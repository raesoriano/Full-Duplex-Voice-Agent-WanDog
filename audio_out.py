"""A speaker transport that can come back after the audio device goes away.

Why this exists
---------------
The TTS did not stop working. The *speaker* did, and Pipecat then decided it
was gone for good.

On this Pi the speaker is a USB audio class device ("USB Composite Device",
Jieli Technology, ALSA card 3 / hw:3,0). It re-enumerates on the bus by
itself -- once at 358.8 s of uptime in the session that produced the report:

    [  358.778094] usb 3-2: USB disconnect, device number 3
    [  359.098108] usb 3-2: new full-speed USB device number 4 using xhci-hcd

It came back 0.32 s later. The PortAudio stream the agent was holding did
not. PortAudio binds to the device node, not to the card, so the old handle
pointed at a device that no longer existed. Writing to it did not raise: it
just never returned, because the USB endpoint it was writing to had nothing
draining it any more.

That is the log in the report:

    22:01:47.394  PiperTTSService  Finished TTS [...]        <- TTS fine
    22:01:55.174  LocalAudioOutputTransport  no longer usable
    22:01:55.179  timed out after 10.0s writing audio ...
                  the peer has stopped reading

The write began ~1.2 s after the device left the bus and timed out 10 s
later. ``BaseOutputTransport._internal_write_audio_frame`` waits
``audio_out_write_timeout_secs`` (default 10.0), and on ``TimeoutError`` it
calls ``push_error(..., force_treat_as_permanent=True)``, which calls
``set_usable(False)``.

Two things then make it permanent, and neither is recoverable from outside:

1. ``_internal_write_audio_frame`` starts with
   ``if not self._transport.is_usable: return False`` -- so **every** later
   audio frame is dropped silently. The bot is mute for the rest of the
   process, and nothing is logged per frame to say so.
2. ``agent.py`` built ``PipelineTask(pipeline)`` with no params, so the
   unusable-processor policy is the default ``CONTINUE``: the pipeline keeps
   running, the VAD keeps hearing you, turns keep being taken, and only the
   audio never comes out. The agent looks alive and answers nothing.

``agent.py`` registers no ``on_error`` handler and never calls
``set_usable(True)``, so once that flag is False it stays False until the
process is restarted. That is the "TTS suddenly stopped working".

What this class changes
-----------------------
``RecoveringAudioOutputTransport`` owns the stream instead of inheriting one
opened once in ``setup()``, so it can replace it:

* **Its own write timeout, shorter than Pipecat's.** We want to catch the
  stall ourselves, before ``BaseOutputTransport`` turns it into a permanent
  verdict. ``recover_after_secs`` defaults to 3.0 s -- well under the 10 s
  Pipecat would wait, and ~2 s more than the longest healthy write needs.
* **A fresh executor on every recovery.** This is the subtle one: the stuck
  write is a *thread* blocked inside ``stream.write()``.
  ``asyncio.wait_for`` cancels the await, not the thread. The inherited
  transport uses ``ThreadPoolExecutor(max_workers=1)``, so that one thread is
  the only writer slot there is -- leaving it wedged means every future write
  queues behind a write that will never finish. Reopening the stream is not
  enough; the executor has to be abandoned and replaced, and the orphaned
  thread has to be daemonised so it cannot block interpreter shutdown.
* **Re-resolve the device by name, not by index.** After a re-enumeration the
  PortAudio index can shift. ``WANDOG_OUT_DEVICE`` is an index, which is fine
  at startup and wrong afterwards, so recovery looks the device up by the
  name it was opened with and falls back to the configured index only if the
  name is gone.
* **Reinitialise PortAudio when the device list is stale.** PortAudio caches
  its device table; after an unplug it can still list a device that is not
  there. Recovery calls ``PyAudio.terminate()`` + a fresh ``PyAudio()`` so the
  table is re-read from ALSA.
* **Restore usability.** After a successful reopen, ``set_usable(True)`` --
  the call ``agent.py`` was missing -- otherwise Pipecat keeps dropping frames
  no matter how healthy the new stream is.

Everything is reported through the event bus, so the dashboard shows the
recovery instead of the agent just going quiet.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pyaudio
from pipecat.transports.local.audio import (
    LocalAudioOutputTransport,
    LocalAudioTransportParams,
)

logger = logging.getLogger(__name__)

# How long a single write may take before we treat the stream as wedged.
# Healthy writes on this Pi finish in a few milliseconds (the device reports
# 8 ms low latency); 3 s is two orders of magnitude of headroom while still
# being comfortably inside Pipecat's own 10 s deadline.
DEFAULT_RECOVER_AFTER_SECS = 3.0

# Reopening the device needs to wait for ALSA to finish re-registering the
# card. The observed re-enumeration took 0.32 s end to end; a second is enough
# for the card to be usable again and still fast enough to be unnoticeable.
DEFAULT_REOPEN_SETTLE_SECS = 1.0

# How many times to try reopening before giving up for this write. The device
# may be genuinely gone (unplugged), in which case there is nothing to do and
# retrying forever would just stall the audio task.
DEFAULT_REOPEN_ATTEMPTS = 3


class RecoveringAudioOutputTransport(LocalAudioOutputTransport):
    """``LocalAudioOutputTransport`` that survives the device disappearing.

    Args:
        py_audio: PortAudio instance (only used for its host API table).
        params: Transport parameters, as for the base class.
        recover_after_secs: A write taking longer than this is treated as a
            wedged stream and triggers recovery. Keep it below
            ``audio_out_write_timeout_secs`` so we get there first.
        on_event: Optional callable ``(kind, fields)`` for reporting. Called
            from the audio task, so it must not block.
    """

    def __init__(
        self,
        py_audio: pyaudio.PyAudio,
        params: LocalAudioTransportParams,
        recover_after_secs: float = DEFAULT_RECOVER_AFTER_SECS,
        reopen_settle_secs: float = DEFAULT_REOPEN_SETTLE_SECS,
        reopen_attempts: int = DEFAULT_REOPEN_ATTEMPTS,
        on_event=None,
        py_audio_factory=None,
    ):
        super().__init__(py_audio, params)
        self._recover_after_secs = recover_after_secs
        self._reopen_settle_secs = reopen_settle_secs
        self._reopen_attempts = reopen_attempts
        self._on_event = on_event
        # How to rebuild PortAudio after a re-enumeration. Injectable so tests
        # can supply a fake; constructing a real PyAudio() costs ~3 s of ALSA
        # probing, which must not happen inside a running pipeline.
        self._py_audio_factory = py_audio_factory or pyaudio.PyAudio

        # What we opened, so recovery can open it again the same way.
        self._opened_name: str | None = None
        self._opened_rate: int = 0
        self._opened_channels: int = 0
        self._opened_index: int | None = None

        # Recovery must not run twice at once: the base class writes from one
        # task, but ``_send_silence`` can arrive from another.
        self._recover_lock = threading.Lock()

        self.recoveries = 0
        self.last_recovery_at: float | None = None
        # Bumped every time the stream is replaced. A writer that stalled on
        # generation N and wakes to find generation N+1 does not need to
        # recover anything -- someone already did -- it just retries.
        self._generation = 0

    # -- reporting ---------------------------------------------------------

    def _emit(self, kind: str, **fields) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, fields)
        except Exception:  # noqa: BLE001 - reporting must never break audio
            logger.debug("audio_out: event callback failed", exc_info=True)

    # -- stream lifecycle --------------------------------------------------

    def _open_stream(self):
        """Open the output stream and remember what we opened.

        Resolves the device by name when we have one, so a re-enumerated
        device is found again even if its index moved.
        """
        index = self._resolve_index()
        stream = self._py_audio.open(
            format=self._py_audio.get_format_from_width(2),
            channels=self._opened_channels,
            rate=self._opened_rate,
            output=True,
            output_device_index=index,
        )
        stream.start_stream()
        self._opened_index = index
        if index is not None:
            try:
                self._opened_name = self._py_audio.get_device_info_by_index(index)["name"]
            except Exception:  # noqa: BLE001 - name is best-effort
                pass
        return stream

    def _resolve_index(self) -> int | None:
        """Find the device we were opened on, preferring its name.

        An index is only valid until the next re-enumeration; the name
        survives it.
        """
        if self._opened_name:
            try:
                for i in range(self._py_audio.get_device_count()):
                    info = self._py_audio.get_device_info_by_index(i)
                    if info["name"] == self._opened_name and info["maxOutputChannels"] > 0:
                        return i
            except Exception:  # noqa: BLE001 - fall through to the index
                logger.debug("audio_out: name lookup failed", exc_info=True)
        return self._opened_index

    def _reinit_portaudio(self) -> None:
        """Move to a PortAudio instance with a fresh device table.

        After a re-enumeration PortAudio can keep advertising the device that
        is no longer there, so an ``open`` would succeed against a dead node
        and stall exactly as before.

        The old instance is NOT terminated. ``agent.py`` builds one
        ``PyAudio()`` and shares it with the input transport, so terminating
        it here would take the microphone down with the speaker -- turning a
        recoverable speaker glitch into a deaf agent. PortAudio reference
        counts its instances (a second ``PyAudio()`` is a second reference, not
        a second initialisation), so constructing a new one is cheap and
        leaves the mic alone.

        Blocking: ``PyAudio()`` probes every ALSA card (~3 s measured here).
        The caller runs this off the event loop.
        """
        try:
            self._py_audio = self._py_audio_factory()
        except Exception:  # noqa: BLE001
            logger.warning("audio_out: could not re-create PyAudio", exc_info=True)

    def _drop_stream(self) -> None:
        """Close the wedged stream and abandon its writer thread.

        ``stream.stop_stream()`` on a stream whose device is gone can block,
        so it runs on a daemon thread we are willing to walk away from.
        """
        stream = self._out_stream
        self._out_stream = None
        if stream is not None:
            def _close():
                try:
                    stream.stop_stream()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    stream.close()
                except Exception:  # noqa: BLE001
                    pass

            t = threading.Thread(target=_close, name="audio-out-close", daemon=True)
            t.start()

        # The executor's single worker is blocked inside write() on the dead
        # stream. Abandon it: cancelling the asyncio future does not unblock
        # the thread, so the slot has to be replaced, not reused.
        old = self._executor
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audio-out")
        # wait=False: do not block on the wedged thread. Python's
        # ThreadPoolExecutor registers an atexit hook that joins workers, which
        # would hang shutdown, so the old executor is deliberately leaked
        # rather than shut down; the thread is a daemon-equivalent blocked in
        # a C call and dies with the process.
        del old

    async def _recover_async(self) -> bool:
        """Run :meth:`_recover` off the event loop.

        Reopening the device blocks: ``time.sleep`` for the settle, and
        ``PyAudio()`` probing every ALSA card (~3 s measured here). Running
        that on the loop would freeze the whole pipeline -- including the
        VAD and the LLM -- for the duration, which is worse than the fault.
        """
        loop = self.get_event_loop()
        return await loop.run_in_executor(None, self._recover)

    def _recover(self) -> bool:
        """Replace the wedged stream. Returns True if audio can flow again."""
        if not self._recover_lock.acquire(blocking=False):
            # Another caller is already recovering the device. Report success
            # so this caller retries on whatever stream that recovery leaves
            # behind, instead of opening a second stream of its own.
            return True
        try:
            started = time.monotonic()
            self._emit("audio_out_recovering", reason="write stalled")

            self._drop_stream()
            self._reinit_portaudio()

            for attempt in range(1, self._reopen_attempts + 1):
                if self._reopen_settle_secs > 0:
                    time.sleep(self._reopen_settle_secs)
                try:
                    self._out_stream = self._open_stream()
                except Exception as e:  # noqa: BLE001 - device may still be gone
                    logger.warning(
                        "audio_out: reopen attempt %d/%d failed: %s",
                        attempt, self._reopen_attempts, e,
                    )
                    continue

                self.recoveries += 1
                self.last_recovery_at = time.time()
                self._generation += 1
                elapsed = time.monotonic() - started
                logger.warning(
                    "audio_out: speaker stream recovered in %.1fs "
                    "(attempt %d, device %r, index %s)",
                    elapsed, attempt, self._opened_name, self._opened_index,
                )
                self._emit(
                    "audio_out_recovered",
                    device=self._opened_name,
                    index=self._opened_index,
                    seconds=round(elapsed, 2),
                    attempt=attempt,
                )
                return True

            logger.error(
                "audio_out: could not reopen the speaker after %d attempts; "
                "audio will stay silent until the device is back",
                self._reopen_attempts,
            )
            self._emit("audio_out_recovery_failed", attempts=self._reopen_attempts)
            return False
        finally:
            self._recover_lock.release()

    # -- Pipecat hooks -----------------------------------------------------

    async def setup(self, setup) -> None:
        """Let the base class open the stream, then record what it opened.

        The base ``setup`` already opens ``self._out_stream``; reopening it
        here would leave a second handle on the device for no gain. What
        recovery needs is the stream's *identity* -- rate, channels and the
        device name -- so it can open the same thing again after the device
        re-enumerates and its index moves.
        """
        await super().setup(setup)
        self._opened_rate = self.sample_rate
        self._opened_channels = self._params.audio_out_channels
        self._opened_index = self._params.output_device_index
        if self._opened_index is not None:
            try:
                self._opened_name = self._py_audio.get_device_info_by_index(
                    self._opened_index
                )["name"]
            except Exception:  # noqa: BLE001 - name is best-effort
                logger.debug("audio_out: could not read device name", exc_info=True)

    async def write_audio_frame(self, frame) -> bool:
        """Write a frame, recovering the stream if the write stalls.

        This is the whole point of the class: a stalled write means the device
        is gone, and a device that is gone can be reopened. The frame is
        retried once on the new stream, so the audio that was playing when the
        device dropped is not silently lost.

        The stream and its executor are captured before the write so that a
        concurrent recovery swapping them mid-write cannot make this call
        write to a stream that has already been closed.
        """
        for attempt in (1, 2):
            stream = self._out_stream
            executor = self._executor
            generation = self._generation
            if not stream:
                return False
            try:
                await asyncio.wait_for(
                    self.get_event_loop().run_in_executor(
                        executor, stream.write, frame.audio
                    ),
                    timeout=self._recover_after_secs,
                )
                return True
            except (asyncio.TimeoutError, TimeoutError):
                logger.warning(
                    "audio_out: write stalled for %.1fs -- assuming the device "
                    "was re-enumerated; reopening",
                    self._recover_after_secs,
                )
                if attempt == 2:
                    return False
                if self._generation != generation:
                    # Another writer already replaced the stream; retry on it.
                    continue
                if not await self._recover_async():
                    return False
            except Exception as e:  # noqa: BLE001 - a closed device raises
                logger.warning("audio_out: write failed (%s); reopening", e)
                if attempt == 2:
                    return False
                if self._generation != generation:
                    continue
                if not await self._recover_async():
                    return False
        return False

    async def cleanup(self) -> None:
        """Close the stream and stop the writer pool."""
        self._out_stream = None
        await super().cleanup()
