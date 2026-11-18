"""The speaker transport must survive the audio device going away.

The report was "the TTS suddenly stopped working". It had not: Piper finished
the utterance and the *output device* stopped accepting audio. These tests
pin the three things that made that permanent, using a fake device that
behaves like the USB speaker did -- it stops draining, so ``write()`` blocks
instead of raising.

Run:
    .venv/bin/python test_audio_out_recovery.py
"""

from __future__ import annotations

import asyncio
import threading
import time
import types

from pipecat.frames.frames import OutputAudioRawFrame
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.local.audio import LocalAudioTransportParams

from audio_out import RecoveringAudioOutputTransport

PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# --------------------------------------------------------------------------
# A fake PortAudio whose stream can be told to stop draining.
# --------------------------------------------------------------------------


class FakeStream:
    """Behaves like a PortAudio stream on a device that was unplugged.

    While ``wedged`` is set, ``write`` blocks forever -- exactly what the log
    shows: no exception, no error, just a write that never returns.

    ``wedged`` is per-stream, not per-device, because that is what a
    re-enumeration does: the handle bound to the old device node dies, while a
    freshly opened handle works. A device-wide flag would model an unplugged
    device, which is a different failure and is covered by test 5.
    """

    def __init__(self, device: "FakeDevice"):
        self.device = device
        self.closed = False
        self.started = False
        self.bytes_written = 0
        self.wedged = False
        self._release = threading.Event()

    def write(self, data: bytes) -> None:
        if self.wedged:
            # Block until the test releases us or the process ends. A real
            # wedged USB write never returns either.
            self._release.wait(timeout=30)
            raise OSError("device went away")
        if self.closed:
            raise OSError("stream is closed")
        self.bytes_written += len(data)

    def start_stream(self) -> None:
        if self.closed:
            raise OSError("stream is closed")
        self.started = True

    def stop_stream(self) -> None:
        pass

    def close(self) -> None:
        self.closed = True
        self._release.set()


class FakeDevice:
    def __init__(self):
        self.streams: list[FakeStream] = []
        self.opens = 0
        self.terminated = 0
        self.portaudio_instances = 0
        self.name = "USB Composite Device: Audio (hw:3,0)"

    def unplug(self):
        """Re-enumerate the device: every open handle dies, new ones work.

        This is the 22:01:43 event from dmesg -- the old stream is bound to a
        device node that no longer exists and blocks forever; a stream opened
        afterwards is healthy.
        """
        for st in self.streams:
            if not st.closed:
                st.wedged = True

    def open(self, **kwargs):
        self.opens += 1
        st = FakeStream(self)
        self.streams.append(st)
        return st


class FakePyAudio:
    """Minimal stand-in for ``pyaudio.PyAudio``."""

    def __init__(self, device: FakeDevice):
        self.device = device
        device.portaudio_instances += 1

    def get_format_from_width(self, _w):
        return 8

    def get_device_count(self):
        return 1

    def get_device_info_by_index(self, _i):
        return {"name": self.device.name, "maxOutputChannels": 2}

    def open(self, **kwargs):
        return self.device.open(**kwargs)

    def terminate(self):
        self.device.terminated += 1


def make_params(**over) -> LocalAudioTransportParams:
    kw = dict(
        audio_out_enabled=True,
        audio_out_sample_rate=48000,
        audio_out_channels=1,
        audio_out_10ms_chunks=2,
        audio_out_write_timeout_secs=10.0,  # Pipecat's default, the one that bites
        output_device_index=1,
    )
    kw.update(over)
    return LocalAudioTransportParams(**kw)


def make_setup() -> types.SimpleNamespace:
    """A setup good enough for the real ``setup()`` chain to run.

    ``FrameProcessor.setup`` reaches ``self.task_manager``, which raises if it
    was never wired, so a real ``TaskManager`` is supplied rather than a mock:
    the point of these tests is that the real code path runs.
    """
    from pipecat.utils.asyncio.task_manager import TaskManager

    return types.SimpleNamespace(
        audio_in_sample_rate=48000,
        audio_out_sample_rate=48000,
        clock=None,
        task_manager=TaskManager(),
        pipeline_worker=None,
        observer=None,
        tool_resources=None,
        tracing_context=None,
    )


def frame(nbytes: int = 1920) -> OutputAudioRawFrame:
    return OutputAudioRawFrame(audio=b"\x00" * nbytes, sample_rate=48000, num_channels=1)


# --------------------------------------------------------------------------
# 1. The bug, reproduced: the stock transport goes permanently silent.
# --------------------------------------------------------------------------


def test_stock_transport_goes_silent_forever():
    """The unpatched transport must lose audio permanently. This is the bug."""
    print("\n1. Stock transport, device wedged mid-utterance")

    async def run():
        dev = FakeDevice()
        t = __import__(
            "pipecat.transports.local.audio", fromlist=["LocalAudioOutputTransport"]
        ).LocalAudioOutputTransport(FakePyAudio(dev), make_params())
        t._params.audio_out_write_timeout_secs = 0.2  # keep the test quick
        await t.setup(make_setup())

        ok_before = await t.write_audio_frame(frame())
        dev.unplug()  # the USB device leaves the bus and comes back

        # This is what BaseOutputTransport does on TimeoutError. The real code
        # calls push_error(..., force_treat_as_permanent=True), whose only
        # effect on usability is the set_usable(False) below (frame_processor
        # line 1017). push_error itself is not called here because this
        # transport is not linked into a pipeline, so pushing the ErrorFrame
        # upstream has nowhere to go and recurses.
        try:
            await asyncio.wait_for(t.write_audio_frame(frame()), timeout=0.2)
            timed_out = False
        except (asyncio.TimeoutError, TimeoutError):
            timed_out = True
            await t.set_usable(False)

        return ok_before, timed_out, t.is_usable

    ok_before, timed_out, usable = asyncio.run(run())
    check("stock transport writes while healthy", ok_before)
    check("stock transport stalls when the device leaves", timed_out)
    check("stock transport is marked permanently unusable", usable is False,
          "this is the flag nothing in agent.py ever resets")

    # And prove that is the real path, not a simulation of one.
    src = open(
        ".venv/lib/python3.12/site-packages/pipecat/transports/base_output.py",
        encoding="utf-8",
    ).read()
    check("Pipecat forces the verdict on a write timeout",
          "force_treat_as_permanent=True" in src,
          "base_output._internal_write_audio_frame, on TimeoutError")
    check("an unusable transport drops every later frame",
          "if not self._transport.is_usable:" in src,
          "the silent drop that makes the bot mute")


def test_agent_never_resets_usability():
    """Prove the missing reset is real, not assumed."""
    print("\n2. Nothing in agent.py restores usability")
    src = open("agent.py", encoding="utf-8").read()
    check("agent.py never calls set_usable(True)", "set_usable(True)" not in src)
    check("agent.py registers no on_error handler", "on_error" not in src)


# --------------------------------------------------------------------------
# 3. The fix: recovery reopens the device and audio flows again.
# --------------------------------------------------------------------------


def test_recovering_transport_resumes_audio():
    """After a stall the recovering transport reopens and keeps writing."""
    print("\n3. Recovering transport, device wedged mid-utterance")

    events = []

    async def run():
        dev = FakeDevice()
        t = RecoveringAudioOutputTransport(
            FakePyAudio(dev),
            make_params(),
            recover_after_secs=0.2,
            reopen_settle_secs=0.0,
            on_event=lambda kind, fields: events.append((kind, fields)),
            py_audio_factory=lambda: FakePyAudio(dev),
        )
        await t.setup(make_setup())

        ok_before = await t.write_audio_frame(frame())
        first_stream = t._out_stream
        first_executor = t._executor

        dev.unplug()  # the held stream is bound to a device node that is gone
        ok_wedged = await t.write_audio_frame(frame())

        # No flag to clear: the device came back on the bus 0.32 s later, so a
        # stream opened now is healthy. The assertion is that one was opened.
        ok_after = await t.write_audio_frame(frame())

        return dict(
            ok_before=ok_before,
            ok_wedged=ok_wedged,
            ok_after=ok_after,
            first_stream=first_stream,
            first_executor=first_executor,
            t=t,
            dev=dev,
        )

    r = asyncio.run(run())
    t, dev = r["t"], r["dev"]

    check("writes while healthy", r["ok_before"])
    check("survives the stall instead of dropping the frame", r["ok_wedged"],
          "the stock transport returns False and stays silent")
    check("a NEW stream was opened", dev.opens >= 2, f"opens={dev.opens}")
    check("the wedged stream was abandoned", t._out_stream is not r["first_stream"])
    check("the wedged writer thread was replaced", t._executor is not r["first_executor"],
          "a blocked thread in max_workers=1 would stall every later write")
    check("PortAudio was reinitialised", dev.portaudio_instances >= 2,
          "its cached device table is stale after a re-enumeration")
    check("the shared PyAudio instance was NOT terminated", dev.terminated == 0,
          "terminating it would take the microphone down with the speaker")
    check("audio flows again after recovery", r["ok_after"])
    check("recovery is reported to the dashboard",
          any(k == "audio_out_recovered" for k, _ in events),
          f"events={[k for k, _ in events]}")

    # The decisive one: the flag Pipecat checks before every write.
    check("transport is still usable after recovery", t.is_usable,
          "stock transport is False here and never comes back")


# --------------------------------------------------------------------------
# 4. Through the real BaseOutputTransport, so the 10 s deadline is covered.
# --------------------------------------------------------------------------


def test_pipecat_deadline_never_fires():
    """The recovering transport must return inside Pipecat's own deadline.

    ``BaseOutputTransport._internal_write_audio_frame`` marks the transport
    permanently unusable when its 10 s ``wait_for`` expires. If recovery
    finishes sooner, that code never runs.
    """
    print("\n4. Through Pipecat's BaseOutputTransport, with its real deadline")

    class MediaSender(BaseOutputTransport.MediaSender):
        """The real sender, so the real timeout logic runs."""

    async def run():
        dev = FakeDevice()
        t = RecoveringAudioOutputTransport(
            FakePyAudio(dev),
            make_params(audio_out_write_timeout_secs=1.0),
            recover_after_secs=0.2,
            reopen_settle_secs=0.0,
            py_audio_factory=lambda: FakePyAudio(dev),
        )
        await t.setup(make_setup())

        params = make_params(audio_out_write_timeout_secs=1.0)
        sender = MediaSender(
            t,
            destination="audio",
            sample_rate=48000,
            audio_chunk_size=1920,
            params=params,
        )
        # The private helper is what BaseOutputTransport uses for every frame.
        first = await sender._internal_write_audio_frame(frame())

        dev.unplug()
        started = time.monotonic()
        second = await sender._internal_write_audio_frame(frame())
        elapsed = time.monotonic() - started

        third = await sender._internal_write_audio_frame(frame())
        return first, second, third, elapsed, t.is_usable

    first, second, third, elapsed, usable = asyncio.run(run())
    check("Pipecat's sender writes while healthy", first)
    check("Pipecat's sender does not give up on the stalled frame", second,
          "False here means the frame was dropped and the bot went mute")
    check(f"recovery finished inside Pipecat's deadline ({elapsed:.2f}s)",
          elapsed < 1.0, "past it, force_treat_as_permanent=True fires")
    check("audio flows again afterwards", third)
    check("Pipecat still considers the transport usable", usable)


# --------------------------------------------------------------------------
# 5. A device that is genuinely gone must not hang the audio task.
# --------------------------------------------------------------------------


def test_absent_device_does_not_hang():
    """If reopening keeps failing, give up rather than block forever."""
    print("\n5. Device stays gone")

    async def run():
        dev = FakeDevice()
        t = RecoveringAudioOutputTransport(
            FakePyAudio(dev),
            make_params(),
            recover_after_secs=0.2,
            reopen_settle_secs=0.0,
            reopen_attempts=2,
            py_audio_factory=lambda: FakePyAudio(dev),
        )
        await t.setup(make_setup())
        await t.write_audio_frame(frame())

        # Every reopen from here on fails.
        orig = dev.open
        def failing_open(**kwargs):
            dev.opens += 1
            raise OSError("No such device")
        dev.open = failing_open
        dev.unplug()

        started = time.monotonic()
        ok = await t.write_audio_frame(frame())
        return ok, time.monotonic() - started, t

    ok, elapsed, t = asyncio.run(run())
    check("reports failure instead of claiming success", ok is False)
    check(f"gives up promptly ({elapsed:.2f}s)", elapsed < 5.0,
          "the audio task must not be held hostage by a missing device")


# --------------------------------------------------------------------------
# 6. Recovery must not run twice at once.
# --------------------------------------------------------------------------


def test_concurrent_recovery_is_serialised():
    """Two callers hitting the stall must not both reopen the device."""
    print("\n6. Concurrent recovery")

    async def run():
        dev = FakeDevice()
        t = RecoveringAudioOutputTransport(
            FakePyAudio(dev),
            make_params(),
            recover_after_secs=0.2,
            reopen_settle_secs=0.0,
            py_audio_factory=lambda: FakePyAudio(dev),
        )
        await t.setup(make_setup())
        dev.unplug()
        results = await asyncio.gather(
            t.write_audio_frame(frame()), t.write_audio_frame(frame())
        )
        return results, dev, t

    results, dev, t = asyncio.run(run())
    check("both concurrent writes complete", len(results) == 2)
    # setup opens one stream; a single recovery opens one more. Two
    # recoveries would mean both callers reopened the device.
    check("only one recovery reopened the device", t.recoveries == 1,
          f"recoveries={t.recoveries}, opens={dev.opens}")


def main() -> int:
    print("=" * 70)
    print("Speaker transport recovery")
    print("=" * 70)
    test_stock_transport_goes_silent_forever()
    test_agent_never_resets_usability()
    test_recovering_transport_resumes_audio()
    test_pipecat_deadline_never_fires()
    test_absent_device_does_not_hang()
    test_concurrent_recovery_is_serialised()
    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
