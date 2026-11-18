"""WanDog -- the heart of the voice agent.

A local, full-duplex voice command agent built on Pipecat:

    mic -> VAD (Silero ONNX) -> LLM (OnIt, OpenAI-compatible) -> TTS (Piper) -> speaker

No ASR in this stage: the LLM consumes raw audio directly (OpenAI-compatible
audio input). Barge-in is handled by Pipecat's input transport (interrupting
the bot when the user starts speaking). The rest of the pipeline (ASR, MCP
tools, dashboard) is added on top of this core later.

Run:
    .venv/bin/python agent.py
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
from pathlib import Path

import pyaudio
from loguru import logger

import json

import config
import mcp_bridge
from resample import AudioResampleProcessor
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    LLMTextFrame,
    StartFrame,
    STTMuteFrame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMAssistantAggregatorParams,
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_mute.function_call_user_mute_strategy import (
    FunctionCallUserMuteStrategy,
)
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from streaming_asr import SherpaStreamingSTTService
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

logging.basicConfig(level=logging.INFO)


class MicControlWatcher(FrameProcessor):
    """Honour the dashboard's mic toggle.

    The dashboard writes ``<log>.mic_control.json`` with ``{"enabled": bool}``.
    Nothing used to read that file, so the toggle was decorative. This
    processor polls it and, on change, pushes an ``STTMuteFrame`` upstream so
    the ASR stops transcribing. Muting the ASR (rather than dropping audio)
    keeps VAD and barge-in behaviour intact and is the officially supported
    way to stop a Pipecat STT service.
    """

    def __init__(self, control_path: str, poll_secs: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._path = Path(control_path)
        self._poll = poll_secs
        self._last: bool | None = None
        self._task: asyncio.Task | None = None

    def _read(self) -> bool:
        """True = mic enabled. Missing/corrupt file means enabled."""
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            return bool(data.get("enabled", True))
        except Exception:
            return True

    async def _watch(self):
        # The FIRST read is applied, not just remembered.
        #
        # It used to be skipped (`if self._last is not None`), on the reasoning
        # that the STT starts unmuted so "enabled" is already the truth. That
        # holds only for enabled. With the dashboard left muted, the agent
        # started, read enabled=False, recorded it, pushed nothing -- and then
        # never pushed again, because the value never changed. Measured: the
        # agent was started against {"enabled": false} and the STT still
        # transcribed ("What in the dog?"). The mic was live while the UI said
        # it was muted, so speech the person thought was ignored still became a
        # turn.
        #
        # Pushing the initial state is harmless when it agrees (the STT is
        # already unmuted) and is the only thing that makes a persisted "muted"
        # true when it disagrees.
        first = True
        while True:
            enabled = self._read()
            if first or enabled != self._last:
                logger.info("Mic {}", "enabled" if enabled else "muted")
                # Upstream: the STT sits before us in the pipeline.
                await self.push_frame(
                    STTMuteFrame(mute=not enabled), FrameDirection.UPSTREAM
                )
                self._last = enabled
                first = False
            await asyncio.sleep(self._poll)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._task is None:
            self._task = asyncio.create_task(self._watch())
        await self.push_frame(frame, direction)

    async def cleanup(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None
        await super().cleanup()


class DashboardControlWatcher(FrameProcessor):
    """Consume the dashboard's Send (typed text) and Stop (interrupt) buttons.

    Both are plain files next to the event log, the same mechanism the mic
    toggle uses, for the same reason: the dashboard is a separate process that
    may crash, restart or never start, and none of that may take the voice loop
    down. The agent owns the files -- it deletes each one after acting on it --
    so a message is answered exactly once even if the dashboard is left open
    and its rerun loop writes again.

    Why a pipeline stage rather than a callback
    -------------------------------------------
    Both actions are frames, and frames enter a pipeline through the task. A
    typed message is appended to the SAME LLM context the spoken turns use
    (``LLMMessagesAppendFrame`` with ``run_llm=True``), so it is
    indistinguishable from a spoken turn afterwards: the context is shared, the
    model sees one conversation, and the reply is spoken by the same TTS. A
    stop is an ``InterruptionFrame``, which is exactly what a barge-in
    broadcasts -- so the stop button cannot drift from real barge-in behaviour
    (it stops the LLM mid-generation and drops the TTS's interruptible audio).

    Placement: this stage sits BEFORE the user aggregator, because
    ``queue_frame()`` pushes the typed message into this stage's own input
    queue and it then flows DOWNSTREAM from here -- so the aggregator that
    handles ``LLMMessagesAppendFrame`` must be downstream of this point. A stop
    is an ``InterruptionFrame`` broadcast in both directions, so it reaches the
    STT upstream and the LLM/TTS downstream regardless.
    """

    def __init__(self, typed_path: str, stop_path: str, bus=None,
                 poll_secs: float = 0.2, clear_path: str | None = None,
                 **kwargs):
        super().__init__(**kwargs)
        self._typed_path = Path(typed_path)
        self._stop_path = Path(stop_path)
        self._clear_path = Path(clear_path) if clear_path else None
        self._bus = bus
        self._poll = poll_secs
        self._task: asyncio.Task | None = None
        # Last id acted on, so a file that could not be deleted (permissions,
        # a racing writer) still cannot be answered twice.
        self._last_typed_id = None
        self._last_stop_id = None
        self._last_clear_id = None

    # -- file plumbing -----------------------------------------------------

    def _take(self, path: Path) -> dict | None:
        """Read a control file and delete it. Returns None when there is
        nothing new to do.

        Deleting before acting (rather than after) is deliberate: if acting
        raises, the message is lost rather than answered twice. A dropped
        question is a visible, recoverable problem -- the person sees no reply
        and asks again -- while a duplicate is a confusing double answer with
        two replies spoken over each other.
        """
        try:
            if not path.exists():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 -- a half-written file is retried next tick
            return None
        try:
            path.unlink()
        except OSError:
            pass
        return payload if isinstance(payload, dict) else None

    # -- actions -----------------------------------------------------------

    async def _handle_typed(self, payload: dict) -> None:
        text = (payload.get("text") or "").strip()
        if not text:
            return
        msg_id = payload.get("id")
        if msg_id is not None and msg_id == self._last_typed_id:
            return
        self._last_typed_id = msg_id

        logger.info(" typed: {}", text)
        if self._bus is not None:
            # source="typed" lets the dashboard tell a typed turn from a
            # spoken one -- they render the same otherwise, and a person
            # reading back the log should be able to see which was which.
            await self._bus.emit("user_text", text=text, source="typed")

        # Appending to the context (rather than injecting a fake transcription
        # frame) is the supported path: the aggregator adds the message and,
        # with run_llm=True, pushes an LLMContextFrame that starts inference.
        # A fake TranscriptionFrame would instead have to satisfy the turn
        # strategies -- VAD start, VAD stop, transcript timeout -- none of
        # which a typed message has any reason to fake.
        from pipecat.frames.frames import LLMMessagesAppendFrame
        await self.queue_frame(
            LLMMessagesAppendFrame(
                messages=[{"role": "user", "content": text}], run_llm=True
            )
        )

    async def _handle_stop(self, payload: dict) -> None:
        stop_id = payload.get("id")
        if stop_id is not None and stop_id == self._last_stop_id:
            return
        self._last_stop_id = stop_id

        logger.info(" stop requested from the dashboard")
        if self._bus is not None:
            await self._bus.emit("interrupted", source="dashboard")
        # Same frame a barge-in broadcasts: stops LLM generation and drops the
        # TTS's interruptible audio contexts, so the bot goes quiet mid-word.
        await self.broadcast_interruption()

    async def _handle_clear(self, payload: dict) -> None:
        clear_id = payload.get("id")
        if clear_id is not None and clear_id == self._last_clear_id:
            return
        self._last_clear_id = clear_id

        logger.info(" clear history requested from the dashboard")
        # Stop whatever is being said first. Clearing the context of a turn that
        # is still generating would let the in-flight reply finish and then be
        # written back into the freshly emptied context by the assistant
        # aggregator -- the conversation would reappear one turn at a time.
        await self.broadcast_interruption()
        # LLMMessagesUpdateFrame with an empty list and run_llm=False: the model
        # forgets the conversation, and no new inference is started. Clearing is
        # not a question, so it must not produce an answer.
        from pipecat.frames.frames import LLMMessagesUpdateFrame
        await self.queue_frame(LLMMessagesUpdateFrame(messages=[], run_llm=False))
        if self._bus is not None:
            await self._bus.emit("cleared")

    async def _watch(self):
        while True:
            typed = self._take(self._typed_path)
            if typed is not None:
                await self._handle_typed(typed)
            stop = self._take(self._stop_path)
            if stop is not None:
                await self._handle_stop(stop)
            if self._clear_path is not None:
                clear = self._take(self._clear_path)
                if clear is not None:
                    await self._handle_clear(clear)
            await asyncio.sleep(self._poll)

    # -- pipeline ----------------------------------------------------------

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame) and self._task is None:
            self._task = asyncio.create_task(self._watch())
        await self.push_frame(frame, direction)

    async def cleanup(self):
        if self._task is not None:
            self._task.cancel()
            self._task = None
        await super().cleanup()


class WanDogLogger(FrameProcessor):
    """Observer that prints what the agent hears and says, and records it to
    the event bus so the dashboard can show the interaction live.

    Why this is split in two
    ------------------------
    The user's words and the bot's words are NOT available as frames by the
    time any observer at the end of the pipeline could see them:

      * ``TranscriptionFrame`` is consumed by ``LLMUserAggregator``
        (``_handle_transcription``) and never pushed downstream;
      * ``LLMTextFrame`` is consumed by ``LLMAssistantAggregator``, whose
        ``process_frame`` has no TextFrame branch and therefore never pushes it.

    An earlier version of this class listened for exactly those two frame types
    from the tail of the pipeline, so ``user_text`` and ``bot_text`` were never
    emitted and the dashboard chat stayed empty no matter what was said. That is
    what the transcript events below fix: they come from the aggregators' own
    event handlers (``on_user_turn_message_added`` and
    ``on_assistant_turn_stopped``), which fire with the finished text.

    The frame half still handles what genuinely IS broadcast -- the speaking
    start/stop frames -- so the dashboard can show a live "listening" /
    "speaking" caption.
    """

    def __init__(self, bus=None):
        super().__init__()
        self.bus = bus

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info(" started speaking")
            if self.bus is not None:
                await self.bus.emit("user_started")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info(" stopped speaking")
            if self.bus is not None:
                await self.bus.emit("user_stopped")
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info(" started speaking")
            if self.bus is not None:
                await self.bus.emit("bot_started")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info(" stopped speaking")
            if self.bus is not None:
                await self.bus.emit("bot_stopped")
        elif isinstance(frame, ErrorFrame):
            logger.error(" {}", frame.error)
            # An error is part of the record, not just the console log.
            #
            # This used to be logger.error() and nothing else, so a failed turn
            # was indistinguishable from a slow one on the dashboard: the
            # question appeared, no reply appeared, and the composer sat on Stop
            # because no bot_stopped ever came. The person could not tell
            # "still thinking" from "gave up", and had nothing to act on.
            #
            # ``fatal`` is carried through so the page can say whether the
            # pipeline died or just this turn failed. Errors also end the turn
            # (see dashboard.turn_in_flight), which is what puts the Send button
            # back.
            if self.bus is not None:
                await self.bus.emit(
                    "error",
                    text=str(getattr(frame, "error", "") or ""),
                    fatal=bool(getattr(frame, "fatal", False)),
                )


class PartialTranscriptTap(FrameProcessor):
    """Publish live partial transcripts to the event bus.

    The streaming ASR emits an ``InterimTranscriptionFrame`` every few hundred
    milliseconds while the user is talking. Those frames are consumed by the
    user aggregator (they never reach the tail of the pipeline), so the
    dashboard cannot see them from ``WanDogLogger``. This tap sits between the
    STT and the aggregator and forwards each partial as a ``user_partial``
    event, which the dashboard renders as a growing draft line above the chat.

    The final ``TranscriptionFrame`` still arrives through the aggregator's
    ``on_user_turn_message_added`` event as ``user_text``; the dashboard clears
    the draft when it sees that.
    """

    def __init__(self, bus=None):
        super().__init__()
        self.bus = bus
        self._last = ""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # super() is NOT optional: FrameProcessor.process_frame() is what handles
        # StartFrame (marking this stage started and letting the pipeline become
        # ready) and CancelFrame/InterruptionFrame. Without it this stage never
        # reports itself started, the StartFrame handshake stalls, and every
        # frame queued behind it -- including the final TranscriptionFrame that
        # closes the turn -- waits forever. Symptom: the transcript is produced
        # and logged by the STT, the VAD stop is seen, and then the turn only
        # ever ends on the 5 s inactivity ceiling with an EMPTY transcript, so
        # the LLM is asked "the user said something but it was not recognized".
        await super().process_frame(frame, direction)

        # Partial transcripts are ours to forward; everything else is pushed
        # through untouched so the aggregator behaves exactly as before.
        if isinstance(frame, InterimTranscriptionFrame):
            text = (frame.text or "").strip()
            if self.bus is not None and text and text != self._last:
                self._last = text
                await self.bus.emit("user_partial", text=text)
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, TranscriptionFrame):
            self._last = ""

        await self.push_frame(frame, direction)


class ClockedLLMService(OpenAILLMService):
    """An OpenAI-compatible LLM service that re-reads the clock per inference.

    The system prompt carries the current date and time (see
    ``config.clock_line``). Building it once in ``build_pipeline`` is not
    enough: this agent runs for hours, so a prompt composed at startup tells
    the model the time the agent was launched, and "what time is it" answers
    with a stale clock -- which is the same class of bug as inventing one, and
    harder to notice.

    ``get_chat_completions`` is the one call every inference passes through
    (``_process_context`` builds its request there and nowhere else), so the
    clock is refreshed there. Only the clock sentence is replaced; the rest of
    the prompt is a constant and is not rebuilt.

    Why not a runtime ``LLMUpdateSettingsFrame`` per turn: that would need a
    frame injected before each user turn, and the clock would then be a turn
    behind whenever the turn boundary moved (barge-in, a speculative turn that
    is confirmed). Overriding here cannot drift.
    """

    # Matches the clock sentence wherever it sits in the prompt. The persona
    # comes first and the clock sits inside the tools section, so this cannot
    # be a startswith/prefix swap -- a prefix swap would prepend a second
    # clock line and leave the stale one in the middle, growing the prompt on
    # every turn.
    _CLOCK_RE = re.compile(
        r"The current date and time here is .*?\([^()]*UTC[+-]\d{4}\)\. "
    )

    async def get_chat_completions(self, context):
        self._refresh_clock()
        return await super().get_chat_completions(context)

    def _refresh_clock(self) -> None:
        """Replace the clock sentence in the system instruction with a fresh one."""
        current = self._settings.system_instruction
        if not isinstance(current, str):
            return
        fresh = config.clock_line()
        if self._CLOCK_RE.search(current):
            updated = self._CLOCK_RE.sub(lambda _m: fresh, current, count=1)
        else:
            # No clock to replace (a prompt built by something else, or the
            # guard fired before build_pipeline composed one). Insert the clock
            # after the persona rather than in front of it, and keep the rest
            # of the prompt as it stands instead of rebuilding from config --
            # rebuilding would discard any instruction a caller had added.
            #
            # The anchor is persona+owner, not the persona alone: the owner
            # block sits between the persona and the clock, so inserting after
            # the persona alone would push the clock in front of it and split
            # the "who is speaking" line from the identity it belongs to.
            anchor = config.SYSTEM_PROMPT + " " + config.OWNER_PROMPT
            if current.startswith(anchor):
                updated = anchor + " " + fresh + current[len(anchor):].lstrip()
            elif current.startswith(config.SYSTEM_PROMPT):
                updated = (config.SYSTEM_PROMPT + " " + fresh
                           + current[len(config.SYSTEM_PROMPT):].lstrip())
            else:
                updated = fresh + current
        if updated != current:
            self._settings.system_instruction = updated
            # The base prompt is the snapshot composition rebuilds from; if it
            # is not updated the next _compose_system_instruction reverts us.
            self._base_system_instruction = updated


def wire_transcript_events(user_agg, assistant_agg, bus) -> None:
    """Record the user's words and the bot's reply on the event bus.

    These two texts exist only inside the aggregators (see ``WanDogLogger``),
    so we subscribe to the events they emit when a turn completes rather than
    trying to catch a frame that is never pushed downstream.
    """
    if bus is None:
        return

    if user_agg is not None:

        @user_agg.event_handler("on_user_turn_message_added")
        async def _on_user_turn_message_added(aggregator, message):
            text = (getattr(message, "content", "") or "").strip()
            if text:
                logger.info(" {}", text)
                await bus.emit("user_text", text=text)

    if assistant_agg is not None:

        @assistant_agg.event_handler("on_assistant_turn_stopped")
        async def _on_assistant_turn_stopped(aggregator, message):
            text = (getattr(message, "content", "") or "").strip()
            if text:
                logger.info(" {}", text)
                await bus.emit("bot_text", text=text)


def report_mic_gain() -> None:
    """Log the mic's current capture gain. Read-only -- never changes it.

    The gain is deliberately NOT set by the agent: it is a property of the
    room and the user's distance from the mic, so it belongs to the person
    sitting in front of it. 100% was too hot for this desk -- it clipped on
    speech (peak 1.02) and let Silero score the keyboard and the speaker's
    own output as speech. Set it yourself and it stays:

        amixer -c 2 sset Mic 70% cap
    """
    if not config.MIC_CARD:
        return
    try:
        import subprocess

        for ctl in ("Mic", "Mic Capture"):
            r = subprocess.run(
                ["amixer", "-c", config.MIC_CARD, "sget", ctl],
                capture_output=True, text=True, timeout=5,
            )
            if r.returncode != 0:
                continue
            # The level line is the one carrying "[N%]"; "Capture channels"
            # and "Limits" also match on "Capture" but carry no level.
            line = next(
                (ln.strip() for ln in r.stdout.splitlines() if "[" in ln and "%]" in ln),
                "",
            )
            logger.info("  Mic gain (unchanged): {} card {}: {}",
                        ctl, config.MIC_CARD, line or "no level reported")
            return
        logger.debug("Could not read mic gain on card {}", config.MIC_CARD)
    except Exception as e:  # noqa: BLE001 -- never block startup on this
        logger.debug("Could not read mic gain: {}", e)


def describe_asr() -> str:
    """One-line label for the ASR stage, for the log and the dashboard."""
    if not config.ASR_ENABLED:
        return "off (raw-audio LLM)"
    if config.ASR_BACKEND == "streaming":
        label = "streaming " + Path(config.ASR_STREAM_MODEL_DIR).name
        if config.ASR_FINAL_PASS == "whisper":
            label += " + final " + config.ASR_FINAL_MODEL
        return label
    return "whisper " + config.ASR_MODEL


def pipeline_stage_names() -> list[str]:
    """The stage order ``build_pipeline`` produces, as class names.

    Exists so a test can assert on the ORDER without starting the pipeline --
    starting it needs a microphone and an LLM. Order is not cosmetic here: the
    dashboard's typed text is queued as a frame that flows DOWNSTREAM from the
    stage that queues it, so a watcher placed after the aggregator that handles
    it is a silent no-op. That exact mistake was made once already.
    """
    pipeline, _task, _inp, _out = build_pipeline("stage-order-probe")
    return [type(p).__name__ for p in pipeline._processors]


def build_pipeline(api_key: str, bridge: "mcp_bridge.ToolBridgeProcessor | None" = None,
                   bus=None, transports=None):
    """Construct the Pipecat pipeline.

    Returns (pipeline, task, input, output). When ``bridge`` is given, its
    tools are registered on the LLM context so the model can call OnIt's
    tools; the system prompt is extended to tell it so. ``bus`` is the event
    bus the logger records to (the dashboard's data source).
    """

    # --- Audio transport (mic + speaker) -----------------------------------
    # ``transports`` lets a test inject a transport pair (see test_pipeline_asr.py)
    # so the whole chain -- VAD, streaming ASR, aggregators, LLM -- can be
    # driven from a wav file without a microphone. Production passes None.
    capture_rate = config.INPUT_SAMPLE_RATE
    # Everything downstream of the resampler sees this rate.
    proc_rate = config.ASR_SAMPLE_RATE if config.ASR_ENABLED else capture_rate
    if transports is not None:
        input_transport, output_transport = transports
    else:
        # Read-only: the agent must never touch the mic gain (see report_mic_gain).
        report_mic_gain()
        pa = pyaudio.PyAudio()
        # The mic is ALWAYS captured at the hardware rate (48 kHz). This card
        # accepts no other capture rate -- 8k/16k/22.05k/32k/44.1k/96k all fail
        # with "Errno -9997 Invalid sample rate". We therefore capture at 48 kHz
        # and convert down to what VAD and ASR need (16 kHz) in the pipeline,
        # using AudioResampleProcessor right after the input transport.
        transport_params = LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_in_sample_rate=capture_rate,
            audio_in_channels=1,
            audio_out_enabled=True,
            audio_out_sample_rate=config.OUTPUT_SAMPLE_RATE,
            audio_out_channels=1,
            audio_out_volume=1.0,
            # Pin the mic (USB PnP Audio Device) and speaker (USB Composite
            # Device) so the agent never picks the wrong card.
            input_device_index=config.INPUT_DEVICE_INDEX,
            output_device_index=config.OUTPUT_DEVICE_INDEX,
        )
        input_transport = LocalAudioInputTransport(pa, transport_params)
        output_transport = LocalAudioOutputTransport(pa, transport_params)

    # --- VAD (Silero ONNX, torch-free) -------------------------------------
    # In Pipecat 1.15 barge-in is NOT driven by a standalone VADProcessor.
    # The analyzer is handed to the LLM user aggregator (below), whose
    # internal UserTurnController + VADUserTurnStartStrategy turn "user
    # started speaking" into an InterruptionFrame that stops the bot's TTS.
    # The analyzer must run at the rate of the InputAudioRawFrames it will
    # actually see, which is AFTER the resampler (16 kHz when ASR is on).
    # Silero rejects anything but 8000/16000, so this is not optional.
    #
    # ONE ANALYZER PER CONSUMER. A VADAnalyzer is stateful -- it accumulates
    # bytes in ``_vad_buffer`` and advances ``_vad_stopping_count`` inside
    # ``_run_analyzer``. Handing the SAME instance to both ``VADProcessor``
    # and ``LLMUserAggregatorParams`` makes both controllers feed it every
    # audio frame, so each 32 ms window is counted twice and every gate runs
    # at half its configured value:
    #
    #     stop_secs 0.7 -> 11 quiet windows instead of 22 -> 0.35 s
    #     start_secs 0.25 -> 4 speaking windows instead of 8  -> 0.13 s
    #
    # Measured on a two-part utterance ("Senate elections." / pause /
    # "As in senators.") fed through the real pipeline: shared -> the VAD
    # reported a stop after 0.33 s of silence and the question was split in
    # two at every pause tested; separate -> 0.71 s, as configured.
    #
    # This is also why ``VAD_STOP_SECS`` "felt" shorter than 0.7 s for the
    # whole life of this file, and why ``VAD_CONFIDENCE`` had to be pushed to
    # 0.6 to stop keyboard clicks starting turns: the start debounce was
    # 0.13 s, not 0.25 s, so a click outlasted it.
    def _make_vad_analyzer() -> SileroVADAnalyzer:
        return SileroVADAnalyzer(
            sample_rate=proc_rate,
            params=VADParams(
                confidence=config.VAD_CONFIDENCE,
                start_secs=config.VAD_START_SECS,
                stop_secs=config.VAD_STOP_SECS,
                min_volume=config.VAD_MIN_VOLUME,
            ),
        )

    vad_analyzer = _make_vad_analyzer()          # for VADProcessor (downstream)
    aggregator_vad_analyzer = _make_vad_analyzer()  # for the user aggregator

    # --- LLM (OnIt, OpenAI-compatible, raw-audio input) --------------------
    # The clock goes in even without the bridge. Before this, the whole prompt
    # extension was gated on tools, so an agent with tools off had no date and
    # no time AND no bash to read them with -- it could only say it did not
    # know. The clock is one line and answers "what time is it" by itself.
    system_prompt = (config.SYSTEM_PROMPT + " " + config.OWNER_PROMPT
                     + " " + config.clock_line()
                     + " " + config.CLOCK_GUARD).strip()
    if bridge is not None:
        system_prompt = (system_prompt + " " + config.TOOLS_SYSTEM_PROMPT).strip()
    llm_settings: dict = dict(
        model=config.LLM_MODEL,
        system_instruction=system_prompt,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    if not config.LLM_THINKING:
        # Thinking models (Qwen3.x) return empty `content` by default; their
        # reasoning goes to a separate field. Disable it so the voice agent
        # actually gets text to speak. vLLM-specific, so it rides in
        # `extra` -> `extra_body`.
        llm_settings["extra"] = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        }
    llm = ClockedLLMService(
        api_key=api_key,
        base_url=config.LLM_BASE_URL,
        settings=OpenAILLMService.Settings(**llm_settings),
    )

    # --- TTS (Piper, local) -------------------------------------------------
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=config.PIPER_VOICE),
        download_dir=config.PIPER_VOICE_DIR,
    )

    # --- ASR (local, no API key) --------------------------------------------
    # Two backends (see config.ASR_BACKEND):
    #
    #   streaming -- sherpa-onnx zipformer decodes the mic as it arrives, so
    #       the words appear while you are still talking and the final text is
    #       ready the moment you stop. This is the default.
    #   whisper   -- the old segmented faster-whisper path: silence until you
    #       stop, then ~4.9 s of transcription on this Pi.
    #
    # Either way the mic's 48 kHz is resampled to 16 kHz first, and the LLM
    # consumes TEXT rather than raw audio, which is what lets the dashboard
    # show your words.
    stt = None
    if config.ASR_ENABLED:
        if config.ASR_BACKEND == "streaming":
            stt = SherpaStreamingSTTService(
                model_dir=config.ASR_STREAM_MODEL_DIR,
                num_threads=config.ASR_THREADS,
                final_pass=config.ASR_FINAL_PASS,
                final_model=config.ASR_FINAL_MODEL,
                final_compute=config.ASR_COMPUTE,
                language=config.ASR_LANGUAGE,
                pre_roll_secs=config.ASR_PREROLL_SECS,
                # Measured: the streaming final text is ready at speech end, so
                # the only wait is the faster-whisper pass (~0.9 s for a short
                # turn). This is the safety net the turn stop strategy uses if
                # a transcript never arrives.
                ttfs_p99_latency=1.2,
            )
        else:
            stt = WhisperSTTService(
                settings=WhisperSTTService.Settings(
                    model=config.ASR_MODEL,
                    language=config.ASR_LANGUAGE,
                ),
                device=config.ASR_DEVICE,
                compute_type=config.ASR_COMPUTE,
            )

    # --- LLM user aggregator (context + barge-in) ----------------------------
    # This is the stage that drives barge-in in 1.15: it owns the VAD analyzer
    # and the UserTurnController, and when the user starts speaking it
    # broadcasts an InterruptionFrame that stops the bot's TTS. It must ALWAYS
    # be in the pipeline -- even with no tools -- or barge-in silently does not
    # work. (Previously it was only created when a tool bridge existed, which
    # disabled barge-in in the default no-tools, ASR-on configuration.)
    user_params = LLMUserAggregatorParams(
        vad_analyzer=aggregator_vad_analyzer,
        # VAD-only turn start: the turn (and the barge-in) fires the moment the
        # analyzer hears speech, without waiting for a transcript.
        #
        # The stop side is what used to cost 3 of the 5.2 s on every turn. The
        # 1.15 default is Smart Turn (TurnAnalyzerUserTurnStopStrategy), which
        # waits 3 s of silence and then asks the STT for text it does not have.
        # With streaming ASR the text is already in hand when you stop, so a
        # plain speech timeout is both faster and more predictable: the turn
        # ends TURN_STOP_SECS after the VAD stop.
        user_turn_strategies=UserTurnStrategies(
            start=[VADUserTurnStartStrategy()],
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=config.TURN_STOP_SECS)],
        ),
        # Echo guard for the agent's own fixed announcements.
        #
        # There is no AEC anywhere in this pipeline (no pywebrtc_audio filter,
        # no Speex), so the mic hears the speaker. The tool bridge speaks a
        # fixed line ("Searching for that.") before every tool call, and that
        # line comes back through the mic as speech: measured in
        # /tmp/wandog_events.jsonl, 5 of the 7 cancelled tool calls were
        # cancelled by the agent's own announcement -- user_partial "Search"
        # -> user_text "Searching for that." 0.6-1.1s after the tool started,
        # which raises a VAD turn start, which broadcasts InterruptionFrame,
        # which cancels the in-flight search. The model then had to say
        # "it looks like that search was cancelled" (idx 2097, 2123).
        #
        # FunctionCallUserMuteStrategy suppresses user frames -- VAD frames
        # included, they are in the mute list in
        # llm_response_universal.py:_should_mute -- for exactly as long as a
        # function call is in progress, which is precisely the window the
        # announcement occupies. Barge-in on the agent's *answers* is
        # untouched: the mute lifts on FunctionCallResultFrame.
        user_mute_strategies=[FunctionCallUserMuteStrategy()],
        # Hard ceiling so a turn can never hang open waiting for text.
        user_turn_stop_timeout=config.TURN_STOP_TIMEOUT_SECS,
    )
    #
    # BOTH aggregators must share ONE context. The user half is what pushes
    # LLMContextFrame -- that object IS the prompt the model receives. The
    # assistant half records the bot's replies by writing them into ITS
    # context. Give them two different LLMContext objects and the model's
    # replies are written to a context nobody ever sends: every prompt becomes
    #
    #     [system, user("Q1"), user("Q2"), user("Q3"), ...]
    #
    # with no assistant turns at all. The model then sees a stack of
    # unanswered questions and picks whichever one it likes -- which is
    # exactly the "it answers the previous question" symptom. Tools went to
    # the assistant context too, so the model was also told nothing about
    # them. One shared context fixes both.
    # The prompt lives on the LLM service's `system_instruction`, NOT in the
    # context. Putting it in both sends it TWICE: the adapter prepends the
    # settings copy and keeps the context copy, so every request carried two
    # identical 1400-char system messages (measured -- see test_clock.py).
    # Pipecat warns about exactly this and calls the context copy deprecated.
    # It also made the clock unfixable: the stale copy in the context would
    # have shadowed any refresh of the settings one.
    context_kwargs = {"messages": []}
    if bridge is not None and bridge.tools:
        context_kwargs["tools"] = bridge.tools
    context = LLMContext(**context_kwargs)
    user_agg = LLMUserAggregator(context, params=user_params)

    # The assistant aggregator records the bot's replies into the SAME context.
    # It is also the ONLY place the bot's finished reply text is available (it
    # consumes LLMTextFrame), so the dashboard's bot_text event comes from it.
    # It is therefore always built, not just when tools are present.
    assistant_agg = LLMAssistantAggregator(
        context,
        params=LLMAssistantAggregatorParams(),
    )
    # Hook the transcript events before the pipeline runs so no turn is missed.
    wire_transcript_events(user_agg, assistant_agg, bus)

    # --- Pipeline ------------------------------------------------------------
    # mic -> resample -> ASR -> user agg (VAD + barge-in) -> LLM -> [assistant
    # agg] -> TTS. The user aggregator sits after ASR so it sees the user's
    # audio (for VAD) and the transcript (for context) and pushes the
    # LLMContextFrame to the LLM. When ASR is off the LLM takes raw audio.
    # VAD MUST run before the STT. SegmentedSTTService only transcribes when it
    # receives VADUserStoppedSpeakingFrame, and its process_frame() swallows
    # both VAD frames without pushing them downstream. With the VAD living
    # only in the user aggregator (downstream of the STT), the STT waited
    # forever for a stop frame that could never arrive -- so Whisper never
    # flushed a segment and the agent appeared completely deaf. VADProcessor
    # broadcasts its frames to every processor, so the STT and the aggregator
    # both receive them.
    stages = [
        input_transport,
        AudioResampleProcessor(capture_rate, proc_rate),
        VADProcessor(vad_analyzer=vad_analyzer),
    ]
    if stt is not None:
        stages.append(stt)
        # Publish the live partial transcripts the streaming ASR emits. Must sit
        # between the STT and the aggregator: the aggregator consumes them.
        stages.append(PartialTranscriptTap(bus))
    # The mic toggle pushes STTMuteFrame UPSTREAM, so it must sit downstream of
    # the STT for that frame to reach it.
    stages.append(MicControlWatcher(config.DASHBOARD_MIC_FILE))
    # Typed text and the Stop button from the dashboard.
    #
    # MUST sit BEFORE the user aggregator. A typed message is delivered by
    # queue_frame(), which pushes the frame into this stage's own input queue
    # and lets it flow DOWNSTREAM from here -- so anything upstream of this
    # point never sees it, and the aggregator that actually handles
    # LLMMessagesAppendFrame is downstream. Placing this after the aggregator
    # made Send a no-op: the frame was queued, flowed away from the aggregator,
    # and the question was never added to the context.
    #
    # A stop is an InterruptionFrame, which broadcast_interruption() sends in
    # both directions, so it reaches the STT and the transports upstream and
    # the LLM and TTS downstream from here either way.
    stages.append(DashboardControlWatcher(
        config.DASHBOARD_TYPED_FILE,
        config.DASHBOARD_STOP_FILE,
        bus=bus,
        poll_secs=config.DASHBOARD_CONTROL_POLL_SECS,
        clear_path=config.DASHBOARD_CLEAR_FILE,
    ))
    stages.append(user_agg)
    stages.append(llm)
    # TTS comes straight after the LLM. The assistant aggregator must come AFTER
    # it, not before: LLMAssistantAggregator.process_frame() has no TextFrame
    # branch, so it CONSUMES LLMTextFrame and never pushes it downstream. With
    # the aggregator between the LLM and the TTS the TTS received no text at
    # all and the agent was silent -- while the LLM had in fact answered.
    # This is the canonical Pipecat order: llm -> tts -> output -> assistant_agg.
    stages.append(tts)
    stages.append(output_transport)
    if assistant_agg is not None:
        stages.append(assistant_agg)
    # The logger observes speaking frames, which are pushed downstream all the
    # way to the end of the pipeline, so it stays last. The mic toggle is NOT
    # here: it pushes STTMuteFrame UPSTREAM, so it must sit after the STT.
    stages.append(WanDogLogger(bus))
    pipeline = Pipeline(stages)
    task = PipelineTask(pipeline)

    # The tool bridge speaks its acknowledgements by queuing a TTSSpeakFrame,
    # and a frame enters the pipeline through the task. The bridge is not a
    # stage in the list above (the tools live on the LLM context, not in the
    # chain), so hand it the task explicitly.
    if bridge is not None:
        bridge.task = task

    return pipeline, task, input_transport, output_transport


async def run() -> None:
    api_key = config.resolve_llm_api_key()

    # Event bus: the record of the interaction. The dashboard tails the JSONL
    # log this writes and pings it for live approval decisions.
    import eventbus
    bus = eventbus.BUS

    # Build the OnIt MCP tool bridge (spawns OnIt's stdio servers). None when
    # tools are disabled or OnIt's interpreter is missing.
    #
    # The Guard is the harness side of OnIt's approval safeguard: when a
    # command is not on the allowlist, the OnIt server returns a
    # needs_approval ticket and the Guard asks a person (dashboard first, then
    # the console) whether to redeem it. Fail-closed: no answer within the
    # timeout = deny.
    guard = mcp_bridge.Guard(timeout=config.MCP_APPROVAL_TIMEOUT, bus=bus)
    bridge = mcp_bridge.build_bridge(guard=guard)
    if bridge is not None:
        await bridge.discover()

    pipeline, task, input_transport, output_transport = build_pipeline(
        api_key, bridge, bus)

    await bus.emit("session_start",
                   llm=config.LLM_MODEL, base_url=config.LLM_BASE_URL,
                   asr=describe_asr(),
                   tools=len(bridge.tools) if bridge else 0)

    logger.info("WanDog starting")
    logger.info("  LLM : {} @ {}", config.LLM_MODEL, config.LLM_BASE_URL)
    logger.info("  ASR : {}", describe_asr())
    logger.info("  TTS : Piper {} ({})", config.PIPER_VOICE, config.PIPER_VOICE_DIR)
    if config.DASHBOARD_ENABLED:
        logger.info("  Dash: on (log: {}, approval via {} / {})",
                    config.DASHBOARD_LOG,
                    Path(config.DASHBOARD_LOG).with_name(
                        Path(config.DASHBOARD_LOG).name + ".pending.json"),
                    Path(config.DASHBOARD_LOG).with_name(
                        Path(config.DASHBOARD_LOG).name + ".decision.json"))
    if bridge is not None:
        logger.info("  Tools: {} OnIt MCP tools (jail: {})",
                    len(bridge.tools), config.MCP_DATA_PATH)
        logger.info("  Guard: approval gate on (timeout {}s, fail-closed)",
                    config.MCP_APPROVAL_TIMEOUT)
        # Which tool keys resolved, and from where. Names and sources only --
        # never a value. A MISSING key is not fatal: that one tool degrades.
        for line in config.describe_tool_secrets():
            logger.info("    secret: {}", line)
    logger.info(
        "  Audio: in {} Hz, out {} Hz", config.INPUT_SAMPLE_RATE, config.OUTPUT_SAMPLE_RATE
    )
    logger.info("  Ctrl+C to stop")

    runner = WorkerRunner()
    await runner.add_workers(task)

    # runner.run() is what actually STARTS the registered workers. Adding a
    # worker only attaches it to the bus and registry; nothing is started
    # until run() is entered (see WorkerRunner.add_workers docstring). This
    # was missing, so the pipeline never started: no StartFrame, the input
    # transport never opened the mic, VAD never ran, ASR never transcribed,
    # and the dashboard only ever saw session_start.
    #
    # auto_end=False keeps the runner alive until we cancel it, instead of
    # exiting the moment the single pipeline worker finishes.
    try:
        await runner.run(auto_end=False)
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down...")
    finally:
        await runner.cancel()
        if bridge is not None:
            await bridge.close()
        await bus.emit("session_end")
        logger.info("Bye!")


def main() -> None:
    try:
        asyncio.run(run())
    except SystemExit:
        raise
    except Exception as e:
        logger.exception("Fatal: {}", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
