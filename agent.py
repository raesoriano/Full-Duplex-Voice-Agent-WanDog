"""Fweddie -- the heart of the voice agent.

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
        while True:
            enabled = self._read()
            if enabled != self._last:
                if self._last is not None:
                    logger.info("Mic {}", "enabled" if enabled else "muted")
                    # Upstream: the STT sits before us in the pipeline.
                    await self.push_frame(
                        STTMuteFrame(mute=not enabled), FrameDirection.UPSTREAM
                    )
                self._last = enabled
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


class FweddieLogger(FrameProcessor):
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


class PartialTranscriptTap(FrameProcessor):
    """Publish live partial transcripts to the event bus.

    The streaming ASR emits an ``InterimTranscriptionFrame`` every few hundred
    milliseconds while the user is talking. Those frames are consumed by the
    user aggregator (they never reach the tail of the pipeline), so the
    dashboard cannot see them from ``FweddieLogger``. This tap sits between the
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


def wire_transcript_events(user_agg, assistant_agg, bus) -> None:
    """Record the user's words and the bot's reply on the event bus.

    These two texts exist only inside the aggregators (see ``FweddieLogger``),
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
    vad_analyzer = SileroVADAnalyzer(
        sample_rate=proc_rate,
        params=VADParams(
            confidence=config.VAD_CONFIDENCE,
            start_secs=config.VAD_START_SECS,
            stop_secs=config.VAD_STOP_SECS,
            min_volume=config.VAD_MIN_VOLUME,
        ),
    )

    # --- LLM (OnIt, OpenAI-compatible, raw-audio input) --------------------
    system_prompt = config.SYSTEM_PROMPT
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
    llm = OpenAILLMService(
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
        vad_analyzer=vad_analyzer,
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
    context_kwargs = {"messages": [{"role": "system", "content": system_prompt}]}
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
    stages.append(MicControlWatcher(
        str(Path(config.DASHBOARD_LOG).with_suffix(".mic_control.json"))))
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
    stages.append(FweddieLogger(bus))
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

    logger.info("Fweddie starting")
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
