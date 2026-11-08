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

import config
import mcp_bridge
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    LLMTextFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.services.whisper.stt import WhisperSTTService
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

logging.basicConfig(level=logging.INFO)


class FweddieLogger(FrameProcessor):
    """Observer that prints what the agent hears and says, and records it to
    the event bus so the dashboard can show the interaction live.

    The LLM streams its reply as many small ``LLMTextFrame`` chunks; we buffer
    them and emit one ``bot_text`` event per reply (when the bot stops
    speaking), so the dashboard shows a whole sentence, not fragments.
    """

    def __init__(self, bus=None):
        super().__init__()
        self.bus = bus
        self._bot_buf: list[str] = []

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("[user] started speaking")
            if self.bus is not None:
                await self.bus.emit("user_started")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("[user] stopped speaking")
            if self.bus is not None:
                await self.bus.emit("user_stopped")
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info("[fweddie] started speaking")
            self._bot_buf = []
            if self.bus is not None:
                await self.bus.emit("bot_started")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info("[fweddie] stopped speaking")
            text = "".join(self._bot_buf).strip()
            self._bot_buf = []
            if text:
                logger.info("[fweddie] {}", text)
                if self.bus is not None:
                    await self.bus.emit("bot_text", text=text)
            if self.bus is not None:
                await self.bus.emit("bot_stopped")
        elif isinstance(frame, TranscriptionFrame):
            # The ASR stage emits the user's words here.
            text = (frame.text or "").strip()
            if text:
                logger.info("[user] {}", text)
                if self.bus is not None:
                    await self.bus.emit("user_text", text=text)
        elif isinstance(frame, LLMTextFrame):
            # Streamed bot reply chunk -- buffer it for the bot_text event.
            self._bot_buf.append(frame.text or "")
        elif isinstance(frame, LLMTextFrame):
            logger.info("[llm] {}", frame.text)
        elif isinstance(frame, ErrorFrame):
            logger.error("[error] {}", frame.error)


def build_pipeline(api_key: str, bridge: "mcp_bridge.ToolBridgeProcessor | None" = None,
                   bus=None):
    """Construct the Pipecat pipeline.

    Returns (pipeline, task, input, output). When ``bridge`` is given, its
    tools are registered on the LLM context so the model can call OnIt's
    tools; the system prompt is extended to tell it so. ``bus`` is the event
    bus the logger records to (the dashboard's data source).
    """

    # --- Audio transport (mic + speaker) -----------------------------------
    pa = pyaudio.PyAudio()
    # When ASR is on, the mic captures at 16 kHz (what faster-whisper wants)
    # instead of the 48 kHz the capture card natively produces. Output stays
    # at the TTS rate. This must be set BEFORE the transports are built.
    in_rate = config.ASR_SAMPLE_RATE if config.ASR_ENABLED else config.INPUT_SAMPLE_RATE
    transport_params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=in_rate,
        audio_in_channels=1,
        audio_out_enabled=True,
        audio_out_sample_rate=config.OUTPUT_SAMPLE_RATE,
        audio_out_channels=1,
        audio_out_volume=1.0,
    )
    input_transport = LocalAudioInputTransport(pa, transport_params)
    output_transport = LocalAudioOutputTransport(pa, transport_params)

    # --- VAD (Silero ONNX, torch-free) -------------------------------------
    # VADProcessor is the FrameProcessor that wraps the analyzer and pushes
    # UserStarted/StoppedSpeaking frames into the pipeline (drives barge-in).
    vad = VADProcessor(
        vad_analyzer=SileroVADAnalyzer(
            sample_rate=config.INPUT_SAMPLE_RATE,
            params=VADParams(
                confidence=config.VAD_CONFIDENCE,
                start_secs=config.VAD_START_SECS,
                stop_secs=config.VAD_STOP_SECS,
                min_volume=config.VAD_MIN_VOLUME,
            ),
        )
    )

    # --- LLM (OnIt, OpenAI-compatible, raw-audio input) --------------------
    system_prompt = config.SYSTEM_PROMPT
    if bridge is not None:
        system_prompt = (system_prompt + " " + config.TOOLS_SYSTEM_PROMPT).strip()
    llm = OpenAILLMService(
        api_key=api_key,
        base_url=config.LLM_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=config.LLM_MODEL,
            system_instruction=system_prompt,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        ),
    )

    # --- TTS (Piper, local) -------------------------------------------------
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=config.PIPER_VOICE),
        download_dir=config.PIPER_VOICE_DIR,
    )

    # --- ASR (faster-whisper, local -- no API key) --------------------------
    # The mic captures at 48 kHz; faster-whisper wants 16 kHz, so we resample
    # the transport to 16 kHz and let the LLM consume the transcribed TEXT
    # instead of raw audio. This is what makes the dashboard able to show the
    # user's words. Disabled (ASR_ENABLED=0) falls back to raw-audio LLM.
    stt = None
    if config.ASR_ENABLED:
        stt = WhisperSTTService(
            settings=WhisperSTTService.Settings(
                model=config.ASR_MODEL,
                language=config.ASR_LANGUAGE,
            ),
            device=config.ASR_DEVICE,
            compute_type=config.ASR_COMPUTE,
        )

    # --- LLM context (tools) -------------------------------------------------
    # In Pipecat 1.15 the LLM context (conversation history + tools) is
    # delivered to the LLM by a pair of context aggregators in the pipeline:
    # the user aggregator sits before the LLM (pushes an LLMContextFrame with
    # the current context + tools each turn) and the assistant aggregator sits
    # after it (records the reply). Without a bridge there is no context
    # (Fweddie just talks), which is the original behaviour.
    pair = None
    if bridge is not None and bridge.tools:
        context = LLMContext(
            messages=[{"role": "system", "content": system_prompt}],
            tools=bridge.tools,
        )
        pair = LLMContextAggregatorPair(context)

    # --- Pipeline ------------------------------------------------------------
    # mic -> VAD -> ASR -> [user agg] -> LLM -> [assistant agg] -> TTS -> speaker.
    # The ASR stage (faster-whisper) turns the user's audio into text; the LLM
    # consumes that text. When ASR is off, the LLM takes raw audio directly
    # (the original path).
    stages = [input_transport, vad]
    if stt is not None:
        stages.append(stt)
    if pair is not None:
        stages.append(pair.user())
    stages.append(llm)
    if pair is not None:
        stages.append(pair.assistant())
    stages += [tts, output_transport, FweddieLogger(bus)]
    pipeline = Pipeline(stages)
    task = PipelineTask(pipeline)

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
                   asr=("on:" + config.ASR_MODEL) if config.ASR_ENABLED else "off",
                   tools=len(bridge.tools) if bridge else 0)

    logger.info("Fweddie starting")
    logger.info("  LLM : {} @ {}", config.LLM_MODEL, config.LLM_BASE_URL)
    if config.ASR_ENABLED:
        logger.info("  ASR : {} ({})", config.ASR_MODEL, config.ASR_COMPUTE)
    else:
        logger.info("  ASR : off (raw-audio LLM)")
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
    logger.info(
        "  Audio: in {} Hz, out {} Hz", config.INPUT_SAMPLE_RATE, config.OUTPUT_SAMPLE_RATE
    )
    logger.info("  Ctrl+C to stop")

    runner = WorkerRunner()
    await runner.add_workers(task)

    # The task's start flows a StartFrame through the pipeline, bringing up
    # both transports. Block until the user hits Ctrl+C.
    try:
        while True:
            await asyncio.sleep(1)
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
