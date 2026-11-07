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

import pyaudio
from loguru import logger

import config
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    LLMTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.piper.tts import PiperTTSService
from pipecat.transports.local.audio import (
    LocalAudioInputTransport,
    LocalAudioOutputTransport,
    LocalAudioTransportParams,
)
from pipecat.workers.runner import WorkerRunner

logging.basicConfig(level=logging.INFO)


class FweddieLogger(FrameProcessor):
    """Tiny observer that prints what the agent hears and says."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await self.push_frame(frame, direction)

        if isinstance(frame, UserStartedSpeakingFrame):
            logger.info("[user] started speaking")
        elif isinstance(frame, UserStoppedSpeakingFrame):
            logger.info("[user] stopped speaking")
        elif isinstance(frame, BotStartedSpeakingFrame):
            logger.info("[fweddie] started speaking")
        elif isinstance(frame, BotStoppedSpeakingFrame):
            logger.info("[fweddie] stopped speaking")
        elif isinstance(frame, LLMTextFrame):
            logger.info("[llm] {}", frame.text)
        elif isinstance(frame, ErrorFrame):
            logger.error("[error] {}", frame.error)


def build_pipeline(api_key: str):
    """Construct the Pipecat pipeline. Returns (pipeline, task, input, output)."""

    # --- Audio transport (mic + speaker) -----------------------------------
    pa = pyaudio.PyAudio()
    transport_params = LocalAudioTransportParams(
        audio_in_enabled=True,
        audio_in_sample_rate=config.INPUT_SAMPLE_RATE,
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
    llm = OpenAILLMService(
        api_key=api_key,
        base_url=config.LLM_BASE_URL,
        settings=OpenAILLMService.Settings(
            model=config.LLM_MODEL,
            system_instruction=config.SYSTEM_PROMPT,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        ),
    )

    # --- TTS (Piper, local) -------------------------------------------------
    tts = PiperTTSService(
        settings=PiperTTSService.Settings(voice=config.PIPER_VOICE),
        download_dir=config.PIPER_VOICE_DIR,
    )

    # --- Pipeline ------------------------------------------------------------
    # Input transport emits InputAudioRawFrame; the LLM service (OpenAI
    # compatible) accepts raw audio input frames directly, so no STT stage.
    # The system prompt (persona) is carried on the LLM service's settings.
    pipeline = Pipeline(
        [
            input_transport,
            vad,
            llm,
            tts,
            output_transport,
            FweddieLogger(),
        ]
    )
    task = PipelineTask(pipeline)

    return pipeline, task, input_transport, output_transport


async def run() -> None:
    api_key = config.resolve_llm_api_key()

    pipeline, task, input_transport, output_transport = build_pipeline(api_key)

    logger.info("Fweddie starting")
    logger.info("  LLM : {} @ {}", config.LLM_MODEL, config.LLM_BASE_URL)
    logger.info("  TTS : Piper {} ({})", config.PIPER_VOICE, config.PIPER_VOICE_DIR)
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
