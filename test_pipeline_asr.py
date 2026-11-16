"""Whole-pipeline test: wav in -> VAD -> streaming ASR -> turn -> LLM prompt.

``test_stream_asr.py`` proves the ASR *stage* streams. This proves the
*pipeline* turns that into a prompt quickly, and that the prompt carries the
words just spoken.

The real LLM and TTS are replaced by recorders: they are slow and
network-bound and can say nothing about ASR latency. What is measured is the
part that was broken --

    speech ends  ->  how long until the turn is submitted to the LLM,
    and does that prompt hold the CURRENT question?

Audio is fed at 48 kHz, so the 48k -> 16k resampler is exercised too.

Reference figure: 5.2 s from "stopped speaking" to turn-submitted, on every
turn of a live session before this change.

Run:  .venv/bin/python test_pipeline_asr.py
"""
from __future__ import annotations

import asyncio
import sys
import time
import wave

import numpy as np

import config
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TranscriptionFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
    SpeechTimeoutUserTurnStopStrategy,
)
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner
from resample import AudioResampleProcessor
from streaming_asr import SherpaStreamingSTTService

CAPTURE_RATE = 48000
PROC_RATE = 16000
CHUNK = 960          # 20 ms at 48 kHz
TAIL_SECS = 3.0

TURN_SUBMIT_BUDGET = 2.5   # VAD stop -> user message added. Was 5.2 s.

QUESTIONS = [
    ("/tmp/q_math.wav", ("2 plus 2", "two plus two")),
    ("/tmp/q_dog.wav", ("dog",)),
    ("/tmp/q_long.wav", ("plus", "dog")),   # 3.5 s: longer than the old 3 s stop timeout
    ("/tmp/q_vlong.wav", ("plus", "dog")),  # 5.6 s: longer than even the 5 s ceiling
]


def load48k(path: str) -> np.ndarray:
    w = wave.open(path)
    sr, ch = w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != CAPTURE_RATE:
        idx = np.arange(int(len(a) * CAPTURE_RATE / sr)) * sr / CAPTURE_RATE
        a = np.interp(idx, np.arange(len(a)), a).astype(np.float32)
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)


class Recorder(FrameProcessor):
    """Records what the ASR produced, from the one place it is still visible.

    It must sit BEFORE the aggregator: the aggregator consumes
    ``TranscriptionFrame`` and ``InterimTranscriptionFrame``, so nothing
    downstream of it ever sees them.
    """

    def __init__(self):
        super().__init__()
        self.partials: list[tuple[float, str]] = []
        self.finals: list[tuple[float, str]] = []
        self.vad_stops: list[float] = []
        self.ready = asyncio.Event()
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            # The pipeline can accept real-time audio from here on.
            self._t0 = time.monotonic()
            self.ready.set()
        elif isinstance(frame, InterimTranscriptionFrame):
            self.partials.append((self.now(), frame.text))
        elif isinstance(frame, TranscriptionFrame):
            self.finals.append((self.now(), frame.text))
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self.vad_stops.append(self.now())
        await self.push_frame(frame, direction)


async def run_one(path: str) -> dict | None:
    try:
        clip = load48k(path)
    except FileNotFoundError:
        print(f"SKIP: {path} is not present")
        return None

    vad = SileroVADAnalyzer(
        sample_rate=PROC_RATE,
        params=VADParams(
            confidence=config.VAD_CONFIDENCE,
            start_secs=config.VAD_START_SECS,
            stop_secs=config.VAD_STOP_SECS,
            min_volume=config.VAD_MIN_VOLUME,
        ),
    )
    stt = SherpaStreamingSTTService(
        model_dir=config.ASR_STREAM_MODEL_DIR,
        num_threads=config.ASR_THREADS,
        final_pass=config.ASR_FINAL_PASS,
        final_model=config.ASR_FINAL_MODEL,
        final_compute=config.ASR_COMPUTE,
        language=config.ASR_LANGUAGE,
        pre_roll_secs=config.ASR_PREROLL_SECS,
        ttfs_p99_latency=1.2,
    )
    user_agg = LLMUserAggregator(
        LLMContext(messages=[{"role": "system", "content": "test"}]),
        params=LLMUserAggregatorParams(
            vad_analyzer=vad,
            user_turn_strategies=UserTurnStrategies(
                start=[VADUserTurnStartStrategy()],
                stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=config.TURN_STOP_SECS)],
            ),
            user_turn_stop_timeout=config.TURN_STOP_TIMEOUT_SECS,
        ),
    )
    rec = Recorder()

    # "Turn submitted to the LLM" is exactly this event -- the same one
    # agent.py wires to publish user_text to the dashboard.
    submitted: list[tuple[float, str]] = []

    @user_agg.event_handler("on_user_turn_message_added")
    async def _on_user_message(aggregator, message):
        submitted.append((rec.now(), (getattr(message, "content", "") or "").strip()))

    task = PipelineTask(
        Pipeline(
            [
                AudioResampleProcessor(CAPTURE_RATE, PROC_RATE),
                VADProcessor(vad_analyzer=vad),
                stt,
                rec,
                user_agg,
            ]
        )
    )
    runner = WorkerRunner()
    await runner.add_workers(task)

    pad = lambda secs: np.zeros(int(CAPTURE_RATE * secs), dtype=np.int16)
    audio = np.concatenate([pad(0.2), clip, pad(TAIL_SECS)])

    async def feed():
        # Wait until the pipeline is live, or the queued audio is consumed
        # faster than real time and the VAD sees a burst instead of speech.
        await asyncio.wait_for(rec.ready.wait(), timeout=180)
        await asyncio.sleep(0.2)
        for i in range(0, len(audio), CHUNK):
            await task.queue_frame(
                InputAudioRawFrame(
                    audio=audio[i : i + CHUNK].tobytes(),
                    sample_rate=CAPTURE_RATE,
                    num_channels=1,
                )
            )
            await asyncio.sleep(0.02)
        await asyncio.sleep(3.0)
        await task.queue_frame(EndFrame())

    async def guard():
        # A hard stop so a wedged pipeline fails the test instead of hanging it.
        await asyncio.sleep(40)
        await task.queue_frame(CancelFrame())

    try:
        await asyncio.wait_for(asyncio.gather(runner.run(), feed(), guard()), timeout=60)
    except asyncio.TimeoutError:
        print("  WARNING: pipeline did not shut down cleanly (timeout)")

    return {
        "partials": rec.partials,
        "finals": rec.finals,
        "stops": rec.vad_stops,
        "submitted": submitted,
    }


async def main() -> int:
    ok = True
    for path, expect in QUESTIONS:
        r = await run_one(path)
        if r is None:
            return 2

        print(f"\n=== {path}")
        print("  --- partials (what the dashboard shows while you talk) ---")
        for t, text in r["partials"]:
            print(f"    {t:6.2f}s  {text!r}")
        print("  --- final transcripts ---")
        for t, text in r["finals"]:
            print(f"    {t:6.2f}s  {text!r}")
        print("  --- turns submitted to the LLM ---")
        for t, text in r["submitted"]:
            print(f"    {t:6.2f}s  {text!r}")

        if not r["partials"]:
            print("  FAIL: no partial transcript -- nothing would appear while you talk")
            ok = False
        else:
            print(f"  PASS: {len(r['partials'])} partial(s) while speaking, first {r['partials'][0][1]!r}")

        if not r["submitted"]:
            print("  FAIL: the turn never reached the LLM")
            ok = False
            continue
        if not r["stops"]:
            print("  FAIL: no VAD stop recorded")
            ok = False
            continue

        latency = r["submitted"][0][0] - r["stops"][0]
        verdict = "PASS" if latency <= TURN_SUBMIT_BUDGET else "FAIL"
        print(
            f"  {verdict}: turn submitted {latency:5.2f}s after speech ended"
            f" (budget {TURN_SUBMIT_BUDGET}s, was 5.2s)"
        )
        ok &= latency <= TURN_SUBMIT_BUDGET

        said = r["submitted"][0][1].lower()
        if not said:
            print("  FAIL: the turn reached the LLM with no text")
            ok = False
        else:
            hit = any(e in said for e in expect)
            print(f"  {'PASS' if hit else 'FAIL'}: the LLM was asked {said!r}")
            ok &= hit

    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
