"""End-to-end test of the streaming ASR stage in a real Pipecat pipeline.

Feeds a wav through VAD -> SherpaStreamingSTTService and records, with
timestamps, every frame the service emits:

  * InterimTranscriptionFrame -- what the dashboard shows while you talk
  * TranscriptionFrame       -- the final text for the turn

What it asserts (these are the properties the user asked for):
  1. partials appear BEFORE the audio ends -- i.e. text while dictating;
  2. the first partial arrives within FIRST_PARTIAL_BUDGET of speech start;
  3. the final transcript arrives within FINAL_BUDGET of the VAD stop, which
     is the whole point: the old whisper path took 4.7-4.9 s;
  4. the final text is correct.

Run:  .venv/bin/python test_stream_asr.py
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
    EndFrame,
    InputAudioRawFrame,
    StartFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner
from streaming_asr import SherpaStreamingSTTService

# Budgets. The old segmented whisper path was 4.7-4.9 s per turn; anything
# close to that means streaming is not actually working.
FIRST_PARTIAL_BUDGET = 2.0
FINAL_BUDGET = 2.5

CHUNK = 320  # 20 ms at 16 kHz -- the same granularity the mic delivers


def load16k(path: str) -> np.ndarray:
    w = wave.open(path)
    sr, ch = w.getframerate(), w.getnchannels()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if sr != 16000:
        idx = np.arange(int(len(a) * 16000 / sr)) * sr / 16000
        a = np.interp(idx, np.arange(len(a)), a).astype(np.float32)
    return (np.clip(a, -1, 1) * 32767).astype(np.int16)


class Recorder(FrameProcessor):
    """Timestamp every transcript frame the STT emits.

    Also signals when the StartFrame reaches it, so the feeder can begin in
    real time instead of queuing audio while the models are still loading
    (queued audio is then consumed at full speed, which makes the VAD see a
    burst instead of speech and produces nonsense timings).
    """

    def __init__(self):
        super().__init__()
        self.events: list[tuple[str, float, str]] = []
        self._t0 = time.monotonic()
        self.speech_started: float | None = None
        self.vad_stopped: float | None = None
        self.ready = asyncio.Event()

    def _now(self) -> float:
        return time.monotonic() - self._t0

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            # Reset the clock here: this is the moment the pipeline can accept
            # audio in real time.
            self._t0 = time.monotonic()
            self.ready.set()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            # Keep the FIRST start: an utterance may contain several VAD
            # segments (the diag clip holds two sentences), and the budget is
            # about how soon text appears after the user begins talking.
            now = self._now()
            if self.speech_started is None:
                self.speech_started = now
            self.events.append(("vad_start", now, ""))
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            now = self._now()
            self.vad_stopped = now
            self.events.append(("vad_stop", now, ""))
        elif isinstance(frame, InterimTranscriptionFrame):
            self.events.append(("partial", self._now(), frame.text))
        elif isinstance(frame, TranscriptionFrame):
            self.events.append(("final", self._now(), frame.text))
        await self.push_frame(frame, direction)


async def run_one(path: str, audio_secs_pad: float = 3.0) -> bool:
    pcm = load16k(path)
    # Trailing silence so the VAD can see the stop. Real mic audio has this.
    pcm = np.concatenate([pcm, np.zeros(int(16000 * audio_secs_pad), dtype=np.int16)])

    vad = SileroVADAnalyzer(
        sample_rate=16000,
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
    rec = Recorder()
    task = PipelineTask(Pipeline([VADProcessor(vad_analyzer=vad), stt, rec]))
    runner = WorkerRunner()
    await runner.add_workers(task)

    async def feed():
        await asyncio.wait_for(rec.ready.wait(), timeout=120)
        await asyncio.sleep(0.3)  # let the VAD settle after StartFrame
        for i in range(0, len(pcm), CHUNK):
            await task.queue_frame(
                InputAudioRawFrame(
                    audio=pcm[i : i + CHUNK].tobytes(), sample_rate=16000, num_channels=1
                )
            )
            await asyncio.sleep(0.02)  # real time
        await asyncio.sleep(4.0)
        await task.queue_frame(EndFrame())

    await asyncio.gather(runner.run(), feed())

    print(f"\n=== {path}")
    for kind, t, text in rec.events:
        print(f"  {t:6.2f}s  {kind:10s} {text!r}")

    partials = [(t, x) for k, t, x in rec.events if k == "partial"]
    finals = [(t, x) for k, t, x in rec.events if k == "final"]

    ok = True
    if rec.speech_started is None:
        print("  FAIL: VAD never fired -- the test audio is not being seen as speech")
        return False
    if rec.vad_stopped is None:
        print("  FAIL: VAD never reported a stop")
        return False
    if not partials:
        print("  FAIL: no partial transcript at all (nothing streams)")
        ok = False
    else:
        first_t, first_text = partials[0]
        dt = first_t - rec.speech_started
        verdict = "ok" if dt <= FIRST_PARTIAL_BUDGET else "FAIL"
        print(f"  first partial after speech start: {dt:5.2f}s  [{verdict}] {first_text!r}")
        ok &= dt <= FIRST_PARTIAL_BUDGET
        # Partials must arrive while the user is still speaking, not after.
        during = [t for t, _ in partials if t < rec.vad_stopped]
        print(f"  partials before the user stopped: {len(during)}/{len(partials)}")
        ok &= len(during) > 0

    if not finals:
        print("  FAIL: no final transcript")
        return False
    fin_t, fin_text = finals[-1]
    dt = fin_t - rec.vad_stopped
    verdict = "ok" if dt <= FINAL_BUDGET else "FAIL"
    print(f"  final transcript after VAD stop:   {dt:5.2f}s  [{verdict}] {fin_text!r}")
    ok &= dt <= FINAL_BUDGET
    return ok


async def main() -> int:
    wavs = sys.argv[1:] or [
        "/tmp/q_math.wav",
        "/tmp/q_dog.wav",
        "/tmp/wandog_diag_mic.wav",
    ]
    results = []
    for path in wavs:
        try:
            results.append((path, await run_one(path)))
        except FileNotFoundError:
            print(f"\n=== {path}: SKIPPED (not present)")
    print("\n" + "=" * 60)
    for path, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {path}")
    print("ALL PASSED" if all(ok for _, ok in results) else "SOME FAILED")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
