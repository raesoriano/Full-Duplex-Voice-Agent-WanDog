"""Guard: a muted mic must stop the VAD, not only the STT.

The bug this covers
-------------------
Reported as "the VAD trips even when I am muted from the button on the
dashboard".

``MicControlWatcher`` pushes ``STTMuteFrame`` upstream when the dashboard is
muted, and that frame is honoured in exactly one place: ``STTService``
(``services/stt_service.py``, ``elif isinstance(frame, STTMuteFrame)`` sets
``self._muted``). ``VADProcessor`` has no such branch -- it passes the frame
straight through -- so the analyzer kept running over the microphone while the
UI said muted.

The pipeline is ordered transport -> resample -> VADProcessor -> STT, so the
VAD is UPSTREAM of the STT and never saw the mute at all. The user-visible
result: with the mic muted, speech still produced
``VADUserStartedSpeakingFrame``, the user aggregator turned that into
``UserStartedSpeakingFrame``, ``WanDogLogger`` emitted ``user_started`` on the
bus, and the dashboard opened a turn and showed the agent listening. No
transcript ever followed, because the STT was discarding the audio.

This test measures the VAD directly against real audio, so it fails on the
behaviour rather than on the wiring: it does not care how the mute is plumbed
through, only that speech while muted produces no speech-start frames.

Run:  .venv/bin/python test_muted_vad.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
import soxr

import agent
import config
from pipecat.frames.frames import (
    InputAudioRawFrame,
    StartFrame,
    UserSpeakingFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.audio.vad_processor import VADProcessor
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

CLIP = "/tmp/wandog_speech.wav"   # any short speech clip; skipped if absent
RATE = 16000                      # Silero accepts only 8000/16000
CHUNK_MS = 20


def load_speech(path: str = CLIP) -> bytes:
    """Return the clip as 16 kHz mono s16le, which is what the VAD sees."""
    with wave.open(path) as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    if rate != RATE:
        audio = soxr.resample(
            audio.astype(np.float32), rate, RATE, quality="HQ"
        ).astype(np.int16)
    return audio.tobytes()


class SpeechSink(FrameProcessor):
    """Counts the frames that mean 'the agent heard something'."""

    def __init__(self):
        super().__init__()
        self.started = 0        # VADUserStartedSpeakingFrame
        self.speaking = 0       # UserSpeakingFrame (periodic activity)
        self.audio = 0          # InputAudioRawFrame that got this far
        self.ready = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.ready.set()
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self.started += 1
        elif isinstance(frame, UserSpeakingFrame):
            self.speaking += 1
        elif isinstance(frame, InputAudioRawFrame):
            self.audio += 1
        await self.push_frame(frame, direction)


def make_analyzer():
    """The same analyzer agent.py builds for VADProcessor."""
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    return SileroVADAnalyzer(
        sample_rate=RATE,
        params=VADParams(
            confidence=config.VAD_CONFIDENCE,
            start_secs=config.VAD_START_SECS,
            stop_secs=config.VAD_STOP_SECS,
            min_volume=config.VAD_MIN_VOLUME,
        ),
    )


async def run_clip(muted: bool) -> SpeechSink:
    """Feed the clip through VAD -> MicControlWatcher, exactly as agent.py wires
    it, with the dashboard file saying either muted or live."""
    tmp = Path(tempfile.mkdtemp())
    mic = tmp / "mic.json"
    mic.write_text(json.dumps({"enabled": not muted}), encoding="utf-8")

    sink = SpeechSink()
    # The real order (agent.py): the mic gate sits after the resampler and
    # BEFORE the VADProcessor, so it can swallow the microphone before either
    # analyzer sees it. Reproduced verbatim -- the gate ahead of the VAD is the
    # fix, so a test that kept the gate behind the VAD would measure the bug.
    pipeline = Pipeline([
        agent.MicControlWatcher(str(mic), poll_secs=0.05),
        VADProcessor(vad_analyzer=make_analyzer()),
        sink,
    ])
    task = PipelineTask(pipeline)
    runner = WorkerRunner()
    await runner.add_workers(task)
    running = asyncio.create_task(runner.run(auto_end=False))

    try:
        await asyncio.wait_for(sink.ready.wait(), timeout=5)
        # Let the watcher complete its first poll and push the mute upstream
        # before any audio is fed, so this measures the steady state and not a
        # race with startup.
        await asyncio.sleep(0.4)

        audio = load_speech()
        step = RATE * CHUNK_MS // 1000 * 2      # bytes per chunk (s16le)
        for i in range(0, len(audio) - step + 1, step):
            await task.queue_frame(InputAudioRawFrame(
                audio=audio[i:i + step], sample_rate=RATE, num_channels=1,
            ))
            await asyncio.sleep(CHUNK_MS / 1000)
        await asyncio.sleep(0.6)                # let the VAD settle
        return sink
    finally:
        await runner.cancel()
        running.cancel()
        try:
            await running
        except (asyncio.CancelledError, Exception):
            pass


async def main() -> int:
    if not Path(CLIP).exists():
        print(f"SKIP: no speech clip at {CLIP}")
        return 0

    failures: list[str] = []
    print("1. speech with the mic LIVE reaches the VAD")
    live = await run_clip(muted=False)
    ok = live.started > 0
    print(f"   VAD start frames {live.started:<3} activity {live.speaking:<3} "
          f"{'ok' if ok else 'WRONG'}")
    if not ok:
        # If the control case does not fire, the muted case proves nothing.
        failures.append(
            "the VAD did not fire on real speech even with the mic live, so "
            "the muted comparison below would be vacuous"
        )
        print("\nFAIL: control case did not fire; cannot judge the mute")
        for f in failures:
            print("FAIL:", f)
        return 1

    print("2. the same speech while MUTED must not reach the VAD")
    muted = await run_clip(muted=True)
    ok = muted.started == 0 and muted.speaking == 0
    print(f"   VAD start frames {muted.started:<3} activity {muted.speaking:<3} "
          f"{'ok' if ok else 'WRONG'}")
    if not ok:
        failures.append(
            f"the mic was muted and the VAD still fired {muted.started} "
            f"start frame(s) and {muted.speaking} activity frame(s). The "
            "dashboard will open a turn and show the agent listening while no "
            "transcript can possibly follow."
        )

    print("3. muted audio must not reach the aggregator's analyzer")
    ok = muted.audio == 0
    print(f"   raw audio past the watcher {muted.audio:<3} "
          f"{'ok' if ok else 'WRONG'}")
    if not ok:
        failures.append(
            f"{muted.audio} InputAudioRawFrame(s) still reached the stage after "
            "the mute. The user aggregator owns a SECOND VAD analyzer and "
            "would trip on them."
        )

    if failures:
        print()
        for f in failures:
            print("FAIL:", f)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
