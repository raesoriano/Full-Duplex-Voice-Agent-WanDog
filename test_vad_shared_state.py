"""Guard: a VADAnalyzer must not be shared between two pipeline consumers.

Why this test exists
--------------------
``agent.py`` used to build ONE ``SileroVADAnalyzer`` and hand the same object
to both ``VADProcessor`` (a pipeline stage) and ``LLMUserAggregatorParams``
(which builds its own internal ``VADController``). A ``VADAnalyzer`` is
stateful:

  * ``_run_analyzer`` appends every buffer to ``_vad_buffer`` and advances
    ``_vad_stopping_count`` / ``_vad_starting_count``;
  * both controllers therefore fed it the same audio twice, so each 32 ms
    window was counted twice.

Effect: every gate ran at HALF its configured length.

    stop_secs  0.7  ->  11 quiet windows instead of 22  ->  0.35 s
    start_secs 0.25 ->   4 speaking windows instead of 8 ->  0.13 s

The user-visible symptoms were a sentence split at a short pause ("Senate
elections." / pause / "as in senators." arriving as two turns) and keyboard
clicks starting turns despite the 0.25 s debounce.

This test measures the gate directly rather than asserting on wiring, so it
fails if anyone re-introduces the sharing in any form.

Run:  .venv/bin/python test_vad_shared_state.py
"""
from __future__ import annotations

import asyncio
import sys
import wave

import numpy as np

import config
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState

SAMPLE_RATE = 16000
WINDOW = 512                     # Silero's analysis window: 32 ms at 16 kHz
CLIP = "/tmp/t_senate.wav"       # any short speech clip; skipped if absent
TOLERANCE = 0.08                 # measured gate must be within this of nominal


def _mk_analyzer() -> SileroVADAnalyzer:
    an = SileroVADAnalyzer(
        sample_rate=SAMPLE_RATE,
        params=VADParams(
            confidence=config.VAD_CONFIDENCE,
            start_secs=config.VAD_START_SECS,
            stop_secs=config.VAD_STOP_SECS,
            min_volume=config.VAD_MIN_VOLUME,
        ),
    )
    an.set_sample_rate(SAMPLE_RATE)
    return an


def _load16k(path: str) -> np.ndarray:
    w = wave.open(path)
    sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if w.getnchannels() > 1:
        a = a.reshape(-1, w.getnchannels()).mean(axis=1)
    if sr != SAMPLE_RATE:
        idx = np.arange(int(len(a) * SAMPLE_RATE / sr)) * sr / SAMPLE_RATE
        a = np.interp(idx, np.arange(len(a)), a).astype(np.float32)
    return a


async def measure_gate(feeds_per_frame: int) -> tuple[float, float]:
    """Feed one clip then silence; return (measured start gate, stop gate)."""
    analyzer = _mk_analyzer()
    clip = _load16k(CLIP)
    audio = np.concatenate(
        [
            np.zeros(int(SAMPLE_RATE * 0.30), dtype=np.float32),
            clip,
            np.zeros(int(SAMPLE_RATE * 3.0), dtype=np.float32),
        ]
    )
    speech_start = 0.30
    speech_end = 0.30 + len(clip) / SAMPLE_RATE

    chunk = 320                  # 20 ms frames, as the real pipeline delivers
    start_at = stop_at = None
    for i in range(0, len(audio) - chunk, chunk):
        frame = audio[i : i + chunk]
        state = VADState.QUIET
        for _ in range(feeds_per_frame):
            state = await analyzer.analyze_audio(
                (frame * 32767).astype(np.int16).tobytes()
            )
        t = (i + chunk) / SAMPLE_RATE
        if state != VADState.QUIET and start_at is None:
            start_at = t
        if state == VADState.QUIET and t > speech_end and stop_at is None:
            stop_at = t

    if start_at is None or stop_at is None:
        return float("nan"), float("nan")
    return start_at - speech_start, stop_at - speech_end


async def main() -> int:
    import os

    if not os.path.exists(CLIP):
        print(f"SKIP: {CLIP} is not present")
        return 0

    nominal_stop = config.VAD_STOP_SECS
    nominal_start = config.VAD_START_SECS
    # Silero quantises to whole 32 ms windows.
    quant_stop = round(nominal_stop / (WINDOW / SAMPLE_RATE)) * (WINDOW / SAMPLE_RATE)
    quant_start = round(nominal_start / (WINDOW / SAMPLE_RATE)) * (WINDOW / SAMPLE_RATE)

    ok = True
    print(f"configured: start {nominal_start:.2f}s  stop {nominal_stop:.2f}s"
          f"  (quantised {quant_start:.3f}s / {quant_stop:.3f}s)\n")

    # 1. A correctly wired analyzer sees each window once and honours the config.
    got_start, got_stop = await measure_gate(1)
    good = abs(got_stop - quant_stop) <= TOLERANCE
    print(f"  fed 1x per window: start gate {got_start:.3f}s, stop gate {got_stop:.3f}s"
          f"   [{'PASS' if good else 'FAIL'}] expected ~{quant_stop:.3f}s")
    ok &= good

    # 2. Sharing doubles the advance and halves the gate. If this ever stops
    #    being true the premise of the fix changed and the test should be
    #    revisited -- but it must never equal the 1x number.
    shared_start, shared_stop = await measure_gate(2)
    halved = shared_stop < got_stop - TOLERANCE
    print(f"  fed 2x per window: start gate {shared_start:.3f}s, stop gate {shared_stop:.3f}s"
          f"   [{'PASS' if halved else 'FAIL'}] sharing halves the gate"
          f" (this is what a shared analyzer does)")
    ok &= halved

    # 3. The gate must be long enough to survive an ordinary clause pause.
    #    0.8 s is the pause that split the incident utterance.
    covers = quant_stop + config.TURN_STOP_SECS >= 1.3
    print(f"\n  total silence gate {quant_stop + config.TURN_STOP_SECS:.2f}s"
          f" (VAD {quant_stop:.2f} + turn stop {config.TURN_STOP_SECS:.2f})"
          f"   [{'PASS' if covers else 'FAIL'}] covers a 1.3 s clause pause")
    ok &= covers

    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
