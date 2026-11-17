#!/usr/bin/env python3
"""WanDog -- is the microphone actually hearing you?

Run this, say something, and read the verdict. It uses the exact capture
rate, resampler, VAD settings and Whisper model the agent uses, so a PASS
here means the voice input path works end to end.

    .venv/bin/python diag_mic.py            # 10 second listen
    .venv/bin/python diag_mic.py --secs 20

It reports, in order:
  1. mic gain (the single most common cause of "it can't hear me")
  2. raw RMS / peak of what the mic returns  (is audio arriving at all?)
  3. how many 32 ms frames Silero scored as speech
  4. the Whisper transcript of whatever it heard

Interpretation guide:
  RMS < 0.002                 -> the mic is essentially silent. Hardware,
                                 gain, or the wrong device. Not a code bug.
  RMS >= 0.01, no VAD frames  -> VAD thresholds too strict for this room;
                                 lower WANDOG_VAD_CONFIDENCE.
  VAD fires, transcript empty -> speech is too faint/muffled for Whisper;
                                 move closer or raise the gain.
  Transcript has words        -> the input path is fine.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import wave

import numpy as np

import config

CAPTURE_RATE = 48000
CHUNK = 1920  # 20 ms at 48 kHz, mono, s16
WAV_PATH = "/tmp/wandog_diag_mic.wav"


def report_gain() -> None:
    print("1. MIC GAIN")
    if not config.MIC_CARD:
        print("   (skipped: WANDOG_MIC_CARD is empty)")
        return
    try:
        r = subprocess.run(
            ["amixer", "-c", config.MIC_CARD, "sget", "Mic"],
            capture_output=True, text=True, timeout=5,
        )
        line = [ln for ln in r.stdout.splitlines() if "Capture" in ln or "Mono:" in ln]
        print(f"   card {config.MIC_CARD}: " + (" | ".join(s.strip() for s in line) or "no data"))
        if "100%" in r.stdout:
            print("   NOTE: gain is at 100%. That is the maximum and it clips on")
            print("         speech (peak > 1.0), so the keyboard and the speaker")
            print("         get scored as speech. Consider 60-75%:")
            print(f"            amixer -c {config.MIC_CARD} sset Mic 70% cap")
    except Exception as e:  # noqa: BLE001
        print(f"   could not read gain: {e}")


def capture(secs: int) -> np.ndarray:
    import pyaudio

    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=CAPTURE_RATE,
        input=True,
        input_device_index=config.INPUT_DEVICE_INDEX,
        frames_per_buffer=CHUNK,
    )
    frames = []
    n_chunks = int(secs * CAPTURE_RATE / CHUNK)
    print(f"   recording {secs}s from device {config.INPUT_DEVICE_INDEX} at {CAPTURE_RATE} Hz...")
    for _ in range(n_chunks):
        frames.append(stream.read(CHUNK, exception_on_overflow=False))
    stream.stop_stream()
    stream.close()
    pa.terminate()

    raw = b"".join(frames)
    with wave.open(WAV_PATH, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(CAPTURE_RATE)
        w.writeframes(raw)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--secs", type=int, default=10, help="seconds to listen (default 10)")
    args = ap.parse_args()

    print("=" * 62)
    print("WanDog microphone diagnostic")
    print("=" * 62)

    report_gain()

    print("\n2. CAPTURE")
    audio = capture(args.secs)
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.abs(audio).max())
    print(f"   RMS {rms:.5f}   peak {peak:.5f}   ({len(audio)/CAPTURE_RATE:.1f}s saved to {WAV_PATH})")
    if rms < 0.002:
        print("   VERDICT: essentially silent. The mic is not picking up sound.")
        print("   Check: physical mic plug, the right device index, capture gain.")
        print("   This is a hardware/routing problem, not the agent's code.")
        return 1
    print("   OK: the mic is returning audio.")

    print("\n3. VOICE ACTIVITY (Silero, the agent's settings)")
    import asyncio

    import soxr
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams

    a16 = soxr.resample(audio, CAPTURE_RATE, 16000)
    a16_i = (np.clip(a16, -1, 1) * 32768).astype(np.int16)

    vad = SileroVADAnalyzer(
        sample_rate=16000,
        params=VADParams(
            confidence=config.VAD_CONFIDENCE,
            start_secs=config.VAD_START_SECS,
            stop_secs=config.VAD_STOP_SECS,
            min_volume=config.VAD_MIN_VOLUME,
        ),
    )
    vad.set_sample_rate(16000)
    n = vad.num_frames_required()

    async def scan() -> tuple[int, int, float]:
        states = []
        best = 0.0
        for i in range(0, len(a16_i) - n, n):
            buf = a16_i[i : i + n].tobytes()
            conf = float(np.asarray(vad.voice_confidence(buf)).ravel()[0])
            best = max(best, conf)
            states.append((await vad.analyze_audio(buf)).name)
        speech = sum(1 for s in states if s in ("STARTING", "SPEAKING"))
        return speech, len(states), best

    speech, total, best = asyncio.run(scan())
    print(f"   {speech}/{total} frames scored as speech   (peak confidence {best:.2f})")
    print(f"   thresholds: confidence>={config.VAD_CONFIDENCE} volume>={config.VAD_MIN_VOLUME}")
    if speech == 0:
        print("   VERDICT: VAD never fired. Either you were silent, or the")
        print("   thresholds are too strict. Try:")
        print("     WANDOG_VAD_CONFIDENCE=0.2 WANDOG_VAD_MIN_VOLUME=0.2 \\")
        print("       .venv/bin/python diag_mic.py")
        return 1
    print("   OK: VAD detected speech.")

    print("\n4. TRANSCRIPT (faster-whisper, the agent's model)")
    from faster_whisper import WhisperModel

    model = WhisperModel(config.ASR_MODEL, device="cpu", compute_type=config.ASR_COMPUTE)
    segments, _ = model.transcribe(a16, language=config.ASR_LANGUAGE, beam_size=1)
    text = " ".join(s.text for s in segments).strip()
    print(f"   heard: {text!r}")
    if not text:
        print("   VERDICT: VAD fired but Whisper heard no words.")
        print("   The audio is probably too faint or too muffled. Move closer,")
        print("   or raise the mic gain. (Whisper does not need more volume,")
        print("   but it does need a reasonably clean signal.)")
        return 1

    print("\n" + "=" * 62)
    print("PASS: mic -> VAD -> ASR all work. The voice input path is good.")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    sys.exit(main())
