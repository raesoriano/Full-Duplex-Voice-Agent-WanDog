"""Barge-in test for Fweddie (Pipecat 1.15.0).

Barge-in in 1.15 is driven by the LLMUserAggregator's internal
UserTurnController + VADUserTurnStartStrategy. When the user starts
speaking, the controller fires an "on_user_turn_started" event with
enable_interruptions=True, and the aggregator broadcasts an
InterruptionFrame that stops the bot's TTS.

This test verifies the two things that must be true for barge-in to work:

  1. WIRING: the real Fweddie pipeline (build_pipeline) contains an
     LLMUserAggregator configured with a VAD analyzer and a
     VADUserTurnStartStrategy. Without it, barge-in silently does nothing.

  2. TRIGGER: the exact turn-start mechanism the aggregator uses
     (UserTurnController + VADUserTurnStartStrategy) fires
     on_user_turn_started with enable_interruptions=True when it sees a
     VADUserStartedSpeakingFrame. This is the deterministic proof that a
     real user speaking into the mic will interrupt the bot.

  3. VAD: the real SileroVADAnalyzer reaches SPEAKING on audio, so a real
     user's voice will produce the VADUserStartedSpeakingFrame that drives
     the trigger above.

Run:  .venv/bin/python test_bargein.py
"""

from __future__ import annotations

import asyncio
import math
import struct
import sys

import config
from agent import build_pipeline
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import VADUserStartedSpeakingFrame
from pipecat.processors.aggregators.llm_response_universal import LLMUserAggregator
from pipecat.turns.user_start.vad_user_turn_start_strategy import VADUserTurnStartStrategy
from pipecat.turns.user_turn_controller import UserTurnController
from pipecat.turns.user_turn_strategies import UserTurnStrategies


def _tone_bytes(freq: float, seconds: float, sample_rate: int, amplitude: int = 12000) -> bytes:
    """A sine tone as 16-bit mono PCM (what the mic produces)."""
    n = int(seconds * sample_rate)
    out = bytearray()
    for i in range(n):
        v = int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
        out += struct.pack("<h", v)
    return bytes(out)


def test_wiring():
    """The real pipeline must contain a VAD-configured LLMUserAggregator."""
    print("\n=== TEST 1: pipeline wiring (LLMUserAggregator + VAD) ===")
    pipeline, *_ = build_pipeline(api_key="dummy", bridge=None, bus=None)
    user_agg = next((s for s in pipeline.processors if isinstance(s, LLMUserAggregator)), None)
    if user_agg is None:
        print("  FAIL: no LLMUserAggregator in pipeline -> barge-in NOT wired.")
        return False
    params = user_agg._params
    has_vad = params.vad_analyzer is not None
    start_strats = params.user_turn_strategies.start if params.user_turn_strategies else []
    has_vad_start = any(isinstance(s, VADUserTurnStartStrategy) for s in start_strats)
    print(f"  user aggregator present : True")
    print(f"  vad_analyzer configured : {has_vad}")
    print(f"  VAD turn-start strategy : {has_vad_start}")
    ok = has_vad and has_vad_start
    print(f"  {'PASS' if ok else 'FAIL'}: barge-in wiring is {'present' if ok else 'incomplete'}.")
    return ok


async def test_trigger():
    """UserTurnController + VADUserTurnStartStrategy fires the interruption
    event on a VADUserStartedSpeakingFrame (the exact mechanism the
    aggregator uses)."""
    print("\n=== TEST 2: turn-start trigger fires interruption ===")
    controller = UserTurnController(
        user_turn_strategies=UserTurnStrategies(start=[VADUserTurnStartStrategy()]),
    )
    fired = {}

    def _on_started(controller, strategy, params):
        fired.setdefault("params", params)

    controller.add_event_handler("on_user_turn_started", _on_started)
    from pipecat.clocks.system_clock import SystemClock
    from pipecat.processors.frame_processor import FrameProcessorSetup
    from pipecat.utils.asyncio.task_manager import TaskManager

    setup = FrameProcessorSetup(
        clock=SystemClock(),
        task_manager=TaskManager(),
        pipeline_worker=None,
        audio_in_sample_rate=config.ASR_SAMPLE_RATE,
        audio_out_sample_rate=config.OUTPUT_SAMPLE_RATE,
    )
    await controller.setup(setup)
    await controller.start()

    await controller.process_frame(VADUserStartedSpeakingFrame(start_secs=0.2))
    await asyncio.sleep(0.2)

    if "params" in fired:
        p = fired["params"]
        print(f"  on_user_turn_started fired: True")
        print(f"  enable_interruptions      : {p.enable_interruptions}")
        ok = p.enable_interruptions
        print(f"  {'PASS' if ok else 'FAIL'}: user speech triggers a bot interruption.")
        return ok
    print("  FAIL: on_user_turn_started never fired on VADUserStartedSpeakingFrame.")
    return False


def _speech_pcm() -> bytes:
    """Real English speech as 16-bit mono PCM at ASR_SAMPLE_RATE.

    Generated once with espeak-ng (from the Piper voice dir) into /tmp, then
    resampled to the mic rate. Falls back to a tone if generation fails.
    """
    import subprocess
    import wave

    piper = "/home/raesoriano/.piper"
    wav = "/tmp/fweddie_speech.wav"
    try:
        import shutil, os
        os.makedirs("/tmp/espeak", exist_ok=True)
        for f in ("espeak-ng", "libespeak-ng.so.1", "libespeak-ng.so.1.52.0.1"):
            shutil.copy(f"{piper}/{f}", f"/tmp/espeak/{f}")
        if not os.path.exists("/tmp/espeak/espeak-ng-data"):
            shutil.copytree(f"{piper}/espeak-ng-data", "/tmp/espeak/espeak-ng-data")
        os.chmod("/tmp/espeak/espeak-ng", 0o755)
        env = dict(os.environ, LD_LIBRARY_PATH="/tmp/espeak",
                   ESPEAK_DATA_PATH="/tmp/espeak/espeak-ng-data")
        subprocess.run(
            ["/tmp/espeak/espeak-ng", "-v", "en-us", "-s", "150", "-w", wav,
             "Hello Fweddie, this is a test of the voice activity detector"],
            check=True, env=env, capture_output=True,
        )
        with wave.open(wav) as w:
            rate, frames = w.getframerate(), w.readframes(w.getnframes())
        # Resample to ASR_SAMPLE_RATE (linear) if needed.
        if rate != config.ASR_SAMPLE_RATE:
            import array
            a = array.array("h")
            a.frombytes(frames)
            n = len(a)
            out = array.array("h")
            for i in range(int(n * config.ASR_SAMPLE_RATE / rate)):
                src = i * rate / config.ASR_SAMPLE_RATE
                j = int(src)
                frac = src - j
                v = a[j] * (1 - frac) + (a[j + 1] if j + 1 < n else a[j]) * frac
                out.append(int(v))
            frames = out.tobytes()
        return frames
    except Exception as e:  # noqa: BLE001
        print(f"  (espeak unavailable: {e}; using tone fallback)")
        return _tone_bytes(440.0, 2.0, config.ASR_SAMPLE_RATE)


async def test_vad():
    """The real SileroVADAnalyzer reaches SPEAKING on real speech."""
    print("\n=== TEST 3: VAD detects speech (real analyzer + real speech) ===")
    analyzer = SileroVADAnalyzer(
        sample_rate=config.ASR_SAMPLE_RATE,
        params=VADParams(
            confidence=config.VAD_CONFIDENCE,
            start_secs=config.VAD_START_SECS,
            stop_secs=config.VAD_STOP_SECS,
            min_volume=config.VAD_MIN_VOLUME,
        ),
    )
    analyzer.set_sample_rate(config.ASR_SAMPLE_RATE)  # initialize internals
    pcm = _speech_pcm()
    chunk_size = 2 * config.ASR_SAMPLE_RATE // 50  # 20 ms, 16-bit mono
    state = VADState.QUIET
    for off in range(0, len(pcm) - chunk_size, chunk_size):
        state = await analyzer.analyze_audio(pcm[off:off + chunk_size])
        if state == VADState.SPEAKING:
            break
    if state == VADState.SPEAKING:
        print(f"  PASS: VAD reached SPEAKING on real speech (state={state.name}).")
        return True
    print(f"  FAIL: VAD never reached SPEAKING (final state={state.name}).")
    return False


async def main():
    print(f"ASR_ENABLED={config.ASR_ENABLED}  LLM_MODEL={config.LLM_MODEL}")
    ok1 = test_wiring()
    ok2 = await test_trigger()
    ok3 = await test_vad()
    print("\n=== SUMMARY ===")
    print(f"  pipeline wiring (user agg + VAD) : {'PASS' if ok1 else 'FAIL'}")
    print(f"  turn-start trigger -> interrupt  : {'PASS' if ok2 else 'FAIL'}")
    print(f"  VAD detects speech               : {'PASS' if ok3 else 'FAIL'}")
    return 0 if (ok1 and ok2 and ok3) else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
