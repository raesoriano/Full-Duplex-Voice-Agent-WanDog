"""End-to-end: does the REAL agent.py pipeline actually ANSWER a question?

test_pipeline_asr.py proves the turn reaches the LLM. It stops there. This
test goes one step further and proves the bot's reply comes back out, which is
the only thing that matters when the user says "it is not answering".

It drives the production ``agent.build_pipeline()`` -- same stages, same
aggregators, same LLM, same turn strategies -- with a wav file injected in
place of the microphone, and fails unless a ``bot_text`` event appears.

Run:  .venv/bin/python test_endtoend_answer.py [/tmp/q_math.wav]
"""

from __future__ import annotations

import asyncio
import sys
import time
import wave

import numpy as np

import config
import agent
from pipecat.frames.frames import EndFrame, InputAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor

CAPTURE_RATE = config.INPUT_SAMPLE_RATE
CHUNK = 960              # 20 ms at 48 kHz
TAIL_SECS = 6.0          # enough silence for the turn to close AND be answered
ANSWER_BUDGET = 20.0     # speech end -> bot reply. Generous: the LLM is remote.


def load48k(path: str) -> np.ndarray:
    w = wave.open(path)
    sr, ch = w.getframerate(), w.getnchannels()
    raw = w.readframes(w.getnframes())
    a = np.frombuffer(raw, dtype=np.int16)
    if ch == 2:
        a = a.reshape(-1, 2).mean(axis=1).astype(np.int16)
    if sr != CAPTURE_RATE:
        n = int(len(a) * CAPTURE_RATE / sr)
        a = np.interp(
            np.linspace(0, len(a) - 1, n), np.arange(len(a)), a.astype(np.float32)
        ).astype(np.int16)
    return a


class FakeTransport(FrameProcessor):
    """Stands in for the mic/speaker pair.

    Must be a real FrameProcessor: ``Pipeline.__init__`` calls ``link()`` on
    every stage, so a plain object fails at construction. It forwards every
    frame it is given (notably StartFrame) and never produces audio.
    """

    def __init__(self):
        super().__init__()
        self.started = asyncio.Event()

    async def process_frame(self, frame, direction):
        from pipecat.frames.frames import StartFrame

        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.started.set()
        await self.push_frame(frame, direction)


class Recorder:
    """Collects the event-bus traffic agent.py already publishes."""

    def __init__(self):
        self.events: list[tuple[float, str, str]] = []
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0

    def hook(self, bus):
        original = bus.emit

        async def emit(kind, **kwargs):
            text = kwargs.get("text", "")
            self.events.append((self.now(), kind, text))
            return await original(kind, **kwargs)

        bus.emit = emit


async def main() -> int:
    paths = sys.argv[1:] or ["/tmp/q_math.wav"]
    clips = [load48k(p) for p in paths]
    for p, c in zip(paths, clips):
        print(f"clip: {p}  {len(c) / CAPTURE_RATE:.2f}s")

    rec = Recorder()
    import eventbus

    rec.hook(eventbus.BUS)

    # A real LLM key is required; skip cleanly rather than fail if absent.
    try:
        api_key = config.resolve_llm_api_key()
    except Exception as e:  # noqa: BLE001
        print(f"SKIP: no LLM key ({e})")
        return 2

    pipeline, task, _in, _out = agent.build_pipeline(
        api_key, bridge=None, bus=eventbus.BUS, transports=(FakeTransport(), FakeTransport())
    )
    runner = agent.WorkerRunner()
    await runner.add_workers(task)

    pad = lambda secs: np.zeros(int(CAPTURE_RATE * secs), dtype=np.int16)
    # Separate questions by a pause long enough that each closes its own turn.
    pieces = [pad(0.5)]
    for c in clips:
        pieces += [c, pad(TAIL_SECS)]
    audio = np.concatenate(pieces)

    async def feed():
        await asyncio.sleep(8.0)  # let ASR models load (sherpa + whisper)
        for i in range(0, len(audio), CHUNK):
            await task.queue_frame(
                InputAudioRawFrame(
                    audio=audio[i : i + CHUNK].tobytes(),
                    sample_rate=CAPTURE_RATE,
                    num_channels=1,
                )
            )
            await asyncio.sleep(0.02)
        await asyncio.sleep(TAIL_SECS)
        await task.queue_frame(EndFrame())

    async def guard():
        await asyncio.sleep(120)
        await task.queue_frame(EndFrame())

    try:
        await asyncio.wait_for(
            asyncio.gather(runner.run(auto_end=False), feed(), guard()), timeout=140
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    finally:
        await runner.cancel()

    print("\n--- event bus ---")
    for t, kind, text in rec.events:
        print(f"  {t:6.2f}s  {kind:12s} {text!r}")

    user = [e for e in rec.events if e[1] == "user_text"]
    bot = [e for e in rec.events if e[1] == "bot_text"]
    stops = [e for e in rec.events if e[1] == "user_stopped"]

    ok = True
    if not user:
        print("\nFAIL: no user_text -- the turn never reached the LLM")
        ok = False
    else:
        print(f"\nPASS: user_text {user[0][2]!r}")

    if not bot:
        print("FAIL: no bot_text -- THE AGENT DID NOT ANSWER")
        ok = False
    else:
        print(f"PASS: bot_text {bot[0][2]!r}")
        if stops:
            lat = bot[0][0] - stops[0][0]
            verdict = "PASS" if lat <= ANSWER_BUDGET else "FAIL"
            print(f"{verdict}: answered {lat:.2f}s after speech ended "
                  f"(budget {ANSWER_BUDGET}s)")
            ok &= lat <= ANSWER_BUDGET

    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
