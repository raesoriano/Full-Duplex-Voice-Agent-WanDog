"""A turn that ends without words must end, and a mute must actually mute.

Both come from one report: "when the message is just typed while muted, the
agent does not answer and the turn is not ending sometimes".

Two independent faults are covered here.

1. ``dashboard.turn_in_flight`` -- the Send/Stop swap. A voice turn opens on
   ``user_started`` (the VAD fired) and only produces a reply if a transcript
   follows. Muted, or a cough, or a door slamming gives ``user_started`` ->
   ``user_stopped`` with NO ``user_text``. ``user_stopped`` did not clear
   ``busy``, so the button stayed on Stop for a turn that was already over and
   would never be answered -- measured on the real log, the final
   ``user_stopped`` left it returning True until the 90 s staleness guard.

2. ``agent.MicControlWatcher`` -- a mic file that already said "muted" when the
   agent started was read, recorded, and never applied, because the first read
   pushed nothing. Measured: started against {"enabled": false}, the STT still
   transcribed. So "muted" could be a lie, and speech the person believed was
   ignored still became a turn.

No microphone, no model, no network.

Run:  .venv/bin/python test_muted_turn.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path

import dashboard
import agent
from pipecat.frames.frames import STTMuteFrame, StartFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.workers.runner import WorkerRunner

T0 = 1_800_000_000.0


def ev(kind: str, offset: float, **kw) -> dict:
    """One event, `offset` seconds after T0."""
    return {"kind": kind, "ts": T0 + offset, **kw}


def at(ts: float):
    """Freeze dashboard.time.time() so the staleness guard is deterministic."""
    dashboard.time.time = lambda: ts


# ---------------------------------------------------------------------------
# 1. turn_in_flight
# ---------------------------------------------------------------------------

def check_turn_state() -> list[str]:
    failures: list[str] = []

    def expect(label, events, now, want):
        at(now)
        got = dashboard.turn_in_flight(events)
        print(f"   {label:<46} {'STOP' if got else 'send':<5} "
              f"{'ok' if got == want else 'WRONG'}")
        if got != want:
            failures.append(f"{label}: expected {'STOP' if want else 'send'}, "
                            f"got {'STOP' if got else 'send'}")

    print("1. turn_in_flight")

    # THE BUG. A phantom turn: VAD opened and closed, no transcript ever came.
    phantom = [ev("user_started", 0), ev("user_stopped", 5)]
    expect("phantom turn (user_stopped, no transcript)", phantom, T0 + 6, False)

    # A real turn: the transcript arrived before user_stopped.
    answered = [ev("user_started", 0), ev("user_partial", 2, text="What"),
                ev("user_text", 4, text="What is a cat?"), ev("user_stopped", 4.1)]
    expect("transcript arrived before user_stopped", answered, T0 + 5, True)

    # The reply is in flight.
    expect("bot_started (answering)",
           answered + [ev("bot_started", 6)], T0 + 7, True)
    expect("bot_stopped (turn over)",
           answered + [ev("bot_started", 6), ev("bot_stopped", 9)], T0 + 10, False)

    # While the person is talking the button is Stop: a barge-in is stoppable.
    expect("user started talking", [ev("user_started", 0)], T0 + 1, True)

    # ...and a long sentence must not time out mid-sentence. Partials are the
    # proof of life, so they reset the staleness clock.
    long_speech = [ev("user_started", 0), ev("user_partial", 10, text="a"),
                   ev("user_partial", 100, text="a b")]
    expect("still talking, last partial 5 s ago", long_speech, T0 + 105, True)

    # The dead-agent guard still works: nothing at all for over 90 s.
    expect("nothing for 120 s (agent likely dead)", long_speech, T0 + 220, False)

    # An error ends the turn, or the button waits out the guard with nothing to
    # stop -- the "gave up but still says Stop" case.
    expect("error ends the turn",
           answered + [ev("bot_started", 6), ev("error", 7, text="boom")],
           T0 + 8, False)

    # A typed message while idle must still start a turn.
    expect("typed message, idle", [ev("user_text", 0, source="typed")], T0 + 1, True)

    # Pressing Stop must put Send back immediately, without waiting for the
    # agent to acknowledge.
    expect("interrupted",
           answered + [ev("interrupted", 6, source="dashboard")], T0 + 7, False)

    return failures


# ---------------------------------------------------------------------------
# 2. the mute must be applied on the first read
# ---------------------------------------------------------------------------

class RecordingSTT(FrameProcessor):
    """Stands in for the STT service: records the mute frames it receives."""

    def __init__(self):
        super().__init__()
        self.mutes: list[bool] = []
        self.ready = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.ready.set()
        elif isinstance(frame, STTMuteFrame):
            self.mutes.append(frame.mute)
        await self.push_frame(frame, direction)


class Sink(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


async def mute_frames_on_start(start_muted: bool) -> list[bool]:
    tmp = Path(tempfile.mkdtemp())
    mic = tmp / "mic.json"
    mic.write_text(json.dumps({"enabled": not start_muted}), encoding="utf-8")

    stt, sink = RecordingSTT(), Sink()
    watcher = agent.MicControlWatcher(str(mic), poll_secs=0.05)
    task = PipelineTask(Pipeline([stt, watcher, sink]))
    runner = WorkerRunner()
    await runner.add_workers(task)
    running = asyncio.create_task(runner.run(auto_end=False))
    try:
        await asyncio.wait_for(stt.ready.wait(), timeout=5)
        await asyncio.sleep(0.8)
        return list(stt.mutes)
    finally:
        await runner.cancel()
        running.cancel()
        try:
            await running
        except (asyncio.CancelledError, Exception):
            pass


async def check_mute() -> list[str]:
    failures: list[str] = []
    print("2. mute is applied on the first read")

    muted = await mute_frames_on_start(start_muted=True)
    ok = muted == [True]
    print(f"   started MUTED  -> STT mute frames {muted!r:<10} "
          f"{'ok' if ok else 'WRONG (mute never applied)'}")
    if not ok:
        failures.append(
            "the agent started while the dashboard was muted and never sent "
            f"STTMuteFrame(mute=True); got {muted!r}. The mic is live while the "
            "UI says it is muted."
        )

    live = await mute_frames_on_start(start_muted=False)
    ok = live == [False]
    print(f"   started LIVE   -> STT mute frames {live!r:<10} "
          f"{'ok' if ok else 'WRONG'}")
    if not ok:
        failures.append(
            f"the agent started unmuted but sent {live!r} instead of [False]"
        )
    return failures


# ---------------------------------------------------------------------------
# 3. an error reaches the dashboard
# ---------------------------------------------------------------------------

async def check_error_event() -> list[str]:
    """An ErrorFrame must be published, not only logged.

    Otherwise a failed turn looks exactly like a slow one: the question sits
    there, no reply comes, and nothing says why.
    """
    from pipecat.frames.frames import EndFrame, ErrorFrame

    published: list[tuple[str, dict]] = []

    class Bus:
        async def emit(self, kind, **kw):
            published.append((kind, kw))

    logger = agent.WanDogLogger(Bus())
    await logger.process_frame(StartFrame(), agent.FrameDirection.DOWNSTREAM)
    await logger.process_frame(
        ErrorFrame(error="LLM completion timeout"), agent.FrameDirection.DOWNSTREAM
    )
    await logger.process_frame(EndFrame(), agent.FrameDirection.DOWNSTREAM)

    errs = [kw for kind, kw in published if kind == "error"]
    print("3. an error reaches the dashboard")
    ok = len(errs) == 1 and "timeout" in errs[0].get("text", "")
    shown = errs[0].get("text", "") if errs else "NONE"
    print(f"   error event published                      "
          f"{shown!r:<5} {'ok' if ok else 'WRONG'}")
    return [] if ok else [
        f"an ErrorFrame produced {len(errs)} error event(s); the dashboard cannot "
        "show a failure it is never told about"
    ]


async def main() -> int:
    failures = check_turn_state()
    failures += await check_mute()
    failures += await check_error_event()

    if failures:
        print()
        for f in failures:
            print("FAIL:", f)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
