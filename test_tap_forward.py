"""Does PartialTranscriptTap forward a final TranscriptionFrame? No audio.

The stage-isolation run showed the turn disappearing as soon as
``PartialTranscriptTap`` was added to an otherwise working pipeline. This
checks the tap in isolation: push one frame through [source, tap, sink] and
assert the sink saw it.

Run:  .venv/bin/python test_tap_forward.py
"""

from __future__ import annotations

import asyncio
import sys

from pipecat.frames.frames import (
    EndFrame,
    InterimTranscriptionFrame,
    StartFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

import agent


class Source(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.ready = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.ready.set()
        await self.push_frame(frame, direction)


class Sink(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.seen: list[str] = []
        self.ready = asyncio.Event()

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.ready.set()
        elif isinstance(frame, TranscriptionFrame):
            self.seen.append(f"final:{frame.text}")
        elif isinstance(frame, InterimTranscriptionFrame):
            self.seen.append(f"interim:{frame.text}")
        elif isinstance(frame, TextFrame):
            self.seen.append(f"text:{frame.text}")
        await self.push_frame(frame, direction)


async def run_one(with_tap: bool) -> list[str]:
    src, sink = Source(), Sink()
    stages = [src]
    if with_tap:
        stages.append(agent.PartialTranscriptTap(None))
    stages.append(sink)

    task = PipelineTask(Pipeline(stages))
    runner = WorkerRunner()
    await runner.add_workers(task)

    async def feed():
        await asyncio.wait_for(sink.ready.wait(), timeout=30)
        await asyncio.sleep(0.2)
        await task.queue_frame(
            InterimTranscriptionFrame(text="what is", user_id="u", timestamp="t")
        )
        await asyncio.sleep(0.1)
        await task.queue_frame(
            TranscriptionFrame(text="What is 2 plus 2?", user_id="u", timestamp="t", finalized=True)
        )
        await asyncio.sleep(0.5)
        await task.queue_frame(EndFrame())

    async def guard():
        await asyncio.sleep(20)
        await task.queue_frame(EndFrame())

    try:
        await asyncio.wait_for(
            asyncio.gather(runner.run(), feed(), guard()), timeout=30
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    return sink.seen


async def main() -> int:
    ok = True
    for with_tap in (False, True):
        label = "with tap" if with_tap else "no tap (control)"
        seen = await run_one(with_tap)
        finals = [s for s in seen if s.startswith("final:")]
        print(f"{label:20s} sink saw: {seen}")
        if not finals:
            print(f"{'':20s} >>> FINAL DROPPED")
            ok = False
        else:
            print(f"{'':20s} final forwarded: {finals[0]}")

    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
