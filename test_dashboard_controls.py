"""Prove the dashboard's Send and Stop buttons reach the pipeline.

Both are file drops that a watcher stage turns into frames, so there are two
things that can silently break and neither is visible from the dashboard:

  1. the frame is queued at a stage that sits AFTER the processor which handles
     it, so it flows away from its consumer and nothing happens (this is exactly
     the bug this test was written to catch: the watcher was first placed after
     the user aggregator and Send did nothing at all);
  2. the typed message is added to the context but inference is never started,
     so the question is remembered and never answered.

The LLM is replaced by a recorder, so this runs in well under a second and says
nothing about a model. What it measures is whether the question arrives.

Run:  .venv/bin/python test_dashboard_controls.py
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import time
from pathlib import Path

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMTextFrame,
    StartFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMUserAggregator,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.workers.runner import WorkerRunner

import agent
import config


class RecordingLLM(FrameProcessor):
    """Stands in for the LLM: records each context it is asked to answer."""

    def __init__(self):
        super().__init__()
        self.contexts: list[LLMContext] = []
        self.interruptions = 0
        # The shared context, set by build(). Clear is observed here rather than
        # as a frame: LLMMessagesUpdateFrame is CONSUMED by the user aggregator
        # upstream, so the LLM never sees it. What matters is the state it
        # leaves behind.
        self.context: LLMContext | None = None
        self.ready = asyncio.Event()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            self.ready.set()
        elif isinstance(frame, LLMContextFrame):
            self.contexts.append(frame.context)
            # Answer, so the assistant aggregator has something to finish and
            # the pipeline exercises the same path a real turn would.
            await self.push_frame(LLMTextFrame(text="ok"))
        elif isinstance(frame, InterruptionFrame):
            self.interruptions += 1
            await self.push_frame(frame, direction)
            return
        await self.push_frame(frame, direction)

    def last_user_text(self) -> str:
        if not self.contexts:
            return ""
        msgs = self.contexts[-1].get_messages()
        for m in reversed(msgs):
            if m.get("role") == "user":
                return m.get("content", "")
        return ""


async def build(typed: Path, stop: Path, clear: Path | None = None):
    """A minimal pipeline: watcher -> user agg -> recorder.

    The same ORDER as agent.py: the watcher is upstream of the aggregator. That
    ordering is the thing under test.
    """
    context = LLMContext()
    user_agg = LLMUserAggregator(context, params=LLMUserAggregatorParams())
    llm = RecordingLLM()
    llm.context = context
    watcher = agent.DashboardControlWatcher(
        str(typed), str(stop), bus=None, poll_secs=0.02,
        clear_path=str(clear) if clear else None,
    )
    pipeline = Pipeline([watcher, user_agg, llm])
    task = PipelineTask(pipeline)
    return pipeline, task, llm


async def drive(typed: Path, stop: Path, clear: Path | None = None):
    pipeline, task, llm = await build(typed, stop, clear)
    # WorkerRunner, not task.run(): PipelineTask.run() is the worker's internal
    # entry point and needs a params argument. The runner is what agent.py uses.
    runner = WorkerRunner()
    await runner.add_workers(task)
    worker = asyncio.create_task(runner.run(auto_end=False))
    try:
        await asyncio.wait_for(llm.ready.wait(), timeout=5)
        # Send a typed question.
        typed.write_text(json.dumps({"id": 1, "text": "what is on my calendar"}), encoding="utf-8")
        for _ in range(200):  # up to ~2 s
            if llm.last_user_text():
                break
            await asyncio.sleep(0.01)
        text = llm.last_user_text()

        # The file must be consumed, not merely read: a second identical write
        # has to be answered, but the SAME file must not be read twice.
        consumed = not typed.exists()

        # Stop button.
        stop.write_text(json.dumps({"id": 1, "ts": time.time()}), encoding="utf-8")
        for _ in range(200):
            if llm.interruptions:
                break
            await asyncio.sleep(0.01)
        interruptions = llm.interruptions

        # Clear: the conversation must be forgotten, and no answer started.
        history_before = len(llm.context.get_messages()) if llm.context else 0
        contexts_before = len(llm.contexts)
        if clear is not None:
            clear.write_text(json.dumps({"id": 1, "ts": time.time()}), encoding="utf-8")
            for _ in range(200):
                if llm.context and not llm.context.get_messages():
                    break
                await asyncio.sleep(0.01)
            # Give a wrongly-started turn a chance to show up as a new context.
            await asyncio.sleep(0.15)
        history_after = len(llm.context.get_messages()) if llm.context else 0
        contexts_after = len(llm.contexts)
    finally:
        await runner.cancel()
        worker.cancel()
        try:
            await worker
        except (asyncio.CancelledError, Exception):
            pass
    return {
        "llm": llm,
        "text": text,
        "consumed": consumed,
        "interruptions": interruptions,
        "history_before": history_before,
        "history_after": history_after,
        "new_contexts": contexts_after - contexts_before,
    }


async def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    typed = tmp / "typed.json"
    stop = tmp / "stop.json"

    clear = tmp / "clear.json"
    r = await drive(typed, stop, clear)
    text, consumed, interruptions = r["text"], r["consumed"], r["interruptions"]

    failures = []
    if text != "what is on my calendar":
        failures.append(f"typed text never reached the LLM context (got {text!r})")
    if not consumed:
        failures.append("typed file was not deleted, so it would be answered again")
    if interruptions != 1:
        failures.append(f"stop did not interrupt exactly once (got {interruptions})")

    print(f"1. typed text reached the LLM as a user turn : {text!r}")
    print(f"2. control file consumed after acting       : {consumed}")
    print(f"3. stop broadcast an interruption            : {interruptions}")

    # 4. The real pipeline must place the watcher UPSTREAM of the aggregator.
    #    The isolated test above proves the frame reaches an aggregator placed
    #    after the watcher; this proves agent.py actually builds it that way.
    #    It is a static check on the stage list -- no microphone, no model.
    import agent

    stages = agent.pipeline_stage_names()
    if "DashboardControlWatcher" not in stages:
        failures.append(f"the real pipeline has no DashboardControlWatcher: {stages}")
    elif "LLMUserAggregator" not in stages:
        failures.append(f"the real pipeline has no LLMUserAggregator: {stages}")
    elif stages.index("DashboardControlWatcher") > stages.index("LLMUserAggregator"):
        failures.append(
            "the watcher sits AFTER the aggregator in the real pipeline, so a "
            f"typed message would flow away from it: {stages}"
        )
    print(f"4. real pipeline order                       : {stages}")

    # 5. Clear empties the shared context and starts no inference.
    if r["history_before"] < 1:
        failures.append(
            "nothing was in the context before Clear, so clearing it proves nothing"
        )
    if r["history_after"] != 0:
        failures.append(
            f"Clear left {r['history_after']} message(s) in the context"
        )
    if r["new_contexts"] != 0:
        failures.append(
            f"Clear started {r['new_contexts']} new LLM turn(s); clearing is not a question"
        )
    print(
        f"5. Clear emptied the context                : "
        f"{r['history_before']} -> {r['history_after']} messages, "
        f"{r['new_contexts']} new turn(s)"
    )

    if failures:
        for f in failures:
            print("FAIL:", f)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
