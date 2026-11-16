"""Turn-context test for Fweddie: does the LLM see the CURRENT question?

The report: the agent answers the PREVIOUS question, not the current one.

The hypothesis: build_pipeline() gives the user aggregator and the assistant
aggregator two DIFFERENT LLMContext objects.

    user_agg  = LLMUserAggregator(LLMContext(messages=[system]))
    asst_agg  = LLMAssistantAggregator(LLMContext(messages=[system]))

Only the user half's context is ever sent to the LLM (the user aggregator is
what pushes LLMContextFrame). The assistant half writes its replies into its
OWN context, which nothing reads. So the LLM receives a conversation of:

    [system, user("Q1"), user("Q2"), user("Q3"), ...]

with no assistant turns at all -- the model never sees its own answers, and
every earlier question stays in the prompt as an unanswered request.

Test 2 uses the aggregators' real context-writing code (no frames pushed, so
no TaskManager needed) to show exactly which messages reach the model.

Run:  .venv/bin/python test_turn_context.py
"""

from __future__ import annotations

import asyncio

from pipecat.processors.aggregators.llm_response_universal import (
    LLMAssistantAggregator,
    LLMUserAggregator,
)
from pipecat.utils.string import TextPartForConcatenation

import agent


def test_shared_context():
    """The real pipeline's two aggregators must share one LLMContext."""
    print("\n=== TEST 1: user and assistant aggregators share one context ===")
    pipeline, *_ = agent.build_pipeline(api_key="dummy", bridge=None, bus=None)

    user_agg = next((s for s in pipeline.processors if isinstance(s, LLMUserAggregator)), None)
    asst_agg = next((s for s in pipeline.processors if isinstance(s, LLMAssistantAggregator)), None)

    if user_agg is None or asst_agg is None:
        print("  FAIL: an aggregator is missing from the pipeline.")
        return False

    same = user_agg._context is asst_agg._context
    print(f"  user_agg context  : {id(user_agg._context)}")
    print(f"  asst_agg context  : {id(asst_agg._context)}")
    print(f"  same object       : {same}")
    print(f"  {'PASS' if same else 'FAIL'}: context is "
          f"{'shared' if same else 'SPLIT (replies never reach the LLM)'}.")
    return same


async def test_turn_sequence():
    """Write a user turn and a bot turn, then show the prompt the LLM gets.

    ``_push_aggregation`` is the user aggregator's real path: it appends the
    aggregated user text to its context and then asks for inference. We stub
    only the inference trigger so no frames are pushed. ``_add_aggregation_to_context``
    is the assistant aggregator's real path for committing its reply.
    """
    print("\n=== TEST 2: what the LLM actually receives on each turn ===")
    pipeline, *_ = agent.build_pipeline(api_key="dummy", bridge=None, bus=None)

    user_agg = next(s for s in pipeline.processors if isinstance(s, LLMUserAggregator))
    asst_agg = next(s for s in pipeline.processors if isinstance(s, LLMAssistantAggregator))

    captured: list[list[dict]] = []

    # The user half's context IS the prompt: it is what LLMContextFrame carries.
    async def fake_run_llm():
        captured.append(list(user_agg._context.messages))

    user_agg._run_llm = fake_run_llm  # type: ignore[method-assign]

    async def turn(question: str, answer: str):
        # --- user speaks: real aggregation path ---
        user_agg._aggregation = [TextPartForConcatenation(question, includes_inter_part_spaces=False)]
        await user_agg._push_aggregation()

        # --- bot answers: real context-write path ---
        asst_agg._aggregation = [TextPartForConcatenation(answer, includes_inter_part_spaces=False)]
        await asst_agg._add_aggregation_to_context()

    await turn("What is 2 plus 2?", "Two plus two is four.")
    await turn("What is a dog?", "A dog is a domesticated mammal.")

    for i, msgs in enumerate(captured, 1):
        print(f"\n  --- prompt sent to the LLM on inference {i} "
              f"({len(msgs)} messages) ---")
        for m in msgs:
            role = m.get("role", "?")
            content = str(m.get("content", ""))[:64]
            print(f"    {role:10s} | {content}")

    if len(captured) < 2:
        print(f"\n  FAIL: expected 2 inferences, captured {len(captured)}.")
        return False

    second = captured[1]
    roles = [m.get("role") for m in second]
    has_assistant = "assistant" in roles
    n_user = sum(1 for r in roles if r == "user")

    print(f"\n  roles on 2nd inference : {roles}")
    print(f"  assistant reply present: {has_assistant}")
    print(f"  user messages stacked  : {n_user}")

    ok = has_assistant
    print(f"  {'PASS' if ok else 'FAIL'}: "
          f"{'the model sees its own replies' if ok else 'the model never sees its replies -- earlier questions pile up unanswered'}")
    return ok


async def main():
    results = [test_shared_context(), await test_turn_sequence()]
    print("\n" + "=" * 62)
    if all(results):
        print("ALL PASSED")
        return 0
    print("FAILED -- turn history is not reaching the LLM.")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
