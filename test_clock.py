"""The clock: can the agent still tell you what time it is?

The report: "it can no longer tell the time right now." It used to. The event
log shows the agent running `date` and answering "The current time is 2:52 PM
on Tuesday, November 17, 2026" at 14:52, then at 16:13 answering "It is
currently Tuesday, November 17, 2026" and, pressed, "I am sorry, but I just do
not have a tool to check the current clock."

Two defects, both introduced by 0850bb2a ("Make search results dated and
readable by the model"):

1. A DATE WITH NO TIME. That commit put "Today is Tuesday, 17 November 2026"
   in front of the model so it could judge what "latest" means -- which it
   needed for news. But a model handed a date and asked the time will invent
   one rather than call a tool. Measured on the live model with the real
   toolset, 3 trials per question:

       prompt                      "What is the time?"   bash calls
       no date line (pre-0850bb2a)  ran `date`            12/12
       date line only (0850bb2a)    invented 11:44 AM     9/12
                                    invented 1:15 PM
                                    invented 11:38 AM
       date + clock + guidance      answered correctly    12/12

   The invented times are the tell: three consecutive calls, three different
   times, no tool call. Fixed by config.clock_line(), which states the time as
   well as the date, plus a line in TOOLS_SYSTEM_PROMPT that forbids stating a
   clock time that was not given or read.

2. THE PROMPT WAS SENT TWICE. build_pipeline put the prompt in the LLM
   service's `system_instruction` AND in the context's first message. The
   OpenAI adapter prepends the settings copy and keeps the context copy, so
   every request carried two identical ~1400-char system messages. Pipecat
   warns about this and calls the context copy deprecated. It also made the
   clock unfixable: the stale copy in the context would have shadowed any
   refresh of the settings one. Fixed by leaving the context empty.

Plus the freeze: a prompt composed once at startup tells the model the time
the agent launched. This agent runs for hours. Fixed by ClockedLLMService,
which re-reads the clock on every inference.

Run: .venv/bin/python test_clock.py
"""
from __future__ import annotations

import asyncio
import re

import config
import agent

CLOCK_RE = re.compile(
    r"The current date and time here is .*?\([^()]*UTC[+-]\d{4}\)\. "
)


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return ok


def make_llm(system_instruction: str | None = None) -> agent.ClockedLLMService:
    """A ClockedLLMService with no network use -- construction only."""
    prompt = system_instruction if system_instruction is not None else (
        config.SYSTEM_PROMPT + " " + config.tools_system_prompt()
    )
    settings = dict(
        model=config.LLM_MODEL,
        system_instruction=prompt,
        temperature=config.LLM_TEMPERATURE,
        max_tokens=config.LLM_MAX_TOKENS,
    )
    if not config.LLM_THINKING:
        settings["extra"] = {
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        }
    return agent.ClockedLLMService(
        api_key="not-a-real-key",
        base_url=config.LLM_BASE_URL,
        settings=agent.OpenAILLMService.Settings(**settings),
    )


# ---------------------------------------------------------------------------
# 1. The prompt states a time, not just a date
# ---------------------------------------------------------------------------
def test_prompt_has_a_clock() -> bool:
    print("\n1. the prompt states the time, not only the date")
    prompt = config.tools_system_prompt()
    ok = check("prompt carries a clock line", bool(CLOCK_RE.search(prompt)),
               CLOCK_RE.search(prompt).group(0) if CLOCK_RE.search(prompt) else "none")
    ok &= check("the clock has an hour and minute",
                bool(re.search(r"\d{1,2}:\d{2}\s*[AP]M", prompt)))
    ok &= check("the clock names a timezone", "UTC" in prompt)
    ok &= check("the date is still there for the news logic",
                bool(re.search(r"\d{2} \w+ \d{4}", prompt)))
    ok &= check("TOOLS_SYSTEM_PROMPT itself stays clock-free",
                "The current date and time here is" not in config.TOOLS_SYSTEM_PROMPT,
                "the value is injected at call time")
    return ok


# ---------------------------------------------------------------------------
# 2. The prompt forbids inventing a time
# ---------------------------------------------------------------------------
def test_prompt_forbids_inventing() -> bool:
    print("\n2. the prompt forbids a clock time it was not given")
    prompt = config.TOOLS_SYSTEM_PROMPT
    ok = check("it forbids stating an unread clock time",
               "Never state a clock time" in prompt)
    ok &= check("it names the escape hatch for other timezones",
                "TZ=" in prompt and "bash" in prompt)
    ok &= check("it still tells the model to use news for recency",
                "type set to news" in prompt)
    return ok


# ---------------------------------------------------------------------------
# 3. The clock is re-read, so a long-running agent cannot freeze
# ---------------------------------------------------------------------------
def test_clock_refreshes() -> bool:
    print("\n3. the clock is re-read on every inference")
    llm = make_llm()
    first = llm._settings.system_instruction
    found = CLOCK_RE.search(first)
    ok = check("the prompt starts with a clock line", bool(found))

    # Pretend the process has been up a while: plant an obviously stale clock.
    stale = CLOCK_RE.sub(
        lambda _m: "The current date and time here is Tuesday, 17 November 2026, "
                   "03:07 AM (PST, UTC+0800). ", first, count=1)
    llm._settings.system_instruction = stale
    ok &= check("a stale clock is present before the refresh", "03:07 AM" in stale)

    llm._refresh_clock()
    refreshed = llm._settings.system_instruction
    ok &= check("the stale clock is gone", "03:07 AM" not in refreshed)
    ok &= check("the fresh clock matches the host",
                CLOCK_RE.search(refreshed).group(0) == config.clock_line(),
                config.clock_line()[:60])
    return ok


# ---------------------------------------------------------------------------
# 4. Refreshing must not grow or duplicate the prompt
# ---------------------------------------------------------------------------
def test_refresh_is_idempotent() -> bool:
    print("\n4. refreshing does not duplicate or grow the prompt")
    llm = make_llm()
    original = llm._settings.system_instruction
    length = len(original)
    for _ in range(25):
        llm._refresh_clock()
    after = llm._settings.system_instruction
    ok = check("exactly one clock line after 25 refreshes",
               len(CLOCK_RE.findall(after)) == 1,
               f"{len(CLOCK_RE.findall(after))} found")
    ok &= check("the prompt did not grow", len(after) == length,
                f"{length} -> {len(after)}")
    ok &= check("the persona is still first", after.startswith(config.SYSTEM_PROMPT))
    ok &= check("the news guidance survived", "type set to news" in after)
    return ok


# ---------------------------------------------------------------------------
# 5. The clock sits in the tools section, not in front of the persona
# ---------------------------------------------------------------------------
def test_clock_position() -> bool:
    print("\n5. the clock is placed after the persona, not before it")
    # tools_system_prompt() is clock + tools text; it never carried the persona.
    prompt = config.tools_system_prompt()
    clock_at = prompt.find("The current date and time here is")
    tools_at = prompt.find(config.TOOLS_SYSTEM_PROMPT[:40])
    ok = check("the clock comes before the tools text", clock_at < tools_at,
               f"clock at {clock_at}, tools at {tools_at}")

    # The real pipeline prompt is persona, then clock, then tools.
    from test_endtoend_answer import FakeTransport

    pipeline, *_ = agent.build_pipeline(
        api_key="not-a-real-key", bridge=None, bus=None,
        transports=(FakeTransport(), FakeTransport()))
    llm = next((p for p in pipeline.processors
                if isinstance(p, agent.ClockedLLMService)), None)
    ok &= check("the pipeline builds a ClockedLLMService", llm is not None)
    if llm is not None:
        full = llm._settings.system_instruction
        ok &= check("the persona still comes first",
                    full.startswith(config.SYSTEM_PROMPT))
        ok &= check("the clock follows the persona",
                    full.find("The current date and time here is")
                    > full.find(config.SYSTEM_PROMPT[-40:]))
        ok &= check("the no-tools prompt still forbids inventing a time",
                    config.CLOCK_GUARD in full)
    return ok


# ---------------------------------------------------------------------------
# 6. The prompt is sent ONCE, not twice
# ---------------------------------------------------------------------------
async def test_single_system_message() -> bool:
    print("\n6. the system prompt is sent once, not twice")
    llm = make_llm()
    from pipecat.processors.aggregators.llm_context import LLMContext

    # A context built the way build_pipeline now builds it: no system message.
    ctx = LLMContext(messages=[{"role": "user", "content": "What is the time?"}])
    inv = await llm._adapter.get_llm_invocation_params(
        ctx, system_instruction=llm._settings.system_instruction,
        convert_developer_to_user=True)
    msgs = inv["messages"]
    systems = [m for m in msgs if m.get("role") == "system"]
    ok = check("exactly one system message reaches the model",
               len(systems) == 1, f"{len(systems)} found")
    ok &= check("it carries the clock",
                bool(CLOCK_RE.search(systems[0]["content"] if systems else "")))
    ok &= check("the user turn is still there",
                any(m.get("role") == "user" for m in msgs))
    return ok


# ---------------------------------------------------------------------------
# 7. build_pipeline no longer puts a second copy in the context
# ---------------------------------------------------------------------------
def test_pipeline_context_is_clean() -> bool:
    print("\n7. build_pipeline leaves the context free of a system message")
    from test_endtoend_answer import FakeTransport

    pipeline, *_ = agent.build_pipeline(
        api_key="not-a-real-key", bridge=None, bus=None,
        transports=(FakeTransport(), FakeTransport()))

    llm = next((p for p in pipeline.processors
                if isinstance(p, agent.ClockedLLMService)), None)
    ok = check("the pipeline uses ClockedLLMService", llm is not None)
    if llm is None:
        return False

    agg = next((p for p in pipeline.processors
                if type(p).__name__ == "LLMUserAggregator"), None)
    ok &= check("the user aggregator is present", agg is not None)
    if agg is None:
        return False

    systems = [m for m in agg._context.messages if m.get("role") == "system"]
    ok &= check("the context carries no system message", len(systems) == 0,
                f"{len(systems)} found")
    ok &= check("the prompt lives on the service instead",
                bool(CLOCK_RE.search(llm._settings.system_instruction)))
    return ok


# ---------------------------------------------------------------------------
# 8. The clock format is what a person would say
# ---------------------------------------------------------------------------
def test_clock_format() -> bool:
    print("\n8. the clock reads like something a person would say")
    line = config.clock_line()
    ok = check("it names the weekday", bool(re.search(
        r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)", line)))
    ok &= check("it gives a 12-hour time", bool(re.search(r"\d{1,2}:\d{2}\s*[AP]M", line)))
    ok &= check("it names the local zone", bool(re.search(r"\([A-Z]{2,5}, UTC[+-]\d{4}\)", line)))
    ok &= check("it is a complete sentence ending in a period", line.rstrip().endswith("."))
    return ok


async def main() -> int:
    results = [
        test_prompt_has_a_clock(),
        test_prompt_forbids_inventing(),
        test_clock_refreshes(),
        test_refresh_is_idempotent(),
        test_clock_position(),
        await test_single_system_message(),
        test_pipeline_context_is_clean(),
        test_clock_format(),
    ]
    print("\n--- summary ---")
    passed = sum(1 for r in results if r)
    print(f"  {passed}/{len(results)} groups passed")
    print("\n" + ("ALL PASSED" if all(results) else "FAILURES ABOVE"))
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
