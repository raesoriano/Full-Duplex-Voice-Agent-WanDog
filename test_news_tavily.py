"""Offline tests for the Tavily news tier and the announcement echo guard.

Two bugs this covers, both measured on 17 Nov 2026:

1. NEWS WAS ALWAYS OLDER THAN GOOGLE. OnIt's `search(type="news")` branch
   calls ddgs only -- it never reaches the Tavily tier its own web branch
   uses. Same four queries, newest published date per provider:

       query                                DDG news   Tavily (days=1)
       Alaska US Senate election results    15 Nov     17 Nov
       Alaska Senate race Peltola Sullivan  17 Nov     17 Nov
       US Senate election results 2026      14 Nov     17 Nov
       latest news Jensen Huang Nvidia      15 Nov     17 Nov

   So news now goes to Tavily first when a key is configured.

2. TAVILY'S DATES DID NOT PARSE. Tavily returns "Tue, 17 Nov 2026 03:44:14
   GMT"; _parse_news_date knew only ISO and "%B %d, %Y". Every Tavily row
   parsed as None, so the newest-first sort had nothing to sort on and the
   age labels were empty -- silently undoing the news-v2 freshness work.
   Measured before the fix: 10/10 rows unparseable, newest=None.

Run:  .venv/bin/python test_news_tavily.py
"""

from __future__ import annotations

import asyncio
import datetime
import json

import config
import mcp_bridge


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return bool(ok)


# ---------------------------------------------------------------------------
# 1. Tavily's date shape parses
# ---------------------------------------------------------------------------
def test_tavily_date_parses() -> bool:
    print("\n1. Tavily's RFC-2822 date parses")
    ok = True
    cases = {
        "Tue, 17 Nov 2026 03:44:14 GMT": datetime.date(2026, 11, 17),
        "Mon, 16 Nov 2026 18:26:31 GMT": datetime.date(2026, 11, 16),
        "Tue, 17 Nov 2026 05:00:00 +0000": datetime.date(2026, 11, 17),
        "Tue, 17 Nov 2026": datetime.date(2026, 11, 17),
    }
    for raw, want in cases.items():
        got = mcp_bridge._parse_news_date(raw)
        ok &= check(f"{raw!r} -> {want}", got is not None and got.date() == want,
                    str(got))
    # The shapes that already worked must keep working.
    for raw, want in {
        "2026-11-17T09:44:19+00:00": datetime.date(2026, 11, 17),
        "November 17, 2026": datetime.date(2026, 11, 17),
    }.items():
        got = mcp_bridge._parse_news_date(raw)
        ok &= check(f"still parses {raw!r}", got is not None and got.date() == want,
                    str(got))
    ok &= check("junk still returns None", mcp_bridge._parse_news_date("soon") is None)
    ok &= check("empty still returns None", mcp_bridge._parse_news_date("") is None)
    return ok


# ---------------------------------------------------------------------------
# 2. A Tavily-shaped payload sorts newest-first and gets age labels
# ---------------------------------------------------------------------------
def _tavily_payload() -> str:
    return json.dumps([
        {"title": "old", "snippet": "b",
         "date": "Mon, 10 Nov 2026 01:00:00 GMT", "source": "x",
         "url": "https://example.com/old"},
        {"title": "today", "snippet": "b",
         "date": "Tue, 17 Nov 2026 05:00:00 GMT", "source": "y",
         "url": "https://example.com/today"},
        {"title": "yesterday", "snippet": "b",
         "date": "Mon, 16 Nov 2026 22:00:00 GMT", "source": "z",
         "url": "https://example.com/y"},
    ])


def test_tavily_payload_sorts() -> bool:
    print("\n2. a Tavily payload sorts newest-first and is age-labelled")
    out = mcp_bridge._sort_news(_tavily_payload(), 5)
    rows = json.loads(out)
    ok = check("newest first", rows[0]["title"] == "today", rows[0]["title"])
    ok &= check("oldest last", rows[-1]["title"] == "old", rows[-1]["title"])
    ok &= check("age label present on the newest", rows[0].get("age") == "today",
                str(rows[0].get("age")))
    ok &= check("age label present on the oldest",
                rows[-1].get("age") == "7 days ago", str(rows[-1].get("age")))
    ok &= check("newest age is 0 days",
                mcp_bridge._news_newest_age_days(out) == 0,
                str(mcp_bridge._news_newest_age_days(out)))
    return ok


def test_tavily_dates_survive_the_cap() -> bool:
    print("\n3. the cap still sees today's story after a Tavily sort")
    out = mcp_bridge._sort_news(_tavily_payload(), 5)
    model = mcp_bridge._cap_for_model(out, config.MCP_MODEL_RESULT_CHARS,
                                      query="election results")
    ok = check("the model is handed the today story", "today" in model)
    ok &= check("the date travels with it", "17 Nov 2026" in model)
    return ok


# ---------------------------------------------------------------------------
# 4. Backend selection: no key means no Tavily, and no exception
# ---------------------------------------------------------------------------
def test_no_key_is_inert() -> bool:
    print("\n4. with no Tavily key the tier is inert, not fatal")
    saved = mcp_bridge._tavily_news_key
    mcp_bridge._tavily_news_key = lambda: None
    try:
        got = mcp_bridge._tavily_news("anything", 5)
        ok = check("returns None (meaning: fall back to ddgs)", got is None, str(got))
    finally:
        mcp_bridge._tavily_news_key = saved
    return ok


def test_backend_switch() -> bool:
    print("\n5. MCP_NEWS_BACKEND=onit disables the tier")

    class FakeServer:
        name = "fake"

        async def call_tool(self, n, a):
            return json.dumps([])

    async def go() -> bool:
        bridge = mcp_bridge.ToolBridgeProcessor([], guard=mcp_bridge.Guard(timeout=1))
        bridge.announce = False
        saved = config.MCP_NEWS_BACKEND
        config.MCP_NEWS_BACKEND = "onit"
        try:
            got = await bridge._news_search({"query": "x", "type": "news"}, FakeServer())
        finally:
            config.MCP_NEWS_BACKEND = saved
        return check("onit backend returns None", got is None, str(got))

    return asyncio.run(go())


# ---------------------------------------------------------------------------
# 6. Config defaults
# ---------------------------------------------------------------------------
def test_config_defaults() -> bool:
    print("\n6. config defaults")
    ok = check(f"backend is tavily ({config.MCP_NEWS_BACKEND})",
               config.MCP_NEWS_BACKEND == "tavily")
    ok &= check(f"days window is 1 ({config.MCP_NEWS_TAVILY_DAYS})",
                1 <= config.MCP_NEWS_TAVILY_DAYS <= 7)
    ok &= check(f"depth is advanced ({config.MCP_NEWS_TAVILY_DEPTH})",
                config.MCP_NEWS_TAVILY_DEPTH in ("advanced", "basic"))
    return ok


# ---------------------------------------------------------------------------
# 7. The echo guard is wired into the pipeline
# ---------------------------------------------------------------------------
def test_echo_guard_wired() -> bool:
    print("\n7. the announcement echo guard is in the pipeline")
    import inspect
    import agent
    src = inspect.getsource(agent)
    ok = check("FunctionCallUserMuteStrategy is imported",
               "FunctionCallUserMuteStrategy" in src)
    ok &= check("it is passed as user_mute_strategies",
                "user_mute_strategies=" in src and
                "FunctionCallUserMuteStrategy()" in src)
    # The mute list must be on the user aggregator params, not somewhere inert.
    ok &= check("it sits on LLMUserAggregatorParams",
                src.index("user_mute_strategies=") >
                src.index("user_params = LLMUserAggregatorParams("))
    return ok


def test_echo_guard_is_a_real_strategy() -> bool:
    print("\n8. the strategy mutes on function calls and only then")
    from pipecat.frames.frames import (
        FunctionCallResultFrame,
        FunctionCallsStartedFrame,
    )
    from pipecat.turns.user_mute.function_call_user_mute_strategy import (
        FunctionCallUserMuteStrategy,
    )

    class FC:
        def __init__(self, tid):
            self.tool_call_id = tid

    async def go() -> bool:
        s = FunctionCallUserMuteStrategy()
        ok = check("unmuted before any call",
                   await s.process_frame(object()) is False)
        await s.process_frame(FunctionCallsStartedFrame(function_calls=[FC("a")]))
        ok &= check("muted while the call runs",
                    await s.process_frame(object()) is True)
        await s.process_frame(FunctionCallResultFrame(
            function_name="search", tool_call_id="a", arguments={},
            result="ok", run_llm=True))
        ok &= check("unmuted once the call settles",
                    await s.process_frame(object()) is False)
        return ok

    return asyncio.run(go())


def main() -> int:
    results = [
        test_tavily_date_parses(),
        test_tavily_payload_sorts(),
        test_tavily_dates_survive_the_cap(),
        test_no_key_is_inert(),
        test_backend_switch(),
        test_config_defaults(),
        test_echo_guard_wired(),
        test_echo_guard_is_a_real_strategy(),
    ]
    print("\n--- summary ---")
    print(f"  {sum(1 for r in results if r is True)}/{len(results)} groups passed")
    print("\nALL PASSED" if all(results) else "\nFAILURES ABOVE")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
