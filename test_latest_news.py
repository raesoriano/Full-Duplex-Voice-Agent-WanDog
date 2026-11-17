"""The latest-news fixes: does a search result reach the model intact and dated?

Three defects were measured on the live agent, all of them on the Fweddie side
of the bridge. This test pins each one:

1. NO DATES. The model passed no `type` argument on 13 out of 13 real searches
   in /tmp/fweddie_events.jsonl, so every call hit the general web index, whose
   results carry no `date` field. It cannot rank by recency it cannot see.
   Fixed by _shape_search_args (fills the argument in) and by putting today's
   date in the system prompt (config.tools_system_prompt).

2. UNPARSEABLE HEAD. config.MCP_MODEL_RESULT_CHARS was applied as a blind
   `text[:limit]`, so a 1796-char / 5-result news payload arrived as 600 chars
   of invalid JSON: one title and a dangling fragment. Fixed by
   _cap_for_model's structural cap, which keeps whole results and stays valid
   JSON.

3. NO RETRY. OnIt's news branch calls ddgs once and turns any exception into
   {"error": "Search failed: ..."}; ddgs raises rather than returning [] when
   rate-limited, measured at ~2 failures in 8 calls. The web branch retries
   inside OnIt, the news branch does not, and we are not touching OnIt here.
   Fixed by retrying `search` in the bridge.

Offline: the MCP server is stubbed, so no OnIt process, key, or mic is needed.

Run: .venv/bin/python test_latest_news.py
"""
from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace

import config
import mcp_bridge

PAYLOAD = "/tmp/news_payload.json"


def news_payload() -> str:
    """The real 5-result news payload captured from the live search tool."""
    if os.path.exists(PAYLOAD):
        return open(PAYLOAD, encoding="utf-8").read()
    # Fallback so the test is not hostage to /tmp being cleared.
    return json.dumps([
        {"title": f"Result {i}", "snippet": "s" * 280,
         "date": "November 16, 2026", "source": "example.com",
         "url": f"https://example.com/{i}"}
        for i in range(5)
    ])


class FakeServer:
    """Stands in for the OnIt net MCP server."""

    name = "ToolsNetMCPServer"

    def __init__(self, results: list):
        self.results = list(results)
        self.calls: list = []

    async def call_tool(self, name, args):
        self.calls.append(dict(args))
        if not self.results:
            return json.dumps({"error": "Search failed: No results found."})
        return self.results.pop(0)

    async def list_tools(self):
        return []

    async def close(self):
        pass


def make_search_tool():
    """The `search` schema exactly as the live server advertises it.

    Note `type` has default "web": Pipecat fills that in from the signature, so
    the handler cannot tell "the model chose web" from "the model said nothing".
    """
    return SimpleNamespace(
        name="search",
        description="Search the web for news or general information.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"anyOf": [{"type": "string"}, {"type": "null"}],
                          "default": None},
                "type": {"default": "web", "type": "string"},
                "max_results": {"default": 5, "type": "integer"},
            },
            "additionalProperties": False,
        },
    )


async def run_tool(server, **kwargs) -> str:
    """Invoke the bridge's search handler and return what the model would see."""
    bridge = mcp_bridge.ToolBridgeProcessor([server], guard=mcp_bridge.Guard(timeout=1))
    bridge.announce = False
    handler = bridge._make_tool(server, make_search_tool())

    delivered: list = []

    async def result_callback(result, **cb_kwargs):
        delivered.append(result)

    params = SimpleNamespace(function_name="search", tool_call_id="call_1",
                              arguments=kwargs, result_callback=result_callback)
    await handler(params, **kwargs)
    assert delivered, "handler settled nothing"
    return delivered[0]


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  [{detail}]" if detail else ""))
    return ok


# ---------------------------------------------------------------------------
# 1. The cap must keep the payload parseable
# ---------------------------------------------------------------------------
def test_cap_keeps_json_valid() -> bool:
    print("\n1. structural cap keeps the payload parseable")
    text = news_payload()
    ok = True
    total = len(json.loads(text))
    for limit in (300, 600, 900, 1200, len(text) + 1):
        out = mcp_bridge._cap_for_model(text, limit)
        try:
            data = json.loads(out)
            valid = True
        except ValueError:
            data, valid = None, False
        kept = len(data.get("results", [])) if isinstance(data, dict) else (
            len(data) if isinstance(data, list) else 0)
        dates = all(r.get("date") for r in
                    (data.get("results", []) if isinstance(data, dict) else data or []))
        fits = len(out) <= limit or limit >= len(text)
        ok &= check(f"limit {limit:>5}: valid={valid} kept={kept}/{total} "
                    f"dates={dates} fits={fits}", valid and dates and fits,
                    f"{len(out)} chars")
    return ok


def test_cap_was_broken_before() -> bool:
    """Prove the old behaviour failed, so the fix is not cosmetic."""
    print("\n2. the old blind slice was invalid JSON (regression evidence)")
    text = news_payload()
    ok = True
    for limit in (600, 900, 1200):
        old = text[:limit]
        try:
            json.loads(old)
            was_valid = True
        except ValueError:
            was_valid = False
        ok &= check(f"blind text[:{limit}] parses: {was_valid}", not was_valid,
                    "expected broken" if not was_valid else "unexpectedly valid")
    return ok


def test_cap_non_json_unchanged() -> bool:
    print("\n3. plain text still falls back to the head slice")
    text = "x" * 5000
    out = mcp_bridge._cap_for_model(text, 600)
    return check("plain text is sliced and labelled",
                 out.startswith("x" * 100) and "truncated" in out,
                 f"{len(out)} chars")


# ---------------------------------------------------------------------------
# 2. A recency query must reach the server as type="news"
# ---------------------------------------------------------------------------
def test_news_autodetect() -> bool:
    print("\n4. a recency query is sent as type=news")
    cases = [
        ("who won the most recent US Senate election in Alaska", "news"),
        ("what is the latest news on the election", "news"),
        ("what is the capital of France", "web"),
        ("how do I bake sourdough bread", "web"),
    ]
    ok = True
    for query, expected in cases:
        shaped = mcp_bridge._shape_search_args({"query": query, "type": "web"})
        ok &= check(f"{query[:44]!r} -> type={shaped.get('type')}",
                    shaped.get("type") == expected, f"wanted {expected}")
    # An explicit non-default choice must survive.
    shaped = mcp_bridge._shape_search_args(
        {"query": "latest news", "type": "news", "max_results": 3})
    ok &= check("explicit type=news is preserved",
                shaped.get("type") == "news" and shaped.get("max_results") == 3,
                str(shaped))
    return ok


async def test_news_autodetect_reaches_server() -> bool:
    print("\n5. the shaped arguments actually reach the server")
    server = FakeServer([news_payload()])
    await run_tool(server, query="what is the latest news about the election",
                   type="web", max_results=5)
    ok = check("server saw type=news",
               server.calls and server.calls[0].get("type") == "news",
               str(server.calls[0] if server.calls else None))
    return ok


# ---------------------------------------------------------------------------
# 3. The retry
# ---------------------------------------------------------------------------
async def test_search_retry() -> bool:
    print("\n6. a failed search is retried and recovers")
    fail = json.dumps({"error": "Search failed: No results found."})
    server = FakeServer([fail, fail, news_payload()])
    out = await run_tool(server, query="latest news", type="news")
    ok = check(f"3 attempts, recovered on the third ({len(server.calls)} calls)",
               len(server.calls) == 3 and '"error"' not in out[:40])
    ok &= check("the model got dated results",
                "November" in out and '"date"' in out)

    print("\n7. a non-search tool is never retried")
    server = FakeServer([fail, news_payload()])
    bridge = mcp_bridge.ToolBridgeProcessor([server], guard=mcp_bridge.Guard(timeout=1))
    bridge.announce = False
    tool = SimpleNamespace(name="bash", description="Run a command.",
                           input_schema={"type": "object",
                                         "properties": {"command": {"type": "string"}},
                                         "required": ["command"]})
    handler = bridge._make_tool(server, tool)
    delivered: list = []

    async def cb(result, **kw):
        delivered.append(result)

    await handler(SimpleNamespace(function_name="bash", tool_call_id="c",
                                  arguments={}, result_callback=cb), command="ls")
    ok &= check("bash called exactly once", len(server.calls) == 1,
                f"{len(server.calls)} calls")
    return ok


# ---------------------------------------------------------------------------
# 4. The prompt carries today's date
# ---------------------------------------------------------------------------
def test_prompt_has_date() -> bool:
    print("\n8. the system prompt states today's date and asks for news")
    from datetime import date
    prompt = config.tools_system_prompt()
    today = date.today().strftime("%d %B %Y")
    ok = check(f"prompt contains {today!r}", today in prompt)
    ok &= check("prompt asks for type=news on recency",
                "type" in prompt and "news" in prompt)
    ok &= check("prompt warns against calling stale results current",
                "older than the question" in prompt or "Never call" in prompt)
    ok &= check("TOOLS_SYSTEM_PROMPT itself is unchanged and date-free",
                "Today is" not in config.TOOLS_SYSTEM_PROMPT,
                "date is added at call time, not import time")
    return ok


def test_cap_default() -> bool:
    print("\n9. the default cap is large enough for two or three results")
    ok = check(f"MCP_MODEL_RESULT_CHARS = {config.MCP_MODEL_RESULT_CHARS}",
               config.MCP_MODEL_RESULT_CHARS >= 900)
    text = news_payload()
    out = mcp_bridge._cap_for_model(text, config.MCP_MODEL_RESULT_CHARS)
    data = json.loads(out)
    kept = len(data.get("results", [])) if isinstance(data, dict) else len(data)
    ok &= check(f"a 5-result payload keeps {kept} whole results at the default",
                kept >= 2, f"{len(out)} chars")
    return ok


async def main() -> int:
    results = [
        test_cap_keeps_json_valid(),
        test_cap_was_broken_before(),
        test_cap_non_json_unchanged(),
        test_news_autodetect(),
        await test_news_autodetect_reaches_server(),
        await test_search_retry(),
        test_prompt_has_date(),
        test_cap_default(),
    ]
    print("\n--- summary ---")
    print(f"  {sum(results)}/{len(results)} groups passed")
    print("\nALL PASSED" if all(results) else "\nFAILURES ABOVE")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
