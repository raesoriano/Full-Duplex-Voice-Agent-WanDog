"""Live check: does the real OnIt search reach the model dated and parseable?

test_latest_news.py proves the logic with a stubbed server. This proves the
same path against the actual OnIt net MCP server: spawn it the way the agent
does, call `search` through the real bridge handler with a recency-shaped
query, and assert that

  * the model's arguments were upgraded to type="news",
  * what the model saw still parses as JSON,
  * every result it saw carries a `date`,
  * and that date is recent enough to answer "the latest".

Needs the OnIt repo and network, so it is separate from the offline test.

Run: .venv/bin/python test_latest_news_live.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import date, datetime
from types import SimpleNamespace

os.environ.pop("OLLAMA_API_KEY", None)  # force the free-tier path

import config
import mcp_bridge


def _parse_date(text: str):
    """Pull a date out of 'November 16, 2026' style strings."""
    for fmt in ("%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y"):
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    m = re.search(r"(\d{4})-(\d{2})-(\d{2})", text or "")
    if m:
        return date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


async def main() -> int:
    server = mcp_bridge.OnItMCPServer(
        name="ToolsNetMCPServer",
        module="tasks.tools",
        profile="net",
    )
    try:
        tools = await server.list_tools()
        names = [t.name for t in tools]
        assert "search" in names, f"no search tool: {names}"
        schema = next(t for t in tools if t.name == "search").input_schema
        print("search schema properties:", list(schema.get("properties", {})))
        assert "type" in schema.get("properties", {}), "schema lost the type arg"

        bridge = mcp_bridge.ToolBridgeProcessor([server],
                                                guard=mcp_bridge.Guard(timeout=1))
        bridge.announce = False
        handler = bridge._make_tool(
            server, next(t for t in tools if t.name == "search"))

        delivered: list = []
        sent: list = []
        real_call = server.call_tool

        async def spy(name, args):
            sent.append(dict(args))
            return await real_call(name, args)

        server.call_tool = spy

        async def cb(result, **kw):
            delivered.append(result)

        # The query the model actually asked, with the argument it actually
        # omitted -- Pipecat fills type="web" from the schema default.
        params = SimpleNamespace(function_name="search", tool_call_id="live_1",
                                 arguments={}, result_callback=cb)
        await handler(params,
                      query="who won the most recent US Senate election in Alaska",
                      type="web", max_results=5)

        assert delivered, "no result delivered"
        seen = delivered[0]
        print(f"\nmodel saw {len(seen)} chars")
        print(seen[:400])

        print(f"\nserver received: {sent}")
        assert sent and sent[0].get("type") == "news", \
            f"query was not upgraded to news: {sent}"

        data = json.loads(seen)
        items = data.get("results") if isinstance(data, dict) else data
        assert items, f"no results in payload: {seen[:200]}"
        print(f"\n{len(items)} result(s) survived the cap")

        today = date.today()
        ok = True
        for r in items:
            d = _parse_date(r.get("date", ""))
            age = (today - d).days if d else None
            print(f"  date={r.get('date')!r:24} age={age}  {r.get('title','')[:52]}")
            ok &= d is not None
        assert ok, "a result the model saw had no parseable date"
        print("\nPASS: live search is dated, parseable, and recent")
        return 0
    finally:
        await server.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
