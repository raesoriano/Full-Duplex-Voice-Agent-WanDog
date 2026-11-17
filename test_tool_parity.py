"""Does WanDog expose the same tools OnIt does?

Task 1 asks that the voice agent be able to use the same tools as OnIt. That
is not a feature to build -- it is a property to pin down. WanDog spawns
OnIt's own MCP servers, so the toolset is whatever OnIt serves today. This
test fails the moment that stops being true: a renamed tool, a dropped server,
or an OnIt release that moves a tool into another profile.

It compares two sets, live:
  expected  -- the @mcp.tool functions registered in OnIt's task-tools server
  actual    -- the tools WanDog's bridge discovers by spawning that server

Run: .venv/bin/python test_tool_parity.py
"""
from __future__ import annotations

import asyncio
import os
import re
import sys

import config
import mcp_bridge

# Tools OnIt's task-tools server has registered, per the plan's inventory.
# Kept explicit on purpose: a tool silently disappearing should fail loudly,
# which a comparison against a freshly-scraped list would not catch.
EXPECTED = {
    "search", "fetch_content", "get_weather", "bash",
    "read_file", "write_file", "edit_file", "serve",
    "grep", "send_file", "github_repo", "search_document",
    "index_documents", "local_search",
}


def scrape_onit_tools() -> set[str]:
    """Read the @mcp.tool names out of OnIt's source, as an independent check."""
    src = os.path.join(config.ONIT_REPO, "src", "mcp", "servers", "tasks", "tools",
                       "mcp_server.py")
    if not os.path.exists(src):
        return set()
    text = open(src, encoding="utf-8").read()
    return set(re.findall(r"@mcp\.tool[^\n]*\n(?:async\s+)?def\s+(\w+)", text))


async def main() -> int:
    print(f"OnIt repo: {config.ONIT_REPO}")
    if not os.path.exists(config.ONIT_PYTHON):
        print(f"SKIP: no OnIt interpreter at {config.ONIT_PYTHON}")
        return 2

    from_src = scrape_onit_tools()
    if from_src:
        print(f"OnIt source registers {len(from_src)} tools")
        if from_src != EXPECTED:
            print(f"  only in source : {sorted(from_src - EXPECTED)}")
            print(f"  only in expected: {sorted(EXPECTED - from_src)}")

    bridge = mcp_bridge.build_bridge(guard=mcp_bridge.Guard(timeout=5))
    if bridge is None:
        print("FAIL: the bridge did not build (tools disabled or OnIt missing)")
        return 1
    try:
        await bridge.discover()
        # bridge.tools holds plain callables; the name is __name__.
        actual = {t.__name__ for t in bridge.tools}
    finally:
        await bridge.close()

    print(f"WanDog bridge exposes {len(actual)} tools")
    print("  " + ", ".join(sorted(actual)))

    missing = EXPECTED - actual
    extra = actual - EXPECTED

    ok = True
    if missing:
        print(f"\nFAIL: missing from WanDog: {sorted(missing)}")
        ok = False
    if extra:
        # Not a failure: OnIt may add tools, and WanDog should gain them.
        print(f"\nNOTE: new tools not in the expected list: {sorted(extra)}")
        print("      add them to EXPECTED so a future removal is caught.")
    if from_src and from_src != EXPECTED:
        print("\nFAIL: OnIt's source and this test's EXPECTED list disagree")
        ok = False

    print("\n" + ("PASS: tool parity holds" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
