"""Does a tool handler actually deliver its result to Pipecat?

This is the regression test for the bug where a tool ran, the result appeared
on the dashboard, and the agent said nothing. The cause was a contract
violation, not a model problem:

    Pipecat 1.15 does ``await item.handler(params)`` and DISCARDS the return
    value. A handler must call ``params.result_callback(result)``. Ours
    returned a string, so no FunctionCallResultFrame was ever broadcast, the
    assistant aggregator never appended the tool message, the LLM was never
    re-invoked, and the turn ended silently at the tool result.

The test is offline: it stubs the MCP server, so it needs no OnIt process, no
LLM key, and no microphone. It fails if the handler goes back to returning a
value instead of calling the callback, if the callback is skipped on error or
timeout, or if the model-visible result is not capped.

Run: .venv/bin/python test_tool_result.py
"""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import config
import mcp_bridge


class FakeServer:
    """Stands in for an OnIt MCP server; returns whatever it is told to."""

    name = "FakeServer"

    def __init__(self, result: str = '{"ok": true}', delay: float = 0.0):
        self.result = result
        self.delay = delay

    async def call_tool(self, name, args):
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.result

    async def list_tools(self):
        return []

    async def close(self):
        pass


def make_tool(name: str = "search"):
    """Build one bridge tool the way discover() would, with a stub server."""
    tool = SimpleNamespace(
        name=name,
        description="Search the web.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Terms."}},
            "required": ["query"],
        },
    )
    return tool


async def run_case(label: str, server, *, name: str = "search") -> tuple[bool, list, str]:
    """Invoke the handler and report whether it settled the call."""
    bridge = mcp_bridge.ToolBridgeProcessor([server], guard=mcp_bridge.Guard(timeout=1))
    bridge.announce = False          # no TTS in a headless test
    tool = make_tool(name)
    handler = bridge._make_tool(server, tool)

    delivered: list = []

    async def result_callback(result, **kwargs):
        delivered.append(result)

    params = SimpleNamespace(
        function_name=name,
        tool_call_id="call_1",
        arguments={"query": "x"},
        result_callback=result_callback,
    )

    returned = await handler(params, query="x")
    ok = bool(delivered)
    print(f"\n=== {label} ===")
    print(f"  handler returned      : {returned!r}  (must be None)")
    print(f"  result_callback calls : {len(delivered)}  (must be 1)")
    if delivered:
        text = delivered[0]
        print(f"  delivered {len(text)} chars: {text[:100]!r}")
    if returned is not None:
        print("  !! handler returned a value; Pipecat discards it")
    if not delivered:
        print("  !! no result delivered: the turn would die silently here")
    return ok and returned is None, delivered, label


async def main() -> int:
    ok = True
    results: list[tuple[bool, list, str]] = []

    # 1. A normal result settles the call.
    results.append(await run_case("normal result", FakeServer('{"ok": true}')))

    # 2. A big result is capped for the model but delivered.
    big = "x" * 5000
    passed, delivered, _ = await run_case("oversized result", FakeServer(big))
    capped = bool(delivered) and len(delivered[0]) <= config.MCP_MODEL_RESULT_CHARS + 80
    print(f"  model saw {len(delivered[0]) if delivered else 0} of {len(big)} chars "
          f"(cap {config.MCP_MODEL_RESULT_CHARS})")
    if not capped:
        print("  !! the cap did not apply, or nothing was delivered")
    results.append((passed and capped, delivered, "oversized result"))

    # 3. A tool that raises still settles the call, with the error as the result.
    class Boom(FakeServer):
        async def call_tool(self, name, args):
            raise RuntimeError("server exploded")

    passed, delivered, _ = await run_case("tool raises", Boom())
    has_error = bool(delivered) and "tool error" in delivered[0]
    if not has_error:
        print("  !! the error was not delivered to the model")
    results.append((passed and has_error, delivered, "tool raises"))

    # 4. A tool that times out still settles the call.
    old = config.MCP_TOOL_TIMEOUT
    config.MCP_TOOL_TIMEOUT = 0.2
    try:
        passed, delivered, _ = await run_case("tool times out", FakeServer(delay=2.0))
    finally:
        config.MCP_TOOL_TIMEOUT = old
    has_timeout = bool(delivered) and "timeout" in delivered[0]
    if not has_timeout:
        print("  !! the timeout was not delivered to the model")
    results.append((passed and has_timeout, delivered, "tool times out"))

    print("\n--- summary ---")
    for good, _delivered, label in results:
        print(f"  {'PASS' if good else 'FAIL'}  {label}")
    ok = all(good for good, _d, _l in results)
    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
