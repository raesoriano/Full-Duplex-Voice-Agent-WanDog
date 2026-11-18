"""The per-session jail: does a tool call actually get confined?

Phase 2. The claim being tested is narrow and mechanical: when a session is
live, the bridge adds ``data_path=<session>/work`` to every tool call whose
schema declares it, overwriting anything the model sent, and adds nothing to
the tools that do not declare it.

The enforcement itself lives inside OnIt's server, so this test does not
re-implement it -- but it does drive a real spawned server to confirm the
boundary rejects a sibling path, because "the argument is sent" and "the
argument confines" are two different claims.

Run:  .venv/bin/python test_session_jail.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
_TMP = Path(tempfile.mkdtemp(prefix="fw_jail_"))
os.environ["WANDOG_SESSIONS_ROOT"] = str(_TMP)
os.environ["WANDOG_SESSION_ROLE"] = "agent"
os.environ.pop("WANDOG_DASHBOARD_LOG", None)

sys.path.insert(0, str(REPO))

import session  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name}{(' -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# The argument, without spawning anything
# ---------------------------------------------------------------------------
print("\nargument injection")

import mcp_bridge  # noqa: E402

d = session.create()
work = str(d / "work")
bridge = mcp_bridge.ToolBridgeProcessor(servers=[])
bridge.session_data_path = work


seen: dict[str, dict] = {}


class _Tool:
    """The shape OnItMCPServer.list_tools() returns."""

    def __init__(self, name, props):
        self.name = name
        self.description = f"{name} tool"
        self.input_schema = {"type": "object", "properties": props,
                             "required": []}


class _Server:
    """A stub OnIt server. The handler closes over the server it was built for,
    so the recording has to happen here rather than in _tool_server."""

    name = "ToolsLocalMCPServer"

    async def call_tool(self, name, payload):
        seen[name] = dict(payload)
        return type("R", (), {"content": []})()


# _make_tool builds one callable; discover() is what appends it to self.tools.
for tool_name in ("read_file", "write_file", "bash", "local_search", "search"):
    bridge.tools.append(
        bridge._make_tool(_Server(), _Tool(tool_name, {"data_path": {"type": "string"},
                                                      "path": {"type": "string"}})))
bridge.tools.append(
    bridge._make_tool(_Server(), _Tool("get_weather", {"place": {"type": "string"}})))

check("tools declaring data_path are recorded",
      {"read_file", "write_file", "bash", "local_search", "search"}
      <= bridge._accepts_data_path, str(sorted(bridge._accepts_data_path)))
check("a tool without data_path is not recorded",
      "get_weather" not in bridge._accepts_data_path)

# Drive a handler and watch what reaches the server. The stub server above
# records the payload; nothing is spawned, so this is purely about argument
# assembly.
async def drive(tool_name: str, **kwargs) -> None:
    handler = next(t for t in bridge.tools if t.__name__ == tool_name)

    async def _cb(*a, **k):
        return None

    params = type("P", (), {"result_callback": staticmethod(_cb)})()
    bridge.announce = False
    await handler(params, **kwargs)


async def main() -> None:
    await drive("read_file", path="a.txt")
    check("data_path is injected on a tool that declares it",
          seen.get("read_file", {}).get("data_path") == work,
          str(seen.get("read_file")))

    await drive("read_file", path="a.txt", data_path="/etc")
    check("the model's data_path is OVERWRITTEN, not merged",
          seen["read_file"].get("data_path") == work,
          str(seen.get("read_file")))

    await drive("get_weather", place="Tokyo")
    check("a tool without data_path gets none",
          "data_path" not in seen.get("get_weather", {}),
          str(seen.get("get_weather")))

    bridge.session_data_path = ""
    await drive("read_file", path="a.txt")
    check("no session => no injection (the pre-session behaviour)",
          "data_path" not in seen["read_file"], str(seen["read_file"]))


asyncio.run(main())

# ---------------------------------------------------------------------------
# The boundary, against a real spawned server
# ---------------------------------------------------------------------------
print("\nboundary (real server)")


async def real_boundary() -> None:
    import config
    if not Path(config.ONIT_PYTHON).exists():
        print("  skip: OnIt's interpreter is not on this machine")
        return

    guard = mcp_bridge.Guard(timeout=5)
    servers = [mcp_bridge.OnItMCPServer(s["name"], s["module"], s.get("profile"))
               for s in config.MCP_SERVERS
               if s.get("name") == "ToolsLocalMCPServer"]
    if not servers:
        print("  skip: no local tool server configured")
        return
    bridge2 = mcp_bridge.ToolBridgeProcessor(servers, guard=guard)
    try:
        await bridge2.discover()
    except Exception as e:  # noqa: BLE001
        print(f"  skip: could not spawn the server ({e})")
        return
    if not bridge2.tools:
        print("  skip: the server offered no tools")
        await bridge2.close()
        return

    sibling = _TMP / "sibling-session" / "work"
    sibling.mkdir(parents=True, exist_ok=True)
    (sibling / "secret.txt").write_text("not yours")

    bridge2.session_data_path = work
    bridge2.announce = False
    got: dict[str, str] = {}

    async def call(name: str, **kwargs) -> None:
        handler = next((t for t in bridge2.tools if t.__name__ == name), None)
        if handler is None:
            got[name] = "[no such tool]"
            return

        async def _cb(text):
            got[name] = text if isinstance(text, str) else str(text)

        params = type("P", (), {"result_callback": staticmethod(_cb)})()
        await handler(params, **kwargs)
        # The handler settles the call itself; give the callback a tick.
        await asyncio.sleep(0.1)

    (Path(work) / "mine.txt").write_text("mine")
    await call("read_file", path=str(Path(work) / "mine.txt"))
    check("the session's own file is readable",
          "mine" in got.get("read_file", ""), got.get("read_file", "")[:200])

    await call("read_file", path=str(sibling / "secret.txt"))
    body = got.get("read_file", "")
    check("a sibling session's file is refused",
          "not yours" not in body, body[:300])
    check("the refusal is explicit",
          any(w in body.lower() for w in ("error", "outside", "denied", "not allowed",
                                          "must be within", "refus")),
          body[:300])

    await call("write_file", path=str(sibling / "planted.txt"), content="x")
    check("a sibling path cannot be written",
          not (sibling / "planted.txt").exists())

    await bridge2.close()


asyncio.run(real_boundary())

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all jail checks passed")
