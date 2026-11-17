"""End-to-end proof that tool secrets reach the MCP server process.

Spawns the real OnIt MCP server through the real `_params()` spawn environment
and calls `get_weather` -- the tool that used to return
"OpenWeather API key not set" because nothing forwarded the key.

This makes one network call to OpenWeather. It is the only way to prove the
key arrived: a locally-resolved key proves nothing about what the subprocess
saw.

The stdio sessions are held open for the life of the process by design (see
OnItMCPServer._ensure_session), so this script closes the bridge and then
exits hard -- otherwise the non-daemon child processes keep the interpreter
alive after the work is done.

Run:  .venv/bin/python test_tool_secrets_live.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import mcp_bridge


async def run() -> int:
    bridge = mcp_bridge.build_bridge()
    if bridge is None:
        print("build_bridge() returned None (tools disabled or OnIt missing)")
        return 1
    servers = bridge.servers
    wanted = set(config.TOOL_SECRETS.values())

    print("--- what the spawn environment carries ---")
    for server in servers:
        params = server._params()
        carried = {
            k: len(v) for k, v in (params.env or {}).items()
            if k in wanted and v
        }
        print(f"  {server.name}: {carried or 'NO TOOL KEYS'}")

    print("\n--- calling get_weather through the bridge ---")
    try:
        for server in servers:
            if server.name != "ToolsNetMCPServer":
                continue
            try:
                raw = await server.call_tool("get_weather", {"place": "Tokyo, Japan"})
            except Exception as exc:  # noqa: BLE001
                print(f"  call raised: {exc}")
                return 1

            try:
                data = json.loads(raw)
            except Exception:
                print(f"  non-JSON: {raw[:300]}")
                return 1

            if "error" in data:
                print(f"  FAIL: tool returned an error -> {data.get('error')}")
                return 1

            cur = data.get("current", {})
            print(f"  location   : {data.get('location')}")
            print(f"  description: {cur.get('description')}")
            print(f"  temperature: {cur.get('temperature_c')} C")
            print("  PASS: the key reached the server")
            return 0

        print("  no ToolsNetMCPServer found")
        return 1
    finally:
        try:
            await bridge.close()
            print("\n  bridge closed")
        except Exception as exc:  # noqa: BLE001
            print(f"\n  bridge close failed: {exc}")


def main() -> int:
    code = 1
    try:
        code = asyncio.run(run())
    except BaseException as exc:  # noqa: BLE001
        print(f"unexpected failure: {exc}")
        code = 1
    finally:
        # Child MCP processes keep non-daemon pipes open, so the interpreter
        # would hang here. Flush, then leave with the real exit code.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)


if __name__ == "__main__":
    main()
