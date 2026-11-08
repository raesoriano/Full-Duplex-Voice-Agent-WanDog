"""Tests for the OnIt approval guard (mcp_bridge.Guard).

These exercise the harness side of OnIt's human-in-the-loop safeguard: when a
command is not on the allowlist, the OnIt server returns a ``needs_approval``
ticket and the Guard asks a person whether to redeem it. No OnIt server or
network is needed -- the tool call is stubbed.

Run:
    .venv/bin/python test_guard.py
"""
from __future__ import annotations

import asyncio
import json

import mcp_bridge


def _payload(cmd: str) -> str:
    """A realistic needs_approval payload, as the OnIt server emits it."""
    return json.dumps({
        "status": "needs_approval",
        "approval_id": "TOK123",
        "command": cmd,
        "reason": "not on allowlist",
        "subjects": ["lscpu"],
        "scopes": ["once", "session"],
        "expires_in": 900,
        "note": "The command was not run.",
    })


async def _allow_through() -> None:
    g = mcp_bridge.Guard(prompt=lambda p: "once")
    calls = []

    async def call(name, args):
        calls.append(dict(args))
        return "hello world"

    out = await g.handle("bash", {"command": "echo hi"}, None, call)
    assert out == "hello world"
    assert len(calls) == 1
    assert "approval_token" not in calls[0]  # no ticket for an allowed call
    print("1. allow-through OK")


async def _redeem_once() -> None:
    g = mcp_bridge.Guard(prompt=lambda p: "once")
    calls = []

    async def call(name, args):
        calls.append(dict(args))
        if "approval_token" not in args:
            return _payload("lscpu")
        assert args["approval_token"] == "TOK123"
        assert args["approval_scope"] == "once"
        return "cpu output"

    out = await g.handle("bash", {"command": "lscpu"}, None, call)
    assert out == "cpu output"
    assert len(calls) == 2 and calls[1]["approval_token"] == "TOK123"
    print("2. redeem(once) OK")


async def _refuse_and_remember() -> None:
    g = mcp_bridge.Guard(prompt=lambda p: "deny")
    calls = []

    async def call(name, args):
        calls.append(dict(args))
        if "approval_token" not in args:
            return _payload("sudo reboot")
        return "should not run"

    out = await g.handle("bash", {"command": "sudo reboot"}, None, call)
    assert out.startswith("[denied]")
    assert len(calls) == 1 and g.is_refused("sudo reboot")

    # The same command again is refused without re-asking a person.
    calls.clear()
    out2 = await g.handle("bash", {"command": "sudo reboot"}, None, call)
    assert "already declined" in out2
    print("3. refuse + remember OK")


async def _redeem_session() -> None:
    g = mcp_bridge.Guard(prompt=lambda p: "session")
    calls = []

    async def call(name, args):
        calls.append(dict(args))
        if "approval_token" not in args:
            return _payload("pip install foo==1.0")
        assert args["approval_scope"] == "session"
        return "installed"

    out = await g.handle("bash", {"command": "pip install foo==1.0"}, None, call)
    assert out == "installed"
    assert g.is_granted("pip install foo==1.0")
    print("4. redeem(session) OK")


async def _prose_passthrough() -> None:
    g = mcp_bridge.Guard(prompt=lambda p: "once")

    async def call(name, args):
        return "just some prose, no json"

    out = await g.handle("search", {"query": "x"}, None, call)
    assert out == "just some prose, no json"
    print("5. prose passthrough OK")


def _parse_approval() -> None:
    p = mcp_bridge.Guard._parse_approval(
        '{"status":"needs_approval","approval_id":"t","command":"c"}')
    assert p and p["approval_id"] == "t"
    assert mcp_bridge.Guard._parse_approval("hello") is None
    assert mcp_bridge.Guard._parse_approval('{"other":1}') is None
    assert mcp_bridge.Guard._parse_approval(123) is None
    print("6. _parse_approval OK")


async def _main() -> None:
    await _allow_through()
    await _redeem_once()
    await _refuse_and_remember()
    await _redeem_session()
    await _prose_passthrough()
    _parse_approval()
    print("\nALL GUARD TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(_main())
