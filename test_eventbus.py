"""Fweddie -- tests for the event bus and the file-based approval handshake.

Covers:
  1. EventBus.emit appends JSONL events and keeps an in-memory ring.
  2. The dashboard's log reader (_read_events) picks up new events and skips
     a half-written last line.
  3. The file-based approval handshake: agent writes <log>.pending.json,
     "dashboard" writes <log>.decision.json, agent reads + consumes it.
  4. request_approval returns None (console fallback) when the dashboard is
     off.

Run:  .venv/bin/python test_eventbus.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import threading
import time

import config


def _fresh_log() -> str:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(path)
    return path


def test_emit_and_read() -> None:
    import eventbus
    log = _fresh_log()
    bus = eventbus.EventBus(log_path=log)

    async def go():
        await bus.emit("session_start", llm="m", asr="on:small", tools=14)
        await bus.emit("user_text", text="hi")
        await bus.emit("bot_text", text="hello!")

    asyncio.run(go())

    # Read back the way the dashboard does.
    import importlib
    import dashboard
    importlib.reload(dashboard)
    # Point the dashboard at our temp log.
    from pathlib import Path
    dashboard.LOG_PATH = Path(log)
    events = dashboard._read_events(0)
    kinds = [e[1]["kind"] for e in events]
    assert kinds == ["session_start", "user_text", "bot_text"], kinds
    assert len(bus.recent()) == 3
    print("1. emit + dashboard read OK")


def test_half_written_line() -> None:
    import dashboard
    from pathlib import Path
    log = _fresh_log()
    p = Path(log)
    # One complete event + one half-written line (no closing brace).
    p.write_text(
        json.dumps({"kind": "user_text", "text": "ok", "ts": 1.0}) + "\n"
        + '{"kind": "bot_text", "text": "cut off',  # no closing brace
        encoding="utf-8",
    )
    dashboard.LOG_PATH = p
    events = dashboard._read_events(0)
    assert len(events) == 1, events  # only the complete line
    assert events[0][1]["kind"] == "user_text"
    print("2. half-written line skipped OK")


def test_approval_handshake() -> None:
    import eventbus
    log = _fresh_log()
    bus = eventbus.EventBus(log_path=log)
    config.DASHBOARD_ENABLED = True
    config.MCP_APPROVAL_TIMEOUT = 5

    # The "dashboard": when the pending file appears, write a decision.
    def fake_dashboard():
        pend = Path(log).with_name(Path(log).name + ".pending.json")
        for _ in range(200):
            if pend.exists():
                time.sleep(0.2)
                dec = Path(log).with_name(Path(log).name + ".decision.json")
                dec.write_text(json.dumps({"decision": "session"}), encoding="utf-8")
                return
            time.sleep(0.05)

    t = threading.Thread(target=fake_dashboard)
    t.start()
    # Mirror the Guard: write the pending file, then wait for the decision.
    bus.write_pending({"command": "lscpu"})
    decision = asyncio.run(bus.request_approval({"command": "lscpu"}))
    t.join()
    assert decision == "session", decision
    # Decision file must be consumed (unlinked) by the agent.
    dec = Path(log).with_name(Path(log).name + ".decision.json")
    assert not dec.exists(), "decision file not consumed"
    print("3. approval handshake (dashboard -> agent) OK")


def test_approval_fallback_when_dashboard_off() -> None:
    import eventbus
    log = _fresh_log()
    bus = eventbus.EventBus(log_path=log)
    config.DASHBOARD_ENABLED = False
    decision = asyncio.run(bus.request_approval({"command": "lscpu"}))
    assert decision is None, decision
    print("4. dashboard off -> None (console fallback) OK")


if __name__ == "__main__":
    from pathlib import Path
    test_emit_and_read()
    test_half_written_line()
    test_approval_handshake()
    test_approval_fallback_when_dashboard_off()
    print("\nALL EVENTBUS TESTS PASSED")
