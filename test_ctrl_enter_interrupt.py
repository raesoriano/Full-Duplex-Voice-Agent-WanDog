"""Live: Ctrl+Enter must stop a turn, and must still send when idle.

The JS is a backstop in the parent document, so nothing short of a real browser
proves it. This drives headless Chromium over CDP against a running dashboard
and presses the key the way a person does: a real key event through the
browser's input pipeline, aimed at the focused textarea.

Two states, both required:

  * busy  -> Ctrl+Enter must interrupt. Before this the handler returned early
             when the Send button was absent, so the shortcut was dead exactly
             when a turn was running -- the one moment it is wanted.
  * idle  -> Ctrl+Enter must still send, i.e. it must not have become a
             stop-only key.

What is asserted is the LOG, not the control file. The dashboard writes
<log>.stop.json and the agent consumes it, so by the time a test looks the file
is usually already gone. The log is the record of the action actually being
taken, which is what matters.

The test rewrites the dashboard log (it has to, to put the UI into a busy
state) and restores it afterwards, so a real session is not lost.

Run:  .venv/bin/python test_ctrl_enter_interrupt.py
Needs the dashboard on :8501, the agent running, and chromium installed.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import config

URL = "http://127.0.0.1:8501"
PORT = 9411
LOG = Path(config.DASHBOARD_LOG)
STOP = Path(config.DASHBOARD_STOP_FILE)
TYPED = Path(config.DASHBOARD_TYPED_FILE)

BUSY_EVENTS = [{"kind": "user_started"}, {"kind": "user_text", "text": "what is a cat"},
               {"kind": "bot_started"}]
IDLE_EVENTS = BUSY_EVENTS + [{"kind": "bot_stopped"},
                             {"kind": "bot_text", "text": "a cat"}]


def ws_url(port: int) -> str:
    for _ in range(50):
        try:
            tabs = json.loads(urllib.request.urlopen(
                f"http://127.0.0.1:{port}/json", timeout=2).read())
            for t in tabs:
                if t.get("type") == "page":
                    return t["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("chromium devtools never came up")


def write_log(events: list[dict]) -> None:
    now = time.time()
    lines = [json.dumps({"kind": "session_start", "llm": "test", "ts": now - 5})]
    for e in events:
        lines.append(json.dumps({**e, "ts": now - 3}))
    LOG.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_log() -> list[dict]:
    try:
        return [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def cleanup_files() -> None:
    for p in (STOP, TYPED):
        try:
            p.unlink()
        except OSError:
            pass


async def press_ctrl_enter(busy: bool, wait_kind: str) -> dict:
    """Load the page in the given state, press Ctrl+Enter, wait for `wait_kind`."""
    import websockets

    write_log(BUSY_EVENTS if busy else IDLE_EVENTS)
    cleanup_files()
    before = len(read_log())

    proc = subprocess.Popen(
        ["chromium", "--headless=new", "--disable-gpu", "--no-sandbox",
         f"--remote-debugging-port={PORT}", "--window-size=1100,800", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        async with websockets.connect(ws_url(PORT), max_size=64 * 1024 * 1024) as ws:
            n = 0

            async def cmd(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg.get("result", {})

            async def js(expr):
                r = await cmd("Runtime.evaluate", expression=expr,
                              returnByValue=True, awaitPromise=True)
                return r.get("result", {}).get("value")

            await cmd("Page.enable")
            await cmd("Runtime.enable")
            await cmd("Page.navigate", url=URL)
            for _ in range(150):
                if await js("!!document.querySelector('.st-key-composer textarea')"):
                    break
                await asyncio.sleep(0.4)
            await asyncio.sleep(3.0)

            control = await js(
                "document.querySelector('.st-key-stopbtn button') ? 'stop'"
                " : (document.querySelector('.st-key-sendbtn button') ? 'send' : 'none')")

            focused = await js(
                "(() => { const t = document.querySelector('.st-key-composer textarea');"
                " if (!t) return false; t.focus(); return document.activeElement === t; })()")

            # Type through the browser, not by assigning .value: the box is a
            # React-controlled textarea, so a scripted assignment does not reach
            # the app's state and the Send handler would read an empty string.
            await cmd("Input.insertText", text="what is a cat")
            await asyncio.sleep(0.5)

            # A real key event, not a synthetic dispatch: this goes through the
            # browser's input pipeline, so it exercises the same path a person's
            # keypress does (including whether the handler is installed at all).
            for kind in ("rawKeyDown", "keyUp"):
                await cmd("Input.dispatchKeyEvent", type=kind, modifiers=2,
                          key="Enter", code="Enter", windowsVirtualKeyCode=13,
                          nativeVirtualKeyCode=13)

            found = None
            for _ in range(40):                     # up to ~8 s
                await asyncio.sleep(0.2)
                for ev in read_log()[before:]:
                    if ev.get("kind") == wait_kind:
                        found = ev
                        break
                if found:
                    break
            return {"control": control, "focused": focused, "event": found}
    finally:
        proc.terminate()
        proc.wait(timeout=10)


async def main() -> int:
    if not Path("/usr/bin/chromium").exists():
        print("SKIP: chromium not installed")
        return 2

    backup = LOG.read_text() if LOG.exists() else None
    failures: list[str] = []
    try:
        print("1. while a turn is in flight, Ctrl+Enter interrupts")
        r = await press_ctrl_enter(busy=True, wait_kind="interrupted")
        print(f"   control showing : {r['control']!r}   (want 'stop')")
        print(f"   textarea focused: {r['focused']}")
        print(f"   interrupted     : {bool(r['event'])}   (want True)")
        if r["control"] != "stop":
            failures.append(
                f"the composer showed {r['control']!r} while the log said a turn "
                "was in flight, so the busy case was never exercised")
        elif not r["event"]:
            failures.append(
                "Ctrl+Enter during a turn produced no 'interrupted' event; the "
                "shortcut is dead exactly when a turn is running")

        print("2. while idle, Ctrl+Enter still sends")
        r = await press_ctrl_enter(busy=False, wait_kind="user_text")
        print(f"   control showing : {r['control']!r}   (want 'send')")
        print(f"   message sent    : {bool(r['event'])}   (want True)")
        if r["control"] != "send":
            failures.append(f"the composer showed {r['control']!r} while idle")
        elif not r["event"]:
            failures.append("Ctrl+Enter while idle did not send the message")
    finally:
        if backup is not None:
            LOG.write_text(backup)
            print("\n(log restored)")
        cleanup_files()

    if failures:
        print()
        for f in failures:
            print("FAIL:", f)
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
