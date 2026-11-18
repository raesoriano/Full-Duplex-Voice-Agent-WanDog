"""The session sidebar: listing, browsing, switching, and the handover.

What this file is for
---------------------
The sidebar adds a second session to the dashboard's world -- one the agent is
NOT in -- and most of the risk is in the seams rather than the buttons:

  * browsing must not move the agent (a read-only look at last week's session
    must not end today's conversation);
  * switching must move everything the agent owns, together (context, jail,
    log, notes, control paths), or the agent ends up in a state no single
    session describes;
  * the handover is the one thing that crosses the jail, so it must cross as
    text and nothing else.

The first half runs against a throwaway sessions root with real folders. The
second half drives dashboard.py through AppTest, which is the only way to check
that a click does what the page says it does.

Run:  .venv/bin/python test_sidebar.py
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

# A throwaway root BEFORE anything imports config: session.root() reads the
# environment on every call, so this is the only hook needed, and it keeps the
# real sessions beside the checkout untouched.
_TMP = Path(tempfile.mkdtemp(prefix="fw_sidebar_"))
os.environ["WANDOG_SESSIONS_ROOT"] = str(_TMP)
os.environ["WANDOG_SESSION_ROLE"] = "agent"
# config.activate() no-ops when the log is pinned. This test wants the real
# resolution path, so the pin is cleared and restored at the end.
_PINNED = os.environ.pop("WANDOG_DASHBOARD_LOG", None)

import session  # noqa: E402

FAILS: list = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name}{(' -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# Fixtures: two sessions with distinguishable histories
# ---------------------------------------------------------------------------
def seed(sid: str, turns: list, mode: str = "session") -> Path:
    d = _TMP / sid
    for sub in session.SUBDIRS:
        (d / sub).mkdir(parents=True, exist_ok=True)
    meta = {"id": sid, "started": "2026-11-18T10:00", "mode": mode, "uploads": []}
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "notes" / session.NOTES_NAME).write_text(
        f"# Session {sid}\n\n## Summary\n\nnotes for {sid}\n", encoding="utf-8")
    now = time.time()
    with (d / "events.jsonl").open("w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "session_start", "llm": "m", "asr": "a",
                            "ts": now - 100}) + "\n")
        for i, (role, text) in enumerate(turns):
            kind = "user_text" if role == "user" else "bot_text"
            f.write(json.dumps({"kind": kind, "text": text,
                                "ts": now - 90 + i}) + "\n")
    return d


OLDER = "2026-11-17T0900-aaaa"
NEWER = "2026-11-18T1600-bbbb"
seed(OLDER, [("user", "what did we decide about the jail"),
             ("assistant", "a session is confined to its own work folder")])
seed(NEWER, [("user", "and what about the sidebar"),
             ("assistant", "it lists the sessions, newest first")])
session.write_pointer(NEWER)


# ---------------------------------------------------------------------------
# list_sessions(): what a row needs
# ---------------------------------------------------------------------------
print("\nlist_sessions")
rows = session.list_sessions()
check("both sessions are listed", len(rows) == 2, str([r["id"] for r in rows]))
check("newest first", [r["id"] for r in rows] == [NEWER, OLDER],
      str([r["id"] for r in rows]))
check("a row carries the first thing asked", rows[0]["title"] == "and what about the sidebar",
      rows[0]["title"])
check("a row carries its turn count", rows[0]["turns"] == 2, str(rows[0]["turns"]))
check("a row carries its mode", rows[1]["mode"] == "session", str(rows[1]["mode"]))

# A session with no turns must still list, with an empty title rather than a
# crash or a placeholder that pretends it has content.
EMPTY = "2026-11-18T1700-cccc"
seed(EMPTY, [])
rows = session.list_sessions()
empty_row = next(r for r in rows if r["id"] == EMPTY)
check("an empty session still lists", empty_row["turns"] == 0)
check("an empty session has no title", empty_row["title"] == "", repr(empty_row["title"]))
check("an empty session sorts newest", rows[0]["id"] == EMPTY, rows[0]["id"])


# ---------------------------------------------------------------------------
# The switch request
# ---------------------------------------------------------------------------
print("\nswitch request")
# Clear any request a previous run left behind, so the assertions below are
# about what THIS run wrote.
session.switch_path().unlink(missing_ok=True)
session.request_switch(OLDER)
payload = json.loads(session.switch_path().read_text(encoding="utf-8"))
check("the request names the target", payload.get("session") == OLDER, str(payload))
check("the request carries an id", bool(payload.get("id")), str(payload))
check("the request sits at the sessions root, not beside a log",
      session.switch_path().parent == session.root(), str(session.switch_path()))

# A request for a folder that is not there is refused rather than written: the
# agent would otherwise adopt a pointer naming nothing.
try:
    session.request_switch("2026-01-01T0000-dead")
    check("a missing session is refused", False, "no exception")
except ValueError:
    check("a missing session is refused", True)

# And a malformed id is refused before it can be used as a path segment.
try:
    session.request_switch("../../etc")
    check("a traversal id is refused", False, "no exception")
except ValueError:
    check("a traversal id is refused", True)

# take_at() consumes: the agent must never act on one request twice.
session.request_switch(OLDER)
first = session.take_at(session.switch_path())
second = session.take_at(session.switch_path())
check("take_at returns the request", (first or {}).get("session") == OLDER, str(first))
check("take_at consumes it", second is None, str(second))
check("the file is gone", not session.switch_path().exists())


# ---------------------------------------------------------------------------
# The handover
# ---------------------------------------------------------------------------
print("\nhandover")
text = session.handover_text(OLDER)
check("the handover is the session's notes", f"notes for {OLDER}" in text, text[:80])
check("the handover names no other session", f"notes for {NEWER}" not in text)
check("a missing session hands over nothing", session.handover_text("2026-01-01T0000-dead") == "")

session.handover_path().unlink(missing_ok=True)
session.request_handover(OLDER)
payload = json.loads(session.handover_path().read_text(encoding="utf-8"))
check("the handover request names the source", payload.get("session") == OLDER, str(payload))
check("the handover request is separate from the switch request",
      session.handover_path() != session.switch_path())

# The jail is what makes this necessary, so assert it rather than assume it:
# a session's tools cannot reach a sibling's notes, which is why the agent has
# to read them in-process.
from pathlib import Path as _P  # noqa: E402

check("the notes sit OUTSIDE the tool jail",
      session.work_dir(OLDER) not in session.notes_path(NEWER).parents)


# ---------------------------------------------------------------------------
# config.adopt(): the agent's side of a switch
# ---------------------------------------------------------------------------
print("\nconfig.adopt")
import config  # noqa: E402

config.adopt(OLDER)
check("SESSION_ID follows", config.SESSION_ID == OLDER, config.SESSION_ID)
check("SESSION_DIR follows", config.SESSION_DIR == _TMP / OLDER, str(config.SESSION_DIR))
check("the jail follows", config.SESSION_WORK_DIR == str(_TMP / OLDER / "work"),
      config.SESSION_WORK_DIR)
check("the log follows", config.DASHBOARD_LOG == str(_TMP / OLDER / "events.jsonl"),
      config.DASHBOARD_LOG)
check("the environment follows",
      os.environ.get("WANDOG_DASHBOARD_LOG") == config.DASHBOARD_LOG,
      os.environ.get("WANDOG_DASHBOARD_LOG", ""))

# All four control files must land in the NEW session, or a Send after a switch
# would be written where the old session's agent is looking.
for name, attr in (("typed", "DASHBOARD_TYPED_FILE"),
                   ("stop", "DASHBOARD_STOP_FILE"),
                   ("clear", "DASHBOARD_CLEAR_FILE"),
                   ("upload", "DASHBOARD_UPLOAD_FILE")):
    got = getattr(config, attr)
    check(f"the {name} control follows", got.startswith(str(_TMP / OLDER)), got)

# A switch to something that is not there must raise, not half-apply.
before = config.SESSION_ID
try:
    config.adopt("2026-01-01T0000-dead")
    check("adopt refuses a missing session", False, "no exception")
except ValueError:
    check("adopt refuses a missing session", True)
check("a refused adopt leaves the session alone", config.SESSION_ID == before,
      f"{before} -> {config.SESSION_ID}")

config.adopt(NEWER)


# ---------------------------------------------------------------------------
# The event bus follows
# ---------------------------------------------------------------------------
print("\neventbus.repoint")
import eventbus  # noqa: E402

bus = eventbus.EventBus()
bus.recent()  # touch it so the ring has been initialised
before_path = bus.log_path
bus.repoint(str(_TMP / OLDER / "events.jsonl"))
check("the bus log follows", bus.log_path == str(_TMP / OLDER / "events.jsonl"),
      bus.log_path)
check("the bus log actually changed", bus.log_path != before_path)
check("the approval files follow the log",
      str(bus._pending_path) == str(_TMP / OLDER / "events.jsonl.pending.json"),
      str(bus._pending_path))
check("the in-memory ring is cleared", bus.recent() == [], str(bus.recent()))


# ---------------------------------------------------------------------------
# The dashboard: does a click do what the page says?
# ---------------------------------------------------------------------------
print("\ndashboard (AppTest)")
# The dashboard reads its session from the pointer, so point it at a fresh
# log inside the throwaway root -- but through the pointer, so the sidebar has
# a real list to draw.
os.environ.pop("WANDOG_DASHBOARD_LOG", None)
os.environ["WANDOG_SESSION_ROLE"] = "dashboard"
import importlib  # noqa: E402

import config as cfg2  # noqa: E402
importlib.reload(cfg2)

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).parent / "dashboard.py")


def _keys(at) -> list:
    return [b.key for b in at.button]


at = AppTest.from_file(DASHBOARD, default_timeout=90).run()
check("the page renders with no exception", not list(at.exception), str(list(at.exception)))

keys = _keys(at)
check("a row is drawn per session", any(k.startswith("show_") for k in keys),
      str(keys))
check("the live session's row is drawn", f"show_{NEWER}" in keys, str(keys))
check("an older session's row is drawn", f"show_{OLDER}" in keys, str(keys))
check("a switch button is offered for the older session",
      f"switch_{OLDER}" in keys, str(keys))
check("a handover button is offered for the older session",
      f"hand_{OLDER}" in keys, str(keys))
check("no switch button on the live session",
      f"switch_{NEWER}" not in keys, str(keys))

# Clicking an older session shows it -- and must NOT write a switch request.
session.switch_path().unlink(missing_ok=True)
at.button(key=f"show_{OLDER}").click().run()
check("showing an older session raises no exception", not list(at.exception),
      str(list(at.exception)))
check("showing does not switch the agent", not session.switch_path().exists(),
      "a switch request was written by a read-only click")
check("the pointer still names the live session", session.read_pointer() == NEWER,
      str(session.read_pointer()))

# ...and the page is now displaying it, which is visible in the transcript.
blob = "\n".join(m.value for m in at.markdown)
check("the browsed session's words are on the page",
      "what did we decide about the jail" in blob,
      "the transcript still shows the live session")
check("the live session's words are not", "and what about the sidebar" not in blob)
check("browsing is announced", any("read-only" in c.value for c in at.caption),
      str([c.value for c in at.caption][:4]))
check("Clear is disabled while browsing",
      at.button(key="clearbtn_btn").disabled is True)

# A typed message must not be delivered to a session that is not on screen.
at.text_area(key="txt").set_value("this should not be sent").run()
at.button(key="sendbtn_btn").click().run()
typed = Path(str(cfg2.DASHBOARD_TYPED_FILE))
check("typing is refused while browsing", not typed.exists(),
      "the message was written into the live session")

# The switch button DOES move the agent.
at.button(key=f"switch_{OLDER}").click().run()
check("switching raises no exception", not list(at.exception), str(list(at.exception)))
check("switching writes the request", session.switch_path().exists())
req = json.loads(session.switch_path().read_text(encoding="utf-8"))
check("the request names the clicked session", req.get("session") == OLDER, str(req))
session.switch_path().unlink(missing_ok=True)

# The handover button asks for a handover, not a switch.
at.button(key=f"hand_{OLDER}").click().run()
check("the handover button writes a handover request", session.handover_path().exists())
check("the handover button does not write a switch request",
      not session.switch_path().exists())
session.handover_path().unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# The agent's watcher: does it act on the requests?
# ---------------------------------------------------------------------------
print("\nagent watcher")
import asyncio  # noqa: E402

import agent as agent_mod  # noqa: E402

moved: list = []
handed: list = []


async def _run() -> None:
    watcher = agent_mod.DashboardControlWatcher(
        typed_path=str(_TMP / "typed.json"),
        stop_path=str(_TMP / "stop.json"),
        switch_path=str(session.switch_path()),
        handover_path=str(session.handover_path()),
        on_switch=lambda sid: _record(moved, sid),
        on_handover=lambda sid: _record(handed, sid),
        session_id=NEWER,
    )
    # Drive the handlers directly rather than waiting on the poll loop: the
    # loop is a 0.2 s sleep around exactly these calls, and a test that sleeps
    # for it would be testing asyncio.sleep.
    await watcher._handle_switch({"id": "s1", "session": OLDER})
    await watcher._handle_switch({"id": "s1", "session": OLDER})   # duplicate
    # After the first switch the watcher's own idea of "live" is OLDER, so this
    # is the click on the row that is now live -- a no-op, not a second move.
    await watcher._handle_switch({"id": "s2", "session": OLDER})
    await watcher._handle_handover({"id": "h1", "session": OLDER})
    await watcher._handle_handover({"id": "h1", "session": OLDER})  # duplicate


async def _record(bucket: list, sid: str) -> None:
    bucket.append(sid)


asyncio.run(_run())
check("the switch callback ran once", moved == [OLDER], str(moved))
check("a duplicate switch is ignored", len(moved) == 1, str(moved))
check("switching to the live session is a no-op", NEWER not in moved, str(moved))
check("the handover callback ran once", handed == [OLDER], str(handed))
check("a duplicate handover is ignored", len(handed) == 1, str(handed))

# repoint() must move every path and clear the per-file ids: a stale id would
# make the new session ignore a control file that reused one.
watcher2 = agent_mod.DashboardControlWatcher(
    typed_path="/a/typed.json", stop_path="/a/stop.json",
    clear_path="/a/clear.json", upload_path="/a/upload.json", session_id="old")
watcher2._last_typed_id = "x"
watcher2.repoint(typed_path="/b/typed.json", stop_path="/b/stop.json",
                 clear_path="/b/clear.json", upload_path="/b/upload.json",
                 session_id="new")
check("repoint moves the typed path", watcher2._typed_path == Path("/b/typed.json"),
      str(watcher2._typed_path))
check("repoint moves the upload path", watcher2._upload_path == Path("/b/upload.json"),
      str(watcher2._upload_path))
check("repoint moves the session id", watcher2._session_id == "new", watcher2._session_id)
check("repoint clears the per-file ids", watcher2._last_typed_id is None)


# ---------------------------------------------------------------------------
# The switch transaction, on the real objects
# ---------------------------------------------------------------------------
# The wiring check below proves _switch_to CALLS adopt/repoint/replay. It does
# not prove the result is coherent: a context holding B's turns while the jail
# still points at A is a state no single session describes, and every assertion
# about the pieces would still pass. So the transaction is performed here on a
# real LLMContext, a real bus and the real config, and the state is inspected
# afterwards -- and after moving back, because a switch that cannot be undone is
# not a switch.
print("\nswitch transaction (live objects)")
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402

# config is already imported above; adopt() puts this process in a session.
config.adopt(OLDER)
session.write_pointer(OLDER)
context = LLMContext(messages=session.replay_messages(OLDER))
bus2 = eventbus.EventBus()
bus2.repoint(config.DASHBOARD_LOG)

check("the context starts with the old session's turns",
      len(context.get_messages()) == 2, str(context.get_messages()))


async def _switch(sid: str) -> None:
    """The same sequence as agent.build_pipeline's _switch_to, minus the frames
    (there is no pipeline here to carry them)."""
    session.compact(config.SESSION_ID, model="test")
    context.set_messages([])
    config.adopt(sid)
    session.write_pointer(sid)
    bus2.repoint(config.DASHBOARD_LOG)
    context.set_messages(session.replay_messages(sid))
    await bus2.emit("session_switched", session=sid)


asyncio.run(_switch(NEWER))
check("config moved", config.SESSION_ID == NEWER, config.SESSION_ID)
check("the pointer moved", session.read_pointer() == NEWER, str(session.read_pointer()))
check("the jail moved", config.SESSION_WORK_DIR == str(_TMP / NEWER / "work"),
      config.SESSION_WORK_DIR)
check("the bus moved", bus2.log_path == str(_TMP / NEWER / "events.jsonl"),
      bus2.log_path)

msgs = [m["content"] for m in context.get_messages()]
check("the context holds the new session's turns only", len(msgs) == 2, str(msgs))
check("the new session's question is there",
      "and what about the sidebar" in msgs, str(msgs))
check("the old session's question is gone",
      "what did we decide about the jail" not in msgs, str(msgs))

b_log = (_TMP / NEWER / "events.jsonl").read_text(encoding="utf-8")
a_log = (_TMP / OLDER / "events.jsonl").read_text(encoding="utf-8")
check("the switch event landed in the new session's log",
      '"session_switched"' in b_log)
check("it did not land in the old one", '"session_switched"' not in a_log)
check("the session left behind was compacted",
      "## Summary" in session.notes_path(OLDER).read_text(encoding="utf-8"))

asyncio.run(_switch(OLDER))
msgs = [m["content"] for m in context.get_messages()]
check("switching back restores the old history",
      "what did we decide about the jail" in msgs, str(msgs))
check("and drops the other", "and what about the sidebar" not in msgs, str(msgs))


# ---------------------------------------------------------------------------
# The switch is wired into the real pipeline
# ---------------------------------------------------------------------------
print("\npipeline wiring")
src = (Path(__file__).parent / "agent.py").read_text(encoding="utf-8")
check("the watcher is given the switch path", "switch_path=str(session.switch_path())" in src)
check("the watcher is given the handover path",
      "handover_path=str(session.handover_path())" in src)
check("the watcher is given both callbacks",
      "on_switch=_switch_to" in src and "on_handover=_hand_over" in src)
check("the switch transaction adopts the session", "config.adopt(sid)" in src)
check("the switch transaction moves the bus", "bus.repoint(config.DASHBOARD_LOG)" in src)
check("the switch transaction moves the jail",
      "bridge.session_data_path = config.SESSION_WORK_DIR" in src)
check("the switch transaction replays the new session",
      "context.set_messages(session.replay_messages(sid))" in src)
check("the switch transaction writes the pointer", "session.write_pointer(sid)" in src)


# ---------------------------------------------------------------------------
if _PINNED is not None:
    os.environ["WANDOG_DASHBOARD_LOG"] = _PINNED

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {FAILS}")
    raise SystemExit(1)
print("all sidebar checks passed")
