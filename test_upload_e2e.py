"""One upload, all the way through: dashboard stores it, agent speaks about it.

The other tests check the pieces. This one checks the join, which is where the
two-process design actually has to hold: the dashboard writes the file and a
control file, and a live DashboardControlWatcher -- the real pipeline stage --
picks the control file up, emits the event, and queues the spoken question.

Run:  .venv/bin/python test_upload_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
_TMP = Path(tempfile.mkdtemp(prefix="fw_uploade2e_"))
os.environ["WANDOG_SESSIONS_ROOT"] = str(_TMP / "sessions")
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


d = session.create()
sid = d.name
os.environ["WANDOG_DASHBOARD_LOG"] = str(session.events_path(sid))

import config  # noqa: E402  (imports after the session is pinned)

check("config sees the session", config.SESSION_ID == sid, config.SESSION_ID)
check("the upload control file sits in the session",
      Path(config.DASHBOARD_UPLOAD_FILE).parent == d,
      config.DASHBOARD_UPLOAD_FILE)

# ---------------------------------------------------------------------------
# The dashboard's half: store the file, drop the notification
# ---------------------------------------------------------------------------
print("\ndashboard half")
stored = session.store_upload(sid, "report.pdf", b"%PDF-1.4 the bridge is late")
payload = {
    "id": "u1",
    "name": stored.name,
    "path": str(stored),
    "bytes": stored.stat().st_size,
    "ts": 0,
}
Path(config.DASHBOARD_UPLOAD_FILE).write_text(json.dumps(payload))
check("the file is in the session's uploads/",
      stored.parent == session.uploads_dir(sid))
check("the control file is present", Path(config.DASHBOARD_UPLOAD_FILE).exists())

# ---------------------------------------------------------------------------
# The agent's half: the real watcher stage
# ---------------------------------------------------------------------------
print("\nagent half")


class _Bus:
    def __init__(self):
        self.events = []

    async def emit(self, kind, **fields):
        self.events.append({"kind": kind, **fields})


class _FakeTask:
    """Stands in for PipelineTask.queue_frame: records what would be pushed."""

    def __init__(self):
        self.frames = []

    async def queue_frame(self, frame):
        self.frames.append(frame)


async def main() -> None:
    import agent as agent_mod

    bus = _Bus()
    watcher = agent_mod.DashboardControlWatcher(
        config.DASHBOARD_TYPED_FILE,
        config.DASHBOARD_STOP_FILE,
        bus=bus,
        poll_secs=0.01,
        clear_path=config.DASHBOARD_CLEAR_FILE,
        upload_path=config.DASHBOARD_UPLOAD_FILE,
        session_id=sid,
    )
    pushed = _FakeTask()

    async def capture(frame):
        pushed.frames.append(frame)

    watcher.queue_frame = capture  # type: ignore[assignment]

    # Drive one poll by hand: _watch loops forever, so take the single control
    # file the same way the loop does.
    up = watcher._take(watcher._upload_path)
    check("the watcher found the upload", up is not None)
    if up is None:
        return
    check("the watcher consumed the control file (once only)",
          not Path(config.DASHBOARD_UPLOAD_FILE).exists())

    await watcher._handle_upload(up)

    upload_events = [e for e in bus.events if e["kind"] == "upload"]
    check("an upload event was emitted", len(upload_events) == 1, str(bus.events))
    if upload_events:
        ev = upload_events[0]
        check("the event names the file", ev.get("name") == "report.pdf",
              str(ev.get("name")))
        check("the event carries the session", ev.get("session") == sid)
        check("the event records the size",
              ev.get("bytes") == len(b"%PDF-1.4 the bridge is late"),
              str(ev.get("bytes")))

    spoken = [f for f in pushed.frames
              if type(f).__name__ == "TTSSpeakFrame"]
    check("the agent SPOKE rather than ingesting silently", len(spoken) == 1,
          str([type(f).__name__ for f in pushed.frames]))
    if spoken:
        text = spoken[0].text
        check("the spoken line names the file", "report.pdf" in text, text)
        check("the spoken line asks about scope",
              "only from that file" in text, text)
        check("the spoken line offers the wider tiers",
              "search the web" in text, text)

    appended = [f for f in pushed.frames
                if type(f).__name__ == "LLMMessagesAppendFrame"]
    check("the question also lands in the context", len(appended) == 1,
          str([type(f).__name__ for f in pushed.frames]))
    if appended:
        check("the appended message is the question, not a user turn",
              appended[0].messages[0]["role"] == "assistant",
              str(appended[0].messages[0]["role"]))
        check("the append does not start an inference",
              appended[0].run_llm is False, str(appended[0].run_llm))

    # The same file must not be announced twice: a dashboard rerun rewrites the
    # control file only on a NEW upload, but a leftover file must not re-ask.
    await watcher._handle_upload(up)
    check("handling the same payload twice is a no-op",
          len([e for e in bus.events if e["kind"] == "upload"]) == 1,
          str(len([e for e in bus.events if e["kind"] == "upload"])))


asyncio.run(main())

# ---------------------------------------------------------------------------
# The file the agent reads is the file the dashboard stored
# ---------------------------------------------------------------------------
print("\nthe agent can read what the user sent")
check("the upload is inside the tool jail",
      str(stored).startswith(str(session.work_dir(sid)) + os.sep), str(stored))
check("the bytes are what was uploaded",
      stored.read_bytes() == b"%PDF-1.4 the bridge is late")

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all end-to-end upload checks passed")
