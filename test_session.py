"""The session pointer, the folder layout, and the restart flag.

Phase 0 + 1 of the file-upload work. Everything here runs against a throwaway
root -- the real one is beside the checkout and must never be touched by a test.

Run:  .venv/bin/python test_session.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
_TMP = Path(tempfile.mkdtemp(prefix="fw_session_"))

# Point the module at the throwaway root BEFORE importing it: `root()` reads
# the environment on every call, so this is the only hook needed.
os.environ["WANDOG_SESSIONS_ROOT"] = str(_TMP)
# config.activate() no-ops when the log is already set, and the tests that pin
# it must keep passing. This test wants the real resolution path, so it clears
# it -- but keeps the pre-set value to restore for the no-op check below.
_PINNED = os.environ.pop("WANDOG_DASHBOARD_LOG", None)
os.environ["WANDOG_SESSION_ROLE"] = "agent"

import session  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name}{(' -- ' + detail) if detail else ''}")


# ---------------------------------------------------------------------------
# root()
# ---------------------------------------------------------------------------
print("\nroot")
r = session.root()
check("root honours WANDOG_SESSIONS_ROOT", r == _TMP.resolve(), str(r))
check("root is not inside the repo", not str(r).startswith(str(REPO) + os.sep),
      f"{r} vs {REPO}")

saved = os.environ.pop("WANDOG_SESSIONS_ROOT")
default = session.root()
os.environ["WANDOG_SESSIONS_ROOT"] = str(_TMP)
check("default root is beside the checkout, not inside it",
      default == REPO.parent / "wandog-sessions", str(default))
check("default root is repo-relative, not machine-specific",
      str(default).startswith(str(REPO.parent)), str(default))

# ---------------------------------------------------------------------------
# create()
# ---------------------------------------------------------------------------
print("\ncreate")
d1 = session.create()
sid1 = d1.name
check("session id matches the shape", bool(session.SESSION_ID_RE.match(sid1)), sid1)
check("folder exists", d1.is_dir())
check("notes/ exists", (d1 / "notes").is_dir())
check("work/uploads/ exists", (d1 / "work" / "uploads").is_dir())
check("work/created/ exists", (d1 / "work" / "created").is_dir())
check("work/tool-results/ exists", (d1 / "work" / "tool-results").is_dir())
check("notes are OUTSIDE the jail",
      not str(session.notes_path(sid1)).startswith(str(session.work_dir(sid1))))
check("events log is OUTSIDE the jail",
      not str(session.events_path(sid1)).startswith(str(session.work_dir(sid1))))

meta = json.loads((d1 / "meta.json").read_text())
check("meta has an id", meta.get("id") == sid1, str(meta.get("id")))
check("meta has a start time", bool(meta.get("started")))
check("meta mode starts at 'session'", meta.get("mode") == "session",
      str(meta.get("mode")))
check("notes file is not empty", session.notes_path(sid1).stat().st_size > 0)

check("pointer names the new session", session.read_pointer() == sid1,
      str(session.read_pointer()))

# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------
print("\ntraversal")
for bad in ("../etc", "2026-01-01T0000-zzzz", "", "../../root", "a/b",
            "2026-11-18T1620-a3f1/../../x"):
    try:
        session.session_dir(bad)
    except ValueError:
        check(f"refuses {bad!r}", True)
    else:
        check(f"refuses {bad!r}", False)

# ---------------------------------------------------------------------------
# Continue vs restart
# ---------------------------------------------------------------------------
print("\ncontinue / restart")
os.environ.pop("WANDOG_DASHBOARD_LOG", None)
again = session.activate()
check("activate() without the flag continues the same session",
      again == d1, f"{again} vs {d1}")
check("continuing did not mint a second folder",
      len([p for p in _TMP.iterdir() if p.is_dir()]) == 1,
      str(sorted(p.name for p in _TMP.iterdir())))

os.environ.pop("WANDOG_DASHBOARD_LOG", None)
fresh = session.activate(restart=True)
check("activate(restart=True) mints a new session", fresh != d1, str(fresh))
check("the new session is pointed at", session.read_pointer() == fresh.name)
check("the old session survives a restart", d1.is_dir())
check("the old session's notes survive", session.notes_path(sid1).exists())
check("a restart gives an empty log",
      not session.events_path(fresh.name).exists()
      or session.events_path(fresh.name).stat().st_size == 0)

check("wants_restart() sees the flag", session.wants_restart(["--restart-session"]))
check("wants_restart() sees --restart-session=x",
      session.wants_restart(["--restart-session=1"]))
check("wants_restart() ignores other flags",
      not session.wants_restart(["--verbose", "-x"]))
check("wants_restart() ignores a substring",
      not session.wants_restart(["--restart-session-please"]))

# ---------------------------------------------------------------------------
# The dashboard never mints
# ---------------------------------------------------------------------------
print("\nroles")
os.environ["WANDOG_SESSION_ROLE"] = "dashboard"
os.environ.pop("WANDOG_DASHBOARD_LOG", None)
followed = session.activate()
check("dashboard follows the agent's pointer", followed == fresh, str(followed))
os.environ.pop("WANDOG_DASHBOARD_LOG", None)
check("dashboard ignores --restart-session", session.activate(restart=True) == fresh)
os.environ["WANDOG_SESSION_ROLE"] = "agent"

# A dashboard with no session at all: empty root, no agent has run.
empty = _TMP / "empty-root"
empty.mkdir()
os.environ["WANDOG_SESSIONS_ROOT"] = str(empty)
os.environ["WANDOG_SESSION_ROLE"] = "dashboard"
os.environ.pop("WANDOG_DASHBOARD_LOG", None)
check("dashboard with no session returns None", session.activate() is None)
check("dashboard with no session creates no folder", list(empty.iterdir()) == [])
os.environ["WANDOG_SESSION_ROLE"] = "agent"
os.environ["WANDOG_SESSIONS_ROOT"] = str(_TMP)

# ---------------------------------------------------------------------------
# The no-op
# ---------------------------------------------------------------------------
print("\nno-op when the log is pinned")
os.environ["WANDOG_DASHBOARD_LOG"] = "/tmp/somewhere_else.jsonl"
os.environ.pop("WANDOG_SESSION_ROLE", None)
before = len([p for p in _TMP.iterdir() if p.is_dir()])
session.activate()
after = len([p for p in _TMP.iterdir() if p.is_dir()])
check("a pinned log mints nothing", before == after, f"{before} -> {after}")
check("a pinned log is left alone",
      os.environ["WANDOG_DASHBOARD_LOG"] == "/tmp/somewhere_else.jsonl",
      os.environ["WANDOG_DASHBOARD_LOG"])
os.environ.pop("WANDOG_DASHBOARD_LOG", None)

# ---------------------------------------------------------------------------
# list_sessions()
# ---------------------------------------------------------------------------
print("\nlist")
listed = session.list_sessions()
check("both sessions list", len(listed) == 2, str(len(listed)))
check("newest first", listed[0]["id"] > listed[1]["id"],
      f"{listed[0]['id']} vs {listed[1]['id']}")
check("each entry carries its mode", all(s.get("mode") for s in listed))

# ---------------------------------------------------------------------------
# config.py integration -- the whole point of resolving at import
# ---------------------------------------------------------------------------
print("\nconfig integration")
os.environ["WANDOG_SESSIONS_ROOT"] = str(_TMP)
os.environ["WANDOG_SESSION_ROLE"] = "agent"
os.environ.pop("WANDOG_DASHBOARD_LOG", None)
code = r"""
import json, os, sys
sys.path.insert(0, %r)
import config, session
print(json.dumps({
    "session_id": config.SESSION_ID,
    "session_dir": str(config.SESSION_DIR) if config.SESSION_DIR else None,
    "work": config.SESSION_WORK_DIR,
    "log": config.DASHBOARD_LOG,
    "typed": config.DASHBOARD_TYPED_FILE,
    "stop": config.DASHBOARD_STOP_FILE,
    "clear": config.DASHBOARD_CLEAR_FILE,
    "mic": config.DASHBOARD_MIC_FILE,
    "jail": config.MCP_DATA_PATH,
    "root": str(session.root()),
}))
""" % str(REPO)
env = dict(os.environ)
out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                     text=True, env=env, cwd=str(REPO))
if out.returncode != 0:
    check("config imports with a session", False, out.stderr[-800:])
else:
    got = json.loads(out.stdout.strip().splitlines()[-1])
    sid = session.read_pointer()
    check("config imported with a session", bool(got["session_id"]),
          str(got["session_id"]))
    check("config's session is the live one", got["session_id"] == sid,
          f"{got['session_id']} vs {sid}")
    check("DASHBOARD_LOG is the session's log",
          got["log"].endswith(f"{sid}/events.jsonl"), got["log"])
    for key in ("typed", "stop", "clear", "mic"):
        check(f"{key} control file sits in the session folder",
              Path(got[key]).parent == Path(got["log"]).parent,
              f"{got[key]} vs {got['log']}")
    check("the four control files are distinct",
          len({got["typed"], got["stop"], got["clear"], got["mic"]}) == 4)
    check("jail is the sessions root", got["jail"] == got["root"],
          f"{got['jail']} vs {got['root']}")
    check("the session's work dir is inside the jail",
          got["work"].startswith(got["jail"] + os.sep),
          f"{got['work']} vs {got['jail']}")

# --restart-session reaches config through session.activate()
code_restart = r"""
import json, sys
sys.path.insert(0, %r)
import config, session
print(json.dumps({"session_id": config.SESSION_ID, "pointer": session.read_pointer()}))
""" % str(REPO)
before_ptr = session.read_pointer()
out = subprocess.run([sys.executable, "-c", code_restart, "--restart-session"],
                     capture_output=True, text=True, env=env, cwd=str(REPO))
if out.returncode != 0:
    check("--restart-session reaches config", False, out.stderr[-800:])
else:
    got = json.loads(out.stdout.strip().splitlines()[-1])
    check("--restart-session reaches config",
          got["session_id"] and got["session_id"] != before_ptr,
          f"{got['session_id']} vs {before_ptr}")
    check("--restart-session moves the pointer", got["pointer"] == got["session_id"])

# ---------------------------------------------------------------------------
# A stale pointer (the folder is gone) must not wedge the agent
# ---------------------------------------------------------------------------
print("\nstale pointer")
stale_root = _TMP / "stale"
stale_root.mkdir()
(stale_root / ".current").write_text(json.dumps({"id": "2026-01-01T0000-dead"}))
os.environ["WANDOG_SESSIONS_ROOT"] = str(stale_root)
os.environ.pop("WANDOG_DASHBOARD_LOG", None)
d = session.activate()
check("a pointer at a missing folder mints a fresh session",
      d is not None and d.is_dir() and d.name != "2026-01-01T0000-dead",
      str(d))

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all session checks passed")
