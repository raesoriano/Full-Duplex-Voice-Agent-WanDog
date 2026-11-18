"""Launch the agent's config resolution three times, the way agent.py does.

No audio hardware, no LLM: this only answers "which session am I, and where do
my files go", which is decided at config import. Run:

    .venv/bin/python test_session_launch.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

CODE = r'''
import sys, json
sys.path.insert(0, %r)
# argv[0] is what tells session._may_mint() this is the agent's entry point,
# and argv[1:] is where --restart-session has to be visible. python -c puts the
# script in argv[0] and everything after it in argv[1:], so rebuild both.
sys.argv = ["agent.py"] + sys.argv[1:]
import config, session
print(json.dumps({
    "sid": config.SESSION_ID,
    "dir": str(config.SESSION_DIR),
    "work": config.SESSION_WORK_DIR,
    "log": config.DASHBOARD_LOG,
    "jail": config.MCP_DATA_PATH,
    "mode": session.get_mode(config.SESSION_ID) if config.SESSION_ID else None,
}))
''' % str(REPO)


def run(*extra: str) -> dict:
    child = dict(os.environ)
    child.pop("WANDOG_DASHBOARD_LOG", None)
    child.pop("WANDOG_SESSION_ROLE", None)
    r = subprocess.run([sys.executable, "-c", CODE, *extra],
                       capture_output=True, text=True, cwd=str(REPO), env=child)
    if r.returncode != 0:
        raise SystemExit("child failed:\n" + r.stderr[-2000:])
    return json.loads(r.stdout.strip().splitlines()[-1])


FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILS.append(name)
        print(f"  FAIL {name}{(' -- ' + detail) if detail else ''}")


one = run()
print("first launch :", one["sid"])
check("a first launch mints a session", bool(one["sid"]), str(one["sid"]))
check("the session folder is the resolved one",
      one["dir"].endswith(one["sid"]), one["dir"])
check("the log is inside the session",
      one["log"] == f"{one['dir']}/events.jsonl", one["log"])
check("the jail is this session's work/",
      one["work"] == f"{one['dir']}/work", one["work"])
check("the work folder is inside the sessions root",
      one["work"].startswith(one["jail"] + os.sep),
      f"{one['work']} vs {one['jail']}")
check("a new session starts in the default tier", one["mode"] == "session",
      str(one["mode"]))

two = run()
print("second launch:", two["sid"])
check("a plain relaunch CONTINUES the same session", two["sid"] == one["sid"],
      f"{two['sid']} vs {one['sid']}")

three = run("--restart-session")
print("--restart-session:", three["sid"])
check("--restart-session mints a new session", three["sid"] != two["sid"],
      f"{three['sid']} vs {two['sid']}")
check("the old session's folder is still there",
      Path(one["dir"]).is_dir(), one["dir"])
check("the old session's notes survived",
      (Path(one["dir"]) / "notes" / "session.md").exists())
check("the new session has its own log",
      three["log"] != one["log"], three["log"])

four = run()
check("a relaunch after a restart continues the NEW session",
      four["sid"] == three["sid"], f"{four['sid']} vs {three['sid']}")

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all launch checks passed")
