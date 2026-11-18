"""Replay and the handover notes.

Phase 3 + 4. Two claims:

  * a continued session's context starts with the turns it already had, and a
    restarted one starts empty -- so "clear chat history" is what the flag
    means, with no clear-file involved;
  * the notes file is written as turns land and compacted on shutdown, and it
    is readable as a plain file from inside the jail the session's tools see.

Run:  .venv/bin/python test_session_replay.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
_TMP = Path(tempfile.mkdtemp(prefix="fw_replay_"))
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


def write_log(sid: str, events: list[dict]) -> None:
    p = session.events_path(sid)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
print("\nreplay")
d = session.create()
sid = d.name

write_log(sid, [
    {"kind": "session_start", "llm": "m"},
    {"kind": "user_text", "text": "what is the weather"},
    {"kind": "bot_text", "text": "Sunny, 21 degrees."},
    {"kind": "user_text", "text": "and tomorrow"},
    {"kind": "bot_text", "text": "Rain in the afternoon."},
    {"kind": "user_text", "text": "thanks"},
    {"kind": "bot_text", "text": "Any time."},
])

msgs = session.replay_messages(sid)
check("three turns replay as six messages", len(msgs) == 6, str(len(msgs)))
check("order is preserved", [m["content"] for m in msgs][:2]
      == ["what is the weather", "Sunny, 21 degrees."],
      str([m["content"] for m in msgs][:2]))
check("roles alternate user/assistant",
      [m["role"] for m in msgs] == ["user", "assistant"] * 3,
      str([m["role"] for m in msgs]))
check("tool traffic is not replayed", all("tool" not in m["content"] for m in msgs))
check("the system prompt is not in the replay",
      not any("You are WanDog" in m["content"] for m in msgs))

turns = session.read_turns(sid)
check("read_turns returns the turns", len(turns) == 6, str(len(turns)))

limited = session.read_turns(sid, limit=2)
check("the limit keeps the NEWEST turns", [t["text"] for t in limited]
      == ["thanks", "Any time."], str([t["text"] for t in limited]))

# A clear truncates history rather than being replayed past.
write_log(sid, [
    {"kind": "user_text", "text": "old question"},
    {"kind": "bot_text", "text": "old answer"},
    {"kind": "cleared"},
    {"kind": "user_text", "text": "new question"},
    {"kind": "bot_text", "text": "new answer"},
])
after_clear = session.replay_messages(sid)
check("a cleared event truncates the replay", len(after_clear) == 2,
      str(len(after_clear)))
check("only the post-clear turns survive",
      [m["content"] for m in after_clear] == ["new question", "new answer"],
      str([m["content"] for m in after_clear]))

# A corrupt line is skipped, not fatal.
p = session.events_path(sid)
with p.open("a", encoding="utf-8") as f:
    f.write('{"kind": "user_text", "text": "half a li\n')  # truncated
    f.write(json.dumps({"kind": "user_text", "text": "after the mess"}) + "\n")
check("a truncated line does not break the replay",
      len(session.replay_messages(sid)) == 3, str(session.replay_messages(sid)))

# An empty or missing log replays nothing.
empty = session.create()
check("a new session replays nothing",
      session.replay_messages(empty.name) == [])
check("a missing log replays nothing",
      session.replay_messages("2026-01-01T0000-abcd") == [])

# A restarted session has no history, which is what "clear chat history" means.
os.environ.pop("WANDOG_DASHBOARD_LOG", None)
session.write_pointer(sid)
restarted = session.activate(restart=True)
check("a restarted session replays nothing",
      session.replay_messages(restarted.name) == [])
check("the continued session still replays its history",
      len(session.replay_messages(sid)) > 0)

# ---------------------------------------------------------------------------
# Notes
# ---------------------------------------------------------------------------
print("\nnotes")
n = session.create()
nsid = n.name
before = session.notes_path(nsid).read_text()

session.append_turn(nsid, "user", "how tall is the tower")
session.append_turn(nsid, "assistant", "Three hundred metres.")
text = session.notes_path(nsid).read_text()
check("a user turn is appended", "user: how tall is the tower" in text)
check("a bot turn is appended", "assistant: Three hundred metres." in text)
check("appending did not truncate the header", text.startswith(before[:40]),
      text[:40])

# Long text is flattened to one line so the file stays greppable.
session.append_turn(nsid, "user", "a\n\nmulti   line\n" + "x" * 500)
body = session.notes_path(nsid).read_text().splitlines()[-1]
check("a turn is one line", "\n" not in body)
check("a long turn is truncated with an ellipsis",
      body.endswith("\u2026") and len(body) < 260, f"{len(body)}: {body[-20:]}")

# Empty text writes nothing.
lines_before = len(session.notes_path(nsid).read_text().splitlines())
session.append_turn(nsid, "user", "   ")
check("an empty turn writes nothing",
      len(session.notes_path(nsid).read_text().splitlines()) == lines_before)

# An upload is recorded in both the notes and meta.json.
session.note_upload(nsid, "report.pdf", 1234)
check("an upload is noted", "uploaded: report.pdf (1234 bytes)"
      in session.notes_path(nsid).read_text())
meta = json.loads((n / "meta.json").read_text())
check("an upload lands in meta.json",
      meta["uploads"][0]["name"] == "report.pdf", str(meta.get("uploads")))

# Compaction replaces the running lines with a summary.
write_log(nsid, [
    {"kind": "user_text", "text": "what does the report say"},
    {"kind": "bot_text", "text": "It says the bridge is late."},
])
body = session.compact(nsid, model="test-model", note="handover for the next one")
compacted = session.notes_path(nsid).read_text()
check("compaction writes a summary section", "## Summary" in compacted)
check("compaction names the session", nsid in compacted)
check("compaction records the model", "test-model" in compacted)
check("compaction lists what was asked",
      "what does the report say" in compacted, compacted[:400])
check("compaction lists what was answered",
      "It says the bridge is late." in compacted)
check("compaction keeps the note", "handover for the next one" in compacted)
check("compaction keeps the upload", "report.pdf" in compacted)
check("compaction drops the running turn lines",
      "- user: how tall is the tower" not in compacted)
check("compaction returns what it wrote", body == compacted)

# Idempotent: compacting twice does not double the content.
session.compact(nsid, model="test-model")
twice = session.notes_path(nsid).read_text()
check("compacting twice is stable", twice.count("## Summary") == 1,
      str(twice.count("## Summary")))

# The notes live outside the jail, so a session's own tools cannot rewrite them.
check("notes are outside the work/ jail",
      not str(session.notes_path(nsid)).startswith(str(session.work_dir(nsid))))

# ---------------------------------------------------------------------------
# The agent wires it up
# ---------------------------------------------------------------------------
print("\nagent wiring")
src = (REPO / "agent.py").read_text()
check("the agent imports session", "\nimport session" in src)
check("the agent replays into the context",
      "replayed = session.replay_messages(config.SESSION_ID)" in src)
check("the agent appends turns to the notes",
      'session.append_turn(config.SESSION_ID, "user"' in src
      and 'session.append_turn(config.SESSION_ID, "assistant"' in src)
check("the agent compacts on shutdown",
      "session.compact(config.SESSION_ID" in src)
check("session_start carries the id", '"session": ' in src
      or "session=config.SESSION_ID" in src)

# A real LLMContext accepts the replayed messages. This is the one claim that
# cannot be checked by reading the source: the message shape has to be one the
# adapter will actually send.
try:
    from pipecat.processors.aggregators.llm_context import LLMContext

    ctx = LLMContext(messages=session.replay_messages(sid))
    got = ctx.get_messages()
    check("LLMContext accepts the replayed messages", len(got) == len(
        session.replay_messages(sid)), str(len(got)))
    first = got[0]
    content = first.get("content") if isinstance(first, dict) else getattr(
        first, "content", None)
    check("a replayed message survives as user text",
          "new question" in str(content), str(content)[:120])
except Exception as e:  # noqa: BLE001
    check("LLMContext accepts the replayed messages", False, repr(e))

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all replay/notes checks passed")
