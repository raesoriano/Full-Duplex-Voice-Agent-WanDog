"""WanDog -- sessions: which folder is live, and what goes in it.

One session is one folder. It holds the event log, the handover notes, and the
working files; the folder is also the jail a session's tool calls are confined
to (see ``mcp_bridge``, which injects ``data_path`` per call).

The root
--------
The root is ``<repo>/../wandog-sessions`` -- beside the checkout, never inside
it, so sessions are not committed and a ``git clean`` cannot delete a person's
work. It is derived from ``__file__`` rather than hardcoded so a clone anywhere
lands its sessions next to itself. ``WANDOG_SESSIONS_ROOT`` overrides it.

The consequence worth knowing: moving or re-cloning the repo does not move the
sessions with it. "My sessions are gone" is almost always "I cloned somewhere
else".

    <root>/
      .current                    pointer: the live session id
      <session-id>/
        meta.json                 id, started, model, mode, uploads
        events.jsonl              the log the dashboard tails
        notes/session.md          handover material, compacted on shutdown
        work/                     the tool jail -- uploaded and created files
          uploads/  created/  tool-results/

``notes/`` sits OUTSIDE ``work/`` on purpose: the agent writes notes in-process
(no tool involved), so a session's tools can neither rewrite the log nor edit
the notes. That is the separation the feature asked for between the handover
file and the temporary folder.

Who mints a session
-------------------
Only the agent. The dashboard follows the pointer the agent wrote, and never
creates one -- otherwise opening the dashboard first would mint an empty session
that the agent then adopts, and the "new session" the user asked for by restart
would be indistinguishable from one the dashboard made by accident.

``--restart-session`` on the agent's command line mints a new session: new
folder, empty log, empty chat history. Without it the agent continues whatever
``.current`` names.

Why this module imports nothing from the project
------------------------------------------------
``config`` calls ``activate()`` at import time, before any path constant is
derived from the environment. If this module imported ``config`` that would be
a cycle. It uses the standard library only, and reads the environment directly.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import sys
import time
from pathlib import Path

# Beside the checkout, not inside it. See the module docstring.
REPO_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = REPO_DIR.parent / "wandog-sessions"

POINTER_NAME = ".current"

# A session id is a sortable timestamp plus 4 hex chars for uniqueness:
#     2026-11-18T1620-a3f1
# The pattern is also the traversal guard -- an id is used as a path segment,
# so anything that is not exactly this shape is refused rather than sanitised.
SESSION_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{4}-[0-9a-f]{4}$")

SUBDIRS = ("notes", "work/uploads", "work/created", "work/tool-results")

RESTART_FLAG = "--restart-session"

NOTES_NAME = "session.md"


# ---------------------------------------------------------------------------
# Where things live
# ---------------------------------------------------------------------------
def root() -> Path:
    """The sessions root. ``WANDOG_SESSIONS_ROOT`` wins when set."""
    raw = os.environ.get("WANDOG_SESSIONS_ROOT")
    if raw:
        return Path(raw).expanduser().resolve()
    return DEFAULT_ROOT


def pointer_path() -> Path:
    return root() / POINTER_NAME


def session_dir(sid: str) -> Path:
    """The folder for one session id. Refuses anything that is not an id."""
    if not SESSION_ID_RE.match(sid or ""):
        raise ValueError(f"not a session id: {sid!r}")
    return root() / sid


def notes_path(sid: str) -> Path:
    return session_dir(sid) / "notes" / NOTES_NAME


def events_path(sid: str) -> Path:
    return session_dir(sid) / "events.jsonl"


def work_dir(sid: str) -> Path:
    """The jail a session's tool calls are confined to."""
    return session_dir(sid) / "work"


# ---------------------------------------------------------------------------
# The pointer
# ---------------------------------------------------------------------------
def read_pointer() -> str | None:
    """The live session id, or None when there is no session.

    JSON is what we write; a bare id is accepted too so a pointer edited by
    hand still works. A malformed pointer reads as "no session" rather than
    raising -- the agent then mints one, which is the recoverable outcome.
    """
    try:
        raw = pointer_path().read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    if raw.startswith("{"):
        try:
            raw = str(json.loads(raw).get("id") or "")
        except ValueError:
            return None
    return raw if SESSION_ID_RE.match(raw) else None


def write_pointer(sid: str) -> Path:
    """Point ``.current`` at a session. Atomic: a crash cannot leave it half
    written, because the rename is what publishes it."""
    if not SESSION_ID_RE.match(sid or ""):
        raise ValueError(f"not a session id: {sid!r}")
    p = pointer_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps({"id": sid, "ts": time.time()}) + "\n",
                   encoding="utf-8")
    os.replace(tmp, p)
    return p


# ---------------------------------------------------------------------------
# Creating and resolving
# ---------------------------------------------------------------------------
def _new_id(now: float | None = None) -> str:
    stamp = time.strftime("%Y-%m-%dT%H%M", time.localtime(now))
    return f"{stamp}-{secrets.token_hex(2)}"


def _notes_header(sid: str) -> str:
    started = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    return (
        f"# Session {sid}\n"
        f"\n"
        f"Started: {started}\n"
        f"Mode: session (this session's files, then the web)\n"
        f"\n"
        f"This file is the handover record. A turn appends one line below the\n"
        f"marker; a clean shutdown compacts those lines into the summary block\n"
        f"under it. A different session can read this file when asked what\n"
        f"happened here.\n"
        f"\n"
        f"## Turns\n"
        f"\n"
    )


def create(now: float | None = None) -> Path:
    """Make a session folder, write its pointer, return the folder."""
    sid = _new_id(now)
    d = root() / sid
    for sub in SUBDIRS:
        (d / sub).mkdir(parents=True, exist_ok=True)
    meta = {
        "id": sid,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "created": now if now is not None else time.time(),
        "schema": 1,
        # The search tier this session answers from. Phase 6 widens it on the
        # user's say-so; the three tiers are: this file, this session, the web.
        "mode": "session",
        "uploads": [],
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                 encoding="utf-8")
    (d / "notes" / NOTES_NAME).write_text(_notes_header(sid), encoding="utf-8")
    write_pointer(sid)
    return d


def role() -> str:
    """``"agent"`` or ``"dashboard"``.

    Only the agent mints. ``streamlit run dashboard.py`` puts the streamlit
    binary in ``argv[0]``, which is the one reliable difference between the two
    entry points -- both otherwise run as ``__main__``. ``WANDOG_SESSION_ROLE``
    overrides, which is what tests use.
    """
    forced = (os.environ.get("WANDOG_SESSION_ROLE") or "").strip().lower()
    if forced in ("agent", "dashboard"):
        return forced
    argv0 = Path(sys.argv[0] or "").name.lower()
    return "dashboard" if "streamlit" in argv0 else "agent"


def _may_mint() -> bool:
    """Is this process allowed to create a session?

    Only an entry point, and only the agent. A session folder is a side effect
    of RUNNING WanDog, not of importing it -- without this rule every test that
    does ``import config`` would leave a session folder beside the checkout,
    and a throwaway ``python -c "import config"`` would too.

    ``WANDOG_SESSION_ROLE`` is an explicit statement of intent, so setting it
    counts as being an entry point. That is what the tests use.
    """
    if (os.environ.get("WANDOG_SESSION_ROLE") or "").strip().lower() == "agent":
        return True
    if role() != "agent":
        return False
    return Path(sys.argv[0] or "").name.lower() == "agent.py"


def wants_restart(argv: list[str] | None = None) -> bool:
    """Is ``--restart-session`` on this process's command line?

    A plain scan rather than argparse: ``agent.py`` parses no arguments today,
    and argparse at import time would choke on the dashboard's streamlit
    arguments in the process that shares this module.
    """
    args = sys.argv[1:] if argv is None else argv
    return any(a == RESTART_FLAG or a.startswith(RESTART_FLAG + "=")
               for a in args)


def activate(restart: bool | None = None) -> Path | None:
    """Resolve the live session and point ``WANDOG_DASHBOARD_LOG`` at it.

    Called from ``config`` at import, so every path constant derived from the
    log -- the log itself and the four control files -- follows automatically
    and the two processes cannot disagree about which session is live.

    No-op when ``WANDOG_DASHBOARD_LOG`` is already set: that is how a caller
    (and the tests that pin it) says "use this log, not a session".

    Returns the session folder, or None when there is no session (a dashboard
    with no agent having run yet).
    """
    if os.environ.get("WANDOG_DASHBOARD_LOG"):
        sid = read_pointer()
        if sid is None:
            return None
        d = session_dir(sid)
        return d if d.is_dir() else None

    if restart is None:
        restart = wants_restart()

    sid = None
    if not (restart and _may_mint()):
        sid = read_pointer()

    d = session_dir(sid) if sid else None
    if d is not None and not d.is_dir():
        # A pointer at a folder that is gone (a wiped root, a deleted session).
        # The folder is the session, so there is nothing to continue.
        d = None

    if d is None:
        if not _may_mint():
            return None
        d = create()

    os.environ["WANDOG_DASHBOARD_LOG"] = str(d / "events.jsonl")
    return d


# ---------------------------------------------------------------------------
# Reading the set back (the sidebar's data source, when it lands)
# ---------------------------------------------------------------------------
def list_sessions() -> list[dict]:
    """Every session on disk, newest first. Reads ``meta.json`` and tolerates
    a folder without one, so a half-created session still lists."""
    out: list[dict] = []
    try:
        entries = list(root().iterdir())
    except OSError:
        return out
    for d in entries:
        if not d.is_dir() or not SESSION_ID_RE.match(d.name):
            continue
        meta: dict = {}
        try:
            meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}
        out.append({
            "id": d.name,
            "dir": d,
            "started": meta.get("started") or "",
            "mode": meta.get("mode") or "session",
            "uploads": meta.get("uploads") or [],
        })
    out.sort(key=lambda s: s["id"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Replay: continuing a session means continuing its conversation
# ---------------------------------------------------------------------------
# How many turns of history a continued session carries. 20 turns is ~40
# messages, which is a few minutes of conversation and comfortably inside the
# model's window; the notes file is what carries anything older.
REPLAY_TURNS = int(os.environ.get("WANDOG_SESSION_REPLAY_TURNS", "20"))

# Only these two kinds are replayed. Tool calls and results are deliberately
# NOT reconstructed: they are not cleanly rebuildable as messages, and the
# duplicated-system-prompt bug (agent.py:838) is exactly what happens when the
# context is handed something it should not have. A continued session
# remembers what was said, not every tool's raw output.
_REPLAY_KINDS = {"user_text", "bot_text"}


def read_turns(sid: str, limit: int = REPLAY_TURNS) -> list[dict]:
    """The last ``limit`` user/bot texts of a session, oldest first.

    Stops at the most recent ``cleared`` event: Clear means the history starts
    there, so a replayed session must not resurrect what the user wiped. A
    corrupt or half-written line is skipped rather than fatal -- the log is
    appended by a process that can be killed mid-write.
    """
    path = events_path(sid)
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []

    parsed: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if isinstance(ev, dict) and ev.get("kind"):
            parsed.append(ev)

    # Find the last clear; everything before it is not this session's history.
    cut = 0
    for i, ev in enumerate(parsed):
        if ev.get("kind") == "cleared":
            cut = i + 1

    turns = []
    for ev in parsed[cut:]:
        if ev.get("kind") not in _REPLAY_KINDS:
            continue
        text = (ev.get("text") or "").strip()
        if text:
            turns.append({"role": "user" if ev["kind"] == "user_text"
                          else "assistant", "text": text})
    return turns[-limit:] if limit else turns


def replay_messages(sid: str, limit: int = REPLAY_TURNS) -> list[dict]:
    """The turns as LLMContext-shaped messages.

    ``{"role", "content"}`` dicts are what ``LLMContext.set_messages`` accepts.
    The system prompt is NOT here: it lives on the LLM service's
    ``system_instruction`` (see agent.build_pipeline), and putting it in the
    context as well sends it twice.
    """
    return [{"role": t["role"], "content": t["text"]}
            for t in read_turns(sid, limit)]


# ---------------------------------------------------------------------------
# The notes file: the handover record
# ---------------------------------------------------------------------------
# Two writers, on purpose:
#
#   1. ``append_turn`` runs as each turn lands. One compact line, written
#      immediately, so a hard kill -- a power cut, a SIGKILL -- still leaves a
#      record of what was said. This is the same reasoning as the event log:
#      the durable thing is the file, not the process.
#
#   2. ``compact`` runs on shutdown and replaces the running lines with a
#      summary block. That is what a person (or another session) reads: not a
#      transcript, but "what was this session about".
#
# The file lives in notes/, outside work/, so a session's own file tools cannot
# rewrite its history. The agent writes it in-process.

TURNS_MARKER = "## Turns"
SUMMARY_MARKER = "## Summary"
_LINE_LIMIT = 200


def _one_line(text: str, limit: int = _LINE_LIMIT) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "\u2026"


def append_turn(sid: str, role: str, text: str) -> None:
    """Record one turn in the notes file. Best-effort: a failure here must
    never break a voice turn."""
    text = _one_line(text)
    if not text:
        return
    try:
        p = notes_path(sid)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(f"- {role}: {text}\n")
    except OSError:
        pass


def note_upload(sid: str, name: str, size: int) -> None:
    """Record an uploaded file in the notes and in meta.json, so the session
    that inherits this one knows what it was given."""
    try:
        p = notes_path(sid)
        with p.open("a", encoding="utf-8") as f:
            f.write(f"- uploaded: {name} ({size} bytes)\n")
    except OSError:
        pass
    meta_file = session_dir(sid) / "meta.json"
    try:
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    uploads = meta.get("uploads") or []
    uploads.append({"name": name, "bytes": size, "ts": time.time()})
    meta["uploads"] = uploads
    try:
        meta_file.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    except OSError:
        pass


def compact(sid: str, model: str = "", note: str = "") -> str:
    """Rewrite the notes file as a handover summary. Returns the new text.

    Replacing rather than appending is the point: a summary that grows forever
    is a transcript, and the whole reason to compact is that the next reader
    should not have to read one.
    """
    turns = read_turns(sid, limit=0)
    meta: dict = {}
    try:
        meta = json.loads((session_dir(sid) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}

    asked = [_one_line(t["text"], 160) for t in turns if t["role"] == "user"]
    said = [_one_line(t["text"], 160) for t in turns if t["role"] == "assistant"]

    lines = [
        f"# Session {sid}",
        "",
        f"Started: {meta.get('started') or 'unknown'}",
        f"Compacted: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Model: {model or meta.get('model') or 'unknown'}",
        f"Mode: {meta.get('mode') or 'session'}",
        f"Turns: {len(turns)}",
    ]
    uploads = meta.get("uploads") or []
    if uploads:
        names = ", ".join(u.get("name", "?") for u in uploads)
        lines.append(f"Uploads: {names}")
    lines += ["", SUMMARY_MARKER, ""]
    if note.strip():
        lines += [_one_line(note, 1000), ""]
    lines += ["What was asked:", ""]
    lines += [f"- {q}" for q in asked] or ["- (nothing)"]
    lines += ["", "What was answered:", ""]
    lines += [f"- {a}" for a in said] or ["- (nothing)"]
    lines += [
        "",
        "---",
        "",
        "Read this file to continue this session's work in another one.",
        "",
    ]
    body = "\n".join(lines)
    try:
        p = notes_path(sid)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(body, encoding="utf-8")
        os.replace(tmp, p)
    except OSError:
        pass
    return body


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------
# The allowlist is by extension and it is short on purpose: the point of an
# upload is text the agent can read, and everything here is either text or a
# PDF the read_file tool can already extract. An extension that is not listed is
# refused rather than guessed at -- a wrong guess means the agent speaks about a
# file it never actually read.
UPLOAD_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".json", ".yaml", ".yml",
    ".log", ".py", ".sh", ".html", ".xml", ".rst", ".pdf",
}

# 2 MB. A voice turn is not a document-ingestion pipeline: a bigger file takes
# long enough to read that the "one moment" the agent says becomes a lie, and
# the result gets capped for the model anyway.
UPLOAD_MAX_BYTES = 2 * 1024 * 1024

UPLOADS_SUBDIR = "uploads"


def uploads_dir(sid: str) -> Path:
    return work_dir(sid) / UPLOADS_SUBDIR


def safe_upload_name(name: str) -> str:
    """Reduce a browser-supplied filename to one path segment.

    A filename arrives from the browser, so it is attacker-controlled: ``..``,
    absolute paths and separators are all possible. Rather than sanitising a
    path, this keeps only the last segment and refuses anything whose extension
    is not on the allowlist.
    """
    base = Path(str(name or "").replace("\\", "/")).name.strip()
    base = base.lstrip(".") or "upload"
    base = re.sub(r"[^A-Za-z0-9._ -]", "_", base)
    base = re.sub(r"\s+", " ", base).strip() or "upload"
    if len(base) > 120:
        stem, dot, ext = base.rpartition(".")
        base = (stem[: 110] + dot + ext) if dot else base[:120]
    if Path(base).suffix.lower() not in UPLOAD_EXTENSIONS:
        raise ValueError(
            f"{base!r} is not a file type WanDog reads. Allowed: "
            + ", ".join(sorted(UPLOAD_EXTENSIONS)))
    return base


def store_upload(sid: str, name: str, data: bytes) -> Path:
    """Write one uploaded file into the session's uploads/ folder.

    Returns the path. Raises ValueError with a sentence a person can hear when
    the file is refused, because the dashboard shows it and the agent says it.
    """
    if not data:
        raise ValueError("that file is empty")
    if len(data) > UPLOAD_MAX_BYTES:
        raise ValueError(
            f"that file is {len(data) / 1e6:.1f} MB; the limit is "
            f"{UPLOAD_MAX_BYTES / 1e6:.0f} MB")
    safe = safe_upload_name(name)
    d = uploads_dir(sid)
    d.mkdir(parents=True, exist_ok=True)
    target = d / safe
    # Never overwrite a previous upload: a second report.pdf in one session
    # would silently replace the file the agent was told to answer from.
    if target.exists():
        stem, ext = target.stem, target.suffix
        n = 2
        while (d / f"{stem}-{n}{ext}").exists():
            n += 1
        target = d / f"{stem}-{n}{ext}"
    target.write_bytes(data)
    note_upload(sid, target.name, len(data))
    return target


def uploaded_files(sid: str) -> list[dict]:
    """The files this session was given, newest last."""
    d = uploads_dir(sid)
    out: list[dict] = []
    try:
        for p in sorted(d.iterdir()):
            if p.is_file():
                out.append({"name": p.name, "path": str(p),
                            "bytes": p.stat().st_size})
    except OSError:
        pass
    return out


# ---------------------------------------------------------------------------
# The source-of-truth question
# ---------------------------------------------------------------------------
# Three tiers, narrowest first. The user's hierarchy, verbatim: this file, then
# the other files in this session, then the web.
#
#   strict    only the uploaded file(s). If the answer is not there, say so.
#   session   + the rest of the session's files          (the default)
#   expanded  + the web
#
# The tier is enforced by which tools the context is given, not by asking the
# model nicely -- see agent.apply_search_mode. A mode the model can ignore is
# not a promise.
SEARCH_MODES = ("strict", "session", "expanded")
DEFAULT_MODE = "session"

# Tools that reach outside the session. Dropped in strict and session.
_WEB_TOOLS = {"search", "fetch_content", "get_weather", "github_repo"}


def read_meta(sid: str) -> dict:
    try:
        return json.loads((session_dir(sid) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def get_mode(sid: str) -> str:
    mode = (read_meta(sid).get("mode") or DEFAULT_MODE).strip().lower()
    return mode if mode in SEARCH_MODES else DEFAULT_MODE


def set_mode(sid: str, mode: str) -> str:
    mode = (mode or "").strip().lower()
    if mode not in SEARCH_MODES:
        raise ValueError(f"unknown search mode {mode!r}; use one of "
                         + ", ".join(SEARCH_MODES))
    d = session_dir(sid)
    meta = read_meta(sid)
    meta["mode"] = mode
    meta.setdefault("id", sid)
    try:
        (d / "meta.json").write_text(json.dumps(meta, indent=2) + "\n",
                                     encoding="utf-8")
    except OSError:
        pass
    return mode


def allowed_tool_names(mode: str, names: list[str]) -> list[str]:
    """Which of the bridge's tools this mode may offer the model.

    ``strict`` keeps only file tools: the model must answer from the file it was
    given, and the honest way to enforce that is to leave it nothing else to
    route around the promise with. ``session`` adds local_search, which reaches
    the rest of this session's files. ``expanded`` adds the web.
    """
    mode = mode if mode in SEARCH_MODES else DEFAULT_MODE
    if mode == "expanded":
        return list(names)
    if mode == "session":
        # local_search is IN; it is the tier-1 tool. Web stays out.
        return [n for n in names if n not in _WEB_TOOLS]
    return [n for n in names if n not in _WEB_TOOLS and n != "local_search"]


def mode_prompt(mode: str, uploads: list[dict] | None = None) -> str:
    """The sentence the model is told about what it may answer from."""
    names = ", ".join(u["name"] for u in (uploads or []))
    if mode == "strict":
        src = names or "the file you were given"
        return (
            f"The user asked you to answer ONLY from {src}. Everything you say "
            f"about the subject must come from that file. If the file does not "
            f"contain the answer, say plainly that it is not in {src} -- do not "
            f"guess, and do not offer what you know from elsewhere. Then ask "
            f"whether they would like you to look further: the rest of this "
            f"session's files first, and the web after that."
        )
    if mode == "session":
        return (
            "Answer from the files in this session first. If they do not "
            "contain what was asked, say so and ask whether to search the web."
        )
    return (
        "You may search this session's files and the web. Prefer the session's "
        "files when they cover the question."
    )


# Phrases that change the tier. Deliberately narrow and deterministic: this is a
# permission change, so it is matched in the harness and not inferred by the
# model. A wrong match here is a promise broken silently, which is worse than
# the user having to repeat themselves.
_STRICT_PHRASES = (
    "only that file", "only from that file", "just that file", "just the file",
    "only the file", "only this file", "only from this file",
    "that file only", "this file only", "only use the file",
)
_EXPANDED_PHRASES = (
    "search the web", "yes search", "go ahead and search", "you can search",
    "use the web", "check online", "look online", "expand the search",
    "search online", "yes please search", "search for it",
)
_SESSION_PHRASES = (
    "other files", "the session", "this session", "session files",
    "use the files", "yes use the files", "check the files",
)


def mode_from_text(text: str) -> str | None:
    """The tier a spoken or typed answer selects, or None if it selects none.

    Expanded is checked first: "yes, search the web" contains no strict phrase,
    but "only search the web" does, and the wider intent is the one the user
    said last and most explicitly. A refusal is not a mode change either --
    "no" leaves the tier alone rather than silently narrowing it.
    """
    low = " ".join((text or "").lower().split())
    if not low:
        return None
    if any(p in low for p in _EXPANDED_PHRASES):
        return "expanded"
    if any(p in low for p in _STRICT_PHRASES):
        return "strict"
    if any(p in low for p in _SESSION_PHRASES):
        return "session"
    return None
