"""The upload, and the three tiers it selects between.

Phase 5 + 6. Three claims:

  * an upload lands in the session's work/uploads/, exactly once, under a name
    that cannot escape the folder, and is refused (with a sentence a person can
    hear) when it is too big or the wrong type;
  * the agent turns the upload into a spoken question rather than ingesting it
    silently;
  * the answer to that question is ENFORCED -- strict really does remove the
    tools that could reach past the file, and the wider tiers really do put
    them back.

Run:  .venv/bin/python test_upload.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent
_TMP = Path(tempfile.mkdtemp(prefix="fw_upload_"))
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


d = session.create()
sid = d.name
work = session.work_dir(sid)

# ---------------------------------------------------------------------------
# Storing
# ---------------------------------------------------------------------------
print("\nstore")
p = session.store_upload(sid, "report.pdf", b"%PDF-1.4 hello")
check("the file lands in work/uploads/",
      p.parent == session.uploads_dir(sid) == work / "uploads", str(p.parent))
check("the bytes are intact", p.read_bytes() == b"%PDF-1.4 hello")
check("the upload is inside the tool jail",
      str(p).startswith(str(work) + os.sep))
check("meta.json records it",
      json.loads((d / "meta.json").read_text())["uploads"][0]["name"] == "report.pdf")
check("the notes record it", "uploaded: report.pdf"
      in session.notes_path(sid).read_text())

# A second file with the same name must not replace the first.
p2 = session.store_upload(sid, "report.pdf", b"%PDF-1.4 second")
check("a same-named upload is not overwritten",
      p2.name != p.name and p.read_bytes() == b"%PDF-1.4 hello",
      f"{p.name} / {p2.name}")
check("the second one is numbered", p2.name == "report-2.pdf", p2.name)

# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------
print("\nrefusals")
for bad, why in [("evil.exe", "disallowed type"),
                 ("script.sh", None)]:
    if why is None:
        continue
    try:
        session.store_upload(sid, bad, b"x")
    except ValueError as e:
        check(f"{why} is refused", "not a file type" in str(e), str(e))
    else:
        check(f"{why} is refused", False)

try:
    session.store_upload(sid, "empty.txt", b"")
except ValueError as e:
    check("an empty file is refused", "empty" in str(e), str(e))
else:
    check("an empty file is refused", False)

big = b"x" * (session.UPLOAD_MAX_BYTES + 1)
try:
    session.store_upload(sid, "big.txt", big)
except ValueError as e:
    check("an oversize file is refused", "limit is" in str(e), str(e))
else:
    check("an oversize file is refused", False)

# Traversal: the browser supplies this name, so it is hostile by assumption.
for hostile in ("../../etc/passwd.txt", "/etc/passwd.txt", "..\\..\\win.md",
                "a/b/c/notes.md"):
    try:
        safe = session.safe_upload_name(hostile)
    except ValueError as e:
        check(f"refuses {hostile!r}", False, str(e))
        continue
    check(f"{hostile!r} becomes one path segment",
          "/" not in safe and "\\" not in safe and not safe.startswith("."),
          safe)

try:
    session.store_upload(sid, "../../escape.txt", b"x")
except ValueError:
    check("a traversal name is refused outright", True)
else:
    landed = list((work / "uploads").glob("*escape*"))
    check("a traversal name lands inside uploads/",
          bool(landed) and all(p.parent == work / "uploads" for p in landed),
          str(landed))
check("nothing was written outside the session",
      not (_TMP / "escape.txt").exists() and not (_TMP.parent / "escape.txt").exists())

# ---------------------------------------------------------------------------
# The spoken gate
# ---------------------------------------------------------------------------
print("\nthe spoken question")
src = (REPO / "agent.py").read_text()
check("the agent has an upload handler", "async def _handle_upload" in src)
check("the handler asks rather than ingests",
      "Should I answer only from that file" in src)
check("the question is spoken as a literal",
      "TTSSpeakFrame(text=prompt)" in src)
check("the upload is watched like the other controls",
      "self._upload_path" in src and "config.DASHBOARD_UPLOAD_FILE" in src)
check("the upload event is recorded", 'emit("upload"' in src)

# ---------------------------------------------------------------------------
# The tiers
# ---------------------------------------------------------------------------
print("\ntiers")
tools = ["read_file", "write_file", "bash", "search_document", "index_documents",
         "local_search", "search", "fetch_content", "get_weather", "github_repo"]

strict = session.allowed_tool_names("strict", tools)
ses = session.allowed_tool_names("session", tools)
exp = session.allowed_tool_names("expanded", tools)

check("strict keeps the file tools", "read_file" in strict and "bash" in strict)
check("strict drops local_search", "local_search" not in strict, str(strict))
check("strict drops the web", not ({"search", "fetch_content", "get_weather"}
                                   & set(strict)), str(strict))
check("session keeps local_search", "local_search" in ses, str(ses))
check("session drops the web", not ({"search", "fetch_content"} & set(ses)), str(ses))
check("expanded keeps everything", exp == tools, str(exp))
check("strict is a subset of session", set(strict) <= set(ses))
check("session is a subset of expanded", set(ses) <= set(exp))

# The mode is remembered per session, not globally.
check("a new session starts at the default", session.get_mode(sid) == "session",
      session.get_mode(sid))
session.set_mode(sid, "strict")
check("the mode is written to meta.json", session.get_mode(sid) == "strict")
other = session.create()
check("the mode does not leak to another session",
      session.get_mode(other.name) == "session", session.get_mode(other.name))
try:
    session.set_mode(sid, "everything")
except ValueError:
    check("an unknown mode is refused", True)
else:
    check("an unknown mode is refused", False)

# The prompt says what to say when the answer is not in the file.
strict_prompt = session.mode_prompt("strict", [{"name": "report.pdf"}])
check("the strict prompt names the file", "report.pdf" in strict_prompt)
check("the strict prompt says to admit a gap",
      "not in report.pdf" in strict_prompt, strict_prompt[:200])
check("the strict prompt offers to expand",
      "look further" in strict_prompt)
check("the strict prompt forbids answering from elsewhere",
      "do not guess" in strict_prompt)

# The spoken answer selects a tier, deterministically.
cases = {
    "only that file": "strict",
    "just the file please": "strict",
    "yes, search the web": "expanded",
    "go ahead and search online": "expanded",
    "use the other files in this session": "session",
    "no": None,
    "what is the population of Lisbon": None,
}
for text, want in cases.items():
    got = session.mode_from_text(text)
    check(f"{text!r} -> {want}", got == want, f"got {got}")

# ---------------------------------------------------------------------------
# Enforcement, on a real LLMContext
# ---------------------------------------------------------------------------
print("\nenforcement")
try:
    from pipecat.processors.aggregators.llm_context import LLMContext
    import agent as agent_mod
    import mcp_bridge

    # Real tool handlers, built the way the bridge builds them. A stub class is
    # not enough: Pipecat's DirectFunctionWrapper introspects the callable's
    # signature, so the tools have to be actual functions or the context
    # refuses them outright ("Unsupported tool type").
    class _Srv:
        name = "stub"

        async def call_tool(self, name, payload):
            return type("R", (), {"content": []})()

    bridge = mcp_bridge.ToolBridgeProcessor(servers=[])
    real = []
    for name in tools:
        props = {"data_path": {"type": "string"}, "path": {"type": "string"}}
        schema = type("T", (), {"name": name, "description": f"{name} tool",
                                "input_schema": {"type": "object",
                                                 "properties": props,
                                                 "required": []}})()
        real.append(bridge._make_tool(_Srv(), schema))

    class _LLM:
        def __init__(self):
            self._settings = type("S", (), {"system_instruction": "base prompt"})()
            self._base_system_instruction = "base prompt"

    ctx = LLMContext(messages=[], tools=real)
    llm = _LLM()
    check("the context starts with every tool",
          len(ctx.tools.direct_functions) == len(tools),
          str(len(ctx.tools.direct_functions)))

    # The bridge's list is the full set; apply_search_mode filters from it, so
    # give the stub bridge the real handlers.
    bridge.tools = list(real)

    agent_mod.apply_search_mode(llm, ctx, bridge, "strict")
    names = [t.function.__name__ for t in ctx.tools.direct_functions]
    check("strict really removes the web tools from the context",
          "search" not in names and "fetch_content" not in names, str(names))
    check("strict really removes local_search from the context",
          "local_search" not in names, str(names))
    check("strict keeps read_file", "read_file" in names, str(names))
    check("strict keeps the handlers (not just the schemas)",
          all(callable(t.function) for t in ctx.tools.direct_functions))
    check("the system prompt gained the scope sentence",
          "Answering scope:" in llm._settings.system_instruction,
          llm._settings.system_instruction[-200:])

    agent_mod.apply_search_mode(llm, ctx, bridge, "expanded")
    names = [t.function.__name__ for t in ctx.tools.direct_functions]
    check("expanded puts the web tools back", "search" in names, str(names))
    check("expanded puts local_search back", "local_search" in names, str(names))
    check("applying twice does not stack the scope sentence",
          llm._settings.system_instruction.count("Answering scope:") == 1,
          str(llm._settings.system_instruction.count("Answering scope:")))
    check("the base prompt survives",
          llm._settings.system_instruction.startswith("base prompt"),
          llm._settings.system_instruction[:40])

    agent_mod.apply_search_mode(llm, ctx, bridge, "session")
    names = [t.function.__name__ for t in ctx.tools.direct_functions]
    check("session restores local_search", "local_search" in names)
    check("session keeps the web out", "search" not in names, str(names))

    # The narrowing has to be real on the wire, not just in the object: the
    # adapter reads the context's tools when it builds each request.
    from pipecat.adapters.llm.schemas.tools_schema import ToolsSchema
    check("the context still holds a ToolsSchema",
          isinstance(ctx.tools, ToolsSchema), type(ctx.tools).__name__)
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check("enforcement runs", False, repr(e))

# ---------------------------------------------------------------------------
# The dashboard side
# ---------------------------------------------------------------------------
print("\ndashboard")
dash = (REPO / "dashboard.py").read_text()
check("the placeholder button is gone", 'disabled=True' not in dash
      or 'help="Attach a file (coming soon)"' not in dash)
check("the attach key is kept for the layout test", "attachph" in dash)
check("the uploader is present", "st.file_uploader" in dash)
check("the uploader is limited to the allowed types",
      "UPLOAD_EXTENSIONS" in dash)
check("the dashboard stores into the session", "session.store_upload" in dash)
check("the dashboard hands the agent a notification",
      "_drop(UPLOAD_FILE" in dash)
check("an upload is stored once", "_SEEN_UPLOADS" in dash)
check("a refusal is shown, not swallowed", "st.error(err)" in dash)

print()
if FAILS:
    print(f"FAILED ({len(FAILS)}): " + ", ".join(FAILS))
    sys.exit(1)
print("all upload/tier checks passed")
