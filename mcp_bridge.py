"""Fweddie -- OnIt MCP tool bridge.

Spawns OnIt's own stdio MCP servers and exposes their tools to the LLM as
Pipecat function calls. Fweddie does NOT re-implement OnIt's tools or its
safeguards: it talks to the real OnIt servers over the MCP protocol, so the
AST command allowlist, the approval gate, and the DATA_PATH file jail all run
inside the spawned server process and apply to every call Fweddie makes.

    Fweddie LLM  --function call-->  ToolBridgeProcessor  --MCP-->  OnIt server (stdio)
                                                                              |
                                                                          bash / read_file /
                                                                          search / weather ...
                                                                          (policy + jail here)

Run the servers with OnIt's interpreter (it has fastmcp + mcp), from OnIt's
package root so `src.*` imports resolve. The MCP client lives in Fweddie's own
venv.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import tempfile
import time
from typing import Any, Callable, Optional

from loguru import logger

import config
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.services.llm_service import FunctionCallParams


# ---------------------------------------------------------------------------
# One OnIt stdio server
# ---------------------------------------------------------------------------
class OnItMCPServer:
    """A single OnIt stdio MCP server: a subprocess Fweddie owns.

    Spawned with OnIt's interpreter running `python -m src.mcp.servers.run
    --stdio ...`. The MCP session is created lazily on first use (it needs an
    event loop, which only exists once the Pipecat pipeline is running).
    """

    def __init__(self, name: str, module: str, profile: Optional[str] = None):
        self.name = name
        self.module = module
        self.profile = profile
        self._session: Optional[ClientSession] = None
        self._tools: Optional[list] = None
        self._lock = asyncio.Lock()

    def _params(self) -> StdioServerParameters:
        args = [
            "-m", "src.mcp.servers.run",
            "--stdio",
            "--name", self.name,
            "--module", self.module,
            "--log-level", "INFO",
        ]
        if self.profile:
            args += ["--profile", self.profile]

        # Inherit the ambient environment, then pin the OnIt knobs the server
        # reads at import time. ONIT_DATA_PATH is the file jail root; the
        # allowlist stays on its default (allowlisted commands run, anything
        # else is refused) -- Fweddie never loosens it.
        #
        # ONIT_APPROVAL_CHANNEL=1 tells the server a person is reachable, so a
        # command the allowlist will not decide alone returns a needs_approval
        # ticket (which the Guard hands to a human) instead of collapsing to a
        # hard refusal. Without it every ask is a deny -- the fail-closed
        # default for a server with no one to ask.
        #
        # ONIT_COMMAND_ALLOWLIST=1 turns the AST command allowlist on. On a
        # host (outside the onit container) it is off by default, which would
        # let Fweddie run any executable -- the allowlist is the whole point of
        # the safeguard, so we pin it on.
        spawn_env = dict(os.environ)
        spawn_env["ONIT_DATA_PATH"] = config.MCP_DATA_PATH
        if config.MCP_DOCUMENTS_PATH:
            spawn_env["DOCUMENTS_PATH"] = config.MCP_DOCUMENTS_PATH
        spawn_env["ONIT_STDIO_SPAWNED"] = "1"
        spawn_env["ONIT_APPROVAL_CHANNEL"] = "1"
        spawn_env["ONIT_COMMAND_ALLOWLIST"] = "1"

        # Tool secrets (OpenWeather, Tavily, GitHub, HuggingFace). The server
        # is a separate process and cannot read the OS keychain, so the keys
        # must travel in its environment. Resolved per server at spawn time --
        # secrets.yaml first, then the keychain -- and scoped to this
        # subprocess only: Fweddie's own process never gets them.
        #
        # An ambient env var still wins, because we only fill in keys that are
        # not already present. That keeps an explicit `export` in control.
        for var, value in config.tool_secret_env().items():
            spawn_env.setdefault(var, value)

        return StdioServerParameters(
            command=config.ONIT_PYTHON,
            args=args,
            env=spawn_env,
            cwd=config.ONIT_REPO,
        )

    async def _ensure_session(self) -> ClientSession:
        if self._session is not None:
            return self._session
        async with self._lock:
            if self._session is not None:
                return self._session
            # stdio_client is an async CM that owns the subprocess; we hold it
            # open for the life of the process.
            cm = stdio_client(self._params())
            read, write = await cm.__aenter__()
            session = ClientSession(read, write)
            await session.__aenter__()
            await session.initialize()
            self._cm = cm
            self._session = session
            logger.info("MCP server connected: {}", self.name)
            return session

    async def list_tools(self) -> list:
        """Return the server's tools (cached). Empty on failure -- a server
        that will not start simply contributes no tools, it does not take the
        voice loop down with it."""
        if self._tools is not None:
            return self._tools
        try:
            session = await self._ensure_session()
            result = await session.list_tools()
            self._tools = list(result.tools)
            logger.info("MCP server {} offers {} tools", self.name, len(self._tools))
        except Exception as e:  # noqa: BLE001 - one bad server must not kill the rest
            logger.error("MCP server {} failed to list tools: {}", self.name, e)
            self._tools = []
        return self._tools

    async def call_tool(self, name: str, arguments: dict) -> str:
        """Call a tool and return its text result. Refusals come back as text
        (the server's policy message), not as exceptions, so the LLM can hear
        about them and explain them to the user."""
        session = await self._ensure_session()
        result = await session.call_tool(name, arguments)
        return _result_to_text(result)

    async def close(self) -> None:
        if self._session is not None:
            try:
                await self._session.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._session = None
        cm = getattr(self, "_cm", None)
        if cm is not None:
            try:
                await cm.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._cm = None


def _json_type_to_python(pschema: Any) -> Any:
    """Map a JSON-schema property to a Python type for the signature hint.

    Pipecat's `_typehint_to_jsonschema` understands the basics (str/int/float/
    bool/list/dict/Optional/Union), so we only need to produce those. Anything
    exotic falls back to `str`, which is safe for a voice agent's tool calls.
    """
    if not isinstance(pschema, dict):
        return str
    # anyOf: [string, null] -> Optional[str]; [string, integer] -> Union[...]
    anyof = pschema.get("anyOf")
    if isinstance(anyof, list) and anyof:
        subs = [_json_type_to_python(p) for p in anyof]
        has_null = any(s is type(None) for s in subs)
        non_null = [s for s in subs if s is not type(None)]
        if has_null:
            if len(non_null) == 1:
                return Optional[non_null[0]]
            return Union[tuple(non_null), type(None)]
        if len(non_null) == 1:
            return non_null[0]
        return Union[tuple(non_null)]
    t = pschema.get("type")
    if t == "string":
        return str
    if t == "integer":
        return int
    if t == "number":
        return float
    if t == "boolean":
        return bool
    if t == "array":
        return list
    if t == "object":
        return dict
    if t == "null":
        return type(None)
    return str


def _json_type_default(ann: Any) -> Any:
    """A sensible default for an optional (non-required) parameter."""
    if ann is str:
        return ""
    if ann is int:
        return 0
    if ann is float:
        return 0.0
    if ann is bool:
        return False
    if ann is list:
        return []
    if ann is dict:
        return {}
    if ann is type(None):
        return None
    # Optional[X] / Union -> None is always acceptable.
    return None


def _result_to_text(result: Any) -> str:
    """Flatten an MCP CallToolResult to plain text for the LLM."""
    parts: list[str] = []
    for content in getattr(result, "content", []) or []:
        ctype = getattr(content, "type", "")
        if ctype == "text":
            parts.append(getattr(content, "text", "") or "")
        elif ctype == "image":
            parts.append("[image returned]")
        else:
            parts.append(str(content))
    text = "\n".join(p for p in parts if p).strip()
    if getattr(result, "isError", False):
        return f"[tool error] {text or 'the tool reported an error'}"
    return text or "(no output)"


def _short(value: Any, limit: int = 600) -> str:
    """Render a tool arg/result for the dashboard, capped so one huge file
    read cannot bloat the event log."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except Exception:  # noqa: BLE001
            text = str(value)
    text = " ".join(text.split())  # collapse newlines/whitespace to one line
    return text if len(text) <= limit else text[:limit] + " …"


# What the agent says before a tool runs. Keyed by tool name so the phrase
# matches the act; anything unlisted gets a neutral line. These are spoken
# verbatim, so they are short, lowercase-friendly, and free of punctuation the
# TTS would read out.
_ANNOUNCEMENTS = {
    "search": "Searching for that.",
    "local_search": "Checking my notes.",
    "search_document": "Looking through the documents.",
    "index_documents": "Indexing the documents.",
    "fetch_content": "Fetching that page.",
    "get_weather": "Checking the weather.",
    "bash": "Running that command.",
    "serve": "Starting that process.",
    "read_file": "Reading the file.",
    "write_file": "Writing the file.",
    "edit_file": "Editing the file.",
    "grep": "Searching the files.",
    "github_repo": "Checking the repository.",
    "send_file": "Sending the file.",
}


def _announcement(tool_name: str) -> str:
    """A short spoken line to say before ``tool_name`` runs."""
    return _ANNOUNCEMENTS.get(tool_name, "One moment.")


# Words that mean the answer is time-sensitive. Used only to fill in a missing
# `type` argument on a `search` call -- never to override one the model set.
# Deliberately narrow: "news", "latest", "today" and the like. A false positive
# costs one dated search instead of an undated one, which is the cheaper error.
_RECENCY_WORDS = (
    "news", "latest", "today", "tonight", "yesterday", "this week",
    "this morning", "right now", "currently", "current", "recent",
    "recently", "breaking", "headline", "headlines", "update", "updates",
    "who won", "election", "score", "scores", "price", "prices", "stock",
    "stocks", "weather", "forecast", "just happened", "newest",
)


def _looks_time_sensitive(query: str) -> bool:
    """Does this query need dated results?"""
    q = " ".join((query or "").lower().split())
    if not q:
        return False
    return any(w in q for w in _RECENCY_WORDS)


def _shape_search_args(args: dict) -> dict:
    """Fill in ``type="news"`` when the query is clearly about recency.

    The model has the tool description in front of it and still passed no
    `type` on 13 out of 13 real searches, so every one hit the general index,
    which returns no dates at all. This is the deterministic half of the fix:
    an argument the model omitted is supplied here.

    Note which case this can and cannot see. The tool schema declares
    ``type: {default: "web"}``, and Pipecat fills defaults in from the
    signature, so by the time the handler runs, ``args["type"] == "web"``
    whether the model chose it or said nothing at all. Those two are
    indistinguishable here, so "web" is treated as "unspecified" and a
    recency-shaped query is upgraded. Only an explicit non-web choice --
    ``"news"`` itself, or anything else the server later adds -- is left alone.
    """
    if not config.MCP_NEWS_AUTODETECT:
        return args
    if not isinstance(args, dict):
        return args
    if "type" in args and args.get("type") not in (None, "", "web"):
        return args          # an explicit, non-default choice; respect it
    if not _looks_time_sensitive(args.get("query") or ""):
        return args
    shaped = dict(args)
    shaped["type"] = "news"
    # Five is the news branch's useful ceiling for a spoken answer; a bigger
    # number only pushes more of the payload past the model's cap.
    try:
        if int(shaped.get("max_results") or 0) < 3:
            shaped["max_results"] = 5
    except (TypeError, ValueError):
        shaped["max_results"] = 5
    return shaped


def _search_failed(text: str) -> bool:
    """Did a `search` call come back as an error object?"""
    if not isinstance(text, str):
        return False
    stripped = text.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped)
    except (ValueError, TypeError):
        return False
    return isinstance(data, dict) and bool(data.get("error"))


def _cap_for_model(text: str, limit: int, query: Optional[str] = None) -> str:
    """Return the part of a tool result the LLM is allowed to see.

    The voice reply is one or two sentences, so the model needs the head of the
    result, not all of it -- but it needs a *parseable* head. A blind
    ``text[:limit]`` slice lands mid-string, so a 5-result news payload came
    back as one title plus a dangling fragment and the model could not see the
    ``date`` field of any result. That is the difference between "the search
    gave me old news" and "the search gave me nothing I can read".

    So the cap is structural where the result is a JSON list (which is what
    every OnIt search returns): whole results are kept, the count of dropped
    results is stated, and the text stays valid JSON. Anything else -- plain
    text, a dict, a malformed payload -- falls back to the old head-slice,
    which is right for a file read or a shell transcript.

    ``query`` is the search that produced the result. When a result has to be
    shortened, the passage matching the question is kept instead of only the
    opening -- see ``_cap_best_window``.

    The full text is untouched: it goes to the dashboard and the session
    folder. ``limit <= 0`` disables the cap.
    """
    if limit <= 0 or len(text) <= limit:
        return text

    capped = _cap_json_list(text, limit, query)
    if capped is not None:
        return capped

    return (text[:limit].rstrip()
            + f"\n[truncated: showing {limit} of {len(text)} chars]")


# Fields that stay whole however tight the budget gets. A voice model needs the
# url to cite a result and the date to tell old news from new; cutting either
# costs it the very thing it searched for, and neither is the bulky field.
_CAP_KEEP_INTACT = frozenset({
    "url", "link", "href", "title", "date", "published", "published_date",
    "source", "id", "name", "type", "error",
})

# How much prose to keep per result, tried in order. ``None`` means "try whole
# results first", so a payload that already fits is passed through untouched:
# news results are short and must not be abbreviated needlessly.
_CAP_CLIP_LADDER = (None, 700, 500, 350, 250, 180, 120, 80, 50)

# Words too common to identify the passage that answers a question.
_CAP_STOPWORDS = frozenset(
    "a an the is are was were be been being what when where who whom whose why "
    "how does do did doing of in on at to for from by with about into over after "
    "and or but if then than that this these those it its i you we they he she "
    "me my your our their there here can could should would will shall may might "
    "please tell give show find latest new news".split()
)

# Share of the budget given to the opening of a passage before the window that
# matches the question starts. A page's first lines are its title and nav; the
# answer is usually a paragraph or two in.
_CAP_HEAD_SHARE = 0.4


def _cap_terms(query: Optional[str]) -> list:
    """The content words of a query, for locating the passage that answers it."""
    if not query:
        return []
    words = re.findall(r"[a-z0-9]+", str(query).lower())
    return [w for w in words if len(w) > 1 and w not in _CAP_STOPWORDS]


def _cap_best_window(text: str, terms: list, size: int):
    """Start of the ``size``-char window holding the most query terms.

    A prefix is the wrong shape for a web result: the first 250 chars of a CES
    page are "About CES: CES is the most powerful tech event in the world...",
    while "CES 2025 takes place Jan. 7-10, 2025, in Las Vegas" sits at char 388.
    The model was given the boilerplate and asked again. So the clip keeps the
    opening *and* jumps to the densest passage for the question's own words.
    """
    if not terms or len(text) <= size:
        return 0
    low = text.lower()
    hits = sorted(m.start() for t in terms for m in re.finditer(re.escape(t), low))
    if not hits:
        return 0
    best, best_hits = hits[0], -1
    for h in hits:
        n = sum(1 for x in hits if h <= x < h + size)
        if n > best_hits:
            best_hits, best = n, h
    # Start a little before the first matching term so the sentence is whole.
    return max(0, min(best - size // 4, len(text) - size))


def _cap_clip(value, clip: int, terms: Optional[list] = None):
    """Shrink the prose inside a result, leaving identity fields whole."""
    if isinstance(value, str):
        if len(value) <= clip:
            return value
        if terms:
            head = max(40, int(clip * _CAP_HEAD_SHARE))
            start = _cap_best_window(value, terms, clip - head)
            # Only jump when the window is actually past the head; otherwise the
            # two pieces would overlap and repeat the same sentence.
            if start > head:
                return (value[:head].rstrip() + " \u2026 "
                        + value[start:start + clip - head].strip() + "\u2026")
        return value[:clip].rstrip() + "\u2026"
    if isinstance(value, list):
        return [_cap_clip(v, clip, terms) for v in value]
    if isinstance(value, dict):
        return {k: (v if k in _CAP_KEEP_INTACT else _cap_clip(v, clip, terms))
                for k, v in value.items()}
    return value


def _cap_json_list(text: str, limit: int, query: Optional[str] = None) -> Optional:
    """Fit a JSON list into ``limit`` chars without ever emptying it.

    Two stages, because there are two different reasons a payload is too big:

    1. **Too many results.** Drop whole results from the tail and say how many
       were dropped. This is the news shape -- a handful of short, dated
       results -- and it is the only stage the cap used to have.
    2. **Each result is too big.** A web search returns full page bodies
       (~1000-1600 chars each), so five results is 5400-7800 chars and *not
       one* fits under the cap. Stage 1 alone therefore dropped all five and
       handed the model ``{"results": [], "note": "all 5 results were too large
       to show"}`` -- an empty answer to a question the server had answered.
       That is why the model re-asked "CES 2025 dates" four times: it was
       reading an empty list, not a stale one.

    Stage 2 clips the prose inside each result while keeping titles, urls,
    dates and sources intact, and clips around the passage that matches
    ``query`` rather than blindly from the front. The model gets at least one
    result whenever the server returned one, and the note says the text was cut
    so it knows the snippet is abbreviated rather than whole.
    """
    stripped = text.strip()
    if not stripped.startswith("["):
        return None
    try:
        items = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if not isinstance(items, list) or not items:
        return None

    total = len(items)
    terms = _cap_terms(query)

    def render(keep: int, clip: Optional[int]) -> str:
        if keep >= total and clip is None:
            return json.dumps(items, ensure_ascii=False)
        shown = items[:keep]
        if clip is not None:
            shown = [_cap_clip(item, clip, terms) for item in shown]
        note = f"showing {keep} of {total} results"
        if keep < total:
            note += f"; {total - keep} dropped to fit the model's context"
        if clip is not None:
            note += (f"; each result's text is cut to {clip} chars, while its "
                     f"title, url and date are intact")
        # The trailer is deliberately plain: a voice model reads it as a
        # sentence ("showing 2 of 5 results"), not as another result object.
        return json.dumps(
            {"results": shown, "showing": keep, "total": total, "note": note},
            ensure_ascii=False,
        )

    # Items vary in size, so walk down from the full list rather than assuming
    # an average. Collect every (clip, keep) that fits instead of taking the
    # first one: the first fit is always the *longest* text, which means one
    # result quoted at 700 chars rather than three at 250. For a voice answer
    # that is the wrong trade -- two independent sources beat one long one --
    # so the choice is made explicitly below.
    options = []
    for clip in _CAP_CLIP_LADDER:
        for keep in range(total, 0, -1):
            out = render(keep, clip)
            if len(out) <= limit:
                options.append((keep, clip, out))
                break

    if options:
        # Prefer a configuration that keeps at least two results, and among
        # those prefer the least clipping (largest clip, then most results).
        # A news payload that already fits is chosen here untouched, because
        # clip=None is the largest clip and it keeps its usual 2-3 results.
        def rank(o):
            return (o[1] if o[1] is not None else 10 ** 6, o[0])

        many = [o for o in options if o[0] >= 2]
        return max(many or options, key=rank)[2]

    # Even one result clipped to the shortest step will not fit, which means
    # the identity fields themselves (a long url, a long title) blow the budget.
    # Give the model the first result's skeleton rather than an empty list.
    skeleton = [
        {k: v for k, v in item.items() if k in _CAP_KEEP_INTACT}
        if isinstance(item, dict) else item
        for item in items[:1]
    ]
    out = json.dumps(
        {"results": skeleton, "showing": 1, "total": total,
         "note": f"only the first of {total} results fits within {limit} chars, "
                 f"and only its title, url, date and source; the body text was "
                 f"dropped"},
        ensure_ascii=False,
    )
    if len(out) <= limit:
        return out
    # A single result's identity alone is over budget (a very long url). Say so
    # with a valid answer that still states a result existed.
    return json.dumps(
        {"results": [], "showing": 0, "total": total,
         "note": f"the first result alone is larger than {limit} chars (its url "
                 f"and title), so no result could be shown"},
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# The approval guard: human-in-the-loop for commands OnIt's policy will not
# decide alone
# ---------------------------------------------------------------------------
# OnIt's bash/serve tools gate commands through an AST allowlist. A command the
# allowlist does not cover is not refused outright: the server returns a
# ``needs_approval`` payload (a single-use ticket bound to that exact command)
# and waits for the harness to re-issue the call carrying the ticket. The
# harness -- that is us -- is responsible for showing a person the command and
# deciding whether to redeem (run it) or refuse (burn it).
#
# This class is that harness side. It mirrors OnIt's own approvals.py:
#   * one pending ticket at a time (a voice agent asks one question at a time)
#   * the ticket is single-use and bound to the exact command
#   * a refusal is remembered for the session so the model cannot retry the
#     same command
# The actual policy decision (allow / ask / deny) still happens inside the
# spawned OnIt server -- the guard only handles the human half.
class Guard:
    """Human-in-the-loop approval for OnIt's ``needs_approval`` payloads."""

    def __init__(self, prompt: Optional[Callable[[dict], str]] = None,
                 timeout: Optional[float] = None,
                 bus: Any = None):
        # ``prompt`` shows the person the pending command and returns their
        # answer: "once" (run it this time), "session" (run it and remember),
        # or "deny". Defaults to an interactive stdin prompt.
        self.prompt = prompt or self._stdin_prompt
        self.timeout = timeout if timeout is not None else config.MCP_APPROVAL_TIMEOUT
        # Optional event bus (eventbus.EventBus). When set, every tool call and
        # approval decision is recorded so the dashboard can show it live.
        self.bus = bus
        self._pending: Optional[dict] = None
        self._refused: set[str] = set()   # command digests a person said no to
        self._granted: set[str] = set()   # command digests approved for the session

    # -- the human half ------------------------------------------------------
    @staticmethod
    def _stdin_prompt(payload: dict) -> str:
        """Interactive fallback: ask on the console. Returns once/session/deny."""
        print("\n" + "=" * 62, file=sys.stderr)
        print("  OnIt needs a person's approval before it can run this:",
              file=sys.stderr)
        print(f"  command : {payload.get('command', '')}", file=sys.stderr)
        if payload.get("reason"):
            print(f"  reason  : {payload['reason']}", file=sys.stderr)
        if payload.get("subjects"):
            print(f"  about   : {', '.join(map(str, payload['subjects']))}",
                  file=sys.stderr)
        print("  [o]nce  [s]ession  [d]eny", file=sys.stderr)
        print("=" * 62, file=sys.stderr)
        while True:
            try:
                ans = input("  approve? ").strip().lower()
            except EOFError:
                return "deny"
            if ans in ("o", "once", "y", "yes"):
                return "once"
            if ans in ("s", "session"):
                return "session"
            if ans in ("d", "deny", "n", "no"):
                return "deny"
            print("  (o)nce / (s)ession / (d)eny", file=sys.stderr)

    # -- the ticket half -----------------------------------------------------
    @staticmethod
    def _digest(command: str) -> str:
        import hashlib
        return hashlib.sha256((command or "").encode("utf-8")).hexdigest()

    def is_refused(self, command: str) -> bool:
        return self._digest(command) in self._refused

    def is_granted(self, command: str) -> bool:
        return self._digest(command) in self._granted

    async def handle(self, tool_name: str, args: dict, server,
                     call: Callable[..., Any]) -> str:
        """Run a tool call through the approval gate.

        ``call`` is the plain ``server.call_tool`` (returns a text string).
        This method:
          1. calls the tool;
          2. if the result is a ``needs_approval`` payload, asks the person and
             re-issues the call carrying the ticket + the person's scope;
          3. records refusals so the same command is not re-asked.
        Returns the final text to hand to the LLM.
        """
        # Record the call so the dashboard can show what the agent is doing.
        if self.bus is not None:
            await self.bus.emit("tool_call", tool=tool_name,
                                args=_short(args), server=server.name)

        result = await call(tool_name, args)
        payload = self._parse_approval(result)
        if payload is None:
            if self.bus is not None:
                await self.bus.emit("tool_result", tool=tool_name,
                                    result=_short(result))
            return result

        # A person already said no to this exact command this session.
        if self.is_refused(payload.get("command", "")):
            if self.bus is not None:
                await self.bus.emit("approval_denied", tool=tool_name,
                                    command=payload.get("command", ""),
                                    reason="already declined this session")
            return ("[denied] A person already declined this command. Do not "
                    "retry it or try another route around the policy.")

        self._pending = payload
        if self.bus is not None:
            self.bus.set_pending_approval(payload)
            self.bus.write_pending(payload)
            await self.bus.emit("approval_requested", tool=tool_name,
                                command=payload.get("command", ""),
                                reason=payload.get("reason", ""))

        # Ask a person. Prefer the dashboard (a person on the LAN can click a
        # button); fall back to the console prompt if the dashboard is down.
        decision: Optional[str] = None
        if self.bus is not None:
            decision = await self.bus.request_approval(payload)
        if decision is None:
            # May block on stdin; run in a thread so we don't stall the loop.
            decision = await asyncio.to_thread(self.prompt, payload)
        self._pending = None
        if self.bus is not None:
            self.bus.clear_pending_approval()

        if decision == "deny":
            self._refused.add(self._digest(payload.get("command", "")))
            if self.bus is not None:
                await self.bus.emit("approval_denied", tool=tool_name,
                                    command=payload.get("command", ""))
            return ("[denied] A person declined to approve this command. It "
                    "was not run. Do not retry it or try another route around "
                    "the policy.")
        if decision == "session":
            self._granted.add(self._digest(payload.get("command", "")))
        if self.bus is not None:
            await self.bus.emit("approval_granted", tool=tool_name,
                                command=payload.get("command", ""),
                                scope=decision)

        # Re-issue the call carrying the single-use ticket + the person's scope.
        retry_args = dict(args)
        retry_args["approval_token"] = payload.get("approval_id", "")
        retry_args["approval_scope"] = decision  # "once" or "session"
        result2 = await call(tool_name, retry_args)
        # If it still asks (ticket raced / policy changed), surface it plainly
        # rather than looping.
        if self._parse_approval(result2) is not None:
            if self.bus is not None:
                await self.bus.emit("tool_result", tool=tool_name,
                                    result="[denied] approval could not be redeemed")
            return ("[denied] The approval could not be redeemed. The command "
                    "was not run.")
        if self.bus is not None:
            await self.bus.emit("tool_result", tool=tool_name,
                                result=_short(result2))
        return result2

    @staticmethod
    def _parse_approval(text: Any) -> Optional[dict]:
        """Return the ``needs_approval`` payload if the tool result carries one.

        ``call_tool`` already flattens the MCP result to a text string, so this
        parses the string (the server emits the payload as a JSON object).
        """
        if not isinstance(text, str):
            return None
        stripped = text.strip()
        if not stripped.startswith("{"):
            return None
        try:
            data = json.loads(stripped)
        except (ValueError, TypeError):
            return None
        if isinstance(data, dict) and data.get("status") == "needs_approval":
            return data
        return None


# ---------------------------------------------------------------------------
# The bridge: OnIt tools -> Pipecat function calls
# ---------------------------------------------------------------------------
class ToolBridgeProcessor(FrameProcessor):
    """Registers every OnIt MCP tool as a Pipecat function call.

    Pipecat 1.15 auto-extracts the name, description, and argument schema from
    the function's signature and docstring. Each generated handler forwards the
    model's arguments straight to the OnIt server, which is where the policy
    and the jail actually enforce anything.
    """

    def __init__(self, servers: list[OnItMCPServer],
                 guard: Optional[Guard] = None):
        super().__init__()
        self.servers = servers
        self._tool_server: dict[str, OnItMCPServer] = {}
        self.tools: list[Callable] = []
        # The approval gate. When present, every tool call is routed through
        # it so OnIt's ``needs_approval`` payloads reach a person.
        self.guard = guard or Guard()

        # Set by agent.build_pipeline once the PipelineTask exists. A spoken
        # acknowledgement has to be queued through the task (the pipeline's own
        # entry point), because this processor is not itself in the stage list.
        self.task = None
        # Layer 2 switch: say a short line before a tool runs, so the user hears
        # something even while a slow tool is still working.
        self.announce = True

    async def _say(self, text: str) -> None:
        """Queue a fixed utterance to be spoken, without going through the LLM.

        ``TTSSpeakFrame`` is Pipecat's supported way to speak a literal string.
        Failing to speak must never break the tool call, so this is best-effort.
        """
        if not self.announce or self.task is None or not text:
            return
        from pipecat.frames.frames import TTSSpeakFrame
        try:
            await self.task.queue_frame(TTSSpeakFrame(text=text))
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not queue spoken acknowledgement: {}", e)

    async def discover(self) -> list[Callable]:
        """Connect to every server, build a callable per tool, and return the
        list to hand to LLMContext(tools=[...])."""
        for server in self.servers:
            for tool in await server.list_tools():
                self.tools.append(self._make_tool(server, tool))
                self._tool_server[tool.name] = server
        logger.info("Tool bridge ready: {} tools from {} server(s)",
                    len(self.tools), len(self.servers))
        return self.tools

    def _make_tool(self, server: OnItMCPServer, tool: Any) -> Callable:
        name = tool.name
        description = (getattr(tool, "description", "") or "").strip() or \
            f"Tool {name} from OnIt server {server.name}."
        # The MCP SDK (mcp-types) exposes the schema as `input_schema`.
        schema = getattr(tool, "input_schema", None) or {}
        props = schema.get("properties", {}) if isinstance(schema, dict) else {}
        required = set(schema.get("required", [])) if isinstance(schema, dict) else set()

        # Build a CONCRETE signature from the real OnIt schema. Pipecat's
        # DirectFunctionWrapper introspects the signature (and its type hints)
        # to build the function schema, so a `**kwargs` catch-all is not an
        # option -- it has no hint and hangs the conversion. Each schema
        # property becomes a named, typed parameter.
        import inspect
        import types as _types

        params = [inspect.Parameter("params", inspect.Parameter.POSITIONAL_ONLY,
                                    annotation=FunctionCallParams)]
        for pname, pschema in props.items():
            ann = _json_type_to_python(pschema)
            if pname in required:
                params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY,
                                                annotation=ann))
            else:
                default = pschema.get("default") if isinstance(pschema, dict) else None
                if default is None:
                    default = _json_type_default(ann)
                params.append(inspect.Parameter(pname, inspect.Parameter.KEYWORD_ONLY,
                                                default=default, annotation=ann))

        # Docstring: description + per-arg lines (Pipecat parses the "Args:"
        # section for parameter descriptions).
        doc_lines = [description, "", "Args:"]
        for pname, pschema in props.items():
            pdesc = (pschema.get("description") or "").strip() if isinstance(pschema, dict) else ""
            doc_lines.append(f"    {pname}: {pdesc or pname}.")
        docstring = "\n".join(doc_lines)

        # Pipecat 1.15 contract: a direct function MUST deliver its result by
        # calling ``params.result_callback``. ``_run_function_call`` does
        # ``await item.handler(params)`` and DISCARDS the return value, so a
        # handler that merely ``return``s a string broadcasts no
        # FunctionCallResultFrame at all. Without that frame the assistant
        # aggregator never appends the tool message to the context, the LLM is
        # never re-invoked, and the turn dies silently at the tool result --
        # which is exactly the "tool use is not spoken" bug. The return value
        # is therefore irrelevant; the callback is the whole contract.
        async def _handler(params: FunctionCallParams, **kwargs: Any) -> None:
            started = time.monotonic()
            # Layer 2: say what is happening before the tool runs. A tool can
            # take seconds (a web search) or wait on a person (an approval), and
            # silence during that time reads as "it stopped working". This is a
            # fixed phrase on purpose: it must not depend on the model.
            await self._say(_announcement(name))
            call_args = dict(kwargs)
            if name == "search":
                call_args = _shape_search_args(call_args)
            try:
                # One deadline for the whole call, retries included. Wrapping
                # only the individual attempt would let a 3-attempt search hold
                # a voice turn for 3x MCP_TOOL_TIMEOUT, which is minutes of
                # silence -- the very failure the spoken announcement exists to
                # avoid.
                text = await asyncio.wait_for(
                    self._call_with_search_retry(name, call_args, server),
                    timeout=config.MCP_TOOL_TIMEOUT,
                )
            except asyncio.TimeoutError:
                text = f"[tool timeout] {name} did not finish within " \
                       f"{config.MCP_TOOL_TIMEOUT:.0f}s."
            except Exception as e:  # noqa: BLE001
                logger.error("Tool {} failed: {}", name, e)
                text = f"[tool error] {name}: {e}"

            full = text if isinstance(text, str) else str(text)
            # Layer 3: the model sees a bounded, parseable head of the result;
            # the full text stays in the session folder and on the dashboard.
            # This is what keeps a huge MCP payload from being the thing that
            # decides whether the agent speaks.
            model_text = _cap_for_model(full, config.MCP_MODEL_RESULT_CHARS,
                                        query=call_args.get("query"))

            # Layer 1: make the tool turn observable. The event log used to
            # record the result but not what happened next, which is why the
            # silent-turn bug could not be diagnosed from it.
            if self.guard.bus is not None:
                await self.guard.bus.emit(
                    "tool_turn", tool=name, ms=int((time.monotonic() - started) * 1000),
                    result_chars=len(full), model_chars=len(model_text),
                    truncated=model_text != full,
                )

            # Always settle the call, even on failure: a handler that raises
            # out of here would leave the call unsettled until its timeout.
            await params.result_callback(model_text)

        _handler.__name__ = name
        _handler.__doc__ = docstring
        _handler.__signature__ = inspect.Signature(parameters=params)
        _handler.__onit_schema__ = schema  # type: ignore[attr-defined]
        _handler.__onit_server__ = server.name  # type: ignore[attr-defined]
        return _handler

    async def _call_with_search_retry(self, name: str, call_args: dict,
                                      server) -> str:
        """Run a tool, retrying a failed ``search`` a bounded number of times.

        OnIt's news branch calls ddgs once and converts any exception into
        ``{"error": "Search failed: ..."}``. ddgs *raises* rather than returning
        ``[]`` when its aggregator is rate-limited, measured at roughly 2
        failures in 8 news searches. The web branch already retries three times
        inside OnIt; the news branch never got that fix, and we are not editing
        OnIt, so the retry lives here.

        Only ``search`` is retried. It is read-only, so a second attempt cannot
        run a command twice -- a retried ``bash`` or ``write_file`` could.
        """
        text = await self.guard.handle(name, call_args, server, server.call_tool)
        if name != "search":
            return text

        attempts = max(1, config.MCP_SEARCH_RETRIES)
        for attempt in range(1, attempts):
            if not _search_failed(text):
                return text
            logger.info("Search failed ({}), retry {}/{}",
                        str(text)[:120], attempt, attempts - 1)
            await asyncio.sleep(config.MCP_SEARCH_RETRY_DELAY * attempt)
            text = await self.guard.handle(name, call_args, server,
                                           server.call_tool)
        return text

    async def close(self) -> None:
        for server in self.servers:
            await server.close()


# ---------------------------------------------------------------------------
# Convenience: build the bridge from config
# ---------------------------------------------------------------------------
def build_bridge(guard: Optional[Guard] = None) -> Optional[ToolBridgeProcessor]:
    """Construct the bridge from config.MCP_SERVERS, or None when tools are
    disabled or the OnIt interpreter is missing (Fweddie still runs, just
    without tools)."""
    if not config.MCP_TOOLS_ENABLED:
        logger.info("MCP tools disabled (FWEDDIE_MCP_TOOLS=0)")
        return None
    if not os.path.exists(config.ONIT_PYTHON):
        logger.warning(
            "OnIt interpreter not found at {}; starting without tools. "
            "Set ONIT_PYTHON to OnIt's venv python.", config.ONIT_PYTHON,
        )
        return None
    os.makedirs(config.MCP_DATA_PATH, exist_ok=True)
    servers = [
        OnItMCPServer(s["name"], s["module"], s.get("profile"))
        for s in config.MCP_SERVERS
    ]
    return ToolBridgeProcessor(servers, guard=guard)
