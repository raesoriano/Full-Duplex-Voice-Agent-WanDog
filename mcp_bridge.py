"""WanDog -- OnIt MCP tool bridge.

Spawns OnIt's own stdio MCP servers and exposes their tools to the LLM as
Pipecat function calls. WanDog does NOT re-implement OnIt's tools or its
safeguards: it talks to the real OnIt servers over the MCP protocol, so the
AST command allowlist, the approval gate, and the DATA_PATH file jail all run
inside the spawned server process and apply to every call WanDog makes.

    WanDog LLM  --function call-->  ToolBridgeProcessor  --MCP-->  OnIt server (stdio)
                                                                              |
                                                                          bash / read_file /
                                                                          search / weather ...
                                                                          (policy + jail here)

Run the servers with OnIt's interpreter (it has fastmcp + mcp), from OnIt's
package root so `src.*` imports resolve. The MCP client lives in WanDog's own
venv.
"""

from __future__ import annotations

import asyncio
import datetime
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
    """A single OnIt stdio MCP server: a subprocess WanDog owns.

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
        # else is refused) -- WanDog never loosens it.
        #
        # ONIT_APPROVAL_CHANNEL=1 tells the server a person is reachable, so a
        # command the allowlist will not decide alone returns a needs_approval
        # ticket (which the Guard hands to a human) instead of collapsing to a
        # hard refusal. Without it every ask is a deny -- the fail-closed
        # default for a server with no one to ask.
        #
        # ONIT_COMMAND_ALLOWLIST=1 turns the AST command allowlist on. On a
        # host (outside the onit container) it is off by default, which would
        # let WanDog run any executable -- the allowlist is the whole point of
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
        # subprocess only: WanDog's own process never gets them.
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
        # Private bookkeeping keys (leading underscore) are ours, not the
        # server's: its signature does not declare them and the MCP SDK
        # validates arguments against the tool schema. Drop them at the wire.
        payload = {k: v for k, v in (arguments or {}).items()
                   if not str(k).startswith("_")}
        result = await session.call_tool(name, payload)
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
    # Ask for MORE than the caller wanted. ddgs orders news by relevance, not
    # by date, so the newest story is routinely not in the first five and the
    # model cap -- which keeps only the first two results -- never sees it.
    # Measured on "Alaska US Senate election results", 17 Nov 2026: the
    # five-result page was 17, 12, 17, 14, 11 Nov, so the agent answered
    # "November 12" while two 17 Nov stories were in the same payload. The
    # over-fetch is trimmed back to `wanted` after the date sort.
    wanted = 5
    try:
        if int(shaped.get("max_results") or 0) >= 3:
            wanted = int(shaped["max_results"])
    except (TypeError, ValueError):
        pass
    shaped["max_results"] = max(wanted, config.MCP_NEWS_OVERFETCH)
    # Remember what the caller actually asked for; _trim_news uses it. Kept
    # under a private key so it never reaches the server's signature.
    shaped["_news_wanted"] = wanted
    return shaped


def _shape_github_args(args: dict) -> dict:
    """Qualify a bare repository name as ``owner/repo`` for the actions that
    need one.

    ``github_repo`` is one tool with two naming conventions, and its
    description says so in a parenthetical: ``name`` is "Repository name" for
    ``create`` and "``owner/repo``" for ``get``/``fork``/``delete``. The model
    reads the first line -- "Repository name" -- and sends a bare name, which
    the server pastes straight into ``/repos/{name}``. GitHub then answers 404,
    the same status it uses for a repository that really is gone, so the
    failure is indistinguishable from "already deleted".

    Measured on the real account, 18 Nov 2026, against a repo created seconds
    earlier::

        create name="wandog-probe2"              -> 201 raesoriano/wandog-probe2
        delete name="wandog-probe2"              -> 404 Not Found
        delete name="raesoriano/wandog-probe2"   -> 204 deleted

    The live event log shows the same shape: 13 of 13 delete calls sent a bare
    name, all 13 got 404, and the agent told the person the repositories "might
    have already been removed" four times over two minutes while they sat in
    plain sight in its own ``list`` output.

    ``create`` is left alone -- it takes a bare name, and qualifying it would
    try to create a repository with a slash in its name. An ``org`` argument is
    left alone too: there the caller has already said which namespace it means.
    """
    if not isinstance(args, dict):
        return args
    if args.get("action") not in ("get", "fork", "delete"):
        return args
    name = args.get("name")
    if not isinstance(name, str) or not name.strip():
        return args
    name = name.strip()
    if "/" in name:
        return args                    # already owner/repo; respect it
    if args.get("org"):
        return args
    owner = (config.OWNER_GITHUB or "").strip()
    if not owner:
        return args
    shaped = dict(args)
    shaped["name"] = f"{owner}/{name}"
    return shaped


def _tavily_news_key() -> Optional[str]:
    """The Tavily API key, read through config's secret store.

    Returns None when unset, which makes the Tavily news tier inert rather
    than fatal: news simply stays on OnIt's ddgs branch.
    """
    try:
        secrets = config.tool_secret_env()
    except Exception as e:  # noqa: BLE001
        logger.debug("Could not read tool secrets for Tavily: {}", e)
        return None
    if not isinstance(secrets, dict):
        return None
    key = (secrets.get("TAVILY_API_KEY") or "").strip()
    return key or None


def _tavily_news(query: str, max_results: int) -> Optional[str]:
    """Fetch news from Tavily's news topic, in OnIt's news payload shape.

    Why this exists at all: OnIt's ``search(type="news")`` branch calls ddgs
    only and never touches its own Tavily tier. Measured on 17 Nov 2026 the
    newest story ddgs returned was 0-3 days old for the same queries where
    Tavily returned a same-day one (see config.MCP_NEWS_BACKEND).

    Returns the same ``[{title, snippet, date, source, url}]`` JSON string
    OnIt returns, so everything downstream -- the date sort, the age labels,
    the model cap -- works unchanged. Returns None on any failure, which is
    the signal to fall back to ddgs.
    """
    key = _tavily_news_key()
    if not key:
        return None
    try:
        import requests
    except Exception:  # noqa: BLE001
        return None

    body = {
        "api_key": key,
        "query": query,
        "max_results": max(1, min(int(max_results), 20)),
        "topic": "news",
        "days": max(1, min(int(config.MCP_NEWS_TAVILY_DAYS), 7)),
        "search_depth": config.MCP_NEWS_TAVILY_DEPTH,
        "include_answer": False,
    }
    try:
        resp = requests.post("https://api.tavily.com/search", json=body,
                             timeout=config.MCP_TOOL_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        logger.warning("Tavily news failed ({}); falling back to ddgs",
                       str(e)[:120])
        return None

    rows = []
    for r in data.get("results", []) or []:
        # Tavily dates look like "Tue, 17 Nov 2026 03:44:14 GMT". Keep the
        # raw string: _parse_news_date handles several shapes and the age
        # labels are computed from it, so there is no need to normalise here.
        rows.append({
            "title": (r.get("title") or "").strip(),
            "snippet": (r.get("content") or "").strip()[:300],
            "date": (r.get("published_date") or "").strip(),
            "source": (r.get("source") or _host_of(r.get("url") or "")).strip(),
            "url": (r.get("url") or "").strip(),
        })
    if not rows:
        logger.info("Tavily news returned nothing for {!r}; falling back to ddgs",
                    query[:60])
        return None
    return json.dumps(rows, ensure_ascii=False)


def _host_of(url: str) -> str:
    """Bare hostname from a URL, for the `source` field Tavily may omit."""
    try:
        host = url.split("//", 1)[-1].split("/", 1)[0]
        return host[4:] if host.startswith("www.") else host
    except Exception:  # noqa: BLE001
        return ""


def _news_count(text: str) -> int:
    """How many rows are in a news payload, for logging. 0 if unparseable."""
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return 0
    return len(data) if isinstance(data, list) else 0


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


def _parse_news_date(value: Any) -> Optional[datetime.datetime]:
    """Parse a news ``date`` into a datetime, or None.

    Three shapes reach us:
      * ddgs' raw ISO stamp ("2026-11-17T09:44:19+00:00");
      * OnIt's reformatted "November 17, 2026" (it strftimes the ISO stamp);
      * Tavily's RFC-2822-ish "Tue, 17 Nov 2026 03:44:14 GMT".
    The Tavily shape is why this list exists: without it every Tavily row
    parsed as None, so the date sort had nothing to sort on and the age labels
    came out empty -- the two things the freshness work depends on. Measured
    before the fix: 10/10 Tavily rows unparseable, newest=None.
    Anything still unparseable returns None and sorts last but is never
    dropped -- a result with an unreadable date is still a result.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        pass
    for fmt in (
        "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y",
        # Tavily / RFC 2822: "Tue, 17 Nov 2026 03:44:14 GMT"
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S %z",
        "%a, %d %b %Y",
    ):
        try:
            return datetime.datetime.strptime(text, fmt)
        except ValueError:
            continue
    # Last resort: strip a leading weekday and try the common date-only forms,
    # so an unexpected timezone suffix cannot hide the date entirely.
    stripped = re.sub(r"^[A-Za-z]{3},\s*", "", text)
    for fmt in ("%d %b %Y %H:%M:%S %Z", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.datetime.strptime(stripped, fmt)
        except ValueError:
            continue
    return None


def _news_age_label(when: Optional[datetime.datetime],
                    now: Optional[datetime.datetime] = None) -> str:
    """Human age of a news item: ``today``, ``yesterday``, ``3 days ago``."""
    if when is None:
        return ""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    if when.tzinfo is not None:
        when = when.astimezone()
    else:
        when = when.replace(tzinfo=None)
        now = now.astimezone().replace(tzinfo=None)
    days = (now.date() - when.date()).days
    if days <= 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago"


def _news_newest_age_days(text: str) -> Optional:
    """Age in days of the newest result in a news payload, or None.

    Used to decide whether a news page is worth re-asking for. None means the
    payload carries no readable date at all, which is treated as "cannot
    judge" -- the caller then does not retry, because there is nothing to
    compare against and a retry would be a coin flip.
    """
    stripped = text.strip() if isinstance(text, str) else ""
    if not stripped.startswith("["):
        return None
    try:
        items = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if not isinstance(items, list):
        return None
    now = datetime.datetime.now(datetime.timezone.utc)
    ages = []
    for item in items:
        if not isinstance(item, dict):
            continue
        when = _parse_news_date(item.get("date"))
        if when is None:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=datetime.timezone.utc)
        ages.append((now.astimezone().date() - when.astimezone().date()).days)
    return min(ages) if ages else None

def _sort_news(text: str, wanted: Optional[int] = None) -> str:
    """Order a news payload newest-first and label each result with its age.

    Why this exists: ddgs returns news by relevance, not recency. On 17 Nov
    2026 the five results for "Alaska US Senate election results" came back
    dated 17, 12, 17, 14 and 11 November. The model cap keeps only the first
    two, so the agent said "as of November 12" -- while two stories from that
    same day sat further down the very same payload. The user had Google open
    showing hours-old coverage, which is what made the news search look broken.

    Sorting here, before the cap, is the fix: the cap can only keep a prefix,
    so the prefix has to be the newest. ``wanted`` trims the over-fetched list
    back to the number of results the caller actually asked for.

    ``age`` is added as its own field rather than folded into the snippet: the
    model reads ``date`` for the citation and ``age`` for the judgement, and a
    voice answer wants both ("today" is what makes it say "just now" rather
    than reading an old date aloud).

    Returns the payload untouched when it is not a JSON list of dicts.
    """
    if not config.MCP_NEWS_SORT:
        return text
    stripped = text.strip()
    if not stripped.startswith("["):
        return text
    try:
        items = json.loads(stripped)
    except (ValueError, TypeError):
        return text
    if not isinstance(items, list) or not items:
        return text
    if not all(isinstance(item, dict) for item in items):
        return text

    now = datetime.datetime.now(datetime.timezone.utc)
    epoch = datetime.datetime.min.replace(tzinfo=datetime.timezone.utc)

    def key(item: dict) -> datetime.datetime:
        parsed = _parse_news_date(item.get("date"))
        if parsed is None:
            return epoch
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=datetime.timezone.utc)
        return parsed

    # A list where NOTHING carries a readable date is not a news payload --
    # it is a general web result that happens to be a list. Sorting it would
    # reorder by nothing and re-encode the JSON for no benefit, so leave it
    # byte-for-byte alone.
    if all(key(item) == epoch for item in items):
        return text

    ordered = sorted(items, key=key, reverse=True)
    if wanted is not None and wanted > 0:
        ordered = ordered[:wanted]

    labelled = []
    for item in ordered:
        enriched = dict(item)
        label = _news_age_label(_parse_news_date(item.get("date")), now)
        if label:
            enriched["age"] = label
        labelled.append(enriched)
    return json.dumps(labelled, ensure_ascii=False)


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
    capped = _cap_json_obj(text, limit, query)
    if capped is not None:
        return capped

    return _cap_head_slice(text, limit)


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


def _cap_json_obj(text: str, limit: int, query: Optional = None) -> Optional:
    """Fit a JSON **dict** into ``limit`` chars without breaking the JSON.

    The list path below covers every ``search`` result. It does not cover the
    other tools, and that gap is a live defect: ``local_search``, ``grep``,
    ``read_file``, ``search_document`` and ``get_weather`` all return a dict,
    and a dict fell through to the blind head-slice, which cut the JSON
    mid-string. Measured on the four shapes the model can actually receive:

        local_search     1718 chars -> 1240 chars of INVALID JSON
        read_file       12094 chars -> 1241 chars of INVALID JSON
        grep             3990 chars -> 1240 chars of INVALID JSON
        search_document  2828 chars -> 1240 chars of INVALID JSON

    The model cannot parse any of those, so a dict-shaped result is worth
    nothing at all once it exceeds the cap -- the same failure as the empty
    ``{"results": []}`` this cap was written to stop, just for the other tools.
    It stayed invisible because the event log has only ``search``,
    ``get_weather`` and ``bash`` calls in it: the payloads that break are the
    ones the agent has not happened to use yet.

    So a dict is capped structurally too: the bulkiest list field loses whole
    records from its tail, prose is clipped inside the records that remain, and
    the note says how many were dropped. Identity fields stay whole, as above.
    ``None`` means "not a JSON dict" -- the caller then falls back to text.
    """
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        obj = json.loads(stripped)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    terms = _cap_terms(query)

    # The bulk of a dict payload is a list of records, so that is what gets
    # trimmed. Pick the biggest one by rendered size, not item count: one
    # `documents` entry carries an `opening` several times the size of a
    # `results` chunk.
    list_keys = [k for k, v in obj.items() if isinstance(v, list) and v]
    list_keys.sort(
        key=lambda k: len(json.dumps(obj[k], ensure_ascii=False)), reverse=True)
    biggest = list_keys[0] if list_keys else None
    total = len(obj[biggest]) if biggest else 0

    def render(clip: Optional, keep: Optional) -> str:
        out = {}
        dropped = 0
        for k, v in obj.items():
            if biggest and k == biggest and keep is not None and isinstance(v, list):
                dropped = len(v) - keep
                v = v[:keep]
            if clip is not None:
                v = _cap_clip(v, clip, terms)
            out[k] = v
        if dropped > 0:
            # Totals are left as the server reported them: they are true, and
            # the note is what explains why fewer records are visible.
            out["note"] = (f"showing {keep} of {total} {biggest}; "
                           f"{dropped} dropped to fit the model's context")
        elif clip is not None:
            out["note"] = (f"each text field is cut to {clip} chars to fit the "
                           f"model's context; urls, dates and counts are intact")
        return json.dumps(out, ensure_ascii=False)

    options = []
    for clip in _CAP_CLIP_LADDER:
        keeps = range(total, 0, -1) if biggest else [None]
        for keep in keeps:
            out = render(clip, keep)
            if len(out) <= limit:
                options.append((keep or 0, clip, out))
                break

    if options:
        # Same preference as the list path: two records beat one long one, and
        # among equals the least clipping wins.
        def rank(o):
            return (o[1] if o[1] is not None else 10 ** 6, o[0])

        many = [o for o in options if o[0] >= 2]
        return max(many or options, key=rank)[2]

    # Nothing fit even with the tightest clip. Return a valid object that still
    # names the tool's own keys, so the model can say what it did get rather
    # than reading half a string.
    return json.dumps(
        {"note": f"the result was {len(text)} chars and no part of it fits "
                 f"within {limit} chars; its fields were {sorted(obj)[:8]}"},
        ensure_ascii=False,
    )


def _cap_head_slice(text: str, limit: int) -> str:
    """Last resort for plain text: a head slice, marked as truncated.

    Right for a shell transcript or a file body, where the opening is what the
    model can use. It must never be applied to JSON -- that is what the two
    structural caps above exist to prevent -- so it stays a separate function
    rather than an inline fallback, and the caller reaches it only after both
    JSON paths have declined.

    Measured limit: the tail is lost. ``bash`` output whose verdict is on the
    last line (a test summary, ``git status``) is not readable through this.
    Not fixed here because it does not bite this workload yet: in the 62-call
    event log every ``bash`` result was 140-172 chars, well under the cap. If a
    long command output ever needs its tail, this is the function to change.
    """
    return (text[:limit].rstrip()
            + f"\n[truncated: showing {limit} of {len(text)} chars]")


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
    # A single result's identity alone is over budget. That used to end in
    # ``{"results": []}`` -- an empty list, which is precisely the failure the
    # two-stage cap was written to remove, just reachable only by a 1200+ char
    # url or title (measured: a 2000-char url produced "no result could be
    # shown" for a payload holding one real result). Empty is the wrong answer
    # when the server did return something, so clip the identity fields too and
    # keep the result recognisable instead.
    for ident_clip in (600, 300, 150, 80, 40):
        skeleton = []
        for item in items[:1]:
            if not isinstance(item, dict):
                skeleton.append(str(item)[:ident_clip])
                continue
            skeleton.append({
                k: (v[:ident_clip].rstrip() + "\u2026"
                    if isinstance(v, str) and len(v) > ident_clip else v)
                for k, v in item.items() if k in _CAP_KEEP_INTACT
            })
        out = json.dumps(
            {"results": skeleton, "showing": 1, "total": total,
             "note": f"only the first of {total} results fits within {limit} "
                     f"chars, and its url and title are cut to {ident_clip} "
                     f"chars; the body text was dropped"},
            ensure_ascii=False,
        )
        if len(out) <= limit:
            return out
    # Only reachable if the budget is smaller than the wrapper itself, i.e. a
    # limit of a few dozen chars. Still not an empty list: name the count.
    return json.dumps(
        {"showing": 0, "total": total,
         "note": f"the first result's title and url alone exceed {limit} chars"},
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
            # How many results the caller asked for. _shape_search_args
            # over-fetches so the date sort has something to choose from, and
            # records the real number under a private key. It is popped here
            # because call_args is forwarded verbatim to the MCP server, which
            # would reject an argument its signature does not declare.
            news_wanted = None
            if name == "search":
                call_args = _shape_search_args(call_args)
                news_wanted = call_args.pop("_news_wanted", None)
            # Same idea as the search shaping above, for the other tool whose
            # arguments the model gets half right. github_repo takes a bare
            # name to create and owner/repo to get, fork or delete; the model
            # sends a bare name for all four, and every delete then 404s.
            if name == "github_repo":
                call_args = _shape_github_args(call_args)
            try:
                # One deadline for the whole call, retries included. Wrapping
                # only the individual attempt would let a 3-attempt search hold
                # a voice turn for 3x MCP_TOOL_TIMEOUT, which is minutes of
                # silence -- the very failure the spoken announcement exists to
                # avoid.
                # News goes to Tavily first when configured, because OnIt's
                # news branch is ddgs-only and measurably older. A None here
                # means "not applicable" and the normal path runs unchanged.
                text = None
                if name == "search" and call_args.get("type") == "news":
                    text = await self._news_search(call_args, server)
                if text is None:
                    text = await asyncio.wait_for(
                        self._call_with_search_retry(name, call_args, server),
                        timeout=config.MCP_TOOL_TIMEOUT,
                    )
            except asyncio.TimeoutError:
                text = f"[tool timeout] {name} did not finish within " \
                       f"{config.MCP_TOOL_TIMEOUT:.0f}s."
            except asyncio.CancelledError:
                # The turn was interrupted (barge-in) while the tool was still
                # running. Pipecat cancels this task and settles the call itself
                # in _cancel_function_call_tasks -- it marks runner_item.settled
                # BEFORE cancelling, then broadcasts FunctionCallCancelFrame,
                # which the aggregator records as the tool result "CANCELLED".
                # Reporting our own result here would be rejected ("Ignoring
                # result for function call ... already settled"), so this only
                # logs. The real fix is upstream: stop the barge-in happening.
                # Measured in /tmp/wandog_events.jsonl: 7 tool_calls were
                # cancelled 0.6-4.2s after starting, all 7 of them a `search`,
                # and 5 of the 7 were triggered by the agent's own "Searching
                # for that." announcement being picked up by the mic.
                logger.info("Tool {} cancelled after {:.1f}s (call settled as "
                            "CANCELLED by Pipecat)", name,
                            time.monotonic() - started)
                raise
            except Exception as e:  # noqa: BLE001
                logger.error("Tool {} failed: {}", name, e)
                text = f"[tool error] {name}: {e}"

            full = text if isinstance(text, str) else str(text)
            # News only: order newest-first and label the age, BEFORE the cap.
            # The cap keeps a prefix, so the prefix has to be the newest item;
            # ddgs' relevance order puts old stories first (measured: 17, 12,
            # 17, 14, 11 Nov for the Alaska race on 17 Nov), which is why the
            # agent kept reporting a days-old date. Also trims the over-fetch
            # in _shape_search_args back to what the caller asked for.
            if name == "search" and call_args.get("type") == "news":
                full = _sort_news(full, news_wanted)
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

    async def _refresh_stale_news(self, text: str, name: str, call_args: dict,
                                  server) -> str:
        """Re-run a news search when the newest result is not from today.

        ddgs returns a *different* set of stories for the same query on
        consecutive calls -- measured on "Alaska US Senate election results"
        on 17 Nov 2026, six calls split into two distinct pages, one of which
        contained no 17 Nov story at all. So a stale page is often not "no news
        exists" but "this draw missed it", and asking again usually finds it:
        measured over 6 rounds, the newest item was from today on 3 of 6 first
        calls, and on 5 of 6 when two more calls were allowed. Cost: mean
        latency 0.8s -> 2.2s, hidden behind the spoken announcement.

        The best (newest) payload is kept, never the last one, so a retry can
        only improve the answer. Bounded by MCP_NEWS_REFRESH_RETRIES and
        skipped entirely when the newest result is already fresh, so a topic
        with genuinely no news today pays one extra call, not a loop.
        """
        attempts = max(0, config.MCP_NEWS_REFRESH_RETRIES)
        if not attempts or _search_failed(text):
            return text
        best = text
        best_age = _news_newest_age_days(best)
        # The first call already happened; the budget covers the retries only.
        deadline = time.monotonic() + config.MCP_NEWS_REFRESH_BUDGET
        for _ in range(attempts):
            if best_age is not None and best_age <= config.MCP_NEWS_FRESH_DAYS:
                break
            await asyncio.sleep(config.MCP_NEWS_REFRESH_DELAY)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            # Bound the retry itself, not just the loop: a single ddgs call
            # was measured taking 10.5s, which would blow the budget the loop
            # check was meant to enforce. On timeout the payload we already
            # have wins, so the worst case is the first page's freshness.
            try:
                candidate = await asyncio.wait_for(
                    self.guard.handle(name, call_args, server, server.call_tool),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                logger.info("News refresh hit its {:.1f}s budget; keeping the "
                            "first page", config.MCP_NEWS_REFRESH_BUDGET)
                break
            if _search_failed(candidate):
                continue
            age = _news_newest_age_days(candidate)
            if age is not None and (best_age is None or age < best_age):
                best, best_age = candidate, age
        if best is not text:
            logger.info("News refreshed: newest result now {} day(s) old",
                        best_age)
        return best

    async def _news_search(self, call_args: dict, server) -> Optional[str]:
        """Serve a news search from Tavily when configured, else from OnIt.

        Returns the payload string, or None to mean "let the normal path run"
        (Tavily disabled, no key, or Tavily came back empty/failed).

        This sits ahead of ``server.call_tool`` rather than inside it because
        OnIt's news branch is ddgs-only and we are not editing OnIt. Everything
        after this point -- date sort, age labels, model cap -- is identical
        whichever provider answered, because the payload shape is the same.
        """
        if config.MCP_NEWS_BACKEND != "tavily":
            return None
        query = (call_args.get("query") or "").strip()
        if not query:
            return None
        wanted = call_args.get("_news_wanted") or call_args.get("max_results") or 5
        try:
            wanted = int(wanted)
        except (TypeError, ValueError):
            wanted = 5
        # Run the blocking HTTP call off the event loop: this is a voice turn
        # and a stalled loop would delay audio, not just the tool.
        try:
            payload = await asyncio.get_running_loop().run_in_executor(
                None, _tavily_news, query, max(wanted, config.MCP_NEWS_OVERFETCH))
        except Exception as e:  # noqa: BLE001
            logger.warning("Tavily news path failed: {}", str(e)[:120])
            return None
        if payload is None:
            return None
        logger.info("News served by Tavily ({} result(s))", _news_count(payload))
        return payload

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

        # A page that came back successfully can still be days stale, because
        # ddgs orders by relevance and varies its draw. Only news is re-asked;
        # a general web result has no date to judge.
        if call_args.get("type") == "news":
            text = await self._refresh_stale_news(text, name, call_args, server)
        return text

    async def close(self) -> None:
        for server in self.servers:
            await server.close()


# ---------------------------------------------------------------------------
# Convenience: build the bridge from config
# ---------------------------------------------------------------------------
def build_bridge(guard: Optional[Guard] = None) -> Optional[ToolBridgeProcessor]:
    """Construct the bridge from config.MCP_SERVERS, or None when tools are
    disabled or the OnIt interpreter is missing (WanDog still runs, just
    without tools)."""
    if not config.MCP_TOOLS_ENABLED:
        logger.info("MCP tools disabled (WANDOG_MCP_TOOLS=0)")
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
