"""WanDog -- shared event bus between the agent and the dashboard.

The agent and the dashboard are separate processes. They coordinate through
two things, both deliberately simple (plain files) so a crash on either side
cannot take the other down and no version-specific Streamlit/Flask API is
needed:

  1. A JSONL event log. The agent appends one JSON object per line; the
     dashboard tails it. Events are the *record* of the interaction (who said
     what, which tool ran, what it returned, what was approved).

  2. A pair of small files for live approvals. When a command needs a person's
     approval, the agent writes the pending payload to ``<log>.pending.json``.
     The dashboard reads it, shows it, and the person clicks a button; the
     dashboard then writes the decision to ``<log>.decision.json``. The agent
     polls for that file and acts on it. If the dashboard is never up, the
     agent falls back to the console prompt, so approvals never block on the
     browser.

The paths come from config so the same code runs headless (dashboard off) or
with the dashboard on.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

import config

# A ring of recent events kept in memory so the dashboard can also read them
# without re-parsing the whole log on every poll. The log is the source of
# truth; this is just a fast cache for the agent's own process.
_MAX_MEMORY_EVENTS = 500


def _now() -> float:
    return time.time()


def _emit(path: str, event: dict) -> None:
    """Append one event to the JSONL log. Never raises -- a full disk or a
    missing directory must not crash the voice loop."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        event = dict(event)
        event.setdefault("ts", _now())
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:  # noqa: BLE001 -- telemetry must be best-effort
        pass


class EventBus:
    """Append-only event log + in-memory ring, with file-based live approval.

    ``log`` is the JSONL path (config.DASHBOARD_LOG). When the dashboard is
    off, events are still written (cheap) so a later dashboard can show the
    session; nothing is lost and nothing is required.
    """

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = log_path or config.DASHBOARD_LOG
        self._log = Path(self.log_path)
        self._pending_path = self._log.with_name(self._log.name + ".pending.json")
        self._decision_path = self._log.with_name(self._log.name + ".decision.json")
        self._recent: list[dict] = []
        import threading
        self._lock = threading.Lock()
        # Live approval: the pending request the dashboard is deciding.
        self._approval_event: Optional[dict] = None

    # -- emit ---------------------------------------------------------------
    async def emit(self, kind: str, **fields: Any) -> None:
        """Record one event. Runs the (blocking) file write off the event
        loop so a slow disk never stalls audio."""
        event = {"kind": kind, **fields}
        await asyncio.to_thread(self._append, event)

    def _append(self, event: dict) -> None:
        _emit(self.log_path, event)
        with self._lock:
            self._recent.append(event)
            if len(self._recent) > _MAX_MEMORY_EVENTS:
                del self._recent[: len(self._recent) - _MAX_MEMORY_EVENTS]

    def recent(self, limit: int = 200) -> list[dict]:
        return list(self._recent[-limit:])

    # -- live approval (file-based) -----------------------------------------
    def set_pending_approval(self, payload: dict) -> None:
        self._approval_event = dict(payload)

    def clear_pending_approval(self) -> None:
        self._approval_event = None

    @property
    def pending_approval(self) -> Optional[dict]:
        return self._approval_event

    def write_pending(self, payload: dict) -> None:
        """Write the pending approval to disk for the dashboard to pick up."""
        try:
            self._pending_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._pending_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(self._pending_path)
        except Exception:  # noqa: BLE001 -- best effort
            pass

    def _read_decision(self) -> Optional[str]:
        """Read and clear a decision the dashboard wrote. Returns
        once/session/deny or None if none is present."""
        try:
            if not self._decision_path.exists():
                return None
            data = json.loads(self._decision_path.read_text(encoding="utf-8"))
            decision = data.get("decision")
            self._decision_path.unlink(missing_ok=True)
            return decision
        except Exception:  # noqa: BLE001
            return None

    async def request_approval(self, payload: dict) -> Optional[str]:
        """Wait for the dashboard to decide. Returns once/session/deny, or
        None if the dashboard is not up (caller falls back to the console).
        Polls the decision file until it appears or the timeout is hit."""
        if not config.DASHBOARD_ENABLED:
            return None
        # If there is no dashboard process, the decision file will never
        # appear; bail out quickly so we fall back to the console.
        deadline = time.time() + config.MCP_APPROVAL_TIMEOUT
        while time.time() < deadline:
            decision = await asyncio.to_thread(self._read_decision)
            if decision:
                return decision
            await asyncio.sleep(0.4)
        # Timed out waiting on the dashboard.
        return None

    def pending_file_exists(self) -> bool:
        return self._pending_path.exists()


# Module-level default bus the agent uses. The dashboard reads the same log
# file directly (it does not import this object across processes).
BUS = EventBus()
