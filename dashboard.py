"""Fweddie -- live interaction dashboard (Streamlit).

Shows what the voice agent is doing, in real time, from any device on the
LAN:

  * a running transcript of the user's words (ASR) and Fweddie's replies
  * the OnIt tools the agent calls and what they return
  * a live approval panel -- when a command needs a person's approval, it
    appears here and you approve/deny it with a button

It is a separate process from the agent. They coordinate through plain files
(no version-specific Streamlit/Flask API, so it is robust):
  * the dashboard tails the agent's JSONL event log (config.DASHBOARD_LOG)
  * the agent writes a pending approval to <log>.pending.json; this app shows
    it and, when you click a button, writes your decision to
    <log>.decision.json, which the agent polls.

Run (from the repo, in the venv):
    .venv/bin/python -m streamlit run dashboard.py \
        --server.address 0.0.0.0 --server.port 8501
Then open http://<this-host-ip>:8501 from another device on the LAN.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

import config

LOG_PATH = Path(config.DASHBOARD_LOG)
PENDING_PATH = LOG_PATH.with_name(LOG_PATH.name + ".pending.json")
DECISION_PATH = LOG_PATH.with_name(LOG_PATH.name + ".decision.json")
POLL_SECS = 0.6  # how often to re-read the log / re-render


def _write_decision(decision: str) -> None:
    """Write the person's decision for the agent to pick up."""
    try:
        DECISION_PATH.parent.mkdir(parents=True, exist_ok=True)
        DECISION_PATH.write_text(
            json.dumps({"decision": decision, "ts": time.time()}), encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 -- best effort
        pass


def _read_pending() -> dict:
    """Read the agent's pending approval, if any. Returns {} when none."""
    try:
        if not PENDING_PATH.exists():
            return {}
        return json.loads(PENDING_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def _read_events(since: float) -> list[tuple[float, dict]]:
    """Read events newer than ``since`` from the JSONL log. Tolerates a
    missing file and a half-written last line."""
    events = []
    if not LOG_PATH.exists():
        return events
    try:
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue  # half-written line; next poll catches it
                ts = ev.get("ts", 0)
                if ts > since:
                    events.append((ts, ev))
    except FileNotFoundError:
        pass
    return events


def _fmt_ts(ts: float) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts))


def main() -> None:
    st.set_page_config(page_title="Fweddie", page_icon="🎙️", layout="wide")
    st.title("🎙️ Fweddie — live interaction")

    # Streamlit reruns the script top-to-bottom on every interaction / auto-
    # rerun. We keep the "last seen ts" in session_state so we only render new
    # events, and we auto-rerun on a timer to poll the log.
    if "last_ts" not in st.session_state:
        st.session_state.last_ts = time.time() - 5  # show a little history on load
    if "events" not in st.session_state:
        st.session_state.events = []

    new = _read_events(st.session_state.last_ts)
    if new:
        st.session_state.events.extend(ev for _, ev in new)
        st.session_state.last_ts = max(ts for ts, _ in new)
        if len(st.session_state.events) > 2000:
            st.session_state.events = st.session_state.events[-2000:]

    events = st.session_state.events

    # -- Approval panel (top, so it is never missed) ------------------------
    pending = _read_pending()
    if pending:
        st.warning("⚠️  Fweddie needs your approval before it can run a command")
        st.markdown(f"**Command:** `{pending.get('command', '')}`")
        if pending.get("reason"):
            st.caption(f"Reason: {pending['reason']}")
        if pending.get("subjects"):
            st.caption(f"About: {', '.join(map(str, pending['subjects']))}")
        c1, c2, c3 = st.columns(3)
        if c1.button("✅ Approve once", key="ap-once", type="primary"):
            _write_decision("once")
            st.rerun()
        if c2.button("🔁 Approve this session", key="ap-session"):
            _write_decision("session")
            st.rerun()
        if c3.button("❌ Deny", key="ap-deny"):
            _write_decision("deny")
            st.rerun()
        st.divider()
    else:
        st.caption("No pending approvals.")

    # -- Status row ---------------------------------------------------------
    started = next((e for e in reversed(events) if e.get("kind") == "session_start"), None)
    ended = any(e.get("kind") == "session_end" for e in events[-5:])
    status = "🔴 ended" if ended else ("🟢 live" if started else "⚪ waiting for agent")
    asr = (started or {}).get("asr", "?")
    tools = (started or {}).get("tools", "?")
    s1, s2, s3 = st.columns(3)
    s1.metric("Status", status)
    s2.metric("ASR", asr)
    s3.metric("OnIt tools", tools)
    st.divider()

    # -- Transcript + tool activity -----------------------------------------
    left, right = st.columns([3, 2])

    with left:
        st.subheader("Transcript")
        if not events:
            st.info("Waiting for the agent to start talking…")
        for ev in events:
            kind = ev.get("kind")
            if kind == "user_text":
                st.markdown(f"**🧑 You**  ·  {_fmt_ts(ev.get('ts', 0))}\n\n{ev.get('text','')}")
                st.divider()
            elif kind == "bot_text":
                st.markdown(f"**🤖 Fweddie**  ·  {_fmt_ts(ev.get('ts', 0))}\n\n{ev.get('text','')}")
                st.divider()
            elif kind == "session_start":
                st.caption(f"Session started · {ev.get('llm','?')} @ {ev.get('base_url','?')}")
            elif kind == "session_end":
                st.caption(f"Session ended · {_fmt_ts(ev.get('ts', 0))}")

    with right:
        st.subheader("Tools & safeguards")
        tool_events = [e for e in events if e.get("kind") in (
            "tool_call", "tool_result", "approval_requested",
            "approval_granted", "approval_denied")]
        if not tool_events:
            st.info("No tool activity yet.")
        for ev in tool_events:
            kind = ev.get("kind")
            ts = _fmt_ts(ev.get("ts", 0))
            if kind == "tool_call":
                st.markdown(f"🔧 **{ev.get('tool','?')}**  ·  {ts}")
                if ev.get("args"):
                    st.code(str(ev.get("args")), language=None)
            elif kind == "tool_result":
                st.markdown(f"↳ **result**  ·  {ts}")
                st.code(str(ev.get("result", "")), language=None)
            elif kind == "approval_requested":
                st.markdown(f"🛑 **approval needed**  ·  {ts}")
                st.code(str(ev.get("command", "")), language=None)
            elif kind == "approval_granted":
                st.markdown(f"✅ **approved ({ev.get('scope','?')})**  ·  {ts}")
            elif kind == "approval_denied":
                st.markdown(f"❌ **denied**  ·  {ts}")
                if ev.get("reason"):
                    st.caption(ev.get("reason"))
            st.divider()

    # Auto-rerun to poll the log.
    time.sleep(POLL_SECS)
    st.rerun()


if __name__ == "__main__":
    main()
