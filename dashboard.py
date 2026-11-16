"""Fweddie -- Streamlit dashboard: live chat transcript + mic control.

Two processes talk through files (see ``eventbus.py``):

  * the agent appends one JSON object per line to ``config.DASHBOARD_LOG``;
  * this page tails that file and renders it as a chat.

The event vocabulary the agent emits is ``kind``-based, NOT role-based:

    session_start  {llm, base_url, asr, tools, ts}
    session_end    {}
    user_started   {}                      -- VAD fired
    user_text      {text}                  -- ASR transcript   (what you said)
    user_stopped   {}
    bot_started    {}
    bot_text       {text}                  -- the reply        (what it said)
    bot_stopped    {}
    tool_call      {name, args}
    tool_result    {name, result}
    approval       {...}

An earlier version of this file read ``role`` / ``content`` / ``timestamp``
from each line, which the agent never writes -- so the log always rendered
empty no matter what you said. That is fixed here: every event is mapped by
its ``kind``.

Mic control is a small JSON file next to the log
(``<log>.mic_control.json``). The agent polls it (see ``MicControlWatcher``
in ``agent.py``) and mutes/unmutes the ASR to match. Before this, the toggle
wrote the file but no agent code ever read it, so it did nothing.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import streamlit as st

import config

LOG_PATH = Path(config.DASHBOARD_LOG)
MIC_CONTROL_FILE = Path(config.DASHBOARD_LOG).with_suffix(".mic_control.json")

st.set_page_config(page_title="Fweddie Voice Dashboard", layout="wide")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def read_events(limit: int = 200) -> list[dict]:
    """Return the last ``limit`` events, oldest first. Tolerates a line being
    written at the moment we read it (JSONDecodeError -> skip)."""
    if not LOG_PATH.exists():
        return []
    try:
        lines = LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    events = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def mic_state() -> bool:
    try:
        return bool(json.loads(MIC_CONTROL_FILE.read_text(encoding="utf-8"))["enabled"])
    except Exception:
        return True


def set_mic_state(enabled: bool) -> None:
    MIC_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MIC_CONTROL_FILE.write_text(json.dumps({"enabled": enabled}), encoding="utf-8")


def fmt_time(ts) -> str:
    try:
        return time.strftime("%H:%M:%S", time.localtime(float(ts)))
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Sidebar: mic control
# ---------------------------------------------------------------------------

st.sidebar.header("Controls")


def _on_mic_toggle() -> None:
    """Write the toggle straight to the control file, in the widget callback.

    Doing it here rather than by comparing the widget value to the file after
    the fact avoids a fight with Streamlit's widget state: with the comparison
    approach the widget could re-render showing a stale value and the file
    silently disagreed with the switch.
    """
    set_mic_state(bool(st.session_state["mic_on"]))


# Seed session state from the file only on the first run, so a value the user
# set is not overwritten on every rerun.
if "mic_on" not in st.session_state:
    st.session_state["mic_on"] = mic_state()

st.sidebar.toggle(
    "Microphone active",
    key="mic_on",
    on_change=_on_mic_toggle,
    help="Mutes the agent's speech recognition. It keeps listening for "
         "barge-in, but will not transcribe or answer.",
)

if not st.session_state["mic_on"]:
    st.sidebar.warning("The agent will not transcribe while muted.")

st.sidebar.divider()
st.sidebar.caption(f"Event log\n\n`{LOG_PATH}`")
st.sidebar.caption(f"Mic control\n\n`{MIC_CONTROL_FILE}`")

events = read_events()
if events:
    last = events[-1].get("ts")
    st.sidebar.caption(f"Last event: {fmt_time(last)}")
else:
    st.sidebar.caption("No events yet.")

# ---------------------------------------------------------------------------
# Main: chat transcript
# ---------------------------------------------------------------------------

st.title("Fweddie Voice Agent")

if not LOG_PATH.exists():
    st.info(
        "Waiting for the agent to start. Run `.venv/bin/python agent.py` "
        "in the repo, then speak."
    )
else:

    @st.fragment(run_every=1.0)
    def render_log():
        events = read_events()
        if not events:
            st.info("The log is empty. Is the agent running?")
            return

        for ev in events:
            kind = ev.get("kind", "")
            ts = fmt_time(ev.get("ts"))

            if kind == "user_text":
                with st.chat_message("user", avatar="🎤"):
                    st.write(ev.get("text", ""))
                    st.caption(f"you · {ts}")

            elif kind == "bot_text":
                with st.chat_message("assistant", avatar="🔊"):
                    st.write(ev.get("text", ""))
                    st.caption(f"fweddie · {ts}")

            elif kind == "tool_call":
                st.caption(
                    f"🛠️ {ev.get('tool', 'tool')} "
                    f"({json.dumps(ev.get('args', {}), ensure_ascii=False)[:200]}) · {ts}"
                )

            elif kind == "tool_result":
                st.caption(
                    f"↳ {ev.get('tool', 'tool')}: "
                    f"{str(ev.get('result', ''))[:300]} · {ts}"
                )

            elif kind == "approval_requested":
                st.warning(
                    f"Approval needed for {ev.get('tool', 'a tool')}: "
                    f"{str(ev.get('command', ev.get('args', '')))[:200]}"
                )

            elif kind == "approval_granted":
                st.caption(f"✅ approved {ev.get('tool', '')} · {ts}")

            elif kind == "approval_denied":
                st.caption(f"🚫 denied {ev.get('tool', '')} · {ts}")

            elif kind == "session_start":
                st.caption(
                    f"▶ session started · LLM `{ev.get('llm')}` @ `{ev.get('base_url')}` · "
                    f"ASR `{ev.get('asr')}` · {ev.get('tools')} tools · {ts}"
                )

            elif kind == "session_end":
                st.caption(f"■ session ended · {ts}")

            elif kind == "user_started":
                st.caption(f"🎤 listening… · {ts}")

            elif kind == "bot_started":
                st.caption(f"🔊 speaking… · {ts}")

        st.caption(f"{len(events)} events · refreshes every second")

    render_log()
