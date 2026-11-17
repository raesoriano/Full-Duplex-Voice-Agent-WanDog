"""Fweddie -- Streamlit dashboard: a dark chat interface with voice and text.

Modelled on the Ollama chat window: one column of messages, a composer pinned at
the bottom, and the controls inside the composer rather than in a sidebar. The
differences are the ones this agent needs -- a microphone toggle that doubles as
the voice-agent indicator, and a Send button that becomes Stop while a turn is
in flight.

Three processes/files are involved, and they stay deliberately decoupled:

  * the agent appends one JSON object per line to ``config.DASHBOARD_LOG``; this
    page tails that file and renders it as a chat;
  * the mic toggle writes ``<log>.mic_control.json``, which the agent's
    ``MicControlWatcher`` polls;
  * Send writes ``<log>.typed.json`` and Stop writes ``<log>.stop.json``, which
    the agent's ``DashboardControlWatcher`` polls and turns into an
    ``LLMMessagesAppendFrame`` and an ``InterruptionFrame`` respectively.

Nothing here can break the voice loop: the worst case is that a control file is
written and no agent is running to read it.

The event vocabulary the agent emits is ``kind``-based, NOT role-based:

    session_start  {llm, base_url, asr, tools, ts}
    session_end    {}
    user_started   {}                      -- VAD fired
    user_partial   {text}                  -- live ASR draft, re-sent as the
                                              words arrive while you talk
    user_text      {text, source?}         -- final turn text ("typed" when it
                                              came from the text box)
    user_stopped   {}
    bot_started    {}
    bot_text       {text}                  -- the reply
    bot_stopped    {}
    interrupted    {source?}               -- Stop pressed, or a barge-in
    tool_call      {tool, args}
    tool_result    {tool, result}
    approval       {...}
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import streamlit as st

import config

LOG_PATH = Path(config.DASHBOARD_LOG)
MIC_CONTROL_FILE = Path(config.DASHBOARD_MIC_FILE)
TYPED_FILE = Path(config.DASHBOARD_TYPED_FILE)
STOP_FILE = Path(config.DASHBOARD_STOP_FILE)

ASSETS = Path(__file__).parent / "assets"
HUSKY = ASSETS / "husky.png"

st.set_page_config(
    page_title="Fweddie",
    page_icon=str(ASSETS / "husky-32.png") if (ASSETS / "husky-32.png").exists() else None,
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Palette and styling
# ---------------------------------------------------------------------------
# The theme in .streamlit/config.toml sets the base colours; this adds what a
# theme cannot express -- the composer's pill shape, the blue "listening" mic,
# and hiding the Streamlit chrome that makes a chat page look like a dashboard.
#
# Selectors are used defensively: anything that is only cosmetic is written so
# that a Streamlit release renaming a class degrades to "less pretty" rather
# than "unreadable". The colours that matter for legibility come from the theme.

BLUE = "#3b9eff"
GREY = "#8b8f96"

CSS = f"""
<style>
/* --- page frame ------------------------------------------------------- */
/* Selectors here were checked against Streamlit 1.67's own bundle, not
   guessed. Where a name could not be confirmed it is not used: a rule that
   silently matches nothing is worse than no rule, because it looks like the
   styling is in place. */
.block-container {{
    padding-top: 1.1rem;
    padding-bottom: 8rem;   /* room for the pinned composer */
    max-width: 800px;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer {{ visibility: hidden; }}

/* --- header ----------------------------------------------------------- */
.fw-head {{
    display: flex; align-items: center; gap: .6rem;
    margin: 0 0 1.2rem 0;
}}
.fw-head img {{ width: 30px; height: 30px; display: block; }}
.fw-head .fw-name {{ font-size: 1.02rem; font-weight: 600; letter-spacing: .01em; }}
.fw-head .fw-sub {{ font-size: .74rem; color: {GREY}; margin-top: -1px; }}

/* --- messages --------------------------------------------------------- */
/* The transcript is rendered as plain blocks rather than st.chat_message:
   that helper centres a 700 px column of its own and puts a fixed avatar
   beside every bubble, which fights the pinned composer. */
.fw-row {{ display: flex; gap: .7rem; margin: 0 0 1.15rem 0; align-items: flex-start; }}
.fw-av {{
    flex: 0 0 26px; width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .72rem; font-weight: 700; overflow: hidden;
}}
.fw-av img {{ width: 100%; height: 100%; object-fit: cover; }}
.fw-av-bot {{ background: #f2f3f5; }}
.fw-av-you {{ background: #26282c; color: #cfd3d8; }}
.fw-body {{ flex: 1 1 auto; min-width: 0; }}
.fw-who {{ font-size: .78rem; font-weight: 600; margin-bottom: .15rem; }}
.fw-text {{ font-size: .95rem; line-height: 1.55; white-space: pre-wrap; word-wrap: break-word; }}
.fw-meta {{ font-size: .7rem; color: {GREY}; margin-top: .2rem; }}
.fw-draft {{ color: {GREY}; font-style: italic; }}

/* A tool call is a quiet inset card: it is context, not conversation. */
.fw-tool {{
    border: 1px solid #26282c; border-radius: .6rem; padding: .55rem .7rem;
    margin: .2rem 0 .9rem 2.4rem; background: #101113;
}}
.fw-tool .fw-tool-name {{ font-size: .78rem; font-weight: 600; color: #cfd3d8; }}

/* --- composer --------------------------------------------------------- */
/* st.chat_input cannot hold a button beside it, so the composer is a container
   of columns styled into a pill and pinned to the bottom of the viewport --
   that is what makes it read as a chat box rather than a form at the end of a
   page. The keyed container gives us .st-key-composer for free. */
.st-key-composer {{
    position: fixed; left: 0; right: 0; bottom: 0; z-index: 100;
    padding: .85rem 1rem 1.05rem 1rem;
    background: linear-gradient(to top, #0a0a0a 62%, rgba(10,10,10,0));
}}
.st-key-composer > div {{ max-width: 800px; margin: 0 auto; }}
.st-key-composer [data-testid="stHorizontalBlock"] {{
    align-items: center; gap: .35rem;
    background: #141416;
    border: 1px solid #2a2c30;
    border-radius: 1.4rem;
    padding: .28rem .38rem;
}}
/* The text field loses its own box: the pill above is the box. */
.st-key-composer [data-testid="stTextInputRootElement"] {{
    background: transparent; border: none; box-shadow: none;
}}
.st-key-composer [data-testid="stTextInputRootElement"] input {{
    background: transparent; border: none; box-shadow: none;
    padding-left: .3rem;
}}
.st-key-composer [data-testid="stTextInputRootElement"] input:focus {{ box-shadow: none; }}

/* Buttons: circular and quiet. data-testid="stBaseButton-secondary" is the
   name Streamlit 1.67 actually emits for a plain st.button (confirmed in its
   bundle: data-testid is stBaseButton-<kind>); a bare `button` element
   selector is kept as the fallback so the shape survives a rename. */
.st-key-composer button,
.st-key-composer [data-testid="stBaseButton-secondary"] {{
    border-radius: 50%; min-height: 2.3rem; min-width: 2.3rem;
    height: 2.3rem; width: 2.3rem; padding: 0;
    border: 1px solid #2a2c30; background: #1b1d21; color: #e8eaed;
    font-size: .95rem; line-height: 1;
}}
.st-key-composer button:hover {{ background: #23262b; border-color: #3a3d42; }}

/* The per-button rules below repeat the [data-testid=...] attribute selector
   on purpose. The shared rule above is two classes deep
   (.st-key-composer + [data-testid]) and so outranks a plain
   ".st-key-sendbtn button" (one class + one element) -- the first version of
   this stylesheet set the Send background and watched it lose, leaving Send
   the same grey as everything else. Matching the attribute selector makes the
   specificity equal, and these come later, so they win. */

/* Send: light, because it is the primary action. */
.st-key-sendbtn button,
.st-key-sendbtn [data-testid="stBaseButton-secondary"] {{
    background: #e8eaed; border-color: #e8eaed; color: #0a0a0a;
}}
.st-key-sendbtn button:hover {{ background: #ffffff; border-color: #ffffff; }}

/* Stop: red, so it can never be mistaken for Send. */
.st-key-stopbtn button,
.st-key-stopbtn [data-testid="stBaseButton-secondary"] {{
    background: #b3261e; border-color: #b3261e; color: #ffffff;
}}
.st-key-stopbtn button:hover {{ background: #d1342a; border-color: #d1342a; }}

/* Mic: monochrome when idle. The blue "accepting audio" state is a separate
   rule emitted only while the mic is on (see mic_button) rather than a class
   toggled on the container -- Streamlit owns that container's class list, so
   there is no supported way to add one to it. */
.st-key-micbtn button,
.st-key-micbtn [data-testid="stBaseButton-secondary"] {{ color: {GREY}; }}

/* Placeholders for the features that come later (file upload, model picker).
   Disabled buttons so the layout is final and the labels are visible, but
   nothing pretends to work. */
.st-key-attachph button, .st-key-modelph button,
.st-key-attachph [data-testid="stBaseButton-secondary"],
.st-key-modelph [data-testid="stBaseButton-secondary"] {{
    color: #5c6067 !important; border-style: dashed !important;
    cursor: not-allowed;
}}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def read_events(limit: int = 200) -> list:
    """Return the last ``limit`` events, oldest first.

    Tolerates a line being written at the moment we read it (JSONDecodeError ->
    skip). Consecutive ``user_partial`` events collapse to the newest one: each
    supersedes the last, so keeping the history would scroll the conversation
    away. A draft is dropped entirely once a later ``user_text`` arrives -- that
    final transcript is the same turn, spelled correctly.
    """
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
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("kind") == "user_partial" and events and events[-1].get("kind") == "user_partial":
            events[-1] = ev
            continue
        events.append(ev)

    last_user_text = max(
        (i for i, e in enumerate(events) if e.get("kind") == "user_text"), default=-1
    )
    if last_user_text >= 0:
        events = [
            e
            for i, e in enumerate(events)
            if not (e.get("kind") == "user_partial" and i < last_user_text)
        ]
    return events


def mic_state() -> bool:
    try:
        return bool(json.loads(MIC_CONTROL_FILE.read_text(encoding="utf-8"))["enabled"])
    except Exception:
        return True


def set_mic_state(enabled: bool) -> None:
    MIC_CONTROL_FILE.parent.mkdir(parents=True, exist_ok=True)
    MIC_CONTROL_FILE.write_text(json.dumps({"enabled": enabled}), encoding="utf-8")


def _drop(path: Path, payload: dict) -> None:
    """Write a control file the agent consumes. Best effort: a failure here must
    not raise into the page, because the agent being down is not an error the
    person can act on."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    except OSError:
        pass


def send_typed(text: str) -> None:
    """Hand the typed message to the agent.

    The ``id`` lets the agent recognise a file it has already acted on, in case
    it could not delete it. It is not used for ordering -- the agent consumes
    the file, so one message is one write.
    """
    _drop(TYPED_FILE, {"id": uuid.uuid4().hex, "text": text, "ts": time.time()})


def request_stop() -> None:
    _drop(STOP_FILE, {"id": uuid.uuid4().hex, "ts": time.time()})


def fmt_time(ts) -> str:
    try:
        return time.strftime("%H:%M", time.localtime(float(ts)))
    except (TypeError, ValueError):
        return ""


# ---------------------------------------------------------------------------
# Turn state
# ---------------------------------------------------------------------------
# The Send button becomes Stop while a turn is in flight. That state is derived
# from the tail of the event log rather than tracked in the browser, because the
# agent -- not the page -- decides when a turn begins and ends: a voice turn can
# start while the page is open, and a page reload must not lose it.
#
# The staleness guard matters. If the agent is killed mid-turn no bot_stopped
# ever arrives, and without a timeout the button would be stuck on Stop forever
# with nothing to stop.

TURN_STALE_SECS = 90.0


def turn_in_flight(events: list) -> bool:
    """True when the agent is working on a turn the person could still stop."""
    busy = False
    last_ts = None
    for ev in events:
        kind = ev.get("kind")
        if kind in ("user_text", "bot_started", "tool_call", "user_started"):
            busy = True
        elif kind in ("bot_stopped", "session_end", "interrupted"):
            busy = False
        if kind in ("user_text", "bot_started", "bot_stopped", "tool_call", "interrupted"):
            last_ts = ev.get("ts")
    if busy and last_ts is not None:
        try:
            if time.time() - float(last_ts) > TURN_STALE_SECS:
                return False
        except (TypeError, ValueError):
            pass
    return busy


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

husky_uri = ""
if HUSKY.exists():
    import base64

    husky_uri = "data:image/png;base64," + base64.b64encode(HUSKY.read_bytes()).decode()

events = read_events()
session = next((e for e in reversed(events) if e.get("kind") == "session_start"), None)
subtitle = "voice + text"
if session:
    bits = []
    if session.get("llm"):
        bits.append(str(session["llm"]))
    if session.get("asr"):
        bits.append(str(session["asr"]))
    if bits:
        subtitle = " · ".join(bits)

st.markdown(
    f"""<div class="fw-head">
      <img src="{husky_uri}" alt="" />
      <div>
        <div class="fw-name">Fweddie</div>
        <div class="fw-sub">{subtitle}</div>
      </div>
    </div>""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


def render_event(ev: dict) -> None:
    kind = ev.get("kind", "")
    ts = fmt_time(ev.get("ts"))

    if kind == "user_text":
        source = ev.get("source")
        who = "You · typed" if source == "typed" else "You"
        st.markdown(
            f"""<div class="fw-row">
              <div class="fw-av fw-av-you">Y</div>
              <div class="fw-body">
                <div class="fw-who">{who}</div>
                <div class="fw-text">{_esc(ev.get("text", ""))}</div>
                <div class="fw-meta">{ts}</div>
              </div></div>""",
            unsafe_allow_html=True,
        )

    elif kind == "user_partial":
        # The draft: re-emitted every ~0.3 s while you are still talking. Shown
        # as one line, and dropped by read_events() once the final arrives.
        st.markdown(
            f"""<div class="fw-row">
              <div class="fw-av fw-av-you">Y</div>
              <div class="fw-body"><div class="fw-text fw-draft">
                {_esc(ev.get("text", ""))}…</div></div></div>""",
            unsafe_allow_html=True,
        )

    elif kind == "bot_text":
        st.markdown(
            f"""<div class="fw-row">
              <div class="fw-av fw-av-bot"><img src="{husky_uri}" alt="" /></div>
              <div class="fw-body">
                <div class="fw-who">Fweddie</div>
                <div class="fw-text">{_esc(ev.get("text", ""))}</div>
                <div class="fw-meta">{ts}</div>
              </div></div>""",
            unsafe_allow_html=True,
        )

    elif kind == "tool_call":
        args = ev.get("args", "")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        body = f'<div class="fw-tool-name">{_esc(ev.get("tool", "tool"))}</div>'
        if args:
            body += f'<div class="fw-meta">{_esc(args[:300])}</div>'
        body += f'<div class="fw-meta">called · {ts}</div>'
        st.markdown(f'<div class="fw-tool">{body}</div>', unsafe_allow_html=True)

    elif kind == "tool_result":
        result = str(ev.get("result", ""))
        st.markdown(
            f"""<div class="fw-tool">
              <div class="fw-tool-name">{_esc(ev.get("tool", "tool"))} — result</div>
              <div class="fw-meta">{len(result)} chars · {ts}</div>
            </div>""",
            unsafe_allow_html=True,
        )
        # The payload itself goes in a code block so it can be copied; a long
        # one scrolls instead of pushing the conversation off the screen.
        st.code(result[:4000], language="json" if result[:1] in "{[" else None)

    elif kind == "tool_turn":
        bits = [f"{ev.get('ms', 0)} ms"]
        if ev.get("truncated"):
            bits.append(
                f"model saw {ev.get('model_chars', 0)} of {ev.get('result_chars', 0)} chars"
            )
        st.markdown(
            f'<div class="fw-meta" style="margin-left:2.4rem">'
            f'↳ {_esc(ev.get("tool", "tool"))}: {" · ".join(bits)}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "interrupted":
        who = ev.get("source")
        label = "stopped" if who == "dashboard" else "interrupted"
        st.markdown(
            f'<div class="fw-meta" style="margin-left:2.4rem">⏹ {label} · {ts}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "approval_requested":
        st.warning(
            f"Approval needed for {ev.get('tool', 'a tool')}: "
            f"{str(ev.get('command', ev.get('args', '')))[:200]}"
        )

    elif kind == "approval_granted":
        st.markdown(
            f'<div class="fw-meta">✅ approved {_esc(ev.get("tool", ""))} · {ts}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "approval_denied":
        st.markdown(
            f'<div class="fw-meta">⛔ denied {_esc(ev.get("tool", ""))} · {ts}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "user_started":
        st.markdown(
            f'<div class="fw-meta" style="margin-left:2.4rem">🎤 listening… · {ts}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "bot_started":
        st.markdown(
            f'<div class="fw-meta" style="margin-left:2.4rem">🔊 speaking… · {ts}</div>',
            unsafe_allow_html=True,
        )


def _esc(text) -> str:
    """Escape for the HTML blocks above.

    Everything rendered through st.markdown(unsafe_allow_html=True) is user or
    model text, so a reply containing '<' would otherwise be parsed as markup
    and silently vanish. html.escape is the whole fix; it is defined here rather
    than imported at the top only because it is used nowhere else.
    """
    import html

    return html.escape(str(text))


@st.fragment(run_every=1.0)
def transcript() -> None:
    events = read_events()
    if not events:
        if not LOG_PATH.exists():
            st.markdown(
                '<div class="fw-meta">Waiting for the agent. Start it with '
                "<code>.venv/bin/python agent.py</code>, then speak or type.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="fw-meta">The log is empty. Is the agent running?</div>',
                unsafe_allow_html=True,
            )
        return

    # Everything except the session bookkeeping is conversation.
    for ev in events:
        if ev.get("kind") in ("session_start", "session_end"):
            continue
        render_event(ev)


transcript()


# ---------------------------------------------------------------------------
# Composer: mic toggle, text box, Send/Stop, and the two placeholders
# ---------------------------------------------------------------------------
# st.chat_input() would be the obvious choice for the text box, but it cannot
# hold anything beside it -- no mic button, no Send/Stop, no future file or model
# controls. So the composer is a container of columns, styled into a pill, and
# pinned to the bottom of the viewport by the CSS above.
#
# The button order is Ollama's: the toggle that affects how the prompt is sent
# sits on the LEFT of the input, the send control on the right. Ollama puts web
# search there; this puts the microphone there.

ss = st.session_state

# Clearing a keyed text_input cannot be done in the same run that rendered it
# (Streamlit refuses to modify a widget's state after instantiation), so the
# Send handler raises a flag and the next run clears the box before it renders.
if ss.pop("clear_box", False):
    ss["txt"] = ""

if "mic_on" not in ss:
    ss["mic_on"] = mic_state()

busy = turn_in_flight(read_events())


def mic_button() -> None:
    live = ss["mic_on"]
    # A plain button rather than st.toggle: the switch widget cannot be made to
    # look like Ollama's inline icon button, and this state has to be obvious at
    # a glance -- blue means the agent is accepting audio.
    #
    # The blue is applied by a class on the container below, not by type
    # ("primary"), because type expresses emphasis and this expresses state.
    with st.container(key="micbtn"):
        clicked = st.button(
            "🎤" if live else "🔇",
            key="micbtn_btn",
            help=(
                "Listening — the agent is accepting audio. Click to mute."
                if live
                else "Muted — the agent will not transcribe. Click to listen."
            ),
        )
    if clicked:
        ss["mic_on"] = not live
        set_mic_state(ss["mic_on"])


with st.container(key="composer"):
    c_attach, c_mic, c_text, c_send, c_model = st.columns(
        [0.7, 0.7, 8.2, 0.9, 0.7], vertical_alignment="center"
    )

    with c_attach:
        # Placeholder: file upload comes later.
        with st.container(key="attachph"):
            st.button(
                "📎",
                key="attachph_btn",
                disabled=True,
                help="Attach a file (coming soon)",
            )

    with c_mic:
        mic_button()

    with c_text:
        typed = st.text_input(
            "Message",
            key="txt",
            label_visibility="collapsed",
            placeholder="Message Fweddie, or just start talking…",
        )

    with c_send:
        if busy:
            # Wrapped in a keyed container so the CSS can reach it: a
            # container keyed "stopbtn" becomes .st-key-stopbtn, which is how
            # the red styling is applied without depending on widget order.
            with st.container(key="stopbtn"):
                stop_clicked = st.button(
                    "■", key="stopbtn_btn", help="Stop this turn (like barge-in)"
                )
            if stop_clicked:
                request_stop()
                # Reflect the intent immediately: waiting for the agent's
                # interrupted event would leave the button looking unresponsive
                # for up to one poll interval.
                st.rerun()
        else:
            with st.container(key="sendbtn"):
                send_clicked = st.button(
                    "➤", key="sendbtn_btn", help="Send (or press Enter)"
                )
            if send_clicked:
                text = (typed or "").strip()
                if text:
                    send_typed(text)
                    ss["clear_box"] = True
                    st.rerun()

    with c_model:
        # Placeholder: model selection comes later.
        with st.container(key="modelph"):
            st.button(
                "⌄",
                key="modelph_btn",
                disabled=True,
                help="Choose a model (coming soon)",
            )

# Blue while the agent is accepting audio, monochrome while muted.
#
# Emitted as its own rule, only when live, instead of a class toggled on the
# container: Streamlit builds that container's class list from the widget key
# and offers no supported way to add to it, so a ".fw-live" selector would have
# silently matched nothing and the mic would have stayed grey in both states.
if ss["mic_on"]:
    st.markdown(
        "<style>.st-key-micbtn button{"
        "background:#3b9eff !important;border-color:#3b9eff !important;"
        "color:#04121f !important;}"
        ".st-key-micbtn button:hover{"
        "background:#57aeff !important;border-color:#57aeff !important;}</style>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------
# Kept minimal on purpose: this is a chat window, not a debug console. The paths
# are here because "the dashboard is up but shows nothing" is almost always a
# log-path mismatch between the two processes, and that is worth being able to
# see at a glance.

tail = read_events(limit=5)
if tail:
    st.caption(
        f"{len(read_events())} events · {fmt_time(tail[-1].get('ts'))} · "
        f"`{LOG_PATH}`"
    )
