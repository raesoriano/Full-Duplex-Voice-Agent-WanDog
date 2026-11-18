"""WanDog -- Streamlit dashboard: a chat interface with voice and text.

Modelled on the Ollama chat window: one column of messages, a composer pinned at
the bottom, and the controls inside the composer rather than in a sidebar. The
differences are the ones this agent needs -- a microphone toggle that doubles as
the voice-agent indicator, a Send button that becomes Stop while a turn is in
flight, and a light/dark switch.

The palette and the theme switch are adapted from the OnIt web interface
(~/onit/src/ui/static/style.css and app.js): a warm paper-white light theme and a
warm charcoal dark theme, both driven by CSS custom properties that a
``data-theme`` attribute on ``<html>`` flips.

Why the theme is switched in the browser and not in Streamlit
--------------------------------------------------------------
Streamlit's own theme is fixed when the server starts. ``st._config.set_option
("theme.base", ...)`` from a running script does NOT repaint -- measured: set to
"light", the page stayed rgb(10,10,10) through a click *and* a full reload, and
``st.context.theme`` kept reporting the old value (its docstring admits it may be
wrong around a theme change). So the theme here is a set of CSS variables plus a
``data-theme`` attribute set from a component iframe, which is exactly what the
OnIt UI does with its own ``data-theme``. Measured after a toggle: app background
rgb(250,249,245) <-> rgb(38,38,36), text and code-block colours following.

Four processes/files are involved, and they stay deliberately decoupled:

  * the agent appends one JSON object per line to ``config.DASHBOARD_LOG``; this
    page tails that file and renders it as a chat;
  * the mic toggle writes ``<log>.mic_control.json``, which the agent's
    ``MicControlWatcher`` polls;
  * Send writes ``<log>.typed.json``, Stop writes ``<log>.stop.json`` and Clear
    writes ``<log>.clear.json``, which the agent's ``DashboardControlWatcher``
    polls and turns into an ``LLMMessagesAppendFrame``, an ``InterruptionFrame``
    and an ``LLMMessagesUpdateFrame(messages=[])`` respectively.

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
    cleared        {}                      -- Clear pressed; history starts here
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
import session as sess

LOG_PATH = Path(config.DASHBOARD_LOG)
MIC_CONTROL_FILE = Path(config.DASHBOARD_MIC_FILE)
TYPED_FILE = Path(config.DASHBOARD_TYPED_FILE)
STOP_FILE = Path(config.DASHBOARD_STOP_FILE)
CLEAR_FILE = Path(config.DASHBOARD_CLEAR_FILE)
UPLOAD_FILE = Path(config.DASHBOARD_UPLOAD_FILE)

# Upload ids this process has already stored. A full-app rerun (Send, Clear,
# the theme toggle) rebuilds this script from the top, so this cannot live in
# session_state -- and re-storing the same bytes would duplicate the file and
# ask the source-of-truth question out loud a second time.
_SEEN_UPLOADS: set = set()

ASSETS = Path(__file__).parent / "assets"
# The three icons in assets/, committed as-is:
#   husky-idle  -- mouth open, tongue out. The bar's resting state, and the
#                  avatar beside every reply.
#   husky-think -- muzzle up. The bar's face while a turn is in flight.
#   husky-32    -- the favicon.
# Rasters, not SVG: tracing the drawings produced blobs (the eyes fused with the
# muzzle). See "The husky icons are three committed PNGs" in WanDog-Notes.md.
HUSKY_IDLE_PNG = ASSETS / "husky-idle.png"
HUSKY_THINK_PNG = ASSETS / "husky-think.png"

st.set_page_config(
    page_title="WanDog",
    page_icon=str(ASSETS / "husky-32.png") if (ASSETS / "husky-32.png").exists() else None,
    layout="centered",
    # Expanded: the session list is the feature, and a rail nobody opens is a
    # feature nobody has. Streamlit remembers the user collapsing it.
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
# Two complete palettes rather than one set of colours with the odd override.
# The OnIt UI does the same (its :root and [data-theme="dark"] blocks each define
# every variable), and it is what makes a switch look deliberate: nothing is
# left inheriting a colour from the other theme.
#
# Values copied from ~/onit/src/ui/static/style.css so the two interfaces read as
# the same family -- warm off-white paper in light, warm charcoal in dark, a
# terracotta accent, and muted greys for the secondary text.

THEMES = {
    "light": {
        "bg": "#faf9f5",
        "panel": "#f0eee6",
        "hover": "#e8e6dc",
        "active": "#e0ddd1",
        "input": "#ffffff",
        "bubble": "#f0eee6",
        "code": "#f6f6f4",
        "border": "#dedcd1",
        "border-strong": "#c9c6b8",
        "text": "#1f1e1d",
        "muted": "#6e6d66",
        "faint": "#a3a196",
        "accent": "#c96442",
        "accent-hover": "#b4522f",
        "on-accent": "#ffffff",
        "danger": "#c0392b",
        "shadow": "0 2px 12px rgba(0, 0, 0, 0.06)",
    },
    "dark": {
        "bg": "#262624",
        "panel": "#1f1e1c",
        "hover": "#32312e",
        "active": "#3a3835",
        "input": "#30302e",
        "bubble": "#33322f",
        "code": "#1e1d1b",
        "border": "#3d3c38",
        "border-strong": "#4c4a45",
        "text": "#eceae4",
        "muted": "#a5a396",
        "faint": "#706e64",
        "accent": "#d97757",
        "accent-hover": "#e08b6e",
        "on-accent": "#ffffff",
        "danger": "#e07b6a",
        "shadow": "0 2px 12px rgba(0, 0, 0, 0.35)",
    },
}

# The microphone is blue in both themes while the agent is accepting audio -- it
# is a state, not decoration -- but not the SAME blue. One blue cannot be legible
# on both a near-black and a near-white background: #3b9eff is 5.4:1 on the dark
# page and only 2.7:1 on the light one, under the 3:1 floor for a graphical
# control. So each palette carries its own shade, picked to clear that floor in
# its own theme (5.4:1 dark, 4.7:1 light).
MIC_LIVE = {"light": "#0b6fd4", "dark": "#3b9eff"}
MIC_LIVE_HOVER = {"light": "#0a5cb0", "dark": "#57aeff"}


def _var_block(palette: dict) -> str:
    return "".join(f"--fw-{k}: {v}; " for k, v in palette.items())


# ---------------------------------------------------------------------------
# Which theme is active
# ---------------------------------------------------------------------------
# Resolved BEFORE any styling is emitted, because the palette that goes into
# :root is the active one -- that is what makes the first paint correct.
#
# The choice lives in the URL (``?theme=dark``) rather than only in
# session_state, for two reasons: it survives a reload, and the server can read
# it, so the toggle shows the right glyph on the first render. localStorage
# cannot do the second -- the browser could repaint itself from it, but Python
# would not know, and the button would show the wrong icon.
#
# Why not Streamlit's own theme setting: it is fixed when the server starts.
# st._config.set_option("theme.base", ...) from a running script does NOT
# repaint. Measured on Streamlit 1.67: set to "light", the page stayed
# rgb(10,10,10) through the click AND through a full reload, while
# st.context.theme kept reporting the old value (its docstring admits it may be
# wrong around a theme change). So the theme here is CSS variables the server
# emits, which is also how the OnIt UI does it.

VALID_THEMES = ("light", "dark")
# Light is the default: a first-time reader gets the paper-white page, and the
# switch is there for whoever wants the dark one. It has to agree with
# .streamlit/config.toml (base = "light", same palette), which paints the frame
# before this stylesheet exists -- if the two disagreed, the page would flash the
# other theme on every load.
DEFAULT_THEME = "light"

ss = st.session_state
if "theme" not in ss:
    wanted = st.query_params.get("theme")
    ss["theme"] = wanted if wanted in VALID_THEMES else DEFAULT_THEME
theme = ss["theme"]
_ACTIVE = theme if theme in THEMES else DEFAULT_THEME


# ---------------------------------------------------------------------------
# Which session is this page showing?
# ---------------------------------------------------------------------------
# Two different questions, and the sidebar is what separates them:
#
#   LIVE    -- the session the agent is in. There is exactly one, and it is
#              whatever .current names. It is what a new turn goes into.
#   VIEWED  -- the session this page is displaying. Usually the live one, but
#              clicking an older session in the sidebar makes it something else.
#
# They have to be separate because browsing is not switching. Reading an old
# session must not move the agent into it: the person may be looking for what
# they said last week while continuing today's conversation. So VIEWED is a
# read-only view of a folder, and only an explicit "switch" moves the agent.
#
# VIEWED is held in session_state, not recomputed from the pointer each rerun,
# or the first poll would yank the reader back to the live session. It is
# validated on every rerun -- a session deleted underneath the page falls back
# to the live one rather than rendering an empty folder forever.
def _live_id() -> str:
    return sess.read_pointer() or ""


def _viewed_id() -> str:
    """The session this page is displaying, guaranteed to exist.

    Falls back to the live session when nothing is chosen, when the chosen id
    is malformed, or when its folder has gone. Falling back rather than raising
    is the recoverable behaviour: a page that cannot decide what to show has
    nothing useful to say.
    """
    want = ss.get("viewing") or ""
    if want:
        try:
            if sess.session_dir(want).is_dir():
                return want
        except ValueError:
            pass
        ss.pop("viewing", None)
    return _live_id()


LIVE_ID = _live_id()
VIEWED_ID = _viewed_id()
# True when the page is looking at something other than the live session. The
# transcript uses it to say so, and the composer uses it to refuse to type into
# a session the agent is not in -- a message sent from a browsed view would
# arrive in a different conversation than the one on screen.
BROWSING = bool(VIEWED_ID) and VIEWED_ID != LIVE_ID
VIEWED_LOG = Path(sess.events_path(VIEWED_ID)) if VIEWED_ID else LOG_PATH


# The static rules below use var(--fw-*) everywhere, so they are written as a
# plain string: nothing here needs Python interpolation, which keeps the braces
# of the CSS out of an f-string. Only the two variable blocks are generated.
_BASE_CSS = """
/* --- page frame ------------------------------------------------------- */
/* Selectors were checked against Streamlit 1.67's own bundle (the data-testid
   names come from the rendered DOM, not from documentation). Anything purely
   cosmetic degrades to "less pretty" if a name changes; the colours that matter
   for legibility come from the variables above. */
html, body, .stApp,
[data-testid="stAppViewContainer"], [data-testid="stMain"],
[data-testid="stMainBlockContainer"] {
    background: var(--fw-bg) !important;
    color: var(--fw-text) !important;
    font-family: var(--fw-font) !important;
}
/* Streamlit's own header is an overlay: `position:absolute; top:0; height:60px;
   z-index:999990` (measured), and it sits ON TOP of the page -- elementFromPoint
   at y=20 returned it, so it would swallow clicks on a top bar placed there. It
   is emptied by our own CSS anyway, so it is removed from the layout and the
   page's own bar takes the top of the screen. */
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"], [data-testid="stAppDeployButton"] { display: none !important; }
#MainMenu, footer { visibility: hidden; }

/* The page does not scroll; the transcript does.
   -------------------------------------------------
   Measured before this change: [data-testid="stMain"] is the scroller
   (overflow-y:auto, clientHeight 657, scrollHeight 936), and the composer is
   pinned over the bottom of it with a background gradient. That is why the
   scrollbar "disappeared" at the end of the conversation: the thumb's travel
   runs to the bottom of stMain, and the last ~82 px of that travel is behind the
   composer's gradient, which fades it out. The thumb was still there -- the last
   6 px column of a screenshot showed it fading from rgb(115,115,111) at y=575 to
   rgb(38,38,36) at y=606 -- it had simply been painted over.

   The fix is to give the transcript a scroller of its own that ends ABOVE the
   composer, so the thumb's travel is never under it. */
.block-container {
    /* No vertical padding: the transcript carries its own margin below the bar
       (see .st-key-transcript), so the gap is one measured number rather than a
       padding here plus an unknown block gap inside. Padding it here left 60 px
       of dead space between the bar and the first message. */
    padding-top: 0;
    padding-bottom: 0;
    /* No HORIZONTAL padding either, and that is what lets the transcript reach
       the main area's edge. Measured: with Streamlit's 1rem of side padding the
       scroller's content box was x=316..1084 inside a main area of 300..1100, so
       its scrollbar sat 16 px inboard -- and no negative margin fixes that,
       because `width: 100%` is already the padded width; the negative margin
       only shifts the box, it does not widen it.
       The reading column keeps its own 1rem of side padding (see the child rule
       below), so the text is still inset from the edges. */
    padding-left: 0;
    padding-right: 0;
    /* And no max-width, for the same reason. `max-width: 820px` here is what the
       old `width: 100vw` was working around: the container caps itself at 820
       and centres, so anything sized as a percentage of it stops short of the
       main area. Measured at 1280 px with the rail open: main area 300..1280 but
       the transcript 380..1200 -- an 80 px gap either side, which only looked
       right at 1100 because there the main area happened to BE 800 px wide.
       Nothing needs the cap: the bar's and composer's own children carry their
       own 820 px reading column, and the transcript's child does too. */
    max-width: none;
}

/* --- sidebar ---------------------------------------------------------- */
/* Streamlit's sidebar is a sibling of the main block, and it has its own
   background and border from Streamlit's theme -- which is NOT this page's
   theme (see the note at the top: the palette is ours, switched in the browser,
   and .streamlit/config.toml cannot follow it). Without these rules the rail is
   a white slab beside a charcoal page, which is the one thing a dark theme
   makes impossible to ignore.

   The collapse control is left alone: it is the only way back to the page on a
   narrow window, and hiding it would strand the sidebar open. */
[data-testid="stSidebar"] {
    background: var(--fw-panel) !important;
    border-right: 1px solid var(--fw-border) !important;
}
[data-testid="stSidebar"] * { font-family: var(--fw-font) !important; }
[data-testid="stSidebar"] [data-testid="stSidebarHeader"] { padding: .5rem .6rem; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: .2rem; }
/* The rail's own vertical rhythm. Each session is a button, a caption and
   sometimes two more buttons, so the default block gap stacks up into a list
   that is mostly whitespace. */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: .15rem; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    margin: 0 0 .1rem .35rem; padding: 0; font-size: .66rem;
}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { margin: 0; }
/* A session row is a label, not a form control: left-aligned text reads as a
   list of things to pick, while centred text reads as a row of buttons. */
[data-testid="stSidebar"] .stButton button {
    justify-content: flex-start;
    text-align: left;
    padding: .32rem .55rem;
    font-size: .8rem;
    font-weight: 500;
    border-radius: .5rem;
    min-height: 0;
}
/* The switch/notes pair under a row is secondary to the row itself, so it is
   quieter: smaller type, no fill until hovered. */
[data-testid="stSidebar"] [class*="st-key-sess_"] [data-testid="stHorizontalBlock"] .stButton button {
    justify-content: center;
    font-size: .68rem;
    padding: .18rem .4rem;
    color: var(--fw-muted);
}
.fw-side-head {
    font-size: .72rem; font-weight: 700; letter-spacing: .04em;
    text-transform: uppercase; color: var(--fw-faint);
    margin: .1rem 0 .35rem .35rem;
}
.fw-side-empty {
    font-size: .74rem; color: var(--fw-faint);
    line-height: 1.5; padding: 0 .35rem;
}

/* --- top bar ---------------------------------------------------------- */
/* Identity on the left, controls on the right, and it stays put: `position:
   fixed` rather than `sticky` because the page itself no longer scrolls, so
   there is nothing for a sticky element to stick to. The keyed container
   (.st-key-topbar) is ours; the bar's own background is opaque so the
   transcript scrolls UNDER it rather than through it. */
/* The bar spans the MAIN AREA, not the window. `left: 0` was right when the
   main block started at the window's edge; with the session rail open it would
   put the bar underneath the rail, hiding the identity and the Clear button
   behind it. --fw-rail is published by the layout script from the sidebar's
   measured width (see _AUTOSCROLL_JS) and falls back to 0, which is exactly the
   collapsed case -- so one rule covers both states. */
.st-key-topbar {
    position: fixed; top: 0; left: var(--fw-rail, 0px); right: 0; z-index: 200;
    /* `width: auto` is load-bearing with both left and right set. Measured:
       with a width of 100% (inherited from the block's own rule) the box was
       1100 px wide starting at x=300, so it ran to x=1400 in a 1100 px window
       and the bar's right-hand controls were off-screen. With left+right and
       width:auto the box is sized by the two offsets, which is what "span the
       main area" means. */
    width: auto;
    padding: .55rem 1rem .6rem 1rem;
    background: var(--fw-bg);
    border-bottom: 1px solid var(--fw-border);
}
.st-key-topbar > div { max-width: 820px; margin: 0 auto; }
/* The status caption is the last thing in the bar. Tighter than Streamlit's
   default caption spacing, and its own colour, because it is metadata. */
.st-key-topbar [data-testid="stCaptionContainer"] {
    margin: .1rem 0 0 0; padding: 0; font-size: .68rem;
}
.st-key-topbar [data-testid="stCaptionContainer"] p { margin: 0; }
/* No gap between the two columns: the identity block and the buttons are one
   row, and Streamlit's default column gap would push the buttons inwards. */
.st-key-topbar [data-testid="stHorizontalBlock"] { gap: .2rem; align-items: center; }

.fw-head {
    display: flex; align-items: center; gap: .6rem;
    margin: 0;
}
/* The face: two images stacked, one visible. Both are the same box, so the
   name plate never shifts when the icon changes, and both are already decoded
   when the swap happens (see the identity fragment). The idle face is the one
   with the tongue out; the thinking face is the one looking up. */
.fw-face { position: relative; width: 30px; height: 30px; flex: 0 0 30px; display: block; }
.fw-face img {
    position: absolute; inset: 0; width: 30px; height: 30px;
    display: block; object-fit: contain;
    transition: opacity .18s ease;
}
/* The thinking face is a little wider than it is tall, so it is inset by a
   pixel to read the same size as the idle one. */
.fw-face .fw-face-think { opacity: 0; transform: scale(.96); }
.fw-head.fw-busy .fw-face-idle { opacity: 0; }
.fw-head.fw-busy .fw-face-think { opacity: 1; transform: scale(1); }
.fw-head .fw-name { font-size: 1.02rem; font-weight: 600; letter-spacing: .01em; }
.fw-head .fw-sub { font-size: .74rem; color: var(--fw-muted); margin-top: -1px; }

/* The header's icon buttons (theme, clear). Quiet by default -- they are
   secondary to the conversation -- and they only show a surface on hover, the
   way the OnIt header's buttons behave. */
.st-key-themebtn button, .st-key-clearbtn button,
.st-key-themebtn [data-testid="stBaseButton-secondary"],
.st-key-clearbtn [data-testid="stBaseButton-secondary"] {
    border-radius: 50%; min-height: 2.2rem; min-width: 2.2rem;
    height: 2.2rem; width: 2.2rem; padding: 0;
    border: 1px solid transparent; background: transparent;
    color: var(--fw-muted); font-size: 1rem; line-height: 1;
}
.st-key-themebtn button:hover, .st-key-clearbtn button:hover {
    background: var(--fw-hover); border-color: var(--fw-border);
    color: var(--fw-text);
}

/* --- transcript scroller ---------------------------------------------- */
/* The transcript is its own scroll box, sized to end above the composer, so a)
   the thumb never travels behind the composer's gradient (which is what made it
   look like the scrollbar vanished), and b) opening the page can put the view at
   the newest message without scrolling the whole document.

   st.fragment renders into .st-key-transcript, and Streamlit's own inner
   vertical block is the box that scrolls.

   The height is NOT a hard-coded rem subtraction. It was, and it kept being
   wrong: the bar's height depends on whether the status line is present, on the
   theme's line heights and on Streamlit's own block padding, so every attempt to
   guess it in rem left the transcript either under the composer or floating
   above it (measured 607 px bottom against a composer at 575, i.e. 32 px of the
   transcript hidden behind it). A one-line script in the bar measures the two
   real edges and sets --fw-scroll-h on :root; the CSS falls back to a viewport
   calculation only until that script has run. */
.st-key-transcript {
    /* Both height and max-height: Streamlit's chain of column flex containers
       gives this block flex-grow:1, so `height` alone is a basis the box grows
       past. max-height is the one that actually caps it (measured: height
       calc(-11.1rem + 100vh) = 479 px, used height 631 px, until max-height was
       set as well). */
    margin-top: var(--fw-top-pad, 7.4rem);
    height: var(--fw-scroll-h, calc(100vh - 13.5rem));
    max-height: var(--fw-scroll-h, calc(100vh - 13.5rem));

    /* FULL BLEED, so the scrollbar lands on the window's right edge.
       -----------------------------------------------------------------
       Measured before this: the box was x=206..994 inside a 1200 px window,
       because it inherits Streamlit's 820 px centred column. The scrollbar
       therefore sat 206 px from the right edge -- not where a window's own
       scrollbar is, and not where the reader's eye expects it after scrolling a
       full-width page.
       The box is now pulled out to the viewport (negative margin of exactly the
       column's own left offset, which is what `50% - 50vw` computes) and the
       CONTENT is put back in an 820 px column by the child rule below. One
       scroller, still, and it ends above the composer -- the thumb never
       travels under the composer's gradient. */
    /* FULL BLEED TO THE MAIN AREA'S EDGES, not to the viewport's.
       ---------------------------------------------------------
       `100vw` and `calc(50% - 50vw)` were written when the main block started at
       the window's left edge. With the session rail open it does not: the block
       begins one sidebar-width in, so 100vw overhangs the rail by exactly that
       much and `50% - 50vw` pulls the box under it -- the transcript would run
       behind the sidebar, and its scrollbar would sit off-screen to the right.

       100% is the containing block's width, which is the main area, so the box
       spans it and stops at the rail. The negative margin is then unnecessary:
       there is nothing to pull out from. The scrollbar still lands on the main
       area's right edge, which is the window's edge when the rail is closed and
       the rail's edge when it is open -- in both cases, the edge of the space
       the conversation actually occupies. */
    width: 100%;
    /* max-width is required, not decoration: Streamlit's own .stVerticalBlock
       carries `max-width: 100%`, which would otherwise clamp the box back to the
       820 px column (measured before this: width 788 px, the column's content
       box, with the box starting at x=0 and ending at 788 in a 1200 px window). */
    max-width: 100% !important;
    margin-left: 0;
    margin-right: 0;
    box-sizing: border-box;
    padding-left: 0;
    padding-right: 0;
    overflow-y: auto;
    overflow-x: hidden;
    /* The gutter is stable so the text does not shift sideways by half a line
       when the transcript grows past one screen. */
    scrollbar-gutter: stable;
}
/* The reading column. The scroller's child is Streamlit's layout wrapper (not
   the vertical block -- that is one level further in, and a selector for it here
   matched nothing), and constraining the wrapper is what keeps the text at the
   width of the bar and the composer while the scroller itself is full width. */
.st-key-transcript > [data-testid="stLayoutWrapper"] {
    max-width: 820px;
    margin-left: auto;
    margin-right: auto;
    padding-left: 1rem;
    padding-right: 1rem;
}
/* Keep the newest message clear of the composer even when the view is at the
   very bottom of the scroll. */
.st-key-transcript [data-testid="stVerticalBlock"] > div:last-child { padding-bottom: .4rem; }

/* --- messages --------------------------------------------------------- */
/* The transcript is rendered as plain blocks rather than st.chat_message: that
   helper centres a 700 px column of its own and puts a fixed avatar beside every
   bubble, which fights the pinned composer. */
.fw-row { display: flex; gap: .7rem; margin: 0 0 1.15rem 0; align-items: flex-start; }
.fw-av {
    flex: 0 0 26px; width: 26px; height: 26px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: .72rem; font-weight: 700; overflow: hidden;
}
.fw-av img { width: 100%; height: 100%; object-fit: cover; }
.fw-av-bot { background: var(--fw-bubble); }
.fw-av-you { background: var(--fw-active); color: var(--fw-muted); }
.fw-body { flex: 1 1 auto; min-width: 0; }
.fw-who { font-size: .78rem; font-weight: 600; margin-bottom: .15rem; }
.fw-text { font-size: .95rem; line-height: 1.6; white-space: pre-wrap; word-wrap: break-word; }
.fw-meta { font-size: .7rem; color: var(--fw-faint); margin-top: .2rem; }
.fw-draft { color: var(--fw-muted); font-style: italic; }

/* A tool call is a quiet inset card: it is context, not conversation. */
.fw-tool {
    border: 1px solid var(--fw-border); border-radius: .6rem; padding: .55rem .7rem;
    margin: .2rem 0 .9rem 2.4rem; background: var(--fw-panel);
}
.fw-tool .fw-tool-name { font-size: .78rem; font-weight: 600; color: var(--fw-text); }

/* Tool payloads arrive through st.code so they can be copied; give them the
   palette's code surface instead of Streamlit's own. */
[data-testid="stCode"], [data-testid="stCode"] pre, [data-testid="stCode"] code {
    background: var(--fw-code) !important;
    color: var(--fw-text) !important;
    border: 1px solid var(--fw-border) !important;
    border-radius: .6rem !important;
}

/* --- composer --------------------------------------------------------- */
/* st.chat_input cannot hold a button beside it, so the composer is a container
   of columns styled into a pill and pinned to the bottom of the viewport --
   that is what makes it read as a chat box rather than a form at the end of a
   page. The keyed container gives us .st-key-composer for free. */
.st-key-composer {
    /* Same reason as the bar, including `width: auto`: it belongs to the main
       area, not the window, and a set width with both offsets over-constrains. */
    position: fixed; left: var(--fw-rail, 0px); right: 0; bottom: 0; z-index: 100;
    width: auto;
    padding: .85rem 1rem 1.05rem 1rem;
    background: linear-gradient(to top, var(--fw-bg) 62%, transparent);
}
.st-key-composer > div { max-width: 820px; margin: 0 auto; }
.st-key-composer [data-testid="stHorizontalBlock"] {
    align-items: center; gap: .35rem;
    background: var(--fw-input);
    border: 1px solid var(--fw-border-strong);
    border-radius: 1.4rem;
    padding: .3rem .4rem;
    box-shadow: var(--fw-shadow);
}
/* The text field loses its own box: the pill above is the box. Both selectors
   are kept -- the textarea's testid and the input's -- so a revert to the
   single-line widget would not silently get its own border back. */
.st-key-composer [data-testid="stTextInputRootElement"],
.st-key-composer [data-testid="stTextArea"] {
    background: transparent !important; border: none !important;
    box-shadow: none !important;
}
.st-key-composer [data-testid="stTextInputRootElement"] input,
.st-key-composer [data-testid="stTextArea"] textarea {
    background: transparent !important; border: none !important;
    box-shadow: none !important; color: var(--fw-text) !important;
    padding-left: .35rem;
}
.st-key-composer [data-testid="stTextInputRootElement"] input::placeholder,
.st-key-composer [data-testid="stTextArea"] textarea::placeholder {
    color: var(--fw-faint) !important;
}
.st-key-composer [data-testid="stTextInputRootElement"] input:focus,
.st-key-composer [data-testid="stTextArea"] textarea:focus {
    box-shadow: none !important;
}
/* A textarea arrives with a drag handle and its own resize; both break the
   pill. The box grows with the text instead (Streamlit auto-expands it).
   The resting height is pinned here rather than through the widget's `height`
   argument: that argument sets ROWS (height=44 still produced rows=3 and a
   66 px box), and the pill then stood 78 px tall against the 46 px single-line
   version it replaced. min-height is what the auto-expand measures from, so
   both values have to agree. */
.st-key-composer [data-testid="stTextArea"] textarea {
    resize: none !important;
    line-height: 1.45;
    height: 2.1rem !important;
    min-height: 2.1rem !important;
    padding-top: .35rem !important; padding-bottom: .35rem !important;
}
.st-key-composer [data-testid="stTextArea"] > div { border: none !important;
    background: transparent !important; box-shadow: none !important; }

/* Buttons: circular and quiet. data-testid="stBaseButton-secondary" is the name
   Streamlit 1.67 actually emits for a plain st.button; a bare `button` element
   selector is kept as the fallback so the shape survives a rename. */
.st-key-composer button,
.st-key-composer [data-testid="stBaseButton-secondary"] {
    border-radius: 50%; min-height: 2.3rem; min-width: 2.3rem;
    height: 2.3rem; width: 2.3rem; padding: 0;
    border: 1px solid var(--fw-border); background: var(--fw-panel);
    color: var(--fw-text); font-size: 1rem; line-height: 1;
}
.st-key-composer button:hover { background: var(--fw-hover); border-color: var(--fw-border-strong); }

/* The per-button rules below repeat the [data-testid=...] attribute selector on
   purpose. The shared rule above is two classes deep (.st-key-composer + ...)
   and so outranks a plain ".st-key-sendbtn button" (one class + one element) --
   the first version of this stylesheet set the Send background and watched it
   lose, leaving Send the same grey as everything else. Matching the attribute
   selector makes the specificity equal, and these come later, so they win. */

/* Send: the accent, because it is the primary action. */
.st-key-sendbtn button,
.st-key-sendbtn [data-testid="stBaseButton-secondary"] {
    background: var(--fw-accent); border-color: var(--fw-accent);
    color: var(--fw-on-accent);
}
.st-key-sendbtn button:hover { background: var(--fw-accent-hover); border-color: var(--fw-accent-hover); }

/* Stop: red, so it can never be mistaken for Send. */
.st-key-stopbtn button,
.st-key-stopbtn [data-testid="stBaseButton-secondary"] {
    background: var(--fw-danger); border-color: var(--fw-danger); color: #ffffff;
}
.st-key-stopbtn button:hover { filter: brightness(1.12); }

/* Mic: monochrome when idle -- the icon inherits the button's colour, which is
   why the live state can be expressed as one colour on the button. */
.st-key-micbtn button,
.st-key-micbtn [data-testid="stBaseButton-secondary"] { color: var(--fw-text); }

/* Placeholders for the features that come later (file upload, model picker).
   Disabled buttons so the layout is final and the labels are visible, but
   nothing pretends to work. */
.st-key-attachph button, .st-key-modelph button,
.st-key-attachph [data-testid="stBaseButton-secondary"],
.st-key-modelph [data-testid="stBaseButton-secondary"] {
    color: var(--fw-faint) !important; border-style: dashed !important;
    cursor: not-allowed;
}

/* --- status ----------------------------------------------------------- */
[data-testid="stCaptionContainer"] { color: var(--fw-faint) !important; }
[data-testid="stCaptionContainer"] code {
    background: var(--fw-panel) !important; color: var(--fw-muted) !important;
}

/* The theme-sync iframe: its only job is to run one line of JavaScript, so it
   must take no space. It cannot be height:0 (st.iframe rejects that), hence a
   1 px frame removed from the flow here. */
.st-key-themesync, .st-key-autoscrollsync { display: none !important; }
"""

_CSS = (
    "<style>\n"
    # The ACTIVE theme's palette goes on :root, so the first paint is already
    # correct -- no flash of the other theme while a script catches up. This is
    # the rule that actually paints.
    ":root { " + _var_block(THEMES[_ACTIVE]) + "}\n"
    # Both palettes are also defined against the attribute, which is what the
    # OnIt UI keys off, and a component below sets that attribute on <html> so
    # the page's state is visible in the DOM. These blocks are not load-bearing
    # (the :root rule above is), but they are not dead either: the attribute is
    # always present once the component has run.
    '[data-theme="light"] { ' + _var_block(THEMES["light"]) + "}\n"
    '[data-theme="dark"] { ' + _var_block(THEMES["dark"]) + "}\n"
    # The font stack is the OnIt UI's, so the two read alike.
    ":root { --fw-font: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif; }\n"
    + _BASE_CSS
    + "</style>"
)
st.markdown(_CSS, unsafe_allow_html=True)

# Record the theme on <html>, the way the OnIt UI does (app.js:
# document.documentElement.setAttribute("data-theme", theme)).
#
# A same-origin iframe can reach window.parent.document, which is the only way to
# set an attribute on the page's <html> from Streamlit. st.html() is not an
# option: it sanitises <script> away -- measured, the attribute stayed null.
# st.iframe is used rather than st.components.v1.html because the latter is
# deprecated ("will be removed after 2026-06-01"); its docstring names
# st.iframe as the replacement and says same-origin HTML runs JavaScript. The
# frame is 0 px tall so it occupies no space.
_THEME_JS = (
    "<script>window.parent.document.documentElement"
    '.setAttribute("data-theme", "__THEME__");</script>'
)
# height must be a POSITIVE pixel count ("height must be either a positive
# integer, 'stretch', or 'content'" -- 0 is rejected), so the frame is hidden by
# its keyed container instead. A display:none iframe still loads and runs its
# script, which is all this one is for.
with st.container(key="themesync"):
    st.iframe(_THEME_JS.replace("__THEME__", _ACTIVE), height=1)


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------


def read_events(limit: int = 400, log: Path | None = None) -> list:
    """Return the last ``limit`` events, oldest first.

    ``log`` defaults to the LIVE session's log. It is a parameter because the
    page can be showing a different session than the one the agent is in -- the
    sidebar makes that an ordinary state, not an edge case -- and reading the
    live log while displaying another session would show the wrong conversation
    under the right heading.

    Tolerates a line being written at the moment we read it (JSONDecodeError ->
    skip). Consecutive ``user_partial`` events collapse to the newest one: each
    supersedes the last, so keeping the history would scroll the conversation
    away. A draft is dropped entirely once a later ``user_text`` arrives -- that
    final transcript is the same turn, spelled correctly.
    """
    path = Path(log) if log is not None else LOG_PATH
    if not path.exists():
        return []
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
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


def cleared_at(events: list) -> float:
    """The timestamp the visible conversation starts from.

    Two sources, because the two halves of "clear" happen at different moments:

      * the agent's ``cleared`` event, which is authoritative and survives a
        reload -- it is in the log, so any browser sees the same history;
      * a session marker, set the instant the button is pressed, so the view
        empties immediately instead of waiting up to one poll interval for the
        agent to notice the control file.

    The later of the two wins. Without the second, pressing Clear looks like it
    did nothing for a fifth of a second; without the first, reloading the page
    would bring the whole conversation back.
    """
    marker = 0.0
    for ev in events:
        if ev.get("kind") == "cleared":
            try:
                marker = max(marker, float(ev.get("ts") or 0))
            except (TypeError, ValueError):
                pass
    try:
        marker = max(marker, float(ss.get("cleared_ts") or 0))
    except (TypeError, ValueError):
        pass
    return marker


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


def send_upload(name: str, data: bytes) -> str:
    """Store an uploaded file in the session and hand the agent a notification.

    The file is written HERE, into the session's work/uploads/ -- the folder the
    agent's tools are jailed to -- so the agent reads it with read_file like any
    other file instead of receiving its contents through a control file. What
    travels through the control file is the name, which is all the spoken
    question needs.

    Returns "" on success, or a sentence to show the person on refusal. The
    refusals are the interesting part: an oversized file, a type WanDog does not
    read, or no live session to put it in.
    """
    if not config.SESSION_ID:
        return ("No session is running yet. Start the agent first, then "
                "attach the file.")
    try:
        stored = sess.store_upload(config.SESSION_ID, name, data)
    except ValueError as e:
        return str(e)
    except OSError as e:
        return f"Could not save that file: {e}"
    _drop(UPLOAD_FILE, {
        "id": uuid.uuid4().hex,
        "name": stored.name,
        "path": str(stored),
        "bytes": stored.stat().st_size,
        "ts": time.time(),
    })
    return ""


def request_stop() -> None:
    _drop(STOP_FILE, {"id": uuid.uuid4().hex, "ts": time.time()})


def request_clear() -> None:
    """Ask the agent to forget the conversation, and empty the view now.

    ``ts`` matters here: if no agent is running when this is pressed the file
    stays on disk, and the agent that starts later must not treat a clear from
    minutes ago as an instruction about the new session.
    """
    _drop(CLEAR_FILE, {"id": uuid.uuid4().hex, "ts": time.time()})
    ss["cleared_ts"] = time.time()


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
    """True when the agent is working on a turn the person could still stop.

    Two things end a turn that used to be missed, both reported as "the turn
    does not end sometimes":

    ``user_stopped`` with no transcript. A turn opens on ``user_started`` (the
    VAD fired). It is only answered if a transcript follows. When the mic is
    muted, or the person coughs and says nothing, the turn closes with
    ``user_stopped`` and NO ``user_text`` -- and since ``user_stopped`` did not
    clear ``busy``, the button stayed on Stop for a turn that had already
    finished and would never be answered. Measured on the real log: the final
    ``user_stopped`` (10:19:28) left this function returning True, so Stop
    stayed up until the 90 s staleness guard expired.

    So ``user_stopped`` ends the turn only when no transcript arrived during it.
    The distinction matters both ways: with no words there is nothing coming and
    nothing to stop, while with words the answer is still owed -- the real log
    shows 1.4-1.8 s between ``user_stopped`` and ``bot_started``, and Stop must
    stay available through that gap.

    ``error``. A failed inference pushes an ErrorFrame and no bot_stopped, so
    without this the button waited out the staleness guard with nothing to
    stop. An error ends the turn.

    ``interrupted`` deliberately stays an ender rather than a starter: pressing
    Stop must put Send back at once, not wait for the agent to acknowledge.
    """
    busy = False
    last_ts = None
    # Did a transcript arrive during the current spoken turn? Set False when the
    # person starts talking, True when words arrive. user_stopped reads it.
    turn_has_words = False
    for ev in events:
        kind = ev.get("kind")
        if kind in ("user_text", "bot_started", "tool_call"):
            busy = True
            if kind == "user_text":
                turn_has_words = True
        elif kind == "user_started":
            # Listening. Stoppable, but no words yet, so this turn may end with
            # nothing to answer.
            busy = True
            turn_has_words = False
        elif kind == "user_stopped":
            # The person stopped talking. If words arrived, the answer is still
            # owed and the button stays Stop; if not, the turn is over.
            busy = turn_has_words
        elif kind in ("bot_stopped", "session_end", "interrupted", "cleared",
                      "error"):
            busy = False
        if kind in ("user_text", "bot_started", "bot_stopped", "tool_call",
                    "interrupted", "user_stopped", "user_started", "error",
                    "user_partial"):
            # user_started and user_partial belong here. user_started now sets
            # busy, and partials are proof the person is still talking, so
            # neither may be measured against an older event: a sentence longer
            # than TURN_STALE_SECS would otherwise read as idle mid-sentence.
            last_ts = ev.get("ts")
    if busy and last_ts is not None:
        try:
            if time.time() - float(last_ts) > TURN_STALE_SECS:
                return False
        except (TypeError, ValueError):
            pass
    return busy


# ---------------------------------------------------------------------------
# Header: identity, theme toggle, clear history
# ---------------------------------------------------------------------------

# Both faces are inlined as data URIs rather than served as files: the page has
# no static route to assets/, and a data URI cannot 404 or be cached stale while
# the icon is being re-cut. Each is 192 px wide and drawn at 26-30 px, so it
# survives any device pixel ratio.
import base64


def _data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


husky_idle_uri = _data_uri(HUSKY_IDLE_PNG)
husky_think_uri = _data_uri(HUSKY_THINK_PNG)

# The bar's subtitle: which model and which ASR, from the VIEWED session's own
# start event. It reads the viewed log rather than the live one for the same
# reason the transcript does -- a heading that describes a different session
# than the text below it is worse than no heading.
events = read_events(log=VIEWED_LOG)
session = next((e for e in reversed(events) if e.get("kind") == "session_start"), None)
subtitle = "voice + text"
if session:
    bits = []
    if session.get("llm"):
        bits.append(str(session["llm"]))
    if session.get("asr"):
        bits.append(str(session["asr"]))
    if bits:
        subtitle = " \u00b7 ".join(bits)

# ---------------------------------------------------------------------------
# Sidebar: the sessions, newest first
# ---------------------------------------------------------------------------
# Two actions per row, and they mean different things:
#
#   the row label -- SHOW this session. Read-only. The agent stays where it is.
#   "switch"      -- MOVE the agent into this session. New turns land here.
#
# Splitting them is what makes browsing safe. One button doing both would mean
# that looking up something said last week silently ended today's conversation.
#
# The action is taken from the button's own return value, INSIDE the fragment.
# That is not a style choice: a click inside a fragment re-runs the fragment
# alone, not the page, so a handler in the main script would never see the
# press. (The opposite trap -- a main-script widget whose press is read in a
# fragment -- is why the composer's Send is handled where it is drawn.)
#
# The list refreshes on a 2 s timer so a session the agent just started appears
# without a click. Not 1.0 s like the transcript: nothing here changes per
# second, and re-reading every session's event log is the one real cost.


def _short_when(sid: str) -> str:
    """``2026-11-18T1620-a3f1`` -> ``18 Nov 16:20``.

    The id is already sortable, so the label is a slice and a strptime rather
    than anything stored separately. A malformed id falls back to the raw
    prefix instead of raising: a listing that cannot format one row must still
    list the others.
    """
    try:
        return time.strftime("%d %b %H:%M", time.strptime(sid[:15], "%Y-%m-%dT%H%M"))
    except (ValueError, TypeError):
        return sid[:15]


@st.fragment(run_every=2.0)
def session_list() -> None:
    rows = sess.list_sessions()
    live = _live_id()

    st.markdown('<div class="fw-side-head">Sessions</div>', unsafe_allow_html=True)
    if not rows:
        st.markdown(
            '<div class="fw-side-empty">No sessions yet. Start the agent and '
            "one is created.</div>",
            unsafe_allow_html=True,
        )
        return

    for row in rows:
        sid = row["id"]
        is_live = sid == live
        is_viewed = sid == VIEWED_ID
        label = row["title"] or "(no turns yet)"

        with st.container(key=f"sess_{sid}"):
            # The live session is marked in the label as well as by colour: a
            # dot survives a colour-blind reader and a greyscale screenshot,
            # and the two themes do not both have a "this is the one" hue.
            shown = st.button(
                ("\u25cf " if is_live else "") + f"{_short_when(sid)}  {label}",
                key=f"show_{sid}",
                help=("The live session -- where new turns land" if is_live
                      else f"Show this session, read-only \u00b7 {row['turns']} turns"),
                type="primary" if is_viewed else "secondary",
                width="stretch",
            )
            if shown and not is_viewed:
                ss["viewing"] = sid
                st.rerun()

            meta = f"{row['turns']} turns"
            if row["uploads"]:
                meta += f" \u00b7 {len(row['uploads'])} file(s)"
            st.caption(("live \u00b7 " if is_live else "") + meta)

            if not is_live:
                c_sw, c_ho = st.columns(2)
                with c_sw:
                    if st.button("switch", key=f"switch_{sid}",
                                 help="Make this the live session: the agent "
                                      "moves here and answers from it",
                                 width="stretch"):
                        try:
                            sess.request_switch(sid)
                            ss["viewing"] = sid
                            st.toast(f"Switching the agent to {_short_when(sid)}\u2026")
                        except ValueError as e:
                            st.error(str(e))
                        st.rerun()
                with c_ho:
                    if st.button("notes", key=f"hand_{sid}",
                                 help="Read this session's handover notes into "
                                      "the live one",
                                 width="stretch"):
                        try:
                            sess.request_handover(sid)
                            st.toast(f"Reading {_short_when(sid)}'s notes in\u2026")
                        except ValueError as e:
                            st.error(str(e))
                        st.rerun()


with st.sidebar:
    session_list()


# One container keyed "topbar", fixed to the top of the viewport by the CSS
# above. Identity on the left, the two controls on the right, so the bar keeps
# its shape whether or not a session is running.
with st.container(key="topbar"):
    h_name, h_theme, h_clear = st.columns([8.4, 0.8, 0.8], vertical_alignment="center")

    if BROWSING:
        # The page is showing a session the agent is not in. Said plainly and
        # once, in the bar, because everything below it -- the transcript, the
        # status line, the composer's refusal -- follows from this one fact.
        st.caption(
            f"Browsing {_short_when(VIEWED_ID)} \u00b7 read-only. "
            f"The agent is in {_short_when(LIVE_ID) if LIVE_ID else 'no session'}."
        )
        if st.button("Back to the live session", key="backlive_btn",
                     icon=":material/arrow_back:"):
            ss.pop("viewing", None)
            st.rerun()

    with h_name:

        @st.fragment(run_every=1.0)
        def identity() -> None:
            """The name plate, with a face that changes with the turn.

            This has to be a fragment: the icon is the one thing in the bar that
            has to move on its own, and the rest of the bar (theme switch, Clear,
            status line) must not re-run on a timer. run_every=1.0 is the same
            cadence the transcript uses, so the two agree.

            Both images are always in the DOM and CSS decides which is visible
            (.fw-face-idle / .fw-face-think under [data-busy]). Swapping the
            src instead would leave a frame with no image while the new data URI
            decodes, which at 30 px is a visible blink.
            """
            busy = turn_in_flight(read_events())
            cls = "fw-head fw-busy" if busy else "fw-head"
            st.markdown(
                f"""<div class="{cls}" data-busy="{'1' if busy else '0'}">
                  <span class="fw-face">
                    <img class="fw-face-idle" src="{husky_idle_uri}" alt="" />
                    <img class="fw-face-think" src="{husky_think_uri}" alt="" />
                  </span>
                  <div>
                    <div class="fw-name">WanDog</div>
                    <div class="fw-sub">{subtitle}</div>
                  </div>
                </div>""",
                unsafe_allow_html=True,
            )

        identity()

    # The answering scope, when it is not the default. A strict session is
    # answering from one file only, and that has to be visible: a limit the
    # person cannot see is indistinguishable from the agent not knowing
    # something. Shown only when narrowed, so the normal case stays quiet.
    if config.SESSION_ID:
        _mode = sess.get_mode(config.SESSION_ID)
        if _mode != sess.DEFAULT_MODE:
            _ups = sess.uploaded_files(config.SESSION_ID)
            _src = (", ".join(u["name"] for u in _ups)
                    if _ups else "no file yet")
            st.caption(
                "Answering from " + _src if _mode == "strict"
                else "Answering from this session's files and the web"
            )

    with h_theme:
        # The glyph is the theme you would switch TO, which is the OnIt
        # convention: its stylesheet hides the sun in light mode and the moon in
        # dark.
        with st.container(key="themebtn"):
            if st.button(
                "",
                key="themebtn_btn",
                icon=":material/dark_mode:" if theme == "light" else ":material/light_mode:",
                help="Switch to light theme" if theme == "dark" else "Switch to dark theme",
            ):
                ss["theme"] = "light" if theme == "dark" else "dark"
                st.query_params["theme"] = ss["theme"]
                st.rerun()

    with h_clear:
        with st.container(key="clearbtn"):
            if st.button(
                "",
                key="clearbtn_btn",
                icon=":material/delete_sweep:",
                help=("Clearing is disabled while browsing: it would clear the "
                      "live session, not the one on screen" if BROWSING else
                      "Clear chat history (also clears the model's memory of it)"),
                disabled=BROWSING,
            ):
                request_clear()
                st.rerun()

    # Status line: event count, the time of the newest event, and the log path.
    # Kept minimal on purpose -- this is a chat window, not a debug console -- but
    # the path is here because "the dashboard is up but shows nothing" is almost
    # always a log-path mismatch between the two processes.
    #
    # It lives in the fixed bar rather than under the transcript so it can never
    # push the transcript's bottom edge down (the transcript's height is a
    # viewport calculation) and can never scroll out of sight.
    tail = read_events(limit=5, log=VIEWED_LOG)
    if tail:
        st.caption(
            f"{len(read_events(log=VIEWED_LOG))} events \u00b7 "
            f"{fmt_time(tail[-1].get('ts'))} \u00b7 `{VIEWED_LOG}`"
        )


# ---------------------------------------------------------------------------
# Transcript
# ---------------------------------------------------------------------------


def render_event(ev: dict) -> None:
    kind = ev.get("kind", "")
    ts = fmt_time(ev.get("ts"))

    if kind == "user_text":
        source = ev.get("source")
        who = "You \u00b7 typed" if source == "typed" else "You"
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
                {_esc(ev.get("text", ""))}\u2026</div></div></div>""",
            unsafe_allow_html=True,
        )

    elif kind == "bot_text":
        st.markdown(
            f"""<div class="fw-row">
              <div class="fw-av fw-av-bot"><img src="{husky_idle_uri}" alt="" /></div>
              <div class="fw-body">
                <div class="fw-who">WanDog</div>
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
        body += f'<div class="fw-meta">called \u00b7 {ts}</div>'
        st.markdown(f'<div class="fw-tool">{body}</div>', unsafe_allow_html=True)

    elif kind == "tool_result":
        result = str(ev.get("result", ""))
        st.markdown(
            f"""<div class="fw-tool">
              <div class="fw-tool-name">{_esc(ev.get("tool", "tool"))} \u2014 result</div>
              <div class="fw-meta">{len(result)} chars \u00b7 {ts}</div>
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
            f'\u21b3 {_esc(ev.get("tool", "tool"))}: {" \u00b7 ".join(bits)}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "interrupted":
        who = ev.get("source")
        label = "stopped" if who == "dashboard" else "interrupted"
        st.markdown(
            f'<div class="fw-meta" style="margin-left:2.4rem">\u23f9 {label} \u00b7 {ts}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "error":
        # The agent failed this turn. Before this existed the failure was only
        # in the agent's console: the question sat there with no reply and no
        # explanation, which reads as "the agent ignored me".
        detail = str(ev.get("text", "") or "").strip()
        st.error(
            ("The agent stopped with an error" if ev.get("fatal")
             else "That turn failed")
            + (f": {detail[:300]}" if detail else ".")
            + f"  \n`{ts}`"
        )

    elif kind == "approval_requested":
        st.warning(
            f"Approval needed for {ev.get('tool', 'a tool')}: "
            f"{str(ev.get('command', ev.get('args', '')))[:200]}"
        )

    elif kind == "approval_granted":
        st.markdown(
            f'<div class="fw-meta">\u2705 approved {_esc(ev.get("tool", ""))} \u00b7 {ts}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "approval_denied":
        st.markdown(
            f'<div class="fw-meta">\u26d4 denied {_esc(ev.get("tool", ""))} \u00b7 {ts}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "user_started":
        st.markdown(
            f'<div class="fw-meta" style="margin-left:2.4rem">\U0001f3a4 listening\u2026 \u00b7 {ts}</div>',
            unsafe_allow_html=True,
        )

    elif kind == "bot_started":
        st.markdown(
            f'<div class="fw-meta" style="margin-left:2.4rem">\U0001f50a speaking\u2026 \u00b7 {ts}</div>',
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
    events = read_events(log=VIEWED_LOG)
    marker = cleared_at(events)

    if marker:
        visible = []
        for ev in events:
            try:
                if float(ev.get("ts") or 0) > marker:
                    visible.append(ev)
            except (TypeError, ValueError):
                continue
        events = visible

    if not events:
        if not VIEWED_LOG.exists():
            st.markdown(
                '<div class="fw-meta">Waiting for the agent. Start it with '
                "<code>.venv/bin/python agent.py</code>, then speak or type.</div>",
                unsafe_allow_html=True,
            )
        elif marker:
            st.markdown(
                '<div class="fw-meta">History cleared. Say something, or type below.</div>',
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
        if ev.get("kind") in ("session_start", "session_end", "cleared"):
            continue
        render_event(ev)


# The transcript scrolls itself (see the CSS note), so the fragment needs no
# anchor. It only has to be the thing that owns the scroll box, which is what
# gives us .st-key-transcript to size.
with st.container(key="transcript"):
    transcript()


# --- layout sync: transcript height, and open at the newest message ------
# Two jobs, one script, because they share the same trigger: a DOM change.
#
# 1. HEIGHT. The transcript's scroll box has to end just above the composer. A
#    hard-coded rem subtraction could not do that reliably -- the bar's height
#    depends on whether the status line is present, and Streamlit's own block
#    padding is in em -- so it is measured here instead: the bar's bottom edge and
#    the composer's top edge are real numbers, and the difference is published as
#    --fw-scroll-h. Measured before this: transcript bottom 607 px against a
#    composer at 575, i.e. 32 px of conversation permanently hidden behind the
#    composer's gradient.
#
# 2. SCROLL POSITION. Streamlit starts an overflow box at scrollTop 0, so a
#    returning reader landed at the OLDEST message. The rule is: follow the bottom
#    only while the reader is already at the bottom, and stop as soon as they
#    scroll up, so reading back is never yanked away every second the transcript
#    fragment re-runs.
#
# Both live in the PARENT document, installed once and guarded by a flag. A script
# inside the 1 px iframe would re-run on every fragment rerun and could not share
# state between runs -- and it could not observe the parent's DOM at all, which is
# what the height measurement needs.
_AUTOSCROLL_JS = """
<script>
(function () {
  var W = window.parent, D = W.document;

  // The Ctrl+Enter backstop is installed BEFORE the one-shot guard below, and
  // guarded separately, because the two have different lifetimes. The layout
  // work must happen once; this listener must exist in every tab. It used to
  // sit after `if (__fwLayout) return`, so a tab that had loaded the page before
  // that line was added kept __fwLayout = true and never installed the
  // listener -- Ctrl+Enter and Enter both did nothing there while the Send
  // button, which is a real Streamlit widget, kept working. That is the report
  // this fixes.
  //
  // It is a BACKSTOP, not the mechanism: st.text_area already commits on
  // Ctrl+Enter itself. This only matters if that ever changes, and it fires on
  // the capture phase so it wins before the widget's own handler.
  //
  // Ctrl+Enter means Send when idle and Interrupt while a turn is running --
  // the same two meanings the composer's one button carries.
  if (!W.__fwKeys) {
    W.__fwKeys = true;
    D.addEventListener('keydown', function (ev) {
      if (ev.key !== 'Enter' || !(ev.ctrlKey || ev.metaKey)) return;
      var t = ev.target;
      if (!t || t.tagName !== 'TEXTAREA') return;
      if (!t.closest('.st-key-composer')) return;
      // One key, two meanings, chosen by what the control is showing. While a
      // turn is in flight the composer's button IS Stop, so Ctrl+Enter must
      // interrupt -- that is the whole point of the shortcut, and returning
      // early here (as this did) made Ctrl+Enter dead exactly when it was most
      // wanted: mid-turn, with the mouse needed for the button.
      //
      // While busy the box is not sent, so the typed text stays where it is:
      // an interrupt should not swallow a half-written message.
      var stop = D.querySelector('.st-key-stopbtn button');
      if (stop) { setTimeout(function () { stop.click(); }, 0); return; }
      // Not busy: let the widget commit the value itself. Clicking Send here
      // would race the commit and send an empty string -- measured: the box
      // cleared and the message was lost.
      var b = D.querySelector('.st-key-sendbtn button');
      if (!b) return;
      setTimeout(function () { b.click(); }, 0);
    }, true);
  }

  if (W.__fwLayout) return;
  W.__fwLayout = true;

  function box() { return D.querySelector('.st-key-transcript'); }
  function bar() { return D.querySelector('.st-key-topbar'); }
  function composer() { return D.querySelector('.st-key-composer'); }
  function rail() { return D.querySelector('[data-testid="stSidebar"]'); }

  // How much of the window the session rail occupies, in px. Published as
  // --fw-rail because the bar and the composer are position:fixed against the
  // WINDOW, and without this they sit underneath the rail.
  //
  // Measured rather than assumed: the rail is resizable by dragging, and it is
  // 0 when collapsed, so any hard-coded width is wrong in at least one of the
  // states a person will actually use. getBoundingClientRect().right is the
  // rail's right edge, which is the offset the main area starts at.
  function railWidth() {
    var r = rail();
    if (!r) return 0;
    var cs = W.getComputedStyle(r);
    if (cs.display === 'none' || cs.visibility === 'hidden') return 0;
    var rect = r.getBoundingClientRect();
    if (rect.width < 1) return 0;
    return Math.max(0, Math.round(rect.right));
  }

  function fit() {
    var root = D.documentElement.style;
    // The rail offset is published FIRST, and before the guard below: the bar
    // and the composer are positioned by it, and they exist even when there is
    // no transcript to size (an empty session, the first paint). Publishing it
    // after the guard left the bar under the rail on exactly those screens.
    var railPx = railWidth();
    if (root.getPropertyValue('--fw-rail') !== railPx + 'px') {
      root.setProperty('--fw-rail', railPx + 'px');
    }

    var t = box(), b = bar(), c = composer();
    if (!t || !b) return;

    var barBottom = b.getBoundingClientRect().bottom;
    var lower = c ? c.getBoundingClientRect().top : W.innerHeight;

    // Where the transcript would sit with NO margin of ours. Streamlit places the
    // block a little way down on its own (48 px here), so the margin has to be
    // "the distance I want" MINUS that offset -- setting it to the distance itself
    // left 60 px of dead space under the bar. Reading the applied margin back and
    // subtracting it is self-correcting, and idempotent: once the margin is right,
    // the next run computes the same number.
    var applied = parseFloat(W.getComputedStyle(t).marginTop) || 0;
    var natural = t.getBoundingClientRect().top - applied;
    var want = barBottom + 12;
    root.setProperty('--fw-top-pad', Math.max(0, Math.round(want - natural)) + 'px');

    // Height is measured to the composer's upper edge, so the transcript's last
    // line is never behind the composer's gradient.
    root.setProperty('--fw-scroll-h',
                     Math.max(120, Math.round(lower - want - 12)) + 'px');
  }

  function atBottom(el) {
    return el.scrollHeight - el.scrollTop - el.clientHeight < 40;
  }

  function follow() {
    var el = box();
    if (!el) return;
    if (el.__fwFollow === undefined) el.__fwFollow = true;  // first sight: start at the end
    if (el.__fwFollow) el.scrollTop = el.scrollHeight;
  }

  function sync() { fit(); follow(); }

  // A reader who scrolls up stops the follow; scrolling back down resumes it.
  // The flag lives on the element, so a re-render that replaces the box starts at
  // the bottom again.
  D.addEventListener('scroll', function (ev) {
    var el = box();
    if (el && ev.target === el) el.__fwFollow = atBottom(el);
  }, true);

  // MutationObserver rather than an interval: it also fires when a tool result or
  // a reply grows the box, which is exactly when the height needs recomputing.
  new W.MutationObserver(sync).observe(D.body, {
    childList: true, subtree: true, characterData: true
  });
  W.addEventListener('resize', sync);
  W.addEventListener('load', sync);

  // Collapsing or dragging the rail changes no child list of D.body, so the
  // observer above does not fire for it -- but the bar's left offset and the
  // transcript's width both depend on it. A ResizeObserver on the rail is the
  // event that actually corresponds to the change. Attached lazily because the
  // rail can appear after this script runs (Streamlit renders it on first
  // paint, and a page that loads before any session exists has no rows to make
  // it visible).
  if (W.ResizeObserver) {
    var ro = new W.ResizeObserver(function () { sync(); });
    var attach = function () {
      var r = rail();
      if (r && !r.__fwObserved) { r.__fwObserved = true; ro.observe(r); }
    };
    attach();
    new W.MutationObserver(attach).observe(D.body, { childList: true, subtree: true });
  }
  sync();

})();
</script>
"""
with st.container(key="autoscrollsync"):
    st.iframe(_AUTOSCROLL_JS, height=1)


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

# Clearing a keyed text_input cannot be done in the same run that rendered it
# (Streamlit refuses to modify a widget's state after instantiation), so the
# Send handler raises a flag and the next run clears the box before it renders.
if ss.pop("clear_box", False):
    ss["txt"] = ""

if "mic_on" not in ss:
    ss["mic_on"] = mic_state()

# NOTE: there is deliberately no module-level `busy` here. The Send/Stop swap
# lives in its own timed fragment below, because a value read once in the main
# script cannot notice the agent finishing: the main script only re-runs when a
# widget is touched. That is exactly the "stuck on Stop" bug -- measured with the
# turn ending in the log and data-busy already 0 in the bar, while the composer
# still showed Stop for 16 s and only reverted on the next click.


def mic_button() -> None:
    live = ss["mic_on"]
    # A plain button rather than st.toggle: the switch widget cannot be made to
    # look like Ollama's inline icon button, and this state has to be obvious at
    # a glance -- blue means the agent is accepting audio.
    #
    # The glyph is Material's "mic" (a capsule head, a stem and a base -- the
    # same shape as the microphone the request came with) rather than the 🎤
    # emoji, which renders in the font's own colours and so cannot be tinted.
    # The icon span inherits the button's colour, which is what lets one CSS
    # rule turn it blue while listening and leave it the text colour when not.
    with st.container(key="micbtn"):
        clicked = st.button(
            "",
            key="micbtn_btn",
            icon=":material/mic:",
            help=(
                "Listening \u2014 the agent is accepting audio. Click to mute."
                if live
                else "Muted \u2014 the agent will not transcribe. Click to listen."
            ),
        )
    if clicked:
        ss["mic_on"] = not live
        set_mic_state(ss["mic_on"])


# The Send/Stop swap, in a fragment that re-runs on a timer.
#
# Why a fragment and not an `if busy:` in the main script: the main script only
# re-runs when a widget is touched, so a value read there is frozen at the moment
# of the last click. The turn ends in the LOG, not in the page, so nothing told
# the composer to swap back and the button sat on Stop until the next click.
# Measured before this fix: with bot_stopped appended to the log, the bar's
# data-busy was already "0" one second later while .st-key-stopbtn was still the
# only control present -- 16 s, unchanged, until the probe clicked something.
#
# The bar's face already worked this way (its own run_every=1.0 fragment); this
# is the same pattern applied to the control that has to agree with it. The
# transcript and the face run at 1.0 s too, so the three cannot disagree by more
# than one tick.
#
# Nothing else moved into the fragment. The text box stays in the main script on
# purpose: a fragment that re-runs every second around a widget the person is
# typing into is how you lose half-typed text.
@st.fragment(run_every=1.0)
def send_control() -> None:
    """Render Stop while a turn is in flight, Send otherwise."""
    busy = turn_in_flight(read_events())

    if busy:
        # Wrapped in a keyed container so the CSS can reach it: a container
        # keyed "stopbtn" becomes .st-key-stopbtn, which is how the red
        # styling is applied without depending on widget order.
        with st.container(key="stopbtn"):
            stop_clicked = st.button(
                "", key="stopbtn_btn", icon=":material/stop:",
                help="Stop this turn (or press Ctrl+Enter)",
            )
        if stop_clicked:
            request_stop()
            # Reflect the intent immediately: waiting for the agent's
            # interrupted event would leave the button looking unresponsive
            # for up to one poll interval.
            st.rerun()
        return

    with st.container(key="sendbtn"):
        send_clicked = st.button(
            "", key="sendbtn_btn", icon=":material/arrow_upward:",
            help="Send (or press Ctrl+Enter) \u00b7 Ctrl+Enter stops a turn in progress",
        )
    if send_clicked:
        # Read from session_state, not from the `typed` local: a fragment
        # re-executes on its own, so a name bound in the main script's scope is
        # not part of its context. The keyed text box keeps its value here.
        text = (ss.get("txt") or "").strip()
        if text and BROWSING:
            # A typed message is delivered to the LIVE session. Sending it from
            # a browsed view would put the words in a conversation the person is
            # not looking at -- the worst kind of wrong: it looks like it worked.
            st.toast("This is a read-only view. Switch to the session to type into it.")
            return
        if text:
            send_typed(text)
            ss["clear_box"] = True
            # Full-app rerun (the default scope), so the box actually clears:
            # that happens at the top of the main script, which a
            # fragment-scoped rerun would not reach.
            st.rerun()

with st.container(key="composer"):
    c_attach, c_mic, c_text, c_send, c_model = st.columns(
        [0.7, 0.7, 8.2, 0.9, 0.7], vertical_alignment="center"
    )

    with c_attach:
        # The file upload. A popover rather than a bare uploader: the composer
        # is one row of controls, and st.file_uploader is a dropzone that would
        # push the row open. The popover keeps the row's height fixed, which is
        # what test_dashboard_layout.py's geometry assertions rely on.
        #
        # st.popover is a button plus a panel, so the key stays attachph_btn --
        # the same key the placeholder used, which is also the key the UI test
        # looks for.
        with st.popover("", icon=":material/attach_file:",
                        help="Attach a file for this session"):
            st.caption(
                f"Files go in this session's folder. Up to "
                f"{sess.UPLOAD_MAX_BYTES // (1024 * 1024)} MB: "
                + ", ".join(sorted(e.lstrip('.') for e in sess.UPLOAD_EXTENSIONS))
            )
            picked = st.file_uploader(
                "Attach a file",
                key="attach_file",
                label_visibility="collapsed",
                accept_multiple_files=False,
                type=[e.lstrip(".") for e in sorted(sess.UPLOAD_EXTENSIONS)],
            )
            if picked is not None and BROWSING:
                # An upload goes into the LIVE session's work/uploads/, so from
                # a browsed view it would land in a conversation that is not the
                # one on screen -- and the spoken question would be asked in it.
                st.warning("Switch to this session before attaching a file to it.")
            elif picked is not None:
                # One write per file, guarded on the id Streamlit gives the
                # upload: the popover reruns on every keystroke in the
                # composer, and without this the same file would be stored
                # (and spoken about) on every one of them. A module-level
                # set, not session_state: Send does a full-app rerun, which
                # rebuilds this script from the top.
                if picked.file_id not in _SEEN_UPLOADS:
                    err = send_upload(picked.name, picked.getvalue())
                    if err:
                        st.error(err)
                    else:
                        _SEEN_UPLOADS.add(picked.file_id)
                        st.success(f"Attached {picked.name}")

    with c_mic:
        mic_button()

    with c_text:
        # st.text_area, not st.text_input, and not because of looks. The input
        # was a single-line <input>, which CANNOT hold a newline: there is no
        # character in it that Enter could insert, so "Enter starts a new line"
        # was unachievable with that widget. A textarea holds one.
        #
        # The key handling then comes free. Streamlit's textarea has a built-in
        # keydown handler -- TextArea.js, `s = !a || metaKey || ctrlKey` with a
        # = true -- which commits the value on Ctrl+Enter and leaves plain Enter
        # to the browser, i.e. a newline. That is exactly the requested
        # behaviour, so no listener of ours is needed for it. Ours also could
        # not be relied on: it was installed after `if (__fwLayout) return`, so
        # a tab opened before that script changed never got it at all -- which
        # is why Ctrl+Enter did nothing while the button still worked.
        #
        # `enter_to_submit` does not exist on this widget, and `height` keeps the
        # pill one line tall until the text wraps.
        #
        # The return value is not used: the Send handler reads ss["txt"] instead,
        # because it lives in a fragment and so cannot see a name bound here.
        st.text_area(
            "Message",
            key="txt",
            height=44,
            label_visibility="collapsed",
            placeholder="Message WanDog, or just start talking\u2026",
        )

    with c_send:
        send_control()


    with c_model:
        # Placeholder: model selection comes later.
        with st.container(key="modelph"):
            st.button(
                "",
                key="modelph_btn",
                icon=":material/expand_more:",
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
    live_blue = MIC_LIVE[_ACTIVE]
    hover_blue = MIC_LIVE_HOVER[_ACTIVE]
    st.markdown(
        "<style>.st-key-micbtn button,"
        '.st-key-micbtn [data-testid="stBaseButton-secondary"]{'
        f"color:{live_blue} !important;border-color:{live_blue} !important;"
        f"background:color-mix(in srgb, {live_blue} 14%, transparent) !important;}}"
        ".st-key-micbtn button:hover{"
        f"color:{hover_blue} !important;border-color:{hover_blue} !important;}}"
        "</style>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------
# (The status line itself is rendered inside the top bar, above.)
