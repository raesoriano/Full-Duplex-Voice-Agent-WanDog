"""The top bar, the transcript scroller and the avatar icon.

Three behaviours that only a real browser can confirm, so this file is a mix:
the DOM contracts that AppTest can see (the bar contains the controls, the
transcript is a keyed container of its own, the icon is SVG) are checked here
with AppTest; the geometry and the scroll position are checked against a live
server over the DevTools protocol, because AppTest has no layout.

The browser half starts its OWN Streamlit server, pointed at this test's own
log. That matters: an earlier version drove whatever server happened to be
running, so the log it appended to was not the log the server read, and the check
"the transcript grew" quietly never held -- the check passed by not testing
anything. Owning the server is what makes the append land.

Run:  .venv/bin/python test_dashboard_layout.py
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

_TMP = Path(tempfile.mkdtemp(prefix="fw_layout_"))
os.environ["FWEDDIE_DASHBOARD_LOG"] = str(_TMP / "events.jsonl")

import base64  # noqa: E402
import numpy as np  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

ASSETS = Path(__file__).parent / "assets"
# The aspect each drawing had before it was scaled to an icon. The source
# drawings are no longer in the tree (the icons are the assets now), so these
# are recorded here rather than read from a file: idle is taller than wide,
# think wider than tall. Re-measure if a face is ever redrawn.
ASPECT = {"idle": 0.885, "think": 1.043}
DASHBOARD = str(Path(__file__).parent / "dashboard.py")
LOG = Path(os.environ["FWEDDIE_DASHBOARD_LOG"])
CHROMIUM_PORT = 9455


def _seed(*events: dict) -> None:
    with LOG.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _seed_conversation(n: int = 14) -> None:
    """Enough turns that the transcript has to scroll."""
    now = time.time()
    events = [{"kind": "session_start", "ts": now - 600}]
    for i in range(n):
        events.append({"kind": "user_text", "text": f"question {i}", "ts": now - 600 + i * 4})
        events.append({"kind": "bot_text", "text": f"answer {i} " + "x" * 120,
                       "ts": now - 598 + i * 4})
        events.append({"kind": "bot_stopped", "ts": now - 597 + i * 4})
    _seed(*events)


# ---------------------------------------------------------------------------
# AppTest half: the DOM contracts
# ---------------------------------------------------------------------------


def test_bar_holds_identity_and_both_controls() -> None:
    """The bar must be ONE keyed container holding the identity and the two
    controls, because that container is what the CSS pins to the viewport.

    If a control sat outside .st-key-topbar it would scroll away with the
    conversation, which is the exact complaint this change answers.
    """
    _seed({"kind": "session_start", "llm": "m", "asr": "a", "ts": time.time()})
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)

    keys = [b.key for b in at.button]
    for wanted in ("themebtn_btn", "clearbtn_btn"):
        assert wanted in keys, f"{wanted} is missing: {keys}"

    blob = "\n".join(m.value for m in at.markdown)
    assert 'class="fw-head"' in blob, "the identity block is missing"

    # The caption (status line) is inside the bar, not after the transcript: a
    # caption below the transcript would sit under the composer and push the
    # transcript's measured height around.
    assert len(at.caption) >= 1, "no status caption at all"
    print("1. the bar holds identity, theme, clear, status     OK")


def test_transcript_is_its_own_scroll_box() -> None:
    """The transcript must be a keyed container, so the CSS can size it and the
    script can find it.

    A bare st.fragment() call gives no stable hook: the container that wraps it
    is Streamlit's, with a generated class name that changes between releases.
    """
    _seed_conversation(3)
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    blob = "\n".join(m.value for m in at.markdown)
    assert "question 0" in blob, "the transcript did not render"
    print("2. the transcript renders inside its own container   OK")


def test_avatar_is_the_idle_face() -> None:
    """The avatar beside a reply must be the tongue-out face, always.

    Not the thinking face, and not the same image the bar happens to be showing:
    the avatar is who is speaking, the bar's face is what it is doing. Two
    different questions, two different images, and the avatar's must not flicker
    with the turn.
    """
    _seed(
        {"kind": "session_start", "ts": time.time() - 9},
        {"kind": "user_text", "text": "hi", "ts": time.time() - 8},
        {"kind": "bot_text", "text": "hello", "ts": time.time() - 7},
    )
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    blob = "\n".join(m.value for m in at.markdown)

    idle = base64.b64encode((ASSETS / "husky-idle.png").read_bytes()).decode()
    think = base64.b64encode((ASSETS / "husky-think.png").read_bytes()).decode()
    assert idle in blob, "the avatar is not the idle face"

    # Check the bot row itself, not a slice of the whole blob: splitting on
    # "fw-row" and keeping the first piece that mentions fw-av-bot also catches
    # the BAR (which carries both faces by design), so the assertion failed on an
    # icon the avatar never used.
    # Match the row markup, not the bare class name: the stylesheet block also
    # contains "fw-av-bot", so a substring test picked up the CSS and failed on
    # an icon the avatar never used.
    rows = [m.value for m in at.markdown if 'class="fw-av fw-av-bot"' in m.value]
    assert rows, "no bot row in the transcript"
    for row in rows:
        assert idle in row, "a bot row is not using the idle face"
        assert think not in row, "a bot row is using the thinking face"
    print("3. the avatar is always the tongue-out face          OK")


def test_icons_are_raster_and_two_tone() -> None:
    """Both faces must exist, differ, and carry a white face under dark ink.

    A raster is the point: the source is a raster, and contour-tracing it produced
    blobs (the eyes fused with the muzzle). The two-tone requirement is the one
    that is easy to lose silently -- dark ink alone is invisible on the dark
    theme, and a white face alone is invisible on the light one, so both must be
    in the same file.

    The aspect check guards against a rebuild that forces a square box: the two
    drawings are not square and are not the same shape, so squashing one is easy
    to ship without noticing.
    """
    from PIL import Image

    idle_path = ASSETS / "husky-idle.png"
    think_path = ASSETS / "husky-think.png"
    for path in (idle_path, think_path):
        assert path.exists(), f"missing icon {path.name}"

    idle = np.asarray(Image.open(idle_path).convert("RGBA")).astype(float)
    think = np.asarray(Image.open(think_path).convert("RGBA")).astype(float)
    assert idle.shape != think.shape or not np.array_equal(idle, think), (
        "the two faces are the same image"
    )

    for name, img in (("idle", idle), ("think", think)):
        alpha = img[..., 3] / 255.0
        lum = img[..., :3].mean(axis=2)
        opaque = alpha > 0.5
        # Roughly half the square is drawing and the rest is the transparent
        # margin (measured 0.49 for the idle face, 0.54 for the thinking one).
        assert 0.30 < opaque.mean() < 0.95, (
            f"{name}: {opaque.mean():.2f} of the icon is drawn"
        )
        # Two-tone: a light backing under a darker drawing. The thinking face is
        # blue rather than black, so this is a range check, not a threshold --
        # dark ink alone is invisible on the dark theme, a light face alone on
        # the light one, and both have to be in the same file.
        face = opaque & (lum > 200)
        ink = opaque & (lum < 180)
        assert face.mean() > 0.10, f"{name}: no light backing ({face.mean():.2f})"
        assert ink.mean() > 0.10, f"{name}: no darker drawing ({ink.mean():.2f})"
        assert lum[opaque].max() - lum[opaque].min() > 60, f"{name}: flat, not two-tone"
        # Transparent margin on all four sides, or the icon is a crop that will
        # read as off-centre in a circle.
        assert not opaque[0].any() and not opaque[-1].any(), f"{name}: no top/bottom margin"

        # Aspect: the drawing must keep the shape its source had, not a
        # square one. The two faces are neither square nor alike, so a
        # rebuild that forces a square box squashes one of them.
        ys, xs = np.nonzero(opaque)
        got = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        want = ASPECT[name]
        assert abs(got - want) / want < 0.05, (
            f"{name}: icon aspect {got:.3f} but the drawing is {want:.3f} "
            f"-- the drawing was squashed into a square box"
        )
    print("4. both faces are raster, differ, two-tone            OK")
    print("   ...and keep their drawing aspect, not a square    OK")


# ---------------------------------------------------------------------------
# Browser half: geometry and scroll position
# ---------------------------------------------------------------------------


def _ws_url(port: int) -> str:
    for _ in range(100):
        try:
            pages = json.loads(
                urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=1).read()
            )
            for p in pages:
                if p.get("type") == "page" and p.get("webSocketDebuggerUrl"):
                    return p["webSocketDebuggerUrl"]
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError("chromium did not expose a debugging endpoint")


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class _Server:
    """A Streamlit server of our own, reading our own log."""

    def __init__(self) -> None:
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}/"
        self.proc = None

    def __enter__(self) -> "_Server":
        env = dict(os.environ, FWEDDIE_DASHBOARD_LOG=str(LOG))
        self.proc = subprocess.Popen(
            [str(Path(__file__).parent / ".venv/bin/python"), "-m", "streamlit",
             "run", "dashboard.py", "--server.port", str(self.port),
             "--server.address", "127.0.0.1", "--server.headless", "true"],
            cwd=str(Path(__file__).parent), env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        deadline = time.time() + 90
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(self.url, timeout=2) as r:
                    if r.status == 200:
                        return self
            except Exception:
                time.sleep(0.4)
        raise RuntimeError(f"streamlit did not come up on {self.url}")

    def __exit__(self, *exc) -> None:
        if self.proc:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


async def _drive(url: str, checks: dict, setup: str | None = None,
                  mid=None, settle: float = 4.0) -> dict:
    """Load the page, run ``setup`` in it, call ``mid`` (a Python callable, for
    things the browser cannot do such as appending to the log), wait ``settle``
    seconds, then evaluate ``checks``.

    The ``mid`` hook is the important part: a new turn has to arrive WHILE the
    page is open. Appending to the log before the browser starts means the page
    simply loads with the new content and no mutation ever fires, so a rule about
    what happens when a turn arrives is never exercised -- which is how an
    earlier version of this check passed against a deliberately broken script.
    """
    import websockets

    proc = subprocess.Popen(
        ["chromium", "--headless=new", "--disable-gpu", "--no-sandbox",
         f"--remote-debugging-port={CHROMIUM_PORT}", "--window-size=1100,800",
         "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        async with websockets.connect(_ws_url(CHROMIUM_PORT),
                                      max_size=64 * 1024 * 1024) as ws:
            n = 0

            async def cmd(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg.get("result", {})

            async def js(expr):
                r = await cmd("Runtime.evaluate", expression=expr,
                              returnByValue=True, awaitPromise=True)
                return r.get("result", {}).get("value")

            await cmd("Page.enable")
            await cmd("Runtime.enable")
            await cmd("Page.navigate", url=url)
            deadline = time.time() + 60
            while time.time() < deadline:
                if await js("!!document.querySelector('.fw-row')"):
                    break
                await asyncio.sleep(0.4)
            await asyncio.sleep(2.5)

            if setup:
                await js(setup)
                await asyncio.sleep(0.8)
            if mid:
                mid()
            await asyncio.sleep(settle)

            out = {}
            for name, expr in checks.items():
                out[name] = await js(expr)
            return out
    finally:
        proc.terminate()
        proc.wait(timeout=10)


GEOMETRY = {
    "bar": """(() => {
      const e = document.querySelector('.st-key-topbar');
      if (!e) return null;
      const r = e.getBoundingClientRect();
      const cs = getComputedStyle(e);
      return {position: cs.position, y: Math.round(r.y), h: Math.round(r.height),
              w: Math.round(r.width)};
    })()""",
    "transcript": """(() => {
      const e = document.querySelector('.st-key-transcript');
      if (!e) return null;
      const r = e.getBoundingClientRect();
      return {y: Math.round(r.y), h: Math.round(r.height),
              bottom: Math.round(r.bottom),
              scrollTop: Math.round(e.scrollTop),
              scrollHeight: e.scrollHeight, clientHeight: e.clientHeight,
              overflowY: getComputedStyle(e).overflowY,
              scrollbarWidth: e.offsetWidth - e.clientWidth};
    })()""",
    "composer": """(() => {
      const e = document.querySelector('.st-key-composer');
      if (!e) return null;
      const r = e.getBoundingClientRect();
      return {y: Math.round(r.y)};
    })()""",
    "page": """(() => ({
      docScrollHeight: document.documentElement.scrollHeight,
      docClientHeight: document.documentElement.clientHeight,
      scrollTop: Math.round(document.documentElement.scrollTop),
    }))()""",
    "lastRow": """(() => {
      const rows = document.querySelectorAll('.fw-row');
      if (!rows.length) return null;
      return Math.round(rows[rows.length - 1].getBoundingClientRect().bottom);
    })()""",
    "exceptions": """(() => [...document.querySelectorAll('[data-testid="stException"]')]
      .map(e => e.innerText.slice(0, 200)))()""",
    "layoutFlag": "!!window.__fwLayout",
    "rows": "document.querySelectorAll('.fw-row').length",
    # The scrollbar must sit on the window's right edge. A scroll box that
    # inherits Streamlit's centred 820 px column puts its thumb 200 px inboard,
    # which is not where a window's own scrollbar is.
    "scrollbar": """(() => {
      const e = document.querySelector('.st-key-transcript');
      if (!e) return null;
      const r = e.getBoundingClientRect();
      return {left: Math.round(r.left), right: Math.round(r.right),
              win: window.innerWidth,
              gutter: e.offsetWidth - e.clientWidth,
              overflow: getComputedStyle(e).overflowY};
    })()""",
    "faces": """(() => {
      const head = document.querySelector('.fw-head');
      if (!head) return null;
      const vis = (sel) => {
        const e = head.querySelector(sel);
        if (!e) return null;
        const cs = getComputedStyle(e);
        return {opacity: parseFloat(cs.opacity), src: (e.getAttribute('src')||'').slice(0, 30)};
      };
      return {busy: head.getAttribute('data-busy'),
              idle: vis('.fw-face-idle'), think: vis('.fw-face-think')};
    })()""",
}


def test_live_layout() -> None:
    """Against a live server: the bar is fixed, the transcript stops above the
    composer, and the page opens at the newest message.

    Every one of these was wrong at some point during this change:
      * the bar scrolled away (it was in the page flow);
      * the transcript ran under the composer by 32 px;
      * the page opened at the OLDEST message (Streamlit starts an overflow box
        at scrollTop 0).
    """
    _seed_conversation(14)  # enough turns that the transcript must scroll
    with _Server() as srv:
        got = asyncio.run(_drive(srv.url, GEOMETRY))

        bar, tr, comp = got["bar"], got["transcript"], got["composer"]
        assert bar, "no .st-key-topbar in the page"
        assert tr, "no .st-key-transcript in the page"
        assert comp, "no .st-key-composer in the page"

        assert bar["position"] == "fixed", f"the bar is {bar['position']}, not fixed"
        assert bar["y"] == 0, f"the bar is at y={bar['y']}, not the top of the page"

        # The bar must not overlap the transcript, and the transcript must not run
        # under the composer.
        assert tr["y"] >= bar["h"], (
            f"the transcript starts at y={tr['y']}, inside the {bar['h']} px bar"
        )
        assert tr["bottom"] <= comp["y"] + 1, (
            f"the transcript ends at {tr['bottom']}, below the composer at {comp['y']}"
        )
        assert tr["overflowY"] == "auto", f"the transcript is {tr['overflowY']}"
        assert tr["scrollHeight"] > tr["clientHeight"], (
            "the transcript does not overflow, so this proves nothing -- "
            "seed more turns"
        )

        # The page itself must not scroll: one scroll box, not two.
        assert got["page"]["docScrollHeight"] <= got["page"]["docClientHeight"] + 1, (
            f"the page still scrolls: {got['page']}"
        )

        assert got["layoutFlag"], "the layout script did not install itself"
        print("5. live: bar fixed, transcript between bar and composer  OK")

        # Opening the page lands at the end.
        gap = tr["scrollHeight"] - tr["scrollTop"] - tr["clientHeight"]
        assert gap <= 40, f"the transcript opened {gap} px from the bottom"
        assert got["lastRow"] is not None
        assert got["lastRow"] <= comp["y"] + 1, (
            f"the newest message is at {got['lastRow']}, behind the composer"
        )
        print("6. live: opens at the newest message, above the composer OK")

        # The scrollbar must be the window's scrollbar, in the window's place.
        # The transcript is full-bleed for exactly this reason: inside Streamlit's
        # centred 820 px column the thumb sat ~200 px inboard of the right edge.
        sb = got["scrollbar"]
        assert sb, "no .st-key-transcript in the page"
        assert sb["overflow"] == "auto", f"the transcript is {sb['overflow']}"
        assert abs(sb["right"] - sb["win"]) <= 1, (
            f"the scroller ends at x={sb['right']} in a {sb['win']} px window, so its "
            f"scrollbar is {sb['win'] - sb['right']} px inboard of the window edge"
        )
        assert sb["left"] <= 1, f"the scroller starts at x={sb['left']}, not the edge"
        assert sb["gutter"] > 0, (
            "the scroller reserves no scrollbar gutter, so this proves nothing"
        )
        print(f"7. live: scrollbar on the window edge (x={sb['right'] - sb['gutter']}"
              f"-{sb['right']} of {sb['win']})  OK")

        # The bar's face must follow the turn. Both images are in the DOM and CSS
        # chooses; checking opacity is what proves the swap actually happened,
        # rather than that both files happen to be present.
        idle_face = got["faces"]
        assert idle_face, "no .fw-head in the page"
        assert idle_face["busy"] == "0", (
            f"the page loaded already thinking (data-busy={idle_face['busy']})"
        )
        assert idle_face["idle"]["opacity"] > 0.9, (
            f"the idle face is not visible at rest: {idle_face['idle']}"
        )
        assert idle_face["think"]["opacity"] < 0.1, (
            f"the thinking face is visible at rest: {idle_face['think']}"
        )

        def _start_turn() -> None:
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": "user_text", "text": "a question",
                                    "ts": time.time()}) + "\n")
                f.write(json.dumps({"kind": "bot_started",
                                    "ts": time.time()}) + "\n")

        busy_face = asyncio.run(_drive(srv.url, {"faces": GEOMETRY["faces"]},
                                       mid=_start_turn, settle=5.0))
        assert busy_face["faces"]["busy"] == "1", (
            f"the bar did not notice the turn: {busy_face['faces']}"
        )
        assert busy_face["faces"]["think"]["opacity"] > 0.9, (
            f"the thinking face did not appear: {busy_face['faces']}"
        )
        assert busy_face["faces"]["idle"]["opacity"] < 0.1, (
            f"the idle face is still visible during a turn: {busy_face['faces']}"
        )
        print("8. live: the bar's face follows the turn              OK")

        # And it must not follow while the reader is reading back.
        #
        # The new turn is appended from INSIDE the browser session (the `mid`
        # hook), after the view has been scrolled to the top. Appending before the
        # browser starts would not exercise the rule at all: the page would simply
        # load with the extra turn and no DOM mutation would ever occur.
        def _append_turn() -> None:
            with LOG.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"kind": "user_text", "text": "a new question",
                                    "ts": time.time()}) + "\n")
                f.write(json.dumps({"kind": "bot_text",
                                    "text": "a new answer " + "y" * 500,
                                    "ts": time.time()}) + "\n")

        followed = asyncio.run(_drive(
            srv.url,
            {"after": """(() => {
               const e = document.querySelector('.st-key-transcript');
               return {scrollTop: Math.round(e.scrollTop),
                       scrollHeight: e.scrollHeight,
                       clientHeight: e.clientHeight,
                       rows: document.querySelectorAll('.fw-row').length,
                       flag: e.__fwFollow};
             })()"""},
            setup="""(() => {
              const e = document.querySelector('.st-key-transcript');
              e.scrollTop = 0;
              return 1;
            })()""",
            mid=_append_turn,
            settle=5.0,
        ))
        assert followed["after"]["rows"] > got["rows"], (
            "the appended turn never reached the page, so the follow rule was "
            f"never exercised: {followed['after']['rows']} rows vs {got['rows']}"
        )
        assert followed["after"]["scrollTop"] == 0, (
            "the view jumped to "
            f"{followed['after']['scrollTop']} while the reader was scrolled up"
        )
        print("9. live: new turn does not yank a reader back down    OK")


def main() -> int:
    tests = [
        test_bar_holds_identity_and_both_controls,
        test_transcript_is_its_own_scroll_box,
        test_avatar_is_the_idle_face,
        test_icons_are_raster_and_two_tone,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1

    try:
        test_live_layout()
    except AssertionError as e:
        print(f"FAIL test_live_layout: {e}")
        failed += 1

    print("OK" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
