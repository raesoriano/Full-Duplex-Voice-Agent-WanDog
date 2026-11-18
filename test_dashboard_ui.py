"""Drive the dashboard the way a person does: type, Send, Stop, mute.

Streamlit has no server to call in a test, so this uses AppTest, which runs
dashboard.py in-process and lets a test click widgets and read what was
rendered. That makes the file-level contract checkable end to end:

    Send  -> <log>.typed.json exists with the typed text, and the box is empty
    Stop  -> <log>.stop.json exists
    mic   -> <log>.mic_control.json exists with the new state
    Clear -> <log>.clear.json exists, and the transcript is empty immediately

It also pins the Send/Stop swap: the button must be Send when no turn is in
flight and Stop when one is, because that is what makes interruption possible
without a keyboard.

Run:  .venv/bin/python test_dashboard_ui.py
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

# Point the dashboard at a private log BEFORE it (or config) is imported.
_TMP = Path(tempfile.mkdtemp(prefix="fw_dash_"))
os.environ["WANDOG_DASHBOARD_LOG"] = str(_TMP / "events.jsonl")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).parent / "dashboard.py")
LOG = Path(os.environ["WANDOG_DASHBOARD_LOG"])
TYPED = Path(str(LOG) + ".typed.json")
STOP = Path(str(LOG) + ".stop.json")
# From config, never guessed: the mic file is named differently from the
# typed/stop files (see the note in config.py), and guessing cost a debug cycle.
import config  # noqa: E402
MIC = Path(config.DASHBOARD_MIC_FILE)
CLEAR = Path(config.DASHBOARD_CLEAR_FILE)


def _seed(*events: dict) -> None:
    with LOG.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _append(*events: dict) -> None:
    with LOG.open("a", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def _keys(at) -> list:
    return [b.key for b in at.button]


def test_send_writes_typed_and_clears_box() -> None:
    _seed({"kind": "session_start", "llm": "m", "asr": "a", "ts": time.time() - 60})
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    assert "sendbtn_btn" in _keys(at), _keys(at)

    at.text_area(key="txt").set_value("  what is the weather  ").run()
    at.button(key="sendbtn_btn").click().run()
    assert not list(at.exception), list(at.exception)

    assert TYPED.exists(), "Send did not write the typed-text file"
    payload = json.loads(TYPED.read_text(encoding="utf-8"))
    # Stripped: a stray space from a paste must not become the prompt.
    assert payload["text"] == "what is the weather", payload
    assert payload.get("id"), payload
    assert at.text_area(key="txt").value == "", "the box was not cleared"
    print("1. Send writes the typed file and clears the box  OK")


def test_empty_send_does_nothing() -> None:
    _seed({"kind": "session_start", "ts": time.time()})
    TYPED.unlink(missing_ok=True)
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    at.text_area(key="txt").set_value("   ").run()
    at.button(key="sendbtn_btn").click().run()
    assert not list(at.exception), list(at.exception)
    assert not TYPED.exists(), "a whitespace-only message was sent"
    print("2. whitespace-only Send is ignored               OK")


def test_stop_button_replaces_send_during_a_turn() -> None:
    _seed(
        {"kind": "session_start", "ts": time.time() - 60},
        {"kind": "user_text", "text": "hi", "ts": time.time() - 3},
        {"kind": "bot_started", "ts": time.time() - 2},
    )
    STOP.unlink(missing_ok=True)
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    assert "stopbtn_btn" in _keys(at), f"no Stop button while a turn is in flight: {_keys(at)}"
    assert "sendbtn_btn" not in _keys(at), _keys(at)

    at.button(key="stopbtn_btn").click().run()
    assert not list(at.exception), list(at.exception)
    assert STOP.exists(), "Stop did not write the stop file"
    print("3. Stop replaces Send, and writes the stop file   OK")


def test_send_returns_after_the_turn_ends() -> None:
    _seed(
        {"kind": "session_start", "ts": time.time() - 60},
        {"kind": "user_text", "text": "hi", "ts": time.time() - 3},
        {"kind": "bot_started", "ts": time.time() - 2},
        {"kind": "bot_text", "text": "hello", "ts": time.time() - 1},
        {"kind": "bot_stopped", "ts": time.time()},
    )
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert "sendbtn_btn" in _keys(at), f"Send did not come back after bot_stopped: {_keys(at)}"
    assert "stopbtn_btn" not in _keys(at), _keys(at)
    print("4. Send returns once bot_stopped arrives           OK")


def test_stale_turn_does_not_stick_on_stop() -> None:
    """The agent can be killed mid-turn, so no bot_stopped ever arrives. Without
    a staleness guard the button would be stuck on Stop forever, offering to
    interrupt something that is not running."""
    _seed(
        {"kind": "session_start", "ts": time.time() - 600},
        {"kind": "bot_started", "ts": time.time() - 300},
    )
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert "sendbtn_btn" in _keys(at), f"a 300 s old turn still shows Stop: {_keys(at)}"
    print("5. an abandoned turn falls back to Send            OK")


def test_mic_toggle_writes_state() -> None:
    _seed({"kind": "session_start", "ts": time.time()})
    MIC.unlink(missing_ok=True)
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    # Missing file means "enabled", so the first click must mute.
    at.button(key="micbtn_btn").click().run()
    assert not list(at.exception), list(at.exception)
    assert MIC.exists(), "the mic toggle wrote no control file"
    assert json.loads(MIC.read_text())["enabled"] is False, MIC.read_text()

    at.button(key="micbtn_btn").click().run()
    assert json.loads(MIC.read_text())["enabled"] is True, MIC.read_text()
    print("6. mic toggle writes, and flips back               OK")


def test_attach_is_live_and_model_is_still_a_placeholder() -> None:
    """Attach is a working control now; the model picker is still a placeholder.

    The attach control used to be checked here as a disabled button. It is a
    popover holding a file uploader since the upload feature landed, so the
    check moved to what is actually true: the model picker is disabled, and the
    attach control exists and is NOT disabled. The layout test owns the
    geometry; this one owns the enabled/disabled contract.
    """
    _seed({"kind": "session_start", "ts": time.time()})
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)

    model = [b for b in at.button if b.key == "modelph_btn"]
    assert model, "modelph_btn placeholder is missing"
    assert model[0].disabled, "modelph_btn placeholder is not disabled"

    # Attach is a popover, and AppTest does not surface a popover's trigger as
    # a button -- it appears as a Block holding the panel's children. So the
    # check is on what the panel actually contains: a file uploader, which is
    # the thing that makes the control live. Asserting on a button here would
    # pass only while the control was the disabled placeholder.
    uploaders = list(at.get("file_uploader"))
    assert uploaders, "the attach popover holds no file uploader"
    assert uploaders[0].key == "attach_file", uploaders[0].key
    print("7. attach is live, model is a placeholder        OK")


def test_transcript_renders_without_markup_leak() -> None:
    """Model text must not be parsed as HTML: a reply containing '<' would
    otherwise be swallowed by st.markdown(unsafe_allow_html=True)."""
    _seed(
        {"kind": "session_start", "ts": time.time() - 60},
        {"kind": "user_text", "text": "hi", "ts": time.time() - 50},
        {"kind": "bot_text", "text": "a < b and <script>x</script>", "ts": time.time() - 49},
    )
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    blob = "\n".join(m.value for m in at.markdown)
    assert "&lt;script&gt;" in blob, "the reply was not HTML-escaped"
    print("8. transcript escapes model text                   OK")


def test_clear_writes_file_and_empties_the_view() -> None:
    """Clear must do both halves at once.

    The agent's side is the control file (the model forgets the conversation).
    The page's side is the transcript going blank in the SAME run -- the file
    cannot be observed by the browser, so waiting for the agent's ``cleared``
    event would leave the button looking dead for a poll interval. Both are
    asserted here because either one alone is a half-working button.
    """
    _seed(
        {"kind": "session_start", "ts": time.time() - 60},
        {"kind": "user_text", "text": "first question", "ts": time.time() - 50},
        {"kind": "bot_text", "text": "first answer", "ts": time.time() - 49},
        {"kind": "bot_stopped", "ts": time.time() - 48},
    )
    CLEAR.unlink(missing_ok=True)
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    assert "clearbtn_btn" in _keys(at), _keys(at)

    blob = "\n".join(m.value for m in at.markdown)
    assert "first question" in blob, "the seeded conversation did not render"

    at.button(key="clearbtn_btn").click().run()
    assert not list(at.exception), list(at.exception)

    assert CLEAR.exists(), "Clear did not write the clear file"
    payload = json.loads(CLEAR.read_text(encoding="utf-8"))
    assert payload.get("id"), payload
    assert payload.get("ts"), payload

    blob = "\n".join(m.value for m in at.markdown)
    assert "first question" not in blob, "the conversation is still on screen after Clear"
    assert "first answer" not in blob, "the reply is still on screen after Clear"
    print("9. Clear writes the file and empties the view    OK")


def test_clear_survives_a_reload() -> None:
    """A reload must not bring the cleared conversation back.

    The page's own marker lives in session_state, which a fresh session does
    not have -- so the only thing that can hold the history down across a
    reload is the agent's ``cleared`` event in the log. Without it, Clear would
    appear to work and then undo itself on the next refresh.
    """
    now = time.time()
    _seed(
        {"kind": "session_start", "ts": now - 60},
        {"kind": "user_text", "text": "forgotten question", "ts": now - 50},
        {"kind": "bot_text", "text": "forgotten answer", "ts": now - 49},
        {"kind": "bot_stopped", "ts": now - 48},
        {"kind": "cleared", "ts": now - 10},
    )
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    blob = "\n".join(m.value for m in at.markdown)
    assert "forgotten question" not in blob, "history came back after a reload"
    assert "forgotten answer" not in blob, "the reply came back after a reload"
    print("10. cleared history stays cleared across a reload  OK")


def test_theme_toggle_switches_and_persists() -> None:
    """The theme must change in the page AND survive a reload.

    Streamlit's own theme cannot be changed from a running script (measured:
    st._config.set_option("theme.base") left the page at rgb(10,10,10) through
    both a click and a reload), so the dashboard emits its palette as CSS
    variables and carries the choice in the URL. This pins both halves: the
    palette really changes, and the choice is readable on the next run.

    Light is the default, so this also pins the direction of the first render:
    a fresh session must land on the light palette with the dark-mode glyph
    showing, which is the one that says "switch to dark".
    """
    _seed({"kind": "session_start", "ts": time.time()})
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    assert at.session_state["theme"] == "light", at.session_state["theme"]

    # The light palette is the one on :root; the dark one is not.
    blob = "\n".join(m.value for m in at.markdown)
    assert "--fw-bg: #faf9f5" in blob, "the light palette was not emitted"
    assert at.button(key="themebtn_btn").icon == ":material/dark_mode:", (
        at.button(key="themebtn_btn").icon
    )

    at.button(key="themebtn_btn").click().run()
    assert not list(at.exception), list(at.exception)
    assert at.session_state["theme"] == "dark", at.session_state["theme"]
    assert at.query_params.get("theme") == "dark", dict(at.query_params)

    blob = "\n".join(m.value for m in at.markdown)
    assert "--fw-bg: #262624" in blob, "the dark palette was not emitted"
    assert at.button(key="themebtn_btn").icon == ":material/light_mode:", (
        at.button(key="themebtn_btn").icon
    )
    print("11. theme toggles both palettes and the glyph    OK")


def test_mic_glyph_is_a_material_icon() -> None:
    """The mic must be a Material icon, not an emoji.

    An emoji is drawn by the font in its own colours, so a CSS colour on the
    button cannot reach it -- the blue "listening" state would have been
    impossible. The Material glyph inherits `color`, which is what the live
    rule relies on.
    """
    _seed({"kind": "session_start", "ts": time.time()})
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    assert not list(at.exception), list(at.exception)
    btn = [b for b in at.button if b.key == "micbtn_btn"][0]
    assert btn.icon == ":material/mic:", btn.icon
    assert btn.label == "", f"the mic still has a text label: {btn.label!r}"
    print("12. mic is a tintable Material icon              OK")


def main() -> int:
    tests = [
        test_send_writes_typed_and_clears_box,
        test_empty_send_does_nothing,
        test_stop_button_replaces_send_during_a_turn,
        test_send_returns_after_the_turn_ends,
        test_stale_turn_does_not_stick_on_stop,
        test_mic_toggle_writes_state,
        test_attach_is_live_and_model_is_still_a_placeholder,
        test_transcript_renders_without_markup_leak,
        test_clear_writes_file_and_empties_the_view,
        test_clear_survives_a_reload,
        test_theme_toggle_switches_and_persists,
        test_mic_glyph_is_a_material_icon,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            failed += 1
    print("OK" if not failed else f"{failed} FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
