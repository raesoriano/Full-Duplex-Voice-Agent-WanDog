"""Drive the dashboard the way a person does: type, Send, Stop, mute.

Streamlit has no server to call in a test, so this uses AppTest, which runs
dashboard.py in-process and lets a test click widgets and read what was
rendered. That makes the file-level contract checkable end to end:

    Send  -> <log>.typed.json exists with the typed text, and the box is empty
    Stop  -> <log>.stop.json exists
    mic   -> <log>.mic_control.json exists with the new state

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
os.environ["FWEDDIE_DASHBOARD_LOG"] = str(_TMP / "events.jsonl")

from streamlit.testing.v1 import AppTest  # noqa: E402

DASHBOARD = str(Path(__file__).parent / "dashboard.py")
LOG = Path(os.environ["FWEDDIE_DASHBOARD_LOG"])
TYPED = Path(str(LOG) + ".typed.json")
STOP = Path(str(LOG) + ".stop.json")
# From config, never guessed: the mic file is named differently from the
# typed/stop files (see the note in config.py), and guessing cost a debug cycle.
import config  # noqa: E402
MIC = Path(config.DASHBOARD_MIC_FILE)


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

    at.text_input(key="txt").set_value("  what is the weather  ").run()
    at.button(key="sendbtn_btn").click().run()
    assert not list(at.exception), list(at.exception)

    assert TYPED.exists(), "Send did not write the typed-text file"
    payload = json.loads(TYPED.read_text(encoding="utf-8"))
    # Stripped: a stray space from a paste must not become the prompt.
    assert payload["text"] == "what is the weather", payload
    assert payload.get("id"), payload
    assert at.text_input(key="txt").value == "", "the box was not cleared"
    print("1. Send writes the typed file and clears the box  OK")


def test_empty_send_does_nothing() -> None:
    _seed({"kind": "session_start", "ts": time.time()})
    TYPED.unlink(missing_ok=True)
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    at.text_input(key="txt").set_value("   ").run()
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


def test_placeholders_are_disabled() -> None:
    _seed({"kind": "session_start", "ts": time.time()})
    at = AppTest.from_file(DASHBOARD, default_timeout=60).run()
    for key in ("attachph_btn", "modelph_btn"):
        btn = [b for b in at.button if b.key == key]
        assert btn, f"{key} placeholder is missing"
        assert btn[0].disabled, f"{key} placeholder is not disabled"
    print("7. attach + model placeholders are disabled       OK")


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


def main() -> int:
    tests = [
        test_send_writes_typed_and_clears_box,
        test_empty_send_does_nothing,
        test_stop_button_replaces_send_during_a_turn,
        test_send_returns_after_the_turn_ends,
        test_stale_turn_does_not_stick_on_stop,
        test_mic_toggle_writes_state,
        test_placeholders_are_disabled,
        test_transcript_renders_without_markup_leak,
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
