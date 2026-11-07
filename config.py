"""Fweddie -- configuration.

All knobs for the voice agent live here. Values can be overridden with
environment variables so the same code runs on the Pi and on a dev box.

LLM key resolution order (first hit wins) -- mirrors how OnIt itself
resolves endpoint keys, so a key stored once by `onit setup` is picked up
by Fweddie with no manual export:
  1. ONIT_LLM_API_KEY env var
  2. OS keychain via `keyring` (service "onit", user
     "endpoint_key:<base_url>" -- the exact key OnIt writes)
  3. OS keychain legacy key "vllm_api_key" (OnIt's fallback for
     non-provider hosts)
  4. ~/.onit/secrets.yaml (OnIt's file fallback when no keychain)
  5. OPENAI_API_KEY env var (last resort)

If none is found the agent refuses to start with a clear message.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PIPER_VOICE_DIR = Path(os.environ.get("FWEDDIE_PIPER_DIR", "/home/raesoriano/.piper"))
PIPER_VOICE = os.environ.get("FWEDDIE_PIPER_VOICE", "en_US-lessac-medium")

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
# The USB capture card on the Pi only accepts 48 kHz for capture
# (16/8/32 kHz fail). Capture at 48 kHz; VAD/ASR run at 16 kHz.
INPUT_SAMPLE_RATE = int(os.environ.get("FWEDDIE_IN_RATE", 48000))
OUTPUT_SAMPLE_RATE = int(os.environ.get("FWEDDIE_OUT_RATE", 24000))

# Optional explicit PyAudio device indices (None = system default).
INPUT_DEVICE_INDEX = os.environ.get("FWEDDIE_IN_DEVICE")
OUTPUT_DEVICE_INDEX = os.environ.get("FWEDDIE_OUT_DEVICE")

# ---------------------------------------------------------------------------
# VAD (Silero ONNX via onnxruntime -- no torch)
# ---------------------------------------------------------------------------
VAD_CONFIDENCE = float(os.environ.get("FWEDDIE_VAD_CONFIDENCE", 0.7))
VAD_START_SECS = float(os.environ.get("FWEDDIE_VAD_START", 0.2))
VAD_STOP_SECS = float(os.environ.get("FWEDDIE_VAD_STOP", 0.8))
VAD_MIN_VOLUME = int(os.environ.get("FWEDDIE_VAD_MIN_VOLUME", 400))

# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible endpoint -- OnIt)
# ---------------------------------------------------------------------------
LLM_BASE_URL = os.environ.get("ONIT_LLM_BASE_URL", "http://202.92.159.240:8001/v1")
LLM_MODEL = os.environ.get("ONIT_LLM_MODEL", "gpt-4o-mini")
LLM_TEMPERATURE = float(os.environ.get("ONIT_LLM_TEMPERATURE", 0.7))
LLM_MAX_TOKENS = int(os.environ.get("ONIT_LLM_MAX_TOKENS", 300))

# ---------------------------------------------------------------------------
# Persona / system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Fweddie, a friendly, upbeat voice assistant. "
    "Keep replies to one or two short sentences -- you are speaking out loud, "
    "not writing. Never use markdown, bullet points, or emoji. "
    "If asked to do something you cannot do yet, say so briefly and cheerfully."
)


def _ensure_dbus_session() -> None:
    """Point `keyring` at the running D-Bus session bus if the address is
    missing. In a headless / sandboxed process the SecretService backend
    cannot reach the gnome-keyring daemon without it, even when the daemon
    and its socket are both present. No-op when the address is already set
    (e.g. a normal desktop login)."""
    if os.environ.get("DBUS_SESSION_BUS_ADDRESS"):
        return
    xdg = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    bus = Path(xdg) / "bus"
    if bus.exists():
        os.environ.setdefault("XDG_RUNTIME_DIR", xdg)
        os.environ.setdefault("DBUS_SESSION_BUS_ADDRESS", f"unix:path={bus}")


def _key_from_keyring(user: str) -> str | None:
    try:
        import keyring

        return keyring.get_password("onit", user)
    except Exception:
        return None  # no usable keyring backend


def _key_from_secrets_yaml() -> str | None:
    """Read OnIt's file fallback (~/.onit/secrets.yaml). OnIt writes the
    key under 'endpoint_key:<url>' (or the legacy 'vllm_api_key') when no
    keychain is available, so we look for the same keys."""
    path = Path.home() / ".onit" / "secrets.yaml"
    if not path.exists():
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return None
    url = LLM_BASE_URL.rstrip("/")
    for candidate in (f"endpoint_key:{url}", "vllm_api_key"):
        val = data.get(candidate)
        if val:
            return str(val)
    return None


def resolve_llm_api_key() -> str:
    """Return the LLM API key or raise with a clear message.

    Resolution mirrors OnIt's own order (see module docstring) so a key
    stored once via `onit setup` is shared with Fweddie.
    """
    # 1. Explicit env var.
    key = os.environ.get("ONIT_LLM_API_KEY")
    if key:
        return key

    # 2-3. OS keychain -- the exact keys OnIt writes for this endpoint.
    _ensure_dbus_session()
    url = LLM_BASE_URL.rstrip("/")
    for user in (f"endpoint_key:{url}", "vllm_api_key"):
        key = _key_from_keyring(user)
        if key:
            return key

    # 4. OnIt's file fallback.
    key = _key_from_secrets_yaml()
    if key:
        return key

    # 5. Last resort.
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    raise SystemExit(
        "No LLM API key found. Run `onit setup` to store one, set "
        "ONIT_LLM_API_KEY, or set OPENAI_API_KEY."
    )
