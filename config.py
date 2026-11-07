"""Fweddie -- configuration.

All knobs for the voice agent live here. Values can be overridden with
environment variables so the same code runs on the Pi and on a dev box.

LLM key resolution order (first hit wins):
  1. ONIT_LLM_API_KEY env var
  2. OS keychain via `keyring` (service "onit", user "onit")
  3. OPENAI_API_KEY env var (last resort)

If none is found the agent refuses to start with a clear message.
"""

from __future__ import annotations

import os
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


def resolve_llm_api_key() -> str:
    """Return the LLM API key or raise with a clear message."""
    key = os.environ.get("ONIT_LLM_API_KEY")
    if key:
        return key

    try:
        import keyring

        key = keyring.get_password("onit", "onit")
        if key:
            return key
    except Exception:
        pass  # no usable keyring backend -- fall through

    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    raise SystemExit(
        "No LLM API key found. Set ONIT_LLM_API_KEY, store one in the OS "
        'keychain (service "onit", user "onit"), or set OPENAI_API_KEY.'
    )
