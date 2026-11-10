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

# PyAudio device indices. Pinned by default to the two USB devices on the Pi:
#   index 0 = "USB PnP Audio Device"   (hw:2,0) -- the MIC (verified: it is the
#              card that actually returns audio on capture)
#   index 1 = "USB Composite Device"   (hw:3,0) -- the SPEAKER (playback)
# Override with FWEDDIE_IN_DEVICE / FWEDDIE_OUT_DEVICE (an empty string means
# "use the system default" instead of a pinned index).
def _device_index(env: str, default: int) -> int | None:
    raw = os.environ.get(env)
    if raw is None:
        return default
    raw = raw.strip()
    return int(raw) if raw else None


INPUT_DEVICE_INDEX = _device_index("FWEDDIE_IN_DEVICE", 0)
OUTPUT_DEVICE_INDEX = _device_index("FWEDDIE_OUT_DEVICE", 1)

# ---------------------------------------------------------------------------
# ASR (faster-whisper, local -- no API key)
# ---------------------------------------------------------------------------
# The agent used to feed raw audio straight to the LLM. It now runs a local
# ASR stage (faster-whisper) so the LLM consumes TEXT. The `small` model is
# already cached under ~/.cache/huggingface, so no download is needed.
ASR_ENABLED = os.environ.get("FWEDDIE_ASR", "1") != "0"
ASR_MODEL = os.environ.get("FWEDDIE_ASR_MODEL", "small")
ASR_DEVICE = os.environ.get("FWEDDIE_ASR_DEVICE", "auto")   # auto|cpu|cuda
ASR_COMPUTE = os.environ.get("FWEDDIE_ASR_COMPUTE", "int8")  # int8 is fast on the Pi
ASR_LANGUAGE = os.environ.get("FWEDDIE_ASR_LANGUAGE", "en")
# Resample the 48 kHz capture to 16 kHz for ASR (faster-whisper wants 16k).
ASR_SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# VAD (Silero ONNX via onnxruntime -- no torch)
# ---------------------------------------------------------------------------
VAD_CONFIDENCE = float(os.environ.get("FWEDDIE_VAD_CONFIDENCE", 0.7))
VAD_START_SECS = float(os.environ.get("FWEDDIE_VAD_START", 0.2))
VAD_STOP_SECS = float(os.environ.get("FWEDDIE_VAD_STOP", 0.8))
VAD_MIN_VOLUME = int(os.environ.get("FWEDDIE_VAD_MIN_VOLUME", 400))

# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible endpoint -- OnIt's DGX cluster)
# ---------------------------------------------------------------------------
# The DGX cluster serves Qwen3.8-27B through vLLM with an OpenAI-compatible
# API. Ollama's free cloud models (e.g. gemma4:31b-cloud) are no longer
# available, so the default is the on-prem DGX model.
LLM_BASE_URL = os.environ.get("ONIT_LLM_BASE_URL", "http://202.92.159.240:8001/v1")
LLM_MODEL = os.environ.get("ONIT_LLM_MODEL", "Qwen/Qwen3.8-27B")
LLM_TEMPERATURE = float(os.environ.get("ONIT_LLM_TEMPERATURE", 0.7))
LLM_MAX_TOKENS = int(os.environ.get("ONIT_LLM_MAX_TOKENS", 300))
# Qwen3.8 is a thinking model: by default it spends its token budget on
# internal reasoning and returns an empty `content` (the reasoning arrives in
# a separate field), which would make the voice agent say nothing. vLLM
# accepts `chat_template_kwargs` to switch the template's thinking off.
# Set ONIT_LLM_THINKING=1 to restore the model's default (thinking on).
LLM_THINKING = os.environ.get("ONIT_LLM_THINKING", "0") == "1"

# ---------------------------------------------------------------------------
# OnIt MCP tools (spawn OnIt's own stdio servers and expose their tools)
# ---------------------------------------------------------------------------
# Fweddie talks to OnIt's tools the same way OnIt does: it spawns OnIt's own
# stdio MCP servers (the bash/file/search/web servers, launched by
# `python -m src.mcp.servers.run`) and speaks the MCP protocol to them. The
# safeguards -- the AST command allowlist, the approval gate, and the
# DATA_PATH file jail -- all live INSIDE those spawned server processes, so
# Fweddie inherits them for free without re-implementing any of it.
ONIT_REPO = os.environ.get("ONIT_REPO", "/home/raesoriano/onit")
ONIT_PYTHON = os.environ.get(
    "ONIT_PYTHON", os.path.join(ONIT_REPO, ".venv", "bin", "python")
)

# The jail root for the file tools. OnIt's bash server defaults its DATA_PATH
# to <tmp>/onit/data; we pin the same value so a tool call can only read/write
# inside it (read_file may also reach DOCUMENTS_PATH). Keep this a directory
# Fweddie is allowed to touch.
MCP_DATA_PATH = os.environ.get(
    "FWEDDIE_MCP_DATA_PATH", "/tmp/onit/data"
)
# Extra read-only documents dir for read_file (OnIt's DOCUMENTS_PATH).
MCP_DOCUMENTS_PATH = os.environ.get("FWEDDIE_MCP_DOCS_PATH", "")

# Which OnIt servers to spawn. Each becomes a stdio subprocess of OnIt's
# interpreter. 'local' = bash + file + local search; 'net' = web search +
# weather. 'prompts' is not a tool server, so it is not listed here.
MCP_SERVERS = [
    {"name": "ToolsLocalMCPServer", "module": "tasks.tools", "profile": "local"},
    {"name": "ToolsNetMCPServer", "module": "tasks.tools", "profile": "net"},
]

# Master switch for the tool bridge (off = Fweddie just talks, no tools).
MCP_TOOLS_ENABLED = os.environ.get("FWEDDIE_MCP_TOOLS", "1") != "0"
# Per-tool-call wall clock (seconds). OnIt's bash tool caps itself at 300s;
# this is a slightly looser outer guard so a wedged call cannot hang the
# voice loop.
MCP_TOOL_TIMEOUT = float(os.environ.get("FWEDDIE_MCP_TOOL_TIMEOUT", 320))
# How long to wait for a person to answer an approval prompt before treating
# it as a refusal (fail-closed). OnIt's own tickets expire after 900s; we are
# tighter because a voice agent should not hang mid-conversation.
MCP_APPROVAL_TIMEOUT = float(os.environ.get("FWEDDIE_MCP_APPROVAL_TIMEOUT", 120))

# ---------------------------------------------------------------------------
# Dashboard (Streamlit, reachable from other devices on the LAN)
# ---------------------------------------------------------------------------
# The dashboard is a separate process (dashboard.py) that tails a shared
# event log the agent appends to, and exchanges live approval decisions with
# the agent through two small files next to that log. Off by default so the
# agent runs headless on the Pi with no extra surface; set FWEDDIE_DASHBOARD=1
# to turn it on (the agent then also writes the log so a dashboard can show
# the session).
DASHBOARD_ENABLED = os.environ.get("FWEDDIE_DASHBOARD", "0") == "1"
# Where the agent writes the JSONL event log the dashboard tails. The
# approval files live next to it: <log>.pending.json and <log>.decision.json.
DASHBOARD_LOG = os.environ.get("FWEDDIE_DASHBOARD_LOG", "/tmp/fweddie_events.jsonl")
# Streamlit bind: 0.0.0.0 so other devices on the LAN can open it.
DASHBOARD_HOST = os.environ.get("FWEDDIE_DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.environ.get("FWEDDIE_DASHBOARD_PORT", 8501))

# ---------------------------------------------------------------------------
# Persona / system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are Fweddie, a friendly, upbeat voice assistant. "
    "Keep replies to one or two short sentences -- you are speaking out loud, "
    "not writing. Never use markdown, bullet points, or emoji. "
    "If asked to do something you cannot do yet, say so briefly and cheerfully."
)

# Appended to the system prompt only when the tool bridge is active, so the
# model knows it has tools and how to use them.
TOOLS_SYSTEM_PROMPT = (
    "You have tools you can call to act: run shell commands, read and write "
    "files, search the web and local documents, and check the weather. "
    "When a request needs one, call the tool with the right arguments, then "
    "answer in one or two short spoken sentences using what it returned. "
    "If a tool refuses a command, say plainly what it refused and why -- do "
    "not try to route around the refusal. Never invent tool results. "
    "Some commands need a person's approval before they run. When a tool "
    "tells you it needs approval, say briefly that you are asking a person, "
    "then wait for the tool to finish -- do not retry it, do not rephrase it, "
    "and do not try another route around it. If the person declines, say so "
    "and move on."
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
