"""WanDog -- configuration.

All knobs for the voice agent live here. Values can be overridden with
environment variables so the same code runs on the Pi and on a dev box.

LLM key resolution order (first hit wins) -- mirrors how OnIt itself
resolves endpoint keys, so a key stored once by `onit setup` is picked up
by WanDog with no manual export:
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
PIPER_VOICE_DIR = Path(os.environ.get("WANDOG_PIPER_DIR", "/home/raesoriano/.piper"))
# Ryan (US English, medium quality). The ONNX model + its .onnx.json config
# live in PIPER_VOICE_DIR; PiperTTSService downloads them on first use if they
# are missing. 61 MB -- see "The TTS voice is Piper Ryan" in WanDog-Notes.md.
PIPER_VOICE = os.environ.get("WANDOG_PIPER_VOICE", "en_US-ryan-medium")

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
# The USB audio devices on this Pi only accept 48 kHz -- for BOTH capture
# and playback. Every other rate we tried (8k/16k/22.05k/32k/44.1k/96k)
# fails with "Errno -9997 Invalid sample rate", so we always open the
# hardware at 48 kHz and convert in software where a lower rate is needed
# (ASR/VAD want 16 kHz; Piper TTS emits 22.05 kHz and gets resampled up by
# the output transport).
INPUT_SAMPLE_RATE = int(os.environ.get("WANDOG_IN_RATE", 48000))
OUTPUT_SAMPLE_RATE = int(os.environ.get("WANDOG_OUT_RATE", 48000))

# PyAudio device indices. Pinned by default to the two USB devices on the Pi:
#   index 0 = "USB PnP Audio Device"   (hw:2,0) -- the MIC (verified: it is the
#              card that actually returns audio on capture)
#   index 1 = "USB Composite Device"   (hw:3,0) -- the SPEAKER (playback)
# Override with WANDOG_IN_DEVICE / WANDOG_OUT_DEVICE (an empty string means
# "use the system default" instead of a pinned index).
def _device_index(env: str, default: int) -> int | None:
    raw = os.environ.get(env)
    if raw is None:
        return default
    raw = raw.strip()
    return int(raw) if raw else None


INPUT_DEVICE_INDEX = _device_index("WANDOG_IN_DEVICE", 0)
# ALSA card number for the capture device. PortAudio index 0 == ALSA card 2
# on this Pi. Used only to READ and report the current gain at startup --
# the agent never writes it. The gain is yours to set, because it depends on
# the room and how close you sit:
#
#     amixer -c 2 sset Mic 70% cap
#
# (34% is too quiet -- Silero scores real speech at ~0.05 and the agent looks
# deaf. 100% is too hot -- it clips on speech and picks up the keyboard and
# the speaker's own output. Somewhere in the middle is the sweet spot.)
# Set WANDOG_MIC_CARD="" to skip even the read.
MIC_CARD = os.environ.get("WANDOG_MIC_CARD", "2")
OUTPUT_DEVICE_INDEX = _device_index("WANDOG_OUT_DEVICE", 1)

# ---------------------------------------------------------------------------
# ASR (local, no API key)
# ---------------------------------------------------------------------------
# Two backends, selected with WANDOG_ASR_BACKEND:
#
#   "streaming" (default) -- sherpa-onnx streaming zipformer. Decodes the mic
#       audio as it arrives, so the words appear WHILE you are still talking.
#       Measured here: first partial ~1.0 s after you start speaking, and the
#       final text ready the instant you stop. Accuracy on this mic is good
#       for ordinary words but poor on numbers ("2 plus 2" -> "TOO PLUS"),
#       so by default a faster-whisper pass produces the authoritative final
#       text (see ASR_FINAL_MODEL) while the streaming model drives the live
#       partials. Speech end -> final transcript: ~1.6 s.
#
#   "whisper" -- the old segmented faster-whisper path: nothing at all until
#       you stop speaking, then ~4.9 s of transcription. Kept as a fallback
#       and for A/B comparison (WANDOG_ASR_BACKEND=whisper).
#
# Both want 16 kHz, so the 48 kHz capture is resampled down once, before VAD.
ASR_ENABLED = os.environ.get("WANDOG_ASR", "1") != "0"
ASR_BACKEND = os.environ.get("WANDOG_ASR_BACKEND", "streaming").strip().lower()

# --- streaming backend ------------------------------------------------------
# Directory holding the sherpa-onnx streaming zipformer. Downloaded once into
# the repo (models/ is gitignored); the service raises a clear error naming the
# download command if it is missing.
ASR_STREAM_MODEL_DIR = os.environ.get(
    "WANDOG_ASR_STREAM_DIR",
    str(Path(__file__).with_name("models")
        / "sherpa-onnx-streaming-zipformer-en-2023-06-26"),
)
# ONNX Runtime threads for the encoder. The Pi has 4 cores and must also run
# VAD, the LLM client and Piper, so leave headroom.
ASR_THREADS = int(os.environ.get("WANDOG_ASR_THREADS", 2))
# Second, non-streaming pass that produces the final text. "none" disables it
# and uses the streaming model's own output as the final transcript.
ASR_FINAL_PASS = os.environ.get("WANDOG_ASR_FINAL", "whisper").strip().lower()
# tiny.en transcribes this Pi's audio in 0.86 s and is word-perfect on the test
# captures; base.en is 1.6 s and small.en 4.9 s for no gain on short turns.
ASR_FINAL_MODEL = os.environ.get("WANDOG_ASR_FINAL_MODEL", "tiny.en")
# Audio kept from before the VAD fires (it needs ~0.25 s of speech to trigger,
# and without this the streaming decoder starts mid-word).
ASR_PREROLL_SECS = float(os.environ.get("WANDOG_ASR_PREROLL", 1.0))

# --- whisper backend (segmented) -------------------------------------------
ASR_MODEL = os.environ.get("WANDOG_ASR_MODEL", "small")
ASR_DEVICE = os.environ.get("WANDOG_ASR_DEVICE", "auto")   # auto|cpu|cuda
ASR_COMPUTE = os.environ.get("WANDOG_ASR_COMPUTE", "int8")  # int8 is fast on the Pi
ASR_LANGUAGE = os.environ.get("WANDOG_ASR_LANGUAGE", "en")
# Resample the 48 kHz capture to 16 kHz for ASR (both backends want 16k).
ASR_SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# VAD (Silero ONNX via onnxruntime -- no torch)
# ---------------------------------------------------------------------------
# Tuned for this Pi's USB mic, from measurements on real captures
# (/tmp/wandog_diag_mic.wav, /tmp/wandog_speech.wav, resampled to 16 kHz):
#   Silero confidence is sharply bimodal -- quiet p50 0.067, speech p50 1.000
#   -- so the confidence gate is the ONLY gate that actually discriminates.
#   0.35 let keyboard clicks and room noise through, and the agent kept
#   interrupting itself (18 user_started events in one live session). 0.6 sits
#   in the empty gap between the two modes: still fires on real speech, ignores
#   clicks. The old comment here claimed speech peaks at 0.45-0.47; measured on
#   this mic it reaches 1.00, so that was wrong.
# start_secs is the debounce: at 0.15s a single keypress counted as speech.
# 0.25s requires a sound to last longer than a click before a turn starts.
VAD_CONFIDENCE = float(os.environ.get("WANDOG_VAD_CONFIDENCE", 0.6))
VAD_START_SECS = float(os.environ.get("WANDOG_VAD_START", 0.25))
# The real silence gate, in combination with TURN_STOP_SECS below.
#
# WARNING: until now this value was never actually in force. agent.py handed
# ONE SileroVADAnalyzer instance to both VADProcessor and the LLM user
# aggregator, and a VADAnalyzer is stateful: both controllers called
# ``analyze_audio`` on every frame, so each 32 ms window was counted twice and
# every gate ran at half its configured length -- 0.7 s behaved as 0.35 s,
# start_secs 0.25 as 0.13. That is why keyboard clicks kept starting turns
# (the 0.25 s debounce was really 0.13 s) and why a 0.8 s pause split a
# sentence. Each consumer now gets its own analyzer, so the number below is
# the number that runs.
VAD_STOP_SECS = float(os.environ.get("WANDOG_VAD_STOP", 0.7))

# ---------------------------------------------------------------------------
# Turn end ("the user has finished asking")
# ---------------------------------------------------------------------------
# Pipecat 1.15 defaults to the Smart Turn analyzer, which waits 3 s of silence
# after the VAD stop before it will call the turn over -- and only then does
# the STT get asked for its text. Measured on this Pi, the gap from "VAD says
# you stopped talking" to "turn submitted to the LLM" was 5.2 s on every single
# turn:
#
#     0.7 s VAD  +  3.0 s Smart Turn  +  1.0 s STT wait  ~=  5.2 s
#
# With streaming ASR the transcript is already in hand when you stop, so none
# of that wait buys anything. This switches the stop strategy to a plain
# speech-timeout: end the turn TURN_STOP_SECS after the VAD stop, once a
# transcript has arrived.
#
# The turn cannot end before a transcript arrives, so this timer only costs
# what it exceeds the final pass. Measured on this Pi (one 0.95 s utterance,
# separate VAD analyzers, twice each, /tmp/t_senate.wav):
#
#     TURN_STOP   VAD stop -> turn end     silence gate
#       0.6 s          0.94 s                1.30 s
#       1.0 s          1.00 s                1.70 s
#       1.2 s          1.20 s                1.90 s
#       1.6 s          1.60 s                2.30 s
#
# tiny.en finishes ~0.91 s after the VAD stop, so everything up to ~0.9 s is
# FREE -- the turn is waiting for the transcript anyway. 0.6 -> 1.0 therefore
# costs 0.06 s and buys 0.4 s of extra pause tolerance. Past 1.0 the timer
# becomes the binding constraint and every extra 0.2 s is paid in full.
#
# Why not lower, and why this matters: a pause longer than the gate ends the
# turn, and the words said after it open a NEW turn. Measured in
# /tmp/wandog_events.jsonl (19:41:05, "Senate elections." [pause] "as in
# senators. ..."): the pause split one question into two and the clarifying
# phrase was answered as if it were a new question. 1.0 s covers an ordinary
# clause pause without adding perceptible lag.
TURN_STOP_SECS = float(os.environ.get("WANDOG_TURN_STOP", 1.0))
# Hard ceiling: seconds without ANY user activity before an open turn is
# closed regardless of what the stop strategy thinks.
#
# This is Pipecat's own default (5.0) and must not be lowered to a value
# near the length of a spoken question. It is an inactivity ceiling, not a
# transcription deadline: the timer is reset by audio activity, but the
# transcript for the segment is produced after the VAD stop and takes
# ~0.9-2.0 s. Set it to 3.0 and any question longer than ~3 s from its first
# word to the moment the transcript lands gets force-closed with NO user
# message at all -- the aggregator then answers a "user spoke but was not
# recognised, ask them to repeat" prompt. Measured on a 3.5 s question:
# turn closed 31 ms before the VAD stop, prompt contained system+developer
# only, and the reply was "could you repeat that" -- the agent "not
# answering". Leave this at 5.0.
TURN_STOP_TIMEOUT_SECS = float(os.environ.get("WANDOG_TURN_STOP_TIMEOUT", 5.0))

# Pipecat 1.15 measures volume on a 0.0-1.0 scale (BS.1770 loudness), not the
# old int16 amplitude. A value like 400 (the pre-1.15 default) makes
# `volume >= min_volume` always False, so VAD never fires and barge-in dies.
#
# Measured on this Pi, the loudness curve saturates almost immediately: a pure
# tone at amplitude 0.01 already scores 0.66, 0.05 -> 0.80, 0.30 -> 0.96,
# 0.50 -> 1.00. Speech and room noise on this mic both land around 0.15-0.19,
# so this gate passes nearly everything and CANNOT separate them. It is kept
# only as a floor against digital silence; VAD_CONFIDENCE does the real work.
# Do not raise it hoping to reject noise -- it will just deafen the agent.
# (The old comment claiming far-field speech peaks at 0.75 was wrong.)
VAD_MIN_VOLUME = float(os.environ.get("WANDOG_VAD_MIN_VOLUME", 0.35))

# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible endpoint -- OnIt's DGX cluster)
# ---------------------------------------------------------------------------
# Default is the Ollama cloud OpenAI-compatible API (DGX cluster is down).
# The /v1 suffix is REQUIRED: pipecat POSTs to {base_url}/chat/completions and
# Ollama only serves that route under /v1. Without it you get a 404 (not a
# 401), which looks like a bad model name but is a bad path.
#   https://ollama.com/v1        <- canonical, works (200)
#   https://api.ollama.com/v1    <- 301-redirects to ollama.com/v1
#   https://api.ollama.com       <- 301 -> 404, the original failure
LLM_BASE_URL = os.environ.get("ONIT_LLM_BASE_URL", "https://ollama.com/v1")
LLM_MODEL = os.environ.get("ONIT_LLM_MODEL", "gemma4:31b-cloud")
LLM_TEMPERATURE = float(os.environ.get("ONIT_LLM_TEMPERATURE", 0.7))
LLM_MAX_TOKENS = int(os.environ.get("ONIT_LLM_MAX_TOKENS", 300))

# ---------------------------------------------------------------------------
# The cap for a turn that calls a tool, which is NOT the cap for speech.
# ---------------------------------------------------------------------------
# LLM_MAX_TOKENS above is a SPEECH budget: one or two sentences, ~40 tokens.
# Applying it to a turn that emits a tool call is a category error, because the
# tool's arguments come out of the same budget and are not speech.
#
# Measured 18 Nov 2026 (gemma4:31b-cloud, the real 14-tool schemas). Asking for
# a flappy bird clone, the model emits its arguments in the order
#
#     {"content": "<the entire 5 KB HTML file>", "path": "flappy_bird.html"}
#
# -- the file body first, the path LAST. At 300 tokens the stream is cut inside
# `content`, so the JSON is closed by the server without ever reaching `path`:
#
#     max_tokens= 300  finish=length     arglen~1127  keys=['content']
#     max_tokens= 600  finish=length     arglen~2030  keys=['content']
#     max_tokens=1000  finish=length     arglen~3900  keys=['content']
#     max_tokens=4096  finish=tool_calls arglen~5840  keys=['content','path']
#
# The tool then answers "Missing required argument(s): path." -- which is true,
# and completely misleading: nothing was missing from the model's intent, the
# argument was simply past 300 tokens. In /tmp/wandog_events.jsonl that is 24
# of 26 write_file calls, every one of them that same error, including a run of
# 11 consecutive attempts and the agent apologising that it "keeps forgetting to
# include the file path". It never forgot; it was never allowed to finish.
#
# 8192 covers a ~20 KB file (measured: 16.6-19.5 KB arguments, finish=tool_calls,
# path present, 3 of 3). It costs no extra latency on speech: the cap is a
# ceiling, not a target, and "What is the capital of France?" measured 31 chars
# and 0.6 s at both 300 and 8192.
LLM_TOOL_MAX_TOKENS = int(os.environ.get("ONIT_LLM_TOOL_MAX_TOKENS", 8192))
# Qwen3.8 is a thinking model: by default it spends its token budget on
# internal reasoning and returns an empty `content` (the reasoning arrives in
# a separate field), which would make the voice agent say nothing. vLLM
# accepts `chat_template_kwargs` to switch the template's thinking off.
# Set ONIT_LLM_THINKING=1 to restore the model's default (thinking on).
LLM_THINKING = os.environ.get("ONIT_LLM_THINKING", "0") == "1"

# ---------------------------------------------------------------------------
# OnIt MCP tools (spawn OnIt's own stdio servers and expose their tools)
# ---------------------------------------------------------------------------
# WanDog talks to OnIt's tools the same way OnIt does: it spawns OnIt's own
# stdio MCP servers (the bash/file/search/web servers, launched by
# `python -m src.mcp.servers.run`) and speaks the MCP protocol to them. The
# safeguards -- the AST command allowlist, the approval gate, and the
# DATA_PATH file jail -- all live INSIDE those spawned server processes, so
# WanDog inherits them for free without re-implementing any of it.
ONIT_REPO = os.environ.get("ONIT_REPO", "/home/raesoriano/onit")
ONIT_PYTHON = os.environ.get(
    "ONIT_PYTHON", os.path.join(ONIT_REPO, ".venv", "bin", "python")
)

# The jail root for the file tools. OnIt's bash server defaults its DATA_PATH
# to <tmp>/onit/data; we pin the same value so a tool call can only read/write
# inside it (read_file may also reach DOCUMENTS_PATH). Keep this a directory
# WanDog is allowed to touch.
MCP_DATA_PATH = os.environ.get(
    "WANDOG_MCP_DATA_PATH", "/tmp/onit/data"
)
# Extra read-only documents dir for read_file (OnIt's DOCUMENTS_PATH).
MCP_DOCUMENTS_PATH = os.environ.get("WANDOG_MCP_DOCS_PATH", "")

# Which OnIt servers to spawn. Each becomes a stdio subprocess of OnIt's
# interpreter. 'local' = bash + file + local search; 'net' = web search +
# weather. 'prompts' is not a tool server, so it is not listed here.
MCP_SERVERS = [
    {"name": "ToolsLocalMCPServer", "module": "tasks.tools", "profile": "local"},
    {"name": "ToolsNetMCPServer", "module": "tasks.tools", "profile": "net"},
]

# Master switch for the tool bridge (off = WanDog just talks, no tools).
MCP_TOOLS_ENABLED = os.environ.get("WANDOG_MCP_TOOLS", "1") != "0"
# Per-tool-call wall clock (seconds). OnIt's bash tool caps itself at 300s;
# this is a slightly looser outer guard so a wedged call cannot hang the
# voice loop.
MCP_TOOL_TIMEOUT = float(os.environ.get("WANDOG_MCP_TOOL_TIMEOUT", 320))
# How long to wait for a person to answer an approval prompt before treating
# it as a refusal (fail-closed). OnIt's own tickets expire after 900s; we are
# tighter because a voice agent should not hang mid-conversation.
MCP_APPROVAL_TIMEOUT = float(os.environ.get("WANDOG_MCP_APPROVAL_TIMEOUT", 120))
# How much of a tool result the MODEL is allowed to see, in characters.
# The full result is kept for the dashboard and the session folder; only this
# much is handed back to the LLM.
#
# The cap is applied STRUCTURALLY by mcp_bridge._cap_for_model: a JSON list
# keeps whole results and says how many were dropped, instead of being cut
# mid-string. A blind slice at 600 chars turned a 5-result news payload into
# 1.6 broken results -- the model got a title and a dangling fragment, and
# could not see the `date` field it needed to tell old news from new.
#
# WHY 24000 AND NOT 1200. The old 1200 was chosen for a voice reply of one or
# two sentences, before anything was measured. Measured on 17 Nov 2026, every
# shape the tools actually return was being truncated at 1200, and two of them
# lost the answer rather than just the bulk:
#
#   ddgs news, sorted to 5 results   2,073 ch   1200 -> 2 of 5   24000 -> 5/5
#   Tavily news, 5 results           6,694 ch   1200 -> 2 of 5   24000 -> 5/5
#   web search, 5 full page bodies   7,432 ch   1200 -> 2 of 5   24000 -> 5/5
#   ollama web, 5 results           22,970 ch   1200 -> 2 of 5   24000 -> 5/5
#   read_file                       12,094 ch   1200 -> clipped  24000 -> whole
#
# 24000 is the smallest round number at which *no observed payload* is cut at
# all, so the cap goes back to being what it was meant to be -- a guard against
# a pathological result, not the thing that decides the answer. The headroom is
# real, not assumed: the endpoint accepts 1,000,000 chars (172,443 prompt
# tokens) and only refuses at 2,000,000 ("exceeded max context length by
# 82,713 tokens"), i.e. a ceiling near 1.5M chars. A heavy session in the event
# log was 13 tool calls / 60,583 raw chars; at 24000 each that is 312,000 chars,
# about a fifth of the ceiling. Latency is not a factor either: 120k chars
# answered in 2.5s, 600k in 4.6s.
#
# The cap stays HARD rather than becoming a prompt suggestion. A prompt can ask
# the model to be brief, but the model does not choose the size of a tool
# result -- the server does, and one 23k-char web page arrives whole. Past the
# ceiling the endpoint returns HTTP 400, which reaches the pipeline as an
# ErrorFrame that agent.py only logs, so the turn goes silent instead of short.
# The prompt therefore *describes* the cap (see TOOLS_SYSTEM_PROMPT) while this
# number enforces it.
#
# 0 disables the cap (the old behaviour).
MCP_MODEL_RESULT_CHARS = int(os.environ.get("WANDOG_MCP_MODEL_RESULT_CHARS", 24000))

# Search retries. OnIt's news branch calls ddgs once and turns any exception
# into {"error": "Search failed: No results found."} -- ddgs RAISES rather than
# returning an empty list when its aggregator is rate-limited, which measured
# at roughly 2 failures in 8 calls. The web branch already retries 3x inside
# OnIt; the news branch never got that fix. We cannot change OnIt here, so the
# bridge retries the call itself. Only read-only `search` calls are retried:
# a retried `bash` or `write_file` could run twice.
MCP_SEARCH_RETRIES = int(os.environ.get("WANDOG_MCP_SEARCH_RETRIES", 3))
MCP_SEARCH_RETRY_DELAY = float(os.environ.get("WANDOG_MCP_SEARCH_RETRY_DELAY", 1.5))

# Should a recency-shaped search be sent as type="news" even when the model
# said nothing about it? Measured over 13 real `search` calls in
# /tmp/wandog_events.jsonl: the model passed no `type` argument once, so
# every call hit the general index, whose results carry NO date. The tool
# description does tell it to use "news" for recency -- it does not do it.
# This is the deterministic backstop. Set to 0 to let the model decide alone.
MCP_NEWS_AUTODETECT = os.environ.get("WANDOG_MCP_NEWS_AUTODETECT", "1") == "1"

# How many news results to ASK OnIt for, regardless of how many the model
# wants. ddgs returns news in relevance order, not date order, so the newest
# item is often not in the first five: measured on "Alaska US Senate election
# results" on 17 Nov 2026, the five-result page was 17, 12, 17, 14, 11 Nov --
# and the model cap only has room for the first two, so the agent reported
# "November 12" while 17 Nov stories sat further down the same payload. The
# fix is to over-fetch, sort by date, then trim back to what was asked for.
# 10 is ddgs' practical ceiling for one news page.
MCP_NEWS_OVERFETCH = int(os.environ.get("WANDOG_MCP_NEWS_OVERFETCH", 10))

# Sort a news payload newest-first before the model sees it, and label each
# result with its age ("today", "3 days ago"). Off = keep ddgs' own order.
MCP_NEWS_SORT = os.environ.get("WANDOG_MCP_NEWS_SORT", "1") == "1"

# ddgs returns a DIFFERENT page for the same query on consecutive calls: six
# calls on "Alaska US Senate election results" split into two distinct sets,
# one with a same-day story and one without. So a stale page often means "this
# draw missed it", not "no news exists". When the newest result is older than
# MCP_NEWS_FRESH_DAYS, the search is re-run up to MCP_NEWS_REFRESH_RETRIES more
# times and the NEWEST payload wins. Measured over 6 rounds: newest item was
# from today on 3/6 first calls and 5/6 with two extra calls; mean latency
# 0.8s -> 2.2s, hidden behind the spoken announcement.
MCP_NEWS_REFRESH_RETRIES = int(os.environ.get("WANDOG_MCP_NEWS_REFRESH_RETRIES", 2))
MCP_NEWS_REFRESH_DELAY = float(os.environ.get("WANDOG_MCP_NEWS_REFRESH_DELAY", 0.6))
# A page whose newest item is this many days old or newer counts as fresh
# enough to stop re-asking. 0 = only a same-day story stops the retry.
MCP_NEWS_FRESH_DAYS = int(os.environ.get("WANDOG_MCP_NEWS_FRESH_DAYS", 0))
# Hard ceiling on the whole news refresh, in seconds. ddgs occasionally takes
# 5-10s for a single call, and this is a voice turn: a retry that pushes the
# answer past this budget is worse than a page that is a day old. Measured
# worst case without it: 10.6s to first speech.
MCP_NEWS_REFRESH_BUDGET = float(os.environ.get("WANDOG_MCP_NEWS_REFRESH_BUDGET", 4.0))

# Which backend serves a news search.
#
# OnIt's `search(type="news")` branch calls ddgs ONLY -- it never reaches the
# Tavily tier that its own web branch uses (mcp_server.py: `if type == "news":
# results = ddgs.news(...)`, no Tavily call at all). That is why news was
# consistently older than a Google search of the same query.
#
# Measured 17 Nov 2026, newest published date per provider, same four queries:
#
#   query                                DDG news     Tavily (topic=news, days=1)
#   Alaska US Senate election results    15 Nov       17 Nov
#   Alaska Senate race Peltola Sullivan  17 Nov       17 Nov
#   US Senate election results 2026      14 Nov       17 Nov
#   latest news Jensen Huang Nvidia      15 Nov       17 Nov
#
# DDG was 0-3 days behind; Tavily was same-day on all four. Tavily's news topic
# also carries a real `published_date` and honours a `days` window, which ddgs
# does not. So when a key is configured, news goes to Tavily first and falls
# back to OnIt's ddgs branch on failure or an empty page.
#
# "tavily" (default) = Tavily first, ddgs fallback. "onit" = always ddgs.
# The key is read from the same secrets store as the other tool secrets; with
# no key this setting is inert and news stays on ddgs.
MCP_NEWS_BACKEND = os.environ.get("WANDOG_MCP_NEWS_BACKEND", "tavily").strip().lower()
# Tavily's news recency window, in days. Tavily accepts 1-7 here. 1 = "today",
# which is what "latest news" means; widen it if a query legitimately has no
# same-day coverage (a race called three days ago).
MCP_NEWS_TAVILY_DAYS = int(os.environ.get("WANDOG_MCP_NEWS_TAVILY_DAYS", 1))
# Tavily "advanced" depth returns real published dates and better recall than
# "basic" on news queries. Costs the same credits per call.
MCP_NEWS_TAVILY_DEPTH = os.environ.get(
    "WANDOG_MCP_NEWS_TAVILY_DEPTH", "advanced").strip().lower()

# ---------------------------------------------------------------------------
# Dashboard (Streamlit, reachable from other devices on the LAN)
# ---------------------------------------------------------------------------
# The dashboard is a separate process (dashboard.py) that tails a shared
# event log the agent appends to, and exchanges live approval decisions with
# the agent through two small files next to that log. Off by default so the
# agent runs headless on the Pi with no extra surface; set WANDOG_DASHBOARD=1
# to turn it on (the agent then also writes the log so a dashboard can show
# the session).
DASHBOARD_ENABLED = os.environ.get("WANDOG_DASHBOARD", "0") == "1"
# Where the agent writes the JSONL event log the dashboard tails. The
# approval files live next to it: <log>.pending.json and <log>.decision.json.
DASHBOARD_LOG = os.environ.get("WANDOG_DASHBOARD_LOG", "/tmp/wandog_events.jsonl")
# Streamlit bind: 0.0.0.0 so other devices on the LAN can open it.
DASHBOARD_HOST = os.environ.get("WANDOG_DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.environ.get("WANDOG_DASHBOARD_PORT", 8501))

# How often the agent polls the dashboard's control files (mic toggle, typed
# text, stop button). These are plain files next to the log -- the same
# mechanism the mic toggle has always used -- so the dashboard can stay a
# separate process that may crash or never start without taking the voice loop
# with it.
#
# 0.2 s: a person who types a question and clicks Send expects the agent to
# start answering immediately, and this is the only latency the poll adds. The
# cost is four small stat() calls per second, which is nothing next to the
# audio callback. Do not raise this to 1 s "to save cycles": the stop button
# would then take up to a second to actually stop the bot talking, which reads
# as the button being broken.
DASHBOARD_CONTROL_POLL_SECS = float(
    os.environ.get("WANDOG_DASHBOARD_CONTROL_POLL", "0.2"))

# The typed-text inbox. The dashboard writes {"id": ..., "text": ...} here when
# the Send button is pressed; the agent consumes the file (deletes it) so the
# same message is never answered twice.
DASHBOARD_TYPED_FILE = os.environ.get(
    "WANDOG_DASHBOARD_TYPED",
    str(Path(DASHBOARD_LOG).with_name(Path(DASHBOARD_LOG).name + ".typed.json")))

# The stop button. The dashboard writes {"id": ..., "ts": ...}; the agent
# consumes it and interrupts the turn exactly as a barge-in would.
DASHBOARD_STOP_FILE = os.environ.get(
    "WANDOG_DASHBOARD_STOP",
    str(Path(DASHBOARD_LOG).with_name(Path(DASHBOARD_LOG).name + ".stop.json")))

# Clear chat history. The dashboard writes {"id": ..., "ts": ...}; the agent
# consumes it and empties the shared LLM context, so "clear" means the model
# forgets the conversation rather than the page merely hiding it. Both halves
# are needed: hiding alone would come back on reload, and clearing alone would
# leave the page looking unchanged until the agent's next poll.
DASHBOARD_CLEAR_FILE = os.environ.get(
    "WANDOG_DASHBOARD_CLEAR",
    str(Path(DASHBOARD_LOG).with_name(Path(DASHBOARD_LOG).name + ".clear.json")))

# The mic toggle. Unlike the two above this file is *state*, not an action: it
# persists, and the agent reads it every poll rather than consuming it.
#
# Named by appending to the log's filename, not with Path.with_suffix(). With
# with_suffix() the name was "wandog_events.mic_control.json" -- the ".jsonl"
# was REPLACED -- while the other two control files appended. Two naming rules
# for three sibling files in one directory is a trap: the dashboard test looked
# for the appended name, found nothing, and reported the working toggle as
# broken. Default unchanged from the old with_suffix() path so an existing
# deployment keeps reading the same file.
DASHBOARD_MIC_FILE = os.environ.get(
    "WANDOG_DASHBOARD_MIC",
    str(Path(DASHBOARD_LOG).with_suffix(".mic_control.json")))

# ---------------------------------------------------------------------------
# Persona / system prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are WanDog, a friendly, upbeat voice assistant. "
    "Keep replies to one or two short sentences -- you are speaking out loud, "
    "not writing. Never use markdown, bullet points, or emoji. "
    "If asked to do something you cannot do yet, say so briefly and cheerfully."
)

# Who the agent is talking to. Kept separate from SYSTEM_PROMPT so the persona
# can be reused by a second person without editing it, and appended in the same
# place ``agent.py`` composes the prompt.
#
# WHY A NAME AT ALL. Without it the model has no anchor for "who am I talking
# to" and answers "I don't know your name" -- measured on the live pipeline,
# where "What is my name?" produced exactly that. The GitHub handle is here
# because the github_repo tool and the `git` in the bash tool both act on
# raesoriano's account: a question like "what did I push last?" needs the model
# to know whose repos "mine" means, and the token in secrets.yaml is that same
# account's, so the two agree.
OWNER_NAME = "Ron Andrei Soriano"
OWNER_GITHUB = "raesoriano"

OWNER_PROMPT = (
    f"You are speaking with {OWNER_NAME}, whose GitHub username is "
    f"{OWNER_GITHUB}. When they say 'I', 'me' or 'my', they mean him -- "
    f"including 'my repos', which are the GitHub repositories owned by "
    f"{OWNER_GITHUB}. If asked who they are, say his name. Never guess a "
    f"different name, and never claim not to know it."
)

# Appended to the system prompt only when the tool bridge is active, so the
# model knows it has tools and how to use them.
#
# The date line is added at call time by ``tools_system_prompt()`` rather than
# baked in here: a module-level constant would freeze whatever day the process
# imported it on, and a long-running agent would then believe it is still that
# day. Without a date the model cannot judge "latest" at all -- the search tool
# description tells it to use type="news" when recency matters, but it has no
# way to know what "recent" means.
# The rule that stops the model inventing a clock time. Defined before
# TOOLS_SYSTEM_PROMPT because that string is built from it, and kept as its own
# constant so it can also apply when the bridge is off: an agent with no tools
# has a clock in its prompt but no `date` to run, and it still must not fill the
# gap from memory.
CLOCK_GUARD = (
    "Never state a clock time you have not been given or read from a tool."
)

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
    "and move on. "
    # Recency guidance. This is the half of the fix that lives in the prompt;
    # the other half is config.MCP_NEWS_AUTODETECT, which sets the argument
    # when the model forgets it. Both were missing, and the event log shows
    # the model never passed `type` on its own in 13 out of 13 calls.
    "For anything about news, current events, prices, scores, or 'the latest' "
    "or 'who won', search with type set to news so the results carry dates. "
    "Prefer the most recent result and mention its date when you speak. If a "
    "result looks older than the question asks for, say so instead of "
    "presenting it as current. Never call a result 'the latest' unless its "
    "date supports that. "
    # Clock guidance. The date line in ``clock_line()`` is what broke the
    # clock: it says "Today is Tuesday, 17 November 2026", and a model holding
    # a date with no time will happily invent one -- measured at 3 out of 3
    # trials on "What is the time?", answering 11:44 AM, 1:15 PM, 11:38 AM in
    # three consecutive calls, never touching the bash tool it had been using
    # for the same question before the date line existed. The line below is
    # what stops the invention; ``clock_line()`` supplies the real value.
    "Never state a clock time you have not been given or read from a tool. "
    "For a time in another city or timezone, run `date` with the bash tool "
    "and a TZ prefix -- for example TZ=\"America/Anchorage\" date -- and read "
    "the time from its output. "
    # GitHub. Two separate failures were measured on the live model before this
    # paragraph existed, and both are about *guessing* rather than calling:
    #
    #   "Can you push to my GitHub?"     -> "I can't push directly to GitHub
    #                                        yet" -- false; git and curl are
    #                                        allowlisted and the token works
    #   "Can you see my GitHub repos?"   -> "I sure can!" -- no call, so it
    #                                        never found out either way
    #   "Do you have my GitHub token?"   -> "I do not have access to your
    #                                        GitHub token." -- false
    #
    # The token was present and working the whole time: github_repo action=list
    # returned real repos in 1.8s. The model was inferring its own capability
    # from the tool description, and the description says "Requires
    # GITHUB_TOKEN", which reads as a caveat rather than as "this is set".
    #
    # The rule is therefore the same shape as the clock rule: never answer a
    # question about its own access from memory, and confirm with a call before
    # it acts. A read is its own confirmation; a write is not.
    "You have GitHub access through the `github_repo` tool and through `git` "
    "and `curl` in bash; the token is configured as GITHUB_TOKEN. Never "
    "claim you cannot reach "
    "GitHub, and never claim you can, without checking first. For a question "
    "about access or about what is there, call `github_repo` with action list "
    "or get and answer from what it returned. Before any call that changes "
    "something -- create, fork or delete a repository, a repository setting "
    "such as Pages or visibility, or a push -- say which "
    "account and which repository you are about to change and wait for the "
    "person to confirm. A read needs no confirmation; a write always does. "
    "If GitHub refuses because the token is missing or lacks a scope, say "
    "exactly that; do not describe a missing permission as not having access. "
    # The name shape. github_repo takes a BARE name to create but owner/repo
    # to get, fork and delete, and the model sent a bare name for all four.
    # The server pastes `name` into /repos/{name}, so a bare name 404s -- and
    # 404 is the same answer GitHub gives for a repository that is really
    # gone. On 18 Nov 2026 that produced 13 failed deletes in a row while the
    # repositories sat in plain sight in the model's own `list` output, and
    # the person was told four times that they "might have already been
    # removed". mcp_bridge._shape_github_args now qualifies the name, so the
    # call works whether or not the model remembers; this line is what makes
    # the model's own reasoning and its report to the person correct.
    "In `github_repo`, `name` is a bare name only for create; for get, fork "
    "and delete it must be owner/repo -- for you that means "
    f"{OWNER_GITHUB}/<name>. A 404 on a repository your own list just showed "
    "is almost always a malformed name, not a deleted repository: check the "
    "name before you conclude it is gone, and never tell the person a "
    "repository has already been removed on the strength of a 404 alone. "
    # Repository settings the tool does not cover. github_repo has exactly
    # five actions -- create, get, list, fork, delete -- so the model answered
    # "enable GitHub Pages" with "I don't have a tool that can change the
    # internal settings of a GitHub repository" and, asked twice, repeated
    # "I am sure." (event log 18 Nov 2026, idx 2038 and 2050). That is false:
    # `curl` is allowlisted, the token is in the bash environment, and
    #
    #   POST /repos/{owner}/{repo}/pages  {"source":{"branch":"main","path":"/"}}
    #   -> 201 with html_url          (measured on a public repo, 18 Nov 2026)
    #   DELETE /repos/{owner}/{repo}/pages -> 204
    #
    # The distinction to teach is narrow: "this one tool lacks an action" is
    # not "I have no way to do it". A missing action means reach for the REST
    # API, and say so only after the call, not instead of it.
    "`github_repo` covers only create, get, list, fork and delete. Anything "
    "else about a repository -- GitHub Pages, visibility, topics, settings -- "
    "is still reachable through the REST API with `curl` in bash, because "
    "GITHUB_TOKEN is in that environment and `curl` is allowed. Never answer "
    "\"I have no tool for that\" about a repository setting: call the API and "
    "report what it said. To enable Pages, POST to "
    "https://api.github.com/repos/<owner>/<repo>/pages with "
    "{\"source\":{\"branch\":\"main\",\"path\":\"/\"}}; a 422 saying the plan "
    "does not support Pages means the repository is private, which is a fact "
    "to report, not a missing tool. "
    # Long tasks. `bash` is a FOREGROUND call with a wall clock on it, and the
    # voice loop is suspended while it runs -- a measured 60s `sleep` held the
    # tool call for the full 60s. `serve` is the detached alternative: it
    # returns in about half a second with a pid and a log path, and `action
    # logs` reads it later.
    #
    # Measured on the live model before this paragraph existed:
    #   "run this command and tell me when it finishes: ./build.sh"
    #       -> bash("./build.sh")            unbounded, could hold the turn for
    #                                        the whole 320s tool timeout
    #   "the test suite takes about 15 minutes"
    #       -> serve(...)                    correct, but only because the
    #                                        question happened to say so
    #
    # So the rule is stated rather than left to the word "minutes" appearing in
    # the question: anything that may outlast a sentence goes to serve.
    "A command that may take more than a few seconds -- a build, a test "
    "suite, an install, a download, a script -- must go through the `serve` "
    "tool, not `bash`. `bash` is foreground and the person hears nothing "
    "until it returns; `serve` starts it detached and returns at once. After "
    "starting one, tell the person it is running and that you can check on "
    "it. Use `serve` with action logs to report progress, and action status "
    "to see whether it finished. Do not claim a long task is done until "
    "`serve` says it stopped. "
    # Truncation guidance. The hard cap in MCP_MODEL_RESULT_CHARS is the
    # enforcement; this is the half that lives in the prompt, and it is the
    # half the model can actually act on. A truncated payload says so in a
    # `note` field ("showing 3 of 5 results"), and without this line the model
    # reads the note as part of the answer -- or worse, does not read it and
    # reports the three it can see as everything the search found. Asking for
    # fewer results is a real lever: a 5-result news payload fits whole at
    # 24000 chars, and asking for 20 of them does not.
    "A tool result may be shortened to fit your context; when it is, a note "
    "field says how much was shown and how much was cut. Treat what you got "
    "as partial when the note says so, and never describe a shortened list as "
    "everything that exists. If you need more than you were shown, ask the "
    "tool again for fewer results or a narrower query rather than repeating "
    "the same call. "
    # Writing a file. The model emits tool arguments in the order it likes, and
    # for write_file it puts the whole body first and `path` LAST. A reply cut
    # off by the token limit therefore loses exactly the one argument the tool
    # cannot work without, and the tool's own error ("Missing required
    # argument(s): path.") reads as a mistake the model made rather than a
    # truncated reply. Measured 18 Nov 2026: 24 of 26 write_file calls failed
    # that way, 11 of them in a row, with the agent apologising that it "keeps
    # forgetting to include the file path".
    #
    # The cap is now large enough (config.LLM_TOOL_MAX_TOKENS) and the bridge
    # explains a pathless call, but the model can also avoid the situation:
    # name the file first, and build a long file in parts instead of one
    # enormous argument.
    "When you write a file, always pass `path`, and put it in the call before "
    "the content. A very long file may not fit in one reply: if a write is cut "
    "off, do not repeat the same call -- write the first part with "
    "mode=\"write\", then add the rest with further calls using mode=\"append\"."
)


def _zone_name() -> str:
    """The IANA zone name of the host, e.g. ``Asia/Manila``.

    ``tzname()`` is not enough on its own. This host is on Asia/Manila, whose
    abbreviation is **PST** -- Philippine Standard Time. The clock line used to
    read "11:26 AM (PST, UTC+0800)", and PST is far more widely read as
    *Pacific* Standard Time, which is UTC-0800. The line contradicted itself:
    a reader who trusts the letters gets a 16-hour error, and a reader who
    trusts the offset gets the right answer by ignoring the letters. Measured
    on the live prompt, the model read the offset correctly, but the ambiguity
    is free to remove and would bite a person reading the transcript.

    Falls back to the abbreviation when /etc/localtime does not name a zone.
    """
    from pathlib import Path
    try:
        target = Path("/etc/localtime").resolve()
        parts = target.parts
        if "zoneinfo" in parts:
            name = "/".join(parts[parts.index("zoneinfo") + 1:])
            if name:
                return name
    except OSError:
        pass
    from datetime import datetime
    return datetime.now().astimezone().tzname() or "local"


def clock_line() -> str:
    """One line stating the real date AND time, to put in front of the model.

    Both halves matter, and the first version of this shipped only one. A date
    with no clock reads to the model as "you know what time it is", so it
    fills the gap: the event log has it answering "It is currently Tuesday,
    November 17, 2026" to "What is the time?", and after the date line landed
    it answered "The current time is 12:00 PM" -- a value it was never given.

    Measured against the real toolset, 3 trials per question:

        prompt                      "What is the time?"   bash calls
        date only (0850bb2a)        invented a time       9/12
        date + clock, no hint       correct, no tool      12/12
        date + clock + hint above   correct, no tool      12/12

    The clock is re-read on every call, so a long-running agent cannot drift.
    It is the local zone: ``datetime.now().astimezone()`` follows /etc/localtime,
    which is what a person standing in front of the mic means by "the time".

    The zone is named as well as offset. See ``_zone_name`` for why: this host's
    abbreviation is PST, which reads as Pacific, not Philippine.
    """
    from datetime import datetime
    now = datetime.now().astimezone()
    return (
        f"The current date and time here is "
        f"{now.strftime('%A, %d %B %Y, %I:%M %p')} "
        f"({_zone_name()}, {now.tzname()}, UTC{now.strftime('%z')}). "
    )


def tools_system_prompt() -> str:
    """The tools section of the system prompt, with the current date and time.

    Call this once per pipeline build, not once per import: a module-level
    constant would freeze whatever moment the process imported it on, and a
    long-running agent would then believe it is still that moment. Without a
    date the model cannot judge "latest" at all -- the search tool description
    tells it to use type="news" when recency matters, but it has no way to
    know what "recent" means.
    """
    return clock_line() + TOOLS_SYSTEM_PROMPT


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
    stored once via `onit setup` is shared with WanDog.
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


# ---------------------------------------------------------------------------
# Tool secrets
# ---------------------------------------------------------------------------
# The tool servers are separate processes, so a key reaches them only through
# the environment we hand to `stdio_client`. They cannot consult the keychain
# themselves -- `keyring` lives in this process, not in theirs.
#
# Resolution order for every tool key (first hit wins):
#   1. repo-local secrets.yaml      -- per-project override, reviewed in a diff
#   2. OS keychain, service "onit"  -- exactly what `onit setup` writes
#   3. leave it unset               -- the tool degrades, the agent still runs
#
# OnIt's own order is env -> keychain -> file (setup.get_secret). We keep its
# keyring names, so a machine already provisioned by `onit setup` needs no new
# file at all: the file is an override on top, never a prerequisite. That is
# deliberate -- a fresh clone with no secrets.yaml must still work.
#
# Passing these into the spawn environment is also what fixes the weather
# tool: nothing used to forward them, so the server saw no
# OPENWEATHERMAP_API_KEY even with the key sitting in the keychain, and
# returned "OpenWeather API key not set".
TOOL_SECRETS_PATH = Path(
    os.environ.get(
        "WANDOG_SECRETS_FILE",
        str(Path(__file__).resolve().parent / "secrets.yaml"),
    )
)

# keyring key -> env var the tool server actually reads. The left side is the
# name OnIt stores under; the right side is what the server consults.
#
# `ollama_api_key` is deliberately NOT forwarded. OnIt's search chain is Tavily
# -> DuckDuckGo only (web_search.py:180-213); it has no Ollama tier, so the key
# bought nothing there. Ollama's web search is metered per call, so a key that
# is merely present is a liability: any server that reads OLLAMA_API_KEY would
# start spending on the first search. The key stays in secrets.yaml because the
# LLM endpoint uses it via ONIT_LLM_BASE_URL; it is never handed to a tool
# server.
TOOL_SECRETS: dict[str, str] = {
    "openweathermap_api_key": "OPENWEATHERMAP_API_KEY",  # get_weather
    "tavily_api_key": "TAVILY_API_KEY",                  # search, first tier
    "github_token": "GITHUB_TOKEN",                      # github_repo, git push
    "huggingface_token": "HF_TOKEN",                     # model downloads
}

# Secrets we refuse to forward even if a future TOOL_SECRETS entry names them.
# Separate from the map so the reason survives, and so a test can assert it.
FORBIDDEN_TOOL_SECRETS: dict[str, str] = {
    "ollama_api_key": "Ollama web search is metered per call; never expose it "
                      "to a tool server",
}

_secrets_cache: dict | None = None
_secrets_cache_mtime: float | None = None


def _clean_secret(value) -> str | None:
    """Coerce a YAML scalar to a usable string, or None.

    Unquoted YAML turns `12345` into an int and `on`/`no` into a bool; str()
    keeps the value intact instead of letting a `len()` or `.strip()` fail
    later. Empty and whitespace-only values mean "not configured".
    """
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _repo_secrets() -> dict:
    """Read the repo-local secrets.yaml, re-reading only when it changes.

    Cached on mtime so a long-running agent picks up an edit without a
    restart, and the file is parsed once per change rather than once per
    tool call. A missing or malformed file is not an error -- it just means
    "no overrides", and resolution falls through to the keychain.
    """
    global _secrets_cache, _secrets_cache_mtime

    path = TOOL_SECRETS_PATH
    try:
        stat = path.stat()
    except OSError:
        _secrets_cache, _secrets_cache_mtime = None, None
        return {}

    if _secrets_cache is not None and _secrets_cache_mtime == stat.st_mtime:
        return _secrets_cache

    # A world-readable secrets file is a mistake worth naming, but not worth
    # refusing to start over.
    if stat.st_mode & 0o077:
        print(
            f"[config] warning: {path} is readable by other users "
            f"(mode {oct(stat.st_mode & 0o777)}); consider chmod 600",
            file=sys.stderr,
        )

    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # malformed YAML, permissions, bad encoding
        print(f"[config] ignoring unreadable {path}: {exc}", file=sys.stderr)
        data = {}

    if not isinstance(data, dict):
        print(f"[config] {path} is not a mapping -- ignoring it", file=sys.stderr)
        data = {}

    _secrets_cache, _secrets_cache_mtime = data, stat.st_mtime
    return data


def _secret_source(name: str) -> tuple[str | None, str]:
    """Resolve one secret and say where it came from.

    Returns (value, source) where source is "file", "keychain" or "unset".
    """
    value = _clean_secret(_repo_secrets().get(name))
    if value:
        return value, "file"

    _ensure_dbus_session()
    value = _key_from_keyring(name)
    if value:
        return value, "keychain"

    return None, "unset"


def resolve_tool_secret(name: str) -> str | None:
    """Return one tool secret by its OnIt keyring name, or None."""
    return _secret_source(name)[0]


def tool_secret_env() -> dict[str, str]:
    """Every resolvable tool key as {env_var: value}.

    Missing keys are omitted, never set to "". An empty env var reads as
    present-but-blank to some HTTP clients, which produces a worse error than
    an absent one (401 instead of "not configured").
    """
    resolved: dict[str, str] = {}
    for name, env_var in TOOL_SECRETS.items():
        if name in FORBIDDEN_TOOL_SECRETS:
            continue
        value = resolve_tool_secret(name)
        if value:
            resolved[env_var] = value
    return resolved


def describe_tool_secrets() -> list[str]:
    """One human-readable line per tool key, for the startup log.

    Names and sources only -- never a value or a prefix of one.
    """
    lines = []
    for name, env_var in TOOL_SECRETS.items():
        value, source = _secret_source(name)
        detail = f"len={len(value)}" if value else "MISSING"
        lines.append(f"{name} -> {env_var} [{source}] {detail}")
    return lines
