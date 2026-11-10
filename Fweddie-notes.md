# Fweddie — Voice Command Agent

> **Started:** 2026-11-07 · **Machine:** Raspberry Pi 5, 8 GB, aarch64, Debian 13, no GPU
> **Repo:** `Fweddie-repo/` · **Venv:** `Fweddie-repo/.venv` (Python 3.12.16, torch-free)
> **Status:** core voice loop built and import-verified; first live mic test pending

---

## 1. What this is

Fweddie is a **local, full-duplex voice command agent** — the "heart" of the
project. You speak into a mic; it listens, thinks, and answers out loud through
the speaker. The rest of the pipeline (ASR, MCP tools, smart-home dashboard,
music) is added on top of this core **later**.

```
mic ──▶ VAD (Silero ONNX) ──▶ LLM (OnIt, OpenAI-compatible) ──▶ TTS (Piper) ──▶ speaker
        (onnxruntime, no torch)        (raw audio in)                        (local, 24 kHz)
```

## 2. Components

| Component | Choice | Why |
|---|---|---|
| Framework | **Pipecat 1.15.0** | Full-duplex pipeline, barge-in, VAD, transports |
| VAD | **Silero ONNX** via `VADProcessor(SileroVADAnalyzer)` | Runs on `onnxruntime` (CPU) — **no torch** |
| LLM | **OnIt DGX cluster** (`http://202.92.159.240:8001/v1`) via `OpenAILLMService` | OpenAI-compatible vLLM endpoint; model **`Qwen/Qwen3.8-27B`** (Ollama's free cloud models, e.g. `gemma4:31b-cloud`, are no longer available) |
| TTS | **Piper** `en_US-lessac-medium` via `PiperTTSService` | Local, low-latency; voice already on disk |
| Audio I/O | `LocalAudioInput/OutputTransport` (PyAudio/ALSA) | Mic + speaker on the Pi |
| Persona | "Fweddie" — short spoken replies, no markdown | Set in `config.py:SYSTEM_PROMPT` |

### Files
- `agent.py` — the pipeline: transport → VAD → LLM → TTS → output + a small
  `FweddieLogger` observer that logs user/bot speech and LLM text.
- `config.py` — every knob, overridable via env vars (`FWEDDIE_*`, `ONIT_LLM_*`).
- `requirements.txt` — torch-free dependency set (see §6).

## 3. Features (core, v1)
- **Full-duplex / barge-in** — Pipecat's input transport interrupts the bot
  when the user starts speaking (Silero VAD drives it).
- **Raw-audio LLM** — mic audio goes straight to the OnIt endpoint; no separate
  ASR in this stage (keeps the loop minimal and fast).
- **Local TTS** — Piper on CPU, no cloud, ~24 kHz output.
- **Config-driven** — model, base URL, voice, VAD thresholds, sample rates all
  in `config.py` / env.
- **Key-safe** — LLM key resolved the same way OnIt does: `ONIT_LLM_API_KEY`
  env → OS keychain (`onit` / `endpoint_key:<base_url>`, then legacy
  `vllm_api_key`) → `~/.onit/secrets.yaml` → `OPENAI_API_KEY`. A key stored
  once via `onit setup` is picked up with **no manual export**.

## 4. Issues & challenges faced

### Pipecat 1.15 API is NOT the old docs
The classic tutorials (`OpenAILLMContext`, `OpenAILLMContextAggregator`,
`basic_cleaners`, `WebRTCVADAnalyzer`, `pipecat.services.openai.OpenAILLMService`)
**do not exist in 1.15**. Verified working imports:

| Old (docs) | 1.15 actual |
|---|---|
| `pipecat.services.openai.OpenAILLMService` | `pipecat.services.openai.llm.OpenAILLMService` |
| `pipecat.services.ollama.OllamaLLMService` | `pipecat.services.ollama.llm.OLLamaLLMService` (note **OLLama**) |
| `pipecat.services.piper.PiperTTSService` | `pipecat.services.piper.tts.PiperTTSService` |
| `OpenAILLMContext` / `OpenAILLMContextAggregator` | `pipecat.processors.aggregators.llm_context.LLMContext` (aggregator gone) |
| `pipecat.utils.text.basic_cleaners` | **removed** |
| `WebRTCVADAnalyzer` | **removed** (only `silero` + `aic_quail` + `krisp_viva` remain) |
| `SileroVADAnalyzer.VADParams` | `VADParams` lives in `pipecat.audio.vad.vad_analyzer` |
| `SileroVADAnalyzer` in pipeline | **Not a FrameProcessor** — wrap it: `VADProcessor(vad_analyzer=SileroVADAnalyzer(...))` from `pipecat.processors.audio.vad_processor` |
| `PipelineRunner` (pipeline.runner) | Deprecated → `WorkerRunner` from `pipecat.workers.runner`; `add_workers(task)` then `run()`/`cancel()` |
| `OpenAILLMContext` / `OpenAILLMContextAggregator` | `LLMContext` (aggregators moved to `llm_response_universal.LLMContextAggregatorPair`) |
| `StartFrame(context=...)` | **No `context` kwarg** — system prompt goes on `OpenAILLMService.Settings(system_instruction=...)` |
| `BotInterruptionFrame` | `InterruptionFrame` |

Settings are now **dataclass-style `Settings` objects** passed as
`settings=Service.Settings(model=...)`; the old `model=`/`voice_id=` init args
still work but are deprecated (warn, move to settings).

### `PiperTTSService.download_dir` must be a `pathlib.Path`
Passing a `str` raises `AttributeError: 'str' object has no attribute 'exists'`.

### ALSA noise on a headless box
`pyaudio`/ALSA spews `Unknown PCM front / surround…` warnings when there is no
real sound card. Harmless — on the Pi the USB capture card + speaker are the
real devices. (Sandbox has only a null/loopback device, so a live mic test can't
be done here.)

### Keyring needs the D-Bus session address in a headless process
`keyring`'s SecretService backend can't reach the `gnome-keyring-daemon`
without `DBUS_SESSION_BUS_ADDRESS`, even when the daemon and its socket
(`/run/user/1000/bus`) are both present. `config.py:_ensure_dbus_session()`
exports it from `XDG_RUNTIME_DIR` before importing `keyring`. Verified: a
clean shell (no key env vars, no D-Bus address) resolves the 64-char key from
the keychain with no manual export. (Earlier note that the keychain was
"unreachable in this sandbox" is stale — it was just missing the D-Bus
address.)

## 5. How to run
```bash
cd /home/raesoriano/sandbox/Fweddie-repo
export ONIT_LLM_API_KEY=sk-...        # or rely on keychain / OPENAI_API_KEY
.venv/bin/python agent.py
```
Stop with **Ctrl+C**.

## 6. Dependency notes (torch-free)
- **No torch.** Silero VAD runs through `onnxruntime` (CPU). The `silero`
  pipecat extra is intentionally omitted (it pulls the torch-requiring
  `silero-vad` wrapper).
- `webrtcvad` kept as a lightweight fallback (zero deps); `setuptools<81` for
  its `pkg_resources` build.
- Venv: 154 packages, 1.4 GB, verified `import torch` → `ModuleNotFoundError`.

## 7. Safeguards (adapted from OnIt)

Fweddie does **not** re-implement OnIt's safeguards. It spawns OnIt's own stdio
MCP servers (`mcp_bridge.py`), so the policy runs inside the spawned process
and applies to every tool call. Three layers:

1. **AST command allowlist** (`command_policy.py`). Every executable in a
   command -- including inside `$(...)`, pipelines, and loops -- must be on the
   allowlist; package installs are gated and must be version-pinned. Fail-closed:
   anything it cannot statically parse is blocked. On a host the allowlist is off
   by default, so `mcp_bridge` pins `ONIT_COMMAND_ALLOWLIST=1` to turn it on.
2. **DATA_PATH file jail**. `read_file`/`write_file`/`edit_file`/`serve`/`grep`
   are confined to `MCP_DATA_PATH` (default `/tmp/onit/data`); paths outside it
   are refused.
3. **Human-in-the-loop approval** (`approvals.py` + `mcp_bridge.Guard`). A
   command the allowlist will not decide alone is not refused outright: the
   server returns a `needs_approval` ticket (single-use, bound to that exact
   command) and the **Guard** -- the harness side -- shows a person the command
   and re-issues the call carrying the ticket (`approval_token` +
   `approval_scope`). A refusal is remembered for the session so the model
   cannot retry it. Fail-closed: no answer within `MCP_APPROVAL_TIMEOUT` = deny.

**Web search** (`search` tool, 'net' profile). Ollama ended its free cloud
web-search API, so OnIt's search chain no longer has an ollama tier: it is now
Tavily (only when `TAVILY_API_KEY` is set) -> DuckDuckGo (`ddgs`, free, no
key). DuckDuckGo intermittently returns "No results found" under load, so
`_search_ddgs` retries up to 3 times with a 1s pause. Verified live over MCP:
`test_search_live.py` spawns the real 'net' server and checks `search`
returns results for both `type="web"` and `type="news"`.

The two env vars `mcp_bridge` sets when spawning the servers are what make the
safeguards actually fire (without them every ask collapses to a hard deny):

| Var | Effect |
|---|---|
| `ONIT_COMMAND_ALLOWLIST=1` | enforce the AST allowlist (off by default on a host) |
| `ONIT_APPROVAL_CHANNEL=1` | a person is reachable, so ASK -> ticket (else ASK -> deny) |

The `approval_token`/`approval_scope` parameters are **harness-filled**: the
model never sees them, and the server overwrites them on every call, so a value
the model made up never reaches the policy.

Verified end-to-end against the real OnIt server:
- `echo hi` -> runs (allowlisted)
- `lscpu` (unlisted) -> `needs_approval` ticket -> Guard redeems -> runs
- `sudo reboot` (never-ask) -> hard deny, no ticket
- a refused command -> remembered, re-ask is refused without prompting

`test_guard.py` covers the Guard's logic (allow-through, redeem once/session,
refuse + remember, prose passthrough) with a stubbed tool call.

## 8. ASR (local, faster-whisper)

The agent now runs a real ASR stage instead of feeding raw audio to the LLM:

    mic(48k) -> VAD(Silero) -> ASR(faster-whisper small, 16k) -> LLM(text) -> TTS -> speaker

- `WhisperSTTService` (Pipecat) with the `small` model, `int8` compute, `en`.
  The model is already cached under `~/.cache/huggingface`, so **no download and
  no API key**.
- The mic captures at **16 kHz** when ASR is on (faster-whisper wants 16k), not
  the 48 kHz the capture card natively produces. Output stays at the TTS rate.
- Knobs in `config.py`: `FWEDDIE_ASR` (on/off), `FWEDDIE_ASR_MODEL`,
  `FWEDDIE_ASR_DEVICE`, `FWEDDIE_ASR_COMPUTE`, `FWEDDIE_ASR_LANGUAGE`.
  `FWEDDIE_ASR=0` restores the original raw-audio-to-LLM path.
- The ASR stage emits `TranscriptionFrame`; the LLM streams its reply as
  `LLMTextFrame` chunks. `FweddieLogger` buffers the reply and emits one
  `bot_text` event per turn (see section 9).

## 9. Dashboard (Streamlit, LAN-reachable)

`dashboard.py` is a separate Streamlit process that shows the live interaction
from any device on the LAN: a running transcript (your words + Fweddie's
replies), the OnIt tools the agent calls and their results, and a live
**approval panel** where you approve/deny a pending command with a button.

The two processes coordinate through **plain files** (no version-specific
Streamlit/Flask API, so it is robust):

| File | Writer | Reader | Purpose |
|---|---|---|---|
| `/tmp/fweddie_events.jsonl` | agent | dashboard | the interaction record (one JSON per line) |
| `/tmp/fweddie_events.jsonl.pending.json` | agent | dashboard | a command awaiting approval |
| `/tmp/fweddie_events.jsonl.decision.json` | dashboard | agent | the person's decision (once/session/deny) |

- The agent appends events via `eventbus.EventBus` (best-effort; a full disk
  never crashes the voice loop). Events: `session_start/end`, `user_text`,
  `bot_text`, `tool_call`, `tool_result`, `approval_requested/granted/denied`.
- **Live approval:** when a command needs approval, the Guard writes the
  pending file and polls for the decision file. The dashboard shows the command
  and, on button click, writes the decision. If the dashboard is never up, the
  Guard falls back to the **console** prompt, so approvals never block on the
  browser. (Fail-closed: no decision by the timeout = deny.)
- Turn it on with `FWEDDIE_DASHBOARD=1` (off by default). The agent writes the
  event log regardless, so a dashboard started later still shows the session.

Run the dashboard (from the repo, in the venv):

    .venv/bin/python -m streamlit run dashboard.py \
        --server.address 0.0.0.0 --server.port 8501

Then open `http://<host-ip>:8501` from another device on the LAN.

Tests: `test_eventbus.py` covers emit + dashboard read, half-written-line
tolerance, the approval handshake (dashboard -> agent), and the console
fallback when the dashboard is off.

## 10. Audio devices (pinned)

The Pi has two USB audio cards; the agent now pins them so it never picks the
wrong one:

| Role    | PyAudio index | ALSA      | USB device            |
|---------|---------------|-----------|-----------------------|
| Mic     | `0`           | `hw:2,0`  | USB PnP Audio Device  |
| Speaker | `1`           | `hw:3,0`  | USB Composite Device  |

Verified by live capture: the PnP card (index 0) returns real audio on
capture; the Composite card (index 1) is silent (it is the playback/speaker
device). `config.py` sets `INPUT_DEVICE_INDEX=0` and `OUTPUT_DEVICE_INDEX=1`
by default and passes them to `LocalAudioTransportParams`. Override with
`FWEDDIE_IN_DEVICE` / `FWEDDIE_OUT_DEVICE` (an empty value = system default).

## 11. Open questions / next steps
- [ ] **Live mic test on the Pi** — confirm capture card, speaker out, and
      barge-in feel. (Can't be done in this sandbox — no real audio device.)
      No `export` needed: run `onit setup` once to store the key, then just
      `.venv/bin/python agent.py`.
- [ ] Add **music** (YouTube download + playback) as an OnIt tool.
- [ ] Tune VAD thresholds (`confidence`, `stop_secs`) and TTS volume for the room.
- [ ] Decide on a wake word / "always listening" policy.
- [ ] (Optional) Voice the approval prompt: have Fweddie *say* "I need approval
      to run X" and take the answer by ASR, instead of the dashboard button /
      console. The Guard's `prompt` is a pluggable callback, so it's a drop-in.
