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
| VAD | **Silero ONNX** `SileroVADAnalyzer`, owned by the `LLMUserAggregator` | Runs on `onnxruntime` (CPU) — **no torch**. In 1.15 the analyzer is *not* a standalone `VADProcessor`; it is handed to the user aggregator, whose `UserTurnController` + `VADUserTurnStartStrategy` drive barge-in (see §13) |
| LLM | **OnIt DGX cluster** (`http://202.92.159.240:8001/v1`) via `OpenAILLMService` | OpenAI-compatible vLLM endpoint; model **`Qwen/Qwen3.8-27B`** (Ollama's free cloud models, e.g. `gemma4:31b-cloud`, are no longer available). Qwen3.8 is a *thinking* model — by default it spends its tokens on internal reasoning and returns empty `content`, so `agent.py` sends `chat_template_kwargs.enable_thinking=false` (vLLM) unless `ONIT_LLM_THINKING=1` |
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

# One-time: the streaming ASR model (321 MB, gitignored)
cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
cd ..

export ONIT_LLM_API_KEY=sk-...        # or rely on keychain / OPENAI_API_KEY
.venv/bin/python agent.py
```
Stop with **Ctrl+C**. Skip the download with `FWEDDIE_ASR_BACKEND=whisper`
(segmented, slower, no 321 MB).

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

## 8. ASR (streaming, 2026-11-17)

    mic(48k) -> VAD(Silero) -> ASR(streaming zipformer partials + whisper tiny.en final) -> LLM(text) -> TTS -> speaker

**Why it changed.** The old path was segmented: `WhisperSTTService` buffered the
whole utterance and only started transcribing after the VAD reported the user
stopped. `small` int8 takes ~4.9 s per segment on this Pi, and it processed
segments sequentially, so turn 2's words arrived ~7.3 s after the user stopped
talking -- long after that turn had closed. The LLM then answered whatever it
had, which was the previous question.

**Now** (`streaming_asr.py`, `SherpaStreamingSTTService`):

| measurement | before | now |
|---|---|---|
| first partial word after speech start | never | 0.4-0.7 s |
| final transcript after speech end | 4.7-4.9 s | 0.86 s |
| VAD stop -> turn submitted to the LLM | 5.2 s | 0.93 s |

- **zipformer streaming** (sherpa-onnx, int8, RTF ~0.10) decodes 20 ms chunks as
  they arrive and emits `InterimTranscriptionFrame` continuously. It is fast but
  weak on numbers (`"2 plus 2"` -> `"TOO PLUS"`), so it is not the final word.
- **faster-whisper `tiny.en`** re-transcribes the same buffered audio for the
  authoritative `TranscriptionFrame`. `tiny.en` is 0.86 s and word-perfect on the
  test captures; `base.en` is 1.6 s and `small.en` 4.9 s for no gain on short
  turns. The final pass is **skipped when the streaming model found no words** --
  otherwise Whisper hallucinated a full sentence (`"Well did I not? What's in the
  nog?"`) out of a silent segment and put it in the prompt as if the user said it.
- Model weights: 321 MB, downloaded into `models/` (gitignored). The service
  raises a `FileNotFoundError` naming the exact `curl` command when missing.
  `FWEDDIE_ASR_BACKEND=whisper` restores the old segmented path with no download.
- Knobs: `FWEDDIE_ASR_BACKEND`, `FWEDDIE_ASR_FINAL`, `FWEDDIE_ASR_FINAL_MODEL`,
  `FWEDDIE_ASR_PREROLL` (1.0 s kept from before the VAD fires, so the decoder does
  not start mid-word), `FWEDDIE_ASR_THREADS` (2 -- the Pi also runs VAD, the LLM
  client and Piper), `FWEDDIE_TURN_STOP` (0.6 s speech-timeout).
- **Rejected:** the 20M zipformer returned empty strings on this mic's audio. The
  2023-06-26 model works on every capture. `WhisperSTTService`'s segmented
  default remains available for A/B.

**The regression this introduced and its fix.** The tap that publishes the live
draft line overrode `FrameProcessor.process_frame` without calling `super()`,
which skipped `StartFrame` handling and stalled the pipeline behind it -- the
transcript decoded but never reached `LLMUserAggregator`. See section 12(b).

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

## 11. LLM: DGX Qwen3.8-27B (post-Ollama-cloud)
- Ollama ended its free cloud models, so the LLM is the on-prem DGX cluster:
  `Qwen/Qwen3.8-27B` served by vLLM at `http://202.92.159.240:8001/v1`
  (OpenAI-compatible).
- **API key lives in the OS keychain** (gnome-keyring, service `onit`, entry
  `endpoint_key:http://202.92.159.240:8001/v1`) — stored by `onit setup`,
  resolved by `config.resolve_llm_api_key()` (env → keychain →
  `~/.onit/secrets.yaml` → `OPENAI_API_KEY`). No export needed at run time.
- **Thinking disabled by default.** Qwen3.8 is a thinking model: with thinking
  on, vLLM returns empty `content` (the reasoning goes to a separate field),
  which would make the voice agent silent. `agent.py` therefore adds
  `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`;
  `ONIT_LLM_THINKING=1` restores the model default.
- Verified end-to-end: `test_llm_live.py` builds pipecat's `OpenAILLMService`
  exactly as `agent.py` does (keychain key, thinking off) and gets a real
  reply from the DGX model.

## 12. Turn handling & the silent-agent regression (fixed 2026-11-17)

Three separate faults produced what looked like one symptom ("it answers the
previous question, or not at all"). They are worth keeping straight, because
each one masked the next.

**a) The LLM never saw its own answers.** `LLMUserAggregator` and
`LLMAssistantAggregator` were each given their own `LLMContext`. Only the user
half pushes `LLMContextFrame`, and that frame carries `user_agg._context` --
that object *is* the prompt. The assistant half wrote replies into a context
nothing ever sent, so every prompt was a growing stack of unanswered `user`
messages with no `assistant` turns, and the model picked whichever it liked.
Every `Generating LLM response` line in the log showed `messages=2/3/4` with
roles `[system, developer]`, `[system, developer, user]` -- never a completed
exchange. **Fix:** one shared `LLMContext`. Prompt 2 is now
`[system, user, assistant, user]`.

**b) The partial-transcript tap stalled the pipeline.** The tap added to publish
the live draft overrode `FrameProcessor.process_frame` without calling
`super()`. That base method handles `StartFrame` -- it marks the stage started,
which is what lets the pipeline become ready -- plus `CancelFrame` and
`InterruptionFrame`. Skipping it left the handshake stalled behind that stage,
so every later frame queued behind it, including the final
`TranscriptionFrame`. The words decoded and appeared in the terminal, then
vanished. The aggregator saw an empty aggregation and asked the user to repeat.
**Fix:** call `super()`. `test_tap_forward.py` hangs before, forwards now;
`test_endtoend_answer.py` drives the real pipeline and fails unless `bot_text`
appears.

**c) The turn inactivity ceiling was too short.** It had been lowered to 3.0 s
(Pipecat's default is 5.0). It is an inactivity ceiling, not a transcription
deadline, and 3.0 s sits right at the length of a spoken question -- a 3.5 s
question was force-closed 31 ms *before* the VAD stop and landed on the same
empty-turn path. **Fix:** reverted to 5.0 s, with the 0.6 s
`SpeechTimeoutUserTurnStopStrategy` doing the real turn closing.

**Why the tests passed while the live agent failed:** `test_pipeline_asr.py`
used 1.4 s and 1.8 s clips -- shorter than the 3 s timeout -- never included the
tap, and never checked for a reply. It now covers 3.5 s and 5.6 s, and
`test_endtoend_answer.py` asserts on the spoken answer.

**Mic gain.** `maximise_mic_gain()` used to run unconditionally from
`build_pipeline()` and force the capture card to 496/496 (+31 dB) on every
start. At that level the mic clips on speech and Silero scores the keyboard and
the speaker's own output as speech. It is now `report_mic_gain()`: it reads the
level with `amixer ... sget` and logs it, and changes nothing. Set the gain
yourself; the agent leaves it alone.

## 13. Barge-in (fixed 2026-11-10)

Barge-in — the bot stops talking the instant you start speaking — was **broken
in two independent ways** in the Pipecat 1.15 upgrade. Both are fixed and
verified by `test_bargein.py` (all 3 checks PASS).

**How barge-in works in 1.15.** It is *not* driven by a standalone
`VADProcessor` (the old pattern). The `SileroVADAnalyzer` is handed to the
`LLMUserAggregator`; its internal `UserTurnController` +
`VADUserTurnStartStrategy` turn a `VADUserStartedSpeakingFrame` into an
`InterruptionFrame`, which every processor (including the TTS output transport)
handles by stopping the bot's speech.

**Bug 1 — the user aggregator was only created when a tool bridge existed.**
In the default config (ASR on, no tools) there was *no* `LLMUserAggregator` in
the pipeline at all, so there was no turn controller and barge-in silently did
nothing. **Fix:** `agent.py` now *always* builds an `LLMUserAggregator` with the
VAD analyzer and a `VADUserTurnStartStrategy`, and puts it in the pipeline
(`mic → ASR → user agg → LLM → [assistant agg] → TTS`). The assistant aggregator
is still only added when there are tools (it records replies into context).

**Bug 2 — `VAD_MIN_VOLUME` was on the wrong scale.** Pipecat 1.15 measures
volume as a **0.0–1.0** float (BS.1770 loudness), but `config.py` still used the
old int16-amplitude value `400`. The state machine requires
`confidence >= VAD_CONFIDENCE AND volume >= VAD_MIN_VOLUME`, so
`volume (0–1) >= 400` was *always* False → VAD never reached SPEAKING → no
barge-in even with the wiring. **Fix:** `VAD_MIN_VOLUME` is now a float, default
`0.6` (the Pipecat default), overridable via `FWEDDIE_VAD_MIN_VOLUME`.

**Verification** (`test_bargein.py`):
1. **Wiring** — the real `build_pipeline` output contains an
   `LLMUserAggregator` with a VAD analyzer + `VADUserTurnStartStrategy`.
2. **Trigger** — the exact `UserTurnController` + `VADUserTurnStartStrategy`
   the aggregator uses fires `on_user_turn_started` with
   `enable_interruptions=True` on a `VADUserStartedSpeakingFrame`.
3. **VAD** — the real `SileroVADAnalyzer` reaches `SPEAKING` on real English
   speech (generated with espeak-ng), proving a real user's voice will trigger
   the path above.

## 14. Open questions / next steps
- [x] **Streaming ASR** — text appears while the user is still talking; see
      section 8.
- [x] **Silent agent / previous-question turns** — see section 12.
- [x] **Mic gain reset** — the agent no longer forces 100%; it only reports.
- [ ] **Live mic test on the Pi** — confirm capture card, speaker out, and
      barge-in feel. (Can't be done in this sandbox — no real audio device.)
      No `export` needed: the key is in the OS keychain, so just
      `.venv/bin/python agent.py`.
- [ ] Numbers in the streaming partials (`"2 plus 2"` → `"TOO PLUS"`). The final
      transcript is correct because Whisper produces it, but the live draft can
      show the wrong digits mid-sentence.
- [ ] Add **music** (YouTube download + playback) as an OnIt tool.
- [ ] Tune VAD thresholds (`confidence`, `stop_secs`) and TTS volume for the room.
- [ ] Decide on a wake word / "always listening" policy.
- [ ] (Optional) Voice the approval prompt: have Fweddie *say* "I need approval
      to run X" and take the answer by ASR, instead of the dashboard button /
      console. The Guard's `prompt` is a pluggable callback, so it's a drop-in.
