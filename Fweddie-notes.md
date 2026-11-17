# Fweddie — Voice Command Agent

> **Started:** 2026-11-07 · **Machine:** Raspberry Pi 5, 8 GB, aarch64, Debian 13, no GPU
> **Repo:** `Fweddie-repo/` · **Venv:** `Fweddie-repo/.venv` (Python 3.12.16, torch-free)
> **Status:** **Working** (2026-11-17). The user confirmed on real hardware:
> "this is working nicely." Streaming partials appear while speaking, the turn
> closes ~0.9 s after speech end, and the answer is about the question just
> asked. Pushed to GitHub at `f4836c85` and verified from a fresh clone.

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
mkdir -p models && cd models
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

**Web search** (`search` tool, 'net' profile). OnIt's search chain is Tavily
(only when `TAVILY_API_KEY` is set) -> DuckDuckGo (`ddgs`, free, no key). There
is no Ollama tier: `ollama_api_key` is deliberately not forwarded to the tool
servers (`config.FORBIDDEN_TOOL_SECRETS`), because Ollama's web search is
metered per call and the chain never used it. DuckDuckGo intermittently returns
"No results found" under load, so `_search_ddgs` retries up to 3 times with a
1s pause. Verified live over MCP: `test_search_live.py` spawns the real 'net'
server and checks `search` returns results for both `type="web"` and
`type="news"`.

**Result size cap** (`mcp_bridge._cap_for_model`). The model sees at most
`MCP_MODEL_RESULT_CHARS` (24000; it was 1200 until 17 Nov 2026) of any tool
result; the full text still goes to the dashboard and the session folder. A web
search returns full page bodies, so five results are 5400-7800 chars and *no
single result* fits a 1200-char cap. Until the fix, the cap dropped all five and
the model received
`{"results": [], "note": "all 5 results were too large to show"}` — an empty
answer to a question the server had answered, which is why the agent re-asked
"CES 2025 dates" four times. The cap now clips the prose inside each result
(keeping title, url, date and source intact) and clips *around the passage that
matches the query*, not blindly from the front. Measured on 12 real queries:
0 empty payloads, and the answer text present in what the model sees 12/12
(previously 4 of 5 web searches came back empty).

**The cap only covered half the tools** (`_cap_json_obj`). The structural cap
above handled a JSON *list*, which is what `search` returns, and nothing else.
Every other tool returns a JSON *dict* -- `local_search`, `grep`, `read_file`,
`search_document`, `get_weather` -- and a dict fell through to the blind
`text[:limit]` slice, which cut the JSON mid-string. Measured on the four
shapes the model can actually receive:

| tool             | payload  | what the model got        |
|------------------|----------|---------------------------|
| `local_search`   |  1718 ch | 1240 ch, invalid JSON     |
| `read_file`      | 12094 ch | 1241 ch, invalid JSON     |
| `grep`           |  3990 ch | 1240 ch, invalid JSON     |
| `search_document`|  2828 ch | 1240 ch, invalid JSON     |

Unparseable output is worth nothing, so a dict result over the cap was as
useless as the empty `{"results": []}` this cap was written to remove. It went
unnoticed because the event log holds only `search`, `get_weather` and `bash`
calls: 35 of 36 searches exceeded the cap, but every `bash` result was 140-172
chars, and the tools that break simply had not been used yet. A dict is now
capped structurally too -- its biggest list field loses whole records from the
tail, prose is clipped inside the rest, identity fields and the server's own
totals stay intact, and a `note` says what was trimmed.

The list path's own last resort had the same flaw in a narrower form: when one
result's title or url alone exceeded the budget it answered
`{"results": [], "showing": 0}`. Reachable only with a 1200+ char url, but it
is the same empty answer, so it now clips the identity fields and keeps the
result recognisable instead.

**What the budget buys, measured** (so the number can be argued with).
At the old 1200, a 5-result web page (full page bodies, ~1400 chars each) kept
**2 of 5** sources, and a real 5-result *news* payload kept **2-3 of 5**. Every
shape the tools return was over 1200:

| payload                          | chars  | at 1200    | at 24000 |
|----------------------------------|--------|------------|----------|
| ddgs news, sorted to 5           |  2,073 | 2-3 of 5   | whole    |
| Tavily news, 5 results           |  6,694 | 2 of 5     | whole    |
| web search, 5 full page bodies   |  7,432 | 2 of 5     | whole    |
| `read_file`                      | 12,094 | clipped    | whole    |
| ollama web, 5 results            | 22,970 | 2 of 5     | whole    |

So the default was raised to **24000** on 17 Nov 2026 — the smallest round
number at which *no observed payload* is cut at all. The headroom is measured,
not assumed: the endpoint accepts 1,000,000 chars (172,443 prompt tokens) and
refuses at 2,000,000 ("exceeded max context length by 82,713 tokens"), a ceiling
near 1.5M chars. The heaviest session in the event log was 13 tool calls /
60,583 raw chars; at 24000 each that is 312,000, about a fifth of the ceiling.
Latency is not a factor either — 120k chars answered in 2.5s, 600k in 4.6s.

**Why it stays a hard cap and not a prompt suggestion.** A prompt can ask the
model to be brief, but the model does not choose the size of a tool result — the
server does, and one 23k-char web page arrives whole. Past the ceiling the
endpoint returns HTTP 400, which reaches the pipeline as an `ErrorFrame` that
`agent.py` only logs, so the turn goes silent rather than short. The prompt
therefore *describes* the cap (the truncation paragraph in
`TOOLS_SYSTEM_PROMPT`) while `MCP_MODEL_RESULT_CHARS` enforces it. The one thing
the model can act on is asking for fewer results or a narrower query, and that
is what the paragraph says.

**Plain text still loses its tail.** `bash` output is head-sliced, so a verdict
on the last line (a test summary, `git status`) is unreadable through the cap.
Left alone because it does not bite this workload: every `bash` result in the
62-call log was 140-172 chars. `_cap_head_slice` is the one function to change
if a long command ever needs its tail.

**News freshness** (`mcp_bridge._sort_news`, `_refresh_stale_news`). On 17 Nov
2026 the agent answered "the latest news is November 12" for the Alaska Senate
race while Google showed stories from that morning. Nothing was broken in the
transport: the model was handed a correct, dated, well-formed payload whose
first two entries were days old. The cause is that `ddgs.news` orders by
**relevance, not recency**, and the model cap keeps a *prefix* — so the
prefix decided the answer. The same five-result page measured 17, 12, 17, 14,
11 Nov, with the two same-day stories sitting third and fifth.

Three fixes, all on Fweddie's side (OnIt's `search` has no `timelimit` and we
do not edit it):

1. **Over-fetch, then trim.** `_shape_search_args` asks for
   `MCP_NEWS_OVERFETCH` (10, ddgs' practical ceiling) and records the caller's
   real number under a private `_news_wanted` key, which `call_tool` strips
   before the arguments reach the server.
2. **Sort newest-first, before the cap.** `_sort_news` orders by date and adds
   an `age` field ("today", "yesterday", "3 days ago") next to `date`. The cap
   can only keep a prefix, so the prefix has to be the newest.
3. **Re-ask when stale.** ddgs returns a *different* page for the same query on
   consecutive calls, so a stale page often means "this draw missed it".
   `_refresh_stale_news` re-runs the search up to `MCP_NEWS_REFRESH_RETRIES`
   (2) times when the newest item is not from today, keeps the newest payload
   (never the last), and gives up at `MCP_NEWS_REFRESH_BUDGET` (4s) — bounded
   with `wait_for`, because a single ddgs call was measured at 10.5s.

Measured on 10 collected payloads, so ddgs' variance is held constant: mean
newest-result age 1.40 → 0.80 days, median 2 → 1 days, 5 rows fresher,
**0 rows worse**. With the refresh enabled, "the model sees a same-day story"
went from 3/6 to 5/6 over 6 rounds, at a mean 0.8s → 2.2s, hidden behind the
spoken announcement. `test_news_v2.py` covers all of it offline (11 groups).

### 11b. The news backend, and the cancelled searches

**News never used Tavily.** OnIt's `search(type="news")` branch calls ddgs and
nothing else — it does not reach the Tavily tier its own *web* branch uses. ddgs'
news engine is Bing-backed (`ddgs/engines/duckduckgo_news.py`, `provider =
"bing"`). Same four queries, newest published date, 17 Nov 2026: ddgs returned
15/17/14/15 Nov where Tavily's news topic returned 17 Nov on all four. So news
now goes to Tavily first (`MCP_NEWS_BACKEND=tavily`, `topic=news`, `days=1`,
`advanced`), falling back to OnIt's ddgs branch on failure or an empty page.
`MCP_NEWS_BACKEND=onit` restores the old path. Both return the same payload
shape, so the sort, the `age` labels and the cap are untouched.

**Tavily's dates needed parsing.** Tavily stamps `"Tue, 17 Nov 2026 03:44:14
GMT"`, which `_parse_news_date` did not know. Every Tavily row parsed as `None`,
so the newest-first sort had nothing to sort on and the `age` labels came out
empty — silently undoing all of the above. Measured before the fix: 10/10 rows
unparseable. `test_news_tavily.py` (8 groups) now pins the date shapes, the sort,
the cap, the backend switch and the echo guard.

**Why searches were being cancelled.** 7 of 62 tool calls in
`/tmp/fweddie_events.jsonl` were cancelled before returning, all 7 of them a
`search`, 0.59–4.22s after starting. Five were triggered by the agent's *own*
spoken announcement: `_ANNOUNCEMENTS["search"]` is `"Searching for that."`, sent
as a `TTSSpeakFrame`; the mic heard it, the transcript read `"Searching for
that."` or `"Searching."`, the VAD raised a turn start, and the resulting
`InterruptionFrame` cancelled the in-flight call. The agent then told the user
"it looks like that search was cancelled" (idx 2097, 2123).

The pipeline has **no echo cancellation at all** — the v4 AEC (`aec.py`,
`audio_io.py`, `ECHO_REF_DELAY_MS`, the mic duck) was left behind in the Pipecat
rewrite, and the repo's history starts after it. The fix here is
`FunctionCallUserMuteStrategy` as `user_mute_strategies` on
`LLMUserAggregatorParams`: it suppresses user frames (VAD frames included) for
exactly the window a function call is in progress, which is where the
announcement lives, and lifts on `FunctionCallResultFrame` so barge-in on real
answers is unchanged.

Pipecat already settles a cancelled call itself — `_cancel_function_call_tasks`
marks the runner `settled` *before* cancelling and the aggregator records
`"CANCELLED"` — so a handler that reports its own result on the way out is
rejected with "Ignoring result … already settled". The handler only logs.

**Self-echo during ordinary answers: closed in hardware, not in code.** A
`bot_started` followed by a `user_started` inside the bot's own utterance
happened 26% of the time after real answers (vs 15% after announcements), so the
echo was there too — it just cost a false barge-in rather than a cancelled tool.
The v4 defence would have been restoring the mic duck while the bot speaks, or
`pywebrtc_audio.EchoCanceller` (installed but unused).

It was not needed. Moving the speaker away from the mic and lowering both the
output volume and the mic gain removed the acoustic coupling at the source, and
the agent now runs without false barge-ins. No AEC code, no mic duck, no extra
latency. This is the intended state: **do not add `pywebrtc_audio.EchoCanceller`
or a mic duck for this** — the coupling is gone at the physical layer, and either
change would spend CPU and complicate barge-in for a problem that no longer
exists. If the echo ever returns, fix the placement and gain first; treat code
AEC as the fallback, not the default.

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
  client and Piper), `FWEDDIE_TURN_STOP` (1.0 s speech-timeout).
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
empty-turn path. **Fix:** reverted to 5.0 s, with the
`SpeechTimeoutUserTurnStopStrategy` (now 1.0 s) doing the real turn closing.

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
- [x] **Live mic test on the Pi** — **confirmed working** (2026-11-17). The
      user reported: "this is working nicely." Words appear while speaking and
      the agent answers the question just asked. No `export` needed: the key is
      in the OS keychain, so just `.venv/bin/python agent.py`.
- [ ] **Multi-turn from the mic** — one turn is confirmed; two in a row, checking
      that the prompt carries `assistant` messages and ends on the new question,
      has not been watched live.
- [ ] `test_pipeline_asr.py` latency budget — measured 2.77 s against a 2.5 s
      assertion (`TURN_SUBMIT_BUDGET`, line 65). Transcript and prompt were both
      correct, so this is a threshold that got tighter than the Pi holds under
      load. Decide whether 2.5 s or 3.0 s is the standard; do not tune it to
      whatever the machine happens to do.
- [ ] Numbers in the streaming partials (`"2 plus 2"` → `"TOO PLUS"`). The final
      transcript is correct because Whisper produces it, but the live draft can
      show the wrong digits mid-sentence.
- [ ] Add **music** (YouTube download + playback) as an OnIt tool.
- [ ] Tune VAD thresholds (`confidence`, `stop_secs`) and TTS volume for the room.
- [ ] Decide on a wake word / "always listening" policy.
- [ ] (Optional) Voice the approval prompt: have Fweddie *say* "I need approval
      to run X" and take the answer by ASR, instead of the dashboard button /
      console. The Guard's `prompt` is a pluggable callback, so it's a drop-in.

---

## The pause that split a sentence: one VAD analyzer, two consumers

**Symptom.** At 19:41:05 the user said "Senate elections." paused, then said
"as in senators. What is the latest news on the US Senate elections?". The
agent answered only "Senate elections." and asked for a repeat; "as in
senators" then arrived attached to the *next* question. Same session, an hour
earlier, the same shape: "What is the latest news under u s senate elec" was
closed and "Specifically alaska and dex" opened a new turn.

**The gate was never what the config said.** `agent.py` built ONE
`SileroVADAnalyzer` and passed that same object to both `VADProcessor` and
`LLMUserAggregatorParams`. The aggregator builds its own internal
`VADController` from it, so two controllers called `analyze_audio` on every
audio frame. A `VADAnalyzer` is stateful: `_run_analyzer` appends to
`_vad_buffer` and advances `_vad_stopping_count`. Fed twice per frame, each
32 ms window is counted twice and every gate runs at half its length.

Measured directly (`test_vad_shared_state.py`):

    fed 1x per window -> start 0.180 s, stop 0.710 s   (matches 0.25 / 0.70)
    fed 2x per window -> start 0.060 s, stop 0.330 s   (exactly half)

So the configured 0.7 s VAD stop behaved as **0.35 s**, and `VAD_START_SECS`
0.25 behaved as **0.13 s**. The total silence gate was ~0.95 s, not 1.30 s.

This explains two older entries in this file that were written as if they were
tuning problems: the keyboard clicks that kept starting turns (the 0.25 s
debounce was really 0.13 s, so a click outlasted it), and the need to raise
`VAD_CONFIDENCE` to 0.6. Both were symptoms of the halved debounce.

**Why no test caught it.** `test_pipeline_asr.py` had the identical bug -- it
built one analyzer and passed it to both places -- so it measured the halved
gate and agreed with the live agent.

**Fix.** Each consumer gets its own analyzer (`_make_vad_analyzer()` called
twice). Same fix in the harness. `test_vad_shared_state.py` pins it by
measuring the gate at 1x and 2x feeds, so re-introducing sharing fails the test.

**And the actual ask: raise the pause tolerance a little.** With the gate
honest again the two knobs are independent and additive:

    silence gate = VAD_STOP_SECS + TURN_STOP_SECS

`TURN_STOP_SECS` went 0.6 -> 1.0. The cost is small because the turn cannot end
before a transcript arrives and tiny.en lands ~0.91 s after the VAD stop, so
everything up to ~0.9 s of the timer is free. Measured on a 0.95 s utterance,
twice each, separate analyzers:

    TURN_STOP   VAD stop -> turn end   silence gate
      0.6 s          0.94 s              1.30 s
      1.0 s          1.00 s              1.70 s
      1.2 s          1.20 s              1.90 s
      1.6 s          1.60 s              2.30 s

So 0.6 -> 1.0 costs 0.06 s and buys 0.4 s of pause tolerance. Past 1.0 the
timer becomes the binding constraint and each extra 0.2 s is paid in full,
which is why it stopped at 1.0.

End-to-end on the incident's own two halves ("Senate elections." / pause /
"As in senators.") through the real pipeline:

    pause 0.5 s -> 1 turn  "Senate elections. As in senators."
    pause 0.8 s -> 1 turn  "Senate elections. As in senators."
    pause 1.0 s -> 1 turn  "Senate elections. As in senators."
    pause 1.3 s -> 2 turns (still split, as it must be)

Before the fix it split at every pause tested, including 0.5 s.

---

# Dashboard: Ollama-style chat, with voice and typing

Branch `dashboard`. The old dashboard was a sidebar toggle plus a transcript
read from the event log. It is now a chat window modelled on Ollama's, dark,
with the microphone where Ollama puts web search and a Send button that becomes
Stop mid-turn.

## What the agent had to gain

Two controls did not exist: typing a question, and stopping the bot. Both are
files next to the event log, the mechanism the mic toggle already used, and both
are turned into frames by one new stage, `DashboardControlWatcher`.

    Send -> <log>.typed.json -> LLMMessagesAppendFrame(run_llm=True)
    Stop -> <log>.stop.json  -> broadcast_interruption()

`LLMMessagesAppendFrame` appends to the SAME LLMContext the spoken turns use, so
a typed question is indistinguishable from a spoken one afterwards: one
conversation, one TTS. An `InterruptionFrame` is exactly what a barge-in
broadcasts, so Stop cannot drift from real barge-in behaviour -- it stops the
LLM mid-generation and drops the TTS's interruptible audio contexts.

## The bug worth recording

The watcher was first placed AFTER `LLMUserAggregator`, on the reasoning that
the aggregator is what handles the frame. It is -- but `queue_frame()` pushes
into the *calling stage's* input queue, and the frame then flows DOWNSTREAM from
there. A watcher downstream of the aggregator therefore hands its frame to the
LLM and the TTS, and the aggregator never sees it. Send did nothing at all, and
nothing logged an error.

`test_dashboard_controls.py` pins both halves: that a frame queued by the
watcher reaches an aggregator placed after it, and that `build_pipeline` really
does put the watcher first (`agent.pipeline_stage_names()`). Reordering the test
pipeline to the broken order makes check 1 fail with an empty context, which is
how the test was validated.

## Streamlit details that cost time

* `st.button` cannot go inside `st.form` (raises). The composer is therefore a
  container of columns, not a form. Buttons already trigger a full rerun, so no
  `st.rerun(scope="fragment")` is needed -- and that scope raises outside a
  fragment anyway.
* A keyed `text_input` cannot be cleared in the run that rendered it. Send
  raises a `clear_box` flag and the next run clears it before the widget is
  built.
* `st.chat_message` rejects SVG avatars (`image_to_url` uses `channels="RGB"`),
  so the husky head ships as PNG, drawn with PIL. `cairosvg` is not installed.
* Keyed widgets get a `.st-key-<key>` class, which is how the mic and the Send
  button are styled without depending on widget order. There is no supported
  way to add a class to a Streamlit container, so the blue "listening" state is
  a rule emitted only while the mic is on, not a toggled class.
* Specificity: a rule `.st-key-composer [data-testid="stBaseButton-secondary"]`
  (two classes) outranks `.st-key-sendbtn button` (one class + element). The
  first stylesheet set Send's background and watched it lose. The per-button
  rules now repeat the attribute selector so they tie on specificity and win on
  order. Verified by reading computed styles over CDP, not by eye.
* `chromium --screenshot` captures Streamlit's blank white shell because the
  page fills in over a websocket after load. `tools/shot.py` drives Chromium
  over the DevTools protocol and waits for the app's own markup.

## Deliberately left alone

`test_pipeline_asr.py` still fails its two clips with internal 0.65-0.70 s
pauses (3.20 s and 6.01 s against a 2.5 s budget). That is pre-existing on
`main` (3.22 s / 6.11 s there, measured by stashing this branch's changes) and
is the known consequence of holding those pauses inside one utterance instead
of splitting it. Not touched here.

---

# Dashboard v2: Material mic, a real light/dark switch, Clear history

Same branch. Three changes, one of which turned out to be a Streamlit limitation
rather than a styling choice.

## The microphone is now a Material icon, not an emoji

The 🎤 emoji is drawn by the font in its own colours, so no CSS colour can reach
it -- the blue "listening" state the previous version wanted was impossible with
it. `st.button(icon=":material/mic:")` renders the glyph as text in the
"Material Symbols Rounded" font, and that span inherits the button's `color`.

Measured over CDP, with the rule emitted only while the mic is on:

    dark  live  : button rgb(59,158,255)   icon rgb(59,158,255)   glyph "mic"
    dark  muted : button rgb(236,234,228)  icon rgb(236,234,228)  glyph "mic"
    light live  : button rgb(11,111,212)   icon rgb(11,111,212)   glyph "mic"
    light muted : button rgb(31,30,29)     icon rgb(31,30,29)     glyph "mic"

The glyph is the same shape as the Flaticon microphone asked for (capsule head,
stem, base). The button also gained a 14% blue tint background while live, via
`color-mix(in srgb, <that blue> 14%, transparent)`.

The blue is per-theme, not one shared value. Measured contrast against the page:
#3b9eff is 5.4:1 on the dark page but only 2.7:1 on the light one, under the 3:1
floor for a graphical control, so light mode uses #0b6fd4 (4.7:1 on light).

## The theme switch: Streamlit's own theme cannot be changed at runtime

This is the part worth recording. `st._config.set_option("theme.base", "light")`
from a running script does NOT repaint. Measured on Streamlit 1.67:

    before click : rgb(10,10,10)
    after click  : rgb(10,10,10)     (server said "light")
    after RELOAD : rgb(10,10,10)     (server still said "light")

and `st.context.theme.type` kept reporting the OLD value throughout -- its own
docstring warns it may be wrong around a theme change (streamlit issue #11920).
Streamlit also caches the active theme in localStorage under
`stActiveTheme-<pathname>-v<n>`, which wins over the server on the next load.

So the theme is CSS variables, and the palette is chosen SERVER-side before the
stylesheet is emitted: the active palette goes on `:root`, so the first paint is
already right and there is no flash. A `data-theme` attribute is also set on
`<html>`, from a 1 px `st.iframe`, matching what the OnIt UI does. Measured after
a toggle, both palettes and the attribute:

    dark  : app bg rgb(38,38,36)  text rgb(236,234,228)  attr "dark"
    light : app bg rgb(250,249,245) text rgb(31,30,29)   attr "light"

Palette values are copied from ~/onit/src/ui/static/style.css (warm paper light,
warm charcoal dark, terracotta accent), which is why the two interfaces read as
the same family.

Two dead ends, both measured rather than assumed:

* `st.html()` sanitises `<script>` away -- the attribute stayed `null`.
* `st.iframe(..., height=0)` raises: height must be a POSITIVE integer. The
  frame is 1 px and its keyed container is `display:none`.
* `st.components.v1.html` works but is deprecated ("will be removed after
  2026-06-01"); its docstring names `st.iframe` as the replacement, and same
  origin HTML there does run JavaScript.

The choice lives in `?theme=`, not only in session_state, so it survives a reload
AND the server can see it -- the toggle's glyph is right on the first render.

## Clear history is two halves

    Clear -> <log>.clear.json -> broadcast_interruption()
                              -> LLMMessagesUpdateFrame(messages=[], run_llm=False)
                              -> "cleared" event

`run_llm=False` matters: clearing is not a question and must not be answered.
The interruption first matters too -- clearing the context of a turn still
generating would let the in-flight reply finish and be written back into the
freshly emptied context, so the conversation would reappear one turn at a time.

The page empties immediately (a session marker) rather than waiting for the
agent's `cleared` event, which can take a poll interval. The event is what makes
the clear survive a reload -- without it, a refresh would bring the whole
conversation back. Both halves are pinned by tests.

Measured end to end in the browser: 4 transcript rows -> 0 rows on click, and
`events.jsonl.clear.json` written.
