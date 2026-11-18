# WanDog — Voice Command Agent

> **Started:** 2026-11-07 · **Machine:** Raspberry Pi 5, 8 GB, aarch64, Debian 13, no GPU
> **Repo:** `WanDog-repo/` · **Venv:** `WanDog-repo/.venv` (Python 3.12.16, torch-free)
> **Status:** **Working** (2026-11-18). The user confirmed on real hardware on
> 2026-11-17: "this is working nicely." Streaming partials appear while speaking,
> the turn closes ~0.9 s after speech end, and the answer is about the question
> just asked.
> **HEAD:** `main` = `df786c3b` · `files` = `9800d94a` (**not yet merged**), 62 commits.

This file is the **development log**: what was built, in what order, and every
fault worth not repeating. The two registers that make it navigable are §21
(features, each with the commit that added it) and §22 (issues, each with the
commit that fixed it).

### Contents

| § | Topic | § | Topic |
|---|---|---|---|
| 1 | What this is | 12 | Turn handling & the silent-agent regression |
| 2 | Components | 13 | Barge-in |
| 3 | Features (core, v1) | 14 | Open questions / next steps |
| 4 | Issues & challenges faced | 15 | TTS voice Piper Ryan, owner identity |
| 5 | How to run | 16 | Wordless turns, mute, Ctrl+Enter |
| 6 | Dependency notes (torch-free) | 17 | Time, GitHub, long tasks |
| 7 | Safeguards (adapted from OnIt) | 18 | A repo the agent could create but not delete |
| 8 | ASR (streaming) | 19 | "I have no tool for that" was false |
| 9 | Dashboard (Streamlit, LAN) | 20 | `write_file` ran out of tokens |
| 10 | Audio devices (pinned) | **21** | **Feature register, by commit** |
| 11 | LLM: the endpoint | **22** | **Issue register, by commit** |

---

## 1. What this is

WanDog is a **local, full-duplex voice command agent** — the "heart" of the
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
| TTS | **Piper** `en_US-ryan-medium` via `PiperTTSService` | Local, low-latency; Ryan voice downloaded to `/home/raesoriano/.piper` (61 MB ONNX + its `.onnx.json`), see §15 |
| Audio I/O | `LocalAudioInput/OutputTransport` (PyAudio/ALSA) | Mic + speaker on the Pi |
| Persona | "WanDog" — short spoken replies, no markdown | Set in `config.py:SYSTEM_PROMPT`, with the owner block in `OWNER_PROMPT` |

### Files
- `agent.py` — the pipeline: transport → VAD → LLM → TTS → output + a small
  `WanDogLogger` observer that logs user/bot speech and LLM text.
- `config.py` — every knob, overridable via env vars (`WANDOG_*`, `ONIT_LLM_*`).
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
cd /home/raesoriano/sandbox/WanDog-repo

# One-time: the streaming ASR model (321 MB, gitignored)
mkdir -p models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
cd ..

export ONIT_LLM_API_KEY=sk-...        # or rely on keychain / OPENAI_API_KEY
.venv/bin/python agent.py
```
Stop with **Ctrl+C**. Skip the download with `WANDOG_ASR_BACKEND=whisper`
(segmented, slower, no 321 MB).

## 6. Dependency notes (torch-free)
- **No torch.** Silero VAD runs through `onnxruntime` (CPU). The `silero`
  pipecat extra is intentionally omitted (it pulls the torch-requiring
  `silero-vad` wrapper).
- `webrtcvad` kept as a lightweight fallback (zero deps); `setuptools<81` for
  its `pkg_resources` build.
- Venv: 154 packages, 1.4 GB, verified `import torch` → `ModuleNotFoundError`.

## 7. Safeguards (adapted from OnIt)

WanDog does **not** re-implement OnIt's safeguards. It spawns OnIt's own stdio
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

Three fixes, all on WanDog's side (OnIt's `search` has no `timelimit` and we
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
`/tmp/wandog_events.jsonl` were cancelled before returning, all 7 of them a
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
  `WANDOG_ASR_BACKEND=whisper` restores the old segmented path with no download.
- Knobs: `WANDOG_ASR_BACKEND`, `WANDOG_ASR_FINAL`, `WANDOG_ASR_FINAL_MODEL`,
  `WANDOG_ASR_PREROLL` (1.0 s kept from before the VAD fires, so the decoder does
  not start mid-word), `WANDOG_ASR_THREADS` (2 -- the Pi also runs VAD, the LLM
  client and Piper), `WANDOG_TURN_STOP` (1.0 s speech-timeout).
- **Rejected:** the 20M zipformer returned empty strings on this mic's audio. The
  2023-06-26 model works on every capture. `WhisperSTTService`'s segmented
  default remains available for A/B.

**The regression this introduced and its fix.** The tap that publishes the live
draft line overrode `FrameProcessor.process_frame` without calling `super()`,
which skipped `StartFrame` handling and stalled the pipeline behind it -- the
transcript decoded but never reached `LLMUserAggregator`. See section 12(b).

## 9. Dashboard (Streamlit, LAN-reachable)

`dashboard.py` is a separate Streamlit process that shows the live interaction
from any device on the LAN: a running transcript (your words + WanDog's
replies), the OnIt tools the agent calls and their results, and a live
**approval panel** where you approve/deny a pending command with a button.

The two processes coordinate through **plain files** (no version-specific
Streamlit/Flask API, so it is robust):

| File | Writer | Reader | Purpose |
|---|---|---|---|
| `/tmp/wandog_events.jsonl` | agent | dashboard | the interaction record (one JSON per line) |
| `/tmp/wandog_events.jsonl.pending.json` | agent | dashboard | a command awaiting approval |
| `/tmp/wandog_events.jsonl.decision.json` | dashboard | agent | the person's decision (once/session/deny) |

- The agent appends events via `eventbus.EventBus` (best-effort; a full disk
  never crashes the voice loop). Events: `session_start/end`, `user_text`,
  `bot_text`, `tool_call`, `tool_result`, `approval_requested/granted/denied`.
- **Live approval:** when a command needs approval, the Guard writes the
  pending file and polls for the decision file. The dashboard shows the command
  and, on button click, writes the decision. If the dashboard is never up, the
  Guard falls back to the **console** prompt, so approvals never block on the
  browser. (Fail-closed: no decision by the timeout = deny.)
- Turn it on with `WANDOG_DASHBOARD=1` (off by default). The agent writes the
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
`WANDOG_IN_DEVICE` / `WANDOG_OUT_DEVICE` (an empty value = system default).

## 11. LLM: the endpoint

> **Corrected 2026-11-18.** This section used to say the LLM was the on-prem DGX
> cluster (`Qwen/Qwen3.8-27B` at `http://202.92.159.240:8001/v1`). That is no
> longer what runs. `config.py` defaults to **`gemma4:31b-cloud`** at
> **`https://ollama.com/v1`**. The Qwen/DGX path still works when overridden
> with `ONIT_LLM_BASE_URL` / `ONIT_LLM_MODEL`, and the keychain entry for it is
> still present — but the default is Ollama cloud.

- Current default: `LLM_BASE_URL = https://ollama.com/v1`,
  `LLM_MODEL = gemma4:31b-cloud`.
- The URL matters: `https://ollama.com/v1` is canonical and answers 200;
  `https://api.ollama.com/v1` 301-redirects to it, and `https://api.ollama.com`
  301s to a 404. That last form was an original failure worth not repeating.
- **Thinking disabled by default.** The Qwen path needed this: it is a thinking
  model and with thinking on vLLM returns empty `content` (the reasoning goes to
  a separate field), which makes the voice agent silent. `agent.py` adds
  `extra_body={"chat_template_kwargs": {"enable_thinking": false}}`;
  `ONIT_LLM_THINKING=1` restores the model default. Harmless on the Ollama path.
- **API key** resolves through `config.resolve_llm_api_key()`: env →
  OS keychain (service `onit`, entry `endpoint_key:<base_url>`) →
  `~/.onit/secrets.yaml` → `OPENAI_API_KEY`. A key stored once via `onit setup`
  is picked up with no manual export.
- Verified end-to-end: `test_llm_live.py` builds pipecat's `OpenAILLMService`
  exactly as `agent.py` does and gets a real reply.

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
`0.6` (the Pipecat default), overridable via `WANDOG_VAD_MIN_VOLUME`.

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
- [ ] (Optional) Voice the approval prompt: have WanDog *say* "I need approval
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
  so the transcript uses plain markdown rows with an `<img>` of our own.
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

# Dashboard v3: a sticky bar, a measured transcript, and a husky drawn here

Same branch. Four changes, three of which were wrong in a way that looked right
until it was measured.

## The scrollbar never disappeared -- it was painted over

The complaint was that the scrollbar vanishes at the end of the conversation.
It does not vanish; it fades. Streamlit's scroller is `[data-testid="stMain"]`
(measured: `overflow-y:auto`, `clientHeight` 657, `scrollHeight` 936) and the
composer is pinned over the bottom of it with a gradient. A screenshot of the
last 6 px column at the bottom of the scroll:

    y=575  rgb(115,115,111)     <- thumb, opaque
    y=590  rgb( 76, 76, 73)
    y=606  rgb( 38, 38, 36)     <- page background, thumb gone

So the thumb's travel ran under the composer's `linear-gradient(to top, bg 62%,
transparent)`. The fix is structural: the transcript gets its own scroll box
that ENDS above the composer, so its thumb is never under the gradient. The page
itself no longer scrolls at all (`documentElement.scrollHeight == clientHeight`,
one scroll box instead of two).

## The transcript's height is measured, not guessed

Four attempts at `calc(100vh - Nrem)` were all wrong, for reasons that are not
visible in the source: the bar's height depends on whether the status line is
present, Streamlit's block padding is in `em`, and the container has
`flex-grow:1`, so `height` alone is a basis the box grows past (measured: height
`calc(-11.1rem + 100vh)` = 479 px, used height 631 px, until `max-height` was set
too). The measured failures:

    11.1rem -> transcript bottom 718, composer at 575   (143 px hidden)
    11.6rem -> bottom 639                              ( 64 px hidden)
    13.5rem -> bottom 607                              ( 32 px hidden)

Now a one-line script in the bar measures the bar's lower edge and the
composer's upper edge and publishes `--fw-scroll-h`. Final geometry, measured:
bar 0-90, transcript 102-563, composer 575-. The margin is computed as "distance
wanted MINUS the offset Streamlit already applies" (48 px here), read back from
the applied margin so it is idempotent -- setting it to the distance itself left
60 px of dead space under the bar.

## Opening at the newest message

Streamlit starts an overflow box at `scrollTop 0`, so a returning reader landed at
the oldest message. The rule is: follow the bottom only while the reader is
already at the bottom (40 px tolerance), and stop the moment they scroll up. The
script lives in the PARENT document, installed once behind a flag, driven by a
MutationObserver -- not inside the 1 px iframe, which re-runs on every fragment
rerun and cannot share state.

A trap worth recording: an idle page produces NO mutation. The fragment re-runs
every second, but with an unchanged log it renders identical markup, so even an
unconditional `follow()` never runs. The check that guards this therefore appends
a turn from inside the browser session. The first version of that check appended
before the browser started and passed against a deliberately broken script.

## The husky icons are three committed PNGs

`assets/` holds exactly three files, and they are the assets -- there is no
source drawing in the tree and no script that rebuilds them:

* `husky-idle.png` -- 192 px, mouth open, tongue out. The bar at rest, and the
  avatar beside every reply.
* `husky-think.png` -- 192 px, muzzle up. The bar while a turn is in flight.
* `husky-32.png` -- 32 px favicon, from the idle drawing.

An earlier revision also carried the original drawings and a script that
rescaled them on demand. Both are gone: the icons are used at 26-30 px and never
re-derived, so the drawings were only ever inputs to one command, and carrying
two copies of each face meant the tree could disagree with itself about which
one was current. Redrawing a face now means replacing the PNG in place.

### Why rasters, and why two-tone

Rasters because the drawings are raster, and tracing them to SVG produced blobs
the first time round (the eyes fused with the muzzle). Two-tone because the
icons sit on both themes: the light backing is what shows on the dark theme, the
darker drawing is what shows on the light one, and both have to be in the same
file. The thinking face's drawing is blue rather than black, so the test checks
a luminance range, not a fixed ink colour.

### What the test pins

`test_dashboard_layout.py` test 4: both icons exist and differ, roughly half the
square is drawn with a transparent margin on every side, both a light backing
and a darker drawing are present, and the drawn shape keeps the aspect it had
before it was scaled. The last one catches a replacement that forces a square
box: the two faces are neither square nor alike (the idle drawing is taller than
it is wide, the thinking one wider than it is tall), so squashing one is easy to
ship without noticing.

The expected aspects live in `ASPECT` at the top of that test -- 0.885 for idle,
1.043 for think -- because the source drawings are no longer in the tree to be
measured. That is the one number to re-measure if a face is ever redrawn.

The PNGs are inlined as data URIs: there is no static route to `assets/`, and a
data URI cannot 404 or go stale while an icon is being replaced.
`assets/husky-32.png` remains the favicon, which Streamlit requires to be raster.

## The bar's face follows the turn

The identity block is its own `@st.fragment(run_every=1.0)`, so the face can change
without re-running the theme switch, Clear, or the status line on a timer. Both
images are always in the DOM (`.fw-face-idle`, `.fw-face-think`) and CSS picks one
through `.fw-head.fw-busy`; swapping the `src` instead left a frame with no image
while the new data URI decoded, which at 30 px is a visible blink.

The avatar beside a reply is always the idle (tongue-out) face. That is a
different question from the bar's: the avatar is *who is speaking*, the bar is
*what it is doing*. A test pins both -- bot rows must carry the idle face and must
not carry the thinking one, which is easy to break because the bar legitimately
holds both.

## The scrollbar sits on the window edge

The transcript is full-bleed (`width: 100vw` plus a negative `50% - 50vw` margin)
and the reading column is constrained inside it. Before this the scroller
inherited Streamlit's centred 820 px column, so its thumb sat ~200 px inboard of
the window edge -- measured, scroller x=206..994 in a 1200 px window. That is not
where a window's scrollbar lives, and it is what "the scrollbar is in the wrong
place" was about.

Two details that cost time:

* `max-width: 100vw` is required, not decoration. Streamlit's own
  `.stVerticalBlock` carries `max-width: 100%`, which clamped 100vw straight back
  to the column (width 788 px, starting at x=0).
* the reading column is the scroller's child `[data-testid="stLayoutWrapper"]`,
  not the `stVerticalBlock` -- the latter is one level further in, and a selector
  for it matched nothing.

Measured after: scroller x=0..1200 in a 1200 px window, gutter 10 px, and the thumb
on hover at x=1190-1198. Streamlit paints the thumb `transparent` until `*:hover`,
which is why a still screenshot shows no bar at all.

## The bar

`.st-key-topbar` is `position:fixed`, not `sticky`: the page no longer scrolls, so
there is nothing for a sticky element to stick to. Streamlit's own header had to
go (`display:none`), because it is an overlay -- `position:absolute; top:0;
height:60px; z-index:999990` -- and `elementFromPoint(300, 20)` returned it, so it
would have swallowed clicks on anything placed at the top of the page.

Verified by clicking, not by reading: real mouse events at the buttons' own
coordinates flipped the palette `dark -> light` (`rgb(38,38,36)` ->
`rgb(250,249,245)`) and took the transcript from 4 rows to 0.

## Tests

`test_dashboard_layout.py`, 9 checks: 4 through AppTest, 5 against a live server
the test starts itself (on its own port, reading its own log -- driving whatever
server happens to be running meant appending to a log the server did not read,
which is how one check once passed by testing nothing).

Every check was confirmed able to fail, by sabotaging the code it protects:

    avatar on the thinking face  -> "a bot row is not using the idle face"
    icon never thinks            -> "the thinking face did not appear"
    scroller back in the column  -> "the scroller ends at x=788 in a 1100 px
                                     window, so its scrollbar is 312 px inboard"
    unconditional follow         -> "the view jumped to 2926 while the reader was
                                     scrolled up"
    bar in the page flow         -> "the bar is static, not fixed"
    no measured height           -> "the transcript ends at 3186, below the
                                     composer at 575"

Two of the new checks had to be fixed before they were worth anything. The avatar
check sliced the page blob on `"fw-row"` and kept the first piece mentioning
`fw-av-bot`, which also matches the stylesheet and the bar; it now matches the row
markup exactly. The icon check asserted "more than half the icon is drawn", which
was wrong once the icons were trimmed and padded to a square -- half is exactly
right.

`test_pipeline_asr.py` still fails its 2.5 s budget (5.83 s here). Pre-existing and
unrelated: stashing this change and re-running gives 6.06 s.

## 15. The TTS voice is Piper Ryan, and the agent knows who it is talking to

Both changes are on branch `TTS-Ryan-raesoriano-WanDog`.

### The voice

`config.PIPER_VOICE` is now `en_US-ryan-medium` (was `en_US-lessac-medium`).
Piper's voices ship as a pair -- the ONNX graph and a `.onnx.json` beside it
holding the phoneme map and the sample rate -- and `PiperTTSService` downloads
both into `PIPER_VOICE_DIR` on first use if they are absent.

| | |
|---|---|
| file | `en_US-ryan-medium.onnx` |
| size | 63,221,984 B (61 MB) -- under the 200 MB budget, so it was fetched |
| config | `en_US-ryan-medium.onnx.json`, 5,466 B |
| location | `/home/raesoriano/.piper/` |
| sample rate | 22,050 Hz, which the output transport resamples to 48 kHz as before |

Verified by loading it outside the pipeline: `PiperVoice.load()` returns a voice
reporting `config.sample_rate == 22050`, and synthesizing a sentence yields
115,200 bytes of int16 -- so the model is intact, not just downloaded.

Ryan also exists as `low` (63.1 MB) and `high` (120.8 MB). Medium was chosen to
match the quality tier of the voice it replaces; `high` is still under the budget
if the extra 58 MB is ever worth it.

### The owner block

`config.OWNER_PROMPT` sits between the persona and the clock:

    SYSTEM_PROMPT + " " + OWNER_PROMPT + " " + clock_line() + " " + CLOCK_GUARD

It states the name (Ron Andrei Soriano) and the GitHub handle (raesoriano), and
resolves "I / me / my" -- including "my repos" -- to him. Without it the model
answers "I don't know your name", which is true from its point of view: nothing
in the prompt ever said.

**One thing that had to change with it.** `ClockedLLMService._refresh_clock()`
rebuilds the prompt when it cannot find a clock sentence to replace, and it used
to splice at `len(config.SYSTEM_PROMPT)` -- the end of the persona. With the
owner block now sitting there, that splice would drop the clock *between* the
persona and the owner line, leaving "who you are speaking with" attached to the
wrong paragraph. The anchor is persona+owner, with the old persona-only path kept
as a fallback for a prompt built by something else. Confirmed by stripping the
clock from a real pipeline prompt and re-running the refresh: the owner line
survives, appears once, and the clock lands after it.

### Identity and the tools

`test_tool_secrets_live.py` reaches the spawned MCP subprocesses and reports the
keys they carry: `GITHUB_TOKEN`, `OPENWEATHERMAP_API_KEY`, `TAVILY_API_KEY`,
`HF_TOKEN`, on both servers, and a live `get_weather` call through the bridge
returns real data -- so the keys are not merely in the file, they arrive. The
GitHub token in `secrets.yaml` authenticates as **raesoriano** and holds admin on
the repo, which is what the `github_repo` tool and a `git push` from the `bash`
tool need.

### The rename

`Fweddie` became `WanDog` everywhere: 26 tracked files, `FweddieLogger` ->
`WanDogLogger`, `Fweddie-notes.md` -> `WanDog-Notes.md`, and the configuration
prefix `FWEDDIE_` -> `WANDOG_` (53 distinct knobs, 99 occurrences). The GitHub repo was
renamed `Full-Duplex-Voice-Agent-Fweddie` -> `Full-Duplex-Voice-Agent-WanDog`
(GitHub redirects the old URL, so an existing clone still fetches), and the
local checkout `Fweddie-repo/` -> `WanDog-repo/`.

The directory rename is the one part that is not just a `mv`: a venv records its
own absolute path in the shebang of every script under `.venv/bin` and in
`pyvenv.cfg`. Fifty scripts were rewritten to the new path, and
`.venv/bin/python -c "import agent"` was run afterwards to prove the interpreter
still resolves its packages. Leftover `.pyc` files inside `.venv` still carry the
old path in their embedded filenames; that is harmless, they are byte-code caches
and get rewritten when their source is next compiled.

`secrets.yaml` stays gitignored; only `secrets.example.yaml` is tracked.

## 16. A wordless turn now ends, a mute now mutes, and Ctrl+Enter interrupts

Reported: "when the message is just typed while muted, the agent does not answer
and the turn is not ending sometimes", plus "allow ctrl+enter to also interrupt
similarly to the stop button".

### The turn that did not end

`turn_in_flight()` reads the tail of the event log to decide whether the
composer shows Send or Stop. It cleared `busy` on `bot_stopped`, `interrupted`,
`cleared` and `session_end` -- but **not on `user_stopped`**.

A voice turn opens on `user_started` (the VAD fired) and is only answered if a
transcript follows. Muted, or a cough, or a door gives `user_started` ->
`user_stopped` with no `user_text`: the turn is over, nothing will be answered,
and nothing is running. The button stayed on Stop anyway, until the 90 s
staleness guard expired. That is the "turn is not ending".

Measured on the real log, replaying it through the function: four turns ended
with no transcript, and three of them left the button on **STOP** -- the tail of
the session (10:19:28) being one of them.

`user_stopped` now ends the turn, but only when no transcript arrived during it.
The distinction is load-bearing in both directions: with no words there is
nothing coming and nothing to stop, while with words the answer is still owed --
the log shows 1.4-1.8 s between `user_stopped` and `bot_started`, and Stop has
to stay available across that gap. All 19 real turns now read correctly: the 4
wordless ones say Send, the 15 answered ones keep Stop.

Two smaller things fell out of the same function:

- `user_started` and `user_partial` now refresh the staleness clock. `user_started`
  sets `busy`, and partials are proof the person is still talking, so a sentence
  longer than 90 s would otherwise read as idle mid-sentence.
- An `error` ends the turn. A failed inference pushes an ErrorFrame and no
  `bot_stopped`, so the button previously waited out the guard with nothing to
  stop -- "gave up, but still says Stop".

### A mute that did not mute

`MicControlWatcher._watch()` skipped the first read (`if self._last is not
None`), on the reasoning that the STT starts unmuted so "enabled" is already the
truth. That holds only for enabled. With the dashboard left muted, the agent
started, read `enabled=False`, recorded it, pushed nothing -- and never pushed
again, because the value never changed.

Measured: the agent was started against `{"enabled": false}` and the STT still
transcribed ("What in the dog?"). The mic was live while the UI said muted, so
speech the person believed was ignored still became a turn. The first read is
now applied; started muted sends `STTMuteFrame(mute=True)`, started live sends
`mute=False`.

### Errors were invisible

`WanDogLogger` did `logger.error()` on an ErrorFrame and nothing else, so a
failed turn looked exactly like a slow one: the question sat there, no reply
came, and nothing said why. Errors are now published on the bus and rendered in
the transcript. The agent cannot answer what it never heard, but it can stop
lying about why it went quiet.

### Ctrl+Enter interrupts

The JS backstop returned early when the Send button was absent -- which is
exactly when the button is Stop, i.e. mid-turn. So Ctrl+Enter was dead at the
one moment it is wanted, with the mouse needed for the button.

It now reads the control that is showing: Stop present -> interrupt, Send
present -> send. While busy the box is not sent, so an interrupt does not
swallow a half-written message. Verified in headless Chromium with a real key
event through the browser's input pipeline, both states.

### Testing notes

`test_muted_turn.py` (no browser, no model): 10 turn-state cases, the mute on
first read both ways, and the error event. `test_ctrl_enter_interrupt.py` (live,
headless Chromium over CDP): Ctrl+Enter interrupts when busy and sends when idle.

Two traps worth remembering. The repro scripts set `config.DASHBOARD_LOG` but
`eventbus` binds its output path at import, so they appended to the **live** log;
24 test events had to be trimmed back out. And `test_ctrl_enter_interrupt.py`
rewrites the log to force the busy state, so it backs the log up and restores it
-- an early version without the backup overwrote a real session. Both control
files are consumed by the agent within a poll, so the test asserts on the log,
not on the file: the click was working all along and the file was already gone.

## 17. Time, GitHub and long tasks: the tools were fine, the prompt was not

Reported as three tool failures. Only one of them is a tool failure, and it is
the one nobody reported: the clock line contradicted itself.

### Time

"How do you tell time as OnIt?" -- OnIt does not carry a clock in its prompt at
all. Its Context block gives only `Today's date: {current_date}`
(`src/mcp/prompts/prompts.py:117,159`), and it answers a time question by
running `date` through bash. `date` and `cal` are in OnIt's
`DEFAULT_ALLOWED_COMMANDS` (`os/bash/command_policy.py:496`), so that works.

WanDog took a different route on purpose: `config.clock_line()` puts the date
**and** the time in the prompt, because a date with no time made the model
invent a time (measured 9/12 before that line existed, 12/12 after). Keeping
that was right -- it is faster and it costs no tool call.

Measured on the live model with the real 14-tool set, 13 questions:

    9 plain questions   answered from the prompt clock, no tool call   correct
    4 timezone questions  bash TZ="..." date                          correct

So "it fails to use the tool for telling the time" does not reproduce as a tool
problem. What does reproduce is the clock going **stale**, which is the failure
the prompt makes possible and the guard does not catch:

    clock frozen at 06:30, real time 11:33
    "What time is it?"  ->  no call, "It is 6:30 AM."        wrong by 5 hours

The guard says "never state a clock time you have not been given". A stale
clock *was* given, so the guard is satisfied and the model answers confidently.
`ClockedLLMService` already re-reads the clock on every inference and is
idempotent (verified: the stale time is gone, exactly one clock remains, the
prompt does not grow). What was missing was a test that says so, and that is
`test_tool_guidance.py` group 4.

**The real bug was in the clock line itself.** It read:

    (PST, UTC+0800)

This host is on Asia/Manila. Its abbreviation is PST -- *Philippine* Standard
Time -- and PST is far more widely read as *Pacific* Standard Time, UTC-0800.
The line contradicted itself. The model happened to trust the offset and got it
right, but a person reading the transcript gets a 16-hour error. `_zone_name()`
now resolves the IANA name out of `/etc/localtime`, so the line reads:

    (Asia/Manila, PST, UTC+0800)

### GitHub

"Do you have my GitHub token?" -> "I do not have access to your GitHub token."
"Can you push to my GitHub?" -> "I can't push directly to GitHub yet."

Both false. The token was in `secrets.yaml` (93 chars, `github_pat_...`),
`config.tool_secret_env()` forwarded it, and `github_repo` with
`action=list` returned real repositories in 1.8 s. Nothing was broken except
the model's belief about itself: `TOOLS_SYSTEM_PROMPT` never mentioned GitHub,
so it inferred its capability from the tool description, which says "Requires
GITHUB_TOKEN" -- a caveat, not a statement of fact.

The worse failure is the other direction. "Can you see my GitHub repos?" ->
"I sure can!" with **no call**. It claimed access without finding out, which is
the same defect as claiming it had none.

The fix names GitHub in the prompt and makes the claim conditional on a call:

  * never claim it can or cannot reach GitHub without checking;
  * a read (`action list` / `get`) is its own confirmation;
  * a write -- create, fork, delete, push -- must state the account and the
    repository and wait for the person to confirm;
  * a missing scope is not "no access", and must be described as what it is.

Measured after: "Do you have access to my GitHub?" now calls `github_repo
action=list` and answers from the result, instead of asserting.

### Long tasks

`bash` is a foreground call and the voice loop waits on it -- a measured 60 s
`sleep` held the call for the full 60 s, and `MCP_TOOL_TIMEOUT` is 320 s.
`serve` is the detached path: it returned in 0.5 s with a pid and a log path,
and `action logs` read `tick-1..tick-4` six seconds later.

The prompt said nothing about which to use, so routing was luck:

    "run ./build.sh and tell me when it finishes"  -> bash("./build.sh")
    "the test suite takes about 15 minutes"        -> serve(...)   (only
                                                     because the question
                                                     said "minutes")

After the change all four phrasings route to `serve`, including the one that
never says how long it takes. The rule is stated rather than inferred from the
word "minutes": anything that may outlast a sentence goes to `serve`, and the
agent must not call it done until `serve` says it stopped.

### Tests

`test_clock.py` 8/8 (the zone assertion was tightened to require the IANA name
and to reject `PST` beside a negative offset). `test_tool_guidance.py` 5/5.
Also re-ran `test_tool_parity` (14 tools), `test_tool_secrets` and
`test_tool_secrets_live` (GITHUB_TOKEN reaches both spawned servers),
`test_eventbus`, `test_turn_context`, `test_turn_live`, and
`test_endtoend_answer`, which spoke a live reply with the new clock line.

## 18. A repository the agent could create but not delete

Reported: "the agent was able to create the repositories, see them, and list
them but is unable to delete the newly created ones."

Not a permissions problem and not a token problem. The token has `admin` on
every repo in the account (`permissions` in the `list` payload says so), and a
delete with the right name returns 204. The name is what was wrong.

### The shape of the argument

`github_repo` is one tool with two naming conventions, and its own description
states both -- in a parenthetical that the model reads as one blob:

    - name : Repository name — required for get/fork/delete (owner/repo)

`create` takes a **bare** name (`POST /user/repos` with `{"name": ...}`), and
`get`/`fork`/`delete` take **owner/repo**, because the server pastes the value
straight into the path: `DELETE /repos/{name}`. Send a bare name to that path
and GitHub answers **404**, which is the same status it returns for a
repository that really is gone. So the failure is indistinguishable from
success-adjacent state: the agent concluded the repos had already been deleted.

Reproduced on the real account, 18 Nov 2026, against a repo created seconds
earlier:

    create name="wandog-probe2"              -> 201 raesoriano/wandog-probe2
    delete name="wandog-probe2"              -> 404 Not Found
    delete name="raesoriano/wandog-probe2"   -> 204 deleted

### What the live log shows

From `/tmp/wandog_events.jsonl`, the whole session:

| | count |
|---|---|
| `github_repo` calls | 20 |
| deletes sent with a bare name | 13 of 13 |
| 404s returned | 13 of 13 |
| times the person was told the repos "might have already been removed" | 4 |

The four repositories were present the entire time, and the agent's own `list`
call showed them with both `name` (bare) and `full_name` (qualified). It kept
reaching for `name`.

### The fix

Two layers, same shape as the clock and search fixes in §17 -- a deterministic
one that works whether or not the model remembers, and a prompt line that
makes the model's reasoning and its report to the person correct.

1. `mcp_bridge._shape_github_args` qualifies a bare name for
   `get`/`fork`/`delete` before the call reaches the server, using
   `config.OWNER_GITHUB`. It leaves `create` alone (a slash would be a
   malformed repository name), leaves an already-qualified name alone, and
   leaves an `org` call alone.
2. `config.TOOLS_SYSTEM_PROMPT` says `name` is bare only for create, gives the
   owner to prefix, and forbids reporting a repository as already removed on
   the strength of a 404 alone.

Verified end-to-end through the real MCP server with the exact arguments the
live log recorded: `delete name="wandog-e2e-probe"` -- shaped to
`raesoriano/wandog-e2e-probe` -- now returns `{"status": "deleted"}`.

Note what the fix does *not* claim. Replaying the logged conversation against
the live model in a clean context, it qualifies the name by itself in 6 of 6
trials, with or without the new prompt line. The 13 bare names came out of a
long, noisy session where the model had a successful bare-name `create` and a
`list` full of bare `name` fields in front of it. That is precisely the case a
prompt rule cannot be relied on to win, and the reason the shim exists.

`test_tool_guidance.py` grew two groups (7/7 now): the shaping table, including
the cases it must not touch, and the 404 wording.

## 19. "I have no tool for that" was false, and `gh` was never installed

Asked to enable GitHub Pages, the agent said:

> I'm sorry, but I don't have a tool that can change the internal settings of a
> GitHub repository, like enabling GitHub Pages. You will need to go to the
> Settings tab of your repository...

Asked whether it was sure, it said **"I am sure."** The event log
(`/tmp/wandog_events.jsonl`, idx 2038 and 2050) has both, and only afterwards
does it call `github_repo action=get`.

The diagnosis was right. `jeolee.github.io` was public, its default branch was
`main`, `index.html` (693 B) was committed — and `has_pages` was `false`, so the
404 the person kept hitting was real. What was wrong was the claim of inability.

### It is not true: `curl` is allowlisted and the token is in bash

`github_repo` really does have five actions (create, get, list, fork, delete).
But the bash tool is a separate door, and the REST API behind it works:

| Check, through the real spawned MCP bash server | Result |
|---|---|
| `GITHUB_TOKEN` in the bash environment | present, 93 chars |
| `GH_TOKEN` | present |
| `curl` private repo, with the token | **200** |
| same `curl`, without the token | 404 — so the 200 was the token |
| `POST /repos/…/pages` on a **private** repo | **422** plan does not support |
| `POST /repos/…/pages` on a **public** repo | **201** with `html_url` |

The last two matter together. A 422 is the plan refusing a private repo; it is
proof the authenticated write reached the endpoint, and it is not a missing
tool. On a public repo the same call returns 201.

So the fix is a prompt paragraph that draws the narrow line the model was
missing: *"this one tool lacks an action"* is not *"I have no way to do this"*.
A missing action means reach for the REST API — and say so after the call, not
instead of it.

Measured on the live model with the real 14-tool schemas, two trials each:

| Question | Before | After |
|---|---|---|
| "Can you enable GitHub Pages on raesoriano/jeolee.github.io?" | "I cannot directly enable GitHub Pages through the API" ×2, no call | confirms, then `bash curl -X POST …/pages` ×2 |
| "…now, I confirm." | `gh repo edit --enable-pages` ×2 | `bash curl -X POST …/pages` ×2 |
| "Do you have a way to change a repo's settings, like visibility?" | "No, I do not have a tool that can change a repository's settings" ×2 | "Yes … via `curl` in the bash tool" ×2 |

Then the emitted command was run through the real bash tool against the real
repository, and it worked:

```
POST https://api.github.com/repos/raesoriano/jeolee.github.io/pages
     {"source":{"branch":"main","path":"/"}}      -> 201
pages status: building -> built
https://raesoriano.github.io/jeolee.github.io/  -> 200, 693 B, the Jeolee page
```

The person's site is live. That is the point: it was one `curl` away the whole
time.

### The `gh` bug underneath

The prompt said GitHub access came "through `git` and `gh` in bash". **`gh` is
not installed on this host** — `gh: not found`, exit 127, in the event log at
idx 583 (`gh repo rename …`) and again at 1627 (`gh repo clone …`). That line
sent the model to a command that can only fail, and the before-column above
shows it reaching for `gh repo edit --enable-pages` — a flag that does not
exist anyway. The prompt now names `git` and `curl` and drops `gh`.

### One thing to know about this account

GitHub Pages on a free plan is **public repositories only**. `POST /pages` on a
private repo answers 422 *"Your current plan does not support GitHub Pages for
this repository."* The prompt says so, so a 422 is reported as a private-repo
fact rather than as a missing tool.

`test_tool_guidance.py` group 8 pins the wording, including a check that the
prompt does not advertise `gh`. 8/8 groups pass. Probe repositories
(`wandog-pages-probe`, `-2`, `-3`) were created and deleted during measurement;
the account is back to its 22.

## 20. `write_file` was never forgetting the path — it was out of tokens

Reported as "it is having trouble writing files", from a session where the
person asked for a flappy bird clone. The agent answered "Writing the file."
eleven times in a row, never wrote anything, and apologised: *"I am so sorry, I
keep forgetting to include the file path!"* It was wrong about itself.

### What the log shows

`/tmp/wandog_events.jsonl`, 18 Nov 2026:

| | |
|---|---|
| `write_file` calls | 26 |
| failed | **24** |
| succeeded | 2 (idx 586, 1640 — both small files) |
| failures that were `Missing required argument(s): path.` | **24 of 24** |
| longest unbroken run | 11 attempts (idx 1756-1854), then 11 more (2328-2409) |

Every single failure is the same one argument, and every call *did* return a
result — so the JSON parsed fine and `path` was simply not in it. Nothing was
silently dropped.

### Why

`config.LLM_MAX_TOKENS` was **300**, and it was applied to every request. That
is a *speech* budget — one or two sentences, about 40 tokens. A turn that calls
`write_file` spends that same budget on the file.

The model emits the arguments body-first, path-last:

```
{"content": "<the entire 5 KB HTML file>", "path": "flappy_bird.html"}
```

So a cap that cuts the stream inside `content` closes the JSON **before `path`
is ever emitted**. Measured against the live model, the real 14-tool schemas,
and the actual question:

| `max_tokens` | `finish_reason` | argument length | keys emitted |
|---|---|---|---|
| 300 | `length` | ~1127 | `['content']` |
| 600 | `length` | ~2030 | `['content']` |
| 1000 | `length` | ~3900 | `['content']` |
| 4096 | `tool_calls` | ~5840 | `['content', 'path']` |
| 8192 | `tool_calls` | ~5760 | `['content', 'path']` |

A 5 KB file needs roughly 1500 tokens. **300 could never fit one**, so retrying
was futile — which is exactly the loop the log shows.

The error message made it worse. `Missing required argument(s): path.` is
literally true and actively misleading: it points at the model's intent when
the cause is the token budget. The model believed it, apologised for it, and
retried identically.

### The fix, in two layers

1. **`config.LLM_TOOL_MAX_TOKENS` (8192), applied only when the request carries
   tools** (`agent.py`, `ClockedLLMService.build_chat_completion_params`). A
   plain spoken turn keeps the 300-token speech ceiling.

   It is a ceiling, not a target, so speech is unchanged: "What is the capital
   of France?" measured **31 characters and 0.6 s at both 300 and 8192**. The
   only cost is on a turn that actually writes something — 1.9 s to 3.3 s for a
   5 KB file.

2. **`mcp_bridge._missing_path_hint`** — if a call still arrives pathless with
   a body, the error names truncation, says the path comes after the body, warns
   that an identical retry fails, and offers `mode="append"` chunking.

Measured on the live model, 4 trials per question:

| Question | Old cap 300 | New cap 8192 |
|---|---|---|
| flappy bird clone → one HTML file | 4/4 body, no path | **4/4 path present** |
| portfolio site → `index.html` | 4/4 body, no path | **4/4 path present** |

And through the real `ClockedLLMService` → real MCP server, the file landed:

```
request max_tokens = 8192  tools present = True
finish = tool_calls  keys = ['content', 'path']  content_len = 4954
write_file -> {"path": "/tmp/onit/data/flappy_bird.html", "size_bytes": 4954,
               "status": "success"}                     (0.0s)
```

`/tmp/onit/data/flappy_bird.html` is a complete, well-formed document.

### The guard changes the next step, not just the words

Feeding a truncated call back to the model with each error text, for a body too
large for even the new cap (90 KB):

| Error text fed back | What the model did next |
|---|---|
| old: `Missing required argument(s): path.` | blind retry, same call, 15752 chars |
| new: the truncation hint | **no retry** — "I cannot write a file of that size in a single call because it exceeds the system's output limits…" |

That is the point of the second layer: it turns a loop into a different
outcome, including an honest refusal.

### The log hid this, so the log now says more

`tool_call` recorded `args=_short(args)`, capped at 600 chars. Every one of
those flappy calls therefore read **`argslen=602`** regardless of whether the
model sent 1 KB or 19 KB — the evidence was flattened out of the log. A new
`args_chars` field records the true size beside it (6015 for the 6 KB probe).

### Notes

- Schema reordering does **not** help: putting `path` first in the schema, and
  marking it `required`, both left the emitted order as `content, path` — 6/6
  trials each. The budget was the whole problem, so that is what was changed.
- `bash` called through the handler with `**kwargs` answers *"1 validation
  error for call: command — Unexpected keyword argument"*. That is
  **pre-existing** (reproduced on the stashed baseline) and is a property of the
  positional-only handler signature; it is not related to this change.

`test_tool_guidance.py` is 12/12 groups, with four new ones: the budget split,
the pathless hint (including every case it must *not* touch), the true-size
logging, and the prompt wording. `test_tool_parity`, `test_tool_secrets`,
`test_tool_result`, `test_results_cap`, `test_clock`, `test_guard`,
`test_turn_context`, `test_eventbus`, `test_muted_turn` and
`test_ctrl_enter_interrupt` all still pass.
## 21. Feature register, by commit

Every feature in the running agent, with the commit that added it and the
section that explains it. `git show <hash>` is the full argument.

| Feature                                      | Commit   | See    |
|----------------------------------------------|----------|--------|
| Full-duplex voice loop: VAD -> LLM -> TTS    | 175d90c8 | 1, 2   |
| Torch-free VAD via onnxruntime               | b31ee5e5 | 2, 6   |
| LLM key resolved as OnIt does (no export)    | 759d96dd | 3, 11  |
| Approval gate / safeguards from OnIt         | 647e1da9 | 7      |
| Local ASR (faster-whisper) + LAN dashboard   | 55fd9bdb | 8, 9   |
| Audio devices pinned by index                | 8e8dc7e8 | 10     |
| Web search tool reachable by the model       | 40e4de8f | 11b    |
| Barge-in: user speech interrupts the bot     | 5281e0b5 | 13     |
| README + fd-benchmark suite                  | 726bac17 | README |
| Streaming ASR: text while still speaking     | 640798ac | 8, 12  |
| News served by Tavily, not ddgs              | 3d1333dd | 11b    |
| News freshness refresh + retry budget        | bad13039 | 11b    |
| Search results dated and readable            | 0850bb2a | 11b    |
| Prompt clock carries date AND time           | 33861ac5 | 17     |
| Tool use is spoken: call, then outcome       | bd7d8825 | 12     |
| Tool secrets from secrets.yaml               | e2f3c119 | 7      |
| Model result cap 1 200 -> 24 000 chars       | 33e92b12 | 11b    |
| Cut results tell the model they were cut     | 3ac4ea49 | 11b    |
| One VAD analyzer per consumer                | ae567ee9 | 12     |
| Dashboard: Ollama-style chat, typing, Stop   | fe5bb202 | 9      |
| Light/dark switch, tintable mic, Clear       | 6f617c27 | 9      |
| Sticky bar, measured transcript, husky       | 22752df6 | 9      |
| Ctrl+Enter sends; Enter makes a newline      | f55b90bd | 9      |
| Default light; Send no longer sticks on Stop | 9211480f | 9      |
| Piper Ryan voice + owner identity block      | 36d7337e | 15     |
| Rename Fweddie -> WanDog, venv path          | 36d7337e | 15     |
| Wordless turn ends; mute persists            | 7cb8fbd5 | 16     |
| Ctrl+Enter interrupts the bot                | 7cb8fbd5 | 16     |
| Prompt states GitHub, long tasks, clock      | 199fcb2c | 17     |
| github_repo qualifies a bare repo name       | 06e809c7 | 18     |
| Prompt: Pages via curl, not 'I cannot'       | df786c3b | 19     |
| Tool-call token budget separate from speech  | 9800d94a | 20     |
| Sessions: continue by default, restart flag  | 97956495 | 23     |
| Per-session tool jail (data_path per call)   | 97956495 | 23     |
| Handover notes, compacted at exit            | 97956495 | 23     |
| Chat history replayed on continue            | 97956495 | 23     |
| File upload into the session folder          | 97956495 | 23     |
| strict/session/expanded search tiers         | 97956495 | 23     |

## 22. Issue register, by commit

Every fault that cost real time, with the commit that fixed it. A fault with
no commit here is still open -- see §14.

| Issue                                          | Fixed in | See   |
|------------------------------------------------|----------|-------|
| Pipecat 1.15 API differs from the old docs     | 5281e0b5 | 4     |
| Agent silent; blank dashboard; mic too hot     | c9760693 | 12    |
| Pipeline never started; VAD ordered after STT  | 0fc1dcec | 12    |
| Two LLMContexts split one turn in two          | 491f6e59 | 12    |
| Mic gain forced to 100% at every startup       | a1372ad2 | 10    |
| Partial-transcript tap stalled the pipeline    | f9088cdd | 8, 12 |
| Dashboard crashed on every tool result         | 45aeeee9 | 9     |
| Model search handed an empty result            | 4f3d4727 | 11b   |
| Search results arrived undated                 | 0850bb2a | 11b   |
| Clock gave the date but not the time           | 33861ac5 | 17    |
| Self-echo: speaker placement, not AEC          | 48b828b3 | 13    |
| News returned last week's stories              | bad13039 | 11b   |
| Mic cancelled the agent's own searches         | 3d1333dd | 11b   |
| Results clipped at 1 200 chars, silently       | 33e92b12 | 11b   |
| One pause split a sentence into two turns      | ae567ee9 | 12    |
| Send button stuck on Stop                      | 9211480f | 9     |
| Enter did nothing in the text box              | 00aa5c10 | 9     |
| Download command failed in a fresh clone       | f4836c85 | 5     |
| A wordless turn never ended                    | 7cb8fbd5 | 16    |
| A persisted mute did not mute                  | 7cb8fbd5 | 16    |
| GitHub delete 404'd on a bare repo name        | 06e809c7 | 18    |
| "I have no tool for that" was false            | df786c3b | 19    |
| write_file dropped path: token budget          | 9800d94a | 20    |
| A session folder created by every import       | 97956495 | 23    |
| dashboard: local `session` shadowed the import | 97956495 | 23    |
| Widening the search tier was impossible        | 97956495 | 23    |
| apply_search_mode: ToolsSchema, not a list     | 97956495 | 23    |
| DOCUMENTS_PATH set; servers read ONIT_*        | 97956495 | 23    |
| Sessions rooted in /tmp (lost on reboot)       | 97956495 | 23    |
| Transcript 80 px short of the main area        | PENDING  | 23    |
| The bar ran 300 px past the window edge        | PENDING  | 23    |
| Scrollbar inboard: container padding, not gap  | PENDING  | 23    |
| Switch compared against a stale session id     | PENDING  | 23    |

## 23. Sessions: one folder per conversation, and an upload that asks first

Built 2026-11-18 on branch `file-upload` (`97956495`). Implements `NEXT-STEPS.md`
§5, which had been plan-only.

### What the feature is

One session is one folder. The agent continues the last session by default;
`--restart-session` starts a new one with an empty chat history. An uploaded file
lands in that folder, and the folder is the boundary the session's tools are
confined to. A notes file per session holds a handover summary, so a later
session can read what happened in an earlier one.

```
.venv/bin/python agent.py                       # continue the last session
.venv/bin/python agent.py --restart-session     # start a new one
```

```
wandog-sessions/                 <- MCP_DATA_PATH (the jail root)
  .current                       pointer: the live session id
  <id>/                          e.g. 2026-11-18T1620-a3f1
    meta.json                    id, started, model, mode, uploads[]
    events.jsonl                 the log the dashboard tails
    notes/session.md             handover record
    work/                        <- injected as data_path: the tool jail
      uploads/  created/  tool-results/
```

### The three corrections the plan needed

**The root is not `/tmp`.** The plan put sessions under `/tmp/onit/data/sessions/`.
`/tmp` is tmpfs on this host (`findmnt -no FSTYPE,TARGET /tmp` → `tmpfs`), so it
is RAM: the folder, the log and the handover notes would all be wiped on reboot.
That makes "continues the previous session by default" false exactly when someone
comes back to it, and destroys the one artefact the feature exists to produce. It
had already happened in this project — the 15:20 reboot took the event log and
four test fixtures with it.

The root is now `../wandog-sessions` relative to the checkout, derived from
`__file__` rather than hardcoded, so it is durable, portable to another clone, and
outside the repo so it is never committed and `git clean` cannot delete a
person's work. `WANDOG_SESSIONS_ROOT` overrides it.

The consequence worth knowing: moving or re-cloning the repo does not move the
sessions with it. "My sessions are gone" is almost always "I cloned somewhere
else".

**The jail root had to become the sessions root.** The plan kept `MCP_DATA_PATH`
flat and put session folders inside it, on the reasoning that a session folder is
*inside* the jail rather than a new jail. That reasoning is right about the
mechanism and wrong about the effect: one shared jail cannot confine a session,
because every session sits in the same boundary.

OnIt's servers accept a per-call `data_path` and enforce it as the boundary —
but only when it sits inside the server-wide `DATA_PATH`:

```python
if base != abs_data and not base.startswith(abs_data + os.sep):
    raise ValueError("data_path must be within the server data directory")
```

So the root has to *contain* every session for a session to be its own jail. With
`MCP_DATA_PATH = <sessions root>`, `mcp_bridge` injects `data_path=<session>/work`
on every call whose schema declares the parameter, and one session cannot read a
sibling's files. That is stronger than the plan, which never attempted separation.

The injection is per tool, not blanket: OnIt's server validates arguments against
the tool's schema, so sending `data_path` to a tool that does not declare it is a
hard rejection. The bridge records which tools accept it at discovery time.

**The tier filter had to run from the full tool list.** This was a real bug found
by a test, not a design preference. Narrowing to `strict` removes the web tools
from the `LLMContext`. Filtering *the context's* tools on the next change
therefore found nothing to restore, and the change was one-way: strict worked,
and nothing could ever widen again. `apply_search_mode` now filters from
`bridge.tools` — the complete set the pipeline was built with — every time.

### The middle tier was free

The user's hierarchy is: the uploaded file, then the other files in this session,
then the web. Three tiers, narrowest first:

| Tier       | Reaches                      | Tools offered              |
|------------|------------------------------|----------------------------|
| `strict`   | only the uploaded file       | file tools only            |
| `session`  | + the rest of this session   | + `local_search` (default) |
| `expanded` | + the web                    | everything                 |

`local_search` already indexes the session base directory when given a
`data_path`, so "search the other files in this session" needed no new code. Only
the two ends of the ladder are new work.

Enforcement is by tool removal, not by asking the model nicely — the same
principle as the approval Guard. `context.tools` is a `ToolsSchema`, and Pipecat's
OpenAI adapter reads it on every request, so `set_tools` changes what the next
request carries without rebuilding the pipeline. The prompt is the second half:
it says what to say when the answer is not in the file ("that isn't in
report.pdf"), and that widening is the user's decision.

A tier change is matched deterministically in the harness
(`session.mode_from_text`), not inferred by the model: a permission change the
model can talk itself out of is not a permission change. It is recorded in
`meta.json` and in the event log, and the dashboard shows a caption whenever the
tier is not the default, because a limit you cannot see is indistinguishable from
the agent not knowing something.

### Continuity means replaying the log

Continuing a session has two halves. Files are free — the folder is reused. Chat
history is not: the context is in-memory, so a restart starts blank.

`session.read_turns` replays the last 20 `user_text`/`bot_text` events into
`context.set_messages`. Tool traffic is deliberately *not* reconstructed — it is
not cleanly rebuildable as messages, and the duplicated-system-prompt bug
documented at `agent.py:838` is exactly what happens when the context is handed
something it should not have. A continued session remembers what was said, not
every tool's raw output. Replay stops at the most recent `cleared` event, so Clear
still means clear.

A restarted session replays nothing, so "clear chat history" falls out of the flag
for free, with no clear-file involved.

### The notes file is written twice over

`notes/session.md` has two writers on purpose:

1. One line per turn as the turn lands, written immediately, so a hard kill — a
   power cut, a SIGKILL — still leaves a record of what was said.
2. A compaction on shutdown that replaces those running lines with a summary:
   session id, times, model, tier, uploads, what was asked, what was answered.

Replacing rather than appending is the point. A summary that grows forever is a
transcript, and the reason to compact is that the next reader should not have to
read one.

`notes/` sits outside `work/` deliberately: the agent writes it in-process, so a
session's own file tools can neither rewrite its log nor edit its history. That is
the "separate folder from the temporary folder" the requirement asked for, made
literal.

### Who mints a session

Only the agent. The dashboard follows the pointer the agent wrote. Otherwise
opening the dashboard first would mint an empty session that the agent then
adopts, and the "new session" the user asked for by restart would be
indistinguishable from one the dashboard made by accident.

And only on *running* WanDog, not on importing it. Without that rule every test
that does `import config` would leave a session folder beside the checkout — which
is exactly what happened on the first run of the suite, and why
`session._may_mint()` exists. `sys.argv[0]` is what distinguishes the two entry
points: `streamlit run dashboard.py` puts streamlit there, `python agent.py` puts
`agent.py` there.

### The upload

The composer's attach control was a disabled placeholder (`attachph`, "coming
soon"). It is now a popover holding `st.file_uploader` — a popover rather than a
bare dropzone because the composer is one row of controls and the layout test's
geometry assertions depend on its height.

The file is written by the **dashboard**, into the session's `work/uploads/` —
already inside the agent's jail — so the agent reads it with `read_file` like any
other file rather than receiving its bytes through a control file. What travels
through `<log>.upload.json` is the name, which is all the spoken question needs.

Refused with a sentence a person can hear: over 2 MB, an extension outside the
allowlist, an empty file, a name that traverses. A same-named upload is numbered
rather than overwriting the file the agent was told to answer from.

An upload is not a silent ingestion. The agent says which file it has and asks
what it may use — spoken as a `TTSSpeakFrame` literal, not through the LLM,
because the question is fixed and letting the model paraphrase it risks it
answering the upload as if the upload were the request.

### Issues hit while building

| Issue | Cause | Fixed by |
|---|---|---|
| Every `import config` left a session folder beside the checkout | `activate()` minted on import, and any process counts as an entry point | `session._may_mint()`: agent only, and only from `agent.py` |
| `dashboard.py` crashed with `'NoneType' has no attribute 'UPLOAD_MAX_BYTES'` | the module already binds a local `session` (the `session_start` event), shadowing the import | import as `sess` |
| `--restart-session` was ignored in the launch test | the harness overwrote `sys.argv` and dropped the flag | rebuild `argv` as `["agent.py"] + sys.argv[1:]` |
| Widening the tier was impossible | filtering the context's current tools, not the bridge's full list | filter from `bridge.tools` |
| `apply_search_mode` raised `Unsupported tool type` | `context.tools` is a `ToolsSchema`, not a list; the callables live on `.direct_functions` | unwrap `.function`, hand back the callables |
| `test_dashboard_ui` asserted attach was a *disabled button* | it was, until the feature landed | assert the uploader is present instead |
| `mcp_bridge` set `DOCUMENTS_PATH`; nothing read it | all three servers read `ONIT_DOCUMENTS_PATH` | renamed; harmless until now only because it defaulted to empty |

Two pre-existing failures were confirmed against unmodified `HEAD` rather than
assumed: `test_ctrl_enter_interrupt.py` (the composer never shows a busy state)
and `test_pipeline_asr.py` / `test_vad_shared_state.py` (missing `/tmp` fixtures,
skipped). Neither is caused by this work.

### Tests

| Test | Asserts |
|---|---|
| `test_session.py` | pointer round-trip; restart mints and the old folder survives; the dashboard never mints; traversal refused; `config` derives every path from the live session |
| `test_session_launch.py` | the same through four real launches of `agent.py`: folder, log and jail each time |
| `test_session_jail.py` | `data_path` injected only where declared, the model's value overwritten, and a real spawned OnIt server refusing a sibling's file |
| `test_session_replay.py` | 3 turns replay as 6 messages; restart replays 0; a `cleared` event truncates; a corrupt line is skipped; notes append per turn and compact on shutdown |
| `test_upload.py` | size/type/traversal refusals; the strict prompt; the tier phrases; `set_tools` really removing and restoring the web tools |
| `test_upload_e2e.py` | the join: dashboard stores the file, the real watcher consumes the control file once, the agent speaks the question rather than ingesting silently |

`test_dashboard_ui`, `test_dashboard_layout`, `test_dashboard_controls`,
`test_eventbus`, `test_guard`, `test_clock`, `test_turn_context`,
`test_tool_guidance`, `test_results_cap`, `test_tool_parity`, `test_bargein` and
`test_muted_turn` all still pass.

### The sidebar, and what it changed

Built on the `sidebar` branch, from `file-upload`. The rail lists every session
newest first; a row carries the time it started and the first thing that was
asked, because an id does not tell you which conversation it was.

The design decision worth keeping: **browsing is not switching.** Clicking a row
shows that session read-only and leaves the agent where it is. One button doing
both would mean that looking something up silently ended today's conversation.
The consequence is that the page has TWO session ids — LIVE (what `.current`
names, where new turns go) and VIEWED (what is on screen) — and everything that
acts on the live session has to say so: Send, Clear and Attach are refused while
browsing, because otherwise the message lands in a conversation that is not
visible.

Switching is a transaction, and it lives in one place (`_switch_to` in
`build_pipeline`) because it has to move five things together: the LLM context,
the tool jail, the event log, the four control files, and the notes. The context
is cleared FIRST, before anything is re-pointed — the assistant aggregator writes
the reply back into the shared context as it finishes, so a reply still in flight
from the old session would otherwise land in the new one's history.

The two sidebar requests (`.switch`, `.handover`) live at the sessions ROOT, not
beside the event log. Every other control file is derived from the live log, and
that is exactly wrong here: a request naming a different session would move the
moment it was acted on.

`notes` is the one thing that crosses the jail. It has to be in-process — a
session's tools are confined to its own `work/`, so no tool can open a sibling's
`notes/`. The agent reads the file itself and injects it as a **system** message,
not a user turn: it is background the model was given, and a user turn would look
like a question that still needs answering.

### The three bugs the live checks caught

1. **`width: 100vw` had to become `100%` — but the container was capped.** With
   the rail open the main area no longer starts at the window's edge, so `100vw`
   overhung it. The obvious fix (100% of the containing block) then inherited
   `.block-container { max-width: 820px }`, and at 1280 px the transcript was
   380..1200 inside a main area of 300..1280 — 80 px of dead space either side.
   The 1100 px window the live checks use made this INVISIBLE: there the main
   area is exactly 800 px, so 100% of it is 800 and everything lined up. Fixed by
   dropping the cap, and the geometry is now checked at a second width
   (`test_live_layout_at_a_wide_window`) precisely because a rule that only holds
   where the numbers coincide is not a rule.

2. **`left` and `right` together with an inherited `width: 100%` over-constrains.**
   The bar computed to 1100 px wide starting at x=300, so it ran to x=1400 in a
   1100 px window and its right-hand controls were off-screen. `width: auto` is
   what makes two offsets size the box.

3. **The scrollbar sat inboard because of container padding, not the gutter.** The
   first fix pulled a negative margin equal to the scrollbar's width; that only
   SHIFTS a box, it does not widen it, and the real bound was Streamlit's 1rem of
   `.block-container` side padding. Removing the padding (the reading column keeps
   its own) is what put the thumb on the main area's edge.

### Still not built

Nothing prunes old sessions. A long-lived deployment accumulates folders and there
is no `--forget` yet, and no way to delete one from the rail.

The handover is manual: you press `notes` on a row. It is not automatic when you
switch, because importing a summary the user did not ask for is a silent change to
what the model knows.

---
