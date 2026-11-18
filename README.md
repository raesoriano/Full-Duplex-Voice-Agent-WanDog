# Full-Duplex Voice Agent: WanDog

WanDog is a local, full-duplex voice command agent designed to run on resource-constrained hardware (e.g., Raspberry Pi 5). It provides a low-latency "listen-think-speak" loop with native support for barge-in (interruptions) and extensible tool use via MCP.

> **Status (2026-11-17):** working on the Pi. Streaming partials appear while you
> speak, the turn closes ~0.9 s after you stop, and the reply is about the
> question just asked. Commit `f4836c85`, verified from a fresh clone.

## 🏗️ System Architecture

WanDog is built on the **Pipecat** framework, utilizing a linear pipeline that transforms raw audio into spoken responses.

### The Voice Loop
`Mic` $\rightarrow$ `VAD (Silero ONNX)` $\rightarrow$ `ASR (streaming zipformer + whisper final pass)` $\rightarrow$ `LLM (OnIt/vLLM)` $\rightarrow$ `TTS (Piper)` $\rightarrow$ `Speaker`

- **VAD (Voice Activity Detection):** Uses `SileroVADAnalyzer` (ONNX runtime) to detect speech without requiring heavy PyTorch dependencies.
- **ASR (Speech-to-Text):** Streams by default, so text appears while you are
  still talking. See the Streaming ASR section below.
- **LLM:** Connects to an OpenAI-compatible endpoint (typically a DGX cluster running vLLM) using `OpenAILLMService`.
- **TTS (Text-to-Speech):** Uses **Piper** with the **`en_US-ryan-medium`**
  voice, a fast local neural TTS that generates high-quality speech in
  real-time on CPU. The 61 MB ONNX model and its `.onnx.json` config are
  fetched into `~/.piper` on first use.
- **Barge-in:** Implemented via `LLMUserAggregator` and `VADUserTurnStartStrategy`. When the VAD detects the user speaking while the bot is talking, an `InterruptionFrame` is broadcast to immediately silence the TTS.

## 🧠 Context & Prompt Engineering

### System Prompt
WanDog's persona is defined in `config.py`. It is designed for **spoken** interaction:
- **Persona:** Friendly, upbeat, and concise.
- **Owner:** `config.OWNER_PROMPT` tells the model it is speaking with **Ron Andrei
  Soriano** (GitHub `raesoriano`), so "who am I", "what did I push" and "my
  repos" resolve to him instead of "I don't know your name".
- **Constraints:** Replies are limited to 1-2 short sentences. No markdown, bullet points, or emojis.
- **Clock:** The prompt states the current date *and time* (`config.clock_line()`), and `ClockedLLMService` re-reads it on every inference so a long-running agent cannot drift. It is placed on the service's `system_instruction` only — putting it in the context as well sends it twice.
- **Tool Awareness:** When MCP tools are enabled, a secondary prompt is appended, instructing the model on how to call tools, handle approvals, and report refusals without "routing around" them.

### Context Management
- **Input:** The agent consumes audio/text turns.
- **State:** Managed by Pipecat's pipeline state.
- **Tool Integration:** Tool results are fed back into the LLM context, which then generates a concise spoken summary for the user.

## 🛠️ MCP Services (Model Context Protocol)

WanDog extends its capabilities through a bridge to **OnIt's MCP servers**, allowing the voice agent to interact with the local system and the web.

| Server | Profile | Capabilities |
|---|---|---|
| `ToolsLocalMCPServer` | `local` | Bash shell execution, file system R/W, local document search |
| `ToolsNetMCPServer` | `net` | Web search (Ollama/DuckDuckGo), weather updates |

**Security & Guardrails:**
- **Jailroot:** File tools are restricted to a specific `MCP_DATA_PATH` (e.g., `/tmp/onit/data`).
- **Human-in-the-Loop:** Critical commands trigger an approval request. The agent notifies the user it is "asking a person" and waits for a decision via the dashboard or API.

### Tool secrets (API keys)

The MCP servers are separate processes, so a key reaches them only through the
environment the bridge hands to `stdio_client`. Each key resolves in this order,
first hit wins:

| # | Source | Notes |
|---|---|---|
| 1 | `secrets.yaml` in this repo | per-project override; git-ignored, template is `secrets.example.yaml` |
| 2 | OS keychain, service `onit` | exactly what `onit setup` writes, so a provisioned machine needs no file |
| 3 | unset | that one tool degrades; the agent still runs |

Because the keychain is consulted second, **deleting `secrets.yaml` must change
nothing** on a machine already set up — `test_tool_secrets.py` asserts this.

```bash
cp secrets.example.yaml secrets.yaml
chmod 600 secrets.yaml
$EDITOR secrets.yaml          # quote every value
```

| Key | Env var | Tool that needs it |
|---|---|---|
| `openweathermap_api_key` | `OPENWEATHERMAP_API_KEY` | `get_weather` |
| `tavily_api_key` | `TAVILY_API_KEY` | `search` (first tier; else DuckDuckGo) |
| `ollama_api_key` | — not forwarded | LLM endpoint only; never a tool |
| `github_token` | `GITHUB_TOKEN` | `github_repo`, `git push` |
| `huggingface_token` | `HF_TOKEN` | HuggingFace model downloads |

### News freshness

`search` with `type="news"` is the path for anything current. WanDog tunes it
on its own side, because OnIt's news branch takes no time filter and orders by
relevance rather than date:

| Env var | Default | What it does |
|---|---|---|
| `WANDOG_MCP_NEWS_OVERFETCH` | `10` | results to request before trimming |
| `WANDOG_MCP_NEWS_SORT` | `1` | sort newest-first and add an `age` field |
| `WANDOG_MCP_NEWS_REFRESH_RETRIES` | `2` | extra calls when the newest hit is stale |
| `WANDOG_MCP_NEWS_REFRESH_BUDGET` | `4.0` | seconds the refresh may spend |
| `WANDOG_MCP_NEWS_FRESH_DAYS` | `0` | newest age that counts as fresh |
| `WANDOG_MCP_NEWS_BACKEND` | `tavily` | `tavily` = Tavily first, ddgs fallback; `onit` = always ddgs |
| `WANDOG_MCP_NEWS_TAVILY_DAYS` | `1` | Tavily's recency window (1-7 days) |
| `WANDOG_MCP_NEWS_TAVILY_DEPTH` | `advanced` | Tavily search depth for news |

Results now reach the model newest-first and carry `age` ("today", "3 days
ago") beside `date`, so the agent says "as of today" only when that is true.
Set `WANDOG_MCP_NEWS_SORT=0` to restore the server's own ordering.

#### Why news does not go to OnIt's news branch by default

OnIt's `search(type="news")` calls ddgs and nothing else — it never reaches the
Tavily tier its own *web* branch uses. ddgs' news engine is Bing-backed
(`ddgs/engines/duckduckgo_news.py`, `provider = "bing"`). Measured on 17 Nov
2026, newest published date per provider, same four queries:

| Query | ddgs (Bing) | Tavily `topic=news, days=1` |
|---|---|---|
| Alaska US Senate election results | 15 Nov | 17 Nov |
| Alaska Senate race Peltola Sullivan | 17 Nov | 17 Nov |
| US Senate election results 2026 | 14 Nov | 17 Nov |
| latest news Jensen Huang Nvidia | 15 Nov | 17 Nov |

ddgs was 0-3 days behind; Tavily was same-day on all four, and it returns a real
`published_date` plus honours a `days` window. So with a Tavily key present,
news is served by Tavily and falls back to OnIt's ddgs branch on failure or an
empty page. Without a key the setting is inert and news stays on ddgs.

Both providers return the same payload shape, so the date sort, the `age`
labels and the model cap are unchanged whichever one answers.

#### Why apnews.com is not the same source here

Google surfaces AP's **live results project** pages, e.g.
`https://apnews.com/projects/elections-2026/alaska-us-senate-results`. Those are
JavaScript applications: the HTML is 142 KB but the visible text is ~3.8 KB and
contains no candidate names and no tallies, so `fetch_content` on one returns
boilerplate. AP has not published a new *dated* Alaska article recently, which
is why no dated news provider — ddgs or Tavily — returns one. ddgs does carry AP
when the query is site-scoped (`... site:apnews.com` returns 10 AP rows), but
they are all older than the race's last article.

`ollama_api_key` is deliberately kept out of the spawned servers' environment.
OnIt's search chain is Tavily → DuckDuckGo and has no Ollama tier, while
Ollama's web search is metered per call — a forwarded key would be a cost
liability with no benefit. See `FORBIDDEN_TOOL_SECRETS` in `config.py`.

Secrets are injected into the spawned subprocess only — never into WanDog's
own process, and an ambient `export` still wins over both sources. Startup
logs one line per key with its source and length, never its value. Note that
the `bash` tool runs as you, so it can read `secrets.yaml`; keep high-value
tokens in the keychain.

## 📈 Evaluation & Benchmarking

To ensure "full-duplex" performance, WanDog is evaluated using the `fd-benchmark` suite.

### Key Challenges & Metrics
- **Barge-in Latency:** Measuring the delta between user speech start and bot silence.
- **Turn-Taking:** Evaluating the smoothness of transitions between user and assistant.
- **VAD Accuracy:** Tuning `VAD_MIN_VOLUME` to prevent false triggers from background noise while remaining sensitive to human speech.
- **End-to-End Latency:** Tracking the time from the end of user speech to the first audio byte of the response.

## 🎙️ Streaming ASR

The ASR streams, so text appears **while you are still talking** instead of after you stop.

| measurement | segmented whisper `small` | streaming (default) |
|---|---|---|
| first words on screen | never | **0.4–0.7 s** after speech start |
| final transcript after you stop | 4.7–4.9 s | **0.86 s** |
| turn submitted to the LLM | 5.2 s | **0.93 s** |

Two models run together:

- **sherpa-onnx streaming zipformer** (int8, RTF ≈ 0.10) decodes 20 ms chunks as they arrive and drives the live partials. It is fast but weak on numbers (`"2 plus 2"` → `"TOO PLUS"`).
- **faster-whisper `tiny.en`** produces the authoritative final text from the same buffered audio, so accuracy does not regress. The final pass is skipped when the streaming model found no words, which stops Whisper hallucinating sentences out of noise.

Both are configurable: `WANDOG_ASR_BACKEND` (`streaming` | `whisper`), `WANDOG_ASR_FINAL` (`whisper` | `none`), `WANDOG_ASR_FINAL_MODEL`, `WANDOG_TURN_STOP`.

**The model weights (321 MB) are not in the repo** — `models/` is gitignored. Download once per machine:

```bash
mkdir -p models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
```

The service names this exact command in its error message if the directory is missing. Set `WANDOG_ASR_BACKEND=whisper` to skip the download and use the old segmented path.

## 🚀 Quick Start

1. **Clone** (the benchmark suite is a submodule):
   ```bash
   git clone --recurse-submodules \
     https://github.com/raesoriano/Full-Duplex-Voice-Agent-WanDog.git
   cd Full-Duplex-Voice-Agent-WanDog
   ```
2. **Setup Keys:** Run `onit setup` to store API keys in the OS keychain.
3. **Environment:**
   ```bash
   python -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
4. **ASR model** (321 MB, one time — see Streaming ASR above).
   **TTS voice** (`en_US-ryan-medium`, 61 MB) needs no step: Piper downloads
   it into `~/.piper` the first time `agent.py` starts.
5. **Run:**
   ```bash
   .venv/bin/python agent.py
   ```

## 🧪 Tests

Run the suite with the venv interpreter; every file is standalone (`python test_x.py`).

| test | proves |
|---|---|
| `test_endtoend_answer.py` | a real pipeline turn produces `bot_text` — the regression test for the silent agent |
| `test_tap_forward.py` | the partial-transcript tap forwards frames instead of stalling the pipeline |
| `test_turn_context.py` | both aggregators share one `LLMContext`, so answers reach history |
| `test_stream_asr.py` | streaming partials arrive before speech ends |
| `test_pipeline_asr.py` | long questions (3.5 s, 5.6 s) still close their turn |
| `test_vad_shared_state.py` | the VAD gate matches its config — one analyzer per consumer, not one shared (sharing halves every gate) |
| `test_tool_result.py` | every tool handler settles its call via `result_callback`, including on error and timeout — the regression test for tools that run but are never spoken |
| `test_tool_parity.py` | the bridge exposes OnIt's full toolset (14 tools), compared against a fixed list |
| `test_tool_secrets.py` | tool-key resolution: file overrides keychain, deleting the file falls back to the keychain, blanks are omitted, YAML retyping is coerced (offline) |
| `test_tool_secrets_live.py` | the key actually reaches the MCP subprocess — calls `get_weather` end to end (network) |
| `test_bargein.py` / `test_guard.py` / `test_eventbus.py` | barge-in wiring, tool guardrails, dashboard events |
| `test_dashboard_ui.py` | the dashboard's controls write the right control files: Send writes the typed text and clears the box, Send becomes Stop while a turn is in flight, Stop writes the stop file, the mic toggle flips (8 checks, via Streamlit's AppTest) |
| `test_dashboard_controls.py` | the agent side of those controls: a typed message reaches the LLM context as a user turn, the control file is consumed so it is never answered twice, Stop broadcasts exactly one interruption, and the watcher really is upstream of the aggregator in `build_pipeline` |
| `test_session.py` | the session pointer and folder layout: continue by default, `--restart-session` mints a new one, the dashboard never mints, traversal in an id is refused, and `config` derives every path from the live session |
| `test_session_launch.py` | the same thing through four real launches of `agent.py`, asserting the folder, the log and the jail each time |
| `test_session_jail.py` | the per-call `data_path`: injected only on tools that declare it, the model's own value overwritten, and a real spawned server refusing a sibling session's file |
| `test_session_replay.py` | continuity and the handover notes: a continued session replays its turns, a restarted one replays nothing, a Clear truncates the replay, and compaction replaces the running lines with a summary |
| `test_upload.py` | uploads and the three tiers: size, type and traversal refusals, the strict prompt, the tier phrases, and `set_tools` really removing and restoring the web tools |
| `test_upload_e2e.py` | the join: the dashboard stores the file, the real watcher consumes the control file once, and the agent speaks the question rather than ingesting the file silently |

## 🖥️ Dashboard

A dark chat interface, modelled on the Ollama chat window, that drives the agent
by voice **or** by typing. It is a separate process:

```
WANDOG_DASHBOARD=1 .venv/bin/python agent.py          # terminal 1
WANDOG_DASHBOARD_LOG=/tmp/wandog_events.jsonl \
  .venv/bin/python -m streamlit run dashboard.py       # terminal 2
```

The two talk only through files next to the event log, so either can restart
without taking the other down:

| control | file | what the agent does |
|---|---|---|
| Mic toggle | `<log>.mic_control.json` | pushes `STTMuteFrame` to the ASR (barge-in still works) |
| Send (typed text) | `<log>.typed.json` | appends the message to the LLM context and runs inference — the reply is spoken like any other |
| Stop | `<log>.stop.json` | broadcasts an `InterruptionFrame`: the same path a barge-in takes, so it stops the LLM mid-generation and cuts the TTS |
| Clear | `<log>.clear.json` | `LLMMessagesUpdateFrame(messages=[])` with `run_llm=False`: the model forgets, nothing is spoken |
| Attach a file | `<log>.upload.json` | says which file it has and asks what may be used, then records the answer |

The composer is pinned to the bottom, with the attach control on the left, the
mic beside it, the text box in the middle, and Send — which becomes Stop
while a turn is in flight — on the right. Model selection is still a
disabled placeholder, kept so the layout is already final.

The mic is blue while the agent is accepting audio and monochrome while muted.

## 📁 Sessions

One session is one folder. The agent continues the previous session by default;
`--restart-session` starts a new one with an empty chat history.

```
.venv/bin/python agent.py                       # continue the last session
.venv/bin/python agent.py --restart-session     # start a new one
```

The root is `../wandog-sessions` relative to this checkout — beside the repo,
never inside it, so sessions are never committed and a `git clean` cannot delete
someone's work. `WANDOG_SESSIONS_ROOT` overrides it. The consequence worth
knowing: moving or re-cloning the repo does not move the sessions with it.

```
wandog-sessions/
  .current                    which session is live
  <id>/                       e.g. 2026-11-18T1620-a3f1
    meta.json                 id, start time, model, search mode, uploads
    events.jsonl              the log the dashboard tails
    notes/session.md          the handover record
    work/                     the folder the agent's tools are jailed to
      uploads/  created/  tool-results/
```

`notes/` is deliberately outside `work/`: the agent writes it in-process, so a
session's own file tools can neither rewrite its log nor edit its history.

Continuing a session replays the last 20 turns of `user_text`/`bot_text` into the
model's context, stopping at the last Clear — so "continue" means the
conversation continues, not just the folder. Tool traffic is not replayed.

The notes file is written twice over: one line per turn as the turn lands (so a
hard kill still leaves a record), then compacted into a summary on shutdown. That
summary is what a different session reads when asked what happened here.

Each session's tool calls are confined to its own `work/` folder by injecting
`data_path` per call, which OnIt's servers enforce as the boundary — so one
session cannot read a sibling's files.

## 📎 Uploading a file, and the search tiers

Attach a file from the composer and it is stored in the session's
`work/uploads/`. The agent does not ingest it silently: it says which file it has
and asks what it may use. The answer picks one of three tiers, narrowest first.
The order is enforced by which tools the model is given, not by asking it
nicely:

| tier | reaches | tools |
|---|---|---|
| `strict` | only the uploaded file | file tools only — no `local_search`, no web |
| `session` | + the rest of this session's files | + `local_search` (the default) |
| `expanded` | + the web | everything |

In `strict` the agent is told to say plainly that something is *not in* the file
rather than guessing, and to offer to widen the search. Saying "only that file",
"use the other files", or "yes, search the web" changes the tier and is recorded
in `meta.json`. The dashboard shows a caption whenever the tier is not the
default, because a limit you cannot see is indistinguishable from not knowing.

`tools/shot.py` screenshots the running page after it hydrates (plain
`chromium --screenshot` captures Streamlit's blank shell instead).

## 📚 References
- **Framework:** [Pipecat](https://github.com/pipecat-ai/pipecat)
- **TTS:** [Piper](https://github.com/rhasspy/piper)
- **VAD:** [Silero VAD](https://github.com/snakers4/silero-vad)
- **Internal Docs:** `WanDog-Notes.md` (Detailed development log)
