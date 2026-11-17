# Full-Duplex Voice Agent: Fweddie

Fweddie is a local, full-duplex voice command agent designed to run on resource-constrained hardware (e.g., Raspberry Pi 5). It provides a low-latency "listen-think-speak" loop with native support for barge-in (interruptions) and extensible tool use via MCP.

> **Status (2026-11-17):** working on the Pi. Streaming partials appear while you
> speak, the turn closes ~0.9 s after you stop, and the reply is about the
> question just asked. Commit `f4836c85`, verified from a fresh clone.

## 🏗️ System Architecture

Fweddie is built on the **Pipecat** framework, utilizing a linear pipeline that transforms raw audio into spoken responses.

### The Voice Loop
`Mic` $\rightarrow$ `VAD (Silero ONNX)` $\rightarrow$ `ASR (streaming zipformer + whisper final pass)` $\rightarrow$ `LLM (OnIt/vLLM)` $\rightarrow$ `TTS (Piper)` $\rightarrow$ `Speaker`

- **VAD (Voice Activity Detection):** Uses `SileroVADAnalyzer` (ONNX runtime) to detect speech without requiring heavy PyTorch dependencies.
- **ASR (Speech-to-Text):** Streams by default, so text appears while you are
  still talking. See the Streaming ASR section below.
- **LLM:** Connects to an OpenAI-compatible endpoint (typically a DGX cluster running vLLM) using `OpenAILLMService`.
- **TTS (Text-to-Speech):** Uses **Piper**, a fast, local neural TTS that generates high-quality speech in real-time on CPU.
- **Barge-in:** Implemented via `LLMUserAggregator` and `VADUserTurnStartStrategy`. When the VAD detects the user speaking while the bot is talking, an `InterruptionFrame` is broadcast to immediately silence the TTS.

## 🧠 Context & Prompt Engineering

### System Prompt
Fweddie's persona is defined in `config.py`. It is designed for **spoken** interaction:
- **Persona:** Friendly, upbeat, and concise.
- **Constraints:** Replies are limited to 1-2 short sentences. No markdown, bullet points, or emojis.
- **Tool Awareness:** When MCP tools are enabled, a secondary prompt is appended, instructing the model on how to call tools, handle approvals, and report refusals without "routing around" them.

### Context Management
- **Input:** The agent consumes audio/text turns.
- **State:** Managed by Pipecat's pipeline state.
- **Tool Integration:** Tool results are fed back into the LLM context, which then generates a concise spoken summary for the user.

## 🛠️ MCP Services (Model Context Protocol)

Fweddie extends its capabilities through a bridge to **OnIt's MCP servers**, allowing the voice agent to interact with the local system and the web.

| Server | Profile | Capabilities |
|---|---|---|
| `ToolsLocalMCPServer` | `local` | Bash shell execution, file system R/W, local document search |
| `ToolsNetMCPServer` | `net` | Web search (Ollama/DuckDuckGo), weather updates |

**Security & Guardrails:**
- **Jailroot:** File tools are restricted to a specific `MCP_DATA_PATH` (e.g., `/tmp/onit/data`).
- **Human-in-the-Loop:** Critical commands trigger an approval request. The agent notifies the user it is "asking a person" and waits for a decision via the dashboard or API.

## 📈 Evaluation & Benchmarking

To ensure "full-duplex" performance, Fweddie is evaluated using the `fd-benchmark` suite.

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

Both are configurable: `FWEDDIE_ASR_BACKEND` (`streaming` | `whisper`), `FWEDDIE_ASR_FINAL` (`whisper` | `none`), `FWEDDIE_ASR_FINAL_MODEL`, `FWEDDIE_TURN_STOP`.

**The model weights (321 MB) are not in the repo** — `models/` is gitignored. Download once per machine:

```bash
mkdir -p models && cd models
curl -LO https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-en-2023-06-26.tar.bz2
```

The service names this exact command in its error message if the directory is missing. Set `FWEDDIE_ASR_BACKEND=whisper` to skip the download and use the old segmented path.

## 🚀 Quick Start

1. **Clone** (the benchmark suite is a submodule):
   ```bash
   git clone --recurse-submodules \
     https://github.com/raesoriano/Full-Duplex-Voice-Agent-Fweddie.git
   cd Full-Duplex-Voice-Agent-Fweddie
   ```
2. **Setup Keys:** Run `onit setup` to store API keys in the OS keychain.
3. **Environment:**
   ```bash
   python -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
4. **ASR model** (321 MB, one time — see Streaming ASR above).
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
| `test_bargein.py` / `test_guard.py` / `test_eventbus.py` | barge-in wiring, tool guardrails, dashboard events |

## 📚 References
- **Framework:** [Pipecat](https://github.com/pipecat-ai/pipecat)
- **TTS:** [Piper](https://github.com/rhasspy/piper)
- **VAD:** [Silero VAD](https://github.com/snakers4/silero-vad)
- **Internal Docs:** `Fweddie-notes.md` (Detailed development log)
