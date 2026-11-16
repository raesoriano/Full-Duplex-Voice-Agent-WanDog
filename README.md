# Full-Duplex Voice Agent: Fweddie

Fweddie is a local, full-duplex voice command agent designed to run on resource-constrained hardware (e.g., Raspberry Pi 5). It provides a low-latency "listen-think-speak" loop with native support for barge-in (interruptions) and extensible tool use via MCP.

## 🏗️ System Architecture

Fweddie is built on the **Pipecat** framework, utilizing a linear pipeline that transforms raw audio into spoken responses.

### The Voice Loop
`Mic` $\rightarrow$ `VAD (Silero ONNX)` $\rightarrow$ `LLM (OnIt/vLLM)` $\rightarrow$ `TTS (Piper)` $\rightarrow$ `Speaker`

- **VAD (Voice Activity Detection):** Uses `SileroVADAnalyzer` (ONNX runtime) to detect speech without requiring heavy PyTorch dependencies.
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

## 🚀 Quick Start

1. **Setup Keys:** Run `onit setup` to store API keys in the OS keychain.
2. **Environment:**
   ```bash
   cd Fweddie-repo
   python -m venv .venv
   .venv/bin/pip install -r requirements.txt
   ```
3. **Run:**
   ```bash
   .venv/bin/python agent.py
   ```

## 📚 References
- **Framework:** [Pipecat](https://github.com/pipecat-ai/pipecat)
- **TTS:** [Piper](https://github.com/rhasspy/piper)
- **VAD:** [Silero VAD](https://github.com/snakers4/silero-vad)
- **Internal Docs:** `Fweddie-notes.md` (Detailed development log)
