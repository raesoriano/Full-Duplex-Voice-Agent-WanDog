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
| LLM | **OnIt endpoint** (`http://202.92.159.240:8001/v1`) via `OpenAILLMService` | OpenAI-compatible; consumes **raw audio** input (no STT stage in v1) |
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
- **Key-safe** — LLM key from `ONIT_LLM_API_KEY` → OS keychain → `OPENAI_API_KEY`.

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

### No keyring backend in the venv
`keyring` reports "No recommended backend was available" in this environment,
so keychain lookup fails gracefully and the agent falls back to env vars. On the
Pi (with a keyring backend) the keychain path works.

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

## 7. Open questions / next steps
- [ ] **Live mic test on the Pi** — confirm capture card at 48 kHz, speaker out,
      and barge-in feel. (Can't be done in this sandbox — no real audio device.)
- [ ] Confirm the OnIt endpoint accepts **raw audio input** on the model we use;
      if it's text-only, add an ASR stage (faster-whisper / sherpa-onnx, already
      in requirements) between VAD and LLM.
- [ ] Add the rest of the pipeline: **ASR**, **MCP tools** (weather, web,
      smart-home), **dashboard**, **music**.
- [ ] Tune VAD thresholds (`confidence`, `stop_secs`) and TTS volume for the room.
- [ ] Decide on a wake word / "always listening" policy.
