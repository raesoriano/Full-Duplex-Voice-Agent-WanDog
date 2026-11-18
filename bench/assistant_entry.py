"""fd-benchmark entry point for WanDog.

fd-benchmark (trueshotjames/ai231-me3-full-duplex-benchmarking) starts an
assistant by importing an *entry* module that exposes ``PROJECT_MARKER``,
``audio_rates()``, optional ``describe()`` and an async ``build()``. Its
shipped entries target the course layout (``assistant/main.py`` with
``Assistant.build(transport=...)``) and Pipecat apps built inline.

WanDog is Pipecat, but its pipeline is constructed in ``agent.build_pipeline``,
which already takes an injectable transport pair (that hook exists for
``test_pipeline_asr.py``). This file is the whole adaptation: it hands
fd-benchmark's websocket transport to that hook instead of the local mic and
speaker. Nothing in agent.py, config.py or the pipeline changes.

Point the launcher at this file with FDBENCH_ENTRY:

    FDBENCH_ENTRY=bench/assistant_entry.py \
        .venv/bin/python fd-benchmark/launch_assistant.py --check

Two deliberate choices, both documented in bench/README.md:

* **Sample rates are forced to 16 kHz.** fdbench always speaks 16 kHz and
  refuses an input rate that is not 16000. WanDog's defaults are 48 kHz
  (``WANDOG_IN_RATE``/``WANDOG_OUT_RATE``) because its USB card accepts nothing
  else. In benchmark mode there is no card, so the capture rate is 16 kHz and
  the resampler becomes a no-op (16k -> ASR 16k). Pipecat resamples the TTS
  output to the transport's ``out_sr``, so 16 kHz out is safe too.

* **Tools are off.** The benchmark measures turn-taking and per-stage latency,
  not tool use. The OnIt MCP bridge costs ~5 s of startup per run and would add
  tool round-trips to the LLM TTFT it is trying to measure.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 1. A file that identifies the project folder (relative to it).
PROJECT_MARKER = "agent.py"


# 2. fdbench always sends 16 kHz mono and scores what comes back; both 16 kHz.
def audio_rates() -> tuple[int, int]:
    return 16000, 16000


# (optional) Model versions for the results card in the report.
def describe() -> dict:
    try:
        import config
        return {
            "ASR": f"sherpa-onnx streaming zipformer ({config.ASR_BACKEND})",
            "LLM": config.LLM_MODEL,
            "TTS": f"Piper {config.PIPER_VOICE}",
            "VAD": "Silero ONNX",
        }
    except Exception:  # noqa: BLE001  (never break a run over a label)
        return {}


async def build(transport, log_dir: Path):
    """Build WanDog's pipeline around fdbench's transport. Return (run, cancel)."""
    import config

    # --- Benchmark-mode overrides, set before build_pipeline reads them ------
    # capture rate 16 kHz: what fdbench sends. proc_rate is already 16 kHz
    # (ASR_SAMPLE_RATE) when ASR is on, so the resampler is 16k -> 16k.
    config.INPUT_SAMPLE_RATE = 16000
    config.OUTPUT_SAMPLE_RATE = 16000

    import agent
    import eventbus

    bus = eventbus.BUS

    # A fresh bus log per run, so the dashboard's JSONL does not mix benchmark
    # turns with real sessions.
    log_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(bus, "path"):
        try:
            bus.path = log_dir / "events.jsonl"
        except Exception:  # noqa: BLE001
            pass

    api_key = config.resolve_llm_api_key()

    # bridge=None: tools off (see module docstring). The transport pair is the
    # same hook test_pipeline_asr.py uses, so the chain is WanDog's real one --
    # VAD, streaming ASR, user aggregator, LLM, Piper TTS -- with only the
    # mic/speaker replaced.
    pipeline, task, input_transport, output_transport = agent.build_pipeline(
        api_key, None, bus, transports=(transport.input(), transport.output()))

    from pipecat.pipeline.runner import WorkerRunner

    runner = WorkerRunner()
    await runner.add_workers(task)

    async def run():
        try:
            # auto_end=False: stay alive until fdbench closes the socket.
            await runner.run(auto_end=False)
        finally:
            await runner.cancel()

    async def cancel():
        await runner.cancel()

    return run(), cancel
