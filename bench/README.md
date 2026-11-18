# bench/ — the full-duplex benchmark, adapted to WanDog

`fd-benchmark/` is a pinned git submodule of
[trueshotjames/ai231-me3-full-duplex-benchmarking](https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking)
— the course's black-box benchmark for full-duplex voice assistants, adapted
from Full-Duplex-Bench v1 (Lin et al., 2025) and v1.5 (Lin et al., 2026). It
feeds the papers' test clips to an assistant, records what the assistant says,
and scores turn-taking and per-stage latency.

This folder is the **adaptation**: everything needed to point that harness at
WanDog, kept outside the submodule so the pinned upstream stays clean.

| file | what it is |
|---|---|
| `assistant_entry.py` | the fdbench entry module — the whole adaptation |
| `run_bench.sh` | `smoke` / `full` / `report` |
| `results/` | committed copies of the runs that matter |
| `BENCHMARK.md` | what the numbers mean, and the measured results |

## Why an entry module and not a patch

fdbench starts an assistant by importing an *entry* module that exposes
`PROJECT_MARKER`, `audio_rates()`, `describe()` and an async `build()`. It ships
two: `adapters/entry_course.py` for the course layout (`assistant/main.py` with
`Assistant.build(transport=...)`) and `adapters/assistant_entry_template.py` for
Pipecat apps that build their transport inline.

WanDog is neither. It is Pipecat 1.15, but its pipeline is assembled in
`agent.build_pipeline()`, which already accepts an injectable transport pair —
a hook that exists for `test_pipeline_asr.py`. So the adapter is short: hand
fdbench's websocket transport to that hook. **`agent.py` and `config.py` are not
modified**, and `git diff main -- agent.py config.py` is empty on this branch.

## Running it

```bash
bench/run_bench.sh smoke     # dev dataset, 1 round    (~7 min)
bench/run_bench.sh full      # 280-clip dataset, 1 round
bench/run_bench.sh report    # print the last full run
```

Prerequisites, all one-time:

```bash
# the submodule and its datasets
git submodule update --init fd-benchmark
.venv/bin/pip install "scipy>=1.11"          # only core dep not already present
.venv/bin/python fd-benchmark/tools/fetch_datasets.py --force   # 30 MB, 280 clips
```

The run needs, and this machine has: the sherpa-onnx streaming zipformer in
`models/`, a Piper voice in `~/.piper`, and a working `ONIT_LLM_*` key.

## The three decisions that matter

**16 kHz, not 48 kHz.** fdbench always speaks 16 kHz and refuses an input rate
that is not 16000. WanDog's defaults are 48 kHz because its USB card accepts
nothing else, so `build()` sets `INPUT_SAMPLE_RATE`/`OUTPUT_SAMPLE_RATE` to
16000 before `build_pipeline` reads them. In benchmark mode there is no card:
the resampler becomes 16k → 16k (a no-op), and Pipecat resamples Piper's output
down to the transport's `out_sr`.

**Tools off.** `build_pipeline(api_key, None, bus, ...)` — no MCP bridge. The
benchmark measures turn-taking and stage latency, not tool use; the bridge costs
~5 s of startup per run and would put tool round-trips inside the LLM TTFT it is
trying to measure. This is a deliberate scope limit: **these numbers describe
WanDog's conversational loop, not its tool use.**

**`--no-asr` on scoring.** fdbench's scorer has two independent switches that
are easy to confuse:

- `--no-asr` stops it transcribing **the assistant's own output**
  (`scorer.py:69`, `_asr_for` returns `None`). It is what turns off the v1.5
  behaviour labels and the voice metrics (WPM, pitch, intensity, UTMOS).
- **WER is unaffected.** WER comes from the stage-latency phrases, where the
  clip's known transcript is compared against the assistant's ASR — it never
  needs the assistant's audio transcribed. That is why a run with `--no-asr`
  still reports WER 1.4 % over n=30.

So this run measures timing, turn-taking and ASR accuracy, and does **not**
measure whether the assistant's spoken replies were sensible or well-delivered.
No judge is configured either, which is a second, separate reason the v1.5
`respond`/`resume` labels come back empty.

## What is *not* measured

- **Tool use.** Tools are off, as above.
- **Interruption relevance.** No judge model is configured, so fdbench records
  `content: judge` and scores only *whether* the assistant stopped, not whether
  the reply made sense.
- **The dashboard.** `--assistant-cmd` starts the agent directly, so no
  Streamlit process runs during a benchmark.
- **A real microphone or speaker.** Both are replaced by the socket. This is the
  point of the harness, but it means acoustic echo cancellation is not exercised.

## Reproducing

`fd-benchmark` is pinned at `57eaddbc`. The datasets are a GitHub Release
(`datasets-v2`), checksum-verified by the pre-flight on every run. Runs are
seeded (`--seed`), so the clip order is reproducible.
