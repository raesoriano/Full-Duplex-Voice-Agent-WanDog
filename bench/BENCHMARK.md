# BENCHMARK — WanDog measured against Full-Duplex-Bench

Run `2026-11-18_18-27-10` · Raspberry Pi 5 (8 GB) · 2026-11-18
Harness: [fd-benchmark](https://github.com/trueshotjames/ai231-me3-full-duplex-benchmarking)
@ `57eaddbc`, adapted via `bench/assistant_entry.py`.

Models under test: **ASR** sherpa-onnx streaming zipformer · **LLM**
gemma4:31b-cloud (Ollama cloud, Wi-Fi) · **TTS** Piper `en_US-ryan-medium` ·
**VAD** Silero ONNX. Tools off. See `README.md` for why.

Raw files: `results/full-2026-11-18/` (40 trials, 30 stage-latency turns).
`results/smoke-2026-11-18/` is the 15-trial dev-dataset run used to validate the
adapter — its numbers mean nothing and are kept only as a smoke record.

## Latency

| Metric | Definition | Target | Mean | p95 | n | |
|---|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.00 s | 2.07 s | 2.70 s | 30 | met |
| ASR final | end of speech → transcript | < 0.80 s | 0.96 s | 1.01 s | 30 | **over** |
| LLM TTFT | transcript → first token | < 1.50 s | 0.63 s | 1.17 s | 30 | met |
| TTS TTFB | text → first audio chunk | < 0.60 s | 0.40 s | 0.56 s | 30 | met |
| Barge-in | speech → playback stopped | < 0.80 s | 0.48 s | 0.48 s | 2 | met |
| WER | ASR accuracy on test set | < 15 % | 1.4 % | — | 30 | met |

E2E passes with 1.9 s of headroom, but the **breakdown** is the useful part:

| Stage | Mean | Share of E2E |
|---|---|---|
| ASR final | 0.956 s | 46 % |
| LLM TTFT | 0.633 s | 31 % |
| TTS TTFB | 0.396 s | 19 % |
| hand-offs + playout | 0.080 s | 4 % |
| **E2E** | **2.065 s** | |

**ASR is the bottleneck and the only missed target** — 0.96 s against a 0.80 s
budget, and 46 % of end-to-end. The LLM and TTS both have comfortable margin.
Hand-offs and playout are 80 ms total, so the pipeline itself is not the
problem; the fix is in the streaming ASR's final pass, not in the plumbing.

`course.e2e_heard` is also reported: **2.71 s** (p95 3.38 s). That is the
stricter measure, timed from the *true* end of speech rather than VAD's idea of
it, and the 0.65 s gap between the two is the endpointer's own latency
(`course.endpointing` = 0.652 s).

## Turn-taking — the finding that matters

The v1.5 groups test whether the assistant stays quiet when audio is *not*
addressed to it. It does not.

| Group | What the audio is | Expected | Result |
|---|---|---|---|
| `false_tt` · TV/radio | "Actually, can you tell me about sculpting?" | stay silent | **took turn**, 3.4 s |
| `false_tt` · side talk | "Chris, your laptop's still on the table, man." | stay silent | **took turn**, 3.7 s |

**False turn-taking rate: 100 % (2 of 2).** Missed turn-taking rate is 0 % —
it never fails to answer when addressed (`v1_turntaking.tor` 1.0,
`v1_interrupt.tor` 1.0, `v1_pause_*.tor` 0.0, i.e. it correctly does *not*
answer into a mid-sentence pause).

So the failure is one-sided and specific: **WanDog answers speech that is not
meant for it.** It has no notion of address. This is exactly what a
VAD-gated turn detector cannot do — Silero answers "is this speech?", not "is
this speech for me?", and with tools off there is no wake word either.

Caveat on the sample: n=2 for `false_tt` here (one side-talk clip, one TV/radio
clip). The rate is 100 % of two, not of forty. It is a clear signal and a real
defect, but the confidence interval is wide. Re-running with the full 40-clip
`false_tt` group is the first thing to do next.

The barge-in numbers are thin for the same reason — n=2. `v15_backchannel`
(0.51 s to stop, 2.34 s to resume) and `v15_talking_to_other` (0.54 s / 2.47 s)
each ran a single clip.

## Resources

| Metric | Value | n |
|---|---|---|
| Peak CPU temperature | 68.9 °C | 794 |
| Throttled share | 0.00 | 794 |
| CPU median / p95 | 6.9 % / 67.7 % | 794 |
| Busiest core p95 | 72.3 % | 794 |
| Min clock | 1500 MHz | 794 |
| Min available RAM | 5904 MB | 794 |

No throttling, 69 °C peak, and a low median CPU with 68 % spikes — the Pi is
not saturated, which is consistent with the LLM being a cloud call and the
remaining local work being ASR + Piper.

## Context growth

| Metric | Median | Max |
|---|---|---|
| Prompt tokens per LLM call (est.) | 879 | 1674 |
| Messages in context | 48 | 94 |

Context is **never trimmed across a 40-trial session** and doubles between the
first and last call (7 → 1674 tokens). Nothing in the run hit a limit, but this
is unbounded growth in a long-lived session.

## What is not measured

- **Tool use.** `course.tool_calls` is empty because tools were off by design.
- **Reply quality.** No judge configured, and `--no-asr` stops transcription of
  the assistant's own audio, so the v1.5 `respond`/`resume` labels and the voice
  metrics (WPM, pitch, intensity, UTMOS) are all empty. WER is unaffected — it
  compares the clip transcript against the assistant's ASR and never needs the
  assistant's audio transcribed.
- **Acoustic echo.** Both mic and speaker are replaced by a socket, so AEC is
  not exercised. A barge-in number here says nothing about whether WanDog can
  hear you over its own speaker.

## Reproducing

```bash
bench/run_bench.sh full      # same command that produced this run
```

Seeded clip order (seed `450687`), checksum-verified datasets, pinned submodule
(`57eaddbc`), fdbench 0.1.0. The wall clock was ~19 minutes for 40 trials
including 30 stage-latency turns.

`session.json` in `results/full-2026-11-18/` records the machine and toolchain:
Linux 6.18.50+rpt-rpi-2712, aarch64, Python 3.12.16, 7.9 GB RAM, Wi-Fi,
`throttled_ever: false`.
