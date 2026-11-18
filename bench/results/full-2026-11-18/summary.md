# Full-Duplex Voice Agent on a Raspberry Pi

Student wandog · run 2026-11-18_18-27-10 · 2026-11-18

## 5 · Evaluation & Benchmark

| Metric | Definition | Target | Mean | p95 | n |
|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.00 s | 2.07 s | 2.70 s | 30 |
| ASR final | end of speech → transcript | < 0.80 s | 0.96 s | 1.01 s | 30 |
| LLM TTFT | transcript → first token | < 1.50 s | 0.63 s | 1.17 s | 30 |
| TTS TTFB | text → first audio chunk | < 0.60 s | 0.40 s | 0.56 s | 30 |
| Barge-in | speech → playback stopped | < 0.80 s | 0.48 s | 0.48 s | 2 |
| WER | ASR accuracy on test set | < 15 % | 1.4 % | — | 30 |

- One clock: the assistant machine's monotonic clock; every stage timestamped (VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker)
- E2E = speaker − VAD end; barge-in = stop − speech start
- 10 phrases × 3 runs; mean + p95; JSON log per run: stage_latency.json (per turn: trials/<id>/stages.jsonl)

## Results (deck 2)

Latency chart: latency_chart.svg

- RPi model / RAM: Raspberry Pi 5 Model B Rev 1.0 · 8 GB
- Cooling: not stated · peak 69 °C · throttled never
- Network: cloud: Wi-Fi · benchmark link: same machine, 2 ms round trip
- Date: 2026-11-18
- Model versions: harness pipecat 1.15.0, ASR sherpa-onnx streaming zipformer (streaming), LLM gemma4:31b-cloud, TTS Piper en_US-ryan-medium, VAD Silero ONNX

## 3 · Context engineering: context size per run

| Context size | Median | Max | First → last call |
|---|---|---|---|
| Prompt tokens per LLM call (estimate) | 879 | 1674 | 7 → 1674 |
| Messages in context | 48 | 94 | 1 → 94 |
