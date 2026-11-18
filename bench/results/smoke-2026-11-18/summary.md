# Full-Duplex Voice Agent on a Raspberry Pi

Student wandog-smoke · run 2026-11-18_18-20-02 · 2026-11-18

## 5 · Evaluation & Benchmark

| Metric | Definition | Target | Mean | p95 | n |
|---|---|---|---|---|---|
| E2E latency | end of speech → audio out | < 4.00 s | 2.39 s | 2.39 s | 1 |
| ASR final | end of speech → transcript | < 0.80 s | 1.04 s | 1.04 s | 1 |
| LLM TTFT | transcript → first token | < 1.50 s | 0.67 s | 0.67 s | 1 |
| TTS TTFB | text → first audio chunk | < 0.60 s | 0.62 s | 0.62 s | 1 |
| Barge-in | speech → playback stopped | < 0.80 s | 0.48 s | 0.48 s | 2 |
| WER | ASR accuracy on test set | < 15 % | 0.0 % | — | 1 |

- One clock: the assistant machine's monotonic clock; every stage timestamped (VAD end → ASR final → LLM 1st token → TTS 1st audio → speaker)
- E2E = speaker − VAD end; barge-in = stop − speech start
- 1 phrases × 1 runs; mean + p95; JSON log per run: stage_latency.json (per turn: trials/<id>/stages.jsonl)

## Results (deck 2)

Latency chart: latency_chart.svg

- RPi model / RAM: Raspberry Pi 5 Model B Rev 1.0 · 8 GB
- Cooling: not stated · peak 68 °C · throttled never
- Network: cloud: Wi-Fi · benchmark link: same machine, 1 ms round trip
- Date: 2026-11-18
- Model versions: harness pipecat 1.15.0, ASR sherpa-onnx streaming zipformer (streaming), LLM gemma4:31b-cloud, TTS Piper en_US-ryan-medium, VAD Silero ONNX

## 3 · Context engineering: context size per run

| Context size | Median | Max | First → last call |
|---|---|---|---|
| Prompt tokens per LLM call (estimate) | 322 | 592 | 16 → 592 |
| Messages in context | 18 | 35 | 1 → 35 |
