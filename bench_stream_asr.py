"""Measure streaming ASR latency: sherpa-onnx zipformer vs faster-whisper.

For each test wav we feed 20 ms chunks (like the mic) and record:
  * time to first partial word
  * time from last chunk of speech -> each new partial
  * time from speech end -> the final transcript (endpoint or flush)

This is the number the turn stop strategy cares about. Whisper's number was
~5.4 s/segment (segmented: nothing at all until the utterance ends).
"""
from __future__ import annotations

import time
import wave

import numpy as np
import sherpa_onnx

MODEL_DIR = "models/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
CHUNK = 320  # 20 ms at 16 kHz


def load16k(path):
    w = wave.open(path)
    sr = w.getframerate()
    a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if sr != 16000:
        idx = np.arange(int(len(a) * 16000 / sr)) * sr / 16000
        a = np.interp(idx, np.arange(len(a)), a).astype(np.float32)
    return a


def make_recognizer():
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=f"{MODEL_DIR}/tokens.txt",
        encoder=f"{MODEL_DIR}/encoder-epoch-99-avg-1.int8.onnx",
        decoder=f"{MODEL_DIR}/decoder-epoch-99-avg-1.onnx",
        joiner=f"{MODEL_DIR}/joiner-epoch-99-avg-1.int8.onnx",
        num_threads=2,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        enable_endpoint_detection=True,
        rule1_min_trailing_silence=2.4,
        rule2_min_trailing_silence=1.2,
        rule3_min_utterance_length=20.0,
    )


def run(path):
    rec = make_recognizer()
    stream = rec.create_stream()
    audio = load16k(path)
    print(f"\n=== {path}  {len(audio)/16000:.2f}s ===")
    t0 = time.perf_counter()
    first_partial = None
    last_text = ""
    t_end_speech = None
    finals = []
    n = len(audio)
    i = 0
    while i < n:
        chunk = audio[i : i + CHUNK]
        i += CHUNK
        stream.accept_waveform(16000, chunk)
        while rec.is_ready(stream):
            rec.decode_stream(stream)
        txt = rec.get_result(stream)
        now = time.perf_counter() - t0
        if txt and txt != last_text:
            if first_partial is None:
                first_partial = now
                print(f"  first partial @ {now:5.2f}s  {txt!r}")
            else:
                print(f"  partial       @ {now:5.2f}s  {txt!r}")
            last_text = txt
        if rec.is_endpoint(stream):
            finals.append((now, txt))
            print(f"  ENDPOINT      @ {now:5.2f}s  {txt!r}")
            rec.reset(stream)
            last_text = ""
        # pace it like a real mic (20 ms of audio per 20 ms of wall clock)
        target = t0 + (i / 16000.0)
        sleep = target - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
    t_end_speech = time.perf_counter() - t0
    stream.input_finished()
    while rec.is_ready(stream):
        rec.decode_stream(stream)
    final = rec.get_result(stream)
    t_final = time.perf_counter() - t0
    print(f"  speech ended  @ {t_end_speech:5.2f}s")
    print(f"  FINAL         @ {t_final:5.2f}s  {final!r}  (tail {t_final - t_end_speech:.2f}s)")
    return first_partial, t_end_speech, t_final - t_end_speech, final


if __name__ == "__main__":
    for p in ["/tmp/q_math.wav", "/tmp/q_dog.wav", f"{MODEL_DIR}/test_wavs/0.wav"]:
        try:
            run(p)
        except Exception as e:
            print(f"  FAILED {p}: {e}")
