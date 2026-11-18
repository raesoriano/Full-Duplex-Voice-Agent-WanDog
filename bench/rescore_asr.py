#!/usr/bin/env python3
"""Recompute the ASR stage of a finished fd-benchmark run. No rerun, no audio.

The question
------------
fd-benchmark reports one number for the ASR stage:

    asr_final = asr_final_mark.t - last_vad_end_mark.t

both on the assistant's monotonic clock (fdbench/score/stages.py:74-99). The
harness describes that number two different ways:

    report/files.py:32   "ASR final (VAD end -> transcript)"
    report/deck.py:13    "ASR final", "end of speech -> transcript"

The deck wording is the one that reaches a reader, and it is not what is
measured. `vad_end` is the assistant's own Silero crossing its stop threshold,
which trails the acoustic end of speech by the endpointing delay. So the
reported figure excludes endpointing, and is not "speech end -> transcript".

This script recomputes the stage under every definition worth naming, from the
marks the run already wrote. It also re-derives the shipped number through
fdbench's own `turn_stages`, so a disagreement with the original report shows
up as a diff instead of hiding.

Definitions
-----------
  A  vad_end        -> asr_final mark      the shipped number
  B  true speech end -> asr_final mark      A + endpointing; the deck's wording
  C  true speech end -> last asr_partial    when the text first existed
  D  vad_end        -> last asr_partial     negative = text beat the VAD
  E  whisper pass alone (assistant.log)     the second model's wall time

A and B are recoverable from the committed stage_latency.json (it carries
endpointing and asr_final per turn). C, D and E need the raw marks under
fd-benchmark/runs/, which the submodule git-ignores.

Usage
-----
    .venv/bin/python bench/rescore_asr.py fd-benchmark/runs/wandog/<run-id>
    .venv/bin/python bench/rescore_asr.py --from-committed bench/results/full-2026-11-18
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
FDBENCH = REPO / "fd-benchmark"
TARGET_ASR = 0.8          # fdbench/targets.py:14
TARGET_WER = 0.15


# --------------------------------------------------------------------------- io

def _read_jsonl(p: Path) -> list:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def _stage_latency_trials(run_dir: Path) -> list:
    """(trial name, timing.json, marks sorted by t) for each stage_latency trial."""
    rows = []
    for tp in sorted((run_dir / "trials").glob("*/timing.json")):
        t = json.loads(tp.read_text(encoding="utf-8"))
        if t.get("group") != "stage_latency":
            continue
        ms = sorted(_read_jsonl(tp.parent / "stages.jsonl"), key=lambda m: float(m["t"]))
        rows.append((t["trial"], t, ms))
    return rows


# ------------------------------------------------------------------- statistics

def _stats(vals) -> dict:
    """Delegate to fdbench's own latency(), so p95 here means what p95 there means."""
    v = [x for x in vals if x is not None]
    sys.path.insert(0, str(FDBENCH))
    try:
        from fdbench.score import metrics
        out = metrics.latency(v)
    except Exception:  # noqa: BLE001
        if not v:
            return {"n": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
        out = {"n": len(v), "mean": statistics.mean(v), "median": statistics.median(v),
               "p95": sorted(v)[min(len(v) - 1, int(round(0.95 * len(v))) - 1)],
               "min": min(v), "max": max(v)}
    return {k: (round(out[k], 4) if isinstance(out.get(k), float) else out.get(k))
            for k in ("n", "mean", "median", "p95", "min", "max")}


def _verdict(mean, target: float = TARGET_ASR) -> str:
    if mean is None:
        return "-"
    return "met" if mean <= target else "over"


def _mean(vals):
    v = [x for x in vals if x is not None]
    return round(statistics.mean(v), 4) if v else None


# --------------------------------------------------------------------- variants

def _last_partial(ms: list, before_t: float | None = None):
    """Last interim transcript mark, optionally before a given monotonic time."""
    hit = None
    for m in ms:
        if m.get("stage") != "asr_partial":
            continue
        if before_t is not None and float(m["t"]) > before_t:
            continue
        hit = m
    return hit


def _log_passes(run_dir: Path) -> list:
    """Every (streaming, final, compute_s, audio_s) the assistant logged, in order.

    faster-whisper logs "Processing audio with duration H:MM:SS.sss" when the
    transcribe call starts and streaming_asr logs the streaming/final pair when
    it returns; the delta is the second pass alone. That is a second, independent
    clock (the log's wall clock) against the marks' monotonic one.
    """
    log = run_dir / "assistant.log"
    if not log.exists():
        return []
    stamp = re.compile(r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)")
    dur_re = re.compile(r"Processing audio with duration (\d+):(\d+\.\d+)")
    pair_re = re.compile(r"Streaming ASR: streaming=(.*) final=(.*)$")

    def unq(x: str) -> str:
        x = x.strip()
        if len(x) > 1 and x[0] == x[-1] and x[0] in "'\"":
            x = x[1:-1]
        return x

    out, started, audio, last = [], None, None, None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        m = stamp.match(line)
        if m:
            last = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f")
        # faster-whisper's lines carry no prefix of their own: use the last stamp
        m = dur_re.search(line)
        if m:
            started = last
            audio = float(m.group(1)) * 60 + float(m.group(2))
            continue
        m = pair_re.search(line)
        if m and started is not None:
            out.append({
                "streaming": unq(m.group(1)),
                "final": unq(m.group(2)),
                "compute_s": round((last - started).total_seconds(), 4),
                "audio_s": round(audio, 3),
            })
            started = None
    return out


def _match_passes(keys: list, passes: list) -> list:
    """Greedy unique match: turn's last partial text -> the log line that produced it.

    Exact lower-case match wins; otherwise containment either way. Each log line
    is used at most once, in order, so a repeated phrase cannot consume the same
    line twice. Returns one pass dict (or None) per key.
    """
    used, out = set(), []
    for key in keys:
        k = (key or "").lower().strip().rstrip(".")
        hit = None
        if k:
            for i, p in enumerate(passes):
                if i not in used and p["streaming"].lower().strip().rstrip(".") == k:
                    hit = i
                    break
            if hit is None:
                for i, p in enumerate(passes):
                    if i in used:
                        continue
                    s = p["streaming"].lower()
                    if k in s or s in k:
                        hit = i
                        break
        if hit is None:
            out.append(None)
        else:
            used.add(hit)
            out.append(passes[hit])
    return out


def _wer(pairs) -> dict:
    """Corpus WER over (reference, hypothesis) pairs, via fdbench's own scorer."""
    sys.path.insert(0, str(FDBENCH))
    try:
        from fdbench.score import wer as W
    except Exception:  # noqa: BLE001
        return {"wer": None, "n": 0}
    items = [W.wer(ref, hyp) for ref, hyp in pairs if ref]
    n_ref = sum(i["ref_words"] for i in items)
    n_err = sum(i["edits"] for i in items)
    return {"wer": (round(n_err / n_ref, 4) if n_ref else None), "n": len(items),
            "ref_words": n_ref, "edits": n_err}


# ------------------------------------------------------------------------- main

def rescore(run_dir: Path) -> dict:
    sys.path.insert(0, str(FDBENCH))
    from fdbench.score import stages as S

    trials = _stage_latency_trials(run_dir)
    if not trials:
        raise SystemExit(f"no stage_latency trials under {run_dir}/trials")

    passes = _log_passes(run_dir)
    rows = []
    for name, t, ms in trials:
        ref = t.get("clip_meta", {}).get("text", "")
        st = S.turn_stages(ms, t.get("user_end"), t.get("first_start"), ref)
        user_end = t.get("user_end")

        # Pick the same marks turn_stages picks, so the recomputation is
        # comparable to the shipped one rather than to a private reading.
        reply = next((m for m in ms if m.get("stage") == "audio_out"
                      and m.get("tt", 0) > (user_end or 0.0) - 0.2), None)
        t_reply = reply["t"] if reply else None
        finals = [m for m in ms if m.get("stage") == "asr_final" and m.get("tt", 0) >= -0.5
                  and (t_reply is None or m["t"] < t_reply)]
        asr = finals[-1] if finals else None
        vads = [m for m in ms if m.get("stage") == "vad_end" and (t_reply is None or m["t"] < t_reply)]
        vad = vads[-1] if vads else None
        part = _last_partial(ms, before_t=(asr["t"] if asr else None))

        a = st.get("asr_final")                      # shipped: vad_end -> final mark
        ep = st.get("endpointing")                   # true end -> vad_end
        rows.append({
            "trial": name,
            "ref": ref,
            "A_vad_end_to_mark": a,
            "endpointing": ep,
            "B_speech_end_to_mark": round(ep + a, 4) if (a is not None and ep is not None) else None,
            "C_speech_end_to_partial": (round(float(part["tt"]) - user_end, 4)
                                        if (part and user_end is not None) else None),
            "D_vad_end_to_partial": (round(float(part["t"]) - float(vad["t"]), 4)
                                     if (part and vad) else None),
            "llm_ttft": st.get("llm_ttft"),
            "tts_ttfb": st.get("tts_ttfb"),
            "e2e": st.get("e2e"),
            "e2e_heard": st.get("e2e_heard"),
            "partial_text": (part.get("text", "") if part else ""),
            "final_text": st.get("transcript", ""),
            "whisper_s": None,
            "streaming_flushed": None,
        })

    # Attach the log-derived numbers by matching each turn's last partial.
    matched = _match_passes([r["partial_text"] for r in rows], passes)
    for r, p in zip(rows, matched):
        if p:
            r["whisper_s"] = p["compute_s"]
            r["audio_s"] = p["audio_s"]
            r["streaming_flushed"] = p["streaming"]

    def col(k):
        return _stats([r[k] for r in rows])

    variants = {
        "A_vad_end_to_mark": dict(label="A  VAD end -> transcript (as shipped)",
                                  **col("A_vad_end_to_mark")),
        "B_speech_end_to_mark": dict(label="B  true speech end -> transcript (deck wording)",
                                     **col("B_speech_end_to_mark")),
        "C_speech_end_to_partial": dict(label="C  true speech end -> last streaming partial",
                                        **col("C_speech_end_to_partial")),
        "D_vad_end_to_partial": dict(label="D  VAD end -> last streaming partial",
                                     **col("D_vad_end_to_partial")),
        "E_whisper_pass": dict(label="E  whisper second pass alone (log clock)",
                               **col("whisper_s")),
    }
    for v in variants.values():
        v["verdict"] = _verdict(v["mean"])

    # Does this script reproduce what was reported?
    original, orig_mean = None, None
    for cand in (run_dir / "stage_latency.json",
                 run_dir.parent / "stage_latency.json",          # raw/ copied next to the results
                 REPO / "bench" / "results" / run_dir.name / "stage_latency.json"):
        if cand.exists():
            try:
                original = json.loads(cand.read_text(encoding="utf-8"))
                break
            except ValueError:
                pass
    if original:
        orig_mean = _mean([r.get("asr_final") for r in original.get("turns", [])])

    # Decomposition of the heard end-to-end, all measured, no assumption.
    handoff = [round(r["e2e"] - (r["A_vad_end_to_mark"] or 0) - (r["llm_ttft"] or 0)
                     - (r["tts_ttfb"] or 0), 4) for r in rows]
    decomp = {
        "endpointing_true_end_to_vad": _mean([r["endpointing"] for r in rows]),
        "asr_vad_end_to_transcript": _mean([r["A_vad_end_to_mark"] for r in rows]),
        "llm_ttft": _mean([r["llm_ttft"] for r in rows]),
        "tts_ttfb": _mean([r["tts_ttfb"] for r in rows]),
        "handoff_and_playout": _mean(handoff),
        "e2e_heard_speech_end_to_reply_audio": _mean([r["e2e_heard"] for r in rows]),
    }
    # Where the transcript could first exist, on the assistant's own clock.
    # The streaming text is flushed by _end_segment(), which runs on
    # VADUserStoppedSpeakingFrame -- so even though the last interim arrives
    # BEFORE that frame (D is negative), the aggregator cannot hand the turn to
    # the LLM until vad_end. So the honest floor is: transcript at vad_end.
    streaming_only = {
        "note": ("How much of the reported ASR stage is the streaming recogniser and how much is the "
                 "optional second pass. D < 0 means the streaming text was already complete before the "
                 "VAD declared the turn over, so the recogniser contributes no stage latency at all."),
        "last_partial_relative_to_vad_end": _stats([r["D_vad_end_to_partial"] for r in rows]),
        "whisper_second_pass": _stats([r["whisper_s"] for r in rows]),
        "reported_stage_A": _stats([r["A_vad_end_to_mark"] for r in rows]),
    }

    # Counterfactual: ASR_FINAL_PASS=none. Transcript ready at vad_end, so the
    # ASR term of the heard end-to-end becomes zero and endpointing still applies.
    cf = []
    for r in rows:
        parts = (r["endpointing"], r["llm_ttft"], r["tts_ttfb"], r["A_vad_end_to_mark"], r["e2e"])
        if any(x is None for x in parts):
            continue
        handoff_r = r["e2e"] - r["A_vad_end_to_mark"] - r["llm_ttft"] - r["tts_ttfb"]
        cf.append(r["endpointing"] + r["llm_ttft"] + r["tts_ttfb"] + handoff_r)
    counterfactual = {
        "note": ("If ASR_FINAL_PASS were 'none' (answer on the streaming text): true speech end -> "
                 "reply audio = endpointing + TTFT + TTFB + handoff, per turn. The ASR term goes to zero "
                 "because the streaming text is already complete at vad_end; endpointing still applies, "
                 "since the aggregator triggers on the VAD stop. Arithmetic on measured values, not a run."),
        "e2e_heard_no_final_pass": _stats(cf),
        "measured_e2e_heard": _stats([r["e2e_heard"] for r in rows]),
        "saving": (round(_mean([r["e2e_heard"] for r in rows]) - _mean(cf), 4)
                   if cf else None),
    }

    return {
        "about": ("ASR stage recomputed from marks the run already wrote; no rerun, no audio. "
                  "A is fdbench's number (VAD end -> transcript). B adds endpointing, which is what the "
                  "deck wording 'end of speech -> transcript' promises. C/D use the streaming model's last "
                  "interim: when the text already existed, before the second pass."),
        "run_dir": str(run_dir),
        "turns": len(rows),
        "log_passes_found": len(passes),
        "turns_matched_to_log": sum(1 for r in rows if r["whisper_s"] is not None),
        "reproduces_shipped": {
            "reported_mean_from_stage_latency_json": orig_mean,
            "recomputed_mean_A": variants["A_vad_end_to_mark"]["mean"],
            "agree": (orig_mean is not None
                      and abs(orig_mean - variants["A_vad_end_to_mark"]["mean"]) < 0.005),
        },
        "variants": variants,
        "decomposition": decomp,
        "streaming_only": streaming_only,
        "counterfactual": counterfactual,
        "wer": {
            "target": TARGET_WER,
            "last_partial": _wer([(r["ref"], r["partial_text"]) for r in rows]),
            "streaming_flushed": _wer([(r["ref"], r["streaming_flushed"]) for r in rows
                                       if r["streaming_flushed"]]),
            "whisper_final": _wer([(r["ref"], r["final_text"]) for r in rows]),
            "note": ("last_partial = the interim mark, often truncated mid-word, so pessimistic. "
                     "streaming_flushed = what final_pass=none would actually emit. "
                     "whisper_final = the transcript the run used."),
        },
        "per_turn": rows,
    }


def rescore_committed(result_dir: Path) -> dict:
    """A and B only, from the committed stage_latency.json (no raw marks needed)."""
    data = json.loads((result_dir / "stage_latency.json").read_text(encoding="utf-8"))
    turns = data.get("turns", [])
    a = [t.get("asr_final") for t in turns]
    ep = [t.get("endpointing") for t in turns]
    b = [(x + y) if (x is not None and y is not None) else None for x, y in zip(a, ep)]
    sa, sb = _stats(a), _stats(b)
    return {
        "about": "A and B recovered from the committed stage_latency.json alone (endpointing + asr_final).",
        "run_dir": str(result_dir),
        "turns": len(turns),
        "variants": {
            "A_vad_end_to_mark": dict(label="A  VAD end -> transcript (as shipped)", **sa,
                                      verdict=_verdict(sa["mean"])),
            "B_speech_end_to_mark": dict(label="B  true speech end -> transcript (deck wording)", **sb,
                                         verdict=_verdict(sb["mean"])),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", nargs="?", help="fd-benchmark/runs/<student>/<run-id>")
    ap.add_argument("--from-committed", metavar="DIR",
                    help="recover A and B from a committed bench/results/<name>/ folder")
    ap.add_argument("--json", metavar="PATH", help="also write the full result as JSON")
    args = ap.parse_args(argv)

    if args.from_committed:
        res = rescore_committed(Path(args.from_committed).resolve())
    else:
        if not args.run_dir:
            ap.error("give a run directory, or --from-committed")
        res = rescore(Path(args.run_dir).resolve())

    print(f"{'variant':52s} {'mean':>7} {'p95':>7} {'n':>3}  verdict")
    for v in res["variants"].values():
        mean = "-" if v["mean"] is None else f"{v['mean']:.4f}"
        p95 = "-" if v["p95"] is None else f"{v['p95']:.4f}"
        print(f"{v['label']:52s} {mean:>7} {p95:>7} {v['n']:>3}  {v['verdict']}")

    r = res.get("reproduces_shipped")
    if r:
        print(f"\nreproduces the shipped figure: {r['agree']} "
              f"(reported {r['reported_mean_from_stage_latency_json']}, "
              f"recomputed {r['recomputed_mean_A']})")
    if res.get("log_passes_found"):
        print(f"log passes matched: {res['turns_matched_to_log']} of {res['turns']} turns "
              f"({res['log_passes_found']} in the log)")
    w = res.get("wer")
    if w:
        print(f"WER  last partial {w['last_partial']['wer']}  "
              f"streaming flushed {w['streaming_flushed']['wer']}  "
              f"whisper final {w['whisper_final']['wer']}  (target {w['target']})")
    d = res.get("decomposition")
    if d:
        print("\ndecomposition of the heard end-to-end (all measured):")
        for k, v in d.items():
            print(f"  {k:42s} {v}")
    c = res.get("counterfactual")
    if c:
        print(f"\n{c['note']}")
        print(f"  e2e_heard measured            {c['measured_e2e_heard']['mean']}")
        print(f"  e2e_heard, no final pass      {c['e2e_heard_no_final_pass']['mean']}")

    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
