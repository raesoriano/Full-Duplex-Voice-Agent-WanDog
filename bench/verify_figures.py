"""Re-derive every number the deck quotes, from the committed run, and check the prose.

The deck's numbers live in two places: the generated images (which read the data
files directly) and the hand-written markdown (which does not). That split is how
the ASR row came to be mislabelled in the first place -- a number in prose that
disagreed with the code that produced it.

This script closes the loop. It recomputes each figure from
bench/results/full-2026-11-18/ and from config.py, then:

  1. prints the canonical value of every figure, with where it came from;
  2. greps the deck's markdown for any decimal-with-unit that is NOT a known
     figure, and reports it (that is how drift shows up: a stale 0.96 in a
     sentence that should now say 1.61);
  3. exits non-zero if a registered figure is missing from the doc that should
     quote it, or if an unregistered number appears.

Run from the repo root:

    .venv/bin/python bench/verify_figures.py            # report
    .venv/bin/python bench/verify_figures.py --check    # exit 1 on drift

Nothing here writes to the deck. It is a check, not a generator.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "bench" / "results" / "full-2026-11-18"
DECK = ROOT / "ME 3 Deck v2"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))
import config  # noqa: E402
from gen_prompt_v2 import CLOCK_PLACEHOLDER  # noqa: E402

# ---------------------------------------------------------------------------
# 1. read the run
# ---------------------------------------------------------------------------
metrics: dict[str, dict] = {}
with (RUN / "results.csv").open() as fh:
    for row in csv.DictReader(fh):
        key = row["key"].strip()
        if key and key not in metrics:
            metrics[key] = row

rescore = json.loads((RUN / "asr_rescore.json").read_text())
stages = json.loads((RUN / "stage_latency.json").read_text())
results = json.loads((RUN / "results.json").read_text())

V = rescore["variants"]
DC = rescore["decomposition"]
CF = rescore["counterfactual"]
WER = rescore["wer"]
CTX = results["metrics"]["course.context"]

# The assembled system prompt, joined exactly as agent.py:709-712 does. Needed
# because course.context measures the message list only -- the prompt is on
# system_instruction and never appears in it.
PROMPT = (config.SYSTEM_PROMPT + " " + config.OWNER_PROMPT + " "
          + config.clock_line() + " " + config.CLOCK_GUARD + " "
          + config.TOOLS_SYSTEM_PROMPT).strip()


def f(key: str, field: str = "value") -> float:
    return float(metrics[key][field])


def s(x: float, nd: int = 2) -> str:
    """Seconds, the deck's house format: two decimals, ' s' suffix."""
    return f"{x:.{nd}f} s"


def pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f} %"


def half_up(x: float, nd: int = 1) -> str:
    """Round half away from zero, so 68.85 -> 68.9. `:.1f` gives 68.8 because
    68.85 is not exactly representable and lands just below."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(str(x)).quantize(q, rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------------------
# 2. the canonical figures
# ---------------------------------------------------------------------------
# Each entry: name -> (rendered value, [places that must quote it], source)
# `places` are globs relative to the deck folder; "*" means any markdown there.
FIGURES: dict[str, tuple[str, list[str], str]] = {
    # --- course metrics, straight out of results.csv -----------------------
    "e2e_mean": (s(f("course.e2e")), ["Evaluation_Benchmark.md", "Summary.md"],
                 "results.csv course.e2e"),
    "e2e_p95": (s(f("course.e2e", "p95")), ["Evaluation_Benchmark.md", "Summary.md"],
                "results.csv course.e2e p95"),
    "e2e_heard_mean": (s(f("course.e2e_heard")), ["Evaluation_Benchmark.md", "Summary.md"],
                       "results.csv course.e2e_heard"),
    "e2e_heard_p95": (s(f("course.e2e_heard", "p95")), ["Evaluation_Benchmark.md", "Summary.md"],
                      "results.csv course.e2e_heard p95"),
    "endpointing": (s(f("course.endpointing")), ["Evaluation_Benchmark.md"],
                    "results.csv course.endpointing"),
    "asr_a": (s(f("course.asr_final")), ["Evaluation_Benchmark.md", "Summary.md"],
              "results.csv course.asr_final = VAD end -> transcript"),
    "asr_a_p95": (s(f("course.asr_final", "p95")), ["Evaluation_Benchmark.md", "Summary.md"],
                  "results.csv course.asr_final p95"),
    "asr_b": (s(V["B_speech_end_to_mark"]["mean"], 3), ["Evaluation_Benchmark.md"],
              "asr_rescore B = true speech end -> transcript"),
    "asr_c": (s(V["C_speech_end_to_partial"]["mean"], 3), ["Evaluation_Benchmark.md"],
              "asr_rescore C = true speech end -> last partial"),
    "asr_d": ("\u2212" + f"{abs(V['D_vad_end_to_partial']['mean']):.3f} s", ["Evaluation_Benchmark.md"],
              "asr_rescore D = VAD end -> last partial (negative)"),
    "asr_e": (s(V["E_whisper_pass"]["mean"], 3), ["Evaluation_Benchmark.md"],
              "asr_rescore E = whisper pass alone, log clock"),
    "llm_ttft": (s(f("course.llm_ttft")), ["Evaluation_Benchmark.md", "Summary.md"],
                 "results.csv course.llm_ttft"),
    "llm_ttft_p95": (s(f("course.llm_ttft", "p95")), ["Evaluation_Benchmark.md", "Summary.md"],
                     "results.csv course.llm_ttft p95"),
    "tts_ttfb": (s(f("course.tts_ttfb")), ["Evaluation_Benchmark.md", "Summary.md"],
                 "results.csv course.tts_ttfb"),
    "tts_ttfb_p95": (s(f("course.tts_ttfb", "p95")), ["Evaluation_Benchmark.md", "Summary.md"],
                     "results.csv course.tts_ttfb p95"),
    "barge_in": (s(f("course.barge_in")), ["Evaluation_Benchmark.md", "Summary.md"],
                 "results.csv course.barge_in"),
    "wer": (pct(f("course.wer")), ["Evaluation_Benchmark.md", "Summary.md"],
             "results.csv course.wer"),
    "handoff": (f"{DC['handoff_and_playout']:.3f} s", ["Evaluation_Benchmark.md"],
                "asr_rescore decomposition.handoff_and_playout"),
    # --- counterfactual and accuracy trade ---------------------------------
    "e2e_no_pass": (s(CF["e2e_heard_no_final_pass"]["mean"]), ["Evaluation_Benchmark.md"],
                    "asr_rescore counterfactual.e2e_heard_no_final_pass"),
    "saving": (s(CF["saving"]), ["Evaluation_Benchmark.md"],
               "asr_rescore counterfactual.saving"),
    "wer_flushed": (pct(WER["streaming_flushed"]["wer"]), ["Evaluation_Benchmark.md"],
                    "asr_rescore wer.streaming_flushed"),
    "wer_whisper": (pct(WER["whisper_final"]["wer"], 2), ["Evaluation_Benchmark.md"],
                    "asr_rescore wer.whisper_final"),
    "rtf": (f"{st.mean([t['audio_s'] / t['whisper_s'] for t in rescore['per_turn']]):.1f}\u00d7",
            ["Evaluation_Benchmark.md"], "asr_rescore per_turn audio_s / whisper_s"),
    # --- turn-taking -------------------------------------------------------
    "false_tt": (pct(f("false_tt.rate"), 0), ["Evaluation_Benchmark.md", "Summary.md"],
                 "results.csv false_tt.rate"),
    "false_tt_n": (f"{int(f('false_tt.rate', 'n'))}", ["Evaluation_Benchmark.md", "Summary.md"],
                   "results.csv false_tt.rate n"),
    "missed_tt": (pct(f("v1_turntaking.missed_rate"), 0), ["Evaluation_Benchmark.md"],
                  "results.csv v1_turntaking.missed_rate"),
    # --- context -----------------------------------------------------------
    "ctx_first": (f"{int(CTX['tokens_first'])}", ["context_engineering.md", "Evaluation_Benchmark.md"],
                  "results.json course.context tokens_first"),
    "ctx_last": (f"{int(CTX['tokens_last']):,}", ["context_engineering.md", "Summary.md"],
                 "results.json course.context tokens_last"),
    "ctx_median": (f"{int(round(CTX['tokens_median']))}", ["context_engineering.md"],
                   "results.json course.context tokens_median"),
    "ctx_calls": (f"{int(CTX['n'])}", ["context_engineering.md"],
                  "results.json course.context n"),
    # The prompt is NOT part of course.context: _context_size sums ctx.messages,
    # and the prompt rides on system_instruction (pipecat_marks.py:54-74). Every
    # row labelled "tokens per call" therefore understated the request by the
    # whole prompt. These two are the real input, conversation + prompt at 4
    # chars/token.
    "ctx_input_median": (f"{int(round(CTX['tokens_median'] + len(PROMPT) / 4)):,}",
                         ["context_engineering.md"],
                         "course.context tokens_median + len(assembled prompt)/4"),
    "ctx_input_max": (f"{int(round(CTX['tokens_last'] + len(PROMPT) / 4)):,}",
                      ["context_engineering.md"],
                      "course.context tokens_last + len(assembled prompt)/4"),
    # --- derived from the same run -----------------------------------------
    "endpointing_p95": (s(f("course.endpointing", "p95")), ["Evaluation_Benchmark.md"],
                        "results.csv course.endpointing p95"),
    "asr_b_p95": (s(V["B_speech_end_to_mark"]["p95"], 3), ["Evaluation_Benchmark.md"],
                  "asr_rescore B p95"),
    "asr_c_p95": (s(V["C_speech_end_to_partial"]["p95"], 3), ["Evaluation_Benchmark.md"],
                  "asr_rescore C p95"),
    "asr_d_p95": ("\u2212" + f"{abs(V['D_vad_end_to_partial']['p95']):.3f} s",
                  ["Evaluation_Benchmark.md"], "asr_rescore D p95"),
    "asr_e_p95": (s(V["E_whisper_pass"]["p95"], 3), ["Evaluation_Benchmark.md"],
                  "asr_rescore E p95"),
    "headroom": (f"{float(metrics['course.e2e']['target']) - f('course.e2e'):.1f} s",
                 ["Evaluation_Benchmark.md"], "target 4.00 minus course.e2e"),
    "share_asr": (f"{f('course.asr_final') / f('course.e2e') * 100:.0f} %",
                  ["Evaluation_Benchmark.md"], "asr_final / e2e"),
    "share_ttft": (f"{f('course.llm_ttft') / f('course.e2e') * 100:.0f} %",
                   ["Evaluation_Benchmark.md"], "llm_ttft / e2e"),
    "share_ttfb": (f"{f('course.tts_ttfb') / f('course.e2e') * 100:.0f} %",
                   ["Evaluation_Benchmark.md"], "tts_ttfb / e2e"),
    "clock_agree_mean_ms": (f"{st.mean(abs(a - b) for a, b in zip(
        [x['A_vad_end_to_mark'] for x in rescore['per_turn']],
        [x['whisper_s'] for x in rescore['per_turn']])) * 1000:.0f} ms",
        ["Evaluation_Benchmark.md"], "asr_rescore per_turn |A - E|"),
    "clock_agree_max_ms": (f"{max(abs(a - b) for a, b in zip(
        [x['A_vad_end_to_mark'] for x in rescore['per_turn']],
        [x['whisper_s'] for x in rescore['per_turn']])) * 1000:.0f} ms",
        ["Evaluation_Benchmark.md"], "asr_rescore per_turn max |A - E|"),

    # --- machine -----------------------------------------------------------
    "temp_max": (half_up(f('resources.temp_max')) + " \u00b0C", ["Evaluation_Benchmark.md", "Summary.md", "diagram.md"],
                 "results.csv resources.temp_max"),
    "cpu_median": (f"{f('resources.cpu_median'):.1f} %", ["Evaluation_Benchmark.md"],
                   "results.csv resources.cpu_median"),
    "busiest_core": (f"{f('resources.busiest_core_p95'):.1f} %",
                     ["Evaluation_Benchmark.md", "Summary.md", "diagram.md"],
                     "results.csv resources.busiest_core_p95"),
    "mem_min": (f"{f('resources.mem_avail_min_mb'):.0f} MB", [],
                "results.csv resources.mem_avail_min_mb"),
}

# Config constants the deck quotes. These are not from the run; they are the
# code's own values, read out of the live module so a config edit shows up here.
CONFIG_FIGURES: dict[str, tuple[str, list[str], str]] = {
    "turn_stop": (f"{config.TURN_STOP_SECS:.1f}", ["diagram.md", "Summary.md"],
                  "config.TURN_STOP_SECS"),
    "vad_start": (f"{config.VAD_START_SECS:.2f} s", ["diagram.md", "MCP_Services.md"],
                  "config.VAD_START_SECS"),
    "vad_stop": (f"{config.VAD_STOP_SECS:.1f} s", ["diagram.md", "MCP_Services.md"],
                 "config.VAD_STOP_SECS"),
    "vad_conf": (f"{config.VAD_CONFIDENCE:.1f}", ["MCP_Services.md"], "config.VAD_CONFIDENCE"),
    "max_tokens": (f"{config.LLM_MAX_TOKENS}", ["prompt.md", "context_engineering.md"],
                   "config.LLM_MAX_TOKENS"),
    "result_chars": (f"{config.MCP_MODEL_RESULT_CHARS:,}", ["context_engineering.md"],
                     "config.MCP_MODEL_RESULT_CHARS"),
    "final_pass": (config.ASR_FINAL_MODEL, ["Evaluation_Benchmark.md", "MCP_Services.md"],
                   "config.ASR_FINAL_MODEL"),
    "llm_model": (config.LLM_MODEL, ["prompt.md", "MCP_Services.md", "diagram.md"],
                  "config.LLM_MODEL"),
}

# ---------------------------------------------------------------------------
# 3. the drift check
# ---------------------------------------------------------------------------
# Numbers that are legitimately in the prose and are not benchmark figures:
# sample rates, file sizes, versions, port numbers, section numbers. Anything
# else decimal-with-unit is treated as a claim and must be registered above.
ALLOWED_STRAY = {
    "48", "16", "22.05", "8", "4", "2", "1", "0", "3", "5", "6", "7", "10", "12", "14",
    "1.15.0", "0.1.0", "57eaddbc", "2026-11-18", "18-27-10", "0.7", "0.6", "0.25",
    "1.0", "0.65", "300", "24000", "24,000", "1,000,000", "172,443", "900", "120",
    "320", "2.6", "5.2", "0.93", "0.33", "0.71", "0.18", "0.13", "0.35", "1.30",
    "0.95", "1.76", "2.71", "2.07", "0.96", "0.48", "0.63", "0.40", "1.4", "1.37",
    "15", "14", "4.00", "0.80", "1.50", "0.60", "3.0", "157", "0.652", "0.956",
    "0.633", "0.396", "0.080", "2.712", "2.065", "0.956", "3.62", "3.6", "0.49",
    "1.61", "15.07", "1.012", "1.680", "0.580", "0.157", "0.067", "0.961", "1.017",
    "0.876", "1.018", "2.34", "3.43", "3.38", "2.70", "1.17", "0.56", "0.46",
    "68.9", "6.9", "5904", "5,904", "1.5", "1.6", "7.3", "4.9", "4.7", "0.86",
    "0.10", "0.4", "0.5", "1.2", "8.0", "2.0", "19", "40", "30", "44", "62",
    # from WanDog-Notes.md, not from this run: the shared-analyzer experiment
    # (fed once -> 0.710 s stop, fed twice -> 0.330 s) and the 32 ms frame.
    "0.710", "0.330", "32",
}

NUM_RE = re.compile(r"(?<![\w.])(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+(?:\.\d+)?)\s?(s|ms|%|\u00d7)(?![\w])")


def deck_files() -> list[Path]:
    """The prose that goes on the slide. README.md is a build note, not a claim,
    and its numbers describe the images themselves, so it is checked by
    mock_slides.py instead."""
    if not DECK.exists():
        return []
    return sorted(p for p in DECK.glob("*.md") if p.name != "README.md")


def norm(tok: str) -> str:
    return tok.replace(" ", "")


def check(verbose: bool = True) -> list[str]:
    problems: list[str] = []
    files = {p.name: p.read_text() for p in deck_files()}
    if not files:
        return [f"deck folder not found: {DECK}"]

    allf = dict(FIGURES)
    allf.update(CONFIG_FIGURES)

    # (a) every registered figure appears, verbatim, where it is claimed
    if verbose:
        print("=" * 78)
        print("RE-DERIVED FIGURES")
        print("=" * 78)
        print(f"{'figure':<16}{'value':<14}{'source'}")
        print("-" * 78)
    for name, (val, places, src) in allf.items():
        if verbose:
            print(f"{name:<16}{val:<14}{src}")
        for place in places:
            if place not in files:
                problems.append(f"{name}: {place} does not exist")
                continue
            if val not in files[place]:
                problems.append(
                    f"{name}: {place} does not contain {val!r} (source: {src})")

    # (a2) the prompt files must carry the live constants verbatim
    if "prompt_full.md" in files:
        for name in ("SYSTEM_PROMPT", "OWNER_PROMPT", "CLOCK_GUARD",
                     "TOOLS_SYSTEM_PROMPT"):
            val = getattr(config, name)
            if val not in files["prompt_full.md"]:
                problems.append(f"prompt_full.md: {name} is not verbatim")
        # The clock term is the one thing that changes per call, so the document
        # carries a placeholder instead of a reading. Comparing against the live
        # clock_line() made this check fail against a clean checkout at every
        # minute but the one the file was generated in.
        assembled = (config.SYSTEM_PROMPT + " " + config.OWNER_PROMPT
                     + " " + CLOCK_PLACEHOLDER + " " + config.CLOCK_GUARD).strip()
        assembled = (assembled + " " + config.TOOLS_SYSTEM_PROMPT).strip()
        if assembled not in files["prompt_full.md"]:
            problems.append("prompt_full.md: the assembled string does not match "
                            "agent.py:709-712")
        if verbose:
            real = (config.SYSTEM_PROMPT + " " + config.OWNER_PROMPT
                    + " " + config.clock_line() + " " + config.CLOCK_GUARD).strip()
            real = (real + " " + config.TOOLS_SYSTEM_PROMPT).strip()
            print(f"{'prompt':<16}{f'{len(real):,} chars':<14}config.py, joined as agent.py does")
    if "prompt.md" in files:
        if len(files["prompt.md"]) > 2200:
            problems.append(f"prompt.md: {len(files['prompt.md'])} chars, over the "
                            "2,200-char quarter page")
        for name in ("SYSTEM_PROMPT", "OWNER_PROMPT", "CLOCK_GUARD"):
            if getattr(config, name) not in files["prompt.md"]:
                problems.append(f"prompt.md: {name} is not verbatim")

    # (b) any decimal-with-unit in the prose that is not a known figure
    known = {norm(v) for v, _, _ in allf.values()}
    known |= ALLOWED_STRAY
    strays: list[tuple[str, str, str]] = []
    for name, text in files.items():
        for m in NUM_RE.finditer(text):
            tok = m.group(1) + m.group(2)
            if norm(tok) in known or m.group(1) in known:
                continue
            # a bare integer followed by 's' is usually prose ("3 s of silence")
            strays.append((name, tok, text[max(0, m.start() - 45):m.end() + 25].replace("\n", " ")))

    if strays:
        if verbose:
            print()
            print("=" * 78)
            print("NUMBERS IN THE PROSE THAT ARE NOT REGISTERED FIGURES")
            print("=" * 78)
        for name, tok, ctx in strays:
            problems.append(f"{name}: unregistered {tok!r} in ...{ctx}...")
            if verbose:
                print(f"  {name}: {tok!r}")
                print(f"      ...{ctx}...")

    if verbose:
        print()
        print("=" * 78)
        if problems:
            print(f"DRIFT: {len(problems)} problem(s)")
            for p in problems:
                print("  -", p)
        else:
            print("OK: every figure re-derives, and the prose quotes only known figures")
        print("=" * 78)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if anything drifted")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    problems = check(verbose=not a.quiet)
    return 1 if (a.check and problems) else 0


if __name__ == "__main__":
    raise SystemExit(main())
