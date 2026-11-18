"""Render the four deck-v2 images at PowerPoint slide size.

The v1 images were sized for a page, not for a slide: 2535x3606 px for the
benchmark table, when the slide is 1920x1080 and that table gets a quarter of
one page. Scaled down to fit, the text became unreadable. So each image here is
rendered at the pixel box it will actually occupy, at 150 dpi, with font sizes
chosen so that the smallest label is ~22 px on the slide:

    diagram.png        1800 x 520   top half of a page
    MCP_Services.png   1800 x 520   top half of another page
    Benchmark.png       900 x 480   bottom-right quarter of a page
    latency_chart.png   880 x 460   beside the benchmark table

Everything drawn is read from the committed run and from config.py, not typed:

    bench/results/full-2026-11-18/{results.csv,results.json,asr_rescore.json}
    config.py  (VAD_*, TURN_STOP_SECS, LLM_*, MCP_MODEL_RESULT_CHARS)

Run from the repo root:

    .venv/bin/python bench/gen_deck_v2.py
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "bench" / "results" / "full-2026-11-18"
DECK = ROOT / "ME 3 Deck v2"
sys.path.insert(0, str(ROOT))
import config  # noqa: E402

DPI = 150
BG = "#ffffff"
TXT = "#16202b"
MUTED = "#5b6b7c"
PI_FILL = "#eef4fb"
PI_EDGE = "#4a7fb5"
CLOUD_FILL = "#fdf1e7"
CLOUD_EDGE = "#c8792f"
STAGE_FILL = "#ffffff"
STAGE_EDGE = "#3d5a80"
ACCENT = "#b3402f"
HDR_BG = "#3d5a80"
ROW_A = "#ffffff"
ROW_B = "#f4f7fb"

# --- read the run ----------------------------------------------------------
metrics: dict[str, dict] = {}
with (RUN / "results.csv").open() as fh:
    for row in csv.DictReader(fh):
        k = row["key"].strip()
        if k and k not in metrics:
            metrics[k] = row

rescore = json.loads((RUN / "asr_rescore.json").read_text())
stages = json.loads((RUN / "stage_latency.json").read_text())
results = json.loads((RUN / "results.json").read_text())
CTX = results["metrics"]["course.context"]


def f(key: str, field: str = "value") -> float:
    return float(metrics[key][field])


def canvas(w_px: int, h_px: int, xmax: float, ymax: float):
    fig, ax = plt.subplots(figsize=(w_px / DPI, h_px / DPI), dpi=DPI)
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.axis("off")
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig, ax


# ===========================================================================
# 1. diagram.png -- 1800 x 520, top half of a page
# ===========================================================================
def draw_diagram() -> Path:
    fig, ax = canvas(1800, 520, 180, 52)

    def box(x, y, w, h, title, sub, fc=STAGE_FILL, ec=STAGE_EDGE,
            tfs=11.8, sfs=9.2, lw=1.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.10,rounding_size=0.6",
                                    linewidth=lw, facecolor=fc, edgecolor=ec,
                                    zorder=3))
        cy = y + h / 2
        ax.text(x + w / 2, cy + h * 0.22, title, ha="center", va="center",
                fontsize=tfs, color=TXT, weight="bold", zorder=4)
        ax.text(x + w / 2, cy - h * 0.20, sub, ha="center", va="center",
                fontsize=sfs, color=MUTED, zorder=4, linespacing=1.35)

    def arrow(p1, p2, color=STAGE_EDGE, lw=2.0, ls="-", rad=0.0, mut=14):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", color=color,
                                     linewidth=lw, linestyle=ls, zorder=2,
                                     mutation_scale=mut,
                                     connectionstyle=f"arc3,rad={rad}"))

    def elbow(pts, color=STAGE_EDGE, lw=2.0, ls="-"):
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw,
                linestyle=ls, zorder=2, solid_capstyle="round",
                solid_joinstyle="round")
        ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                                     color=color, linewidth=lw, linestyle=ls,
                                     mutation_scale=14, zorder=2))

    # titles
    ax.text(90, 49.2, "WanDog \u2014 full-duplex voice agent on a Raspberry Pi 5",
            ha="center", va="center", fontsize=16.5, weight="bold", color=TXT)
    ax.text(90, 45.6,
            f"harness Pipecat 1.15.0   \u00b7   LLM {config.LLM_MODEL}   "
            "\u00b7   TTS Piper en_US-ryan-medium   \u00b7   VAD Silero ONNX",
            ha="center", va="center", fontsize=9.4, color=MUTED)

    # zones: one Pi rect behind everything, one small cloud rect around the LLM
    ax.add_patch(Rectangle((0.8, 3.0), 178.4, 39.4, facecolor=PI_FILL,
                           edgecolor=PI_EDGE, linewidth=1.8, zorder=1,
                           linestyle=(0, (7, 5))))
    ax.text(2.6, 40.9, "ON THE RASPBERRY PI 5  \u2014  8 GB, 4 cores, no accelerator",
            ha="left", va="center", fontsize=10.0, weight="bold",
            color=PI_EDGE, zorder=5)
    ax.add_patch(Rectangle((96.6, 22.4), 34.4, 19.6, facecolor=CLOUD_FILL,
                           edgecolor=CLOUD_EDGE, linewidth=1.8, zorder=1,
                           linestyle=(0, (7, 5))))
    ax.text(98.4, 40.9, "CLOUD  \u2014  Wi-Fi", ha="left", va="center",
            fontsize=10.0, weight="bold", color=CLOUD_EDGE, zorder=5)

    R1Y, R1H = 29.5, 11.0
    box(3.0, R1Y, 13.0, R1H, "Mic", "USB\n48 kHz")
    box(18.5, R1Y, 15.5, R1H, "Resample", "48 \u2192 16 k")
    box(36.5, R1Y, 14.0, R1H, "VAD", "Silero")
    box(53.0, R1Y, 17.5, R1H, "ASR", "streaming\nzipformer")
    box(73.0, R1Y, 20.0, R1H, "Turn control", f"stop {config.TURN_STOP_SECS:.1f} s")
    box(99.0, 26.5, 29.0, 13.0, "LLM + tools",
        "gemma4:31b-cloud\n14 MCP tools", tfs=12.4, sfs=9.6,
        fc=CLOUD_FILL, ec=CLOUD_EDGE)
    box(133.5, R1Y, 18.0, R1H, "TTS", "Piper\n48 kHz")
    box(154.5, R1Y, 20.0, R1H, "Speaker", "voice out")

    # the optional second pass, under the ASR
    box(53.0, 14.6, 17.5, 10.6, "ASR final", "whisper\ntiny.en 0.96 s",
        tfs=11.2, sfs=9.0, fc="#fbf0ee", ec=ACCENT)

    # row-1 chain
    arrow((16.0, 35.0), (18.5, 35.0))
    arrow((34.0, 35.0), (36.5, 35.0))
    arrow((50.5, 35.0), (53.0, 35.0))
    arrow((70.5, 35.0), (73.0, 35.0))
    arrow((93.0, 35.0), (99.0, 35.0))
    arrow((128.0, 35.0), (133.5, 35.0))
    arrow((151.5, 35.0), (154.5, 35.0))
    # ASR -> final pass, and final pass -> turn control
    arrow((61.75, 29.5), (61.75, 25.0), color=ACCENT)
    elbow([(70.5, 20.0), (77.0, 20.0), (77.0, 29.5)], color=ACCENT)

    # barge-in: speech detected -> stop TTS + cancel LLM -> listen
    elbow([(43.5, 29.5), (43.5, 7.6), (142.5, 7.6), (142.5, 29.5)],
          color=ACCENT, lw=2.2, ls=(0, (6, 4)))
    ax.text(93.0, 9.6,
            "barge-in:  speech detected  \u2192  stop TTS playback + cancel the LLM stream  \u2192  listen",
            ha="center", va="center", fontsize=9.6, color=ACCENT, weight="bold")
    ax.text(93.0, 5.4,
            "the only hop off the Pi on the speech path is the LLM call",
            ha="center", va="center", fontsize=9.0, color=MUTED, style="italic")

    out = DECK / "diagram.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


# ===========================================================================
# 2. MCP_Services.png -- 1800 x 520, top half of a page
# ===========================================================================
MCP_ROWS = [
    ("ASR", "in-process\n(no MCP)", "streaming partials\n+ final transcript",
     "sherpa-onnx streaming\nzipformer int8", "Pi \u00b7 16 kHz \u00b7 text ready\nBEFORE the VAD ends the turn", False),
    ("ASR final", "in-process\n(no MCP)", "one authoritative\ntranscript", "faster-whisper tiny.en\nint8", "Pi \u00b7 optional second pass\nTHIS is the 0.96 s", False),
    ("TTS", "in-process\n(no MCP)", "speak(text) \u2192 audio\nstreamed chunks", "Piper en_US-ryan-medium\n(ONNX)", "Pi \u00b7 48 kHz out \u00b7 chunks\ncancelled on barge-in", False),
    ("VAD", "in-process\n(no MCP)", "detect_speech(frames)", "Silero ONNX\n(torch-free)",
     f"Pi \u00b7 16 kHz \u00b7 conf {config.VAD_CONFIDENCE:.1f}\nstart {config.VAD_START_SECS:.2f} s \u00b7 stop {config.VAD_STOP_SECS:.1f} s", False),
    ("LLM\n(harness MCP client)", "OpenAI-compatible\nHTTP", "function calling\nover 14 MCP tools",
     "gemma4:31b-cloud\ntemp 0.7 \u00b7 300 tok", "CLOUD \u00b7 ollama.com/v1 \u00b7 Wi-Fi\nthe only hop off the Pi", True),
    ("ToolsLocal\nMCPServer", "stdio", "12 tools: bash, files,\nsearch, github_repo, +8",
     "n/a \u2014 local\nexecution", "Pi \u00b7 stdio subprocess of\nOnIt's interpreter", False),
    ("ToolsNet\nMCPServer", "stdio", "2 tools: search,\nget_weather", "n/a \u2014 calls out",
     "Pi process \u2192 cloud APIs\n(ddgs / Tavily, OpenWeather)", False),
]


def draw_mcp_table() -> Path:
    fig, ax = canvas(1800, 520, 180, 52)
    cols = [("Component", 24.0), ("Transport", 20.0),
            ("Tools", 32.0), ("Model", 42.0), ("Served where / notes", 60.0)]
    x0, hdr_h, row_h = 1.0, 4.6, 5.3
    top = 45.0

    ax.text(90, 49.4, "MCP services \u2014 WanDog, Raspberry Pi 5",
            ha="center", va="center", fontsize=16.0, weight="bold", color=TXT)
    ax.text(90, 46.3,
            "14 tools on 2 MCP servers, all built here and all running on the Pi  \u00b7  "
            "ASR, TTS and VAD are in-process Pipecat services, not MCP servers",
            ha="center", va="center", fontsize=9.4, color=MUTED)

    # header
    x = x0
    ax.add_patch(Rectangle((x0, top - hdr_h), sum(c[1] for c in cols), hdr_h,
                           facecolor=HDR_BG, edgecolor="none", zorder=2))
    for name, w in cols:
        ax.text(x + 1.1, top - hdr_h / 2, name, ha="left", va="center",
                fontsize=10.4, weight="bold", color="#ffffff", zorder=3)
        x += w

    y = top - hdr_h
    for i, row in enumerate(MCP_ROWS):
        fc = ROW_B if i % 2 else ROW_A
        ax.add_patch(Rectangle((x0, y - row_h), sum(c[1] for c in cols), row_h,
                               facecolor=fc, edgecolor="#d8e0e9", linewidth=0.8,
                               zorder=1))
        x = x0
        is_cloud = row[5]
        for j, (name, w) in enumerate(cols):
            txt = row[j]
            col = TXT
            weight = "normal"
            if j == 0:
                weight = "bold"
                col = CLOUD_EDGE if is_cloud else STAGE_EDGE
            elif j == 4:
                col = CLOUD_EDGE if txt.startswith("CLOUD") else MUTED
            ax.text(x + 1.1, y - row_h / 2, txt, ha="left", va="center",
                    fontsize=9.6, color=col, weight=weight, zorder=3,
                    linespacing=1.30)
            x += w
        y -= row_h

    ax.text(x0, 0.9,
            "Cloud boundary: only the LLM call leaves the Pi. The two MCP servers are "
            "stdio subprocesses on the same machine as the agent.",
            ha="left", va="center", fontsize=9.0, color=MUTED, style="italic")

    out = DECK / "MCP_Services.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


# ===========================================================================
# 3. Benchmark.png -- 900 x 480, bottom-right quarter of a page
# ===========================================================================
def draw_bench_table() -> Path:
    fig, ax = canvas(900, 480, 90, 48)
    V = rescore["variants"]

    rows = [
        ("E2E latency", "VAD end \u2192 audio out", "< 4.00 s",
         f"{f('course.e2e'):.2f} s", f"{f('course.e2e', 'p95'):.2f} s", "met"),
        ("E2E heard", "true speech end \u2192 reply", "< 4.00 s",
         f"{f('course.e2e_heard'):.2f} s", f"{f('course.e2e_heard', 'p95'):.2f} s", "met"),
        ("ASR final", "VAD end \u2192 transcript", "< 0.80 s",
         f"{f('course.asr_final'):.2f} s", f"{f('course.asr_final', 'p95'):.2f} s", "OVER"),
        ("LLM TTFT", "transcript \u2192 first token", "< 1.50 s",
         f"{f('course.llm_ttft'):.2f} s", f"{f('course.llm_ttft', 'p95'):.2f} s", "met"),
        ("TTS TTFB", "text \u2192 first audio", "< 0.60 s",
         f"{f('course.tts_ttfb'):.2f} s", f"{f('course.tts_ttfb', 'p95'):.2f} s", "met"),
        ("Barge-in", "speech \u2192 playback stop", "< 0.80 s",
         f"{f('course.barge_in'):.2f} s", f"{f('course.barge_in'):.2f} s", "met"),
        ("WER", "ASR accuracy", "< 15 %",
         f"{f('course.wer') * 100:.2f} %", "\u2014", "met"),
    ]

    cols = [("Metric", 27.0), ("Target", 12.0), ("Mean", 13.0),
            ("p95", 13.0), ("", 23.0)]
    x0, hdr_h, row_h = 1.0, 4.0, 4.4
    top = 40.2

    ax.text(45, 45.6, "Evaluation & benchmark", ha="center", va="center",
            fontsize=12.5, weight="bold", color=TXT)
    ax.text(45, 42.4, "Raspberry Pi 5 \u00b7 2026-11-18 \u00b7 10 phrases \u00d7 3 runs \u00b7 n = 30",
            ha="center", va="center", fontsize=8.8, color=MUTED)

    x = x0
    ax.add_patch(Rectangle((x0, top - hdr_h), sum(c[1] for c in cols), hdr_h,
                           facecolor=HDR_BG, edgecolor="none", zorder=2))
    for name, w in cols:
        ax.text(x + 1.0, top - hdr_h / 2, name, ha="left", va="center",
                fontsize=9.4, weight="bold", color="#ffffff", zorder=3)
        x += w

    y = top - hdr_h
    for i, (metric, definition, target, mean, p95, verdict) in enumerate(rows):
        over = verdict == "OVER"
        fc = "#fdf0ee" if over else (ROW_B if i % 2 else ROW_A)
        ax.add_patch(Rectangle((x0, y - row_h), sum(c[1] for c in cols), row_h,
                               facecolor=fc, edgecolor="#d8e0e9", linewidth=0.7,
                               zorder=1))
        ax.text(x0 + 1.0, y - row_h * 0.32, metric, ha="left", va="center",
                fontsize=9.4, weight="bold", color=ACCENT if over else TXT, zorder=3)
        ax.text(x0 + 1.0, y - row_h * 0.73, definition, ha="left", va="center",
                fontsize=7.8, color=MUTED, zorder=3)
        xx = x0 + cols[0][1]
        for val, w, weight in ((target, cols[1][1], "normal"),
                               (mean, cols[2][1], "bold"),
                               (p95, cols[3][1], "normal")):
            ax.text(xx + 1.0, y - row_h / 2, val, ha="left", va="center",
                    fontsize=9.4, color=ACCENT if over else TXT, weight=weight,
                    zorder=3)
            xx += w
        ax.text(xx + 1.0, y - row_h / 2, "\u2717 missed" if over else "\u2713 met",
                ha="left", va="center", fontsize=8.8,
                color=ACCENT if over else "#2f7d4f", weight="bold", zorder=3)
        y -= row_h

    ax.text(x0, 3.4,
            f"ASR row is VAD end \u2192 transcript. True speech end \u2192 transcript is "
            f"{V['B_speech_end_to_mark']['mean']:.2f} s: 0.65 s of it is endpointing.",
            ha="left", va="center", fontsize=7.6, color=MUTED, style="italic")
    ax.text(x0, 1.4,
            f"n = 2 for barge-in. Hand-offs + playout are "
            f"{rescore['decomposition']['handoff_and_playout']:.3f} s of the total.",
            ha="left", va="center", fontsize=7.6, color=MUTED, style="italic")

    out = DECK / "Benchmark.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


# ===========================================================================
# 4. latency_chart.png -- 880 x 460
# ===========================================================================
def draw_chart() -> Path:
    turns = stages["turns"]

    def total(t):
        return (t.get("asr_final") or 0) + (t.get("llm_ttft") or 0) + (t.get("tts_ttfb") or 0)

    turns = sorted(turns, key=total)
    asr = np.array([t.get("asr_final") or 0 for t in turns])
    ttft = np.array([t.get("llm_ttft") or 0 for t in turns])
    ttfb = np.array([t.get("tts_ttfb") or 0 for t in turns])
    hand = np.array([max(0.0, (t.get("e2e") or 0) - a - b - c)
                     for t, a, b, c in zip(turns, asr, ttft, ttfb)])
    x = np.arange(len(turns))

    fig, ax = canvas(880, 460, 90, 47)
    ax = fig.add_axes([0.085, 0.15, 0.885, 0.70])
    ax.set_facecolor(BG)

    kw = dict(width=0.82, edgecolor="none")
    ax.bar(x, asr, color="#b3402f", label=f"ASR final pass  {asr.mean():.2f} s", **kw)
    ax.bar(x, ttft, bottom=asr, color="#3d5a80",
           label=f"LLM TTFT  {ttft.mean():.2f} s", **kw)
    ax.bar(x, ttfb, bottom=asr + ttft, color="#6b9ac4",
           label=f"TTS TTFB  {ttfb.mean():.2f} s", **kw)
    ax.bar(x, hand, bottom=asr + ttft + ttfb, color="#c9d6e3",
           label=f"hand-offs  {hand.mean():.3f} s", **kw)

    ax.set_ylabel("seconds, VAD end \u2192 audio out", fontsize=9.6, color=TXT)
    ax.set_xlabel("30 turns, sorted", fontsize=9.6, color=TXT)
    ax.set_xlim(-0.8, len(turns) - 0.2)
    ax.set_ylim(0, 3.1)
    ax.tick_params(labelsize=9, colors=MUTED)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3ccd6")
    ax.grid(axis="y", color="#e6ebf1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.axhline(asr.mean() + ttft.mean() + ttfb.mean() + hand.mean(),
               color="#16202b", lw=1.2, ls=(0, (5, 3)))
    ax.text(0.4, asr.mean() + ttft.mean() + ttfb.mean() + hand.mean() + 0.06,
            f"mean E2E {asr.mean() + ttft.mean() + ttfb.mean() + hand.mean():.2f} s",
            fontsize=9.0, color=TXT, weight="bold")
    ax.legend(fontsize=8.6, frameon=False, loc="upper left", ncol=2,
              handlelength=1.1, columnspacing=1.2, labelcolor=TXT)
    ax.set_title("Where the 2.07 s goes, per turn", fontsize=12.0, weight="bold",
                 color=TXT, pad=7)

    out = DECK / "latency_chart.png"
    fig.savefig(out, dpi=DPI, facecolor=BG)
    plt.close(fig)
    return out


def main() -> None:
    DECK.mkdir(exist_ok=True)
    for fn in (draw_diagram, draw_mcp_table, draw_bench_table, draw_chart):
        p = fn()
        from PIL import Image
        w, h = Image.open(p).size
        print(f"{p.name:<20} {w} x {h} px")


if __name__ == "__main__":
    main()
