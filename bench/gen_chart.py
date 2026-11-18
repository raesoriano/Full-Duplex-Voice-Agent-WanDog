"""Render "ME 3 Deck/latency_chart.png" — where the 2.07 s goes, per turn.

Reads the committed per-turn marks in
bench/results/full-2026-11-18/stage_latency.json. The stages stack to the measured
E2E, which is the point of the chart: it shows that ASR is the biggest block and
that the pipeline overhead is negligible.

Run from the repo root:  .venv/bin/python bench/gen_chart.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "bench" / "results" / "full-2026-11-18"
OUT = ROOT / "ME 3 Deck" / "latency_chart.png"

data = json.loads((RUN / "stage_latency.json").read_text())
turns = data["turns"]

# Sort by total so the stack reads as a staircase rather than noise.
def total(t):
    return (t.get("asr_final") or 0) + (t.get("llm_ttft") or 0) + (t.get("tts_ttfb") or 0)

turns = sorted(turns, key=total)

asr = np.array([t.get("asr_final") or 0 for t in turns])
ttft = np.array([t.get("llm_ttft") or 0 for t in turns])
ttfb = np.array([t.get("tts_ttfb") or 0 for t in turns])
e2e = np.array([t.get("e2e") or 0 for t in turns])
heard = np.array([t.get("e2e_heard") or 0 for t in turns])

# the remainder is hand-offs + playout, which is what makes the stack add up
handoff = np.clip(e2e - (asr + ttft + ttfb), 0, None)

BG = "#ffffff"
TXT = "#16202b"
MUTED = "#5b6b7c"
C_ASR = "#b3402f"
C_TTFT = "#3d5a80"
C_TTFB = "#4a7fb5"
C_HAND = "#b9c3cf"

fig, (ax1, ax2) = plt.subplots(
    1, 2, figsize=(15.5, 6.4), dpi=175,
    gridspec_kw={"width_ratios": [2.35, 1.0], "wspace": 0.22},
)
fig.patch.set_facecolor(BG)

# --- left: stacked per turn ----------------------------------------------
x = np.arange(len(turns))
ax1.bar(x, asr, color=C_ASR, label="ASR final pass", width=0.82, zorder=3)
ax1.bar(x, ttft, bottom=asr, color=C_TTFT, label="LLM TTFT", width=0.82, zorder=3)
ax1.bar(x, ttfb, bottom=asr + ttft, color=C_TTFB, label="TTS TTFB", width=0.82, zorder=3)
ax1.bar(x, handoff, bottom=asr + ttft + ttfb, color=C_HAND,
        label="hand-offs + playout", width=0.82, zorder=3)

ax1.axhline(4.0, color="#2f7d4f", linewidth=1.8, linestyle="--", zorder=4)
ax1.text(len(turns) - 0.4, 4.06, "target < 4.00 s", ha="right", va="bottom",
         fontsize=9, color="#2f7d4f", weight="bold")
ax1.axhline(e2e.mean(), color=TXT, linewidth=1.2, linestyle=":", zorder=4)
ax1.text(0.2, e2e.mean() + 0.05, f"mean {e2e.mean():.2f} s", ha="left", va="bottom",
         fontsize=9, color=TXT, weight="bold")

ax1.set_title("Where the 2.07 s goes — every timed turn, sorted",
              fontsize=13, weight="bold", color=TXT, pad=12, loc="left")
ax1.set_xlabel("turn (30 stage-latency turns)", fontsize=9.5, color=MUTED)
ax1.set_ylabel("seconds from VAD end", fontsize=9.5, color=MUTED)
ax1.set_ylim(0, 4.5)
ax1.set_xlim(-0.8, len(turns) - 0.2)
ax1.tick_params(labelsize=8.5, colors=MUTED)
ax1.grid(axis="y", color="#e8ecf1", linewidth=0.8, zorder=0)
ax1.set_axisbelow(True)
for s in ("top", "right"):
    ax1.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax1.spines[s].set_color("#c8d0d9")
ax1.legend(fontsize=8.6, frameon=False, loc="upper left", ncol=2)

# --- right: the two clocks ------------------------------------------------
labels = ["VAD end →\naudio out\n(as shipped)", "true speech end →\naudio out\n(what is heard)"]
means = [e2e.mean(), heard.mean()]
p95s = [np.percentile(e2e, 95), np.percentile(heard, 95)]
xs = np.arange(2)
bars = ax2.bar(xs, means, width=0.55, color=[C_TTFT, C_ASR], zorder=3)
for i, (m, p) in enumerate(zip(means, p95s)):
    ax2.text(i, m + 0.08, f"{m:.2f} s", ha="center", va="bottom",
             fontsize=11, weight="bold", color=TXT)
    ax2.plot([i - 0.28, i + 0.28], [p, p], color=TXT, linewidth=1.6, zorder=4)
    ax2.text(i + 0.31, p, f"p95 {p:.2f}", ha="left", va="center",
             fontsize=8.4, color=MUTED)

ax2.axhline(4.0, color="#2f7d4f", linewidth=1.8, linestyle="--", zorder=2)
ax2.text(1.45, 4.05, "target", ha="right", va="bottom", fontsize=8.8,
         color="#2f7d4f", weight="bold")
ax2.annotate("", xy=(1, heard.mean()), xytext=(1, e2e.mean()),
             arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.3))
ax2.text(1.33, (heard.mean() + e2e.mean()) / 2, "+0.65 s\nendpointing",
         ha="left", va="center", fontsize=8.4, color=MUTED, linespacing=1.4)

ax2.set_xticks(xs)
ax2.set_xticklabels(labels, fontsize=8.8, color=MUTED)
ax2.set_title("Two ways to time the same turn",
              fontsize=13, weight="bold", color=TXT, pad=12, loc="left")
ax2.set_ylim(0, 4.5)
ax2.set_xlim(-0.6, 1.75)
ax2.tick_params(labelsize=8.5, colors=MUTED)
ax2.grid(axis="y", color="#e8ecf1", linewidth=0.8, zorder=0)
ax2.set_axisbelow(True)
for s in ("top", "right"):
    ax2.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax2.spines[s].set_color("#c8d0d9")

fig.text(0.008, 0.012,
         "Source: bench/results/full-2026-11-18/stage_latency.json (30 turns, run "
         "2026-11-18_18-27-10). Stack order follows the pipeline: ASR final → LLM TTFT → "
         "TTS TTFB → hand-offs + playout.",
         ha="left", va="bottom", fontsize=7.8, color=MUTED, style="italic")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=175, bbox_inches="tight", facecolor=BG)
print(f"wrote {OUT}")
