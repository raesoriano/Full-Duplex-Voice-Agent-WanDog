"""Draw "ME 3 Deck/diagram.png" — WanDog's actual audio and tool path.

Everything drawn here is read from the code, not from the template:

  pipeline stage order .... agent.py:875-924
  cloud boundary .......... config.py:234 (LLM_BASE_URL)
  MCP transport ........... config.py:305 (MCP_SERVERS, stdio subprocesses)
  sample rates ............ config.py:44,45,126
  barge-in ................ agent.py:797-801
  no AEC / mute ........... agent.py:807-820

Run from the repo root:  .venv/bin/python bench/gen_diagram.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "ME 3 Deck" / "diagram.png"

BG = "#ffffff"
PI_FILL = "#eef4fb"
PI_EDGE = "#4a7fb5"
CLOUD_FILL = "#fdf1e7"
CLOUD_EDGE = "#c8792f"
STAGE_FILL = "#ffffff"
STAGE_EDGE = "#3d5a80"
TXT = "#16202b"
MUTED = "#5b6b7c"
ACCENT = "#b3402f"
TOOL_FILL = "#f5f2fb"
TOOL_EDGE = "#6b5b95"

fig, ax = plt.subplots(figsize=(16.5, 12.0), dpi=165)
ax.set_xlim(0, 100)
ax.set_ylim(0, 74)
ax.axis("off")
fig.patch.set_facecolor(BG)


def box(x, y, w, h, title, sub=None, fc=STAGE_FILL, ec=STAGE_EDGE,
        fs=10.2, subfs=8.0, lw=1.5, tcol=TXT, subcol=MUTED, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.16,rounding_size=0.65",
                                linewidth=lw, facecolor=fc, edgecolor=ec, zorder=z))
    cy = y + h / 2
    if sub:
        ax.text(x + w / 2, cy + h * 0.20, title, ha="center", va="center",
                fontsize=fs, color=tcol, weight="bold", zorder=z + 1)
        ax.text(x + w / 2, cy - h * 0.21, sub, ha="center", va="center",
                fontsize=subfs, color=subcol, zorder=z + 1, linespacing=1.5)
    else:
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                fontsize=fs, color=tcol, weight="bold", zorder=z + 1)


def arrow(p1, p2, color=STAGE_EDGE, lw=2.0, style="-|>", ls="-", rad=0.0,
          z=2, mut=16):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, color=color,
                                 linewidth=lw, linestyle=ls, zorder=z,
                                 mutation_scale=mut,
                                 connectionstyle=f"arc3,rad={rad}"))


def poly(points, color=CLOUD_EDGE, lw=1.9, ls="-", z=2, head=True):
    """Elbow connector through a list of (x, y) points, arrow at the last."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color=color, lw=lw, linestyle=ls, zorder=z,
            solid_capstyle="round", solid_joinstyle="round")
    if head:
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                     color=color, linewidth=lw, linestyle=ls,
                                     mutation_scale=16, zorder=z))


def label(x, y, t, fs=8.4, col=MUTED, ha="center", va="center", style="italic",
          weight="normal", rot=0, bg=None, z=8):
    kw = {}
    if bg:
        kw["bbox"] = dict(boxstyle="round,pad=0.24", fc=bg, ec="none", zorder=z - 1)
    ax.text(x, y, t, ha=ha, va=va, fontsize=fs, color=col, style=style,
            weight=weight, rotation=rot, zorder=z, linespacing=1.5, **kw)


# --- titles ----------------------------------------------------------------
ax.text(50, 71.6, "WanDog — full-duplex voice agent on a Raspberry Pi 5",
        ha="center", va="center", fontsize=19, weight="bold", color=TXT)
ax.text(50, 69.2,
        "harness Pipecat 1.15.0   ·   ASR sherpa-onnx streaming zipformer   ·   "
        "LLM gemma4:31b-cloud   ·   TTS Piper en_US-ryan-medium   ·   VAD Silero ONNX",
        ha="center", va="center", fontsize=9.2, color=MUTED)

# --- zones -----------------------------------------------------------------
ax.add_patch(Rectangle((1.5, 10.0), 68.0, 56.5, facecolor=PI_FILL,
                       edgecolor=PI_EDGE, linewidth=1.8, zorder=1,
                       linestyle=(0, (7, 5))))
ax.text(3.2, 64.9, "ON THE RASPBERRY PI 5  —  8 GB RAM, 4 cores, no accelerator",
        ha="left", va="center", fontsize=11, weight="bold", color=PI_EDGE, zorder=5)

ax.add_patch(Rectangle((72.0, 10.0), 26.5, 56.5, facecolor=CLOUD_FILL,
                       edgecolor=CLOUD_EDGE, linewidth=1.8, zorder=1,
                       linestyle=(0, (7, 5))))
ax.text(73.4, 64.9, "CLOUD  —  reached over Wi-Fi",
        ha="left", va="center", fontsize=11, weight="bold", color=CLOUD_EDGE, zorder=5)

# ===========================================================================
# ROW 1 — audio in, left to right
# ===========================================================================
R1Y, R1H, R1W = 55.0, 8.0, 11.0
row1 = [
    (3.0,   "Mic",           "USB PnP card\n48 kHz mono"),
    (16.6,  "Resample",      "48 k → 16 k\nAudioResample\nProcessor"),
    (30.2,  "VAD",           "Silero ONNX\nconf 0.6 · start .25 s\nstop 0.7 s"),
    (43.8,  "ASR",           "sherpa-onnx\nstreaming zipformer\n16 k · 2 threads"),
    (57.4,  "ASR final pass", "faster-whisper\ntiny.en int8\n(~0.96 s)"),
]
for x, t, s in row1:
    box(x, R1Y, R1W, R1H, t, s)

for x0, x1 in [(14.0, 16.6), (27.6, 30.2), (41.2, 43.8), (54.8, 57.4)]:
    arrow((x0, R1Y + R1H / 2), (x1, R1Y + R1H / 2))

for x, t in [(15.3, "analog"), (28.9, "48 k"), (42.5, "16 k"), (56.1, "16 k")]:
    label(x, 64.0, t, fs=7.9)

# ===========================================================================
# ROW 2 — the agent
# ===========================================================================
R2Y, R2H = 41.0, 8.0
box(3.0, R2Y, 14.0, R2H, "Turn controller",
    "UserTurnController\nstart: VADUserTurnStart\nstop: SpeechTimeout 1.0 s", fc="#f8fafc")
box(20.1, R2Y, 14.0, R2H, "LLM client",
    "ClockedLLMService\ntemp 0.7 · 300 tok\nthinking off", fc="#f8fafc")
box(37.3, R2Y, 14.0, R2H, "TTS", "Piper\nen_US-ryan-medium\n48 kHz out")
box(54.4, R2Y, 14.0, R2H, "Speaker", "USB Composite\n48 kHz mono", fc="#f8fafc")

arrow((17.0, R2Y + R2H / 2), (20.1, R2Y + R2H / 2))
arrow((34.1, R2Y + R2H / 2), (37.3, R2Y + R2H / 2))
arrow((51.3, R2Y + R2H / 2), (54.4, R2Y + R2H / 2))

# transcript: ASR final pass -> turn controller (long diagonal in the empty band)
arrow((60.0, R1Y), (11.0, R2Y + R2H), rad=0.10)
label(19.5, 50.9, "final transcript (text, not audio)", fs=8.3, bg=PI_FILL)

# ===========================================================================
# BOTTOM-LEFT — MCP tool path
# ===========================================================================
MX, MY, MW, MH = 3.0, 14.0, 28.0, 10.0
ax.add_patch(FancyBboxPatch((MX, MY), MW, MH,
                            boxstyle="round,pad=0.16,rounding_size=0.65",
                            linewidth=1.5, facecolor=TOOL_FILL,
                            edgecolor=TOOL_EDGE, zorder=3))
ax.text(MX + MW / 2, MY + MH - 2.2, "MCP tool path", ha="center", va="center",
        fontsize=10.2, weight="bold", color=TOOL_EDGE, zorder=4)
ax.text(MX + MW / 2, MY + 3.5,
        "2 OnIt stdio servers\n14 tools, spawned as\nsubprocesses of OnIt's\ninterpreter",
        ha="center", va="center", fontsize=8.0, color="#4a3f6b",
        zorder=4, linespacing=1.5)

# LLM <-> MCP
arrow((24.0, R2Y), (24.0, MY + MH), color=TOOL_EDGE, lw=1.9, style="<|-|>")
label(25.4, 32.0, "function call ↓   result ↑", fs=8.2, col=TOOL_EDGE,
      ha="left", bg=PI_FILL)

# MCP -> cloud APIs (the servers are local, the APIs are not)
poly([(31.0, MY + 4.0), (68.0, MY + 4.0), (68.0, 33.0), (72.0, 33.0)],
     color=TOOL_EDGE, lw=1.5, ls=(0, (4, 3)))

# ===========================================================================
# BOTTOM-RIGHT — barge-in
# ===========================================================================
BX, BY, BW, BH = 34.5, 14.0, 33.9, 10.0
ax.add_patch(FancyBboxPatch((BX, BY), BW, BH,
                            boxstyle="round,pad=0.16,rounding_size=0.65",
                            linewidth=1.6, facecolor="#fdf0ee",
                            edgecolor=ACCENT, zorder=3))
ax.text(BX + BW / 2, BY + BH - 2.1, "BARGE-IN", ha="center", va="center",
        fontsize=10.6, weight="bold", color=ACCENT, zorder=4)
ax.text(BX + BW / 2, BY + 4.6,
        "speech detected → InterruptionFrame\nbroadcast both directions →\n"
        "TTS playback + LLM stream cancelled",
        ha="center", va="center", fontsize=8.0, color="#7a2b1f",
        zorder=4, linespacing=1.5)
ax.text(BX + BW / 2, BY + 1.1, "measured 0.48 s   ·   target < 0.80 s",
        ha="center", va="center", fontsize=8.4, color=ACCENT,
        weight="bold", zorder=4)

# VAD speech start -> barge-in (straight down the corridor between LLM and TTS)
arrow((35.7, R1Y), (35.7, BY + BH), color=ACCENT, lw=1.8, ls=(0, (5, 3)))
label(37.4, 51.4, "speech start", fs=7.8, col=ACCENT, ha="left", bg=PI_FILL)

# barge-in -> cancels LLM, cancels TTS
arrow((43.0, BY + BH), (32.0, R2Y), color=ACCENT, lw=1.8, ls=(0, (5, 3)), rad=0.12)
label(35.6, 36.6, "cancels LLM", fs=8.1, col=ACCENT, style="normal",
      weight="bold", bg=PI_FILL)
arrow((50.0, BY + BH), (45.5, R2Y), color=ACCENT, lw=1.8, ls=(0, (5, 3)), rad=-0.10)
label(48.6, 36.6, "cancels TTS", fs=8.1, col=ACCENT, style="normal",
      weight="bold", bg=PI_FILL)

# ===========================================================================
# CLOUD COLUMN
# ===========================================================================
box(73.2, R1Y, 24.1, R1H, "Ollama cloud API",
    "https://ollama.com/v1\nOpenAI-compatible · streaming",
    fc=CLOUD_FILL, ec=CLOUD_EDGE)

# LLM client -> cloud, elbow over the TTS/Speaker tops
poly([(30.0, R2Y + R2H), (30.0, 51.6), (73.2, 51.6)], color=CLOUD_EDGE, lw=1.9)
label(55.0, 53.1, "prompt + context out   ·   tokens back", fs=8.2, col=CLOUD_EDGE)

box(73.2, 29.0, 24.1, 8.0, "Third-party APIs",
    "web search · page fetch\nweather · GitHub REST",
    fc=CLOUD_FILL, ec=CLOUD_EDGE, fs=9.8)
label(85.2, 26.6, "called by the tool servers", fs=8.0, col=CLOUD_EDGE)

# ===========================================================================
# boundary callout
# ===========================================================================
label(36.0, 12.3, "▲  the only hop that leaves the Pi is the LLM call  ▲",
      fs=9.2, col=CLOUD_EDGE, style="normal", weight="bold")

ax.text(1.5, 5.4,
        "Source: agent.py:875-924 (stage order) · config.py:234 (LLM base URL) · "
        "config.py:305 (MCP servers) · config.py:44,45,126 (sample rates)",
        ha="left", va="center", fontsize=7.7, color=MUTED, style="italic")
ax.text(1.5, 3.2,
        "No AEC in this pipeline: the mic hears the speaker, so user frames are muted "
        "for the duration of a tool call (agent.py:807-820).",
        ha="left", va="center", fontsize=7.7, color=MUTED, style="italic")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=165, bbox_inches="tight", facecolor=BG)
print(f"wrote {OUT}")
