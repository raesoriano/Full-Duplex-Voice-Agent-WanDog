"""Draw "ME 3 Deck/diagram.png" — WanDog's actual audio and tool path.

Everything drawn here is read from the code, not from the template:

  pipeline stage order .... agent.py:875-924
  cloud boundary .......... config.py:234 (LLM_BASE_URL)
  MCP transport ........... config.py:305 (MCP_SERVERS, stdio subprocesses)
  sample rates ............ config.py:44,45,126
  barge-in ................ agent.py:797-801
  no AEC / mute ........... agent.py:807-820

Run from the repo root:  .venv/bin/python bench/gen_diagram_v2.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "ME 3 Deck v2" / "diagram.png"

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

fig, ax = plt.subplots(figsize=(18.0, 13.5), dpi=165)
ax.set_xlim(0, 100)
ax.set_ylim(0, 82)
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
ax.text(50, 79.6, "WanDog — full-duplex voice agent on a Raspberry Pi 5",
        ha="center", va="center", fontsize=20, weight="bold", color=TXT)
ax.text(50, 76.8,
        "harness Pipecat 1.15.0   ·   ASR sherpa-onnx streaming zipformer   ·   "
        "LLM gemma4:31b-cloud   ·   TTS Piper en_US-ryan-medium   ·   VAD Silero ONNX",
        ha="center", va="center", fontsize=9.5, color=MUTED)

# --- zones -----------------------------------------------------------------
# Pi zone — left side, covers audio pipeline + MCP tool path + barge-in
PI_X, PI_Y = 1.5, 10.0
PI_W, PI_H = 63.0, 63.0
ax.add_patch(Rectangle((PI_X, PI_Y), PI_W, PI_H, facecolor=PI_FILL,
                       edgecolor=PI_EDGE, linewidth=1.8, zorder=1,
                       linestyle=(0, (7, 5))))
ax.text(3.2, PI_Y + PI_H - 2.0, "ON THE RASPBERRY PI 5  —  8 GB RAM, 4 cores, no accelerator",
        fontsize=9.5, weight="bold", color=PI_EDGE)

# Cloud zone — right side, clearly separated
CLOUD_X = 68.0
CLOUD_W = 30.5
ax.add_patch(Rectangle((CLOUD_X, PI_Y), CLOUD_W, PI_H, facecolor=CLOUD_FILL,
                       edgecolor=CLOUD_EDGE, linewidth=1.8, zorder=1,
                       linestyle=(0, (7, 5))))
ax.text(CLOUD_X + CLOUD_W / 2, PI_Y + PI_H - 2.0, "CLOUD  —  reached over Wi-Fi",
        fontsize=9.5, weight="bold", color=CLOUD_EDGE, ha="center")

# --- audio pipeline rows ---------------------------------------------------
R1Y = 56.0
R1H = 7.5
R2Y = 46.0
R2H = 7.5

# Row 1: Mic → Resample → VAD → ASR → ASR final pass
box(3.0, R1Y, 10.5, R1H, "Mic", "USB PnP card\n48 kHz mono")
box(15.0, R1Y, 10.5, R1H, "Resample", "48 kHz → 16 kHz\nSilero needs 8/16 k")
box(27.0, R1Y, 10.5, R1H, "VAD", "Silero ONNX\nconf 0.6 · start 0.25 s · stop 0.7 s")
box(39.0, R1Y, 10.5, R1H, "ASR", "sherpa-onnx streaming\nzipformer · 16 kHz\n2 ONNX threads")
box(52.0, R1Y, 10.5, R1H, "ASR final pass", "faster-whisper\ntiny.en int8\n0.96 s on this Pi", fc="#fff8f0")

arrow((14.5, R1Y + R1H / 2), (15.0, R1Y + R1H / 2))
arrow((25.5, R1Y + R1H / 2), (27.0, R1Y + R1H / 2))
arrow((37.5, R1Y + R1H / 2), (39.0, R1Y + R1H / 2))
arrow((49.5, R1Y + R1H / 2), (52.0, R1Y + R1H / 2))

# Row 2: Turn controller → LLM client → TTS → Speaker
box(3.0, R2Y, 14.0, R2H, "Turn controller",
    "UserTurnController\nstart: VADUserTurnStart\nstop: SpeechTimeout 1.0 s", fc="#f8fafc")
box(18.5, R2Y, 14.0, R2H, "LLM client",
    "ClockedLLMService\ntemp 0.7 · 300 tok\nthinking off", fc="#f8fafc")
box(35.0, R2Y, 14.0, R2H, "TTS", "Piper\nen_US-ryan-medium\n48 kHz out")
box(51.5, R2Y, 14.0, R2H, "Speaker", "USB Composite\n48 kHz mono", fc="#f8fafc")

arrow((17.5, R2Y + R2H / 2), (18.5, R2Y + R2H / 2))
arrow((34.0, R2Y + R2H / 2), (35.0, R2Y + R2H / 2))
arrow((50.0, R2Y + R2H / 2), (51.5, R2Y + R2H / 2))

# transcript: ASR final pass -> turn controller (long diagonal in the empty band)
arrow((62.5, R1Y), (11.0, R2Y + R2H), rad=0.10)
label(22.0, 53.5, "final transcript (text, not audio)", fs=8.5, bg=PI_FILL)

# ===========================================================================
# BOTTOM-LEFT — MCP tool path (on Pi)
# ===========================================================================
MX, MY, MW, MH = 3.0, 14.0, 28.0, 11.0
ax.add_patch(FancyBboxPatch((MX, MY), MW, MH,
                            boxstyle="round,pad=0.16,rounding_size=0.65",
                            linewidth=1.5, facecolor=TOOL_FILL, edgecolor=TOOL_EDGE,
                            linestyle=(0, (5, 3)), zorder=2))
ax.text(MX + 1.0, MY + MH - 1.2, "MCP TOOL PATH", fontsize=8.5, weight="bold",
        color=TOOL_EDGE)

box(MX + 1.0, MY + 3.0, 12.5, 6.5, "ToolsLocalMCPServer",
    "read_file · write_file\nedit · grep · bash\nAST allowlist · approval gate",
    fc="#fff", ec=TOOL_EDGE, fs=9.5, subfs=7.5)
box(MX + 15.0, MY + 3.0, 12.0, 6.5, "ToolsNetMCPServer",
    "search · fetch_content\nget_weather · github_repo\n14 tools total",
    fc="#fff", ec=TOOL_EDGE, fs=9.5, subfs=7.5)

# LLM client -> MCP servers (dashed, within Pi)
poly([(25.5, R2Y), (25.5, MY + MH + 0.5), (17.0, MY + MH + 0.5), (17.0, MY + MH)],
     color=TOOL_EDGE, lw=1.6, ls=(0, (5, 3)), head=False)
label(22.0, MY + MH + 2.0, "tool calls", fs=8.0, col=TOOL_EDGE, ha="center", bg=PI_FILL)

# ===========================================================================
# BOTTOM-RIGHT — Barge-in (on Pi)
# ===========================================================================
BX, BY, BW, BH = 34.0, 14.0, 28.0, 11.0
ax.add_patch(FancyBboxPatch((BX, BY), BW, BH,
                            boxstyle="round,pad=0.16,rounding_size=0.65",
                            linewidth=1.5, facecolor="#fff5f5", edgecolor=ACCENT,
                            linestyle=(0, (5, 3)), zorder=2))
ax.text(BX + 1.0, BY + BH - 1.2, "BARGE-IN", fontsize=8.5, weight="bold",
        color=ACCENT)

box(BX + 1.0, BY + 3.0, 12.5, 6.5, "InterruptionFrame",
    "VAD turn start fires\non first speech frame\n0.48 s to cancel",
    fc="#fff", ec=ACCENT, fs=9.5, subfs=7.5)
box(BX + 15.5, BY + 3.0, 11.5, 6.5, "FunctionCallUserMuteStrategy",
    "mutes user frames\nduring tool calls\nprevents self-echo",
    fc="#fff", ec=ACCENT, fs=9.5, subfs=7.5)

# VAD speech start -> barge-in (straight down)
arrow((37.2, R1Y), (37.2, BY + BH), color=ACCENT, lw=1.8, ls=(0, (5, 3)))
label(39.0, 54.0, "speech start", fs=8.0, col=ACCENT, ha="left", bg=PI_FILL)

# barge-in -> cancels LLM, cancels TTS
arrow((43.5, BY + BH), (31.5, R2Y), color=ACCENT, lw=1.8, ls=(0, (5, 3)), rad=0.12)
label(36.5, 39.0, "cancels LLM", fs=8.2, col=ACCENT, style="normal",
      weight="bold", bg=PI_FILL)
arrow((50.0, BY + BH), (43.0, R2Y), color=ACCENT, lw=1.8, ls=(0, (5, 3)), rad=-0.10)
label(47.0, 39.0, "cancels TTS", fs=8.2, col=ACCENT, style="normal",
      weight="bold", bg=PI_FILL)

# ===========================================================================
# CLOUD COLUMN — clearly outside the Pi box
# ===========================================================================

# Ollama cloud API — top of cloud zone
box(CLOUD_X + 2.5, R1Y, 25.5, R1H, "Ollama cloud API",
    "https://ollama.com/v1\nOpenAI-compatible · streaming",
    fc=CLOUD_FILL, ec=CLOUD_EDGE)

# LLM client -> cloud, elbow going right into cloud zone
poly([(30.0, R2Y + R2H), (30.0, R2Y + R2H + 2.0), (CLOUD_X, R2Y + R2H + 2.0), (CLOUD_X, 56.0)],
     color=CLOUD_EDGE, lw=2.0)
label(38.0, R2Y + R2H + 3.5, "prompt + context out   ·   tokens back", fs=8.5, col=CLOUD_EDGE)

# Third-party APIs — below Ollama in cloud zone
box(CLOUD_X + 2.5, 38.0, 25.5, 9.0, "Third-party APIs",
    "web search · page fetch\nweather · GitHub REST",
    fc=CLOUD_FILL, ec=CLOUD_EDGE, fs=10.5)

# MCP tool servers -> Third-party APIs (crossing from Pi to Cloud)
# Arrow goes from MCP box right edge, across the boundary, into cloud
poly([(MX + MW, MY + MH / 2), (CLOUD_X, MY + MH / 2), (CLOUD_X, 42.5)],
     color=CLOUD_EDGE, lw=1.8, ls=(0, (5, 3)))
label((MX + MW + CLOUD_X) / 2, MY + MH / 2 + 1.5, "tool servers call cloud APIs",
      fs=8.2, col=CLOUD_EDGE, bg=PI_FILL)

label(CLOUD_X + CLOUD_W / 2, 33.5, "called by the tool servers", fs=8.2, col=CLOUD_EDGE)

# ===========================================================================
# boundary callout
# ===========================================================================
label(34.0, 9.0, "▲  the only hop that leaves the Pi is the LLM call  ▲",
      fs=9.5, col=CLOUD_EDGE, style="normal", weight="bold")

ax.text(1.5, 5.8,
        "Source: agent.py:875-924 (stage order) · config.py:234 (LLM base URL) · "
        "config.py:305 (MCP servers) · config.py:44,45,126 (sample rates)",
        ha="left", va="center", fontsize=7.8, color=MUTED, style="italic")
ax.text(1.5, 3.5,
        "No AEC in this pipeline: the mic hears the speaker, so user frames are muted "
        "for the duration of a tool call (agent.py:807-820).",
        ha="left", va="center", fontsize=7.8, color=MUTED, style="italic")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=165, bbox_inches="tight", facecolor=BG)
print(f"wrote {OUT}")
