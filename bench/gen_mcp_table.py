"""Render "ME 3 Deck/MCP_Services.png" — the MCP services table as an image.

Rows mirror ME 3 Deck/MCP_Services.md, which in turn mirrors what the live bridge
reports. The 14 tool names and their server split were enumerated at runtime, not
typed by hand:

    mcp_bridge.build_bridge().discover()

Run from the repo root:  .venv/bin/python bench/gen_mcp_table.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "ME 3 Deck" / "MCP_Services.png"

BG = "#ffffff"
TXT = "#16202b"
MUTED = "#5b6b7c"
HDR_BG = "#3d5a80"
ROW_A = "#ffffff"
ROW_B = "#f4f7fb"
PI = "#4a7fb5"
CLOUD = "#c8792f"
NONE = "#8a97a4"

# (component, transport, tools, model, where, is_cloud, is_mcp)
ROWS = [
    ("ASR", "in-process\n(no MCP)", "—",
     "sherpa-onnx streaming\nzipformer en-2023-06-26\nint8 · 2 threads",
     "Pi · 16 kHz mono\nstreams partials; final text\nready BEFORE VAD end", False, False),
    ("ASR final pass", "in-process\n(no MCP)", "—",
     "faster-whisper tiny.en\nint8",
     "Pi · optional second pass\nTHIS is the 0.96 s,\nnot the recogniser", False, False),
    ("TTS", "in-process\n(no MCP)", "—",
     "Piper en_US-ryan-medium\n(ONNX)",
     "Pi · 48 kHz out\nstreams chunks; cancelled\non barge-in", False, False),
    ("VAD", "in-process\n(no MCP)", "—",
     "Silero ONNX (torch-free)",
     "Pi · 16 kHz\nconf 0.6 · start 0.25 s\nstop 0.7 s · 2 instances", False, False),
    ("LLM\n(harness MCP client)", "OpenAI-compatible\nHTTP", "function calling\nover the 14 tools",
     "gemma4:31b-cloud\ntemp 0.7 · 300 tok\nthinking off",
     "CLOUD · ollama.com/v1\nover Wi-Fi\nthe only hop off the Pi", True, False),
    ("Tool bridge\nToolsLocalMCPServer", "stdio", "bash · serve · read_file\nwrite_file · edit_file\ngrep · send_file · github_repo\nlocal_search · search_document\nindex_documents · fetch_content",
     "n/a — local\nexecution",
     "Pi · subprocess of OnIt's\ninterpreter · module tasks.tools\nprofile local · 12 tools", False, True),
    ("Tool bridge\nToolsNetMCPServer", "stdio", "search\nget_weather",
     "n/a — calls out",
     "Pi process, cloud APIs\n(ddgs / Tavily,\nOpenWeather) · 2 tools", False, True),
    ("Task tools", "—", "none", "—",
     "No task / home-automation\nMCP server exists\nin this build", False, False),
]

COLS = [
    ("Component", 15.5),
    ("MCP server / transport", 17.5),
    ("Tools", 25.0),
    ("Model", 21.0),
    ("Served where / notes", 25.5),
]
X0 = 2.0
ROW_H = 10.4
HDR_H = 4.4
TOTAL_W = sum(c[1] for c in COLS)

# one unit == one "tenth of an inch"; the axes span exactly the content height so
# there is no empty canvas below the last row.
TITLE_UNITS = 11.0
LEGEND_UNITS = 9.0
TOTAL_H = TITLE_UNITS + HDR_H + ROW_H * len(ROWS) + LEGEND_UNITS

fig_w = 17.0
fig_h = TOTAL_H / 10.0
fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=170)
ax.set_xlim(0, TOTAL_W + X0 * 2)
ax.set_ylim(0, TOTAL_H)
ax.axis("off")
fig.patch.set_facecolor(BG)

# title
top = TOTAL_H
ax.text(X0, top - 3.4, "WanDog — MCP services, transports and models",
        ha="left", va="center", fontsize=17, weight="bold", color=TXT)
ax.text(X0, top - 6.6,
        "Verified by enumerating the live bridge. ASR, TTS and VAD are in-process "
        "Pipecat services — not MCP servers. MCP carries only the 14 tools.",
        ha="left", va="center", fontsize=9.0, color=MUTED, style="italic")

y = top - TITLE_UNITS

# header
ax.add_patch(Rectangle((X0, y - HDR_H), TOTAL_W, HDR_H,
                       facecolor=HDR_BG, edgecolor=HDR_BG, zorder=2))
x = X0
for name, w in COLS:
    ax.text(x + 0.7, y - HDR_H / 2, name, ha="left", va="center",
            fontsize=9.3, weight="bold", color="#ffffff", zorder=3)
    x += w
y -= HDR_H

for i, (comp, transport, tools, model, where, is_cloud, is_mcp) in enumerate(ROWS):
    fill = ROW_B if i % 2 else ROW_A
    ax.add_patch(Rectangle((X0, y - ROW_H), TOTAL_W, ROW_H,
                           facecolor=fill, edgecolor="#d7dee6", linewidth=0.8,
                           zorder=2))
    # a colour stripe says "cloud" vs "Pi" vs "absent" at a glance
    stripe = CLOUD if is_cloud else (PI if not transport.startswith("—") or "in-process" in transport else NONE)
    if transport.startswith("—"):
        stripe = NONE
    ax.add_patch(Rectangle((X0, y - ROW_H), 0.42, ROW_H, facecolor=stripe,
                           edgecolor="none", zorder=3))

    cy = y - ROW_H / 2
    x = X0
    # component
    ax.text(x + 1.1, cy, comp, ha="left", va="center", fontsize=9.2,
            weight="bold", color=TXT, zorder=4, linespacing=1.45)
    x += COLS[0][1]
    # transport
    tcol = MUTED if transport.startswith(("in-process", "—")) else "#3a2f66"
    ax.text(x + 0.7, cy, transport, ha="left", va="center", fontsize=8.5,
            color=tcol, zorder=4, linespacing=1.5,
            style="italic" if transport.startswith(("in-process", "—")) else "normal")
    x += COLS[1][1]
    # tools
    ax.text(x + 0.7, cy, tools, ha="left", va="center", fontsize=7.5,
            color=TXT, zorder=4, linespacing=1.62)
    x += COLS[2][1]
    # model
    ax.text(x + 0.7, cy, model, ha="left", va="center", fontsize=7.9,
            color=TXT, zorder=4, linespacing=1.55)
    x += COLS[3][1]
    # where
    wcol = CLOUD if is_cloud else MUTED
    ax.text(x + 0.7, cy, where, ha="left", va="center", fontsize=7.9,
            color=wcol, zorder=4, linespacing=1.55,
            weight="bold" if is_cloud else "normal")
    y -= ROW_H

# legend
ly = y - 2.6
ax.add_patch(Rectangle((X0, ly), 0.42, 2.0, facecolor=PI, edgecolor="none"))
ax.text(X0 + 1.1, ly + 1.0, "runs on the Pi", ha="left", va="center",
        fontsize=8.4, color=MUTED)
ax.add_patch(Rectangle((X0 + 15.0, ly), 0.42, 2.0, facecolor=CLOUD, edgecolor="none"))
ax.text(X0 + 16.1, ly + 1.0, "leaves the Pi", ha="left", va="center",
        fontsize=8.4, color=MUTED)
ax.add_patch(Rectangle((X0 + 27.0, ly), 0.42, 2.0, facecolor=NONE, edgecolor="none"))
ax.text(X0 + 28.1, ly + 1.0, "does not exist in this build", ha="left", va="center",
        fontsize=8.4, color=MUTED)

ax.text(X0, ly - 3.0,
        "Sources: agent.py:690,735,764 (in-process services) · config.py:305 (MCP_SERVERS) · "
        "config.py:234 (LLM base URL) · runtime tool enumeration (14 tools: 12 local + 2 net)",
        ha="left", va="center", fontsize=7.8, color=MUTED, style="italic")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170, bbox_inches="tight", facecolor=BG)
print(f"wrote {OUT}")
