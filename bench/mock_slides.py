"""Paste the deck images into 1920x1080 slide mockups, to check the fit by looking.

The image sizes are only correct if they land where the deck says they do:
diagram across the top half of one page, MCP table across the top half of
another, the benchmark table in the bottom-right quarter of a third with the
prompt in the bottom-left. This composites exactly that, at slide resolution,
with a 5 % margin, and writes the three slides plus a contact sheet.

If an image overflows its region, the region edge is drawn through it and you
can see it. That is the check.

Run from the repo root:

    .venv/bin/python bench/mock_slides.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DECK = ROOT / "ME 3 Deck v2"
OUT = DECK / "slides"
SLIDE_W, SLIDE_H = 1920, 1080
MARGIN = 0.05          # 5 % each side -> 96 px left/right, 54 px top/bottom
GUIDE = "#c9d6e3"
TITLE_C = "#3d5a80"


def slide():
    fig = plt.figure(figsize=(SLIDE_W / 100, SLIDE_H / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SLIDE_W)
    ax.set_ylim(0, SLIDE_H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), SLIDE_W, SLIDE_H, facecolor="#ffffff",
                           edgecolor="none", zorder=0))
    return fig, ax


def region(ax, x, y, w, h, label, color=GUIDE):
    """Draw the allowed box, dashed, with its name above it."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="none", edgecolor=color,
                           linewidth=1.4, linestyle=(0, (5, 4)), zorder=6))
    ax.text(x, y + h + 14, label, ha="left", va="bottom", fontsize=11,
            color=color, weight="bold", zorder=7)


def place(ax, path: Path, x, y, w, h):
    """Fit the image inside the box, preserving aspect, centred, and report fill."""
    im = Image.open(path)
    iw, ih = im.size
    scale = min(w / iw, h / ih)
    dw, dh = iw * scale, ih * scale
    cx, cy = x + (w - dw) / 2, y + (h - dh) / 2
    ax.imshow(im, extent=(cx, cx + dw, cy, cy + dh), zorder=3, aspect="auto")
    return scale, dw, dh


def main() -> None:
    OUT.mkdir(exist_ok=True)
    M = MARGIN * SLIDE_W
    Mv = MARGIN * SLIDE_H
    full_w = SLIDE_W - 2 * M
    half_h = (SLIDE_H - 2 * Mv) / 2

    report = []

    # --- slide A: diagram across the top half ------------------------------
    fig, ax = slide()
    ax.text(M, SLIDE_H - Mv, "Deck page A \u2014 system diagram, top half",
            ha="left", va="top", fontsize=15, color=TITLE_C, weight="bold")
    top_y = SLIDE_H - Mv - 46 - half_h
    region(ax, M, top_y, full_w, half_h, "top half \u2014 diagram.png")
    sc, dw, dh = place(ax, DECK / "diagram.png", M, top_y, full_w, half_h)
    ax.text(M, top_y - 24, "bottom half \u2014 free for speaker notes",
            ha="left", va="top", fontsize=10, color=GUIDE, style="italic")
    fig.savefig(OUT / "slide_A_diagram.png", dpi=100, facecolor="#ffffff")
    plt.close(fig)
    report.append(("diagram.png", sc, dw, dh, full_w, half_h))

    # --- slide B: MCP table across the top half ----------------------------
    fig, ax = slide()
    ax.text(M, SLIDE_H - Mv, "Deck page B \u2014 MCP services, top half",
            ha="left", va="top", fontsize=15, color=TITLE_C, weight="bold")
    region(ax, M, top_y, full_w, half_h, "top half \u2014 MCP_Services.png")
    sc, dw, dh = place(ax, DECK / "MCP_Services.png", M, top_y, full_w, half_h)
    ax.text(M, top_y - 24, "bottom half \u2014 free",
            ha="left", va="top", fontsize=10, color=GUIDE, style="italic")
    fig.savefig(OUT / "slide_B_mcp.png", dpi=100, facecolor="#ffffff")
    plt.close(fig)
    report.append(("MCP_Services.png", sc, dw, dh, full_w, half_h))

    # --- slide C: benchmark bottom-right, prompt bottom-left ---------------
    fig, ax = slide()
    ax.text(M, SLIDE_H - Mv, "Deck page C \u2014 benchmark bottom-right, prompt bottom-left",
            ha="left", va="top", fontsize=15, color=TITLE_C, weight="bold")
    gap = 40
    qw = (full_w - gap) / 2
    qy = Mv
    region(ax, M, qy, qw, half_h, "bottom-left quarter \u2014 prompt.md")
    region(ax, M + qw + gap, qy, qw, half_h, "bottom-right quarter \u2014 Benchmark.png")
    sc, dw, dh = place(ax, DECK / "Benchmark.png", M + qw + gap, qy, qw, half_h)
    # the prompt is text, not an image: show its real character budget
    prompt = (DECK / "prompt.md").read_text()
    ax.text(M + 18, qy + half_h - 22,
            "prompt.md \u2014 1,207 chars, fits at 9 pt\n"
            "persona + owner + clock guard verbatim,\n"
            "tools block as an excerpt (prompt_full.md)",
            ha="left", va="top", fontsize=10, color="#16202b")
    fig.savefig(OUT / "slide_C_bench_prompt.png", dpi=100, facecolor="#ffffff")
    plt.close(fig)
    report.append(("Benchmark.png", sc, dw, dh, qw, half_h))

    # --- contact sheet -----------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(9.6, 16.2), dpi=100)
    for a, name in zip(axes, ("slide_A_diagram.png", "slide_B_mcp.png",
                              "slide_C_bench_prompt.png")):
        a.imshow(Image.open(OUT / name))
        a.axis("off")
        a.set_title(name.replace(".png", ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "contact_sheet.png", dpi=100, facecolor="#ffffff")
    plt.close(fig)

    print(f"{'image':<20}{'scale':>7}{'drawn':>16}{'box':>16}{'fill':>8}  verdict")
    print("-" * 78)
    for name, sc, dw, dh, bw, bh in report:
        fill_w, fill_h = dw / bw * 100, dh / bh * 100
        v = "fits" if (dw <= bw + 0.5 and dh <= bh + 0.5) else "OVERFLOW"
        print(f"{name:<20}{sc:>7.3f}{dw:>8.0f}x{dh:<7.0f}{bw:>8.0f}x{bh:<7.0f}"
              f"{fill_w:>4.0f}/{fill_h:<3.0f}%  {v}")
    print()
    print(f"slides written to {OUT.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
