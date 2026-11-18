"""Geometric check on the deck images: no text overlaps, nothing off-canvas.

Rendered images can only be judged by looking, and a script cannot look. What it
*can* do is measure every text artist's bounding box in pixels and report the
two failures that actually happen when a layout is re-sized:

  * two labels overlapping (the v1 diagram had this)
  * a label running past the canvas edge (the v1 table had this)

Run from the repo root:

    .venv/bin/python bench/check_layout.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_deck_v2 as g  # noqa: E402

_orig_close = plt.close
plt.close = lambda *a, **k: None          # keep the figure for inspection


def boxes(fig):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    out = []
    for ax in fig.axes:
        for t in ax.texts:
            s = t.get_text().strip()
            if not s:
                continue
            bb = t.get_window_extent(renderer=r)
            out.append((s, bb))
    return out


def check(name, fn):
    plt.close("all")
    fn()
    fig = plt.figure(plt.get_fignums()[-1]) if plt.get_fignums() else None
    fig = plt.gcf()
    W, H = fig.canvas.get_width_height()
    bs = boxes(fig)
    problems = []

    for s, bb in bs:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            problems.append(f"off-canvas: {s[:44]!r} "
                            f"x {bb.x0:.0f}..{bb.x1:.0f} of {W}, "
                            f"y {bb.y0:.0f}..{bb.y1:.0f} of {H}")

    for i in range(len(bs)):
        for j in range(i + 1, len(bs)):
            a, b = bs[i][1], bs[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0)
            oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > 2 and oy > 2:
                problems.append(f"overlap: {bs[i][0][:30]!r} / {bs[j][0][:30]!r} "
                                f"({ox:.0f} x {oy:.0f} px)")

    # smallest glyph height, as a proxy for "readable when projected"
    sizes = sorted({round(t.get_fontsize(), 1) for ax in fig.axes for t in ax.texts})
    print(f"{name:<18} {W}x{H} px  {len(bs):>2} labels  "
          f"font sizes {sizes[0]}-{sizes[-1]} pt  "
          f"smallest ~{sizes[0] / 72 * 150:.0f} px tall")
    for p in problems:
        print("   ", p)
    if not problems:
        print("    clean")
    return problems


def main() -> int:
    all_p = []
    for name, fn in (("diagram", g.draw_diagram),
                     ("MCP_Services", g.draw_mcp_table),
                     ("Benchmark", g.draw_bench_table),
                     ("latency_chart", g.draw_chart)):
        all_p += check(name, fn)
    print()
    print("LAYOUT OK" if not all_p else f"{len(all_p)} layout problem(s)")
    return 1 if all_p else 0


if __name__ == "__main__":
    raise SystemExit(main())
