"""Cut the two husky heads out of the reference sheet and turn them into icons.

The sheet is a grid of hand-drawn husky heads on white. Two cells are used:

    row 2, column 1  -- mouth open, tongue out, facing front  -> husky-idle.png
    row 3, column 4  -- muzzle up, "thinking"                  -> husky-think.png

WHY THESE ARE RASTERS AND NOT TRACED SVG
----------------------------------------
The previous icon was an SVG I drew by hand, and it was traced by contour. That
worked because the shapes were mine: clean ellipses and beziers. These are a
photograph of a drawing, ~110x120 px of JPEG, and contour tracing at that
resolution produced blobs -- the outline broke into fragments, the eyes fused
with the muzzle, and the result was worse than a plain downscale. So the icons
are rasters, cut with an alpha matte, at 4x the largest size they are drawn at
(the top bar shows them at 30 px, the chat avatar at 26 px).

THE MATTE
----------
The drawing is dark ink on white paper, so ink density is a linear ramp of the
inverse grey level: alpha = (232 - v) / (232 - 70), gamma 0.85 to keep the thin
lines from greying out. That single number is used twice:

  * as the ink's own alpha -- so the lines stay anti-aliased at every size;
  * thresholded at 0.42 and flood-filled from the border to find the SILHOUETTE,
    which is painted white underneath. Without it the icon is invisible on the
    dark theme (dark ink on a dark page) and on the light theme it is only an
    outline. With it, the white face carries the shape on dark and the dark
    outline carries it on light. Both are checked in test_dashboard_layout.py.

The flood fill is done on a closed copy of the thresholded ink: the outline has
pinholes at this resolution, and a leak lets the fill escape and erase the face.

Run:  .venv/bin/python tools/make_husky.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"
SHEET = ASSETS / "reference" / "husky-sheet.jpg"

# Cell boundaries, measured from the white gutters between the heads (gutters at
# x = 180-220, 339-383, 515-553 and y = 151-179, 290-316, 426-453).
# x0, x1, y0, y1, and the kernel that closes the outline before the fill.
#
# The kernel is per-icon because the two drawings are not equally closed. The
# idle head's outline is one continuous stroke and closes at 9 px; the thinking
# head's jaw leaves a gap at the throat that needs 13 px. Measured as the point
# where the filled area stops growing: idle 0.858 -> 0.923 between k=7 and k=9
# and then flat; think 0.651 -> 0.767 -> 0.853 between k=7, 9 and 11, flat after
# 13. Below those numbers the fill leaks out through the gap and the white face
# never appears -- which is what the first version of this file did.
CELLS = {
    "husky-idle": (58, 171, 172, 297, 9),
    "husky-think": (560, 685, 309, 429, 13),
}

SIZES = {"": 192, "-64": 64}   # suffix -> width in px
PAD = 3                        # px of air around the ink, in source pixels
HI, LO = 232.0, 70.0           # grey levels mapped to alpha 0 and 1
GAMMA = 0.85                   # >1 thins the lines, <1 fattens them
SIL_THRESHOLD = 0.42
MARGIN = 0.04          # air around the drawing, as a fraction of its longer side
LINE = (31, 30, 29)            # the dark, near-black rather than #000
FACE = (255, 255, 255)


def _ink(cell: np.ndarray) -> np.ndarray:
    """Ink alpha, 0..1, cropped to the drawing with PAD of air."""
    alpha = np.clip((HI - cell.astype(np.float64)) / (HI - LO), 0.0, 1.0) ** GAMMA
    ys, xs = np.nonzero(alpha > 0.08)
    y0 = max(0, int(ys.min()) - PAD)
    y1 = min(cell.shape[0], int(ys.max()) + 1 + PAD)
    x0 = max(0, int(xs.min()) - PAD)
    x1 = min(cell.shape[1], int(xs.max()) + 1 + PAD)
    return alpha[y0:y1, x0:x1]


def _silhouette(alpha: np.ndarray, kernel: int) -> np.ndarray:
    """The head as a soft-edged mask: everything the border fill cannot reach."""
    hard = (alpha > SIL_THRESHOLD).astype(np.uint8)

    # PAD FIRST, then close. The closing kernel grows the ink by about half its
    # width in every direction, so on a crop that is only PAD px of air the
    # outline reaches the border and seals the surrounding background into
    # pockets the corner fill can never enter. Measured with a 9 px kernel on a
    # 3 px pad: only 1070 of ~8000 background pixels were reached, the rest stayed
    # unfilled and were then counted as "inside", so the white face covered the
    # entire rectangle and the icon became a white square.
    pad = kernel
    padded = np.pad(hard, pad, mode="constant", constant_values=0)

    # Close the pinholes, or the fill leaks in and the face disappears.
    closed = cv2.morphologyEx(
        padded, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8)
    )
    fill = closed.copy()
    mask = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), np.uint8)
    cv2.floodFill(fill, mask, (0, 0), 2)
    inside = (fill != 2).astype(np.uint8)[pad:-pad, pad:-pad]

    # One head, not a head plus stray marks: a fleck beside the ear would
    # otherwise be filled white and become part of the face.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inside, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        inside = (labels == biggest).astype(np.uint8)

    soft = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), 1.0)
    return np.clip((soft - 0.35) / 0.30, 0.0, 1.0)


def _resize(a: np.ndarray, width: int) -> np.ndarray:
    h, w = a.shape
    height = max(1, round(width * h / w))
    im = Image.fromarray((a * 255.0).astype(np.uint8)).resize(
        (width, height), Image.LANCZOS
    )
    return np.asarray(im).astype(np.float64) / 255.0


def build(name: str, box: tuple[int, int, int, int, int]) -> dict:
    sheet = np.asarray(Image.open(SHEET).convert("L"))
    x0, x1, y0, y1, kernel = box
    alpha = _ink(sheet[y0:y1, x0:x1])
    sil = _silhouette(alpha, kernel)

    made = {}
    for suffix, width in SIZES.items():
        a = _resize(alpha, width)
        s = _resize(sil, width)
        out = _compose(a, s)
        path = ASSETS / f"{name}{suffix}.png"
        Image.fromarray(out, "RGBA").save(path, optimize=True)
        made[suffix] = (path, out.shape[1], out.shape[0])
    return {"name": name, "made": made, "source": (x1 - x0, y1 - y0)}


def _compose(ink: np.ndarray, sil: np.ndarray) -> np.ndarray:
    """White face under dark ink, trimmed to the drawing, then squared.

    The trim is not cosmetic. The source crop is taken with a fixed PAD of air,
    but the ink's faintest edge and the silhouette's soft edge disagree by a
    pixel or two, so the composed image reaches its own border -- measured on the
    first cut: 19 opaque pixels on the bottom row of the idle icon, which in a
    30 px round avatar reads as a cropped chin. Trimming to the drawing and then
    padding to a square guarantees air on all four sides, and a square frame is
    also what keeps the two faces the same size when the bar swaps between them.
    """
    h, w = ink.shape
    out = np.zeros((h, w, 4), np.uint8)
    face = np.array(FACE, float) / 255.0
    line = np.array(LINE, float) / 255.0
    rgb = face * sil[..., None] * (1.0 - ink[..., None]) + line * ink[..., None]
    out[..., :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.maximum(sil, ink) * 255.0, 0, 255).astype(np.uint8)

    drawn = out[..., 3] > 2
    ys, xs = np.nonzero(drawn)
    out = out[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]

    side = max(out.shape[0], out.shape[1])
    margin = max(1, round(side * MARGIN))
    box = side + 2 * margin
    canvas = np.zeros((box, box, 4), np.uint8)
    top = (box - out.shape[0]) // 2
    left = (box - out.shape[1]) // 2
    canvas[top : top + out.shape[0], left : left + out.shape[1]] = out
    return canvas


def main() -> int:
    if not SHEET.exists():
        print(f"missing reference sheet: {SHEET}", file=sys.stderr)
        return 2
    for name, box in CELLS.items():
        info = build(name, box)
        sizes = ", ".join(f"{p.name} {w}x{h}" for p, w, h in info["made"].values())
        print(f"{name}: from {info['source'][0]}x{info['source'][1]}  ->  {sizes}")

    # The favicon has to be raster and small; it is the idle head.
    fav = ASSETS / "husky-32.png"
    Image.open(ASSETS / "husky-idle-64.png").resize((32, 32), Image.LANCZOS).save(fav)
    print(f"favicon: {fav.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
