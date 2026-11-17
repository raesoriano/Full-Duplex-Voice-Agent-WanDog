"""Scale the two ASCII-art husky heads into the PNG icons the dashboard uses.

    assets/ascii-idle.png   -> husky-idle.png   (the bar's face, and the avatar)
    assets/ascii-think.png  -> husky-think.png  (the bar's face while busy)
    assets/ascii-idle.png   -> husky-32.png     (page icon)

The two `ascii-*.png` files are the sources: they are renders of the ASCII art,
one text glyph per cell, 100 columns by 55 rows on a 6.59 x 12.00 px cell. The
generated icons are what the dashboard loads; the sources are kept so the icons
can be rebuilt.

WHY THE RENDER IS DECODED BACK TO ITS CELL GRID
-----------------------------------------------
A glyph render is a halftone: the drawing's tone is carried by WHICH character
sits in each cell, so the ink inside a cell is a few thin strokes rather than a
flat patch. Downscaling that render directly keeps the halftone -- at 192 px the
character texture is visible, and by 30 px it has aliased into flecks that read
as dirt on the face. Measured on the idle head: direct downscale leaves ink in
0.078 of the face at 192 px against 0.167 in the source.

So each cell is averaged to a single density value first, which throws the
glyph shapes away and leaves the tone the artist drew, and that 100x55 grid is
then re-rendered as a continuous image. Re-rendering also restores the cell
aspect: the source render already stretches its cells (12.00/6.59 = 1.82, which
is the 1.84 the art implies), and the grid is re-rendered at the aspect measured
from the source, so the head keeps the shape it has rather than a squashed one.

WHY RASTERS, AND WHY TWO-TONE
-----------------------------
Rasters because the source is a raster, and tracing it to SVG produced blobs the
first time round (the eyes fused with the muzzle). Two-tone (white face under
dark ink) because dark ink alone is invisible on the dark theme and a white face
alone is invisible on the light one. The white backing is the silhouette the
border flood fill cannot reach.

Run:  .venv/bin/python tools/make_husky.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"

# The glyph grid of the source renders, as (columns, rows). Measured, not
# guessed: the cell boundaries sit on columns 0, 6.59, 13.18 ... and on rows
# 0, 12, 24 ..., which is where the all-white gutter lines land.
GRID = (100, 55)

# Pixels per cell when re-rendering the decoded grid. 16 px across gives the
# idle head ~1600 px of ink, 8x the largest shipped width, so the Lanczos
# downscale has plenty to work with.
CELL_PX = 16

# Density window mapped to full ink, after the grid is normalised to its own
# 99.5th percentile. Chosen so the ink coverage inside the silhouette stays flat
# across sizes instead of thinning out as the strokes get fewer pixels: measured
# 0.256 at 192 px and 0.268 at 30 px, against 0.167 of solid ink in the source.
INK_LO, INK_HI = 0.09, 0.80

# Air around the drawing, as a fraction of its longer side.
MARGIN = 0.045
LINE = (26, 25, 24)
FACE = (255, 255, 255)

# Closing kernel for the silhouette. Both heads close by 9 px and are flat after
# (idle 0.496 -> 0.508, think 0.322 -> 0.553 by k=9). The thinking head needs the
# full 9 because its jaw leaves a gap at the throat; below that the flood fill
# escapes and the white face never appears.
KERNEL = 9

ICON_WIDTH = 192   # the one size the dashboard loads, drawn at 26-30 px
FAVICON = ("ascii-idle", 32)   # (art, width) -> husky-32.png
ASPECT_TOLERANCE = 0.03
COVERAGE_TOLERANCE = 0.10


def load_ink(name: str) -> np.ndarray:
    """Ink coverage, 0..1, from a source render. White is no ink."""
    grey = np.asarray(Image.open(ASSETS / f"{name}.png").convert("L")).astype(np.float64)
    return np.clip((255.0 - grey) / 255.0, 0.0, 1.0)


def cell_aspect(ink: np.ndarray) -> float:
    """Height of one source cell as a multiple of its width.

    Read off the image rather than declared: the render stretched its cells, and
    re-rendering the decoded grid at any other aspect would squash the head.
    """
    rows, cols = GRID[1], GRID[0]
    return (ink.shape[0] / rows) / (ink.shape[1] / cols)


def to_grid(ink: np.ndarray) -> np.ndarray:
    """One density value per source cell: the halftone decoded back to tone."""
    rows, cols = GRID[1], GRID[0]
    ch, cw = ink.shape[0] / rows, ink.shape[1] / cols
    grid = np.zeros((rows, cols), np.float64)
    for r in range(rows):
        y0, y1 = int(round(r * ch)), int(round((r + 1) * ch))
        for c in range(cols):
            x0, x1 = int(round(c * cw)), int(round((c + 1) * cw))
            cell = ink[y0:y1, x0:x1]
            if cell.size:
                grid[r, c] = cell.mean()
    if not (grid > 0).any():
        raise ValueError("the source render has no ink")
    # Normalise to the art's own darkest tone, so the tone window below means the
    # same thing for both heads and does not drift if a render is re-exported.
    grid = grid / np.percentile(grid[grid > 0], 99.5)
    return np.clip(grid, 0.0, 1.0)


def silhouette(ink: np.ndarray) -> np.ndarray:
    """The head as a soft mask: everything a border flood fill cannot reach."""
    hard = (ink > 0.30).astype(np.uint8)
    # Pad before closing: the kernel grows the ink by about half its width, and
    # on an unpadded crop that seals the background into pockets the corner fill
    # can never enter, so the white face covers the whole square instead.
    pad = KERNEL
    padded = np.pad(hard, pad, mode="constant", constant_values=0)
    closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, np.ones((KERNEL, KERNEL), np.uint8))
    fill = closed.copy()
    mask = np.zeros((closed.shape[0] + 2, closed.shape[1] + 2), np.uint8)
    cv2.floodFill(fill, mask, (0, 0), 2)
    inside = (fill != 2).astype(np.uint8)[pad:-pad, pad:-pad]

    # One head, not a head plus a stray mark: a fleck beside the ear would
    # otherwise be filled white and become part of the face.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inside, connectivity=8)
    if n > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        inside = (labels == biggest).astype(np.uint8)

    soft = cv2.GaussianBlur(inside.astype(np.float32), (0, 0), CELL_PX * 0.19)
    return np.clip((soft - 0.35) / 0.30, 0.0, 1.0)


def rerender(grid: np.ndarray, aspect: float) -> np.ndarray:
    """Continuous ink from the decoded grid, at the source's own cell aspect."""
    rows, cols = grid.shape
    w = cols * CELL_PX
    h = max(1, round(rows * CELL_PX * aspect))
    up = Image.fromarray((grid * 255.0).astype(np.uint8)).resize((w, h), Image.BICUBIC)
    return np.asarray(up).astype(np.float64) / 255.0


def square(ink: np.ndarray, sil: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim to the drawing, then pad back to a square with a small margin.

    Squaring on the longer side makes the two heads read the same size: each
    fills the box along its own longer axis, so the bar's name plate never
    shifts when the face swaps.
    """
    ys, xs = np.nonzero(ink > 0.02)
    ink = ink[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    sil = sil[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = ink.shape
    side = max(h, w)
    margin = max(1, round(side * MARGIN))
    box = side + 2 * margin
    oi = np.zeros((box, box), np.float64)
    os_ = np.zeros((box, box), np.float64)
    top = (box - h) // 2
    left = (box - w) // 2
    oi[top:top + h, left:left + w] = ink
    os_[top:top + h, left:left + w] = sil
    return oi, os_


def compose(ink: np.ndarray, sil: np.ndarray) -> np.ndarray:
    """White face under dark ink, as RGBA."""
    rgb = (np.array(FACE, float) / 255.0) * sil[..., None] * (1.0 - ink[..., None]) \
        + (np.array(LINE, float) / 255.0) * ink[..., None]
    out = np.zeros(ink.shape + (4,), np.uint8)
    out[..., :3] = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
    out[..., 3] = np.clip(np.maximum(sil, ink) * 255.0, 0, 255).astype(np.uint8)
    return out


def scale(rgba: np.ndarray, width: int) -> np.ndarray:
    """Lanczos to a given width, height following the square."""
    h, w = rgba.shape[:2]
    height = max(1, round(width * h / w))
    return np.asarray(
        Image.fromarray(rgba, "RGBA").resize((width, height), Image.LANCZOS)
    )


def master(name: str) -> tuple[np.ndarray, float]:
    """The composed icon at render resolution, plus the source's ink aspect."""
    ink = load_ink(name)
    sil = silhouette(ink)
    # Drop anything the silhouette does not cover, so a stray mark beside the ear
    # cannot become part of the face. Boolean, not a multiply: multiplying by the
    # soft mask would fade the outline itself.
    ink = ink * (sil > 0.02)
    ys, xs = np.nonzero(ink > 0.30)
    source_aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)

    grid = to_grid(ink)
    drawn = rerender(grid, cell_aspect(ink))
    # The re-rendered grid has its own silhouette; take it from the same place so
    # the two masks line up with the ink they belong to.
    sil_drawn = np.asarray(
        Image.fromarray((sil * 255).astype(np.uint8)).resize(
            (drawn.shape[1], drawn.shape[0]), Image.LANCZOS
        )
    ).astype(np.float64) / 255.0
    oi, os_ = square(drawn, sil_drawn)
    ink_out = np.clip((oi - INK_LO) / (INK_HI - INK_LO), 0.0, 1.0) * os_
    return compose(ink_out, os_), source_aspect


def build(name: str) -> tuple[str, int, int]:
    """Write one face at icon size. Returns (file, w, h)."""
    rgba, source_aspect = master(name)
    img = scale(rgba, ICON_WIDTH)
    path = ASSETS / f"husky-{name.split('-', 1)[1]}.png"
    Image.fromarray(img, "RGBA").save(path)
    written = [(path.name, img.shape[1], img.shape[0])]

    # The drawn head must keep the shape the source art has. This is the check
    # that catches a re-render at the wrong cell aspect, which is a squashed
    # husky and is easy to ship without noticing.
    full = np.asarray(Image.open(ASSETS / written[0][0]).convert("RGBA")).astype(float)
    opaque = full[..., 3] > 128
    drawn = opaque & (full[..., :3].mean(axis=2) < 128)
    ys, xs = np.nonzero(drawn)
    got = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    if abs(got - source_aspect) / source_aspect > ASPECT_TOLERANCE:
        raise SystemExit(
            f"{name}: drawn aspect {got:.3f} but the source art is "
            f"{source_aspect:.3f} -- the cell aspect was not applied"
        )

    # Ink coverage must survive the downscale, or the strokes grey out at 30 px.
    # Measured against the opaque face, not the whole square (the square carries
    # a transparent margin): the icons this replaced sit at 0.55, and anything
    # much below that is a face whose strokes have washed out.
    cover = drawn.sum() / max(opaque.sum(), 1)
    if not (0.35 < cover < 0.70):
        raise SystemExit(f"{name}: ink covers {cover:.2f} of the face at {written[0][1]} px")
    return written[0]


def main() -> None:
    for icon in ("ascii-idle", "ascii-think"):
        name, w, h = build(icon)
        print(f"{icon:12s} -> {name} {w}x{h}")

    rgba, _ = master(FAVICON[0])
    fav = scale(rgba, FAVICON[1])
    Image.fromarray(fav, "RGBA").save(ASSETS / "husky-32.png")
    print(f"{FAVICON[0]:12s} -> husky-32.png {fav.shape[1]}x{fav.shape[0]}")


if __name__ == "__main__":
    main()
