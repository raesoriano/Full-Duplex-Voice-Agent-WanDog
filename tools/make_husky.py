"""Scale the two uploaded husky drawings into the PNG icons the dashboard uses.

    assets/husky-idle2.png  -> husky-idle.png   (the bar's face, and the avatar)
    assets/husky-think3.png -> husky-think.png  (the bar's face while busy)
    assets/husky-idle2.png  -> husky-32.png     (page icon)

The uploads are finished drawings, so this is a plain resize: nothing is
decoded, traced or re-rendered. The one thing that needs care is the alpha
channel -- both drawings carry fully transparent pixels whose RGB is black, and
resampling straight RGBA drags that black into the edge, leaving a dark halo
around a face that is only ever drawn 26-30 px wide. Premultiplying before the
Lanczos pass and dividing it back out afterwards keeps the edge clean.

Run:  .venv/bin/python tools/make_husky.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "assets"

# (source upload, icon basename) -> husky-<basename>.png
SOURCES = (
    ("husky-idle2", "idle"),
    ("husky-think3", "think"),
)

ICON_WIDTH = 192   # the one size the dashboard loads, drawn at 26-30 px
FAVICON = ("husky-idle2", 32)   # (source, width) -> husky-32.png


def scale(source: str, width: int) -> Image.Image:
    """Lanczos to `width`, premultiplied so the transparent edge stays clean."""
    img = np.asarray(Image.open(ASSETS / f"{source}.png").convert("RGBA")).astype(np.float64)
    alpha = img[..., 3:4] / 255.0
    premultiplied = np.concatenate([img[..., :3] * alpha, img[..., 3:4]], axis=2)

    height = max(1, round(width * img.shape[0] / img.shape[1]))
    small = np.asarray(
        Image.fromarray(premultiplied.round().astype(np.uint8), "RGBA").resize(
            (width, height), Image.LANCZOS
        )
    ).astype(np.float64)

    alpha = small[..., 3:4] / 255.0
    rgb = np.where(alpha > 0, small[..., :3] / np.maximum(alpha, 1e-6), 0.0)
    out = np.concatenate([np.clip(rgb, 0, 255), small[..., 3:4]], axis=2)
    return Image.fromarray(out.round().astype(np.uint8), "RGBA")


def main() -> None:
    for source, name in SOURCES:
        img = scale(source, ICON_WIDTH)
        img.save(ASSETS / f"husky-{name}.png", optimize=True)
        print(f"{source:12s} -> husky-{name}.png {img.width}x{img.height}")

    fav = scale(*FAVICON)
    fav.save(ASSETS / "husky-32.png", optimize=True)
    print(f"{FAVICON[0]:12s} -> husky-32.png {fav.width}x{fav.height}")


if __name__ == "__main__":
    main()
