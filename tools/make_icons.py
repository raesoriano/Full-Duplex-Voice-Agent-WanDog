"""Regenerate assets/husky.png and husky-32.png from assets/husky.svg.

The favicon has to be raster (Streamlit's page_icon wants a file it can hand to
the browser), so the SVG is rendered once at each size with Chromium and cropped
to its own bounding box. The SVG is the source of truth; these two files are
outputs, not artwork.

Usage: python tools/make_icons.py
"""
from __future__ import annotations

import pathlib
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image

ASSETS = pathlib.Path(__file__).resolve().parent.parent / "assets"
SVG = ASSETS / "husky.svg"
SIZES = {"husky.png": 512, "husky-32.png": 32}


def render(size: int) -> Image.Image:
    svg = SVG.read_text()
    ET.fromstring(svg)  # an unparseable SVG renders as nothing at all
    inner = svg.split("?>", 1)[1].strip()
    with tempfile.TemporaryDirectory() as tmp:
        page = pathlib.Path(tmp) / "page.html"
        png = pathlib.Path(tmp) / "out.png"
        # A transparent page background is what makes the alpha channel real;
        # without it the screenshot has an opaque white box behind the dog.
        page.write_text(
            "<!doctype html><html><head><meta charset='utf-8'><style>"
            "html,body{margin:0;padding:0;background:transparent;}"
            "svg{display:block;width:100vw;height:100vh;}"
            "</style></head><body>" + inner + "</body></html>"
        )
        subprocess.run(
            ["chromium", "--headless=new", "--disable-gpu", "--no-sandbox",
             "--hide-scrollbars", "--default-background-color=00000000",
             f"--window-size={size},{size}", f"--screenshot={png}",
             f"file://{page}"],
            check=True, capture_output=True,
        )
        im = Image.open(png).convert("RGBA")
    # The SVG's viewBox is 64 with the art inside it, so the raw render has a
    # margin. Crop to the ink and pad a little, which is what makes the dog fill
    # a round avatar instead of floating in it.
    bbox = im.getbbox()
    if bbox:
        im = im.crop(bbox)
    side = max(im.size)
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(im, ((side - im.width) // 2, (side - im.height) // 2), im)
    return canvas


def main() -> int:
    for name, size in SIZES.items():
        im = render(size * 2 if size < 64 else size).resize(
            (size, size), Image.LANCZOS
        )
        out = ASSETS / name
        im.save(out)
        print(f"wrote {out} ({size}x{size}, {out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
