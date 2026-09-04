#!/usr/bin/env python3
"""Cut a region of the A1 poster into 1:1 A4 tiles you can actually print.

Why this exists
---------------
Every check up to now has been a screenshot: the whole A1 scaled onto a monitor.
That answers "is the layout balanced" and cannot answer "can this be read from a
metre away", because the sheet is never at its real size. The only honest test is
paper at 1:1, and the only 1:1 paper most people have is A4.

So this cuts a named region of the poster — by default the six results panels,
the part whose type is smallest — into A4 landscape tiles at true scale. Print
them at **100% / Actual size** (NOT "fit to page", which silently rescales and
destroys the entire point), lay them out, and stand back.

The judgement it is for, from the author's own brief:

    without reading the small type — heading, figure shape, one caption —
    can you say what each panel concludes?

Usage::

    .venv/bin/python3 deliverables/showcase/print_test_tiles.py
    .venv/bin/python3 deliverables/showcase/print_test_tiles.py --top 132 --bottom 420
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PageObject, PdfReader, PdfWriter, Transformation

HERE = Path(__file__).resolve().parent
SRC = HERE / "poster_v9_jiaming_wei.pdf"
OUT = HERE / "print_test_tiles.pdf"

MM = 72 / 25.4
A4_W, A4_H = 297 * MM, 210 * MM      # landscape
MARGIN = 8 * MM                       # printers need some; keeps the scale honest
SHEET_H = 841                         # A1 portrait, mm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=SRC)
    ap.add_argument("--out", type=Path, default=OUT)
    # defaults frame the six results panels, whose type is the smallest on the sheet
    ap.add_argument("--left", type=float, default=20.2)
    ap.add_argument("--right", type=float, default=577.5)
    ap.add_argument("--top", type=float, default=520.0, help="mm from the sheet's top")
    ap.add_argument("--bottom", type=float, default=790.0)
    a = ap.parse_args()

    src = PdfReader(str(a.src)).pages[0]
    # PDF's origin is bottom-left; the poster's geometry is quoted from the top
    x0, x1 = a.left * MM, a.right * MM
    y0, y1 = (SHEET_H - a.bottom) * MM, (SHEET_H - a.top) * MM
    tile_w, tile_h = A4_W - 2 * MARGIN, A4_H - 2 * MARGIN
    cols = int(-(-(x1 - x0) // tile_w))
    rows = int(-(-(y1 - y0) // tile_h))

    w = PdfWriter()
    for r in range(rows):
        for c in range(cols):
            page = PageObject.create_blank_page(width=A4_W, height=A4_H)
            # top row first, so the printed stack reads the way the poster does
            dx = MARGIN - (x0 + c * tile_w)
            dy = MARGIN - (y1 - (r + 1) * tile_h)
            tile = PdfReader(str(a.src)).pages[0]
            tile.add_transformation(Transformation().translate(dx, dy))
            page.merge_page(tile)
            page.cropbox.lower_left = (0, 0)
            page.cropbox.upper_right = (A4_W, A4_H)
            w.add_page(page)
    with a.out.open("wb") as f:
        w.write(f)
    print(f"wrote {a.out.name}  —  {rows}x{cols} A4 landscape tiles at 1:1")
    print(f"  region: {a.right - a.left:.0f} x {a.bottom - a.top:.0f} mm of the A1 sheet")
    print("  print at 100% / Actual size — 'fit to page' rescales and voids the test")


if __name__ == "__main__":
    main()
