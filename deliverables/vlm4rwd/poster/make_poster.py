#!/usr/bin/env python3
"""VLM4RWD poster = the REALM @ EMNLP 2026 poster (A0) with the venue line replaced and the page
scaled to the VLM4RWD board size, 61 x 85 cm portrait (organiser email 2026-10-19).

Only the two footer lines naming the venue change; every other mark of the REALM PDF is kept as is.
The old lines are deleted from the content stream (so no hidden REALM text survives in the text layer),
their area is painted in the footer colour, and the new lines are set in Helvetica
(metric-compatible with the LiberationSans the original uses), at the size that reproduces the
original line width. Scaling is uniform: A0 is 1:1.414 and the board 1:1.393, so the poster fills the
height and leaves ~0.44 cm of white at each side.

  python make_poster.py <realm_poster_A0.pdf> <out.pdf>
"""
from __future__ import annotations

import io
import sys

from pypdf import PdfReader, PdfWriter, Transformation
from pypdf.generic import ContentStream, NameObject, RectangleObject
from reportlab.lib.colors import Color
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen import canvas

CM = 72 / 2.54
BOARD_W, BOARD_H = 61 * CM, 85 * CM

# measured on the REALM PDF with `pdftotext -bbox-layout` (top-down y, points) and a 150 dpi render
OLD_LINE1 = "REALM @ EMNLP 2026  ·  Non-archival poster  ·  Paper 192"
OLD_LINE2 = "29 October 2026  \u00b7  Budapest / online"
OLD_LINE1_W = 2059.9 - 1715.6
CENTER_X = (1715.6 + 2059.9) / 2
BASE1_Y, BASE2_Y = 3258.0, 3271.2          # baselines, top-down
COVER = (1700.0, 3244.0, 2075.0, 3277.0)   # x0, y0, x1, y1 top-down; the QR code starts at x ~2182
BG = Color(20 / 255, 30 / 255, 65 / 255)
FG = Color(219 / 255, 227 / 255, 255 / 255)
FONT = "Helvetica"

NEW_LINE1 = "VLM4RWD @ NeurIPS 2026  ·  Non-archival poster  ·  Paper 18"
NEW_LINE2 = "11 December 2026  ·  Sydney"


def overlay(w: float, h: float) -> PdfReader:
    size = OLD_LINE1_W / stringWidth(OLD_LINE1, FONT, 1.0)
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(w, h))
    x0, y0, x1, y1 = COVER
    c.setFillColor(BG)
    c.rect(x0, h - y1, x1 - x0, y1 - y0, stroke=0, fill=1)
    c.setFillColor(FG)
    c.setFont(FONT, size)
    c.drawCentredString(CENTER_X, h - BASE1_Y, NEW_LINE1)
    c.drawCentredString(CENTER_X, h - BASE2_Y, NEW_LINE2)
    c.save()
    buf.seek(0)
    return PdfReader(buf)


def drop_old_lines(reader: PdfReader, page) -> None:
    """Remove the two text-show operators that print the REALM venue lines (plain WinAnsi strings)."""
    old = {OLD_LINE1.encode("cp1252"), OLD_LINE2.encode("cp1252")}
    cs = ContentStream(page.get_contents(), reader)
    keep, dropped = [], 0
    for operands, op in cs.operations:
        if op == b"TJ":
            raw = b"".join(x.get_original_bytes() for x in operands[0] if hasattr(x, "get_original_bytes"))
            if raw in old:
                dropped += 1
                continue
        keep.append((operands, op))
    if dropped != 2:
        raise SystemExit(f"expected to drop the 2 REALM venue lines, dropped {dropped}")
    cs.operations = keep
    page[NameObject("/Contents")] = cs


def main(src: str, dst: str) -> int:
    reader = PdfReader(src)
    page = reader.pages[0]
    drop_old_lines(reader, page)
    w, h = float(page.mediabox.width), float(page.mediabox.height)
    page.merge_page(overlay(w, h).pages[0])

    s = min(BOARD_W / w, BOARD_H / h)
    dx, dy = (BOARD_W - w * s) / 2, (BOARD_H - h * s) / 2
    page.add_transformation(Transformation().scale(s, s).translate(dx, dy))
    box = RectangleObject([0, 0, BOARD_W, BOARD_H])
    page.mediabox = box
    page.cropbox = box
    page.trimbox = box

    out = PdfWriter()
    out.add_page(page)
    out.add_metadata({"/Title": "Routing Is Least Learnable Where It Is Most Valuable - VLM4RWD 2026 poster"})
    with open(dst, "wb") as f:
        out.write(f)
    print(f"scale {s:.5f}, side margin {dx / CM:.2f} cm, font {OLD_LINE1_W / stringWidth(OLD_LINE1, FONT, 1.0):.2f} pt",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
