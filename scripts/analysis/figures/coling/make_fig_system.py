#!/usr/bin/env python3
"""COLING Figure 1 (fig_system.pdf): panels 1-2 of the REALM/VLM4RWD overview figure.

fig_overview.pdf has no source file, so this edits the PDF itself (PyMuPDF):
  - panel 3 (oracle / cascade / ceilings) is removed: none of those appear in the COLING draft;
  - "text-only variants" -> "screenshot-free variants": the task's reference image reaches every
    mode, so the page-screenshot-free modes are not text-only (paper wording, 2026-10-08 audit);
  - "reactive execution" is re-set inside its pill (the original clipped its first letter).
Inserted text uses Arial Bold, metrically the same as the figure's Arimo-Bold, embedded as a subset.

usage: python -X utf8 scripts/analysis/figures/coling/make_fig_system.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import fitz

REPO = Path(__file__).resolve().parents[4]
SRC = REPO / "deliverables/vlm4rwd/figures/fig_overview.pdf"
OUT = REPO / "deliverables/coling2027/figures/fig_system.pdf"
BOLD_TTF = "C:/Windows/Fonts/arialbd.ttf"

CROP_X = 995.0                           # panel 2's border is at x=993.5, width 1.275
PILL = fitz.Rect(42, 330.1, 121, 353.2)  # the "reactive execution" pill, right of its green dot
BORDER = (0.720703, 0.785156, 0.839844)
# (old span text, new text, box to centre in or None = the old span's box, size or None = the old size)
EDITS = [("reactive execution", "reactive execution", PILL, 8.0),
         ("text-only variants", "screenshot-free variants", None, None)]


def main() -> int:
    doc = fitz.open(SRC)
    page = doc[0]
    spans = {}
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            for s in line["spans"]:
                spans.setdefault(s["text"].strip(), []).append(s)

    placed = []
    for old, new, box, size in EDITS:
        (s,) = spans[old]   # exactly one occurrence, or the source figure changed under us
        r = fitz.Rect(s["bbox"])
        page.add_redact_annot(r)
        c = s["color"]
        rgb = ((c >> 16) & 255) / 255, ((c >> 8) & 255) / 255, (c & 255) / 255
        placed.append((new, box or r, size or s["size"], rgb, s["origin"][1], box is PILL))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE, graphics=fitz.PDF_REDACT_LINE_ART_NONE)

    # drop panel 3 so nothing past the crop edge survives in the file, not even as extractable text.
    # REMOVE_IF_TOUCHED would also take panel 2's border (it shares a path with the page background).
    page.add_redact_annot(fitz.Rect(994.3, 0, page.rect.x1, page.rect.y1))
    page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_REMOVE, graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
    # the arrow into panel 3 (x=995..1001, round cap r=2.25) survives redaction and overlaps panel 2's border;
    # pdflatex also shows a sliver past the MediaBox edge, so paint over the whole line
    page.draw_rect(fitz.Rect(992.6, 259.0, 1004.0, 265.0), color=None, fill=(1, 1, 1), overlay=True)
    page.draw_line((993.5, 257.0), (993.5, 267.0), color=BORDER, width=1.275, overlay=True)

    bold = fitz.Font(fontfile=BOLD_TTF)
    page.insert_font(fontname="ArialBd", fontbuffer=bold.buffer)
    for text, box, size, rgb, baseline, centre_v in placed:
        x = (box.x0 + box.x1) / 2 - bold.text_length(text, fontsize=size) / 2
        y = (box.y0 + box.y1) / 2 + size * 0.35 if centre_v else baseline
        page.insert_text((x, y), text, fontname="ArialBd", fontsize=size, color=rgb)

    page.set_mediabox(fitz.Rect(0, 0, CROP_X, page.rect.y1))
    doc.subset_fonts()
    doc.save(OUT, garbage=4, deflate=True)
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
