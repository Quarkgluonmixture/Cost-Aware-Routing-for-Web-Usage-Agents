#!/usr/bin/env python3
"""Build the Holistic AI x UCL CDI showcase poster from the supplied A1 template.

v5 — "Look, read, or both?" (2026-09-02)
-----------------------------------------
The poster now stands next to a laptop that replays the same task through three
ways of seeing the page. That split decides what goes on silk:

    the DEMO shows the phenomenon   — one task, three eyes, three behaviours,
                                      three bills, step by step
    the POSTER shows the system and — where the decision sits in the agent loop,
    the measurement                   what it was worth in hindsight, whether it
                                      could be learned, and why not

Reviewer brief (Holistic AI, 2026-09-02) still holds: one main system diagram,
compact, one type scale, the template's own skeleton. So:

    TITLE        Look, read, or both? / Web agents can't yet learn how to see a page
    STANDFIRST   one sentence, both baselines named, two lines at the template's 28pt
    FIG 1        the system diagram, drawn in native shapes at full width:
                 task + page -> WHO DECIDES HOW TO SEE? -> LOOK / READ / BOTH ->
                 agent step -> outcome + bill. The decision box is the experiment.
    COLUMN 1     ON THE SCREEN BESIDE YOU: the three demo tasks, one frame each,
                 with every mode's real outcome / steps / bill
    COLUMNS 2-3  RESULTS: Fig 2 (thesis F13) + three verdicts; WHY IT CANNOT BE
                 LEARNED; TAKEAWAY

Type scale is the template's own (read from its placeholders):
    title 41 Georgia · standfirst 28 Georgia · section header 14 Consolas bold
    body 17.65 Arial · caption 12.7 Arial grey · marks 32.5 Consolas bold

Every number on the demo strip is parsed from the episode summaries by
``poster_figures.py`` (``figures/demo_strip.json``) — nothing on that strip is
typed by hand, and a task whose outcome flips on rerun fails the build there.

Hard constraint from the organisers: **do not resize the slide.** ``verify``
re-asserts it and fails the build if any panel overruns its box.

Usage::

    .venv/bin/python3 deliverables/showcase/poster_figures.py   # figures + strip data
    .venv/bin/python3 deliverables/showcase/build_poster.py
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import qrcode
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Mm, Pt

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "Showcase-Poster-Template-A1.pptx"
OUT = HERE / "poster_jiaming_wei.pptx"
FIGDIR = HERE / "figures"

REPO_URL = "https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents"

# ---------------------------------------------------------------- palette / type
INK = RGBColor(0x12, 0x16, 0x2E)
INK_STRONG = RGBColor(0x14, 0x1E, 0x41)
MUTED = RGBColor(0x5D, 0x67, 0x87)
ACCENT = RGBColor(0x50, 0x49, 0xF9)
HAIRLINE = RGBColor(0xDD, 0xE3, 0xF2)
FIG_FILL = RGBColor(0xF7, 0xF9, 0xFD)
GREY = RGBColor(0xC9, 0xCF, 0xE0)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
# Thesis figure palette: orange = text only, green = text + image, blue = image
# only. The demo uses the same three colours for READ / BOTH / LOOK.
C_TEXT = RGBColor(0xE8, 0x72, 0x0C)
C_BOTH = RGBColor(0x14, 0x85, 0x5F)
C_IMAGE = RGBColor(0x1F, 0x5F, 0xD6)
C_FAIL = RGBColor(0xC2, 0x35, 0x2B)

SERIF, SANS, MONO = "Georgia", "Arial", "Consolas"

FONT_FILE = {
    (SANS, False): "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
    (SANS, True): "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
    (SERIF, False): "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    (SERIF, True): "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
}
# Calibrated to the renderer (LibreOffice), not the face file: see §498.2.
LINE_HEIGHT = {SANS: 1.20, SERIF: 1.20}

SZ_TITLE = Pt(40.96)
SZ_STANDFIRST = Pt(28)
SZ_BODY = Pt(17.65)
SZ_CAPTION = Pt(12.71)
SZ_MARK = Pt(32.48)

BODY_SPACING = 1.2
CAPTION_SPACING = 1.32
PARA_GAP = Mm(3.5)

# ------------------------------------------------------------------------- grid
COL_X = (Mm(20.2), Mm(209.3), Mm(398.3))
COL_W = Mm(179.2)
SPAN_23_X, SPAN_23_W = Mm(209.3), Mm(368.2)
FULL_X, FULL_W = Mm(20.2), Mm(557.3)

SYS_Y = Mm(140)
ROW_Y = Mm(300)
ROW_BOTTOM = Mm(782)

MODES = (("look", "LOOK", C_IMAGE), ("read", "READ", C_TEXT), ("both", "BOTH", C_BOTH))


# ------------------------------------------------------------------- measurement
_MARK_RE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*")
_SCALE = 8


def segments(text: str):
    out, pos = [], 0
    for m in _MARK_RE.finditer(text):
        if m.start() > pos:
            out.append((text[pos : m.start()], False, False))
        if m.group(1) is not None:
            out.append((m.group(1), True, False))
        else:
            out.append((m.group(2), False, True))
        pos = m.end()
    if pos < len(text):
        out.append((text[pos:], False, False))
    out = out or [("", False, False)]
    leftover = "".join(chunk for chunk, _, _ in out)
    assert "*" not in leftover, f"unpaired '*' would print literally: {text!r}"
    return out


_font_cache: dict = {}


def _font(family: str, bold: bool, size_pt: float):
    key = (family, bold, round(size_pt, 2))
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(
            FONT_FILE[(family, bold)], int(round(size_pt * _SCALE)))
    return _font_cache[key]


def line_count(text: str, family: str, size_pt: float, width_emu: int) -> int:
    width_px = width_emu / 12700 * _SCALE
    words = [(w, b) for seg, b, _ in segments(text) for w in seg.split(" ") if w]
    lines, cur_px, started = 1, 0.0, False
    for word, bold in words:
        f = _font(family, bold, size_pt)
        w_px = f.getlength(word if not started else " " + word)
        if started and cur_px + w_px > width_px:
            lines += 1
            cur_px = f.getlength(word)
        else:
            cur_px += w_px
            started = True
    return lines


def text_height(paragraphs, family, size, width_emu, *, spacing, gap) -> int:
    n = sum(line_count(p, family, size.pt, width_emu) for p in paragraphs)
    return int(n * size.pt * LINE_HEIGHT[family] * spacing * 12700) + gap * (len(paragraphs) - 1)


# ------------------------------------------------------------------- pptx helpers
def find(slide, name):
    for shape in slide.shapes:
        if shape.name == name:
            return shape
    raise KeyError(f"shape {name!r} not in template")


def drop(slide, *shapes):
    for shape in shapes:
        el = shape._element if hasattr(shape, "_element") else find(slide, shape)._element
        el.getparent().remove(el)


def clone(slide, shape, left, top):
    el = copy.deepcopy(shape._element)
    shape._element.getparent().append(el)
    new = slide.shapes[-1]
    new.left, new.top = left, top
    return new


def _run(paragraph, text, font, size, color, bold, italic=False):
    run = paragraph.add_run()
    run.text = text
    run.font.name = font
    run.font.size = size
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color


def _emit(paragraph, text, *, font, size, color, bold=False):
    for chunk, is_bold, is_italic in segments(text):
        _run(paragraph, chunk, font, size, color, bold or is_bold, is_italic)


def textbox(slide, left, top, width, height, paragraphs, *, font=SANS, size=SZ_BODY,
            color=INK, line_spacing=BODY_SPACING, space_after=PARA_GAP,
            align=PP_ALIGN.LEFT, bold=False, anchor=None):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    if anchor is not None:
        frame.vertical_anchor = anchor
    for i, text in enumerate(paragraphs):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.line_spacing = line_spacing
        para.space_after = space_after
        para.alignment = align
        _emit(para, text, font=font, size=size, color=color, bold=bold)
    return box


def body(slide, x, y, w, paragraphs, *, after=Mm(4), size=SZ_BODY, color=INK):
    h = text_height(paragraphs, SANS, size, w, spacing=BODY_SPACING, gap=PARA_GAP)
    textbox(slide, x, y, w, h, paragraphs, size=size, color=color)
    return y + h + after


def caption(slide, x, y, w, paragraphs, *, after=Mm(4), color=MUTED):
    h = text_height(paragraphs, SANS, SZ_CAPTION, w, spacing=CAPTION_SPACING, gap=Mm(1.5))
    textbox(slide, x, y, w, h, paragraphs, size=SZ_CAPTION, color=color,
            line_spacing=CAPTION_SPACING, space_after=Mm(1.5))
    return y + h + after


def set_text(shape, paragraphs, *, size=None, align=None):
    frame = shape.text_frame
    first = frame.paragraphs[0]
    proto = first.runs[0]
    font, color, bold = proto.font.name, proto.font.color.rgb, proto.font.bold
    size = size or proto.font.size
    spacing = first.line_spacing
    frame.clear()
    for i, text in enumerate(paragraphs):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.line_spacing = spacing
        if align is not None:
            para.alignment = align
        _emit(para, text, font=font, size=size, color=color, bold=bool(bold))


def rect(slide, x, y, w, h, fill=None, line=None, *, lw=Pt(0.7), radius=None, dash=False):
    kind = MSO_SHAPE.ROUNDED_RECTANGLE if radius is not None else MSO_SHAPE.RECTANGLE
    shape = slide.shapes.add_shape(kind, x, y, w, h)
    if fill is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    if line is None:
        shape.line.fill.background()
    else:
        shape.line.color.rgb = line
        shape.line.width = lw
        if dash:
            shape.line.dash_style = MSO_LINE.DASH
    shape.shadow.inherit = False
    if radius is not None:
        shape.adjustments[0] = radius
    shape.text_frame.word_wrap = True
    return shape


def arrow_right(slide, x, y, w, h, fill=GREY):
    shape = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, x, y, w, h)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


HEADER_PARTS: tuple = ()


def panel_header(slide, x, y, label, width):
    block, title, rule_a, rule_b = HEADER_PARTS
    base = block.top
    clone(slide, block, x, y)
    t = clone(slide, title, x + (title.left - block.left), y - Mm(1.2))
    ra = clone(slide, rule_a, x, y + (rule_a.top - base))
    rb = clone(slide, rule_b, x + (rule_b.left - block.left), y + (rule_b.top - base))
    rb.width = width - ra.width - (rb.left - x - ra.width)
    set_text(t, [label])
    return y + Mm(15)


def fig_box(slide, x, y, w, png: Path, cap: str):
    inset = Mm(3)
    with Image.open(png) as im:
        pic_w = w - 2 * inset
        pic_h = int(pic_w * im.height / im.width)
    box_h = Mm(1.4) + inset + pic_h + inset
    rect(slide, x, y, w, box_h, fill=FIG_FILL, line=HAIRLINE)
    rect(slide, x, y, w, Mm(1.4), fill=ACCENT)
    slide.shapes.add_picture(str(png), x + inset, y + Mm(1.4) + inset, width=pic_w)
    return caption(slide, x, y + box_h + Mm(2.5), w, [cap], after=Mm(3))


# --------------------------------------------------------------- system diagram
def card(slide, x, y, w, h, *, bar=ACCENT, dash=False, line=HAIRLINE):
    rect(slide, x, y, w, h, fill=FIG_FILL, line=line, dash=dash)
    if bar is not None:
        rect(slide, x, y, w, Mm(1.4), fill=bar)


def label(slide, x, y, w, text, color=MUTED):
    textbox(slide, x, y, w, Mm(6), [text], font=MONO, size=Pt(10.59), color=color,
            space_after=Pt(0))
    return y + Mm(6.5)


def build_system(slide):
    """Fig 1: where the decision sits in the agent loop, and what is measured.

    Drawn in native shapes so it uses the template's type and colours; nothing
    here is a screenshot of a figure. The five stages read left to right and
    the arrows carry no data — they are the loop's order."""
    y = panel_header(slide, FULL_X, SYS_Y, "WHERE THE DECISION SITS IN THE AGENT LOOP", FULL_W)
    top, H = y, Mm(118)
    x0 = FULL_X
    pad = Mm(5)
    # column geometry, mm from x0
    cols = {"task": (0, 86), "decide": (100, 120), "eyes": (234, 118), "agent": (366, 92),
            "measure": (472, 85.3)}
    arrows = [(88, 10), (222, 10), (354, 10), (460, 10)]
    for ax, aw in arrows:
        arrow_right(slide, x0 + Mm(ax), top + H / 2 - Mm(7), Mm(aw), Mm(14))

    # -- task + page
    cx, cw = (x0 + Mm(cols["task"][0]), Mm(cols["task"][1]))
    card(slide, cx, top, cw, H)
    yy = label(slide, cx + pad, top + Mm(4), cw - 2 * pad, "TASK + LIVE PAGE")
    yy = body(slide, cx + pad, yy, cw - 2 * pad,
              ["“Show me the cheapest bike with red handlebars between $900–950.”"],
              after=Mm(3))
    caption(slide, cx + pad, yy, cw - 2 * pad,
            ["Part of the intent is in the pictures, part in the text — and which part "
             "matters changes task by task."])

    # -- who decides how to see
    cx, cw = (x0 + Mm(cols["decide"][0]), Mm(cols["decide"][1]))
    card(slide, cx, top, cw, H, bar=None, dash=True, line=ACCENT)
    yy = label(slide, cx + pad, top + Mm(4), cw - 2 * pad, "WHO DECIDES HOW TO SEE?", color=ACCENT)
    pills = [("Fixed mode", "the same choice for every task"),
             ("Hindsight oracle", "the best choice, known afterwards"),
             ("Learned router", "a choice made before the task runs")]
    ph = Mm(29)
    for i, (name, note) in enumerate(pills):
        py = yy + Emu(i * int(ph + Mm(3)))
        rect(slide, cx + pad, py, cw - 2 * pad, ph, fill=PAPER, line=HAIRLINE, radius=0.12)
        textbox(slide, cx + pad + Mm(4), py + Mm(3.5), cw - 2 * pad - Mm(8), Mm(8),
                [name], bold=True, color=INK_STRONG, space_after=Pt(0))
        textbox(slide, cx + pad + Mm(4), py + Mm(13), cw - 2 * pad - Mm(8), Mm(13),
                [note], size=SZ_CAPTION, color=MUTED, line_spacing=CAPTION_SPACING,
                space_after=Pt(0))

    # -- the three eyes
    cx, cw = (x0 + Mm(cols["eyes"][0]), Mm(cols["eyes"][1]))
    eyes = [("LOOK", C_IMAGE, "screenshot only", "3,123 tokens on one real page"),
            ("READ", C_TEXT, "accessibility-tree text only",
             "3,314 tokens · plus three text-only variants"),
            ("BOTH", C_BOTH, "marked screenshot + text", "4,335 tokens")]
    eh = Mm(36)
    for i, (name, colour, what, price) in enumerate(eyes):
        ey = top + Emu(i * int(eh + Mm(5)))
        card(slide, cx, ey, cw, eh, bar=colour)
        textbox(slide, cx + pad, ey + Mm(4.5), Mm(40), Mm(8), [name], font=MONO,
                size=SZ_BODY, bold=True, color=colour, space_after=Pt(0))
        textbox(slide, cx + pad, ey + Mm(14), cw - 2 * pad, Mm(8), [what], bold=True,
                color=INK_STRONG, space_after=Pt(0))
        textbox(slide, cx + pad, ey + Mm(23), cw - 2 * pad, Mm(11), [price],
                size=SZ_CAPTION, color=MUTED, line_spacing=CAPTION_SPACING, space_after=Pt(0))

    # -- the agent
    cx, cw = (x0 + Mm(cols["agent"][0]), Mm(cols["agent"][1]))
    card(slide, cx, top, cw, H)
    yy = label(slide, cx + pad, top + Mm(4), cw - 2 * pad, "AGENT, ONE STEP AT A TIME")
    yy = body(slide, cx + pad, yy, cw - 2 * pad, ["**think → act**"], after=Mm(2))
    yy = caption(slide, cx + pad, yy, cw - 2 * pad,
                 ["click · type · scroll · go back · finish",
                  "repeats until it finishes or hits 30 steps"], after=Mm(3))
    caption(slide, cx + pad, yy, cw - 2 * pad,
            ["Same model, same prompt, same step budget and same cost accounting in "
             "every mode — only what it is shown changes."], color=INK)

    # -- measured
    cx, cw = (x0 + Mm(cols["measure"][0]), Mm(cols["measure"][1]))
    card(slide, cx, top, cw, H)
    yy = label(slide, cx + pad, top + Mm(4), cw - 2 * pad, "MEASURED")
    yy = body(slide, cx + pad, yy, cw - 2 * pad, ["**✓ / ✗ per task**", "**$ billed per episode**"],
              after=Mm(3))
    caption(slide, cx + pad, yy, cw - 2 * pad,
            ["8 website–model settings · two benchmarks · 6 modes · 8,934 episodes"])

    y = top + H + Mm(3)
    return caption(slide, FULL_X, y, FULL_W, [
        "**Fig 1.** Everything is held fixed except how the page is shown. The dashed "
        "box is what this work measures: a fixed choice, the best choice in hindsight, "
        "and a choice a router has to learn — each judged on **both** success and cost. "
        "The laptop beside this poster replays the three eyes on the three tasks below."],
        after=Mm(0))


# ------------------------------------------------------------------- demo strip
def build_strip(slide):
    x, w = COL_X[0], COL_W
    y = panel_header(slide, x, ROW_Y, "ON THE SCREEN BESIDE YOU", w)
    strip = json.loads((FIGDIR / "demo_strip.json").read_text())
    tile_w = int((w - 2 * Mm(3)) / 3)
    tile_h = Mm(24)
    for task, d in strip.items():
        y = body(slide, x, y, w, [f"**“{d['intent']}”**"], after=Mm(2.5))
        png = FIGDIR / d["thumb"]
        with Image.open(png) as im:
            ph = int(w * im.height / im.width)
        rect(slide, x, y, w, ph, fill=None, line=HAIRLINE)
        slide.shapes.add_picture(str(png), x, y, width=w)
        y += ph + Mm(2.5)
        for i, (key, name, colour) in enumerate(MODES):
            m = d["modes"][key]
            tx = x + Emu(i * (tile_w + int(Mm(3))))
            rect(slide, tx, y, Emu(tile_w), tile_h, fill=FIG_FILL, line=HAIRLINE)
            rect(slide, tx, y, Emu(tile_w), Mm(1.2), fill=colour)
            textbox(slide, tx + Mm(3), y + Mm(3), Mm(30), Mm(6), [name], font=MONO,
                    size=Pt(10.59), bold=True, color=colour, space_after=Pt(0))
            mark, mc = ("✓", C_BOTH) if m["success"] else ("✗", C_FAIL)
            textbox(slide, tx + Mm(3), y + Mm(7.5), Mm(16), Mm(15), [mark], font=MONO,
                    size=SZ_MARK, bold=True, color=mc, line_spacing=1.0, space_after=Pt(0))
            textbox(slide, tx + Mm(20), y + Mm(9), Emu(tile_w) - Mm(22), Mm(14),
                    [f"{m['steps']} steps", f"${m['cost_usd']:.3f}"], size=SZ_CAPTION,
                    color=INK, line_spacing=CAPTION_SPACING, space_after=Pt(0))
        y += tile_h + Mm(8)
    return caption(slide, x, y - Mm(2), w, [
        "One recorded run per mode, B0 · classifieds. Every ✓ / ✗ above came out the "
        "same on an independent rerun; steps and cost differ run to run — across all "
        "tasks a rerun flips 10–14% of outcomes."], after=Mm(0))


# ---------------------------------------------------------------------- results
def metric_strip(slide, x, y, w, tiles):
    """The template's metric strip (Rectangle 52 + three number/label pairs).
    Each label names its own baseline: the strip is exactly the place where two
    different baselines would otherwise be read as one comparison."""
    h = Mm(32.5)
    rect(slide, x, y, w, h, fill=RGBColor(0xEC, 0xEF, 0xFA))
    pad = Mm(7)
    tile_w = int((w - 2 * pad) / len(tiles))
    for i, (number, lbl) in enumerate(tiles):
        assert len(lbl) * 10.59 * 0.6 * 12700 < tile_w - Mm(4), f"metric label overprints: {lbl!r}"
        tx = x + pad + Emu(i * tile_w)
        textbox(slide, tx, y + Mm(5), Emu(tile_w), Mm(17), [number], font=MONO,
                size=SZ_MARK, color=INK_STRONG, bold=True, line_spacing=1.18, space_after=Pt(0))
        textbox(slide, tx, y + Mm(20.5), Emu(tile_w), Mm(9), [lbl], font=MONO,
                size=Pt(10.59), color=MUTED, line_spacing=1.18, space_after=Pt(0))
    return y + h + Mm(4)


def build_results(slide):
    x, w = SPAN_23_X, SPAN_23_W
    y = panel_header(slide, x, ROW_Y, "RESULTS ACROSS 8,934 EPISODES", w)
    y = metric_strip(slide, x, y, w, [
        ("+16.35 pp", "CEILING, LARGEST OF 8 · VS BEST FIXED MODE"),
        ("0 of 8", "LEARNED ROUTERS BEAT ALWAYS-CHEAPEST"),
        ("1 of 8", "HINDSIGHT ORACLES BEAT ALWAYS-CHEAPEST"),
    ])
    y = fig_box(
        slide, x, y, w, FIGDIR / "poster_dominance_plane.png",
        "**Fig 2.** Every policy in every setting against one fixed baseline, **always "
        "use the cheapest mode** (★). A win lands in the shaded region: cheaper *and* no "
        "worse. Always-cheapest is cheapest on average, not per episode, which is why a "
        "few points sit left of it. Nested cross-validation; 10,000 bundle permutations.")
    y = body(slide, x, y, w, [
        "**The ceiling is real.** In hindsight, choosing the eyes per task solves "
        "**+3.45 to +16.35 pp** more than the best single fixed mode, at 1.6–35.3% "
        "lower cost, in 8 of 8 settings.",
        "**Nothing we trained wins.** **0 of 8** learned routers beat always-cheapest "
        "on both success and cost — and even the hindsight oracle does so in only "
        "**1 of 8**.",
        "**What survives is a bound, not a router.** Sending the tasks nobody solves to "
        "the cheapest mode saves 9.5–30.6% at identical success in 8 of 8 — against the "
        "best-success fixed mode, and plain always-cheapest usually saves more."],
        after=Mm(6))

    y = panel_header(slide, x, y, "WHY IT CANNOT BE LEARNED", w)
    y = body(slide, x, y, w, [
        "A routing label exists only when the agent solves a task. Here the best single "
        "mode solves just **2–36%** of tasks, leaving typically **15–97** usable labels "
        "per setting — **the agents that would gain most from routing produce the least "
        "supervision to learn it.** Deliberately shrinking the training data confirms "
        "scarcity is the mechanism and prices it: the failing settings would need at "
        "least **2.1–4.2×** more tasks than the benchmarks contain.",
        "Rerunning **one unchanged mode** flips **10–14%** of outcomes and by itself "
        "buys **2.0–7.6 pp** (B0 · classifieds, six replicated modes, n=224); every "
        "gain on this sheet is read against that band, not against zero."],
        after=Mm(6))

    y = panel_header(slide, x, y, "TAKEAWAY", w)
    y = body(slide, x, y, w, [
        "**Routing is not only a model-selection problem: its learnability depends on "
        "the competence of the agent producing the labels.** So, in this order: improve "
        "the agent, then generate reliable supervision, then learn selective perception."],
        after=Mm(1.5))
    return caption(slide, x, y, w, [
        "Measured inside the 2–36% success regime we observed. This conclusion need not "
        "hold for stronger agents."], after=Mm(0))


# --------------------------------------------------------------------------- run
TITLE = ["Look, read, or both?", "Web agents can't yet learn how to see a page"]
STANDFIRST = (
    "Seen the right way, a page would let the agent solve up to 16 more tasks in 100 "
    "than the best fixed mode — yet none of 8 learned routers beat always using the "
    "cheapest mode on both success and cost."
)


def main():
    prs = Presentation(str(TEMPLATE))
    slide = prs.slides[0]

    title = find(slide, "TextBox 6")
    title.top, title.height = Mm(8), Mm(46)
    for line in TITLE:
        n = line_count(line, SERIF, SZ_TITLE.pt, int(title.width * 0.94))
        assert n == 1, f"title line wraps to {n} lines: {line!r}"
    set_text(title, TITLE)

    set_text(find(slide, "TextBox 7"),
             ["Jiaming Wei          Supervisors: Prof. María Pérez-Ortiz  ·  Zekun Wu"])
    set_text(find(slide, "TextBox 8"),
             ["UCL Centre for Artificial Intelligence          Holistic AI"])
    standfirst = find(slide, "TextBox 12")
    n = line_count(STANDFIRST, SERIF, SZ_STANDFIRST.pt, int(standfirst.width * 0.94))
    assert n <= 2, f"standfirst wraps to {n} lines at 28pt and will leave its band"
    set_text(standfirst, [STANDFIRST])

    global HEADER_PARTS
    HEADER_PARTS = tuple(find(slide, n) for n in
                         ("Rectangle 13", "TextBox 14", "Rectangle 15", "Rectangle 16"))
    drop(slide, "TextBox 17", "Rectangle 33", "TextBox 34", "Rectangle 35",
         "Rectangle 36", "TextBox 37", "Rectangle 38", "Rectangle 39", "TextBox 42",
         "Rectangle 48", "TextBox 49", "Rectangle 50", "Rectangle 51",
         "Rectangle 52", "TextBox 53", "TextBox 54", "TextBox 55", "TextBox 56",
         "TextBox 57", "TextBox 58")

    fig_end = build_system(slide)
    assert fig_end <= ROW_Y, f"Fig 1 ends at {fig_end / 36000:.1f}mm, past the row start {ROW_Y / 36000:.0f}mm"
    ends = {"column 1": (build_strip(slide), ROW_BOTTOM),
            "columns 2-3": (build_results(slide), ROW_BOTTOM)}

    drop(slide, *HEADER_PARTS)

    set_text(find(slide, "TextBox 71"),
             ["jiaming.wei.22@ucl.ac.uk",
              "Paper, code & full results: github.com/Quarkgluonmixture/"
              "Cost-Aware-Routing-for-Web-Usage-Agents"])
    qr_png = FIGDIR / "qr_repo.png"
    qrcode.make(REPO_URL, box_size=20, border=1).save(qr_png)
    set_text(find(slide, "TextBox 73"), ["Explore all 8 settings →"])
    slot = find(slide, "Rectangle 74")
    slide.shapes.add_picture(str(qr_png), slot.left + Mm(2), slot.top + Mm(2),
                             width=slot.width - Mm(4))
    drop(slide, "Rectangle 74", "TextBox 75")

    prs.save(str(OUT))
    verify(prs, ends, fig_end)


def verify(prs, ends, fig_end):
    mm = lambda emu: emu / 914400 * 25.4  # noqa: E731
    w, h = mm(prs.slide_width), mm(prs.slide_height)
    assert abs(w - 594) < 0.5 and abs(h - 841) < 0.5, "slide was resized!"
    print(f"wrote {OUT.relative_to(REPO)}   ({w:.0f}x{h:.0f}mm, A1, not resized)")
    print(f"  {'fig 1':12s} ends {mm(fig_end):6.1f}mm   (row starts {mm(ROW_Y):.0f}mm)")
    bad = False
    for name, (end, limit) in ends.items():
        slack = mm(limit - end)
        flag = "" if slack >= 0 else "   <-- OVERRUNS ITS BOX"
        bad |= slack < 0
        print(f"  {name:12s} ends {mm(end):6.1f}mm   (box to {mm(limit):.0f}mm, slack {slack:+6.1f}mm){flag}")
    if bad:
        raise SystemExit("a panel overran its box — shorten it or move the grid")


if __name__ == "__main__":
    main()
