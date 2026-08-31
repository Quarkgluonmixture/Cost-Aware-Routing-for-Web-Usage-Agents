#!/usr/bin/env python3
"""Build the Holistic AI x UCL CDI showcase poster from the supplied A1 template.

Why this is a grid and not a flow
---------------------------------
The first version poured nine numbered sections through a bin-packing routine
that split them into three equal columns. It produced a clean sheet and the
wrong artefact: bin-packing solves a *document* problem, so its optimum is
always "heading, body, next heading". A poster is a *hierarchy* problem — one
figure has to dominate, and the argument has to run along a path the eye takes
on its own. No amount of copy-editing moves a column packer off its optimum.

So the layout is now explicit and asymmetric, and the reading path is designed:

    HERO      three numbers that form one causal sentence
    MAIN      left: what we actually did (a real task, run six ways)
              right: the result, as the one dominant figure
    SECOND    left: why it failed (the paradox)  right: calibration + what survives
    TAKEAWAY  the transferable claim, with its scope attached

What is load-bearing
--------------------
*Text is measured, not estimated.* Arimo is metric-compatible with Arial, so
measuring with it reproduces PowerPoint's line breaks; and a rendered line
advances by the font's ascent + descent, then by the paragraph's line spacing.
Dropping that second factor under-counts every paragraph by ~12%, which is
exactly enough to slide body copy under the next heading.

*The template's header assembly is a mould, not a part.* It is cloned per panel
and the originals dropped, or the first panel lands on top of the placeholder.

*Two baselines exist and must never be silently merged.* The hindsight ceiling
(+16.35pp, 13.7-35.3% cheaper) is measured against the BEST-SUCCESS FIXED MODE;
the 0/8 learnability result is measured against ALWAYS-CHEAPEST. Putting them
side by side in the hero is good narrative and a factual trap, so every hero
number carries its baseline in its own label.

Hard constraint from the organisers: **do not resize the slide.** Nothing here
touches ``prs.slide_width`` / ``slide_height``; ``verify`` re-asserts it and
fails the build if any panel overruns its box.

Usage::

    .venv/bin/python3 deliverables/showcase/poster_figures.py   # figures first
    .venv/bin/python3 deliverables/showcase/build_poster.py
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import qrcode
from PIL import Image, ImageFont
from pptx import Presentation
from pptx.dml.color import RGBColor
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
ACCENT_PALE = RGBColor(0xE8, 0xE7, 0xFF)
PAPER = RGBColor(0xFF, 0xFF, 0xFF)
GREY = RGBColor(0xD8, 0xDC, 0xE8)
HAIRLINE = RGBColor(0xE2, 0xE5, 0xEF)
WASH = RGBColor(0xF7, 0xF8, 0xFC)
# Carried from _style.py so a visitor who reads the figure reads the same colour
# as the same thing in the panels beside it.
C_TEXTSIDE = RGBColor(0xE8, 0x72, 0x0C)
C_WIN = RGBColor(0x14, 0x85, 0x5F)
# For the illustrative listing only — two blue items and one red, so a viewer
# performs the agent's task themselves: filter by sight, then compare by text.
C_BLUE = RGBColor(0x3B, 0x6F, 0xD4)
C_RED = RGBColor(0xC2, 0x35, 0x2B)
CHROME = RGBColor(0xEC, 0xEE, 0xF4)

SERIF, SANS, MONO = "Georgia", "Arial", "Consolas"

# Metric-compatible stand-ins for measurement.
FONT_FILE = {
    (SANS, False): "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
    (SANS, True): "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
    (SERIF, False): "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    (SERIF, True): "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
    (MONO, False): "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    (MONO, True): "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
}
# A rendered line advances by ascent + descent, measured from the face files.
LINE_HEIGHT = {SANS: 1.118, MONO: 1.165, SERIF: 1.20}

SZ_TITLE = Pt(42)
SZ_STANDFIRST = Pt(25)
SZ_HERO_NUM = Pt(76)
SZ_HERO_LABEL = Pt(15)
SZ_ZERO = Pt(200)
SZ_ZERO_TAIL = Pt(52)
SZ_HERO_SAYS = Pt(19)
SZ_PANEL = Pt(20)
SZ_BODY = Pt(19)
SZ_SMALL = Pt(15)
SZ_CAPTION = Pt(14)
SZ_TASK = Pt(23)
SZ_TAKEAWAY = Pt(30)

LINE_SPACING = 1.26
PARA_GAP = Mm(4.5)

# ------------------------------------------------------------------------- grid
# Explicit, asymmetric, and designed around the reading path. The right-hand
# main panel is 343mm so the dominance plane can dominate.
MARGIN = Mm(20)
SHEET_W = Mm(594)
FULL_W = Mm(554)

HERO_Y, HERO_H = Mm(140), Mm(110)
MAIN_Y, MAIN_BOTTOM = Mm(262), Mm(578)
MAIN_L_X, MAIN_L_W = Mm(20), Mm(197)
MAIN_R_X, MAIN_R_W = Mm(231), Mm(343)
SECOND_Y, SECOND_BOTTOM = Mm(584), Mm(728)
SECOND_L_X, SECOND_L_W = Mm(20), Mm(266)
SECOND_R_X, SECOND_R_W = Mm(300), Mm(274)
TAKE_Y, TAKE_H = Mm(732), Mm(50)


# ------------------------------------------------------------------- measurement
_MARK_RE = re.compile(r"\*\*(.+?)\*\*|\*(.+?)\*")
_SCALE = 8


def segments(text: str):
    """Split marked-up copy into [(text, bold, italic)] runs.

    Italics are parsed as well as bold because an unhandled single ``*`` is not
    an error anywhere in this pipeline — it renders as a literal asterisk, and
    the build, the export and the geometry check all still pass. Two of them
    reached a printed draft that way. The assert below is the real fix: a stray
    asterisk now fails the build instead of reaching silk.
    """
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
            FONT_FILE[(family, bold)], int(round(size_pt * _SCALE))
        )
    return _font_cache[key]


def line_count(text: str, family: str, size_pt: float, width_emu: int) -> int:
    """Lines this paragraph occupies once wrapped — measured, not guessed."""
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


def text_height(paragraphs, family, size, width_emu, *, spacing=LINE_SPACING) -> int:
    size_pt = size.pt
    n = sum(line_count(p, family, size_pt, width_emu) for p in paragraphs)
    advance = size_pt * LINE_HEIGHT[family] * spacing
    return int(n * advance * 12700) + PARA_GAP * (len(paragraphs) - 1)


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
            color=INK, line_spacing=LINE_SPACING, space_after=PARA_GAP,
            align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    frame = box.text_frame
    frame.word_wrap = True
    frame.margin_left = frame.margin_right = 0
    frame.margin_top = frame.margin_bottom = 0
    for i, text in enumerate(paragraphs):
        para = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        para.line_spacing = line_spacing
        para.space_after = space_after
        para.alignment = align
        _emit(para, text, font=font, size=size, color=color)
    return box


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


def rect(slide, x, y, w, h, fill=None, line=None, *, radius=None, lw=Pt(1)):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius is not None else MSO_SHAPE.RECTANGLE,
        x, y, w, h)
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
    shape.shadow.inherit = False
    if radius is not None:
        shape.adjustments[0] = radius
    shape.text_frame.word_wrap = True
    return shape


def chip(slide, x, y, w, h, label, *, fill, ink, size=Pt(16)):
    box = rect(slide, x, y, w, h, fill=fill,
               line=MUTED if fill == GREY else ACCENT, radius=0.14, lw=Pt(1.25))
    frame = box.text_frame
    frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = frame.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    _run(para, label, MONO, size, ink, True)
    return box


HEADER_PARTS: tuple = ()


def panel_header(slide, x, y, label, width):
    """Clone the template's section-header assembly at an arbitrary position."""
    block, title, rule_a, rule_b = HEADER_PARTS
    base = block.top
    clone(slide, block, x, y)
    t = clone(slide, title, x + (title.left - block.left), y - Mm(1.2))
    ra = clone(slide, rule_a, x, y + (rule_a.top - base))
    rb = clone(slide, rule_b, x + (rule_b.left - block.left),
               y + (rule_b.top - base))
    rb.width = width - ra.width - (rb.left - x - ra.width) - Mm(0)
    set_text(t, [label], size=SZ_PANEL)
    return y + Mm(16)


def arrow_down(slide, cx, y, h, *, label=None, width=Mm(90)):
    """A downward step marker in the workflow. Drawn rather than typed so it
    survives at A1 without depending on a glyph the printer may not have."""
    shaft = rect(slide, cx - Mm(0.6), y, Mm(1.2), h - Mm(3), fill=GREY)
    head = slide.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE,
                                  cx - Mm(2.6), y + h - Mm(4), Mm(5.2), Mm(4))
    head.rotation = 180
    head.fill.solid()
    head.fill.fore_color.rgb = GREY
    head.line.fill.background()
    head.shadow.inherit = False
    if label:
        textbox(slide, cx + Mm(6), y + h / 2 - Mm(5), width, Mm(10), [label],
                font=MONO, size=Pt(13), color=MUTED, space_after=Pt(0))
    return shaft


# ------------------------------------------------------------------------ panels
def build_hero(slide):
    """The contradiction, as the poster's one oversized element.

    The first version set three numbers at equal weight in a tinted band. That
    reads as a KPI row: nothing is the result, everything is a metric. Two
    changes fix it. `2-36%` leaves — it is the *first step of the paradox*, not
    a headline, and it was diluting the other two. And the `0` is set at 200pt
    against whitespace, so from three metres the only thing legible besides the
    question is a very large zero. The hook is arithmetic curiosity: why zero?

    The two baselines still differ, so nothing here may read as one measurement
    against another. They are separated by a rule, each carries its own
    baseline, and the connective is narrative ("in hindsight" / "when actually
    learned") rather than comparative (no arrow, no "vs", no equals).
    """
    split = Mm(292)
    rect(slide, split, HERO_Y + Mm(6), Mm(0.6), HERO_H - Mm(16), fill=HAIRLINE)

    # Left: the opportunity.
    lx = MARGIN
    textbox(slide, lx, HERO_Y, Mm(240), Mm(7), ["IN HINDSIGHT"], font=MONO,
            size=SZ_HERO_LABEL, color=MUTED, space_after=Pt(0))
    textbox(slide, lx, HERO_Y + Mm(9), Mm(250), Mm(30), ["+16.35 pp"], font=MONO,
            size=SZ_HERO_NUM, color=INK_STRONG, line_spacing=1.0,
            space_after=Pt(0))
    textbox(slide, lx, HERO_Y + Mm(41), Mm(250), Mm(7),
            ["MORE TASKS SOLVED PER 100"], font=MONO, size=SZ_HERO_LABEL,
            color=INK_STRONG, space_after=Pt(0))
    textbox(slide, lx, HERO_Y + Mm(48), Mm(250), Mm(7),
            ["vs the best single fixed mode"], font=MONO, size=SZ_HERO_LABEL,
            color=MUTED, space_after=Pt(0))
    textbox(slide, lx, HERO_Y + Mm(78), Mm(258), Mm(24),
            ["So there is something genuinely worth routing to."],
            size=Pt(21), color=INK, line_spacing=1.22, space_after=Pt(0))

    # Right: the result. Orange is not decoration — it is the colour the
    # dominance plane already uses for `learned, nested`, so a visitor who
    # walks up to the figure meets the same encoding twice.
    rx = split + Mm(20)
    textbox(slide, rx, HERO_Y, Mm(240), Mm(7), ["WHEN ACTUALLY LEARNED"],
            font=MONO, size=SZ_HERO_LABEL, color=MUTED, space_after=Pt(0))
    textbox(slide, rx, HERO_Y + Mm(2), Mm(90), Mm(76), ["0"], font=MONO,
            size=SZ_ZERO, color=C_TEXTSIDE, line_spacing=1.0, space_after=Pt(0))
    textbox(slide, rx + Mm(51), HERO_Y + Mm(50), Mm(90), Mm(22), ["of 8"],
            font=MONO, size=SZ_ZERO_TAIL, color=C_TEXTSIDE, line_spacing=1.0,
            space_after=Pt(0))
    textbox(slide, rx + Mm(112), HERO_Y + Mm(20), Mm(160), Mm(14),
            ["LEARNED ROUTERS"], font=MONO, size=Pt(28), color=INK_STRONG,
            line_spacing=1.1, space_after=Pt(0))
    textbox(slide, rx + Mm(112), HERO_Y + Mm(35), Mm(158), Mm(20),
            ["beat always-cheapest on both success and cost"], font=MONO,
            size=SZ_HERO_LABEL, color=MUTED, line_spacing=1.35,
            space_after=Pt(0))
    textbox(slide, rx, HERO_Y + Mm(78), Mm(262), Mm(24),
            ["Not one beat always-cheapest on both counts."],
            size=Pt(21), color=INK, line_spacing=1.22, space_after=Pt(0))


def listing_mock(slide, x, y, w):
    """A deliberately crude search result, drawn in native shapes.

    Not a screenshot and not a UI exercise: it exists so a visitor performs the
    task themselves in about two seconds — spot the blue ones by sight, then
    compare their prices by reading. That is the routing decision, experienced
    rather than described. It is labelled illustrative because inventing a
    benchmark screenshot would misrepresent the data.
    """
    row_h, chrome_h = Mm(12), Mm(10)
    rows = [(C_BLUE, "Sea kayak — blue", "£340"),
            (C_RED, "Kayak — red", "£295"),
            (C_BLUE, "Blue kayak, 2 seat", "£410")]
    total = chrome_h + row_h * len(rows)

    frame = rect(slide, x, y, w, total, fill=PAPER, line=GREY, radius=0.05)
    frame.text_frame.text = ""
    rect(slide, x, y, w, chrome_h, fill=CHROME)
    for i in range(3):
        dot = slide.shapes.add_shape(MSO_SHAPE.OVAL, x + Mm(5) + Emu(i * int(Mm(5))),
                                     y + Mm(3.6), Mm(2.6), Mm(2.6))
        dot.fill.solid()
        dot.fill.fore_color.rgb = GREY
        dot.line.fill.background()
        dot.shadow.inherit = False
    textbox(slide, x + Mm(23), y + Mm(2.9), w - Mm(26), Mm(6), ["classifieds"],
            font=MONO, size=Pt(11), color=MUTED, space_after=Pt(0))

    top = y + chrome_h
    for colour, name, price in rows:
        rect(slide, x + Mm(6), top + Mm(2.4), Mm(11), Mm(7.2), fill=colour,
             radius=0.12)
        textbox(slide, x + Mm(22), top + Mm(3), Mm(105), Mm(7), [name],
                size=Pt(15), color=INK, space_after=Pt(0))
        textbox(slide, x + w - Mm(40), top + Mm(3), Mm(34), Mm(7), [price],
                font=MONO, size=Pt(15), color=INK_STRONG, space_after=Pt(0),
                align=PP_ALIGN.RIGHT)
        top += row_h
        if (colour, name, price) != rows[-1]:
            rect(slide, x + Mm(6), top, w - Mm(12), Mm(0.3), fill=HAIRLINE)

    textbox(slide, x, y + total + Mm(2), w, Mm(6),
            ["ILLUSTRATIVE — NOT A BENCHMARK SCREENSHOT"], font=MONO, size=Pt(10),
            color=MUTED, space_after=Pt(0))
    return total + Mm(8)


def build_main_left(slide):
    """What we actually did — a real task, run six ways, then compared.

    This is the panel a visitor who has never met the problem needs first, and
    the one the previous draft did not have at all.
    """
    x, w = MAIN_L_X, MAIN_L_W
    y = panel_header(slide, x, MAIN_Y, "WHAT WE ACTUALLY DID", w)

    textbox(slide, x, y, w, Mm(6), ["THE AGENT IS GIVEN"], font=MONO, size=Pt(12),
            color=MUTED, space_after=Pt(0))
    textbox(slide, x, y + Mm(7), w, Mm(16),
            ["“Find me the cheapest blue kayak on this site.”"], font=SERIF,
            size=Pt(21), color=INK_STRONG, line_spacing=1.2, space_after=Pt(0))
    textbox(slide, x, y + Mm(24), w, Mm(6),
            ["A REAL TASK — VISUALWEBARENA, CLASSIFIEDS #0"], font=MONO,
            size=Pt(11), color=MUTED, space_after=Pt(0))
    y += Mm(35)

    y += listing_mock(slide, x, y, w)

    intuition = ["**“blue”** can be visual. **“cheapest”** is textual."]
    h = text_height(intuition, SANS, SZ_BODY, w)
    textbox(slide, x, y, w, h, intuition, size=SZ_BODY)
    y += h + Mm(3)
    textbox(slide, x, y, w, Mm(12), ["Should the agent pay to look?"], font=SERIF,
            size=Pt(23), color=ACCENT, line_spacing=1.2, space_after=Pt(0))
    y += Mm(15)

    arrow_down(slide, x + Mm(9), y, Mm(13), label="RUN THE SAME TASK SIX WAYS")
    y += Mm(16)

    groups = [
        ("NO IMAGE AT ALL", ["DOM", "P-text", "P-prompt", "P-SoM"], ACCENT_PALE,
         INK_STRONG),
        ("TEXT + SCREENSHOT", ["SoM"], ACCENT, PAPER),
        ("SCREENSHOT ONLY", ["Vision"], GREY, INK_STRONG),
    ]
    for label, modes, fill, ink in groups:
        textbox(slide, x, y, w, Mm(6), [label], font=MONO, size=Pt(12.5),
                color=MUTED, space_after=Pt(0))
        y += Mm(7)
        gap = Mm(2.5)
        cw = int((w - gap * (len(modes) - 1)) / len(modes))
        for i, mode in enumerate(modes):
            chip(slide, x + Emu(i * (cw + gap)), y, Emu(cw), Mm(13), mode,
                 fill=fill, ink=ink, size=Pt(15) if len(modes) > 1 else Pt(17))
        y += Mm(19)

    arrow_down(slide, x + Mm(9), y, Mm(13),
               label="MEASURE SUCCESS × COST, THEN COMPARE", width=Mm(150))
    y += Mm(17)

    for i, (name, note) in enumerate([
        ("FIXED MODE", "the same choice every task"),
        ("HINDSIGHT ORACLE", "the best choice, known afterwards"),
        ("LEARNED ROUTER", "a choice made before the task runs"),
    ]):
        rect(slide, x, y, Mm(2.2), Mm(11), fill=ACCENT if i == 2 else GREY)
        textbox(slide, x + Mm(7), y, w - Mm(7), Mm(6), [name], font=MONO,
                size=Pt(14), color=INK_STRONG, space_after=Pt(0))
        textbox(slide, x + Mm(7), y + Mm(6), w - Mm(7), Mm(6), [note],
                size=SZ_CAPTION, color=MUTED, space_after=Pt(0))
        y += Mm(16)

    y += Mm(3)
    rect(slide, x, y, w, Mm(0.35), fill=HAIRLINE)
    textbox(slide, x, y + Mm(4), w, Mm(14),
            ["6 modes · 8 benchmark–model settings · 8,934 episodes"],
            font=MONO, size=Pt(16), color=INK_STRONG, line_spacing=1.3,
            space_after=Pt(0))
    return y + Mm(20)


def build_main_right(slide):
    """The result, as the one dominant figure, with its conclusion stated in
    poster type rather than left for the reader to derive from the axes."""
    x, w = MAIN_R_X, MAIN_R_W
    y = panel_header(slide, x, MAIN_Y, "NOTHING WE TRAINED WON", w)

    lead = ["Every policy, in every pair, measured against one simple fixed "
            "baseline: **always use the cheapest mode**. A win means landing in "
            "the shaded region — cheaper *and* no worse."]
    h = text_height(lead, SANS, SZ_BODY, w)
    textbox(slide, x, y, w, h, lead, size=SZ_BODY)
    y += h + Mm(6)

    png = FIGDIR / "poster_dominance_plane.png"
    with Image.open(png) as im:
        pic_h = int(w * im.height / im.width)
    slide.shapes.add_picture(str(png), x, y, width=w)
    y += pic_h + Mm(5)

    verdicts = [
        (C_TEXTSIDE, "0 of 8", "learned routers land in the win region."),
        (C_WIN, "1 of 8", "hindsight oracles clear the same bar — so learning "
                          "failure alone cannot explain this."),
    ]
    for colour, number, rest in verdicts:
        rect(slide, x, y, Mm(2.4), Mm(13), fill=colour)
        box = slide.shapes.add_textbox(x + Mm(8), y - Mm(1), w - Mm(8), Mm(14))
        frame = box.text_frame
        frame.word_wrap = True
        frame.margin_left = frame.margin_right = 0
        frame.margin_top = frame.margin_bottom = 0
        para = frame.paragraphs[0]
        para.line_spacing = 1.2
        _run(para, number + "  ", SANS, Pt(22), colour, True)
        _run(para, rest, SANS, Pt(19), INK, False)
        y += Mm(16)

    textbox(slide, x, y + Mm(1), w, Mm(12),
            ["Always-cheapest is the cheapest fixed policy **on average**, not a "
             "per-episode floor — which is why a few points sit left of it. "
             "Nested cross-validation; 10,000 bundle permutations."],
            size=Pt(15), color=MUTED, line_spacing=1.22, space_after=Pt(0))
    return y + Mm(14)


def build_second_left(slide):
    """The paradox as an editorial statement, not a module.

    It was four rounded boxes joined by arrows — which made the poster's one
    memorable *idea* look like the smallest thing on the sheet. Boxes imply a
    process; this is a claim. So it is set large, unboxed, against whitespace,
    and the two quantities that cause it sit underneath as evidence rather than
    as steps in a flow.

    `2-36%` lives here rather than in the hero: it is the *reason* for the
    paradox, and in the hero it was a third KPI diluting the contradiction.
    """
    x, w = SECOND_L_X, SECOND_L_W
    y = panel_header(slide, x, SECOND_Y, "THE ROUTING PARADOX", w)

    claim = ["Weak agents need routing most —",
             "but produce the least supervision to learn it."]
    h = text_height(claim, SERIF, Pt(31), w, spacing=1.3)
    textbox(slide, x, y + Mm(3), w, h, claim, font=SERIF, size=Pt(31),
            color=INK_STRONG, line_spacing=1.3, space_after=Mm(2))
    y += h + Mm(16)

    for i, (value, label) in enumerate([
        ("2–36%", "OF TASKS THESE AGENTS SOLVE"),
        ("15–97", "USABLE ROUTING LABELS, TYPICALLY"),
    ]):
        cx = x + Emu(i * int(Mm(136)))
        textbox(slide, cx, y, Mm(130), Mm(18), [value], font=MONO, size=Pt(44),
                color=INK_STRONG, line_spacing=1.0, space_after=Pt(0))
        textbox(slide, cx, y + Mm(19), Mm(130), Mm(14), [label], font=MONO,
                size=Pt(14), color=MUTED, line_spacing=1.3, space_after=Pt(0))
    y += Mm(36)

    textbox(slide, x, y, w, Mm(10),
            ["A routing label is born only when the agent solves something."],
            size=SZ_CAPTION, color=MUTED, space_after=Pt(0))
    return y + Mm(10)


def build_second_right(slide):
    """Calibration, then the one deployable-shaped result. Both are asides to
    the main finding and are sized as asides."""
    x, w = SECOND_R_X, SECOND_R_W
    y = panel_header(slide, x, SECOND_Y, "CALIBRATION  ·  WHAT SURVIVES", w)

    card_h = Mm(80)
    rect(slide, x, y, w, card_h, fill=WASH, line=HAIRLINE, radius=0.04)
    cx = x + Mm(8)
    cw = w - Mm(16)
    textbox(slide, cx, y + Mm(5), cw, Mm(14),
            ["Rerunning **one mode on the same tasks** flips **12–14%** of "
             "outcomes. So we measured the ceiling against that ruler:"],
            size=SZ_SMALL, color=INK, line_spacing=1.22, space_after=Pt(0))

    # Two bars on a shared 0-9pp scale. The overlap IS the finding, so they are
    # drawn to scale rather than described; ``FULL`` is the length of 9pp, and
    # every bar is a fraction of it. (Setting FULL to the bar's own width made
    # both bars collapse into stubs in the first draft.)
    # One shared 0-9pp axis, because the two quantities are NOT the same kind of
    # thing and drawing them as two bars said they were: +7.14pp is a point
    # estimate, the rerun floor is an interval. So the rerun is a band and the
    # new-mode gain is a rule through it — which makes the finding visible
    # rather than argued: the gain lands inside the range a plain rerun already
    # covers. It also matches the thesis figure's own blue-point / orange-band
    # encoding, so the two read as the same claim.
    ax_x, FULL, span = cx + Mm(16), Mm(196), 9.0
    ax_y = y + Mm(38)
    rect(slide, ax_x, ax_y + Mm(13), FULL, Mm(0.4), fill=MUTED)
    for tick in (0, 3, 6, 9):
        tx = ax_x + Emu(int(FULL * tick / span))
        rect(slide, tx, ax_y + Mm(13), Mm(0.4), Mm(2), fill=MUTED)
        textbox(slide, tx - Mm(6), ax_y + Mm(15.5), Mm(12), Mm(6), [str(tick)],
                font=MONO, size=Pt(11), color=MUTED, space_after=Pt(0),
                align=PP_ALIGN.CENTER)
    textbox(slide, ax_x, ax_y + Mm(21), FULL, Mm(6),
            ["gain from adding a representation  [pp]"], font=MONO, size=Pt(11),
            color=MUTED, space_after=Pt(0), align=PP_ALIGN.CENTER)

    lo, hi = 4.46, 7.59
    band_x = ax_x + Emu(int(FULL * lo / span))
    rect(slide, band_x, ax_y + Mm(3), Emu(int(FULL * (hi - lo) / span)), Mm(10),
         fill=RGBColor(0xF6, 0xCF, 0xA8))
    textbox(slide, band_x, ax_y + Mm(5.5), Emu(int(FULL * (hi - lo) / span)),
            Mm(6), [f"{lo:.2f}–{hi:.2f}"], font=MONO, size=Pt(12),
            color=RGBColor(0x9A, 0x4A, 0x00), space_after=Pt(0),
            align=PP_ALIGN.CENTER)
    textbox(slide, band_x - Mm(62), ax_y + Mm(4.5), Mm(59), Mm(7),
            ["same-mode rerun"], font=MONO, size=Pt(11.5),
            color=C_TEXTSIDE, space_after=Pt(0), align=PP_ALIGN.RIGHT)

    point = ax_x + Emu(int(FULL * 7.14 / span))
    rect(slide, point - Mm(1), ax_y - Mm(3), Mm(2), Mm(18), fill=ACCENT)
    textbox(slide, point - Mm(24), ax_y - Mm(10), Mm(48), Mm(7), ["7.14"],
            font=MONO, size=Pt(15), color=ACCENT, space_after=Pt(0),
            align=PP_ALIGN.CENTER)
    textbox(slide, point - Mm(48), ax_y - Mm(16), Mm(96), Mm(7),
            ["new representation"], font=MONO, size=Pt(11.5), color=ACCENT,
            space_after=Pt(0), align=PP_ALIGN.CENTER)

    textbox(slide, cx, y + Mm(64), cw, Mm(12),
            ["The gain from adding a representation lands **inside** the range a "
             "plain rerun already covers — not distinguishable here. "
             "B0 · classifieds, n=224, three replicated modes."],
            size=SZ_CAPTION, color=MUTED, line_spacing=1.2, space_after=Pt(0))
    y += card_h + Mm(7)

    rect(slide, x, y, Mm(2.4), Mm(38), fill=C_WIN)
    textbox(slide, x + Mm(8), y - Mm(1), w - Mm(8), Mm(38),
            ["**9.5–30.6% cheaper at identical success, in 8 of 8 pairs** — send "
             "the tasks *nobody* solves to the cheapest mode.",
             "Against the best-success fixed mode, and still a hindsight bound; "
             "in most pairs plain always-cheapest saves more. What makes it "
             "worth naming is the label: *solvable or not* is far easier to "
             "supply than *which mode*."],
            size=SZ_SMALL, color=INK, line_spacing=1.22, space_after=Mm(2))
    return y + Mm(40)


def build_takeaway(slide):
    rect(slide, MARGIN, TAKE_Y, FULL_W, TAKE_H, fill=INK_STRONG)
    rect(slide, MARGIN, TAKE_Y, Mm(3.4), TAKE_H, fill=ACCENT)
    textbox(slide, MARGIN + Mm(14), TAKE_Y + Mm(6), Mm(330), Mm(30),
            ["Routing is not only a model-selection problem. Its learnability "
             "depends on the competence of the agent producing the labels."],
            font=SERIF, size=SZ_TAKEAWAY, color=PAPER, line_spacing=1.2,
            space_after=Pt(0))
    textbox(slide, MARGIN + Mm(14), TAKE_Y + Mm(41), Mm(330), Mm(10),
            ["Measured inside the 2–36% success regime we observed. This "
             "conclusion need not hold for stronger agents."],
            size=Pt(13), color=RGBColor(0xC9, 0xC6, 0xFF), line_spacing=1.2,
            space_after=Pt(0))

    px = MARGIN + Mm(360)
    textbox(slide, px, TAKE_Y + Mm(7), Mm(180), Mm(7), ["SO, IN THIS ORDER"],
            font=MONO, size=Pt(12), color=RGBColor(0x8A, 0x92, 0xB0),
            space_after=Pt(0))
    for i, step in enumerate(["Improve the agent",
                              "Generate reliable supervision",
                              "Then learn selective perception"]):
        top = TAKE_Y + Mm(16) + Emu(i * int(Mm(11)))
        textbox(slide, px, top, Mm(12), Mm(8), [f"{i + 1}"], font=MONO, size=Pt(18),
                color=ACCENT_PALE, space_after=Pt(0))
        textbox(slide, px + Mm(12), top, Mm(170), Mm(8), [step], size=Pt(18),
                color=PAPER, space_after=Pt(0))


# --------------------------------------------------------------------------- run
def main():
    prs = Presentation(str(TEMPLATE))
    slide = prs.slides[0]

    # The template anchors the title box to its BOTTOM edge, so an over-long
    # title grows upward and off the sheet rather than down into the byline.
    title = find(slide, "TextBox 6")
    title.top, title.height = Mm(10), Mm(44)
    title_lines = ["Can a web agent learn when a screenshot is worth the cost?"]
    for line in title_lines:
        n = line_count(line, SERIF, SZ_TITLE.pt, title.width)
        assert n == 1, f"title wraps to {n} lines and will overflow: {line!r}"
    set_text(title, title_lines, size=SZ_TITLE)

    set_text(find(slide, "TextBox 7"),
             ["Jiaming Wei          Supervisors: Prof. María Pérez-Ortiz  ·  "
              "Zekun Wu"])
    set_text(find(slide, "TextBox 8"),
             ["UCL Centre for Artificial Intelligence          Holistic AI"])
    set_text(find(slide, "TextBox 12"),
             ["Hindsight reveals a real routing opportunity. Against "
              "always-cheapest, none of eight learned routers — and only one "
              "of eight hindsight oracles — win on both success and cost. "
              "These agents solve just 2–36% of tasks, starving the router of "
              "the supervision it would need."],
             size=SZ_STANDFIRST)

    global HEADER_PARTS
    HEADER_PARTS = tuple(
        find(slide, n)
        for n in ("Rectangle 13", "TextBox 14", "Rectangle 15", "Rectangle 16"))
    drop(slide, "TextBox 17", "Rectangle 33", "TextBox 34", "Rectangle 35",
         "Rectangle 36", "TextBox 37", "Rectangle 38", "Rectangle 39", "TextBox 42",
         "Rectangle 48", "TextBox 49", "Rectangle 50", "Rectangle 51",
         "Rectangle 52", "TextBox 53", "TextBox 54", "TextBox 55", "TextBox 56",
         "TextBox 57", "TextBox 58")

    build_hero(slide)
    ends = {
        "main-left": (build_main_left(slide), MAIN_BOTTOM),
        "main-right": (build_main_right(slide), MAIN_BOTTOM),
        "second-left": (build_second_left(slide), SECOND_BOTTOM),
        "second-right": (build_second_right(slide), SECOND_BOTTOM),
    }
    build_takeaway(slide)

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
    verify(prs, ends)


def verify(prs, ends):
    """Assert what cannot be checked by eye on a 594x841mm sheet."""
    mm = lambda emu: emu / 914400 * 25.4  # noqa: E731
    w, h = mm(prs.slide_width), mm(prs.slide_height)
    assert abs(w - 594) < 0.5 and abs(h - 841) < 0.5, "slide was resized!"
    print(f"wrote {OUT.relative_to(REPO)}   ({w:.0f}x{h:.0f}mm, A1, not resized)")

    bad = False
    for name, (end, limit) in ends.items():
        slack = mm(limit - end)
        flag = "" if slack >= 0 else "   <-- OVERRUNS ITS BOX"
        bad |= slack < 0
        print(f"  {name:12s} ends {mm(end):6.1f}mm   (box to {mm(limit):.0f}mm, "
              f"slack {slack:+6.1f}mm){flag}")
    if bad:
        raise SystemExit("a panel overran its box — shorten it or move the grid")


if __name__ == "__main__":
    main()
