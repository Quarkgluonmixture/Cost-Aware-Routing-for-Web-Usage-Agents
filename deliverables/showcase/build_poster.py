#!/usr/bin/env python3
"""Build the Holistic AI x UCL CDI showcase poster from the supplied A1 template.

v4 — the template's own skeleton, one system diagram, one type scale (2026-09-02)
-----------------------------------------------------------------------------------
Reviewer brief (Zekun's colleague, 2026-09-02, against the v2 exhibition
design): *one main system diagram; compact; font sizes and formats consistent;
stay on the template.* v4 is that brief and nothing else. Every number, scope
line and baseline name is carried over from v2/v3 unchanged.

    STANDFIRST   the template band: one sentence, both baselines named, 2 lines
    FIG 1        the system diagram (thesis Fig 1.1) across the full width, in
                 the template's figure box
    COLUMN 1     THE PROBLEM -> WHY IT CANNOT BE LEARNED -> HOW MUCH IS NOISE
    COLUMNS 2-3  RESULTS: the template's metric strip, Fig 2 (thesis F13) in
                 the template's figure box, three verdicts; then TAKEAWAY

Type scale — read from the template's own placeholders, not chosen here:

    title 41 Georgia · standfirst 28 Georgia · section header 14 Consolas bold
    body 17.65 Arial · caption 12.7 Arial grey · metric 32.5 / 10.6 Consolas

Body emphasis is bold only. There is no pull-quote size, no hero size and no
serif in the body: the v2 sheet used fourteen sizes across three families, and
"consistent" was the reviewer's word for what that cost.

What is load-bearing (unchanged from v2)
-----------------------------------------
*Text is measured, not estimated.* Arimo is metric-compatible with Arial, so
measuring with it reproduces PowerPoint's line breaks; a rendered line advances
by the font's ascent + descent, then by the paragraph's line spacing.

*The template's header assembly is a mould, not a part.* It is cloned per panel
and the originals dropped, or the first panel lands on top of the placeholder.
The figure box and the metric strip are rebuilt from the placeholders' own
fill / line / type values (they are read once from the template at build time).

*Two baselines exist and must never be silently merged.* The hindsight ceiling
(+3.45 to +16.35pp, 1.6-35.3% cheaper) is measured against the BEST-SUCCESS
FIXED MODE; the 0/8 learnability result against ALWAYS-CHEAPEST. Every number
prints its own baseline — in the metric strip, in its own label line.

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
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Mm, Pt

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "Showcase-Poster-Template-A1.pptx"
OUT = HERE / "poster_jiaming_wei.pptx"
FIGDIR = HERE / "figures"

REPO_URL = "https://github.com/Quarkgluonmixture/Cost-Aware-Routing-for-Web-Usage-Agents"

# ---------------------------------------------------------------- palette / type
# Colours are the template's own (read off its placeholders) plus the thesis
# figure palette, so a colour met in a panel means the same thing in the figure
# beside it: orange = text only, green = text plus image, blue = image only.
INK = RGBColor(0x12, 0x16, 0x2E)
INK_STRONG = RGBColor(0x14, 0x1E, 0x41)
MUTED = RGBColor(0x5D, 0x67, 0x87)
ACCENT = RGBColor(0x50, 0x49, 0xF9)
HAIRLINE = RGBColor(0xDD, 0xE3, 0xF2)
FIG_FILL = RGBColor(0xF7, 0xF9, 0xFD)
METRIC_FILL = RGBColor(0xEC, 0xEF, 0xFA)
C_TEXT = RGBColor(0xE8, 0x72, 0x0C)
C_BOTH = RGBColor(0x14, 0x85, 0x5F)
C_IMAGE = RGBColor(0x1F, 0x5F, 0xD6)

SERIF, SANS, MONO = "Georgia", "Arial", "Consolas"

# Metric-compatible stand-ins for measurement.
FONT_FILE = {
    (SANS, False): "/usr/share/fonts/truetype/croscore/Arimo-Regular.ttf",
    (SANS, True): "/usr/share/fonts/truetype/croscore/Arimo-Bold.ttf",
    (SERIF, False): "/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf",
    (SERIF, True): "/usr/share/fonts/truetype/noto/NotoSerif-Bold.ttf",
}
# A rendered line advances by (ascent + descent) x line spacing. The face files
# give 1.118 for Arimo, but the PDF is exported by LibreOffice, which advances
# Liberation Sans 9.3mm per line at 17.65pt x 1.25 (measured on the 100dpi
# render of the v4 draft), i.e. a factor of 1.20. Using the file value
# under-counts every paragraph by ~7%, which is exactly enough to close the gap
# above the next section header. Calibrate to the renderer, not the font.
LINE_HEIGHT = {SANS: 1.20, SERIF: 1.20}

# The template's type scale. Values are the placeholders' own, to the hundredth,
# so the sheet cannot drift from the organisers' design by rounding.
SZ_TITLE = Pt(40.96)
SZ_STANDFIRST = Pt(28)
SZ_BODY = Pt(17.65)
SZ_CAPTION = Pt(12.71)
SZ_METRIC = Pt(32.48)
SZ_METRIC_LABEL = Pt(10.59)

BODY_SPACING = 1.2       # the template's 1.36 is airy; the brief said compact
CAPTION_SPACING = 1.32   # template
PARA_GAP = Mm(3.5)

# ------------------------------------------------------------------------- grid
# The template's three columns, exactly as placed in the file (x / width in mm).
COL_X = (Mm(20.2), Mm(209.3), Mm(398.3))
COL_W = Mm(179.2)
SPAN_23_X, SPAN_23_W = Mm(209.3), Mm(368.2)          # columns 2+3 with the gutter
FULL_X, FULL_W = Mm(20.2), Mm(557.3)                 # column 1 to column 3

FIG_Y = Mm(140)
ROW_Y = Mm(392)
ROW_BOTTOM = Mm(782)                                 # footer band starts at 789


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


def text_height(paragraphs, family, size, width_emu, *, spacing, gap) -> int:
    size_pt = size.pt
    n = sum(line_count(p, family, size_pt, width_emu) for p in paragraphs)
    advance = size_pt * LINE_HEIGHT[family] * spacing
    return int(n * advance * 12700) + gap * (len(paragraphs) - 1)


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
            align=PP_ALIGN.LEFT, bold=False):
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
        _emit(para, text, font=font, size=size, color=color, bold=bold)
    return box


def body(slide, x, y, w, paragraphs, *, after=Mm(4)):
    """Body copy at the template's body size; returns the y just below it."""
    h = text_height(paragraphs, SANS, SZ_BODY, w, spacing=BODY_SPACING, gap=PARA_GAP)
    textbox(slide, x, y, w, h, paragraphs)
    return y + h + after


def caption(slide, x, y, w, paragraphs, *, after=Mm(4)):
    """Caption / scope copy at the template's caption size and grey."""
    h = text_height(paragraphs, SANS, SZ_CAPTION, w, spacing=CAPTION_SPACING,
                    gap=Mm(1.5))
    textbox(slide, x, y, w, h, paragraphs, size=SZ_CAPTION, color=MUTED,
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


def rect(slide, x, y, w, h, fill=None, line=None, *, lw=Pt(0.7)):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
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
    shape.text_frame.word_wrap = True
    return shape


HEADER_PARTS: tuple = ()


def panel_header(slide, x, y, label, width):
    """Clone the template's section-header assembly at an arbitrary position,
    at the template's own header size."""
    block, title, rule_a, rule_b = HEADER_PARTS
    base = block.top
    clone(slide, block, x, y)
    t = clone(slide, title, x + (title.left - block.left), y - Mm(1.2))
    ra = clone(slide, rule_a, x, y + (rule_a.top - base))
    rb = clone(slide, rule_b, x + (rule_b.left - block.left),
               y + (rule_b.top - base))
    rb.width = width - ra.width - (rb.left - x - ra.width)
    set_text(t, [label])
    return y + Mm(15)


def fig_box(slide, x, y, w, png: Path, cap: str):
    """The template's figure box (Rectangle 38/39): off-white card, hairline,
    accent bar along the top, then the caption in the template's caption style.
    Returns the y just below the caption."""
    inset = Mm(3)
    with Image.open(png) as im:
        pic_w = w - 2 * inset
        pic_h = int(pic_w * im.height / im.width)
    box_h = Mm(1.4) + inset + pic_h + inset
    rect(slide, x, y, w, box_h, fill=FIG_FILL, line=HAIRLINE)
    rect(slide, x, y, w, Mm(1.4), fill=ACCENT)
    slide.shapes.add_picture(str(png), x + inset, y + Mm(1.4) + inset, width=pic_w)
    return caption(slide, x, y + box_h + Mm(2.5), w, [cap], after=Mm(3))


def metric_strip(slide, x, y, w, tiles):
    """The template's metric strip (Rectangle 52 + three number/label pairs).
    Each label names its own baseline: the strip is exactly the place where two
    different baselines would otherwise be read as one comparison."""
    h = Mm(32.5)
    rect(slide, x, y, w, h, fill=METRIC_FILL)
    pad = Mm(7)
    tile_w = int((w - 2 * pad) / len(tiles))
    for i, (number, label) in enumerate(tiles):
        # DejaVu Sans Mono stands in for Consolas; both are ~0.6em per glyph.
        label_w = len(label) * SZ_METRIC_LABEL.pt * 0.6 * 12700
        assert label_w < tile_w - Mm(4), f"metric label overprints the next tile: {label!r}"
        tx = x + pad + Emu(i * tile_w)
        textbox(slide, tx, y + Mm(5), Emu(tile_w), Mm(17), [number], font=MONO,
                size=SZ_METRIC, color=INK_STRONG, bold=True, line_spacing=1.18,
                space_after=Pt(0))
        textbox(slide, tx, y + Mm(20.5), Emu(tile_w), Mm(9), [label], font=MONO,
                size=SZ_METRIC_LABEL, color=MUTED, line_spacing=1.18,
                space_after=Pt(0))
    return y + h + Mm(4)


# ------------------------------------------------------------------------ panels
def build_system(slide):
    """Fig 1, full width: the one system diagram (thesis Fig 1.1)."""
    y = panel_header(slide, FULL_X, FIG_Y, "HOW A WEB AGENT SEES A PAGE, AND WHAT WE COMPARED",
                     FULL_W)
    return fig_box(
        slide, FULL_X, y, FULL_W, FIGDIR / "poster_overview.png",
        "**Fig 1.** The agent (①) is held fixed — same task, actions, step limit "
        "and cost accounting — and only the page encoding it is handed (②) "
        "varies: six observation modes, DOM, three text-only variants (P-text, "
        "P-prompt, P-SoM), SoM and Vision. Each runs on 8 website–model settings "
        "from VisualWebArena and WebArena (8,934 episodes). Three policies (③) "
        "are compared: a fixed mode, a hindsight oracle that picks the best mode "
        "per task after the fact, and a learned router that must choose before "
        "the task runs.")


def build_column_1(slide):
    x, w = COL_X[0], COL_W
    y = panel_header(slide, x, ROW_Y, "THE PROBLEM", w)
    y = body(slide, x, y, w, [
        "A web agent looks at a page, decides what to click or type, acts, and "
        "repeats. Before each step it must be handed some encoding of the page.",
        "That encoding is usually chosen once and paid for at every step: cheap "
        "text, an expensive annotated screenshot, or both.",
    ])

    # One real page, sent three ways. Values from the thesis F1 pipeline
    # (fig_f1_motivating_example.gather(): B0 x classifieds task 0, step 000).
    y = caption(slide, x, y, w, ["**The same page, sent three ways** · B0 · "
                                 "classifieds task 0 · first step"], after=Mm(1))
    rows = [("DOM", C_TEXT, "3,314 tokens", "text only, no image"),
            ("SoM", C_BOTH, "4,335 tokens", "text + 143 KB marked image"),
            ("Vision", C_IMAGE, "3,123 tokens", "110 KB screenshot, no text")]
    for name, colour, tokens, note in rows:
        rect(slide, x, y, w, Mm(0.3), fill=HAIRLINE)
        textbox(slide, x, y + Mm(1.5), Mm(28), Mm(7), [name], color=colour,
                bold=True, space_after=Pt(0))
        textbox(slide, x + Mm(28), y + Mm(1.5), Mm(48), Mm(7), [tokens],
                color=INK_STRONG, bold=True, space_after=Pt(0))
        textbox(slide, x + Mm(78), y + Mm(2.6), w - Mm(78), Mm(6), [note],
                size=SZ_CAPTION, color=MUTED, space_after=Pt(0))
        y += Mm(9)
    rect(slide, x, y, w, Mm(0.3), fill=HAIRLINE)
    y = caption(slide, x, y + Mm(2), w, [
        "SoM's text is within 1% of DOM's; nearly all of its extra cost is the "
        "image."], after=Mm(3))
    y = body(slide, x, y, w, [
        "**Is the screenshot needed at every step — and can the steps that need "
        "it be identified cheaply enough to be worth identifying?**"],
        after=Mm(7))

    y = panel_header(slide, x, y, "WHY IT CANNOT BE LEARNED", w)
    y = body(slide, x, y, w, [
        "A routing label exists only when the agent solves a task. Here the best "
        "single mode solves just **2–36%** of tasks, which leaves typically "
        "**15–97** usable labels per setting.",
        "**The agents that would gain most from routing produce the least "
        "supervision to learn it.**",
        "Deliberately shrinking the training data confirms scarcity is the "
        "mechanism, and prices it: the failing settings would need at least "
        "**2.1–4.2×** more tasks than the benchmarks contain — a specification, "
        "not an impossibility."], after=Mm(7))

    y = panel_header(slide, x, y, "HOW MUCH OF THIS IS NOISE?", w)
    y = body(slide, x, y, w, [
        "Rerunning **one unchanged mode** on the same tasks flips **10–14%** of "
        "outcomes and by itself buys **2.0–7.6 pp** of success (B0 · "
        "classifieds, six replicated modes, n=224).",
        "Every gain on this sheet is read against that band, not against zero."],
        after=Mm(0))
    return y


def build_columns_2_3(slide):
    x, w = SPAN_23_X, SPAN_23_W
    y = panel_header(slide, x, ROW_Y, "RESULTS", w)
    y = metric_strip(slide, x, y, w, [
        ("+16.35 pp", "CEILING, LARGEST OF 8 · VS BEST FIXED MODE"),
        ("0 of 8", "LEARNED ROUTERS BEAT ALWAYS-CHEAPEST"),
        ("1 of 8", "HINDSIGHT ORACLES BEAT ALWAYS-CHEAPEST"),
    ])
    y = fig_box(
        slide, x, y, w, FIGDIR / "poster_dominance_plane.png",
        "**Fig 2.** Every policy in every setting against one fixed baseline, "
        "**always use the cheapest mode** (★). A win lands in the shaded region: "
        "cheaper *and* no worse. Always-cheapest is cheapest on average, not per "
        "episode, which is why a few points sit left of it. Nested "
        "cross-validation; 10,000 bundle permutations.")

    y = body(slide, x, y, w, [
        "**The ceiling is real.** In hindsight, choosing the mode per task solves "
        "**+3.45 to +16.35 pp** more than the best single fixed mode, at "
        "1.6–35.3% lower cost, in 8 of 8 settings.",
        "**Nothing we trained wins.** **0 of 8** learned routers beat "
        "always-cheapest on both success and cost — and even the hindsight "
        "oracle does so in only **1 of 8**.",
        "**What survives is a bound, not a router.** Sending the tasks nobody "
        "solves to the cheapest mode saves 9.5–30.6% at identical success in "
        "8 of 8 — against the best-success fixed mode, and plain always-cheapest "
        "usually saves more."], after=Mm(5))

    y = panel_header(slide, x, y, "TAKEAWAY", w)
    y = body(slide, x, y, w, [
        "**Routing is not only a model-selection problem: its learnability "
        "depends on the competence of the agent producing the labels.** So, in "
        "this order: improve the agent, then generate reliable supervision, then "
        "learn selective perception."], after=Mm(1.5))
    y = caption(slide, x, y, w, [
        "Measured inside the 2–36% success regime we observed. This conclusion "
        "need not hold for stronger agents."], after=Mm(0))
    return y


# --------------------------------------------------------------------------- run
STANDFIRST = (
    "Choosing the page representation per task could solve up to 16 more tasks "
    "in 100 than the best fixed mode — yet none of 8 learned routers beat "
    "always using the cheapest mode on both success and cost."
)


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
    set_text(title, title_lines)

    set_text(find(slide, "TextBox 7"),
             ["Jiaming Wei          Supervisors: Prof. María Pérez-Ortiz  ·  "
              "Zekun Wu"])
    set_text(find(slide, "TextBox 8"),
             ["UCL Centre for Artificial Intelligence          Holistic AI"])
    standfirst = find(slide, "TextBox 12")
    # Georgia is wider than the Noto Serif used to measure it; 6% is the margin
    # observed between the two on this sentence's own words.
    n = line_count(STANDFIRST, SERIF, SZ_STANDFIRST.pt, int(standfirst.width * 0.94))
    assert n <= 2, f"standfirst wraps to {n} lines at 28pt and will leave its band"
    set_text(standfirst, [STANDFIRST])

    global HEADER_PARTS
    HEADER_PARTS = tuple(
        find(slide, n)
        for n in ("Rectangle 13", "TextBox 14", "Rectangle 15", "Rectangle 16"))
    drop(slide, "TextBox 17", "Rectangle 33", "TextBox 34", "Rectangle 35",
         "Rectangle 36", "TextBox 37", "Rectangle 38", "Rectangle 39", "TextBox 42",
         "Rectangle 48", "TextBox 49", "Rectangle 50", "Rectangle 51",
         "Rectangle 52", "TextBox 53", "TextBox 54", "TextBox 55", "TextBox 56",
         "TextBox 57", "TextBox 58")

    fig_end = build_system(slide)
    assert fig_end <= ROW_Y, f"Fig 1 ends at {fig_end / 36000:.1f}mm, past the row start"
    ends = {
        "column 1": (build_column_1(slide), ROW_BOTTOM),
        "columns 2-3": (build_columns_2_3(slide), ROW_BOTTOM),
    }

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
    """Assert what cannot be checked by eye on a 594x841mm sheet."""
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
        print(f"  {name:12s} ends {mm(end):6.1f}mm   (box to {mm(limit):.0f}mm, "
              f"slack {slack:+6.1f}mm){flag}")
    if bad:
        raise SystemExit("a panel overran its box — shorten it or move the grid")


if __name__ == "__main__":
    main()
