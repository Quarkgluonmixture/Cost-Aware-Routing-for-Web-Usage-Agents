#!/usr/bin/env python3
"""Prepare the v9 poster's raster assets.

v9 is a picture-led sheet: the supervisor's note was that v8 was too text-heavy
and that the reference poster (a NeurIPS three-column landscape sheet) leads
with figures. The paper stays on the organiser's A1 **portrait** template, so
every asset here is cut to a width the portrait grid actually has.

Outputs (all under ``figures/v9/``):

* ``system.png``   the three-panel routing diagram, white margin trimmed. Source
                   is the SVG in ``deliverables/``, rasterised at poster scale.
* ``lane_*.png``   the frames of the two-lane strip (task 76, READ vs LOOK).
                   Enough of each page is kept that its **shape** is readable at
                   column width — a listing grid, a form, an item page. An
                   earlier cut kept only the top 50% and every frame came out as
                   the same navigation bar, which destroyed the comparison the
                   strip exists to make. Nobody reads the body text at this size
                   and nothing here asks them to.
* ``page_*.png``   six real pages, for the "what the agent sees" band.
* ``venn_b0.png``  the task-overlap Venn, cropped to its two B0 panels. The
                   full figure has six, but the B1 and B2 rows are nearly empty
                   circles — those models solve so little that the overlap is
                   not readable at column width, and printing them wastes a
                   third of a column to say nothing.

Every screenshot is a real artifact from a recorded replicate run; the label of
each is checked against that step's own ``obs_url``, because the artifacts
directory name is the *task's* site, not necessarily the *page's* site — one of
these frames is the agent having walked off to a different site entirely.

Usage::

    .venv/bin/python3 deliverables/showcase/poster_figures_v9.py
"""

from __future__ import annotations

from pathlib import Path

import cairosvg
from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures" / "v9"
REPL = REPO / "results" / "repro_replicates"
DOM = REPL / "B0_dom_classifieds_R31194_clean_replicate" / "phase1_dom_router_0"
VIS = REPL / "B0_vision_classifieds_R24792_clean_replicate" / "phase1_vision_router_0"
SVG = REPO / "deliverables" / "representation_routing_merged_three_sections.svg"

# (lane, run, task, step) — the frames that carry the story. LOOK's 8 and 17 are
# the point of the whole strip: two near-identical screenshots of the same empty
# form, sixteen dollars-cents and nine steps apart.
# LOOK's four frames are two round trips: the listings page and the edit form,
# then the *same two pages again* seventeen steps later, pixel for pixel. Two
# earlier cuts misread and are recorded so they are not tried again:
#   * three frames whose actions happened to be `scroll` read as "it scrolled and
#     nothing moved" — the opposite of the run (every scroll succeeds, and LOOK
#     scrolls more than READ);
#   * a round trip through the blank new-listing form read as "same page, form
#     cleared", because OsClass renders `item_add` and `item_edit` from one
#     template — at thumbnail size the two pages differ only in a breadcrumb.
# Alternating two visibly different pages fixes both: the movement is visible,
# and so is the fact that it goes nowhere.
#
# THREE frames, not four. Four made LOOK two pairs (1=18, 3=20), which the reader
# has to match up twice; three makes it form -> listings -> form, where the first
# and last frames are the same page and the loop reads at a glance. It also buys
# 34% more width per frame, and frame width is what this strip kept losing to.
# READ drops its home page: "12 steps" already says it travelled.
#
# Timing, verified against the URLs: `obs_url` is recorded BEFORE the action and
# the screenshot AFTER it, so a frame is the page that step ENDED on.
LANES = {
    "read": [(DOM, 76, 2), (DOM, 76, 5), (DOM, 76, 11)],
    "look": [(VIS, 76, 3), (VIS, 76, 18), (VIS, 76, 20)],
}
# (name, run, task, step) — label checked against the step's obs_url, see module docstring
PAGES = [
    ("front", VIS, 18, 0),
    ("category", DOM, 52, 6),
    ("mine", DOM, 76, 2),
    ("item", DOM, 76, 11),
    ("form", VIS, 76, 8),
    ("offsite", VIS, 200, 9),
]

SHOT_W = 1400          # frames are downscaled here, not in pptx: LibreOffice's
CROP_FRAC = 0.58       # scaler is poorer than PIL's, and the crop keeps the
                       # part of the page where anything actually happens
VENN = REPO / "results" / "phantom_paper" / "figures" / "fig_phantom_structure_venn.png"
VENN_KEEP = 0.34       # the two B0 panels, measured off the rendered figure
CARBON = REPO / "results" / "phantom_paper" / "figures" / "fig3_regional_carbon.png"


def trim(im: Image.Image) -> Image.Image:
    bg = Image.new(im.mode, im.size, (255, 255, 255))
    box = ImageChops.difference(im, bg).getbbox()
    return im.crop(box) if box else im


def shot(run: Path, task: int, step: int, *, crop: float | None = None) -> Image.Image:
    p = run / "artifacts" / f"classifieds_task_{task}" / f"step_{step:03d}" / "screenshot.png"
    if not p.exists():
        raise SystemExit(f"missing screenshot: {p}")
    im = Image.open(p).convert("RGB")
    if crop:
        im = im.crop((0, 0, im.width, int(im.height * crop)))
    return im.resize((SHOT_W, int(SHOT_W * im.height / im.width)), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw = OUT / "_system_raw.png"
    cairosvg.svg2png(url=str(SVG), write_to=str(raw), output_width=3600)
    with Image.open(raw) as im:
        sysfig = trim(im.convert("RGB"))
    sysfig.save(OUT / "system.png")
    raw.unlink()
    print(f"  system.png       {sysfig.size}  ({sysfig.width / sysfig.height:.2f}:1)")

    for lane, frames in LANES.items():
        for i, (run, task, step) in enumerate(frames):
            im = shot(run, task, step, crop=CROP_FRAC)
            im.save(OUT / f"lane_{lane}_{i}.png")
        print(f"  lane_{lane}_*.png   {len(frames)} frames  {im.size}")

    for name, run, task, step in PAGES:
        shot(run, task, step, crop=CROP_FRAC).save(OUT / f"page_{name}.png")
    print(f"  page_*.png       {len(PAGES)} pages")

    with Image.open(VENN) as im:
        v = im.convert("RGB").crop((0, 0, im.width, int(im.height * VENN_KEEP)))
    v.save(OUT / "venn_b0.png")
    print(f"  venn_b0.png      {v.size}  ({v.width / v.height:.2f}:1)")

    # The carbon figure is two side-by-side panels; at column width its region
    # labels are unreadable. One panel makes the same point at twice the type size.
    # its title spans both panels, so keeping half of it reads as a truncation;
    # the caption on the sheet says what the panel is instead.
    with Image.open(CARBON) as im:
        c = im.convert("RGB").crop((0, int(im.height * 0.09), im.width // 2, im.height))
    c.save(OUT / "carbon_cls.png")
    print(f"  carbon_cls.png   {c.size}  ({c.width / c.height:.2f}:1)")




# ---------------------------------------------------------------- new for v9.3
# Two figures the poster needs and the existing libraries do not have in a form
# that survives a 179mm column. `fig1c_strategy_gradient` is 24 small panels and
# `fig_failure_modes_per_cell` is 30 stacked rows: both are readable on a screen
# and unreadable on a board. These draw the same underlying quantities at poster
# scale, one question each.

PROFILE = REPO / "docs/analysis/cross_sites/per_mode_four_dimension_profile_with_wa.json"
MODE_ORDER = ["DOM", "P-text", "P-prompt", "P-SoM", "SoM", "Vision"]
MODE_COLOUR = {"DOM": "#E8720C", "P-text": "#E8720C", "P-prompt": "#E8720C",
               "P-SoM": "#E8720C", "SoM": "#14855F", "Vision": "#1F5FD6"}
POSTER_COL_IN = 173 / 25.4      # inner width of a poster column's figure plate


def behaviour() -> None:
    """How the six views spend their steps, over all 8 cells.

    One question: does the choice of view change what the agent *does*? Two
    measures answer it and disagree in direction, which is why both are drawn:
    Vision scrolls far more and types far less. The min–max whisker is the point
    of the figure — Vision's scroll range does not overlap any other view's, so
    this is not a difference of averages."""
    import json
    import statistics as st
    import matplotlib.pyplot as plt

    cells = json.loads(PROFILE.read_text())["cells"]

    def series(metric):
        return {m: [c["per_mode"][m][metric] for c in cells if m in c["per_mode"]]
                for m in MODE_ORDER}

    fig, axes = plt.subplots(1, 2, figsize=(POSTER_COL_IN, 2.55), sharey=True)
    for ax, (metric, title) in zip(axes, [("scroll_frac", "scrolling"),
                                          ("type_frac", "typing")]):
        data = series(metric)
        ys = range(len(MODE_ORDER))
        for y, m in zip(ys, MODE_ORDER):
            v = data[m]
            lo, hi, mid = min(v), max(v), st.median(v)
            ax.plot([lo * 100, hi * 100], [y, y], lw=2.2, solid_capstyle="round",
                    color=MODE_COLOUR[m], alpha=0.35, zorder=2)
            ax.plot([mid * 100], [y], "o", ms=9, color=MODE_COLOUR[m], zorder=3)
        ax.set_yticks(list(ys))
        ax.set_yticklabels(MODE_ORDER, fontsize=13)
        ax.set_xlabel(f"% of steps {title}", fontsize=13)
        ax.tick_params(labelsize=12)
        ax.invert_yaxis()
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.4, w_pad=1.2)
    out = OUT / "behaviour.png"
    fig.savefig(out, dpi=340, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  behaviour.png    {out.stat().st_size // 1024}KB")


# Table 41 of the REALM evidence inventory. Enrichment = how often the losing
# side hit that failure rule on the disagreement set, over how often it hits it
# across all of its failures. 1.0x means "it failed here the way it fails
# everywhere". The two blocks are NOT comparable to each other (four text arms
# against two image arms), which is why each is drawn against the same 1.0x
# reference rather than against the other block.
FAILURE = [
    ("When the picture won,\nthe text-only agent…", "#E8720C", [
        ("gave up once it could not find it", 2.31),
        ("clicked and went back, over and over", 2.25),
        ("could not see what the task asked about", 2.24),
        ("was on a visual page with no screenshot", 1.65),
    ]),
    ("When the text won,\nthe picture-only agent…", "#1F5FD6", [
        ("clicked and went back, over and over", 1.17),
        ("never turned the page", 0.93),
        ("ran out of budget, unfinished", 0.91),
    ]),
]


def failure() -> None:
    """Do the two sides fail the same way? One question, one reading direction.

    Everything is drawn against the same 1.0x line, never against the other
    block: the text side pools four arms and the image side two, so their task
    counts are not comparable. Omitted from the lower block is a 3.61x row that
    rests on 8 hits, all of them in the two WebArena cells — the poster caption
    says so rather than the figure implying a named cause that is not there."""
    import matplotlib.pyplot as plt

    # lay the rows out first, with a slot reserved for each block heading, so
    # that a heading never lands on top of a bar
    slots, y = [], 0.0
    for head, col, items in FAILURE:
        slots.append(("head", head, col, y))
        y += 1.55                      # the headings run to two lines
        for lab, v in items:
            slots.append(("bar", (lab, v), col, y))
            y += 1.0
        y += 0.5

    fig, ax = plt.subplots(figsize=(POSTER_COL_IN, 3.15))
    for kind, payload, col, yy in slots:
        if kind == "head":
            ax.text(0.0, yy - 0.1, payload, ha="left", va="center", fontsize=13,
                    fontweight="bold", color=col, linespacing=1.3)
            continue
        lab, v = payload
        ax.barh(yy, v, height=0.66, color=col, alpha=0.9, zorder=3)
        ax.text(v + 0.05, yy, f"{v:.1f}×", va="center", fontsize=13.5,
                fontweight="bold", color=col, zorder=4)
        ax.text(-0.05, yy, lab, ha="right", va="center", fontsize=12.5,
                color="#222222")
    ax.axvline(1.0, color="#555555", lw=1.6, zorder=5)
    ax.text(1.0, y - 0.3, "as often as it fails anywhere", ha="center",
            va="center", fontsize=11.5, color="#555555")
    ax.set_xlim(0, 2.8)
    ax.set_ylim(y, -0.7)
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout(pad=0.3)
    out = OUT / "failure.png"
    fig.savefig(out, dpi=340, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  failure.png      {out.stat().st_size // 1024}KB")


if __name__ == "__main__":
    print("preparing v9 assets")
    main()
    behaviour()
    failure()
