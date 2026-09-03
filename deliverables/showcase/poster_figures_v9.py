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

import sys
from pathlib import Path

import cairosvg
from PIL import Image, ImageChops

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "analysis" / "figures" / "thesis"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
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
# Three frames, laid out as a controlled comparison. Every pairing below was
# picked by measured frame difference (mean abs diff on 160x90 greyscale), not
# by step number — neighbouring steps in these runs are often the same page.
#
#   frame 1  both lanes filling in the same form   READ 5 vs LOOK 3 : 0.16
#   frame 2  each lane after it has diverged       READ 9 vs LOOK 18: 19.87
#   frame 3  READ finishes; LOOK is back on its own frame 1
#                                                  LOOK 20 vs LOOK 3: 0.00
#
# Timing, verified on READ step 0, whose screenshot is the site's front page
# while its `obs_url` is already `user&action=items`. The only self-consistent
# reading: **the screenshot is the page BEFORE the action, the `obs_url` is the
# URL AFTER it**. So `step_N/screenshot.png` shows what step N-1 produced.
#
# The step numbers below are therefore the step whose action CAUSED the frame;
# `shot()` reads N+1's screenshot. The sheet then says "step 4 · type" over the
# page that typing produced, instead of over the page it was typed on.
LANES = {
    "read": [(DOM, 76, 4), (DOM, 76, 8), (DOM, 76, 10)],
    "look": [(VIS, 76, 2), (VIS, 76, 17), (VIS, 76, 19)],
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


def shot(run: Path, task: int, step: int, *, crop: float | None = None,
         after: bool = False) -> Image.Image:
    """``after`` reads the screenshot that this step's action produced, which
    the runner files under the NEXT step (see the timing note above LANES)."""
    if after:
        step += 1
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
            im = shot(run, task, step, crop=CROP_FRAC, after=True)
            im.save(OUT / f"lane_{lane}_{i}.png")
        print(f"  lane_{lane}_*.png   {len(frames)} frames  {im.size}")

    for name, run, task, step in PAGES:
        shot(run, task, step, crop=CROP_FRAC).save(OUT / f"page_{name}.png")
    print(f"  page_*.png       {len(PAGES)} pages")

    with Image.open(VENN) as im:
        # from just under the figure's own title, which is unreadable at column
        # width and only adds noise, down through the two B0 panels
        v = im.convert("RGB").crop((0, int(im.height * 0.052), im.width,
                                    int(im.height * VENN_KEEP)))
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
    ("TEXT-ONLY", "#E8720C", [
        ("early give-up", 2.31),
        ("action loop", 2.25),
        ("target unseen", 2.24),
        ("visual state unseen", 1.65),
    ]),
    ("IMAGE-ONLY", "#1F5FD6", [
        ("action loop", 1.17),
        ("no page advance", 0.93),
        ("budget exhausted", 0.91),
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
        y += 1.15                      # one-line headings
        for lab, v in items:
            slots.append(("bar", (lab, v), col, y))
            y += 1.0
        y += 0.5

    fig, ax = plt.subplots(figsize=(POSTER_COL_IN, 2.6))
    for kind, payload, col, yy in slots:
        if kind == "head":
            ax.text(0.0, yy - 0.1, payload, ha="left", va="center", fontsize=15,
                    fontweight="bold", color=col)
            continue
        lab, v = payload
        ax.barh(yy, v, height=0.66, color=col, alpha=0.9, zorder=3)
        ax.text(v + 0.05, yy, f"{v:.1f}×", va="center", fontsize=16,
                fontweight="bold", color=col, zorder=4)
        ax.text(-0.05, yy, lab, ha="right", va="center", fontsize=15,
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




# ------------------------------------------------------------- v9.7 de-noising
# Two thesis figures carry annotations written for a page a reader holds at
# 40cm. On a board they are noise: the reader has three seconds, and a
# three-line gloss inside the plot spends all of it. These redraw the same
# figures with the explanation moved out to the caption, where the poster
# already says it.

def routing() -> None:
    """Panel 5 at poster scale: the win region named, and nothing else labelled.

    Everything removed here (the Pareto gloss, the per-cell point names, the
    'costs nothing to implement' note) is still true and still in the thesis —
    it is simply not what this panel is for. The panel has one job: show that
    the shaded region is empty."""
    import matplotlib.pyplot as plt
    import _style as S
    import fig_f13_dominance_plane as f

    rows, _ = f.load(f.SRC)
    S.FS_TICK, S.FS_LABEL, S.FS_VALUE, S.FS_PANEL = 17.0, 20.0, 18.0, 22.0
    S.apply()
    fig, ax = plt.subplots(figsize=(POSTER_COL_IN, 3.05))
    f.build(ax, rows)

    # the shaded band is only about a fifth of the axis wide, so its label has
    # to be sized to the band, not to the panel
    for t in list(ax.texts):
        if "Pareto-dominates" in t.get_text():
            t.set_text("WIN REGION\ncheaper, no worse")
            t.set_fontsize(14)
            t.set_linespacing(1.2)
        else:
            t.remove()      # per-cell point names, and the star's own gloss —
                            # the x axis already says what the origin is
    ax.set_xlabel("cost, relative to always-cheapest", fontsize=18)
    ax.set_ylabel("success gain (pp)", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.legend(loc="lower right", frameon=False, handletextpad=0.3,
              borderpad=0.1, labelspacing=0.2, fontsize=13.5, markerscale=0.85)
    fig.tight_layout(pad=0.3)
    out = OUT / "routing.png"
    fig.savefig(out, dpi=340, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  routing.png      {out.stat().st_size // 1024}KB")


def label_supply() -> None:
    """Panel 6 at poster scale, with the x axis reading in the caption's
    direction: the caption says *more routing upside, less usable signal*, so
    the axis has to say which way "more upside" is. Left, because a weaker
    agent has more to gain — that is the whole point and it is not obvious."""
    import json
    import re
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import _style as S
    import poster_figures as P8

    rows, trainable = {}, {}
    text = P8.LABEL_SUPPLY_MD.read_text(encoding="utf-8")
    for m in re.finditer(r"^\| (B\d)_(classifieds|reddit) \| (\d+) \| \*\*(\d+)\*\* \| ([\d.]+)% \| (\d)/6 \|", text, re.M):
        rows[(m.group(2), m.group(1))] = int(m.group(4))
    for m in re.finditer(r"^\| (B\d)_(classifieds|reddit) \| (\d+) \| \d \| [^|]+ \| (\*\*no\*\*|yes) \|", text, re.M):
        trainable[(m.group(2), m.group(1))] = (m.group(4) == "yes")
    assert len(rows) == 6 and len(trainable) == 6, (rows, trainable)
    sr = {(c["site"], c["baseline_model"]): c["baseline_policy"]["sr_pct"]
          for c in json.loads(P8.LEARN_JSON.read_text())["cells"]}

    S.apply()
    fig, ax = plt.subplots(figsize=(POSTER_COL_IN, 2.75))
    for key, n in rows.items():
        ax.scatter([sr[key]], [n], s=300, marker="o",
                   facecolors=S.C_INK if trainable[key] else "white",
                   edgecolors=S.C_INK, linewidths=2.4, zorder=3)
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 112)
    # the direction the caption depends on, said on the axis itself rather than
    # floated next to it, where it lands on the label
    ax.set_xlabel("tasks the best single view solves  (%)\n← more routing upside",
                  fontsize=18, linespacing=1.5)
    ax.set_ylabel("usable labels\nfor “which view”", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.legend([Line2D([], [], marker="o", ls="", ms=15, mfc=S.C_INK, mec=S.C_INK),
               Line2D([], [], marker="o", ls="", ms=15, mfc="white", mec=S.C_INK, mew=2.4)],
              ["enough to train a classifier", "not enough"], loc="upper left",
              frameon=False, fontsize=18, handletextpad=0.3, labelspacing=0.3)
    fig.tight_layout(pad=0.3)
    out = OUT / "label_supply.png"
    fig.savefig(out, dpi=340, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  label_supply.png {out.stat().st_size // 1024}KB")


if __name__ == "__main__":
    print("preparing v9 assets")
    main()
    behaviour()
    failure()
    routing()
    label_supply()
