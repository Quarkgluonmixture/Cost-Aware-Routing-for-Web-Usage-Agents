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
# LOOK's last three are pixel-identical to each other (they belong to a group
# of seven identical frames in that run); READ's four are four different pages.
# That contrast is the strip, so the frames are chosen by what the screenshots
# actually show, not by the step's obs_url — the URL is recorded before the
# action and the screenshot after it, so they disagree.
LANES = {
    "read": [(DOM, 76, 0), (DOM, 76, 2), (DOM, 76, 5), (DOM, 76, 11)],
    "look": [(VIS, 76, 0), (VIS, 76, 3), (VIS, 76, 9), (VIS, 76, 20)],
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
CROP_FRAC = 0.75       # scaler is poorer than PIL's, and the crop keeps the
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


if __name__ == "__main__":
    print("preparing v9 assets")
    main()
