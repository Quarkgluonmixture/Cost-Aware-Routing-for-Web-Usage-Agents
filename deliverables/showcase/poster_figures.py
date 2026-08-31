#!/usr/bin/env python3
"""Render the poster's figures at the size they will actually be printed.

``_style.py`` opens with the rule this script obeys: *author at the printed
width*. A figure drawn 5.12in wide for the A4 text block and then dropped into a
poster panel is barely scaled, so its 8.5pt labels land on the A1 sheet at about
10pt — legible at arm's length, not from across a room.

The dominance plane is the poster's one quantitative hero, so it is authored at
343mm (13.5in) and its type is set for a two-metre read. Nothing in
``final_dissertation/`` is touched: this imports the thesis script's own
``build()`` and overrides only the shared style constants, so the poster cannot
drift from the artefact the thesis parses.

The rerun-noise result is deliberately NOT rendered here. It used to be an
eight-row forest plot competing with the main result for visual weight; on the
poster it is a calibration aside, and ``build_poster.py`` draws it as two
overlapping bars in native shapes — crisper at A1 and readable in one glance.

Usage::

    .venv/bin/python3 deliverables/showcase/poster_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THESIS_FIGS = REPO / "scripts" / "analysis" / "figures" / "thesis"
OUTDIR = Path(__file__).resolve().parent / "figures"

sys.path.insert(0, str(THESIS_FIGS))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import _style as S  # noqa: E402

# Width of the hero panel on the A1 sheet. The figure is authored at exactly
# this, so the point sizes below are literal: FS_LABEL 22 renders 22pt on paper.
HERO_W_IN = 343 / 25.4

S.FS_TICK = 18.0
S.FS_LABEL = 22.0
S.FS_VALUE = 20.0
S.FS_PANEL = 24.0


def save(fig, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    path = OUTDIR / f"{name}.png"
    fig.savefig(path, dpi=350, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path.relative_to(REPO)}")


def dominance_plane() -> None:
    import fig_f13_dominance_plane as f

    rows, _proto = f.load(f.SRC)
    S.apply()
    fig, ax = plt.subplots(figsize=(HERO_W_IN, 8.3))
    f.build(ax, rows)

    x0, _ = ax.get_xlim()
    for artist in list(ax.texts):
        text = artist.get_text()
        if "Pareto" in text:
            # A three-line claim anchored inside the axes; at poster type it
            # covers the y-axis label and two oracle marks. It becomes a label
            # set vertically inside the (empty) region it names — the claim
            # itself is stated beside the figure, in poster type, where a
            # visitor reads it without decoding the axes first.
            artist.set_text("CHEAPER  AND  NO WORSE")
            artist.set_position((x0 / 2, 4.1))
            artist.set_ha("center")
            artist.set_va("center")
            artist.set_rotation(90)
            artist.set_fontsize(S.FS_LABEL)
        elif "always-cheapest" in text:
            # Needs ~45mm of clear plane in a corner that has ~32mm; every
            # placement collides with the y-axis label, the legend or cls-B2.
            # The star is unambiguous once named beside the figure.
            artist.remove()
        elif text.startswith("WA-red·B1"):
            artist.set_position((9, -26))  # collides with red-B1

    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles,
              ["learned router", "learned (in-sample)", "hindsight oracle"],
              loc="lower right", frameon=False, handletextpad=0.4,
              borderpad=0.2, fontsize=S.FS_LABEL)
    ax.set_xlabel("cost, relative to always using the cheapest mode   "
                  "[$\\log_2$ ratio]", fontsize=S.FS_LABEL)
    ax.set_ylabel("success rate over always-cheapest  [pp]", fontsize=S.FS_LABEL)

    save(fig, "poster_dominance_plane")


if __name__ == "__main__":
    print("rendering poster-scale figures")
    dominance_plane()
