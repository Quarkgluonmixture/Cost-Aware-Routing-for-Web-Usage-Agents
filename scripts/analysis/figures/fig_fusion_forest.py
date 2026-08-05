#!/usr/bin/env python3
"""Fusion premium per cell: SoM against each single channel, with the rerun threshold.

Committed 2026-08-05, replacing a version whose script lived on one machine only.
Two things changed with it:

1. The legend carried ``measured rerun band (|Δ| ≤ 2.23)``. That is the *observed*
   range across two replicate draws, and ``fusion_premium.json`` says in its own
   ``reading`` field that it "must never be quoted as a threshold on its own".
   Shading it is exactly that. The figure now shades ``one_sided_95_*`` (3.8--4.2pp),
   the level a mean difference has to reach before a single rerun would be unlikely
   to produce it, which is the threshold Section 4 adopts.
2. Numbers moved out of the plot and into the caption.

The verdict is unchanged and slightly safer under the wider band: against the
workload-matched channel, 0 of 8 cells clear it.

Usage:
    python3 scripts/analysis/figures/fig_fusion_forest.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "docs" / "analysis" / "cross_sites" / "fusion_premium.json"

BLUE = "#2A78D6"
ORANGE = "#EB6834"
GREY = "#52514E"
BAND = "#E5E4E0"
FIGSIZE = (3.03, 2.28)

PRETTY = {"cls_B0": "cls·B0", "cls_B1": "cls·B1", "cls_B2": "cls·B2",
          "red_B0": "red·B0", "red_B1": "red·B1", "red_B2": "red·B2",
          "wa_red_B0": "WA·B0", "wa_red_B1": "WA·B1"}
ORDER = ["cls_B0", "cls_B1", "cls_B2", "red_B0", "red_B1", "red_B2",
         "wa_red_B1", "wa_red_B0"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path.home() / "overleaf-aaai27" / "figures")
    args = ap.parse_args()

    d = json.loads(SRC.read_text())
    cells, band = d["cells"], d["floor_band"]
    for key in ("one_sided_95_min_pp", "one_sided_95_max_pp"):
        if key not in band:
            raise SystemExit(f"{SRC.name}: floor_band lost '{key}'; refusing to guess a threshold")
    thr_lo, thr_hi = band["one_sided_95_min_pp"], band["one_sided_95_max_pp"]
    missing = [c for c in ORDER if c not in cells]
    if missing:
        raise SystemExit(f"{SRC.name} is missing {missing}; the plot would silently drop rows")

    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Shade to the WIDER edge: a cell has to clear the most permissive reading of the
    # threshold before we call its premium resolved.
    ax.axvspan(-thr_hi, thr_hi, color=BAND, zorder=0)
    ax.axvline(0, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)

    for i, cid in enumerate(ORDER):
        for comparator, colour, off in (("vision", BLUE, +0.17), ("dom", ORANGE, -0.17)):
            b = cells[cid][comparator]
            lo, hi = b["ci"]
            y = i + off
            ax.plot([lo, hi], [y, y], color=colour, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot([b["est_pp"]], [y], marker="o", ms=2.6, color=colour, zorder=3)

    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([PRETTY[c] for c in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("fusion premium (pp), paired bootstrap 95% CI", fontsize=6)
    ax.tick_params(length=2, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)

    handles = [plt.Line2D([], [], color=BLUE, lw=1.0, marker="o", ms=2.6, label="SoM − Vision"),
               plt.Line2D([], [], color=ORANGE, lw=1.0, marker="o", ms=2.6, label="SoM − DOM"),
               plt.Rectangle((0, 0), 1, 1, color=BAND, label="rerun threshold")]
    # Outside the axes: the WA rows run to -12.5pp, so any in-axes corner sits on data.
    # An inside legend here obscured two error bars in the version this replaces.
    ax.legend(handles=handles, fontsize=5.5, frameon=False, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2, columnspacing=1.1, borderaxespad=0.15)

    fig.tight_layout(pad=0.25)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "fig_fusion_forest.pdf"
    fig.savefig(dest)
    print(f"wrote {dest}")
    print(f"shaded threshold for the caption: ±{thr_lo:.1f}–{thr_hi:.1f}pp "
          f"(one-sided 95% under the exchangeability null)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
