#!/usr/bin/env python3
"""Thesis F5 — what was actually run: modes x cells, coverage at a glance.

Guide §7.1 F4: worth a figure when the experiment count is large enough that a
reader cannot otherwise judge coverage. 48 conditions over 8 cells qualifies.

The terminology lock matters here and is enforced in the labels rather than left
to the caption: a **condition** is one (site, model, mode) launch unit; a
**cell** is one (site, model) statistical stratification unit. A filled square
is a condition; a column is a cell. Mixing the two is the single most common
error in this project's own prose, so the figure names both explicitly.

Output: final_dissertation/figures/fig_f5_design_matrix.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ordering_parse import (  # noqa: E402
    C_SIDE, MODES, SIDE_OF, SIDES, VWA_CELLS, WA_CELLS, load)

OUT = (Path(__file__).resolve().parents[4]
       / "final_dissertation/figures/fig_f5_design_matrix")


def build(ax, cells, n_of, sr):
    ncol, nrow = len(cells), len(MODES)
    for j, cell in enumerate(cells):
        for i, mode in enumerate(MODES):
            y = nrow - 1 - i
            filled = (cell, mode) in sr
            col = C_SIDE[SIDE_OF[mode]] if filled else "#FFFFFF"
            ax.add_patch(Rectangle((j - 0.42, y - 0.40), 0.84, 0.80,
                                   facecolor=col, edgecolor="#BBBBBB",
                                   lw=0.8, alpha=0.85 if filled else 1.0))
            if filled:
                ax.text(j, y, f"{sr[(cell, mode)]:.1f}", ha="center",
                        va="center", fontsize=8.0, color="white",
                        fontweight="bold")

    # Separate the primary sweep from the external-validation cells.
    if WA_CELLS and all(c in cells for c in WA_CELLS):
        cut = len(VWA_CELLS) - 0.5
        ax.axvline(cut, color="#333333", lw=1.4)
        ax.text(cut - 0.08, nrow - 0.30, "primary sweep  ·  VisualWebArena",
                ha="right", fontsize=8.6, color="#333333", fontweight="bold")
        ax.text(cut + 0.08, nrow - 0.30, "external validation  ·  WebArena",
                ha="left", fontsize=8.6, color="#333333", fontweight="bold")

    ax.set_xticks(range(ncol),
                  [f"{c}\n$n$={n_of[c]}" for c in cells], fontsize=9.0)
    ax.set_yticks(range(nrow), MODES[::-1], fontsize=9.4)
    ax.set_xlim(-0.6, ncol - 0.4)
    ax.set_ylim(-0.6, nrow - 0.15)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=9,
                                  color=C_SIDE[i], label=lab)
                       for i, (lab, _m) in enumerate(SIDES)],
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3,
              frameon=False, fontsize=8.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    cells, n_of, sr, _cost = load()
    n_cond = len(sr)

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    build(ax, cells, n_of, sr)
    ax.set_title("Every mode was run in every cell — the grid is complete, "
                 "so every comparison is paired",
                 fontsize=11.4, fontweight="bold", loc="left", pad=40)
    ax.text(0.0, 1.015,
            f"{n_cond} conditions over {len(cells)} cells. A **condition** is "
            "one (site, model, mode) launch unit — one square. A **cell** is "
            "one (site, model) stratification unit — one column.\nSquares carry "
            "task success rate (%); colour marks what is actually sent to the "
            "model. Within a column all six modes run the identical scored "
            "task-ID set.",
            transform=ax.transAxes, fontsize=8.4, color="#444444",
            linespacing=1.5, va="bottom")
    fig.text(0.012, 0.005,
             "Source: docs/analysis/cross_sites/router_objective_ordering.md, "
             "cross-checked against sr_per_mode.json (script refuses to plot on "
             "disagreement). VWA shopping conditions exist but are excluded "
             "from analysis (they predate pipeline corrections) and are not "
             "shown.",
             fontsize=7.0, color="#888888")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({n_cond} conditions, {len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
