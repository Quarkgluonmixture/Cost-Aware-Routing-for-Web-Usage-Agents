#!/usr/bin/env python3
"""Thesis F6 — success rate per mode and cell; no column wins everywhere.

The design decision that makes this figure readable is row-wise normalisation.
Absolute success spans 0.45%-35.58% across cells — a factor of ~80 — so an
unnormalised heatmap shows only which backbone is stronger, which is not the
question. Normalising within each cell shows the *ordering*, which is the
precondition routing depends on. Absolute values are printed in the cells so
nothing is hidden by the transform.

Output: final_dissertation/figures/fig_f6_sr_heatmap.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ordering_parse import MODES, VWA_CELLS, load  # noqa: E402

OUT = (Path(__file__).resolve().parents[4]
       / "final_dissertation/figures/fig_f6_sr_heatmap")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    cells, n_of, sr, _cost = load()

    M = np.array([[sr[(c, m)] for c in cells] for m in MODES])
    # Row-wise here means per cell, i.e. per COLUMN of M.
    denom = M.max(axis=0)
    if (denom <= 0).any():
        raise SystemExit("a cell has zero success in every mode — refusing to "
                         "normalise")
    Z = M / denom

    fig, ax = plt.subplots(figsize=(10.2, 4.6))
    ax.imshow(Z, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")

    winners = []
    for j, cell in enumerate(cells):
        best = max(range(len(MODES)), key=lambda i: M[i, j])
        winners.append(MODES[best])
        for i, mode in enumerate(MODES):
            is_best = i == best
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=8.2, fontweight="bold" if is_best else "normal",
                    color="white" if Z[i, j] > 0.62 else "#333333")
            if is_best:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, edgecolor="#D55E00",
                                           lw=2.0, zorder=3))

    if all(c in cells for c in VWA_CELLS):
        ax.axvline(len(VWA_CELLS) - 0.5, color="#333333", lw=1.4)

    ax.set_xticks(range(len(cells)),
                  [f"{c}\n$n$={n_of[c]}" for c in cells], fontsize=9.0)
    ax.set_yticks(range(len(MODES)), MODES, fontsize=9.4)
    ax.tick_params(length=0)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)

    n_distinct = len(set(winners))
    ax.set_title("No observation mode wins everywhere — which is the only "
                 "reason selection could be worth anything",
                 fontsize=11.4, fontweight="bold", loc="left", pad=42)
    ax.text(0.0, 1.018,
            f"Task success rate (%), printed absolutely and shaded relative to "
            f"the best mode within each cell (absolute success spans a factor "
            f"of ~80 across cells, so an unnormalised map would\nshow only which "
            f"backbone is stronger). Orange outline = best mode in that cell: "
            f"{n_distinct} different modes win across {len(cells)} cells. The "
            "rightmost two columns are the external-validation benchmark.",
            transform=ax.transAxes, fontsize=8.4, color="#444444",
            linespacing=1.5, va="bottom")
    fig.text(0.012, 0.005,
             "Source: docs/analysis/cross_sites/router_objective_ordering.md, "
             "cross-checked against sr_per_mode.json for the six VWA cells; the "
             "two WA cells are single-source. Denominator is the scored set "
             "(Table 3.2). Task-set identity within a cell is enforced by "
             "SHA-256 in the upstream aggregators, not re-verified here.",
             fontsize=7.0, color="#888888")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({n_distinct} distinct winners over "
          f"{len(cells)} cells: {winners})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
