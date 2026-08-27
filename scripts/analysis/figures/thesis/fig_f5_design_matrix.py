#!/usr/bin/env python3
"""Thesis F5 — what was actually run: modes x cells, coverage at a glance.

⚠️ SCOPE OF WHAT THIS FIGURE VERIFIES. It reads success rates out of one
markdown table and cross-checks the VWA subset against the canonical SR JSON. It
does NOT read completeness flags, episode counts, artifact grades or task-set
SHAs, so it can only claim "this condition has a row", not "this condition ran
to completion on the full scored set". Completeness is enforced upstream by the
aggregators, which refuse a condition whose task-set hash does not match. Do not
let the caption promise more than the parser checks.

Guide §7.1 F4: worth a figure when the experiment count is large enough that a
reader cannot otherwise judge coverage. 48 conditions over 8 cells qualifies.

The terminology lock matters here and is enforced in the labels rather than left
to the caption: a **condition** is one (site, model, mode) launch unit; a
**cell** is one (site, model) statistical stratification unit. A filled square
is a condition; a column is a cell. Mixing the two is the single most common
error in this project's own prose, so the figure names both explicitly.

Output: final_dissertation/figures/fig_f5_design_matrix.{png,pdf}

Run via ``make thesis-figures``, not on its own: this script writes only the
working tree, and the copy LaTeX embeds is refreshed by that target.
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
import _style as S  # noqa: E402
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
                        va="center", fontsize=S.FS_VALUE, color="white",
                        fontweight="bold")

    # Separate the primary sweep from the external-validation cells. The two
    # tags name the benchmark and nothing else; which of them is the primary
    # sweep and which the external validation is stated in the caption.
    if WA_CELLS and all(c in cells for c in WA_CELLS):
        cut = len(VWA_CELLS) - 0.5
        ax.axvline(cut, color=S.C_INK, lw=1.2)
        ax.text(cut - 0.10, nrow - 0.28, "VisualWebArena", ha="right",
                fontsize=S.FS_LABEL, color=S.C_INK, fontweight="bold")
        ax.text(cut + 0.10, nrow - 0.28, "WebArena", ha="left",
                fontsize=S.FS_LABEL, color=S.C_INK, fontweight="bold")

    # Eight cell codes will not sit side by side on a 13cm text block: at this
    # width they ran into each other ("red-B2wa_red-B0"). Rotating them costs a
    # head-tilt but keeps every code readable, which is the cheaper of the two.
    ax.set_xticks(range(ncol), [f"{c}  $n$={n_of[c]}" for c in cells])
    ax.tick_params(axis="x", labelrotation=45)
    for lbl in ax.get_xticklabels():
        lbl.set_horizontalalignment("right")
    ax.set_yticks(range(nrow), MODES[::-1], fontsize=S.FS_LABEL)
    ax.set_xlim(-0.6, ncol - 0.4)
    ax.set_ylim(-0.6, nrow - 0.15)
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[plt.Line2D([], [], marker="s", ls="", ms=9,
                                  color=C_SIDE[i], label=lab)
                       for i, (lab, _m) in enumerate(SIDES)],
              loc="upper center", bbox_to_anchor=(0.5, -0.42), ncol=3,
              frameon=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    cells, n_of, sr, _cost = load()
    n_cond = len(sr)

    S.apply()
    fig, ax = plt.subplots(figsize=(S.PRINT_W_IN, 3.0))
    build(ax, cells, n_of, sr)
    # Nothing is written above the axes. The claim this grid supports (every
    # mode ran in every cell, so within-cell comparisons are paired), the
    # condition-versus-cell definition, and the provenance note all now live in
    # the LaTeX caption, where they are body text rather than 8pt grey pixels.

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   ({n_cond} conditions, {len(cells)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
