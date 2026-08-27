#!/usr/bin/env python3
"""Thesis F6 — every mode's success rate, one row per cell.

This replaces two earlier attempts at the same question.

The first was a 6x8 heatmap carrying 48 printed numbers and a within-column
normalisation that had to be explained before it could be read. The second was
the conference figure `fig_sr_by_class.pdf`, which has the right shape but two
problems in this document: its palette puts the image-free modes in blue and
Vision in orange, the reverse of every other figure here, so a reader has to
re-learn the key halfway through Chapter 4; and its generating script does not
exist in this repository, so the numbers in it cannot be re-derived or checked.

The chart form is kept because it was the right one: one row per cell, one dot
per mode, the best mode in that row ringed. What changes is that the colours now
come from the shared palette and the values come from `_ordering_parse.load()`,
which cross-checks all 36 VisualWebArena values against `sr_per_mode.json` and
refuses to plot on any disagreement.

Reading direction is top to bottom, cells grouped by site, so a reader compares
backbones within a site by moving one row and compares sites by moving one
block.

Output: final_dissertation/figures/fig_f6_sr_by_class.{png,pdf}

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
from _ordering_parse import C_SIDE, MODES, SIDE_OF, SIDES, load  # noqa: E402

OUT = (Path(__file__).resolve().parents[4]
       / "final_dissertation/figures/fig_f6_sr_by_class")

# Rows are grouped by site rather than following the parser's declaration order,
# so the three backbones of a site sit together and the two benchmarks form
# separate blocks. The separator between blocks is drawn, not labelled.
ROW_ORDER = ["cls·B0", "cls·B1", "cls·B2",
             "red·B0", "red·B1", "red·B2",
             "wa_red·B0", "wa_red·B1"]
BLOCKS = (3, 6)   # a rule is drawn above these row indices


def build(ax, cells, sr):
    rows = [c for c in ROW_ORDER if c in cells]
    y = list(range(len(rows)))[::-1]

    for yi, cell in zip(y, rows):
        vals = {m: sr[(cell, m)] for m in MODES}
        top = max(vals.values())
        # Ties are real and are stated in the text: on cls-B2, SoM and Vision
        # both reach 2.23%. Ringing only one of them would break a documented
        # tie silently, so every mode at the maximum gets a ring.
        best = [m for m in MODES if abs(vals[m] - top) < 1e-9]

        # Two modes on exactly the same value would otherwise plot as one dot,
        # hiding a mode. Identical values are fanned out vertically by a hair,
        # which is below the row spacing and so cannot be misread as a value.
        seen: dict[float, int] = {}
        for mode in MODES:
            v = vals[mode]
            k = seen.get(v, 0)
            seen[v] = k + 1
            dy = 0.0 if k == 0 else (0.11 if k % 2 else -0.11) * ((k + 1) // 2)
            ax.scatter([v], [yi + dy], s=34, color=C_SIDE[SIDE_OF[mode]],
                       zorder=3, edgecolor="white", lw=0.5)
            if mode in best:
                # The ring is an annotation, not a category, so it is drawn in
                # ink rather than introducing a fourth colour.
                ax.scatter([v], [yi + dy], s=132, facecolor="none",
                           edgecolor=S.C_INK, lw=1.3, zorder=5)

    for cut in BLOCKS:
        if cut < len(rows):
            ax.axhline(len(rows) - cut - 0.5, color="#DDDDDD", lw=0.9, zorder=0)

    ax.set_yticks(y, rows)
    ax.set_xlabel("task success rate (%)")
    ax.set_xlim(-1.2, max(sr[(c, m)] for c in rows for m in MODES) + 2.5)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)

    handles = [plt.Line2D([], [], marker="o", ls="", ms=6, color=C_SIDE[i],
                          label=lab) for i, (lab, _m) in enumerate(SIDES)]
    handles.append(plt.Line2D([], [], marker="o", ls="", ms=9,
                              markerfacecolor="none", markeredgecolor=S.C_INK,
                              markeredgewidth=1.3, label="best in cell"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=4, frameon=False, handletextpad=0.4, columnspacing=1.4)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    cells, _n_of, sr, _cost = load()
    missing = [c for c in ROW_ORDER if c not in cells]
    if missing:
        raise SystemExit(f"cells absent from the source table: {missing}")

    S.apply()
    fig, ax = plt.subplots(figsize=(S.PRINT_W_IN, 3.4))
    rows = build(ax, cells, sr)

    winners = {}
    for cell in rows:
        vals = {m: sr[(cell, m)] for m in MODES}
        top = max(vals.values())
        winners[cell] = [m for m in MODES if abs(vals[m] - top) < 1e-9]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    n_distinct = len({m for ms in winners.values() for m in ms})
    print(f"wrote {a.out}.png / .pdf   ({len(rows)} cells, {n_distinct} distinct "
          f"winners: {winners})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
