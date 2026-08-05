#!/usr/bin/env python3
"""The zero-token partition: what the screenshot is worth on regex-flagged tasks.

Committed 2026-08-05. The Overleaf copy of this figure was produced by a script that
lived only on one machine, so its legend could not be edited by anyone else --- which
is how ``(71/63 tasks)`` stayed baked into the artwork after the decision to keep
counts in captions rather than inside plots. Regenerating it here puts the figure back
under version control; the counts now go in the caption.

Style is matched to the sibling forest plots (same size, palette and type scale) so
the three data figures still read as one set.

Usage:
    python3 scripts/analysis/figures/fig_partition_forest.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "docs" / "analysis" / "cross_sites" / "visual_intent_routing.json"

# Sampled from the figure this replaces, so the set stays visually consistent.
BLUE = "#2A78D6"
ORANGE = "#EB6834"
GREY = "#52514E"
FIGSIZE = (3.03, 1.85)

PRETTY = {"cls_B0": "cls·B0", "cls_B1": "cls·B1", "cls_B2": "cls·B2",
          "red_B0": "red·B0", "red_B1": "red·B1", "red_B2": "red·B2"}
ORDER = ["cls_B0", "cls_B1", "cls_B2", "red_B0", "red_B1", "red_B2"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path.home() / "overleaf-aaai27" / "figures")
    ap.add_argument("--contrast", default="vision", choices=["vision", "som"])
    args = ap.parse_args()

    sites = json.loads(SRC.read_text())["sites"]
    cells = {cid: c for s in sites.values() for cid, c in s["cells"].items()}
    missing = [c for c in ORDER if c not in cells]
    if missing:
        raise SystemExit(f"{SRC.name} is missing {missing}; the plot would silently drop rows")

    plt.rcParams.update({"font.size": 6, "axes.linewidth": 0.6,
                         "xtick.labelsize": 6, "ytick.labelsize": 6})
    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Rows run top-to-bottom in ORDER, so invert the numeric axis rather than
    # reversing the list -- keeps the data order and the visual order the same object.
    for i, cid in enumerate(ORDER):
        block = cells[cid][args.contrast]
        for stratum, colour, off in (("flagged", BLUE, +0.17), ("rest", ORANGE, -0.17)):
            b = block[stratum]
            lo, hi = b["ci"]
            y = i + off
            ax.plot([lo, hi], [y, y], color=colour, lw=1.0, solid_capstyle="butt", zorder=2)
            ax.plot([b["est_pp"]], [y], marker="o", ms=2.6, color=colour, zorder=3)

    ax.axvline(0, color=GREY, lw=0.6, ls=(0, (3, 2)), zorder=1)
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels([PRETTY[c] for c in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("Δ success (Vision − DOM, pp), 95% CI" if args.contrast == "vision"
                  else "Δ success (SoM − DOM, pp), 95% CI", fontsize=6)
    ax.tick_params(length=2, width=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color="#E5E4E0", lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    # No counts in the legend: stratum sizes vary by site and belong to the caption.
    # Placed outside the axes to match the sibling forest plot, where an in-axes
    # legend sat on top of two error bars.
    handles = [plt.Line2D([], [], color=BLUE, lw=1.0, marker="o", ms=2.6, label="flagged"),
               plt.Line2D([], [], color=ORANGE, lw=1.0, marker="o", ms=2.6, label="the rest")]
    ax.legend(handles=handles, fontsize=5.5, frameon=False, ncol=2,
              loc="lower center", bbox_to_anchor=(0.5, 1.0),
              handlelength=1.2, columnspacing=1.4, borderaxespad=0.15)

    fig.tight_layout(pad=0.25)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / "fig_partition_forest.pdf"
    fig.savefig(dest)
    print(f"wrote {dest}")
    print("flagged counts for the caption: " +
          ", ".join(f"{s} {v['n_flagged']}" for s, v in sites.items() if "n_flagged" in v))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
