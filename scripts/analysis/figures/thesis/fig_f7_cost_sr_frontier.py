#!/usr/bin/env python3
"""Thesis F7 — cost against success, one panel per cell, dominated modes shown.

Two rules govern this figure and both exist because of a specific way it could
mislead.

1. Dominated modes are drawn, not hidden. A frontier plot that shows only the
   non-dominated set makes every mode look reasonable; the interesting fact is
   that several modes cost more AND succeed less than an alternative.

2. Panels are never compared across the API/local boundary. B0's cost axis is
   billed API dollars; B1/B2's is an electricity equivalent from device
   telemetry. Each panel therefore carries its own x-axis and its own unit
   label, and no shared axis invites the ratio.

Output: final_dissertation/figures/fig_f7_cost_sr_frontier.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ordering_parse import (  # noqa: E402
    C_SIDE, MODES, SIDE_OF, SIDES, load)

OUT = (Path(__file__).resolve().parents[4]
       / "final_dissertation/figures/fig_f7_cost_sr_frontier")


def pareto(points):
    """Indices of non-dominated points: cheaper is better, higher SR is better."""
    keep = []
    for i, (ci, si) in enumerate(points):
        if not any((cj <= ci and sj >= si) and (cj < ci or sj > si)
                   for j, (cj, sj) in enumerate(points) if j != i):
            keep.append(i)
    return keep


def panel(ax, cell, n, sr, cost, unit):
    pts = [(cost[(cell, m)], sr[(cell, m)]) for m in MODES]
    front = sorted(pareto(pts), key=lambda i: pts[i][0])
    ax.plot([pts[i][0] for i in front], [pts[i][1] for i in front],
            color="#666666", lw=1.1, ls="-", zorder=1, alpha=0.7)
    for i, m in enumerate(MODES):
        c, s = pts[i]
        on = i in front
        ax.scatter([c], [s], s=74 if on else 46, color=C_SIDE[SIDE_OF[m]],
                   edgecolor="#333333" if on else "none",
                   lw=1.1, zorder=3, alpha=1.0 if on else 0.55)
        ax.annotate(m, (c, s), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=7.4,
                    color="#222222" if on else "#888888")
    ax.set_title(f"{cell}   ($n$={n}, {len(front)} non-dominated)",
                 fontsize=9.4, loc="left", pad=6)
    ax.set_xlabel(unit, fontsize=8.0)
    ax.grid(color="#F2F2F2", lw=0.8)
    ax.set_axisbelow(True)
    for s_ in ("top", "right"):
        ax.spines[s_].set_visible(False)
    lo = min(p[0] for p in pts)
    hi = max(p[0] for p in pts)
    ax.set_xlim(lo - (hi - lo) * 0.18, hi + (hi - lo) * 0.18)
    ymax = max(p[1] for p in pts)
    ax.set_ylim(-ymax * 0.14, ymax * 1.30)
    return len(front)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    cells, n_of, sr, cost = load()

    fig, axes = plt.subplots(2, 4, figsize=(14.6, 7.2))
    fronts = []
    for ax, cell in zip(axes.ravel(), cells):
        unit = ("billed API \\$ / episode" if "B0" in cell
                else "electricity-equivalent \\$ / episode")
        fronts.append(panel(ax, cell, n_of[cell], sr, cost, unit))
    for ax in axes.ravel()[len(cells):]:
        ax.axis("off")
    for ax in axes[:, 0]:
        ax.set_ylabel("task success rate (%)", fontsize=8.6)

    fig.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=8,
                                   color=C_SIDE[i], label=lab)
                        for i, (lab, _m) in enumerate(SIDES)],
               loc="lower center", ncol=3, frameon=False, fontsize=8.8,
               bbox_to_anchor=(0.5, -0.005))

    fig.suptitle("Cost and success do not move together: every cell has "
                 "several non-dominated modes, and several dominated ones",
                 fontsize=12.0, fontweight="bold", x=0.008, ha="left", y=1.010)
    fig.text(0.008, 0.965,
             f"Filled outlines are non-dominated ({min(fronts)}-{max(fronts)} "
             "per cell); faded points cost more AND succeed less than some "
             "alternative, and are shown rather than hidden. Cost is measured "
             "per episode, not\nassumed from a token schedule — which is why a "
             "weaker mode can be the more expensive one (it spends the step "
             "budget failing). Panels use separate axes on purpose: B0 is "
             "billed API dollars,\nB1/B2 an electricity equivalent from device "
             "telemetry, and the two are never divided by one another.",
             fontsize=8.4, color="#444444", linespacing=1.5, va="top")
    fig.text(0.008, -0.030,
             "Source: docs/analysis/cross_sites/router_objective_ordering.md "
             "(cost = total_billed_cost_usd, the canonical estimand), "
             "cross-checked against sr_per_mode.json.",
             fontsize=7.0, color="#888888")
    fig.tight_layout(rect=(0, 0.03, 1, 0.90))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   (non-dominated per cell: {fronts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
