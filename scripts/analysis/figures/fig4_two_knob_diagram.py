#!/usr/bin/env python3
"""Schematic two-knob mechanism: representation controls exploration; prompt controls confidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "docs/analysis/figures/fig4_two_knob_diagram.png"


def cell(ax, xy, w, h, title, subtitle, face, edge="#333333"):
    rect = Rectangle(xy, w, h, facecolor=face, edgecolor=edge, lw=1.4)
    ax.add_patch(rect)
    x, y = xy
    ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.36, subtitle, ha="center", va="center", fontsize=9.5, color="#333333")


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    ax.text(5.0, 6.58, "Two-Knob Ablation", ha="center", fontsize=16, fontweight="bold")
    ax.text(5.0, 6.22, "Text representation shapes exploration; prompt wording tunes commitment confidence", ha="center", fontsize=10.5, color="#444444")

    ax.text(3.25, 5.62, "DOM prompt", ha="center", fontsize=12, fontweight="bold")
    ax.text(6.75, 5.62, "SoM prompt", ha="center", fontsize=12, fontweight="bold")
    ax.text(0.72, 4.45, "AXTree obs", ha="center", rotation=90, fontsize=12, fontweight="bold")
    ax.text(0.72, 2.05, "[SOM_MARKS] obs", ha="center", rotation=90, fontsize=12, fontweight="bold")

    cell(
        ax,
        (1.65, 3.65),
        3.1,
        1.55,
        "DOM",
        "search-loop 22.7%\nFP gap 6.25 pp",
        "#dce8f5",
    )
    cell(
        ax,
        (5.25, 3.65),
        3.1,
        1.55,
        "Theoretical cell",
        "AXTree with SoM prompt\nnot run in Phase 2.1",
        "#eeeeee",
        edge="#888888",
    )
    cell(
        ax,
        (1.65, 1.35),
        3.1,
        1.55,
        "Phantom-DOM",
        "search-loop 10.8%\nFP gap 6.25 pp",
        "#f8dddd",
    )
    cell(
        ax,
        (5.25, 1.35),
        3.1,
        1.55,
        "Phantom-SoM",
        "search-loop 10.8%\nFP gap 2.08 pp",
        "#eadff0",
    )

    ax.add_patch(
        FancyArrowPatch(
            (4.95, 4.42),
            (4.95, 2.15),
            arrowstyle="<->",
            mutation_scale=18,
            lw=2.0,
            color="#222222",
        )
    )
    ax.text(
        4.95,
        3.23,
        "representation knob\nchanges exploration",
        ha="center",
        va="center",
        fontsize=10.5,
        bbox=dict(facecolor="white", edgecolor="none", pad=2.5),
    )

    ax.add_patch(
        FancyArrowPatch(
            (3.15, 1.08),
            (6.85, 1.08),
            arrowstyle="<->",
            mutation_scale=18,
            lw=2.0,
            color="#222222",
        )
    )
    ax.text(
        5.0,
        0.68,
        "prompt knob changes terminal calibration",
        ha="center",
        va="center",
        fontsize=10.5,
    )

    ax.text(
        5.0,
        0.18,
        "All numeric labels use the verified reddit ablation subset (N=48; raw-to-adjusted FP gap where shown).",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.savefig(OUT, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
