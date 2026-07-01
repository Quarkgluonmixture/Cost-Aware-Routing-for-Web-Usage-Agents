#!/usr/bin/env python3
"""Paper F1 — 2x2 ablation-diamond concept schematic (no data dependency).

Visual form of aaai27_main.md Table 1 / §3.1: two textual knobs (text-payload
format x prompt family) span a 2x2 diamond of screenshot-free arms; full SoM is
the screenshot-on endpoint of the P-SoM corner; Vision anchors screenshot-only.

Output: results/phantom_paper/figures/fig_f1_diamond_schematic.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results/phantom_paper/figures/fig_f1_diamond_schematic"

# colorblind-safe (Okabe-Ito)
C_FREE = "#0072B2"      # screenshot-free arms
C_DOM = "#000000"       # DOM baseline (screenshot-free but not phantom)
C_IMG = "#D55E00"       # screenshot-carrying arms
C_REGION = "#56B4E9"    # phantom region fill

NODES = {
    # (x, y): axes = (text format, prompt family)
    "DOM": (0.0, 0.0),
    "P-text": (1.0, 0.0),
    "P-prompt": (0.0, 1.0),
    "P-SoM": (1.0, 1.0),
}
SUB = {
    "DOM": "AXTree text\nDOM-style prompt",
    "P-text": "[SOM_MARKS] text\nDOM-style prompt",
    "P-prompt": "AXTree text\nSoM-style prompt",
    "P-SoM": "[SOM_MARKS] text\nSoM-style prompt",
}
SOM_XY = (1.85, 1.0)
VISION_XY = (1.85, -0.05)


def main() -> int:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=300)

    # phantom region: the three screenshot-free non-DOM corners
    region = Polygon(
        [NODES["P-text"], NODES["P-SoM"], NODES["P-prompt"]],
        closed=True, facecolor=C_REGION, alpha=0.15, edgecolor="none", zorder=0,
    )
    ax.add_patch(region)
    ax.text(0.72, 0.7, "phantom\nrouting space", ha="center", va="center",
            fontsize=10, style="italic", color=C_FREE, zorder=1)

    # diamond edges (axis moves)
    for a, b in [("DOM", "P-text"), ("DOM", "P-prompt"),
                 ("P-text", "P-SoM"), ("P-prompt", "P-SoM")]:
        (x1, y1), (x2, y2) = NODES[a], NODES[b]
        ax.plot([x1, x2], [y1, y2], color="#888888", lw=1.2, zorder=2)

    # axis-move labels
    ax.text(0.5, -0.14, "flatten text payload →", ha="center", fontsize=8.5, color="#555555")
    ax.text(-0.33, 0.5, "swap prompt family →", ha="center", fontsize=8.5,
            color="#555555", rotation=90)

    # screenshot-free nodes
    for name, (x, y) in NODES.items():
        c = C_DOM if name == "DOM" else C_FREE
        ax.scatter([x], [y], s=1500, facecolor="white", edgecolor=c, lw=2.2, zorder=3)
        ax.annotate(name, (x, y), ha="center", va="center", fontsize=8,
                    fontweight="bold", color=c, zorder=4)
        dy = -0.24 if y == 0.0 else 0.24
        ax.annotate(SUB[name], (x, y + dy), ha="center", va="center",
                    fontsize=7.2, color="#333333", zorder=4)

    # SoM endpoint: P-SoM + marked screenshot
    ax.scatter(*SOM_XY, s=1500, facecolor=C_IMG, edgecolor=C_IMG, alpha=0.9, zorder=3)
    ax.annotate("SoM", SOM_XY, ha="center", va="center", fontsize=8,
                fontweight="bold", color="white", zorder=4)
    ax.annotate("+ marked screenshot\n(the bundle)", (SOM_XY[0], SOM_XY[1] + 0.24),
                ha="center", fontsize=7.2, color="#333333")
    arrow = FancyArrowPatch(NODES["P-SoM"], SOM_XY, arrowstyle="-|>",
                            mutation_scale=14, lw=1.4, linestyle="--",
                            color=C_IMG, shrinkA=24, shrinkB=24, zorder=2)
    ax.add_patch(arrow)
    ax.text(1.42, 1.09, "add screenshot", fontsize=7.5, color=C_IMG, ha="center")

    # Vision anchor
    ax.scatter(*VISION_XY, s=1500, facecolor="white", edgecolor=C_IMG,
               lw=2.2, linestyle=":", zorder=3)
    ax.annotate("Vision", VISION_XY, ha="center", va="center", fontsize=8,
                fontweight="bold", color=C_IMG, zorder=4)
    ax.annotate("raw screenshot only\n(no AXTree text)", (VISION_XY[0], VISION_XY[1] - 0.25),
                ha="center", fontsize=7.2, color="#333333")

    # legend line
    ax.text(0.5, 1.42,
            "screenshot-free boundary: all four left nodes receive no per-step page screenshot\n"
            "(task-supplied reference images preserved identically in every mode)",
            ha="center", fontsize=7.8, color="#555555")

    ax.set_xlim(-0.45, 2.35)
    ax.set_ylim(-0.42, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
        print(f"Wrote: {OUT}.{ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
