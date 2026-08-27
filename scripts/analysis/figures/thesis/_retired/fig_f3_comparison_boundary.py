#!/usr/bin/env python3
"""Thesis F3 — the causal comparison boundary (no data dependency).

CHAPTER_CHAIN.md:235-237 fixes the spec, and explicitly rules out the obvious
alternative: "Not a decorative architecture diagram." The chain it must show is

    same task + same agent -> router/selected mode -> mode-specific observation
    construction -> same action loop -> benchmark outcome + measured cost

So this figure answers ONE question: why is a difference measured downstream
attributable to the representation rather than to something else? It does that
by drawing everything held constant inside one boundary and leaving exactly one
stage marked as varied. A module-dependency diagram of `p79/` would answer a
different question ("how is the code organised") and is deliberately not drawn.

Output: final_dissertation/figures/fig_f3_comparison_boundary.{png,pdf}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "final_dissertation/figures/fig_f3_comparison_boundary"

C_FIXED = "#5A5A5A"      # held constant
C_VAR = "#D55E00"        # the one varied stage
C_OUT = "#0072B2"        # measured outputs
C_BOUND = "#000000"

STAGES = [
    ("Task config", "same task set\n(224 cls · 203 red)"),
    ("Browser state", "same reset policy\nsame start URL"),
    ("Observation\nconstruction", None),          # the variable
    ("Prompt shell", "same template\nsame action schema"),
    ("Backbone", "same model\nsame decoding"),
]

# Grouped by what is sent (TERMS §1.1); the text-side four all send no image.
SIDES = [("DOM · P-text · P-prompt · P-SoM", "text side — no image", "#0072B2"),
         ("SoM", "combined — text + image", "#009E73"),
         ("Vision", "visual — image only", "#D55E00")]


def _box(ax, x, y, w, h, label, sub=None, ec="#000000", fc="none", lw=1.4,
         fs=10, subfs=7.8, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                ec=ec, fc=fc, lw=lw, zorder=z))
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), label, ha="center",
            va="center", fontsize=fs, fontweight="bold", color=ec, zorder=z + 1,
            linespacing=1.25)
    if sub:
        ax.text(x + w / 2, y + h * 0.24, sub, ha="center", va="center",
                fontsize=subfs, color="#444444", zorder=z + 1, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, color="#000000", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, lw=lw, color=color,
                                 linestyle=ls, zorder=5, shrinkA=0, shrinkB=0))


def build(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(-4, 50)
    ax.axis("off")

    # ---- the boundary --------------------------------------------------
    ax.add_patch(FancyBboxPatch((1.5, 20.5), 97, 24,
                                boxstyle="round,pad=0.02,rounding_size=0.03",
                                ec=C_BOUND, fc="#FAFAFA", lw=2.2, zorder=1,
                                linestyle=(0, (6, 3))))
    ax.text(3.2, 41.6, "Comparison boundary — everything inside is held identical "
                       "across all conditions",
            fontsize=10, fontweight="bold", color=C_BOUND, va="center")

    # ---- the pipeline --------------------------------------------------
    n = len(STAGES)
    x0, w, gap = 3.2, 16.4, 2.9
    for i, (label, sub) in enumerate(STAGES):
        x = x0 + i * (w + gap)
        varied = sub is None
        _box(ax, x, 24.5, w, 12.4, label,
             "★ the only\nvaried stage" if varied else sub,
             ec=C_VAR if varied else C_FIXED,
             fc="#FFF4EC" if varied else "white",
             lw=2.2 if varied else 1.3,
             fs=10 if varied else 9.5,
             subfs=8.0 if varied else 7.6)
        if i < n - 1:
            _arrow(ax, x + w, 30.7, x + w + gap, 30.7, color="#666666", lw=1.4)

    # ---- what the varied stage ranges over ------------------------------
    vx = x0 + 2 * (w + gap)
    _arrow(ax, vx + w / 2, 24.3, vx + w / 2, 20.6, color=C_VAR, lw=1.8)
    ax.add_patch(FancyBboxPatch((vx - 15.0, 8.6), w + 30, 11.6,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                ec=C_VAR, fc="#FFF4EC", lw=1.6, zorder=3))
    # one row per side: the text-side list is too wide to sit beside the others
    for i, (names, side, col) in enumerate(SIDES):
        yy = 17.6 - i * 3.3
        ax.text(vx + w / 2 - 1.5, yy, names, ha="right", va="center",
                fontsize=8.6, fontweight="bold", color=col, zorder=4)
        ax.text(vx + w / 2 + 1.5, yy, side, ha="left", va="center",
                fontsize=7.4, color=col, zorder=4)
    ax.text(vx + w / 2, 6.2, "six observation modes — the same page, encoded "
            "six ways, grouped by what is actually sent",
            ha="center", va="center", fontsize=8.0, color="#555555")

    # ---- outputs ---------------------------------------------------------
    # The arrows leave the BOUNDARY, not the mode box: the pipeline produces the
    # outcome, while the mode box is an annotation of what the varied stage ranges
    # over. Routing them at x=15/x=85 keeps them clear of the mode box (27-73).
    _arrow(ax, 15, 20.3, 15, 5.0, color=C_OUT, lw=1.6)
    _arrow(ax, 85, 20.3, 85, 5.0, color=C_OUT, lw=1.6)
    _box(ax, 1, -2.2, 32, 7.0, "Benchmark outcome",
         "same evaluator, same scored task universe", ec=C_OUT, fc="white",
         fs=10, subfs=8.0)
    _box(ax, 67, -2.2, 32, 7.0, "Measured cost",
         "same accounting boundary (total billed · latency)", ec=C_OUT,
         fc="white", fs=10, subfs=8.0)

    # the claim the figure licenses
    ax.text(50, 46.9,
            "Any downstream difference is attributable to the representation, "
            "because nothing else was allowed to move.",
            ha="center", fontsize=9.6, color="#333333", style="italic")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(13.0, 6.6))
    build(ax)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
