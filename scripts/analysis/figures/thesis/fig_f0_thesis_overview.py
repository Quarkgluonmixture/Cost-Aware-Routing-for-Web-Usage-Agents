#!/usr/bin/env python3
"""Thesis F0 — one-figure overview (no data dependency).

Guide §7.1 F0: a reader who sees only this figure must be able to answer
  (1) what the input is, (2) what stages the study has,
  (3) where the new thing is, (4) where the evaluation comes from.

This is the argument map, NOT a system performance figure. The bottom band is
the thesis spine itself, so the figure and the chapter chain cannot drift apart.

Numbers shown are deliberately few, and each carries its scope:
  +3.45-16.07pp   VWA 6-cell oracle-vs-best-single gain (router_objective_ordering.md)
  2.0-7.6pp       what one rerun of the SAME arm already buys (noise_floor_inventory.md §3)
  0/8             cells where learned triage Pareto-beats always-cheapest
                  (router_triage_learnability_with_wa.md:124)
  2-27%           base SR = best-SR fixed policy's own SR across the six VWA cells

Output: final_dissertation/figures/fig_f0_thesis_overview.{png,pdf}
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
OUT = ROOT / "final_dissertation/figures/fig_f0_thesis_overview"

# Okabe-Ito, colourblind-safe
C_TASK = "#000000"
C_CHEAP = "#0072B2"     # screenshot-free / cheap representations
C_EXPENSIVE = "#D55E00"  # screenshot-carrying / expensive
C_ROUTER = "#009E73"    # the decision this thesis is about
C_EVAL = "#5A5A5A"
C_POS = "#0072B2"
C_NEG = "#D55E00"
C_MECH = "#000000"

MODES = [
    ("DOM", C_CHEAP), ("P-text", C_CHEAP), ("P-prompt", C_CHEAP),
    ("P-SoM", C_CHEAP), ("SoM", C_EXPENSIVE), ("Vision", C_EXPENSIVE),
]


def _box(ax, x, y, w, h, label, sub=None, ec="#000000", fc="none", lw=1.4,
         fs=11, subfs=8.5, style="round,pad=0.02,rounding_size=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                ec=ec, fc=fc, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h * (0.62 if sub else 0.5), label, ha="center",
            va="center", fontsize=fs, fontweight="bold", color=ec, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h * 0.26, sub, ha="center", va="center",
                fontsize=subfs, color="#333333", zorder=3, linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, color="#000000", lw=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, lw=lw, color=color,
                                 linestyle=ls, zorder=4,
                                 shrinkA=0, shrinkB=0))


def build(ax):
    ax.set_xlim(0, 100)
    ax.set_ylim(-17, 64)
    ax.axis("off")

    # ---------- row 1: the loop, left to right ----------
    ax.text(1, 61.5, "A web task, one agent, one browser loop — only the page "
                     "representation changes",
            fontsize=10.5, color="#333333", style="italic")

    _box(ax, 1, 40, 16, 14.5, "Web task",
         "VisualWebArena\nclassifieds · reddit\n+ WebArena reddit", ec=C_TASK, subfs=8.2)
    _arrow(ax, 17, 47.2, 20.2, 47.2)

    # representation space
    ax.add_patch(FancyBboxPatch((20.5, 35.4), 34, 22.6,
                                boxstyle="round,pad=0.02,rounding_size=0.02",
                                ec="#888888", fc="#F7F7F7", lw=1.2, zorder=1))
    ax.text(37.5, 55.9, "Representation space  (6 modes)", ha="center",
            fontsize=10, fontweight="bold")
    for i, (name, col) in enumerate(MODES):
        r, c = divmod(i, 3)
        bx, by = 22.2 + c * 10.4, 47.0 - r * 7.6
        _box(ax, bx, by, 9.2, 5.6, name, ec=col, fc="white", lw=1.3, fs=9.5)
    ax.text(37.5, 37.2, "cheap: text only          expensive: + screenshot",
            ha="center", fontsize=8.2, color="#555555")

    _arrow(ax, 54.5, 47.2, 57.7, 47.2)
    _box(ax, 58, 40, 19, 14.5, "Same agent loop",
         "observe → decide → act\nidentical everywhere", ec=C_TASK, subfs=8.2)
    _arrow(ax, 77, 47.2, 80.2, 47.2)
    _box(ax, 80.5, 40, 18, 14.5, "Measured", "task success\ncost · latency",
         ec=C_EVAL, subfs=8.2)

    # ---------- the router: the object of study ----------
    _box(ax, 20.5, 25.0, 34, 7.6, "Router:  which mode for this task?",
         "the decision this thesis is about", ec=C_ROUTER, fc="#F2FBF8", lw=2.0,
         fs=10.5, subfs=8.5)
    _arrow(ax, 37.5, 32.6, 37.5, 35.2, color=C_ROUTER, lw=2.0)
    ax.text(56.2, 28.8, "serving-time features only —\nno peeking at the outcome",
            fontsize=8.4, color=C_ROUTER, va="center", linespacing=1.5)
    _arrow(ax, 55.6, 28.8, 54.7, 28.8, color=C_ROUTER, lw=1.3)

    # ---------- row 2: the spine ----------
    ax.plot([1, 99], [20.0, 20.0], color="#CCCCCC", lw=1.0)
    ax.text(1, 17.0, "What the thesis finds", fontsize=10.5, fontweight="bold")

    spine = [
        (C_POS, "1", "The headroom is real",
         "Oracle mode choice beats the best fixed mode by "
         "$\\bf{+3.45}$–$\\bf{16.07pp}$ (6 VWA cells).\n"
         "But one rerun of the SAME mode already buys 2.0–7.6pp — "
         "so the gap is real, not all of it is representation."),
        (C_NEG, "2", "The choice is not predictable",
         "With nested CV and two controls, $\\bf{0/8}$ cells learn a triage "
         "rule that Pareto-beats\n'always use the cheapest mode' — a policy "
         "that costs nothing to implement."),
        (C_MECH, "3", "And the reason is structural",
         "A which-mode label only exists once a task is solved, and base SR is "
         "$\\bf{2}$–$\\bf{27\\%}$.\n"
         "4 of 6 cells never reach a trainable label supply — "
         "resplitting cannot manufacture events."),
    ]
    y = 13.0
    for col, num, head, body in spine:
        ax.add_patch(plt.Circle((3.0, y), 1.3, color=col, zorder=3))
        ax.text(3.0, y, num, ha="center", va="center", color="white",
                fontsize=10, fontweight="bold", zorder=4)
        ax.text(6.0, y, head, fontsize=10.5, fontweight="bold", color=col,
                va="center")
        ax.text(6.0, y - 2.3, body, fontsize=8.6, color="#333333", va="top",
                linespacing=1.7)
        y -= 9.6

    ax.text(1, -15.4,
            "Scope: VisualWebArena (classifieds, reddit) × three backbones, plus "
            "WebArena-reddit × two. Oracle bounds are retrospective, not deployable.",
            fontsize=7.8, color="#666666")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    fig, ax = plt.subplots(figsize=(12.5, 9.4))
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
