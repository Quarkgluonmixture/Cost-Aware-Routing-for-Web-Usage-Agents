#!/usr/bin/env python3
"""Thesis F16 — is the negative result a mechanism, or just too few rows?

The figure exists to settle one attack, and it is built so that it can settle it
in EITHER direction rather than only the flattering one. Three panels, three
different things the "you just had too little data" objection could mean:

  A  more of the same data -> does out-of-fold AUROC keep climbing, or saturate?
  B  is there signal at all -> near-unregularised train AUROC against the same
     model fitted to permuted labels (the memorisation baseline)
  C  the bound that decides it -> more data moves the learned predictor toward
     the ORACLE, and Ch6 measured what the oracle itself achieves. If the oracle
     does not enter the winning region, no predictor reaching for it can either.

Panel C is the load-bearing one and it is the reason A and B are allowed to come
back partly against us: they do. The curves are still rising and the signal is
real, so the objection has force against the AUROC claim. It has no purchase on
the Pareto claim, because that boundary is above the oracle.

Output: final_dissertation/figures/fig_f16_undersampling_control.{png,pdf}

Run via ``make thesis-figures``, not on its own: this script writes only the
working tree, and the copy LaTeX embeds is refreshed by that target.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _style as S  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "docs/analysis/cross_sites/router_undersampling_control.json"
SRC_E2E = ROOT / "docs/analysis/cross_sites/router_triage_learnability_with_wa.md"
OUT = ROOT / "final_dissertation/figures/fig_f16_undersampling_control"

C_DEP = "#0072B2"     # deployment feature set (18)
C_ANN = "#999999"     # annotated set (20) — includes a column no deployment has
C_HOT = "#D55E00"
C_TH = "#333333"


def load():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    if not d.get("cells"):
        raise SystemExit(f"{SRC}: no cells")
    return d


def _endpoint_labels(ax, ends):
    """Stack the right-edge cell labels so they cannot print on top of each other.

    The six deployment curves converge into an AUROC band about 0.06 wide, so a
    fixed (5, -2) point offset put three of them --- classifieds/B1, reddit/B2
    and classifieds/B0 --- in the same few points of vertical space, and the
    middle label was struck through by its neighbours. Labels are pushed apart
    to a minimum gap and joined back to their own endpoint by a leader line, so
    de-overlapping never costs the reader the label-to-curve correspondence.
    """
    if not ends:
        return
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    xmax = max(x for _, x, _ in ends)
    # A gutter on the right; without it the labels sit outside the axes and are
    # clipped by the figure's bounding box rather than by anything visible.
    ax.set_xlim(x0, max(x1, xmax + 0.34 * (x1 - x0)))
    lab_x = xmax + 0.05 * (x1 - x0)

    # 0.050 was the label height itself, leaving no leading; 0.070 gives ~1.4x.
    gap = 0.070 * (y1 - y0)
    ends = sorted(ends, key=lambda t: -t[2])
    ys = []
    for _, _, y in ends:
        ys.append(min(y, ys[-1] - gap) if ys else y)
    # If the stack has been pushed below the axes, lift the whole column.
    if ys and ys[-1] < y0 + 0.02 * (y1 - y0):
        shift = (y0 + 0.02 * (y1 - y0)) - ys[-1]
        ys = [v + shift for v in ys]

    for (cell, x, y_end), y_lab in zip(ends, ys):
        if abs(y_lab - y_end) > 1e-9:
            ax.plot([x, lab_x], [y_end, y_lab], color=C_DEP, lw=0.5,
                    alpha=0.5, zorder=3)
        ax.text(lab_x + 0.008 * (x1 - x0), y_lab, S.cell_label(cell),
                va="center", fontsize=S.FS_VALUE, color=C_DEP)


def panel_a(ax, d):
    ends = []
    for cell, e in d["cells"].items():
        for tag, col, ls, lw in (("f20_annotated", C_ANN, "--", 1.0),
                                 ("f18_deployment", C_DEP, "-", 1.7)):
            c = e[tag]["curve"]
            ks = sorted(c, key=float)
            n = [c[k]["n_train_median"] for k in ks]
            a = [c[k]["auroc_mean"] for k in ks]
            ax.plot(n, a, ls, color=col, lw=lw, alpha=0.85, zorder=3 if lw > 1 else 2)
            if tag == "f18_deployment":
                ax.scatter(n[-1:], a[-1:], s=22, color=col, zorder=4)
                ends.append((cell, n[-1], a[-1]))
    _endpoint_labels(ax, ends)
    ax.axhline(0.5, color=C_TH, lw=1.0, ls=":", zorder=1)
    ax.text(ax.get_xlim()[0], 0.505, " chance", fontsize=S.FS_VALUE, color=C_TH, va="bottom")
    ax.set_xlabel("training rows")
    ax.set_ylabel("out-of-fold AUROC")
    S.panel_label(ax, "A  Learning curves")
    sat = sum(1 for e in d["cells"].values()
              for t in ("f20_annotated", "f18_deployment")
              if _sat(e[t]["curve"]))
    tot = 2 * len(d["cells"])
    ax.legend(handles=[plt.Line2D([], [], color=C_DEP, lw=1.7,
                                  label="18 features (deployment)"),
                       plt.Line2D([], [], color=C_ANN, lw=1.0, ls="--",
                                  label="20 features (annotated)")],
              loc="upper left", frameon=False)


SPLIT = 0.55   # must match router_undersampling_control.py's saturation table


def _sat(curve):
    """Saturating iff the SECOND half of the sweep buys less AUROC than the first.

    The split is at 0.55, not at the midpoint index, and the choice matters: it
    gives the second half the LARGER span (0.55->1.00 vs 0.25->0.55), so it is
    biased towards calling a curve "still climbing". Using the midpoint index
    instead would report 12/12 saturating; this reports 9/12. The conservative
    count is the one quoted, because the conclusion leans on it.
    """
    g = lambda f: (curve[str(f)] if str(f) in curve else curve[f])["auroc_mean"]  # noqa: E731
    return (g(1.00) - g(SPLIT)) < (g(SPLIT) - g(0.25))


def panel_b(ax, d):
    cells = list(d["cells"])
    y = np.arange(len(cells))[::-1]
    for yi, cell in zip(y, cells):
        i = d["cells"][cell]["f18_deployment"]["insample"]
        ax.plot([i["train_auroc_perm_mean"], i["train_auroc_real"]], [yi, yi],
                color=C_DEP, lw=2.0, alpha=0.5, zorder=2)
        ax.scatter([i["train_auroc_perm_mean"]], [yi], s=42, color=C_ANN,
                   zorder=3, edgecolor="white", lw=0.8)
        ax.scatter([i["train_auroc_real"]], [yi], s=46, color=C_DEP, zorder=4)
        ax.text(i["train_auroc_real"] + 0.008, yi,
                f"+{i['excess_over_perm']:.3f}", va="center", fontsize=S.FS_VALUE,
                color=C_DEP, fontweight="bold")
    ax.set_yticks(y, [S.cell_label(c) for c in cells], fontsize=S.FS_LABEL)
    ax.set_xlabel("in-sample AUROC, near-unregularised fit")
    S.panel_label(ax, "B  Real labels against permuted")
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=7, color=C_ANN,
                                  label="permuted"),
                       plt.Line2D([], [], marker="o", ls="", ms=7, color=C_DEP,
                                  label="real")],
              loc="upper right", frameon=False)


def panel_c(ax, d):
    """The bound that settles it: more data reaches toward the oracle, and the
    oracle was already measured."""
    ax.axis("off")
    best = max(e["f18_deployment"]["curve"][k]["auroc_mean"]
               for e in d["cells"].values()
               for k in e["f18_deployment"]["curve"])
    ax.add_patch(plt.Rectangle((0.02, 0.62), 0.96, 0.28, transform=ax.transAxes,
                               facecolor="#F2F2F2", edgecolor="#CCCCCC", lw=0.8))
    ax.text(0.5, 0.845, "what more data can buy, and what it cannot",
            transform=ax.transAxes, ha="center", fontsize=S.FS_PANEL, fontweight="bold")
    ax.text(0.5, 0.70,
            f"more rows move the learned predictor\nfrom AUROC {best:.2f} upward — at "
            "best toward\na perfect predictor, AUROC = 1.00",
            transform=ax.transAxes, ha="center", va="center", fontsize=S.FS_VALUE,
            linespacing=1.6)
    ax.annotate("", xy=(0.5, 0.57), xytext=(0.5, 0.62),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C_TH, lw=1.4))
    ax.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.50, transform=ax.transAxes,
                               facecolor="#FDF0E6", edgecolor=C_HOT, lw=1.3))
    ax.text(0.5, 0.470, "a perfect predictor IS the oracle —\nand we measured it",
            transform=ax.transAxes, ha="center", va="center", fontsize=S.FS_LABEL,
            fontweight="bold", color=C_HOT, linespacing=1.5)
    ax.text(0.5, 0.255,
            "the retrospective oracle Pareto-beats\nalways-cheapest in 1 of 8 cells "
            "(Ch 6).\nThe learned router does so in 0 of 8.\n\n"
            f"Undersampling explains the gap from {best:.2f} to 1.00.\n"
            "It cannot explain a boundary the oracle\nalso fails to cross.",
            transform=ax.transAxes, ha="center", va="center", fontsize=S.FS_VALUE,
            linespacing=1.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    d = load()

    # Panel C was a rectangle of prose arguing that a perfect predictor is the
    # oracle and the oracle was already measured. That argument is the chapter's
    # to make in sentences; it was never data, so it is no longer drawn.
    S.apply()
    fig, axes = plt.subplots(2, 1, figsize=(S.PRINT_W_IN, 6.0))
    panel_a(axes[0], d)
    panel_b(axes[1], d)
    axes[1].spines["left"].set_visible(False)
    fig.tight_layout()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    sat = sum(1 for e in d["cells"].values()
              for t in ("f20_annotated", "f18_deployment") if _sat(e[t]["curve"]))
    print(f"wrote {a.out}.png / .pdf   ({sat}/{2 * len(d['cells'])} curves "
          f"saturating; best deployment AUROC {best_auroc(d):.3f})")
    return 0


def best_auroc(d):
    return max(e["f18_deployment"]["curve"][k]["auroc_mean"]
               for e in d["cells"].values() for k in e["f18_deployment"]["curve"])


if __name__ == "__main__":
    sys.exit(main())
