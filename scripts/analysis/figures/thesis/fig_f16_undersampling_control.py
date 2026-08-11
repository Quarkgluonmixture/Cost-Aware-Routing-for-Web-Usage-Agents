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
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
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


def panel_a(ax, d):
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
                ax.annotate(cell, (n[-1], a[-1]), textcoords="offset points",
                            xytext=(5, -2), fontsize=7.0, color=C_DEP)
    ax.axhline(0.5, color=C_TH, lw=1.0, ls=":", zorder=1)
    ax.text(ax.get_xlim()[0], 0.505, " chance", fontsize=7.2, color=C_TH, va="bottom")
    ax.set_xlabel("training rows (test folds never thinned)", fontsize=8.6)
    ax.set_ylabel("out-of-fold AUROC", fontsize=8.6)
    ax.set_title("A · does more of the same data help?", fontsize=9.8,
                 loc="left", fontweight="bold", pad=6)
    sat = sum(1 for e in d["cells"].values()
              for t in ("f20_annotated", "f18_deployment")
              if _sat(e[t]["curve"]))
    tot = 2 * len(d["cells"])
    ax.text(0.97, 0.05,
            f"yes — but with decelerating returns:\n{sat} of {tot} curves buy less over "
            "0.55→1.00\nthan over 0.25→0.55, despite the later span\nbeing the wider "
            "one of the two",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.8,
            color=C_TH, linespacing=1.55)
    ax.legend(handles=[plt.Line2D([], [], color=C_DEP, lw=1.7,
                                  label="18 features (deployment)"),
                       plt.Line2D([], [], color=C_ANN, lw=1.0, ls="--",
                                  label="20 (adds the benchmark's own\ndifficulty annotation)")],
              loc="upper left", frameon=False, fontsize=7.6)


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
                f"+{i['excess_over_perm']:.3f}", va="center", fontsize=7.4,
                color=C_DEP, fontweight="bold")
    ax.set_yticks(y, cells, fontsize=8.4)
    ax.set_xlabel("in-sample AUROC, near-unregularised fit", fontsize=8.6)
    ax.set_title("B · is there signal to find at all?", fontsize=9.8,
                 loc="left", fontweight="bold", pad=6)
    ax.tick_params(axis="y", length=0)
    ax.text(0.03, 0.06,
            "yes. Every cell separates its own rows further than the\n"
            "same model fitted to PERMUTED labels does (grey).\n"
            "So the features are not noise — which means the\n"
            "objection cannot be dismissed on a no-signal finding.",
            transform=ax.transAxes, fontsize=7.8, color=C_TH, linespacing=1.55)
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=7, color=C_ANN,
                                  label="permuted labels (memorisation floor)"),
                       plt.Line2D([], [], marker="o", ls="", ms=7, color=C_DEP,
                                  label="real labels")],
              loc="upper right", frameon=False, fontsize=7.6)


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
            transform=ax.transAxes, ha="center", fontsize=9.2, fontweight="bold")
    ax.text(0.5, 0.70,
            f"more rows move the learned predictor\nfrom AUROC {best:.2f} upward — at "
            "best toward\na perfect predictor, AUROC = 1.00",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.0,
            linespacing=1.6)
    ax.annotate("", xy=(0.5, 0.57), xytext=(0.5, 0.62),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color=C_TH, lw=1.4))
    ax.add_patch(plt.Rectangle((0.02, 0.06), 0.96, 0.50, transform=ax.transAxes,
                               facecolor="#FDF0E6", edgecolor=C_HOT, lw=1.3))
    ax.text(0.5, 0.470, "a perfect predictor IS the oracle —\nand we measured it",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.8,
            fontweight="bold", color=C_HOT, linespacing=1.5)
    ax.text(0.5, 0.255,
            "the retrospective oracle Pareto-beats\nalways-cheapest in 1 of 8 cells "
            "(Ch 6).\nThe learned router does so in 0 of 8.\n\n"
            f"Undersampling explains the gap from {best:.2f} to 1.00.\n"
            "It cannot explain a boundary the oracle\nalso fails to cross.",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.0,
            linespacing=1.6)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    d = load()

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.9))
    panel_a(axes[0], d)
    panel_b(axes[1], d)
    panel_c(axes[2], d)
    for ax in axes[:2]:
        ax.grid(color="#F2F2F2", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[1].spines["left"].set_visible(False)

    fig.suptitle("The undersampling objection has force against the predictor, "
                 "and none against the result",
                 fontsize=12.0, fontweight="bold", x=0.006, ha="left", y=1.045)
    fig.text(0.006, 1.000,
             "\"You did not measure a mechanism, you measured a small sample\" is "
             "the cheapest way to dismiss a negative result, so it is answered "
             "here with measurements rather than prose. Two of the three panels "
             "come back\nPARTLY AGAINST this dissertation: the curves are still "
             "rising and the signal is real. The third is why that does not "
             "rescue the router.",
             fontsize=8.4, color="#444444", linespacing=1.5, va="top")
    fig.text(0.006, -0.045,
             f"Source: docs/analysis/cross_sites/router_undersampling_control.json "
             f"({d['n_splits']}-fold × {d['n_repeat']} repeats, seed {d['seed']}; "
             "triage label; training rows thinned stratified, test folds intact). "
             "Oracle comparison from router_triage_learnability_with_wa.md.",
             fontsize=7.0, color="#888888")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

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
