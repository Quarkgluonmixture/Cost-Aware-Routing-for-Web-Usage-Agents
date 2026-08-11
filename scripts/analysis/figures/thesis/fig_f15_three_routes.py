#!/usr/bin/env python3
"""Thesis F15 — three routes out of the label-supply constraint, each blocked.

The mechanism claim in Ch7 is that the which-mode router fails on label SUPPLY,
not on hypothesis class. A supply claim is only convincing if the obvious ways
of manufacturing more supervision are shown to fail — and, crucially, to fail
for *different* reasons. Three failures with one shared cause would just be one
failure counted thrice.

So the figure's job is to make the three failure modes visually distinct:
    A  continuous label   -> blocked by the instrument (evaluator emits binary)
    B  pool across cells  -> blocked by identifiability (same task, different
                             label under a different backbone)
    C  coarsen the target -> open, but converts the problem into a smaller one

Every number is read from router_label_supply_diagnosis.json. The one quantity
the source itself flags as commonly misread — the "Bayes ceiling" columns are
resubstitution estimates, not bounds — is relabelled in the axis text rather
than reproduced under the misleading name.

Output: final_dissertation/figures/fig_f15_three_routes.{png,pdf}
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "results/phantom_paper/router_label_supply_diagnosis.json"
SRC_EVAL = ROOT / "docs/analysis/cross_sites/evaluator_score_granularity.md"
OUT = ROOT / "final_dissertation/figures/fig_f15_three_routes"

C_BLOCK = "#D55E00"
C_OPEN = "#0072B2"
C_GREY = "#999999"


def load():
    d = json.loads(SRC.read_text(encoding="utf-8"))
    sup, ident = d["supply"], d["identifiability"]["per_site"]
    if sup["n_cells"] != 6:
        raise SystemExit(f"{SRC}: expected 6 cells, got {sup['n_cells']}")
    return d, sup, ident


def panel_a(ax, sup):
    """Route 1 — a graded label would make every task informative."""
    labels = ["tasks\nscored", "tasks any\nmode solves", "which-mode\nlabels"]
    per = sup["per_cell"]
    total_universe = sum(v["n_universe"] for v in per.values())
    total_labels = sup["pooled_total"]
    vals = [total_universe, total_labels, total_labels]
    ax.bar(labels, vals, color=[C_GREY, C_BLOCK, C_BLOCK], width=0.62)
    for x, v in enumerate(vals):
        ax.text(x, v + total_universe * 0.025, f"{v:,}", ha="center",
                fontsize=9.0, fontweight="bold")
    ax.set_ylim(0, total_universe * 1.20)
    ax.set_title("A · make the label continuous", fontsize=9.8, loc="left",
                 fontweight="bold", pad=6)
    ax.text(0.5, 0.62,
            "blocked by the instrument\n\nthe evaluator returns a single\n"
            "conjunctive pass/fail — there is\nno partial-credit channel to\n"
            "expose, so the second bar\ncannot be widened",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.2,
            color=C_BLOCK, linespacing=1.6)
    ax.set_ylabel("tasks (pooled over 6 VWA cells)", fontsize=8.4)


def panel_b(ax, ident):
    """Route 2 — pooling multiplies rows and destroys identifiability."""
    sites = list(ident.keys())
    shared = [ident[s]["n_tasks_shared_by_2plus_cells"] for s in sites]
    conflict = [ident[s]["n_tasks_conflicting"] for s in sites]
    x = range(len(sites))
    ax.bar(x, shared, color=C_GREY, width=0.5, label="shared by $\\geq$2 cells")
    ax.bar(x, conflict, color=C_BLOCK, width=0.5, label="carry conflicting labels")
    for i, s in enumerate(sites):
        r = ident[s]["conflict_rate_pct"]
        ax.text(i, shared[i] + max(shared) * 0.04,
                f"{conflict[i]}/{shared[i]}\n= {r:.1f}%", ha="center",
                fontsize=8.6, fontweight="bold", color=C_BLOCK,
                linespacing=1.4)
    ax.set_xticks(list(x), sites, fontsize=9.0)
    ax.set_ylim(0, max(shared) * 1.34)
    ax.set_title("B · pool across backbones", fontsize=9.8, loc="left",
                 fontweight="bold", pad=6)
    ax.text(0.5, 0.55,
            "blocked by identifiability\n\nthe same task gets a different\n"
            "label under a different backbone,\nso the pooled target is not a\n"
            "function of the pooled features",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.2,
            color=C_BLOCK, linespacing=1.6)
    ax.legend(frameon=False, fontsize=7.8, loc="upper left")
    ax.set_ylabel("tasks", fontsize=8.4)


def panel_c(ax, ident):
    """Route 3 — coarsening is open, and leads somewhere smaller."""
    sites = list(ident.keys())
    which = [ident[s]["bayes_ceiling_which_mode_pct"] for s in sites]
    tier = [ident[s]["bayes_ceiling_cost_tier_pct"] for s in sites]
    x = range(len(sites))
    w = 0.34
    ax.bar([i - w / 2 for i in x], which, width=w, color=C_GREY,
           label="target = which of 6 modes")
    ax.bar([i + w / 2 for i in x], tier, width=w, color=C_OPEN,
           label="target = cost tier")
    for i in x:
        ax.text(i - w / 2, which[i] + 1.4, f"{which[i]:.1f}", ha="center",
                fontsize=8.4)
        ax.text(i + w / 2, tier[i] + 1.4, f"{tier[i]:.1f}", ha="center",
                fontsize=8.4, fontweight="bold", color=C_OPEN)
    ax.set_xticks(list(x), sites, fontsize=9.0)
    ax.set_ylim(0, 118)
    ax.set_title("C · coarsen the target", fontsize=9.8, loc="left",
                 fontweight="bold", pad=6)
    ax.text(0.5, 0.40,
            "open — but smaller\n\nsame features, same solve events,\n"
            "a better-posed question. It buys a\ncoarser decision, not more\n"
            "supervision.",
            transform=ax.transAxes, ha="center", va="center", fontsize=8.2,
            color=C_OPEN, linespacing=1.6)
    ax.legend(frameon=False, fontsize=7.8, loc="lower left")
    ax.set_ylabel("in-sample modal agreement (%)", fontsize=8.4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    d, sup, ident = load()

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.0))
    panel_a(axes[0], sup)
    panel_b(axes[1], ident)
    panel_c(axes[2], ident)
    for ax in axes:
        ax.grid(axis="y", color="#F2F2F2", lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.suptitle("Three ways out of the label shortage — and three different "
                 "reasons none of them works",
                 fontsize=12.0, fontweight="bold", x=0.006, ha="left", y=1.045)
    fig.text(0.006, 1.000,
             f"Labels for the which-mode router exist only where some mode "
             f"succeeded: {sup['labels_min']}-{sup['labels_max']} per cell, "
             f"{sup['pooled_total']} pooled over six cells, and "
             f"{d['trainability']['n_cells_untrainable']} of "
             f"{d['trainability']['n_cells']} cells with no trainable classifier "
             "at all. If the shortage were a slicing problem one of these\n"
             "routes would relieve it. Each fails in a structurally different "
             "way, which is the argument that the constraint is a production "
             "rate rather than a labelling choice.",
             fontsize=8.4, color="#444444", linespacing=1.5, va="top")
    fig.text(0.006, -0.055,
             "Source: results/phantom_paper/router_label_supply_diagnosis.json. "
             "Panel C's percentages are RESUBSTITUTION estimates (each scores "
             "the rows it took its modal label from, so singleton groups are "
             "correct by construction — 29.8% / 37.0% of rows); they are "
             "in-sample modal agreement, not Bayes ceilings, and are not used "
             "as bounds anywhere.",
             fontsize=7.0, color="#888888", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{a.out}.{ext}", dpi=220, bbox_inches="tight",
                    facecolor="white")
    plt.close(fig)
    print(f"wrote {a.out}.png / .pdf   (pooled labels {sup['pooled_total']}, "
          f"{d['trainability']['n_cells_untrainable']}/"
          f"{d['trainability']['n_cells']} cells untrainable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
