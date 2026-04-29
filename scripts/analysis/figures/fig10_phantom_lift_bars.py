#!/usr/bin/env python3
"""fig10: Phantom routing lift — Section 1/4 paper hook visualization.

Reads results/phantom_paper/phantom_lift.csv (produced by aggregate_phantom_lift.py)
and visualizes 4 oracle ceilings per (baseline, site) cell:
  3-mode | +P-DOM (4-mode) | +P-SoM (4-mode) | 5-mode

Each bar = oracle SR%. Error bars from bootstrap 95% CI on the lift vs 3-mode.
Text annotations show absolute ceiling % and lift Δ vs 3-mode.

Output: results/phantom_paper/figures/fig10_phantom_lift_bars.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "results/phantom_paper/phantom_lift.csv"
OUT = ROOT / "results/phantom_paper/figures/fig10_phantom_lift_bars.png"


def load_rows() -> list[dict]:
    with CSV.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_rows()
    if not rows:
        print(f"[warn] no rows in {CSV} — run `make phantom-lift` first")
        return

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.0), sharey=True)
    if n == 1:
        axes = [axes]

    plt.rcParams.update({"font.size": 10})

    BARS = ["3-mode", "+P-DOM", "+P-SoM", "5-mode"]
    COLORS = ["#a0a0a0", "#9e6da8", "#b279a2", "#54a24b"]

    for ax, r in zip(axes, rows):
        sr3 = float(r["oracle_3mode_pp"])
        sr_pdom = float(r["oracle_4mode_pdom_pp"])
        sr_psom = float(r["oracle_4mode_psom_pp"])
        sr5 = float(r["oracle_5mode_pp"])

        # err bars only on the lift bars (vs 3-mode, anchored at 3-mode level)
        # Use the bootstrap CI from CSV; convert to error magnitudes
        ci_lo_pdom = sr3 + float(r["lift_4pdom_vs_3_ci95_lo_pp"])
        ci_hi_pdom = sr3 + float(r["lift_4pdom_vs_3_ci95_hi_pp"])
        ci_lo_psom = sr3 + float(r["lift_4psom_vs_3_ci95_lo_pp"])
        ci_hi_psom = sr3 + float(r["lift_4psom_vs_3_ci95_hi_pp"])
        ci_lo_5    = sr3 + float(r["lift_5_vs_3_ci95_lo_pp"])
        ci_hi_5    = sr3 + float(r["lift_5_vs_3_ci95_hi_pp"])

        sr_vals  = [sr3, sr_pdom, sr_psom, sr5]
        err_low  = [0, max(0, sr_pdom - ci_lo_pdom), max(0, sr_psom - ci_lo_psom), max(0, sr5 - ci_lo_5)]
        err_high = [0, max(0, ci_hi_pdom - sr_pdom), max(0, ci_hi_psom - sr_psom), max(0, ci_hi_5 - sr5)]

        x = np.arange(len(BARS))
        bars = ax.bar(x, sr_vals, color=COLORS, width=0.66,
                      yerr=[err_low, err_high], ecolor="#222222", capsize=4,
                      error_kw={"linewidth": 1.0})

        for i, (bar, val) in enumerate(zip(bars, sr_vals)):
            if i == 0:
                lift_label = "(baseline)"
            else:
                delta = val - sr3
                lift_label = f"Δ +{delta:.2f}pp"
            ax.text(bar.get_x() + bar.get_width()/2, val + max(err_high[i], 0.0) + 0.5,
                    f"{val:.2f}%\n{lift_label}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold" if i == 3 else "normal")

        n_label = (f"N={r['n_common']}/{r['n_expected']}†" if r["is_partial"].lower() == "true"
                   else f"N={r['n_common']}")
        ax.set_title(f"{r['baseline']} {r['site']} ({n_label})", fontsize=11, fontweight="bold")
        ax.set_xticks(x, BARS, fontsize=9.5)
        ax.set_ylabel("Oracle ceiling SR (%)" if ax is axes[0] else "")
        ax.set_ylim(0, max(sr_vals) * 1.35)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Phantom routing lift: 3-mode → +P-DOM → +P-SoM → 5-mode oracle ceiling",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.012,
             "Bars = oracle ceiling success rate (any-of-modes solves task). Error bars: 95% bootstrap CI. "
             "Δ = lift vs 3-mode baseline. Phantom modes are zero-cost (text-only inference).",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
