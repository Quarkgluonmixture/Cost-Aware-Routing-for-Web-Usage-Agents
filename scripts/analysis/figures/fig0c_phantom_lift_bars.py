#!/usr/bin/env python3
"""[Outcome 0c viz] Outcome dimension — phantom routing lift visualization.

Output:
- results/phantom_paper/figures/fig0c_phantom_lift_bars.png

Visual companion to Outcome 0c routing oracle lift.

See docs/checkpoints/paper_planning.md §3 Outcome dimension framework.

fig10: Phantom routing ADDITIVE lift — Appendix-D / H1-deploy sensitivity figure.

⚠️ AMENDMENT_02 §3 / AMENDMENT_04 (2026-05-24): this figure visualizes the 4-mode ADD
estimand (3→{4,5}-mode incremental oracle lift, `4psom_vs_3` etc.) = Appendix-D /
H1-deploy sensitivity, NOT the 6-mode-strict H1 drop-one PRIMARY gate
(`_cell_drop_one_theta_se`; 6-mode strict ≤ 4-mode ADD by construction). Do NOT cite this
as the §1 H1 hero number — the canonical H1 hero = bootstrap-percentile FE pool from
`phase1_full_prereg_decision`.

Reads results/phantom_paper/phantom_lift.csv (produced by aggregate_phantom_lift.py)
and visualizes 4 oracle ceilings per (baseline, site) cell:
  3-mode | +P-text (4-mode) | +P-SoM (4-mode) | 5-mode

Each bar = oracle SR% (ADD ceiling). Error bars from bootstrap 95% CI on the lift vs 3-mode.
Text annotations show absolute ceiling % and lift Δ vs 3-mode.

Output: results/phantom_paper/figures/fig0c_phantom_lift_bars.png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CSV = ROOT / "results/phantom_paper/phantom_lift.csv"
OUT = ROOT / "results/phantom_paper/figures/fig0c_phantom_lift_bars.png"


def load_rows() -> list[dict]:
    with CSV.open() as f:
        return list(csv.DictReader(f))


def main() -> None:
    rows = load_rows()
    if not rows:
        print(f"[warn] no rows in {CSV} — run `make phantom-lift` first")
        return

    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 5.0), sharey=False)
    if n == 1:
        axes = [axes]

    plt.rcParams.update({"font.size": 10})

    BARS = ["3-mode", "+P-text", "+P-prompt", "+P-SoM", "5-mode", "6-mode"]
    COLORS = ["#a0a0a0", "#9e6da8", "#9467bd", "#b279a2", "#54a24b", "#2e7d32"]

    def _to_float(value: str) -> float | None:
        if value is None or value == "" or value.lower() == "none":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    for ax, r in zip(axes, rows):
        sr3 = _to_float(r["oracle_3mode_pp"]) or 0.0
        sr_pdom = _to_float(r["oracle_4mode_pdom_pp"])  # may be None when P-text pending
        sr_pprompt = _to_float(r.get("oracle_4mode_pprompt_pp"))  # may be None when P-prompt pending
        sr_psom = _to_float(r["oracle_4mode_psom_pp"]) or 0.0
        sr5 = _to_float(r["oracle_5mode_pp"])  # may be None
        sr6 = _to_float(r.get("oracle_6mode_pp"))  # may be None when P-prompt pending

        ci_lo_pdom_lift = _to_float(r["lift_4pdom_vs_3_ci95_lo_pp"])
        ci_hi_pdom_lift = _to_float(r["lift_4pdom_vs_3_ci95_hi_pp"])
        ci_lo_pprompt_lift = _to_float(r.get("lift_4pprompt_vs_3_ci95_lo_pp"))
        ci_hi_pprompt_lift = _to_float(r.get("lift_4pprompt_vs_3_ci95_hi_pp"))
        ci_lo_5_lift = _to_float(r["lift_5_vs_3_ci95_lo_pp"])
        ci_hi_5_lift = _to_float(r["lift_5_vs_3_ci95_hi_pp"])
        ci_lo_psom_lift = _to_float(r["lift_4psom_vs_3_ci95_lo_pp"]) or 0.0
        ci_hi_psom_lift = _to_float(r["lift_4psom_vs_3_ci95_hi_pp"]) or 0.0
        ci_lo_6_lift = _to_float(r.get("lift_6_vs_3_ci95_lo_pp"))
        ci_hi_6_lift = _to_float(r.get("lift_6_vs_3_ci95_hi_pp"))

        # /stress A1.20 P0-2-A* (2026-05-17, A1.19 B-429 figure-layer fill):
        # use per-comparison universe baseline for CI rendering. A1.19 B-429 fix
        # added `lift_4psom_vs_3_pp` = `sr_4_psom - sr_3_psom_only` (over u_psom).
        # Bar height is `sr_psom` (over u_psom). CI must be rooted at same baseline
        # universe — derive `sr_3_psom_only = sr_psom - lift_4psom_vs_3_pp` to
        # back-compute baseline-of-comparison and root CI bars to it. Pre-fix:
        # `sr3 + ci_lo_psom_lift` mixed universe_5 baseline (sr3) with u_psom lift.
        lift_psom_pp = _to_float(r.get("lift_4psom_vs_3_pp")) or 0.0
        sr3_psom_universe = sr_psom - lift_psom_pp
        ci_lo_psom = sr3_psom_universe + ci_lo_psom_lift
        ci_hi_psom = sr3_psom_universe + ci_hi_psom_lift

        # Build sr_vals; pdom / pprompt / 5-mode / 6-mode may be None (pending)
        sr_pdom_plot = 0.0 if sr_pdom is None else sr_pdom
        sr_pprompt_plot = 0.0 if sr_pprompt is None else sr_pprompt
        sr5_plot = 0.0 if sr5 is None else sr5
        sr6_plot = 0.0 if sr6 is None else sr6

        sr_vals = [sr3, sr_pdom_plot, sr_pprompt_plot, sr_psom, sr5_plot, sr6_plot]
        err_low = [
            0,
            0 if sr_pdom is None or ci_lo_pdom_lift is None else max(0, sr_pdom - (sr3 + ci_lo_pdom_lift)),
            0 if sr_pprompt is None or ci_lo_pprompt_lift is None else max(0, sr_pprompt - (sr3 + ci_lo_pprompt_lift)),
            max(0, sr_psom - ci_lo_psom),
            0 if sr5 is None or ci_lo_5_lift is None else max(0, sr5 - (sr3 + ci_lo_5_lift)),
            0 if sr6 is None or ci_lo_6_lift is None else max(0, sr6 - (sr3 + ci_lo_6_lift)),
        ]
        err_high = [
            0,
            0 if sr_pdom is None or ci_hi_pdom_lift is None else max(0, (sr3 + ci_hi_pdom_lift) - sr_pdom),
            0 if sr_pprompt is None or ci_hi_pprompt_lift is None else max(0, (sr3 + ci_hi_pprompt_lift) - sr_pprompt),
            max(0, ci_hi_psom - sr_psom),
            0 if sr5 is None or ci_hi_5_lift is None else max(0, (sr3 + ci_hi_5_lift) - sr5),
            0 if sr6 is None or ci_hi_6_lift is None else max(0, (sr3 + ci_hi_6_lift) - sr6),
        ]

        x = np.arange(len(BARS))
        # Use grey placeholder for pending bars
        bar_colors = list(COLORS)
        pending_idx: list[int] = []
        if sr_pdom is None:
            bar_colors[1] = "#dddddd"
            pending_idx.append(1)
        if sr_pprompt is None:
            bar_colors[2] = "#dddddd"
            pending_idx.append(2)
        if sr5 is None:
            bar_colors[4] = "#dddddd"
            pending_idx.append(4)
        if sr6 is None:
            bar_colors[5] = "#dddddd"
            pending_idx.append(5)

        bars = ax.bar(x, sr_vals, color=bar_colors, width=0.66,
                      yerr=[err_low, err_high], ecolor="#222222", capsize=4,
                      error_kw={"linewidth": 1.0})
        for idx in pending_idx:
            bars[idx].set_hatch("//")
            bars[idx].set_edgecolor("#999999")

        for i, (bar, val) in enumerate(zip(bars, sr_vals)):
            if i in pending_idx:
                ax.text(bar.get_x() + bar.get_width()/2, max(val, 1.0) + 0.5,
                        "(pending)", ha="center", va="bottom", fontsize=8, color="#666666")
                continue
            if i == 0:
                lift_label = "(baseline)"
            else:
                delta = val - sr3
                lift_label = f"Δ +{delta:.2f}pp"
            ax.text(bar.get_x() + bar.get_width()/2, val + max(err_high[i], 0.0) + 0.5,
                    f"{val:.2f}%\n{lift_label}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold" if i in (4, 5) else "normal")

        n_label = (f"N={r['n_common']}/{r['n_expected']}†" if r["is_partial"].lower() == "true"
                   else f"N={r['n_common']}")
        ax.set_title(f"{r['baseline']} {r['site']} ({n_label})", fontsize=11, fontweight="bold")
        ax.set_xticks(x, BARS, fontsize=9.5)
        ax.set_ylabel("Oracle ceiling SR (%)")
        # Per-panel zoomed y-axis (sharey=False) so each cell's lift differences are visible
        max_with_err = max(sr_vals[i] + err_high[i] for i in range(len(BARS))) or 1.0
        ax.set_ylim(0, max_with_err * 1.30)
        ax.grid(axis="y", color="#dddddd", linewidth=0.8)
        ax.set_axisbelow(True)

    fig.suptitle("Phantom routing lift: 3-mode → +P-text → +P-prompt → +P-SoM → 5-mode → 6-mode oracle ceiling",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.012,
             "Bars = oracle ceiling success rate (any-of-modes solves task). Error bars: 95% bootstrap CI. "
             "Δ = lift vs 3-mode baseline. Phantom modes (P-text/P-prompt/P-SoM) are zero-extra-image-cost (text+optional SoM marks).",
             ha="center", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(OUT)


if __name__ == "__main__":
    main()
