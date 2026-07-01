#!/usr/bin/env python3
"""Paper F2 — H1 forest: per-cell P-SoM strict 6-mode drop-one + FE pooled diamond.

Estimand discipline: this figure plots the H1 GATE estimand (6-mode strict
drop-one loss for P-SoM, task-paired bootstrap CI per cell; FE inverse-variance
pool with percentile CI). It deliberately does NOT reuse fig_forest_drop_one.py,
whose "vs 3-mode oracle" ADD-lift estimand is the Amendment-02-flagged sibling
that OVERSTATES the strict gate effect — mixing them in the paper would hand a
reviewer the estimand-conflation attack.

Sources (read-only, no recomputation):
  results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv   (per-cell per-arm strict drop-one)
  results/phantom_paper/phase1_full_prereg_decision.json  (pooled_h1_fe / pooled_h1_bootstrap / gate_status)

Guards: panels with <6 modes are skipped (3-mode partial portfolio ≠ 6-mode
estimand, NUMBERS_TODO §0); gate_status != COMPLETE stamps an INTERIM watermark
so a rehearsal render cannot silently end up in the submission.

Output: results/phantom_paper/figures/fig_f2_h1_forest.{png,pdf}
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
FIG0C = ROOT / "results/phantom_paper/fig0c_drop_one_bootstrap_ci.csv"
DECISION = ROOT / "results/phantom_paper/phase1_full_prereg_decision.json"
OUT = ROOT / "results/phantom_paper/figures/fig_f2_h1_forest"

DELTA_PP = 1.0
ARM = "P-SoM"
C_CELL = "#0072B2"
C_POOL = "#D55E00"


def main() -> int:
    rows = list(csv.DictReader(FIG0C.open()))
    dec = json.loads(DECISION.read_text())

    n_modes = {}
    for r in rows:
        n_modes[r["site_baseline"]] = n_modes.get(r["site_baseline"], 0) + 1
    cells, skipped = [], []
    for r in rows:
        if r["mode"] != ARM:
            continue
        if n_modes[r["site_baseline"]] < 6:
            skipped.append(r["site_baseline"])
            continue
        cells.append((
            r["site_baseline"],
            float(r["drop_one_loss_pp"]),
            float(r["ci95_low_pp"]),
            float(r["ci95_high_pp"]),
        ))
    if skipped:
        print(f"note: skipped partial (<6-mode) panels: {sorted(set(skipped))}")

    fe = dec.get("pooled_h1_fe", {})
    boot = dec.get("pooled_h1_bootstrap", {})
    gate_status = dec.get("gate_status", "MISSING")
    theta = fe.get("theta_FE_pp")
    ci_lo = boot.get("ci95_lo_pp_bootstrap")
    ci_hi = boot.get("ci95_hi_pp_bootstrap")
    passed = boot.get("gate_passed_bootstrap")
    k = boot.get("k_cells")

    n = len(cells)
    fig, ax = plt.subplots(figsize=(6.6, 0.62 * (n + 2) + 1.4), dpi=300)
    ys = list(range(n, 0, -1))
    for (label, d, lo, hi), y in zip(cells, ys):
        ax.plot([lo, hi], [y, y], color=C_CELL, lw=1.8, zorder=2)
        ax.plot([lo, lo], [y - 0.12, y + 0.12], color=C_CELL, lw=1.8)
        ax.plot([hi, hi], [y - 0.12, y + 0.12], color=C_CELL, lw=1.8)
        ax.scatter([d], [y], s=54, color=C_CELL, zorder=3)
        ax.annotate(f"{d:+.2f} [{lo:.2f}, {hi:.2f}]", (hi, y), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=8, color="#333333")
    if isinstance(theta, (int, float)) and isinstance(ci_lo, (int, float)):
        y0 = 0
        ax.fill([ci_lo, theta, ci_hi, theta], [y0, y0 + 0.22, y0, y0 - 0.22],
                color=C_POOL, alpha=0.85, zorder=3)
        verdict = "PASS" if passed else "fail"
        ax.annotate(
            f"FE pool (k={k}): {theta:+.2f} [{ci_lo:.2f}, {ci_hi:.2f}]  gate({DELTA_PP:+.1f}pp): {verdict}",
            (max(ci_hi, DELTA_PP), y0), xytext=(8, 0), textcoords="offset points",
            va="center", fontsize=8, fontweight="bold", color=C_POOL)

    ax.axvline(0.0, color="#555555", lw=0.9, ls="--", zorder=1)
    ax.axvline(DELTA_PP, color=C_POOL, lw=0.9, ls=":", zorder=1)
    ax.text(DELTA_PP, n + 0.75, f"gate {DELTA_PP:+.1f}pp", fontsize=7.5,
            color=C_POOL, ha="center")

    ax.set_yticks(ys + [0])
    ax.set_yticklabels([c[0] for c in cells] + ["FE pool"], fontsize=9)
    ax.set_xlabel(f"{ARM} drop-one loss from 6-mode portfolio (pp; task-paired bootstrap 95% CI)",
                  fontsize=9)
    ax.set_ylim(-0.8, n + 1.1)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("H1: pooled P-SoM drop-one vs +1.0pp substantive threshold", fontsize=10)

    if gate_status != "COMPLETE":
        fig.text(0.5, 0.5, f"INTERIM ({gate_status}) — NOT A VERDICT",
                 fontsize=22, color="#CC0000", alpha=0.25, ha="center",
                 va="center", rotation=18, zorder=10)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", bbox_inches="tight")
        print(f"Wrote: {OUT}.{ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
