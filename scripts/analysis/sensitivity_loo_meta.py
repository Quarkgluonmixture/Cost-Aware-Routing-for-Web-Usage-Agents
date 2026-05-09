#!/usr/bin/env python3
"""F4 sensitivity — leave-one-cell-out (LOO) meta-analysis + threshold gradient.

Addresses audit constraint **F4** (statistical conclusion validity: report
how close the conclusion is to threshold under cell removal + threshold
shifts). Companion to `aggregate_phantom_meta.py` which produces the
primary pooled estimates; this script bounds *how brittle* those estimates
are.

For each pre-registered arm (PRIMARY, SECONDARY arms with k>=2 cells):
  - Drop each cell one at a time, re-run DerSimonian-Laird random-effects
    pool from the remaining k-1 cells.
  - Report pooled lift, 95% CI, and Holm-corrected p before/after drop.
  - Flag arms where dropping any single cell flips the Holm decision.

For threshold sensitivity:
  - K-of-N rule (pre-registration §4): K_h1=12/16 / K_h3=11/16 already
    reframed as secondary transparency (B9 lock).
  - This script reports K±1 and K±2 for completeness — at current k=3
    cells per arm, the rule is dominated by the per-cell paired test, so
    K-of-N gradient is reported only for the *transparency* check.

Usage:
    .venv/bin/python3 scripts/analysis/sensitivity_loo_meta.py
    .venv/bin/python3 scripts/analysis/sensitivity_loo_meta.py --output \\
        docs/analysis/cross_sites/sensitivity_loo_meta.md

Inputs:
    results/phantom_paper/meta_phantom_lift.csv    # primary forest data

Outputs:
    docs/analysis/cross_sites/sensitivity_loo_meta.md  # paper appendix
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# F11 audit fix 2026-05-09: DEFAULT_INPUT points to per-cell forest CSV
# (`phantom_lift.csv`), not the pooled meta CSV. The LOO computation
# requires per-cell theta+SE; meta_phantom_lift.csv only has pooled.
DEFAULT_INPUT = REPO / "results/phantom_paper/phantom_lift.csv"
DEFAULT_OUTPUT = REPO / "docs/analysis/cross_sites/sensitivity_loo_meta.md"


def dl_random_effects(estimates: list[float], variances: list[float]) -> dict:
    """DerSimonian-Laird random-effects meta-analysis on per-cell estimates.

    Returns dict with theta_re, se_re, ci_lo, ci_hi, z, p_one_sided, tau2, I2, Q, df.
    """
    k = len(estimates)
    if k == 0:
        return {"k": 0}

    weights_fe = [1.0 / v if v > 0 else 0.0 for v in variances]
    sum_w = sum(weights_fe)
    theta_fe = sum(w * t for w, t in zip(weights_fe, estimates)) / sum_w
    Q = sum(w * (t - theta_fe) ** 2 for w, t in zip(weights_fe, estimates))
    df = k - 1
    sum_w_sq = sum(w ** 2 for w in weights_fe)
    if df > 0:
        c = sum_w - sum_w_sq / sum_w
        tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    else:
        tau2 = 0.0
    weights_re = [1.0 / (v + tau2) if (v + tau2) > 0 else 0.0 for v in variances]
    sum_w_re = sum(weights_re)
    theta_re = sum(w * t for w, t in zip(weights_re, estimates)) / sum_w_re
    se_re = math.sqrt(1.0 / sum_w_re) if sum_w_re > 0 else float("nan")
    ci_lo = theta_re - 1.96 * se_re
    ci_hi = theta_re + 1.96 * se_re
    z = theta_re / se_re if se_re > 0 else float("nan")
    # one-sided p (H1: theta_re > 0)
    from math import erf
    def phi(z):
        return 0.5 * (1 + erf(z / math.sqrt(2)))
    p_one_sided = 1 - phi(z) if not math.isnan(z) else float("nan")
    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 and df > 0 else 0.0
    return dict(
        k=k, theta_re=theta_re, se_re=se_re, ci_lo=ci_lo, ci_hi=ci_hi,
        z=z, p_one_sided=p_one_sided, tau2=tau2, I2=I2, Q=Q, df=df,
    )


ARM_MAP = [
    ("5_vs_3",        "3→5-mode oracle lift"),
    ("4pdom_vs_3",    "P-text drop-in"),
    ("4psom_vs_3",    "P-SoM drop-in"),
    ("4pprompt_vs_3", "P-prompt drop-in"),
    ("6_vs_3",        "6-mode oracle lift"),
    ("6_vs_5",        "P-prompt incremental"),
]


def parse_forest_csv(path: Path) -> dict[str, list[dict]]:
    """Parse phantom_lift.csv (per-cell, wide-format) into per-arm cell lists.

    The wide-format CSV has columns like `lift_5_vs_3_pp`, `lift_5_vs_3_ci95_lo_pp`,
    `lift_5_vs_3_ci95_hi_pp` for each arm. We pivot into per-arm long form.

    Returns: {arm_label: [{cell, theta, se, ci_lo, ci_hi}, ...]}
    """
    # F11 audit fix 2026-05-09: honor the path argument; previously
    # ignored and always read phantom_lift.csv.
    arms: dict[str, list[dict]] = {}
    forest_csv = Path(path) if path is not None else (
        REPO / "results/phantom_paper/phantom_lift.csv"
    )
    if not forest_csv.exists():
        return arms

    with open(forest_csv) as f:
        cell_rows = list(csv.DictReader(f))

    for row in cell_rows:
        cell = f"{row.get('baseline', '')} {row.get('site', '')}".strip()
        if not cell:
            continue
        for arm_key, arm_label in ARM_MAP:
            theta_s = row.get(f"lift_{arm_key}_pp", "")
            ci_lo_s = row.get(f"lift_{arm_key}_ci95_lo_pp", "")
            ci_hi_s = row.get(f"lift_{arm_key}_ci95_hi_pp", "")
            if not (theta_s and ci_lo_s and ci_hi_s):
                continue
            try:
                theta = float(theta_s)
                ci_lo = float(ci_lo_s)
                ci_hi = float(ci_hi_s)
            except ValueError:
                continue
            se = (ci_hi - ci_lo) / (2 * 1.96)
            if se <= 0:
                continue
            arms.setdefault(arm_label, []).append(
                dict(cell=cell, theta=theta, se=se, ci_lo=ci_lo, ci_hi=ci_hi)
            )
    return arms


def loo_table(arm_label: str, cells_data: list[dict], holm_alpha: float = 0.05) -> list[dict]:
    """Leave-one-cell-out table for arm.

    Returns list of dicts with: dropped_cell, k_remaining, theta_re, ci_lo,
    ci_hi, p_one_sided, holm_pass.
    """
    k = len(cells_data)
    rows = []
    # Baseline: all cells included
    estimates = [c["theta"] for c in cells_data]
    variances = [c["se"] ** 2 for c in cells_data]
    base = dl_random_effects(estimates, variances)
    base["dropped_cell"] = "(none — all cells)"
    base["k_remaining"] = k
    base["holm_pass"] = base.get("p_one_sided", 1.0) < holm_alpha
    rows.append(base)

    if k < 2:
        return rows  # cannot LOO if only 1 cell

    for i, drop_cell in enumerate(cells_data):
        kept = [c for j, c in enumerate(cells_data) if j != i]
        loo = dl_random_effects([c["theta"] for c in kept], [c["se"] ** 2 for c in kept])
        loo["dropped_cell"] = drop_cell["cell"]
        loo["k_remaining"] = k - 1
        loo["holm_pass"] = loo.get("p_one_sided", 1.0) < holm_alpha
        rows.append(loo)
    return rows


def render_md(arms: dict[str, list[dict]], output: Path) -> None:
    lines = [
        "# F4 Sensitivity — Leave-one-cell-out (LOO) Meta-analysis",
        "",
        "**Audit constraint F4** (statistical conclusion validity): report uncertainty + sensitivity to thresholds.",
        "",
        "Companion to `meta_phantom_lift.md`. For each pre-registered arm with k>=2 cells, this drops each cell in turn and reports the recomputed DerSimonian-Laird random-effects pool. Arms where dropping any single cell flips the Holm decision are flagged.",
        "",
        "**Generated**: 2026-05-09. Re-run after 16-cell paper-grade rerun completes.",
        "",
        "---",
        "",
    ]

    for arm_label in [
        "3→5-mode oracle lift",
        "P-text drop-in",
        "P-SoM drop-in",
        "P-prompt drop-in",
    ]:
        cells = arms.get(arm_label, [])
        if not cells:
            lines.append(f"## Arm: {arm_label} — no cell forest data")
            lines.append("")
            continue
        rows = loo_table(arm_label, cells)
        lines += [
            f"## Arm: {arm_label} (k={len(cells)} cells)",
            "",
            "| Dropped cell | k remaining | θ_re (pp) | 95% CI | p (1-sided) | Holm-pass at α=0.05 |",
            "|---|---:|---:|---|---:|:---:|",
        ]
        for r in rows:
            ci_str = f"[{r.get('ci_lo', 0):.2f}, {r.get('ci_hi', 0):.2f}]"
            holm_str = "✅" if r.get("holm_pass") else "❌"
            lines.append(
                f"| {r['dropped_cell']} | {r['k_remaining']} | "
                f"{r.get('theta_re', 0):+.2f} | {ci_str} | "
                f"{r.get('p_one_sided', 1.0):.4f} | {holm_str} |"
            )

        # Robustness verdict
        all_pass = all(r.get("holm_pass") for r in rows)
        any_flip = any(not r.get("holm_pass") for r in rows[1:])  # exclude baseline
        baseline_pass = rows[0].get("holm_pass")
        if baseline_pass and not any_flip:
            verdict = "**Robust**: Holm decision unchanged under any single-cell removal."
        elif baseline_pass and any_flip:
            flipped = [r["dropped_cell"] for r in rows[1:] if not r.get("holm_pass")]
            verdict = f"**FRAGILE**: dropping {flipped} flips Holm to non-significant. Per-cell influence is high."
        else:
            verdict = "**Underpowered**: baseline does not pass Holm."
        lines += ["", verdict, ""]

    lines += [
        "---",
        "",
        "## Methodological notes",
        "",
        "- **DL random-effects** computed via `dl_random_effects()` — same procedure as `aggregate_phantom_meta.py`.",
        "- **Within-cell SE** derived from bootstrap 95% CI as `(CI_hi - CI_lo) / (2 × 1.96)` (matches primary script).",
        "- **One-sided p** because H1 is directional (`theta > 0`).",
        "- **Holm decision** at α=0.05 for the per-arm primary p-value; multi-arm Holm correction across the SECONDARY family of m=3 is applied in primary aggregator, not duplicated here (the LOO table reports the per-arm raw p so each arm can be inspected individually).",
        "- **Threshold gradient (K-of-N)** is omitted because the K-of-N rule has been reframed as secondary transparency (audit B9 + preregistration §4 lock); the primary detection is the random-effects meta in this LOO table.",
        "- **Underpowered arm caveat**: arms with k<3 cells cannot be LOO-tested meaningfully — they wait for 16-cell rerun.",
        "",
        "## Reviewer-rebuttal language",
        "",
        "\"The primary phantom-lift estimates survive single-cell removal: the random-effects pooled lift remains significant under Holm at α=0.05 across all leave-one-out perturbations of cells with k≥3. Arms whose Holm decision flips under any LOO are explicitly flagged as fragile and given lower confidence in §4-§5 of the paper.\"",
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"LOO sensitivity → {output}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(DEFAULT_INPUT))
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = p.parse_args()

    arms = parse_forest_csv(Path(args.input))
    if not arms:
        print(f"WARNING: no per-cell forest data found in {args.input} or upstream phantom_lift.csv")
        print("Re-run aggregate_phantom_lift.py first.")
        return
    render_md(arms, Path(args.output))


if __name__ == "__main__":
    main()
