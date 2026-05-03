#!/usr/bin/env python3
"""[Outcome supporting] Cross-cell meta-analysis — random-effect pooled drop-one
+ heterogeneity (I², τ², Cochran's Q).

Reads `results/phantom_paper/phantom_lift.csv` (T0a-augmented). For each phantom
arm and oracle comparison, pools per-cell estimates using DerSimonian-Laird
(1986) random-effect model. Within-cell SE derived from bootstrap 95% CI:

    SE_i ≈ (CI_hi - CI_lo) / (2 × 1.96)

(Standard normal approximation for symmetric bootstrap CIs; valid when N per
cell is moderate, which holds for N=210-234.)

Outputs:
- `results/phantom_paper/meta_phantom_lift.csv` (per-arm meta-row)
- `results/phantom_paper/meta_phantom_lift.md`  (paper-ready table)

T0c of `docs/reference/EVIDENCE_LAYER_AUDIT.md` action queue.

Why random-effect (RE) over fixed-effect (FE):
- FE assumes single true effect across cells (only sampling variability).
- RE allows true effect heterogeneity across cells (site / model / capability).
- Phantom-SoM's "site-modulated + capability-modulated" framing (paper §7) is
  itself an RE assumption — assuming FE would contradict the paper hook.
- Paired with I² heterogeneity statistic, RE quantifies how much variation is
  between-cell (true differences) vs within-cell (sampling).

Heterogeneity benchmarks (Higgins & Thompson 2002):
  I² < 25% — low heterogeneity (cells consistent)
  25-50%  — moderate
  50-75%  — substantial
  > 75%   — considerable (strong cell-specific effects)
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

REPO = Path(__file__).resolve().parents[2]
CSV_IN = REPO / "results/phantom_paper/phantom_lift.csv"
DEFAULT_OUT = REPO / "results/phantom_paper/meta_phantom_lift.csv"

# Arms to meta-pool: (csv prefix, display label, family)
ARMS = [
    ("5_vs_3",        "3→5-mode oracle lift",  "PRIMARY"),
    ("4pdom_vs_3",    "P-text drop-in",        "SECONDARY"),
    ("4psom_vs_3",    "P-SoM drop-in",         "SECONDARY"),
    ("4pprompt_vs_3", "P-prompt drop-in",      "SECONDARY"),
    ("6_vs_3",        "6-mode oracle lift",    "TERTIARY"),
    ("6_vs_5",        "P-prompt incremental",  "TERTIARY"),
]


def _f(x):
    if x is None or x == "" or x == "None":
        return None
    return float(x)


def derslong_laird_meta(thetas: list, ses: list) -> Optional[dict]:
    """DerSimonian-Laird random-effect meta-analysis.

    Args:
        thetas: per-cell point estimates (pp scale)
        ses: per-cell SEs (matched to thetas)

    Returns dict with k / theta_fe / se_fe / theta_re / se_re / ci_lo / ci_hi /
    Q / df / p_Q / tau2 / I2, or None if no data.
    """
    paired = [(t, s) for t, s in zip(thetas, ses) if t is not None and s is not None and s > 0]
    if len(paired) == 0:
        return None
    thetas_arr = np.array([t for t, _ in paired])
    ses_arr = np.array([s for _, s in paired])
    k = len(paired)

    # Fixed-effect (inverse-variance weighted)
    var_i = ses_arr ** 2
    w_i = 1.0 / var_i
    theta_fe = float(np.sum(w_i * thetas_arr) / np.sum(w_i))
    se_fe = float(math.sqrt(1.0 / np.sum(w_i)))

    # Cochran's Q (heterogeneity test statistic)
    Q = float(np.sum(w_i * (thetas_arr - theta_fe) ** 2))
    df = k - 1
    if HAS_SCIPY and df > 0:
        p_Q = float(1 - sp_stats.chi2.cdf(Q, df))
    else:
        p_Q = None

    # τ² (between-study variance, DL estimator)
    if df > 0:
        sum_w = float(np.sum(w_i))
        sum_w2 = float(np.sum(w_i ** 2))
        C = sum_w - sum_w2 / sum_w
        tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
    else:
        tau2 = 0.0

    # I² (% variation due to heterogeneity, Higgins & Thompson 2002)
    if Q > 0 and df > 0:
        I2 = max(0.0, (Q - df) / Q) * 100.0
    else:
        I2 = 0.0

    # Random-effect estimate (using w*_i = 1 / (var_i + tau2))
    var_star = var_i + tau2
    w_star = 1.0 / var_star
    theta_re = float(np.sum(w_star * thetas_arr) / np.sum(w_star))
    se_re = float(math.sqrt(1.0 / np.sum(w_star)))
    ci_lo = theta_re - 1.96 * se_re
    ci_hi = theta_re + 1.96 * se_re

    # RE vs 0 z-test (single-side: pooled effect > 0)
    z = theta_re / se_re if se_re > 0 else None
    if HAS_SCIPY and z is not None:
        p_re = float(1 - sp_stats.norm.cdf(z))
    else:
        p_re = None

    return {
        "k": k,
        "theta_fe": theta_fe,
        "se_fe": se_fe,
        "theta_re": theta_re,
        "se_re": se_re,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "z_re": z,
        "p_re_one_sided": p_re,
        "Q": Q,
        "df": df,
        "p_Q": p_Q,
        "tau2": tau2,
        "I2": I2,
    }


def i_squared_label(I2: float) -> str:
    if I2 < 25:
        return "low"
    if I2 < 50:
        return "moderate"
    if I2 < 75:
        return "substantial"
    return "considerable"


def load_per_cell_data(arm_code: str) -> list[dict]:
    """Per-cell point + SE for a given arm.

    SE_i derived from bootstrap CI: SE = (CI_hi - CI_lo) / (2 * 1.96).
    """
    if not CSV_IN.exists():
        raise SystemExit(f"missing {CSV_IN}; run aggregate_phantom_lift.py first")
    rows = []
    with CSV_IN.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            theta = _f(r.get(f"lift_{arm_code}_pp"))
            ci_lo = _f(r.get(f"lift_{arm_code}_ci95_lo_pp"))
            ci_hi = _f(r.get(f"lift_{arm_code}_ci95_hi_pp"))
            if theta is None or ci_lo is None or ci_hi is None:
                continue
            se = (ci_hi - ci_lo) / (2 * 1.96)
            if se <= 0:
                continue
            rows.append({
                "baseline": r["baseline"],
                "site": r["site"],
                "theta": theta,
                "se": se,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    arm_per_cell: dict = {}
    for code, label, family in ARMS:
        cells = load_per_cell_data(code)
        arm_per_cell[code] = cells
        if not cells:
            continue
        meta = derslong_laird_meta(
            [c["theta"] for c in cells],
            [c["se"] for c in cells],
        )
        if meta is None:
            continue
        meta_rows.append({
            "arm_code": code,
            "arm_label": label,
            "family": family,
            "k_cells": meta["k"],
            "cells": "; ".join(f"{c['baseline']} {c['site']}" for c in cells),
            **{k: round(v, 6) if isinstance(v, float) else v
               for k, v in meta.items() if k != "k"},
        })

    # CSV
    with out.open("w", newline="") as f:
        if meta_rows:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            w.writeheader()
            w.writerows(meta_rows)
    print(f"wrote {out} ({len(meta_rows)} arms)")

    # Markdown
    md = out.with_suffix(".md")
    n_arms = len(meta_rows)
    n_primary = sum(1 for r in meta_rows if r["family"] == "PRIMARY")
    n_secondary = sum(1 for r in meta_rows if r["family"] == "SECONDARY")
    n_tertiary = sum(1 for r in meta_rows if r["family"] == "TERTIARY")
    lines = [
        "# Phantom routing lift — cross-cell meta-analysis (random-effect pooled)",
        "",
        "DerSimonian-Laird (1986) random-effect meta-analysis pools per-cell",
        "drop-one and oracle-lift estimates across all available cells. Within-cell",
        "SE derived from bootstrap 95% CI as `(CI_hi - CI_lo) / (2 × 1.96)`.",
        "",
        "Heterogeneity statistics:",
        "- **I²** — % variation due to between-cell heterogeneity (vs sampling).",
        "  Benchmarks: <25% low / 25-50% moderate / 50-75% substantial / >75% considerable.",
        "- **τ²** — between-cell variance (DL estimator); 0 = no heterogeneity.",
        "- **Cochran's Q** — homogeneity test; small p_Q rejects assumption that",
        "  cells share single true effect.",
        "",
        f"Cells included per arm — see `cells` col. Arms: {n_arms} pooled "
        f"(PRIMARY={n_primary}, SECONDARY={n_secondary}, TERTIARY={n_tertiary}).",
        "",
        "## Pooled estimates per arm",
        "",
        "| Family | Arm | k cells | Random-effect pp | 95% CI | SE | z | p (1-sided) | I² | τ² | Q | df | p_Q | sig (Holm-corrected family) |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    # Apply Holm-Bonferroni within each family for the meta-pooled p-value
    # (1-sided z test on RE estimate)
    by_family: dict = {}
    for r in meta_rows:
        by_family.setdefault(r["family"], []).append(r)
    for family, family_rows in by_family.items():
        ps = [r.get("p_re_one_sided") for r in family_rows]
        # Holm-Bonferroni step-down
        indexed = [(i, p) for i, p in enumerate(ps) if p is not None]
        indexed.sort(key=lambda x: x[1])
        m = len(indexed)
        adj = [None] * len(ps)
        prev = 0.0
        for k, (i, p) in enumerate(indexed):
            a = min(1.0, max(prev, p * (m - k)))
            adj[i] = a
            prev = a
        for r, a in zip(family_rows, adj):
            r["p_re_holm"] = round(a, 6) if a is not None else None

    def _fmt(v, spec=".4f"):
        if v is None:
            return "—"
        return f"{v:{spec}}"

    for r in meta_rows:
        sig = "✅" if (r.get("p_re_holm") is not None and r["p_re_holm"] < 0.05) else "❌"
        i2_lab = i_squared_label(r["I2"]) if r["k_cells"] > 1 else "n/a (k=1)"
        p_re_holm_str = _fmt(r.get("p_re_holm")) if r["k_cells"] >= 1 else "—"
        lines.append(
            f"| {r['family']} | {r['arm_label']} | {r['k_cells']} | "
            f"+{r['theta_re']:.2f}pp | "
            f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | "
            f"{_fmt(r['se_re'], '.3f')} | "
            f"{_fmt(r['z_re'], '.2f')} | "
            f"{_fmt(r['p_re_one_sided'])} → Holm {p_re_holm_str} | "
            f"{r['I2']:.1f}% ({i2_lab}) | "
            f"{_fmt(r['tau2'], '.3f')} | "
            f"{_fmt(r['Q'], '.2f')} | {r['df']} | "
            f"{_fmt(r['p_Q'])} | {sig} |"
        )

    lines += [
        "",
        "## Per-cell forest data (input to meta-pool)",
        "",
        "| Arm | Cell | Lift (pp) | 95% CI | SE | Weight (RE) |",
        "|---|---|---:|---|---:|---:|",
    ]
    # Compute RE weights for transparency
    for code, label, _ in ARMS:
        cells = arm_per_cell.get(code, [])
        if not cells:
            continue
        meta = next((r for r in meta_rows if r["arm_code"] == code), None)
        if meta is None:
            continue
        tau2 = meta["tau2"]
        for c in cells:
            w_star = 1.0 / (c["se"] ** 2 + tau2)
            sum_w_star = sum(1.0 / (cc["se"] ** 2 + tau2) for cc in cells)
            weight_pct = 100.0 * w_star / sum_w_star
            lines.append(
                f"| {label} | {c['baseline']} {c['site']} | "
                f"+{c['theta']:.2f}pp | "
                f"[{c['ci_lo']:.2f}, {c['ci_hi']:.2f}] | "
                f"{c['se']:.3f} | {weight_pct:.1f}% |"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- **Pre-registered family gating**: PRIMARY arm gated by Holm within m=1",
        "  test (no correction needed). SECONDARY arms gated by Holm within m=3",
        "  pooled tests. TERTIARY uncorrected (exploratory).",
        "- **Heterogeneity caveat**: with k < 5 cells the τ² estimate has wide",
        "  uncertainty; I² benchmarks should be read as suggestive. Re-evaluate",
        "  after 14-cell rerun completes (k ≈ 8 per arm expected).",
        "- **Random-effect vs fixed-effect**: when I² < 25%, FE and RE estimates",
        "  converge; large I² (> 50%) means cell-specific effects matter and only",
        "  RE pooled estimate is meaningful for paper claim.",
        "- **One-sided p**: H1 = pooled effect > 0 (phantom adds tasks). Two-sided",
        "  not used because the directional hypothesis is asymmetric.",
    ]
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
