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

    F5 fix 2026-05-14 (codex /stress v6): this is now a THIN ADAPTER over the
    CANONICAL DL estimator `preregistration_decision_test.dersimonian_laird_meta`.
    Previously this module carried a parallel hand-rolled DL implementation —
    two implementations of the same estimator that could silently drift. The
    canonical version (used by the paper §1 framing-rule decision script) is now
    the single source of truth for τ² / Q / I² / RE weights / pooled CI.

    This adapter preserves THIS module's API contract:
      - input: (thetas, ses) — per-cell SEs (canonical takes variances; converted here)
      - output keys: k / theta_fe / se_fe / theta_re / se_re / ci_lo / ci_hi /
        z_re / p_re_one_sided / Q / df / p_Q / tau2 / I2
      - p_re_one_sided is ONE-SIDED (canonical reports two-sided; recomputed here)

    Returns None if no valid data.
    """
    paired = [(t, s) for t, s in zip(thetas, ses) if t is not None and s is not None and s > 0]
    if len(paired) == 0:
        return None
    thetas_list = [t for t, _ in paired]
    variances = [s * s for _, s in paired]  # SE → variance for canonical API
    k = len(paired)

    # Import canonical DL estimator (sibling script in scripts/analysis/).
    # Import is function-local to avoid module-load circular-import + to keep
    # this adapter self-contained.
    import sys as _sys
    from pathlib import Path as _Path
    _this_dir = str(_Path(__file__).resolve().parent)
    if _this_dir not in _sys.path:
        _sys.path.insert(0, _this_dir)
    from preregistration_decision_test import dersimonian_laird_meta as _canonical_dl

    m = _canonical_dl(thetas_list, variances)
    theta_re = m["pooled_effect"]
    se_re = m["pooled_se"]
    Q = m["Q"]
    df = m["Q_df"] if m.get("Q_df") is not None else (k - 1)
    tau2 = m["tau_squared"] if m.get("tau_squared") is not None else 0.0
    I2 = m["I_squared_pct"] if m.get("I_squared_pct") is not None else 0.0
    if m.get("pooled_ci_95") and m["pooled_ci_95"][0] is not None:
        ci_lo, ci_hi = m["pooled_ci_95"]
    else:
        ci_lo, ci_hi = theta_re - 1.96 * se_re, theta_re + 1.96 * se_re

    # Fixed-effect (backward-compat keys; not in canonical return)
    var_arr = np.array(variances)
    w_i = 1.0 / np.maximum(var_arr, 1e-12)
    theta_fe = float(np.sum(w_i * np.array(thetas_list)) / np.sum(w_i))
    se_fe = float(math.sqrt(1.0 / np.sum(w_i)))

    # THIS module's p-value convention: ONE-SIDED (pooled effect > 0).
    z = theta_re / se_re if se_re > 0 else None
    if HAS_SCIPY and z is not None:
        p_re = float(1 - sp_stats.norm.cdf(z))
    elif z is not None:
        p_re = 0.5 * (1.0 - math.erf(z / math.sqrt(2.0)))  # erf fallback (matches canonical)
    else:
        p_re = None
    if HAS_SCIPY and Q is not None and df > 0:
        p_Q = float(1 - sp_stats.chi2.cdf(Q, df))
    else:
        p_Q = None

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


def hartung_knapp_sidik_jonkman(thetas: list, ses: list, tau2: float
                                 ) -> Optional[dict]:
    """Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment to DL random-effects pooling.

    /stress A1.19 P1-3-B* (2026-05-17, codex Mode B OOB): DL-Wald is anti-conservative
    at k=6 (per IntHout et al. 2014 / Veroniki et al. 2016 — DL τ² downward-biased at
    k<10, RE Wald CI false-positive rate inflated). HKSJ replaces Wald z + 1.96 with
    t_{k-1, 0.975} on Hartung's variance estimator, giving nominal coverage at small k.

    Returns dict with theta_re (same as DL) + se_hk + ci_lo_hk + ci_hi_hk + t_stat +
    p_one_sided_hk, OR None if insufficient cells.

    Per IntHout et al. 2014 formulation:
      w*_i = 1 / (se_i² + τ²)                  -- DL random-effects weights
      θ_RE  = Σ(w*_i · θ_i) / Σw*_i             -- pooled point (same as DL)
      q_HK  = Σ(w*_i · (θ_i - θ_RE)²) / (k - 1) -- Hartung's residual-variance estimator
      se_HK = sqrt(q_HK / Σw*_i)                -- HK pooled SE
      CI    = θ_RE ± t_{k-1, 0.975} · se_HK     -- t-distribution CI

    Note: q_HK / Σw*_i = se_DL² × q_HK × Σw*_i (algebra check); equivalently,
    se_HK = se_DL · sqrt(q_HK · Σw*_i) where the multiplier ≥ 1 is the HK correction.
    """
    paired = [(t, s) for t, s in zip(thetas, ses) if t is not None and s is not None and s > 0]
    k = len(paired)
    if k < 2:
        return None
    arr_t = np.array([t for t, _ in paired])
    arr_s = np.array([s for _, s in paired])
    variances = arr_s ** 2
    w_star = 1.0 / (variances + tau2 + 1e-12)
    sum_w_star = float(np.sum(w_star))
    theta_re = float(np.sum(w_star * arr_t) / sum_w_star)
    q_hk = float(np.sum(w_star * (arr_t - theta_re) ** 2) / (k - 1))
    se_hk = float(math.sqrt(q_hk / sum_w_star))
    # t_{k-1, 0.975} via scipy if available; otherwise hard-coded table for k=2..10
    if HAS_SCIPY:
        t_crit = float(sp_stats.t.ppf(0.975, df=k - 1))
    else:
        # Approximation table (two-sided 0.975 quantile of Student-t):
        t_table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
                   6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}
        t_crit = t_table.get(k - 1, 1.96)  # large-k → Wald approx
    ci_lo_hk = theta_re - t_crit * se_hk
    ci_hi_hk = theta_re + t_crit * se_hk
    # One-sided test H1: θ > 0 via t-statistic on k-1 df
    t_stat = theta_re / se_hk if se_hk > 0 else None
    if HAS_SCIPY and t_stat is not None:
        p_one_sided_hk = float(1.0 - sp_stats.t.cdf(t_stat, df=k - 1))
    else:
        p_one_sided_hk = None
    return {
        "k": k,
        "theta_re": theta_re,
        "se_hk": se_hk,
        "ci_lo_hk": ci_lo_hk,
        "ci_hi_hk": ci_hi_hk,
        "t_stat": t_stat,
        "t_crit": t_crit,
        "df": k - 1,
        "p_one_sided_hk": p_one_sided_hk,
    }


# F08 audit fix 2026-05-09: B8 preregistration lock requires N_common >= 10
# per cell for inclusion in random-effects meta. Cells below floor are
# excluded with reason logged. See `preregistration.md §4` row "Heterogeneity
# (random-effects, Q, I², τ²) pre-spec".
MIN_N_COMMON_FOR_META = 10


def load_per_cell_data(arm_code: str) -> tuple[list[dict], list[dict]]:
    """Per-cell point + SE for a given arm, with B8 N>=10 floor enforced.

    Returns (included, excluded) cell-row dicts. SE_i derived from bootstrap
    CI: SE = (CI_hi - CI_lo) / (2 * 1.96).
    """
    if not CSV_IN.exists():
        raise SystemExit(f"missing {CSV_IN}; run aggregate_phantom_lift.py first")
    included, excluded = [], []
    with CSV_IN.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            theta = _f(r.get(f"lift_{arm_code}_pp"))
            ci_lo = _f(r.get(f"lift_{arm_code}_ci95_lo_pp"))
            ci_hi = _f(r.get(f"lift_{arm_code}_ci95_hi_pp"))
            n_common = _f(r.get("n_common"))
            cell_label = f"{r.get('baseline','?')} {r.get('site','?')}"
            if theta is None or ci_lo is None or ci_hi is None:
                continue
            se = (ci_hi - ci_lo) / (2 * 1.96)
            if se <= 0:
                continue
            row = {
                "baseline": r["baseline"],
                "site": r["site"],
                "n_common": int(n_common) if n_common is not None else None,
                "theta": theta,
                "se": se,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
            if n_common is not None and n_common < MIN_N_COMMON_FOR_META:
                row["exclude_reason"] = f"N_common={int(n_common)} < {MIN_N_COMMON_FOR_META} (B8 lock)"
                excluded.append(row)
                continue
            included.append(row)
    return included, excluded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta_rows = []
    arm_per_cell: dict = {}
    arm_excluded: dict = {}  # F08: track B8 N>=10 floor exclusions
    for code, label, family in ARMS:
        cells, excluded = load_per_cell_data(code)
        arm_per_cell[code] = cells
        arm_excluded[code] = excluded
        if excluded:
            for ex in excluded:
                print(
                    f"  [B8 floor] arm={code} excluded "
                    f"{ex['baseline']} {ex['site']}: {ex['exclude_reason']}"
                )
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
            "excluded_b8": "; ".join(
                f"{c['baseline']} {c['site']} (N={c['n_common']})"
                for c in excluded
            ) or "none",
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
        # B-182 (/stress A1.4b-i codex B3, P1): clarify family scope + gating
        # status. Pre-fix prose said "Pre-registered family gating" with Holm
        # over the SECONDARY pooled tests (m=3), but `preregistration.md:292-320`
        # declares PRIMARY family m=1 (single FE superiority test) — the RE
        # meta in THIS script is appendix sensitivity-only, NOT a paper gate.
        # H4 exploratory family is `m = 2 × N_cells` per-cell drop-one, distinct
        # from this 3-arm pooled Holm. Make the family labels explicit so
        # methods reviewer can answer "Holm over what family?".
        "**Family scoping clarification (B-182)**: this table reports RE meta-pooled",
        "estimates per arm. The PRIMARY paper gate is the FE one-sided superiority",
        "test (H0: θ_FE ≤ +1.0pp) computed separately by `phase1_prereg_gate.csv`",
        "(currently MISSING — see issue tracker for B-185 follow-up). The Holm-corrected",
        "p_re_holm column below is family-corrected WITHIN this 3-arm SECONDARY family",
        "(`family_scope=APPENDIX_RE_SENSITIVITY_m3`, NOT `H1_PRIMARY`). Treat",
        "`sig` symbols as appendix sensitivity signal, not paper gate verdict.",
        "",
        "## Pooled estimates per arm",
        "",
        "| Family | family_scope | gating_status | Arm | k cells | Random-effect pp | 95% CI | SE | z | p (1-sided) | I² | τ² | Q | df | p_Q | p_re_holm (within family) | sig |",
        "|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
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

    # B-182: family_scope + gating_status are derived from the family label
    # already attached to each meta_row. PRIMARY family in this RE script is
    # appendix sensitivity, not a paper gate — the paper PRIMARY (FE
    # superiority) lives in `phase1_prereg_gate.csv` (currently MISSING per
    # B-185 follow-up issue).
    family_scope_map = {
        "PRIMARY":   ("APPENDIX_RE_SENSITIVITY_m1", "appendix-only"),
        "SECONDARY": ("APPENDIX_RE_SENSITIVITY_m3", "appendix-only"),
        "TERTIARY":  ("APPENDIX_RE_SENSITIVITY_m2", "exploratory"),
    }
    for r in meta_rows:
        sig = "✅" if (r.get("p_re_holm") is not None and r["p_re_holm"] < 0.05) else "❌"
        i2_lab = i_squared_label(r["I2"]) if r["k_cells"] > 1 else "n/a (k=1)"
        p_re_holm_str = _fmt(r.get("p_re_holm")) if r["k_cells"] >= 1 else "—"
        f_scope, gating = family_scope_map.get(
            r["family"], (f"UNKNOWN_FAMILY_{r['family']}", "—")
        )
        # B-182: stamp family_scope + gating_status onto the row so downstream
        # CSV / JSON consumers also see the disambiguation.
        r["family_scope"] = f_scope
        r["gating_status"] = gating
        lines.append(
            f"| {r['family']} | {f_scope} | {gating} | {r['arm_label']} | {r['k_cells']} | "
            f"+{r['theta_re']:.2f}pp | "
            f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | "
            f"{_fmt(r['se_re'], '.3f')} | "
            f"{_fmt(r['z_re'], '.2f')} | "
            f"{_fmt(r['p_re_one_sided'])} | "
            f"{r['I2']:.1f}% ({i2_lab}) | "
            f"{_fmt(r['tau2'], '.3f')} | "
            f"{_fmt(r['Q'], '.2f')} | {r['df']} | "
            f"{_fmt(r['p_Q'])} | {p_re_holm_str} | {sig} |"
        )

    # /stress A1.19 P1-3-B* (2026-05-17): Hartung-Knapp-Sidik-Jonkman row alongside
    # the DL-Wald row above. At k=6 (Phase 1a) the DL-Wald CI is anti-conservative
    # per IntHout et al. 2014; HKSJ uses t_{k-1, 0.975} on Hartung's residual-variance
    # estimator and gives nominal coverage. Decision-grade RE inference at k≤10 should
    # cite HKSJ; DL-Wald is retained above for backward-compat with archive prose.
    lines += [
        "",
        "## HKSJ adjustment (decision-grade RE inference at k≤10)",
        "",
        "Hartung-Knapp-Sidik-Jonkman (IntHout et al. 2014, BMC Med Res Method 14:25) ",
        "replaces Wald z + 1.96 with t_{k−1, 0.975} on Hartung's residual-variance ",
        "estimator. At Phase 1a k=6 cells, DL-Wald above is anti-conservative ",
        "(false-positive rate inflated; Veroniki et al. 2016). Cite HKSJ row for ",
        "decision-grade RE inference; DL-Wald row is appendix backward-compat only.",
        "",
        "| Arm | k | θ_RE (pp) | 95% CI (HK) | SE_HK | t-stat (df=k-1) | p (HK, 1-sided) | sig (α=.05) |",
        "|---|---:|---:|---|---:|---:|---:|:---:|",
    ]
    for code, _arm_label, _family in ARMS:
        cells = arm_per_cell.get(code, [])
        if not cells:
            continue
        meta = next((r for r in meta_rows if r["arm_code"] == code), None)
        if meta is None:
            continue
        hk = hartung_knapp_sidik_jonkman(
            [c["theta"] for c in cells],
            [c["se"] for c in cells],
            meta["tau2"],
        )
        if hk is None:
            continue
        sig_hk = "✅" if (hk["p_one_sided_hk"] is not None and hk["p_one_sided_hk"] < 0.05) else "❌"
        lines.append(
            f"| {meta['arm_label']} | {hk['k']} | "
            f"+{hk['theta_re']:.2f} | "
            f"[{hk['ci_lo_hk']:.2f}, {hk['ci_hi_hk']:.2f}] | "
            f"{hk['se_hk']:.3f} | "
            f"{_fmt(hk['t_stat'], '.2f')} | "
            f"{_fmt(hk['p_one_sided_hk'])} | {sig_hk} |"
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
        "- **DL-Wald is legacy descriptive** (/stress A1.19 P1-3-B*, 2026-05-17): ",
        "  Wald z+1.96 CI on DL pooled estimate is anti-conservative at k≤10 ",
        "  (IntHout et al. 2014; Veroniki et al. 2016 τ² downward bias at k<10). ",
        "  **Cite HKSJ row above for decision-grade RE inference**; DL-Wald row is ",
        "  retained for backward-compat with archive prose.",
        "- **Heterogeneity caveat**: with k < 5 cells the τ² estimate has wide",
        "  uncertainty; I² benchmarks should be read as suggestive. Re-evaluate",
        "  after Phase 1a 6-cell rerun completes (k = 6 per arm expected).",
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
