r"""Preregistration decision test — Phase 1a 36-condition / 6-cell H1 / H3 / H2 evaluation.

⚠️ A1.21 RESCOPE 2026-05-17 (P0-2 + P0-3 + P0-9 + P1-7/8/9/10/11 batch, B-513~B-521):
   This script is now a **synthetic test fixture + appendix-sensitivity producer**, NOT
   the canonical paper §1 gate substrate. The canonical full prereg decision (H1+H2(a)
   +H3 axes + I² cap + R1-R5 framing) is now `aggregate_phase1_full_prereg_decision.py`
   (B-515, A1.21 P0-4). This script retains the synthetic fixture scenarios for CI
   smoke testing and emits the DL random-effects estimand into an `appendix_dl_sensitivity`
   block as transparency reporting only.

   Changes from prior version (codex Mode B A1.21 catches):
   - **P0-2**: `evaluate_h1` DL-meta gate path retired — primary verdict now FE superiority
     only (matches prereg §1 lock L68-86). DL meta kept as `appendix_dl_sensitivity`.
   - **P0-3**: `_effective_gate_pass` heterogeneity-rescue branch retired — high I² caps
     framing power R1/R2 → R3 (cap-only, prereg §2 L323) but does NOT rescue failed
     FE-H1 (prior code violated prereg L340-342 by rescuing R5 → R3 via per-cell rule).
   - **P0-6**: `_paired_bootstrap` adds `p_percentile_two_sided` for method coherence
     with percentile CI (was: percentile CI + normal-approx p-value mixed method).
   - **P0-9**: `evaluate_h2_cost` rewrite per-task median ratio (paired) — prior code
     computed median-of-marginals (`median(P-SoM costs) / median(DOM costs)`) which is
     wrong estimand for paper §1 line 9 per-task framing. Prereg §2 H2(a) prose locked
     2026-05-17 to "per-task median ratio".
   - **P1-7**: H2(a) 3-state per cell (within / falsified / cannot_evaluate) — framing
     rule uses `n_falsified > 0` (NOT `pass_count == total` which conflated missing data
     with falsification).
   - **P1-8**: `dersimonian_laird_meta` k<2 raises `InsufficientCellsError` (silent 0.0
     fallback retired — gave wrong R5 diagnosis when root cause was data missing).
   - **P1-9**: `--data-seed` + `--bootstrap-seed` split (was single `--seed` double-binding).
   - **P1-10**: synthetic generator B2 capability adjustment fixed (was: only B1 down-scaled
     so B2 = B0 by accident, violating advisor §138 B2 ≈ B1 matched-capability assumption).
   - **P1-11**: `_phi` scipy fallback path with overflow warning when |z| > 6.
   - **P2-2**: stale "Phase 1a 24-condition / 4-cell" output metadata fixed to current scope.

⚠️ REWRITTEN 2026-05-13 (historical):
   - PRIMARY GATE = pooled DerSimonian-Laird random-effects meta + one-sided superiority
   - K-of-N reclassified gate → transparency consistency check
   - H1 formula = P-SoM drop-one oracle ceiling lift (NOT P-SoM ≥ best single mode)
   - H3 family = axis-1 (P-text \ P-SoM) + axis-2 (P-prompt \ P-SoM), both pooled

Definitions (per preregistration.md §2 + §4):
  - cell = 1 (site, model) statistical stratification unit. Phase 1a N=6 cells:
    (cls, B0), (cls, B1), (cls, B2), (red, B0), (red, B1), (red, B2).
  - condition = 1 (site, model, mode) operational launch unit. Phase 1a N=36.
  - Drop-one per cell: oracle ceiling SR over {6 modes} − oracle ceiling SR over
    {5 modes drop P-SoM}, per task, averaged across task pool. Paired bootstrap CI.
  - Pooled meta (current impl): DerSimonian-Laird random-effects across 6 cell estimates.
    (Decision 3A specifies FE; advisor lock pending — see banner above.)
  - One-sided superiority test (PRIMARY for H1(ii)): H0: θ ≤ +δ vs H1: θ > +δ at α=0.05.
  - TOST equivalence (INFORMATIONAL secondary for H1): two one-sided tests for
    H0 |θ| ≥ δ vs H1 |θ| < δ at δ=1.0pp. Reported in JSON output but NOT gating.

PRIMARY GATES (gate paper hook framing R1-R5):
  H1(i)  pooled meta on P-SoM drop-one, Holm α=0.05 sig (m=1)
  H1(ii) pooled magnitude θ ≥ 1.0pp AND one-sided superiority test
         (H0: θ ≤ +1.0pp vs H1: θ > +1.0pp) rejected at α=0.05 (m=1)
  H3(i)  pooled meta on |P-text \ P-SoM| axis-1, Holm α=0.05 sig (m=1)
  H3(ii) pooled meta on |P-prompt \ P-SoM| axis-2, Holm α=0.05 sig (m=1)
  H2(a)  median cost(P-SoM) within ±20% of median cost(DOM) per cell (by-construction
         falsification check — see preregistration.md §2 H2(a) revision 2026-05-14)

TRANSPARENCY (NOT gating, reported alongside primary; B-120 2026-05-15 — K-ratios
retired per Decision 3A 2026-05-14 fake-precision argument, see preregistration.md
§4 K-of-N row. Default thresholds retained as descriptive benchmarks only):
  n-of-6 cells individually Holm-sig on drop-one (descriptive; "4-5/6 = strong")
  n-of-6 cells with axis-1 CI > 0 (descriptive)
  n-of-6 cells with axis-2 CI > 0 (descriptive)

Usage:
    # With actual per-task data:
    python3 scripts/analysis/preregistration_decision_test.py \\
        --per-task-csv results/phantom_paper/per_task_sr.csv \\
        --primary-gate drop_one_pooled_meta_TOST \\
        --TOST-delta-pp 1.0 \\
        --transparency-K_h1 4 --transparency-K_h3 4 \\
        --out results/phantom_paper/preregistration_test_results.json

    # Smoke test on synthetic data:
    python3 scripts/analysis/preregistration_decision_test.py --synthetic --seed 42

Input CSV schema (per-task wide format, one row per (cell_id, task_id)):
    cell_id,site,model,task_id,sr_dom,sr_som,sr_vision,sr_ptext,sr_pprompt,sr_psom,
        cost_dom,cost_psom
    cls_B0,classifieds,B0,task_0001,0.0,1.0,0.0,1.0,0.0,1.0,0.043,0.044
    ...

Each SR cell ∈ {0, 1} (binary per-task evaluator verdict, post-source-level
FP fix — B-91 patch in VWA submodule p79-patches branch f0c835b, see
preregistration.md §4 FP-filter-architecture row 2026-05-14).
Costs in any consistent unit (token-normalized $); only ratio used.

Tied to:
- preregistration.md §2 (H1/H3 hypotheses) + §4 (locked analysis choices) +
  Appendix A 2026-05-13 (codex stress audit propagation) + 2026-05-14 (Decision
  3A FE estimand + B2 baseline addition + §139.8 FP retire) + 2026-05-15 (A100
  host migration + this script B-120 rescope)
- osf_lock_manifest.md §2.2 (canonical threshold table)
- run_manifest.yaml (cell scope = 6 Phase 1a cells)
- 笔记 §132 (codex stress audit), §138/§142 (2026-05-14 advisor scope收口 + B2
  addition), §143 (this rescope batch)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("preregistration-test")

# Phase 1a canonical cells (must match preregistration.md §4 N_cells row)
# B-120 (2026-05-15, codex Mode B P0-1): k=4 → k=6 per B2 addition 2026-05-14.
# Cells = (site, model) tuples for 2 sites (cls, red) × 3 baselines (B0, B1, B2).
PHASE_1A_CELLS = [
    ("classifieds", "B0"),
    ("classifieds", "B1"),
    ("classifieds", "B2"),  # Gemma3-VL google/gemma-3-4b-it (added 2026-05-14)
    ("reddit", "B0"),
    ("reddit", "B1"),
    ("reddit", "B2"),       # Gemma3-VL google/gemma-3-4b-it (added 2026-05-14)
]
PHANTOM_MODE_KEYS = ["sr_psom", "sr_ptext", "sr_pprompt"]
BASELINE_MODE_KEYS = ["sr_dom", "sr_som", "sr_vision"]
ALL_MODE_KEYS = BASELINE_MODE_KEYS + PHANTOM_MODE_KEYS


# ---------------------------------------------------------------------------
# Per-cell drop-one + unique-count computation (paired bootstrap)
# ---------------------------------------------------------------------------

def _oracle_per_task(task_row: dict, mode_keys: list[str]) -> int:
    """Oracle ceiling for one task = 1 if ANY mode in mode_keys solved it, else 0.

    A5 fix 2026-05-13: cast via float() first to handle CSV string SR values like
    "0.0" / "1.0" (int("0.0") raises ValueError; int(float("0.0")) works).
    """
    return 1 if any(int(float(task_row[k])) >= 1 for k in mode_keys) else 0


def _drop_one_lift_per_cell(cell_tasks: list[dict], drop_mode: str = "sr_psom") -> float:
    """Drop-one oracle ceiling lift for a cell.

    Returns the mean over the cell's task pool of:
        oracle({all 6 modes}, task) − oracle({all 6 modes} \\ {drop_mode}, task)

    Result is in [0, 1] (probability units; multiply by 100 for pp).
    """
    full = ALL_MODE_KEYS
    reduced = [k for k in full if k != drop_mode]
    deltas = [_oracle_per_task(t, full) - _oracle_per_task(t, reduced) for t in cell_tasks]
    return sum(deltas) / max(1, len(deltas))


def _unique_count_per_cell(cell_tasks: list[dict], axis_mode: str, ref_mode: str = "sr_psom") -> int:
    """|axis_mode \\ ref_mode| = number of tasks where axis_mode solved but ref_mode didn't.

    Used for H3 axis-1 (axis_mode=sr_ptext) and H3 axis-2 (axis_mode=sr_pprompt).
    A5 fix 2026-05-13: float() coercion for CSV string SR values.
    """
    return sum(1 for t in cell_tasks
               if int(float(t[axis_mode])) >= 1 and int(float(t[ref_mode])) < 1)


def _paired_bootstrap(cell_tasks: list[dict], statistic_fn, n_resamples: int = 1000,
                       seed: int = 42) -> tuple[float, float, float, float, float, float]:
    """1000-resample paired task-level bootstrap.

    Returns (point_estimate, ci_lo_95, ci_hi_95, bootstrap_se,
             p_percentile_two_sided, p_percentile_one_sided_gt_zero).

    A1.21 P0-6 fix (B-517): added percentile p-values alongside percentile CI for method
    coherence. Pre-fix returned only (point, ci_lo, ci_hi, se); callers (evaluate_h1
    / evaluate_h3_axis) computed p-value via normal-approx z-score on bootstrap SE —
    mixed method (percentile CI + normal-approx p). Cross-AI 3-AI overlap finding
    (Claude P0-2 + codex F1 + gemini F5). Bootstrap distribution skew on small-N
    sparse cells (B2) → percentile CI excludes zero but normal-approx p > 0.05 →
    conflicting verdict on same H1 cell. Now: callers can use `p_percentile_*` for
    coherent percentile-based decision.

    Resamples task rows with replacement (preserves all modes' SR for that task → paired).
    """
    import random
    rng = random.Random(seed)
    point = statistic_fn(cell_tasks)
    n = len(cell_tasks)
    boot_vals = []
    for _ in range(n_resamples):
        resample = [cell_tasks[rng.randrange(n)] for _ in range(n)]
        boot_vals.append(statistic_fn(resample))
    boot_vals.sort()
    ci_lo = boot_vals[int(0.025 * n_resamples)]
    ci_hi = boot_vals[int(0.975 * n_resamples)]
    se = statistics.stdev(boot_vals) if len(boot_vals) > 1 else 0.0
    # A1.21 P0-6: percentile p-values
    # Two-sided p = 2 × min(P(boot ≤ 0), P(boot ≥ 0))
    p_lo = sum(1 for b in boot_vals if b <= 0) / n_resamples
    p_hi = sum(1 for b in boot_vals if b >= 0) / n_resamples
    p_percentile_two_sided = 2.0 * min(p_lo, p_hi)
    p_percentile_one_sided_gt_zero = p_lo  # P(boot ≤ 0) = P(true effect ≤ 0 | data)
    return point, ci_lo, ci_hi, se, p_percentile_two_sided, p_percentile_one_sided_gt_zero


# ---------------------------------------------------------------------------
# DerSimonian-Laird random-effects meta-analysis
# ---------------------------------------------------------------------------

class InsufficientCellsError(ValueError):
    """A1.21 P1-8: raised when pool requires ≥2 cells but received <2.

    Pre-fix: `dersimonian_laird_meta` silently returned `pooled_effect=0.0` when k<2 →
    `magnitude_check` FAIL → R5 "paper death" framing → user spend hours debugging
    statistical setup when root cause was `run_manifest.yaml` had 0 paper-grade entries
    (A1.21 P0-8). Fail-loud with explicit cause hint.
    """


def dersimonian_laird_meta(effects: list[float], variances: list[float]) -> dict:
    """Pool effect estimates across cells via DerSimonian-Laird random-effects.

    ⚠️ A1.21 P0-2 (B-513): DEMOTED to appendix-sensitivity reporting. Canonical paper §1
    H1 PRIMARY gate is FE inverse-variance superiority test (see
    `aggregate_phase1_full_prereg_decision.py` for the canonical full-pipeline).
    This function retained for `appendix_dl_sensitivity` transparency block, NOT for
    PRIMARY gating decision.

    Args:
        effects: per-cell effect estimates (same scale, e.g., pp or unique-count)
        variances: per-cell variance estimates (= SE^2 from bootstrap)

    Returns dict with: pooled_effect, pooled_se, pooled_ci_95, Q, I_squared, tau_squared,
                       p_value_two_sided.

    Method (Higgins & Thompson 2002; DerSimonian & Laird 1986):
      1. Fixed-effects pooled mean θ_FE = Σ(w_i × θ_i) / Σw_i where w_i = 1 / v_i
      2. Q = Σw_i × (θ_i − θ_FE)^2
      3. τ^2 = max(0, (Q − (k − 1)) / (Σw_i − Σw_i^2 / Σw_i))
      4. Random-effects weights w*_i = 1 / (v_i + τ^2)
      5. Pooled θ_RE = Σ(w*_i × θ_i) / Σw*_i; SE_RE = sqrt(1 / Σw*_i)
      6. I^2 = max(0, (Q − (k − 1)) / Q) × 100  (% heterogeneity)

    Raises:
        InsufficientCellsError: if k < 2 (A1.21 P1-8 — was silent 0.0 fallback)
    """
    k = len(effects)
    if k < 2:
        raise InsufficientCellsError(
            f"DL meta requires ≥2 cells; got {k}. "
            "Likely cause: `run_manifest.yaml` has 0 paper-grade entries (A1.21 P0-8) "
            "OR CSV has missing cells. Check `generate_per_task_sr.py` output before "
            "invoking this script. Synthetic mode (--synthetic) always emits all "
            f"{len(PHASE_1A_CELLS)} cells."
        )

    w_fe = [1.0 / max(v, 1e-12) for v in variances]
    theta_fe = sum(w * t for w, t in zip(w_fe, effects)) / sum(w_fe)
    Q = sum(w * (t - theta_fe) ** 2 for w, t in zip(w_fe, effects))
    sum_w = sum(w_fe)
    sum_w_sq = sum(w * w for w in w_fe)
    tau_sq_num = Q - (k - 1)
    tau_sq_den = sum_w - (sum_w_sq / sum_w)
    tau_sq = max(0.0, tau_sq_num / max(tau_sq_den, 1e-12))

    # Guard against degenerate variance=0 (e.g., bootstrap SE=0 when all per-task
    # values identical, occurs in r5_fail synthetic when drop-one ≡ 0 for all tasks)
    w_re = [1.0 / max(v + tau_sq, 1e-12) for v in variances]
    theta_re = sum(w * t for w, t in zip(w_re, effects)) / sum(w_re)
    se_re = math.sqrt(1.0 / max(sum(w_re), 1e-12))
    ci_lo = theta_re - 1.96 * se_re
    ci_hi = theta_re + 1.96 * se_re

    z = theta_re / max(se_re, 1e-12)
    # Two-sided p from standard normal (using error function approximation)
    p_two_sided = 2.0 * (1.0 - _phi(abs(z)))

    i_sq = max(0.0, (Q - (k - 1)) / Q) * 100.0 if Q > 0 else 0.0

    return {
        "pooled_effect": theta_re,
        "pooled_se": se_re,
        "pooled_ci_95": [ci_lo, ci_hi],
        "Q": Q,
        "Q_df": k - 1,
        "I_squared_pct": i_sq,
        "tau_squared": tau_sq,
        "p_value_two_sided": p_two_sided,
        "z_statistic": z,
        "k": k,
    }


try:
    from scipy.stats import norm as _scipy_norm
    _HAS_SCIPY = True
except ImportError:  # paper-grade env may or may not have scipy
    _HAS_SCIPY = False


def _phi(z: float) -> float:
    """Standard normal CDF.

    A1.21 P1-11 fix (B-520): scipy when available (log-space stable for |z|>6);
    erf fallback otherwise + emit warning when |z|>6 to flag numeric saturation.
    Pre-fix: erf saturates at ±1.0 → silent p=0 for |z|>6, doesn't distinguish
    z=6 (p ≈ 1e-9) from z=10 (p ≈ 7.6e-24) — advisor-grade reproducibility hygiene.
    """
    if _HAS_SCIPY:
        return float(_scipy_norm.cdf(z))
    if abs(z) > 6:
        import warnings as _w
        _w.warn(
            f"_phi(z={z:.3f}): erf saturates at |z|>6 (returns silent ±1.0); "
            "install scipy for log-space stable evaluation. Reported p-value "
            "may underflow to 0.0; treat as 'p < 1e-9' only.",
            RuntimeWarning, stacklevel=2,
        )
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------------------
# TOST equivalence test
# ---------------------------------------------------------------------------

def superiority_test(pooled_effect: float, pooled_se: float, threshold: float,
                      alpha: float = 0.05) -> dict:
    """One-sided superiority test: H0: θ ≤ threshold vs H1: θ > threshold.

    Used for H1(ii) per prereg 2026-05-13 wording revision: "effect is significantly
    ABOVE the +threshold substantive-effect floor". Reject H0 when pooled effect is
    significantly larger than threshold (z = (θ̂ - threshold)/SE > z_α).

    Args:
        pooled_effect: pooled effect estimate (same units as threshold)
        pooled_se: pooled SE
        threshold: substantive-effect floor (positive; e.g., 1.0pp)
        alpha: one-sided significance level (default 0.05)

    Returns dict with: z, p_one_sided, threshold, decision.

    Note: This replaces prior TOST-rejection logic which had ambiguous semantic
    direction ("TOST equivalence rejected" could mean either equivalence-demonstrated
    OR equivalence-not-demonstrated). One-sided superiority is the unambiguous test
    for "effect substantively exceeds threshold".
    """
    z = (pooled_effect - threshold) / max(pooled_se, 1e-12)
    p_one_sided = 1.0 - _phi(z)
    return {
        "threshold": threshold,
        "alpha": alpha,
        "pooled_effect": pooled_effect,
        "pooled_se": pooled_se,
        "z_statistic": z,
        "p_one_sided": p_one_sided,
        "decision": "reject_H0_substantively_above_threshold" if p_one_sided < alpha else "fail_reject",
    }


def tost_equivalence(pooled_effect: float, pooled_se: float, delta: float,
                      alpha: float = 0.05) -> dict:
    """Two one-sided tests for equivalence (Schuirmann 1987).

    Tests H0: |θ| ≥ δ (effect non-equivalent) vs H1: |θ| < δ (effect equivalent).
    Both one-sided tests must reject H0 to demonstrate equivalence.

    Used in P79 paper-1 as **informational only** (reported alongside H1 superiority
    test, NOT used for H1 PRIMARY gating per 2026-05-13 prereg revision).
    """
    t_lo = (pooled_effect - (-delta)) / max(pooled_se, 1e-12)  # tests θ > -δ
    t_hi = ((+delta) - pooled_effect) / max(pooled_se, 1e-12)  # tests θ < +δ
    p_lo = 1.0 - _phi(t_lo)
    p_hi = 1.0 - _phi(t_hi)
    max_p = max(p_lo, p_hi)
    equivalence_demonstrated = (p_lo < alpha) and (p_hi < alpha)
    return {
        "delta": delta,
        "alpha_per_side": alpha,
        "pooled_effect": pooled_effect,
        "pooled_se": pooled_se,
        "p_lower_bound_test": p_lo,
        "p_upper_bound_test": p_hi,
        "max_p_value": max_p,
        "equivalence_demonstrated": equivalence_demonstrated,
        "decision": "equivalence_demonstrated" if equivalence_demonstrated else "equivalence_not_demonstrated",
    }


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------

def holm_correct(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm-Bonferroni step-down correction for a family of m tests.

    Returns list of dicts (in original order) with: p_raw, p_holm, rejected.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * m
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = (m - rank) * p
        adj = max(adj, prev_adj)
        adj = min(adj, 1.0)
        results[orig_idx] = {
            "p_raw": p,
            "p_holm": adj,
            "rejected": adj < alpha,
        }
        prev_adj = adj
    return results


# ---------------------------------------------------------------------------
# Hypothesis evaluators
# ---------------------------------------------------------------------------

def evaluate_h1(cells_by_id: dict[str, list[dict]], delta_pp: float = 1.0,
                 magnitude_threshold_pp: float = 1.0, alpha: float = 0.05,
                 transparency_K_h1: int = 3, bootstrap_seed: int = 42) -> dict:
    """H1: P-SoM drop-one oracle ceiling lift > 0, pooled across cells.

    PRIMARY: pooled DL meta sig at Holm α=0.05 (m=1) + θ_RE ≥ magnitude_threshold_pp
             + one-sided superiority test rejected at α=alpha (H0: θ ≤ +magnitude_threshold_pp).
             TOST equivalence test computed for informational reporting (NOT gating).
    TRANSPARENCY: K_h1 = transparency_K_h1 of N cells individually Holm-sig (m=N).
    """
    per_cell = {}
    effects_pp = []
    variances_pp = []  # variances of per-cell drop-one in pp^2
    per_cell_p_values = []

    for cell_id, tasks in cells_by_id.items():
        # F7 fix 2026-05-14 (codex /stress v6): per-call seed stratification so
        # each (cell, statistic) bootstrap draws an independent resample sequence.
        # Prior shared seed=42 made same-cell H1/H3 bootstrap CIs correlated.
        # /stress A1.19 P1-1-B* (2026-05-17, codex Mode B OOB reproducibility):
        # swap `hash()` → `hashlib.sha256()` so per-cell seed is process-stable
        # (PYTHONHASHSEED no longer affects bootstrap stream). Pre-fix empirically
        # produced pooled_effect 4.22569 vs 4.23549 across runs with same --seed 42.
        cell_seed = bootstrap_seed + (
            int(hashlib.sha256(f"{cell_id}|h1_drop_one".encode()).hexdigest()[:8], 16)
            % 100000
        )
        # A1.21 P0-6 fix: unpack 6 values (was 4); use percentile p-value not normal-approx
        point, ci_lo, ci_hi, se, p_pct_2s, _p_pct_1s = _paired_bootstrap(
            tasks,
            statistic_fn=lambda t: _drop_one_lift_per_cell(t, drop_mode="sr_psom"),
            seed=cell_seed,
        )
        # Convert to pp
        effect_pp = point * 100.0
        se_pp = se * 100.0
        # A1.21 P0-6: report BOTH percentile p (canonical, method-coherent with percentile CI)
        # AND normal-approx p (legacy backward-compat for synthetic test fixtures).
        z = effect_pp / max(se_pp, 1e-12)
        p_normal_approx = 2.0 * (1.0 - _phi(abs(z)))
        per_cell[cell_id] = {
            "drop_one_lift_pp": effect_pp,
            "ci_95_pp": [ci_lo * 100.0, ci_hi * 100.0],
            "se_pp": se_pp,
            "p_value_two_sided": p_pct_2s,  # A1.21 P0-6: percentile-based (canonical)
            "p_value_normal_approx_legacy": p_normal_approx,  # legacy z-score (NOT canonical)
            "n_tasks": len(tasks),
        }
        effects_pp.append(effect_pp)
        variances_pp.append(se_pp ** 2)
        per_cell_p_values.append(p_pct_2s)  # A1.21 P0-6: percentile p (canonical)

    # A1.21 P0-2 fix (B-513, codex Mode B): retire DL meta from PRIMARY H1 gate.
    # Canonical prereg lock is FE inverse-variance pool + one-sided superiority test
    # (single test, NOT 3-test compound). DL meta + magnitude check moved to
    # `appendix_dl_sensitivity` block for transparency reporting only.
    #
    # PRIMARY: FE inverse-variance pool + one-sided superiority (matches prereg §1 L68-86)
    import numpy as _np
    _thetas = _np.array(effects_pp)
    _ses = _np.array([math.sqrt(v) for v in variances_pp])
    # A1.19 B-426 SE floor 1.0pp (Agresti-Coull-style, prereg §2 H1 anchor)
    _n_zero_se = int((_ses <= 0).sum())
    if _n_zero_se > 0:
        _ses = _np.where(_ses <= 0, 1.0, _ses)
    _w = 1.0 / (_ses ** 2)
    _theta_fe = float(_np.sum(_w * _thetas) / _np.sum(_w))
    _se_fe = float(math.sqrt(1.0 / _np.sum(_w)))
    # A1.21 P0-11 fix: compute I² + Q on FE pool so `_effective_gate_pass` cap can fire.
    # Higgins & Thompson 2002: Q = Σ w_i (θ_i − θ_FE)², df = k-1, I² = max(0, (Q-df)/Q)·100
    _k_h1 = len(_thetas)
    if _k_h1 >= 2:
        _Q = float(_np.sum(_w * (_thetas - _theta_fe) ** 2))
        _df = _k_h1 - 1
        _isq = max(0.0, (_Q - _df) / _Q) * 100.0 if _Q > 0 else 0.0
    else:
        _Q, _df, _isq = None, 0, None
    fe_pool = {
        "pooled_effect": _theta_fe,
        "pooled_se": _se_fe,
        "pooled_ci_95": [_theta_fe - 1.96 * _se_fe, _theta_fe + 1.96 * _se_fe],
        "k": _k_h1,
        "n_zero_se_floored_cells": _n_zero_se,
        # A1.21 P0-11: I² + Q exposed for framing rule cap-only logic
        "Q": _Q,
        "Q_df": _df,
        "I_squared_pct": _isq,
    }
    superiority = superiority_test(_theta_fe, _se_fe,
                                     threshold=magnitude_threshold_pp, alpha=alpha)
    # TOST kept for informational reporting (NOT used in H1 gating decision)
    tost_info = tost_equivalence(_theta_fe, _se_fe, delta=delta_pp, alpha=alpha)

    # A1.21 P0-2: PRIMARY gate = FE superiority decision ONLY (single test, no compound)
    primary_h1_pass = superiority["decision"] == "reject_H0_substantively_above_threshold"

    # Appendix-DL sensitivity (transparency only, NOT a gate path — A1.21 P0-2)
    try:
        dl_meta_appendix = dersimonian_laird_meta(effects_pp, variances_pp)
    except InsufficientCellsError:
        dl_meta_appendix = {"k": len(effects_pp), "error": "k<2, DL undefined"}

    # TRANSPARENCY: K-of-N Holm
    holm_per_cell = holm_correct(per_cell_p_values, alpha=alpha)
    for (cell_id, _), h in zip(per_cell.items(), holm_per_cell):
        per_cell[cell_id]["holm_p"] = h["p_holm"]
        per_cell[cell_id]["individually_holm_sig"] = h["rejected"]
    n_individually_sig = sum(1 for h in holm_per_cell if h["rejected"])
    transparency_pass = n_individually_sig >= transparency_K_h1

    return {
        "primary_gate": {
            # A1.21 P0-2 fix: pooled_meta is now FE pool (NOT DL); see canonical
            # `aggregate_phase1_full_prereg_decision.py` for bit-identical FE path.
            "pooled_meta": fe_pool,
            "estimand": "FE inverse-variance pool + one-sided superiority test",
            "superiority_test": superiority,
            "tost_informational": tost_info,
            "decision": "PASS" if primary_h1_pass else "FAIL",
        },
        "appendix_dl_sensitivity": {
            "estimand": "DerSimonian-Laird random-effects (appendix sensitivity only)",
            "note": "A1.21 P0-2 (B-513): retired from H1 PRIMARY gate; reported here "
                    "for transparency only. Canonical paper §1 H1 = FE superiority above.",
            "dl_meta": dl_meta_appendix,
        },
        "transparency_K_h1": {
            "K": transparency_K_h1,
            "N": len(cells_by_id),
            "n_individually_holm_sig": n_individually_sig,
            "consistent": transparency_pass,
            "note": "transparency-only, NOT a gate on H1 (per prereg 2026-05-13 reclassification)",
        },
        "per_cell": per_cell,
    }


def evaluate_h3_axis(cells_by_id: dict[str, list[dict]], axis_mode_key: str,
                      ref_mode_key: str = "sr_psom", min_unique_count: int = 2,
                      alpha: float = 0.05, transparency_K_h3: int = 3,
                      bootstrap_seed: int = 42) -> dict:
    """H3 axis test: |axis_mode \\ ref_mode| > 0, pooled across cells.

    axis_mode_key examples: sr_ptext (axis-1), sr_pprompt (axis-2).

    PRIMARY: pooled DL meta on unique-count, CI excluding 0 at Holm α=0.05 (m=1).
    TRANSPARENCY: K_h3 of N cells with bootstrap CI > 0 AND unique-count ≥ min_unique_count.
    """
    per_cell = {}
    effects = []
    variances = []
    per_cell_p_values = []
    per_cell_ci_excludes_zero = []

    for cell_id, tasks in cells_by_id.items():
        # Statistic: count of tasks where axis solved but ref did not, normalized by task count
        # (using count as the statistic per prereg H3 wording)
        # F7 fix 2026-05-14: per-call seed stratification (see evaluate_h1).
        # /stress A1.19 P1-1-B* (2026-05-17): hashlib.sha256 swap for reproducibility (see evaluate_h1).
        cell_seed = bootstrap_seed + (
            int(hashlib.sha256(f"{cell_id}|h3_{axis_mode_key}".encode()).hexdigest()[:8], 16)
            % 100000
        )
        # A1.21 P0-6: unpack 6 values (was 4); use percentile one-sided p
        count, ci_lo, ci_hi, se, _p_pct_2s, p_pct_1s = _paired_bootstrap(
            tasks,
            statistic_fn=lambda t: float(_unique_count_per_cell(t, axis_mode_key, ref_mode_key)),
            seed=cell_seed,
        )
        # Per-cell pass: CI > 0 AND count ≥ min_unique_count (≥2 floor for noise)
        ci_excludes_zero = ci_lo > 0
        count_above_floor = count >= min_unique_count
        per_cell_pass = ci_excludes_zero and count_above_floor
        # A1.21 P0-6: percentile one-sided p-value (was normal-approx z-score)
        z = count / max(se, 1e-12)
        p_normal_approx = 1.0 - _phi(z)
        per_cell[cell_id] = {
            "unique_count": count,
            "ci_95": [ci_lo, ci_hi],
            "se": se,
            "p_value_one_sided": p_pct_1s,  # A1.21 P0-6: percentile (canonical)
            "p_value_normal_approx_legacy": p_normal_approx,
            "ci_excludes_zero": ci_excludes_zero,
            "count_above_min": count_above_floor,
            "per_cell_pass": per_cell_pass,
            "n_tasks": len(tasks),
        }
        effects.append(count)
        variances.append(se ** 2)
        per_cell_p_values.append(p_pct_1s)  # A1.21 P0-6: percentile p (canonical)
        per_cell_ci_excludes_zero.append(per_cell_pass)

    # A1.21 P0-2 fix (B-513): H3 PRIMARY gate FE-only (matches H1 + canonical
    # `aggregate_phase1_full_prereg_decision.py`); DL moved to appendix sensitivity.
    import numpy as _np
    _effs = _np.array(effects)
    _ses_h3 = _np.array([math.sqrt(v) for v in variances])
    _n_zero_se_h3 = int((_ses_h3 <= 0).sum())
    if _n_zero_se_h3 > 0:
        _ses_h3 = _np.where(_ses_h3 <= 0, 1.0, _ses_h3)
    _w_h3 = 1.0 / (_ses_h3 ** 2)
    _theta_fe_h3 = float(_np.sum(_w_h3 * _effs) / _np.sum(_w_h3))
    _se_fe_h3 = float(math.sqrt(1.0 / _np.sum(_w_h3)))
    _z_h3 = _theta_fe_h3 / max(_se_fe_h3, 1e-12)
    _p_one_sided_h3 = 1.0 - _phi(_z_h3)
    # A1.21 P0-11: I² + Q for framing cap fidelity (same as H1)
    _k_h3 = len(_effs)
    if _k_h3 >= 2:
        _Q_h3 = float(_np.sum(_w_h3 * (_effs - _theta_fe_h3) ** 2))
        _df_h3 = _k_h3 - 1
        _isq_h3 = max(0.0, (_Q_h3 - _df_h3) / _Q_h3) * 100.0 if _Q_h3 > 0 else 0.0
    else:
        _Q_h3, _df_h3, _isq_h3 = None, 0, None
    meta = {
        "pooled_effect": _theta_fe_h3,
        "pooled_se": _se_fe_h3,
        "pooled_ci_95": [_theta_fe_h3 - 1.96 * _se_fe_h3, _theta_fe_h3 + 1.96 * _se_fe_h3],
        "p_value_one_sided": _p_one_sided_h3,
        "p_value_two_sided": 2.0 * (1.0 - _phi(abs(_z_h3))),
        "k": _k_h3,
        "n_zero_se_floored_cells": _n_zero_se_h3,
        "estimand": "FE inverse-variance pool",
        "Q": _Q_h3,
        "Q_df": _df_h3,
        "I_squared_pct": _isq_h3,
    }
    pooled_ci_lo = meta["pooled_ci_95"][0]
    primary_pass = (_p_one_sided_h3 < alpha and pooled_ci_lo > 0)
    # Appendix DL sensitivity (transparency)
    try:
        dl_meta_appendix_h3 = dersimonian_laird_meta(effects, variances)
    except InsufficientCellsError:
        dl_meta_appendix_h3 = {"k": len(effects), "error": "k<2, DL undefined"}

    # TRANSPARENCY
    holm_per_cell = holm_correct(per_cell_p_values, alpha=alpha)
    for (cell_id, _), h in zip(per_cell.items(), holm_per_cell):
        per_cell[cell_id]["holm_p"] = h["p_holm"]
        per_cell[cell_id]["individually_holm_sig"] = h["rejected"]
    n_per_cell_pass = sum(per_cell_ci_excludes_zero)
    transparency_pass = n_per_cell_pass >= transparency_K_h3

    return {
        "axis_mode": axis_mode_key,
        "ref_mode": ref_mode_key,
        "primary_gate": {
            "pooled_meta": meta,  # A1.21 P0-2: FE pool (was DL)
            "ci_excludes_zero": pooled_ci_lo > 0,
            "decision": "PASS" if primary_pass else "FAIL",
        },
        "appendix_dl_sensitivity": {
            "estimand": "DerSimonian-Laird random-effects (appendix sensitivity only)",
            "note": "A1.21 P0-2 (B-513): retired from H3 PRIMARY gate; reported for transparency.",
            "dl_meta": dl_meta_appendix_h3,
        },
        "transparency_K_h3": {
            "K": transparency_K_h3,
            "N": len(cells_by_id),
            "n_cells_pass": n_per_cell_pass,
            "consistent": transparency_pass,
            "note": "transparency-only, NOT a gate on H3 (per prereg 2026-05-13 reclassification)",
        },
        "per_cell": per_cell,
    }


def evaluate_h2_cost(cells_by_id: dict[str, list[dict]], cost_margin_pct: float = 20.0,
                      transparency_K_h2: int | None = None) -> dict:
    """H2(a): per-task median cost ratio cost(P-SoM)/cost(DOM) within ±cost_margin_pct% per cell.

    A1.21 P0-9 fix (B-518, prereg §2 H2(a) prose lock amend 2026-05-17):
    Estimand rewrite — was median(P-SoM costs) / median(DOM costs) (marginal medians,
    paired info ignored), now median over tasks of (cost_psom[t] / cost_dom[t]) per cell.

    Why: paper §1 line 9 "the cost of obtaining this configuration is essentially the
    cost of the DOM baseline" is a **per-task claim** (each task should cost about the
    same under P-SoM as DOM). Marginal medians measure cost-distribution-shape — a
    different claim. When cost variance is heterogeneous across tasks (LLM token counts
    vary 10-100×), marginal-median can collapse to a number unrelated to per-task ratio.
    Prereg §2 H2(a) prose lock 2026-05-17 added "per-task median ratio" disambiguation.

    A1.21 P1-7 fix (B-519): 3-state per cell (within_band / falsified / cannot_evaluate).
    Pre-fix: missing cost data → `per_cell_pass: False` → counted as falsification by
    framing rule. Now: distinct state, framing rule uses `n_falsified > 0` (NOT
    `pass_count == total`), missing data does NOT trigger R4 framing degradation.

    H2(a) test margin is a RELATIVE PERCENTAGE (e.g., ±20% of DOM cost), distinct from
    H1 TOST δ which is an SR percentage-point margin.
    """
    if transparency_K_h2 is not None:
        import warnings as _w
        _w.warn(
            "evaluate_h2_cost: `transparency_K_h2` arg is DEPRECATED and ignored "
            "(/stress A1.19 P0-3 2026-05-17 — prereg lock 2026-05-14 H2(a) is strict "
            "ALL-cells-pass falsification check, not K-of-N transparency). "
            "Remove the arg; falsification = ALL cells within band.",
            DeprecationWarning, stacklevel=2,
        )
    per_cell = {}
    n_within = 0
    n_falsified = 0
    n_cannot_evaluate = 0
    n_cells_total = len(cells_by_id)
    for cell_id, tasks in cells_by_id.items():
        # A1.21 P0-9: per-task ratio (paired, NOT marginal medians).
        # A1.21 P0-1: `is not None` check, NOT truthy `or` (avoid 0.0 short-circuit drop).
        per_task_ratios = []
        n_dom_zero = 0
        n_missing = 0
        for t in tasks:
            cd_raw = t.get("cost_dom")
            cp_raw = t.get("cost_psom")
            if cd_raw is None or cd_raw == "" or cp_raw is None or cp_raw == "":
                n_missing += 1
                continue
            try:
                cd = float(cd_raw)
                cp = float(cp_raw)
            except (TypeError, ValueError):
                n_missing += 1
                continue
            if cd <= 0:
                n_dom_zero += 1
                continue
            per_task_ratios.append(cp / cd)
        if not per_task_ratios:
            # A1.21 P1-7: cannot_evaluate state (distinct from falsified)
            per_cell[cell_id] = {
                "state": "cannot_evaluate",
                "reason": f"no valid per-task ratios (n_missing={n_missing}, "
                          f"n_dom_zero={n_dom_zero})",
                "per_cell_pass": None,
            }
            n_cannot_evaluate += 1
            continue
        med_ratio = statistics.median(per_task_ratios)
        rel_diff_pct = (med_ratio - 1.0) * 100.0
        within_band = abs(rel_diff_pct) <= cost_margin_pct
        per_cell[cell_id] = {
            "n_per_task_ratios": len(per_task_ratios),
            "n_missing": n_missing,
            "n_dom_zero_skipped": n_dom_zero,
            "median_per_task_ratio": med_ratio,
            "relative_diff_pct": rel_diff_pct,
            "margin_pct": cost_margin_pct,
            "state": "within_band" if within_band else "falsified",
            "per_cell_pass": within_band,
        }
        if within_band:
            n_within += 1
        else:
            n_falsified += 1
    # A1.21 P1-7 fix: framing rule uses `n_falsified > 0` not `pass_count == total`.
    # Missing data is NOT falsification.
    not_falsified = (n_falsified == 0)
    return {
        "h2a_cost_equivalence": {
            "N": n_cells_total,
            "n_cells_within_band": n_within,
            "n_cells_falsified": n_falsified,
            "n_cells_cannot_evaluate": n_cannot_evaluate,
            # `consistent` = NOT falsified (any cell within +/- margin counts; missing data ignored)
            "consistent": not_falsified,
            "n_cells_pass": n_within,  # backward-compat field
            "margin_pct": cost_margin_pct,
            "semantics": "per_task_ratio_strict_no_cell_falsification (P0-9+P1-7 reframe)",
            "estimand": "per-task median ratio cost(P-SoM)/cost(DOM) (paired)",
            "prereg_anchor": "preregistration.md §2 H2(a) line 120-145 + 2026-05-17 prose lock amend",
        },
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# Framing rule R1-R5 mapper
# ---------------------------------------------------------------------------

def _effective_gate_pass(gate_result: dict, gate_kind: str,
                          heterogeneity_threshold_pct: float) -> tuple[bool, bool, dict]:
    """Determine whether a primary gate passes + report I² heterogeneity status.

    A1.21 P0-3 fix (B-514, codex Mode B OOB): heterogeneity-rescue branch RETIRED.
    Prior code (F3 2026-05-14) rescued failed pooled gate via per-cell consistency
    (≥3 direction-positive AND ≥2 individually sig). This VIOLATED prereg §2 L323
    ("high heterogeneity does NOT block pooling, only caps the hook") + L340-342
    ("p ≥ 0.05 → H1 FAILS → R5"). Code could rescue R5 → R3 via per-cell rule.

    Post-fix: pooled gate decision is canonical regardless of I². I² serves
    `apply_framing_rule` as cap-only signal (R1/R2 → R3 when primary passes,
    NOT to rescue R5). FE pool is pre-registered estimand (Decision 3A 2026-05-14)
    so per-cell consistency is not a backup mode.

    Returns (passes, heterogeneous, detail) where:
      - passes = pooled primary gate PASS decision (unchanged regardless of I²)
      - heterogeneous = I² > threshold (used as framing CAP by apply_framing_rule)

    gate_kind ∈ {"h1", "h3_axis"} retained for API compat but not used in decision.
    """
    meta = gate_result.get("primary_gate", {}).get("pooled_meta", {})
    i_sq = meta.get("I_squared_pct")  # NB: FE pool doesn't compute I²; canonical
    # producer computes Q + I² separately. Here we just report what meta carries.
    heterogeneous = (i_sq is not None and i_sq > heterogeneity_threshold_pct)
    passes = gate_result["primary_gate"]["decision"] == "PASS"
    return passes, heterogeneous, {
        "I_squared_pct": i_sq,
        "via": "pooled_meta (canonical, A1.21 P0-3 per-cell rescue retired)",
        "_unused_gate_kind": gate_kind,
    }


def apply_framing_rule(h1: dict, h2: dict, h3_axis1: dict, h3_axis2: dict,
                        heterogeneity_threshold_pct: float = 75.0) -> dict:
    """Apply preregistration §2 R1-R5 framing rule to test outcomes.

    F3 fix 2026-05-14 (codex /stress v6): heterogeneity check now applies to ALL
    primary gates via `_effective_gate_pass()`, not just H1. If ANY primary gate
    has pooled meta I² > 75%, the pooled "2-axis empirical structure" claim is not
    supported — hook caps at R3 (heterogeneity-conditional per-cell consistency).

    Rationale: R1/R2 framing claims a pooled empirical structure. I² > 75% on any
    component gate means that gate's effect is NOT consistent across cells →
    pooling is invalid for that gate → the structural claim it underwrites cannot
    be made at pooled-meta strength. R3 (per-cell consistency) is the ceiling.
    """
    h1_pass, h1_het, h1_det = _effective_gate_pass(h1, "h1", heterogeneity_threshold_pct)
    h3a_pass, h3a_het, h3a_det = _effective_gate_pass(h3_axis1, "h3_axis", heterogeneity_threshold_pct)
    h3b_pass, h3b_het, h3b_det = _effective_gate_pass(h3_axis2, "h3_axis", heterogeneity_threshold_pct)
    h2a_cost_pass = h2["h2a_cost_equivalence"]["consistent"]  # H2(a) only per 2026-05-13 T2 scope

    any_heterogeneous = h1_het or h3a_het or h3b_het
    heterogeneity_detail = {"h1": h1_det, "h3_axis1": h3a_det, "h3_axis2": h3b_det}

    # A1.21 P0-3 fix (B-514): heterogeneity = CAP-ONLY (R1/R2 → R3), NOT rescue (R5 → R3).
    # Prereg §2 L323-342: high I² caps framing power but H1 FAIL → R5 always.
    # Prior code (heterogeneity_override branch) rescued failed H1 → prereg violation.

    # H1 failed → R5 (paper death) regardless of I² or H2(a) or H3 — prereg L340-342
    if not h1_pass:
        return {"rule": "R5", "framing": "Paper death scenario — H1 FE superiority failed; pivot needed",
                "hook_power": "n/a", "heterogeneity_override": False,
                "heterogeneity_detail": heterogeneity_detail}
    # H1 passed but H2(a) falsified → R4
    if not h2a_cost_pass:
        return {"rule": "R4", "framing": "Phantom-SoM partial drop-in (H2(a) cost equivalence fails)",
                "hook_power": "WEAK", "heterogeneity_override": False,
                "heterogeneity_detail": heterogeneity_detail}
    # H1 passed + H2(a) not falsified — primary R-rule by H3 axes
    if h3a_pass and h3b_pass:
        primary = ("R1", "Phantom routing space (2-axis empirical structure)", "STRONGEST")
    elif h3a_pass or h3b_pass:
        primary = ("R2", "Phantom routing space (single-axis empirical structure)", "MODERATE-STRONG")
    else:
        primary = ("R3", "Phantom-SoM is hidden 4th routing arm (workshop-grade R3)", "MODERATE")
    # I² cap-only — caps R1/R2 → R3, does NOT change H1 pass decision (prereg L323)
    if any_heterogeneous and primary[0] in ("R1", "R2"):
        return {"rule": "R3",
                "framing": f"Heterogeneity-capped R3 — ≥1 gate I² > 75%; original "
                           f"rule {primary[0]} capped per prereg §2 L323 (cap-only)",
                "hook_power": "MODERATE",
                "heterogeneity_override": True,
                "heterogeneity_detail": heterogeneity_detail,
                "original_rule_pre_cap": primary[0]}
    return {"rule": primary[0], "framing": primary[1], "hook_power": primary[2],
            "heterogeneity_override": False, "heterogeneity_detail": heterogeneity_detail}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_per_task_csv(csv_path: Path) -> dict[str, list[dict]]:
    """Load per-task CSV, return dict of cell_id → list of task rows."""
    cells_by_id: dict[str, list[dict]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            cells_by_id[row["cell_id"]].append(row)
    return dict(cells_by_id)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Synthetic data generator (24-condition / 4-cell smoke test)
# ---------------------------------------------------------------------------

def generate_synthetic_per_task(seed: int = 42, n_tasks_per_cell: int = 200,
                                  scenario: str = "r1_pass") -> dict[str, list[dict]]:
    """Generate Phase 1a 4-cell × n_tasks per-task data.

    A6 fix 2026-05-13: scenarios with H1/H3 failure modes now enforce per-task
    correlation (NOT independent Bernoulli) so the test fixture actually exhibits
    the failure mode. Prior generator's independent-Bernoulli design meant even
    "fail" scenarios accidentally passed (codex Round A Flaw A6).

    Scenarios:
      - r1_pass:   H1 strong (drop-one ~2pp pooled), H2(a) cost equiv hold, H3 both axes pass.
                   Independent Bernoulli + favorable base rates.
      - r3_pass:   H1 holds (P-SoM ⊋ {DOM ∪ SoM ∪ Vision} on some tasks → drop-one > 0)
                   AND H3 fails BOTH axes (P-text ⊆ P-SoM AND P-prompt ⊆ P-SoM by construction).
      - r5_fail:   H1 fails (P-SoM ⊆ {DOM ∪ SoM ∪ Vision} on ALL tasks → drop-one = 0).
                   Also H3 fails by similar subset construction.
      - heterogeneity_test: H1 pooled magnitude OK but I² > 75% from injected cell variance.
    """
    import random
    rng = random.Random(seed)
    cells_by_id = {}
    for cell_idx, (site, model) in enumerate(PHASE_1A_CELLS):
        cell_id = f"{site}_{model}"
        # Base per-task SR rates (per mode)
        base_rate = {"sr_dom": 0.30, "sr_som": 0.32, "sr_vision": 0.20,
                     "sr_ptext": 0.31, "sr_pprompt": 0.28, "sr_psom": 0.34}
        # Capability adjustment. r1_pass is the deliberately-homogeneous happy-path
        # fixture (all 6 cells identical distribution) so it demonstrably routes R1.
        # B1 + B2 both get 0.6× capability multiplier (advisor §138 lock: B2 ≈ B1
        # matched-capability cross-family control). A1.21 P1-10 fix (B-520): pre-fix
        # only B1 was scaled → B2 = B0 by accident, violating advisor matched-capability
        # assumption and inducing wrong-shaped heterogeneity in test scenarios.
        if model in ("B1", "B2") and scenario != "r1_pass":
            base_rate = {k: v * 0.6 for k, v in base_rate.items()}
        # Cell-level effect-size variance for heterogeneity test — large bimodal
        # shift so between-cell variance >> within-cell bootstrap SE → I² > 75%.
        # A1.21 P1-10 sibling fix (B-520): 4-element hardcoded list extended to 6
        # to match Phase 1a 6-cell scope (was IndexError when B2 added).
        if scenario == "heterogeneity_test":
            _cell_shifts = [+0.25, -0.20, +0.25, -0.20, +0.25, -0.20]
            cell_shift = _cell_shifts[cell_idx % len(_cell_shifts)]
            base_rate["sr_psom"] = max(0.0, min(1.0, base_rate["sr_psom"] + cell_shift))

        rows = []
        for i in range(n_tasks_per_cell):
            # Per-task latent solvability bias (correlates modes within task)
            bias = rng.uniform(-0.1, 0.1)
            row = {"cell_id": cell_id, "site": site, "model": model,
                   "task_id": f"{cell_id}_t{i:04d}"}

            # Sample baseline modes (DOM/SoM/Vision) independently per task
            for mode_key in ("sr_dom", "sr_som", "sr_vision"):
                eff_rate = max(0.0, min(1.0, base_rate[mode_key] + bias))
                row[mode_key] = 1 if rng.random() < eff_rate else 0

            baseline_union = 1 if any(row[k] for k in ("sr_dom", "sr_som", "sr_vision")) else 0

            # Sample phantom modes per scenario logic
            if scenario == "r5_fail":
                # H1 fails: P-SoM strict subset of baseline union → drop-one = 0
                # P-SoM = baseline_union AND p_psom_subset_rate
                p_psom_subset_rate = 0.85
                row["sr_psom"] = baseline_union * (1 if rng.random() < p_psom_subset_rate else 0)
                # H3 also fails: P-text/P-prompt ⊆ P-SoM
                row["sr_ptext"] = row["sr_psom"] * (1 if rng.random() < 0.95 else 0)
                row["sr_pprompt"] = row["sr_psom"] * (1 if rng.random() < 0.95 else 0)
            elif scenario == "r3_pass":
                # H1 holds: P-SoM independent of baseline → drop-one > 0
                eff_psom = max(0.0, min(1.0, base_rate["sr_psom"] + bias))
                row["sr_psom"] = 1 if rng.random() < eff_psom else 0
                # H3 fails: P-text + P-prompt are SPARSE subsets of P-SoM. When P-SoM
                # solves, P-text fires ~30% of the time (so 70% of P-SoM-solved tasks
                # have psom-unique contribution = drop-one > 0). H3 axis-1 unique = 0
                # because ptext=1 only when psom=1 (sparse subset by construction).
                row["sr_ptext"] = row["sr_psom"] * (1 if rng.random() < 0.30 else 0)
                row["sr_pprompt"] = row["sr_psom"] * (1 if rng.random() < 0.30 else 0)
            else:
                # r1_pass and heterogeneity_test: independent per-mode Bernoulli
                for mode_key in ("sr_ptext", "sr_pprompt", "sr_psom"):
                    eff_rate = max(0.0, min(1.0, base_rate[mode_key] + bias))
                    row[mode_key] = 1 if rng.random() < eff_rate else 0

            # Cost: P-SoM ~ DOM cost (regex filter property)
            row["cost_dom"] = 0.040 + rng.uniform(-0.005, 0.005)
            row["cost_psom"] = row["cost_dom"] * (1.0 + rng.uniform(-0.05, 0.05))
            rows.append(row)
        cells_by_id[cell_id] = rows
    return cells_by_id


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-task-csv",
                   help="Per-task CSV path (cell_id, site, model, task_id, sr_*, cost_*)")
    p.add_argument("--synthetic", action="store_true",
                   help="Run smoke test on synthetic 4-cell × 200-task data")
    p.add_argument("--scenario", default="r1_pass",
                   choices=["r1_pass", "r3_pass", "r5_fail", "heterogeneity_test"])
    p.add_argument("--seed", type=int, default=42,
                   help="Combined seed (DEPRECATED — use --data-seed + --bootstrap-seed). "
                        "If --data-seed/--bootstrap-seed not given, falls back to --seed. "
                        "A1.21 P1-9 fix (B-520): split for reproducibility audit clarity.")
    p.add_argument("--data-seed", type=int, default=None,
                   help="Seed for synthetic data generation (A1.21 P1-9 split from --seed)")
    p.add_argument("--bootstrap-seed", type=int, default=None,
                   help="Seed for paired bootstrap resampling (A1.21 P1-9 split from --seed)")
    p.add_argument("--primary-gate", default="drop_one_pooled_meta_TOST",
                   help="Primary gate flavor (informational; method is fixed in this rewrite)")
    p.add_argument("--TOST-delta-pp", type=float, default=1.0,
                   help="TOST equivalence margin in SR pp (default 1.0 per prereg lock)")
    p.add_argument("--H1-magnitude-pp", type=float, default=1.0,
                   help="H1 pooled magnitude threshold (default 1.0pp per prereg lock)")
    p.add_argument("--H2-cost-margin-pct", type=float, default=20.0,
                   help="H2(a) cost equivalence margin in %% (default 20%% per prereg lock "
                        "L120-137: 1.20× median cost ratio = +20%% relative). "
                        "/stress A1.19 P0-3 (2026-05-17): default fixed 10.0→20.0 to "
                        "match prereg lock. ANY cell exceeding margin → falsified.")
    p.add_argument("--H3-min-unique-count", type=int, default=2,
                   help="H3 per-cell unique-count noise floor (default 2 tasks)")
    p.add_argument("--transparency-K_h1", type=int, default=3,
                   help="K_h1 transparency ratio cells count (default 3 of 4)")
    p.add_argument("--transparency-K_h3", type=int, default=3,
                   help="K_h3 transparency ratio cells count per axis (default 3 of 4)")
    p.add_argument("--transparency-K_h2", type=int, default=None,
                   help="DEPRECATED (/stress A1.19 P0-3 2026-05-17): H2(a) is strict "
                        "ALL-cells-pass falsification check per prereg L131-132 + L368, "
                        "not K-of-N transparency. Arg retained for backward-compat with "
                        "older CLI invocations but ignored; emits DeprecationWarning.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out", default="-", help="Output JSON path (- = stdout)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # A1.21 P1-9 fix: separate data + bootstrap seeds; fall back to --seed if not given
    data_seed = args.data_seed if args.data_seed is not None else args.seed
    bootstrap_seed = args.bootstrap_seed if args.bootstrap_seed is not None else args.seed

    # Load data
    if args.synthetic:
        cells_by_id = generate_synthetic_per_task(seed=data_seed, scenario=args.scenario)
        input_sha = f"synthetic:{args.scenario}:data_seed={data_seed}:bootstrap_seed={bootstrap_seed}"
        logger.info(f"Synthetic mode: {len(cells_by_id)} cells, scenario={args.scenario}, "
                    f"data_seed={data_seed}, bootstrap_seed={bootstrap_seed}")
    else:
        if not args.per_task_csv:
            logger.error("Must provide --per-task-csv or --synthetic")
            sys.exit(2)
        csv_path = Path(args.per_task_csv)
        cells_by_id = load_per_task_csv(csv_path)
        input_sha = _file_sha256(csv_path)
        logger.info(f"Loaded {len(cells_by_id)} cells from {csv_path} (sha256={input_sha[:12]}...)")

    if len(cells_by_id) < 2:
        logger.error(f"Need ≥2 cells for pooled meta; got {len(cells_by_id)}")
        sys.exit(2)

    # Evaluate hypotheses (A1.21 P1-9: bootstrap_seed split from data_seed)
    h1 = evaluate_h1(cells_by_id, delta_pp=args.TOST_delta_pp,
                      magnitude_threshold_pp=args.H1_magnitude_pp,
                      alpha=args.alpha, transparency_K_h1=args.transparency_K_h1,
                      bootstrap_seed=bootstrap_seed)
    h2 = evaluate_h2_cost(cells_by_id, cost_margin_pct=args.H2_cost_margin_pct,
                           transparency_K_h2=args.transparency_K_h2)
    h3_axis1 = evaluate_h3_axis(cells_by_id, axis_mode_key="sr_ptext",
                                  ref_mode_key="sr_psom",
                                  min_unique_count=args.H3_min_unique_count,
                                  alpha=args.alpha,
                                  transparency_K_h3=args.transparency_K_h3,
                                  bootstrap_seed=bootstrap_seed)
    h3_axis2 = evaluate_h3_axis(cells_by_id, axis_mode_key="sr_pprompt",
                                  ref_mode_key="sr_psom",
                                  min_unique_count=args.H3_min_unique_count,
                                  alpha=args.alpha,
                                  transparency_K_h3=args.transparency_K_h3,
                                  bootstrap_seed=bootstrap_seed)
    framing = apply_framing_rule(h1, h2, h3_axis1, h3_axis2)

    result = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        # A1.21 P2-2 fix (B-521): stale "24-condition / 4-cell" scope updated.
        # Current Phase 1a scope = 36 baseline + 6 router = 42 conditions / 6 cells statistical.
        # Note: this script is now synthetic-fixture + appendix-DL-sensitivity producer;
        # canonical paper §1 producer is `aggregate_phase1_full_prereg_decision.py`.
        "scope": "Phase 1a 42-condition (36 baseline + 6 router) / 6-cell synthetic test "
                 "fixture + appendix DL sensitivity producer (A1.21 P2-2 scope update)",
        "n_cells": len(cells_by_id),
        "n_tasks_total": sum(len(t) for t in cells_by_id.values()),
        "cell_ids": list(cells_by_id.keys()),
        "input_data_sha256": input_sha,
        "thresholds": {
            "primary_gate_method": "pooled_DerSimonian_Laird_meta + one_sided_superiority + magnitude (TOST informational)",
            "TOST_delta_pp": args.TOST_delta_pp,
            "H1_magnitude_pp": args.H1_magnitude_pp,
            "H2_cost_margin_pct": args.H2_cost_margin_pct,
            "H3_min_unique_count": args.H3_min_unique_count,
            "transparency_K_h1": args.transparency_K_h1,
            "transparency_K_h3": args.transparency_K_h3,
            "transparency_K_h2": args.transparency_K_h2,
            "alpha": args.alpha,
        },
        "H1_psom_drop_one": h1,
        "H2_cost_equivalence": h2,
        "H3_axis1_ptext_unique": h3_axis1,
        "H3_axis2_pprompt_unique": h3_axis2,
        "framing_rule": framing,
        "primary_gate_summary": {
            "H1": h1["primary_gate"]["decision"],
            "H2": "PASS" if h2["h2a_cost_equivalence"]["consistent"] else "FAIL",
            "H3_axis1": h3_axis1["primary_gate"]["decision"],
            "H3_axis2": h3_axis2["primary_gate"]["decision"],
        },
        "transparency_summary": {
            "K_h1": f"{h1['transparency_K_h1']['n_individually_holm_sig']}/{h1['transparency_K_h1']['N']} ≥ {h1['transparency_K_h1']['K']}?  {'YES' if h1['transparency_K_h1']['consistent'] else 'NO'}",
            "K_h3_axis1": f"{h3_axis1['transparency_K_h3']['n_cells_pass']}/{h3_axis1['transparency_K_h3']['N']} ≥ {h3_axis1['transparency_K_h3']['K']}?  {'YES' if h3_axis1['transparency_K_h3']['consistent'] else 'NO'}",
            "K_h3_axis2": f"{h3_axis2['transparency_K_h3']['n_cells_pass']}/{h3_axis2['transparency_K_h3']['N']} ≥ {h3_axis2['transparency_K_h3']['K']}?  {'YES' if h3_axis2['transparency_K_h3']['consistent'] else 'NO'}",
        },
    }

    payload = json.dumps(result, indent=2, default=float)
    if args.out == "-":
        print(payload)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
        logger.info(f"Result → {out_path}")
        logger.info(f"Framing rule: {framing['rule']} — {framing['framing']} (hook power: {framing['hook_power']})")
        logger.info(f"  H1: {h1['primary_gate']['decision']} (pooled drop-one {h1['primary_gate']['pooled_meta']['pooled_effect']:.2f}pp, "
                    f"superiority p={h1['primary_gate']['superiority_test']['p_one_sided']:.4f}, "
                    f"TOST equiv {h1['primary_gate']['tost_informational']['decision']})")
        logger.info(f"  H2: {'PASS' if h2['h2a_cost_equivalence']['consistent'] else 'FAIL'} "
                    f"({h2['h2a_cost_equivalence']['n_cells_pass']}/{h2['h2a_cost_equivalence']['N']} cells within ±{args.H2_cost_margin_pct}% cost)")
        logger.info(f"  H3 axis-1 (P-text): {h3_axis1['primary_gate']['decision']} "
                    f"(pooled unique={h3_axis1['primary_gate']['pooled_meta']['pooled_effect']:.2f})")
        logger.info(f"  H3 axis-2 (P-prompt): {h3_axis2['primary_gate']['decision']} "
                    f"(pooled unique={h3_axis2['primary_gate']['pooled_meta']['pooled_effect']:.2f})")
        logger.info(f"  Transparency K_h1: {result['transparency_summary']['K_h1']}")
        logger.info(f"  Transparency K_h3 axis-1: {result['transparency_summary']['K_h3_axis1']}")
        logger.info(f"  Transparency K_h3 axis-2: {result['transparency_summary']['K_h3_axis2']}")


if __name__ == "__main__":
    main()
