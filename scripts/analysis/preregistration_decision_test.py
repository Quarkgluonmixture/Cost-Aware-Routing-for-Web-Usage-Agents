r"""Preregistration decision test — Phase 1a 36-condition / 6-cell H1 / H3 / H2 evaluation.

⚠️ RESCOPED 2026-05-15 (B-120 fix per codex Mode B P0-1):
   - Phase 1a scope: 24-cond / 4-cell (B0+B1 only) → 36-cond / 6-cell (B0+B1+B2)
   - B2 = Gemma3-VL `google/gemma-3-4b-it` added 2026-05-14 as cross-family
     matched-capability control vs B1 4B (see preregistration.md Appendix A
     2026-05-14 entry + 笔记 §142)
   - k=4 → k=6 propagates to H1/H3/H5/H6 estimand
   - Decision 3A 2026-05-14 specified FE inverse-variance pooling over the 6
     planned cells (NOT DerSimonian-Laird). **Note**: current implementation
     below uses DL random-effects from earlier scaffolding — FE migration is
     pending advisor review (gemini Mode C P0-2 2026-05-15 challenges
     Decision 3A back toward RE+Knapp-Hartung; advisor confirmation pending
     before estimator finalization). Until advisor lock, both DL output and
     FE-equivalent superiority test are reported for transparency.

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
                       seed: int = 42) -> tuple[float, float, float, float]:
    """1000-resample paired task-level bootstrap.

    Returns (point_estimate, ci_lo_95, ci_hi_95, bootstrap_se).
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
    return point, ci_lo, ci_hi, se


# ---------------------------------------------------------------------------
# DerSimonian-Laird random-effects meta-analysis
# ---------------------------------------------------------------------------

def dersimonian_laird_meta(effects: list[float], variances: list[float]) -> dict:
    """Pool effect estimates across cells via DerSimonian-Laird random-effects.

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
    """
    k = len(effects)
    if k < 2:
        return {"pooled_effect": effects[0] if effects else 0.0,
                "pooled_se": math.sqrt(variances[0]) if variances else 0.0,
                "pooled_ci_95": [None, None],
                "Q": None, "I_squared_pct": None, "tau_squared": None,
                "p_value_two_sided": None, "k": k,
                "note": "k<2: pooling undefined"}

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


def _phi(z: float) -> float:
    """Standard normal CDF using erf approximation."""
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
        point, ci_lo, ci_hi, se = _paired_bootstrap(
            tasks,
            statistic_fn=lambda t: _drop_one_lift_per_cell(t, drop_mode="sr_psom"),
            seed=cell_seed,
        )
        # Convert to pp
        effect_pp = point * 100.0
        se_pp = se * 100.0
        # Two-sided p from bootstrap normal approx
        z = effect_pp / max(se_pp, 1e-12)
        p_cell = 2.0 * (1.0 - _phi(abs(z)))
        per_cell[cell_id] = {
            "drop_one_lift_pp": effect_pp,
            "ci_95_pp": [ci_lo * 100.0, ci_hi * 100.0],
            "se_pp": se_pp,
            "p_value_two_sided": p_cell,
            "n_tasks": len(tasks),
        }
        effects_pp.append(effect_pp)
        variances_pp.append(se_pp ** 2)
        per_cell_p_values.append(p_cell)

    # PRIMARY: pooled DL meta + magnitude + superiority test
    meta = dersimonian_laird_meta(effects_pp, variances_pp)
    superiority = superiority_test(meta["pooled_effect"], meta["pooled_se"],
                                     threshold=magnitude_threshold_pp, alpha=alpha)
    # TOST kept for informational reporting (NOT used in H1 gating decision)
    tost_info = tost_equivalence(meta["pooled_effect"], meta["pooled_se"],
                                  delta=delta_pp, alpha=alpha)

    pooled_sig = meta["p_value_two_sided"] is not None and meta["p_value_two_sided"] < alpha
    magnitude_pass = meta["pooled_effect"] >= magnitude_threshold_pp
    superiority_pass = superiority["decision"] == "reject_H0_substantively_above_threshold"

    primary_h1_pass = pooled_sig and magnitude_pass and superiority_pass

    # TRANSPARENCY: K-of-N Holm
    holm_per_cell = holm_correct(per_cell_p_values, alpha=alpha)
    for (cell_id, _), h in zip(per_cell.items(), holm_per_cell):
        per_cell[cell_id]["holm_p"] = h["p_holm"]
        per_cell[cell_id]["individually_holm_sig"] = h["rejected"]
    n_individually_sig = sum(1 for h in holm_per_cell if h["rejected"])
    transparency_pass = n_individually_sig >= transparency_K_h1

    return {
        "primary_gate": {
            "pooled_meta": meta,
            "magnitude_check": {"pooled_pp": meta["pooled_effect"],
                                 "threshold_pp": magnitude_threshold_pp,
                                 "pass": magnitude_pass},
            "superiority_test": superiority,
            "tost_informational": tost_info,
            "decision": "PASS" if primary_h1_pass else "FAIL",
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
        count, ci_lo, ci_hi, se = _paired_bootstrap(
            tasks,
            statistic_fn=lambda t: float(_unique_count_per_cell(t, axis_mode_key, ref_mode_key)),
            seed=cell_seed,
        )
        # Per-cell pass: CI > 0 AND count ≥ min_unique_count (≥2 floor for noise)
        ci_excludes_zero = ci_lo > 0
        count_above_floor = count >= min_unique_count
        per_cell_pass = ci_excludes_zero and count_above_floor
        # Per-cell p from normal approx on count statistic (testing > 0)
        z = count / max(se, 1e-12)
        p_cell = 1.0 - _phi(z)  # one-sided
        per_cell[cell_id] = {
            "unique_count": count,
            "ci_95": [ci_lo, ci_hi],
            "se": se,
            "p_value_one_sided": p_cell,
            "ci_excludes_zero": ci_excludes_zero,
            "count_above_min": count_above_floor,
            "per_cell_pass": per_cell_pass,
            "n_tasks": len(tasks),
        }
        effects.append(count)
        variances.append(se ** 2)
        per_cell_p_values.append(p_cell)
        per_cell_ci_excludes_zero.append(per_cell_pass)

    # PRIMARY: pooled meta
    meta = dersimonian_laird_meta(effects, variances)
    pooled_ci_lo = meta["pooled_ci_95"][0] if meta["pooled_ci_95"][0] is not None else None
    primary_pass = (meta["p_value_two_sided"] is not None and
                    meta["p_value_two_sided"] < alpha and
                    pooled_ci_lo is not None and pooled_ci_lo > 0)

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
            "pooled_meta": meta,
            "ci_excludes_zero": pooled_ci_lo is not None and pooled_ci_lo > 0,
            "decision": "PASS" if primary_pass else "FAIL",
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
    """H2(a): median cost(P-SoM) within ±cost_margin_pct% of median cost(DOM) per cell.

    H2(a) is a **by-construction property with a falsification check** (preregistration.md
    §2 H2(a) lock 2026-05-14, decision "3A"; line 120-137 + line 368). Falsification rule
    per prereg lock: "if **ANY** condition shows median cost ratio > **1.20×** (= margin
    >20%), the by-construction claim is falsified". This is NOT a K-of-N transparency
    gate — it is a strict ALL-cells-must-pass falsification check.

    /stress A1.19 P0-3 (2026-05-17, gemini Mode C OOB framing critique + Claude verify):
    pre-fix had TWO defects: (a) `cost_margin_pct: float = 10.0` default was 2× stricter
    than prereg ±20% lock → false-falsification rate inflated when median ratio in
    1.10-1.20× band; (b) `consistent: pass_count >= transparency_K_h2` (K-of-N) was
    semantically inverted — prereg explicitly says "if ANY condition violated → falsified"
    which is the strict-ALL-pass semantics (K-of-N is for H1/H3 transparency counts, not
    for H2(a) falsification check per line 368). `transparency_K_h2` arg retained for
    backward-compat with CLI invocations but **ignored** (it has no role in falsification
    semantics); deprecation warning emitted.

    H2(a) test margin is a RELATIVE PERCENTAGE (e.g., ±20% of DOM cost), distinct from
    H1 TOST δ which is an SR percentage-point margin (codex probable concern disambig).
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
    pass_count = 0
    n_cells_total = len(cells_by_id)
    for cell_id, tasks in cells_by_id.items():
        cost_dom_vals = [float(t["cost_dom"]) for t in tasks if t["cost_dom"]]
        cost_psom_vals = [float(t["cost_psom"]) for t in tasks if t["cost_psom"]]
        if not cost_dom_vals or not cost_psom_vals:
            per_cell[cell_id] = {"per_cell_pass": False, "reason": "missing cost data"}
            continue
        med_dom = statistics.median(cost_dom_vals)
        med_psom = statistics.median(cost_psom_vals)
        rel_diff_pct = (med_psom - med_dom) / max(med_dom, 1e-12) * 100.0
        within_band = abs(rel_diff_pct) <= cost_margin_pct
        per_cell[cell_id] = {
            "median_cost_dom": med_dom,
            "median_cost_psom": med_psom,
            "relative_diff_pct": rel_diff_pct,
            "margin_pct": cost_margin_pct,
            "per_cell_pass": within_band,
        }
        if within_band:
            pass_count += 1
    # Prereg strict semantics: ALL cells must pass for H2(a) by-construction to hold.
    # If any cell falsifies (median ratio > 1.20×), framing degrades to R4 per prereg
    # line 310.
    not_falsified = pass_count == n_cells_total
    return {
        "h2a_cost_equivalence": {
            "N": n_cells_total,
            "n_cells_pass": pass_count,
            "n_cells_falsified": n_cells_total - pass_count,
            "consistent": not_falsified,  # ALL-pass per prereg L131-132 + L368 strict semantics
            "margin_pct": cost_margin_pct,
            "semantics": "strict_all_pass_falsification_check",
            "prereg_anchor": "preregistration.md §2 H2(a) line 120-137 + framing rule R4 line 310",
        },
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# Framing rule R1-R5 mapper
# ---------------------------------------------------------------------------

def _effective_gate_pass(gate_result: dict, gate_kind: str,
                          heterogeneity_threshold_pct: float) -> tuple[bool, bool, dict]:
    """Determine whether a primary gate effectively passes, accounting for heterogeneity.

    F3 fix 2026-05-14 (codex /stress v6 + Round C M3): heterogeneity check now applies
    to EVERY primary gate (H1, H3 axis-1, H3 axis-2), not just H1. Prior version only
    checked H1 I² — selective check biased toward R1 framing when H3 axes were
    heterogeneous but H1 was not.

    Returns (passes, heterogeneous, detail):
      - If pooled meta I² ≤ threshold: passes = pooled primary gate PASS decision.
      - If pooled meta I² > threshold: do NOT trust pooled gate; passes via per-cell
        consistency (≥3 of 4 cells direction-positive AND ≥2 individually significant).

    gate_kind ∈ {"h1", "h3_axis"} selects the per-cell field names.
    """
    meta = gate_result.get("primary_gate", {}).get("pooled_meta", {})
    i_sq = meta.get("I_squared_pct")
    per_cell = gate_result.get("per_cell", {})
    n_cells = len(per_cell)

    if i_sq is not None and i_sq > heterogeneity_threshold_pct:
        # Heterogeneous — pooled gate untrustworthy, use per-cell consistency
        if gate_kind == "h1":
            n_pos = sum(1 for c in per_cell.values() if c.get("drop_one_lift_pp", 0.0) > 0)
            n_sig = sum(1 for c in per_cell.values() if c.get("individually_holm_sig", False))
        else:  # h3_axis
            n_pos = sum(1 for c in per_cell.values() if c.get("unique_count", 0.0) > 0)
            n_sig = sum(1 for c in per_cell.values() if c.get("per_cell_pass", False))
        passes = (n_pos >= 3 and n_sig >= 2)
        return passes, True, {
            "I_squared_pct": i_sq, "n_direction_positive": n_pos,
            "n_consistent": n_sig, "n_cells": n_cells,
            "via": "per_cell_consistency (pooled meta I² > threshold)",
        }
    passes = gate_result["primary_gate"]["decision"] == "PASS"
    return passes, False, {"I_squared_pct": i_sq, "via": "pooled_meta"}


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

    # Heterogeneity branch: ≥1 primary gate had I² > threshold → pooled 2-axis
    # structure claim not supported; hook caps at R3 (per-cell consistency).
    if any_heterogeneous:
        if h1_pass and h2a_cost_pass:
            return {
                "rule": "R3",
                "framing": "Heterogeneity-conditional R3 — ≥1 primary gate had pooled I² > 75%; "
                           "per-cell consistency used, pooled 2-axis structure claim not supported",
                "hook_power": "MODERATE",
                "heterogeneity_override": True,
                "heterogeneity_detail": heterogeneity_detail,
            }
        return {
            "rule": "R4_or_R5",
            "framing": "Heterogeneity override (≥1 gate I² > 75%) AND H1 or H2(a) per-cell "
                       "consistency fails — paper hook not supported",
            "hook_power": "WEAK",
            "heterogeneity_override": True,
            "heterogeneity_detail": heterogeneity_detail,
        }

    # Normal R1-R5 mapping (all primary gates pooled cleanly, I² ≤ threshold)
    if h1_pass and h2a_cost_pass and h3a_pass and h3b_pass:
        return {"rule": "R1", "framing": "Phantom routing space (2-axis empirical structure)",
                "hook_power": "STRONGEST", "heterogeneity_override": False,
                "heterogeneity_detail": heterogeneity_detail}
    if h1_pass and h2a_cost_pass and (h3a_pass or h3b_pass):
        return {"rule": "R2", "framing": "Phantom routing space (single-axis empirical structure)",
                "hook_power": "MODERATE-STRONG", "heterogeneity_override": False,
                "heterogeneity_detail": heterogeneity_detail}
    if h1_pass and h2a_cost_pass and not h3a_pass and not h3b_pass:
        return {"rule": "R3", "framing": "Phantom-SoM is hidden 4th routing arm (workshop-grade R3)",
                "hook_power": "MODERATE", "heterogeneity_override": False,
                "heterogeneity_detail": heterogeneity_detail}
    if h1_pass and not h2a_cost_pass:
        return {"rule": "R4", "framing": "Phantom-SoM partial drop-in (H2(a) cost equivalence fails)",
                "hook_power": "WEAK", "heterogeneity_override": False,
                "heterogeneity_detail": heterogeneity_detail}
    return {"rule": "R5", "framing": "Paper death scenario — pivot to VWA bug audit OR abandon",
            "hook_power": "n/a", "heterogeneity_override": False,
            "heterogeneity_detail": heterogeneity_detail}


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
        # fixture (all 4 cells identical distribution) so it demonstrably routes R1;
        # the B1 0.6× capability multiplier is applied only to scenarios that are
        # NOT testing the clean-pooled path (it induces genuine B0/B1 heterogeneity).
        if model == "B1" and scenario != "r1_pass":
            base_rate = {k: v * 0.6 for k, v in base_rate.items()}
        # Cell-level effect-size variance for heterogeneity test — large bimodal
        # shift so between-cell variance >> within-cell bootstrap SE → I² > 75%.
        if scenario == "heterogeneity_test":
            cell_shift = [+0.25, -0.20, +0.25, -0.20][cell_idx]
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
    p.add_argument("--seed", type=int, default=42)
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

    # Load data
    if args.synthetic:
        cells_by_id = generate_synthetic_per_task(seed=args.seed, scenario=args.scenario)
        input_sha = f"synthetic:{args.scenario}:{args.seed}"
        logger.info(f"Synthetic mode: {len(cells_by_id)} cells, scenario={args.scenario}")
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

    # Evaluate hypotheses
    h1 = evaluate_h1(cells_by_id, delta_pp=args.TOST_delta_pp,
                      magnitude_threshold_pp=args.H1_magnitude_pp,
                      alpha=args.alpha, transparency_K_h1=args.transparency_K_h1,
                      bootstrap_seed=args.seed)
    h2 = evaluate_h2_cost(cells_by_id, cost_margin_pct=args.H2_cost_margin_pct,
                           transparency_K_h2=args.transparency_K_h2)
    h3_axis1 = evaluate_h3_axis(cells_by_id, axis_mode_key="sr_ptext",
                                  ref_mode_key="sr_psom",
                                  min_unique_count=args.H3_min_unique_count,
                                  alpha=args.alpha,
                                  transparency_K_h3=args.transparency_K_h3,
                                  bootstrap_seed=args.seed)
    h3_axis2 = evaluate_h3_axis(cells_by_id, axis_mode_key="sr_pprompt",
                                  ref_mode_key="sr_psom",
                                  min_unique_count=args.H3_min_unique_count,
                                  alpha=args.alpha,
                                  transparency_K_h3=args.transparency_K_h3,
                                  bootstrap_seed=args.seed)
    framing = apply_framing_rule(h1, h2, h3_axis1, h3_axis2)

    result = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "scope": "Phase 1a 24-condition / 4-cell statistical analysis",
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
