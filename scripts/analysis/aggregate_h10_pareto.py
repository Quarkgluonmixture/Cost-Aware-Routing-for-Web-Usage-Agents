#!/usr/bin/env python3
"""H10 Pareto non-dominance operational deployment gate producer (B-1002, /stress A2.5 Chunk C;
/stress A2.8 prose alignment B-1550~B-1552 2026-05-18).

Reads Pass-1 baseline outcomes + Pass-2 learned-router outcomes per cell, computes:

OPERATIONAL DEPLOYMENT GATE (locked 2026-05-18 per /stress A2.8 P0-2-AB* + P0-3-A* +
P0-1-A* user-directive resolution — supersedes prior "PRIMARY (Q4=A) K-of-6 descriptive"
framing that contradicted preregistration §2.4 K-of-N transparency-only doctrine):
    Two-layer criterion. Cell-level (statistical): per cell, paired bootstrap (B=1000,
    seed=42) on (Cost, SR); cell passes if router Pareto non-dominated vs 5 single-mode
    baselines in >=95% bootstrap replicates. Grid-level (operational robustness): H10
    deployable iff >=5 of 6 pre-registered cells pass cell-level. The 5/6 threshold is
    a fixed-cell operational deployment criterion, NOT a binomial significance test
    (no across-cells alpha, no Type-I/II coupling). See preregistration.md §H10
    OPERATIONAL DEPLOYMENT GATE for the full two-layer specification.

APPENDIX-D SENSITIVITY: continuous θ_i = SR_router_i - max_m SR_baseline_m_i  (subject to
                Cost_baseline_m_i ≤ Cost_router_i Pareto feasibility) + FE inverse-variance
                pool over 6 cells (mirrors H1 §2.5 estimand structure). FE pool is a
                transparency row, NOT the operational gate.

Site-asymmetric viability is a pre-hoc theoretical prediction (preregistration §H10
hypothesis prose — NOT archive-simulation-derived per /stress A2.8 P0-3-A* archive-
deletion-from-prereg-justification 2026-05-18): visual-rich classifieds cells expected
to pass cell-level via task-conditional routing benefit; text-dominated reddit cells
expected to collapse toward always_phantom_som baseline. Phase 1a clean-rerun is the
falsification test.

Output:
  results/phantom_paper/h10_pareto_verdict.csv  # per-cell tabular
  results/phantom_paper/h10_pareto_verdict.md   # human-readable summary
  results/phantom_paper/h10_pareto_verdict.json # machine-readable raw stats

Usage:
    python3 scripts/analysis/aggregate_h10_pareto.py --all
    python3 scripts/analysis/aggregate_h10_pareto.py --pass1-glob 'results/visualwebarena/phase1/B*_*'
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

from p79.policies.pass1_manifest import discover_runs

REPO = Path(__file__).resolve().parents[2]
PHASE1_ROOT = REPO / "results/visualwebarena/phase1"
OUT_DIR = REPO / "results/phantom_paper"
# P0-2 (AMENDMENT_04 H10 entropy DEFER gate): train_l1_router.main() emits the
# train-fold label-entropy artifact here; run_h10_verdict reads it to DEFER H10
# (prereg §H10 L238-240) when any required cell concentrates on ≤ 2 modes.
ROUTER_ARTIFACT_DIR = REPO / "results/phantom_paper/l1_router"

# AMENDMENT_01 (2026-05-21): H10 Pareto Cost-axis = total_billed_cost (within-cell,
# consistent with §1/§4). Fail closed if an episode summary lacks total_billed; legacy
# archive vintages can opt in to the old total_cost_usd via P79_ALLOW_LEGACY_COST=1.
_LEGACY_COST_OK = os.environ.get("P79_ALLOW_LEGACY_COST", "0") == "1"

# Phase 1a 6 cells
CELLS = [
    ("B0", "classifieds"), ("B0", "reddit"),
    ("B1", "classifieds"), ("B1", "reddit"),
    ("B2", "classifieds"), ("B2", "reddit"),
]

# 5 single-mode baselines (P-prompt excluded per preregistration:199 line 204 —
# cls archive aborted at 4 ep; baseline set expands to 6 if Phase 1a B0+B1+B2 cls
# all produce ≥50 ep P-prompt outcomes).
SINGLE_MODE_BASELINES = ["dom", "som", "vision", "phantom_text", "phantom_som"]
# C7 (B-1820): phantom_prompt is in the router candidate_modes (yaml) + oracle label
# space but excluded from the fixed baseline set (prereg: cls P-prompt archive aborted
# at 4 ep). If it gathers >= this many Pass-1 episodes it is added as a 6th baseline so
# the router cannot be credited for a phantom_prompt arm with no always-phantom_prompt
# comparison. Implements the prereg "baseline set expands to 6 if >=50 ep P-prompt".
PHANTOM_PROMPT_BASELINE_MIN_EP = 50

# Router condition_id (single per-cell Pass-2 fire)
ROUTER_CONDITION_PATTERN = "phase1_learned_router_"  # cond_id prefix for Pass-2

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
PARETO_NON_DOMINANCE_THRESHOLD = 0.95  # 95% paired bootstrap support per cell
DELTA_PP = 1.0  # H1-mirror δ for Appendix-D FE pool sensitivity row only (NOT the
# operational deployment gate). The operational gate uses cell-level paired-bootstrap
# Pareto non-dominance + grid-level >=5/6 robustness criterion, neither of which uses
# delta_pp as a threshold. See preregistration.md §H10 OPERATIONAL DEPLOYMENT GATE.
SCHEMA_VERSION = "2026-05-18-a2.5-chunk-c-h10"


def find_pass1_run_dirs(baseline: str, site: str) -> list[Path]:
    """Discover canonical Pass-1 baseline run dirs for (baseline, site) cell.

    C2 (B-1810): shared manifest-aware discovery (rejects smoke/test/debug runs; uses
    the manifest whitelist when present) — was a bare glob excluding only router_learned,
    which folded smoke / partial / stale runs into the H10 estimand.
    """
    runs, _ = discover_runs(PHASE1_ROOT, baseline, site, router=False)
    return runs


def find_pass2_router_dirs(baseline: str, site: str) -> list[Path]:
    """Discover canonical Pass-2 router run dirs (C2 B-1810: shared discovery)."""
    runs, _ = discover_runs(PHASE1_ROOT, baseline, site, router=True)
    return runs


def collect_per_task_outcomes_with_metrics(
    run_dirs: list[Path],
    site: str,
    cond_prefix_filter: Optional[str] = None,
) -> dict[int, dict[str, dict[str, Any]]]:
    """Read per-task SR + Cost + Latency from condition_summary_v2.json episodes.

    Returns {task_id: {mode: {success, cost_usd, latency_ms}, ...}}.
    cond_prefix_filter: only include conditions whose cond_id starts with this prefix
                       (e.g. "phase1_learned_router_" for Pass-2).
    """
    matrix: dict[int, dict[str, dict[str, Any]]] = {}
    for run_dir in run_dirs:
        for cond_dir in run_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            cond_id = cond_dir.name
            if cond_prefix_filter and not cond_id.startswith(cond_prefix_filter):
                continue
            if (not cond_prefix_filter) and cond_id.startswith("phase1_learned_router"):
                continue
            parts = cond_id.split("_")
            if len(parts) < 3 or parts[0] != "phase1":
                continue
            mode_tokens = parts[1:-2]
            mode = "_".join(mode_tokens)
            if mode == "phantom_dom":
                mode = "phantom_text"

            ep_dir = cond_dir / "episodes"
            if not ep_dir.is_dir():
                continue
            for summary_f in ep_dir.glob(f"{site}_task_*_summary_v2.json"):
                try:
                    rec = json.loads(summary_f.read_text())
                except (OSError, json.JSONDecodeError):
                    continue
                tid = int(rec["task_id"])
                success = bool(rec.get("success", False))
                # AMENDMENT_01 (2026-05-21): H10 Pareto Cost-axis = total_billed_cost
                # (within-cell, consistent with §1/§4). Fail closed if a paper-grade
                # episode lacks total_billed; legacy total_cost_usd / cost_usd.total only
                # in legacy mode (P79_ALLOW_LEGACY_COST=1 for archive vintages).
                cost = rec.get("total_billed_cost_usd")
                if cost is None:
                    if _LEGACY_COST_OK:
                        cost = rec.get("total_cost_usd")
                        if cost is None:
                            cost = (rec.get("cost_usd") or {}).get("total", 0.0)
                    else:
                        raise RuntimeError(
                            f"H10 Pareto cost-axis: {summary_f} lacks total_billed_cost_usd "
                            "(AMENDMENT_01 H10 cost-axis = total_billed). Set "
                            "P79_ALLOW_LEGACY_COST=1 for archive vintages, or re-run the "
                            "condition to emit the canonical billed-cost field."
                        )
                latency = rec.get("total_latency_ms")
                if latency is None:
                    latency = (rec.get("latency_ms") or {}).get("total", 0.0)
                # B-1601 (/stress 深入审 Mode A P0-2-A*, 2026-05-18): capture
                # `cost_unit_basis` per-episode for downstream cell-level
                # homogeneity check + cross-cell FE-pool basis-mix detection.
                # Pre-fix: this script entirely missed the cost_unit_basis
                # propagation that A2.7 B-1409 wired into
                # `aggregate_cross_site.py` — sibling-script propagation gap.
                # B0 cells = "api_usd" (proxy `usage.cost`), B1/B2 cells =
                # "electricity_usd_derived" (NVML wallclock × $/kWh); 1000×
                # scale gap. Within-cell legacy "unknown" rows would silently
                # corrupt per-cell paired-bootstrap cost-feasibility filter
                # (`b_cost <= r_cost`); cross-cell appendix FE-pool of any
                # cost-derived statistic would mix bases.
                cost_unit_basis = rec.get("cost_unit_basis", "unknown")
                matrix.setdefault(tid, {})[mode] = {
                    "success": int(success),
                    "cost_usd": float(cost or 0.0),
                    "latency_ms": float(latency or 0.0),
                    "cost_unit_basis": str(cost_unit_basis) if cost_unit_basis else "unknown",
                }
    return matrix


def _compute_modal_basis(
    outcomes: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """B-1601 (/stress 深入审 Mode A P0-2-A*, 2026-05-18): compute modal
    `cost_unit_basis` across all (task, mode) entries in a per-cell outcome
    matrix + flag heterogeneity. Returns:

        {modal_basis: str, basis_counts: {basis: int}, homogeneous: bool,
         minority_rows: int, total_rows: int}

    `homogeneous=False` when minority basis count > 0 (any non-modal record).
    Within-cell `cost_unit_basis` mixing breaks per-cell paired-bootstrap
    cost-feasibility filter at borderline tasks — under paper_grade gate this
    should fail-loud, but under default behaviour we surface the diagnostic
    and let the caller decide.
    """
    basis_counts: dict[str, int] = {}
    total = 0
    for _tid, modes in outcomes.items():
        for _mode, m in modes.items():
            basis = m.get("cost_unit_basis", "unknown")
            basis_counts[basis] = basis_counts.get(basis, 0) + 1
            total += 1
    if not basis_counts:
        return {
            "modal_basis": "unknown",
            "basis_counts": {},
            "homogeneous": True,
            "minority_rows": 0,
            "total_rows": 0,
        }
    # Modal = highest-count basis; ties broken alphabetically deterministic
    modal_basis = max(basis_counts.items(), key=lambda kv: (kv[1], -ord(kv[0][0]) if kv[0] else 0))[0]
    minority_rows = total - basis_counts.get(modal_basis, 0)
    return {
        "modal_basis": modal_basis,
        "basis_counts": basis_counts,
        "homogeneous": minority_rows == 0,
        "minority_rows": minority_rows,
        "total_rows": total,
    }


def aggregate_arm_metrics(
    task_outcomes: dict[int, dict[str, dict[str, Any]]],
    arm: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Extract per-task (success, cost, latency) arrays for one arm/mode.

    Returns (success_arr, cost_arr, latency_arr, task_ids) where task_ids is the
    sorted list of tasks that have data for this arm.
    """
    rows = []
    for tid in sorted(task_outcomes.keys()):
        modes = task_outcomes[tid]
        if arm not in modes:
            continue
        m = modes[arm]
        rows.append((tid, m["success"], m["cost_usd"], m["latency_ms"]))
    if not rows:
        return np.array([]), np.array([]), np.array([]), []
    arr = np.array(rows, dtype=float)
    task_ids = [int(r[0]) for r in rows]
    return arr[:, 1], arr[:, 2], arr[:, 3], task_ids


def paired_bootstrap_arm_metrics(
    success: np.ndarray,
    cost: np.ndarray,
    B: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute mean SR + Cost per bootstrap replicate (task-level resampling).

    Returns {sr_mean, sr_ci, cost_mean, cost_ci, sr_replicates, cost_replicates}.
    """
    rng = np.random.default_rng(seed)
    n = len(success)
    if n == 0:
        return {
            "n": 0, "sr_mean": float("nan"), "sr_ci": (float("nan"),) * 2,
            "cost_mean": float("nan"), "cost_ci": (float("nan"),) * 2,
            "sr_replicates": [], "cost_replicates": [],
        }
    sr_reps = []
    cost_reps = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        sr_reps.append(float(success[idx].mean()))
        cost_reps.append(float(cost[idx].mean()))
    return {
        "n": n,
        "sr_mean": float(success.mean()),
        "sr_ci": (
            float(np.percentile(sr_reps, 2.5)),
            float(np.percentile(sr_reps, 97.5)),
        ),
        "cost_mean": float(cost.mean()),
        "cost_ci": (
            float(np.percentile(cost_reps, 2.5)),
            float(np.percentile(cost_reps, 97.5)),
        ),
        "sr_replicates": sr_reps,
        "cost_replicates": cost_reps,
    }


def check_pareto_non_dominance_paired_bootstrap(
    router_success: np.ndarray,
    router_cost: np.ndarray,
    baseline_metrics: dict[str, dict[str, np.ndarray]],
    common_task_ids: list[int],
    B: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Per-cell paired bootstrap: for each task, resample → compute (Cost, SR) for
    router + each baseline → check Pareto non-dominance.

    Pareto non-dominance: NO baseline (Cost_b, SR_b) satisfies Cost_b ≤ Cost_router
    AND SR_b ≥ SR_router with at least one strict.

    Returns dict with fraction of bootstrap reps where router is non-dominated,
    pass = fraction > PARETO_NON_DOMINANCE_THRESHOLD.
    """
    rng = np.random.default_rng(seed)
    n = len(common_task_ids)
    if n == 0:
        return {
            "n_common_tasks": 0,
            "fraction_non_dominated": float("nan"),
            "pass_threshold": PARETO_NON_DOMINANCE_THRESHOLD,
            "passes": False,
            "reason": "no_common_tasks",
        }

    non_dominated_count = 0
    delta_max_replicates = []  # SR_router - max_feasible_SR_baseline (continuous θ for APPENDIX FE)

    for _ in range(B):
        idx = rng.integers(0, n, n)
        r_sr = float(router_success[idx].mean())
        r_cost = float(router_cost[idx].mean())
        max_feasible_sr = -float("inf")
        is_dominated = False
        for arm, m in baseline_metrics.items():
            b_sr = float(m["success"][idx].mean())
            b_cost = float(m["cost"][idx].mean())
            # Pareto domination: arm dominates router if cost ≤ router AND sr ≥ router (strict ≥1)
            if (b_cost <= r_cost) and (b_sr >= r_sr) and ((b_cost < r_cost) or (b_sr > r_sr)):
                is_dominated = True
                break
            # Track max-feasible baseline SR (for θ_i computation)
            if b_cost <= r_cost and b_sr > max_feasible_sr:
                max_feasible_sr = b_sr
        if not is_dominated:
            non_dominated_count += 1
        # θ for this replicate = r_sr - max_feasible_baseline_sr
        if max_feasible_sr > -float("inf"):
            delta_max_replicates.append(r_sr - max_feasible_sr)

    frac = non_dominated_count / B
    passes = frac >= PARETO_NON_DOMINANCE_THRESHOLD
    theta_mean = float(np.mean(delta_max_replicates)) if delta_max_replicates else float("nan")
    theta_ci = (
        float(np.percentile(delta_max_replicates, 2.5)),
        float(np.percentile(delta_max_replicates, 97.5)),
    ) if delta_max_replicates else (float("nan"), float("nan"))
    theta_se = float(np.std(delta_max_replicates)) if delta_max_replicates else float("nan")

    # F7/G4 (B-1815): non-dominance is a low bar — a router that collapses to a single
    # baseline mode (e.g. always phantom_som) is non-dominated BY DEFINITION (it IS a
    # baseline) and would pass the gate while contributing zero learning. Add a
    # marginal-utility flag: the router is STRICTLY better than the best cost-feasible
    # baseline iff θ (= SR_router − max feasible baseline SR) has a 95% CI lower bound
    # > 0. This is the "the routing logic actually helped" signal §6 must report
    # alongside non-dominance, so a degenerate router cannot masquerade as success.
    router_strictly_better = bool(delta_max_replicates) and (theta_ci[0] > 0)
    return {
        "n_common_tasks": n,
        "fraction_non_dominated": frac,
        "pass_threshold": PARETO_NON_DOMINANCE_THRESHOLD,
        "passes": passes,
        "theta_mean_pp": theta_mean * 100,
        "theta_ci_95_pp": (theta_ci[0] * 100, theta_ci[1] * 100),
        "theta_se_pp": theta_se * 100,
        "router_strictly_better_than_envelope": router_strictly_better,
        "n_bootstrap": B,
        "reason": "ok",
    }


def fe_inverse_variance_pool(
    per_cell_theta: list[float],
    per_cell_se: list[float],
) -> dict[str, Any]:
    """FE inverse-variance pool over per-cell θ_i estimates (APPENDIX SENSITIVITY).

    Mirrors H1 estimand structure (preregistration §2.5 + §625).
    Returns {theta_pool, se_pool, ci_95, z_one_sided_vs_delta, p_one_sided}.
    """
    valid = [(t, s) for t, s in zip(per_cell_theta, per_cell_se)
             if not (np.isnan(t) or np.isnan(s)) and s > 0]
    if len(valid) < 2:
        return {
            "n_cells_pooled": len(valid),
            "theta_pool_pp": float("nan"),
            "se_pool_pp": float("nan"),
            "ci_95_pp": (float("nan"), float("nan")),
            "z_vs_delta_1pp": float("nan"),
            "p_one_sided": float("nan"),
            "reason": "insufficient_cells_for_pool",
        }
    theta_arr = np.array([t for t, _ in valid])
    se_arr = np.array([s for _, s in valid])
    var_arr = se_arr ** 2
    weights = 1.0 / var_arr
    theta_pool = float(np.sum(weights * theta_arr) / np.sum(weights))
    se_pool = float(np.sqrt(1.0 / np.sum(weights)))
    ci_low = theta_pool - 1.96 * se_pool
    ci_high = theta_pool + 1.96 * se_pool
    z = (theta_pool - DELTA_PP) / se_pool if se_pool > 0 else float("nan")
    # One-sided superiority p-value (H0: θ_pool ≤ +δ)
    from scipy.stats import norm
    p_one_sided = float(1.0 - norm.cdf(z))
    return {
        "n_cells_pooled": len(valid),
        "theta_pool_pp": theta_pool,
        "se_pool_pp": se_pool,
        "ci_95_pp": (ci_low, ci_high),
        "z_vs_delta_1pp": float(z),
        "p_one_sided": p_one_sided,
        "delta_pp": DELTA_PP,
        "reason": "ok",
    }


def analyze_cell(
    baseline: str, site: str, require_full_coverage: bool = False
) -> dict[str, Any]:
    """Per-cell H10 Pareto analysis: load Pass-1 + Pass-2, paired bootstrap, verdict.

    require_full_coverage (C8 B-1811): when True, fail-closed if the router/baseline
    task intersection is not the full router task set (paper-grade); otherwise warn and
    proceed on the intersection subset (dev / partial-fire inspection).
    """
    cell_id = f"{baseline}_{site}"
    print(f"\n=== {cell_id} ===")

    pass1_runs = find_pass1_run_dirs(baseline, site)
    pass2_runs = find_pass2_router_dirs(baseline, site)
    print(f"  Pass-1 runs: {len(pass1_runs)}; Pass-2 runs: {len(pass2_runs)}")

    if not pass1_runs:
        return {
            "cell_id": cell_id,
            "status": "no_pass1_runs",
            "passes": False,
        }
    if not pass2_runs:
        return {
            "cell_id": cell_id,
            "status": "no_pass2_runs",
            "n_pass1_runs": len(pass1_runs),
            "passes": False,
        }

    pass1_outcomes = collect_per_task_outcomes_with_metrics(pass1_runs, site)
    pass2_outcomes = collect_per_task_outcomes_with_metrics(
        pass2_runs, site, cond_prefix_filter=ROUTER_CONDITION_PATTERN
    )
    print(f"  Pass-1 tasks with outcomes: {len(pass1_outcomes)}")
    print(f"  Pass-2 tasks with outcomes: {len(pass2_outcomes)}")

    # B-1601 (/stress 深入审 Mode A P0-2-A*, 2026-05-18): per-cell modal
    # cost_unit_basis + within-cell homogeneity diagnostic. Within-cell
    # mixed basis (e.g., paper-grade rerun episodes pooled with legacy
    # "unknown" archive episodes) silently breaks paired-bootstrap cost
    # feasibility filter `b_cost <= r_cost` at borderline tasks. Diagnostic
    # surfaces basis_counts so caller can decide whether to (a) re-filter
    # archive rows, (b) flip P79_PAPER_GRADE=1 hard-fail, or (c) accept
    # the warning row for non-paper-grade development runs.
    cell_basis = _compute_modal_basis({**pass1_outcomes, **pass2_outcomes})
    print(
        f"  cell_cost_unit_basis: modal={cell_basis['modal_basis']} "
        f"homogeneous={cell_basis['homogeneous']} "
        f"counts={cell_basis['basis_counts']}"
    )

    # Extract router metrics: Pass-2 has a single condition "phase1_learned_router_<N>"
    # Look for the "learned" mode key (sentinel; actual routed mode varies per task)
    router_success: list[int] = []
    router_cost: list[float] = []
    router_latency: list[float] = []
    router_task_ids: list[int] = []
    for tid in sorted(pass2_outcomes.keys()):
        modes = pass2_outcomes[tid]
        # Router runs under "learned" mode label OR routed mode label
        # Take any mode present (typically only one entry per task in Pass-2)
        if not modes:
            continue
        mode_entries = list(modes.values())
        if not mode_entries:
            continue
        # Combine: if multiple modes per task (shouldn't happen for router), take first
        m = mode_entries[0]
        router_success.append(m["success"])
        router_cost.append(m["cost_usd"])
        router_latency.append(m["latency_ms"])
        router_task_ids.append(tid)
    router_success_arr = np.array(router_success, dtype=float)
    router_cost_arr = np.array(router_cost, dtype=float)
    router_latency_arr = np.array(router_latency, dtype=float)

    # Per-arm baseline metrics (Pass-1)
    baseline_metrics: dict[str, dict[str, np.ndarray]] = {}
    baseline_paired_summaries: dict[str, dict[str, Any]] = {}
    # C7 (B-1820): expand the baseline set to include phantom_prompt when it has enough
    # Pass-1 data (prereg conditional). The router can route to phantom_prompt, so
    # without this an always-phantom_prompt baseline that might dominate the router would
    # never be tested → H10 could over-credit the router.
    arms = list(SINGLE_MODE_BASELINES)
    pp_s, _, _, _ = aggregate_arm_metrics(pass1_outcomes, "phantom_prompt")
    phantom_prompt_in_baselines = len(pp_s) >= PHANTOM_PROMPT_BASELINE_MIN_EP
    if phantom_prompt_in_baselines and "phantom_prompt" not in arms:
        arms = arms + ["phantom_prompt"]
        print(
            f"  C7: phantom_prompt has {len(pp_s)} ep >= {PHANTOM_PROMPT_BASELINE_MIN_EP} "
            f"→ added as 6th baseline arm"
        )
    for arm in arms:
        s, c, l, tids = aggregate_arm_metrics(pass1_outcomes, arm)
        if len(s) == 0:
            print(f"  arm={arm}: NO DATA, skipping")
            continue
        # Align to router task set (intersection only)
        common = set(router_task_ids).intersection(set(tids))
        if not common:
            print(f"  arm={arm}: NO common task overlap with router, skipping")
            continue
        arm_tid_to_idx = {t: i for i, t in enumerate(tids)}
        router_tid_to_idx = {t: i for i, t in enumerate(router_task_ids)}
        common_sorted = sorted(common)
        arm_idx = [arm_tid_to_idx[t] for t in common_sorted]
        baseline_metrics[arm] = {
            "success": s[arm_idx],
            "cost": c[arm_idx],
            "latency": l[arm_idx],
            "task_ids": common_sorted,
        }
        baseline_paired_summaries[arm] = paired_bootstrap_arm_metrics(
            s[arm_idx], c[arm_idx]
        )
        print(
            f"  arm={arm}: n_common={len(common_sorted)}, "
            f"SR={baseline_paired_summaries[arm]['sr_mean']:.3f}, "
            f"Cost={baseline_paired_summaries[arm]['cost_mean']:.4f}"
        )

    if not baseline_metrics:
        return {
            "cell_id": cell_id,
            "status": "no_baseline_arms_with_data",
            "passes": False,
        }

    # Intersect to common task set across router + all baselines
    common_tasks = set(router_task_ids)
    for arm, m in baseline_metrics.items():
        common_tasks = common_tasks.intersection(set(m["task_ids"]))
    common_sorted = sorted(common_tasks)
    if not common_sorted:
        return {
            "cell_id": cell_id,
            "status": "no_common_tasks_router_vs_baselines",
            "passes": False,
        }

    # C8 (B-1811): coverage disclosure + paper-grade fail-closed. The H10 estimand is
    # the router's task set; silently intersecting down to whatever every arm happens to
    # have completed turns a partial fire into a biased small-subset verdict that still
    # reports "passes". Record coverage; under require_full_coverage refuse to emit a
    # verdict on an incomplete intersection.
    router_set = set(router_task_ids)
    coverage = {
        "n_router_tasks": len(router_set),
        "n_common": len(common_tasks),
        "common_fraction": (len(common_tasks) / len(router_set)) if router_set else 0.0,
        "per_arm": {
            arm: {
                "n_tasks": len(set(m["task_ids"])),
                "n_missing_vs_router": len(router_set - set(m["task_ids"])),
            }
            for arm, m in baseline_metrics.items()
        },
    }
    coverage_incomplete = coverage["n_common"] < coverage["n_router_tasks"] or any(
        a["n_missing_vs_router"] > 0 for a in coverage["per_arm"].values()
    )
    if coverage_incomplete and require_full_coverage:
        return {
            "cell_id": cell_id,
            "status": "incomplete_coverage_paper_grade",
            "passes": False,
            "coverage": coverage,
            "missing_per_arm": {
                arm: sorted(router_set - set(m["task_ids"]))[:50]
                for arm, m in baseline_metrics.items()
                if router_set - set(m["task_ids"])
            },
        }
    if coverage_incomplete:
        print(
            f"  ⚠️  C8 coverage incomplete: common={coverage['n_common']}/"
            f"{coverage['n_router_tasks']} router tasks "
            f"(fraction={coverage['common_fraction']:.2f}); verdict is on the "
            f"intersection subset — pass --require-full-coverage for paper-grade."
        )

    # Re-align all arms to common task set
    router_tid_to_idx = {t: i for i, t in enumerate(router_task_ids)}
    r_idx = [router_tid_to_idx[t] for t in common_sorted]
    router_success_common = router_success_arr[r_idx]
    router_cost_common = router_cost_arr[r_idx]
    aligned_baseline = {}
    for arm, m in baseline_metrics.items():
        m_tid_to_idx = {t: i for i, t in enumerate(m["task_ids"])}
        b_idx = [m_tid_to_idx[t] for t in common_sorted]
        aligned_baseline[arm] = {
            "success": m["success"][b_idx],
            "cost": m["cost"][b_idx],
        }

    # Router paired bootstrap summary (for descriptive)
    router_paired = paired_bootstrap_arm_metrics(router_success_common, router_cost_common)
    print(
        f"  router: n_common={len(common_sorted)}, "
        f"SR={router_paired['sr_mean']:.3f}, Cost={router_paired['cost_mean']:.4f}"
    )

    # Pareto non-dominance test
    pareto = check_pareto_non_dominance_paired_bootstrap(
        router_success_common, router_cost_common, aligned_baseline, common_sorted
    )
    print(
        f"  Pareto: fraction_non_dominated={pareto['fraction_non_dominated']:.3f} "
        f"(threshold {PARETO_NON_DOMINANCE_THRESHOLD}) → "
        f"{'PASS' if pareto['passes'] else 'FAIL'}"
    )
    print(f"  θ_pp={pareto['theta_mean_pp']:.2f} [CI {pareto['theta_ci_95_pp'][0]:.2f}, {pareto['theta_ci_95_pp'][1]:.2f}]")

    return {
        "cell_id": cell_id,
        "baseline": baseline,
        "site": site,
        "status": "ok",
        "n_common_tasks": len(common_sorted),
        "coverage": coverage,  # C8 (B-1811): estimand coverage disclosure
        "phantom_prompt_in_baselines": phantom_prompt_in_baselines,  # C7 (B-1820)
        "router_sr_mean": router_paired["sr_mean"],
        "router_sr_ci_95": router_paired["sr_ci"],
        "router_cost_mean": router_paired["cost_mean"],
        "router_cost_ci_95": router_paired["cost_ci"],
        "baseline_paired_summaries": baseline_paired_summaries,
        "pareto_non_dominance": pareto,
        "passes": pareto["passes"],
        "theta_mean_pp": pareto["theta_mean_pp"],
        "theta_se_pp": pareto["theta_se_pp"],
        # F7/G4 (B-1815): marginal-utility flag — router strictly > best feasible
        # baseline (θ CI lower bound > 0), NOT merely non-dominated.
        "router_strictly_better": pareto.get("router_strictly_better_than_envelope", False),
        # B-1601 (/stress 深入审 Mode A P0-2-A*, 2026-05-18): cell-level
        # cost_unit_basis diagnostic for downstream cross-cell pool homogeneity
        # check in run_h10_verdict (paper §6 Appendix-D FE-pool transparency).
        "cell_cost_unit_basis": cell_basis,
    }


def _load_h10_entropy_gate() -> Optional[dict]:
    """Read the H10 train-fold label-entropy DEFER artifact (train_l1_router output).

    Returns the parsed `h10_entropy_gate.json` (emitted by `train_l1_router.main`) or
    None if absent/unreadable. `run_h10_verdict` fail-closes H10 deployability when this
    is None OR when `h10_entropy_gate_passed` is False (prereg §H10 L238-240 DEFER).
    """
    path = ROUTER_ARTIFACT_DIR / "h10_entropy_gate.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_h10_verdict(
    cells: Optional[list[tuple[str, str]]] = None,
    require_full_coverage: bool = False,
) -> dict[str, Any]:
    """Top-level H10 verdict: per-cell analysis + K-of-6 PRIMARY + FE pool APPENDIX."""
    cells = cells or CELLS
    per_cell_results = {}
    for baseline, site in cells:
        rec = analyze_cell(baseline, site, require_full_coverage=require_full_coverage)
        per_cell_results[rec["cell_id"]] = rec

    # Operational deployment gate (two-layer: cell-level + grid-level)
    # /stress A2.8 B-1551 — renamed from "K-of-6 PRIMARY descriptive verdict (Q4=A)"
    # to "operational_deployment_gate" per user-directive operational-gate-not-significance-test reframing.
    ok_cells = [r for r in per_cell_results.values() if r["status"] == "ok"]
    k_pass = sum(1 for r in ok_cells if r["passes"])
    n_total = len(ok_cells)
    operational_gate = {
        "estimand": (
            "Two-layer operational deployment gate (preregistration §H10 OPERATIONAL "
            "DEPLOYMENT GATE locked 2026-05-18): cell-level statistical = per-cell "
            "paired bootstrap (B=1000, seed=42) on (Cost, SR), cell passes if router "
            "Pareto non-dominated vs 5 single-mode baselines in >=95% bootstrap "
            "replicates; grid-level operational robustness = H10 deployable iff "
            ">=5 of 6 pre-registered cells pass cell-level. The 5/6 threshold is a "
            "fixed-cell operational deployment criterion for the engineering "
            "deployability claim, NOT a binomial significance test over a population "
            "of cells (no across-cells alpha, no Type-I/II coupling)."
        ),
        "n_cells_with_data": n_total,
        "k_cells_passing_cell_level": k_pass,
        "k_of_n_string": f"{k_pass}/{n_total}",
        "deployment_threshold": ">= 5/6 cells pass cell-level (operational robustness criterion)",
        "operational_gate_passed": (k_pass >= 5 and n_total >= 6),
        # F7/G4 (B-1815): marginal-utility layer — how many cells show the router
        # STRICTLY beating the best feasible single-mode baseline (θ CI lower > 0),
        # not merely non-dominated. High non-dominance count + low strictly-better
        # count = router collapsing toward a baseline (learning illusion).
        "k_cells_router_strictly_better": sum(
            1 for r in ok_cells if r.get("router_strictly_better")
        ),
        "marginal_utility_note": (
            "Non-dominance only proves 'not worse'; router_strictly_better (θ CI lower "
            "bound > 0) proves the routing logic added value over the best single mode. "
            "If k_cells_router_strictly_better << k_cells_passing_cell_level, the router "
            "is largely reproducing a single-mode baseline (e.g. always phantom_som) and "
            "§6 must not claim a learned-routing benefit."
        ),
    }

    # P0-2 (AMENDMENT_04 H10 entropy DEFER gate, prereg §H10 L238-240): read the
    # train-fold label-entropy artifact (train_l1_router output). If any required cell's
    # min-over-folds entropy < 1.0 bit (labels concentrate on ≤ 2 modes), H10 is DEFERRED
    # to §5 descriptive — operational_gate_passed is suppressed regardless of Pareto
    # non-dominance, because a router collapsing to a single mode can pass non-dominance
    # while having no learnable routing policy. Fail-closed: a MISSING entropy artifact
    # also suppresses deployability (no entropy verdict ⇒ no deployability claim). The
    # pre-entropy Pareto verdict is retained (operational_gate_passed_pre_entropy).
    _entropy_gate = _load_h10_entropy_gate()
    if _entropy_gate is None:
        operational_gate["operational_gate_passed_pre_entropy"] = (
            operational_gate["operational_gate_passed"]
        )
        operational_gate["operational_gate_passed"] = False
        operational_gate["h10_status"] = "entropy_unavailable"
        operational_gate["entropy_defer_reason"] = (
            "H10 deployability NOT claimable: entropy gate artifact "
            "results/phantom_paper/l1_router/h10_entropy_gate.json absent (fail-closed "
            "per prereg §H10 — run train_l1_router.main() to emit it; no entropy verdict "
            "means no deployability claim)."
        )
        operational_gate["h10_entropy_gate"] = {"h10_status": "unavailable"}
    else:
        operational_gate["h10_entropy_gate"] = _entropy_gate
        if not _entropy_gate.get("h10_entropy_gate_passed", False):
            operational_gate["operational_gate_passed_pre_entropy"] = (
                operational_gate["operational_gate_passed"]
            )
            operational_gate["operational_gate_passed"] = False
            operational_gate["h10_status"] = "deferred_entropy"
            operational_gate["entropy_defer_reason"] = (
                "H10 DEFERRED to §5 descriptive: train-fold label entropy < 1.0 bit on "
                f"cells {_entropy_gate.get('low_entropy_cells')} (global_min="
                f"{_entropy_gate.get('global_entropy_min_bits')} bits). Pareto "
                "non-dominance suppressed per prereg §H10 L238-240 DEFER condition."
            )
        else:
            operational_gate["h10_status"] = "ok"

    # APPENDIX-D SENSITIVITY: FE inverse-variance pool over θ_i (transparency, NOT gating)
    thetas = [r["theta_mean_pp"] for r in ok_cells]
    ses = [r["theta_se_pp"] for r in ok_cells]
    fe_pool = fe_inverse_variance_pool(thetas, ses)
    fe_pool["estimand"] = (
        "Continuous θ_i = SR_router - max-feasible-baseline-SR per cell, "
        "FE inverse-variance pool over 6 cells (Appendix-D sensitivity row, "
        "H1-mirror estimand parallelism — NOT the operational deployment gate; "
        "operational gate is the two-layer cell-level + grid-level criterion above)."
    )

    # B-1601 (/stress 深入审 Mode A P0-2-A*, 2026-05-18): cross-cell
    # cost_unit_basis homogeneity diagnostic on FE pool. θ_i is a SR delta
    # (pp, dimensionless) so basis-mixing does NOT directly corrupt pool
    # arithmetic, BUT (a) each θ_i's `max_feasible_baseline_SR` calculation
    # used per-cell cost-feasibility filter `b_cost <= r_cost` whose ordering
    # depends on basis-consistent cost magnitudes — within-cell unknown-row
    # mix is the actual risk vector, (b) any downstream paper §6 prose
    # quoting "average router cost across cells" would silently mix
    # api_usd ($0.005/1K tok) with electricity_usd_derived ($0.0000005/1K tok)
    # at ~1000× scale gap. Emit warning fields so paper §6 figure script +
    # OSF artifact replay can stratify.
    cell_basis_summary: dict[str, Any] = {
        "per_cell": {
            r["cell_id"]: r.get("cell_cost_unit_basis", {})
            for r in ok_cells
        },
        "cells_with_homogeneous_basis": sum(
            1 for r in ok_cells
            if r.get("cell_cost_unit_basis", {}).get("homogeneous", True)
        ),
        "cells_with_mixed_basis": sum(
            1 for r in ok_cells
            if not r.get("cell_cost_unit_basis", {}).get("homogeneous", True)
        ),
        "modal_bases_distinct": sorted({
            r.get("cell_cost_unit_basis", {}).get("modal_basis", "unknown")
            for r in ok_cells
        }),
    }
    cell_basis_summary["pool_basis_homogeneous"] = (
        len([b for b in cell_basis_summary["modal_bases_distinct"] if b != "unknown"]) <= 1
    )
    if not cell_basis_summary["pool_basis_homogeneous"]:
        cell_basis_summary["pool_warning"] = (
            "FE pool aggregates θ_i (SR delta, pp) across cells with mixed "
            "cost_unit_basis: " + ", ".join(cell_basis_summary["modal_bases_distinct"])
            + ". θ_i pooling is SR-based and dimensionally sound, BUT any "
            "downstream paper §6 prose / figure quoting absolute or pooled "
            "cost (USD) across these cells MUST stratify by basis or it "
            "produces a unit-collision artifact (api_usd vs "
            "electricity_usd_derived = ~1000× scale gap). Reference: "
            "section1_intro.md [^cost-basis-cross-baseline] footnote."
        )
    fe_pool["cell_cost_unit_basis_summary"] = cell_basis_summary

    return {
        "schema_version": SCHEMA_VERSION,
        # NOTE: legacy key "primary_k_of_n" retained as alias for downstream consumers
        # (paper §6 figure scripts, OSF artifact replay); new canonical key is
        # "operational_deployment_gate". /stress A2.8 B-1551 transitional schema.
        "operational_deployment_gate": operational_gate,
        "primary_k_of_n": operational_gate,  # legacy alias (A2.8 B-1551 transitional)
        "appendix_fe_pool": fe_pool,
        "per_cell": per_cell_results,
        "note_site_asymmetric_pre_hoc_hypothesis": (
            "Site-asymmetric viability is a pre-hoc theoretical prediction "
            "(preregistration §H10 hypothesis prose — NOT archive-simulation-derived "
            "per /stress A2.8 P0-3-A* archive-deletion-from-prereg-justification "
            "2026-05-18): visual-rich classifieds cells (3 cells x B0/B1/B2) "
            "hypothesized to pass cell-level via task-conditional routing benefit; "
            "text-dominated reddit cells (3 cells x B0/B1/B2) hypothesized to collapse "
            "toward always_phantom_som baseline. Phase 1a clean-rerun is the "
            "falsification test of this hypothesis."
        ),
    }


def write_outputs(verdict: dict[str, Any], out_dir: Path) -> None:
    """Write CSV + JSON + Markdown outputs."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # JSON
    json_path = out_dir / "h10_pareto_verdict.json"
    json_path.write_text(json.dumps(verdict, indent=2, default=str))
    print(f"\nWrote: {json_path}")

    # CSV per-cell
    csv_lines = [
        "cell_id,baseline,site,status,n_common_tasks,router_sr,router_cost,"
        "theta_pp,theta_se_pp,fraction_non_dominated,passes"
    ]
    for cell_id, rec in verdict["per_cell"].items():
        if rec.get("status") != "ok":
            csv_lines.append(
                f"{cell_id},,,{rec.get('status')},,,,,,,{rec.get('passes', False)}"
            )
            continue
        pn = rec["pareto_non_dominance"]
        csv_lines.append(
            f"{cell_id},{rec['baseline']},{rec['site']},ok,"
            f"{rec['n_common_tasks']},{rec['router_sr_mean']:.4f},"
            f"{rec['router_cost_mean']:.6f},{pn['theta_mean_pp']:.2f},"
            f"{pn['theta_se_pp']:.2f},{pn['fraction_non_dominated']:.4f},"
            f"{pn['passes']}"
        )
    csv_path = out_dir / "h10_pareto_verdict.csv"
    csv_path.write_text("\n".join(csv_lines) + "\n")
    print(f"Wrote: {csv_path}")

    # Markdown summary
    md = ["# H10 Pareto Non-Dominance Verdict (paper §6 source)", ""]
    md.append(f"Schema: `{verdict['schema_version']}`")
    md.append("")
    md.append("## Operational deployment gate verdict (two-layer: cell-level + grid-level)")
    md.append("")
    md.append("> /stress A2.8 B-1551 reframing 2026-05-18: K-of-6 is a fixed-cell operational")
    md.append("> deployment criterion, NOT a binomial significance test. The 5/6 threshold is")
    md.append("> an engineering deployability margin, not an alpha=0.05 cells-population test.")
    md.append("")
    pv = verdict.get("operational_deployment_gate", verdict["primary_k_of_n"])
    md.append(f"- Cells passing cell-level (>=95% paired-bootstrap Pareto non-dominance): {pv['k_of_n_string']}")
    md.append(f"- Grid-level deployment threshold: {pv['deployment_threshold']}")
    md.append(f"- **Operational deployment gate passed**: {pv['operational_gate_passed']}")
    md.append("")
    md.append("## Appendix-D sensitivity: FE inverse-variance pool (transparency row, NOT gating)")
    fp = verdict["appendix_fe_pool"]
    if not np.isnan(fp.get("theta_pool_pp", float("nan"))):
        md.append(
            f"- Pooled θ = {fp['theta_pool_pp']:.2f}pp "
            f"[CI {fp['ci_95_pp'][0]:.2f}, {fp['ci_95_pp'][1]:.2f}]"
        )
        md.append(f"- Z vs δ=1.0pp: {fp['z_vs_delta_1pp']:.3f}, p_one_sided={fp['p_one_sided']:.4f}")
        md.append(f"- N cells pooled: {fp['n_cells_pooled']}")
    else:
        md.append(f"- Insufficient cells for FE pool ({fp.get('reason')})")
    md.append("")
    md.append("## Per-cell breakdown")
    md.append("")
    md.append("| Cell | Status | N common | Router SR | Router Cost | θ (pp) | Frac non-dom | Pass |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for cell_id, rec in verdict["per_cell"].items():
        if rec.get("status") != "ok":
            md.append(f"| {cell_id} | {rec.get('status')} | — | — | — | — | — | — |")
            continue
        pn = rec["pareto_non_dominance"]
        md.append(
            f"| {cell_id} | ok | {rec['n_common_tasks']} | "
            f"{rec['router_sr_mean']:.3f} | {rec['router_cost_mean']:.4f} | "
            f"{pn['theta_mean_pp']:.2f} | {pn['fraction_non_dominated']:.3f} | "
            f"{'✓' if pn['passes'] else '✗'} |"
        )
    md.append("")
    md.append("## Site-asymmetric viability note")
    # F5 (B-1818): verdict key is note_site_asymmetric_pre_hoc_hypothesis (defined
    # L718); the bare 'note_site_asymmetric' KeyError'd at §6 markdown write once data
    # landed. Use the correct key + .get() so a future rename degrades gracefully.
    md.append(f"{verdict.get('note_site_asymmetric_pre_hoc_hypothesis', '')}")

    md_path = out_dir / "h10_pareto_verdict.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"Wrote: {md_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="Verdict all 6 cells (default)")
    ap.add_argument("--baseline", help="B0 | B1 | B2 (subset)")
    ap.add_argument("--site", help="classifieds | reddit (subset)")
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory")
    ap.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="(retained for back-compat; full coverage is now the DEFAULT per "
        "P2-3/AMENDMENT_04 unless --allow-partial-dev). No-op when set.",
    )
    ap.add_argument(
        "--allow-partial-dev",
        action="store_true",
        help="P2-3 (codex B-F9, AMENDMENT_04): opt into dev-mode partial-intersection "
        "verdict. Paper-grade DEFAULT is fail-closed full coverage — without this flag a "
        "cell whose router/baseline task intersection is incomplete reports "
        "status=incomplete_coverage_paper_grade + passes=False instead of a biased subset.",
    )
    args = ap.parse_args()

    if args.baseline and args.site:
        cells = [(args.baseline, args.site)]
    else:
        cells = CELLS

    # P2-3 (codex B-F9, AMENDMENT_04): paper-grade DEFAULT = fail-closed full coverage;
    # --allow-partial-dev opts into the dev-mode partial-intersection subset verdict.
    require_full = not args.allow_partial_dev
    verdict = run_h10_verdict(cells, require_full_coverage=require_full)
    write_outputs(verdict, Path(args.out_dir))

    pv = verdict["primary_k_of_n"]
    print(f"\n=== H10 Verdict: {pv['k_of_n_string']} cells pass (K-of-6 PRIMARY) ===")
    fp = verdict["appendix_fe_pool"]
    if not np.isnan(fp.get("theta_pool_pp", float("nan"))):
        print(
            f"=== FE pool θ = {fp['theta_pool_pp']:.2f}pp "
            f"[CI {fp['ci_95_pp'][0]:.2f}, {fp['ci_95_pp'][1]:.2f}] (APPENDIX) ==="
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
