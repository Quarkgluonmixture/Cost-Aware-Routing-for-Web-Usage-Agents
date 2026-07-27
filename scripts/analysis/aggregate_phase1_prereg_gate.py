#!/usr/bin/env python3
"""B-184: H1 drop-one θ_i / SE_i producer — TRANSPARENCY-ONLY legacy (normal-Z).

⚠️ NON-CANONICAL for the gate DECISION (AMENDMENT 03, 2026-05-24). The paper §1 H1
PRIMARY gate is the **bootstrap percentile** test in
`aggregate_phase1_full_prereg_decision._pool_bootstrap_percentile_p`
(prereg §2 H1 L98 + AMENDMENT_02 §2 line 99; `h1_pass =
pooled_h1_bootstrap.gate_passed_bootstrap`). This file's per-cell
`_cell_drop_one_theta_se` is the SHARED θ_i/SE_i kernel the canonical producer
imports (UNCHANGED per AMENDMENT_02 §2); its own `_fe_pool` normal-Z `gate_passed`
+ `theta_FE_pp` are retained ONLY as a transparency column (prereg L98: "retained as
a transparency check column ... does NOT drive the gate decision"). SE-floor here is
aligned to the canonical 0.68pp threshold (`_fe_pool`) so the transparency θ_FE is
bit-identical to the canonical point estimate. §1 hero cites the canonical producer,
NOT this file.

Implements the prereg H1 per-cell kernel (preregistration.md:68-86 lock):

    Per-cell kernel = "FE inverse-variance pooled P-SoM drop-one effect θ_FE vs the
                   +1.0pp substantive-effect threshold, one-sided superiority test:
                   reject H0: θ_FE ≤ +1.0pp at α=0.05 (PRIMARY family m=1) —
                   gate DECISION uses canonical bootstrap percentile; this normal-Z
                   is transparency-only"

Per-cell drop-one (preregistration.md:77-82):

    For each (site, model) cell containing all 6 modes
    (DOM, SoM, Vision, P-text, P-prompt, P-SoM):
      per-task indicator d_t = [t ∈ oracle_6] - [t ∈ oracle_5_drop_PSoM]
      θ_i = mean_t(d_t) × 100  (pp units)
      SE_i = paired 1000-resample task-level bootstrap (seed=42 per B-176)

FE pool:
    w_i = 1 / SE_i²
    θ_FE = Σ(w_i · θ_i) / Σ(w_i)
    SE_FE = sqrt(1 / Σ(w_i))

One-sided superiority z-statistic (TRANSPARENCY-ONLY — not the gate decision):
    z = (θ_FE - 1.0) / SE_FE
    p_one_sided = 1 - Φ(z)
    gate_passed = (p_one_sided < 0.05)   # transparency column; canonical gate =
                                         # full_prereg_decision bootstrap percentile

This producer is **complementary** to `aggregate_phantom_lift.py` (legacy 3→5 lift,
exploratory) and **SUBORDINATE** to `aggregate_phase1_full_prereg_decision.py`
(canonical H1 PRIMARY). The prereg PRIMARY gate is the canonical full producer, NOT
this file; this file supplies the shared θ_i/SE_i kernel + a normal-Z transparency
cross-check.

Pre-data behavior: if <6 cells contain all 6 modes (e.g., Phase 1a still
mid-rerun), emits `gate_status="INSUFFICIENT_DATA"` + lists available cells +
exits 0 (not a fail) — `make analysis` does not block on this.

Cross-link: B-184 follow-up issue
`docs/checkpoints/_status/issues/issue_phase1_canonical_artifacts_2026-05-16.md`.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Reuse the SAME cell enumeration + episode loader the existing phantom-lift
# producer uses → guarantees the gate operates on the same data slices and
# refuses the same partial cells (MIN_EP_FOR_CELL filter).
from scripts.analysis.aggregate_phantom_lift import CELLS, MIN_EP_FOR_CELL  # noqa: E402
from scripts.analysis.lib.atomic_io import atomic_write_text  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import (  # noqa: E402
    expected_scored_ids,
    protocol_excluded_in_universe,
    task_id_set_sha256,
)
from scripts.analysis.lib.canonical_cells import PHASE_1A_PLANNED_CELLS  # noqa: E402
from scripts.analysis.lib.episode_rows import (  # noqa: E402
    load_cell_task_rows,
    load_task_rows,
)

# B-176 lock: bootstrap seed=42, B=1000 per prereg "1000-resample".
# (Note: prereg explicitly says 1000, NOT the 10_000 used in `analyze_run`
# bootstrap CIs. Different parts of the pipeline have different B; the per-cell
# drop-one SE here is the prereg-locked B=1000 path.)
PREREG_B = 1000
PREREG_SEED = 42
DELTA_PP = 1.0    # prereg superiority threshold (preregistration.md:341 lock)
ALPHA = 0.05      # prereg α
# SE-floor (degenerate-cell protocol, prereg §2 H1 L103-111 + L98/L718 B-1003 codify).
# AMENDMENT 03 (2026-05-24): threshold aligned to the 0.68pp Agresti-Coull anchor — the
# SINGLE source mirrored by the canonical primary producer
# `aggregate_phase1_full_prereg_decision` (which uses the same 0.68/1.0 values). Floor
# REPLACE value 1.0pp + δ unchanged → implementation alignment, NOT an estimand change.
SE_FLOOR_THRESHOLD_PP = 0.68
SE_FLOOR_REPLACE_PP = 1.0
SIX_MODES = ("DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM")

DEFAULT_OUT_CSV = REPO / "results/phantom_paper/phase1_prereg_gate.csv"
DEFAULT_OUT_JSON = REPO / "results/phantom_paper/phase1_prereg_gate.json"
DEFAULT_OUT_MD = REPO / "results/phantom_paper/phase1_prereg_gate.md"


def load(episodes_dir: Path) -> tuple[set[int], set[int]]:
    """Legacy H1 set view over the shared identity-checked row loader."""
    rows = load_task_rows(episodes_dir)
    observed = set(rows)
    succeeded = {task_id for task_id, row in rows.items() if row["success"] is True}
    return succeeded, observed


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via erf; scipy-free."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _cell_drop_one_theta_se(
    cell: Dict, *, B: int = PREREG_B, seed: int = PREREG_SEED,
    expected_ids: Optional[frozenset[int] | set[int]] = None,
    rows_by_mode: Optional[Dict[str, Dict[int, Dict]]] = None,
    tolerate_extra_ids: Optional[frozenset[int] | set[int]] = None,
) -> Dict:
    """Compute per-cell drop-one effect + bootstrap SE per prereg spec.

    Returns an explicit ``complete_exact=False`` diagnostic if any mode's
    observed IDs differ from the canonical scored task universe.  Missing and
    extra IDs therefore fail closed instead of being hidden by an intersection.

    Otherwise returns:
        {baseline, site, n_tasks, theta_pp, se_pp, ci95_lo_pp, ci95_hi_pp,
         oracle_6_pp, oracle_5_no_psom_pp, n_psom_only}
    """
    if expected_ids is None:
        expected, task_set_sha = expected_scored_ids(cell["site"])
    else:
        expected = frozenset(int(t) for t in expected_ids)
        task_set_sha = task_id_set_sha256(expected)

    # Load every mode through one shared, identity-checked task_id -> row map.
    # The canonical full producer passes the same map onward to H2/H3, so the
    # three hypotheses cannot silently use different task identities.
    if rows_by_mode is None:
        rows_by_mode = load_cell_task_rows(cell, modes=SIX_MODES)
    succ: Dict[str, set] = {}
    obs: Dict[str, set] = {}
    for mode in SIX_MODES:
        rows = rows_by_mode.get(mode, {})
        obs[mode] = set(rows)
        succ[mode] = {
            task_id for task_id, row in rows.items() if row["success"] is True
        }

    # AMENDMENT_08: the runner still COLLECTS the protocol-excluded tasks, so a
    # landed reddit cell holds 205 episodes against a 203-task scored set. Those
    # two are expected, not contamination. Without this carve-out all three
    # reddit cells fail `complete_exact`, get skipped, and the gate silently
    # runs at k=3 on classifieds alone — which reads out as framing=R5 (paper
    # death) purely as an artifact. `expected_ids` callers pass their own
    # universe and opt out.
    # An explicit `expected_ids` caller states its own universe, so the default
    # carve-out does not apply — but a sensitivity arm (AMENDMENT_08 §5) needs to
    # say "this narrower universe, and these landed-but-unscored ids are still
    # expected". `tolerate_extra_ids` is that knob; without it every arm except
    # the pre-amendment one fails `complete_exact` and the comparison the
    # amendment promises cannot be reproduced.
    if expected_ids is None:
        protocol_excluded = protocol_excluded_in_universe(cell["site"])
    else:
        protocol_excluded = frozenset(int(t) for t in (tolerate_extra_ids or ()))
    observed_n = {m: len(obs[m]) for m in SIX_MODES}
    missing_ids = {m: sorted(expected - obs[m]) for m in SIX_MODES}
    extra_ids = {m: sorted(obs[m] - expected - protocol_excluded) for m in SIX_MODES}
    complete_exact = all(
        not missing_ids[m] and not extra_ids[m] for m in SIX_MODES
    )
    diagnostics = {
        "baseline": cell["baseline"],
        "site": cell["site"],
        "complete_exact": complete_exact,
        "expected_n": len(expected),
        "observed_n": observed_n,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        # AMENDMENT_08 transparency: collected-but-not-scored IDs actually seen.
        "protocol_excluded_observed": {
            m: sorted(obs[m] & protocol_excluded) for m in SIX_MODES
        },
        "task_set_sha256": task_set_sha,
    }
    if not complete_exact:
        diagnostics["incomplete_reason"] = (
            "one or more modes do not exactly match the canonical scored task-ID set"
        )
        return diagnostics

    # Exact equality makes the prereg universe the canonical scored set, not a
    # data-dependent intersection of whichever tasks happened to be observed.
    universe = set(expected)
    n = len(expected)

    universe_sorted = sorted(universe)

    # Build per-task oracle indicators.
    succ_r = {m: succ[m] & universe for m in SIX_MODES}
    oracle_6 = set().union(*[succ_r[m] for m in SIX_MODES])
    oracle_5_no_psom = set().union(*[succ_r[m] for m in SIX_MODES if m != "P-SoM"])

    in_6 = np.array([t in oracle_6 for t in universe_sorted], dtype=np.int8)
    in_5_no_psom = np.array(
        [t in oracle_5_no_psom for t in universe_sorted], dtype=np.int8,
    )
    diff = (in_6 - in_5_no_psom).astype(np.int8)
    # `diff` is element-wise in {0, 1}: 1 iff oracle_6 covers task AND
    # oracle_5_no_psom does NOT — i.e. P-SoM is the only-saver for that task.
    # (`diff = -1` is impossible because oracle_5_no_psom ⊆ oracle_6 by construction.)
    if (diff < 0).any():
        # Safety net: should never happen given subset construction; surface if it does
        raise AssertionError("oracle_5_no_psom contained a task absent from oracle_6")

    theta_pp = 100.0 * float(diff.mean())

    # Paired task-level bootstrap SE_i (prereg B=1000, seed pinned per B-176).
    rng = np.random.default_rng(seed)
    boot_thetas = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot_thetas[b] = 100.0 * float(diff[idx].mean())
    # SE = std of bootstrap distribution (ddof=1 matches sample-std convention).
    se_pp = float(boot_thetas.std(ddof=1))
    ci_lo = float(np.quantile(boot_thetas, 0.025))
    ci_hi = float(np.quantile(boot_thetas, 0.975))

    return {
        **diagnostics,
        "n_tasks": n,
        "theta_pp": theta_pp,
        "se_pp": se_pp,
        "ci95_lo_pp": ci_lo,
        "ci95_hi_pp": ci_hi,
        # B-1301 (/stress A2.3d P0-1-AB*, 2026-05-18): expose the 1000-rep paired
        # bootstrap distribution so the canonical producer can compute the
        # prereg-locked bootstrap percentile FE pool p-value (B-1009 amend).
        # Pre-fix this array was discarded after se_pp/ci derivation; the B-1009
        # amendment promised `P(θ_FE* ≤ 1.0pp)` over 1000 paired-bootstrap pool
        # replicates but never had the per-cell substrate to pool. Returning the
        # B-length float32 vector lets `aggregate_phase1_full_prereg_decision`
        # IV-weight per-iteration θ_i_b → θ_FE_b distribution.
        "boot_pp": boot_thetas.astype(np.float32),
        "oracle_6_pp": 100.0 * len(oracle_6) / n,
        "oracle_5_no_psom_pp": 100.0 * len(oracle_5_no_psom) / n,
        "n_psom_only": int(diff.sum()),
    }


def _fe_pool(per_cell: List[Dict]) -> Optional[Dict]:
    """Fixed-effects inverse-variance pool over per-cell drop-one estimates.

    w_i = 1 / SE_i²
    θ_FE = Σ(w_i · θ_i) / Σ(w_i)
    SE_FE = sqrt(1 / Σ(w_i))
    z = (θ_FE - δ) / SE_FE  with δ=1.0pp
    p_one_sided = 1 - Φ(z)

    Returns None if fewer than 2 cells (FE pool ill-defined at k=1).
    """
    if len(per_cell) < 2:
        return None
    thetas = np.array([r["theta_pp"] for r in per_cell])
    ses = np.array([r["se_pp"] for r in per_cell])
    # /stress A1.19 P0-1 (2026-05-17, gemini Mode C OOB + Claude data-grounded analysis):
    # SE_floor = 1.0pp is now PRE-REGISTERED disclosed in preregistration.md §2 H1
    # "Degenerate-cell SE floor protocol" paragraph (Agresti-Coull-style finite lower
    # bound). Data-anchored to archive 2026-05-09 P-SoM cells (median SE 0.98pp at
    # N≈200-234, p≈10-22%); no post-hoc floor tuning permitted post-data-lock.
    # Implementation invariant: only applies when bootstrap SE_i = 0 exactly (degenerate
    # cell where drop-one diff vector is identically constant under all resamples);
    # `n_zero_se_floored_cells` is emitted in payload for paper §6 disclosure.
    # Implementation-alignment (AMENDMENT 03, 2026-05-24): SE-floor uses the module-level
    # SE_FLOOR_THRESHOLD_PP (0.68pp Agresti-Coull anchor), the SINGLE source mirrored by
    # the canonical primary producer
    # `aggregate_phase1_full_prereg_decision._pool_bootstrap_percentile_p`. This legacy
    # producer was previously on a literal `<= 0` floor — the B-1003 "code-bug fix" that
    # codified 0.68 in prereg prose (L98/L718) but never landed in code. NO estimand
    # change (REPLACE value 1.0pp + δ unchanged); only makes the transparency θ_FE
    # bit-identical to the canonical point estimate.
    n_zero_se = int((ses <= 0).sum())  # legacy exact-zero transparency stat (back-compat)
    n_below_floor = int((ses < SE_FLOOR_THRESHOLD_PP).sum())
    if n_below_floor > 0:
        ses = np.where(ses < SE_FLOOR_THRESHOLD_PP, SE_FLOOR_REPLACE_PP, ses)
    w = 1.0 / (ses ** 2)
    theta_fe = float(np.sum(w * thetas) / np.sum(w))
    se_fe = float(math.sqrt(1.0 / np.sum(w)))
    z = (theta_fe - DELTA_PP) / se_fe
    p_one_sided = 1.0 - _norm_cdf(z)
    return {
        "k_cells": len(per_cell),
        "theta_FE_pp": theta_fe,
        "se_FE_pp": se_fe,
        "ci95_FE_lo_pp": theta_fe - 1.96 * se_fe,
        "ci95_FE_hi_pp": theta_fe + 1.96 * se_fe,
        "delta_pp": DELTA_PP,
        "z_one_sided": z,
        "p_one_sided": p_one_sided,
        "alpha": ALPHA,
        "gate_passed": bool(p_one_sided < ALPHA),
        # v6 fix (P0-9) + AMENDMENT 03 alignment (2026-05-24): SE-floor transparency.
        # `n_zero_se_floored_cells` = legacy exact-zero count (back-compat);
        # `n_below_se_floor_cells` = cells floored under the canonical 0.68pp
        # Agresti-Coull threshold. paper §6 discloses if > 0. NOTE: this normal-Z
        # `gate_passed` is TRANSPARENCY-ONLY; the canonical H1 PRIMARY gate is
        # `aggregate_phase1_full_prereg_decision` bootstrap percentile (prereg L98).
        "n_zero_se_floored_cells": n_zero_se,
        "n_below_se_floor_cells": n_below_floor,
        "se_floor_threshold_pp": SE_FLOOR_THRESHOLD_PP,
    }


def build_gate(
    cells: List[Dict], *,
    expected_ids_by_site: Optional[Dict[str, frozenset[int] | set[int]]] = None,
) -> Dict:
    """End-to-end gate computation across the provided cell list.

    Returns a structured payload with per-cell rows + pooled FE result +
    gate_status ∈ {"PASS", "FAIL", "INSUFFICIENT_DATA", "PARTIAL_DATA"}.
    """
    per_cell: List[Dict] = []
    skipped: List[Dict] = []
    for cell in cells:
        expected_ids = (
            expected_ids_by_site.get(cell["site"])
            if expected_ids_by_site is not None else None
        )
        result = _cell_drop_one_theta_se(cell, expected_ids=expected_ids)
        if not result["complete_exact"]:
            skipped.append({
                **result,
                "reason": result["incomplete_reason"],
            })
        else:
            per_cell.append(result)

    payload: Dict = {
        "prereg_section": (
            "preregistration.md §1 H1 normal-Z transparency check; "
            "NOT canonical verdict (canonical bootstrap gate is full decision producer)"
        ),
        "estimand": "FE inverse-variance pooled P-SoM drop-one over 6 planned cells",
        "delta_pp": DELTA_PP,
        "alpha": ALPHA,
        "bootstrap_B": PREREG_B,
        "bootstrap_seed": PREREG_SEED,
        "per_cell": per_cell,
        "skipped_cells": skipped,
    }

    fe = _fe_pool(per_cell)
    planned = {(site, baseline) for site, baseline in PHASE_1A_PLANNED_CELLS}
    exact_cells = {(r["site"], r["baseline"]) for r in per_cell}
    if fe is None:
        analysis_status = "INSUFFICIENT"
    elif len(per_cell) == len(planned) and exact_cells == planned:
        analysis_status = "COMPLETE"
    else:
        analysis_status = "PARTIAL"
    payload["analysis_status"] = analysis_status
    payload["h1_verdict_normal_approx_transparency"] = "NOT_EVALUATED"
    if len(per_cell) == 0:
        payload["gate_status"] = "INSUFFICIENT_DATA"
        payload["gate_status_reason"] = (
            "No cells contain all 6 modes (DOM/SoM/Vision/P-text/P-prompt/P-SoM). "
            "Phase 1a rerun likely still in flight; gate cannot be evaluated."
        )
    elif fe is None:
        payload["gate_status"] = "INSUFFICIENT_DATA"
        payload["gate_status_reason"] = (
            f"Only {len(per_cell)} cell(s) with all 6 modes; FE pool requires ≥2."
        )
    elif len(per_cell) < 6:
        # Partial = some cells available but not the full 6 planned.
        payload["pooled_fe"] = fe
        payload["gate_status"] = "PARTIAL_DATA"
        payload["gate_status_reason"] = (
            f"{len(per_cell)} of 6 planned cells have all 6 modes; "
            "pooled result is reported but does NOT yet match the prereg "
            "estimand (which is exactly 6 cells)."
        )
    else:
        payload["pooled_fe"] = fe
        payload["gate_status"] = "PASS" if fe["gate_passed"] else "FAIL"
        payload["gate_status_reason"] = (
            f"Pooled θ_FE={fe['theta_FE_pp']:.3f}pp, z={fe['z_one_sided']:.3f}, "
            f"p_one_sided={fe['p_one_sided']:.4f}, α={ALPHA}, δ={DELTA_PP}pp."
        )

    if analysis_status == "COMPLETE" and fe is not None:
        payload["h1_verdict_normal_approx_transparency"] = (
            "PASS" if fe["gate_passed"] else "FAIL"
        )

    return payload


def write_csv(payload: Dict, out_csv: Path) -> None:
    """Per-cell × 6 + pooled FE row → flat CSV for paper §1 prose to cite."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "row_type,baseline,site,k_cells,n_tasks,theta_pp,se_pp,ci95_lo_pp,"
        "ci95_hi_pp,oracle_6_pp,oracle_5_no_psom_pp,n_psom_only,"
        "z_one_sided,p_one_sided,gate_passed,gate_status,analysis_status,"
        "h1_verdict_normal_approx_transparency",
    ]
    gs = payload.get("gate_status", "UNKNOWN")
    analysis_status = payload.get("analysis_status", "INSUFFICIENT")
    h1_verdict = payload.get(
        "h1_verdict_normal_approx_transparency", "NOT_EVALUATED"
    )
    for r in payload["per_cell"]:
        lines.append(
            f"cell,{r['baseline']},{r['site']},,{r['n_tasks']},"
            f"{r['theta_pp']:.4f},{r['se_pp']:.4f},"
            f"{r['ci95_lo_pp']:.4f},{r['ci95_hi_pp']:.4f},"
            f"{r['oracle_6_pp']:.4f},{r['oracle_5_no_psom_pp']:.4f},"
            f"{r['n_psom_only']},,,,{gs},{analysis_status},{h1_verdict}"
        )
    fe = payload.get("pooled_fe")
    if fe is not None:
        lines.append(
            f"pooled_FE,,,{fe['k_cells']},,"
            f"{fe['theta_FE_pp']:.4f},{fe['se_FE_pp']:.4f},"
            f"{fe['ci95_FE_lo_pp']:.4f},{fe['ci95_FE_hi_pp']:.4f},"
            f",,,{fe['z_one_sided']:.4f},{fe['p_one_sided']:.6f},"
            f"{fe['gate_passed']},{gs},{analysis_status},{h1_verdict}"
        )
    atomic_write_text(out_csv, "\n".join(lines) + "\n")


def _json_default(o):
    """JSON fallback for numpy scalars/arrays the gate payload carries.

    `compute_cell()` stows the raw bootstrap distribution under `boot_pp`
    (an `np.float32` ndarray) and several stats arrive as `np.floating` /
    `np.integer`; the stdlib encoder rejects all of these. Convert to native
    Python so `write_json` round-trips losslessly.
    """
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def write_json(payload: Dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        out_json,
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
    )


def write_md(payload: Dict, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 legacy H1 normal-approximation transparency check",
        "",
        "**Estimand** (preregistration.md §1, lines 68-86 lock):",
        "",
        "> FE inverse-variance pooled P-SoM drop-one over 6 planned (site, model) cells,",
        "> one-sided superiority test: reject H0: θ_FE ≤ +1.0pp at α=0.05.",
        "",
        f"- **δ = {DELTA_PP}pp** (substantive-effect threshold, prereg lock)",
        f"- **α = {ALPHA}** (one-sided)",
        f"- **B = {PREREG_B}** (paired task-level bootstrap; prereg-locked)",
        f"- **seed = {PREREG_SEED}** (B-176 pinned)",
        "",
        f"**Gate status**: `{payload.get('gate_status', 'UNKNOWN')}`",
        f"**Analysis status**: `{payload.get('analysis_status', 'INSUFFICIENT')}`",
        "**NOT the canonical H1 verdict.** Canonical = full decision producer's "
        "bootstrap-percentile gate.",
        f"**Normal-approximation transparency verdict**: "
        f"`{payload.get('h1_verdict_normal_approx_transparency', 'NOT_EVALUATED')}`",
        "",
        payload.get("gate_status_reason", ""),
        "",
        "## Per-cell drop-one",
        "",
        "| Baseline | Site | n_tasks | θ (pp) | SE (pp) | 95% CI | oracle_6 (pp) | oracle_5∖P-SoM (pp) | n_PSoM_only |",
        "|---|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for r in payload["per_cell"]:
        lines.append(
            f"| {r['baseline']} | {r['site']} | {r['n_tasks']} | "
            f"+{r['theta_pp']:.2f} | {r['se_pp']:.3f} | "
            f"[{r['ci95_lo_pp']:.2f}, {r['ci95_hi_pp']:.2f}] | "
            f"{r['oracle_6_pp']:.2f} | {r['oracle_5_no_psom_pp']:.2f} | "
            f"{r['n_psom_only']} |"
        )

    if payload.get("skipped_cells"):
        lines += ["", "### Skipped cells (missing one of 6 modes / below MIN_EP)", ""]
        for s in payload["skipped_cells"]:
            lines.append(f"- **{s['baseline']} {s['site']}** — {s['reason']}")

    fe = payload.get("pooled_fe")
    if fe is not None:
        sig = "✅ **PASSED**" if fe["gate_passed"] else "❌ **NOT YET**"
        lines += [
            "",
            "## Pooled FE normal-Z transparency check (non-canonical)",
            "",
            f"- **k = {fe['k_cells']}** cells",
            f"- **θ_FE = +{fe['theta_FE_pp']:.3f}pp** (SE = {fe['se_FE_pp']:.3f}pp)",
            f"- **95% CI**: [{fe['ci95_FE_lo_pp']:.3f}, {fe['ci95_FE_hi_pp']:.3f}]pp",
            f"- **z** = (θ_FE − δ) / SE_FE = ({fe['theta_FE_pp']:.3f} − {fe['delta_pp']}) / {fe['se_FE_pp']:.3f} = **{fe['z_one_sided']:.3f}**",
            f"- **p_one_sided** = 1 − Φ(z) = **{fe['p_one_sided']:.4f}**",
            f"- **Gate (p < α={ALPHA})**: {sig}",
        ]

    lines += [
        "",
        "---",
        "_Producer: `scripts/analysis/aggregate_phase1_prereg_gate.py` (B-184)._",
        "_Demoted-to-appendix alternative estimands: see `phantom_lift.csv` (3→5 lift,_",
        "_codex B2 catch — different estimand) and `meta_phantom_lift.csv` (DerSimonian-Laird_",
        "_random-effects, B-182 marked appendix-only)._",
    ]
    atomic_write_text(out_md, "\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    # B-1014 (/stress A2.4a P1-12-B codex F4, 2026-05-18): manifest_path parity
    # with canonical full producer. Pre-fix B-184 legacy main() had no
    # `--run-manifest` arg → B-184 artifact JSON could diverge from canonical
    # `aggregate_phase1_full_prereg_decision.py` post-fire even though kernel
    # was bit-identical (different cells_to_use under different env/manifest).
    # Now propagates manifest_path through `get_aggregator_cells(manifest_path=)`
    # — same as canonical (A1.21 P0-5 B-524 + B-530 lazy fn).
    ap.add_argument("--run-manifest", default=None,
                    help="Path to run_manifest.yaml (default: registry default). "
                         "B-1014 propagates to get_aggregator_cells for cross-producer "
                         "cells_to_use consistency.")
    ap.add_argument("--output-csv", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--output-md", default=str(DEFAULT_OUT_MD))
    args = ap.parse_args(argv)

    # B-1014: lazy fn re-evaluates env var + manifest at call time (mirror
    # canonical full producer pattern). Falls back to module-level frozen CELLS
    # only for the expected legacy-import compatibility failure; registry and
    # manifest errors must propagate instead of silently selecting other data.
    try:
        from scripts.analysis.aggregate_phantom_lift import (
            get_aggregator_cells as _get,
        )
    except (ImportError, AttributeError) as exc:
        print(
            "warning: aggregate_phantom_lift.get_aggregator_cells unavailable; "
            f"falling back to legacy frozen CELLS ({exc})",
            file=sys.stderr,
        )
        cells_to_use = CELLS
    else:
        manifest_path = (Path(args.run_manifest) if args.run_manifest
                         else REPO / "results/phantom_paper/run_manifest.yaml")
        cells_to_use = _get(manifest_path=manifest_path)

    payload = build_gate(cells_to_use)

    write_csv(payload, Path(args.output_csv))
    write_json(payload, Path(args.output_json))
    write_md(payload, Path(args.output_md))

    print(f"[B-184] gate_status={payload['gate_status']} "
          f"(k_cells={len(payload['per_cell'])}, skipped={len(payload['skipped_cells'])})")
    print(f"        → {args.output_csv}")
    print(f"        → {args.output_json}")
    print(f"        → {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
