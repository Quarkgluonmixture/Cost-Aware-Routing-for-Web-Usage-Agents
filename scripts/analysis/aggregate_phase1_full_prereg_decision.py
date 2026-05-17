#!/usr/bin/env python3
r"""Canonical full Phase 1 prereg decision producer — H1 + H2(a) + H3 axes + framing.

Built /stress A1.21 P0-2 + P0-3 + P0-4 + P0-11 (2026-05-17, B-481).

`aggregate_phase1_prereg_gate.py` (B-184) is H1-only. `preregistration_decision_test.py`
covers H1+H2+H3 but uses retired DerSimonian-Laird estimand (A1.21 codex P0-1 catch:
prereg ↔ code lock breach) AND contains a heterogeneity-rescue branch that violates
prereg §2 R5 flow (A1.21 codex P0-2 catch). Result: `make analysis` has no canonical
full R1-R5 artifact (A1.21 codex P0-3).

This producer fills that hole:

  1. H1 — FE inverse-variance pool over P-SoM drop-one (reuses B-184 _fe_pool path)
  2. H2(a) — per-task cost ratio median falsification check (NOT median-of-marginals;
            A1.21 Claude+codex+gemini P0-1 catch: paper §1 line 9 is per-task claim,
            old decision_test was median(P-SoM) / median(DOM) → wrong estimand)
  3. H3 axis-1/2 — FE pool over per-cell unique-count (P-text \ P-SoM, P-prompt \ P-SoM)
  4. I² + Cochran's Q on H1 — CAP-ONLY (high I² caps framing power R1→R3, does NOT
     rescue failed FE-H1; A1.21 codex+gemini P0-11 catch + prereg L323 explicit)
  5. Framing rule R1-R5 — per prereg §2 mapping with I² cap-only override

Output:
  results/phantom_paper/phase1_full_prereg_decision.{csv,json,md}

Provenance lock:
  - `code_sha256` = sha256 of this script + reused B-184 helpers
  - `manifest_sha256` = sha256 of run_manifest.yaml (provenance audit)
  - `input_csv_sha256` = sha256 of per_task_sr.csv if --per-task-csv supplied
  - `commit_sha` = `git rev-parse HEAD` if available

Tied to:
  - preregistration.md §2 H1/H2(a)/H3 + §2 R1-R5 mapping + §3 statistical methods
  - aggregate_phase1_prereg_gate.py B-184 (FE H1 reuse via direct import)
  - generate_per_task_sr.py B-122 (bridge for per-task CSV input — preferred path)
  - osf_lock_manifest.md (audit chain)
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Reuse the same cell enumeration + H1 FE pool + Q/I² helpers from B-184.
# This guarantees H1 numbers are bit-identical between phase1_prereg_gate
# (H1-only legacy) and phase1_full_prereg_decision (canonical full).
from scripts.analysis.aggregate_phantom_lift import (  # noqa: E402
    MIN_EP_FOR_CELL,
    get_aggregator_cells,  # A1.21 P1-3 (B-499): lazy fn, was frozen CELLS constant
)
from scripts.analysis.aggregate_phase1_prereg_gate import (  # noqa: E402
    _cell_drop_one_theta_se,
    _fe_pool,
    _norm_cdf,
    ALPHA,
    DELTA_PP,
    PREREG_B,
    PREREG_SEED,
    SIX_MODES,
)

DEFAULT_OUT_CSV = REPO / "results/phantom_paper/phase1_full_prereg_decision.csv"
DEFAULT_OUT_JSON = REPO / "results/phantom_paper/phase1_full_prereg_decision.json"
DEFAULT_OUT_MD = REPO / "results/phantom_paper/phase1_full_prereg_decision.md"

# Prereg §2 H2(a) lock (2026-05-14 Decision 3A + A1.21 P0-9 prereg amend):
# per-task median ratio cost(P-SoM)/cost(DOM) within ±20% of 1.0 per cell.
# ANY cell violation → falsified → R4 framing.
H2A_MARGIN_PCT = 20.0  # +20% = 1.20× per prereg lock L120-145

# Prereg §2 heterogeneity rule (2026-05-14 Decision 3A + A1.21 P0-3 cap-only):
# I² > 75% caps framing power at R3, does NOT rescue failed H1.
HETEROGENEITY_CAP_PCT = 75.0


def _git_commit_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL,
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _file_sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _self_code_sha() -> str:
    """SHA of this script + the B-184 helper module (covers H1 + FE pool path)."""
    h = hashlib.sha256()
    for p in (Path(__file__), REPO / "scripts/analysis/aggregate_phase1_prereg_gate.py"):
        with p.open("rb") as f:
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cell per-task data loader (success + cost, paired across modes)
# ---------------------------------------------------------------------------

def _load_cell_per_task(cell: Dict) -> Dict[str, Dict[str, Dict]]:
    """Load per-mode per-task {success, cost} dict for one (baseline, site) cell.

    A1.21 P0-1 fix: `cost_raw = data.get('total_cost_usd')` 用 `is None` 检查
    避免 `or` short-circuit drop valid 0.0. Same fix in `generate_per_task_sr.py`.

    Returns: dict[mode] -> dict[task_id_str] -> {"success": float|None, "cost": float|None}
    """
    by_mode: Dict[str, Dict[str, Dict]] = {}
    for mode, ep_dir in cell["modes"].items():
        outcomes: Dict[str, Dict] = {}
        if not ep_dir.exists():
            by_mode[mode] = outcomes
            continue
        for summary_path in sorted(ep_dir.glob("*_summary_v2.json")):
            try:
                with summary_path.open() as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            tid = data.get("task_id")
            if tid is None:
                continue
            success_raw = data.get("success")
            # P0-1 fix: `is None` check, NOT truthy `or` short-circuit
            cost_raw = data.get("total_cost_usd")
            if cost_raw is None:
                cost_raw = data.get("total_model_cost_usd")
            try:
                success = float(success_raw) if success_raw is not None else None
            except (TypeError, ValueError):
                success = None
            try:
                cost = float(cost_raw) if cost_raw is not None else None
            except (TypeError, ValueError):
                cost = None
            outcomes[str(tid)] = {"success": success, "cost": cost}
        by_mode[mode] = outcomes
    return by_mode


def _h2a_per_task_ratio(per_task: Dict[str, Dict[str, Dict]]) -> Optional[Dict]:
    """H2(a) per-task cost ratio median falsification check.

    A1.21 P0-9 fix (prereg §2 H2(a) lock amend 2026-05-17):
    pre-fix `evaluate_h2_cost` computed median(P-SoM costs) / median(DOM costs)
    — marginal medians, paired info ignored. Paper §1 line 9 "the cost of obtaining
    this configuration is essentially the cost of the DOM baseline" is a per-task
    claim → estimand should be median over tasks of (cost_psom[t] / cost_dom[t]).

    Returns None if no paired tasks with positive DOM cost.
    """
    dom = per_task.get("DOM", {})
    psom = per_task.get("P-SoM", {})
    common = sorted(set(dom.keys()) & set(psom.keys()))
    ratios: List[float] = []
    n_dom_zero = 0
    n_dom_missing = 0
    n_psom_missing = 0
    for tid in common:
        cd = dom[tid]["cost"]
        cp = psom[tid]["cost"]
        if cd is None:
            n_dom_missing += 1
            continue
        if cp is None:
            n_psom_missing += 1
            continue
        if cd <= 0:
            # cost=0 is real (e.g., GLM fallback or proxy edge); skip ratio to avoid
            # div-by-zero but count for transparency
            n_dom_zero += 1
            continue
        ratios.append(cp / cd)
    if not ratios:
        return None
    med_ratio = statistics.median(ratios)
    rel_diff_pct = (med_ratio - 1.0) * 100.0
    within_band = abs(rel_diff_pct) <= H2A_MARGIN_PCT
    return {
        "n_paired_tasks": len(common),
        "n_ratios_computed": len(ratios),
        "n_dom_zero_skipped": n_dom_zero,
        "n_dom_missing": n_dom_missing,
        "n_psom_missing": n_psom_missing,
        "median_ratio": med_ratio,
        "relative_diff_pct": rel_diff_pct,
        "margin_pct": H2A_MARGIN_PCT,
        "per_cell_pass": within_band,
        "per_cell_falsified": not within_band,
    }


def _h3_axis_per_cell(per_task: Dict[str, Dict[str, Dict]], axis_mode: str,
                       ref_mode: str = "P-SoM", *,
                       B: int = PREREG_B, seed: int = PREREG_SEED) -> Optional[Dict]:
    """H3 axis test: count tasks where axis_mode solved AND ref_mode did NOT.

    axis_mode = "P-text" (axis-1) or "P-prompt" (axis-2).
    Per-cell unique-count statistic + paired bootstrap SE → FE pool input.

    Returns None if either mode missing or below MIN_EP_FOR_CELL.
    """
    axis = per_task.get(axis_mode, {})
    ref = per_task.get(ref_mode, {})
    if not axis or not ref:
        return None
    common = sorted(set(axis.keys()) & set(ref.keys()))
    if len(common) < MIN_EP_FOR_CELL:
        return None
    axis_solved = np.array(
        [1 if axis[t]["success"] is not None and axis[t]["success"] >= 1 else 0
         for t in common], dtype=np.int8,
    )
    ref_solved = np.array(
        [1 if ref[t]["success"] is not None and ref[t]["success"] >= 1 else 0
         for t in common], dtype=np.int8,
    )
    unique = axis_solved * (1 - ref_solved)
    n = len(common)
    count_pp = 100.0 * float(unique.mean())

    # Paired task-level bootstrap SE (prereg-locked B=1000, seed=42).
    rng = np.random.default_rng(seed)
    boot = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        boot[b] = 100.0 * float(unique[idx].mean())
    se_pp = float(boot.std(ddof=1))
    ci_lo = float(np.quantile(boot, 0.025))
    ci_hi = float(np.quantile(boot, 0.975))
    # A1.21 P0-6 fix: also emit percentile p-value for method coherence
    p_percentile_one_sided = float((boot <= 0).mean())  # H0: count ≤ 0
    return {
        "axis_mode": axis_mode,
        "ref_mode": ref_mode,
        "n_tasks": n,
        "unique_count_pp": count_pp,
        "n_unique": int(unique.sum()),
        "se_pp": se_pp,
        "ci95_lo_pp": ci_lo,
        "ci95_hi_pp": ci_hi,
        "p_percentile_one_sided": p_percentile_one_sided,
    }


# ---------------------------------------------------------------------------
# I² + Cochran's Q on FE pool (cap-only, NOT rescue)
# ---------------------------------------------------------------------------

def _compute_q_isq(thetas: np.ndarray, ses: np.ndarray) -> Dict:
    """Cochran's Q + I² for FE pool — per Higgins & Thompson 2002.

    Q = Σ w_i × (θ_i − θ_FE)²
    df = k − 1
    I² = max(0, (Q − df) / Q) × 100

    A1.21 P0-11 fix: this lives in the CANONICAL producer so paper §1 framing
    can apply prereg §2 heterogeneity cap-only rule (high I² caps R1/R2 → R3,
    does NOT rescue failed FE-H1 — A1.21 P0-3 / prereg L323).
    """
    k = len(thetas)
    if k < 2:
        return {"k": k, "Q": None, "df": 0, "I_squared_pct": None,
                "note": "k<2: heterogeneity undefined"}
    w = 1.0 / (ses ** 2)
    theta_fe = float(np.sum(w * thetas) / np.sum(w))
    Q = float(np.sum(w * (thetas - theta_fe) ** 2))
    df = k - 1
    isq = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0
    return {
        "k": k,
        "Q": Q,
        "df": df,
        "I_squared_pct": isq,
        "heterogeneity_cap_at_r3": isq > HETEROGENEITY_CAP_PCT,
        "heterogeneity_threshold_pct": HETEROGENEITY_CAP_PCT,
    }


# ---------------------------------------------------------------------------
# Framing rule R1-R5 (with I² cap-only, NOT rescue)
# ---------------------------------------------------------------------------

def _apply_framing(h1_pass: bool, h2a_falsified: bool,
                    h3_axis1_pass: bool, h3_axis2_pass: bool,
                    h1_isq_cap_at_r3: bool) -> Dict:
    """Apply prereg §2 R1-R5 framing rule with I² cap-only override.

    A1.21 P0-3 + P0-11 fix: I² > 75% caps R1/R2 → R3, but does NOT rescue
    failed H1 (prereg L323 + L340-342). Per-cell consistency substitution
    (which the retired decision_test had) is RETIRED.
    """
    # H1 failed → R5 (paper death) regardless of I² or H2(a) or H3
    if not h1_pass:
        return {"rule": "R5",
                "framing": "Paper death scenario — H1 FE superiority failed; pivot needed",
                "hook_power": "n/a",
                "heterogeneity_override": False}
    # H1 passed but H2(a) falsified → R4
    if h2a_falsified:
        return {"rule": "R4",
                "framing": "Phantom-SoM partial drop-in (H2(a) cost equivalence falsified — "
                           "some cell median ratio > 1.20×); §4 disclosure + investigation",
                "hook_power": "WEAK",
                "heterogeneity_override": False}
    # H1 passed + H2(a) not falsified — primary R-rule by H3 axes
    if h3_axis1_pass and h3_axis2_pass:
        primary = ("R1", "Phantom routing space (2-axis empirical structure)", "STRONGEST")
    elif h3_axis1_pass or h3_axis2_pass:
        primary = ("R2", "Phantom routing space (single-axis empirical structure)", "MODERATE-STRONG")
    else:
        primary = ("R3", "Phantom-SoM is hidden 4th routing arm (workshop-grade)", "MODERATE")
    # I² cap-only — caps R1/R2 → R3, does NOT change H1 pass decision
    if h1_isq_cap_at_r3 and primary[0] in ("R1", "R2"):
        return {"rule": "R3",
                "framing": f"Heterogeneity-capped R3 — H1 FE pooled I² > {HETEROGENEITY_CAP_PCT:.0f}%; "
                           f"original R-rule {primary[0]} capped at R3 per prereg §2 L323 cap-only",
                "hook_power": "MODERATE",
                "heterogeneity_override": True,
                "original_rule_pre_cap": primary[0]}
    return {"rule": primary[0], "framing": primary[1], "hook_power": primary[2],
            "heterogeneity_override": False}


# ---------------------------------------------------------------------------
# Build canonical full decision
# ---------------------------------------------------------------------------

def build_full_decision(cells: List[Dict]) -> Dict:
    """End-to-end H1 + H2(a) + H3 axes + I² cap + framing rule."""
    per_cell_data = []
    skipped = []
    for cell in cells:
        # H1 per cell (reuses B-184 path → bit-identical to phase1_prereg_gate)
        h1_per_cell = _cell_drop_one_theta_se(cell)
        if h1_per_cell is None:
            skipped.append({"baseline": cell["baseline"], "site": cell["site"],
                            "reason": "H1 missing one of 6 modes OR below MIN_EP_FOR_CELL"})
            continue
        per_task = _load_cell_per_task(cell)
        h2a = _h2a_per_task_ratio(per_task)
        h3_axis1 = _h3_axis_per_cell(per_task, "P-text", ref_mode="P-SoM")
        h3_axis2 = _h3_axis_per_cell(per_task, "P-prompt", ref_mode="P-SoM")
        per_cell_data.append({
            "baseline": cell["baseline"],
            "site": cell["site"],
            "h1": h1_per_cell,
            "h2a": h2a,
            "h3_axis1": h3_axis1,
            "h3_axis2": h3_axis2,
        })

    payload: Dict = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "producer": "aggregate_phase1_full_prereg_decision.py (A1.21 B-481)",
        "prereg_section": "preregistration.md §2 H1/H2(a)/H3 + R1-R5 framing rule",
        "estimands": {
            "H1": "FE inverse-variance pool over P-SoM drop-one (6-mode universe), "
                  "one-sided superiority test H0: θ_FE ≤ +1.0pp at α=0.05",
            "H2(a)": "Per-task median cost ratio cost(P-SoM)/cost(DOM) per cell; "
                     f"ANY cell median ratio > 1.{int(H2A_MARGIN_PCT):02d}× → falsified → R4",
            "H3_axis1": "FE pool over per-cell |P-text \\ P-SoM| unique-count (paired bootstrap SE)",
            "H3_axis2": "FE pool over per-cell |P-prompt \\ P-SoM| unique-count (paired bootstrap SE)",
            "I_squared_cap": f"H1 FE pool I² > {HETEROGENEITY_CAP_PCT:.0f}% caps framing R1/R2 → R3 "
                             "(cap-only, does NOT rescue failed H1 per prereg L323)",
        },
        "alpha": ALPHA,
        "delta_pp": DELTA_PP,
        "bootstrap_B": PREREG_B,
        "bootstrap_seed": PREREG_SEED,
        "h2a_margin_pct": H2A_MARGIN_PCT,
        "heterogeneity_cap_pct": HETEROGENEITY_CAP_PCT,
        "per_cell": per_cell_data,
        "skipped_cells": skipped,
    }

    # Insufficient data branch
    if not per_cell_data:
        payload["gate_status"] = "INSUFFICIENT_DATA"
        payload["gate_status_reason"] = (
            "No cells contain all 6 modes. Phase 1a rerun likely still in flight."
        )
        return payload

    # H1 FE pool (reuses B-184 _fe_pool — bit-identical to phase1_prereg_gate.json)
    h1_per_cell_list = [c["h1"] for c in per_cell_data]
    fe = _fe_pool(h1_per_cell_list)
    if fe is None:
        payload["gate_status"] = "INSUFFICIENT_DATA"
        payload["gate_status_reason"] = (
            f"Only {len(per_cell_data)} cell(s) with all 6 modes; FE pool needs ≥2."
        )
        return payload

    # I² + Q on H1 FE pool (cap-only)
    thetas = np.array([r["theta_pp"] for r in h1_per_cell_list])
    ses = np.array([r["se_pp"] for r in h1_per_cell_list])
    n_zero_se = int((ses <= 0).sum())
    if n_zero_se > 0:
        # A1.19 B-426 floor (also used by _fe_pool → bit-identical FE pool)
        ses = np.where(ses <= 0, 1.0, ses)
    isq_payload = _compute_q_isq(thetas, ses)

    payload["pooled_h1_fe"] = fe
    payload["h1_heterogeneity"] = isq_payload

    # H2(a) falsification check — ANY cell falsified → R4
    h2a_per_cell = [c["h2a"] for c in per_cell_data if c["h2a"] is not None]
    h2a_falsified_cells = [c for c in per_cell_data
                            if c["h2a"] is not None and c["h2a"]["per_cell_falsified"]]
    h2a_cannot_evaluate = [c for c in per_cell_data if c["h2a"] is None]
    # A1.21 P1-7 fix: 3-state per cell (within / falsified / cannot_evaluate),
    # framing rule uses `n_falsified == 0` (NOT `pass_count == total`).
    payload["h2a_summary"] = {
        "n_cells_with_data": len(h2a_per_cell),
        "n_cells_within_band": sum(1 for c in h2a_per_cell if c["per_cell_pass"]),
        "n_cells_falsified": len(h2a_falsified_cells),
        "n_cells_cannot_evaluate": len(h2a_cannot_evaluate),
        "falsified": len(h2a_falsified_cells) > 0,
        "falsified_cells": [
            {"baseline": c["baseline"], "site": c["site"],
             "median_ratio": c["h2a"]["median_ratio"],
             "relative_diff_pct": c["h2a"]["relative_diff_pct"]}
            for c in h2a_falsified_cells
        ],
    }

    # H3 axis-1 FE pool
    h3a_per_cell = [c["h3_axis1"] for c in per_cell_data if c["h3_axis1"] is not None]
    if len(h3a_per_cell) >= 2:
        h3a_thetas = np.array([r["unique_count_pp"] for r in h3a_per_cell])
        h3a_ses = np.array([r["se_pp"] for r in h3a_per_cell])
        h3a_zero_se = int((h3a_ses <= 0).sum())
        if h3a_zero_se > 0:
            h3a_ses = np.where(h3a_ses <= 0, 1.0, h3a_ses)
        h3a_w = 1.0 / (h3a_ses ** 2)
        h3a_theta_fe = float(np.sum(h3a_w * h3a_thetas) / np.sum(h3a_w))
        h3a_se_fe = float(math.sqrt(1.0 / np.sum(h3a_w)))
        h3a_z = h3a_theta_fe / max(h3a_se_fe, 1e-12)
        h3a_p = 1.0 - _norm_cdf(h3a_z)  # one-sided H0: pooled count ≤ 0
        payload["h3_axis1_pooled_fe"] = {
            "k_cells": len(h3a_per_cell),
            "theta_FE_pp": h3a_theta_fe,
            "se_FE_pp": h3a_se_fe,
            "ci95_FE_lo_pp": h3a_theta_fe - 1.96 * h3a_se_fe,
            "ci95_FE_hi_pp": h3a_theta_fe + 1.96 * h3a_se_fe,
            "z_one_sided": h3a_z,
            "p_one_sided": h3a_p,
            "alpha": ALPHA,
            "passed": bool(h3a_p < ALPHA),
            "n_zero_se_floored_cells": h3a_zero_se,
        }
    else:
        payload["h3_axis1_pooled_fe"] = None

    # H3 axis-2 FE pool
    h3b_per_cell = [c["h3_axis2"] for c in per_cell_data if c["h3_axis2"] is not None]
    if len(h3b_per_cell) >= 2:
        h3b_thetas = np.array([r["unique_count_pp"] for r in h3b_per_cell])
        h3b_ses = np.array([r["se_pp"] for r in h3b_per_cell])
        h3b_zero_se = int((h3b_ses <= 0).sum())
        if h3b_zero_se > 0:
            h3b_ses = np.where(h3b_ses <= 0, 1.0, h3b_ses)
        h3b_w = 1.0 / (h3b_ses ** 2)
        h3b_theta_fe = float(np.sum(h3b_w * h3b_thetas) / np.sum(h3b_w))
        h3b_se_fe = float(math.sqrt(1.0 / np.sum(h3b_w)))
        h3b_z = h3b_theta_fe / max(h3b_se_fe, 1e-12)
        h3b_p = 1.0 - _norm_cdf(h3b_z)
        payload["h3_axis2_pooled_fe"] = {
            "k_cells": len(h3b_per_cell),
            "theta_FE_pp": h3b_theta_fe,
            "se_FE_pp": h3b_se_fe,
            "ci95_FE_lo_pp": h3b_theta_fe - 1.96 * h3b_se_fe,
            "ci95_FE_hi_pp": h3b_theta_fe + 1.96 * h3b_se_fe,
            "z_one_sided": h3b_z,
            "p_one_sided": h3b_p,
            "alpha": ALPHA,
            "passed": bool(h3b_p < ALPHA),
            "n_zero_se_floored_cells": h3b_zero_se,
        }
    else:
        payload["h3_axis2_pooled_fe"] = None

    # Apply framing rule with I² cap-only
    h1_pass = fe["gate_passed"]
    h2a_falsified = payload["h2a_summary"]["falsified"]
    h3a_pass = (payload["h3_axis1_pooled_fe"] is not None
                and payload["h3_axis1_pooled_fe"]["passed"])
    h3b_pass = (payload["h3_axis2_pooled_fe"] is not None
                and payload["h3_axis2_pooled_fe"]["passed"])
    h1_isq_cap = isq_payload.get("heterogeneity_cap_at_r3", False)

    framing = _apply_framing(h1_pass, h2a_falsified, h3a_pass, h3b_pass, h1_isq_cap)
    payload["framing_rule"] = framing

    # Gate status overall
    if len(per_cell_data) < 6:
        payload["gate_status"] = "PARTIAL_DATA"
        payload["gate_status_reason"] = (
            f"{len(per_cell_data)} of 6 planned cells with all 6 modes; pooled result reported "
            "but does NOT yet match the prereg estimand (which is exactly 6 cells)."
        )
    elif h1_pass and not h2a_falsified:
        payload["gate_status"] = "PASS"
        payload["gate_status_reason"] = (
            f"H1 FE superiority θ={fe['theta_FE_pp']:.3f}pp p={fe['p_one_sided']:.4f}; "
            f"H2(a) {payload['h2a_summary']['n_cells_within_band']}/"
            f"{payload['h2a_summary']['n_cells_with_data']} within band; "
            f"H3 axis-1 {'PASS' if h3a_pass else 'FAIL'}, axis-2 {'PASS' if h3b_pass else 'FAIL'}; "
            f"I²={isq_payload['I_squared_pct']:.1f}% (cap_at_r3={h1_isq_cap}); "
            f"framing rule {framing['rule']}."
        )
    else:
        payload["gate_status"] = "FAIL"
        payload["gate_status_reason"] = (
            f"H1 superiority {'PASS' if h1_pass else 'FAIL'}; "
            f"H2(a) {'falsified' if h2a_falsified else 'not falsified'}; "
            f"framing rule {framing['rule']}."
        )

    return payload


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(payload: Dict, out_json: Path, *,
                manifest_path: Optional[Path] = None,
                input_csv_path: Optional[Path] = None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Provenance lock — manifest + code + git SHAs (paper §1 OSF audit chain)
    payload["provenance"] = {
        "code_sha256": _self_code_sha(),
        "manifest_sha256": _file_sha256(manifest_path) if manifest_path else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "input_csv_sha256": _file_sha256(input_csv_path) if input_csv_path else None,
        "input_csv_path": str(input_csv_path) if input_csv_path else None,
        "git_commit_sha": _git_commit_sha(),
    }
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n",
                        encoding="utf-8")


def write_csv(payload: Dict, out_csv: Path) -> None:
    """Per-cell × 6 + pooled FE summary rows."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "row_type,baseline,site,k_cells,n_tasks,"
        "h1_theta_pp,h1_se_pp,h1_ci_lo_pp,h1_ci_hi_pp,h1_p_one_sided,"
        "h2a_median_ratio,h2a_rel_diff_pct,h2a_within_band,"
        "h3a_unique_count_pp,h3a_se_pp,h3b_unique_count_pp,h3b_se_pp,"
        "i_squared_pct,framing_rule,gate_status",
    ]
    gs = payload.get("gate_status", "UNKNOWN")
    framing_rule = payload.get("framing_rule", {}).get("rule", "")
    isq = payload.get("h1_heterogeneity", {}).get("I_squared_pct")
    isq_str = f"{isq:.2f}" if isq is not None else ""
    def _f(v: Optional[float]) -> str:
        return f"{v:.4f}" if v is not None else ""

    for r in payload["per_cell"]:
        h1 = r["h1"] or {}
        h2a = r["h2a"] or {}
        h3a = r["h3_axis1"] or {}
        h3b = r["h3_axis2"] or {}
        cell_line_parts = [
            "cell", r["baseline"], r["site"], "", str(h1.get("n_tasks", "")),
            _f(h1.get("theta_pp")), _f(h1.get("se_pp")),
            _f(h1.get("ci95_lo_pp")), _f(h1.get("ci95_hi_pp")),
            "",  # H1 per-cell p not directly emitted (FE pool gives pooled p)
            _f(h2a.get("median_ratio")), _f(h2a.get("relative_diff_pct")),
            str(h2a.get("per_cell_pass", "")),
            _f(h3a.get("unique_count_pp")), _f(h3a.get("se_pp")),
            _f(h3b.get("unique_count_pp")), _f(h3b.get("se_pp")),
            "", "", gs,
        ]
        lines.append(",".join(cell_line_parts))
    fe = payload.get("pooled_h1_fe")
    if fe is not None:
        h2a_sum = payload.get("h2a_summary", {})
        h3a_fe = payload.get("h3_axis1_pooled_fe") or {}
        h3b_fe = payload.get("h3_axis2_pooled_fe") or {}
        lines.append(
            f"pooled,,,{fe.get('k_cells', '')},,"
            f"{fe.get('theta_FE_pp', 0):.4f},{fe.get('se_FE_pp', 0):.4f},"
            f"{fe.get('ci95_FE_lo_pp', 0):.4f},{fe.get('ci95_FE_hi_pp', 0):.4f},"
            f"{fe.get('p_one_sided', 0):.6f},"
            f",,{'true' if not h2a_sum.get('falsified', False) else 'false'},"
            f"{h3a_fe.get('theta_FE_pp', 0):.4f},{h3a_fe.get('se_FE_pp', 0):.4f},"
            f"{h3b_fe.get('theta_FE_pp', 0):.4f},{h3b_fe.get('se_FE_pp', 0):.4f},"
            f"{isq_str},{framing_rule},{gs}"
        )
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_md(payload: Dict, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    fe = payload.get("pooled_h1_fe")
    isq = payload.get("h1_heterogeneity", {})
    h2a_sum = payload.get("h2a_summary", {})
    h3a_fe = payload.get("h3_axis1_pooled_fe")
    h3b_fe = payload.get("h3_axis2_pooled_fe")
    framing = payload.get("framing_rule", {})

    lines = [
        "# Phase 1 full prereg decision — H1 + H2(a) + H3 axes + framing rule",
        "",
        "**Producer**: `aggregate_phase1_full_prereg_decision.py` (A1.21 P0-2/P0-3/P0-4/P0-11, B-481).",
        "Canonical replacement for the H1-only `phase1_prereg_gate.{csv,json,md}` "
        "(B-184) + retired `preregistration_decision_test.py` (DL-contaminated path retired A1.21).",
        "",
        f"**Gate status**: `{payload.get('gate_status', 'UNKNOWN')}`",
        "",
        payload.get("gate_status_reason", ""),
        "",
        f"**Framing rule**: **{framing.get('rule', '?')}** — {framing.get('framing', '?')} "
        f"(hook_power: `{framing.get('hook_power', '?')}`)",
        "",
        "---",
        "",
        "## H1 — FE inverse-variance pool over P-SoM drop-one",
        "",
    ]
    if fe is not None:
        sig = "✅ **PASSED**" if fe["gate_passed"] else "❌ **NOT YET**"
        lines += [
            f"- **k = {fe['k_cells']}** cells",
            f"- **θ_FE = +{fe['theta_FE_pp']:.3f}pp** (SE = {fe['se_FE_pp']:.3f}pp)",
            f"- **95% CI**: [{fe['ci95_FE_lo_pp']:.3f}, {fe['ci95_FE_hi_pp']:.3f}]pp",
            f"- **z** = (θ_FE − {fe['delta_pp']}) / SE_FE = **{fe['z_one_sided']:.3f}**",
            f"- **p_one_sided** = 1 − Φ(z) = **{fe['p_one_sided']:.4f}**",
            f"- **Gate (p < α={fe['alpha']})**: {sig}",
        ]
        if fe.get("n_zero_se_floored_cells", 0) > 0:
            lines.append(f"- ⚠️ **SE floor fired**: {fe['n_zero_se_floored_cells']} cell(s) with bootstrap SE=0 "
                         "floored to 1.0pp per A1.19 B-426 + prereg §2 H1 anchor")
    lines.append("")

    if isq.get("I_squared_pct") is not None:
        cap_note = " ⚠️ **HIGH → cap framing at R3**" if isq.get("heterogeneity_cap_at_r3") else ""
        lines += [
            "### H1 heterogeneity (cap-only, NOT rescue per prereg L323)",
            "",
            f"- **Cochran's Q** = {isq['Q']:.3f} (df = {isq['df']})",
            f"- **I²** = {isq['I_squared_pct']:.1f}%{cap_note}",
            f"- **Cap threshold**: I² > {isq['heterogeneity_threshold_pct']:.0f}% → framing R1/R2 capped at R3",
            "",
        ]

    lines += [
        "## H2(a) — per-task cost ratio falsification check",
        "",
        f"- **Margin**: per-task median ratio cost(P-SoM)/cost(DOM) within ±{payload['h2a_margin_pct']:.0f}%",
        f"- **Cells with H2(a) data**: {h2a_sum.get('n_cells_with_data', 0)} / {len(payload['per_cell'])}",
        f"- **Within band**: {h2a_sum.get('n_cells_within_band', 0)}",
        f"- **Falsified** (median ratio > {1 + payload['h2a_margin_pct']/100:.2f}×): "
        f"{h2a_sum.get('n_cells_falsified', 0)}",
        f"- **Cannot evaluate**: {h2a_sum.get('n_cells_cannot_evaluate', 0)}",
        f"- **Overall H2(a) falsified**: {'❌ YES — R4 framing' if h2a_sum.get('falsified', False) else '✅ NO'}",
        "",
    ]
    if h2a_sum.get("falsified_cells"):
        lines.append("### Falsified cells (median ratio > 1.20×)")
        lines.append("")
        for c in h2a_sum["falsified_cells"]:
            lines.append(f"- **{c['baseline']} {c['site']}**: median ratio={c['median_ratio']:.3f} "
                         f"({c['relative_diff_pct']:+.2f}% rel diff)")
        lines.append("")

    for axis_idx, (axis_name, h3_fe) in enumerate([("axis-1 P-text", h3a_fe), ("axis-2 P-prompt", h3b_fe)], 1):
        if h3_fe is None:
            lines += [f"## H3 {axis_name} — insufficient data", ""]
            continue
        sig = "✅ **PASSED**" if h3_fe["passed"] else "❌ **NOT YET**"
        lines += [
            f"## H3 {axis_name} — FE inverse-variance pool over unique-count",
            "",
            f"- **k = {h3_fe['k_cells']}** cells",
            f"- **θ_FE = +{h3_fe['theta_FE_pp']:.3f}pp** (SE = {h3_fe['se_FE_pp']:.3f}pp)",
            f"- **95% CI**: [{h3_fe['ci95_FE_lo_pp']:.3f}, {h3_fe['ci95_FE_hi_pp']:.3f}]pp",
            f"- **z** = θ_FE / SE_FE = **{h3_fe['z_one_sided']:.3f}**",
            f"- **p_one_sided** = 1 − Φ(z) = **{h3_fe['p_one_sided']:.4f}**",
            f"- **Gate (p < α={h3_fe['alpha']})**: {sig}",
            "",
        ]

    if payload.get("skipped_cells"):
        lines += ["## Skipped cells", ""]
        for s in payload["skipped_cells"]:
            lines.append(f"- **{s['baseline']} {s['site']}** — {s['reason']}")
        lines.append("")

    # Provenance footer
    prov = payload.get("provenance", {})
    lines += [
        "---",
        "## Provenance lock (OSF audit chain)",
        "",
        f"- **Code SHA256**: `{prov.get('code_sha256', 'n/a')[:32]}...`",
        f"- **Manifest SHA256**: `{(prov.get('manifest_sha256') or 'n/a')[:32]}...` ({prov.get('manifest_path', 'default')})",
        f"- **Git commit SHA**: `{prov.get('git_commit_sha', 'n/a')}`",
        "",
    ]
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--run-manifest", default=None,
                    help="Path to run_manifest.yaml (default: results/phantom_paper/run_manifest.yaml via registry). "
                    "A1.21 P0-5 fix: this arg actually propagates to data discovery (was provenance theater).")
    ap.add_argument("--output-csv", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--output-md", default=str(DEFAULT_OUT_MD))
    args = ap.parse_args()

    # A1.21 P1-3 (B-499): lazy fn re-evaluates env var + manifest at call time
    cells_to_use = get_aggregator_cells()

    payload = build_full_decision(cells_to_use)
    manifest_path = (Path(args.run_manifest) if args.run_manifest
                     else REPO / "results/phantom_paper/run_manifest.yaml")

    write_csv(payload, Path(args.output_csv))
    write_json(payload, Path(args.output_json), manifest_path=manifest_path)
    write_md(payload, Path(args.output_md))

    framing = payload.get("framing_rule", {})
    print(f"[A1.21 B-481] gate_status={payload['gate_status']} "
          f"framing={framing.get('rule', '?')} "
          f"k_cells={len(payload['per_cell'])} skipped={len(payload['skipped_cells'])}")
    print(f"        → {args.output_csv}")
    print(f"        → {args.output_json}")
    print(f"        → {args.output_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
