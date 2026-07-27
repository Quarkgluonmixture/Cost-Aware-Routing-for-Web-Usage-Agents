#!/usr/bin/env python3
r"""Canonical full Phase 1 prereg decision producer — H1 + H2(a) + H3 axes + framing.

Built /stress A1.21 P0-2 + P0-3 + P0-4 + P0-11 (2026-05-17, B-515).

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
import os
import shutil
import statistics
import subprocess
import sys
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

# Reuse the same cell enumeration + H1 FE pool + Q/I² helpers from B-184.
# This guarantees H1 numbers are bit-identical between phase1_prereg_gate
# (H1-only legacy) and phase1_full_prereg_decision (canonical full).
from scripts.analysis.aggregate_phantom_lift import (  # noqa: E402
    MIN_EP_FOR_CELL,
    get_aggregator_cells,  # A1.21 P1-3 (B-530): lazy fn, was frozen CELLS constant
)
from scripts.analysis.lib.canonical_task_universe import (  # noqa: E402
    expected_scored_ids,
    protocol_excluded_in_universe,
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
from scripts.analysis.lib.atomic_io import (  # noqa: E402
    atomic_write_text,
    exclusive_file_lock,
    fsync_directory,
)
from scripts.analysis.lib.canonical_cells import PHASE_1A_PLANNED_CELLS  # noqa: E402
from scripts.analysis.lib.episode_rows import load_cell_task_rows  # noqa: E402

DEFAULT_OUT_CSV = REPO / "results/phantom_paper/phase1_full_prereg_decision.csv"
DEFAULT_OUT_JSON = REPO / "results/phantom_paper/phase1_full_prereg_decision.json"
DEFAULT_OUT_MD = REPO / "results/phantom_paper/phase1_full_prereg_decision.md"

# PROTOCOL_NOTE_06 is an explicitly isolated, temporary k=5 verdict channel.
# Its paths and metadata are deliberately not configurable through the canonical
# --output-* flags: invoking the authorization must never replace the registered
# six-cell artifact, even accidentally.
PROTOCOL_NOTE_06_PATH = (
    REPO / "docs/prereg_amendments/PROTOCOL_NOTE_06_K5_EARLY_VERDICT_20260716.md"
)
PROTOCOL_NOTE_06_OUT_JSON = (
    REPO / "results/phantom_paper/phase1_full_prereg_decision_pn06_k5.json"
)
PROTOCOL_NOTE_06_OUT_MD = (
    REPO / "results/phantom_paper/phase1_full_prereg_decision_pn06_k5.md"
)
PROTOCOL_NOTE_06_STATUS = "COMPLETE_K5_PROTOCOL_NOTE_06"
PROTOCOL_NOTE_06_QUALIFIER = "on the five landed cells"
PROTOCOL_NOTE_06_WITNESS_TAG = (
    "protocol-note-06-k5-early-verdict-signed-20260715"
)
PROTOCOL_NOTE_06_FIXED_CELLS = (
    ("classifieds", "B0"),
    ("classifieds", "B1"),
    ("classifieds", "B2"),
    ("reddit", "B0"),
    ("reddit", "B1"),
)
PROTOCOL_NOTE_06_FIXED_CELL_IDS = (
    "B0_classifieds",
    "B1_classifieds",
    "B2_classifieds",
    "B0_reddit",
    "B1_reddit",
)

# Prereg §2 H2(a) lock (2026-05-14 Decision 3A + A1.21 P0-9 prereg amend):
# per-task median ratio cost(P-SoM)/cost(DOM) within ±20% of 1.0 per cell.
# ANY cell violation → falsified → R4 framing.
H2A_MARGIN_PCT = 20.0  # +20% = 1.20× per prereg lock L120-145

# Prereg §2 heterogeneity rule (2026-05-14 Decision 3A + A1.21 P0-3 cap-only):
# I² > 75% caps framing power at R3, does NOT rescue failed H1.
HETEROGENEITY_CAP_PCT = 75.0

# Degenerate-cell SE floor, prereg §2 H1 (trigger codified at 0.68pp on
# 2026-05-18 per B-1003 / Appendix A; code aligned 2026-05-24 per AMENDMENT_03,
# witness tag `prereg-amendment-03-implementation-alignment-20260524`).
# B-1898 (2026-07-27): hoisted from function-local literals to module scope.
# AMENDMENT_03 calls this producer the SINGLE source that the transparency
# producer mirrors, but while both values were local literals that mirroring
# was untestable and held only by coincidence.  `test_h1_canonical_alignment`
# now imports both and asserts they agree.
SE_FLOOR_THRESHOLD_PP = 0.68
SE_FLOOR_REPLACE_PP = 1.0


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


def _prereg_sha() -> Optional[str]:
    """B-1004 (/stress A2.4a P1-7-A Claude, 2026-05-18): SHA of preregistration.md
    at runtime. Pre-fix `_self_code_sha` covered script + helper but NOT prereg —
    prereg amendments (A1.21 P0-9 per-task ratio, P0-10 Agresti-Coull anchor)
    after code freeze invisible to canonical output. Reviewer T₂ replay couldn't
    tell which prereg version was binding. Now `provenance.prereg_sha` field
    completes the audit trail.
    """
    return _file_sha256(REPO / "docs/checkpoints/pre_run/preregistration.md")


def _self_code_sha() -> str:
    """SHA of this script + the B-184 helper module (covers H1 + FE pool path)."""
    h = hashlib.sha256()
    for p in (
        Path(__file__),
        REPO / "scripts/analysis/aggregate_phase1_prereg_gate.py",
        REPO / "scripts/analysis/lib/canonical_task_universe.py",
        REPO / "scripts/analysis/lib/episode_rows.py",
    ):
        with p.open("rb") as f:
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Cell per-task data loader (success + cost, paired across modes)
# ---------------------------------------------------------------------------

def _load_cell_per_task(
    cell: Dict,
    *,
    rows_by_mode: Optional[Dict[str, Dict[int, Dict]]] = None,
) -> Dict[str, Dict[str, Dict]]:
    """Load per-mode per-task {success, cost} dict for one (baseline, site) cell.

    A1.21 P0-1 fix: `cost_raw = data.get('total_cost_usd')` 用 `is None` 检查
    避免 `or` short-circuit drop valid 0.0. Same fix in `generate_per_task_sr.py`.

    B-542 (/stress A1.5b Phase 2 P0-3-B codex OOB, 2026-05-17): paper-grade
    canonical first-number producer was bypassing `load_episode_summary_strict`
    (B-283) AND not rejecting B-486 quarantined episodes — quarantined rows
    entered H1/H2/H3 universe as failed denominator. Now strict + reject
    needs_reevaluation. Pre-fix audit trail: if any current archive row has
    quarantined state, this raises in strict mode → CI gate catches archive
    pollution before paper-grade aggregation. Use P79_STRICT=0 env to opt
    into lenient diagnostic mode (skip rather than raise).

    Returns: dict[mode] -> dict[task_id_str] -> {"success": float|None, "cost": float|None}
    """
    if rows_by_mode is None:
        rows_by_mode = load_cell_task_rows(cell, modes=SIX_MODES)
    by_mode: Dict[str, Dict[str, Dict]] = {}
    for mode in SIX_MODES:
        outcomes: Dict[str, Dict] = {}
        for tid, data in rows_by_mode.get(mode, {}).items():
            success_raw = data.get("success")  # bool (strict loader guaranteed)
            # P0-1 fix: `is None` check, NOT truthy `or` short-circuit
            # P1-3 (AMENDMENT_04 cost alignment 2026-05-24): canonical cost =
            # total_billed_cost_usd per AMENDMENT_01 §1 + AMENDMENT_03 §3. The H2(a)
            # per-task ratio + framing gate consume this column. AMENDMENT_03 §3 migrated
            # aggregate_cost_electricity + aggregate_h10_pareto to total_billed but missed
            # THIS canonical-gate producer (sibling-propagation gap). Legacy total_cost_usd
            # / total_model_cost_usd retained as fallback ONLY under P79_ALLOW_LEGACY_COST=1
            # (archive vintage); paper-grade fails closed to None if billed absent so a
            # missing-billed cell reports H2(a) state=cannot_evaluate, not a wrong-basis ratio.
            cost_raw = data.get("total_billed_cost_usd")
            if cost_raw is None:
                _allow_legacy_cost = os.environ.get(
                    "P79_ALLOW_LEGACY_COST", "0").lower() in ("1", "true", "yes")
                if _allow_legacy_cost:
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


def _psom_unique_ids(per_task: Dict[str, Dict[str, Dict]],
                     universe: Optional[set] = None) -> set:
    """Within-cell P-SoM-unique task IDs (NUMBERS_TODO §1.1 UNIQ slot rule).

    A task qualifies iff P-SoM success == 1.0 AND every one of the five other
    arms has an explicit success == 0.0.  Any None/missing success among the
    six arms excludes the task (fail-closed): uniqueness cannot be asserted
    against an arm whose outcome is unknown.
    """
    # AMENDMENT_08 (B-1901): restrict to the canonical SCORED set. Without this the
    # "each arm uniquely solves tasks no other arm does" slot — a paper-facing
    # figure, and one Paper A leans on harder now that H1 has failed — counts
    # protocol-excluded tasks. Measured before the fix: reddit union reported
    # `[11, 12, 58, 179]` (n=4), where 58 is a task AMENDMENT_08 removes; correct
    # count is 3. Missed by the same-day H1/H2a/H3 universe fix because this is a
    # separate function outside those three code paths.
    psom_rows = per_task.get("P-SoM", {})
    if universe is not None:
        _keep = {int(x) for x in universe}
        psom_rows = {t: r for t, r in psom_rows.items() if int(t) in _keep}
    other_modes = [m for m in SIX_MODES if m != "P-SoM"]
    unique: set = set()
    for tid, row in psom_rows.items():
        if row.get("success") != 1.0:
            continue
        others = [per_task.get(m, {}).get(tid, {}).get("success") for m in other_modes]
        if all(s == 0.0 for s in others):
            unique.add(int(tid))
    return unique


def _h2a_per_task_ratio(per_task: Dict[str, Dict[str, Dict]],
                        universe: Optional[set] = None) -> Optional[Dict]:
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
    # AMENDMENT_08: restrict to the canonical SCORED set. Without this, H2(a)
    # ran on every task present on disk (reddit 205) while H1 ran on the scored
    # set (203) — three hypotheses, two denominators, one output table.
    if universe is not None:
        common = [t for t in common if t in universe]
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
            # cost=0 is real (e.g., proxy edge / early-exit before any billed call);
            # (GLM-fallback path retired B-991 2026-05-17); skip ratio to avoid
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


def _six_arm_complete_case_universe(per_task: Dict[str, Dict[str, Dict]]) -> Optional[list]:
    """B-948 (/stress A2.3a P0-6-B*, 2026-05-17): six-arm complete-case task universe.

    Returns sorted list of task_ids where ALL 6 modes (DOM/SoM/Vision/P-text/
    P-prompt/P-SoM) ran and have a non-None `success` value. This matches
    `aggregate_phantom_lift.py:797-810` `universe_6` semantics (corrected from
    earlier comment citing :655-658 which was the 5-arm region; per Mode B
    A2.4a F3 P1-11 OOB B-1013 disclosure).

    B-1013 (/stress A2.4a P1-11-B* codex F3 OOB, 2026-05-18): task_id type
    assertion — pre-fix relied on string equality of task_id but lift script
    keys task ids from filenames as `int` (per `aggregate_phantom_lift.py:164-200`),
    while this script keys from summary `task_id` which may be str OR int
    depending on loader. Now: assert task_id type is int (post-load) OR
    string convertible to int; mixed types → fail-loud with diagnostic for
    operator. Defends against silent universe drift on str-vs-int comparison.
    """
    six_modes = ("DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM")
    mode_keysets = []
    for m in six_modes:
        mp = per_task.get(m, {})
        if not mp:
            return None  # any missing mode → six-arm universe undefined
        mode_keysets.append({t for t, rec in mp.items()
                              if rec.get("success") is not None})
    common = set.intersection(*mode_keysets) if mode_keysets else set()
    # B-1013: task_id type assertion (defends Mode B F3 attack vector). Detect
    # if loader produced mixed-type task_ids (e.g., some int, some "001" str).
    # Mixed types make set intersection silently smaller (str("1") != int(1)) →
    # universe drift vs lift script. Fail-loud diagnostic.
    if common:
        sample = next(iter(common))
        types_seen = {type(t).__name__ for t in common}
        if len(types_seen) > 1:
            raise TypeError(
                f"_six_arm_complete_case_universe task_id type mismatch: "
                f"observed types {sorted(types_seen)} in common set (sample={sample!r}). "
                f"Loader must produce uniform task_id type (int recommended per "
                f"aggregate_phantom_lift.py:164-200 convention). B-1013 fail-loud."
            )
    return sorted(common)


def _h3_axis_per_cell(per_task: Dict[str, Dict[str, Dict]], axis_mode: str,
                       ref_mode: str = "P-SoM", *,
                       universe: Optional[list] = None,
                       B: int = PREREG_B, seed: int = PREREG_SEED) -> Optional[Dict]:
    """H3 axis test: count tasks where axis_mode solved AND ref_mode did NOT.

    axis_mode = "P-text" (axis-1) or "P-prompt" (axis-2).
    Per-cell unique-count statistic + paired bootstrap SE → FE pool input.

    B-948 (/stress A2.3a P0-6-B*, 2026-05-17): accepts `universe` param —
    the six-arm complete-case task list from `_six_arm_complete_case_universe`.
    If None, falls back to legacy `axis ∩ ref` for backward compatibility,
    but the CANONICAL caller (`compute_full_decision`) now passes the
    six-arm universe matching `aggregate_phantom_lift.py:655-658`.

    Returns None if either mode missing or below MIN_EP_FOR_CELL.
    """
    axis = per_task.get(axis_mode, {})
    ref = per_task.get(ref_mode, {})
    if not axis or not ref:
        return None
    if universe is not None:
        # B-948: use six-arm complete-case universe (canonical, matches lift script)
        common = sorted(set(universe) & set(axis.keys()) & set(ref.keys()))
    else:
        # Legacy axis-only universe (kept for backward compat)
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
        # B-948: emit universe label so reviewers can verify which task
        # universe H3 was computed on (six-arm vs axis-only).
        "universe_label": "six_arm_complete_case" if universe is not None else "axis_intersection_legacy",
        "unique_count_pp": count_pp,
        "n_unique": int(unique.sum()),
        # Prereg H3(iii): the >=2-task threshold is a per-cell label only.
        # It MUST NOT filter the six-cell FE estimand (F4/P0-4).
        "cell_pass": bool(int(unique.sum()) >= 2),
        "se_pp": se_pp,
        "ci95_lo_pp": ci_lo,
        "ci95_hi_pp": ci_hi,
        "p_percentile_one_sided": p_percentile_one_sided,
        # B-1302 (/stress A2.3d P1-6-AC sibling of P0-1, 2026-05-18): expose the
        # per-cell bootstrap distribution so the H3 axis FE pool can compute the
        # B-1009 sibling bootstrap-percentile CI gate (H3 prereg §2 H3(i)/(ii)
        # gate = "FE CI excludes 0", consistent with H1 P0-1 fix).
        "boot_pp": boot.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# B-1009 paired-bootstrap pool replicate percentile p-value
# (B-1301 /stress A2.3d P0-1-AB*, 2026-05-18 — implementation companion to
# preregistration.md §2 H1 line 85 amendment which had been prose-only)
# ---------------------------------------------------------------------------

def _pool_bootstrap_percentile_p(per_cell_with_boot: List[Dict], *,
                                  theta_null_pp: float = 1.0,
                                  alpha: float = 0.05,
                                  floor_nonpositive_only: bool = False) -> Optional[Dict]:
    """Paired-bootstrap pool replicate FE percentile p-value.

    Implements the prereg-locked B-1009 primary gate (preregistration.md §2 H1
    line 85): one-sided bootstrap percentile p = `P(θ_FE* ≤ theta_null_pp)`
    over the per-cell bootstrap replicates pooled with IV weights.

    Operational semantics (B-1303 prereg L85 operational lock):
      - per-cell paired bootstrap replicates θ_i_b are reused from
        `_cell_drop_one_theta_se`'s cached `boot_pp` array (B=1000, seed=42 per
        B-176 + per-axis SHA seed per B-1006); NOT re-resampled here
      - per-cell IV weights `w_i = 1 / SE_i²` use the *point-estimate* SE_i
        from each cell's bootstrap std (subject to B-1003 Agresti-Coull
        threshold + B-426 floor); weights are held fixed across the B pool
        iterations (Davidson & MacKinnon 2000 / Hall 1992 standard paired
        bootstrap pool: bootstrap captures cell-level signal variability under
        fixed precision weighting; per-iter SE re-estimation would require
        nested 1M-resample inner loop with marginal accuracy gain at k=6)
      - per iter b: θ_FE_b = Σ(w_i · θ_i_b) / Σw_i ; pooled bootstrap
        distribution is the k-vector × B array stacked + IV-pooled
      - p_one_sided_bootstrap = (1/B) · |{b : θ_FE_b ≤ theta_null_pp}|
      - bootstrap percentile 95% two-sided CI = [q_0.025(θ_FE*), q_0.975(θ_FE*)]

    Why this implementation matches B-1009 amendment intent (codex Mode B F1 +
    Claude Mode A F1 OOB 2026-05-18 /stress A2.3d catch): pre-fix the canonical
    producer gated H1 on `p_one_sided = 1 - Φ(z)` (normal-Z Wald against
    δ=1.0pp) while prereg L85 promised bootstrap percentile. This left the
    prose↔code gap that broke OSF audit-trail reproducibility — Phase 1a fire
    would emit normal-Z primary in artifact JSON despite prereg-promised
    bootstrap percentile. This function closes the gap by computing both
    quantities (bootstrap percentile primary + normal-Z transparency
    transparency_p_one_sided_normal_approx via existing `_fe_pool`).
    """
    k = len(per_cell_with_boot)
    if k < 2:
        return None
    # Stack per-cell bootstrap replicate matrix: shape (k, B)
    boot_matrix = np.array([np.asarray(c["boot_pp"], dtype=np.float64)
                             for c in per_cell_with_boot])
    if boot_matrix.shape[0] != k:
        return None
    B_count = boot_matrix.shape[1]
    # Per-cell point-estimate SE (post B-1003 threshold-aware floor).
    # B-1898 (2026-07-27): the two constants were function-LOCAL here while
    # AMENDMENT_03 described this producer as the "SINGLE source mirrored by"
    # the transparency producer.  Local literals cannot be imported, so nothing
    # could actually check the mirroring — the two copies were merely equal by
    # coincidence.  Hoisted to module scope so the alignment is testable.
    ses = np.array([float(c["se_pp"]) for c in per_cell_with_boot])
    floor_mask = (ses <= 0) if floor_nonpositive_only else (ses < SE_FLOOR_THRESHOLD_PP)
    n_below_floor = int(floor_mask.sum())
    if n_below_floor > 0:
        ses = np.where(floor_mask, SE_FLOOR_REPLACE_PP, ses)
    w = 1.0 / (ses ** 2)
    sum_w = float(np.sum(w))
    # IV-weighted pool per iter b: θ_FE_b = Σ(w_i · θ_i_b) / Σw_i
    theta_fe_boot = (w[:, None] * boot_matrix).sum(axis=0) / sum_w
    p_one_sided_bootstrap = float((theta_fe_boot <= theta_null_pp).mean())
    ci95_lo_pp_bootstrap = float(np.quantile(theta_fe_boot, 0.025))
    ci95_hi_pp_bootstrap = float(np.quantile(theta_fe_boot, 0.975))
    # Bootstrap-distribution median (point estimate for symmetric reporting)
    theta_fe_bootstrap_median_pp = float(np.median(theta_fe_boot))
    return {
        "k_cells": k,
        "B_replicates": int(B_count),
        "theta_null_pp": theta_null_pp,
        "alpha": alpha,
        "p_one_sided_bootstrap": p_one_sided_bootstrap,
        "ci95_lo_pp_bootstrap": ci95_lo_pp_bootstrap,
        "ci95_hi_pp_bootstrap": ci95_hi_pp_bootstrap,
        "theta_fe_bootstrap_median_pp": theta_fe_bootstrap_median_pp,
        "gate_passed_bootstrap": bool(p_one_sided_bootstrap < alpha),
        "n_below_se_floor": n_below_floor,
        "n_zero_se_floored_cells": int((np.array(
            [float(c["se_pp"]) for c in per_cell_with_boot]
        ) <= 0).sum()),
        "se_floor_threshold_pp": SE_FLOOR_THRESHOLD_PP,
        "se_floor_replace_pp": SE_FLOOR_REPLACE_PP,
        "se_floor_rule": "ses<=0" if floor_nonpositive_only else "ses<0.68pp",
        "method_note": (
            "Davidson-MacKinnon 2000 / Hall 1992 paired-bootstrap pool: "
            "fixed point-estimate IV weights × per-cell bootstrap θ_i_b → "
            "pooled θ_FE_b distribution; one-sided percentile p at H0: "
            "θ_FE ≤ theta_null_pp; bootstrap percentile two-sided 95% CI."
        ),
    }


H3_MIN_UNIQUE_TASKS = 2
H3_REQUIRED_CELLS = 6


def _h3_axis_pooled_fe(
    per_cell_list: List[Dict], axis_name: str, *, k_required: int = H3_REQUIRED_CELLS,
) -> Dict:
    """Pool every available planned-cell H3 estimate without outcome filtering.

    ``n_unique < 2`` is recorded only through each row's ``cell_pass`` label and
    the noise-floor count.  Numeric interim pools are emitted at k=2..5, but an
    axis verdict is evaluated only for the preregistered exact-six-cell design.
    """
    k_input = len(per_cell_list)
    n_noise = sum(
        1 for r in per_cell_list if r.get("n_unique", 0) < H3_MIN_UNIQUE_TASKS
    )
    base = {
        "axis": axis_name,
        "k_cells": k_input,
        "k_cells_input": k_input,
        "k_cells_required": k_required,
        "n_noise_floor_cells": n_noise,
        # Backward-compatible field retained, now truthfully zero because F4
        # restores all data-bearing planned cells to the pool.
        "n_noise_floor_cells_skipped": 0,
        "n_cell_pass": k_input - n_noise,
        "noise_floor_threshold_unique_tasks": H3_MIN_UNIQUE_TASKS,
        "analysis_status": (
            "INSUFFICIENT" if k_input < 2
            else "COMPLETE" if k_input == k_required
            else "PARTIAL"
        ),
        "axis_verdict": "NOT_EVALUATED",
        "passed": None,
    }
    if k_input < 2:
        return {
            **base,
            "theta_FE_pp": None,
            "se_FE_pp": None,
            "ci95_FE_lo_pp": None,
            "ci95_FE_hi_pp": None,
            "ci95_lo_pp_bootstrap": None,
            "ci95_hi_pp_bootstrap": None,
            "p_one_sided_bootstrap": None,
            "theta_fe_bootstrap_median_pp": None,
            "z_one_sided": None,
            "p_one_sided": None,
            "alpha": ALPHA,
            "n_zero_se_floored_cells": 0,
            "reason": "fewer than 2 data-bearing cells; FE pool undefined",
        }

    thetas = np.array([r["unique_count_pp"] for r in per_cell_list])
    ses_raw = np.array([r["se_pp"] for r in per_cell_list])
    zero_mask = ses_raw <= 0
    n_zero_se = int(zero_mask.sum())
    # Normative A1.21 degenerate-cell rule for H3: only non-positive SEs are
    # replaced with the fixed 1.0pp floor; low-but-positive SEs remain inputs.
    ses = np.where(zero_mask, 1.0, ses_raw)
    w = 1.0 / (ses ** 2)
    theta_fe = float(np.sum(w * thetas) / np.sum(w))
    se_fe = float(math.sqrt(1.0 / np.sum(w)))
    ci_lo = theta_fe - 1.96 * se_fe
    ci_hi = theta_fe + 1.96 * se_fe
    z = theta_fe / max(se_fe, 1e-12)
    p_one_sided = 1.0 - _norm_cdf(z)
    boot_payload = _pool_bootstrap_percentile_p(
        per_cell_list,
        theta_null_pp=0.0,
        alpha=ALPHA,
        floor_nonpositive_only=True,
    )
    passed_ci_bootstrap = bool(
        boot_payload is not None
        and boot_payload.get("ci95_lo_pp_bootstrap") is not None
        and boot_payload["ci95_lo_pp_bootstrap"] > 0.0
    )
    complete = k_input == k_required
    return {
        **base,
        "theta_FE_pp": theta_fe,
        "se_FE_pp": se_fe,
        "ci95_FE_lo_pp": ci_lo,
        "ci95_FE_hi_pp": ci_hi,
        "ci95_lo_pp_bootstrap": (boot_payload or {}).get("ci95_lo_pp_bootstrap"),
        "ci95_hi_pp_bootstrap": (boot_payload or {}).get("ci95_hi_pp_bootstrap"),
        "p_one_sided_bootstrap": (boot_payload or {}).get("p_one_sided_bootstrap"),
        "theta_fe_bootstrap_median_pp": (boot_payload or {}).get(
            "theta_fe_bootstrap_median_pp"
        ),
        "z_one_sided": z,
        "p_one_sided": p_one_sided,
        "alpha": ALPHA,
        "gate_rule": (
            "bootstrap_percentile_CI_lower_bound > 0, evaluated only when "
            f"k_cells_input == {k_required}"
        ),
        "axis_verdict": (
            "PASS" if complete and passed_ci_bootstrap
            else "FAIL" if complete
            else "NOT_EVALUATED"
        ),
        "passed": passed_ci_bootstrap if complete else None,
        "interim_ci_excludes_zero": passed_ci_bootstrap,
        "passed_wald_ci_legacy": bool(ci_lo > 0.0),
        "passed_p_one_sided_legacy": bool(p_one_sided < ALPHA),
        "n_zero_se_floored_cells": n_zero_se,
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
# Holm-Bonferroni per-cell transparency counts (B-1054 /stress A2.3c P0-4-AB)
# ---------------------------------------------------------------------------

def _holm_per_cell_transparency(per_cell_theta_se: List[Tuple[float, float]],
                                alpha: float = 0.05) -> Dict:
    """B-1054 (/stress A2.3c Mode A F1 + Mode B B5, 2026-05-18): port
    `individually_holm_sig` transparency count from `preregistration_decision_test.py:507-560`
    to canonical producer. Pre-fix the canonical paper-grade gate artifact
    `phase1_prereg_gate.{csv,md}` did NOT emit the per-cell Holm-significance
    counts promised by prereg §3 line 408 + §4 line 446 + line 468:

      \"For H1 and for each H3 axis, report the count of cells (out of 6)
       whose per-cell bootstrap CI excludes 0, and the count individually
       Holm-significant.\"

    Only synthetic test fixture had this — canonical gate artifact was
    silent on the prereg-promised transparency row. Reviewer would diff
    paper §4 line 468 promise against artifact CSV and find nothing.

    Algorithm:
      1. Per-cell p_one_sided = 1 − Φ(θ_i / SE_i) (H0: θ_i ≤ 0)
      2. Apply Holm-Bonferroni step-down across N cells (m = N)
      3. Cell rejected at Holm-α=0.05 if p_holm_i < α
      4. Count rejected as n_individually_holm_sig

    Returns: {N, n_individually_holm_sig, alpha, per_cell_p_raw, per_cell_p_holm,
              per_cell_rejected, note}
    """
    n = len(per_cell_theta_se)
    if n == 0:
        return {"N": 0, "n_individually_holm_sig": 0, "alpha": alpha,
                "per_cell_p_raw": [], "per_cell_p_holm": [], "per_cell_rejected": [],
                "note": "n=0, no cells"}
    # Per-cell one-sided p value (H0: θ ≤ 0; positive lift in tail)
    p_raw_list = []
    for theta, se in per_cell_theta_se:
        if se is None or se <= 0:
            p_raw_list.append(1.0)  # degenerate cell, can't reject
            continue
        z = theta / se
        # 1 − Φ(z) one-sided upper tail
        p_one_sided = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2)))
        p_raw_list.append(float(p_one_sided))
    # Holm-Bonferroni step-down (Holm 1979 original)
    indexed = sorted(enumerate(p_raw_list), key=lambda x: x[1])
    p_holm_list = [None] * n
    rejected_list = [False] * n
    prev_adj = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = (n - rank) * p
        adj = max(adj, prev_adj)
        adj = min(adj, 1.0)
        p_holm_list[orig_idx] = adj
        rejected_list[orig_idx] = bool(adj < alpha)
        prev_adj = adj
    n_individually_holm_sig = sum(1 for r in rejected_list if r)
    return {
        "N": n,
        "n_individually_holm_sig": n_individually_holm_sig,
        "alpha": alpha,
        "per_cell_p_raw": [round(p, 6) for p in p_raw_list],
        "per_cell_p_holm": [round(p, 6) if p is not None else None for p in p_holm_list],
        "per_cell_rejected": rejected_list,
        "note": "Transparency count per prereg §3 line 408 + §4 line 468 (NOT a gate; primary H1/H3 gates are FE pooled superiority/CI-excludes-0 per §2.5/§2 H3)",
    }


# ---------------------------------------------------------------------------
# Framing rule R1-R5 (with I² cap-only, NOT rescue)
# ---------------------------------------------------------------------------

def _load_h10_operational_gate_passed() -> Optional[bool]:
    """Read H10 `operational_gate_passed` from h10_pareto_verdict.json for post-R5 routing.

    Returns True/False if the H10 verdict artifact exists + has the field, else None
    (H10 not yet computed, pre-Pass-2). amendment-02 §4 post-R5 route uses this to pick
    C_prime_router_only (H10 pass) vs F_failure (H10 fail) when H3 also fails.
    """
    path = DEFAULT_OUT_JSON.parent / "h10_pareto_verdict.json"
    if not path.exists():
        return None
    try:
        v = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    gate = v.get("operational_deployment_gate", {})
    val = gate.get("operational_gate_passed")
    return bool(val) if isinstance(val, bool) else None


def _apply_framing(h1_pass: bool, h2a_falsified: bool,
                    h3_axis1_pass: bool, h3_axis2_pass: bool,
                    h1_isq_cap_at_r3: bool, h10_pass: Optional[bool] = None,
                    design_scope: str = "over 6-cell design") -> Dict:
    """Apply prereg §2 R1-R5 framing rule with I² cap-only override.

    A1.21 P0-3 + P0-11 fix: I² > 75% caps R1/R2 → R3, but does NOT rescue
    failed H1 (prereg L323 + L340-342). Per-cell consistency substitution
    (which the retired decision_test had) is RETIRED.

    P1-1 (AMENDMENT_04, amendment-02 §4): on R5 (H1-fail) the framing TIER is unchanged
    (still R5) but the REPORTING route is mechanically determined by the independently
    pre-registered H3 / H10 gates → post_r5_pivot ∈ {C_prime_structure, C_prime_router_only,
    F_failure}. h10_pass=None (H10 not yet computed, pre-Pass-2) → pivot resolves to a
    pending marker. NO framing-tier rescue (R5 stays R5; amendment-02 anti-rescue guard).
    """
    # H1 failed → R5 regardless of I² or H2(a) or H3 (framing tier unchanged)
    # B-1288 /stress A2.6b P1-12-B 2026-05-18: R5 framing precise scope — falsifies
    # P-SoM deployment-arm superiority over the 6-cell design ONLY; does NOT
    # falsify phantom concept space existence or P-text/P-prompt structural
    # ablation evidence. Cross-link: paper_planning.md §5 R5 row + preregistration.md
    # §2.5 step 8 cross-family claim-tier gate (Qwen anchor is load-bearing).
    if not h1_pass:
        # P1-1 (amendment-02 §4 post-R5 reporting route): H3-pass → structure pivot;
        # H3-fail + H10-pass → router-only pivot; H3-fail + H10-fail → failure (Track B).
        h3_any = h3_axis1_pass or h3_axis2_pass
        if h3_any:
            pivot, pivot_desc = ("C_prime_structure",
                "H3 structural axis survives → lower-claim phantom-space-structure paper "
                "(P-text/P-prompt axis decomposition; NOT P-SoM deployment hero).")
        elif h10_pass is True:
            pivot, pivot_desc = ("C_prime_router_only",
                "H3 fails (neither axis) but H10 router deployable → lower-claim "
                "router-only / systems paper.")
        elif h10_pass is False:
            pivot, pivot_desc = ("F_failure",
                "H1 + H3 + H10 all fail → negative/methodology result; Track B B-91 "
                "evaluation-systems workshop note (prereg §2.5 R5 row).")
        else:
            pivot, pivot_desc = ("C_prime_router_only_or_F_pending_h10",
                "H3 fails; H10 verdict not yet computed (Pass-2 pending) → route resolves "
                "to C'-R (H10 pass) or F (H10 fail) once H10 lands.")
        return {"rule": "R5",
                "post_r5_pivot": pivot,
                "post_r5_pivot_desc": pivot_desc,
                "framing": f"H1 FE superiority failed {design_scope} (falsifies P-SoM "
                           "deployment-arm superiority claim, NOT phantom concept space "
                           "existence or P-text/P-prompt structural ablation evidence). "
                           "R5 tier unchanged; reporting route = " + pivot
                           + " per amendment-02 §4.",
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


def _apply_protocol_note_06_b1284_downgrade(framing: Dict) -> Dict:
    """Apply NOTE_06 §2's automatic one-tier downgrade to an available R-tier.

    This is intentionally separate from the registered six-cell B-1284 helper:
    the missing B2 Reddit cell itself activates the modifier, independent of the
    observed direction in the single landed Gemma-Classifieds cell.
    """
    rule = framing.get("rule")
    downgraded = {"R1": "R2", "R2": "R3"}.get(rule, rule)
    return {
        **framing,
        "rule": downgraded,
        "protocol_note_06_rule_pre_b1284_downgrade": rule,
        "protocol_note_06_b1284_modifier": (
            "automatic one-tier downgrade under NOTE_06 §2; R-tier capped at R2; "
            "no cross-site Gemma replication claim"
        ),
    }


def _apply_b2_cross_family_downgrade(framing: Dict, per_cell_data: List[Dict]) -> Dict:
    """prereg §2.5 step-8 (B-1284): B2 (Gemma) cross-family claim-tier downgrade.

    Per-cell H1 pass = bootstrap CI excludes 0 (ci95_lo_pp > 0). Rules:
    - Qwen-lineage {B0,B1}×{cls,red} 4-cell: ANY per-cell fail → R5 (load-bearing
      within-family anchor; prereg L412), overrides R1/R2/R3.
    - B2 {cls,red} 2-cell: ANY fail while Qwen 4-cell all pass → R-tier downgrade ONE
      step (R1→R2, R2→R3; R3 already lowest) — cross-family non-replication (prereg L411).
    - Incomplete data (None, e.g. pre-Pass-1 or missing cell) → no downgrade
      (conservative; await full 6-cell data).
    Only applies to R1/R2/R3 (H1 passed); R4/R5 terminal, returned unchanged.
    """
    rule = framing.get("rule")
    if rule in ("R4", "R5"):
        return framing

    def _cell_h1_pass(baseline: str, site: str) -> Optional[bool]:
        for c in per_cell_data:
            if c.get("baseline") == baseline and c.get("site") == site:
                h1 = c.get("h1")
                if not h1:
                    return None
                lo = h1.get("ci95_lo_pp")
                return (lo > 0) if isinstance(lo, (int, float)) else None
        return None

    qwen_cells = [("B0", "classifieds"), ("B0", "reddit"),
                  ("B1", "classifieds"), ("B1", "reddit")]
    b2_cells = [("B2", "classifieds"), ("B2", "reddit")]
    qwen_results = {f"{b}_{s}": _cell_h1_pass(b, s) for b, s in qwen_cells}
    b2_results = {f"{b}_{s}": _cell_h1_pass(b, s) for b, s in b2_cells}
    cross_family = {
        "qwen_lineage_per_cell_h1_pass": qwen_results,
        "b2_lineage_per_cell_h1_pass": b2_results,
        "rule_pre_cross_family": rule,
    }
    qwen_any_fail = any(v is False for v in qwen_results.values())
    qwen_all_pass = all(v is True for v in qwen_results.values())
    b2_any_fail = any(v is False for v in b2_results.values())

    if qwen_any_fail:
        return {**framing, "rule": "R5",
                "cross_family_override": "qwen_anchor_fail_r5",
                "cross_family_detail": {**cross_family,
                    "note": "prereg §2.5 L412: Qwen-lineage per-cell H1 fail → R5 "
                            "(load-bearing within-family replication), overrides FE-pool R-tier."}}
    if b2_any_fail and qwen_all_pass:
        downgrade = {"R1": "R2", "R2": "R3", "R3": "R3"}
        new_rule = downgrade.get(rule, rule)
        return {**framing, "rule": new_rule,
                "cross_family_override": "b2_nonreplication_downgrade",
                "cross_family_detail": {**cross_family,
                    "downgraded_to": new_rule,
                    "note": "prereg §2.5 L411: B2 (Gemma) per-cell H1 fail while Qwen 4-cell "
                            "pass → R-tier downgrade one step; phantom space + Qwen-validated "
                            "deployment claim survive."}}
    return {**framing, "cross_family_detail": cross_family}


# ---------------------------------------------------------------------------
# Build canonical full decision
# ---------------------------------------------------------------------------

def build_full_decision(
    cells: List[Dict], *,
    expected_ids_by_site: Optional[Dict[str, frozenset[int] | set[int]]] = None,
    protocol_note_06_k5: bool = False,
) -> Dict:
    """End-to-end H1 + H2(a) + H3 axes + I² cap + framing rule."""
    per_cell_data = []
    skipped = []
    # UNIQ_CLS/UNIQ_RED registered slot source (NUMBERS_TODO §1.1 rows 78-79):
    # per site, union across landed backbones of {t: P-SoM succeeds AND every
    # other menu arm explicitly fails within that cell}, deduplicated task IDs.
    # Tasks with any None success among the six arms are excluded (fail-closed).
    site_unique_psom: Dict[str, set] = {}
    site_unique_cells: Dict[str, List[str]] = {}
    for cell in cells:
        rows_by_mode = load_cell_task_rows(cell, modes=SIX_MODES)
        # H1 per cell (reuses B-184 path → bit-identical to phase1_prereg_gate)
        expected_ids = (
            expected_ids_by_site.get(cell["site"])
            if expected_ids_by_site is not None else None
        )
        # AMENDMENT_08: resolve the canonical scored set HERE rather than relying on
        # `_cell_drop_one_theta_se`'s internal fallback. The fallback gave H1 the
        # right universe while leaving `expected_ids=None` at this call site, so the
        # H2(a) / H3 intersection below silently no-op'd and those two hypotheses kept
        # running on the wider on-disk set. One resolved variable, three hypotheses.
        if expected_ids is None:
            expected_ids = expected_scored_ids(cell["site"])[0]
        h1_per_cell = _cell_drop_one_theta_se(
            cell, expected_ids=expected_ids, rows_by_mode=rows_by_mode,
            # Passing `expected_ids` explicitly opts out of the gate's own
            # AMENDMENT_08 carve-out, so hand it the tolerated ids directly —
            # otherwise every reddit cell fails `complete_exact` again.
            tolerate_extra_ids=protocol_excluded_in_universe(cell["site"]),
        )
        if not h1_per_cell["complete_exact"]:
            skipped.append({
                **h1_per_cell,
                "reason": h1_per_cell["incomplete_reason"],
            })
            continue
        per_task = _load_cell_per_task(cell, rows_by_mode=rows_by_mode)
        # B-948 (/stress A2.3a P0-6-B*, 2026-05-17): compute six-arm complete-
        # case universe ONCE per cell + pass to both axis tests → matches
        # `aggregate_phantom_lift.py:655-658` universe semantics. Pre-fix
        # axis-only intersection drifted H3 universe vs lift script.
        six_arm_universe = _six_arm_complete_case_universe(per_task)
        # AMENDMENT_08: `_six_arm_complete_case_universe` derives the universe from
        # WHAT IS ON DISK (every task with all six modes present). That was the right
        # semantics when the scored set equalled the collected set; post-amendment it
        # is 2 reddit tasks too wide, and it feeds H2(a) + both H3 axes — including
        # the H3 verdict the R5 fallback route rests on. Intersect with the canonical
        # scored set so all three hypotheses share one denominator.
        if six_arm_universe is not None and expected_ids is not None:
            # `per_task` keys are str (loader convention) while
            # `expected_scored_ids` yields int — B-1013's fail-loud assertion only
            # rejects MIXED types inside the common set, so a uniform-but-different
            # type against an outside comparison set slips through it. Normalise
            # both sides to int before intersecting; a naive `t in set(expected_ids)`
            # silently yields the empty set (verified: 0 vs 203).
            _keep = {int(x) for x in expected_ids}
            six_arm_universe = [t for t in six_arm_universe if int(t) in _keep]
            if not six_arm_universe:
                raise ValueError(
                    f"{cell['baseline']}_{cell['site']}: six-arm universe is EMPTY after "
                    f"intersecting with the canonical scored set. Almost certainly a "
                    f"task_id type mismatch, not real data loss. Refusing to emit — an "
                    f"empty H2(a) reports 'not falsified' because falsification is "
                    f"impossible on zero tasks, which reads as a pass."
                )
        h2a = _h2a_per_task_ratio(
            per_task, universe=set(six_arm_universe) if six_arm_universe is not None else None
        )
        # B-1006 (/stress A2.4a P2-20-B* codex F8 OOB, 2026-05-18): per-axis seed
        # stratification. Pre-fix both axis1+axis2 called `_h3_axis_per_cell`
        # with identical `PREREG_SEED=42` → identical bootstrap resample indices →
        # artificial covariance in axis1/axis2 reported uncertainty (CI bands
        # share resample noise). Point estimates unchanged; this is uncertainty
        # hygiene per retired script `preregistration_decision_test.py:449-458`
        # SHA-derived seed stratification pattern.
        # Root seed PREREG_SEED preserved in metadata for OSF replay.
        cell_id = f"{cell['baseline']}_{cell['site']}"
        axis1_seed = int(hashlib.sha256(
            f"{cell_id}|axis1|{PREREG_SEED}".encode()).hexdigest()[:8], 16)
        axis2_seed = int(hashlib.sha256(
            f"{cell_id}|axis2|{PREREG_SEED}".encode()).hexdigest()[:8], 16)
        h3_axis1 = _h3_axis_per_cell(per_task, "P-text", ref_mode="P-SoM",
                                      universe=six_arm_universe, seed=axis1_seed)
        h3_axis2 = _h3_axis_per_cell(per_task, "P-prompt", ref_mode="P-SoM",
                                      universe=six_arm_universe, seed=axis2_seed)
        cell_unique = _psom_unique_ids(per_task, universe=set(expected_ids))
        site_unique_psom.setdefault(cell["site"], set()).update(cell_unique)
        site_unique_cells.setdefault(cell["site"], []).append(
            f"{cell['baseline']}_{cell['site']} (+{len(cell_unique)})"
        )
        per_cell_data.append({
            "baseline": cell["baseline"],
            "site": cell["site"],
            "h1": h1_per_cell,
            "h2a": h2a,
            "h3_axis1": h3_axis1,
            "h3_axis2": h3_axis2,
        })

    # B-1017 (/stress A2.4a P0-2-AC* Claude+gemini 2-AI overlap, 2026-05-18):
    # paper §1 hero substitution table emit at metadata level — appears in
    # output regardless of gate_status (so reviewer can read substitution map
    # even when Phase 1a not yet fired, INSUFFICIENT_DATA branch). Filled in
    # later post-H1-FE-pool with `<INSUFFICIENT_DATA>` if early-return triggers.
    paper_hero_substitution_table = {
        "purpose": (
            "Maps archive-era paper §1 hero numbers (4-mode universe, pre-§139.8) "
            "to post-Phase-1a-fire 6-mode FE-pool canonical equivalents. Reviewer / "
            "OSF auditor / paper §1 substitution-checklist consumer: read this field "
            "post-fire, mechanically swap section1_intro.md hardcoded numbers per row. "
            "B-1017 closes A2.4a P0-2-AC* 'no binding mechanism for hero substitution' "
            "attack (fn 13 PRE-FIRE STUB was textual promise pre-B-1017)."
        ),
        "section1_intro_md_line_reference": "line 11 hero paragraph + fn 13 [^hero-estimand-scope]",
        "substitutions": [
            {"claim_text_archive": "Phantom-SoM contributes 3.33pp incremental oracle on reddit",
             "archive_value": "3.33pp",
             "canonical_field": "pooled_h1_fe.theta_FE_pp",
             "canonical_value_post_fire": "<filled-in-post-FE-pool>",
             "scope_archive": "4-mode {DOM, SoM, Vision, P-SoM} per-cell drop-one B0 reddit",
             "scope_canonical": "6-mode FE-pool over 6 planned cells"},
            {"claim_text_archive": "task-resampling 95% two-sided CI [+0.95, +6.19] reddit",
             "archive_value": "[+0.95, +6.19]",
             "canonical_field": "pooled_h1_fe.ci95_lo_pp + .ci95_hi_pp (B-1009 bootstrap percentile)",
             "canonical_value_post_fire": "<filled-in-post-FE-pool>",
             "scope_archive": "per-cell paired bootstrap 95% CI B0 reddit",
             "scope_canonical": "6-mode FE-pool 95% two-sided percentile CI"},
            {"claim_text_archive": "2.56pp classifieds CI [+0.85, +4.70]",
             "archive_value": "2.56pp / [+0.85, +4.70]",
             "canonical_field": "per-cell list filtered by site=classifieds + pooled_h1_fe FE-aggregated",
             "canonical_value_post_fire": "<see per_cell entries>",
             "scope_archive": "per-cell drop-one B0 classifieds archive 4-mode",
             "scope_canonical": "6-mode universe; per-site disagg in §6 forest if site-level claim retained"},
        ],
        "substitution_checklist": [
            "1. Read paper_hero_substitution_table[i].canonical_value_post_fire from this JSON.",
            "2. If '<INSUFFICIENT_DATA>' / '<filled-in-post-FE-pool>' → Phase 1a not yet fired; do not update §1.",
            "3. Else: section1_intro.md grep for substitutions[i].archive_value → replace.",
            "4. Verify CI is 95% TWO-SIDED PERCENTILE BOOTSTRAP (B-1009 method amend post-2026-05-18).",
            "5. Move archive-era values to Appendix-D as cross-rerun sensitivity (fn 13 promise).",
            "6. Re-run aggregate_phase1_full_prereg_decision after edit to verify substitution self-consistency.",
            "7. OSF lock manifest commit references this paper_hero_substitution_table by code_sha + prereg_sha (B-1004).",
        ],
    }

    payload: Dict = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "producer": "aggregate_phase1_full_prereg_decision.py (A1.21 B-515)",
        "paper_hero_substitution_table": paper_hero_substitution_table,  # B-1017
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
        # Registered NUMBERS_TODO §1.1 UNIQ_CLS/UNIQ_RED source fields.
        # Union covers exactly the landed complete_exact cells in `per_cell`
        # (at k=5 this is the PROTOCOL_NOTE_06 fixed set; at k=6 all backbones).
        "site_unique_psom_union": {
            site: {
                "n_unique_task_ids": len(ids),
                "task_ids": sorted(ids),
                "cells_in_union": site_unique_cells.get(site, []),
                "rule": "P-SoM success AND all five other arms explicit fail, per cell; task IDs deduplicated across backbones",
            }
            for site, ids in sorted(site_unique_psom.items())
        },
    }
    if protocol_note_06_k5:
        payload.update({
            "protocol_note": "PROTOCOL_NOTE_06",
            "verdict_qualifier": PROTOCOL_NOTE_06_QUALIFIER,
            "b1284_one_tier_downgrade": True,
            "r_tier_cap": "R2",
            "fixed_cell_set": list(PROTOCOL_NOTE_06_FIXED_CELL_IDS),
            "witness_tag": PROTOCOL_NOTE_06_WITNESS_TAG,
        })

    planned_source = (
        PROTOCOL_NOTE_06_FIXED_CELLS
        if protocol_note_06_k5 else PHASE_1A_PLANNED_CELLS
    )
    planned = {(site, baseline) for site, baseline in planned_source}
    exact_cells = {(c["site"], c["baseline"]) for c in per_cell_data}
    if len(per_cell_data) < 2:
        payload["analysis_status"] = "INSUFFICIENT"
    elif len(per_cell_data) == len(planned) and exact_cells == planned:
        payload["analysis_status"] = (
            PROTOCOL_NOTE_06_STATUS if protocol_note_06_k5 else "COMPLETE"
        )
    else:
        payload["analysis_status"] = "PARTIAL"
    payload["h1_verdict"] = "NOT_EVALUATED"

    # B-1002 (/stress A2.4a P0-1-A* Claude OOB, 2026-05-18): paper-grade strict
    # k=6 gate. Pre-fix code branched k<2→INSUFFICIENT_DATA, k=2-5→silently pools.
    # prereg §2 H1 line 68-86 locks "FE pool over 6 planned cells" — silent k<6
    # pool would let Phase 1a fire emit hero θ_FE on degraded data without paper
    # §1 prose disclosure (reviewer R3 silent-fallback attack). New 3-state:
    #   k=0       → INSUFFICIENT_DATA
    #   1≤k<2     → INSUFFICIENT_DATA (FE needs ≥2)
    #   2≤k<6 + paper-grade strict → DEGRADED (emit + warn, but distinct status)
    #   k=6       → 正常 emit
    # Strict mode enabled via P79_PAPER_GRADE env (default 1 per A2.2 B-548 lib).
    K_REQUIRED_PAPER_GRADE = 5 if protocol_note_06_k5 else 6
    paper_grade_strict = os.environ.get("P79_PAPER_GRADE", "1") == "1"

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

    # B-1002 strict k=6 gate (paper-grade only; dev opt-out via P79_PAPER_GRADE=0)
    k_actual = len(per_cell_data)
    if paper_grade_strict and k_actual < K_REQUIRED_PAPER_GRADE:
        payload["gate_status"] = "DEGRADED"
        payload["gate_status_reason"] = (
            f"k_actual={k_actual} < k_required={K_REQUIRED_PAPER_GRADE} planned cells "
            f"(P79_PAPER_GRADE=1 strict). Missing: {[(s['baseline'], s['site']) for s in skipped]}. "
            f"FE θ_FE = {fe['theta_FE_pp']:.3f}pp on degraded pool — paper §1 prose MUST "
            f"disclose degradation per prereg §2 Appendix-E k-degradation protocol (TBD pre-fire)."
        )
        payload["k_actual"] = k_actual
        payload["k_required"] = K_REQUIRED_PAPER_GRADE
        payload["pooled_h1_fe"] = fe  # still emit for transparency
        # Continue to emit H2(a) / H3 / framing for transparency, but gate_status
        # signals DEGRADED so downstream consumers don't substitute into paper §1 silently.

    # I² + Q on H1 FE pool (cap-only)
    thetas = np.array([r["theta_pp"] for r in h1_per_cell_list])
    ses = np.array([r["se_pp"] for r in h1_per_cell_list])
    # B-1003 (/stress A2.4a P1-5-A* Claude OOB, 2026-05-18): SE floor anchor at
    # Agresti-Coull bound 0.68pp per prereg §2 H1 line 99 (was literal `<= 0`).
    # Pre-fix floor fired only on literal zero; SE = 0.05pp from a single-unique-
    # task bootstrap (degenerate-but-nonzero scenario) didn't trigger floor →
    # FE weight 1/SE² = 400× hijacks pool, opposite of prereg amendment intent.
    # Threshold = 0.68pp (Agresti-Coull anchor `√(p_AC × (1-p_AC) / (N+z²))`
    # at x=0, N=200). Archive empirical median SE 0.98pp ≈ post-floor anchor.
    SE_FLOOR_THRESHOLD_PP = 0.68
    SE_FLOOR_REPLACE_PP = 1.0  # prereg §2 H1 lock — replace below-threshold with 1.0pp finite bound
    n_below_floor = int((ses < SE_FLOOR_THRESHOLD_PP).sum())
    n_zero_se = int((ses <= 0).sum())  # legacy stat retained for transparency
    if n_below_floor > 0:
        # A1.19 B-426 floor + B-1003 threshold-aware (also used by _fe_pool path)
        ses = np.where(ses < SE_FLOOR_THRESHOLD_PP, SE_FLOOR_REPLACE_PP, ses)
    isq_payload = _compute_q_isq(thetas, ses)

    payload["pooled_h1_fe"] = fe
    payload["h1_heterogeneity"] = isq_payload

    # B-1301 (/stress A2.3d P0-1-AB* 3-AI overlap OOB, 2026-05-18): compute the
    # B-1009 prereg-locked primary bootstrap percentile FE pool p-value via
    # `_pool_bootstrap_percentile_p` over per-cell bootstrap distributions now
    # exposed by `_cell_drop_one_theta_se` (B-1301 substrate fix). Pre-fix
    # `fe["gate_passed"]` was the primary gate but computed off normal-Z Wald
    # p_one_sided despite prereg L85 amend promising bootstrap percentile; this
    # left the prose↔code gap that broke OSF audit-trail reproducibility.
    # Both p-values now emit: `pooled_h1_bootstrap.p_one_sided_bootstrap`
    # (PRIMARY, drives `h1_pass`) + `pooled_h1_fe.p_one_sided` (legacy normal-Z,
    # retained as transparency channel per amendment).
    pooled_h1_bootstrap = _pool_bootstrap_percentile_p(
        h1_per_cell_list, theta_null_pp=DELTA_PP, alpha=ALPHA,
    )
    payload["pooled_h1_bootstrap"] = pooled_h1_bootstrap

    # B-1054 (/stress A2.3c Mode A F1 + Mode B B5, 2026-05-18): per-cell
    # Holm-significance transparency count for H1 (prereg §3 line 408 + §4
    # line 446 + line 468 promise). Uses post-floor SE per B-1003 / B-426.
    h1_per_cell_theta_se = [(float(t), float(s)) for t, s in zip(thetas, ses)]
    payload["transparency_H1"] = _holm_per_cell_transparency(h1_per_cell_theta_se, alpha=ALPHA)

    # H2(a) falsification check — ANY cell falsified → R4
    h2a_per_cell = [c["h2a"] for c in per_cell_data if c["h2a"] is not None]
    h2a_falsified_cells = [c for c in per_cell_data
                            if c["h2a"] is not None and c["h2a"]["per_cell_falsified"]]
    h2a_cannot_evaluate = [c for c in per_cell_data if c["h2a"] is None]
    # A1.21 P1-7 fix: 3-state per cell (within / falsified / cannot_evaluate),
    # framing rule uses `n_falsified == 0` (NOT `pass_count == total`).
    # B-1018 (/stress A2.4a P1-15-C gemini F2, 2026-05-18): zero-cost task share
    # disclosure. Pre-fix `_h2a_per_task_ratio` skipped n_dom_zero tasks (divide-
    # by-zero) silently — if DOM-failure tasks systematically have cost=0
    # (e.g., proxy edge / DOM early-exit before billed call; GLM-rescue retired
    # B-991), median ratio computed
    # only over DOM-SUCCESS subset → survival bias on H2(a) cost-equivalence
    # claim, paper §1 "cost ≈ DOM" actually proven only for DOM-pass tasks.
    # Emit per-cell + aggregate n_dom_zero_skipped/n_paired_tasks ratio so
    # reviewer can quantify survival-bias risk + invoke sensitivity check.
    n_dom_zero_total = sum((c["h2a"].get("n_dom_zero_skipped", 0) or 0)
                           for c in per_cell_data if c["h2a"] is not None)
    n_paired_total = sum((c["h2a"].get("n_paired_tasks", 0) or 0)
                          for c in per_cell_data if c["h2a"] is not None)
    dom_zero_ratio = (100.0 * n_dom_zero_total / n_paired_total) if n_paired_total > 0 else None

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
        # B-1018: zero-cost survival-bias disclosure (paper §1 cost ≈ DOM claim
        # only holds on DOM-pass subset when n_dom_zero is non-trivial)
        "n_dom_zero_skipped_total": n_dom_zero_total,
        "n_paired_tasks_total": n_paired_total,
        "dom_zero_ratio_pct": dom_zero_ratio,
        "dom_zero_disclosure_required": (dom_zero_ratio is not None and dom_zero_ratio > 5.0),
    }

    # H3 axis-1 FE pool
    h3a_per_cell = [c["h3_axis1"] for c in per_cell_data if c["h3_axis1"] is not None]
    # B-1007 (/stress A2.4a P1-10-B* codex F2 OOB, 2026-05-18): Holm m=2 across
    # {axis1, axis2} H3 sub-family for the legacy p-value transparency channel.
    # Note: canonical CI-based gate (B-949) is CLOSED-FORM decision rule, NOT
    # p-family — no FWER correction technically required for the `passed` field.
    # But the legacy `passed_p_one_sided_legacy` transparency p-value should
    # carry Holm m=2 so reviewer demanding FWER (m=2 H3 sub-family per prereg
    # §3 family rules) finds the correction emitted. Compute Holm AFTER both
    # axes pool, attach `passed_p_holm_m2` field to each axis result.
    h3a_result = _h3_axis_pooled_fe(
        h3a_per_cell, "axis1", k_required=K_REQUIRED_PAPER_GRADE,
    )
    payload["h3_axis1_pooled_fe"] = h3a_result

    # B-1054 (/stress A2.3c Mode A F1 + Mode B B5, 2026-05-18): per-cell
    # Holm-significance transparency count for H3 axis-1 (prereg §3 line 408 +
    # §4 line 446 + line 468 promise — H3 axes parity with H1).
    h3a_theta_se = [(float(c.get("unique_count_pp") or 0.0), float(c.get("se_pp") or 0.0))
                    for c in h3a_per_cell if c.get("se_pp") is not None]
    payload["transparency_H3_axis1"] = _holm_per_cell_transparency(h3a_theta_se, alpha=ALPHA)

    # H3 axis-2 FE pool
    h3b_per_cell = [c["h3_axis2"] for c in per_cell_data if c["h3_axis2"] is not None]
    h3b_result = _h3_axis_pooled_fe(
        h3b_per_cell, "axis2", k_required=K_REQUIRED_PAPER_GRADE,
    )
    payload["h3_axis2_pooled_fe"] = h3b_result

    # B-1054 H3 axis-2 transparency count (parity with H1 + axis-1)
    h3b_theta_se = [(float(c.get("unique_count_pp") or 0.0), float(c.get("se_pp") or 0.0))
                    for c in h3b_per_cell if c.get("se_pp") is not None]
    payload["transparency_H3_axis2"] = _holm_per_cell_transparency(h3b_theta_se, alpha=ALPHA)

    # B-1007 Holm m=2 correction across H3 axis-1 + axis-2 transparency p-values.
    # Sorted ascending: smallest p compared against α/m, next against α/(m-1).
    # Closed-form CI gate (B-949) is independent and already canonical.
    h3_p_pairs = []
    if h3a_result.get("p_one_sided") is not None:
        h3_p_pairs.append(("axis1", h3a_result["p_one_sided"], h3a_result))
    if h3b_result.get("p_one_sided") is not None:
        h3_p_pairs.append(("axis2", h3b_result["p_one_sided"], h3b_result))
    h3_p_pairs.sort(key=lambda x: x[1])
    m = len(h3_p_pairs)
    for rank, (axis_name, p_raw, result_dict) in enumerate(h3_p_pairs):
        # Holm step-down: p_holm = min(1, max(over rejected so far) of p_raw * (m - rank))
        # Simple form for m=2: p_holm[0] = p_raw[0]*2, p_holm[1] = max(p_holm[0], p_raw[1]*1)
        p_holm = min(1.0, p_raw * (m - rank))
        if rank > 0:
            p_holm = max(p_holm, h3_p_pairs[rank - 1][2].get("p_holm_m2_legacy", 0.0))
        result_dict["p_holm_m2_legacy"] = p_holm
        result_dict["passed_p_holm_m2_legacy"] = bool(p_holm < ALPHA)
        # B-1897 (/stress 2026-07-27, Mode C P0 OOB): the previous wording read
        # "Canonical gate is CI_lower_bound > 0 (closed-form, no p-family). Holm
        # m=2 applied to legacy p-value for transparency only." That rationale is
        # wrong: a 95% CI excluding 0 is isomorphic to rejecting at α=0.05, so two
        # uncorrected 95% CIs over {axis1, axis2} inflate FWER exactly as two
        # uncorrected p-values do. "Closed-form" is not an exemption from
        # multiplicity — being an interval rather than a p-value changes the
        # presentation, not the error rate.
        #
        # The correction was always COMPUTED (p_holm_m2_legacy); only the stated
        # justification for treating it as optional was defective. This note now
        # reports the Holm verdict as load-bearing rather than decorative, which
        # matters more after 2026-07-27 than before: with H1 failed, the R5
        # fallback route (C_prime_structure) rests entirely on these two axes, so
        # the family is the paper's claim, not a side panel.
        result_dict["family_correction_note"] = (
            "Family = {H3 axis1, axis2}, m=2. Both the CI gate and the Holm-adjusted "
            "p-value are reported and BOTH must hold; the CI is not exempt from "
            "multiplicity (a 95% CI excluding 0 is isomorphic to a test at alpha=0.05). "
            "For a strictly FWER-controlled interval read the CI at 1-alpha/m. "
            "Supersedes the pre-B-1897 note that called the CI gate 'closed-form, no "
            "p-family'."
        )
        result_dict["family_size_m"] = 2
        result_dict["ci_level_for_fwer_control"] = 1.0 - ALPHA / 2.0

    # Apply framing rule with I² cap-only
    # B-1301 (/stress A2.3d P0-1-AB* 3-AI overlap OOB, 2026-05-18): primary H1
    # gate switched from normal-Z `fe["gate_passed"]` to bootstrap percentile
    # `pooled_h1_bootstrap["gate_passed_bootstrap"]` per prereg §2 H1 L85
    # B-1009 amendment. `fe["gate_passed"]` retained as legacy transparency
    # column ONLY (do NOT use for downstream decisions). Defensive `.get` lets
    # `_pool_bootstrap_percentile_p` return None at k<2 propagate to h1_pass=False
    # without KeyError CRASH (mirrors B-1001 H3 defensive pattern).
    if pooled_h1_bootstrap is not None:
        h1_pass = bool(pooled_h1_bootstrap.get("gate_passed_bootstrap", False))
        h1_primary_p = pooled_h1_bootstrap.get("p_one_sided_bootstrap")
        h1_primary_p_method = "bootstrap_percentile_B1000"
    else:
        h1_pass = False
        h1_primary_p = None
        h1_primary_p_method = "INSUFFICIENT_DATA"
    payload["h1_primary_gate_method"] = h1_primary_p_method
    payload["h1_primary_p_one_sided"] = h1_primary_p
    payload["h1_transparency_p_one_sided_normal_approx"] = fe.get("p_one_sided")
    if payload["analysis_status"] in {"COMPLETE", PROTOCOL_NOTE_06_STATUS}:
        payload["h1_verdict"] = "PASS" if h1_pass else "FAIL"
    h2a_falsified = payload["h2a_summary"]["falsified"]
    # F4/F5 (2026-07-14): H3 now returns one explicit status-union schema.  At
    # interim k<6, ``passed`` is None and ``axis_verdict=NOT_EVALUATED``; the
    # defensive bool conversion keeps interim evidence out of final framing.
    h3a_pass = bool(payload.get("h3_axis1_pooled_fe", {}).get("passed", False))
    h3b_pass = bool(payload.get("h3_axis2_pooled_fe", {}).get("passed", False))
    h1_isq_cap = isq_payload.get("heterogeneity_cap_at_r3", False)

    # P1-1 (amendment-02 §4): h10_pass flag for post-R5 reporting route (None if H10
    # verdict not yet computed, pre-Pass-2). P1-2 (prereg §2.5 step-8): B2 cross-family
    # claim-tier downgrade applied AFTER base framing (may downgrade R-tier one step or
    # force R5 on Qwen-anchor failure).
    h10_pass = _load_h10_operational_gate_passed()
    framing = _apply_framing(
        h1_pass,
        h2a_falsified,
        h3a_pass,
        h3b_pass,
        h1_isq_cap,
        h10_pass=h10_pass,
        design_scope=(
            PROTOCOL_NOTE_06_QUALIFIER if protocol_note_06_k5 else "over 6-cell design"
        ),
    )
    if protocol_note_06_k5:
        framing = _apply_protocol_note_06_b1284_downgrade(framing)
    else:
        framing = _apply_b2_cross_family_downgrade(framing, per_cell_data)
    payload["framing_rule"] = framing
    payload["h10_pass_for_post_r5"] = h10_pass

    # Gate status overall
    if len(per_cell_data) < K_REQUIRED_PAPER_GRADE:
        payload["gate_status"] = "PARTIAL_DATA"
        payload["gate_status_reason"] = (
            f"{len(per_cell_data)} of {K_REQUIRED_PAPER_GRADE} required cells with all 6 modes; "
            "pooled result reported but does NOT match the active decision estimand."
        )
    elif h1_pass and not h2a_falsified:
        payload["gate_status"] = "PASS"
        # B-1301: gate status reports bootstrap percentile p (primary) +
        # parenthetical normal-Z transparency p; normal-Z retained for
        # reviewer reproducibility but does NOT drive gate decision.
        p_boot_str = (f"{h1_primary_p:.4f}" if h1_primary_p is not None else "n/a")
        p_normz_str = (f"{fe.get('p_one_sided', 0.0):.4f}" if fe.get("p_one_sided") is not None else "n/a")
        payload["gate_status_reason"] = (
            f"H1 FE bootstrap-percentile p={p_boot_str} (transparency normal-Z p={p_normz_str}) "
            f"on θ_FE={fe['theta_FE_pp']:.3f}pp; "
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
    if protocol_note_06_k5:
        payload["gate_status_reason"] = (
            f"{payload.get('gate_status_reason', '')} Verdict authorized by "
            f"PROTOCOL_NOTE_06 {PROTOCOL_NOTE_06_QUALIFIER}; B-1284 automatic "
            "one-tier downgrade and R2 cap apply."
        ).strip()

    # B-1017 post-FE-pool back-fill: when H1 FE pool succeeded, replace
    # placeholder `<filled-in-post-FE-pool>` strings with actual numbers.
    pfe = payload.get("pooled_h1_fe", {})
    if pfe:
        sub = payload["paper_hero_substitution_table"]["substitutions"]
        sub[0]["canonical_value_post_fire"] = f"{pfe.get('theta_FE_pp', '<missing>')}pp"
        ci_lo = pfe.get("ci95_lo_pp")
        ci_hi = pfe.get("ci95_hi_pp")
        if ci_lo is not None and ci_hi is not None:
            sub[1]["canonical_value_post_fire"] = f"[{ci_lo:+.2f}, {ci_hi:+.2f}]"

    return payload


# ---------------------------------------------------------------------------
# PROTOCOL_NOTE_06 authorization boundary
# ---------------------------------------------------------------------------

def _require_protocol_note_06_in_force(
    note_path: Path = PROTOCOL_NOTE_06_PATH,
) -> None:
    """Fail closed unless NOTE_06 exists and its frontmatter status is in force."""
    if not note_path.is_file():
        raise RuntimeError(f"PROTOCOL_NOTE_06 file missing: {note_path}")
    try:
        text = note_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"cannot read PROTOCOL_NOTE_06: {note_path}: {exc}") from exc
    if not text.startswith("---\n"):
        raise RuntimeError("PROTOCOL_NOTE_06 frontmatter missing")
    end = text.find("\n---", 4)
    if end < 0:
        raise RuntimeError("PROTOCOL_NOTE_06 frontmatter is not terminated")
    status: Optional[str] = None
    for line in text[4:end].splitlines():
        if line.startswith("status:"):
            status = line.split(":", 1)[1].strip()
            break
    normalized = (status or "").upper().replace("-", " ")
    if "IN FORCE" not in normalized or "NOT IN FORCE" in normalized:
        raise RuntimeError(
            "PROTOCOL_NOTE_06 frontmatter status must contain 'IN FORCE'; "
            f"got {status!r}"
        )


def _cell_has_all_six_bound_modes(cell: Dict) -> bool:
    modes = cell.get("modes")
    return isinstance(modes, dict) and set(modes) == set(SIX_MODES)


def build_protocol_note_06_k5_decision(
    cells: List[Dict], *,
    note_path: Path = PROTOCOL_NOTE_06_PATH,
    expected_ids_by_site: Optional[Dict[str, frozenset[int] | set[int]]] = None,
) -> Dict:
    """Validate the NOTE_06 authorization boundary, then compute the fixed k=5 gate.

    No output is written here.  Callers must not invoke an output writer unless
    this function returns successfully, preserving the no-artifact-on-rejection
    contract for every precondition failure.
    """
    _require_protocol_note_06_in_force(note_path)

    b2_reddit = [
        cell for cell in cells
        if cell.get("baseline") == "B2" and cell.get("site") == "reddit"
    ]
    if any(_cell_has_all_six_bound_modes(cell) for cell in b2_reddit):
        raise RuntimeError(
            "k=6 upgrade rule: regenerate the full six-cell verdict instead"
        )

    fixed_keys = set(PROTOCOL_NOTE_06_FIXED_CELLS)
    selected = [
        cell for cell in cells
        if (cell.get("site"), cell.get("baseline")) in fixed_keys
    ]
    selected_keys = {(cell.get("site"), cell.get("baseline")) for cell in selected}
    if selected_keys != fixed_keys or len(selected) != len(fixed_keys):
        missing = sorted(fixed_keys - selected_keys)
        raise RuntimeError(
            "PROTOCOL_NOTE_06 requires exactly the fixed five manifest-bound cells; "
            f"missing={missing}, selected_count={len(selected)}"
        )

    payload = build_full_decision(
        selected,
        expected_ids_by_site=expected_ids_by_site,
        protocol_note_06_k5=True,
    )
    completed_ids = {
        f"{cell.get('baseline')}_{cell.get('site')}"
        for cell in payload.get("per_cell", [])
    }
    required_ids = set(PROTOCOL_NOTE_06_FIXED_CELL_IDS)
    if (
        completed_ids != required_ids
        or payload.get("skipped_cells")
        or payload.get("analysis_status") != PROTOCOL_NOTE_06_STATUS
    ):
        skipped = [
            {
                "cell": f"{cell.get('baseline')}_{cell.get('site')}",
                "reason": cell.get("reason", cell.get("incomplete_reason")),
            }
            for cell in payload.get("skipped_cells", [])
        ]
        raise RuntimeError(
            "PROTOCOL_NOTE_06 requires exactly the fixed five cells to be "
            "complete_exact under the canonical task-universe/provenance checks; "
            f"missing={sorted(required_ids - completed_ids)}, skipped={skipped}"
        )
    if payload.get("h1_verdict") not in {"PASS", "FAIL"}:
        raise RuntimeError("PROTOCOL_NOTE_06 H1 verdict was not evaluated")
    return payload


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_json(payload: Dict, out_json: Path, *,
                manifest_path: Optional[Path] = None,
                input_csv_path: Optional[Path] = None) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    # Provenance lock — manifest + code + git + prereg SHAs (paper §1 OSF audit chain)
    payload["provenance"] = {
        "code_sha256": _self_code_sha(),
        # B-1004 (/stress A2.4a P1-7-A, 2026-05-18): prereg.md SHA closes the
        # provenance gap. A1.21 amendments (per-task ratio + Agresti-Coull anchor)
        # post-code-freeze were invisible — now T₂ reviewer can verify prereg
        # version at run time matches OSF-locked baseline.
        "prereg_sha256": _prereg_sha(),
        "prereg_path": "docs/checkpoints/pre_run/preregistration.md",
        "manifest_sha256": _file_sha256(manifest_path) if manifest_path else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "input_csv_sha256": _file_sha256(input_csv_path) if input_csv_path else None,
        "input_csv_path": str(input_csv_path) if input_csv_path else None,
        "git_commit_sha": _git_commit_sha(),
    }
    atomic_write_text(
        out_json,
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=lambda o: o.tolist() if isinstance(o, np.ndarray) else float(o),
        ) + "\n",
    )


def write_csv(payload: Dict, out_csv: Path) -> None:
    """Per-cell × 6 + pooled FE summary rows.

    B-1054 (/stress A2.3c Mode A F1 + Mode B B5, 2026-05-18): added
    `n_h1_holm_sig`, `n_h3a_holm_sig`, `n_h3b_holm_sig` transparency
    columns per prereg §3 line 408 + §4 line 446 + line 468 promise.
    Per-cell rows have empty (cell-level not aggregate); pooled row has
    counts. Reviewer can now diff prereg promise against CSV without
    opening JSON.
    """
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # B-1301 (/stress A2.3d P0-1-AB*, 2026-05-18): add bootstrap percentile
    # primary p + percentile CI columns; existing `h1_p_one_sided` retained as
    # legacy normal-Z transparency column.
    lines = [
        "row_type,baseline,site,k_cells,n_tasks,"
        "h1_theta_pp,h1_se_pp,h1_ci_lo_pp,h1_ci_hi_pp,h1_p_one_sided,"
        "h1_p_one_sided_bootstrap,h1_ci_lo_pp_bootstrap,h1_ci_hi_pp_bootstrap,"
        "h2a_median_ratio,h2a_rel_diff_pct,h2a_within_band,"
        "h3a_unique_count_pp,h3a_se_pp,h3b_unique_count_pp,h3b_se_pp,"
        "i_squared_pct,framing_rule,gate_status,analysis_status,h1_verdict,"
        "n_h1_holm_sig,n_h3a_holm_sig,n_h3b_holm_sig",
    ]
    gs = payload.get("gate_status", "UNKNOWN")
    analysis_status = payload.get("analysis_status", "INSUFFICIENT")
    h1_verdict = payload.get("h1_verdict", "NOT_EVALUATED")
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
            "", "", "",  # B-1301: bootstrap percentile fields are pool-level only
            _f(h2a.get("median_ratio")), _f(h2a.get("relative_diff_pct")),
            str(h2a.get("per_cell_pass", "")),
            _f(h3a.get("unique_count_pp")), _f(h3a.get("se_pp")),
            _f(h3b.get("unique_count_pp")), _f(h3b.get("se_pp")),
            "", "", gs, analysis_status, h1_verdict,
            "", "", "",  # B-1054 n_holm_sig fields aggregate; cell-level empty
        ]
        lines.append(",".join(cell_line_parts))
    fe = payload.get("pooled_h1_fe")
    if fe is not None:
        h2a_sum = payload.get("h2a_summary", {})
        h3a_fe = payload.get("h3_axis1_pooled_fe") or {}
        h3b_fe = payload.get("h3_axis2_pooled_fe") or {}
        # B-1054 transparency Holm-sig counts (prereg §3+§4 promise emission)
        n_h1 = payload.get("transparency_H1", {}).get("n_individually_holm_sig", "")
        n_h3a = payload.get("transparency_H3_axis1", {}).get("n_individually_holm_sig", "")
        n_h3b = payload.get("transparency_H3_axis2", {}).get("n_individually_holm_sig", "")
        # B-1301: bootstrap percentile pool emit (primary), normal-Z transparency
        pooled_boot = payload.get("pooled_h1_bootstrap") or {}
        p_boot = pooled_boot.get("p_one_sided_bootstrap")
        ci_lo_boot = pooled_boot.get("ci95_lo_pp_bootstrap")
        ci_hi_boot = pooled_boot.get("ci95_hi_pp_bootstrap")
        lines.append(
            f"pooled,,,{fe.get('k_cells', '')},,"
            f"{fe.get('theta_FE_pp', 0):.4f},{fe.get('se_FE_pp', 0):.4f},"
            f"{fe.get('ci95_FE_lo_pp', 0):.4f},{fe.get('ci95_FE_hi_pp', 0):.4f},"
            f"{fe.get('p_one_sided', 0):.6f},"
            f"{_f(p_boot) or ''},{_f(ci_lo_boot) or ''},{_f(ci_hi_boot) or ''},"
            f",,{'true' if not h2a_sum.get('falsified', False) else 'false'},"
            f"{_f(h3a_fe.get('theta_FE_pp'))},{_f(h3a_fe.get('se_FE_pp'))},"
            f"{_f(h3b_fe.get('theta_FE_pp'))},{_f(h3b_fe.get('se_FE_pp'))},"
            f"{isq_str},{framing_rule},{gs},{analysis_status},{h1_verdict},"
            f"{n_h1},{n_h3a},{n_h3b}"
        )
    atomic_write_text(out_csv, "\n".join(lines) + "\n")


def write_md(payload: Dict, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    fe = payload.get("pooled_h1_fe")
    isq = payload.get("h1_heterogeneity", {})
    h2a_sum = payload.get("h2a_summary", {})
    h3a_fe = payload.get("h3_axis1_pooled_fe")
    h3b_fe = payload.get("h3_axis2_pooled_fe")
    framing = payload.get("framing_rule", {})

    is_pn06 = payload.get("analysis_status") == PROTOCOL_NOTE_06_STATUS
    title = (
        "# PROTOCOL_NOTE_06 k=5 decision — H1 + H2(a) + H3 axes"
        if is_pn06
        else "# Phase 1 full prereg decision — H1 + H2(a) + H3 axes + framing rule"
    )
    lines = [
        title,
        "",
        "**Producer**: `aggregate_phase1_full_prereg_decision.py` (A1.21 P0-2/P0-3/P0-4/P0-11, B-515).",
        "Canonical replacement for the H1-only `phase1_prereg_gate.{csv,json,md}` "
        "(B-184) + retired `preregistration_decision_test.py` (DL-contaminated path retired A1.21).",
        "",
        f"**Gate status**: `{payload.get('gate_status', 'UNKNOWN')}`",
        f"**Analysis status**: `{payload.get('analysis_status', 'INSUFFICIENT')}`",
        f"**H1 verdict**: `{payload.get('h1_verdict', 'NOT_EVALUATED')}`",
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
    if is_pn06:
        lines[8:8] = [
            f"**Protocol note**: `{payload.get('protocol_note')}`",
            f"**Verdict qualifier**: **{payload.get('verdict_qualifier')}**",
            f"**Fixed cell set**: `{payload.get('fixed_cell_set')}`",
            "**B-1284 modifier**: automatic one-tier downgrade; **R2 cap**",
            f"**Witness tag**: `{payload.get('witness_tag')}`",
            "",
        ]
    if fe is not None:
        # B-1301 (/stress A2.3d P0-1-AB*, 2026-05-18): MD primary section now
        # reads bootstrap percentile (PRIMARY per prereg L85 B-1009 amend);
        # normal-Z section retained as TRANSPARENCY ONLY.
        pooled_boot = payload.get("pooled_h1_bootstrap")
        primary_sig = (
            "✅ **PASSED**" if (pooled_boot is not None and pooled_boot.get("gate_passed_bootstrap"))
            else ("❌ **NOT YET**" if pooled_boot is not None else "⚠️ **INSUFFICIENT_DATA**")
        )
        lines += [
            f"- **k = {fe['k_cells']}** cells (point-estimate weights)",
            f"- **θ_FE point estimate = +{fe['theta_FE_pp']:.3f}pp** (SE = {fe['se_FE_pp']:.3f}pp, point IV weights)",
        ]
        if pooled_boot is not None:
            lines += [
                "",
                "### Primary gate — bootstrap percentile FE pool (B-1009 amend, B-1301 code)",
                "",
                f"- **θ_FE bootstrap-distribution median = +{pooled_boot['theta_fe_bootstrap_median_pp']:.3f}pp**",
                f"- **95% two-sided percentile CI**: "
                f"[{pooled_boot['ci95_lo_pp_bootstrap']:.3f}, {pooled_boot['ci95_hi_pp_bootstrap']:.3f}]pp",
                f"- **p_one_sided_bootstrap** = P(θ_FE\\* ≤ {pooled_boot['theta_null_pp']}pp) over B={pooled_boot['B_replicates']} "
                f"paired-bootstrap pool replicates = **{pooled_boot['p_one_sided_bootstrap']:.4f}**",
                f"- **Gate (p_bootstrap < α={pooled_boot['alpha']})**: {primary_sig}",
                f"- _Method: {pooled_boot['method_note']}_",
            ]
            if pooled_boot.get("n_below_se_floor", 0) > 0:
                lines.append(
                    f"- ⚠️ **SE floor fired** (B-1003 Agresti-Coull threshold < "
                    f"{pooled_boot['se_floor_threshold_pp']}pp → replaced with "
                    f"{pooled_boot['se_floor_replace_pp']}pp): "
                    f"{pooled_boot['n_below_se_floor']} cell(s)"
                )
        lines += [
            "",
            "### Transparency channel — legacy normal-Z Wald (NOT the gate)",
            "",
            f"- **95% Wald CI**: [{fe['ci95_FE_lo_pp']:.3f}, {fe['ci95_FE_hi_pp']:.3f}]pp",
            f"- **z** = (θ_FE − {fe['delta_pp']}) / SE_FE = **{fe['z_one_sided']:.3f}**",
            f"- **p_one_sided_normal_approx** = 1 − Φ(z) = **{fe['p_one_sided']:.4f}** "
            f"(NOT primary gate per B-1009 amendment; report-only)",
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
        if not isinstance(h3_fe, dict) or h3_fe.get("theta_FE_pp") is None:
            status = h3_fe.get("analysis_status", "INSUFFICIENT") if isinstance(h3_fe, dict) else "INSUFFICIENT"
            reason = h3_fe.get("reason", "FE pool unavailable") if isinstance(h3_fe, dict) else "FE pool unavailable"
            lines += [
                f"## H3 {axis_name} — {status.lower()}",
                "",
                f"- **Axis verdict**: `NOT_EVALUATED`",
                f"- **Reason**: {reason}",
                "",
            ]
            continue
        axis_verdict = h3_fe.get("axis_verdict", "NOT_EVALUATED")
        sig = (
            "✅ **PASSED**" if axis_verdict == "PASS"
            else "❌ **FAILED**" if axis_verdict == "FAIL"
            else "⚠️ **NOT_EVALUATED (INTERIM)**"
        )
        lines += [
            f"## H3 {axis_name} — FE inverse-variance pool over unique-count",
            "",
            f"- **k = {h3_fe['k_cells']}** cells",
            f"- **Analysis status**: `{h3_fe.get('analysis_status', 'UNKNOWN')}`; "
            f"**axis verdict**: `{axis_verdict}`",
            f"- **θ_FE = +{h3_fe['theta_FE_pp']:.3f}pp** (SE = {h3_fe['se_FE_pp']:.3f}pp)",
            f"- **95% CI**: [{h3_fe['ci95_FE_lo_pp']:.3f}, {h3_fe['ci95_FE_hi_pp']:.3f}]pp",
            f"- **z** = θ_FE / SE_FE = **{h3_fe['z_one_sided']:.3f}**",
            f"- **p_one_sided** = 1 − Φ(z) = **{h3_fe['p_one_sided']:.4f}** (transparency only — gate uses CI not p)",
            # B-1055 (/stress A2.3c Mode B P1-9-B*, 2026-05-18): MD writer
            # mislabel fix. Pre-fix wrote "Gate (p < α=...)" but B-949 H3 gate
            # refactor (/stress A2.3a) changed gate semantics to
            # `passed = ci_lo > 0.0` (CI excludes 0, per prereg §2 H3 line
            # 163 "FE CI excludes 0"). MD writer was stale — code computed
            # CI gate, prose said p-threshold gate. Same artifact internal
            # contradiction. Reviewer attack: "code switched gate semantics
            # from p<α to CI-excludes-0, prose still says p<α".
            f"- **Gate (CI lower > 0)**: {sig}",
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
    atomic_write_text(out_md, "\n".join(lines) + "\n")


def write_outputs_atomic(
    payload: Dict,
    out_csv: Path,
    out_json: Path,
    out_md: Path,
    *,
    manifest_path: Optional[Path] = None,
    input_csv_path: Optional[Path] = None,
) -> None:
    """Render before replace, then commit under a lock with rollback backups.

    This is deliberately described as render-before-replace with rollback, not
    as a filesystem transaction.  A same-directory ``flock`` serializes
    writers; old destinations are copied to sibling backups before the first
    replace; a commit error triggers best-effort restoration of all three; and
    the parent directory is fsynced after commit or rollback.
    """
    destinations = [Path(out_csv), Path(out_json), Path(out_md)]
    parent_dirs = {path.parent.resolve() for path in destinations}
    if len(parent_dirs) != 1:
        raise ValueError(
            "phase1 full-decision outputs must share one parent directory for "
            "locked render-before-replace with rollback"
        )
    parent = destinations[0].parent
    parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    staged = [p.with_name(f".{p.name}.{token}.staged") for p in destinations]
    backups = [p.with_name(f".{p.name}.{token}.backup") for p in destinations]
    lock_path = parent / ".phase1_full_prereg_decision.outputs.lock"
    try:
        write_csv(payload, staged[0])
        write_json(
            payload,
            staged[1],
            manifest_path=manifest_path,
            input_csv_path=input_csv_path,
        )
        write_md(payload, staged[2])
        with exclusive_file_lock(lock_path):
            had_old: list[bool] = []
            for dst, backup in zip(destinations, backups):
                exists = dst.exists()
                had_old.append(exists)
                if not exists:
                    continue
                with dst.open("rb") as source, backup.open("xb") as target:
                    shutil.copyfileobj(source, target)
                    target.flush()
                    os.fsync(target.fileno())
            fsync_directory(parent)

            try:
                for src, dst in zip(staged, destinations):
                    os.replace(src, dst)
                fsync_directory(parent)
            except BaseException as commit_error:
                rollback_errors: list[str] = []
                for dst, backup, existed in zip(destinations, backups, had_old):
                    try:
                        if existed and backup.exists():
                            os.replace(backup, dst)
                        elif not existed:
                            dst.unlink(missing_ok=True)
                    except BaseException as rollback_error:
                        rollback_errors.append(f"{dst}: {rollback_error}")
                try:
                    fsync_directory(parent)
                except BaseException as rollback_fsync_error:
                    rollback_errors.append(f"directory fsync: {rollback_fsync_error}")
                if rollback_errors and hasattr(commit_error, "add_note"):
                    commit_error.add_note(
                        "Best-effort output rollback encountered: "
                        + "; ".join(rollback_errors)
                    )
                raise
    finally:
        for path in staged + backups:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def write_protocol_note_06_outputs_atomic(
    payload: Dict,
    out_json: Path = PROTOCOL_NOTE_06_OUT_JSON,
    out_md: Path = PROTOCOL_NOTE_06_OUT_MD,
    *,
    manifest_path: Optional[Path] = None,
) -> None:
    """Atomically replace only the two isolated NOTE_06 artifacts."""
    destinations = [Path(out_json), Path(out_md)]
    parent_dirs = {path.parent.resolve() for path in destinations}
    if len(parent_dirs) != 1:
        raise ValueError("PROTOCOL_NOTE_06 outputs must share one parent directory")
    parent = destinations[0].parent
    parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    staged = [p.with_name(f".{p.name}.{token}.staged") for p in destinations]
    backups = [p.with_name(f".{p.name}.{token}.backup") for p in destinations]
    lock_path = parent / ".phase1_full_prereg_decision_pn06_k5.outputs.lock"
    try:
        write_json(payload, staged[0], manifest_path=manifest_path)
        write_md(payload, staged[1])
        with exclusive_file_lock(lock_path):
            had_old: list[bool] = []
            for dst, backup in zip(destinations, backups):
                exists = dst.exists()
                had_old.append(exists)
                if exists:
                    with dst.open("rb") as source, backup.open("xb") as target:
                        shutil.copyfileobj(source, target)
                        target.flush()
                        os.fsync(target.fileno())
            fsync_directory(parent)
            try:
                for src, dst in zip(staged, destinations):
                    os.replace(src, dst)
                fsync_directory(parent)
            except BaseException as commit_error:
                rollback_errors: list[str] = []
                for dst, backup, existed in zip(destinations, backups, had_old):
                    try:
                        if existed and backup.exists():
                            os.replace(backup, dst)
                        elif not existed:
                            dst.unlink(missing_ok=True)
                    except BaseException as rollback_error:
                        rollback_errors.append(f"{dst}: {rollback_error}")
                try:
                    fsync_directory(parent)
                except BaseException as rollback_fsync_error:
                    rollback_errors.append(f"directory fsync: {rollback_fsync_error}")
                if rollback_errors and hasattr(commit_error, "add_note"):
                    commit_error.add_note(
                        "Best-effort NOTE_06 output rollback encountered: "
                        + "; ".join(rollback_errors)
                    )
                raise
    finally:
        for path in staged + backups:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--run-manifest", default=None,
                    help="Path to run_manifest.yaml (default: results/phantom_paper/run_manifest.yaml via registry). "
                    "A1.21 P0-5 fix: this arg actually propagates to data discovery (was provenance theater).")
    ap.add_argument("--output-csv", default=None)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--output-md", default=None)
    ap.add_argument(
        "--protocol-note-06-k5",
        action="store_true",
        help=(
            "authorize the isolated fixed-five PROTOCOL_NOTE_06 verdict; writes only "
            "phase1_full_prereg_decision_pn06_k5.{json,md}"
        ),
    )
    args = ap.parse_args(argv)

    if args.protocol_note_06_k5 and any(
        value is not None
        for value in (args.output_csv, args.output_json, args.output_md)
    ):
        ap.error(
            "--protocol-note-06-k5 uses isolated fixed output paths and does not "
            "accept --output-csv/--output-json/--output-md"
        )

    # B-534 (/stress A1.5b Phase 2 P0-2-B codex OOB, 2026-05-17): manifest
    # propagation closure. Pre-fix `--run-manifest` only fed `write_json()`
    # provenance hash → consumer thought "data discovery via this manifest"
    # but cells_to_use was loaded from default registry → JSON file hash
    # was for a manifest that NEVER GOVERNED THE DATA. Now pass manifest_path
    # through `get_aggregator_cells(manifest_path=...)` so the provenance
    # SHA in the output is computed against the SAME manifest that drove
    # data discovery. A1.21 P0-5-B / B-524 / B-530 already added
    # `manifest_path` to `get_cells` + `get_aggregator_cells` → this is
    # the canonical-first-number producer plug-in for the param trail.
    manifest_path = (Path(args.run_manifest) if args.run_manifest
                     else REPO / "results/phantom_paper/run_manifest.yaml")
    # A1.21 P1-3 (B-530): lazy fn re-evaluates env var + manifest at call time
    cells_to_use = get_aggregator_cells(manifest_path=manifest_path)

    # B-1015 (/stress A2.4a P1-14-B codex F7, 2026-05-18): structural enforcement
    # of canonical_cells planned scope. Pre-fix lib/canonical_cells.py exposed
    # `assert_cells_match_planned()` (A1.21 B-526) but no producer actually
    # called it → triple-source-of-truth still leaked (preregistration_decision_test
    # hardcoded PHASE_1A_CELLS / aggregate_phantom_lift frozen CELLS / canonical
    # via registry could diverge silently). Now canonical full producer calls
    # the helper; require_complete=False (pre-fire partial state tolerated, but
    # extra/unknown cells raise — closes "wrong scope CSV loaded" attack).
    try:
        from scripts.analysis.lib.canonical_cells import assert_cells_match_planned, cell_id_for
        loaded_ids = [cell_id_for(c["site"], c["baseline"]) for c in cells_to_use]
        assert_cells_match_planned(loaded_ids, require_complete=False)
    except ImportError:
        pass  # canonical_cells.py not present in legacy paths; B-526 require_complete still defends

    if args.protocol_note_06_k5:
        try:
            payload = build_protocol_note_06_k5_decision(
                cells_to_use, note_path=PROTOCOL_NOTE_06_PATH,
            )
        except (RuntimeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        write_protocol_note_06_outputs_atomic(
            payload,
            manifest_path=manifest_path,
        )
        output_paths = (
            PROTOCOL_NOTE_06_OUT_JSON,
            PROTOCOL_NOTE_06_OUT_MD,
        )
    else:
        payload = build_full_decision(cells_to_use)
        out_csv = Path(args.output_csv) if args.output_csv else DEFAULT_OUT_CSV
        out_json = Path(args.output_json) if args.output_json else DEFAULT_OUT_JSON
        out_md = Path(args.output_md) if args.output_md else DEFAULT_OUT_MD
        write_outputs_atomic(
            payload,
            out_csv,
            out_json,
            out_md,
            manifest_path=manifest_path,
        )
        output_paths = (out_csv, out_json, out_md)

    framing = payload.get("framing_rule", {})
    channel = "PN06-K5" if args.protocol_note_06_k5 else "A1.21 B-515"
    print(f"[{channel}] gate_status={payload['gate_status']} "
          f"framing={framing.get('rule', '?')} "
          f"k_cells={len(payload['per_cell'])} skipped={len(payload['skipped_cells'])}")
    for path in output_paths:
        print(f"        → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
