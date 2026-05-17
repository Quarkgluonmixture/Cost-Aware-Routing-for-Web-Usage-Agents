#!/usr/bin/env python3
"""B-184: canonical Phase 1 paper §1 PRIMARY gate producer.

Implements the prereg H1 spec (preregistration.md:68-86 lock):

    Primary gate = "FE inverse-variance pooled P-SoM drop-one effect θ_FE
                   significantly exceeds the +1.0pp substantive-effect threshold
                   via a one-sided superiority test:
                   reject H0: θ_FE ≤ +1.0pp at α=0.05
                   (PRIMARY family m=1, no within-family correction)"

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

One-sided superiority z-statistic:
    z = (θ_FE - 1.0) / SE_FE
    p_one_sided = 1 - Φ(z)
    gate_passed = (p_one_sided < 0.05)

This producer is **complementary** to `aggregate_phantom_lift.py`, which still
runs the legacy 3→5 lift estimand (now demoted to exploratory; the prereg
PRIMARY gate is THIS file).

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
from scripts.analysis.aggregate_phantom_lift import (  # noqa: E402
    CELLS,
    MIN_EP_FOR_CELL,
    load,
)

# B-176 lock: bootstrap seed=42, B=1000 per prereg "1000-resample".
# (Note: prereg explicitly says 1000, NOT the 10_000 used in `analyze_run`
# bootstrap CIs. Different parts of the pipeline have different B; the per-cell
# drop-one SE here is the prereg-locked B=1000 path.)
PREREG_B = 1000
PREREG_SEED = 42
DELTA_PP = 1.0    # prereg superiority threshold (preregistration.md:341 lock)
ALPHA = 0.05      # prereg α
SIX_MODES = ("DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM")

DEFAULT_OUT_CSV = REPO / "results/phantom_paper/phase1_prereg_gate.csv"
DEFAULT_OUT_JSON = REPO / "results/phantom_paper/phase1_prereg_gate.json"
DEFAULT_OUT_MD = REPO / "results/phantom_paper/phase1_prereg_gate.md"


def _norm_cdf(x: float) -> float:
    """Standard-normal CDF via erf; scipy-free."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _cell_drop_one_theta_se(
    cell: Dict, *, B: int = PREREG_B, seed: int = PREREG_SEED,
) -> Optional[Dict]:
    """Compute per-cell drop-one effect + bootstrap SE per prereg spec.

    Returns None if the cell does NOT contain all 6 modes (oracle_6 undefined)
    OR any present mode is below MIN_EP_FOR_CELL.

    Otherwise returns:
        {baseline, site, n_tasks, theta_pp, se_pp, ci95_lo_pp, ci95_hi_pp,
         oracle_6_pp, oracle_5_no_psom_pp, n_psom_only}
    """
    # Load per-mode (success, observed) sets via the shared `load` primitive.
    succ: Dict[str, set] = {}
    obs: Dict[str, set] = {}
    for mode, ep_dir in cell["modes"].items():
        s, o = load(ep_dir)
        if len(o) < MIN_EP_FOR_CELL:
            continue  # skip below-threshold modes (won't make universe anyway)
        succ[mode] = s
        obs[mode] = o

    # Require ALL 6 modes present (prereg: "cells containing all 6 modes").
    if any(m not in succ for m in SIX_MODES):
        return None

    # Universe = intersection of observed task IDs across all 6 modes
    # (only tasks where every mode reports a result are eligible).
    universe = set.intersection(*[obs[m] for m in SIX_MODES])
    n = len(universe)
    if n < MIN_EP_FOR_CELL:
        return None

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
        "baseline": cell["baseline"],
        "site": cell["site"],
        "n_tasks": n,
        "theta_pp": theta_pp,
        "se_pp": se_pp,
        "ci95_lo_pp": ci_lo,
        "ci95_hi_pp": ci_hi,
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
    n_zero_se = int((ses <= 0).sum())
    if n_zero_se > 0:
        ses = np.where(ses <= 0, 1.0, ses)  # 1.0 pp floor — see prereg.md §2 H1
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
        # v6 fix (P0-9): n_zero_se transparency — cells with SE=0 got floored to 1pp to
        # prevent degenerate-cell hijack of FE pool weight. paper §6 must disclose.
        "n_zero_se_floored_cells": n_zero_se,
    }


def build_gate(cells: List[Dict]) -> Dict:
    """End-to-end gate computation across the provided cell list.

    Returns a structured payload with per-cell rows + pooled FE result +
    gate_status ∈ {"PASS", "FAIL", "INSUFFICIENT_DATA", "PARTIAL_DATA"}.
    """
    per_cell: List[Dict] = []
    skipped: List[Dict] = []
    for cell in cells:
        result = _cell_drop_one_theta_se(cell)
        if result is None:
            skipped.append({
                "baseline": cell["baseline"],
                "site": cell["site"],
                "reason": "missing one or more of the 6 modes OR below MIN_EP_FOR_CELL",
            })
        else:
            per_cell.append(result)

    payload: Dict = {
        "prereg_section": "preregistration.md §1 H1 PRIMARY gate (line 68-86 lock)",
        "estimand": "FE inverse-variance pooled P-SoM drop-one over 6 planned cells",
        "delta_pp": DELTA_PP,
        "alpha": ALPHA,
        "bootstrap_B": PREREG_B,
        "bootstrap_seed": PREREG_SEED,
        "per_cell": per_cell,
        "skipped_cells": skipped,
    }

    fe = _fe_pool(per_cell)
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

    return payload


def write_csv(payload: Dict, out_csv: Path) -> None:
    """Per-cell × 6 + pooled FE row → flat CSV for paper §1 prose to cite."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "row_type,baseline,site,k_cells,n_tasks,theta_pp,se_pp,ci95_lo_pp,"
        "ci95_hi_pp,oracle_6_pp,oracle_5_no_psom_pp,n_psom_only,"
        "z_one_sided,p_one_sided,gate_passed,gate_status",
    ]
    gs = payload.get("gate_status", "UNKNOWN")
    for r in payload["per_cell"]:
        lines.append(
            f"cell,{r['baseline']},{r['site']},,,{r['n_tasks']},"
            f"{r['theta_pp']:.4f},{r['se_pp']:.4f},"
            f"{r['ci95_lo_pp']:.4f},{r['ci95_hi_pp']:.4f},"
            f"{r['oracle_6_pp']:.4f},{r['oracle_5_no_psom_pp']:.4f},"
            f"{r['n_psom_only']},,,,{gs}"
        )
    fe = payload.get("pooled_fe")
    if fe is not None:
        lines.append(
            f"pooled_FE,,,{fe['k_cells']},,"
            f"{fe['theta_FE_pp']:.4f},{fe['se_FE_pp']:.4f},"
            f"{fe['ci95_FE_lo_pp']:.4f},{fe['ci95_FE_hi_pp']:.4f},"
            f",,,{fe['z_one_sided']:.4f},{fe['p_one_sided']:.6f},"
            f"{fe['gate_passed']},{gs}"
        )
    out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(payload: Dict, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8")


def write_md(payload: Dict, out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Phase 1 prereg gate — H1 PRIMARY (P-SoM drop-one over 6 cells)",
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
            "## Pooled FE (paper §1 hero claim source)",
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
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-csv", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--output-json", default=str(DEFAULT_OUT_JSON))
    ap.add_argument("--output-md", default=str(DEFAULT_OUT_MD))
    args = ap.parse_args()

    payload = build_gate(CELLS)

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
