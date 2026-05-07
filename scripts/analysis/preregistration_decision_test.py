"""Preregistration decision test — H1 / H3 / TOST canonical implementation.

Single source of truth for the paper §5 / Table 5 decision rules.
Once advisor's email reply confirms thresholds (K_h1 / K_h3 / TOST δ),
this script takes 16-cell SR data + emits deterministic pass/fail JSON.

Tied to:
- preregistration.md (commitment text)
- osf_lock_manifest.md (lock SHA chain)
- run_manifest.yaml (cell scope)
- 笔记 §114 (provenance hardening)

Hypotheses:
- H1: Phantom-SoM ≥ best (DOM, SoM, Vision) in ≥ K_h1 cells
- H3: Drop-one oracle lift from Phantom-SoM ≥ <delta_pp> in ≥ K_h3 cells
- TOST: |cost(P-SoM) - cost(DOM)| < δ (equivalence test)

Usage:
    # With actual data:
    python3 scripts/analysis/preregistration_decision_test.py \\
        --cells-csv results/phantom_paper/cells_aggregated.csv \\
        --thresholds K_h1=12 K_h3=11 TOST_delta=1.0 \\
        --out results/phantom_paper/preregistration_test_results.json

    # Smoke test on synthetic data:
    python3 scripts/analysis/preregistration_decision_test.py --synthetic --seed 0

Inputs (CSV schema):
    cell_id,baseline,site,phantom_axis,sr_dom,sr_som,sr_vision,sr_phantom_som,
        sr_phantom_text,sr_phantom_prompt,oracle_3mode,oracle_drop_one_psom,
        cost_dom,cost_psom

Outputs (JSON):
    {
      "captured_at": "...",
      "n_cells": 16,
      "thresholds": {"K_h1": 12, "K_h3": 11, "TOST_delta_pp": 1.0},
      "H1": {"per_cell_winners": [...], "n_pass": 12, "decision": "PASS|FAIL", "p_binomial": ...},
      "H3": {"per_cell_lift_pp": [...], "n_pass": 10, "decision": "PASS|FAIL"},
      "TOST": {"observed_delta_pp": 0.7, "ci_95": [-0.4, 1.8], "decision": "PASS|FAIL"},
      "overall_decision": "...",
      "input_data_sha256": "..."
    }
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger("preregistration-test")


# ---------------------------------------------------------------------------
# H1: Phantom-SoM ≥ best of (DOM, SoM, Vision) per cell
# ---------------------------------------------------------------------------

def evaluate_h1(cells: list[dict], k_threshold: int) -> dict:
    """For each cell, count if SR_phantom_som >= max(SR_dom, SR_som, SR_vision)."""
    per_cell = []
    for c in cells:
        psom = float(c["sr_phantom_som"])
        baselines = [float(c["sr_dom"]), float(c["sr_som"]), float(c["sr_vision"])]
        best_baseline = max(baselines)
        winner = psom >= best_baseline
        per_cell.append({
            "cell_id": c["cell_id"],
            "sr_phantom_som": psom,
            "best_baseline_sr": best_baseline,
            "delta_pp": (psom - best_baseline) * 100.0,  # in percentage points
            "phantom_som_wins": winner,
        })
    n_pass = sum(1 for r in per_cell if r["phantom_som_wins"])
    n_total = len(per_cell)

    # Two-sided binomial p-value under H0: p = 0.5 (random tie-breaking)
    p_binomial = _binomial_test_two_sided(n_pass, n_total, 0.5)

    return {
        "n_cells": n_total,
        "n_pass": n_pass,
        "k_threshold": k_threshold,
        "decision": "PASS" if n_pass >= k_threshold else "FAIL",
        "p_binomial_h0_random": p_binomial,
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# H3: Drop-one oracle lift from Phantom-SoM ≥ <delta_pp> in K_h3 cells
# ---------------------------------------------------------------------------

def evaluate_h3(cells: list[dict], k_threshold: int, lift_pp_min: float = 0.5) -> dict:
    """Drop-one oracle lift: oracle_with_psom - oracle_3mode (DOM/SoM/Vision)
    expressed in percentage points; pass if lift >= lift_pp_min for K_h3 cells."""
    per_cell = []
    for c in cells:
        oracle_3 = float(c["oracle_3mode"])
        oracle_drop_one = float(c["oracle_drop_one_psom"])
        # Drop-one: oracle 4-mode - oracle 4-mode-without-PSoM
        # Convention here: oracle_drop_one_psom = oracle 4-mode lift attributable to PSoM
        # i.e., the marginal SR contribution if you remove PSoM from oracle bag
        lift_pp = oracle_drop_one * 100.0
        passes = lift_pp >= lift_pp_min
        per_cell.append({
            "cell_id": c["cell_id"],
            "oracle_3mode_sr": oracle_3,
            "drop_one_lift_pp": lift_pp,
            "passes_min_lift": passes,
        })
    n_pass = sum(1 for r in per_cell if r["passes_min_lift"])

    return {
        "n_cells": len(cells),
        "n_pass": n_pass,
        "k_threshold": k_threshold,
        "min_lift_pp": lift_pp_min,
        "decision": "PASS" if n_pass >= k_threshold else "FAIL",
        "per_cell": per_cell,
    }


# ---------------------------------------------------------------------------
# TOST: equivalence test for cost(P-SoM) ≈ cost(DOM)
# ---------------------------------------------------------------------------

def evaluate_tost(cells: list[dict], delta_pp: float) -> dict:
    """TOST (two one-sided tests) for cost equivalence within ±delta_pp.
    Compute mean cost difference + 95% CI, check if CI ⊂ [-delta_pp, +delta_pp].
    Cost expressed in percentage units (e.g., relative tokens or relative latency)."""
    diffs = []
    for c in cells:
        cost_dom = float(c["cost_dom"])
        cost_psom = float(c["cost_psom"])
        if cost_dom == 0:
            continue  # avoid div-by-zero
        rel_diff_pp = (cost_psom - cost_dom) / cost_dom * 100.0
        diffs.append(rel_diff_pp)

    if not diffs:
        return {"decision": "FAIL", "reason": "no valid diffs"}

    n = len(diffs)
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / max(1, n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n)
    # 95% CI using t-distribution approx (df = n-1, t_0.975 ≈ 1.96 for n>=30, ~2.13 for n=16)
    t_crit = 2.131 if n <= 16 else 1.96
    ci_lo = mean_diff - t_crit * sem
    ci_hi = mean_diff + t_crit * sem

    inside_band = (ci_lo > -delta_pp) and (ci_hi < delta_pp)

    return {
        "n_cells": n,
        "delta_pp": delta_pp,
        "mean_diff_pp": mean_diff,
        "ci_95": [ci_lo, ci_hi],
        "decision": "PASS" if inside_band else "FAIL",
        "interpretation": (
            f"95% CI [{ci_lo:.2f}, {ci_hi:.2f}] {'⊂' if inside_band else '⊄'} "
            f"[-{delta_pp:.1f}, +{delta_pp:.1f}]"
        ),
        "raw_diffs_pp": diffs,
    }


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _binomial_test_two_sided(k: int, n: int, p: float) -> float:
    """Two-sided binomial p-value for k successes in n trials under H0: P(success)=p.
    Sums tail probabilities ≤ observed."""
    from math import comb
    obs_p = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for kk in range(0, n + 1):
        pp = comb(n, kk) * (p ** kk) * ((1 - p) ** (n - kk))
        if pp <= obs_p + 1e-15:
            total += pp
    return min(1.0, total)


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Synthetic data generator (smoke test, no actual data needed)
# ---------------------------------------------------------------------------

def generate_synthetic_cells(seed: int = 0, n_cells: int = 16, scenario: str = "h1_pass") -> list[dict]:
    """Generate 16-cell synthetic data for smoke test.
    Scenarios:
      - h1_pass: PSoM wins in 13/16 cells (above K_h1=12)
      - h1_fail: PSoM wins in 8/16 cells
      - tost_pass: cost diff ~0.5pp ± 0.3 (inside band 1.0pp)
      - tost_fail: cost diff ~3pp (outside band)
    """
    import random
    rng = random.Random(seed)
    sites = ["classifieds", "reddit", "shopping"]
    baselines = ["B0", "B1"]
    phantom_axes = ["P-SoM", "P-text", "P-prompt"]

    cells = []
    cell_id = 0
    for b in baselines:
        for s in sites:
            for p in phantom_axes:
                if cell_id >= n_cells:
                    break
                # Base SR levels
                base_sr = 0.3 + rng.uniform(-0.05, 0.10)
                sr_dom = base_sr + rng.uniform(-0.02, 0.02)
                sr_som = base_sr + rng.uniform(-0.02, 0.05)
                sr_vision = base_sr + rng.uniform(-0.05, 0.02)

                if scenario == "h1_pass":
                    # PSoM wins in most cells
                    sr_psom = max(sr_dom, sr_som, sr_vision) + rng.uniform(-0.005, 0.04)
                else:  # h1_fail
                    sr_psom = base_sr + rng.uniform(-0.04, 0.01)

                # Cost
                cost_dom = 1.0  # normalized
                if scenario == "tost_pass":
                    cost_psom = 1.0 + rng.uniform(-0.005, 0.005)  # ±0.5%
                elif scenario == "tost_fail":
                    cost_psom = 1.03 + rng.uniform(-0.005, 0.005)  # +3%
                else:
                    cost_psom = 1.0 + rng.uniform(-0.008, 0.008)  # within band

                cells.append({
                    "cell_id": f"{b}_{s}_{p}_{cell_id}",
                    "baseline": b, "site": s, "phantom_axis": p,
                    "sr_dom": sr_dom, "sr_som": sr_som, "sr_vision": sr_vision,
                    "sr_phantom_som": sr_psom,
                    "sr_phantom_text": sr_psom * 0.97,
                    "sr_phantom_prompt": sr_psom * 0.95,
                    "oracle_3mode": max(sr_dom, sr_som, sr_vision),
                    "oracle_drop_one_psom": rng.uniform(0.005, 0.025),  # 0.5-2.5pp lift
                    "cost_dom": cost_dom, "cost_psom": cost_psom,
                })
                cell_id += 1
    return cells[:n_cells]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells-csv", help="Path to cells_aggregated.csv (skip if --synthetic)")
    p.add_argument("--synthetic", action="store_true",
                   help="Run smoke test on synthetic data (no real data needed)")
    p.add_argument("--scenario", default="h1_pass",
                   choices=["h1_pass", "h1_fail", "tost_pass", "tost_fail"],
                   help="Synthetic data scenario (only with --synthetic)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--K_h1", type=int, default=12,
                   help="H1 threshold (cells where PSoM wins). Pre-registration default 12.")
    p.add_argument("--K_h3", type=int, default=11,
                   help="H3 threshold (cells with drop-one lift ≥ min). Pre-registration default 11.")
    p.add_argument("--TOST-delta", type=float, default=1.0,
                   help="TOST equivalence margin in pp. Pre-registration default 1.0.")
    p.add_argument("--H3-min-lift-pp", type=float, default=0.5,
                   help="H3 per-cell minimum drop-one lift in pp.")
    p.add_argument("--out", default="-", help="Output JSON path (- = stdout)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Load cells
    if args.synthetic:
        cells = generate_synthetic_cells(seed=args.seed, scenario=args.scenario)
        input_sha = "synthetic"
        logger.info(f"Synthetic mode: {len(cells)} cells, scenario={args.scenario}")
    else:
        if not args.cells_csv:
            logger.error("Must provide --cells-csv or --synthetic")
            sys.exit(2)
        csv_path = Path(args.cells_csv)
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            cells = list(reader)
        input_sha = _file_sha256(csv_path)
        logger.info(f"Loaded {len(cells)} cells from {csv_path} (sha256={input_sha[:12]}...)")

    # Run hypothesis tests
    h1 = evaluate_h1(cells, k_threshold=args.K_h1)
    h3 = evaluate_h3(cells, k_threshold=args.K_h3, lift_pp_min=args.H3_min_lift_pp)
    tost = evaluate_tost(cells, delta_pp=args.TOST_delta)

    overall = "PASS" if (h1["decision"] == "PASS" and h3["decision"] == "PASS"
                          and tost["decision"] == "PASS") else "PARTIAL_OR_FAIL"

    result = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "n_cells": len(cells),
        "thresholds": {
            "K_h1": args.K_h1,
            "K_h3": args.K_h3,
            "TOST_delta_pp": args.TOST_delta,
            "H3_min_lift_pp": args.H3_min_lift_pp,
        },
        "input_data_sha256": input_sha,
        "H1": h1,
        "H3": h3,
        "TOST": tost,
        "overall_decision": overall,
        "summary_paper_table5": {
            "H1_text": f"H1: Phantom-SoM ≥ best baseline in {h1['n_pass']}/{h1['n_cells']} cells "
                      f"(threshold {h1['k_threshold']}, decision {h1['decision']})",
            "H3_text": f"H3: drop-one oracle lift ≥ {h3['min_lift_pp']:.1f}pp in {h3['n_pass']}/{h3['n_cells']} cells "
                      f"(threshold {h3['k_threshold']}, decision {h3['decision']})",
            "TOST_text": f"TOST: cost equivalence δ=±{tost.get('delta_pp', 'N/A')}pp, "
                        f"observed {tost.get('mean_diff_pp', 'N/A'):.2f}pp, "
                        f"CI95 {tost.get('ci_95', 'N/A')}, decision {tost['decision']}"
                        if tost.get("ci_95") else f"TOST: {tost.get('reason', 'N/A')}",
        },
    }

    payload = json.dumps(result, indent=2)
    if args.out == "-":
        print(payload)
    else:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
        logger.info(f"Result → {out_path}")
        logger.info(f"Overall decision: {overall}")
        logger.info(f"  H1: {h1['decision']} ({h1['n_pass']}/{h1['n_cells']}, p_binomial={h1['p_binomial_h0_random']:.4f})")
        logger.info(f"  H3: {h3['decision']} ({h3['n_pass']}/{h3['n_cells']})")
        logger.info(f"  TOST: {tost['decision']}")


if __name__ == "__main__":
    main()
