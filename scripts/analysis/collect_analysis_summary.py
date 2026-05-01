#!/usr/bin/env python3
"""[Efficiency supporting] Efficiency dimension — run-level metadata collector.

Outputs:
- results/phantom_paper/run_summary_collect.json

Supports Efficiency 3a token/cost, 3b image embedding-token summaries, and
3c latency by consolidating per-run condition_summary_v2.json artifacts.

See docs/checkpoints/paper_planning.md §3 Efficiency dimension framework.

Collect all key analysis outputs into a single consolidated JSON.

Replaces 10+ manual Read calls in write-analysis/report SKILLs by gathering
adjusted SR, FP counts, McNemar p-values, condition metrics, oracle ceiling,
reason distribution, and signal AUROC into one file.

Usage:
  python scripts/analysis/collect_analysis_summary.py \
      --run-dir <run_dir>

  # Save to file
  python scripts/analysis/collect_analysis_summary.py \
      --run-dir <run_dir> \
      --output collected_summary.json

  # Multiple runs (e.g., for cross-baseline)
  python scripts/analysis/collect_analysis_summary.py \
      --run-dir <b0_run_dir> \
      --run-dir <b1_run_dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from scripts.analysis.lib.run_registry import get_run_dirs_paper_vwa
except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from scripts.analysis.lib.run_registry import get_run_dirs_paper_vwa


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_csv_dicts(path: Path) -> Optional[List[Dict]]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return list(csv.DictReader(f))
    except OSError:
        return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Per-run collection
# ---------------------------------------------------------------------------

def collect_run(run_dir: Path) -> Dict[str, Any]:
    """Collect all analysis data from a single run directory."""
    a = run_dir / "analysis"
    result: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
    }

    # --- 1. analysis_summary.json (adjusted SR, FP counts) ---
    summary = _read_json(a / "analysis_summary.json")
    if summary:
        result["adjusted_success_rates"] = summary.get("adjusted_success_rates", {})
        result["episode_count"] = summary.get("episode_count")
        result["step_count"] = summary.get("step_count")
        result["na_reference_task_count"] = summary.get("na_reference_task_count")
    else:
        result["_missing"] = result.get("_missing", []) + ["analysis_summary.json"]

    # --- 2. condition_metrics.csv (per-condition: raw_sr, avg_steps, cost, latency) ---
    metrics_rows = _read_csv_dicts(a / "results/_overview/tables/condition_metrics.csv")
    if metrics_rows:
        metrics = {}
        for row in metrics_rows:
            cid = row.get("condition_id", "")
            metrics[cid] = {
                k: _safe_float(row.get(k)) for k in [
                    "raw_success_rate", "avg_steps", "avg_total_cost_usd",
                    "avg_total_latency_ms", "p95_step_latency_ms",
                    "avg_total_tokens", "no_op_rate",
                ]
            }
        result["condition_metrics"] = metrics
    else:
        result["_missing"] = result.get("_missing", []) + ["condition_metrics.csv"]

    # --- 3. statistical_tests.json (McNemar, Bootstrap CI) ---
    stats = _read_json(a / "results/_overview/reports/statistical_tests.json")
    if stats:
        result["statistical_tests"] = stats
    else:
        result["_missing"] = result.get("_missing", []) + ["statistical_tests.json"]

    # --- 4. cross_representation_summary.json (oracle, headroom, set analysis) ---
    cross_rep = _read_json(a / "results/cross_representation/cross_representation_summary.json")
    if cross_rep:
        # Flatten per-site data
        per_site = cross_rep.get("per_site", {})
        cross = {}
        for site, sdata in per_site.items():
            sdata = sdata or {}  # tolerate None entries (e.g. site present but no analysis)
            cross[site] = {
                "oracle_ceiling": sdata.get("oracle_ceiling"),
                "routing_headroom": sdata.get("routing_headroom"),
                "per_mode_sr": sdata.get("per_mode_sr"),
                "per_mode_sr_adjusted": sdata.get("per_mode_sr_adjusted"),
                "oracle_ceiling_adjusted": sdata.get("oracle_ceiling_adjusted"),
                "routing_headroom_adjusted": sdata.get("routing_headroom_adjusted"),
                "na_fp_count": sdata.get("na_fp_count"),
                "eval_fp_count": sdata.get("eval_fp_count"),
            }
        result["cross_representation"] = cross
    else:
        result["_missing"] = result.get("_missing", []) + ["cross_representation_summary.json"]

    # --- 5. reason_diagnostics summary ---
    reason_summary = _read_json(a / "reason_diagnostics/reason_diagnostics_summary.json")
    if reason_summary:
        result["reason_diagnostics"] = reason_summary

    # condition_reason_summary.csv (failure reason distribution)
    reason_rows = _read_csv_dicts(a / "reason_diagnostics/condition_reason_summary.csv")
    if reason_rows:
        reason_dist: Dict[str, Dict[str, int]] = {}
        for row in reason_rows:
            cid = row.get("condition_id", "")
            if cid not in reason_dist:
                reason_dist[cid] = {}
            reason = row.get("reason", "unknown")
            count = int(row.get("count", 0)) if row.get("count") else 0
            reason_dist[cid][reason] = count
        result["failure_reason_distribution"] = reason_dist

    # condition_overview.csv
    overview_rows = _read_csv_dicts(a / "reason_diagnostics/condition_overview.csv")
    if overview_rows:
        overview = {}
        for row in overview_rows:
            cid = row.get("condition_id", "")
            overview[cid] = {
                k: _safe_float(row.get(k)) for k in [
                    "total_episodes", "fp_count", "early_finish",
                    "avg_steps", "avg_no_op_rate",
                ]
            }
        result["condition_overview"] = overview

    # --- 6. signals / AUROC ---
    signals = _read_json(a / "signals/combined/confidence_summary.json")
    if signals:
        # Extract key AUROC metrics
        auroc = {}
        for key in ["auroc_all", "auroc_per_mode", "calibration_metrics",
                     "ece", "mce", "brier_score", "auroc"]:
            if key in signals:
                auroc[key] = signals[key]
        if auroc:
            result["signal_auroc"] = auroc

    # AUROC table
    auroc_rows = _read_csv_dicts(a / "signals/combined/tables/auroc_all_metrics.csv")
    if auroc_rows:
        result["auroc_all_metrics"] = auroc_rows

    cross_mode_auroc = _read_csv_dicts(a / "signals/combined/tables/cross_mode_auroc.csv")
    if cross_mode_auroc:
        result["cross_mode_auroc"] = cross_mode_auroc

    # --- 7. FP lists (compact) ---
    na_tasks = _read_csv_dicts(a / "benchmark_noise/na_reference_tasks.csv")
    if na_tasks:
        result["na_task_count"] = len(na_tasks)
        result["na_task_ids"] = sorted(set(
            int(r.get("task_id", -1)) for r in na_tasks if r.get("task_id")
        ))

    # --- 8. Exclusive sets (adjusted) ---
    excl_adj = _read_csv_dicts(
        a / "results/cross_representation/tables/A3_exclusive_sets_summary_adjusted.csv"
    )
    if excl_adj:
        result["exclusive_sets_adjusted"] = excl_adj

    # --- 9. Oracle decomposition ---
    oracle = _read_json(a / "results/cross_representation/R3_oracle_decomposition.json")
    if oracle:
        result["oracle_decomposition"] = oracle

    # --- 9b. Task type × mode SR (A5) ---
    a5_rows = _read_csv_dicts(a / "results/cross_representation/tables/A5_task_type_success_rate.csv")
    if a5_rows:
        result["task_type_mode_sr"] = a5_rows

    # --- 9c. Reason stability (B2) ---
    b2_rows = _read_csv_dicts(a / "results/cross_representation/tables/B2_reason_stability.csv")
    if b2_rows:
        result["reason_stability"] = b2_rows

    # --- 9d. Fail reason cost stats (A4b) ---
    a4b_rows = _read_csv_dicts(a / "results/cross_representation/tables/A4b_fail_reason_cost_stats.csv")
    if a4b_rows:
        result["fail_reason_cost_stats"] = a4b_rows

    # --- 9e. State change reason distribution ---
    sc_rows = _read_csv_dicts(a / "results/_overview/tables/state_change_reason_distribution.csv")
    if sc_rows:
        result["state_change_distribution"] = sc_rows

    # --- 9f. Action execution summary ---
    ax_rows = _read_csv_dicts(a / "reason_diagnostics/action_execution_summary.csv")
    if ax_rows:
        result["action_execution_stats"] = ax_rows

    # --- 9g. Per-mode calibration (ECE columns) ---
    pm_rows = _read_csv_dicts(a / "signals/combined/tables/per_mode_summary.csv")
    if pm_rows and any("ECE" in r for r in pm_rows):
        result["per_mode_calibration"] = [
            {k: v for k, v in r.items() if k in (
                "observation_mode", "n", "success_rate",
                "ECE", "MCE", "Brier",
                "verbalized_ECE", "verbalized_MCE", "verbalized_Brier",
            )}
            for r in pm_rows
        ]

    # --- 9h. Temporal SR ---
    ts_rows = _read_csv_dicts(a / "reason_diagnostics/temporal_sr.csv")
    if ts_rows:
        result["temporal_sr"] = ts_rows

    # --- 9i. State change by outcome ---
    scbo_rows = _read_csv_dicts(a / "reason_diagnostics/state_change_by_outcome.csv")
    if scbo_rows:
        result["state_change_by_outcome"] = scbo_rows

    # --- 10. Reason diagnostics: cost + intent features + plots ---
    episode_rows = _read_csv_dicts(a / "reason_diagnostics/episode_reason_rows.csv")
    if episode_rows:
        # Per-condition cost aggregation
        cost_by_cond: Dict[str, Dict[str, list]] = {}
        for row in episode_rows:
            cid = row.get("condition_id", "")
            if cid not in cost_by_cond:
                cost_by_cond[cid] = {"total": [], "effective": [], "no_op": [], "loop": []}
            for key, csv_col in [("total", "total_cost_usd"), ("effective", "effective_cost_usd"),
                                  ("no_op", "no_op_cost_usd"), ("loop", "loop_cost_usd")]:
                v = _safe_float(row.get(csv_col))
                if v is not None:
                    cost_by_cond[cid][key].append(v)
        reason_cost: Dict[str, Dict[str, Optional[float]]] = {}
        for cid, vals in cost_by_cond.items():
            reason_cost[cid] = {
                k: (sum(v) / len(v) if v else None) for k, v in vals.items()
            }
        result["reason_diagnostics_cost"] = reason_cost

        # Intent feature SR
        intent_keys = [k for k in episode_rows[0].keys() if k.startswith("intent_")]
        if intent_keys:
            intent_sr: Dict[str, Dict[str, Optional[float]]] = {}
            for ik in intent_keys:
                positives = [r for r in episode_rows if str(r.get(ik, "")).lower() in ("true", "1")]
                if len(positives) >= 3:
                    sr = sum(1 for r in positives if str(r.get("adjusted_success", "")).lower() in ("true", "1")) / len(positives)
                    intent_sr[ik] = {"count": len(positives), "adjusted_sr": round(sr, 4)}
                else:
                    intent_sr[ik] = {"count": len(positives), "adjusted_sr": None}
            result["intent_feature_sr"] = intent_sr

    # List diagnostic plots
    plots_dir = a / "reason_diagnostics/plots"
    if plots_dir.is_dir():
        result["reason_diagnostics_plots"] = sorted(p.name for p in plots_dir.iterdir() if p.suffix == ".png")

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect all analysis outputs into a consolidated JSON",
    )
    parser.add_argument("--run-dir", type=Path, action="append", default=None,
                        help="Run directory (can specify multiple; default: paper VWA runs from run_manifest.yaml)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: stdout)")

    args = parser.parse_args()

    results = []
    run_dirs = args.run_dir if args.run_dir else get_run_dirs_paper_vwa()
    for rd in run_dirs:
        if not rd.is_dir():
            print(f"Warning: {rd} is not a directory, skipping", file=sys.stderr)
            continue
        data = collect_run(rd)
        results.append(data)
        missing = data.get("_missing", [])
        if missing:
            print(f"  {rd.name}: missing {missing}", file=sys.stderr)
        else:
            print(f"  {rd.name}: all files found", file=sys.stderr)

    output = {
        "collect_time": datetime.now(timezone.utc).isoformat(),
        "runs": results if len(results) > 1 else results[0] if results else {},
    }

    output_json = json.dumps(output, ensure_ascii=False, indent=2)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json, encoding="utf-8")
        print(f"Output written to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
