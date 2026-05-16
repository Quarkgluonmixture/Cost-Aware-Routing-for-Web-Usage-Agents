#!/usr/bin/env python3
"""F1 about:blank systematic study — Phase 1 frequency measurement.

Cross site × mode × baseline, count steps where the recovery branch
(``p79/experiment/runner/main.py:1052-1090``) fires: page_change_reasons
contains ``about_blank_recovery`` OR ``state_digest.url_after`` starts
with ``about:blank``.

Output: stdout table + JSON dump for cross-cell aggregation.

Mini-investigation tracker: ``docs/checkpoints/_status/issues/
issue_about_blank_systematic_2026-05-16.md``.

Usage:
    python3 scripts/analysis/about_blank_frequency.py \\
        --runs results/visualwebarena/phase1/B*_3mode_*_20260413 \\
        --output docs/analysis/about_blank_frequency.json

    # Or scan all phase1 runs:
    python3 scripts/analysis/about_blank_frequency.py \\
        --runs 'results/visualwebarena/phase1/*' \\
        --output -

Phase 2 (action-type classification): extend the step record with
``action_type_preceding_about_blank`` and aggregate. Not implemented yet
— see issue file ``Phase 2`` block.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from p79.experiment.io_utils import read_jsonl_dedup


def _is_about_blank_step(step: Dict[str, Any]) -> bool:
    """True iff this step triggered the about:blank recovery branch."""
    reasons = step.get("page_change_reasons", []) or []
    if isinstance(reasons, list) and "about_blank_recovery" in reasons:
        return True
    state_digest = step.get("state_digest", {}) or {}
    url_after = str(state_digest.get("url_after", "") or "")
    if url_after.startswith("about:blank"):
        return True
    return False


def _infer_mode_baseline(condition_id: str) -> Dict[str, str]:
    """Best-effort split of condition_id into (mode, baseline) labels.

    Examples:
        phase1_dom_router_0 → mode=dom, baseline=?
        phase1_phantom_som_router_0 → mode=phantom_som, baseline=?

    Baseline (B0/B1/B2) is harder to recover from condition_id alone;
    callers should pair with condition_meta.json or the run-dir name
    prefix (e.g. B0_3mode_*).
    """
    cid = condition_id.lower()
    if cid.startswith("phase1_"):
        rest = cid[len("phase1_"):]
        # rest looks like "som_router_0" or "phantom_som_router_0"
        if "_router_" in rest:
            mode = rest.split("_router_")[0]
        else:
            mode = rest
    else:
        mode = cid
    return {"mode": mode}


def _infer_baseline_from_runname(run_name: str) -> str:
    """B0_3mode_classifieds_20260413 → B0; falls back to 'unknown' if
    run name doesn't follow the project convention."""
    upper = run_name.upper()
    for tag in ("B0", "B1", "B2"):
        if upper.startswith(tag + "_") or "_" + tag + "_" in upper:
            return tag
    return "unknown"


def scan_run_dir(run_dir: Path) -> List[Dict[str, Any]]:
    """Walk a Phase 1 run dir and return per-condition stats:

    [{run, condition_id, mode, baseline, site, total_steps, about_blank_steps,
      about_blank_rate}, ...]
    """
    if not run_dir.is_dir():
        return []

    baseline = _infer_baseline_from_runname(run_dir.name)
    results: List[Dict[str, Any]] = []

    for cond_dir in sorted(run_dir.iterdir()):
        if not cond_dir.is_dir():
            continue
        condition_id = cond_dir.name
        episodes_dir = cond_dir / "episodes"
        if not episodes_dir.is_dir():
            continue

        # Aggregate per-condition counters (across all tasks)
        # We sub-aggregate by site (from JSONL benchmark_site field, more
        # robust than parsing config_meta)
        per_site: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "ab": 0})

        for jsonl in episodes_dir.glob("*_steps_v2.jsonl"):
            try:
                rows = read_jsonl_dedup(jsonl)
            except Exception as exc:
                print(f"  [WARN] failed to read {jsonl}: {exc}", file=sys.stderr)
                continue
            for row in rows:
                site = str(row.get("benchmark_site", "unknown"))
                per_site[site]["total"] += 1
                if _is_about_blank_step(row):
                    per_site[site]["ab"] += 1

        cond_meta = _infer_mode_baseline(condition_id)
        for site, counts in per_site.items():
            total = counts["total"]
            ab = counts["ab"]
            results.append({
                "run": run_dir.name,
                "condition_id": condition_id,
                "mode": cond_meta["mode"],
                "baseline": baseline,
                "site": site,
                "total_steps": total,
                "about_blank_steps": ab,
                "about_blank_rate_pct": (100.0 * ab / total) if total else 0.0,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs", nargs="+", required=True,
        help="One or more run-dir globs, e.g. 'results/visualwebarena/phase1/B*_3mode_*'",
    )
    parser.add_argument(
        "--output", default="-",
        help="Output JSON path; '-' for stdout. Default: -",
    )
    args = parser.parse_args()

    all_results: List[Dict[str, Any]] = []
    for pattern in args.runs:
        for run_path in sorted(glob.glob(pattern)):
            run_dir = Path(run_path)
            if not run_dir.is_dir():
                continue
            print(f"Scanning {run_dir.name}...", file=sys.stderr)
            results = scan_run_dir(run_dir)
            all_results.extend(results)

    # Print summary table to stderr
    print("\n=== Per-condition about:blank rate ===", file=sys.stderr)
    print(
        f"{'run':<50} {'mode':<14} {'baseline':<8} {'site':<14} "
        f"{'steps':>7} {'AB':>4} {'AB%':>6}",
        file=sys.stderr,
    )
    for r in all_results:
        print(
            f"{r['run'][:50]:<50} {r['mode']:<14} {r['baseline']:<8} {r['site'][:14]:<14} "
            f"{r['total_steps']:>7} {r['about_blank_steps']:>4} "
            f"{r['about_blank_rate_pct']:>6.2f}",
            file=sys.stderr,
        )

    # Aggregate by (mode, baseline) — paper §3.5 cell-level transparency
    by_cell: Dict[tuple, Dict[str, int]] = defaultdict(lambda: {"total": 0, "ab": 0})
    for r in all_results:
        key = (r["mode"], r["baseline"], r["site"])
        by_cell[key]["total"] += r["total_steps"]
        by_cell[key]["ab"] += r["about_blank_steps"]

    print("\n=== Aggregated by (mode, baseline, site) ===", file=sys.stderr)
    print(
        f"{'mode':<14} {'baseline':<8} {'site':<14} "
        f"{'steps':>7} {'AB':>4} {'AB%':>6}",
        file=sys.stderr,
    )
    cell_summary = []
    for (mode, baseline, site), counts in sorted(by_cell.items()):
        total = counts["total"]
        ab = counts["ab"]
        rate = (100.0 * ab / total) if total else 0.0
        print(
            f"{mode:<14} {baseline:<8} {site[:14]:<14} "
            f"{total:>7} {ab:>4} {rate:>6.2f}",
            file=sys.stderr,
        )
        cell_summary.append({
            "mode": mode, "baseline": baseline, "site": site,
            "total_steps": total, "about_blank_steps": ab,
            "about_blank_rate_pct": rate,
        })

    output_payload = {
        "scope": "F1 about:blank Phase 1 frequency",
        "issue_file": "docs/checkpoints/_status/issues/issue_about_blank_systematic_2026-05-16.md",
        "per_condition": all_results,
        "by_cell": cell_summary,
    }

    if args.output == "-":
        json.dump(output_payload, sys.stdout, indent=2, ensure_ascii=False)
        print()
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
