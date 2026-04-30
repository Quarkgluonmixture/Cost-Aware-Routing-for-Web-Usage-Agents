#!/usr/bin/env python3
"""B-37 pilot gate: compare pilot T=0 SR vs existing paper-grade DOM SR.

Decision matrix:
  PASS:     pilot SR within ±5pp existing → green-light Phase A full re-run
  MARGINAL: -5pp to -15pp                  → tune top_p / consider mild T=0.05
  FAIL:     < -15pp or mode collapse       → revert T=0→0.1, paper takes disclosure path

Mode collapse signature: ≥80% of pilot tasks share same first-action element_id
across episodes (model getting stuck on first visible element).

Usage:
  .venv/bin/python3 scripts/analysis/compare_pilot_t0_vs_paper_grade.py
  .venv/bin/python3 scripts/analysis/compare_pilot_t0_vs_paper_grade.py --site classifieds

Output: stdout summary + docs/analysis/cross_sites/pilot_t0_decision.{md,json}
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/visualwebarena/phase1"
OUT_JSON = ROOT / "docs/analysis/cross_sites/pilot_t0_decision.json"
OUT_MD = ROOT / "docs/analysis/cross_sites/pilot_t0_decision.md"

# Existing paper-grade DOM run roots per site (most recent paper-grade clean)
PAPER_GRADE_DOM = {
    "classifieds": ("B0_3mode_classifieds_20260413", "phase1_dom_router_0"),
    "reddit": ("B0_3mode_reddit_20260422", "phase1_dom_router_0"),
    "shopping": ("B0_dom_shopping_20260428", "phase1_dom_router_0"),
}


def find_pilot_run(site: str) -> Path | None:
    cands = sorted(RESULTS.glob(f"B0_dom_pilot_T0_{site}_*"))
    return cands[-1] if cands else None


def load_summaries(run_dir: Path, condition_id: str, site: str) -> List[Dict[str, Any]]:
    ep_dir = run_dir / condition_id / "episodes"
    if not ep_dir.exists():
        return []
    out = []
    for p in sorted(ep_dir.glob(f"{site}_task_*_summary_v2.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            pass
    return out


def task_subset_ids(summaries: List[Dict[str, Any]]) -> set[int]:
    return {int(s.get("task_id", -1)) for s in summaries if s.get("task_id") is not None}


def compute_sr(summaries: List[Dict[str, Any]], task_filter: set[int] | None = None) -> Dict[str, Any]:
    rows = summaries if task_filter is None else [s for s in summaries if int(s.get("task_id", -1)) in task_filter]
    if not rows:
        return {"n": 0, "sr_raw": None, "n_success": 0}
    n = len(rows)
    n_success = sum(1 for s in rows if bool(s.get("success", False)))
    return {
        "n": n,
        "n_success": n_success,
        "sr_raw": round(100.0 * n_success / n, 2) if n else None,
    }


def detect_mode_collapse(pilot_summaries: List[Dict[str, Any]], pilot_run_dir: Path, site: str) -> Dict[str, Any]:
    """Mode collapse signature: ≥80% of episodes share same first-action element_id."""
    first_actions = []
    ep_dir = pilot_run_dir / "phase1_dom_router_0" / "episodes"
    for s in pilot_summaries:
        tid = s.get("task_id")
        if tid is None:
            continue
        steps_path = ep_dir / f"{site}_task_{tid}_steps_v2.jsonl"
        if not steps_path.exists():
            continue
        try:
            for line in steps_path.read_text().splitlines():
                if not line.strip():
                    continue
                step = json.loads(line)
                if step.get("step_idx") == 0:
                    action = step.get("action") or {}
                    eid = action.get("element_id") if isinstance(action, dict) else None
                    at = action.get("action_type") if isinstance(action, dict) else None
                    first_actions.append((at, eid))
                    break
        except Exception:
            pass
    if not first_actions:
        return {"checked": False, "n_episodes": 0}
    counter = Counter(first_actions)
    most_common, n_most = counter.most_common(1)[0]
    pct = 100.0 * n_most / len(first_actions)
    return {
        "checked": True,
        "n_episodes": len(first_actions),
        "most_common_first_action": most_common,
        "most_common_count": n_most,
        "most_common_pct": round(pct, 1),
        "collapse_detected": pct >= 80.0,
    }


def verdict(pilot_sr: float | None, paper_sr: float | None, collapsed: bool) -> str:
    if collapsed:
        return "FAIL_MODE_COLLAPSE"
    if pilot_sr is None or paper_sr is None:
        return "INCONCLUSIVE_INSUFFICIENT_DATA"
    delta = pilot_sr - paper_sr
    if delta >= -5.0:
        return "PASS"
    if delta >= -15.0:
        return f"MARGINAL ({delta:+.1f}pp)"
    return f"FAIL ({delta:+.1f}pp)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", choices=["classifieds", "reddit", "shopping", "all"], default="all")
    args = ap.parse_args()

    sites = ["classifieds", "reddit", "shopping"] if args.site == "all" else [args.site]
    results: Dict[str, Any] = {"audit_date": "2026-04-30", "per_site": {}}

    for site in sites:
        pilot_dir = find_pilot_run(site)
        if pilot_dir is None:
            results["per_site"][site] = {"verdict": "NOT_RUN_YET"}
            print(f"[{site}] no pilot run found", file=sys.stderr)
            continue

        pilot_summaries = load_summaries(pilot_dir, "phase1_dom_router_0", site)
        if not pilot_summaries:
            results["per_site"][site] = {"verdict": "NO_SUMMARIES_YET", "pilot_run": pilot_dir.name}
            print(f"[{site}] pilot {pilot_dir.name} has no summaries yet (still running)", file=sys.stderr)
            continue

        pilot_sr = compute_sr(pilot_summaries)
        pilot_task_ids = task_subset_ids(pilot_summaries)

        # Compare against paper-grade SAME task subset (matched by task_id)
        paper_run, paper_cond = PAPER_GRADE_DOM[site]
        paper_dir = RESULTS / paper_run
        paper_summaries = load_summaries(paper_dir, paper_cond, site)
        paper_full_sr = compute_sr(paper_summaries)
        paper_matched_sr = compute_sr(paper_summaries, pilot_task_ids)

        collapse = detect_mode_collapse(pilot_summaries, pilot_dir, site)

        ver = verdict(pilot_sr["sr_raw"], paper_matched_sr["sr_raw"], collapse.get("collapse_detected", False))

        results["per_site"][site] = {
            "pilot_run": pilot_dir.name,
            "paper_grade_run": paper_run,
            "pilot_sr": pilot_sr,
            "paper_grade_full_sr": paper_full_sr,
            "paper_grade_matched_subset_sr": paper_matched_sr,
            "task_subset_size": len(pilot_task_ids),
            "mode_collapse_check": collapse,
            "verdict": ver,
        }
        print(
            f"[{site}] pilot SR={pilot_sr['sr_raw']}% (n={pilot_sr['n']}) "
            f"vs paper-grade matched {paper_matched_sr['sr_raw']}% (n={paper_matched_sr['n']}) "
            f"→ {ver}",
            file=sys.stderr,
        )

    # Aggregate verdict across sites
    verdicts = [r.get("verdict", "?") for r in results["per_site"].values()]
    if all(v == "PASS" for v in verdicts):
        results["aggregate_verdict"] = "GREEN_LIGHT_PHASE_A"
    elif any("FAIL" in v for v in verdicts):
        results["aggregate_verdict"] = "RED_LIGHT_REVERT_T0"
    else:
        results["aggregate_verdict"] = "YELLOW_INVESTIGATE"

    OUT_JSON.write_text(json.dumps(results, indent=2))

    md = ["# B-37 Pilot T=0 Decision Gate", ""]
    md.append(f"**Audit date**: {results['audit_date']}")
    md.append(f"**Aggregate verdict**: **{results['aggregate_verdict']}**")
    md.append("")
    md.append("## Per-site summary")
    md.append("")
    md.append("| Site | Pilot run | Pilot SR (N) | Paper-grade matched SR (N) | Mode collapse % | Verdict |")
    md.append("|---|---|---:|---:|---:|---|")
    for site, r in results["per_site"].items():
        if "verdict" not in r or r.get("verdict") in ("NOT_RUN_YET", "NO_SUMMARIES_YET"):
            md.append(f"| {site} | — | — | — | — | {r.get('verdict','?')} |")
            continue
        psr = r["pilot_sr"]
        msr = r["paper_grade_matched_subset_sr"]
        col = r["mode_collapse_check"]
        md.append(
            f"| {site} | `{r['pilot_run']}` | {psr['sr_raw']}% ({psr['n']}) | "
            f"{msr['sr_raw']}% ({msr['n']}) | {col.get('most_common_pct','—')}% | "
            f"**{r['verdict']}** |"
        )
    md.append("")
    md.append("## Decision matrix")
    md.append("")
    md.append("- **PASS** (within ±5pp): green-light Phase A full re-run with T=0 baseline")
    md.append("- **MARGINAL** (-5 to -15pp): investigate top_p / try T=0.05 / check first-action distribution")
    md.append("- **FAIL** (< -15pp or mode collapse): revert T=0→0.1, paper takes B-37 disclosure path")
    md.append("")
    md.append("Mode collapse signature: ≥80% of episodes share same first action (action_type, element_id)")

    OUT_MD.write_text("\n".join(md))

    print(f"\n=== Aggregate verdict: {results['aggregate_verdict']} ===")
    print(f"Wrote {OUT_MD}")
    print(f"      {OUT_JSON}")


if __name__ == "__main__":
    main()
