#!/usr/bin/env python3
"""Which episodes have a steps JSONL that does not belong to their summary?

`axis_effect_size.read_steps` reads step records under `strict_identity=True`,
so any episode whose `*_steps_v2.jsonl` disagrees with its `*_summary_v2.json`
raises instead of returning a trajectory. Exactly one of the 36 (model, site,
mode) combinations was known to trip it — B0·reddit·P-SoM — and the per-mode
four-dimension profile (笔记 §108 / INDEX §7) cannot be computed over the Macro
and Micro dimensions until the blast radius is known.

Root cause found 2026-07-28 on that combination: the quarantine → resume-rerun
path writes a NEW summary but leaves the ORIGINAL steps JSONL in place. On
reddit task 149 the quarantined summary (18 steps, 68741 tokens, killed by an
AWS proxy 503 mid-episode) matches the current steps file exactly, while the
live summary is the 2026-07-08 clean rerun (13 steps, 49657 tokens). The two
files describe different executions. `.stale_<ts>.jsonl` siblings exist, so the
backup half of that path ran; the replace half did not.

This script answers the only question that matters downstream: for every
paper-grade condition, which episodes can supply a trustworthy trajectory.

  - summary-only metrics (Outcome / Efficiency) are unaffected: the live
    summary is the clean rerun and the condition was bound by
    validate_fire_manifest.
  - trajectory metrics (Macro / Micro) must exclude the mismatched episodes and
    say how many were excluded.

Usage:
  .venv/bin/python3 scripts/analysis/audit_steps_summary_identity.py
  .venv/bin/python3 scripts/analysis/audit_steps_summary_identity.py --json-out out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import scripts.analysis.axis_effect_size as A  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

BASELINES = ("B0", "B1", "B2")
SITES = ("classifieds", "reddit")


def audit_condition(baseline: str, site: str, mode: str) -> dict | None:
    ep_dir = A.STEP_DIRS.get(baseline, {}).get(site, {}).get(mode)
    if ep_dir is None or not ep_dir.exists():
        return None
    # AMENDMENT_08 keeps the runner COLLECTING the protocol-excluded reddit tasks
    # (58, 160), so a landed condition holds 205 step files against a 203-task
    # SCORED set. Auditing the raw glob would report a denominator no downstream
    # consumer uses. Excluded files are counted separately, not silently dropped.
    universe, _ = expected_scored_ids(site)
    all_paths = sorted(ep_dir.glob(f"{site}_task_*_steps_v2.jsonl"))
    paths = [p for p in all_paths if A.step_task_id(p) in universe]
    n_unscored = len(all_paths) - len(paths)
    ok, bad = [], []
    for p in paths:
        tid = A.step_task_id(p)
        try:
            A.read_steps(p)
            ok.append(tid)
        except Exception as exc:  # noqa: BLE001 — we want the message, not the type
            bad.append({"task_id": tid, "error": type(exc).__name__,
                        "msg": str(exc)[:200]})
    stale = sorted(q.name for q in ep_dir.glob("*_steps_v2.stale_*.jsonl"))
    quarantined = sorted(q.name for q in (ep_dir / "quarantine").glob("*.json")) \
        if (ep_dir / "quarantine").is_dir() else []
    return {
        "cell": f"{baseline}_{site}", "mode": mode,
        "episodes_dir": str(ep_dir.relative_to(REPO)),
        "n_step_files": len(paths), "n_ok": len(ok), "n_mismatch": len(bad),
        "n_collected_but_unscored": n_unscored,
        "mismatched_tasks": sorted(b["task_id"] for b in bad),
        "mismatch_detail": bad,
        "stale_backups": stale, "quarantined_summaries": quarantined,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json-out", type=Path)
    a = ap.parse_args()

    modes = list(A._REGISTRY_MODE_TO_AXIS_KEY.values())
    rows, total_bad, total_ep, missing = [], 0, 0, []
    for baseline in BASELINES:
        for site in SITES:
            for mode in modes:
                r = audit_condition(baseline, site, mode)
                if r is None:
                    missing.append(f"{baseline}_{site}/{mode}")
                    continue
                rows.append(r)
                total_bad += r["n_mismatch"]
                total_ep += r["n_step_files"]

    print(f"{'cell':16s} {'mode':14s} {'scored':>6s} {'unscd':>6s} {'ok':>5s} "
          f"{'MISMATCH':>9s} {'stale':>6s} {'quar':>5s}  mismatched task ids")
    print("-" * 110)
    for r in rows:
        flag = "  <<<" if r["n_mismatch"] else ""
        print(f"{r['cell']:16s} {r['mode']:14s} {r['n_step_files']:6d} "
              f"{r['n_collected_but_unscored']:6d} "
              f"{r['n_ok']:5d} {r['n_mismatch']:9d} {len(r['stale_backups']):6d} "
              f"{len(r['quarantined_summaries']):5d}  "
              f"{r['mismatched_tasks'] if r['mismatched_tasks'] else ''}{flag}")

    total_unscored = sum(r["n_collected_but_unscored"] for r in rows)
    print(f"\n{len(rows)} combinations audited ({total_ep} SCORED episodes; "
          f"{total_unscored} further step files were collected but are outside the "
          f"scored universe — AMENDMENT_08 protocol-excluded — and were not audited), "
          f"{total_bad} step files disagree with their summary.")
    if missing:
        print(f"⚠️ {len(missing)} combinations had no episode dir: {missing}")
    affected = [r for r in rows if r["n_mismatch"]]
    if affected:
        print("\nTrajectory (Macro / Micro) metrics must exclude these and report the count.")
        print("Summary-only (Outcome / Efficiency) metrics are unaffected — the live "
              "summary is the clean rerun.")
        for r in affected:
            print(f"  {r['cell']}/{r['mode']}: {r['mismatched_tasks']} "
                  f"({r['n_mismatch']}/{r['n_step_files']} = "
                  f"{100.0*r['n_mismatch']/r['n_step_files']:.1f}%)")
    else:
        print("\nNo mismatches — trajectory metrics can use every episode.")

    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(
            {"n_combinations": len(rows), "n_episodes": total_ep,
             "n_mismatch": total_bad, "missing_combinations": missing,
             "per_condition": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nwrote {a.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
