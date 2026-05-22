#!/usr/bin/env python3
"""B-1825 (Fire-6 /stress P0-3-AC*): manifest-bound ghost-run gate.

The cross-condition aggregators glob `results/visualwebarena/phase1/{baseline}_
{mode}_{site}_*` and (per phase1a_status.sh:94) take the `ls -dt` LATEST run for a
condition. If a relaunch produces a 2nd copy of an already-completed condition
(e.g. R9755 B0-dom-cls re-run), the latest-run heuristic silently double-counts or
cherry-picks — a reviewer-defense liability (gemini C-P0 "ghost run").

This gate binds aggregation to `docs/checkpoints/pre_run/fire_manifest.json`:
a paper-grade condition may have AT MOST its manifest-listed authoritative run on
disk. Any additional paper-grade run for a manifest-listed condition = a GHOST →
exit 1 (aggregation MUST halt; operator clears the ghost or re-binds the manifest).

Run this BEFORE any cross-condition aggregation (wire into `make analysis`).

Usage:
  python3 scripts/analysis/validate_fire_manifest.py
  python3 scripts/analysis/validate_fire_manifest.py --min-date 20260516
  python3 scripts/analysis/validate_fire_manifest.py --populate   # suggest entries for unbound, unambiguous conditions

Exit 0 = clean (every manifest-listed condition has only its authoritative run).
Exit 1 = ghost run(s) detected, or manifest missing/invalid.
Exit 2 = usage error.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO / "docs/checkpoints/pre_run/fire_manifest.json"
RESULTS_ROOT = REPO / "results/visualwebarena/phase1"
MODES = ("dom", "som", "vision", "phantom_text", "phantom_som", "phantom_prompt")
BASELINES = ("B0", "B1", "B2")
SITES = ("classifieds", "reddit", "shopping")
DATE_RE = re.compile(r"(\d{8})")


def condition_id_for_mode(mode: str) -> str:
    # Matches phase1a_status.sh:condition_id_for_mode + queue conditions.py labels.
    return f"phase1_{mode}_router_0"


def episodes_in(summary: Path) -> int:
    try:
        return int(json.loads(summary.read_text()).get("episodes", 0))
    except Exception:
        return 0


def discover_runs(baseline: str, mode: str, site: str, min_date: int) -> list[str]:
    """Paper-grade run dirs for a condition (post-min-date), newest first."""
    pat = f"{baseline}_{mode}_{site}_"
    out = []
    if not RESULTS_ROOT.is_dir():
        return out
    for d in sorted(RESULTS_ROOT.glob(f"{pat}*"), key=lambda p: p.name, reverse=True):
        if not d.is_dir():
            continue
        if "smoke" in d.name or "_test" in d.name:
            continue  # smoke / test runs are never paper-grade
        m = DATE_RE.search(d.name)
        if m and int(m.group(1)) >= min_date:
            out.append(d.name)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-date", type=int, default=20260516,
                    help="only count runs at-or-after this YYYYMMDD as paper-grade")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--populate", action="store_true",
                    help="print manifest entries for unbound conditions with exactly 1 paper-grade run")
    ap.add_argument("--apply", action="store_true",
                    help="with --populate: actually WRITE the unbound unambiguous bindings to the "
                         "manifest (atomic + flock). NARROW semantics — only adds a condition that "
                         "is unbound AND has exactly ONE complete valid run. NEVER overwrites an "
                         "existing binding, never glob-latest, never picks by SR/mtime, never binds "
                         "partial/ambiguous/wrong-scored. Ambiguous/ghost/write-error → fail-closed "
                         "(exit 1). For the in-pipeline watchdog auto-bind hook (B-1830 followup).")
    args = ap.parse_args()

    man_path = Path(args.manifest)
    try:
        manifest = json.loads(man_path.read_text())
    except Exception as e:
        print(f"[validate_fire_manifest][FATAL] cannot read manifest {man_path}: {e}", file=sys.stderr)
        return 1

    conditions = manifest.get("conditions", {})
    scored = manifest.get("scored_task_count", {})
    bound_run_ids = {c["run_id"] for c in conditions.values() if "run_id" in c}

    ghosts: list[str] = []
    unbound_singletons: dict[str, str] = {}
    unbound_ambiguous: list[str] = []
    ok_bound = 0

    for site in SITES:
        for bl in BASELINES:
            for mode in MODES:
                key = f"{site}|{bl}|{mode}"
                runs = discover_runs(bl, mode, site, args.min_date)
                if not runs:
                    continue
                if key in conditions:
                    auth = conditions[key]["run_id"]
                    cid = condition_id_for_mode(mode)
                    exp = int(scored.get(site, 10**9))
                    # Ghost = a non-authoritative run that is COMPLETE (episodes >=
                    # scored) → real double-count risk. A partial/aborted extra (e.g.
                    # an interrupted re-fire) is NOT a ghost: it's below the scored
                    # count so the aggregator's scored-count gate already excludes it.
                    complete_ghosts = [
                        r for r in runs if r != auth
                        and episodes_in(RESULTS_ROOT / r / cid / "condition_summary_v2.json") >= exp
                    ]
                    if complete_ghosts:
                        for r in complete_ghosts:
                            eps = episodes_in(RESULTS_ROOT / r / cid / "condition_summary_v2.json")
                            ghosts.append(f"{key}: COMPLETE ghost '{r}' ({eps} ep) ≠ authoritative '{auth}'")
                    else:
                        ok_bound += 1
                else:
                    # Not yet bound. A single paper-grade run is bindable; >1 = ambiguous ghost.
                    valid = [r for r in runs
                             if episodes_in(RESULTS_ROOT / r / condition_id_for_mode(mode) / "condition_summary_v2.json")
                             >= int(scored.get(site, 10**9))]
                    if len(valid) == 1:
                        unbound_singletons[key] = valid[0]
                    elif len(valid) > 1:
                        unbound_ambiguous.append(f"{key}: {len(valid)} complete runs, none bound → {valid}")

    if args.populate:
        if unbound_singletons:
            print("# Suggested fire_manifest.json entries (unbound, unambiguous):")
            for key, run_id in unbound_singletons.items():
                mode = key.split("|")[2]
                print(json.dumps({key: {"run_id": run_id,
                                        "condition_id": condition_id_for_mode(mode),
                                        "episodes": episodes_in(RESULTS_ROOT / run_id / condition_id_for_mode(mode) / "condition_summary_v2.json"),
                                        "bound_date": "TODO"}}, indent=2))
        else:
            print("# No unbound, unambiguous complete conditions to suggest.")
        if unbound_ambiguous:
            print("\n# ⚠️  Ambiguous (multiple complete runs, manual pick required):", file=sys.stderr)
            for a in unbound_ambiguous:
                print(f"#   {a}", file=sys.stderr)

        # B-1830 followup (2026-05-22, user directive): --apply WRITES the
        # unbound + unambiguous bindings (narrow semantics). flock + atomic
        # replace; re-read under lock to avoid lost-update. NEVER overwrites an
        # existing binding (idempotent skip). Ambiguous (unbound_ambiguous) +
        # ghosts are NOT touched here — they fall through to the fail-closed
        # verdict below (exit 1). Write error → fail-closed (exit 1) too.
        if args.apply:
            if not unbound_singletons:
                print("[validate_fire_manifest][APPLY] nothing to bind (0 unbound unambiguous).")
            else:
                import fcntl
                _lockp = man_path.with_suffix(".applylock")
                try:
                    with open(_lockp, "w") as _lk:
                        fcntl.flock(_lk, fcntl.LOCK_EX)
                        _m = json.loads(man_path.read_text())  # re-read under lock (lost-update guard)
                        _conds = _m.setdefault("conditions", {})
                        _added = []
                        for key, run_id in unbound_singletons.items():
                            if key in _conds:
                                continue  # bound since discovery → idempotent, NEVER overwrite
                            mode = key.split("|")[2]
                            _conds[key] = {
                                "run_id": run_id,
                                "condition_id": condition_id_for_mode(mode),
                                "episodes": episodes_in(RESULTS_ROOT / run_id / condition_id_for_mode(mode) / "condition_summary_v2.json"),
                                "bound_date": datetime.now().strftime("%Y-%m-%d"),
                                "bound_by": "validate_fire_manifest --apply (in-pipeline auto-bind)",
                            }
                            _added.append(f"{key} → {run_id}")
                        _tmp = man_path.with_suffix(".json.tmp")
                        _tmp.write_text(json.dumps(_m, ensure_ascii=False, indent=1) + "\n")
                        os.replace(str(_tmp), str(man_path))
                    for a in _added:
                        print(f"[validate_fire_manifest][APPLY] bound {a}")
                    # Reflect writes into in-memory state so the verdict below is accurate.
                    for key in list(unbound_singletons):
                        conditions.setdefault(key, _conds.get(key, {}))
                    ok_bound += len(unbound_singletons)
                    unbound_singletons = {}
                except Exception as e:
                    print(f"[validate_fire_manifest][FAIL] --apply write error: {e}", file=sys.stderr)
                    return 1

    # Verdict
    print(f"[validate_fire_manifest] bound-clean conditions: {ok_bound}")
    print(f"[validate_fire_manifest] unbound singletons (bindable): {len(unbound_singletons)}")
    if unbound_ambiguous:
        print(f"[validate_fire_manifest] ⚠️  unbound AMBIGUOUS (multi-run): {len(unbound_ambiguous)}")
        for a in unbound_ambiguous:
            print(f"  {a}")
    if ghosts:
        print(f"[validate_fire_manifest][FAIL] {len(ghosts)} GHOST run(s) — aggregation must halt:", file=sys.stderr)
        for g in ghosts:
            print(f"  ✗ {g}", file=sys.stderr)
        print("  Fix: remove the ghost run dir (scripts/maintenance/clear_tasks.py) OR re-bind "
              "the manifest if the ghost is the new authoritative.", file=sys.stderr)
        return 1
    # Ambiguous unbound = also fail-closed (can't aggregate without knowing authoritative).
    if unbound_ambiguous:
        print("[validate_fire_manifest][FAIL] ambiguous unbound conditions — bind them in the "
              "manifest before aggregating.", file=sys.stderr)
        return 1
    print("[validate_fire_manifest][OK] no ghost runs; all paper-grade runs are manifest-consistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
