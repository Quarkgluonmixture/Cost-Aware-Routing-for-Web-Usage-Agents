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


NOISE_INVENTORY = REPO / "scripts/analysis/aggregate_noise_floor_inventory.py"
_PHASE1_PREFIX = "results/visualwebarena/phase1/"


def registered_replicate_run_ids() -> frozenset[str]:
    """Run IDs registered as the REPLICATE arm of a same-condition pair.

    B-1951 (2026-08-03): a deliberate same-condition replicate — the second run
    used to measure the stochastic noise floor — is not a ghost. Two of the three
    existing replicates (B0.cls.dom, B0.cls.vision) live under
    ``results/repro_replicates/`` and were therefore invisible to this scanner by
    construction. The third (B0.cls.som, run R30696) was deliberately kept under
    ``results/visualwebarena/phase1/`` — see 实验笔记 §424.4(2), which records
    the resulting halt as *"B-1927 的守卫按设计工作 … 非故障"* — and it is the arm
    that finally answered §242 (the drop-one oracle's 1.7-3.3pp does NOT clear the
    2.32-2.53pp exchangeability floor).

    So the scanner needs to tell "second run I meant to make" from "second run
    that should not exist". The registry already exists and is authoritative:
    ``CLEAN_PAIRS`` in ``aggregate_noise_floor_inventory.py`` names, per pair, the
    canonical arm and the replicate arm. Read via ``ast.literal_eval`` rather than
    import so this stays side-effect-free and keeps working if that module grows
    heavier imports.

    Only arm_b entries UNDER phase1/ matter — a replicate parked in
    ``repro_replicates/`` never reaches ghost detection anyway.
    """
    import ast

    try:
        tree = ast.parse(NOISE_INVENTORY.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        # Registry unreadable → register nothing. Fail CLOSED: every extra run
        # stays a ghost, which is the pre-B-1951 behaviour.
        return frozenset()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == "CLEAN_PAIRS" for t in node.targets):
            continue
        try:
            pairs = ast.literal_eval(node.value)
        except ValueError:
            return frozenset()
        out = set()
        for entry in pairs:
            if len(entry) != 3:
                continue
            _label, _arm_a, arm_b = entry
            if isinstance(arm_b, str) and arm_b.startswith(_PHASE1_PREFIX):
                out.add(arm_b[len(_PHASE1_PREFIX):].split("/")[0])
        return frozenset(out)
    return frozenset()


def registered_replicate_pairs() -> list[tuple[str, str, str]]:
    """(label, canonical_run_id, replicate_run_id) for pairs whose replicate is under phase1/."""
    import ast

    try:
        tree = ast.parse(NOISE_INVENTORY.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "CLEAN_PAIRS" for t in node.targets
        ):
            try:
                pairs = ast.literal_eval(node.value)
            except ValueError:
                return []
            out = []
            for entry in pairs:
                if len(entry) != 3:
                    continue
                label, arm_a, arm_b = entry
                if not (isinstance(arm_b, str) and arm_b.startswith(_PHASE1_PREFIX)):
                    continue
                a = arm_a[len(_PHASE1_PREFIX):].split("/")[0] if arm_a.startswith(_PHASE1_PREFIX) else arm_a
                out.append((label, a, arm_b[len(_PHASE1_PREFIX):].split("/")[0]))
            return out
    return []


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
    replicate_notes: list[str] = []
    # B-1951: the registry of deliberate second runs, plus the inverse guard.
    _replicate_run_ids = registered_replicate_run_ids()
    inverted_bindings: list[str] = []
    for _label, _canon, _rep in registered_replicate_pairs():
        for _k, _c in conditions.items():
            if _c.get("run_id") == _rep:
                inverted_bindings.append(
                    f"{_k}: manifest binds '{_rep}', which {NOISE_INVENTORY.name} "
                    f"registers as the REPLICATE arm of pair '{_label}' "
                    f"(canonical is '{_canon}'). Binding the replicate as "
                    f"authoritative collapses the noise-floor comparison — both "
                    f"arms become the same run and the floor it measures vanishes."
                )
    unbound_singletons: dict[str, str] = {}
    unbound_ambiguous: list[str] = []
    over_complete: list[str] = []  # B-1834: episodes > scored = contamination
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
                    # B-1834 (3-AI /stress 2026-05-22, codex F1): an authoritative
                    # binding whose run is OVER-complete (episodes > scored) is a
                    # corrupted binding (dedup failure / double-run) → fail-closed.
                    _auth_eps = episodes_in(RESULTS_ROOT / auth / cid / "condition_summary_v2.json")
                    if _auth_eps > exp:
                        over_complete.append(
                            f"{key}: authoritative '{auth}' has {_auth_eps} ep > scored {exp} (corrupted binding)"
                        )
                    # Ghost = a non-authoritative run that is COMPLETE (episodes >=
                    # scored) → real double-count risk. A partial/aborted extra (e.g.
                    # an interrupted re-fire) is NOT a ghost: it's below the scored
                    # count so the aggregator's scored-count gate already excludes it.
                    # B-1951: a run registered as the replicate arm of a
                    # same-condition pair is a deliberate second run, not a ghost.
                    complete_ghosts = [
                        r for r in runs if r != auth
                        and r not in _replicate_run_ids
                        and episodes_in(RESULTS_ROOT / r / cid / "condition_summary_v2.json") >= exp
                    ]
                    _excused = [r for r in runs if r != auth and r in _replicate_run_ids]
                    for r in _excused:
                        replicate_notes.append(f"{key}: registered replicate '{r}' (not a ghost)")
                    if complete_ghosts:
                        for r in complete_ghosts:
                            eps = episodes_in(RESULTS_ROOT / r / cid / "condition_summary_v2.json")
                            ghosts.append(f"{key}: COMPLETE ghost '{r}' ({eps} ep) ≠ authoritative '{auth}'")
                    else:
                        ok_bound += 1
                else:
                    # Not yet bound. A run is bindable iff EXACTLY scored. B-1834
                    # (3-AI /stress 2026-05-22, codex F1): the prior `>= scored`
                    # would auto-bind an over-complete (e.g. 225/224, dedup-failure
                    # contaminated) run and then skip it forever = paper-grade
                    # denominator corruption. Require `== scored`; flag `> scored`
                    # fail-closed (never silently bind OR silently ignore it).
                    _cid_u = condition_id_for_mode(mode)
                    _exp_u = int(scored.get(site, 10**9))
                    valid = [r for r in runs
                             if episodes_in(RESULTS_ROOT / r / _cid_u / "condition_summary_v2.json") == _exp_u]
                    for r in runs:
                        _e_u = episodes_in(RESULTS_ROOT / r / _cid_u / "condition_summary_v2.json")
                        if _e_u > _exp_u:
                            over_complete.append(
                                f"{key}: unbound '{r}' has {_e_u} ep > scored {_exp_u} (contamination — never auto-bind)"
                            )
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
    if over_complete:
        print(f"[validate_fire_manifest][FAIL] {len(over_complete)} OVER-COMPLETE run(s) "
              "(episodes > scored) — contamination, must halt (B-1834):", file=sys.stderr)
        for o in over_complete:
            print(f"  ✗ {o}", file=sys.stderr)
        print("  Fix: a run with MORE than scored episodes = dedup failure / double-run. "
              "Investigate (scripts/maintenance/clear_tasks.py the duplicate task data) "
              "before binding/aggregating.", file=sys.stderr)
        return 1
    if replicate_notes:
        for n in replicate_notes:
            print(f"[validate_fire_manifest][replicate] {n}")
    if inverted_bindings:
        print("[validate_fire_manifest][FAIL] manifest binds a REPLICATE arm as authoritative:")
        for n in inverted_bindings:
            print(f"  ✗ {n}")
        print("  Fix: rebind the condition to the canonical arm (arm_a of the pair).")
        return 1
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
