#!/usr/bin/env python3
"""Validate run_manifest.yaml for paper-grade promotion correctness.

A1.21 P0-8 fix (B-525, /stress Claude unique 2026-05-17):

Pre-fix: paper-grade promotion = user manually edits yaml to add 36 cells (2 sites
× 3 baselines × 6 modes). No schema validator. No disk-existence check. No canonical
`expected_n` enforcement. `_entry_to_cell` only `RuntimeWarning` on actual_n==0.
Human typo (e.g., `condition_subdir: phase1_phantom_dom_router_0` vs canonical
`phase1_phantom_text_router_0`) → silent drop cell → paper §1 hero pool k=5 instead
of k=6 → reviewer R3 attack "为何 k=5?"

Post-fix: this validator MUST pass before `make analysis` runs.

Checks (all must pass for `--strict` mode):
  (a) Every paper-grade (baseline, site) has all 6 PAPER_MODES present
  (b) `expected_n` matches `scored_task_count(site, "visualwebarena", strict=True)`
      for paper-grade tier (post-§139.8 canonical)
  (c) `run_dir/condition_subdir` exists on disk
  (d) Episode summary count ≥ MIN_EP_FOR_CELL (= 50, per aggregate_phantom_lift)
  (e) yaml section ↔ grade alignment (`cells:` only paper-grade*; `in_flight:` only
      in-flight; `archived:` only archived) — P2-1 / codex catch
  (f) No duplicate (baseline, site, mode, grade) within manifest
  (g) Phase 1a planned cells (6 stratification units) all present in paper-grade tier

Output:
  - exit 0 + "VALIDATION PASSED" if all checks pass
  - exit 2 + per-check error list if any fail
  - --strict raises after first failure; default reports all + summary

Usage:
  python3 scripts/analysis/validate_run_manifest.py              # default check
  python3 scripts/analysis/validate_run_manifest.py --strict     # fail-fast for CI
  python3 scripts/analysis/validate_run_manifest.py --no-disk    # skip disk checks (faster)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.analysis.lib.run_registry import (  # noqa: E402
    PAPER_MODES,
    GRADES,
    EXPECTED_N,
    load_manifest,
    _iter_manifest_entries,
    canonical_mode,
)
from scripts.analysis.lib.canonical_cells import (  # noqa: E402
    PHASE_1A_PLANNED_CELLS,
)

MIN_EP_FOR_CELL = 50  # matches scripts/analysis/aggregate_phantom_lift.MIN_EP_FOR_CELL


def validate_manifest(
    manifest_path: Path,
    *,
    strict: bool = False,
    check_disk: bool = True,
) -> list[str]:
    """Run all validation checks. Returns list of error strings."""
    errors: list[str] = []
    manifest = load_manifest(manifest_path)

    # Build entries with section attribution for (e) check
    section_entries: dict[str, list[dict]] = {}
    for section in ("cells", "in_flight", "archived"):
        section_entries[section] = manifest.get(section) or []

    # (e) section ↔ grade alignment
    for section, entries in section_entries.items():
        for entry in entries:
            if isinstance(entry, str):
                continue  # string entries valid only in `archived` (handled by _iter_manifest_entries)
            grade = entry.get("grade")
            if section == "cells":
                if grade not in ("paper-grade", "paper-grade-pre-bug"):
                    errors.append(
                        f"[section] entry under `cells:` has grade={grade!r} "
                        f"(should be paper-grade or paper-grade-pre-bug): "
                        f"{entry.get('baseline')}/{entry.get('site')}/{entry.get('mode')}"
                    )
            elif section == "in_flight":
                if grade != "in-flight":
                    errors.append(
                        f"[section] entry under `in_flight:` has grade={grade!r} "
                        f"(should be in-flight): {entry.get('baseline')}/{entry.get('site')}/{entry.get('mode')}"
                    )
            elif section == "archived":
                if grade != "archived":
                    errors.append(
                        f"[section] entry under `archived:` has grade={grade!r} "
                        f"(should be archived): {entry.get('baseline')}/{entry.get('site')}/{entry.get('mode')}"
                    )

    all_entries = _iter_manifest_entries(manifest)

    # Build paper-grade cell map for (a), (g), (b), (c), (d)
    paper_grade_by_cell: dict[tuple[str, str], dict[str, dict]] = {}
    seen_keys: set[tuple[str, str, str, str]] = set()
    for entry in all_entries:
        grade = entry.get("grade")
        baseline = entry.get("baseline", "")
        site = entry.get("site", "")
        mode_raw = entry.get("mode", "")
        try:
            mode = canonical_mode(str(mode_raw))
        except Exception:
            mode = mode_raw

        # (f) duplicate detection
        key = (baseline, site, mode, grade)
        if key in seen_keys and grade != "archived":
            errors.append(f"[duplicate] (baseline={baseline}, site={site}, mode={mode}, grade={grade}) appears twice")
        seen_keys.add(key)

        if grade in ("paper-grade", "paper-grade-pre-bug"):
            paper_grade_by_cell.setdefault((baseline, site), {})[mode] = entry

            # (b) expected_n canonical check — C2 fix 2026-05-24: unconditional.
            # Pre-fix: only checked when expected_n was present in yaml (opt-in bypass).
            # Post-fix: ALWAYS compare actual episode count on disk against EXPECTED_N[site].
            # yaml `expected_n:` field is still cross-checked when present, but the hard
            # gate uses EXPECTED_N (registry canonical) not the yaml field.
            raw_en = entry.get("expected_n")
            canonical_en = EXPECTED_N.get(site, 0)
            if raw_en is not None:
                if int(raw_en) != canonical_en:
                    errors.append(
                        f"[expected_n] {baseline}/{site}/{mode}: yaml expected_n={raw_en}, "
                        f"canonical scored_task_count({site}) = {canonical_en}. Either drop "
                        f"the `expected_n:` line or set to {canonical_en}."
                    )

            # (c) disk existence + (d) episode count
            # C2 fix 2026-05-24: (d) threshold changed from >=50 (MIN_EP_FOR_CELL) to
            # == EXPECTED_N[site] (exact canonical count). MIN_EP_FOR_CELL=50 was a
            # completeness proxy that accepted partial runs with >50 episodes as
            # "complete" for sites with canonical N=224/205 — paper-grade promotion gate
            # must be exact match, not a lower-bound heuristic.
            if check_disk:
                run_dir_str = entry.get("run_dir", "")
                cond_sub = entry.get("condition_subdir", "")
                # Resolve via registry logic
                from scripts.analysis.lib.run_registry import _resolve_run_dir
                full_dir = _resolve_run_dir(run_dir_str) / cond_sub
                if not full_dir.exists():
                    errors.append(
                        f"[disk] {baseline}/{site}/{mode}: run_dir/condition_subdir does NOT exist: {full_dir}"
                    )
                else:
                    ep_dir = full_dir / "episodes"
                    if ep_dir.exists():
                        ep_count = len(list(ep_dir.glob("*_summary_v2.json")))
                        expected_ep = EXPECTED_N.get(site, 0)
                        if expected_ep > 0 and ep_count != expected_ep:
                            errors.append(
                                f"[episode-count] {baseline}/{site}/{mode}: {ep_count} episodes "
                                f"!= EXPECTED_N({site})={expected_ep}. "
                                "Run incomplete or has extra episodes — not paper-grade promotable."
                            )
                        elif expected_ep == 0 and ep_count < MIN_EP_FOR_CELL:
                            # Fallback for unknown sites: retain the original >=50 heuristic
                            errors.append(
                                f"[episode-count] {baseline}/{site}/{mode}: {ep_count} episodes "
                                f"< MIN_EP_FOR_CELL ({MIN_EP_FOR_CELL}). Likely partial run."
                            )
                    else:
                        errors.append(f"[disk] {baseline}/{site}/{mode}: episodes/ subdir missing: {ep_dir}")

        if strict and errors:
            raise RuntimeError(f"Validator strict mode: failing on first error:\n  {errors[-1]}")

    # (a) all 6 PAPER_MODES present per paper-grade (baseline, site)
    for (baseline, site), modes_present in paper_grade_by_cell.items():
        missing = [m for m in PAPER_MODES if m not in modes_present]
        if missing:
            errors.append(
                f"[missing-modes] paper-grade {baseline}/{site}: missing modes {missing} "
                f"(only have {sorted(modes_present.keys())})"
            )

    # (g) Phase 1a planned cells all present (paper-grade)
    # C2 fix 2026-05-24: removed dead-code variable `paper_grade_planned_present` which
    # built a set via a self-cancelling filter `(b,s) in {(bb,ss) for (ss,bb) in PLANNED}`
    # — the comprehension swapped (site,baseline) order so the membership test was always
    # True when PLANNED uses (site,baseline) tuples but the filter compared (baseline,site);
    # net result: set was always == PHASE_1A_PLANNED_CELLS but variable was never read.
    planned_missing = []
    for site, baseline in PHASE_1A_PLANNED_CELLS:
        if (baseline, site) not in paper_grade_by_cell:
            planned_missing.append(f"{baseline}/{site}")
    if planned_missing:
        errors.append(
            f"[planned-missing] Phase 1a planned cells absent from paper-grade tier: "
            f"{planned_missing}. After A100 launch completes, edit run_manifest.yaml "
            "to add 6 modes × these (baseline, site) pairs. (A1.21 B-525 P0-8 validator)"
        )

    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--run-manifest",
                    default=str(REPO / "results/phantom_paper/run_manifest.yaml"))
    ap.add_argument("--strict", action="store_true",
                    help="Fail-fast on first error (for CI gates)")
    ap.add_argument("--no-disk", action="store_true",
                    help="Skip disk existence + episode count checks (faster)")
    args = ap.parse_args()

    print(f"[validate_run_manifest] checking {args.run_manifest} ...")
    errors = validate_manifest(
        Path(args.run_manifest),
        strict=args.strict,
        check_disk=not args.no_disk,
    )

    if not errors:
        print("✅ VALIDATION PASSED — all checks (a)-(g) pass")
        return 0

    print(f"❌ VALIDATION FAILED — {len(errors)} error(s):")
    for i, e in enumerate(errors, 1):
        print(f"  {i}. {e}")
    print()
    print("Fix yaml entries to address these, OR if pre-fire (Phase 1a not launched yet) "
          "the `[planned-missing]` errors are expected — gate `make analysis` on this only "
          "AFTER Phase 1a rerun completes.")
    return 2


if __name__ == "__main__":
    sys.exit(main())
