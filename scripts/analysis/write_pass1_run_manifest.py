#!/usr/bin/env python3
"""Generate the Pass-1 run whitelist from the canonical run_manifest. — B-1896

`p79/policies/pass1_manifest.discover_runs` restricts a cell to exactly the runs
a manifest lists; absent a manifest it falls back to globbing and merely *warns*
when more than one canonical run matches, then lets
`collect_per_task_outcomes` overwrite newest-wins with no precedence rule.

That fallback is not hypothetical. Extracting all six cells on 2026-07-27 hit it
in two of them:

    B0_reddit        7 runs — two DOM runs, ...R819 (superseded) and ...R11344
                     (the one run_manifest.yaml names)
    B1_classifieds   7 runs — including B1_3mode_classifieds_20260413, the exact
                     stale run 笔记 §367 warned would be picked up

笔记 §367 flagged this before any canonical router training and said the
whitelist had to land first. It never did, so this script lands it — and it
derives the whitelist from `results/phantom_paper/run_manifest.yaml` rather than
hand-listing, so the router's notion of "canonical run" cannot drift away from
the aggregators'.

Usage:
  .venv/bin/python3 scripts/analysis/write_pass1_run_manifest.py           # write
  .venv/bin/python3 scripts/analysis/write_pass1_run_manifest.py --check   # verify only
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

RUN_MANIFEST = REPO / "results/phantom_paper/run_manifest.yaml"
OUT = REPO / "results/phantom_paper/l1_router/pass1_run_manifest.json"


def build() -> dict:
    manifest = yaml.safe_load(RUN_MANIFEST.read_text(encoding="utf-8"))
    pass1: dict[str, list[str]] = defaultdict(list)
    for cell in manifest.get("cells") or []:
        if cell.get("grade") != "paper-grade":
            continue          # archived / in-flight runs are not label substrate
        baseline = cell.get("baseline")
        site = cell.get("site")
        run_dir = cell.get("run_dir")
        if not (baseline and site and run_dir):
            continue
        pass1[f"{baseline}_{site}"].append(run_dir)
    return {
        "_source": "results/phantom_paper/run_manifest.yaml (grade == paper-grade)",
        "_generated_by": "scripts/analysis/write_pass1_run_manifest.py",
        "_why": (
            "B-1896 / 笔记 §367 — without this whitelist, discover_runs globs and "
            "collect_per_task_outcomes silently overwrites newest-wins, folding "
            "superseded and stale runs into the router's oracle labels."
        ),
        "pass1": {k: sorted(v) for k, v in sorted(pass1.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the on-disk manifest matches run_manifest.yaml; exit 1 if not")
    args = ap.parse_args()

    built = build()
    counts = {k: len(v) for k, v in built["pass1"].items()}
    off = {k: n for k, n in counts.items() if n != 6}

    if args.check:
        if not OUT.is_file():
            print(f"✗ manifest missing: {OUT}")
            return 1
        cur = json.loads(OUT.read_text(encoding="utf-8"))
        if cur.get("pass1") != built["pass1"]:
            print("✗ on-disk manifest differs from run_manifest.yaml")
            for k in sorted(set(cur.get("pass1", {})) | set(built["pass1"])):
                a, b = cur.get("pass1", {}).get(k), built["pass1"].get(k)
                if a != b:
                    print(f"    {k}:\n      on-disk: {a}\n      derived: {b}")
            return 1
        print(f"✓ manifest matches run_manifest.yaml ({len(counts)} cells)")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(built, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    for k, v in sorted(built["pass1"].items()):
        print(f"  {k:18s} {len(v)} runs")
    if off:
        print(f"\n⚠️  cells whose run count is not 6: {off}")
        print("    (6 = one per observation mode; anything else means the cell is "
              "partial or carries a duplicate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
