#!/usr/bin/env python3
"""Tier-1 /diag scan of BOTH arms of every registered rerun pair, at the current ruleset.

Why (2026-10-08). The 58 per-condition digests cover the canonical run of each condition only;
of the 24 registered rerun pairs (`CLEAN_PAIRS`), almost no arm b has a scan. Any question of
the form "does the same task fail the same way when the condition is rerun" therefore had only
the coarse reason buckets (`rerun_flip_failure_anatomy`) to compare. This scans arm a AND arm b
with one `RULESET_VERSION`, so a symptom difference between the arms cannot be a ruleset or
run-identity difference (arm a is rescanned rather than borrowed from `v11_*`: the canonical
scans resolved cls runs by glob, B-1927).

Zero model calls: `diag_pattern_match.py` is rule matching over step files.

Output: results/diag_scans/v11_rerun_pairs/<label>/{a,b}.json   (results/diag_scans -> E:)
Usage:  python scripts/analysis/rerun_pairs_diag_scan.py [--workers 6] [--only B0.cls.dom ...]
Fails loud: any arm that does not scan, or a ruleset mismatch across arms, exits non-zero.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.analysis.lib.replicate_pairs import clean_pairs  # noqa: E402

SCANNER = REPO / "scripts/analysis/diag_pattern_match.py"
OUT_ROOT = REPO / "results/diag_scans/v11_rerun_pairs"


def scan_arm(label: str, side: str, cond_dir: Path) -> tuple[str, str, str | None, str]:
    out = OUT_ROOT / label / f"{side}.json"
    r = subprocess.run(
        [sys.executable, str(SCANNER), "--run-dir", str(cond_dir.parent),
         "--condition", cond_dir.name, "--output", str(out)],
        capture_output=True, cwd=str(REPO),
    )
    err = r.stderr.decode("utf-8", errors="replace")
    if r.returncode != 0 or not out.exists():
        return label, side, None, f"rc={r.returncode} {err.strip()[-300:]}"
    d = json.loads(out.read_text(encoding="utf-8"))
    return label, side, d.get("ruleset_version"), f"n={d['total_episodes']} hits={d['episodes_with_hits']}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()

    jobs = []
    for label, a_dir, b_dir in clean_pairs():
        if args.only and label not in args.only:
            continue
        for side, d in (("a", a_dir), ("b", b_dir)):
            if not d.is_dir():
                print(f"✗ {label}/{side}: condition dir missing {d}")
                return 1
            jobs.append((label, side, d))

    print(f"scanning {len(jobs)} arms → {OUT_ROOT}")
    versions, failed = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for label, side, ver, msg in ex.map(lambda j: scan_arm(*j), jobs):
            mark = "✓" if ver else "✗"
            print(f"  {mark} {label:18s} {side}  {msg}")
            if ver:
                versions.setdefault(ver, []).append(f"{label}/{side}")
            else:
                failed.append(f"{label}/{side}")

    print(f"\nruleset versions: { {v: len(k) for v, k in versions.items()} }")
    if failed:
        print(f"✗ {len(failed)} arms failed: {failed}")
        return 1
    if len(versions) != 1:
        print("✗ ruleset mismatch across arms — pairwise comparison forbidden")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
