#!/usr/bin/env python3
"""Export the small-text evidence substrate for every run in run_manifest.yaml.

Why this exists
---------------
`results/*` is gitignored (`.gitignore:33`). The per-episode records that every
aggregator reads live only on the paper-grade run host (Condenser A100,
`/mnt/scratch/p79_results_active_visualwebarena`). `results/phantom_paper/`
carries `run_manifest.yaml` and nothing else, so a checkout cannot recompute
`per_task_sr.csv` — and therefore cannot re-cut any routing claim — once that
host is gone.

This script copies only the *small text* layer (condition metadata + per-episode
summary JSON). It deliberately does NOT copy screenshots, trajectories or raw
model I/O: the goal is a bundle that can live in git next to the manifest.

Run ON the host that holds the results tree:

    python3 scripts/maintenance/export_evidence_bundle.py \
        --results-root results/visualwebarena/phase1 \
        --results-root results/webarena/phase1 \
        --out /tmp/p79_evidence_bundle

Then rsync the out dir back. Pair it with:

    python3 scripts/analysis/generate_per_task_sr.py \
        --out results/phantom_paper/per_task_sr.csv

`per_task_sr.csv` is the task x mode outcome matrix consumed by
aggregate_fusion_premium / aggregate_label_instability /
aggregate_noise_floor_inventory / aggregate_phase1_full_prereg_decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("export_evidence_bundle")

# The three small-text kinds every downstream aggregator reads.
CONDITION_FILES = ("condition_meta.json", "condition_summary_v2.json")
EPISODE_GLOB = "episodes/*_summary_v2.json"


def load_manifest(path: Path) -> list[dict[str, Any]]:
    """Return manifest entries. yaml is required — a hand-rolled parser here
    would be a second source of truth for the run registry."""
    try:
        import yaml
    except ImportError:
        logger.exception(
            "PyYAML missing. run_manifest.yaml is the canonical run registry; "
            "refusing to glob the results tree instead (that would silently "
            "pick up archived / non-paper-grade run dirs)."
        )
        raise

    with open(path, "r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)

    entries: list[dict[str, Any]] = []
    for section in ("cells", "in_flight", "archived"):
        for entry in doc.get(section) or []:
            entry = dict(entry)
            entry["_section"] = section
            entries.append(entry)
    return entries


def resolve_run_dir(run_dir: str, roots: list[Path]) -> Path | None:
    candidate = Path(run_dir)
    if candidate.is_absolute():
        return candidate if candidate.is_dir() else None
    for root in roots:
        probe = root / run_dir
        if probe.is_dir():
            return probe
    return None


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_entry(
    entry: dict[str, Any], roots: list[Path], out_dir: Path
) -> dict[str, Any]:
    """Copy one manifest entry's small-text layer. Returns a record; never
    swallows a miss — a missing paper-grade cell is a hard failure upstream."""
    run_dir_name = entry.get("run_dir")
    condition_subdir = entry.get("condition_subdir")
    label = f"{entry.get('baseline')}/{entry.get('site')}/{entry.get('mode')}"

    record: dict[str, Any] = {
        "baseline": entry.get("baseline"),
        "site": entry.get("site"),
        "mode": entry.get("mode"),
        "grade": entry.get("grade"),
        "section": entry["_section"],
        "run_dir": run_dir_name,
        "condition_subdir": condition_subdir,
        "files": [],
        "episodes": 0,
        "bytes": 0,
        "status": "ok",
    }

    if not run_dir_name or not condition_subdir:
        record["status"] = "manifest_incomplete"
        logger.error("%s: manifest entry lacks run_dir/condition_subdir", label)
        return record

    resolved = resolve_run_dir(run_dir_name, roots)
    if resolved is None:
        record["status"] = "run_dir_not_found"
        logger.error(
            "%s: run_dir %r not found under any of %s",
            label,
            run_dir_name,
            [str(r) for r in roots],
        )
        return record

    cond_dir = resolved / condition_subdir
    if not cond_dir.is_dir():
        record["status"] = "condition_subdir_not_found"
        logger.error("%s: condition subdir missing: %s", label, cond_dir)
        return record

    dest = out_dir / run_dir_name / condition_subdir
    (dest / "episodes").mkdir(parents=True, exist_ok=True)

    for name in CONDITION_FILES:
        src = cond_dir / name
        if not src.is_file():
            # Loud: a condition without its summary cannot be re-aggregated.
            logger.warning("%s: missing %s", label, src)
            continue
        shutil.copy2(src, dest / name)
        record["files"].append(name)
        record["bytes"] += src.stat().st_size

    episodes = sorted(cond_dir.glob(EPISODE_GLOB))
    for src in episodes:
        shutil.copy2(src, dest / "episodes" / src.name)
        record["bytes"] += src.stat().st_size
    record["episodes"] = len(episodes)

    if not episodes:
        record["status"] = "no_episode_summaries"
        logger.error("%s: zero *_summary_v2.json under %s", label, cond_dir)

    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/phantom_paper/run_manifest.yaml"),
        help="canonical run registry (default: %(default)s)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        dest="results_roots",
        required=True,
        help="root that relative run_dir values resolve under; repeatable",
    )
    parser.add_argument("--out", type=Path, required=True, help="bundle output dir")
    parser.add_argument(
        "--grade",
        action="append",
        dest="grades",
        help="only export these grades (default: paper-grade). Pass --grade all "
        "to take every manifest entry.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr
    )

    grades = args.grades or ["paper-grade"]
    take_all = "all" in grades

    entries = load_manifest(args.manifest)
    selected = [e for e in entries if take_all or e.get("grade") in grades]
    logger.info(
        "manifest %s: %d entries, %d selected (grades=%s)",
        args.manifest,
        len(entries),
        len(selected),
        grades,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    records = [export_entry(e, args.results_roots, args.out) for e in selected]

    total_bytes = sum(r["bytes"] for r in records)
    total_eps = sum(r["episodes"] for r in records)
    failed = [r for r in records if r["status"] != "ok"]

    manifest_out = {
        "source_manifest": str(args.manifest),
        "source_manifest_sha256": sha256_of(args.manifest),
        "results_roots": [str(r) for r in args.results_roots],
        "grades": grades,
        "n_conditions": len(records),
        "n_episode_summaries": total_eps,
        "total_bytes": total_bytes,
        "n_failed": len(failed),
        "conditions": records,
    }
    with open(args.out / "BUNDLE_MANIFEST.json", "w", encoding="utf-8") as fh:
        json.dump(manifest_out, fh, indent=2, ensure_ascii=False)

    print(f"conditions exported : {len(records) - len(failed)}/{len(records)}")
    print(f"episode summaries   : {total_eps}")
    print(f"bundle size         : {total_bytes / 1e6:.1f} MB")
    print(f"bundle manifest     : {args.out / 'BUNDLE_MANIFEST.json'}")

    if failed:
        print("\nFAILED conditions:", file=sys.stderr)
        for r in failed:
            print(
                f"  {r['baseline']}/{r['site']}/{r['mode']}: {r['status']}",
                file=sys.stderr,
            )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
