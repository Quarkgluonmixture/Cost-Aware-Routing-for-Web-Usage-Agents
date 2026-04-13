#!/usr/bin/env python3
"""Delete result files for specific tasks so the runner retry pass can re-run them.

Deletes: summary JSON, steps JSONL, artifacts directory, and digest records for each task.

Examples:
    # Delete tasks 85-131 for classifieds
    python scripts/clear_tasks.py --run-dir results/.../B1_run \
        --condition phase1_dom_router_0 --site classifieds --tasks 85-131

    # Delete specific tasks
    python scripts/clear_tasks.py --run-dir results/.../B1_run \
        --condition phase1_dom_router_0 --site classifieds --tasks 80,95,104

    # Dry run (show what would be deleted)
    python scripts/clear_tasks.py --run-dir results/.../B1_run \
        --condition phase1_dom_router_0 --site classifieds --tasks 85-131 --dry-run

    # Clean orphan artifact dirs (no summary file) across all conditions
    python scripts/clear_tasks.py --run-dir results/.../B1_run --clean-orphan-artifacts

    # Clean orphan artifacts for a specific condition
    python scripts/clear_tasks.py --run-dir results/.../B1_run \
        --condition phase1_som_router_0 --clean-orphan-artifacts
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Set

_EXCLUDED_DIRS = {"analysis", "task_configs", "_vwa"}


def _parse_task_ids(spec: str) -> List[int]:
    """Parse '85-131' or '80,95,104' or '85-90,95,100-104' into sorted list."""
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            ids.update(range(int(lo), int(hi) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def _clean_orphan_artifacts(
    run_dir: Path,
    condition: Optional[str],
    dry_run: bool,
    stale_mins: int = 10,
) -> int:
    """Delete artifact dirs and orphan steps files that have no corresponding summary.

    Items modified within the last `stale_mins` minutes are skipped — they may
    belong to an in-progress episode (runner creates artifacts/steps before writing
    the summary).
    """
    import time as _time
    cutoff = _time.time() - stale_mins * 60

    if condition:
        cond_dirs = [run_dir / condition]
    else:
        cond_dirs = sorted(
            p for p in run_dir.iterdir()
            if p.is_dir() and p.name not in _EXCLUDED_DIRS and not p.suffix
        )

    deleted = 0
    skipped_recent = 0
    for cond_dir in cond_dirs:
        if not cond_dir.is_dir():
            continue
        art_dir = cond_dir / "artifacts"
        ep_dir = cond_dir / "episodes"

        # 1. Orphan artifact directories (no summary)
        if art_dir.exists():
            for artifact in sorted(art_dir.iterdir()):
                if not artifact.is_dir():
                    continue
                if (ep_dir / f"{artifact.name}_summary_v2.json").exists():
                    continue
                if artifact.stat().st_mtime > cutoff:
                    skipped_recent += 1
                    continue
                rel = artifact.relative_to(run_dir)
                if dry_run:
                    print(f"  [dry-run] rm -rf {rel}  (orphan artifact — no summary)")
                else:
                    shutil.rmtree(artifact)
                    print(f"  deleted orphan artifact: {rel}")
                deleted += 1

        # 2. Orphan steps files (steps JSONL without corresponding summary)
        if ep_dir.exists():
            for steps_file in sorted(ep_dir.glob("*_steps_v2.jsonl")):
                # Derive the expected summary filename
                summary = ep_dir / steps_file.name.replace("_steps_v2.jsonl", "_summary_v2.json")
                if summary.exists():
                    continue
                if steps_file.stat().st_mtime > cutoff:
                    skipped_recent += 1
                    continue
                rel = steps_file.relative_to(run_dir)
                if dry_run:
                    print(f"  [dry-run] rm {rel}  (orphan steps — no summary)")
                else:
                    steps_file.unlink()
                    print(f"  deleted orphan steps: {rel}")
                deleted += 1

    if skipped_recent:
        print(f"  (skipped {skipped_recent} recently-modified item(s) — may be in-progress)")
    return deleted


def main() -> int:
    p = argparse.ArgumentParser(description="Delete task results for runner retry")
    p.add_argument("--run-dir", required=True, help="Run directory")
    p.add_argument("--condition", default=None, help="Condition ID (e.g. phase1_dom_router_0)")
    p.add_argument("--site", default=None, help="Site name (e.g. classifieds)")
    p.add_argument("--tasks", default=None, help="Task IDs: '85-131' or '80,95,104' or '85-90,100-104'")
    p.add_argument("--clean-orphan-artifacts", action="store_true",
                    help="Delete artifact dirs that have no corresponding summary file")
    p.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")
    p.add_argument("--force", action="store_true",
                    help="Also delete tasks that may be in-progress (has steps but no summary)")
    args = p.parse_args()

    # Validate: either --tasks or --clean-orphan-artifacts must be provided
    if not args.tasks and not args.clean_orphan_artifacts:
        p.error("one of --tasks or --clean-orphan-artifacts is required")
    if args.tasks and not args.condition:
        p.error("--condition is required when using --tasks")
    if args.tasks and not args.site:
        p.error("--site is required when using --tasks")

    run_dir = Path(args.run_dir).resolve()

    # --- Orphan artifact cleanup mode ---
    if args.clean_orphan_artifacts:
        orphans_deleted = _clean_orphan_artifacts(run_dir, args.condition, args.dry_run)
        action = "would delete" if args.dry_run else "deleted"
        print(f"\nDone: {action} {orphans_deleted} orphan artifact dir(s)")
        if not args.tasks:
            return 0

    # --- Task-level cleanup ---
    if not args.condition:
        p.error("--condition is required when using --tasks")
    cond_dir = run_dir / args.condition
    if not cond_dir.exists():
        print(f"ERROR: condition dir not found: {cond_dir}", file=sys.stderr)
        return 1

    ep_dir = cond_dir / "episodes"
    art_dir = cond_dir / "artifacts"
    task_ids = _parse_task_ids(args.tasks)
    site = args.site

    deleted = 0
    skipped = 0
    in_progress_skipped = 0

    for tid in task_ids:
        prefix = f"{site}_task_{tid}"
        summary_file = ep_dir / f"{prefix}_summary_v2.json"
        steps_file = ep_dir / f"{prefix}_steps_v2.jsonl"
        artifact_dir = art_dir / prefix

        # Safety: skip tasks that may be currently running
        # (has steps JSONL or artifacts but no summary yet)
        if not summary_file.exists() and (steps_file.exists() or artifact_dir.exists()):
            if not args.force:
                print(f"  SKIP {prefix} — in-progress (has steps/artifacts but no summary). Use --force to override")
                in_progress_skipped += 1
                continue

        files = [summary_file, steps_file]
        dirs = [artifact_dir]

        found_any = False
        for f in files:
            if f.exists():
                found_any = True
                if args.dry_run:
                    print(f"  [dry-run] rm {f.relative_to(run_dir)}")
                else:
                    f.unlink()
        for d in dirs:
            if d.exists():
                found_any = True
                if args.dry_run:
                    print(f"  [dry-run] rm -rf {d.relative_to(run_dir)}")
                else:
                    shutil.rmtree(d)

        if found_any:
            deleted += 1
            if not args.dry_run:
                print(f"  deleted {prefix}")
        else:
            skipped += 1

    # --- Clean digest records for deleted tasks ---
    digest_cleaned = 0
    task_id_set: Set[int] = set(task_ids)
    digest_dir = run_dir / "analysis" / "digest"
    if digest_dir.exists():
        for jsonl_file in sorted(digest_dir.glob("digest_*.jsonl")):
            try:
                lines = jsonl_file.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            keep = []
            removed_here = 0
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    keep.append(line)
                    continue
                tid = rec.get("task_id", -1)
                cid = rec.get("condition_id", "")
                if tid in task_id_set and (not cid or cid == args.condition):
                    removed_here += 1
                else:
                    keep.append(line)
            if removed_here:
                if args.dry_run:
                    print(f"  [dry-run] remove {removed_here} records from {jsonl_file.name}")
                else:
                    jsonl_file.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
                digest_cleaned += removed_here

    action = "would delete" if args.dry_run else "deleted"
    parts = [f"{action} {deleted} tasks"]
    if digest_cleaned:
        parts.append(f"{action} {digest_cleaned} digest records")
    if skipped:
        parts.append(f"skipped {skipped} (not found)")
    if in_progress_skipped:
        parts.append(f"skipped {in_progress_skipped} (in-progress)")
    print(f"\nDone: {', '.join(parts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
