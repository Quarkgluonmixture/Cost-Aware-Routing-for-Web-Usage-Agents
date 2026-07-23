"""Normalize canonical fire artifacts into the layout mechanistic scripts expect.

WHY: the mechanistic pipeline (curate_mirage_tasks / extract_archive_subset /
run_stage2b / run_stage3 / run_stage4) was written against the 2026-04 archive
layout, where the SoM-annotated screenshot lived at

    <task>/step_NNN/screenshot_annotated.png

Canonical paper-grade fire runs (2026-06+) write it elsewhere, under a per-task
`som/` subdir with a different filename (p79/experiment/som.py:606, B-1828 P0-1
deferred-save refactor):

    <task>/som/step_NNN_som.png

Both files are the same artifact — the bbox-drawn PIL image the SoM model
consumed. Only path + name changed. Rather than fork the read contract across 9
downstream extractors (see B-82 for what that costs), this script materializes
the legacy name as a **relative symlink** next to observation_dom.txt, so every
downstream script keeps its existing `step_dir / "screenshot_annotated.png"`
lookup and reads canonical pixels.

Symlink (not copy): the annotated PNGs are ~200KB each × 4k steps per site.
Relative targets keep the tree movable (rsync -l / tar preserves them).

Usage:
    python3 scripts/mechanistic/normalize_canonical_artifacts.py \
      --artifacts-dir results/mechanistic/_canonical_artifacts/B1_som_classifieds_R31705

    # dry run first:
    python3 scripts/mechanistic/normalize_canonical_artifacts.py \
      --artifacts-dir <dir> --dry-run

Idempotent: re-running relinks existing symlinks and leaves real files alone
(a pre-existing regular screenshot_annotated.png is reported and never
clobbered — that would be a legacy-layout tree, not a canonical one).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("normalize-artifacts")

LEGACY_NAME = "screenshot_annotated.png"
SOM_PNG_RE = re.compile(r"^step_(\d+)_som\.png$")


def normalize_task(task_dir: Path, dry_run: bool) -> dict:
    """Link <task>/som/step_NNN_som.png → <task>/step_NNN/screenshot_annotated.png."""
    stats = {"linked": 0, "relinked": 0, "skipped_real_file": 0, "orphan_som": 0, "no_obs": 0}
    som_dir = task_dir / "som"
    if not som_dir.is_dir():
        return stats

    for som_png in sorted(som_dir.iterdir()):
        m = SOM_PNG_RE.match(som_png.name)
        if not m:
            continue
        step_dir = task_dir / f"step_{int(m.group(1)):03d}"
        if not step_dir.is_dir():
            # An annotated image with no sibling step dir: the step's DOM text
            # never landed, so the pair is unusable downstream regardless.
            stats["orphan_som"] += 1
            continue
        if not (step_dir / "observation_dom.txt").exists():
            stats["no_obs"] += 1
            continue

        target = step_dir / LEGACY_NAME
        # os.path.relpath so the link survives moving/copying the whole tree.
        rel = os.path.relpath(som_png, step_dir)

        if target.is_symlink():
            if os.readlink(target) == rel:
                continue  # already correct
            if not dry_run:
                target.unlink()
                target.symlink_to(rel)
            stats["relinked"] += 1
            continue
        if target.exists():
            # Real file → this is a legacy-layout tree. Never overwrite.
            stats["skipped_real_file"] += 1
            continue

        if not dry_run:
            target.symlink_to(rel)
        stats["linked"] += 1

    return stats


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--artifacts-dir", required=True,
        help="Directory containing <site>_task_<id>/ subdirs (the canonical run's "
             "artifacts/ dir, or an rsync'd copy of it).",
    )
    p.add_argument("--dry-run", action="store_true", help="Report only; write nothing.")
    args = p.parse_args()

    root = Path(args.artifacts_dir)
    if not root.is_dir():
        logger.error(f"artifacts dir not found: {root}")
        return 1

    task_dirs = sorted(d for d in root.iterdir() if d.is_dir() and "_task_" in d.name)
    if not task_dirs:
        logger.error(f"no <site>_task_<id> subdirs under {root}")
        return 1

    total = {"linked": 0, "relinked": 0, "skipped_real_file": 0, "orphan_som": 0, "no_obs": 0}
    tasks_with_pairs = 0
    for task_dir in task_dirs:
        s = normalize_task(task_dir, args.dry_run)
        for k in total:
            total[k] += s[k]
        if s["linked"] or s["relinked"]:
            tasks_with_pairs += 1

    prefix = "[DRY RUN] " if args.dry_run else ""
    logger.info(f"{prefix}tasks scanned: {len(task_dirs)}")
    logger.info(f"{prefix}symlinks created: {total['linked']} (relinked {total['relinked']})")
    if total["skipped_real_file"]:
        logger.warning(
            f"{prefix}{total['skipped_real_file']} steps already had a REAL "
            f"{LEGACY_NAME} — left untouched (legacy-layout tree?)"
        )
    if total["orphan_som"]:
        logger.warning(f"{prefix}{total['orphan_som']} annotated PNGs had no sibling step_NNN/ dir")
    if total["no_obs"]:
        logger.warning(f"{prefix}{total['no_obs']} steps had an annotated PNG but no observation_dom.txt")

    if not args.dry_run:
        manifest = {
            "artifacts_dir": str(root),
            "legacy_name": LEGACY_NAME,
            "source_pattern": "som/step_NNN_som.png",
            "link_type": "relative symlink",
            "tasks_scanned": len(task_dirs),
            "tasks_with_pairs": tasks_with_pairs,
            **total,
        }
        out = root / "normalize_manifest.json"
        out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"manifest → {out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
