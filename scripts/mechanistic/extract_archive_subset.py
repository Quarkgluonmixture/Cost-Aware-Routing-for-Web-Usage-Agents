"""Extract 24 strong + 11 reverse mirage candidate task artifacts to a compact
subset suitable for git commit + A100/Myriad transfer.

Reads `results/mechanistic/curate_mirage_b1_classifieds/candidates.jsonl`
(produced by `curate_mirage_tasks.py`, 笔记 §113), filters by composite +
token_overlap criteria, and copies per-(task, step) artifacts:
- observation_dom.txt
- screenshot_annotated.png

Output: `results/mechanistic/archive_subset_b1_cls/<site>_task_<id>/step_<NNN>/`

Total ~25MB (vs full archive 1.8GB), git-committable for A100 launch.

Usage:
    python3 scripts/mechanistic/extract_archive_subset.py
    # or with custom thresholds:
    python3 scripts/mechanistic/extract_archive_subset.py \
      --strong-min-composite 1.0 --strong-max-overlap 0.5 \
      --reverse-max-composite -1.5

Result manifest: `results/mechanistic/archive_subset_b1_cls/manifest.json`
listing all extracted task IDs + step indices + tier (strong/reverse).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract-subset")

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_artifacts_dir(run_dir: Path) -> Path:
    for child in run_dir.iterdir():
        if child.is_dir() and (child / "artifacts").is_dir():
            return child / "artifacts"
    raise FileNotFoundError(f"No condition subdir with artifacts/ in {run_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--candidates-jsonl",
        default=str(REPO_ROOT / "results/mechanistic/curate_mirage_b1_classifieds/candidates.jsonl"),
    )
    p.add_argument(
        "--archived-run-dir",
        default=str(REPO_ROOT / "results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428"),
    )
    p.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/mechanistic/archive_subset_b1_cls"),
    )
    p.add_argument("--site", default="classifieds")
    p.add_argument(
        "--steps", nargs="+", type=int, default=[2, 5],
        help="Step indices to extract per task (default [2, 5]).",
    )
    p.add_argument(
        "--strong-min-composite", type=float, default=1.0,
        help="Strong tier: composite >= this (paper-grade mirage candidates)",
    )
    p.add_argument(
        "--strong-max-overlap", type=float, default=0.5,
        help="Strong tier: token_overlap < this (real divergence not envelope)",
    )
    p.add_argument(
        "--reverse-max-composite", type=float, default=-1.5,
        help="Reverse tier: composite <= this (paper §5 robustness check)",
    )
    p.add_argument(
        "--artifacts-subdir", default=None,
        help="Override condition subdir name. For multi-mode archived runs "
             "(e.g. B1_3mode_reddit_20260413 has phase1_{dom,som,vision}_router_0), "
             "find_artifacts_dir picks first-iterated which may be wrong condition. "
             "Set explicitly: e.g. --artifacts-subdir phase1_som_router_0.",
    )
    args = p.parse_args()

    candidates_jsonl = Path(args.candidates_jsonl)
    if not candidates_jsonl.exists():
        logger.error(f"candidates.jsonl not found: {candidates_jsonl}")
        logger.error("  Run scripts/mechanistic/curate_mirage_tasks.py first.")
        sys.exit(1)

    archived_dir = Path(args.archived_run_dir)
    if not archived_dir.is_dir():
        logger.error(f"archived run dir not found: {archived_dir}")
        sys.exit(1)
    if args.artifacts_subdir:
        artifacts_dir = archived_dir / args.artifacts_subdir / "artifacts"
        if not artifacts_dir.is_dir():
            logger.error(f"--artifacts-subdir resolved to {artifacts_dir} (does not exist)")
            sys.exit(1)
    else:
        artifacts_dir = find_artifacts_dir(archived_dir)
    logger.info(f"Source artifacts: {artifacts_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output: {output_dir}")

    # 1. Load candidates
    candidates = [json.loads(line) for line in candidates_jsonl.open()]
    logger.info(f"Loaded {len(candidates)} candidates from jsonl")

    # 2. Filter by tier
    strong = [
        c for c in candidates
        if c["composite"] >= args.strong_min_composite
        and c["token_overlap"] < args.strong_max_overlap
    ]
    reverse = [
        c for c in candidates
        if c["composite"] <= args.reverse_max_composite
    ]

    # Order: strong by composite desc, reverse by composite asc (most negative first)
    strong.sort(key=lambda c: c["composite"], reverse=True)
    reverse.sort(key=lambda c: c["composite"])

    logger.info(
        f"Filtered: {len(strong)} strong (composite ≥ {args.strong_min_composite}, "
        f"overlap < {args.strong_max_overlap}), {len(reverse)} reverse "
        f"(composite ≤ {args.reverse_max_composite})"
    )

    # 3. Copy artifacts
    manifest = {
        "config": vars(args),
        "site": args.site,
        "steps": args.steps,
        "strong": [],
        "reverse": [],
        "skipped": [],
    }

    total_bytes = 0

    for tier_name, tier in [("strong", strong), ("reverse", reverse)]:
        for c in tier:
            task_id = c["task_id"]
            entry = {
                "task_id": task_id,
                "intent": c["intent"],
                "composite": c["composite"],
                "token_overlap": c["token_overlap"],
                "src_neg": c["src_neg"],
                "src_aff": c["src_aff"],
                "tgt_neg": c["tgt_neg"],
                "tgt_aff": c["tgt_aff"],
                "source_text": c["source_text"],
                "target_text": c["target_text"],
                "steps_extracted": [],
            }
            task_src = artifacts_dir / f"{args.site}_task_{task_id}"
            task_dst = output_dir / f"{args.site}_task_{task_id}"

            for step_idx in args.steps:
                step_src = task_src / f"step_{step_idx:03d}"
                if not step_src.is_dir():
                    continue
                step_dst = task_dst / f"step_{step_idx:03d}"
                step_dst.mkdir(parents=True, exist_ok=True)

                copied_step_files = []
                for fname in ["observation_dom.txt", "screenshot_annotated.png"]:
                    fsrc = step_src / fname
                    if fsrc.exists():
                        fdst = step_dst / fname
                        shutil.copy2(fsrc, fdst)
                        bytes_copied = fdst.stat().st_size
                        total_bytes += bytes_copied
                        copied_step_files.append(fname)

                if copied_step_files:
                    entry["steps_extracted"].append({
                        "step_idx": step_idx,
                        "files": copied_step_files,
                    })

            if entry["steps_extracted"]:
                manifest[tier_name].append(entry)
            else:
                manifest["skipped"].append({"task_id": task_id, "tier": tier_name, "reason": "no artifact files"})

    # 4. Save manifest
    manifest_file = output_dir / "manifest.json"
    with manifest_file.open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Saved manifest: {manifest_file}")

    # 5. Human-readable README
    readme_lines = [
        f"# Mirage Candidate Subset — {args.site} (B1 Qwen3-VL-4B)",
        "",
        f"Extracted from `results/mechanistic/curate_mirage_b1_classifieds/candidates.jsonl` "
        f"(笔记 §113, commit `cd50c34`). Used as paper-grade mirage dataset for Stage 2B "
        f"curated scale-up + Stage 2C reverse-direction asymmetry confirm on A100.",
        "",
        "## Filter criteria",
        f"- **Strong tier** (paper-grade mirage candidates): composite ≥ {args.strong_min_composite} "
        f"AND token_overlap < {args.strong_max_overlap}",
        f"- **Reverse tier** (asymmetry robustness): composite ≤ {args.reverse_max_composite}",
        "",
        "## Counts",
        f"- Strong: {len(manifest['strong'])} tasks × {len(args.steps)} steps = up to {len(manifest['strong']) * len(args.steps)} (task, step) artifacts",
        f"- Reverse: {len(manifest['reverse'])} tasks × {len(args.steps)} steps",
        f"- Skipped (no artifact): {len(manifest['skipped'])}",
        f"- **Total disk**: {total_bytes / 1e6:.1f} MB",
        "",
        "## Strong candidates (top 24, sorted by composite desc)",
        "",
        "| Rank | task_id | composite | overlap | intent (50 char) |",
        "|---|---|---|---|---|",
    ]
    for i, e in enumerate(manifest["strong"][:24], 1):
        readme_lines.append(
            f"| {i} | {e['task_id']} | {e['composite']:+.2f} | {e['token_overlap']:.2f} | "
            f"{e['intent'][:50]} |"
        )
    readme_lines.append("")
    readme_lines.append("## Reverse candidates (sorted by composite asc)")
    readme_lines.append("")
    readme_lines.append("| Rank | task_id | composite | overlap | intent (50 char) |")
    readme_lines.append("|---|---|---|---|---|")
    for i, e in enumerate(manifest["reverse"], 1):
        readme_lines.append(
            f"| {i} | {e['task_id']} | {e['composite']:+.2f} | {e['token_overlap']:.2f} | "
            f"{e['intent'][:50]} |"
        )
    (output_dir / "README.md").write_text("\n".join(readme_lines))

    logger.info(
        f"\n{'='*60}\n"
        f"Extracted: {len(manifest['strong'])} strong + {len(manifest['reverse'])} reverse tasks\n"
        f"Total disk: {total_bytes / 1e6:.1f} MB\n"
        f"Output: {output_dir}\n"
        f"{'='*60}"
    )


if __name__ == "__main__":
    main()
