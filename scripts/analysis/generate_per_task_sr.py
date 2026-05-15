#!/usr/bin/env python3
r"""Generate per-task SR + cost wide-format CSV for preregistration_decision_test.py.

**B-122 fix (2026-05-15, codex Mode B P0-4)**:
Launcher (`queue_phase1_paper_grade.sh:394`) commanded users to feed
`preregistration_decision_test.py` with
`--per-task-csv results/phantom_paper/per_task_sr.csv` — but no script
produced this file. Aggregators output `phantom_lift.csv` (different schema).
Reviewers / users following the launcher hit `FileNotFoundError` and decision
test could not run → paper hero verdict uncomputable.

**Fix**: this script. Reads run_manifest.yaml + per-task summary JSONs across
all paper-grade cells, pivots to wide format (one row per (cell_id, task_id),
columns = `sr_<mode>` for 6 modes + `cost_dom` + `cost_psom`).

Usage (post-rerun, after manifest paper-grade promotion):
    python3 scripts/analysis/generate_per_task_sr.py \
        --run-manifest results/phantom_paper/run_manifest.yaml \
        --out results/phantom_paper/per_task_sr.csv

Then feed the CSV into the decision test:
    python3 scripts/analysis/preregistration_decision_test.py \
        --per-task-csv results/phantom_paper/per_task_sr.csv \
        --primary-gate drop_one_pooled_meta_superiority \
        --TOST-delta-pp 1.0 --H1-magnitude-pp 1.0 \
        --transparency-K_h1 4 --transparency-K_h3 4 \
        --out results/phantom_paper/preregistration_test_results.json

Output schema (per `preregistration_decision_test.py` docstring):
    cell_id,site,model,task_id,sr_dom,sr_som,sr_vision,sr_ptext,sr_pprompt,sr_psom,cost_dom,cost_psom
    cls_B0,classifieds,B0,task_0001,0.0,1.0,0.0,1.0,0.0,1.0,0.043,0.044
    ...

Tied to:
- preregistration.md §4 row "Cell inclusion (Phase 1a main)" (B0+B1+B2, 36 cond / 6 cells)
- run_manifest.yaml (paper-grade promotion gates this script's input)
- 笔记 §143 (this fix batch chronicle)
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.lib.run_registry import (  # noqa: E402
    PAPER_MODES,
    PHASE1_ROOT,
    get_all_cells,
)

LOGGER = logging.getLogger("generate-per-task-sr")

# Map paper mode name → CSV column suffix
MODE_TO_KEY = {
    "DOM": "dom",
    "SoM": "som",
    "Vision": "vision",
    "P-text": "ptext",
    "P-prompt": "pprompt",
    "P-SoM": "psom",
}

SITE_ABBREV = {"classifieds": "cls", "reddit": "red", "shopping": "shop"}


def load_task_outcomes(condition_dir: Path) -> dict[str, dict[str, Any]]:
    """Read all *_summary_v2.json in episodes/ dir. Returns task_id → {success, cost}."""
    outcomes: dict[str, dict[str, Any]] = {}
    episodes_dir = condition_dir / "episodes"
    if not episodes_dir.is_dir():
        LOGGER.warning("no episodes dir under %s", condition_dir)
        return outcomes
    for summary_path in episodes_dir.glob("*_summary_v2.json"):
        try:
            with summary_path.open() as f:
                data = json.load(f)
        except Exception as e:
            LOGGER.warning("skip %s: %s", summary_path, e)
            continue
        task_id = data.get("task_id")
        if task_id is None:
            continue
        success_raw = data.get("success")
        cost_raw = data.get("total_cost_usd") or data.get("total_model_cost_usd")
        try:
            success = float(success_raw) if success_raw is not None else None
        except (TypeError, ValueError):
            success = None
        try:
            cost = float(cost_raw) if cost_raw is not None else None
        except (TypeError, ValueError):
            cost = None
        outcomes[str(task_id)] = {"success": success, "cost": cost}
    return outcomes


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument(
        "--run-manifest",
        default=str(REPO_ROOT / "results/phantom_paper/run_manifest.yaml"),
        help="run_manifest.yaml path",
    )
    p.add_argument(
        "--out",
        default=str(REPO_ROOT / "results/phantom_paper/per_task_sr.csv"),
        help="output per_task_sr.csv path",
    )
    p.add_argument(
        "--grade",
        default="paper-grade",
        help="manifest grade filter (paper-grade / paper-grade-pre-bug / archived). "
        "Default = paper-grade. Use paper-grade-pre-bug for Appendix-D legacy check.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        cells = get_all_cells(grade_filter=[args.grade])
    except Exception as e:
        LOGGER.error("Failed loading manifest %s: %s", args.run_manifest, e)
        return 2

    if not cells:
        LOGGER.error(
            "No %s cells in manifest %s. "
            "Did paper-grade promotion happen? See B-121 in master_bug_catalog.",
            args.grade,
            args.run_manifest,
        )
        return 2

    # Group cells by (site, baseline) → dict[mode → outcomes]
    by_cell: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]] = {}
    for cs in cells:
        cond_dir = PHASE1_ROOT / cs.run_dir / cs.condition_subdir
        outcomes = load_task_outcomes(cond_dir)
        key = (cs.site, cs.baseline)
        by_cell.setdefault(key, {})[cs.mode] = outcomes
        LOGGER.info(
            "loaded %3d tasks: %s %s %s (%s)",
            len(outcomes),
            cs.baseline,
            cs.site,
            cs.mode,
            cs.run_dir,
        )

    if not by_cell:
        LOGGER.error("No cells loaded any outcomes.")
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        ["cell_id", "site", "model", "task_id"]
        + [f"sr_{MODE_TO_KEY[m]}" for m in PAPER_MODES]
        + ["cost_dom", "cost_psom"]
    )

    n_rows = 0
    incomplete_cells: list[tuple[str, str]] = []
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for (site, baseline), mode_data in sorted(by_cell.items()):
            present_modes = sorted(mode_data.keys())
            missing = [m for m in PAPER_MODES if m not in mode_data]
            if missing:
                LOGGER.warning(
                    "cell %s/%s missing modes %s — skipped (need all 6 for paired analysis)",
                    site,
                    baseline,
                    missing,
                )
                incomplete_cells.append((site, baseline))
                continue
            cell_id = f"{SITE_ABBREV[site]}_{baseline}"
            mode_task_sets = [set(d.keys()) for d in mode_data.values()]
            common_tasks = sorted(set.intersection(*mode_task_sets))
            LOGGER.info(
                "cell %s: %d common tasks across %d modes",
                cell_id,
                len(common_tasks),
                len(present_modes),
            )
            for task_id in common_tasks:
                row: dict[str, Any] = {
                    "cell_id": cell_id,
                    "site": site,
                    "model": baseline,
                    "task_id": task_id,
                }
                for m in PAPER_MODES:
                    row[f"sr_{MODE_TO_KEY[m]}"] = mode_data[m][task_id]["success"]
                row["cost_dom"] = mode_data["DOM"][task_id]["cost"]
                row["cost_psom"] = mode_data["P-SoM"][task_id]["cost"]
                writer.writerow(row)
                n_rows += 1

    LOGGER.info("wrote %d rows to %s (%d complete cells, %d incomplete)",
                n_rows, out_path, len(by_cell) - len(incomplete_cells), len(incomplete_cells))
    if incomplete_cells:
        LOGGER.warning("incomplete cells (skipped): %s", incomplete_cells)
    return 0 if n_rows > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
