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
    MODE_TO_KEY,  # A1.21 P1-13 (B-498): canonical shared source — was duplicated here
    get_all_cells,
)
from scripts.analysis.lib.canonical_cells import (  # noqa: E402
    SITE_ABBREV,  # A1.21 P0-7 (B-495): canonical shared source
)

LOGGER = logging.getLogger("generate-per-task-sr")


def load_task_outcomes(condition_dir: Path) -> dict[str, dict[str, Any]]:
    """Read all *_summary_v2.json in episodes/ dir. Returns task_id → {success, cost}.

    A1.21 P0-1 fix (B-496, 3-AI overlap A+B+C): `cost_raw = data.get('total_cost_usd')`
    now uses explicit `is None` check (was `or` short-circuit which dropped valid 0.0
    costs because Python `0.0 or x = x`). Affects: B0 proxy GLM-fallback edge cases
    (rare 0.0 cost) + any transient error episode emitting 0.0.
    """
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
        # A1.21 P0-1 fix: explicit `is None` check, NOT `or` short-circuit
        cost_raw = data.get("total_cost_usd")
        if cost_raw is None:
            cost_raw = data.get("total_model_cost_usd")
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


def cost_unit_basis_for(baseline: str) -> str:
    """A1.21 P0-1 cross-baseline cost unit disambiguation.

    B0 (proxy) reports `total_cost_usd` = API margin USD (paid).
    B1 + B2 (local) report `total_cost_usd` = electricity-derived USD
    (avg_total_energy_kwh × electricity_rate, NOT API margin).

    Paper §1 cost-equivalence claim aggregating across baselines mixes API-USD
    + electricity-USD → reviewer R5 attack "cost-DOM ratio 是 token-cost 还是
    wall-cost?" — must stratify by unit basis when pooling.
    """
    if baseline == "B0":
        return "api_usd"
    if baseline in ("B1", "B2"):
        return "electricity_usd_derived"
    return "unknown"


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
    # A1.21 P1-12 (B-497): --grade accepts list (was single string). Appendix-D
    # mixed-grade analysis can pass `--grade paper-grade --grade paper-grade-pre-bug`.
    p.add_argument(
        "--grade",
        action="append",
        default=None,
        help="manifest grade filter (paper-grade / paper-grade-pre-bug / archived / in-flight). "
        "Default = paper-grade. Pass repeatedly for multi-grade analysis. "
        "(A1.21 P1-12 B-497: nargs='append', was single string).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()
    if args.grade is None:
        args.grade = ["paper-grade"]

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    try:
        # A1.21 P0-5 fix (B-493): pass manifest_path to actually use --run-manifest arg.
        # P1-12 (B-497): args.grade is now a list.
        cells = get_all_cells(
            grade_filter=args.grade,
            manifest_path=Path(args.run_manifest),
        )
    except Exception as e:
        LOGGER.error("Failed loading manifest %s: %s", args.run_manifest, e)
        return 2

    if not cells:
        LOGGER.error(
            "No %s cells in manifest %s. "
            "Did paper-grade promotion happen? See B-121 in master_bug_catalog. "
            "Run: scripts/analysis/validate_run_manifest.py --no-disk (A1.21 B-494)",
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
    # A1.21 P0-1 (B-496): cost_unit_basis column for cross-baseline cost-unit transparency
    fieldnames = (
        ["cell_id", "site", "model", "task_id"]
        + [f"sr_{MODE_TO_KEY[m]}" for m in PAPER_MODES]
        + ["cost_dom", "cost_psom", "cost_unit_basis"]
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
            cost_basis = cost_unit_basis_for(baseline)  # A1.21 P0-1 disambiguation
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
                row["cost_unit_basis"] = cost_basis
                writer.writerow(row)
                n_rows += 1

    LOGGER.info("wrote %d rows to %s (%d complete cells, %d incomplete)",
                n_rows, out_path, len(by_cell) - len(incomplete_cells), len(incomplete_cells))
    if incomplete_cells:
        LOGGER.warning("incomplete cells (skipped): %s", incomplete_cells)

    # A1.21 P0-5 (B-493): emit manifest provenance sidecar so canonical full producer
    # can lock manifest_sha256 + csv_sha256 + grade_filter in audit chain.
    import hashlib
    manifest_path = Path(args.run_manifest)
    sidecar_path = out_path.with_suffix(out_path.suffix + ".provenance.json")
    sidecar = {
        "csv_path": str(out_path),
        "csv_sha256": hashlib.sha256(out_path.read_bytes()).hexdigest() if out_path.exists() else None,
        "manifest_path": str(manifest_path),
        "manifest_sha256": (hashlib.sha256(manifest_path.read_bytes()).hexdigest()
                            if manifest_path.exists() else None),
        "grade_filter": args.grade,
        "n_rows": n_rows,
        "n_complete_cells": len(by_cell) - len(incomplete_cells),
        "n_incomplete_cells": len(incomplete_cells),
        "incomplete_cells": [list(c) for c in incomplete_cells],
        "producer": "generate_per_task_sr.py (A1.21 B-493 P0-5 manifest_path wired)",
    }
    sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n")
    LOGGER.info("provenance sidecar → %s", sidecar_path)
    return 0 if n_rows > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
