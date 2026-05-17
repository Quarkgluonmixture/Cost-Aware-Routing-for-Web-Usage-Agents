"""Canonical Phase 1a cell enumeration — single source of truth.

A1.21 P0-7 fix (B-495, /stress Claude unique OOB 2026-05-17): pre-fix Phase 1a
"6 cells" was scattered across 3 places — `preregistration_decision_test.PHASE_1A_CELLS`
hardcoded (always 6) / `aggregate_phantom_lift.CELLS` module-import frozen via
`_build_cells(_GRADE_LIST)` / `lib/run_registry.get_*` live registry call. Different
process state → different cell count, no cross-validation, no fail-loud.

This module is THE canonical source. All consumers should:
  - Import `PHASE_1A_PLANNED_CELLS` for the static planned cell list (always 6)
  - Call `get_phase1a_actual_cells(grade_filter, manifest_path)` for live registry view
  - Use `assert_cells_match_planned(loaded_cells)` to fail-loud on Phase 1a partial-data

Tied to:
- preregistration.md §1 Phase 1a scope (6 cells = 2 sites × 3 baselines)
- aggregate_phase1_full_prereg_decision.py (canonical paper §1 producer)
- generate_per_task_sr.py (bridge CSV producer)
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

from scripts.analysis.lib.run_registry import CellSpec, get_all_cells

# Phase 1a planned scope (2 sites × 3 baselines = 6 statistical cells).
# This is the prereg-locked design, NOT a configurable enum — changing requires
# preregistration amendment + advisor sync + Git/OSF re-witness.
PHASE_1A_PLANNED_CELLS: list[tuple[str, str]] = [
    ("classifieds", "B0"),
    ("classifieds", "B1"),
    ("classifieds", "B2"),  # Gemma3-VL (added 2026-05-14 advisor lock)
    ("reddit", "B0"),
    ("reddit", "B1"),
    ("reddit", "B2"),
]

# Cell-id encoding for CSV/JSON `cell_id` field (matches generate_per_task_sr SITE_ABBREV).
SITE_ABBREV: dict[str, str] = {"classifieds": "cls", "reddit": "red", "shopping": "shop"}


def cell_id_for(site: str, baseline: str) -> str:
    """Canonical `cell_id` string for (site, baseline) — e.g., 'cls_B0'."""
    return f"{SITE_ABBREV[site]}_{baseline}"


def planned_cell_ids() -> list[str]:
    """Phase 1a planned cell_ids in canonical sort order."""
    return [cell_id_for(s, b) for (s, b) in PHASE_1A_PLANNED_CELLS]


def get_phase1a_actual_cells(
    grade_filter: list[str] | None = None,
    manifest_path: Path | None = None,
    *,
    strict_paper_grade: bool = False,
) -> list[CellSpec]:
    """Live registry view of Phase 1a cells matching the planned (site, baseline) set.

    Returns CellSpec objects (one per mode per (site, baseline) in PHASE_1A_PLANNED_CELLS).
    For Phase 1a complete state, returns 36 CellSpec (6 cells × 6 modes).
    For partial state (e.g., B2 not launched), returns subset.

    A1.21 P0-5 (B-493): `manifest_path` propagates to registry.
    A1.21 P0-8 (B-492): `strict_paper_grade=True` raises on actual_n==0.
    """
    planned_keys = set(PHASE_1A_PLANNED_CELLS)
    cells = get_all_cells(grade_filter=grade_filter, manifest_path=manifest_path,
                          strict_paper_grade=strict_paper_grade)
    return [c for c in cells if (c.site, c.baseline) in planned_keys]


def assert_cells_match_planned(
    loaded_cell_ids: Iterable[str],
    *,
    require_complete: bool = True,
) -> None:
    """Fail-loud if loaded cell set doesn't match Phase 1a planned 6-cell scope.

    Args:
        loaded_cell_ids: e.g., from CSV `cell_id` column unique values
        require_complete: if True, missing planned cells raise; if False, only
            extra/unknown cells raise (partial-data tolerated for pre-fire smoke)

    A1.21 P0-7 fix: enforces cross-script consistency. Use in
    `preregistration_decision_test.main()` after CSV load, in canonical full
    producer after registry load, etc.
    """
    loaded = set(loaded_cell_ids)
    planned = set(planned_cell_ids())
    extra = loaded - planned
    missing = planned - loaded
    if extra:
        raise ValueError(
            f"Cell scope mismatch: loaded cells {sorted(extra)} are NOT in Phase 1a "
            f"planned set {sorted(planned)}. Did you load a different scope's CSV? "
            "(A1.21 P0-7 B-495)"
        )
    if require_complete and missing:
        raise ValueError(
            f"Phase 1a partial scope: missing cells {sorted(missing)} from planned "
            f"set {sorted(planned)}. Either run scripts/analysis/generate_per_task_sr.py "
            "to refresh CSV from latest registry OR pass require_complete=False to "
            "tolerate pre-fire partial state. (A1.21 P0-7 B-495)"
        )
