from scripts.analysis.lib.run_registry import (
    LEGACY_MODE_ALIAS,
    canonical_mode,
    get_all_cells,
    get_cell,
    get_cells,
    load_manifest,
)


def test_load_manifest_succeeds():
    manifest = load_manifest()
    assert "cells" in manifest
    assert len(manifest["cells"]) >= 10


def test_get_cells_by_baseline():
    # F01 audit 2026-05-09: DEFAULT_GRADE_FILTER restricts to
    # ['paper-grade'] only. Pre-rerun manifest cells are graded
    # `archived` (legacy phase-A pilot data). Pass all grades to ensure
    # the registry round-trip works regardless of paper-grade promotion.
    b0_cells = get_cells(
        baseline="B0",
        grade=["paper-grade", "paper-grade-pre-bug", "in-flight", "archived"],
    )
    assert b0_cells
    assert all(cell.baseline == "B0" for cell in b0_cells)


def test_get_cell_returns_none_when_missing():
    assert get_cell("B9", "classifieds", "DOM") is None


def test_episodes_dir_exists():
    # F01 audit 2026-05-09: pre-rerun manifest has only `archived` cells.
    cells = get_all_cells(
        grade_filter=["paper-grade", "paper-grade-pre-bug", "in-flight", "archived"],
    )
    complete = [cell for cell in cells if cell.actual_n > 0]
    assert complete
    assert complete[0].episodes_dir.exists()


def test_legacy_mode_alias_resolves():
    assert LEGACY_MODE_ALIAS["phantom_dom"] == "P-text"
    assert canonical_mode("phantom_dom") == "P-text"
    # F01 audit 2026-05-09: pre-rerun manifest has only `archived` cells.
    p_text = get_cells(
        mode="phantom_dom",
        grade=["paper-grade", "paper-grade-pre-bug", "in-flight", "archived"],
    )
    assert p_text
    assert all(cell.mode == "P-text" for cell in p_text)
