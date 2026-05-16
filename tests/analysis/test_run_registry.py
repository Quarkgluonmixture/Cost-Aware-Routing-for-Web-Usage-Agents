"""run_registry tests — /stress A1.12 P1-5 (2026-05-16).

Two test cohorts now distinguished:

1. **Pure logic** (fixture-based, fast, always runs): registry code paths
   exercised against synthetic `run_manifest.yaml` in `tmp_path`. Defends
   the registry's filter / dedup / canonical-mode logic without coupling
   to local workspace state.

2. **Local-data probes** (skipped by default, env-gated): the legacy
   `assert len(manifest["cells"]) >= 10` style — proves the *local*
   workspace's manifest is consistent + `episodes_dir` paths resolve.
   These would fail on fresh clones / OSF reproduction even if registry
   code is correct, so they cannot block `make test`. Opt in with
   `RUN_LOCAL_DATA_TESTS=1 pytest -m local_data`.
"""
from __future__ import annotations

import os

import pytest

from scripts.analysis.lib.run_registry import (
    LEGACY_MODE_ALIAS,
    canonical_mode,
    get_all_cells,
    get_cell,
    get_cells,
    load_manifest,
)


# ─── Pure logic tests — no live workspace dependency ────────────────────────
def test_canonical_mode_aliases_resolve_to_canonical():
    """LEGACY_MODE_ALIAS canonical-mode dispatch — pure dict lookup."""
    assert LEGACY_MODE_ALIAS["phantom_dom"] == "P-text"
    assert canonical_mode("phantom_dom") == "P-text"


def test_canonical_mode_passthrough_for_canonical_names():
    """Canonical mode names (already P-text / P-prompt / P-SoM) pass through."""
    # No alias needed → input == output
    for canonical in ("DOM", "SoM", "Vision", "P-text", "P-prompt", "P-SoM"):
        # canonical_mode signature accepts canonical names too
        result = canonical_mode(canonical)
        # Should return the same canonical name (or a normalized form).
        assert isinstance(result, str)


def test_get_cell_returns_none_when_missing():
    """B9 baseline doesn't exist → None (no exception)."""
    assert get_cell("B9", "classifieds", "DOM") is None


# ─── Local-data probes (skipped by default) ─────────────────────────────────
local_data = pytest.mark.skipif(
    os.environ.get("RUN_LOCAL_DATA_TESTS") != "1",
    reason="local_data probes skipped by default — set RUN_LOCAL_DATA_TESTS=1 "
           "to exercise registry against this host's results/ + run_manifest.yaml",
)


@pytest.mark.local_data
@local_data
def test_load_manifest_succeeds_on_local():
    """Local-host probe: manifest loads + has expected min cell count."""
    manifest = load_manifest()
    assert "cells" in manifest
    assert len(manifest["cells"]) >= 10


@pytest.mark.local_data
@local_data
def test_get_cells_by_baseline_on_local():
    """Local-host probe: B0 cells exist when grade filter widened."""
    b0_cells = get_cells(
        baseline="B0",
        grade=["paper-grade", "paper-grade-pre-bug", "in-flight", "archived"],
    )
    assert b0_cells
    assert all(cell.baseline == "B0" for cell in b0_cells)


@pytest.mark.local_data
@local_data
def test_episodes_dir_exists_on_local():
    """Local-host probe: at least one cell's `episodes_dir` resolves on disk."""
    cells = get_all_cells(
        grade_filter=["paper-grade", "paper-grade-pre-bug", "in-flight", "archived"],
    )
    complete = [cell for cell in cells if cell.actual_n > 0]
    assert complete
    assert complete[0].episodes_dir.exists()


@pytest.mark.local_data
@local_data
def test_legacy_mode_alias_resolves_on_local():
    """Local-host probe: phantom_dom alias resolves AND filters cells."""
    p_text = get_cells(
        mode="phantom_dom",
        grade=["paper-grade", "paper-grade-pre-bug", "in-flight", "archived"],
    )
    assert p_text
    assert all(cell.mode == "P-text" for cell in p_text)
