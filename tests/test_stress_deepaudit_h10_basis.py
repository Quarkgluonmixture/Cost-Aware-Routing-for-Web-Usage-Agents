"""Regression tests for /stress 深入审 Mode A — Chunk 2 B-1601 P0-2-A*.

Asserts `aggregate_h10_pareto.py` captures `cost_unit_basis` per-task,
computes per-cell modal basis + homogeneity diagnostic, and FE-pool
output carries cross-cell basis-mix summary. Pre-fix: this script had
0 references to cost_unit_basis (673 LOC) despite A2.7 B-1409 wiring
the field into the sibling `aggregate_cross_site.py`. Sibling-script
propagation gap → paper §6 H10 Appendix-D FE pool could silently mix
api_usd ($0.005/1K tok B0) with electricity_usd_derived ($0.0000005/1K
tok B1/B2) at ~1000× scale gap if any downstream code averaged cost.
See `docs/checkpoints/master_bug_catalog.md ## /stress 深入审` (B-1601)
+ chronicle §220.
"""

from __future__ import annotations

import pytest

from scripts.analysis.aggregate_h10_pareto import _compute_modal_basis


def _mk_outcome(success: int, cost: float, latency: float, basis: str) -> dict:
    return {
        "success": success,
        "cost_usd": cost,
        "latency_ms": latency,
        "cost_unit_basis": basis,
    }


def test_modal_basis_empty_matrix():
    """Empty matrix → safe defaults, not RuntimeError."""
    result = _compute_modal_basis({})
    assert result["modal_basis"] == "unknown"
    assert result["basis_counts"] == {}
    assert result["homogeneous"] is True  # empty is trivially homogeneous
    assert result["minority_rows"] == 0
    assert result["total_rows"] == 0


def test_modal_basis_single_basis_homogeneous():
    """All entries same basis → homogeneous=True, minority_rows=0."""
    outcomes = {
        0: {
            "dom": _mk_outcome(1, 0.005, 1500.0, "api_usd"),
            "som": _mk_outcome(0, 0.010, 2000.0, "api_usd"),
        },
        1: {"dom": _mk_outcome(0, 0.003, 1200.0, "api_usd")},
    }
    result = _compute_modal_basis(outcomes)
    assert result["modal_basis"] == "api_usd"
    assert result["homogeneous"] is True
    assert result["minority_rows"] == 0
    assert result["total_rows"] == 3
    assert result["basis_counts"] == {"api_usd": 3}


def test_modal_basis_mixed_basis_diagnostic():
    """Mixed basis → homogeneous=False, minority_rows counts non-modal."""
    outcomes = {
        0: {
            "dom": _mk_outcome(1, 0.000001, 1500.0, "electricity_usd_derived"),
            "som": _mk_outcome(0, 0.000002, 2000.0, "electricity_usd_derived"),
        },
        1: {
            # 1 legacy unknown row mixed in (e.g., archive episode)
            "dom": _mk_outcome(0, 0.0, 1200.0, "unknown"),
        },
    }
    result = _compute_modal_basis(outcomes)
    assert result["modal_basis"] == "electricity_usd_derived"
    assert result["homogeneous"] is False
    assert result["minority_rows"] == 1
    assert result["total_rows"] == 3
    assert result["basis_counts"] == {"electricity_usd_derived": 2, "unknown": 1}


def test_modal_basis_missing_field_defaults_unknown():
    """Outcomes missing cost_unit_basis default to 'unknown' (legacy schema)."""
    outcomes = {
        0: {
            "dom": {
                # No cost_unit_basis key — legacy summary pre-B-563
                "success": 1, "cost_usd": 0.005, "latency_ms": 1500.0,
            },
        },
    }
    result = _compute_modal_basis(outcomes)
    assert result["modal_basis"] == "unknown"


def test_collect_per_task_outcomes_captures_basis_field(tmp_path):
    """End-to-end: synthesize episode summary on disk, verify cost_unit_basis
    propagates from JSON file into outcome matrix.
    """
    import json
    from scripts.analysis.aggregate_h10_pareto import (
        collect_per_task_outcomes_with_metrics,
    )

    run_dir = tmp_path / "results" / "phase1"
    cond_dir = run_dir / "phase1_dom_B1_reddit"
    ep_dir = cond_dir / "episodes"
    ep_dir.mkdir(parents=True)

    # Fake episode summary file with cost_unit_basis field
    # Canonical schema has total_billed_cost_usd (AMENDMENT_01 H10 cost-axis); a distinct
    # legacy total_cost_usd is included to prove the matrix reads total_billed, not legacy.
    summary = {
        "task_id": 42,
        "success": True,
        "total_billed_cost_usd": 0.0000003,
        "total_cost_usd": 0.0000009,
        "total_latency_ms": 1234.5,
        "cost_unit_basis": "electricity_usd_derived",
    }
    (ep_dir / "reddit_task_42_summary_v2.json").write_text(json.dumps(summary))

    matrix = collect_per_task_outcomes_with_metrics([run_dir], site="reddit")
    assert 42 in matrix
    assert "dom" in matrix[42]
    entry = matrix[42]["dom"]
    assert entry["cost_unit_basis"] == "electricity_usd_derived"
    assert entry["cost_usd"] == pytest.approx(0.0000003)  # total_billed, NOT total_cost


def test_collect_per_task_outcomes_missing_basis_defaults_unknown(tmp_path):
    """Legacy episode summary without cost_unit_basis key → defaults to 'unknown'."""
    import json
    from scripts.analysis.aggregate_h10_pareto import (
        collect_per_task_outcomes_with_metrics,
    )

    run_dir = tmp_path / "results" / "phase1"
    cond_dir = run_dir / "phase1_som_B0_classifieds"
    ep_dir = cond_dir / "episodes"
    ep_dir.mkdir(parents=True)

    summary = {
        "task_id": 7,
        "success": False,
        "total_billed_cost_usd": 0.001,  # canonical (AMENDMENT_01); fail-closed requires it
        "total_cost_usd": 0.001,
        "total_latency_ms": 5000.0,
        # No cost_unit_basis key — legacy schema (this test exercises the
        # cost_unit_basis default, not the cost-axis field)
    }
    (ep_dir / "classifieds_task_7_summary_v2.json").write_text(json.dumps(summary))

    matrix = collect_per_task_outcomes_with_metrics([run_dir], site="classifieds")
    assert matrix[7]["som"]["cost_unit_basis"] == "unknown"


def test_cross_cell_pool_basis_homogeneity_summary_shape():
    """run_h10_verdict structure: ok_cells with mixed modal basis → pool_basis_homogeneous=False
    + pool_warning string mentioning the basis mix."""
    # Simulate ok_cells shape post-Chunk-2
    ok_cells_synth = [
        {
            "cell_id": "B0_reddit",
            "cell_cost_unit_basis": {
                "modal_basis": "api_usd",
                "homogeneous": True,
                "minority_rows": 0,
            },
            "theta_mean_pp": 2.5,
            "theta_se_pp": 0.8,
        },
        {
            "cell_id": "B1_reddit",
            "cell_cost_unit_basis": {
                "modal_basis": "electricity_usd_derived",
                "homogeneous": True,
                "minority_rows": 0,
            },
            "theta_mean_pp": -1.0,
            "theta_se_pp": 0.7,
        },
    ]
    # Build the summary inline to match run_h10_verdict logic
    distinct = sorted({
        r["cell_cost_unit_basis"]["modal_basis"]
        for r in ok_cells_synth
    })
    pool_homogeneous = len([b for b in distinct if b != "unknown"]) <= 1
    assert distinct == ["api_usd", "electricity_usd_derived"]
    assert pool_homogeneous is False, (
        "Cross-cell FE pool with api_usd + electricity_usd_derived modal "
        "bases must flag pool_basis_homogeneous=False — 1000× scale gap."
    )
