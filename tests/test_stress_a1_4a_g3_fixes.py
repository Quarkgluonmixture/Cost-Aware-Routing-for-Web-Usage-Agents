"""Invariant tests for /stress A1.4a v8 Commit G3 (B-168 + B-169) — JSONL /
resume hardening: partial-step crash recovery + resume identity check.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from p79.experiment.runner.main import ExperimentRunner


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# B-168 — Partial-step crash recovery: _aggregate_partial_steps
# ---------------------------------------------------------------------------


def test_aggregate_partial_steps_empty_returns_zero_summary():
    """Empty step list (no JSONL on disk before crash) → zero-step summary
    (preserves pre-B-168 behavior as defensive fallback)."""
    agg = ExperimentRunner._aggregate_partial_steps([])
    assert agg["steps"] == 0
    assert agg["total_tokens"] == 0
    assert agg["total_cost_usd"] == 0.0
    assert agg["total_latency_ms"] == 0.0


def test_aggregate_partial_steps_sums_tokens_and_costs():
    """B-168 critical: 12 partial JSONL rows on disk → summary aggregates
    them, not zero-step erasure. Paper-grade evidence layer no longer
    splits between JSONL truth and summary truth on mid-episode crash."""
    partial = [
        {
            "step_idx": i,
            "tokens": {"total": 100 + i},
            "cost_usd": {"model": 0.01, "router_overhead": 0.001, "obs_prepare": 0.0},
            "latency_ms": {"total": 200.0 + i * 10},
            "action_success": True,
            "page_changed": True,
            "action": {"action_type": "click"},
            "retry_count": 0,
            "router": {"overhead_ms": {"router_decision_ms": 0.5}},
        }
        for i in range(12)
    ]
    agg = ExperimentRunner._aggregate_partial_steps(partial)
    assert agg["steps"] == 12
    assert agg["total_tokens"] == sum(100 + i for i in range(12))
    # 12 * (0.01 model + 0.001 router_overhead) = 0.132
    assert abs(agg["total_cost_usd"] - 0.132) < 1e-6
    assert agg["total_model_cost_usd"] == 12 * 0.01
    assert agg["total_router_overhead_cost_usd"] == pytest.approx(12 * 0.001)
    assert agg["total_latency_ms"] == sum(200.0 + i * 10 for i in range(12))


def test_aggregate_partial_steps_handles_missing_fields():
    """B-168 robustness: rows from older schema versions / corrupt-but-
    parsed JSONL may lack fields. Helper must not KeyError, treats missing
    as zero. Otherwise the catch-all except path would mask a recoverable
    crash with its own crash."""
    partial = [
        {"step_idx": 0},  # bare-minimum row
        {"step_idx": 1, "tokens": {"total": 50}},  # partial fields
    ]
    agg = ExperimentRunner._aggregate_partial_steps(partial)
    assert agg["steps"] == 2
    assert agg["total_tokens"] == 50  # only second row had tokens
    assert agg["total_cost_usd"] == 0.0  # neither had cost


def test_aggregate_partial_steps_no_op_and_unchanged_rates():
    """no_op_rate / page_unchanged_rate computed correctly from partial
    rows. Finish/stop steps excluded from unchanged count (consistency
    with main episode aggregation)."""
    partial = [
        {"action_success": False, "page_changed": False, "action": {"action_type": "click"}},
        {"action_success": True, "page_changed": True, "action": {"action_type": "click"}},
        {"action_success": False, "page_changed": False, "action": {"action_type": "finish"}},  # excluded from unchanged
    ]
    agg = ExperimentRunner._aggregate_partial_steps(partial)
    assert agg["steps"] == 3
    # 2 of 3 had action_success=False → no_op_rate = 2/3
    assert abs(agg["no_op_rate"] - 2.0 / 3.0) < 1e-6
    # 1 of 3 had !page_changed AND non-finish → 1/3
    assert abs(agg["page_unchanged_rate"] - 1.0 / 3.0) < 1e-6


# ---------------------------------------------------------------------------
# B-169 — Resume identity check: _validate_resume_identity
# ---------------------------------------------------------------------------


def test_validate_resume_identity_returns_none_on_match():
    """Identity tuple matches → resume gate accepts (returns None)."""
    expected = {
        "schema_version": "2.0",
        "run_id": "run_42",
        "condition_id": "phase1_dom_router_0",
        "seed": 42,
        "benchmark_site": "classifieds",
        "task_id": 123,
    }
    loaded = dict(expected)  # exact match
    result = ExperimentRunner._validate_resume_identity(loaded, expected)
    assert result is None


def test_validate_resume_identity_catches_run_id_drift():
    """Pre-B-169 critical scenario: output_root reused, run_id changed
    (e.g. yesterday's run vs today's). Loaded summary belongs to old run.
    Resume gate must detect via run_id mismatch."""
    expected = {
        "schema_version": "2.0",
        "run_id": "run_today",
        "condition_id": "phase1_dom_router_0",
        "seed": 42,
        "benchmark_site": "classifieds",
        "task_id": 123,
    }
    loaded = dict(expected)
    loaded["run_id"] = "run_yesterday"  # stale
    result = ExperimentRunner._validate_resume_identity(loaded, expected)
    assert result is not None
    assert "run_id" in result
    assert result["run_id"] == ("run_yesterday", "run_today")


def test_validate_resume_identity_catches_seed_drift():
    """Multi-seed run reused output: seed=42 summary loaded when expecting
    seed=43 (e.g. manual cp from sibling directory)."""
    expected = {
        "schema_version": "2.0",
        "run_id": "run_42",
        "condition_id": "phase1_dom_router_0",
        "seed": 43,
        "benchmark_site": "classifieds",
        "task_id": 123,
    }
    loaded = dict(expected)
    loaded["seed"] = 42  # mismatched seed
    result = ExperimentRunner._validate_resume_identity(loaded, expected)
    assert result is not None
    assert "seed" in result


def test_validate_resume_identity_catches_site_drift():
    """Worst-case identity bug: same task_id present on multiple sites
    (cls task 5 vs reddit task 5). Without site check, cross-site
    contamination silent."""
    expected = {
        "schema_version": "2.0", "run_id": "run_42",
        "condition_id": "phase1_dom_router_0", "seed": 42,
        "benchmark_site": "classifieds", "task_id": 5,
    }
    loaded = dict(expected)
    loaded["benchmark_site"] = "reddit"
    result = ExperimentRunner._validate_resume_identity(loaded, expected)
    assert result is not None
    assert "benchmark_site" in result


def test_validate_resume_identity_catches_schema_version_drift():
    """Older schema version (e.g. v1.x or future v2.1 with retired fields)
    must trigger quarantine — analysis may depend on schema-specific
    fields."""
    expected = {
        "schema_version": "2.0", "run_id": "run_42",
        "condition_id": "phase1_dom_router_0", "seed": 42,
        "benchmark_site": "classifieds", "task_id": 5,
    }
    loaded = dict(expected)
    loaded["schema_version"] = "1.0"
    result = ExperimentRunner._validate_resume_identity(loaded, expected)
    assert result is not None
    assert "schema_version" in result


def test_validate_resume_identity_collects_all_mismatches():
    """Multiple drifted fields → all reported (not just first). Useful
    for diagnosis when an entire run dir got mv'd."""
    expected = {
        "schema_version": "2.0", "run_id": "run_42",
        "condition_id": "phase1_dom_router_0", "seed": 42,
        "benchmark_site": "classifieds", "task_id": 5,
    }
    loaded = {
        "schema_version": "1.0", "run_id": "run_yesterday",
        "condition_id": "phase1_som_router_0", "seed": 43,
        "benchmark_site": "reddit", "task_id": 99,
    }
    result = ExperimentRunner._validate_resume_identity(loaded, expected)
    assert result is not None
    # All 6 fields should mismatch
    assert len(result) == 6


def test_validate_resume_identity_missing_field_counts_as_mismatch():
    """If loaded summary lacks an identity field (e.g. older schema didn't
    write seed), treat as mismatch (None != expected_value)."""
    expected = {"run_id": "run_42", "seed": 42}
    loaded = {"run_id": "run_42"}  # no seed
    result = ExperimentRunner._validate_resume_identity(loaded, expected)
    assert result is not None
    assert "seed" in result


# ---------------------------------------------------------------------------
# Code-level invariants (B-168 + B-169 wired into runner.run / _run_and_record_episode)
# ---------------------------------------------------------------------------


def test_runner_except_path_calls_aggregate_partial_steps():
    """Verify runner's _run_and_record_episode except path actually invokes
    the recovery helper (not just zero-step summary). Otherwise the helper
    is dead code."""
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    # The except block must read partial JSONL via read_jsonl_dedup AND
    # call _aggregate_partial_steps
    assert "from p79.experiment.io_utils import read_jsonl_dedup" in src, (
        "B-168 missing read_jsonl_dedup import in runner"
    )
    assert "read_jsonl_dedup(_jsonl_path)" in src, (
        "B-168 except path must call read_jsonl_dedup on JSONL"
    )
    assert "self._aggregate_partial_steps" in src, (
        "B-168 except path must call _aggregate_partial_steps"
    )
    # Error summary now uses _agg values instead of zero-step literals
    assert 'steps=_agg["steps"]' in src
    assert 'total_tokens=_agg["total_tokens"]' in src


def test_runner_resume_gate_uses_identity_check():
    """Verify resume gate actually invokes _validate_resume_identity (not
    just exists() check)."""
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    assert "_validate_resume_identity" in src, (
        "B-169 missing _validate_resume_identity call in run() resume gate"
    )
    # Quarantine path must be set up under episodes/quarantine/
    assert "quarantine" in src
    assert 'shutil.move(' in src or "shutil.move" in src, (
        "B-169 quarantine path must move file (not just rename) for cross-fs safety"
    )


def test_runner_imports_required_for_g3():
    """All G3 imports present at module top."""
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    assert "from p79.experiment.io_utils import read_jsonl_dedup" in src
    # shutil and time already imported at top
    assert re.search(r"^import shutil$", src, re.MULTILINE) or "import shutil" in src
    assert re.search(r"^import time$", src, re.MULTILINE) or "import time" in src


import re
