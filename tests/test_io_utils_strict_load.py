"""Invariant tests for `p79.experiment.io_utils` — /stress A1.12 P0-3.

Pre-2026-05-16 status: `load_episode_summary_strict` (B-283) +
`read_jsonl_dedup` (B-180/B-196/B-287/B-288/B-293) had ZERO direct unit
tests despite being the canonical paper-grade FP-control loader. Any
silent refactor of the strict/lenient boundary, identity-tuple keys, or
integrity-log shape could silently degrade aggregator behaviour with no
test signal.

These tests pin the contract so future maintainers cannot regress:

- strict mode raises on every type mismatch listed in io_utils.py:57-62
- lenient mode logs + returns None for type mismatch
- read_jsonl_dedup integrity log carries (corrupt_lines, dedup_discarded,
  summary_identity_mismatch, step_idx_non_monotonic) shape
- B-180 identity-tuple mismatch (schema_version / run_id / condition_id /
  seed / benchmark_site / task_id) is detected
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from p79.experiment.io_utils import (
    _JSONL_INTEGRITY_LOG,
    load_episode_summary_strict,
    read_jsonl_dedup,
)


# ─── Fixture helpers ────────────────────────────────────────────────────────
def _valid_summary(task_id: int = 1, success: bool = True) -> dict:
    """Minimal schema-valid summary per io_utils.py:57-62."""
    return {
        "schema_version": "2.0",
        "task_id": task_id,
        "success": success,
    }


def _write_summary(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


# ─── load_episode_summary_strict: valid path ────────────────────────────────
def test_strict_load_valid_summary_returns_dict(tmp_path):
    path = _write_summary(tmp_path / "task_0_summary_v2.json", _valid_summary(task_id=0))
    result = load_episode_summary_strict(path)
    assert result is not None
    assert result["task_id"] == 0
    assert result["success"] is True
    assert result["schema_version"] == "2.0"


def test_strict_load_lenient_valid_summary_returns_dict(tmp_path):
    """Lenient mode on valid payload behaves identically to strict mode."""
    path = _write_summary(tmp_path / "task_1_summary_v2.json", _valid_summary())
    assert load_episode_summary_strict(path, mode="lenient") is not None


# ─── load_episode_summary_strict: type-mismatch surface ─────────────────────
def test_strict_load_missing_schema_version_raises(tmp_path):
    """B-283 contract: schema_version is the strict-load gate.

    Without it, `aggregate_phase1_prereg_gate.py` silently treats summaries as
    corrupt-skip → θ collapses to 0 (caught by /stress A1.12 P0-1).
    """
    bad = _valid_summary()
    del bad["schema_version"]
    path = _write_summary(tmp_path / "task_2_summary_v2.json", bad)
    with pytest.raises(ValueError, match=r"schema_version"):
        load_episode_summary_strict(path)


def test_strict_load_non_bool_success_raises(tmp_path):
    """`success="false"` would have been truthy pre-B-283 — must now raise."""
    bad = _valid_summary()
    bad["success"] = "false"  # string, not bool
    path = _write_summary(tmp_path / "task_3_summary_v2.json", bad)
    with pytest.raises(ValueError, match=r"success="):
        load_episode_summary_strict(path)


def test_strict_load_non_int_task_id_raises(tmp_path):
    bad = _valid_summary()
    bad["task_id"] = "1"  # string, not int
    path = _write_summary(tmp_path / "task_4_summary_v2.json", bad)
    with pytest.raises(ValueError, match=r"task_id="):
        load_episode_summary_strict(path)


def test_strict_load_corrupt_json_raises(tmp_path):
    path = tmp_path / "task_5_summary_v2.json"
    path.write_text("{not valid json")
    with pytest.raises(ValueError, match=r"Corrupt JSON"):
        load_episode_summary_strict(path)


def test_strict_load_missing_file_raises_filenotfound(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_episode_summary_strict(tmp_path / "missing.json")


# ─── load_episode_summary_strict: lenient downgrade contract ────────────────
def test_lenient_load_missing_schema_version_returns_none(tmp_path, caplog):
    """B-283 lenient: type mismatch → log warning + return None (the path
    aggregate_phase1_prereg_gate.py exercises via aggregate_phantom_lift.load)."""
    bad = _valid_summary()
    del bad["schema_version"]
    path = _write_summary(tmp_path / "task_6_summary_v2.json", bad)
    with caplog.at_level("WARNING"):
        result = load_episode_summary_strict(path, mode="lenient")
    assert result is None
    assert any("B-283" in r.message and "schema_version" in r.message for r in caplog.records)


def test_lenient_load_corrupt_json_returns_none(tmp_path, caplog):
    path = tmp_path / "task_7_summary_v2.json"
    path.write_text("{garbage")
    with caplog.at_level("WARNING"):
        result = load_episode_summary_strict(path, mode="lenient")
    assert result is None
    assert any("corrupt json" in r.message.lower() for r in caplog.records)


# ─── read_jsonl_dedup: integrity log shape ──────────────────────────────────
def test_read_jsonl_dedup_records_corrupt_lines(tmp_path):
    """B-196: per-file integrity stats include corrupt_lines + dedup_discarded."""
    _JSONL_INTEGRITY_LOG.clear()
    jsonl = tmp_path / "task_8_steps_v2.jsonl"
    jsonl.write_text(
        '{"step_idx": 0, "x": 1}\n'
        'corrupt-not-json\n'
        '{"step_idx": 1, "x": 2}\n'
        'also-broken\n'
        '{"step_idx": 2, "x": 3}\n'
    )
    segment = read_jsonl_dedup(jsonl)
    assert len(segment) == 3  # 5 lines − 2 corrupt = 3
    assert len(_JSONL_INTEGRITY_LOG) == 1
    rec = _JSONL_INTEGRITY_LOG[0]
    assert rec["lines_read"] == 5
    assert rec["corrupt_lines"] == 2
    assert rec["dedup_discarded"] == 0
    # B-293: identity_mismatch = None when summary_path not provided
    assert rec["summary_identity_mismatch"] is None
    # B-287: monotonic 0/1/2 → no flag
    assert rec["step_idx_non_monotonic"] is False


def test_read_jsonl_dedup_restart_artifact_keeps_last_segment(tmp_path):
    """B-180/B-287: when step_idx resets to 0 mid-file (restart), keep tail only."""
    _JSONL_INTEGRITY_LOG.clear()
    jsonl = tmp_path / "task_9_steps_v2.jsonl"
    jsonl.write_text(
        '{"step_idx": 0, "x": "first_run"}\n'
        '{"step_idx": 1, "x": "first_run"}\n'
        '{"step_idx": 0, "x": "restart"}\n'
        '{"step_idx": 1, "x": "restart"}\n'
        '{"step_idx": 2, "x": "restart"}\n'
    )
    segment = read_jsonl_dedup(jsonl)
    assert len(segment) == 3
    assert all(r["x"] == "restart" for r in segment)
    rec = _JSONL_INTEGRITY_LOG[0]
    assert rec["dedup_discarded"] == 2


def test_read_jsonl_dedup_identity_mismatch_flagged(tmp_path):
    """B-180: summary identity tuple mismatch surfaces in integrity log."""
    _JSONL_INTEGRITY_LOG.clear()
    jsonl = tmp_path / "task_10_steps_v2.jsonl"
    jsonl.write_text(
        '{"step_idx": 0, "schema_version": "2.0", "task_id": 10, "condition_id": "phase1_dom_router_0"}\n'
        '{"step_idx": 1, "schema_version": "2.0", "task_id": 10, "condition_id": "phase1_dom_router_0"}\n'
    )
    summary = tmp_path / "task_10_summary_v2.json"
    summary.write_text(json.dumps({
        "schema_version": "2.0",
        "task_id": 10,
        "condition_id": "phase1_DIFFERENT_condition_id",  # mismatch
        "success": True,
        "steps": 2,
    }))
    read_jsonl_dedup(jsonl, summary_path=summary)
    rec = _JSONL_INTEGRITY_LOG[0]
    assert rec["summary_identity_mismatch"] is True


def test_read_jsonl_dedup_identity_match_no_flag(tmp_path):
    """B-180: matching identity tuple → integrity log marks False (not None)."""
    _JSONL_INTEGRITY_LOG.clear()
    jsonl = tmp_path / "task_11_steps_v2.jsonl"
    jsonl.write_text(
        '{"step_idx": 0, "schema_version": "2.0", "task_id": 11, "condition_id": "x"}\n'
    )
    summary = tmp_path / "task_11_summary_v2.json"
    summary.write_text(json.dumps({
        "schema_version": "2.0", "task_id": 11, "condition_id": "x",
        "success": True, "steps": 1,
    }))
    read_jsonl_dedup(jsonl, summary_path=summary)
    rec = _JSONL_INTEGRITY_LOG[0]
    assert rec["summary_identity_mismatch"] is False  # explicit False, not None


def test_read_jsonl_dedup_empty_file_handles_gracefully(tmp_path):
    _JSONL_INTEGRITY_LOG.clear()
    jsonl = tmp_path / "task_12_steps_v2.jsonl"
    jsonl.write_text("")
    assert read_jsonl_dedup(jsonl) == []
    rec = _JSONL_INTEGRITY_LOG[0]
    assert rec["lines_read"] == 0
    assert rec["corrupt_lines"] == 0
