"""Invariant tests for /stress A1.4b-ii G4 defensive validators (B-195~B-200).

- B-195: obs_prepare aggregate comment + cost field emission preserved
- B-196: jsonl_integrity_report.csv emitted on corrupt-line discovery
- B-197: cost_efficiency_ratio returns None when no cost data
- B-198: logger_v2 calls _fsync_dir after every os.replace
- B-199: detect_benchmark_noise has api_rate_limit + auth_expired_or_session_invalid
         categories; aggregator emits benchmark_noise_category_distribution
- B-200: p95 filters None / NaN gracefully (returns 0.0 on empty valid set)
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest


# ─── B-196 ──────────────────────────────────────────────────────────────────
def test_b196_integrity_log_records_corrupt_lines(tmp_path):
    from p79.experiment.io_utils import read_jsonl_dedup, _JSONL_INTEGRITY_LOG
    _JSONL_INTEGRITY_LOG.clear()

    jsonl = tmp_path / "task_1_steps_v2.jsonl"
    jsonl.write_text(
        '{"step_idx": 0, "x": 1}\n'
        'this is not valid JSON\n'  # corrupt
        '{"step_idx": 1, "x": 2}\n'
        'also broken\n'              # corrupt
        '{"step_idx": 2, "x": 3}\n'
    )
    read_jsonl_dedup(jsonl)
    assert len(_JSONL_INTEGRITY_LOG) == 1
    rec = _JSONL_INTEGRITY_LOG[0]
    assert rec["lines_read"] == 5
    assert rec["corrupt_lines"] == 2
    # B-293 fix (2026-05-16, A1.8): summary_path=None → identity_mismatch=None
    # (was False pre-fix, semantically misleading "checked + matched"). None
    # means "not checked"; False/True means "checked + matched/mismatch".
    assert rec["summary_identity_mismatch"] is None


def test_b196_integrity_report_emitted_by_analyze_run(tmp_path):
    """End-to-end: analyze_run writes jsonl_integrity_report.csv when JSONL had corrupt lines."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    from p79.experiment.analysis import analyze_run

    cond = tmp_path / "phase1_dom_router_0"
    eps = cond / "episodes"
    eps.mkdir(parents=True)
    # condition_summary exists so analyze_run runs the collect path
    (cond / "condition_summary_v2.json").write_text(json.dumps({
        "condition_id": "phase1_dom_router_0",
        "seed": 42, "phase": "phase1", "backend_id": "b1",
        "som_on": False, "observation_mode": "dom", "router_on": False,
        "module_flags": {}, "episodes": 1, "success_rate": 1.0,
        "avg_steps": 1.0, "p95_step_latency_ms": 100.0,
        "avg_total_model_cost_usd": 0.0, "avg_total_cost_usd": 0.0,
        "avg_router_overhead_cost_usd": 0.0,
        "avg_total_energy_kwh": None, "avg_total_co2e_kg": None,
        "avg_retries": 0.0, "avg_no_op_rate": 0.0, "avg_page_unchanged_rate": 0.0,
        "avg_escalation_count": 0.0, "trigger_distribution": {},
        "state_change_reason_distribution": {},
        "avg_checklist_completion_rate": None,
        "checklist_failure_episode_rate": None,
        "benchmark_noise_rate": 0.0, "wasted_energy_kwh": 0.0,
        "avg_wasted_cost_usd": 0.0, "avg_wasted_energy_kwh": 0.0,
        "cost_efficiency_ratio": 0.0,
    }))
    # one valid summary + one corrupt JSONL row in steps.
    # B-599 (/stress A1.6a, 2026-05-17): `_collect_step_records` now passes
    # `strict_identity=True` to `read_jsonl_dedup` (B-571 paper-grade
    # fail-loud). The test scenario has 2 valid JSONL rows (1 corrupt
    # dropped) so summary.steps must equal 2 to satisfy identity check;
    # the B-196 corrupt-line integrity report logging is orthogonal and
    # still emits.
    (eps / "1_summary_v2.json").write_text(json.dumps({
        "schema_version": "2.0", "run_id": "r1",
        "condition_id": "phase1_dom_router_0",
        "benchmark": "vwa", "benchmark_site": "classifieds",
        "task_id": 1, "seed": 42, "success": True, "score": 1.0,
        "steps": 2, "retries": 0, "no_op_rate": 0.0, "page_unchanged_rate": 0.0,
        "total_latency_ms": 100.0, "p95_step_latency_ms": 100.0,
        "total_tokens": 0, "total_model_cost_usd": 0.0, "total_cost_usd": 0.0,
        "total_router_overhead_cost_usd": 0.0, "total_router_overhead_ms": 0.0,
        "total_energy_kwh": None, "total_co2e_kg": None,
        "escalation_count": 0, "trigger_distribution": {},
        "benchmark_noise": False, "benchmark_noise_category": None,
        "artifacts_dir": "",
    }))
    (eps / "1_steps_v2.jsonl").write_text(
        '{"step_idx": 0, "x": 1}\n'
        'corrupt-line-not-json\n'
        '{"step_idx": 1, "x": 2}\n'
    )

    analyze_run(str(tmp_path))
    integrity_csv = tmp_path / "analysis" / "jsonl_integrity_report.csv"
    assert integrity_csv.exists()
    text = integrity_csv.read_text()
    # The 1_steps_v2.jsonl with 1 corrupt line should show up
    assert "1_steps_v2.jsonl" in text
    assert ",1," in text or ",1\n" in text  # 1 corrupt line column value


# ─── B-197 ──────────────────────────────────────────────────────────────────
def test_b197_cost_efficiency_none_when_no_cost():
    from p79.experiment.metrics import aggregate_condition_metrics
    from conftest import complete_episode_summary
    # Fire-6 RCA Stage C1 fixture-drift fix (2026-05-20): derive from DEFAULTS
    # so canonical require_present metrics populate (see conftest helper).
    eps_zero_cost = [
        complete_episode_summary(success=True, total_cost_usd=0.0),
        complete_episode_summary(success=False, total_cost_usd=0.0),
    ]
    out = aggregate_condition_metrics(eps_zero_cost)
    assert out["cost_efficiency_ratio"] is None


def test_b197_cost_efficiency_correct_when_data_present():
    from p79.experiment.metrics import aggregate_condition_metrics
    from conftest import complete_episode_summary
    eps = [
        complete_episode_summary(success=True, total_cost_usd=0.10),
        complete_episode_summary(success=False, total_cost_usd=0.30),
    ]
    out = aggregate_condition_metrics(eps)
    # cost_on_success / total = 0.10 / 0.40 = 0.25
    assert out["cost_efficiency_ratio"] == pytest.approx(0.25)


# ─── B-198 ──────────────────────────────────────────────────────────────────
def test_b198_logger_v2_imports_fsync_dir():
    src = (Path(__file__).resolve().parents[1] /
           "p79" / "experiment" / "logger_v2.py").read_text()
    assert "def _fsync_dir(" in src
    assert "B-198" in src
    # 3 callsites — write_condition_meta + write_episode_summary + write_condition_summary
    assert src.count("_fsync_dir(path.parent)") >= 3


def test_b198_fsync_dir_swallows_eopnotsupp(tmp_path):
    """Best-effort: _fsync_dir must not raise on platforms without dir fsync support."""
    from p79.experiment.logger_v2 import _fsync_dir
    _fsync_dir(tmp_path)  # should not raise
    # nonexistent path → OSError swallowed
    _fsync_dir(tmp_path / "does_not_exist")


# ─── B-199 ──────────────────────────────────────────────────────────────────
def test_b199_detect_benchmark_noise_classifies_new_categories():
    from p79.experiment.metrics import detect_benchmark_noise
    cases = [
        ("HTTP 429 Too Many Requests", "api_rate_limit"),
        ("rate limit exceeded", "api_rate_limit"),
        ("Session expired, please login", "auth_expired_or_session_invalid"),
        ("401 Unauthorized auth expired", "auth_expired_or_session_invalid"),
    ]
    for msg, expected_cat in cases:
        is_noise, cat = detect_benchmark_noise(msg)
        assert is_noise is True
        assert cat == expected_cat, f"{msg!r} → got {cat}, expected {expected_cat}"


def test_b199_noise_category_distribution_emitted():
    from p79.experiment.metrics import aggregate_condition_metrics
    from conftest import complete_episode_summary
    eps = [
        complete_episode_summary(success=False, benchmark_noise=True, benchmark_noise_category="api_rate_limit"),
        complete_episode_summary(success=False, benchmark_noise=True, benchmark_noise_category="timeout"),
        complete_episode_summary(success=False, benchmark_noise=True, benchmark_noise_category="api_rate_limit"),
        complete_episode_summary(success=True, benchmark_noise=False, benchmark_noise_category=None),
    ]
    out = aggregate_condition_metrics(eps)
    assert out["benchmark_noise_category_distribution"] == {
        "api_rate_limit": 2,
        "timeout": 1,
    }


# ─── B-200 ──────────────────────────────────────────────────────────────────
def test_b200_p95_handles_none_and_nan():
    from p79.experiment.metrics import p95
    # All None → empty valid set → 0.0
    assert p95([None, None]) == 0.0
    # Mixed None + valid → just valid considered (sorted [1, 2, 3] → P95 ≈ 2.9)
    res = p95([None, 1.0, 2.0, 3.0])
    assert res == pytest.approx(2.9, abs=1e-9)
    # NaN filtered
    assert p95([float("nan"), 1.0, 2.0, 3.0]) == pytest.approx(2.9, abs=1e-9)
    # All NaN → empty → 0.0
    assert p95([float("nan"), float("nan")]) == 0.0


# ─── B-456 ──────────────────────────────────────────────────────────────────
def test_b456_p95_strict_mode_raises_on_empty():
    """B-456 (/stress A1.4 P1-8-C gemini OOB, 2026-05-17): opt-in strict mode
    on empty input raises ValueError so figure renderers / cross-arm
    aggregators can display "N/A" rather than injecting 0.0 into mean(p95)
    and falsely advantaging the most-failing arm.

    Default (strict=False) keeps legacy 0.0 contract; explicit strict=True
    fails loud so callers must handle the catastrophic-empty case.
    """
    from p79.experiment.metrics import p95

    # Legacy contract preserved on default strict=False
    assert p95([]) == 0.0
    assert p95([None, None]) == 0.0
    assert p95([float("nan"), float("nan")]) == 0.0

    # strict=True raises on empty / all-None / all-NaN
    with pytest.raises(ValueError, match=r"empty valid input set"):
        p95([], strict=True)
    with pytest.raises(ValueError, match=r"empty valid input set"):
        p95([None, None], strict=True)
    with pytest.raises(ValueError, match=r"empty valid input set"):
        p95([float("nan")], strict=True)

    # strict=True with non-empty valid set works normally (no raise, returns p95)
    assert p95([1.0, 2.0, 3.0], strict=True) == pytest.approx(2.9, abs=1e-9)
    assert p95([None, 1.0, 2.0, 3.0], strict=True) == pytest.approx(2.9, abs=1e-9)
