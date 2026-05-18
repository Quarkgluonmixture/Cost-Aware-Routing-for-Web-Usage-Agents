"""Regression tests for /stress 深入审 Mode A — Chunk 1 B-1600 P0-1-A*.

Asserts the runner's `_aggregate_partial_steps` + success-path episode
summary construction writes `total_latency_minus_retry_ms` to
`EpisodeSummaryV2`. Pre-fix the schema field existed in
`schema_migrations/v2.py:51` (default None), `metrics.py:_avg` consumed it,
and `aggregate_cross_site.py:281/442` carried it — but no code anywhere
wrote it. Pass-1 fire would have produced empty rollup data → paper §1
retry-adjusted canonical latency claim + paper §3.5.1 disclosure has no
data substrate. See `docs/checkpoints/master_bug_catalog.md ## /stress
深入审` (B-1600) + chronicle §220.
"""

from __future__ import annotations

import pytest

from p79.experiment.types import EpisodeSummaryV2


def _mk_step(total_ms: float, retry_wait_ms: float = 0.0, **extras: float) -> dict:
    """Synthetic step record with `latency_ms` shaped like runner main L2535-2579."""
    return {
        "latency_ms": {
            "total": total_ms,
            "total_minus_retry": total_ms - retry_wait_ms,
            "backend_infer": total_ms * 0.8,
            "env_step": total_ms * 0.1,
        },
        "tokens": {"total": 100},
        "cost_usd": {"model": 0.001},
        "router": {"overhead_ms": {}},
        "action_success": True,
        "page_changed": True,
        "retry_count": 0,
        "observation_mode": "som",
        "action": {"action_type": "click"},
        **extras,
    }


def test_episode_summary_dataclass_has_field():
    """EpisodeSummaryV2 dataclass must declare total_latency_minus_retry_ms."""
    fields = {f.name for f in EpisodeSummaryV2.__dataclass_fields__.values()}
    assert "total_latency_minus_retry_ms" in fields, (
        "B-1600 P0-1-A*: dataclass field missing — runner writer cannot land. "
        "Add `total_latency_minus_retry_ms: Optional[float] = None` near "
        "`component_breakdown` in types.py."
    )


def test_episode_summary_default_is_none():
    """Field defaults to None (pre-fire compat with B-1410 schema migration)."""
    summary = EpisodeSummaryV2(
        schema_version="v2",
        run_id="test_run",
        condition_id="test_cond",
        benchmark="vwa",
        benchmark_site="reddit",
        task_id=0,
        seed=42,
        success=True,
        score=1.0,
        steps=1,
        retries=0,
        no_op_rate=0.0,
        page_unchanged_rate=0.0,
        total_latency_ms=1234.5,
        p95_step_latency_ms=1234.5,
        total_tokens=100,
        total_model_cost_usd=0.001,
        total_cost_usd=0.001,
        total_router_overhead_cost_usd=0.0,
        total_router_overhead_ms=0.0,
        total_energy_kwh=None,
        total_co2e_kg=None,
        escalation_count=0,
        trigger_distribution={},
        benchmark_noise=False,
        benchmark_noise_category=None,
        artifacts_dir="/tmp/test",
    )
    assert summary.total_latency_minus_retry_ms is None, (
        "Default factory must be None — legacy summaries without the field "
        "must serialize as None, NOT 0.0 (0.0 would silently equal raw total)."
    )


def test_episode_summary_accepts_explicit_value():
    """Field accepts explicit float (the runner write path B-1600 emits)."""
    summary = EpisodeSummaryV2(
        schema_version="v2",
        run_id="test_run",
        condition_id="test_cond",
        benchmark="vwa",
        benchmark_site="reddit",
        task_id=0,
        seed=42,
        success=True,
        score=1.0,
        steps=2,
        retries=0,
        no_op_rate=0.0,
        page_unchanged_rate=0.0,
        total_latency_ms=10_000.0,
        total_latency_minus_retry_ms=7_500.0,  # retry-adjusted = 2.5s less
        p95_step_latency_ms=5_000.0,
        total_tokens=200,
        total_model_cost_usd=0.002,
        total_cost_usd=0.002,
        total_router_overhead_cost_usd=0.0,
        total_router_overhead_ms=0.0,
        total_energy_kwh=None,
        total_co2e_kg=None,
        escalation_count=0,
        trigger_distribution={},
        benchmark_noise=False,
        benchmark_noise_category=None,
        artifacts_dir="/tmp/test",
    )
    assert summary.total_latency_minus_retry_ms == 7_500.0


def test_aggregate_partial_steps_returns_field():
    """_aggregate_partial_steps return dict includes total_latency_minus_retry_ms."""
    from p79.experiment.runner.main import ExperimentRunner
    # Construct a minimal ExperimentRunner without going through __init__ —
    # we only need the bound method's logic. _aggregate_partial_steps is a
    # method but doesn't use self state beyond logger imports.
    runner = ExperimentRunner.__new__(ExperimentRunner)

    # Empty case
    agg_empty = runner._aggregate_partial_steps([], "som")
    assert "total_latency_minus_retry_ms" in agg_empty
    assert agg_empty["total_latency_minus_retry_ms"] == 0.0

    # Non-empty case — B0-style step with retry_wait_ms
    steps = [
        _mk_step(total_ms=10_000.0, retry_wait_ms=2_000.0),  # B0 with 2s retry
        _mk_step(total_ms=5_000.0, retry_wait_ms=0.0),       # B1/B2 style
    ]
    agg = runner._aggregate_partial_steps(steps, "som")
    assert "total_latency_minus_retry_ms" in agg
    assert agg["total_latency_minus_retry_ms"] == 13_000.0  # 10k - 2k + 5k - 0
    assert agg["total_latency_ms"] == 15_000.0  # raw sum
    assert agg["total_latency_minus_retry_ms"] < agg["total_latency_ms"]


def test_aggregate_partial_steps_backward_compat_missing_minus_retry():
    """Step records pre-B-143 (no total_minus_retry key) fall back to total."""
    from p79.experiment.runner.main import ExperimentRunner
    runner = ExperimentRunner.__new__(ExperimentRunner)

    # Legacy step record without total_minus_retry — should fall back to total
    legacy_steps = [
        {
            "latency_ms": {"total": 8_000.0},  # no total_minus_retry
            "tokens": {"total": 50},
            "cost_usd": {"model": 0.0005},
            "router": {"overhead_ms": {}},
            "action_success": True,
            "page_changed": True,
            "retry_count": 0,
            "observation_mode": "dom",
            "action": {"action_type": "type"},
        }
    ]
    agg = runner._aggregate_partial_steps(legacy_steps, "dom")
    # Legacy fallback: minus_retry == total (no retry data available)
    assert agg["total_latency_minus_retry_ms"] == agg["total_latency_ms"] == 8_000.0


def test_schema_validation_accepts_new_field():
    """_EPISODE_FIELD_TYPES validation map includes total_latency_minus_retry_ms."""
    from p79.experiment.types import _EPISODE_FIELD_TYPES
    assert "total_latency_minus_retry_ms" in _EPISODE_FIELD_TYPES
    # Must accept int, float, AND None (legacy summaries pre-B-1600 = None)
    expected_types = _EPISODE_FIELD_TYPES["total_latency_minus_retry_ms"]
    assert int in expected_types
    assert float in expected_types
    assert type(None) in expected_types
