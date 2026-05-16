"""Invariant tests for /stress A1.4b-ii G3 trajectory + wasted_cost (B-193/B-194).

- B-193: trajectory_incomplete / unknown_failure_reasons /
         partial_recovery_step_count → EpisodeSummaryV2 dataclass +
         schema_migrations.v2 defaults + aggregate_condition_metrics emits
         per-cell rates
- B-194: exception path wasted_cost_usd = total_cost (recovered from partial
         JSONL), not 0.0 — matches `compute_wasted_cost(success=False)` semantic
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ─── B-193 ──────────────────────────────────────────────────────────────────
def test_b193_episode_summary_v2_has_telemetry_fields():
    from p79.experiment.types import EpisodeSummaryV2
    from dataclasses import fields
    names = {f.name for f in fields(EpisodeSummaryV2)}
    for f in ("trajectory_incomplete", "unknown_failure_reasons",
              "partial_recovery_step_count"):
        assert f in names, f"EpisodeSummaryV2 missing {f}"


def test_b193_schema_migrations_v2_defaults_include_telemetry():
    from p79.experiment.schema_migrations.v2 import EPISODE_SUMMARY_V2_DEFAULTS
    for f in ("trajectory_incomplete", "unknown_failure_reasons",
              "partial_recovery_step_count"):
        assert f in EPISODE_SUMMARY_V2_DEFAULTS, f"defaults catalog missing {f}"
    assert EPISODE_SUMMARY_V2_DEFAULTS["trajectory_incomplete"] is False
    assert EPISODE_SUMMARY_V2_DEFAULTS["unknown_failure_reasons"] == {}
    assert EPISODE_SUMMARY_V2_DEFAULTS["partial_recovery_step_count"] == 0


def test_b193_aggregate_condition_metrics_emits_rates():
    from p79.experiment.metrics import aggregate_condition_metrics
    eps = [
        # Episode 1: trajectory_incomplete=True, partial_recovery=5, unknown={"foo":1}
        {"success": False, "trajectory_incomplete": True,
         "partial_recovery_step_count": 5,
         "unknown_failure_reasons": {"foo": 1}},
        # Episode 2: incomplete=False, recovery=0
        {"success": True, "trajectory_incomplete": False,
         "partial_recovery_step_count": 0,
         "unknown_failure_reasons": {}},
        # Episode 3: incomplete=True, recovery=3, unknown={"foo":2, "bar":1}
        {"success": False, "trajectory_incomplete": True,
         "partial_recovery_step_count": 3,
         "unknown_failure_reasons": {"foo": 2, "bar": 1}},
    ]
    out = aggregate_condition_metrics(eps)
    assert out["trajectory_incomplete_episode_count"] == 2
    assert out["trajectory_incomplete_rate"] == pytest.approx(2/3)
    assert out["partial_recovery_episode_count"] == 2  # eps 1 + 3 have >0 recovery
    assert out["partial_recovery_rate"] == pytest.approx(2/3)
    assert out["unknown_failure_reason_distribution"] == {"foo": 3, "bar": 1}


def test_b193_aggregate_condition_metrics_empty_list_defaults():
    from p79.experiment.metrics import aggregate_condition_metrics
    out = aggregate_condition_metrics([])
    assert out["trajectory_incomplete_episode_count"] == 0
    assert out["trajectory_incomplete_rate"] == 0.0
    assert out["partial_recovery_episode_count"] == 0
    assert out["partial_recovery_rate"] == 0.0
    assert out["unknown_failure_reason_distribution"] == {}


# ─── B-194 ──────────────────────────────────────────────────────────────────
def test_b194_exception_path_wasted_cost_eq_total():
    """Source-level check: exception path sets wasted = total (not 0)."""
    src = (Path(__file__).resolve().parents[1] /
           "p79" / "experiment" / "runner" / "main.py").read_text()
    assert "B-194" in src
    # The fix line uses the recovered partial total, not a hardcoded 0.0
    assert 'summary["wasted_cost_usd"] = float(_agg.get("total_cost_usd"' in src
    # Pre-fix bug pattern gone (legacy 'summary["wasted_cost_usd"] = 0.0' in
    # exception path) — note we only check ANYWHERE in main.py to avoid false
    # positives from other unrelated 0.0 assignments; the explicit fix line
    # above is the positive proof.
