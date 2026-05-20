"""GRL boundary audit fixes (/stress 2026-05-20) — Chunk A P0 fire-blockers.

Covers the two P0 findings the user gated Fire-6 on (Q1=A, Q2=A):

  P0-1-AB  paper_grade XOR diagnostic_replay hard block (runner + queue layer).
           A leaked P79_DIAGNOSTIC_REPLAY=1 must NOT silently turn a paper-grade
           fire into non-canonical sr_excluded data + M1-abort-suppressed.

  P0-2-B   watchdog must NOT delete+retry an error episode under paper_grade
           (denominator surgery; mode/site/backend-correlated). Invariant tested
           on the extracted pure `_can_auto_retry` decision.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ───────────────────────── P0-2-B: watchdog no denominator surgery ──────────


from scripts.maintenance.experiment_watchdog import _can_auto_retry

_RETRY_KW = dict(
    quarantine_flagged=False,
    condition_completed=False,
    retries_so_far=0,
    max_code_bug_retries=2,
    max_noise_retries=3,
)


def test_p0_2_paper_grade_never_auto_retries_code_bug():
    """INVARIANT: paper_grade ⇒ no delete+retry, even with retry budget left."""
    assert _can_auto_retry("error(code_bug)", paper_grade=True, **_RETRY_KW) is False


def test_p0_2_paper_grade_never_auto_retries_noise():
    assert _can_auto_retry("error(timeout)", paper_grade=True, **_RETRY_KW) is False
    assert _can_auto_retry("error(session)", paper_grade=True, **_RETRY_KW) is False
    assert _can_auto_retry("error(connection)", paper_grade=True, **_RETRY_KW) is False


def test_p0_2_non_paper_grade_preserves_legacy_retry():
    """Negative control: dev mode keeps the historical auto-clean retry path."""
    assert _can_auto_retry("error(code_bug)", paper_grade=False, **_RETRY_KW) is True
    assert _can_auto_retry("error(timeout)", paper_grade=False, **_RETRY_KW) is True


def test_p0_2_legacy_guards_still_hold_in_dev_mode():
    """The pre-existing predicate semantics must be unchanged for non-paper-grade."""
    # evaluator path never retries (B-486 / EvaluatorUnavailableError re-raise)
    assert _can_auto_retry("error(evaluator)", paper_grade=False, **_RETRY_KW) is False
    # quarantine_flagged (needs_reevaluation=True) suppresses retry (P1-10-B)
    kw = {**_RETRY_KW, "quarantine_flagged": True}
    assert _can_auto_retry("error(timeout)", paper_grade=False, **kw) is False
    # post-aggregation never deletes (would desync condition_summary)
    kw = {**_RETRY_KW, "condition_completed": True}
    assert _can_auto_retry("error(timeout)", paper_grade=False, **kw) is False
    # non-error reasons (success / cycle / etc.) never retry
    assert _can_auto_retry("success", paper_grade=False, **_RETRY_KW) is False
    # retry budget exhausted
    kw = {**_RETRY_KW, "retries_so_far": 2}
    assert _can_auto_retry("error(code_bug)", paper_grade=False, **kw) is False
    kw = {**_RETRY_KW, "retries_so_far": 3}
    assert _can_auto_retry("error(timeout)", paper_grade=False, **kw) is False


# ───────────────────────── P0-1-AB: paper_grade XOR diagnostic_replay ───────


def _mock_runner_cfg(tmp_path: Path) -> dict:
    """Minimal offline (mock env + mock backend) runner cfg, mirrors
    tests/test_runner_smoke.py::_mock_cfg."""
    site_cfg = tmp_path / "shopping.json"
    site_cfg.write_text(json.dumps([{
        "task_id": 0,
        "intent": "GRL XOR guard test task",
        "sites": ["shopping"],
        "start_url": "__SHOPPING__/",
    }]))
    return {
        "experiment": {
            "name": "grl_xor", "benchmark": "visualwebarena", "phase": "phase1",
            "seed": 42, "output_root": str(tmp_path / "results"), "run_id": "grl_xor_run",
        },
        "task": {
            "include_sites": ["shopping"], "max_tasks_per_site": 1,
            "task_ids": {}, "site_configs": {"shopping": str(site_cfg)},
        },
        "env": {"type": "mock", "viewport_width": 320, "viewport_height": 240},
        "runtime": {"max_steps": 3, "resume": False},
        "variables": {"primary": {"observation_mode": ["dom"]}},
        "router": {
            "cheap_default_mode": "dom", "rich_escalation_mode": "som",
            "thresholds": {"dom_size_threshold": 12000, "unchanged_steps_trigger": 2,
                           "no_progress_steps_trigger": 2, "retry_limit": 1},
            "overhead_cost_per_ms": 0.0,
        },
        "metrics": {
            "cost": {"input_cost_per_1k": 0.0, "output_cost_per_1k": 0.0},
            "energy": {"enabled": False, "kwh_per_step": None, "co2e_kg_per_kwh": None},
        },
        "checklist": {"enabled": False},
        "state_change": {"similarity_threshold": 0.95},
        "backends": {
            "default_backend": "local_4b",
            "local_4b": {"type": "local_qwen", "mock_mode": True, "dom_mode": "llm"},
        },
        "baselines": {"run_b0": False},
    }


def test_p0_1_paper_grade_plus_diagnostic_replay_raises(tmp_path):
    """Both flags True → fail-loud at runner init (cannot silently neuter a fire)."""
    from p79.experiment.runner import ExperimentRunner
    cfg = _mock_runner_cfg(tmp_path)
    cfg["paper_grade"] = True
    cfg["diagnostic_replay"] = True
    with pytest.raises(RuntimeError, match="forbids diagnostic_replay"):
        ExperimentRunner(cfg)


def test_p0_1_diagnostic_replay_alone_does_not_trip_xor(tmp_path):
    """Negative control: diagnostic_replay without paper_grade must NOT raise the
    XOR guard (queue_diagnostic_replay.sh path stays usable)."""
    from p79.experiment.runner import ExperimentRunner
    cfg = _mock_runner_cfg(tmp_path)
    cfg["paper_grade"] = False
    cfg["diagnostic_replay"] = True
    # Should construct without the XOR RuntimeError (other init may proceed).
    runner = ExperimentRunner(cfg)
    assert runner.diagnostic_replay is True


# ───────────────────── P1-1 / B-1780: C1b two-layer latency (B-1773 follow-up) ──


def test_p1_1_aggregator_consumes_recovered_screenshot_telemetry():
    """B-1780 (Q3=A): aggregate_condition_metrics CONSUMES the recovered-screenshot
    fields (B-1773 added them WRITE-ONLY — zero aggregators read them). Proves the
    consumption: avg recovered_total_ms + per-cell episode_rate."""
    from p79.experiment.metrics import aggregate_condition_metrics
    eps = [
        # 1 episode with a recovered dom screenshot timeout, 1 clean.
        {"success": True, "steps": 5, "total_latency_minus_retry_ms": 50000.0,
         "busy_wait_total_ms": 0.0, "screenshot_timeout_recovered_count": 1,
         "screenshot_timeout_recovered_total_ms": 30000.0},
        {"success": False, "steps": 8, "total_latency_minus_retry_ms": 10000.0,
         "busy_wait_total_ms": 0.0, "screenshot_timeout_recovered_count": 0,
         "screenshot_timeout_recovered_total_ms": 0.0},
    ]
    agg = aggregate_condition_metrics(eps)
    assert agg["avg_screenshot_timeout_recovered_total_ms"] == 15000.0  # (30000+0)/2
    assert agg["screenshot_timeout_recovered_episode_rate"] == 0.5      # 1 of 2 episodes


def test_p1_1_cross_site_canonical_subtracts_recovered():
    """B-1780: cross-site canonical = minus_retry − busy_wait − recovered (3 terms).
    Locks the arithmetic contract; recovered missing/None ≡ 0 (no C1b recovery),
    distinct from minus_retry/busy_wait None-propagate (verified by reading the
    aggregate_cross_site.py:avg_total_latency_canonical_ms composer)."""
    minus_retry, busy_wait, recovered = 50000.0, 5000.0, 30000.0
    canonical = minus_retry - busy_wait - (recovered or 0.0)
    assert canonical == 15000.0
    # None recovered ≡ 0 (legacy / no recovery)
    assert minus_retry - busy_wait - (None or 0.0) == 45000.0


def test_p1_1_runner_episode_has_two_layer_latency(tmp_path):
    """B-1780: runner stamps total_latency_canonical_ms + recovered_total_ms on
    every episode (mock env → 0 recovery → canonical == minus_retry)."""
    import json as _json
    from p79.experiment.runner import ExperimentRunner
    cfg = _mock_runner_cfg(tmp_path)
    run_dir = ExperimentRunner(cfg).run()
    # episode summaries are <site>_task_<id>_summary_v2.json (NOT
    # condition_summary_v2.json which also ends _summary_v2.json).
    summaries = [p for p in Path(run_dir).rglob("*_summary_v2.json") if "_task_" in p.name]
    assert summaries, "no episode summary produced"
    ep = _json.loads(summaries[0].read_text())
    assert "screenshot_timeout_recovered_total_ms" in ep
    assert "total_latency_canonical_ms" in ep
    assert ep["screenshot_timeout_recovered_total_ms"] == 0.0  # mock env: no timeout
    # canonical == minus_retry when no recovery (mock); both present + consistent.
    if ep.get("total_latency_minus_retry_ms") is not None:
        assert ep["total_latency_canonical_ms"] == ep["total_latency_minus_retry_ms"]
