"""Runner smoke test: protects §97 audit invariants + future refactors.

This is a lighter-weight cousin of `test_runner_integration.py` that focuses
on the schema invariants introduced by the §97 audit. It runs ONE
condition × ONE task in mock mode and asserts:

  1. Runner.run() succeeds and writes the canonical files.
  2. EpisodeSummaryV2 carries the §97-added fields:
       - busy_wait_total_ms (RU-4)
       - energy_partial / energy_step_complete_count (RU-5)
       - agent_finished
  3. page_unchanged_rate excludes finish/stop steps (RU-1 invariant).
  4. condition_summary_v2.json aggregates §97 fields:
       - avg_total_latency_ms (M-6)
       - avg_busy_wait_total_ms / energy_partial_episode_count (M-6)
       - avg_router_overhead_ms
  5. ExperimentRunner is importable via the canonical path
       (`from p79.experiment.runner import ExperimentRunner`) — guards
       against Step 3's planned package-split breakage.

Run: pytest tests/test_runner_smoke.py -v
"""
from __future__ import annotations

import json
from pathlib import Path

# Canonical import — Step 3 must keep this working post-split.
from p79.experiment.runner import ExperimentRunner
from p79.experiment.types import EpisodeSummaryV2, validate_step_record_v2


def _mock_cfg(tmp_path: Path) -> dict:
    """Minimal single-task single-condition mock config."""
    site_cfg = tmp_path / "shopping.json"
    site_cfg.write_text(json.dumps([{
        "task_id": 0,
        "intent": "Smoke test task",
        "sites": ["shopping"],
        "start_url": "__SHOPPING__/",
    }]))
    return {
        "experiment": {
            "name": "smoke",
            "benchmark": "visualwebarena",
            "phase": "phase1",
            "seed": 42,
            "output_root": str(tmp_path / "results"),
            "run_id": "smoke_run",
        },
        "task": {
            "include_sites": ["shopping"],
            "max_tasks_per_site": 1,
            "task_ids": {},
            "site_configs": {"shopping": str(site_cfg)},
        },
        "env": {"type": "mock", "viewport_width": 320, "viewport_height": 240},
        "runtime": {"max_steps": 3, "resume": False},
        "variables": {"primary": {"observation_mode": ["dom"]}},
        "router": {
            "cheap_default_mode": "dom",
            "rich_escalation_mode": "som",
            "thresholds": {
                "dom_size_threshold": 12000,
                "unchanged_steps_trigger": 2,
                "no_progress_steps_trigger": 2,
                "retry_limit": 1,
            },
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
            "local_4b": {
                "type": "local_qwen",
                "mock_mode": True,
                "dom_mode": "heuristic",
            },
        },
        "baselines": {"run_b0": False},
    }


def test_smoke_canonical_import_path():
    """Step 3 (runner.py split) must keep this import working."""
    from p79.experiment.runner import ExperimentRunner as ER  # noqa: F401
    # Sanity: the imported symbol is callable as a class
    assert callable(ER)


def test_smoke_runner_writes_canonical_artifacts(tmp_path):
    """Runner end-to-end: 1 condition × 1 task → canonical files exist."""
    cfg = _mock_cfg(tmp_path)
    runner = ExperimentRunner(cfg)
    run_dir = runner.run()

    assert (run_dir / "run_summary_v2.json").exists()
    assert (run_dir / "run_meta.json").exists()
    summaries = list(run_dir.glob("*/episodes/*_summary_v2.json"))
    assert len(summaries) == 1, f"expected 1 episode summary, got {len(summaries)}"
    cond_summaries = list(run_dir.glob("*/condition_summary_v2.json"))
    assert len(cond_summaries) == 1
    step_logs = list(run_dir.glob("*/episodes/*_steps_v2.jsonl"))
    assert len(step_logs) == 1


def test_smoke_episode_summary_audit_fields(tmp_path):
    """§97 audit added busy_wait_total_ms / energy_partial / energy_step_complete_count."""
    cfg = _mock_cfg(tmp_path)
    run_dir = ExperimentRunner(cfg).run()
    summary_path = next(run_dir.glob("*/episodes/*_summary_v2.json"))
    summary = json.loads(summary_path.read_text())

    # §97 RU-4: busy_wait_total_ms field
    assert "busy_wait_total_ms" in summary, "RU-4: busy_wait_total_ms missing"
    assert isinstance(summary["busy_wait_total_ms"], (int, float))
    # §97 RU-5: energy_partial diagnostics
    assert "energy_partial" in summary, "RU-5: energy_partial missing"
    assert "energy_step_complete_count" in summary, "RU-5: energy_step_complete_count missing"
    assert isinstance(summary["energy_partial"], bool)
    # agent_finished — standalone diagnostic (§139.8: no longer feeds an
    # adjusted_success layer; that post-hoc layer is retired)
    assert "agent_finished" in summary
    # Existing fields still present
    assert "wasted_cost_usd" in summary
    assert "page_unchanged_rate" in summary


def test_smoke_page_unchanged_rate_excludes_finish(tmp_path):
    """§97 RU-1 invariant: page_unchanged_rate must exclude finish/stop steps."""
    cfg = _mock_cfg(tmp_path)
    run_dir = ExperimentRunner(cfg).run()
    summary_path = next(run_dir.glob("*/episodes/*_summary_v2.json"))
    summary = json.loads(summary_path.read_text())
    steps_path = summary_path.with_name(
        summary_path.name.replace("_summary_v2.json", "_steps_v2.jsonl")
    )
    steps = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
    # Reproduce the §97 formula and compare
    n_total = len(steps)
    if n_total == 0:
        return  # vacuous
    expected_unchanged = sum(
        1 for s in steps
        if not bool(s.get("page_changed", False))
        and str((s.get("action") or {}).get("action_type", "")).lower() not in ("finish", "stop")
    )
    expected_rate = expected_unchanged / n_total
    actual_rate = float(summary["page_unchanged_rate"])
    # Allow tiny float epsilon
    assert abs(actual_rate - expected_rate) < 1e-9, (
        f"page_unchanged_rate {actual_rate} != expected {expected_rate} "
        f"(steps={n_total}, unchanged_excl_finish={expected_unchanged})"
    )


def test_smoke_condition_aggregate_audit_fields(tmp_path):
    """§97 M-6: aggregate must export busy_wait/energy_partial/avg_total_latency_ms."""
    cfg = _mock_cfg(tmp_path)
    run_dir = ExperimentRunner(cfg).run()
    cond_path = next(run_dir.glob("*/condition_summary_v2.json"))
    cond = json.loads(cond_path.read_text())

    # §97 M-6 new aggregate fields
    assert "avg_total_latency_ms" in cond, "M-6: avg_total_latency_ms missing"
    assert "avg_router_overhead_ms" in cond, "M-6: avg_router_overhead_ms missing"
    assert "avg_busy_wait_total_ms" in cond, "M-6: avg_busy_wait_total_ms missing"
    assert "energy_partial_episode_count" in cond, "M-6: energy_partial_episode_count missing"
    assert "energy_partial_episode_rate" in cond, "M-6: energy_partial_episode_rate missing"


def test_smoke_step_record_schema_valid(tmp_path):
    """Each step record passes StepRecordV2 schema validation."""
    cfg = _mock_cfg(tmp_path)
    run_dir = ExperimentRunner(cfg).run()
    steps_path = next(run_dir.glob("*/episodes/*_steps_v2.jsonl"))
    for line in steps_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        validate_step_record_v2(rec)  # raises on schema mismatch


def test_smoke_episode_summary_dataclass_compatible(tmp_path):
    """EpisodeSummaryV2 dataclass can re-construct from emitted summary
    (without losing non-required fields). Guards Step 6 schema migration."""
    cfg = _mock_cfg(tmp_path)
    run_dir = ExperimentRunner(cfg).run()
    summary_path = next(run_dir.glob("*/episodes/*_summary_v2.json"))
    summary = json.loads(summary_path.read_text())
    # Filter to dataclass fields (extras like wasted_cost_usd / agent_finished
    # are runner-added beyond the dataclass; we only validate the core schema).
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(EpisodeSummaryV2)}
    core = {k: v for k, v in summary.items() if k in field_names}
    obj = EpisodeSummaryV2(**core)
    # Round-trip must preserve required fields.
    assert obj.condition_id == summary["condition_id"]
    assert obj.task_id == summary["task_id"]
