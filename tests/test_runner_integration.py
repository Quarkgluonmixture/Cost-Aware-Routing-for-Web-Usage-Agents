import json
from pathlib import Path

from p79.experiment.analysis import analyze_run
from p79.experiment.runner import ExperimentRunner
from p79.experiment.types import validate_step_record_v2


def _write_site_tasks(path: Path, site: str, task_id: int):
    payload = [
        {
            "task_id": task_id,
            "intent": f"Test task for {site}",
            "sites": [site],
            "start_url": f"__{site.upper()}__/",
        }
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


def test_runner_and_analysis_end_to_end_with_mock_env(tmp_path):
    shopping_file = tmp_path / "shopping.json"
    reddit_file = tmp_path / "reddit.json"
    classifieds_file = tmp_path / "classifieds.json"

    _write_site_tasks(shopping_file, "shopping", 0)
    _write_site_tasks(reddit_file, "reddit", 1)
    _write_site_tasks(classifieds_file, "classifieds", 2)

    cfg = {
        "experiment": {
            "name": "test_run",
            "benchmark": "visualwebarena",
            "phase": "phase1",
            "seed": 42,
            "output_root": str(tmp_path / "results"),
            "run_id": "run_test",
        },
        "task": {
            "include_sites": ["shopping", "reddit", "classifieds"],
            "max_tasks_per_site": 1,
            "task_ids": {},
            "site_configs": {
                "shopping": str(shopping_file),
                "reddit": str(reddit_file),
                "classifieds": str(classifieds_file),
            },
        },
        "env": {
            "type": "mock",
            "viewport_width": 320,
            "viewport_height": 240,
        },
        "runtime": {
            "max_steps": 3,
            "resume": False,
        },
        "variables": {
            "primary": {
                "observation_mode": ["dom", "som", "vision"],
            }
        },
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
            "cost": {
                "input_cost_per_1k": 0.0,
                "output_cost_per_1k": 0.0,
            },
            "energy": {
                "enabled": False,
                "kwh_per_step": None,
                "co2e_kg_per_kwh": None,
            },
        },
        "checklist": {
            "enabled": True,
            "inject_into_prompt": True,
            "max_items": 3,
        },
        "state_change": {
            "similarity_threshold": 0.95,
        },
        "backends": {
            "default_backend": "local_4b",
            "local_4b": {
                "type": "local_qwen",
                "mock_mode": True,
                "dom_mode": "heuristic",
            },
            "api_strong": {
                "type": "api_proxy",
                "mock_mode": True,
                "dom_mode": "heuristic",
            },
        },
        # B-269 fix (2026-05-16, A1.7): baselines.run_b0=True retired for phase1.
        # The integration test runs on phase1; B0 is now a top-level baseline via
        # per-condition yaml, not an additive flag. Remove this block.
        "baselines": {
            "run_b0": False,
        },
    }

    runner = ExperimentRunner(cfg)
    run_dir = runner.run()

    run_summary_path = run_dir / "run_summary_v2.json"
    assert run_summary_path.exists()

    with open(run_summary_path, "r", encoding="utf-8") as f:
        run_summary = json.load(f)

    assert run_summary["total_conditions"] == 3  # 3 phase1 modes (b0 upper-bound removed B-269 2026-05-16, A1.7)
    assert run_summary["total_episodes"] > 0

    step_logs = list(run_dir.glob("*/episodes/*_steps_v2.jsonl"))
    assert step_logs, "No step logs generated"

    first_step_log = step_logs[0]
    with open(first_step_log, "r", encoding="utf-8") as f:
        first_record = json.loads(f.readline())
    validate_step_record_v2(first_record)
    assert "page_change_reasons" in first_record
    assert "text_similarity" in first_record
    assert first_record.get("checklist") is not None
    assert "state_digest" in first_record

    # Verify state_digest contains new fields
    state_digest = first_record.get("state_digest", {})
    assert "dom_complexity" in state_digest
    assert "text_length" in state_digest

    first_summary_path = next(run_dir.glob("*/episodes/*_summary_v2.json"))
    with open(first_summary_path, "r", encoding="utf-8") as f:
        first_summary = json.load(f)
    assert "total_model_cost_usd" in first_summary
    assert "state_change_reason_distribution" in first_summary
    assert "checklist_completion_rate" in first_summary

    # Verify wasted cost and component breakdown in episode summary
    assert "wasted_cost_usd" in first_summary
    assert "wasted_energy_kwh" in first_summary
    assert "component_breakdown" in first_summary
    assert isinstance(first_summary["component_breakdown"], dict)
    assert "model_cost_usd" in first_summary["component_breakdown"]

    # Verify aggregate has new fields
    condition_summary_path = next(run_dir.glob("*/condition_summary_v2.json"))
    with open(condition_summary_path, "r", encoding="utf-8") as f:
        cond_summary = json.load(f)
    assert "avg_wasted_cost_usd" in cond_summary
    assert "avg_wasted_energy_kwh" in cond_summary
    assert "cost_efficiency_ratio" in cond_summary

    try:
        analysis_dir = analyze_run(str(run_dir))
    except RuntimeError as exc:
        assert "requires pandas and matplotlib" in str(exc)
    else:
        assert (analysis_dir / "results" / "_overview" / "tables" / "phase1_representation_screening.csv").exists()
