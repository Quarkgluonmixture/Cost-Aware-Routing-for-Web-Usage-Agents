import pytest
from PIL import Image

from p79.envs.vwa_wrapper import P79Observation
from p79.experiment.som import apply_som
from p79.experiment.types import SCHEMA_VERSION_V2, validate_step_record_v2


def test_som_degrades_without_bbox(tmp_path):
    obs = P79Observation(
        text="[1] Search textbox\n[2] Submit button",
        image=Image.new("RGB", (200, 100), color="white"),
        raw={"text": "no bbox here"},
    )

    result = apply_som(obs=obs, som_on=True, artifact_dir=tmp_path, step_idx=0)
    assert result.mark_count >= 1
    assert result.degraded_som is True


def test_step_schema_validation_required_fields():
    record = {
        "schema_version": SCHEMA_VERSION_V2,
        "run_id": "run_x",
        "condition_id": "c1",
        "benchmark": "visualwebarena",
        "benchmark_site": "shopping",
        "task_id": 0,
        "seed": 42,
        "step_idx": 0,
        "som": {},
        "observation_mode": "dom",
        "router": {},
        "module_flags": {},
        "action_type": "wait",
        "action": {"action_type": "wait"},
        "action_success": False,
        "page_changed": False,
        "latency_ms": {"total": 0.0},
        "tokens": {"input": 0, "output": 0, "total": 0},
        "cost_usd": {"total": 0.0},
        "energy": {"kwh": None, "co2e_kg": None},
        "retry_count": 0,
        "error_category": None,
        "artifact_paths": {},
        "reward": 0.0,
        "done": False,
    }

    validate_step_record_v2(record)

    broken = dict(record)
    broken.pop("router")
    with pytest.raises(ValueError):
        validate_step_record_v2(broken)
