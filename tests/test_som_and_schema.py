import pytest
from PIL import Image

from p79.envs.vwa_wrapper import P79Observation
from p79.experiment.som import apply_som, prepare_observation_for_mode
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


def test_phantom_som_strips_image_keeps_text(tmp_path):
    """Phantom-SoM (§25): same SOM_MARKS text as full SoM, but no image to model."""
    obs = P79Observation(
        text="[1] Search textbox\n[2] Submit button",
        image=Image.new("RGB", (200, 100), color="white"),
        obs_nodes_info={
            "1": {"union_bound": [10, 20, 80, 30]},
            "2": {"union_bound": [100, 20, 90, 30]},
        },
    )

    full_som = prepare_observation_for_mode(obs, "som", tmp_path, step_idx=0)
    phantom = prepare_observation_for_mode(obs, "phantom_som", tmp_path, step_idx=0)

    # Same textual SOM_MARKS content
    assert phantom.som_text == full_som.som_text
    assert phantom.mark_count == full_som.mark_count

    # But model receives NO image in phantom
    assert phantom.marked_image is None
    assert full_som.marked_image is not None


def test_phantom_som_text_differs_from_dom(tmp_path):
    """Phantom-SoM text should be wrapped in [SOM_MARKS] block; DOM passes raw obs.text."""
    obs = P79Observation(
        text="[1] link 'Home'\n[2] button 'Submit'",
        image=Image.new("RGB", (100, 100), color="white"),
        obs_nodes_info={"1": {"union_bound": [0, 0, 50, 20]}},
    )

    dom = prepare_observation_for_mode(obs, "dom", tmp_path, step_idx=0)
    phantom = prepare_observation_for_mode(obs, "phantom_som", tmp_path, step_idx=0)

    # DOM passes raw obs.text; phantom wraps in [SOM_MARKS] block
    assert "[SOM_MARKS]" in phantom.som_text
    assert "[SOM_MARKS]" not in dom.som_text
    # Both have no image visible to the model
    assert dom.marked_image is None
    assert phantom.marked_image is None


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
