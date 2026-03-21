import math

from PIL import Image

from p79.backends.image_utils import encode_image_data_url
from p79.envs.vwa_wrapper import P79Observation
from p79.experiment.checklist_module import ChecklistManagerLite
from p79.experiment.energy_tracker import LightweightEnergyTracker
from p79.experiment.state_change import build_page_state, detect_page_state_change


def test_state_change_detection_uses_url_and_content():
    obs_before = P79Observation(text="[1] Home link\n[2] Search", url="http://mock.local/a")
    obs_after = P79Observation(text="[1] Checkout\n[2] Confirm", url="http://mock.local/b")

    state_before = build_page_state(obs_before, {"title": "A"})
    state_after = build_page_state(obs_after, {"title": "B"})

    ok, reasons, similarity = detect_page_state_change(
        state_before=state_before,
        state_after=state_after,
        action_type="CLICK",
    )
    assert ok is True
    assert "url_changed" in reasons
    assert "title_changed" in reasons
    assert similarity < 0.95


def test_checklist_lite_progress_updates():
    manager = ChecklistManagerLite("Search product and open detail and complete checkout", max_items=3)
    assert len(manager.task_checklist) >= 1
    assert len(manager.task_checklist) <= 3

    manager.update_after_action(action_success=True, error=None)
    status1 = manager.get_status()
    assert status1["in_progress"] + status1["completed"] >= 1

    manager.update_after_action(action_success=True, error=None)
    status2 = manager.get_status()
    assert status2["completed"] >= 1


def test_lightweight_energy_tracker_fixed_power():
    tracker = LightweightEnergyTracker(
        {
            "enabled": True,
            "fixed_power_watts": 100.0,
            "carbon_intensity_g_per_kwh": 500.0,
        }
    )
    energy = tracker.estimate_step(duration_seconds=10.0)

    expected_kwh = 100.0 * (10.0 / 3600.0) / 1000.0
    expected_co2_kg = expected_kwh * 500.0 / 1000.0
    assert math.isclose(float(energy["kwh"]), expected_kwh, rel_tol=1e-5)
    assert math.isclose(float(energy["co2e_kg"]), expected_co2_kg, rel_tol=1e-5)


def test_image_utils_compresses_to_payload_limit():
    image = Image.new("RGB", (3200, 2400), color="white")
    payload = encode_image_data_url(image=image, max_payload_bytes=200_000)

    assert str(payload["data_url"]).startswith("data:image/jpeg;base64,")
    assert int(payload["payload_bytes"]) <= 200_000
    assert int(payload["width"]) <= 3200
    assert int(payload["height"]) <= 2400
