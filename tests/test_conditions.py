from p79.experiment.conditions import generate_conditions


def _base_cfg(phase: str):
    return {
        "experiment": {"phase": phase},
        "variables": {
            "primary": {
                "observation_mode": ["dom", "som", "vision"],
            }
        },
        "backends": {
            "default_backend": "local_4b",
            "local_4b": {"type": "mock"},
        },
        "router": {"cheap_default_mode": "dom", "rich_escalation_mode": "som"},
        "baselines": {"run_b0": False},
    }


def test_phase1_has_3_mode_conditions():
    cfg = _base_cfg("phase1")
    conditions = generate_conditions(cfg)
    assert len(conditions) == 3
    assert all(not c.router_on for c in conditions)
    modes = {c.observation_mode for c in conditions}
    assert modes == {"dom", "som", "vision"}


def test_phase1_som_on_derived_correctly():
    cfg = _base_cfg("phase1")
    conditions = generate_conditions(cfg)
    for c in conditions:
        assert c.som_on == (c.observation_mode == "som"), f"som_on mismatch for mode={c.observation_mode}"


def test_phase2_has_fixed_and_routed_pair():
    cfg = _base_cfg("phase2")
    cfg["variables"]["phase2"] = {
        "fixed_condition": {"observation_mode": "som"}
    }
    conditions = generate_conditions(cfg)
    assert {c.condition_id for c in conditions} == {"phase2_fixed_best", "phase2_routed"}
    assert sum(1 for c in conditions if c.router_on) == 1


def test_phase3_enforces_one_module_at_a_time():
    cfg = _base_cfg("phase3")
    cfg["variables"]["phase3"] = {
        "base_condition": {
            "observation_mode": "dom",
            "router_on": True,
        }
    }
    conditions = generate_conditions(cfg)
    assert len(conditions) == 5

    for c in conditions:
        enabled = sum(1 for v in c.modules.as_dict().values() if v)
        assert enabled <= 1


def test_b0_is_added_when_enabled():
    """B-269 fix (2026-05-16, A1.7): `baselines.run_b0` is retired for Phase 1a
    3-baseline architecture but kept for Phase 2/3 paper-2 substrate. Test
    on phase2 since phase1 now raises on run_b0=True.
    """
    cfg = _base_cfg("phase2")
    cfg["baselines"] = {
        "run_b0": True,
        "b0_backend": "api_strong",
        "b0_observation_mode": "som",
    }
    cfg["backends"]["api_strong"] = {"type": "mock"}

    conditions = generate_conditions(cfg)
    assert any(c.condition_id == "b0_strong_upper_bound" for c in conditions)


def test_b0_run_b0_raises_in_phase1():
    """B-269 fix (2026-05-16, A1.7): paper-1 3-baseline architecture means
    `baselines.run_b0=True` in phase1 is a configuration error (would create
    duplicate b0_strong_upper_bound condition alongside per-baseline B0 cell).
    """
    import pytest as _pytest
    cfg = _base_cfg("phase1")
    cfg["baselines"] = {"run_b0": True, "b0_backend": "api_strong"}
    cfg["backends"]["api_strong"] = {"type": "mock"}
    with _pytest.raises(ValueError, match="baselines.run_b0=True is retired"):
        generate_conditions(cfg)
