from p79.experiment.conditions import generate_conditions


def _base_cfg(phase: str):
    return {
        "experiment": {"phase": phase},
        "variables": {
            "primary": {
                "som": [False, True],
                "observation_mode": ["dom_only", "hybrid"],
            }
        },
        "backends": {
            "default_backend": "local_4b",
            "local_4b": {"type": "mock"},
        },
        "router": {"cheap_default_mode": "dom_only"},
        "baselines": {"run_b0": False},
    }


def test_phase1_matrix_has_2x2_conditions():
    cfg = _base_cfg("phase1")
    conditions = generate_conditions(cfg)
    assert len(conditions) == 4
    assert all(not c.router_on for c in conditions)


def test_phase2_has_fixed_and_routed_pair():
    cfg = _base_cfg("phase2")
    cfg["variables"]["phase2"] = {
        "fixed_condition": {"som_on": True, "observation_mode": "hybrid"}
    }
    conditions = generate_conditions(cfg)
    assert {c.condition_id for c in conditions} == {"phase2_fixed_best", "phase2_routed"}
    assert sum(1 for c in conditions if c.router_on) == 1


def test_phase3_enforces_one_module_at_a_time():
    cfg = _base_cfg("phase3")
    cfg["variables"]["phase3"] = {
        "base_condition": {
            "som_on": True,
            "observation_mode": "dom_only",
            "router_on": True,
        }
    }
    conditions = generate_conditions(cfg)
    assert len(conditions) == 5

    for c in conditions:
        enabled = sum(1 for v in c.modules.as_dict().values() if v)
        assert enabled <= 1


def test_b0_is_added_when_enabled():
    cfg = _base_cfg("phase1")
    cfg["baselines"] = {
        "run_b0": True,
        "b0_backend": "api_strong",
        "b0_som": True,
        "b0_observation_mode": "hybrid",
    }
    cfg["backends"]["api_strong"] = {"type": "mock"}

    conditions = generate_conditions(cfg)
    assert any(c.condition_id == "b0_strong_upper_bound" for c in conditions)
