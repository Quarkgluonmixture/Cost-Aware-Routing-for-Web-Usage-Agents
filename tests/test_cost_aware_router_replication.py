from __future__ import annotations

import copy

import pytest

from scripts.analysis.cost_aware_router_replication import (
    FROZEN_TAU,
    assert_primary_threshold_frozen,
    build_reddit_fold_map,
    decide_cost_aware_mode,
    decide_frozen_primary_mode,
    validate_frozen_recipe_payload,
)
from scripts.analysis.router_offline_replay import DISPLAY_MODES


def _six(value: float) -> dict[str, float]:
    return {mode: value for mode in DISPLAY_MODES}


def _valid_recipe() -> dict:
    return {
        "cell_id": "B0_classifieds",
        "protocol": {
            "outer_folds": 5,
            "feature_assets_reused": "fold-local vectorizer + MI-18 selector from locked router",
            "lr_C": 1.0,
            "lr_max_iter": 2000,
            "random_seed": 42,
            "class_weight": None,
            "thresholds": [0.05, 0.10, 0.15],
            "threshold_comparator": ">=",
            "cost_for_decision": "fold-training mean total_billed_cost_usd by mode",
            "fallback": "fold-training best single by SR, then lower mean cost",
        },
        "curve": [
            {
                "threshold": 0.10,
                "success_rate_pct": 29.9107142857,
                "mean_cost_usd": 0.070513058,
            }
        ],
    }


def test_decision_selects_cheapest_mode_at_or_above_threshold():
    probabilities = _six(0.01)
    probabilities["dom"] = 0.10  # equality is eligible: comparator is >=
    probabilities["vision"] = 0.80
    costs = _six(0.20)
    costs["dom"] = 0.05
    costs["vision"] = 0.07
    assert decide_cost_aware_mode(probabilities, costs, 0.10, "som") == ("dom", False)


def test_decision_falls_back_when_no_mode_is_eligible():
    assert decide_cost_aware_mode(_six(0.09), _six(0.1), 0.10, "som") == ("som", True)


def test_decision_cost_tie_uses_locked_display_order():
    probabilities = _six(0.0)
    probabilities["dom"] = probabilities["som"] = 0.8
    costs = _six(1.0)
    costs["dom"] = costs["som"] = 0.1
    assert decide_cost_aware_mode(probabilities, costs, 0.5, "vision") == ("dom", False)


def test_primary_decision_has_no_reddit_threshold_argument():
    probabilities = _six(0.0)
    probabilities["vision"] = FROZEN_TAU
    costs = _six(1.0)
    costs["vision"] = 0.01
    assert decide_frozen_primary_mode(probabilities, costs, "dom") == ("vision", False)
    with pytest.raises(TypeError):
        decide_frozen_primary_mode(probabilities, costs, "dom", 0.15)  # type: ignore[call-arg]


def test_primary_threshold_guard_rejects_reddit_side_change():
    assert_primary_threshold_frozen(0.10)
    with pytest.raises(ValueError, match="recipe-frozen"):
        assert_primary_threshold_frozen(0.15)


def test_frozen_recipe_validator_rejects_source_or_comparator_drift():
    recipe = validate_frozen_recipe_payload(_valid_recipe())
    assert recipe.threshold == 0.10

    wrong_cell = copy.deepcopy(_valid_recipe())
    wrong_cell["cell_id"] = "B0_reddit"
    with pytest.raises(ValueError, match="must come from B0_classifieds"):
        validate_frozen_recipe_payload(wrong_cell)

    wrong_comparator = copy.deepcopy(_valid_recipe())
    wrong_comparator["protocol"]["threshold_comparator"] = ">"
    with pytest.raises(ValueError, match="threshold_comparator"):
        validate_frozen_recipe_payload(wrong_comparator)


def test_reddit_fold_map_is_seeded_deterministic_and_complete():
    task_ids = list(range(205))
    first = build_reddit_fold_map(task_ids)
    second = build_reddit_fold_map(reversed(task_ids))
    assert first == second
    assert set(first) == set(task_ids)
    assert set(first.values()) == {0, 1, 2, 3, 4}
    assert sorted(first.values()).count(0) == 41
