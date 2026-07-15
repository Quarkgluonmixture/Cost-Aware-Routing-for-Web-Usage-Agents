from __future__ import annotations

from scripts.analysis.router_pareto_analysis import (
    PolicyPoint,
    dominance_relation,
    pareto_frontier,
    select_cheapest_eligible_mode,
)


def point(name: str, cost: float, sr: float) -> PolicyPoint:
    return PolicyPoint(name, name, "test", cost, sr, 100, int(sr))


def test_strict_dominance_requires_both_axes_strictly_better():
    better = point("better", 0.05, 80.0)
    worse = point("worse", 0.10, 60.0)
    assert dominance_relation(better, worse) == "a_strictly_dominates_b"
    assert dominance_relation(worse, better) == "b_strictly_dominates_a"


def test_weak_dominance_allows_one_axis_tie():
    cheaper = point("cheaper", 0.05, 60.0)
    dearer = point("dearer", 0.10, 60.0)
    assert dominance_relation(cheaper, dearer) == "a_weakly_dominates_b"


def test_two_axis_tie_is_equivalent_and_both_points_remain_on_frontier():
    first = point("first", 0.05, 60.0)
    second = point("second", 0.05, 60.0)
    assert dominance_relation(first, second) == "equivalent"
    assert [row.policy_id for row in pareto_frontier([first, second])] == ["first", "second"]


def test_tradeoff_is_incomparable_and_both_points_remain_on_frontier():
    cheap_low = point("cheap_low", 0.05, 40.0)
    dear_high = point("dear_high", 0.10, 80.0)
    assert dominance_relation(cheap_low, dear_high) == "incomparable"
    assert {row.policy_id for row in pareto_frontier([cheap_low, dear_high])} == {
        "cheap_low",
        "dear_high",
    }


def test_dominated_point_is_removed_from_frontier():
    cheap_low = point("cheap_low", 0.05, 40.0)
    dear_high = point("dear_high", 0.10, 80.0)
    dominated = point("dominated", 0.12, 30.0)
    assert {row.policy_id for row in pareto_frontier([cheap_low, dear_high, dominated])} == {
        "cheap_low",
        "dear_high",
    }


def test_cheapest_eligible_threshold_and_fallback():
    probabilities = {mode: 0.2 for mode in ("dom", "som", "vision", "phantom_text", "phantom_prompt", "phantom_som")}
    probabilities["dom"] = 0.8
    probabilities["vision"] = 0.9
    costs = {mode: 0.2 for mode in probabilities}
    costs["dom"] = 0.05
    costs["vision"] = 0.07
    assert select_cheapest_eligible_mode(probabilities, costs, 0.5, "som") == ("dom", False)
    assert select_cheapest_eligible_mode(probabilities, costs, 0.95, "som") == ("som", True)
