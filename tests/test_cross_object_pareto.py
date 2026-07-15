from __future__ import annotations

from scripts.analysis.cross_object_pareto import arm_stats, hindsight_oracle, union_solved


def _row(success: bool, tokens: float, latency_s: float, usd: float) -> dict:
    return {
        "success": success,
        "tokens": tokens,
        "latency_s": latency_s,
        "usd": usd,
        "cost_unit_basis": "api_usd",
    }


ARMS = {
    # task 1: only A succeeds; task 2: both succeed (B cheaper on tokens,
    # A cheaper on latency); task 3: neither succeeds (B token-cheapest).
    ("M0", "A"): {
        1: _row(True, 100.0, 10.0, 0.10),
        2: _row(True, 300.0, 5.0, 0.30),
        3: _row(False, 500.0, 50.0, 0.50),
    },
    ("M0", "B"): {
        1: _row(False, 80.0, 20.0, 0.08),
        2: _row(True, 200.0, 9.0, 0.20),
        3: _row(False, 400.0, 60.0, 0.40),
    },
}
SUBSET = [("M0", "A"), ("M0", "B")]


def test_oracle_picks_metric_cheapest_success():
    o = hindsight_oracle(ARMS, SUBSET, "tokens")
    # task1 -> A (only success), task2 -> B (200 < 300), task3 fallback -> B (400)
    assert o["n_solved"] == 2
    assert o["sr"] == 2 / 3
    assert o["picks"] == {1: "M0/A", 2: "M0/B", 3: "M0/B"}
    assert o["mean_tokens"] == (100.0 + 200.0 + 400.0) / 3


def test_oracle_metric_switch_changes_pick_not_sr():
    o = hindsight_oracle(ARMS, SUBSET, "latency_s")
    # task2 now prefers A (5s < 9s); SR unchanged.
    assert o["picks"][2] == "M0/A"
    assert o["sr"] == 2 / 3
    assert o["mean_latency_s"] == (10.0 + 5.0 + 50.0) / 3


def test_union_solved_is_success_union():
    assert union_solved(ARMS, SUBSET) == {1, 2}
    assert union_solved(ARMS, [("M0", "B")]) == {2}


def test_arm_stats_means():
    s = arm_stats(ARMS[("M0", "A")])
    assert s["n_tasks"] == 3
    assert s["sr"] == 2 / 3
    assert s["mean_tokens"] == 300.0
    assert s["cost_unit_basis"] == "api_usd"
