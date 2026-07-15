from __future__ import annotations

import pytest

from scripts.analysis.router_offline_replay import DISPLAY_MODES
from scripts.analysis.router_prior_baselines import (
    DISCLAIMER,
    aggregate_episode_confidence,
    efficiency_metrics,
    majority_oracle_vote,
    performance_gap_recovered,
    render_markdown,
    simulate_cascade_task,
)


def test_performance_gap_recovered_uses_requested_offline_anchors():
    assert performance_gap_recovered(30.0, 20.0, 40.0) == pytest.approx(0.5)
    assert performance_gap_recovered(15.0, 20.0, 40.0) == pytest.approx(-0.25)
    assert performance_gap_recovered(20.0, 20.0, 20.0) is None


def test_cost_normalized_efficiency_is_signed_delta_ratio():
    got = efficiency_metrics(
        30.0,
        0.08,
        best_single_success_rate_pct=20.0,
        best_single_mean_cost_usd=0.06,
        oracle_success_rate_pct=40.0,
    )
    assert got["performance_gap_recovered"] == pytest.approx(0.5)
    assert got["delta_vs_best_single_sr_pp"] == pytest.approx(10.0)
    assert got["delta_vs_best_single_cost_usd"] == pytest.approx(0.02)
    assert got["delta_sr_pp_per_delta_usd"] == pytest.approx(500.0)


def test_majority_vote_ties_by_mean_distance_then_locked_order():
    assert majority_oracle_vote(
        ["dom", "dom", "som"], [0.4, 0.3, 0.01]
    ) == "dom"
    assert majority_oracle_vote(
        ["dom", "som"], [0.4, 0.1]
    ) == "som"
    assert majority_oracle_vote(
        ["dom", "phantom_som"], [0.2, 0.2]
    ) == "dom"


def test_episode_confidence_aggregation_matches_existing_semantics():
    steps = [
        {
            "confidence": {
                "mean_logprob": -0.2,
                "min_logprob": -1.0,
                "mean_margin": 2.0,
                "min_margin": 0.5,
            }
        },
        {
            "confidence": {
                "mean_logprob": -0.4,
                "min_logprob": -2.0,
                "mean_margin": 4.0,
                "min_margin": 0.2,
            }
        },
    ]
    got = aggregate_episode_confidence(steps)
    assert got == {
        "mean_logprob": pytest.approx(-0.3),
        "min_logprob": pytest.approx(-2.0),
        "mean_margin": pytest.approx(3.0),
        "min_margin": pytest.approx(0.2),
    }


def test_cascade_sums_all_executed_trajectories_and_uses_final_success():
    outcomes = {
        mode: {"cost_usd": float(index + 1), "success": mode == "vision"}
        for index, mode in enumerate(DISPLAY_MODES)
    }
    confidence = {mode: 0.9 for mode in DISPLAY_MODES}
    confidence["dom"] = 0.1
    row = simulate_cascade_task(
        outcomes,
        confidence,
        DISPLAY_MODES,
        threshold=0.5,
    )
    assert row["executed_modes"] == ["dom", "som"]
    assert row["total_billed_cost_usd"] == pytest.approx(3.0)
    assert row["final_mode"] == "som"
    assert row["success"] is False


def test_cascade_full_escalation_reaches_final_mode():
    outcomes = {
        mode: {"cost_usd": 1.0, "success": mode == DISPLAY_MODES[-1]}
        for mode in DISPLAY_MODES
    }
    row = simulate_cascade_task(
        outcomes,
        {mode: 0.0 for mode in DISPLAY_MODES},
        DISPLAY_MODES,
        threshold=1.0,
    )
    assert row["n_executed"] == len(DISPLAY_MODES)
    assert row["total_billed_cost_usd"] == pytest.approx(float(len(DISPLAY_MODES)))
    assert row["success"] is True


def test_markdown_starts_with_non_gate_banner():
    payload = {
        "points": [
            {
                "policy_id": "fixed_som",
                "label": "Always SoM",
                "category": "fixed",
                "mean_cost_usd": 0.1,
                "success_rate_pct": 30.0,
                "n_tasks": 10,
                "n_success": 3,
                "performance_gap_recovered": 0.0,
                "delta_sr_pp_per_delta_usd": None,
            },
            {
                "policy_id": "router_oof",
                "label": "OOF learned router",
                "category": "router",
                "mean_cost_usd": 0.11,
                "success_rate_pct": 20.0,
                "n_tasks": 10,
                "n_success": 2,
                "performance_gap_recovered": -0.5,
                "delta_sr_pp_per_delta_usd": -1000.0,
            },
            {
                "policy_id": "oracle",
                "label": "Oracle",
                "category": "hindsight_oracle",
                "mean_cost_usd": 0.09,
                "success_rate_pct": 50.0,
                "n_tasks": 10,
                "n_success": 5,
                "performance_gap_recovered": 1.0,
                "delta_sr_pp_per_delta_usd": -2000.0,
            },
            {
                "policy_id": "knn_k5",
                "label": "kNN",
                "category": "prior_knn",
                "mean_cost_usd": 0.1,
                "success_rate_pct": 25.0,
                "n_tasks": 10,
                "n_success": 2,
                "performance_gap_recovered": -0.25,
                "delta_sr_pp_per_delta_usd": None,
            },
            {
                "policy_id": "random_uniform",
                "label": "Random",
                "category": "random_noise_floor",
                "mean_cost_usd": 0.1,
                "success_rate_pct": 20.0,
                "n_tasks": 10,
                "n_success": 2,
                "success_rate_pct_sd": 1.0,
                "mean_cost_usd_sd": 0.01,
                "performance_gap_recovered": -0.5,
                "delta_sr_pp_per_delta_usd": None,
            },
        ],
        "pareto": {
            "deployable_frontier": ["fixed_som"],
            "hindsight_augmented_frontier": ["oracle"],
            "dominated_by": {
                "router_oof": [{"policy_id": "fixed_som", "type": "strict"}],
                "knn_k5": [],
            },
        },
        "knn": {"points": []},
        "random_noise_floors": {"points": []},
        "cascade": {
            "primary_signal": "mean_logprob",
            "curves": {
                "mean_logprob": [
                    {
                        "policy_id": "cascade_mean_logprob_q0.00",
                        "label": "cascade",
                        "tau_quantile": 0.0,
                        "success_rate_pct": 30.0,
                        "mean_cost_usd": 0.1,
                        "performance_gap_recovered": 0.0,
                        "mean_trajectories_executed": 1.0,
                        "escalated_task_fraction": 0.0,
                    }
                ]
            },
        },
        "confidence_probe": {
            "n_steps_with_confidence_dict": 10,
            "n_steps": 10,
            "step_confidence_coverage": 1.0,
            "n_task_mode_episodes": 10,
        },
    }
    assert render_markdown(payload).startswith(f"# {DISCLAIMER}\n")
