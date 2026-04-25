import math

from p79.experiment.metrics import (
    aggregate_condition_metrics,
    compute_component_breakdown,
    compute_wasted_cost,
    detect_benchmark_noise,
    net_saving,
)
from p79.experiment.router import RouterState, RuleBasedRouter


# ---------------------------------------------------------------------------
# metrics: compute_wasted_cost
# ---------------------------------------------------------------------------


def test_wasted_cost_zero_on_success():
    steps = [
        {"cost_usd": {"total": 0.05}, "energy": {"kwh": 0.002}},
        {"cost_usd": {"total": 0.03}, "energy": {"kwh": 0.001}},
    ]
    result = compute_wasted_cost(steps, success=True)
    assert result["wasted_cost_usd"] == 0.0
    assert result["wasted_energy_kwh"] == 0.0


def test_wasted_cost_equals_total_on_failure():
    steps = [
        {"cost_usd": {"total": 0.05}, "energy": {"kwh": 0.002}},
        {"cost_usd": {"total": 0.03}, "energy": {"kwh": 0.001}},
    ]
    result = compute_wasted_cost(steps, success=False)
    assert math.isclose(result["wasted_cost_usd"], 0.08, rel_tol=1e-9)
    assert math.isclose(result["wasted_energy_kwh"], 0.003, rel_tol=1e-9)


def test_wasted_cost_empty_steps():
    assert compute_wasted_cost([], success=False) == {"wasted_cost_usd": 0.0, "wasted_energy_kwh": 0.0}
    assert compute_wasted_cost([], success=True) == {"wasted_cost_usd": 0.0, "wasted_energy_kwh": 0.0}


def test_wasted_cost_missing_fields():
    """Steps with missing cost_usd or energy keys should default to 0."""
    steps = [{"other": 1}, {"cost_usd": {"total": 0.01}}]
    result = compute_wasted_cost(steps, success=False)
    assert math.isclose(result["wasted_cost_usd"], 0.01, rel_tol=1e-9)
    assert result["wasted_energy_kwh"] == 0.0


# ---------------------------------------------------------------------------
# metrics: compute_component_breakdown
# ---------------------------------------------------------------------------


def test_component_breakdown_aggregates_correctly():
    steps = [
        {"cost_usd": {"model": 0.01, "router_overhead": 0.002}, "energy": {"kwh": 0.001}},
        {"cost_usd": {"model": 0.02, "router_overhead": 0.003}, "energy": {"kwh": 0.002}},
    ]
    bd = compute_component_breakdown(steps)
    assert math.isclose(bd["model_cost_usd"], 0.03, rel_tol=1e-9)
    assert math.isclose(bd["router_overhead_usd"], 0.005, rel_tol=1e-9)
    assert math.isclose(bd["total_energy_kwh"], 0.003, rel_tol=1e-9)


def test_component_breakdown_empty_steps():
    bd = compute_component_breakdown([])
    assert bd == {"model_cost_usd": 0.0, "router_overhead_usd": 0.0, "total_energy_kwh": 0.0}


def test_component_breakdown_missing_fields():
    steps = [{"cost_usd": {"model": 0.01}}, {}]
    bd = compute_component_breakdown(steps)
    assert math.isclose(bd["model_cost_usd"], 0.01, rel_tol=1e-9)
    assert bd["router_overhead_usd"] == 0.0
    assert bd["total_energy_kwh"] == 0.0


# ---------------------------------------------------------------------------
# metrics: aggregate_condition_metrics — new fields
# ---------------------------------------------------------------------------


def test_aggregate_empty_returns_new_fields():
    agg = aggregate_condition_metrics([])
    assert agg["avg_wasted_cost_usd"] == 0.0
    assert agg["avg_wasted_energy_kwh"] == 0.0
    assert agg["cost_efficiency_ratio"] is None


def test_aggregate_wasted_and_efficiency():
    episodes = [
        {
            "success": True,
            "total_cost_usd": 1.0,
            "wasted_cost_usd": 0.0,
            "wasted_energy_kwh": 0.0,
            "p95_step_latency_ms": 10.0,
            "steps": 2,
            "total_model_cost_usd": 0.9,
            "total_router_overhead_cost_usd": 0.1,
            "retries": 0,
            "no_op_rate": 0.0,
            "page_unchanged_rate": 0.0,
            "escalation_count": 0,
            "trigger_distribution": {},
            "state_change_reason_distribution": {},
        },
        {
            "success": False,
            "total_cost_usd": 2.0,
            "wasted_cost_usd": 2.0,
            "wasted_energy_kwh": 0.01,
            "p95_step_latency_ms": 20.0,
            "steps": 3,
            "total_model_cost_usd": 1.8,
            "total_router_overhead_cost_usd": 0.2,
            "retries": 0,
            "no_op_rate": 0.0,
            "page_unchanged_rate": 0.0,
            "escalation_count": 0,
            "trigger_distribution": {},
            "state_change_reason_distribution": {},
        },
    ]
    agg = aggregate_condition_metrics(episodes)
    # avg_wasted_cost_usd = mean(0.0, 2.0) = 1.0
    assert math.isclose(agg["avg_wasted_cost_usd"], 1.0, rel_tol=1e-9)
    # avg_wasted_energy_kwh = mean(0.0, 0.01) = 0.005
    assert math.isclose(agg["avg_wasted_energy_kwh"], 0.005, rel_tol=1e-9)
    # cost_efficiency_ratio = success_cost / total_cost = 1.0 / 3.0
    assert math.isclose(agg["cost_efficiency_ratio"], 1.0 / 3.0, rel_tol=1e-9)


def test_aggregate_all_success_efficiency_ratio_is_one():
    episodes = [
        {
            "success": True,
            "total_cost_usd": 1.0,
            "wasted_cost_usd": 0.0,
            "wasted_energy_kwh": 0.0,
            "p95_step_latency_ms": 5.0,
            "steps": 1,
            "total_model_cost_usd": 1.0,
            "total_router_overhead_cost_usd": 0.0,
            "retries": 0,
            "no_op_rate": 0.0,
            "page_unchanged_rate": 0.0,
            "escalation_count": 0,
            "trigger_distribution": {},
            "state_change_reason_distribution": {},
        },
    ]
    agg = aggregate_condition_metrics(episodes)
    assert math.isclose(agg["cost_efficiency_ratio"], 1.0, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# router: existing tests
# ---------------------------------------------------------------------------


def test_router_escalates_on_failure_signal():
    cfg = {
        "router": {
            "cheap_default_mode": "dom",
            "thresholds": {
                "dom_size_threshold": 12000,
                "unchanged_steps_trigger": 2,
                "no_progress_steps_trigger": 2,
            },
        }
    }
    router = RuleBasedRouter(cfg)
    state = RouterState()

    decision, triggers, _, state = router.decide(
        router_enabled=True,
        preferred_mode="som",
        obs_text="short text",
        state=state,
        prev_action_success=None,
        prev_page_changed=None,
    )
    assert decision == "dom"
    assert triggers == []

    decision2, triggers2, _, _ = router.decide(
        router_enabled=True,
        preferred_mode="som",
        obs_text="short text",
        state=state,
        prev_action_success=False,
        prev_page_changed=False,
    )
    assert decision2 == "som"
    assert "action_failed" in triggers2


def test_net_saving_formula():
    assert net_saving(10.0, 6.0, 1.0) == 3.0


def test_router_escalates_on_checklist_stall_when_enabled():
    cfg = {
        "router": {
            "cheap_default_mode": "dom",
            "thresholds": {
                "dom_size_threshold": 12000,
                "unchanged_steps_trigger": 2,
                "no_progress_steps_trigger": 2,
            },
            "checklist_trigger": {
                "enabled": True,
                "stalled_steps_trigger": 1,
                "failed_item_trigger": False,
            },
        }
    }
    router = RuleBasedRouter(cfg)
    state = RouterState()

    decision1, triggers1, _, state = router.decide(
        router_enabled=True,
        preferred_mode="som",
        obs_text="short text",
        state=state,
        prev_action_success=True,
        prev_page_changed=True,
        checklist_status={"total": 3, "completed": 0, "failed": 0},
    )
    assert decision1 == "dom"
    assert "checklist_progress_stalled" not in triggers1

    decision2, triggers2, _, _ = router.decide(
        router_enabled=True,
        preferred_mode="som",
        obs_text="short text",
        state=state,
        prev_action_success=True,
        prev_page_changed=True,
        checklist_status={"total": 3, "completed": 0, "failed": 0},
    )
    assert decision2 == "som"
    assert "checklist_progress_stalled" in triggers2


def test_aggregate_condition_metrics_includes_reason_distribution_and_checklist_metrics():
    episodes = [
        {
            "success": True,
            "p95_step_latency_ms": 10.0,
            "steps": 2,
            "total_model_cost_usd": 1.1,
            "total_cost_usd": 1.2,
            "total_router_overhead_cost_usd": 0.1,
            "total_energy_kwh": 0.01,
            "total_co2e_kg": 0.005,
            "retries": 0,
            "no_op_rate": 0.1,
            "page_unchanged_rate": 0.2,
            "escalation_count": 1,
            "trigger_distribution": {"action_failed": 2},
            "state_change_reason_distribution": {"content_changed": 2, "url_changed": 1},
            "checklist_completion_rate": 0.66,
            "checklist_failed_items": 0,
        },
        {
            "success": False,
            "p95_step_latency_ms": 20.0,
            "steps": 3,
            "total_model_cost_usd": 1.6,
            "total_cost_usd": 1.8,
            "total_router_overhead_cost_usd": 0.2,
            "total_energy_kwh": 0.02,
            "total_co2e_kg": 0.01,
            "retries": 1,
            "no_op_rate": 0.2,
            "page_unchanged_rate": 0.4,
            "escalation_count": 2,
            "trigger_distribution": {"action_failed": 1, "page_unchanged_streak": 3},
            "state_change_reason_distribution": {"content_changed": 1, "form_fields_changed": 3},
            "checklist_completion_rate": 0.33,
            "checklist_failed_items": 1,
        },
    ]
    agg = aggregate_condition_metrics(episodes)
    assert agg["avg_total_model_cost_usd"] == 1.35
    assert agg["trigger_distribution"]["action_failed"] == 3
    assert agg["trigger_distribution"]["page_unchanged_streak"] == 3
    assert agg["state_change_reason_distribution"]["content_changed"] == 3
    assert agg["state_change_reason_distribution"]["url_changed"] == 1
    assert agg["state_change_reason_distribution"]["form_fields_changed"] == 3
    assert agg["avg_checklist_completion_rate"] == 0.495
    assert agg["checklist_failure_episode_rate"] == 0.5


# ---------------------------------------------------------------------------
# router: dom_complexity_high / text_length_high triggers
# ---------------------------------------------------------------------------


def _make_router(extra_thresholds=None, modes=None):
    cfg = {
        "router": {
            "cheap_default_mode": "dom",
            "rich_escalation_mode": "som",
            "thresholds": {
                "dom_size_threshold": 50000,  # high — won't fire from obs_text alone
                "unchanged_steps_trigger": 99,
                "no_progress_steps_trigger": 99,
                "dom_complexity_trigger": 100,
                "text_length_trigger": 5000,
                "deescalation_streak": 3,
                "history_window": 5,
                **(extra_thresholds or {}),
            },
        }
    }
    if modes is not None:
        cfg["router"]["modes"] = modes
    return RuleBasedRouter(cfg)


def test_router_dom_complexity_high_trigger():
    router = _make_router()
    state = RouterState()
    state.dom_complexity_history = [200]  # > 100 threshold

    decision, triggers, _, state = router.decide(
        router_enabled=True,
        preferred_mode="dom",
        obs_text="short",
        state=state,
        prev_action_success=True,
        prev_page_changed=True,
    )
    assert "dom_complexity_high" in triggers
    assert decision == "som"


def test_router_text_length_high_trigger():
    router = _make_router()
    state = RouterState()
    state.text_length_history = [8000]  # > 5000 threshold

    decision, triggers, _, state = router.decide(
        router_enabled=True,
        preferred_mode="dom",
        obs_text="short",
        state=state,
        prev_action_success=True,
        prev_page_changed=True,
    )
    assert "text_length_high" in triggers
    assert decision == "som"


def test_router_no_complexity_trigger_below_threshold():
    router = _make_router()
    state = RouterState()
    state.dom_complexity_history = [50]   # < 100
    state.text_length_history = [3000]    # < 5000

    decision, triggers, _, _ = router.decide(
        router_enabled=True,
        preferred_mode="dom",
        obs_text="short",
        state=state,
        prev_action_success=True,
        prev_page_changed=True,
    )
    assert "dom_complexity_high" not in triggers
    assert "text_length_high" not in triggers
    assert decision == "dom"


# ---------------------------------------------------------------------------
# router: 3-way escalation (dom → som → vision)
# ---------------------------------------------------------------------------


def test_router_3way_escalation_step_by_step():
    router = _make_router(modes=["dom", "som", "vision"])
    state = RouterState()
    state.current_mode = "dom"

    # Step 1: trigger → dom→som
    state.dom_complexity_history = [200]
    decision, triggers, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=True, prev_page_changed=True,
    )
    assert decision == "som"
    assert state.current_mode == "som"

    # Step 2: trigger again → som→vision
    state.dom_complexity_history = [200]
    decision2, triggers2, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=True, prev_page_changed=True,
    )
    assert decision2 == "vision"
    assert state.current_mode == "vision"

    # Step 3: trigger again → stays at vision (already max)
    state.dom_complexity_history = [200]
    decision3, _, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=True, prev_page_changed=True,
    )
    assert decision3 == "vision"


def test_router_3way_deescalation():
    router = _make_router(
        extra_thresholds={"deescalation_streak": 2},
        modes=["dom", "som", "vision"],
    )
    state = RouterState()
    state.current_mode = "vision"
    state.success_streak = 2  # meets deescalation_streak=2

    # No triggers, success_streak >= deescalation_streak → vision→som
    decision, triggers, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=True, prev_page_changed=True,
    )
    assert triggers == []
    assert decision == "som"
    assert state.current_mode == "som"
    assert state.success_streak == 0  # reset after de-escalation


def test_router_deescalation_does_not_go_below_cheapest():
    router = _make_router(
        extra_thresholds={"deescalation_streak": 1},
        modes=["dom", "som", "vision"],
    )
    state = RouterState()
    state.current_mode = "dom"
    state.success_streak = 5  # high, but already at cheapest mode

    decision, _, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=True, prev_page_changed=True,
    )
    # Already at cheapest → stays dom (no deescalation branch entered)
    assert decision == "dom"


def test_router_success_streak_tracks_correctly():
    router = _make_router()
    state = RouterState()

    # 3 successes
    for _ in range(3):
        _, _, _, state = router.decide(
            router_enabled=True, preferred_mode="dom", obs_text="short",
            state=state, prev_action_success=True, prev_page_changed=True,
        )
    assert state.success_streak == 3

    # 1 failure resets
    _, _, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=False, prev_page_changed=True,
    )
    assert state.success_streak == 0


def test_router_current_mode_persists_across_calls():
    router = _make_router()
    state = RouterState()

    # First call: no triggers → dom
    _, _, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=True, prev_page_changed=True,
    )
    assert state.current_mode == "dom"

    # Trigger escalation → som
    state.dom_complexity_history = [200]
    _, _, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=True, prev_page_changed=True,
    )
    assert state.current_mode == "som"


def test_router_disabled_ignores_triggers_and_returns_preferred():
    router = _make_router(modes=["dom", "som", "vision"])
    state = RouterState()
    state.dom_complexity_history = [9999]

    decision, triggers, _, _ = router.decide(
        router_enabled=False, preferred_mode="vision", obs_text="short",
        state=state, prev_action_success=False, prev_page_changed=False,
    )
    # Triggers are still computed but decision follows preferred_mode
    assert "dom_complexity_high" in triggers
    assert decision == "vision"


def test_router_holds_mode_when_no_trigger_after_escalation():
    """After escalation, mode should be held (not reset to cheapest) when no triggers fire."""
    router = _make_router(modes=["dom", "som", "vision"])
    state = RouterState()
    state.current_mode = "dom"

    # Step 1: trigger → escalate dom→som
    state.dom_complexity_history = [200]
    decision, _, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=True, prev_page_changed=True,
    )
    assert decision == "som"
    assert state.current_mode == "som"

    # Step 2: no trigger, success_streak < deescalation_streak → should HOLD som
    state.dom_complexity_history = [50]  # below threshold
    state.text_length_history = [100]    # below threshold
    decision2, triggers2, _, state = router.decide(
        router_enabled=True, preferred_mode="dom", obs_text="short",
        state=state, prev_action_success=True, prev_page_changed=True,
    )
    assert triggers2 == []
    assert decision2 == "som", "Mode should be held, not reset to cheapest"
    assert state.current_mode == "som"


def test_router_backward_compat_default_modes():
    """Default config produces 2-way modes=[dom, som]."""
    router = RuleBasedRouter({})
    assert router.modes == ["dom", "som"]
    assert router.dom_complexity_trigger == 500
    assert router.deescalation_streak == 3
    assert router.history_window == 5


# ---------------------------------------------------------------------------
# metrics: detect_benchmark_noise
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# analysis: compute_adjusted_success — eval_fp program_html supplementary rule
# ---------------------------------------------------------------------------

from p79.experiment.analysis import compute_adjusted_success


def test_eval_fp_program_html_no_effective_action():
    """program_html + ~agent_finished + ~has_effective_action → E-FP (§95 simplified rule)."""
    ok, reason = compute_adjusted_success(
        69, "reddit", True,
        agent_finished=False, eval_type="program_html",
        has_effective_action=False,
    )
    assert ok is False
    assert reason == "eval_fp"


def test_eval_fp_program_html_effective_action_safe():
    """program_html + ~agent_finished + effective=True → NOT E-FP."""
    ok, reason = compute_adjusted_success(
        160, "reddit", True,
        agent_finished=False, eval_type="program_html",
        has_effective_action=True,
    )
    assert ok is True
    assert reason == ""


def test_eval_fp_program_html_backward_compat_none():
    """has_effective_action=None defaults True → NOT E-FP."""
    ok, reason = compute_adjusted_success(
        42, "reddit", True,
        agent_finished=False, eval_type="program_html",
        has_effective_action=None,
    )
    assert ok is True
    assert reason == ""


def test_no_visual_fp_layer():
    """Confirm visual_fp reason no longer exists (§95)."""
    ok, reason = compute_adjusted_success(
        100, "classifieds", True,
        agent_finished=True,
    )
    assert ok is True
    assert reason != "visual_fp"


def test_detect_benchmark_noise_site_infra_error():
    is_noise, label = detect_benchmark_noise(
        "site_infra_error: title='Osclass Error' detected at step 2 for task classifieds/155"
    )
    assert is_noise is True
    assert label == "site_infra_error"


def test_detect_benchmark_noise_api_infra():
    is_noise, label = detect_benchmark_noise(
        "503 Server Error for url: https://model-api.example.com/v1/chat"
    )
    assert is_noise is True
    assert label == "api_infra"


def test_detect_benchmark_noise_api_infra_execute_api():
    is_noise, label = detect_benchmark_noise(
        "ReadTimeout for url: https://execute-api.us-east-1.amazonaws.com/..."
    )
    assert is_noise is True
    assert label == "api_infra"


def test_detect_benchmark_noise_none_input():
    is_noise, label = detect_benchmark_noise(None)
    assert is_noise is False
    assert label is None


def test_detect_benchmark_noise_normal_error():
    is_noise, label = detect_benchmark_noise("some random agent error")
    assert is_noise is False
    assert label is None
