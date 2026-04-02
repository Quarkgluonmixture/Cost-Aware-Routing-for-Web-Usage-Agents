from p79.experiment.metrics import aggregate_condition_metrics, net_saving
from p79.experiment.router import RouterState, RuleBasedRouter


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
