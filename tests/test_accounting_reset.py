"""Protocol Reset #6/#7/#8 (§244 canonical, 2026-05-20): two-budget accounting
+ three-column cost regression tests.

Covers:
  - classify_step_accounting: WAIT sink / genuine action / policy-blocked goto
    (B-1782) / finish — the valid_agent_action vs consumes_agent_action_budget
    divergence is the load-bearing distinction (goto: valid=F, consumes=T).
  - compute_three_column_cost: additive invariant canonical + wasted == billed.
  - config budget defaults: max_agent_actions inherits max_steps (zero yaml
    churn) + max_model_attempts derives + explicit override honored.
  - MockEnv integration: the 3 step flags + 5 episode counters + 3 cost columns
    actually land in the written JSONL/summary and the additive invariant holds
    on a real runner episode (B0/B1/B2-agnostic — the runner stamps for all).
"""
import json

import pytest

from p79.experiment.metrics import (
    classify_step_accounting,
    compute_three_column_cost,
)


# ─── classify_step_accounting (§244 #6/#7) ──────────────────────────────────
@pytest.mark.parametrize(
    "parse_valid,action_type,goto_blocked,exp_valid,exp_consumes,exp_sink",
    [
        # genuine parse-valid action → valid + consumes + not a sink
        (True, "click", False, True, True, False),
        (True, "type", False, True, True, False),
        (True, "scroll", False, True, True, False),
        # finish/stop are genuine agent decisions
        (True, "finish", False, True, True, False),
        (True, "stop", False, True, True, False),
        # WAIT is never a canonical action (parse-fail / structural sink)
        (False, "wait", False, False, False, True),
        # an agent-emitted *valid* wait is still not a canonical action
        (True, "wait", False, False, False, True),
        # policy-blocked off-site goto (B-1782): parsed, NOT valid, BUT consumes
        # the turn → the one place valid != consumes
        (True, "goto", True, False, True, False),
    ],
)
def test_classify_step_accounting(
    parse_valid, action_type, goto_blocked, exp_valid, exp_consumes, exp_sink
):
    out = classify_step_accounting(
        parse_valid=parse_valid, action_type=action_type, goto_blocked=goto_blocked
    )
    assert out["valid_agent_action"] is exp_valid
    assert out["consumes_agent_action_budget"] is exp_consumes
    assert out["is_injected_wait_sink"] is exp_sink
    # counts_as_runner_iteration is always True at this call site (a model ran)
    assert out["counts_as_runner_iteration"] is True


def test_classify_policy_blocked_goto_is_the_only_valid_consumes_divergence():
    """The goto-blocked case is the sole branch where consumes != valid; every
    other branch keeps them equal. Guards against a future refactor collapsing
    the two flags into one (which would silently let off-site gotos escape the
    budget OR count parse-errors against it)."""
    goto = classify_step_accounting(parse_valid=True, action_type="goto", goto_blocked=True)
    assert goto["valid_agent_action"] != goto["consumes_agent_action_budget"]
    for at in ("click", "wait", "finish", "type"):
        for pv in (True, False):
            o = classify_step_accounting(parse_valid=pv, action_type=at, goto_blocked=False)
            assert o["valid_agent_action"] == o["consumes_agent_action_budget"]


# ─── compute_three_column_cost (§244 #8) ────────────────────────────────────
def _rec(model_cost, valid):
    return {"cost_usd": {"model": model_cost}, "valid_agent_action": valid}


def test_three_column_cost_additive_invariant():
    recs = [
        _rec(1.0, True),    # canonical
        _rec(0.5, False),   # parse-error sink → wasted
        _rec(0.3, False),   # policy-blocked goto → wasted
        _rec(0.7, True),    # canonical
    ]
    c = compute_three_column_cost(recs)
    assert c["total_billed_cost_usd"] == pytest.approx(2.5)
    assert c["canonical_action_cost_usd"] == pytest.approx(1.7)
    assert c["protocol_wasted_cost_usd"] == pytest.approx(0.8)
    # the load-bearing invariant
    assert c["canonical_action_cost_usd"] + c["protocol_wasted_cost_usd"] == pytest.approx(
        c["total_billed_cost_usd"]
    )


def test_three_column_cost_edge_cases():
    assert compute_three_column_cost([]) == {
        "total_billed_cost_usd": 0.0,
        "canonical_action_cost_usd": 0.0,
        "protocol_wasted_cost_usd": 0.0,
    }
    # all valid → wasted 0
    allv = compute_three_column_cost([_rec(1.0, True), _rec(2.0, True)])
    assert allv["canonical_action_cost_usd"] == pytest.approx(3.0)
    assert allv["protocol_wasted_cost_usd"] == pytest.approx(0.0)
    # all wasted → canonical 0
    allw = compute_three_column_cost([_rec(1.0, False), _rec(2.0, False)])
    assert allw["canonical_action_cost_usd"] == pytest.approx(0.0)
    assert allw["protocol_wasted_cost_usd"] == pytest.approx(3.0)


def test_three_column_cost_falls_back_to_input_plus_output():
    """When cost_usd.model absent (legacy rows), fall back to input + output."""
    recs = [{"cost_usd": {"input": 0.4, "output": 0.6}, "valid_agent_action": True}]
    c = compute_three_column_cost(recs)
    assert c["total_billed_cost_usd"] == pytest.approx(1.0)
    assert c["canonical_action_cost_usd"] == pytest.approx(1.0)


def test_three_column_cost_none_valid_flag_is_not_canonical():
    """A row with valid_agent_action absent/None (legacy vintage) is NOT counted
    as canonical — only an explicit True qualifies (avoids legacy rows silently
    inflating the canonical numerator)."""
    recs = [{"cost_usd": {"model": 1.0}}, {"cost_usd": {"model": 2.0}, "valid_agent_action": None}]
    c = compute_three_column_cost(recs)
    assert c["canonical_action_cost_usd"] == pytest.approx(0.0)
    assert c["protocol_wasted_cost_usd"] == pytest.approx(3.0)


# ─── config budget defaults (§244 #7) ───────────────────────────────────────
def test_config_budget_inherits_max_steps():
    from p79.experiment.config import normalize_config

    rt = normalize_config({"runtime": {"max_steps": 30}})["runtime"]
    assert rt["max_agent_actions"] == 30  # inherits max_steps (zero yaml churn)
    assert rt["max_model_attempts"] == 45  # 30 + 5 + 10
    assert rt["max_consecutive_parse_errors"] == 3
    assert rt["max_total_parse_errors"] == 5


def test_config_budget_base_default_and_explicit_override():
    from p79.experiment.config import normalize_config

    base = normalize_config({})["runtime"]
    assert base["max_agent_actions"] == 40 and base["max_model_attempts"] == 55
    over = normalize_config({"runtime": {"max_steps": 30, "max_agent_actions": 25}})["runtime"]
    assert over["max_agent_actions"] == 25 and over["max_model_attempts"] == 40


def test_config_max_model_attempts_never_cuts_before_primary_budget():
    """Safety ceiling MUST exceed the primary budget + parse cap so it can never
    terminate an episode before the agent has spent its action budget."""
    from p79.experiment.config import normalize_config

    for ms in (10, 30, 50, 100):
        rt = normalize_config({"runtime": {"max_steps": ms}})["runtime"]
        assert rt["max_model_attempts"] >= rt["max_agent_actions"] + rt["max_total_parse_errors"]


# ─── MockEnv integration: fields land + invariant holds on a real episode ───
def _minimal_cfg(tmp_path, site="classifieds"):
    task_file = tmp_path / f"{site}.json"
    with open(task_file, "w", encoding="utf-8") as f:
        json.dump(
            [{"task_id": 0, "intent": f"Test task for {site}", "sites": [site],
              "start_url": f"__{site.upper()}__/"}],
            f,
        )
    return {
        "experiment": {
            "name": "acct_test", "benchmark": "visualwebarena", "phase": "phase1",
            "seed": 42, "output_root": str(tmp_path / "results"), "run_id": "run_acct",
        },
        "task": {
            "include_sites": [site], "max_tasks_per_site": 1, "task_ids": {},
            "site_configs": {site: str(task_file)},
        },
        "env": {"type": "mock", "viewport_width": 320, "viewport_height": 240},
        "runtime": {"max_steps": 3, "resume": False},
        "variables": {"primary": {"observation_mode": ["dom"]}},
        "router": {
            "cheap_default_mode": "dom", "rich_escalation_mode": "som",
            "thresholds": {"dom_size_threshold": 12000, "unchanged_steps_trigger": 2,
                           "no_progress_steps_trigger": 2, "retry_limit": 1},
            "overhead_cost_per_ms": 0.0,
        },
        "metrics": {
            "cost": {"input_cost_per_1k": 0.001, "output_cost_per_1k": 0.002},
            "energy": {"enabled": False, "kwh_per_step": None, "co2e_kg_per_kwh": None},
        },
        "state_change": {"similarity_threshold": 0.95},
        "backends": {
            "default_backend": "local_4b",
            "local_4b": {"type": "local_qwen", "mock_mode": True, "dom_mode": "llm"},
        },
        "baselines": {"run_b0": False},
    }


def test_accounting_fields_land_in_runner_output(tmp_path):
    from p79.experiment.runner import ExperimentRunner

    runner = ExperimentRunner(_minimal_cfg(tmp_path))
    run_dir = runner.run()

    # step records carry the 3 two-budget flags (KEY-present, bool-valued)
    step_logs = list(run_dir.glob("*/episodes/*_steps_v2.jsonl"))
    assert step_logs, "no step logs"
    with open(step_logs[0], "r", encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    assert rows
    for r in rows:
        for k in ("valid_agent_action", "consumes_agent_action_budget", "counts_as_runner_iteration"):
            assert k in r, f"step missing {k}"
            assert isinstance(r[k], bool), f"{k} not bool: {r[k]!r}"
        # the runner only persists steps that ran a model call
        assert r["counts_as_runner_iteration"] is True

    # episode summary carries the 5 counters + 3 cost columns + invariant holds
    summary_path = next(run_dir.glob("*/episodes/*_summary_v2.json"))
    with open(summary_path, "r", encoding="utf-8") as f:
        s = json.load(f)
    for k in ("agent_action_step_count", "valid_action_step_count",
              "model_call_attempt_count", "runner_iteration_count",
              "parse_error_injected_wait_count"):
        assert k in s and isinstance(s[k], int), f"counter {k} missing/non-int"
    for k in ("total_billed_cost_usd", "canonical_action_cost_usd", "protocol_wasted_cost_usd"):
        assert k in s, f"cost col {k} missing"
        assert s[k] is not None, f"runner must stamp {k} (not None on a real fire)"

    # model_call_attempt_count == persisted steps; runner_iteration >= steps
    assert s["model_call_attempt_count"] == len(rows)
    assert s["runner_iteration_count"] >= len(rows)
    # additive invariant on REAL output
    assert s["canonical_action_cost_usd"] + s["protocol_wasted_cost_usd"] == pytest.approx(
        s["total_billed_cost_usd"]
    )
    # a clean mock run has no parse errors → agent_action == valid_action
    assert s["parse_error_injected_wait_count"] == 0
    assert s["agent_action_step_count"] == s["valid_action_step_count"]
