"""B-1891 — a stuck episode must be countable, without disturbing what already landed.

Measured on `B2_phantom_text_reddit_20260720` task 103 (31 steps): 29 steps
carried `locator_route_meta.error = walk_fail:*`, yet `action_success` was True
31/31 and `page_changed` True 31/31 (29 of them `scroll_changed`), so
`trigger_distribution` came out `{}`. An episode that spent its entire budget
failing to reach its target recorded no signal at all.

The fix is additive on purpose. `action_success` keeps its meaning and values —
it feeds the agent-visible FAILED line in `format_history()` and every landed
step record — and the existing trigger names keep their exact counting, because
the WA cross-benchmark arm is collected after this change and reusing a key
would silently make the two arms incomparable. The new signal gets a new field
and a new trigger name.
"""

from __future__ import annotations

import pytest

from p79.experiment.router import RouterState, RuleBasedRouter
from p79.experiment.runner.helpers import _action_intent_fulfilled


# --------------------------------------------------------------------------- #
# the classifier
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("err", [
    "walk_fail:no_actionable_within_walk",
    "walk_fail:no_input_within_walk",
    "obs_nodes_info missing union_bound",
])
def test_locator_failure_means_intent_unfulfilled_even_when_action_succeeded(err):
    assert _action_intent_fulfilled(True, {"error": err}) is False


def test_clean_step_is_fulfilled():
    assert _action_intent_fulfilled(True, {"error": None}) is True
    assert _action_intent_fulfilled(True, {}) is True
    assert _action_intent_fulfilled(True, None) is True


def test_outright_failure_is_unfulfilled_regardless_of_locator():
    assert _action_intent_fulfilled(False, None) is False
    assert _action_intent_fulfilled(False, {"error": None}) is False


def test_unknown_locator_error_does_not_count_as_unfulfilled():
    """The marker list is an allowlist, not a truthy check on the error string.
    A new locator error class should have to be classified deliberately — the
    alternative is that adding any diagnostic note silently starts suppressing
    the signal this field exists to carry."""
    assert _action_intent_fulfilled(True, {"error": "some_new_dispatch_note"}) is True


# --------------------------------------------------------------------------- #
# the trigger
# --------------------------------------------------------------------------- #

def _router() -> RuleBasedRouter:
    return RuleBasedRouter({"router": {"thresholds": {"no_progress_steps_trigger": 2}}})


def _decide(r, st, *, success=True, changed=True, fulfilled=None):
    return r.decide(
        router_enabled=False, preferred_mode="dom", obs_text="x", state=st,
        prev_action_success=success, prev_page_changed=changed,
        prev_action_intent_fulfilled=fulfilled,
    )


def test_the_task_103_shape_now_fires_a_trigger():
    """Exactly the observed field combination: success True, page changed True
    (scroll), locator saying the target was unreachable."""
    r, st = _router(), RouterState()
    fired = []
    for _ in range(4):
        _, triggers, _, st = _decide(r, st, success=True, changed=True, fulfilled=False)
        fired.append(triggers)
    assert "intent_unfulfilled_streak" in fired[-1]
    # and the pre-existing triggers stay silent, because their inputs are unchanged
    assert "no_progress_streak" not in fired[-1]
    assert "page_unchanged_streak" not in fired[-1]
    assert "action_failed" not in fired[-1]


def test_streak_resets_on_a_fulfilled_step():
    r, st = _router(), RouterState()
    for _ in range(3):
        _decide(r, st, fulfilled=False)
    assert st.intent_unfulfilled_streak == 3
    _decide(r, st, fulfilled=True)
    assert st.intent_unfulfilled_streak == 0


def test_none_neither_increments_nor_resets():
    """First step of an episode has no previous step; it must not be read as
    either evidence."""
    r, st = _router(), RouterState()
    _decide(r, st, fulfilled=False)
    _decide(r, st, fulfilled=None)
    assert st.intent_unfulfilled_streak == 1


def test_existing_trigger_semantics_are_untouched():
    """The comparability guarantee, asserted rather than asserted-in-prose: with
    the new argument absent, every pre-existing streak and trigger behaves
    exactly as before."""
    r, st = _router(), RouterState()
    for _ in range(3):
        _, triggers, _, st = r.decide(
            router_enabled=False, preferred_mode="dom", obs_text="x", state=st,
            prev_action_success=False, prev_page_changed=False,
        )
    assert st.no_progress_streak == 3
    assert st.unchanged_streak == 3
    assert st.intent_unfulfilled_streak == 0
    assert "no_progress_streak" in triggers
    assert "page_unchanged_streak" in triggers
    assert "action_failed" in triggers
    assert "intent_unfulfilled_streak" not in triggers


def test_threshold_defaults_to_the_no_progress_threshold_but_is_separable():
    assert _router().intent_unfulfilled_steps_trigger == 2
    r = RuleBasedRouter({"router": {"thresholds": {
        "no_progress_steps_trigger": 2, "intent_unfulfilled_steps_trigger": 5,
    }}})
    assert r.intent_unfulfilled_steps_trigger == 5
    st = RouterState()
    for _ in range(4):
        _, triggers, _, st = _decide(r, st, fulfilled=False)
    assert "intent_unfulfilled_streak" not in triggers
    _, triggers, _, st = _decide(r, st, fulfilled=False)
    assert "intent_unfulfilled_streak" in triggers


def test_action_success_semantics_were_not_changed():
    """Guard the decision itself. Flipping `action_success` on walk_fail was the
    more natural fix and was rejected: it feeds the agent-visible FAILED line in
    `format_history()`, so it would change trajectories, and it would silently
    re-file every landed step record's meaning. If a later change wants it, that
    is an estimand-adjacent move needing its own witness — not a quiet edit."""
    import inspect

    from p79.experiment.runner import main as runner_main

    src = inspect.getsource(runner_main)
    marker = 'step_record["action_intent_fulfilled"] = _action_intent_fulfilled('
    assert marker in src, "the additive field is gone — B-1891 regressed"
    assert "action_success=action_success" in src, (
        "action_success is no longer passed through unmodified"
    )
