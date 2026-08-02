"""P47 / P48 — the two rules landed from the WA reddit Tier-2/3 round (2026-08-02).

Both were validated success-safe over the full 624-episode WA cell before landing, and two
sibling candidates from the same round were rejected on exactly that test (R2 fired on 17% of
successes, R4 on 36%). These tests pin the properties that made them landable, so a later
loosening has to break a test rather than a habit.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis import diag_pattern_match as D  # noqa: E402


def _step(idx, action_type, url, **extra):
    s = {"step_idx": idx, "action_type": action_type, "obs_url": url}
    s.update(extra)
    return s


def _finish(idx, url, answer=""):
    return {"step_idx": idx, "action_type": "finish", "obs_url": url,
            "action": {"action_type": "finish", "answer": answer}}


# --------------------------------------------------------------------------- P47
FORM_URL = "http://localhost:9999/submit/AskReddit"
PAGE_URL = "http://localhost:9999/f/AskReddit"


def test_p47_fires_on_type_then_finish_while_on_form():
    steps = [_step(0, "click", PAGE_URL), _step(1, "type", FORM_URL), _finish(2, FORM_URL)]
    hits = D.check_p47(steps, {"success": False}, {}, "")
    assert len(hits) == 1 and hits[0].rule_id == "P47"


def test_p47_silent_when_a_click_follows_the_type():
    """A click after the type is the submit attempt; the mechanism does not apply."""
    steps = [_step(0, "type", FORM_URL), _step(1, "click", FORM_URL), _finish(2, FORM_URL)]
    assert D.check_p47(steps, {"success": False}, {}, "") == []


def test_p47_silent_off_form_routes():
    steps = [_step(0, "type", PAGE_URL), _finish(1, PAGE_URL)]
    assert D.check_p47(steps, {"success": False}, {}, "") == []


def test_p47_silent_on_success():
    steps = [_step(0, "type", FORM_URL), _finish(1, FORM_URL)]
    assert D.check_p47(steps, {"success": True}, {}, "") == []


# --------------------------------------------------------------------------- P48
SEARCH_URL = "http://localhost:9999/search?q=Hrekires"


def test_p48_fires_on_short_search_then_negative_assertion():
    steps = [_step(0, "type", SEARCH_URL),
             _finish(1, SEARCH_URL, "No submissions found for that user.")]
    hits = D.check_p48(steps, {"success": False}, {}, "")
    assert len(hits) == 1 and hits[0].rule_id == "P48"


def test_p48_silent_above_the_step_bound():
    """The four-step bound is the only thing separating this from 'searched and was right'."""
    steps = [_step(i, "click", SEARCH_URL) for i in range(5)]
    steps.append(_finish(5, SEARCH_URL, "No submissions found."))
    assert D.check_p48(steps, {"success": False}, {}, "") == []


def test_p48_silent_without_a_search():
    steps = [_step(0, "click", PAGE_URL), _finish(1, PAGE_URL, "No submissions found.")]
    assert D.check_p48(steps, {"success": False}, {}, "") == []


def test_p48_silent_when_the_answer_is_not_negative():
    steps = [_step(0, "type", SEARCH_URL), _finish(1, SEARCH_URL, "Found three posts.")]
    assert D.check_p48(steps, {"success": False}, {}, "") == []


# --------------------------------------------------------------------------- registry
def test_both_rules_are_registered_and_version_bumped():
    assert "P47" in D.ALL_RULES and "P48" in D.ALL_RULES
    # Adding rules without bumping would let mixed-ruleset scans aggregate silently;
    # aggregate_conditional_failure_attribution asserts a single version and would raise.
    assert D.RULESET_VERSION.startswith("9-"), D.RULESET_VERSION


def test_rejected_siblings_did_not_sneak_in():
    """R2 / R4 were rejected on the success side. Keep them out by name."""
    names = {getattr(fn, "__name__", "") for fn in D.ALL_RULES.values()}
    assert not any(n.endswith(("p49", "p50")) for n in names), (
        "a new rule appeared without going through the success-safe + three-site check")
