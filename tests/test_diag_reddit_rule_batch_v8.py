"""Regression tests for the reddit discover rule batch (ruleset 8-reddit-p41p46-b1890fix).

Each test pins the *reason* a rule fires or stays silent, because every rule in this
batch encodes a claim that was verified against the full 36-condition scan — and in
three cases contradicted a sub-agent's proposal. The tests exist so a later edit
cannot quietly re-introduce the rejected framing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "diag_pattern_match", REPO / "scripts" / "analysis" / "diag_pattern_match.py"
)
dpm = importlib.util.module_from_spec(_spec)
sys.modules["diag_pattern_match"] = dpm          # @dataclass needs the module registered
_spec.loader.exec_module(dpm)


def _step(idx=0, action_type="click", element_id=1, success=True, page_changed=True,
          url="http://localhost:9999/", err=None, input_image=0):
    return {
        "step_idx": idx,
        # Real step records carry action_type BOTH at top level and nested inside
        # `action` (verified against a live episode); helpers read either, so the
        # fixture must supply both or it silently tests a shape that never occurs.
        "action_type": action_type,
        "action": {"action_type": action_type, "element_id": element_id},
        "action_success": success,
        "page_changed": page_changed,
        "obs_url": url,
        "tokens": {"input_image": input_image},
        "locator_route_meta": {"error": err} if err else {"error": None},
    }


def _finish(idx=1, answer="done"):
    s = _step(idx=idx, action_type="finish", element_id=None)
    s["action"] = {"action_type": "finish", "answer": answer}
    return s



# --------------------------------------------------------------------------- #
# ruleset identity
# --------------------------------------------------------------------------- #

def test_ruleset_version_and_registry():
    assert dpm.RULESET_VERSION == "8-reddit-p41p46-b1890fix"
    for rid in ("P41", "P42", "P43", "P44", "P45", "P46"):
        assert rid in dpm.ALL_RULES, f"{rid} not registered"


# --------------------------------------------------------------------------- #
# B-1890: the dead-field guard must be gone from P35 / P39
# --------------------------------------------------------------------------- #

def test_b1890_p35_p39_no_longer_read_the_dead_field():
    """`effective_mutating_action_count` is never populated (always 0), so guarding on
    it was a no-op that made P35/P39 looser than their docstrings claimed."""
    src = (REPO / "scripts" / "analysis" / "diag_pattern_match.py").read_text()
    body = src[src.index("def check_p35"):src.index("def check_p40")]
    # Comments legitimately name the field (they explain why it was removed); only
    # executable lines must be free of it.
    code = [ln for ln in body.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")]
    offenders = [ln for ln in code if "effective_mutating_action_count" in ln]
    assert not offenders, \
        "P35/P39 must derive the mutation count from step records, not the dead " \
        f"schema slot; still referenced in: {offenders}"


def test_count_mutating_steps_requires_success_and_page_change():
    steps = [
        _step(0, "type", success=True, page_changed=True),      # counts
        _step(1, "click", success=True, page_changed=False),    # no page change
        _step(2, "click", success=False, page_changed=True),    # action failed
        _step(3, "scroll", success=True, page_changed=True),    # not a mutating type
    ]
    assert dpm._count_mutating_steps(steps) == 1


# --------------------------------------------------------------------------- #
# P33: reddit raw-image path added, classifieds pattern preserved
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("url", [
    "http://localhost:9980/oc-content/uploads/0/123.jpg",          # cls (pre-existing)
    "http://localhost:9999/submission_images/7c409257444b7.jpg",   # reddit (added)
    "http://localhost:9999/submission_images/abc.gif?x=1",
])
def test_p33_raw_image_regex_covers_both_sites(url):
    assert dpm.RAW_IMAGE_URL_RE.search(url), url


def test_p33_regex_still_rejects_ordinary_pages():
    for url in ("http://localhost:9999/f/food/60721", "http://localhost:9980/index.php?page=item"):
        assert not dpm.RAW_IMAGE_URL_RE.search(url), url


# --------------------------------------------------------------------------- #
# P41 — passive must_exclude FP (B-1889)
# --------------------------------------------------------------------------- #

_MUST_EXCLUDE_ONLY = {
    "eval": {"eval_types": ["program_html"],
             "program_html": [{"required_contents": {"must_exclude": ["IAmA"]}}]},
    "intent": "subscribe to all subreddits starting with i",
}


def test_p41_fires_even_when_the_agent_acted():
    """Regression on the corrected gate: 'passable by doing nothing' is a property of
    the eval shape, not of the trajectory. A mutation gate suppressed 12 of the 13
    true positives because every task-160 episode typed in the search box."""
    steps = [_step(0, "type", success=True, page_changed=True), _finish(1)]
    hits = dpm.check_p41(steps, {"success": True}, _MUST_EXCLUDE_ONLY, "dom")
    assert len(hits) == 1
    assert "derived mutating steps=1" in hits[0].detail


def test_p41_silent_when_eval_has_a_positive_check():
    cfg = {"eval": {"eval_types": ["program_html"],
                    "program_html": [{"required_contents": {"must_include": ["x"],
                                                            "must_exclude": ["y"]}}]}}
    assert dpm.check_p41([_finish()], {"success": True}, cfg, "dom") == []


def test_p41_is_success_side_only():
    assert dpm.check_p41([_finish()], {"success": False}, _MUST_EXCLUDE_ONLY, "dom") == []


# --------------------------------------------------------------------------- #
# P42 — multi-site task grounded in one site (B-1892)
# --------------------------------------------------------------------------- #

def test_p42_fires_when_a_declared_site_was_never_visited():
    cfg = {"sites": ["wikipedia", "reddit"]}
    steps = [_step(0, url="http://localhost:9999/f/x"), _finish(1)]
    hits = dpm.check_p42(steps, {"success": True}, cfg, "dom")
    assert len(hits) == 1 and "visited only 1 host" in hits[0].detail


def test_p42_silent_when_both_sites_visited():
    cfg = {"sites": ["wikipedia", "reddit"]}
    steps = [_step(0, url="http://localhost:9999/f/x"),
             _step(1, url="http://localhost:8888/wiki/Y"), _finish(2)]
    assert dpm.check_p42(steps, {"success": True}, cfg, "dom") == []


def test_p42_silent_on_single_site_tasks():
    assert dpm.check_p42([_step()], {"success": True}, {"sites": ["reddit"]}, "dom") == []


# --------------------------------------------------------------------------- #
# P43 — page-embedded visual info, mode withholds the screenshot
# --------------------------------------------------------------------------- #

_VIS_NO_REF = {"intent": "How many kirbies are in the image?", "image": None}


@pytest.mark.parametrize("mode", ["dom", "phantom_text", "phantom_prompt", "phantom_som"])
def test_p43_fires_only_for_modes_without_a_page_screenshot(mode):
    hits = dpm.check_p43([_step(0), _finish(1)], {"success": False}, _VIS_NO_REF, mode)
    assert len(hits) == 1


@pytest.mark.parametrize("mode", ["som", "vision"])
def test_p43_silent_for_screenshot_modes(mode):
    assert dpm.check_p43([_step(0), _finish(1)], {"success": False}, _VIS_NO_REF, mode) == []


def test_p43_silent_when_a_task_reference_image_exists():
    """That case belongs to P34; and reference images reach every mode
    (runner/main.py:2628-2631), so it is not a blindness signal."""
    cfg = dict(_VIS_NO_REF, image=["input_0.png"])
    assert dpm.check_p43([_step(0), _finish(1)], {"success": False}, cfg, "dom") == []


def test_p43_silent_when_image_tokens_actually_reached_the_model():
    steps = [_step(0, input_image=768), _finish(1)]
    assert dpm.check_p43(steps, {"success": False}, _VIS_NO_REF, "dom") == []


def test_p43_detail_states_the_measured_null_effect():
    """§387.10 measured dom→som on this exact task set at +0.00/+1.56/+0.00 pp, so the
    label must not read as 'unsolvable'. Five sub-agents proposed that framing."""
    hits = dpm.check_p43([_step(0), _finish(1)], {"success": False}, _VIS_NO_REF, "dom")
    assert "neutral label" in hits[0].detail
    assert hits[0].rule_name == "PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT"


# --------------------------------------------------------------------------- #
# P44 — hallucinated element reference
# --------------------------------------------------------------------------- #

def test_p44_fires_on_missing_union_bound():
    steps = [_step(0, err="obs_nodes_info missing union_bound"), _finish(1)]
    hits = dpm.check_p44(steps, {"success": False}, {}, "dom")
    assert len(hits) == 1 and hits[0].rule_name == "HALLUCINATED_ELEMENT_REF"


def test_p44_does_not_fire_on_walk_fail():
    """The two locator branches are orthogonal: walk_fail = element resolved but no
    actionable ancestor; missing union_bound = element absent from the observation."""
    steps = [_step(0, err="walk_fail:no_actionable_within_walk"), _finish(1)]
    assert dpm.check_p44(steps, {"success": False}, {}, "dom") == []


def test_p44_success_safe():
    steps = [_step(0, err="obs_nodes_info missing union_bound")]
    assert dpm.check_p44(steps, {"success": True}, {}, "dom") == []


# --------------------------------------------------------------------------- #
# P45 — identical failed action streak
# --------------------------------------------------------------------------- #

def test_p45_fires_at_three_consecutive_identical_failures():
    steps = [_step(i, element_id=42, success=False,
                   err="walk_fail:no_actionable_within_walk") for i in range(3)]
    hits = dpm.check_p45(steps, {"success": False}, {}, "dom")
    assert len(hits) == 1 and "repeated 3x" in hits[0].detail


def test_p45_silent_below_threshold():
    steps = [_step(i, element_id=42, success=False, err="walk_fail:x") for i in range(2)]
    assert dpm.check_p45(steps, {"success": False}, {}, "dom") == []


def test_p45_streak_broken_by_a_different_target():
    steps = [_step(0, element_id=42, success=False, err="e"),
             _step(1, element_id=99, success=False, err="e"),
             _step(2, element_id=42, success=False, err="e")]
    assert dpm.check_p45(steps, {"success": False}, {}, "dom") == []


def test_p45_ignores_successful_repeats():
    steps = [_step(i, element_id=42, success=True) for i in range(5)]
    assert dpm.check_p45(steps, {"success": False}, {}, "dom") == []


# --------------------------------------------------------------------------- #
# P46 — comment intent never committed text
# --------------------------------------------------------------------------- #

_CMT = {"intent": "Reply to the post with my comment 'hello'"}


def test_p46_fires_when_the_answer_went_into_finish_instead_of_a_type():
    steps = [_step(0, "click"), _finish(1, answer="hello")]
    hits = dpm.check_p46(steps, {"success": False}, _CMT, "vision")
    assert len(hits) == 1 and hits[0].rule_name == "COMMENT_INTENT_NO_TYPE"


def test_p46_silent_when_text_was_committed():
    steps = [_step(0, "type", success=True), _finish(1, answer="hello")]
    assert dpm.check_p46(steps, {"success": False}, _CMT, "vision") == []


def test_p46_silent_without_a_finish_step():
    """Budget exhaustion is P31's domain, not an action-modality mismatch."""
    steps = [_step(i, "click") for i in range(5)]
    assert dpm.check_p46(steps, {"success": False}, _CMT, "vision") == []


def test_p46_intent_regex_stays_narrow():
    """Widening to any mutation verb erases the effect it encodes: comment/reply-intent
    tasks run 2.11% vs 8.49% (4.0x, 18/18 cells), but the broad mutation set is
    7.23% vs 6.01% — no gap (§387.8)."""
    assert dpm.COMMENT_INTENT_RE.search("Reply to the post")
    assert dpm.COMMENT_INTENT_RE.search("post a comment saying hi")
    for negative in ("Create a post about cats", "Upvote the top submission",
                     "Subscribe to r/food", "Edit my biography"):
        assert not dpm.COMMENT_INTENT_RE.search(negative), negative
