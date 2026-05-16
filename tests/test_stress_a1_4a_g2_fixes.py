"""Invariant tests for /stress A1.4a v8 Commit G2 (B-167) — invalid_action
sub-category 细分 + unknown_failure bucket + paper §3.5 informative error
taxonomy.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from p79.backends.action_utils import (
    validate_action,
    validate_action_detailed,
)
from p79.experiment.runner.main import ExperimentRunner


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# validate_action_detailed: sub-category failure_reason emission
# ---------------------------------------------------------------------------


def test_validate_detailed_emits_invalid_schema_dict_for_non_dict():
    """Non-dict input → ``invalid_schema_dict`` reason."""
    action, valid, reason = validate_action_detailed("not a dict")
    assert valid is False
    assert reason == "invalid_schema_dict"
    assert action == {"action_type": "wait"}


def test_validate_detailed_emits_invalid_action_type_for_unknown_action():
    """Unknown action_type (e.g. agent emits 'clik' or 'tap') → specific
    reason so router can apply prompt-level fix (no mode change needed)."""
    action, valid, reason = validate_action_detailed({"action_type": "clik"})
    assert valid is False
    assert reason == "invalid_action_type"


def test_validate_detailed_emits_invalid_element_id_for_click_missing_target():
    """Click without element_id and without valid coord → invalid_element_id.
    Router could route this to SoM mode (explicit `[N]` listing) as
    follow-up policy enhancement (deferred to paper-2 scope)."""
    action, valid, reason = validate_action_detailed({"action_type": "click"})
    assert valid is False
    assert reason == "invalid_element_id"


def test_validate_detailed_emits_invalid_coord_for_malformed_coord():
    """Coord supplied but malformed (NaN / out-of-shape) → invalid_coord.
    Router could route this to vision mode (visual reposition) as policy
    enhancement."""
    action, valid, reason = validate_action_detailed(
        {"action_type": "click", "coordinate": [float("nan"), 0.5]}
    )
    assert valid is False
    assert reason == "invalid_coord"


def test_validate_detailed_emits_invalid_select_option_when_no_option_field():
    """select_option with element_id but no option_{label,value,index} →
    invalid_select_option (not catch-all invalid_action)."""
    action, valid, reason = validate_action_detailed(
        {"action_type": "select_option", "element_id": 5}
    )
    assert valid is False
    assert reason == "invalid_select_option"


def test_validate_detailed_returns_none_reason_on_valid():
    """Valid action → reason=None (not empty string, not absent)."""
    action, valid, reason = validate_action_detailed(
        {"action_type": "click", "element_id": 5}
    )
    assert valid is True
    assert reason is None


# ---------------------------------------------------------------------------
# Backward-compat 2-tuple wrapper
# ---------------------------------------------------------------------------


def test_validate_action_backward_compat_returns_2tuple():
    """``validate_action`` (legacy 2-tuple) must keep its signature so 20+
    existing callers (tests, proxy_api_agent, runner) don't break."""
    result = validate_action({"action_type": "click", "element_id": 3})
    assert isinstance(result, tuple)
    assert len(result) == 2
    action, valid = result  # must unpack to 2
    assert valid is True


def test_validate_action_backward_compat_on_invalid():
    """Same 2-tuple shape for invalid actions."""
    action, valid = validate_action({"action_type": "tap"})  # tap is not allowed
    assert valid is False
    assert action == {"action_type": "wait"}


# ---------------------------------------------------------------------------
# _normalize_error_category: 7+ category mapping
# ---------------------------------------------------------------------------


def test_normalize_invalid_action_type_category():
    """``invalid_action_type`` reason → ``invalid_action_type`` category."""
    cat = ExperimentRunner._normalize_error_category(
        failure_reason="invalid_action_type",
        action_success=False, page_changed=False,
    )
    assert cat == "invalid_action_type"


def test_normalize_invalid_element_id_category():
    cat = ExperimentRunner._normalize_error_category(
        failure_reason="invalid_element_id",
        action_success=False, page_changed=False,
    )
    assert cat == "invalid_element_id"


def test_normalize_invalid_coord_category():
    cat = ExperimentRunner._normalize_error_category(
        failure_reason="invalid_coord",
        action_success=False, page_changed=False,
    )
    assert cat == "invalid_coord"


def test_normalize_invalid_select_option_category():
    cat = ExperimentRunner._normalize_error_category(
        failure_reason="invalid_select_option",
        action_success=False, page_changed=False,
    )
    assert cat == "invalid_select_option"


def test_normalize_invalid_schema_for_dict_shape_gap():
    cat = ExperimentRunner._normalize_error_category(
        failure_reason="invalid_schema_dict",
        action_success=False, page_changed=False,
    )
    assert cat == "invalid_schema"


def test_normalize_runner_invalid_action_preserved():
    """B-134 runner-rescue category must survive (no collapse into
    parse_error or invalid_*)."""
    cat = ExperimentRunner._normalize_error_category(
        failure_reason="runner_invalid_action",
        action_success=False, page_changed=False,
    )
    assert cat == "runner_invalid_action"


def test_normalize_parse_error_still_parse():
    """parse_failed / repaired_* / keyword_* → parse_error (no regress)."""
    for reason in ("parse_failed", "repaired_fenced", "repaired_raw_decode",
                   "keyword_finish", "multiple_actions"):
        cat = ExperimentRunner._normalize_error_category(
            failure_reason=reason,
            action_success=False, page_changed=False,
        )
        assert cat == "parse_error", f"reason={reason!r} → {cat!r}"


def test_normalize_env_error_still_env():
    """timeout / playwright / browser / network → env_error (no regress)."""
    for reason in ("timeout", "playwright_error", "browser_crashed",
                   "connection_refused", "network_unreachable"):
        cat = ExperimentRunner._normalize_error_category(
            failure_reason=reason,
            action_success=False, page_changed=False,
        )
        assert cat == "env_error", f"reason={reason!r} → {cat!r}"


def test_normalize_unknown_failure_bucket_for_new_reasons():
    """B-167 critical: any UNKNOWN failure_reason (not in any whitelist)
    falls to ``unknown_failure``, not silent ``invalid_action`` catch-all.

    This is the future-proof tripwire — if a new backend emits a novel
    failure_reason string (e.g. agent gets a new safety filter result),
    it surfaces as ``unknown_failure`` and downstream Counter telemetry
    catches the actual reason for taxonomy bump.
    """
    for novel_reason in ("model_oom", "image_too_large",
                         "glm_fallback_exhausted", "safety_filter_triggered"):
        cat = ExperimentRunner._normalize_error_category(
            failure_reason=novel_reason,
            action_success=False, page_changed=False,
        )
        assert cat == "unknown_failure", (
            f"B-167 regression: novel reason {novel_reason!r} collapsed to {cat!r} "
            f"instead of unknown_failure (silently lost to invalid_action catch-all?)"
        )


def test_normalize_no_progress_still_no_progress():
    """No failure_reason + action failed + page unchanged → no_progress."""
    cat = ExperimentRunner._normalize_error_category(
        failure_reason=None,
        action_success=False, page_changed=False,
    )
    assert cat == "no_progress"


# ---------------------------------------------------------------------------
# Episode-summary unknown_failure_reasons Counter telemetry
# ---------------------------------------------------------------------------


def test_episode_summary_has_unknown_failure_reasons_field():
    """Runner main.py must populate ``unknown_failure_reasons`` Counter on
    every episode (empty dict when nothing unknown). This is the paper-grade
    tripwire — if a previously-unseen reason appears frequently, the dict
    surfaces it for catalog inclusion."""
    src = (REPO_ROOT / "p79/experiment/runner/main.py").read_text(encoding="utf-8")
    assert 'episode_summary["unknown_failure_reasons"]' in src, (
        "B-167 episode_summary missing unknown_failure_reasons Counter field"
    )
    # Counter is built from step_records filtered by error_category=='unknown_failure'
    assert 'error_category") == "unknown_failure"' in src or \
        '"error_category"] == "unknown_failure"' in src, (
        "B-167 Counter must filter step_records by error_category=='unknown_failure'"
    )


# ---------------------------------------------------------------------------
# parse_action_text propagation of detailed reason
# ---------------------------------------------------------------------------


def test_parse_action_text_propagates_invalid_action_type_reason():
    """When clean JSON parses but action_type is unknown, parse_action_text
    must propagate ``invalid_action_type`` (not generic ``invalid_action``)
    so runner's _normalize_error_category gets the discriminator."""
    from p79.backends.action_utils import parse_action_text
    action, valid, reason = parse_action_text('{"action_type": "clik"}')
    assert valid is False
    assert reason == "invalid_action_type", (
        f"Expected specific 'invalid_action_type', got {reason!r}"
    )
