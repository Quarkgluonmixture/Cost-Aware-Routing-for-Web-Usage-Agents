"""Regression tests for /stress A1.2 cold-start v2 fixes (2026-05-17).

Covers P0-1-B / P0-2-B / P0-3-AB / P0-4-B P0 fixes + 9 P1 + 9 P2 fixes.
B-799~B-815 from master_bug_catalog.md.

Sibling tests in `test_stress_a1_2_fixes.py` cover the earlier B-149 / B-155 /
B-406-416 family. This file is exclusively the cold-start re-audit catches.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from p79.backends.action_utils import (
    _is_strict_int,
    _is_valid_coordinate_pair,
    _is_valid_delta_pair,
    parse_action_text,
    validate_action_detailed,
)
from p79.backends._shared_stage_prefix import build_stage_prefix
from p79.backends.base import BackendStepContext


REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# B-799 P0-1-B*  bool-as-int silent dispatch — 4 surfaces
# ---------------------------------------------------------------------------


def test_b799_strict_int_helper_rejects_bool():
    assert _is_strict_int(1) is True
    assert _is_strict_int(0) is True
    assert _is_strict_int(True) is False
    assert _is_strict_int(False) is False
    assert _is_strict_int(1.0) is False
    assert _is_strict_int("1") is False
    assert _is_strict_int(None) is False


def test_b799_click_element_id_true_rejected():
    _, valid, reason = validate_action_detailed({"action_type": "click", "element_id": True})
    assert valid is False
    assert reason == "invalid_element_id"


def test_b799_tab_focus_page_number_false_rejected():
    _, valid, reason = validate_action_detailed(
        {"action_type": "tab_focus", "page_number": False}
    )
    assert valid is False
    assert reason == "invalid_schema_dict"


def test_b799_scroll_delta_bool_components_rejected():
    _, valid, reason = validate_action_detailed(
        {"action_type": "scroll", "delta": [True, True]}
    )
    assert valid is False
    assert reason == "invalid_coord"


def test_b799_select_option_index_true_rejected():
    # option_index=True must not satisfy has_option (was: isinstance(int) accepted bool)
    _, valid, reason = validate_action_detailed(
        {"action_type": "select_option", "element_id": 5, "option_index": True}
    )
    assert valid is False
    assert reason == "invalid_select_option"


def test_b799_coord_bool_components_rejected():
    assert _is_valid_coordinate_pair([True, False], coordinate_type="normalized") is False
    assert _is_valid_coordinate_pair([True, 0.5], coordinate_type="normalized") is False
    # Sanity: real coords still pass.
    assert _is_valid_coordinate_pair([0.5, 0.5], coordinate_type="normalized") is True


# ---------------------------------------------------------------------------
# B-801 P0-4-B*  multiple fenced JSON actions → multiple_actions
# ---------------------------------------------------------------------------


def test_b801_multiple_fenced_actions_classified_ambiguity():
    text = """
```json
{"action_type":"click","element_id":1}
```
some prose
```json
{"action_type":"click","element_id":2}
```
"""
    _, valid, reason = parse_action_text(text)
    assert valid is False
    assert reason == "multiple_actions"


def test_b801_single_fenced_action_still_repaired_fenced():
    text = '```json\n{"action_type":"click","element_id":42}\n```'
    action, valid, reason = parse_action_text(text)
    assert valid is True
    assert reason == "repaired_fenced"
    assert action["element_id"] == 42


def test_b801_multiple_identical_fenced_collapses_to_one():
    text = """
```json
{"action_type":"click","element_id":7}
```
echo
```json
{"action_type":"click","element_id":7}
```
"""
    action, valid, reason = parse_action_text(text)
    assert valid is True
    assert reason == "repaired_multiple_identical"
    assert action["element_id"] == 7


# ---------------------------------------------------------------------------
# B-805 P0-3-AB*  scroll_direction enum {up,down,left,right} cross-baseline
# ---------------------------------------------------------------------------


def test_b805_scroll_direction_left_accepted():
    action, valid, reason = validate_action_detailed(
        {"action_type": "scroll", "scroll_direction": "left"}
    )
    assert valid is True
    assert reason is None
    assert action["scroll_direction"] == "left"


def test_b805_scroll_direction_right_accepted():
    action, valid, reason = validate_action_detailed(
        {"action_type": "scroll", "scroll_direction": "right"}
    )
    assert valid is True
    assert action["scroll_direction"] == "right"


# ---------------------------------------------------------------------------
# B-802 P1-2-B  unknown coordinate_type rejected
# ---------------------------------------------------------------------------


def test_b802_unknown_coordinate_type_rejected():
    assert _is_valid_coordinate_pair([0.5, 0.5], coordinate_type="screen") is False
    assert _is_valid_coordinate_pair([0.5, 0.5], coordinate_type="css") is False
    # Canonical enums still accepted
    assert _is_valid_coordinate_pair([0.5, 0.5], coordinate_type="normalized") is True
    assert _is_valid_coordinate_pair([100, 200], coordinate_type="pixel") is True


# ---------------------------------------------------------------------------
# B-803 P1-5-B*  zero delta [0,0] rejected
# ---------------------------------------------------------------------------


def test_b803_zero_delta_rejected():
    assert _is_valid_delta_pair([0, 0]) is False
    assert _is_valid_delta_pair([0.0, 0.0]) is False
    # Real deltas still pass.
    assert _is_valid_delta_pair([0, 0.8]) is True
    assert _is_valid_delta_pair([0.5, 0]) is True


def test_b803_zero_scroll_propagates_invalid_coord():
    _, valid, reason = validate_action_detailed(
        {"action_type": "scroll", "delta": [0, 0]}
    )
    assert valid is False
    assert reason == "invalid_coord"


# ---------------------------------------------------------------------------
# B-804 P1-4-A*  element_id "007" rejected (comment-lie fix)
# ---------------------------------------------------------------------------


def test_b804_element_id_leading_zero_rejected():
    action, valid, reason = validate_action_detailed(
        {"action_type": "click", "element_id": "007"}
    )
    assert valid is False
    assert reason == "invalid_element_id"


def test_b804_element_id_decimal_string_rejected():
    _, valid, reason = validate_action_detailed(
        {"action_type": "click", "element_id": "1.0"}
    )
    assert valid is False


def test_b804_element_id_canonical_string_still_coerced():
    action, valid, reason = validate_action_detailed(
        {"action_type": "click", "element_id": "12"}
    )
    assert valid is True
    assert action["element_id"] == 12
    assert action.get("element_id_coerced_from_string") is True


# ---------------------------------------------------------------------------
# B-810 P1-3-A*  reference_images Tuple immutable migration
# ---------------------------------------------------------------------------


def test_b810_reference_images_default_is_tuple():
    ctx = BackendStepContext(observation_mode="dom", som_enabled=False, som_text="")
    assert isinstance(ctx.reference_images, tuple)
    assert ctx.reference_images == ()


def test_b810_reference_images_list_input_frozen_to_tuple():
    ctx = BackendStepContext(
        observation_mode="dom", som_enabled=False, som_text="",
        reference_images=["img_a", "img_b"],
    )
    assert isinstance(ctx.reference_images, tuple)
    assert ctx.reference_images == ("img_a", "img_b")


def test_b810_reference_images_append_raises():
    ctx = BackendStepContext(
        observation_mode="dom", som_enabled=False, som_text="",
        reference_images=["img1"],
    )
    with pytest.raises(AttributeError):
        ctx.reference_images.append("img2")


# ---------------------------------------------------------------------------
# B-811 P1-6-B*  stage enum strict-validated
# ---------------------------------------------------------------------------


def test_b811_stage_typo_raises_value_error():
    for typo in ("planer", "groudner", "Single", "SINGLE", "", "  "):
        with pytest.raises(ValueError, match="BackendStepContext.stage"):
            BackendStepContext(
                observation_mode="dom", som_enabled=False, som_text="", stage=typo,
            )


def test_b811_stage_canonical_values_accepted():
    for stage in ("single", "planner", "grounder"):
        ctx = BackendStepContext(
            observation_mode="dom", som_enabled=False, som_text="", stage=stage,
        )
        assert ctx.stage == stage


# ---------------------------------------------------------------------------
# B-812 P1-1-A*  stage_prefix shared single-source byte-identical invariant
# ---------------------------------------------------------------------------


def test_b812_stage_prefix_byte_identical_across_baselines():
    """Smoke + sibling-propagation invariant: 3 wrappers must use the
    same shared prefix builder so paper §3.4 planner/grounder ablation
    contract holds.
    """
    qwen_src = (REPO_ROOT / "p79/backends/local_qwen.py").read_text(encoding="utf-8")
    gemma_src = (REPO_ROOT / "p79/backends/local_gemma.py").read_text(encoding="utf-8")
    proxy_src = (REPO_ROOT / "p79/backends/api_proxy.py").read_text(encoding="utf-8")

    for name, src in [("local_qwen", qwen_src), ("local_gemma", gemma_src), ("api_proxy", proxy_src)]:
        assert "from p79.backends._shared_stage_prefix import build_stage_prefix" in src, (
            f"{name}.py must import shared build_stage_prefix (B-812)"
        )
        assert "build_stage_prefix(context.stage" in src, (
            f"{name}.py must call build_stage_prefix(context.stage, ...) (B-812)"
        )
        # No inline copy of the stage-prefix string blocks should remain.
        assert "[Stage: planner] Based on the task and interaction history" not in src, (
            f"{name}.py still has inline planner prefix — must use shared module (B-812)"
        )
        assert "[Stage: grounder] Sub-goal:" not in src, (
            f"{name}.py still has inline grounder prefix — must use shared module (B-812)"
        )


def test_b812_build_stage_prefix_stable_strings():
    """Lock the exact byte sequences so prose-tuning churn fails this test
    before silently drifting cross-baseline."""
    assert build_stage_prefix("single") == ""
    # Planner has stable content.
    planner = build_stage_prefix("planner")
    assert planner.startswith("[Stage: planner]")
    assert planner.endswith("\n\n")
    # Grounder includes sub-goal.
    grounder = build_stage_prefix("grounder", "click search button")
    assert grounder.startswith("[Stage: grounder] Sub-goal: click search button")
    assert "produce a concrete action JSON" in grounder
    assert grounder.endswith("\n\n")


# ---------------------------------------------------------------------------
# B-805 / scroll alias path direction:"left" still accepted (regression)
# ---------------------------------------------------------------------------


def test_b805_legacy_direction_alias_still_accepted():
    action, valid, reason = validate_action_detailed(
        {"action_type": "scroll", "direction": "left"}
    )
    assert valid is True
    assert action["scroll_direction"] == "left"
    assert action.get("direction_raw_alias") == "left"


# ---------------------------------------------------------------------------
# B-806 P1-8-A  validate_action 2-tuple shim emits DeprecationWarning
# ---------------------------------------------------------------------------


def test_b806_validate_action_emits_deprecation_warning():
    from p79.backends.action_utils import validate_action

    with pytest.warns(DeprecationWarning, match="failure_reason"):
        validate_action({"action_type": "click", "element_id": 1})


# ---------------------------------------------------------------------------
# B-809 P2-1-A* factory dom_mode guard
# ---------------------------------------------------------------------------


def test_b809_dom_mode_heuristic_only_rejected():
    from p79.backends.factory import create_backend

    with pytest.raises(ValueError, match="dom_mode"):
        create_backend("B1_test", {"type": "local_qwen", "dom_mode": "heuristic_only"})


def test_b809_dom_mode_llm_passes():
    from p79.backends.factory import create_backend

    b = create_backend("B1_test", {"type": "local_qwen", "dom_mode": "llm", "mock_mode": True})
    assert b.backend_id == "B1_test"


# ---------------------------------------------------------------------------
# B-816 P2-7-B* curated TypeError for non-string api_key_env
# ---------------------------------------------------------------------------


def test_b816_api_key_env_non_string_curated_value_error():
    from p79.backends.api_proxy import ApiProxyBackend

    with pytest.raises(ValueError, match="api_key_env must be a string"):
        ApiProxyBackend._validate_api_key_env(["PROXY_API_KEY"])
    with pytest.raises(ValueError, match="api_key_env must be a string"):
        ApiProxyBackend._validate_api_key_env(None)
    with pytest.raises(ValueError, match="api_key_env must be a string"):
        ApiProxyBackend._validate_api_key_env(123)


# ---------------------------------------------------------------------------
# B-817 P1-9-C* B0 sampling defense parity
# ---------------------------------------------------------------------------


def test_b817_api_proxy_warns_on_yaml_temperature_nonzero(caplog):
    import logging
    from p79.backends.api_proxy import ApiProxyBackend

    with caplog.at_level(logging.WARNING, logger="p79.backends.api_proxy"):
        ApiProxyBackend("B0_test", {
            "type": "api_proxy", "mock_mode": True,
            "api_key_env": "PROXY_API_KEY",
            "temperature": 0.5,
        })
    assert any("cross-baseline drift" in rec.message and "B-817" in rec.message for rec in caplog.records)


def test_b817_api_proxy_paper_grade_raises_on_nonzero_temperature():
    from p79.backends.api_proxy import ApiProxyBackend
    from p79.backends.base import BackendError

    with pytest.raises(BackendError, match="paper_grade=True"):
        ApiProxyBackend("B0_test", {
            "type": "api_proxy", "mock_mode": True,
            "api_key_env": "PROXY_API_KEY",
            "temperature": 0.5,
            "paper_grade": True,
        })


# ---------------------------------------------------------------------------
# B-819 codex F4 honest-gap: assert → explicit raise (python -O robust)
# ---------------------------------------------------------------------------


def test_b819_image_utils_pillow_lower_bound_uses_explicit_raise():
    src = (REPO_ROOT / "p79/backends/image_utils.py").read_text(encoding="utf-8")
    # Must use explicit raise (not assert) so python -O cannot strip the guard.
    assert "raise RuntimeError" in src
    # Lower bound check uses `< (10, 0)` pattern (replacing the old `>= (10, 0)` assert).
    assert "< (10, 0)" in src or "<(10, 0)" in src


# ---------------------------------------------------------------------------
# B-813 P0-2-B* None metadata preserve defense (wrapper-side smoke)
# ---------------------------------------------------------------------------


def test_b813_wrapper_meta_none_branch_present():
    """Sibling-propagation check: all 3 wrappers must guard against
    present-but-None metadata (model_calls / backend_type / token counts).
    """
    for path in (
        "p79/backends/local_qwen.py",
        "p79/backends/local_gemma.py",
        "p79/backends/api_proxy.py",
    ):
        src = (REPO_ROOT / path).read_text(encoding="utf-8")
        assert "meta.get(\"model_calls\") is None" in src, (
            f"{path} must guard model_calls=None (B-813)"
        )
        assert "meta.get(\"backend_type\") is None" in src, (
            f"{path} must guard backend_type=None (B-813)"
        )
        assert "input_tokens" in src and "output_tokens" in src, (
            f"{path} must guard token-count None under paper_grade (B-813)"
        )
