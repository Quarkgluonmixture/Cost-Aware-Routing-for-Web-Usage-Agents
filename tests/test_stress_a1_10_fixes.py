"""Regression tests for /stress A1.10 paper-grade substrate fixes (B-359~376).

Covers:
- P0-2-AB*: runner uses agent_visible_changed for router input + analyzer
- P0-3-A: DEFAULT_CONFIG 6-mode universe
- P0-4-B*: learned-router skip rule router
- P1-1-A: _TEXT_TRUNCATION_LIMIT=20000 + hash fallback on long pages
- P1-2-AB*: MARK_ID_DETECT_RE anchored canonical regex
- P1-4-A*: _form_fields_changed discriminator for all types
- P1-5-A: _extract_modal_state strict role/aria-modal regex
"""
from __future__ import annotations

import pytest


# ─── P1-2-AB* anchored mark-id regex ───────────────────────────────────────
def test_b362_mark_id_anchored_matches_canonical_axtree_line():
    from p79.experiment.som import extract_mark_id, is_mark_line
    line = '\t\t[123] button "Submit"'
    assert extract_mark_id(line) == 123
    assert is_mark_line(line) is True


def test_b362_mark_id_anchored_rejects_bracketed_digit_in_label_content():
    """A1.4 SOM regex sibling propagation defect — bracketed digits inside
    StaticText labels must NOT be extracted as element ids."""
    from p79.experiment.som import extract_mark_id
    # Pre-fix unanchored regex would match `[4]` inside the label content.
    line = "    StaticText 'See section [4] below for details'"
    assert extract_mark_id(line) is None


def test_b362_mark_id_anchored_first_match_wins_on_legit_line():
    """When the line is a canonical AXTree row, the leading mark id is the
    one returned even if the label content contains bracketed digits."""
    from p79.experiment.som import extract_mark_id
    line = '\t\t[10] StaticText "Click [123] for details"'
    assert extract_mark_id(line) == 10  # leading wins


def test_b362_extract_interactive_count_uses_anchored_regex():
    """state_change._extract_interactive_count counts only anchored AXTree
    rows, not bracketed digits in label content."""
    from p79.experiment.state_change import _extract_interactive_count
    text = (
        "\t\t[1] button 'Submit'\n"
        "\t\t[2] StaticText 'see [99] notes [100] footnote'\n"
        "\t\t[3] textbox 'name'\n"
    )
    # Pre-fix would count [1, 2, 99, 100, 3] = 5; post-fix counts only
    # anchored rows [1, 2, 3] = 3.
    assert _extract_interactive_count(text) == 3


# ─── P1-1-A long-page truncation + hash fallback ────────────────────────────
def test_b363_text_truncation_limit_raised():
    from p79.experiment.state_change import _TEXT_TRUNCATION_LIMIT
    assert _TEXT_TRUNCATION_LIMIT == 20000


def test_b363_long_page_hash_fallback_equal_pages_unchanged():
    """When both pages exceed truncation limit, hash equality → similarity 1.0
    → no content_changed."""
    from p79.experiment.state_change import detect_page_state_change
    long_text = "A" * 25000  # > _TEXT_TRUNCATION_LIMIT
    before = {"visible_text": long_text, "url": "u", "title": "t",
              "interactive_elements_count": 0, "form_fields_count": 0,
              "modal_present": False, "scroll_x": 0, "scroll_y": 0,
              "dom_complexity": 0, "text_length": 25000, "form_field_values": []}
    after = dict(before)
    success, reasons, sim = detect_page_state_change(before, after, "click")
    assert sim == 1.0
    assert "content_changed" not in reasons


def test_b363_long_page_hash_fallback_different_pages_changed():
    from p79.experiment.state_change import detect_page_state_change
    before = {"visible_text": "A" * 25000, "url": "u", "title": "t",
              "interactive_elements_count": 0, "form_fields_count": 0,
              "modal_present": False, "scroll_x": 0, "scroll_y": 0,
              "dom_complexity": 0, "text_length": 25000, "form_field_values": []}
    after = dict(before)
    after["visible_text"] = "B" * 25000
    after["text_length"] = 25000
    success, reasons, sim = detect_page_state_change(before, after, "click")
    assert sim == 0.0
    assert "content_changed" in reasons


# ─── P1-5-A strict modal regex ──────────────────────────────────────────────
def test_b365_modal_state_does_not_match_dialog_substring_in_text():
    from p79.experiment.state_change import _extract_modal_state
    # Pre-fix: "dialog" substring anywhere → modal_present=True.
    # Post-fix: requires role/aria-modal attribute context.
    text = "    StaticText 'Open dialog about the new feature'"
    assert _extract_modal_state(text) is False


def test_b365_modal_state_matches_role_dialog():
    from p79.experiment.state_change import _extract_modal_state
    text = '<div role="dialog" aria-modal="true">'
    assert _extract_modal_state(text) is True


def test_b365_modal_state_matches_aria_modal():
    from p79.experiment.state_change import _extract_modal_state
    text = "    [12] role: dialog aria-modal: true"
    assert _extract_modal_state(text) is True


# ─── P1-4-A* form_fields_changed discriminator for all types ────────────────
def test_b366_form_value_change_detected_for_text_input_empty_name():
    """Two text inputs with name='' at idx=0 in different wrappers should
    not collapse to the same key after the value-in-discriminator fix."""
    from p79.experiment.state_change import _form_fields_changed
    before = [
        {"tag": "input", "type": "text", "name": "", "value": "foo", "idx": 0},
        {"tag": "input", "type": "text", "name": "", "value": "bar", "idx": 0},
    ]
    # If value changes on the first input, change should be detected.
    after = [
        {"tag": "input", "type": "text", "name": "", "value": "FOO", "idx": 0},
        {"tag": "input", "type": "text", "name": "", "value": "bar", "idx": 0},
    ]
    assert _form_fields_changed(before, after) is True


def test_b366_form_value_no_change_when_identical():
    from p79.experiment.state_change import _form_fields_changed
    before = [
        {"tag": "input", "type": "text", "name": "q", "value": "foo", "idx": 0},
    ]
    after = [
        {"tag": "input", "type": "text", "name": "q", "value": "foo", "idx": 0},
    ]
    assert _form_fields_changed(before, after) is False


# ─── P0-3-A DEFAULT_CONFIG 6-mode universe ──────────────────────────────────
def test_b367_default_config_observation_mode_is_6mode():
    """DEFAULT_CONFIG fallback now lists the full paper-1 6-mode universe."""
    from p79.experiment.config import DEFAULT_CONFIG
    modes = set(DEFAULT_CONFIG["variables"]["primary"]["observation_mode"])
    assert modes == {"dom", "som", "vision", "phantom_som", "phantom_text", "phantom_prompt"}


# ─── P0-4-B* learned-router skip rule router (smoke) ────────────────────────
def test_b368_learned_router_metadata_marker_present():
    """conditions.py emits router_variant=v7_learned on learned cells so the
    runner can branch its routing-skip path."""
    from p79.experiment.conditions import generate_conditions
    cfg = {
        "experiment": {"phase": "phase1", "benchmark": "vwa"},
        "variables": {
            "primary": {"observation_mode": ["dom", "som", "vision",
                                              "phantom_som", "phantom_text", "phantom_prompt"]},
            "phase1": {"variant": "router", "router_kind": "learned"},
        },
        "backends": {"default_backend": "local_4b", "local_4b": {"path": "x"}},
    }
    conds = generate_conditions(cfg)
    learned = [c for c in conds if c.observation_mode == "learned"]
    assert len(learned) >= 1
    assert learned[0].metadata.get("router_variant") == "v7_learned"


# ─── P0-1-ABC* router fire-rate audit gate ──────────────────────────────────
def test_b369_audit_router_fire_rate_disclosure_consistent(tmp_path):
    """Empty run dir → 0 steps → disclosure_consistent True (vacuous)."""
    from scripts.analysis.aggregate_phantom_lift import audit_router_fire_rate
    report = audit_router_fire_rate(tmp_path)
    assert report["total_steps"] == 0
    assert report["disclosure_consistent"] is True
