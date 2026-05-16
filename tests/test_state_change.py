"""Tests for p79.experiment.state_change.

Covers the swatch_form_change_audit.md regression: same-name radio groups
where each radio is the sole child of its wrapper (idx_in_parent=0 for all)
must be individually addressable in the snapshot diff.
"""
from p79.experiment.state_change import detect_page_state_change


def _state(form_fields):
    return {
        "url": "http://shop/test.html",
        "title": "T",
        "visible_text": "x",
        "interactive_elements_count": 1,
        "form_fields_count": len(form_fields),
        "modal_present": False,
        "scroll_x": 0,
        "scroll_y": 0,
        "scroll_height": 0,
        "client_height": 720,
        "dom_complexity": 1,
        "text_length": 1,
        "form_field_values": form_fields,
    }


def _radio(value, checked):
    # idx=0 mimics Magento custom-option layout: each radio is the sole child
    # of its <div class="field choice"> wrapper.
    return {"tag": "input", "type": "radio", "name": "options[10527]", "idx": 0,
            "value": value, "checked": checked}


def test_same_name_radio_group_first_member_flip_detected():
    """Click on Style 3 (first of three same-name radios) must produce form_value_changed."""
    before = [_radio("64440", False), _radio("64441", False), _radio("64442", False)]
    after = [_radio("64440", True), _radio("64441", False), _radio("64442", False)]
    success, reasons, _ = detect_page_state_change(_state(before), _state(after), "click")
    assert "form_value_changed" in reasons
    assert success is True


def test_same_name_radio_group_middle_member_flip_detected():
    """Click on Style 4 (middle radio) must produce form_value_changed."""
    before = [_radio("64440", False), _radio("64441", False), _radio("64442", False)]
    after = [_radio("64440", False), _radio("64441", True), _radio("64442", False)]
    success, reasons, _ = detect_page_state_change(_state(before), _state(after), "click")
    assert "form_value_changed" in reasons
    assert success is True


def test_same_name_radio_group_swap_detected():
    """Switching from Style 3 to Style 4 (one off, one on) must be detected."""
    before = [_radio("64440", True), _radio("64441", False), _radio("64442", False)]
    after = [_radio("64440", False), _radio("64441", True), _radio("64442", False)]
    success, reasons, _ = detect_page_state_change(_state(before), _state(after), "click")
    assert "form_value_changed" in reasons
    assert success is True


def test_same_name_radio_group_no_change_no_false_positive():
    """Identical snapshots must not report form_value_changed."""
    before = [_radio("64440", False), _radio("64441", False), _radio("64442", False)]
    after = [_radio("64440", False), _radio("64441", False), _radio("64442", False)]
    success, reasons, _ = detect_page_state_change(_state(before), _state(after), "click")
    assert "form_value_changed" not in reasons
    assert success is False


def test_text_input_change_still_detected():
    """Sanity: text input value change still detected (non-radio path unchanged)."""
    before = [{"tag": "input", "type": "text", "name": "q", "idx": 0, "value": ""}]
    after = [{"tag": "input", "type": "text", "name": "q", "idx": 0, "value": "red blanket"}]
    success, reasons, _ = detect_page_state_change(_state(before), _state(after), "type")
    assert "form_value_changed" in reasons
    assert success is True
