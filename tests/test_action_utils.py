from p79.backends.action_utils import parse_action_text, validate_action


# ---------------------------------------------------------------------------
# validate_action: returns (action, is_valid)
# ---------------------------------------------------------------------------


def test_validate_valid_click():
    action, valid = validate_action({"action_type": "click", "element_id": 3})
    assert valid is True
    assert action["action_type"] == "click"


def test_validate_invalid_action_type():
    action, valid = validate_action({"action_type": "DROP_TABLE"})
    assert valid is False
    assert action == {"action_type": "wait"}


def test_validate_non_dict():
    action, valid = validate_action("not a dict")
    assert valid is False
    assert action == {"action_type": "wait"}


def test_validate_click_missing_target():
    action, valid = validate_action({"action_type": "click"})
    assert valid is False
    assert action == {"action_type": "wait"}


def test_validate_stop_normalized_to_finish():
    action, valid = validate_action({"action_type": "stop", "answer": "42"})
    assert valid is True
    assert action["action_type"] == "finish"
    assert action["answer"] == "42"


# ---------------------------------------------------------------------------
# parse_action_text: valid flag reflects semantic validity
# ---------------------------------------------------------------------------


def test_parse_valid_json_valid_action():
    action, valid, reason = parse_action_text('{"action_type":"click","element_id":1}')
    assert valid is True
    assert reason is None
    assert action["action_type"] == "click"


def test_parse_valid_json_invalid_action_type():
    """JSON parses OK but action_type is not allowed → valid=False."""
    action, valid, reason = parse_action_text('{"action_type":"DROP_TABLE"}')
    assert valid is False
    assert action == {"action_type": "wait"}
    assert reason == "invalid_action"


def test_parse_valid_json_click_no_target():
    """JSON parses OK but click has no coordinate or element_id → valid=False."""
    action, valid, reason = parse_action_text('{"action_type":"click"}')
    assert valid is False
    assert reason == "invalid_action"


def test_parse_repaired_regex_valid():
    action, valid, reason = parse_action_text('thinking... {"action_type":"scroll","direction":"down"}')
    assert valid is True
    assert reason == "repaired_regex"


def test_parse_repaired_regex_invalid_action():
    action, valid, reason = parse_action_text('thinking... {"action_type":"DESTROY"}')
    assert valid is False
    assert reason == "invalid_action_repaired"


def test_parse_keyword_fallback():
    action, valid, reason = parse_action_text("let me scroll down")
    assert valid is False
    assert reason == "keyword_scroll"
    assert action["action_type"] == "scroll"


def test_parse_total_failure():
    action, valid, reason = parse_action_text("gibberish")
    assert valid is False
    assert reason == "parse_failed"
    assert action["action_type"] == "wait"
    # thought preserves raw model output for diagnostics
    assert action.get("thought") == "gibberish"
