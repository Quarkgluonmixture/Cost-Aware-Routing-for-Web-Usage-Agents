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
    """JSON parses OK but action_type is not allowed → valid=False.

    B-165 (/stress A1.4a, 2026-05-16): the failure_reason is now the specific
    sub-category ``invalid_action_type`` (was generic ``invalid_action``)
    so paper §3.5 error taxonomy can distinguish action-type-typo failures
    from structural-schema failures, element-id failures, etc.
    """
    action, valid, reason = parse_action_text('{"action_type":"DROP_TABLE"}')
    assert valid is False
    assert action == {"action_type": "wait"}
    assert reason == "invalid_action_type"


def test_parse_valid_json_click_no_target():
    """JSON parses OK but click has no coordinate or element_id → valid=False.

    B-165 (/stress A1.4a, 2026-05-16): specific sub-category
    ``invalid_element_id`` (was generic ``invalid_action``).
    """
    action, valid, reason = parse_action_text('{"action_type":"click"}')
    assert valid is False
    assert reason == "invalid_element_id"


def test_parse_repaired_regex_valid():
    """B-141 (/stress A1.1 v8 codex F6, 2026-05-15): repair label changed
    from "repaired_regex" → "repaired_raw_decode" since parser now uses
    json.JSONDecoder().raw_decode() scan instead of greedy `\\{.*\\}` regex.
    Behavior preserved (valid JSON object embedded in prose still repairs)."""
    action, valid, reason = parse_action_text('thinking... {"action_type":"scroll","direction":"down"}')
    assert valid is True
    assert reason == "repaired_raw_decode"


def test_parse_repaired_regex_invalid_action():
    action, valid, reason = parse_action_text('thinking... {"action_type":"DESTROY"}')
    assert valid is False
    assert reason == "invalid_action_repaired"


def test_parse_unparseable_with_scroll_word_falls_to_wait():
    """/stress A1.1 codex Mode B C3 fix (2026-05-15): keyword scroll fallback
    removed. Previously this returned ``action_type=scroll`` with valid=False,
    but the runner executed the scroll anyway because it does not gate env.step
    on parse_valid. Cross-validate Q4 confirmed archived runs with
    ``keyword_scroll`` actions had ``action_success=True`` — silent partial
    automation driven by substring match, not by model action. Behaviour now
    mirrors §67 ``keyword_finish`` removal: incidental keywords in unparseable
    text are not actions.
    """
    action, valid, reason = parse_action_text("let me scroll down")
    assert valid is False
    assert reason == "parse_failed"
    assert action["action_type"] == "wait"


def test_parse_unparseable_with_back_word_falls_to_wait():
    """/stress A1.1 codex Mode B C3 fix: same as scroll — `back` keyword
    fallback removed. Falls through to wait with valid=False.
    """
    action, valid, reason = parse_action_text("Let me go back to the previous page")
    assert valid is False
    assert reason == "parse_failed"
    assert action["action_type"] == "wait"


def test_parse_total_failure():
    action, valid, reason = parse_action_text("gibberish")
    assert valid is False
    assert reason == "parse_failed"
    assert action["action_type"] == "wait"
    # thought preserves raw model output for diagnostics
    assert action.get("thought") == "gibberish"


# ---------------------------------------------------------------------------
# B-141 (/stress A1.1 v8 codex F6, 2026-05-15) — robust JSON repair
# ---------------------------------------------------------------------------


def test_parse_fenced_json_block_repair():
    """Fenced ```json {...} ``` block should be repair-preferred over raw
    scan — models often echo "Output ONLY valid JSON" instruction by
    wrapping it in markdown fence anyway."""
    text = '''
    Let me think about this.

    ```json
    {"action_type": "click", "element_id": 42}
    ```
    '''
    action, valid, reason = parse_action_text(text)
    assert valid is True
    assert action["action_type"] == "click"
    assert action["element_id"] == 42
    assert reason == "repaired_fenced"


def test_parse_multiple_identical_actions_repairs_to_one():
    """Model repetition (same action emitted twice) should collapse to one
    valid action, not flag as ambiguity."""
    text = '''
    {"action_type":"scroll","scroll_direction":"down"}
    {"action_type":"scroll","scroll_direction":"down"}
    '''
    action, valid, reason = parse_action_text(text)
    assert valid is True
    assert action["action_type"] == "scroll"
    assert reason == "repaired_multiple_identical"


def test_parse_multiple_distinct_actions_flags_ambiguity():
    """Two genuinely different actions in same output should NOT silently
    pick the first one (previous greedy regex behavior). Surface as
    parse_valid=False with explicit failure_reason="multiple_actions"."""
    text = '''
    Option A: {"action_type":"click","element_id":42}
    Option B: {"action_type":"type","element_id":7,"text":"hello"}
    '''
    action, valid, reason = parse_action_text(text)
    assert valid is False
    assert reason == "multiple_actions"


def test_parse_invalid_action_after_raw_decode_emits_repaired_invalid():
    """raw_decode finds a JSON object but validate_action rejects the
    action_type → emit invalid_action_repaired (distinguishable from
    parse_failed where no JSON was found at all)."""
    text = 'I will {"action_type":"NUKE","target":"all"}'
    action, valid, reason = parse_action_text(text)
    assert valid is False
    assert reason == "invalid_action_repaired"


def test_parse_no_json_at_all_falls_to_parse_failed():
    """No JSON anywhere → parse_failed (no candidates found)."""
    action, valid, reason = parse_action_text("just prose with no JSON")
    assert valid is False
    assert reason == "parse_failed"


# ---------------------------------------------------------------------------
# B-142 (/stress A1.1 v8 codex F7, 2026-05-15) — coord/delta shape check
# ---------------------------------------------------------------------------


def test_validate_click_rejects_malformed_coord_pair():
    """Previously {"action_type":"click","coordinate":[2,"x"]} validated
    True (only `coord is None` check). Now coord shape + numeric type
    enforced: reject malformed payload."""
    action, valid = validate_action({"action_type": "click", "coordinate": [2, "x"]})
    assert valid is False
    assert action == {"action_type": "wait"}


def test_validate_click_rejects_wrong_coord_length():
    """Coord must be exactly 2 elements (x, y). Reject [0.5] / [0.5,0.6,0.7]."""
    for bad_coord in ([0.5], [0.5, 0.6, 0.7], []):
        action, valid = validate_action({"action_type": "click", "coordinate": bad_coord})
        assert valid is False, f"coord {bad_coord!r} should be rejected"


def test_validate_click_rejects_nan_inf_coord():
    """NaN / inf in coord → reject (not a meaningful screen position)."""
    for bad_coord in ([float("nan"), 0.5], [0.5, float("inf")], [float("-inf"), 0.5]):
        action, valid = validate_action({"action_type": "click", "coordinate": bad_coord})
        assert valid is False, f"coord {bad_coord!r} should be rejected"


def test_validate_click_accepts_valid_normalized_coord():
    """Sanity: valid coord (no element_id) should pass."""
    action, valid = validate_action({"action_type": "click", "coordinate": [0.5, 0.7]})
    assert valid is True
    assert action["coordinate_type"] == "normalized"


def test_validate_click_accepts_valid_pixel_coord():
    """Pixel coords (>1) accepted as long as non-negative finite (vision
    mode may emit pixel coordinates)."""
    action, valid = validate_action({"action_type": "click", "coordinate": [120, 350]})
    assert valid is True


def test_validate_scroll_rejects_malformed_delta():
    """Scroll delta must be 2 finite floats."""
    for bad_delta in (None, [0.5], "down", [float("nan"), 0.1], ["x", "y"]):
        a = {"action_type": "scroll"}
        if bad_delta is not None:
            a["delta"] = bad_delta
        action, valid = validate_action(a)
        if bad_delta is None:
            # No delta at all — scroll still valid (env applies default).
            assert valid is True, "scroll without delta should remain valid"
        else:
            assert valid is False, f"scroll delta {bad_delta!r} should be rejected"


def test_validate_scroll_accepts_valid_delta():
    action, valid = validate_action({"action_type": "scroll", "delta": [0, 0.8]})
    assert valid is True
    action, valid = validate_action({"action_type": "scroll", "delta": [0, -0.5]})
    assert valid is True


def test_validate_tab_focus_requires_int_page_number():
    """tab_focus needs page_number int. Previously no check at all."""
    for bad in (None, "1", 1.5, -1):
        a = {"action_type": "tab_focus"}
        if bad is not None:
            a["page_number"] = bad
        action, valid = validate_action(a)
        assert valid is False, f"tab_focus page_number {bad!r} should be rejected"
    action, valid = validate_action({"action_type": "tab_focus", "page_number": 2})
    assert valid is True


def test_validate_select_option_rejects_non_int_element_id():
    """select_option element_id must be int (was: any truthy value)."""
    action, valid = validate_action({
        "action_type": "select_option",
        "element_id": "42",  # string, not int
        "option_label": "Red",
    })
    assert valid is False
    # Sanity: int element_id passes
    action, valid = validate_action({
        "action_type": "select_option",
        "element_id": 42,
        "option_label": "Red",
    })
    assert valid is True
