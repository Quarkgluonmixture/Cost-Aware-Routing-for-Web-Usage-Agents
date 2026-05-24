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

    B-167 (/stress A1.4a, 2026-05-16): the failure_reason is now the specific
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

    B-167 (/stress A1.4a, 2026-05-16): specific sub-category
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
    """B-413 (/stress A1.2 v8 Mode B P1-6, 2026-05-16): repair path now
    surfaces specific sub-category reason instead of generic
    `invalid_action_repaired`. `{"action_type":"DESTROY"}` → repair path
    finds candidate via raw_decode → validate_action_detailed returns
    `invalid_action_type` → propagated to parse_action_text caller.
    """
    action, valid, reason = parse_action_text('thinking... {"action_type":"DESTROY"}')
    assert valid is False
    assert reason == "invalid_action_type"


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


def test_parse_invalid_action_after_raw_decode_emits_specific_reason():
    """B-413 (/stress A1.2 v8 Mode B P1-6, 2026-05-16): raw_decode finds a
    JSON object but validate_action_detailed rejects it → emit specific
    sub-category reason (here `invalid_action_type`) rather than the old
    generic `invalid_action_repaired`. Distinguishable from `parse_failed`
    (no JSON at all) because we still got a parseable JSON object, but
    schema-invalid. Paper §3.5 taxonomy now sees the real reason."""
    text = 'I will {"action_type":"NUKE","target":"all"}'
    action, valid, reason = parse_action_text(text)
    assert valid is False
    assert reason == "invalid_action_type"


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
    """Scroll delta must be 2 finite floats. B-412 (/stress A1.2 v8 Mode B
    P1-5, 2026-05-16): naked scroll (no delta + no scroll_direction + no
    direction alias) now rejected — pre-fix the validator silently passed
    naked scroll and VWA env couldn't execute it (`vwa_wrapper.py:356`
    required `delta` or `scroll_direction`)."""
    for bad_delta in ([0.5], "down", [float("nan"), 0.1], ["x", "y"]):
        a = {"action_type": "scroll", "delta": bad_delta}
        action, valid = validate_action(a)
        assert valid is False, f"scroll delta {bad_delta!r} should be rejected"

    # B-412: naked scroll without targeting field → invalid_schema_dict.
    action, valid = validate_action({"action_type": "scroll"})
    assert valid is False, "naked scroll without delta/direction now invalid"


def test_validate_scroll_accepts_direction_aliases():
    """B-412: scroll with WebArena-legacy `direction` field stays valid
    for cross-benchmark compat (`vwa_wrapper.py:800` reads this field).
    Also accept tool-calling-schema `scroll_direction`."""
    for direction in ("up", "down"):
        action, valid = validate_action({"action_type": "scroll", "scroll_direction": direction})
        assert valid is True, f"scroll_direction={direction} should be valid"
    for direction in ("up", "down", "left", "right"):
        action, valid = validate_action({"action_type": "scroll", "direction": direction})
        assert valid is True, f"direction={direction} (WA alias) should be valid"


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
    """select_option element_id must resolve to int > 0.

    B-572 (/stress A1.22 P1-14-B* codex OOB, 2026-05-17): digit-string
    element_id like `"42"` is now canonicalized to int(42) BEFORE per-action
    validation — closes the B0 tool_use vs B1/B2 text JSON asymmetry where
    quoted-int element_id was rejected on text path but accepted on tool
    path. Non-digit strings, floats ("1.0"), negative, and zero are still
    rejected. Test updated to match the new contract:
      - `"42"` → coerce to int(42) → valid
      - `"hello"` → not coerced → invalid (per-action `isinstance(int)` fails)
      - `0` and `-1` → still invalid (per-action `> 0` requirement)
    Provenance flag `element_id_coerced_from_string` records the coercion.
    """
    # B-572: digit-string is now coerced; was rejected pre-B-572.
    action, valid = validate_action({
        "action_type": "select_option",
        "element_id": "42",  # string-digit, coerced to int(42)
        "option_label": "Red",
    })
    assert valid is True
    assert action.get("element_id") == 42
    assert action.get("element_id_coerced_from_string") is True

    # Non-digit string still rejected.
    action, valid = validate_action({
        "action_type": "select_option",
        "element_id": "hello",
        "option_label": "Red",
    })
    assert valid is False

    # Sanity: int element_id passes unchanged + no coercion flag.
    action, valid = validate_action({
        "action_type": "select_option",
        "element_id": 42,
        "option_label": "Red",
    })
    assert valid is True
    assert action.get("element_id_coerced_from_string") is None

    # Zero / negative int still rejected (per-action `> 0`).
    action, valid = validate_action({
        "action_type": "select_option",
        "element_id": 0,
        "option_label": "Red",
    })
    assert valid is False


def test_undeclared_coord_infers_pixel_not_blind_normalized():
    """B-452 (/stress A1.4 P1-1-B codex OOB, 2026-05-17): undeclared
    coordinate_type (caller did not pass) must be inferred from coord
    values, not blindly stamped as "normalized".

    Pre-B-452 the validator's "auto-add coordinate_type when missing"
    branch (action_utils.py:299/317/344-345) stamped `"normalized"` for
    every valid (positive finite) coord including obvious pixel pairs
    like [100, 200]. The env wrapper at vwa_wrapper.py:352-358 then
    silently divides by viewport, but the step JSONL audit trail
    claimed normalized — paper §3 error-taxonomy + cross-baseline
    coord-failure analysis were mislabeled.

    Post-B-452: `max(x, y) > 1.0` → "pixel"; else → "normalized".
    """
    # click with pixel coords + no coordinate_type → infer "pixel"
    action, valid = validate_action({
        "action_type": "click", "coordinate": [100, 200],
    })
    assert valid is True
    assert action["coordinate_type"] == "pixel", (
        f"pixel coord [100, 200] should infer 'pixel', "
        f"got {action.get('coordinate_type')!r}"
    )

    # click with normalized coords + no coordinate_type → infer "normalized"
    action, valid = validate_action({
        "action_type": "click", "coordinate": [0.5, 0.5],
    })
    assert valid is True
    assert action["coordinate_type"] == "normalized", (
        f"normalized coord [0.5, 0.5] should infer 'normalized', "
        f"got {action.get('coordinate_type')!r}"
    )

    # type with pixel coord + no coordinate_type → infer "pixel"
    action, valid = validate_action({
        "action_type": "type", "text": "x", "coordinate": [50, 80],
    })
    assert valid is True
    assert action["coordinate_type"] == "pixel"

    # select_option with pixel coord + no coordinate_type → infer "pixel"
    action, valid = validate_action({
        "action_type": "select_option", "coordinate": [30, 40], "option_label": "X",
    })
    assert valid is True
    assert action["coordinate_type"] == "pixel"

    # Explicit declaration is preserved (no inference override).
    # The declared "pixel" passes _is_valid_coordinate_pair's pixel branch
    # (x >= 0 and y >= 0), so this is structurally valid even though the
    # values [0.5, 0.5] look normalized — caller intent wins.
    action, valid = validate_action({
        "action_type": "click", "coordinate": [0.5, 0.5],
        "coordinate_type": "pixel",
    })
    assert valid is True
    assert action["coordinate_type"] == "pixel", (
        "explicit coordinate_type must not be overridden by inference"
    )


# ---------------------------------------------------------------------------
# B-1860: Qwen 0-1000 coordinate contract
#
# Qwen3-VL natively emits a 0-1000 coordinate system (probe-confirmed B0 + B1
# 2026-05-24) but sometimes also returns normalized [0,1] (mixed-format probe).
# Contract: judge EACH dimension independently BY VALUE — `<= 1.1` is a
# normalized [0,1] coord, `> 1.1` is a Qwen 0-1000 coord (divide by 1000) —
# IGNORING the model's `coordinate_type` declaration (empirically unreliable:
# model stamps "normalized" while emitting 0-1000). Save the format layer
# only, NOT the grounding layer (no target snapping). True malformed
# (NaN / inf / non-number / shape != 2 / negative / bool) still rejects.
# ---------------------------------------------------------------------------


def test_b1860_validator_accepts_qwen_0_1000_even_when_declared_normalized():
    """The pre-B-1860 hard reject (`coordinate_type=="normalized" → [0,1]`)
    turned 0-1000 coords into parse errors (vision parse_error 13.6%). Now a
    0-1000 coord is VALID regardless of the (ignored) coordinate_type label.
    """
    from p79.backends.action_utils import _is_valid_coordinate_pair

    # 0-1000 coord declared "normalized" (the empirical failure signature) → valid.
    assert _is_valid_coordinate_pair([598, 125], coordinate_type="normalized") is True
    assert _is_valid_coordinate_pair([728, 920], coordinate_type="normalized") is True
    # Mixed (x normalized, y 0-1000) declared "normalized" → valid.
    assert _is_valid_coordinate_pair([0.842, 117], coordinate_type="normalized") is True
    # Pure normalized declared "normalized" → still valid.
    assert _is_valid_coordinate_pair([0.5, 0.5], coordinate_type="normalized") is True
    # No declaration (None) → by-value, both regimes valid.
    assert _is_valid_coordinate_pair([598, 125]) is True
    assert _is_valid_coordinate_pair([0.5, 0.5]) is True


def test_b1860_validator_still_rejects_true_malformed():
    """Format-layer recovery must NOT relax true-malformed rejection."""
    from p79.backends.action_utils import _is_valid_coordinate_pair

    # NaN / inf component → reject.
    assert _is_valid_coordinate_pair([float("nan"), 5]) is False
    assert _is_valid_coordinate_pair([float("inf"), 5]) is False
    # Wrong shape (1 number) → reject.
    assert _is_valid_coordinate_pair([500]) is False
    # Wrong shape (3 numbers) → reject.
    assert _is_valid_coordinate_pair([500, 600, 700]) is False
    # Negative component → reject (grounding-nonsense, not a format issue).
    assert _is_valid_coordinate_pair([-5, 500]) is False
    assert _is_valid_coordinate_pair([500, -1]) is False
    # Non-number component → reject.
    assert _is_valid_coordinate_pair([500, "x"]) is False
    # Bool component (int subclass) → reject (B-799 preserved).
    assert _is_valid_coordinate_pair([True, 500]) is False
    # Unknown coordinate_type enum → reject (B-802 schema guard preserved).
    assert _is_valid_coordinate_pair([0.5, 0.5], coordinate_type="screen") is False


def test_b1860_validate_action_qwen_0_1000_click_is_parse_valid():
    """End-to-end through validate_action: a 0-1000 click (the kind that was
    cap-killing vision episodes) is now parse_valid=True (no invalid_coord)."""
    from p79.backends.action_utils import validate_action_detailed

    _, valid, reason = validate_action_detailed(
        {"action_type": "click", "coordinate": [598, 125], "coordinate_type": "normalized"}
    )
    assert valid is True
    assert reason is None
    # Mixed-format click also parse_valid.
    _, valid2, _ = validate_action_detailed(
        {"action_type": "click", "coordinate": [0.842, 117]}
    )
    assert valid2 is True
    # NaN coord still surfaces invalid_coord (true malformed → feeds parse caps).
    _, valid3, reason3 = validate_action_detailed(
        {"action_type": "click", "coordinate": [float("nan"), 5]}
    )
    assert valid3 is False
    assert reason3 == "invalid_coord"


def _b1860_fake_wrapper(monkeypatch, viewport_width, viewport_height):
    """Build a VWAWrapper backed by a fake browser_env + env that captures the
    create_mouse_click_action (left, top) the wrapper computes.

    Uses monkeypatch.setitem for sys.modules so the fake `browser_env` is
    auto-restored at test teardown (a raw `sys.modules[...]=` assignment would
    leak the SimpleNamespace into later tests and break real `import
    browser_env.env_config` — observed polluting test_vwa_evaluator_b91_guard).
    """
    import sys
    import types

    from p79.envs.vwa_wrapper import VWAWrapper

    captured = {}

    fake_browser_env = types.SimpleNamespace(
        create_id_based_action=lambda s: {"kind": "id", "action_str": s},
        create_mouse_click_action=lambda left, top: captured.__setitem__("mouse_click", (left, top)) or {"kind": "mouse", "left": left, "top": top},
        create_mouse_hover_action=lambda left, top: {"kind": "hover_coord", "left": left, "top": top},
        create_scroll_action=lambda direction: {"kind": "scroll", "direction": direction},
        create_stop_action=lambda answer: {"kind": "stop", "answer": answer},
        create_go_back_action=lambda: {"kind": "back"},
        create_go_forward_action=lambda: {"kind": "forward"},
        create_page_focus_action=lambda page_number: {"kind": "tab", "page_number": page_number},
        create_keyboard_type_action=lambda text: {"kind": "type", "text": text},
        create_none_action=lambda: {"kind": "none"},
        create_playwright_action=lambda s: {"kind": "playwright", "action_str": s},
        create_goto_url_action=lambda url: {"kind": "goto", "url": url},
        ScriptBrowserEnv=None,
    )
    monkeypatch.setitem(sys.modules, "browser_env", fake_browser_env)

    class _FakeEnv:
        def step(self, action):
            return {"text": "[1] link"}, 0.0, False, False, {"url": "http://mock.local"}

    wrapper = VWAWrapper(viewport_width=viewport_width, viewport_height=viewport_height)
    wrapper._env = _FakeEnv()
    return wrapper, captured


def test_b1860_wrapper_normalizes_qwen_0_1000_per_dimension(monkeypatch):
    """Wrapper click path maps each dimension by the 0-1000 contract:
    `> 1.1` → /1000; `<= 1.1` → kept. viewport != 1000 (1280x720, the real
    default) proves the divisor is 1000, NOT the viewport. The spec's expected
    normalized outputs are asserted exactly.
    """
    # Both 0-1000 → (598/1000, 125/1000) = (0.598, 0.125).
    w, cap = _b1860_fake_wrapper(monkeypatch, 1280, 720)
    w.step({"action_type": "click", "coordinate": [598, 125]})
    assert cap["mouse_click"] == (0.598, 0.125)

    # Both 0-1000, B1-probe values → (728/1000, 920/1000) = (0.728, 0.920).
    # 920 > 720 viewport_height — under the OLD /viewport bug this would have
    # been 920/720 = 1.28 (clamped to ~1.0, far off). /1000 gives 0.920.
    w, cap = _b1860_fake_wrapper(monkeypatch, 1280, 720)
    w.step({"action_type": "click", "coordinate": [728, 920]})
    assert cap["mouse_click"] == (0.728, 0.92)

    # Mixed: x normalized (<= 1.1 kept), y 0-1000 (/1000) → (0.842, 0.117).
    w, cap = _b1860_fake_wrapper(monkeypatch, 1280, 720)
    w.step({"action_type": "click", "coordinate": [0.842, 117]})
    assert cap["mouse_click"] == (0.842, 0.117)

    # Pure normalized: both kept as-is → (0.5, 0.5).
    w, cap = _b1860_fake_wrapper(monkeypatch, 1280, 720)
    w.step({"action_type": "click", "coordinate": [0.5, 0.5]})
    assert cap["mouse_click"] == (0.5, 0.5)
