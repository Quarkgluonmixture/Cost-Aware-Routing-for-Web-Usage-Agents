"""B-1794: B0 tool schema `required` must mirror validate_action_detailed.

Cross-baseline semantic consistency invariant (advisor/user directive 2026-05-21):
B0 (tool-calling, tool_choice="required") forces a tool call; under forcing the
proxy emits a MINIMAL call satisfying only the required-array, so any field the
validator semantically requires but the schema marks optional gets dropped (the
original element_id-on-search bug). B1/B2 (prose-JSON) pass the SAME
validate_action_detailed gate but emit fields naturally.

For B0 to be held to the SAME standard as B1/B2 (not stricter, not looser), the
schema's per-action required fields must EQUAL the validator's per-action
requirements. This test locks that invariant so future edits to either side
cannot silently re-introduce cross-baseline asymmetry.
"""
import pytest

from p79.agents.proxy_api_agent import _WEB_ACTION_TOOL
from p79.backends.action_utils import validate_action_detailed


# (action_type, schema-minimal VALID action, deficient action missing a required field)
_CASES = [
    ("click", {"action_type": "click", "element_id": 1}, {"action_type": "click"}),
    ("type", {"action_type": "type", "element_id": 1}, {"action_type": "type"}),
    ("hover", {"action_type": "hover", "element_id": 1}, {"action_type": "hover"}),
    ("select_option",
     {"action_type": "select_option", "element_id": 1, "option_label": "X"},
     {"action_type": "select_option", "element_id": 1}),
    ("scroll", {"action_type": "scroll", "scroll_direction": "down"}, {"action_type": "scroll"}),
    ("tab_focus", {"action_type": "tab_focus", "page_number": 1}, {"action_type": "tab_focus"}),
    ("press", {"action_type": "press", "key": "Enter"}, {"action_type": "press"}),
    ("goto", {"action_type": "goto", "url": "/x"}, {"action_type": "goto"}),
]


@pytest.mark.parametrize("at,good,bad", _CASES, ids=[c[0] for c in _CASES])
def test_schema_minimal_satisfies_validator(at, good, bad):
    """A schema-minimal action validates, and dropping a schema-required field fails."""
    _, valid_good, _ = validate_action_detailed(dict(good))
    assert valid_good is True, f"{at}: schema-minimal action rejected by validator"
    _, valid_bad, reason = validate_action_detailed(dict(bad))
    assert valid_bad is False, f"{at}: deficient action wrongly accepted by validator"
    assert reason, f"{at}: rejection must carry a sub-category reason"


def test_schema_not_stricter_than_validator_for_type_text():
    """Regression guard: `type` must NOT require `text` (validator allows
    type-with-element_id-no-text). A stricter schema would force B0 alone to emit
    text, an asymmetry vs B1/B2."""
    params = _WEB_ACTION_TOOL["function"]["parameters"]
    for clause in params.get("allOf", []):
        cond = clause.get("if", {}).get("properties", {}).get("action_type", {})
        types = cond.get("enum", []) + ([cond["const"]] if "const" in cond else [])
        if "type" in types:
            then = clause["then"]
            assert "text" not in then.get("required", []), \
                "schema requires `text` for type but validator does not (B0-only over-strictness)"


def test_schema_covers_all_validator_required_actions():
    """Every action_type the validator has a required field for must appear in an
    allOf clause (so forcing can't drop it)."""
    params = _WEB_ACTION_TOOL["function"]["parameters"]
    covered = set()
    for clause in params.get("allOf", []):
        cond = clause.get("if", {}).get("properties", {}).get("action_type", {})
        covered.update(cond.get("enum", []))
        if "const" in cond:
            covered.add(cond["const"])
    for at, _, _ in _CASES:
        assert at in covered, f"{at} has a validator requirement but no schema clause"


# B-1796 (P0-1, /stress 2026-05-21 Claude Mode A OOB): reverse-direction
# invariant. The forward tests above only prove schema-minimal ⊆ validator-valid.
# The fix's contract is schema == validator (bidirectional), so we ALSO lock
# validator-valid ⊆ schema-accepted for the per-action GROUNDING clauses. The
# canonical witness is select_option-by-coordinate: the validator accepts it
# (coordinate path, no element_id — action_utils.py:502), and the pre-B-1796
# schema rejected it (required element_id), a VISION-mode B0-only over-strictness.
# Top-level `thought` is intentionally NOT checked — it is an always-on reasoning
# field required of every baseline by prompt convention, not a per-action
# grounding requirement the validator gates on.
def _then_satisfied(action, then):
    if "required" in then and not all(k in action for k in then["required"]):
        return False
    if "anyOf" in then and not any(_then_satisfied(action, s) for s in then["anyOf"]):
        return False
    if "allOf" in then and not all(_then_satisfied(action, s) for s in then["allOf"]):
        return False
    return True


def _schema_grounding_accepts(action):
    """True iff `action` satisfies every applicable per-action allOf clause."""
    params = _WEB_ACTION_TOOL["function"]["parameters"]
    at = action.get("action_type")
    for clause in params.get("allOf", []):
        cond = clause.get("if", {}).get("properties", {}).get("action_type", {})
        types = list(cond.get("enum", [])) + ([cond["const"]] if "const" in cond else [])
        if at in types and not _then_satisfied(action, clause["then"]):
            return False
    return True


# validator-valid representatives, incl. the coordinate path the validator allows.
_REVERSE_CASES = [
    {"action_type": "click", "coordinate": [0.5, 0.5]},
    {"action_type": "type", "coordinate": [0.5, 0.5], "text": "x"},
    {"action_type": "hover", "coordinate": [0.5, 0.5]},
    # P0-1 canonical witness: select_option by coordinate (no element_id).
    {"action_type": "select_option", "coordinate": [0.5, 0.5], "option_label": "X"},
    {"action_type": "select_option", "element_id": 1, "option_index": 2},
    {"action_type": "scroll", "scroll_direction": "up"},
    {"action_type": "tab_focus", "page_number": 1},
    {"action_type": "press", "key": "Enter"},
    {"action_type": "goto", "url": "/x"},
    {"action_type": "back"},
    {"action_type": "finish", "answer": "a"},
    {"action_type": "wait"},
]


@pytest.mark.parametrize(
    "action", _REVERSE_CASES,
    ids=[c["action_type"] + ("_coord" if "coordinate" in c else "") for c in _REVERSE_CASES],
)
def test_validator_valid_action_satisfies_schema_grounding(action):
    """Reverse direction: a validator-valid action must satisfy the B0 tool
    schema's per-action grounding clauses (else schema is stricter than the
    validator → B0-only over-strictness, the P0-1 select_option-by-coordinate
    asymmetry)."""
    _, valid, reason = validate_action_detailed(dict(action))
    assert valid, f"test fixture bug: {action} should be validator-valid (got {reason})"
    assert _schema_grounding_accepts(action), (
        f"{action['action_type']}: validator-valid but B0 schema rejects "
        f"(schema stricter than validator — cross-baseline asymmetry)"
    )
