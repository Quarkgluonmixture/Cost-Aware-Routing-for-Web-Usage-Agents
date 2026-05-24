from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple

ALLOWED_ACTION_TYPES = {
    "click",
    "type",
    "scroll",
    "wait",
    "back",
    "forward",
    "finish",
    "stop",
    "tab_focus",
    "select_option",
    # Protocol Reset #5 (action-set restore, 2026-05-20): upstream-compatible
    # id-based action space restored for paper-grade VWA fidelity. These were
    # dropped by the P79 custom prompt; cls/reddit empirically use tab_focus
    # for cross-site (pre-opened |AND| tabs) so demand is ~0, but restoring
    # closes the "crippled action space" reviewer attack. Execution: hover/
    # press/new_tab/close_tab route through the wrapper escape-hatch to upstream
    # `create_id_based_action`; goto has an explicit wrapper branch with a VWA
    # domain whitelist (off-site goto → no-op). See 实验笔记 §245.
    "hover",
    "press",
    "new_tab",
    "close_tab",
    "goto",
}


def _is_strict_int(value: Any) -> bool:
    """B-799 (/stress A1.2 cold-start P0-1-B* codex OOB, 2026-05-17): strict
    int test that REJECTS Python ``bool`` (which is an ``int`` subclass).

    Pre-fix the per-action validators used ``isinstance(x, int)``, so a JSON
    payload like ``{"action_type":"click","element_id":true}`` validated:
    Python evaluates ``isinstance(True, int) == True`` then ``True > 0 ==
    True`` → silently dispatched to element_id=1. Same pattern at
    ``tab_focus page_number``, ``select_option option_index``, and inside
    coord/delta lists (``isinstance(False, (int,float))``). Empirical codex
    probe 4/4 attack surfaces returned ``valid=True``; paper §3.5 sub-
    category enum (parse_valid + invalid_element_id) cannot honestly
    report this without strict typing.

    Use this helper at every integer-schema gate.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _extract_fallback_thought(text: str, max_len: int = 500) -> str:
    """Extract thought from raw model output when JSON parsing fails.

    Tries to find a "thought" value in partial/malformed JSON first,
    then falls back to the raw text (truncated).
    """
    m = re.search(r'"thought"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if m:
        return m.group(1)[:max_len]
    return text.strip()[:max_len]


_FENCED_JSON_RE = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _iter_json_objects(text: str):
    """Yield (object, start_index) for every JSON object that can be parsed
    starting at any '{' position in `text`. Uses ``json.JSONDecoder().raw_decode``
    which stops at the first complete object — this lets us find ALL candidate
    actions in a model output (multiple `{...}` blocks, JSON followed by prose,
    etc.) without the brittle greedy `\\{.*\\}` regex that captures everything
    from first `{` to last `}` including unrelated trailing content.
    """
    decoder = json.JSONDecoder()
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        yield obj, i
        i = end


def parse_action_text(text: str) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    text = (text or "").strip()
    # Strip <think>...</think> blocks emitted by extended-thinking models (e.g. Qwen3-235B).
    # Must happen before any JSON extraction so the regex below doesn't greedily capture
    # JSON fragments embedded inside the thinking block.
    # B-800 (/stress A1.2 cold-start P2-5-B codex, 2026-05-17): add IGNORECASE
    # so `<THINK>` / `<Think>` (uppercase or mixed) are also stripped. Pre-fix
    # case-sensitive regex left such blocks intact → JSON fragments inside
    # them surfaced as candidate actions to _iter_json_objects → spurious
    # multiple_actions classification on rare model emit pattern.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

    # Path 1: whole text is one valid JSON object (clean case).
    # B-167 (/stress A1.4a v8, 2026-05-16): use validate_action_detailed so the
    # sub-category reason (invalid_action_type / invalid_element_id /
    # invalid_coord / invalid_text / invalid_schema_dict / invalid_select_option)
    # surfaces in failure_reason instead of the catch-all "invalid_action".
    try:
        parsed = json.loads(text)
        action, is_valid, detail_reason = validate_action_detailed(parsed)
        return action, is_valid, None if is_valid else detail_reason
    except json.JSONDecodeError:
        pass

    # B-141 (/stress A1.1 v8 codex F6, 2026-05-15): parser robust repair.
    # Path 2a: prefer fenced ```json {...} ``` blocks (common when models echo
    # the system-prompt "Output ONLY valid JSON" with markdown despite the
    # instruction).
    # B-413 (/stress A1.2 v8 Mode B P1-6, 2026-05-16): use detailed validator
    # so repair path failures carry sub-category reason (was: 2-tuple
    # validate_action() dropped reason, fenced/raw_decode failures collapsed
    # to generic `invalid_action_repaired`).
    # B-801 (/stress A1.2 cold-start P0-4-B* codex OOB, 2026-05-17): collect
    # ALL fenced candidates and route through the same multiple_actions
    # ambiguity guard as raw_decode Path 2b. Pre-fix `return ...` on first
    # valid fenced block bypassed the guard entirely → empirical codex
    # probe: model emit two fenced `{click,eid=1}` and `{click,eid=2}` →
    # silent dispatch to eid=1 with reason="repaired_fenced", paper §3.5
    # multiple_actions disclosure contract false-negative on fenced path.
    fenced_candidates = []
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            parsed = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        fenced_action, fenced_valid, fenced_reason = validate_action_detailed(parsed)
        fenced_candidates.append((fenced_action, fenced_valid, fenced_reason))
    fenced_valid_actions = [c for c in fenced_candidates if c[1]]
    if len(fenced_valid_actions) == 1:
        return fenced_valid_actions[0][0], True, "repaired_fenced"
    if len(fenced_valid_actions) >= 2:
        # Ambiguity logic shared with raw_decode Path 2b: identical full-field
        # signature → repaired_multiple_identical; otherwise multiple_actions.
        _sig = lambda a: (
            a.get("action_type"),
            tuple(a.get("coordinate")) if isinstance(a.get("coordinate"), (list, tuple)) else None,
            a.get("coordinate_type"),
            a.get("element_id"),
            a.get("text", ""),
            tuple(a.get("delta")) if isinstance(a.get("delta"), (list, tuple)) else None,
            a.get("scroll_direction"),
            a.get("answer", ""),
            a.get("option_label", ""),
            a.get("option_value", ""),
            a.get("option_index"),
            a.get("page_number"),
            # P1-3-B* (cross-AI 2026-05-20): restored-action identifying fields.
            # Pre-fix two DIFFERENT press (key=Enter vs Escape) or two different
            # goto URLs collapsed to the same signature → repaired_multiple_
            # identical → silently executed the first instead of flagging
            # multiple_actions. Include key/url so distinct restored actions are
            # disambiguated like click/scroll/select_option already are.
            a.get("key") or a.get("key_comb"),
            a.get("url"),
        )
        sigs = {_sig(a) for a, _v, _r in fenced_valid_actions}
        if len(sigs) == 1:
            return fenced_valid_actions[0][0], True, "repaired_multiple_identical"
        thought = _extract_fallback_thought(text)
        first_action = fenced_valid_actions[0][0]
        first_action["thought"] = thought
        return first_action, False, "multiple_actions"

    # Path 2b: scan for ALL valid JSON objects in the text via raw_decode.
    # Previously the greedy regex `\{.*\}` captured first-{ to last-} which
    # silently mismatched on outputs like
    #   "{action} then maybe {alternative}" (captured "{...} then maybe {...}")
    # or "{action} // notes"  (captured "{action} // notes" → JSON error).
    # raw_decode finds each well-formed object boundary. Distinguish:
    #   • 0 valid candidates → parse_failed
    #   • 1 valid candidate → repair to that one
    #   • >1 valid candidates with DIFFERENT actions → ambiguity, explicit
    #     failure_reason="multiple_actions" instead of silently picking first
    # B-413: store (action, is_valid, reason) tuples so invalid candidates
    # surface the FIRST specific reason rather than `invalid_action_repaired`.
    candidates = []
    for parsed, _start in _iter_json_objects(text):
        action, is_valid, reason = validate_action_detailed(parsed)
        candidates.append((action, is_valid, reason))

    valid_candidates = [c for c in candidates if c[1]]
    if len(valid_candidates) >= 2:
        # Ambiguous — multiple parseable actions. If all share the same
        # action_type + ALL key identifying fields, treat as single action
        # (model repetition). Otherwise flag ambiguity explicitly.
        # B-409 (/stress A1.2 v8 Mode B P1-2 OOB unique, 2026-05-16): pre-fix
        # signature was only `(action_type, element_id, text)` so two clicks
        # at DIFFERENT coords were considered "identical" → executed first
        # → system bias toward first-hallucinated candidate (vision-mode
        # paper claim especially exposed). Post-fix: include coord / delta /
        # answer / option_* in signature, per user Q2=A "full-field".
        firsts = {
            (
                a.get("action_type"),
                tuple(a.get("coordinate")) if isinstance(a.get("coordinate"), (list, tuple)) else None,
                a.get("coordinate_type"),
                a.get("element_id"),
                a.get("text", ""),
                tuple(a.get("delta")) if isinstance(a.get("delta"), (list, tuple)) else None,
                a.get("scroll_direction"),
                a.get("answer", ""),
                a.get("option_label", ""),
                a.get("option_value", ""),
                a.get("option_index"),
                a.get("page_number"),
                # P1-3-B* (cross-AI 2026-05-20): restored-action identifying
                # fields — see fenced-path _sig note above. Keep both signature
                # tuples in lock-step.
                a.get("key") or a.get("key_comb"),
                a.get("url"),
            )
            for a, _v, _r in valid_candidates
        }
        if len(firsts) == 1:
            return valid_candidates[0][0], True, "repaired_multiple_identical"
        thought = _extract_fallback_thought(text)
        # Pick first valid for downstream (won't be executed if runner gates
        # on parse_valid, but action_type is still a reasonable best-guess).
        first_action = valid_candidates[0][0]
        first_action["thought"] = thought
        return first_action, False, "multiple_actions"
    if len(valid_candidates) == 1:
        return valid_candidates[0][0], True, "repaired_raw_decode"

    # Path 3 (parse_failed): no valid JSON object recoverable.
    # Keyword fallback removed by codex Mode B C3 (B-114) — ALL parse
    # failures fall through to wait with valid=False so failure taxonomy
    # is honest. Salvage thought for annotation only.
    thought = _extract_fallback_thought(text)
    # B-413: if raw_decode found candidates but none validated, surface the
    # FIRST candidate's specific reason (invalid_element_id / invalid_coord /
    # invalid_action_type / invalid_text / invalid_select_option /
    # invalid_schema_dict) rather than collapsing to `invalid_action_repaired`.
    # This unblocks paper §3.5 sub-category taxonomy on markdown/prose-wrapped
    # JSON case (common VLM output pattern).
    if candidates:
        first_invalid_reason = next((r for _a, _v, r in candidates if not _v and r), None)
        if first_invalid_reason is None:
            first_invalid_reason = "invalid_action_repaired"
        return {"action_type": "wait", "thought": thought}, False, first_invalid_reason
    return {"action_type": "wait", "thought": thought}, False, "parse_failed"


def _is_valid_coordinate_pair(
    coord: Any,
    coordinate_type: Optional[str] = None,
    allow_pixel: bool = True,
) -> bool:
    """B-142 + B-406 (/stress A1.2 v8 Mode A+B P0-1 2-AI OOB overlap,
    2026-05-16): per-action strict coord shape check with coordinate_type
    semantics enforcement. Normalized coords must be 2 finite floats in [0,1];
    pixel coords must be 2 non-negative finite numbers.

    B-406: pre-fix `allow_pixel=True` (default) accepted any non-negative
    finite pair regardless of declared `coordinate_type`. Empirical archive
    spot-check (Mode A + Mode B independent): 851/3561 normalized-declared
    rows had coord >1 (B0 15.6% / B1 35.3% — 2.3× cross-baseline asymmetry),
    all parse_valid=true. env wrapper `vwa_wrapper.py:336` then silently
    auto-normalizes by viewport division → schema violation collapsed into
    env behavior / no_progress → paper §3.5 error taxonomy contaminated and
    §1 hero number cross-baseline averaging biased.

    Post-fix: when `coordinate_type` is explicitly passed, strictly enforce
    the declared semantics. `coordinate_type is None` falls back to legacy
    `allow_pixel` behavior for backward compatibility with callsites that
    haven't been updated yet (those still get the old permissive check).
    """
    if not isinstance(coord, (list, tuple)):
        return False
    if len(coord) != 2:
        return False
    # B-799 (/stress A1.2 cold-start P0-1-B* codex OOB, 2026-05-17): reject
    # bool components BEFORE float coercion. Python `bool` is `int` subclass
    # → `float(True) == 1.0` silently succeeds → coord=[True, False] passed
    # the legacy shape check, then `x=1.0, y=0.0` slipped through the
    # normalized [0,1] gate. Cross-baseline action-shape aggregator counted
    # bool-typed coords as valid clicks at (1,0). Reject early.
    if isinstance(coord[0], bool) or isinstance(coord[1], bool):
        return False
    try:
        x = float(coord[0])
        y = float(coord[1])
    except (TypeError, ValueError):
        return False
    # NaN / inf
    if not (x == x) or not (y == y):
        return False
    if x in (float("inf"), float("-inf")) or y in (float("inf"), float("-inf")):
        return False
    # B-802 (/stress A1.2 cold-start P1-2-B codex non-OOB, 2026-05-17):
    # unknown coordinate_type enum must reject, not silently fall to pixel.
    # Pre-fix `coordinate_type="screen"` / `"css"` / typos passed via the
    # `allow_pixel` legacy fallback path → bogus enum survived into step
    # JSONL audit trail → paper §3.5 schema-taxonomy false expansion. Only
    # the canonical 2 enum members + `None` (legacy unset) are accepted.
    # (B-1860 keeps this schema-level enum reject: a garbage type string is
    # still true-malformed; the change below only drops the enum's authority
    # over the per-dimension VALUE-RANGE judgment.)
    if coordinate_type is not None and coordinate_type not in ("normalized", "pixel"):
        return False
    # B-1860: Qwen 0-1000 contract — judge each dimension BY VALUE, ignoring
    # the model's `coordinate_type` declaration (probe-confirmed unreliable:
    # the model stamps "normalized" while emitting 0-1000 coords). A dimension
    # `> 1.1` is a legal Qwen 0-1000 coordinate (NOT a [0,1] violation to
    # reject); a dimension `<= 1.1` is a legal normalized [0,1] coordinate.
    # Both regimes are non-negative finite numbers, so the single non-negative
    # check accepts both — and the env wrapper auto-judges per dimension again
    # (`<= 1.1` keep / `> 1.1` divide by 1000) to map to pixels. This replaces
    # the pre-fix B-406 hard `coordinate_type == "normalized" → [0,1]` reject
    # that was rejecting 0-1000 coords as parse errors (vision parse_error
    # 13.6%). True malformed (NaN / inf / non-number / wrong shape / bool /
    # NEGATIVE) is still rejected above + here. Save format layer, NOT
    # grounding layer (no target snapping / nearest-element correction).
    return x >= 0 and y >= 0


def _infer_coordinate_type(coord: Any) -> str:
    """B-452 (/stress A1.4 P1-1-B codex OOB, 2026-05-17): infer the natural
    coordinate_type for a coord pair when the caller did not declare one.

    Pre-fix the validator stamped any valid (positive finite) coord as
    ``coordinate_type="normalized"`` even when the values were obviously
    pixels (e.g. ``[100, 200]``). The env wrapper at
    ``vwa_wrapper.py:352-358`` then silently divides by viewport when
    ``max(x,y) > 1.0`` to recover pixel semantics, but the step JSONL
    audit trail still claimed ``"normalized"`` — paper §3 error-taxonomy
    aggregator and cross-baseline coord-failure analysis were mislabeled.

    Inference rule (cheap, no env access):
      - any coord component > 1.0  → ``"pixel"`` (large absolute values
        cannot be inside [0,1] normalized space)
      - all components ≤ 1.0       → ``"normalized"`` (canonical default)

    The caller is responsible for already validating the coord shape via
    ``_is_valid_coordinate_pair``; this helper assumes positive finite
    floats and just picks a label. Returns the inferred type string.
    """
    try:
        x = float(coord[0])
        y = float(coord[1])
    except (TypeError, ValueError, IndexError):
        # Defensive fallback — should never happen after _is_valid_coordinate_pair.
        return "normalized"
    if x > 1.0 or y > 1.0:
        return "pixel"
    return "normalized"


def _is_valid_delta_pair(delta: Any) -> bool:
    """B-142: scroll delta shape check. 2 finite floats; sign indicates
    direction; magnitude usually 0-1 normalized but pixel deltas are
    allowed for backward compat. Reject NaN / inf / non-numeric / wrong
    shape.

    B-799 (/stress A1.2 cold-start P0-1-B* codex OOB, 2026-05-17): also
    reject bool components — same JSON bool-as-int slip as coord path.
    B-803 (/stress A1.2 cold-start P1-5-B* codex OOB, 2026-05-17): also
    reject zero delta [0,0]. Pre-fix `delta:[0,0]` passed shape gate and
    the scroll canonicalizer at L476 then inferred `down` from `dy=0 > 0`
    being False → silent `scroll_direction="up"` valid record. paper §3
    action-distribution counted no-op scrolls as up-scrolls.
    """
    if not isinstance(delta, (list, tuple)):
        return False
    if len(delta) != 2:
        return False
    # B-799: reject bool components before float coerce.
    if isinstance(delta[0], bool) or isinstance(delta[1], bool):
        return False
    try:
        dx = float(delta[0])
        dy = float(delta[1])
    except (TypeError, ValueError):
        return False
    if not (dx == dx) or not (dy == dy):
        return False
    if dx in (float("inf"), float("-inf")) or dy in (float("inf"), float("-inf")):
        return False
    # B-803: zero-magnitude both-axes = no-op scroll. Reject so canonicalizer
    # cannot synthesize a fake direction enum.
    if dx == 0.0 and dy == 0.0:
        return False
    return True


def validate_action_detailed(action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    """B-167 (/stress A1.4a v8 Claude F3 expanded scope, 2026-05-16): detailed
    validation that emits a sub-category failure_reason. Pre-B-167
    ``validate_action`` returned only ``(action, valid)`` — runner had no way
    to distinguish *why* the action was invalid (action_type unknown vs
    element_id missing vs coord malformed vs schema gap), so paper §3.5
    error taxonomy collapsed every invalid emission into ``invalid_action``.

    Returns:
        (action, valid, reason)
        - valid=True → reason=None
        - valid=False → reason ∈ {"invalid_schema_dict",
            "invalid_action_type", "invalid_element_id", "invalid_coord",
            "invalid_text", "invalid_select_option"}

    Mapped by runner ``_normalize_error_category`` into the corresponding
    error_category enum. Router-aware escalation policy (per-category
    target mode) deferred to Phase 2 / paper-2 scope.
    """
    if not isinstance(action, dict):
        return {"action_type": "wait"}, False, "invalid_schema_dict"

    action_type = str(action.get("action_type", "wait")).lower().strip()
    if action_type == "stop":
        action_type = "finish"
    if action_type not in ALLOWED_ACTION_TYPES:
        return {"action_type": "wait"}, False, "invalid_action_type"

    action["action_type"] = action_type

    # B-572 (/stress A1.22 P1-14-B* codex OOB, 2026-05-17): element_id
    # digit-string → int canonicalization. LLM text-JSON output commonly
    # emits `"element_id": "12"` (quoted by JSON formatter habit). Pre-fix
    # `isinstance(int)` checks in the per-action branches rejected this as
    # `invalid_element_id` even though semantically identical to int 12.
    # B0 tool_use Path-1 uses `validate_action` which coerces via tool
    # schema → typed int; B1+B2 text JSON path through `parse_action_text
    # → json.loads → str/int as emitted` → asymmetric rejection rate
    # treating serialization style as capability. Coerce digit-string here
    # (BEFORE per-action validation) + record provenance via inline flag.
    # Reject only outside int range (negative, zero, non-digit).
    _eid_raw = action.get("element_id")
    if isinstance(_eid_raw, str) and _eid_raw.strip():
        _eid_stripped = _eid_raw.strip()
        # B-572 + B-804 (/stress A1.2 cold-start P1-4-A* Claude OOB,
        # 2026-05-17): strict regex match int-only — accept "12" / "+12",
        # reject "0" / "-1" / "007" / "1.0" / "1e3" / "0x12".
        # Pre-B-804 the inline comment claimed "leading-zero (007) all
        # rejected" but the actual implementation used
        # `lstrip("+").isdigit()` which returns True for "007" → silent
        # coerce to int 7. comment lied → paper §3.5 element_id_coerced_
        # from_string disclosure miscounted "007" as legitimate coercion
        # rather than malformed emission.
        if re.match(r"^\+?[1-9][0-9]*$", _eid_stripped):
            try:
                action["element_id"] = int(_eid_stripped)
                # Provenance flag so paper §3.5 parse_valid disclosure can
                # report "B1/B2 element_id coerced from string" count per
                # cell — measures the gap relative to B0 tool_use.
                action.setdefault("element_id_coerced_from_string", True)
            except (ValueError, TypeError):
                pass  # leave raw; per-action validator will reject

    # B-1101 (/stress A2.3b P0-1-AC* OOB, 2026-05-18): 1-element-list
    # coerce for B0 tool_calling. AWS Bedrock proxy does NOT enforce tools
    # schema on output — model self-decides emission format under
    # `tool_choice="auto"`. Empirical probe `docs/checkpoints/probes/
    # proxy_full_stack_225749.json` shows 30/30 runs emit
    # `"element_id": [37]` (1-element int list) instead of integer 37,
    # despite `_WEB_ACTION_TOOL.parameters.properties.element_id.type =
    # "integer"`. Pre-fix `validate_action_detailed` rejected list-typed
    # _eid via `_is_strict_int([37]) = False` → Path-2 text parse on
    # empty content → 30 wait/episode contamination → 0% SR. Coerce
    # strict-int 1-element list → int with explicit `len==1` guard
    # (reject 2-element `[37, 38]` to surface as `invalid_element_id`
    # rather than silent first-element pick). Mirror digit-string coerce
    # above (B-572) for paper-grade audit symmetry.
    if isinstance(_eid_raw, list) and len(_eid_raw) == 1 and _is_strict_int(_eid_raw[0]):
        action["element_id"] = _eid_raw[0]
        # Provenance flag so paper §3.5 parse_valid disclosure can report
        # "B0 element_id coerced from 1-element list" count per cell.
        action.setdefault("element_id_coerced_from_list", True)

    if action_type == "select_option":
        # B-506 (/stress A1.25 GRL Chunk 3 P0-1-B* codex OOB, 2026-05-17):
        # element_id must be `int > 0`. Pre-fix `isinstance(int)` alone
        # accepted `0` and `-1` (common LLM sentinel emissions) as valid
        # targets; wrapper's legacy keyboard-fallback then typed into
        # focused element silently, producing `parse_valid=true` records
        # with no actual element_id-based dispatch. Closes paper §3
        # action-primitive evidence-layer hole.
        _eid = action.get("element_id")
        # B-799 (/stress A1.2 cold-start P0-1-B* codex OOB, 2026-05-17):
        # strict-int helper rejects bool. Same fix at click/type/tab_focus
        # branches + option_index below.
        has_id = _is_strict_int(_eid) and _eid > 0
        coord = action.get("coordinate")
        coord_present = coord is not None
        # B-406: pass declared coordinate_type into validator so normalized
        # declarations strictly enforce [0,1]; absence falls back to legacy.
        coord_ctype = action.get("coordinate_type")
        coord_valid_shape = coord_present and _is_valid_coordinate_pair(
            coord, coordinate_type=coord_ctype
        )
        # B-167: priority — if agent INTENDED to use coord (supplied) but it's
        # malformed, surface as invalid_coord even when element_id is also
        # missing. Pre-fix the "neither id nor valid coord" branch fired first,
        # collapsing the specific coord-shape failure into invalid_element_id.
        if coord_present and not coord_valid_shape:
            return {"action_type": "wait"}, False, "invalid_coord"
        if not has_id and not coord_valid_shape:
            return {"action_type": "wait"}, False, "invalid_element_id"
        has_option = bool(
            action.get("option_label") or action.get("option_value")
            or _is_strict_int(action.get("option_index"))
        )
        if not has_option:
            return {"action_type": "wait"}, False, "invalid_select_option"
        if coord_valid_shape and "coordinate_type" not in action:
            # B-452 (/stress A1.4 P1-1-B codex OOB, 2026-05-17): infer the
            # natural coord type from values rather than blindly stamping
            # "normalized". Pixel inputs (e.g. [100, 200]) now correctly
            # label as "pixel"; env wrapper still auto-normalizes downstream
            # but the audit trail no longer lies.
            action["coordinate_type"] = _infer_coordinate_type(action["coordinate"])

    if action_type == "click":
        coord = action.get("coordinate")
        elem_id = action.get("element_id")
        # B-506 (/stress A1.25 GRL Chunk 3 P0-1-B*): element_id must be int > 0.
        # B-799 (/stress A1.2 cold-start P0-1-B* codex OOB, 2026-05-17): strict
        # int helper rejects bool (was: isinstance(int) accepted True/False).
        has_id = _is_strict_int(elem_id) and elem_id > 0
        coord_present = coord is not None
        coord_ctype = action.get("coordinate_type")
        coord_valid_shape = coord_present and _is_valid_coordinate_pair(
            coord, coordinate_type=coord_ctype
        )
        # B-167: priority — coord-present-but-malformed → invalid_coord
        # regardless of element_id presence. Specific reason beats generic.
        if coord_present and not coord_valid_shape:
            return {"action_type": "wait"}, False, "invalid_coord"
        if not has_id and not coord_valid_shape:
            return {"action_type": "wait"}, False, "invalid_element_id"
        if coord_valid_shape and "coordinate_type" not in action:
            # B-452 (/stress A1.4 P1-1-B codex OOB, 2026-05-17): infer the
            # natural coord type from values rather than blindly stamping
            # "normalized". Pixel inputs (e.g. [100, 200]) now correctly
            # label as "pixel"; env wrapper still auto-normalizes downstream
            # but the audit trail no longer lies.
            action["coordinate_type"] = _infer_coordinate_type(action["coordinate"])

    if action_type == "type":
        action["text"] = str(action.get("text", ""))
        # Vision mode may supply a coordinate to indicate which input field to target.
        coord = action.get("coordinate")
        elem_id = action.get("element_id")
        # B-506 (/stress A1.25 GRL Chunk 3 P0-1-B*): element_id must be int > 0.
        # B-799 (/stress A1.2 cold-start P0-1-B* codex OOB, 2026-05-17): strict
        # int helper rejects bool.
        has_id = _is_strict_int(elem_id) and elem_id > 0
        coord_ctype = action.get("coordinate_type")
        coord_valid_shape = coord is not None and _is_valid_coordinate_pair(
            coord, coordinate_type=coord_ctype
        )
        # B-406: coord-present-but-malformed → invalid_coord even when
        # element_id also missing. Specific reason beats generic.
        if coord is not None and not coord_valid_shape:
            return {"action_type": "wait"}, False, "invalid_coord"
        # B-407 (/stress A1.2 v8 Mode B P0-2 OOB unique, 2026-05-16): `type`
        # must declare a target — either `element_id:int` or a valid
        # `coordinate`. Pre-fix the validator only checked `text` was present
        # and accepted targetless type (empirical 23 rows in archive: B0 cls
        # task 204 step 1 `{"action_type":"type","text":"\n"}` parse_valid=true
        # error_category=no_progress). env wrapper cannot execute targetless
        # type → schema-violation silently grouped under no_progress → paper
        # §3.5 invalid-action taxonomy + cross-baseline SR contaminated.
        if not has_id and not coord_valid_shape:
            return {"action_type": "wait"}, False, "invalid_element_id"
        if coord_valid_shape and "coordinate_type" not in action:
            # B-452 (/stress A1.4 P1-1-B codex OOB, 2026-05-17): infer the
            # natural coord type from values rather than blindly stamping
            # "normalized". Pixel inputs (e.g. [100, 200]) now correctly
            # label as "pixel"; env wrapper still auto-normalizes downstream
            # but the audit trail no longer lies.
            action["coordinate_type"] = _infer_coordinate_type(action["coordinate"])

    if action_type == "scroll":
        delta = action.get("delta")
        scroll_dir = action.get("scroll_direction")
        # B-412 (/stress A1.2 v8 Mode B P1-5 unique, 2026-05-16): naked
        # `{"action_type":"scroll"}` was parse_valid=true (delta None branch
        # fell through). env wrapper `vwa_wrapper.py:356` (VWA path) requires
        # `delta` OR `scroll_direction` to execute. Empirical 2 rows in
        # archive (B1 cls task_174 step_1). Validator now requires at least
        # one of `delta` / `scroll_direction` / `direction` (`direction`
        # kept as WebArena-legacy alias used by `vwa_wrapper.py:800`).
        # Single source of truth at validator avoids cross-benchmark drift.
        if delta is not None and not _is_valid_delta_pair(delta):
            return {"action_type": "wait"}, False, "invalid_coord"
        wa_direction = action.get("direction")  # WA legacy alias
        # B-805 (/stress A1.2 cold-start P0-3-AB* Claude+codex 2-AI OOB,
        # 2026-05-17): entry validator must accept the FULL canonical enum
        # `{up,down,left,right}` (was `{up,down}` only). Pre-fix
        # `scroll_direction:"left"` was rejected as invalid_schema_dict
        # while `direction:"left"` (legacy alias) was accepted and
        # canonicalized to `scroll_direction:"left"` at L464 — same
        # semantic intent, two validity outcomes depending on field name.
        # Reviewer 5-line probe could cancel paper §3 action-taxonomy
        # claim. Align entry gate to canonical enum.
        if (
            delta is None
            and scroll_dir not in {"up", "down", "left", "right"}
            and (wa_direction or "").lower() not in {"up", "down", "left", "right"}
        ):
            return {"action_type": "wait"}, False, "invalid_schema_dict"
        # B-573 (/stress A1.22 P1-15-B* codex OOB, 2026-05-17): scroll
        # action shape canonicalization. Pre-fix step_record `action` could
        # carry `scroll_direction:"down"` (B0 tool-schema) OR `direction:
        # "down"` (WA legacy alias used by B1/B2 free-form JSON) OR
        # `delta:[0,0.8]` (B1/B2 native VLM emit). Three shapes for the
        # same semantic intent → cross-baseline action-shape aggregator
        # (paper §3 action-taxonomy) groups by ANY ONE shape misses the
        # others. Now: validator emits canonical `scroll_direction` enum
        # `{up,down,left,right}` (`down` default for legacy delta with
        # `dy>0`). Legacy `direction` field preserved as
        # `direction_raw_alias` so reviewer can audit which baseline
        # emitted which form; canonical `scroll_direction` is the
        # cross-baseline-stable consumer field.
        if "scroll_direction" not in action or action["scroll_direction"] not in {"up", "down", "left", "right"}:
            _canonical: Optional[str] = None
            if isinstance(wa_direction, str) and wa_direction.lower() in {
                "up", "down", "left", "right"
            }:
                _canonical = wa_direction.lower()
                # Preserve WA-legacy alias for reviewer audit; canonical
                # field is `scroll_direction`.
                action["direction_raw_alias"] = wa_direction
            elif isinstance(delta, (list, tuple)) and len(delta) == 2:
                try:
                    _dy = float(delta[1])
                    _canonical = "down" if _dy > 0 else "up"
                except (ValueError, TypeError):
                    _canonical = None
            if _canonical is not None:
                action["scroll_direction"] = _canonical

    if action_type == "tab_focus":
        page_no = action.get("page_number")
        # B-799 (/stress A1.2 cold-start P0-1-B* codex OOB, 2026-05-17): strict
        # int helper rejects bool. Pre-fix `page_number:true` validated to 1.
        if not _is_strict_int(page_no) or page_no < 0:
            return {"action_type": "wait"}, False, "invalid_schema_dict"

    if action_type == "hover":
        # Protocol Reset #5 (2026-05-20): hover is click-like — needs an
        # element_id (int > 0) or a valid coordinate. Reuses the click coord
        # priority so coord-present-but-malformed surfaces as invalid_coord.
        coord = action.get("coordinate")
        elem_id = action.get("element_id")
        has_id = _is_strict_int(elem_id) and elem_id > 0
        coord_present = coord is not None
        coord_ctype = action.get("coordinate_type")
        coord_valid_shape = coord_present and _is_valid_coordinate_pair(
            coord, coordinate_type=coord_ctype
        )
        if coord_present and not coord_valid_shape:
            return {"action_type": "wait"}, False, "invalid_coord"
        if not has_id and not coord_valid_shape:
            return {"action_type": "wait"}, False, "invalid_element_id"
        if coord_valid_shape and "coordinate_type" not in action:
            action["coordinate_type"] = _infer_coordinate_type(action["coordinate"])

    if action_type == "press":
        # Protocol Reset #5 (2026-05-20): press needs a non-empty key string.
        # Accept `key` (canonical) or `key_comb` (upstream id-based alias).
        # Canonicalize onto `key` so the wrapper serializer reads one field.
        key = action.get("key") or action.get("key_comb")
        if not isinstance(key, str) or not key.strip():
            return {"action_type": "wait"}, False, "invalid_schema_dict"
        action["key"] = key.strip()

    if action_type == "goto":
        # Protocol Reset #5 (2026-05-20): goto needs a non-empty url string.
        # The VWA-domain whitelist is a RUNTIME policy enforced in the env
        # wrapper (`vwa_wrapper._goto_allowed_hosts`) because it depends on the
        # configured site hosts + currently-open tabs — not a pure schema fact.
        # Here we only assert schema validity (non-empty string url).
        url = action.get("url")
        if not isinstance(url, str) or not url.strip():
            return {"action_type": "wait"}, False, "invalid_schema_dict"
        action["url"] = url.strip()

    # new_tab / close_tab take no arguments — always schema-valid once the
    # action_type passed the ALLOWED_ACTION_TYPES gate above.

    if action_type in ("finish", "stop"):
        answer = action.get("answer", "")
        action["answer"] = "" if answer is None else str(answer)

    return action, True, None


def validate_action(action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Backward-compat 2-tuple wrapper around ``validate_action_detailed``.

    Existing callers (tests, proxy_api_agent, runner) keep their 2-tuple
    unpacking; new callsites that need the failure_reason discriminator
    call ``validate_action_detailed`` directly.

    B-806 (/stress A1.2 cold-start P1-8-A Claude, 2026-05-17): the 2-tuple
    shim silently drops ``failure_reason``. Re-validation callsites
    (runner/env wrapper round-trip checks) lose the sub-category enum →
    paper §3.5 disclosure does not see those failures. DeprecationWarning
    fires once per process at first call so the audit trail surfaces
    remaining callsites; migration target is ``validate_action_detailed``.
    """
    import warnings
    warnings.warn(
        "validate_action() drops failure_reason — migrate to "
        "validate_action_detailed() to preserve paper §3.5 sub-category enum",
        DeprecationWarning,
        stacklevel=2,
    )
    action, valid, _reason = validate_action_detailed(action)
    return action, valid


_ROLE_RE = re.compile(r"\[\s*\d+\s*\]\s+(\S+)")


def first_element_id_by_keyword(obs_text: str, keywords: Tuple[str, ...]) -> Optional[int]:
    # /stress A1.10 P1-2-AB* (2026-05-16): replaced unanchored
    # `re.search(r"\[(\d+)\]", line)` with canonical anchored extractor.
    # Pre-fix attacked AXTree StaticText containing bracketed digits in label
    # content — e.g. line "[10] StaticText 'see [4] section'" returned eid=10
    # (OK), but line "[StaticText 'click 12 for details']" containing the
    # `combobox` keyword via parent context could return eid=12 from text.
    #
    # B-414 (/stress A1.2 v8 Mode B P2-1 OOB, 2026-05-16): role-anchored
    # keyword match. Pre-A1.10 fix anchored mark id extraction, but role
    # keyword check (`any(k in lower for k in keywords)`) still matched
    # against the WHOLE line including label / body / url. e.g. line
    # `[12] StaticText 'click the blue button'` matched "button" → returned
    # eid=12 (StaticText, not clickable). M1/M2 fallback + heuristic_dom
    # could pick StaticText as click target → fallback dead-path. Now: parse
    # the role token after `[N]` and match keyword ONLY against the
    # whitespace-normalized role string. Sibling propagation: 7 callsites
    # share this primitive (heuristic.py x2, modules.py M1, runner/helpers
    # M2, etc.) — single-source fix.
    from p79.experiment.som import extract_mark_id
    normalized_keywords = [k.lower().replace(" ", "").replace("_", "") for k in keywords]
    for line in (obs_text or "").splitlines():
        eid = extract_mark_id(line)
        if eid is None:
            continue
        m = _ROLE_RE.match(line.strip())
        if not m:
            continue
        role_norm = m.group(1).lower().replace(" ", "").replace("_", "")
        if any(k in role_norm for k in normalized_keywords):
            return eid
    return None


def extract_candidate_query(instruction: str) -> str:
    """Extract a fallback search query string from a task instruction.

    B-807 (/stress A1.2 cold-start P2-9-A Claude, 2026-05-17): removed
    silent 80-char truncation. Pre-fix shop/WA instructions with 200+ char
    target descriptions had the latter half silently dropped → fallback
    `no_progress` misclassification when the long context contained the
    distinguishing search keyword. Truncation was load-bearing only for
    the historical no-quoted-keyword-no-recognized-prefix fallback; the
    full string is acceptable downstream and any downstream length cap
    should be enforced explicitly at the consumer (not silently here).
    """
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", instruction or "")
    if quoted:
        return quoted[0].strip()

    instruction = (instruction or "").strip()
    for prefix in ("find", "search for", "look for", "buy", "add"):
        if instruction.lower().startswith(prefix):
            return instruction[len(prefix):].strip(" .")
    return instruction.strip()
