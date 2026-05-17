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
}


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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

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
    # Path 2a: prefer fenced ```json {...} ``` block (common when models echo
    # the system-prompt "Output ONLY valid JSON" with markdown despite the
    # instruction). Pick first fenced block that validates.
    # B-413 (/stress A1.2 v8 Mode B P1-6, 2026-05-16): use detailed validator
    # so repair path failures carry sub-category reason (was: 2-tuple
    # validate_action() dropped reason, fenced/raw_decode failures collapsed
    # to generic `invalid_action_repaired`).
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            parsed = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        action, is_valid, _reason = validate_action_detailed(parsed)
        if is_valid:
            return action, True, "repaired_fenced"

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
    # B-406: explicit coordinate_type enforcement (new strict path).
    if coordinate_type == "normalized":
        return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0
    if coordinate_type == "pixel":
        return x >= 0 and y >= 0
    # Backward-compat fallback (coordinate_type is None — caller didn't pass).
    if allow_pixel:
        return x >= 0 and y >= 0
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


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
    """
    if not isinstance(delta, (list, tuple)):
        return False
    if len(delta) != 2:
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

    if action_type == "select_option":
        # B-506 (/stress A1.25 GRL Chunk 3 P0-1-B* codex OOB, 2026-05-17):
        # element_id must be `int > 0`. Pre-fix `isinstance(int)` alone
        # accepted `0` and `-1` (common LLM sentinel emissions) as valid
        # targets; wrapper's legacy keyboard-fallback then typed into
        # focused element silently, producing `parse_valid=true` records
        # with no actual element_id-based dispatch. Closes paper §3
        # action-primitive evidence-layer hole.
        _eid = action.get("element_id")
        has_id = isinstance(_eid, int) and _eid > 0
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
            or (isinstance(action.get("option_index"), int))
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
        has_id = isinstance(elem_id, int) and elem_id > 0
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
        has_id = isinstance(elem_id, int) and elem_id > 0
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
        if (
            delta is None
            and scroll_dir not in {"up", "down"}
            and (wa_direction or "").lower() not in {"up", "down", "left", "right"}
        ):
            return {"action_type": "wait"}, False, "invalid_schema_dict"

    if action_type == "tab_focus":
        page_no = action.get("page_number")
        if not isinstance(page_no, int) or page_no < 0:
            return {"action_type": "wait"}, False, "invalid_schema_dict"

    if action_type in ("finish", "stop"):
        answer = action.get("answer", "")
        action["answer"] = "" if answer is None else str(answer)

    return action, True, None


def validate_action(action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """Backward-compat 2-tuple wrapper around ``validate_action_detailed``.

    Existing callers (tests, proxy_api_agent, runner) keep their 2-tuple
    unpacking; new callsites that need the failure_reason discriminator
    call ``validate_action_detailed`` directly.
    """
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
    quoted = re.findall(r"['\"]([^'\"]+)['\"]", instruction or "")
    if quoted:
        return quoted[0].strip()

    instruction = (instruction or "").strip()
    for prefix in ("find", "search for", "look for", "buy", "add"):
        if instruction.lower().startswith(prefix):
            return instruction[len(prefix):].strip(" .")
    return instruction[:80].strip()
