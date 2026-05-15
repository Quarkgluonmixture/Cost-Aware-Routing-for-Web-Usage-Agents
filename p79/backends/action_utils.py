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
    try:
        parsed = json.loads(text)
        action, is_valid = validate_action(parsed)
        return action, is_valid, None if is_valid else "invalid_action"
    except json.JSONDecodeError:
        pass

    # B-141 (/stress A1.1 v8 codex F6, 2026-05-15): parser robust repair.
    # Path 2a: prefer fenced ```json {...} ``` block (common when models echo
    # the system-prompt "Output ONLY valid JSON" with markdown despite the
    # instruction). Pick first fenced block that validates.
    for m in _FENCED_JSON_RE.finditer(text):
        try:
            parsed = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        action, is_valid = validate_action(parsed)
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
    candidates = []
    for parsed, _start in _iter_json_objects(text):
        action, is_valid = validate_action(parsed)
        candidates.append((action, is_valid))

    valid_candidates = [c for c in candidates if c[1]]
    if len(valid_candidates) >= 2:
        # Ambiguous — multiple parseable actions. If all share the same
        # action_type + key identifiers, treat as single action (model
        # repetition). Otherwise flag ambiguity explicitly.
        firsts = {(a.get("action_type"), a.get("element_id"), a.get("text", "")) for a, _ in valid_candidates}
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
    # If raw_decode found candidates but none validated (e.g. unknown
    # action_type, malformed coord), surface that as repaired-but-invalid
    # so analysis can distinguish "no JSON at all" from "JSON but invalid".
    if candidates:
        return {"action_type": "wait", "thought": thought}, False, "invalid_action_repaired"
    return {"action_type": "wait", "thought": thought}, False, "parse_failed"


def _is_valid_coordinate_pair(coord: Any, allow_pixel: bool = True) -> bool:
    """B-142 (/stress A1.1 v8 codex F7, 2026-05-15): per-action strict coord
    shape check. Normalized coords must be 2 finite floats in [0,1]; pixel
    coords must be 2 non-negative finite numbers. Previously the validator
    only checked coord presence (`coord is not None`), so malformed payloads
    like `[2, "x"]` or `[None, None]` or `"42,7"` passed validation and
    reached env.step → schema failure converted into env behavior /
    no-progress → cross-baseline parse/invalid-action accounting polluted.
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
    if allow_pixel:
        # Pixel coords just need to be non-negative finite. Normalized coords
        # are a subset (0 ≤ x,y ≤ 1) — we accept both since coordinate_type
        # may declare either. Negative pixel coords always invalid.
        return x >= 0 and y >= 0
    # Normalized-only strict check (used when coordinate_type explicitly says so)
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


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


def validate_action(action: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    if not isinstance(action, dict):
        return {"action_type": "wait"}, False

    action_type = str(action.get("action_type", "wait")).lower().strip()
    if action_type == "stop":
        action_type = "finish"
    if action_type not in ALLOWED_ACTION_TYPES:
        return {"action_type": "wait"}, False

    action["action_type"] = action_type

    if action_type == "select_option":
        has_id = "element_id" in action and isinstance(action.get("element_id"), int)
        coord = action.get("coordinate")
        has_coord = coord is not None and _is_valid_coordinate_pair(coord)
        if not has_id and not has_coord:
            return {"action_type": "wait"}, False
        has_option = bool(
            action.get("option_label") or action.get("option_value")
            or (isinstance(action.get("option_index"), int))
        )
        if not has_option:
            return {"action_type": "wait"}, False
        if has_coord and "coordinate_type" not in action:
            action["coordinate_type"] = "normalized"

    if action_type == "click":
        coord = action.get("coordinate")
        elem_id = action.get("element_id")
        has_id = elem_id is not None and isinstance(elem_id, int)
        has_coord = coord is not None and _is_valid_coordinate_pair(coord)
        if not has_id and not has_coord:
            return {"action_type": "wait"}, False
        if coord is not None and not has_coord:
            # coord supplied but malformed — reject (was: silently accepted)
            return {"action_type": "wait"}, False
        if has_coord and "coordinate_type" not in action:
            action["coordinate_type"] = "normalized"

    if action_type == "type":
        action["text"] = str(action.get("text", ""))
        # Vision mode may supply a coordinate to indicate which input field to target.
        coord = action.get("coordinate")
        if coord is not None and not _is_valid_coordinate_pair(coord):
            return {"action_type": "wait"}, False
        if coord is not None and "coordinate_type" not in action:
            action["coordinate_type"] = "normalized"

    if action_type == "scroll":
        delta = action.get("delta")
        if delta is not None and not _is_valid_delta_pair(delta):
            return {"action_type": "wait"}, False

    if action_type == "tab_focus":
        page_no = action.get("page_number")
        if not isinstance(page_no, int) or page_no < 0:
            return {"action_type": "wait"}, False

    if action_type in ("finish", "stop"):
        answer = action.get("answer", "")
        action["answer"] = "" if answer is None else str(answer)

    return action, True


def first_element_id_by_keyword(obs_text: str, keywords: Tuple[str, ...]) -> Optional[int]:
    for line in (obs_text or "").splitlines():
        lower = line.lower()
        if not any(k in lower for k in keywords):
            continue
        match = re.search(r"\[(\d+)\]", line)
        if match:
            return int(match.group(1))
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
