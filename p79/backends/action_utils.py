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


def parse_action_text(text: str) -> Tuple[Dict[str, Any], bool, Optional[str]]:
    text = (text or "").strip()
    # Strip <think>...</think> blocks emitted by extended-thinking models (e.g. Qwen3-235B).
    # Must happen before any JSON extraction so the regex below doesn't greedily capture
    # JSON fragments embedded inside the thinking block.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    lowered = text.lower()

    try:
        parsed = json.loads(text)
        action, is_valid = validate_action(parsed)
        return action, is_valid, None if is_valid else "invalid_action"
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            action, is_valid = validate_action(parsed)
            return action, is_valid, "repaired_regex" if is_valid else "invalid_action_repaired"
        except json.JSONDecodeError:
            pass

    # Keyword fallback: salvage thought from raw text for annotation/debug only.
    # The action itself falls through to `wait` — keyword scroll/back fallback
    # was removed by /stress A1.1 codex Mode B C3 (2026-05-15):
    #
    # `_action` returned with `valid=False` but `action_type` set to `scroll` /
    # `back` was still executed downstream (runner records `parse_valid=False`
    # but does NOT skip env.step). Cross-validate Q4 confirmed archived runs
    # contain `keyword_scroll/back` episodes with `action_success=True` — silent
    # partial automation where ~1.5% of fp-population steps were substring-match
    # driven, not model-action driven. Behaviour mirrors §67 `keyword_finish`
    # removal (incidental "finish"/"stop" in unparseable text is not an action).
    # Now ALL parse failures fall through to wait with valid=False.
    thought = _extract_fallback_thought(text)

    return {"action_type": "wait", "thought": thought}, False, "parse_failed"


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
        has_id = "element_id" in action
        has_coord = "coordinate" in action
        if not has_id and not has_coord:
            return {"action_type": "wait"}, False
        has_option = bool(
            action.get("option_label") or action.get("option_value")
            or action.get("option_index") is not None
        )
        if not has_option:
            return {"action_type": "wait"}, False
        if has_coord and "coordinate_type" not in action:
            action["coordinate_type"] = "normalized"

    if action_type == "click":
        coord = action.get("coordinate")
        if coord is None and "element_id" not in action:
            return {"action_type": "wait"}, False
        if coord is not None and "coordinate_type" not in action:
            action["coordinate_type"] = "normalized"

    if action_type == "type":
        action["text"] = str(action.get("text", ""))
        # Vision mode may supply a coordinate to indicate which input field to target.
        # Keep the coordinate fields so the environment can optionally use them.
        coord = action.get("coordinate")
        if coord is not None and "coordinate_type" not in action:
            action["coordinate_type"] = "normalized"

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
