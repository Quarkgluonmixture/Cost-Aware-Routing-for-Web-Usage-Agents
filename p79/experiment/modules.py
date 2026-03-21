from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from p79.backends.action_utils import first_element_id_by_keyword


def apply_m1_dom_select_fallback(action: Dict[str, Any], obs_text: str, enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return action
    if action.get("action_type") != "click" or "element_id" in action:
        return action

    eid = first_element_id_by_keyword(obs_text, ("combobox", "select", "option", "dropdown"))
    if eid is None:
        return action

    patched = dict(action)
    patched["element_id"] = eid
    patched["m1_patch"] = True
    return patched


def apply_m2_dom_first_input_fallback(action: Dict[str, Any], obs_text: str, enabled: bool) -> Dict[str, Any]:
    if not enabled:
        return action
    if action.get("action_type") != "type" or "element_id" in action:
        return action

    eid = first_element_id_by_keyword(obs_text, ("textbox", "input", "search", "edit"))
    if eid is None:
        return action

    patched = dict(action)
    patched["element_id"] = eid
    patched["m2_patch"] = True
    return patched


def apply_secondary_modules(action: Dict[str, Any], obs_text: str, module_flags: Dict[str, bool]) -> Dict[str, Any]:
    out = dict(action)
    out = apply_m1_dom_select_fallback(out, obs_text, module_flags.get("m1_dom_select_fallback", False))
    out = apply_m2_dom_first_input_fallback(out, obs_text, module_flags.get("m2_dom_first_input_fallback", False))
    return out


def should_trigger_m3_retry(
    action_success: bool,
    page_changed: bool,
    retry_count: int,
    retry_limit: int,
    module_flags: Dict[str, bool],
) -> bool:
    if not module_flags.get("m3_failure_trigger_retry", False):
        return False
    if retry_count >= retry_limit:
        return False
    if action_success:
        return False
    if page_changed:
        return False
    return True


def m3_retry_action() -> Dict[str, Any]:
    return {
        "action_type": "scroll",
        "delta": [0, 0.8],
        "coordinate_type": "normalized",
        "thought": "M3 retry policy: force one corrective scroll.",
    }
