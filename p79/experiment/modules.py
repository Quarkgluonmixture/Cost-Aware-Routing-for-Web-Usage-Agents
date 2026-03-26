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


def m3_retry_action(
    failed_action: Optional[Dict[str, Any]] = None,
    obs_text: str = "",
) -> Dict[str, Any]:
    """Choose a retry action based on the type of failure."""
    failed_type = str((failed_action or {}).get("action_type", "")).lower()

    if failed_type == "click":
        # Click failed — scroll to reveal the target element
        return {
            "action_type": "scroll",
            "delta": [0, 0.5],
            "coordinate_type": "normalized",
            "thought": "M3 retry: click failed, scroll down to reveal target.",
        }
    elif failed_type == "type":
        # Type failed — try clicking the input field first
        eid = first_element_id_by_keyword(obs_text, ("textbox", "input", "search", "edit"))
        if eid is not None:
            return {
                "action_type": "click",
                "element_id": eid,
                "thought": "M3 retry: type failed, click input field to focus.",
            }
        return {
            "action_type": "scroll",
            "delta": [0, 0.3],
            "coordinate_type": "normalized",
            "thought": "M3 retry: type failed, no input found, scroll to reveal.",
        }
    elif failed_type in ("go_back", "goto"):
        # Navigation failed — wait briefly via a no-op scroll
        return {
            "action_type": "scroll",
            "delta": [0, 0.0],
            "coordinate_type": "normalized",
            "thought": "M3 retry: navigation failed, minimal scroll as wait.",
        }
    else:
        # Default: scroll to gather more context
        return {
            "action_type": "scroll",
            "delta": [0, 0.8],
            "coordinate_type": "normalized",
            "thought": "M3 retry: force one corrective scroll.",
        }
