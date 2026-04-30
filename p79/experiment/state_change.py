from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from p79.envs.vwa_wrapper import P79Observation

_TEXT_TRUNCATION_LIMIT = 5000


# Adapted from external_code/page_state_utils.py (Aiden Yiliu Li, Apache-2.0)
def _safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _extract_interactive_count(text: str) -> int:
    if not text:
        return 0
    # AXTree lines typically contain [id] markers for interactable nodes.
    return len(re.findall(r"\[(\d+)\]", text))


def _extract_form_fields_count(text: str) -> int:
    if not text:
        return 0
    markers = ("textbox", "input", "textarea", "select", "combobox", "search")
    low = text.lower()
    return sum(low.count(m) for m in markers)


def _extract_focused_tag(text: str) -> Optional[str]:
    """Extract the tag/role of the focused element from AXTree text."""
    if not text:
        return None
    m = re.search(r"focused.*?\b(\w+)(?:\s|$)", text, re.IGNORECASE)
    return m.group(1) if m else None


def _extract_modal_state(text: str) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(k in low for k in ("dialog", "modal", "popup", "overlay", "aria-modal"))


def _extract_title_from_html(html: str) -> str:
    """Extract <title> from HTML content."""
    if not html:
        return ""
    m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else ""


def build_page_state(
    obs: P79Observation,
    info: Optional[Dict[str, Any]],
    form_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    text = _safe_str(getattr(obs, "text", ""))
    info = info or {}
    fs = form_snapshot or {}

    # VWA stores url/content inside info["page"] (DetachedPage dataclass)
    page_obj = info.get("page")
    page_url = ""
    page_title = ""
    if page_obj is not None:
        if hasattr(page_obj, "url"):
            page_url = _safe_str(page_obj.url)
        if hasattr(page_obj, "content"):
            page_title = _extract_title_from_html(_safe_str(page_obj.content))

    state = {
        "url": _safe_str(info.get("url") or info.get("current_url") or getattr(obs, "url", "")) or page_url,
        "title": _safe_str(info.get("title") or info.get("page_title") or "") or page_title,
        "visible_text": text[:_TEXT_TRUNCATION_LIMIT],
        "interactive_elements_count": _extract_interactive_count(text),
        "form_fields_count": _extract_form_fields_count(text),
        "modal_present": _extract_modal_state(text),
        "scroll_x": int(fs.get("scroll_x", 0) or 0),
        "scroll_y": int(fs.get("scroll_y", 0) or 0),
        "scroll_height": int(fs.get("scroll_height", 0) or 0),
        "client_height": int(fs.get("client_height", 0) or 0),
        "dom_complexity": text.count("\n") + 1,
        "text_length": len(text),
        "active_element_tag": _extract_focused_tag(text),
        "form_field_values": fs.get("fields", []),
    }
    return state


def _form_fields_changed(before_fields: List[Dict[str, Any]], after_fields: List[Dict[str, Any]]) -> bool:
    """Compare form field snapshots by matching (tag, type, name, idx) keys."""
    if not before_fields and not after_fields:
        return False

    def _key(f: Dict[str, Any]) -> Tuple[str, str, str, str, int]:
        # For radio/checkbox in same-name groups, each radio is typically the
        # sole child of its wrapper (idx=0 for all), so include value to keep
        # group members individually addressable. See
        # docs/analysis/cross_sites/swatch_form_change_audit.md.
        ftype = str(f.get("type", ""))
        discriminator = str(f.get("value", "")) if ftype in ("radio", "checkbox") else ""
        return (
            str(f.get("tag", "")),
            ftype,
            str(f.get("name", "")),
            discriminator,
            int(f.get("idx", 0)),
        )

    before_map = {_key(f): f for f in before_fields}
    after_map = {_key(f): f for f in after_fields}

    # Field count change (new/removed fields)
    if set(before_map.keys()) != set(after_map.keys()):
        return True

    # Value/checked/selectedIndex change on matched fields
    for k, bf in before_map.items():
        af = after_map.get(k)
        if af is None:
            return True
        if bf.get("value") != af.get("value"):
            return True
        if bf.get("checked") != af.get("checked"):
            return True
        if bf.get("selectedIndex") != af.get("selectedIndex"):
            return True

    return False


# B-09 fix: split page_changed into two derivations:
#   - runner_page_changed = bool(any reason)  → for cycle/retry decision (retains
#     historical behavior; needed because form_value_changed / dom_complexity
#     correctly indicate "framework should not early-stop")
#   - agent_visible_changed = bool(any AGENT_VISIBLE_REASONS reason) → for SR
#     derivation, fig0a metrics, search-loop detection (excludes form_value /
#     dom_complexity / text_length / interactive_elements / form_fields which
#     fire even when agent cannot perceive the change in obs_text)
#
# Probe self-verify (probe_b01_b13_self_verify.py 2026-04-30) found 6/8 I2
# violations are page_changed=True with no agent-visible delta — those should
# be agent_visible_changed=False.
AGENT_VISIBLE_REASONS = frozenset({
    "url_changed",
    "title_changed",
    "content_changed",
    "scroll_changed",
    "modal_state_changed",
})

RUNNER_INTERNAL_REASONS = frozenset({
    "interactive_elements_changed",
    "form_fields_changed",
    "dom_complexity_changed",
    "text_length_changed",
    "form_value_changed",
})


def is_agent_visible_change(reasons: List[str]) -> bool:
    """Return True if any reason in `reasons` is agent-perceivable.

    Used to derive `agent_visible_changed` step record field for paper-grade
    SR computation that excludes runner-internal noise (form_value_changed
    et al firing on form edits that don't change obs_text).
    """
    return bool(set(reasons or []) & AGENT_VISIBLE_REASONS)


def detect_page_state_change(
    state_before: Dict[str, Any],
    state_after: Dict[str, Any],
    action_type: str,
    similarity_threshold: float = 0.95,
) -> Tuple[bool, List[str], float]:
    """
    Detect whether action changed page state.

    Returns:
      (action_successful, change_reasons, text_similarity)
    """
    before = state_before or {}
    after = state_after or {}

    changes: List[str] = []

    text_before = _safe_str(before.get("visible_text"))
    text_after = _safe_str(after.get("visible_text"))

    if text_before and text_after:
        similarity = SequenceMatcher(None, text_before, text_after).ratio()
        if similarity < similarity_threshold:
            changes.append("content_changed")
    elif not text_before and not text_after:
        similarity = 1.0  # both empty — genuinely unchanged (e.g. blank page)
    else:
        # one side empty, the other not — treat as content changed
        similarity = 0.0
        changes.append("content_changed")

    if int(before.get("interactive_elements_count", 0) or 0) != int(after.get("interactive_elements_count", 0) or 0):
        changes.append("interactive_elements_changed")

    if int(before.get("form_fields_count", 0) or 0) != int(after.get("form_fields_count", 0) or 0):
        changes.append("form_fields_changed")

    if _safe_str(before.get("title")) != _safe_str(after.get("title")):
        changes.append("title_changed")

    if _safe_str(before.get("url")) != _safe_str(after.get("url")):
        changes.append("url_changed")

    if bool(before.get("modal_present", False)) != bool(after.get("modal_present", False)):
        changes.append("modal_state_changed")

    sbx = int(before.get("scroll_x", 0) or 0)
    sby = int(before.get("scroll_y", 0) or 0)
    sax = int(after.get("scroll_x", 0) or 0)
    say = int(after.get("scroll_y", 0) or 0)
    if abs(sax - sbx) >= 5 or abs(say - sby) >= 5:
        changes.append("scroll_changed")

    # DOM complexity change (>20% relative change)
    dc_before = int(before.get("dom_complexity", 0) or 0)
    dc_after = int(after.get("dom_complexity", 0) or 0)
    if dc_before > 0 and abs(dc_after - dc_before) / dc_before > 0.20:
        changes.append("dom_complexity_changed")

    # Text length change (>30% relative change)
    tl_before = int(before.get("text_length", 0) or 0)
    tl_after = int(after.get("text_length", 0) or 0)
    if tl_before > 0 and abs(tl_after - tl_before) / tl_before > 0.30:
        changes.append("text_length_changed")

    # Form field value change (catches type edits, select_option, checkbox toggle)
    if _form_fields_changed(
        before.get("form_field_values", []),
        after.get("form_field_values", []),
    ):
        changes.append("form_value_changed")

    action_upper = (action_type or "").upper()
    if action_upper in ("SCROLL", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM"):
        # Use real scroll_y delta instead of unconditional True
        if abs(say - sby) >= 5:
            if "scroll_changed" not in changes:
                changes.append("scroll_changed")
            return True, changes, similarity
        else:
            # Scroll didn't actually move — only report success if other evidence exists
            evidence = [c for c in changes if c != "focus_blur"]
            return len(evidence) > 0, changes, similarity

    evidence = [c for c in changes if c != "focus_blur"]
    return len(evidence) > 0, changes, similarity
