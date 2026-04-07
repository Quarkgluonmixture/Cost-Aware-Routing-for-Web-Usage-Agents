from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from p79.envs.vwa_wrapper import P79Observation


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


def build_page_state(obs: P79Observation, info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    text = _safe_str(getattr(obs, "text", ""))
    info = info or {}

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
        "visible_text": text[:2000],
        "interactive_elements_count": _extract_interactive_count(text),
        "form_fields_count": _extract_form_fields_count(text),
        "modal_present": _extract_modal_state(text),
        "scroll_x": int(info.get("scroll_x", 0) or 0),
        "scroll_y": int(info.get("scroll_y", 0) or 0),
        "dom_complexity": text.count("\n") + 1,
        "text_length": len(text),
        "active_element_tag": _extract_focused_tag(text),
    }
    return state


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
    else:
        similarity = 1.0

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

    action_upper = (action_type or "").upper()
    if action_upper in ("SCROLL", "SCROLL UP", "SCROLL DOWN", "SCROLL TOP", "SCROLL BOTTOM"):
        return True, changes, similarity

    evidence = [c for c in changes if c != "focus_blur"]
    return len(evidence) > 0, changes, similarity
