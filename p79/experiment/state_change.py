from __future__ import annotations

import hashlib
import re
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

from p79.envs.vwa_wrapper import P79Observation
from p79.experiment.som import MARK_ID_DETECT_RE  # /stress A1.10 P1-2-AB* canonical regex

# /stress A1.10 P1-1-A (2026-05-16): raised from 5000 to 20000.
# Empirical text_length distribution (5001-step B1 cls sample): p95=4675,
# max=46591 — pre-fix ~5% of pages exceeded the 5000-char prefix used by
# SequenceMatcher similarity, masking real content_change on long pages
# (cls search-results pages with 30+ listings). 20000 covers the empirical
# p99 (~8000) with 2× safety margin. Long pages > 20000 chars fall back to
# content-hash equality (cheap O(n) md5) instead of O(n²) SequenceMatcher.
_TEXT_TRUNCATION_LIMIT = 20000


# /stress A1.10 P1-5-A (2026-05-16): tightened modal-state detection.
# Pre-fix: `any(k in low for k in ("dialog","modal","popup",...))` matched
# any of these substrings anywhere in AXTree text — reddit subforum
# descriptions containing the word "dialog" caused modal_present to flip
# noisily, polluting modal_state_changed (an AGENT_VISIBLE_REASON).
# Post-fix: require the strings to appear inside role/aria-modal attribute
# context, which is the canonical accessibility-tree dialog signal.
_MODAL_STATE_RE = re.compile(
    r"\b(?:role|aria-modal)\s*[=:]\s*[\"']?(?:dialog|alertdialog|modal)\b",
    re.IGNORECASE,
)


# Adapted from external_code/page_state_utils.py (Aiden Yiliu Li, Apache-2.0)
def _safe_str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _extract_interactive_count(text: str) -> int:
    if not text:
        return 0
    # /stress A1.10 P1-2-AB* (2026-05-16): use canonical anchored mark regex
    # from som.py. Pre-fix `re.findall(r"\[(\d+)\]", text)` counted any
    # bracketed digit anywhere in the AXTree dump including footnote
    # references inside StaticText labels — Mode A F4 + Mode B F7 dual-catch
    # of A1.4 SOM regex sibling propagation defect.
    return sum(1 for line in text.splitlines() if MARK_ID_DETECT_RE.match(line))


def _extract_form_fields_count(text: str) -> int:
    if not text:
        return 0
    markers = ("textbox", "input", "textarea", "select", "combobox", "search")
    low = text.lower()
    return sum(low.count(m) for m in markers)


def _extract_modal_state(text: str) -> bool:
    if not text:
        return False
    return bool(_MODAL_STATE_RE.search(text))


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
        "form_field_values": fs.get("fields", []),
    }
    return state


def _form_fields_changed(before_fields: List[Dict[str, Any]], after_fields: List[Dict[str, Any]]) -> bool:
    """Compare form field snapshots by matching (tag, type, name, idx) keys."""
    if not before_fields and not after_fields:
        return False

    def _key(f: Dict[str, Any]) -> Tuple[str, str, str, str, int]:
        # /stress A1.10 P1-4-A* (2026-05-16): discriminator now includes value
        # for ALL field types, not just radio/checkbox. Pre-fix: text/textarea
        # with empty `name=""` (cls search filters frequently use unnamed
        # inputs) collapsed to identical `(tag, type, "", "", idx)` keys when
        # wrapper-relative idx collided, causing form_value_changed to miss
        # real edits. Including value as a sub-discriminator for all types
        # restores per-field addressability across the empty-name regime.
        ftype = str(f.get("type", ""))
        return (
            str(f.get("tag", "")),
            ftype,
            str(f.get("name", "")),
            str(f.get("value", "")),
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
        # B-424 (/stress A1.3 v9 Mode B P2-2, 2026-05-17): full-fidelity check
        # — `value` is the 200-char prefix; suffix edits beyond 200 chars
        # would silently match. value_len + value_djb2 hash (when present
        # from the upgraded _FORM_SNAPSHOT_JS) captures the full content.
        # Legacy snapshots without these keys default to None == None →
        # this clause is no-op when running against archived data.
        if bf.get("value_len") != af.get("value_len"):
            return True
        if bf.get("value_djb2") != af.get("value_djb2"):
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
        # /stress A1.10 P1-1-A (2026-05-16): long-page hash-equality fallback.
        # SequenceMatcher.ratio() is O(n²) and gets expensive past ~10k chars.
        # For pages exceeding the truncation limit on either side, fall back
        # to cheap O(n) md5 hash equality. Hash-equal pages report similarity
        # 1.0 (no content_change); hash-different pages report 0.0 (changed).
        # This trades the soft similarity threshold for a hard equality on
        # long pages, which is more conservative (fewer false-positive matches
        # because identical hash requires byte-exact text).
        if len(text_before) >= _TEXT_TRUNCATION_LIMIT or len(text_after) >= _TEXT_TRUNCATION_LIMIT:
            h_before = hashlib.md5(text_before.encode("utf-8", errors="replace")).digest()
            h_after = hashlib.md5(text_after.encode("utf-8", errors="replace")).digest()
            if h_before == h_after:
                similarity = 1.0
            else:
                similarity = 0.0
                changes.append("content_changed")
        else:
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
