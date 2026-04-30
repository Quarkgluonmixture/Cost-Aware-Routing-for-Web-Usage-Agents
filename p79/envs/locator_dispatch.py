"""Locator-route dispatch wrapper (Cluster 1 patch — B-01/02/03/04/05/25/32/33).

Replaces framework's `mouse.click(union_bound_center)` pattern (actions.py:1280-
1430) with Playwright locator-based dispatch on the resolved actionable DOM
ancestor. Avoids the B-33 family bug where bbox center hits a child span/icon
instead of the actionable parent (`<a>`, `<button>`, `<input>`).

Tier 10 sweep (probe_tier10_dispatch_target.py 2026-04-30) measured 94.4%
off-target on failed clicks; this module aims to lift that to >80% ON_TARGET.

Architecture (per docs/analysis/cross_sites/cluster1_locator_route_design.md):
- Looks up obs_nodes_info[eid]["union_bound"] for pixel center
- Uses page.evaluate to find DOM element at center
- Walks up to find actionable ancestor (max 6 levels)
- Dispatches via Playwright locator (real mouse event + actionability check)
- Returns dict with success/target_tag/fallback_used for runner observability

Falls back to framework dispatch (existing `mouse.click(center)`) if walk-up
fails — safe, preserves current behavior for unrecognized targets.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# JS resolvers — find actionable ancestor by walking up the DOM
# Returns the resolved ElementHandle (via evaluate_handle) or null

_JS_RESOLVE_CLICK = """([cx, cy]) => {
    let el = document.elementFromPoint(cx, cy);
    if (!el) return null;
    for (let i = 0; i < 6 && el && el !== document.body; i++) {
        if (el.tagName === 'A' && el.href) return el;
        if (el.tagName === 'BUTTON') return el;
        const role = el.getAttribute('role');
        if (role === 'link' || role === 'button' || role === 'menuitem' ||
            role === 'tab' || role === 'option' || role === 'menuitemcheckbox') return el;
        if (el.tagName === 'INPUT' && (el.type === 'submit' || el.type === 'button' ||
            el.type === 'checkbox' || el.type === 'radio')) return el;
        if (el.tagName === 'SUMMARY') return el;  // <details>/<summary>
        el = el.parentElement;
    }
    return null;
}"""

_JS_RESOLVE_INPUT = """([cx, cy]) => {
    let el = document.elementFromPoint(cx, cy);
    if (!el) return null;
    for (let i = 0; i < 6 && el && el !== document.body; i++) {
        if (el.tagName === 'INPUT' && el.type !== 'hidden' &&
            el.type !== 'submit' && el.type !== 'button' &&
            el.type !== 'checkbox' && el.type !== 'radio') return el;
        if (el.tagName === 'TEXTAREA') return el;
        if (el.isContentEditable) return el;
        // Label associated with input via for=""
        if (el.tagName === 'LABEL' && el.htmlFor) {
            const target = document.getElementById(el.htmlFor);
            if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {
                if (target.type !== 'hidden' && target.type !== 'submit' && target.type !== 'button') {
                    return target;
                }
            }
        }
        el = el.parentElement;
    }
    return null;
}"""

_JS_RESOLVE_UPLOAD = """([cx, cy]) => {
    let el = document.elementFromPoint(cx, cy);
    if (!el) return null;
    for (let i = 0; i < 6 && el && el !== document.body; i++) {
        if (el.tagName === 'INPUT' && el.type === 'file') return el;
        // "Choose File" button often near a hidden file input
        const parent = el.parentElement;
        if (parent) {
            const fileInput = parent.querySelector('input[type=file]');
            if (fileInput) return fileInput;
        }
        el = el.parentElement;
    }
    return null;
}"""


def _bbox_center(union_bound: Optional[list]) -> Optional[tuple]:
    """Compute (x_px, y_px) from union_bound = [x, y, w, h]."""
    if not union_bound or len(union_bound) < 4:
        return None
    x, y, w, h = union_bound[0], union_bound[1], union_bound[2], union_bound[3]
    return (float(x) + float(w) / 2.0, float(y) + float(h) / 2.0)


def dispatch_id_based_click(
    page: Any,
    obs_nodes_info: Optional[Dict[str, Any]],
    element_id: int,
    *,
    sleep_after_ms: int = 0,
) -> Dict[str, Any]:
    """Locator-route CLICK for id-based action.

    Returns: {success: bool, fallback_used: bool, target_tag: Optional[str],
              error: Optional[str]}
    """
    node_info = (obs_nodes_info or {}).get(str(element_id))
    if not node_info or "union_bound" not in node_info:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": "obs_nodes_info missing union_bound"}
    center = _bbox_center(node_info["union_bound"])
    if center is None:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": "invalid union_bound shape"}
    cx, cy = center
    try:
        handle = page.evaluate_handle(_JS_RESOLVE_CLICK, [cx, cy])
        # JSHandle resolves to None if no match
        as_element = handle.as_element() if handle is not None else None
        if as_element is None:
            try:
                handle.dispose()
            except Exception:
                pass
            return {"success": False, "fallback_used": True, "target_tag": None,
                    "error": "no actionable ancestor within walk-up depth"}
        target_tag = as_element.evaluate("el => el.tagName")
        as_element.click(timeout=5000)
        if sleep_after_ms > 0:
            page.wait_for_timeout(int(sleep_after_ms))
        try:
            as_element.dispose()
        except Exception:
            pass
        return {"success": True, "fallback_used": False, "target_tag": str(target_tag), "error": None}
    except Exception as e:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": f"{type(e).__name__}: {str(e)[:160]}"}


def dispatch_id_based_type(
    page: Any,
    obs_nodes_info: Optional[Dict[str, Any]],
    element_id: int,
    text: str,
    *,
    sleep_after_ms: int = 0,
    press_enter: bool = False,
) -> Dict[str, Any]:
    """Locator-route TYPE for id-based action — uses locator.fill() which auto-
    clears + dispatches input event WITHOUT global Meta+A (avoiding 全选变蓝
    §52/§64). press_enter optionally simulates Enter key after fill.
    """
    node_info = (obs_nodes_info or {}).get(str(element_id))
    if not node_info or "union_bound" not in node_info:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": "obs_nodes_info missing union_bound"}
    center = _bbox_center(node_info["union_bound"])
    if center is None:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": "invalid union_bound shape"}
    cx, cy = center
    # Strip trailing newline from text — use press_enter flag instead.
    fill_text = text
    if fill_text.endswith("\n"):
        fill_text = fill_text[:-1]
        press_enter = True
    try:
        handle = page.evaluate_handle(_JS_RESOLVE_INPUT, [cx, cy])
        as_element = handle.as_element() if handle is not None else None
        if as_element is None:
            try:
                handle.dispose()
            except Exception:
                pass
            return {"success": False, "fallback_used": True, "target_tag": None,
                    "error": "no input ancestor within walk-up depth"}
        target_tag = as_element.evaluate("el => el.tagName")
        as_element.fill(fill_text, timeout=5000)
        if press_enter:
            as_element.press("Enter", timeout=5000)
        if sleep_after_ms > 0:
            page.wait_for_timeout(int(sleep_after_ms))
        try:
            as_element.dispose()
        except Exception:
            pass
        return {"success": True, "fallback_used": False, "target_tag": str(target_tag), "error": None}
    except Exception as e:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": f"{type(e).__name__}: {str(e)[:160]}"}


def dispatch_id_based_hover(
    page: Any,
    obs_nodes_info: Optional[Dict[str, Any]],
    element_id: int,
    *,
    sleep_after_ms: int = 0,
) -> Dict[str, Any]:
    """Locator-route HOVER for id-based action."""
    node_info = (obs_nodes_info or {}).get(str(element_id))
    if not node_info or "union_bound" not in node_info:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": "obs_nodes_info missing union_bound"}
    center = _bbox_center(node_info["union_bound"])
    if center is None:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": "invalid union_bound shape"}
    cx, cy = center
    try:
        handle = page.evaluate_handle(_JS_RESOLVE_CLICK, [cx, cy])
        as_element = handle.as_element() if handle is not None else None
        if as_element is None:
            try:
                handle.dispose()
            except Exception:
                pass
            return {"success": False, "fallback_used": True, "target_tag": None,
                    "error": "no hover target within walk-up depth"}
        target_tag = as_element.evaluate("el => el.tagName")
        as_element.hover(timeout=5000)
        if sleep_after_ms > 0:
            page.wait_for_timeout(int(sleep_after_ms))
        try:
            as_element.dispose()
        except Exception:
            pass
        return {"success": True, "fallback_used": False, "target_tag": str(target_tag), "error": None}
    except Exception as e:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": f"{type(e).__name__}: {str(e)[:160]}"}


def dispatch_id_based_clear(
    page: Any,
    obs_nodes_info: Optional[Dict[str, Any]],
    element_id: int,
    *,
    sleep_after_ms: int = 0,
) -> Dict[str, Any]:
    """Locator-route CLEAR — fill input with empty string."""
    return dispatch_id_based_type(page, obs_nodes_info, element_id, "",
                                   sleep_after_ms=sleep_after_ms, press_enter=False)


def dispatch_id_based_upload(
    page: Any,
    obs_nodes_info: Optional[Dict[str, Any]],
    element_id: int,
    file_path: str,
    *,
    sleep_after_ms: int = 0,
) -> Dict[str, Any]:
    """Locator-route UPLOAD — find file input ancestor + set_input_files()."""
    node_info = (obs_nodes_info or {}).get(str(element_id))
    if not node_info or "union_bound" not in node_info:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": "obs_nodes_info missing union_bound"}
    center = _bbox_center(node_info["union_bound"])
    if center is None:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": "invalid union_bound shape"}
    cx, cy = center
    try:
        handle = page.evaluate_handle(_JS_RESOLVE_UPLOAD, [cx, cy])
        as_element = handle.as_element() if handle is not None else None
        if as_element is None:
            try:
                handle.dispose()
            except Exception:
                pass
            return {"success": False, "fallback_used": True, "target_tag": None,
                    "error": "no file input within walk-up depth"}
        target_tag = as_element.evaluate("el => el.tagName")
        as_element.set_input_files(file_path, timeout=5000)
        if sleep_after_ms > 0:
            page.wait_for_timeout(int(sleep_after_ms))
        try:
            as_element.dispose()
        except Exception:
            pass
        return {"success": True, "fallback_used": False, "target_tag": str(target_tag), "error": None}
    except Exception as e:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": f"{type(e).__name__}: {str(e)[:160]}"}
