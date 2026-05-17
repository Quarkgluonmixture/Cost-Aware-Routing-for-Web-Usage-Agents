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

/stress A1.3 (2026-05-15):
- F1 fix: handle + as_element disposed in `finally` regardless of return / raise
  path. Previously the outer `except` branch silently leaked both. Playwright
  JSHandle is a V8 reference; episode-level page-reset gates the blast radius
  but the leak is paper-grade hygiene gap.
- F6 fix: standardize the walk-fail `error` field to a `walk_fail:<category>`
  format prefix so downstream telemetry aggregation can group by failure type
  rather than fuzzy-matching free-text strings. The category currently
  collapses to `no_actionable_within_walk` (no subcategorization); future
  calibration runs can extend the diagnostic JS to distinguish e.g.
  `no_href_anchor` / `walk_exhausted` / `aria_unknown` / `no_element_at_point`.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# /stress A1.3 F2 backlog sweep (2026-05-15): expose walk-up depth as a named
# constant so any future calibration / tuning is a one-line change instead of
# a 3-place magic-number edit.
WALK_UP_MAX_DEPTH = 6

# /stress A1.3 F4 backlog sweep (2026-05-15): ARIA actionable role accept list
# is now a JS constant. ARIA 1.2 actionable roles that map to a single click
# intent — added `menuitemradio`, `switch`, `treeitem`, `gridcell`, `radio`,
# `checkbox` (non-native), `combobox` (some popup dropdowns), `slider` (drag
# target — accepting because click still meaningful as focus).
_ACTIONABLE_ARIA_ROLES_JS = (
    "['link','button','menuitem','tab','option','menuitemcheckbox',"
    "'menuitemradio','switch','treeitem','gridcell','radio','checkbox',"
    "'combobox','slider']"
)

# B-161 (/stress A1.3 v8 gemini C4, 2026-05-16): shadow-DOM penetration helper.
# ``document.elementFromPoint`` returns the *shadow host* (e.g. ``<custom-search>``)
# instead of the inner ``<button>``/``<a>`` when the page uses Shadow DOM (Reddit
# redesign, modern SPAs, web components). Pre-fix the walk-up loop hit the host
# without an actionable ancestor, fell through to walk_fail → framework
# bbox-center fallback (= the B-33 buggy path Cluster 1 was meant to retire).
# Recursive descent: if the hit element has a ``shadowRoot`` and its inner
# ``elementFromPoint`` returns a *different* element, follow into that. Depth
# capped at 5 (deeply nested shadow tree is exotic — most are 1-2 levels).
_JS_SHADOW_DESCENT_FN = """
function _shadowDescend(root, cx, cy, depth) {
    if (!root || depth >= 5) return null;
    const inner = (root.elementFromPoint ? root.elementFromPoint(cx, cy) : null);
    if (!inner) return null;
    // If this inner element has its own shadow, recurse one level deeper.
    if (inner.shadowRoot) {
        const deeper = _shadowDescend(inner.shadowRoot, cx, cy, depth + 1);
        if (deeper && deeper !== inner) return deeper;
    }
    return inner;
}
function _pierceElementFromPoint(cx, cy) {
    let el = document.elementFromPoint(cx, cy);
    if (!el) return null;
    // If the top-level hit is a shadow host, pierce into it.
    if (el.shadowRoot) {
        const pierced = _shadowDescend(el.shadowRoot, cx, cy, 1);
        if (pierced && pierced !== el) return pierced;
    }
    return el;
}
"""


# JS resolvers — find actionable ancestor by walking up the DOM
# Returns the resolved ElementHandle (via evaluate_handle) or null
#
# /stress A1.3 F3 backlog sweep: `<a>` without `href` is accepted iff it has
# an `onclick` attribute (or non-empty `data-*` markers used by some sites
# for JS-only links). Previously these would fall through to walk-fail →
# framework bbox-center fallback (= B-33 regression risk).

_JS_RESOLVE_CLICK = f"""([cx, cy]) => {{
    {_JS_SHADOW_DESCENT_FN}
    let el = _pierceElementFromPoint(cx, cy);
    if (!el) return null;
    const ACTIONABLE_ROLES = {_ACTIONABLE_ARIA_ROLES_JS};
    for (let i = 0; i < {WALK_UP_MAX_DEPTH} && el && el !== document.body; i++) {{
        if (el.tagName === 'A' && el.href) return el;
        // F3: <a> without href but with onclick handler / JS-link role
        if (el.tagName === 'A' && (el.onclick || el.getAttribute('onclick'))) return el;
        if (el.tagName === 'BUTTON') return el;
        const role = el.getAttribute('role');
        if (role && ACTIONABLE_ROLES.indexOf(role) !== -1) return el;
        if (el.tagName === 'INPUT' && (el.type === 'submit' || el.type === 'button' ||
            el.type === 'checkbox' || el.type === 'radio' ||
            el.type === 'image' || el.type === 'reset')) return el;
        // B-443 (/stress A1.25 P1-6-A Claude, 2026-05-17): added INPUT
        // type=image (image submit button, common in Magento/shopping sprites)
        // and type=reset (form reset). Pre-fix both fell through to walk-fail
        // → framework bbox-center fallback = silent B-33 regression on those
        // specific clicks (Shopping "Add to Cart" image-submit pattern).
        if (el.tagName === 'SUMMARY') return el;  // <details>/<summary>
        if (el.tagName === 'AREA' && el.href) return el;  // <map>/<area> image map
        if (el.isContentEditable) return el;  // [contenteditable] divs
        // B-161: traverse out of shadow roots when walking up (parentElement
        // stops at shadow-root boundary; need ``getRootNode().host`` to escape).
        const next = el.parentElement || (el.getRootNode && el.getRootNode().host) || null;
        el = next;
    }}
    return null;
}}"""

_JS_RESOLVE_INPUT = f"""([cx, cy]) => {{
    {_JS_SHADOW_DESCENT_FN}
    let el = _pierceElementFromPoint(cx, cy);
    if (!el) return null;
    for (let i = 0; i < {WALK_UP_MAX_DEPTH} && el && el !== document.body; i++) {{
        if (el.tagName === 'INPUT' && el.type !== 'hidden' &&
            el.type !== 'submit' && el.type !== 'button' &&
            el.type !== 'checkbox' && el.type !== 'radio') return el;
        if (el.tagName === 'TEXTAREA') return el;
        if (el.isContentEditable) return el;
        // Label associated with input via for=""
        if (el.tagName === 'LABEL' && el.htmlFor) {{
            const target = document.getElementById(el.htmlFor);
            if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) {{
                if (target.type !== 'hidden' && target.type !== 'submit' && target.type !== 'button') {{
                    return target;
                }}
            }}
        }}
        const next = el.parentElement || (el.getRootNode && el.getRootNode().host) || null;
        el = next;
    }}
    return null;
}}"""

_JS_RESOLVE_UPLOAD = f"""([cx, cy]) => {{
    {_JS_SHADOW_DESCENT_FN}
    let el = _pierceElementFromPoint(cx, cy);
    if (!el) return null;
    for (let i = 0; i < {WALK_UP_MAX_DEPTH} && el && el !== document.body; i++) {{
        if (el.tagName === 'INPUT' && el.type === 'file') return el;
        // "Choose File" button often near a hidden file input
        const parent = el.parentElement;
        if (parent) {{
            const fileInput = parent.querySelector('input[type=file]');
            if (fileInput) return fileInput;
        }}
        const next = el.parentElement || (el.getRootNode && el.getRootNode().host) || null;
        el = next;
    }}
    return null;
}}"""


def _bbox_center(union_bound: Optional[list]) -> Optional[tuple]:
    """Compute (x_px, y_px) from union_bound = [x, y, w, h]."""
    if not union_bound or len(union_bound) < 4:
        return None
    x, y, w, h = union_bound[0], union_bound[1], union_bound[2], union_bound[3]
    return (float(x) + float(w) / 2.0, float(y) + float(h) / 2.0)


def _dispose_all(*handles) -> None:
    """Dispose any Playwright JSHandles, suppressing exceptions individually."""
    for h in handles:
        if h is not None:
            try:
                h.dispose()
            except Exception:
                pass


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
    handle = None
    as_element = None
    try:
        handle = page.evaluate_handle(_JS_RESOLVE_CLICK, [cx, cy])
        as_element = handle.as_element() if handle is not None else None
        if as_element is None:
            return {"success": False, "fallback_used": True, "target_tag": None,
                    "error": "walk_fail:no_actionable_within_walk"}
        target_tag = as_element.evaluate("el => el.tagName")
        as_element.click(timeout=5000)
        if sleep_after_ms > 0:
            page.wait_for_timeout(int(sleep_after_ms))
        return {"success": True, "fallback_used": False, "target_tag": str(target_tag), "error": None}
    except Exception as e:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": f"{type(e).__name__}: {str(e)[:160]}"}
    finally:
        _dispose_all(as_element, handle)


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
    handle = None
    as_element = None
    try:
        handle = page.evaluate_handle(_JS_RESOLVE_INPUT, [cx, cy])
        as_element = handle.as_element() if handle is not None else None
        if as_element is None:
            return {"success": False, "fallback_used": True, "target_tag": None,
                    "error": "walk_fail:no_input_within_walk"}
        target_tag = as_element.evaluate("el => el.tagName")
        as_element.fill(fill_text, timeout=5000)
        if press_enter:
            as_element.press("Enter", timeout=5000)
        if sleep_after_ms > 0:
            page.wait_for_timeout(int(sleep_after_ms))
        return {"success": True, "fallback_used": False, "target_tag": str(target_tag), "error": None}
    except Exception as e:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": f"{type(e).__name__}: {str(e)[:160]}"}
    finally:
        _dispose_all(as_element, handle)


# B-439 (/stress A1.25 P0-4-AC* OOB, 2026-05-17): hover/clear/upload locator-route
# dispatch functions DELETED as dead code. Grep verified zero production callsites
# in `p79/envs/vwa_wrapper.py` or anywhere in `p79/` — only `tests/test_locator_
# dispatch.py` exercised them. Production hover/clear/upload action_types (zero
# observed in `results/visualwebarena/phase1/*/step_*.jsonl`) fall through to VWA
# framework dispatch via `_json_to_id_action_str` → `create_id_based_action`.
# Workshop sub-paper "VWA upstream bug fix" framing now honestly scoped to
# click+type only. Paper §3 disclosure updated separately (P0-5).


def dispatch_coord_based_type(
    page: Any,
    cx: float,
    cy: float,
    text: str,
    *,
    sleep_after_ms: int = 0,
    press_enter: bool = False,
) -> Dict[str, Any]:
    """Locator-route TYPE for COORDINATE-based action (vision-mode focus-click).

    B-442 (/stress A1.25 P0-3-AC* OOB, 2026-05-17): closes the cross-mode
    asymmetry where DOM/SoM mode TYPE got locator walk-up (B-01 fix) but
    vision-mode TYPE used direct ``page.mouse.click(px, py)`` (vwa_wrapper.py
    pre-fix line 394-397) — still B-01-prone bbox-pattern. The Control+a
    ``is_editable`` guard at vwa_wrapper.py:405-413 only prevented the visible
    全选变蓝 symptom; the focus-落空 root cause persisted in vision mode.

    Reuses ``_JS_RESOLVE_INPUT`` (already accounts for shadow-DOM pierce per
    B-161 and 6-level walk-up depth) — vision-mode TYPE now gets the same
    walk-up coverage as id-based TYPE. ``cx``/``cy`` are pixel coordinates
    (caller is responsible for coord normalization → pixel conversion).
    """
    handle = None
    as_element = None
    fill_text = text
    if fill_text.endswith("\n"):
        fill_text = fill_text[:-1]
        press_enter = True
    try:
        handle = page.evaluate_handle(_JS_RESOLVE_INPUT, [cx, cy])
        as_element = handle.as_element() if handle is not None else None
        if as_element is None:
            return {"success": False, "fallback_used": True, "target_tag": None,
                    "error": "walk_fail:no_input_within_walk"}
        target_tag = as_element.evaluate("el => el.tagName")
        as_element.fill(fill_text, timeout=5000)
        if press_enter:
            as_element.press("Enter", timeout=5000)
        if sleep_after_ms > 0:
            page.wait_for_timeout(int(sleep_after_ms))
        return {"success": True, "fallback_used": False, "target_tag": str(target_tag), "error": None}
    except Exception as e:
        return {"success": False, "fallback_used": True, "target_tag": None,
                "error": f"{type(e).__name__}: {str(e)[:160]}"}
    finally:
        _dispose_all(as_element, handle)
