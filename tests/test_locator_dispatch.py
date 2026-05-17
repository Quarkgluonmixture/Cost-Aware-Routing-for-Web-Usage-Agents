"""locator_dispatch F1 (dispose always) + F6 (walk_fail error category) invariants.

/stress A1.3 (2026-05-15) fix verification.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# B-439 (/stress A1.25 P0-4-AC* OOB, 2026-05-17): hover/clear/upload imports
# removed alongside function deletion — production code never invoked these,
# tests were the only callsite. Workshop sub-paper scope honestly = click+type.
from p79.envs.locator_dispatch import (
    dispatch_id_based_click,
    dispatch_id_based_type,
)


def _make_page(handle_returns_element: bool, click_raises: bool = False):
    """Build a mock Playwright page that returns a JSHandle on evaluate_handle.

    Tracks dispose calls on both the wrapper handle and the as_element so
    tests can verify F1 dispose-always semantics.
    """
    page = MagicMock(name="page")
    handle = MagicMock(name="handle")
    if handle_returns_element:
        elem = MagicMock(name="elem")
        elem.evaluate.return_value = "BUTTON"
        if click_raises:
            elem.click.side_effect = RuntimeError("element not visible")
            elem.fill.side_effect = RuntimeError("element not visible")
            elem.hover.side_effect = RuntimeError("element not visible")
            elem.set_input_files.side_effect = RuntimeError("element not visible")
        handle.as_element.return_value = elem
    else:
        elem = None
        handle.as_element.return_value = None
    page.evaluate_handle.return_value = handle
    return page, handle, elem


_NODES = {"42": {"union_bound": [10, 20, 100, 50]}}  # x, y, w, h


# ---------------------------------------------------------------------------
# F6 — walk_fail error format
# ---------------------------------------------------------------------------


def test_click_walk_fail_returns_walk_fail_prefixed_error():
    page, handle, _ = _make_page(handle_returns_element=False)
    r = dispatch_id_based_click(page, _NODES, 42)
    assert r["success"] is False
    assert r["error"].startswith("walk_fail:"), f"got {r['error']!r}"
    assert r["error"] == "walk_fail:no_actionable_within_walk"


def test_type_walk_fail_distinct_category():
    page, handle, _ = _make_page(handle_returns_element=False)
    r = dispatch_id_based_type(page, _NODES, 42, "search\n")
    assert r["error"] == "walk_fail:no_input_within_walk"


# B-439: hover/upload walk-fail tests retired (functions deleted as dead code)


# ---------------------------------------------------------------------------
# F1 — dispose-always invariant
# ---------------------------------------------------------------------------


def test_click_disposes_both_handles_on_walk_fail():
    page, handle, _ = _make_page(handle_returns_element=False)
    dispatch_id_based_click(page, _NODES, 42)
    handle.dispose.assert_called()


def test_click_disposes_both_handles_on_action_raise():
    """The F1 fix-critical case: click() raises → dispose must still fire.

    Prior code: handle + as_element leaked because the outer except returned
    before the inner dispose blocks ran. New code's `finally` runs both.
    """
    page, handle, elem = _make_page(handle_returns_element=True, click_raises=True)
    r = dispatch_id_based_click(page, _NODES, 42)
    assert r["success"] is False
    assert r["error"].startswith("RuntimeError:")
    handle.dispose.assert_called()
    elem.dispose.assert_called()


def test_click_disposes_on_success_path():
    page, handle, elem = _make_page(handle_returns_element=True)
    r = dispatch_id_based_click(page, _NODES, 42)
    assert r["success"] is True
    handle.dispose.assert_called()
    elem.dispose.assert_called()


def test_type_disposes_on_fill_raise():
    page, handle, elem = _make_page(handle_returns_element=True, click_raises=True)
    dispatch_id_based_type(page, _NODES, 42, "x")
    handle.dispose.assert_called()
    elem.dispose.assert_called()


# B-439: hover/upload/clear dispose tests retired (functions deleted as dead code)


# ---------------------------------------------------------------------------
# No-regression: input validation paths still return same shape
# ---------------------------------------------------------------------------


def test_missing_node_info_unchanged():
    page = MagicMock()
    r = dispatch_id_based_click(page, None, 42)
    assert r["success"] is False
    assert r["fallback_used"] is True
    assert r["error"] == "obs_nodes_info missing union_bound"
    page.evaluate_handle.assert_not_called()


def test_invalid_union_bound_unchanged():
    page = MagicMock()
    r = dispatch_id_based_click(page, {"42": {"union_bound": []}}, 42)
    assert r["error"] == "invalid union_bound shape"


# ---------------------------------------------------------------------------
# A1.3 backlog sweep — F2/F3/F4 invariants
# ---------------------------------------------------------------------------


def test_walk_up_max_depth_is_named_constant():
    """F2: walk-up depth is exposed as named constant, not buried magic 6."""
    from p79.envs.locator_dispatch import WALK_UP_MAX_DEPTH, _JS_RESOLVE_CLICK

    assert isinstance(WALK_UP_MAX_DEPTH, int)
    assert WALK_UP_MAX_DEPTH == 6  # current value; change requires audit
    # B-439: _JS_RESOLVE_UPLOAD kept in module (still imported below for shadow-DOM
    # test coverage) but the dispatch_id_based_upload function deleted. The JS
    # constant remains in case future workshop-sub-paper expansion wires upload.
    from p79.envs.locator_dispatch import _JS_RESOLVE_INPUT, _JS_RESOLVE_UPLOAD
    for js_name, js in [("CLICK", _JS_RESOLVE_CLICK), ("INPUT", _JS_RESOLVE_INPUT), ("UPLOAD", _JS_RESOLVE_UPLOAD)]:
        assert f"i < {WALK_UP_MAX_DEPTH}" in js, f"{js_name} resolver missing depth constant"


def test_click_resolver_accepts_anchor_without_href_with_onclick():
    """F3: <a> without href but with onclick handler is treated as actionable."""
    from p79.envs.locator_dispatch import _JS_RESOLVE_CLICK

    # Source check — the resolver must inspect el.onclick / onclick attribute
    assert "el.onclick" in _JS_RESOLVE_CLICK
    assert "getAttribute('onclick')" in _JS_RESOLVE_CLICK


def test_click_resolver_accepts_extended_aria_roles():
    """F4: ARIA accept list extended beyond original 6 roles."""
    from p79.envs.locator_dispatch import _JS_RESOLVE_CLICK

    expected_added_roles = {
        "menuitemradio",  # ARIA 1.1
        "switch",          # ARIA 1.1
        "treeitem",        # ARIA 1.0
        "gridcell",        # ARIA 1.0 interactive
        "radio",           # ARIA-only (not native input)
        "checkbox",        # ARIA-only (not native input)
        "combobox",        # popup dropdowns
        "slider",          # focusable interactive
    }
    for role in expected_added_roles:
        assert f"'{role}'" in _JS_RESOLVE_CLICK, (
            f"ARIA role {role!r} missing from _JS_RESOLVE_CLICK accept list"
        )
