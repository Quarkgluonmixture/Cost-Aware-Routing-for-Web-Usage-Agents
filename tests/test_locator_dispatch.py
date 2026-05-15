"""locator_dispatch F1 (dispose always) + F6 (walk_fail error category) invariants.

/stress A1.3 (2026-05-15) fix verification.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from p79.envs.locator_dispatch import (
    dispatch_id_based_click,
    dispatch_id_based_clear,
    dispatch_id_based_hover,
    dispatch_id_based_type,
    dispatch_id_based_upload,
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


def test_hover_walk_fail_distinct_category():
    page, handle, _ = _make_page(handle_returns_element=False)
    r = dispatch_id_based_hover(page, _NODES, 42)
    assert r["error"] == "walk_fail:no_hover_target_within_walk"


def test_upload_walk_fail_distinct_category():
    page, handle, _ = _make_page(handle_returns_element=False)
    r = dispatch_id_based_upload(page, _NODES, 42, "/tmp/x.txt")
    assert r["error"] == "walk_fail:no_file_input_within_walk"


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


def test_hover_disposes_on_hover_raise():
    page, handle, elem = _make_page(handle_returns_element=True, click_raises=True)
    dispatch_id_based_hover(page, _NODES, 42)
    handle.dispose.assert_called()
    elem.dispose.assert_called()


def test_upload_disposes_on_set_files_raise():
    page, handle, elem = _make_page(handle_returns_element=True, click_raises=True)
    dispatch_id_based_upload(page, _NODES, 42, "/tmp/x.txt")
    handle.dispose.assert_called()
    elem.dispose.assert_called()


def test_clear_routes_through_type_and_disposes():
    """clear() is a thin wrapper around type(text='') — dispose still works."""
    page, handle, elem = _make_page(handle_returns_element=True)
    r = dispatch_id_based_clear(page, _NODES, 42)
    assert r["success"] is True
    handle.dispose.assert_called()
    elem.dispose.assert_called()


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
