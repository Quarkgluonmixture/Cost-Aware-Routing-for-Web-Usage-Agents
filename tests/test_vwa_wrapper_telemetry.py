"""B-421 (/stress A1.3 v9 Mode A P1-8, 2026-05-17): regression test for
locator_route_meta + select_option_meta dispatch telemetry landing in
the post-step `info` dict.

Why a separate test file: the existing `test_vwa_wrapper_coordinate_click.py`
covers the coord-click path which DOES NOT invoke locator-route dispatch
(coord paths use `create_mouse_click_action`, not `dispatch_id_based_*`).
The id-based click + id-based type paths are the locator-route consumers
and they're what paper §3 evidence-layer ON_TARGET rate audit reads from
JSONL. Pre-fix, B-156 added the field to step_record but no integration
test asserted the wire was actually connected — fix could silently regress
without detection (archive coverage 0/53924 confirmed pre-B-156 data, no
post-fix data yet because Phase 1a re-fire pending).

These tests use fake `_lr_click` + `_lr_type` modules so we exercise the
wrapper integration path without needing a real Playwright browser.
"""

import sys
import types

import pytest


def _build_fake_browser_env(captured):
    return types.SimpleNamespace(
        create_id_based_action=lambda action_str: {"kind": "id", "action_str": action_str},
        create_mouse_click_action=lambda left, top: {"kind": "mouse", "left": left, "top": top},
        create_scroll_action=lambda direction: {"kind": "scroll", "direction": direction},
        create_stop_action=lambda answer: {"kind": "stop", "answer": answer},
        create_go_back_action=lambda: {"kind": "back"},
        create_go_forward_action=lambda: {"kind": "forward"},
        create_page_focus_action=lambda page_number: {"kind": "tab", "page_number": page_number},
        create_keyboard_type_action=lambda text: {"kind": "type", "text": text},
        create_none_action=lambda: {"kind": "none"},
        create_playwright_action=lambda s: {"kind": "playwright", "action_str": s},
        ScriptBrowserEnv=None,
    )


def test_locator_route_meta_lands_in_info_on_click(monkeypatch):
    """B-421: id-based click MUST surface locator_route_meta in info dict."""
    from p79.envs.vwa_wrapper import VWAWrapper

    captured = {}

    fake_lr_result = {
        "success": True,
        "fallback_used": False,
        "target_tag": "A",
        "error": None,
    }

    def fake_lr_click(page, obs_nodes_info, eid, *, sleep_after_ms=0):
        captured["lr_click_eid"] = eid
        return dict(fake_lr_result)

    fake_locator_dispatch = types.SimpleNamespace(
        dispatch_id_based_click=fake_lr_click,
        dispatch_id_based_type=lambda *a, **k: fake_lr_result,
    )
    monkeypatch.setitem(sys.modules, "p79.envs.locator_dispatch", fake_locator_dispatch)
    monkeypatch.setitem(sys.modules, "browser_env", _build_fake_browser_env(captured))

    class _FakeContext:
        def __init__(self):
            self.pages = []

    class _FakeEnv:
        def __init__(self):
            self.context = _FakeContext()
            self.page = types.SimpleNamespace()

        def step(self, action):
            return {"text": "[1] link"}, 0.0, False, False, {"url": "http://mock.local"}

    wrapper = VWAWrapper(viewport_width=1000, viewport_height=500)
    wrapper._env = _FakeEnv()
    # Pre-populate obs_nodes_info so locator-route does NOT short-circuit
    wrapper._last_obs_nodes_info = {"42": {"union_bound": [10, 20, 30, 40]}}

    _, _, _, _, info = wrapper.step({"action_type": "click", "element_id": 42})

    # B-421 assertions
    assert info.get("locator_route_meta") is not None, "click MUST emit locator_route_meta"
    assert info["locator_route_meta"]["action_kind"] == "click"
    assert info["locator_route_meta"]["success"] is True
    assert info["locator_route_meta"]["target_tag"] == "A"
    # Symmetric: select_option_meta MUST be None on non-select_option steps
    assert info.get("select_option_meta") is None


def test_locator_route_meta_lands_in_info_on_type(monkeypatch):
    """B-421 sibling: id-based type MUST surface locator_route_meta too."""
    from p79.envs.vwa_wrapper import VWAWrapper

    captured = {}

    fake_lr_result = {
        "success": True,
        "fallback_used": False,
        "target_tag": "INPUT",
        "error": None,
    }

    def fake_lr_type(page, obs_nodes_info, eid, text, *, sleep_after_ms=0, press_enter=False):
        captured["lr_type_eid"] = eid
        captured["lr_type_text"] = text
        return dict(fake_lr_result)

    fake_locator_dispatch = types.SimpleNamespace(
        dispatch_id_based_click=lambda *a, **k: fake_lr_result,
        dispatch_id_based_type=fake_lr_type,
    )
    monkeypatch.setitem(sys.modules, "p79.envs.locator_dispatch", fake_locator_dispatch)
    monkeypatch.setitem(sys.modules, "browser_env", _build_fake_browser_env(captured))

    class _FakeContext:
        def __init__(self):
            self.pages = []

    class _FakeEnv:
        def __init__(self):
            self.context = _FakeContext()
            self.page = types.SimpleNamespace()

        def step(self, action):
            return {"text": "[1] textbox"}, 0.0, False, False, {"url": "http://mock.local"}

    wrapper = VWAWrapper(viewport_width=1000, viewport_height=500)
    wrapper._env = _FakeEnv()
    wrapper._last_obs_nodes_info = {"42": {"union_bound": [10, 20, 30, 40]}}

    _, _, _, _, info = wrapper.step(
        {"action_type": "type", "element_id": 42, "text": "hello"}
    )

    assert info.get("locator_route_meta") is not None
    assert info["locator_route_meta"]["action_kind"] == "type"
    assert info["locator_route_meta"]["success"] is True


def test_locator_route_meta_none_on_scroll(monkeypatch):
    """B-421: non-locator-route action types MUST set locator_route_meta=None."""
    from p79.envs.vwa_wrapper import VWAWrapper

    captured = {}
    monkeypatch.setitem(sys.modules, "browser_env", _build_fake_browser_env(captured))

    class _FakeEnv:
        def __init__(self):
            self.context = types.SimpleNamespace(pages=[])
            self.page = types.SimpleNamespace()

        def step(self, action):
            return {"text": ""}, 0.0, False, False, {"url": "http://mock.local"}

    wrapper = VWAWrapper(viewport_width=1000, viewport_height=500)
    wrapper._env = _FakeEnv()
    wrapper._last_obs_nodes_info = None

    _, _, _, _, info = wrapper.step({"action_type": "scroll", "delta": [0, 0.8]})

    assert "locator_route_meta" in info, "key MUST be present (paper §3 evidence layer)"
    assert info["locator_route_meta"] is None
    assert "select_option_meta" in info
    assert info["select_option_meta"] is None
