"""Protocol Reset #5 (action-set restore, 2026-05-20).

Covers the 4-layer restore of the upstream-compatible id-based action space
(hover / press / new_tab / close_tab / goto) that the P79 custom prompt had
dropped:

  - validator  (action_utils.validate_action_detailed + ALLOWED_ACTION_TYPES)
  - wrapper    (VWAWrapper._json_to_id_action_str escape-hatch + goto whitelist)
  - upstream   (create_id_based_action round-trip — the escape-hatch target)

Prompt + B0 tool schema are string/dict constants exercised by other suites;
here we lock the executable contract (validity + serialization + whitelist).
"""
import os
import sys
import types

import pytest

from p79.backends.action_utils import (
    ALLOWED_ACTION_TYPES,
    validate_action_detailed,
)
from p79.envs.vwa_wrapper import VWAWrapper

_VWA_HOST_VARS = (
    "REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB",
    "WIKIPEDIA", "MAP", "HOMEPAGE", "CLASSIFIEDS",
)


# ---------------------------------------------------------------------------
# Layer 1 — validator: restored action_types are accepted with proper schema
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("atype", ["hover", "press", "new_tab", "close_tab", "goto"])
def test_restored_types_in_allowed_set(atype):
    assert atype in ALLOWED_ACTION_TYPES


def test_hover_valid_element_id():
    action, valid, reason = validate_action_detailed(
        {"action_type": "hover", "element_id": 7}
    )
    assert valid is True and reason is None
    assert action["action_type"] == "hover"


def test_hover_valid_coordinate():
    action, valid, _ = validate_action_detailed(
        {"action_type": "hover", "coordinate": [0.5, 0.5], "coordinate_type": "normalized"}
    )
    assert valid is True


def test_hover_missing_target_invalid():
    action, valid, reason = validate_action_detailed({"action_type": "hover"})
    assert valid is False
    assert reason == "invalid_element_id"
    assert action == {"action_type": "wait"}


def test_hover_rejects_bool_element_id():
    # bool is an int subclass — must NOT validate as element_id (B-799 family).
    _a, valid, _r = validate_action_detailed({"action_type": "hover", "element_id": True})
    assert valid is False


def test_press_valid_key():
    action, valid, reason = validate_action_detailed(
        {"action_type": "press", "key": "Ctrl+Enter"}
    )
    assert valid is True and reason is None
    assert action["key"] == "Ctrl+Enter"


def test_press_accepts_key_comb_alias():
    action, valid, _ = validate_action_detailed(
        {"action_type": "press", "key_comb": "Enter"}
    )
    assert valid is True
    assert action["key"] == "Enter"  # canonicalized onto `key`


def test_press_empty_key_invalid():
    _a, valid, reason = validate_action_detailed({"action_type": "press", "key": "  "})
    assert valid is False
    assert reason == "invalid_schema_dict"


def test_new_tab_and_close_tab_take_no_args():
    for atype in ("new_tab", "close_tab"):
        action, valid, reason = validate_action_detailed({"action_type": atype})
        assert valid is True and reason is None
        assert action["action_type"] == atype


def test_goto_valid_url():
    action, valid, reason = validate_action_detailed(
        {"action_type": "goto", "url": "http://reddit.com/f/news"}
    )
    assert valid is True and reason is None
    assert action["url"] == "http://reddit.com/f/news"


def test_goto_empty_url_invalid():
    _a, valid, reason = validate_action_detailed({"action_type": "goto", "url": ""})
    assert valid is False
    assert reason == "invalid_schema_dict"


# ---------------------------------------------------------------------------
# Layer 2 — wrapper: escape-hatch serializer produces upstream id-based strings
# ---------------------------------------------------------------------------


def _bare_wrapper():
    # _json_to_id_action_str / _goto_allowed_hosts do not touch __init__ state
    # except self._env (guarded), so a __new__ instance is sufficient + cheap.
    return VWAWrapper.__new__(VWAWrapper)


def test_serialize_hover():
    w = _bare_wrapper()
    assert w._json_to_id_action_str({"action_type": "hover", "element_id": 12}) == "hover [12]"


def test_serialize_press():
    w = _bare_wrapper()
    assert w._json_to_id_action_str({"action_type": "press", "key": "Ctrl+v"}) == "press [Ctrl+v]"


def test_serialize_new_tab_close_tab():
    w = _bare_wrapper()
    assert w._json_to_id_action_str({"action_type": "new_tab"}) == "new_tab"
    assert w._json_to_id_action_str({"action_type": "close_tab"}) == "close_tab"


def test_serialize_press_missing_key_raises():
    w = _bare_wrapper()
    with pytest.raises(ValueError):
        w._json_to_id_action_str({"action_type": "press"})


# ---------------------------------------------------------------------------
# Layer 2b — goto domain whitelist (env-var portion; _env=None → tabs skipped)
# ---------------------------------------------------------------------------


def test_goto_allowed_origins_are_netloc_with_port(monkeypatch):
    # P1-2-B* (cross-AI 2026-05-20): the whitelist is netloc (host:port), NOT
    # bare hostname — on the A100 self-host every site is localhost:<port>, so a
    # hostname-only set would collapse to "any localhost port".
    monkeypatch.setenv("REDDIT", "http://127.0.0.1:9999")
    monkeypatch.setenv("WIKIPEDIA", "http://127.0.0.1:8888/wiki")
    for v in ("SHOPPING", "HOMEPAGE", "CLASSIFIEDS", "GITLAB", "MAP", "SHOPPING_ADMIN"):
        monkeypatch.delenv(v, raising=False)
    w = _bare_wrapper()
    w._env = None  # context.pages access raises → caught → env-var origins only
    origins = w._goto_allowed_hosts()
    assert "127.0.0.1:9999" in origins
    assert "127.0.0.1:8888" in origins
    # bare hostname must NOT be present (would re-open the port-collapse hole)
    assert "127.0.0.1" not in origins
    # a different port on the same host is NOT whitelisted
    assert "127.0.0.1:7770" not in origins


def test_goto_offsite_origin_not_whitelisted(monkeypatch):
    monkeypatch.setenv("REDDIT", "http://reddit.local:9999")
    for v in ("SHOPPING", "WIKIPEDIA", "HOMEPAGE", "CLASSIFIEDS", "GITLAB", "MAP", "SHOPPING_ADMIN"):
        monkeypatch.delenv(v, raising=False)
    w = _bare_wrapper()
    w._env = None
    origins = w._goto_allowed_hosts()
    assert "evil.example.com" not in origins
    assert "reddit.local:9999" in origins


# ---------------------------------------------------------------------------
# Layer 3 — upstream create_id_based_action parses the escape-hatch strings
# ---------------------------------------------------------------------------


def test_upstream_parses_restored_action_strings(monkeypatch):
    # The escape-hatch (`_json_to_id_action_str` → `create_id_based_action`)
    # relies on the upstream parser actually accepting these strings. Importing
    # the upstream package runs `browser_env/env_config.py` which requires the
    # VWA host env vars; set dummies + add the submodule to sys.path, then skip
    # cleanly if the upstream tree is unavailable in this environment.
    import sys

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(repo, "external", "visualwebarena"))
    monkeypatch.setenv("DATASET", "visualwebarena")
    for k in (
        "REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB",
        "WIKIPEDIA", "MAP", "HOMEPAGE", "CLASSIFIEDS",
    ):
        monkeypatch.setenv(k, os.environ.get(k) or "http://localhost")
    try:
        from browser_env.actions import create_id_based_action, ActionTypes
    except Exception as e:  # pragma: no cover - env-dependent
        pytest.skip(f"upstream browser_env not importable here: {e}")

    assert create_id_based_action("hover [5]")["action_type"] == ActionTypes.HOVER
    assert create_id_based_action("new_tab")["action_type"] == ActionTypes.NEW_TAB
    assert create_id_based_action("close_tab")["action_type"] == ActionTypes.PAGE_CLOSE
    assert create_id_based_action("press [Ctrl+v]")["action_type"] == ActionTypes.KEY_PRESS
    assert create_id_based_action("goto [http://reddit.com]")["action_type"] == ActionTypes.GOTO_URL


# ---------------------------------------------------------------------------
# Layer 2c — step()-level executable contract (P2-1 cross-AI fix: prior tests
# only exercised the private serializer; these drive VWAWrapper.step() with a
# fake VWA env so the dispatch + telemetry + goto whitelist are tested end-to-end)
# ---------------------------------------------------------------------------


def _step_wrapper(monkeypatch, *, open_urls=()):
    fake = types.SimpleNamespace(
        create_id_based_action=lambda s: {"kind": "id", "action_str": s},
        create_mouse_click_action=lambda left, top: {"kind": "mouse_click", "left": left, "top": top},
        create_mouse_hover_action=lambda left, top: {"kind": "hover_coord", "left": left, "top": top},
        create_scroll_action=lambda direction: {"kind": "scroll", "direction": direction},
        create_stop_action=lambda answer: {"kind": "stop", "answer": answer},
        create_go_back_action=lambda: {"kind": "back"},
        create_go_forward_action=lambda: {"kind": "forward"},
        create_page_focus_action=lambda page_number: {"kind": "tab", "page_number": page_number},
        create_keyboard_type_action=lambda text: {"kind": "type", "text": text},
        create_none_action=lambda: {"kind": "none"},
        create_playwright_action=lambda s: {"kind": "playwright", "action_str": s},
        create_goto_url_action=lambda url: {"kind": "goto", "url": url},
        ScriptBrowserEnv=None,
    )
    monkeypatch.setitem(sys.modules, "browser_env", fake)

    class _Ctx:
        pages = [types.SimpleNamespace(url=u) for u in open_urls]

    class _FakeEnv:
        context = _Ctx()

        def step(self, action):
            return {"text": "[1] link"}, 0.0, False, False, {"url": "http://mock.local"}

    w = VWAWrapper(viewport_width=1000, viewport_height=500)
    w._env = _FakeEnv()
    return w


def test_step_hover_coordinate_executes_not_noop(monkeypatch):
    # P1-1: coordinate hover must dispatch create_mouse_hover_action, NOT silently
    # no-op (the bug the cross-AI review caught).
    w = _step_wrapper(monkeypatch)
    _, _, _, _, info = w.step({"action_type": "hover", "coordinate": [100, 200]})
    assert info["raw_action"]["kind"] == "hover_coord"
    assert info["action_executed"]["dispatch_path"] == "coord_mouse_hover"


def test_step_hover_element_id(monkeypatch):
    w = _step_wrapper(monkeypatch)
    _, _, _, _, info = w.step({"action_type": "hover", "element_id": 5})
    assert info["raw_action"]["action_str"] == "hover [5]"
    assert info["action_executed"]["dispatch_path"] == "element_id"


def test_step_goto_allowed_whitelisted_origin(monkeypatch):
    for v in _VWA_HOST_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("REDDIT", "http://site.local:9999")
    w = _step_wrapper(monkeypatch)
    _, _, _, _, info = w.step({"action_type": "goto", "url": "http://site.local:9999/f/news"})
    assert info["goto_meta"]["allowed"] is True
    assert info["raw_action"]["kind"] == "goto"
    assert info["action_executed"]["dispatch_path"] == "whitelisted"


def test_step_goto_blocked_offsite(monkeypatch):
    for v in _VWA_HOST_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("REDDIT", "http://site.local:9999")
    w = _step_wrapper(monkeypatch)
    _, _, _, _, info = w.step({"action_type": "goto", "url": "http://evil.example.com/x"})
    assert info["goto_meta"]["allowed"] is False
    assert info["goto_meta"]["error"] == "offsite_blocked"
    assert info["raw_action"]["kind"] == "none"


def test_step_goto_blocked_wrong_port_same_host(monkeypatch):
    # P1-2-B*: port matters — a different port on a whitelisted host is off-site.
    for v in _VWA_HOST_VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("REDDIT", "http://localhost:9999")
    w = _step_wrapper(monkeypatch)
    _, _, _, _, info = w.step({"action_type": "goto", "url": "http://localhost:7770/admin"})
    assert info["goto_meta"]["allowed"] is False


def test_step_goto_relative_allowed(monkeypatch):
    for v in _VWA_HOST_VARS:
        monkeypatch.delenv(v, raising=False)
    w = _step_wrapper(monkeypatch)
    _, _, _, _, info = w.step({"action_type": "goto", "url": "/settings"})
    assert info["goto_meta"]["relative"] is True
    assert info["goto_meta"]["allowed"] is True


def test_step_goto_javascript_scheme_blocked(monkeypatch):
    # empty netloc but non-empty scheme is NOT a relative path → must block.
    for v in _VWA_HOST_VARS:
        monkeypatch.delenv(v, raising=False)
    w = _step_wrapper(monkeypatch)
    _, _, _, _, info = w.step({"action_type": "goto", "url": "javascript:alert(1)"})
    assert info["goto_meta"]["allowed"] is False


def test_step_press_via_escape_hatch(monkeypatch):
    w = _step_wrapper(monkeypatch)
    _, _, _, _, info = w.step({"action_type": "press", "key": "Enter"})
    assert info["raw_action"]["action_str"] == "press [Enter]"
    assert info["action_executed"]["dispatch_path"] == "id_based_escape_hatch"


def test_step_new_tab_and_close_tab_via_escape_hatch(monkeypatch):
    w = _step_wrapper(monkeypatch)
    _, _, _, _, info_new = w.step({"action_type": "new_tab"})
    assert info_new["raw_action"]["action_str"] == "new_tab"
    assert info_new["action_executed"]["dispatch_path"] == "id_based_escape_hatch"
    _, _, _, _, info_close = w.step({"action_type": "close_tab"})
    assert info_close["raw_action"]["action_str"] == "close_tab"
