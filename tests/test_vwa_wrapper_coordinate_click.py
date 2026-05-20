import sys
import types

from p79.envs.vwa_wrapper import VWAWrapper


def test_coordinate_click_with_integer_values_stays_mouse_click(monkeypatch):
    captured = {}

    def create_id_based_action(action_str):
        captured["id_action"] = action_str
        return {"kind": "id", "action_str": action_str}

    def create_mouse_click_action(left, top):
        captured["mouse_click"] = (left, top)
        return {"kind": "mouse", "left": left, "top": top}

    fake_browser_env = types.SimpleNamespace(
        create_id_based_action=create_id_based_action,
        create_mouse_click_action=create_mouse_click_action,
        create_mouse_hover_action=lambda left, top: {"kind": "hover_coord", "left": left, "top": top},
        create_scroll_action=lambda direction: {"kind": "scroll", "direction": direction},
        create_stop_action=lambda answer: {"kind": "stop", "answer": answer},
        create_go_back_action=lambda: {"kind": "back"},
        create_go_forward_action=lambda: {"kind": "forward"},
        create_page_focus_action=lambda page_number: {"kind": "tab", "page_number": page_number},
        create_keyboard_type_action=lambda text: {"kind": "type", "text": text},
        create_none_action=lambda: {"kind": "none"},
        create_playwright_action=lambda s: {"kind": "playwright", "action_str": s},
        # Protocol Reset #5 (action-set restore, 2026-05-20): wrapper now imports
        # create_goto_url_action for the goto whitelist branch — fake must match.
        create_goto_url_action=lambda url: {"kind": "goto", "url": url},
        ScriptBrowserEnv=None,
    )
    monkeypatch.setitem(sys.modules, "browser_env", fake_browser_env)

    class _FakeEnv:
        def step(self, action):
            captured["env_action"] = action
            return {"text": "[1] link"}, 0.0, False, False, {"url": "http://mock.local"}

    wrapper = VWAWrapper(viewport_width=1000, viewport_height=500)
    wrapper._env = _FakeEnv()

    _, _, _, _, info = wrapper.step({"action_type": "click", "coordinate": [100, 200]})

    assert "id_action" not in captured
    assert captured["env_action"]["kind"] == "mouse"
    assert captured["mouse_click"] == (0.1, 0.4)
    assert info["raw_action"]["kind"] == "mouse"

