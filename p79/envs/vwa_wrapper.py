from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os

@dataclass
class P79Observation:
    text: str
    image: Optional[Any] = None   # 可能是 PIL / np / base64 / path，先 Any
    url: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

class VWAWrapper:
    """
    Thin wrapper around (Visual)WebArena ScriptBrowserEnv.

    - reset(options={"config_file": ...})
    - step(action)
    """

    def __init__(
        self,
        headless: bool = True,
        observation_type: str = "accessibility_tree",
        current_viewport_only: bool = True,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        dry_run: bool = False,
    ) -> None:
        self.headless = headless
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.dry_run = dry_run

        self._env = None  # lazy init

    def _lazy_init(self) -> None:
        if self._env is not None:
            return

        # Ensure environment variables are set to avoid crash on import
        # User should set these to real values for actual tasks
        if "DATASET" not in os.environ:
            os.environ["DATASET"] = "visualwebarena"
        
        required_vars = ["REDDIT", "SHOPPING", "WIKIPEDIA", "HOMEPAGE", "CLASSIFIEDS", "CLASSIFIEDS_RESET_TOKEN"]
        for var in required_vars:
            if var not in os.environ:
                # Set dummy values if not present
                # Use example.com to allow page load without local server
                os.environ[var] = "https://example.com"

        from browser_env import ScriptBrowserEnv  # provided by (Visual)WebArena package

        self._env = ScriptBrowserEnv(
            headless=self.headless,
            observation_type=self.observation_type,
            current_viewport_only=self.current_viewport_only,
            viewport_size={"width": self.viewport_width, "height": self.viewport_height},
        )

    def reset(self, config_file: str) -> Tuple[P79Observation, Dict[str, Any]]:
        if self.dry_run:
            return P79Observation(text="[DRY_RUN]"), {"dry_run": True}

        self._lazy_init()
        assert self._env is not None

        obs, info = self._env.reset(options={"config_file": config_file})

        p79_obs = self._to_p79_obs(obs, info)
        return p79_obs, info

    def step(self, action_json: Dict[str, Any]) -> Tuple[P79Observation, float, bool, bool, Dict[str, Any]]:
        if self.dry_run:
            return P79Observation(text="[DRY_RUN]"), 0.0, False, False, {"dry_run": True}

        self._lazy_init()
        assert self._env is not None

        action_str = self._json_to_id_action_str(action_json)

        from browser_env import (
            create_id_based_action,
            create_mouse_click_action,
            create_scroll_action,
            create_stop_action,
            create_go_back_action,
            create_go_forward_action,
            create_keyboard_type_action,
            create_none_action
        )

        action_type = (action_json.get("action_type") or "").lower().strip()
        action = None

        if action_type == "click" and "coordinate" in action_json:
            coord = action_json["coordinate"]
            # Assuming normalized coordinates [0-1]
            action = create_mouse_click_action(left=coord[0], top=coord[1])
        elif action_type == "scroll" and "delta" in action_json:
            dy = action_json["delta"][1]
            direction = "down" if dy > 0 else "up"
            action = create_scroll_action(direction=direction)
        elif action_type == "type" and "text" in action_json and "element_id" not in action_json:
            # Type without ID -> keyboard type
            action = create_keyboard_type_action(action_json["text"])
        elif action_type == "back":
            action = create_go_back_action()
        elif action_type == "forward":
            action = create_go_forward_action()
        elif action_type in ("finish", "stop"):
            action = create_stop_action(action_json.get("answer", ""))
        elif action_type == "wait":
            action = create_none_action()

        if action is None:
            # Fallback to ID based
            action_str = self._json_to_id_action_str(action_json)
            action = create_id_based_action(action_str)

        obs, reward, terminated, truncated, info = self._env.step(action)
        p79_obs = self._to_p79_obs(obs, info)
        return p79_obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None

    # ---------- helpers ----------

    def _json_to_id_action_str(self, a: Dict[str, Any]) -> str:
        t = (a.get("action_type") or "").lower().strip()

        if t == "click":
            eid = a.get("element_id")
            if eid is None:
                raise ValueError(f"click requires element_id, got: {a}")
            return f"click [{int(eid)}]"

        if t == "type":
            eid = a.get("element_id")
            text = a.get("text", "")
            if eid is None:
                raise ValueError(f"type requires element_id, got: {a}")
            # 注意：文本里如果有 ']' 等符号，后续可以做转义；先跑通再说
            return f"type [{int(eid)}] [{text}]"

        if t == "scroll":
            direction = (a.get("direction") or "down").lower()
            # WebArena 常见方向：up/down/left/right
            return f"scroll [{direction}]"

        if t in ("stop", "finish", "done"):
            return "stop"

        if t == "wait":
            # 有些实现支持 wait；如果不支持就用 noop/stop 替代
            return "wait"

        # 兜底：如果 agent 直接给了 action_str
        if "action_str" in a:
            return str(a["action_str"])

        raise ValueError(f"Unknown action_type: {t}, raw={a}")

    def _to_p79_obs(self, obs: Dict[str, Any], info: Dict[str, Any]) -> P79Observation:
        # WebArena 文档提到可以从 obs["text"] 取文本观测（如 html / accessibility tree）:contentReference[oaicite:4]{index=4}
        text = ""
        if isinstance(obs, dict):
            text = obs.get("text", "") or ""

        # VWA 可能会包含 screenshot / image（不同版本 key 名可能不一样）
        image = None
        for k in ("image", "screenshot", "pixel", "rgb"):
            if isinstance(obs, dict) and k in obs:
                image = obs[k]
                break

        url = None
        if isinstance(info, dict):
            url = info.get("url") or info.get("current_url")

        return P79Observation(text=text, image=image, url=url, raw=obs)
