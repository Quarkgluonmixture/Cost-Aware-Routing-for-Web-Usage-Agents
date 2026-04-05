from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
import os
import re
try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None
from PIL import Image

@dataclass
class P79Observation:
    text: str
    image: Optional[Any] = None   # 可能是 PIL / np / base64 / path，先 Any
    url: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None
    # VWA obs_nodes_info: maps str(element_id) -> {"union_bound": [x,y,w,h], ...}
    # Populated from info["observation_metadata"]["text"]["obs_nodes_info"] by _to_p79_obs.
    obs_nodes_info: Optional[Dict[str, Any]] = None

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
        sleep_after_execution: float = 0.5,
        dry_run: bool = False,
    ) -> None:
        self.headless = headless
        self.observation_type = observation_type
        self.current_viewport_only = current_viewport_only
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.sleep_after_execution = sleep_after_execution
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
            sleep_after_execution=self.sleep_after_execution,
        )

    def reset(self, config_file: str) -> Tuple[P79Observation, Dict[str, Any]]:
        if self.dry_run:
            # Return dummy black image for dry run to satisfy agent
            dummy_img = Image.new('RGB', (self.viewport_width, self.viewport_height), color='black')
            return P79Observation(text="[DRY_RUN]", image=dummy_img), {"dry_run": True}

        self._lazy_init()
        assert self._env is not None

        try:
            obs, info = self._env.reset(options={"config_file": config_file})
        except Exception:
            # Keep wrapper recoverable across episodes after init/reset failures.
            self.close()
            raise

        p79_obs = self._to_p79_obs(obs, info)
        return p79_obs, info

    def step(self, action_json: Dict[str, Any]) -> Tuple[P79Observation, float, bool, bool, Dict[str, Any]]:
        if self.dry_run:
            dummy_img = Image.new('RGB', (self.viewport_width, self.viewport_height), color='black')
            return P79Observation(text="[DRY_RUN]", image=dummy_img), 0.0, False, False, {"dry_run": True}

        self._lazy_init()
        assert self._env is not None

        from browser_env import (
            create_id_based_action,
            create_mouse_click_action,
            create_scroll_action,
            create_stop_action,
            create_go_back_action,
            create_go_forward_action,
            create_page_focus_action,
            create_keyboard_type_action,
            create_none_action,
            create_playwright_action
        )

        action_type = (action_json.get("action_type") or "").lower().strip()
        action = None

        if action_type == "click" and "element_id" in action_json:
            # Prefer element_id click (id-based action via AXTree node)
            try:
                eid = int(action_json["element_id"])
                action = create_id_based_action(f"click [{eid}]")
            except (TypeError, ValueError):
                action = None
        elif action_type == "click" and "coordinate" in action_json:
            coord = action_json.get("coordinate")
            if not (
                isinstance(coord, (list, tuple))
                and len(coord) == 2
                and coord[0] is not None
                and coord[1] is not None
            ):
                coord = None
            if coord is not None:
                left = float(coord[0])
                top = float(coord[1])
                # Accept either normalized [0-1] or pixel coordinates.
                # Coordinate clicks are always treated as mouse clicks.
                if left > 1.0 or top > 1.0:
                    left = left / float(self.viewport_width)
                    top = top / float(self.viewport_height)
                # Avoid 0.0 which triggers VWA create_mouse_click_action validation
                eps = 1e-6
                if left <= 0.0:
                    left = eps
                elif left >= 1.0:
                    left = 1.0 - eps
                if top <= 0.0:
                    top = eps
                elif top >= 1.0:
                    top = 1.0 - eps
                action = create_mouse_click_action(left=left, top=top)
            else:
                action = None
        elif action_type == "scroll" and "delta" in action_json:
            dy = action_json["delta"][1]
            direction = "down" if dy > 0 else "up"
            action = create_scroll_action(direction=direction)
        elif action_type == "type" and "text" in action_json and "element_id" not in action_json:
            # Type without ID -> keyboard type
            action = create_keyboard_type_action(action_json["text"])
        elif action_type == "type" and "text" in action_json and "element_id" in action_json:
            # Treat invalid/zero element_id as keyboard typing fallback
            try:
                element_id = int(action_json.get("element_id"))
            except (TypeError, ValueError):
                element_id = None
            if element_id is None or element_id <= 0:
                action = create_keyboard_type_action(action_json["text"])
        elif action_type == "back":
            action = create_go_back_action()
        elif action_type == "forward":
            action = create_go_forward_action()
        elif action_type in ("tab", "tab_focus", "page_focus"):
            page_number = action_json.get("page_number")
            if page_number is None:
                page_number = action_json.get("tab_index")
            if page_number is None:
                thought = action_json.get("thought", "")
                match = re.search(r"tab\s*(\d+)", thought, re.IGNORECASE)
                if match:
                    page_number = int(match.group(1))
            if page_number is None:
                action = create_none_action()
            else:
                action = create_page_focus_action(page_number=int(page_number))
        elif action_type in ("finish", "stop"):
            answer = action_json.get("answer", "")
            action = create_stop_action("" if answer is None else str(answer))
        elif action_type == "wait":
            action = create_none_action()

        if action is None and action_type == "click" and "element_id" not in action_json:
            action = create_none_action()

        if action is None:
            # Fallback to action_str or ID based
            if "action_str" in action_json:
                action = create_playwright_action(str(action_json["action_str"]))
            else:
                action_str = self._json_to_id_action_str(action_json)
                action = create_id_based_action(action_str)

        try:
            obs, reward, terminated, truncated, info = self._env.step(action)
        except Exception:
            # Reset underlying resources so next episode can re-initialize cleanly.
            self.close()
            raise
        if action_type in ("finish", "stop"):
            terminated = True
        info["raw_action"] = action  # Expose the raw VWA action for trajectory recording
        p79_obs = self._to_p79_obs(obs, info)
        return p79_obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            finally:
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
            # VWA id-based parser cannot handle literal newlines inside text
            if isinstance(text, str):
                text = text.replace("\n", " ").replace("\r", " ")
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
                raw_img = obs[k]
                if np is not None and isinstance(raw_img, np.ndarray):
                    image = Image.fromarray(raw_img)
                else:
                    image = raw_img
                break

        url = None
        if isinstance(info, dict):
            url = info.get("url") or info.get("current_url")
            # VWA stores url inside info["page"].url (DetachedPage dataclass)
            if not url:
                page_obj = info.get("page")
                if page_obj is not None and hasattr(page_obj, "url"):
                    url = page_obj.url or None

        # Extract per-element bounding boxes from VWA observation metadata.
        # info["observation_metadata"]["text"]["obs_nodes_info"] maps str(element_id)
        # to {"union_bound": [x, y, width, height], ...} in pixel coordinates.
        obs_nodes_info: Optional[Dict[str, Any]] = None
        try:
            obs_nodes_info = (
                info.get("observation_metadata", {})
                    .get("text", {})
                    .get("obs_nodes_info")
            ) or None
        except Exception:
            pass

        return P79Observation(text=text, image=image, url=url, raw=obs, obs_nodes_info=obs_nodes_info)
