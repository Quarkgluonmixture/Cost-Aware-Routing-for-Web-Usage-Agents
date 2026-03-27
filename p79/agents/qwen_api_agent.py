import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image
from openai import OpenAI

from p79.backends.image_utils import DEFAULT_MAX_IMAGE_PAYLOAD_BYTES, encode_image_data_url

logger = logging.getLogger(__name__)


class QwenApiAgent:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_cfg = config.get("model", {})
        self.model_name = model_cfg.get("api_name", model_cfg.get("name", "qwen-vl-max"))
        self.base_url = model_cfg.get("base_url") or os.getenv(
            "QWEN_API_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        )
        self.api_key = os.getenv("QWEN_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("QWEN_API_KEY (or DASHSCOPE_API_KEY) is not set")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.system_prompt = self._get_system_prompt()

    def _get_system_prompt(self) -> str:
        return """You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

Core Rules:
1) Do NOT answer or finish immediately. You MUST navigate to find the item.
2) If the target category (e.g., "Blankets & Throws") is not visible, look for a parent category (e.g., "Home & Kitchen") or use the search bar.
3) NEVER give up early. If you don't see the item, SEARCH for it using the search bar.
4) Only use "finish" when you have successfully completed the task (e.g., found the item, placed order) or if you have searched everywhere and are 100% sure it's missing.
5) If you are on the homepage, DO NOT go back. Start by searching or clicking a category.
6) If you are stuck, use scroll or try a different category/search.

Response Format (JSON):
{
  "thought": "Brief reasoning about what to do next. Why are you choosing this action? What is your plan?",
  "action_type": "click" | "type" | "scroll" | "wait" | "back" | "forward" | "finish" | "tab_focus",
  ... (other action parameters) ...
}

Action Schema:
1. Click: {"action_type": "click", "coordinate": [x, y], "coordinate_type": "normalized"}
   - x, y are floats 0.0-1.0.
2. Type: {"action_type": "type", "text": "string", "element_id": int (optional)}
   - To submit a search or form, append "\\n" to the text (e.g., "red blanket\\n").
3. Scroll: {"action_type": "scroll", "delta": [dx, dy], "coordinate_type": "normalized"}
4. Wait: {"action_type": "wait"}
5. Back: {"action_type": "back"}
6. Forward: {"action_type": "forward"}
7. Finish: {"action_type": "finish", "answer": "optional string"}
8. Tab focus: {"action_type": "tab_focus", "page_number": int}

Tab Rule:
- If the Accessibility Tree lists tabs like "Tab 0" / "Tab 1", use tab_focus to switch to the tab that matches the site you need (e.g., Wikipedia). Do NOT click random coordinates to switch tabs.
- If the task says "Wikipedia site in the second tab", immediately use {"action_type":"tab_focus","page_number":1} before any clicks.

CRITICAL:
- You MUST include a "thought" field to explain your reasoning.
- DO NOT use "finish" to report failure. "finish" is ONLY for success or after EXHAUSTIVE search (at least 3 different search queries/attempts).
- If you are in the wrong category, click "back" or use the search bar.
- PREFER clicking on Categories (e.g., "Home & Kitchen" -> "Blankets & Throws") over searching if search results are poor.
- If search returns unrelated items (e.g. seafood instead of blankets), STOP searching immediately. Navigate via Categories.
- Do NOT output literal newlines inside JSON strings. Use \\n for newline.
- If search results appear, CLICK on the most promising item to verify details (price, color). Do not just stare at the list.
- Avoid repeating the same search query or action. If something doesn't work, change your strategy.
"""

    @staticmethod
    def _format_history(history: List[Dict[str, Any]]) -> str:
        if not history:
            return ""
        lines = []
        for rec in history:
            act = rec.get("action", {})
            atype = act.get("action_type", "?")
            detail = ""
            if atype == "click":
                coord = act.get("coordinate", "?")
                detail = f" {coord}"
            elif atype == "type":
                detail = f' "{act.get("text", "")}"'
            elif atype == "scroll":
                detail = f' delta={act.get("delta", "?")}'
            success = rec.get("action_success", None)
            changed = rec.get("page_changed", None)
            if success is False or changed is False:
                result = "FAILED (page unchanged)"
            elif changed:
                result = "OK (page changed)"
            else:
                result = "OK"
            lines.append(f"  Step {rec.get('step_idx', '?')}: {atype}{detail} -> {result}")
        return "Previous actions:\n" + "\n".join(lines) + "\n"

    def step(self, instruction: str, obs: Any, history: Optional[List[Dict[str, Any]]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image = obs.image
        obs_text = ""
        if hasattr(obs, "text") and obs.text:
            obs_text = obs.text
            max_chars = self.config.get("agent", {}).get("max_obs_chars", 8000)
            if len(obs_text) > max_chars:
                obs_text = obs_text[:max_chars] + "\n[TRUNCATED]"

        history_text = self._format_history(history or [])

        max_size = self.config.get("agent", {}).get("image_max_size", 1024)
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        image_payload = self._image_to_data_url(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"Task: {instruction}\nSystem: {self.system_prompt}\n"
                            f"{history_text}"
                            f"Accessibility Tree:\n{obs_text}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": image_payload["data_url"]},
                    },
                ],
            }
        ]

        gen_cfg = self.config.get("model", {})
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=gen_cfg.get("temperature", 0.1),
            top_p=gen_cfg.get("top_p", 0.9),
            max_tokens=gen_cfg.get("max_new_tokens", 256),
        )

        output_text = response.choices[0].message.content or ""
        action, valid, fail_reason = self._parse_and_validate(output_text)

        if action.get("action_type") == "type":
            text = action.get("text", "")
            thought = action.get("thought", "").lower()
            if ("search" in thought or "find" in thought or "look for" in thought) and not text.endswith("\n"):
                action["text"] = text + "\n"
                logger.info("Auto-appended newline to search query.")

        usage = getattr(response, "usage", None)
        meta = {
            "raw_output": output_text,
            "valid": valid,
            "failure_reason": fail_reason,
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "image_payload_bytes": image_payload.get("payload_bytes"),
            "image_quality": image_payload.get("quality"),
            "image_compressed": image_payload.get("compressed"),
        }

        return action, meta

    def _image_to_data_url(self, image: Image.Image) -> Dict[str, Any]:
        max_payload = self.config.get("agent", {}).get("max_image_payload_bytes", DEFAULT_MAX_IMAGE_PAYLOAD_BYTES)
        return encode_image_data_url(image=image, max_payload_bytes=int(max_payload))

    def _parse_and_validate(self, text: str) -> Tuple[Dict[str, Any], bool, Optional[str]]:
        text = text.strip()
        lower_text = text.lower()

        try:
            action = json.loads(text)
            return self._validate_schema(action), True, None
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                action = json.loads(match.group(0))
                return self._validate_schema(action), True, "repaired_regex"
            except json.JSONDecodeError:
                pass

        if "scroll" in lower_text:
            return {
                "action_type": "scroll",
                "delta": [0, 0.8],
                "coordinate_type": "normalized",
            }, False, "keyword_scroll"
        if "back" in lower_text:
            return {"action_type": "back"}, False, "keyword_back"
        if "finish" in lower_text or "stop" in lower_text:
            return {"action_type": "finish", "answer": ""}, False, "keyword_finish"
        if "wait" in lower_text:
            return {"action_type": "wait"}, False, "keyword_wait"

        logger.warning("Failed to parse action from model output.")
        return {"action_type": "wait"}, False, "parse_failed"

    def _validate_schema(self, action: Dict[str, Any]) -> Dict[str, Any]:
        if "action_type" not in action:
            return {"action_type": "wait"}
        if action["action_type"] == "click":
            if "coordinate" not in action and "element_id" not in action:
                return {"action_type": "wait"}
            if "coordinate" in action and "coordinate_type" not in action:
                action["coordinate_type"] = "normalized"
        return action
