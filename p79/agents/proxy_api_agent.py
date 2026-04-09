"""ProxyApiAgent — calls a custom proxy API (Anthropic Messages style)."""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from p79.backends.action_utils import parse_action_text
from p79.backends.image_utils import DEFAULT_MAX_IMAGE_PAYLOAD_BYTES, encode_image_data_url

logger = logging.getLogger(__name__)

# Default timeout for API requests (seconds).
_DEFAULT_TIMEOUT = 120


class ProxyApiAgent:
    """Agent that talks to a custom proxy endpoint (Anthropic Messages API style).

    Request:  POST endpoint  { model, messages, max_tokens, temperature, system }
    Response: { content: [{type:"text", text:"..."}], model, usage, metadata }
    Auth:     X-Api-Key header
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        model_cfg = config.get("model", {})
        self.model_name = model_cfg.get("api_name", "qwen.qwen3-vl-235b-a22b")
        self.endpoint = model_cfg.get("base_url") or os.getenv("PROXY_API_ENDPOINT", "")
        if not self.endpoint:
            raise RuntimeError(
                "Proxy API endpoint not set. "
                "Set model.base_url in config or PROXY_API_ENDPOINT env var."
            )

        self.api_key = os.getenv("PROXY_API_KEY") or ""
        if not self.api_key:
            raise RuntimeError("PROXY_API_KEY environment variable is not set")

        self.timeout = model_cfg.get("timeout", _DEFAULT_TIMEOUT)
        self.system_prompt = self._get_system_prompt()

    # ---- system prompt (identical to QwenApiAgent) ----

    def _get_system_prompt(self) -> str:
        return """You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

Core Rules:
1) Do NOT answer or finish immediately. You MUST navigate to find the item.
2) You are logged in as a user. For tasks involving your own content (e.g., "my listing", "my post", "my message"),
   navigate to account/profile sections instead of searching publicly.
3) If the target category (e.g., "Blankets & Throws") is not visible, look for a parent category (e.g., "Home & Kitchen") or use the search bar.
4) NEVER give up early. If you don't see the item, SEARCH for it using the search bar.
5) Only use "finish" when you have successfully completed the task (e.g., found the item, placed order) or if you have searched everywhere and are 100% sure it's missing.
6) For single-item tasks (find and navigate to ONE specific item/page), you MUST open that item's detail page before "finish".
   For collection tasks (return links/info for MULTIPLE items), you MAY "finish" from a list/search page
   after recording the required items in your answer.
7) If you are on the homepage, DO NOT go back. Start by searching or clicking a category.
8) If you are stuck, use scroll or try a different category/search.

Response Format (JSON):
{
  "thought": "Brief reasoning about what to do next. Why are you choosing this action? What is your plan?",
  "action_type": "click" | "type" | "scroll" | "wait" | "back" | "forward" | "finish" | "tab_focus",
  ... (other action parameters) ...
}

Action Schema:
1. Click: {"action_type": "click", "element_id": N}
   - N is the numeric ID from the Accessibility Tree (e.g., [175] link 'Comments' -> element_id: 175).
   - This is the PREFERRED way to click. Use element IDs from the Accessibility Tree.
   - Alternative (only if no element ID): {"action_type": "click", "coordinate": [x, y], "coordinate_type": "normalized"} with x, y as floats 0.0-1.0.
2. Type: {"action_type": "type", "text": "string", "element_id": N}
   - ALWAYS specify element_id to target the correct input field (e.g., search box [397], text field [132]).
   - To submit a search or form, append "\\n" to the text (e.g., "red blanket\\n").
   - Without element_id, text goes to whatever is focused, which is often WRONG.
3. Scroll: {"action_type": "scroll", "delta": [dx, dy], "coordinate_type": "normalized"}
4. Wait: {"action_type": "wait"}
5. Back: {"action_type": "back"}
   - WARNING: Do NOT use "back" if you are on the first page (homepage). Going back from the first page leads to a blank page (about:blank) and you will be stuck.
6. Forward: {"action_type": "forward"}
7. Finish: {"action_type": "finish", "answer": "optional string"}
8. Tab focus: {"action_type": "tab_focus", "page_number": int}

Tab Rule:
- If the Accessibility Tree lists tabs like "Tab 0" / "Tab 1", use tab_focus to switch to the tab that matches the site you need (e.g., Wikipedia). Do NOT click random coordinates to switch tabs.
- If the task says "Wikipedia site in the second tab", immediately use {"action_type":"tab_focus","page_number":1} before any clicks.
- Multi-site tasks may open multiple websites in different tabs. If the target site is in another tab, switch via tab_focus first.
- Element IDs are page-local to the current tab. Do NOT reuse IDs from a different tab/site.
- Do NOT search for a cross-site navigation link on the current page when the target site is already in another tab.

CRITICAL:
- You MUST include a "thought" field to explain your reasoning.
- DO NOT use "finish" to report failure. "finish" is ONLY for success or after EXHAUSTIVE search (at least 3 different search queries/attempts).
- If you are in the wrong category, use the search bar or click a navigation link. Avoid "back" unless you are sure it won't lead to about:blank.
- PREFER clicking on Categories (e.g., "Home & Kitchen" -> "Blankets & Throws") over searching if search results are poor.
- If search returns unrelated items (e.g. seafood instead of blankets), STOP searching immediately. Navigate via Categories.
- Do NOT output literal newlines inside JSON strings. Use \\n for newline.
- If search results appear, CLICK on the most promising item to verify details (price, color). Do not just stare at the list.
- Avoid repeating the same search query or action. If something doesn't work, change your strategy.
- ALWAYS use element_id from the Accessibility Tree for click and type actions. Do NOT guess coordinates or type blindly.
"""

    # ---- history formatting (identical to QwenApiAgent) ----

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
            url = str(rec.get("obs_url", "") or "")
            if not url:
                state_digest = rec.get("state_digest", {}) or {}
                url = str(state_digest.get("url_after", "") or "")
            url_suffix = f" [{url[:100]}]" if url else ""
            lines.append(f"  Step {rec.get('step_idx', '?')}: {atype}{detail} -> {result}{url_suffix}")
        return "Previous actions:\n" + "\n".join(lines) + "\n"

    # ---- main step ----

    def step(
        self,
        instruction: str,
        obs: Any,
        history: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image = obs.image
        obs_text = ""
        if hasattr(obs, "text") and obs.text:
            obs_text = obs.text
            max_chars = self.config.get("agent", {}).get("max_obs_chars", 8000)
            if len(obs_text) > max_chars:
                obs_text = obs_text[:max_chars] + "\n[TRUNCATED]"

        history_text = self._format_history(history or [])

        # Build user message content (Anthropic Messages style).
        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Task: {instruction}\n"
                    f"{history_text}"
                    f"Accessibility Tree:\n{obs_text}"
                ),
            }
        ]

        image_payload = None
        if image is not None:
            try:
                max_size = self.config.get("agent", {}).get("image_max_size", 1024)
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                image_payload = self._image_to_data_url(image)
                # Anthropic style: {type:"image", source:{type:"base64", media_type, data}}
                data_url: str = image_payload["data_url"]
                # Strip "data:image/jpeg;base64," prefix to get raw base64
                raw_b64 = data_url.split(",", 1)[1] if "," in data_url else data_url
                user_content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": raw_b64,
                        },
                    }
                )
            except Exception:
                logger.warning("Failed to encode image; continuing without image.", exc_info=True)
                image_payload = None

        messages = [{"role": "user", "content": user_content}]

        gen_cfg = self.config.get("model", {})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": gen_cfg.get("max_new_tokens", 512),
            "temperature": gen_cfg.get("temperature", 0.1),
            "system": self.system_prompt,
        }

        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

        resp = requests.post(
            self.endpoint,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        resp_json = resp.json()

        # Parse response.  content may be a plain string or a list of blocks.
        raw_content = resp_json.get("content", "")
        if isinstance(raw_content, str):
            output_text = raw_content
        elif isinstance(raw_content, list):
            output_text = ""
            for block in raw_content:
                if isinstance(block, dict) and block.get("type") == "text":
                    output_text = block.get("text", "")
                    break
        else:
            output_text = str(raw_content)

        action, valid, fail_reason = parse_action_text(output_text)

        # Auto-append newline for search queries (same as QwenApiAgent).
        if action.get("action_type") == "type":
            text = action.get("text", "")
            thought = action.get("thought", "").lower()
            if ("search" in thought or "find" in thought or "look for" in thought) and not text.endswith("\n"):
                action["text"] = text + "\n"
                logger.info("Auto-appended newline to search query.")

        usage = resp_json.get("usage", {})
        metadata = resp_json.get("metadata", {})
        meta = {
            "raw_output": output_text,
            "valid": valid,
            "failure_reason": fail_reason,
            "input_tokens": usage.get("inputTokens"),
            "output_tokens": usage.get("outputTokens"),
            "thinking_tokens": None,
            "image_payload_bytes": image_payload.get("payload_bytes") if image_payload else None,
            "image_quality": image_payload.get("quality") if image_payload else None,
            "image_compressed": image_payload.get("compressed") if image_payload else None,
            "reasoning_content": None,
            "enable_thinking": False,
            # Proxy-specific fields for analysis.
            "proxy_cost": usage.get("cost"),
            "proxy_remaining_quota": metadata.get("remaining_quota"),
        }

        return action, meta

    def _image_to_data_url(self, image: Image.Image) -> Dict[str, Any]:
        max_payload = self.config.get("agent", {}).get(
            "max_image_payload_bytes", DEFAULT_MAX_IMAGE_PAYLOAD_BYTES
        )
        return encode_image_data_url(image=image, max_payload_bytes=int(max_payload))
