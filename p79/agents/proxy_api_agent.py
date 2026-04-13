"""ProxyApiAgent — calls a custom proxy API (Anthropic Messages style)."""

import logging
import os
import time
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
        self._system_prompts = self._get_system_prompts()

    # ---- system prompts (per observation mode) ----

    def _get_system_prompts(self) -> Dict[str, str]:
        _COMMON_RULES = """Core Rules:
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
8) If you are stuck, use scroll or try a different category/search."""

        _COMMON_RESPONSE_FORMAT = """{
  "thought": "Brief reasoning about what to do next. Why are you choosing this action? What is your plan?",
  "confidence": 0.7,
  "action_type": "click" | "type" | "scroll" | "wait" | "back" | "forward" | "finish" | "tab_focus",
  ... (other action parameters) ...
}"""

        _COMMON_SCROLL_AND_NAV = """3. Scroll: {"action_type": "scroll", "delta": [dx, dy], "coordinate_type": "normalized"}
   - dy>0 scrolls DOWN, dy<0 scrolls UP. Use scroll up when the target is above the current view.
4. Wait: {"action_type": "wait"}
5. Back: {"action_type": "back"}
   - WARNING: Do NOT use "back" if you are on the first page (homepage). Going back from the first page leads to a blank page (about:blank) and you will be stuck.
6. Forward: {"action_type": "forward"}
7. Finish: {"action_type": "finish", "answer": "optional string"}
8. Tab focus: {"action_type": "tab_focus", "page_number": int}"""

        _COMMON_TAB_RULE = """Tab Rule:
- Multi-site tasks may open multiple websites in different tabs. If the target site is in another tab, switch via tab_focus first.
- Element IDs are page-local to the current tab. Do NOT reuse IDs from a different tab/site.
- Do NOT search for a cross-site navigation link on the current page when the target site is already in another tab."""

        _COMMON_CRITICAL = """CRITICAL:
- You MUST include a "thought" field to explain your reasoning.
- "confidence" is a float 0.0-1.0 reflecting your certainty about this action being correct.
- DO NOT use "finish" to report failure. "finish" is ONLY for success or after EXHAUSTIVE search (at least 3 different search queries/attempts).
- Do NOT output literal newlines inside JSON strings. Use \\n for newline.
- Avoid repeating the same search query or action. If something doesn't work, change your strategy."""

        dom_prompt = f"""You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

Observation: You receive an Accessibility Tree (plain text) describing the page structure and interactive elements.

{_COMMON_RULES}

Response Format (JSON):
{_COMMON_RESPONSE_FORMAT}

Action Schema:
1. Click: {{"action_type": "click", "element_id": N}}
   - N is the numeric ID from the Accessibility Tree (e.g., [175] link 'Comments' -> element_id: 175).
   - This is the PREFERRED way to click. Use element IDs from the Accessibility Tree.
   - Alternative (only if no element ID): {{"action_type": "click", "coordinate": [x, y], "coordinate_type": "normalized"}} with x, y as floats 0.0-1.0.
2. Type: {{"action_type": "type", "text": "string", "element_id": N}}
   - ALWAYS specify element_id to target the correct input field.
   - To submit a search or form, append "\\n" to the text (e.g., "red blanket\\n").
   - Without element_id, text goes to whatever is focused, which is often WRONG.
{_COMMON_SCROLL_AND_NAV}

{_COMMON_TAB_RULE}
- If the Accessibility Tree lists tabs like "Tab 0" / "Tab 1", use tab_focus to switch to the tab that matches the site you need. Do NOT click random coordinates to switch tabs.

{_COMMON_CRITICAL}
- ALWAYS use element_id from the Accessibility Tree for click and type actions. Do NOT guess coordinates or type blindly.
- If you are in the wrong category, use the search bar or click a navigation link. Avoid "back" unless you are sure it won't lead to about:blank.
- PREFER clicking on Categories over searching if search results are poor.
"""

        som_prompt = f"""You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

Observation: You receive a [SOM_MARKS]...[/SOM_MARKS] list of labeled elements (each with an element_id and description) AND an annotated screenshot with numbered bounding boxes overlaid on the page. Use element_id from the marks for interaction; use normalized coordinates only when no element_id is available.

Note: If [SOM_MARKS] is empty (no elements detected), no bounding boxes will appear in the screenshot. In that case, fall back to coordinate-based interaction using what you can see in the screenshot.

{_COMMON_RULES}

Response Format (JSON):
{_COMMON_RESPONSE_FORMAT}

Action Schema:
1. Click: {{"action_type": "click", "element_id": N}}
   - N is the numeric ID from the SOM_MARKS list (e.g., [42] button 'Submit' -> element_id: 42).
   - This is the PREFERRED way to click. Use element IDs from SOM_MARKS.
   - Alternative (only if no element ID in marks): {{"action_type": "click", "coordinate": [x, y], "coordinate_type": "normalized"}} with x, y as floats 0.0-1.0.
2. Type: {{"action_type": "type", "text": "string", "element_id": N}}
   - ALWAYS specify element_id from SOM_MARKS to target the correct input field.
   - To submit a search or form, append "\\n" to the text (e.g., "red blanket\\n").
{_COMMON_SCROLL_AND_NAV}

{_COMMON_TAB_RULE}
- If the screenshot shows multiple tabs, use tab_focus to switch to the correct tab.

{_COMMON_CRITICAL}
- ALWAYS use element_id from SOM_MARKS for click and type. Use coordinates only as fallback when no ID is available.
- The annotated screenshot shows numbered boxes — match element_id numbers to the boxes in the image.
- If you are stuck, look at the screenshot carefully for visual cues not captured in SOM_MARKS.
"""

        vision_prompt = f"""You are a precise web navigation agent.
Output ONLY valid JSON. No markdown blocks, no explanations.

Observation: You receive ONLY a screenshot of the page. There is NO text-based element list. You must rely entirely on the visual content to navigate. Use normalized coordinates [x, y] (floats 0.0-1.0, origin top-left) for all click and type interactions.

{_COMMON_RULES}

Response Format (JSON):
{_COMMON_RESPONSE_FORMAT}

Action Schema:
1. Click: {{"action_type": "click", "coordinate": [x, y], "coordinate_type": "normalized"}}
   - x, y are floats 0.0-1.0 (e.g., center of screen is [0.5, 0.5]).
   - Estimate coordinates from the screenshot carefully. Click the center of the target element.
2. Type: {{"action_type": "type", "text": "string", "coordinate": [x, y], "coordinate_type": "normalized"}}
   - This action automatically clicks the target coordinate to focus it, then types the text.
   - ALWAYS use "type" (not "click") when you want to enter text into an input field.
   - To submit a search or form, append "\\n" to the text.
{_COMMON_SCROLL_AND_NAV}

{_COMMON_TAB_RULE}
- If the screenshot shows multiple tabs at the top, use tab_focus to switch tabs.

{_COMMON_CRITICAL}
- You MUST use normalized coordinates for all click/type actions. There are no element IDs.
- Look carefully at the screenshot to identify buttons, links, input fields, and other interactive elements by their visual appearance.
- If you are stuck, scroll to reveal more content or try a different visual element.
"""

        return {
            "dom": dom_prompt,
            "som": som_prompt,
            "vision": vision_prompt,
        }

    # ---- history formatting ----

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
                eid = act.get("element_id")
                coord = act.get("coordinate", "?")
                detail = f" [id={eid}]" if eid is not None else f" {coord}"
            elif atype == "type":
                detail = f' "{act.get("text", "")}"'
            elif atype == "scroll":
                detail = f' delta={act.get("delta", "?")}'
            success = rec.get("action_success", None)
            changed = rec.get("page_changed", None)
            if success is False:
                result = "FAILED"
            elif changed:
                result = "OK (page changed)"
            else:
                result = "OK (page unchanged)"
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
        observation_mode: str = "dom",
        reference_images: Optional[List[Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        image = obs.image
        obs_text = ""
        if hasattr(obs, "text") and obs.text:
            obs_text = obs.text
            max_chars = self.config.get("agent", {}).get("max_obs_chars", 8000)
            if len(obs_text) > max_chars:
                obs_text = obs_text[:max_chars] + "\n[TRUNCATED]"

        history_text = self._format_history(history or [])

        # Build obs_section per mode — mirrors qwen3vl_agent.py exactly.
        if observation_mode == "vision":
            obs_section = ""  # no text — screenshot only
        elif observation_mode == "som":
            # obs_text already contains [SOM_MARKS]...[/SOM_MARKS]; pass through directly.
            obs_section = obs_text if obs_text else ""
        else:
            obs_section = f"Accessibility Tree:\n{obs_text}"

        system_prompt = self._system_prompts.get(observation_mode, self._system_prompts["dom"])

        # Build user message content (Anthropic Messages style).
        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Task: {instruction}\n"
                    f"System: {system_prompt}\n"
                    f"{history_text}"
                    f"{obs_section}"
                ),
            }
        ]

        max_size = self.config.get("agent", {}).get("image_max_size", 1024)

        # Inject task reference images (e.g. product photos) before the screenshot.
        # Mirrors qwen3vl_agent.py reference_images handling.
        if reference_images:
            for idx, ref_img in enumerate(reference_images):
                try:
                    if max(ref_img.size) > max_size:
                        ratio = max_size / max(ref_img.size)
                        new_size = (int(ref_img.size[0] * ratio), int(ref_img.size[1] * ratio))
                        ref_img = ref_img.resize(new_size, Image.Resampling.LANCZOS)
                    ref_label = (
                        f"[Reference image {idx + 1}] "
                        f"This image shows the target item described in the task. "
                        f"Use it to identify which element to interact with."
                    )
                    ref_payload = self._image_to_data_url(ref_img)
                    user_content.append({"type": "text", "text": ref_label})
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": ref_payload["data_url"]},
                    })
                except Exception:
                    logger.warning("Failed to encode reference image %d; skipping.", idx + 1, exc_info=True)

        image_payload = None
        if image is not None:
            try:
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                image_payload = self._image_to_data_url(image)
                data_url: str = image_payload["data_url"]
                if reference_images:
                    # With reference images: append screenshot at end with label
                    user_content.append({"type": "text", "text": "[Current screenshot]"})
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
                else:
                    # No reference images: insert screenshot at position 0 (before text)
                    user_content.insert(0, {
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
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
        }

        headers = {
            "X-Api-Key": self.api_key,
            "Content-Type": "application/json",
        }

        _retryable_codes = {429, 500, 502, 503, 504}
        _max_retries = 3
        _backoff = 10  # seconds; doubles each attempt
        for _attempt in range(_max_retries + 1):
            resp = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
            if resp.status_code not in _retryable_codes or _attempt == _max_retries:
                break
            wait = _backoff * (2 ** _attempt)
            logger.warning(
                "API %s (attempt %d/%d), retrying in %ds...",
                resp.status_code, _attempt + 1, _max_retries, wait,
            )
            time.sleep(wait)
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

        # Auto-append newline for search queries.
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
            "input_tokens": (
                usage.get("inputTokens")
                or usage.get("input_tokens")
                or usage.get("prompt_tokens")
            ),
            "output_tokens": (
                usage.get("outputTokens")
                or usage.get("output_tokens")
                or usage.get("completion_tokens")
            ),
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
