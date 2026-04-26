"""ProxyApiAgent — calls a custom proxy API (Anthropic Messages style)."""

import base64
import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from p79.backends.action_utils import parse_action_text, validate_action
from p79.backends.image_utils import DEFAULT_MAX_IMAGE_PAYLOAD_BYTES, encode_image_data_url

logger = logging.getLogger(__name__)

# Default timeout for API requests (seconds).
_DEFAULT_TIMEOUT = 120

# ---------------------------------------------------------------------------
# Tool-calling schema — forces structured output via the API's tool_use
# mechanism, completely eliminating parse_error from free-form text.
# Enabled via config: model.use_tool_calling = true
# ---------------------------------------------------------------------------
_WEB_ACTION_TOOL = {
    "name": "web_action",
    "description": (
        "Execute a web navigation action on the current page. "
        "Call this tool with your chosen action for every step."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "thought": {
                "type": "string",
                "description": "Brief reasoning about what to do next and why.",
            },
            "confidence": {
                "type": "number",
                "description": "Confidence 0.0-1.0 that this action is correct.",
            },
            "action_type": {
                "type": "string",
                "enum": [
                    "click", "type", "scroll", "wait", "back",
                    "forward", "finish", "select_option", "tab_focus",
                ],
                "description": "The type of action to perform.",
            },
            "element_id": {
                "type": "integer",
                "description": "Element ID from Accessibility Tree or SOM marks.",
            },
            "coordinate": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2,
                "description": "Normalized [x, y] coordinates (0.0-1.0).",
            },
            "text": {
                "type": "string",
                "description": "Text to type (for type action). Append \\n to submit.",
            },
            "scroll_direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "Scroll direction: 'down' to reveal content below, 'up' to reveal content above.",
            },
            "option_label": {
                "type": "string",
                "description": "Visible option text for select_option.",
            },
            "option_value": {
                "type": "string",
                "description": "Option value for select_option.",
            },
            "option_index": {
                "type": "integer",
                "description": "Option index for select_option.",
            },
            "answer": {
                "type": "string",
                "description": "Answer for finish action.",
            },
            "page_number": {
                "type": "integer",
                "description": "Tab number for tab_focus.",
            },
        },
        "required": ["action_type"],
    },
}


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

        # API format: "anthropic" (proxy default) or "openai" (DashScope, etc.)
        self._api_format = model_cfg.get("api_format", "anthropic")
        # Image format in messages: "openai" (image_url) or "anthropic_native" (source base64).
        # Auto-detect from model name if not set explicitly.
        self._image_format = model_cfg.get("image_format", "auto")
        if self._image_format == "auto":
            if "anthropic" in self.model_name.lower() or "claude" in self.model_name.lower():
                self._image_format = "anthropic_native"
            else:
                self._image_format = "openai"

        self.endpoint = model_cfg.get("base_url") or os.getenv("PROXY_API_ENDPOINT", "")
        if not self.endpoint:
            raise RuntimeError(
                "Proxy API endpoint not set. "
                "Set model.base_url in config or PROXY_API_ENDPOINT env var."
            )
        # OpenAI format: ensure endpoint ends with /chat/completions
        if self._api_format == "openai":
            ep = self.endpoint.rstrip("/")
            if not ep.endswith("/chat/completions"):
                self.endpoint = ep + "/chat/completions"
            else:
                self.endpoint = ep

        api_key_env = model_cfg.get("api_key_env", "PROXY_API_KEY")
        self.api_key = os.getenv(api_key_env) or ""
        if not self.api_key:
            raise RuntimeError(f"{api_key_env} environment variable is not set")

        self.timeout = model_cfg.get("timeout", _DEFAULT_TIMEOUT)
        self._use_tool_calling = model_cfg.get("use_tool_calling", False)

        # GLM fallback for parse-error recovery (Solution B).
        # Reads .auth/glm (3-line file: endpoint, model, api_key).
        # Cost is NOT counted in experiment metrics — purely scaffold overhead.
        self._glm_config: Optional[Dict[str, str]] = None
        glm_cfg_path = model_cfg.get("glm_config", ".auth/glm")
        if model_cfg.get("use_glm_fallback", False):
            self._glm_config = self._load_glm_config(glm_cfg_path)

        self._system_prompts = self._get_system_prompts()

        # When tool calling is enabled, replace the "output JSON" instruction
        # with a tool-use instruction.  This is a format-only change — the
        # rules, action schema, and all task-relevant guidance are unchanged.
        if self._use_tool_calling:
            _old = "Output ONLY valid JSON. No markdown blocks, no explanations."
            _new = "Use the web_action tool for every action. Put reasoning in the thought parameter."
            for mode in self._system_prompts:
                self._system_prompts[mode] = self._system_prompts[mode].replace(_old, _new)

    # ---- GLM fallback (Solution B) ----

    @staticmethod
    def _load_glm_config(cfg_path: str) -> Optional[Dict[str, str]]:
        """Load GLM config from a 3-line file: endpoint, model, api_key."""
        p = Path(cfg_path)
        if not p.exists():
            logger.warning("GLM config %s not found; GLM fallback disabled.", cfg_path)
            return None
        lines = [
            ln.strip()
            for ln in p.read_text(encoding="utf-8").splitlines()
            if ln.strip() and not ln.strip().startswith("#")
        ]
        if len(lines) < 3:
            logger.warning("GLM config %s needs 3 lines (endpoint/model/key); got %d. Disabled.", cfg_path, len(lines))
            return None
        cfg = {"endpoint": lines[0], "model": lines[1], "api_key": lines[2]}
        logger.info("GLM fallback enabled: model=%s", cfg["model"])
        return cfg

    def _call_glm_extract(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """Ask GLM to extract a JSON action from raw model output.

        Returns validated action dict on success, None on failure.
        This is a scaffold-level format repair — cost is NOT experiment cost.
        """
        if not self._glm_config:
            return None

        # Truncate to avoid sending huge payloads for a simple extraction task.
        truncated = raw_output[:4000]
        extract_prompt = (
            "Extract or infer the intended web navigation action from the following agent output.\n"
            "The output may be valid JSON, malformed JSON, or natural language describing the intended action.\n\n"
            "Rules:\n"
            '- Output a single JSON object with "action_type" (one of: click, type, scroll, wait, back, '
            "forward, finish, select_option, tab_focus).\n"
            "- Include relevant fields: element_id, coordinate, text, scroll_direction (up/down), "
            "option_label, answer, thought.\n"
            "- If the output contains JSON (possibly malformed), extract and fix it.\n"
            "- If the output is natural language, infer the action from the described intent.\n"
            '- For finish/stop actions, extract the answer from context (do NOT leave answer as "").\n'
            "- Output ONLY the JSON object. No explanation, no markdown.\n\n"
            "Examples:\n"
            'Input: "I need to click on the submit button which is element 42"\n'
            'Output: {"action_type": "click", "element_id": 42, "thought": "click submit button"}\n\n'
            'Input: "Let me scroll down to see more results on this page"\n'
            'Output: {"action_type": "scroll", "scroll_direction": "down", "thought": "scroll to see more results"}\n\n'
            'Input: "The answer to the question is $25.99, I should finish now"\n'
            'Output: {"action_type": "finish", "answer": "$25.99", "thought": "found the answer"}\n\n'
            f"Agent output:\n{truncated}"
        )

        messages = [
            {"role": "system", "content": "You extract or infer structured JSON actions from agent output. Output ONLY valid JSON."},
            {"role": "user", "content": extract_prompt},
        ]
        payload = json.dumps({
            "model": self._glm_config["model"],
            "messages": messages,
            "temperature": 0.0,
            # GLM-5.1 is a thinking model: reasoning_content consumes most of the
            # token budget.  512 causes content truncation for complex outputs.
            "max_tokens": 2048,
        }).encode("utf-8")

        ep = self._glm_config["endpoint"].rstrip("/")
        urls = [f"{ep}/chat/completions", ep] if not ep.endswith("/chat/completions") else [ep]

        for url in urls:
            req = urllib.request.Request(
                url,
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._glm_config['api_key']}",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices") or []
                if not choices:
                    continue
                import re as _re

                msg_obj = (choices[0].get("message") or {})
                text = msg_obj.get("content") or ""
                if not text.strip():
                    # Thinking model: try reasoning_content
                    text = msg_obj.get("reasoning_content") or ""
                text = text.strip()
                # Strip markdown code fence
                if text.startswith("```"):
                    text = text.strip("`")
                    if text.lower().startswith("json"):
                        text = text[4:].strip()
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    # Thinking model may embed JSON inside reasoning prose;
                    # try regex extraction as last resort.
                    # First try: allow nested braces (for coordinate arrays etc.)
                    m = _re.search(r"\{[^{}]*\"action_type\"[^}]*\}", text)
                    if not m:
                        # Second try: non-greedy match from action_type to end
                        m = _re.search(r"\{.*?\"action_type\".*?\}", text, _re.DOTALL)
                    if m:
                        parsed = json.loads(m.group())
                    else:
                        raise
                action, is_valid = validate_action(parsed)
                if is_valid:
                    logger.info("GLM fallback extracted action: %s", action.get("action_type"))
                    return action
                logger.warning("GLM fallback returned invalid action: %s", parsed)
                return None
            except Exception as exc:
                logger.warning("GLM fallback call failed (%s): %s", url, exc)
                continue
        return None

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
  "action_type": "click" | "type" | "select_option" | "scroll" | "wait" | "back" | "forward" | "finish" | "tab_focus",
  ... (other action parameters) ...
}"""

        _COMMON_SCROLL_AND_NAV = """3. Scroll: {"action_type": "scroll", "scroll_direction": "down"}
   - "down" reveals content below the current view, "up" reveals content above. Use "up" when the target is above.
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
2.5. Select Option: {{"action_type": "select_option", "element_id": N, "option_label": "Option Name"}}
   - Use ONLY for <select> dropdown elements (shown as "combobox" in the Accessibility Tree).
   - Clicking a combobox does NOT open the dropdown. Use select_option instead.
   - option_label must match the visible option text exactly (e.g., "Electronics", "Jewelry & Watches").
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
2.5. Select Option: {{"action_type": "select_option", "element_id": N, "option_label": "Option Name"}}
   - Use ONLY for <select> dropdown elements (shown as "combobox" in the SOM_MARKS list).
   - Clicking a combobox does NOT open the dropdown. Use select_option instead.
   - option_label must match the visible option text exactly (e.g., "Electronics", "Jewelry & Watches").
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
2.5. Select Option: {{"action_type": "select_option", "coordinate": [x, y], "option_label": "Option Name"}}
   - Use ONLY for <select> dropdown visible in the screenshot.
   - Clicking a dropdown does NOT open it. Use select_option to set the value directly.
   - option_label must match the visible option text exactly.
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
            # Phantom-SoM (§25): identical SoM prompt + SoM marks text, but no image.
            # Tests whether the model can complete tasks using SoM textual labels alone
            # ("mirage" mode — preserves prompt that mentions screenshot).
            "phantom_som": som_prompt,
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
        elif observation_mode in ("som", "phantom_som"):
            # obs_text already contains [SOM_MARKS]...[/SOM_MARKS]; pass through directly.
            # phantom_som receives the same text but no image (see som.py).
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

        # Convert image blocks for Claude/Anthropic native format if needed.
        if self._image_format == "anthropic_native":
            user_content = self._convert_images_to_anthropic(user_content)

        messages = [{"role": "user", "content": user_content}]

        gen_cfg = self.config.get("model", {})
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": gen_cfg.get("max_new_tokens", 512),
            "temperature": gen_cfg.get("temperature", 0.1),
        }

        if self._use_tool_calling:
            payload["tools"] = [_WEB_ACTION_TOOL]
            # Force the model to call web_action — guarantees structured output.
            payload["tool_choice"] = {"type": "tool", "name": "web_action"}

        if self._api_format == "openai":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        else:
            headers = {
                "X-Api-Key": self.api_key,
                "Content-Type": "application/json",
            }

        _retryable_codes = {429, 500, 502, 503, 504}
        _max_retries = 3
        _backoff = 10  # seconds; doubles each attempt
        resp = None
        for _attempt in range(_max_retries + 1):
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as net_exc:
                if _attempt == _max_retries:
                    raise
                wait = _backoff * (2 ** _attempt)
                logger.warning(
                    "API network error %s (attempt %d/%d), retrying in %ds...",
                    net_exc, _attempt + 1, _max_retries, wait,
                )
                time.sleep(wait)
                continue
            if resp.status_code not in _retryable_codes or _attempt == _max_retries:
                break
            wait = _backoff * (2 ** _attempt)
            logger.warning(
                "API %s (attempt %d/%d), retrying in %ds...",
                resp.status_code, _attempt + 1, _max_retries, wait,
            )
            time.sleep(wait)
        assert resp is not None, "API request failed: resp is None after all retries"
        resp.raise_for_status()
        try:
            resp_json = resp.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"API returned non-JSON response (status {resp.status_code}): {resp.text[:200]}") from exc

        # ----- Parse response -----
        # Normalize response: OpenAI format → extract from choices[0].message
        if self._api_format == "openai" and "choices" in resp_json:
            choices = resp_json["choices"]
            if not choices:
                raise RuntimeError("API returned empty choices list")
            msg = choices[0].get("message", {})
            raw_content = msg.get("content", "")
            reasoning_text_openai = msg.get("reasoning_content") or None
        else:
            raw_content = resp_json.get("content", "")
            reasoning_text_openai = None

        output_text = ""
        action = None
        valid = False
        fail_reason: Optional[str] = None
        reasoning_text: Optional[str] = reasoning_text_openai

        # Path 1: tool_use extraction (when enabled).
        if self._use_tool_calling and isinstance(raw_content, list):
            text_parts: List[str] = []
            tool_input: Optional[Dict[str, Any]] = None
            for block in raw_content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "tool_use" and block.get("name") == "web_action":
                    tool_input = block.get("input", {})
                elif block.get("type") == "text":
                    text_parts.append(block.get("text", ""))

            reasoning_text = "\n".join(text_parts).strip() or None

            if tool_input is not None:
                # Inject thought from text blocks if not already provided.
                if not tool_input.get("thought") and reasoning_text:
                    tool_input["thought"] = reasoning_text[:500]
                action, valid = validate_action(tool_input)
                fail_reason = None if valid else "invalid_tool_input"
                output_text = json.dumps(tool_input, ensure_ascii=False)
                if valid:
                    logger.info("Tool-use parsed: %s", action.get("action_type"))
                else:
                    logger.warning("Tool-use input invalid, falling back to text parse.")
                    action = None  # trigger fallback below

        # Path 2: text parsing fallback (original logic).
        if action is None:
            if isinstance(raw_content, str):
                output_text = raw_content
            elif isinstance(raw_content, list):
                for block in raw_content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        output_text = block.get("text", "")
                        break
                if not output_text:
                    output_text = str(raw_content)
            else:
                output_text = str(raw_content)
            action, valid, fail_reason = parse_action_text(output_text)

        # Path 3: GLM extraction fallback — only when parse failed.
        glm_fallback_used = False
        glm_fallback_attempted = False
        glm_fallback_ms = 0.0
        glm_original_fail_reason: Optional[str] = None
        if not valid and self._glm_config:
            glm_fallback_attempted = True
            glm_original_fail_reason = fail_reason  # remember what failed
            _t0 = time.monotonic()
            glm_action = self._call_glm_extract(output_text)
            glm_fallback_ms = (time.monotonic() - _t0) * 1000
            if glm_action is not None:
                action = glm_action
                valid = True
                fail_reason = None  # clear so runner doesn't mis-categorize
                glm_fallback_used = True

        # Convert semantic scroll_direction → delta for environment compatibility.
        if action.get("action_type") == "scroll" and "scroll_direction" in action:
            sd = action.pop("scroll_direction")
            action["delta"] = [0, 0.8] if sd == "down" else [0, -0.8]
            action.setdefault("coordinate_type", "normalized")

        # Auto-append newline for search queries.
        if action.get("action_type") == "type":
            text = action.get("text", "") or ""
            # action.get("thought") may be None — guard before .lower() (same as B1).
            thought = (action.get("thought") or "").lower()
            if ("search" in thought or "find" in thought or "look for" in thought) and not text.endswith("\n"):
                action["text"] = text + "\n"
                logger.info("Auto-appended newline to search query.")

        usage = resp_json.get("usage") or {}
        metadata = resp_json.get("metadata") or {}
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
            "reasoning_content": reasoning_text,
            "enable_thinking": False,
            "tool_calling": self._use_tool_calling,
            # GLM fallback tracking (cost NOT in model_cost — scaffold overhead only).
            "glm_fallback_used": glm_fallback_used,
            "glm_fallback_attempted": glm_fallback_attempted if glm_fallback_attempted else None,
            "glm_fallback_latency_ms": glm_fallback_ms if glm_fallback_attempted else None,
            "glm_original_fail_reason": glm_original_fail_reason,
            # Proxy-specific fields for analysis.
            "proxy_cost": usage.get("cost"),
            "proxy_remaining_quota": metadata.get("remaining_quota"),
        }

        return action, meta

    @staticmethod
    def _convert_images_to_anthropic(content: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI image_url blocks to Anthropic native image blocks.

        OpenAI:    {type: "image_url", image_url: {url: "data:image/jpeg;base64,..."}}
        Anthropic: {type: "image", source: {type: "base64", media_type: "image/jpeg", data: "..."}}
        """
        converted = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                data_url = block.get("image_url", {}).get("url", "")
                m = re.match(r"data:(image/\w+);base64,(.+)", data_url, re.DOTALL)
                if m:
                    converted.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": m.group(1),
                            "data": m.group(2),
                        },
                    })
                else:
                    logger.warning("Cannot convert image_url to anthropic format; skipping.")
            else:
                converted.append(block)
        return converted

    def _image_to_data_url(self, image: Image.Image) -> Dict[str, Any]:
        max_payload = self.config.get("agent", {}).get(
            "max_image_payload_bytes", DEFAULT_MAX_IMAGE_PAYLOAD_BYTES
        )
        return encode_image_data_url(image=image, max_payload_bytes=int(max_payload))
