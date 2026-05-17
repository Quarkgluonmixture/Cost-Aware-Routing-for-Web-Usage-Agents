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
        #
        # ⚠️ DEPRECATED — MARKED FOR FULL RETIRE (B-145, /stress A1.2 v8, 2026-05-16).
        # Cross-baseline cost-fairness violation: B1/B2 have no equivalent
        # recovery model, so enabling GLM gives B0 a unique "free retry" that
        # invalidates paper §1 cost-fair comparison. Default is now
        # ``use_glm_fallback: false`` (configs/exp_v2_base.yaml:160). This block
        # is preserved only as a fallback during the transition window pending
        # advisor sync on the Qwen official API channel (which exposes
        # tool_choice and removes the parse-error root cause). Once that lands,
        # delete this entire ``_glm_config`` / ``_call_glm_extract`` / Solution-B
        # codepath; the corresponding config key in exp_v2_base.yaml is removed
        # simultaneously.
        self._glm_config: Optional[Dict[str, str]] = None
        glm_cfg_path = model_cfg.get("glm_config", ".auth/glm")
        if model_cfg.get("use_glm_fallback", False):
            import warnings
            warnings.warn(
                "GLM fallback (Solution B) is deprecated and marked for retire; "
                "enabling it violates paper §1 cross-baseline cost-fairness. Set "
                "use_glm_fallback: false (now the default) for any paper-grade run.",
                DeprecationWarning,
                stacklevel=2,
            )
            # B-340 (/stress A1.9 Mode C F4 defense-in-depth, 2026-05-16):
            # paper-grade mode hard-blocks GLM fallback enable (defense
            # against config drift / accidental yaml override). The
            # DeprecationWarning above is easy to miss in noisy log; this
            # explicit raise makes any paper-grade run with GLM enabled
            # fail at construction rather than silently corrupting cost-
            # fairness mid-fire. Set `paper_grade: false` for dev/legacy.
            if config.get("paper_grade", False):
                raise RuntimeError(
                    "use_glm_fallback=true is forbidden in paper-grade mode "
                    "(B-340). GLM fallback gives B0 an asymmetric 'free parse-"
                    "error rescue' service that B1/B2 do not have → "
                    "violates cross-baseline cost-fairness in paper §1. "
                    "Set `use_glm_fallback: false` or `paper_grade: false`."
                )
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

    @staticmethod
    def _get_system_prompts() -> Dict[str, str]:
        """Build the per-mode system-prompt dispatch table for B0.

        B-451 (/stress A1.4 P0-5-A* OOB, 2026-05-17): use the canonical
        `build_mode_prompt_dispatch_table` so B0/B1/B2 + mechanistic
        extractor consume the same 7-key dict from one place. Pre-B-451
        each consumer re-listed the same dict locally; B-103 (DOM/phantom_prompt
        missing `Accessibility Tree:\\n` prefix in mechanistic path) was caused
        by exactly this drift surface.

        Historical context retained for reviewers tracing prompt parity:
        - /stress A1.1 F1 (2026-05-15): B0 reuses canonical builders from
          `_shared_vl_utils` (was diverged with shopping-specific examples
          leaking domain prior into cross-baseline comparison).
        - B-402 (/stress A1.1 v8 Mode A P1-4, 2026-05-16): direct import from
          `_shared_vl_utils` (was: `Qwen3VLAgent._make_*_prompt()` indirection
          which transitively pulled heavy Qwen-VL deps into pure-network B0).
        - B-451 (current): consolidate to single dispatch-table factory.
        """
        from p79.agents._shared_vl_utils import build_mode_prompt_dispatch_table

        return build_mode_prompt_dispatch_table()

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
        # B-84: no max_obs_chars truncation. It fired on ~0.2% of steps but only
        # on AXTree modes (marks modes derive from the untruncated text), an
        # axis-1 page-coverage asymmetry. The viewport filter is the real input
        # bound (empirically median 3306 / p99 7656 / max 46592 chars).

        history_text = self._format_history(history or [])

        # Build obs_section per mode — mirrors qwen3vl_agent.py exactly.
        if observation_mode == "vision":
            obs_section = ""  # no text — screenshot only
        elif observation_mode in ("som", "phantom_som", "phantom_dom", "phantom_text"):
            # obs_text already contains [SOM_MARKS]...[/SOM_MARKS]; pass through directly.
            # phantom_som receives the same text but no image (see som.py).
            # phantom_text is the current name for phantom_dom (legacy alias preserved).
            obs_section = obs_text if obs_text else ""
        else:
            # "dom" or "phantom_prompt": AXTree text. phantom_prompt uses SoM-prompt
            # (set above) but the obs payload is AXTree (no [SOM_MARKS] markers).
            obs_section = f"Accessibility Tree:\n{obs_text}"

        # /stress A1.4 F2: strict mode validation (was silent .get(mode, dom_default)).
        # som.py's prepare_observation_for_mode already raises on unknown mode at the
        # observation-pipeline entry; this is defense-in-depth for any code path that
        # reaches the agent directly with a typo'd mode.
        if observation_mode not in self._system_prompts:
            raise ValueError(
                f"Unknown observation_mode {observation_mode!r}; expected one of "
                f"{sorted(self._system_prompts)}"
            )
        system_prompt = self._system_prompts[observation_mode]

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

        # /stress A1.1 codex Mode B C2: track image-encode failures so meta can
        # surface them to the step record (audit-able from JSONL) instead of the
        # silent text-only-episode contamination pattern. B1/B2 raise on encode
        # failure; B0 keeps lenient try/except (proxy-side transient errors) but
        # the meta flag lets downstream symmetric-exclude / paper-grade audit.
        _image_encode_error_count = 0
        # B-400 (/stress A1.1 v8 Mode A+C overlap P1-2, 2026-05-16): track
        # reference-image payload bytes so cost reporting includes the full
        # B0 egress, not only the screenshot. Pre-fix: meta emitted only
        # screenshot `image_payload_bytes`; tasks with N reference images
        # (each ~30-100KB) silently under-reported absolute cost. Mode A
        # F5 + Mode C F7 (Gemini) two-AI overlap. Backward-compat: existing
        # `image_payload_bytes` retained = screenshot bytes; new
        # `image_payload_bytes_ref` + `image_payload_bytes_total` exposed.
        _ref_payload_bytes_total = 0

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
                    # B-400: accumulate ref bytes for paper §1 cost claim.
                    _ref_payload_bytes_total += int(ref_payload.get("payload_bytes") or 0)
                    user_content.append({"type": "text", "text": ref_label})
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": ref_payload["data_url"]},
                    })
                except Exception:
                    _image_encode_error_count += 1
                    logger.warning("Failed to encode reference image %d; skipping.", idx + 1, exc_info=True)

        image_payload = None
        if image is not None:
            try:
                if max(image.size) > max_size:
                    ratio = max_size / max(image.size)
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                image_payload = self._image_to_data_url(image)
                # /stress A1.2 F1: warn loudly when the cap was exceeded so the
                # over-budget payload is auditable from logs as well as meta.
                if image_payload.get("over_cap"):
                    logger.warning(
                        "Image payload exceeded max_payload_bytes cap "
                        "(payload_bytes=%d, quality=%d, %dx%d). "
                        "Shipping over-budget JPEG to the proxy; downstream "
                        "audit should check meta['image_over_cap'].",
                        image_payload.get("payload_bytes", -1),
                        image_payload.get("quality", -1),
                        image_payload.get("width", -1),
                        image_payload.get("height", -1),
                    )
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
                _image_encode_error_count += 1
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
            # B-135 (/stress A1.1 v8 Claude F3, 2026-05-15): default 512 → 4096
            # to match B1/B2 (`qwen3vl_agent.py:538` + `gemma3vl_agent.py:240`).
            # Current 107 active configs explicit-set this (codex Mode B Q2
            # scan); the stale default was a defense-in-depth leak surface
            # for future configs missing the key. 512 silently truncates
            # typical thought+JSON envelope (~400-1500 tok) → parse_error →
            # GLM fallback fires → cross-baseline parse_fail rate asymmetric.
            "max_tokens": gen_cfg.get("max_new_tokens", 4096),
            # B-37 fix: default 0.1 → 0 for paper-grade reproducibility. T=0 is
            # greedy decoding (top-1 token deterministic given prefix). Override
            # via config only if mode-collapse signature appears in pilot.
            "temperature": gen_cfg.get("temperature", 0.0),
            # B-37 fix: explicit top_p=1.0 to prevent provider-default top_p<1
            # from introducing token-boundary non-determinism even at T=0.
            "top_p": gen_cfg.get("top_p", 1.0),
        }

        # B-37 best-effort: forward seed if provider supports OpenAI-compat seed.
        # Anthropic native protocol ignores unknown fields, so safe to include.
        # Some proxies (DashScope OpenAI-format) honor seed; others ignore.
        _seed_from_cfg = gen_cfg.get("seed")
        if _seed_from_cfg is not None:
            payload["seed"] = int(_seed_from_cfg)

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

        # B-143 (/stress A1.1 v8 Claude F7, 2026-05-15): track retry count
        # and total wait time so runner can emit `latency_ms_minus_retry`
        # for cross-baseline fair comparison. B0 retry adds 10-70s to step
        # latency (10s base × 2^attempt backoff, 3 attempts) — B1/B2 have
        # no equivalent (local inference, no network retry). Without this
        # separation, §C router latency feature is asymmetric across
        # baselines and paper §1 latency claim cannot be reported fairly.
        # Retry overhead is scaffold-level — NOT counted in agent cost.
        _retryable_codes = {429, 500, 502, 503, 504}
        _max_retries = 3
        _backoff = 10  # seconds; doubles each attempt
        _retry_count = 0
        _retry_wait_ms_total = 0.0
        # B-399 (/stress A1.1 v8 Mode A P1-1, 2026-05-16): accumulate the
        # failed-attempt elapsed (not only the sleep) into the retry-wait
        # total so `total_minus_retry` truly reflects "what the request
        # would have cost without scaffold overhead". Pre-fix: a 120s
        # timeout + 10s sleep + 30s success scored 150s fair-latency while
        # the actual no-scaffold cost was 30s — retry-frequent sites were
        # systematically inflated.
        resp = None
        for _attempt in range(_max_retries + 1):
            _attempt_start = time.time()
            try:
                resp = requests.post(
                    self.endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as net_exc:
                _attempt_elapsed_ms = (time.time() - _attempt_start) * 1000.0
                if _attempt == _max_retries:
                    raise
                wait = _backoff * (2 ** _attempt)
                logger.warning(
                    "API network error %s (attempt %d/%d, %.0fms), retrying in %ds...",
                    net_exc, _attempt + 1, _max_retries, _attempt_elapsed_ms, wait,
                )
                _retry_count += 1
                # B-399: charge failed-attempt elapsed + sleep to scaffold.
                _retry_wait_ms_total += _attempt_elapsed_ms + wait * 1000.0
                time.sleep(wait)
                continue
            _attempt_elapsed_ms = (time.time() - _attempt_start) * 1000.0
            if resp.status_code not in _retryable_codes or _attempt == _max_retries:
                # Success path (or last-attempt 5xx that we surface up):
                # this attempt's elapsed is the LEGITIMATE network cost,
                # NOT scaffold. Do not accumulate.
                break
            wait = _backoff * (2 ** _attempt)
            logger.warning(
                "API %s (attempt %d/%d, %.0fms), retrying in %ds...",
                resp.status_code, _attempt + 1, _max_retries, _attempt_elapsed_ms, wait,
            )
            _retry_count += 1
            # B-399: failed-status attempt (will be retried) → scaffold cost.
            _retry_wait_ms_total += _attempt_elapsed_ms + wait * 1000.0
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
            # B-401 (/stress A1.1 v8 Mode A P1-3, 2026-05-16): explicit None
            # for latency-split fields B0 cannot expose at the API boundary.
            # Pre-fix the keys were absent; runner default 0.0 made B0
            # latency-split rows look like "preprocessing=0, generate=0,
            # backend_infer=full_network" — visually different from B1/B2
            # but not principled (B0 internal preprocess + generate exist
            # inside the proxy/provider, just not surfaced). None is the
            # honest contract; runner now records None instead of 0 and
            # paper §3 latency-split disclosure can document the asymmetry.
            "preprocess_ms": None,
            "generate_ms": None,
            "image_payload_bytes": image_payload.get("payload_bytes") if image_payload else None,
            # B-400 (/stress A1.1 v8 Mode A+C overlap P1-2, 2026-05-16):
            # separate screenshot-only + ref-only + total payload bytes so
            # paper §1 absolute cost claim covers full B0 egress. Old field
            # `image_payload_bytes` retained for backward compat = screenshot
            # bytes (matches legacy semantic). Aggregator should prefer
            # `image_payload_bytes_total` for cross-task cost comparison.
            "image_payload_bytes_screenshot": (
                image_payload.get("payload_bytes") if image_payload else None
            ),
            "image_payload_bytes_ref": (
                _ref_payload_bytes_total if _ref_payload_bytes_total else None
            ),
            "image_payload_bytes_total": (
                (image_payload.get("payload_bytes") if image_payload else 0)
                + _ref_payload_bytes_total
            ) or None,
            "image_quality": image_payload.get("quality") if image_payload else None,
            "image_compressed": image_payload.get("compressed") if image_payload else None,
            # /stress A1.2 F1: surface the over-cap condition through meta so
            # downstream audit can detect images that exceeded the payload limit
            # (previously the encoder silently returned an over-budget payload).
            "image_over_cap": image_payload.get("over_cap", False) if image_payload else None,
            # /stress A1.1 codex Mode B C2: count of image-encode failures in
            # this step. Persisted via runner step_record.image_meta so paper-
            # grade audit can detect silent text-only episodes (B0 lenient path)
            # without grepping warning logs. 0 = clean step; >0 = N images
            # silently dropped (1 screenshot + N reference images attempted).
            "image_encode_error": _image_encode_error_count if _image_encode_error_count else None,
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
            # B-143 (/stress A1.1 v8 Claude F7, 2026-05-15): network retry
            # accounting — scaffold-level overhead NOT counted in agent
            # cost, but included in raw step wallclock latency. Runner
            # emits latency_ms_minus_retry for cross-baseline-fair latency
            # comparison (B1/B2 have no network retry equivalent).
            "network_retry_count": _retry_count if _retry_count else None,
            "network_retry_wait_ms": _retry_wait_ms_total if _retry_count else None,
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
