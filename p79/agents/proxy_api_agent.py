"""ProxyApiAgent — calls a custom proxy API (Anthropic Messages style)."""

import base64
import json
import logging
import os
import re
import time
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
# Tool-calling schema — OpenAI-format tool def for the AWS proxy (hybrid shim).
# Proxy probe 2026-05-17 (`docs/checkpoints/probes/proxy_capability_v2_223704.json`)
# confirmed: endpoint accepts OpenAI-style tools `{type:"function", function:
# {name, parameters}}` + returns top-level `body["tool_calls"][0].function.
# arguments` (NOT Anthropic-style `content[].tool_use` block, NOT OpenAI-
# style `choices[0].message.tool_calls`). `thought` added to required so
# tool args carry reasoning for `_format_history` parity with B1/B2.
# ---------------------------------------------------------------------------
_WEB_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "web_action",
        "description": (
            "Execute a web navigation action on the current page. "
            "Call this tool with your chosen action for every step."
        ),
        "parameters": {
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
            "required": ["action_type", "thought"],
        },
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

        # B-991~B-993 (/stress A1.2-followup, 2026-05-17): GLM fallback fully
        # retired. AWS proxy probe 2026-05-17 (`probes/proxy_capability_v2_
        # 223704.json`) confirmed native OpenAI-format tool_choice + logprobs
        # support, so parse-error rescue via GLM-5.1 is no longer needed.
        # Step record fields `glm_fallback_*` are preserved as schema v2
        # zombie fields (always None) to keep archive read paths valid; v3
        # migration deferred. `use_glm_fallback` config key is accepted but
        # hard-rejected if true (defense against stale yaml).
        if model_cfg.get("use_glm_fallback", False):
            raise RuntimeError(
                "use_glm_fallback=true is no longer supported (B-991 retire). "
                "AWS proxy supports native tool_choice; set use_tool_calling=true "
                "and use_glm_fallback=false."
            )

        self._system_prompts = self._get_system_prompts()

        # When tool calling is enabled, replace the "output JSON" instruction
        # with a tool-use instruction.  This is a format-only change — the
        # rules, action schema, and all task-relevant guidance are unchanged.
        if self._use_tool_calling:
            _old = "Output ONLY valid JSON. No markdown blocks, no explanations."
            _new = "Use the web_action tool for every action. Put reasoning in the thought parameter."
            for mode in self._system_prompts:
                self._system_prompts[mode] = self._system_prompts[mode].replace(_old, _new)

    # ---- GLM fallback fully retired 2026-05-17 (B-991 migration); methods
    # `_load_glm_config` + `_call_glm_extract` deleted. Step record schema
    # v2 fields `glm_fallback_*` preserved as None (zombie) for archive
    # read-path compatibility.

    # ---- B0 confidence extraction from proxy logprobs ----

    @staticmethod
    def _compute_confidence_from_proxy_logprobs(
        logprobs_content: Optional[List[Dict[str, Any]]],
    ) -> Dict[str, Optional[float]]:
        """Compute B0 confidence metrics from AWS proxy top-level logprobs.

        Proxy shape: `body["logprobs"]["content"][i] = {"token", "logprob",
        "top_logprobs": [{"token", "logprob"}, ...]}` (OpenAI-style with
        `top_logprobs=2`). Returns 6-field dict aligned with
        `_shared_vl_utils.compute_confidence` schema (B1/B2 path) for
        cross-baseline runner consumption — entropy fields are None because
        full-vocab entropy is not recoverable from top-2 truncation.

        Empirical proxy shape: see `docs/checkpoints/probes/
        proxy_capability_v2_223704.json` V4/V5.
        """
        empty = {
            "mean_logprob": None, "min_logprob": None,
            "mean_margin": None, "min_margin": None,
            "mean_entropy": None, "max_entropy": None,
        }
        if not logprobs_content or not isinstance(logprobs_content, list):
            return empty
        logprobs_list: List[float] = []
        margins_list: List[float] = []
        for entry in logprobs_content:
            if not isinstance(entry, dict):
                continue
            chosen = entry.get("logprob")
            if chosen is None:
                continue
            try:
                logprobs_list.append(float(chosen))
            except (TypeError, ValueError):
                continue
            top = entry.get("top_logprobs") or []
            if isinstance(top, list) and len(top) >= 2:
                try:
                    margin = float(top[0].get("logprob")) - float(top[1].get("logprob"))
                    margins_list.append(margin)
                except (TypeError, ValueError, AttributeError):
                    pass
        if not logprobs_list:
            return empty
        n = len(logprobs_list)
        out: Dict[str, Optional[float]] = {
            "mean_logprob": sum(logprobs_list) / n,
            "min_logprob": min(logprobs_list),
            # Entropy fields intentionally None: top-2 truncation cannot
            # recover full-vocab entropy (B-991 F1 / codex Mode B, 2026-05-17).
            "mean_entropy": None,
            "max_entropy": None,
        }
        if margins_list:
            out["mean_margin"] = sum(margins_list) / len(margins_list)
            out["min_margin"] = min(margins_list)
        else:
            out["mean_margin"] = None
            out["min_margin"] = None
        return out

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
            # B-991 (Q1=A 推荐, 2026-05-17): tool_choice="auto" — model self-
            # decides call-vs-free-text. Forced tool_choice would impose
            # grammar-constrained decoding (alternative tokens masked) →
            # mean_logprob systematically inflated vs B1/B2 free decoding →
            # §C router cross-baseline confidence feature contamination.
            # "auto" preserves logprob symmetry; if N=30 pilot emit_rate <95%
            # this falls back to "forced" + paper §3.5 disclose constrained
            # asymmetry (parking lot §8 Q1 option B).
            payload["tool_choice"] = "auto"
            # Logprobs for cross-baseline confidence feature parity with
            # B1/B2 _compute_confidence (mean/min logprob + mean/min margin;
            # entropy fields None per top-2 truncation, see
            # _compute_confidence_from_proxy_logprobs).
            payload["logprobs"] = True
            payload["top_logprobs"] = 2

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
        # B-568 (/stress A1.22 P1-10-A Claude, 2026-05-17): yaml-expose retry
        # policy hyperparams. Pre-fix `_max_retries=3 _backoff=10
        # _retryable_codes={429,500,502,503,504}` were hardcoded — reviewer
        # cannot reproduce exact retry behavior from yaml + commit SHA alone,
        # local "all hyperparams in yaml" reproducibility claim partially
        # failed. Defaults preserved so existing configs unchanged; explicit
        # override path exists via yaml (`configs/exp_v2_base.yaml:backends.
        # *.{max_retries,retry_backoff_s,retryable_codes}`).
        _retryable_codes = set(gen_cfg.get(
            "retryable_codes", [429, 500, 502, 503, 504],
        ))
        _max_retries = int(gen_cfg.get("max_retries", 3))
        _backoff = int(gen_cfg.get("retry_backoff_s", 10))  # seconds; doubles each attempt
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

        # B-991 (2026-05-17): AWS proxy hybrid shape — top-level `tool_calls`
        # field (NOT inside `content[]` Anthropic block, NOT inside
        # `choices[0].message` OpenAI). Probe v2 confirmed `body[tool_calls]
        # [0].function.{name, arguments}` shape; `arguments` is a JSON string.
        # Try this first when use_tool_calling is enabled; fall through to
        # legacy Anthropic content-block parsing or Path-2 text parse on miss.
        proxy_tool_calls = resp_json.get("tool_calls") if isinstance(resp_json, dict) else None
        if (
            self._use_tool_calling
            and isinstance(proxy_tool_calls, list)
            and proxy_tool_calls
        ):
            first_call = proxy_tool_calls[0] or {}
            fn_block = first_call.get("function") if isinstance(first_call, dict) else None
            if isinstance(fn_block, dict) and fn_block.get("name") == "web_action":
                args_str = fn_block.get("arguments") or ""
                try:
                    tool_input = json.loads(args_str) if isinstance(args_str, str) else args_str
                except json.JSONDecodeError:
                    tool_input = None
                    fail_reason = "tool_arguments_json_decode"
                    logger.warning("Proxy tool_calls.arguments JSON decode failed; fallback to text parse.")
                if isinstance(tool_input, dict):
                    if not tool_input.get("thought") and isinstance(raw_content, str) and raw_content:
                        tool_input["thought"] = raw_content.strip()[:500]
                    action, valid = validate_action(tool_input)
                    output_text = json.dumps(tool_input, ensure_ascii=False)
                    if valid:
                        _action_pt, _valid_pt, _fail_pt = parse_action_text(output_text)
                        fail_reason = _fail_pt if _valid_pt else None
                        logger.info("Proxy tool_calls parsed: %s", action.get("action_type"))
                    else:
                        fail_reason = "invalid_tool_input"
                        logger.warning("Proxy tool_calls validate_action invalid, falling back to text parse.")
                        action = None

        # Path 1: tool_use extraction (when enabled).
        if action is None and self._use_tool_calling and isinstance(raw_content, list):
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
                # B-570 (/stress A1.22 P1-12-A Claude OOB, 2026-05-17): re-route
                # Path-1 success through `parse_action_text(json.dumps(...))`
                # so the success path produces the **same** failure-mode
                # taxonomy as B1/B2 (`parse_action_text` 10+ classes vs the
                # legacy `validate_action` single "invalid_tool_input" string).
                # Pre-fix: when use_tool_calling activates, paper §3.5
                # parse_failure_reason distribution showed B0 1 class /
                # B1+B2 10+ classes — cross-baseline granularity
                # asymmetric ⟹ reviewer cannot compare parse_valid rate
                # apples-to-apples. Post-fix: dual-validate (validate_action
                # for schema integrity + parse_action_text for taxonomy),
                # the strict-schema `validate_action` decides Path-1 vs
                # Path-2 routing while `parse_action_text` produces the
                # canonical fail_reason used by §3.5 disclosure.
                # `use_tool_calling` default `false` keeps Path-2 active
                # on current paper-grade fire; this fix activates when
                # advisor sync 2026-05-14 decides Qwen official API
                # `tool_choice` channel.
                action, valid = validate_action(tool_input)
                output_text = json.dumps(tool_input, ensure_ascii=False)
                if valid:
                    # Cross-validate via parse_action_text so fail_reason
                    # taxonomy matches B1/B2 even on the success path.
                    _action_pt, _valid_pt, _fail_pt = parse_action_text(
                        output_text
                    )
                    # Path-1 trusts validate_action for routing (it has
                    # tool-schema awareness Path-2 lacks); parse_action_text
                    # output is consulted only for the canonical fail_reason
                    # taxonomy when both agree the input is parseable.
                    fail_reason = _fail_pt if _valid_pt else None
                    logger.info("Tool-use parsed: %s", action.get("action_type"))
                else:
                    fail_reason = "invalid_tool_input"
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

        # Path 3 GLM extraction fallback RETIRED 2026-05-17 (B-991). Proxy
        # native tool_calling + free-text path together cover parse-error
        # surface without cross-baseline cost-fairness violation. Step
        # record fields preserved as None for schema v2 zombie compat.
        glm_fallback_used = False
        glm_fallback_attempted = False
        glm_fallback_ms = 0.0
        glm_original_fail_reason: Optional[str] = None

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

        # B-991 (2026-05-17): extract logprob-derived confidence fields from
        # proxy top-level `logprobs.content` (when `logprobs=True` in payload).
        # 4 of 6 fields populate (mean/min logprob + mean/min margin); entropy
        # fields remain None per top-2 truncation. See
        # `_compute_confidence_from_proxy_logprobs` docstring.
        _proxy_logprobs_content = None
        if isinstance(resp_json, dict):
            _lp = resp_json.get("logprobs")
            if isinstance(_lp, dict):
                _proxy_logprobs_content = _lp.get("content")
        _confidence = self._compute_confidence_from_proxy_logprobs(_proxy_logprobs_content)

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
            # B-991 (2026-05-17): logprob-derived confidence for cross-baseline
            # §C router input parity with B1/B2 _compute_confidence. Entropy
            # fields None per top-2 truncation (full-vocab unobservable).
            **_confidence,
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
