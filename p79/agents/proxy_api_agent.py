"""ProxyApiAgent — calls a custom proxy API (Anthropic Messages style)."""

import base64
import fcntl
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from PIL import Image

from p79.backends.action_utils import parse_action_text, validate_action_detailed
from p79.backends.image_utils import DEFAULT_MAX_IMAGE_PAYLOAD_BYTES, encode_image_data_url


# B-1668 (/stress A2.11 P1-2-B*C 2026-05-18, user Q8=A): cross-process file lock
# for B0 AWS proxy. Defends against simultaneous fire from cls + red chains
# (default sequential post-P0-5 B-1663, but accidents possible) hammering same
# proxy endpoint + API key → sync exponential backoff on 429/5xx tail latency
# amplification. Non-blocking acquire + sleep-retry up to PROXY_LOCK_MAX_WAIT
# secs; on timeout, proceed without lock + log warning (degrade-not-block).
#
# B-1700 (/stress A2.12 P0-1-B* OOB codex unique, 2026-05-18, user Q2=A):
# `paper_grade=True` flips contract to FAIL-CLOSED — timeout → raise
# RuntimeError instead of degrade-to-unlocked. Pre-fix the warning-then-
# proceed semantics defeated the lock's purpose at the very moment serialization
# matters (contention window where another holder ran >60s with retries/backoff).
# Dev mode keeps degrade-not-block for iteration UX.
@contextmanager
def _proxy_global_lock(api_key: str, max_wait_secs: int = 60, paper_grade: bool = False):
    """File-based semaphore on /tmp/p79_proxy_<hash>.lock — serializes B0 proxy
    requests across all runners on same host. Yields True if lock acquired,
    False on timeout in dev mode. In paper_grade mode raises RuntimeError on
    timeout (B-1700 fail-closed contract — caller cannot proceed unsynchronized
    under paper-grade contention)."""
    lock_path = Path(tempfile.gettempdir()) / f"p79_proxy_{hashlib.sha1(api_key.encode()).hexdigest()[:8]}.lock"
    lock_fd = None
    try:
        lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
        _start = time.time()
        acquired = False
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except OSError:
                if time.time() - _start >= max_wait_secs:
                    if paper_grade:
                        try:
                            os.close(lock_fd)
                        except OSError:
                            pass
                        lock_fd = None
                        raise RuntimeError(
                            f"B-1700 (paper_grade): proxy global lock {lock_path} "
                            f"acquire timeout {max_wait_secs}s — refusing to proceed "
                            f"unsynchronized (cls+red parallel B0 contention would "
                            f"pollute latency canonical). Check for stale runner / "
                            f"lock-holder PID / set PHASE1A_PARALLEL=0 for sequential."
                        )
                    logging.getLogger(__name__).warning(
                        "B-1668: proxy global lock %s acquire timeout %ds — proceeding without (dev mode)",
                        lock_path, max_wait_secs,
                    )
                    try:
                        os.close(lock_fd)
                    except OSError:
                        pass
                    lock_fd = None
                    break
                time.sleep(0.5)
        yield acquired
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
            except OSError:
                pass

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
            "Execute a web navigation action on the current page. Call this tool "
            "with your chosen action for every step. RULES (match the text protocol "
            "shared with the other baselines): for click / type / hover / "
            "select_option you MUST provide `element_id` (the numeric [N] id from "
            "the Accessibility Tree); ALWAYS prefer `element_id` over `coordinate` "
            "(coordinate is a last resort only). Use `type` (not `click`) to enter "
            "text into an input field; `click` is for buttons/links/navigation. "
            "`url` is ONLY for the goto action. Do NOT mix fields from different "
            "actions (e.g. never put `url` on a type action)."
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
                        # Protocol Reset #5 (action-set restore, 2026-05-20):
                        # upstream-compatible id-based action space. hover/press/
                        # new_tab/close_tab execute via the wrapper escape-hatch;
                        # goto via an explicit branch with a VWA-domain whitelist.
                        "hover", "press", "new_tab", "close_tab", "goto",
                    ],
                    "description": "The type of action to perform.",
                },
                "element_id": {
                    "type": "integer",
                    "description": (
                        "Numeric [N] element ID from the Accessibility Tree / SOM "
                        "marks. REQUIRED for click / type / hover / select_option — "
                        "ALWAYS specify it to target the correct element; do NOT "
                        "omit it and do NOT guess coordinates for these actions."
                    ),
                },
                "coordinate": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 2,
                    "maxItems": 2,
                    "description": (
                        # B-1860: Qwen 0-1000 contract (was "Normalized 0.0-1.0").
                        "[x, y] coordinates in a 0-1000 system ([0,0]=top-left, "
                        "[1000,1000]=bottom-right). LAST RESORT only — use ONLY "
                        "when the target has no element_id; always prefer element_id."
                    ),
                },
                "text": {
                    "type": "string",
                    "description": (
                        "Text to type (for the type action; also requires "
                        "element_id). Append \\n to submit."
                    ),
                },
                "scroll_direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Scroll direction: 'down' to reveal content below, 'up' to reveal content above.",
                },
                "option_label": {
                    "type": "string",
                    "description": (
                        "Visible option text for select_option (must match exactly). "
                        "Use select_option for native <select> comboboxes and "
                        "[DROPDOWN OPTIONS]-annotated triggers — clicking a combobox "
                        "does NOT open the menu, so use select_option to set the value."
                    ),
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
                "key": {
                    "type": "string",
                    "description": "Key combination for press action (e.g. 'Ctrl+Enter').",
                },
                "url": {
                    "type": "string",
                    "description": (
                        "Target URL — ONLY for the goto action. Must be a URL on the "
                        "current task's own websites (a relative path like '/page' "
                        "also works); off-site URLs are ignored. Do NOT include url "
                        "on any non-goto action (e.g. type/click)."
                    ),
                },
            },
            "required": ["action_type", "thought"],
            # B-1794 REAL fix (2026-05-21): per-action conditional required fields,
            # MIRRORING validate_action_detailed (the shared semantic gate that
            # B1/B2 prose-JSON also pass through). Root cause of B0's search/type
            # element_id omission is NOT the prose (B-1794 description-only attempt
            # failed) but `tool_choice="required"` forcing a MINIMAL tool call that
            # satisfies only the required-array (action_type, thought), dropping
            # the OPTIONAL element_id (model had a competing url prior). Confirmed
            # on the real proxy: tc="auto" emits element_id, tc="required" omits
            # it; conditional required restores it (probe 6/6 valid). Cross-baseline
            # CONSISTENCY requires this schema == validator exactly: each clause
            # below matches a validate_action_detailed rule (NOT stricter — e.g.
            # `type` does NOT require `text`, mirroring `type(eid,no-text)=valid`;
            # being stricter would force B0 alone beyond what B1/B2 face).
            # P1-10 (/stress 2026-05-21 Claude Mode A): the AWS/Bedrock proxy does
            # NOT hard-enforce this JSON schema on output (see B-1101 note in
            # action_utils.py — the model self-decides emission format); the schema
            # acts as a SOFT constraint that conditions the model to include the
            # grounding field, and `validate_action_detailed` is the HARD runtime
            # gate (post-hoc → invalid steps surface in the §3.5 taxonomy). The
            # probe (6/6 valid) + 30-step dom smoke (0 invalid) confirm the model
            # honors it empirically; they are NOT a proof of proxy-side enforcement,
            # so the validator gate remains the authority. click/type/hover use
            # anyOf(element_id|coordinate) so VISION mode (no AXTree → coordinate
            # only) stays valid; AXTree modes pick element_id.
            "allOf": [
                {
                    "if": {"properties": {"action_type": {
                        "enum": ["click", "type", "hover"]}}},
                    "then": {"anyOf": [{"required": ["element_id"]},
                                       {"required": ["coordinate"]}]},
                },
                {
                    "if": {"properties": {"action_type": {"const": "select_option"}}},
                    # B-1796 (P0-1, /stress 2026-05-21 Claude Mode A OOB): mirror
                    # validate_action_detailed exactly — select_option needs
                    # (element_id OR coordinate) AND one option specifier. Pre-fix
                    # `required: ["element_id"]` was STRICTER than the validator
                    # (action_utils.py:502 accepts a valid coordinate when element_id
                    # is absent), so under tool_choice="required" a VISION-mode B0
                    # (no AXTree → coordinate-only) could not emit a select_option
                    # that B1/B2 free-gen can — a cross-baseline asymmetry the B-1794
                    # schema≡validator contract is supposed to eliminate. allOf keeps
                    # the two independent requirements (grounding + option) both live.
                    "then": {"allOf": [
                        {"anyOf": [{"required": ["element_id"]},
                                   {"required": ["coordinate"]}]},
                        {"anyOf": [{"required": ["option_label"]},
                                   {"required": ["option_value"]},
                                   {"required": ["option_index"]}]},
                    ]},
                },
                {
                    "if": {"properties": {"action_type": {"const": "scroll"}}},
                    "then": {"required": ["scroll_direction"]},
                },
                {
                    "if": {"properties": {"action_type": {"const": "tab_focus"}}},
                    "then": {"required": ["page_number"]},
                },
                {
                    "if": {"properties": {"action_type": {"const": "press"}}},
                    "then": {"required": ["key"]},
                },
                {
                    "if": {"properties": {"action_type": {"const": "goto"}}},
                    "then": {"required": ["url"]},
                },
            ],
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
        # B-1102 (/stress A2.3b P1-4-A, 2026-05-18): read top-level
        # `paper_grade` flag so init-time + step-time invariant guards can
        # fail-loud on paper-grade contract violations (vs silently
        # degrading dev runs). See B-340 paper-grade hard-block precedent.
        self._paper_grade = bool(config.get("paper_grade", False))

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

        # B-1102 (/stress A2.3b P1-4-A, 2026-05-18): paper-grade B0 MUST
        # use native tool_calling. Post-B-991 GLM rescue is physically
        # deleted, so a paper-grade run with `use_tool_calling=false`
        # falls through to Path-2 text-parse-only mode (~30% parse_error
        # historically) with no rescue → cross-baseline silent
        # contamination. Fail-loud at init so misconfigured yamls (e.g.
        # partial override skipping base merge) surface immediately, NOT
        # mid-fire on cell N at episode K.
        if self._paper_grade and not self._use_tool_calling:
            raise RuntimeError(
                "paper-grade B0 requires use_tool_calling=true post-B-991 "
                "(GLM rescue physically deleted; Path-2 text-parse-only "
                "mode would cause silent cross-baseline contamination). "
                "Set backends.api_strong.use_tool_calling: true OR clear "
                "the paper_grade flag for dev runs."
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
                # B-1105 (/stress A2.3b P0-5-A OOB, 2026-05-18): assert
                # top[0] is the chosen token before computing margin. At
                # T=0 greedy decoding (current paper-grade preregistration
                # lock) the chosen token IS argmax = top[0] by
                # construction. BUT (a) OpenAI top_logprobs spec does NOT
                # mandate top[0]==chosen ordering — provider drift could
                # return chosen at any list position; (b) future T>0
                # sampling ablation (preregistration §7 future scope)
                # would break the assumption silently. Without this
                # guard, `top[0].logprob - top[1].logprob` computes "two
                # arbitrary non-chosen alternatives" → margin signal
                # silently corrupted → §C router cross-mode confidence
                # feature meaningless. Skip margin (NOT logprob) on
                # mismatch + warn once. Symmetric assumption holds in B1/
                # B2 `_shared_vl_utils.compute_confidence:393` (top2 from
                # torch.topk(log_softmax) is sorted by construction, and
                # do_sample=False means chosen=argmax=top2[0]).
                chosen_token = entry.get("token")
                if chosen_token is not None and top[0].get("token") != chosen_token:
                    logger.warning(
                        "Proxy logprob top[0]=%r != chosen=%r; skipping margin "
                        "for this token (B-1105). Provider drift OR T>0 ablation?",
                        top[0].get("token"), chosen_token,
                    )
                    continue
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
            # tool_choice="required" (NOT "auto"). B-991 (2026-05-17) chose
            # "auto" to preserve logprob symmetry, but Fire-6 RCA (2026-05-20)
            # found "auto" emits tool_calls 0% under the real dom system prompt:
            # its "Output ONLY valid JSON. No markdown blocks, no explanations."
            # (B-451 byte-identical contract, required by B1/B2 text-parse)
            # suppresses tool calling → 31% parse_error → injected-wait spirals
            # (masked pre-B-991 by GLM rescue, exposed when B-991 retired it).
            # probe (2026-05-20): emit 0%→100% with "required" (OpenAI string;
            # {type:function} object proxy-ignored, Anthropic {type:tool}/{any}
            # → HTTP 400), valid≈95%, logprobs intact. No system-prompt change
            # needed (sidesteps the byte-identical contract). Logprob-symmetry
            # worry deferred to Pass-2 router (§C): Pass-1 baseline does not
            # consume logprobs; paper §3.5 discloses B0 grammar-constrained vs
            # B1/B2 free decoding.
            payload["tool_choice"] = "required"
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
        # B-1668 (/stress A2.11 P1-2-B*C 2026-05-18, user Q8=A): file-based
        # global semaphore around B0 proxy retry loop. Other process holding
        # lock waits up to 60s; on timeout proceed without serialization
        # (degrade-not-block in dev / raise in paper-grade per B-1700).
        # B-1700 (/stress A2.12 P0-1-B* OOB codex, 2026-05-18, user Q2=A):
        # paper-grade mode flips contract to FAIL-CLOSED (raise instead of
        # degrade) so cls+red parallel B0 contention can never produce
        # unsynchronized proxy traffic under paper-grade. Lock released in
        # finally after loop exit (success / max retries / exception / raise).
        _proxy_lock_ctx = _proxy_global_lock(self.api_key, paper_grade=self._paper_grade)
        _proxy_lock_ctx.__enter__()
        try:
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
        finally:
            _proxy_lock_ctx.__exit__(None, None, None)
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
        # B-1588 (/stress A1.24 post-fire P1-8-B codex Mode B F6 OOB, 2026-05-18):
        # per-step tool-call emission audit. `tool_calling` bool config alone
        # doesn't tell us whether the proxy actually emitted a `tool_calls`
        # block — text-parse fallback can masquerade as structured emission if
        # the free text happens to parse valid JSON. Paper §3 B0 substrate
        # claim ("native tool calling via AWS proxy hybrid shim per B-991")
        # requires per-step evidence. Three tracker fields, surfaced via meta:
        #   tool_call_emitted: bool — proxy actually returned tool_calls list?
        #   tool_call_parse_path: str — "tool_calls" / "text_json" / "fallback_regex"
        #   tool_call_fallback_reason: Optional[str] — why fallback if it happened
        _tool_call_emitted: bool = False
        _tool_call_parse_path: str = "text_json"  # default; overridden if tool_calls path taken
        _tool_call_fallback_reason: Optional[str] = None
        # P1-3-B (/stress GRL audit 2026-05-20, user Q4=A): explicit action-
        # provenance trackers. text_fallback_used = Path-2 text parse ran;
        # tool_call_valid = the emitted tool_call validated (None if no emit);
        # action_source ∈ {tool_call, text_json, fallback, invalid} surfaced via
        # meta so paper §3.5.1 cross-baseline action-channel disclosure is
        # reproducible from disk (B0 native tool_calls vs B1/B2 text JSON).
        _text_fallback_used: bool = False

        # B-991 (2026-05-17): AWS proxy hybrid shape — top-level `tool_calls`
        # field (NOT inside `content[]` Anthropic block, NOT inside
        # `choices[0].message` OpenAI). Probe v2 confirmed `body[tool_calls]
        # [0].function.{name, arguments}` shape; `arguments` is a JSON string.
        # Try this first when use_tool_calling is enabled; fall through to
        # legacy Anthropic content-block parsing or Path-2 text parse on miss.
        proxy_tool_calls = resp_json.get("tool_calls") if isinstance(resp_json, dict) else None
        # B-1588: capture emission intent regardless of parse-success outcome.
        if isinstance(proxy_tool_calls, list) and proxy_tool_calls:
            _tool_call_emitted = True
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
                    # B-1588: emission JSON malformed → text-parse fallback.
                    _tool_call_fallback_reason = "tool_arguments_json_decode"
                    logger.warning("Proxy tool_calls.arguments JSON decode failed; fallback to text parse.")
                if isinstance(tool_input, dict):
                    if not tool_input.get("thought") and isinstance(raw_content, str) and raw_content:
                        tool_input["thought"] = raw_content.strip()[:500]
                    action, valid, _tool_detail_reason = validate_action_detailed(tool_input)
                    output_text = json.dumps(tool_input, ensure_ascii=False)
                    if valid:
                        _action_pt, _valid_pt, _fail_pt = parse_action_text(output_text)
                        fail_reason = _fail_pt if _valid_pt else None
                        # B-1588: native tool-call path succeeded — record provenance.
                        _tool_call_parse_path = "tool_calls"
                        logger.info("Proxy tool_calls parsed: %s", action.get("action_type"))
                    else:
                        # Smoke 2026-05-21: capture the SPECIFIC validate reason
                        # (invalid_element_id / invalid_coord / invalid_action_type /
                        # invalid_select_option / invalid_schema_dict / ...) via
                        # validate_action_detailed instead of the generic
                        # "invalid_tool_input" — so B0 tool-call failures are
                        # classifiable from disk (Fire-6 §3.5 B0 failure analysis +
                        # parse_error_rate disclosure; pre-fix all invalids collapsed
                        # to one label + mapped to error_category=unknown_failure).
                        # Raw emitted args logged (not persisted as a schema field)
                        # for forensics.
                        fail_reason = _tool_detail_reason or "invalid_tool_input"
                        # B-1588: emission attempted but validator rejected → fallback.
                        _tool_call_fallback_reason = fail_reason
                        logger.warning(
                            "Proxy tool_calls validate_action_detailed invalid "
                            "(reason=%s); emitted_args=%s",
                            fail_reason, output_text[:500],
                        )
                        action = None

        # B-1110 (/stress A2.3b P1-5-A, 2026-05-18): legacy Path-1 Anthropic
        # `content[].tool_use` block parser DELETED. AWS proxy probe v2
        # (`docs/checkpoints/probes/proxy_capability_v2_223704.json`)
        # empirically confirmed: proxy returns `body["content"]` as STRING
        # (not list-of-blocks Anthropic style) — list-content path was
        # 50 LOC dead code never fired in production. The current active
        # tool-call parser is the top-level `body["tool_calls"]` branch
        # at L613 (B-991 native AWS proxy hybrid shim). If a future
        # provider drift returns list-shaped content with embedded
        # `tool_use` blocks, the contract assertion below will fail-loud
        # — signal to re-introduce the parser, NOT silently fall through
        # to Path-2 text parse that would lose the structured tool_use
        # input.
        assert not isinstance(raw_content, list), (
            "proxy returned list-shaped content; AWS proxy contract expects "
            "string (B-991 + B-1110). Re-introduce Anthropic content-block "
            "parser if a non-AWS proxy provider is being used."
        )

        # P1-3-B (/stress GRL audit 2026-05-20, user Q4=A): paper-grade B0 must
        # NOT let an EMITTED-but-invalid tool_call silently fall through to a
        # text-parsed DIFFERENT action. That changes B0 action provenance + masks
        # the tool-call failure rate — cross-baseline asymmetry (B0 = native
        # tool_calls per B-991; B1/B2 = text JSON), so silent recovery would make
        # B0's effective parse-success look better than it is. Record an emitted-
        # but-invalid tool_call as a protocol failure (invalid no-op, valid=False)
        # and SKIP Path-2; the runner re-validates (B-134) + records
        # parse_valid=False (failed step). Dev / non-paper-grade keeps the lenient
        # text fallback below. Covers BOTH invalid emission paths above
        # (tool_arguments_json_decode → tool_input=None; invalid_tool_input →
        # validate_action rejected → action=None).
        _tool_call_emitted_invalid = (
            self._use_tool_calling
            and _tool_call_emitted
            and action is None
            and _tool_call_fallback_reason is not None
        )
        if self._paper_grade and _tool_call_emitted_invalid:
            _thought = raw_content.strip()[:500] if isinstance(raw_content, str) else ""
            action = {"action_type": "none", "thought": _thought}
            valid = False
            _tool_call_parse_path = "invalid_tool_call_protocol_failure"
            if fail_reason is None:
                fail_reason = _tool_call_fallback_reason or "invalid_tool_input"
            logger.warning(
                "Proxy paper_grade: tool_call emitted but invalid (%s) — recorded "
                "as protocol failure (invalid no-op), NO text-parse fallback "
                "(P1-3-B).", fail_reason,
            )

        # Path 2: text parsing fallback (original logic; dev / non-paper-grade,
        # or paper-grade with NO tool_call emitted at all).
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
            _text_fallback_used = True

        # Path 3 GLM extraction fallback RETIRED 2026-05-17 (B-991). Proxy
        # native tool_calling + free-text path together cover parse-error
        # surface without cross-baseline cost-fairness violation. Step
        # record fields preserved as None for schema v2 zombie compat.
        # B-1111 (/stress A2.3b P1-6-A, 2026-05-18): uniform-None zombie
        # serialization. Pre-fix `glm_fallback_used = False` (bool) +
        # `glm_fallback_attempted = None` (None) inconsistent — archive
        # aggregator + paper §3.5 disclosure table mixed `False=never tried`
        # with `attempted=None=never relevant`. Both semantically equal
        # "GLM module non-existent post-B-991"; emit None uniformly so
        # downstream readers cannot confuse "tried-but-failed" with
        # "never-relevant". Schema v3 will drop the keys entirely (paper-2
        # prep); v2 zombie kept for archive read-path back-compat.
        glm_fallback_used = None
        glm_fallback_attempted = None
        glm_fallback_ms = None
        glm_original_fail_reason: Optional[str] = None

        # /stress A2.4b Chunk α (2026-05-18): B0 scroll_direction→delta conversion
        # deleted for cross-baseline JSONL symmetry. Pre-removal asymmetry: B0
        # step_record.action recorded `delta:[0, ±0.8]`, B1/B2 recorded raw `delta`
        # canonicalized to `scroll_direction` at validator (action_utils.py:575-591).
        # vwa_wrapper L462-493 (B-512 2026-05-17) collapses both forms to
        # `create_scroll_action(direction=...)` and VWA upstream `execute_scroll`
        # (external/visualwebarena/browser_env/actions.py:936) executes at fixed
        # `±window.innerHeight` magnitude — magnitude info in delta was always
        # discarded at execution. Removing conversion lets B0 RAW emit
        # `scroll_direction` survive to step JSONL, identical schema across baselines.

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

        # B-1103 (/stress A2.3b P0-4-B* codex OOB, 2026-05-18): fail-loud
        # when paper-grade B0 advertises logprob-derived confidence but
        # proxy silently omits it. Provider drift / proxy quota mode /
        # response shape change can erase `body.logprobs.content` without
        # failing the HTTP request — SR/cost data finishes clean but §C
        # confidence analysis silently has missing/incomplete B0 coverage.
        # Without this guard, Phase 1a substrate ships and reviewer asks
        # "why advertised at launch but not invariant?" — paper-grade
        # contract violation surfaces only at OSF audit.
        # Non-paper-grade dev runs: persist `confidence_error` for audit
        # but proceed (preserves existing F3 behavior).
        _confidence_error: Optional[str] = None
        if self._use_tool_calling and not _proxy_logprobs_content:
            if self._paper_grade:
                raise RuntimeError(
                    "B0 paper-grade run requires body.logprobs.content "
                    "(use_tool_calling=True advertises logprob-derived "
                    "confidence). Proxy response missing logprobs — "
                    "check provider drift / quota mode / response shape "
                    "change. B-1103 /stress A2.3b P0-4-B*."
                )
            _confidence_error = "missing_proxy_logprobs"

        _confidence = self._compute_confidence_from_proxy_logprobs(_proxy_logprobs_content)

        # P1-3-B (/stress GRL audit 2026-05-20, user Q4=A): derive explicit
        # action provenance for paper §3.5.1 cross-baseline disclosure.
        if _tool_call_parse_path == "tool_calls":
            _action_source = "tool_call"
        elif _tool_call_parse_path == "invalid_tool_call_protocol_failure":
            _action_source = "invalid"
        elif _text_fallback_used and _tool_call_emitted:
            _action_source = "fallback"  # dev: emitted-but-invalid → text recovered
        else:
            _action_source = "text_json"
        # tool_call_valid: True if a native tool_call validated, False if emitted-
        # but-invalid, None if no tool_call was emitted (text-only path).
        if not _tool_call_emitted:
            _tool_call_valid: Optional[bool] = None
        else:
            _tool_call_valid = (_tool_call_parse_path == "tool_calls")

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
            # B-1588 (/stress A1.24 post-fire P1-8-B codex Mode B F6 OOB, 2026-05-18):
            # per-step tool-call emission audit fields. `tool_calling` (bool config)
            # alone cannot prove the proxy emitted native tool_calls — text-parse
            # fallback can masquerade as structured emission. These 3 fields
            # surface the actual parse path per step so paper §3 substrate claim
            # ("B-991 native tool calling via AWS proxy hybrid shim") gains
            # per-step empirical backing. Downstream aggregator can compute
            # `tool_call_emit_rate = mean(tool_call_emitted)` per condition and
            # paper-grade gate at e.g. ≥0.95 to admit B0 evidence layer.
            "tool_call_emitted": _tool_call_emitted,
            "tool_call_parse_path": _tool_call_parse_path,
            "tool_call_fallback_reason": _tool_call_fallback_reason,
            # P1-3-B (/stress GRL audit 2026-05-20, user Q4=A): explicit action
            # provenance. action_source ∈ {tool_call, text_json, fallback,
            # invalid}; tool_call_valid (None if no emit); text_fallback_used.
            # Under paper_grade an emitted-but-invalid tool_call → action_source
            # "invalid" + valid=False + NO text fallback (no silent action swap).
            "action_source": _action_source,
            "tool_call_valid": _tool_call_valid,
            "text_fallback_used": _text_fallback_used,
            # B-1103 (/stress A2.3b P0-4-B*, 2026-05-18): non-paper-grade
            # dev-run signal for missing proxy logprobs. Paper-grade raises
            # at extraction point; dev runs persist this for downstream
            # audit (e.g. analyze_confidence_calibration coverage report).
            "confidence_error": _confidence_error,
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
