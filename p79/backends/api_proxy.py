"""ApiProxyBackend — backend wrapper for the custom proxy API agent."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from p79.backends.base import BackendError, BackendStepContext
from p79.backends._shared_stage_prefix import build_stage_prefix

logger = logging.getLogger(__name__)

# B-425 (/stress A1.3 v9 D1, 2026-05-17): HeuristicDomBackend retired.


# B-148 (/stress A1.2 v8 gemini C4, 2026-05-16): allowlist of env-var names the
# yaml is allowed to point ``api_key_env`` at. Without this guard a malicious
# or accidentally-copied yaml could request, e.g., ``AWS_SECRET_ACCESS_KEY``;
# any subsequent verbose log of the agent config would then echo the secret
# into stdout / git-committed logs. Allowlist is intentionally narrow — extend
# only when adding a new vetted API surface.
# B-815 (/stress A1.2 cold-start P2-8-C gemini, 2026-05-17): allowlist kept
# in-module rather than promoted to a constants file. Each new entry should
# require explicit code review (security boundary), and a separate constants
# module would shift the review surface without adding extensibility — the
# existing frozenset is the simplest enforcement mechanism. Promote to
# constants only when a 4th legitimate API surface lands and the static
# enumeration becomes the bottleneck (not now).
_ALLOWED_API_KEY_ENVS = frozenset({
    "PROXY_API_KEY",     # P79 AWS API Gateway → Bedrock proxy (B0)
    "DASHSCOPE_API_KEY", # Qwen official API (pending advisor migration)
    "GLM_API_KEY",       # GLM-5.1 fallback (deprecated, marked for retire)
})


class ApiProxyBackend:
    @staticmethod
    def _validate_api_key_env(name: Any) -> str:
        # B-148: surface bad yaml early with a clear message instead of letting
        # the agent quietly read whatever env var the config requested.
        # B-816 (/stress A1.2 cold-start P2-7-B* codex OOB, 2026-05-17):
        # explicit non-string type-check with curated ValueError before set
        # membership. Pre-fix passing a list / dict (operator yaml typo) raised
        # raw ``TypeError: unhashable type: 'list'`` deep inside frozenset
        # `__contains__` — opaque to preflight / CI logs. Curated message
        # tells the operator the exact field + expected shape.
        if not isinstance(name, str):
            raise ValueError(
                f"api_key_env must be a string env-var name, got "
                f"{type(name).__name__}={name!r}. Check yaml backends.B0."
                f"api_key_env shape (B-816)."
            )
        if name not in _ALLOWED_API_KEY_ENVS:
            raise ValueError(
                f"api_key_env={name!r} is not in the allowlist "
                f"{sorted(_ALLOWED_API_KEY_ENVS)}; security boundary violation. "
                f"Add to ``_ALLOWED_API_KEY_ENVS`` in p79/backends/api_proxy.py "
                f"after explicit review if this is a legitimate new API surface."
            )
        return name

    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config
        self.mock_mode = bool(config.get("mock_mode", False))
        # B-425 (/stress A1.3 v9 D1, 2026-05-17): HeuristicDomBackend retired;
        # `dom_mode` field accepted but no longer dispatched (yaml backward-compat).
        self._agent = None

        # B-817 (/stress A1.2 cold-start P1-9-C* gemini OOB, 2026-05-17):
        # mirror local_qwen / local_gemma sampling-defense warn-loud for
        # cross-baseline parity. Pre-fix B0 wrapper silently accepted any
        # yaml temperature / top_p and forwarded them to the proxy → if a
        # future yaml drift set `temperature: 0.5` on B0 only, the asymmetry
        # would land silently (B1/B2 warn, B0 silent honors). Currently all
        # 41 paper-grade yamls are 0.0/1.0; this is preventive defense for
        # cross-baseline drift gate. paper_grade=True escalates to raise so
        # paper-grade run cannot fire with B0 honoring a non-greedy temp.
        _temp = config.get("temperature", 0.0)
        _topp = config.get("top_p", 1.0)
        if _temp != 0.0:
            msg = (
                f"ApiProxyBackend yaml temperature={_temp} ≠ 0.0 — cross-"
                f"baseline drift: B1/B2 hardcode greedy and would IGNORE this "
                f"override, but B0 (api_proxy) HONORS it. Set 0.0 unless you "
                f"explicitly want asymmetric sampling. (B-817)"
            )
            if bool(config.get("paper_grade", False)):
                raise BackendError(msg + " paper_grade=True hard-fails this.")
            logger.warning(msg)
        if _topp != 1.0:
            msg = (
                f"ApiProxyBackend yaml top_p={_topp} ≠ 1.0 — same cross-"
                f"baseline drift as temperature (B-817)."
            )
            if bool(config.get("paper_grade", False)):
                raise BackendError(msg + " paper_grade=True hard-fails this.")
            logger.warning(msg)

        # B-818 (/stress A1.2 cold-start P2-4-A Claude, 2026-05-17): validate
        # api_key_env BEFORE partial-state agent_cfg construction so a
        # malformed yaml surfaces immediately instead of after a half-built
        # cfg. Pre-fix the validate ran inline at agent_cfg["model"]
        # ["api_key_env"] = self._validate_api_key_env(...) — exception
        # raised after several other cfg fields were already constructed
        # (cheap but defense-in-depth weaker than fail-first).
        _api_key_env_validated = self._validate_api_key_env(
            config.get("api_key_env", "PROXY_API_KEY"),
        )

        if not self.mock_mode:
            from p79.agents.proxy_api_agent import ProxyApiAgent

            agent_cfg = {
                "model": {
                    "api_name": config.get("api_name", config.get("name", "qwen.qwen3-vl-235b-a22b")),
                    "base_url": config.get("base_url"),
                    # B-147 (/stress A1.2 v8 Claude A1, 2026-05-16): default
                    # aligned 512 → 4096 to match local_qwen / local_gemma
                    # wrappers + agent layer's own default (B-135). Previously
                    # the wrapper masked the agent's 4096 with stale 512 — yaml
                    # had to set explicitly to avoid silent truncation of the
                    # ~400-1500 tok thought + JSON envelope. Defense-in-depth
                    # alignment so future config refactor cannot regress this.
                    "max_new_tokens": config.get("max_new_tokens", 4096),
                    # B-37 fix: defaults 0.1→0 (greedy), 0.9→1.0 (no nucleus pruning).
                    # yaml configs may still override but new default is reproducibility-first.
                    "temperature": config.get("temperature", 0.0),
                    "top_p": config.get("top_p", 1.0),
                    # B-37 fix: forward seed from runner-injected backend cfg.
                    "seed": config.get("seed"),
                    "timeout": config.get("timeout", 120),
                    # API format: "anthropic" (proxy) or "openai" (DashScope)
                    "api_format": config.get("api_format", "anthropic"),
                    # B-148 + B-818 (/stress A1.2 cold-start P2-4-A): validated
                    # at __init__ entry (fail-first), reused here so the
                    # agent_cfg construction is partial-state-safe.
                    "api_key_env": _api_key_env_validated,
                    # Plan A/B: tool_use + GLM fallback (§67)
                    "use_tool_calling": config.get("use_tool_calling", False),
                    # B-1985 (2026-08-20). These three were read by the agent and
                    # NEVER forwarded here, so a yaml that set them was obeyed by
                    # nobody. `agent_cfg["model"]` is an explicit allowlist, and an
                    # allowlist silently drops what it does not name — the same shape
                    # B-340 fixed for `paper_grade` ("agent_cfg was a strict subset
                    # that dropped top-level config keys ... B-340 raise inert"), back
                    # again for three more keys. What it cost: B5's
                    # `structured_output: "response_format"` never reached the agent,
                    # which defaulted to "tool_calls", attached `tools` to a model that
                    # rejects them, and every B5 episode died at step 0 with HTTP 400.
                    # The B-1990 guard written to catch exactly that misconfiguration
                    # was disarmed by the same dropped key: it fires only when
                    # `_structured_output == "response_format"`, which could never be
                    # true. tests/test_b1985_model_cfg_forwarding.py now derives both
                    # sides from source so the next omission fails a test, not a fire.
                    "structured_output": config.get("structured_output", "tool_calls"),
                    "logprobs_unavailable": config.get("logprobs_unavailable", False),
                    "image_format": config.get("image_format", "auto"),
                    "use_glm_fallback": config.get("use_glm_fallback", False),
                    "glm_config": config.get("glm_config", ".auth/glm"),
                    # B-568 (/stress A1.22 P1-10-A Claude, 2026-05-17): forward
                    # yaml-controlled retry policy hyperparams. Pre-fix these
                    # were hardcoded inside `proxy_api_agent.py:585-587`
                    # (`_max_retries=3 _backoff=10 _retryable_codes`). Yaml
                    # `configs/exp_v2_base.yaml:backends.B0.{max_retries,
                    # retry_backoff_s,retryable_codes}` now overridable for
                    # reproducibility checks (reviewer running with a faster
                    # proxy can lower `max_retries=0` to match local latency
                    # signature).
                    "max_retries": config.get("max_retries", 3),
                    "retry_backoff_s": config.get("retry_backoff_s", 10),
                    # B-1880 (reddit chain abort #3, 2026-06-19): cap exponential
                    # backoff (None = uncapped, back-compat). Lets max_retries be
                    # raised to survive multi-minute proxy 503 outages without a
                    # single 5min+ sleep. See proxy_api_agent.py retry loop RCA.
                    "retry_backoff_max_s": config.get("retry_backoff_max_s", None),
                    "retryable_codes": config.get(
                        "retryable_codes", [429, 500, 502, 503, 504],
                    ),
                },
                "agent": {
                    "image_max_size": config.get("image_max_size", 1024),
                    # B-84: max_obs_chars removed — the agent no longer truncates
                    # obs_text (viewport filter is the real bound).
                    "max_image_payload_bytes": config.get("max_image_payload_bytes", 5 * 1024 * 1024),
                },
                # B-340 (/stress A1.9 Mode C F4 defense-in-depth, 2026-05-16):
                # forward paper_grade flag to agent layer so the GLM-fallback
                # hard-block in `ProxyApiAgent.__init__` can fire. Pre-fix
                # agent_cfg was a strict subset that dropped top-level config
                # keys; paper_grade flag would always read False in agent →
                # B-340 raise inert.
                "paper_grade": bool(config.get("paper_grade", False)),
            }
            self._agent = ProxyApiAgent(agent_cfg)

    def step(self, instruction: str, obs: Any, context: BackendStepContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        # B-425 (/stress A1.3 v9 D1, 2026-05-17): heuristic dispatch retired.
        if self.mock_mode:
            # B-149 (/stress A1.2 v8 Claude A3 + gemini C5, 2026-05-16):
            # mock action aligned with local_qwen / local_gemma / MockBackend
            # mock_mode (all emit scroll [0, 0.8]). Previously this returned
            # click element_id=1, breaking the "Mock Parity" invariant — tests
            # asserting that swapping mocked backends preserves action shape
            # silently failed only on api_proxy. factory.py:15-18 comment
            # marks scroll as canonical.
            # B-808 (/stress A1.2 cold-start P2-2-AC): removed dead
            # coordinate_type field (see local_qwen.py / local_gemma.py
            # / factory.MockBackend siblings).
            action = {
                "action_type": "scroll",
                "delta": [0, 0.8],
                "thought": "Mock API proxy backend action.",
            }
            return action, {
                "raw_output": action,
                "valid": True,
                "failure_reason": None,
                "input_tokens": 1,
                "output_tokens": 1,
                "model_calls": 1,
                # B-415 (/stress A1.2 v8 Mode A P2-3, 2026-05-16): canonical
                # `mock_<backend_id>` naming (was: "api_proxy_mock"). Aligned
                # with factory.MockBackend + local_qwen + local_gemma so
                # cross-baseline mock invariance tests grep one pattern.
                "backend_type": f"mock_{self.backend_id}",
            }

        assert self._agent is not None

        # B-812 (/stress A1.2 cold-start P1-1-A*): shared single-source
        # stage_prefix from _shared_stage_prefix.build_stage_prefix.
        prompt = f"{build_stage_prefix(context.stage, context.planner_sub_goal)}{instruction}"
        start = time.time()
        action, meta = self._agent.step(
            prompt, obs,
            history=context.history,
            observation_mode=context.observation_mode,
            reference_images=context.reference_images,
        )
        infer_ms = (time.time() - start) * 1000.0

        meta = dict(meta)
        # B-813 (/stress A1.2 cold-start P0-2-B*): preserve-None defense
        # for cost-aware paper §1 hero telemetry contract.
        if meta.get("model_calls") is None:
            meta["model_calls"] = 1
        if meta.get("backend_type") is None:
            meta["backend_type"] = "api_proxy"
        if bool(self.config.get("paper_grade", False)):
            for _k in ("input_tokens", "output_tokens"):
                if meta.get(_k) is None:
                    raise BackendError(
                        f"api_proxy step returned meta[{_k!r}]=None under "
                        f"paper_grade=True — cost telemetry contract violation. "
                        f"Likely proxy `usage` block missing / provider drift. "
                        f"(B-813)"
                    )
        meta["infer_ms"] = infer_ms
        return action, meta
