"""ApiProxyBackend — backend wrapper for the custom proxy API agent."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from p79.backends.base import BackendStepContext

# B-425 (/stress A1.3 v9 D1, 2026-05-17): HeuristicDomBackend retired.


# B-148 (/stress A1.2 v8 gemini C4, 2026-05-16): allowlist of env-var names the
# yaml is allowed to point ``api_key_env`` at. Without this guard a malicious
# or accidentally-copied yaml could request, e.g., ``AWS_SECRET_ACCESS_KEY``;
# any subsequent verbose log of the agent config would then echo the secret
# into stdout / git-committed logs. Allowlist is intentionally narrow — extend
# only when adding a new vetted API surface.
_ALLOWED_API_KEY_ENVS = frozenset({
    "PROXY_API_KEY",     # P79 AWS API Gateway → Bedrock proxy (B0)
    "DASHSCOPE_API_KEY", # Qwen official API (pending advisor migration)
    "GLM_API_KEY",       # GLM-5.1 fallback (deprecated, marked for retire)
})


class ApiProxyBackend:
    @staticmethod
    def _validate_api_key_env(name: str) -> str:
        # B-148: surface bad yaml early with a clear message instead of letting
        # the agent quietly read whatever env var the config requested.
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
                    # B-148 (/stress A1.2 v8 gemini C4, 2026-05-16): allowlist
                    # guard against config-injected env-var redirection (see
                    # ``_ALLOWED_API_KEY_ENVS`` at module level for rationale).
                    "api_key_env": self._validate_api_key_env(
                        config.get("api_key_env", "PROXY_API_KEY"),
                    ),
                    # Plan A/B: tool_use + GLM fallback (§67)
                    "use_tool_calling": config.get("use_tool_calling", False),
                    "use_glm_fallback": config.get("use_glm_fallback", False),
                    "glm_config": config.get("glm_config", ".auth/glm"),
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
            action = {
                "action_type": "scroll",
                "delta": [0, 0.8],
                "coordinate_type": "normalized",
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

        stage_prefix = ""
        if context.stage == "planner":
            stage_prefix = (
                "[Stage: planner] Based on the task and interaction history, "
                "identify the immediate sub-goal for this step. Output ONLY a "
                "short sub-goal description (one sentence), not an action.\n\n"
            )
        elif context.stage == "grounder":
            sub_goal = context.planner_sub_goal or ""
            stage_prefix = (
                f"[Stage: grounder] Sub-goal: {sub_goal}\n"
                "Based on the sub-goal above and the current page state, "
                "produce a concrete action JSON.\n\n"
            )

        prompt = f"{stage_prefix}{instruction}"
        start = time.time()
        action, meta = self._agent.step(
            prompt, obs,
            history=context.history,
            observation_mode=context.observation_mode,
            reference_images=context.reference_images,
        )
        infer_ms = (time.time() - start) * 1000.0

        meta = dict(meta)
        meta.setdefault("model_calls", 1)
        meta.setdefault("backend_type", "api_proxy")
        meta["infer_ms"] = infer_ms
        return action, meta
