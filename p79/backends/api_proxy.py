"""ApiProxyBackend — backend wrapper for the custom proxy API agent."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from p79.backends.base import BackendStepContext
from p79.backends.heuristic import HeuristicDomBackend


class ApiProxyBackend:
    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config
        self.mock_mode = bool(config.get("mock_mode", False))
        self.dom_mode = config.get("dom_mode", "llm")
        self._heuristic = HeuristicDomBackend()
        self._agent = None

        if not self.mock_mode:
            from p79.agents.proxy_api_agent import ProxyApiAgent

            agent_cfg = {
                "model": {
                    "api_name": config.get("api_name", config.get("name", "qwen.qwen3-vl-235b-a22b")),
                    "base_url": config.get("base_url"),
                    "max_new_tokens": config.get("max_new_tokens", 512),
                    "temperature": config.get("temperature", 0.1),
                    "top_p": config.get("top_p", 0.9),
                    "timeout": config.get("timeout", 120),
                },
                "agent": {
                    "image_max_size": config.get("image_max_size", 1024),
                    "max_obs_chars": config.get("max_obs_chars", 12000),
                    "max_image_payload_bytes": config.get("max_image_payload_bytes", 5 * 1024 * 1024),
                },
            }
            self._agent = ProxyApiAgent(agent_cfg)

    def step(self, instruction: str, obs: Any, context: BackendStepContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if context.observation_mode == "dom" and self.dom_mode == "heuristic_only":
            return self._heuristic.step(instruction, obs, context)

        if self.mock_mode:
            action = {
                "action_type": "click",
                "element_id": 1,
                "thought": "Mock API proxy backend action.",
            }
            return action, {
                "raw_output": action,
                "valid": True,
                "failure_reason": None,
                "input_tokens": 2,
                "output_tokens": 2,
                "model_calls": 1,
                "backend_type": "api_proxy_mock",
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
        )
        infer_ms = (time.time() - start) * 1000.0

        meta = dict(meta)
        meta.setdefault("model_calls", 1)
        meta.setdefault("backend_type", "api_proxy")
        meta["infer_ms"] = infer_ms
        return action, meta
