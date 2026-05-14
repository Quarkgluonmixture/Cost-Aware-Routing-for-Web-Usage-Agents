from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from p79.backends.base import BackendStepContext


class LocalGemmaBackend:
    """Backend wrapper for the local Gemma 3 vision-language agent (B2 baseline).

    Mirrors ``LocalQwenBackend``'s contract so the runner stays model-agnostic.
    Two deliberate differences from the Qwen backend:
      1. loads ``Gemma3VLAgent`` instead of ``Qwen3VLAgent``;
      2. forwards ``revision`` from the backend config into the agent config,
         so the loaded model SHA is provable from run metadata. (The Qwen path
         does not forward it and falls back to a hard-coded default — see codex
         cross-review,
         docs/checkpoints/codex_outputs/gemma3vl_integration_crossreview_2026-05-14.md.)

    No ``dom_mode``/heuristic branch: Gemma always runs the LLM (no paper-grade
    config uses ``heuristic_only``).
    """

    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config
        self.mock_mode = bool(config.get("mock_mode", False))
        self._agent = None

        if not self.mock_mode:
            from p79.agents.gemma3vl_agent import Gemma3VLAgent

            agent_cfg = {
                "model": {
                    "path": config.get("path", "google/gemma-3-4b-it"),
                    # Forwarded so the loaded SHA lands in run metadata.
                    "revision": config.get("revision"),
                    "quantization": config.get("quantization", "none"),
                    "device": config.get("device", "cuda"),
                    "max_new_tokens": config.get("max_new_tokens", 4096),
                    "temperature": config.get("temperature", 0.0),
                    "top_p": config.get("top_p", 1.0),
                    "seed": config.get("seed"),
                    "min_free_vram_gb": config.get("min_free_vram_gb", 0),
                },
                "agent": {
                    "image_max_size": config.get("image_max_size", 1024),
                    "max_obs_chars": config.get("max_obs_chars", 12000),
                },
            }
            self._agent = Gemma3VLAgent(agent_cfg)

    def step(
        self, instruction: str, obs: Any, context: BackendStepContext
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.mock_mode:
            action = {
                "action_type": "scroll",
                "delta": [0, 0.8],
                "coordinate_type": "normalized",
                "thought": "Mock local Gemma backend action.",
            }
            return action, {
                "raw_output": action,
                "valid": True,
                "failure_reason": None,
                "input_tokens": 1,
                "output_tokens": 1,
                "model_calls": 1,
                "backend_type": "local_gemma_mock",
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
            prompt,
            obs,
            history=context.history,
            observation_mode=context.observation_mode,
            reference_images=context.reference_images,
        )
        infer_ms = (time.time() - start) * 1000.0

        meta = dict(meta)
        meta.setdefault("model_calls", 1)
        meta.setdefault("backend_type", "local_gemma")
        meta["infer_ms"] = infer_ms
        return action, meta
