from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from p79.backends.base import BackendStepContext
from p79.backends.heuristic import HeuristicDomBackend

logger = logging.getLogger(__name__)


class LocalQwenBackend:
    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config
        self.mock_mode = bool(config.get("mock_mode", False))
        self.dom_mode = config.get("dom_mode", "llm")
        self._heuristic = HeuristicDomBackend()
        self._agent = None

        # B-410 (/stress A1.2 v8 Mode A P1-3, 2026-05-16): yaml temperature /
        # top_p are dead config on B1 (qwen3vl_agent.py:299 hardcodes
        # do_sample=False). Warn loud when yaml deviates so a stale yaml temp
        # cannot silently land. See local_gemma.py for B2 mirror.
        _temp = config.get("temperature", 0.0)
        _topp = config.get("top_p", 1.0)
        if _temp != 0.0:
            logger.warning(
                "LocalQwenBackend yaml temperature=%s ignored — agent "
                "hardcodes do_sample=False (greedy). Cross-baseline drift "
                "risk: B0 (proxy) honors yaml temperature, B1/B2 don't. "
                "Set temperature=0.0 to remove the asymmetry.", _temp,
            )
        if _topp != 1.0:
            logger.warning(
                "LocalQwenBackend yaml top_p=%s ignored — agent hardcodes "
                "do_sample=False. Same cross-baseline drift as temperature.",
                _topp,
            )

        if not self.mock_mode:
            from p79.agents.qwen3vl_agent import Qwen3VLAgent

            agent_cfg = {
                "model": {
                    "path": config.get("path", "Qwen/Qwen3-VL-4B-Instruct"),
                    # B-83 + B-136: forward HF revision SHA so the agent loads
                    # the pinned weights. Post-B-136 strict mode: missing
                    # revision raises ``RuntimeError`` at agent init (the
                    # previous "falls back to default + warns" behavior was
                    # retired so paper-grade reproducibility cannot silently
                    # regress).
                    "revision": config.get("revision"),
                    "quantization": config.get("quantization", "none"),
                    "device": config.get("device", "cuda"),
                    # Default raised 512 → 4096 (§45 alignment, §97 audit):
                    # 512 truncates typical thought+JSON envelope (~400-1500 tok)
                    # → silent parse errors. Configs should set this explicitly.
                    "max_new_tokens": config.get("max_new_tokens", 4096),
                    # B-37 fix: B1 already greedy via do_sample=False, but kept as
                    # config for fallback paths (BLIP-2 captioning etc.). Default
                    # 0.1→0 here for consistency, but qwen3vl_agent ignores at
                    # generate() since do_sample=False is hardcoded.
                    "temperature": config.get("temperature", 0.0),
                    "top_p": config.get("top_p", 1.0),
                    # B-37 fix: forward seed for torch.manual_seed in agent's generate path.
                    "seed": config.get("seed"),
                    "min_free_vram_gb": config.get("min_free_vram_gb", 0),
                },
                "agent": {
                    "image_max_size": config.get("image_max_size", 1024),
                    # B-84: max_obs_chars removed — the agent no longer truncates
                    # obs_text (viewport filter is the real bound).
                },
                # B-411 (/stress A1.2 v8 Mode A P1-4, 2026-05-16):
                # defense-in-depth — forward paper_grade flag so any future
                # paper-grade gate added inside Qwen3VLAgent can fire.
                # Currently inert at the agent layer; the wire matches
                # api_proxy.py:94 contract.
                "paper_grade": bool(config.get("paper_grade", False)),
            }
            self._agent = Qwen3VLAgent(agent_cfg)

    def step(self, instruction: str, obs: Any, context: BackendStepContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if context.observation_mode == "dom" and self.dom_mode == "heuristic_only":
            return self._heuristic.step(instruction, obs, context)

        if self.mock_mode:
            action = {
                "action_type": "scroll",
                "delta": [0, 0.8],
                "coordinate_type": "normalized",
                "thought": "Mock local backend action.",
            }
            return action, {
                "raw_output": action,
                "valid": True,
                "failure_reason": None,
                "input_tokens": 1,
                "output_tokens": 1,
                "model_calls": 1,
                # B-415 (/stress A1.2 v8 Mode A P2-3, 2026-05-16): canonical
                # `mock_<backend_id>` naming (was: "local_qwen_mock"). See
                # factory.MockBackend + local_gemma + api_proxy for sibling
                # alignment. Cross-baseline mock invariance tests now grep
                # one pattern.
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
        action, meta = self._agent.step(prompt, obs, history=context.history, observation_mode=context.observation_mode, reference_images=context.reference_images)
        infer_ms = (time.time() - start) * 1000.0

        meta = dict(meta)
        meta.setdefault("model_calls", 1)
        meta.setdefault("backend_type", "local_qwen")
        meta["infer_ms"] = infer_ms
        return action, meta
