"""B3 dev-pilot backend wrapper — MiMo-VL-7B-RL (cross-family, DEV ONLY).

Thin wrapper mirroring LocalQwenBackend: builds agent_cfg from the backend
config and drives MiMoVLAgent (subclass of Qwen3VLAgent). step() is
model-agnostic (stage-prefix + delegate + meta None-guards + paper_grade cost
contract) — copied verbatim from local_qwen.py so cross-baseline meta parity
holds, with backend_type relabelled to "local_mimo". NOT in the paper-grade
fire import path; lazily imported from factory.py only when type=="local_mimo".
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from p79.backends.base import BackendError, BackendStepContext
from p79.backends._shared_stage_prefix import build_stage_prefix

logger = logging.getLogger(__name__)


class LocalMiMoBackend:
    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config
        self.mock_mode = bool(config.get("mock_mode", False))
        self._agent = None

        # Mirror B-410: yaml temperature/top_p are dead config — the inherited
        # MiMoVLAgent.step() (from Qwen3VLAgent) hardcodes do_sample=False
        # (greedy), matching B0/B1/B2 and the Stage-0 probe. Warn loud on drift.
        _temp = config.get("temperature", 0.0)
        _topp = config.get("top_p", 1.0)
        if _temp != 0.0:
            logger.warning(
                "LocalMiMoBackend yaml temperature=%s ignored — agent hardcodes "
                "do_sample=False (greedy). Set temperature=0.0 to remove asymmetry.",
                _temp,
            )
        if _topp != 1.0:
            logger.warning(
                "LocalMiMoBackend yaml top_p=%s ignored — agent hardcodes "
                "do_sample=False (greedy).", _topp,
            )

        if not self.mock_mode:
            from p79.agents.mimo_vl_agent import MiMoVLAgent

            agent_cfg = {
                "model": {
                    "path": config.get("path", "XiaomiMiMo/MiMo-VL-7B-RL-2508"),
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
                },
                "paper_grade": bool(config.get("paper_grade", False)),
            }
            self._agent = MiMoVLAgent(agent_cfg)

    def step(self, instruction: str, obs: Any, context: BackendStepContext) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if self.mock_mode:
            action = {
                "action_type": "scroll",
                "delta": [0, 0.8],
                "thought": "Mock local backend action.",
            }
            return action, {
                "raw_output": action,
                "valid": True,
                "failure_reason": None,
                "input_tokens": 1,
                "output_tokens": 1,
                "model_calls": 1,
                "backend_type": f"mock_{self.backend_id}",
            }

        assert self._agent is not None

        prompt = f"{build_stage_prefix(context.stage, context.planner_sub_goal)}{instruction}"
        start = time.time()
        action, meta = self._agent.step(
            prompt, obs, history=context.history,
            observation_mode=context.observation_mode,
            reference_images=context.reference_images,
        )
        infer_ms = (time.time() - start) * 1000.0

        meta = dict(meta)
        # B-813: replace present-but-None descriptive fields with defaults;
        # paper_grade=True raises on None cost fields (token counts).
        if meta.get("model_calls") is None:
            meta["model_calls"] = 1
        if meta.get("backend_type") is None:
            meta["backend_type"] = "local_mimo"
        if bool(self.config.get("paper_grade", False)):
            for _k in ("input_tokens", "output_tokens"):
                if meta.get(_k) is None:
                    raise BackendError(
                        f"local_mimo step returned meta[{_k!r}]=None under "
                        f"paper_grade=True — cost telemetry contract violation. "
                        f"Unset paper_grade for dev pilot runs (B-813)."
                    )
        meta["infer_ms"] = infer_ms
        return action, meta
