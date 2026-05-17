from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from p79.backends.base import BackendError, BackendStepContext
from p79.backends._shared_stage_prefix import build_stage_prefix

logger = logging.getLogger(__name__)


class LocalQwenBackend:
    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config
        self.mock_mode = bool(config.get("mock_mode", False))
        # B-425 (/stress A1.3 v9 D1, 2026-05-17): HeuristicDomBackend retired.
        # The `dom_mode` field is preserved in config schema for backward
        # compat with the 41 paper-grade yamls but no longer dispatched.
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
        # B-425 (/stress A1.3 v9 D1, 2026-05-17): heuristic dispatch retired.
        if self.mock_mode:
            # B-808 (/stress A1.2 cold-start P2-2-AC): removed dead
            # coordinate_type field (scroll uses delta, not coord). Aligns
            # with factory.MockBackend / local_gemma / api_proxy mock paths
            # so cross-baseline mock invariance test signature is canonical.
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
                # B-415 (/stress A1.2 v8 Mode A P2-3, 2026-05-16): canonical
                # `mock_<backend_id>` naming (was: "local_qwen_mock"). See
                # factory.MockBackend + local_gemma + api_proxy for sibling
                # alignment. Cross-baseline mock invariance tests now grep
                # one pattern.
                "backend_type": f"mock_{self.backend_id}",
            }

        assert self._agent is not None

        # B-812 (/stress A1.2 cold-start P1-1-A* Claude OOB, 2026-05-17):
        # stage_prefix now sourced from _shared_stage_prefix.build_stage_prefix
        # so all three backend wrappers (local_qwen / local_gemma / api_proxy)
        # share a single byte-identical prefix definition. See module docstring
        # for paper §3.4 planner/grounder ablation rationale.
        prompt = f"{build_stage_prefix(context.stage, context.planner_sub_goal)}{instruction}"
        start = time.time()
        action, meta = self._agent.step(prompt, obs, history=context.history, observation_mode=context.observation_mode, reference_images=context.reference_images)
        infer_ms = (time.time() - start) * 1000.0

        meta = dict(meta)
        # B-813 (/stress A1.2 cold-start P0-2-B* codex OOB, 2026-05-17): handle
        # present-but-None metadata values. `setdefault(k, default)` does NOT
        # replace an explicit `None` — agent layer returning
        # ``{"backend_type": None, "model_calls": None, "input_tokens": None,
        # "output_tokens": None}`` would slip through downstream
        # `int(meta.get("input_tokens") or 0)` → silent zero-cost record.
        # Cost-aware paper §1 hero metric depends on non-silent failure here.
        # Replace present-but-None with default for descriptive fields;
        # paper_grade=True raises for cost fields (token counts).
        if meta.get("model_calls") is None:
            meta["model_calls"] = 1
        if meta.get("backend_type") is None:
            meta["backend_type"] = "local_qwen"
        if bool(self.config.get("paper_grade", False)):
            for _k in ("input_tokens", "output_tokens"):
                if meta.get(_k) is None:
                    raise BackendError(
                        f"local_qwen step returned meta[{_k!r}]=None under "
                        f"paper_grade=True — cost telemetry contract violation. "
                        f"Fix upstream agent or unset paper_grade for smoke "
                        f"runs (B-813)."
                    )
        meta["infer_ms"] = infer_ms
        return action, meta
