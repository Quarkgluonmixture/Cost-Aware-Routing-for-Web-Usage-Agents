from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from p79.backends.base import BackendStepContext

logger = logging.getLogger(__name__)


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

    No ``dom_mode``/heuristic branch: Gemma always runs the LLM. B-408
    fail-loud below converts the previous silent-no-op into an explicit
    raise if a config requests ``heuristic_only`` for the B2 backend.
    """

    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config
        self.mock_mode = bool(config.get("mock_mode", False))
        # B-408 (/stress A1.2 v8 Mode A+B+C P1-1 3-AI overlap OOB, 2026-05-16):
        # B2 does not implement the heuristic_dom fallback path. Pre-fix
        # `dom_mode: heuristic_only` on a B2 yaml would silently fall through
        # to the LLM (mismatching B0/B1 behavior that actually run the
        # heuristic). Fail loud at init so configs cannot accidentally
        # diverge cross-baseline.
        _dom_mode = config.get("dom_mode", "llm")
        if _dom_mode != "llm":
            raise NotImplementedError(
                f"LocalGemmaBackend (B2) only supports dom_mode='llm', got "
                f"dom_mode={_dom_mode!r} (backend_id={backend_id}). Cross-"
                f"baseline contract: B2 has no HeuristicDomBackend wrapper. "
                f"If you need heuristic-only DOM ablation, use B0 or B1, OR "
                f"add the heuristic branch + tests + paper §3.5 disclosure."
            )
        self._agent = None

        # B-410 (/stress A1.2 v8 Mode A P1-3, 2026-05-16): yaml temperature /
        # top_p are dead config on B1/B2 (agent hardcodes do_sample=False).
        # Warn loud when yaml deviates from paper-grade greedy so a stale
        # `temperature: 0.7` does not silently land on B2 looking like B0
        # honored it.
        _temp = config.get("temperature", 0.0)
        _topp = config.get("top_p", 1.0)
        if _temp != 0.0:
            logger.warning(
                "LocalGemmaBackend yaml temperature=%s ignored — agent "
                "hardcodes do_sample=False (greedy). Cross-baseline drift "
                "risk: B0 (proxy) honors yaml temperature, B1/B2 don't. "
                "Set temperature=0.0 to remove the asymmetry.", _temp,
            )
        if _topp != 1.0:
            logger.warning(
                "LocalGemmaBackend yaml top_p=%s ignored — agent hardcodes "
                "do_sample=False. Same cross-baseline drift as temperature.",
                _topp,
            )

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
                    # B-84: max_obs_chars removed — the agent no longer truncates
                    # obs_text (viewport filter is the real input bound).
                },
                # B-411 (/stress A1.2 v8 Mode A P1-4, 2026-05-16):
                # defense-in-depth — forward paper_grade flag so any future
                # paper-grade gate added inside Gemma3VLAgent (e.g.,
                # torch.use_deterministic_algorithms / cudnn benchmark off /
                # revision drift fail-fast) can fire. Currently inert at
                # the agent layer (no consumer); the wire is here so the
                # contract is symmetric with api_proxy.py:94.
                "paper_grade": bool(config.get("paper_grade", False)),
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
                # B-415 (/stress A1.2 v8 Mode A P2-3, 2026-05-16): align with
                # factory.MockBackend `mock_<backend_id>` naming so cross-
                # baseline mock invariance tests can grep one canonical
                # pattern. Pre-fix string "local_gemma_mock" diverged from
                # factory's `mock_<id>` and api_proxy's `api_proxy_mock`.
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
