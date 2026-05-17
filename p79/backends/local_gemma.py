from __future__ import annotations

import logging
import time
from typing import Any, Dict, Tuple

from p79.backends.base import BackendError, BackendStepContext
from p79.backends._shared_stage_prefix import build_stage_prefix

logger = logging.getLogger(__name__)


class LocalGemmaBackend:
    """Backend wrapper for the local Gemma 3 vision-language agent (B2 baseline).

    Mirrors ``LocalQwenBackend``'s contract so the runner stays model-agnostic.
    The only deliberate difference is loading ``Gemma3VLAgent`` instead of
    ``Qwen3VLAgent``; both wrappers forward ``revision`` from the backend
    config into the agent config under B-83 + B-136 strict mode so the loaded
    model SHA is provable from run metadata.

    B-814 (/stress A1.2 cold-start P1-7-A Claude, 2026-05-17): docstring
    previously claimed the Qwen path "does not forward revision and falls back
    to a hard-coded default" citing the 2026-05-14 codex cross-review. That
    note is OBSOLETE — local_qwen.py:54 explicitly forwards ``revision`` and
    B-136 strict mode raises ``RuntimeError`` when missing. The stale
    docstring let a reviewer infer asymmetric SHA pinning between baselines
    and attack paper §1 cross-family reproducibility defense.

    No ``dom_mode``/heuristic branch: Gemma always runs the LLM. B-425
    (/stress A1.3 v9 D1, 2026-05-17) retired the HeuristicDomBackend family
    entirely, so the prior B-408 NotImplementedError raise is no longer
    necessary — `dom_mode` is now a no-op config field across all backends.
    """

    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config
        self.mock_mode = bool(config.get("mock_mode", False))
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
            # B-808 (/stress A1.2 cold-start P2-2-AC): removed dead
            # coordinate_type field (see local_qwen.py / factory.py siblings).
            action = {
                "action_type": "scroll",
                "delta": [0, 0.8],
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

        # B-812: shared single-source stage_prefix (see local_qwen.py).
        prompt = f"{build_stage_prefix(context.stage, context.planner_sub_goal)}{instruction}"
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
        # B-813 (/stress A1.2 cold-start P0-2-B*): preserve-None defense
        # mirrored from local_qwen. paper §1 cost telemetry contract.
        if meta.get("model_calls") is None:
            meta["model_calls"] = 1
        if meta.get("backend_type") is None:
            meta["backend_type"] = "local_gemma"
        if bool(self.config.get("paper_grade", False)):
            for _k in ("input_tokens", "output_tokens"):
                if meta.get(_k) is None:
                    raise BackendError(
                        f"local_gemma step returned meta[{_k!r}]=None under "
                        f"paper_grade=True — cost telemetry contract violation. "
                        f"Fix upstream agent or unset paper_grade for smoke "
                        f"runs (B-813)."
                    )
        meta["infer_ms"] = infer_ms
        return action, meta
