from __future__ import annotations

from typing import Any, Dict

from p79.backends.heuristic import HeuristicDomBackend
from p79.backends.local_qwen import LocalQwenBackend


class MockBackend:
    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config

    def step(self, instruction: str, obs: Any, context: Any):
        # /stress A1.2 F5: aligned scroll delta=0.8 with LocalQwenBackend /
        # LocalGemmaBackend mock_mode (both emit 0.8). Previously 0.5 here
        # silently broke any cross-baseline parity test that used factory.MockBackend
        # alongside the local-backend mock_mode paths.
        action = {
            "action_type": "scroll",
            "delta": [0, 0.8],
            "coordinate_type": "normalized",
            "thought": f"Mock backend {self.backend_id}",
        }
        meta = {
            "raw_output": action,
            "valid": True,
            "failure_reason": None,
            "input_tokens": 1,
            "output_tokens": 1,
            "model_calls": 1,
            # B-154 (/stress A1.2 v8 Claude A6, 2026-05-16): tag with the
            # actual backend_id so test_runner_smoke / cross-baseline mock
            # invariants can distinguish mock_B0 vs mock_B1 vs mock_B2
            # vs the bare factory.MockBackend default. Pre-fix: all mocks
            # reported backend_type="mock" so tests couldn't tell which
            # baseline they were exercising.
            "backend_type": f"mock_{self.backend_id}",
        }
        return action, meta


def create_backend(backend_id: str, cfg: Dict[str, Any]):
    # /stress A1.2 F2: backend `type` must be explicit. Previously the
    # `cfg.get("type", "local_qwen")` default silently dispatched a B1 backend
    # whenever a malformed config (typo / missing key / merge bug) reached
    # this function, masking the failure as a successful but mis-labeled run.
    if "type" not in cfg:
        raise ValueError(
            f"Backend cfg missing required 'type' field (backend_id={backend_id}). "
            f"Explicit dispatch only — see configs/exp_v2_base.yaml for examples."
        )
    backend_type = cfg["type"]
    if backend_type == "local_qwen":
        return LocalQwenBackend(backend_id, cfg)
    if backend_type == "local_gemma":
        from p79.backends.local_gemma import LocalGemmaBackend
        return LocalGemmaBackend(backend_id, cfg)
    if backend_type == "api_proxy":
        from p79.backends.api_proxy import ApiProxyBackend
        return ApiProxyBackend(backend_id, cfg)
    if backend_type == "heuristic_dom":
        # /stress A1.2 F3: pass cfg through normally (was: HeuristicDomBackend()
        # + post-construct backend_id = backend_id, which silently dropped cfg).
        return HeuristicDomBackend(backend_id, cfg)
    if backend_type == "mock":
        return MockBackend(backend_id, cfg)
    raise ValueError(f"Unsupported backend type: {backend_type} (backend_id={backend_id})")
