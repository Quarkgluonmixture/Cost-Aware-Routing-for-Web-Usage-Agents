from __future__ import annotations

from typing import Any, Dict

from p79.backends.heuristic import HeuristicDomBackend
from p79.backends.local_qwen import LocalQwenBackend


# B-408 (/stress A1.2 v8 Mode A+B+C P1-1 3-AI overlap OOB, 2026-05-16):
# canonical enum for `dom_mode`. Pre-fix three drift sources:
#   1. tests/test_runner_smoke.py:80 uses `dom_mode:"heuristic"` (no _only)
#   2. p79/backends/{api_proxy,local_qwen}.py recognise only "heuristic_only"
#   3. p79/backends/local_gemma.py has no dom_mode branch at all
# → same primitive 3 different semantics across B0/B1/B2 → smoke / router /
# ablation results cross-baseline inconsistent. Now the factory rejects any
# value outside this enum at init time so config typos surface immediately.
_ALLOWED_DOM_MODES = frozenset({"llm", "heuristic_only"})


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
            # B-154 + B-415 (/stress A1.2 v8 Mode A P2-3, 2026-05-16): tag
            # with the actual backend_id so test_runner_smoke /
            # cross-baseline mock invariants can distinguish mock_B0 vs
            # mock_B1 vs mock_B2 vs the bare factory.MockBackend default.
            # Naming canonical: `mock_<backend_id>` (B-415 aligns to this);
            # local_qwen / local_gemma / api_proxy mock_mode paths use the
            # same scheme below.
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
    # B-408 (P1-1): validate dom_mode enum at dispatch time. heuristic_dom +
    # mock + future backend variants skip this since they don't honor
    # dom_mode. Local_gemma raises NotImplementedError on heuristic_only at
    # construct time (see local_gemma.py B-408 fix).
    backend_type = cfg["type"]
    if backend_type in {"local_qwen", "local_gemma", "api_proxy"}:
        _dom_mode = cfg.get("dom_mode", "llm")
        if _dom_mode not in _ALLOWED_DOM_MODES:
            raise ValueError(
                f"Backend cfg dom_mode={_dom_mode!r} not in allowed set "
                f"{sorted(_ALLOWED_DOM_MODES)} (backend_id={backend_id}). "
                f"Common drift: yaml/tests use 'heuristic' but code expects "
                f"'heuristic_only'. Fix the config or use 'llm'."
            )
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
