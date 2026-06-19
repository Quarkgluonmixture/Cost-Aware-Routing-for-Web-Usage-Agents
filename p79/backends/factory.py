from __future__ import annotations

from typing import Any, Dict

from p79.backends.local_qwen import LocalQwenBackend


# B-425 (/stress A1.3 v9 D1, 2026-05-17): HeuristicDomBackend family retired.
# Prior `_ALLOWED_DOM_MODES = {"llm", "heuristic_only"}` enum + B-408 cross-
# baseline drift guard removed because 3-AI deeper audit (Mode A archive grep
# + codex Mode B numeric receipts + gemini Mode C framing) confirmed 0/53924
# step rows + 0/119 yaml configs ever exercised the heuristic dispatch. The
# `dom_mode` config field is preserved for backward-compat with the 41
# paper-grade yamls that explicitly set `dom_mode: "llm"`, but the value is
# now a no-op (LocalQwenBackend / ApiProxyBackend no longer read it). Future
# paper-2 module-ablation work can revive HeuristicDomBackend from git
# history; the per-baseline backend wrappers will then need to opt back in.


class MockBackend:
    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config

    def step(self, instruction: str, obs: Any, context: Any):
        # /stress A1.2 F5: aligned scroll delta=0.8 with LocalQwenBackend /
        # LocalGemmaBackend mock_mode (both emit 0.8). Previously 0.5 here
        # silently broke any cross-baseline parity test that used factory.MockBackend
        # alongside the local-backend mock_mode paths.
        # B-808 (/stress A1.2 cold-start P2-2-AC Claude+gemini, 2026-05-17):
        # removed dead `coordinate_type:"normalized"` field — scroll uses
        # `delta`, not `coordinate`, so coordinate_type was never consumed
        # by the env wrapper but DID leak into downstream aggregator slices
        # that filter on coordinate_type semantics (paper §3.5 coord-type
        # distribution figure). Aligns mock to action_utils canonical scroll
        # shape: {action_type, delta, scroll_direction(canonicalized)}.
        action = {
            "action_type": "scroll",
            "delta": [0, 0.8],
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
    # B-809 (/stress A1.2 cold-start P2-1-A* Claude OOB, 2026-05-17): `dom_mode`
    # field defense parity with retired heuristic_dom backend. Pre-fix only
    # `type:"heuristic_dom"` raised; `dom_mode:"heuristic_only"` (the field-
    # level path that yaml configs use) was silently ignored by the per-
    # baseline wrappers post-B-425 retirement. The 41 paper-grade yamls all
    # currently set `dom_mode:"llm"`, but future drift (operator edits yaml
    # to "heuristic_only" mid-experiment) would silently no-op rather than
    # fail loud. Validate at the single dispatch gate.
    dom_mode = cfg.get("dom_mode", "llm")
    if dom_mode != "llm":
        raise ValueError(
            f"dom_mode={dom_mode!r} unsupported — heuristic_dom backend family "
            f"retired 2026-05-17 (B-425). Only 'llm' is accepted; resurrect "
            f"HeuristicDomBackend from git history if paper-2 module ablation "
            f"resumes. (backend_id={backend_id})"
        )
    backend_type = cfg["type"]
    if backend_type == "local_qwen":
        return LocalQwenBackend(backend_id, cfg)
    if backend_type == "local_gemma":
        from p79.backends.local_gemma import LocalGemmaBackend
        return LocalGemmaBackend(backend_id, cfg)
    # B3 dev pilot (§340, 2026-06-16): MiMo-VL-7B-RL ships as the Qwen2.5-VL
    # deployment class; LocalMiMoBackend drives MiMoVLAgent (subclass of
    # Qwen3VLAgent, same processing stack). Lazy import keeps the Qwen2.5-VL
    # load path off the B0/B1/B2 paper-grade fire import path.
    if backend_type == "local_mimo":
        from p79.backends.local_mimo import LocalMiMoBackend
        return LocalMiMoBackend(backend_id, cfg)
    if backend_type == "api_proxy":
        from p79.backends.api_proxy import ApiProxyBackend
        return ApiProxyBackend(backend_id, cfg)
    if backend_type == "mock":
        return MockBackend(backend_id, cfg)
    # B-425: `heuristic_dom` retired alongside HeuristicDomBackend. Raise an
    # explicit error so any stale yaml / test still requesting it surfaces.
    if backend_type == "heuristic_dom":
        raise ValueError(
            f"Backend type 'heuristic_dom' was retired 2026-05-17 (B-425, "
            f"/stress A1.3 v9 D1). HeuristicDomBackend had 0/53924 paper-grade "
            f"usage; resurrect from git history when paper-2 module ablation "
            f"resumes. (backend_id={backend_id})"
        )
    raise ValueError(f"Unsupported backend type: {backend_type} (backend_id={backend_id})")
