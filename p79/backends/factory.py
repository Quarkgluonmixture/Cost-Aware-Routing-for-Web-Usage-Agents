from __future__ import annotations

from typing import Any, Dict

from p79.backends.api_qwen import ApiQwenBackend
from p79.backends.heuristic import HeuristicDomBackend
from p79.backends.local_qwen import LocalQwenBackend


class MockBackend:
    def __init__(self, backend_id: str, config: Dict[str, Any]):
        self.backend_id = backend_id
        self.config = config

    def step(self, instruction: str, obs: Any, context: Any):
        action = {
            "action_type": "scroll",
            "delta": [0, 0.5],
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
            "backend_type": "mock",
        }
        return action, meta


def create_backend(backend_id: str, cfg: Dict[str, Any]):
    backend_type = cfg.get("type", "local_qwen")
    if backend_type == "local_qwen":
        return LocalQwenBackend(backend_id, cfg)
    if backend_type == "api_qwen":
        return ApiQwenBackend(backend_id, cfg)
    if backend_type == "heuristic_dom":
        b = HeuristicDomBackend()
        b.backend_id = backend_id
        return b
    if backend_type == "mock":
        return MockBackend(backend_id, cfg)
    raise ValueError(f"Unsupported backend type: {backend_type} (backend_id={backend_id})")
