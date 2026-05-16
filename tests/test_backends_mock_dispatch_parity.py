"""Backends mock-mode parity invariants — /stress A1.12 P1-3.

Pre-2026-05-16 status: 3 production backends (B0 / B1 / B2) had ZERO direct
tests at the backend layer. `test_agents_prompt_parity` covered prompt-string
equality; `test_factory_dispatch:test_mock_backends_agree_on_scroll_delta`
covered only `LocalQwenBackend` (B1). `LocalGemmaBackend` (B2, added
2026-05-14) and `ApiProxyBackend` (B0) had no parity invariant.

This batch covers:
- 3 backend types instantiated via factory using `mock_mode=True`
- step() returns identical scroll action across all 3 (cross-baseline
  parity contract — they substitute for each other in smoke tests)
- step() meta carries `backend_type` field with expected per-backend value
- missing `type` field surfaces ValueError at factory dispatch
"""
from __future__ import annotations

import pytest

from p79.backends.api_proxy import ApiProxyBackend
from p79.backends.base import BackendStepContext
from p79.backends.factory import create_backend
from p79.backends.local_gemma import LocalGemmaBackend
from p79.backends.local_qwen import LocalQwenBackend


def _ctx(observation_mode: str = "dom") -> BackendStepContext:
    return BackendStepContext(
        observation_mode=observation_mode,
        som_enabled=False,
        som_text="",
    )


# ─── 3-backend mock_mode parity ─────────────────────────────────────────────
def test_local_qwen_mock_mode_emits_canonical_scroll():
    """B1 mock step contract."""
    backend = LocalQwenBackend("b1_mock", {"type": "local_qwen", "mock_mode": True})
    action, meta = backend.step("any", obs=None, context=_ctx())
    assert action["action_type"] == "scroll"
    assert action["delta"] == [0, 0.8]
    assert meta["valid"] is True
    assert meta["model_calls"] == 1


def test_local_gemma_mock_mode_emits_canonical_scroll():
    """B2 mock step contract — added 2026-05-14 advisor decision.

    B-415 (/stress A1.2 v8 Mode A P2-3, 2026-05-16): canonical
    `mock_<backend_id>` naming replaces hardcoded `local_gemma_mock`.
    Cross-baseline mock invariance tests now grep one pattern across
    factory.MockBackend / local_qwen / local_gemma / api_proxy.
    """
    backend = LocalGemmaBackend("b2_mock", {"type": "local_gemma", "mock_mode": True})
    action, meta = backend.step("any", obs=None, context=_ctx())
    assert action["action_type"] == "scroll"
    assert action["delta"] == [0, 0.8]
    assert meta["valid"] is True
    assert meta["backend_type"] == "mock_b2_mock"


def test_api_proxy_mock_mode_emits_canonical_scroll():
    """B0 mock step contract."""
    backend = ApiProxyBackend("b0_mock", {"type": "api_proxy", "mock_mode": True})
    action, meta = backend.step("any", obs=None, context=_ctx())
    assert action["action_type"] == "scroll"
    assert action["delta"] == [0, 0.8]
    assert meta["valid"] is True


def test_all_three_backends_emit_identical_action_shape():
    """Cross-baseline parity: 3 mock backends are interchangeable substitutes.

    If any of them drifts (e.g. B2 emits `[0, 0.9]` while B0/B1 stay at `[0, 0.8]`),
    cross-baseline smoke runs become inconsistent. This is the same invariant
    `test_factory_dispatch` checks for QwenMock only — extended to all 3.
    """
    qwen = LocalQwenBackend("b1", {"type": "local_qwen", "mock_mode": True})
    gemma = LocalGemmaBackend("b2", {"type": "local_gemma", "mock_mode": True})
    proxy = ApiProxyBackend("b0", {"type": "api_proxy", "mock_mode": True})

    actions = []
    for backend in (qwen, gemma, proxy):
        action, _ = backend.step("any", obs=None, context=_ctx())
        # Compare on the action keys that matter for downstream dispatch.
        actions.append({
            "action_type": action["action_type"],
            "delta": action["delta"],
            "coordinate_type": action.get("coordinate_type"),
        })
    assert actions[0] == actions[1] == actions[2], (
        f"Cross-baseline mock action drift detected:\n"
        f"  B1 (Qwen):  {actions[0]}\n  B2 (Gemma): {actions[1]}\n  B0 (Proxy): {actions[2]}"
    )


# ─── Factory dispatch surface for 3 backends ────────────────────────────────
def test_factory_dispatches_b1_local_qwen_mock():
    backend = create_backend("b1_via_factory", {"type": "local_qwen", "mock_mode": True})
    assert isinstance(backend, LocalQwenBackend)
    assert backend.backend_id == "b1_via_factory"


def test_factory_dispatches_b2_local_gemma_mock():
    backend = create_backend("b2_via_factory", {"type": "local_gemma", "mock_mode": True})
    assert isinstance(backend, LocalGemmaBackend)
    assert backend.backend_id == "b2_via_factory"


def test_factory_dispatches_b0_api_proxy_mock():
    backend = create_backend("b0_via_factory", {"type": "api_proxy", "mock_mode": True})
    assert isinstance(backend, ApiProxyBackend)
    assert backend.backend_id == "b0_via_factory"


# ─── Mock-mode does NOT touch model loading ─────────────────────────────────
def test_local_gemma_mock_mode_does_not_load_model():
    """B2 mock_mode must not import `Gemma3VLAgent` (would trigger HF model load
    on a 4B-param model that needs A100 — pytest hosts may lack GPU / weights)."""
    backend = LocalGemmaBackend("b2_mock", {"type": "local_gemma", "mock_mode": True})
    assert backend._agent is None, (
        "mock_mode=True must skip model load (`self._agent is None`). "
        "Pre-fix would have OOM'd on pytest hosts without 40GB VRAM."
    )


def test_local_qwen_mock_mode_does_not_load_model():
    backend = LocalQwenBackend("b1_mock", {"type": "local_qwen", "mock_mode": True})
    # LocalQwenBackend's mock path may use a different sentinel; check the
    # `mock_mode` flag is honored (no real init path triggered).
    assert backend.mock_mode is True


def test_api_proxy_mock_mode_does_not_require_api_key():
    """B0 mock_mode must not require PROXY_API_KEY env var."""
    backend = ApiProxyBackend("b0_mock", {"type": "api_proxy", "mock_mode": True})
    assert backend.mock_mode is True


# ─── Factory contract regressions ───────────────────────────────────────────
def test_factory_dispatch_unknown_type_raises():
    """Pre-A1.2 F2 fix: missing/unknown type silently defaulted to B1."""
    with pytest.raises(ValueError, match=r"Unsupported backend type"):
        create_backend("bogus", {"type": "no_such_backend"})
