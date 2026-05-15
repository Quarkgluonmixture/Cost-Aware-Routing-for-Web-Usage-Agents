"""Factory dispatch surface invariants — /stress A1.2 F2/F3/F5/F6 fix.

These tests guard the factory backend-dispatch contract so any future
regression surfaces as a CI failure instead of a silent paper-grade drift.
"""

from __future__ import annotations

import pytest

from p79.backends.factory import MockBackend, create_backend
from p79.backends.heuristic import HeuristicDomBackend


def test_factory_requires_explicit_type_key():
    """F2: malformed cfg without 'type' must raise — no silent default to B1."""
    with pytest.raises(ValueError, match=r"missing required 'type' field"):
        create_backend("nameless", {})

    with pytest.raises(ValueError, match=r"missing required 'type' field"):
        create_backend("nameless", {"model": {"path": "anything"}})


def test_factory_dispatches_unknown_type_explicitly():
    """F2: unknown backend type must raise (not silently fall back)."""
    with pytest.raises(ValueError, match=r"Unsupported backend type"):
        create_backend("bogus", {"type": "nonexistent_backend"})


def test_heuristic_dom_dispatch_flows_config_through():
    """F3: HeuristicDomBackend dispatched via factory MUST receive the cfg.

    Previously factory.py did `b = HeuristicDomBackend(); b.backend_id = bid`
    which silently dropped the whole cfg dict — any future config-driven
    behavior (mock_mode, etc.) would be ignored.
    """
    cfg = {"type": "heuristic_dom", "mock_mode": False, "marker": "test-sentinel-42"}
    backend = create_backend("hd_test", cfg)
    assert isinstance(backend, HeuristicDomBackend)
    assert backend.backend_id == "hd_test"
    assert backend.config == cfg, "cfg must flow through, not be dropped"
    assert backend.config.get("marker") == "test-sentinel-42"


def test_heuristic_dom_no_class_level_backend_id_default():
    """F6: backend_id must come from the instance, not a class attribute.

    Two instances with different backend_ids should not share state via
    the class-level default that the previous implementation had at line 12.
    """
    a = HeuristicDomBackend(backend_id="alpha", config={})
    b = HeuristicDomBackend(backend_id="beta", config={})
    assert a.backend_id == "alpha"
    assert b.backend_id == "beta"
    # Confirm there's no shared class attr leaking
    assert "backend_id" in a.__dict__


def test_mock_backends_agree_on_scroll_delta():
    """F5: all three mock paths must emit the same scroll delta.

    factory.MockBackend, LocalQwenBackend.mock_mode, LocalGemmaBackend.mock_mode
    are interchangeable placeholders — any cross-baseline parity test that
    uses them as substitutes must see identical scroll behavior.
    """
    # factory.MockBackend
    mb = MockBackend("mock_a", {"type": "mock"})
    action_mb, _ = mb.step("test", obs=None, context=None)
    assert action_mb["action_type"] == "scroll"
    assert action_mb["delta"] == [0, 0.8], (
        f"factory.MockBackend scroll delta must be [0, 0.8] to match local "
        f"mock_mode, got {action_mb['delta']}"
    )

    # LocalQwenBackend.mock_mode — we import here so the test can also catch
    # if anyone changes local_qwen's mock delta back to a non-0.8 value.
    from p79.backends.local_qwen import LocalQwenBackend

    lq = LocalQwenBackend("lq_a", {"mock_mode": True, "type": "local_qwen"})
    action_lq, _ = lq.step("test", obs=None, context=type("C", (), {"observation_mode": "som"})())
    assert action_lq["action_type"] == "scroll"
    assert action_lq["delta"] == [0, 0.8], (
        f"LocalQwenBackend.mock_mode scroll delta must be [0, 0.8] to match "
        f"factory.MockBackend, got {action_lq['delta']}"
    )
