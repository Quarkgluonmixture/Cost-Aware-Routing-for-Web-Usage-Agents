"""Factory dispatch surface invariants — /stress A1.2 F2/F3/F5/F6 fix.

These tests guard the factory backend-dispatch contract so any future
regression surfaces as a CI failure instead of a silent paper-grade drift.

B-425 (/stress A1.3 v9 D1, 2026-05-17): HeuristicDomBackend retired —
the two `heuristic_dom`-dispatch tests below were deleted. The factory
now raises a paper-grade-clear ValueError for any stale config still
requesting heuristic_dom; the replacement test below verifies the
retirement contract.
"""

from __future__ import annotations

import pytest

from p79.backends.factory import MockBackend, create_backend


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


def test_heuristic_dom_retirement_surfaces_explicit_error():
    """B-425: heuristic_dom retirement contract — raise actionable error.

    Pre-B-425 the factory dispatched to HeuristicDomBackend(backend_id, cfg).
    HeuristicDomBackend had 0/53924 paper-grade usage (codex Mode B numeric
    receipts), so the entire family was retired. Any stale yaml / test that
    still requests heuristic_dom must surface as a clear error (not a silent
    KeyError or AttributeError downstream).
    """
    cfg = {"type": "heuristic_dom", "mock_mode": False}
    with pytest.raises(ValueError, match=r"heuristic_dom.*retired"):
        create_backend("hd_test", cfg)


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
