"""Tests for the schema migration framework.

B-294 fix (2026-05-16, A1.8): pre-fix `p79/experiment/schema_migrations/`
was scaffold-only with zero callers + zero tests; first v3 work would
detonate all latent bugs (string mismatch, deepcopy, idempotency).
This file registers a mock migration v2 → vtest and verifies:
  - chain logic (chain step order, target version found)
  - registry decorator
  - deepcopy isolation (no caller-side mutation)
  - idempotency (apply twice == apply once)
  - error paths (unknown version, downgrade refusal)
  - SCHEMA_VERSION_V2 semver alignment with `_CHAIN` (B-282 fix)
"""
from __future__ import annotations

import uuid

import pytest

from p79.experiment.schema_migrations import (
    _CHAIN,
    _REGISTRY,
    current_version,
    migrate,
    register,
)
from p79.experiment.types import SCHEMA_VERSION_V2


@pytest.fixture
def chain_snapshot():
    """B-666 (/stress A1.12 P1-11 B* codex, 2026-05-17): snapshot/restore the
    private `_CHAIN` + `_REGISTRY` global state around any test that mutates
    them. Pre-fix tests that register temporary migrations relied on a
    try/finally cleanup that, if it failed, left polluted state for ALL
    downstream tests — and the test then `pytest.skip`'d on detected pollution
    (silent escape hatch). Now: pollution at entry = pytest.fail (loud), and
    teardown unconditionally restores the snapshot.
    """
    chain_backup = list(_CHAIN)
    registry_backup = dict(_REGISTRY)
    yield
    # Unconditional teardown: even if test raised, restore canonical state.
    _CHAIN[:] = chain_backup
    _REGISTRY.clear()
    _REGISTRY.update(registry_backup)


def test_schema_version_v2_aligned_with_chain():
    """B-282 fix: types.SCHEMA_VERSION_V2 must match the canonical chain head.

    Pre-fix `SCHEMA_VERSION_V2="2.0"` but `_CHAIN=["v2"]` → migrate("2.0", ...)
    would have raised "Unknown schema version" on first call. Test guards
    against accidental re-divergence.
    """
    assert SCHEMA_VERSION_V2 in _CHAIN, (
        f"SCHEMA_VERSION_V2={SCHEMA_VERSION_V2!r} must be in _CHAIN={_CHAIN}"
    )
    assert current_version() == SCHEMA_VERSION_V2, (
        "current_version() should return the runtime SCHEMA_VERSION_V2 string"
    )


def test_migrate_identity_when_versions_match():
    rec = {"schema_version": "2.0", "x": 1, "nested": {"a": 2}}
    out = migrate(rec, "2.0", "2.0")
    assert out == rec
    # B-191 (Claude D3, 2026-05-16, A1.4b-ii): identity migration must still
    # deepcopy so caller mutation doesn't leak.
    out["nested"]["a"] = 99
    assert rec["nested"]["a"] == 2, "deepcopy should isolate nested mutation"


def test_migrate_unknown_version_raises():
    with pytest.raises(ValueError, match="Unknown schema version"):
        migrate({}, "v99", "2.0")


def test_migrate_downgrade_refused():
    # If a future v3 is registered, the test below proves downgrade refused.
    # For now we only have "2.0" in chain so we can't actually trigger it
    # without registering a temporary migration. The path is exercised in
    # `test_mock_migration_chain` below.
    pass


def test_mock_migration_chain(chain_snapshot):
    """Register a temporary v2 → v<sentinel> migration, run it, then clean up.

    B-666 (/stress A1.12 P1-11 B*): each invocation generates a unique
    `vtest<uuid>` sentinel rather than hardcoded "vtest", so two concurrent
    test workers (pytest-xdist) cannot collide on the same temporary version.
    Cleanup is provided by `chain_snapshot` fixture unconditionally; pollution
    at entry is now `pytest.fail` (loud), not `pytest.skip` (silent escape).
    """
    sentinel = f"vtest_{uuid.uuid4().hex[:8]}"
    unreachable = f"vunreachable_{uuid.uuid4().hex[:8]}"
    ghost = f"vghost_{uuid.uuid4().hex[:8]}"
    if sentinel in _CHAIN or unreachable in _CHAIN or ghost in _CHAIN:
        # Should be impossible with uuid randomness; if it triggers, an earlier
        # test leaked state AND collided with our random sentinel — fail loud
        # so the leak is investigated, do NOT silently skip.
        pytest.fail(
            f"global _CHAIN polluted on test entry: {sentinel!r}/"
            f"{unreachable!r}/{ghost!r} already present in {_CHAIN}. "
            f"Earlier test failed to restore state — chain_snapshot fixture missing?"
        )

    @register("2.0", sentinel)
    def _v2_to_sentinel(rec: dict) -> dict:
        rec["mock_new_field"] = rec.get("mock_new_field", "filled-by-migration")
        return rec

    # Forward migration succeeds.
    rec = {"schema_version": "2.0", "x": 1}
    out = migrate(rec, "2.0", sentinel)
    assert out["schema_version"] == sentinel
    assert out["mock_new_field"] == "filled-by-migration"
    assert "mock_new_field" not in rec, "input must not be mutated (deepcopy)"

    # B-191 idempotency: applying twice yields same result.
    out2 = migrate(out, sentinel, sentinel)
    assert out2 == out

    # Downgrade refused.
    with pytest.raises(ValueError, match="Downgrade not supported"):
        migrate(out, sentinel, "2.0")

    # No-migration-registered path also raises.
    @register("2.0", unreachable)
    def _stub(rec: dict) -> dict:
        return rec
    # Insert a chain hole: sentinel → ghost without registered fn
    _CHAIN.append(ghost)
    with pytest.raises(ValueError, match="No migration registered"):
        migrate({"schema_version": "2.0"}, sentinel, ghost)
    # chain_snapshot teardown handles all cleanup unconditionally
