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

import pytest

from p79.experiment.schema_migrations import (
    _CHAIN,
    _REGISTRY,
    current_version,
    migrate,
    register,
)
from p79.experiment.types import SCHEMA_VERSION_V2


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


def test_mock_migration_chain():
    """Register a temporary v2 → vtest migration, run it, then clean up."""
    if "vtest" in _CHAIN:
        pytest.skip("vtest already registered (test artifact)")

    @register("2.0", "vtest")
    def _v2_to_vtest(rec: dict) -> dict:
        rec["mock_new_field"] = rec.get("mock_new_field", "filled-by-migration")
        return rec

    try:
        # Forward migration succeeds.
        rec = {"schema_version": "2.0", "x": 1}
        out = migrate(rec, "2.0", "vtest")
        assert out["schema_version"] == "vtest"
        assert out["mock_new_field"] == "filled-by-migration"
        assert "mock_new_field" not in rec, "input must not be mutated (deepcopy)"

        # B-191 idempotency: applying twice yields same result.
        out2 = migrate(out, "vtest", "vtest")
        assert out2 == out

        # Downgrade refused.
        with pytest.raises(ValueError, match="Downgrade not supported"):
            migrate(out, "vtest", "2.0")

        # No-migration-registered path also raises.
        @register("2.0", "vunreachable")
        def _stub(rec: dict) -> dict:
            return rec
        # Insert a chain hole: vtest → vunreachable without registered fn
        _CHAIN.append("vghost")
        try:
            with pytest.raises(ValueError, match="No migration registered"):
                migrate({"schema_version": "2.0"}, "vtest", "vghost")
        finally:
            _CHAIN.remove("vghost")
    finally:
        # Cleanup: remove the mock migration so it doesn't pollute other tests
        _REGISTRY.pop(("2.0", "vtest"), None)
        _REGISTRY.pop(("2.0", "vunreachable"), None)
        if "vtest" in _CHAIN:
            _CHAIN.remove("vtest")
        if "vunreachable" in _CHAIN:
            _CHAIN.remove("vunreachable")
