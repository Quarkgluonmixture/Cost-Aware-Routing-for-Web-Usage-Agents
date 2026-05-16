"""Invariant tests for /stress A1.4b-ii G2 schema integrity (B-190/B-191/B-192).

- B-190: `STEP_RECORD_V2_DEFAULTS` catalog + `fill_step_defaults()` helper exist
- B-191: `schema_migrations.migrate` uses deepcopy (nested dicts isolated)
- B-192: `_collect_episode_summaries` actively exercises `fill_defaults` so
         framework is no longer dead infrastructure
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ─── B-190 ──────────────────────────────────────────────────────────────────
def test_b190_step_record_defaults_catalog_exists():
    from p79.experiment.schema_migrations.v2 import (
        STEP_RECORD_V2_DEFAULTS,
        fill_step_defaults,
        SCHEMA_VERSION_V2,
    )
    # All 5 paper-grade-critical optionals listed
    for k in ("parse_valid", "parse_failure_reason", "fallback_finish",
              "image_meta", "locator_route_meta", "agent_visible_changed"):
        assert k in STEP_RECORD_V2_DEFAULTS, f"missing critical field {k}"
    # All defaults are None or default-typed (no nonzero arbitrary values)
    assert STEP_RECORD_V2_DEFAULTS["parse_valid"] is None
    assert STEP_RECORD_V2_DEFAULTS["image_meta"] is None
    assert STEP_RECORD_V2_DEFAULTS["locator_route_meta"] is None
    # Schema version matches runtime constant
    assert STEP_RECORD_V2_DEFAULTS["schema_version"] == SCHEMA_VERSION_V2


def test_b190_fill_step_defaults_preserves_actual_values():
    """fill_step_defaults must overlay actual values over defaults, not overwrite."""
    from p79.experiment.schema_migrations.v2 import fill_step_defaults
    raw = {
        "schema_version": "2.0", "run_id": "r1", "condition_id": "c1",
        "benchmark": "vwa", "benchmark_site": "classifieds", "task_id": 5,
        "seed": 42, "step_idx": 3,
        # legacy row missing image_meta + locator_route_meta + parse_valid
    }
    filled = fill_step_defaults(raw)
    # Actuals preserved
    assert filled["task_id"] == 5
    assert filled["step_idx"] == 3
    # Defaults filled
    assert "image_meta" in filled and filled["image_meta"] is None
    assert "locator_route_meta" in filled and filled["locator_route_meta"] is None
    assert "parse_valid" in filled and filled["parse_valid"] is None


# ─── B-191 ──────────────────────────────────────────────────────────────────
def test_b191_migrate_uses_deepcopy_not_shallow():
    """migrate() must not share nested dict references with caller."""
    from p79.experiment.schema_migrations import migrate
    src = {
        "schema_version": "v2",
        "trigger_distribution": {"a": 1, "b": 2},
        "module_flags": {"m1": True},
    }
    out = migrate(src, "v2", "v2")  # no-op migration; tests entry deepcopy
    # Returned dict is a new object
    assert out is not src
    # Nested dicts are also new objects (deepcopy not shallow)
    assert out["trigger_distribution"] is not src["trigger_distribution"]
    assert out["module_flags"] is not src["module_flags"]
    # Mutation of returned data does not propagate to source
    out["trigger_distribution"]["a"] = 999
    assert src["trigger_distribution"]["a"] == 1


def test_b191_migrate_source_uses_deepcopy_keyword():
    """Source-level check: schema_migrations/__init__.py uses deepcopy."""
    src = (Path(__file__).resolve().parents[1] /
           "p79" / "experiment" / "schema_migrations" / "__init__.py").read_text()
    assert "from copy import deepcopy" in src
    assert "B-191" in src
    # The shallow `dict(record)` entry pattern is gone from migrate()
    # (only acceptable use is in unrelated registry storage)
    in_migrate = src.split("def migrate(")[1].split("\ndef ")[0]
    assert "deepcopy(record)" in in_migrate
    # The functional pre-fix `out = dict(record)` and `return dict(record)`
    # entry patterns must be gone (the docstring can still mention them
    # historically, but no executable line uses them in migrate body).
    lines = [l for l in in_migrate.splitlines()
             if l.strip() and not l.strip().startswith(("#", '"', "'"))]
    code_text = "\n".join(lines)
    assert "out = dict(record)" not in code_text
    assert "return dict(record)" not in code_text


# ─── B-192 ──────────────────────────────────────────────────────────────────
def test_b192_collect_episode_summaries_calls_fill_defaults(tmp_path):
    """Legacy summary missing optional fields gets backfilled by fill_defaults."""
    pytest.importorskip("pandas")
    from p79.experiment.analysis import _collect_episode_summaries

    cond = tmp_path / "phase1_dom_router_0"
    eps = cond / "episodes"
    eps.mkdir(parents=True)
    # Write a LEGACY summary missing newer optional fields like
    # `state_change_reason_distribution`, `busy_wait_total_ms`, `energy_partial`
    legacy = {
        "schema_version": "2.0", "run_id": "legacy_run", "condition_id": "c1",
        "benchmark": "vwa", "benchmark_site": "classifieds", "task_id": 1,
        "seed": 42, "success": True, "score": 1.0,
        "steps": 5, "retries": 0, "no_op_rate": 0.0, "page_unchanged_rate": 0.0,
        "total_latency_ms": 500.0, "p95_step_latency_ms": 100.0,
        "total_tokens": 100, "total_model_cost_usd": 0.01, "total_cost_usd": 0.01,
        "total_router_overhead_cost_usd": 0.0, "total_router_overhead_ms": 0.0,
        "total_energy_kwh": None, "total_co2e_kg": None,
        "escalation_count": 0, "trigger_distribution": {},
        "benchmark_noise": False, "benchmark_noise_category": None,
        "artifacts_dir": "",
    }
    (eps / "1_summary_v2.json").write_text(json.dumps(legacy))

    rows = _collect_episode_summaries(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    # Legacy values preserved
    assert r["task_id"] == 1
    assert r["success"] is True
    # B-192: fill_defaults added missing optionals
    assert "state_change_reason_distribution" in r  # default {} added
    assert "busy_wait_total_ms" in r                # default 0.0 added
    assert "energy_partial" in r                    # default False added


def test_b192_source_imports_fill_defaults():
    src = (Path(__file__).resolve().parents[1] /
           "p79" / "experiment" / "analysis.py").read_text()
    assert "B-192" in src
    assert "from p79.experiment.schema_migrations.v2 import" in src
    assert "fill_defaults" in src
