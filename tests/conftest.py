"""Shared test fixture builders — derive from the canonical schema defaults.

Fire-6 RCA Stage C1 fixture-drift cleanup (2026-05-20): step-record / episode-
summary test fixtures used to hardcode field lists that drifted whenever a new
paper-grade optional key landed (Phase-2 intervention fields commit 8d2a327;
`total_latency_minus_retry_ms` commit e20c6ef). Deriving every fixture from
`STEP_RECORD_V2_DEFAULTS` / `EPISODE_SUMMARY_V2_DEFAULTS` (the 4-place-synced
source of truth — see tests/test_schema_4place_sync.py) makes them auto-sync
with future schema additions, killing the whole drift class.

Import in a test module with `from conftest import complete_step_record,
complete_episode_summary` (tests/ has no __init__.py → pytest prepend import
mode puts this dir on sys.path).
"""
from __future__ import annotations

from typing import Any, Dict


def complete_step_record(**overrides: Any) -> Dict[str, Any]:
    """A step-record dict that passes `validate_step_record_v2`, with every
    schema-v2 key present (incl. the full `PAPER_GRADE_STEP_OPTIONAL_KEYS` set).

    Two DEFAULTS sentinels are re-seeded because they encode "legacy-read"
    neutral values that the paper-grade write-boundary validator rejects:
    `cost_usd` ships as `{}` in DEFAULTS but B-338 requires the 5 nested keys
    {input, output, model, router_overhead, total}. Override any field via
    kwargs (e.g. to perturb one field for a poisoned-input test).
    """
    from p79.experiment.schema_migrations.v2 import STEP_RECORD_V2_DEFAULTS
    rec = dict(STEP_RECORD_V2_DEFAULTS)
    rec["cost_usd"] = {
        "input": 0.0, "output": 0.0, "model": 0.0,
        "router_overhead": 0.0, "total": 0.0,
    }
    rec.update(overrides)
    return rec


def complete_episode_summary(**overrides: Any) -> Dict[str, Any]:
    """An episode-summary dict that passes `validate_episode_summary_v2` and is
    safe to feed to `aggregate_condition_metrics`, with every schema-v2 key
    present (incl. the full `PAPER_GRADE_EPISODE_OPTIONAL_KEYS` set).

    `total_latency_minus_retry_ms` ships as `None` in DEFAULTS (legacy-vintage
    sentinel) but `aggregate_condition_metrics` calls
    `_avg('total_latency_minus_retry_ms', require_present=True)` which raises
    when every episode leaves it None. A fresh paper-grade episode has no B0
    network-retry scaffold, so retry-adjusted latency equals raw latency —
    mirror `total_latency_ms` unless the caller sets the field explicitly
    (pass `total_latency_minus_retry_ms=None` to exercise the mixed-vintage
    raise path).
    """
    from p79.experiment.schema_migrations.v2 import EPISODE_SUMMARY_V2_DEFAULTS
    ep = dict(EPISODE_SUMMARY_V2_DEFAULTS)
    ep.update(overrides)
    if "total_latency_minus_retry_ms" not in overrides:
        ep["total_latency_minus_retry_ms"] = ep.get("total_latency_ms", 0.0)
    return ep
