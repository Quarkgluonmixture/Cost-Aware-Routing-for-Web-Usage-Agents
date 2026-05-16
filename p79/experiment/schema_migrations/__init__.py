"""Schema migration framework for episode/condition/run summaries.

Why this exists
---------------
Pre-§97 the team accumulated ~10 schema additions to `EpisodeSummaryV2`
without a migration path: `busy_wait_total_ms`, `energy_partial`,
`adjusted_success`, `wasted_cost_usd`, etc. Each was added by appending an
optional dataclass field — old data files lacked them and downstream
analyses had ad-hoc `r.get(field)` defenses.

This module provides a **registry-based migration path** so the next schema
bump (v2 → v3) has a documented upgrade routine. Old summaries can be
upgraded in place via `migrate(record, "v2", "v3")`.

Status: scaffolding only. The current canonical version is "v2"; a migration
to "v3" will be added when the next breaking-or-additive batch lands.

Usage
-----
    from p79.experiment.schema_migrations import migrate, current_version

    record = json.load(open("episode_summary_v2.json"))
    migrated = migrate(record, from_v=record.get("schema_version", "2.0"),
                       to_v=current_version())

Design
------
- Migrations are pure functions `dict -> dict` registered per `(from, to)` pair.
- `migrate()` chains registered migrations in order (e.g. v2 → v3 → v4).
- Each migration must be idempotent: applying twice yields same result.
- Field defaults live in `v2.py` / `v3.py` as the source of truth for "what
  fields exist at this schema version + their default value".
"""
from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, List, Tuple

# Migration registry: (from_version, to_version) -> migration function
_REGISTRY: Dict[Tuple[str, str], Callable[[dict], dict]] = {}

# B-282 fix (2026-05-16, A1.8): canonical chain uses semver matching
# `types.SCHEMA_VERSION_V2 = "2.0"`. Pre-fix `_CHAIN = ["v2"]` did not match
# what the runner writes to disk; any migrate("2.0", "3.0") call would have
# raised "Unknown schema version". Now both forms live as semver strings;
# v3 lands as "3.0" via `_CHAIN.append("3.0")` + register a `("2.0", "3.0")`
# migration function.
_CHAIN: List[str] = ["2.0"]


def register(from_v: str, to_v: str):
    """Decorator: register a migration `from_v -> to_v`.

    Example::

        @register("2.0", "3.0")
        def upgrade_v2_to_v3(rec: dict) -> dict:
            rec["new_field"] = rec.get("new_field", default_value)
            return rec
    """
    def _decorate(fn: Callable[[dict], dict]) -> Callable[[dict], dict]:
        _REGISTRY[(from_v, to_v)] = fn
        if to_v not in _CHAIN:
            _CHAIN.append(to_v)
        return fn
    return _decorate


def current_version() -> str:
    """Return the latest registered schema version."""
    return _CHAIN[-1]


def migrate(record: dict, from_v: str, to_v: str) -> dict:
    """Apply chained migrations to upgrade `record` from `from_v` to `to_v`.

    Returns a NEW dict (does not mutate input). Raises if no migration path
    exists between the requested versions.

    B-191 (/stress A1.4b-ii Claude D3, P0 OOB): use `deepcopy` not the
    shallow `dict(record)` at entry — nested dicts (`trigger_distribution`,
    `state_change_reason_distribution`, `module_flags`, etc.) would otherwise
    share references with the caller's source data, and any future migration
    that mutates a nested dict in-place would silently corrupt caller state.
    Same defensive class as B-164 in `runner/main.py`.
    """
    if from_v == to_v:
        return deepcopy(record)

    try:
        i_from = _CHAIN.index(from_v)
        i_to = _CHAIN.index(to_v)
    except ValueError as exc:
        raise ValueError(f"Unknown schema version: {exc}") from exc

    if i_to < i_from:
        raise ValueError(f"Downgrade not supported: {from_v} -> {to_v}")

    out = deepcopy(record)
    for i in range(i_from, i_to):
        f, t = _CHAIN[i], _CHAIN[i + 1]
        if (f, t) not in _REGISTRY:
            raise ValueError(f"No migration registered for {f} -> {t}")
        out = _REGISTRY[(f, t)](out)
        out["schema_version"] = t
    return out


# Import side-effect: register all built-in migrations.
# (When v3 lands, add `from . import v2_to_v3`)
from p79.experiment.schema_migrations import v2  # noqa: E402, F401
