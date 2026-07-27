"""Canonical scored task-ID universes for paper-grade analysis.

The scored-set selection is deliberately delegated to
``p79.experiment.tasks.load_tasks`` so analysis producers cannot drift from the
runner's site filtering or N/A exclusion rules.  This module only turns the
resulting ``TaskSpec`` objects into an immutable ID set plus a stable digest.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from functools import lru_cache
from pathlib import Path


def task_id_set_sha256(task_ids: frozenset[int] | set[int]) -> str:
    """Return a stable SHA256 for an integer task-ID set."""
    canonical = json.dumps(sorted(int(t) for t in task_ids), separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@lru_cache(maxsize=8)
def collected_task_ids(
    site: str,
    benchmark: str = "visualwebarena",
) -> frozenset[int]:
    """Task IDs a run is required to produce episodes for (the COLLECTION set).

    ``load_tasks`` is the single source for site matching and the preregistered
    §139.8 N/A exclusion.  A temporary output directory absorbs its normal
    resolved task-config writes; the cached immutable result makes this a
    one-time cost per site/process.  ``scored_task_count(strict=True)`` is
    retained as a fail-closed cross-check against config-resolution or
    duplicate-ID drift.

    This set is NOT the scoring denominator post-AMENDMENT_08 — see
    ``expected_scored_ids``.
    """
    from p79.experiment.analysis import _resolve_site_config, scored_task_count
    from p79.experiment.tasks import load_tasks

    site = site.lower()
    config_path = _resolve_site_config(site, benchmark)
    if config_path is None:
        raise FileNotFoundError(
            f"No task config for site={site!r}, benchmark={benchmark!r}"
        )

    cfg = {
        "experiment": {"benchmark": benchmark},
        "task": {
            "include_sites": [site],
            "site_configs": {site: str(config_path)},
            "exclude_na_tasks": True,
        },
    }
    with tempfile.TemporaryDirectory(prefix=f"p79-scored-{site}-") as tmp:
        specs = load_tasks(cfg, Path(tmp))

    ids = frozenset(int(spec.task_id) for spec in specs)
    if len(ids) != len(specs):
        raise ValueError(
            f"Duplicate scored task IDs for {site}: {len(specs)} specs, "
            f"{len(ids)} unique IDs"
        )
    expected_n = scored_task_count(site, benchmark, strict=True)
    if len(ids) != expected_n:
        raise ValueError(
            f"Collected task-universe mismatch for {site}: load_tasks returned "
            f"{len(ids)} IDs but scored_task_count returned {expected_n}"
        )
    return ids


@lru_cache(maxsize=16)
def expected_scored_ids(
    site: str,
    benchmark: str = "visualwebarena",
    tiers: tuple[str, ...] = ("A", "B"),
) -> tuple[frozenset[int], str]:
    """Return the SCORED task IDs and their stable SHA256.

    = ``collected_task_ids`` minus the AMENDMENT_08 protocol exclusions. Every
    success RATE in the paper is formed over this set, so changing it changes
    the returned SHA256 — which is recorded in each artifact's
    ``outcome_provenance.canonical_task_universe_sha256``. That makes any
    artifact generated under the pre-amendment universe fail the existing
    cross-artifact SHA check rather than silently mixing denominators.

    ``tiers`` selects which exclusion warrants apply; ``()`` reproduces the
    pre-amendment universe exactly and is how the sensitivity arms are built.
    """
    from p79.experiment.tasks import protocol_excluded_task_ids

    site = site.lower()
    ids = collected_task_ids(site, benchmark)
    excluded = protocol_excluded_task_ids(site, benchmark, tiers=tuple(tiers))
    scored = ids - excluded
    return scored, task_id_set_sha256(scored)


def protocol_excluded_in_universe(
    site: str,
    benchmark: str = "visualwebarena",
    tiers: tuple[str, ...] = ("A", "B"),
) -> frozenset[int]:
    """AMENDMENT_08 exclusions that are actually present in the collected set.

    Aggregators need this to tell "the run produced an extra episode we did not
    ask for" (contamination) apart from "the run produced an episode we asked
    for but no longer score" (expected, post-amendment).
    """
    from p79.experiment.tasks import protocol_excluded_task_ids

    site = site.lower()
    return collected_task_ids(site, benchmark) & protocol_excluded_task_ids(
        site, benchmark, tiers=tuple(tiers)
    )
