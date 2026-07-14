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
def expected_scored_ids(
    site: str,
    benchmark: str = "visualwebarena",
) -> tuple[frozenset[int], str]:
    """Return the exact runner-scored task IDs and their stable SHA256.

    ``load_tasks`` is the single source for site matching and the preregistered
    N/A exclusion.  A temporary output directory absorbs its normal resolved
    task-config writes; the cached immutable result makes this a one-time cost
    per site/process.  ``scored_task_count(strict=True)`` is retained as a
    fail-closed cross-check against config-resolution or duplicate-ID drift.
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
            f"Scored task-universe mismatch for {site}: load_tasks returned "
            f"{len(ids)} IDs but scored_task_count returned {expected_n}"
        )
    return ids, task_id_set_sha256(ids)

