"""Canonical Pass-1 / Pass-2 run discovery + optional manifest whitelist.

router /stress 2026-05-21 (B-1810 C2): single discovery path shared by
`scripts/analysis/extract_50_features.py` (Stage 1 labels) and
`scripts/analysis/aggregate_h10_pareto.py` (§6 H10). Both previously globbed
`{baseline}_*_{site}_*` and excluded only `router_learned` — which matches
`*_smoke_*`, aborted/partial reruns, and stale pre-fix local runs and silently
folds them into paper-grade labels + H10 with no provenance trail.

Resolution order (per cell):
  1. If a manifest exists and lists this cell, restrict to exactly those run-dir
     names (paper-grade strict). This is the post-Pass-1-land state.
  2. Else glob and REJECT non-canonical runs (smoke/test/debug/dryrun) with a
     logged warning, and warn when >1 canonical run remains (outcome-overwrite
     ambiguity — `collect_per_task_outcomes` keeps the newest, no precedence).

The manifest is intentionally optional so the mechanism exists now (zero-cost,
pre-Pass-1) and is filled with the exact paper-grade run IDs once Pass-1 lands.
Manifest schema:
    {"pass1": {"B0_classifieds": ["B0_dom_classifieds_..._R9755", ...], ...},
     "pass2_router": {"B0_classifieds": ["B0_router_learned_classifieds_..."]}}
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Substrings that mark a run dir as NOT paper-grade canonical.
NON_CANONICAL_MARKERS = ("smoke", "_test_", "_debug_", "_dryrun_", "_scratch_")

# Default manifest location (alongside the L1 router artifacts).
DEFAULT_MANIFEST = "results/phantom_paper/l1_router/pass1_run_manifest.json"


def is_non_canonical(run_name: str) -> bool:
    """True if a run-dir name looks like a smoke / test / debug / scratch run."""
    low = run_name.lower()
    return any(marker in low for marker in NON_CANONICAL_MARKERS)


def load_manifest(manifest_path: str | Path | None = None) -> Optional[dict[str, Any]]:
    """Load the run manifest JSON, or None if absent/unreadable.

    Defaults to DEFAULT_MANIFEST relative to CWD. Absence is normal pre-Pass-1.
    """
    path = Path(manifest_path) if manifest_path is not None else Path(DEFAULT_MANIFEST)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("[pass1_manifest] manifest %s unreadable (%s); ignoring", path, e)
        return None


def discover_runs(
    phase1_root: str | Path,
    baseline: str,
    site: str,
    *,
    router: bool = False,
    manifest_path: str | Path | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    """Discover Pass-1 baseline (router=False) or Pass-2 router (router=True) run dirs.

    Returns (sorted_run_dirs, provenance). `provenance` records the resolution mode +
    rejected dirs + any overwrite-ambiguity warning so the caller can persist it into
    its metadata for paper-grade audit.
    """
    phase1_root = Path(phase1_root)
    cell_id = f"{baseline}_{site}"
    key = "pass2_router" if router else "pass1"
    provenance: dict[str, Any] = {
        "cell_id": cell_id,
        "kind": key,
        "mode": None,
        "manifest_used": False,
        "kept": [],
        "rejected_non_canonical": [],
        "warnings": [],
    }
    if not phase1_root.is_dir():
        provenance["mode"] = "no_phase1_root"
        return [], provenance

    # Raw glob, split pass1 vs router by the router_learned marker.
    raw = [
        d
        for d in phase1_root.glob(f"{baseline}_*_{site}_*")
        if d.is_dir() and (("router_learned" in d.name) == router)
    ]

    manifest = load_manifest(manifest_path)
    if manifest and cell_id in manifest.get(key, {}):
        allowed = set(manifest[key][cell_id])
        kept = [d for d in raw if d.name in allowed]
        missing = sorted(allowed - {d.name for d in raw})
        provenance.update(
            mode="manifest",
            manifest_used=True,
            kept=[d.name for d in kept],
            rejected_non_canonical=[d.name for d in raw if d.name not in allowed],
        )
        if missing:
            msg = f"manifest lists {len(missing)} run(s) not present on disk: {missing}"
            provenance["warnings"].append(msg)
            logger.warning("[pass1_manifest] %s: %s", cell_id, msg)
        return sorted(kept), provenance

    # No manifest → reject non-canonical, warn on multi-run overwrite ambiguity.
    kept, rejected = [], []
    for d in raw:
        (rejected if is_non_canonical(d.name) else kept).append(d)
    provenance.update(
        mode="glob_reject_noncanonical",
        kept=[d.name for d in kept],
        rejected_non_canonical=[d.name for d in rejected],
    )
    if rejected:
        logger.warning(
            "[pass1_manifest] %s: rejected %d non-canonical run(s): %s",
            cell_id, len(rejected), [d.name for d in rejected],
        )
    if len(kept) > 1 and not router:
        msg = (
            f"{len(kept)} canonical Pass-1 runs with no manifest; per-(task,mode) "
            f"outcomes will be overwritten newest-wins with no precedence rule. Add a "
            f"manifest entry for {cell_id} for paper-grade reproducibility."
        )
        provenance["warnings"].append(msg)
        logger.warning("[pass1_manifest] %s: %s", cell_id, msg)
    return sorted(kept), provenance
