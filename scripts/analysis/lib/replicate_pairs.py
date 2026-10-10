"""Registered same-condition replicate pairs, by cell — one reader for `CLEAN_PAIRS`.

`CLEAN_PAIRS` in `aggregate_noise_floor_inventory.py` is the single registry of which second run
of a condition was made on purpose (label `<baseline>.<site>.<mode>`, arm a = the canonical run,
arm b = the replicate; arm a equals the manifest run for every VWA pair, checked 2026-10-07).
Before this module, five band-relative producers each hard-coded one cell and two of them kept
their own copy of the pair paths (实验笔记 §536). They now ask this module instead.

Read with `ast.literal_eval`, not import, like `lib/wa_runs.py`: the registry module has heavy
imports and this must stay side-effect free.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path

from scripts.analysis.lib.canonical_task_universe import expected_scored_ids
from scripts.analysis.lib.episode_rows import load_task_rows

REPO = Path(__file__).resolve().parents[3]
NOISE_INVENTORY = REPO / "scripts/analysis/aggregate_noise_floor_inventory.py"
MODE_KEYS = ("dom", "som", "vision", "ptext", "pprompt", "psom")
# label site token -> (site, benchmark, cell-id prefix used in products)
SITE_OF = {
    "cls": ("classifieds", "visualwebarena", "cls"),
    "red": ("reddit", "visualwebarena", "red"),
    "wared": ("reddit", "webarena", "wared"),
    "shop": ("shopping", "visualwebarena", "shop"),  # first shopping pair registered 2026-10-10 (§557)
}


@lru_cache(maxsize=1)
def clean_pairs() -> tuple[tuple[str, Path, Path], ...]:
    """(label, arm-a condition dir, arm-b condition dir) for every registered pair. Fails loud."""
    tree = ast.parse(NOISE_INVENTORY.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "CLEAN_PAIRS"
                                                for t in node.targets):
            rows = ast.literal_eval(node.value)
            out = []
            for label, a, b in rows:
                parts = label.split(".")
                if len(parts) != 3 or parts[1] not in SITE_OF or parts[2] not in MODE_KEYS:
                    raise ValueError(f"CLEAN_PAIRS label {label!r} is not <baseline>.<site>.<mode>")
                out.append((label, REPO / a, REPO / b))
            return tuple(out)
    raise ValueError(f"no CLEAN_PAIRS assignment in {NOISE_INVENTORY}")


def cell_id(baseline: str, site_key: str) -> str:
    return f"{SITE_OF[site_key][2]}_{baseline}"


def cell_pairs(baseline: str, site_key: str) -> dict[str, tuple[Path, Path]]:
    """mode key -> (arm-a condition dir, arm-b condition dir) for one cell."""
    out = {}
    for label, a, b in clean_pairs():
        bl, st, md = label.split(".")
        if (bl, st) == (baseline, site_key):
            out[md] = (a, b)
    return out


def full_paired_cells() -> list[tuple[str, str]]:
    """(baseline, site_key) cells where all six arms carry a registered replicate."""
    cells = sorted({tuple(lbl.split(".")[:2]) for lbl, _, _ in clean_pairs()})
    return [(b, s) for b, s in cells if set(cell_pairs(b, s)) == set(MODE_KEYS)]


def scored_universe(site_key: str) -> set[int]:
    site, bench, _ = SITE_OF[site_key]
    return set(expected_scored_ids(site, bench)[0])


def outcome_matrix(baseline: str, site_key: str, arm: str = "a") -> dict[int, dict[str, int]]:
    """task -> {mode: 0/1} over the canonical scored set, from one arm of the cell's pairs.

    Identity-checked (`episode_rows.load_task_rows`); every mode must cover the whole scored set.
    """
    pairs = cell_pairs(baseline, site_key)
    missing = set(MODE_KEYS) - set(pairs)
    if missing:
        raise ValueError(f"{cell_id(baseline, site_key)}: no replicate pair for {sorted(missing)}")
    scored = scored_universe(site_key)
    idx = 0 if arm == "a" else 1
    out: dict[int, dict[str, int]] = {t: {} for t in scored}
    for mode, dirs in pairs.items():
        rows = load_task_rows(dirs[idx] / "episodes")
        got = {t for t in rows if t in scored}
        if got != scored:
            raise ValueError(f"{cell_id(baseline, site_key)}/{mode}/arm {arm}: "
                             f"{len(got)} of {len(scored)} scored tasks present")
        for t in scored:
            out[t][mode] = 1 if rows[t].get("success") else 0
    return out
