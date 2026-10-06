"""`lib.wa_runs.drop_registered_replicates` — keep canonical WA runs, drop registered replicates.

The bug it guards (实验笔记 §531.6): after the 2026-09-15 replicate chain put a second
complete run next to each WA·B1 arm, producers that glob `{b}_{stem}_wa_reddit_2026*_R*`
either dropped WA·B1 silently or read the September replicate as the canonical run.
"""
from __future__ import annotations

from pathlib import Path

from scripts.analysis.lib.wa_runs import (
    drop_registered_replicates,
    registered_wa_replicate_run_ids,
)

WA = "results/webarena/phase1/"


def _wa_pairs():
    import ast
    src = Path("scripts/analysis/aggregate_noise_floor_inventory.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == "CLEAN_PAIRS" for t in node.targets):
            return [(a, b) for _l, a, b in ast.literal_eval(node.value) if b.startswith(WA)]
    raise AssertionError("CLEAN_PAIRS not found")


def test_every_registered_wa_replicate_is_dropped_and_its_canonical_kept():
    pairs = _wa_pairs()
    assert pairs, "no WA pairs registered; the fixture assumption is broken"
    for arm_a, arm_b in pairs:
        run_a = Path(arm_a).parent      # glob hits are run dirs, not condition dirs
        run_b = Path(arm_b).parent
        assert drop_registered_replicates([run_a, run_b]) == [run_a]
        # str hits (glob.glob) behave the same as Path hits (Path.glob)
        assert drop_registered_replicates([str(run_b), str(run_a)]) == [str(run_a)]


def test_vwa_replicates_are_not_in_the_wa_set():
    """The WA filter must not start dropping VWA runs: VWA replicates are handled by the
    fire validator's own reader, and a VWA canonical arm is never a WA glob hit anyway."""
    assert all("_wa_" in r for r in registered_wa_replicate_run_ids())
