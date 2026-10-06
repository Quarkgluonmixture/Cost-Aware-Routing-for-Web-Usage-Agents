"""lib/replicate_pairs — the one reader of CLEAN_PAIRS used by the band-relative producers (§537).

Bug it guards: five producers each hard-coded B0 x classifieds and two kept their own copy of
the replicate paths. The library now feeds them; if its outcome matrix ever disagrees with
per_task_sr.csv (the other source of the same per-task outcomes) one of the two has drifted.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.analysis.lib.replicate_pairs import (
    MODE_KEYS, cell_id, full_paired_cells, outcome_matrix,
)

REPO = Path(__file__).resolve().parents[1]
PER_TASK = REPO / "results/phantom_paper/per_task_sr.csv"


def test_full_paired_cells_have_every_arm():
    from scripts.analysis.lib.replicate_pairs import cell_pairs
    for b, s in full_paired_cells():
        assert set(cell_pairs(b, s)) == set(MODE_KEYS)


@pytest.mark.skipif(not PER_TASK.is_file(), reason="per_task_sr.csv is generated, not tracked")
def test_outcome_matrix_matches_per_task_sr_on_vwa_cells():
    rows = {}
    with PER_TASK.open() as fh:
        for r in csv.DictReader(fh):
            rows[(r["cell_id"], int(r["task_id"]))] = {m: int(float(r[f"sr_{m}"]) > 0) for m in MODE_KEYS}
    checked = 0
    for b, s in full_paired_cells():
        if s == "wared":
            continue                      # per_task_sr carries no WA rows
        cid = cell_id(b, s)
        m = outcome_matrix(b, s)
        assert all(rows[(cid, t)] == m[t] for t in m), cid
        checked += 1
    assert checked >= 1
