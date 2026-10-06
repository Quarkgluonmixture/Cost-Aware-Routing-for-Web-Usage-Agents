"""Which WebArena run directory is the canonical one for a (baseline, mode).

WA cells are not in `run_manifest.yaml`; every producer finds its WA run by globbing
`{baseline}_{stem}_wa_reddit_2026*_R*` and assumes the glob hits exactly one directory.
That held until the 2026-09-15 local replicate chain put a second complete run next to
each WA·B1 arm (实验笔记 §529, §531.6). From then on the producers split three ways:
raise (fine), return None and silently drop WA·B1 (`len(hits) == 1 else None`), or take
`sorted(hits)[-1]` and silently read the SEPTEMBER replicate as if it were the canonical
run (`aggregate_outcome_efficiency`).

The replicate arms are already registered, once, in `CLEAN_PAIRS`
(`aggregate_noise_floor_inventory.py`); that registry is what tells "the second run we
meant to make" from "a run that should not exist". This module reads it the same way
`validate_fire_manifest.registered_replicate_run_ids` does — `ast.literal_eval`, no
import — but for the WebArena tree, which that function deliberately ignores.
"""
from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Iterable, TypeVar

REPO = Path(__file__).resolve().parents[3]
NOISE_INVENTORY = REPO / "scripts/analysis/aggregate_noise_floor_inventory.py"
_WA_PREFIX = "results/webarena/phase1/"

P = TypeVar("P", str, Path)


@lru_cache(maxsize=1)
def registered_wa_replicate_run_ids() -> frozenset[str]:
    """Run ids registered as the REPLICATE arm (arm_b) of a WebArena CLEAN_PAIR.

    Fails loud: unlike the fire validator, an unreadable registry here would let a
    replicate pass as canonical, which is the silent-substitution this module exists
    to stop.
    """
    tree = ast.parse(NOISE_INVENTORY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "CLEAN_PAIRS" for t in node.targets):
            pairs = ast.literal_eval(node.value)
            return frozenset(
                arm_b[len(_WA_PREFIX):].split("/")[0]
                for _label, _arm_a, arm_b in pairs
                if isinstance(arm_b, str) and arm_b.startswith(_WA_PREFIX))
    raise RuntimeError(f"CLEAN_PAIRS not found in {NOISE_INVENTORY}")


def drop_registered_replicates(hits: Iterable[P]) -> list[P]:
    """Filter glob hits down to non-replicate run dirs, preserving type and order.

    A hit is matched on its run-directory name, so callers may pass either the run
    dir itself or a path below it.
    """
    reps = registered_wa_replicate_run_ids()
    out = []
    for h in hits:
        parts = Path(h).parts
        if not any(p in reps for p in parts):
            out.append(h)
    return out
