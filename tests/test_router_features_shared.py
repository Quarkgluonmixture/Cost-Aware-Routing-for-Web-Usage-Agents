"""Regression tests for the shared router feature/oracle single source of truth.

router /stress 2026-05-21:
  B-1805 (F1) difficulty parsing — VWA stores "easy"/"medium"/"hard" strings,
    int("medium") crashed Stage 1 train + was silently zeroed at serve.
  B-1806 (F2) oracle tie-break — MODES must be ascending prior cost so
    derive_oracle_label returns the *cheapest* successful mode.
  B-1807 (F6) single source — extract_50_features (train), learned_router (serve),
    and l1_archive_simulation must bind the SAME regex/MODES/oracle objects so the
    "must match" copy-paste drift can no longer occur.
"""
import sys
from pathlib import Path

from p79.policies import router_features as rf

# extract_50_features lives under scripts/ (not a package); mirror the existing
# test convention of putting scripts/analysis on sys.path.
_SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ── B-1805 F1: difficulty parsing ──────────────────────────────────────────────
def test_b1805_difficulty_to_int_maps_vwa_ordinal_strings():
    assert rf.difficulty_to_int("easy") == 0
    assert rf.difficulty_to_int("medium") == 1
    assert rf.difficulty_to_int("hard") == 2
    assert rf.difficulty_to_int("HARD") == 2          # case-insensitive
    assert rf.difficulty_to_int("  medium  ") == 1     # whitespace-tolerant


def test_b1805_difficulty_to_int_non_strings_and_unknown_do_not_crash():
    assert rf.difficulty_to_int(None) == 0             # missing → default
    assert rf.difficulty_to_int(3) == 3               # int passthrough
    assert rf.difficulty_to_int("2") == 2             # numeric string
    assert rf.difficulty_to_int("-1") == -1           # signed numeric string
    assert rf.difficulty_to_int("xyzzy") == 0         # unknown label → default
    assert rf.difficulty_to_int(True) == 0            # bool guarded (int subclass)
    assert rf.difficulty_to_int(None, default=-1) == -1


# ── B-1806 F2: oracle picks cheapest successful mode ───────────────────────────
def test_b1806_modes_ascending_prior_cost():
    idx = {m: i for i, m in enumerate(rf.MODES)}
    text_only = {"dom", "phantom_som", "phantom_text", "phantom_prompt"}
    image = {"som", "vision"}
    # every text-only mode is cheaper-ranked than every image mode
    assert max(idx[m] for m in text_only) < min(idx[m] for m in image)
    # HERO phantom_som ranks ahead of the other phantom arms
    assert idx["phantom_som"] < idx["phantom_text"]
    assert idx["phantom_som"] < idx["phantom_prompt"]
    # cost tiers consistent with the ordering
    assert rf.MODE_COST_TIER["phantom_som"] < rf.MODE_COST_TIER["som"]


def test_b1806_oracle_returns_cheapest_successful_mode():
    # som (expensive) AND phantom_som (cheap) both succeed → cheap one wins.
    assert rf.derive_oracle_label({"som": True, "phantom_som": True}) == "phantom_som"
    # dom is cheapest overall → wins whenever it succeeds.
    assert rf.derive_oracle_label({"dom": True, "som": True, "vision": True}) == "dom"
    # only an expensive mode succeeds → it is the label.
    assert rf.derive_oracle_label({"vision": True}) == "vision"
    # B-995: no success → None (NOT a "dom" fallback that collapses label semantics).
    assert rf.derive_oracle_label({"dom": False, "som": False}) is None
    assert rf.derive_oracle_label({}) is None


# ── B-1807 F6: single source of truth (identity, not equality) ─────────────────
def test_b1807_train_serve_archive_bind_same_objects():
    import extract_50_features as train  # noqa: E402
    from p79.policies import learned_router as serve

    # train (Stage 1) and serve (Pass-2 runtime) use the SAME compiled regex bank,
    # MODES list, and oracle function — drift is impossible by construction.
    assert train.INTENT_REGEX is rf.INTENT_REGEX
    assert serve.INTENT_REGEX is rf.INTENT_REGEX
    assert train.MODES is rf.MODES
    assert train.derive_oracle_label is rf.derive_oracle_label
    # back-compat aliases re-exported by serve point at the shared objects.
    assert serve.COLOR_RE is rf.COLOR_RE
    assert serve.difficulty_to_int is rf.difficulty_to_int


# ── B-1808 C1: fold_assignment covers the full routable universe ───────────────
def test_b1808_fold_assignment_covers_no_success_tasks():
    """No-success tasks (dropped from labeled training by B-995) must still resolve to
    a fold, so the Pass-2 runtime never hard-fails on a task it wasn't trained on."""
    import importlib

    import numpy as np

    mi = importlib.import_module("train_l1_router_with_mi")
    cell_ids = np.array(["B0_classifieds"] * 10)
    task_ids = np.array(list(range(10)))
    labels = np.array(["dom"] * 5 + ["phantom_som"] * 5)
    # full universe = 10 labeled + 5 no-success (ids 100..104 absent from labeled rows)
    all_cell_ids = np.array(["B0_classifieds"] * 15)
    all_task_ids = np.array(list(range(10)) + [100, 101, 102, 103, 104])

    fa = mi.generate_per_cell_fold_assignments(
        cell_ids, task_ids, labels,
        all_cell_ids=all_cell_ids, all_task_ids=all_task_ids,
        seed=42, n_splits=5,
    )["B0_classifieds"]
    for t in list(range(10)) + [100, 101, 102, 103, 104]:
        assert t in fa, f"task {t} missing from fold_assignment (C1 regression → runtime abort)"
    assert all(0 <= fk < 5 for fk in fa.values())


def test_b1808_fold_assignment_labeled_only_fallback():
    """Pre-C1 callers (no full universe supplied) still get labeled-only coverage."""
    import importlib

    import numpy as np

    mi = importlib.import_module("train_l1_router_with_mi")
    cell_ids = np.array(["B0_classifieds"] * 10)
    task_ids = np.array(list(range(10)))
    labels = np.array(["dom"] * 5 + ["phantom_som"] * 5)
    fa = mi.generate_per_cell_fold_assignments(
        cell_ids, task_ids, labels, seed=42, n_splits=5
    )["B0_classifieds"]
    assert set(fa.keys()) == set(range(10))
