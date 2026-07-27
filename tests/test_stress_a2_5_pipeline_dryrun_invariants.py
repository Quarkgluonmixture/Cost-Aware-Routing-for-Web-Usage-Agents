"""A2.5 learned-router pipeline invariants surfaced by the 2026-06-02 end-to-end
dry-run (router /stress dry-run, wt-s1-router-pipeline).

These cover gaps the existing test_stress_a2_5_*.py suite did not exercise, found while
running the full chain (Stage 1 → Stage 2 → Stage 3 → runtime → aggregate_h10_pareto) on
the real B0_classifieds Pass-1 data + an isolated archive-replay Pass-2:

  G3. Stage 1 task accounting — a labeled task whose VWA config is unreadable must be
      COUNTED (n_dropped_no_config), not silently skipped. Identity:
      n_total == n_filtered_no_success + n_dropped_no_config + n_kept.
      (Dry-run symptom: B0_classifieds reported n_kept=0 for 97 labeled tasks when the
      worktree's external/visualwebarena submodule was unchecked — a silent empty pool.)

  C1. Stage 2 fold coverage — generate_per_cell_fold_assignments must cover the FULL
      routable universe (labeled + no-success), so the runtime (predict_mode_fold_aware)
      never hard-fails on a Pass-2 task missing from fold_assignment (B-1640 / B-1808).

  E2E. The real Stage 2 + Stage 3 + runtime contract holds end to end on a full-universe
      synthetic cell: every routable task predicts a valid mode with 0
      LearnedRouterArtifactError (the train==serve [tfidf|numeric|binary] dim invariant).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

import extract_50_features as e50  # noqa: E402
from train_l1_router_with_mi import (  # noqa: E402
    N_SPLITS,
    generate_per_cell_fold_assignments,
    run_stage2,
)
from train_l1_router import (  # noqa: E402
    load_chunk_a_artifacts,
    train_one_cell,
)
from p79.policies.learned_router import (  # noqa: E402
    LearnedRouterArtifactError,
    extract_raw_features,
    predict_mode_fold_aware,
)


# ── G3 fixtures: synthetic Pass-1 run dir + VWA config tree ──────────────────


def _write_summary(ep_dir: Path, site: str, tid: int, success: bool) -> None:
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / f"{site}_task_{tid}_summary_v2.json").write_text(
        json.dumps({"task_id": tid, "success": bool(success)})
    )


def _build_synthetic_pass1(tmp_path: Path, site: str = "classifieds"):
    """One canonical Pass-1 run dir with all six mode conditions + a partial config tree.

    Tasks 0,1,2 -> dom success (label dom); task 3 -> som-only success (label som);
    task 4 -> no success (filtered); task 5 -> dom success BUT config MISSING (dropped).

    The four phantom/vision conditions fail on every task, so every oracle label above is
    unchanged — they exist because B-1887 requires a cell to carry the full six-mode menu
    before any label is derived (an absent mode would otherwise be coerced to "failed").
    Returns (phase1_root, vwa_config_root).
    """
    phase1_root = tmp_path / "phase1"
    run_dir = phase1_root / f"B0_dom_{site}_20260101_R1"
    dom_ep = run_dir / "phase1_dom_router_0" / "episodes"
    som_ep = run_dir / "phase1_som_router_0" / "episodes"
    dom_success = {0: True, 1: True, 2: True, 3: False, 4: False, 5: True}
    som_success = {0: False, 1: False, 2: False, 3: True, 4: False, 5: False}
    other_eps = [
        run_dir / f"phase1_{mode}_router_0" / "episodes"
        for mode in ("phantom_som", "phantom_text", "phantom_prompt", "vision")
    ]
    for tid in range(6):
        _write_summary(dom_ep, site, tid, dom_success[tid])
        _write_summary(som_ep, site, tid, som_success[tid])
        for ep in other_eps:
            _write_summary(ep, site, tid, False)

    vwa_root = tmp_path / "vwa_config"
    cfg_dir = vwa_root / f"test_{site}"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # Configs for tasks 0-4 only; task 5 deliberately omitted to trigger the drop path.
    for tid in range(5):
        (cfg_dir / f"{tid}.json").write_text(
            json.dumps(
                {
                    "intent": f"find the cheapest blue item number {tid}",
                    "image": None,
                    "reasoning_difficulty": "medium",
                }
            )
        )
    return phase1_root, vwa_root


def _patch_roots(monkeypatch, phase1_root, vwa_root):
    monkeypatch.setattr(e50, "PHASE1_ROOT", phase1_root)
    monkeypatch.setattr(e50, "VWA_CONFIG", vwa_root)
    # B-1896: these tests build their own synthetic run universe, so they must also
    # detach from the repo's real Pass-1 whitelist. Before that whitelist existed
    # (2026-07-27) `discover_runs` silently fell through to globbing and the tests
    # passed by accident; once it existed the synthetic runs were filtered out and
    # every one of them failed. A test that constructs its own world should say so
    # rather than depend on a file not being there.
    import p79.policies.pass1_manifest as _pm

    monkeypatch.setattr(_pm, "DEFAULT_MANIFEST", str(phase1_root / "_no_manifest.json"))


def test_b1887_absent_mode_raises_before_any_label_is_derived(tmp_path, monkeypatch):
    """A mode with NO data must hard-fail, not be coerced to 'failed on every task'.

    derive_oracle_label reads outcomes.get(mode, False), so an unrun mode silently
    becomes a failure and 'cheapest successful mode' is computed over a truncated menu.
    Empirically this let a mid-collection B2_reddit (phantom_prompt unstarted,
    phantom_som at 74/205) emit 16 plausible-but-invalid oracle labels with no warning.
    """
    phase1_root, vwa_root = _build_synthetic_pass1(tmp_path)
    _patch_roots(monkeypatch, phase1_root, vwa_root)
    run_dir = phase1_root / "B0_dom_classifieds_20260101_R1"
    shutil.rmtree(run_dir / "phase1_vision_router_0")

    with pytest.raises(ValueError, match=r"absent=\['vision'\]"):
        e50.build_cell_records("B0", "classifieds")


def test_b1887_partial_mode_raises(tmp_path, monkeypatch):
    """A mode present but covering fewer tasks than the cell is equally invalid."""
    phase1_root, vwa_root = _build_synthetic_pass1(tmp_path)
    _patch_roots(monkeypatch, phase1_root, vwa_root)
    ep = phase1_root / "B0_dom_classifieds_20260101_R1" / "phase1_vision_router_0" / "episodes"
    (ep / "classifieds_task_5_summary_v2.json").unlink()

    with pytest.raises(ValueError, match=r"partial=.*vision"):
        e50.build_cell_records("B0", "classifieds")


def test_b1887_allow_incomplete_skips_cell_and_records_reason(tmp_path, monkeypatch):
    """The opt-out skips the cell loudly — it must never silently include it."""
    phase1_root, vwa_root = _build_synthetic_pass1(tmp_path)
    _patch_roots(monkeypatch, phase1_root, vwa_root)
    run_dir = phase1_root / "B0_dom_classifieds_20260101_R1"
    shutil.rmtree(run_dir / "phase1_vision_router_0")

    rec = e50.build_cell_records("B0", "classifieds", allow_incomplete=True)
    assert rec["n_kept"] == 0
    assert rec["labels"] == []
    assert "incomplete mode menu" in rec["error"]
    assert rec["mode_completeness"]["absent_modes"] == ["vision"]
    assert rec["mode_completeness"]["cell_complete"] is False


def test_b1887_complete_cell_passes_and_records_provenance(tmp_path, monkeypatch):
    """The happy path still yields labels and now carries mode-coverage provenance."""
    phase1_root, vwa_root = _build_synthetic_pass1(tmp_path)
    _patch_roots(monkeypatch, phase1_root, vwa_root)

    # B-1904 (2026-07-27): these fixtures use a 6-task synthetic universe on
    # purpose — they assert the G3 accounting identity / B-1887 provenance, not
    # scored-universe coverage. `allow_incomplete=True` is the explicit opt-out
    # the extractor now requires for a non-paper-grade cache; without it the
    # universe gate (6 of 224 scored tasks) fires before the assertion under test.
    rec = e50.build_cell_records("B0", "classifieds", allow_incomplete=True)
    mc = rec["mode_completeness"]
    assert mc["cell_complete"] is True
    assert mc["absent_modes"] == [] and mc["partial_modes"] == []
    assert mc["n_tasks_covered"] == 6
    assert all(m["consistent"] for m in mc["by_mode"].values())
    # The synthetic cell is deliberately smaller than the canonical scored universe:
    # that is reported, never raised on (gating consumers enforce the SHA separately).
    assert mc["universe_complete"] is False


def test_g3_task_accounting_identity_all_configs_present(tmp_path, monkeypatch):
    """n_total == n_filtered_no_success + n_dropped_no_config + n_kept (no missing cfg)."""
    phase1_root, vwa_root = _build_synthetic_pass1(tmp_path)
    # Give task 5 a config too so nothing is dropped.
    (vwa_root / "test_classifieds" / "5.json").write_text(
        json.dumps({"intent": "sort by newest", "image": None, "reasoning_difficulty": "easy"})
    )
    _patch_roots(monkeypatch, phase1_root, vwa_root)

    # B-1904 (2026-07-27): these fixtures use a 6-task synthetic universe on
    # purpose — they assert the G3 accounting identity / B-1887 provenance, not
    # scored-universe coverage. `allow_incomplete=True` is the explicit opt-out
    # the extractor now requires for a non-paper-grade cache; without it the
    # universe gate (6 of 224 scored tasks) fires before the assertion under test.
    rec = e50.build_cell_records("B0", "classifieds", allow_incomplete=True)
    assert rec["n_total_tasks"] == 6
    assert rec["n_filtered_no_success"] == 1  # task 4
    assert rec["n_dropped_no_config"] == 0
    assert rec["n_kept"] == 5  # tasks 0,1,2,3,5
    assert (
        rec["n_total_tasks"]
        == rec["n_filtered_no_success"] + rec["n_dropped_no_config"] + rec["n_kept"]
    )


def test_g3_missing_config_counted_not_silently_dropped(tmp_path, monkeypatch):
    """G3 bug: a labeled task with no VWA config is COUNTED + recorded, not silent.

    Pre-fix this task vanished with no trace and n_total != filtered + kept; the whole
    cell could report n_kept=0 (submodule unchecked) and look like a benign empty pool.
    """
    phase1_root, vwa_root = _build_synthetic_pass1(tmp_path)  # task 5 cfg omitted
    _patch_roots(monkeypatch, phase1_root, vwa_root)

    # B-1904 (2026-07-27): these fixtures use a 6-task synthetic universe on
    # purpose — they assert the G3 accounting identity / B-1887 provenance, not
    # scored-universe coverage. `allow_incomplete=True` is the explicit opt-out
    # the extractor now requires for a non-paper-grade cache; without it the
    # universe gate (6 of 224 scored tasks) fires before the assertion under test.
    rec = e50.build_cell_records("B0", "classifieds", allow_incomplete=True)
    assert rec["n_total_tasks"] == 6
    assert rec["n_filtered_no_success"] == 1  # task 4 (no success)
    assert rec["n_dropped_no_config"] == 1  # task 5 (labeled dom, config missing)
    assert 5 in rec["dropped_no_config_task_ids"]
    assert rec["n_kept"] == 4  # tasks 0,1,2,3
    # The accounting identity must hold so the loss is never invisible.
    assert (
        rec["n_total_tasks"]
        == rec["n_filtered_no_success"] + rec["n_dropped_no_config"] + rec["n_kept"]
    )
    # The dropped task must NOT leak into the trained rows.
    assert 5 not in rec["task_ids"]


def test_g3_dropped_count_surfaced_in_saved_meta(tmp_path, monkeypatch):
    """The persisted companion JSON must expose n_dropped_no_config per cell (diagnosable
    from the artifact alone, not just stdout)."""
    phase1_root, vwa_root = _build_synthetic_pass1(tmp_path)
    _patch_roots(monkeypatch, phase1_root, vwa_root)

    # B-1904: synthetic 6-task universe (see the sibling fixtures above).
    extracted = e50.extract_all_cells(
        [("B0", "classifieds")], allow_incomplete=True
    )
    out = tmp_path / "raw_features_phase1a.npz"
    e50.save_npz(extracted, out)
    meta = json.loads((tmp_path / "raw_features_phase1a.json").read_text())
    cell_summary = meta["per_cell_summary"]["B0_classifieds"]
    assert cell_summary["n_dropped_no_config"] == 1
    assert 5 in cell_summary["dropped_no_config_task_ids"]


# ── C1: full-universe fold coverage (runtime never hard-fails) ───────────────


def _labeled_and_universe(n_labeled: int = 30, n_universe: int = 50, cell: str = "B0_classifieds"):
    """Labeled rows 0..n_labeled-1; universe 0..n_universe-1 (tail = no-success tasks)."""
    rng = np.random.RandomState(3)
    labels = np.array(["dom" if rng.rand() < 0.6 else "som" for _ in range(n_labeled)])
    cell_ids = np.array([cell] * n_labeled)
    task_ids = np.arange(n_labeled)
    all_cell_ids = np.array([cell] * n_universe)
    all_task_ids = np.arange(n_universe)
    return cell_ids, task_ids, labels, all_cell_ids, all_task_ids


def test_c1_fold_assignment_covers_full_universe():
    """Every routable task (labeled + no-success) resolves to exactly one valid fold."""
    cell_ids, task_ids, labels, all_cell_ids, all_task_ids = _labeled_and_universe()
    fa = generate_per_cell_fold_assignments(
        cell_ids, task_ids, labels,
        all_cell_ids=all_cell_ids, all_task_ids=all_task_ids,
        seed=42, n_splits=N_SPLITS,
    )
    fmap = fa["B0_classifieds"]
    # Coverage: every universe task id present (this is what the runtime depends on).
    assert set(fmap.keys()) == set(int(t) for t in all_task_ids)
    # Validity: every fold index in range.
    assert all(0 <= fk < N_SPLITS for fk in fmap.values())


def test_c1_no_success_tasks_resolve_to_valid_fold():
    """The no-success tail (universe minus labeled) must each get a fold — predict_mode_
    fold_aware raises LearnedRouterArtifactError on any task missing from fold_assignment
    (B-1640), so a gap here = a Pass-2 cell-run crash."""
    cell_ids, task_ids, labels, all_cell_ids, all_task_ids = _labeled_and_universe(
        n_labeled=30, n_universe=50
    )
    fa = generate_per_cell_fold_assignments(
        cell_ids, task_ids, labels,
        all_cell_ids=all_cell_ids, all_task_ids=all_task_ids,
        seed=42, n_splits=N_SPLITS,
    )
    fmap = fa["B0_classifieds"]
    no_success = set(range(30, 50))
    assert no_success.issubset(set(fmap.keys()))
    assert all(0 <= fmap[t] < N_SPLITS for t in no_success)


def test_c1_labeled_only_when_no_universe_supplied():
    """Back-compat: without all_task_ids the assignment still covers the labeled rows
    (pre-C1 NPZ files)."""
    cell_ids, task_ids, labels, _, _ = _labeled_and_universe(n_labeled=25, n_universe=25)
    fa = generate_per_cell_fold_assignments(
        cell_ids, task_ids, labels, seed=42, n_splits=N_SPLITS
    )
    fmap = fa["B0_classifieds"]
    assert set(fmap.keys()) == set(range(25))


# ── E2E: real Stage 2 + Stage 3 + runtime contract over the full universe ────

_WORD_BANK = [
    "find", "cheapest", "blue", "search", "compare", "price", "sort", "newest",
    "listing", "color", "image", "filter", "item", "post", "today", "account",
]


def _intent_for(tid: int) -> str:
    """Deterministic shared-vocabulary intent so TF-IDF min_df=3 yields a real vocab and
    train (NPZ) == serve (runtime) for the same task_id."""
    w = _WORD_BANK
    return f"{w[tid % len(w)]} {w[(tid * 3 + 1) % len(w)]} {w[(tid * 7 + 2) % len(w)]} item {tid % 5}"


def _make_full_universe_npz(out_dir: Path, cell: str = "B0_classifieds",
                            n_dom: int = 55, n_som: int = 35, n_no_success: int = 30):
    """Write a Stage-1-shaped NPZ + companion JSON for one cell with a routable universe
    strictly larger than the labeled rows (the no-success tail has fold coverage only)."""
    rng = np.random.RandomState(7)
    n_labeled = n_dom + n_som
    labels = ["dom"] * n_dom + ["som"] * n_som
    task_ids = list(range(n_labeled))
    intents = [_intent_for(t) for t in task_ids]
    X_numeric = rng.rand(n_labeled, 5)
    X_binary = (rng.rand(n_labeled, 15) < 0.3).astype(int)
    cell_ids = [cell] * n_labeled
    # Universe = labeled + no-success tail (ids n_labeled .. n_labeled+n_no_success-1)
    all_task_ids = list(range(n_labeled + n_no_success))
    all_cell_ids = [cell] * len(all_task_ids)

    feature_names_numeric = ["dom_complexity", "text_length", "tokens_input_text",
                             "intent_token_count", "reasoning_difficulty"]
    feature_names_binary = ["has_reference_image"] + sorted(e50.INTENT_REGEX.keys())

    np.savez_compressed(
        out_dir / "raw_features_phase1a.npz",
        X_numeric=X_numeric, X_binary=X_binary,
        labels=np.array(labels), task_ids=np.array(task_ids, dtype=int),
        cell_ids=np.array(cell_ids), intents=np.array(intents, dtype=object),
        all_task_ids=np.array(all_task_ids, dtype=int),
        all_cell_ids=np.array(all_cell_ids),
    )
    (out_dir / "raw_features_phase1a.json").write_text(json.dumps({
        "schema_version": "dryrun-test",
        "cells_in_pool": [cell],
        "cells_present_in_pool": [cell],
        "feature_names_numeric": feature_names_numeric,
        "feature_names_binary": feature_names_binary,
    }))
    return all_task_ids


def test_e2e_real_stage2_stage3_runtime_full_chain(tmp_path):
    """Run the REAL Stage 2 + Stage 3 on a full-universe synthetic cell, then exercise the
    runtime over EVERY routable task. Asserts the train==serve dim invariant end to end:
    0 LearnedRouterArtifactError + every prediction a valid mode.

    This is the test-form of the day-data-lands one-click guarantee: it would fail on a
    [tfidf|numeric|binary] order/dim regression, a fold-coverage gap, or a corrupt
    artifact contract — the failure modes the manual dry-run had to catch by hand.
    """
    out_dir = tmp_path
    universe = _make_full_universe_npz(out_dir)

    # Stage 2 (real)
    s2 = run_stage2(out_dir / "raw_features_phase1a.npz", out_dir)
    assert s2.get("status") != "no_data_yet"
    for k in range(N_SPLITS):
        assert (out_dir / f"vectorizer_fold{k}.pkl").exists()
        assert (out_dir / f"selected_idx_fold{k}.json").exists()
    assert (out_dir / "B0_classifieds_fold_assignment.json").exists()

    # Stage 3 (real)
    artifacts = load_chunk_a_artifacts(out_dir)
    assert artifacts["status"] == "ok"
    cell_meta = train_one_cell("B0_classifieds", artifacts, out_dir)
    assert cell_meta["cell_complete"], (
        f"cell not deployable: incomplete folds {cell_meta.get('incomplete_folds')}"
    )
    for k in range(N_SPLITS):
        assert (out_dir / f"B0_classifieds_lr_fold{k}.pkl").exists()

    # Runtime (real) over the FULL universe — the contract the runner relies on.
    cache: dict = {}
    errors = []
    modes_seen = set()
    for tid in universe:
        rf = extract_raw_features(
            intent=_intent_for(tid), has_reference_image=False,
            dom_complexity=10, text_length=200, tokens_input_text=50,
            reasoning_difficulty=1,
        )
        try:
            mode, diag = predict_mode_fold_aware(
                "B0_classifieds", tid, out_dir, cache, rf
            )
            modes_seen.add(mode)
            assert diag["fold_k_used"] in range(N_SPLITS)
        except LearnedRouterArtifactError as exc:
            errors.append((tid, str(exc)[:120]))
    assert not errors, f"runtime hard-failed on {len(errors)} routable tasks: {errors[:3]}"
    # Predictions must be valid modes (trained classes ∪ phantom_som fallback).
    assert modes_seen, "no predictions produced"
    assert modes_seen.issubset(set(e50.MODES)), f"invalid modes: {modes_seen - set(e50.MODES)}"
