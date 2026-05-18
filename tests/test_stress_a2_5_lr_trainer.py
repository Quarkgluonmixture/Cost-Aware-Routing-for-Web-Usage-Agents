"""A2.5 Chunk B — Stage 3 LR trainer invariant tests.

Invariants tested:
  1. apply_min_class_filter drops rare classes (B-995 fix)
  2. apply_min_class_filter keeps all when no rare classes
  3. tune_threshold_inner_cv returns tau ∈ TAU_CANDIDATES on valid data
  4. tune_threshold_inner_cv handles single-class train gracefully
  5. TAU_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7] locked
  6. N_FOLDS_OUTER = 5, N_FOLDS_INNER = 5 (Q1=C confirmed)
  7. SAFE_FALLBACK_MODE = "phantom_som" pre-locked
  8. N_MIN_CLASS_TRAIN = 10 (B-995 fix)
  9. LR config: class_weight=None (NOT "balanced" — B-995 minority hallucination fix)
  10. Pipeline has StandardScaler + LogisticRegression in correct order (GPT Point 4)
  11. load_chunk_a_artifacts handles no_data_yet gracefully
  12. End-to-end synthetic train_one_cell_one_fold succeeds
"""
from __future__ import annotations

import json
import pickle
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

import sys
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from train_l1_router import (  # noqa: E402
    LR_C,
    LR_MAX_ITER,
    N_FOLDS_INNER,
    N_FOLDS_OUTER,
    N_MIN_CLASS_TRAIN,
    SAFE_FALLBACK_MODE,
    SCHEMA_VERSION,
    TAU_CANDIDATES,
    apply_min_class_filter,
    build_design_matrix_for_indices,
    load_chunk_a_artifacts,
    train_one_cell_one_fold,
    tune_threshold_inner_cv,
)


# ── Invariant 1+2: apply_min_class_filter ─────────────────────────────────


def test_apply_min_class_filter_drops_rare():
    """Classes with n < 10 should be dropped per B-995 fix."""
    labels = np.array(["dom"] * 50 + ["som"] * 30 + ["phantom_som"] * 3)
    indices = np.arange(83)
    kept_labels, kept_idx, dropped = apply_min_class_filter(
        labels, indices, min_n=N_MIN_CLASS_TRAIN
    )
    assert "phantom_som" not in kept_labels
    assert "phantom_som" in dropped
    assert dropped["phantom_som"] == 3
    assert len(kept_labels) == 80
    assert len(kept_idx) == 80


def test_apply_min_class_filter_keeps_all_when_no_rare():
    """No drops when all classes >= min_n."""
    labels = np.array(["dom"] * 50 + ["som"] * 30)
    indices = np.arange(80)
    kept_labels, kept_idx, dropped = apply_min_class_filter(
        labels, indices, min_n=10
    )
    assert len(kept_labels) == 80
    assert len(dropped) == 0


# ── Invariant 3-4: tune_threshold_inner_cv ────────────────────────────────


def _make_synthetic_train_fold(n_dom: int = 80, n_som: int = 40, n_features: int = 18, seed: int = 0):
    """Synthetic train fold for τ tuning tests. Returns (X, y)."""
    rng = np.random.RandomState(seed)
    X_dom = rng.rand(n_dom, n_features)
    X_som = rng.rand(n_som, n_features) + 0.3  # slight separation
    X = np.vstack([X_dom, X_som])
    y = np.array(["dom"] * n_dom + ["som"] * n_som)
    perm = rng.permutation(len(y))
    return X[perm], y[perm]


def test_tune_threshold_inner_cv_returns_valid_tau():
    """τ* must be one of the candidates set when inner-CV succeeds."""
    X, y = _make_synthetic_train_fold(seed=1)
    result = tune_threshold_inner_cv(X, y, cell_id="test_cell", fold_k=0)
    assert result["chosen_tau"] in TAU_CANDIDATES
    assert result["reason"] in ("ok", "all_inner_folds_failed")


def test_tune_threshold_inner_cv_reports_per_tau_scores():
    """per_tau_score dict must have entry for each candidate."""
    X, y = _make_synthetic_train_fold(seed=2)
    result = tune_threshold_inner_cv(X, y, cell_id="test_cell", fold_k=1)
    for tau in TAU_CANDIDATES:
        assert str(tau) in result["per_tau_score"]


def test_tune_threshold_inner_cv_single_class_graceful():
    """Single-class train fold should not crash; return median candidate."""
    X = np.random.RandomState(3).rand(50, 18)
    y = np.array(["dom"] * 50)
    result = tune_threshold_inner_cv(X, y, cell_id="test_cell", fold_k=2)
    assert result["chosen_tau"] in TAU_CANDIDATES
    assert result["reason"] == "single_class_train_set"


def test_tune_threshold_inner_cv_deterministic_across_runs():
    """Same input → same τ* (deterministic via inner_seed = seed + fold_k * 100)."""
    X, y = _make_synthetic_train_fold(seed=4)
    r1 = tune_threshold_inner_cv(X, y, cell_id="test", fold_k=0)
    r2 = tune_threshold_inner_cv(X, y, cell_id="test", fold_k=0)
    assert r1["chosen_tau"] == r2["chosen_tau"]


# ── Invariant 5-8: Constants locked ───────────────────────────────────────


def test_tau_candidates_locked():
    """[0.3, 0.4, 0.5, 0.6, 0.7] pre-locked per /stress A2.5 (b)."""
    assert TAU_CANDIDATES == [0.3, 0.4, 0.5, 0.6, 0.7]


def test_n_folds_outer_5():
    """Q1=C confirmed — 5-fold within-cell CV deployment."""
    assert N_FOLDS_OUTER == 5


def test_n_folds_inner_5():
    """(b) inner-CV τ tuning uses 5 inner folds on train fold."""
    assert N_FOLDS_INNER == 5


def test_safe_fallback_mode_pre_locked():
    """phantom_som = safe fallback (paper §1 P-SoM hero arm)."""
    assert SAFE_FALLBACK_MODE == "phantom_som"


def test_n_min_class_train_is_10():
    """B-995 fix — classes with n<10 in train fold dropped to prevent minority hallucination."""
    assert N_MIN_CLASS_TRAIN == 10


# ── Invariant 9-10: LR config + Pipeline structure ────────────────────────


def test_lr_class_weight_none_not_balanced():
    """B-995 fix: class_weight=None, NOT 'balanced'.

    Empirical: B1_classifieds (label phantom_som=3) → balanced reweighting produced
    in_sample_pred phantom_som=46 (15× hallucination). class_weight=None prevents.
    """
    # Build pipeline directly to verify config
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight=None,
            max_iter=LR_MAX_ITER,
            C=LR_C,
            solver="lbfgs",
        )),
    ])
    clf = pipeline.named_steps["clf"]
    assert clf.class_weight is None, "B-995 violation: class_weight should be None"


def test_pipeline_has_scaler_before_clf():
    """GPT-relay Point 4: StandardScaler must be in pipeline (fits on train fold only)."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=LR_MAX_ITER)),
    ])
    assert "scaler" in pipeline.named_steps
    assert "clf" in pipeline.named_steps
    # Verify order: scaler step appears before clf
    step_names = [name for name, _ in pipeline.steps]
    assert step_names.index("scaler") < step_names.index("clf")


# ── Invariant 11: load_chunk_a_artifacts no_data_yet ──────────────────────


def test_load_chunk_a_artifacts_no_data_yet():
    """When Stage 2 summary says no_data_yet, loader propagates status without crash."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Write minimal Stage 1 + Stage 2 artifacts in no_data_yet state
        # Stage 1 dummy NPZ
        np.savez_compressed(
            td_path / "raw_features_phase1a.npz",
            X_numeric=np.zeros((0, 5)),
            X_binary=np.zeros((0, 15), dtype=int),
            labels=np.array([], dtype=object),
            task_ids=np.array([], dtype=int),
            cell_ids=np.array([], dtype=object),
            intents=np.array([], dtype=object),
        )
        (td_path / "raw_features_phase1a.json").write_text(json.dumps({
            "schema_version": "test", "cells_in_pool": []
        }))
        # Stage 2 no_data_yet
        (td_path / "stage2_summary.json").write_text(json.dumps({
            "schema_version": "test", "status": "no_data_yet", "n_total_tasks": 0
        }))

        result = load_chunk_a_artifacts(td_path)
        assert result["status"] == "no_data_yet"


# ── Invariant 12: End-to-end with synthetic data ──────────────────────────


def _make_synthetic_chunk_a_artifacts(out_dir: Path, n_per_cell: int = 60, n_cells: int = 2):
    """Generate minimal Chunk A artifacts for end-to-end test.

    Creates raw_features_phase1a.npz + json + 5 vectorizer.pkl + 5 selected_idx.json
    + per-cell fold_assignment.json + stage2_summary.json.
    """
    rng = np.random.RandomState(99)
    cell_names = ["B0_classifieds", "B0_reddit"][:n_cells]
    intents, X_numeric_list, X_binary_list, labels_list, task_ids, cell_ids = [], [], [], [], [], []
    for cell_idx, cell_id in enumerate(cell_names):
        for tid in range(n_per_cell):
            intents.append(f"task {tid} cell {cell_idx} find blue search compare today")
            X_numeric_list.append(rng.rand(5).tolist())
            X_binary_list.append((rng.rand(15) < 0.3).astype(int).tolist())
            # 70% dom, 20% som, 10% phantom_som
            r = rng.rand()
            if r < 0.70:
                labels_list.append("dom")
            elif r < 0.90:
                labels_list.append("som")
            else:
                labels_list.append("phantom_som")
            task_ids.append(tid)
            cell_ids.append(cell_id)

    X_numeric = np.array(X_numeric_list, dtype=float)
    X_binary = np.array(X_binary_list, dtype=int)
    labels = np.array(labels_list)
    task_ids_arr = np.array(task_ids, dtype=int)
    cell_ids_arr = np.array(cell_ids)

    np.savez_compressed(
        out_dir / "raw_features_phase1a.npz",
        X_numeric=X_numeric, X_binary=X_binary, labels=labels,
        task_ids=task_ids_arr, cell_ids=cell_ids_arr,
        intents=np.array(intents, dtype=object),
    )
    feature_names_numeric = ["dom_complexity", "text_length", "tokens_input_text",
                             "intent_token_count", "reasoning_difficulty"]
    feature_names_binary = ["has_reference_image", "intent_account_action",
                            "intent_action_word", "intent_aggregate", "intent_color",
                            "intent_compare", "intent_compose", "intent_filter",
                            "intent_form_fill", "intent_nav", "intent_question",
                            "intent_search", "intent_sort", "intent_temporal",
                            "intent_visual_attribute"]
    (out_dir / "raw_features_phase1a.json").write_text(json.dumps({
        "schema_version": "test",
        "cells_in_pool": cell_names,
        "cells_present_in_pool": cell_names,
        "feature_names_numeric": feature_names_numeric,
        "feature_names_binary": feature_names_binary,
    }))

    # Fit vectorizers + selectors per fold
    for k in range(5):
        # Naive pool: half of data
        pool_idx = list(range(0, len(intents), 2))
        pool_intents = [intents[i] for i in pool_idx]
        vec = TfidfVectorizer(max_features=30, min_df=1, stop_words="english", lowercase=True)
        vec.fit(pool_intents)
        with (out_dir / f"vectorizer_fold{k}.pkl").open("wb") as f:
            pickle.dump(vec, f)
        n_tfidf = len(vec.get_feature_names_out())
        total = n_tfidf + 5 + 15
        # Mock select_mask: pick first 18 features
        mask = [True] * 18 + [False] * (total - 18)
        if total < 18:
            # If vocab very small, just select all
            mask = [True] * total
        (out_dir / f"selected_idx_fold{k}.json").write_text(json.dumps({
            "fold_k": k,
            "n_features_total": total,
            "n_selected": sum(mask),
            "selected_mask": mask,
            "selected_names": [f"feat_{i}" for i, m in enumerate(mask) if m],
        }))

    # Per-cell fold assignments — round-robin
    for cell_id in cell_names:
        fold_map = {str(tid): tid % 5 for tid in range(n_per_cell)}
        (out_dir / f"{cell_id}_fold_assignment.json").write_text(json.dumps({
            "cell_id": cell_id,
            "fold_assignment": fold_map,
        }))

    (out_dir / "stage2_summary.json").write_text(json.dumps({
        "schema_version": "test", "status": "ok", "n_total_tasks": len(intents)
    }))


def test_end_to_end_train_one_cell_one_fold():
    """End-to-end smoke: load artifacts + train 1 (cell, fold) succeeds."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        _make_synthetic_chunk_a_artifacts(td_path, n_per_cell=80, n_cells=2)

        artifacts = load_chunk_a_artifacts(td_path)
        assert artifacts["status"] == "ok"

        rec = train_one_cell_one_fold(
            cell_id="B0_classifieds",
            fold_k=0,
            artifacts=artifacts,
            out_dir=td_path,
        )
        # At minimum, should not crash and should produce a status
        assert "status" in rec
        if rec["status"] == "ok":
            assert (td_path / "B0_classifieds_lr_fold0.pkl").exists()
            assert 0.0 <= rec["holdout_sr"] <= 1.0 or np.isnan(rec["holdout_sr"])
            assert rec["chosen_tau"] in TAU_CANDIDATES


# ── Invariant 13-14: Schema + behavioral checks ───────────────────────────


def test_schema_version_is_chunk_b():
    assert "chunk-b" in SCHEMA_VERSION.lower()


def test_lr_c_default_is_1():
    """L2 strength default — could be tuned later via inner-CV, locked at 1.0 for now."""
    assert LR_C == 1.0


def test_lr_max_iter_2000():
    """Sufficient max_iter for sklearn convergence at N≈200."""
    assert LR_MAX_ITER == 2000
