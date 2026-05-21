#!/usr/bin/env python3
"""Stage 2: Fold-local TF-IDF + global fold-local pooled MI feature selection.

A2.5 Chunk A (B-996 expanded, /stress 2026-05-18). Reads Stage 1 raw features
(`raw_features_phase1a.npz`), generates per-cell 5-fold StratifiedKFold splits, and
for each fold k:

1. Computes `pool_idx_k = all_indices \\ {union over cells of holdout_C_k}`.
2. Fits TfidfVectorizer(max_features=30, min_df=3) on pool intent texts → vectorizer_k.
3. Transforms pool to 50-dim X_pool_k (30 TF-IDF + 5 numeric + 15 binary).
4. Fits SelectKBest(mutual_info_classif, k) on X_pool_k, y_pool_k → selected_idx_k.
   k defaults to N_SELECTED=18, overridable via --k for K-sensitivity (B-1804).
5. Dumps vectorizer_fold{k}.pkl + selected_idx_fold{k}.json (incl. full mi_scores).

This is "global fold-local pooled MI" per user OOB-catch #4 — 5 unified selectors per
fold, shared across cells within that fold. Stage 3 (per-cell × per-fold LR training)
is handled by refactored train_l1_router.py in Chunk B.

Properties:
- Leak: ZERO. Selector_k never sees fold_k holdouts of any cell.
- Stability: N=~1124 per MI fit (vs N=40 in per-fold-within-cell MI).
- Sklearn pattern: equivalent to Pipeline-in-CV with selector as first step.
- MI estimator hygiene (B-1804): the 15 binary indicators are passed via
  `discrete_features` so the k-NN estimator uses discrete entropy for them rather than
  treating {0,1} as continuous (sklearn's dense-X default), which biases binary MI
  downward; TF-IDF + numeric stay continuous. score_func is functools.partial (not a
  lambda) so the selector stays picklable.

Output artifacts (in OUT_DIR):
- vectorizer_fold{k}.pkl × 5     # fitted TfidfVectorizer per fold
- selected_idx_fold{k}.json × 5  # 18-bool mask per fold
- {cell_id}_fold_assignment.json × 6  # task_id → fold_index
- stage2_summary.json            # feature stability + MI rankings
"""
from __future__ import annotations

import argparse
import functools
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results/phantom_paper/l1_router"

FOLD_SEED = 42
MI_SEED = 42  # mutual_info_classif uses random_state for k-NN entropy estimation
N_SPLITS = 5
TFIDF_MAX_FEATURES = 30
TFIDF_MIN_DF = 3
N_SELECTED = 18

SCHEMA_VERSION = "2026-05-18-a2.5-chunk-a-stage2"


def load_raw_features(npz_path: Path) -> dict[str, Any]:
    """Load Stage 1 raw features. Returns dict with arrays + metadata."""
    if not npz_path.exists():
        raise FileNotFoundError(f"Stage 1 raw features not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    meta_path = npz_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Stage 1 metadata JSON not found: {meta_path}")
    meta = json.loads(meta_path.read_text())
    return {
        "X_numeric": data["X_numeric"],
        "X_binary": data["X_binary"],
        "labels": data["labels"],
        "task_ids": data["task_ids"],
        "cell_ids": data["cell_ids"],
        "intents": list(data["intents"]),
        # C1 (B-1808): full routable universe (incl. no-success) for fold coverage;
        # fall back to labeled-only for back-compat with pre-C1 NPZ files.
        "all_task_ids": data["all_task_ids"] if "all_task_ids" in data else data["task_ids"],
        "all_cell_ids": data["all_cell_ids"] if "all_cell_ids" in data else data["cell_ids"],
        "meta": meta,
    }


def generate_per_cell_fold_assignments(
    cell_ids: np.ndarray,
    task_ids: np.ndarray,
    labels: np.ndarray,
    all_cell_ids: np.ndarray | None = None,
    all_task_ids: np.ndarray | None = None,
    seed: int = FOLD_SEED,
    n_splits: int = N_SPLITS,
) -> dict[str, dict[int, int]]:
    """Per-cell StratifiedKFold split over labeled rows + full-universe coverage.

    For each cell, partition the *labeled* tasks into n_splits folds stratified on
    oracle label (the rows Stage 3 trains on). Then — C1 (B-1808) — extend the
    assignment to the FULL routable universe (`all_cell_ids`/`all_task_ids`, incl.
    no-success tasks dropped from training by B-995) so every Pass-2 task resolves to
    a fold (runtime hard-fails on any task missing from fold_assignment, B-1640).
    No-success tasks are mapped round-robin; they were never in a training split, so
    the fold only selects which fold's LR scores them (all out-of-sample). Falls back
    to labeled-only coverage when the full universe is not supplied.

    Returns: {cell_id: {task_id: fold_index}}.
    """
    fold_assignments: dict[str, dict[int, int]] = {}
    unique_cells = sorted(set(cell_ids.tolist()))
    # C1: full routable universe per cell (incl. no-success), for fold coverage.
    full_by_cell: dict[str, list[int]] = {}
    if all_cell_ids is not None and all_task_ids is not None:
        for c, t in zip(all_cell_ids.tolist(), all_task_ids.tolist()):
            full_by_cell.setdefault(str(c), []).append(int(t))
    for cell_id in unique_cells:
        cell_mask = cell_ids == cell_id
        cell_task_ids = task_ids[cell_mask]
        cell_labels = labels[cell_mask]
        if len(cell_task_ids) == 0:
            fold_assignments[cell_id] = {}
            continue

        # StratifiedKFold may fail if some class has < n_splits members.
        # If so, drop those classes from stratification — assign them to fold 0.
        label_counter = Counter(cell_labels.tolist())
        rare_classes = {lbl for lbl, ct in label_counter.items() if ct < n_splits}
        if rare_classes:
            print(
                f"[{cell_id}] WARNING: classes {rare_classes} have <{n_splits} samples;"
                f" stratification may be approximate"
            )

        # Use rare-aware stratification: merge rare classes for stratification only
        strat_labels = np.array(
            ["__rare__" if lbl in rare_classes else lbl for lbl in cell_labels]
        )
        if len(set(strat_labels)) < 2:
            # Single class after merging — fallback to simple K-fold
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            # Use dummy labels to enable split
            dummy = np.zeros(len(cell_task_ids), dtype=int)
            dummy[: n_splits] = np.arange(n_splits)  # ensure each fold has one item
            np.random.RandomState(seed).shuffle(dummy)
            splits = list(skf.split(cell_task_ids, dummy))
        else:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            splits = list(skf.split(cell_task_ids, strat_labels))

        cell_fold_map = {}
        for fold_k, (_train_local_idx, holdout_local_idx) in enumerate(splits):
            for local_idx in holdout_local_idx:
                cell_fold_map[int(cell_task_ids[local_idx])] = fold_k

        # C1 (B-1808): extend coverage to the full routable universe — map no-success
        # (unlabeled) tasks round-robin so fold_assignment covers every Pass-2 task.
        if full_by_cell:
            labeled_tasks = set(cell_fold_map.keys())
            rr = 0
            for t in sorted(full_by_cell.get(cell_id, [])):
                if t not in labeled_tasks:
                    cell_fold_map[t] = rr % n_splits
                    rr += 1

        fold_assignments[cell_id] = cell_fold_map

    return fold_assignments


def build_pool_mask_for_fold(
    cell_ids: np.ndarray,
    task_ids: np.ndarray,
    fold_assignments: dict[str, dict[int, int]],
    target_fold_k: int,
) -> np.ndarray:
    """Build boolean mask: True = index in pool_idx_k (NOT in any cell's fold_k holdout).

    Implements global fold-local pooled rule: exclude union over all cells of fold_k holdouts.
    """
    n_total = len(task_ids)
    in_holdout = np.zeros(n_total, dtype=bool)
    for i in range(n_total):
        cell_id = str(cell_ids[i])
        tid = int(task_ids[i])
        fold_map = fold_assignments.get(cell_id, {})
        if fold_map.get(tid) == target_fold_k:
            in_holdout[i] = True
    return ~in_holdout  # pool = NOT in holdout


def fit_fold_local_tfidf(
    intents_pool: list[str],
    max_features: int = TFIDF_MAX_FEATURES,
    min_df: int = TFIDF_MIN_DF,
) -> TfidfVectorizer:
    """Fit TfidfVectorizer on pool intents only (Point 1 GPT-relay fix).

    Vocabulary + IDF estimated from pool — no holdout leak into vectorizer state.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        min_df=min_df,
        stop_words="english",
        lowercase=True,
    )
    vectorizer.fit(intents_pool)
    return vectorizer


def build_design_matrix(
    intents: list[str],
    X_numeric: np.ndarray,
    X_binary: np.ndarray,
    vectorizer: TfidfVectorizer,
) -> tuple[np.ndarray, list[str]]:
    """Combine TF-IDF + numeric + binary → full 50-dim design matrix.

    Returns (X_full, feature_names).
    """
    X_tfidf = vectorizer.transform(intents).toarray()
    n_tfidf = X_tfidf.shape[1]  # actual cols may be < max_features if vocab smaller
    X_full = np.hstack([X_tfidf, X_numeric, X_binary])
    tfidf_names = [f"tfidf_{t}" for t in vectorizer.get_feature_names_out()]
    return X_full, tfidf_names


def build_discrete_mask(n_features: int, n_binary: int) -> np.ndarray:
    """Boolean mask marking the trailing `n_binary` columns as discrete (B-1804).

    Design-matrix column order is [TF-IDF | numeric | binary] (`build_design_matrix`),
    so the binary indicator block is ALWAYS the last `n_binary` columns regardless of
    how many TF-IDF columns the fold's vocab produced. TF-IDF (continuous magnitude)
    and numeric (counts / lengths / ordinal) stay continuous. `n_binary=0` → all-False
    (legacy all-continuous behavior).
    """
    mask = np.zeros(n_features, dtype=bool)
    if n_binary > 0:
        mask[-n_binary:] = True
    return mask


def fit_pooled_mi_selector(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    k: int = N_SELECTED,
    seed: int = MI_SEED,
    n_binary: int = 0,
) -> tuple[SelectKBest, np.ndarray]:
    """Fit SelectKBest(mutual_info_classif, k) on pool data.

    B-1804: the binary indicators are passed via `discrete_features` so the k-NN MI
    estimator uses the discrete-entropy path for them instead of treating {0,1} as
    continuous (sklearn's dense-X default). The continuous treatment adds tie-breaking
    noise on the degenerate {0,1} axis and biases binary-feature MI downward, which
    systematically under-ranks binary indicators in the top-k. `n_binary=0` preserves
    the legacy all-continuous behavior. `functools.partial` (not a lambda) keeps the
    score_func picklable.

    Returns (selector, selected_idx_boolean_mask).
    """
    discrete_arg: Any
    if n_binary > 0:
        discrete_arg = build_discrete_mask(X_pool.shape[1], n_binary)
    else:
        discrete_arg = False  # sklearn dense-X default: all continuous
    selector = SelectKBest(
        score_func=functools.partial(
            mutual_info_classif, discrete_features=discrete_arg, random_state=seed
        ),
        k=k,
    )
    selector.fit(X_pool, y_pool)
    selected_idx = selector.get_support()
    return selector, selected_idx


def compute_feature_stability(
    selected_masks_per_fold: dict[int, np.ndarray],
    all_feature_names: list[str],
) -> dict[str, Any]:
    """Cross-fold feature stability — inclusion count per feature.

    Returns dict mapping feature name → {selected_in_folds: [...], inclusion_count: int}.
    """
    stability = {}
    for feat_idx, feat_name in enumerate(all_feature_names):
        selected_in = [
            k for k, mask in selected_masks_per_fold.items() if mask[feat_idx]
        ]
        stability[feat_name] = {
            "selected_in_folds": selected_in,
            "inclusion_count": len(selected_in),
        }
    # Categorize stability bands
    n_folds = len(selected_masks_per_fold)
    bands = {
        "stable_core": [
            f for f, s in stability.items() if s["inclusion_count"] >= n_folds
        ],
        "high_stability": [
            f
            for f, s in stability.items()
            if n_folds - 1 <= s["inclusion_count"] < n_folds
        ],
        "moderate": [
            f
            for f, s in stability.items()
            if 2 <= s["inclusion_count"] < n_folds - 1
        ],
        "unstable": [
            f for f, s in stability.items() if 1 <= s["inclusion_count"] < 2
        ],
        "never_selected": [
            f for f, s in stability.items() if s["inclusion_count"] == 0
        ],
    }
    return {"per_feature": stability, "bands": bands}


def run_stage2(npz_path: Path, out_dir: Path, k: int = N_SELECTED) -> dict[str, Any]:
    """Stage 2 main entry: per-fold pooled TF-IDF + MI selection across 5 folds.

    `k` = number of features SelectKBest retains per fold (default N_SELECTED=18).
    Override via `--k` for K-sensitivity sweeps (B-1804). Note: per-fold `mi_scores`
    are always dumped, so feature-selection K-sensitivity is also reconstructable
    post-hoc from a single run; only router-performance K-sensitivity needs a re-run.
    """
    print(f"\n=== Stage 2: Fold-local TF-IDF + global pooled MI ===")
    print(f"Reading Stage 1: {npz_path}")
    raw = load_raw_features(npz_path)
    X_numeric = raw["X_numeric"]
    X_binary = raw["X_binary"]
    labels = raw["labels"]
    task_ids = raw["task_ids"]
    cell_ids = raw["cell_ids"]
    intents = raw["intents"]
    n_total = len(intents)
    print(f"Pooled raw data: n_total={n_total} tasks")
    print(f"  X_numeric: {X_numeric.shape}, X_binary: {X_binary.shape}")

    if n_total == 0:
        msg = (
            "No pooled tasks available — Stage 2 cannot run without Pass-1 outcomes. "
            "Wait for Phase 1a Pass-1 fire to land, then re-run extract_50_features.py "
            "followed by this script."
        )
        print(f"\n⚠️  {msg}")
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "status": "no_data_yet",
            "message": msg,
            "n_total_tasks": 0,
        }
        (out_dir / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    # Generate per-cell fold assignments
    print(f"\nGenerating per-cell {N_SPLITS}-fold splits (seed={FOLD_SEED})...")
    fold_assignments = generate_per_cell_fold_assignments(
        cell_ids, task_ids, labels,
        all_cell_ids=raw.get("all_cell_ids"),
        all_task_ids=raw.get("all_task_ids"),
        seed=FOLD_SEED, n_splits=N_SPLITS,
    )
    for cell_id, fold_map in fold_assignments.items():
        fold_counts = Counter(fold_map.values())
        print(
            f"  {cell_id}: {len(fold_map)} tasks, fold sizes = "
            f"{dict(sorted(fold_counts.items()))}"
        )

    # For each fold k, build pool, fit TF-IDF, fit MI, dump artifacts
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_masks_per_fold: dict[int, np.ndarray] = {}
    pool_sizes_per_fold: dict[int, int] = {}
    mi_scores_per_fold: dict[int, np.ndarray] = {}
    vectorizer_vocabs_per_fold: dict[int, list[str]] = {}
    feature_names_per_fold: dict[int, list[str]] = {}

    feature_names_numeric = raw["meta"]["feature_names_numeric"]
    feature_names_binary = raw["meta"]["feature_names_binary"]

    for fold_k in range(N_SPLITS):
        print(f"\n--- Fold {fold_k} ---")
        pool_mask = build_pool_mask_for_fold(
            cell_ids, task_ids, fold_assignments, fold_k
        )
        n_pool = int(pool_mask.sum())
        pool_sizes_per_fold[fold_k] = n_pool
        print(f"  Pool size: {n_pool} / {n_total} (excluded {n_total - n_pool} holdouts)")

        # Pool-local arrays
        pool_intents = [intents[i] for i in range(n_total) if pool_mask[i]]
        pool_X_numeric = X_numeric[pool_mask]
        pool_X_binary = X_binary[pool_mask]
        pool_labels = labels[pool_mask]

        # Step 1: Fit TF-IDF on pool intents only
        vectorizer_k = fit_fold_local_tfidf(pool_intents)
        vocab_k = list(vectorizer_k.get_feature_names_out())
        vectorizer_vocabs_per_fold[fold_k] = vocab_k
        print(f"  TF-IDF vocab size: {len(vocab_k)}")

        # Step 2: Build pool design matrix (TF-IDF + numeric + binary)
        X_pool_full, tfidf_names = build_design_matrix(
            pool_intents, pool_X_numeric, pool_X_binary, vectorizer_k
        )
        feat_names_k = tfidf_names + feature_names_numeric + feature_names_binary
        feature_names_per_fold[fold_k] = feat_names_k
        print(f"  Design matrix: {X_pool_full.shape} (= {len(feat_names_k)} features)")

        # Step 3: Fit MI selector on pool (B-1804: binary block marked discrete)
        selector_k, selected_mask_k = fit_pooled_mi_selector(
            X_pool_full,
            pool_labels,
            k=k,
            seed=MI_SEED,
            n_binary=len(feature_names_binary),
        )
        mi_scores_per_fold[fold_k] = selector_k.scores_
        selected_masks_per_fold[fold_k] = selected_mask_k
        selected_names_k = [
            feat_names_k[i] for i, s in enumerate(selected_mask_k) if s
        ]
        print(f"  Top-{k} MI-selected features ({sum(selected_mask_k)} total):")
        for nm in selected_names_k[:10]:
            idx = feat_names_k.index(nm)
            print(f"    {nm}: MI={selector_k.scores_[idx]:.4f}")
        if len(selected_names_k) > 10:
            print(f"    ... +{len(selected_names_k) - 10} more")

        # Dump artifacts: vectorizer + selected_idx
        with (out_dir / f"vectorizer_fold{fold_k}.pkl").open("wb") as f:
            pickle.dump(vectorizer_k, f)
        (out_dir / f"selected_idx_fold{fold_k}.json").write_text(
            json.dumps(
                {
                    "fold_k": fold_k,
                    "n_features_total": len(feat_names_k),
                    "n_selected": int(sum(selected_mask_k)),
                    "selected_mask": selected_mask_k.tolist(),
                    "selected_names": selected_names_k,
                    "feature_names_all": feat_names_k,
                    "mi_scores": selector_k.scores_.tolist(),
                    "pool_size": n_pool,
                    "tfidf_vocab": vocab_k,
                    "mi_seed": MI_SEED,
                    "schema_version": SCHEMA_VERSION,
                },
                indent=2,
            )
        )

    # Dump per-cell fold assignments. fold_assignment now covers the FULL routable
    # universe (C1 B-1808); labeled-vs-unlabeled split disclosed for paper-grade audit.
    labeled_by_cell = Counter(str(c) for c in cell_ids.tolist())
    for cell_id, fold_map in fold_assignments.items():
        n_labeled = labeled_by_cell.get(cell_id, 0)
        (out_dir / f"{cell_id}_fold_assignment.json").write_text(
            json.dumps(
                {
                    "cell_id": cell_id,
                    "n_splits": N_SPLITS,
                    "seed": FOLD_SEED,
                    "fold_assignment": {str(tid): fk for tid, fk in fold_map.items()},
                    "n_tasks": len(fold_map),
                    "n_labeled_trained": n_labeled,
                    "n_unlabeled_routed": len(fold_map) - n_labeled,
                    "coverage_note": (
                        "fold_assignment covers the FULL routable task universe (C1 "
                        "B-1808). Only n_labeled_trained rows fit the LR; n_unlabeled_routed "
                        "(no-success) tasks are routed out-of-sample by their round-robin "
                        "fold's LR so the runtime never hard-fails on an unseen Pass-2 task."
                    ),
                    "fold_sizes": dict(Counter(fold_map.values())),
                    "schema_version": SCHEMA_VERSION,
                },
                indent=2,
            )
        )

    # Cross-fold feature stability
    # Note: feature names differ across folds because TF-IDF vocabs differ. Compute
    # stability per fold's own feature ordering; surface "stable_core" candidates from
    # non-TF-IDF features (deterministic across folds).
    nontfidf_names = feature_names_numeric + feature_names_binary
    nontfidf_stability = {}
    for fold_k, mask in selected_masks_per_fold.items():
        names_k = feature_names_per_fold[fold_k]
        for nm in nontfidf_names:
            if nm in names_k:
                idx = names_k.index(nm)
                if mask[idx]:
                    nontfidf_stability.setdefault(nm, []).append(fold_k)
    nontfidf_bands = {
        "stable_core_20_raw": [
            nm for nm in nontfidf_names if len(nontfidf_stability.get(nm, [])) >= N_SPLITS
        ],
        "high_stability_4_5": [
            nm
            for nm in nontfidf_names
            if N_SPLITS - 1 <= len(nontfidf_stability.get(nm, [])) < N_SPLITS
        ],
        "moderate_2_3": [
            nm
            for nm in nontfidf_names
            if 2 <= len(nontfidf_stability.get(nm, [])) < N_SPLITS - 1
        ],
        "unstable_1": [
            nm
            for nm in nontfidf_names
            if len(nontfidf_stability.get(nm, [])) == 1
        ],
        "never_selected": [
            nm for nm in nontfidf_names if nm not in nontfidf_stability
        ],
    }

    # Dump stage2 summary
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_total_tasks": n_total,
        "n_splits": N_SPLITS,
        "fold_seed": FOLD_SEED,
        "mi_seed": MI_SEED,
        "n_selected_per_fold": k,
        "mi_estimator": {
            "method": "sklearn.feature_selection.mutual_info_classif (k-NN entropy)",
            "n_neighbors": 3,
            "random_state": MI_SEED,
            "discrete_features": (
                f"trailing {len(feature_names_binary)} binary indicators marked "
                "discrete (B-1804); TF-IDF + numeric treated as continuous"
            ),
            "k_selected": k,
            "note_k_sensitivity": (
                "Per-fold mi_scores are dumped in selected_idx_fold{k}.json — "
                "feature-selection K-sensitivity is reconstructable post-hoc by "
                "taking top-K' from those scores without re-running Stage 2. "
                "Router-performance K-sensitivity requires Stage 3 re-run with --k."
            ),
        },
        "tfidf_max_features": TFIDF_MAX_FEATURES,
        "tfidf_min_df": TFIDF_MIN_DF,
        "pool_sizes_per_fold": pool_sizes_per_fold,
        "nontfidf_stability_inclusion": nontfidf_stability,
        "nontfidf_stability_bands": nontfidf_bands,
        "tfidf_vocab_overlap_across_folds": (
            len(
                set.intersection(
                    *(set(v) for v in vectorizer_vocabs_per_fold.values())
                )
            )
            if vectorizer_vocabs_per_fold
            else 0
        ),
        "tfidf_vocab_per_fold_size": {
            k: len(v) for k, v in vectorizer_vocabs_per_fold.items()
        },
        "fold_assignment_cells": list(fold_assignments.keys()),
        "artifacts": {
            "vectorizers": [f"vectorizer_fold{k}.pkl" for k in range(N_SPLITS)],
            "selected_idx": [f"selected_idx_fold{k}.json" for k in range(N_SPLITS)],
            "fold_assignments": [
                f"{cell_id}_fold_assignment.json"
                for cell_id in fold_assignments.keys()
            ],
        },
        "note_canonical_pattern": (
            "Global fold-local pooled MI per user OOB-catch #4 (canonical sklearn "
            "Pipeline-in-CV pattern). Selector_k trained on training side of fold k "
            "across all cells; never sees holdout tasks from any cell."
        ),
    }
    (out_dir / "stage2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nWrote stage2_summary.json + {N_SPLITS} vectorizers + {N_SPLITS} selected_idx")
    print(f"Wrote {len(fold_assignments)} fold_assignment files")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--raw-features",
        default=str(OUT_DIR / "raw_features_phase1a.npz"),
        help="Stage 1 raw features NPZ path",
    )
    ap.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for artifacts")
    ap.add_argument(
        "--k",
        type=int,
        default=N_SELECTED,
        help=f"SelectKBest k per fold (default {N_SELECTED}); vary for K-sensitivity (B-1804)",
    )
    args = ap.parse_args()

    summary = run_stage2(Path(args.raw_features), Path(args.out_dir), k=args.k)
    print("\n=== Summary ===")
    if summary.get("status") == "no_data_yet":
        print(f"Status: {summary['status']} (waiting for Phase 1a Pass-1)")
        return 0
    print(f"Pool sizes per fold: {summary['pool_sizes_per_fold']}")
    print(f"TF-IDF vocab overlap (intersection): {summary['tfidf_vocab_overlap_across_folds']}")
    print(f"NonTF-IDF stable_core (=5/5 folds): {summary['nontfidf_stability_bands']['stable_core_20_raw']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
