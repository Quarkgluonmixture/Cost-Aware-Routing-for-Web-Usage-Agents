#!/usr/bin/env python3
"""Post-hoc offline model/feature sweep for the B0_classifieds router.

This producer is intentionally isolated from the canonical learned-router and the
2026-07-15 offline-replay artifacts.  It reuses those artifacts read-only, preserves
their outer folds, min-class filter, inner-CV threshold semantics, and P-SoM signal
fallback, then varies only classifier family and feature breadth.

Nothing emitted here is preregistered or H10-eligible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
import sys
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Direct ``python scripts/analysis/...`` execution puts only this script's
# directory on sys.path.  Add the repository root before importing sibling
# analysis modules (the package itself may be editable-installed, ``scripts`` is not).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.analysis.router_offline_replay import (
    DISPLAY_MODES,
    FALLBACK_MODE,
    MODE_LABELS,
    build_offline_raw_features,
    collect_cell_outcomes,
    fold_map_sha256,
    load_paper_grade_entries,
    policy_metrics,
    reference_points,
    sha256_file,
)
from scripts.analysis.train_l1_router import (
    INNER_CV_SEED,
    LR_C,
    LR_MAX_ITER,
    N_FOLDS_INNER,
    N_MIN_CLASS_TRAIN,
    TAU_CANDIDATES,
    apply_min_class_filter,
)
from scripts.analysis.train_l1_router_with_mi import build_pool_mask_for_fold


REPO = _REPO_ROOT
SOURCE_DIR = REPO / "results/phantom_paper/l1_router_offline_20260715"
DEFAULT_OUT_DIR = REPO / "results/phantom_paper/l1_router_sweep_20260715"
DEFAULT_REPORT = (
    REPO / "docs/checkpoints/codex_outputs/router_model_sweep_2026-07-15.md"
)
RUN_MANIFEST = REPO / "results/phantom_paper/run_manifest.yaml"
PHASE1_ROOT = REPO / "results/visualwebarena/phase1"
CANONICAL_DIR = REPO / "results/phantom_paper/l1_router"

CELL_ID = "B0_classifieds"
N_OUTER_FOLDS = 5
SEED = 42
TFIDF_MIN_DF = 3
RANDOM_FOREST_TREES = 300
MLP_HIDDEN_UNITS = 32
CALIBRATION_FOLDS = 3

SCHEMA_VERSION = "2026-07-15-posthoc-router-model-sweep-v1"
BANNER = (
    "POST-HOC EXPLORATORY MODEL SWEEP — NOT the preregistered router, "
    "NOT H10-eligible"
)
ARTIFACT_STATUS = BANNER

MODEL_ORDER = [
    "logistic_regression",
    "gradient_boosting",
    "random_forest",
    "linear_svc_calibrated",
    "mlp_1hidden",
]
MODEL_LABELS = {
    "logistic_regression": "LogisticRegression",
    "gradient_boosting": "GradientBoostingClassifier",
    "random_forest": "RandomForest",
    "linear_svc_calibrated": "LinearSVC + calibration",
    "mlp_1hidden": "MLPClassifier (1 hidden)",
}
FEATURE_ORDER = ["locked_mi18", "full_50", "full_tfidf"]
FEATURE_LABELS = {
    "locked_mi18": "① locked MI-18",
    "full_50": "② all 50, no MI",
    "full_tfidf": "③ raw-20 + full fold-local TF-IDF",
}


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(payload: Any) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(canonical.encode("utf-8"))


def _json_native(value: Any) -> Any:
    """Recursively convert NumPy scalars/arrays before strict JSON serialization."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_native(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_native(item) for item in value]
    return value


def _make_estimator(model_id: str, seed: int = SEED) -> Pipeline:
    """Return a deterministic probability-producing classifier pipeline.

    StandardScaler is kept for every family so the preprocessing contract is fixed
    across the grid.  It is immaterial to threshold-order tree splits but required by
    LR, LinearSVC, and MLP.
    """
    if model_id == "logistic_regression":
        classifier: Any = LogisticRegression(
            class_weight=None,
            max_iter=LR_MAX_ITER,
            C=LR_C,
            solver="lbfgs",
        )
    elif model_id == "gradient_boosting":
        classifier = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            subsample=1.0,
            random_state=seed,
        )
    elif model_id == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=RANDOM_FOREST_TREES,
            criterion="gini",
            max_features="sqrt",
            class_weight=None,
            random_state=seed,
            n_jobs=-1,
        )
    elif model_id == "linear_svc_calibrated":
        classifier = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.0,
                class_weight=None,
                dual="auto",
                max_iter=5000,
                random_state=seed,
            ),
            method="sigmoid",
            cv=CALIBRATION_FOLDS,
            n_jobs=1,
        )
    elif model_id == "mlp_1hidden":
        classifier = MLPClassifier(
            hidden_layer_sizes=(MLP_HIDDEN_UNITS,),
            activation="relu",
            solver="lbfgs",
            alpha=0.0001,
            max_iter=2000,
            random_state=seed,
        )
    else:
        raise ValueError(f"Unknown model grid id: {model_id}")
    return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])


def _fit_with_warning_count(estimator: Pipeline, X: np.ndarray, y: np.ndarray) -> int:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        estimator.fit(X, y)
    return sum(issubclass(w.category, ConvergenceWarning) for w in caught)


def tune_threshold_inner_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_id: str,
    fold_k: int,
) -> dict[str, Any]:
    """Generalized copy of the locked Stage-3 τ protocol.

    It preserves the same 5-fold stratified inner split, fold-specific seed, inner
    min-class=2 filter, mode-match objective, candidate set, strict ``> τ`` routing
    rule, P-SoM fallback, and higher-τ tie break.  Only the estimator factory varies.
    """
    counts = Counter(y_train.tolist())
    if len(counts) < 2:
        return {
            "chosen_tau": float(TAU_CANDIDATES[len(TAU_CANDIDATES) // 2]),
            "per_tau_score": {str(t): None for t in TAU_CANDIDATES},
            "n_inner_folds_used": 0,
            "reason": "single_class_train_set",
            "convergence_warning_count": 0,
        }

    rare_for_inner = {label for label, n in counts.items() if n < N_FOLDS_INNER}
    strat = np.array(
        ["__rare__" if label in rare_for_inner else label for label in y_train]
    )
    inner_seed = INNER_CV_SEED + fold_k * 100
    try:
        splitter = StratifiedKFold(
            n_splits=N_FOLDS_INNER, shuffle=True, random_state=inner_seed
        )
        splits = list(splitter.split(X_train, strat))
    except ValueError:
        return {
            "chosen_tau": float(TAU_CANDIDATES[len(TAU_CANDIDATES) // 2]),
            "per_tau_score": {str(t): None for t in TAU_CANDIDATES},
            "n_inner_folds_used": 0,
            "reason": "inner_cv_split_failed",
            "convergence_warning_count": 0,
        }

    scores: dict[float, list[float]] = {tau: [] for tau in TAU_CANDIDATES}
    convergence_warnings = 0
    n_inner_used = 0
    for inner_train_idx, inner_holdout_idx in splits:
        inner_y = y_train[inner_train_idx]
        kept_y, kept_relative_idx, _ = apply_min_class_filter(
            inner_y, inner_train_idx, min_n=2
        )
        if len(set(kept_y)) < 2:
            continue
        keep_mask = np.isin(inner_train_idx, kept_relative_idx)
        estimator = _make_estimator(model_id)
        convergence_warnings += _fit_with_warning_count(
            estimator, X_train[inner_train_idx][keep_mask], kept_y
        )
        probabilities = estimator.predict_proba(X_train[inner_holdout_idx])
        max_probabilities = probabilities.max(axis=1)
        argmax_modes = estimator.classes_[probabilities.argmax(axis=1)]
        for tau in TAU_CANDIDATES:
            decided = np.where(
                max_probabilities > tau, argmax_modes, FALLBACK_MODE
            )
            scores[tau].append(float((decided == y_train[inner_holdout_idx]).mean()))
        n_inner_used += 1

    means = {
        tau: (float(np.mean(values)) if values else None)
        for tau, values in scores.items()
    }
    valid = {tau: score for tau, score in means.items() if score is not None}
    if not valid:
        chosen = float(TAU_CANDIDATES[len(TAU_CANDIDATES) // 2])
        reason = "all_inner_folds_failed"
    else:
        max_score = max(valid.values())
        chosen = float(
            max(tau for tau, score in valid.items() if abs(score - max_score) < 1e-9)
        )
        reason = "ok"
    return {
        "chosen_tau": chosen,
        "per_tau_score": {str(tau): means[tau] for tau in TAU_CANDIDATES},
        "n_inner_folds_used": n_inner_used,
        "reason": reason,
        "convergence_warning_count": convergence_warnings,
    }


def _load_raw_pool(source_dir: Path) -> dict[str, Any]:
    npz_path = source_dir / "raw_features_phase1a.npz"
    meta_path = source_dir / "raw_features_phase1a.json"
    raw = np.load(npz_path, allow_pickle=True)
    return {
        "X_numeric": raw["X_numeric"],
        "X_binary": raw["X_binary"],
        "labels": raw["labels"],
        "task_ids": raw["task_ids"],
        "cell_ids": raw["cell_ids"],
        "intents": list(raw["intents"]),
        "meta": json.loads(meta_path.read_text()),
        "npz_path": npz_path,
        "meta_path": meta_path,
    }


def _load_fold_maps(
    source_dir: Path, cell_ids: Iterable[str]
) -> dict[str, dict[int, int]]:
    result: dict[str, dict[int, int]] = {}
    for cell_id in cell_ids:
        payload = json.loads(
            (source_dir / f"{cell_id}_fold_assignment.json").read_text()
        )
        result[cell_id] = {
            int(task_id): int(fold)
            for task_id, fold in payload["fold_assignment"].items()
        }
    return result


def _load_capped_fold_assets(
    source_dir: Path,
) -> tuple[dict[int, TfidfVectorizer], dict[int, np.ndarray]]:
    vectorizers: dict[int, TfidfVectorizer] = {}
    masks: dict[int, np.ndarray] = {}
    for fold_k in range(N_OUTER_FOLDS):
        with (source_dir / f"vectorizer_fold{fold_k}.pkl").open("rb") as handle:
            vectorizers[fold_k] = pickle.load(handle)
        selected = json.loads(
            (source_dir / f"selected_idx_fold{fold_k}.json").read_text()
        )
        masks[fold_k] = np.array(selected["selected_mask"], dtype=bool)
        if int(masks[fold_k].sum()) != 18:
            raise ValueError(f"Fold {fold_k} source MI mask is not 18-dimensional")
    return vectorizers, masks


def _fit_full_vocab_vectorizers(
    raw: dict[str, Any], fold_maps: dict[str, dict[int, int]]
) -> tuple[dict[int, TfidfVectorizer], dict[int, int]]:
    vectorizers: dict[int, TfidfVectorizer] = {}
    pool_sizes: dict[int, int] = {}
    for fold_k in range(N_OUTER_FOLDS):
        pool_mask = build_pool_mask_for_fold(
            raw["cell_ids"], raw["task_ids"], fold_maps, fold_k
        )
        intents_pool = [
            raw["intents"][idx] for idx in np.where(pool_mask)[0].tolist()
        ]
        vectorizer = TfidfVectorizer(
            max_features=None,
            min_df=TFIDF_MIN_DF,
            stop_words="english",
            lowercase=True,
        )
        vectorizer.fit(intents_pool)
        vectorizers[fold_k] = vectorizer
        pool_sizes[fold_k] = len(intents_pool)
    return vectorizers, pool_sizes


def _matrix_from_raw(
    intents: list[str],
    numeric: np.ndarray,
    binary: np.ndarray,
    vectorizer: TfidfVectorizer,
    feature_id: str,
    selected_mask: np.ndarray | None,
) -> np.ndarray:
    X_tfidf = vectorizer.transform(intents).toarray()
    X_full = np.hstack([X_tfidf, numeric, binary])
    if feature_id == "locked_mi18":
        if selected_mask is None or X_full.shape[1] != len(selected_mask):
            raise ValueError(
                f"MI-mask mismatch: design={X_full.shape[1]}, "
                f"mask={None if selected_mask is None else len(selected_mask)}"
            )
        return X_full[:, selected_mask]
    return X_full


def _collect_all_task_raw_features(
    entries: dict[str, dict[str, Any]],
    outcomes: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    intents: list[str] = []
    numeric: list[np.ndarray] = []
    binary: list[np.ndarray] = []
    provenance: dict[str, Any] = {}
    task_ids = sorted(outcomes)
    for task_id in task_ids:
        raw, row_provenance = build_offline_raw_features(
            entries, PHASE1_ROOT, "classifieds", task_id
        )
        intents.append(raw["intent_text"])
        numeric.append(raw["numeric"])
        binary.append(raw["binary"])
        provenance[str(task_id)] = row_provenance
    return {
        "task_ids": np.array(task_ids, dtype=int),
        "intents": intents,
        "numeric": np.vstack(numeric),
        "binary": np.vstack(binary),
        "provenance": provenance,
    }


def _validate_raw_train_serve_parity(
    raw: dict[str, Any], all_tasks: dict[str, Any]
) -> dict[str, Any]:
    task_row = {int(task_id): idx for idx, task_id in enumerate(all_tasks["task_ids"])}
    checked = 0
    for idx in np.where(raw["cell_ids"] == CELL_ID)[0]:
        task_id = int(raw["task_ids"][idx])
        all_idx = task_row[task_id]
        if raw["intents"][idx] != all_tasks["intents"][all_idx]:
            raise ValueError(f"Intent train/replay drift for task {task_id}")
        if not np.array_equal(raw["X_numeric"][idx], all_tasks["numeric"][all_idx]):
            raise ValueError(f"Numeric train/replay drift for task {task_id}")
        if not np.array_equal(raw["X_binary"][idx], all_tasks["binary"][all_idx]):
            raise ValueError(f"Binary train/replay drift for task {task_id}")
        checked += 1
    return {"n_labeled_rows_checked": checked, "exact_match": True}


def _fold_reference(
    outcomes: dict[int, dict[str, dict[str, Any]]], task_ids: list[int]
) -> dict[str, Any]:
    refs = reference_points(outcomes, task_ids)
    som = refs["single_modes"]["som"]
    return {
        "som_n_success": som["n_success"],
        "som_success_rate_pct": som["success_rate_pct"],
        "som_mean_total_billed_cost_usd": som["mean_total_billed_cost_usd"],
        "best_single_mode": refs["best_single_mode"]["mode"],
        "best_single_success_rate_pct": refs["best_single_mode"]["success_rate_pct"],
    }


def _run_combination(
    model_id: str,
    feature_id: str,
    raw: dict[str, Any],
    all_tasks: dict[str, Any],
    outcomes: dict[int, dict[str, dict[str, Any]]],
    fold_map: dict[int, int],
    capped_vectorizers: dict[int, TfidfVectorizer],
    selected_masks: dict[int, np.ndarray],
    full_vectorizers: dict[int, TfidfVectorizer],
    full_refs: dict[str, Any],
) -> dict[str, Any]:
    cell_global_idx = np.where(raw["cell_ids"] == CELL_ID)[0]
    cell_task_ids = raw["task_ids"][cell_global_idx]
    cell_labels = raw["labels"][cell_global_idx]
    full_task_ids = all_tasks["task_ids"]

    selected_modes: dict[int, str] = {}
    task_records: list[dict[str, Any]] = []
    fold_records: dict[str, Any] = {}
    dimensions_per_fold: dict[str, int] = {}
    total_convergence_warnings = 0

    for fold_k in range(N_OUTER_FOLDS):
        train_local = np.array(
            [idx for idx, task_id in enumerate(cell_task_ids) if fold_map[int(task_id)] != fold_k],
            dtype=int,
        )
        train_global = cell_global_idx[train_local]
        y_train = cell_labels[train_local]
        y_kept, kept_global, dropped_classes = apply_min_class_filter(
            y_train, train_global, min_n=N_MIN_CLASS_TRAIN
        )
        if len(set(y_kept)) < 2:
            raise RuntimeError(
                f"{model_id}/{feature_id}/fold{fold_k}: min-class filter left "
                f"{len(set(y_kept))} class(es)"
            )

        if feature_id in {"locked_mi18", "full_50"}:
            vectorizer = capped_vectorizers[fold_k]
            mask = selected_masks[fold_k] if feature_id == "locked_mi18" else None
        else:
            vectorizer = full_vectorizers[fold_k]
            mask = None

        X_train = _matrix_from_raw(
            [raw["intents"][idx] for idx in kept_global],
            raw["X_numeric"][kept_global],
            raw["X_binary"][kept_global],
            vectorizer,
            feature_id,
            mask,
        )
        dimensions_per_fold[str(fold_k)] = int(X_train.shape[1])
        tau_result = tune_threshold_inner_cv(X_train, y_kept, model_id, fold_k)
        if tau_result["reason"] != "ok":
            raise RuntimeError(
                f"{model_id}/{feature_id}/fold{fold_k}: τ tuning failed: "
                f"{tau_result['reason']}"
            )

        estimator = _make_estimator(model_id)
        outer_warning_count = _fit_with_warning_count(estimator, X_train, y_kept)
        total_convergence_warnings += (
            tau_result["convergence_warning_count"] + outer_warning_count
        )

        holdout_indices = np.array(
            [idx for idx, task_id in enumerate(full_task_ids) if fold_map[int(task_id)] == fold_k],
            dtype=int,
        )
        holdout_task_ids = [int(full_task_ids[idx]) for idx in holdout_indices]
        X_holdout = _matrix_from_raw(
            [all_tasks["intents"][idx] for idx in holdout_indices],
            all_tasks["numeric"][holdout_indices],
            all_tasks["binary"][holdout_indices],
            vectorizer,
            feature_id,
            mask,
        )
        probabilities = estimator.predict_proba(X_holdout)
        if not np.isfinite(probabilities).all():
            raise ValueError(f"{model_id}/{feature_id}/fold{fold_k}: non-finite probability")
        max_probabilities = probabilities.max(axis=1)
        argmax_modes = estimator.classes_[probabilities.argmax(axis=1)]
        tau = float(tau_result["chosen_tau"])
        decided_modes = np.where(
            max_probabilities > tau, argmax_modes, FALLBACK_MODE
        )

        fold_policy: dict[int, str] = {}
        for task_id, argmax, max_probability, decided in zip(
            holdout_task_ids,
            argmax_modes.tolist(),
            max_probabilities.tolist(),
            decided_modes.tolist(),
        ):
            selected_mode = str(decided)
            selected_modes[task_id] = selected_mode
            fold_policy[task_id] = selected_mode
            row = outcomes[task_id][selected_mode]
            task_records.append(
                {
                    "task_id": task_id,
                    "fold_k": fold_k,
                    "tau": tau,
                    "argmax_mode": str(argmax),
                    "max_probability": float(max_probability),
                    "signal_strength_fallback_fired": bool(max_probability <= tau),
                    "selected_mode": selected_mode,
                    "success": bool(row["success"]),
                    "total_billed_cost_usd": float(row["cost_usd"]),
                    "outcome_summary_path": row["summary_path"],
                }
            )

        fold_metric = policy_metrics(outcomes, fold_policy)
        fold_ref = _fold_reference(outcomes, holdout_task_ids)
        fold_records[str(fold_k)] = {
            **fold_metric,
            **fold_ref,
            "delta_vs_fold_som_pp": (
                fold_metric["success_rate_pct"] - fold_ref["som_success_rate_pct"]
            ),
            "tau": tau,
            "fallback_count": sum(max_probabilities <= tau),
            "fallback_rate": float(np.mean(max_probabilities <= tau)),
            "train_n_before_min_class": int(len(y_train)),
            "train_n_after_min_class": int(len(y_kept)),
            "train_label_distribution": dict(Counter(y_kept.tolist())),
            "dropped_classes": dropped_classes,
            "tau_tuning": tau_result,
            "outer_fit_convergence_warning_count": outer_warning_count,
        }

    if set(selected_modes) != set(outcomes):
        raise AssertionError(
            f"{model_id}/{feature_id}: OOF coverage {len(selected_modes)} != {len(outcomes)}"
        )
    metric = policy_metrics(outcomes, selected_modes)
    best = full_refs["best_single_mode"]
    oracle = full_refs["six_mode_oracle_ceiling"]
    som_success = {
        task_id: bool(outcomes[task_id]["som"]["success"]) for task_id in outcomes
    }
    router_success = {
        task_id: bool(outcomes[task_id][selected_modes[task_id]]["success"])
        for task_id in outcomes
    }
    paired = {
        "both_success": sum(router_success[t] and som_success[t] for t in outcomes),
        "router_only_success": sum(router_success[t] and not som_success[t] for t in outcomes),
        "som_only_success": sum(not router_success[t] and som_success[t] for t in outcomes),
        "neither_success": sum(not router_success[t] and not som_success[t] for t in outcomes),
    }
    fold_srs = [fold_records[str(k)]["success_rate_pct"] for k in range(N_OUTER_FOLDS)]
    task_records.sort(key=lambda row: row["task_id"])
    return {
        "model_id": model_id,
        "model": MODEL_LABELS[model_id],
        "feature_id": feature_id,
        "feature_set": FEATURE_LABELS[feature_id],
        "dimensions_per_fold": dimensions_per_fold,
        **metric,
        "delta_vs_best_single_pp": metric["success_rate_pct"] - best["success_rate_pct"],
        "delta_vs_lr_baseline_pp": None,
        "oracle_gap_pp": oracle["success_rate_pct"] - metric["success_rate_pct"],
        "fallback_count": sum(
            row["signal_strength_fallback_fired"] for row in task_records
        ),
        "fallback_rate": float(
            np.mean(
                [row["signal_strength_fallback_fired"] for row in task_records]
            )
        ),
        "thresholds_per_fold": {
            str(k): fold_records[str(k)]["tau"] for k in range(N_OUTER_FOLDS)
        },
        "fold_success_rate_pct_values": fold_srs,
        "fold_success_rate_pct_mean_unweighted": float(np.mean(fold_srs)),
        "fold_success_rate_pct_sd_unweighted": float(np.std(fold_srs, ddof=0)),
        "fold_success_rate_pct_min": float(min(fold_srs)),
        "fold_success_rate_pct_max": float(max(fold_srs)),
        "paired_vs_som": paired,
        "convergence_warning_count": total_convergence_warnings,
        "fold_records": fold_records,
        "task_records": task_records,
    }


def _model_specs() -> dict[str, Any]:
    return {
        "shared_preprocessing": "StandardScaler fitted inside every train split",
        "logistic_regression": {
            "classifier": "LogisticRegression",
            "C": LR_C,
            "solver": "lbfgs",
            "max_iter": LR_MAX_ITER,
            "class_weight": None,
        },
        "gradient_boosting": {
            "classifier": "GradientBoostingClassifier",
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 3,
            "subsample": 1.0,
            "random_state": SEED,
        },
        "random_forest": {
            "classifier": "RandomForestClassifier",
            "n_estimators": RANDOM_FOREST_TREES,
            "criterion": "gini",
            "max_features": "sqrt",
            "class_weight": None,
            "random_state": SEED,
        },
        "linear_svc_calibrated": {
            "classifier": "CalibratedClassifierCV(LinearSVC)",
            "C": 1.0,
            "method": "sigmoid",
            "calibration_cv": CALIBRATION_FOLDS,
            "random_state": SEED,
        },
        "mlp_1hidden": {
            "classifier": "MLPClassifier",
            "hidden_layer_sizes": [MLP_HIDDEN_UNITS],
            "solver": "lbfgs",
            "alpha": 0.0001,
            "max_iter": 2000,
            "random_state": SEED,
        },
    }


def _feature_specs(
    dimensions: dict[str, dict[str, int]], full_pool_sizes: dict[int, int]
) -> dict[str, Any]:
    return {
        "locked_mi18": {
            "definition": (
                "Read-only reuse of the 2026-07-15 fold-local top-30 TF-IDF + raw-20 "
                "design and its fold-local pooled MI top-18 masks"
            ),
            "dimensions_per_fold": dimensions["locked_mi18"],
        },
        "full_50": {
            "definition": (
                "Read-only reuse of the same fold-local top-30 TF-IDF + raw-20 "
                "design, retaining all 50 columns without MI selection"
            ),
            "dimensions_per_fold": dimensions["full_50"],
        },
        "full_tfidf": {
            "definition": (
                "Raw-20 plus fold-local TF-IDF fitted on the identical pooled outer-"
                "training side with max_features=None, min_df=3. This replaces the "
                "capped top-30 block rather than duplicating it."
            ),
            "dimensions_per_fold": dimensions["full_tfidf"],
            "pool_sizes_per_fold": {str(k): v for k, v in full_pool_sizes.items()},
        },
    }


def run_sweep(source_dir: Path) -> dict[str, Any]:
    raw = _load_raw_pool(source_dir)
    source_cells = list(raw["meta"]["cells_in_pool"])
    fold_maps = _load_fold_maps(source_dir, source_cells)
    fold_map = fold_maps[CELL_ID]
    capped_vectorizers, selected_masks = _load_capped_fold_assets(source_dir)
    full_vectorizers, full_pool_sizes = _fit_full_vocab_vectorizers(raw, fold_maps)

    entries_by_cell = load_paper_grade_entries(RUN_MANIFEST, [CELL_ID])
    entries = entries_by_cell[CELL_ID]
    outcomes, outcome_provenance = collect_cell_outcomes(
        CELL_ID, entries, PHASE1_ROOT
    )
    if set(fold_map) != set(outcomes):
        raise ValueError("Source fold map does not cover the B0_classifieds universe")
    all_tasks = _collect_all_task_raw_features(entries, outcomes)
    raw_parity = _validate_raw_train_serve_parity(raw, all_tasks)
    full_refs = reference_points(outcomes)

    combinations: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        for feature_id in FEATURE_ORDER:
            print(f"[{model_id} × {feature_id}]", flush=True)
            combinations.append(
                _run_combination(
                    model_id,
                    feature_id,
                    raw,
                    all_tasks,
                    outcomes,
                    fold_map,
                    capped_vectorizers,
                    selected_masks,
                    full_vectorizers,
                    full_refs,
                )
            )

    lr_baseline = next(
        row
        for row in combinations
        if row["model_id"] == "logistic_regression"
        and row["feature_id"] == "locked_mi18"
    )
    source_replay_path = source_dir / "router_offline_replay.json"
    source_replay = json.loads(source_replay_path.read_text())
    source_lr = source_replay["cells"][CELL_ID]["offline_routed"]
    source_tau = source_replay["cells"][CELL_ID]["thresholds_per_fold"]
    baseline_checks = {
        "n_success_exact": lr_baseline["n_success"] == source_lr["n_success"],
        "success_rate_exact": math.isclose(
            lr_baseline["success_rate"], source_lr["success_rate"], abs_tol=1e-15
        ),
        "mean_cost_exact": math.isclose(
            lr_baseline["mean_total_billed_cost_usd"],
            source_lr["mean_total_billed_cost_usd"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "thresholds_exact": lr_baseline["thresholds_per_fold"]
        == {str(k): float(v) for k, v in source_tau.items()},
    }
    if not all(baseline_checks.values()):
        raise AssertionError(f"Locked LR baseline reproduction failed: {baseline_checks}")

    for row in combinations:
        row["delta_vs_lr_baseline_pp"] = (
            row["success_rate_pct"] - lr_baseline["success_rate_pct"]
        )

    best_single = full_refs["best_single_mode"]
    oracle = full_refs["six_mode_oracle_ceiling"]
    ranked = sorted(
        combinations,
        key=lambda row: (
            -row["success_rate_pct"],
            row["mean_total_billed_cost_usd"],
            MODEL_ORDER.index(row["model_id"]),
            FEATURE_ORDER.index(row["feature_id"]),
        ),
    )
    winners = [
        row
        for row in combinations
        if row["success_rate_pct"] > best_single["success_rate_pct"]
    ]
    dimensions = {
        feature_id: next(
            row["dimensions_per_fold"]
            for row in combinations
            if row["feature_id"] == feature_id
        )
        for feature_id in FEATURE_ORDER
    }

    source_fold_path = source_dir / f"{CELL_ID}_fold_assignment.json"
    input_files = [
        source_dir / "raw_features_phase1a.npz",
        source_dir / "raw_features_phase1a.json",
        source_dir / "stage2_summary.json",
        source_dir / f"{CELL_ID}_lr_meta.json",
        source_replay_path,
        source_fold_path,
        RUN_MANIFEST,
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "banner": BANNER,
        "post_hoc_exploratory": True,
        "preregistered_router": False,
        "h10_eligible": False,
        "gate_eligible": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cell_id": CELL_ID,
        "question": (
            "Is the negative locked-LR offline result caused by a weak classifier "
            "class, or is it robust to stronger sklearn model classes and broader "
            "fold-local features?"
        ),
        "protocol": {
            "outer_folds": N_OUTER_FOLDS,
            "fold_map_sha256": fold_map_sha256(fold_map),
            "fold_map_file_sha256": sha256_file(source_fold_path),
            "fold_map_source": _relative(source_fold_path),
            "fold_map_reused_read_only": True,
            "min_class_n_train": N_MIN_CLASS_TRAIN,
            "inner_cv_folds": N_FOLDS_INNER,
            "inner_cv_seed_rule": "42 + outer_fold * 100",
            "tau_candidates": TAU_CANDIDATES,
            "tau_objective": "mode_match_accuracy",
            "tau_decision_rule": "argmax class iff max_probability > tau, else P-SoM",
            "fallback_mode": FALLBACK_MODE,
            "random_seed": SEED,
            "raw_train_replay_feature_parity": raw_parity,
            "classifier_specs": _model_specs(),
            "feature_specs": _feature_specs(dimensions, full_pool_sizes),
        },
        "inputs": {
            "source_artifacts_dir": _relative(source_dir),
            "source_artifacts_read_only": True,
            "run_manifest": _relative(RUN_MANIFEST),
            "phase1_root": _relative(PHASE1_ROOT),
            "sha256": {_relative(path): sha256_file(path) for path in input_files},
            "outcome_provenance": outcome_provenance,
        },
        "reference_points": {
            "best_single": best_single,
            "six_mode_oracle": oracle,
            "locked_lr_offline": {
                "n_success": lr_baseline["n_success"],
                "n_tasks": lr_baseline["n_tasks"],
                "success_rate_pct": lr_baseline["success_rate_pct"],
                "mean_total_billed_cost_usd": lr_baseline[
                    "mean_total_billed_cost_usd"
                ],
            },
        },
        "baseline_reproduction_checks": baseline_checks,
        "selection_summary": {
            "n_model_classes": len(MODEL_ORDER),
            "n_feature_sets": len(FEATURE_ORDER),
            "n_combinations": len(combinations),
            "best_combination_id": (
                f"{ranked[0]['model_id']}__{ranked[0]['feature_id']}"
            ),
            "best_success_rate_pct": ranked[0]["success_rate_pct"],
            "best_n_success": ranked[0]["n_success"],
            "best_delta_vs_best_single_pp": ranked[0]["delta_vs_best_single_pp"],
            "n_combinations_above_best_single": len(winners),
            "combinations_above_best_single": [
                f"{row['model_id']}__{row['feature_id']}" for row in winners
            ],
            "selection_effect_warning": (
                "The maximum is selected from 15 correlated post-hoc cells on the "
                "same 224 tasks. It is optimistically selected and is not a corrected "
                "confirmatory estimate or an H10 result."
            ),
        },
        "combinations": combinations,
        "content_sha256_excluding_self": _json_sha256(
            {
                "fold_map": fold_map,
                "combination_metrics": [
                    {
                        "model_id": row["model_id"],
                        "feature_id": row["feature_id"],
                        "n_success": row["n_success"],
                        "mean_cost": row["mean_total_billed_cost_usd"],
                        "thresholds": row["thresholds_per_fold"],
                    }
                    for row in combinations
                ],
            }
        ),
    }


def _pct(value: float) -> str:
    return f"{value:.2f}%"


def _cost(value: float) -> str:
    return f"{value:.8f}"


def _dims(row: dict[str, Any]) -> str:
    values = list(row["dimensions_per_fold"].values())
    return str(values[0]) if len(set(values)) == 1 else f"{min(values)}–{max(values)}"


def render_report(payload: dict[str, Any]) -> str:
    refs = payload["reference_points"]
    best_single = refs["best_single"]
    oracle = refs["six_mode_oracle"]
    locked_lr = refs["locked_lr_offline"]
    combos = payload["combinations"]
    ranked = sorted(
        combos,
        key=lambda row: (
            -row["success_rate_pct"], row["mean_total_billed_cost_usd"]
        ),
    )
    winners = [row for row in combos if row["delta_vs_best_single_pp"] > 0]
    best = ranked[0]

    if winners:
        conclusion = (
            f"负结果**不对本次模型类网格完全稳健**：15 个 post-hoc 格子中有 "
            f"**{len(winners)}** 个超过 SoM；最高为 {best['model']} × "
            f"{best['feature_set']}，{best['n_success']}/{best['n_tasks']} = "
            f"**{best['success_rate_pct']:.2f}%**（vs SoM "
            f"{best['delta_vs_best_single_pp']:+.2f} pp）。这只说明 LR/18维假设类可能"
            "约束了点估计；由于同一数据上从 15 格挑最大值，不能据此声称有可泛化的"
            "router gain。"
        )
    else:
        conclusion = (
            "负结果在本次 5×3 post-hoc 网格内**稳健**：没有任何更强模型/更宽特征"
            f"组合超过 best-single SoM 的 {best_single['success_rate_pct']:.2f}%。"
            f"最高组合为 {best['model']} × {best['feature_set']}，"
            f"{best['n_success']}/{best['n_tasks']} = {best['success_rate_pct']:.2f}% "
            f"（{best['delta_vs_best_single_pp']:+.2f} pp vs SoM）。因此当前证据更"
            "符合可学习路由信号偏弱和/或单次 oracle 标签噪声、样本支持不足，而不是"
            "单纯 LR 假设类太弱；这仍"
            "是离线、单 cell、post-hoc 结论。"
        )

    lines = [
        f"# {BANNER}",
        "",
        f"> **{BANNER}**  ",
        "> `gate_eligible=false` · `h10_eligible=false` · 仅 B0_classifieds 离线回放；禁止写入任何 H10 gate。",
        "",
        "## 结论",
        "",
        conclusion,
        "",
        "三参考点保持不变：best-single SoM = "
        f"**{best_single['n_success']}/{best_single['n_tasks']} = "
        f"{best_single['success_rate_pct']:.2f}%**，六-mode oracle = "
        f"**{oracle['n_success']}/{oracle['n_tasks']} = "
        f"{oracle['success_rate_pct']:.2f}%**，locked LR baseline = "
        f"**{locked_lr['n_success']}/{locked_lr['n_tasks']} = "
        f"{locked_lr['success_rate_pct']:.2f}%**。LR/18 维复现检查四项均通过。",
        "",
        "## 15 格汇总",
        "",
        "| 模型 | 特征 | 每折维数 | OOF success | Routed SR | Mean billed cost | Δ vs SoM | Δ vs locked LR | Oracle gap | Fallback | τ (fold 0→4) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for model_id in MODEL_ORDER:
        for feature_id in FEATURE_ORDER:
            row = next(
                item
                for item in combos
                if item["model_id"] == model_id and item["feature_id"] == feature_id
            )
            taus = "/".join(
                f"{row['thresholds_per_fold'][str(k)]:.1f}"
                for k in range(N_OUTER_FOLDS)
            )
            lines.append(
                f"| {row['model']} | {row['feature_set']} | {_dims(row)} | "
                f"{row['n_success']}/{row['n_tasks']} | **{_pct(row['success_rate_pct'])}** | "
                f"{_cost(row['mean_total_billed_cost_usd'])} | "
                f"{row['delta_vs_best_single_pp']:+.2f} pp | "
                f"{row['delta_vs_lr_baseline_pp']:+.2f} pp | "
                f"{row['oracle_gap_pp']:.2f} pp | "
                f"{row['fallback_count']}/{row['n_tasks']} | {taus} |"
            )

    lines.extend(
        [
            "",
            "特征格 ③ 的“全量 TF-IDF”按无重复列实现：在与原协议相同的每折 pooled train side 上设置 `max_features=None, min_df=3`，以完整词表**替换**原 top-30 TF-IDF block，再拼 20 个 raw 特征；因此实际维数按折报告，不把重复的 top-30 再加一遍。",
            "",
            "## OOF 细节与防假阳检查",
            "",
        ]
    )
    if winners:
        lines.append(
            "下表列出所有超过 27.23% 的格子在五个 outer holdout fold 上的波动；"
            "`Δ vs fold SoM` 使用同一 fold 的任务子集，不能用全 cell 参考值替代。"
        )
        for row in sorted(
            winners,
            key=lambda item: (
                -item["success_rate_pct"], item["mean_total_billed_cost_usd"]
            ),
        ):
            paired = row["paired_vs_som"]
            lines.extend(
                [
                    "",
                    f"### {row['model']} × {row['feature_set']}",
                    "",
                    f"整-cell {row['n_success']}/{row['n_tasks']} = "
                    f"{row['success_rate_pct']:.2f}%（{row['delta_vs_best_single_pp']:+.2f} pp）；"
                    f"五折 SR unweighted mean±SD = "
                    f"{row['fold_success_rate_pct_mean_unweighted']:.2f}±"
                    f"{row['fold_success_rate_pct_sd_unweighted']:.2f}%（range "
                    f"{row['fold_success_rate_pct_min']:.2f}–"
                    f"{row['fold_success_rate_pct_max']:.2f}%）。与 SoM 配对："
                    f"both={paired['both_success']}，router-only={paired['router_only_success']}，"
                    f"SoM-only={paired['som_only_success']}，neither={paired['neither_success']}。",
                    "",
                    "| Fold | n | Router success | Router SR | Fold SoM SR | Δ vs fold SoM | Cost | τ | Fallback | Route counts |",
                    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
                ]
            )
            for fold_k in range(N_OUTER_FOLDS):
                fold = row["fold_records"][str(fold_k)]
                routes = ", ".join(
                    f"{MODE_LABELS[mode]}={count}"
                    for mode, count in fold["selected_mode_counts"].items()
                )
                lines.append(
                    f"| {fold_k} | {fold['n_tasks']} | {fold['n_success']} | "
                    f"{fold['success_rate_pct']:.2f}% | "
                    f"{fold['som_success_rate_pct']:.2f}% | "
                    f"{fold['delta_vs_fold_som_pp']:+.2f} pp | "
                    f"{fold['mean_total_billed_cost_usd']:.8f} | {fold['tau']:.1f} | "
                    f"{fold['fallback_count']}/{fold['n_tasks']} | {routes} |"
                )
    else:
        lines.append(
            "没有格子超过 27.23%，故不存在需按纪律展开的“翻盘组合”。作为最接近"
            "边界的 sanity check，机器可读 JSON 仍保存全部 15 格 × 5 折指标和逐任务"
            "预测。"
        )

    lines.extend(
        [
            "",
            "## 多重比较警示",
            "",
            "这是在同一 224 个任务上比较 5 个模型 × 3 个特征格，并从 15 个高度相关的 post-hoc 格子挑最大值。最大 SR 带有 selection effect / winner's curse；本报告未做独立验证、nested model selection、multiplicity correction 或 fresh replay。因此任何超过 27.23% 的点也只能称为 exploratory flip，不能称显著改进、不能替代 live Pass-2、不能进入 H10。五折波动只是透明度检查，不消除选择偏差。",
            "",
            "## 协议与 provenance",
            "",
            f"- Cell：`{payload['cell_id']}`，97 个 oracle-labeled train rows，224/224 OOF replay coverage。",
            "- 可选 B1_classifieds 4/5 diagnostic 未纳入：它不是合法整-cell OOF 点；本次多重比较 family 固定为 B0 的 15 格。",
            f"- Fold map 内容 SHA-256：`{payload['protocol']['fold_map_sha256']}`；与 2026-07-15 offline replay 完全相同。",
            f"- 外层 min-class：`n_train >= {N_MIN_CLASS_TRAIN}`；τ 候选：`{TAU_CANDIDATES}`；inner-CV objective 仍为 mode-match accuracy；判断仍为 `max_prob > τ`，否则 P-SoM。",
            "- Mean billed cost 复用被选中 Pass-1 episode 的 `total_billed_cost_usd`；B0 cost basis 为 `api_usd`，不含真实 router serving overhead/latency。",
            f"- Machine-readable：`results/phantom_paper/l1_router_sweep_20260715/router_model_sweep.json`；summary CSV 与 fold CSV 同目录。",
            "- Source `results/phantom_paper/l1_router_offline_20260715/` 只读；未写 canonical `results/phantom_paper/l1_router/`，未改 paper drafts，未触发 fire。",
            "",
            "## 限制",
            "",
            "标签与回放 outcome 都来自每 mode 单次 Pass-1 realization；OOF 只防 task-level 训练泄漏，不能消除 oracle-label 噪声、template sibling 相关性、trajectory 随机性或离线状态反事实缺失。更强模型没有引入新信号；它只能更灵活地拟合现有 97 个标签。",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_csvs(payload: dict[str, Any], out_dir: Path) -> None:
    summary_path = out_dir / "router_model_sweep_summary.csv"
    summary_fields = [
        "artifact_status",
        "post_hoc_exploratory",
        "preregistered_router",
        "h10_eligible",
        "model_id",
        "model",
        "feature_id",
        "feature_set",
        "dimensions_per_fold",
        "n_tasks",
        "n_success",
        "success_rate_pct",
        "mean_total_billed_cost_usd",
        "delta_vs_best_single_pp",
        "delta_vs_lr_baseline_pp",
        "oracle_gap_pp",
        "fallback_count",
        "fallback_rate",
        "thresholds_per_fold",
    ]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in payload["combinations"]:
            writer.writerow(
                {
                    "artifact_status": ARTIFACT_STATUS,
                    "post_hoc_exploratory": True,
                    "preregistered_router": False,
                    "h10_eligible": False,
                    **{key: row[key] for key in summary_fields if key in row},
                    "dimensions_per_fold": json.dumps(row["dimensions_per_fold"], sort_keys=True),
                    "thresholds_per_fold": json.dumps(row["thresholds_per_fold"], sort_keys=True),
                }
            )

    fold_path = out_dir / "router_model_sweep_folds.csv"
    fold_fields = [
        "artifact_status",
        "post_hoc_exploratory",
        "preregistered_router",
        "h10_eligible",
        "model_id",
        "feature_id",
        "fold_k",
        "n_tasks",
        "n_success",
        "success_rate_pct",
        "som_success_rate_pct",
        "delta_vs_fold_som_pp",
        "mean_total_billed_cost_usd",
        "tau",
        "fallback_count",
        "fallback_rate",
        "selected_mode_counts",
    ]
    with fold_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fold_fields)
        writer.writeheader()
        for row in payload["combinations"]:
            for fold_k, fold in row["fold_records"].items():
                writer.writerow(
                    {
                        "artifact_status": ARTIFACT_STATUS,
                        "post_hoc_exploratory": True,
                        "preregistered_router": False,
                        "h10_eligible": False,
                        "model_id": row["model_id"],
                        "feature_id": row["feature_id"],
                        "fold_k": fold_k,
                        **{
                            key: fold[key]
                            for key in fold_fields
                            if key in fold and key != "selected_mode_counts"
                        },
                        "selected_mode_counts": json.dumps(
                            fold["selected_mode_counts"], sort_keys=True
                        ),
                    }
                )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    report_path = args.report.resolve()
    forbidden = {SOURCE_DIR.resolve(), CANONICAL_DIR.resolve()}
    if out_dir in forbidden:
        parser.error("Refusing to write post-hoc sweep into source offline or canonical dir")
    if not source_dir.is_dir():
        parser.error(f"Source artifacts directory not found: {source_dir}")
    if source_dir != SOURCE_DIR.resolve():
        print(f"WARNING: non-default read-only source directory: {source_dir}")

    payload = _json_native(run_sweep(source_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "router_model_sweep.json"
    results_report_path = out_dir / "router_model_sweep.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    report = render_report(payload)
    results_report_path.write_text(report)
    report_path.write_text(report)
    _write_csvs(payload, out_dir)
    print(f"Wrote: {json_path}")
    print(f"Wrote: {results_report_path}")
    print(f"Wrote: {report_path}")
    print(BANNER)
    return 0


if __name__ == "__main__":
    sys.exit(main())
