#!/usr/bin/env python3
"""Recipe-frozen B0-reddit replication of the cost-aware success router.

Primary design (D2): fit six binary success heads within B0-reddit using a
task-held-out five-fold protocol, then apply the B0-classifieds-selected
``tau=0.10`` exactly once.  The output is an offline Pass-1 replay and is never
eligible for the preregistered live H10 gate.

The cheap sensitivity points and the fully frozen classifieds-to-reddit transfer
(D1) are emitted as explicitly secondary diagnostics.  Neither can change the
primary threshold or conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from p79.policies.learned_router import build_runtime_feature_vector  # noqa: E402
from p79.policies.router_features import INTENT_REGEX, derive_oracle_label  # noqa: E402
from scripts.analysis.router_offline_replay import (  # noqa: E402
    DISPLAY_MODES,
    MODE_LABELS,
    build_offline_raw_features,
    collect_cell_outcomes,
    fold_map_sha256,
    load_paper_grade_entries,
    policy_metrics,
    reference_points,
    sha256_file,
)
from scripts.analysis.router_pareto_analysis import (  # noqa: E402
    PolicyPoint,
    dominance_relation,
    pareto_frontier,
)
from scripts.analysis.train_l1_router import (  # noqa: E402
    LR_C,
    LR_MAX_ITER,
    N_MIN_CLASS_TRAIN,
    SAFE_FALLBACK_MODE,
    apply_min_class_filter,
    tune_threshold_inner_cv,
)
from scripts.analysis.train_l1_router_with_mi import (  # noqa: E402
    FOLD_SEED,
    MI_SEED,
    N_SELECTED,
    N_SPLITS,
    TFIDF_MAX_FEATURES,
    TFIDF_MIN_DF,
    build_design_matrix,
    fit_fold_local_tfidf,
    fit_pooled_mi_selector,
    generate_per_cell_fold_assignments,
)


CELL_ID = "B0_reddit"
SOURCE_CELL_ID = "B0_classifieds"
FROZEN_TAU = 0.10
SENSITIVITY_TAUS = (0.05, 0.10, 0.15)
SUCCESS_HEAD_SEED = 42
BOOTSTRAP_SEED = 20260716
DEFAULT_BOOTSTRAP_REPLICATES = 5_000
MIN_BOOTSTRAP_REPLICATES = 2_000

DEFAULT_RUN_MANIFEST = REPO / "results/phantom_paper/run_manifest.yaml"
DEFAULT_PHASE1_ROOT = REPO / "results/visualwebarena/phase1"
DEFAULT_FROZEN_RECIPE = (
    REPO
    / "results/phantom_paper/l1_router_offline_20260715/pareto"
    / "cost_aware_success_oof.json"
)
DEFAULT_OUT_DIR = (
    REPO / "results/phantom_paper/l1_router_offline_20260716_red_replication"
)

SCHEMA_VERSION = "2026-07-16-b0red-cost-aware-replication-v1"
ARTIFACT_STATUS = "OFFLINE / NON-GATE / RECIPE-FROZEN-ON-CLS"
DISCLAIMER = (
    "OFFLINE Pass-1 trajectory replay; not the preregistered live H10 gate and "
    "not an independent trajectory realization"
)
NUMERIC_FEATURE_NAMES = [
    "dom_complexity",
    "text_length",
    "tokens_input_text",
    "intent_token_count",
    "reasoning_difficulty",
]
BINARY_FEATURE_NAMES = ["has_reference_image", *sorted(INTENT_REGEX)]


@dataclass(frozen=True)
class FrozenRecipe:
    """The decision degrees of freedom selected on B0-classifieds only."""

    source_cell_id: str
    threshold: float
    threshold_comparator: str
    outer_folds: int
    fold_seed: int
    mi_features: int
    mi_seed: int
    lr_c: float
    lr_max_iter: int
    lr_seed: int
    class_weight: None
    decision_cost: str
    fallback: str
    source_curve_sr_pct: float
    source_curve_mean_cost_usd: float


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_frozen_recipe_payload(payload: Mapping[str, Any]) -> FrozenRecipe:
    """Fail closed unless the supplied artifact encodes the frozen cls recipe."""

    if payload.get("cell_id") != SOURCE_CELL_ID:
        raise ValueError(
            f"Frozen recipe must come from {SOURCE_CELL_ID}, got {payload.get('cell_id')!r}"
        )
    protocol = payload.get("protocol")
    if not isinstance(protocol, Mapping):
        raise ValueError("Frozen recipe lacks a protocol mapping")
    required = {
        "outer_folds": N_SPLITS,
        "threshold_comparator": ">=",
        "lr_C": LR_C,
        "lr_max_iter": LR_MAX_ITER,
        "random_seed": SUCCESS_HEAD_SEED,
        "class_weight": None,
        "cost_for_decision": "fold-training mean total_billed_cost_usd by mode",
        "fallback": "fold-training best single by SR, then lower mean cost",
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError(
                f"Frozen recipe protocol drift for {key}: "
                f"expected {expected!r}, got {protocol.get(key)!r}"
            )
    if "MI-18" not in str(protocol.get("feature_assets_reused", "")):
        raise ValueError("Frozen recipe no longer declares the MI-18 feature convention")
    thresholds = [float(value) for value in protocol.get("thresholds", [])]
    if FROZEN_TAU not in thresholds:
        raise ValueError(f"Frozen source curve does not contain tau={FROZEN_TAU:.2f}")
    curve_rows = [
        row
        for row in payload.get("curve", [])
        if math.isclose(float(row.get("threshold", -1)), FROZEN_TAU, abs_tol=1e-12)
    ]
    if len(curve_rows) != 1:
        raise ValueError("Frozen source must contain exactly one tau=0.10 curve row")
    source_point = curve_rows[0]
    return FrozenRecipe(
        source_cell_id=SOURCE_CELL_ID,
        threshold=FROZEN_TAU,
        threshold_comparator=">=",
        outer_folds=N_SPLITS,
        fold_seed=FOLD_SEED,
        mi_features=N_SELECTED,
        mi_seed=MI_SEED,
        lr_c=float(protocol["lr_C"]),
        lr_max_iter=int(protocol["lr_max_iter"]),
        lr_seed=int(protocol["random_seed"]),
        class_weight=None,
        decision_cost=str(protocol["cost_for_decision"]),
        fallback=str(protocol["fallback"]),
        source_curve_sr_pct=float(source_point["success_rate_pct"]),
        source_curve_mean_cost_usd=float(source_point["mean_cost_usd"]),
    )


def load_frozen_recipe(path: Path) -> tuple[FrozenRecipe, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen B0-classifieds recipe: {path}")
    payload = json.loads(path.read_text())
    return validate_frozen_recipe_payload(payload), payload


def assert_primary_threshold_frozen(threshold: float) -> None:
    """Guard against introducing a reddit-side primary-threshold choice."""

    if not math.isclose(float(threshold), FROZEN_TAU, abs_tol=1e-12, rel_tol=0.0):
        raise ValueError(
            f"Primary tau is recipe-frozen at {FROZEN_TAU:.2f}; "
            f"reddit-side value {threshold!r} is forbidden"
        )


def decide_cost_aware_mode(
    probabilities: Mapping[str, float],
    mean_train_costs: Mapping[str, float],
    threshold: float,
    fallback_mode: str,
) -> tuple[str, bool]:
    """Choose the cheapest eligible mode; ties follow the locked display order."""

    expected = set(DISPLAY_MODES)
    if set(probabilities) != expected or set(mean_train_costs) != expected:
        raise ValueError("Decision inputs must contain exactly the canonical six modes")
    if fallback_mode not in expected:
        raise ValueError(f"Invalid fallback mode: {fallback_mode!r}")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"Invalid success threshold: {threshold!r}")
    for mode in DISPLAY_MODES:
        probability = float(probabilities[mode])
        cost = float(mean_train_costs[mode])
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError(f"Invalid P(success) for {mode}: {probability!r}")
        if not math.isfinite(cost) or cost < 0.0:
            raise ValueError(f"Invalid train cost for {mode}: {cost!r}")
    eligible = [
        mode for mode in DISPLAY_MODES if float(probabilities[mode]) >= float(threshold)
    ]
    if not eligible:
        return fallback_mode, True
    order = {mode: index for index, mode in enumerate(DISPLAY_MODES)}
    return (
        min(eligible, key=lambda mode: (float(mean_train_costs[mode]), order[mode])),
        False,
    )


def decide_frozen_primary_mode(
    probabilities: Mapping[str, float],
    mean_train_costs: Mapping[str, float],
    fallback_mode: str,
) -> tuple[str, bool]:
    """Primary D2 decision with no caller-visible threshold degree of freedom."""

    assert_primary_threshold_frozen(FROZEN_TAU)
    return decide_cost_aware_mode(
        probabilities, mean_train_costs, FROZEN_TAU, fallback_mode
    )


def build_reddit_fold_map(task_ids: Iterable[int]) -> dict[int, int]:
    """Use the canonical seed-42 pure-KFold derivation on the full task universe."""

    ids = sorted({int(task_id) for task_id in task_ids})
    if not ids:
        raise ValueError("Cannot create a fold map for an empty task universe")
    cell_ids = np.asarray([CELL_ID] * len(ids), dtype=object)
    task_array = np.asarray(ids, dtype=int)
    # Fold generation no longer uses labels, but the public canonical function keeps
    # the argument for compatibility.  Dummy values cannot affect the pure KFold.
    assignments = generate_per_cell_fold_assignments(
        cell_ids,
        task_array,
        np.asarray(["__universe__"] * len(ids), dtype=object),
        all_cell_ids=cell_ids,
        all_task_ids=task_array,
        seed=FOLD_SEED,
        n_splits=N_SPLITS,
    )
    fold_map = assignments[CELL_ID]
    if set(fold_map) != set(ids) or set(fold_map.values()) != set(range(N_SPLITS)):
        raise AssertionError("Canonical reddit fold-map derivation failed coverage")
    return fold_map


def _feature_matrix(
    task_ids: list[int],
    raw_by_task: Mapping[int, Mapping[str, Any]],
    vectorizer: Any,
    selected_mask: np.ndarray | None = None,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for task_id in task_ids:
        raw = raw_by_task[task_id]
        if selected_mask is None:
            tfidf = vectorizer.transform([raw["intent_text"]]).toarray().ravel()
            rows.append(np.concatenate([tfidf, raw["numeric"], raw["binary"]]))
        else:
            rows.append(build_runtime_feature_vector(raw, vectorizer, selected_mask))
    if not rows:
        width = int(selected_mask.sum()) if selected_mask is not None else 0
        return np.empty((0, width), dtype=float)
    return np.vstack(rows)


def _fit_feature_asset(
    train_ids: list[int],
    raw_by_task: Mapping[int, Mapping[str, Any]],
    oracle_labels: Mapping[int, str | None],
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    """Fold/full-train TF-IDF + oracle-label MI-18, matching the cls convention."""

    labeled_ids = [task_id for task_id in train_ids if oracle_labels[task_id] is not None]
    if len(labeled_ids) < 2 or len({oracle_labels[t] for t in labeled_ids}) < 2:
        raise ValueError("MI-18 needs at least two oracle classes in the training split")
    intents = [str(raw_by_task[task_id]["intent_text"]) for task_id in labeled_ids]
    vectorizer = fit_fold_local_tfidf(
        intents, max_features=TFIDF_MAX_FEATURES, min_df=TFIDF_MIN_DF
    )
    numeric = np.vstack([raw_by_task[task_id]["numeric"] for task_id in labeled_ids])
    binary = np.vstack([raw_by_task[task_id]["binary"] for task_id in labeled_ids])
    X_full, tfidf_names = build_design_matrix(intents, numeric, binary, vectorizer)
    selector, selected_mask = fit_pooled_mi_selector(
        X_full,
        np.asarray([oracle_labels[task_id] for task_id in labeled_ids], dtype=object),
        k=N_SELECTED,
        seed=MI_SEED,
        n_binary=len(BINARY_FEATURE_NAMES),
    )
    feature_names = [*tfidf_names, *NUMERIC_FEATURE_NAMES, *BINARY_FEATURE_NAMES]
    if len(feature_names) != len(selected_mask) or int(selected_mask.sum()) != N_SELECTED:
        raise AssertionError("MI selector did not produce the locked 18-feature mask")
    scores = [None if not math.isfinite(float(v)) else float(v) for v in selector.scores_]
    return vectorizer, selected_mask, {
        "n_train_total": len(train_ids),
        "n_train_oracle_labeled": len(labeled_ids),
        "n_train_no_success_excluded_from_mi": len(train_ids) - len(labeled_ids),
        "tfidf_vocabulary_size": len(vectorizer.get_feature_names_out()),
        "n_candidate_features": len(feature_names),
        "n_selected_features": int(selected_mask.sum()),
        "selected_feature_names": [
            name for name, selected in zip(feature_names, selected_mask) if selected
        ],
        "feature_names_all": feature_names,
        "mi_scores_all": scores,
        "mi_target": "cheapest-prior successful-mode oracle label; no-success tasks excluded",
    }


def _fit_binary_success_heads(
    train_ids: list[int],
    predict_ids: list[int],
    outcomes: Mapping[int, Mapping[str, Mapping[str, Any]]],
    raw_by_task: Mapping[int, Mapping[str, Any]],
    vectorizer: Any,
    selected_mask: np.ndarray,
) -> tuple[dict[int, dict[str, float]], list[dict[str, Any]]]:
    X_train = _feature_matrix(train_ids, raw_by_task, vectorizer, selected_mask)
    X_predict = _feature_matrix(predict_ids, raw_by_task, vectorizer, selected_mask)
    probabilities = {task_id: {} for task_id in predict_ids}
    fit_rows: list[dict[str, Any]] = []
    for mode in DISPLAY_MODES:
        y_train = np.asarray(
            [int(outcomes[task_id][mode]["success"] is True) for task_id in train_ids],
            dtype=int,
        )
        if len(np.unique(y_train)) == 1:
            predicted = np.full(len(predict_ids), float(y_train[0]))
            fit_kind = "constant_single_class"
        else:
            estimator = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            class_weight=None,
                            max_iter=LR_MAX_ITER,
                            C=LR_C,
                            solver="lbfgs",
                            random_state=SUCCESS_HEAD_SEED,
                        ),
                    ),
                ]
            )
            estimator.fit(X_train, y_train)
            positive_index = int(np.where(estimator.classes_ == 1)[0][0])
            predicted = estimator.predict_proba(X_predict)[:, positive_index]
            fit_kind = "binary_logistic_regression"
        if not np.isfinite(predicted).all() or np.any((predicted < 0) | (predicted > 1)):
            raise ValueError(f"Invalid success probability for mode={mode}")
        for task_id, probability in zip(predict_ids, predicted.tolist()):
            probabilities[task_id][mode] = float(probability)
        fit_rows.append(
            {
                "mode": mode,
                "fit_kind": fit_kind,
                "n_train": len(train_ids),
                "n_positive_train": int(y_train.sum()),
                "n_predict": len(predict_ids),
            }
        )
    return probabilities, fit_rows


def _decision_context(
    train_ids: list[int], outcomes: Mapping[int, Mapping[str, Mapping[str, Any]]]
) -> dict[str, Any]:
    order = {mode: index for index, mode in enumerate(DISPLAY_MODES)}
    costs = {
        mode: float(np.mean([outcomes[t][mode]["cost_usd"] for t in train_ids]))
        for mode in DISPLAY_MODES
    }
    success_rates = {
        mode: float(np.mean([outcomes[t][mode]["success"] for t in train_ids]))
        for mode in DISPLAY_MODES
    }
    fallback = min(
        DISPLAY_MODES,
        key=lambda mode: (-success_rates[mode], costs[mode], order[mode]),
    )
    return {
        "n_train": len(train_ids),
        "mean_train_cost_usd_by_mode": costs,
        "train_success_rate_by_mode": success_rates,
        "fallback_best_single_mode": fallback,
    }


def _policy_point(policy_id: str, category: str, metric: Mapping[str, Any]) -> PolicyPoint:
    return PolicyPoint(
        policy_id=policy_id,
        label=policy_id,
        category=category,
        mean_cost_usd=float(metric["mean_total_billed_cost_usd"]),
        success_rate_pct=float(metric["success_rate_pct"]),
        n_tasks=int(metric["n_tasks"]),
        n_success=int(metric["n_success"]),
    )


def _paired_bootstrap(
    outcomes: Mapping[int, Mapping[str, Mapping[str, Any]]],
    selected: Mapping[int, str],
    best_mode: str,
    *,
    n_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if n_replicates < MIN_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"Paired bootstrap requires at least {MIN_BOOTSTRAP_REPLICATES} replicates"
        )
    task_ids = sorted(outcomes)
    if set(selected) != set(task_ids):
        raise ValueError("Bootstrap policy does not cover the full task universe")
    routed_success = np.asarray(
        [float(outcomes[t][selected[t]]["success"] is True) for t in task_ids]
    )
    routed_cost = np.asarray([float(outcomes[t][selected[t]]["cost_usd"]) for t in task_ids])
    best_success = np.asarray(
        [float(outcomes[t][best_mode]["success"] is True) for t in task_ids]
    )
    best_cost = np.asarray([float(outcomes[t][best_mode]["cost_usd"]) for t in task_ids])
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(task_ids), size=(n_replicates, len(task_ids)))
    delta_sr = 100.0 * (
        routed_success[sample_indices].mean(axis=1)
        - best_success[sample_indices].mean(axis=1)
    )
    delta_cost = (
        routed_cost[sample_indices].mean(axis=1)
        - best_cost[sample_indices].mean(axis=1)
    )
    atol = 1e-12
    router_dominates = (
        (delta_sr >= -atol)
        & (delta_cost <= atol)
        & ((delta_sr > atol) | (delta_cost < -atol))
    )
    best_dominates = (
        (delta_sr <= atol)
        & (delta_cost >= -atol)
        & ((delta_sr < -atol) | (delta_cost > atol))
    )
    return {
        "n_tasks": len(task_ids),
        "n_replicates": n_replicates,
        "seed": seed,
        "delta_definition": "router minus best single",
        "best_single_mode": best_mode,
        "delta_sr_pp": float(100.0 * (routed_success - best_success).mean()),
        "delta_sr_pp_percentile_95_ci": [
            float(value) for value in np.percentile(delta_sr, [2.5, 97.5])
        ],
        "delta_mean_cost_usd": float((routed_cost - best_cost).mean()),
        "delta_mean_cost_usd_percentile_95_ci": [
            float(value) for value in np.percentile(delta_cost, [2.5, 97.5])
        ],
        "router_pareto_dominates_best_single_fraction": float(router_dominates.mean()),
        "best_single_pareto_dominates_router_fraction": float(best_dominates.mean()),
        "router_strictly_dominates_best_single_fraction": float(
            ((delta_sr > atol) & (delta_cost < -atol)).mean()
        ),
        "best_single_strictly_dominates_router_fraction": float(
            ((delta_sr < -atol) & (delta_cost > atol)).mean()
        ),
    }


def _select_policy(
    probabilities: Mapping[int, Mapping[str, float]],
    fold_map: Mapping[int, int],
    contexts: Mapping[int, Mapping[str, Any]],
    threshold: float,
    *,
    frozen_primary: bool,
) -> tuple[dict[int, str], int]:
    if frozen_primary:
        assert_primary_threshold_frozen(threshold)
    selected: dict[int, str] = {}
    fallback_count = 0
    for task_id in sorted(probabilities):
        context = contexts[fold_map[task_id]]
        if frozen_primary:
            mode, fallback = decide_frozen_primary_mode(
                probabilities[task_id],
                context["mean_train_cost_usd_by_mode"],
                context["fallback_best_single_mode"],
            )
        else:
            mode, fallback = decide_cost_aware_mode(
                probabilities[task_id],
                context["mean_train_cost_usd_by_mode"],
                threshold,
                context["fallback_best_single_mode"],
            )
        selected[task_id] = mode
        fallback_count += int(fallback)
    return selected, fallback_count


def _fit_preregistered_lr_control(
    outcomes: Mapping[int, Mapping[str, Mapping[str, Any]]],
    raw_by_task: Mapping[int, Mapping[str, Any]],
    oracle_labels: Mapping[int, str | None],
    fold_map: Mapping[int, int],
    feature_assets: Mapping[int, tuple[Any, np.ndarray]],
) -> dict[str, Any]:
    """Run the locked multiclass LR protocol; fail closed on untrainable folds."""

    selected: dict[int, str] = {}
    fold_records: dict[str, Any] = {}
    task_records: list[dict[str, Any]] = []
    folds_ok: list[int] = []
    for fold_k in range(N_SPLITS):
        train_ids = sorted(
            task_id
            for task_id in outcomes
            if fold_map[task_id] != fold_k and oracle_labels[task_id] is not None
        )
        holdout_ids = sorted(task_id for task_id in outcomes if fold_map[task_id] == fold_k)
        y_train = np.asarray([oracle_labels[task_id] for task_id in train_ids], dtype=object)
        y_filtered, kept_local_indices, dropped = apply_min_class_filter(
            y_train, np.arange(len(train_ids), dtype=int), min_n=N_MIN_CLASS_TRAIN
        )
        kept_ids = [train_ids[int(index)] for index in kept_local_indices]
        base_record = {
            "fold_k": fold_k,
            "n_train_oracle_labeled": len(train_ids),
            "n_train_kept": len(kept_ids),
            "n_holdout_full_universe": len(holdout_ids),
            "raw_train_label_counts": dict(Counter(y_train.tolist())),
            "dropped_classes": dropped,
        }
        if len(y_filtered) < 2 or len(set(y_filtered.tolist())) < 2:
            fold_records[str(fold_k)] = {
                **base_record,
                "status": "insufficient_train_data_after_preregistered_min_class_filter",
                "n_classes_remaining": len(set(y_filtered.tolist())),
            }
            continue
        vectorizer, selected_mask = feature_assets[fold_k]
        X_train = _feature_matrix(kept_ids, raw_by_task, vectorizer, selected_mask)
        tau_result = tune_threshold_inner_cv(X_train, y_filtered, CELL_ID, fold_k)
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        class_weight=None,
                        max_iter=LR_MAX_ITER,
                        C=LR_C,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_filtered)
        X_holdout = _feature_matrix(holdout_ids, raw_by_task, vectorizer, selected_mask)
        proba = pipeline.predict_proba(X_holdout)
        maximum = proba.max(axis=1)
        argmax = pipeline.classes_[proba.argmax(axis=1)]
        tau_star = float(tau_result["chosen_tau"])
        decisions = np.where(maximum > tau_star, argmax, SAFE_FALLBACK_MODE)
        folds_ok.append(fold_k)
        for task_id, mode, max_probability, argmax_mode in zip(
            holdout_ids, decisions.tolist(), maximum.tolist(), argmax.tolist()
        ):
            selected[task_id] = str(mode)
            task_records.append(
                {
                    "task_id": task_id,
                    "fold_k": fold_k,
                    "selected_mode": str(mode),
                    "argmax_mode": str(argmax_mode),
                    "max_probability": float(max_probability),
                    "tau": tau_star,
                    "fallback_fired": bool(max_probability <= tau_star),
                }
            )
        fold_records[str(fold_k)] = {
            **base_record,
            "status": "ok",
            "chosen_tau": tau_star,
            "tau_tuning_reason": tau_result["reason"],
            "tau_tuning_n_inner_folds": tau_result["n_inner_folds_used"],
            "train_label_counts_after_filter": dict(Counter(y_filtered.tolist())),
        }
    complete = folds_ok == list(range(N_SPLITS)) and set(selected) == set(outcomes)
    return {
        "protocol": {
            "label": "cheapest-prior successful-mode oracle label",
            "feature_pipeline": "fold-local TF-IDF + oracle-label MI-18",
            "min_class_n_train": N_MIN_CLASS_TRAIN,
            "classifier": "StandardScaler + multiclass LogisticRegression",
            "lr_C": LR_C,
            "lr_max_iter": LR_MAX_ITER,
            "class_weight": None,
            "threshold": "fold-train inner-CV over prereg candidates",
            "fallback": SAFE_FALLBACK_MODE,
        },
        "training_status": "TRAINED_COMPLETE" if complete else "UNTRAINABLE",
        "folds_ok": folds_ok,
        "fold_records": fold_records,
        "oof_routed": policy_metrics(dict(outcomes), selected) if complete else None,
        "fallback_reference_when_untrainable": (
            reference_points(dict(outcomes))["always_p_som"] if not complete else None
        ),
        "task_records": task_records,
    }


def run_d2(
    outcomes: dict[int, dict[str, dict[str, Any]]],
    entries: dict[str, dict[str, Any]],
    phase1_root: Path,
    *,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[int, int]]:
    task_ids = sorted(outcomes)
    fold_map = build_reddit_fold_map(task_ids)
    raw_by_task: dict[int, dict[str, Any]] = {}
    raw_provenance: dict[str, Any] = {}
    for task_id in task_ids:
        raw, provenance = build_offline_raw_features(
            entries, phase1_root, "reddit", task_id
        )
        raw_by_task[task_id] = raw
        raw_provenance[str(task_id)] = provenance
    oracle_labels = {
        task_id: derive_oracle_label(
            {mode: bool(outcomes[task_id][mode]["success"]) for mode in DISPLAY_MODES}
        )
        for task_id in task_ids
    }

    probabilities: dict[int, dict[str, float]] = {task_id: {} for task_id in task_ids}
    contexts: dict[int, dict[str, Any]] = {}
    feature_meta: dict[str, Any] = {}
    fit_rows: list[dict[str, Any]] = []
    feature_assets: dict[int, tuple[Any, np.ndarray]] = {}
    for fold_k in range(N_SPLITS):
        train_ids = sorted(task_id for task_id in task_ids if fold_map[task_id] != fold_k)
        holdout_ids = sorted(task_id for task_id in task_ids if fold_map[task_id] == fold_k)
        vectorizer, selected_mask, metadata = _fit_feature_asset(
            train_ids, raw_by_task, oracle_labels
        )
        feature_assets[fold_k] = (vectorizer, selected_mask)
        feature_meta[str(fold_k)] = metadata
        fold_probabilities, fold_fits = _fit_binary_success_heads(
            train_ids,
            holdout_ids,
            outcomes,
            raw_by_task,
            vectorizer,
            selected_mask,
        )
        for task_id in holdout_ids:
            probabilities[task_id] = fold_probabilities[task_id]
        fit_rows.extend({"fold_k": fold_k, **row} for row in fold_fits)
        contexts[fold_k] = {
            **_decision_context(train_ids, outcomes),
            "n_holdout": len(holdout_ids),
        }
    if any(set(row) != set(DISPLAY_MODES) for row in probabilities.values()):
        raise AssertionError("Incomplete six-mode OOF probability matrix")

    references = reference_points(outcomes)
    best = references["best_single_mode"]
    policies: dict[str, Any] = {}
    selections: dict[float, dict[int, str]] = {}
    for threshold in SENSITIVITY_TAUS:
        is_primary = math.isclose(threshold, FROZEN_TAU, abs_tol=1e-12)
        selected, fallback_count = _select_policy(
            probabilities,
            fold_map,
            contexts,
            threshold,
            frozen_primary=is_primary,
        )
        selections[threshold] = selected
        metric = policy_metrics(outcomes, selected)
        metric.update(
            {
                "threshold": threshold,
                "fallback_count": fallback_count,
                "fallback_rate": fallback_count / len(task_ids),
                "status": (
                    "PRIMARY_RECIPE_FROZEN_ON_CLS"
                    if is_primary
                    else "POST_HOC_SENSITIVITY_NOT_FOR_CONCLUSION"
                ),
                "delta_vs_best_single_pp": (
                    metric["success_rate_pct"] - best["success_rate_pct"]
                ),
                "delta_vs_best_single_mean_cost_usd": (
                    metric["mean_total_billed_cost_usd"]
                    - best["mean_total_billed_cost_usd"]
                ),
            }
        )
        metric["dominance_relation_router_vs_best_single"] = dominance_relation(
            _policy_point(f"cost_aware_tau_{threshold:.2f}", "router", metric),
            _policy_point(f"fixed_{best['mode']}", "fixed", best),
        )
        policies[f"{threshold:.2f}"] = metric

    primary_selected = selections[FROZEN_TAU]
    primary = policies[f"{FROZEN_TAU:.2f}"]
    bootstrap = _paired_bootstrap(
        outcomes,
        primary_selected,
        str(best["mode"]),
        n_replicates=bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    fixed_points = [
        _policy_point(f"fixed_{mode}", "fixed", references["single_modes"][mode])
        for mode in DISPLAY_MODES
    ]
    primary_point = _policy_point("cost_aware_tau_0.10", "router", primary)
    oracle_point = _policy_point(
        "six_mode_oracle", "hindsight_oracle", references["six_mode_oracle_ceiling"]
    )
    deployable_frontier = [
        point.policy_id for point in pareto_frontier([*fixed_points, primary_point])
    ]
    hindsight_frontier = [
        point.policy_id
        for point in pareto_frontier([*fixed_points, primary_point, oracle_point])
    ]

    primary_task_records = []
    for task_id in task_ids:
        mode = primary_selected[task_id]
        primary_task_records.append(
            {
                "task_id": task_id,
                "fold_k": fold_map[task_id],
                "probability_success_by_mode": probabilities[task_id],
                "selected_mode": mode,
                "success": bool(outcomes[task_id][mode]["success"]),
                "total_billed_cost_usd": float(outcomes[task_id][mode]["cost_usd"]),
                "fallback_fired": all(
                    probabilities[task_id][candidate] < FROZEN_TAU
                    for candidate in DISPLAY_MODES
                ),
                "outcome_summary_path": outcomes[task_id][mode]["summary_path"],
            }
        )

    lr_control = _fit_preregistered_lr_control(
        outcomes, raw_by_task, oracle_labels, fold_map, feature_assets
    )
    return {
        "design": "D2 primary: recipe-frozen threshold + within-B0_reddit OOF training",
        "primary_threshold": FROZEN_TAU,
        "threshold_selected_on": SOURCE_CELL_ID,
        "reddit_primary_thresholds_tried": [FROZEN_TAU],
        "reddit_threshold_selection_performed": False,
        "n_tasks": len(task_ids),
        "n_oracle_labeled_tasks": sum(label is not None for label in oracle_labels.values()),
        "oracle_label_counts": dict(Counter(oracle_labels.values())),
        "fold_protocol": {
            "n_splits": N_SPLITS,
            "seed": FOLD_SEED,
            "derivation": (
                "generate_per_cell_fold_assignments: full-universe, site-level shared "
                "pure KFold(shuffle=True, random_state=42); single B0_reddit cell"
            ),
            "fold_sizes": dict(Counter(fold_map.values())),
            "fold_map_content_sha256": fold_map_sha256(fold_map),
        },
        "feature_protocol": {
            "fold_local": True,
            "tfidf_max_features": TFIDF_MAX_FEATURES,
            "tfidf_min_df": TFIDF_MIN_DF,
            "mi_selected_features": N_SELECTED,
            "mi_seed": MI_SEED,
            "mi_target": "multiclass cheapest-prior successful-mode oracle label",
            "binary_heads_train_on_all_fold-train_tasks": True,
            "per_fold": feature_meta,
            "step0_zero_filled_count": sum(
                int(row["step0_zero_filled"]) for row in raw_provenance.values()
            ),
        },
        "success_head_protocol": {
            "n_heads_per_fold": len(DISPLAY_MODES),
            "pipeline": "StandardScaler + LogisticRegression",
            "C": LR_C,
            "max_iter": LR_MAX_ITER,
            "solver": "lbfgs",
            "random_seed": SUCCESS_HEAD_SEED,
            "class_weight": None,
            "fit_rows": fit_rows,
        },
        "fold_decision_context": {str(key): value for key, value in contexts.items()},
        "reference_points": references,
        "primary_cost_aware_router": primary,
        "paired_bootstrap_vs_best_single": bootstrap,
        "deployable_frontier": deployable_frontier,
        "hindsight_augmented_frontier": hindsight_frontier,
        "lr_mode_router_control": lr_control,
        "sensitivity_appendix": {
            "status": "POST-HOC CURVE; NOT USED FOR THE CONCLUSION",
            "thresholds": list(SENSITIVITY_TAUS),
            "results": policies,
        },
        "primary_task_records": primary_task_records,
        "sensitivity_selections_by_threshold": {
            f"{threshold:.2f}": {
                str(task_id): mode for task_id, mode in sorted(selection.items())
            }
            for threshold, selection in selections.items()
        },
    }, fold_map


def run_d1_transfer(
    red_outcomes: dict[int, dict[str, dict[str, Any]]],
    red_entries: dict[str, dict[str, Any]],
    cls_outcomes: dict[int, dict[str, dict[str, Any]]],
    cls_entries: dict[str, dict[str, Any]],
    phase1_root: Path,
    *,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    """Bonus D1: train once on all cls tasks and freeze everything for reddit."""

    cls_ids = sorted(cls_outcomes)
    red_ids = sorted(red_outcomes)
    cls_raw = {
        task_id: build_offline_raw_features(
            cls_entries, phase1_root, "classifieds", task_id
        )[0]
        for task_id in cls_ids
    }
    red_raw = {
        task_id: build_offline_raw_features(red_entries, phase1_root, "reddit", task_id)[0]
        for task_id in red_ids
    }
    cls_oracle_labels = {
        task_id: derive_oracle_label(
            {mode: bool(cls_outcomes[task_id][mode]["success"]) for mode in DISPLAY_MODES}
        )
        for task_id in cls_ids
    }
    vectorizer, selected_mask, feature_meta = _fit_feature_asset(
        cls_ids, cls_raw, cls_oracle_labels
    )
    # The head helper expects one raw-feature mapping for train and predict; the task
    # IDs overlap across sites, so use disjoint synthetic IDs without changing rows.
    offset = max(cls_ids) + 1
    synthetic_red_ids = [offset + index for index in range(len(red_ids))]
    synthetic_to_red = dict(zip(synthetic_red_ids, red_ids))
    combined_raw = dict(cls_raw)
    combined_outcomes = dict(cls_outcomes)
    for synthetic_id, red_id in synthetic_to_red.items():
        combined_raw[synthetic_id] = red_raw[red_id]
        combined_outcomes[synthetic_id] = red_outcomes[red_id]
    synthetic_probabilities, fit_rows = _fit_binary_success_heads(
        cls_ids,
        synthetic_red_ids,
        combined_outcomes,
        combined_raw,
        vectorizer,
        selected_mask,
    )
    probabilities = {
        red_id: synthetic_probabilities[synthetic_id]
        for synthetic_id, red_id in synthetic_to_red.items()
    }
    context = _decision_context(cls_ids, cls_outcomes)
    selected: dict[int, str] = {}
    fallback_count = 0
    for task_id in red_ids:
        mode, fallback = decide_frozen_primary_mode(
            probabilities[task_id],
            context["mean_train_cost_usd_by_mode"],
            context["fallback_best_single_mode"],
        )
        selected[task_id] = mode
        fallback_count += int(fallback)
    metric = policy_metrics(red_outcomes, selected)
    refs = reference_points(red_outcomes)
    best = refs["best_single_mode"]
    metric.update(
        {
            "threshold": FROZEN_TAU,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_count / len(red_ids),
            "delta_vs_reddit_best_single_pp": (
                metric["success_rate_pct"] - best["success_rate_pct"]
            ),
            "delta_vs_reddit_best_single_mean_cost_usd": (
                metric["mean_total_billed_cost_usd"]
                - best["mean_total_billed_cost_usd"]
            ),
            "dominance_relation_transfer_vs_reddit_best_single": dominance_relation(
                _policy_point("D1_cls_to_red", "transfer", metric),
                _policy_point(f"fixed_{best['mode']}", "fixed", best),
            ),
        }
    )
    return {
        "design": "D1 bonus: all-224 B0_classifieds heads + feature pipeline frozen to reddit",
        "status": "SECONDARY_TRANSFER_BONUS_NOT_PRIMARY",
        "training_site": "classifieds",
        "evaluation_site": "reddit",
        "n_train_tasks": len(cls_ids),
        "n_evaluation_tasks": len(red_ids),
        "threshold": FROZEN_TAU,
        "decision_cost_source": "B0_classifieds all-task mean billed cost by mode",
        "fallback_source": "B0_classifieds best single by SR then lower cost",
        "feature_meta": feature_meta,
        "success_head_fit_rows": fit_rows,
        "cls_decision_context": context,
        "reddit_metric": metric,
        "paired_bootstrap_vs_reddit_best_single": _paired_bootstrap(
            red_outcomes,
            selected,
            str(best["mode"]),
            n_replicates=bootstrap_replicates,
            seed=BOOTSTRAP_SEED,
        ),
        "task_records": [
            {
                "task_id": task_id,
                "probability_success_by_mode": probabilities[task_id],
                "selected_mode": selected[task_id],
                "success": bool(red_outcomes[task_id][selected[task_id]]["success"]),
                "total_billed_cost_usd": float(
                    red_outcomes[task_id][selected[task_id]]["cost_usd"]
                ),
            }
            for task_id in red_ids
        ],
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def _fmt_cost(value: float) -> str:
    return f"${value:.6f}"


def _fmt_ci(values: list[float], *, suffix: str = "") -> str:
    return f"[{values[0]:+.3f}, {values[1]:+.3f}]{suffix}"


def render_markdown(payload: Mapping[str, Any]) -> str:
    d2 = payload["d2_primary"]
    refs = d2["reference_points"]
    primary = d2["primary_cost_aware_router"]
    best = refs["best_single_mode"]
    oracle = refs["six_mode_oracle_ceiling"]
    boot = d2["paired_bootstrap_vs_best_single"]
    lines = [
        f"# {ARTIFACT_STATUS}",
        "",
        f"**Artifact status:** `{ARTIFACT_STATUS}` · **Gate eligible:** `false`",
        "",
        f"> {DISCLAIMER}.",
        "",
        "## Freeze declaration",
        "",
        (
            f"The efficiency recipe was selected on **B0_classifieds only** and frozen "
            f"before B0_reddit was evaluated. The primary reddit decision uses exactly "
            f"one threshold, **tau*={FROZEN_TAU:.2f}**; no reddit-side threshold "
            "selection was performed. The 0.05/0.15 points appear only in the "
            "post-hoc appendix and are forbidden from the conclusion."
        ),
        "",
        "## D2 primary result — B0_reddit within-cell OOF",
        "",
        "| Policy | n | Success | SR | Mean billed cost/task | Delta SR vs best | Delta cost vs best | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for mode in DISPLAY_MODES:
        metric = refs["single_modes"][mode]
        lines.append(
            f"| Always {MODE_LABELS[mode]} | {metric['n_tasks']} | {metric['n_success']} | "
            f"{_fmt_pct(metric['success_rate_pct'])} | "
            f"{_fmt_cost(metric['mean_total_billed_cost_usd'])} | "
            f"{metric['success_rate_pct'] - best['success_rate_pct']:+.2f} pp | "
            f"{metric['mean_total_billed_cost_usd'] - best['mean_total_billed_cost_usd']:+.6f} | fixed |"
        )
    lines.extend(
        [
            f"| **Best single ({MODE_LABELS[best['mode']]})** | {best['n_tasks']} | "
            f"{best['n_success']} | **{_fmt_pct(best['success_rate_pct'])}** | "
            f"{_fmt_cost(best['mean_total_billed_cost_usd'])} | +0.00 pp | +0.000000 | reference |",
            f"| **Cost-aware OOF, frozen tau=0.10** | {primary['n_tasks']} | "
            f"{primary['n_success']} | **{_fmt_pct(primary['success_rate_pct'])}** | "
            f"**{_fmt_cost(primary['mean_total_billed_cost_usd'])}** | "
            f"**{primary['delta_vs_best_single_pp']:+.2f} pp** | "
            f"**{primary['delta_vs_best_single_mean_cost_usd']:+.6f}** | PRIMARY offline replication |",
            f"| Six-mode oracle | {oracle['n_tasks']} | {oracle['n_success']} | "
            f"{_fmt_pct(oracle['success_rate_pct'])} | "
            f"{_fmt_cost(oracle['mean_total_billed_cost_usd'])} | "
            f"{oracle['success_rate_pct'] - best['success_rate_pct']:+.2f} pp | "
            f"{oracle['mean_total_billed_cost_usd'] - best['mean_total_billed_cost_usd']:+.6f} | hindsight ceiling |",
            "",
            f"Point relation (router vs best single): `{primary['dominance_relation_router_vs_best_single']}`. "
            f"Deployable frontier: `{', '.join(d2['deployable_frontier'])}`.",
            "",
            "## Paired bootstrap",
            "",
            (
                f"Task-paired percentile bootstrap ({boot['n_replicates']} resamples, "
                f"seed {boot['seed']}): router-best DeltaSR "
                f"**{boot['delta_sr_pp']:+.2f} pp**, 95% CI "
                f"**{_fmt_ci(boot['delta_sr_pp_percentile_95_ci'], suffix=' pp')}**; "
                f"Delta mean cost **{boot['delta_mean_cost_usd']:+.6f} USD/task**, "
                f"95% CI **{_fmt_ci(boot['delta_mean_cost_usd_percentile_95_ci'])}**. "
                f"Router dominance rate={boot['router_pareto_dominates_best_single_fraction']:.1%}; "
                f"best-single dominance rate={boot['best_single_pareto_dominates_router_fraction']:.1%}."
            ),
            "",
            "## Preregistered-style LR mode-router control",
            "",
        ]
    )
    lr = d2["lr_mode_router_control"]
    if lr["oof_routed"] is None:
        fallback = lr["fallback_reference_when_untrainable"]
        lines.append(
            f"Status: **{lr['training_status']}** ({len(lr['folds_ok'])}/5 folds trainable "
            f"under the locked min-class rule). No full-cell LR OOF estimate is reported; "
            f"the fail-closed reference is always P-SoM at "
            f"{_fmt_pct(fallback['success_rate_pct'])}, "
            f"{_fmt_cost(fallback['mean_total_billed_cost_usd'])}/task."
        )
    else:
        metric = lr["oof_routed"]
        lines.append(
            f"Status: **TRAINED_COMPLETE**; OOF SR={_fmt_pct(metric['success_rate_pct'])}, "
            f"mean billed cost={_fmt_cost(metric['mean_total_billed_cost_usd'])}/task."
        )
    lines.extend(
        [
            "",
            "## Classifiers side by side",
            "",
            "| Cell/design | Threshold | SR | Mean billed cost/task | Selection status |",
            "|---|---:|---:|---:|---|",
            f"| B0_classifieds cost-aware OOF | 0.10 | "
            f"{_fmt_pct(payload['frozen_recipe']['source_curve_sr_pct'])} | "
            f"{_fmt_cost(payload['frozen_recipe']['source_curve_mean_cost_usd'])} | selected on cls |",
            f"| B0_reddit D2 within-cell OOF | **0.10 frozen** | "
            f"**{_fmt_pct(primary['success_rate_pct'])}** | "
            f"**{_fmt_cost(primary['mean_total_billed_cost_usd'])}** | one-shot primary |",
            "",
            "## Sensitivity appendix — post-hoc, not for the conclusion",
            "",
            "| tau | SR | Mean billed cost/task | Fallback | Status |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for threshold in SENSITIVITY_TAUS:
        metric = d2["sensitivity_appendix"]["results"][f"{threshold:.2f}"]
        lines.append(
            f"| {threshold:.2f} | {_fmt_pct(metric['success_rate_pct'])} | "
            f"{_fmt_cost(metric['mean_total_billed_cost_usd'])} | "
            f"{metric['fallback_rate']:.1%} | {metric['status']} |"
        )
    d1 = payload.get("d1_transfer_bonus")
    if d1 is not None:
        metric = d1["reddit_metric"]
        lines.extend(
            [
                "",
                "## D1 bonus — fully frozen classifieds-to-reddit transfer",
                "",
                (
                    f"All 224 B0_classifieds tasks retrain the frozen feature pipeline and "
                    f"six heads; classifier, tau, decision costs, and fallback are then "
                    f"applied unchanged to reddit. Result: **{_fmt_pct(metric['success_rate_pct'])}** "
                    f"at **{_fmt_cost(metric['mean_total_billed_cost_usd'])}/task** "
                    f"({metric['delta_vs_reddit_best_single_pp']:+.2f} pp and "
                    f"{metric['delta_vs_reddit_best_single_mean_cost_usd']:+.6f} USD/task "
                    f"vs reddit best single). This is secondary transfer evidence."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Provenance and limits",
            "",
            f"- Reddit fold-map content SHA-256: `{d2['fold_protocol']['fold_map_content_sha256']}`; "
            f"fold file SHA-256: `{payload['inputs']['reddit_fold_assignment_file_sha256']}`.",
            f"- Run manifest: `{payload['inputs']['run_manifest']}` "
            f"(SHA-256 `{payload['inputs']['run_manifest_sha256']}`).",
            f"- Frozen cls recipe: `{payload['inputs']['frozen_recipe']}` "
            f"(SHA-256 `{payload['inputs']['frozen_recipe_sha256']}`).",
            "- Costs replay selected held-out Pass-1 `total_billed_cost_usd`; router serving "
            "overhead, fresh-state interaction, and trajectory stochasticity are absent.",
            "- The oracle uses hindsight and is not deployable. D1 and the sensitivity "
            "curve are secondary; only D2 at frozen tau=0.10 may support the replication conclusion.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_replication(
    *,
    run_manifest: Path,
    phase1_root: Path,
    frozen_recipe_path: Path,
    out_dir: Path,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    assert_primary_threshold_frozen(FROZEN_TAU)
    if bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES:
        raise ValueError(
            f"bootstrap_replicates must be >= {MIN_BOOTSTRAP_REPLICATES}"
        )
    recipe, _source_payload = load_frozen_recipe(frozen_recipe_path)
    entries_by_cell = load_paper_grade_entries(
        run_manifest, [CELL_ID, SOURCE_CELL_ID]
    )
    red_outcomes, red_provenance = collect_cell_outcomes(
        CELL_ID, entries_by_cell[CELL_ID], phase1_root
    )
    cls_outcomes, cls_provenance = collect_cell_outcomes(
        SOURCE_CELL_ID, entries_by_cell[SOURCE_CELL_ID], phase1_root
    )
    if len(red_outcomes) != 205:
        raise ValueError(f"B0_reddit must contain exactly 205 tasks, got {len(red_outcomes)}")
    if len(cls_outcomes) != 224:
        raise ValueError(
            f"B0_classifieds transfer training must contain 224 tasks, got {len(cls_outcomes)}"
        )
    d2, fold_map = run_d2(
        red_outcomes,
        entries_by_cell[CELL_ID],
        phase1_root,
        bootstrap_replicates=bootstrap_replicates,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / "B0_reddit_fold_assignment.json"
    fold_payload = {
        "schema_version": SCHEMA_VERSION,
        "cell_id": CELL_ID,
        "site": "reddit",
        "n_splits": N_SPLITS,
        "seed": FOLD_SEED,
        "derivation": d2["fold_protocol"]["derivation"],
        "canonical_task_universe_sha256": red_provenance[
            "canonical_task_universe_sha256"
        ],
        "fold_assignment_content_sha256": fold_map_sha256(fold_map),
        "fold_assignment": {str(task_id): fold for task_id, fold in sorted(fold_map.items())},
    }
    fold_path.write_text(
        json.dumps(fold_payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )

    d1 = run_d1_transfer(
        red_outcomes,
        entries_by_cell[CELL_ID],
        cls_outcomes,
        entries_by_cell[SOURCE_CELL_ID],
        phase1_root,
        bootstrap_replicates=bootstrap_replicates,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_status": ARTIFACT_STATUS,
        "gate_eligible": False,
        "h10_eligible": False,
        "disclaimer": DISCLAIMER,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_recipe": recipe.__dict__,
        "selection_audit": {
            "selected_on_cell": SOURCE_CELL_ID,
            "frozen_before_evaluating_cell": CELL_ID,
            "primary_tau": FROZEN_TAU,
            "reddit_primary_thresholds_tried": [FROZEN_TAU],
            "reddit_threshold_selection_performed": False,
            "sensitivity_thresholds": list(SENSITIVITY_TAUS),
            "sensitivity_status": "POST-HOC CURVE; NOT USED FOR THE CONCLUSION",
            "conclusion_may_reference_only_tau": FROZEN_TAU,
        },
        "inputs": {
            "run_manifest": _relative(run_manifest),
            "run_manifest_sha256": sha256_file(run_manifest),
            "phase1_root": _relative(phase1_root),
            "frozen_recipe": _relative(frozen_recipe_path),
            "frozen_recipe_sha256": sha256_file(frozen_recipe_path),
            "reddit_fold_assignment": _relative(fold_path),
            "reddit_fold_assignment_file_sha256": sha256_file(fold_path),
            "reddit_outcome_provenance": red_provenance,
            "classifieds_outcome_provenance": cls_provenance,
        },
        "d2_primary": d2,
        "d1_transfer_bonus": d1,
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--phase1-root", type=Path, default=DEFAULT_PHASE1_ROOT)
    parser.add_argument("--frozen-recipe", type=Path, default=DEFAULT_FROZEN_RECIPE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    args = parser.parse_args()

    run_manifest = args.run_manifest.resolve()
    phase1_root = args.phase1_root.resolve()
    frozen_recipe = args.frozen_recipe.resolve()
    out_dir = args.out_dir.resolve()
    if not run_manifest.is_file():
        parser.error(f"Run manifest does not exist: {run_manifest}")
    if not phase1_root.is_dir():
        parser.error(f"Phase-1 root does not exist: {phase1_root}")
    if not frozen_recipe.is_file():
        parser.error(f"Frozen recipe does not exist: {frozen_recipe}")
    if args.bootstrap_replicates < MIN_BOOTSTRAP_REPLICATES:
        parser.error(
            f"--bootstrap-replicates must be >= {MIN_BOOTSTRAP_REPLICATES}"
        )

    payload = run_replication(
        run_manifest=run_manifest,
        phase1_root=phase1_root,
        frozen_recipe_path=frozen_recipe,
        out_dir=out_dir,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    json_path = out_dir / "cost_aware_router_b0reddit_replication.json"
    markdown_path = out_dir / "cost_aware_router_b0reddit_replication.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )
    markdown_path.write_text(render_markdown(payload))
    primary = payload["d2_primary"]["primary_cost_aware_router"]
    print(f"Wrote: {json_path}")
    print(f"Wrote: {markdown_path}")
    print(
        f"{ARTIFACT_STATUS}: B0_reddit tau={FROZEN_TAU:.2f}, "
        f"SR={primary['success_rate_pct']:.2f}%, "
        f"mean_cost=${primary['mean_total_billed_cost_usd']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
