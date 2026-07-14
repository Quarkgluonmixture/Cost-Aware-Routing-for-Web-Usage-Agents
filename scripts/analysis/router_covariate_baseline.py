#!/usr/bin/env python3
"""Router covariate-baseline + template-disjoint sensitivity (2606.22864 defense).

Attack surface (lit 2606.22864 "When AUC 0.998 Is Not Enough", CUA + Mind2Web +
Qwen-VL): a router/classifier's apparent learnability can be driven by trivial
covariates (site / text length / task template) rather than task semantics. VWA
tasks are TEMPLATE-INSTANTIATED (`intent_template_id` in every task config;
cls = 75 templates / 234 tasks, red = 87 / 210), and the canonical Stage-2
per-site KFold (B-1871) splits by TASK, so instances of the same template
appear on both sides of the split -> "learnable" may partly be "memorizable".

Two defenses, both evaluated under the EXACT canonical eval protocol
(per-site shared KFold seed=42, B-995 min-class filter, StandardScaler +
LogisticRegression(class_weight=None, C=1.0, max_iter=2000), pooled
out-of-fold prediction; Stage-2 fold-local TF-IDF + pooled-MI replicated
verbatim via imports from train_l1_router_with_mi / train_l1_router):

1. Scalar-covariate baselines: retrain the same LR head on trivial covariates
   only and compare AUROC / accuracy side-by-side with the full 18-feature LR.
     - scalar_min  : intent_char_len + intent_word_count + has_reference_image
     - scalar_plus : scalar_min + reasoning_difficulty
     - template_onehot_oracle : intent_template_id one-hot (explicit
       memorization UPPER reference, not a defensible feature set)
   Site one-hot is intentionally absent: the canonical architecture is
   per-cell (site is cell-constant, excluded by design — extract_50_features
   "note_cell_constant_excluded").

2. Template-disjoint split sensitivity: regenerate the per-site fold map with
   GroupKFold grouped by intent_template_id (all instances of a template share
   a fold), rerun every feature set, and report the AUROC/accuracy drop of the
   full LR standard -> disjoint. The template one-hot oracle collapses to the
   intercept under this regime by construction (unseen template -> zero row).

Metrics (per cell x feature set x split regime, pooled out-of-fold):
  - argmax mode-match accuracy (tau-free; canonical Stage-3 `cv_mode_match_acc`
    additionally applies the tau fallback rule — deployment rule, orthogonal to
    the feature-capacity question probed here)
  - macro one-vs-rest AUROC over classes with both positives and negatives in
    the pooled OOF set (multiclass mode-prediction AUROC; NOT the per-mode
    confidence-signal AUROC of aggregate_routing_auroc.py — different estimand)
  - majority-class accuracy reference

Data generation: reads a Stage-1 NPZ produced by extract_50_features.py.
Input and output paths are always explicit.  A path containing ``rehearsal``
requires ``--allow-rehearsal`` and is marked NON_PAPER_GRADE.

Usage:
    .venv/bin/python scripts/analysis/router_covariate_baseline.py \
        --raw-features results/phantom_paper/l1_router/raw_features_phase1a.npz \
        --out-json results/phantom_paper/l1_router/covariate_baseline.json

Analysis-layer only (not in the fire import path). No JSONL reads (Stage-1 NPZ
already aggregates episode outcomes via extract_50_features, which reads
summary JSONs).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from scripts.analysis.lib.atomic_io import atomic_write_text  # noqa: E402
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids  # noqa: E402

# Reuse the canonical pipeline pieces — do NOT copy logic (drift lesson B-1805..07).
from scripts.analysis.train_l1_router import (  # noqa: E402
    LR_C,
    LR_MAX_ITER,
    N_MIN_CLASS_TRAIN,
    apply_min_class_filter,
    build_design_matrix_for_indices,
)
from scripts.analysis.train_l1_router_with_mi import (  # noqa: E402
    FOLD_SEED,
    MI_SEED,
    N_SELECTED,
    N_SPLITS,
    _site_of_cell,
    build_design_matrix,
    build_pool_mask_for_fold,
    fit_fold_local_tfidf,
    fit_pooled_mi_selector,
    generate_per_cell_fold_assignments,
    load_raw_features,
)

VWA_CONFIG = REPO / "external/visualwebarena/config_files/vwa"

FEATURE_SETS = ["full_lr", "scalar_min", "scalar_plus", "template_onehot_oracle"]
SPLIT_REGIMES = ["standard", "template_disjoint"]
SCHEMA_VERSION = "2026-07-14-router-covariate-baseline-v2"
CANONICAL_CELL_IDS = {
    f"{baseline}_{site}"
    for baseline in ("B0", "B1", "B2")
    for site in ("classifieds", "reddit")
}
CONTRAST_BOOTSTRAP_B = 2000
CONTRAST_BOOTSTRAP_SEED = 42


# ---------------------------------------------------------------------------
# Template map
# ---------------------------------------------------------------------------

def load_template_map_with_diagnostics(site: str) -> tuple[dict[int, int], dict[str, Any]]:
    """Load task->template IDs while retaining parse/duplicate diagnostics."""
    site_dir = VWA_CONFIG / f"test_{site}"
    out: dict[int, int] = {}
    parse_errors: list[str] = []
    missing_template_ids: list[int] = []
    conflicting_task_ids: list[int] = []
    for f in site_dir.glob("[0-9]*.json"):
        try:
            cfg = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            parse_errors.append(str(f))
            continue
        if not isinstance(cfg, dict):
            parse_errors.append(str(f))
            continue
        tid = cfg.get("intent_template_id")
        try:
            task_id = int(f.stem)
        except ValueError:
            parse_errors.append(str(f))
            continue
        if tid is None:
            missing_template_ids.append(task_id)
            continue
        try:
            template_id = int(tid)
        except (TypeError, ValueError):
            missing_template_ids.append(task_id)
            continue
        if task_id in out and out[task_id] != template_id:
            conflicting_task_ids.append(task_id)
            continue
        out[task_id] = template_id
    return out, {
        "site": site,
        "parse_error_files": sorted(parse_errors),
        "config_tasks_missing_template_id": sorted(set(missing_template_ids)),
        "conflicting_task_ids": sorted(set(conflicting_task_ids)),
    }


def load_template_map(site: str) -> dict[int, int]:
    """Backward-compatible task_id -> intent_template_id loader."""
    return load_template_map_with_diagnostics(site)[0]


def template_coverage_report(
    raw: dict[str, Any],
    template_maps: dict[str, dict[int, int]],
    load_diagnostics: dict[str, dict[str, Any]],
    *,
    n_splits: int = N_SPLITS,
) -> dict[str, Any]:
    """Validate 100% one-template-per-evaluated-task coverage by site."""
    all_cell_ids = raw.get("all_cell_ids", raw["cell_ids"])
    all_task_ids = raw.get("all_task_ids", raw["task_ids"])
    tasks_by_site: dict[str, set[int]] = {}
    for cell_id, task_id in zip(all_cell_ids.tolist(), all_task_ids.tolist()):
        tasks_by_site.setdefault(_site_of_cell(str(cell_id)), set()).add(int(task_id))

    sites: dict[str, dict[str, Any]] = {}
    valid = True
    for site, task_ids in sorted(tasks_by_site.items()):
        tmap = template_maps.get(site, {})
        missing = sorted(task_ids - set(tmap))
        present = sorted(task_ids & set(tmap))
        n_groups = len({tmap[t] for t in present})
        diag = load_diagnostics.get(site, {})
        conflicts = sorted(set(diag.get("conflicting_task_ids", [])) & task_ids)
        site_valid = not missing and not conflicts and n_groups >= n_splits
        valid = valid and site_valid
        sites[site] = {
            "n_evaluated_tasks": len(task_ids),
            "n_tasks_with_exactly_one_template": len(present) - len(conflicts),
            "coverage_pct": 100.0 * (len(present) - len(conflicts)) / len(task_ids)
            if task_ids else 0.0,
            "missing_task_ids": missing,
            "conflicting_task_ids": conflicts,
            "n_unique_groups": n_groups,
            "required_n_splits": n_splits,
            "valid": site_valid,
            "parse_error_files": diag.get("parse_error_files", []),
        }
    return {"valid": valid and bool(sites), "sites": sites}


# ---------------------------------------------------------------------------
# Fold maps
# ---------------------------------------------------------------------------

def standard_fold_assignments(raw: dict[str, Any]) -> dict[str, dict[int, int]]:
    """Canonical per-site shared pure KFold (B-1871, seed=42) — exact Stage-2 call."""
    return generate_per_cell_fold_assignments(
        raw["cell_ids"],
        raw["task_ids"],
        raw["labels"],
        all_cell_ids=raw.get("all_cell_ids"),
        all_task_ids=raw.get("all_task_ids"),
        seed=FOLD_SEED,
        n_splits=N_SPLITS,
    )


def template_disjoint_fold_assignments(
    raw: dict[str, Any],
    template_maps: dict[str, dict[int, int]],
    n_splits: int = N_SPLITS,
    *,
    diagnostic: bool = False,
) -> dict[str, dict[int, int]]:
    """Per-site GroupKFold with groups = intent_template_id (same shape as canonical).

    Mirrors generate_per_cell_fold_assignments' universe construction (labeled +
    no-success tasks, per-site shared, cells of a site agree on every task) but
    the split is group-disjoint on template: all instances of one template land
    in the same fold, so a holdout task's template is never seen at train time.
    GroupKFold is deterministic (greedy size-balanced), no shuffle/seed.
    """
    cell_ids = raw["cell_ids"]
    task_ids = raw["task_ids"]
    universe_by_cell: dict[str, set[int]] = {}
    for c, t in zip(cell_ids.tolist(), task_ids.tolist()):
        universe_by_cell.setdefault(str(c), set()).add(int(t))
    all_cell_ids = raw.get("all_cell_ids")
    all_task_ids = raw.get("all_task_ids")
    if all_cell_ids is not None and all_task_ids is not None:
        for c, t in zip(all_cell_ids.tolist(), all_task_ids.tolist()):
            universe_by_cell.setdefault(str(c), set()).add(int(t))

    site_universe: dict[str, set[int]] = {}
    for cell_id, tasks in universe_by_cell.items():
        site_universe.setdefault(_site_of_cell(cell_id), set()).update(tasks)

    site_fold_map: dict[str, dict[int, int]] = {}
    for site, tasks in site_universe.items():
        tmap = template_maps.get(site, {})
        tasks_sorted = np.array(sorted(tasks), dtype=int)
        missing = [int(t) for t in tasks_sorted if int(t) not in tmap]
        if missing and not diagnostic:
            raise ValueError(
                f"{site}: missing template IDs for {len(missing)} evaluated tasks; "
                "singleton fallback requires --diagnostic"
            )
        # Diagnostic-only fallback: missing template IDs become singleton groups.
        groups = np.array(
            [tmap.get(int(t), 10_000_000 + int(t)) for t in tasks_sorted], dtype=int
        )
        n_groups = len(set(groups.tolist()))
        if n_groups < n_splits and not diagnostic:
            raise ValueError(
                f"{site}: only {n_groups} unique template groups; need >= {n_splits}"
            )
        n_splits_eff = min(n_splits, n_groups, len(tasks_sorted))
        fold_map: dict[int, int] = {}
        if len(tasks_sorted) == 0:
            site_fold_map[site] = fold_map
            continue
        if n_splits_eff < 2:
            if not diagnostic:
                raise ValueError(f"{site}: template-disjoint split needs >=2 groups")
            for t in tasks_sorted:
                fold_map[int(t)] = 0
            site_fold_map[site] = fold_map
            continue
        gkf = GroupKFold(n_splits=n_splits_eff)
        for fold_k, (_tr, ho) in enumerate(gkf.split(tasks_sorted, groups=groups)):
            for local_idx in ho:
                fold_map[int(tasks_sorted[local_idx])] = fold_k
        site_fold_map[site] = fold_map

    out: dict[str, dict[int, int]] = {}
    for cell_id in sorted(universe_by_cell.keys()):
        shared = site_fold_map[_site_of_cell(cell_id)]
        out[cell_id] = {t: shared[t] for t in sorted(universe_by_cell[cell_id])}
    return out


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def scalar_features(
    raw: dict[str, Any], plus_difficulty: bool
) -> tuple[np.ndarray, list[str]]:
    """Trivial-covariate design matrix over ALL pooled labeled rows.

    intent_char_len / intent_word_count are template-level shallow features
    (instantiation shifts them by a few chars); has_reference_image is a
    template-level property; reasoning_difficulty is a task-config annotation.
    NPZ column order contract (extract_50_features feature_names_numeric):
    numeric = [dom_complexity, text_length, tokens_input_text,
    intent_token_count, reasoning_difficulty]; binary[:, 0] = has_reference_image.
    """
    intents = raw["intents"]
    char_len = np.array([len(s) for s in intents], dtype=float)
    word_cnt = raw["X_numeric"][:, 3].astype(float)  # intent_token_count
    has_img = raw["X_binary"][:, 0].astype(float)  # has_reference_image
    cols = [char_len, word_cnt, has_img]
    names = ["intent_char_len", "intent_word_count", "has_reference_image"]
    if plus_difficulty:
        cols.append(raw["X_numeric"][:, 4].astype(float))  # reasoning_difficulty
        names.append("reasoning_difficulty")
    return np.column_stack(cols), names


def template_onehot(
    train_templates: np.ndarray, holdout_templates: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """One-hot on train-fold template categories; unseen holdout template -> zero row."""
    cats = sorted(set(train_templates.tolist()))
    idx = {c: j for j, c in enumerate(cats)}
    X_tr = np.zeros((len(train_templates), len(cats)))
    for i, t in enumerate(train_templates):
        X_tr[i, idx[t]] = 1.0
    X_ho = np.zeros((len(holdout_templates), len(cats)))
    for i, t in enumerate(holdout_templates):
        j = idx.get(t)
        if j is not None:
            X_ho[i, j] = 1.0
    return X_tr, X_ho


def make_pipeline() -> Pipeline:
    """Canonical Stage-3 head (train_l1_router.py hyperparams, imported constants)."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    class_weight=None, max_iter=LR_MAX_ITER, C=LR_C, solver="lbfgs"
                ),
            ),
        ]
    )


def fold_local_majority_prediction(
    y_train: np.ndarray | list[str], holdout_n: int,
) -> tuple[list[str], dict[str, Any]]:
    """Choose a fold's majority from training labels only.

    Ties are resolved by the lexicographically smallest class name.  Returning
    predictions and metadata from one helper keeps the reported fold audit and
    the actual OOF baseline predictions on the same deterministic rule.
    """
    labels = [str(y) for y in list(y_train)]
    if not labels:
        raise ValueError("cannot choose fold-local majority from empty training labels")
    counts = Counter(labels)
    max_count = max(counts.values())
    majority_class = min(c for c, count in counts.items() if count == max_count)
    return [majority_class] * int(holdout_n), {
        "majority_class": majority_class,
        "majority_count": int(max_count),
        "train_n": len(labels),
        "class_counts": {c: int(counts[c]) for c in sorted(counts)},
        "tie_rule": "lexicographically_smallest_class",
    }


# ---------------------------------------------------------------------------
# Stage-2 replication per fold (full_lr only)
# ---------------------------------------------------------------------------

def fit_stage2_for_fold(
    raw: dict[str, Any],
    fold_assignments: dict[str, dict[int, int]],
    fold_k: int,
    k_select: int = N_SELECTED,
) -> tuple[Any, np.ndarray]:
    """Replicate Stage 2 for one fold: pool -> TF-IDF -> pooled-MI top-18 mask."""
    pool_mask = build_pool_mask_for_fold(
        raw["cell_ids"], raw["task_ids"], fold_assignments, fold_k
    )
    n_total = len(raw["intents"])
    pool_intents = [raw["intents"][i] for i in range(n_total) if pool_mask[i]]
    vectorizer = fit_fold_local_tfidf(pool_intents)
    X_pool_full, _ = build_design_matrix(
        pool_intents,
        raw["X_numeric"][pool_mask],
        raw["X_binary"][pool_mask],
        vectorizer,
    )
    _, selected_mask = fit_pooled_mi_selector(
        X_pool_full,
        raw["labels"][pool_mask],
        k=k_select,
        seed=MI_SEED,
        n_binary=raw["X_binary"].shape[1],
    )
    return vectorizer, selected_mask


# ---------------------------------------------------------------------------
# Per-cell OOF evaluation
# ---------------------------------------------------------------------------

def eval_cell(
    cell_id: str,
    feature_set: str,
    raw: dict[str, Any],
    fold_assignments: dict[str, dict[int, int]],
    stage2_cache: dict[int, tuple[Any, np.ndarray]],
    template_maps: dict[str, dict[int, int]],
) -> dict[str, Any]:
    """Pooled out-of-fold eval of one (cell, feature_set) under one fold map."""
    cell_ids = raw["cell_ids"]
    task_ids = raw["task_ids"]
    labels = raw["labels"]
    fa = fold_assignments.get(cell_id, {})

    cell_mask = cell_ids == cell_id
    cell_indices = np.where(cell_mask)[0]
    if len(cell_indices) == 0 or not fa:
        return {"status": "no_data", "cell_id": cell_id, "feature_set": feature_set}

    site = _site_of_cell(cell_id)
    tmap = template_maps.get(site, {})

    scalar_X = None
    if feature_set in ("scalar_min", "scalar_plus"):
        scalar_X, scalar_names = scalar_features(
            raw, plus_difficulty=(feature_set == "scalar_plus")
        )

    global_classes = sorted(set(labels[cell_indices].tolist()))
    cls_idx = {c: j for j, c in enumerate(global_classes)}

    oof_true: list[str] = []
    oof_proba: list[np.ndarray] = []
    oof_task_ids: list[int] = []
    oof_majority_pred: list[str] = []
    per_fold_majority: list[dict[str, Any]] = []
    n_skipped_holdout = 0
    folds_ok: list[int] = []
    folds_failed: dict[int, str] = {}
    fold_values = sorted(set(fa.values()))

    for fold_k in fold_values:
        train_local, hold_local = [], []
        for local_i, gi in enumerate(cell_indices):
            tid = int(task_ids[gi])
            if fa.get(tid) == fold_k:
                hold_local.append(local_i)
            else:
                train_local.append(local_i)
        if not hold_local:
            continue
        train_global = cell_indices[np.array(train_local, dtype=int)]
        hold_global = cell_indices[np.array(hold_local, dtype=int)]
        y_train = labels[train_global]
        y_hold = labels[hold_global]

        # B-995 min-class filter — identical to canonical Stage 3.
        y_train_f, train_kept_global, _dropped = apply_min_class_filter(
            y_train, train_global, min_n=N_MIN_CLASS_TRAIN
        )
        if len(y_train_f) < 2 or len(set(y_train_f.tolist())) < 2:
            folds_failed[fold_k] = "insufficient_train_data_post_min_class_filter"
            n_skipped_holdout += len(hold_global)
            continue

        try:
            if feature_set == "full_lr":
                vectorizer, sel_mask = stage2_cache[fold_k]
                X_tr = build_design_matrix_for_indices(
                    train_kept_global, raw["intents"], raw["X_numeric"],
                    raw["X_binary"], vectorizer, sel_mask,
                )
                X_ho = build_design_matrix_for_indices(
                    hold_global, raw["intents"], raw["X_numeric"],
                    raw["X_binary"], vectorizer, sel_mask,
                )
            elif feature_set in ("scalar_min", "scalar_plus"):
                X_tr = scalar_X[train_kept_global]
                X_ho = scalar_X[hold_global]
            elif feature_set == "template_onehot_oracle":
                tr_templates = np.array(
                    [tmap.get(int(task_ids[g]), -1) for g in train_kept_global]
                )
                ho_templates = np.array(
                    [tmap.get(int(task_ids[g]), -1) for g in hold_global]
                )
                X_tr, X_ho = template_onehot(tr_templates, ho_templates)
            else:
                raise ValueError(f"unknown feature_set {feature_set}")

            pipe = make_pipeline()
            pipe.fit(X_tr, y_train_f)
            proba = pipe.predict_proba(X_ho)
        except (ValueError, KeyError) as exc:
            folds_failed[fold_k] = f"fit_or_transform_failed: {exc}"
            n_skipped_holdout += len(hold_global)
            continue

        proba_full = np.zeros((len(hold_global), len(global_classes)))
        for j, c in enumerate(pipe.classes_):
            if c in cls_idx:
                proba_full[:, cls_idx[c]] = proba[:, j]
        oof_true.extend(y_hold.tolist())
        oof_proba.append(proba_full)
        oof_task_ids.extend(int(task_ids[g]) for g in hold_global)
        majority_pred, majority_meta = fold_local_majority_prediction(
            y_train_f, len(hold_global),
        )
        oof_majority_pred.extend(majority_pred)
        per_fold_majority.append({
            "fold": int(fold_k),
            "holdout_n": int(len(hold_global)),
            **majority_meta,
        })
        folds_ok.append(fold_k)

    if not oof_proba:
        return {
            "status": "untrainable",
            "cell_id": cell_id,
            "feature_set": feature_set,
            "folds_failed": folds_failed,
            "n_labeled": int(len(cell_indices)),
        }

    P = np.vstack(oof_proba)
    y = np.array(oof_true)
    pred = np.array([global_classes[j] for j in P.argmax(axis=1)])
    acc = float((pred == y).mean())
    majority_pred_arr = np.array(oof_majority_pred)
    majority_acc = float((y == majority_pred_arr).mean())

    per_class_auroc: dict[str, float] = {}
    per_class_n: dict[str, int] = {}
    for c in global_classes:
        pos = (y == c).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue
        try:
            per_class_auroc[c] = float(roc_auc_score(pos, P[:, cls_idx[c]]))
            per_class_n[c] = int(pos.sum())
        except ValueError:
            continue
    macro_auroc = float(np.mean(list(per_class_auroc.values()))) if per_class_auroc else None
    total_pos = sum(per_class_n.values())
    weighted_auroc = (
        float(
            sum(per_class_auroc[c] * per_class_n[c] for c in per_class_auroc) / total_pos
        )
        if total_pos
        else None
    )
    metric_status = "ok" if per_class_auroc else "undefined_single_class"

    return {
        "status": "ok",
        "cell_id": cell_id,
        "feature_set": feature_set,
        "n_labeled": int(len(cell_indices)),
        "n_oof": int(len(y)),
        # Per-row OOF dump (paired-bootstrap substrate: rows keyed by task_id are
        # alignable across feature sets / regimes evaluated on the same cell).
        "oof_rows": {
            "task_ids": oof_task_ids,
            "y_true": y.tolist(),
            "y_pred": pred.tolist(),
            "majority_pred": majority_pred_arr.tolist(),
            "classes": global_classes,
            "proba": np.round(P, 6).tolist(),
        },
        "n_skipped_holdout": int(n_skipped_holdout),
        "folds_ok": folds_ok,
        "folds_failed": folds_failed,
        "accuracy": acc,
        "majority_strategy": "fold_local_train_labels",
        "majority_tie_rule": "lexicographically_smallest_class",
        "per_fold_majority": per_fold_majority,
        "majority_acc": majority_acc,
        "macro_ovr_auroc": macro_auroc,
        "weighted_ovr_auroc": weighted_auroc,
        "metric_status": metric_status,
        "per_class_auroc": per_class_auroc,
        "per_class_n_pos": per_class_n,
        "n_classes_scored": len(per_class_auroc),
    }


# ---------------------------------------------------------------------------
# Canonical input validation + paired contrasts
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json_dumps(payload: dict[str, Any]) -> str:
    """Serialize standards-compliant JSON; NaN/Infinity are hard errors."""
    def default(value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(
            f"Object of type {type(value).__name__} is not JSON serializable"
        )

    return json.dumps(
        payload, indent=2, allow_nan=False, default=default,
    ) + "\n"


def validate_canonical_raw(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Require exact six cells and canonical full task universes."""
    errors: list[str] = []
    labeled_cells = {str(c) for c in raw["cell_ids"].tolist()}
    all_cell_ids = raw.get("all_cell_ids", raw["cell_ids"])
    all_task_ids = raw.get("all_task_ids", raw["task_ids"])
    universe_cells = {str(c) for c in all_cell_ids.tolist()}
    if labeled_cells != CANONICAL_CELL_IDS:
        errors.append(
            "labeled cell IDs must equal canonical six: "
            f"missing={sorted(CANONICAL_CELL_IDS - labeled_cells)} "
            f"extra={sorted(labeled_cells - CANONICAL_CELL_IDS)}"
        )
    if universe_cells != CANONICAL_CELL_IDS:
        errors.append(
            "full-universe cell IDs must equal canonical six: "
            f"missing={sorted(CANONICAL_CELL_IDS - universe_cells)} "
            f"extra={sorted(universe_cells - CANONICAL_CELL_IDS)}"
        )

    cells: dict[str, Any] = {}
    for cell_id in sorted(CANONICAL_CELL_IDS):
        labeled_tids = [
            int(t) for c, t in zip(raw["cell_ids"].tolist(), raw["task_ids"].tolist())
            if str(c) == cell_id
        ]
        universe_tids = [
            int(t) for c, t in zip(all_cell_ids.tolist(), all_task_ids.tolist())
            if str(c) == cell_id
        ]
        site = _site_of_cell(cell_id)
        expected_ids, expected_sha = expected_scored_ids(site)
        if not labeled_tids:
            errors.append(f"{cell_id}: zero labeled canonical rows")
        if len(labeled_tids) != len(set(labeled_tids)):
            errors.append(f"{cell_id}: duplicate labeled task rows")
        if not set(labeled_tids).issubset(expected_ids):
            errors.append(f"{cell_id}: labeled rows contain non-canonical task IDs")
        if set(universe_tids) != expected_ids or len(universe_tids) != len(expected_ids):
            errors.append(
                f"{cell_id}: full task universe is not exact canonical set "
                f"({len(set(universe_tids))}/{len(expected_ids)})"
            )
        cells[cell_id] = {
            "n_labeled_rows": len(labeled_tids),
            "n_canonical_rows": len(universe_tids),
            "expected_n": len(expected_ids),
            "task_set_sha256": expected_sha,
            "complete_exact": set(universe_tids) == expected_ids
            and len(universe_tids) == len(expected_ids),
        }
    return {"cells": cells, "complete_exact": not errors}, errors


def _macro_ovr_from_arrays(
    y_true: np.ndarray,
    classes: list[str],
    proba: np.ndarray,
) -> Optional[float]:
    values: list[float] = []
    for j, class_name in enumerate(classes):
        pos = (y_true == class_name).astype(int)
        if pos.sum() == 0 or pos.sum() == len(pos):
            continue
        try:
            values.append(float(roc_auc_score(pos, proba[:, j])))
        except ValueError:
            continue
    return float(np.mean(values)) if values else None


def paired_auroc_contrast(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    contrast_id: str,
    B: int = CONTRAST_BOOTSTRAP_B,
    seed: int = CONTRAST_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired macro-OVR AUROC delta on the task-ID intersection."""
    base = {
        "contrast_id": contrast_id,
        "cell_id": left.get("cell_id") or right.get("cell_id"),
        "left": {
            "feature_set": left.get("feature_set"),
            "split_regime": left.get("split_regime"),
        },
        "right": {
            "feature_set": right.get("feature_set"),
            "split_regime": right.get("split_regime"),
        },
    }
    if left.get("status") != "ok" or right.get("status") != "ok":
        return {
            **base,
            "status": "NOT_EVALUABLE",
            "metric_status": "missing_oof_result",
            "n_left": 0,
            "n_right": 0,
            "n_common": 0,
            "dropped_ids": {"left_only": [], "right_only": []},
            "delta_auroc": None,
            "ci95": None,
            "bootstrap_B": B,
            "bootstrap_seed": seed,
            "bootstrap_valid_replicates": 0,
        }

    def unpack(record: dict[str, Any]) -> tuple[dict[int, tuple[str, np.ndarray]], list[str]]:
        rows = record["oof_rows"]
        task_ids = [int(t) for t in rows["task_ids"]]
        if len(task_ids) != len(set(task_ids)):
            raise ValueError(
                f"{record.get('cell_id')}/{record.get('feature_set')}/"
                f"{record.get('split_regime')}: duplicate OOF task IDs"
            )
        classes = [str(c) for c in rows["classes"]]
        mapping = {
            task_id: (str(y), np.asarray(p, dtype=float))
            for task_id, y, p in zip(task_ids, rows["y_true"], rows["proba"])
        }
        return mapping, classes

    left_rows, left_classes = unpack(left)
    right_rows, right_classes = unpack(right)
    left_ids, right_ids = set(left_rows), set(right_rows)
    common_ids = sorted(left_ids & right_ids)
    dropped = {
        "left_only": sorted(left_ids - right_ids),
        "right_only": sorted(right_ids - left_ids),
    }
    result = {
        **base,
        "n_left": len(left_ids),
        "n_right": len(right_ids),
        "n_common": len(common_ids),
        "dropped_ids": dropped,
        "bootstrap_B": B,
        "bootstrap_seed": seed,
    }
    if not common_ids:
        return {
            **result,
            "status": "NOT_EVALUABLE",
            "metric_status": "empty_task_intersection",
            "delta_auroc": None,
            "ci95": None,
            "bootstrap_valid_replicates": 0,
        }

    y_left = np.array([left_rows[t][0] for t in common_ids])
    y_right = np.array([right_rows[t][0] for t in common_ids])
    if not np.array_equal(y_left, y_right):
        mismatched = [
            t for t, yl, yr in zip(common_ids, y_left.tolist(), y_right.tolist())
            if yl != yr
        ]
        raise AssertionError(
            f"{contrast_id}: y_true mismatch on common task IDs {mismatched}"
        )
    left_p = np.vstack([left_rows[t][1] for t in common_ids])
    right_p = np.vstack([right_rows[t][1] for t in common_ids])
    left_auc = _macro_ovr_from_arrays(y_left, left_classes, left_p)
    right_auc = _macro_ovr_from_arrays(y_left, right_classes, right_p)
    if left_auc is None or right_auc is None:
        return {
            **result,
            "status": "NOT_EVALUABLE",
            "metric_status": "undefined_single_class",
            "left_auroc_on_common": left_auc,
            "right_auroc_on_common": right_auc,
            "delta_auroc": None,
            "ci95": None,
            "bootstrap_valid_replicates": 0,
        }

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n = len(common_ids)
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        left_b = _macro_ovr_from_arrays(y_left[idx], left_classes, left_p[idx])
        right_b = _macro_ovr_from_arrays(y_left[idx], right_classes, right_p[idx])
        if left_b is not None and right_b is not None:
            deltas.append(left_b - right_b)
    ci95 = (
        [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))]
        if deltas else None
    )
    return {
        **result,
        "status": "ok" if deltas else "NOT_EVALUABLE",
        "metric_status": "ok" if deltas else "undefined_single_class",
        "left_auroc_on_common": left_auc,
        "right_auroc_on_common": right_auc,
        "delta_auroc": left_auc - right_auc,
        "ci95": ci95,
        "bootstrap_valid_replicates": len(deltas),
    }


def build_predefined_contrasts(
    results: list[dict[str, Any]],
    *,
    B: int = CONTRAST_BOOTSTRAP_B,
    seed: int = CONTRAST_BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    """Emit the locked full-vs-scalar and standard-vs-template contrasts."""
    index = {
        (r.get("cell_id"), r.get("feature_set"), r.get("split_regime")): r
        for r in results
    }
    cells = sorted({str(r.get("cell_id")) for r in results if r.get("cell_id")})
    contrasts: list[dict[str, Any]] = []
    missing = {"status": "missing"}
    for cell_id in cells:
        for regime in SPLIT_REGIMES:
            left = index.get((cell_id, "full_lr", regime), missing)
            right = index.get((cell_id, "scalar_min", regime), missing)
            contrasts.append(paired_auroc_contrast(
                left, right,
                contrast_id=f"full-vs-scalar:{regime}", B=B, seed=seed,
            ))
        left = index.get((cell_id, "full_lr", "standard"), missing)
        right = index.get((cell_id, "full_lr", "template_disjoint"), missing)
        contrasts.append(paired_auroc_contrast(
            left, right,
            contrast_id="standard-vs-template-disjoint:full_lr", B=B, seed=seed,
        ))
    return contrasts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-features", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--cells", nargs="+", default=None, help="subset of cell_ids")
    ap.add_argument(
        "--allow-rehearsal", action="store_true",
        help="Permit paths containing 'rehearsal'; output is marked NON_PAPER_GRADE.",
    )
    ap.add_argument(
        "--diagnostic", action="store_true",
        help="Allow invalid template coverage with singleton fallback; never paper-grade.",
    )
    args = ap.parse_args()

    npz_path = Path(args.raw_features)
    out_path = Path(args.out_json)
    rehearsal_path = "rehearsal" in str(npz_path).lower() or "rehearsal" in str(out_path).lower()
    if rehearsal_path and not args.allow_rehearsal:
        print(
            "error: rehearsal path requires explicit --allow-rehearsal",
            file=sys.stderr,
        )
        return 2
    if not npz_path.is_file():
        print(f"error: raw feature NPZ not found: {npz_path}", file=sys.stderr)
        return 2

    raw = load_raw_features(npz_path)
    cells = sorted(set(str(c) for c in raw["cell_ids"].tolist()))
    if args.cells:
        cells = [c for c in cells if c in set(args.cells)]
    paper_grade = not rehearsal_path and not args.diagnostic
    canonical_input_report: dict[str, Any] = {}
    if paper_grade:
        canonical_input_report, canonical_errors = validate_canonical_raw(raw)
        if set(cells) != CANONICAL_CELL_IDS:
            canonical_errors.append(
                "selected output cells must equal the exact canonical six; "
                f"got {sorted(cells)}"
            )
        if canonical_errors:
            for error in canonical_errors:
                print(f"error: {error}", file=sys.stderr)
            return 2

    sites = sorted({_site_of_cell(c) for c in cells})
    loaded = {s: load_template_map_with_diagnostics(s) for s in sites}
    template_maps = {s: loaded[s][0] for s in sites}
    load_diagnostics = {s: loaded[s][1] for s in sites}
    coverage = template_coverage_report(raw, template_maps, load_diagnostics)
    for s in sites:
        n_tasks_with = len(template_maps[s])
        n_templates = len(set(template_maps[s].values()))
        print(f"[template map] {s}: {n_tasks_with} tasks, {n_templates} templates")
    if not coverage["valid"] and not args.diagnostic:
        print(
            "error: INVALID_TEMPLATE_COVERAGE; use --diagnostic only for "
            "singleton-fallback diagnostics",
            file=sys.stderr,
        )
        for site, report in coverage["sites"].items():
            if not report["valid"]:
                print(
                    f"error: {site}: coverage={report['coverage_pct']:.1f}% "
                    f"groups={report['n_unique_groups']}/{report['required_n_splits']} "
                    f"missing={report['missing_task_ids']} "
                    f"conflicts={report['conflicting_task_ids']}",
                    file=sys.stderr,
                )
        return 2

    fold_maps = {
        "standard": standard_fold_assignments(raw),
        "template_disjoint": template_disjoint_fold_assignments(
            raw, template_maps, diagnostic=args.diagnostic,
        ),
    }

    # Template-leak audit of the canonical split: fraction of labeled holdout
    # tasks whose template also appears in the same cell's train fold.
    leak_audit: dict[str, dict[str, float]] = {}
    for regime, fa_all in fold_maps.items():
        leak_audit[regime] = {}
        for cell_id in cells:
            fa = fa_all.get(cell_id, {})
            cmask = raw["cell_ids"] == cell_id
            tids = raw["task_ids"][cmask]
            tmap = template_maps.get(_site_of_cell(cell_id), {})
            n_leak, n_tot = 0, 0
            for tid in tids.tolist():
                fk = fa.get(int(tid))
                if fk is None:
                    continue
                tmpl = tmap.get(int(tid))
                n_tot += 1
                train_templates = {
                    tmap.get(int(t2))
                    for t2 in tids.tolist()
                    if fa.get(int(t2)) != fk
                }
                if tmpl in train_templates:
                    n_leak += 1
            leak_audit[regime][cell_id] = (n_leak / n_tot) if n_tot else None

    results: list[dict[str, Any]] = []
    for regime in SPLIT_REGIMES:
        fa_all = fold_maps[regime]
        fold_values = sorted(
            {fk for cell_id in cells for fk in fa_all.get(cell_id, {}).values()}
        )
        stage2_cache = {
            fk: fit_stage2_for_fold(raw, fa_all, fk) for fk in fold_values
        }
        for feature_set in FEATURE_SETS:
            for cell_id in cells:
                rec = eval_cell(
                    cell_id, feature_set, raw, fa_all, stage2_cache, template_maps
                )
                rec["split_regime"] = regime
                results.append(rec)
                if rec["status"] == "ok":
                    macro = rec["macro_ovr_auroc"]
                    weighted = rec["weighted_ovr_auroc"]
                    macro_s = f"{macro:.3f}" if macro is not None else "undefined"
                    weighted_s = f"{weighted:.3f}" if weighted is not None else "undefined"
                    print(
                        f"[{regime:17s}] {cell_id:15s} {feature_set:22s} "
                        f"n_oof={rec['n_oof']:3d} acc={rec['accuracy']:.3f} "
                        f"(maj={rec['majority_acc']:.3f}) "
                        f"macroAUROC={macro_s} wAUROC={weighted_s}"
                    )
                else:
                    print(
                        f"[{regime:17s}] {cell_id:15s} {feature_set:22s} "
                        f"status={rec['status']}"
                    )

    contrasts = build_predefined_contrasts(results)
    if args.diagnostic and not coverage["valid"]:
        analysis_status = "INVALID_TEMPLATE_COVERAGE"
    elif paper_grade:
        analysis_status = "COMPLETE"
    else:
        analysis_status = "NON_PAPER_GRADE"
    out = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "grade": "PAPER_GRADE" if paper_grade else "NON_PAPER_GRADE",
        "analysis_status": analysis_status,
        "raw_features_npz": str(npz_path),
        "raw_features_sha256": _sha256_file(npz_path),
        "raw_features_schema": raw["meta"].get("schema_version"),
        "n_pooled_labeled": int(len(raw["task_ids"])),
        "cells": cells,
        "feature_sets": FEATURE_SETS,
        "split_regimes": SPLIT_REGIMES,
        "canonical_protocol": {
            "fold_seed": FOLD_SEED,
            "n_splits": N_SPLITS,
            "min_class_n_train": N_MIN_CLASS_TRAIN,
            "lr_c": LR_C,
            "lr_max_iter": LR_MAX_ITER,
            "mi_k": N_SELECTED,
            "note": (
                "standard regime = canonical B-1871 per-site shared pure KFold; "
                "template_disjoint = GroupKFold(intent_template_id) over the same "
                "site universe. full_lr replicates Stage 2 (fold-local TF-IDF + "
                "pooled-MI top-18) + Stage 3 head per fold via imports."
            ),
        },
        "canonical_input_validation": canonical_input_report,
        "template_coverage": coverage,
        "template_leak_audit_holdout_frac_with_template_in_train": leak_audit,
        "results": results,
        "paired_contrasts": contrasts,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(
        out_path,
        strict_json_dumps(out),
    )
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
