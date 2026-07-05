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
Default = the 2026-07-02 rehearsal artifact built from landed Pass-1
paper-grade runs (4 cells present: B0_cls 97 / B0_red 51 / B1_cls 55 /
B2_cls 16 labeled tasks). Pre-Pass-2 vintage: oracle labels are N=1 draws.

Usage:
    .venv/bin/python scripts/analysis/router_covariate_baseline.py \
        [--raw-features results/phantom_paper/l1_router_rehearsal_20260702/raw_features_phase1a.npz] \
        [--out-json results/phantom_paper/l1_router_rehearsal_20260702/covariate_baseline.json]

Analysis-layer only (not in the fire import path). No JSONL reads (Stage-1 NPZ
already aggregates episode outcomes via extract_50_features, which reads
summary JSONs).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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

DEFAULT_NPZ = (
    REPO / "results/phantom_paper/l1_router_rehearsal_20260702/raw_features_phase1a.npz"
)
VWA_CONFIG = REPO / "external/visualwebarena/config_files/vwa"

FEATURE_SETS = ["full_lr", "scalar_min", "scalar_plus", "template_onehot_oracle"]
SPLIT_REGIMES = ["standard", "template_disjoint"]
SCHEMA_VERSION = "2026-07-05-router-covariate-baseline-v1"


# ---------------------------------------------------------------------------
# Template map
# ---------------------------------------------------------------------------

def load_template_map(site: str) -> dict[int, int]:
    """task_id -> intent_template_id from the per-task VWA config JSONs."""
    site_dir = VWA_CONFIG / f"test_{site}"
    out: dict[int, int] = {}
    for f in site_dir.glob("[0-9]*.json"):
        try:
            cfg = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        tid = cfg.get("intent_template_id")
        if tid is not None:
            out[int(f.stem)] = int(tid)
    return out


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
        # Tasks with no template id (should not happen — coverage verified 234/234
        # + 210/210) get a sentinel singleton group = their own task_id + offset.
        groups = np.array(
            [tmap.get(int(t), 10_000_000 + int(t)) for t in tasks_sorted], dtype=int
        )
        n_groups = len(set(groups.tolist()))
        n_splits_eff = min(n_splits, n_groups, len(tasks_sorted))
        fold_map: dict[int, int] = {}
        if len(tasks_sorted) == 0:
            site_fold_map[site] = fold_map
            continue
        if n_splits_eff < 2:
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
    majority = Counter(labels[cell_indices].tolist()).most_common(1)[0]
    majority_acc = float((y == majority[0]).mean())

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
    macro_auroc = (
        float(np.mean(list(per_class_auroc.values()))) if per_class_auroc else float("nan")
    )
    total_pos = sum(per_class_n.values())
    weighted_auroc = (
        float(
            sum(per_class_auroc[c] * per_class_n[c] for c in per_class_auroc) / total_pos
        )
        if total_pos
        else float("nan")
    )

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
            "classes": global_classes,
            "proba": np.round(P, 6).tolist(),
        },
        "n_skipped_holdout": int(n_skipped_holdout),
        "folds_ok": folds_ok,
        "folds_failed": folds_failed,
        "accuracy": acc,
        "majority_class": majority[0],
        "majority_acc": majority_acc,
        "macro_ovr_auroc": macro_auroc,
        "weighted_ovr_auroc": weighted_auroc,
        "per_class_auroc": per_class_auroc,
        "per_class_n_pos": per_class_n,
        "n_classes_scored": len(per_class_auroc),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-features", default=str(DEFAULT_NPZ))
    ap.add_argument(
        "--out-json",
        default=str(DEFAULT_NPZ.parent / "covariate_baseline.json"),
    )
    ap.add_argument("--cells", nargs="+", default=None, help="subset of cell_ids")
    args = ap.parse_args()

    npz_path = Path(args.raw_features)
    raw = load_raw_features(npz_path)
    cells = sorted(set(str(c) for c in raw["cell_ids"].tolist()))
    if args.cells:
        cells = [c for c in cells if c in set(args.cells)]
    sites = sorted({_site_of_cell(c) for c in cells})
    template_maps = {s: load_template_map(s) for s in sites}
    for s in sites:
        n_tasks_with = len(template_maps[s])
        n_templates = len(set(template_maps[s].values()))
        print(f"[template map] {s}: {n_tasks_with} tasks, {n_templates} templates")

    fold_maps = {
        "standard": standard_fold_assignments(raw),
        "template_disjoint": template_disjoint_fold_assignments(raw, template_maps),
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
            leak_audit[regime][cell_id] = (n_leak / n_tot) if n_tot else float("nan")

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
                    print(
                        f"[{regime:17s}] {cell_id:15s} {feature_set:22s} "
                        f"n_oof={rec['n_oof']:3d} acc={rec['accuracy']:.3f} "
                        f"(maj={rec['majority_acc']:.3f}) "
                        f"macroAUROC={rec['macro_ovr_auroc']:.3f} "
                        f"wAUROC={rec['weighted_ovr_auroc']:.3f}"
                    )
                else:
                    print(
                        f"[{regime:17s}] {cell_id:15s} {feature_set:22s} "
                        f"status={rec['status']}"
                    )

    out = {
        "schema_version": SCHEMA_VERSION,
        "raw_features_npz": str(npz_path),
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
        "template_leak_audit_holdout_frac_with_template_in_train": leak_audit,
        "results": results,
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
