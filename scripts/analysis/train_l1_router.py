#!/usr/bin/env python3
"""Stage 3: Per-cell × per-fold LR trainer with inner-CV threshold tuning.

A2.5 Chunk B (B-994 + B-995 + B-997 + B-998 fix integrated; /stress 2026-05-18).

REFACTORED to consume Chunk A artifacts (raw_features_phase1a.npz + 5 vectorizer
pickles + 5 selected_idx JSON + 6 fold_assignment JSON). Replaces the prior in-sample
single-pickle-per-cell trainer with the (E''') design:
  - per-cell × per-fold LR training (30 pickles total = 6 cells × 5 folds)
  - StandardScaler INSIDE Pipeline (GPT-relay Point 4: scaler fits on train fold only)
  - class_weight=None (B-995 fix: drop "balanced" reweighting that produced 15× minority
    hallucination)
  - min_class_n_train rule (B-995 fix: drop classes with <N_MIN_CLASS train-fold samples)
  - Inner-CV τ tuning over candidate set [0.3, 0.4, 0.5, 0.6, 0.7] per (cell, fold)
    using inner StratifiedKFold (B-998 (b) GPT-relay Point 5 fix — τ tuned on train-fold
    inner-CV, never on outer holdout)
  - Per-(cell, fold) τ* dict stored in <cell_id>_lr_meta.json

Output (additional to Chunk A artifacts):
  <cell_id>_lr_fold{k}.pkl    × 30  # Pipeline pickles
  <cell_id>_lr_meta.json      × 6   # per-cell summary + thresholds_per_fold dict

Usage:
    # Train all 6 cells × 5 folds (requires Chunk A artifacts present)
    python3 scripts/analysis/train_l1_router.py --all

    # Train one cell × all folds
    python3 scripts/analysis/train_l1_router.py --baseline B0 --site classifieds
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "results/phantom_paper/l1_router"

# Cells in Phase 1a (B0/B1/B2 × cls/red).
CELLS = [
    ("B0", "classifieds"), ("B0", "reddit"),
    ("B1", "classifieds"), ("B1", "reddit"),
    ("B2", "classifieds"), ("B2", "reddit"),
]

# Hyperparameters locked at Chunk B (pre-fire per /stress A2.5).
N_FOLDS_OUTER = 5
N_FOLDS_INNER = 5
INNER_CV_SEED = 42
N_MIN_CLASS_TRAIN = 10  # B-995 fix: classes with fewer train samples → drop or merge
LR_MAX_ITER = 2000
LR_C = 1.0  # default L2 strength; tunable later via inner-CV if needed
TAU_CANDIDATES = [0.3, 0.4, 0.5, 0.6, 0.7]
SAFE_FALLBACK_MODE = "phantom_som"
SCHEMA_VERSION = "2026-05-18-a2.5-chunk-b-stage3"


def load_chunk_a_artifacts(out_dir: Path) -> dict[str, Any]:
    """Load Stage 1 raw features + Stage 2 vectorizers + selectors + fold assignments."""
    raw_npz = out_dir / "raw_features_phase1a.npz"
    raw_meta = out_dir / "raw_features_phase1a.json"
    stage2_summary = out_dir / "stage2_summary.json"
    if not raw_npz.exists():
        raise FileNotFoundError(f"Stage 1 artifact missing: {raw_npz}")
    if not stage2_summary.exists():
        raise FileNotFoundError(f"Stage 2 summary missing: {stage2_summary}")

    raw = np.load(raw_npz, allow_pickle=True)
    meta = json.loads(raw_meta.read_text())
    summary = json.loads(stage2_summary.read_text())

    if summary.get("status") == "no_data_yet":
        return {"status": "no_data_yet", "summary": summary}

    # Load 5 vectorizers + 5 selectors
    vectorizers = {}
    selectors = {}
    for k in range(N_FOLDS_OUTER):
        vec_path = out_dir / f"vectorizer_fold{k}.pkl"
        sel_path = out_dir / f"selected_idx_fold{k}.json"
        if not vec_path.exists() or not sel_path.exists():
            raise FileNotFoundError(f"Stage 2 fold {k} artifact missing")
        with vec_path.open("rb") as f:
            vectorizers[k] = pickle.load(f)
        selectors[k] = json.loads(sel_path.read_text())

    # Load per-cell fold assignments
    fold_assignments = {}
    for cell_id in meta.get("cells_in_pool", []):
        fa_path = out_dir / f"{cell_id}_fold_assignment.json"
        if not fa_path.exists():
            continue
        fa = json.loads(fa_path.read_text())
        fold_assignments[cell_id] = {
            int(tid): fk for tid, fk in fa["fold_assignment"].items()
        }

    return {
        "status": "ok",
        "X_numeric": raw["X_numeric"],
        "X_binary": raw["X_binary"],
        "labels": raw["labels"],
        "task_ids": raw["task_ids"],
        "cell_ids": raw["cell_ids"],
        "intents": list(raw["intents"]),
        "meta": meta,
        "vectorizers": vectorizers,
        "selectors": selectors,
        "fold_assignments": fold_assignments,
    }


def build_design_matrix_for_indices(
    indices: np.ndarray,
    intents: list[str],
    X_numeric: np.ndarray,
    X_binary: np.ndarray,
    vectorizer: Any,
    selected_idx_mask: np.ndarray,
) -> np.ndarray:
    """Transform raw features → 50-dim design matrix → apply selected_idx mask → 18-dim X.

    Used for both train fold (build → fit pipeline) and holdout fold (build → predict).
    """
    sub_intents = [intents[i] for i in indices]
    X_tfidf = vectorizer.transform(sub_intents).toarray()
    X_full = np.hstack([X_tfidf, X_numeric[indices], X_binary[indices]])
    # selected_idx_mask is a 50-dim bool over (tfidf + numeric + binary). Apply it.
    # Note: TF-IDF dim may be < 30 if vocab smaller. Pad mask if needed.
    n_tfidf_actual = X_tfidf.shape[1]
    expected_total = len(selected_idx_mask)
    if X_full.shape[1] != expected_total:
        # Mask was built with a different number of TF-IDF cols. Align:
        # the JSON's selected_mask reflects vocab size at Stage 2 fit. For runtime
        # consistency, this should always match. If mismatched, fail loud.
        raise ValueError(
            f"Design matrix has {X_full.shape[1]} cols, selected_mask expects "
            f"{expected_total}. Stage 2 vocab vs Stage 3 transform mismatch."
        )
    return X_full[:, selected_idx_mask]


def apply_min_class_filter(
    train_labels: np.ndarray,
    train_indices: np.ndarray,
    min_n: int = N_MIN_CLASS_TRAIN,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """B-995 fix: drop training samples whose class has < min_n samples in train fold.

    Returns (kept_labels, kept_indices, dropped_class_counts).
    """
    counts = Counter(train_labels.tolist())
    rare_classes = {c for c, n in counts.items() if n < min_n}
    if not rare_classes:
        return train_labels, train_indices, {}
    keep_mask = np.array([lbl not in rare_classes for lbl in train_labels])
    dropped_counts = {c: counts[c] for c in rare_classes}
    return train_labels[keep_mask], train_indices[keep_mask], dropped_counts


def tune_threshold_inner_cv(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cell_id: str,
    fold_k: int,
    candidates: list[float] = TAU_CANDIDATES,
    n_inner_folds: int = N_FOLDS_INNER,
    seed: int = INNER_CV_SEED,
) -> dict[str, Any]:
    """Inner-CV τ tuning on train fold ONLY (GPT-relay Point 5 fix).

    For each candidate τ, train inner pipeline on inner-train, evaluate the routing
    decision on inner-holdout, score = mode-match accuracy (decided mode == oracle label).

    DISCLOSURE (router /stress 2026-05-21; deferred true-fixes → next_steps.md):
      - F3/G2 (B-1814): the score is mode-match ACCURACY, a proxy for Pareto utility,
        NOT true SR. A miss that picks a more expensive but successful mode is scored the
        same as a cheap failing one. A true Expected-Pareto-Lift objective needs the
        per-task per-mode outcome matrix inside inner-CV (Stage 3 only has oracle labels)
        → deferred. 2nd-order: affects τ selection only, NOT the outer-holdout H10 eval.
      - C3 (B-1816): X_train here is already Stage-2 (outer-pool) MI-selected, so the
        inner-holdout influenced feature selection → τ is mildly optimistic. 2nd-order
        (outer eval leak-free, §254) + 3rd-order magnitude (~30 inner-holdout vs ~1124
        pooled-MI samples). Nested per-inner-fold MI conflicts with the user-confirmed
        E'' pooled-cross-cell selector → deferred.

    Returns dict with chosen tau, per-tau scores, n_inner_folds_used.
    """
    label_counts = Counter(y_train.tolist())
    n_classes = len(label_counts)
    rare_for_inner = {c for c, n in label_counts.items() if n < n_inner_folds}

    if n_classes < 2:
        # Single-class train → τ irrelevant; predict_proba is degenerate
        return {
            "chosen_tau": candidates[len(candidates) // 2],
            "per_tau_score": {str(t): float("nan") for t in candidates},
            "n_inner_folds_used": 0,
            "reason": "single_class_train_set",
        }

    # For stratification, merge rare classes for split only
    if rare_for_inner:
        strat = np.array(["__rare__" if c in rare_for_inner else c for c in y_train])
    else:
        strat = y_train

    inner_seed = seed + fold_k * 100  # per-(cell, fold) deterministic
    try:
        inner_skf = StratifiedKFold(
            n_splits=n_inner_folds, shuffle=True, random_state=inner_seed
        )
        splits = list(inner_skf.split(X_train, strat))
    except ValueError:
        # Fallback: not enough samples per class for inner-CV
        return {
            "chosen_tau": candidates[len(candidates) // 2],
            "per_tau_score": {str(t): float("nan") for t in candidates},
            "n_inner_folds_used": 0,
            "reason": "inner_cv_split_failed",
        }

    per_tau_scores: dict[float, list[float]] = {t: [] for t in candidates}
    for inner_train_idx, inner_holdout_idx in splits:
        inner_X_train = X_train[inner_train_idx]
        inner_y_train = y_train[inner_train_idx]
        inner_X_holdout = X_train[inner_holdout_idx]
        inner_y_holdout = y_train[inner_holdout_idx]

        # B-995 filter inner-train too (only apply to inner-train, evaluate on full holdout)
        ity, iti, _ = apply_min_class_filter(
            inner_y_train, inner_train_idx, min_n=2  # tighter for inner-CV
        )
        if len(set(ity)) < 2:
            continue
        # Refit inner_X_train using the filtered indices (relative to inner_train_idx)
        keep_relative = np.isin(inner_train_idx, iti)
        inner_X_train_filtered = inner_X_train[keep_relative]
        inner_y_train_filtered = ity

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
        try:
            pipeline.fit(inner_X_train_filtered, inner_y_train_filtered)
        except ValueError:
            continue

        probs = pipeline.predict_proba(inner_X_holdout)
        max_probs = probs.max(axis=1)
        pred_modes = pipeline.classes_[probs.argmax(axis=1)]

        for tau in candidates:
            # Cost-weighted decision rule: route to pred_mode if max_prob > τ else fallback
            decided_modes = np.where(max_probs > tau, pred_modes, SAFE_FALLBACK_MODE)
            # F3/G2 (B-1814): mode-match ACCURACY (decided mode == oracle label) — a
            # proxy for Pareto utility, NOT true SR (a miss picking a more expensive but
            # successful mode scores the same as a cheap failing one). Deferred
            # Expected-Pareto-Lift objective → next_steps. 2nd-order: τ-selection only,
            # outer-holdout H10 eval unaffected.
            accuracy = float((decided_modes == inner_y_holdout).mean())
            per_tau_scores[tau].append(accuracy)

    # Aggregate per-tau scores
    per_tau_mean: dict[float, float] = {}
    for tau, scores in per_tau_scores.items():
        per_tau_mean[tau] = float(np.mean(scores)) if scores else float("nan")

    # Pick τ* maximizing mean SR; tie-break = higher τ (more conservative routing)
    valid = {t: s for t, s in per_tau_mean.items() if not np.isnan(s)}
    if not valid:
        chosen = candidates[len(candidates) // 2]
        reason = "all_inner_folds_failed"
    else:
        max_score = max(valid.values())
        # Highest τ among those tied at max (more conservative = prefer fallback)
        best_taus = [t for t, s in valid.items() if abs(s - max_score) < 1e-9]
        chosen = max(best_taus)
        reason = "ok"

    return {
        "chosen_tau": float(chosen),
        "per_tau_score": {str(t): per_tau_mean[t] for t in candidates},
        "n_inner_folds_used": len([s for tau in candidates for s in per_tau_scores[tau]])
        // max(1, len(candidates)),
        "reason": reason,
    }


def label_entropy_bits(labels) -> float:
    """Shannon entropy (base-2, bits) of a label distribution.

    H = −Σ_m p(m)·log_2 p(m) over the distinct mode labels present. Used by the H10
    DEFER gate (prereg §H10 L238-240): if a cell's train-fold best-mode label
    distribution concentrates on ≤ 2 modes (H < 1.0 bit), the learned router has
    insufficient label diversity to learn a non-trivial routing policy → H10 is
    downgraded to §5 descriptive. Computed on RAW train-fold labels (before the B-995
    min-class filter) so the diagnostic reflects true label concentration, not the
    filter-inflated distribution.
    """
    arr = np.asarray(labels)
    n = len(arr)
    if n == 0:
        return float("nan")
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / n
    return float(-np.sum(probs * np.log2(probs)))


def train_one_cell_one_fold(
    cell_id: str,
    fold_k: int,
    artifacts: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Train a single Pipeline for (cell_id, fold_k) and dump pickle + return summary."""
    X_numeric = artifacts["X_numeric"]
    X_binary = artifacts["X_binary"]
    labels = artifacts["labels"]
    task_ids = artifacts["task_ids"]
    cell_ids = artifacts["cell_ids"]
    intents = artifacts["intents"]
    vectorizer = artifacts["vectorizers"][fold_k]
    selector_payload = artifacts["selectors"][fold_k]
    selected_mask = np.array(selector_payload["selected_mask"], dtype=bool)
    fold_assignments = artifacts["fold_assignments"]

    fa = fold_assignments.get(cell_id, {})
    if not fa:
        return {"status": "no_fold_assignment", "cell_id": cell_id, "fold_k": fold_k}

    # Cell-scoped indices in pooled arrays:
    cell_mask = cell_ids == cell_id
    cell_indices = np.where(cell_mask)[0]
    cell_task_ids = task_ids[cell_indices]
    cell_labels = labels[cell_indices]

    # Partition cell_indices into train (fold != fold_k) vs holdout (fold == fold_k)
    train_local_idx, holdout_local_idx = [], []
    for local_i, tid in enumerate(cell_task_ids):
        if fa.get(int(tid)) == fold_k:
            holdout_local_idx.append(local_i)
        else:
            train_local_idx.append(local_i)
    train_local_idx = np.array(train_local_idx, dtype=int)
    holdout_local_idx = np.array(holdout_local_idx, dtype=int)

    train_global_idx = cell_indices[train_local_idx]
    holdout_global_idx = cell_indices[holdout_local_idx]
    y_train = cell_labels[train_local_idx]
    y_holdout = cell_labels[holdout_local_idx]

    # B-995 min-class filter on train fold
    y_train_filtered, train_kept_global_idx, dropped_classes = apply_min_class_filter(
        y_train, train_global_idx, min_n=N_MIN_CLASS_TRAIN
    )

    if len(y_train_filtered) < 2 or len(set(y_train_filtered)) < 2:
        return {
            "status": "insufficient_train_data",
            "cell_id": cell_id,
            "fold_k": fold_k,
            "n_train_total": len(y_train),
            "n_train_kept": len(y_train_filtered),
            "n_classes_remaining": len(set(y_train_filtered)),
            "dropped_classes": dropped_classes,
        }

    # Build design matrices
    X_train_full = build_design_matrix_for_indices(
        train_kept_global_idx, intents, X_numeric, X_binary, vectorizer, selected_mask
    )
    X_holdout_full = build_design_matrix_for_indices(
        holdout_global_idx, intents, X_numeric, X_binary, vectorizer, selected_mask
    )

    # Inner-CV τ tuning on train fold ONLY
    tau_result = tune_threshold_inner_cv(
        X_train_full, y_train_filtered, cell_id, fold_k
    )
    tau_star = tau_result["chosen_tau"]

    # Refit pipeline on full filtered train
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
    pipeline.fit(X_train_full, y_train_filtered)

    # In-fold holdout evaluation — P1-6 (codex B-F7, AMENDMENT_04): DIAGNOSTIC ONLY.
    # This is CV mode-match accuracy (predicted mode == oracle-best label), NOT the H10
    # realized SR/cost gate. True H10 uses Pass-2 condition_summary realized (Cost, SR)
    # in aggregate_h10_pareto.analyze_cell — do NOT cite holdout_sr as H10 evidence.
    holdout_probs = pipeline.predict_proba(X_holdout_full)
    holdout_max_probs = holdout_probs.max(axis=1)
    holdout_argmax_modes = pipeline.classes_[holdout_probs.argmax(axis=1)]
    holdout_decided_modes = np.where(
        holdout_max_probs > tau_star, holdout_argmax_modes, SAFE_FALLBACK_MODE
    )
    # Holdout SR = fraction where decided mode == oracle-best label
    holdout_sr = float((holdout_decided_modes == y_holdout).mean()) if len(y_holdout) else float("nan")
    holdout_fallback_rate = float((holdout_max_probs <= tau_star).mean()) if len(holdout_max_probs) else float("nan")

    # Dump pipeline pickle
    pickle_path = out_dir / f"{cell_id}_lr_fold{fold_k}.pkl"
    with pickle_path.open("wb") as f:
        pickle.dump(pipeline, f)

    return {
        "status": "ok",
        "cell_id": cell_id,
        "fold_k": fold_k,
        "n_train_total": int(len(y_train)),
        "n_train_kept": int(len(y_train_filtered)),
        "dropped_classes": dropped_classes,
        "train_label_distribution": dict(Counter(y_train_filtered.tolist())),
        # P0-2 (AMENDMENT_04 H10 entropy DEFER gate, prereg §H10 L238-240): per-fold
        # train-fold best-mode label entropy in bits, on RAW y_train (pre B-995
        # min-class filter) so the diagnostic reflects true label concentration, not
        # the filter-inflated distribution. H < 1.0 bit → ≤ 2 modes → H10 DEFER.
        "train_label_counts_raw": dict(Counter(y_train.tolist())),
        "train_label_entropy_bits": label_entropy_bits(y_train),
        "n_holdout": int(len(y_holdout)),
        "holdout_label_distribution": dict(Counter(y_holdout.tolist())),
        "chosen_tau": tau_star,
        "tau_tuning_per_score": tau_result["per_tau_score"],
        "tau_tuning_reason": tau_result["reason"],
        "tau_tuning_n_inner_folds": tau_result["n_inner_folds_used"],
        "holdout_sr": holdout_sr,
        # P1-6 (codex B-F7, AMENDMENT_04): clearer alias — CV mode-match accuracy
        # (predicted == oracle-best label), NOT H10 realized SR/cost. Diagnostic only.
        "cv_mode_match_acc": holdout_sr,
        "holdout_fallback_rate": holdout_fallback_rate,
        "holdout_modes_predicted": dict(Counter(holdout_decided_modes.tolist())),
        "pickle_path": pickle_path.name,
    }


def train_one_cell(
    cell_id: str, artifacts: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    """Train all 5 folds for one cell + dump cell meta JSON."""
    print(f"\n=== Training cell {cell_id} ===")
    per_fold = {}
    for fold_k in range(N_FOLDS_OUTER):
        rec = train_one_cell_one_fold(cell_id, fold_k, artifacts, out_dir)
        per_fold[fold_k] = rec
        print(
            f"  fold {fold_k}: status={rec['status']}, "
            f"n_train_kept={rec.get('n_train_kept', 'N/A')}, "
            f"τ={rec.get('chosen_tau', 'N/A')}, "
            f"holdout_sr={rec.get('holdout_sr', 'N/A')}"
        )

    # C5 (B-1812): cell completeness — every fold must be "ok" (pickle written +
    # threshold tuned), else the runtime hard-fails at Pass-2 on the missing fold pickle
    # (B-1640). Surface incomplete cells loudly at train time instead of discovering
    # them at fire time.
    folds_ok = [fk for fk, rec in per_fold.items() if rec["status"] == "ok"]
    incomplete_folds = {
        fk: rec["status"] for fk, rec in per_fold.items() if rec["status"] != "ok"
    }
    cell_complete = len(folds_ok) == N_FOLDS_OUTER

    # Aggregate cell meta
    thresholds_per_fold = {
        fk: rec["chosen_tau"] for fk, rec in per_fold.items() if rec["status"] == "ok"
    }
    holdout_srs = [
        rec["holdout_sr"] for rec in per_fold.values()
        if rec["status"] == "ok" and not np.isnan(rec.get("holdout_sr", float("nan")))
    ]
    # P0-2 (AMENDMENT_04 H10 entropy DEFER gate): per-fold train-label entropy +
    # per-cell min over folds. aggregate_h10_pareto reads cell_entropy_min_bits and
    # DEFERs H10 if any required cell's min < 1.0 bit (prereg §H10 L238-240).
    entropy_per_fold = {
        str(fk): rec["train_label_entropy_bits"]
        for fk, rec in per_fold.items()
        if rec["status"] == "ok" and "train_label_entropy_bits" in rec
    }
    cell_entropy_min_bits = (
        min(entropy_per_fold.values()) if entropy_per_fold else float("nan")
    )
    cell_meta = {
        "schema_version": SCHEMA_VERSION,
        "cell_id": cell_id,
        "n_folds": N_FOLDS_OUTER,
        # C5 (B-1812): completeness contract — runtime needs all 5 fold pickles.
        "cell_complete": cell_complete,
        "folds_ok": folds_ok,
        "incomplete_folds": incomplete_folds,
        "thresholds_per_fold": thresholds_per_fold,
        "min_class_n_train": N_MIN_CLASS_TRAIN,
        "tau_candidates": TAU_CANDIDATES,
        "lr_c": LR_C,
        "lr_max_iter": LR_MAX_ITER,
        "class_weight": None,
        "safe_fallback_mode": SAFE_FALLBACK_MODE,
        "per_fold_records": {str(fk): rec for fk, rec in per_fold.items()},
        "holdout_sr_per_fold_mean": float(np.mean(holdout_srs)) if holdout_srs else float("nan"),
        "holdout_sr_per_fold_values": holdout_srs,
        # P0-2 H10 entropy DEFER gate (prereg §H10 L238-240):
        "train_label_entropy_bits_per_fold": entropy_per_fold,
        "cell_entropy_min_bits": cell_entropy_min_bits,
        # δ-cluster disclosure (router /stress 2026-05-21; deferred true-fixes → next_steps):
        "tau_tuning_disclosure": {
            "objective": "mode_match_accuracy",
            "f3_g2_b1814": (
                "τ chosen by mode-match accuracy (proxy for Pareto utility, not true SR). "
                "Expected-Pareto-Lift objective needs the outcome matrix in inner-CV "
                "(deferred). 2nd-order: τ-selection only, outer H10 eval clean."
            ),
            "c3_b1816": (
                "inner-CV reuses the Stage-2 outer-pool MI selector → inner-holdout "
                "influenced selection → τ mildly optimistic. 2nd-order (outer eval "
                "leak-free §254) + 3rd-order magnitude. Nested MI deferred (conflicts "
                "with user-confirmed E'' pooled selector)."
            ),
        },
        "note_design": (
            "Q1=C + (E''') design — within-cell 5-fold CV deployment with inner-CV τ "
            "tuning (b). Pipeline has internal StandardScaler (GPT-relay Point 4). "
            "class_weight=None (B-995 fix vs 'balanced' minority hallucination). "
            "Cell-constant features (site, capability_tier) implicit via runtime "
            "pickle selection per (baseline, site)."
        ),
    }

    meta_path = out_dir / f"{cell_id}_lr_meta.json"
    meta_path.write_text(json.dumps(cell_meta, indent=2, default=str))
    print(f"  Wrote: {meta_path.name} (thresholds_per_fold: {thresholds_per_fold})")
    return cell_meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", help="B0 | B1 | B2 (omit if --all)")
    ap.add_argument("--site", help="classifieds | reddit (omit if --all)")
    ap.add_argument("--all", action="store_true", help="train all 6 Phase 1a cells")
    ap.add_argument(
        "--out-dir",
        default=str(OUT_DIR),
        help="Directory containing Chunk A artifacts + receiving Chunk B pickles",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    artifacts = load_chunk_a_artifacts(out_dir)
    if artifacts.get("status") == "no_data_yet":
        print(
            "⚠️  Chunk A artifacts indicate no_data_yet — Stage 3 cannot train without "
            "Pass-1 outcomes. Run extract_50_features.py + train_l1_router_with_mi.py "
            "after Phase 1a Pass-1 lands."
        )
        return 0

    cells_in_pool = artifacts["meta"].get("cells_in_pool", [])
    if args.all:
        targets = cells_in_pool
    elif args.baseline and args.site:
        cell_id = f"{args.baseline}_{args.site}"
        targets = [cell_id] if cell_id in cells_in_pool else []
    else:
        ap.error("Either --all or (--baseline + --site) required")

    if not targets:
        print(f"⚠️  No matching cells in pool. Available: {cells_in_pool}")
        return 1

    print(f"Training {len(targets)} cell(s): {targets}")
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_cells_trained": 0,
        "n_cells_incomplete": 0,
        "n_cells_failed": 0,
        "per_cell": {},
    }
    for cell_id in targets:
        try:
            cell_meta = train_one_cell(cell_id, artifacts, out_dir)
            summary["per_cell"][cell_id] = cell_meta
            # C5 (B-1812): only count a cell as trained if ALL folds wrote a pickle. An
            # incomplete cell would hard-fail the runtime at Pass-2 (B-1640) — flag it
            # loudly here so the orchestrator never fires an undeployable cell.
            if cell_meta.get("cell_complete"):
                summary["n_cells_trained"] += 1
            else:
                summary["n_cells_incomplete"] += 1
                print(
                    f"  ⚠️  {cell_id} INCOMPLETE: folds {cell_meta.get('incomplete_folds')} "
                    f"have no pickle — runtime would hard-fail at Pass-2 (B-1640). "
                    f"NOT deployable."
                )
        except Exception as exc:
            print(f"  ✗ {cell_id} FAILED: {exc}")
            summary["per_cell"][cell_id] = {"error": str(exc)}
            summary["n_cells_failed"] += 1

    summary_path = out_dir / "stage3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote: {summary_path}")

    # P0-2 (AMENDMENT_04 H10 entropy DEFER gate, prereg §H10 L238-240): emit a dedicated
    # entropy-gate artifact that aggregate_h10_pareto.run_h10_verdict reads to DEFER H10
    # when any required cell's train-fold label entropy concentrates on ≤ 2 modes
    # (min-over-folds < 1.0 bit). Full per-cell + per-fold breakdown (NOT a single bool)
    # so a reviewer can point at exactly which cell/fold is low-entropy.
    H10_ENTROPY_DEFER_THRESHOLD_BITS = 1.0
    entropy_by_cell: dict[str, float] = {}
    entropy_by_fold: dict[str, dict] = {}
    for cid, cmeta in summary["per_cell"].items():
        if not isinstance(cmeta, dict) or "cell_entropy_min_bits" not in cmeta:
            continue
        entropy_by_cell[cid] = cmeta["cell_entropy_min_bits"]
        entropy_by_fold[cid] = cmeta.get("train_label_entropy_bits_per_fold", {})
    finite_mins = [v for v in entropy_by_cell.values()
                   if isinstance(v, (int, float)) and not np.isnan(v)]
    global_entropy_min_bits = min(finite_mins) if finite_mins else float("nan")
    gate_passed = bool(finite_mins) and global_entropy_min_bits >= H10_ENTROPY_DEFER_THRESHOLD_BITS
    low_entropy_cells = [
        c for c, v in entropy_by_cell.items()
        if isinstance(v, (int, float)) and not np.isnan(v)
        and v < H10_ENTROPY_DEFER_THRESHOLD_BITS
    ]
    entropy_gate = {
        "schema_version": SCHEMA_VERSION,
        "defer_threshold_bits": H10_ENTROPY_DEFER_THRESHOLD_BITS,
        "h10_entropy_gate_passed": gate_passed,
        "h10_status": "ok" if gate_passed else "deferred_entropy",
        "global_entropy_min_bits": global_entropy_min_bits,
        "h10_entropy_by_cell": entropy_by_cell,
        "h10_entropy_by_fold": entropy_by_fold,
        "low_entropy_cells": low_entropy_cells,
        "note": (
            "prereg §H10 L238-240 DEFER condition: per-cell train-fold best-mode label "
            "entropy H = −Σ p·log_2(p); if any required cell's min-over-folds < 1.0 bit "
            "(labels concentrate on ≤ 2 modes) H10 is downgraded to §5 descriptive "
            "(operational deployment gate suppressed). Computed on RAW train labels "
            "(pre B-995 min-class filter). Consumed by aggregate_h10_pareto.run_h10_verdict."
        ),
    }
    entropy_gate_path = out_dir / "h10_entropy_gate.json"
    entropy_gate_path.write_text(json.dumps(entropy_gate, indent=2, default=str))
    print(f"Wrote: {entropy_gate_path} (h10_status={entropy_gate['h10_status']}, "
          f"global_min_bits={global_entropy_min_bits})")
    print(
        f"\n=== Summary: {summary['n_cells_trained']}/{len(targets)} cells fully trained "
        f"({summary['n_cells_incomplete']} incomplete, {summary['n_cells_failed']} failed) ==="
    )
    # C5 (B-1812): incomplete cells are NOT paper-grade deployable (runtime hard-fail),
    # so a non-zero exit signals the orchestrator not to fire Pass-2 on them.
    if summary["n_cells_failed"] > 0 or summary["n_cells_incomplete"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
