"""Learned router runtime predictor (paper-1, Chunk C / A2.5 2026-05-18).

Loads per-(cell, fold) LR pickles + Stage 2 vectorizer + Stage 2 selected_idx +
per-cell fold_assignment, predicts observation_mode for each task at Pass-2 fire
time using the fold that held that task out (Q1=C + (E''') within-cell 5-fold CV
deployment).

Feature pipeline (matches Chunk A/B Stage 1+2+3):
  1. Stage 1: extract 20 raw deterministic features
     (5 numeric: dom_complexity, text_length, tokens_input_text, intent_token_count,
      reasoning_difficulty / 15 binary: has_ref_image + 14 intent regex banks).
     site + capability_tier EXCLUDED (cell-constant; implicit via pickle selection).
  2. Stage 2: apply fold-k TfidfVectorizer (30 cols) + concat + selected_idx mask
     → 18-dim feature vector.
  3. Stage 3: pipeline_{C,k}.predict_proba → argmax mode if max_prob > τ_{C,k}
     else SAFE_FALLBACK_MODE (cost-weighted decision rule, B-998).

Train artifacts (loaded at predictor init per cell):
  <cell_id>_lr_fold{k}.pkl     × 5   # Pipeline (StandardScaler + LR)
  vectorizer_fold{k}.pkl       × 5   # TfidfVectorizer (fold-local vocab)
  selected_idx_fold{k}.json    × 5   # 18-bool mask + feature names
  <cell_id>_fold_assignment.json     # task_id → fold_index
  <cell_id>_lr_meta.json             # thresholds_per_fold dict + class decisions

Used by: `p79/experiment/runner/main.py` when condition.observation_mode == "learned"
"""
from __future__ import annotations

import json
import logging
import pickle
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants matching Chunk A/B Stage 1+2 ─────────────────────────────────────
N_FOLDS = 5
SAFE_FALLBACK_MODE = "phantom_som"

# Back-compat module-level regex aliases (pre-Chunk-C single-pickle callers).
# New code should use INTENT_REGEX dict (matches scripts/analysis/extract_50_features.py).
COLOR_RE = re.compile(
    r"\b(color|red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey)\b",
    re.IGNORECASE,
)
SEARCH_RE = re.compile(r"\b(find|search|locate|how many|how much)\b", re.IGNORECASE)
COMPARE_RE = re.compile(
    r"\b(cheapest|most expensive|highest|lowest|best|worst|biggest|smallest)\b",
    re.IGNORECASE,
)
NAV_RE = re.compile(r"\b(go to|navigate|open|visit)\b", re.IGNORECASE)

# Intent regex banks — must match scripts/analysis/extract_50_features.py:INTENT_REGEX
INTENT_REGEX = {
    "intent_color": re.compile(
        r"\b(color|red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey)\b",
        re.IGNORECASE,
    ),
    "intent_search": re.compile(r"\b(find|search|locate|how many|how much)\b", re.IGNORECASE),
    "intent_compare": re.compile(
        r"\b(cheapest|most expensive|highest|lowest|best|worst|biggest|smallest)\b",
        re.IGNORECASE,
    ),
    "intent_nav": re.compile(r"\b(go to|navigate|open|visit)\b", re.IGNORECASE),
    "intent_filter": re.compile(r"\b(filter|narrow|restrict|limit to|only)\b", re.IGNORECASE),
    "intent_sort": re.compile(r"\b(sort|rank|order by|by date|by price|newest|oldest)\b", re.IGNORECASE),
    "intent_aggregate": re.compile(r"\b(total|sum|average|count of|number of)\b", re.IGNORECASE),
    "intent_compose": re.compile(r"\b(compose|write|post|submit|reply|comment)\b", re.IGNORECASE),
    "intent_form_fill": re.compile(r"\b(fill|enter|type|input)\b", re.IGNORECASE),
    "intent_account_action": re.compile(
        r"\b(login|logout|account|profile|sign in|sign out|subscribe|unsubscribe)\b",
        re.IGNORECASE,
    ),
    "intent_visual_attribute": re.compile(
        r"\b(size|shape|appear|look|tall|wide|small|large|height|width)\b", re.IGNORECASE
    ),
    "intent_question": re.compile(r"\b(what|where|when|why|how)\b|\?", re.IGNORECASE),
    "intent_action_word": re.compile(r"\b(click|select|choose|press|tap)\b", re.IGNORECASE),
    "intent_temporal": re.compile(
        r"\b(today|yesterday|recent|latest|first|newest|oldest|2024|2025|2026)\b",
        re.IGNORECASE,
    ),
}


# ── Artifact loaders (cached per cell on Runner attribute) ─────────────────────


def load_fold_assignment(artifacts_dir: str | Path, cell_id: str) -> dict[int, int]:
    """Load <cell_id>_fold_assignment.json — task_id → fold_index dict."""
    path = Path(artifacts_dir) / f"{cell_id}_fold_assignment.json"
    if not path.exists():
        logger.warning("Fold assignment not found: %s", path)
        return {}
    try:
        data = json.loads(path.read_text())
        fa = data.get("fold_assignment", {})
        return {int(tid): int(fk) for tid, fk in fa.items()}
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load fold_assignment %s: %s", path, e)
        return {}


def load_vectorizer_fold(artifacts_dir: str | Path, fold_k: int) -> Optional[Any]:
    """Load vectorizer_fold{k}.pkl (TfidfVectorizer fitted in Stage 2)."""
    path = Path(artifacts_dir) / f"vectorizer_fold{fold_k}.pkl"
    if not path.exists():
        logger.warning("Vectorizer artifact not found: %s", path)
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError) as e:
        logger.error("Failed to load vectorizer %s: %s", path, e)
        return None


def load_selected_idx_fold(
    artifacts_dir: str | Path, fold_k: int
) -> tuple[Optional[np.ndarray], Optional[list[str]]]:
    """Load selected_idx_fold{k}.json — returns (mask, feature_names_all) or (None, None)."""
    path = Path(artifacts_dir) / f"selected_idx_fold{fold_k}.json"
    if not path.exists():
        logger.warning("Selected idx not found: %s", path)
        return None, None
    try:
        data = json.loads(path.read_text())
        mask = np.array(data["selected_mask"], dtype=bool)
        feature_names = data.get("feature_names_all", [])
        return mask, feature_names
    except (OSError, json.JSONDecodeError, KeyError) as e:
        logger.error("Failed to load selected_idx %s: %s", path, e)
        return None, None


def load_lr_pipeline_fold(
    artifacts_dir: str | Path, cell_id: str, fold_k: int
) -> Optional[Any]:
    """Load <cell_id>_lr_fold{k}.pkl — Pipeline(scaler + LR)."""
    path = Path(artifacts_dir) / f"{cell_id}_lr_fold{fold_k}.pkl"
    if not path.exists():
        logger.warning("LR pipeline not found: %s", path)
        return None
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except (OSError, pickle.UnpicklingError) as e:
        logger.error("Failed to load LR pipeline %s: %s", path, e)
        return None


def load_cell_meta(artifacts_dir: str | Path, cell_id: str) -> dict[str, Any]:
    """Load <cell_id>_lr_meta.json — per-cell summary including thresholds_per_fold."""
    path = Path(artifacts_dir) / f"{cell_id}_lr_meta.json"
    if not path.exists():
        logger.warning("Cell meta not found: %s", path)
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.error("Failed to load cell meta %s: %s", path, e)
        return {}


# ── Feature extraction (matches Chunk A Stage 1 deterministic raw 20-dim) ──────


def load_task_image_field(task_config_file: str | Path) -> bool:
    """Extract has_reference_image from VWA task config JSON."""
    try:
        with Path(task_config_file).open() as f:
            cfg = json.load(f)
        image = cfg.get("image")
        return image not in (None, "None", "", [])
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to read task config %s: %s", task_config_file, e)
        return False


def extract_raw_features(
    intent: str,
    has_reference_image: bool,
    dom_complexity: int,
    text_length: int,
    tokens_input_text: int,
    reasoning_difficulty: int,
) -> dict[str, Any]:
    """Extract 20 raw deterministic features matching Chunk A Stage 1 schema.

    Returns dict with `intent_text`, `numeric` (5,), `binary` (15,).
    """
    intent = intent or ""
    intent_token_count = len(intent.split())
    numeric = np.array(
        [
            int(dom_complexity),
            int(text_length),
            int(tokens_input_text),
            int(intent_token_count),
            int(reasoning_difficulty),
        ],
        dtype=float,
    )
    intent_bins = {
        name: int(bool(pattern.search(intent)))
        for name, pattern in INTENT_REGEX.items()
    }
    binary = np.array(
        [int(has_reference_image)]
        + [intent_bins[name] for name in sorted(INTENT_REGEX.keys())],
        dtype=int,
    )
    return {
        "intent_text": intent,
        "numeric": numeric,
        "binary": binary,
    }


def build_runtime_feature_vector(
    raw_features: dict[str, Any],
    vectorizer: Any,
    selected_mask: np.ndarray,
) -> np.ndarray:
    """Apply fold-k vectorizer + concat + selected_idx mask → (18,) feature vector.

    Mirrors `train_l1_router_with_mi.py:build_design_matrix` runtime equivalent.
    """
    intent_text = raw_features["intent_text"]
    X_tfidf = vectorizer.transform([intent_text]).toarray().ravel()  # (30,)
    X_full = np.concatenate(
        [X_tfidf, raw_features["numeric"], raw_features["binary"]]
    )
    # Defensive length check — vectorizer may have <30 cols if vocab smaller
    if X_full.shape[0] != selected_mask.shape[0]:
        raise ValueError(
            f"Runtime feature vector dim {X_full.shape[0]} mismatch selected_mask "
            f"dim {selected_mask.shape[0]}. Stage 2/Stage 3 vocab consistency violated."
        )
    return X_full[selected_mask]


# ── Top-level predictor entry (used by runner) ─────────────────────────────────


class LearnedRouterArtifactError(RuntimeError):
    """Infrastructure-level error in learned router substrate (missing artifact,
    corrupt pickle, dim mismatch, missing fold_assignment entry). Must NOT be
    silenced into SAFE_FALLBACK_MODE — that contaminates the H10 PRIMARY estimand
    (router operating point ≡ phantom_som operating point = trivial null circular).

    B-1600 (/stress A2.10 P0-3-B + user-mandate hard-fail 2026-05-18):
    user-directed "silent fallback is H10 最大风险之一" — only task-level
    signal-strength fallback (max_prob ≤ τ) stays silent + counted; all
    infrastructure-level paths raise this RuntimeError so the cell loop dies
    loudly + reviewer / user sees the failure immediately rather than days
    later when paper §6 aggregator notices all predictions = phantom_som.
    """


def predict_mode_fold_aware(
    cell_id: str,
    task_id: int,
    artifacts_dir: str | Path,
    cache: dict[str, Any],
    raw_features: dict[str, Any],
    fallback_mode: str = SAFE_FALLBACK_MODE,
) -> tuple[str, dict[str, Any]]:
    """Predict mode using (cell, fold-k) LR where fold_k = fold_assignment[task_id].

    Cache structure (Runner attribute, populated lazily):
        cache[cell_id] = {
            "fold_assignment": {task_id: fold_k, ...},
            "cell_meta": {thresholds_per_fold: {...}, ...},
            "vectorizers": {k: TfidfVectorizer, ...},
            "selected_masks": {k: np.ndarray, ...},
            "pipelines": {k: Pipeline, ...},
        }

    Fold-local feature machinery is **shared across cells** (per user-confirmed
    final E'' design /stress A2.10 P0-3-B 2026-05-18): `vectorizer_fold{k}.pkl`
    and `selected_idx_fold{k}.json` are NOT prefixed by cell_id — one vocab +
    one selected-idx mask per fold, fit on the pooled train-fold tasks across
    all 6 cells. Only `{cell_id}_lr_fold{k}.pkl` is per-cell (one LR head per
    (cell, fold) on the shared 18-dim representation).

    Returns (predicted_mode, diagnostic_dict) where diagnostic_dict contains
    fold_k_used, tau_used, max_prob, fallback_fired flags for paper-grade audit.

    Raises:
        LearnedRouterArtifactError on infrastructure-level failures
        (missing fold_assignment entry / missing artifact files / corrupt
        pickle / feature vector dim mismatch / pipeline.predict_proba exception).
        Task-level signal-strength fallback (max_prob ≤ τ) does NOT raise;
        it returns `fallback_mode` with `diag["fallback_fired"] = True` so
        the runner can count it as a legitimate signal-strength fallback.
    """
    diag = {
        "cell_id": cell_id,
        "task_id": task_id,
        "fold_k_used": None,
        "tau_used": None,
        "max_prob": None,
        "fallback_fired": False,
        "fallback_reason": None,
    }

    # Lazy-load cell-scoped artifacts if not cached
    if cell_id not in cache:
        cache[cell_id] = {
            "fold_assignment": load_fold_assignment(artifacts_dir, cell_id),
            "cell_meta": load_cell_meta(artifacts_dir, cell_id),
            "vectorizers": {},
            "selected_masks": {},
            "pipelines": {},
        }
    cell_cache = cache[cell_id]

    # Resolve fold for this task
    # B-1600 hard-fail: task_id missing from fold_assignment is an infrastructure
    # error (training pipeline didn't include this task), not a signal-strength
    # fallback. Raising kills the cell run so user diagnoses immediately.
    fold_k = cell_cache["fold_assignment"].get(int(task_id))
    if fold_k is None:
        msg = (
            f"[learned_router] task_id={task_id} not in fold_assignment for "
            f"cell={cell_id}; fold_assignment must include every Pass-2 task. "
            f"Hard-fail per /stress A2.10 P0-3-B user-mandate (NO silent "
            f"phantom_som fallback for infrastructure errors). "
            f"Fix: re-run scripts/analysis/extract_50_features.py + "
            f"train_l1_router_with_mi.py to regenerate fold_assignment.json."
        )
        logger.error(msg)
        raise LearnedRouterArtifactError(msg)
    diag["fold_k_used"] = fold_k

    # Lazy-load fold artifacts
    if fold_k not in cell_cache["vectorizers"]:
        cell_cache["vectorizers"][fold_k] = load_vectorizer_fold(artifacts_dir, fold_k)
    if fold_k not in cell_cache["selected_masks"]:
        mask, _ = load_selected_idx_fold(artifacts_dir, fold_k)
        cell_cache["selected_masks"][fold_k] = mask
    if fold_k not in cell_cache["pipelines"]:
        cell_cache["pipelines"][fold_k] = load_lr_pipeline_fold(
            artifacts_dir, cell_id, fold_k
        )

    vectorizer = cell_cache["vectorizers"][fold_k]
    selected_mask = cell_cache["selected_masks"][fold_k]
    pipeline = cell_cache["pipelines"][fold_k]
    # B-1600 hard-fail: missing fold artifact = infrastructure error.
    if vectorizer is None or selected_mask is None or pipeline is None:
        msg = (
            f"[learned_router] cell={cell_id} fold_k={fold_k} missing artifact "
            f"(vec={vectorizer is not None}, mask={selected_mask is not None}, "
            f"pipe={pipeline is not None}). "
            f"Expected paths: {artifacts_dir}/vectorizer_fold{fold_k}.pkl + "
            f"{artifacts_dir}/selected_idx_fold{fold_k}.json + "
            f"{artifacts_dir}/{cell_id}_lr_fold{fold_k}.pkl. "
            f"Hard-fail per /stress A2.10 P0-3-B user-mandate (NO silent "
            f"phantom_som fallback for infrastructure errors)."
        )
        logger.error(msg)
        raise LearnedRouterArtifactError(msg)

    # Build runtime feature vector
    # B-1600 hard-fail: feature vector dim mismatch = vocabulary inconsistency
    # between trained vectorizer + selected_idx, infrastructure-level corruption.
    try:
        x = build_runtime_feature_vector(raw_features, vectorizer, selected_mask)
    except ValueError as e:
        msg = (
            f"[learned_router] feature vector build failed (cell={cell_id} "
            f"fold_k={fold_k}): {e}. "
            f"Indicates Stage 2/3 vocab consistency violation between "
            f"vectorizer_fold{fold_k}.pkl + selected_idx_fold{fold_k}.json. "
            f"Hard-fail per /stress A2.10 P0-3-B user-mandate."
        )
        logger.error(msg)
        raise LearnedRouterArtifactError(msg) from e

    # Resolve τ for this fold
    thresholds_per_fold = cell_cache["cell_meta"].get("thresholds_per_fold", {})
    tau = thresholds_per_fold.get(str(fold_k), thresholds_per_fold.get(fold_k, 0.5))
    diag["tau_used"] = float(tau)

    # Predict + cost-weighted decision rule (B-998)
    # B-1600: pipeline.predict_proba exception is infrastructure-level (numpy
    # version mismatch / sklearn version drift / pickle compat); hard-fail.
    # max_prob ≤ τ is LEGITIMATE signal-strength fallback (per H10 cost-
    # weighted decision rule design B-998); stays silent + counter.
    try:
        probs = pipeline.predict_proba(x.reshape(1, -1))[0]
    except Exception as e:
        msg = (
            f"[learned_router] pipeline.predict_proba failed (cell={cell_id} "
            f"fold_k={fold_k}): {e}. Indicates sklearn / numpy version mismatch "
            f"or corrupt pickle. Hard-fail per /stress A2.10 P0-3-B user-mandate."
        )
        logger.error(msg)
        raise LearnedRouterArtifactError(msg) from e

    max_prob = float(probs.max())
    argmax_idx = int(probs.argmax())
    argmax_mode = str(pipeline.classes_[argmax_idx])
    diag["max_prob"] = max_prob
    diag["argmax_mode"] = argmax_mode

    if max_prob > tau:
        return argmax_mode, diag
    else:
        # B-1600 retained silent: this is the H10 cost-weighted decision rule's
        # legitimate signal-strength fallback per B-998. Counted as fallback in
        # diag so per-cell fallback rate can be disclosed (B-1601 P1-6 surface).
        diag["fallback_fired"] = True
        diag["fallback_reason"] = f"max_prob={max_prob:.3f} <= tau={tau:.3f}"
        return fallback_mode, diag


# ── Back-compat shims (deprecated, will be removed after runner refactor) ─────


def load_lr_pipeline(lr_model_path: str | Path) -> Optional[Any]:
    """DEPRECATED: pre-Chunk-B single-pickle loader. Kept for back-compat during
    runner refactor transition. New code should use load_lr_pipeline_fold."""
    path = Path(lr_model_path)
    if not path.exists():
        logger.warning("LR model artifact not found: %s", path)
        return None
    try:
        with path.open("rb") as f:
            pipeline = pickle.load(f)
        logger.info("Loaded LR router pickle: %s", path)
        return pipeline
    except (OSError, pickle.UnpicklingError) as e:
        logger.error("Failed to load LR pickle %s: %s", path, e)
        return None


def extract_task_features(
    task_intent: str,
    task_has_image: bool,
    site: str,
    axtree_element_count: int,
) -> np.ndarray:
    """DEPRECATED: pre-Chunk-A 8-dim feature schema. Kept for back-compat during
    runner refactor transition. New code should use extract_raw_features +
    build_runtime_feature_vector (Stage 1 → Stage 2)."""
    intent = task_intent or ""
    return np.array(
        [
            1.0 if site == "classifieds" else 0.0,
            1.0 if task_has_image else 0.0,
            1.0 if INTENT_REGEX["intent_color"].search(intent) else 0.0,
            1.0 if INTENT_REGEX["intent_search"].search(intent) else 0.0,
            1.0 if INTENT_REGEX["intent_compare"].search(intent) else 0.0,
            1.0 if INTENT_REGEX["intent_nav"].search(intent) else 0.0,
            float(len(intent.split())),
            float(axtree_element_count),
        ],
        dtype=float,
    ).reshape(1, -1)


def predict_mode(
    pipeline: Any,
    task_intent: str,
    task_has_image: bool,
    site: str,
    axtree_element_count: int,
    fallback_mode: str = "phantom_som",
) -> str:
    """DEPRECATED: pre-Chunk-B single-pickle predictor. Returns argmax without
    cost-weighted decision rule. Kept for back-compat during runner refactor
    transition. New code should use predict_mode_fold_aware."""
    if pipeline is None:
        logger.warning("LR pipeline is None; falling back to %s", fallback_mode)
        return fallback_mode
    try:
        X = extract_task_features(
            task_intent=task_intent,
            task_has_image=task_has_image,
            site=site,
            axtree_element_count=axtree_element_count,
        )
        pred = pipeline.predict(X)[0]
        return str(pred)
    except Exception as e:
        logger.error("LR predict failed; falling back to %s: %s", fallback_mode, e)
        return fallback_mode
