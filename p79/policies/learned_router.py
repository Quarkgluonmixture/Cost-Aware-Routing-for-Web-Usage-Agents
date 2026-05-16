"""Learned router runtime predictor (paper-1 v7, 2026-05-16).

Loads a per-cell LR pickle (trained by `scripts/analysis/train_l1_router.py`)
and predicts the observation_mode for each task at runtime.

Feature schema mirrors `scripts/analysis/l1_archive_simulation.py` exactly:
- site one-hot (cls / red — shop pending Phase 1b)
- has_reference_image (from task config `image` field)
- intent_color_regex / intent_compare_regex / intent_search_regex / intent_nav_regex
- intent_token_count (z-scored in-fold during training)
- axtree_element_count (step-0 obs.text line count; z-scored in-fold)

The LR pickle stores a sklearn Pipeline with ColumnTransformer (StandardScaler
on numeric cols 6+7) + LogisticRegression(class_weight=balanced).

Train artifact: `results/phantom_paper/l1_router/<baseline>_<site>_lr.pkl`
Train script: `scripts/analysis/train_l1_router.py` (TODO, separate session)

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

# Intent regex banks (mechanism-anchored; identical to l1_archive_simulation.py:24-27)
COLOR_RE = re.compile(
    r"\b(color|red|blue|green|yellow|black|white|orange|purple|pink|brown|gray|grey)\b",
    re.I,
)
SEARCH_RE = re.compile(r"\b(find|search|locate|how many|how much)\b", re.I)
COMPARE_RE = re.compile(
    r"\b(cheapest|most expensive|highest|lowest|best|worst|biggest|smallest)\b", re.I
)
NAV_RE = re.compile(r"\b(go to|navigate|open|visit)\b", re.I)


def load_lr_pipeline(lr_model_path: str | Path) -> Optional[Any]:
    """Load a trained LR pickle. Returns None if file missing or unpicklable."""
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
    """Build the 8-dim feature vector matching l1_archive_simulation.py:51-95.

    Note: intent_tok_count + axtree_element_count are NOT z-scored here;
    the trained Pipeline applies StandardScaler internally (fitted on train fold).

    Column order (must match train-time):
        0: site_cls (1.0 if classifieds else 0.0)
        1: has_image (1.0 if has_reference_image else 0.0)
        2: color_intent
        3: search_intent
        4: compare_intent
        5: nav_intent
        6: intent_tok_count (raw — Pipeline scales)
        7: axtree_elements (raw — Pipeline scales)
    """
    intent = task_intent or ""
    return np.array(
        [
            1.0 if site == "classifieds" else 0.0,
            1.0 if task_has_image else 0.0,
            1.0 if COLOR_RE.search(intent) else 0.0,
            1.0 if SEARCH_RE.search(intent) else 0.0,
            1.0 if COMPARE_RE.search(intent) else 0.0,
            1.0 if NAV_RE.search(intent) else 0.0,
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
    """Predict observation_mode for a task. Returns fallback_mode on any failure."""
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
