"""Learned router runtime test — /stress A1.12 P0-5 AB (2026-05-17, B-668).

Companion to `tests/test_stress_a1_10_fixes.py::test_b388_*_learned_router_*`
which only verifies that `conditions.py` emits `router_variant=v7_learned`
string — i.e. that the condition gets GENERATED. This file exercises the
RUNTIME predictor: feature extraction, pickle load, predict_mode fallback,
task_image_field parsing.

Paper-1 contribution 2 (router, paper §6) hinges on `p79/policies/learned_router.py`
behaving correctly at predict time. Pre-fix the entire module had 0 test
coverage; a feature column order drift, missing pickle fallback, or site
one-hot regression would only surface during Pass-2 router fire — too late.
"""
from __future__ import annotations

import json
import pickle
import unittest.mock as mock
from pathlib import Path

import numpy as np
import pytest

from p79.policies.learned_router import (
    COLOR_RE,
    COMPARE_RE,
    NAV_RE,
    SEARCH_RE,
    extract_task_features,
    load_lr_pipeline,
    load_task_image_field,
    predict_mode,
)


# ─── extract_task_features: exact 8-column vector lock ─────────────────────
def test_extract_task_features_returns_8_column_2d_array():
    """Pipeline expects shape (1, 8) — anything else = train/predict drift."""
    X = extract_task_features("find red shoes", task_has_image=False,
                              site="classifieds", axtree_element_count=42)
    assert X.shape == (1, 8), f"expected (1, 8), got {X.shape}"
    assert X.dtype == float


def test_extract_task_features_column_order_locked():
    """Feature column order must match l1_archive_simulation.py train-time.

    Lock the contract: cols 0..7 = site_cls / has_image / color / search /
    compare / nav / intent_tok_count / axtree_element_count. Reorder = silent
    paper §6 prediction drift.
    """
    # Intent hits color + search + compare (intentionally; nav absent)
    X = extract_task_features(
        task_intent="find the cheapest red shoes",
        task_has_image=True,
        site="classifieds",
        axtree_element_count=100,
    )
    flat = X.flatten()
    assert flat[0] == 1.0, "col 0 = site_cls (classifieds → 1.0)"
    assert flat[1] == 1.0, "col 1 = has_image"
    assert flat[2] == 1.0, "col 2 = color intent"
    assert flat[3] == 1.0, "col 3 = search intent"
    assert flat[4] == 1.0, "col 4 = compare intent (cheapest)"
    assert flat[5] == 0.0, "col 5 = nav intent (absent)"
    assert flat[6] == 5.0, "col 6 = intent token count (find/the/cheapest/red/shoes = 5 words)"
    assert flat[7] == 100.0, "col 7 = axtree element count (raw — Pipeline scales)"


def test_extract_task_features_site_one_hot_reddit_is_zero():
    """site_cls = 0.0 for non-classifieds sites (reddit / shopping)."""
    X = extract_task_features("test", task_has_image=False, site="reddit",
                              axtree_element_count=10)
    assert X[0, 0] == 0.0


def test_extract_task_features_empty_intent_handled():
    """Empty intent must not crash + intent_tok_count = 0."""
    X = extract_task_features("", task_has_image=False, site="reddit",
                              axtree_element_count=0)
    assert X.shape == (1, 8)
    assert X[0, 6] == 0.0  # intent_tok_count = len("".split()) = 0


def test_extract_task_features_none_intent_treated_as_empty():
    """None intent (defensive — should not crash)."""
    X = extract_task_features(None, task_has_image=False, site="reddit",  # type: ignore[arg-type]
                              axtree_element_count=0)
    assert X.shape == (1, 8)
    assert X[0, 6] == 0.0


# ─── load_lr_pipeline: missing file + corrupt pickle fallback ──────────────
def test_load_lr_pipeline_returns_none_on_missing_file(tmp_path):
    """Non-existent path → None (do not raise, runner handles gracefully)."""
    result = load_lr_pipeline(tmp_path / "nonexistent.pkl")
    assert result is None


def test_load_lr_pipeline_returns_none_on_corrupt_pickle(tmp_path):
    """Corrupt pickle bytes → None (do not raise; UnpicklingError caught)."""
    bad = tmp_path / "corrupt.pkl"
    bad.write_bytes(b"NOT_A_PICKLE_AT_ALL")
    result = load_lr_pipeline(bad)
    assert result is None


def test_load_lr_pipeline_loads_valid_pickle(tmp_path):
    """Valid pickle → returns the unpickled object (round-trip)."""
    obj = {"fake": "pipeline"}
    good = tmp_path / "valid.pkl"
    good.write_bytes(pickle.dumps(obj))
    result = load_lr_pipeline(good)
    assert result == obj


# ─── predict_mode: fallback contracts ──────────────────────────────────────
def test_predict_mode_returns_fallback_when_pipeline_is_none():
    """Pipeline=None → fallback_mode (no inference attempt)."""
    out = predict_mode(None, task_intent="x", task_has_image=False,
                       site="reddit", axtree_element_count=0,
                       fallback_mode="phantom_som")
    assert out == "phantom_som"


def test_predict_mode_returns_fallback_on_pipeline_exception():
    """Pipeline.predict raises → fallback_mode (no propagation)."""
    fake_pipeline = mock.MagicMock()
    fake_pipeline.predict.side_effect = RuntimeError("simulated sklearn failure")
    out = predict_mode(fake_pipeline, task_intent="x", task_has_image=False,
                       site="reddit", axtree_element_count=0,
                       fallback_mode="dom")
    assert out == "dom"


def test_predict_mode_returns_str_of_pipeline_prediction():
    """Successful predict → str() of pipeline's first prediction."""
    fake_pipeline = mock.MagicMock()
    fake_pipeline.predict.return_value = np.array(["som"])
    out = predict_mode(fake_pipeline, task_intent="x", task_has_image=False,
                       site="reddit", axtree_element_count=0,
                       fallback_mode="dom")
    assert out == "som"


def test_predict_mode_custom_fallback_respected():
    """fallback_mode is parameterizable; default is 'phantom_som' but caller can override."""
    out = predict_mode(None, task_intent="x", task_has_image=False,
                       site="reddit", axtree_element_count=0,
                       fallback_mode="vision")
    assert out == "vision"


# ─── load_task_image_field: 4 image-key shapes ─────────────────────────────
def test_load_task_image_field_returns_true_for_real_image_path(tmp_path):
    """image field is a real path string → True."""
    cfg = tmp_path / "task.json"
    cfg.write_text(json.dumps({"image": "/path/to/img.png"}))
    assert load_task_image_field(cfg) is True


def test_load_task_image_field_returns_false_for_none(tmp_path):
    """image field is JSON null → False."""
    cfg = tmp_path / "task.json"
    cfg.write_text(json.dumps({"image": None}))
    assert load_task_image_field(cfg) is False


def test_load_task_image_field_returns_false_for_string_none(tmp_path):
    """image field is string "None" (VWA legacy quirk) → False."""
    cfg = tmp_path / "task.json"
    cfg.write_text(json.dumps({"image": "None"}))
    assert load_task_image_field(cfg) is False


def test_load_task_image_field_returns_false_for_empty_list(tmp_path):
    """image field is empty list → False."""
    cfg = tmp_path / "task.json"
    cfg.write_text(json.dumps({"image": []}))
    assert load_task_image_field(cfg) is False


def test_load_task_image_field_returns_false_for_missing_image_key(tmp_path):
    """image key absent → False."""
    cfg = tmp_path / "task.json"
    cfg.write_text(json.dumps({"other_field": "x"}))
    assert load_task_image_field(cfg) is False


def test_load_task_image_field_returns_false_on_corrupt_json(tmp_path):
    """Corrupt JSON → False (no propagation; logged warning + safe default)."""
    cfg = tmp_path / "task.json"
    cfg.write_text("NOT-JSON")
    assert load_task_image_field(cfg) is False


def test_load_task_image_field_returns_false_on_missing_file(tmp_path):
    """File doesn't exist → False (no propagation; logged warning + safe default)."""
    assert load_task_image_field(tmp_path / "nonexistent.json") is False


# ─── Regex bank sanity (mechanism-anchored, must match train-time) ──────────
def test_regex_banks_match_canonical_keywords():
    """Sanity-check that mechanism-anchored regexes still fire on expected keywords.

    These regexes are SHARED with `l1_archive_simulation.py:24-27` (train time)
    and `extract_task_features` (predict time). Drift = train/predict mismatch.
    """
    assert COLOR_RE.search("the red shoes")
    assert COLOR_RE.search("Color: blue")
    assert SEARCH_RE.search("find the cheapest")
    assert SEARCH_RE.search("how many items")
    assert COMPARE_RE.search("cheapest item")
    assert COMPARE_RE.search("the most expensive")
    assert NAV_RE.search("go to the homepage")
    assert NAV_RE.search("Navigate to /shop")
    # Negative sanity: nav regex should NOT fire on unrelated text
    assert not NAV_RE.search("post a comment")
