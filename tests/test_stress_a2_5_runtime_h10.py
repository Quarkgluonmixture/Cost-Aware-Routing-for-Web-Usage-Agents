"""A2.5 Chunk C — Runtime fold-aware + H10 verdict producer invariant tests.

Invariants tested:
  1. learned_router.extract_raw_features schema (5 numeric + 15 binary, no site/capability)
  2. learned_router.build_runtime_feature_vector applies selected_mask correctly
  3. learned_router.predict_mode_fold_aware caches per-cell artifacts lazily
  4. learned_router.predict_mode_fold_aware hard-fails (LearnedRouterArtifactError)
     when fold_assignment is missing task_id  [B-1640 / A2.10 P0-3-B: no silent
     phantom_som fallback for infrastructure errors]
  5. learned_router.predict_mode_fold_aware hard-fails when fold artifacts missing
  6. learned_router.INTENT_REGEX = 14 banks (must match Chunk A)
  7. aggregate_h10_pareto.fe_inverse_variance_pool basic correctness
  8. aggregate_h10_pareto.check_pareto_non_dominance returns valid fraction
  9. aggregate_h10_pareto.aggregate_arm_metrics empty handling
  10. Q4=A K-of-6 primary verdict structure
  11. Back-compat shims still importable (load_lr_pipeline / predict_mode / extract_task_features)
  12. SAFE_FALLBACK_MODE = "phantom_som" matches Chunk B
"""
from __future__ import annotations

import json
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "analysis"
sys.path.insert(0, str(SCRIPT_DIR))

from p79.policies.learned_router import (  # noqa: E402
    INTENT_REGEX,
    LearnedRouterArtifactError,
    N_FOLDS,
    SAFE_FALLBACK_MODE,
    build_runtime_feature_vector,
    extract_raw_features,
    extract_task_features,
    load_lr_pipeline,
    load_cell_meta,
    load_fold_assignment,
    load_selected_idx_fold,
    load_vectorizer_fold,
    predict_mode,
    predict_mode_fold_aware,
)
from aggregate_h10_pareto import (  # noqa: E402
    DELTA_PP,
    PARETO_NON_DOMINANCE_THRESHOLD,
    SINGLE_MODE_BASELINES,
    aggregate_arm_metrics,
    check_pareto_non_dominance_paired_bootstrap,
    fe_inverse_variance_pool,
    paired_bootstrap_arm_metrics,
)


# ── Invariant 1-2: extract_raw_features + build_runtime_feature_vector ────


def test_extract_raw_features_shape():
    rf = extract_raw_features(
        intent="Find blue kayak",
        has_reference_image=True,
        dom_complexity=50,
        text_length=2000,
        tokens_input_text=500,
        reasoning_difficulty=2,
    )
    assert rf["numeric"].shape == (5,)
    assert rf["binary"].shape == (15,)
    assert rf["intent_text"] == "Find blue kayak"


def test_extract_raw_features_intent_regex_correct():
    rf = extract_raw_features(
        intent="Find the cheapest blue kayak today",
        has_reference_image=False,
        dom_complexity=100,
        text_length=3000,
        tokens_input_text=750,
        reasoning_difficulty=3,
    )
    # binary[0] = has_ref_image=0
    assert rf["binary"][0] == 0
    # Following 14 in alphabetical order; intent matches color (blue), compare (cheapest),
    # search (find), temporal (today)
    feature_names = ["has_reference_image"] + sorted(INTENT_REGEX.keys())
    color_idx = feature_names.index("intent_color")
    compare_idx = feature_names.index("intent_compare")
    search_idx = feature_names.index("intent_search")
    temporal_idx = feature_names.index("intent_temporal")
    assert rf["binary"][color_idx] == 1
    assert rf["binary"][compare_idx] == 1
    assert rf["binary"][search_idx] == 1
    assert rf["binary"][temporal_idx] == 1


def test_build_runtime_feature_vector_applies_mask():
    """build_runtime_feature_vector should apply selected_mask correctly."""
    # Create a mock vectorizer + transform output
    class MockVec:
        def transform(self, texts):
            # Returns sparse-like with 5 features
            class Sparse:
                def toarray(self):
                    return np.array([[1.0, 0.0, 0.5, 0.0, 0.2]])
            return Sparse()
    vec = MockVec()
    rf = {
        "intent_text": "test",
        "numeric": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "binary": np.array([0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1], dtype=int),
    }
    # Total = 5 TF-IDF + 5 numeric + 15 binary = 25
    # Mask selects features 0, 5, 10, 15, 20 (every 5th)
    mask = np.zeros(25, dtype=bool)
    mask[[0, 5, 10, 15, 20]] = True
    x = build_runtime_feature_vector(rf, vec, mask)
    assert x.shape == (5,)
    # First selected = TF-IDF[0] = 1.0
    assert x[0] == 1.0
    # Second selected = numeric[0] = 1.0
    assert x[1] == 1.0


def test_build_runtime_feature_vector_dim_mismatch_raises():
    class MockVec:
        def transform(self, texts):
            class Sparse:
                def toarray(self):
                    return np.array([[1.0, 0.0]])  # 2 cols
            return Sparse()
    rf = {
        "intent_text": "test",
        "numeric": np.array([1.0] * 5),
        "binary": np.array([0] * 15, dtype=int),
    }
    # Total = 2 + 5 + 15 = 22
    mask = np.ones(50, dtype=bool)  # Wrong size: 50 not 22
    with pytest.raises(ValueError, match="mismatch"):
        build_runtime_feature_vector(rf, MockVec(), mask)


# ── Invariant 3-5: predict_mode_fold_aware caching + fallback ─────────────


def test_predict_mode_fold_aware_raises_when_task_not_in_fold_assignment():
    """B-1640 / A2.10 P0-3-B: a task absent from fold_assignment is an
    infrastructure error (training pipeline didn't cover it), NOT a signal-
    strength fallback — predict_mode_fold_aware must hard-fail loud, never
    silently route to phantom_som."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Write empty fold_assignment
        (td_path / "test_cell_fold_assignment.json").write_text(json.dumps({
            "cell_id": "test_cell",
            "fold_assignment": {},
        }))
        rf = extract_raw_features("test", False, 0, 0, 0, 0)
        cache: dict = {}
        with pytest.raises(LearnedRouterArtifactError, match="not in fold_assignment"):
            predict_mode_fold_aware(
                cell_id="test_cell",
                task_id=999,
                artifacts_dir=td_path,
                cache=cache,
                raw_features=rf,
            )


def test_predict_mode_fold_aware_raises_when_pipeline_missing():
    """B-1640 / A2.10 P0-3-B: missing fold artifacts (vectorizer / selected_idx /
    LR pipeline) is infrastructure-level corruption — hard-fail loud, never
    silent phantom_som fallback."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        # Fold assignment maps task 1 to fold 0
        (td_path / "test_cell_fold_assignment.json").write_text(json.dumps({
            "cell_id": "test_cell",
            "fold_assignment": {"1": 0},
        }))
        # No vectorizer / selected_idx / pipeline files written
        rf = extract_raw_features("test", False, 0, 0, 0, 0)
        cache: dict = {}
        with pytest.raises(LearnedRouterArtifactError, match="missing artifact"):
            predict_mode_fold_aware(
                cell_id="test_cell",
                task_id=1,
                artifacts_dir=td_path,
                cache=cache,
                raw_features=rf,
            )


def test_predict_mode_fold_aware_caches_artifacts():
    """Per-cell fold_assignment is cached on first touch and reused on the next
    call without re-reading files. The cache is populated *before* the B-1640
    missing-artifact hard-fail raises, so the caching invariant holds even when
    the prediction itself aborts (here both calls raise on missing fold pickles —
    we assert the same fold_assignment object is reused across them)."""
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        (td_path / "test_cell_fold_assignment.json").write_text(json.dumps({
            "fold_assignment": {"1": 0, "2": 0},
        }))
        rf = extract_raw_features("test", False, 0, 0, 0, 0)
        cache: dict = {}
        # First call populates cache, then hard-fails on missing fold artifacts.
        with pytest.raises(LearnedRouterArtifactError, match="missing artifact"):
            predict_mode_fold_aware("test_cell", 1, td_path, cache, rf)
        assert "test_cell" in cache
        first_fa = cache["test_cell"]["fold_assignment"]
        # Second call reuses the cached fold_assignment (no re-read) before it
        # too hard-fails on the same missing artifacts.
        with pytest.raises(LearnedRouterArtifactError, match="missing artifact"):
            predict_mode_fold_aware("test_cell", 2, td_path, cache, rf)
        # Cache structure unchanged — same object identity proves reuse.
        assert cache["test_cell"]["fold_assignment"] is first_fa


def test_predict_mode_fold_aware_raises_when_tau_missing():
    """S3 cross-AI P0-3-B* (2026-06-02): a per-fold τ absent from cell_meta
    thresholds_per_fold is artifact corruption (Stage 3 LR training did not emit
    a threshold for this fold), NOT a legitimate signal-strength fallback.
    Pre-fix code silently defaulted to τ=0.5 here, contradicting load_cell_meta's
    own B-1640 'no silent fallback to default τ' contract. Must hard-fail loud.

    Cache is pre-populated with mock-but-valid fold artifacts so the predictor
    reaches the τ-resolution step (which precedes predict_proba); only the
    cell_meta thresholds_per_fold is corrupted (missing fold 2)."""
    class MockVec:
        def transform(self, texts):
            class Sparse:
                def toarray(self):
                    return np.zeros((1, 30))  # 30 TF-IDF cols
            return Sparse()

    class MockPipe:
        classes_ = np.array(["dom", "som"])
        def predict_proba(self, x):  # not reached — τ raises first
            return np.array([[0.7, 0.3]])

    mask = np.zeros(50, dtype=bool)  # 30 tfidf + 5 numeric + 15 binary
    mask[:18] = True
    cache = {
        "test_cell": {
            "fold_assignment": {7: 2},
            # thresholds_per_fold deliberately MISSING fold 2 (has 0/1/3/4 only)
            "cell_meta": {"thresholds_per_fold": {"0": 0.4, "1": 0.4, "3": 0.4, "4": 0.4}},
            "vectorizers": {2: MockVec()},
            "selected_masks": {2: mask},
            "pipelines": {2: MockPipe()},
        }
    }
    rf = extract_raw_features("test", False, 0, 0, 0, 0)
    with pytest.raises(LearnedRouterArtifactError, match="missing τ"):
        predict_mode_fold_aware(
            cell_id="test_cell",
            task_id=7,
            artifacts_dir="/nonexistent",  # not read — cache pre-populated
            cache=cache,
            raw_features=rf,
        )


def test_predict_mode_fold_aware_accepts_tau_zero():
    """Guard the `is None` (not falsy) check in the P0-3-B fix: a legitimately
    trained τ=0.0 must be accepted, not mistaken for 'missing'. Routes to argmax
    since max_prob (0.7) > τ (0.0)."""
    class MockVec:
        def transform(self, texts):
            class Sparse:
                def toarray(self):
                    return np.zeros((1, 30))
            return Sparse()

    class MockPipe:
        classes_ = np.array(["dom", "som"])
        def predict_proba(self, x):
            return np.array([[0.7, 0.3]])

    mask = np.zeros(50, dtype=bool)
    mask[:18] = True
    cache = {
        "test_cell": {
            "fold_assignment": {7: 2},
            "cell_meta": {"thresholds_per_fold": {"2": 0.0}},  # τ=0.0 is legit
            "vectorizers": {2: MockVec()},
            "selected_masks": {2: mask},
            "pipelines": {2: MockPipe()},
        }
    }
    rf = extract_raw_features("test", False, 0, 0, 0, 0)
    mode, diag = predict_mode_fold_aware(
        cell_id="test_cell", task_id=7, artifacts_dir="/nonexistent",
        cache=cache, raw_features=rf,
    )
    assert diag["tau_used"] == 0.0
    assert mode == "dom"  # argmax (prob 0.7) since max_prob > τ=0.0


# ── Invariant 6: INTENT_REGEX has 14 banks ──────────────────────────────


def test_intent_regex_14_banks():
    """Must match scripts/analysis/extract_50_features.py count."""
    assert len(INTENT_REGEX) == 14


def test_n_folds_5():
    assert N_FOLDS == 5


def test_safe_fallback_mode_phantom_som():
    """Must match Chunk B trainer SAFE_FALLBACK_MODE."""
    assert SAFE_FALLBACK_MODE == "phantom_som"


# ── Invariant 7: FE inverse-variance pool ───────────────────────────────


def test_fe_inverse_variance_pool_two_cells():
    """Manual sanity: 2 cells, θ=[+2.0, -4.0], SE=[1.0, 1.0].
    Inverse-variance weights equal → pool = mean = -1.0pp."""
    result = fe_inverse_variance_pool([2.0, -4.0], [1.0, 1.0])
    assert abs(result["theta_pool_pp"] - (-1.0)) < 1e-6
    assert result["n_cells_pooled"] == 2


def test_fe_inverse_variance_pool_unequal_weights():
    """θ=[+2.0, -4.0], SE=[0.5, 2.0]. Weights: 4, 0.25. Pool = (2*4 + (-4)*0.25) / 4.25 ≈ 1.647pp."""
    result = fe_inverse_variance_pool([2.0, -4.0], [0.5, 2.0])
    expected = (2.0 * 4.0 + (-4.0) * 0.25) / 4.25
    assert abs(result["theta_pool_pp"] - expected) < 1e-6


def test_fe_inverse_variance_pool_insufficient_cells():
    result = fe_inverse_variance_pool([2.0], [1.0])
    assert result["reason"] == "insufficient_cells_for_pool"
    assert np.isnan(result["theta_pool_pp"])


def test_fe_inverse_variance_pool_filters_nans():
    """Cells with NaN θ or SE should be filtered out."""
    result = fe_inverse_variance_pool(
        [2.0, float("nan"), -4.0, 1.0],
        [1.0, 1.0, 1.0, float("nan")],
    )
    assert result["n_cells_pooled"] == 2  # Only first + third


# ── Invariant 8: Pareto non-dominance paired bootstrap ──────────────────


def test_pareto_non_dominance_router_dominates_all():
    """Router (Cost=0.05, SR=0.9) dominates all baselines → fraction_non_dominated=1.0."""
    np.random.seed(0)
    n = 100
    router_sr = np.array([1.0] * 90 + [0.0] * 10)  # SR=0.9
    router_cost = np.array([0.05] * n)
    baselines = {
        "dom": {
            "success": np.array([1.0] * 70 + [0.0] * 30),  # SR=0.7
            "cost": np.array([0.05] * n),  # Same cost
        },
        "som": {
            "success": np.array([1.0] * 80 + [0.0] * 20),  # SR=0.8
            "cost": np.array([0.1] * n),  # Higher cost
        },
    }
    result = check_pareto_non_dominance_paired_bootstrap(
        router_sr, router_cost, baselines, list(range(n)), B=100
    )
    # Router should be non-dominated in nearly all bootstrap replicates
    assert result["fraction_non_dominated"] > 0.9
    assert result["passes"]


def test_pareto_non_dominance_router_dominated():
    """Router (Cost=0.10, SR=0.5) dominated by baseline (Cost=0.05, SR=0.7)."""
    np.random.seed(1)
    n = 100
    router_sr = np.array([1.0] * 50 + [0.0] * 50)
    router_cost = np.array([0.10] * n)
    baselines = {
        "dom": {
            "success": np.array([1.0] * 70 + [0.0] * 30),
            "cost": np.array([0.05] * n),
        },
    }
    result = check_pareto_non_dominance_paired_bootstrap(
        router_sr, router_cost, baselines, list(range(n)), B=100
    )
    assert result["fraction_non_dominated"] < 0.5  # Often dominated
    assert not result["passes"]


# ── Invariant 9: aggregate_arm_metrics empty handling ────────────────────


def test_aggregate_arm_metrics_empty():
    s, c, l, tids = aggregate_arm_metrics({}, arm="dom")
    assert len(s) == 0
    assert len(c) == 0
    assert tids == []


def test_aggregate_arm_metrics_extracts_correct_arm():
    outcomes = {
        1: {"dom": {"success": 1, "cost_usd": 0.05, "latency_ms": 1000.0}},
        2: {"dom": {"success": 0, "cost_usd": 0.04, "latency_ms": 800.0},
            "som": {"success": 1, "cost_usd": 0.10, "latency_ms": 1500.0}},
    }
    s, c, l, tids = aggregate_arm_metrics(outcomes, arm="dom")
    assert s.tolist() == [1, 0]
    assert c.tolist() == [0.05, 0.04]
    assert l.tolist() == [1000.0, 800.0]
    assert tids == [1, 2]


# ── Invariant 10: K-of-6 PRIMARY structure ──────────────────────────────


def test_pareto_non_dominance_threshold_95_locked():
    """95% paired bootstrap support pre-locked per preregistration line 188."""
    assert PARETO_NON_DOMINANCE_THRESHOLD == 0.95


def test_delta_pp_locked_at_1():
    """δ=1.0pp mirror H1 estimand (preregistration §625 + line 212)."""
    assert DELTA_PP == 1.0


def test_single_mode_baselines_5_arms_p_prompt_excluded():
    """5-arm baseline set: P-prompt excluded per preregistration line 199-204
    (cls archive aborted at 4 ep; expands to 6 if Phase 1a B0+B1+B2 cls all produce ≥50 ep)."""
    assert SINGLE_MODE_BASELINES == ["dom", "som", "vision", "phantom_text", "phantom_som"]
    assert "phantom_prompt" not in SINGLE_MODE_BASELINES


# ── Invariant 11: Back-compat shims still importable ─────────────────────


def test_backcompat_load_lr_pipeline_still_callable():
    """Back-compat shim for pre-Chunk-B callers."""
    result = load_lr_pipeline("/nonexistent/path.pkl")
    assert result is None  # Returns None on missing file


def test_backcompat_extract_task_features_8dim():
    """Back-compat shim for pre-Chunk-A 8-dim feature extraction."""
    X = extract_task_features(
        task_intent="find blue kayak",
        task_has_image=True,
        site="classifieds",
        axtree_element_count=100,
    )
    assert X.shape == (1, 8)


def test_backcompat_predict_mode_returns_fallback_on_none_pipeline():
    """Back-compat shim returns fallback on None pipeline."""
    mode = predict_mode(
        pipeline=None,
        task_intent="test",
        task_has_image=False,
        site="reddit",
        axtree_element_count=50,
        fallback_mode="dom",
    )
    assert mode == "dom"


# ── Invariant 12: paired_bootstrap_arm_metrics ──────────────────────────


def test_paired_bootstrap_arm_metrics_returns_ci():
    np.random.seed(42)
    success = np.array([1] * 70 + [0] * 30, dtype=float)  # SR=0.7
    cost = np.array([0.05] * 100, dtype=float)
    result = paired_bootstrap_arm_metrics(success, cost, B=200, seed=42)
    assert result["n"] == 100
    assert abs(result["sr_mean"] - 0.7) < 1e-6
    # CI should bracket 0.7
    assert result["sr_ci"][0] <= 0.7 <= result["sr_ci"][1]


def test_paired_bootstrap_arm_metrics_empty():
    result = paired_bootstrap_arm_metrics(np.array([]), np.array([]), B=100)
    assert result["n"] == 0
    assert np.isnan(result["sr_mean"])
