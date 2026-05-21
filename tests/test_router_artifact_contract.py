"""Artifact-contract tests for the learned router (router /stress B-1813 C6).

A corrupt / version-incompatible artifact (sklearn/numpy drift → AttributeError /
ModuleNotFoundError during pickle.load) must raise LearnedRouterArtifactError so the
runtime hard-fails (B-1640), NOT escape the old narrow (OSError, UnpicklingError) catch
into main.py's generic `except Exception` → silent safe_fallback. Missing (not corrupt)
artifacts still return None/{} so the caller's None-check is what hard-fails.
"""
import pytest

from p79.policies.learned_router import (
    LearnedRouterArtifactError,
    load_cell_meta,
    load_fold_assignment,
    load_lr_pipeline_fold,
    load_selected_idx_fold,
    load_vectorizer_fold,
)


def test_b1813_corrupt_pickle_raises(tmp_path):
    bad = tmp_path / "B0_classifieds_lr_fold0.pkl"
    bad.write_text("not a pickle (simulates sklearn/numpy version drift)")
    with pytest.raises(LearnedRouterArtifactError):
        load_lr_pipeline_fold(tmp_path, "B0_classifieds", 0)


def test_b1813_corrupt_vectorizer_raises(tmp_path):
    bad = tmp_path / "vectorizer_fold0.pkl"
    bad.write_bytes(b"\x80\x04corrupt-not-a-valid-pickle-stream")
    with pytest.raises(LearnedRouterArtifactError):
        load_vectorizer_fold(tmp_path, 0)


def test_b1813_corrupt_json_artifacts_raise(tmp_path):
    (tmp_path / "selected_idx_fold0.json").write_text("{ not valid json")
    with pytest.raises(LearnedRouterArtifactError):
        load_selected_idx_fold(tmp_path, 0)

    (tmp_path / "B0_classifieds_fold_assignment.json").write_text("{ broken")
    with pytest.raises(LearnedRouterArtifactError):
        load_fold_assignment(tmp_path, "B0_classifieds")

    (tmp_path / "B0_classifieds_lr_meta.json").write_text("definitely not json")
    with pytest.raises(LearnedRouterArtifactError):
        load_cell_meta(tmp_path, "B0_classifieds")


def test_b1813_missing_artifacts_return_none_not_raise(tmp_path):
    # Missing (not corrupt) → None/{}; the caller's None-check hard-fails, not the
    # loader. This preserves the "missing artifact" path while closing the
    # "corrupt/version-drift" escape hatch.
    assert load_lr_pipeline_fold(tmp_path, "B0_classifieds", 0) is None
    assert load_vectorizer_fold(tmp_path, 0) is None
    assert load_selected_idx_fold(tmp_path, 0) == (None, None)
    assert load_fold_assignment(tmp_path, "B0_classifieds") == {}
    assert load_cell_meta(tmp_path, "B0_classifieds") == {}
