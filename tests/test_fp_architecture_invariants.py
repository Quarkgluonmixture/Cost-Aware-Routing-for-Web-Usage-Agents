"""FP architecture invariants — guard the §139.8 + /stress A1.6 retire.

These tests fail loudly if a future refactor reintroduces the post-hoc
adjusted-success / na_fp / eval_fp / visual_fp dead code, or if the
N/A definition diverges between `tasks._is_na_task` and
`analysis._load_na_task_ids`.

Created 2026-05-16 per /stress A1.6 P2-4 finding.
"""
from __future__ import annotations

import importlib
import inspect

import pytest


# ---------------------------------------------------------------------------
# scored_task_count — single source of truth for EXPECTED_N
# ---------------------------------------------------------------------------

def test_scored_task_count_post_exclusion_values():
    """Canonical post-N/A-exclusion counts per memory `reference_fp_architecture_2026-05-14`."""
    from p79.experiment.analysis import scored_task_count
    assert scored_task_count("classifieds", "visualwebarena") == 224
    assert scored_task_count("reddit", "visualwebarena") == 205
    assert scored_task_count("shopping", "visualwebarena") == 435


def test_scored_task_count_strict_raises_on_missing_config():
    """strict=True must fail loud, not silently return 0."""
    from p79.experiment.analysis import scored_task_count
    with pytest.raises(FileNotFoundError):
        scored_task_count("nonexistent_site", "visualwebarena", strict=True)


def test_scored_task_count_non_strict_falls_back_to_zero():
    """Default (non-strict) keeps the silent-0 fallback for non-paper-grade callers."""
    from p79.experiment.analysis import scored_task_count
    assert scored_task_count("nonexistent_site", "visualwebarena") == 0


# ---------------------------------------------------------------------------
# N/A definition single-source — tasks._is_na_task ↔ analysis._load_na_task_ids
# ---------------------------------------------------------------------------

def test_na_definition_single_sourced():
    """analysis._load_na_task_ids must reuse tasks._is_na_task — no duplicated regex."""
    from p79.experiment import analysis
    src = inspect.getsource(analysis._load_na_task_ids)
    assert "_is_na_task" in src, (
        "_load_na_task_ids must import _is_na_task from tasks.py (DRY single-source); "
        "duplicated `fuzzy_match == 'N/A'` logic risks drift if the N/A definition is extended."
    )


def test_is_na_task_basic_contract():
    """_is_na_task: fuzzy_match == 'N/A' → True; everything else → False."""
    from p79.experiment.tasks import _is_na_task
    assert _is_na_task({"eval": {"reference_answers": {"fuzzy_match": "N/A"}}}) is True
    assert _is_na_task({"eval": {"reference_answers": {"fuzzy_match": "not n/a"}}}) is False
    assert _is_na_task({"eval": {"reference_answers": {"exact_match": "0"}}}) is False
    assert _is_na_task({}) is False


# ---------------------------------------------------------------------------
# EpisodeSummaryV2 schema — retired fields must NOT come back
# ---------------------------------------------------------------------------

def test_episode_summary_v2_has_no_retired_fields():
    """`adjusted_success` / `fp_reason` are §139.8-retired and must not be re-added."""
    from p79.experiment.types import EpisodeSummaryV2
    fields = {f.name for f in EpisodeSummaryV2.__dataclass_fields__.values()}
    assert "adjusted_success" not in fields, (
        "EpisodeSummaryV2.adjusted_success was retired in §139.8; readding it would "
        "resurrect the post-hoc layer that /stress A1.6 hard-deleted on 2026-05-16."
    )
    assert "fp_reason" not in fields, (
        "EpisodeSummaryV2.fp_reason was retired in §139.8; FP-by-type book-keeping "
        "is replaced by upstream B-91 + task-load N/A exclusion."
    )


# ---------------------------------------------------------------------------
# Retired helper / module symbols
# ---------------------------------------------------------------------------

def test_compute_adjusted_success_is_retired():
    """The post-hoc helpers were removed in §139.8 — re-adding them is a regression."""
    analysis = importlib.import_module("p79.experiment.analysis")
    assert not hasattr(analysis, "compute_adjusted_success"), (
        "`compute_adjusted_success` was retired in §139.8 piece 4c; re-adding it "
        "violates the canonical-success invariant."
    )
    assert not hasattr(analysis, "compute_adjusted_success_batch"), (
        "`compute_adjusted_success_batch` was retired in §139.8 piece 4c."
    )


def test_cross_representation_no_mark_false_positives():
    """`_mark_false_positives` thin alias-setter was hard-deleted on 2026-05-16."""
    mod = importlib.import_module("scripts.analysis.analyze_cross_representation")
    assert not hasattr(mod, "_mark_false_positives"), (
        "`_mark_false_positives` was hard-deleted in /stress A1.6 (2026-05-16); "
        "re-adding it would reintroduce the 0-alias output-schema cargo cult."
    )


# ---------------------------------------------------------------------------
# Config defaults — N/A exclusion remains opt-out, not opt-in
# ---------------------------------------------------------------------------

def test_exclude_na_tasks_default_true():
    """task.exclude_na_tasks default must remain True — flipping it changes the SR denominator."""
    from p79.experiment.config import normalize_config
    cfg = normalize_config({})
    assert cfg.get("task", {}).get("exclude_na_tasks") is True, (
        "task.exclude_na_tasks default flipped to False; this changes the "
        "SR denominator from post-exclusion (cls 224 / red 205 / shop 435) "
        "back to pre-exclusion (234 / 210 / 466) and silently breaks paper §3."
    )
