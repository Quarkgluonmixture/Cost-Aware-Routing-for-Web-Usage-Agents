"""Negative tests for paper-grade step / episode / run summary validators.

B-295 fix (2026-05-16, A1.8): pre-fix tests only covered the happy path
(`test_step_schema_validation_required_fields`) + one "missing router"
case. Validator attack surface (type drift, missing critical optionals,
string-truthy success) had no automated coverage. These tests lock the
B-280/B-281/B-283/B-285/B-296 contract behaviour.
"""
from __future__ import annotations

import pytest

from p79.experiment.types import (
    PAPER_GRADE_STEP_OPTIONAL_KEYS,
    SCHEMA_VERSION_V2,
    validate_episode_summary_v2,
    validate_run_summary_v2,
    validate_step_record_v2,
)


# ---------------------------------------------------------------------------
# Fixtures: minimal valid records that pass each validator.
# ---------------------------------------------------------------------------


def _valid_step_record():
    rec = {
        "schema_version": SCHEMA_VERSION_V2,
        "run_id": "run_x", "condition_id": "c1",
        "benchmark": "vwa", "benchmark_site": "cls",
        "task_id": 0, "seed": 42, "step_idx": 0,
        "som": {}, "observation_mode": "dom",
        "router": {}, "module_flags": {},
        "action_type": "wait", "action": {},
        "action_success": False, "page_changed": False,
        "latency_ms": {}, "tokens": {},
        # B-338 (/stress A1.9 Mode B F7, 2026-05-16): cost_usd nested key
        # validator now requires {input, output, model, router_overhead, total}.
        "cost_usd": {
            "input": 0.0, "output": 0.0, "model": 0.0,
            "router_overhead": 0.0, "total": 0.0,
        },
        "energy": {},
        "retry_count": 0, "error_category": None,
        "artifact_paths": {}, "reward": 0.0, "done": False,
    }
    # B-280: critical optionals must be present (value may be None)
    for k in PAPER_GRADE_STEP_OPTIONAL_KEYS:
        rec[k] = None
    return rec


def _valid_episode_summary():
    # Fire-6 RCA Stage C1 fixture-drift fix (2026-05-20): derive from
    # EPISODE_SUMMARY_V2_DEFAULTS via the shared conftest helper so the full
    # PAPER_GRADE_EPISODE_OPTIONAL_KEYS set (B-732 7 sentinels + eval-context
    # provenance + M5 timeout taxonomy + Phase-2 intervention rollup +
    # diagnostic_replay / sr_excluded) auto-populates instead of drifting
    # against a hardcoded field list.
    from conftest import complete_episode_summary
    return complete_episode_summary(
        run_id="run_x", condition_id="c1",
        benchmark="vwa", benchmark_site="cls",
        task_id=0, seed=42,
        success=False, score=0.0, steps=0,
    )


def _valid_run_summary():
    return {
        "schema_version": SCHEMA_VERSION_V2,
        "run_id": "run_x", "benchmark": "vwa", "phase": "phase1",
        "total_conditions": 1, "total_episodes": 0,
        "condition_metrics": [], "assumptions": {},
    }


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_valid_step_passes():
    validate_step_record_v2(_valid_step_record())


def test_valid_episode_passes():
    validate_episode_summary_v2(_valid_episode_summary())


def test_valid_run_summary_passes():
    validate_run_summary_v2(_valid_run_summary())


# ---------------------------------------------------------------------------
# B-280: type-drift attack vector (codex Mode B F1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field,bad_value", [
    ("som", None),
    ("som", "not-a-dict"),
    ("latency_ms", "0"),
    ("tokens", []),
    ("task_id", "0"),
    ("seed", 42.0),  # float not int
    ("action_success", "true"),
    ("page_changed", 1),  # int not bool
    ("reward", "0.0"),  # string not numeric
    # schema_version float case omitted: validator checks equality before
    # type, so float 2.0 → "unexpected schema_version" raise rather than
    # "type mismatch" — different error path, separate test below.
])
def test_step_validator_rejects_type_mismatch(field, bad_value):
    rec = _valid_step_record()
    rec[field] = bad_value
    with pytest.raises(ValueError, match="type mismatch"):
        validate_step_record_v2(rec)


def test_step_validator_rejects_float_schema_version():
    """schema_version equality check fires before type check, so a non-str
    value (e.g. JSON literal 2.0 instead of "2.0") raises the equality error."""
    rec = _valid_step_record()
    rec["schema_version"] = 2.0
    with pytest.raises(ValueError, match="schema_version"):
        validate_step_record_v2(rec)


# ---------------------------------------------------------------------------
# B-280: critical optional keys must be present even when None
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing_key", sorted(PAPER_GRADE_STEP_OPTIONAL_KEYS))
def test_step_validator_requires_critical_optional_keys(missing_key):
    rec = _valid_step_record()
    del rec[missing_key]
    with pytest.raises(ValueError, match="missing paper-grade critical optional keys"):
        validate_step_record_v2(rec)


# ---------------------------------------------------------------------------
# B-283: episode summary string-truthy attack (codex Mode B F3)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_success", ["false", "true", "0", 0, 1, None, 1.0])
def test_episode_validator_rejects_non_bool_success(bad_success):
    rec = _valid_episode_summary()
    rec["success"] = bad_success
    with pytest.raises(ValueError, match="type mismatch"):
        validate_episode_summary_v2(rec)


def test_episode_validator_rejects_non_int_steps():
    rec = _valid_episode_summary()
    rec["steps"] = "5"
    with pytest.raises(ValueError, match="type mismatch"):
        validate_episode_summary_v2(rec)


def test_episode_validator_rejects_non_numeric_score():
    rec = _valid_episode_summary()
    rec["score"] = "1.0"
    with pytest.raises(ValueError, match="type mismatch"):
        validate_episode_summary_v2(rec)


# ---------------------------------------------------------------------------
# B-296: run summary validator
# ---------------------------------------------------------------------------


def test_run_summary_validator_rejects_non_list_condition_metrics():
    rec = _valid_run_summary()
    rec["condition_metrics"] = {}
    with pytest.raises(ValueError, match="condition_metrics: expected list"):
        validate_run_summary_v2(rec)


def test_run_summary_validator_rejects_non_dict_assumptions():
    rec = _valid_run_summary()
    rec["assumptions"] = []
    with pytest.raises(ValueError, match="assumptions: expected dict"):
        validate_run_summary_v2(rec)


# ---------------------------------------------------------------------------
# B-282: schema version semver alignment
# ---------------------------------------------------------------------------


def test_step_validator_rejects_wrong_schema_version():
    rec = _valid_step_record()
    rec["schema_version"] = "v2"  # old short form
    with pytest.raises(ValueError, match="schema_version"):
        validate_step_record_v2(rec)
    rec["schema_version"] = "1.0"
    with pytest.raises(ValueError, match="schema_version"):
        validate_step_record_v2(rec)
