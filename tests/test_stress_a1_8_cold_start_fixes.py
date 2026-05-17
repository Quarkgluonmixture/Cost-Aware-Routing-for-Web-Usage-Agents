"""Regression tests for /stress A1.8 cold-start paper-grade fixes (B-731~B-740).

Cold-start re-audit of A1.8 scope (types.py / io_utils.py / logger_v2.py /
schema_migrations/). 3-AI cycle (Claude Mode A + codex Mode B + gemini Mode C)
caught 13 unique findings (8 Claude / 6 codex / 5 gemini, 2-AI overlap on 2,
1-AI unique on 11). User directive 2026-05-17: "archive 不进 paper scope" →
P0-2 PAPER_GRADE_EPISODE_OPTIONAL_KEYS enforces full 7-sentinel set without
backward-compat hook.

Fixes covered:
- B-731 (P0-1-AC*): _STEP_OPTIONAL_FIELD_TYPES value-type guard
- B-732 (P0-2-C*): PAPER_GRADE_EPISODE_OPTIONAL_KEYS + _EPISODE_OPTIONAL_FIELD_TYPES
- B-733 (P0-3-AB*): _validate_against_summary tri-state Optional[bool] on missing
- B-734 (P0-4-B*): load_episode_summary_strict needs_reevaluation type-guard
- B-735 (P1-1-C*): score bool subclass-of-int exclude
- B-736 (P1-2-A*): fcntl.flock concurrent-append PIPE_BUF protection
- B-737 (P1-3-AB): dedup_restart_lines string-`"0"` type guard
- B-738 (P1-4-B): _validate_against_summary multi-row identity check
- B-739 (P1-5-A): _EPISODE_FIELD_TYPES full hero field coverage
- B-740 (P1-6-C): import-time _STEP_FIELD_TYPES drift invariant
"""
from __future__ import annotations

import json
import multiprocessing as mp
import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest


# ─── B-740 import-time invariant ────────────────────────────────────────────
def test_b740_step_field_types_drift_invariant_passes_on_import():
    """Module import succeeds iff _STEP_FIELD_TYPES is in sync with the
    auto-derived REQUIRED_STEP_FIELDS_V2 set. Any future required dataclass
    field addition without matching _STEP_FIELD_TYPES entry will trigger
    AssertionError at import time. This test verifies the invariant is
    intact in the committed code."""
    from p79.experiment.types import (
        REQUIRED_STEP_FIELDS_V2, _STEP_FIELD_TYPES,
    )
    assert set(_STEP_FIELD_TYPES.keys()) == REQUIRED_STEP_FIELDS_V2, (
        f"_STEP_FIELD_TYPES drift: {set(_STEP_FIELD_TYPES.keys()) ^ REQUIRED_STEP_FIELDS_V2}"
    )


# ─── B-731 step record optional value types ─────────────────────────────────
def _minimal_valid_step_record() -> Dict[str, Any]:
    """Baseline record that passes validate_step_record_v2 — used to perturb
    one field at a time for poisoned-input tests."""
    from p79.experiment.types import SCHEMA_VERSION_V2
    return {
        "schema_version": SCHEMA_VERSION_V2,
        "run_id": "r1", "condition_id": "c1", "benchmark": "vwa",
        "benchmark_site": "classifieds", "task_id": 1, "seed": 42,
        "step_idx": 0, "som": {}, "observation_mode": "dom",
        "router": {}, "module_flags": {}, "action_type": "click",
        "action": {}, "action_success": True, "page_changed": True,
        "latency_ms": {}, "tokens": {},
        "cost_usd": {"input": 0.0, "output": 0.0, "model": 0.0,
                     "router_overhead": 0.0, "total": 0.0},
        "energy": {}, "retry_count": 0, "error_category": None,
        "artifact_paths": {}, "reward": 0.0, "done": False,
        "parse_valid": True, "parse_failure_reason": None,
        "image_meta": None, "image_meta_recorded": False,
        "locator_route_meta": None,
        "locator_route_meta_primary": None,
        "locator_route_meta_retry": None,
        "select_option_meta": None,
        "select_option_meta_primary": None,
        "select_option_meta_retry": None,
        "agent_visible_changed": True, "control_intervention": None,
        "dialog_meta": None, "action_executed": None,
        "fallback_finish": False, "element_bbox": None,
        "cost_unit_basis": None, "cost_total_mixed_unit_warn": False,
        "network_retry_count": None, "network_retry_wait_ms": None,
    }


def test_b731_valid_step_record_passes():
    from p79.experiment.types import validate_step_record_v2
    validate_step_record_v2(_minimal_valid_step_record())


@pytest.mark.parametrize("field,poison_value,expected_type_hint", [
    ("parse_valid", "false", "bool"),
    ("agent_visible_changed", "true", "bool"),
    ("fallback_finish", "true", "bool"),
    ("image_meta", "{'k': 'v'}", "dict"),
    ("cost_total_mixed_unit_warn", "true", "bool"),
    ("network_retry_count", "five", "int"),
    ("network_retry_wait_ms", "100", "int"),
])
def test_b731_step_record_optional_string_coercion_attack_caught(
    field, poison_value, expected_type_hint
):
    """String values for paper-grade optional fields trigger ValueError —
    the B-283 attack vector for `success` now extended to step-level."""
    from p79.experiment.types import validate_step_record_v2
    record = _minimal_valid_step_record()
    record[field] = poison_value
    with pytest.raises(ValueError, match=f"{field}.*expected.*{expected_type_hint}"):
        validate_step_record_v2(record)


def test_b731_step_record_optional_none_accepted():
    """None is always a valid value for optional fields (semantic: not
    measured this step)."""
    from p79.experiment.types import validate_step_record_v2
    record = _minimal_valid_step_record()
    for field in ("parse_valid", "agent_visible_changed", "fallback_finish",
                  "cost_total_mixed_unit_warn", "network_retry_count",
                  "network_retry_wait_ms"):
        record[field] = None
    validate_step_record_v2(record)  # should not raise


# ─── B-732 episode summary optional keys ────────────────────────────────────
def _minimal_valid_episode_summary() -> Dict[str, Any]:
    from p79.experiment.types import SCHEMA_VERSION_V2
    return {
        "schema_version": SCHEMA_VERSION_V2,
        "run_id": "r1", "condition_id": "c1", "benchmark": "vwa",
        "benchmark_site": "classifieds", "task_id": 1, "seed": 42,
        "success": True, "score": 1.0, "steps": 2, "retries": 0,
        "no_op_rate": 0.0, "page_unchanged_rate": 0.0,
        "total_latency_ms": 100.0, "p95_step_latency_ms": 50.0,
        "total_tokens": 100, "total_model_cost_usd": 0.001,
        "total_cost_usd": 0.001, "total_router_overhead_cost_usd": 0.0,
        "total_router_overhead_ms": 0.0, "total_energy_kwh": None,
        "total_co2e_kg": None, "escalation_count": 0,
        "trigger_distribution": {}, "benchmark_noise": False,
        "benchmark_noise_category": None, "artifacts_dir": "",
        # B-732 sentinels (all 7 required per archive-not-in-paper-scope)
        "evaluator_authority_mode": "post_B545_vwa_score_only",
        "reward_override_applied": False,
        "wallclock_start": "2026-05-17T15:00:00+00:00",
        "wallclock_end": "2026-05-17T15:01:00+00:00",
        "resume_fingerprint": "deadbeefcafef00d",
        "needs_reevaluation": False,
        "trajectory_incomplete": False,
    }


def test_b732_valid_episode_summary_with_all_sentinels_passes():
    from p79.experiment.types import validate_episode_summary_v2
    validate_episode_summary_v2(_minimal_valid_episode_summary())


@pytest.mark.parametrize("missing_field", [
    "evaluator_authority_mode", "reward_override_applied",
    "wallclock_start", "wallclock_end", "resume_fingerprint",
    "needs_reevaluation", "trajectory_incomplete",
])
def test_b732_episode_summary_missing_sentinel_raises(missing_field):
    """Each of the 7 paper-grade episode optional keys MUST be present at
    write boundary (value MAY be None for sentinel str fields)."""
    from p79.experiment.types import validate_episode_summary_v2
    summary = _minimal_valid_episode_summary()
    del summary[missing_field]
    with pytest.raises(ValueError, match=f"missing paper-grade.*{missing_field}"):
        validate_episode_summary_v2(summary)


def test_b732_episode_summary_evaluator_authority_string_coercion_attack_caught():
    """String `"false"` for `reward_override_applied` would be truthy-cast
    under `bool(...)` → silent cohort isolation bypass. Value-type check
    catches this at write boundary."""
    from p79.experiment.types import validate_episode_summary_v2
    summary = _minimal_valid_episode_summary()
    summary["reward_override_applied"] = "false"  # STRING attack
    with pytest.raises(ValueError, match=r"reward_override_applied.*expected.*bool"):
        validate_episode_summary_v2(summary)


# ─── B-735 score bool subclass-of-int exclude ───────────────────────────────
def test_b735_episode_summary_score_true_rejected():
    """Python bool subclasses int — `isinstance(True, int) == True`. Pre-fix
    `score=True` would pass validator. Now: bool is explicitly excluded
    from the (int, float) acceptance for score."""
    from p79.experiment.types import validate_episode_summary_v2
    summary = _minimal_valid_episode_summary()
    summary["score"] = True  # bool literal → silent type-coerce attack
    with pytest.raises(ValueError, match=r"score.*expected.*\(NOT bool\)"):
        validate_episode_summary_v2(summary)


def test_b735_episode_summary_steps_true_rejected():
    """Same bool subclass attack on steps/task_id."""
    from p79.experiment.types import validate_episode_summary_v2
    summary = _minimal_valid_episode_summary()
    summary["steps"] = True
    with pytest.raises(ValueError, match=r"steps.*NOT bool"):
        validate_episode_summary_v2(summary)


# ─── B-739 episode full field-type coverage ─────────────────────────────────
def test_b739_episode_summary_total_cost_usd_string_caught():
    """Pre-fix only 4 fields type-checked; total_cost_usd string would slip
    past."""
    from p79.experiment.types import validate_episode_summary_v2
    summary = _minimal_valid_episode_summary()
    summary["total_cost_usd"] = "0.001"  # STRING attack
    with pytest.raises(ValueError, match=r"total_cost_usd.*expected"):
        validate_episode_summary_v2(summary)


def test_b739_episode_summary_total_latency_ms_string_caught():
    from p79.experiment.types import validate_episode_summary_v2
    summary = _minimal_valid_episode_summary()
    summary["total_latency_ms"] = "100"
    with pytest.raises(ValueError, match=r"total_latency_ms.*expected"):
        validate_episode_summary_v2(summary)


def test_b739_episode_summary_total_cost_usd_bool_caught():
    """B-735 extension: cost fields also exclude bool from (int, float)."""
    from p79.experiment.types import validate_episode_summary_v2
    summary = _minimal_valid_episode_summary()
    summary["total_cost_usd"] = True
    with pytest.raises(ValueError, match=r"total_cost_usd.*NOT bool"):
        validate_episode_summary_v2(summary)


# ─── B-733 _validate_against_summary tri-state Optional[bool] ───────────────
def test_b733_validate_against_summary_returns_none_on_missing_summary(tmp_path):
    """Pre-fix returned hard False (= 'checked & OK') on missing summary;
    that bundled into `_JSONL_INTEGRITY_LOG` as silent OK signal. Now: None
    (= 'not checked'), matching B-293 tri-state contract."""
    from p79.experiment.io_utils import _validate_against_summary
    jsonl = tmp_path / "test.jsonl"
    summary = tmp_path / "missing_summary.json"  # not created
    jsonl.write_text(json.dumps({"step_idx": 0, "task_id": 1}) + "\n")
    result = _validate_against_summary(jsonl, [{"step_idx": 0}], summary)
    assert result is None, f"Expected None tri-state, got {result!r}"


def test_b733_read_jsonl_dedup_lenient_logs_summary_missing_telemetry(tmp_path):
    """Lenient mode preserves data return + log telemetry includes
    `summary_missing=True` and `summary_identity_mismatch=None`."""
    from p79.experiment.io_utils import _JSONL_INTEGRITY_LOG, read_jsonl_dedup
    jsonl = tmp_path / "test.jsonl"
    summary = tmp_path / "missing.json"
    jsonl.write_text(json.dumps({"step_idx": 0, "task_id": 1}) + "\n")
    _JSONL_INTEGRITY_LOG.clear()
    lines = read_jsonl_dedup(jsonl, summary_path=summary)
    assert len(lines) == 1
    entry = _JSONL_INTEGRITY_LOG[-1]
    assert entry["summary_missing"] is True
    assert entry["summary_identity_mismatch"] is None


def test_b733_read_jsonl_dedup_strict_raises_on_missing_summary(tmp_path):
    """Strict mode: orphan step JSONL (= step fsync landed but summary atomic
    replace did not before crash) fails-loud — this is exactly the paper-
    grade case strict_identity was supposed to protect."""
    from p79.experiment.io_utils import read_jsonl_dedup
    jsonl = tmp_path / "test.jsonl"
    summary = tmp_path / "missing.json"
    jsonl.write_text(json.dumps({"step_idx": 0, "task_id": 1}) + "\n")
    with pytest.raises(ValueError, match=r"B-733|summary file missing"):
        read_jsonl_dedup(jsonl, summary_path=summary, strict_identity=True)


# ─── B-734 needs_reevaluation truthy-coercion attack ────────────────────────
def test_b734_load_episode_summary_strict_rejects_string_needs_reevaluation(tmp_path):
    """`bool("false") == True` would silently quarantine a valid episode.
    Strict mode raises with explicit type-mismatch."""
    from p79.experiment.io_utils import load_episode_summary_strict
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({
        "schema_version": "2.0", "task_id": 1, "success": True,
        "needs_reevaluation": "false",  # STRING attack
    }))
    with pytest.raises(ValueError, match=r"B-734|needs_reevaluation type mismatch"):
        load_episode_summary_strict(summary, mode="strict")


def test_b734_load_episode_summary_lenient_returns_none_and_warns(tmp_path, caplog):
    from p79.experiment.io_utils import load_episode_summary_strict
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({
        "schema_version": "2.0", "task_id": 1, "success": True,
        "needs_reevaluation": "false",
    }))
    result = load_episode_summary_strict(summary, mode="lenient")
    assert result is None  # lenient mode skips poisoned row


def test_b734_load_episode_summary_strict_accepts_bool_needs_reevaluation(tmp_path):
    """Sanity: actual bool value passes."""
    from p79.experiment.io_utils import load_episode_summary_strict
    summary = tmp_path / "s.json"
    summary.write_text(json.dumps({
        "schema_version": "2.0", "task_id": 1, "success": True,
        "needs_reevaluation": False,
    }))
    result = load_episode_summary_strict(summary, mode="strict")
    assert result is not None
    assert result["needs_reevaluation"] is False


# ─── B-737 dedup_restart_lines string-"0" type guard ────────────────────────
def test_b737_dedup_string_step_idx_does_not_trigger_boundary():
    """JSON literal `"step_idx": "0"` (string) must NOT count as a restart
    boundary (would falsely segment legitimate data)."""
    from p79.experiment.io_utils import dedup_restart_lines
    lines = [
        {"step_idx": 0, "data": "first_run"},
        {"step_idx": 1, "data": "first_run"},
        {"step_idx": "0", "data": "string_zero"},  # not an int — not a boundary
        {"step_idx": "1", "data": "string_one"},
    ]
    seg = dedup_restart_lines(lines)
    assert len(seg) == 4  # all kept (string "0" doesn't trigger boundary)


def test_b737_dedup_int_step_idx_still_triggers_boundary():
    """Regression: legitimate int 0 still triggers restart boundary."""
    from p79.experiment.io_utils import dedup_restart_lines
    lines = [
        {"step_idx": 0, "data": "old"}, {"step_idx": 1, "data": "old"},
        {"step_idx": 0, "data": "new"}, {"step_idx": 1, "data": "new"},
    ]
    seg = dedup_restart_lines(lines)
    assert len(seg) == 2
    assert seg[0]["data"] == "new"


def test_b737_dedup_bool_step_idx_excluded():
    """Python `True == 1` and `False == 0` — bool subclasses int. Must not
    be treated as int step_idx (rare but pathological)."""
    from p79.experiment.io_utils import dedup_restart_lines
    lines = [
        {"step_idx": 0, "data": "real"},
        {"step_idx": 1, "data": "real"},
        {"step_idx": False, "data": "bool"},  # False == 0 but not int per fix
        {"step_idx": True, "data": "bool"},
    ]
    seg = dedup_restart_lines(lines)
    # bool excluded → no second boundary → all 4 returned
    assert len(seg) == 4


def test_b737_read_jsonl_dedup_strict_raises_on_string_step_idx(tmp_path):
    """Strict mode: non-int step_idx in segment is paper-grade fail-loud
    (boundary detection structurally compromised)."""
    from p79.experiment.io_utils import read_jsonl_dedup
    jsonl = tmp_path / "test.jsonl"
    summary = tmp_path / "test_summary.json"
    summary.write_text(json.dumps({
        "schema_version": "2.0", "run_id": "r1", "condition_id": "c1",
        "seed": 42, "benchmark_site": "classifieds", "task_id": 1,
        "steps": 2,
    }))
    jsonl.write_text(
        json.dumps({"step_idx": "0", "task_id": 1, "run_id": "r1"}) + "\n" +
        json.dumps({"step_idx": "1", "task_id": 1, "run_id": "r1"}) + "\n"
    )
    with pytest.raises(ValueError, match=r"B-737|non-int step_idx"):
        read_jsonl_dedup(jsonl, summary_path=summary, strict_identity=True)


# ─── B-738 _validate_against_summary multi-row check ────────────────────────
def test_b738_validate_against_summary_catches_tail_row_mismatch(tmp_path):
    """Pre-fix only `last_segment[0]` was identity-checked → poisoned tail
    survived. Now `strict_rows=True` loops over every row."""
    from p79.experiment.io_utils import _validate_against_summary
    jsonl = tmp_path / "test.jsonl"
    summary_path = tmp_path / "summary.json"
    summary_payload = {"task_id": 1, "run_id": "r1", "condition_id": "c1",
                       "schema_version": "2.0", "seed": 42,
                       "benchmark_site": "classifieds", "steps": 2}
    summary_path.write_text(json.dumps(summary_payload))
    segment = [
        {"step_idx": 0, "task_id": 1, "run_id": "r1", "condition_id": "c1",
         "schema_version": "2.0", "seed": 42, "benchmark_site": "classifieds"},
        {"step_idx": 1, "task_id": 999, "run_id": "r999",  # POISONED tail
         "condition_id": "c1", "schema_version": "2.0", "seed": 42,
         "benchmark_site": "classifieds"},
    ]
    # Default (strict_rows=False): only first row checked, mismatch not seen
    result_lenient = _validate_against_summary(jsonl, segment, summary_path,
                                                strict_rows=False)
    assert result_lenient is False  # only first row checked, matched
    # strict_rows=True: every row checked, tail mismatch detected
    result_strict = _validate_against_summary(jsonl, segment, summary_path,
                                               strict_rows=True)
    assert result_strict is True  # mismatch surfaces


# ─── B-736 fcntl flock concurrent-append protection ─────────────────────────
def _flock_writer_worker(td: str, worker_id: int, n_events: int) -> None:
    """Helper for B-736 mp test — must be top-level for pickling."""
    from p79.experiment.logger_v2 import LoggerV2
    logger = LoggerV2(Path(td))
    # ~6KB payload, well over PIPE_BUF=4096 to force the race window
    payload = "X" * 6000
    for i in range(n_events):
        logger.log_trajectory_event(
            event_type=f"w{worker_id}_evt_{i}",
            task_index=i,
            metadata={"worker": worker_id, "iter": i, "payload": payload},
        )


def test_b736_log_trajectory_event_no_torn_writes_under_concurrency():
    """4 concurrent writers × 10 events each × 6KB payload (> PIPE_BUF=4096)
    must produce 40 valid JSONL lines with no torn writes."""
    with tempfile.TemporaryDirectory() as td:
        procs = []
        for w in range(4):
            p = mp.Process(target=_flock_writer_worker, args=(td, w, 10))
            p.start()
            procs.append(p)
        for p in procs:
            p.join(timeout=30)
            assert not p.is_alive(), "worker hung — flock deadlock?"
        target = Path(td) / "trajectory_events.jsonl"
        valid = 0
        torn = 0
        with open(target, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)
                    valid += 1
                except json.JSONDecodeError:
                    torn += 1
        assert valid == 40, f"expected 40 valid lines, got {valid}"
        assert torn == 0, f"expected 0 torn writes, got {torn}"
