"""Invariant tests for /stress A1.4b-i G3 (B-179 synth canonical + B-180 JSONL identity).

B-179: `_synthesize_condition_summary` delegates aggregation to
       `aggregate_condition_metrics` so synth + complete summaries share schema.
       `_synthesized` flag visually distinguishes partial bars in `_plot_phase1`.
B-180: `read_jsonl_dedup(path, summary_path=...)` warns on identity tuple
       mismatch or step-count divergence between JSONL segment and summary.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PY = REPO_ROOT / "p79" / "experiment" / "analysis.py"
IO_UTILS_PY = REPO_ROOT / "p79" / "experiment" / "io_utils.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ─── B-179 ──────────────────────────────────────────────────────────────────
def test_b179_synth_delegates_to_aggregate_condition_metrics():
    src = _read(ANALYSIS_PY)
    # The new synth function must import + call aggregate_condition_metrics
    assert "from p79.experiment.metrics import aggregate_condition_metrics" in src
    assert "canonical = aggregate_condition_metrics(ep_summaries)" in src
    # No more hand-aggregation with `or 0` silent None→0 pattern
    assert 'costs = [e.get("total_cost_usd", 0) or 0 for e in ep_summaries]' not in src
    assert 'model_costs = [e.get("total_model_cost_usd", 0) or 0' not in src
    # Hard-zero `avg_router_overhead_cost_usd: 0.0` line gone from synth body
    assert '"avg_router_overhead_cost_usd": 0.0,' not in src


def test_b179_synth_carries_canonical_schema():
    """Synthetic call: synth payload must contain every key the canonical aggregator produces."""
    pytest.importorskip("pandas")
    from p79.experiment.analysis import _synthesize_condition_summary
    from p79.experiment.metrics import aggregate_condition_metrics

    ep_summaries = [
        {
            "success": True, "steps": 10, "total_cost_usd": 0.05,
            "total_model_cost_usd": 0.04, "total_router_overhead_cost_usd": 0.0,
            "total_obs_prepare_cost_usd": 0.01,
            "total_input_cost_usd": 0.02, "total_output_cost_usd": 0.02,
            "total_router_overhead_ms": 0.0, "total_latency_ms": 1000.0,
            "total_latency_minus_retry_ms": 1000.0,
            "total_energy_kwh": None, "total_co2e_kg": None,
            "p95_step_latency_ms": 100.0, "retries": 0,
            "no_op_rate": 0.0, "page_unchanged_rate": 0.0,
            "escalation_count": 0, "trigger_distribution": {},
            "state_change_reason_distribution": {},
            "benchmark_noise": False, "busy_wait_total_ms": 0.0,
            "wasted_cost_usd": 0.0, "wasted_energy_kwh": 0.0,
        },
    ]

    canonical_keys = set(aggregate_condition_metrics(ep_summaries).keys())

    # Build a minimal meta file in a temp dir
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        meta = td_path / "condition_meta.json"
        meta.write_text(json.dumps({
            "condition_id": "test_cond",
            "seed": 42, "phase": "phase1", "backend_id": "B1",
            "som_on": False, "observation_mode": "dom", "router_on": False,
            "modules": {},
        }))

        result = _synthesize_condition_summary(td_path, ep_summaries)

    # Every canonical key is present in synth payload
    missing = canonical_keys - set(result.keys())
    assert not missing, f"synth missing canonical keys: {missing}"
    # _synthesized flag is True
    assert result.get("_synthesized") is True
    # Specific keys that pre-fix synth lacked (per codex B6 catch)
    for k in ["avg_total_latency_ms", "avg_obs_prepare_cost_usd",
              "avg_input_cost_usd", "avg_output_cost_usd",
              "avg_busy_wait_total_ms", "energy_partial_episode_count"]:
        assert k in result, f"synth payload still missing {k}"


def test_b179_plot_phase1_hatches_partial_bars():
    """Source-level check: `_plot_phase1` honors `_synthesized` via hatch."""
    src = _read(ANALYSIS_PY)
    assert "B-179" in src
    assert 'hatch=["//"' in src
    assert "partial conditions hatched" in src


# ─── B-180 ──────────────────────────────────────────────────────────────────
def test_b180_signature_accepts_summary_path():
    """API check: read_jsonl_dedup now has `summary_path=` kwarg."""
    import inspect
    from p79.experiment.io_utils import read_jsonl_dedup
    sig = inspect.signature(read_jsonl_dedup)
    assert "summary_path" in sig.parameters


def test_b180_summary_match_no_warning(tmp_path, caplog):
    """When JSONL last segment matches summary identity, no B-180 warning."""
    from p79.experiment.io_utils import read_jsonl_dedup
    jsonl = tmp_path / "1_steps_v2.jsonl"
    summary = tmp_path / "1_summary_v2.json"

    rows = [
        {"step_idx": 0, "schema_version": "2.0", "run_id": "r1", "condition_id": "c1",
         "seed": 42, "benchmark_site": "classifieds", "task_id": 1},
        {"step_idx": 1, "schema_version": "2.0", "run_id": "r1", "condition_id": "c1",
         "seed": 42, "benchmark_site": "classifieds", "task_id": 1},
    ]
    jsonl.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary.write_text(json.dumps({
        "schema_version": "2.0", "run_id": "r1", "condition_id": "c1",
        "seed": 42, "benchmark_site": "classifieds", "task_id": 1,
        "steps": 2,
    }))

    with caplog.at_level(logging.WARNING, logger="p79.experiment.io_utils"):
        result = read_jsonl_dedup(jsonl, summary_path=summary)
    assert len(result) == 2
    b180_warnings = [r for r in caplog.records if "B-180" in r.message]
    assert not b180_warnings, "no warning expected for matching identity"


def test_b180_identity_mismatch_warns(tmp_path, caplog):
    """When run_id differs between JSONL + summary, B-180 warning fires."""
    from p79.experiment.io_utils import read_jsonl_dedup
    jsonl = tmp_path / "1_steps_v2.jsonl"
    summary = tmp_path / "1_summary_v2.json"

    jsonl.write_text(json.dumps({
        "step_idx": 0, "schema_version": "2.0", "run_id": "NEW_RUN",
        "condition_id": "c1", "seed": 42, "benchmark_site": "classifieds",
        "task_id": 1,
    }) + "\n")
    summary.write_text(json.dumps({
        "schema_version": "2.0", "run_id": "OLD_RUN", "condition_id": "c1",
        "seed": 42, "benchmark_site": "classifieds", "task_id": 1, "steps": 1,
    }))

    with caplog.at_level(logging.WARNING, logger="p79.experiment.io_utils"):
        read_jsonl_dedup(jsonl, summary_path=summary)
    msg = " ".join(r.message for r in caplog.records)
    assert "B-180 identity mismatch" in msg
    assert "run_id" in msg


def test_b180_step_count_mismatch_warns(tmp_path, caplog):
    """Steps in JSONL segment != summary.steps → warning."""
    from p79.experiment.io_utils import read_jsonl_dedup
    jsonl = tmp_path / "1_steps_v2.jsonl"
    summary = tmp_path / "1_summary_v2.json"

    jsonl.write_text(json.dumps({
        "step_idx": 0, "schema_version": "2.0", "run_id": "r1",
        "condition_id": "c1", "seed": 42, "benchmark_site": "classifieds",
        "task_id": 1,
    }) + "\n")
    summary.write_text(json.dumps({
        "schema_version": "2.0", "run_id": "r1", "condition_id": "c1",
        "seed": 42, "benchmark_site": "classifieds", "task_id": 1,
        "steps": 10,  # 10 != 1 in JSONL
    }))

    with caplog.at_level(logging.WARNING, logger="p79.experiment.io_utils"):
        read_jsonl_dedup(jsonl, summary_path=summary)
    msg = " ".join(r.message for r in caplog.records)
    assert "B-180 step count mismatch" in msg


def test_b180_missing_summary_silent(tmp_path):
    """If summary_path is None or file absent, no error + no validation."""
    from p79.experiment.io_utils import read_jsonl_dedup
    jsonl = tmp_path / "1_steps_v2.jsonl"
    jsonl.write_text(json.dumps({"step_idx": 0}) + "\n")
    assert len(read_jsonl_dedup(jsonl)) == 1
    assert len(read_jsonl_dedup(jsonl, summary_path=tmp_path / "missing.json")) == 1
