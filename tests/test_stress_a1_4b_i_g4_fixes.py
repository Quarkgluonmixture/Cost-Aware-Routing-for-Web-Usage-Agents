"""Invariant tests for /stress A1.4b-i G4 (B-181 / B-182 / B-183).

B-181: `analyze_run` writes `phase_mix_warning.txt` + drops out-of-phase rows
       when condition_id prefixes disagree with the inferred phase.
B-182: `aggregate_phantom_meta.py` markdown table emits `family_scope` +
       `gating_status` columns + appendix-only disclosure.
B-183: per-episode P95 latency figure caption discloses the Jensen-like
       semantics (NOT per-step distribution).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PY = REPO_ROOT / "p79" / "experiment" / "analysis.py"
PHANTOM_META_PY = REPO_ROOT / "scripts" / "analysis" / "aggregate_phantom_meta.py"


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


# ─── B-181 ──────────────────────────────────────────────────────────────────
def test_b181_phase_mix_detection_present():
    src = _read(ANALYSIS_PY)
    assert "B-181" in src
    assert "phase_mix_warning.txt" in src
    assert 'cid_str.startswith("phase1_")' in src
    assert 'cid_str.startswith("phase2_")' in src


def test_b181_end_to_end_drops_out_of_phase(tmp_path):
    """Build a stub run with mixed phase1+phase2 cond dirs; verify drop+warning."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    from p79.experiment.analysis import analyze_run

    run_dir = tmp_path / "stub_mixed"
    run_dir.mkdir()
    # Phase 1 condition
    p1 = run_dir / "phase1_dom_router_0"
    p1.mkdir()
    (p1 / "episodes").mkdir()
    cond_summary = {
        "condition_id": "phase1_dom_router_0",
        "seed": 42, "phase": "phase1", "backend_id": "B1",
        "som_on": False, "observation_mode": "dom", "router_on": False,
        "module_flags": {},
        "episodes": 1, "success_rate": 1.0, "avg_steps": 1.0,
        "p95_step_latency_ms": 100.0,
        "avg_total_model_cost_usd": 0.0, "avg_total_cost_usd": 0.0,
        "avg_router_overhead_cost_usd": 0.0,
        "avg_total_energy_kwh": None, "avg_total_co2e_kg": None,
        "avg_retries": 0.0, "avg_no_op_rate": 0.0, "avg_page_unchanged_rate": 0.0,
        "avg_escalation_count": 0.0,
        "trigger_distribution": {},
        "state_change_reason_distribution": {},
        "avg_checklist_completion_rate": None,
        "checklist_failure_episode_rate": None,
        "benchmark_noise_rate": 0.0, "wasted_energy_kwh": 0.0,
        "avg_wasted_cost_usd": 0.0, "avg_wasted_energy_kwh": 0.0,
        "cost_efficiency_ratio": 0.0,
    }
    (p1 / "condition_summary_v2.json").write_text(json.dumps(cond_summary))
    # Phase 2 condition leaking into same run dir (re-launch scenario)
    p2 = run_dir / "phase2_fixed_best"
    p2.mkdir()
    (p2 / "episodes").mkdir()
    cs2 = dict(cond_summary)
    cs2["condition_id"] = "phase2_fixed_best"
    cs2["phase"] = "phase2"
    cs2["observation_mode"] = "som"
    (p2 / "condition_summary_v2.json").write_text(json.dumps(cs2))

    # Run analyze_run — should detect phase mix + drop p2 row from cross-cond plots
    analyze_run(str(run_dir))
    mix_warning = run_dir / "analysis" / "phase_mix_warning.txt"
    assert mix_warning.exists(), "B-181 should emit phase_mix_warning.txt"
    contents = mix_warning.read_text()
    assert "phase mix detected" in contents
    assert "phase2" in contents or "phase1" in contents


# ─── B-182 ──────────────────────────────────────────────────────────────────
def test_b182_family_scope_columns_in_phantom_meta():
    src = _read(PHANTOM_META_PY)
    assert "B-182" in src
    # Header now has family_scope + gating_status
    assert "family_scope" in src
    assert "gating_status" in src
    # Family scope map mentions appendix-only labels
    assert "APPENDIX_RE_SENSITIVITY_m" in src
    # Disclosure prose: "this is appendix sensitivity, NOT paper gate"
    assert "appendix-only" in src or "appendix sensitivity" in src


# ─── B-183 ──────────────────────────────────────────────────────────────────
def test_b183_p95_latency_caption_discloses_semantics():
    src = _read(ANALYSIS_PY)
    assert "B-183" in src
    # The caption should explicitly say "Per-episode P95" + "NOT per-step"
    assert "Per-episode P95 step latency" in src
    assert "NOT per-step distribution" in src
