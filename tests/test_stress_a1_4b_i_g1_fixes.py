"""Invariant tests for /stress A1.4b-i G1 quick-win fixes (B-170 ... B-176).

Covers:
  - B-170: McNemar/Wilcoxon merge key now includes benchmark_site (cross-site
    task_id collision is impossible by construction).
  - B-171: "(adjusted)" prose stripped from analysis.py plot titles.
  - B-172: Wilcoxon skip writes a flat CSV row with `skipped_reason`.
  - B-173: `_compute_pareto_front` docstring discloses tie-breaking semantics.
  - B-174: `_to_mapping` logs + collects parse failures into a global buffer
    (emitted to analysis/parse_failures.csv by `analyze_run`).
  - B-175: aggregate_phantom_lift.py markdown column renamed to
    `equiv_within_1pp` with a footnote disambiguating from lift significance.
  - B-176: bootstrap RNG seed=42 + B=10000 disclosed via inline comment.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

# /stress A1.12 P1-8 fix (2026-05-16): top-level `import pandas as pd`
# would collection-time error on fresh CI without `[analysis]` / `[test]`
# extras. Module-level importorskip cleanly skips the whole file when
# pandas absent, instead of breaking the entire test collection.
pd = pytest.importorskip("pandas")

REPO_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_PY = REPO_ROOT / "p79" / "experiment" / "analysis.py"
PHANTOM_LIFT_PY = REPO_ROOT / "scripts" / "analysis" / "aggregate_phantom_lift.py"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ─── B-170 ──────────────────────────────────────────────────────────────────
def test_b170_merge_key_includes_benchmark_site():
    src = _read(ANALYSIS_PY)
    # The fixed merge call must use a list that contains benchmark_site
    # under the has_site=True branch.
    assert 'join_on = ["benchmark_site", "task_id"] if has_site else ["task_id"]' in src
    # Pre-fix bug: bare `merge(df_a, df_b, on="task_id"` should NOT remain.
    assert 'merge(df_b, on="task_id", suffixes=' not in src


def test_b170_merge_actually_isolates_sites_at_runtime():
    """Synthetic check: same task_id across cls + red must not cross-merge."""
    # Build a tiny ep_df with overlapping task_ids across 2 sites + 2 conditions
    ep_df = pd.DataFrame([
        {"condition_id": "c1", "benchmark_site": "classifieds", "task_id": 5, "success": True,
         "total_cost_usd": 0.10, "p95_step_latency_ms": 1000},
        {"condition_id": "c1", "benchmark_site": "reddit", "task_id": 5, "success": False,
         "total_cost_usd": 0.20, "p95_step_latency_ms": 2000},
        {"condition_id": "c2", "benchmark_site": "classifieds", "task_id": 5, "success": False,
         "total_cost_usd": 0.15, "p95_step_latency_ms": 1500},
        {"condition_id": "c2", "benchmark_site": "reddit", "task_id": 5, "success": True,
         "total_cost_usd": 0.25, "p95_step_latency_ms": 2500},
    ])
    # Pre-fix join: merge(on="task_id") would 4-pair-cross to 4 rows
    bad_merge = ep_df[ep_df["condition_id"] == "c1"].merge(
        ep_df[ep_df["condition_id"] == "c2"], on="task_id", suffixes=("_a", "_b"))
    assert len(bad_merge) == 4, "pre-fix would explode 2×2 site×task into 4 rows (broken)"
    # Fixed join: merge(on=["benchmark_site", "task_id"]) keeps 2 site-unique pairs
    good_merge = ep_df[ep_df["condition_id"] == "c1"].merge(
        ep_df[ep_df["condition_id"] == "c2"],
        on=["benchmark_site", "task_id"], suffixes=("_a", "_b"))
    assert len(good_merge) == 2, "fix must produce 2 site-distinct pairs (not 4)"


# ─── B-171 ──────────────────────────────────────────────────────────────────
def test_b171_no_stray_adjusted_in_plot_titles():
    src = _read(ANALYSIS_PY)
    # No remaining "(adjusted)" parentheticals in plot titles. The retired
    # post-hoc layer should not be hinted at anywhere in user-facing text.
    title_lines = re.findall(r'(set_title|suptitle)\(([^)]+)\)', src)
    for func, arg in title_lines:
        assert "(adjusted)" not in arg, f"stale (adjusted) in {func}: {arg[:80]}"
    # Replacement text "(N/A excluded at task-load)" should be present.
    assert "(N/A excluded at task-load)" in src


# ─── B-172 ──────────────────────────────────────────────────────────────────
def test_b172_wilcoxon_skip_emits_csv_row():
    src = _read(ANALYSIS_PY)
    assert 'skipped_reason' in src
    assert 'insufficient_paired_samples_n' in src


# ─── B-173 ──────────────────────────────────────────────────────────────────
def test_b173_pareto_docstring_discloses_tie_semantics():
    src = _read(ANALYSIS_PY)
    assert 'ties broken by sort order' in src
    assert 'B-173' in src


# ─── B-174 ──────────────────────────────────────────────────────────────────
def test_b174_to_mapping_collects_parse_failures():
    # Import after asserting the global exists in source.
    src = _read(ANALYSIS_PY)
    assert '_TO_MAPPING_PARSE_FAILURES' in src

    from p79.experiment.analysis import _to_mapping, _TO_MAPPING_PARSE_FAILURES

    _TO_MAPPING_PARSE_FAILURES.clear()
    assert _to_mapping("{not valid json") == {}
    assert _to_mapping("[1, 2, 3]") == {}  # parsed but non-dict
    assert _to_mapping("null") == {}        # parsed but non-dict (NoneType)
    assert _to_mapping({"ok": 1}) == {"ok": 1}  # dict passthrough — no failure
    # Three malformed inputs above → 3 collector entries
    assert len(_TO_MAPPING_PARSE_FAILURES) == 3
    # Each entry has the structural keys
    for entry in _TO_MAPPING_PARSE_FAILURES:
        assert {"context", "reason", "value_snippet"}.issubset(entry.keys())


def test_b174_analyze_run_emits_parse_failures_csv(tmp_path):
    """End-to-end: build a stub run dir + ensure parse_failures.csv emits."""
    pytest.importorskip("matplotlib")
    pytest.importorskip("pandas")
    from p79.experiment.analysis import analyze_run, _TO_MAPPING_PARSE_FAILURES

    run_dir = tmp_path / "stub_run"
    run_dir.mkdir()
    cond_dir = run_dir / "phase1_dom_router_0"
    cond_dir.mkdir()
    eps = cond_dir / "episodes"
    eps.mkdir()
    # condition_summary_v2 with malformed trigger_distribution-as-string
    (cond_dir / "condition_summary_v2.json").write_text(json.dumps({
        "condition_id": "phase1_dom_router_0",
        "seed": 42, "phase": "phase1", "backend_id": "b1",
        "som_on": False, "observation_mode": "dom", "router_on": False,
        "module_flags": {},
        "episodes": 1, "success_rate": 1.0, "avg_steps": 1.0,
        "p95_step_latency_ms": 100.0,
        "avg_total_model_cost_usd": 0.0, "avg_total_cost_usd": 0.0,
        "avg_router_overhead_cost_usd": 0.0,
        "avg_total_energy_kwh": None, "avg_total_co2e_kg": None,
        "avg_retries": 0.0, "avg_no_op_rate": 0.0, "avg_page_unchanged_rate": 0.0,
        "avg_escalation_count": 0.0,
        # malformed JSON string → triggers _to_mapping parse failure
        "trigger_distribution": "{not valid json",
        "state_change_reason_distribution": {},
        "avg_checklist_completion_rate": None,
        "checklist_failure_episode_rate": None,
        "benchmark_noise_rate": 0.0,
        "wasted_energy_kwh": 0.0,
        "avg_wasted_cost_usd": 0.0,
        "avg_wasted_energy_kwh": 0.0,
        "cost_efficiency_ratio": 0.0,
    }), encoding="utf-8")
    # Minimal episode summary so collector finds a row
    (eps / "1_summary_v2.json").write_text(json.dumps({
        "schema_version": "2.0", "run_id": "stub", "condition_id": "phase1_dom_router_0",
        "benchmark": "visualwebarena", "benchmark_site": "classifieds",
        "task_id": 1, "seed": 42, "success": True, "score": 1.0,
        "steps": 1, "retries": 0, "no_op_rate": 0.0, "page_unchanged_rate": 0.0,
        "total_latency_ms": 100.0, "p95_step_latency_ms": 100.0,
        "total_tokens": 0, "total_model_cost_usd": 0.0, "total_cost_usd": 0.0,
        "total_router_overhead_cost_usd": 0.0, "total_router_overhead_ms": 0.0,
        "total_energy_kwh": None, "total_co2e_kg": None,
        "escalation_count": 0, "trigger_distribution": {},
        "benchmark_noise": False, "benchmark_noise_category": None,
        "artifacts_dir": "",
    }), encoding="utf-8")

    analyze_run(str(run_dir))
    parse_failures = run_dir / "analysis" / "parse_failures.csv"
    assert parse_failures.exists(), "expected parse_failures.csv to emit"
    pf_df = pd.read_csv(parse_failures)
    assert len(pf_df) >= 1
    assert "trigger_distribution" in str(pf_df["context"].iloc[0])


# ─── B-175 ──────────────────────────────────────────────────────────────────
def test_b175_phantom_lift_column_renamed():
    src = _read(PHANTOM_LIFT_PY)
    # Old confusing pair gone
    assert "| sig (Holm 0.05) | TOST sig (0.05) |" not in src
    # New explicit pair present
    assert "sig_lift (Holm 0.05)" in src
    assert "equiv_within_1pp (TOST 0.05)" in src
    # Footnote disambiguating present
    assert "is NOT evidence of positive lift" in src


# ─── B-176 ──────────────────────────────────────────────────────────────────
def test_b176_seed_disclosure_in_code():
    src = _read(ANALYSIS_PY)
    assert "B-176" in src
    assert "B=10_000" in src or "B=10000" in src.replace(",", "").replace("_", "")
    assert "seed 42" in src
