"""Regression locks for toolchain repair Chunk 1 (F1-F5, 2026-07-14)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import aggregate_phase1_full_prereg_decision as full
from scripts.analysis.aggregate_phase1_prereg_gate import (
    SIX_MODES,
    _cell_drop_one_theta_se,
    build_gate,
)
from scripts.analysis.aggregate_sr_fp_per_mode import aggregate_cell
from scripts.analysis.lib.canonical_cells import PHASE_1A_PLANNED_CELLS
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids


def _write_summary(path: Path, task_id: int, success: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "schema_version": "2.0",
        "task_id": task_id,
        "success": success,
    }), encoding="utf-8")


def _six_mode_cell(tmp_path: Path, observed_by_mode: dict[str, set[int]]) -> dict:
    modes = {}
    for mode in SIX_MODES:
        ep_dir = tmp_path / mode
        for tid in observed_by_mode.get(mode, set()):
            _write_summary(ep_dir / f"task_{tid}_summary_v2.json", tid, tid % 3 == 0)
        modes[mode] = ep_dir
    return {"baseline": "B0", "site": "classifieds", "modes": modes}


def test_f1_status_schema_is_orthogonal_in_both_producers(monkeypatch):
    """COMPLETE is exact planned scope; verdict is separate from legacy status."""
    boot = np.full(1000, 5.0, dtype=np.float32)

    def fake_h1(cell, **_kwargs):
        return {
            "baseline": cell["baseline"], "site": cell["site"],
            "complete_exact": True, "expected_n": 100,
            "observed_n": {m: 100 for m in SIX_MODES},
            "missing_ids": {m: [] for m in SIX_MODES},
            "extra_ids": {m: [] for m in SIX_MODES},
            "task_set_sha256": "synthetic", "n_tasks": 100,
            "theta_pp": 5.0, "se_pp": 1.0,
            "ci95_lo_pp": 3.0, "ci95_hi_pp": 7.0,
            "boot_pp": boot, "oracle_6_pp": 20.0,
            "oracle_5_no_psom_pp": 15.0, "n_psom_only": 5,
        }

    monkeypatch.setattr(full, "_cell_drop_one_theta_se", fake_h1)
    monkeypatch.setattr(full, "_load_cell_per_task", lambda _cell, **_kwargs: {})
    cells = [
        {"site": site, "baseline": baseline, "modes": {}}
        for site, baseline in PHASE_1A_PLANNED_CELLS
    ]
    decision = full.build_full_decision(cells)
    assert decision["analysis_status"] == "COMPLETE"
    assert decision["h1_verdict"] == "PASS"
    assert decision["gate_status"] == "PASS"  # legacy field retained unchanged

    partial = build_gate([])
    assert partial["analysis_status"] == "INSUFFICIENT"
    assert partial["h1_verdict_normal_approx_transparency"] == "NOT_EVALUATED"
    assert "h1_verdict" not in partial
    assert partial["gate_status"] == "INSUFFICIENT_DATA"


def test_f2_h1_missing_and_extra_ids_fail_closed_with_persisted_diff(tmp_path):
    expected = set(range(60))
    observed = {mode: set(expected) for mode in SIX_MODES}
    observed["DOM"] = (expected - {7}) | {999}
    cell = _six_mode_cell(tmp_path, observed)

    result = _cell_drop_one_theta_se(cell, expected_ids=expected)
    assert result["complete_exact"] is False
    assert result["expected_n"] == 60
    assert result["observed_n"]["DOM"] == 60
    assert result["missing_ids"]["DOM"] == [7]
    assert result["extra_ids"]["DOM"] == [999]
    assert len(result["task_set_sha256"]) == 64

    payload = build_gate([cell], expected_ids_by_site={"classifieds": expected})
    assert payload["per_cell"] == []
    assert payload["skipped_cells"][0]["missing_ids"]["DOM"] == [7]
    assert payload["analysis_status"] == "INSUFFICIENT"


def test_f2_canonical_helper_matches_locked_operational_counts():
    cls_ids, cls_sha = expected_scored_ids("classifieds")
    red_ids, red_sha = expected_scored_ids("reddit")
    assert len(cls_ids) == 224
    assert len(red_ids) == 205
    assert len(cls_sha) == len(red_sha) == 64
    assert cls_sha != red_sha


def test_f3_sr_exact_set_and_fixed_canonical_denominator(tmp_path):
    ep_dir = tmp_path / "episodes"
    _write_summary(ep_dir / "task_0_summary_v2.json", 0, True)
    _write_summary(ep_dir / "task_1_summary_v2.json", 1, False)
    _write_summary(ep_dir / "task_99_summary_v2.json", 99, True)

    row = aggregate_cell(
        "B0", "classifieds", "DOM", ep_dir, expected_ids={0, 1, 2}
    )
    assert row["complete"] is False
    assert row["missing_ids"] == [2]
    assert row["extra_ids"] == [99]
    assert row["observed_n"] == 3
    assert row["sr_denominator_n"] == 3
    assert row["n_success"] == 1  # extra task 99 cannot enter the numerator
    assert row["sr_pct"] == pytest.approx(100.0 / 3.0, abs=1e-6)
    assert len(row["task_set_sha256"]) == 64


def test_f4_h3_pool_keeps_all_six_cells_and_only_floors_zero_se():
    thetas = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    ses = [0.0, 0.5, 1.0, 1.0, 1.0, 1.0]
    uniques = [0, 1, 2, 3, 4, 5]
    rows = []
    jitter = np.linspace(-0.2, 0.2, 1000, dtype=np.float32)
    for theta, se, n_unique in zip(thetas, ses, uniques):
        rows.append({
            "unique_count_pp": theta,
            "se_pp": se,
            "n_unique": n_unique,
            "cell_pass": n_unique >= 2,
            "boot_pp": theta + jitter,
        })

    pooled = full._h3_axis_pooled_fe(rows, "axis1")
    assert pooled["k_cells_input"] == 6
    assert pooled["k_cells"] == 6
    assert pooled["n_noise_floor_cells"] == 2
    assert pooled["n_noise_floor_cells_skipped"] == 0
    assert pooled["n_zero_se_floored_cells"] == 1
    # Correct weights: SEs [1.0(floored), .5, 1, 1, 1, 1] -> theta_FE=2.0.
    # The retired <0.68 floor would also floor .5 and incorrectly give 2.5.
    assert pooled["theta_FE_pp"] == pytest.approx(2.0)
    assert pooled["analysis_status"] == "COMPLETE"
    assert pooled["axis_verdict"] in {"PASS", "FAIL"}

    interim = full._h3_axis_pooled_fe(rows[:3], "axis1")
    assert interim["k_cells_input"] == 3
    assert interim["analysis_status"] == "PARTIAL"
    assert interim["axis_verdict"] == "NOT_EVALUATED"
    assert interim["passed"] is None
    assert interim["theta_FE_pp"] is not None


def _writer_payload() -> dict:
    return {
        "per_cell": [],
        "skipped_cells": [],
        "gate_status": "PARTIAL_DATA",
        "gate_status_reason": "synthetic interim",
        "analysis_status": "PARTIAL",
        "h1_verdict": "NOT_EVALUATED",
        "h2a_margin_pct": 20.0,
        "pooled_h1_fe": {
            "k_cells": 2, "theta_FE_pp": 2.0, "se_FE_pp": 1.0,
            "ci95_FE_lo_pp": 0.04, "ci95_FE_hi_pp": 3.96,
            "delta_pp": 1.0, "z_one_sided": 1.0,
            "p_one_sided": 0.158655, "gate_passed": False,
        },
        "h3_axis1_pooled_fe": {"n_noise_skipped": 2, "k_after_filter": 0},
        "h3_axis2_pooled_fe": {"n_noise_skipped": 1, "k_after_filter": 1},
    }


def test_f5_insufficient_writers_emit_blank_numeric_and_md_does_not_crash(tmp_path):
    payload = _writer_payload()
    csv_path = tmp_path / "decision.csv"
    md_path = tmp_path / "decision.md"
    full.write_csv(payload, csv_path)
    full.write_md(payload, md_path)

    pooled = list(csv.DictReader(csv_path.open(encoding="utf-8")))[0]
    assert pooled["h3a_unique_count_pp"] == ""
    assert pooled["h3a_se_pp"] == ""
    assert pooled["h3b_unique_count_pp"] == ""
    assert "NOT_EVALUATED" in md_path.read_text(encoding="utf-8")
    assert not list(tmp_path.glob("*.tmp"))


def test_f5_three_output_transaction_does_not_partially_replace(monkeypatch, tmp_path):
    destinations = [tmp_path / name for name in ("decision.csv", "decision.json", "decision.md")]
    for path in destinations:
        path.write_text("OLD\n", encoding="utf-8")

    def fail_md(*_args, **_kwargs):
        raise KeyError("synthetic markdown schema failure")

    monkeypatch.setattr(full, "write_md", fail_md)
    with pytest.raises(KeyError):
        full.write_outputs_atomic(
            _writer_payload(), destinations[0], destinations[1], destinations[2]
        )
    assert [path.read_text(encoding="utf-8") for path in destinations] == ["OLD\n"] * 3
    assert not list(tmp_path.glob("*.staged"))
