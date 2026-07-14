"""Invariant tests for B-184 — Phase 1 prereg gate producer.

Covers prereg H1 spec (preregistration.md:68-86 lock):
- 6-mode requirement (cells missing any mode are skipped)
- Per-task drop-one indicator semantics (oracle_6 ⊇ oracle_5_no_psom)
- FE inverse-variance pool arithmetic
- One-sided z superiority test at δ=1.0pp, α=0.05
- Bootstrap deterministic with seed=42, B=1000
- Gate status reflects data availability (PASS / FAIL / PARTIAL_DATA / INSUFFICIENT_DATA)
"""
from __future__ import annotations

import json
import math
import tempfile
import csv
from pathlib import Path

import numpy as np
import pytest

# Import target module
from scripts.analysis.aggregate_phase1_prereg_gate import (
    DELTA_PP,
    PREREG_B,
    PREREG_SEED,
    SIX_MODES,
    _cell_drop_one_theta_se,
    _fe_pool,
    _norm_cdf,
    build_gate,
    write_csv,
    write_json,
    write_md,
)


# ─── Synthetic cell builder helpers ────────────────────────────────────────
def _make_episodes_dir(tmp_root: Path, mode_name: str, success_task_ids: set[int],
                       observed_task_ids: set[int]) -> Path:
    """Create a fake episode dir matching `load_episode_summary_strict[lenient]`.

    /stress A1.12 P0-1 (2026-05-16): fixture must carry `schema_version` (str)
    in addition to `success` (bool) + `task_id` (int) — without these B-283
    strict-load downgrades the summary to corrupt-skip → θ collapses to 0.
    """
    ep_dir = tmp_root / mode_name
    ep_dir.mkdir(parents=True, exist_ok=True)
    for tid in observed_task_ids:
        d = ep_dir
        f = d / f"task_{tid}_summary_v2.json"
        f.write_text(json.dumps({
            "schema_version": "2.0",
            "task_id": tid,
            "success": tid in success_task_ids,
        }))
    return ep_dir


def _make_synthetic_cell(tmp: Path, baseline: str, site: str,
                         psom_only_count: int, common_count: int = 100) -> dict:
    """Build a synthetic cell with all 6 modes and a controlled `psom_only` count.

    Construction:
      tasks 0..common_count-1 observed in all 6 modes
      DOM / SoM / Vision / P-text / P-prompt each succeed on disjoint halves
        (so oracle_5_no_psom covers everyone except the last `psom_only_count` tasks)
      P-SoM succeeds on the last `psom_only_count` tasks (and ONLY those)
      → oracle_6 covers all tasks; oracle_5_no_psom covers (common_count - psom_only_count)
      → θ_pp = 100 * psom_only_count / common_count
    """
    tasks = set(range(common_count))
    no_psom_covered = set(range(common_count - psom_only_count))
    psom_only_tasks = set(range(common_count - psom_only_count, common_count))
    # Distribute no_psom_covered roughly evenly across 5 non-PSoM modes
    five = ["DOM", "SoM", "Vision", "P-text", "P-prompt"]
    mode_success = {m: set() for m in five}
    for i, tid in enumerate(sorted(no_psom_covered)):
        mode_success[five[i % 5]].add(tid)
    mode_success["P-SoM"] = psom_only_tasks
    cell_dir = tmp / f"{baseline}_{site}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    modes = {}
    for m in SIX_MODES:
        modes[m] = _make_episodes_dir(cell_dir, m, mode_success[m], tasks)
    return {
        "baseline": baseline,
        "site": site,
        "n_expected": common_count,
        "modes": modes,
    }


def _expected_for_cell(cell: dict) -> set[int]:
    return set(range(cell["n_expected"]))


def _expected_by_site(cells: list[dict]) -> dict[str, set[int]]:
    return {cell["site"]: _expected_for_cell(cell) for cell in cells}


# ─── _norm_cdf sanity ───────────────────────────────────────────────────────
def test_norm_cdf_canonical_values():
    """Φ at known points (no scipy needed)."""
    assert abs(_norm_cdf(0.0) - 0.5) < 1e-12
    assert abs(_norm_cdf(1.96) - 0.975) < 1e-3
    assert abs(_norm_cdf(-1.96) - 0.025) < 1e-3


# ─── _cell_drop_one_theta_se ────────────────────────────────────────────────
def test_cell_drop_one_theta_matches_psom_only_fraction(tmp_path):
    """θ_pp should equal `psom_only / common × 100`."""
    cell = _make_synthetic_cell(tmp_path, "B1", "classifieds",
                                psom_only_count=5, common_count=100)
    result = _cell_drop_one_theta_se(cell, expected_ids=_expected_for_cell(cell))
    assert result["complete_exact"] is True
    assert result["n_tasks"] == 100
    assert result["theta_pp"] == pytest.approx(5.0, abs=1e-9)
    assert result["n_psom_only"] == 5
    # oracle_6 covers everyone; oracle_5_no_psom covers 95
    assert result["oracle_6_pp"] == pytest.approx(100.0, abs=1e-9)
    assert result["oracle_5_no_psom_pp"] == pytest.approx(95.0, abs=1e-9)


def test_cell_drop_one_skipped_when_mode_missing(tmp_path):
    """If any of 6 modes is absent → explicit exact-set diagnostic."""
    cell = _make_synthetic_cell(tmp_path, "B1", "classifieds",
                                psom_only_count=5, common_count=100)
    # Remove P-prompt directory entirely
    import shutil
    shutil.rmtree(cell["modes"]["P-prompt"])
    cell["modes"].pop("P-prompt")
    result = _cell_drop_one_theta_se(cell, expected_ids=_expected_for_cell(cell))
    assert result["complete_exact"] is False
    assert result["observed_n"]["P-prompt"] == 0
    assert result["missing_ids"]["P-prompt"] == list(range(100))


def test_bootstrap_se_deterministic(tmp_path):
    """Same cell + same seed → identical SE every call (B-176 + prereg lock)."""
    cell = _make_synthetic_cell(tmp_path, "B1", "reddit",
                                psom_only_count=10, common_count=200)
    expected = _expected_for_cell(cell)
    r1 = _cell_drop_one_theta_se(cell, B=PREREG_B, seed=PREREG_SEED,
                                 expected_ids=expected)
    r2 = _cell_drop_one_theta_se(cell, B=PREREG_B, seed=PREREG_SEED,
                                 expected_ids=expected)
    assert r1["se_pp"] == r2["se_pp"]
    assert r1["ci95_lo_pp"] == r2["ci95_lo_pp"]
    assert r1["ci95_hi_pp"] == r2["ci95_hi_pp"]


def test_bootstrap_se_scales_with_n(tmp_path):
    """Larger n → smaller SE (variance reduction)."""
    cell_small = _make_synthetic_cell(tmp_path / "small", "B1", "classifieds",
                                      psom_only_count=5, common_count=50)
    cell_big = _make_synthetic_cell(tmp_path / "big", "B1", "reddit",
                                    psom_only_count=20, common_count=200)
    se_small = _cell_drop_one_theta_se(
        cell_small, expected_ids=_expected_for_cell(cell_small)
    )["se_pp"]
    se_big = _cell_drop_one_theta_se(
        cell_big, expected_ids=_expected_for_cell(cell_big)
    )["se_pp"]
    assert se_big < se_small, f"SE should shrink with n; got small={se_small} big={se_big}"


# ─── _fe_pool ────────────────────────────────────────────────────────────────
def test_fe_pool_arithmetic():
    """FE pool: equal SEs → simple mean; weighted SE = sqrt(1/Σw).

    SEs chosen ≥ 0.68pp so the AMENDMENT-03 Agresti-Coull floor does NOT fire —
    this test exercises the inverse-variance arithmetic, not the floor.
    """
    per_cell = [
        {"theta_pp": 2.0, "se_pp": 1.0},
        {"theta_pp": 3.0, "se_pp": 1.0},
        {"theta_pp": 4.0, "se_pp": 1.0},
    ]
    fe = _fe_pool(per_cell)
    assert fe is not None
    # Equal weights → arithmetic mean
    assert fe["theta_FE_pp"] == pytest.approx(3.0, abs=1e-9)
    # SE_FE = sqrt(1 / Σw) where w_i = 1/1.0² = 1 each, Σw=3
    expected_se_fe = math.sqrt(1.0 / 3.0)
    assert fe["se_FE_pp"] == pytest.approx(expected_se_fe, abs=1e-9)
    assert fe["n_below_se_floor_cells"] == 0  # no cell below the 0.68pp floor


def test_fe_pool_weighted_by_inverse_variance():
    """Cell with smaller SE should pull θ_FE toward it more.

    SEs 0.7 vs 1.4 (ratio 1:2 → weight ratio 4:1) are both ≥ 0.68pp so the
    AMENDMENT-03 floor does not collapse the contrast (the pre-amendment test used
    SE=0.1, which now floors to 1.0pp and would erase the weighting under test).
    """
    per_cell = [
        {"theta_pp": 2.0, "se_pp": 0.7},   # weight 1/0.49 (4×)
        {"theta_pp": 10.0, "se_pp": 1.4},  # weight 1/1.96 (1×)
    ]
    fe = _fe_pool(per_cell)
    # weight ratio 4:1 → θ_FE = (4·2 + 1·10) / 5 = 3.6, pulled toward the low-SE cell
    assert fe["theta_FE_pp"] == pytest.approx(18.0 / 5.0, abs=1e-9)
    assert fe["n_below_se_floor_cells"] == 0


def test_fe_pool_z_and_p_one_sided_at_threshold():
    """When θ_FE = δ exactly, z=0 and p_one_sided=0.5 (gate not passed)."""
    per_cell = [
        {"theta_pp": 1.0, "se_pp": 0.5},  # θ exactly at δ
        {"theta_pp": 1.0, "se_pp": 0.5},
    ]
    fe = _fe_pool(per_cell)
    assert fe["theta_FE_pp"] == pytest.approx(1.0, abs=1e-12)
    assert abs(fe["z_one_sided"]) < 1e-12  # z = (1 - 1) / SE = 0
    assert fe["p_one_sided"] == pytest.approx(0.5, abs=1e-9)
    assert fe["gate_passed"] is False


def test_fe_pool_returns_none_at_k1():
    """k=1 cell: FE pool ill-defined → None."""
    assert _fe_pool([{"theta_pp": 5.0, "se_pp": 1.0}]) is None


def test_fe_pool_handles_zero_se_via_floor():
    """SE-floor (AMENDMENT 03: `< 0.68pp → 1.0pp` Agresti-Coull threshold).

    History: pre-P0-9 used a 1e-9 floor (zero-SE row dominates, θ_FE≈2.0); P0-9
    floored only the literal-zero SE (threshold `<= 0` → θ_FE=2.8, the SE=0.5 cell
    keeping weight 4); AMENDMENT 03 (2026-05-24) aligns the threshold to the canonical
    0.68pp anchor (prereg L98 / B-1003), so BOTH the SE=0.0 cell AND the SE=0.5 cell
    (0.5 < 0.68) floor to 1.0pp → equal weights → θ_FE = (2 + 3)/2 = 2.5.
    `n_zero_se_floored_cells` counts only exact-zero (1); `n_below_se_floor_cells`
    counts all cells under the 0.68 threshold (2).
    """
    per_cell = [
        {"theta_pp": 2.0, "se_pp": 0.0},
        {"theta_pp": 3.0, "se_pp": 0.5},
    ]
    fe = _fe_pool(per_cell)
    assert fe["theta_FE_pp"] == pytest.approx(2.5, abs=1e-6)
    assert fe["n_zero_se_floored_cells"] == 1
    assert fe["n_below_se_floor_cells"] == 2


# ─── build_gate end-to-end ──────────────────────────────────────────────────
def test_build_gate_empty_cells():
    """0 cells → INSUFFICIENT_DATA, gate not blocking."""
    payload = build_gate([])
    assert payload["gate_status"] == "INSUFFICIENT_DATA"
    assert payload["per_cell"] == []
    assert payload["delta_pp"] == DELTA_PP


def test_build_gate_six_cells_passes(tmp_path):
    """6 cells all with strong P-SoM effect → gate PASSES.

    /stress A1.12 P0-2 (2026-05-16): cell topology aligned to Phase 1a
    canonical = {B0, B1, B2} × {cls, red} = 6 cells. Legacy fixture used
    B0+B1 × cls+red+shop which mismatched Phase 1a scope (B2 missing, shop
    deferred to Phase 1b).
    """
    cells = [
        _make_synthetic_cell(tmp_path / f"{b}_{s}", b, s,
                             psom_only_count=10, common_count=100)
        for b in ["B0", "B1", "B2"] for s in ["classifieds", "reddit"]
    ]
    payload = build_gate(cells, expected_ids_by_site=_expected_by_site(cells))
    assert payload["gate_status"] == "PASS"
    assert payload["analysis_status"] == "COMPLETE"
    assert payload["h1_verdict_normal_approx_transparency"] == "PASS"
    assert "h1_verdict" not in payload
    assert len(payload["per_cell"]) == 6
    fe = payload["pooled_fe"]
    # θ_FE = 10pp (each cell has 10/100=10% of tasks only P-SoM saves), >> δ=1.0pp
    assert fe["theta_FE_pp"] == pytest.approx(10.0, abs=1e-9)
    assert fe["gate_passed"] is True


def test_build_gate_six_cells_at_threshold_fails(tmp_path):
    """6 cells all at exactly δ=1.0pp → gate FAILS (one-sided).

    /stress A1.12 P0-2 (2026-05-16): Phase 1a canonical topology.
    """
    cells = [
        _make_synthetic_cell(tmp_path / f"{b}_{s}", b, s,
                             psom_only_count=1, common_count=100)
        for b in ["B0", "B1", "B2"] for s in ["classifieds", "reddit"]
    ]
    payload = build_gate(cells, expected_ids_by_site=_expected_by_site(cells))
    fe = payload["pooled_fe"]
    # θ_FE ≈ 1.0pp ≈ δ → z ≈ 0, p ≈ 0.5 > α → FAIL
    assert fe["gate_passed"] is False
    assert payload["gate_status"] == "FAIL"
    assert payload["analysis_status"] == "COMPLETE"
    assert payload["h1_verdict_normal_approx_transparency"] == "FAIL"
    assert "h1_verdict" not in payload


def test_build_gate_partial_data_three_cells(tmp_path):
    """3 of 6 cells available → PARTIAL_DATA, pooled FE still reported."""
    cells = [
        _make_synthetic_cell(tmp_path / f"B1_{s}", "B1", s,
                             psom_only_count=10, common_count=100)
        for s in ["classifieds", "reddit", "shopping"]
    ]
    payload = build_gate(cells, expected_ids_by_site=_expected_by_site(cells))
    assert payload["gate_status"] == "PARTIAL_DATA"
    assert payload["analysis_status"] == "PARTIAL"
    assert payload["h1_verdict_normal_approx_transparency"] == "NOT_EVALUATED"
    assert "h1_verdict" not in payload
    assert len(payload["per_cell"]) == 3
    assert "pooled_fe" in payload


# ─── output writers ─────────────────────────────────────────────────────────
def test_write_csv_per_cell_and_pooled_rows(tmp_path):
    """/stress A1.12 P0-2 (2026-05-16): Phase 1a canonical topology."""
    cells = [
        _make_synthetic_cell(tmp_path / f"{b}_{s}", b, s,
                             psom_only_count=10, common_count=100)
        for b in ["B0", "B1", "B2"] for s in ["classifieds", "reddit"]
    ]
    payload = build_gate(cells, expected_ids_by_site=_expected_by_site(cells))
    out_csv = tmp_path / "phase1_prereg_gate.csv"
    write_csv(payload, out_csv)
    text = out_csv.read_text()
    # Header + 6 cell rows + 1 pooled = 8 lines (last newline)
    assert text.startswith("row_type,baseline,site")
    rows = [r for r in text.splitlines() if r.strip()]
    assert len(rows) == 1 + 6 + 1  # header + cells + pool
    assert any(r.startswith("pooled_FE,") for r in rows)
    parsed = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    assert all(row["n_tasks"] == "100" for row in parsed if row["row_type"] == "cell")
    assert all(None not in row for row in parsed)  # no overflow columns


def test_write_json_round_trips(tmp_path):
    cells = [
        _make_synthetic_cell(tmp_path / f"B1_{s}", "B1", s,
                             psom_only_count=5, common_count=100)
        for s in ["classifieds", "reddit"]
    ]
    payload = build_gate(cells, expected_ids_by_site=_expected_by_site(cells))
    out_json = tmp_path / "phase1_prereg_gate.json"
    write_json(payload, out_json)
    loaded = json.loads(out_json.read_text())
    assert loaded["gate_status"] == payload["gate_status"]
    assert loaded["delta_pp"] == DELTA_PP


def test_write_md_renders(tmp_path):
    """/stress A1.12 P0-2 (2026-05-16): Phase 1a canonical topology."""
    cells = [
        _make_synthetic_cell(tmp_path / f"{b}_{s}", b, s,
                             psom_only_count=10, common_count=100)
        for b in ["B0", "B1", "B2"] for s in ["classifieds", "reddit"]
    ]
    payload = build_gate(cells, expected_ids_by_site=_expected_by_site(cells))
    out_md = tmp_path / "phase1_prereg_gate.md"
    write_md(payload, out_md)
    text = out_md.read_text()
    assert "# Phase 1 legacy H1 normal-approximation transparency check" in text
    assert "NOT the canonical H1 verdict" in text
    assert "PASSED" in text
    assert "θ_FE" in text
    assert "B-184" in text
