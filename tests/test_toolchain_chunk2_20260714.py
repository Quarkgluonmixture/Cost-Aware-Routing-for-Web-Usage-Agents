"""Regression coverage for the 2026-07-14 verdict-toolchain Chunk 2 fixes."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis.figures import fig0c_drop_one_oracle as fig0c
from scripts.analysis.figures import fig_f1_diamond_schematic as f1
from scripts.analysis.figures import fig_f2_h1_forest as f2
from scripts.analysis import router_covariate_baseline as router
from scripts.analysis import verdict_day_slotsheet as slots


REPO = Path(__file__).resolve().parents[1]


def test_g1_bootstrap_uses_full_common_universe_including_all_failures():
    common = set(range(100))
    sets = {mode: set() for mode in fig0c.MODES}
    sets["P-SoM"] = set(range(10))
    ci = fig0c.bootstrap_drop_one_ci(
        sets, common, n_bootstrap=10_000, seed=42,
    )["P-SoM"]
    assert ci[0] < 10.0 < ci[1]
    assert ci != (10.0, 10.0)


def test_g2_fig0c_exact_six_mode_and_task_set_guard(monkeypatch):
    expected = set(range(10))
    panel = {
        "key": "b0_cls",
        "title": "B0 classifieds",
        "baseline": "B0",
        "site": "classifieds",
        "expected": 10,
        "expected_task_ids": expected,
        "modes": {mode: Path(mode) for mode in fig0c.MODES},
    }

    def fake_load(path: Path):
        observed = set(range(9)) if path.name == "P-prompt" else set(expected)
        return ({0} & observed), observed

    monkeypatch.setattr(fig0c, "load_success_set", fake_load)
    sets_final, _, common_final, partial, meta_final = fig0c.load_panel_sets(panel)
    assert not sets_final and not common_final
    assert meta_final["complete_exact"] is False
    assert meta_final["n_modes_unique"] == 6
    assert "P-prompt" in partial
    error_row = fig0c._panel_error_row(panel, meta_final, "2026-07-14T00:00:00+00:00", "bad")
    assert error_row["row_type"] == "panel_error"
    assert error_row["is_partial"] is True
    assert "portfolio_modes" in error_row and "task_set_sha256" in error_row

    sets_partial, _, common_partial, _, meta_partial = fig0c.load_panel_sets(
        panel, allow_partial=True,
    )
    assert len(sets_partial) == 6
    assert common_partial == set(range(9))
    assert meta_partial["complete_exact"] is False


def _partial_decision_for_f2() -> dict:
    return {
        "analysis_status": "PARTIAL",
        "h1_verdict": "NOT_EVALUATED",
        "per_cell": [{
            "baseline": "B0",
            "site": "classifieds",
            "h1": {
                "complete_exact": True,
                "observed_n": {mode: 2 for mode in f2.PAPER_MODES},
                "n_tasks": 2,
                "task_set_sha256": "sha",
                "theta_pp": 2.0,
                "ci95_lo_pp": 0.5,
                "ci95_hi_pp": 4.0,
            },
        }],
        "pooled_h1_fe": {"theta_FE_pp": 2.0},
        "pooled_h1_bootstrap": {
            "ci95_lo_pp_bootstrap": 0.5,
            "ci95_hi_pp_bootstrap": 4.0,
            "k_cells": 1,
        },
    }


def test_g2_f2_single_source_interim_and_final_fail_closed(tmp_path, monkeypatch):
    decision = tmp_path / "decision.json"
    decision.write_text(json.dumps(_partial_decision_for_f2()))
    monkeypatch.setattr(f2, "expected_scored_ids", lambda site: (frozenset({1, 2}), "sha"))
    out = tmp_path / "nested" / "f2"
    assert f2.main(["--decision", str(decision), "--out", str(out), "--interim"]) == 0
    assert out.with_suffix(".png").is_file()
    assert out.with_suffix(".pdf").is_file()

    final_out = tmp_path / "must_not_exist" / "f2"
    assert f2.main(["--decision", str(decision), "--out", str(final_out)]) == 2
    assert not final_out.with_suffix(".png").exists()


def test_g3_router_cli_requires_explicit_paths_and_rehearsal_opt_in(tmp_path):
    script = REPO / "scripts/analysis/router_covariate_baseline.py"
    no_args = subprocess.run(
        [sys.executable, str(script)], cwd=REPO, text=True, capture_output=True,
    )
    assert no_args.returncode != 0
    assert "--raw-features" in no_args.stderr and "--out-json" in no_args.stderr

    rehearsal = tmp_path / "router_rehearsal" / "raw.npz"
    rehearsal.parent.mkdir()
    rehearsal.write_bytes(b"not loaded because opt-in guard fires first")
    blocked = subprocess.run(
        [
            sys.executable, str(script),
            "--raw-features", str(rehearsal),
            "--out-json", str(tmp_path / "out.json"),
        ],
        cwd=REPO, text=True, capture_output=True,
    )
    assert blocked.returncode == 2
    assert "--allow-rehearsal" in blocked.stderr


def test_g4_majority_is_fold_local_and_audit_smoke_is_one_third():
    fold0 = np.array(["A"] * 25 + ["B"] * 10)
    fold1 = np.array(["A"] * 15 + ["B"] * 25)
    pred_fold0, meta0 = router.fold_local_majority_prediction(fold1, len(fold0))
    pred_fold1, meta1 = router.fold_local_majority_prediction(fold0, len(fold1))
    y = np.concatenate([fold0, fold1])
    pred = np.array(pred_fold0 + pred_fold1)
    assert float((pred == y).mean()) == pytest.approx(1 / 3)
    assert meta0["majority_class"] == "B" and meta0["majority_count"] == 25
    assert meta1["majority_class"] == "A" and meta1["majority_count"] == 25
    tied, tied_meta = router.fold_local_majority_prediction(["B", "A"], 2)
    assert tied == ["A", "A"]
    assert tied_meta["tie_rule"] == "lexicographically_smallest_class"


def test_g5_template_coverage_fails_closed_and_fallback_is_diagnostic_only():
    raw = {
        "cell_ids": np.array(["B0_classifieds"] * 3),
        "task_ids": np.array([1, 2, 3]),
        "labels": np.array(["A", "B", "A"]),
        "all_cell_ids": np.array(["B0_classifieds"] * 3),
        "all_task_ids": np.array([1, 2, 3]),
    }
    maps = {"classifieds": {1: 10, 2: 10}}
    report = router.template_coverage_report(
        raw, maps, {"classifieds": {}}, n_splits=2,
    )
    assert report["valid"] is False
    assert report["sites"]["classifieds"]["coverage_pct"] < 100.0
    with pytest.raises(ValueError, match="--diagnostic"):
        router.template_disjoint_fold_assignments(
            raw, maps, n_splits=2, diagnostic=False,
        )
    folds = router.template_disjoint_fold_assignments(
        raw, maps, n_splits=2, diagnostic=True,
    )
    assert set(folds["B0_classifieds"]) == {1, 2, 3}


def _oof_record(ids: list[int], *, invert: bool, feature: str, regime: str) -> dict:
    y = ["A" if task_id % 2 == 0 else "B" for task_id in ids]
    proba = []
    for label in y:
        p = [0.95, 0.05] if label == "A" else [0.05, 0.95]
        proba.append(list(reversed(p)) if invert else p)
    return {
        "status": "ok",
        "cell_id": "B0_classifieds",
        "feature_set": feature,
        "split_regime": regime,
        "oof_rows": {
            "task_ids": ids,
            "y_true": y,
            "classes": ["A", "B"],
            "proba": proba,
        },
    }


def test_g6_paired_contrast_intersects_ids_asserts_labels_and_bootstraps():
    left = _oof_record(list(range(8)), invert=False, feature="full_lr", regime="standard")
    right = _oof_record(list(range(2, 10)), invert=True, feature="scalar_min", regime="standard")
    contrast = router.paired_auroc_contrast(
        left, right, contrast_id="full-vs-scalar:standard", B=200, seed=42,
    )
    assert contrast["n_left"] == 8
    assert contrast["n_right"] == 8
    assert contrast["n_common"] == 6
    assert contrast["dropped_ids"] == {"left_only": [0, 1], "right_only": [8, 9]}
    assert contrast["delta_auroc"] == pytest.approx(1.0)
    assert contrast["ci95"] == pytest.approx([1.0, 1.0])

    bad = _oof_record(list(range(2, 10)), invert=True, feature="scalar_min", regime="standard")
    bad["oof_rows"]["y_true"][0] = "B"
    with pytest.raises(AssertionError, match="y_true mismatch"):
        router.paired_auroc_contrast(left, bad, contrast_id="bad", B=10)


def test_g6_predefined_contrast_families_are_built_in():
    records = []
    for regime in router.SPLIT_REGIMES:
        records.append(_oof_record(list(range(8)), invert=False, feature="full_lr", regime=regime))
        records.append(_oof_record(list(range(8)), invert=True, feature="scalar_min", regime=regime))
    contrasts = router.build_predefined_contrasts(records, B=20, seed=42)
    assert {c["contrast_id"] for c in contrasts} == {
        "full-vs-scalar:standard",
        "full-vs-scalar:template_disjoint",
        "standard-vs-template-disjoint:full_lr",
    }


def test_g7_slotsheet_noncomplete_is_no_branch_and_suppresses_copyable_blocks():
    sheet = slots.build_sheet(
        _partial_decision_for_f2(), {}, {"summary_table": []}, [], {},
        rehearsal=True, errors=["synthetic partial"],
        gaps=["SR: captured_at absent"],
    )
    assert sheet.startswith("# INVALID_FOR_DRAFT")
    assert "NO_BRANCH" in sheet
    assert "## C. Canonical slot values" not in sheet
    assert "Copyable §C–§F slots/tables intentionally suppressed" in sheet
    assert "剩余 provenance gap" in sheet


def test_g7_slotsheet_final_missing_artifacts_exits_nonzero_without_output(tmp_path):
    out = tmp_path / "nested" / "sheet.md"
    rc = slots.main([
        "--decision", str(tmp_path / "missing-decision.json"),
        "--h10", str(tmp_path / "missing-h10.json"),
        "--sr", str(tmp_path / "missing-sr.json"),
        "--fig0c", str(tmp_path / "missing.csv"),
        "--router", str(tmp_path / "missing-router.json"),
        "--out", str(out),
    ])
    assert rc == 2
    assert not out.exists()


def test_g8_round_half_up_and_router_contrast_slots_are_canonical():
    assert slots.decimal_format(6.25, 1, signed=False) == "6.3"
    decision = {
        "analysis_status": "COMPLETE",
        "h1_verdict": "PASS",
        "gate_status": "PASS",
        "captured_at": "2026-07-14T00:00:00+00:00",
        "pooled_h1_fe": {"theta_FE_pp": 6.25},
        "pooled_h1_bootstrap": {
            "ci95_lo_pp_bootstrap": 1.25,
            "ci95_hi_pp_bootstrap": 2.35,
            "p_one_sided_bootstrap": 0.01235,
            "k_cells": 6,
        },
    }
    sr = {"summary_table": [
        {"site": "classifieds", "baseline": "B0", "mode": mode, "sr_pct": 6.25}
        for mode in slots.MODE_ORDER
    ]}
    router_payload = {"paired_contrasts": [{
        "cell_id": "B0_classifieds",
        "contrast_id": "full-vs-scalar:standard",
        "delta_auroc": 0.1255,
        "ci95": [-0.005, 0.255],
        "n_common": 20,
    }]}
    sheet = slots.build_sheet(
        decision, {"per_cell": {"B0_classifieds": {}}}, sr, [], router_payload,
        rehearsal=False, errors=[], gaps=[],
    )
    assert "| THETA | +6.25 |" in sheet
    assert "ROUTER_B0_CLASSIFIEDS_FULL_VS_SCALAR_STANDARD" in sheet
    assert "ΔAUROC=+0.126" in sheet
    assert "| B0·classifieds | 6.3 | 6.3 |" in sheet


def test_g9_output_parents_and_strict_json_null_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(f1, "OUT", tmp_path / "deep" / "figures" / "f1")
    assert f1.main() == 0
    assert (tmp_path / "deep/figures/f1.png").is_file()
    assert (tmp_path / "deep/figures/f1.pdf").is_file()

    payload = {"macro_ovr_auroc": None, "metric_status": "undefined_single_class"}
    encoded = router.strict_json_dumps(payload)
    assert '"macro_ovr_auroc": null' in encoded
    with pytest.raises(ValueError):
        router.strict_json_dumps({"bad": float("nan")})

    rehearsal_out = tmp_path / "more" / "parents" / "sheet.md"
    assert slots.main([
        "--decision", str(tmp_path / "missing-decision.json"),
        "--h10", str(tmp_path / "missing-h10.json"),
        "--sr", str(tmp_path / "missing-sr.json"),
        "--fig0c", str(tmp_path / "missing.csv"),
        "--router", str(tmp_path / "missing-router.json"),
        "--out", str(rehearsal_out),
        "--rehearsal",
    ]) == 0
    assert rehearsal_out.is_file()
    assert rehearsal_out.read_text().startswith("# INVALID_FOR_DRAFT")
