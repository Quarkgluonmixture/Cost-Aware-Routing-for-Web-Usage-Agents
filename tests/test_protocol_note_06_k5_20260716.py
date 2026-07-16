"""Regression locks for the isolated PROTOCOL_NOTE_06 k=5 verdict channel."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import aggregate_phase1_full_prereg_decision as full
from scripts.analysis import verdict_day_slotsheet as slots
from scripts.analysis.aggregate_phase1_prereg_gate import SIX_MODES


def _note(path: Path, status: str = "SIGNED / IN FORCE") -> Path:
    path.write_text(
        f"---\nstatus: {status}\nwitness_tag: {full.PROTOCOL_NOTE_06_WITNESS_TAG}\n---\n",
        encoding="utf-8",
    )
    return path


def _cell(site: str, baseline: str) -> dict:
    return {
        "site": site,
        "baseline": baseline,
        "modes": {mode: Path(f"/{baseline}/{site}/{mode}") for mode in SIX_MODES},
    }


def _fixed_cells() -> list[dict]:
    return [_cell(site, baseline) for site, baseline in full.PROTOCOL_NOTE_06_FIXED_CELLS]


def _install_synthetic_statistics(monkeypatch) -> None:
    jitter = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)

    monkeypatch.setattr(full, "load_cell_task_rows", lambda *_args, **_kwargs: {})

    def fake_h1(cell, **_kwargs):
        order = list(full.PROTOCOL_NOTE_06_FIXED_CELLS).index(
            (cell["site"], cell["baseline"])
        )
        theta = 0.5 + order * 0.2
        return {
            "baseline": cell["baseline"],
            "site": cell["site"],
            "complete_exact": True,
            "incomplete_reason": None,
            "expected_n": 100,
            "observed_n": {mode: 100 for mode in SIX_MODES},
            "missing_ids": {mode: [] for mode in SIX_MODES},
            "extra_ids": {mode: [] for mode in SIX_MODES},
            "task_set_sha256": f"sha-{cell['baseline']}-{cell['site']}",
            "n_tasks": 100,
            "theta_pp": theta,
            "se_pp": 1.0,
            "ci95_lo_pp": theta - 0.5,
            "ci95_hi_pp": theta + 0.5,
            "boot_pp": theta + jitter,
            "oracle_6_pp": 20.0,
            "oracle_5_no_psom_pp": 20.0 - theta,
            "n_psom_only": round(theta),
        }

    def fake_h3(_per_task, axis_mode, **_kwargs):
        theta = 1.25 if axis_mode == "P-text" else 2.5
        return {
            "axis_mode": axis_mode,
            "ref_mode": "P-SoM",
            "n_tasks": 100,
            "universe_label": "six_arm_complete_case",
            "unique_count_pp": theta,
            "n_unique": 3,
            "cell_pass": True,
            "se_pp": 1.0,
            "ci95_lo_pp": theta - 0.5,
            "ci95_hi_pp": theta + 0.5,
            "p_percentile_one_sided": 0.0,
            "boot_pp": theta + jitter,
        }

    monkeypatch.setattr(full, "_cell_drop_one_theta_se", fake_h1)
    monkeypatch.setattr(full, "_load_cell_per_task", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(full, "_h2a_per_task_ratio", lambda *_args: {
        "n_paired_tasks": 100,
        "n_ratios_computed": 100,
        "n_dom_zero_skipped": 0,
        "n_dom_missing": 0,
        "n_psom_missing": 0,
        "median_ratio": 1.0,
        "relative_diff_pct": 0.0,
        "margin_pct": 20.0,
        "per_cell_pass": True,
        "per_cell_falsified": False,
    })
    monkeypatch.setattr(full, "_h3_axis_per_cell", fake_h3)
    monkeypatch.setattr(full, "_load_h10_operational_gate_passed", lambda: None)


def test_pn06_rejects_b2_reddit_when_all_six_modes_are_bound(tmp_path, monkeypatch, capsys):
    note = _note(tmp_path / "note.md")
    cells = _fixed_cells() + [_cell("reddit", "B2")]
    monkeypatch.setattr(full, "PROTOCOL_NOTE_06_PATH", note)
    monkeypatch.setattr(full, "get_aggregator_cells", lambda **_kwargs: cells)
    monkeypatch.setattr(
        full, "write_protocol_note_06_outputs_atomic",
        lambda *_args, **_kwargs: pytest.fail("rejection must not write an artifact"),
    )

    assert full.main(["--protocol-note-06-k5"]) == 2
    assert (
        "k=6 upgrade rule: regenerate the full six-cell verdict instead"
        in capsys.readouterr().err
    )


def test_pn06_rejects_when_one_fixed_cell_is_missing(tmp_path, monkeypatch, capsys):
    note = _note(tmp_path / "note.md")
    monkeypatch.setattr(full, "PROTOCOL_NOTE_06_PATH", note)
    monkeypatch.setattr(full, "get_aggregator_cells", lambda **_kwargs: _fixed_cells()[:-1])
    monkeypatch.setattr(
        full, "write_protocol_note_06_outputs_atomic",
        lambda *_args, **_kwargs: pytest.fail("rejection must not write an artifact"),
    )

    assert full.main(["--protocol-note-06-k5"]) == 2
    assert "requires exactly the fixed five manifest-bound cells" in capsys.readouterr().err


@pytest.mark.parametrize("status", [None, "DRAFT / NOT-IN-FORCE"])
def test_pn06_rejects_missing_or_non_in_force_note(tmp_path, status):
    note = tmp_path / "note.md"
    if status is not None:
        _note(note, status)
    with pytest.raises(RuntimeError, match="PROTOCOL_NOTE_06"):
        full.build_protocol_note_06_k5_decision(_fixed_cells(), note_path=note)


def test_pn06_status_metadata_and_numeric_pool_regression_match_partial(
    tmp_path, monkeypatch,
):
    _install_synthetic_statistics(monkeypatch)
    cells = _fixed_cells()
    partial = full.build_full_decision(cells)
    pn06 = full.build_protocol_note_06_k5_decision(
        cells, note_path=_note(tmp_path / "note.md"),
    )

    assert partial["analysis_status"] == "PARTIAL"
    assert partial["h1_verdict"] == "NOT_EVALUATED"
    assert pn06["analysis_status"] == full.PROTOCOL_NOTE_06_STATUS
    assert pn06["h1_verdict"] in {"PASS", "FAIL"}
    assert pn06["protocol_note"] == "PROTOCOL_NOTE_06"
    assert pn06["verdict_qualifier"] == "on the five landed cells"
    assert pn06["b1284_one_tier_downgrade"] is True
    assert pn06["r_tier_cap"] == "R2"
    assert pn06["fixed_cell_set"] == list(full.PROTOCOL_NOTE_06_FIXED_CELL_IDS)
    assert pn06["witness_tag"] == full.PROTOCOL_NOTE_06_WITNESS_TAG

    numeric_paths = (
        ("pooled_h1_fe", "theta_FE_pp"),
        ("pooled_h1_bootstrap", "theta_fe_bootstrap_median_pp"),
        ("pooled_h1_bootstrap", "ci95_lo_pp_bootstrap"),
        ("pooled_h1_bootstrap", "ci95_hi_pp_bootstrap"),
        ("pooled_h1_bootstrap", "p_one_sided_bootstrap"),
        ("h3_axis1_pooled_fe", "theta_FE_pp"),
        ("h3_axis1_pooled_fe", "ci95_lo_pp_bootstrap"),
        ("h3_axis1_pooled_fe", "ci95_hi_pp_bootstrap"),
        ("h3_axis2_pooled_fe", "theta_FE_pp"),
        ("h3_axis2_pooled_fe", "ci95_lo_pp_bootstrap"),
        ("h3_axis2_pooled_fe", "ci95_hi_pp_bootstrap"),
    )
    for section, field in numeric_paths:
        assert pn06[section][field] == partial[section][field]
    assert pn06["h3_axis1_pooled_fe"]["axis_verdict"] == "PASS"
    assert pn06["h3_axis2_pooled_fe"]["axis_verdict"] == "PASS"
    assert partial["h3_axis1_pooled_fe"]["axis_verdict"] == "NOT_EVALUATED"


def _write_slots_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    captured_at = "2026-07-16T12:00:00+00:00"
    decision = {
        "analysis_status": slots.PROTOCOL_NOTE_06_STATUS,
        "h1_verdict": "FAIL",
        "gate_status": "FAIL",
        "captured_at": captured_at,
        "protocol_note": "PROTOCOL_NOTE_06",
        "verdict_qualifier": slots.PROTOCOL_NOTE_06_QUALIFIER,
        "b1284_one_tier_downgrade": True,
        "r_tier_cap": "R2",
        "fixed_cell_set": sorted(slots.PROTOCOL_NOTE_06_CELL_IDS),
        "witness_tag": full.PROTOCOL_NOTE_06_WITNESS_TAG,
        "pooled_h1_fe": {"theta_FE_pp": 0.829511184576794},
        "pooled_h1_bootstrap": {
            "theta_fe_bootstrap_median_pp": 0.7951111799270356,
            "ci95_lo_pp_bootstrap": 0.26683096315504606,
            "ci95_hi_pp_bootstrap": 1.4906966554811916,
            "p_one_sided_bootstrap": 0.743,
            "k_cells": 5,
        },
        "h1_heterogeneity": {
            "I_squared_pct": 0.0,
            "heterogeneity_cap_at_r3": False,
        },
        "h3_axis1_pooled_fe": {
            "theta_FE_pp": 1.2603890712940347,
            "ci95_lo_pp_bootstrap": 0.6806760068610969,
            "ci95_hi_pp_bootstrap": 1.9917264335521296,
            "p_one_sided_bootstrap": 0.0,
            "axis_verdict": "PASS",
            "passed": True,
        },
        "h3_axis2_pooled_fe": {
            "theta_FE_pp": 2.5964581098559973,
            "ci95_lo_pp_bootstrap": 1.6746833078631898,
            "ci95_hi_pp_bootstrap": 3.6268210561227456,
            "p_one_sided_bootstrap": 0.0,
            "axis_verdict": "PASS",
            "passed": True,
        },
        "per_cell": [],
    }
    sr_rows: list[dict] = []
    fig_rows: list[dict] = []
    for cid in sorted(slots.PROTOCOL_NOTE_06_CELL_IDS):
        baseline, site = cid.split("_", 1)
        task_sha = f"sha-{cid}"
        decision["per_cell"].append({
            "baseline": baseline,
            "site": site,
            "h1": {
                "complete_exact": True,
                "observed_n": {mode: 100 for mode in slots.MODE_ORDER},
                "task_set_sha256": task_sha,
                "theta_pp": 1.0,
                "ci95_lo_pp": 0.5,
                "ci95_hi_pp": 1.5,
            },
            "h2a": {"per_cell_pass": True, "median_ratio": 1.0},
        })
        for mode in slots.MODE_ORDER:
            sr_rows.append({
                "baseline": baseline,
                "site": site,
                "mode": mode,
                "n_total": 100,
                "observed_n": 100,
                "expected_n": 100,
                "n_success": 50,
                "sr_denominator_n": 100,
                "sr_pct": 50.0,
                "completeness_ratio": 1.0,
                "complete_exact": True,
                "task_set_sha256": task_sha,
            })
            fig_rows.append({
                "row_type": "numeric",
                "site_baseline": f"{baseline}·{site}",
                "baseline": baseline,
                "site": site,
                "mode": mode,
                "drop_one_loss_pp": "1.0",
                "ci95_low_pp": "0.5",
                "ci95_high_pp": "1.5",
                "complete_exact": "true",
                # The six-panel generator globally demotes all rows while B2
                # Reddit is absent; NOTE_06 must revalidate the authorized five
                # row-by-row rather than trust this global label.
                "grade": "NON_PAPER_GRADE",
                "is_partial": "false",
                "portfolio_modes": json.dumps(slots.MODE_ORDER),
                "n_modes_unique": "6",
                "n_common": "100",
                "n_expected": "100",
                "task_set_sha256": task_sha,
                "captured_at": captured_at,
            })
    # The ordinary six-panel generator may retain the absent B2 Reddit panel as
    # an error row.  NOTE_06 validates and copies only its authorized five.
    fig_rows.append({
        **fig_rows[0],
        "row_type": "panel_error",
        "baseline": "B2",
        "site": "reddit",
        "site_baseline": "B2·reddit",
    })

    decision_path = tmp_path / "decision.json"
    sr_path = tmp_path / "sr.json"
    fig_path = tmp_path / "fig0c.csv"
    decision_path.write_text(json.dumps(decision), encoding="utf-8")
    sr_path.write_text(
        json.dumps({"captured_at": captured_at, "summary_table": sr_rows}),
        encoding="utf-8",
    )
    with fig_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fig_rows[0]))
        writer.writeheader()
        writer.writerows(fig_rows)
    return decision_path, sr_path, fig_path


@pytest.mark.parametrize("wrong_status", ["PARTIAL", "COMPLETE"])
def test_pn06_slotsheet_rejects_wrong_status(tmp_path, capsys, wrong_status):
    decision, sr, fig0c = _write_slots_artifacts(tmp_path)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["analysis_status"] = wrong_status
    decision.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "must-not-exist.md"
    assert slots.main([
        "--protocol-note-06",
        "--decision", str(decision),
        "--sr", str(sr),
        "--fig0c", str(fig0c),
        "--out", str(out),
    ]) == 2
    assert not out.exists()
    assert "analysis_status=COMPLETE_K5_PROTOCOL_NOTE_06" in capsys.readouterr().err


def test_pn06_slotsheet_rejects_missing_qualifier(tmp_path, capsys):
    decision, sr, fig0c = _write_slots_artifacts(tmp_path)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload.pop("verdict_qualifier")
    decision.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "must-not-exist.md"
    assert slots.main([
        "--protocol-note-06", "--decision", str(decision),
        "--sr", str(sr), "--fig0c", str(fig0c), "--out", str(out),
    ]) == 2
    assert not out.exists()
    assert "requires verdict_qualifier" in capsys.readouterr().err


def test_pn06_slotsheet_opens_h1_h3_and_closes_h10(tmp_path):
    decision, sr, fig0c = _write_slots_artifacts(tmp_path)
    out = tmp_path / "slotsheet.md"
    assert slots.main([
        "--protocol-note-06", "--decision", str(decision),
        "--sr", str(sr), "--fig0c", str(fig0c), "--out", str(out),
    ]) == 0
    sheet = out.read_text(encoding="utf-8")
    assert sheet.startswith("# PROTOCOL_NOTE_06 k=5 verdict-day slot sheet")
    assert "r_tier_cap=`R2`" in sheet
    assert "B-1284 one-tier downgrade=`true`" in sheet
    assert "**Branch B**" in sheet
    assert f"FAIL — {slots.PROTOCOL_NOTE_06_QUALIFIER}" in sheet
    assert f"PASS — {slots.PROTOCOL_NOTE_06_QUALIFIER}" in sheet
    assert f"| THETA | +0.83 — {slots.PROTOCOL_NOTE_06_QUALIFIER} |" in sheet
    assert f"| AX1 | +1.26 [+0.68, +1.99] — {slots.PROTOCOL_NOTE_06_QUALIFIER} |" in sheet
    assert slots.H10_PENDING_NOTICE in sheet
    assert slots.H10_PENDING_ABSTRACT in sheet
    assert "ROUTER_" not in sheet
    assert "B2·reddit" not in sheet
    table4 = sheet.split("## F. Table 4 regen (H10)", 1)[1].split(
        "## G. Post-splice checklist", 1,
    )[0]
    assert "numeric rows are intentionally withheld" in table4


def test_pn06_slotsheet_mode_is_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as excinfo:
        slots.main(["--protocol-note-06", "--h10-pending"])
    assert excinfo.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err
