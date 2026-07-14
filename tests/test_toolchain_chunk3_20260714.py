"""Regression coverage for the 2026-07-14 verdict-toolchain Chunk 3 fixes."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.analysis import aggregate_h10_pareto as h10
from scripts.analysis import aggregate_phase1_prereg_gate as legacy_gate
from scripts.analysis import aggregate_phantom_lift as phantom_lift
from scripts.analysis import verdict_day_slotsheet as slots
from scripts.analysis.figures import fig_f1_diamond_schematic as f1
from scripts.analysis.lib.canonical_task_universe import expected_scored_ids


REPO = Path(__file__).resolve().parents[1]


def _partial_decision() -> dict:
    return {
        "analysis_status": "PARTIAL",
        "h1_verdict": "NOT_EVALUATED",
        "gate_status": "PARTIAL_DATA",
        "pooled_h1_fe": {"theta_FE_pp": 1.5},
        "pooled_h1_bootstrap": {
            "ci95_lo_pp_bootstrap": 0.5,
            "ci95_hi_pp_bootstrap": 2.5,
            "p_one_sided_bootstrap": 0.01,
            "k_cells": 3,
        },
        "per_cell": [],
    }


def _rehearsal_router() -> dict:
    return {
        "grade": "NON_PAPER_GRADE",
        "analysis_status": "NON_PAPER_GRADE",
        "captured_at": "2026-07-14T12:00:00+00:00",
        "paired_contrasts": [
            {
                "cell_id": "B0_classifieds",
                "contrast_id": "full-vs-scalar:standard",
                "delta_auroc": 0.1255,
                "ci95": [-0.005, 0.255],
                "n_common": 20,
            },
            {
                "cell_id": "B1_reddit",
                "contrast_id": "standard-vs-template-disjoint:full_lr",
                "delta_auroc": -0.2,
                "ci95": [-0.4, 0.1],
                "n_common": 17,
            },
        ],
    }


def test_k1_rehearsal_has_noncopyable_router_c_prime_and_grade_markers():
    sheet = slots.build_sheet(
        _partial_decision(), {}, {"summary_table": []}, [], _rehearsal_router(),
        rehearsal=True, errors=[], gaps=[],
    )
    assert sheet.startswith("# INVALID_FOR_DRAFT")
    assert (
        "## C'. Router covariate diagnostics "
        "(REHEARSAL — NON-COPYABLE, 禁止进 draft)"
    ) in sheet
    assert "grade=`NON_PAPER_GRADE`" in sheet
    assert "analysis_status=`NON_PAPER_GRADE`" in sheet
    assert "canonical slot source" in sheet
    assert "## C. Canonical slot values" not in sheet


def test_k1_rehearsal_summarizes_every_contrast_in_warning_shaped_rows():
    sheet = slots.build_sheet(
        _partial_decision(), {}, {"summary_table": []}, [], _rehearsal_router(),
        rehearsal=True, errors=[], gaps=[],
    )
    assert "| ⚠ diagnostic only | cell | contrast | delta_AUROC | CI95 | n_common |" in sheet
    assert sheet.count("| ⚠ NON-COPYABLE |") == 2
    assert (
        "| ⚠ NON-COPYABLE | B0_classifieds | full-vs-scalar:standard | "
        "+0.126 | [-0.005, +0.255] | 20 |"
    ) in sheet
    assert (
        "| ⚠ NON-COPYABLE | B1_reddit | "
        "standard-vs-template-disjoint:full_lr | -0.200 | "
        "[-0.400, +0.100] | 17 |"
    ) in sheet
    assert "Copyable §C–§F slots/tables intentionally suppressed" in sheet


def _write_complete_pass1_artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    captured_at = "2026-07-14T12:00:00+00:00"
    decision = {
        "analysis_status": "COMPLETE",
        "h1_verdict": "PASS",
        "gate_status": "PASS",
        "captured_at": captured_at,
        "pooled_h1_fe": {"theta_FE_pp": 1.5},
        "pooled_h1_bootstrap": {
            "ci95_lo_pp_bootstrap": 0.5,
            "ci95_hi_pp_bootstrap": 2.5,
            "p_one_sided_bootstrap": 0.01,
            "k_cells": 6,
        },
        "h1_heterogeneity": {
            "I_squared_pct": 5.0,
            "heterogeneity_cap_at_r3": False,
        },
        "h3_axis1_pooled_fe": {
            "theta_FE_pp": 2.0,
            "ci95_lo_pp_bootstrap": 1.0,
            "ci95_hi_pp_bootstrap": 3.0,
            "passed": True,
        },
        "h3_axis2_pooled_fe": {
            "theta_FE_pp": 1.0,
            "ci95_lo_pp_bootstrap": 0.0,
            "ci95_hi_pp_bootstrap": 2.0,
            "passed": True,
        },
        "per_cell": [],
    }
    sr_rows = []
    fig_rows = []
    for cell_id in sorted(slots.PLANNED_CELL_IDS):
        baseline, site = cell_id.split("_", 1)
        task_sha = expected_scored_ids(site)[1]
        decision["per_cell"].append({
            "baseline": baseline,
            "site": site,
            "h1": {
                "complete_exact": True,
                "observed_n": {mode: 2 for mode in slots.MODE_ORDER},
                "task_set_sha256": task_sha,
                "theta_pp": 1.5,
                "ci95_lo_pp": 0.5,
                "ci95_hi_pp": 2.5,
            },
            "h2a": {"passed": True},
        })
        for mode in slots.MODE_ORDER:
            sr_rows.append({
                "baseline": baseline,
                "site": site,
                "mode": mode,
                "n_total": 2,
                "observed_n": 2,
                "expected_n": 2,
                "n_success": 1,
                "sr_denominator_n": 2,
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
                "drop_one_loss_pp": "1.5",
                "ci95_low_pp": "0.5",
                "ci95_high_pp": "2.5",
                "complete_exact": "true",
                "grade": "PAPER_GRADE",
                "is_partial": "false",
                "portfolio_modes": json.dumps(slots.MODE_ORDER),
                "n_modes_unique": "6",
                "n_common": "2",
                "n_expected": "2",
                "task_set_sha256": task_sha,
                "captured_at": captured_at,
            })

    decision_path = tmp_path / "decision.json"
    sr_path = tmp_path / "sr.json"
    fig0c_path = tmp_path / "fig0c.csv"
    decision_path.write_text(json.dumps(decision))
    sr_path.write_text(json.dumps({
        "captured_at": captured_at,
        "summary_table": sr_rows,
    }))
    with fig0c_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fig_rows[0]))
        writer.writeheader()
        writer.writerows(fig_rows)
    return decision_path, sr_path, fig0c_path


def _build_producer_h10(monkeypatch) -> dict:
    """Exercise the real producer schema without requiring landed Pass-2 data."""
    monkeypatch.setattr(h10, "find_pass1_run_dirs", lambda *_args: [Path("synthetic")])
    monkeypatch.setattr(
        h10, "collect_per_task_outcomes_with_metrics", lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        h10, "aggregate_arm_metrics",
        lambda *_args, **_kwargs: ([0] * h10.PHANTOM_PROMPT_BASELINE_MIN_EP, [], [], []),
    )

    def fake_analyze_cell(baseline, site, **_kwargs):
        return {
            "cell_id": f"{baseline}_{site}",
            "baseline": baseline,
            "site": site,
            "status": "ok",
            "passes": True,
            "theta_mean_pp": 2.0,
            "theta_se_pp": 1.0,
            "router_strictly_better": True,
            "cell_cost_unit_basis": {
                "homogeneous": True,
                "modal_basis": "api_usd",
            },
        }

    monkeypatch.setattr(h10, "analyze_cell", fake_analyze_cell)
    monkeypatch.setattr(
        h10, "_load_h10_entropy_gate",
        lambda: {"h10_entropy_gate_passed": True},
    )
    monkeypatch.setattr(h10, "_load_ladder_disclosure", lambda: None)
    return h10.build_verdict()


def _complete_router_payload(decision: dict) -> dict:
    captured_at = decision["captured_at"]
    task_shas = {
        f"{cell['baseline']}_{cell['site']}": cell["h1"]["task_set_sha256"]
        for cell in decision["per_cell"]
    }
    contrast_ids = (
        "full-vs-scalar:standard",
        "full-vs-scalar:template_disjoint",
        "standard-vs-template-disjoint:full_lr",
    )
    return {
        "grade": "PAPER_GRADE",
        "analysis_status": "COMPLETE",
        "captured_at": captured_at,
        "cells": sorted(slots.PLANNED_CELL_IDS),
        "paired_contrasts": [
            {
                "cell_id": cid,
                "contrast_id": contrast_id,
                "delta_auroc": 0.1,
                "ci95": [0.0, 0.2],
                "n_common": 100,
            }
            for cid in sorted(slots.PLANNED_CELL_IDS)
            for contrast_id in contrast_ids
        ],
        "results": [{"status": "ok"}],
        "canonical_input_validation": {
            "cells": {
                cid: {"task_set_sha256": task_shas[cid]}
                for cid in sorted(slots.PLANNED_CELL_IDS)
            },
        },
    }


def test_n01_full_final_accepts_real_h10_producer_schema_and_rejects_sha_mismatch(
    tmp_path, monkeypatch, capsys,
):
    decision_path, sr_path, fig0c_path = _write_complete_pass1_artifacts(tmp_path)
    decision = json.loads(decision_path.read_text())
    h10_payload = _build_producer_h10(monkeypatch)
    assert "task_set_sha256" not in h10_payload
    assert set(h10_payload["per_cell"]) == slots.PLANNED_CELL_IDS

    h10_path = tmp_path / "h10.json"
    router_path = tmp_path / "router.json"
    h10_path.write_text(json.dumps(h10_payload))
    router_path.write_text(json.dumps(_complete_router_payload(decision)))
    valid_out = tmp_path / "valid-slotsheet.md"
    common_args = [
        "--decision", str(decision_path),
        "--h10", str(h10_path),
        "--sr", str(sr_path),
        "--fig0c", str(fig0c_path),
        "--router", str(router_path),
    ]

    assert slots.main([*common_args, "--out", str(valid_out)]) == 0
    assert valid_out.read_text().startswith("# Verdict-day slot sheet")
    assert capsys.readouterr().err == ""

    h10_payload["per_cell"]["B1_reddit"]["task_set_sha256"] = "wrong-site-universe"
    h10_path.write_text(json.dumps(h10_payload))
    mismatch_out = tmp_path / "mismatch-must-not-exist.md"
    assert slots.main([*common_args, "--out", str(mismatch_out)]) == 2
    assert not mismatch_out.exists()
    assert (
        "task_set_sha256 mismatch decision↔H10 for B1_reddit"
        in capsys.readouterr().err
    )

    h10_payload["task_set_sha256"] = "arbitrary-nonjoinable-global"
    for cell in h10_payload["per_cell"].values():
        cell.pop("task_set_sha256", None)
    h10_path.write_text(json.dumps(h10_payload))
    global_only_out = tmp_path / "global-only-must-not-exist.md"
    assert slots.main([*common_args, "--out", str(global_only_out)]) == 2
    assert not global_only_out.exists()
    global_only_errors = capsys.readouterr().err
    assert "H10 B0_classifieds required task_set_sha256 missing" in global_only_errors
    assert "H10 B2_reddit required task_set_sha256 missing" in global_only_errors


def test_u2_legacy_cli_run_manifest_reaches_live_aggregator(
    tmp_path, monkeypatch,
):
    manifest = tmp_path / "chosen-run-manifest.yaml"
    manifest.write_text("schema_version: test\n")
    selected_cells = [{"baseline": "B9", "site": "reddit", "modes": {}}]
    seen: dict[str, object] = {}

    def fake_get_aggregator_cells(*, manifest_path):
        seen["manifest_path"] = manifest_path
        return selected_cells

    def fake_build_gate(cells):
        seen["cells"] = cells
        return {"gate_status": "INSUFFICIENT_DATA", "per_cell": [], "skipped_cells": []}

    monkeypatch.setattr(phantom_lift, "get_aggregator_cells", fake_get_aggregator_cells)
    monkeypatch.setattr(legacy_gate, "build_gate", fake_build_gate)
    monkeypatch.setattr(legacy_gate, "write_csv", lambda *_args: None)
    monkeypatch.setattr(legacy_gate, "write_json", lambda *_args: None)
    monkeypatch.setattr(legacy_gate, "write_md", lambda *_args: None)

    assert legacy_gate.main(["--run-manifest", str(manifest)]) == 0
    assert seen == {"manifest_path": manifest, "cells": selected_cells}


def test_k3_h10_pending_complete_emits_copyable_pass1_and_pending_partitions(tmp_path):
    decision, sr, fig0c = _write_complete_pass1_artifacts(tmp_path)
    out = tmp_path / "nested" / "slotsheet.md"
    rc = slots.main([
        "--h10-pending",
        "--decision", str(decision),
        "--h10", str(tmp_path / "pass2-not-landed.json"),
        "--sr", str(sr),
        "--fig0c", str(fig0c),
        "--router", str(tmp_path / "router-not-landed.json"),
        "--out", str(out),
    ])
    assert rc == 0
    sheet = out.read_text()
    for heading in (
        "## A. Gate status", "## B. Branch suggestion",
        "## C. Canonical slot values", "## D. Table 2 regen",
        "## E. Table 3 regen", "## F. Table 4 regen",
    ):
        assert heading in sheet
    assert "**Branch A**" in sheet
    assert "| THETA | +1.50 |" in sheet
    assert "| AX1 | +2.00 [+1.00, +3.00] |" in sheet
    assert "| B0·classifieds | 50.0 |" in sheet
    assert "| B0·classifieds | DOM | +1.50 | [+0.50, +2.50] |" in sheet
    assert slots.H10_PENDING_NOTICE in sheet
    assert slots.H10_PENDING_ABSTRACT in sheet
    assert "ROUTER_" not in sheet
    table4 = sheet.split("## F. Table 4 regen (H10)", 1)[1].split(
        "## G. Post-splice checklist", 1,
    )[0]
    assert "numeric rows are intentionally withheld" in table4
    assert "B0_classifieds:" not in table4


def test_k3_h10_pending_rejects_partial_as_non_bypass(tmp_path, capsys):
    decision, sr, fig0c = _write_complete_pass1_artifacts(tmp_path)
    payload = json.loads(decision.read_text())
    payload["analysis_status"] = "PARTIAL"
    payload["h1_verdict"] = "NOT_EVALUATED"
    decision.write_text(json.dumps(payload))
    out = tmp_path / "must-not-exist" / "slotsheet.md"
    rc = slots.main([
        "--h10-pending",
        "--decision", str(decision),
        "--sr", str(sr),
        "--fig0c", str(fig0c),
        "--out", str(out),
    ])
    assert rc == 2
    assert not out.exists()
    stderr = capsys.readouterr().err
    assert "--h10-pending requires decision artifact analysis_status=COMPLETE" in stderr
    assert "it cannot bypass completeness" in stderr
    assert "h1_verdict in {PASS, FAIL}" in stderr


def test_f05_h10_pending_rejects_null_sr_pct_with_exit_2(tmp_path, capsys):
    decision, sr, fig0c = _write_complete_pass1_artifacts(tmp_path)
    payload = json.loads(sr.read_text())
    payload["summary_table"][0]["sr_pct"] = None
    sr.write_text(json.dumps(payload))
    out = tmp_path / "must-not-exist" / "slotsheet.md"

    rc = slots.main([
        "--h10-pending",
        "--decision", str(decision),
        "--sr", str(sr),
        "--fig0c", str(fig0c),
        "--out", str(out),
    ])

    assert rc == 2
    assert not out.exists()
    stderr = capsys.readouterr().err
    assert "SR invalid numeric field" in stderr
    assert "sr_pct=None" in stderr


def test_k3_h10_pending_and_rehearsal_are_argparse_mutually_exclusive(capsys):
    with pytest.raises(SystemExit) as excinfo:
        slots.main(["--h10-pending", "--rehearsal"])
    assert excinfo.value.code == 2
    assert "not allowed with argument" in capsys.readouterr().err


def test_k2_f1_out_writes_scratch_help_exists_and_default_is_canonical(tmp_path):
    expected = REPO / "results/phantom_paper/figures/fig_f1_diamond_schematic"
    assert f1.OUT == expected
    assert f1.build_parser().parse_args([]).out == expected

    out = tmp_path / "nested" / "custom_f1"
    assert f1.main(["--out", str(out)]) == 0
    assert out.with_suffix(".png").is_file()
    assert out.with_suffix(".pdf").is_file()

    help_result = subprocess.run(
        [sys.executable, str(REPO / "scripts/analysis/figures/fig_f1_diamond_schematic.py"),
         "--help"],
        cwd=REPO, text=True, capture_output=True,
    )
    assert help_result.returncode == 0
    assert "--out BASENAME" in help_result.stdout
