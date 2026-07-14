"""Regression locks for day-audit Round A findings F-01/02/03/04/06/10/12."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from scripts.analysis import aggregate_phase1_full_prereg_decision as full
from scripts.analysis import aggregate_phase1_prereg_gate as legacy
from scripts.analysis.figures import fig3_regional_carbon as fig3
from scripts.analysis.lib.canonical_cells import PHASE_1A_PLANNED_CELLS
from scripts.analysis.lib.episode_rows import load_task_rows


REPO = Path(__file__).resolve().parents[1]


def _bootstrap_h1_row(cell: dict, n: int, only_count: int) -> dict:
    diff = np.zeros(n, dtype=np.int8)
    if only_count:
        diff[-only_count:] = 1
    rng = np.random.default_rng(legacy.PREREG_SEED)
    boot = np.empty(legacy.PREREG_B)
    for index in range(legacy.PREREG_B):
        sample = rng.integers(0, n, size=n)
        boot[index] = 100.0 * float(diff[sample].mean())
    return {
        "baseline": cell["baseline"],
        "site": cell["site"],
        "complete_exact": True,
        "expected_n": n,
        "observed_n": {mode: n for mode in legacy.SIX_MODES},
        "missing_ids": {mode: [] for mode in legacy.SIX_MODES},
        "extra_ids": {mode: [] for mode in legacy.SIX_MODES},
        "task_set_sha256": "synthetic-divergence",
        "n_tasks": n,
        "theta_pp": 100.0 * only_count / n,
        "se_pp": float(boot.std(ddof=1)),
        "ci95_lo_pp": float(np.quantile(boot, 0.025)),
        "ci95_hi_pp": float(np.quantile(boot, 0.975)),
        "boot_pp": boot.astype(np.float32),
        "oracle_6_pp": 100.0,
        "oracle_5_no_psom_pp": 100.0 - 100.0 * only_count / n,
        "n_psom_only": only_count,
    }


def test_f01_legacy_field_is_transparency_only_and_full_uses_bootstrap(monkeypatch):
    """A feasible normal-Z PASS/bootstrap FAIL publishes one canonical verdict."""
    # Multiset matches the audit witness (0,0,4,7,7,8); assignment preserves
    # the canonical 3×N=224 + 3×N=205 site topology and creates the divergence.
    only_counts = {
        ("classifieds", "B0"): 0,
        ("classifieds", "B1"): 0,
        ("classifieds", "B2"): 7,
        ("reddit", "B0"): 4,
        ("reddit", "B1"): 7,
        ("reddit", "B2"): 8,
    }
    cells = [
        {"site": site, "baseline": baseline, "modes": {}}
        for site, baseline in PHASE_1A_PLANNED_CELLS
    ]
    rows = {
        (cell["site"], cell["baseline"]): _bootstrap_h1_row(
            cell,
            224 if cell["site"] == "classifieds" else 205,
            only_counts[(cell["site"], cell["baseline"])],
        )
        for cell in cells
    }

    def fake_h1(cell, **_kwargs):
        return rows[(cell["site"], cell["baseline"])]

    monkeypatch.setattr(legacy, "_cell_drop_one_theta_se", fake_h1)
    monkeypatch.setattr(full, "_cell_drop_one_theta_se", fake_h1)
    monkeypatch.setattr(full, "_load_cell_per_task", lambda _cell, **_kwargs: {})

    transparency = legacy.build_gate(cells)
    canonical = full.build_full_decision(cells)

    assert transparency["pooled_fe"]["gate_passed"] is True
    assert transparency["h1_verdict_normal_approx_transparency"] == "PASS"
    assert "h1_verdict" not in transparency
    assert canonical["pooled_h1_bootstrap"]["gate_passed_bootstrap"] is False
    assert canonical["h1_verdict"] == "FAIL"


def _write_summary(path: Path, *, filename_id: int, payload_id: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / f"task_{filename_id}_summary_v2.json").write_text(
        json.dumps({
            "schema_version": "2.0",
            "task_id": payload_id,
            "success": False,
        }),
        encoding="utf-8",
    )


def test_f02_filename_payload_task_id_mismatch_hard_fails_h1_and_full(tmp_path):
    modes: dict[str, Path] = {}
    for mode in legacy.SIX_MODES:
        modes[mode] = tmp_path / mode
        _write_summary(modes[mode], filename_id=42, payload_id=43)
    cell = {"baseline": "B0", "site": "classifieds", "modes": modes}

    with pytest.raises(ValueError, match="filename task_id=42, payload task_id=43"):
        legacy._cell_drop_one_theta_se(cell, expected_ids={42})
    with pytest.raises(ValueError, match="filename task_id=42, payload task_id=43"):
        full.build_full_decision([cell], expected_ids_by_site={"classifieds": {42}})


def test_f02_duplicate_logical_task_id_is_always_a_hard_error(tmp_path):
    episodes = tmp_path / "episodes"
    _write_summary(episodes, filename_id=42, payload_id=42)
    (episodes / "classifieds_task_42_summary_v2.json").write_text(
        json.dumps({"schema_version": "2.0", "task_id": 42, "success": True}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate logical episode summary task_id=42"):
        load_task_rows(episodes)


def test_f04_second_destination_replace_failure_rolls_back_all_three(monkeypatch, tmp_path):
    destinations = [tmp_path / name for name in ("decision.csv", "decision.json", "decision.md")]
    for destination in destinations:
        destination.write_text("OLD\n", encoding="utf-8")

    real_replace = os.replace
    final_destinations = {path.resolve() for path in destinations}
    commit_replace_count = 0
    injected = False

    def fail_second_commit_replace(src, dst):
        nonlocal commit_replace_count, injected
        if Path(dst).resolve() in final_destinations:
            commit_replace_count += 1
            if commit_replace_count == 2 and not injected:
                injected = True
                raise OSError("synthetic second destination replace failure")
        return real_replace(src, dst)

    monkeypatch.setattr(full.os, "replace", fail_second_commit_replace)
    with pytest.raises(OSError, match="second destination replace"):
        full.write_outputs_atomic(
            {
                "per_cell": [],
                "skipped_cells": [],
                "gate_status": "PARTIAL_DATA",
                "gate_status_reason": "synthetic",
                "analysis_status": "PARTIAL",
                "h1_verdict": "NOT_EVALUATED",
                "h2a_margin_pct": 20.0,
            },
            *destinations,
        )
    assert [path.read_text(encoding="utf-8") for path in destinations] == ["OLD\n"] * 3
    assert not list(tmp_path.glob("*.staged"))
    assert not list(tmp_path.glob("*.backup"))


@pytest.mark.skipif(
    any(shutil.which(command) is None for command in (
        "pandoc", "latexmk", "pdflatex", "bibtex", "pdfinfo", "perl", "awk", "rg"
    )),
    reason="full zero-TODO converter regression requires the local LaTeX toolchain",
)
def test_f03_zero_todo_submission_fixture_runs_full_conversion_chain(tmp_path):
    latex_dir = tmp_path / "docs/checkpoints/paper_drafts/aaai27/latex"
    latex_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    shutil.copy2(
        REPO / "docs/checkpoints/paper_drafts/aaai27/latex/convert.sh",
        latex_dir / "convert.sh",
    )
    shutil.copy2(
        REPO / "docs/checkpoints/paper_drafts/aaai27/latex/skeleton.tex",
        latex_dir / "skeleton.tex",
    )

    source = tmp_path / "docs/checkpoints/paper_drafts/aaai27/aaai27_main.md"
    source.write_text(
        """---
title: "Zero TODO Fixture"
---

# Abstract

This sanitized fixture contains no residual slots.

# Introduction

The conversion chain is exercised end to end.

| arm | value |
|---|---:|
| A | 1 |

*Table 1: First fixture table.*

## Setup

| arm | value |
|---|---:|
| B | 2 |

*Table 2: Second fixture table.*

# Results I: The Phenomenon

The strict submission path remains compileable.

| arm | value |
|---|---:|
| C | 3 |

*Table 3: Third fixture table.*

## Discussion

| arm | value |
|---|---:|
| D | 4 |

*Table 4: Fourth fixture table.*
""",
        encoding="utf-8",
    )
    (tmp_path / "docs/checkpoints/paper_drafts/paper.bib").write_text(
        "@article{fixture, title={Fixture}, author={Anonymous}, year={2026}}\n",
        encoding="utf-8",
    )
    figures = tmp_path / "results/phantom_paper/figures"
    figures.mkdir(parents=True)
    for name in ("fig_f1_diamond_schematic.pdf", "fig_f2_h1_forest.pdf"):
        shutil.copy2(REPO / "results/phantom_paper/figures" / name, figures / name)

    result = subprocess.run(
        ["bash", str(latex_dir / "convert.sh"), "--submission"],
        cwd=latex_dir,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "Visible TODO slots: 0" in result.stdout
    assert (latex_dir / "build/main.pdf").is_file()


def test_f10_consumers_reference_canonical_sr_artifact_only():
    layered = (REPO / "scripts/analysis/layered_status.py").read_text(encoding="utf-8")
    mechanism = (REPO / "scripts/analysis/mechanism_per_task.py").read_text(encoding="utf-8")
    assert "sr_per_mode.json" in layered
    assert "sr_per_mode.md" in layered
    assert "sr_fp_per_mode" not in layered
    assert "sr_per_mode.json" in mechanism
    assert "sr_fp_per_mode" not in mechanism
    assert "fp_cross_reference" not in mechanism


def test_f12_b2_caption_is_parameterized_without_b1_lower_bound_claim():
    b1 = fig3.caption_for_baseline("B1")
    b2 = fig3.caption_for_baseline("B2")
    assert "B1 measurement serves as a lower-bound reference" in b1
    assert "B2" in b2
    assert "B1" not in b2
    assert "lower-bound reference" not in b2
