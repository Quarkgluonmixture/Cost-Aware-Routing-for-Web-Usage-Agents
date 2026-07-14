"""Regression coverage for the 2026-07-14 analysis-pipeline hygiene fixes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_h2_fig3_partial_default_renders_available_cells(tmp_path, monkeypatch, capsys):
    from scripts.analysis.figures import fig3_regional_carbon as fig3

    episodes = tmp_path / "episodes"
    episodes.mkdir()
    (episodes / "classifieds_task_0_summary_v2.json").write_text(
        json.dumps({"total_energy_kwh": 0.01}), encoding="utf-8"
    )
    (episodes / "reddit_task_0_summary_v2.json").write_text(
        json.dumps({"total_energy_kwh": 0.02}), encoding="utf-8"
    )

    class Cell:
        episodes_dir = episodes

    def fake_get_cells(*, baseline, site, mode, grade):
        # One intentionally absent paper-grade cell exercises partial rendering.
        return [] if (site, mode) == ("reddit", "Vision") else [Cell()]

    monkeypatch.setattr(fig3, "_get_cells", fake_get_cells)
    monkeypatch.setattr(fig3, "OUT", tmp_path / "fig3.png")

    assert fig3.main([]) == 0
    assert fig3.OUT.exists()
    stdout = capsys.readouterr().out
    assert "PARTIAL/NON_PAPER_GRADE" in stdout
    assert "SKIP missing cell: B1 reddit Vision" in stdout
    assert "missing cell count: 1" in stdout


def test_h2_fig3_strict_preserves_fail_closed_behavior(monkeypatch):
    from scripts.analysis.figures import fig3_regional_carbon as fig3

    monkeypatch.setattr(fig3, "_get_cells", lambda **kwargs: [])
    with pytest.raises(RuntimeError, match="missing paper-grade cells"):
        fig3._resolve_runs(strict=True)


def test_f06_fig3_strict_rejects_one_of_expected_task_energy(tmp_path, monkeypatch):
    from scripts.analysis.figures import fig3_regional_carbon as fig3

    dirs = {}
    for site in ("classifieds", "reddit"):
        episodes = tmp_path / site / "episodes"
        episodes.mkdir(parents=True)
        (episodes / "task_0_summary_v2.json").write_text(
            json.dumps({
                "schema_version": "2.0",
                "task_id": 0,
                "success": False,
                "total_energy_kwh": 0.01,
            }),
            encoding="utf-8",
        )
        dirs[site] = episodes

    class Cell:
        def __init__(self, episodes_dir):
            self.episodes_dir = episodes_dir

    def fake_get_cells(*, baseline, site, mode, grade):
        return [Cell(dirs[site])]

    monkeypatch.setattr(fig3, "_get_cells", fake_get_cells)
    monkeypatch.setattr(fig3, "OUT", tmp_path / "strict.png")

    with pytest.raises(RuntimeError, match="strict task-ID set mismatch"):
        fig3.main(["--strict"])
    assert not fig3.OUT.exists()


def test_h3_partial_mean_jaccard_is_skipped_without_losing_available_cell(capsys):
    from scripts.analysis import mechanism_per_task as mechanism

    e1 = {
        "reddit": {
            "compound_DOM_to_PSoM": {
                "n": 0,
                "skipped": True,
                "left_mode": "DOM",
                "right_mode": "P-SoM",
            }
        },
        "classifieds": {
            "compound_DOM_to_PSoM": {
                "n": 1,
                "mean_jaccard": 0.25,
                "left_mode": "DOM",
                "right_mode": "P-SoM",
            }
        },
    }
    e2 = {
        site: {"DOM_vs_P-SoM": {"early_divergence_rate": None}}
        for site in ("reddit", "classifieds")
    }
    e3 = {
        "cells": {
            "B0/classifieds/DOM": {
                "AUROC_token": 0.6,
                "AUROC_verbal": None,
                "AUROC_behavioral_max": None,
            }
        }
    }
    e4 = {"axis_contrasts": {"reddit": {}, "classifieds": {}}}

    assert mechanism.log_missing_e1_cells(e1) == 1
    implications = mechanism.headline_implications(e1, e2, e3, e4)

    assert "0.250 on classifieds" in implications["E1_headline"]
    stdout = capsys.readouterr().out
    assert "SKIP E1 reddit/compound_DOM_to_PSoM: missing mean_jaccard" in stdout
    assert "SKIP summary: 1 E1 cell(s) missing mean_jaccard" in stdout


def test_h3_axis1_bad_task_input_is_skipped(tmp_path, monkeypatch, capsys):
    from scripts.analysis import axis1_microbehavior as axis1

    episodes = tmp_path / "episodes"
    episodes.mkdir()
    steps = episodes / "reddit_task_149_steps_v2.jsonl"
    steps.write_text("{}\n", encoding="utf-8")
    monkeypatch.setitem(axis1.STEP_DIRS, "B0", {"reddit": {"DOM": episodes}})
    monkeypatch.setattr(axis1, "read_steps", lambda path: (_ for _ in ()).throw(ValueError("identity mismatch")))
    axis1.SKIPPED_TASK_INPUTS.clear()

    result = axis1.per_task_mode_metrics("B0", "reddit", "DOM", {})

    assert result == {}
    assert axis1.SKIPPED_TASK_INPUTS == ["B0/reddit/DOM/task_149"]
    assert "SKIP B0/reddit/DOM/task_149: identity mismatch" in capsys.readouterr().out


def test_partial_cross_site_caption_formats_none_as_na():
    from scripts.analysis.figures.fig2e_cross_site_validity import fmt_ratio

    assert fmt_ratio(None) == "n/a"
    assert fmt_ratio(float("nan")) == "n/a"
    assert fmt_ratio(1.234) == "1.23"
