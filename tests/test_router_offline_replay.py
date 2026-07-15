from __future__ import annotations

import pytest

from scripts.analysis.router_offline_replay import (
    DISCLAIMER,
    DISPLAY_MODES,
    fold_map_sha256,
    normalize_mode,
    policy_metrics,
    reference_points,
    render_markdown,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("DOM", "dom"),
        ("SoM", "som"),
        ("P-text", "phantom_text"),
        ("P-prompt", "phantom_prompt"),
        ("P-SoM", "phantom_som"),
    ],
)
def test_normalize_manifest_modes(raw, expected):
    assert normalize_mode(raw) == expected


def _matrix():
    return {
        1: {
            mode: {"success": mode in {"dom", "som"}, "cost_usd": float(i + 1)}
            for i, mode in enumerate(DISPLAY_MODES)
        },
        2: {
            mode: {"success": mode == "som", "cost_usd": float(i + 2)}
            for i, mode in enumerate(DISPLAY_MODES)
        },
        3: {
            mode: {"success": False, "cost_usd": float(i + 3)}
            for i, mode in enumerate(DISPLAY_MODES)
        },
    }


def test_policy_metrics_replays_selected_outcome_and_cost():
    got = policy_metrics(_matrix(), {1: "dom", 2: "som", 3: "phantom_som"})
    assert got["n_success"] == 2
    assert got["success_rate"] == pytest.approx(2 / 3)
    assert got["mean_total_billed_cost_usd"] == pytest.approx((1 + 3 + 8) / 3)


def test_reference_points_include_best_single_oracle_and_psom():
    got = reference_points(_matrix())
    assert got["best_single_mode"]["mode"] == "som"
    assert got["best_single_mode"]["success_rate"] == pytest.approx(2 / 3)
    assert got["six_mode_oracle_ceiling"]["success_rate"] == pytest.approx(2 / 3)
    assert got["six_mode_oracle_ceiling"]["n_no_success_tasks"] == 1
    assert got["always_p_som"] == got["single_modes"]["phantom_som"]


def test_fold_map_sha_is_order_and_key_type_stable():
    assert fold_map_sha256({"2": 1, "1": 0}) == fold_map_sha256({1: 0, 2: 1})


def test_markdown_starts_with_non_gate_banner():
    payload = {
        "inputs": {
            "run_manifest": "results/phantom_paper/run_manifest.yaml",
            "run_manifest_sha256": "abc",
            "artifacts_dir": "results/phantom_paper/offline",
        },
        "cells": {},
    }
    assert render_markdown(payload).startswith(f"# {DISCLAIMER}\n")
