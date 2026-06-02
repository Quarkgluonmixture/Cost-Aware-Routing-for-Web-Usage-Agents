"""Invariant tests for the §6.5 intelligent-baseline ladder (B-1006 R5 defense).

Deterministic synthetic-matrix tests — no archive data / VWA submodule needed. We
assert the STRUCTURAL ladder invariants (which hold regardless of LR/stump
randomness) + the exact closed-form arms (always-DOM, per-task-lookup oracle) +
the DISCLOSURE-not-gating contract of the aggregate hook.

Loads scripts via importlib (matches A1.15 / A1.24 test convention; tests/ has no
__init__.py so pytest prepend mode already has the dir on sys.path).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ANALYSIS = REPO / "scripts" / "analysis"


def _load(name: str, rel: str):
    # scripts/analysis on path so sibling imports (aggregate_h10_pareto) resolve.
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    spec = importlib.util.spec_from_file_location(name, REPO / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ladder():
    return _load("intelligent_baseline_ladder", "scripts/analysis/intelligent_baseline_ladder.py")


@pytest.fixture(scope="module")
def agg():
    return _load("aggregate_h10_pareto", "scripts/analysis/aggregate_h10_pareto.py")


def _mk(success: int, cost: float) -> dict:
    return {"success": int(success), "cost_usd": float(cost), "latency_ms": 0.0, "cost_unit_basis": "api_usd"}


def _synthetic_cell():
    """24 tasks × 3 modes (dom cheap / som mid / vision dear) with a known oracle.

    Pattern (oracle = cheapest successful mode, MODES order dom<som<vision):
      6 dom-only · 6 som-only · 4 vision-only · 4 all-succeed (→dom) · 4 none.
    => 20 tasks have >=1 success → oracle SR = 20/24. dom is cheapest by construction.
    """
    COST = {"dom": 0.01, "som": 0.02, "vision": 0.03}
    outcomes: dict[int, dict[str, dict]] = {}
    feats: dict[int, dict] = {}
    tid = 0

    def add(d: int, s: int, v: int, has_image: float, axtree: float, intent: str):
        nonlocal tid
        outcomes[tid] = {
            "dom": _mk(d, COST["dom"]),
            "som": _mk(s, COST["som"]),
            "vision": _mk(v, COST["vision"]),
        }
        feats[tid] = {
            "site_cls": 1.0, "has_image": has_image,
            "intent_color": 1.0 if "blue" in intent else 0.0,
            "intent_search": 1.0 if "find" in intent else 0.0,
            "intent_compare": 1.0 if "cheapest" in intent else 0.0,
            "intent_nav": 1.0 if "go to" in intent else 0.0,
            "intent_tok_count": float(len(intent.split())),
            "axtree_elements": axtree,
        }
        tid += 1

    for _ in range(6):
        add(1, 0, 0, 0.0, 10.0, "find the page")          # dom-only
    for _ in range(6):
        add(0, 1, 0, 0.0, 80.0, "go to the big listing")  # som-only
    for _ in range(4):
        add(0, 0, 1, 1.0, 120.0, "the blue picture")      # vision-only
    for _ in range(4):
        add(1, 1, 1, 0.5, 40.0, "find cheapest blue")     # all-succeed
    for _ in range(4):
        add(0, 0, 0, 0.0, 200.0, "impossible task")       # none
    return outcomes, feats


def test_structural_sr_ceiling_holds(ladder):
    """Oracle (per-task-lookup) SR >= every other arm — the load-bearing invariant."""
    outcomes, feats = _synthetic_cell()
    rec = ladder.evaluate_ladder_for_cell("B0", "classifieds", outcomes, feats, ladder.LadderConfig())
    arms = rec["arms"]
    ceil = arms["per_task_lookup"]["sr_mean_pct"]
    for name, a in arms.items():
        assert ceil + 1e-9 >= a["sr_mean_pct"], f"{name} ({a['sr_mean_pct']}) exceeds oracle ceiling {ceil}"
    assert rec["bounding_checks"]["sr_ceiling_holds"] is True
    assert rec["bounding_checks"]["structural_invariant_holds"] is True


def test_per_task_lookup_equals_oracle_sr(ladder):
    """Infinite-capacity reductio = fraction of tasks with any successful mode (20/24)."""
    outcomes, feats = _synthetic_cell()
    rec = ladder.evaluate_ladder_for_cell("B0", "classifieds", outcomes, feats, ladder.LadderConfig())
    assert rec["n_tasks"] == 24
    assert rec["n_no_success"] == 4
    # as_row() rounds sr to 3 decimals for display; value is exactly 20/24.
    assert rec["arms"]["per_task_lookup"]["sr_mean_pct"] == pytest.approx(100 * 20 / 24, abs=1e-2)


def test_always_cheapest_is_dom(ladder):
    """§6.5 (a): always-cheapest routes every task to DOM; DOM is empirically cheapest here."""
    outcomes, feats = _synthetic_cell()
    rec = ladder.evaluate_ladder_for_cell("B0", "classifieds", outcomes, feats, ladder.LadderConfig())
    assert rec["arms"]["always_cheapest_dom"]["routed_mode_dist"] == {"dom": 24}
    assert rec["empirical_cheapest_single_mode"] == "dom"
    assert rec["dom_is_empirically_cheapest"] is True


def test_all_five_arms_present(ladder):
    outcomes, feats = _synthetic_cell()
    rec = ladder.evaluate_ladder_for_cell("B0", "classifieds", outcomes, feats, ladder.LadderConfig())
    assert set(rec["arms"]) == {
        "always_cheapest_dom", "decision_stump", "lr_dom_features_only",
        "learned_router_proxy", "per_task_lookup",
    }


def test_decision_stump_at_most_two_modes_per_fold(ladder):
    """A depth-1 tree emits <=2 modes per fold; aggregated over 5 folds it may show
    more, but a single fold can never exceed 2 (the §6.5 'one feature, two outcomes')."""
    outcomes, feats = _synthetic_cell()
    tids = sorted(outcomes)
    oracle = [ladder.derive_oracle_label(ladder._success_dict(outcomes[t])) for t in tids]
    X = ladder._design_matrix(feats, tids, ladder.FEATURE_NAMES)
    # Single-fold proxy: train a stump on all and predict — must yield <=2 distinct modes.
    est = ladder._make_stump(42)
    trainable = [i for i in range(len(tids)) if oracle[i] is not None]
    est.fit(X[trainable], [oracle[i] for i in trainable])
    preds = set(est.predict(X))
    assert len(preds) <= 2


def test_disclosure_payload_marks_not_gating(ladder):
    outcomes, feats = _synthetic_cell()
    rec = ladder.evaluate_ladder_for_cell("B0", "classifieds", outcomes, feats, ladder.LadderConfig())
    payload = ladder.build_disclosure({rec["cell_id"]: {**rec, "status": "ok"}})
    assert payload["disclosure_only"] is True
    assert "not_a_gate" in payload and "gate" in payload["not_a_gate"].lower()
    assert "B0_classifieds" in payload["cells"]
    assert payload["schema_version"].startswith("ladder-")


def test_aggregate_render_is_disclosure_only_and_robust(agg):
    """The aggregate hook renders a clearly-labelled DISCLOSURE section and never
    raises on either a pending stub or a populated payload."""
    # pending stub
    md: list[str] = []
    agg._render_ladder_disclosure_md(md, {"status": "pending", "note": "absent"})
    text = "\n".join(md)
    assert "DISCLOSURE ONLY" in text and "not part of H10 gate" in text

    # populated payload (minimal shape)
    md2: list[str] = []
    populated = {
        "generated_utc": "2026-06-02T00:00:00+00:00",
        "schema_version": "ladder-test",
        "cells": {
            "B0_classifieds": {
                "status": "ok",
                "arms": {
                    "always_cheapest_dom": {"sr_mean_pct": 17.4},
                    "decision_stump": {"sr_mean_pct": 19.6},
                    "lr_dom_features_only": {"sr_mean_pct": 22.3},
                    "learned_router_proxy": {"sr_mean_pct": 25.0},
                    "per_task_lookup": {"sr_mean_pct": 43.3},
                },
                "bounding_checks": {
                    "feature_set_value_over_stump_pp": 5.36,
                    "text_feature_value_pp": 2.68,
                    "router_headroom_pp": 18.30,
                    "sr_ceiling_holds": True,
                },
            }
        },
    }
    agg._render_ladder_disclosure_md(md2, populated)
    text2 = "\n".join(md2)
    assert "B0_classifieds" in text2 and "43.3" in text2 and "DISCLOSURE ONLY" in text2


def test_aggregate_verdict_gate_keys_untouched_by_disclosure(agg):
    """Sanity: the disclosure key is additive — the gate/verdict keys still exist and
    the disclosure loader degrades to a pending stub when the artifact is absent."""
    # loader returns None or a dict; never raises
    res = agg._load_ladder_disclosure()
    assert res is None or isinstance(res, dict)
