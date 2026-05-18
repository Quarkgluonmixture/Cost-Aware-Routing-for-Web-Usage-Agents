"""Invariant tests for /stress A2.1 phantom-framing fixes (B-892~B-900, 2026-05-17).

Covers:
  - B-893 P0-2: permutation_drop_one_null fixed-marginal null + drop-one
    excess + p-value reproducibility under fixed seed
  - B-896 P0-5: stratified_bootstrap_lift_ci stratum-resample contract +
    fallback to standard bootstrap when only one stratum observed
  - Paper §1 prose pointer invariants (axis-based boundary,
    [^null-framing] footnote, FrugalGPT/RouteLLM preempt, two-knob
    hypothesis-generation wording)
  - Paper §2 lexical-leakage memory-rule compliance (no metadata-rich /
    compact framing)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

from aggregate_phantom_lift import (  # noqa: E402
    bootstrap_lift_ci,
    permutation_drop_one_null,
    stratified_bootstrap_lift_ci,
)


PAPER_DRAFTS = REPO / "docs" / "checkpoints" / "paper_drafts"
SECTION1 = PAPER_DRAFTS / "section1_intro.md"
SECTION2 = PAPER_DRAFTS / "section2_background.md"


# ─────────────────────────── B-893 permutation null ──────────────────────


def _make_arms(n: int, marginals: dict, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for arm, k in marginals.items():
        v = np.zeros(n, dtype=np.int8)
        idx = rng.choice(n, k, replace=False)
        v[idx] = 1
        out[arm] = v
    return out


def test_permutation_null_returns_required_keys():
    arms = _make_arms(100, {"DOM": 20, "SoM": 15, "Vision": 5, "P-SoM": 18})
    res = permutation_drop_one_null(arms, drop_arm="P-SoM", B_perm=500, seed=42)
    required = {
        "observed_drop_one_pp", "null_p50", "null_p95", "null_p99",
        "excess_over_null_p95_pp", "p_value_one_sided", "B_perm",
        "marginal_counts", "n_tasks", "drop_arm",
    }
    assert required.issubset(res.keys())
    assert res["B_perm"] == 500
    assert res["n_tasks"] == 100
    assert res["drop_arm"] == "P-SoM"
    assert res["marginal_counts"]["P-SoM"] == 18


def test_permutation_null_seed_reproducible():
    arms = _make_arms(150, {"DOM": 30, "SoM": 25, "Vision": 8, "P-SoM": 20})
    r1 = permutation_drop_one_null(arms, drop_arm="P-SoM", B_perm=500, seed=42)
    r2 = permutation_drop_one_null(arms, drop_arm="P-SoM", B_perm=500, seed=42)
    assert r1["null_p95"] == r2["null_p95"]
    assert r1["excess_over_null_p95_pp"] == r2["excess_over_null_p95_pp"]
    assert r1["p_value_one_sided"] == r2["p_value_one_sided"]


def test_permutation_null_p_one_sided_in_unit_interval():
    arms = _make_arms(80, {"DOM": 10, "SoM": 8, "Vision": 2, "P-SoM": 12})
    res = permutation_drop_one_null(arms, drop_arm="P-SoM", B_perm=300, seed=42)
    assert 0.0 <= res["p_value_one_sided"] <= 1.0


def test_permutation_null_independent_random_arms_yield_p_near_uniform():
    arms = _make_arms(200, {"DOM": 40, "SoM": 30, "Vision": 8, "P-SoM": 35}, seed=99)
    res = permutation_drop_one_null(arms, drop_arm="P-SoM", B_perm=1000, seed=42)
    # Under independent random marginals, observed drop-one should land near
    # the null median — p-value should not be extreme on either tail.
    assert 0.05 < res["p_value_one_sided"] < 0.95


def test_permutation_null_rejects_missing_drop_arm():
    arms = _make_arms(50, {"DOM": 10, "SoM": 5, "Vision": 2, "P-SoM": 8})
    with pytest.raises(ValueError, match="drop_arm"):
        permutation_drop_one_null(arms, drop_arm="P-MISSING", B_perm=10, seed=42)


def test_permutation_null_rejects_mismatched_arm_lengths():
    arms = {
        "DOM": np.zeros(50, dtype=np.int8),
        "SoM": np.zeros(40, dtype=np.int8),  # mismatched
    }
    with pytest.raises(ValueError, match="mismatched"):
        permutation_drop_one_null(arms, drop_arm="SoM", B_perm=10, seed=42)


def test_permutation_null_zero_marginal_handled():
    arms = {
        "DOM": np.ones(50, dtype=np.int8),
        "SoM": np.zeros(50, dtype=np.int8),  # zero pass count
        "Vision": np.zeros(50, dtype=np.int8),
        "P-SoM": np.ones(50, dtype=np.int8),
    }
    res = permutation_drop_one_null(arms, drop_arm="P-SoM", B_perm=200, seed=42)
    # With zero-pass arms, drop-one of P-SoM with DOM={1..50} and P-SoM={1..50}
    # union = full universe, union-without-P-SoM still = full universe (DOM
    # alone covers everything) → observed drop-one = 0.
    assert res["observed_drop_one_pp"] == 0.0
    assert res["marginal_counts"]["SoM"] == 0


# ─────────────────────── B-896 stratified bootstrap ──────────────────────


def test_stratified_bootstrap_returns_ordered_ci():
    n = 200
    rng = np.random.default_rng(11)
    in_3 = (rng.random(n) < 0.30).astype(int)
    in_5 = (rng.random(n) < 0.45).astype(int)
    strata = np.array(["cls"] * 100 + ["red"] * 100)
    ci_lo, ci_hi = stratified_bootstrap_lift_ci(in_3, in_5, strata, B=300, seed=42)
    assert ci_lo <= ci_hi


def test_stratified_bootstrap_seed_reproducible():
    n = 120
    rng = np.random.default_rng(13)
    in_3 = (rng.random(n) < 0.25).astype(int)
    in_5 = (rng.random(n) < 0.40).astype(int)
    strata = np.array(["fam_A"] * 60 + ["fam_B"] * 60)
    r1 = stratified_bootstrap_lift_ci(in_3, in_5, strata, B=400, seed=42)
    r2 = stratified_bootstrap_lift_ci(in_3, in_5, strata, B=400, seed=42)
    assert r1 == r2


def test_stratified_bootstrap_single_stratum_falls_back():
    n = 100
    rng = np.random.default_rng(17)
    in_3 = (rng.random(n) < 0.20).astype(int)
    in_5 = (rng.random(n) < 0.30).astype(int)
    strata_single = np.array(["only"] * n)
    ci_strat = stratified_bootstrap_lift_ci(in_3, in_5, strata_single, B=500, seed=42)
    ci_paired = bootstrap_lift_ci(in_3, in_5, B=500, seed=42)
    assert ci_strat == ci_paired


def test_stratified_bootstrap_rejects_length_mismatch():
    in_3 = np.zeros(50, dtype=int)
    in_5 = np.zeros(50, dtype=int)
    strata = np.zeros(40)  # mismatched
    with pytest.raises(ValueError, match="strata length"):
        stratified_bootstrap_lift_ci(in_3, in_5, strata, B=10, seed=42)


# ─────────────────── B-892 / B-894 / B-895 / B-897 / B-898 / B-899 prose pointers ──


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def test_section1_axis_based_boundary_present():
    """B-894 P0-3: §1 paragraph 1 uses axis-based boundary formulation."""
    txt = _read(SECTION1)
    assert "varying text-payload format" in txt
    assert "prompt family" in txt
    assert "DOM baseline" in txt
    assert "origin baseline" in txt or "origin baseline (axis-0" in txt


def test_section1_phantom_boundary_formulation_footnote_present():
    """B-894 P0-3: ^phantom-boundary-formulation footnote present and cross-links §2."""
    txt = _read(SECTION1)
    assert "^phantom-boundary-formulation" in txt
    assert "axis-based vs 4-corner" in txt
    assert "axis-based formulation in §1" in txt


def test_section1_drop_in_construction_vs_measured_split():
    """B-892 P0-1: §1 prose splits constructed vs empirically validated."""
    txt = _read(SECTION1)
    assert "Drop-in property — constructed vs measured" in txt
    assert "architectural consequences of the image-off boundary" in txt
    assert "empirically validate two non-trivial signal properties" in txt


def test_section1_task_resampling_ci_label():
    """B-896 P0-5: 'task-resampling 95% CI' label in §1 hero paragraph."""
    txt = _read(SECTION1)
    assert "task-resampling 95% CI" in txt
    assert "per-task paired bootstrap" in txt


def test_section1_null_framing_footnote_present():
    """B-893 P0-2: ^null-framing footnote explains permutation null."""
    txt = _read(SECTION1)
    assert "^null-framing" in txt
    assert "fixed-marginal permutation reference" in txt
    assert "aggregate_phantom_lift.py --permute-marginal-null" in txt


def test_section1_jaccard_null_reference_present():
    """B-897 P1-6: Jaccard prose includes random-independent baseline + unique-pass count."""
    txt = _read(SECTION1)
    assert "3-5× higher than the random-independent baseline" in txt
    assert "E[J] ≈ 0.06-0.10" in txt
    assert "residual unique-pass set" in txt


def test_section1_oracle_vs_realized_router_separation():
    """B-895 P0-4: §1 explicitly separates oracle ceiling from realized router lift."""
    txt = _read(SECTION1)
    assert "oracle ceiling, not a realized router gain" in txt
    assert "router classification overhead and misrouting regret" in txt
    assert "constructed" in txt and "substrate" in txt


def test_section1_cascade_lit_preempt_present():
    """B-898 P1-7: FrugalGPT/RouteLLM preempt sentence in §1 contribution 3."""
    txt = _read(SECTION1)
    assert "FrugalGPT" in txt
    assert "RouteLLM" in txt
    assert "model-routing systems" in txt or "model routing" in txt


def test_section1_two_knob_hypothesis_generation_downgrade():
    """B-899 P1-8: two-knob claim downgraded to hypothesis-generation observation."""
    txt = _read(SECTION1)
    assert "hypothesis-generation observation" in txt
    assert "appears to shape exploration" in txt
    assert "appears to modulate commitment timing" in txt
    # Wording refined by A2.6a Chunk 1 (3ade9b6) + A2.5 Chunk D — tolerate both.
    assert (
        "archive-grade evidence pending Phase 1a clean-rerun confirmation" in txt
        or "archive-grade evidence pending Phase 1a confirmation" in txt
    )


# ─────────────────── B-900 §2 lexical-leakage memory rule ────────────────


def test_section2_no_metadata_rich_compact_framing():
    """B-900 P1-9: §2 no longer uses 'metadata-rich vs compact' framing."""
    txt = _read(SECTION2)
    forbidden_patterns = [
        r"\bmetadata-rich\b",
        r"\bcompact\b",  # in context of [SOM_MARKS]
    ]
    for pat in forbidden_patterns:
        # `compact` may appear in unrelated context; gate on prox to '[SOM_MARKS]'
        if pat == r"\bcompact\b":
            # Allow if not in same paragraph as SOM_MARKS / AXTree dichotomy
            for line in txt.splitlines():
                if "compact" in line and ("[SOM_MARKS]" in line or "AXTree" in line):
                    if "metadata" in line or "flat" in line:
                        pytest.fail(
                            f"§2 still uses 'compact' in AXTree/SOM_MARKS dichotomy: {line[:200]}"
                        )
        else:
            assert not re.search(pat, txt), f"§2 still contains {pat!r}"


def test_section2_preserves_hierarchical_nesting_framing():
    """B-900 P1-9: §2 new framing uses structural-format language."""
    txt = _read(SECTION2)
    assert "same element semantics" in txt
    assert "preserves hierarchical nesting" in txt
    assert "flattened into a sequential indexed list" in txt
    assert "altering the element-level semantic content" in txt or \
           "without altering the element-level semantic content" in txt


# ─────────────────── Cross-doc consistency ───────────────────────────────


def test_section1_strict_positive_oracle_excess_wording():
    """B-893 P0-2: §1 deployment claim uses 'oracle excess over fixed-marginal null'."""
    txt = _read(SECTION1)
    assert "drop-one oracle excess over the fixed-marginal permutation null" in txt or \
           "drop-one oracle excess over fixed-marginal permutation null" in txt


def test_section1_no_4_fold_property_unqualified():
    """B-892 P0-1: §1 paragraph 2 'empirical 4-fold drop-in property' wording retired."""
    txt = _read(SECTION1)
    # Should NOT contain the old packaging
    assert "preserve the empirical 4-fold drop-in property" not in txt
    # Should contain the new construction-vs-measurement split
    assert "By construction, phantom-space arms inherit DOM's cost profile" in txt
