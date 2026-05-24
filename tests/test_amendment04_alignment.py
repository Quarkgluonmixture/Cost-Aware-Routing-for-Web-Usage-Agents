"""Regression tests for AMENDMENT_04 implementation-alignment fixes.

Covers the analysis-layer fixes landed per the 3-AI /stress audit 2026-05-24:
- P1-1: post-R5 reporting route (amendment-02 §4) — `_apply_framing` emits
  `post_r5_pivot` ∈ {C_prime_structure, C_prime_router_only, F_failure, pending}.
  amendment-02 §4 line 209-210 explicitly requires these assertions.
- P1-2: B2 cross-family claim-tier downgrade (prereg §2.5 step-8 / B-1284).
- P0-2: H10 entropy DEFER gate — `label_entropy_bits` bits computation.

These gate logic paths are pre-data (R5/B2-downgrade/entropy cannot fire until
Phase 1a Pass-1+Pass-2 data lands), so the tests exercise the decision functions
directly with synthetic inputs rather than via full pipeline runs.
"""
import math

import pytest

from scripts.analysis.aggregate_phase1_full_prereg_decision import (
    _apply_framing,
    _apply_b2_cross_family_downgrade,
)
from scripts.analysis.train_l1_router import label_entropy_bits


# ---------------------------------------------------------------------------
# P1-1 — post-R5 reporting route (amendment-02 §4)
# ---------------------------------------------------------------------------

class TestPostR5Pivot:
    def test_h1_fail_h3_pass_structure(self):
        # amendment-02 §4 L209: H1 fail + H3 pass → R5 AND post_r5_pivot == C_prime_structure
        f = _apply_framing(h1_pass=False, h2a_falsified=False,
                           h3_axis1_pass=True, h3_axis2_pass=False,
                           h1_isq_cap_at_r3=False)
        assert f["rule"] == "R5"
        assert f["post_r5_pivot"] == "C_prime_structure"

    def test_h1_fail_h3_fail_h10_pass_router(self):
        # amendment-02 §4 L210: H1 fail + H3 fail + H10 pass → R5 AND post_r5_pivot == C_prime_router_only
        f = _apply_framing(False, False, False, False, False, h10_pass=True)
        assert f["rule"] == "R5"
        assert f["post_r5_pivot"] == "C_prime_router_only"

    def test_h1_fail_h3_fail_h10_fail_failure(self):
        f = _apply_framing(False, False, False, False, False, h10_pass=False)
        assert f["rule"] == "R5"
        assert f["post_r5_pivot"] == "F_failure"

    def test_h1_fail_h3_fail_h10_pending(self):
        # pre-Pass-2: H10 verdict not yet computed → pending marker, NOT a silent pass
        f = _apply_framing(False, False, False, False, False, h10_pass=None)
        assert f["rule"] == "R5"
        assert "pending_h10" in f["post_r5_pivot"]

    def test_h1_pass_no_post_r5(self):
        # H1 passes → R1, no post_r5_pivot field
        f = _apply_framing(True, False, True, True, False)
        assert f["rule"] == "R1"
        assert "post_r5_pivot" not in f

    def test_no_framing_tier_rescue(self):
        # amendment-02 anti-rescue: R5 stays R5 even when H3 passes (no R5→R3 rescue)
        f = _apply_framing(False, False, True, True, False, h10_pass=True)
        assert f["rule"] == "R5"  # NOT rescued to R1/R2/R3


# ---------------------------------------------------------------------------
# P1-2 — B2 cross-family claim-tier downgrade (prereg §2.5 step-8)
# ---------------------------------------------------------------------------

def _cells(b2_cls=True, b2_red=True, qwen_pass=True):
    """6-cell per_cell_data; ci95_lo_pp > 0 = per-cell H1 pass (CI excludes 0)."""
    def cell(b, s, passing):
        return {"baseline": b, "site": s,
                "h1": {"ci95_lo_pp": 1.5 if passing else -0.5}}
    return [
        cell("B0", "classifieds", qwen_pass), cell("B0", "reddit", qwen_pass),
        cell("B1", "classifieds", qwen_pass), cell("B1", "reddit", qwen_pass),
        cell("B2", "classifieds", b2_cls), cell("B2", "reddit", b2_red),
    ]


class TestB2CrossFamilyDowngrade:
    def test_all_pass_no_downgrade(self):
        f = _apply_b2_cross_family_downgrade({"rule": "R1"}, _cells())
        assert f["rule"] == "R1"
        assert "cross_family_override" not in f

    def test_b2_fail_downgrade_r1_to_r2(self):
        f = _apply_b2_cross_family_downgrade({"rule": "R1"}, _cells(b2_cls=False))
        assert f["rule"] == "R2"
        assert f["cross_family_override"] == "b2_nonreplication_downgrade"

    def test_b2_fail_downgrade_r2_to_r3(self):
        f = _apply_b2_cross_family_downgrade({"rule": "R2"}, _cells(b2_red=False))
        assert f["rule"] == "R3"

    def test_b2_fail_r3_stays_r3(self):
        # R3 already lowest phenomenon tier; downgrade target is R3 (no R4/R5 from B2)
        f = _apply_b2_cross_family_downgrade({"rule": "R3"}, _cells(b2_cls=False))
        assert f["rule"] == "R3"

    def test_qwen_fail_forces_r5(self):
        # prereg L412: Qwen anchor per-cell fail → R5 regardless of B2
        f = _apply_b2_cross_family_downgrade({"rule": "R1"}, _cells(qwen_pass=False))
        assert f["rule"] == "R5"
        assert f["cross_family_override"] == "qwen_anchor_fail_r5"

    def test_r5_terminal_unchanged(self):
        f = _apply_b2_cross_family_downgrade({"rule": "R5"}, _cells(b2_cls=False))
        assert f["rule"] == "R5"

    def test_r4_terminal_unchanged(self):
        f = _apply_b2_cross_family_downgrade({"rule": "R4"}, _cells(b2_cls=False))
        assert f["rule"] == "R4"

    def test_incomplete_data_no_downgrade(self):
        # missing B2 cells → None per-cell results → conservative no-change (await data)
        data = [{"baseline": "B0", "site": "classifieds",
                 "h1": {"ci95_lo_pp": 1.5}}]
        f = _apply_b2_cross_family_downgrade({"rule": "R1"}, data)
        assert f["rule"] == "R1"


# ---------------------------------------------------------------------------
# P0-2 — H10 entropy DEFER gate label-entropy computation
# ---------------------------------------------------------------------------

class TestEntropyBits:
    def test_uniform_4modes_max_entropy(self):
        assert abs(label_entropy_bits(["a", "b", "c", "d"]) - 2.0) < 1e-9

    def test_two_modes_5050_one_bit(self):
        assert abs(label_entropy_bits(["a", "a", "b", "b"]) - 1.0) < 1e-9

    def test_collapse_single_mode_zero(self):
        assert abs(label_entropy_bits(["a", "a", "a", "a"])) < 1e-9

    def test_skew_7to1_below_defer_threshold(self):
        # 7:1 concentration → ~0.544 bit < 1.0 DEFER threshold
        h = label_entropy_bits(["a"] * 7 + ["b"])
        assert h < 1.0
        assert h == pytest.approx(0.5435644, abs=1e-4)

    def test_empty_nan(self):
        assert math.isnan(label_entropy_bits([]))

    def test_three_uniform_above_threshold(self):
        # 3 modes uniform → log2(3) ≈ 1.585 bit > 1.0 → no DEFER
        assert label_entropy_bits(["a", "b", "c"]) > 1.0
