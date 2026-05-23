"""AMENDMENT 03 (2026-05-24) regression: H1 PRIMARY canonical-source + SE-floor alignment.

Guards the implementation-alignment fix that:
  (1) converged paper §1 H1 hero references onto the canonical bootstrap-percentile
      producer `aggregate_phase1_full_prereg_decision` (away from the legacy normal-Z
      `aggregate_phase1_prereg_gate`);
  (2) aligned the legacy producer's SE-floor threshold to the prereg-locked 0.68pp
      Agresti-Coull anchor (prereg §2 H1 L98 + L718 B-1003), eliminating the
      `<= 0` vs `< 0.68` double-track that would split θ_FE on any cell with
      SE ∈ (0, 0.68pp).

NO estimand / gate / δ / R-ladder change — these tests only prevent the stale
implementation references + the literal-`<=0` floor drift from re-appearing.
See docs/prereg_amendments/AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524.md.
"""
from __future__ import annotations

import importlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts/analysis"

# prereg §2 H1 L98 + L718 (B-1003) Agresti-Coull anchor — the single locked value.
PREREG_SE_FLOOR_THRESHOLD_PP = 0.68
PREREG_SE_FLOOR_REPLACE_PP = 1.0


def test_legacy_gate_se_floor_matches_prereg():
    """Legacy transparency producer's module-level SE-floor == prereg-locked 0.68 / 1.0."""
    from scripts.analysis.aggregate_phase1_prereg_gate import (
        SE_FLOOR_THRESHOLD_PP,
        SE_FLOOR_REPLACE_PP,
    )
    assert SE_FLOOR_THRESHOLD_PP == PREREG_SE_FLOOR_THRESHOLD_PP, (
        "legacy gate SE-floor threshold drifted from the prereg L98 0.68pp anchor "
        "(was a literal `<= 0` pre-AMENDMENT-03 — B-1003 codified 0.68 in prose but "
        "the code never landed it)"
    )
    assert SE_FLOOR_REPLACE_PP == PREREG_SE_FLOOR_REPLACE_PP


def test_canonical_producer_se_floor_matches_legacy():
    """Canonical bootstrap producer uses the SAME 0.68 threshold (no double-track)."""
    src = (SCRIPTS / "aggregate_phase1_full_prereg_decision.py").read_text(encoding="utf-8")
    thresholds = re.findall(r"SE_FLOOR_THRESHOLD_PP\s*=\s*([0-9.]+)", src)
    assert thresholds, "canonical producer must define SE_FLOOR_THRESHOLD_PP"
    for t in thresholds:
        assert float(t) == PREREG_SE_FLOOR_THRESHOLD_PP, (
            f"canonical SE-floor threshold {t} != prereg 0.68 / legacy gate — "
            "SE-floor double-track regression (AMENDMENT 03)"
        )


def test_canonical_primary_is_bootstrap_percentile():
    """The canonical H1 PRIMARY gate producer exposes the bootstrap-percentile pool."""
    mod = importlib.import_module(
        "scripts.analysis.aggregate_phase1_full_prereg_decision"
    )
    assert hasattr(mod, "_pool_bootstrap_percentile_p"), (
        "canonical H1 primary gate must be the bootstrap-percentile test "
        "(prereg §2 H1 L98 + AMENDMENT_02 §2 line 99); the legacy normal-Z is "
        "transparency-only"
    )


def test_legacy_gate_marked_transparency_only():
    """Legacy normal-Z producer docstring must self-declare NON-CANONICAL / transparency."""
    src = (SCRIPTS / "aggregate_phase1_prereg_gate.py").read_text(encoding="utf-8")
    head = src[:2500]
    assert "TRANSPARENCY-ONLY" in head, "legacy gate must mark itself transparency-only"
    assert "NON-CANONICAL" in head
    assert "aggregate_phase1_full_prereg_decision" in head, (
        "legacy gate docstring must point to the canonical primary producer"
    )


def test_no_stale_legacy_gate_as_primary_in_figures_and_prose():
    """No figure / §1-§3 prose may cite phase1_prereg_gate as the §1 H1 PRIMARY hero.

    A line is a violation only if it ties phase1_prereg_gate to a PRIMARY/§1-hero
    claim WITHOUT a transparency/legacy/kernel qualifier on the same line (those
    qualifiers mark the intentionally-retained shared-kernel / transparency mention).
    """
    targets = [
        SCRIPTS / "figures/fig_meta_forest.py",
        SCRIPTS / "aggregate_phantom_lift.py",
        REPO / "docs/checkpoints/paper_drafts/section1_intro.md",
        REPO / "docs/checkpoints/paper_drafts/section3_definition.md",
    ]
    bad: list[str] = []
    for p in targets:
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            low = line.lower()
            if "phase1_prereg_gate" not in low and "prereg_gate.py" not in low:
                continue
            ties_primary = any(
                tok in low for tok in ("h1 primary", "§1 hero", "§1 h1 primary", "for §1 hero")
            )
            qualified = any(
                tok in low for tok in ("transparency", "legacy", "non-canonical", "kernel")
            )
            if ties_primary and not qualified:
                bad.append(f"{p.name}:{i}: {line.strip()[:120]}")
    assert not bad, "stale legacy-gate-as-§1-PRIMARY references:\n" + "\n".join(bad)
