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


# ---------------------------------------------------------------------------
# B-1898 (2026-07-27): prereg PROSE ↔ code constant, not just code ↔ code.
#
# The 0.68pp trigger threshold was codified 2026-05-18 (B-1003) and the code was
# aligned 2026-05-24 (AMENDMENT_03), but preregistration.md §2 H1 L103-111 kept
# describing the superseded "SE = 0 exactly" rule until 2026-07-27.  A /stress
# pass then read that stale paragraph, recomputed θ_FE = 0.6533pp under it, and
# reported the IMPLEMENTATION as deviating from the preregistration — when in
# fact the prose was the stale side.  A reviewer doing the same thing gets a
# number that does not match the paper.
#
# These tests make the two surfaces fail together instead of drifting apart.
# ---------------------------------------------------------------------------

PREREG = REPO / "docs/checkpoints/pre_run/preregistration.md"


def _se_floor_paragraph() -> str:
    text = PREREG.read_text(encoding="utf-8")
    start = text.index("**Degenerate-cell SE floor protocol")
    end = text.index("**Heterogeneity", start)
    return text[start:end]


def test_prereg_se_floor_prose_states_the_locked_threshold():
    """The paragraph must name 0.68pp as the TRIGGER, matching the code."""
    from scripts.analysis.aggregate_phase1_full_prereg_decision import (
        SE_FLOOR_THRESHOLD_PP,
    )

    para = _se_floor_paragraph()
    assert f"{SE_FLOOR_THRESHOLD_PP}pp" in para, (
        "preregistration.md's degenerate-cell paragraph no longer names the "
        f"locked trigger threshold {SE_FLOOR_THRESHOLD_PP}pp that the canonical "
        "producer implements (B-1898)."
    )


def test_prereg_se_floor_prose_does_not_reassert_the_superseded_rule():
    """`SE = 0 exactly` must not reappear as the operative trigger.

    It may still be MENTIONED (the zero case is a subset of `< 0.68pp`, and the
    2026-07-27 sync note quotes the old wording to explain what changed), but it
    must not be stated as the condition under which the floor fires.
    """
    para = _se_floor_paragraph()
    banned = (
        "when a cell's paired bootstrap SE_i = 0 exactly",
        "SE floor fires only when `(ses <= 0)",
    )
    hits = [b for b in banned if b in para]
    assert not hits, (
        "the superseded `SE = 0 exactly` trigger wording is back in the prereg "
        f"paragraph: {hits} — that is the exact drift B-1898 was filed for."
    )


def test_both_producers_still_mirror_one_threshold():
    """Canonical and transparency producers must not re-split (AMENDMENT_03)."""
    from scripts.analysis.aggregate_phase1_full_prereg_decision import (
        SE_FLOOR_THRESHOLD_PP as canonical_threshold,
    )
    from scripts.analysis.aggregate_phase1_prereg_gate import (
        SE_FLOOR_REPLACE_PP as transparency_replace,
        SE_FLOOR_THRESHOLD_PP as transparency_threshold,
    )

    assert canonical_threshold == transparency_threshold == 0.68
    assert transparency_replace == 1.0
