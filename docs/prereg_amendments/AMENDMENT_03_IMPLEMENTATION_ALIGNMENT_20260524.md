---
amendment_id: 03
title: Implementation-alignment of stale §1-hero references + SE-floor + cost-estimand code to DOI-1 / AMENDMENT_01 / AMENDMENT_02 (NO estimand / gate / δ / R-ladder change)
date: 2026-05-24
status: pre-fire witness (DRAFT — pending git tag + push + OSF upload); Phase 1a paper-grade outcome statistics NOT yet computed
parent_prereg: docs/checkpoints/pre_run/preregistration.md (status: locked)
parent_doi: 10.17605/OSF.IO/9QCWU   # DOI 1, pre-canonical-outcome-creation witness, 2026-05-18
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_01a_SCHEMA_VALIDATOR_20260521
  - AMENDMENT_02_GATE_LADDER_20260523
witness_tag: prereg-amendment-03-implementation-alignment-20260524   # to be created at the commit adding this file
relation: >
  Pure implementation-alignment of code + draft prose to estimands ALREADY LOCKED by
  DOI-1 + AMENDMENT_01 + AMENDMENT_02. Recorded BEFORE any Phase 1a paper-grade outcome
  statistic exists (no per-cell drop-one θ_i, no pooled θ_FE, no H10 verdict computed on
  paper-grade data) and externally witnessed before any eligible H1/H3/H10 result is
  available. This amendment changes NO estimand, NO gate test, NO δ threshold, and NO
  R1-R5 framing ladder. It (a) converges stale §1-hero references onto the canonical
  bootstrap-percentile producer, (b) lands the B-1003 SE-floor 0.68pp threshold that
  prereg L98/L718 already locked in prose but whose code still read literal `<= 0`, and
  (c) lands the AMENDMENT_01 `total_billed` cost estimand in the two stale cost producers
  that still read `total_cost_usd`. All three are "code/prose caught up to an
  already-witnessed estimand", not new analytical choices.
---

# Preregistration Amendment 03 — Implementation alignment (NO estimand change)

> **One-line**: Three estimands were already locked by earlier witnesses — the H1 SE-floor
> at 0.68pp (DOI-1 prereg L98/L718, B-1003), the canonical H1 PRIMARY = bootstrap
> percentile producer (AMENDMENT_02 §2 line 99), and §1/H10 cost = `total_billed`
> (AMENDMENT_01). This amendment lands those locks in code + draft prose where stale
> implementations still diverged. It does **not** introduce, weaken, or re-choose any
> estimand, gate, threshold, or framing tier.

## §0 — Pre-data status (the legitimacy anchor)

Witnessed **before any Phase 1a paper-grade outcome statistic exists**. At witness time:
- No per-cell drop-one θ_i and no pooled θ_FE has been computed on paper-grade data
  (`results/phantom_paper/phase1_full_prereg_decision.*` not yet produced from a complete
  6-mode cell; `results/phantom_paper/h10_pareto_verdict.*` = no Pass-2 router data).
- Pass-1 baseline (36 conditions) is incomplete; Pass-2 router (6 conditions) has not fired.
- Only single-mode runs have completed (e.g. B0 cls DOM = `R31194`); a per-cell drop-one θ_i
  requires all six modes present in one cell, so no gate statistic can be computed from them.

Because no paper-grade outcome statistic has been computed, this is recorded as a
**pre-data implementation alignment** and is externally witnessed before any eligible
H1/H3/H10 result is available.

**Honest exposure note**: pre-fix *archive* outcomes are a correlated-population sanity
check, explicitly NON-substrate in DOI-1, known buggy, and were NOT used to motivate any
choice here. The motivation is purely consistency between code/prose and the
already-locked estimands (the alignment was surfaced by a code audit, not by inspecting
any outcome). The SE-floor change moves the published θ_FE by **0** on the current archive
(all archive P-SoM cells have SE ≥ 0.766pp ≥ the 0.68 threshold, so neither the old `<= 0`
nor the new `< 0.68` floor fires); it only matters if a future Phase 1a cell yields
SE ∈ (0, 0.68pp), in which case the new code matches the locked estimand and the old code
would have diverged from it.

## §1 — Why this amendment (three pre-existing locks, three stale implementations)

1. **SE-floor double-track.** Prereg §2 H1 L98 + L718 (B-1003, 2026-05-18) lock the
   degenerate-cell SE-floor at the **0.68pp Agresti-Coull threshold → 1.0pp** and
   explicitly note this was "a code-bug fix … was code-level literal `<= 0`". The canonical
   producer `aggregate_phase1_full_prereg_decision._pool_bootstrap_percentile_p` already
   implements `< 0.68`; the legacy transparency producer
   `aggregate_phase1_prereg_gate._fe_pool` still implemented `<= 0`. Two producers sharing
   the same `_cell_drop_one_theta_se` kernel but applying different floors → θ_FE could
   split on any cell with SE ∈ (0, 0.68pp). B-1003's "fix" had never landed in this code.

2. **§1-hero reference points at the wrong producer.** AMENDMENT_02 §2 line 99 locks the
   H1 PRIMARY gate as the **bootstrap-percentile** test in
   `aggregate_phase1_full_prereg_decision`; the legacy normal-Z `aggregate_phase1_prereg_gate`
   is a transparency column ("does NOT drive the gate decision", prereg L98). Yet several
   draft-prose + figure references still named `phase1_prereg_gate` as the "§1 H1 PRIMARY
   hero" source.

3. **Cost estimand stale in two producers.** AMENDMENT_01 (2026-05-21) locks **§1 PRIMARY
   cost = `total_billed_cost`** (canonical/wasted → §4) and **H10 Pareto Cost-axis =
   `total_billed_cost`**. `aggregate_cross_site.py` already consumes
   `avg_total_billed_cost_usd` (Q1=A); but `aggregate_cost_electricity.py` (B0 paper cost)
   and `aggregate_h10_pareto.py` (episode cost-axis) still read the legacy
   `avg_total_cost_usd` / `total_cost_usd`.

## §2 — What does NOT change (carried forward UNCHANGED)

- **H1 primary gate estimand** (prereg L93-98): 6-mode P-SoM drop-one, FE inverse-variance
  pool over the 6 planned cells, one-sided **bootstrap percentile** gate `p < α=0.05`.
  **UNCHANGED.** Per-cell kernel `_cell_drop_one_theta_se` **UNCHANGED**.
- **δ = 1.0pp** superiority threshold. **UNCHANGED.**
- **SE-floor REPLACE value = 1.0pp** (only the *trigger threshold* code is corrected from
  `<= 0` to the already-locked `< 0.68`; the floored value, the estimand, and the FE pool
  arithmetic are unchanged).
- **R1-R5 framing ladder** + AMENDMENT_02 post-R5 reporting routes + anti-rescue guard.
  **UNCHANGED.**
- **H3 / H10 estimands + thresholds.** **UNCHANGED** (H10 cost-axis was already locked to
  `total_billed` by AMENDMENT_01; this only lands the field name in code).
- **Cost estimand definition** = `total_billed` (already locked AMENDMENT_01); only the two
  stale producers are pointed at the canonical field, with a fail-closed guard + explicit
  `P79_ALLOW_LEGACY_COST=1` legacy escape for archive vintages. **No silent fallback.**

## §3 — What changes (implementation/prose only)

| Change | Files | Nature |
|---|---|---|
| SE-floor `<= 0` → `< 0.68` (module-level constant, single source mirrored by canonical) | `scripts/analysis/aggregate_phase1_prereg_gate.py` | code catches up to prereg L98/L718 (B-1003) |
| Legacy gate docstring + payload marked **TRANSPARENCY-ONLY / NON-CANONICAL**; points to canonical primary | `aggregate_phase1_prereg_gate.py` | provenance honesty |
| §1-hero references converged to `phase1_full_prereg_decision` (away from `phase1_prereg_gate`) | `scripts/analysis/figures/fig_meta_forest.py`, `aggregate_phantom_lift.py`, `docs/checkpoints/paper_drafts/section1_intro.md`, `section3_definition.md` | reference correction per AMENDMENT_02 §2 |
| B0 paper cost + H10 cost-axis → `total_billed` (+ fail-closed, `P79_ALLOW_LEGACY_COST` escape) | `scripts/analysis/aggregate_cost_electricity.py`, `aggregate_h10_pareto.py` | code catches up to AMENDMENT_01 |
| Regression tests for canonical-source + SE-floor==prereg; existing floor tests updated to 0.68; cost fixtures carry `total_billed` | `tests/test_h1_canonical_alignment.py` (new), `tests/test_phase1_prereg_gate.py`, `tests/test_stress_deepaudit_h10_basis.py` | guards the alignment from re-drifting |

**Estimand-scope honesty for prereg L103-111**: the prereg "Degenerate-cell SE floor
protocol" paragraph (L103-111) still describes the *pre-B-1003* `SE = 0 exactly` wording and
points its "Implementation" at `aggregate_phase1_prereg_gate.py:185-187`. The locked
estimand value is 0.68pp (L98/L718, B-1003); L103-111 is stale prose, not a competing lock.
A follow-up honesty-surface edit to L103-111 (rephrase "SE = 0 exactly" → "SE < 0.68pp
Agresti-Coull threshold", repoint implementation to the canonical producer) is recommended
at the next paper-finalize pass; it is a wording sync with **no estimand effect** (the
0.68pp value is unchanged). Recorded here so the L98-vs-L103 internal wording gap is a
documented known-item, not a silent inconsistency.

## §4 — Witness and timing

- **Witness primitive** = the commit SHA that adds this file (content-addressed,
  tamper-evident), with git tag `prereg-amendment-03-implementation-alignment-20260524`
  and an OSF upload as the external-visibility layer. See companion
  `git_witness_IMPLEMENTATION_ALIGNMENT_20260524.txt`.
- **Timing requirement**: this witness MUST precede the computation of any paper-grade
  per-cell drop-one θ_i, pooled θ_FE, or H10 verdict. As of the witness commit no such
  paper-grade statistic exists (§0). The full Phase 1a fire continuation (Pass-1 → Pass-2)
  and any canonical analysis run occur AFTER this witness.

---

*This file is a DRAFT pre-fire witness. It becomes the binding witness only at the commit
that adds it, after which the git tag is created and the OSF upload performed. Fill the
SHA / tag / timestamps in `git_witness_IMPLEMENTATION_ALIGNMENT_20260524.txt` at commit time.*
