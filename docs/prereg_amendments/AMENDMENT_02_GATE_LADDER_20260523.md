---
amendment_id: 02
title: Post-R5 reporting-scope clarification + power-table estimand-label correction (H1-strict primary gate preserved UNCHANGED)
date: 2026-05-23
status: pre-fire witness (DRAFT — pending git tag + push + OSF upload); Phase 1a paper-grade data NOT yet created
parent_prereg: docs/checkpoints/pre_run/preregistration.md (status: locked)
parent_doi: 10.17605/OSF.IO/9QCWU   # DOI 1, pre-canonical-outcome-creation witness, 2026-05-18
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_01a_SCHEMA_VALIDATOR_20260521
witness_tag: prereg-amendment-02-gate-ladder-20260523   # to be created at the commit adding this file
relation: >
  Pre-data clarification of the §2.5 FRAMING DECISION RULE. Recorded BEFORE any Phase 1a
  paper-grade outcome statistic exists (canary R11315 = probe-only, NOT paper data; Pass-1
  incomplete; no per-cell drop-one θ_i and no pooled θ_FE computed on paper-grade data), and
  externally witnessed before any eligible H1/H3/H10 result is available. This amendment does
  NOT alter the H1 primary gate estimand, its δ=1.0pp threshold, its producer, or the
  "R5 fires on H1-failure" principle (preregistration.md:349). It (a) corrects a power-table
  estimand-LABEL inconsistency present since DOI-1 lock, and (b) clarifies that the *reporting
  scope* after an R5 (H1-failure) outcome is conditional on the independently pre-registered
  H3 / H10 results, so an H1-strict failure no longer forces discarding pre-registered
  phantom-space-structure / router evidence. The anti-rescue principle (no R5→R3 framing-tier
  rescue) is PRESERVED UNCHANGED and remains enforced in
  aggregate_phase1_full_prereg_decision.py + preregistration_decision_test.py.
---

# Preregistration Amendment 02 — Post-R5 reporting-scope clarification + power-table estimand-label correction

> **One-line**: The H1 primary gate — the 6-mode P-SoM drop-one superiority test — is kept
> EXACTLY as locked in DOI 1; it remains the genuine high-risk bet of the paper. This
> amendment only (a) corrects a power-table that reported a 4-mode effect size for the
> 6-mode gate, and (b) clarifies that after an R5 (H1-failure) the reporting scope is
> conditional on the independently pre-registered H3 / H10 gates — so that if the
> strict-uniqueness claim dies but the phantom-space (H3) or router (H10) results survive on
> their own gates, those results are not discarded. It does NOT make H1-strict easier,
> non-fatal-for-its-own-claim, or rescuable into a higher framing tier.

## §0 — Pre-data status (the legitimacy anchor)

This amendment is witnessed **before any Phase 1a paper-grade outcome statistic exists**. At
witness time:

- Canary run `R11315` (B0 × SoM × classifieds) is **probe-only, explicitly NOT paper-grade
  data** (per `next_steps.md §0 ④`); it produces no per-cell drop-one estimate.
- Pass-1 baseline (36 conditions) is **incomplete**; Pass-2 router (6 conditions) has **not
  fired**.
- No per-cell θ_i and no pooled θ_FE has been computed on paper-grade data
  (`results/phantom_paper/h10_pareto_verdict.csv` = all `no_pass1_runs`;
  `results/phantom_paper/l1_router/stage2_summary.json` = `status: no_data_yet`).
- Only single-mode runs have completed (B0 cls DOM = `R9755`; B0 cls SoM = `R11315` canary),
  and a per-cell drop-one θ_i requires **all six modes present in one cell** — so no gate
  statistic can be computed from them **regardless of their individual grade**. (`R9755`
  additionally predates the per-condition docker-reset that the 36-condition fresh fire
  applies, so it is treated as non-paper-grade and re-run fresh; this only reinforces that no
  eligible H1 statistic exists at witness time.)

Because no paper-grade outcome statistic has been computed, this amendment is recorded as a
**pre-data clarification** and is externally witnessed before any eligible H1/H3/H10 result is
available.

**Honest exposure note**: pre-fix *archive* outcomes — a correlated-population sanity check,
explicitly designated NON-substrate in DOI-1, and known to contain evaluator/pipeline bugs
that motivated the Phase 1a re-fire — were inspected during design review. To minimise any
"design responded to correlated archive" exposure, this amendment **changes the H1 primary
gate by ZERO** (§2) and **cites no archive effect size as its basis**; its motivation is
purely structural (§1).

## §1 — Why this amendment (logical independence of the claims; structural, not outcome-driven)

**Primary reason — H1, H3, H10 are logically distinct claims with independent pre-registered
gates.** H1-strict asks whether P-SoM is the *only* mode that succeeds among all six modes (a
narrow irreplaceability claim). H3 asks whether the phantom siblings (P-text / P-prompt) carry
unique oracle coverage (the phantom region is a real 2-D structure). H10 asks whether a
learned router over the space is operationally deployable. An H1-strict failure — *"P-SoM is
not uniquely irreplaceable when all six modes are available"* — does **not** logically imply
that H3 or H10 fail. Yet DOI-1 §2.5 routes any H1 failure to R5 → *"main paper-1 abandoned"*,
discarding H3 and H10 **even when they hold their own pre-registered gates**.

**Secondary, structural reinforcement (stated as an estimand property, not a quantified archive
prediction).** By construction the 6-mode strict estimand is reduced whenever any of the other
five modes — including the cheap phantom siblings — also succeeds on a task P-SoM solves; so
the 6-mode uniqueness can be small even when P-SoM is a strong, deployable arm and the phantom
space is clearly real (the 6-mode strict drop-one is ≤ the 4-mode additive lift by
construction). A small H1-strict is therefore *compatible with* — not evidence against — a
real phantom-space contribution. (Note: this is weaker than a clean "stronger H3 ⇒ weaker H1"
law; H3 measures sibling coverage *unique beyond* P-SoM, whereas H1 shrinks from sibling
*overlap on* P-SoM's successes — distinct quantities. The amendment relies only on the
logical-independence reason above, not on a mechanical anti-correlation.)

**The defect being fixed.** Combining these, an H1-strict failure can unilaterally kill
independently-confirmed H3/H10 contributions. This amendment removes that pathology **without
weakening H1-strict's own claim**.

## §2 — What does NOT change (carried forward UNCHANGED from DOI 1)

- **H1 primary gate estimand** (preregistration.md:93-98): per-cell 6-mode drop-one
  `θ_i = SR_oracle({DOM,SoM,Vision,P-text,P-prompt,P-SoM}) − SR_oracle({same 5 except P-SoM})`,
  pooled FE inverse-variance over the 6 planned cells, one-sided bootstrap percentile gate
  `p = P(θ_FE* ≤ 1.0pp) < α=0.05`. **Producer `aggregate_phase1_prereg_gate.py:_cell_drop_one_theta_se`
  is unchanged.** δ=1.0pp unchanged. SE floor (1.0pp) unchanged.
- **R5 fires on H1-failure** (preregistration.md:349): UNCHANGED and reaffirmed. An H1-strict
  failure still produces the R5 framing tier.
- **Anti-rescue principle**: R1/R2/R3 all require "H1 holds" (preregistration.md:343-345); an
  H1 failure CANNOT be rescued into a higher framing tier (no R5→R3 rescue). This amendment
  performs no framing-tier rescue (see §5). The guard remains enforced by
  `aggregate_phase1_full_prereg_decision.py` (R1-R5 mapper) and its regression test
  `preregistration_decision_test.py` (which explicitly forbids R5→R3 rescue) — **both
  unchanged**.
- **H2(a)** cost-equivalence by-construction falsification check: UNCHANGED. It remains a
  non-fatal R4 modifier; this amendment does NOT promote it to a survival/death gate.
- **H3(i)/(ii)** axis-structure gates and **H10** operational deployment gate: estimands and
  thresholds UNCHANGED. This amendment references them only by pointer to their DOI-1
  definitions (§4); it does not redefine them.

## §3 — Correction 1: power-table estimand-label mismatch (a planning-number fix, not a gate change)

DOI-1 §2.5 power table (preregistration.md:361) reports the H1 row at **+2.336pp**. That
effect size is the **4-mode ADD** estimand `4psom_vs_3` =
`SR_oracle({DOM,SoM,Vision,P-SoM}) − SR_oracle({DOM,SoM,Vision})`
(`aggregate_phantom_lift.py:809`; `meta_phantom_lift.csv:4`, flagged there as
`SECONDARY / appendix-only`). It is **NOT** the 6-mode drop-one that the H1 gate computes
(`_cell_drop_one_theta_se`: oracle over all 6 minus oracle over the 5 without P-SoM). The H1
row therefore reports the power of the **wrong (easier, 4-mode) estimand** for the 6-mode gate.

By construction the 6-mode strict drop-one is **≤** the 4-mode additive lift, so H1-strict is
the **stricter** of the two estimands and the +2.336pp figure overstates the H1-strict effect.
**The H1-strict effect size and its power will be established by Phase 1a paper-grade data, not
by the pre-fix archive** (which is buggy non-substrate and is deliberately not cited here).

The +2.336pp 4-mode figure is re-designated the **H1-deploy / Appendix-D sensitivity**
estimand, consistent with the already-locked `section1_intro.md` `[^hero-estimand-scope]`
footnote (which already states the 4-mode drop-one moves to Appendix-D). **H1-deploy is a
reported sensitivity, NOT a gate and NOT a survival leg.** No code change to the gate is implied
by this correction; it relabels a power-calculation number and corrects its estimand provenance.

## §4 — Correction 2: post-R5 reporting-scope clarification (the substantive change)

The single hardcoded R5 reporting scope ("abandon paper-1; write the B-91 evaluation-systems
note") is clarified into reporting routes selected by the **independently pre-registered H3 and
H10 gates, each evaluated under its exact DOI-1 decision rule**. **R5 still fires on
H1-failure** (the framing tier is unchanged); only its reporting scope is conditional. H3-pass
(structure) and H10-pass (router) are kept distinct so that an H10-only result is never
reported as confirmed phantom-space structure:

- **Route P — Primary bet.** H1-strict **passes** its gate. The P-SoM strict-uniqueness claim
  remains eligible for R1/R2/R3 per the existing DOI-1 mapping (H2(a) and H3(i)/(ii) modifiers
  unchanged).

- **Route C'-S — Structure pivot.** H1-strict **fails**, so the strict P-SoM 6-mode
  uniqueness claim fails and **R5 fires**. **If H3 passes for ≥1 axis under its locked DOI-1
  §2 H3(i)/(ii) decision rule**, the authors may report a **lower-claim phantom-space-structure
  paper** ("strict P-SoM 6-mode uniqueness not established; the phantom-space structural result
  holds on its own pre-registered gate"). **This is not an R5→R3 framing rescue** (§5).

- **Route C'-R — Router-only pivot.** H1-strict **fails AND** H3 **fails** (neither axis), but
  **H10 passes under its locked DOI-1 §2 H10 operational deployment gate**. The authors may
  report a **lower-claim learned-routing / systems result, WITHOUT claiming confirmed
  phantom-space structure** ("operational routing result holds despite weak oracle-space
  structure"). **This is not an R5→R3 framing rescue** (§5), and H10 is reported only as
  router/systems evidence, never as phenomenon (oracle-space) evidence.

- **Route F — Failure (preserved death path).** H1-strict **fails AND** H3 **fails AND** H10
  **fails**. The hero / phantom-space / router program fails for paper-1; pivot to a negative /
  methodology / reliability paper (e.g., the B-91 VWA LLM-judge polarity-bug evaluation-systems
  note, per 实验笔记 §179).

**Falsifiability is preserved.** Route F is a genuine "the program failed" death outcome. The
amendment does not make the paper unkillable; it only stops H1-strict's failure from
*unilaterally* killing independently-confirmed H3 or H10 contributions, and it keeps the
H10-only case at a strictly lower (router/systems, non-structure) claim tier. Each route uses
the **exact locked DOI-1 gate** for H3 / H10; no new "meaningful" / "suggestive" soft threshold
is introduced.

## §5 — Why this is NOT a violation of the anti-rescue guard

The regression-tested anti-rescue guard (`preregistration_decision_test.py`, which forbids
rescuing a failed FE-H1 from R5 into R3 via a per-cell rule) concerns the **framing TIER**: it
prevents re-labeling the §1 hook framing upward (R5→R3) after H1 fails.

Routes C'-S and C'-R perform **no framing-tier rescue**:

- Under H1-failure the framing tier is still R5; the P-SoM strict-uniqueness claim still dies;
  no R1/R2/R3 hook is claimed (those still require H1-holds, unchanged).
- The routes open **separate, lower-claim reporting pivots** grounded in H3 / H10 — hypotheses
  with their own pre-registered gates that are not framing-tier-conditioned on H1 (H3 has its
  own STRUCTURAL gate; H10 has its own §6 operational gate; per DOI-1 §39 "H10 fail caps §6 but
  does not collapse the phenomenon", H10 was already decoupled from the R-rule).
- **H10 is not made a rescue condition for the original H1-dependent framing tier.** This
  amendment only permits an H10-positive result to be reported as a lower-claim router/systems
  pivot *after* the H1-dependent paper has failed; it does not let H10 stand in for the H1 hook
  or for confirmed phantom-space structure.

**Code/test contract (SHOULD).** The R1-R5 mapper SHOULD additionally surface the
already-computed H3 / H10 pass flags so the post-R5 reporting route is mechanically determined
from pre-registered gate outputs, with **`r_tier` kept = `R5`** (never re-emitted as R3, never
displayed as "rescued"):

```json
{
  "r_tier": "R5",
  "h1_strict_pass": false,
  "post_r5_pivot": "C_prime_structure" | "C_prime_router_only" | "F_failure",
  "pivot_basis": ["H3"] | ["H10"] | []
}
```

The existing anti-rescue test (`r_tier != R3` after H1-fail) remains valid, and new tests
SHOULD assert: `H1 fail + H3 pass → r_tier == R5 AND post_r5_pivot == C_prime_structure`;
`H1 fail + H3 fail + H10 pass → r_tier == R5 AND post_r5_pivot == C_prime_router_only`;
`H1 fail + H3 fail + H10 fail → r_tier == R5 AND post_r5_pivot == F_failure`. The guard remains
valid and unchanged.

## §6 — Superseded / corrected references

| Reference | DOI-1 text | Amendment 02 effect |
|---|---|---|
| `preregistration.md:347` (R5 row reporting scope) | "Paper death … pivot to Track B workshop (B-91 note); main paper-1 abandoned. NO other pivot pre-registered." | R5 still fires on H1-fail; its reporting scope is now conditional — §4 Route C'-S (H3 pass → structure paper) / Route C'-R (H3 fail, H10 pass → router-only paper) / Route F (H3 & H10 fail → negative/methodology/B-91 note). |
| `preregistration.md:361` (power table H1 row) | H1 power computed at +2.336pp | +2.336pp is the 4-mode ADD (H1-deploy / Appendix-D), not the 6-mode H1-strict gate effect (6-mode ≤ 4-mode by construction; true value TBD from Phase 1a). H1-strict acknowledged as the stricter test (§3). |
| `preregistration.md:395-404` (decision tree) | "p≥0.05 → H1 FAILS → R5 (paper-death/pivot)" | R5 node downstream reporting route split per §4 (R5 still fires; route conditional on H3/H10). |
| `preregistration.md:349` (R5 trigger) | "R5 fires only on the single H1 superiority gate failing" | NOT superseded — explicitly reaffirmed. R5 still fires on H1-failure; only its reporting scope is clarified. |
| `section1_intro.md:21` (`[^hero-estimand-scope]`) | already states 4-mode drop-one → Appendix-D sensitivity, 6-mode = primary gate | NO CHANGE — cited as corroborating §3. |
| `aggregate_phase1_full_prereg_decision.py` (R1-R5 mapper) | emits R5 on H1-fail; no R5→R3 rescue | UNCHANGED gate logic; SHOULD additionally surface `post_r5_pivot` + H3/H10 `pivot_basis` flags (§5), `r_tier` stays `R5`. |
| `preregistration_decision_test.py` (anti-rescue regression test) | forbids R5→R3 framing rescue | UNCHANGED guard; SHOULD add the three `post_r5_pivot` assertions in §5. |

Downstream prose to align (no gate logic change): `section6_router.md` R-rule references;
`section8_limitations.md` framing notes; `_status/tasks/task_rtier_decision.md`;
`_status/issues/issue_advisor_sync_preregistration.md`. A grep sweep for `R5` / `paper death`
/ `R1-R5` MUST confirm completeness before commit.

## §7 — Witness and timing

- **Witness primitive** = the commit SHA that adds this file (content-addressed,
  tamper-evident; commit timestamp hashed into the SHA), with an immovable git tag
  `prereg-amendment-02-gate-ladder-20260523` and an OSF upload as the external visibility
  layer. See companion `git_witness_GATE_LADDER_20260523.txt`.
- **Timing requirement**: this witness MUST precede the computation of any paper-grade per-cell
  drop-one θ_i or pooled θ_FE. As of the witness commit, no such paper-grade statistic exists
  (§0). Full Phase 1a fire (Gate 3 → Pass-1 → Pass-2) must occur AFTER this witness.

---

*This file is a DRAFT pre-fire witness. It becomes the binding witness only at the commit that
adds it, after which the git tag is created and the OSF upload performed. Fill the SHA / tag /
timestamps in `git_witness_GATE_LADDER_20260523.txt` at commit time.*
