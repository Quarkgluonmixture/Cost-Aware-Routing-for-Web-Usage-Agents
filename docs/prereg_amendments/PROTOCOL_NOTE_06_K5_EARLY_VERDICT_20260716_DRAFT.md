---
title: Time-constrained k=5 submission verdict with unconditional k=6 upgrade — temporary deviation from the six-planned-cell estimand
status: DRAFT — pending advisor sign-off 2026-07-16 meeting; NOT in force
parent_doi: 10.17605/OSF.IO/9QCWU
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_02_GATE_LADDER_20260523
  - AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524
  - AMENDMENT_04_ANALYSIS_ALIGNMENT_20260524
  - AMENDMENT_05_COORDINATE_CONTRACT_20260525
  - AMENDMENT_06_REPRODUCIBILITY_SENSITIVITY_20260525
  - AMENDMENT_07_SOM_IDENTIFIER_CONTRACT_20260525
  - PROTOCOL_NOTE_01_SESSION_LOST_PAPER_GRADE_20260527
  - PROTOCOL_NOTE_02_TRANSIENT_PREFLIGHT_RETRY_20260621
  - PROTOCOL_NOTE_03_RESUME_ON_ABORT_20260622
  - PROTOCOL_NOTE_04_REDDIT_IDENTITY_RESET_20260625
  - PROTOCOL_NOTE_05_ANALYSIS_ESTIMAND_CONFORMANCE_20260714
witness_tag: <PENDING-ADVISOR-SIGNOFF-AND-FINALIZING-COMMIT-NO-TAG-CREATED>
osf_notice: DRAFT — pending advisor sign-off; notify OSF amendment log only if activated
decided_by: <PENDING — advisor decision at 2026-07-16 meeting>
effective_at: NOT IN FORCE — no k=5 branch selection, splice, or paper claim is authorized by this draft
---

# PROTOCOL_NOTE_06 — Time-constrained k=5 early verdict

> **DRAFT-PENDING-ADVISOR / NOT-IN-FORCE.** This file is a complete prewrite for
> the 2026-07-16 meeting. It does not authorize a k=5 verdict, branch selection,
> tag, preregistration edit, OSF notice, or paper splice unless the activation
> conditions in §6 are completed.

## 0. Scope — a time-constrained deviation from “over the 6 planned cells”

The locked preregistration defines the H1/H3 fixed-effects estimands over
exactly six planned `(site, model)` cells. This note proposes a **temporary,
time-constrained deviation** for an AAAI-27 submission verdict over five landed
cells, with `(reddit, B2=Gemma-3)` absent. It does not reinterpret five cells as
the original six-cell estimand and does not change the per-cell effect,
bootstrap, weighting, threshold, alpha level, task universe, or missing-data
rules.

The written justification is limited to the following two
outcome-independent facts:

1. AAAI-27 has a hard full-paper deadline of **2026-07-28 23:59 AoE**. The
   observed execution rate for the six-condition B2 Reddit chain is about
   **1.65 days per condition**, putting its completion ETA near **2026-07-25**
   and leaving insufficient deterministic time for bind, full verdict
   regeneration, splice, audit, and submission packaging.
2. The paper-1 H10 router gate is already structurally fail-closed independently
   of the missing B2 Reddit outcomes: `B2_classifieds` and `B1_reddit` do not
   provide trainable router policies, so collecting B2 Reddit cannot restore
   the registered 5-of-6 H10 deployment criterion.

**Prohibited rationale.** No interim H1 point estimate, confidence interval,
per-cell effect, pass probability, branch probability, or direction of change
may be cited to justify activating, rejecting, delaying, or upgrading this
note. Interim outcomes are chronologically available, so this note does not
claim a pre-outcome-blind timestamp; its protection against outcome-dependent
selection is the exhaustive rationale above plus the symmetric, unconditional
upgrade rule in §3.

## 1. Temporary k=5 pooled-gate definition

### 1.1 Fixed landed-cell set

The temporary submission set is fixed in advance as:

`{B0_classifieds, B1_classifieds, B2_classifieds, B0_reddit, B1_reddit}`.

It is not “any five” and cannot be changed after viewing outcomes. Each cell
must be fully landed and bound, contain all six registered observation modes,
and pass the same exact task-universe and provenance checks as the six-cell
analysis. `B2_reddit` is the only omitted cell.

### 1.2 Estimation and gates

For H1 and each H3 axis, apply the **same locked analysis mechanism** to those
five cells: the same per-task paired, per-cell bootstrap; the same fixed
per-cell inverse-variance weights; the same fixed-effects pooling formula; the
same 1,000 pooled bootstrap replicates and percentile confidence interval; and
the same preregistered decision thresholds. H1 therefore retains the
one-sided test against `+1.0pp` at `alpha=0.05`; each H3 axis retains its
one-sided zero-threshold test at `alpha=0.05`.

This produces a temporary finite-design average over the five named landed
cells, **not** an estimate of the omitted sixth cell and **not** the locked
six-cell finite-design average. K-of-N quantities remain transparency-only and
must be reported as counts out of five, never converted into a new threshold.
Per-cell forest estimates remain visible.

Every numerical or branch verdict derived under this note must be qualified
with the exact phrase **“on the five landed cells”** at first mention in the
abstract, §1, §4, §8, figure/table captions, and any public summary. It must not
be labelled simply “the pre-registered six-cell verdict.”

The existing `analysis_status=PARTIAL` artifact and rehearsal slotsheet remain
non-verdict artifacts. Activation requires a separately identified,
Protocol-Note-06-authorized k=5 artifact/slotsheet; neither legacy
`gate_status`, an interim boolean, nor manual reading of a partial sheet may be
used to select a branch.

## 2. B-1284 cross-family modifier under missing B2 Reddit

The locked B-1284 cross-family claim-tier gate requires both Gemma cells
(`B2_classifieds` and `B2_reddit`) for the two-cell Gemma replication check.
With B2 Reddit absent, Gemma evidence consists of **one Classifieds cell only**;
cross-site Gemma replication is not established.

Accordingly, a k=5 submission automatically applies the conservative B-1284
**one-tier downgrade** wherever an R-tier is otherwise available: an R1
candidate becomes R2, and an R2 candidate becomes R3. The k=5 submission is
therefore **capped at R2**. It may not claim cross-family or cross-site Gemma
robustness; it may only describe the observed Gemma-Classifieds direction as a
single-cell result. This modifier cannot rescue a failed H1/H3 gate or change
the Amendment-02 reporting branch.

## 3. Two-track commitment and unconditional k=6 upgrade

Upon activation, both tracks apply immediately and remain independent of
observed effect direction:

- **Submission-baseline track:** once the five fixed cells in §1.1 are landed
  and bound, compute the Protocol-Note-06 k=5 verdict and prepare the paper with
  the required “on the five landed cells” qualifications and §2 downgrade.
- **Completion track:** keep the B2 Reddit six-condition chain running in the
  background toward the original six-cell design.

If B2 Reddit fully lands and is provenance-bound **before the paper is
submitted**, the submission must unconditionally upgrade to k=6: regenerate
the complete analysis artifacts, regenerate the verdict-day slotsheet,
reselect the branch from the six-cell verdict, resplice all affected prose and
tables, rerun audits, and delete all k=5-only disclosure language. The k=5
verdict is then **void for submission**, whether the k=6 result is more or less
favourable. Conversely, if B2 Reddit has not landed and bound by submission,
the signed k=5 verdict remains the submission baseline regardless of whether
its result is favourable or unfavourable. Data completeness—not outcome
direction—chooses which verdict is submitted.

## 4. Mandatory disclosure package

Activation requires all four disclosures below in the same change set; none is
optional:

1. **Paper §4:** insert the `(K5-CONDITIONAL)` one-sentence completion-status
   disclosure prewritten in
   `docs/checkpoints/paper_drafts/aaai27/branch_prewrites_s1_abstract.md` §8.1.
2. **Paper §8:** replace the statistics paragraph with the k=5 fixed-cells
   interpretation and B-1284 downgrade paragraph prewritten in that file §8.2.
3. **Preregistration amendment log:** append, without rewriting the locked
   preregistration body, the following row only after activation:

   `| 2026-07-16 | Advisor-signed PROTOCOL_NOTE_06 temporarily defines the submission verdict as the unchanged FE paired-bootstrap mechanism over the five fixed landed cells (B2 Reddit missing), with all claims qualified “on the five landed cells”, automatic B-1284 one-tier downgrade / R-tier cap R2, continued B2 Reddit collection, and unconditional k=6 regeneration if that cell lands and binds before submission. Decision basis: AAAI-27 hard deadline plus H10 structural fail-closed; no interim H1 quantity was used as a reason. | Advisor signature + witness tag + OSF notice; see PROTOCOL_NOTE_06_K5_EARLY_VERDICT_20260716_DRAFT.md |`

4. **OSF:** post an amendment-log notice containing this note, the advisor
   sign-off date, finalizing Git SHA/tag, the five named cells, the missing B2
   Reddit status, the automatic B-1284 downgrade, and the unconditional k=6
   upgrade commitment. Record the OSF notice URL/timestamp in the finalized
   note.

The abstract and §1 k=5 replacements in the branch-prewrite file §8.3 are also
mandatory because the scope qualifier must accompany verdict claims, not only
appear later in Methods.

## 5. Non-selection and invalid uses

- The decision to activate this note cannot cite any interim H1 quantity or
  which branch appears more likely.
- The five-cell set cannot be changed, reweighted, or reduced after outcome
  inspection.
- A k=5 result cannot be described as evidence about `B2_reddit`, a six-cell
  cross-family replication, or the original six-cell finite-design mean.
- The k=5 and k=6 verdicts cannot coexist as selectable submission branches.
  Once the k=6 upgrade trigger fires, only k=6 may be submitted.
- This draft cannot be invoked by editing its frontmatter alone; §6 must be
  completed.

## 6. Advisor decision and activation record

**Meeting decision (check exactly one):**

- [ ] **APPROVE** the two-track k=5 baseline / unconditional k=6 upgrade policy.
- [ ] **REJECT**; retain the locked six-cell-only verdict policy.

Advisor name: `<PENDING>`  
Advisor signature or witnessed written confirmation: `<PENDING>`  
Decision timestamp (with timezone): `<PENDING>`  
Student acknowledgement: `<PENDING>`

If approved, the ten-minute activation sequence is: fill this block → change
frontmatter from DRAFT/NOT-IN-FORCE to the signed status → create the finalizing
commit and recorded witness tag → post/queue the OSF notice → generate the
Protocol-Note-06 k=5 slotsheet → splice only the k=5 prewrites and disclosures.
Until those steps are complete, the current six-cell runbook remains binding.
