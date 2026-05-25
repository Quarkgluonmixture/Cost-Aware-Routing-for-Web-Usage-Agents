---
amendment_id: 06
title: Non-gating reproducibility sensitivity layer for run-to-run (execution-level) variance — H1 drop-one stochastic-measurement disclosure + self-oracle / replicate-calibrated sensitivity — NO estimand / gate / δ / SE-floor / R-ladder change
date: 2026-05-25
status: DRAFT — disclosure portion pre-fire witnessable NOW; sensitivity NUMBERS post-replicate (clean replicate not yet run; Gate-3 cls chain in progress). Phase 1a paper-grade outcome statistics NOT yet computed.
parent_prereg: docs/checkpoints/pre_run/preregistration.md (status: locked)
parent_doi: 10.17605/OSF.IO/9QCWU   # DOI 1, pre-canonical-outcome-creation witness, 2026-05-18
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_01a_SCHEMA_VALIDATOR_20260521
  - AMENDMENT_02_GATE_LADDER_20260523
  - AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524
  - AMENDMENT_04_ANALYSIS_ALIGNMENT_20260524
  - AMENDMENT_05_COORDINATE_CONTRACT_20260525
witness_tag: prereg-amendment-06-reproducibility-sensitivity-20260525   # to be created at the finalizing commit (post-replicate numbers)
provenance: GPT cross-AI independent hostile review 2026-05-25 (self-contained brief, same role as /stress Mode B/C; reviewer read prereg + amendments + ptext repro analysis cold) + 实验笔记 §292/§293 ptext archive↔current repro deep-dive + user-directed escalation ("波动超 noise floor 会威胁 phantom hero").
relation: >
  ADDS a witnessed, NON-GATING reproducibility/sensitivity layer that discloses and
  quantifies a variance component the pre-registered task-level bootstrap does NOT cover:
  run-to-run (execution-level) variance of the per-task binary success label. This amendment
  changes NO estimand, NO gate test, NO δ threshold, NO SE-floor, and NO R1-R5 framing ladder.
  The canonical H1/H10 gates execute exactly as locked. The added layer is reported as a
  transparency column + Appendix-D-style sensitivity, NOT as a primary gate, NOT as a bias
  correction to the canonical θ_FE, and NEVER replaces canonical episode labels. Recorded
  BEFORE any Phase 1a paper-grade outcome statistic exists; finalized (with empirical
  numbers) post clean-replicate. The analysis layer is NOT in the Gate-3 fire import path,
  so these additions are fire-safe; witnessed nonetheless because the prose-disclosure item
  touches the §1-hero estimand-adjacent surface.
---

# Preregistration Amendment 06 — Non-gating reproducibility sensitivity layer (NO estimand change)

> **One-line**: The pre-registered H1 gate resamples **tasks** (prereg L96-98: per-cell
> drop-one θ_i with *task-level* paired bootstrap SE_i). That bootstrap treats each
> single-run trajectory's success label as a fixed fact, so it captures between-task
> sampling variance but **not** run-to-run (execution-level) variance. For an
> oracle/drop-one estimand this matters, because run-to-run noise can manufacture
> apparent-unique tasks. This amendment **discloses** that gap and **adds a non-gating
> sensitivity layer** to quantify it. It does **not** change the gate, the estimand, the
> threshold, or the framing ladder.

## §0 — Pre-data status (the legitimacy anchor)

Recorded **before any Phase 1a paper-grade outcome statistic exists**. At witness time:
- No per-cell drop-one θ_i, no pooled θ_FE, no H10 verdict computed on paper-grade data.
- Pass-1 baseline (Gate-3 cls chain) in progress; Pass-2 router not fired; no clean replicate run yet.
- Therefore the disclosure here cannot be (and is not) motivated by any observed gate
  outcome. The motivation is structural: an estimand-level reasoning about which variance
  components the locked bootstrap does/does not cover, surfaced by a ptext archive↔current
  task-level repro analysis (笔记 §292) and an independent cross-AI review (§293).

## §1 — The gap (what the locked bootstrap does not see)

- **H1 PRIMARY** (prereg L96-98): per-cell P-SoM 6-mode-strict drop-one θ_i, **task-level**
  paired 1000-resample bootstrap SE_i → FE inverse-variance pool → one-sided bootstrap
  percentile p < α=0.05 at H₀ = +1.0pp.
- Observed unit is `Y_{task, mode, run}`; the bootstrap resamples **task** only, i.e. it
  conditions on `run = 1` as a noiseless fact. Complete uncertainty has at least two layers:
  `Var(θ̂) = Var_task + Var_run-to-run (+ task×run)`. The locked SE estimates only `Var_task`.
- **Why drop-one is structurally exposed** (vs aggregate SR which is run-to-run-stable, 笔记
  §282 "聚合 SR 殊途同归"): drop-one is a *uniqueness* operator — "P-SoM = 1 AND all five
  competitors = 0". A competitor's run-to-run false-failure manufactures a false-unique task;
  there are 5 competitors (union), so the false-unique channel is ~5× the P-SoM-false-success
  channel. Aggregate SR's bidirectional flips cancel; drop-one's do not.
- **Direction is conditional, not necessary** (GPT correction, §293): run-to-run noise on a
  *truly-unique* task lowers observed drop-one; on a *jointly-solvable* or *all-fail* task it
  raises it. Net sign depends on task-pool composition; VWA has many near-boundary
  trajectories, so a positive (anti-conservative) bias is *plausible* but must be reported as
  conditional, not asserted as a theorem.
- **The SE-floor does not address this**: the prereg SE_floor = 1.0pp (B-1003, prereg
  L103-111) is a degenerate-cell backstop (fires when between-task bootstrap SE_i = 0), anchored
  to Agresti-Coull 0.68pp + archive-median 0.98pp. It is not a run-to-run variance model.

## §2 — What does NOT change (gate integrity)

UNCHANGED, exactly as locked under DOI-1 + AMENDMENT_01/02/03:
- H1 estimand (6-mode-strict P-SoM drop-one), test (FE pool + bootstrap percentile), δ=1.0pp, SE-floor=1.0pp.
- H10 operational deployment gate (per-cell paired-bootstrap 95% Pareto non-dominance + 5/6 grid; realized router, not oracle — prereg L234, B-1855).
- H2/H3 estimands + thresholds; R1-R5 framing ladder; K-of-N transparency-only status.
- The canonical producers (`aggregate_phase1_full_prereg_decision`, `aggregate_h10_pareto`) emit the canonical verdict from canonical single-run labels, untouched.

This amendment adds **outputs**, not **decisions**. Per DOI-1 lock, NO gating-family test is added.

## §3 — What is ADDED (non-gating, transparency + Appendix-D-style sensitivity)

1. **Self-oracle discordance diagnostic** (DONE — `scripts/analysis/compare_cross_run_same_condition.py`):
   two independent runs of the **same** mode treated as two arms; their mutual drop-one is a
   pseudo-uniqueness floor (true value 0). Reports **symmetric** self_drop (1→2 and 2→1),
   discordance P(Y1≠Y2), agreement, Cohen κ. Asymmetric one-way self_drop ⇒ version/state
   drift, not pure stochastic noise. **DIAGNOSTIC / instability proxy ONLY — explicitly NOT a
   bias estimate** of the H1 false-unique bias (same-mode discordance ≠ P-SoM-vs-5-competitor
   false-unique; small-N / mixed-code-version cuts make it an upper-bound risk trigger).
2. **Replicate-calibrated sensitivity** (post-replicate; design only here): from clean
   replicates estimate a per-mode flip-rate / discordance matrix → Monte-Carlo perturb the
   canonical single-run success matrix under that flip model → recompute H1-strict θ_FE each
   draw → report canonical p / replicate-calibrated θ_FE distribution / P(θ_FE > 1pp) /
   floor-vs-effect ratio. **Reported as Appendix-D sensitivity, not a gate.**
3. **H1 prose run-to-run disclosure** (paper §1 / §3.6): one explicit paragraph stating the
   task-resampling gate treats each trajectory as a fixed label and therefore excludes
   execution-level variance; the sensitivity layer quantifies whether the observed drop-one
   exceeds the empirical self-oracle / replicate-calibrated noise floor.
4. **Pre-committed prose-downgrade rule** (GPT bottom line): IF H1-strict passes the locked
   gate but the replicate-calibrated noise floor is of the same order as the effect
   (both ≈1-2pp), the hero wording is downgraded from "P-SoM has a stable unique
   task-solving contribution" to "pre-registered single-run oracle evidence, with a
   reproducibility caveat". The gate verdict (pass/fail) is unchanged; only the **claim
   strength prose** adapts. This commitment is recorded pre-data so it cannot be read as
   post-hoc rationalization.

## §4 — Replicate protocol (isolation discipline)

- Replicate runs land in an **independent namespace** `results/repro_replicates/...`, NEVER
  the canonical `phase1a fire` tree. Replicate labels NEVER replace / majority-vote-overwrite
  canonical episode labels, and NEVER enter the primary H1/H10 denominator.
- **P0**: do not patch a running canonical fire; replicate a condition only after its
  canonical run completes (no mid-fire runner patch → no mixed-semantics fire, 笔记 §278).
- Priority: P1 B0 cls/red × **P-SoM** (hero arm self-instability); P2 + competitors
  {SoM, P-text, DOM} (false-unique attribution; tests the falsifiable **H_robust**: is
  P-SoM self-discordance ≤ {DOM,SoM,Vision}? — a win turns the upstream element-ID noise
  into a phantom *robustness advantage*, a deployment selling point); P3 local 4B
  **deterministic** isolation (`torch.use_deterministic_algorithms`) to separate element-ID
  churn from provider nondeterminism **without patching upstream**; P4 P-SoM-unique
  challenge-set re-run (post canonical H1).
- **element-ID churn — SUPERSEDED by AMENDMENT_07 (2026-05-25, same day).** This bullet
  originally recorded a decision NOT to patch element-ID churn, on the premise that "the churn
  IS part of real deployment". That premise was **falsified** later the same day: production /
  standard SoM interfaces (Yang 2023; VWA native `image_som`; WebVoyager; SeeAct-Choice;
  AndroidControl; browser-use) all use **sequential** selection ids, not raw browser nodeIds, so
  P79's SoM-family nodeId churn is a **P79-implementation artifact, not deployment-realistic
  noise** (P79 built SoM from the AXTree path instead of a sequential selection interface). The
  churn is therefore **corrected**, not merely quantified: AMENDMENT_07 changes the SoM-family
  identifier contract to deterministic sequential ids (an ESTIMAND-AFFECTING change, witnessed
  pre-restart, old SoM-family data archived non-canonical, cell re-collected — the AMENDMENT_05
  pattern, NOT a silent in-place patch). DOM / P-prompt keep native nodeId by design (the
  AXTree-native representation arm), so their churn remains and is covered by the sensitivity
  layer below. **After AMENDMENT_07, this §6/§3-item-2 non-gating sensitivity layer covers the
  RESIDUAL run-to-run stochasticity** (provider/MoE nondeterminism, evaluator-judge) that
  sequential ids do not remove — the element-ID component (the dominant source per §282) is
  eliminated upstream. The §4 replicate item P3 (local deterministic isolation) now measures
  that residual MoE/provider floor. See AMENDMENT_07 §1/§5 + B-1862.

## §5 — Cross-AI provenance + honest corrections record

This amendment originates from an independent GPT cross-AI review (§293) that **corrected
three of the author's own positions**, recorded here for audit honesty:
- (i) "H10 only loses power, never produces false positives" was **too optimistic** — H10 has
  no oracle-max selection bias, but a lucky-router / unlucky-baseline draw can still false-pass
  Pareto non-dominance; robustness depends on whether the margin is cost-driven (stable) or
  SR-driven (fragile).
- (ii) "drop-one is systematically positively biased" is **not a theorem** — direction is
  task-pool conditional (§1).
- (iii) the self-oracle floor is a **diagnostic, not a bias estimate** (§3 item 1).

## §6 — Witness plan

- **Disclosure portion (§1 + §2 + §3 items 1,3,4 + §4 + §5)**: witnessable NOW (pre-data),
  same legitimacy class as AMENDMENT_04 — code/prose/disclosure recorded before any eligible
  H1/H10 statistic exists. Witness = git tag `prereg-amendment-06-reproducibility-sensitivity-20260525`
  + push + OSF upload, mirroring the AMENDMENT_04/05 witness pattern.
- **Sensitivity numbers (§3 item 2 empirical fill)**: finalized post clean-replicate; the
  finalizing commit updates this file's `status` and re-stamps the witness tag.
- Cross-link: 实验笔记 §292/§293 · paper_planning Risk 6 · phase1_plan §D4 · preregistration L96-98 (task-level bootstrap) · master_bug_catalog B-12 / B-1858 (element-ID churn) · B-1003 (SE-floor) / B-1855 (H10 realized-not-oracle).
