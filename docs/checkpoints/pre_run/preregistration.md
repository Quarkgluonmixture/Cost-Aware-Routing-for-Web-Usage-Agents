---
type: preregistration
status: draft
created: 2026-05-03
last_revised: 2026-05-13
draft_author: Jiaming
registered_at: <pending advisor sync lock>
registered_git_sha: <pending lock>
witnessed_by: <pending advisor sync>
osf_doi: <pending paper submission stage>
data_lock_until: <pending Phase 1a 24-condition rerun completion (cls+red × B0+B1 × 6 modes)>
scope_revision_2026_05_13: cls+red × B0+B1 × 6 modes = 24 operational conditions across 4 statistical cells; shop deferred to Phase 1b main paper; K-of-N reclassified gate → transparency-only; smoke-gate stopping rule replaced (outcome-independent)
---

# Phantom-SoM Pre-Registration (Draft)

> **Status: draft** — pending advisor sync lock. Once advisor signs (single-line email or co-authored commit), `status` flips to `locked`, `registered_git_sha` records the commit at lock time, and `witnessed_by` records advisor name + lock timestamp. `data_lock_until` records when the Phase 1a 24-condition rerun finishes — between lock-time and completion-time, NO additional analyses may be added to gating-family tests.
>
> **Scope of this DOI claim**: Phase 1a only — 24 operational conditions (2 sites cls+red × 2 models B0+B1 × 6 modes) across 4 statistical cells. The gating hypotheses are **H1 (hero) + H2 (drop-in) + H3 (structural)**. H5/H6 (post-hoc theory) and H7/H8 (router, Phase 2) are retained as **deferred forward stubs in Appendix B — NOT part of this DOI claim**. Mechanism layer-selection disclosure (Stage 2) is in Appendix C.
>
> **Reading order**: §1 epistemic structure → §2 hypotheses (H1-H3 gating + framing rule) → §3 multiple-comparison family declaration → §4 locked analysis choices → §5 exploratory disclosure → §6 witness mechanism → §7 reproducibility → Appendix A decision log → Appendix B deferred hypotheses (H5-H8) → Appendix C mechanism disclosure.
>
> **Companion docs**:
> - `docs/reference/EVIDENCE_LAYER_AUDIT.md` §2 — template + meta-rationale
> - `docs/checkpoints/paper_planning.md` §1 + §2 — paper hook + theory framework
> - `docs/checkpoints/ADVISOR_SYNC.md` §1 — advisor sync prep, lock decision questions

---

## §1 Epistemic Structure (why this pre-registration shape)

This pre-registration adopts a **Hero + Drop-in + Structural + Framing-Rule** hierarchy rather than the more conventional "all hypotheses pre-committed strict" pattern. The rationale (developed via 2026-05-03 design discussion):

1. **Phantom-SoM is the deployment hero**: 4-fold drop-in property (cost ≈ DOM, latency ~50% lower, signal AUROC ≥ baseline, drop-one positive) is the headline practical contribution. This is pre-registered strict.

2. **Phantom space is a structural claim, not a 3-arm deployment claim**: P-text and P-prompt are not pre-registered as deployment-grade routing arms. They are pre-registered as **structural ablation evidence** validating that phantom space (the cube image-off half) is a 2-axis multi-region empirical structure rather than collapsed to the single P-SoM point.

3. **Framing decision is data-conditional, not data-prediction**: paper §1 hook framing depends on which combination of H1-H3 holds. The rule is pre-registered (R1-R5 below) so reviewers can verify the framing-to-data mapping is not post-hoc.

4. **Theory predictions (别扭, capability-modulated reversal) are post-hoc explanatory**: these frameworks were developed *after* observing N=4 pre-Phase-A cells; treating them as pre-registered hypotheses would be epistemically dishonest. Paper prose explicitly marks them as post-hoc.

5. **Multiple-comparison family discipline**: gating tests (PRIMARY + STRUCTURAL) have explicit Holm-corrected family m count. Exploratory tests (EXPLORATORY family + post-hoc) are reported with adjusted p-values for transparency but NOT used to gate paper claims.

This structure compresses garden-of-forking-paths degree-of-freedom to a small, pre-committed set of tests, while preserving epistemic honesty about which findings emerged from data vs were predicted a priori.

---

## §2 Hypotheses

### PRIMARY family (gates paper claim)

#### H1 — Hero deployment claim (P-SoM is hidden routing arm)

**Estimand (locked 2026-05-14, decision "3A")**: H1 estimates the **fixed-effects
inverse-variance-weighted average P-SoM drop-one oracle ceiling lift over the 4
*planned* (site, model) cells** — (cls,B0), (cls,B1), (red,B0), (red,B1). The 4
cells are NOT a random sample from a larger universe of site/model conditions;
they are the specific conditions this study characterizes. Therefore the estimand
is the average effect across *exactly these 4 cells*, not the mean μ of a
hypothetical population. **No between-cell variance τ² is in the estimand** — this
is a deliberate choice (see Appendix A 2026-05-14): it avoids estimating τ² at
k=4 (where DerSimonian-Laird τ² is downward-biased and random-effects Wald CIs are
anti-conservative, per Veroniki et al. 2016 / IntHout et al. 2014), which the prior
random-effects framing required.

**H1 — single PRIMARY gate**: the fixed-effects pooled drop-one effect θ_FE
**significantly exceeds the +1.0pp substantive-effect threshold** via a one-sided
superiority test: reject H0: θ_FE ≤ +1.0pp at α=0.05 (PRIMARY family m=1, no
within-family correction).

This single test implies both (a) θ_FE ≠ 0 and (b) θ_FE point estimate > +1.0pp;
the prior H1(i) "pooled meta ≠ 0" and the separate "magnitude ≥ 1.0pp" check were
redundant with it and have been folded in (2026-05-14 simplification).

**Pooling (operational)**: For each (site, model) cell containing all 6 modes (DOM,
SoM, Vision, P-text, P-prompt, P-SoM), compute oracle ceiling SR over {6 modes}
minus oracle ceiling SR over {5 modes drop P-SoM} per task, then average across the
cell's task pool → per-cell effect θ_i with paired 1000-resample task-level
bootstrap SE_i. Pool via fixed-effects inverse-variance weighting: w_i = 1/SE_i²,
θ_FE = Σ(w_i·θ_i)/Σw_i, SE_FE = sqrt(1/Σw_i). The one-sided superiority z-statistic
is z = (θ_FE − 1.0) / SE_FE. **Why FE Wald is sound at k=4 here**: each per-cell
θ_i is approximately normal (CLT on ~210-234 paired tasks); θ_FE is a linear
combination of 4 approximately-normal estimates → approximately normal. The k=4
fragility of the *prior* design came from τ² estimation, which is now absent.

**Heterogeneity (reported descriptively, NOT in the estimator)**: Cochran's Q, I²,
and a DerSimonian-Laird τ² estimate are computed and reported for transparency
("are the 4 cells consistent?"), and a random-effects pooled estimate is shown as
an Appendix sensitivity row. But they do NOT enter the H1 gate — see the
heterogeneity-conditional rule in the §2 framing-rule block for how high I²
affects *interpretation* (not the FE point estimate).

**Transparency consistency check (NOT gating, reported alongside H1)**: report the
**count of cells (out of 4) whose per-cell drop-one bootstrap CI excludes 0** and
the count individually Holm-significant. No fixed K threshold (at N=4 a "K%
threshold" is fake precision — ⌈0.75×4⌉ = ⌈0.67×4⌉ = 3; the prior K_h1/K_h3
percentages collapsed). A descriptive benchmark "3 of 4 = strong per-cell
consistency" may be used in prose, but it is NOT a decision rule. See §4 + Appendix A.

#### H2 — Drop-in property (P-SoM specifically)

**Scope (revised 2026-05-13 codex stress audit T2; refined 2026-05-14 decision "3A")**:
The R1 framing rule depends on **H2(a) cost** only, and H2(a) is a *by-construction
property with a falsification check* (not a statistical gate — see below). H2(b)
latency and H2(c) signal AUROC are **EXPLORATORY transparency reports** (NOT gating
R1). H2(d) drop-one magnitude is already covered by the H1 superiority gate. Rationale:

- (a) is a by-construction model property (regex filter ≡ no image tokens), verified
  by a falsification check, not a sampling-theory hypothesis test
- (b) latency depends on serving infrastructure (DGX vs A100 vs proxy), not a model
  property; should be characterized but not gated
- (c) signal AUROC depends on routing-signal universe choice (`aggregate_routing_auroc.py`
  top-1 selection), which is exploratory characterization per §5 — including AUROC
  in R1 gating would force a circular dependency
- (d) folded into the H1 superiority gate

**H2(a) — BY-CONSTRUCTION property with falsification check (locked 2026-05-14, decision "3A")**:
P-SoM cost ≈ DOM cost is **not an empirical hypothesis to be gated by a statistical
test** — it is a *by-construction* property. P-SoM's observation is `[SOM_MARKS]` =
a regex filter over the *same* VWA accessibility-tree text the DOM baseline already
consumes, plus *no image tokens*. The token count therefore cannot be substantially
higher than DOM by construction; it is bounded by DOM cost + a regex-pass overhead
that is provably negligible. Running a sampling-theory equivalence test (TOST) or a
K-of-N count on a near-deterministic token-count quantity would be a category error.

- **By-construction argument**: stated in paper §3.2 (regex-filter derivation).
- **Empirical falsification check (pre-registered)**: report observed median cost
  ratio cost(P-SoM)/cost(DOM) per condition in §4. **Falsification threshold**: if
  ANY condition shows median cost ratio > **1.20×**, the by-construction claim is
  contradicted and must be investigated (regex not filtering / hidden overhead /
  tokenizer surprise) before H2(a) can be asserted. The check is a *falsification*
  test, not a confirmation test — passing it (all ratios ≤ 1.20×) does not "prove"
  equivalence, it fails to falsify the by-construction derivation.
- **No K_h2 threshold** — H2(a) is not a per-cell-counted hypothesis test.

**H2(b) — EXPLORATORY transparency report (NOT gating)**:
median latency(P-SoM) ≤ 0.6 × median latency(SoM). Reflects skipping image inference
stage. Reported per cell in §4 table; paper §1 hook can claim "lower latency" via
descriptive characterization, but framing rule R1 does NOT gate on this.

**H2(c) — EXPLORATORY transparency report (NOT gating)**:
top-1 routing-signal AUROC(P-SoM) ≥ AUROC(DOM) − 0.05 (within 5pp). Signal selected
per `aggregate_routing_auroc.py` top-1. Reported per cell in §4 table; paper §1 hook
can claim "usable routing signal" via descriptive characterization, but framing rule R1
does NOT gate on this.

**H2(d) — folded into H1(ii)**:
P-SoM contributes ≥ 1.0pp lift on average via H1(ii) magnitude + superiority gate.

#### H3 — Phantom space 2-axis empirical structural claim

Each phantom-space axis (axis 1 = text payload via P-text; axis 2 = SoM-style prompt via P-prompt) contributes tasks NOT solved by P-SoM, evidencing axis decomposition is empirically non-trivial (i.e., phantom space is a multi-region 2D structure, not a collapsed 0D point).

H3 statistical cells = 4 (one per (site, model)). H3 axis-1 and axis-2 are tested separately within each cell.

- **H3(i) STRUCTURAL GATE** axis 1: fixed-effects inverse-variance pooled mean |P-text ∖ P-SoM| over the 4 planned cells, FE CI excludes 0 (one-sided, α=0.05, m=1 within axis-1 sub-family). Same FE estimand as H1 (decision "3A" 2026-05-14): the 4 cells are the study design, not a population sample; no τ².
- **H3(ii) STRUCTURAL GATE** axis 2: same as H3(i) for |P-prompt ∖ P-SoM|.
- **H3(iii)** Per-cell unique-count noise floor: ≥ 2 tasks (≈ 1pp at N=234 to N=210); 1 task is noise floor, excluded from cell-level pass.

**Transparency consistency check (NOT gating)**: report count of cells (out of 4) whose per-cell unique-count bootstrap CI excludes 0, per axis. No fixed K threshold (see H1 transparency check + §4 K-of-N row — at N=4 a K% is fake precision). Archive caveat (`meta_phantom_lift.md`): P-text drop-in I²=71% and LOO-fragile (one-cell-drop flips Holm) — H3 axes are genuinely more heterogeneous than P-SoM; if descriptive I² > 75%, the heterogeneity-conditional rule caps the hook at R3.

**Test details**:
- Primary gating: bootstrap CI on unique-count, 1000 resamples.
- Secondary report: McNemar exact one-sided directional asymmetry test (informational only — McNemar tests if one axis dominates the other in unique contribution; H3 only requires non-emptiness, not dominance).
- Multiple-comparison: Holm-Bonferroni step-down per axis sub-family (axis 1: m = N_cells; axis 2: m = N_cells).

### EXPLORATORY family (reported with corrections, NOT gating)

#### H4 — P-text / P-prompt drop-one magnitude

Reported per cell + meta-pooled (DerSimonian-Laird) for transparency. Holm-Bonferroni and BH FDR q-values reported. No pre-registered ranking commitment.

Paper §4 prose **must** explicitly flag: "exploratory analysis; not pre-registered for paper hook gating; magnitudes interpreted descriptively."

### POST-HOC family (theory tested on data that motivated it)

#### H5 — 别扭 (mismatch) framework predictions

The 4 distinguishing predictions in 实验笔记 §108.16 are tested against the Phase 1a 24-condition / 4-cell data. The framework was developed after observing pre-Phase-A archive cells; this is **post-hoc**.

Paper §5 prose **must** explicitly flag: "post-hoc theoretical framework, validated on the same data motivating it; no formal significance gating."

#### H6 — Capability-modulated reversal (B0 vs B1 axis preference)

B0 vs B1 ranking direction on text-axis vs image-axis drop-one tested via B0 × B1 × axis logistic GLM interaction term. Post-hoc finding (developed after observing N=4 pre-Phase-A cells).

Paper §7 prose **must** explicitly flag: "post-hoc finding; no pre-registered prediction."

### ROUTER family — H7/H8 — ⚠️ DEFERRED, NOT PART OF THIS DOI CLAIM (logical Appendix B)

> **⚠️ Scope banner (2026-05-14 decision "3A")**: H7/H8 require Phase 2 router data
> that does not exist. They are **not gating the Phase 1a workshop DOI claim** and
> are **not tested in Phase 1a**. They are retained here as a forward stub for a
> future paper-2 preregistration. A reviewer should read H1-H3 (above) as the
> complete gating set for this DOI; H7/H8 below carry no claim weight. (Physical
> relocation to a standalone paper-2 prereg is a follow-up cleanup item.)

#### H7 — Tier 1 oracle router lift over best-single-mode baseline (offline supervised)

Tier 1 router: TF-IDF task-instruction features + binary task features (`has_ref_image`, `has_finish_string_match`) → logistic regression predicting best-mode-per-task. Trained per cell-fold (site-stratified k-fold). Lift = adjusted-SR(router) − adjusted-SR(best-single-mode-baseline) per cell.

- **H7(i)** Pooled DerSimonian-Laird random-effect meta-analysis on lift reaches Holm α=0.05 (PRIMARY family m=1 if paper-1 / SECONDARY informational if paper-2).
- **H7(ii)** ≥ K_h1 of N_cells individually Holm-significant on per-cell lift, bootstrap 95% CI lower-bound > 0.
- **H7(iii)** Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence at margin δ=1.0pp rejected (same δ as H1).

**Test details**:
- 5-fold site-stratified CV on cls+red post-Phase-A task pool (split protocol locked §4 — train/test fold seed + minimum sizes).
- Best-single-mode-baseline = mode with highest mean adjusted-SR on train fold per cell, evaluated on held-out test fold (no test leak).
- Bootstrap 1000 resamples, paired task-level.
- Multiple-comparison: Holm-Bonferroni step-down within H7 sub-family m=N_cells.

**Status**: ⏸️ pending advisor 5/5 lock decision — if paper-1 PRIMARY, H7 gates Section 6 routing claim; if paper-2 deferred, H7 reported as informational with explicit "paper-1 hook does NOT depend on H7-H8".

#### H8 — Tier 2 first-step trigger router (online, test-leak-free)

Tier 2 router: features extracted from agent's first-step observation (task instruction + initial DOM/SoM observation slice + initial action diversity proxy) → predicts which mode to commit for full trajectory. **No test leak**: features use only first-step info, mode commitment thereafter is fixed.

- **H8(i)** Tier 2 router lift over Tier 1 oracle baseline ≥ 0 with bootstrap 95% CI excluding −1.0pp (paper claims Tier 2 ≈ Tier 1 within deployment-grade tolerance, given Tier 2 is leak-free and deployment-realistic).
- **H8(ii)** Tier 2 router lift over best-single-mode-baseline ≥ 1.0pp, ≥ K_h1 cells Holm-significant.

**Status**: ⏸️ pending advisor 5/5 lock — same as H7.

**Companion check** (NOT gating): per-mode AUROC of selected routing signals reported for transparency (Section 6 portfolio characterization, see EXPLORATORY §5).

### FRAMING DECISION RULE (pre-registered, data-conditional)

The paper §1 hook framing maps to data outcomes as follows:

| Rule | Conditions | Paper hook framing | Hook power |
|---|---|---|---|
| **R1** | H1 holds AND H2(a) by-construction property not falsified AND H3(i) holds AND H3(ii) holds (H2(b) latency + H2(c) AUROC reported as exploratory transparency, NOT gating) | "Phantom routing space (M1/M2 2-axis empirical structure); P-SoM as deployment hero, P-text/P-prompt as structural ablation arms validating axis decomposition." | STRONGEST |
| **R2** | H1 holds AND H2(a) not falsified AND only one of H3(i)/(ii) holds | "Phantom routing space (single-axis empirical structure) with P-SoM as deployment hero; remaining axis decomposition theoretical (Zoom 1 architectural argument only)." | MODERATE-STRONG |
| **R3** | H1 holds AND H2(a) not falsified AND neither H3(i)/(ii) holds | "Phantom-SoM is hidden 4th routing arm; M1/M2 axis decomposition supported by Zoom 1 architectural argument only, not empirically validated by ablation." | MODERATE (= 04-30 fallback; workshop-grade) |
| **R4** | H1 holds AND H2(a) **falsified** (median cost ratio P-SoM/DOM > 1.20× in some condition) | "Phantom-SoM partial drop-in (cost-equivalence by-construction claim contradicted by data)" + §4 disclosure + investigation of the falsification. | WEAK; substantial revision |
| **R5** | H1 fails — one-sided **superiority test** fails to reject H0: θ_FE ≤ +1.0pp at α=0.05 (θ_FE = fixed-effects pooled drop-one over 4 planned cells) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |

**Trigger rule update (2026-05-13 → refined 2026-05-14 decision "3A")**: R5 fires only on the single H1 superiority gate failing — no `< K_h1` trigger (K-of-N is transparency-only), no separate "pooled meta ≠ 0" or "magnitude ≥ 1pp" sub-gates (folded into the one superiority test). The estimator is fixed-effects inverse-variance pooling over the 4 planned cells (no DL, no REML, no τ²).

**Heterogeneity-conditional rule (reframed 2026-05-14 under the fixed-effects estimand)**: Under the FE estimand, high heterogeneity does NOT block pooling — the FE average over the 4 planned cells is well-defined regardless of I². But high I² makes "the average" a *less meaningful summary* of 4 genuinely-different cells. Rule: if descriptive I² > 75%, the H1 superiority gate still runs and still determines R5-vs-not, BUT the paper hook is **capped at R3** (cannot claim R1/R2 "empirical structure") and §4 prose must lead with the per-cell forest, presenting θ_FE as a secondary summary. I² ≤ 75% → normal R1-R5 mapping. (This replaces the prior "I²>75% → do not pool" rule, which was incoherent with a superiority test that needs a pooled estimate.)

### §2.4 Power acknowledgment (added 2026-05-14, in-doc per codex prereg-structure review)

Phase 1a has **4 statistical cells** (2 sites × 2 models). This is a small design and the preregistration acknowledges it explicitly:

- **Per-cell power is modest.** Per `power_analysis.md`, at observed phantom-mode effect sizes (1-5pp) and observed adjusted-SR levels (8-15%), per-cell statistical power is ≈ 0.30 — minimum detectable effect at 80% per-cell power is 5-7pp. This is *why* per-cell K-of-N is transparency-only, not a gate.
- **The fixed-effects estimand is the mitigation.** By estimating the average over exactly the 4 planned cells (not a population mean), the design needs no τ² estimation — the k=4 fragility of random-effects τ² (Veroniki et al. 2016) and anti-conservative random-effects Wald CIs (IntHout et al. 2014) do not apply. The FE pooled SE = sqrt(1/Σw_i) aggregates 4 well-powered per-cell estimates (each from 210-234 paired tasks).
- **Honest scope.** The claim is "P-SoM drop-one averaged over cls/red × B0/B1", NOT "P-SoM helps universally". Cross-site / cross-model generalization is Phase 1b (shop) + future work. The per-cell forest is always shown so the reader sees the 4 cells, never just the average.

### §2.5 H1 PASS/FAIL decision flow (added 2026-05-14)

```
1. Compute per-cell drop-one θ_i + paired-bootstrap SE_i for each of the 4 cells.
2. Fixed-effects pool: θ_FE = Σ(w_i·θ_i)/Σw_i,  SE_FE = sqrt(1/Σw_i),  w_i = 1/SE_i².
3. Compute descriptive Q, I², τ² (for transparency + the heterogeneity branch).
4. One-sided superiority test: z = (θ_FE − 1.0) / SE_FE,  p = 1 − Φ(z).
   ├─ p ≥ 0.05  → H1 FAILS → R5 (paper-death / pivot).
   └─ p < 0.05  → H1 PASSES → continue.
5. H2(a) falsification check: any condition with median cost ratio > 1.20×?
   ├─ yes → H2(a) falsified → R4.
   └─ no  → continue.
6. H3 axis-1 + axis-2 FE superiority tests (CI excludes 0):
   ├─ both pass → R1 candidate.
   ├─ one passes → R2 candidate.
   └─ neither   → R3 candidate.
7. Heterogeneity cap: if descriptive I²(H1) > 75% → cap candidate at R3
   (lead §4 prose with per-cell forest; θ_FE is a secondary summary).
8. Final R-rule = the (possibly capped) candidate. Transparency: report
   n-of-4 per-cell CI>0 counts alongside, with NO threshold.
```

---

## §3 Multiple-Comparison Family Declaration

**PRIMARY family** (gating paper hook) — UPDATED 2026-05-14 (decision "3A"):
- **H1**: one-sided fixed-effects superiority test (H0: θ_FE ≤ +1.0pp vs H1: θ_FE > +1.0pp) rejected at α=0.05 — **m = 1, single test** (the prior H1(i) "pooled meta ≠ 0" + separate "magnitude ≥ 1.0pp" sub-tests were redundant with this and have been folded in).
- **H2(a)** is NOT in this family m-count — it is a *by-construction property with a falsification check* (median cost ratio > 1.20× in any condition → falsified), not a sampling-theory hypothesis test. See §2 H2(a).
- Method: PRIMARY family has m = 1; no within-family Holm correction needed.

**STRUCTURAL family** (gating phantom-space framing) — UPDATED 2026-05-13:
- H3(i) pooled axis-1 meta on N=4 cells: m = 1.
- H3(ii) pooled axis-2 meta on N=4 cells: m = 1.
- Method: Holm-Bonferroni step-down per axis sub-family.
- Rationale: structural claim is weaker than deployment, separate family avoids inflating PRIMARY family m count.

**TRANSPARENCY reporting** (NOT gating, NOT a family with thresholds — reported in §4 per-cell table for reviewer transparency):
- For H1 and for each H3 axis, report the **count of cells (out of 4)** whose per-cell bootstrap CI excludes 0, and the count individually Holm-significant.
- **No fixed K threshold** — at N=4, a "K%" threshold is fake precision (⌈0.75×4⌉ = ⌈0.67×4⌉ = 3; the prior K_h1=0.75 / K_h3=0.67 distinction collapses). A descriptive prose benchmark "3 of 4 = strong per-cell consistency" may be used, but it is not a decision rule.
- **Rationale**: power analysis (`docs/analysis/cross_sites/power_analysis.md`, pre-data) shows per-cell power at observed 1-3pp effect sizes is ~0.30; a K-of-N gate would be statistically dysfunctional in this regime. The PRIMARY gate is the fixed-effects superiority test (m=1); per-cell counts are pure transparency. See Appendix A 2026-05-13 + 2026-05-14 entries.

**ROUTER family** (gates Section 6 routing claim — pending advisor 5/5 paper-1-vs-paper-2 lock):
- H7(i) pooled meta lift: m = 1 (no within-family correction).
- H7(ii) per-cell Tier 1 lift Holm: m = N_cells.
- H7(iii) folded into H7(i) magnitude/TOST.
- H8(i) Tier 2 vs Tier 1: m = 1.
- H8(ii) Tier 2 vs best-single-mode-baseline: m = N_cells.
- Method: Holm-Bonferroni step-down per H-sub-family.
- **Status**: ⏸️ if paper-1 PRIMARY, ROUTER family gates §6 contribution; if paper-2 deferred, reported as informational with paper-1 hook independence explicit.

**EXPLORATORY family** (NOT gating, reported only):
- H4 P-text/P-prompt drop-one per cell: m = 2 × N_cells.
- Best-signal-per-mode characterization (Register III AA, Section 6 portfolio finding): per (mode, signal) AUROC reported, Holm-corrected within mode for transparency.
- Method: Holm-corrected and BH q-value reported for transparency.
- **Paper hook does NOT depend on these tests passing.**

**POST-HOC** (no correction, explicit disclosure):
- H5 别扭 4 predictions.
- H6 capability-modulated reversal interaction GLM.
- Disclosed as "post-hoc theoretical analysis tested on motivating data" in paper prose.

---

## §4 Locked Analysis Choices (pre-data)

| Choice | Value | Rationale |
|---|---|---|
| **Primary metric** | Oracle ceiling SR pp lift (binary, paired) | Standard routing-arm contribution metric |
| **CI method** | 1000-resample task-level paired bootstrap, **percentile** intervals (BCa as sensitivity check, not primary) | Existing infra in `aggregate_phantom_lift.py`. Percentile chosen primary because: (a) paired-bootstrap on bounded proportion (SR ∈ [0,1]) → BCa acceleration estimate is unstable at small N per cell; (b) Cohen's h transformation already symmetrizes; (c) percentile is the canonical reporting in WebArena/VWA precedent. BCa shown as appendix sensitivity check. |
| **Bootstrap resampling unit** | **Task-level** (not episode-level, not run-level) | Each (task_id) drawn with replacement N times; same task across modes drawn together to preserve pairing. This is the standard unit for adjusted_success comparisons in VWA/WA. Episode-level would break pairing; run-level would over-conservatively widen CIs. |
| **Bootstrap clustering** | **Single-level (task_id)** for primary, no nested cluster (cell × site) bootstrap | Justification: meta-analysis at cell level is separate (`aggregate_phantom_meta.py` random-effects + I²/τ²); within-cell bootstrap only re-samples tasks. Multi-level cluster would double-count uncertainty already captured by random-effects meta. Lock: percentile + task-id unit + no nested cluster (B2 lock 2026-05-09). |
| **Sig threshold** | Holm α=0.05 within respective family | FWER control |
| **Effect size (binary)** | Cohen's h with bootstrap CI | Standard for proportion comparisons |
| **Effect size (continuous)** | Cohen's d with bootstrap CI | For cost/latency H2(a)(b) |
| **H1 superiority threshold δ** | **1.0pp** | H0 threshold for the one-sided fixed-effects superiority test (H0: θ_FE ≤ +1.0pp). Archive `meta_phantom_lift.md` P-SoM pooled drop-one +2.34pp clears δ=1.0pp (z≈2.5) but LOO-borderline (one-cell-drop p≈0.044-0.046) — δ=1.0pp is the floor, not raisable. ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor. Decision "3A" 2026-05-14 |
| **K-of-N per-cell consistency** | **Transparency count, NO threshold** | Report count of cells (out of 4) with per-cell CI excluding 0 + count individually Holm-sig, for H1 and each H3 axis. At N=4 a K% threshold is fake precision (⌈0.75×4⌉=⌈0.67×4⌉=3) — prior K_h1=0.75 / K_h3=0.67 ratios retired 2026-05-14. Descriptive benchmark "3/4 = strong consistency" allowed in prose, not a decision rule |
| **H3 unique-count floor** | **≥ 2 tasks per cell** | 1 task is sampling noise; 2 tasks ≈ 1pp at N=234 |
| **Cell inclusion (Phase 1a main)** | Phase A post-fix only (commit ≥ 3c15cd7), cls + red sites only, all 6 modes per (site, model) cell freshly rerun | Bug-clean rerun + workshop-target scope (shop deferred to Phase 1b) |
| **Cell inclusion (Phase 1b main paper)** | Phase A post-fix rerun of shop × B0+B1 × 6 modes (12 conditions added on top of Phase 1a 24 conditions) | Cross-site expansion lever for main paper, post-data R1 vs Option D framing decision |
| **Cell inclusion (Appendix D)** | Archived pre-Phase-A data as robustness check | Symmetric contamination disclosure |
| **N inclusion floor** | ≥ 100 ep per (condition) | Statistical power baseline |
| **FP filter primary** | na_fp + eval_fp combined | Per 实验笔记 §95 (visual_fp deprecated — no lit precedent, boundary-undecidable, over-filters 95.3% VWA tasks). Code: `compute_adjusted_success()` returns `fp_reason ∈ {'', 'na_fp', 'eval_fp'}` (`p79/experiment/analysis.py:52`) |
| **FP filter sensitivity** | 3 variants reported (raw_SR / +na_fp only / +na_fp+eval combined) | Robustness disclosure. visual_fp is NOT in the ladder — see §95 decision rationale |
| **Non-visual subset robustness** | 43 VWA + 480 WA = 523 manually-audited non-visual tasks (`docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py`) | Replaces deprecated visual_fp; Appendix D sensitivity check |
| **Mode operational definitions** | 6 modes per paper §3 (text format × prompt × image): DOM (AXTree+DOM-prompt+no image) / SoM ([SOM_MARKS]+SoM-prompt+image) / Vision (no text+image) / P-text ([SOM_MARKS]+DOM-prompt+no image) / P-prompt (AXTree+SoM-prompt+no image) / P-SoM ([SOM_MARKS]+SoM-prompt+no image) | Stipulative — **no post-hoc episode reclassification**. Episodes systematically excluded per (FP filter / N-floor / data-corruption flag), never redefined which mode they belong to. Edge cases (empty AXTree / 0 marks / OCR-empty) follow `condition_meta.json` declared mode |
| **Routing signal universe** | `aggregate_routing_auroc.py` enumerated set: ep_mean_verbalized / ep_min_verbalized / max_repeat_streak / action_diversity / url_revisit_count / url_revisit_max / action_unique_types / url_unique_count / ep_mean_logprob / ep_min_logprob (last 2 B1-only) | **No post-hoc engineered features** for router input. Best-signal-per-mode characterization is exploratory (§5) — paper §6 portfolio finding, not pre-registered prediction |
| **Router train/test split** | 5-fold site-stratified CV on cls+red post-Phase-A task pool, seed=42, min test fold ≥ 40 tasks | Reproducible split via `scripts/analysis/router_split.py` (TBD). **Test fold predictions use ONLY train-fold mode rankings** to prevent oracle leak. Pending advisor 5/5 sync alternative: leave-one-site-out (LOSO) — test cls hold-out trained on red, vice versa |
| **Failure-mode classification rubric** | 5-bucket: `early_finish` / `wrong_commit` / `visual_hijack` / `click_loop` / `persistent_error` per `docs/analysis/disagreement_clusters.md` decision tree | Pre-data inter-annotator agreement target Cohen κ ≥ 0.7 on 30-task pilot (codex prompt + 1 human spot-check). Buckets remain in the rubric but the paper §1 "+43.7pp B0/B1 capability shift" prose was dropped 2026-05-09 (third contribution cut from paper). Failure-mode classification still used for §8 limitations and supplement S.X if needed. |
| **N_conditions Phase 1a (operational)** | **24 conditions** = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM). Each condition launched fresh post-fix via `scripts/queues/queue_phase1_paper_grade.sh` (renamed 2026-05-13 from `queue_16cell_paper_grade.sh`; current scope = 24 conditions Phase 1a + 12 conditions Phase 1b deferred). Sequence: B0 → B1 per site (shared user account); cls + red parallel chains | ✅ **Student-decided 2026-05-13** post-codex stress audit. Workshop-targeted (cls + red only, shop deferred to Phase 1b for main paper). Replaces prior 16-cell phantom-only scope that lacked baseline DOM/SoM/Vision rerun (codex Flaw 1) |
| **N_cells statistical (H1/H3 stratification)** | **4 cells** = (site, model) tuples: (cls, B0), (cls, B1), (red, B0), (red, B1). Drop-one computed per cell using all 6 modes; pooled via **fixed-effects inverse-variance weighting** over the 4 planned cells (decision "3A" 2026-05-14 — NOT random-effects DL/REML; the 4 cells are the design, not a population sample, so no τ²) | Cell = paired-test stratification unit (one per (site, model)), distinct from "condition" (one per (site, model, mode)). 4 cells × 6 modes = 24 conditions |
| **N_conditions Phase 1b (main paper, deferred)** | **+12 conditions** = shop × 2 models × 6 modes. Launches after Phase 1a workshop submission to feed main paper R1 / Option D framing decision. N_cells statistical becomes 6 (= 3 sites × 2 models) when Phase 1b lands | Phase 1b is additive; workshop §1 hook does NOT depend on Phase 1b. Main paper §1 hook upgrade R3 → R1 conditional on shop replicating P-SoM 4-fold within ±2pp tolerance |
| **Best-single-mode baseline (H7/H8 anchor)** | Per cell: mode with highest mean adjusted-SR on train fold | Used as comparison anchor for router lift; **train/test split-stratified** to prevent test leak |
| **Missing-data / crashed-episode policy** (audit B6) | (a) Crashed episodes (uncaught exception, OOM, timeout > 30 min, browser crash) **excluded from paired-N denominators**, **NOT imputed** to success or failure. (b) Episodes with `not_logged_in` or `auth_drift` flag at termination excluded after watchdog refresh fails 3 retries (per `experiment_watchdog.py`). (c) Missing artifacts (no `obs.txt` / `screenshot_annotated.png` at step k) excluded from per-step analyses, NOT imputed. (d) Per-cell exclusion count + reason histogram reported in Appendix C. | Listwise deletion only; mean imputation introduces bias for SR proportions, hot-deck imputation breaks paired-N pairing. Crashed-episode imputation as success/failure would inflate Type I/II error. Lock 2026-05-09. |
| **Stopping rules / contamination halt criteria** (audit B7, REVISED 2026-05-13 to remove outcome-dependent bias per codex Flaw 6) | (a) **Pre-launch**: `make pre-launch-check` validates seed configured + HF SHA pinned + git working tree clean + GPU available + disk free > 20GB; failure halts launch (per audit C10). (b) **Smoke-test gate (outcome-INDEPENDENT)**: first 10 episodes per condition must show auth-state `logged_in=True` on all 10 AND ≥ 9 of 10 episodes produced complete artifact bundle (`obs.txt` + `screenshot.png` + `condition_summary_v2` increment + JSONL flush) AND evaluator returned a parseable verdict (success / failure / `ua_match` N/A — any of these is fine, **success rate itself is NOT checked**). Failures halt for auth refresh / artifact pipeline debug, NOT for low SR observation. Rationale: outcome-dependent smoke gate biases low-SR cells upward (a true 5-10% SR cell has 35-60% probability of "0 successes in first 10" by binomial chance and would be invalidly restarted). (c) **Auth/site contamination halt**: ≥ 5 consecutive episodes with `not_logged_in` ⇒ stop cell, refresh auth, archive partial run as `_dirty_partial`, restart fresh. (d) **Eval drift halt**: if rerun on identical archived episode produces SR delta > 5pp via `validate_run.py --strict`, freeze cell + investigate evaluator code. (e) **OOM / hardware halt**: 3 consecutive job failures ⇒ stop cell, document hardware in incident log, manually re-queue with diagnostic output. | Halt rules protect data purity; halted cells restarted only after root-cause documented in `master_bug_catalog.md` + bug fix committed. Lock 2026-05-09; smoke gate revised 2026-05-13 to outcome-independent variant. |
| **Pooling estimator + heterogeneity pre-spec** (audit B8, REVISED 2026-05-14 decision "3A") | (a) **Primary estimator**: **fixed-effects inverse-variance weighted average over the 4 *planned* (site, model) cells** — w_i = 1/SE_i², θ_FE = Σ(w_i·θ_i)/Σw_i, SE_FE = sqrt(1/Σw_i). The 4 cells are the study's design, NOT a random sample from a population; the estimand is the average over exactly these cells, so no between-cell variance τ² is in the estimand. This avoids estimating τ² at k=4 (DerSimonian-Laird τ² downward-biased + random-effects Wald CIs anti-conservative at k<10 per Veroniki et al. 2016 / IntHout et al. 2014). (b) **Heterogeneity reporting (descriptive only, NOT in estimator)**: report Cochran Q, I², a DL τ² estimate, and a random-effects pooled estimate as an Appendix sensitivity row — all for transparency, none entering the H1 gate. (c) **Interpretation thresholds**: I² < 25% = "pooled FE average is a meaningful summary"; 25-75% = "report FE average + per-cell forest together"; > 75% = "FE average is a weak summary of 4 genuinely-different cells — §4 prose leads with per-cell forest, hook capped at R3 (see §2 heterogeneity-conditional rule); the FE superiority gate still runs and still determines R5-vs-not". (d) when I² > 50%, report meta-regression by site / by model. | Estimand-first design (codex prereg-structure review 2026-05-14): choose the estimand (FE average over planned cells) before the estimator. Fixed-effects pooling is valid at k=4 because θ_FE is a linear combination of 4 approximately-normal per-cell estimates; the k=4 fragility was a τ²-estimation artifact, now absent. Per-cell forest always shown. |
| **K-of-N rule scope** (audit B9, RETIRED-AS-THRESHOLD 2026-05-14 decision "3A") | K-of-N is a **pure transparency count with no threshold** — report "n of 4 cells with per-cell CI > 0" + "n of 4 individually Holm-sig" for H1 and each H3 axis. It is NOT a gate, NOT a family with an m-count, NOT a pass/fail rule. **Primary gate** = the single one-sided fixed-effects superiority test (H0: θ_FE ≤ +1.0pp at α=0.05). Per `power_analysis.md` §3-§5, per-cell power at observed 1-3pp effects ≈ 0.30 → any K-of-N gate is dysfunctional; and at N=4 the K_h1=0.75 / K_h3=0.67 ratios are indistinguishable (both ⌈·×4⌉=3). The prior 0.75/0.67 values are retired; only the descriptive count remains. | Audit B9 (2026-05-09) first reframed K-of-N as transparency; 2026-05-13 propagated it; 2026-05-14 decision "3A" retires the percentage thresholds entirely (fake precision at N=4) — only the raw count is reported. Pre-data; recorded for OSF witness audit trail. |

---

## §5 Exploratory (NOT pre-registered, paper must explicitly flag)

The following analyses are exploratory and cannot be used to gate paper claims. Paper prose **must** mark them explicitly as "exploratory" or "post-hoc":

- Per-task category × mode heatmap exploration (`fig0e`)
- Mechanism per-task qualitative analysis (`mechanism_per_task.json`)
- Axis 3 image-axis 8-channel decomposition (paper §5 axis 3 framework)
- 别扭 framework (H5) — post-hoc, theory developed on motivating data
- Capability-modulated reversal (H6) — post-hoc cross-capability finding
- **Best-signal-per-mode characterization** (Register III AA novelty, Section 6 portfolio finding): which routing signal works best for which mode is reported as exploratory characterization, NOT pre-registered prediction. Per-(mode, signal) AUROC table reported with Holm correction within mode for transparency.
- **Router feature engineering exploration beyond locked signal universe** (§4): any new feature added post-data-lock is exploratory, NOT gating H7/H8 claim.
- **Cross-site asymmetry as site-class adaptive routing primitive** (Register IV HH novelty, §1 + §6): reported as exploratory framing of cls/red mode-preference reversal, NOT pre-registered prediction.
- **Phantom space generalizability speculation** beyond web agent (Register I J, §8 future work): clearly marked as speculative discussion.
- Any post-hoc cell subsetting beyond H1-H8 family scope
- Any analysis added after `data_lock_until` timestamp in this preregistration's frontmatter

### §5.X Post-hoc Layer Selection Disclosure (Stage 2 Mechanism, audit G5) — ⚠️ DEFERRED (logical Appendix C)

> **⚠️ Scope banner (2026-05-14 decision "3A")**: This Stage 2 mechanism
> layer-selection disclosure is **not part of the Phase 1a phantom-space-phenomenon
> DOI claim** (which is H1-H3). It is retained here as the mechanism-paper
> disclosure stub; the Phase 1a workshop prereg gates on H1-H3 only. (Physical
> relocation to a standalone mechanism-paper prereg is a follow-up cleanup item.)

Stage 2 mechanistic activation patching identified mid-layer disruption peaking
at **L17** (3 of 4 cells Holm-significant on `token_overlap_to_target`, p_Holm <
0.05; cell D L11+L17 strongest p_Holm = 0.006/0.008 \*\*). The L11/L17 layer
selection has the following **explicit pre-vs-post-hoc structure**:

| Stage | Layer rationale | Status |
|---|---|---|
| **Stage 2A logit_shift pilot** (5-task aggregate, 笔记 §111.5) | L17 emerged as peak in independent `logit_shift` metric | **Hypothesis-generating** — first-pass discovery |
| **§111 task-0 single-task patching** | L11 flipped 93% match in N=1 task | **Hypothesis-generating** — distribution outlier (acknowledged 笔记 §117.4) |
| **Stage 2B 24-task aggregate (cell A)** | L17 Holm-significant (p_Holm = 0.011 \*\*) — confirmed Stage 2A peak | **Confirmatory** — independent metric agreement |
| **Stage 2C reverse 15-task (cell B)** | L11 + L17 Holm-significant — direction-paired confirmation | **Confirmatory** |
| **Cell D (rev × strong-tier 24)** | L11 + L17 strongest (p_Holm = 0.006/0.008 \*\*) | **Confirmatory** — cross-tier replication |

**Disclosure**: Layers L11 and L17 were not pre-registered before Stage 2 data
collection; they emerged from Stage 2A pilot (the *hypothesis-generating* phase)
and were confirmed by Stage 2B/2C scaled-up data (the *confirmatory* phase). To
mitigate the multiple-comparison concern, all per-direction tests use Holm-
Bonferroni correction across the canonical layer grid (L0/5/11/17/23/29 vs L35
baseline) — this catches the "any layer might pop" multiple-testing concern.

**Reviewer-defense**: We do NOT claim pre-registered layer prediction. We claim
that the **same** mid-layer region (L11-L17) emerges across (a) Stage 2A
logit_shift, (b) Stage 2B forward overlap-to-target, (c) Stage 2C reverse,
(d) Cell D cross-tier rev-on-strong. Convergence across 4 independent analysis
paths constitutes confirmatory evidence even without pre-registered layer
prediction. Cell E random-injection control (G6) further demonstrates content-
specificity, ruling out generic-injection alternative explanations.

**Future paper-grade improvement** (deferred to next iteration): full **leave-
one-out layer-selection** robustness — re-run patching on per-cell holdout
that excludes the layer that informed selection on the training cell, then
report the mid-layer pattern under that holdout.

---

## §6 Witness Mechanism

### (a) Internal witness — Git commit + advisor email

1. Advisor sync session: lock **9 commit decisions** (expanded 5/4 audit + 2026-05-13 + 2026-05-14 "3A" revisions):
   - (1) **Estimand = fixed-effects average over 4 planned (site, model) cells** (decision "3A" 2026-05-14). The 4 cells are the study design, not a population sample → no τ², no DerSimonian-Laird, no REML. Dissolves the k=4 random-effects fragility.
   - (2) **K-of-N = pure transparency count, NO threshold** (2026-05-14). Prior K_h1=0.75 / K_h3=0.67 ratios retired (at N=4, ⌈0.75×4⌉=⌈0.67×4⌉=3 — fake precision). Report n-of-4 per-cell CI>0 counts descriptively.
   - (3) **δ=1.0pp** — H0 threshold for the one-sided fixed-effects superiority test (H0: θ_FE ≤ +1.0pp). H1 is this single test (prior H1(i)+magnitude folded in). H2(a) cost is a by-construction property + falsification check (>1.20× cost ratio), distinct from δ.
   - (4) **Cell inclusion**: Phase 1a = cls + red × B0+B1 × 6 modes (Phase A post-fix only); Phase 1b shop deferred
   - (5) **Witness mechanism**: Git + advisor email + OSF DOI
   - (6) **N_conditions Phase 1a final scope**: **24 operational conditions** (= 2 sites × 2 models × 6 modes) across **4 statistical cells** (= (site, model) tuples) — student-decided 2026-05-13 post-codex stress audit, replaces prior 16-cell phantom-only scope. Advisor email witness pending
   - (7) **Smoke-gate revision** (2026-05-13): outcome-independent (auth + artifact + evaluator parseability only), no SR-based restart
   - (8) **Router H7/H8 = DEFERRED** (paper-2, not part of this DOI claim — see §2 ROUTER-family banner / Appendix B)
   - (9) **Train/test split protocol** (paper-2 router scope): 5-fold site-stratified CV vs leave-one-site-out (LOSO)
   - Plus lock gating H-list: **H1-H3 only** for this DOI; H5/H6 post-hoc disclosure + H7/H8 deferred.
2. Update this file frontmatter: `status: draft` → `status: locked`, fill `registered_at`, `registered_git_sha`, `witnessed_by`.
3. Git commit this file.
4. Advisor sends single-line confirmation email: "I witness pre-registration of phantom-SoM gating hypotheses (H1-H3) and the 9 lock decisions as of <git SHA> <date>." Email archived in `.witness/preregistration_witness.eml` (gitignored, local-only).

### (b) External witness — OSF DOI (optional, paper-time)

Approximately 1 week before paper submission. The detailed 8-step DOI workflow +
artifact-freeze registry is in **`osf_lock_manifest.md` §3** — not re-listed here to
avoid drift. Summary: upload the locked `preregistration.md` + companion docs, mint
the OSF DOI, paper §1 footnote cites "Gating hypotheses (H1-H3) pre-registered prior
to the Phase 1a 24-condition rerun (OSF DOI X.YYYY/osf.io/zzzz, Git SHA abc123,
witnessed by [advisor name] on YYYY-MM-DD)."

---

## §7 Reproducibility Scope Statement (audit A14, F3)

**Public release scope** — what reviewers / replicators can reproduce from the released artifact:

| Component | Reproducibility tier | Mechanism |
|---|---|---|
| **B1 (Qwen3-VL-4B local)** | **Fully reproducible** byte-identical | HF model SHA pinned (`ebb281ec70b05090aa6165b016eac8ec08e71b17`) + greedy decoding + seed=42 (`configs/exp_v2_base.yaml`) + `_seed_global_rng()` per (cond, seed) iteration + env_snapshot.json per run + git commit SHA in run_manifest. Re-running produces byte-identical action traces, hidden states, and aggregate SR. |
| **B1 mechanistic Stage 2** | **Fully reproducible** | Same as B1 plus `--random-seed 42` for `--random-inject` (cell E). `archive_subset_b1_{cls,reddit}/` (curated mirage tasks + cached observations + screenshot_annotated) committed for cross-machine replication without needing full archive. |
| **B0 (Qwen3-VL-235B-A22B via proxy API)** | **Verifiable from traces, replayable subject to API access** | All B0 episodes log full request/response traces + temperature=0 server-side. Re-running depends on: (a) proxy API endpoint availability, (b) model server-side determinism (best-effort, not guaranteed at temperature=0). For paper claims, B0 is "one controlled stochastic sample with bootstrap task uncertainty" — replicators verify via released traces or rerun under same proxy / Anthropic-native API access. |
| **VWA environment** | **Reproducible given containers** | VWA Docker images pulled at submodule commit SHA (recorded at lock time per audit A5/F8 remediation). Reset-before-each-cell protocol (`RESET_BEFORE=1`) ensures clean start state. Site-state snapshot pre/post-cell as additional gate (audit C3 pending). |
| **Evaluator** | **Fully reproducible** byte-identical | `evaluator_code.combined_sha256` recorded per run. T0/T1/T2/T3 evaluator-change protocol (`evaluator_change_protocol.md`) governs post-lock changes — same paper requires dual-reporting for any T0 fix. |
| **Mechanism analysis (Stage 2 patching)** | **Fully reproducible** | Greedy decoding + seed=42 + Holm-corrected paired t-test + 1000-resample percentile bootstrap (seed=42 in `stage2_layer_significance.py`). Per-task per-layer `patching_continuation_results.json` released for re-aggregation. |

**Scope claim language for paper §3**:

> "All B1 (local Qwen3-VL-4B) experiments, including agent traces, mechanistic activation patching, and aggregate analysis, are fully reproducible given the released code (commit SHA), pinned HF model revision, and seed configurations. B0 (proxy-API Qwen3-VL-235B-A22B) results are verifiable from released traces and replayable subject to API access; B0 server-side decoding determinism is best-effort under temperature=0 and reported as a single controlled stochastic sample with task-level bootstrap uncertainty. The VWA environment is reproducible given the pinned VWA submodule commit and Docker images. Cross-benchmark (WebArena) results are out of scope for this paper unless explicitly reported in the appendix."

**External validity scope (audit F3)**:

> "Empirical claims are scoped to the **Qwen-family VWA characterization**: Qwen3-VL-4B (B1) and Qwen3-VL-235B-A22B (B0) on VisualWebArena classifieds / reddit / shopping. Cross-benchmark generalization (WebArena 480 tasks) and cross-model-family generalization (Llama-VL, GPT-4o-V, Gemini-Pro-VL) are explicitly future work. Mechanistic Stage 2 findings are scoped to the curated mirage-disagreement task tiers (composite score-based curation per `curate_mirage_tasks.py`) on classifieds (and reddit if cells F/G replicate); broader phantom-routing-space mechanism universality is conditional on the 2x2 + cross-site control results."

---

## Appendix A — Decision Log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-03 | Pre-registration framework reframed from "3-arm strict" to "Hero + Structural + Framing-rule" | User push back on a-priori 3-arm claim being epistemically dishonest given P-text/P-prompt findings emerged from data; new framework cleaner |
| 2026-05-03 | H3 structural test changed from McNemar exact (asymmetry) to bootstrap CI (non-emptiness) | McNemar tests directional dominance (which axis dominates), but H3 only requires non-empty unique contribution; bootstrap CI on count > 0 is the right test |
| 2026-05-03 | TOST δ = 1.0pp locked (was 0.5pp draft) | 0.5pp = 1 task in N=234 too liberal; 1.0pp = 2 tasks ≈ bootstrap SE noise floor; statistically principled |
| 2026-05-03 | K_h1 = 0.75 cell-pass threshold for H1 | Allows ~25% capability-outlier cells; not so strict as to break on single-cell noise |
| 2026-05-03 | K_h3 = 0.67 cell-pass threshold for H3 | Lower than K_h1 because structural < deployment commit |
| 2026-05-03 | Disconfirmation rule changed from "any cell fail" to data-conditional R1-R5 framing rule | "Any cell fail" too strict given single-cell power limits; framing rule maps data outcomes to paper hook revisions transparently |
| 2026-05-04 | Pre-registration scope expanded — added H7-H8 router family + 6 §4 lock entries (mode operational defs / routing signal universe / train-test split protocol / failure-mode classification rubric / N_cells final scope / best-single-mode baseline anchor) | User audit prompt 5/4: "preregistration.md 还需要锁 Held-out router claim / router baselines train-validation-test split / routing signals / mode definition 这些吗". Claude added 2 more (failure-mode rubric / N_cells). Deferred 3 advisor lock decisions: (a) H7-H8 router family paper-1 vs paper-2 / (b) N_cells 13/14/16 final / (c) split protocol k-fold vs LOSO. Witness §6 expanded from 5 commits → 8 commits |
| 2026-05-05 | Advisor sync 5/5 partial outcome — early-stop A locked (cancel全 mechanism); compute path locked (advisor 5090 → Rancher H100 → RunPod backup); paper split direction discussed but Mechanistic-nested-vs-independent + threshold detail not finalized due to network drop | Advisor explicit confirm early-stop cancel + compute paths; paper split + threshold lock deferred to email follow-up via `docs/checkpoints/advisor_sync_5_5_followup.md` |
| 2026-05-05 | **N_cells = 16** (student-decided post-5/5 sync, advisor email witness pending) | 14 (pre-sync default) → 16 to add B1 shop × {phantom_text, phantom_som} 2 cells for cross-capability shop coverage. K_h1 threshold count: ⌈0.75 × 16⌉ = 12. K_h3 threshold count: ⌈0.67 × 16⌉ = 11 |
| 2026-05-13 | **Codex stress audit triggered 6 paper-grade design fixes** (pre-launch): (a) scope reframe 16-cell phantom-only → 24-condition / 4-cell Phase 1a (cls+red×B0+B1×6modes), Phase 1b shop deferred to main paper; (b) K-of-N reclassified gate → transparency-only (power analysis showing dysfunction at < 7pp effects, re-propagated to H1/H3/R5/§6); (c) H1 drop-one definition disambiguated (oracle ceiling lift with-vs-without P-SoM, per (site, model) cell paired bootstrap); (d) smoke-gate B7 revised outcome-independent (no SR-based restart bias); (e) cell terminology disambiguated ("cell" = 4 statistical strata for K-of-N/meta input, "condition" = 24 operational launch units); (f) Phase 1b shop scope-expansion lever for main paper R3→R1 framing decision | Codex CLI hostile reviewer audit (`docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md`, lean prompt no-enumeration, cross-AI complementary to prior Claude reviews); 6 HIGH severity findings + 3 probable concerns. Workshop-targeted Phase 1a launch this week; main paper Phase 1b after workshop submission |
| 2026-05-14 | **Decision "3A" — estimand + H2(a) reframe** (student-decided after Claude + codex cross-think on bug-fix-pre archive data). **3 = H2(a) by-construction**: P-SoM cost ≈ DOM is a by-construction token-accounting property (regex-filtered AXTree subset, no image tokens) verified by a falsification check (>1.20× cost ratio → falsified), NOT a sampling-theory gate. **A = fixed-effects estimand**: H1/H3 pool drop-one via fixed-effects inverse-variance weighting over the 4 *planned* cells (the cells are the design, not a population sample) → no τ², no DerSimonian-Laird, no REML — dissolves the k=4 random-effects fragility (Veroniki et al. 2016 τ² bias; IntHout et al. 2014 anti-conservative RE Wald CI) that codex /stress v6 F1/F2 surfaced. **Knock-on cleanups**: H1 simplified to a single one-sided FE superiority test (prior H1(i) pooled-meta-≠0 + magnitude check were redundant, folded in); K-of-N percentage thresholds (K_h1=0.75/K_h3=0.67) retired — pure transparency counts only (at N=4 the ratios are indistinguishable); heterogeneity-conditional rule reframed (I²>75% caps hook at R3, does not block FE pooling); H7/H8 router + §5.X mechanism disclosure bannered as DEFERRED (Appendix B/C, not part of this DOI claim); §2.4 in-doc power acknowledgment + §2.5 H1 PASS/FAIL decision flow added; §6(b) OSF workflow → reference osf_lock_manifest.md §3. | Claude /stress v6 + 2 codex cross-think rounds (`threshold_rethink_FINAL_2026-05-14.md` + `prereg_structure_review_FINAL_2026-05-14.md`); archive data (`meta_phantom_lift.md` P-SoM I²=0% pooled +2.34pp; `power_analysis.md` per-cell power ≈0.30 at 1-5pp) grounds the calibration. Estimand-first principle: choose what you estimate before choosing the estimator. |
| \<pending advisor sync\> | \<witness 9 lock decisions per §6 — incl. estimand=FE-over-4-planned-cells, K-of-N transparency-count-no-threshold, δ=1.0pp FE superiority, H2(a) by-construction, H1-H3 gating only, H7/H8 deferred\> | \<advisor email reply timestamp + Git SHA at lock\> |
| \<pending advisor email follow-up\> | \<witness K_h1=0.75 transparency / K_h3=0.67 transparency / TOST δ=1.0pp / N_conditions=24 (Phase 1a) / N_cells=4 / split protocol / paper split / Phase 1b shop / outcome-indep smoke gate / per follow-up doc Q1-Q11\> | \<email reply timestamp + Git SHA at lock\> |
