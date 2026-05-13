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

> **Status: draft** — pending advisor sync lock. Once advisor signs (single-line email or co-authored commit), `status` flips to `locked`, `registered_git_sha` records the commit at lock time, and `witnessed_by` records advisor name + lock timestamp. `data_lock_until` records when 16-cell rerun finishes — between lock-time and completion-time, NO additional analyses may be added to gating-family tests.
>
> **Reading order**: §1 epistemic structure (why this framework) → §2 hypotheses (H1-H6 + framing rule) → §3 multiple-comparison family declaration → §4 locked analysis choices → §5 exploratory disclosure → §6 witness mechanism.
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

P-SoM drop-one oracle ceiling lift > 0 across statistical cells (each cell = one (site, model) stratum), satisfying ALL two PRIMARY sub-conditions:

- **H1(i)** Pooled DerSimonian-Laird random-effect meta-analysis on N=4 (site, model) cells reaches significance at Holm α=0.05 (PRIMARY family m=1 test, no within-family correction needed).
- **H1(ii)** Pooled magnitude θ_RE ≥ 1.0pp AND one-sided **superiority test** rejects H0: θ ≤ 1.0pp at α=0.05 (i.e., effect is significantly ABOVE the +1.0pp substantive-effect threshold; commit-locked). Note 2026-05-13: replaces prior "TOST equivalence rejected at δ" wording which was ambiguous in direction; one-sided superiority is the unambiguous statistical test for "effect substantively > δ".

**Drop-one definition (operational)**: For each (site, model) cell containing all 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM), compute oracle ceiling SR over {6 modes} minus oracle ceiling SR over {5 modes drop P-SoM} per task, then average across the cell's task pool. Paired 1000-resample task-level bootstrap CI per cell; pooled DerSimonian-Laird across 4 cells.

**Transparency consistency check (NOT gating, reported alongside H1)**: K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually clear Holm α=0.05 within the per-cell P-SoM sub-family (m = 4). **K-of-N reclassified pre-data 2026-05-13** from gating threshold to transparency consistency check, based on power analysis (`docs/analysis/cross_sites/power_analysis.md`) showing per-cell power at observed 1-3pp effect sizes is < 10% — calibrated only for ≥7pp effects, smaller than reasonable phenomenon effect size, so K-as-gate is statistically dysfunctional. See §4 audit B9 row + Appendix A 2026-05-13 entry.

#### H2 — Drop-in property (P-SoM specifically)

**Scope revision 2026-05-13 (codex stress audit T2 fix)**: H2 primary gate (gating R1
framing rule) is **cost equivalence only** — H2(a). H2(b) latency and H2(c) signal
AUROC are demoted to **EXPLORATORY transparency reports** (NOT gating R1). H2(d)
drop-one magnitude is already covered by H1(ii) magnitude check. Rationale:

- (a) is a model-property test (regex filter ≡ no image tokens), measurable per-cell with
  bootstrap CI, properly gating
- (b) latency depends on serving infrastructure (DGX vs A100 vs proxy), not a model
  property; should be characterized but not gated
- (c) signal AUROC depends on routing-signal universe choice (`aggregate_routing_auroc.py`
  top-1 selection), which is exploratory characterization per §5 — including AUROC
  in R1 gating would force a circular dependency
- (d) folded into H1(ii) magnitude+superiority gate

**H2(a) — PRIMARY GATE (cost equivalence)**:
median cost(P-SoM) within ±10% of median cost(DOM) per cell, replicated in
≥ K_h2 = 3 of 4 cells (transparency consistency check). Tested empirically per cell.
This reflects the by-construction property that `[SOM_MARKS]` is an AXTree regex
filter (no image embedding tokens).

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

- **H3(i) PRIMARY GATE** axis 1: pooled across N=4 cells, mean |P-text ∖ P-SoM| > 0 with DerSimonian-Laird random-effects meta CI excluding 0 (Holm α=0.05, m=1 within axis-1 sub-family).
- **H3(ii) PRIMARY GATE** axis 2: same as H3(i) for |P-prompt ∖ P-SoM|.
- **H3(iii)** Per-cell unique-count noise floor: ≥ 2 tasks (≈ 1pp at N=234 to N=210); 1 task is noise floor, excluded from cell-level pass.

**Transparency consistency check (NOT gating)**: K_h3 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap 95% CI excluding 0 (m=4 per axis). Same K-of-N reclassification rationale as H1 (see §4 audit B9 + Appendix A 2026-05-13 entry).

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

The 4 distinguishing predictions in 实验笔记 §108.16 are tested against 16-cell data. The framework was developed after observing N=4 pre-Phase-A cells; this is **post-hoc**.

Paper §5 prose **must** explicitly flag: "post-hoc theoretical framework, validated on the same data motivating it; no formal significance gating."

#### H6 — Capability-modulated reversal (B0 vs B1 axis preference)

B0 vs B1 ranking direction on text-axis vs image-axis drop-one tested via B0 × B1 × axis logistic GLM interaction term. Post-hoc finding (developed after observing N=4 pre-Phase-A cells).

Paper §7 prose **must** explicitly flag: "post-hoc finding; no pre-registered prediction."

### ROUTER family (gates Section 6 routing claim — **pending advisor 5/5 lock**: paper-1 PRIMARY vs paper-2 deferred)

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
| **R1** | H1 holds AND H2(a) cost holds AND H3(i) holds AND H3(ii) holds (H2(b) latency + H2(c) AUROC reported as exploratory transparency, NOT gating) | "Phantom routing space (M1/M2 2-axis empirical structure); P-SoM as deployment hero, P-text/P-prompt as structural ablation arms validating axis decomposition." | STRONGEST |
| **R2** | H1 holds AND H2(a) cost holds AND only one of H3(i)/(ii) holds | "Phantom routing space (single-axis empirical structure) with P-SoM as deployment hero; remaining axis decomposition theoretical (Zoom 1 architectural argument only)." | MODERATE-STRONG |
| **R3** | H1 holds AND H2(a) cost holds AND neither H3(i)/(ii) holds | "Phantom-SoM is hidden 4th routing arm; M1/M2 axis decomposition supported by Zoom 1 architectural argument only, not empirically validated by ablation." | MODERATE (= 04-30 fallback; workshop-grade) |
| **R4** | H1 holds AND H2(a) cost fails (latency / AUROC reported but not gating, so R4 triggers only on cost failure) | "Phantom-SoM partial drop-in (cost equivalence fails)" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
| **R5** | H1 fails (pooled DerSimonian-Laird meta Holm α=0.05 fails OR pooled magnitude θ_RE < 1.0pp OR one-sided **superiority test** fails reject H0: θ ≤ +1.0pp at α=0.05) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |

**Trigger rule update 2026-05-13**: R5 no longer fires on `< K_h1` (K-of-N reclassified to transparency-only). Pooled meta + one-sided superiority test primary gate only (TOST replaced 2026-05-13 due to semantic ambiguity, see Appendix A). K-of-N consistency reported in §4 per-cell table as descriptive transparency row.

**Heterogeneity-conditional rule (added 2026-05-13 to resolve §4 audit B8 ↔ H1(i) conflict)**: If pre-specified I² > 75% from random-effects meta (per §4 audit B8 thresholds), do NOT pool — primary inference reverts to per-cell forest + meta-regression by site / model. R1-R5 framing in this branch maps to per-cell direction-consistency: ≥3 of 4 cells direction-positive + ≥2 individually Holm sig → R3-grade hook; otherwise R4/R5.

---

## §3 Multiple-Comparison Family Declaration

**PRIMARY family** (gating paper hook) — UPDATED 2026-05-13 (K-of-N → transparency-only):
- H1(i) pooled meta on N=4 statistical cells: m = 1 (no within-family correction).
- H1(ii) pooled magnitude θ_RE ≥ 1.0pp AND one-sided superiority test (H0: θ ≤ +1.0pp vs H1: θ > +1.0pp) rejected at α=0.05: m = 1.
- H2(a) cost equivalence per cell (PRIMARY): m = 4 statistical cells. H2(b) latency + H2(c) AUROC + H2(d) drop-one folded reported as exploratory transparency (NOT gating, not in this PRIMARY family m count). Scope revision 2026-05-13 codex stress audit T2 fix.
- Method: Holm-Bonferroni step-down per H-sub-family (Holm 1979).

**STRUCTURAL family** (gating phantom-space framing) — UPDATED 2026-05-13:
- H3(i) pooled axis-1 meta on N=4 cells: m = 1.
- H3(ii) pooled axis-2 meta on N=4 cells: m = 1.
- Method: Holm-Bonferroni step-down per axis sub-family.
- Rationale: structural claim is weaker than deployment, separate family avoids inflating PRIMARY family m count.

**TRANSPARENCY family** (NOT gating, reported in §4 per-cell table for reviewer transparency):
- K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually Holm-significant on P-SoM drop-one (m=4 per cell).
- K_h3 axis-1 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap CI excluding 0.
- K_h3 axis-2 = same as axis-1.
- Method: Holm-Bonferroni within transparency sub-family (m=4 per K-test).
- **Rationale for transparency-only reclassification**: power analysis (`docs/analysis/cross_sites/power_analysis.md`, pre-data) shows K-of-N family power at observed 1-3pp effect sizes is < 10%, calibrated only for ≥7pp effects. Per-cell N=234 (cls) / 210 (red) bootstrap power at 1.5pp effect ≈ 0.30. P(≥3 of 4 cells sig | p_cell=0.30) ≈ 8%. K-as-gate is statistically dysfunctional in this regime; K-as-transparency provides per-cell consistency check value alongside pooled meta. See Appendix A 2026-05-13 entry.

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
| **H1(ii) superiority threshold δ** (also informational TOST margin) | **1.0pp** | Used as H0 threshold for one-sided superiority test (H0: θ ≤ +1.0pp, primary gate per H1(ii) revision 2026-05-13) AND as informational TOST equivalence margin (informational secondary report only). ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
| **H1 K_h1 transparency ratio** | **0.75** (= 3/4 cells; **transparency-only, not gating** per 2026-05-13 reclassification) | Reports per-cell consistency alongside pooled meta; not a gate on H1 |
| **H3 K_h3 transparency ratio** | **0.67** (= 3/4 cells; **transparency-only**) | Same as K_h1 reclassification rationale |
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
| **N_cells statistical (H1/H3 stratification)** | **4 cells** = (site, model) tuples: (cls, B0), (cls, B1), (red, B0), (red, B1). Drop-one is computed per cell using all 6 modes; pooled DerSimonian-Laird random-effects meta across 4 cells | Cell = paired-test stratification unit (one per (site, model)), distinct from "condition" (one per (site, model, mode)). 4 cells × 6 modes = 24 conditions. Distinction propagated to all prose / queue / docs 2026-05-13 |
| **N_conditions Phase 1b (main paper, deferred)** | **+12 conditions** = shop × 2 models × 6 modes. Launches after Phase 1a workshop submission to feed main paper R1 / Option D framing decision. N_cells statistical becomes 6 (= 3 sites × 2 models) when Phase 1b lands | Phase 1b is additive; workshop §1 hook does NOT depend on Phase 1b. Main paper §1 hook upgrade R3 → R1 conditional on shop replicating P-SoM 4-fold within ±2pp tolerance |
| **Best-single-mode baseline (H7/H8 anchor)** | Per cell: mode with highest mean adjusted-SR on train fold | Used as comparison anchor for router lift; **train/test split-stratified** to prevent test leak |
| **Missing-data / crashed-episode policy** (audit B6) | (a) Crashed episodes (uncaught exception, OOM, timeout > 30 min, browser crash) **excluded from paired-N denominators**, **NOT imputed** to success or failure. (b) Episodes with `not_logged_in` or `auth_drift` flag at termination excluded after watchdog refresh fails 3 retries (per `experiment_watchdog.py`). (c) Missing artifacts (no `obs.txt` / `screenshot_annotated.png` at step k) excluded from per-step analyses, NOT imputed. (d) Per-cell exclusion count + reason histogram reported in Appendix C. | Listwise deletion only; mean imputation introduces bias for SR proportions, hot-deck imputation breaks paired-N pairing. Crashed-episode imputation as success/failure would inflate Type I/II error. Lock 2026-05-09. |
| **Stopping rules / contamination halt criteria** (audit B7, REVISED 2026-05-13 to remove outcome-dependent bias per codex Flaw 6) | (a) **Pre-launch**: `make pre-launch-check` validates seed configured + HF SHA pinned + git working tree clean + GPU available + disk free > 20GB; failure halts launch (per audit C10). (b) **Smoke-test gate (outcome-INDEPENDENT)**: first 10 episodes per condition must show auth-state `logged_in=True` on all 10 AND ≥ 9 of 10 episodes produced complete artifact bundle (`obs.txt` + `screenshot.png` + `condition_summary_v2` increment + JSONL flush) AND evaluator returned a parseable verdict (success / failure / `ua_match` N/A — any of these is fine, **success rate itself is NOT checked**). Failures halt for auth refresh / artifact pipeline debug, NOT for low SR observation. Rationale: outcome-dependent smoke gate biases low-SR cells upward (a true 5-10% SR cell has 35-60% probability of "0 successes in first 10" by binomial chance and would be invalidly restarted). (c) **Auth/site contamination halt**: ≥ 5 consecutive episodes with `not_logged_in` ⇒ stop cell, refresh auth, archive partial run as `_dirty_partial`, restart fresh. (d) **Eval drift halt**: if rerun on identical archived episode produces SR delta > 5pp via `validate_run.py --strict`, freeze cell + investigate evaluator code. (e) **OOM / hardware halt**: 3 consecutive job failures ⇒ stop cell, document hardware in incident log, manually re-queue with diagnostic output. | Halt rules protect data purity; halted cells restarted only after root-cause documented in `master_bug_catalog.md` + bug fix committed. Lock 2026-05-09; smoke gate revised 2026-05-13 to outcome-independent variant. |
| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
| **K-of-N rule scope** (audit B9 power-corrected, REPROPAGATED 2026-05-13 to H1/H3/R5/§6/Appendix A) | The **K_h1=3/4 / K_h3=3/4** ratios (under 24-condition / 4-cell Phase 1a scope) are **transparency consistency checks** (count of cells *individually* clearing α=0.05 Holm), **NOT gates on H1/H3 paper claims**. **Primary gate** = (a) DerSimonian-Laird random-effects meta-analysis on N=4 (site, model) cells + (b) one-sided superiority test on pooled drop-one effect (H0: θ ≤ +1.0pp vs H1: θ > +1.0pp) at α=0.05. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed 1-3pp effect sizes is < 10%; the rule is calibrated for ≥7pp effects (1.5pp per-cell power ≈ 0.30; P(≥3 of 4 cells sig) ≈ 8%). K-as-gate is statistically dysfunctional in this effect-size regime. **2026-05-13 propagation**: prior prereg text in H1(ii) / H3(i) / H3(ii) / R5 / §6 still gated K-of-N → fixed to "transparency consistency check, reported alongside but NOT gating". This is **pre-data reclassification**: power analysis commit predates Phase 1a launch; reclassification timestamp recorded for OSF witness audit trail. | Original audit B9 lock 2026-05-09 introduced framing but did not propagate to H1/H3/R5/§6 prose (codex stress audit 2026-05-13 Flaw 2 surfaced internal contradiction). Repropagation 2026-05-13 reconciles all references. |

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

### §5.X Post-hoc Layer Selection Disclosure (Stage 2 Mechanism, audit G5)

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

1. Advisor sync session: lock **9 commit decisions** (expanded 5/4 audit + 2026-05-13 revisions):
   - (1) **K_h1=0.75 transparency ratio** (= 3/4 cells; reclassified gate → transparency-only 2026-05-13)
   - (2) **K_h3=0.67 transparency ratio** (= 3/4 cells; reclassified gate → transparency-only 2026-05-13)
   - (3) **δ=1.0pp** — one-sided superiority threshold for H1(ii) primary gate AND informational TOST margin secondary report. SR drop-one effect-size margin, distinct from H2(a) cost ±10% margin — see §4 lock row
   - (4) **Cell inclusion**: Phase 1a = cls + red × B0+B1 × 6 modes (Phase A post-fix only); Phase 1b shop deferred
   - (5) **Witness mechanism**: Git + advisor email + OSF DOI
   - (6) **N_conditions Phase 1a final scope**: **24 operational conditions** (= 2 sites × 2 models × 6 modes) across **4 statistical cells** (= (site, model) tuples) — student-decided 2026-05-13 post-codex stress audit, replaces prior 16-cell phantom-only scope. Advisor email witness pending
   - (7) **Smoke-gate revision** (2026-05-13): outcome-independent (auth + artifact + evaluator parseability only), no SR-based restart
   - (8) **Router paper-1-vs-paper-2 decision**: H7-H8 PRIMARY (paper-1) or SECONDARY-informational (paper-2 deferred)
   - (9) **Train/test split protocol**: 5-fold site-stratified CV vs leave-one-site-out (LOSO)
   - Plus lock H-list (H1-H8 family declaration final).
2. Update this file frontmatter: `status: draft` → `status: locked`, fill `registered_at`, `registered_git_sha`, `witnessed_by`.
3. Git commit this file.
4. Advisor sends single-line confirmation email: "I witness pre-registration of phantom-SoM hypotheses (H1-H8) and 8 lock decisions as of <git SHA> <date>." Email archived in `.witness/preregistration_witness.eml` (gitignored, local-only).

### (b) External witness — OSF DOI (optional, paper-time)

Approximately 1 week before paper submission:

1. Create free OSF account (if not exists) at osf.io.
2. New project: "Phantom-SoM 16-cell pre-registration witness."
3. Upload this `preregistration.md` (locked version) + companion EVIDENCE_LAYER_AUDIT.md §2 + ADVISOR_SYNC.md §1.4 (lock decisions).
4. OSF generates DOI + permanent timestamp.
5. Paper §1 footnote cites the DOI: "Hypotheses pre-registered prior to 16-cell rerun (OSF DOI X.YYYY/osf.io/zzzz, Git SHA abc123, witnessed by [advisor name] on YYYY-MM-DD)."

---

## §7 Reproducibility Scope Statement (audit A14, F3)

**Public release scope** — what reviewers / replicators can reproduce from the released artifact:

| Component | Reproducibility tier | Mechanism |
|---|---|---|
| **B1 (Qwen3-VL-4B local)** | **Fully reproducible** byte-identical | HF model SHA pinned (`ebb281ec70b05090aa6165b016eac8ec08e71b17`) + greedy decoding + seed=42 (`configs/exp_v2_base.yaml`) + `_seed_global_rng()` per (cond, seed) iteration + env_snapshot.json per run + git commit SHA in run_manifest. Re-running produces byte-identical action traces, hidden states, and aggregate SR. |
| **B1 mechanistic Stage 2** | **Fully reproducible** | Same as B1 plus `--random-seed 42` for `--random-inject` (cell E). `archive_subset_b1_{cls,reddit}/` (curated mirage tasks + cached observations + screenshot_annotated) committed for cross-machine replication without needing full archive. |
| **B0 (Qwen3-Omni-235B-Thinking via proxy API)** | **Verifiable from traces, replayable subject to API access** | All B0 episodes log full request/response traces + temperature=0 server-side. Re-running depends on: (a) proxy API endpoint availability, (b) model server-side determinism (best-effort, not guaranteed at temperature=0). For paper claims, B0 is "one controlled stochastic sample with bootstrap task uncertainty" — replicators verify via released traces or rerun under same proxy / Anthropic-native API access. |
| **VWA environment** | **Reproducible given containers** | VWA Docker images pulled at submodule commit SHA (recorded at lock time per audit A5/F8 remediation). Reset-before-each-cell protocol (`RESET_BEFORE=1`) ensures clean start state. Site-state snapshot pre/post-cell as additional gate (audit C3 pending). |
| **Evaluator** | **Fully reproducible** byte-identical | `evaluator_code.combined_sha256` recorded per run. T0/T1/T2/T3 evaluator-change protocol (`evaluator_change_protocol.md`) governs post-lock changes — same paper requires dual-reporting for any T0 fix. |
| **Mechanism analysis (Stage 2 patching)** | **Fully reproducible** | Greedy decoding + seed=42 + Holm-corrected paired t-test + 1000-resample percentile bootstrap (seed=42 in `stage2_layer_significance.py`). Per-task per-layer `patching_continuation_results.json` released for re-aggregation. |

**Scope claim language for paper §3**:

> "All B1 (local Qwen3-VL-4B) experiments, including agent traces, mechanistic activation patching, and aggregate analysis, are fully reproducible given the released code (commit SHA), pinned HF model revision, and seed configurations. B0 (proxy-API Qwen3-Omni-235B) results are verifiable from released traces and replayable subject to API access; B0 server-side decoding determinism is best-effort under temperature=0 and reported as a single controlled stochastic sample with task-level bootstrap uncertainty. The VWA environment is reproducible given the pinned VWA submodule commit and Docker images. Cross-benchmark (WebArena) results are out of scope for this paper unless explicitly reported in the appendix."

**External validity scope (audit F3)**:

> "Empirical claims are scoped to the **Qwen-family VWA characterization**: Qwen3-VL-4B (B1) and Qwen3-Omni-235B-Thinking (B0) on VisualWebArena classifieds / reddit / shopping. Cross-benchmark generalization (WebArena 480 tasks) and cross-model-family generalization (Llama-VL, GPT-4o-V, Gemini-Pro-VL) are explicitly future work. Mechanistic Stage 2 findings are scoped to the curated mirage-disagreement task tiers (composite score-based curation per `curate_mirage_tasks.py`) on classifieds (and reddit if cells F/G replicate); broader phantom-routing-space mechanism universality is conditional on the 2x2 + cross-site control results."

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
| \<pending advisor email follow-up\> | \<witness K_h1=0.75 transparency / K_h3=0.67 transparency / TOST δ=1.0pp / N_conditions=24 (Phase 1a) / N_cells=4 / split protocol / paper split / Phase 1b shop / outcome-indep smoke gate / per follow-up doc Q1-Q11\> | \<email reply timestamp + Git SHA at lock\> |
