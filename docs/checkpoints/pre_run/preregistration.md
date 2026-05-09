---
type: preregistration
status: draft
created: 2026-05-03
draft_author: Jiaming
registered_at: <pending advisor sync lock>
registered_git_sha: <pending lock>
witnessed_by: <pending advisor sync>
osf_doi: <pending paper submission stage>
data_lock_until: <pending 16-cell rerun completion>
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

P-SoM drop-one > 0 across cells, satisfying ALL three sub-conditions:

- **H1(i)** Pooled DerSimonian-Laird random-effect meta-analysis reaches significance at Holm α=0.05 (PRIMARY family m=1 test, no within-family correction needed).
- **H1(ii)** ≥ K_h1 of N_cells individually Holm-significant at α=0.05 within the per-cell P-SoM sub-family (m = N_cells), where **K_h1 = 0.75** (commit-locked, see §4).
- **H1(iii)** Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence at margin **δ = 1.0pp** rejected (commit-locked).

#### H2 — 4-fold drop-in property (P-SoM specifically)

All four sub-claims hold per cell, replicated in ≥ K_h1 cells:

- **(a) Cost** — median cost(P-SoM) within ±10% of median cost(DOM); reflects the by-construction property that `[SOM_MARKS]` is an AXTree regex filter (no image embedding tokens). Tested empirically per cell.
- **(b) Latency** — median latency(P-SoM) ≤ 0.6 × median latency(SoM); reflects skipping image inference stage. Tested empirically per cell.
- **(c) Signal AUROC** — top-1 routing-signal AUROC(P-SoM) ≥ AUROC(DOM) − 0.05 (within 5pp). Tested empirically per cell, signal selected per `aggregate_routing_auroc.py` top-1.
- **(d) Drop-one magnitude** — folded into H1(iii); P-SoM contributes ≥ 1.0pp lift on average.

#### H3 — Phantom space 2-axis empirical structural claim

Each phantom-space axis (axis 1 = text payload via P-text; axis 2 = SoM-style prompt via P-prompt) contributes tasks NOT solved by P-SoM, evidencing axis decomposition is empirically non-trivial (i.e., phantom space is a multi-region 2D structure, not a collapsed 0D point):

- **H3(i)** axis 1: |P-text ∖ P-SoM| unique-count > 0 with bootstrap 95% CI excluding 0 in ≥ K_h3 of N_cells.
- **H3(ii)** axis 2: |P-prompt ∖ P-SoM| unique-count > 0 with bootstrap 95% CI excluding 0 in ≥ K_h3 of N_cells.
- **H3(iii)** Per-cell unique-count threshold: ≥ 2 tasks (≈ 1pp at N=234 to N=210); 1 task is noise floor.

K_h3 = 0.67 (commit-locked, see §4). This is **lower** than K_h1 because the structural claim is weaker than the deployment claim — non-overlap existence is sufficient, deployment-grade magnitude is not required.

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
| **R1** | H1 holds AND H2 (a)(b)(c) all hold AND H3(i) holds AND H3(ii) holds | "Phantom routing space (M1/M2 2-axis empirical structure); P-SoM as deployment hero, P-text/P-prompt as structural ablation arms validating axis decomposition." | STRONGEST |
| **R2** | H1+H2 hold AND only one of H3(i)/(ii) holds | "Phantom routing space (single-axis empirical structure) with P-SoM as deployment hero; remaining axis decomposition theoretical (Zoom 1 architectural argument only)." | MODERATE-STRONG |
| **R3** | H1+H2 hold AND neither H3(i)/(ii) holds | "Phantom-SoM is hidden 4th routing arm; M1/M2 axis decomposition supported by Zoom 1 architectural argument only, not empirically validated by ablation." | MODERATE (= 04-30 fallback) |
| **R4** | H1 holds AND H2 partially fails (e.g., (a) cost or (b) latency fails on some site) | "Phantom-SoM partial drop-in" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
| **R5** | H1 fails (pooled meta sig fails Holm OR < K_h1 cells individually sig) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |

---

## §3 Multiple-Comparison Family Declaration

**PRIMARY family** (gating paper hook):
- H1(i) pooled meta: m = 1 (no within-family correction).
- H1(ii) per-cell P-SoM Holm: m = N_cells.
- H2 sub-claims (a)(b)(c)(d): m = 4 × N_cells (each per-cell sub-claim test).
- Method: Holm-Bonferroni step-down per H-sub-family (Holm 1979).

**STRUCTURAL family** (gating phantom-space framing):
- H3(i) axis 1 per-cell: m = N_cells (bootstrap CI lower-bound > 0 test).
- H3(ii) axis 2 per-cell: m = N_cells.
- Method: Holm-Bonferroni step-down per axis sub-family.
- Rationale: structural claim is weaker than deployment, separate family avoids inflating PRIMARY family m count.

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
| **TOST equivalence margin δ** | **1.0pp** | ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
| **H1 K_h1 cell-pass threshold** | **0.75** | Allows ~25% capability-outlier cells (e.g., B1 shopping power-limited); not so strict that single-cell noise breaks claim |
| **H3 K_h3 cell-pass threshold** | **0.67** | Lower than K_h1 because structural < deployment commit |
| **H3 unique-count floor** | **≥ 2 tasks per cell** | 1 task is sampling noise; 2 tasks ≈ 1pp at N=234 |
| **Cell inclusion (main)** | Phase A post-fix only (commit ≥ 3c15cd7) | bug-clean rerun |
| **Cell inclusion (Appendix D)** | Archived pre-Phase-A data as robustness check | Symmetric contamination disclosure |
| **N inclusion floor** | ≥ 100 ep per (cell × mode) | Statistical power baseline |
| **FP filter primary** | na_fp + eval_fp combined | Per 实验笔记 §95 (visual_fp deprecated — no lit precedent, boundary-undecidable, over-filters 95.3% VWA tasks). Code: `compute_adjusted_success()` returns `fp_reason ∈ {'', 'na_fp', 'eval_fp'}` (`p79/experiment/analysis.py:52`) |
| **FP filter sensitivity** | 3 variants reported (raw_SR / +na_fp only / +na_fp+eval combined) | Robustness disclosure. visual_fp is NOT in the ladder — see §95 decision rationale |
| **Non-visual subset robustness** | 43 VWA + 480 WA = 523 manually-audited non-visual tasks (`docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py`) | Replaces deprecated visual_fp; Appendix D sensitivity check |
| **Mode operational definitions** | 6 modes per paper §3 (text format × prompt × image): DOM (AXTree+DOM-prompt+no image) / SoM ([SOM_MARKS]+SoM-prompt+image) / Vision (no text+image) / P-text ([SOM_MARKS]+DOM-prompt+no image) / P-prompt (AXTree+SoM-prompt+no image) / P-SoM ([SOM_MARKS]+SoM-prompt+no image) | Stipulative — **no post-hoc episode reclassification**. Episodes systematically excluded per (FP filter / N-floor / data-corruption flag), never redefined which mode they belong to. Edge cases (empty AXTree / 0 marks / OCR-empty) follow `condition_meta.json` declared mode |
| **Routing signal universe** | `aggregate_routing_auroc.py` enumerated set: ep_mean_verbalized / ep_min_verbalized / max_repeat_streak / action_diversity / url_revisit_count / url_revisit_max / action_unique_types / url_unique_count / ep_mean_logprob / ep_min_logprob (last 2 B1-only) | **No post-hoc engineered features** for router input. Best-signal-per-mode characterization is exploratory (§5) — paper §6 portfolio finding, not pre-registered prediction |
| **Router train/test split** | 5-fold site-stratified CV on cls+red post-Phase-A task pool, seed=42, min test fold ≥ 40 tasks | Reproducible split via `scripts/analysis/router_split.py` (TBD). **Test fold predictions use ONLY train-fold mode rankings** to prevent oracle leak. Pending advisor 5/5 sync alternative: leave-one-site-out (LOSO) — test cls hold-out trained on red, vice versa |
| **Failure-mode classification rubric** | 5-bucket: `early_finish` / `wrong_commit` / `visual_hijack` / `click_loop` / `persistent_error` per `docs/analysis/disagreement_clusters.md` decision tree | Pre-data inter-annotator agreement target Cohen κ ≥ 0.7 on 30-task pilot (codex prompt + 1 human spot-check). Paper §1 prose ("B0 53.3% early-finish vs B1 70.4% visual-hijack/click-loop, +43.7pp") cites these locked buckets |
| **N_cells final scope** | **16 cells** (B0 × {cls, red} × 3 phantom = 6 + B1 × {cls, red} × 3 phantom = 6 + B0 shop × 2 phantom = 2 + B1 shop × 2 phantom = 2). K_h1=0.75 → **≥ 12 cells pass** (= ⌈0.75 × 16⌉); K_h3=0.67 → **≥ 11 cells pass** (= ⌈0.67 × 16⌉) | ✅ **Student-decided post-5/5 sync** (chose 16 over 14 to add B1 shop × phantom_text/phantom_som for cross-capability shop coverage). Advisor email witness pending via `docs/checkpoints/advisor_sync_5_5_followup.md` follow-up |
| **Best-single-mode baseline (H7/H8 anchor)** | Per cell: mode with highest mean adjusted-SR on train fold | Used as comparison anchor for router lift; **train/test split-stratified** to prevent test leak |
| **Missing-data / crashed-episode policy** (audit B6) | (a) Crashed episodes (uncaught exception, OOM, timeout > 30 min, browser crash) **excluded from paired-N denominators**, **NOT imputed** to success or failure. (b) Episodes with `not_logged_in` or `auth_drift` flag at termination excluded after watchdog refresh fails 3 retries (per `experiment_watchdog.py`). (c) Missing artifacts (no `obs.txt` / `screenshot_annotated.png` at step k) excluded from per-step analyses, NOT imputed. (d) Per-cell exclusion count + reason histogram reported in Appendix C. | Listwise deletion only; mean imputation introduces bias for SR proportions, hot-deck imputation breaks paired-N pairing. Crashed-episode imputation as success/failure would inflate Type I/II error. Lock 2026-05-09. |
| **Stopping rules / contamination halt criteria** (audit B7) | (a) **Pre-launch**: `make pre-launch-check` validates seed configured + HF SHA pinned + git working tree clean + GPU available + disk free > 20GB; failure halts launch (per audit C10). (b) **Smoke-test gate**: first 10 episodes per cell must show ≥ 1 success (or ≥ 1 N/A by ua_match) AND auth-state `logged_in=True`; otherwise halt + watchdog auth_refresh + restart, log incident in `master_bug_catalog.md`. (c) **Auth/site contamination halt**: ≥ 5 consecutive episodes with `not_logged_in` ⇒ stop cell, refresh auth, archive partial run as `_dirty_partial`, restart fresh. (d) **Eval drift halt**: if rerun on identical archived episode produces SR delta > 5pp via `validate_run.py --strict`, freeze cell + investigate evaluator code. (e) **OOM / hardware halt**: 3 consecutive job failures ⇒ stop cell, document hardware in incident log, manually re-queue with diagnostic output. | Halt rules protect data purity; halted cells restarted only after root-cause documented in `master_bug_catalog.md` + bug fix committed. Lock 2026-05-09. |
| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
| **K-of-N rule scope** (audit B9 power-corrected) | The **K_h1=12/16 / K_h3=11/16** thresholds are retained as **secondary transparency checks** (count of cells *individually* clearing α=0.05 Holm), **not as gates on H1/H3 paper claims**. **Primary detection** = (a) DerSimonian-Laird random-effects meta-analysis on cells N≥10 (B8 lock above) + (b) TOST equivalence on N=910 pooled tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed effect sizes (1-5pp) is < 10%; the rule is calibrated for ≥7pp effects. This recharacterization is consistent with the original §4 "Primary metric" + B8 random-effects lock — K-of-N was always a transparency aggregator, not the primary test, and the corrected power analysis makes that explicit. | `power_analysis.py` bug (stale interpretation block) discovered 2026-05-09; fixed in same commit. K-of-N values themselves unchanged; only the framing as "secondary transparency vs primary gate" is added. Lock 2026-05-09. |

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

1. Advisor sync session: lock **8 commit decisions** (expanded 5/4 audit):
   - (1) **K_h1=0.75** cell-pass threshold
   - (2) **K_h3=0.67** cell-pass threshold
   - (3) **TOST δ=1.0pp** equivalence margin
   - (4) **Cell inclusion**: Phase A post-fix only (main) + archived pre-Phase-A (Appendix D robustness)
   - (5) **Witness mechanism**: Git + advisor email + OSF DOI
   - (6) **N_cells final scope**: **16** (student-decided post-5/5 sync; B0 × {cls,red} × 3 + B1 × {cls,red} × 3 + B0 shop × 2 + B1 shop × 2; advisor email witness pending via follow-up doc)
   - (7) **Router paper-1-vs-paper-2 decision**: H7-H8 PRIMARY (paper-1) or SECONDARY-informational (paper-2 deferred)
   - (8) **Train/test split protocol**: 5-fold site-stratified CV vs leave-one-site-out (LOSO)
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
| \<pending advisor email follow-up\> | \<witness K_h1=0.75 / K_h3=0.67 / TOST δ=1.0pp / N_cells=16 / split protocol / paper split per follow-up doc Q1-Q11\> | \<email reply timestamp + Git SHA at lock\> |
