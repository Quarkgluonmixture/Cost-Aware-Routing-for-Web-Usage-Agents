Reading prompt from stdin...
OpenAI Codex v0.130.0
--------
workdir: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: high
reasoning summaries: none
session id: 019e2223-8ae6-7dd2-865f-e0959acbf99d
--------
user
# Round C — Statistical methodology assumptions audit

## Context

P79 paper-1 statistical framework (post 2026-05-13 codex stress audit revisions):

- **N = 4 statistical cells** (Phase 1a scope: 2 sites × 2 models)
- **Primary gate (H1)**: pooled DerSimonian-Laird random-effects meta on per-cell
  P-SoM drop-one oracle ceiling lift, Holm α=0.05 sig (m=1) + pooled magnitude
  θ_RE ≥ 1.0pp + one-sided superiority test (H0: θ ≤ +1.0pp) rejected at α=0.05
- **Primary gate (H3)**: pooled DL meta on axis-1 |P-text \ P-SoM| unique count +
  axis-2 |P-prompt \ P-SoM| (two separate sub-families, each m=1)
- **Primary gate (H2)**: per-cell median cost equivalence within ±10%, replicated
  in ≥ 3 of 4 cells (K_h2 transparency)
- **Transparency-only (NOT gating)**: K-of-N consistency checks per H sub-family.
  K_h1 = 3 of 4 cells; K_h3 = 3 of 4 per axis. Reclassified pre-data 2026-05-13
  from gating threshold to transparency consistency per power analysis showing
  K family power < 10% at observed 1-3pp effect sizes
- **Bootstrap**: 1000-resample paired task-level bootstrap, percentile CI,
  single-level (no nested cluster), seed=42
- **Framing rule R1-R5**: maps (H1, H2, H3_axis1, H3_axis2) primary gate decisions
  to hook power (STRONGEST / MODERATE-STRONG / MODERATE / WEAK / paper-death)

**Your job**: Audit this framework as a hostile **statistical reviewer**. NOT a
methodology blessing. Find anything that:
- (a) A top statistical reviewer (e.g., NeurIPS statistics-area chair) would
      challenge as "the assumption doesn't hold here"
- (b) Has a math error (DL formula transcribed wrong, bootstrap variance wrong,
      Holm step-down wrong, K-of-N power calculation wrong)
- (c) Is statistically valid in isolation but problematic in combination (e.g.,
      separate sub-families with Holm correction but undeclared shared cells)
- (d) Has a edge case where R1-R5 framing rule mapping gives the wrong answer
- (e) Is undefined for the observed data (e.g., what if a cell has 0 tasks, or
      one mode is missing, or I² > 75% triggers "do not pool")

## Input files (read cold)

### Primary statistical method docs

- `docs/checkpoints/pre_run/preregistration.md` §2 H1/H2/H3 (statements + drop-one
  formula + superiority test wording) + §3 family declaration + §4 locked analysis
  choices (TOST δ / K-of-N / heterogeneity / bootstrap / missing data / stopping
  rules) + §5.X Stage 2 layer selection disclosure + Appendix A 2026-05-13 entries
- `docs/analysis/cross_sites/power_analysis.md` — K-of-N power calculation,
  underwriting K-of-N transparency-only reclassification

### Statistical implementation code

- `scripts/analysis/preregistration_decision_test.py` — Round 2 rewrite, the
  canonical implementation of the framework:
  - `dersimonian_laird_meta()` — pooled meta math (Higgins & Thompson 2002 /
    DerSimonian & Laird 1986)
  - `superiority_test()` — one-sided z-test for θ > threshold
  - `tost_equivalence()` — TOST (Schuirmann 1987), now informational only
  - `holm_correct()` — Holm-Bonferroni step-down
  - `_paired_bootstrap()` — task-level resampling
  - `apply_framing_rule()` — R1-R5 mapper
- `scripts/analysis/aggregate_phantom_meta.py` (if exists) — random-effects meta
  on existing pre-Phase-A archived data
- `scripts/analysis/aggregate_phantom_lift.py` (if exists) — TOST on pooled tasks
- `scripts/analysis/aggregate_routing_auroc.py` (if exists) — routing-signal AUROC
  family
- `scripts/analysis/sensitivity_loo_meta.py` (if exists) — leave-one-out
  robustness meta

### Companion sanity references

- `docs/checkpoints/paper_planning.md` §2 (theory framework) — does the
  statistical framework match the theoretical scaffold the paper claims?
- `docs/reference/EVIDENCE_LAYER_AUDIT.md` — each figure traces to a gated
  H1/H3 sub-claim OR exploratory; check consistency

## Output format

### One-sentence statistical verdict

Pick one:
- "Statistical framework is methodologically sound — safe to gate paper claims"
- "Statistical framework has N issue(s) that would be flagged by a top stat reviewer"
- "Statistical framework has reviewer-defensible interpretation choices but no math error"
- "Insufficient time to verify — partial audit only"

### Confirmed math errors

For each: location (file:line, method name), the correct formula vs the implemented
formula, magnitude of inference error, severity (HIGH / MED / LOW), defuse effort.

### Confirmed methodological assumptions violated

E.g., DL random-effects meta typically requires N ≥ 10 cells for stable τ²
estimation — N = 4 is small. Is the prereg disclosed this? Is the I² > 75%
"do not pool" branch in R1-R5 statistically coherent? Is paired bootstrap
single-cluster appropriate (vs nested cell × task two-level)?

### Reviewer ammunition (statistical angle)

What questions would a top-tier statistical reviewer ask? E.g.: "Why DerSimonian-Laird
and not REML for τ²?" / "Why one-sided superiority and not two-sided 90% CI?" /
"Why Holm and not FDR for the transparency sub-families?" / "Why N=4 not N≥10?" /
"Did you correct for the multiple K-of-N transparency reports as multiple comparisons?"

For each, list the question + the user's current answer (or "no answer prepared").

### Framing rule R1-R5 edge cases

Are there observed-data outcomes where R1-R5 mapping is ambiguous or gives the
wrong hook framing? E.g., what if H1 passes but I² > 75% triggers "do not pool"
branch — does the heterogeneity-conditional rule resolve cleanly?

### Verdict on next steps

If framework holds: user can confidently lock prereg + push commit (already done).
If framework has flaws: prioritized list — which can be fixed in prereg prose
before advisor meeting tomorrow vs which require advisor sync to repropose vs
which can be defused as appendix sensitivity check after Phase 1a data lands.

## Calibration

- Paper-grade audit, statistical reviewer mode. Not a textbook intro.
- Don't propose code fixes; identify the suspect, impact, defuse cost
- Negative result valid: if framework holds after 60 min, write verdict and stop
- Don't fabricate. File paths + line numbers must be real
- Set your own attack vectors based on what the docs + code show. Do NOT follow
  any enumerated list — read the actual statistical content cold

## Time budget

Up to 60 min. Tier 3 PID monitor fires when codex exits.

codex
I’ll audit the prereg docs and canonical implementation directly, then cross-check the stated assumptions against the code paths that actually decide gates and framing. I’ll focus on concrete reviewer attack points with file/line evidence rather than rewriting the methodology.
  → Read preregistration, power analysis, and companion docs
  • Inspect statistical implementation and optional aggregate scripts
  • Cross-check math, assumptions, multiplicity, missing-data, and framing logic
  • Produce hostile reviewer audit with line references and prioritized next steps
exec
/bin/bash -lc 'rg -n "''^(##|###|#)|H1|H2|H3|DerSimonian|Holm|bootstrap|missing|heterogeneity|I²|I'"\\"'^2|K-of-N|TOST|superiority|Stage 2|2026-05-13|R[1-5]" docs/checkpoints/pre_run/preregistration.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
5:last_revised: 2026-05-13
12:scope_revision_2026_05_13: cls+red × B0+B1 × 6 modes = 24 operational conditions across 4 statistical cells; shop deferred to Phase 1b main paper; K-of-N reclassified gate → transparency-only; smoke-gate stopping rule replaced (outcome-independent)
15:# Phantom-SoM Pre-Registration (Draft)
19:> **Reading order**: §1 epistemic structure (why this framework) → §2 hypotheses (H1-H6 + framing rule) → §3 multiple-comparison family declaration → §4 locked analysis choices → §5 exploratory disclosure → §6 witness mechanism.
28:## §1 Epistemic Structure (why this pre-registration shape)
36:3. **Framing decision is data-conditional, not data-prediction**: paper §1 hook framing depends on which combination of H1-H3 holds. The rule is pre-registered (R1-R5 below) so reviewers can verify the framing-to-data mapping is not post-hoc.
40:5. **Multiple-comparison family discipline**: gating tests (PRIMARY + STRUCTURAL) have explicit Holm-corrected family m count. Exploratory tests (EXPLORATORY family + post-hoc) are reported with adjusted p-values for transparency but NOT used to gate paper claims.
46:## §2 Hypotheses
48:### PRIMARY family (gates paper claim)
50:#### H1 — Hero deployment claim (P-SoM is hidden routing arm)
54:- **H1(i)** Pooled DerSimonian-Laird random-effect meta-analysis on N=4 (site, model) cells reaches significance at Holm α=0.05 (PRIMARY family m=1 test, no within-family correction needed).
55:- **H1(ii)** Pooled magnitude θ_RE ≥ 1.0pp AND one-sided **superiority test** rejects H0: θ ≤ 1.0pp at α=0.05 (i.e., effect is significantly ABOVE the +1.0pp substantive-effect threshold; commit-locked). Note 2026-05-13: replaces prior "TOST equivalence rejected at δ" wording which was ambiguous in direction; one-sided superiority is the unambiguous statistical test for "effect substantively > δ".
57:**Drop-one definition (operational)**: For each (site, model) cell containing all 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM), compute oracle ceiling SR over {6 modes} minus oracle ceiling SR over {5 modes drop P-SoM} per task, then average across the cell's task pool. Paired 1000-resample task-level bootstrap CI per cell; pooled DerSimonian-Laird across 4 cells.
59:**Transparency consistency check (NOT gating, reported alongside H1)**: K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually clear Holm α=0.05 within the per-cell P-SoM sub-family (m = 4). **K-of-N reclassified pre-data 2026-05-13** from gating threshold to transparency consistency check, based on power analysis (`docs/analysis/cross_sites/power_analysis.md`) showing per-cell power at observed 1-3pp effect sizes is < 10% — calibrated only for ≥7pp effects, smaller than reasonable phenomenon effect size, so K-as-gate is statistically dysfunctional. See §4 audit B9 row + Appendix A 2026-05-13 entry.
61:#### H2 — 4-fold drop-in property (P-SoM specifically)
68:- **(d) Drop-one magnitude** — folded into H1(iii); P-SoM contributes ≥ 1.0pp lift on average.
70:#### H3 — Phantom space 2-axis empirical structural claim
74:H3 statistical cells = 4 (one per (site, model)). H3 axis-1 and axis-2 are tested separately within each cell.
76:- **H3(i) PRIMARY GATE** axis 1: pooled across N=4 cells, mean |P-text ∖ P-SoM| > 0 with DerSimonian-Laird random-effects meta CI excluding 0 (Holm α=0.05, m=1 within axis-1 sub-family).
77:- **H3(ii) PRIMARY GATE** axis 2: same as H3(i) for |P-prompt ∖ P-SoM|.
78:- **H3(iii)** Per-cell unique-count noise floor: ≥ 2 tasks (≈ 1pp at N=234 to N=210); 1 task is noise floor, excluded from cell-level pass.
80:**Transparency consistency check (NOT gating)**: K_h3 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap 95% CI excluding 0 (m=4 per axis). Same K-of-N reclassification rationale as H1 (see §4 audit B9 + Appendix A 2026-05-13 entry).
83:- Primary gating: bootstrap CI on unique-count, 1000 resamples.
84:- Secondary report: McNemar exact one-sided directional asymmetry test (informational only — McNemar tests if one axis dominates the other in unique contribution; H3 only requires non-emptiness, not dominance).
85:- Multiple-comparison: Holm-Bonferroni step-down per axis sub-family (axis 1: m = N_cells; axis 2: m = N_cells).
87:### EXPLORATORY family (reported with corrections, NOT gating)
89:#### H4 — P-text / P-prompt drop-one magnitude
91:Reported per cell + meta-pooled (DerSimonian-Laird) for transparency. Holm-Bonferroni and BH FDR q-values reported. No pre-registered ranking commitment.
95:### POST-HOC family (theory tested on data that motivated it)
97:#### H5 — 别扭 (mismatch) framework predictions
103:#### H6 — Capability-modulated reversal (B0 vs B1 axis preference)
109:### ROUTER family (gates Section 6 routing claim — **pending advisor 5/5 lock**: paper-1 PRIMARY vs paper-2 deferred)
111:#### H7 — Tier 1 oracle router lift over best-single-mode baseline (offline supervised)
115:- **H7(i)** Pooled DerSimonian-Laird random-effect meta-analysis on lift reaches Holm α=0.05 (PRIMARY family m=1 if paper-1 / SECONDARY informational if paper-2).
116:- **H7(ii)** ≥ K_h1 of N_cells individually Holm-significant on per-cell lift, bootstrap 95% CI lower-bound > 0.
117:- **H7(iii)** Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence at margin δ=1.0pp rejected (same δ as H1).
123:- Multiple-comparison: Holm-Bonferroni step-down within H7 sub-family m=N_cells.
127:#### H8 — Tier 2 first-step trigger router (online, test-leak-free)
131:- **H8(i)** Tier 2 router lift over Tier 1 oracle baseline ≥ 0 with bootstrap 95% CI excluding −1.0pp (paper claims Tier 2 ≈ Tier 1 within deployment-grade tolerance, given Tier 2 is leak-free and deployment-realistic).
132:- **H8(ii)** Tier 2 router lift over best-single-mode-baseline ≥ 1.0pp, ≥ K_h1 cells Holm-significant.
138:### FRAMING DECISION RULE (pre-registered, data-conditional)
144:| **R1** | H1 holds AND H2 (a)(b)(c) all hold AND H3(i) holds AND H3(ii) holds | "Phantom routing space (M1/M2 2-axis empirical structure); P-SoM as deployment hero, P-text/P-prompt as structural ablation arms validating axis decomposition." | STRONGEST |
145:| **R2** | H1+H2 hold AND only one of H3(i)/(ii) holds | "Phantom routing space (single-axis empirical structure) with P-SoM as deployment hero; remaining axis decomposition theoretical (Zoom 1 architectural argument only)." | MODERATE-STRONG |
146:| **R3** | H1+H2 hold AND neither H3(i)/(ii) holds | "Phantom-SoM is hidden 4th routing arm; M1/M2 axis decomposition supported by Zoom 1 architectural argument only, not empirically validated by ablation." | MODERATE (= 04-30 fallback; workshop-grade) |
147:| **R4** | H1 holds AND H2 partially fails (e.g., (a) cost or (b) latency fails on some site) | "Phantom-SoM partial drop-in" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
148:| **R5** | H1 fails (pooled meta DerSimonian-Laird Holm α=0.05 fails OR pooled magnitude θ_RE < 1.0pp OR TOST equivalence fails reject at δ=1.0pp) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |
150:**Trigger rule update 2026-05-13**: R5 no longer fires on `< K_h1` (K-of-N reclassified to transparency-only). Pooled meta + TOST primary gate only. K-of-N consistency reported in §4 per-cell table as descriptive transparency row.
152:**Heterogeneity-conditional rule (added 2026-05-13 to resolve §4 audit B8 ↔ H1(i) conflict)**: If pre-specified I² > 75% from random-effects meta (per §4 audit B8 thresholds), do NOT pool — primary inference reverts to per-cell forest + meta-regression by site / model. R1-R5 framing in this branch maps to per-cell direction-consistency: ≥3 of 4 cells direction-positive + ≥2 individually Holm sig → R3-grade hook; otherwise R4/R5.
156:## §3 Multiple-Comparison Family Declaration
158:**PRIMARY family** (gating paper hook) — UPDATED 2026-05-13 (K-of-N → transparency-only):
159:- H1(i) pooled meta on N=4 statistical cells: m = 1 (no within-family correction).
160:- H1(ii) pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence reject at δ=1.0pp: m = 1.
161:- H2 sub-claims (a)(b)(c)(d) per cell: m = 4 × 4 statistical cells = 16 tests (each per-cell sub-claim).
162:- Method: Holm-Bonferroni step-down per H-sub-family (Holm 1979).
164:**STRUCTURAL family** (gating phantom-space framing) — UPDATED 2026-05-13:
165:- H3(i) pooled axis-1 meta on N=4 cells: m = 1.
166:- H3(ii) pooled axis-2 meta on N=4 cells: m = 1.
167:- Method: Holm-Bonferroni step-down per axis sub-family.
171:- K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually Holm-significant on P-SoM drop-one (m=4 per cell).
172:- K_h3 axis-1 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap CI excluding 0.
174:- Method: Holm-Bonferroni within transparency sub-family (m=4 per K-test).
175:- **Rationale for transparency-only reclassification**: power analysis (`docs/analysis/cross_sites/power_analysis.md`, pre-data) shows K-of-N family power at observed 1-3pp effect sizes is < 10%, calibrated only for ≥7pp effects. Per-cell N=234 (cls) / 210 (red) bootstrap power at 1.5pp effect ≈ 0.30. P(≥3 of 4 cells sig | p_cell=0.30) ≈ 8%. K-as-gate is statistically dysfunctional in this regime; K-as-transparency provides per-cell consistency check value alongside pooled meta. See Appendix A 2026-05-13 entry.
179:- H7(ii) per-cell Tier 1 lift Holm: m = N_cells.
180:- H7(iii) folded into H7(i) magnitude/TOST.
183:- Method: Holm-Bonferroni step-down per H-sub-family.
188:- Best-signal-per-mode characterization (Register III AA, Section 6 portfolio finding): per (mode, signal) AUROC reported, Holm-corrected within mode for transparency.
189:- Method: Holm-corrected and BH q-value reported for transparency.
199:## §4 Locked Analysis Choices (pre-data)
204:| **CI method** | 1000-resample task-level paired bootstrap, **percentile** intervals (BCa as sensitivity check, not primary) | Existing infra in `aggregate_phantom_lift.py`. Percentile chosen primary because: (a) paired-bootstrap on bounded proportion (SR ∈ [0,1]) → BCa acceleration estimate is unstable at small N per cell; (b) Cohen's h transformation already symmetrizes; (c) percentile is the canonical reporting in WebArena/VWA precedent. BCa shown as appendix sensitivity check. |
206:| **Bootstrap clustering** | **Single-level (task_id)** for primary, no nested cluster (cell × site) bootstrap | Justification: meta-analysis at cell level is separate (`aggregate_phantom_meta.py` random-effects + I²/τ²); within-cell bootstrap only re-samples tasks. Multi-level cluster would double-count uncertainty already captured by random-effects meta. Lock: percentile + task-id unit + no nested cluster (B2 lock 2026-05-09). |
207:| **Sig threshold** | Holm α=0.05 within respective family | FWER control |
208:| **Effect size (binary)** | Cohen's h with bootstrap CI | Standard for proportion comparisons |
209:| **Effect size (continuous)** | Cohen's d with bootstrap CI | For cost/latency H2(a)(b) |
210:| **TOST equivalence margin δ** | **1.0pp** | ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
211:| **H1 K_h1 transparency ratio** | **0.75** (= 3/4 cells; **transparency-only, not gating** per 2026-05-13 reclassification) | Reports per-cell consistency alongside pooled meta; not a gate on H1 |
212:| **H3 K_h3 transparency ratio** | **0.67** (= 3/4 cells; **transparency-only**) | Same as K_h1 reclassification rationale |
213:| **H3 unique-count floor** | **≥ 2 tasks per cell** | 1 task is sampling noise; 2 tasks ≈ 1pp at N=234 |
215:| **Cell inclusion (Phase 1b main paper)** | Phase A post-fix rerun of shop × B0+B1 × 6 modes (12 conditions added on top of Phase 1a 24 conditions) | Cross-site expansion lever for main paper, post-data R1 vs Option D framing decision |
225:| **N_conditions Phase 1a (operational)** | **24 conditions** = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM). Each condition launched fresh post-fix via `scripts/queues/queue_phase1_paper_grade.sh` (renamed 2026-05-13 from `queue_16cell_paper_grade.sh`; current scope = 24 conditions Phase 1a + 12 conditions Phase 1b deferred). Sequence: B0 → B1 per site (shared user account); cls + red parallel chains | ✅ **Student-decided 2026-05-13** post-codex stress audit. Workshop-targeted (cls + red only, shop deferred to Phase 1b for main paper). Replaces prior 16-cell phantom-only scope that lacked baseline DOM/SoM/Vision rerun (codex Flaw 1) |
226:| **N_cells statistical (H1/H3 stratification)** | **4 cells** = (site, model) tuples: (cls, B0), (cls, B1), (red, B0), (red, B1). Drop-one is computed per cell using all 6 modes; pooled DerSimonian-Laird random-effects meta across 4 cells | Cell = paired-test stratification unit (one per (site, model)), distinct from "condition" (one per (site, model, mode)). 4 cells × 6 modes = 24 conditions. Distinction propagated to all prose / queue / docs 2026-05-13 |
227:| **N_conditions Phase 1b (main paper, deferred)** | **+12 conditions** = shop × 2 models × 6 modes. Launches after Phase 1a workshop submission to feed main paper R1 / Option D framing decision. N_cells statistical becomes 6 (= 3 sites × 2 models) when Phase 1b lands | Phase 1b is additive; workshop §1 hook does NOT depend on Phase 1b. Main paper §1 hook upgrade R3 → R1 conditional on shop replicating P-SoM 4-fold within ±2pp tolerance |
230:| **Stopping rules / contamination halt criteria** (audit B7, REVISED 2026-05-13 to remove outcome-dependent bias per codex Flaw 6) | (a) **Pre-launch**: `make pre-launch-check` validates seed configured + HF SHA pinned + git working tree clean + GPU available + disk free > 20GB; failure halts launch (per audit C10). (b) **Smoke-test gate (outcome-INDEPENDENT)**: first 10 episodes per condition must show auth-state `logged_in=True` on all 10 AND ≥ 9 of 10 episodes produced complete artifact bundle (`obs.txt` + `screenshot.png` + `condition_summary_v2` increment + JSONL flush) AND evaluator returned a parseable verdict (success / failure / `ua_match` N/A — any of these is fine, **success rate itself is NOT checked**). Failures halt for auth refresh / artifact pipeline debug, NOT for low SR observation. Rationale: outcome-dependent smoke gate biases low-SR cells upward (a true 5-10% SR cell has 35-60% probability of "0 successes in first 10" by binomial chance and would be invalidly restarted). (c) **Auth/site contamination halt**: ≥ 5 consecutive episodes with `not_logged_in` ⇒ stop cell, refresh auth, archive partial run as `_dirty_partial`, restart fresh. (d) **Eval drift halt**: if rerun on identical archived episode produces SR delta > 5pp via `validate_run.py --strict`, freeze cell + investigate evaluator code. (e) **OOM / hardware halt**: 3 consecutive job failures ⇒ stop cell, document hardware in incident log, manually re-queue with diagnostic output. | Halt rules protect data purity; halted cells restarted only after root-cause documented in `master_bug_catalog.md` + bug fix committed. Lock 2026-05-09; smoke gate revised 2026-05-13 to outcome-independent variant. |
231:| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
232:| **K-of-N rule scope** (audit B9 power-corrected, REPROPAGATED 2026-05-13 to H1/H3/R5/§6/Appendix A) | The **K_h1=3/4 / K_h3=3/4** ratios (under 24-condition / 4-cell Phase 1a scope) are **transparency consistency checks** (count of cells *individually* clearing α=0.05 Holm), **NOT gates on H1/H3 paper claims**. **Primary gate** = (a) DerSimonian-Laird random-effects meta-analysis on N=4 (site, model) cells + (b) TOST equivalence on pooled cls + red tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed 1-3pp effect sizes is < 10%; the rule is calibrated for ≥7pp effects (1.5pp per-cell power ≈ 0.30; P(≥3 of 4 cells sig) ≈ 8%). K-as-gate is statistically dysfunctional in this effect-size regime. **2026-05-13 propagation**: prior prereg text in H1(ii) / H3(i) / H3(ii) / R5 / §6 still gated K-of-N → fixed to "transparency consistency check, reported alongside but NOT gating". This is **pre-data reclassification**: power analysis commit predates Phase 1a launch; reclassification timestamp recorded for OSF witness audit trail. | Original audit B9 lock 2026-05-09 introduced framing but did not propagate to H1/H3/R5/§6 prose (codex stress audit 2026-05-13 Flaw 2 surfaced internal contradiction). Repropagation 2026-05-13 reconciles all references. |
236:## §5 Exploratory (NOT pre-registered, paper must explicitly flag)
245:- **Best-signal-per-mode characterization** (Register III AA novelty, Section 6 portfolio finding): which routing signal works best for which mode is reported as exploratory characterization, NOT pre-registered prediction. Per-(mode, signal) AUROC table reported with Holm correction within mode for transparency.
249:- Any post-hoc cell subsetting beyond H1-H8 family scope
252:### §5.X Post-hoc Layer Selection Disclosure (Stage 2 Mechanism, audit G5)
254:Stage 2 mechanistic activation patching identified mid-layer disruption peaking
255:at **L17** (3 of 4 cells Holm-significant on `token_overlap_to_target`, p_Holm <
256:0.05; cell D L11+L17 strongest p_Holm = 0.006/0.008 \*\*). The L11/L17 layer
261:| **Stage 2A logit_shift pilot** (5-task aggregate, 笔记 §111.5) | L17 emerged as peak in independent `logit_shift` metric | **Hypothesis-generating** — first-pass discovery |
263:| **Stage 2B 24-task aggregate (cell A)** | L17 Holm-significant (p_Holm = 0.011 \*\*) — confirmed Stage 2A peak | **Confirmatory** — independent metric agreement |
264:| **Stage 2C reverse 15-task (cell B)** | L11 + L17 Holm-significant — direction-paired confirmation | **Confirmatory** |
265:| **Cell D (rev × strong-tier 24)** | L11 + L17 strongest (p_Holm = 0.006/0.008 \*\*) | **Confirmatory** — cross-tier replication |
267:**Disclosure**: Layers L11 and L17 were not pre-registered before Stage 2 data
268:collection; they emerged from Stage 2A pilot (the *hypothesis-generating* phase)
269:and were confirmed by Stage 2B/2C scaled-up data (the *confirmatory* phase). To
270:mitigate the multiple-comparison concern, all per-direction tests use Holm-
275:that the **same** mid-layer region (L11-L17) emerges across (a) Stage 2A
276:logit_shift, (b) Stage 2B forward overlap-to-target, (c) Stage 2C reverse,
289:## §6 Witness Mechanism
291:### (a) Internal witness — Git commit + advisor email
293:1. Advisor sync session: lock **9 commit decisions** (expanded 5/4 audit + 2026-05-13 revisions):
294:   - (1) **K_h1=0.75 transparency ratio** (= 3/4 cells; reclassified gate → transparency-only 2026-05-13)
295:   - (2) **K_h3=0.67 transparency ratio** (= 3/4 cells; reclassified gate → transparency-only 2026-05-13)
296:   - (3) **TOST δ=1.0pp** equivalence margin (interpretation: SR drop-one effect-size margin, distinct from H2(a) cost ±10% margin — see §4 lock row)
299:   - (6) **N_conditions Phase 1a final scope**: **24 operational conditions** (= 2 sites × 2 models × 6 modes) across **4 statistical cells** (= (site, model) tuples) — student-decided 2026-05-13 post-codex stress audit, replaces prior 16-cell phantom-only scope. Advisor email witness pending
300:   - (7) **Smoke-gate revision** (2026-05-13): outcome-independent (auth + artifact + evaluator parseability only), no SR-based restart
303:   - Plus lock H-list (H1-H8 family declaration final).
306:4. Advisor sends single-line confirmation email: "I witness pre-registration of phantom-SoM hypotheses (H1-H8) and 8 lock decisions as of <git SHA> <date>." Email archived in `.witness/preregistration_witness.eml` (gitignored, local-only).
308:### (b) External witness — OSF DOI (optional, paper-time)
320:## §7 Reproducibility Scope Statement (audit A14, F3)
327:| **B1 mechanistic Stage 2** | **Fully reproducible** | Same as B1 plus `--random-seed 42` for `--random-inject` (cell E). `archive_subset_b1_{cls,reddit}/` (curated mirage tasks + cached observations + screenshot_annotated) committed for cross-machine replication without needing full archive. |
328:| **B0 (Qwen3-Omni-235B-Thinking via proxy API)** | **Verifiable from traces, replayable subject to API access** | All B0 episodes log full request/response traces + temperature=0 server-side. Re-running depends on: (a) proxy API endpoint availability, (b) model server-side determinism (best-effort, not guaranteed at temperature=0). For paper claims, B0 is "one controlled stochastic sample with bootstrap task uncertainty" — replicators verify via released traces or rerun under same proxy / Anthropic-native API access. |
331:| **Mechanism analysis (Stage 2 patching)** | **Fully reproducible** | Greedy decoding + seed=42 + Holm-corrected paired t-test + 1000-resample percentile bootstrap (seed=42 in `stage2_layer_significance.py`). Per-task per-layer `patching_continuation_results.json` released for re-aggregation. |
335:> "All B1 (local Qwen3-VL-4B) experiments, including agent traces, mechanistic activation patching, and aggregate analysis, are fully reproducible given the released code (commit SHA), pinned HF model revision, and seed configurations. B0 (proxy-API Qwen3-Omni-235B) results are verifiable from released traces and replayable subject to API access; B0 server-side decoding determinism is best-effort under temperature=0 and reported as a single controlled stochastic sample with task-level bootstrap uncertainty. The VWA environment is reproducible given the pinned VWA submodule commit and Docker images. Cross-benchmark (WebArena) results are out of scope for this paper unless explicitly reported in the appendix."
339:> "Empirical claims are scoped to the **Qwen-family VWA characterization**: Qwen3-VL-4B (B1) and Qwen3-Omni-235B-Thinking (B0) on VisualWebArena classifieds / reddit / shopping. Cross-benchmark generalization (WebArena 480 tasks) and cross-model-family generalization (Llama-VL, GPT-4o-V, Gemini-Pro-VL) are explicitly future work. Mechanistic Stage 2 findings are scoped to the curated mirage-disagreement task tiers (composite score-based curation per `curate_mirage_tasks.py`) on classifieds (and reddit if cells F/G replicate); broader phantom-routing-space mechanism universality is conditional on the 2x2 + cross-site control results."
343:## Appendix A — Decision Log
348:| 2026-05-03 | H3 structural test changed from McNemar exact (asymmetry) to bootstrap CI (non-emptiness) | McNemar tests directional dominance (which axis dominates), but H3 only requires non-empty unique contribution; bootstrap CI on count > 0 is the right test |
349:| 2026-05-03 | TOST δ = 1.0pp locked (was 0.5pp draft) | 0.5pp = 1 task in N=234 too liberal; 1.0pp = 2 tasks ≈ bootstrap SE noise floor; statistically principled |
350:| 2026-05-03 | K_h1 = 0.75 cell-pass threshold for H1 | Allows ~25% capability-outlier cells; not so strict as to break on single-cell noise |
351:| 2026-05-03 | K_h3 = 0.67 cell-pass threshold for H3 | Lower than K_h1 because structural < deployment commit |
352:| 2026-05-03 | Disconfirmation rule changed from "any cell fail" to data-conditional R1-R5 framing rule | "Any cell fail" too strict given single-cell power limits; framing rule maps data outcomes to paper hook revisions transparently |
354:| 2026-05-05 | Advisor sync 5/5 partial outcome — early-stop A locked (cancel全 mechanism); compute path locked (advisor 5090 → Rancher H100 → RunPod backup); paper split direction discussed but Mechanistic-nested-vs-independent + threshold detail not finalized due to network drop | Advisor explicit confirm early-stop cancel + compute paths; paper split + threshold lock deferred to email follow-up via `docs/checkpoints/advisor_sync_5_5_followup.md` |
356:| 2026-05-13 | **Codex stress audit triggered 6 paper-grade design fixes** (pre-launch): (a) scope reframe 16-cell phantom-only → 24-condition / 4-cell Phase 1a (cls+red×B0+B1×6modes), Phase 1b shop deferred to main paper; (b) K-of-N reclassified gate → transparency-only (power analysis showing dysfunction at < 7pp effects, re-propagated to H1/H3/R5/§6); (c) H1 drop-one definition disambiguated (oracle ceiling lift with-vs-without P-SoM, per (site, model) cell paired bootstrap); (d) smoke-gate B7 revised outcome-independent (no SR-based restart bias); (e) cell terminology disambiguated ("cell" = 4 statistical strata for K-of-N/meta input, "condition" = 24 operational launch units); (f) Phase 1b shop scope-expansion lever for main paper R3→R1 framing decision | Codex CLI hostile reviewer audit (`docs/checkpoints/codex_outputs/codex_stress_16cell_design_2026-05-13.md`, lean prompt no-enumeration, cross-AI complementary to prior Claude reviews); 6 HIGH severity findings + 3 probable concerns. Workshop-targeted Phase 1a launch this week; main paper Phase 1b after workshop submission |
357:| \<pending advisor email follow-up\> | \<witness K_h1=0.75 transparency / K_h3=0.67 transparency / TOST δ=1.0pp / N_conditions=24 (Phase 1a) / N_cells=4 / split protocol / paper split / Phase 1b shop / outcome-indep smoke gate / per follow-up doc Q1-Q11\> | \<email reply timestamp + Git SHA at lock\> |

exec
/bin/bash -lc 'rg -n "''^(##|###|#)|power|K|binomial|effect|3 of 4|transparency|family|simulation|alpha" docs/analysis/cross_sites/power_analysis.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
1:# Power Analysis — Observed-SR Update (B9 ✓)
5:This appendix updates the pre-registered power analysis with **observed adjusted-SR levels** from `sr_fp_per_mode.md` (Phase 1 B0 + B1 done cells, pre-paper-grade rerun). The post-rerun version will replace this file once 16-cell aggregation completes.
7:## 1. Observed adjusted-SR ranges (per `sr_fp_per_mode.md`)
15:**Observed effect-size range** (phantom-mode minus best non-phantom baseline):
22:**Modal effect size**: 1-5pp range, with phantom modes clustered at 0-4pp.
24:## 2. Per-cell MDE at observed SR levels (paired design, α=0.05 two-sided, β=0.20)
26:Run: `python3 scripts/analysis/power_analysis.py --baseline-sr {0.10,0.15,0.20}`
34:**Key observation**: minimum detectable effect at 80% per-cell power is **5-7pp** for cls/red, **4-5pp** for shop. The **observed mechanism effect (1-5pp)** is at or below per-cell MDE in 2 of 3 sites — **per-cell power for typical phantom effects is < 50%**.
36:## 3. Family-wise power at observed effects (K-of-N rule, baseline SR=0.15 proxy)
38:| Per-cell power (proxy effect on smallest site) | K_h1=12/16 family power | K_h3=11/16 family power |
48:- **K_h1=12/16** is calibrated for **≥7pp effects** with paper-grade ≥0.80 family power. For typical phantom mechanism effects (1-5pp), K_h1 family power is **<10%**.
49:- **K_h3=11/16** is slightly more permissive but still requires per-cell power ≥0.65 (≈6pp effect at SR=0.15) to reach 0.49 family power.
51:## 4. Methodological implication & paper-§3 framing update
53:The K-of-N family-wise rule was originally pre-registered as a **transparency / aggregation** check, not the primary detection mechanism. With the corrected interpretation:
55:- **Primary effect-detection test** = DerSimonian-Laird random-effects meta-analysis (locked by B8) on cells with N≥10. This is power-adequate at the cross-cell level for effects ≥2pp.
57:- **K-of-N rule** = retained as a **secondary transparency check** documenting how many cells *individually* clear α=0.05; not a gate on the H1/H3 paper claims.
58:- This recharacterization is **not post-hoc cherry-picking**: the random-effects meta + TOST were always the primary tests in `preregistration.md §4`. The K-of-N rule is restated as transparency.
60:## 5. Reviewer-rebuttal language
62:"At observed adjusted-SR levels (8-15% across sites) and observed mechanism effect sizes (1-5pp), per-cell statistical power is below 0.55 in two of three sites. We therefore rely on (a) DerSimonian-Laird random-effects meta-analysis across all cells (B8 lock; cross-cell pooling raises effective power) and (b) TOST equivalence on the full N=910 pooled task set (δ=1.0pp margin) as primary tests. The K-of-N family-wise rule pre-registered for transparency is not powered for sub-5pp effects, and we report its outcome as a secondary observation rather than a gate on the main hypotheses."
64:## 6. Bug history
66:The original `power_analysis.py` (pre-2026-05-09) contained a stale interpretation block claiming "K_h1=12/16 with per-cell power=0.65 → >0.95 family-wise power" — this was numerically inconsistent with the K-of-N table on the same page (actual value 0.289). Fixed in commit (current session) to compute family power **at observed effect sizes** rather than at hypothetical per-cell power levels. The corrected version is what this appendix relies on.
69:- `scripts/analysis/power_analysis.py` (script, fixed)
70:- `docs/analysis/cross_sites/power_analysis_observed_sr10.md` / `sr15.md` / `sr20.md` (full per-baseline tables)
72:- `docs/checkpoints/pre_run/preregistration.md §4` (B8 random-effects lock + TOST policy)
74:## 7. Open items (post-rerun)
77:- [ ] Re-verify K-of-N rule reframing in `preregistration.md` (audit item A1 / advisor review)
78:- [ ] Add `power_analysis_post_rerun.md` once observed effect sizes are final (replace this file)

exec
/bin/bash -lc 'rg -n "''^(##|###|#)|theory|H1|H2|H3|layer|claim|Phase|hook|oracle|cost|route" docs/checkpoints/paper_planning.md docs/reference/EVIDENCE_LAYER_AUDIT.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/reference/EVIDENCE_LAYER_AUDIT.md:10:# Evidence-Layer Audit & Action Queue
docs/reference/EVIDENCE_LAYER_AUDIT.md:14:> **What this doc is NOT**: 不是 finding registry / 不是 paper hook / 不是 result narrative。**纯方法论 + viz 装备清单**, 跟数据 direction agnostic。
docs/reference/EVIDENCE_LAYER_AUDIT.md:20:## §0 Audit scope (where evidence lives now)
docs/reference/EVIDENCE_LAYER_AUDIT.md:23:- `phantom_lift.{md,csv}` — Outcome 0c/0d (3→5/6-mode oracle lift, drop-one, Jaccard)
docs/reference/EVIDENCE_LAYER_AUDIT.md:26:- `cost_per_mode.{md,json}` — Efficiency cost
docs/reference/EVIDENCE_LAYER_AUDIT.md:34:- `aggregate_cost_electricity.py` — cost + electricity
docs/reference/EVIDENCE_LAYER_AUDIT.md:43:- fig3a (token_cost), fig3c (latency), fig3d (Pareto), fig3 (regional_carbon)
docs/reference/EVIDENCE_LAYER_AUDIT.md:48:## §1 Methodology + visualization gap registry
docs/reference/EVIDENCE_LAYER_AUDIT.md:50:按 4 evidence type × 4 cross-X axis 组织 (paper §2 Zoom 1-4 evidence layer skeleton). 每行: stats gap + paired viz gap + tier + ETA.
docs/reference/EVIDENCE_LAYER_AUDIT.md:52:### A. Cross-cutting methodology (优先级最高)
docs/reference/EVIDENCE_LAYER_AUDIT.md:57:| **A2** | **Pre-registration doc** `docs/checkpoints/pre_run/preregistration.md` — primary H1 / secondary H2-Hn / disconfirmation conditions / multiple-comparison family / decision rule. Git commit timestamp = registration time. | **Hypothesis × outcome confirmation matrix** (filled post-rerun): hypothesis row × cell column × {pass/fail/inconclusive} cell coloring (`fig_hypothesis_matrix.py`) | **T0** | 2h doc (need user-decided H list) + 1h viz scaffold |
docs/reference/EVIDENCE_LAYER_AUDIT.md:59:| **A4** | **TOST equivalence test** for "effect ≈ 0" reverse-claim ability — per-arm vs equivalence margin δ=0.5pp, two one-sided tests. | **Equivalence bound viz** (`fig_tost_bounds.py`) — CI bar with ±δ shaded region overlay, per arm | T1 | 2h stats + 1h viz |
docs/reference/EVIDENCE_LAYER_AUDIT.md:60:| **A5** | **Effect-size standardization 跨连续 outcome**: Cohen's d (cost / latency 连续) + Cliff's δ (AUROC non-parametric) + CI for Cohen's h (currently point estimate only) | **Effect-size CI panel** (`fig_effect_size_panel.py`) — h / d / δ as 3-panel forest, per arm | T1 | 2h stats + 1h viz |
docs/reference/EVIDENCE_LAYER_AUDIT.md:63:### B. Outcome (SR) — cross-mode primary
docs/reference/EVIDENCE_LAYER_AUDIT.md:71:| B5 | **Achievable-vs-ceiling SR gap** — oracle ceiling vs single-best-mode achievable | (folded into existing fig0c phantom_lift_bars with new layer) | T2 | 1h stats |
docs/reference/EVIDENCE_LAYER_AUDIT.md:73:### C. Macro (action freq, episode length, finish rate) — cross-mode
docs/reference/EVIDENCE_LAYER_AUDIT.md:82:### D. Micro (per-step) — cross-mode
docs/reference/EVIDENCE_LAYER_AUDIT.md:91:### E. Efficiency (cost / latency / carbon) — cross-mode
docs/reference/EVIDENCE_LAYER_AUDIT.md:95:| E1 | **Cost median + bootstrap CI** in `cost_per_mode.md` (currently point estimate only) | **Cost errorbars in fig3a** — currently point, add CI whiskers | T1 | 2h stats + 0.5h viz |
docs/reference/EVIDENCE_LAYER_AUDIT.md:96:| E2 | **Latency-vs-cost ratio per arm with CI** | (folded into fig3c with CI) | T2 | 1h stats |
docs/reference/EVIDENCE_LAYER_AUDIT.md:97:| E3 | **Cost-effectiveness ratio (lift / cost) with CI** — paper §6 routing decision 用 | **CE ratio forest** (`fig_ce_ratio_forest.py`) — per arm, lift÷cost with CI | T1 | 2h stats + 1h viz |
docs/reference/EVIDENCE_LAYER_AUDIT.md:99:| E5 | **Multi-metric Pareto (cost + latency + carbon)** — already flagged in next_steps §5 | **Trellis or 3D Pareto** (`fig_multi_pareto.py`) | T2 | 2h stats + 2h viz |
docs/reference/EVIDENCE_LAYER_AUDIT.md:101:### F. Cross-X axis (interaction tests)
docs/reference/EVIDENCE_LAYER_AUDIT.md:106:| F2 | **B0 × B1 cross-model interaction test** for "capability-modulated reversal" claim — currently narrative | **B0×B1 interaction crossed line plot** (`fig_capability_interaction.py`) — y=drop-one, x=axis (text/image), lines for B0/B1, crossover visible | **T1** (advisor sync 用) | 3h stats + 1h viz |
docs/reference/EVIDENCE_LAYER_AUDIT.md:110:### G. Reproducibility / determinism reporting
docs/reference/EVIDENCE_LAYER_AUDIT.md:116:| G3 | **Per-cell config diff manifest** — Phase A pre/post / SoM-prompt v1/v2 — unified manifest replacing git-log scrape | (no viz, table only in run_manifest) | T1 | 1h manual |
docs/reference/EVIDENCE_LAYER_AUDIT.md:120:## §2 Pre-registration template (T0e, blocks rerun launch)
docs/reference/EVIDENCE_LAYER_AUDIT.md:125:> - **Hero claim** (P-SoM as deployment routing arm) — pre-registered strict
docs/reference/EVIDENCE_LAYER_AUDIT.md:126:> - **4-fold drop-in property** — pre-registered strict (4 sub-claims a/b/c/d)
docs/reference/EVIDENCE_LAYER_AUDIT.md:127:> - **2-axis structural claim** (phantom space is multi-region, not collapsed point) — pre-registered with low-threshold non-overlap evidence requirement
docs/reference/EVIDENCE_LAYER_AUDIT.md:128:> - **Framing decision rule** — pre-registered, data-conditional (paper hook 升降级 mapping)
docs/reference/EVIDENCE_LAYER_AUDIT.md:142:# Phantom-SoM Pre-Registration
docs/reference/EVIDENCE_LAYER_AUDIT.md:144:## Hypotheses
docs/reference/EVIDENCE_LAYER_AUDIT.md:146:### PRIMARY (gates paper claim)
docs/reference/EVIDENCE_LAYER_AUDIT.md:148:H1 (Hero deployment claim — P-SoM is hidden routing arm):
docs/reference/EVIDENCE_LAYER_AUDIT.md:158:H2 (4-fold drop-in property — P-SoM specifically):
docs/reference/EVIDENCE_LAYER_AUDIT.md:159:  All four sub-claims hold per cell, replicated in ≥ K_h1 cells:
docs/reference/EVIDENCE_LAYER_AUDIT.md:160:    (a) median cost(P-SoM) within ±10% of median cost(DOM)
docs/reference/EVIDENCE_LAYER_AUDIT.md:163:    (d) P-SoM drop-one magnitude ≥ 1.0pp (=H1 (iii); folded)
docs/reference/EVIDENCE_LAYER_AUDIT.md:165:H3 (2-axis empirical structural claim — phantom space is not collapsed point):
docs/reference/EVIDENCE_LAYER_AUDIT.md:171:          (lower threshold than H1: structural claim, NOT deployment)
docs/reference/EVIDENCE_LAYER_AUDIT.md:179:### EXPLORATORY (post-data, no pre-commit threshold)
docs/reference/EVIDENCE_LAYER_AUDIT.md:187:  framework was developed after observing N=4 pre-Phase-A cells.
docs/reference/EVIDENCE_LAYER_AUDIT.md:197:### FRAMING DECISION RULE (pre-registered, data-conditional)
docs/reference/EVIDENCE_LAYER_AUDIT.md:199:R1 IF (H1 holds AND H2 holds AND H3 (i) AND (ii) hold):
docs/reference/EVIDENCE_LAYER_AUDIT.md:203:   → Paper §1 hook: STRONGEST.
docs/reference/EVIDENCE_LAYER_AUDIT.md:205:R2 IF (H1 holds AND H2 holds AND only one of H3 (i)/(ii) holds):
docs/reference/EVIDENCE_LAYER_AUDIT.md:209:   → Paper §1 hook: MODERATE-STRONG.
docs/reference/EVIDENCE_LAYER_AUDIT.md:211:R3 IF (H1 holds AND H2 holds AND neither H3 (i)/(ii) holds):
docs/reference/EVIDENCE_LAYER_AUDIT.md:215:   → Paper §1 hook: MODERATE (= 04-30 fallback framing).
docs/reference/EVIDENCE_LAYER_AUDIT.md:217:R4 IF (H1 holds AND H2 partially fails — e.g., (a) cost or (b) latency
docs/reference/EVIDENCE_LAYER_AUDIT.md:220:                    deployment limitations" + §4 disclosure of failed sub-claim.
docs/reference/EVIDENCE_LAYER_AUDIT.md:221:   → Paper §1 hook: WEAK; substantial revision needed.
docs/reference/EVIDENCE_LAYER_AUDIT.md:223:R5 IF (H1 fails: pooled meta sig fails Holm OR < K_h1 cells individually sig):
docs/reference/EVIDENCE_LAYER_AUDIT.md:229:## Multiple-Comparison Family
docs/reference/EVIDENCE_LAYER_AUDIT.md:232:    H1 (i) pooled meta:    m = 1 (no correction within family)
docs/reference/EVIDENCE_LAYER_AUDIT.md:233:    H1 (ii) per-cell P-SoM: m = N_cells
docs/reference/EVIDENCE_LAYER_AUDIT.md:234:    H2 sub-claims (a)(b)(c)(d): m = 4 × N_cells
docs/reference/EVIDENCE_LAYER_AUDIT.md:238:    H3 (i) axis 1 per-cell:  m = N_cells
docs/reference/EVIDENCE_LAYER_AUDIT.md:239:    H3 (ii) axis 2 per-cell: m = N_cells
docs/reference/EVIDENCE_LAYER_AUDIT.md:241:    Rationale: structural claim is weaker than deployment, separate family
docs/reference/EVIDENCE_LAYER_AUDIT.md:247:    BH FDR q-value reported for transparency, not used for paper claim gating.
docs/reference/EVIDENCE_LAYER_AUDIT.md:249:## Locked Analysis Choices (pre-data)
docs/reference/EVIDENCE_LAYER_AUDIT.md:251:  Primary metric: oracle ceiling SR pp lift (binary, paired)
docs/reference/EVIDENCE_LAYER_AUDIT.md:256:  H1 K_h1 cell-pass threshold: 0.75 (75% of cells must Holm-sig)
docs/reference/EVIDENCE_LAYER_AUDIT.md:257:  H3 K_h3 cell-pass threshold: 0.67 (67%, lower because structural < deployment)
docs/reference/EVIDENCE_LAYER_AUDIT.md:258:  H3 unique-count floor: ≥ 2 tasks per cell
docs/reference/EVIDENCE_LAYER_AUDIT.md:259:  Cell inclusion: Phase A post-fix only (commit ≥ 3c15cd7) for main analysis;
docs/reference/EVIDENCE_LAYER_AUDIT.md:265:## Exploratory (NOT pre-registered, paper must explicitly flag)
docs/reference/EVIDENCE_LAYER_AUDIT.md:270:  - Any post-hoc cell subsetting beyond H1-H6 family scope
docs/reference/EVIDENCE_LAYER_AUDIT.md:271:  - 别扭 / capability-reversal explanations (H5/H6) — post-hoc theory, NOT validation
docs/reference/EVIDENCE_LAYER_AUDIT.md:273:## Witness Mechanism
docs/reference/EVIDENCE_LAYER_AUDIT.md:284:## §3 Action queue (ordered, T0 → T1 → T2)
docs/reference/EVIDENCE_LAYER_AUDIT.md:286:### T0 — Pre-rerun launch (blocks 14-cell start)
docs/reference/EVIDENCE_LAYER_AUDIT.md:292:- [ ] **T0e — A2 pre-registration doc**: 写 `docs/checkpoints/pre_run/preregistration.md` (上方 §2 template). 需要 user lock H1-H5 specifics + advisor 见证. _ETA 2h_
docs/reference/EVIDENCE_LAYER_AUDIT.md:297:### T1 — Advisor sync + paper §5/§7 commit ready
docs/reference/EVIDENCE_LAYER_AUDIT.md:307:- [ ] **T1i — F2 B0 × B1 interaction test + viz** ⭐ (advisor sync, capability-modulated reversal claim)
docs/reference/EVIDENCE_LAYER_AUDIT.md:308:- [ ] T1j — E1 cost CI + E3 CE ratio forest
docs/reference/EVIDENCE_LAYER_AUDIT.md:313:### T2 — Paper end-stage prose
docs/reference/EVIDENCE_LAYER_AUDIT.md:318:- [ ] T2d — E2 latency-vs-cost ratio + E4 Pareto CI + E5 multi-metric Pareto
docs/reference/EVIDENCE_LAYER_AUDIT.md:324:## §4 Tracking
docs/reference/EVIDENCE_LAYER_AUDIT.md:344:## §5 References
docs/checkpoints/paper_planning.md:1:# Paper 1 Strategy & Notes (Phantom-SoM)
docs/checkpoints/paper_planning.md:4:> 含 theory framework / findings 列表 / risks / cascade / router design /
docs/checkpoints/paper_planning.md:8:> - **paper_planning.md** (此文档): paper strategy, theory, findings, risks
docs/checkpoints/paper_planning.md:13:> **Last updated**: 2026-05-04 deepest-evening late (§21.5 fundamental hook reframe to **research-characterization angle** — user push: "工业用 ptext 是为了省花费, 不知道 text 扁平化有独特效果, 因为他们无法把 dom 跟 ptext 对比"。Paper §1 hook 不应 claim "first inference-time substitution" (industry agent-browser/Tarsier 已部署), 改 claim "first systematic peer-reviewed **characterization** of routing behavior across phantom space configurations on Qwen3-VL via controlled cross-mode comparison". Industry deployment ≠ research finding — different epistemic levels. 全部 4 phantom corners 同样 novel-as-research-cells, 不是 P-text trivial vs P-SoM novel.)
docs/checkpoints/paper_planning.md:15:> **2026-05-04 deepest evening**: §21 fact-check correction — prior "interactive-only filter / P79 preserve all elements" over-claim 撤回 after reading `external/visualwebarena/browser_env/processors.py:513-619`. Format-axis orthogonal not scope-axis (scope similar across P79 and industry SDK).
docs/checkpoints/paper_planning.md:19:> **2026-05-04 late evening**: §21 ROUND-3 fact-check integration — Round-2 hallucinations corrected: HMT author Tan/Gao/Wu BIT (not Huang); NLAH dropped (was lying citation); WebAIM 2026 actual 59.1 vs 42 (not 57 vs 27); Operator 41.7% misattribution dropped (was MAI-UI success rate); Doubao bans dropped (no Chinese press verification); K3 Mariner / ActionEngine dropped (hallucinated). New Round-3 verified specifics: Magma uses Qwen3-VL backbone (same family as our paper); ScribeAgent fine-tunes Qwen 7B 6B-token corpus to WebArena 51.3%; NLWeb deployed at Tripadvisor + Shopify with `/ask`+`/mcp` endpoint spec; OmniParser-v2 SPS literal format `<box_start>...<box_end>`; AppAgent-v2 view_state_id JSON schema; Mind2Web 2 WebJudge 3-category taxonomy; cost anchors 241K vs 47-140K tokens; MCP 100% tool spoofing vulnerability
docs/checkpoints/paper_planning.md:21:> **2026-05-04 evening**: §21 EXPANDED — DR audit findings integrated: industry precedent stack mapped to 9-cell matrix [NLWeb / OmniParser-v2 / Magma / AppAgent-v2 / ScribeAgent / UI-TARS / HMT]; (ii)×L3 internal 4-tier sub-gradient identified [pretrain / RAG / inference-time / pure-visual], paper-1 occupies inference-time niche; §21.5 candidate paper §1 hook prose with substitution-gradient framing; §21.6 WebAIM 2026 + WebSuite + CAPTCHA counter-evidence stack
docs/checkpoints/paper_planning.md:23:> **2026-05-04 morning**: §21 NEW — Environment-Agent Intervention Taxonomy 3×3 matrix; 笔记 §1-§108 audit, ~40 entries mapped to 9 cells; paper-1 main hook = (ii)×L3 phantom routing space; identified-but-unfixed (iii)×L2 channel-addition gaps as paper §5 ceiling argument material; paper-level methodology asymmetry inventory §47/§95/§96/§101
docs/checkpoints/paper_planning.md:25:> **Previous**: 2026-05-03 (pre-registration framework reframe: Hero+Structural+Framing-rule replacing 3-arm a-priori commit; preregistration.md draft + EVIDENCE_LAYER_AUDIT.md §2 anchor; T0a-d evidence-layer infra done)
docs/checkpoints/paper_planning.md:27:> **Previous**: 2026-05-01 (hook reframe to phantom space 3 arms; §2 cube boundary definition; axis 1/2 LLM mechanism refine)
docs/checkpoints/paper_planning.md:31:## §1 Paper Hook + Tagline
docs/checkpoints/paper_planning.md:33:> **2026-05-03 reframe note**: Paper hook framing is now **data-conditional** per pre-registered framing decision rule (R1-R5; see `docs/checkpoints/pre_run/preregistration.md` §2). The "core finding" below corresponds to **rule R1 (STRONGEST)** — applies if H1+H2+H3(i)+(ii) all hold post-rerun. If H3 fails, hook falls back to "Phantom-SoM is hidden 4th routing arm" (R3, MODERATE). The Hero (P-SoM deployment) + Structural ablation (P-text/P-prompt non-overlap) + Framing-rule structure replaces the older "3-arm a-priori commit" framing — see `docs/reference/EVIDENCE_LAYER_AUDIT.md` §2 for epistemic rationale.
docs/checkpoints/paper_planning.md:35:**Core finding (under R1, contingent on H3 empirical validation)**: We discover a **hidden phantom routing space** for web agents — defined by the boundary "**skip annotated image**" — containing a **2-axis empirical structure** (axis 1 = text payload via P-text; axis 2 = SoM-style prompt via P-prompt) with **P-SoM (cube center, axis 1 + axis 2 compound) as the deployment hero**. P-SoM satisfies a **4-fold drop-in property**; P-text and P-prompt serve as **structural ablation arms** validating axis decomposition:
docs/checkpoints/paper_planning.md:42:| (d) **Drop-one oracle 1.7-3.8pp per phantom arm** | B0 red: P-text +3.81pp / P-SoM +3.33pp / P-prompt +2.86pp (all sig CI excludes 0); cls: P-text +3.42pp / P-SoM +2.56pp; B1 cls P-SoM +1.71pp. **Phantom space 3 arms 都贡献 unique tasks**, 6-mode oracle vs 3-mode lift +7.14pp [3.81, 10.48] (B0 reddit) |
docs/checkpoints/paper_planning.md:45:> "We discover a hidden **phantom routing space** in SoM-style web agents — defined by the boundary 'skip annotated image' — containing 3 routing arms (P-text / P-prompt / P-SoM) sharing a **4-fold drop-in property**: cost ≈ DOM (no image embedding tax), ~50% lower latency (no image inference stage), signal AUROC ≥ baseline (routing infra drop-in), drop-one oracle 1.7-3.8pp per arm (all sig). Two LLM mechanisms create this space: (i) text-payload flattening (AXTree → `[SOM_MARKS]`) reframes the agent's task ontology from web-browsing to indexed selection (axis 1); (ii) SoM-style visual prompting without image still activates the agent's visual-mark referencing parsing and recovers a substantial fraction of visual structure information textually (axis 2; **Mirage Effect** Asadi et al. 2026 (arXiv:2603.21687) — VLM 无图准确率 ~70-80% of with-image; **Scaffold Effect** Vu & Balloccu 2026 — prompt mentioning modality alone explains 70-80% performance shift independent of image presence). P-SoM (cube center, axis 1 + axis 2 compound) is the space's representative arm; SoM (image-on cube endpoint) and Vision (image-only, outside cube) anchor the comparison. **The 3-axis cube framework (orthogonalizing image-presence as a controllable axis distinct from text payload and prompt format) and cube-center P-SoM (`[SOM_MARKS]` text + SoM-prompt + no image) are paper-level framework contributions** — industry deploys text-only OR SoM-with-image, never the cube-center SoM-text-without-image combination; industry uses these configurations arbitrarily for token economy, never compared P-text vs DOM nor characterized per-dimension routing behavior. Paper discovers text-flattening has independent routing effects beyond cost (drop-one unique tasks, M1 ontology reframe). The space is site-modulated (cls visual-rich requires image; red text-dominated thrives in phantom space) and routing-deployable (B0 red 6-mode oracle lift +7.14pp over 3-mode baseline)."
docs/checkpoints/paper_planning.md:47:### Cascade design (token-monotonic, paper Section 6)
docs/checkpoints/paper_planning.md:59:**Order rationale**: Step 1+2 都 **0 增量 token**，第 3 步才付 image embedding tax — token-monotonic cascade，trigger router 不需要"先加再删"。Vision 是另条独立路径（image-only, no text），适合纯 visual task。
docs/checkpoints/paper_planning.md:63:## §2 Theory Framework — Mechanism Activation + Phantom Space Boundary (大重写 2026-05-01)
docs/checkpoints/paper_planning.md:65:> **Cross-reference (added 2026-05-04)**: §2 是 **mechanism explanation layer** (Zoom 1-4 解释假说); 跟 §21 **intervention taxonomy** (3-spectrum × 3-layer) 是 orthogonal 维度。**§2 关心 phantom routing space 内部 mechanism** (M1/M2 axis activation, etc.); **§21 关心 substitution gradient 上 paper-1 跟 industry precedents 的 substrate 站位** (NLWeb / OmniParser-v2 / Magma / ScribeAgent / AppAgent-v2 / UI-TARS vs phantom routing). 两个 view 互补不替代。
docs/checkpoints/paper_planning.md:68:> - **Explanation layer** (因果假说, §2 主住所): Zoom 1 architectural / Zoom 2 axis behavioral / Zoom 3 named phenomena / Zoom 4 model-internal
docs/checkpoints/paper_planning.md:69:> - **Evidence layer** (观测数据, §3 主住所): Outcome / Macro / Micro / Efficiency × cross-task / mode / site / model
docs/checkpoints/paper_planning.md:77:> - ❌ "Three-layer mechanism argument" (Layer 1/2/3) 命名 — 改用 evidence/explanation 双层 + Zoom 1-4 (§108.6)
docs/checkpoints/paper_planning.md:81:### Zoom 1 (architectural): Phantom space boundary + 2-axis activation by design
docs/checkpoints/paper_planning.md:101:  - (a) cost ≈ DOM — no image embedding tax (`[SOM_MARKS]` 是 AXTree regex filter，~3K text both)
docs/checkpoints/paper_planning.md:104:  - (d) drop-one oracle 1.7-3.8pp positive per arm — emergent (B0 red: P-text +3.81 / P-SoM +3.33 / P-prompt +2.86，all sig CI excludes 0)
docs/checkpoints/paper_planning.md:107:**Why exclude #2/#4/#6 (3 image-on phantom corners)**: 一旦加 annotated image 回去，cost / latency / carbon 都跟 SoM 拉齐 → 失去 4-fold drop-in property → 不再属于 phantom space。这 3 个 corners 在 routing 维度上是 SoM 的 variants (image cost dominate)，不提供 phantom-class deployment value。即 **boundary 是 "no annotated image"，不是 "matched parsing"**。
docs/checkpoints/paper_planning.md:111:- P-prompt 有真 LLM 机制 (axis 2 effect alone)：**visual prompting without image** —— SoM-style prompt 即使无图仍 activate agent 的 visual-mark referencing parsing，agent 在 AXTree 上自动 fallback 到 element_id 引用，仍 recover 部分 visual 结构信息。**Lit anchor stack** (3 互补 mechanism): (a) **Mirage Effect** Asadi et al. 2026 (arXiv:2603.21687, Stanford) — VLM 无图准确率达有图的 ~70-80% (实验笔记 §18); (b) **Scaffold Effect** Vu & Balloccu 2026 — prompt 仅提及 modality 可用就解释 70-80% performance shift independent of image presence (实验笔记 §25 + phantom_som.md §3.5); (c) **Cross-modal flow** Kaduri et al. — middle-layer cross-modal flows store image info in query tokens enabling image-consistent generation without direct image-token attention (phantom_som.md §2.1)
docs/checkpoints/paper_planning.md:113:- 6-mode oracle vs 5-mode +1.90pp [0.48, 3.81] sig 验证 P-prompt 贡献 incremental unique tasks (6 tasks added, 1 unique to P-prompt)
docs/checkpoints/paper_planning.md:129:### Zoom 1 (architectural completeness): Approach 2 deductive argument
docs/checkpoints/paper_planning.md:139:            (T=0 + greedy decoding 假设, Phase A 后真; 但 B0 proxy 仅 decision-level
docs/checkpoints/paper_planning.md:151:### Zoom 2 (behavioral): M1/M2 mechanism activation 2x2 framework
docs/checkpoints/paper_planning.md:178:### Zoom 2.5 (reverse explanation, NEW 2026-05-01 evening): 别扭 framework + Capability-modulated effect
docs/checkpoints/paper_planning.md:180:> **⏸️ Provisional**: 现有数据 N=4 cells (B0 cls/red 含 phantom + B1 cls 5-mode + B1 red 3-mode), 全部 Phase A bug fix 之前 (commit `3c15cd7` 之前). 16-cell rerun 后 statistical commit, 现 framework 标 "provisional pending 16-cell rerun + cross-VLM-family validation"。详 笔记 §108.16。
docs/checkpoints/paper_planning.md:182:**Insight**: M1/M2 framework 是 **forward causal upstream** ("what input change happens"). 别扭 framework 是 **reverse causal downstream** ("what gap between expectation and reality"). 两者描述同一现象的不同 layer, 但 别扭 在 phantom space 内提供更 mechanism-aligned 解释。
docs/checkpoints/paper_planning.md:217:**Empirical cross-cell validation (4 cells, Phase A pre-fix data)**:
docs/checkpoints/paper_planning.md:242:- §1 hook drop-one "1.7-3.8pp per arm" 加 capability-modulated caveat ("magnitude 4× weaker on small VLM, direction reverses text-vs-image axis preference")
docs/checkpoints/paper_planning.md:243:- §2 Theory: forward (M1/M2) + reverse (别扭) **layered framework** — forward describes design + measurement, reverse describes mechanism + interpretation
docs/checkpoints/paper_planning.md:249:- Phase A bug pre-fix data — cycle false positives / dispatch noise affect aggregate metrics
docs/checkpoints/paper_planning.md:254:### Zoom 3 (named phenomena): Lit-anchored mechanism phenomena
docs/checkpoints/paper_planning.md:259:- **Cross-modal flow** (Kaduri et al.): middle-layer cross-modal flows store image info in query tokens, allowing image-consistent generation without direct image-token attention (phantom_som.md §2.1 line 83-89) — actually 这个偏 Zoom 4 mechanism
docs/checkpoints/paper_planning.md:266:### Zoom 3 expansion (5/6 Gemini DR returns 2026-05-01, Q5 pending)
docs/checkpoints/paper_planning.md:282:- ✅ **Paper §1 first-work claim VERIFIED**: "**no study isolates SoM-style flat text as a standalone observation without its accompanying marked screenshot. The target paper fills this unprecedented gap**" — Phantom-SoM "first systematic SoM-text isolation" claim is lit-verified novel
docs/checkpoints/paper_planning.md:303:- **Bidirectional Failure framing 是 "Novel synthesis"** (Q5 doc 自己 line 22 标注): "framing VLM modality interaction as exhibiting **dual failure modes that act in opposite directions** constitutes a novel theoretical synthesis. The current 2023-2026 literature largely treats these failure modes in isolation." → Paper §5 axis 3 prose 可 cite Q5 综述 + claim "first systematic web-agent multi-step application"
docs/checkpoints/paper_planning.md:319:- **Mid-layer attention decay (Liu 2025 "Devils in Middle Layers", arXiv:2512.07730)**: object hallucinations 是 multi-faceted — 同时由 mid-layer visual attention decay + decoding 时 language prior dominance 导致。**Paper §5 axis 3 双因机制 anchor** (从单 cause 升级到 dual cause 解释)
docs/checkpoints/paper_planning.md:326:### Zoom 3 counter-evidence catalog (NEW 2026-05-01 from Gemini DR, mandatory for paper §5 honest framing)
docs/checkpoints/paper_planning.md:333:- → **Paper §1 hook honest framing**: text-only fallback works on **standard web schema with stable DOM**; fails on perception-conditioned, dynamic GUI, behavioral tracking, structurally ambiguous tasks
docs/checkpoints/paper_planning.md:343:- **"Seeing but Not Believing" 2025**: linear probing on visual encoder shows it accurately extracts features; failure is **late-stage generative disconnect inside LLM**, NOT cross-modal interaction failure. Implication: paper §5 axis 3 prose 不应 over-commit "cross-modal failure" mechanism — 我们 8-channel 是 behavioral level taxonomy, 不 claim encoding-stage vs decoding-stage attribution
docs/checkpoints/paper_planning.md:354:> **🆕 Paper §1 first-work claim verified (Q3 Gemini DR 2026-05-01)**: Gemini deep research synthesis explicitly confirms "no study isolates SoM-style flat text as a standalone observation without its accompanying marked screenshot" — Phantom-SoM paper §1 hook "first systematic SoM-text isolation" claim is lit-verified novel. Reviewer attack vector "you are not first" is preempted via Q3 forward-citation-chain analysis (FOCUSAGENT prunes hierarchy but doesn't compare to flat list; HMT compares hierarchical vs flat in memory architecture, not observation; Yang 2023 SoM original always bundles text with marked image).
docs/checkpoints/paper_planning.md:356:### Zoom 4 (model-internal): Mechanistic probe lit anchors (paper §8 future work)
docs/checkpoints/paper_planning.md:358:**Paper 不 self-probe Zoom 4** (因 B0 proxy API 不暴露 router logits internals + local deploy Qwen3-VL-235B-A22B 需 ~120GB VRAM 超 RunPod $200 budget). 但 lit anchors 给 mechanism plausibility:
docs/checkpoints/paper_planning.md:360:- **Cross-modal flow** (Kaduri et al.): layerwise attention probe 显示 middle-layer cross-modal flows enable image-like representations from query tokens — M1 axis activation 的 mechanistic 解释
docs/checkpoints/paper_planning.md:362:- **Tool Calling Linear Steerable Circuit** (Anonymous 2026 ACL, 笔记 §19 archived 2026-04-09): 在 **Qwen3-4B** (跟 B1 = Qwen3-VL-4B 是 architectural cousin) 验证 — 15 tools → 10 PCA 方向 (90.2% var), **cosine gap 捕获 92% action-selection 错误**, L23+ layer steering 切 tool 准确率 80-93%, "**knows but cannot say**" (hidden state 77-89% correct, output layer 3-61%)。**对 phantom-SoM 的暗示**: (a) action selection 在 **action-type axis 线性可分** + argument generation 非线性，给 §1 cascade `DOM→P-text→P-SoM→SoM` 的 token-monotonic path 一个 mechanistic 理由 (轴 selection 比 argument 廉价); (b) hidden-state cosine gap 是 B1 白盒 routing signal candidate，AUROC 可对比 logprob (~2300 forward pass，无需重跑 environment); (c) "knows but cannot say" 跟我们 §70 infra 观测到 Bedrock proxy 静默吞 `tools`/`tool_choice` 参数返回纯文本的现象在不同 stack layer 互相印证 (架构层 vs API 层 stack-wide brittleness)。**对 B1-side 平衡 anchor**: SteerMoE / Cross-modal flow 都偏 B0 (235B-A22B MoE) 路径，Tool Calling Linear Circuit 是 Zoom 4 anchor stack 中**唯一在 4B 模型直接验证**的，给 paper §8 future work B1 self-probe 一个直接 method template (output_hidden_states=True forward pass + PCA + cosine gap AUROC, 仅 B1 白盒可行)
docs/checkpoints/paper_planning.md:367:| Cross-modal flow (Kaduri) | layerwise attention probe | model-agnostic | M1 axis (Mirage / Scaffold) 的 mechanism 解释 |
docs/checkpoints/paper_planning.md:373:### Zoom 4 paper sequence implication
docs/checkpoints/paper_planning.md:384:    需要 local deploy 或 API extension (RunPod 4×4090 ~$400-600 cost)
docs/checkpoints/paper_planning.md:395:### Axis 1: Text payload structure (PRIMARY, first-order SR effect)
docs/checkpoints/paper_planning.md:405:- **Task ontology reframing — web-browsing → indexed selection** (核心 axis 1 effect, NEW 2026-05-01): AXTree (hierarchical tree, agent navigate tree structure 像 browser DOM walk) → `[SOM_MARKS]` (flat indexed list, agent picks ID 像 multiple-choice selection)。改变 LLM 任务 ontology 从 "browse the web" 到 "select from list"，trigger 不同的 in-context behavior。这是 P-text 4-mode drop-one +3.42-3.81pp 的根本机制。**Lit anchor**: deep research `docs/literature/The Novelty and Efficacy of Set-of-Mark Text...` line 84 frame 这个 split 为 "AXTree induces **tree traversal trajectory** (logical deduction over hierarchy) vs flat SoM induces **sequential list scanning trajectory** (rapid spatial approximation)"; paper draft `section2_background.md` line 27 已 adopt: "the flat marks list tends to shift exploration toward **quick element selection**, AXTree hierarchy supports **sustained navigation and search**". Sclar 2024 / Mishra 2022 prompt-format sensitivity theory 提供 transformer-level mechanism (different token distribution → distinct latent state → distinct exploration policy).
docs/checkpoints/paper_planning.md:413:### Axis 2: Prompt (multi-dimensional task-conditional decision prior)
docs/checkpoints/paper_planning.md:417:**Replaced theory**: Prompt acts as task-conditional decision prior over:
docs/checkpoints/paper_planning.md:423:**Visual prompting without image (P-prompt 的核心 LLM 机制, NEW 2026-05-01)**: SoM-style prompt 即使无 image 仍 activate agent 的 "visual-mark referencing" mental model —— prompt 期望 numerical-marker referencing system，agent 在 AXTree 上自动 fallback 到 element_id 引用，仍 recover **substantial fraction (~70-80%) of visual structure information from textual cues**。**Lit anchor stack (3 互补 mechanism)**: (a) **Mirage Effect** Asadi et al. 2026 (arXiv:2603.21687, Stanford) — VLM 无图时仍自信描述视觉特征，**无图准确率达有图的 70-80%** (mirage-mode > guess-mode), 实验笔记 §18 line 457; (b) **Scaffold Effect** Vu & Balloccu 2026 — prompt 仅提及 MRI 可用就解释 **70-80% 性能变化** independent of image presence (clinical VLM 起源, web agent 同样适用), 实验笔记 §25 + phantom_som.md line 281; (c) **Cross-modal flow** Kaduri et al. — middle-layer cross-modal flows enable VLMs to store image info in query tokens, allowing image-consistent generation without direct image-token attention (phantom_som.md §2.1 line 83-89). 即 axis 2 swap **不是** "prompt label 改"，而是 **task ontology 切换 from textual-action-prompt to visual-mark-referencing-prompt**。这与 axis 1 的 "browse → select" 形成对称 (axis 1 改 obs ontology，axis 2 改 action-referencing ontology)。
docs/checkpoints/paper_planning.md:432:- B0 red 6-mode oracle +7.14pp [3.81, 10.48] vs 3-mode (3 phantom arms 都贡献 unique tasks)
docs/checkpoints/paper_planning.md:436:### Axis 3: Image (8-channel multi-dimensional, codex `7106d2e` validated)
docs/checkpoints/paper_planning.md:440:**Replaced theory**: Image is **bidirectional modality fusion** with multiple sub-channels:
docs/checkpoints/paper_planning.md:445:- **Mid-layer mechanism dual-cause** (Liu 2025 "Devils in Middle Layers" arXiv:2512.07730): mid-layer visual attention decay + decoding-time language prior dominance — paper §5 axis 3 dual-cause anchor
docs/checkpoints/paper_planning.md:459:  - Lit: ⭐ Tong 2024 "Eyes wide shut" (CVPR), Bitton-Guetta 2023 WHOOPS! (ICCV); Fu 2024 BLINK (ECCV, perception primitives 24-30%); Liu 2025 Devils mid-layer attention decay; Guan 2024 HallusionBench "language hallucination" axis (Q5)
docs/checkpoints/paper_planning.md:471:### Site-modulated framing (LLM-level explanation)
docs/checkpoints/paper_planning.md:482:### Site mechanical substrate (full characterization, 2026-04-29)
docs/checkpoints/paper_planning.md:486:#### reddit (Postmill, N=210)
docs/checkpoints/paper_planning.md:499:#### classifieds (OSClass, N=234)
docs/checkpoints/paper_planning.md:512:#### shopping (Magento, N=466)
docs/checkpoints/paper_planning.md:523:| Site-specific quirks | Magento FPC cache full-page-cache requires hook + post-restart curl; custom-option radio swatch bug; review form ratings same bug pattern; long product comparison (12 items × 10 fields per Magento aggregation tasks) |
docs/checkpoints/paper_planning.md:526:#### Mechanism three-way table (Section 5 narrative scaffold)
docs/checkpoints/paper_planning.md:547:### Capability layer (B0 vs B1, lazy minimization §101.九)
docs/checkpoints/paper_planning.md:567:### Cross-axis interaction LLM mechanism
docs/checkpoints/paper_planning.md:576:### Paper contribution position (Section 5 framing)
docs/checkpoints/paper_planning.md:582:4. **Drop-in deployment claim** (4-fold drop-in property)
docs/checkpoints/paper_planning.md:587:### Literature gap 5-dimension (§103 anchor)
docs/checkpoints/paper_planning.md:594:| D. Prompt format sensitivity | **Yes (theory anchor)** — 但无 web agent 应用 | Sclar 2023, Mishra 2022 |
docs/checkpoints/paper_planning.md:603:## §3 Findings — 4-dimension Evidence + Mechanism Framework (重组 2026-04-29, 2026-05-01 update: evidence/explanation separation)
docs/checkpoints/paper_planning.md:605:> **Cross-reference (added 2026-05-04)**: §3 是 **paper finding evidence layer** (4 测量 × 4 cross-X = 16 sub-cells); 跟 §21 **intervention taxonomy** (~40 笔记 § environmental work) 是 orthogonal — §3 关心 phantom routing space 的 SR/cost/AUROC 实证, §21 关心 paper 周边 environmental scaffolding work 的 substrate categorization. **环境侧 fix** (§51-62 / §80 / §107 等 9 条) 是 §21 的 (i)+(ii)×L2 cell 内容, **paper §3 evaluation methodology footnote** acknowledge 这些 fix (跟 Avenir Web ignore env issue 形成 rigor differentiator), 但 detail 不在 §3 主分析。
docs/checkpoints/paper_planning.md:607:> **重组动因 (§105)**：之前 10 条 finding 是 flat list，paper 写作时不好定位"哪个证据支持哪个 claim"。重组为 **4-dimension framework** —— 每个证据进对应 dimension，每个 paper claim 引用 dimension (e.g. "Outcome 0d Jaccard 0.447 supports routing-arm complementarity")。四个 dimensions 是 **正交** 的（不是 hierarchical layers）。**所有原 10 条 finding 都映射到对应 dimension，未删除**（见末尾索引）。
docs/checkpoints/paper_planning.md:609:### Evidence vs Explanation Layer Separation (2026-05-01 update)
docs/checkpoints/paper_planning.md:617:   Outcome (SR/oracle/Jaccard)   cross-task   (within-cell aggregation, 统计 foundation)
docs/checkpoints/paper_planning.md:620:   Efficiency (cost/lat/carbon)  cross-model  (跨 capability generalization)
docs/checkpoints/paper_planning.md:636:**关键区分**: §3 4-dim 是 evidence layer 的**测量类型轴**, cross-X 是 evidence layer 的**比较 axis 轴**。两者**正交 organize 同一份数据**。Explanation layer 跟 evidence layer 严格分开 — explanation 是 hypothesis (Zoom 1-4), evidence 是 data。Paper writing 时 reviewer 最忌 evidence-explanation 混淆 ("Macro 1c search-loop 51.9→35.7%" 是 evidence, "M1 axis activates list-scanning trajectory" 是 explanation Zoom 2 — 两者必须分写然后 explicit link)。
docs/checkpoints/paper_planning.md:638:### Cross-X 比较 axis (4 类) 的 paper section mapping
docs/checkpoints/paper_planning.md:649:### 4-dimension framework 概览
docs/checkpoints/paper_planning.md:655:Efficiency  cost / latency / carbon (4-fold drop-in property)
docs/checkpoints/paper_planning.md:662:### Outcome — task 成功 / 路由 arm 证据
docs/checkpoints/paper_planning.md:668:| **0c** Routing oracle uplift (3-mode → 4/5-mode) + drop-one | `phantom_lift.{md,csv}` | red 3→5: **+5.24pp** [2.38, 8.11] Wilcoxon p=0.0009 McNemar p=0.0005 ✅; cls +4.70pp [2.14, 7.69] p=0.0009 ✅. red drop-one P-text +3.81pp / P-SoM +3.33pp; cls P-text +3.42pp / P-SoM +2.56pp |
docs/checkpoints/paper_planning.md:676:### Macro — agent 平均怎么 act
docs/checkpoints/paper_planning.md:680:| **1a** Tier 1 hook (3-mode coarse: DOM/P-SoM/SoM × 8 metric) | `axis_effect_size.py` (FRESH 04-29) + `axis_effect_size_report.md` | P-SoM "fully independent" cells: **red 4/8 vs cls 1/8**. cls P-SoM 主要"瘫向 DOM" (6/8 DOM-like) —— image axis 决定性, **印证 0d 的 task-pool 复杂性** |
docs/checkpoints/paper_planning.md:686:### Micro — per-step 决策
docs/checkpoints/paper_planning.md:698:### Efficiency — 4-fold drop-in property
docs/checkpoints/paper_planning.md:702:| **3a** Token cost per step (input) | `condition_summary_v2.json` | P-SoM ≈ DOM (~3K both); SoM +image embedding tax. **4-fold drop-in (a) cost ≈ DOM ✅** |
docs/checkpoints/paper_planning.md:705:| **3d** B0 (API) vs B1 (local) deployment-class cost gap | `cost_per_mode.{json,md}` (FRESH 04-29) + `fig3d_cost_sr_frontier.png` | **B0 API ~$0.04/ep (Qwen3-VL-235B-A22B token cost)**; **B1 electricity-equivalent ~$0.0004/ep** (DGX Spark `avg_total_energy_kwh × $0.12/kWh` UK industrial rate). **Ratio ~100×** (red 98× / cls 105×) — **deployment-class gap, NOT capability/parameter ratio**. ⚠️ §103 / §3-legacy "30×" claim **superseded** by FRESH data. Paper presents both classes side-by-side, not a single multiplier. |
docs/checkpoints/paper_planning.md:709:### Cross-dimension Mechanism Chain（每个 axis 在哪些 dimension 上 first-order）
docs/checkpoints/paper_planning.md:711:| Axis | Outcome dimension 贡献 | Macro dimension signature | Micro dimension signature | Efficiency dimension cost |
docs/checkpoints/paper_planning.md:715:| **Axis 3 (image)** | secondary (cls SoM 21.37% > P-SoM 14.53%, image 决定性 cls 上) | **cls 5/8 dominant** (finish h=+0.57 medium-effect 最强信号); red 3/8 dominant (efficiency cluster) | image 加上 = URL Jaccard 0.46-0.60 minor change | **+700-1100 image tokens** (Efficiency 3a 主要 cost source) |
docs/checkpoints/paper_planning.md:720:### Evidence chain — paper claims → dimension support
docs/checkpoints/paper_planning.md:722:每个 paper claim 直接 cite dimension + 数字：
docs/checkpoints/paper_planning.md:724:| Paper claim | Dimension support |
docs/checkpoints/paper_planning.md:727:| **C2**: 4-fold drop-in property (cost / latency / signal / drop-one) | (a) Efficiency 3a, (b) Efficiency 3c, (c) Outcome 0g, (d) Outcome 0c |
docs/checkpoints/paper_planning.md:728:| **C3**: 3-axis hierarchical theory | Macro 1b (cascade decomposition), Micro (axis-by-axis micro), Cross-dimension table |
docs/checkpoints/paper_planning.md:735:### Mechanism chain — 三个机制阶段
docs/checkpoints/paper_planning.md:751:### Honest framing (avoid over-claim)
docs/checkpoints/paper_planning.md:761:### Legacy index (原 10 条 finding 映射)
docs/checkpoints/paper_planning.md:764:`B0_phantom_*` completed runs became `B0_phantom_som_*`, and completed `B0_phantom_dom_*` runs became `B0_phantom_text_*`. Internal mode IDs and condition dirs remain unchanged (`phantom_dom` / `phase1_phantom_dom_router_0`, `phantom_som` / `phase1_phantom_som_router_0`) for backward compatibility with recorded JSONL.
docs/checkpoints/paper_planning.md:772:| 5 P-SoM cost ≈ DOM cost | **3a** (4-fold drop-in (a)) |
docs/checkpoints/paper_planning.md:774:| 7 B0 vs B1 cost gap | **3d** (修正 04-29: ~100× deployment-class gap, NOT 30× — see `cost_per_mode.md`) |
docs/checkpoints/paper_planning.md:783:| **N2**: Tier 1 hook macro: red 4/8 cells fully independent / cls 1/8 (cls 主要 DOM-like) | **Macro 1a** |
docs/checkpoints/paper_planning.md:791:### Evidence vs Explanation: framework 的真实定位（2026-04-29 反思）
docs/checkpoints/paper_planning.md:795:#### 4-dimension = Evidence dimensions（paper Section 4）
docs/checkpoints/paper_planning.md:798:- Outcome: 哪些 task 成功（SR / oracle / Jaccard / category / overlap / AUROC）
docs/checkpoints/paper_planning.md:801:- Efficiency: 资源 footprint（cost / latency / carbon）
docs/checkpoints/paper_planning.md:803:四个 **正交 dimensions**（不是 hierarchical layers），从宏观 outcome 到微观 decision。Paper Section 4 是 evidence catalog，每个 sub-finding 引用一个 dimension 的数据 + figure。
docs/checkpoints/paper_planning.md:805:#### LLM mechanism = Explanation layer（paper Section 5）
docs/checkpoints/paper_planning.md:827:#### Axis decomposition（diamond 完整后的 final form）
docs/checkpoints/paper_planning.md:838:P-prompt 是必需的，因为它是 **axis 2 在 AXTree-text context 下的唯一测量点**。如果 interaction term ≈ 0 → paper 写 "axis additive, independent first-order"；如果 interaction term ≠ 0 → honest disclose "axis 1 effect is modulated by prompt context"。任一 verdict 都比 cascade-only 多一个 quantitative claim。
docs/checkpoints/paper_planning.md:840:#### Framework 的 future-data 弹性
docs/checkpoints/paper_planning.md:853:`make analyze-layered` 是 idempotent 的——新数据 commit 后跑一遍 `layered_status.py` 自动 regenerate `layered_evidence_status.md` + 所有 figures。CLI alias 保留 (`analyze-layered`, `layered_status`, `layered_evidence_status.md`) 是 backward compat — paper-facing 命名是 4-dimension。
docs/checkpoints/paper_planning.md:855:#### Caveats / honest framing
docs/checkpoints/paper_planning.md:858:- **命名约定**: paper-facing 用 "Outcome / Macro / Micro / Efficiency" (4 orthogonal dimensions). Sub-codes (0a / 1c / 2a / 3d) 保留作 figure-internal anchors. Code-level CLI 保留 "layered_*" 别名 (Makefile target / `layered_status.py` / `layered_evidence_status.md`) 作 backward compat
docs/checkpoints/paper_planning.md:864:### Mechanism Tier 1/2/3 escalation plan (Section 5 explanation methodology, 2026-04-29)
docs/checkpoints/paper_planning.md:868:#### Tier 1 — Behavioral mechanism (paper-ready now, B0+B1 data, no GPU work)
docs/checkpoints/paper_planning.md:881:#### Tier 2 — Mechanistic interpretability (B1-only, executes 实验笔记 §19 future-work)
docs/checkpoints/paper_planning.md:888:| **M2** B1 hidden state probing | layer L hidden state → probe "task will succeed"; PCA cosine gap (per §19) → AUROC vs logprob | `output_hidden_states=True` forward pass; PCA + LR | 🟡 blocked B1 GPU |
docs/checkpoints/paper_planning.md:889:| **M3** Token-level decision attribution | next-action token distribution per mode; quantify "axis 1 改 token-level decision prior" claim | forward inference, no training | 🟡 blocked B1 GPU |
docs/checkpoints/paper_planning.md:895:#### Tier 3 — Causal mechanistic intervention (heavy, may be future paper)
docs/checkpoints/paper_planning.md:899:| **H1** Activation patching | DOM forward pass at (layer L, step S) → patch hidden state into P-text run → does behavior become DOM-like? | causal scrubbing infrastructure | 🔴 blocked B1 GPU + 1-2 weeks impl |
docs/checkpoints/paper_planning.md:900:| **H2** Steering vectors | train PCA / linear probe to find "mode direction" in activation space; add steering vector at inference to induce mode-like behavior without obs/prompt swap | per §19 future work "L23 steering 修正 'know-but-cant-say'" | 🔴 blocked B1 GPU + advanced technique |
docs/checkpoints/paper_planning.md:901:| **H3** Attention head ablation | systematic zero-out specific heads; find "axis 1 head" / "axis 2 head" responsible for mode-specific behavior | head-by-head intervention scaffold | 🔴 heaviest, possible split paper |
docs/checkpoints/paper_planning.md:903:**Trigger condition**: 顶刊投稿 reviewer 要求 mechanistic 强化 OR 时间允许提前做。可能的 split: H1+H2 进 Section 5, H3 留 future work / paper 2.
docs/checkpoints/paper_planning.md:905:**Paper value**: causal claim, 比 correlation-based mechanism (Tier 2) 更强. ACL/NeurIPS mechanistic interpretability track 期望.
docs/checkpoints/paper_planning.md:907:#### 总体 Section 5 mechanism narrative cascade
docs/checkpoints/paper_planning.md:913:  Tier 3 causal (H1-H3)      ← Section 5 顶刊 differentiator, optional
docs/checkpoints/paper_planning.md:920:## §4 Paper Section Status (2026-04-29, 8 sections final scope; 2026-05-04 update with §21 cross-references)
docs/checkpoints/paper_planning.md:924:| 1 Intro | ✅ 已写 (786w + 4-fold drop-in framing + conservative framing) | done `62c1380` `ef29add` | **Pending advisor sync 5/5**: substitution-gradient framing rewrite (§21.5 candidate prose ~370w with Magma+ScribeAgent same-Qwen-base differentiator) | §21.5 hook prose, §21.2 industry precedent stack |
docs/checkpoints/paper_planning.md:927:| 4 Empirical Findings | 🟡 80% (figures FRESH ✅ + B0 5-mode FRESH, prose 待 update) | data ready | codex #11 fresh prose (~30K) | §21.6 cost/latency anchors (241K vs 47-140K tokens) for §1/§6 quantitative positioning |
docs/checkpoints/paper_planning.md:929:| **6 Routing (Tier 1+2)** ⭐ NEW | 🟡 40% (signal AUROC ≥ baseline `9d7e99f`, infra scaffold) | scaffold ready | Tier 1 prototype (~3 天) + Tier 2 first-step trigger (~7-10 天) | §21.5 CoAct-1 OSWorld 60.76% task-class routing precedent; §21.6 cost anchors |
docs/checkpoints/paper_planning.md:935:> **2026-05-04 §21 alignment audit**: 8 sections 全 cover 现 §21 9-cell taxonomy 内容, 但 explicit cross-reference 没贯通。advisor sync 5/5 lock 后 paper §1 hook 用 §21.5 prose; codex #11/#13 prose 写作时 cross-reference §21 industry precedents + counter-evidence stack。
docs/checkpoints/paper_planning.md:937:### Section 6 Routing — 详细 outline
docs/checkpoints/paper_planning.md:945:  - Target: max adjusted SR / cost-aware / Pareto
docs/checkpoints/paper_planning.md:947:6.2 Tier 1 — task-level oracle router (offline supervised)
docs/checkpoints/paper_planning.md:951:  - Result: routing pool oracle bound vs learned router gap
docs/checkpoints/paper_planning.md:953:6.3 Tier 2 — first-step-trigger router (online cascade)
docs/checkpoints/paper_planning.md:962:  - paper claim: "router trained on baseline 可 directly extend to Phantom"
docs/checkpoints/paper_planning.md:966:  - Fig B: Cumulative SR vs Budget curve ⭐ (cost-aware 顶刊套路)
docs/checkpoints/paper_planning.md:971:### Section 8 Discussion — 详细 outline (含 sustainability + green AI)
docs/checkpoints/paper_planning.md:977:  (c) Signal AUROC ≥ baseline (router infra drop-in)
docs/checkpoints/paper_planning.md:978:  (d) Drop-one oracle 1.7-3.3pp
docs/checkpoints/paper_planning.md:989:  - Multi-metric Pareto: cost + latency + carbon 三向 drop-in
docs/checkpoints/paper_planning.md:994:  - Tier 3 online learning router 留 future work
docs/checkpoints/paper_planning.md:1000:## §5 Final Scope + 顶刊概率
docs/checkpoints/paper_planning.md:1002:### Final scope (paper 完整版)
docs/checkpoints/paper_planning.md:1011:+ Router:  Tier 1+2 (oracle + first-step trigger), 实际 deploy on agent
docs/checkpoints/paper_planning.md:1012:+ Multi-metric: cost / P95 latency / carbon (B1 measured + B0 estimate)
docs/checkpoints/paper_planning.md:1015:### 顶刊概率 — conditional tree on framing rule R1-R5
docs/checkpoints/paper_planning.md:1017:> **Update 2026-05-04**: 旧 §5 是 unconditional 单点估计 (4/27 写). 5/3 pre-registration reframe 后, paper hook 是 **data-conditional R1-R5** (见 §1 + `preregistration.md`), 概率也应该按 framing-rule 分支条件化, 不再是单点数字.
docs/checkpoints/paper_planning.md:1024:> - Phase A 4-cluster bug fix + 5-tier audit + 16-cell rerun 计划 (Risk 1 mitigation 落地)
docs/checkpoints/paper_planning.md:1027:#### 条件概率 (conditional on R-rule outcome 落在哪档)
docs/checkpoints/paper_planning.md:1034:| **R4 (weak)** | Hero partial fail (e.g. cost/latency 不 hold) | 15-30% | 10-25% | 20-35% | 55-70% | 50-65% | 45-60% | 65-78% | ~85% |
docs/checkpoints/paper_planning.md:1037:#### 解读
docs/checkpoints/paper_planning.md:1041:- **R4 是真危险区** — MLSys + TMLR 仍能保底, 但 top-tier <30%. 这就是为啥 16-cell rerun (cost/latency 重测) 必跑.
docs/checkpoints/paper_planning.md:1045:#### 数据未确认前的实际期待 (advisor sync 用)
docs/checkpoints/paper_planning.md:1049:#### Caveats (-)
docs/checkpoints/paper_planning.md:1055:### Multi-metric + Green AI axis 加成的 paper-level 价值
docs/checkpoints/paper_planning.md:1059:3. **三向 drop-in** (cost+latency+carbon) narrative 立体
docs/checkpoints/paper_planning.md:1066:## §6 Critical Risks + Mitigation (4 risks, 决定接收 vs reject)
docs/checkpoints/paper_planning.md:1068:### Risk 1: Execution quality（顶刊成败 #1 因素 ⚠️⚠️⚠️）
docs/checkpoints/paper_planning.md:1081:### Risk 2: Story discipline ⚠️⚠️
docs/checkpoints/paper_planning.md:1085:**Single narrative**: "Phantom-SoM is hidden routing arm + we explain why + we route on it + here's the cost saving".
docs/checkpoints/paper_planning.md:1089:### Risk 3: Router design ⚠️⚠️
docs/checkpoints/paper_planning.md:1094:- **Tier 1 (must-have)**: Oracle router — task feature → best mode lookup, train/test split
docs/checkpoints/paper_planning.md:1095:- **Tier 2 (great-to-have)**: First-step-trigger router — 看 step 1 obs 决定 mode, no test leak
docs/checkpoints/paper_planning.md:1096:- **Tier 3 (stretch)**: Online learning router — mid-trajectory escalation
docs/checkpoints/paper_planning.md:1102:**Minimum viable router** (start ~3 天 prototype):
docs/checkpoints/paper_planning.md:1111:### Risk 4: Negative results 必须诚实报告 ⚠️
docs/checkpoints/paper_planning.md:1115:**Mitigation**: 诚实报告反而强化 mechanism claim ("effect 是 task-type/capability bound, 不是 universal").
docs/checkpoints/paper_planning.md:1117:### Risk 5: B0 vs B1 reproducibility 不对称 (新增 2026-04-30) ⚠️
docs/checkpoints/paper_planning.md:1133:**Cost saved by NOT pursuing this further**: replication study at full 16-cell scale would cost ~$60-200; instead cheap 5-call probe ($0.005) gave us decisive characterization. Paper Section 4 disclosure paragraph is the deliverable.
docs/checkpoints/paper_planning.md:1137:## §7 Investment Cascade Plan
docs/checkpoints/paper_planning.md:1162:期望出版 venue 链 ~99% (5 站 5 model deployed-router scope 没法被全拒).
docs/checkpoints/paper_planning.md:1166:## §8 Router Design (Tier 1+2)
docs/checkpoints/paper_planning.md:1168:### 5 个关键设计决策点 (each requires ablation)
docs/checkpoints/paper_planning.md:1173:| **Target** | max SR / SR-per-cost / Pareto / budget-constrained | multi-obj weight 选 |
docs/checkpoints/paper_planning.md:1174:| **Granularity** | task-level / step-level / confidence-triggered | step-level 重跑 2x cost |
docs/checkpoints/paper_planning.md:1175:| **Cascade** | 单 router / B1→B0 escalation / rule+ML hybrid | escalation 实验代价大 |
docs/checkpoints/paper_planning.md:1176:| **Baseline** | random / best-single-mode / oracle / rule-based | best-single-mode 是 hardest baseline |
docs/checkpoints/paper_planning.md:1178:### Realistic timeline (paper 真正最值钱的工作量)
docs/checkpoints/paper_planning.md:1181:Tier 1 (task-level oracle): ~5-7 天
docs/checkpoints/paper_planning.md:1194:### Routing infra 现状 (paper 1 直接用)
docs/checkpoints/paper_planning.md:1199:- Router scaffold: `p79/experiment/router.py::RuleBasedRouter`
docs/checkpoints/paper_planning.md:1200:- **Phantom modes 直接复用 baseline signal infra** (drop-in routing claim 第 4 fold)
docs/checkpoints/paper_planning.md:1204:## §9 Advisor Align Checklist
docs/checkpoints/paper_planning.md:1206:### Meeting #1 (~Week 3, cls+red+shopping done)
docs/checkpoints/paper_planning.md:1212:| 单 paper vs 双 paper | (a) Integrated (Paper 1 含 router) / (b) Split (Paper 2 router) | **(a) Integrated** (毕设决策) | publication count vs paper depth |
docs/checkpoints/paper_planning.md:1216:### Meeting #2 (~Week 6-7, WA + Claude done)
docs/checkpoints/paper_planning.md:1224:### 关键 strategic 问题 (advisor align 时主动问)
docs/checkpoints/paper_planning.md:1234:## §10 Visualization Plan (cascade router viz)
docs/checkpoints/paper_planning.md:1236:**单纯 2D cost-SR Pareto 不够 striking**. 推荐 4-figure stack:
docs/checkpoints/paper_planning.md:1240:| **Fig A: 3-panel multi-metric Pareto** | 主 figure, fig7 升级 | 3 panel: cost-SR + latency-SR + CO2-SR |
docs/checkpoints/paper_planning.md:1241:| **Fig B: Cumulative SR vs Budget curve** ⭐ | 最 striking, cost-aware 顶刊套路 | x=budget per task, y=cumulative SR; lines: random/best-single/rule/learned/oracle |
docs/checkpoints/paper_planning.md:1242:| **Fig C: Routing decision Sankey** | Section 6 解释 router 学到什么 | task category → routed mode → outcome |
docs/checkpoints/paper_planning.md:1243:| **Fig D: Per-task savings histogram** | Appendix supplementary | distribution: cost saved by routing per task |
docs/checkpoints/paper_planning.md:1248:x: cumulative cost budget per task ($)
docs/checkpoints/paper_planning.md:1253:  --- rule-based router (handcrafted)
docs/checkpoints/paper_planning.md:1254:  ▬▬▬ learned ML router (ours) ⭐
docs/checkpoints/paper_planning.md:1255:  ─── oracle router (upper bound)
docs/checkpoints/paper_planning.md:1256:fill area: ours vs best-single-mode gap; ours vs oracle gap
docs/checkpoints/paper_planning.md:1259:直观论证: 在 $0.04 budget per task → 我们 router 25% SR vs best-single-mode 21%; oracle 边界 ~30%, learned router 缩小 60% gap.
docs/checkpoints/paper_planning.md:1267:## §11 Cost / Latency / Carbon Multi-metric Plan
docs/checkpoints/paper_planning.md:1269:### 已有数据状况 (per `condition_summary_v2.json`)
docs/checkpoints/paper_planning.md:1276:### Carbon tracker 现状 (`p79/experiment/energy_tracker.py`)
docs/checkpoints/paper_planning.md:1282:### Tier 化 paper 利用
docs/checkpoints/paper_planning.md:1286:| **Tier 1 (主体)** | adjusted SR, drop-one oracle, cost/task | Section 1 hook + Section 4 main + fig7 |
docs/checkpoints/paper_planning.md:1287:| **Tier 2 (主体辅)** | P95 latency, CO2/task | Section 4 cost-aware table + Section 7 sustainability |
docs/checkpoints/paper_planning.md:1288:| **Tier 3 (附录)** | wasted cost, energy kWh, cost_efficiency_ratio | supplementary |
docs/checkpoints/paper_planning.md:1290:### Striking findings 已 measured (paper 直接 cite)
docs/checkpoints/paper_planning.md:1295:4. **Phantom-SoM cls cost ≈ DOM** + latency 4× 改进 = triple win
docs/checkpoints/paper_planning.md:1297:### Regional Carbon Sensitivity (fig9 already done, codex `d3dfc8f`)
docs/checkpoints/paper_planning.md:1306:## §12 References / Doc Map
docs/checkpoints/paper_planning.md:1308:### Paper drafts (final prose, `docs/analysis/paper_drafts/`)
docs/checkpoints/paper_planning.md:1320:### paper.bib
docs/checkpoints/paper_planning.md:1324:### Codex analyses (`docs/analysis/phantom_paper/`)
docs/checkpoints/paper_planning.md:1333:### Other analyses
docs/checkpoints/paper_planning.md:1338:### Figures (`results/phantom_paper/figures/`, all FRESH 04-28)
docs/checkpoints/paper_planning.md:1342:fig2 drop-one oracle (2x2)
docs/checkpoints/paper_planning.md:1347:fig7 cost-SR Pareto + deployment callouts
docs/checkpoints/paper_planning.md:1352:### 实验笔记 § index (key findings)
docs/checkpoints/paper_planning.md:1363:### Recent key commits (~04-27 / 04-28)
docs/checkpoints/paper_planning.md:1368:139afb0  router framing fix (paper 1 not paper 2)
docs/checkpoints/paper_planning.md:1373:00124e4  3-axis hierarchical theory framework
docs/checkpoints/paper_planning.md:1379:48db047  Phantom-SoM cost ≈ DOM
docs/checkpoints/paper_planning.md:1380:93e413f  3-layer cost decomposition
docs/checkpoints/paper_planning.md:1388:## §13 Pending TODO (paper-strategic, not action ledger)
docs/checkpoints/paper_planning.md:1390:### A. Codex prose tasks (跟踪 in next_steps §4 codex queue)
docs/checkpoints/paper_planning.md:1400:### B. Data analysis pipeline (Python scripts, not codex tokens)
docs/checkpoints/paper_planning.md:1402:- [x] **统计显著性测试** ✅ done 04-28 — `fig0c_drop_one_oracle.py` 加 `bootstrap_drop_one_ci()` (1000 resample × 4 panel)，error bars + `fig0c_drop_one_bootstrap_ci.csv` 12 rows
docs/checkpoints/paper_planning.md:1406:  - Outputs: `results/phantom_paper/auroc_cross_condition.csv` (188 rows × 5 modes × 4 cells) + `_summary.md` (top-1 per cell, Section 6 claim 证据)
docs/checkpoints/paper_planning.md:1407:  - Section 6 "AUROC ≥ baseline" claim 部分支持: B0 red P-text 0.793 highest; B0 cls P-text 0.737 ≥ SoM 0.709 baseline; B1 cells 待 chain done
docs/checkpoints/paper_planning.md:1409:  - Outputs: `results/phantom_paper/phantom_lift.{csv,md}` — 3-mode → 5-mode oracle ceiling lift + bootstrap CI + per-phantom decomposition
docs/checkpoints/paper_planning.md:1410:  - **Paper Section 1/4 hook 主 evidence**: B0 cls **+4.70pp [2.14, 7.69]** ✅, B0 red **+5.24pp [2.38, 8.11]** ✅ (CI 排除 0)
docs/checkpoints/paper_planning.md:1413:- [ ] **Multi-metric Pareto pipeline** (cost + latency + carbon)
docs/checkpoints/paper_planning.md:1414:  - Section 8 sustainability prose 前置；fig9 已有 carbon B1 only, 需 cost/latency 三向 join
docs/checkpoints/paper_planning.md:1416:  - Implementation: extend `scripts/analysis/figures/fig3d_cost_sr_frontier.py`
docs/checkpoints/paper_planning.md:1417:- [ ] **每 task 特征提取** (Section 6 Tier 1 oracle router 前置)
docs/checkpoints/paper_planning.md:1432:- [x] **H3 structural test (phantom space 2-axis empirical evidence)** ✅ done 2026-05-03 (T0a) — `aggregate_phantom_lift.py` H3 family with bootstrap CI on |arm ∖ P-SoM| unique-count + per-axis Holm correction
docs/checkpoints/paper_planning.md:1433:  - Output: `phantom_lift.md` H3 Structural section
docs/checkpoints/paper_planning.md:1434:  - Tests phantom space is multi-region 2D not collapsed point (paper hook structural claim)
docs/checkpoints/paper_planning.md:1440:### C. Paper end-stage tasks (Week 8+)
docs/checkpoints/paper_planning.md:1454:## §14 Reviewer Attack Anticipation + Pre-Rebuttal
docs/checkpoints/paper_planning.md:1462:| **Single model family (Qwen)** | "Effect Qwen-specific?" | + Claude Opus 4.7 cross-model after advisor align (~$70). B0 (235B) + B1 (4B) shows capability-dependent shift (+50/+33pp cross-site, §101.九 lazy minimization) | §2 capability layer + cross_site_pattern_consolidation.md |
docs/checkpoints/paper_planning.md:1464:| **Effect size small (drop-one 1.7-3.3pp)** | "Statistically marginal" | (i) Pre-registered Hero (P-SoM) requires pooled magnitude ≥ 1.0pp + TOST equivalence at δ=1.0pp rejected. (ii) P-text/P-prompt are framed as **structural ablation evidence** (low-threshold non-overlap proves phantom space is multi-region 2D), NOT as deployment routing arms — so deployment magnitude bar doesn't apply to them. (iii) Holm-Bonferroni multi-comparison correction applied per pre-registered family. | §1 paper hook (data-conditional R1-R5) + `preregistration.md` H1+H3 + `phantom_lift.md` Holm/TOST cols |
docs/checkpoints/paper_planning.md:1465:| **Post-hoc hypothesis cherry-picking** ⭐ NEW pre-rebuttal | "你 H-list 是数据进来后 fit 的" | Pre-registration locked before 16-cell rerun via Git SHA + advisor email witness + OSF DOI (paper-time public). Multi-comparison family declared explicitly. Exploratory analyses (H4/H5/H6) marked "post-hoc" in paper prose with explicit non-gating disclosure. Framing decision rule R1-R5 maps data outcome to hook framing transparently — reviewer can verify framing-to-data mapping is deterministic, not chosen post-hoc. | `docs/checkpoints/pre_run/preregistration.md` + `EVIDENCE_LAYER_AUDIT.md` §2 |
docs/checkpoints/paper_planning.md:1466:| **Latency claim cherry-picked** | "Just one P95 measurement" | §100 SoM probe ground truth (5 imgs × 3 mode × 2 model = 30 cells measured). cls SoM 74s vs Phantom 18s p95 = 4× slower. Across all conditions consistent | §11 + 实验笔记 §100 |
docs/checkpoints/paper_planning.md:1468:| **Router contribution toy** | "Tier 1 oracle is overfit" | Tier 1 train/test split, baseline 对比 (random, best-single-mode, rule-based, oracle, learned). Tier 2 first-step trigger no test leakage | §8 + Section 6 outline §4.6 |
docs/checkpoints/paper_planning.md:1469:| **No production deployment** | "Drop-in claim hypothetical" | 4-fold drop-in property: code-level verified (`som.py::_extract_text_marks` line 24 regex); routing signal AUROC ≥ baseline (5/5 `overall_usable=True`); 实证 cost+latency+CO2 measured | §1 + §3 finding #5 #9 |
docs/checkpoints/paper_planning.md:1471:| **Mechanism not novel** | "Each axis has prior literature" | Contribution = systematic decomposition + web-agent multi-step setting + drop-in deployment claim. NOT new LLM mechanism. Paper §5 framing 已 acknowledge | §2 paper contribution position |
docs/checkpoints/paper_planning.md:1472:| **Overfit to VWA visual specifics** | "Effect won't generalize to WA" | §103 falsifiable prediction: WA Phantom-SoM 5-mode oracle gain. WA pilot ≤50 task verify Jaccard ≤0.5 universal vs >0.7 VWA-specific | §103 generalization prediction; pending data |
docs/checkpoints/paper_planning.md:1481:## §15 Prior Work Comparison Table
docs/checkpoints/paper_planning.md:1489:| **Cost-aware Pareto** | ❌ | ❌ | ❌ | ✅ token cost | ✅ model cost | ✅ **multi-metric** (cost+latency+carbon) ⭐ |
docs/checkpoints/paper_planning.md:1493:| **Drop-in deployment** | ❌ | ❌ | ❌ | partial | partial | ✅ **4-fold property** (cost/latency/signal/oracle) ⭐ |
docs/checkpoints/paper_planning.md:1498:**Closest prior pairing**: FocusAgent (text 压缩, hierarchy 保持) + Yang 2023 SoM (visual marks). 本工作 = unprecedented synthesis + drop-in deployment claim + multi-metric Pareto + green AI differentiator.
docs/checkpoints/paper_planning.md:1504:## §16 Authorship + Advisor Roles + First-Paper Strategy
docs/checkpoints/paper_planning.md:1506:### 毕设 paper authorship plan (TBD with advisor align meeting #1)
docs/checkpoints/paper_planning.md:1517:### Advisor / collaborator roles
docs/checkpoints/paper_planning.md:1525:### Personal context (毕设 backdrop, 本 paper 是 first paper)
docs/checkpoints/paper_planning.md:1528:- First paper, 经历: paper trajectory 从 "magical noise 怀疑" 到 "4-fold drop-in deployment claim"
docs/checkpoints/paper_planning.md:1529:- 多次 critique-driven theory refinement (4 rounds: prompt-only / visual-hijack-only / image-over-text / SoM density) — paper integrity discipline 体现
docs/checkpoints/paper_planning.md:1532:### First-paper psychology + strategic advice
docs/checkpoints/paper_planning.md:1551:### Acknowledgments draft (paper end-stage)
docs/checkpoints/paper_planning.md:1564:## §17 Pre-Submission Checklist (~Week 10-12 paper 终稿前)
docs/checkpoints/paper_planning.md:1566:### Content completeness
docs/checkpoints/paper_planning.md:1578:- [ ] Limitations section honest (no over-claim)
docs/checkpoints/paper_planning.md:1580:### Format / Style
docs/checkpoints/paper_planning.md:1589:### Reproducibility
docs/checkpoints/paper_planning.md:1597:### Authorship + Submission
docs/checkpoints/paper_planning.md:1605:### Pre-rebuttal preparedness
docs/checkpoints/paper_planning.md:1614:## §18 Watchdog Protocol + Paper-Grade Execution Discipline
docs/checkpoints/paper_planning.md:1619:### 6-layer Defense in Depth (per `experiment_watchdog.py`)
docs/checkpoints/paper_planning.md:1640:### Magento history (3 复发 + final fix)
docs/checkpoints/paper_planning.md:1647:                     → 3-layer 持久化 (magento_baseurl_fix.sh + start_vwa_docker.sh
docs/checkpoints/paper_planning.md:1648:                        hook + reset_shopping.sh remove hardcode localhost)
docs/checkpoints/paper_planning.md:1655:                       PowerShell hook 持久化 (reset 后 auto-disable FPC)
docs/checkpoints/paper_planning.md:1658:### Paper-grade clean re-run protocol
docs/checkpoints/paper_planning.md:1681:### Paper integrity 论证 (Section 4 / supplementary)
docs/checkpoints/paper_planning.md:1688:- **Cross-mode comparison preserved**: 5 modes 受同一 protocol, drop-one oracle / Jaccard / cost-SR Pareto 都不被 ~2% noise bias
docs/checkpoints/paper_planning.md:1689:- **Paper-grade discipline**: self-healing data pipeline, 6-layer defense in depth → reviewer 信任 paper data integrity
docs/checkpoints/paper_planning.md:1693:## §19 Decision Log (paper-strategic decisions audit trail)
docs/checkpoints/paper_planning.md:1697:| 2026-04-27 | Final scope: 6 sites × 3 models × 5 modes + deployed router + multi-metric + green AI | NeurIPS/顶刊 viable scope (paper_planning §5) | ✅ in plan |
docs/checkpoints/paper_planning.md:1699:| 2026-04-27 | Future paper 2 转向 Phase 3 modules (router 整合 paper 1) | 毕设决策, paper 1 含完整 contribution | ✅ in plan |
docs/checkpoints/paper_planning.md:1701:| 2026-04-27 | Paper hook 升级到 "drop-in deployment intervention" | Phantom-SoM cost ≈ DOM (regex filter), 4-fold property | ✅ commits 48db047 + ef29add |
docs/checkpoints/paper_planning.md:1708:| 2026-04-28 | 8 sections paper structure (含 Section 6 Routing 独立) | router 是 paper independent contribution, not Section 7 sub | ✅ commit 4ca9f66 |
docs/checkpoints/paper_planning.md:1709:| 2026-05-01 | Paper hook reframe: "P-SoM is hidden 4th routing arm" → "**phantom routing space (3 arms)** sharing 4-fold drop-in" | B0 reddit 6-mode oracle +7.14pp [3.81, 10.48] sig + 3 arms drop-one 全 sig (P-text +3.81 / P-SoM +3.33 / P-prompt +2.86) | ⏸️ provisional pending cls 6-mode + B1 phantom 数据 confirm (advisor sync Q3) |
docs/checkpoints/paper_planning.md:1713:| 2026-05-01 | Evidence vs Explanation layer 严格分离 (paper conceptual structure) | Evidence = 2D organize (4 测量 × 4 cross-X); Explanation = 1D zoom scale (Zoom 1-4); 不混 | ✅ paper_planning §3 顶部 + §2 retract list + 笔记 §108.6 |
docs/checkpoints/paper_planning.md:1714:| 2026-05-01 | 4 zoom scale of explanation layer | Zoom 1 (架构) / Zoom 2 (behavioral M1/M2) / Zoom 3 (named phenomena Mirage/Scaffold/Sclar) / Zoom 4 (model-internal Cross-modal flow/SteerMoE) | ✅ paper_planning §2 + 笔记 §108.6 |
docs/checkpoints/paper_planning.md:1718:| 2026-05-01 | Early-stop mechanism design decision: lean Option A (full cancel), pending advisor align | early-stop 是 cross-dimension systemic confound (不止 micro layer); Option A +$1300 全 cancel / B keep / C hybrid +$200 | ⏸️ advisor sync Q1 重写, lean A pending | 
docs/checkpoints/paper_planning.md:1719:| 2026-05-01 | 别扭 framework refinement (provisional) — reverse-explanation layer + capability-modulated discovery | (a) Cross-cell empirical validation 4 cells: B0 4/4 别扭 predictions confirmed, B1 cls prediction 4 reversed (small VLM single 别扭 negative aggregate); (b) drop-one direction reversal cross-capability (B0 P-text > P-SoM, B1 cls P-SoM > P-text) → 别扭 + Lazy Minimization 联合 framework; (c) compound 别扭 (P-prompt) 实证 negative aggregate (B0 reddit raw 10.48 < DOM 11.43) but positive complementarity (drop-one +2.86pp) — double-edged property | ⏸️ provisional, pending 16-cell rerun statistical commit + B1 reddit phantom 数据 |
docs/checkpoints/paper_planning.md:1721:| 2026-05-03 | H3 structural test = bootstrap CI on \|arm ∖ P-SoM\| unique-count > 0, K_h3=0.67, ≥2 task floor | Structural claim only requires non-emptiness of axis non-overlap, not directional dominance. McNemar tests asymmetry which is wrong test for H3. Bootstrap CI > 0 is correct. | ✅ `aggregate_phantom_lift.py` H3 family + `phantom_lift.md` H3 section |
docs/checkpoints/paper_planning.md:1722:| 2026-05-03 | Pre-registration commits locked: K_h1=0.75 / K_h3=0.67 / TOST δ=1.0pp / Phase A only main + Appendix D archived / Witness=Git+advisor email + OSF DOI | All 5 commits drafted in `preregistration.md` (status:draft); pending advisor sync to flip status:locked + record git SHA + advisor witness. | ⏸️ pending advisor sync |
docs/checkpoints/paper_planning.md:1723:| 2026-05-03 | Evidence layer + visualization audit infra (T0a-T0d done) | `aggregate_phantom_lift.py` Bonferroni/Holm/BH/TOST + H3 structural test cols; `aggregate_phantom_meta.py` DerSimonian-Laird random-effect; `fig_forest_drop_one.py` per-cell forest with Holm-sig markers; `fig_meta_forest.py` Hero+Ablation visual hierarchy; `fig_phantom_structure_venn.py` paper §1 centerpiece Venn; `make analysis [FAST=1]` end-to-end wired. | ✅ `docs/reference/EVIDENCE_LAYER_AUDIT.md` §3 T0 4/6 done |
docs/checkpoints/paper_planning.md:1724:| 2026-05-04 | Bulk archive all 27 manifest cells pre-advisor-sync (run_manifest.yaml) | Phase A 4-cluster fix (3c15cd7 4/30 15:35) makes pre-fix data not directly comparable to post-fix; cross-grade asymmetry from 5/1 + 5/4 post-fix solo runs would contaminate cross-mode comparisons (fix-effect ≠ mode-effect). All cells flipped to `grade: archived` until 16-cell rerun + advisor lock; figures preserved at last paper-grade-pre-bug-only state for 5/5 sync visual aid. | ✅ commit 8a9f595 |
docs/checkpoints/paper_planning.md:1725:| 2026-05-04 | §5 顶刊概率 → conditional tree on R1-R5 framing rule (was unconditional single-point) | 5/3 pre-registration reframe made paper hook data-conditional; probability estimates should follow same discipline. R1 (strongest, K_h1≥0.75 + K_h3≥0.67): top-tier 55-70% / cascade ~99%. R2 (Hero pass, single-axis structural): ~97% cascade. R3 (旧 §5 baseline, hero pass + structural fail): 35-50% top-tier / ~93%. R4 (hero partial fail, e.g. cost not hold): MLSys+TMLR 保底, top-tier <30%. R5 (hero fail): pivot. R2 是 advisor-sync realistic baseline expectation. | ✅ paper_planning §5 rewritten |
docs/checkpoints/paper_planning.md:1726:| 2026-05-04 | §21.5 + §1 hook 三层 novelty hierarchy (refines §109.17 binary reframe) | §109.17 (research-characterization angle) collapsed framework-tier into artifact-vs-characterization binary. User clarification: (a) **P-SoM specifically + 3-axis cube framework + image-axis isolation are paper-level framework contributions** — no industry deploys cube-center SoM-text-without-image combination; (b) industry deploys P-text/DOM-like artifacts **arbitrarily for token economy**, never compared P-text vs DOM, never characterized per-dimension routing behavior — artifact existence ≠ understanding. New three-tier: framework (cube + P-SoM, paper-novel) / artifact (industry has DOM/P-text/SoM analogs but deployed without characterization, NO P-SoM/P-prompt analog) / research (paper discovers + characterizes routing effects industry deployed-without-realizing). Reviewer-defense layered into 3 attack vectors. | ✅ paper_planning §21.5 + §1 hook one-liner |
docs/checkpoints/paper_planning.md:1727:| 2026-05-04 | New §22 Multi-Register Novelty Inventory (5/4 audit consolidation) | Audit prompt "现在的 novelty 还缺什么吗 — 审计下 结合其他文档/figure" 触发. 跨 paper drafts §1-§5 + 24 figures + 笔记 §1-§109 + EVIDENCE_LAYER_AUDIT + preregistration cross-check. 出 5-register layered framework (Theory/Concept · Method/Process · App/Impact · Survey/Position · Future-trajectory) × ~38 items. 用户自列 6 dimensions covered + 4 new dimensions added (J phantom space generalizability / AA routing signal portfolio / HH site-class adaptive routing primitive / LL execution-discipline standalone short-paper). Audit gaps section list 5 main gaps (paper §1 prose stuck at 4/29 framing, figures not referenced in §1, 笔记 finding §32/§72/§94/§100 未 elevate, EVIDENCE_LAYER_AUDIT 12+ pending, industry analog 缺位 not explicit). §1 hook 6-contribution rewrite candidate drafted. Advisor 5/5 sync top-5 / Tier-2 / polish priority 列出. Post-sync action items 8 项. | ✅ paper_planning §22 (147 行 inventory) |
docs/checkpoints/paper_planning.md:1728:| 2026-05-04 | preregistration.md 5/4 audit expansion: H7-H8 router family + 6 §4 lock entries + §5 exploratory expansion | User audit prompt "preregistration.md 还需要锁 Held-out router claim / router baselines train-validation-test split / routing signals / mode definition 这些吗". Claude evaluated 4 items + added 2 (failure-mode rubric / N_cells). Added: §2 H7 Tier 1 oracle router lift family + H8 Tier 2 first-step trigger router family (PRIMARY-vs-SECONDARY pending advisor lock); §3 ROUTER family multi-comparison declaration; §4 6 new lock entries (mode operational defs / routing signal universe / train-test split protocol / failure-mode rubric / N_cells final scope / best-single-mode baseline anchor); §5 exploratory expanded (best-signal-per-mode / router feature engineering / cross-site asymmetry framing / phantom space generalizability); §6 witness 5 commits → 8 commits expansion. ADVISOR_SYNC §0 + §2 同步 5 件 → 8 件 (added (6) N_cells / (7) Router paper-1-vs-2 / (8) Split protocol). Status: draft pending advisor 5/5 sync lock all 8 commits. | ✅ preregistration.md §2-§6 + ADVISOR_SYNC §0 + §2 |
docs/checkpoints/paper_planning.md:1732:## §20 Meta — Doc Update Workflow (when X happens, update which docs)
docs/checkpoints/paper_planning.md:1736:### A. 新 condition 数据 (e.g. B1 phantom_som cls done)
docs/checkpoints/paper_planning.md:1756:### B. 新 figure (e.g. fig10 cumulative SR vs budget)
docs/checkpoints/paper_planning.md:1766:### B'. 新 cross-condition aggregator (e.g. paired permutation table)
docs/checkpoints/paper_planning.md:1775:### C. 新 codex analysis (e.g. trajectory diff diag)
docs/checkpoints/paper_planning.md:1781:🟡 paper_planning §2 theory framework: if mechanism discovery (e.g. axis refinement)
docs/checkpoints/paper_planning.md:1786:### D. 新 paper drafts (e.g. Section 5 prose done by codex)
docs/checkpoints/paper_planning.md:1796:### E. 新 decision (e.g. advisor align meeting #1 outcome)
docs/checkpoints/paper_planning.md:1807:### F. 新 infra fix (e.g. another bug fix or watchdog upgrade)
docs/checkpoints/paper_planning.md:1812:🟡 paper_planning §18 watchdog protocol + execution discipline: update 6-layer description
docs/checkpoints/paper_planning.md:1816:### G. 新 finding (e.g. unexpected mechanism observation)
docs/checkpoints/paper_planning.md:1821:🟡 paper_planning §2 theory framework: if framework refinement needed
docs/checkpoints/paper_planning.md:1823:🟡 next_steps §0: if changes paper hook
docs/checkpoints/paper_planning.md:1826:### H. 新 reviewer attack scenario
docs/checkpoints/paper_planning.md:1833:### I. 新 paper section prose done
docs/checkpoints/paper_planning.md:1843:### General principle
docs/checkpoints/paper_planning.md:1851:### Quick mental check before update
docs/checkpoints/paper_planning.md:1864:## §21 Environment-Agent Intervention Taxonomy (整合 2026-05-04, 笔记 §1-§108 audit, **2026-05-04 deepest-evening 4-round epistemic upgrade**)
docs/checkpoints/paper_planning.md:1873:> 1. **§109.16** Format-axis vs scope-axis distinction (verified by reading processors.py:513-619, retract earlier "interactive-only filter" over-claim)
docs/checkpoints/paper_planning.md:1874:> 2. **§109.17** Research-characterization vs artifact-existence epistemic distinction (paper §1 hook reframe — industry deploys for economy, paper characterizes for behavior; 4 phantom corners equal-novel as research cells)
docs/checkpoints/paper_planning.md:1878:### §21.1 Framework — 3 × 3 Taxonomy
docs/checkpoints/paper_planning.md:1880:**3 个 intervention nature** × **3 个 intervention layer** = 9 cells:
docs/checkpoints/paper_planning.md:1882:#### Spectrum dimension (intervention 性质)
docs/checkpoints/paper_planning.md:1890:#### Layer dimension (intervention 位置)
docs/checkpoints/paper_planning.md:1895:| **L2 Agent-pipeline** | agent script preprocessing / postprocessing layer (改 agent 端 perception 处理, 不动 web) | 中: agent-only 改动; 但 fragile (web 一变就要重写) |
docs/checkpoints/paper_planning.md:1898:#### 9-Cell 一句话总结表
docs/checkpoints/paper_planning.md:1904:| **L3 Agent-compute** | n/a (compute 不能 fix bug) | **Paper hook 4-tier sub-gradient** (重要 distinction):<br/>• Pretraining-time: **Magma** (MS Feb 2025, SoM+ToM grounding 进 weights)<br/>• Offline exploration + RAG retrieve: **AppAgent-v2** (Tencent, agent self-generates text doc, deploy-time RAG)<br/>• Inference-time substitution: **Phantom routing space** ⭐ paper-1 main hook (P-text/P-prompt/P-SoM, no pretraining, no RAG, no offline phase)<br/>• Pure visual VLM: UI-TARS, CogAgent, Magma — opposite end (skip text substitution) | n/a (compute 不能 add absent signal) |
docs/checkpoints/paper_planning.md:1906:### §21.2 Done items (已做 work, mapped to cells)
docs/checkpoints/paper_planning.md:1908:#### (i) × L1 — Server-side bug fix (~6 entries)
docs/checkpoints/paper_planning.md:1919:#### (i) × L2 — Agent-pipeline bug fix (~28 entries)
docs/checkpoints/paper_planning.md:1923:| **Phase A 4-cluster (paper-grade)** | §107 (commit `3c15cd7`): C1 dispatch / C2 page_changed split / C3 fuzzy cycle hash min_reps=5 / C4 RNG seeding+T=0 | 主 paper-grade rerun trigger |
docs/checkpoints/paper_planning.md:1931:#### (ii) × L1 — Server-side affordance synthesis: **0 done** (paper 2 future direction)
docs/checkpoints/paper_planning.md:1942:#### (ii) × L2 — Agent-pipeline affordance synthesis (~9 entries)
docs/checkpoints/paper_planning.md:1953:| §93 | 分析管线 timing/intent/cost/visualization 增强 (eval-side affordance) | `validate_run` 22→27 checks + `reason_diagnostics` 16 cols |
docs/checkpoints/paper_planning.md:1971:| **Tarsier** (Reworkd v0.6.0 Jun 2024) ⭐ closest research precedent | Playwright | Typed SoM brackets `[#23]` input / `[@23]` link / `[$23]` button + text "ASCII art" for non-vision LLM. **Optional `[ID]` plain text mode** (not strictly interactive-only) | (not disclosed) | Internal benchmark claims **"unimodal text beats GPT-4V + Tarsier-Screenshot by 10-20%"** — direct industry analog of phantom routing thesis but **no peer-reviewed systematic characterization**, paper §1/§2 explicit cite |
docs/checkpoints/paper_planning.md:1978:| **Anchor Browser** ($6M seed, 2025; Cloudflare/Coinbase/Groq partners) | Custom infrastructure layer | MCP integration (Claude Desktop / Cursor / Groq agent platform) | (infrastructure-level, not affordance-format-specific) | Production scale: deploys "millions of browser agents"; Groq partnership Jan 2026 |
docs/checkpoints/paper_planning.md:1983:**Convergent industry design point**: agent-browser + Playwright MCP both report ~200-400 tokens per snapshot via accessibility-tree extraction + format trimming + ref-based flat list — **10-15× compression vs full raw HTML/DOM 3000-5000**. Note: 10-15× is vs raw HTML, not vs accessibility-tree-roled output (which is intermediate). **Tarsier directly claims text-only beats text+vision by 10-20%** on internal benchmarks.
docs/checkpoints/paper_planning.md:1991:| Hierarchy preservation | tab-indented tree | flat list typically | minor (re-tokenize cost) |
docs/checkpoints/paper_planning.md:1994:→ Snapshot-alone token estimate: **P79 ~1000-1500** vs **agent-browser/Playwright MCP ~200-400** = ~3-5× format-trim gap. The §1 hook table "3008 tokens cls / 3437 reddit" refers to **full prompt context** (system prompt + task + observation + history) not observation alone.
docs/checkpoints/paper_planning.md:1998:#### (ii) × L3 — Agent-compute affordance synthesis (paper-1 main hook)
docs/checkpoints/paper_planning.md:2003:| §103 | Phantom-SoM 4-mode routing arm finding (B0 reddit 6-mode 完整 cell) | ✅ paper §1 main hook, status:provisional pending 16-cell rerun |
docs/checkpoints/paper_planning.md:2005:**(ii) × L3 内部 4-tier sub-gradient — paper §1 hook 关键 distinction (added 2026-05-04 from DR audit)**:
docs/checkpoints/paper_planning.md:2013:| **L3-inference (我们)** ⭐ | Inference-time, no offline, no retrieval | **Phantom routing space** (paper-1 hook) | Agent reads `[SOM_MARKS]` directly from observation processing — no pretraining, no RAG, no offline exploration phase |
docs/checkpoints/paper_planning.md:2028:| **OmniParser-v2** (Microsoft, Yu/Yang/Wan/Bai) | (ii) × L2 — pipeline preprocessing | Feb 2025 (arXiv:2502.16161) ✅ verified | Canonical instance of L2 affordance synthesis. **MoE token-router shared decoder + Two-Stage Structured-Points-of-Thought (SPOT) prompting**. Literal output token format (Round-3 verified): `<box_start> <x_0.12> <y_0.45> <content_submit> <box_type_button> <box_end>`. Paper §1 / §3 cite as "pipeline preprocessing alternative to inference-time L3 substitution"; literal SPS format vs our `[id=N] role 'label'` format direct contrast |
docs/checkpoints/paper_planning.md:2036:#### (iii) × L1 — Server-side channel addition: **0 done** (paper 2 future)
docs/checkpoints/paper_planning.md:2048:| **A2A protocol** (Google) | Agent-card metadata channel for agent-to-agent negotiation | Apr 2025 | Different layer (agent ↔ agent) but conceptually same channel-addition principle |
docs/checkpoints/paper_planning.md:2050:| **Doubao Mobile `INJECT_EVENTS`** (ByteDance) | OS-level event channel for cross-app actions | Dec 2025 | Mobile-specific; channel-addition at OS layer rather than web |
docs/checkpoints/paper_planning.md:2053:#### (iii) × L2 — Agent-side instrumentation channel addition: **0 done, ≥7 § identified gaps** ⚠️
docs/checkpoints/paper_planning.md:2071:#### (iii) × L3 — Agent-compute channel addition: **n/a (impossible by definition)**
docs/checkpoints/paper_planning.md:2075:### §21.3 Identified-but-not-done items (想做但未做)
docs/checkpoints/paper_planning.md:2077:#### A. (iii) × L2 instrumentation gaps (Paper §4/§5 disclosure 必备 + paper 2 future)
docs/checkpoints/paper_planning.md:2088:#### B. (ii) × L1 — Server-side affordance synthesis (paper 2)
docs/checkpoints/paper_planning.md:2094:#### C. (iii) × L1 — Server-side channel addition (paper 2)
docs/checkpoints/paper_planning.md:2101:#### D. Agent module M-modules (笔记 行动规划)
docs/checkpoints/paper_planning.md:2111:#### E. Cross-stack generalization (paper 2 / future paper)
docs/checkpoints/paper_planning.md:2117:### §21.4 Paper-level methodology asymmetries (跟 9-cell taxonomy 平行的 disclosure list)
docs/checkpoints/paper_planning.md:2131:### §21.5 Substitution Gradient — Paper §1 Hook Positioning (added 2026-05-04 from DR Section E)
docs/checkpoints/paper_planning.md:2140:   substitution at         tokenized list, downstream     grounding 进 weights,      explore 写 doc → deploy-      (paper-1 main hook)
docs/checkpoints/paper_planning.md:2149:**Paper §1 hook 候选 framing** (基于 **research-characterization 角度**, 不是 artifact-existence 角度 — fundamental reframe 2026-05-04 deepest-evening):
docs/checkpoints/paper_planning.md:2151:> "Industry production deployment of web-agent SDKs (Playwright MCP from Microsoft, agent-browser from Vercel Labs, Stagehand from Browserbase, Tarsier from Reworkd, Browser Use SDK, Skyvern, etc.) configures single-mode operation for token economy: typically a11y-tree-roled flat-ref representation (~200-400 tokens per snapshot) chosen over raw DOM/HTML (~3000-5000 tokens) for deployment cost. While these deployments demonstrate that text-only substitution **operationally works** in production at scale — agent-browser alone has 81+ releases integrated with Claude Code, Cursor, Codex, and Gemini agent CLIs; OpenClaw with browser tools as core capability hit 361K GitHub stars by early 2026 — **no published study systematically characterizes which routing behaviors emerge from each substitution dimension**. Industry single-mode deployment by definition cannot isolate the contribution of (a) text payload format (hierarchical AXTree vs flat ref list), (b) prompt-format expectation (DOM-prompt vs SoM-prompt), or (c) image presence (with vs without visual marker overlay): production agents commit to one configuration matched across these dimensions, and cannot run controlled cross-mode comparison on identical task pool. Even Tarsier's anecdotal claim of unimodal text beating GPT-4V + visual SoM by 10-20% (an internal benchmark) lacks systematic per-dimension characterization, controlled experimental design, or cross-task / cross-model / cross-site generalization analysis. We provide this systematic peer-reviewed characterization via the **phantom routing space**: a 4-corner ablation cube (text payload axis × prompt-format axis × image presence axis, with image-off half forming the phantom space). Each phantom corner — DOM (hierarchical AXTree, DOM-prompt), P-text (flat AXTree, DOM-prompt), P-prompt (hierarchical AXTree, SoM-prompt), P-SoM (flat AXTree, SoM-prompt) — contributes unique tasks not solvable by other corners, evidencing each substitution dimension has independent routing-behavior effect. Phantom-SoM (cube center, axis 1+2 compound) is the deployment hero satisfying a 4-fold drop-in property: cost ≈ DOM (no image embedding tax), latency ~50% lower (no image inference), signal AUROC ≥ baseline (routing infra drop-in), drop-one ≥ 1pp pre-registered. The phantom space exhibits 2-axis empirical structure (axis 1 text payload via P-text vs DOM; axis 2 SoM-style prompt via P-prompt vs P-SoM) — both dimensions contribute non-overlapping unique tasks. We use **non-pretrained, non-fine-tuned Qwen3-VL** (the same backbone Magma integrates SoM+ToM into via pretraining and ScribeAgent fine-tunes via 6B-token DOM workflow corpus to WebArena 51.3%) — clean experimental isolation of inference-time prompt-structure contribution from pretraining/fine-tuning contributions. Industry can adopt these specific configurations based on our characterization without re-running controlled comparison; format-axis trims (URL/property emission, ref format width, hierarchy preservation) are stackable deployment optimizations independent of substitution-axis findings."
docs/checkpoints/paper_planning.md:2153:(这版 hook ~530 词, **research-characterization angle** 替代 artifact-existence angle. 含 (a) industry 部署 acknowledgment ("operationally works in production"), (b) **research gap statement** (industry single-mode 无法做 controlled cross-mode comparison), (c) 4-corner ablation cube + per-dimension characterization, (d) **all 4 phantom corners equal-novel** as research cells, (e) Magma+ScribeAgent same-Qwen-base differentiator (pretraining/fine-tuning isolation), (f) format-axis orthogonality disclosure)
docs/checkpoints/paper_planning.md:2160:  - **DOM-like + P-text-like corners**: industry analogs exist — Playwright MCP a11y-tree+refs, agent-browser text-only mode, Tarsier text-mode. **Critical nuance**: industry deploys these **arbitrarily for token economy** (smaller payload = cheaper inference), NOT from understanding per-dimension routing behavior. **No published industry comparison of P-text vs DOM** (hierarchical AXTree); no awareness that text-flattening has independent routing effects beyond token-cost reduction. Industry artifact existence ≠ understanding ≠ characterization.
docs/checkpoints/paper_planning.md:2167:  - **Implication for industry**: paper's findings let practitioners choose configurations based on per-dimension routing behavior (not just cost), e.g. "use P-text not because it's cheap but because it activates flat-list selection ontology useful for tasks X/Y/Z".
docs/checkpoints/paper_planning.md:2169:**Reviewer-defense layering** (updated):
docs/checkpoints/paper_planning.md:2170:- "Industry already does this" → 反驳 1: industry has artifact analogs only for DOM/P-text/SoM corners, NOT for cube-center P-SoM/P-prompt (these emerge from our framework). 反驳 2: industry deploys arbitrarily for cost economy, never compared with DOM, never characterized per-dimension behavior. Paper discovers + characterizes routing effects industry deployed-without-realizing.
docs/checkpoints/paper_planning.md:2172:- "P-text is not novel, agent-browser already deploys text-only" → 反驳: agent-browser deploys text-only **for cost**; we discover text-only has **independent routing effects beyond cost** (drop-one unique tasks, M1 ontology reframe). Different epistemic claim than industry deployment.
docs/checkpoints/paper_planning.md:2176:> **Epistemic-level distinction**: All industry instances below operate at **artifact-deployment level** (single-mode production configuration for cost/economy reasons). Our paper operates at **research-characterization level** (controlled cross-mode comparison on identical task pool to isolate per-dimension routing behavior). Industry deployment ≠ research finding — different epistemic levels, both valid, paper contribution at characterization level not artifact level.
docs/checkpoints/paper_planning.md:2180:| OmniParser-v2 | They do (ii)×L2 pipeline preprocessing with literal SPS format `<box_start>...<box_end>` (pure-vision MoE, no DOM); we do (ii)×L3 LLM compute substitution with `[id=N] role 'label'` format from accessibility tree. Different layer + different engine path. |
docs/checkpoints/paper_planning.md:2186:| **Playwright MCP** (Microsoft) + **agent-browser** (Vercel Labs) | **Their artifact**: ~200-400 token a11y-tree refs deployment SDK, integrated with Claude Code / Cursor / Codex / Gemini. **Their characterization gap**: single-mode production deployment by definition cannot isolate which routing benefits come from text-flattening (hierarchical AXTree → flat ref list) vs accessibility-tree-extraction (raw HTML → a11y-roled). Industry deploys for token economy, not for behavior characterization. We isolate text-flattening contribution via DOM (hierarchical AXTree) vs P-text (flat AXTree) controlled comparison on identical task pool. **Format-axis orthogonal** (verified by reading processors.py:513-619 2026-05-04): scope similar (both a11y-tree-roled), gap from format choices — independent of substitution-axis claim. |
docs/checkpoints/paper_planning.md:2187:| **Stagehand** (Browserbase) + **Browser Use SDK** + **Skyvern** | **Their artifact**: production SDK with hardcoded environmental patches + a11y tree extraction. **Their characterization gap**: same as Playwright MCP — single-mode deployment, no controlled cross-mode comparison. Their environmental patches (popup/cookie/dropdown handling) are deployment scaffolding not characterization study; characterized comparison of patched-vs-unpatched routing requires research harness (which we provide via VWA + Phase A 4-cluster fixes). |
docs/checkpoints/paper_planning.md:2198:| **PageAgent** (Alibaba, frontend pure-JS) ⭐ Chinese P-text-equivalent artifact | (ii)×L2 — pipeline preprocessing | `github.com/alibaba/page-agent` (17.5k stars / v1.8.1 / 2026-04-27) | **Their artifact**: pure-DOM flat-text representation (no multimodal), runs on any text-only LLM, deployed in Alibaba ecosystem (淘宝/天猫/钉钉/阿里云). **Their characterization gap**: single-mode production deployment for cost economy, no controlled DOM-vs-flat-text comparison on isolated routing-behavior axis. Our P-text corner provides this characterization on Qwen3-VL. |
docs/checkpoints/paper_planning.md:2199:| **UI-TARS** (ByteDance + Tsinghua) + **UI-TARS-2** | Outside (ii)×L3 — pure visual VLM L3-pretrain | `arXiv:2501.12326` (UI-TARS) / `arXiv:2509.02544` (UI-TARS-2) / `github.com/bytedance/UI-TARS` (10.2k stars) | UI-TARS-2 (2025-09): OSWorld 47.5 / WindowsAgentArena 50.6 / AndroidWorld 73.3. Pure visual GUI agent via large-scale pretraining (627M+ GUI samples claimed by V2 — fabricated, real number TBD). **Same family as Magma**: pretraining-time substitution vs paper's inference-time substitution on non-pretrained Qwen3-VL. |
docs/checkpoints/paper_planning.md:2213:### §21.6 Industry counter-evidence stack (added 2026-05-04 from DR Section D)
docs/checkpoints/paper_planning.md:2222:| **Online-Mind2Web SOTA Avenir-Web 53.7% with Gemini 3 Pro backbone (Nov 18 2025 public preview)** ✅ VERIFIED Round-3 | Avenir-Web research initiative 2026 / Online-Mind2Web (Xue et al. 2025) | Paper §1 motivation: "Industry SOTA fails ~46% on live web, indicating environment-side hostility is unsolved at any agent-side scale" — **Round-2 "Operator 41.7% failure" claim was misattribution** (actually MAI-UI 41.7% **success** rate on MobileWorld), Round-3 verified Avenir-Web only |
docs/checkpoints/paper_planning.md:2231:| **Industry SDK convergent token economy at ~200-400 per snapshot** ✅ verified Vercel agent-browser + Microsoft Playwright MCP | agent-browser README + Playwright MCP docs | Both Vercel agent-browser and Playwright MCP independently converge on **~200-400 tokens per page snapshot via accessibility-tree extraction + format trimming + compact ref format** = 10-15× compression vs **raw HTML/DOM 3000-5000** (intermediate AXTree-roled output is ~1000-1500 tokens; further trim to ~200-400 via format/URL/property strip). **Paper §3 method footnote**: industry SDK compression at L2 combines (a) accessibility-tree extraction (drops presentational HTML), and (b) format trimming (URL/property strip, compact refs, flat list) — both independent of (ii)×L3 substitution-axis paper-1 main claim. |
docs/checkpoints/paper_planning.md:2232:| **Tarsier text-beats-vision claim** ⭐ direct industry analog of phantom routing thesis | Tarsier (Reworkd) v0.6.0 README internal benchmark | **"unimodal beats GPT-4V + Tarsier-Screenshot by 10-20%"** — production-deployment-level anecdotal evidence that text-only routing matches/beats text+vision. **Paper §1 hook critical cite**: positions our paper as systematic peer-reviewed characterization of what Tarsier deployment hints at without controlled experiment. |
docs/checkpoints/paper_planning.md:2233:| **Format-axis orthogonality (paper §3 method 必备 disclosure)** ⚠️ Round-3 fact-check 修正 prior over-claim 2026-05-04 | Convergent observation across agent-browser, Playwright MCP, Tarsier, Stagehand + verified reading `external/visualwebarena/browser_env/processors.py:513-619` (parse_accessibility_tree + clean_accesibility_tree) | **Element scope similar across P79 paper and industry SDKs** (both a11y-tree-roled via Chrome accessibility tree extraction). **Token-economy gap is format-axis, not scope-axis** — P79 emits URL property (Chrome a11y standard, in P79 not in IGNORED_ACTREE_PROPERTIES list) + a11y properties (`focused: True` / `required: False` / `expanded: False`) + tab-indented hierarchy + longer ref format (`[id=88]` vs `@e88`); industry SDK trims to compact format. Snapshot-alone token estimate: P79 ~1000-1500 vs industry ~200-400 = **~3-5× format-axis trim gap**. The §1 hook table "3008 tokens cls / 3437 reddit" refers to **full prompt context** (system + task + observation + history) not observation alone. **Two orthogonal compression axes**: (1) substitution mechanism (visual marker render → textual ref list) is paper-1 main characterization axis; (2) format trimming (URL/property emission, ref width, hierarchy) is industry deployment optimization stackable on top — INDEPENDENT of substitution claim. **§96 design decision** ("preserve all elements") accurately refers to P79 SoM marker scope vs **VWA-original ImageObservationProcessor** (annotates only interactive elements on screenshot for SoM mode), not P79 vs industry SDK — both within a11y-tree-roled scope. Earlier §21 over-claim about "interactive-only filter" 已撤回 — actual industry SDK behavior (Playwright MCP example output `- heading 'Example Domain' [ref=e1]`) shows headings/structural a11y elements included, not strictly interactive-only. |
docs/checkpoints/paper_planning.md:2238:### §21.6.5 Scope-defense — Cognitive-routing vs SE-engineering distinction (added 2026-05-04 deepest-evening, 笔记 §109.19)
docs/checkpoints/paper_planning.md:2242:> Paper claim is **cognitive routing-behavior characterization** (per-axis representation effect on LLM behavior). SE-engineering modules (site-specific fingerprint databases, short symbolic action grammars, benchmark instrumentation patches) are **deployment optimizations** whose effect is cost/latency engineering — not LLM cognitive behavior. Including them in substitution-axis ablation would conflate cognitive characterization with software engineering benchmarking.
docs/checkpoints/paper_planning.md:2244:#### §21.6.5.1 Distinction grid
docs/checkpoints/paper_planning.md:2250:| **agent-browser `click @7` short action grammar** | ❌ fixed | ✅ SE module | engineering (output-token cost saving) | ❌ exclude — out of cognitive routing scope |
docs/checkpoints/paper_planning.md:2252:| **VWA Magento FPC fix / Postmill PHP gc_maxlifetime fix / Wikipedia ZIM version fix** | ❌ fixed | ✅ SE infrastructure module | engineering (benchmark site-config bug) | ❌ exclude **as substantive finding** — ✅ acknowledge as **evidence-layer prereq** (Appendix D) |
docs/checkpoints/paper_planning.md:2254:| **Phase A 4-cluster fixes (C1 dispatch / C2 page_changed / C3 fuzzy cycle / C4 RNG seeding)** | ❌ fixed | ✅ SE benchmark instrumentation module | engineering (benchmark cleanliness) | ❌ exclude as substantive finding — ✅ acknowledge as paper-grade rigor prereq |
docs/checkpoints/paper_planning.md:2255:| **Watchdog auto-clean protocol (6-layer defense)** | ❌ fixed | ✅ SE data hygiene module | engineering (data cleanliness automation) | ❌ exclude — paper-grade rigor scaffolding |
docs/checkpoints/paper_planning.md:2257:#### §21.6.5.2 Why this distinction matters (reviewer-defense argument)
docs/checkpoints/paper_planning.md:2263:| "Why didn't you ablate FPC fix / Phase A 4-cluster as substitution dimensions?" | (no answer — risks looking like we're hiding methodology asymmetry) | "Because these are **evidence-layer instrumentation** (preventing benchmark Environment-Failure from contaminating cognitive Agent-Failure measurement). They enable controlled comparison; they're not the comparison itself. Paper §3 evaluation methodology + Appendix D explicit acknowledge them as paper-grade rigor prereq, not as cognitive routing findings." |
docs/checkpoints/paper_planning.md:2264:| "Why is your paper not a software-engineering paper?" | (weak — relies on intuitive claim "we focus on routing") | "Because phantom 4-corner ablation **isolates per-axis representation effect on LLM cognitive behavior**, which generalizes across sites/tasks/models. SE modules (fingerprint DB / short grammar / FPC fix) generalize **only within their specific deployment configuration** — different epistemic generalization scope. Paper provides cognitive routing characterization that practitioners deploy on top of any SE-module stack." |
docs/checkpoints/paper_planning.md:2266:#### §21.6.5.3 Phantom 4-corner status under this distinction
docs/checkpoints/paper_planning.md:2270:- **Effect**: reveals **per-axis representation→LLM-behavior** causal isolation (not cost/latency engineering)
docs/checkpoints/paper_planning.md:2280:### §21.6.6 Substitution-axis scope — Observation-axis paper, action-axis future work (added 2026-05-04 deepest-evening)
docs/checkpoints/paper_planning.md:2282:#### §21.6.6.1 Two independent substitution axes
docs/checkpoints/paper_planning.md:2289:These two axes are **orthogonal**: observation substitution affects what LLM sees, action-grammar substitution affects what LLM emits. They independently impact cost (input vs output tokens) and behavior (representation routing vs action-format constraint).
docs/checkpoints/paper_planning.md:2291:#### §21.6.6.2 Where industry sits on action-axis
docs/checkpoints/paper_planning.md:2303:#### §21.6.6.3 Paper §21 explicit limitations prose
docs/checkpoints/paper_planning.md:2309:#### §21.6.6.4 Why this scope is principled (not arbitrary)
docs/checkpoints/paper_planning.md:2312:2. **Industry already characterizes action-axis empirically** (short grammar vs verbose JSON cost saving), but with same SE-engineering caveat as §21.6.5 — no controlled cognitive characterization on action-axis either, parallel research opportunity
docs/checkpoints/paper_planning.md:2313:3. **Phantom routing space hero claim (P-SoM 4-fold drop-in)** uses default verbose action grammar — claim does NOT depend on action-grammar axis. Future action-axis extension is **stackable** on top of observation-axis findings (same orthogonality logic as format-axis trim per §109.16)
docs/checkpoints/paper_planning.md:2315:### §21.7 Pending decisions (后续 discuss / advisor sync)
docs/checkpoints/paper_planning.md:2317:#### Original (5/3)
docs/checkpoints/paper_planning.md:2319:1. **Paper 1 §1 hook contextualization** — 是否加 3-spectrum framing (i)/(ii)/(iii)? 现 §1 是 P-SoM hero + structural ablation, 加 contextualization 段 ~150 词
docs/checkpoints/paper_planning.md:2327:#### Added 2026-05-04 (post DR audit)
docs/checkpoints/paper_planning.md:2329:8. **Adopt §21.5 candidate paper §1 hook prose?** — 用 substitution gradient framing (industry precedent stack contextualization) 替代 / 升级现 §1 hook (~250 词, 含 4-tier sub-gradient contrast + niche positioning)
docs/checkpoints/paper_planning.md:2331:10. **Env-side pilot 实施 — Sweet Spot 设计** — 用户提议 "server emit hidden select options" 是 NLWeb-style 实例。Sweet spot 选 (a) inline `<script type="application/agent-marks">` JSON-LD / (b) HTTP header / (c) sidecar endpoint /agent/v1/page-state? 工作量 + paper claim power 跟 16-cell rerun critical path 优先级冲突
docs/checkpoints/paper_planning.md:2333:12. **OmniParser-v2 跟 phantom routing 对比 prose** — paper §3 / §5 explicit (ii)×L2 vs (ii)×L3 layer 区分; OmniParser 是 industry-side L2 instance, phantom routing 是 paper-side L3 instance
docs/checkpoints/paper_planning.md:2336:#### Added 2026-05-04 late evening (post (ii)×L2 industry sweep)
docs/checkpoints/paper_planning.md:2338:14. **Tarsier explicit cite + differentiate prose (paper §1/§2)** — Tarsier text-beats-vision claim "unimodal beats GPT-4V + Tarsier-Screenshot by 10-20%" 是 closest industry analog of phantom routing thesis; paper §1 hook 必须 cite + differentiate (我们提供 systematic peer-reviewed characterization vs Tarsier deployment-only anecdote). **不 cite Tarsier 是 reviewer 拒稿 trigger**.
docs/checkpoints/paper_planning.md:2345:#### Added 2026-05-04 deeper-evening (post code fact-check correction)
docs/checkpoints/paper_planning.md:2347:20. **Paper §3 method 描述 token gap source 准确化** ⚠️ — 修正 prior over-claim "industry SDK filter to interactive-only / P79 preserve all elements" 为 accurate framing "scope similar (both a11y-tree-roled), gap from format axis (URL/property/hierarchy/ref)". 背景: 用户 push back catches bug, 实际读 `external/visualwebarena/browser_env/processors.py:513-619` 验证. **Pending paper §3 method explicit prose with verified specifics — substitution-axis (paper main) + format-axis (industry deployment) orthogonal, both INDEPENDENT of element-scope axis (both us and industry SDK use a11y-tree extraction)**.
docs/checkpoints/paper_planning.md:2348:21. **§1 hook table token figure clarification** — "3008 tokens cls / 3437 reddit" 是 **full prompt context** (system + task + observation + history) 不是 observation snapshot alone (~1000-1500). Paper writing 时 explicit clarify (avoid reviewer 误解 single-snapshot vs full-context).
docs/checkpoints/paper_planning.md:2350:#### Added 2026-05-04 deepest-evening (research-characterization angle)
docs/checkpoints/paper_planning.md:2352:22. **Paper §1 hook framing 选择** ⭐ critical — 用 **research-characterization angle** ("industry deploys for economy, paper characterizes for behavior", §21.5 prose ~530 词) 还是保留之前的 substitution-gradient-niche framing? **Strong recommend research-characterization angle**: (a) honest about industry-already-deploys-equivalent-artifacts (avoid reviewer attack vector); (b) shifts paper claim to characterization level (industry can't做 controlled comparison only research can) — different epistemic level than artifact deployment; (c) all 4 phantom corners equal-novel as research cells (P-text not less novel than P-SoM); (d) Magma+ScribeAgent same-Qwen-base differentiator becomes pretraining/fine-tuning isolation argument naturally.
docs/checkpoints/paper_planning.md:2353:23. **Artifact-vs-characterization epistemic distinction** explicit 进 paper §1 / §2 prose — "industry deployment ≠ research finding" 是 reviewer-defense critical phrase, paper §1 hook + paper §2 related work 都 explicit acknowledge industry artifact existence + position paper at characterization level. 不 over-claim "first to use these configurations", claim "first to systematically characterize routing behavior of these configurations on Qwen3-VL via controlled cross-mode comparison".
docs/checkpoints/paper_planning.md:2355:#### Added 2026-05-04 deepest-evening latest (post §109.18 fact-check + §109.19 scope-defense)
docs/checkpoints/paper_planning.md:2357:24. **SE-module-vs-cognitive-routing scope-defense explicit prose 进 paper §3 / §8** ⭐ critical — §21.6.5 argument 必须 explicit 写进 paper: "deliberately exclude SE-engineering modules (站点指纹库 / 短 grammar / FPC fix as substantive findings) from substitution-axis ablation because paper claim is cognitive routing characterization not deployment optimization". Paper §3 method 段 + §8 discussion limitations 各一段 prose, parallel to §21.5 research-characterization argument. 不写 = reviewer 攻击"why not ablate site fingerprint DB / short grammar"无 principled defense.
docs/checkpoints/paper_planning.md:2359:25. **Observation-axis vs action-axis scope explicit 进 paper §3 / §8 limitations** — §21.6.6 argument: paper phantom routing space focuses on observation-representation axis (4-corner cube), action-grammar substitution (short symbolic grammar like `click @7`) is orthogonal future-work axis. Paper §3 method explicit "we use VWA default verbose action serialization across all 6 modes for consistent observation-axis ablation control"; paper §8 future work explicit "extending phantom routing to action-axis (8-cell extended cube)". 不写 = reviewer 误以为 paper claims action-axis 也 covered.
docs/checkpoints/paper_planning.md:2361:26. **中国 industry sweep integration into paper §1 / §2** — §109.18 verified arXiv IDs cheat sheet 已就位 (PageAgent / UI-TARS / UI-TARS-2 / AutoWebGLM / AutoGLM / WebRL / WebSailor suite / OS-Atlas / Mobile-Agent v2/v3 / Qwen3-VL technical report / CogAgent), paper.bib 加 ~10 BibTeX entries, paper §1 hook prose 中加 "Chinese industry SDKs (PageAgent from Alibaba, UI-TARS from ByteDance/Tsinghua, AutoGLM from Zhipu, WebSailor suite from Alibaba Tongyi)" parallel 西方 SDK list, achieving dual-region industry sweep coverage. **Special anchor**: Qwen3-VL Technical Report `arXiv:2511.21631` 直接对应 paper backbone (B0=235B-A22B / B1=4B variants), paper §1 / §3 method explicit cite with backbone disclosure.
docs/checkpoints/paper_planning.md:2363:27. **§21.6.5 SE-module exclusion full audit** — 明确列 paper Appendix D "evidence-layer instrumentation" vs "cognitive routing finding" 两 category split: 现 Phase A 4-cluster fixes / FPC fix / watchdog auto-clean / Magento fix / Postmill PHP gc fix / Wikipedia ZIM fix 全列 evidence-layer 不算 finding. 写一段 prose: "Paper §3 evaluation methodology + Appendix D explicit categorize all SE-engineering instrumentation (~37 entries from §21.2 (i)×L1 + (i)×L2) as paper-grade rigor prereq, not as cognitive routing findings. Phantom routing space 4-corner ablation operates on top of clean evidence-layer infrastructure".
docs/checkpoints/paper_planning.md:2367:## §22 Multi-Register Novelty Inventory (advisor 5/5 sync ready, 2026-05-04 audit)
docs/checkpoints/paper_planning.md:2371:> **缘由**: 5/3 pre-registration reframe + §109.16-19 4-round epistemic upgrade 后, paper-strategic novelty 从单 vector "phantom routing arm" 扩到 multi-register layered claim. 用户 5/4 prompt: "现在的 novelty 还缺什么吗 — 审计下, 结合其他文档/figure". 用户自己 list 了 6 个维度 (现象/效果/原因/数据比较/routing/cross-X), audit cover 现 inventory + 加补 dimensions.
docs/checkpoints/paper_planning.md:2373:### §22.1 5-register novelty framework
docs/checkpoints/paper_planning.md:2382:**Standard**: ⭐ = core claim (paper §1 必须 surface) / ☆ = supporting claim (各 section 必 surface) / · = polish / context
docs/checkpoints/paper_planning.md:2384:### §22.2 Inventory by register
docs/checkpoints/paper_planning.md:2386:#### Register I — Theory / Concept (mostly 用户列举的 A-F)
docs/checkpoints/paper_planning.md:2390:| ⭐ | (A) **Phantom routing space phenomenon** (named operational entity, "no annotated image" boundary) | ✅ §1 + §2 + §21 | 笔记 §103 + §108 | §1 hook |
docs/checkpoints/paper_planning.md:2391:| ⭐ | (B) **4-fold drop-in property** as unified deployment criterion (cost ≈ DOM + latency 50% + AUROC ≥ baseline + drop-one ≥ 1pp) | ✅ §1 + §3 + §11 | 笔记 §106 | §1 + §4 |
docs/checkpoints/paper_planning.md:2392:| ⭐ | (C) **3-axis cube + image-axis isolation** as paper framework contribution | 🟡 §21.5 三层 + §1 one-liner 1 句, paper §1 prose 没 reflect | 用户 5/4 push | §1 hook + §3 |
docs/checkpoints/paper_planning.md:2401:#### Register II — Method / Process discipline
docs/checkpoints/paper_planning.md:2406:| ⭐ | (L) **4-dimension Evidence framework** (Outcome/Macro/Micro/Efficiency 正交, replaces hierarchical-layer thinking) | 🟡 paper_planning §3 重组, paper §3-§4 prose 部分 reflect | 笔记 §106 |
docs/checkpoints/paper_planning.md:2407:| ⭐ | (M) **~100× deployment-class cost gap** (B0 API $0.04/ep vs B1 electricity $0.0004/ep, **NOT capability ratio**) | 🟡 笔记 §106 + paper §3 prose 部分, §1 没 explicit | 笔记 §106 |
docs/checkpoints/paper_planning.md:2409:| ☆ | (O) **Phase A 4-cluster bug audit + 5-tier lit-aligned review** | 🟡 笔记 §107, Appendix D candidate, ~37 entries cataloged | VWA_FRAMEWORK_BUGS doc |
docs/checkpoints/paper_planning.md:2412:| ☆ | (R) **Bootstrap CI on H3 unique-count structural test** | ✅ aggregate_phantom_lift.py 实现 | EVIDENCE_LAYER_AUDIT |
docs/checkpoints/paper_planning.md:2414:| ☆ | (T) **Format-axis vs scope-axis orthogonality honesty** (processors.py:513-619 verified, retract over-claim) | ✅ paper_planning §21.6 quantitative anchor | 笔记 §109.16 |
docs/checkpoints/paper_planning.md:2415:| · | (U) **Watchdog 6-layer auto-clean + manifest-grade discipline** (5/4 hardened: reset hard-fail + watchdog self-exit) | ✅ run_manifest.yaml + watchdog protocol commit a912545 | run_manifest.yaml |
docs/checkpoints/paper_planning.md:2417:#### Register III — Application / Impact
docs/checkpoints/paper_planning.md:2421:| ⭐ | (V) **Routing utility — Tier 1+2 router design** (oracle TF-IDF+LR + first-step trigger, no test leak) | 🟡 paper_planning §8 design, infra ready | §8 |
docs/checkpoints/paper_planning.md:2422:| ⭐ | (W) **Multi-metric Pareto + Green AI axis** (cost / P95 latency / regional carbon, B1 measured 45 region) | 🟡 paper_planning §11, fig3_regional_carbon ready | §11 |
docs/checkpoints/paper_planning.md:2423:| ⭐ | (X) **Independent routing effects beyond cost** (industry deploys P-text arbitrarily for cost; paper discovers per-axis routing benefits) | 🟡 §21.5 三层 paper-discovery 段 5/4 add | 用户 5/4 |
docs/checkpoints/paper_planning.md:2425:| ☆ | (Z) **Industry-can-adopt configurations based on paper characterization** (not just cost) | 🟡 §21.5 三层 hierarchy paper-discovery 段 | 5/4 surface |
docs/checkpoints/paper_planning.md:2426:| · | (AA) **Routing signal portfolio per mode** (fig0g 5 signal × N mode AUROC matrix; not just "AUROC ≥ baseline" but **which signal works for which mode**) | (NEW, 5/4 audit add) — fig0g already computed | §6 router design input |
docs/checkpoints/paper_planning.md:2428:#### Register IV — Survey / Position framing
docs/checkpoints/paper_planning.md:2432:| ⭐ | (BB) **9-cell intervention taxonomy + dual-track** (3 spectrum × 3 layer, 12+ verified industry instances 西方+中国) | ✅ §21 + dual_track_taxonomy.canvas | §21 |
docs/checkpoints/paper_planning.md:2440:#### Register V — Future-paper trajectory
docs/checkpoints/paper_planning.md:2449:### §22.3 Audit gaps surfaced 5/4 — 哪些 documented elsewhere 但 paper §1 hook 没 surface
docs/checkpoints/paper_planning.md:2453:| §1 prose stuck at 4/29 framing (4th-arm + 2-knob + capability) | 5/3 reframe (Hero+Structural+R1-R5) + 5/4 framework-tier (cube+image-axis) + ~100× cost gap + 4-dim Evidence + research-characterization angle 全部没 reflect | 顶会 reviewer 看 §1+abstract 决定 contribution claim, 这是最大 gap | 16-cell rerun done 后 codex pass for §1 prose 重写, 6-contribution structure (§22.4 candidate) |
docs/checkpoints/paper_planning.md:2454:| Figures cover quantitative anchor 但 §1 prose 没 reference | fig0d Jaccard 0.29-0.49 / fig3a cost ≈ DOM / fig3c latency 50% / fig0g AUROC ≥ baseline / fig2 cross-model fingerprint | 4-fold drop-in property 缺 visual quantitative grounding | Codex pass §1 加 figure-anchor sentences (1-2 quantitative anchor per contribution) |
docs/checkpoints/paper_planning.md:2459:### §22.4 Paper §1 hook 重写 candidate (6-contribution structure)
docs/checkpoints/paper_planning.md:2466:3. **Industry-vs-research epistemic distinction**: industry deploys P-text arbitrarily for cost; paper discovers per-axis routing effects beyond cost [Register IV EE + Register III X]
docs/checkpoints/paper_planning.md:2468:5. ~~**Capability interaction**: B0 vs B1 failure-mode shift +43.7pp~~ — **DROPPED 2026-05-09**. Paper §1 third contribution cut to focus on phantom routing space + structural axes (H1/H3). B1 retained as cross-capability robustness check, not a separate scientific claim. `fig_capability_b0_b1.png` deleted; `disagreement_clusters.md` retained as supplement material if §8 limitations needs reference.
docs/checkpoints/paper_planning.md:2469:6. **Methodology**: 4-dim Evidence framework + R1-R5 pre-registration + ~100× deployment-class cost gap framing (B0 API $0.04/ep vs B1 electricity $0.0004/ep) [Register II K+L+M]
docs/checkpoints/paper_planning.md:2473:### §22.5 Advisor 5/5 sync priority
docs/checkpoints/paper_planning.md:2479:4. (W) Multi-metric + Green AI axis (cost+latency+carbon)
docs/checkpoints/paper_planning.md:2480:5. (X) **Independent routing effects beyond cost** (industry-vs-research epistemic distinction)
docs/checkpoints/paper_planning.md:2486:- (V) Tier 1+2 router 部署故事
docs/checkpoints/paper_planning.md:2487:- (M) ~100× deployment-class cost gap framing
docs/checkpoints/paper_planning.md:2495:### §22.6 Action items (post-sync)
docs/checkpoints/paper_planning.md:2502:6. **Routing signal portfolio explicit prose** (Register III AA new) → §6 router design
docs/checkpoints/paper_planning.md:2506:### §22.7 Maintenance

exec
/bin/bash -lc "rg --files scripts/analysis | rg 'preregistration_decision_test|aggregate_phantom_meta|aggregate_phantom_lift|aggregate_routing_auroc|sensitivity_loo_meta'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
scripts/analysis/aggregate_phantom_lift.py
scripts/analysis/aggregate_routing_auroc.py
scripts/analysis/preregistration_decision_test.py
scripts/analysis/sensitivity_loo_meta.py
scripts/analysis/aggregate_phantom_meta.py

codex
The first pass already shows a few internal tensions: the prereg text has been updated to N=4 in some places, but the power appendix still talks about 16-cell K thresholds and “N≥10” meta-analysis, while the current primary design uses DL meta with four cells. I’m reading the implementation next because that will decide whether these are just stale prose or actual gate-breaking issues.
exec
/bin/bash -lc "nl -ba scripts/analysis/preregistration_decision_test.py | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	"""Preregistration decision test — Phase 1a 24-condition / 4-cell H1 / H3 / H2 evaluation.
     2	
     3	⚠️ REWRITTEN 2026-05-13 to align with preregistration.md revisions (codex stress audit
     4	   Flaws 2 + 3 fix):
     5	   - PRIMARY GATE = pooled DerSimonian-Laird random-effects meta + TOST equivalence
     6	   - K-of-N reclassified gate → transparency consistency check (per pre-data 2026-05-13
     7	     reclassification, see `preregistration.md` §4 audit B9 + Appendix A 2026-05-13)
     8	   - H1 formula = P-SoM drop-one oracle ceiling lift (NOT P-SoM ≥ best single mode)
     9	   - H3 family = axis-1 (P-text \ P-SoM) + axis-2 (P-prompt \ P-SoM), both pooled
    10	   - Scope = 4 (site, model) statistical cells, each with 6 modes' per-task SR data
    11	
    12	Definitions (per preregistration.md §2 + §4):
    13	  - cell = 1 (site, model) statistical stratification unit. Phase 1a N=4 cells:
    14	    (cls, B0), (cls, B1), (red, B0), (red, B1).
    15	  - condition = 1 (site, model, mode) operational launch unit. Phase 1a N=24.
    16	  - Drop-one per cell: oracle ceiling SR over {6 modes} − oracle ceiling SR over
    17	    {5 modes drop P-SoM}, per task, averaged across task pool. Paired bootstrap CI.
    18	  - Pooled meta: DerSimonian-Laird random-effects across 4 cell effect estimates.
    19	  - TOST: two one-sided tests for H0 |θ| ≥ δ rejected vs H1 |θ| < δ at δ=1.0pp.
    20	
    21	PRIMARY GATES (gate paper hook framing R1-R5):
    22	  H1(i)  pooled DL meta on P-SoM drop-one, Holm α=0.05 sig (m=1)
    23	  H1(ii) pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence rejected at δ=1.0pp
    24	  H3(i)  pooled DL meta on |P-text \ P-SoM| axis-1, Holm α=0.05 sig (m=1)
    25	  H3(ii) pooled DL meta on |P-prompt \ P-SoM| axis-2, Holm α=0.05 sig (m=1)
    26	  H2(a)  median cost(P-SoM) within ±10% of median cost(DOM) per cell, replicated
    27	         in ≥3 of 4 cells (transparency K_h2)
    28	
    29	TRANSPARENCY (NOT gating, reported alongside primary):
    30	  K_h1 = 3 of 4 cells individually Holm-sig on drop-one
    31	  K_h3 axis-1 = 3 of 4 cells individually CI > 0
    32	  K_h3 axis-2 = same
    33	
    34	Usage:
    35	    # With actual per-task data:
    36	    python3 scripts/analysis/preregistration_decision_test.py \\
    37	        --per-task-csv results/phantom_paper/per_task_sr.csv \\
    38	        --primary-gate drop_one_pooled_meta_TOST \\
    39	        --TOST-delta-pp 1.0 \\
    40	        --transparency-K_h1 3 --transparency-K_h3 3 \\
    41	        --out results/phantom_paper/preregistration_test_results.json
    42	
    43	    # Smoke test on synthetic data:
    44	    python3 scripts/analysis/preregistration_decision_test.py --synthetic --seed 42
    45	
    46	Input CSV schema (per-task wide format, one row per (cell_id, task_id)):
    47	    cell_id,site,model,task_id,sr_dom,sr_som,sr_vision,sr_ptext,sr_pprompt,sr_psom,
    48	        cost_dom,cost_psom
    49	    cls_B0,classifieds,B0,task_0001,0.0,1.0,0.0,1.0,0.0,1.0,0.043,0.044
    50	    ...
    51	
    52	Each SR cell ∈ {0, 1} (binary per-task evaluator verdict, post-FP-filter).
    53	Costs in any consistent unit (token-normalized $); only ratio used.
    54	
    55	Tied to:
    56	- preregistration.md §2 (H1/H3 hypotheses) + §4 (locked analysis choices) +
    57	  Appendix A 2026-05-13 (codex stress audit propagation)
    58	- osf_lock_manifest.md §2.2 (canonical threshold table)
    59	- run_manifest.yaml (cell scope = 4 Phase 1a cells)
    60	- 笔记 §132 (codex stress audit + scope reframe chronicle)
    61	"""
    62	
    63	from __future__ import annotations
    64	
    65	import argparse
    66	import csv
    67	import hashlib
    68	import json
    69	import logging
    70	import math
    71	import statistics
    72	import sys
    73	from collections import defaultdict
    74	from datetime import datetime, timezone
    75	from pathlib import Path
    76	from typing import Optional
    77	
    78	logger = logging.getLogger("preregistration-test")
    79	
    80	# Phase 1a canonical cells (must match preregistration.md §4 N_cells row)
    81	PHASE_1A_CELLS = [
    82	    ("classifieds", "B0"),
    83	    ("classifieds", "B1"),
    84	    ("reddit", "B0"),
    85	    ("reddit", "B1"),
    86	]
    87	PHANTOM_MODE_KEYS = ["sr_psom", "sr_ptext", "sr_pprompt"]
    88	BASELINE_MODE_KEYS = ["sr_dom", "sr_som", "sr_vision"]
    89	ALL_MODE_KEYS = BASELINE_MODE_KEYS + PHANTOM_MODE_KEYS
    90	
    91	
    92	# ---------------------------------------------------------------------------
    93	# Per-cell drop-one + unique-count computation (paired bootstrap)
    94	# ---------------------------------------------------------------------------
    95	
    96	def _oracle_per_task(task_row: dict, mode_keys: list[str]) -> int:
    97	    """Oracle ceiling for one task = 1 if ANY mode in mode_keys solved it, else 0."""
    98	    return 1 if any(int(task_row[k]) >= 1 for k in mode_keys) else 0
    99	
   100	
   101	def _drop_one_lift_per_cell(cell_tasks: list[dict], drop_mode: str = "sr_psom") -> float:
   102	    """Drop-one oracle ceiling lift for a cell.
   103	
   104	    Returns the mean over the cell's task pool of:
   105	        oracle({all 6 modes}, task) − oracle({all 6 modes} \\ {drop_mode}, task)
   106	
   107	    Result is in [0, 1] (probability units; multiply by 100 for pp).
   108	    """
   109	    full = ALL_MODE_KEYS
   110	    reduced = [k for k in full if k != drop_mode]
   111	    deltas = [_oracle_per_task(t, full) - _oracle_per_task(t, reduced) for t in cell_tasks]
   112	    return sum(deltas) / max(1, len(deltas))
   113	
   114	
   115	def _unique_count_per_cell(cell_tasks: list[dict], axis_mode: str, ref_mode: str = "sr_psom") -> int:
   116	    """|axis_mode \\ ref_mode| = number of tasks where axis_mode solved but ref_mode didn't.
   117	
   118	    Used for H3 axis-1 (axis_mode=sr_ptext) and H3 axis-2 (axis_mode=sr_pprompt).
   119	    """
   120	    return sum(1 for t in cell_tasks
   121	               if int(t[axis_mode]) >= 1 and int(t[ref_mode]) < 1)
   122	
   123	
   124	def _paired_bootstrap(cell_tasks: list[dict], statistic_fn, n_resamples: int = 1000,
   125	                       seed: int = 42) -> tuple[float, float, float, float]:
   126	    """1000-resample paired task-level bootstrap.
   127	
   128	    Returns (point_estimate, ci_lo_95, ci_hi_95, bootstrap_se).
   129	    Resamples task rows with replacement (preserves all modes' SR for that task → paired).
   130	    """
   131	    import random
   132	    rng = random.Random(seed)
   133	    point = statistic_fn(cell_tasks)
   134	    n = len(cell_tasks)
   135	    boot_vals = []
   136	    for _ in range(n_resamples):
   137	        resample = [cell_tasks[rng.randrange(n)] for _ in range(n)]
   138	        boot_vals.append(statistic_fn(resample))
   139	    boot_vals.sort()
   140	    ci_lo = boot_vals[int(0.025 * n_resamples)]
   141	    ci_hi = boot_vals[int(0.975 * n_resamples)]
   142	    se = statistics.stdev(boot_vals) if len(boot_vals) > 1 else 0.0
   143	    return point, ci_lo, ci_hi, se
   144	
   145	
   146	# ---------------------------------------------------------------------------
   147	# DerSimonian-Laird random-effects meta-analysis
   148	# ---------------------------------------------------------------------------
   149	
   150	def dersimonian_laird_meta(effects: list[float], variances: list[float]) -> dict:
   151	    """Pool effect estimates across cells via DerSimonian-Laird random-effects.
   152	
   153	    Args:
   154	        effects: per-cell effect estimates (same scale, e.g., pp or unique-count)
   155	        variances: per-cell variance estimates (= SE^2 from bootstrap)
   156	
   157	    Returns dict with: pooled_effect, pooled_se, pooled_ci_95, Q, I_squared, tau_squared,
   158	                       p_value_two_sided.
   159	
   160	    Method (Higgins & Thompson 2002; DerSimonian & Laird 1986):
   161	      1. Fixed-effects pooled mean θ_FE = Σ(w_i × θ_i) / Σw_i where w_i = 1 / v_i
   162	      2. Q = Σw_i × (θ_i − θ_FE)^2
   163	      3. τ^2 = max(0, (Q − (k − 1)) / (Σw_i − Σw_i^2 / Σw_i))
   164	      4. Random-effects weights w*_i = 1 / (v_i + τ^2)
   165	      5. Pooled θ_RE = Σ(w*_i × θ_i) / Σw*_i; SE_RE = sqrt(1 / Σw*_i)
   166	      6. I^2 = max(0, (Q − (k − 1)) / Q) × 100  (% heterogeneity)
   167	    """
   168	    k = len(effects)
   169	    if k < 2:
   170	        return {"pooled_effect": effects[0] if effects else 0.0,
   171	                "pooled_se": math.sqrt(variances[0]) if variances else 0.0,
   172	                "pooled_ci_95": [None, None],
   173	                "Q": None, "I_squared_pct": None, "tau_squared": None,
   174	                "p_value_two_sided": None, "k": k,
   175	                "note": "k<2: pooling undefined"}
   176	
   177	    w_fe = [1.0 / max(v, 1e-12) for v in variances]
   178	    theta_fe = sum(w * t for w, t in zip(w_fe, effects)) / sum(w_fe)
   179	    Q = sum(w * (t - theta_fe) ** 2 for w, t in zip(w_fe, effects))
   180	    sum_w = sum(w_fe)
   181	    sum_w_sq = sum(w * w for w in w_fe)
   182	    tau_sq_num = Q - (k - 1)
   183	    tau_sq_den = sum_w - (sum_w_sq / sum_w)
   184	    tau_sq = max(0.0, tau_sq_num / max(tau_sq_den, 1e-12))
   185	
   186	    w_re = [1.0 / (v + tau_sq) for v in variances]
   187	    theta_re = sum(w * t for w, t in zip(w_re, effects)) / sum(w_re)
   188	    se_re = math.sqrt(1.0 / sum(w_re))
   189	    ci_lo = theta_re - 1.96 * se_re
   190	    ci_hi = theta_re + 1.96 * se_re
   191	
   192	    z = theta_re / max(se_re, 1e-12)
   193	    # Two-sided p from standard normal (using error function approximation)
   194	    p_two_sided = 2.0 * (1.0 - _phi(abs(z)))
   195	
   196	    i_sq = max(0.0, (Q - (k - 1)) / Q) * 100.0 if Q > 0 else 0.0
   197	
   198	    return {
   199	        "pooled_effect": theta_re,
   200	        "pooled_se": se_re,
   201	        "pooled_ci_95": [ci_lo, ci_hi],
   202	        "Q": Q,
   203	        "Q_df": k - 1,
   204	        "I_squared_pct": i_sq,
   205	        "tau_squared": tau_sq,
   206	        "p_value_two_sided": p_two_sided,
   207	        "z_statistic": z,
   208	        "k": k,
   209	    }
   210	
   211	
   212	def _phi(z: float) -> float:
   213	    """Standard normal CDF using erf approximation."""
   214	    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
   215	
   216	
   217	# ---------------------------------------------------------------------------
   218	# TOST equivalence test
   219	# ---------------------------------------------------------------------------
   220	
   221	def superiority_test(pooled_effect: float, pooled_se: float, threshold: float,
   222	                      alpha: float = 0.05) -> dict:
   223	    """One-sided superiority test: H0: θ ≤ threshold vs H1: θ > threshold.
   224	
   225	    Used for H1(ii) per prereg 2026-05-13 wording revision: "effect is significantly
   226	    ABOVE the +threshold substantive-effect floor". Reject H0 when pooled effect is
   227	    significantly larger than threshold (z = (θ̂ - threshold)/SE > z_α).
   228	
   229	    Args:
   230	        pooled_effect: pooled effect estimate (same units as threshold)
   231	        pooled_se: pooled SE
   232	        threshold: substantive-effect floor (positive; e.g., 1.0pp)
   233	        alpha: one-sided significance level (default 0.05)
   234	
   235	    Returns dict with: z, p_one_sided, threshold, decision.
   236	
   237	    Note: This replaces prior TOST-rejection logic which had ambiguous semantic
   238	    direction ("TOST equivalence rejected" could mean either equivalence-demonstrated
   239	    OR equivalence-not-demonstrated). One-sided superiority is the unambiguous test
   240	    for "effect substantively exceeds threshold".
   241	    """
   242	    z = (pooled_effect - threshold) / max(pooled_se, 1e-12)
   243	    p_one_sided = 1.0 - _phi(z)
   244	    return {
   245	        "threshold": threshold,
   246	        "alpha": alpha,
   247	        "pooled_effect": pooled_effect,
   248	        "pooled_se": pooled_se,
   249	        "z_statistic": z,
   250	        "p_one_sided": p_one_sided,
   251	        "decision": "reject_H0_substantively_above_threshold" if p_one_sided < alpha else "fail_reject",
   252	    }
   253	
   254	
   255	def tost_equivalence(pooled_effect: float, pooled_se: float, delta: float,
   256	                      alpha: float = 0.05) -> dict:
   257	    """Two one-sided tests for equivalence (Schuirmann 1987).
   258	
   259	    Tests H0: |θ| ≥ δ (effect non-equivalent) vs H1: |θ| < δ (effect equivalent).
   260	    Both one-sided tests must reject H0 to demonstrate equivalence.

exec
/bin/bash -lc "nl -ba scripts/analysis/preregistration_decision_test.py | sed -n '260,620p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   260	    Both one-sided tests must reject H0 to demonstrate equivalence.
   261	
   262	    Used in P79 paper-1 as **informational only** (reported alongside H1 superiority
   263	    test, NOT used for H1 PRIMARY gating per 2026-05-13 prereg revision).
   264	    """
   265	    t_lo = (pooled_effect - (-delta)) / max(pooled_se, 1e-12)  # tests θ > -δ
   266	    t_hi = ((+delta) - pooled_effect) / max(pooled_se, 1e-12)  # tests θ < +δ
   267	    p_lo = 1.0 - _phi(t_lo)
   268	    p_hi = 1.0 - _phi(t_hi)
   269	    max_p = max(p_lo, p_hi)
   270	    equivalence_demonstrated = (p_lo < alpha) and (p_hi < alpha)
   271	    return {
   272	        "delta": delta,
   273	        "alpha_per_side": alpha,
   274	        "pooled_effect": pooled_effect,
   275	        "pooled_se": pooled_se,
   276	        "p_lower_bound_test": p_lo,
   277	        "p_upper_bound_test": p_hi,
   278	        "max_p_value": max_p,
   279	        "equivalence_demonstrated": equivalence_demonstrated,
   280	        "decision": "equivalence_demonstrated" if equivalence_demonstrated else "equivalence_not_demonstrated",
   281	    }
   282	
   283	
   284	# ---------------------------------------------------------------------------
   285	# Holm-Bonferroni correction
   286	# ---------------------------------------------------------------------------
   287	
   288	def holm_correct(p_values: list[float], alpha: float = 0.05) -> list[dict]:
   289	    """Holm-Bonferroni step-down correction for a family of m tests.
   290	
   291	    Returns list of dicts (in original order) with: p_raw, p_holm, rejected.
   292	    """
   293	    m = len(p_values)
   294	    if m == 0:
   295	        return []
   296	    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
   297	    results = [None] * m
   298	    prev_adj = 0.0
   299	    for rank, (orig_idx, p) in enumerate(indexed):
   300	        adj = (m - rank) * p
   301	        adj = max(adj, prev_adj)
   302	        adj = min(adj, 1.0)
   303	        results[orig_idx] = {
   304	            "p_raw": p,
   305	            "p_holm": adj,
   306	            "rejected": adj < alpha,
   307	        }
   308	        prev_adj = adj
   309	    return results
   310	
   311	
   312	# ---------------------------------------------------------------------------
   313	# Hypothesis evaluators
   314	# ---------------------------------------------------------------------------
   315	
   316	def evaluate_h1(cells_by_id: dict[str, list[dict]], delta_pp: float = 1.0,
   317	                 magnitude_threshold_pp: float = 1.0, alpha: float = 0.05,
   318	                 transparency_K_h1: int = 3, bootstrap_seed: int = 42) -> dict:
   319	    """H1: P-SoM drop-one oracle ceiling lift > 0, pooled across cells.
   320	
   321	    PRIMARY: pooled DL meta sig at Holm α=0.05 (m=1) + θ_RE ≥ magnitude_threshold_pp
   322	             + TOST equivalence rejected at δ=delta_pp.
   323	    TRANSPARENCY: K_h1 = transparency_K_h1 of N cells individually Holm-sig (m=N).
   324	    """
   325	    per_cell = {}
   326	    effects_pp = []
   327	    variances_pp = []  # variances of per-cell drop-one in pp^2
   328	    per_cell_p_values = []
   329	
   330	    for cell_id, tasks in cells_by_id.items():
   331	        point, ci_lo, ci_hi, se = _paired_bootstrap(
   332	            tasks,
   333	            statistic_fn=lambda t: _drop_one_lift_per_cell(t, drop_mode="sr_psom"),
   334	            seed=bootstrap_seed,
   335	        )
   336	        # Convert to pp
   337	        effect_pp = point * 100.0
   338	        se_pp = se * 100.0
   339	        # Two-sided p from bootstrap normal approx
   340	        z = effect_pp / max(se_pp, 1e-12)
   341	        p_cell = 2.0 * (1.0 - _phi(abs(z)))
   342	        per_cell[cell_id] = {
   343	            "drop_one_lift_pp": effect_pp,
   344	            "ci_95_pp": [ci_lo * 100.0, ci_hi * 100.0],
   345	            "se_pp": se_pp,
   346	            "p_value_two_sided": p_cell,
   347	            "n_tasks": len(tasks),
   348	        }
   349	        effects_pp.append(effect_pp)
   350	        variances_pp.append(se_pp ** 2)
   351	        per_cell_p_values.append(p_cell)
   352	
   353	    # PRIMARY: pooled DL meta + magnitude + superiority test
   354	    meta = dersimonian_laird_meta(effects_pp, variances_pp)
   355	    superiority = superiority_test(meta["pooled_effect"], meta["pooled_se"],
   356	                                     threshold=magnitude_threshold_pp, alpha=alpha)
   357	    # TOST kept for informational reporting (NOT used in H1 gating decision)
   358	    tost_info = tost_equivalence(meta["pooled_effect"], meta["pooled_se"],
   359	                                  delta=delta_pp, alpha=alpha)
   360	
   361	    pooled_sig = meta["p_value_two_sided"] is not None and meta["p_value_two_sided"] < alpha
   362	    magnitude_pass = meta["pooled_effect"] >= magnitude_threshold_pp
   363	    superiority_pass = superiority["decision"] == "reject_H0_substantively_above_threshold"
   364	
   365	    primary_h1_pass = pooled_sig and magnitude_pass and superiority_pass
   366	
   367	    # TRANSPARENCY: K-of-N Holm
   368	    holm_per_cell = holm_correct(per_cell_p_values, alpha=alpha)
   369	    for (cell_id, _), h in zip(per_cell.items(), holm_per_cell):
   370	        per_cell[cell_id]["holm_p"] = h["p_holm"]
   371	        per_cell[cell_id]["individually_holm_sig"] = h["rejected"]
   372	    n_individually_sig = sum(1 for h in holm_per_cell if h["rejected"])
   373	    transparency_pass = n_individually_sig >= transparency_K_h1
   374	
   375	    return {
   376	        "primary_gate": {
   377	            "pooled_meta": meta,
   378	            "magnitude_check": {"pooled_pp": meta["pooled_effect"],
   379	                                 "threshold_pp": magnitude_threshold_pp,
   380	                                 "pass": magnitude_pass},
   381	            "superiority_test": superiority,
   382	            "tost_informational": tost_info,
   383	            "decision": "PASS" if primary_h1_pass else "FAIL",
   384	        },
   385	        "transparency_K_h1": {
   386	            "K": transparency_K_h1,
   387	            "N": len(cells_by_id),
   388	            "n_individually_holm_sig": n_individually_sig,
   389	            "consistent": transparency_pass,
   390	            "note": "transparency-only, NOT a gate on H1 (per prereg 2026-05-13 reclassification)",
   391	        },
   392	        "per_cell": per_cell,
   393	    }
   394	
   395	
   396	def evaluate_h3_axis(cells_by_id: dict[str, list[dict]], axis_mode_key: str,
   397	                      ref_mode_key: str = "sr_psom", min_unique_count: int = 2,
   398	                      alpha: float = 0.05, transparency_K_h3: int = 3,
   399	                      bootstrap_seed: int = 42) -> dict:
   400	    """H3 axis test: |axis_mode \\ ref_mode| > 0, pooled across cells.
   401	
   402	    axis_mode_key examples: sr_ptext (axis-1), sr_pprompt (axis-2).
   403	
   404	    PRIMARY: pooled DL meta on unique-count, CI excluding 0 at Holm α=0.05 (m=1).
   405	    TRANSPARENCY: K_h3 of N cells with bootstrap CI > 0 AND unique-count ≥ min_unique_count.
   406	    """
   407	    per_cell = {}
   408	    effects = []
   409	    variances = []
   410	    per_cell_p_values = []
   411	    per_cell_ci_excludes_zero = []
   412	
   413	    for cell_id, tasks in cells_by_id.items():
   414	        # Statistic: count of tasks where axis solved but ref did not, normalized by task count
   415	        # (using count as the statistic per prereg H3 wording)
   416	        count, ci_lo, ci_hi, se = _paired_bootstrap(
   417	            tasks,
   418	            statistic_fn=lambda t: float(_unique_count_per_cell(t, axis_mode_key, ref_mode_key)),
   419	            seed=bootstrap_seed,
   420	        )
   421	        # Per-cell pass: CI > 0 AND count ≥ min_unique_count (≥2 floor for noise)
   422	        ci_excludes_zero = ci_lo > 0
   423	        count_above_floor = count >= min_unique_count
   424	        per_cell_pass = ci_excludes_zero and count_above_floor
   425	        # Per-cell p from normal approx on count statistic (testing > 0)
   426	        z = count / max(se, 1e-12)
   427	        p_cell = 1.0 - _phi(z)  # one-sided
   428	        per_cell[cell_id] = {
   429	            "unique_count": count,
   430	            "ci_95": [ci_lo, ci_hi],
   431	            "se": se,
   432	            "p_value_one_sided": p_cell,
   433	            "ci_excludes_zero": ci_excludes_zero,
   434	            "count_above_min": count_above_floor,
   435	            "per_cell_pass": per_cell_pass,
   436	            "n_tasks": len(tasks),
   437	        }
   438	        effects.append(count)
   439	        variances.append(se ** 2)
   440	        per_cell_p_values.append(p_cell)
   441	        per_cell_ci_excludes_zero.append(per_cell_pass)
   442	
   443	    # PRIMARY: pooled meta
   444	    meta = dersimonian_laird_meta(effects, variances)
   445	    pooled_ci_lo = meta["pooled_ci_95"][0] if meta["pooled_ci_95"][0] is not None else None
   446	    primary_pass = (meta["p_value_two_sided"] is not None and
   447	                    meta["p_value_two_sided"] < alpha and
   448	                    pooled_ci_lo is not None and pooled_ci_lo > 0)
   449	
   450	    # TRANSPARENCY
   451	    holm_per_cell = holm_correct(per_cell_p_values, alpha=alpha)
   452	    for (cell_id, _), h in zip(per_cell.items(), holm_per_cell):
   453	        per_cell[cell_id]["holm_p"] = h["p_holm"]
   454	        per_cell[cell_id]["individually_holm_sig"] = h["rejected"]
   455	    n_per_cell_pass = sum(per_cell_ci_excludes_zero)
   456	    transparency_pass = n_per_cell_pass >= transparency_K_h3
   457	
   458	    return {
   459	        "axis_mode": axis_mode_key,
   460	        "ref_mode": ref_mode_key,
   461	        "primary_gate": {
   462	            "pooled_meta": meta,
   463	            "ci_excludes_zero": pooled_ci_lo is not None and pooled_ci_lo > 0,
   464	            "decision": "PASS" if primary_pass else "FAIL",
   465	        },
   466	        "transparency_K_h3": {
   467	            "K": transparency_K_h3,
   468	            "N": len(cells_by_id),
   469	            "n_cells_pass": n_per_cell_pass,
   470	            "consistent": transparency_pass,
   471	            "note": "transparency-only, NOT a gate on H3 (per prereg 2026-05-13 reclassification)",
   472	        },
   473	        "per_cell": per_cell,
   474	    }
   475	
   476	
   477	def evaluate_h2_cost(cells_by_id: dict[str, list[dict]], cost_margin_pct: float = 10.0,
   478	                      transparency_K_h2: int = 3) -> dict:
   479	    """H2(a): median cost(P-SoM) within ±cost_margin_pct% of median cost(DOM) per cell,
   480	    replicated in ≥ transparency_K_h2 of N cells.
   481	
   482	    H2(a) test margin is a RELATIVE PERCENTAGE (e.g., ±10% of DOM cost), distinct from
   483	    H1 TOST δ which is an SR percentage-point margin (codex probable concern disambig).
   484	    """
   485	    per_cell = {}
   486	    pass_count = 0
   487	    for cell_id, tasks in cells_by_id.items():
   488	        cost_dom_vals = [float(t["cost_dom"]) for t in tasks if t["cost_dom"]]
   489	        cost_psom_vals = [float(t["cost_psom"]) for t in tasks if t["cost_psom"]]
   490	        if not cost_dom_vals or not cost_psom_vals:
   491	            per_cell[cell_id] = {"per_cell_pass": False, "reason": "missing cost data"}
   492	            continue
   493	        med_dom = statistics.median(cost_dom_vals)
   494	        med_psom = statistics.median(cost_psom_vals)
   495	        rel_diff_pct = (med_psom - med_dom) / max(med_dom, 1e-12) * 100.0
   496	        within_band = abs(rel_diff_pct) <= cost_margin_pct
   497	        per_cell[cell_id] = {
   498	            "median_cost_dom": med_dom,
   499	            "median_cost_psom": med_psom,
   500	            "relative_diff_pct": rel_diff_pct,
   501	            "margin_pct": cost_margin_pct,
   502	            "per_cell_pass": within_band,
   503	        }
   504	        if within_band:
   505	            pass_count += 1
   506	    return {
   507	        "h2a_cost_equivalence": {
   508	            "K": transparency_K_h2,
   509	            "N": len(cells_by_id),
   510	            "n_cells_pass": pass_count,
   511	            "consistent": pass_count >= transparency_K_h2,
   512	            "margin_pct": cost_margin_pct,
   513	        },
   514	        "per_cell": per_cell,
   515	    }
   516	
   517	
   518	# ---------------------------------------------------------------------------
   519	# Framing rule R1-R5 mapper
   520	# ---------------------------------------------------------------------------
   521	
   522	def apply_framing_rule(h1: dict, h2: dict, h3_axis1: dict, h3_axis2: dict) -> dict:
   523	    """Apply preregistration §2 R1-R5 framing rule to test outcomes."""
   524	    h1_pass = h1["primary_gate"]["decision"] == "PASS"
   525	    h2_pass = h2["h2a_cost_equivalence"]["consistent"]
   526	    h3_axis1_pass = h3_axis1["primary_gate"]["decision"] == "PASS"
   527	    h3_axis2_pass = h3_axis2["primary_gate"]["decision"] == "PASS"
   528	
   529	    if h1_pass and h2_pass and h3_axis1_pass and h3_axis2_pass:
   530	        return {"rule": "R1", "framing": "Phantom routing space (2-axis empirical structure)",
   531	                "hook_power": "STRONGEST"}
   532	    if h1_pass and h2_pass and (h3_axis1_pass or h3_axis2_pass):
   533	        return {"rule": "R2", "framing": "Phantom routing space (single-axis empirical structure)",
   534	                "hook_power": "MODERATE-STRONG"}
   535	    if h1_pass and h2_pass and not h3_axis1_pass and not h3_axis2_pass:
   536	        return {"rule": "R3", "framing": "Phantom-SoM is hidden 4th routing arm (workshop-grade R3)",
   537	                "hook_power": "MODERATE"}
   538	    if h1_pass and not h2_pass:
   539	        return {"rule": "R4", "framing": "Phantom-SoM partial drop-in (cost/latency equivalence fails on some site)",
   540	                "hook_power": "WEAK"}
   541	    return {"rule": "R5", "framing": "Paper death scenario — pivot to VWA bug audit OR abandon",
   542	            "hook_power": "n/a"}
   543	
   544	
   545	# ---------------------------------------------------------------------------
   546	# Data loading
   547	# ---------------------------------------------------------------------------
   548	
   549	def load_per_task_csv(csv_path: Path) -> dict[str, list[dict]]:
   550	    """Load per-task CSV, return dict of cell_id → list of task rows."""
   551	    cells_by_id: dict[str, list[dict]] = defaultdict(list)
   552	    with csv_path.open() as f:
   553	        reader = csv.DictReader(f)
   554	        for row in reader:
   555	            cells_by_id[row["cell_id"]].append(row)
   556	    return dict(cells_by_id)
   557	
   558	
   559	def _file_sha256(path: Path) -> str:
   560	    h = hashlib.sha256()
   561	    with path.open("rb") as f:
   562	        for chunk in iter(lambda: f.read(8192), b""):
   563	            h.update(chunk)
   564	    return h.hexdigest()
   565	
   566	
   567	# ---------------------------------------------------------------------------
   568	# Synthetic data generator (24-condition / 4-cell smoke test)
   569	# ---------------------------------------------------------------------------
   570	
   571	def generate_synthetic_per_task(seed: int = 42, n_tasks_per_cell: int = 200,
   572	                                  scenario: str = "r1_pass") -> dict[str, list[dict]]:
   573	    """Generate Phase 1a 4-cell × n_tasks per-task data.
   574	
   575	    Scenarios:
   576	      - r1_pass:   H1 strong (drop-one lift ~2pp pooled), H2 cost equiv hold, H3 both axes pass
   577	      - r3_pass:   H1 holds, H3 both axes fail (workshop fallback framing)
   578	      - r5_fail:   H1 fails (pooled near 0)
   579	    """
   580	    import random
   581	    rng = random.Random(seed)
   582	    cells_by_id = {}
   583	    for site, model in PHASE_1A_CELLS:
   584	        cell_id = f"{site}_{model}"
   585	        # Base per-task SR rates (per mode)
   586	        base_rate = {"sr_dom": 0.30, "sr_som": 0.32, "sr_vision": 0.20,
   587	                     "sr_ptext": 0.31, "sr_pprompt": 0.28, "sr_psom": 0.34}
   588	        # Capability adjustment
   589	        if model == "B1":
   590	            base_rate = {k: v * 0.6 for k, v in base_rate.items()}
   591	        # Scenario
   592	        if scenario == "r5_fail":
   593	            base_rate["sr_psom"] = base_rate["sr_dom"] - 0.01  # nullify hero
   594	        elif scenario == "r3_pass":
   595	            # Hero passes but axes collapse: ptext/pprompt similar to psom
   596	            base_rate["sr_ptext"] = base_rate["sr_psom"] - 0.005
   597	            base_rate["sr_pprompt"] = base_rate["sr_psom"] - 0.005
   598	
   599	        rows = []
   600	        for i in range(n_tasks_per_cell):
   601	            # Per-task latent solvability bias
   602	            bias = rng.uniform(-0.1, 0.1)
   603	            row = {"cell_id": cell_id, "site": site, "model": model,
   604	                   "task_id": f"{cell_id}_t{i:04d}"}
   605	            for mode_key, rate in base_rate.items():
   606	                eff_rate = max(0.0, min(1.0, rate + bias))
   607	                row[mode_key] = 1 if rng.random() < eff_rate else 0
   608	            # Cost: P-SoM ~ DOM cost (regex filter property)
   609	            row["cost_dom"] = 0.040 + rng.uniform(-0.005, 0.005)
   610	            row["cost_psom"] = row["cost_dom"] * (1.0 + rng.uniform(-0.05, 0.05))
   611	            rows.append(row)
   612	        cells_by_id[cell_id] = rows
   613	    return cells_by_id
   614	
   615	
   616	# ---------------------------------------------------------------------------
   617	# Main
   618	# ---------------------------------------------------------------------------
   619	
   620	def main():

exec
/bin/bash -lc "nl -ba scripts/analysis/aggregate_phantom_meta.py | sed -n '1,320p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""[Outcome supporting] Cross-cell meta-analysis — random-effect pooled drop-one
     3	+ heterogeneity (I², τ², Cochran's Q).
     4	
     5	Reads `results/phantom_paper/phantom_lift.csv` (T0a-augmented). For each phantom
     6	arm and oracle comparison, pools per-cell estimates using DerSimonian-Laird
     7	(1986) random-effect model. Within-cell SE derived from bootstrap 95% CI:
     8	
     9	    SE_i ≈ (CI_hi - CI_lo) / (2 × 1.96)
    10	
    11	(Standard normal approximation for symmetric bootstrap CIs; valid when N per
    12	cell is moderate, which holds for N=210-234.)
    13	
    14	Outputs:
    15	- `results/phantom_paper/meta_phantom_lift.csv` (per-arm meta-row)
    16	- `results/phantom_paper/meta_phantom_lift.md`  (paper-ready table)
    17	
    18	T0c of `docs/reference/EVIDENCE_LAYER_AUDIT.md` action queue.
    19	
    20	Why random-effect (RE) over fixed-effect (FE):
    21	- FE assumes single true effect across cells (only sampling variability).
    22	- RE allows true effect heterogeneity across cells (site / model / capability).
    23	- Phantom-SoM's "site-modulated + capability-modulated" framing (paper §7) is
    24	  itself an RE assumption — assuming FE would contradict the paper hook.
    25	- Paired with I² heterogeneity statistic, RE quantifies how much variation is
    26	  between-cell (true differences) vs within-cell (sampling).
    27	
    28	Heterogeneity benchmarks (Higgins & Thompson 2002):
    29	  I² < 25% — low heterogeneity (cells consistent)
    30	  25-50%  — moderate
    31	  50-75%  — substantial
    32	  > 75%   — considerable (strong cell-specific effects)
    33	"""
    34	from __future__ import annotations
    35	
    36	import argparse
    37	import csv
    38	import math
    39	from pathlib import Path
    40	from typing import Optional
    41	
    42	import numpy as np
    43	
    44	try:
    45	    from scipy import stats as sp_stats
    46	    HAS_SCIPY = True
    47	except ImportError:
    48	    HAS_SCIPY = False
    49	
    50	REPO = Path(__file__).resolve().parents[2]
    51	CSV_IN = REPO / "results/phantom_paper/phantom_lift.csv"
    52	DEFAULT_OUT = REPO / "results/phantom_paper/meta_phantom_lift.csv"
    53	
    54	# Arms to meta-pool: (csv prefix, display label, family)
    55	ARMS = [
    56	    ("5_vs_3",        "3→5-mode oracle lift",  "PRIMARY"),
    57	    ("4pdom_vs_3",    "P-text drop-in",        "SECONDARY"),
    58	    ("4psom_vs_3",    "P-SoM drop-in",         "SECONDARY"),
    59	    ("4pprompt_vs_3", "P-prompt drop-in",      "SECONDARY"),
    60	    ("6_vs_3",        "6-mode oracle lift",    "TERTIARY"),
    61	    ("6_vs_5",        "P-prompt incremental",  "TERTIARY"),
    62	]
    63	
    64	
    65	def _f(x):
    66	    if x is None or x == "" or x == "None":
    67	        return None
    68	    return float(x)
    69	
    70	
    71	def derslong_laird_meta(thetas: list, ses: list) -> Optional[dict]:
    72	    """DerSimonian-Laird random-effect meta-analysis.
    73	
    74	    Args:
    75	        thetas: per-cell point estimates (pp scale)
    76	        ses: per-cell SEs (matched to thetas)
    77	
    78	    Returns dict with k / theta_fe / se_fe / theta_re / se_re / ci_lo / ci_hi /
    79	    Q / df / p_Q / tau2 / I2, or None if no data.
    80	    """
    81	    paired = [(t, s) for t, s in zip(thetas, ses) if t is not None and s is not None and s > 0]
    82	    if len(paired) == 0:
    83	        return None
    84	    thetas_arr = np.array([t for t, _ in paired])
    85	    ses_arr = np.array([s for _, s in paired])
    86	    k = len(paired)
    87	
    88	    # Fixed-effect (inverse-variance weighted)
    89	    var_i = ses_arr ** 2
    90	    w_i = 1.0 / var_i
    91	    theta_fe = float(np.sum(w_i * thetas_arr) / np.sum(w_i))
    92	    se_fe = float(math.sqrt(1.0 / np.sum(w_i)))
    93	
    94	    # Cochran's Q (heterogeneity test statistic)
    95	    Q = float(np.sum(w_i * (thetas_arr - theta_fe) ** 2))
    96	    df = k - 1
    97	    if HAS_SCIPY and df > 0:
    98	        p_Q = float(1 - sp_stats.chi2.cdf(Q, df))
    99	    else:
   100	        p_Q = None
   101	
   102	    # τ² (between-study variance, DL estimator)
   103	    if df > 0:
   104	        sum_w = float(np.sum(w_i))
   105	        sum_w2 = float(np.sum(w_i ** 2))
   106	        C = sum_w - sum_w2 / sum_w
   107	        tau2 = max(0.0, (Q - df) / C) if C > 0 else 0.0
   108	    else:
   109	        tau2 = 0.0
   110	
   111	    # I² (% variation due to heterogeneity, Higgins & Thompson 2002)
   112	    if Q > 0 and df > 0:
   113	        I2 = max(0.0, (Q - df) / Q) * 100.0
   114	    else:
   115	        I2 = 0.0
   116	
   117	    # Random-effect estimate (using w*_i = 1 / (var_i + tau2))
   118	    var_star = var_i + tau2
   119	    w_star = 1.0 / var_star
   120	    theta_re = float(np.sum(w_star * thetas_arr) / np.sum(w_star))
   121	    se_re = float(math.sqrt(1.0 / np.sum(w_star)))
   122	    ci_lo = theta_re - 1.96 * se_re
   123	    ci_hi = theta_re + 1.96 * se_re
   124	
   125	    # RE vs 0 z-test (single-side: pooled effect > 0)
   126	    z = theta_re / se_re if se_re > 0 else None
   127	    if HAS_SCIPY and z is not None:
   128	        p_re = float(1 - sp_stats.norm.cdf(z))
   129	    else:
   130	        p_re = None
   131	
   132	    return {
   133	        "k": k,
   134	        "theta_fe": theta_fe,
   135	        "se_fe": se_fe,
   136	        "theta_re": theta_re,
   137	        "se_re": se_re,
   138	        "ci_lo": ci_lo,
   139	        "ci_hi": ci_hi,
   140	        "z_re": z,
   141	        "p_re_one_sided": p_re,
   142	        "Q": Q,
   143	        "df": df,
   144	        "p_Q": p_Q,
   145	        "tau2": tau2,
   146	        "I2": I2,
   147	    }
   148	
   149	
   150	def i_squared_label(I2: float) -> str:
   151	    if I2 < 25:
   152	        return "low"
   153	    if I2 < 50:
   154	        return "moderate"
   155	    if I2 < 75:
   156	        return "substantial"
   157	    return "considerable"
   158	
   159	
   160	# F08 audit fix 2026-05-09: B8 preregistration lock requires N_common >= 10
   161	# per cell for inclusion in random-effects meta. Cells below floor are
   162	# excluded with reason logged. See `preregistration.md §4` row "Heterogeneity
   163	# (random-effects, Q, I², τ²) pre-spec".
   164	MIN_N_COMMON_FOR_META = 10
   165	
   166	
   167	def load_per_cell_data(arm_code: str) -> tuple[list[dict], list[dict]]:
   168	    """Per-cell point + SE for a given arm, with B8 N>=10 floor enforced.
   169	
   170	    Returns (included, excluded) cell-row dicts. SE_i derived from bootstrap
   171	    CI: SE = (CI_hi - CI_lo) / (2 * 1.96).
   172	    """
   173	    if not CSV_IN.exists():
   174	        raise SystemExit(f"missing {CSV_IN}; run aggregate_phantom_lift.py first")
   175	    included, excluded = [], []
   176	    with CSV_IN.open() as f:
   177	        reader = csv.DictReader(f)
   178	        for r in reader:
   179	            theta = _f(r.get(f"lift_{arm_code}_pp"))
   180	            ci_lo = _f(r.get(f"lift_{arm_code}_ci95_lo_pp"))
   181	            ci_hi = _f(r.get(f"lift_{arm_code}_ci95_hi_pp"))
   182	            n_common = _f(r.get("n_common"))
   183	            cell_label = f"{r.get('baseline','?')} {r.get('site','?')}"
   184	            if theta is None or ci_lo is None or ci_hi is None:
   185	                continue
   186	            se = (ci_hi - ci_lo) / (2 * 1.96)
   187	            if se <= 0:
   188	                continue
   189	            row = {
   190	                "baseline": r["baseline"],
   191	                "site": r["site"],
   192	                "n_common": int(n_common) if n_common is not None else None,
   193	                "theta": theta,
   194	                "se": se,
   195	                "ci_lo": ci_lo,
   196	                "ci_hi": ci_hi,
   197	            }
   198	            if n_common is not None and n_common < MIN_N_COMMON_FOR_META:
   199	                row["exclude_reason"] = f"N_common={int(n_common)} < {MIN_N_COMMON_FOR_META} (B8 lock)"
   200	                excluded.append(row)
   201	                continue
   202	            included.append(row)
   203	    return included, excluded
   204	
   205	
   206	def main() -> int:
   207	    ap = argparse.ArgumentParser()
   208	    ap.add_argument("--output", default=str(DEFAULT_OUT))
   209	    args = ap.parse_args()
   210	
   211	    out = Path(args.output)
   212	    out.parent.mkdir(parents=True, exist_ok=True)
   213	
   214	    meta_rows = []
   215	    arm_per_cell: dict = {}
   216	    arm_excluded: dict = {}  # F08: track B8 N>=10 floor exclusions
   217	    for code, label, family in ARMS:
   218	        cells, excluded = load_per_cell_data(code)
   219	        arm_per_cell[code] = cells
   220	        arm_excluded[code] = excluded
   221	        if excluded:
   222	            for ex in excluded:
   223	                print(
   224	                    f"  [B8 floor] arm={code} excluded "
   225	                    f"{ex['baseline']} {ex['site']}: {ex['exclude_reason']}"
   226	                )
   227	        if not cells:
   228	            continue
   229	        meta = derslong_laird_meta(
   230	            [c["theta"] for c in cells],
   231	            [c["se"] for c in cells],
   232	        )
   233	        if meta is None:
   234	            continue
   235	        meta_rows.append({
   236	            "arm_code": code,
   237	            "arm_label": label,
   238	            "family": family,
   239	            "k_cells": meta["k"],
   240	            "cells": "; ".join(f"{c['baseline']} {c['site']}" for c in cells),
   241	            "excluded_b8": "; ".join(
   242	                f"{c['baseline']} {c['site']} (N={c['n_common']})"
   243	                for c in excluded
   244	            ) or "none",
   245	            **{k: round(v, 6) if isinstance(v, float) else v
   246	               for k, v in meta.items() if k != "k"},
   247	        })
   248	
   249	    # CSV
   250	    with out.open("w", newline="") as f:
   251	        if meta_rows:
   252	            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
   253	            w.writeheader()
   254	            w.writerows(meta_rows)
   255	    print(f"wrote {out} ({len(meta_rows)} arms)")
   256	
   257	    # Markdown
   258	    md = out.with_suffix(".md")
   259	    n_arms = len(meta_rows)
   260	    n_primary = sum(1 for r in meta_rows if r["family"] == "PRIMARY")
   261	    n_secondary = sum(1 for r in meta_rows if r["family"] == "SECONDARY")
   262	    n_tertiary = sum(1 for r in meta_rows if r["family"] == "TERTIARY")
   263	    lines = [
   264	        "# Phantom routing lift — cross-cell meta-analysis (random-effect pooled)",
   265	        "",
   266	        "DerSimonian-Laird (1986) random-effect meta-analysis pools per-cell",
   267	        "drop-one and oracle-lift estimates across all available cells. Within-cell",
   268	        "SE derived from bootstrap 95% CI as `(CI_hi - CI_lo) / (2 × 1.96)`.",
   269	        "",
   270	        "Heterogeneity statistics:",
   271	        "- **I²** — % variation due to between-cell heterogeneity (vs sampling).",
   272	        "  Benchmarks: <25% low / 25-50% moderate / 50-75% substantial / >75% considerable.",
   273	        "- **τ²** — between-cell variance (DL estimator); 0 = no heterogeneity.",
   274	        "- **Cochran's Q** — homogeneity test; small p_Q rejects assumption that",
   275	        "  cells share single true effect.",
   276	        "",
   277	        f"Cells included per arm — see `cells` col. Arms: {n_arms} pooled "
   278	        f"(PRIMARY={n_primary}, SECONDARY={n_secondary}, TERTIARY={n_tertiary}).",
   279	        "",
   280	        "## Pooled estimates per arm",
   281	        "",
   282	        "| Family | Arm | k cells | Random-effect pp | 95% CI | SE | z | p (1-sided) | I² | τ² | Q | df | p_Q | sig (Holm-corrected family) |",
   283	        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
   284	    ]
   285	
   286	    # Apply Holm-Bonferroni within each family for the meta-pooled p-value
   287	    # (1-sided z test on RE estimate)
   288	    by_family: dict = {}
   289	    for r in meta_rows:
   290	        by_family.setdefault(r["family"], []).append(r)
   291	    for family, family_rows in by_family.items():
   292	        ps = [r.get("p_re_one_sided") for r in family_rows]
   293	        # Holm-Bonferroni step-down
   294	        indexed = [(i, p) for i, p in enumerate(ps) if p is not None]
   295	        indexed.sort(key=lambda x: x[1])
   296	        m = len(indexed)
   297	        adj = [None] * len(ps)
   298	        prev = 0.0
   299	        for k, (i, p) in enumerate(indexed):
   300	            a = min(1.0, max(prev, p * (m - k)))
   301	            adj[i] = a
   302	            prev = a
   303	        for r, a in zip(family_rows, adj):
   304	            r["p_re_holm"] = round(a, 6) if a is not None else None
   305	
   306	    def _fmt(v, spec=".4f"):
   307	        if v is None:
   308	            return "—"
   309	        return f"{v:{spec}}"
   310	
   311	    for r in meta_rows:
   312	        sig = "✅" if (r.get("p_re_holm") is not None and r["p_re_holm"] < 0.05) else "❌"
   313	        i2_lab = i_squared_label(r["I2"]) if r["k_cells"] > 1 else "n/a (k=1)"
   314	        p_re_holm_str = _fmt(r.get("p_re_holm")) if r["k_cells"] >= 1 else "—"
   315	        lines.append(
   316	            f"| {r['family']} | {r['arm_label']} | {r['k_cells']} | "
   317	            f"+{r['theta_re']:.2f}pp | "
   318	            f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}] | "
   319	            f"{_fmt(r['se_re'], '.3f')} | "
   320	            f"{_fmt(r['z_re'], '.2f')} | "

exec
/bin/bash -lc "nl -ba scripts/analysis/aggregate_phantom_lift.py | sed -n '1,360p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""[Outcome 0c + 0d] Outcome dimension — routing oracle lift and task-pool Jaccard.
     3	
     4	Outputs:
     5	- results/phantom_paper/phantom_lift.csv
     6	- results/phantom_paper/phantom_lift.md
     7	
     8	Outcome 0c: 3-mode to 4/5-mode oracle lift and significance tests.
     9	Outcome 0d: P-text↔P-SoM task-pool Jaccard Scenario C sentinel.
    10	
    11	See docs/checkpoints/paper_planning.md §3 Outcome dimension framework.
    12	
    13	Aggregate phantom routing lift across (baseline, site) cells.
    14	
    15	For each cell with all 5 modes (DOM / SoM / Vision / P-text / P-SoM) present:
    16	  - Compute 3-mode oracle ceiling (DOM ∪ SoM ∪ Vision)
    17	  - Compute 5-mode oracle ceiling (+ P-text + P-SoM)
    18	  - Routing lift = 5-mode - 3-mode oracle SR (pp)
    19	  - 95% bootstrap CI on lift (n=1000 task resamples)
    20	  - Decomposition: P-text-only, P-SoM-only, both-add-same contributions
    21	
    22	Outputs (results/phantom_paper/):
    23	  - phantom_lift.csv          (one row per (baseline, site, decomposition))
    24	  - phantom_lift_summary.md   (paper-ready table for Section 1/4 hook)
    25	
    26	Usage:
    27	    python3 scripts/analysis/aggregate_phantom_lift.py [--cells <baseline:site:...>]
    28	
    29	Default cells = paper-grade clean B0 cls/red. B1 cells included if all 5 modes
    30	have data (>= 50 ep each). Partial cells use the common observed-task universe.
    31	"""
    32	from __future__ import annotations
    33	
    34	import argparse
    35	import csv
    36	import json
    37	import math
    38	import os
    39	import re
    40	import warnings
    41	from pathlib import Path
    42	from typing import Optional
    43	
    44	import numpy as np
    45	
    46	try:
    47	    from scripts.analysis.lib.run_registry import get_cells
    48	except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    49	    import sys
    50	    sys.path.append(str(Path(__file__).resolve().parents[2]))
    51	    from scripts.analysis.lib.run_registry import get_cells
    52	
    53	try:
    54	    from scipy import stats as sp_stats
    55	    HAS_SCIPY = True
    56	except ImportError:
    57	    HAS_SCIPY = False
    58	
    59	REPO = Path(__file__).resolve().parents[2]
    60	
    61	
    62	# Cell registry: (baseline, site, expected_N, run_paths_per_mode)
    63	def _build_cells(grade_filter: list | None = None) -> list[dict]:
    64	    """Build aggregator cell list. F01 audit: respects grade_filter
    65	    (default = `paper-grade` only). Pass a list to override (e.g. for
    66	    legacy `archived` data in Appendix-D sensitivity figure)."""
    67	    out: list[dict] = []
    68	    for baseline in ("B0", "B1"):
    69	        for site in ("classifieds", "reddit"):
    70	            specs = get_cells(baseline=baseline, site=site, grade=grade_filter)
    71	            if not specs:
    72	                continue
    73	            out.append({
    74	                "baseline": baseline,
    75	                "site": site,
    76	                "n_expected": specs[0].expected_n,
    77	                "modes": {cell.mode: cell.episodes_dir for cell in specs},
    78	            })
    79	    return out
    80	
    81	
    82	# F01 audit 2026-05-09: env override `P79_AGGREGATOR_GRADE` lets the
    83	# Appendix-D legacy sensitivity figure pull `archived` data while the
    84	# default `paper-grade` filter remains the paper-claim path.
    85	_GRADE_OVERRIDE = os.environ.get("P79_AGGREGATOR_GRADE", "")
    86	_GRADE_LIST = [g.strip() for g in _GRADE_OVERRIDE.split(",") if g.strip()] or None
    87	CELLS = _build_cells(_GRADE_LIST)
    88	
    89	MIN_EP_FOR_CELL = 50  # skip cells where any present mode has < 50 ep (too partial)
    90	
    91	
    92	def load(d: Path) -> tuple[set[int], set[int]]:
    93	    """Returns (succ_set, observed_set)."""
    94	    s, o = set(), set()
    95	    if not d.exists():
    96	        return s, o
    97	    # F05 audit fix 2026-05-09: track corrupt summary count instead of
    98	    # silently dropping; warn at end of cell. Set P79_STRICT=1 to fail.
    99	    n_corrupt = 0
   100	    for p in sorted(d.glob("*_summary_v2.json")):
   101	        m = re.search(r"task_(\d+)", p.name)
   102	        if not m:
   103	            continue
   104	        tid = int(m.group(1))
   105	        o.add(tid)
   106	        try:
   107	            rec = json.loads(p.read_text())
   108	        except Exception as _e:
   109	            n_corrupt += 1
   110	            continue
   111	        if rec.get("adjusted_success", rec.get("success", False)):
   112	            s.add(tid)
   113	    if n_corrupt > 0:
   114	        msg = (
   115	            f"  [F05] {d}: {n_corrupt} corrupt summary file(s) skipped "
   116	            "(could change oracle union + pp lift). Set P79_STRICT=1 to fail."
   117	        )
   118	        if os.environ.get("P79_STRICT", "").lower() in ("1", "true", "yes"):
   119	            raise RuntimeError(msg)
   120	        print(f"WARNING: {msg}")
   121	    return s, o
   122	
   123	
   124	def bootstrap_lift_ci(in_3: np.ndarray, in_5: np.ndarray, B: int = 1000, seed: int = 42
   125	                      ) -> tuple[float, float]:
   126	    """Bootstrap 95% CI on (5-mode oracle SR - 3-mode oracle SR)."""
   127	    n = len(in_3)
   128	    rng = np.random.default_rng(seed)
   129	    lifts = np.empty(B)
   130	    for b in range(B):
   131	        idx = rng.integers(0, n, size=n)
   132	        lifts[b] = 100 * (int(in_5[idx].sum()) - int(in_3[idx].sum())) / n
   133	    return float(np.quantile(lifts, 0.025)), float(np.quantile(lifts, 0.975))
   134	
   135	
   136	def cohen_h(p1: float, p2: float) -> float:
   137	    """Cohen's h effect size between two proportions p1, p2 ∈ [0, 1].
   138	
   139	    h = 2 * (arcsin(√p1) - arcsin(√p2))
   140	
   141	    Interpretation: |h|<0.2 small, 0.2-0.5 medium, 0.5-0.8 large, >0.8 huge.
   142	    Sign indicates direction (p1 > p2 → h > 0).
   143	    """
   144	    p1 = max(0.0, min(1.0, p1))
   145	    p2 = max(0.0, min(1.0, p2))
   146	    return 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
   147	
   148	
   149	def cohen_h_label(h: float) -> str:
   150	    a = abs(h)
   151	    if a < 0.2:
   152	        return "small"
   153	    if a < 0.5:
   154	        return "medium"
   155	    if a < 0.8:
   156	        return "large"
   157	    return "huge"
   158	
   159	
   160	def wilcoxon_signed_rank(in_a: np.ndarray, in_b: np.ndarray) -> tuple[Optional[float], Optional[float]]:
   161	    """Wilcoxon signed-rank test on paired binary task outcomes (a vs b).
   162	
   163	    For binary outcomes diff ∈ {-1, 0, +1}; scipy drops zero diffs. When set b
   164	    ⊇ set a (e.g. 5-mode oracle ⊇ 3-mode oracle), all non-zero diffs are
   165	    positive (b solves task that a doesn't) → test reduces to one-sided
   166	    binomial (sign test). Returns (statistic, p_two_sided) or (None, None) if
   167	    scipy unavailable / undefined.
   168	    """
   169	    if not HAS_SCIPY:
   170	        return None, None
   171	    diffs = in_b.astype(int) - in_a.astype(int)
   172	    nonzero = diffs[diffs != 0]
   173	    if len(nonzero) == 0:
   174	        return None, 1.0  # no difference, p = 1
   175	    try:
   176	        with warnings.catch_warnings():
   177	            warnings.simplefilter("ignore")
   178	            stat, p = sp_stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
   179	        return float(stat), float(p)
   180	    except Exception:
   181	        return None, None
   182	
   183	
   184	def mcnemar_exact_one_sided(in_a: np.ndarray, in_b: np.ndarray) -> Optional[float]:
   185	    """McNemar exact one-sided p-value: H1 = b > a (b adds tasks a misses).
   186	
   187	    For monotonic case (b ⊇ a), discordant b-only count = sum(b - a > 0),
   188	    a-only count = 0. Exact binomial: p = 0.5^(b_only).
   189	    """
   190	    if not HAS_SCIPY:
   191	        return None
   192	    a = in_a.astype(int); b = in_b.astype(int)
   193	    a_only = int(((a > b)).sum())
   194	    b_only = int(((b > a)).sum())
   195	    n_disc = a_only + b_only
   196	    if n_disc == 0:
   197	        return 1.0
   198	    # one-sided: H1 = b > a
   199	    return float(sp_stats.binom.cdf(a_only, n_disc, 0.5))
   200	
   201	
   202	def bootstrap_unique_count_ci(in_a: np.ndarray, in_b: np.ndarray,
   203	                              B: int = 1000, seed: int = 42, ci: float = 0.95
   204	                              ) -> tuple[int, float, float]:
   205	    """Bootstrap CI on |a ∖ b| count: tasks where a solves but b doesn't.
   206	
   207	    H3 structural claim test: arm a contributes tasks NOT solved by arm b.
   208	    If lower CI bound > 0, "a has unique non-overlap with b" sig at 1-ci level.
   209	
   210	    Used per-cell for:
   211	      P-text ∖ P-SoM unique count (axis 1 structural evidence)
   212	      P-prompt ∖ P-SoM unique count (axis 2 structural evidence)
   213	    """
   214	    n = len(in_a)
   215	    if n == 0 or len(in_b) != n:
   216	        return 0, 0.0, 0.0
   217	    a = in_a.astype(bool)
   218	    b = in_b.astype(bool)
   219	    observed = int((a & ~b).sum())
   220	    rng = np.random.default_rng(seed)
   221	    counts = np.empty(B)
   222	    for r in range(B):
   223	        idx = rng.integers(0, n, size=n)
   224	        counts[r] = int((a[idx] & ~b[idx]).sum())
   225	    alpha = (1 - ci) / 2
   226	    return observed, float(np.quantile(counts, alpha)), float(np.quantile(counts, 1 - alpha))
   227	
   228	
   229	def bootstrap_tost_equivalence_p(in_a: np.ndarray, in_b: np.ndarray,
   230	                                  delta_pp: float = 1.0, B: int = 1000, seed: int = 42
   231	                                  ) -> Optional[float]:
   232	    """Bootstrap TOST (Two One-Sided Tests) p-value for **equivalence** test.
   233	
   234	    H0: |true lift| >= δ          (effect is meaningful in either direction)
   235	    H1: |true lift| < δ           (effect equivalent to zero within margin)
   236	
   237	    Two one-sided tests:
   238	      H0_lower:  lift <= -δ  → reject if bootstrap dist mostly above -δ
   239	                              p_lower = P(boot_lift <= -δ); small p_lower
   240	                              ⇒ evidence rejects "effect <= -δ"
   241	      H0_upper:  lift >= +δ  → reject if bootstrap dist mostly below +δ
   242	                              p_upper = P(boot_lift >= +δ); small p_upper
   243	                              ⇒ evidence rejects "effect >= +δ"
   244	
   245	    TOST p = max(p_lower, p_upper).
   246	    **If max(p_lower, p_upper) < α, equivalence is ACCEPTED** — both
   247	    one-sided tests reject, so the effect is bounded inside (-δ, +δ).
   248	
   249	    F03 audit fix 2026-05-09: δ default = 1.0pp (was 0.5). Matches
   250	    `preregistration.md §4` lock "TOST equivalence margin δ = 1.0pp".
   251	
   252	    F04 audit fix 2026-05-09: renamed from `bootstrap_tost_p`; clarified
   253	    docstring (previous wording said "equivalence rejected when max < α"
   254	    which inverts the conclusion). Strong positive lift gives p_upper≈1
   255	    correctly (effect is outside +δ equivalence margin), so equivalence
   256	    is correctly NOT accepted.
   257	
   258	    For the **nonzero / one-sided directional** test (the phantom-lift
   259	    hypothesis "lift > 0"), use `bootstrap_one_sided_nonzero_p()` below.
   260	    """
   261	    if len(in_a) != len(in_b):
   262	        return None
   263	    n = len(in_a)
   264	    if n == 0:
   265	        return None
   266	    rng = np.random.default_rng(seed)
   267	    lifts = np.empty(B)
   268	    for b in range(B):
   269	        idx = rng.integers(0, n, size=n)
   270	        lifts[b] = 100 * (int(in_b[idx].sum()) - int(in_a[idx].sum())) / n
   271	    p_lower = float(np.mean(lifts <= -delta_pp))
   272	    p_upper = float(np.mean(lifts >= delta_pp))
   273	    return max(p_lower, p_upper)
   274	
   275	
   276	# F04 audit fix 2026-05-09: alias preserves backward-compat callers; new
   277	# code should use the renamed `bootstrap_tost_equivalence_p()`.
   278	bootstrap_tost_p = bootstrap_tost_equivalence_p
   279	
   280	
   281	def bootstrap_one_sided_nonzero_p(in_a: np.ndarray, in_b: np.ndarray,
   282	                                   B: int = 1000, seed: int = 42,
   283	                                   alternative: str = "greater"
   284	                                   ) -> Optional[float]:
   285	    """Bootstrap one-sided p-value for the directional phantom-lift claim.
   286	
   287	    H0: lift = 0     (no phantom-routing benefit)
   288	    H1: lift > 0     (alternative='greater', default — primary paper claim)
   289	       or lift < 0   (alternative='less')
   290	
   291	    p = fraction of bootstrap resamples where lift contradicts H1
   292	        (alternative='greater' → fraction with lift <= 0)
   293	        (alternative='less' → fraction with lift >= 0)
   294	
   295	    F04 audit fix 2026-05-09: added as the correct test for the paper's
   296	    phantom-lift > 0 claim; the equivalence-style TOST in
   297	    `bootstrap_tost_equivalence_p()` is for the separate "lift is bounded
   298	    inside ±δ" claim and should NOT be substituted for nonzero detection.
   299	    """
   300	    if len(in_a) != len(in_b) or len(in_a) == 0:
   301	        return None
   302	    n = len(in_a)
   303	    rng = np.random.default_rng(seed)
   304	    lifts = np.empty(B)
   305	    for b in range(B):
   306	        idx = rng.integers(0, n, size=n)
   307	        lifts[b] = 100 * (int(in_b[idx].sum()) - int(in_a[idx].sum())) / n
   308	    if alternative == "greater":
   309	        return float(np.mean(lifts <= 0.0))
   310	    elif alternative == "less":
   311	        return float(np.mean(lifts >= 0.0))
   312	    else:
   313	        raise ValueError(f"alternative must be 'greater' or 'less', got {alternative}")
   314	
   315	
   316	def bonferroni_adjust(pvals: list) -> list:
   317	    """Bonferroni: p_adj = min(1, m * p_raw); None entries pass-through."""
   318	    m = sum(1 for p in pvals if p is not None)
   319	    if m == 0:
   320	        return list(pvals)
   321	    return [min(1.0, m * p) if p is not None else None for p in pvals]
   322	
   323	
   324	def holm_bonferroni_adjust(pvals: list) -> list:
   325	    """Holm-Bonferroni step-down (Holm 1979): less conservative than Bonferroni
   326	    while still controlling family-wise error rate at α.
   327	
   328	    Sort non-None p-values ascending; the k-th smallest gets multiplied by
   329	    (m - k + 1) where m = number of non-None tests; running max enforces
   330	    monotonicity.
   331	    """
   332	    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
   333	    indexed.sort(key=lambda x: x[1])
   334	    m = len(indexed)
   335	    out: list = [None] * len(pvals)
   336	    prev = 0.0
   337	    for k, (i, p) in enumerate(indexed):
   338	        adj = min(1.0, max(prev, p * (m - k)))
   339	        out[i] = adj
   340	        prev = adj
   341	    return out
   342	
   343	
   344	def bh_fdr_adjust(pvals: list) -> list:
   345	    """Benjamini-Hochberg FDR adjusted q-values (BH 1995).
   346	
   347	    Less conservative than FWER methods; controls expected proportion of false
   348	    discoveries among rejections rather than family-wise error rate.
   349	    """
   350	    indexed = [(i, p) for i, p in enumerate(pvals) if p is not None]
   351	    indexed.sort(key=lambda x: x[1])
   352	    m = len(indexed)
   353	    out: list = [None] * len(pvals)
   354	    prev = 1.0
   355	    for k in range(m - 1, -1, -1):
   356	        i, p = indexed[k]
   357	        rank = k + 1
   358	        adj = min(prev, p * m / rank)
   359	        out[i] = min(1.0, adj)
   360	        prev = out[i]

exec
/bin/bash -lc "nl -ba scripts/analysis/preregistration_decision_test.py | sed -n '620,900p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   620	def main():
   621	    p = argparse.ArgumentParser()
   622	    p.add_argument("--per-task-csv",
   623	                   help="Per-task CSV path (cell_id, site, model, task_id, sr_*, cost_*)")
   624	    p.add_argument("--synthetic", action="store_true",
   625	                   help="Run smoke test on synthetic 4-cell × 200-task data")
   626	    p.add_argument("--scenario", default="r1_pass",
   627	                   choices=["r1_pass", "r3_pass", "r5_fail"])
   628	    p.add_argument("--seed", type=int, default=42)
   629	    p.add_argument("--primary-gate", default="drop_one_pooled_meta_TOST",
   630	                   help="Primary gate flavor (informational; method is fixed in this rewrite)")
   631	    p.add_argument("--TOST-delta-pp", type=float, default=1.0,
   632	                   help="TOST equivalence margin in SR pp (default 1.0 per prereg lock)")
   633	    p.add_argument("--H1-magnitude-pp", type=float, default=1.0,
   634	                   help="H1 pooled magnitude threshold (default 1.0pp per prereg lock)")
   635	    p.add_argument("--H2-cost-margin-pct", type=float, default=10.0,
   636	                   help="H2(a) cost equivalence margin in % (default 10%% per prereg lock)")
   637	    p.add_argument("--H3-min-unique-count", type=int, default=2,
   638	                   help="H3 per-cell unique-count noise floor (default 2 tasks)")
   639	    p.add_argument("--transparency-K_h1", type=int, default=3,
   640	                   help="K_h1 transparency ratio cells count (default 3 of 4)")
   641	    p.add_argument("--transparency-K_h3", type=int, default=3,
   642	                   help="K_h3 transparency ratio cells count per axis (default 3 of 4)")
   643	    p.add_argument("--transparency-K_h2", type=int, default=3,
   644	                   help="H2 transparency cells count (default 3 of 4)")
   645	    p.add_argument("--alpha", type=float, default=0.05)
   646	    p.add_argument("--out", default="-", help="Output JSON path (- = stdout)")
   647	    args = p.parse_args()
   648	
   649	    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
   650	
   651	    # Load data
   652	    if args.synthetic:
   653	        cells_by_id = generate_synthetic_per_task(seed=args.seed, scenario=args.scenario)
   654	        input_sha = f"synthetic:{args.scenario}:{args.seed}"
   655	        logger.info(f"Synthetic mode: {len(cells_by_id)} cells, scenario={args.scenario}")
   656	    else:
   657	        if not args.per_task_csv:
   658	            logger.error("Must provide --per-task-csv or --synthetic")
   659	            sys.exit(2)
   660	        csv_path = Path(args.per_task_csv)
   661	        cells_by_id = load_per_task_csv(csv_path)
   662	        input_sha = _file_sha256(csv_path)
   663	        logger.info(f"Loaded {len(cells_by_id)} cells from {csv_path} (sha256={input_sha[:12]}...)")
   664	
   665	    if len(cells_by_id) < 2:
   666	        logger.error(f"Need ≥2 cells for pooled meta; got {len(cells_by_id)}")
   667	        sys.exit(2)
   668	
   669	    # Evaluate hypotheses
   670	    h1 = evaluate_h1(cells_by_id, delta_pp=args.TOST_delta_pp,
   671	                      magnitude_threshold_pp=args.H1_magnitude_pp,
   672	                      alpha=args.alpha, transparency_K_h1=args.transparency_K_h1,
   673	                      bootstrap_seed=args.seed)
   674	    h2 = evaluate_h2_cost(cells_by_id, cost_margin_pct=args.H2_cost_margin_pct,
   675	                           transparency_K_h2=args.transparency_K_h2)
   676	    h3_axis1 = evaluate_h3_axis(cells_by_id, axis_mode_key="sr_ptext",
   677	                                  ref_mode_key="sr_psom",
   678	                                  min_unique_count=args.H3_min_unique_count,
   679	                                  alpha=args.alpha,
   680	                                  transparency_K_h3=args.transparency_K_h3,
   681	                                  bootstrap_seed=args.seed)
   682	    h3_axis2 = evaluate_h3_axis(cells_by_id, axis_mode_key="sr_pprompt",
   683	                                  ref_mode_key="sr_psom",
   684	                                  min_unique_count=args.H3_min_unique_count,
   685	                                  alpha=args.alpha,
   686	                                  transparency_K_h3=args.transparency_K_h3,
   687	                                  bootstrap_seed=args.seed)
   688	    framing = apply_framing_rule(h1, h2, h3_axis1, h3_axis2)
   689	
   690	    result = {
   691	        "captured_at": datetime.now(timezone.utc).isoformat(),
   692	        "scope": "Phase 1a 24-condition / 4-cell statistical analysis",
   693	        "n_cells": len(cells_by_id),
   694	        "n_tasks_total": sum(len(t) for t in cells_by_id.values()),
   695	        "cell_ids": list(cells_by_id.keys()),
   696	        "input_data_sha256": input_sha,
   697	        "thresholds": {
   698	            "primary_gate_method": "pooled_DerSimonian_Laird_meta + TOST + magnitude",
   699	            "TOST_delta_pp": args.TOST_delta_pp,
   700	            "H1_magnitude_pp": args.H1_magnitude_pp,
   701	            "H2_cost_margin_pct": args.H2_cost_margin_pct,
   702	            "H3_min_unique_count": args.H3_min_unique_count,
   703	            "transparency_K_h1": args.transparency_K_h1,
   704	            "transparency_K_h3": args.transparency_K_h3,
   705	            "transparency_K_h2": args.transparency_K_h2,
   706	            "alpha": args.alpha,
   707	        },
   708	        "H1_psom_drop_one": h1,
   709	        "H2_cost_equivalence": h2,
   710	        "H3_axis1_ptext_unique": h3_axis1,
   711	        "H3_axis2_pprompt_unique": h3_axis2,
   712	        "framing_rule": framing,
   713	        "primary_gate_summary": {
   714	            "H1": h1["primary_gate"]["decision"],
   715	            "H2": "PASS" if h2["h2a_cost_equivalence"]["consistent"] else "FAIL",
   716	            "H3_axis1": h3_axis1["primary_gate"]["decision"],
   717	            "H3_axis2": h3_axis2["primary_gate"]["decision"],
   718	        },
   719	        "transparency_summary": {
   720	            "K_h1": f"{h1['transparency_K_h1']['n_individually_holm_sig']}/{h1['transparency_K_h1']['N']} ≥ {h1['transparency_K_h1']['K']}?  {'YES' if h1['transparency_K_h1']['consistent'] else 'NO'}",
   721	            "K_h3_axis1": f"{h3_axis1['transparency_K_h3']['n_cells_pass']}/{h3_axis1['transparency_K_h3']['N']} ≥ {h3_axis1['transparency_K_h3']['K']}?  {'YES' if h3_axis1['transparency_K_h3']['consistent'] else 'NO'}",
   722	            "K_h3_axis2": f"{h3_axis2['transparency_K_h3']['n_cells_pass']}/{h3_axis2['transparency_K_h3']['N']} ≥ {h3_axis2['transparency_K_h3']['K']}?  {'YES' if h3_axis2['transparency_K_h3']['consistent'] else 'NO'}",
   723	        },
   724	    }
   725	
   726	    payload = json.dumps(result, indent=2, default=float)
   727	    if args.out == "-":
   728	        print(payload)
   729	    else:
   730	        out_path = Path(args.out)
   731	        out_path.parent.mkdir(parents=True, exist_ok=True)
   732	        out_path.write_text(payload)
   733	        logger.info(f"Result → {out_path}")
   734	        logger.info(f"Framing rule: {framing['rule']} — {framing['framing']} (hook power: {framing['hook_power']})")
   735	        logger.info(f"  H1: {h1['primary_gate']['decision']} (pooled drop-one {h1['primary_gate']['pooled_meta']['pooled_effect']:.2f}pp, "
   736	                    f"superiority p={h1['primary_gate']['superiority_test']['p_one_sided']:.4f}, "
   737	                    f"TOST equiv {h1['primary_gate']['tost_informational']['decision']})")
   738	        logger.info(f"  H2: {'PASS' if h2['h2a_cost_equivalence']['consistent'] else 'FAIL'} "
   739	                    f"({h2['h2a_cost_equivalence']['n_cells_pass']}/{h2['h2a_cost_equivalence']['N']} cells within ±{args.H2_cost_margin_pct}% cost)")
   740	        logger.info(f"  H3 axis-1 (P-text): {h3_axis1['primary_gate']['decision']} "
   741	                    f"(pooled unique={h3_axis1['primary_gate']['pooled_meta']['pooled_effect']:.2f})")
   742	        logger.info(f"  H3 axis-2 (P-prompt): {h3_axis2['primary_gate']['decision']} "
   743	                    f"(pooled unique={h3_axis2['primary_gate']['pooled_meta']['pooled_effect']:.2f})")
   744	        logger.info(f"  Transparency K_h1: {result['transparency_summary']['K_h1']}")
   745	        logger.info(f"  Transparency K_h3 axis-1: {result['transparency_summary']['K_h3_axis1']}")
   746	        logger.info(f"  Transparency K_h3 axis-2: {result['transparency_summary']['K_h3_axis2']}")
   747	
   748	
   749	if __name__ == "__main__":
   750	    main()

exec
/bin/bash -lc "nl -ba scripts/analysis/aggregate_phantom_lift.py | sed -n '360,760p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   360	        prev = out[i]
   361	    return out
   362	
   363	
   364	def analyze_cell(cell: dict) -> Optional[dict]:
   365	    """Compute phantom lift for a single (baseline, site) cell.
   366	
   367	    Required modes: DOM, SoM, Vision, P-SoM. P-text and P-prompt are optional —
   368	    when absent, dependent oracle/lift columns are emitted as None (CSV blank).
   369	
   370	    Returns None if any required mode missing or below MIN_EP_FOR_CELL.
   371	    """
   372	    succ, obs = {}, {}
   373	    for mode, ep_dir in cell["modes"].items():
   374	        s, o = load(ep_dir)
   375	        if len(o) < MIN_EP_FOR_CELL:
   376	            # Skip undersized modes silently; allow rest of cell to still build.
   377	            # P-prompt cell may be partial (e.g. 127/210); keep it if it meets MIN.
   378	            continue
   379	        succ[mode] = s
   380	        obs[mode] = o
   381	
   382	    required = ("DOM", "SoM", "Vision", "P-SoM")
   383	    if any(m not in succ for m in required):
   384	        return None
   385	    has_pdom = "P-text" in succ
   386	    has_pprompt = "P-prompt" in succ
   387	
   388	    # F07 audit fix 2026-05-09: per-comparison universe — each oracle
   389	    # contrast uses ONLY the arms it compares, not a global intersection
   390	    # across all present modes. Previously a partial P-prompt arm could
   391	    # shrink the 3-vs-5 denominator even though P-prompt is not in that
   392	    # estimand. Universes:
   393	    #   universe_psom_only:    obs(DOM, SoM, Vision, P-SoM)
   394	    #   universe_pdom_only:    obs(DOM, SoM, Vision, P-text)
   395	    #   universe_pprompt_only: obs(DOM, SoM, Vision, P-prompt)
   396	    #   universe_5:            obs(DOM, SoM, Vision, P-text, P-SoM)   ← 3-vs-5 denominator
   397	    #   universe_6:            obs(DOM, SoM, Vision, P-text, P-SoM, P-prompt)
   398	    # `n_common` reported in the CSV = |universe_5| if P-text present,
   399	    # else |universe_psom_only| (closest match to historical semantics).
   400	
   401	    def _universe(arms: list) -> set:
   402	        return set.intersection(*[obs[a] for a in arms if a in obs])
   403	
   404	    universe_psom_only = _universe(["DOM", "SoM", "Vision", "P-SoM"])
   405	    universe_pdom_only = _universe(["DOM", "SoM", "Vision", "P-text"]) if has_pdom else set()
   406	    universe_pprompt_only = _universe(["DOM", "SoM", "Vision", "P-prompt"]) if has_pprompt else set()
   407	    if has_pdom:
   408	        universe_5 = _universe(["DOM", "SoM", "Vision", "P-text", "P-SoM"])
   409	    else:
   410	        universe_5 = universe_psom_only
   411	    if has_pdom and has_pprompt:
   412	        universe_6 = _universe(["DOM", "SoM", "Vision", "P-text", "P-SoM", "P-prompt"])
   413	    else:
   414	        universe_6 = set()
   415	
   416	    common = universe_5 if has_pdom else universe_psom_only
   417	    n = len(common)
   418	    if n < MIN_EP_FOR_CELL:
   419	        return None
   420	
   421	    # Restrict each mode's success set to its own comparison's universe
   422	    # at use site (not globally as before).
   423	    def _restrict_set(arms: list) -> tuple[set, dict]:
   424	        u = _universe(arms)
   425	        return u, {a: succ[a] & u for a in arms if a in succ}
   426	
   427	    # P-SoM only (3 → 4_psom)
   428	    u_psom, succ_r_psom = _restrict_set(["DOM", "SoM", "Vision", "P-SoM"])
   429	    union_3_psom_only = succ_r_psom["DOM"] | succ_r_psom["SoM"] | succ_r_psom["Vision"]
   430	    union_4_psom = union_3_psom_only | succ_r_psom["P-SoM"]
   431	    sr_3_psom_only = 100 * len(union_3_psom_only) / max(1, len(u_psom))
   432	    sr_4_psom = 100 * len(union_4_psom) / max(1, len(u_psom))
   433	    universe_psom = sorted(u_psom)
   434	    in_3_psom = np.array([t in union_3_psom_only for t in universe_psom], dtype=bool)
   435	    in_4_psom = np.array([t in union_4_psom for t in universe_psom], dtype=bool)
   436	
   437	    # CSV-reported sr_3 / union_3 use universe_5 (paper-grade primary
   438	    # denominator when P-text present; same as universe_psom otherwise).
   439	    succ_r = {m: s & common for m, s in succ.items()}
   440	    union_3 = succ_r["DOM"] | succ_r["SoM"] | succ_r["Vision"]
   441	    sr_3 = 100 * len(union_3) / n
   442	    universe = sorted(common)
   443	    # Backward-compat aliases for downstream 5-mode / H3 axis tests
   444	    # which use in_3 indexed against universe_5.
   445	    in_3 = np.array([t in union_3 for t in universe], dtype=bool)
   446	
   447	    # Single-P-SoM lift CI (uses P-SoM-specific universe per F07)
   448	    ci_lo_psom, ci_hi_psom = bootstrap_lift_ci(in_3_psom, in_4_psom)
   449	    h_4psom_vs_3 = cohen_h(sr_4_psom / 100, sr_3_psom_only / 100)
   450	    wstat_psom, wp_psom = wilcoxon_signed_rank(in_3_psom, in_4_psom)
   451	    mc_p_psom = mcnemar_exact_one_sided(in_3_psom, in_4_psom)
   452	    tost_p_psom = bootstrap_tost_p(in_3_psom, in_4_psom)
   453	
   454	    psom_adds = succ_r["P-SoM"] - union_3
   455	
   456	    if has_pdom:
   457	        union_4_pdom = union_3 | succ_r["P-text"]
   458	        union_5 = union_3 | succ_r["P-text"] | succ_r["P-SoM"]
   459	        sr_4_pdom = 100 * len(union_4_pdom) / n
   460	        sr_5 = 100 * len(union_5) / n
   461	        in_4_pdom = np.array([t in union_4_pdom for t in universe], dtype=bool)
   462	        in_5 = np.array([t in union_5 for t in universe], dtype=bool)
   463	        ci_lo, ci_hi = bootstrap_lift_ci(in_3, in_5)
   464	        ci_lo_pdom, ci_hi_pdom = bootstrap_lift_ci(in_3, in_4_pdom)
   465	        h_5_vs_3 = cohen_h(sr_5 / 100, sr_3 / 100)
   466	        h_4pdom_vs_3 = cohen_h(sr_4_pdom / 100, sr_3 / 100)
   467	        wstat_5, wp_5 = wilcoxon_signed_rank(in_3, in_5)
   468	        wstat_pdom, wp_pdom = wilcoxon_signed_rank(in_3, in_4_pdom)
   469	        mc_p_5 = mcnemar_exact_one_sided(in_3, in_5)
   470	        mc_p_pdom = mcnemar_exact_one_sided(in_3, in_4_pdom)
   471	        tost_p_5 = bootstrap_tost_p(in_3, in_5)
   472	        tost_p_pdom = bootstrap_tost_p(in_3, in_4_pdom)
   473	        pdom_adds = succ_r["P-text"] - union_3
   474	        both_add = pdom_adds & psom_adds
   475	        pdom_only = pdom_adds - psom_adds
   476	        psom_only = psom_adds - pdom_adds
   477	        inter = succ_r["P-SoM"] & succ_r["P-text"]
   478	        unionj = succ_r["P-SoM"] | succ_r["P-text"]
   479	        jaccard = (len(inter) / len(unionj)) if unionj else 0.0
   480	        jaccard_warn = jaccard > 0.7
   481	    else:
   482	        sr_4_pdom = None
   483	        sr_5 = None
   484	        ci_lo = ci_hi = None
   485	        ci_lo_pdom = ci_hi_pdom = None
   486	        h_5_vs_3 = None
   487	        h_4pdom_vs_3 = None
   488	        wp_5 = wp_pdom = None
   489	        mc_p_5 = mc_p_pdom = None
   490	        tost_p_5 = tost_p_pdom = None
   491	        pdom_adds = both_add = pdom_only = set()
   492	        psom_only = psom_adds  # no overlap with absent P-text
   493	        jaccard = None
   494	        jaccard_warn = False
   495	
   496	    # P-prompt 4-mode lift + 6-mode oracle (when present)
   497	    if has_pprompt:
   498	        # F07 audit fix 2026-05-09: P-prompt-only comparison uses
   499	        # universe_pprompt_only (DOM ∩ SoM ∩ Vision ∩ P-prompt), NOT the
   500	        # 5-mode universe — otherwise the denominator drops by tasks
   501	        # missing in P-text/P-SoM that have nothing to do with this arm.
   502	        u_pprompt, succ_r_pprompt = _restrict_set(["DOM", "SoM", "Vision", "P-prompt"])
   503	        union_3_pprompt_only = succ_r_pprompt["DOM"] | succ_r_pprompt["SoM"] | succ_r_pprompt["Vision"]
   504	        union_4_pprompt = union_3_pprompt_only | succ_r_pprompt["P-prompt"]
   505	        sr_3_pprompt_only = 100 * len(union_3_pprompt_only) / max(1, len(u_pprompt))
   506	        sr_4_pprompt = 100 * len(union_4_pprompt) / max(1, len(u_pprompt))
   507	        u_pprompt_sorted = sorted(u_pprompt)
   508	        in_3_pprompt = np.array([t in union_3_pprompt_only for t in u_pprompt_sorted], dtype=bool)
   509	        in_4_pprompt = np.array([t in union_4_pprompt for t in u_pprompt_sorted], dtype=bool)
   510	        ci_lo_pprompt, ci_hi_pprompt = bootstrap_lift_ci(in_3_pprompt, in_4_pprompt)
   511	        h_4pprompt_vs_3 = cohen_h(sr_4_pprompt / 100, sr_3_pprompt_only / 100)
   512	        wstat_pprompt, wp_pprompt = wilcoxon_signed_rank(in_3_pprompt, in_4_pprompt)
   513	        mc_p_pprompt = mcnemar_exact_one_sided(in_3_pprompt, in_4_pprompt)
   514	        tost_p_pprompt = bootstrap_tost_p(in_3_pprompt, in_4_pprompt)
   515	        pprompt_adds = succ_r["P-prompt"] - union_3
   516	        if has_pdom:
   517	            # F07 audit fix 2026-05-09: 6-mode oracle and 6-vs-5
   518	            # incremental tests must use universe_6 (DOM ∩ SoM ∩
   519	            # Vision ∩ P-text ∩ P-SoM ∩ P-prompt). Previously used
   520	            # universe_5 which can include tasks where P-prompt was
   521	            # not observed → treats missing as failed.
   522	            u6_sorted = sorted(universe_6)
   523	            succ_r_u6 = {m: s & universe_6 for m, s in succ.items()}
   524	            union_3_u6 = succ_r_u6["DOM"] | succ_r_u6["SoM"] | succ_r_u6["Vision"]
   525	            union_5_u6 = union_3_u6 | succ_r_u6["P-text"] | succ_r_u6["P-SoM"]
   526	            union_6 = union_5_u6 | succ_r_u6["P-prompt"]
   527	            sr_3_u6 = 100 * len(union_3_u6) / max(1, len(universe_6))
   528	            sr_5_u6 = 100 * len(union_5_u6) / max(1, len(universe_6))
   529	            sr_6 = 100 * len(union_6) / max(1, len(universe_6))
   530	            in_3_u6 = np.array([t in union_3_u6 for t in u6_sorted], dtype=bool)
   531	            in_5_u6 = np.array([t in union_5_u6 for t in u6_sorted], dtype=bool)
   532	            in_6 = np.array([t in union_6 for t in u6_sorted], dtype=bool)
   533	            ci_lo_6, ci_hi_6 = bootstrap_lift_ci(in_3_u6, in_6)
   534	            ci_lo_6v5, ci_hi_6v5 = bootstrap_lift_ci(in_5_u6, in_6)
   535	            h_6_vs_3 = cohen_h(sr_6 / 100, sr_3_u6 / 100)
   536	            h_6_vs_5 = cohen_h(sr_6 / 100, sr_5_u6 / 100)
   537	            _, wp_6 = wilcoxon_signed_rank(in_3_u6, in_6)
   538	            _, wp_6v5 = wilcoxon_signed_rank(in_5_u6, in_6)
   539	            mc_p_6 = mcnemar_exact_one_sided(in_3_u6, in_6)
   540	            mc_p_6v5 = mcnemar_exact_one_sided(in_5_u6, in_6)
   541	            tost_p_6 = bootstrap_tost_p(in_3_u6, in_6)
   542	            tost_p_6v5 = bootstrap_tost_p(in_5_u6, in_6)
   543	        else:
   544	            sr_6 = None
   545	            ci_lo_6 = ci_hi_6 = ci_lo_6v5 = ci_hi_6v5 = None
   546	            h_6_vs_3 = h_6_vs_5 = None
   547	            wp_6 = wp_6v5 = None
   548	            mc_p_6 = mc_p_6v5 = None
   549	            tost_p_6 = tost_p_6v5 = None
   550	    else:
   551	        sr_4_pprompt = None
   552	        ci_lo_pprompt = ci_hi_pprompt = None
   553	        h_4pprompt_vs_3 = None
   554	        wp_pprompt = None
   555	        mc_p_pprompt = None
   556	        tost_p_pprompt = None
   557	        pprompt_adds = set()
   558	        sr_6 = None
   559	        ci_lo_6 = ci_hi_6 = ci_lo_6v5 = ci_hi_6v5 = None
   560	        h_6_vs_3 = h_6_vs_5 = None
   561	        wp_6 = wp_6v5 = None
   562	        mc_p_6 = mc_p_6v5 = None
   563	        tost_p_6 = tost_p_6v5 = None
   564	
   565	    # H3 structural test: phantom space 2-axis empirical validation.
   566	    # For each axis, bootstrap CI on |arm ∖ P-SoM| unique-count + McNemar exact
   567	    # one-sided. CI lower bound > 0 evidences axis contributes tasks P-SoM
   568	    # doesn't solve (i.e., axis is empirically distinct from compound center,
   569	    # phantom space is multi-region not collapsed point).
   570	    in_psom_raw = np.array([t in succ_r["P-SoM"] for t in universe], dtype=bool)
   571	
   572	    if has_pdom:
   573	        in_pdom_raw = np.array([t in succ_r["P-text"] for t in universe], dtype=bool)
   574	        h3_axis1_count, h3_axis1_ci_lo, h3_axis1_ci_hi = bootstrap_unique_count_ci(
   575	            in_pdom_raw, in_psom_raw)
   576	        # mcnemar_exact_one_sided(a, b) tests H1: b > a (b adds tasks a misses)
   577	        # Set a=P-SoM, b=P-text → H1 asymmetric: P-text adds tasks P-SoM misses
   578	        # more often than vice versa (directional structural asymmetry test).
   579	        h3_axis1_mcnemar_p = mcnemar_exact_one_sided(in_psom_raw, in_pdom_raw)
   580	    else:
   581	        h3_axis1_count = h3_axis1_ci_lo = h3_axis1_ci_hi = h3_axis1_mcnemar_p = None
   582	
   583	    if has_pprompt:
   584	        in_pprompt_raw = np.array([t in succ_r["P-prompt"] for t in universe], dtype=bool)
   585	        h3_axis2_count, h3_axis2_ci_lo, h3_axis2_ci_hi = bootstrap_unique_count_ci(
   586	            in_pprompt_raw, in_psom_raw)
   587	        h3_axis2_mcnemar_p = mcnemar_exact_one_sided(in_psom_raw, in_pprompt_raw)
   588	    else:
   589	        h3_axis2_count = h3_axis2_ci_lo = h3_axis2_ci_hi = h3_axis2_mcnemar_p = None
   590	
   591	    is_partial = (any(len(o) < cell["n_expected"] for o in obs.values()) or not has_pdom
   592	                  or not has_pprompt)
   593	
   594	    def maybe_round(value, ndigits=4):
   595	        return None if value is None else round(value, ndigits)
   596	
   597	    return {
   598	        "baseline": cell["baseline"],
   599	        "site": cell["site"],
   600	        "n_common": n,
   601	        "n_expected": cell["n_expected"],
   602	        "is_partial": is_partial,
   603	        "has_pdom": has_pdom,
   604	        "has_pprompt": has_pprompt,
   605	        "sr_dom":     round(100 * len(succ_r["DOM"]) / n, 4),
   606	        "sr_som":     round(100 * len(succ_r["SoM"]) / n, 4),
   607	        "sr_vision":  round(100 * len(succ_r["Vision"]) / n, 4),
   608	        "sr_pdom":    (round(100 * len(succ_r["P-text"]) / n, 4) if has_pdom else None),
   609	        "sr_psom":    round(100 * len(succ_r["P-SoM"]) / n, 4),
   610	        "sr_pprompt": (round(100 * len(succ_r["P-prompt"]) / n, 4) if has_pprompt else None),
   611	        "oracle_3mode_pp":  round(sr_3, 4),
   612	        "oracle_4mode_pdom_pp": maybe_round(sr_4_pdom),
   613	        "oracle_4mode_psom_pp": round(sr_4_psom, 4),
   614	        "oracle_4mode_pprompt_pp": maybe_round(sr_4_pprompt),
   615	        "oracle_5mode_pp":  maybe_round(sr_5),
   616	        "oracle_6mode_pp":  maybe_round(sr_6),
   617	        "lift_5_vs_3_pp":   (round(sr_5 - sr_3, 4) if sr_5 is not None else None),
   618	        "lift_5_vs_3_ci95_lo_pp":  maybe_round(ci_lo),
   619	        "lift_5_vs_3_ci95_hi_pp":  maybe_round(ci_hi),
   620	        "lift_4pdom_vs_3_pp":   (round(sr_4_pdom - sr_3, 4) if sr_4_pdom is not None else None),
   621	        "lift_4pdom_vs_3_ci95_lo_pp": maybe_round(ci_lo_pdom),
   622	        "lift_4pdom_vs_3_ci95_hi_pp": maybe_round(ci_hi_pdom),
   623	        "lift_4psom_vs_3_pp":   round(sr_4_psom - sr_3, 4),
   624	        "lift_4psom_vs_3_ci95_lo_pp": round(ci_lo_psom, 4),
   625	        "lift_4psom_vs_3_ci95_hi_pp": round(ci_hi_psom, 4),
   626	        "lift_4pprompt_vs_3_pp": (round(sr_4_pprompt - sr_3, 4) if sr_4_pprompt is not None else None),
   627	        "lift_4pprompt_vs_3_ci95_lo_pp": maybe_round(ci_lo_pprompt),
   628	        "lift_4pprompt_vs_3_ci95_hi_pp": maybe_round(ci_hi_pprompt),
   629	        "lift_6_vs_3_pp": (round(sr_6 - sr_3, 4) if sr_6 is not None else None),
   630	        "lift_6_vs_3_ci95_lo_pp": maybe_round(ci_lo_6),
   631	        "lift_6_vs_3_ci95_hi_pp": maybe_round(ci_hi_6),
   632	        "lift_6_vs_5_pp": (round(sr_6 - sr_5, 4) if (sr_6 is not None and sr_5 is not None) else None),
   633	        "lift_6_vs_5_ci95_lo_pp": maybe_round(ci_lo_6v5),
   634	        "lift_6_vs_5_ci95_hi_pp": maybe_round(ci_hi_6v5),
   635	        # Effect sizes (Cohen's h on oracle proportions)
   636	        "cohen_h_5_vs_3":     maybe_round(h_5_vs_3),
   637	        "cohen_h_5_vs_3_label": (cohen_h_label(h_5_vs_3) if h_5_vs_3 is not None else None),
   638	        "cohen_h_4pdom_vs_3": maybe_round(h_4pdom_vs_3),
   639	        "cohen_h_4pdom_vs_3_label": (cohen_h_label(h_4pdom_vs_3) if h_4pdom_vs_3 is not None else None),
   640	        "cohen_h_4psom_vs_3": round(h_4psom_vs_3, 4),
   641	        "cohen_h_4psom_vs_3_label": cohen_h_label(h_4psom_vs_3),
   642	        "cohen_h_4pprompt_vs_3": maybe_round(h_4pprompt_vs_3),
   643	        "cohen_h_4pprompt_vs_3_label": (cohen_h_label(h_4pprompt_vs_3) if h_4pprompt_vs_3 is not None else None),
   644	        "cohen_h_6_vs_3": maybe_round(h_6_vs_3),
   645	        "cohen_h_6_vs_3_label": (cohen_h_label(h_6_vs_3) if h_6_vs_3 is not None else None),
   646	        "cohen_h_6_vs_5": maybe_round(h_6_vs_5),
   647	        "cohen_h_6_vs_5_label": (cohen_h_label(h_6_vs_5) if h_6_vs_5 is not None else None),
   648	        # Wilcoxon (paired sign on binary)
   649	        "wilcoxon_5_vs_3_p":     wp_5,
   650	        "wilcoxon_4pdom_vs_3_p": wp_pdom,
   651	        "wilcoxon_4psom_vs_3_p": wp_psom,
   652	        "wilcoxon_4pprompt_vs_3_p": wp_pprompt,
   653	        "wilcoxon_6_vs_3_p": wp_6,
   654	        "wilcoxon_6_vs_5_p": wp_6v5,
   655	        # McNemar exact 1-sided
   656	        "mcnemar_5_vs_3_p":     mc_p_5,
   657	        "mcnemar_4pdom_vs_3_p": mc_p_pdom,
   658	        "mcnemar_4psom_vs_3_p": mc_p_psom,
   659	        "mcnemar_4pprompt_vs_3_p": mc_p_pprompt,
   660	        "mcnemar_6_vs_3_p": mc_p_6,
   661	        "mcnemar_6_vs_5_p": mc_p_6v5,
   662	        # TOST equivalence p (bootstrap, δ=0.5pp; rejects equivalence if max < α)
   663	        "tost_5_vs_3_p":      tost_p_5,
   664	        "tost_4pdom_vs_3_p":  tost_p_pdom,
   665	        "tost_4psom_vs_3_p":  tost_p_psom,
   666	        "tost_4pprompt_vs_3_p": tost_p_pprompt,
   667	        "tost_6_vs_3_p":      tost_p_6,
   668	        "tost_6_vs_5_p":      tost_p_6v5,
   669	        # Family-adjusted p / q (filled by main() post-collection; see §family decl)
   670	        "mcnemar_5_vs_3_p_holm":     None,
   671	        "mcnemar_5_vs_3_q_bh":       None,
   672	        "mcnemar_5_vs_3_p_bonf":     None,
   673	        "mcnemar_4pdom_vs_3_p_holm": None,
   674	        "mcnemar_4pdom_vs_3_q_bh":   None,
   675	        "mcnemar_4psom_vs_3_p_holm": None,
   676	        "mcnemar_4psom_vs_3_q_bh":   None,
   677	        "mcnemar_4pprompt_vs_3_p_holm": None,
   678	        "mcnemar_4pprompt_vs_3_q_bh":   None,
   679	        # H3 structural — phantom space 2-axis empirical validation
   680	        "h3_axis1_unique_count":      h3_axis1_count,
   681	        "h3_axis1_ci95_lo":           (round(h3_axis1_ci_lo, 4) if h3_axis1_ci_lo is not None else None),
   682	        "h3_axis1_ci95_hi":           (round(h3_axis1_ci_hi, 4) if h3_axis1_ci_hi is not None else None),
   683	        "h3_axis1_mcnemar_p":         (round(h3_axis1_mcnemar_p, 6) if h3_axis1_mcnemar_p is not None else None),
   684	        "h3_axis1_mcnemar_p_holm":    None,  # filled by family correction in main()
   685	        "h3_axis2_unique_count":      h3_axis2_count,
   686	        "h3_axis2_ci95_lo":           (round(h3_axis2_ci_lo, 4) if h3_axis2_ci_lo is not None else None),
   687	        "h3_axis2_ci95_hi":           (round(h3_axis2_ci_hi, 4) if h3_axis2_ci_hi is not None else None),
   688	        "h3_axis2_mcnemar_p":         (round(h3_axis2_mcnemar_p, 6) if h3_axis2_mcnemar_p is not None else None),
   689	        "h3_axis2_mcnemar_p_holm":    None,  # filled by family correction in main()
   690	        # Decomposition
   691	        "pdom_adds_count":      (len(pdom_adds) if has_pdom else None),
   692	        "psom_adds_count":      len(psom_adds),
   693	        "pprompt_adds_count":   (len(pprompt_adds) if has_pprompt else None),
   694	        "pdom_only_count":      (len(pdom_only) if has_pdom else None),
   695	        "psom_only_count":      (len(psom_only) if has_pdom else None),
   696	        "both_phantom_overlap_count": (len(both_add) if has_pdom else None),
   697	        # Scenario C sentinel: P-SoM ↔ P-text Jaccard
   698	        "phantom_pair_jaccard": (round(jaccard, 4) if jaccard is not None else None),
   699	        "phantom_pair_jaccard_warn": jaccard_warn,
   700	    }
   701	
   702	
   703	def main() -> int:
   704	    ap = argparse.ArgumentParser()
   705	    ap.add_argument("--output", default=str(REPO / "results/phantom_paper/phantom_lift.csv"))
   706	    args = ap.parse_args()
   707	
   708	    rows = []
   709	    skipped = []
   710	    for cell in CELLS:
   711	        r = analyze_cell(cell)
   712	        if r is None:
   713	            skipped.append(f"{cell['baseline']} {cell['site']}")
   714	            continue
   715	        rows.append(r)
   716	
   717	    # ── Multiple-comparison correction (per pre-registered family) ────────
   718	    # Comparison families:
   719	    #   PRIMARY (m = N_cells):           3→5-mode lift (one per cell)
   720	    #   SECONDARY (m = 3 × N_cells):     per-arm drop-one (P-text/P-SoM/P-prompt)
   721	    #   TERTIARY (m = 2 × N_cells):      6-mode oracle (vs 3 / vs 5) — exploratory
   722	    # Method: Holm-Bonferroni step-down per family (FWER) + BH FDR (informational)
   723	    # Primary p-value: McNemar exact one-sided (directional H1: phantom adds tasks)
   724	    # Wilcoxon two-sided remains uncorrected as secondary report.
   725	
   726	    def _adjust_inplace(rows, key_p, key_holm, key_bh, key_bonf=None):
   727	        """Run Bonferroni / Holm / BH on a list of rows for a given p-value field."""
   728	        pvals = [r.get(key_p) for r in rows]
   729	        holm = holm_bonferroni_adjust(pvals)
   730	        bh = bh_fdr_adjust(pvals)
   731	        bonf = bonferroni_adjust(pvals) if key_bonf else [None] * len(rows)
   732	        for r, h, q, b in zip(rows, holm, bh, bonf):
   733	            r[key_holm] = round(h, 6) if h is not None else None
   734	            r[key_bh] = round(q, 6) if q is not None else None
   735	            if key_bonf:
   736	                r[key_bonf] = round(b, 6) if b is not None else None
   737	
   738	    # Family A (PRIMARY): 3→5-mode lift
   739	    _adjust_inplace(rows, "mcnemar_5_vs_3_p",
   740	                    "mcnemar_5_vs_3_p_holm", "mcnemar_5_vs_3_q_bh",
   741	                    key_bonf="mcnemar_5_vs_3_p_bonf")
   742	
   743	    # Family B (SECONDARY): per-arm drop-one. Pool across cells × {pdom, psom, pprompt}.
   744	    flat_secondary = []
   745	    for r in rows:
   746	        for arm in ("4pdom", "4psom", "4pprompt"):
   747	            p = r.get(f"mcnemar_{arm}_vs_3_p")
   748	            flat_secondary.append((r, arm, p))
   749	    holm_b = holm_bonferroni_adjust([t[2] for t in flat_secondary])
   750	    bh_b = bh_fdr_adjust([t[2] for t in flat_secondary])
   751	    for (r, arm, _), h, q in zip(flat_secondary, holm_b, bh_b):
   752	        r[f"mcnemar_{arm}_vs_3_p_holm"] = round(h, 6) if h is not None else None
   753	        r[f"mcnemar_{arm}_vs_3_q_bh"] = round(q, 6) if q is not None else None
   754	
   755	    # H3 STRUCTURAL family: per-axis structural test (axis 1 = P-text, axis 2 = P-prompt).
   756	    # Holm-corrected separately within each axis sub-family (axis 1 / axis 2),
   757	    # because structural claim is weaker than deployment — separate family
   758	    # avoids inflating PRIMARY/SECONDARY family m count.
   759	    for axis_key in ("h3_axis1_mcnemar_p", "h3_axis2_mcnemar_p"):
   760	        ps = [r.get(axis_key) for r in rows]

exec
/bin/bash -lc "nl -ba scripts/analysis/sensitivity_loo_meta.py | sed -n '1,300p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""F4 sensitivity — leave-one-cell-out (LOO) meta-analysis + threshold gradient.
     3	
     4	Addresses audit constraint **F4** (statistical conclusion validity: report
     5	how close the conclusion is to threshold under cell removal + threshold
     6	shifts). Companion to `aggregate_phantom_meta.py` which produces the
     7	primary pooled estimates; this script bounds *how brittle* those estimates
     8	are.
     9	
    10	For each pre-registered arm (PRIMARY, SECONDARY arms with k>=2 cells):
    11	  - Drop each cell one at a time, re-run DerSimonian-Laird random-effects
    12	    pool from the remaining k-1 cells.
    13	  - Report pooled lift, 95% CI, and Holm-corrected p before/after drop.
    14	  - Flag arms where dropping any single cell flips the Holm decision.
    15	
    16	For threshold sensitivity:
    17	  - K-of-N rule (pre-registration §4): K_h1=12/16 / K_h3=11/16 already
    18	    reframed as secondary transparency (B9 lock).
    19	  - This script reports K±1 and K±2 for completeness — at current k=3
    20	    cells per arm, the rule is dominated by the per-cell paired test, so
    21	    K-of-N gradient is reported only for the *transparency* check.
    22	
    23	Usage:
    24	    .venv/bin/python3 scripts/analysis/sensitivity_loo_meta.py
    25	    .venv/bin/python3 scripts/analysis/sensitivity_loo_meta.py --output \\
    26	        docs/analysis/cross_sites/sensitivity_loo_meta.md
    27	
    28	Inputs:
    29	    results/phantom_paper/meta_phantom_lift.csv    # primary forest data
    30	
    31	Outputs:
    32	    docs/analysis/cross_sites/sensitivity_loo_meta.md  # paper appendix
    33	"""
    34	from __future__ import annotations
    35	
    36	import argparse
    37	import csv
    38	import math
    39	from pathlib import Path
    40	
    41	REPO = Path(__file__).resolve().parents[2]
    42	# F11 audit fix 2026-05-09: DEFAULT_INPUT points to per-cell forest CSV
    43	# (`phantom_lift.csv`), not the pooled meta CSV. The LOO computation
    44	# requires per-cell theta+SE; meta_phantom_lift.csv only has pooled.
    45	DEFAULT_INPUT = REPO / "results/phantom_paper/phantom_lift.csv"
    46	DEFAULT_OUTPUT = REPO / "docs/analysis/cross_sites/sensitivity_loo_meta.md"
    47	
    48	
    49	def dl_random_effects(estimates: list[float], variances: list[float]) -> dict:
    50	    """DerSimonian-Laird random-effects meta-analysis on per-cell estimates.
    51	
    52	    Returns dict with theta_re, se_re, ci_lo, ci_hi, z, p_one_sided, tau2, I2, Q, df.
    53	    """
    54	    k = len(estimates)
    55	    if k == 0:
    56	        return {"k": 0}
    57	
    58	    weights_fe = [1.0 / v if v > 0 else 0.0 for v in variances]
    59	    sum_w = sum(weights_fe)
    60	    theta_fe = sum(w * t for w, t in zip(weights_fe, estimates)) / sum_w
    61	    Q = sum(w * (t - theta_fe) ** 2 for w, t in zip(weights_fe, estimates))
    62	    df = k - 1
    63	    sum_w_sq = sum(w ** 2 for w in weights_fe)
    64	    if df > 0:
    65	        c = sum_w - sum_w_sq / sum_w
    66	        tau2 = max(0.0, (Q - df) / c) if c > 0 else 0.0
    67	    else:
    68	        tau2 = 0.0
    69	    weights_re = [1.0 / (v + tau2) if (v + tau2) > 0 else 0.0 for v in variances]
    70	    sum_w_re = sum(weights_re)
    71	    theta_re = sum(w * t for w, t in zip(weights_re, estimates)) / sum_w_re
    72	    se_re = math.sqrt(1.0 / sum_w_re) if sum_w_re > 0 else float("nan")
    73	    ci_lo = theta_re - 1.96 * se_re
    74	    ci_hi = theta_re + 1.96 * se_re
    75	    z = theta_re / se_re if se_re > 0 else float("nan")
    76	    # one-sided p (H1: theta_re > 0)
    77	    from math import erf
    78	    def phi(z):
    79	        return 0.5 * (1 + erf(z / math.sqrt(2)))
    80	    p_one_sided = 1 - phi(z) if not math.isnan(z) else float("nan")
    81	    I2 = max(0.0, (Q - df) / Q * 100) if Q > 0 and df > 0 else 0.0
    82	    return dict(
    83	        k=k, theta_re=theta_re, se_re=se_re, ci_lo=ci_lo, ci_hi=ci_hi,
    84	        z=z, p_one_sided=p_one_sided, tau2=tau2, I2=I2, Q=Q, df=df,
    85	    )
    86	
    87	
    88	ARM_MAP = [
    89	    ("5_vs_3",        "3→5-mode oracle lift"),
    90	    ("4pdom_vs_3",    "P-text drop-in"),
    91	    ("4psom_vs_3",    "P-SoM drop-in"),
    92	    ("4pprompt_vs_3", "P-prompt drop-in"),
    93	    ("6_vs_3",        "6-mode oracle lift"),
    94	    ("6_vs_5",        "P-prompt incremental"),
    95	]
    96	
    97	
    98	def parse_forest_csv(path: Path) -> dict[str, list[dict]]:
    99	    """Parse phantom_lift.csv (per-cell, wide-format) into per-arm cell lists.
   100	
   101	    The wide-format CSV has columns like `lift_5_vs_3_pp`, `lift_5_vs_3_ci95_lo_pp`,
   102	    `lift_5_vs_3_ci95_hi_pp` for each arm. We pivot into per-arm long form.
   103	
   104	    Returns: {arm_label: [{cell, theta, se, ci_lo, ci_hi}, ...]}
   105	    """
   106	    # F11 audit fix 2026-05-09: honor the path argument; previously
   107	    # ignored and always read phantom_lift.csv.
   108	    arms: dict[str, list[dict]] = {}
   109	    forest_csv = Path(path) if path is not None else (
   110	        REPO / "results/phantom_paper/phantom_lift.csv"
   111	    )
   112	    if not forest_csv.exists():
   113	        return arms
   114	
   115	    with open(forest_csv) as f:
   116	        cell_rows = list(csv.DictReader(f))
   117	
   118	    for row in cell_rows:
   119	        cell = f"{row.get('baseline', '')} {row.get('site', '')}".strip()
   120	        if not cell:
   121	            continue
   122	        for arm_key, arm_label in ARM_MAP:
   123	            theta_s = row.get(f"lift_{arm_key}_pp", "")
   124	            ci_lo_s = row.get(f"lift_{arm_key}_ci95_lo_pp", "")
   125	            ci_hi_s = row.get(f"lift_{arm_key}_ci95_hi_pp", "")
   126	            if not (theta_s and ci_lo_s and ci_hi_s):
   127	                continue
   128	            try:
   129	                theta = float(theta_s)
   130	                ci_lo = float(ci_lo_s)
   131	                ci_hi = float(ci_hi_s)
   132	            except ValueError:
   133	                continue
   134	            se = (ci_hi - ci_lo) / (2 * 1.96)
   135	            if se <= 0:
   136	                continue
   137	            arms.setdefault(arm_label, []).append(
   138	                dict(cell=cell, theta=theta, se=se, ci_lo=ci_lo, ci_hi=ci_hi)
   139	            )
   140	    return arms
   141	
   142	
   143	def loo_table(arm_label: str, cells_data: list[dict], holm_alpha: float = 0.05) -> list[dict]:
   144	    """Leave-one-cell-out table for arm.
   145	
   146	    Returns list of dicts with: dropped_cell, k_remaining, theta_re, ci_lo,
   147	    ci_hi, p_one_sided, holm_pass.
   148	    """
   149	    k = len(cells_data)
   150	    rows = []
   151	    # Baseline: all cells included
   152	    estimates = [c["theta"] for c in cells_data]
   153	    variances = [c["se"] ** 2 for c in cells_data]
   154	    base = dl_random_effects(estimates, variances)
   155	    base["dropped_cell"] = "(none — all cells)"
   156	    base["k_remaining"] = k
   157	    base["holm_pass"] = base.get("p_one_sided", 1.0) < holm_alpha
   158	    rows.append(base)
   159	
   160	    if k < 2:
   161	        return rows  # cannot LOO if only 1 cell
   162	
   163	    for i, drop_cell in enumerate(cells_data):
   164	        kept = [c for j, c in enumerate(cells_data) if j != i]
   165	        loo = dl_random_effects([c["theta"] for c in kept], [c["se"] ** 2 for c in kept])
   166	        loo["dropped_cell"] = drop_cell["cell"]
   167	        loo["k_remaining"] = k - 1
   168	        loo["holm_pass"] = loo.get("p_one_sided", 1.0) < holm_alpha
   169	        rows.append(loo)
   170	    return rows
   171	
   172	
   173	# F10 audit fix 2026-05-09: SECONDARY family for Holm correction is the
   174	# 3 phantom drop-in arms (P-text / P-SoM / P-prompt). Within each LOO
   175	# scenario, raw per-arm p-values are Holm-corrected across this family
   176	# of m=3 before the `holm_pass` decision. PRIMARY family (3→5-mode
   177	# oracle lift) is m=1 and needs no within-family correction.
   178	SECONDARY_ARMS = ["P-text drop-in", "P-SoM drop-in", "P-prompt drop-in"]
   179	PRIMARY_ARMS = ["3→5-mode oracle lift"]
   180	
   181	
   182	def holm_correct(ps: list[float]) -> list[float]:
   183	    """Holm-Bonferroni step-down adjustment for a list of one-sided p-values."""
   184	    if not ps:
   185	        return []
   186	    indexed = sorted(enumerate(ps), key=lambda x: x[1])
   187	    m = len(ps)
   188	    adjusted = [None] * m
   189	    running_max = 0.0
   190	    for rank, (orig_idx, p) in enumerate(indexed):
   191	        adj = min(1.0, (m - rank) * p)
   192	        running_max = max(running_max, adj)
   193	        adjusted[orig_idx] = running_max
   194	    return adjusted
   195	
   196	
   197	def apply_holm_within_family(arm_rows: dict[str, list[dict]],
   198	                             family_arms: list[str],
   199	                             alpha: float = 0.05) -> None:
   200	    """For each LOO scenario (baseline + each dropped cell), gather the
   201	    family arms' raw p-values, apply Holm across the family, and update
   202	    each row's `holm_pass` + add a `p_holm` field. Mutates `arm_rows`.
   203	    """
   204	    present_arms = [a for a in family_arms if arm_rows.get(a)]
   205	    if not present_arms:
   206	        return
   207	    # Index rows by dropped_cell
   208	    by_cell: dict[str, list[tuple[str, dict]]] = {}
   209	    for arm in present_arms:
   210	        for r in arm_rows[arm]:
   211	            by_cell.setdefault(r["dropped_cell"], []).append((arm, r))
   212	    for cell_key, arm_row_pairs in by_cell.items():
   213	        ps = [pair[1].get("p_one_sided", 1.0) for pair in arm_row_pairs]
   214	        adj = holm_correct(ps)
   215	        for (arm, row), p_h in zip(arm_row_pairs, adj):
   216	            row["p_holm_secondary"] = p_h
   217	            row["holm_pass"] = (p_h is not None and p_h < alpha)
   218	
   219	
   220	def render_md(arms: dict[str, list[dict]], output: Path) -> None:
   221	    lines = [
   222	        "# F4 Sensitivity — Leave-one-cell-out (LOO) Meta-analysis",
   223	        "",
   224	        "**Audit constraint F4** (statistical conclusion validity): report uncertainty + sensitivity to thresholds.",
   225	        "",
   226	        "Companion to `meta_phantom_lift.md`. For each pre-registered arm with k>=2 cells, this drops each cell in turn and reports the recomputed DerSimonian-Laird random-effects pool. Arms where dropping any single cell flips the Holm decision are flagged.",
   227	        "",
   228	        "**Holm correction (F10 fix 2026-05-09)**: SECONDARY family Holm correction is applied across the 3 phantom drop-in arms (P-text / P-SoM / P-prompt) within each LOO scenario before the per-arm `holm_pass` decision. PRIMARY family (3→5-mode oracle lift) is m=1, no within-family correction needed.",
   229	        "",
   230	        "**Generated**: 2026-05-09. Re-run after 16-cell paper-grade rerun completes.",
   231	        "",
   232	        "---",
   233	        "",
   234	    ]
   235	
   236	    # First compute per-arm LOO rows, then apply Holm within secondary family
   237	    arm_rows: dict[str, list[dict]] = {}
   238	    for arm_label in PRIMARY_ARMS + SECONDARY_ARMS:
   239	        cells = arms.get(arm_label, [])
   240	        if not cells:
   241	            arm_rows[arm_label] = []
   242	            continue
   243	        arm_rows[arm_label] = loo_table(arm_label, cells)
   244	
   245	    # F10: Holm-correct SECONDARY family per LOO scenario
   246	    apply_holm_within_family(arm_rows, SECONDARY_ARMS)
   247	
   248	    for arm_label in PRIMARY_ARMS + SECONDARY_ARMS:
   249	        rows = arm_rows[arm_label]
   250	        cells = arms.get(arm_label, [])
   251	        if not rows:
   252	            lines.append(f"## Arm: {arm_label} — no cell forest data")
   253	            lines.append("")
   254	            continue
   255	        is_secondary = arm_label in SECONDARY_ARMS
   256	        p_col_label = "p_Holm (m=3)" if is_secondary else "p (1-sided)"
   257	        lines += [
   258	            f"## Arm: {arm_label} (k={len(cells)} cells, family={'SECONDARY' if is_secondary else 'PRIMARY'})",
   259	            "",
   260	            f"| Dropped cell | k remaining | θ_re (pp) | 95% CI | p (raw 1-sided) | {p_col_label} | Pass at α=0.05 |",
   261	            "|---|---:|---:|---|---:|---:|:---:|",
   262	        ]
   263	        for r in rows:
   264	            ci_str = f"[{r.get('ci_lo', 0):.2f}, {r.get('ci_hi', 0):.2f}]"
   265	            holm_str = "✅" if r.get("holm_pass") else "❌"
   266	            p_raw = r.get("p_one_sided", 1.0)
   267	            p_corrected = r.get("p_holm_secondary") if is_secondary else p_raw
   268	            p_corrected_str = f"{p_corrected:.4f}" if p_corrected is not None else "—"
   269	            lines.append(
   270	                f"| {r['dropped_cell']} | {r['k_remaining']} | "
   271	                f"{r.get('theta_re', 0):+.2f} | {ci_str} | "
   272	                f"{p_raw:.4f} | {p_corrected_str} | {holm_str} |"
   273	            )
   274	
   275	        # Robustness verdict (now based on Holm-corrected pass)
   276	        any_flip = any(not r.get("holm_pass") for r in rows[1:])
   277	        baseline_pass = rows[0].get("holm_pass")
   278	        if baseline_pass and not any_flip:
   279	            verdict = "**Robust**: Holm decision unchanged under any single-cell removal."
   280	        elif baseline_pass and any_flip:
   281	            flipped = [r["dropped_cell"] for r in rows[1:] if not r.get("holm_pass")]
   282	            verdict = f"**FRAGILE**: dropping {flipped} flips Holm to non-significant. Per-cell influence is high."
   283	        else:
   284	            verdict = "**Underpowered**: baseline does not pass Holm at α=0.05 (with secondary-family correction)."
   285	        lines += ["", verdict, ""]
   286	
   287	    lines += [
   288	        "---",
   289	        "",
   290	        "## Methodological notes",
   291	        "",
   292	        "- **DL random-effects** computed via `dl_random_effects()` — same procedure as `aggregate_phantom_meta.py`.",
   293	        "- **Within-cell SE** derived from bootstrap 95% CI as `(CI_hi - CI_lo) / (2 × 1.96)` (matches primary script).",
   294	        "- **One-sided p** because H1 is directional (`theta > 0`).",
   295	        "- **Holm decision** at α=0.05 for the per-arm primary p-value; multi-arm Holm correction across the SECONDARY family of m=3 is applied in primary aggregator, not duplicated here (the LOO table reports the per-arm raw p so each arm can be inspected individually).",
   296	        "- **Threshold gradient (K-of-N)** is omitted because the K-of-N rule has been reframed as secondary transparency (audit B9 + preregistration §4 lock); the primary detection is the random-effects meta in this LOO table.",
   297	        "- **Underpowered arm caveat**: arms with k<3 cells cannot be LOO-tested meaningfully — they wait for 16-cell rerun.",
   298	        "",
   299	        "## Reviewer-rebuttal language",
   300	        "",

exec
/bin/bash -lc "nl -ba scripts/analysis/aggregate_routing_auroc.py | sed -n '1,340p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""[Outcome 0g] Outcome dimension — routing signal quality across conditions.
     3	
     4	Outputs:
     5	- results/phantom_paper/auroc_cross_condition.csv
     6	- results/phantom_paper/auroc_cross_condition.md
     7	- results/phantom_paper/auroc_cross_condition_summary.md
     8	
     9	Outcome 0g: per-mode routing AUROC evidence for router-usable signals.
    10	
    11	See docs/checkpoints/paper_planning.md §3 Outcome dimension framework.
    12	
    13	Aggregate per-mode routing signal AUROC + 95% bootstrap CI across runs.
    14	
    15	Reads existing per-run `analysis/signals/combined/tables/cross_mode_auroc.csv`
    16	(produced by analyze_confidence_calibration.py) and merges them into a single
    17	paper-ready table with run/baseline/site metadata.
    18	
    19	Usage:
    20	    python3 scripts/analysis/aggregate_routing_auroc.py \\
    21	        --output results/phantom_paper/auroc_cross_condition.csv
    22	
    23	Output columns: baseline, site, mode, signal, signal_type, AUROC,
    24	                AUROC_ci_lower, AUROC_ci_upper, n, run_id
    25	
    26	A second markdown summary lands at <output>.md with a paper-ready table
    27	showing top-3 signals per (baseline, site, mode).
    28	"""
    29	from __future__ import annotations
    30	
    31	import argparse
    32	import re
    33	from pathlib import Path
    34	
    35	import pandas as pd
    36	
    37	try:
    38	    from scripts.analysis.lib.run_registry import canonical_mode, get_run_dirs_paper_vwa
    39	except ModuleNotFoundError:  # pragma: no cover - supports direct script execution.
    40	    import sys
    41	    sys.path.append(str(Path(__file__).resolve().parents[2]))
    42	    from scripts.analysis.lib.run_registry import canonical_mode, get_run_dirs_paper_vwa
    43	
    44	
    45	REPO = Path(__file__).resolve().parents[2]
    46	DEFAULT_RUNS = get_run_dirs_paper_vwa()
    47	
    48	
    49	def parse_run_id(run_dir: Path) -> tuple[str, str]:
    50	    """Extract (baseline, site) from a paper run id."""
    51	    name = run_dir.name
    52	    baseline = "B0" if name.startswith("B0") else ("B1" if name.startswith("B1") else "?")
    53	    for site in ("classifieds", "reddit", "shopping_admin", "shopping"):
    54	        if f"_{site}_" in name or name.endswith(f"_{site}"):
    55	            return baseline, site
    56	    m = re.search(r"_(classifieds|reddit|shopping_admin|shopping)", name)
    57	    return baseline, m.group(1) if m else "?"
    58	
    59	
    60	def main() -> int:
    61	    ap = argparse.ArgumentParser()
    62	    ap.add_argument("--runs", nargs="+", default=[str(p) for p in DEFAULT_RUNS],
    63	                    help="run dirs to aggregate")
    64	    ap.add_argument("--output", default=str(REPO / "results/phantom_paper/auroc_cross_condition.csv"))
    65	    args = ap.parse_args()
    66	
    67	    rows: list[pd.DataFrame] = []
    68	    for run_str in args.runs:
    69	        run_dir = Path(run_str)
    70	        cm_path = run_dir / "analysis/signals/combined/tables/cross_mode_auroc.csv"
    71	        single_path = run_dir / "analysis/signals/combined/tables/auroc_all_metrics.csv"
    72	        baseline, site = parse_run_id(run_dir)
    73	        if cm_path.exists():
    74	            df = pd.read_csv(cm_path)
    75	            if df.empty:
    76	                continue
    77	        elif single_path.exists():
    78	            # Single-condition (e.g. phantom) runs — derive mode from condition dir name
    79	            cond_dirs = [d for d in run_dir.glob("phase1_*") if d.is_dir()]
    80	            if not cond_dirs:
    81	                print(f"  [skip] {run_dir.name}: no condition dir")
    82	                continue
    83	            mode = cond_dirs[0].name.replace("phase1_", "").replace("_router_0", "")
    84	            df = pd.read_csv(single_path).rename(columns={"metric": "signal"})
    85	            df = df.assign(mode=canonical_mode(mode))
    86	        else:
    87	            print(f"  [skip] {run_dir.name}: no AUROC tables")
    88	            continue
    89	        if "mode" in df.columns:
    90	            df["mode"] = df["mode"].map(lambda value: canonical_mode(str(value)))
    91	        df = df.assign(baseline=baseline, site=site, run_id=run_dir.name)
    92	        rows.append(df)
    93	
    94	    if not rows:
    95	        print("No AUROC data found in any run.")
    96	        return 1
    97	
    98	    full = pd.concat(rows, ignore_index=True)
    99	    full = full[[
   100	        "baseline", "site", "mode", "signal", "signal_type",
   101	        "AUROC", "AUROC_ci_lower", "AUROC_ci_upper", "n", "run_id",
   102	    ]]
   103	    full = full.sort_values(["baseline", "site", "mode", "AUROC"], ascending=[True, True, True, False])
   104	
   105	    out = Path(args.output)
   106	    out.parent.mkdir(parents=True, exist_ok=True)
   107	    full.to_csv(out, index=False)
   108	    print(f"wrote {out} ({len(full)} rows)")
   109	
   110	    # Markdown top-3 per (baseline, site, mode)
   111	    md = out.with_suffix(".md")
   112	    lines = [
   113	        "# Cross-condition routing signal AUROC (top-3 per cell)",
   114	        "",
   115	        "AUROC ≥ 0.5 means signal correlates with success; CI from 1000-resample bootstrap.",
   116	        "",
   117	        "| Baseline | Site | Mode | Signal | AUROC | 95% CI | n |",
   118	        "|---|---|---|---|---:|---|---:|",
   119	    ]
   120	    grouped = full.groupby(["baseline", "site", "mode"], dropna=False)
   121	    for (b, s, m), grp in grouped:
   122	        top = grp.nlargest(3, "AUROC")
   123	        for _, r in top.iterrows():
   124	            ci = ""
   125	            if pd.notna(r["AUROC_ci_lower"]) and pd.notna(r["AUROC_ci_upper"]):
   126	                ci = f"[{r['AUROC_ci_lower']:.3f}, {r['AUROC_ci_upper']:.3f}]"
   127	            lines.append(
   128	                f"| {b} | {s} | {m} | {r['signal']} | "
   129	                f"{r['AUROC']:.3f} | {ci} | {int(r['n'])} |"
   130	            )
   131	    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
   132	    print(f"wrote {md}")
   133	
   134	    # Cross-mode max-AUROC summary (paper-ready Section 6 claim support)
   135	    summary_md = out.parent / "auroc_cross_condition_summary.md"
   136	    summary_lines = [
   137	        "# Routing signal AUROC summary — max per (baseline, site, mode)",
   138	        "",
   139	        "Section 6 claim: AUROC ≥ baseline (DOM/SoM/Vision) for Phantom modes.",
   140	        "",
   141	        "| Baseline | Site | Mode | Max-AUROC signal | AUROC | 95% CI | n |",
   142	        "|---|---|---|---|---:|---|---:|",
   143	    ]
   144	    for (b, s, m), grp in grouped:
   145	        top = grp.nlargest(1, "AUROC")
   146	        if top.empty:
   147	            continue
   148	        r = top.iloc[0]
   149	        ci = ""
   150	        if pd.notna(r["AUROC_ci_lower"]) and pd.notna(r["AUROC_ci_upper"]):
   151	            ci = f"[{r['AUROC_ci_lower']:.3f}, {r['AUROC_ci_upper']:.3f}]"
   152	        summary_lines.append(
   153	            f"| {b} | {s} | {m} | {r['signal']} | "
   154	            f"{r['AUROC']:.3f} | {ci} | {int(r['n'])} |"
   155	        )
   156	    summary_md.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
   157	    print(f"wrote {summary_md}")
   158	
   159	    return 0
   160	
   161	
   162	if __name__ == "__main__":
   163	    raise SystemExit(main())

exec
/bin/bash -lc "nl -ba docs/checkpoints/pre_run/preregistration.md | sed -n '46,86p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
    46	## §2 Hypotheses
    47	
    48	### PRIMARY family (gates paper claim)
    49	
    50	#### H1 — Hero deployment claim (P-SoM is hidden routing arm)
    51	
    52	P-SoM drop-one oracle ceiling lift > 0 across statistical cells (each cell = one (site, model) stratum), satisfying ALL two PRIMARY sub-conditions:
    53	
    54	- **H1(i)** Pooled DerSimonian-Laird random-effect meta-analysis on N=4 (site, model) cells reaches significance at Holm α=0.05 (PRIMARY family m=1 test, no within-family correction needed).
    55	- **H1(ii)** Pooled magnitude θ_RE ≥ 1.0pp AND one-sided **superiority test** rejects H0: θ ≤ 1.0pp at α=0.05 (i.e., effect is significantly ABOVE the +1.0pp substantive-effect threshold; commit-locked). Note 2026-05-13: replaces prior "TOST equivalence rejected at δ" wording which was ambiguous in direction; one-sided superiority is the unambiguous statistical test for "effect substantively > δ".
    56	
    57	**Drop-one definition (operational)**: For each (site, model) cell containing all 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM), compute oracle ceiling SR over {6 modes} minus oracle ceiling SR over {5 modes drop P-SoM} per task, then average across the cell's task pool. Paired 1000-resample task-level bootstrap CI per cell; pooled DerSimonian-Laird across 4 cells.
    58	
    59	**Transparency consistency check (NOT gating, reported alongside H1)**: K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually clear Holm α=0.05 within the per-cell P-SoM sub-family (m = 4). **K-of-N reclassified pre-data 2026-05-13** from gating threshold to transparency consistency check, based on power analysis (`docs/analysis/cross_sites/power_analysis.md`) showing per-cell power at observed 1-3pp effect sizes is < 10% — calibrated only for ≥7pp effects, smaller than reasonable phenomenon effect size, so K-as-gate is statistically dysfunctional. See §4 audit B9 row + Appendix A 2026-05-13 entry.
    60	
    61	#### H2 — 4-fold drop-in property (P-SoM specifically)
    62	
    63	All four sub-claims hold per cell, replicated in ≥ K_h1 cells:
    64	
    65	- **(a) Cost** — median cost(P-SoM) within ±10% of median cost(DOM); reflects the by-construction property that `[SOM_MARKS]` is an AXTree regex filter (no image embedding tokens). Tested empirically per cell.
    66	- **(b) Latency** — median latency(P-SoM) ≤ 0.6 × median latency(SoM); reflects skipping image inference stage. Tested empirically per cell.
    67	- **(c) Signal AUROC** — top-1 routing-signal AUROC(P-SoM) ≥ AUROC(DOM) − 0.05 (within 5pp). Tested empirically per cell, signal selected per `aggregate_routing_auroc.py` top-1.
    68	- **(d) Drop-one magnitude** — folded into H1(iii); P-SoM contributes ≥ 1.0pp lift on average.
    69	
    70	#### H3 — Phantom space 2-axis empirical structural claim
    71	
    72	Each phantom-space axis (axis 1 = text payload via P-text; axis 2 = SoM-style prompt via P-prompt) contributes tasks NOT solved by P-SoM, evidencing axis decomposition is empirically non-trivial (i.e., phantom space is a multi-region 2D structure, not a collapsed 0D point).
    73	
    74	H3 statistical cells = 4 (one per (site, model)). H3 axis-1 and axis-2 are tested separately within each cell.
    75	
    76	- **H3(i) PRIMARY GATE** axis 1: pooled across N=4 cells, mean |P-text ∖ P-SoM| > 0 with DerSimonian-Laird random-effects meta CI excluding 0 (Holm α=0.05, m=1 within axis-1 sub-family).
    77	- **H3(ii) PRIMARY GATE** axis 2: same as H3(i) for |P-prompt ∖ P-SoM|.
    78	- **H3(iii)** Per-cell unique-count noise floor: ≥ 2 tasks (≈ 1pp at N=234 to N=210); 1 task is noise floor, excluded from cell-level pass.
    79	
    80	**Transparency consistency check (NOT gating)**: K_h3 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap 95% CI excluding 0 (m=4 per axis). Same K-of-N reclassification rationale as H1 (see §4 audit B9 + Appendix A 2026-05-13 entry).
    81	
    82	**Test details**:
    83	- Primary gating: bootstrap CI on unique-count, 1000 resamples.
    84	- Secondary report: McNemar exact one-sided directional asymmetry test (informational only — McNemar tests if one axis dominates the other in unique contribution; H3 only requires non-emptiness, not dominance).
    85	- Multiple-comparison: Holm-Bonferroni step-down per axis sub-family (axis 1: m = N_cells; axis 2: m = N_cells).
    86	

exec
/bin/bash -lc "nl -ba docs/checkpoints/pre_run/preregistration.md | sed -n '138,176p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   138	### FRAMING DECISION RULE (pre-registered, data-conditional)
   139	
   140	The paper §1 hook framing maps to data outcomes as follows:
   141	
   142	| Rule | Conditions | Paper hook framing | Hook power |
   143	|---|---|---|---|
   144	| **R1** | H1 holds AND H2 (a)(b)(c) all hold AND H3(i) holds AND H3(ii) holds | "Phantom routing space (M1/M2 2-axis empirical structure); P-SoM as deployment hero, P-text/P-prompt as structural ablation arms validating axis decomposition." | STRONGEST |
   145	| **R2** | H1+H2 hold AND only one of H3(i)/(ii) holds | "Phantom routing space (single-axis empirical structure) with P-SoM as deployment hero; remaining axis decomposition theoretical (Zoom 1 architectural argument only)." | MODERATE-STRONG |
   146	| **R3** | H1+H2 hold AND neither H3(i)/(ii) holds | "Phantom-SoM is hidden 4th routing arm; M1/M2 axis decomposition supported by Zoom 1 architectural argument only, not empirically validated by ablation." | MODERATE (= 04-30 fallback; workshop-grade) |
   147	| **R4** | H1 holds AND H2 partially fails (e.g., (a) cost or (b) latency fails on some site) | "Phantom-SoM partial drop-in" + §4 disclosure of failed sub-claim. | WEAK; substantial revision |
   148	| **R5** | H1 fails (pooled meta DerSimonian-Laird Holm α=0.05 fails OR pooled magnitude θ_RE < 1.0pp OR TOST equivalence fails reject at δ=1.0pp) | Paper death scenario: pivot to VWA bug audit paper (§107 4-cluster fix as primary) OR abandon. Decision deferred to advisor sync at fail time. | n/a |
   149	
   150	**Trigger rule update 2026-05-13**: R5 no longer fires on `< K_h1` (K-of-N reclassified to transparency-only). Pooled meta + TOST primary gate only. K-of-N consistency reported in §4 per-cell table as descriptive transparency row.
   151	
   152	**Heterogeneity-conditional rule (added 2026-05-13 to resolve §4 audit B8 ↔ H1(i) conflict)**: If pre-specified I² > 75% from random-effects meta (per §4 audit B8 thresholds), do NOT pool — primary inference reverts to per-cell forest + meta-regression by site / model. R1-R5 framing in this branch maps to per-cell direction-consistency: ≥3 of 4 cells direction-positive + ≥2 individually Holm sig → R3-grade hook; otherwise R4/R5.
   153	
   154	---
   155	
   156	## §3 Multiple-Comparison Family Declaration
   157	
   158	**PRIMARY family** (gating paper hook) — UPDATED 2026-05-13 (K-of-N → transparency-only):
   159	- H1(i) pooled meta on N=4 statistical cells: m = 1 (no within-family correction).
   160	- H1(ii) pooled magnitude θ_RE ≥ 1.0pp + TOST equivalence reject at δ=1.0pp: m = 1.
   161	- H2 sub-claims (a)(b)(c)(d) per cell: m = 4 × 4 statistical cells = 16 tests (each per-cell sub-claim).
   162	- Method: Holm-Bonferroni step-down per H-sub-family (Holm 1979).
   163	
   164	**STRUCTURAL family** (gating phantom-space framing) — UPDATED 2026-05-13:
   165	- H3(i) pooled axis-1 meta on N=4 cells: m = 1.
   166	- H3(ii) pooled axis-2 meta on N=4 cells: m = 1.
   167	- Method: Holm-Bonferroni step-down per axis sub-family.
   168	- Rationale: structural claim is weaker than deployment, separate family avoids inflating PRIMARY family m count.
   169	
   170	**TRANSPARENCY family** (NOT gating, reported in §4 per-cell table for reviewer transparency):
   171	- K_h1 = ⌈0.75 × 4⌉ = 3 of 4 cells individually Holm-significant on P-SoM drop-one (m=4 per cell).
   172	- K_h3 axis-1 = ⌈0.67 × 4⌉ = 3 of 4 cells individually with bootstrap CI excluding 0.
   173	- K_h3 axis-2 = same as axis-1.
   174	- Method: Holm-Bonferroni within transparency sub-family (m=4 per K-test).
   175	- **Rationale for transparency-only reclassification**: power analysis (`docs/analysis/cross_sites/power_analysis.md`, pre-data) shows K-of-N family power at observed 1-3pp effect sizes is < 10%, calibrated only for ≥7pp effects. Per-cell N=234 (cls) / 210 (red) bootstrap power at 1.5pp effect ≈ 0.30. P(≥3 of 4 cells sig | p_cell=0.30) ≈ 8%. K-as-gate is statistically dysfunctional in this regime; K-as-transparency provides per-cell consistency check value alongside pooled meta. See Appendix A 2026-05-13 entry.
   176	

exec
/bin/bash -lc "nl -ba docs/checkpoints/pre_run/preregistration.md | sed -n '199,233p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   199	## §4 Locked Analysis Choices (pre-data)
   200	
   201	| Choice | Value | Rationale |
   202	|---|---|---|
   203	| **Primary metric** | Oracle ceiling SR pp lift (binary, paired) | Standard routing-arm contribution metric |
   204	| **CI method** | 1000-resample task-level paired bootstrap, **percentile** intervals (BCa as sensitivity check, not primary) | Existing infra in `aggregate_phantom_lift.py`. Percentile chosen primary because: (a) paired-bootstrap on bounded proportion (SR ∈ [0,1]) → BCa acceleration estimate is unstable at small N per cell; (b) Cohen's h transformation already symmetrizes; (c) percentile is the canonical reporting in WebArena/VWA precedent. BCa shown as appendix sensitivity check. |
   205	| **Bootstrap resampling unit** | **Task-level** (not episode-level, not run-level) | Each (task_id) drawn with replacement N times; same task across modes drawn together to preserve pairing. This is the standard unit for adjusted_success comparisons in VWA/WA. Episode-level would break pairing; run-level would over-conservatively widen CIs. |
   206	| **Bootstrap clustering** | **Single-level (task_id)** for primary, no nested cluster (cell × site) bootstrap | Justification: meta-analysis at cell level is separate (`aggregate_phantom_meta.py` random-effects + I²/τ²); within-cell bootstrap only re-samples tasks. Multi-level cluster would double-count uncertainty already captured by random-effects meta. Lock: percentile + task-id unit + no nested cluster (B2 lock 2026-05-09). |
   207	| **Sig threshold** | Holm α=0.05 within respective family | FWER control |
   208	| **Effect size (binary)** | Cohen's h with bootstrap CI | Standard for proportion comparisons |
   209	| **Effect size (continuous)** | Cohen's d with bootstrap CI | For cost/latency H2(a)(b) |
   210	| **TOST equivalence margin δ** | **1.0pp** | ≈ 2 tasks in N=234, matches per-cell bootstrap SE; smaller is within sampling noise floor |
   211	| **H1 K_h1 transparency ratio** | **0.75** (= 3/4 cells; **transparency-only, not gating** per 2026-05-13 reclassification) | Reports per-cell consistency alongside pooled meta; not a gate on H1 |
   212	| **H3 K_h3 transparency ratio** | **0.67** (= 3/4 cells; **transparency-only**) | Same as K_h1 reclassification rationale |
   213	| **H3 unique-count floor** | **≥ 2 tasks per cell** | 1 task is sampling noise; 2 tasks ≈ 1pp at N=234 |
   214	| **Cell inclusion (Phase 1a main)** | Phase A post-fix only (commit ≥ 3c15cd7), cls + red sites only, all 6 modes per (site, model) cell freshly rerun | Bug-clean rerun + workshop-target scope (shop deferred to Phase 1b) |
   215	| **Cell inclusion (Phase 1b main paper)** | Phase A post-fix rerun of shop × B0+B1 × 6 modes (12 conditions added on top of Phase 1a 24 conditions) | Cross-site expansion lever for main paper, post-data R1 vs Option D framing decision |
   216	| **Cell inclusion (Appendix D)** | Archived pre-Phase-A data as robustness check | Symmetric contamination disclosure |
   217	| **N inclusion floor** | ≥ 100 ep per (condition) | Statistical power baseline |
   218	| **FP filter primary** | na_fp + eval_fp combined | Per 实验笔记 §95 (visual_fp deprecated — no lit precedent, boundary-undecidable, over-filters 95.3% VWA tasks). Code: `compute_adjusted_success()` returns `fp_reason ∈ {'', 'na_fp', 'eval_fp'}` (`p79/experiment/analysis.py:52`) |
   219	| **FP filter sensitivity** | 3 variants reported (raw_SR / +na_fp only / +na_fp+eval combined) | Robustness disclosure. visual_fp is NOT in the ladder — see §95 decision rationale |
   220	| **Non-visual subset robustness** | 43 VWA + 480 WA = 523 manually-audited non-visual tasks (`docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py`) | Replaces deprecated visual_fp; Appendix D sensitivity check |
   221	| **Mode operational definitions** | 6 modes per paper §3 (text format × prompt × image): DOM (AXTree+DOM-prompt+no image) / SoM ([SOM_MARKS]+SoM-prompt+image) / Vision (no text+image) / P-text ([SOM_MARKS]+DOM-prompt+no image) / P-prompt (AXTree+SoM-prompt+no image) / P-SoM ([SOM_MARKS]+SoM-prompt+no image) | Stipulative — **no post-hoc episode reclassification**. Episodes systematically excluded per (FP filter / N-floor / data-corruption flag), never redefined which mode they belong to. Edge cases (empty AXTree / 0 marks / OCR-empty) follow `condition_meta.json` declared mode |
   222	| **Routing signal universe** | `aggregate_routing_auroc.py` enumerated set: ep_mean_verbalized / ep_min_verbalized / max_repeat_streak / action_diversity / url_revisit_count / url_revisit_max / action_unique_types / url_unique_count / ep_mean_logprob / ep_min_logprob (last 2 B1-only) | **No post-hoc engineered features** for router input. Best-signal-per-mode characterization is exploratory (§5) — paper §6 portfolio finding, not pre-registered prediction |
   223	| **Router train/test split** | 5-fold site-stratified CV on cls+red post-Phase-A task pool, seed=42, min test fold ≥ 40 tasks | Reproducible split via `scripts/analysis/router_split.py` (TBD). **Test fold predictions use ONLY train-fold mode rankings** to prevent oracle leak. Pending advisor 5/5 sync alternative: leave-one-site-out (LOSO) — test cls hold-out trained on red, vice versa |
   224	| **Failure-mode classification rubric** | 5-bucket: `early_finish` / `wrong_commit` / `visual_hijack` / `click_loop` / `persistent_error` per `docs/analysis/disagreement_clusters.md` decision tree | Pre-data inter-annotator agreement target Cohen κ ≥ 0.7 on 30-task pilot (codex prompt + 1 human spot-check). Buckets remain in the rubric but the paper §1 "+43.7pp B0/B1 capability shift" prose was dropped 2026-05-09 (third contribution cut from paper). Failure-mode classification still used for §8 limitations and supplement S.X if needed. |
   225	| **N_conditions Phase 1a (operational)** | **24 conditions** = 2 sites (cls, red) × 2 models (B0, B1) × 6 modes (DOM, SoM, Vision, P-text, P-prompt, P-SoM). Each condition launched fresh post-fix via `scripts/queues/queue_phase1_paper_grade.sh` (renamed 2026-05-13 from `queue_16cell_paper_grade.sh`; current scope = 24 conditions Phase 1a + 12 conditions Phase 1b deferred). Sequence: B0 → B1 per site (shared user account); cls + red parallel chains | ✅ **Student-decided 2026-05-13** post-codex stress audit. Workshop-targeted (cls + red only, shop deferred to Phase 1b for main paper). Replaces prior 16-cell phantom-only scope that lacked baseline DOM/SoM/Vision rerun (codex Flaw 1) |
   226	| **N_cells statistical (H1/H3 stratification)** | **4 cells** = (site, model) tuples: (cls, B0), (cls, B1), (red, B0), (red, B1). Drop-one is computed per cell using all 6 modes; pooled DerSimonian-Laird random-effects meta across 4 cells | Cell = paired-test stratification unit (one per (site, model)), distinct from "condition" (one per (site, model, mode)). 4 cells × 6 modes = 24 conditions. Distinction propagated to all prose / queue / docs 2026-05-13 |
   227	| **N_conditions Phase 1b (main paper, deferred)** | **+12 conditions** = shop × 2 models × 6 modes. Launches after Phase 1a workshop submission to feed main paper R1 / Option D framing decision. N_cells statistical becomes 6 (= 3 sites × 2 models) when Phase 1b lands | Phase 1b is additive; workshop §1 hook does NOT depend on Phase 1b. Main paper §1 hook upgrade R3 → R1 conditional on shop replicating P-SoM 4-fold within ±2pp tolerance |
   228	| **Best-single-mode baseline (H7/H8 anchor)** | Per cell: mode with highest mean adjusted-SR on train fold | Used as comparison anchor for router lift; **train/test split-stratified** to prevent test leak |
   229	| **Missing-data / crashed-episode policy** (audit B6) | (a) Crashed episodes (uncaught exception, OOM, timeout > 30 min, browser crash) **excluded from paired-N denominators**, **NOT imputed** to success or failure. (b) Episodes with `not_logged_in` or `auth_drift` flag at termination excluded after watchdog refresh fails 3 retries (per `experiment_watchdog.py`). (c) Missing artifacts (no `obs.txt` / `screenshot_annotated.png` at step k) excluded from per-step analyses, NOT imputed. (d) Per-cell exclusion count + reason histogram reported in Appendix C. | Listwise deletion only; mean imputation introduces bias for SR proportions, hot-deck imputation breaks paired-N pairing. Crashed-episode imputation as success/failure would inflate Type I/II error. Lock 2026-05-09. |
   230	| **Stopping rules / contamination halt criteria** (audit B7, REVISED 2026-05-13 to remove outcome-dependent bias per codex Flaw 6) | (a) **Pre-launch**: `make pre-launch-check` validates seed configured + HF SHA pinned + git working tree clean + GPU available + disk free > 20GB; failure halts launch (per audit C10). (b) **Smoke-test gate (outcome-INDEPENDENT)**: first 10 episodes per condition must show auth-state `logged_in=True` on all 10 AND ≥ 9 of 10 episodes produced complete artifact bundle (`obs.txt` + `screenshot.png` + `condition_summary_v2` increment + JSONL flush) AND evaluator returned a parseable verdict (success / failure / `ua_match` N/A — any of these is fine, **success rate itself is NOT checked**). Failures halt for auth refresh / artifact pipeline debug, NOT for low SR observation. Rationale: outcome-dependent smoke gate biases low-SR cells upward (a true 5-10% SR cell has 35-60% probability of "0 successes in first 10" by binomial chance and would be invalidly restarted). (c) **Auth/site contamination halt**: ≥ 5 consecutive episodes with `not_logged_in` ⇒ stop cell, refresh auth, archive partial run as `_dirty_partial`, restart fresh. (d) **Eval drift halt**: if rerun on identical archived episode produces SR delta > 5pp via `validate_run.py --strict`, freeze cell + investigate evaluator code. (e) **OOM / hardware halt**: 3 consecutive job failures ⇒ stop cell, document hardware in incident log, manually re-queue with diagnostic output. | Halt rules protect data purity; halted cells restarted only after root-cause documented in `master_bug_catalog.md` + bug fix committed. Lock 2026-05-09; smoke gate revised 2026-05-13 to outcome-independent variant. |
   231	| **Heterogeneity (random-effects, Q, I², τ²) pre-spec** (audit B8) | (a) **Primary estimator**: random-effects DerSimonian-Laird via `aggregate_phantom_meta.py` (already implemented). (b) **Heterogeneity reporting**: report Cochran Q (chi² test of homogeneity), I² (% of total variance attributable to between-cell heterogeneity), τ² (between-cell variance). (c) **Interpretation thresholds (pre-specified)**: I² < 25% = "low heterogeneity, pooled mean is primary"; 25%-50% = "moderate, report both pooled + per-cell"; 50%-75% = "high, per-cell estimates are primary, pooled is summary"; > 75% = "very high, do not pool — report only per-cell + heterogeneity-source analysis (site / model / task-pool)". (d) **Heterogeneity-source decomposition**: when I² > 50%, report meta-regression by site (cls / red / shop) and by model (B0 / B1) to identify dominant variance source. | Higgins & Thompson 2002 (I² thresholds). Per-cell estimates always shown alongside pooled, so heterogeneity is never averaged away. Lock 2026-05-09. |
   232	| **K-of-N rule scope** (audit B9 power-corrected, REPROPAGATED 2026-05-13 to H1/H3/R5/§6/Appendix A) | The **K_h1=3/4 / K_h3=3/4** ratios (under 24-condition / 4-cell Phase 1a scope) are **transparency consistency checks** (count of cells *individually* clearing α=0.05 Holm), **NOT gates on H1/H3 paper claims**. **Primary gate** = (a) DerSimonian-Laird random-effects meta-analysis on N=4 (site, model) cells + (b) TOST equivalence on pooled cls + red tasks at δ=1.0pp. Per `docs/analysis/cross_sites/power_analysis.md` §3-§5, K-of-N family power at observed 1-3pp effect sizes is < 10%; the rule is calibrated for ≥7pp effects (1.5pp per-cell power ≈ 0.30; P(≥3 of 4 cells sig) ≈ 8%). K-as-gate is statistically dysfunctional in this effect-size regime. **2026-05-13 propagation**: prior prereg text in H1(ii) / H3(i) / H3(ii) / R5 / §6 still gated K-of-N → fixed to "transparency consistency check, reported alongside but NOT gating". This is **pre-data reclassification**: power analysis commit predates Phase 1a launch; reclassification timestamp recorded for OSF witness audit trail. | Original audit B9 lock 2026-05-09 introduced framing but did not propagate to H1/H3/R5/§6 prose (codex stress audit 2026-05-13 Flaw 2 surfaced internal contradiction). Repropagation 2026-05-13 reconciles all references. |
   233	

exec
/bin/bash -lc "nl -ba docs/analysis/cross_sites/power_analysis.md | sed -n '1,80p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Power Analysis — Observed-SR Update (B9 ✓)
     2	
     3	**Last updated**: 2026-05-09 (B9 audit constraint, post-bugfix)
     4	
     5	This appendix updates the pre-registered power analysis with **observed adjusted-SR levels** from `sr_fp_per_mode.md` (Phase 1 B0 + B1 done cells, pre-paper-grade rerun). The post-rerun version will replace this file once 16-cell aggregation completes.
     6	
     7	## 1. Observed adjusted-SR ranges (per `sr_fp_per_mode.md`)
     8	
     9	| Site | B0 range (DOM/SoM/Vision/P-text/P-prompt/P-SoM) | B1 range (DOM/SoM/Vision/P-text/P-SoM) | Pooled median |
    10	|---|---|---|---|
    11	| classifieds | 13.7-21.4% | 7.3-13.7% | ~13.7% |
    12	| reddit | 6.7-13.8% | 2.4-7.6% | ~9.5% |
    13	| shopping | TBD post-rerun | TBD post-rerun | TBD |
    14	
    15	**Observed effect-size range** (phantom-mode minus best non-phantom baseline):
    16	- B0 reddit P-SoM vs DOM: +4.3pp (largest phantom uplift in done cells)
    17	- B0 cls SoM vs DOM: +7.3pp (largest non-phantom contrast)
    18	- B0 cls P-SoM vs DOM: +0.4pp (smallest contrast)
    19	- B1 cls SoM vs DOM: +5.1pp
    20	- B1 cls P-SoM vs DOM: -0.9pp (negative — phantom does not always uplift)
    21	
    22	**Modal effect size**: 1-5pp range, with phantom modes clustered at 0-4pp.
    23	
    24	## 2. Per-cell MDE at observed SR levels (paired design, α=0.05 two-sided, β=0.20)
    25	
    26	Run: `python3 scripts/analysis/power_analysis.py --baseline-sr {0.10,0.15,0.20}`
    27	
    28	| Site | N | MDE @ SR=0.10 | MDE @ SR=0.15 | MDE @ SR=0.20 |
    29	|---|---:|---:|---:|---:|
    30	| classifieds | 234 | 5.5pp | 6.5pp | 7.4pp |
    31	| reddit | 210 | 5.8pp | 6.9pp | 7.8pp |
    32	| shopping | 466 | 3.9pp | 4.6pp | 5.2pp |
    33	
    34	**Key observation**: minimum detectable effect at 80% per-cell power is **5-7pp** for cls/red, **4-5pp** for shop. The **observed mechanism effect (1-5pp)** is at or below per-cell MDE in 2 of 3 sites — **per-cell power for typical phantom effects is < 50%**.
    35	
    36	## 3. Family-wise power at observed effects (K-of-N rule, baseline SR=0.15 proxy)
    37	
    38	| Per-cell power (proxy effect on smallest site) | K_h1=12/16 family power | K_h3=11/16 family power |
    39	|---|---:|---:|
    40	| 0.06 (1pp) | <0.001 | <0.001 |
    41	| 0.13 (2pp) | <0.001 | <0.001 |
    42	| 0.23 (3pp) | <0.001 | <0.001 |
    43	| 0.53 (5pp) | 0.061 | 0.151 |
    44	| 0.80 (~6.5pp) | 0.798 | 0.918 |
    45	| 0.90 (~7.5pp) | 0.983 | 0.997 |
    46	
    47	**Interpretation**:
    48	- **K_h1=12/16** is calibrated for **≥7pp effects** with paper-grade ≥0.80 family power. For typical phantom mechanism effects (1-5pp), K_h1 family power is **<10%**.
    49	- **K_h3=11/16** is slightly more permissive but still requires per-cell power ≥0.65 (≈6pp effect at SR=0.15) to reach 0.49 family power.
    50	
    51	## 4. Methodological implication & paper-§3 framing update
    52	
    53	The K-of-N family-wise rule was originally pre-registered as a **transparency / aggregation** check, not the primary detection mechanism. With the corrected interpretation:
    54	
    55	- **Primary effect-detection test** = DerSimonian-Laird random-effects meta-analysis (locked by B8) on cells with N≥10. This is power-adequate at the cross-cell level for effects ≥2pp.
    56	- **Equivalence test (TOST)** = pooled across cls+red+shop tasks (N=234+210+466=910), δ=1.0pp margin. Sufficient CI width for 1pp resolution.
    57	- **K-of-N rule** = retained as a **secondary transparency check** documenting how many cells *individually* clear α=0.05; not a gate on the H1/H3 paper claims.
    58	- This recharacterization is **not post-hoc cherry-picking**: the random-effects meta + TOST were always the primary tests in `preregistration.md §4`. The K-of-N rule is restated as transparency.
    59	
    60	## 5. Reviewer-rebuttal language
    61	
    62	"At observed adjusted-SR levels (8-15% across sites) and observed mechanism effect sizes (1-5pp), per-cell statistical power is below 0.55 in two of three sites. We therefore rely on (a) DerSimonian-Laird random-effects meta-analysis across all cells (B8 lock; cross-cell pooling raises effective power) and (b) TOST equivalence on the full N=910 pooled task set (δ=1.0pp margin) as primary tests. The K-of-N family-wise rule pre-registered for transparency is not powered for sub-5pp effects, and we report its outcome as a secondary observation rather than a gate on the main hypotheses."
    63	
    64	## 6. Bug history
    65	
    66	The original `power_analysis.py` (pre-2026-05-09) contained a stale interpretation block claiming "K_h1=12/16 with per-cell power=0.65 → >0.95 family-wise power" — this was numerically inconsistent with the K-of-N table on the same page (actual value 0.289). Fixed in commit (current session) to compute family power **at observed effect sizes** rather than at hypothetical per-cell power levels. The corrected version is what this appendix relies on.
    67	
    68	**Source files**:
    69	- `scripts/analysis/power_analysis.py` (script, fixed)
    70	- `docs/analysis/cross_sites/power_analysis_observed_sr10.md` / `sr15.md` / `sr20.md` (full per-baseline tables)
    71	- `docs/analysis/cross_sites/sr_fp_per_mode.md` (observed SR source)
    72	- `docs/checkpoints/pre_run/preregistration.md §4` (B8 random-effects lock + TOST policy)
    73	
    74	## 7. Open items (post-rerun)
    75	
    76	- [ ] Update with shopping observed adjusted-SR once 16-cell rerun completes (cells G/H pending)
    77	- [ ] Re-verify K-of-N rule reframing in `preregistration.md` (audit item A1 / advisor review)
    78	- [ ] Add `power_analysis_post_rerun.md` once observed effect sizes are final (replace this file)

codex
At implementation level, the largest problems are not the DL algebra itself; it is mostly transcribed correctly. The failure mode is that the prereg says several conditional rules matter, but the canonical script does not enforce them, especially heterogeneity and the full H2 drop-in criteria.
exec
/bin/bash -lc "python3 scripts/analysis/preregistration_decision_test.py --synthetic --seed 42 --scenario r1_pass >/tmp/prereg_synth.json && python3 - <<'PY'
import json
p='/tmp/prereg_synth.json'
d=json.load(open(p))
print(d['framing_rule'])
print(d['primary_gate_summary'])
print(d['H1_psom_drop_one']['primary_gate']['pooled_meta'])
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 939ms:
{'rule': 'R1', 'framing': 'Phantom routing space (2-axis empirical structure)', 'hook_power': 'STRONGEST'}
{'H1': 'PASS', 'H2': 'PASS', 'H3_axis1': 'PASS', 'H3_axis2': 'PASS'}
{'pooled_effect': 4.648785956943824, 'pooled_se': 0.8432433656780929, 'pooled_ci_95': [2.996028960214762, 6.301542953672886], 'Q': 3.8498487271629966, 'Q_df': 3, 'I_squared_pct': 22.0748602709193, 'tau_squared': 0.6342046820184201, 'p_value_two_sided': 3.528032310740059e-08, 'z_statistic': 5.512982545917227, 'k': 4}

exec
/bin/bash -lc "nl -ba docs/reference/EVIDENCE_LAYER_AUDIT.md | sed -n '120,230p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   120	## §2 Pre-registration template (T0e, blocks rerun launch)
   121	
   122	> **创建** `docs/checkpoints/pre_run/preregistration.md` **with this skeleton**, advisor sync 时 lock + sign. Git commit SHA + timestamp = registration time. OSF DOI (optional, paper submission 前 1 周上传) = third-party witness.
   123	>
   124	> **Epistemic 结构** (核心 design move 2026-05-03 reframe):
   125	> - **Hero claim** (P-SoM as deployment routing arm) — pre-registered strict
   126	> - **4-fold drop-in property** — pre-registered strict (4 sub-claims a/b/c/d)
   127	> - **2-axis structural claim** (phantom space is multi-region, not collapsed point) — pre-registered with low-threshold non-overlap evidence requirement
   128	> - **Framing decision rule** — pre-registered, data-conditional (paper hook 升降级 mapping)
   129	> - **Theory predictions (别扭, capability-reversal)** — marked post-hoc explanatory, no gating
   130	
   131	```
   132	---
   133	type: preregistration
   134	status: locked
   135	registered_at: <yyyy-mm-dd HH:MM BST>
   136	registered_git_sha: <40-char>
   137	witnessed_by: <advisor name>
   138	osf_doi: <optional>
   139	data_lock_until: <14-cell rerun completion timestamp>
   140	---
   141	
   142	# Phantom-SoM Pre-Registration
   143	
   144	## Hypotheses
   145	
   146	### PRIMARY (gates paper claim)
   147	
   148	H1 (Hero deployment claim — P-SoM is hidden routing arm):
   149	  P-SoM drop-one > 0 across cells, satisfying ALL three sub-conditions:
   150	    (i)   Pooled DerSimonian-Laird random-effect meta sig at Holm α=0.05
   151	          (PRIMARY family m = 1 test, no correction needed within family)
   152	    (ii)  ≥ K_h1 of N_cells individually Holm-sig at α=0.05
   153	          within SECONDARY family m = N_cells (per-cell P-SoM tests)
   154	          where K_h1 = 0.75 (commit-locked, see Commit #1)
   155	    (iii) Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence rejected
   156	          at margin δ = 1.0pp (commit-locked, see Commit #2)
   157	
   158	H2 (4-fold drop-in property — P-SoM specifically):
   159	  All four sub-claims hold per cell, replicated in ≥ K_h1 cells:
   160	    (a) median cost(P-SoM) within ±10% of median cost(DOM)
   161	    (b) median latency(P-SoM) ≤ 0.6 × median latency(SoM)
   162	    (c) top-1 signal AUROC(P-SoM) ≥ AUROC(DOM) − 0.05
   163	    (d) P-SoM drop-one magnitude ≥ 1.0pp (=H1 (iii); folded)
   164	
   165	H3 (2-axis empirical structural claim — phantom space is not collapsed point):
   166	  Each phantom-space axis (axis 1 = text-payload via P-text;
   167	  axis 2 = SoM-prompt via P-prompt) contributes tasks NOT solved by P-SoM,
   168	  evidencing axis decomposition is empirically non-trivial:
   169	    (i)   axis 1: P-text ∖ P-SoM unique-task count > 0 with bootstrap
   170	          95% CI excluding 0, in ≥ K_h3 of N_cells
   171	          (lower threshold than H1: structural claim, NOT deployment)
   172	    (ii)  axis 2: P-prompt ∖ P-SoM unique-task count > 0 with bootstrap
   173	          95% CI excluding 0, in ≥ K_h3 of N_cells
   174	          where K_h3 = 0.67 (commit-locked, lower than K_h1 because
   175	          structural is weaker commit than deployment)
   176	    (iii) Per-cell unique-count ≥ 2 tasks (≈1pp at N=234); 1 task is noise
   177	          floor. Tested via exact binomial / paired McNemar one-sided.
   178	
   179	### EXPLORATORY (post-data, no pre-commit threshold)
   180	
   181	H4 (P-text / P-prompt drop-one magnitude):
   182	  Reported per cell + meta-pooled. No pre-registered ranking commitment.
   183	  Disclosed as exploratory (paper §4 prose explicit "exploratory analysis").
   184	
   185	H5 (别扭 framework predictions, 笔记 §108.16):
   186	  4 distinguishing predictions tested against 14-cell data. POST-HOC because
   187	  framework was developed after observing N=4 pre-Phase-A cells.
   188	  Reported irrespective of direction. Paper §5 prose explicit "post-hoc
   189	  theoretical framework, validated on same data motivating it; no formal
   190	  significance gating."
   191	
   192	H6 (Capability-modulated reversal):
   193	  B0 vs B1 axis-preference ranking (text-axis drop-one vs image-axis
   194	  drop-one) tested via B0 × B1 × axis GLM logistic interaction. POST-HOC
   195	  exploratory; paper §7 prose explicit "post-hoc finding."
   196	
   197	### FRAMING DECISION RULE (pre-registered, data-conditional)
   198	
   199	R1 IF (H1 holds AND H2 holds AND H3 (i) AND (ii) hold):
   200	   → Hook framing: "Phantom routing space (M1/M2 2-axis empirical structure);
   201	                    P-SoM as deployment hero, P-text/P-prompt as structural
   202	                    ablation arms validating axis decomposition."
   203	   → Paper §1 hook: STRONGEST.
   204	
   205	R2 IF (H1 holds AND H2 holds AND only one of H3 (i)/(ii) holds):
   206	   → Hook framing: "Phantom routing space (single-axis empirical structure)
   207	                    with P-SoM as deployment hero; remaining axis decomp
   208	                    theoretical (Zoom 1 architectural argument only)."
   209	   → Paper §1 hook: MODERATE-STRONG.
   210	
   211	R3 IF (H1 holds AND H2 holds AND neither H3 (i)/(ii) holds):
   212	   → Hook framing: "Phantom-SoM is hidden 4th routing arm; M1/M2 axis
   213	                    decomposition supported by Zoom 1 architectural argument
   214	                    only, not empirically validated by ablation."
   215	   → Paper §1 hook: MODERATE (= 04-30 fallback framing).
   216	
   217	R4 IF (H1 holds AND H2 partially fails — e.g., (a) cost or (b) latency
   218	       fails on some site):
   219	   → Hook framing: "Phantom-SoM partial drop-in, with site-specific
   220	                    deployment limitations" + §4 disclosure of failed sub-claim.
   221	   → Paper §1 hook: WEAK; substantial revision needed.
   222	
   223	R5 IF (H1 fails: pooled meta sig fails Holm OR < K_h1 cells individually sig):
   224	   → Paper death scenario. Reframe:
   225	      Option A: pivot to VWA bug audit paper (§107 4-cluster fix as primary)
   226	      Option B: abandon and merge findings into other paper
   227	   → Decision deferred to advisor sync at fail time.
   228	
   229	## Multiple-Comparison Family
   230	

exec
/bin/bash -lc "nl -ba docs/checkpoints/paper_planning.md | sed -n '31,47p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
    31	## §1 Paper Hook + Tagline
    32	
    33	> **2026-05-03 reframe note**: Paper hook framing is now **data-conditional** per pre-registered framing decision rule (R1-R5; see `docs/checkpoints/pre_run/preregistration.md` §2). The "core finding" below corresponds to **rule R1 (STRONGEST)** — applies if H1+H2+H3(i)+(ii) all hold post-rerun. If H3 fails, hook falls back to "Phantom-SoM is hidden 4th routing arm" (R3, MODERATE). The Hero (P-SoM deployment) + Structural ablation (P-text/P-prompt non-overlap) + Framing-rule structure replaces the older "3-arm a-priori commit" framing — see `docs/reference/EVIDENCE_LAYER_AUDIT.md` §2 for epistemic rationale.
    34	
    35	**Core finding (under R1, contingent on H3 empirical validation)**: We discover a **hidden phantom routing space** for web agents — defined by the boundary "**skip annotated image**" — containing a **2-axis empirical structure** (axis 1 = text payload via P-text; axis 2 = SoM-style prompt via P-prompt) with **P-SoM (cube center, axis 1 + axis 2 compound) as the deployment hero**. P-SoM satisfies a **4-fold drop-in property**; P-text and P-prompt serve as **structural ablation arms** validating axis decomposition:
    36	
    37	| Drop-in property | Evidence |
    38	|---|---|
    39	| (a) **Cost ≈ DOM** | `[SOM_MARKS]` 是 AXTree regex filter, 不需 bbox/image (验 `som.py::_extract_text_marks` line 24); text token ±7% (3437 vs 3661 reddit / 3008 vs 2948 cls) |
    40	| (b) **Latency ~50% lower** | cls SoM p95 74s vs Phantom-SoM 18.2s = **4× faster** (no image encoding stage) |
    41	| (c) **Signal AUROC ≥ baseline** | 5-mode 全 `overall_usable=True`; red P-text verbalized 0.793 是 5-mode 最高 (超 baseline 0.766) |
    42	| (d) **Drop-one oracle 1.7-3.8pp per phantom arm** | B0 red: P-text +3.81pp / P-SoM +3.33pp / P-prompt +2.86pp (all sig CI excludes 0); cls: P-text +3.42pp / P-SoM +2.56pp; B1 cls P-SoM +1.71pp. **Phantom space 3 arms 都贡献 unique tasks**, 6-mode oracle vs 3-mode lift +7.14pp [3.81, 10.48] (B0 reddit) |
    43	
    44	**Paper one-liner (for advisor pitch)**:
    45	> "We discover a hidden **phantom routing space** in SoM-style web agents — defined by the boundary 'skip annotated image' — containing 3 routing arms (P-text / P-prompt / P-SoM) sharing a **4-fold drop-in property**: cost ≈ DOM (no image embedding tax), ~50% lower latency (no image inference stage), signal AUROC ≥ baseline (routing infra drop-in), drop-one oracle 1.7-3.8pp per arm (all sig). Two LLM mechanisms create this space: (i) text-payload flattening (AXTree → `[SOM_MARKS]`) reframes the agent's task ontology from web-browsing to indexed selection (axis 1); (ii) SoM-style visual prompting without image still activates the agent's visual-mark referencing parsing and recovers a substantial fraction of visual structure information textually (axis 2; **Mirage Effect** Asadi et al. 2026 (arXiv:2603.21687) — VLM 无图准确率 ~70-80% of with-image; **Scaffold Effect** Vu & Balloccu 2026 — prompt mentioning modality alone explains 70-80% performance shift independent of image presence). P-SoM (cube center, axis 1 + axis 2 compound) is the space's representative arm; SoM (image-on cube endpoint) and Vision (image-only, outside cube) anchor the comparison. **The 3-axis cube framework (orthogonalizing image-presence as a controllable axis distinct from text payload and prompt format) and cube-center P-SoM (`[SOM_MARKS]` text + SoM-prompt + no image) are paper-level framework contributions** — industry deploys text-only OR SoM-with-image, never the cube-center SoM-text-without-image combination; industry uses these configurations arbitrarily for token economy, never compared P-text vs DOM nor characterized per-dimension routing behavior. Paper discovers text-flattening has independent routing effects beyond cost (drop-one unique tasks, M1 ontology reframe). The space is site-modulated (cls visual-rich requires image; red text-dominated thrives in phantom space) and routing-deployable (B0 red 6-mode oracle lift +7.14pp over 3-mode baseline)."
    46	
    47	### Cascade design (token-monotonic, paper Section 6)

exec
/bin/bash -lc "nl -ba docs/checkpoints/paper_planning.md | sed -n '81,108p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
    81	### Zoom 1 (architectural): Phantom space boundary + 2-axis activation by design
    82	
    83	**Phantom routing space** = subset of 3-axis modal cube characterized by **"skip annotated image"** boundary (axis 3 = no image)。Cube 有 8 corners (3 axes: text payload × system prompt × image)；paper 测 **4 phantom corners (cube image-off 半) + 1 image-on cube endpoint + 1 image-only mode (Vision, cube 之外)** = **5 cube modes + 1 image-only mode = 6 paper modes**:
    84	
    85	**Phantom routing space** = subset of 2×2×2 modal cube characterized by **"skip annotated image"** boundary。Cube 有 8 corners (3 axes: text payload × system prompt × image)；paper 测 **4 phantom corners (cube image-off 半) + 1 image-on cube endpoint + 1 image-only mode (Vision, cube 之外)** = **5 cube modes + 1 image-only mode = 6 paper modes**:
    86	
    87	| # | text | prompt | image | mode | 在 phantom space? |
    88	|---|---|---|---|---|---|
    89	| 1 | AXTree | DOM-prompt | No | **DOM** | ✅ phantom corner (origin baseline) |
    90	| 2 | AXTree | DOM-prompt | Yes | "DOM+image" | ❌ violates boundary (image embedding tax) |
    91	| 3 | AXTree | SoM-prompt | No | **P-prompt** | ✅ phantom corner (axis 2 alone @ AXTree 锚点) |
    92	| 4 | AXTree | SoM-prompt | Yes | (mismatched + image) | ❌ violates boundary |
    93	| 5 | [SOM_MARKS] | DOM-prompt | No | **P-text** | ✅ phantom corner (axis 1 alone @ DOM-prompt 锚点) |
    94	| 6 | [SOM_MARKS] | DOM-prompt | Yes | "P-text+image" | ❌ violates boundary |
    95	| 7 | [SOM_MARKS] | SoM-prompt | No | **P-SoM** | ✅ phantom corner (cube center, axis 1+2 compound) |
    96	| 8 | [SOM_MARKS] | SoM-prompt | Yes | **SoM** | image-on cube endpoint (paper baseline, NOT phantom) |
    97	| — | none | — | Yes | **Vision** | image-only mode (cube 之外, axis 1 = "no text") |
    98	
    99	**Boundary 定义性属性 (axis 3 = "no annotated image")**:
   100	- 4 phantom corners 都 share 4-fold drop-in by construction:
   101	  - (a) cost ≈ DOM — no image embedding tax (`[SOM_MARKS]` 是 AXTree regex filter，~3K text both)
   102	  - (b) latency ~50% lower — no image inference stage
   103	  - (c) signal AUROC ≥ baseline — emergent (5/5 phantom `overall_usable=True`，red P-text 0.793 = 5-mode max)
   104	  - (d) drop-one oracle 1.7-3.8pp positive per arm — emergent (B0 red: P-text +3.81 / P-SoM +3.33 / P-prompt +2.86，all sig CI excludes 0)
   105	- (a)(b) 由 boundary derive (definitional)；(c)(d) 是经验验证的 emergent property —— **definition-then-validation 双层结构**
   106	
   107	**Why exclude #2/#4/#6 (3 image-on phantom corners)**: 一旦加 annotated image 回去，cost / latency / carbon 都跟 SoM 拉齐 → 失去 4-fold drop-in property → 不再属于 phantom space。这 3 个 corners 在 routing 维度上是 SoM 的 variants (image cost dominate)，不提供 phantom-class deployment value。即 **boundary 是 "no annotated image"，不是 "matched parsing"**。
   108	

exec
/bin/bash -lc "nl -ba scripts/analysis/aggregate_phantom_lift.py | sed -n '431,463p;496,535p;565,589p;617,633p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   431	    sr_3_psom_only = 100 * len(union_3_psom_only) / max(1, len(u_psom))
   432	    sr_4_psom = 100 * len(union_4_psom) / max(1, len(u_psom))
   433	    universe_psom = sorted(u_psom)
   434	    in_3_psom = np.array([t in union_3_psom_only for t in universe_psom], dtype=bool)
   435	    in_4_psom = np.array([t in union_4_psom for t in universe_psom], dtype=bool)
   436	
   437	    # CSV-reported sr_3 / union_3 use universe_5 (paper-grade primary
   438	    # denominator when P-text present; same as universe_psom otherwise).
   439	    succ_r = {m: s & common for m, s in succ.items()}
   440	    union_3 = succ_r["DOM"] | succ_r["SoM"] | succ_r["Vision"]
   441	    sr_3 = 100 * len(union_3) / n
   442	    universe = sorted(common)
   443	    # Backward-compat aliases for downstream 5-mode / H3 axis tests
   444	    # which use in_3 indexed against universe_5.
   445	    in_3 = np.array([t in union_3 for t in universe], dtype=bool)
   446	
   447	    # Single-P-SoM lift CI (uses P-SoM-specific universe per F07)
   448	    ci_lo_psom, ci_hi_psom = bootstrap_lift_ci(in_3_psom, in_4_psom)
   449	    h_4psom_vs_3 = cohen_h(sr_4_psom / 100, sr_3_psom_only / 100)
   450	    wstat_psom, wp_psom = wilcoxon_signed_rank(in_3_psom, in_4_psom)
   451	    mc_p_psom = mcnemar_exact_one_sided(in_3_psom, in_4_psom)
   452	    tost_p_psom = bootstrap_tost_p(in_3_psom, in_4_psom)
   453	
   454	    psom_adds = succ_r["P-SoM"] - union_3
   455	
   456	    if has_pdom:
   457	        union_4_pdom = union_3 | succ_r["P-text"]
   458	        union_5 = union_3 | succ_r["P-text"] | succ_r["P-SoM"]
   459	        sr_4_pdom = 100 * len(union_4_pdom) / n
   460	        sr_5 = 100 * len(union_5) / n
   461	        in_4_pdom = np.array([t in union_4_pdom for t in universe], dtype=bool)
   462	        in_5 = np.array([t in union_5 for t in universe], dtype=bool)
   463	        ci_lo, ci_hi = bootstrap_lift_ci(in_3, in_5)
   496	    # P-prompt 4-mode lift + 6-mode oracle (when present)
   497	    if has_pprompt:
   498	        # F07 audit fix 2026-05-09: P-prompt-only comparison uses
   499	        # universe_pprompt_only (DOM ∩ SoM ∩ Vision ∩ P-prompt), NOT the
   500	        # 5-mode universe — otherwise the denominator drops by tasks
   501	        # missing in P-text/P-SoM that have nothing to do with this arm.
   502	        u_pprompt, succ_r_pprompt = _restrict_set(["DOM", "SoM", "Vision", "P-prompt"])
   503	        union_3_pprompt_only = succ_r_pprompt["DOM"] | succ_r_pprompt["SoM"] | succ_r_pprompt["Vision"]
   504	        union_4_pprompt = union_3_pprompt_only | succ_r_pprompt["P-prompt"]
   505	        sr_3_pprompt_only = 100 * len(union_3_pprompt_only) / max(1, len(u_pprompt))
   506	        sr_4_pprompt = 100 * len(union_4_pprompt) / max(1, len(u_pprompt))
   507	        u_pprompt_sorted = sorted(u_pprompt)
   508	        in_3_pprompt = np.array([t in union_3_pprompt_only for t in u_pprompt_sorted], dtype=bool)
   509	        in_4_pprompt = np.array([t in union_4_pprompt for t in u_pprompt_sorted], dtype=bool)
   510	        ci_lo_pprompt, ci_hi_pprompt = bootstrap_lift_ci(in_3_pprompt, in_4_pprompt)
   511	        h_4pprompt_vs_3 = cohen_h(sr_4_pprompt / 100, sr_3_pprompt_only / 100)
   512	        wstat_pprompt, wp_pprompt = wilcoxon_signed_rank(in_3_pprompt, in_4_pprompt)
   513	        mc_p_pprompt = mcnemar_exact_one_sided(in_3_pprompt, in_4_pprompt)
   514	        tost_p_pprompt = bootstrap_tost_p(in_3_pprompt, in_4_pprompt)
   515	        pprompt_adds = succ_r["P-prompt"] - union_3
   516	        if has_pdom:
   517	            # F07 audit fix 2026-05-09: 6-mode oracle and 6-vs-5
   518	            # incremental tests must use universe_6 (DOM ∩ SoM ∩
   519	            # Vision ∩ P-text ∩ P-SoM ∩ P-prompt). Previously used
   520	            # universe_5 which can include tasks where P-prompt was
   521	            # not observed → treats missing as failed.
   522	            u6_sorted = sorted(universe_6)
   523	            succ_r_u6 = {m: s & universe_6 for m, s in succ.items()}
   524	            union_3_u6 = succ_r_u6["DOM"] | succ_r_u6["SoM"] | succ_r_u6["Vision"]
   525	            union_5_u6 = union_3_u6 | succ_r_u6["P-text"] | succ_r_u6["P-SoM"]
   526	            union_6 = union_5_u6 | succ_r_u6["P-prompt"]
   527	            sr_3_u6 = 100 * len(union_3_u6) / max(1, len(universe_6))
   528	            sr_5_u6 = 100 * len(union_5_u6) / max(1, len(universe_6))
   529	            sr_6 = 100 * len(union_6) / max(1, len(universe_6))
   530	            in_3_u6 = np.array([t in union_3_u6 for t in u6_sorted], dtype=bool)
   531	            in_5_u6 = np.array([t in union_5_u6 for t in u6_sorted], dtype=bool)
   532	            in_6 = np.array([t in union_6 for t in u6_sorted], dtype=bool)
   533	            ci_lo_6, ci_hi_6 = bootstrap_lift_ci(in_3_u6, in_6)
   534	            ci_lo_6v5, ci_hi_6v5 = bootstrap_lift_ci(in_5_u6, in_6)
   535	            h_6_vs_3 = cohen_h(sr_6 / 100, sr_3_u6 / 100)
   565	    # H3 structural test: phantom space 2-axis empirical validation.
   566	    # For each axis, bootstrap CI on |arm ∖ P-SoM| unique-count + McNemar exact
   567	    # one-sided. CI lower bound > 0 evidences axis contributes tasks P-SoM
   568	    # doesn't solve (i.e., axis is empirically distinct from compound center,
   569	    # phantom space is multi-region not collapsed point).
   570	    in_psom_raw = np.array([t in succ_r["P-SoM"] for t in universe], dtype=bool)
   571	
   572	    if has_pdom:
   573	        in_pdom_raw = np.array([t in succ_r["P-text"] for t in universe], dtype=bool)
   574	        h3_axis1_count, h3_axis1_ci_lo, h3_axis1_ci_hi = bootstrap_unique_count_ci(
   575	            in_pdom_raw, in_psom_raw)
   576	        # mcnemar_exact_one_sided(a, b) tests H1: b > a (b adds tasks a misses)
   577	        # Set a=P-SoM, b=P-text → H1 asymmetric: P-text adds tasks P-SoM misses
   578	        # more often than vice versa (directional structural asymmetry test).
   579	        h3_axis1_mcnemar_p = mcnemar_exact_one_sided(in_psom_raw, in_pdom_raw)
   580	    else:
   581	        h3_axis1_count = h3_axis1_ci_lo = h3_axis1_ci_hi = h3_axis1_mcnemar_p = None
   582	
   583	    if has_pprompt:
   584	        in_pprompt_raw = np.array([t in succ_r["P-prompt"] for t in universe], dtype=bool)
   585	        h3_axis2_count, h3_axis2_ci_lo, h3_axis2_ci_hi = bootstrap_unique_count_ci(
   586	            in_pprompt_raw, in_psom_raw)
   587	        h3_axis2_mcnemar_p = mcnemar_exact_one_sided(in_psom_raw, in_pprompt_raw)
   588	    else:
   589	        h3_axis2_count = h3_axis2_ci_lo = h3_axis2_ci_hi = h3_axis2_mcnemar_p = None
   617	        "lift_5_vs_3_pp":   (round(sr_5 - sr_3, 4) if sr_5 is not None else None),
   618	        "lift_5_vs_3_ci95_lo_pp":  maybe_round(ci_lo),
   619	        "lift_5_vs_3_ci95_hi_pp":  maybe_round(ci_hi),
   620	        "lift_4pdom_vs_3_pp":   (round(sr_4_pdom - sr_3, 4) if sr_4_pdom is not None else None),
   621	        "lift_4pdom_vs_3_ci95_lo_pp": maybe_round(ci_lo_pdom),
   622	        "lift_4pdom_vs_3_ci95_hi_pp": maybe_round(ci_hi_pdom),
   623	        "lift_4psom_vs_3_pp":   round(sr_4_psom - sr_3, 4),
   624	        "lift_4psom_vs_3_ci95_lo_pp": round(ci_lo_psom, 4),
   625	        "lift_4psom_vs_3_ci95_hi_pp": round(ci_hi_psom, 4),
   626	        "lift_4pprompt_vs_3_pp": (round(sr_4_pprompt - sr_3, 4) if sr_4_pprompt is not None else None),
   627	        "lift_4pprompt_vs_3_ci95_lo_pp": maybe_round(ci_lo_pprompt),
   628	        "lift_4pprompt_vs_3_ci95_hi_pp": maybe_round(ci_hi_pprompt),
   629	        "lift_6_vs_3_pp": (round(sr_6 - sr_3, 4) if sr_6 is not None else None),
   630	        "lift_6_vs_3_ci95_lo_pp": maybe_round(ci_lo_6),
   631	        "lift_6_vs_3_ci95_hi_pp": maybe_round(ci_hi_6),
   632	        "lift_6_vs_5_pp": (round(sr_6 - sr_5, 4) if (sr_6 is not None and sr_5 is not None) else None),
   633	        "lift_6_vs_5_ci95_lo_pp": maybe_round(ci_lo_6v5),

codex
Statistical framework has 9 issues that would be flagged by a top stat reviewer.

**Confirmed Math Errors**
1. `scripts/analysis/aggregate_phantom_lift.py:623,626,629,632` mixes denominators for reported point lifts.
Correct formula: `lift = sr_augmented(U_same) - sr_baseline(U_same)`.
Implemented: e.g. `sr_4_psom` is computed on `u_psom` at lines 431-432, but subtracts `sr_3` computed on `universe_5` at lines 437-441. Same pattern for P-prompt and 6-mode lifts.
Magnitude: 0 if all mode observation sets are identical; otherwise can shift point estimates by roughly the missing-task imbalance over N, enough to desynchronize point estimates from their CIs.
Severity: HIGH for archived/aggregate path; defuse effort LOW-MED.

2. `scripts/analysis/aggregate_phantom_lift.py:583-587` tests H3 axis-2 on `universe_5`, not the P-prompt/P-SoM common universe.
Correct formula: compare `P-prompt \ P-SoM` only on tasks observed by both P-prompt and P-SoM, ideally the full H3 axis-2 paired universe.
Implemented: `in_pprompt_raw` is indexed over `universe` from `common = universe_5`, so tasks not observed for P-prompt can be treated as P-prompt failures.
Magnitude: biases axis-2 unique counts downward when P-prompt is partial; could flip H3(ii) to fail.
Severity: HIGH if partial cells exist; defuse effort MED.

I did not find a DL algebra transcription error in `preregistration_decision_test.py:150-209`; the DL τ², Q, I², RE weights, and normal CI are formula-consistent.

**Confirmed Methodological Assumptions Violated**
1. DL random-effects meta with `N=4` cells is fragile. The prereg explicitly gates H1/H3 on DL meta over 4 cells at `docs/checkpoints/pre_run/preregistration.md:54-57,76-77,226`, but DL τ² and I² are unstable with k=4 and normal Wald RE CIs are commonly anti-conservative. A reviewer will ask for REML/Hartung-Knapp or a design-level justification. Severity HIGH.

2. The power appendix does not support the current design. It still analyzes `K_h1=12/16` and `K_h3=11/16` at `docs/analysis/cross_sites/power_analysis.md:36-49`, says DL meta is on cells with `N≥10` at line 55, and discusses pooled `N=910` TOST across cls+red+shop at lines 56 and 62. Current Phase 1a is 4 cells and no shop. Severity HIGH for prereg coherence.

3. Heterogeneity override is preregistered but not implemented. The prereg says I² > 75% means “do NOT pool” and maps to per-cell direction consistency at `docs/checkpoints/pre_run/preregistration.md:152,231`. The canonical script still gates on pooled meta at `scripts/analysis/preregistration_decision_test.py:361-365` and `:443-448`, and `apply_framing_rule()` at `:522-542` has no heterogeneity branch. Severity HIGH.

4. H2 is not the preregistered 4-fold drop-in gate in the canonical script. The prereg requires cost, latency, signal AUROC, and folded drop-one at `docs/checkpoints/pre_run/preregistration.md:61-68` and R1 requires H2(a)(b)(c) at line 144. Code only checks median cost point estimates at `scripts/analysis/preregistration_decision_test.py:477-515`, then maps that single boolean to H2 at `:524-525`. Severity HIGH.

5. H2 “equivalence” is a point-estimate band, not an equivalence test. Cost passes if median ratio is within ±10% at `scripts/analysis/preregistration_decision_test.py:493-503`; no uncertainty, bootstrap CI, or TOST-style decision is used. A top reviewer will not accept “median inside band” as statistical equivalence. Severity MED-HIGH.

6. H3 pools raw unique counts, not rates. The prereg gates on mean `|P-text \ P-SoM|` and `|P-prompt \ P-SoM|` counts at `docs/checkpoints/pre_run/preregistration.md:76-78`; code pools counts directly at `scripts/analysis/preregistration_decision_test.py:416-444`. This is tolerable only because cls/red N are close; it becomes indefensible when shop N=466 enters Phase 1b. Severity MED.

7. H3 axis multiplicity is under-declared. Axis 1 and axis 2 are each `m=1` at `docs/checkpoints/pre_run/preregistration.md:164-168`. For R2, “only one axis holds” creates a two-test opportunity for a single-axis claim without correcting across axes. Severity MED.

8. Missing-data edge cases are undefined in the canonical gate. The script only requires `len(cells_by_id) >= 2` at `scripts/analysis/preregistration_decision_test.py:665-667`; it does not enforce exactly the four preregistered cells, all six mode columns, or nonzero tasks per cell. `_paired_bootstrap()` will fail on zero-task cells at `:134-138`. Severity MED.

9. Prereg prose still contradicts the 2026-05-13 superiority-test revision. H1 says superiority at `docs/checkpoints/pre_run/preregistration.md:55`, but R5 and family declaration still mention TOST rejection at `:148,160,232`; code treats TOST as informational at `scripts/analysis/preregistration_decision_test.py:357-359`. Severity MED.

**Reviewer Ammunition**
- “Why DerSimonian-Laird with only four cells, and why not REML plus Hartung-Knapp?” Current answer: no adequate answer prepared; prereg says DL is locked.
- “Why does I² > 75% cancel pooling in prose but not in the canonical implementation?” Current answer: no answer prepared.
- “Why is H2 called equivalence when you only check median point estimates?” Current answer: no answer prepared.
- “Why are H3 axes separate m=1 families when R2 lets either axis support a structural hook?” Current answer: structural family is claimed weaker/separate, but this does not address familywise error for R2.
- “Why raw unique counts instead of per-task rates or paired binomial/logistic models?” Current answer: noise floor is task-count based; weak for unequal N.
- “How does the power appendix justify N=4 when it still discusses 16 cells, N≥10, and shop?” Current answer: stale appendix, not current support.
- “How are missing cells or partial mode runs handled?” Current answer: prereg says all 6 modes per cell; canonical code does not enforce it.

**Framing Rule Edge Cases**
- H1/H3 pass with I² > 75%: prereg says do not pool and downgrade to per-cell consistency; code can still emit R1/R2/R3 from pooled gates.
- Cost passes but latency or AUROC fails: code can emit R1 because H2 means cost-only.
- Only one H3 axis passes at raw α=0.05: R2 can fire without two-axis family correction.
- Three cells loaded instead of four: script still runs and may frame claims outside the preregistered N=4 design.
- All four cells have one unique H3 task each: pooled count may look positive, but per-cell noise floor says one task is noise; primary handling is ambiguous.

**Verdict on Next Steps**
Before advisor meeting: fix prereg prose contradictions around superiority vs TOST, update the power appendix to the 4-cell design, and explicitly state whether H2 is cost-only or the full 4-fold gate.

Advisor-sync required: decide whether DL with k=4 is acceptable as the primary gate, or whether the gate needs REML/Hartung-Knapp, permutation/randomization, or a fixed-effect/stratified paired model sensitivity.

Appendix sensitivity is sufficient for: DL vs REML/HK, H3 count vs rate, one-sided vs two-sided H1 meta p, and leave-one-cell-out influence. The heterogeneity branch and H2 gate mismatch are not appendix-only issues; they affect the primary framing rule.
tokens used
135,659
Statistical framework has 9 issues that would be flagged by a top stat reviewer.

**Confirmed Math Errors**
1. `scripts/analysis/aggregate_phantom_lift.py:623,626,629,632` mixes denominators for reported point lifts.
Correct formula: `lift = sr_augmented(U_same) - sr_baseline(U_same)`.
Implemented: e.g. `sr_4_psom` is computed on `u_psom` at lines 431-432, but subtracts `sr_3` computed on `universe_5` at lines 437-441. Same pattern for P-prompt and 6-mode lifts.
Magnitude: 0 if all mode observation sets are identical; otherwise can shift point estimates by roughly the missing-task imbalance over N, enough to desynchronize point estimates from their CIs.
Severity: HIGH for archived/aggregate path; defuse effort LOW-MED.

2. `scripts/analysis/aggregate_phantom_lift.py:583-587` tests H3 axis-2 on `universe_5`, not the P-prompt/P-SoM common universe.
Correct formula: compare `P-prompt \ P-SoM` only on tasks observed by both P-prompt and P-SoM, ideally the full H3 axis-2 paired universe.
Implemented: `in_pprompt_raw` is indexed over `universe` from `common = universe_5`, so tasks not observed for P-prompt can be treated as P-prompt failures.
Magnitude: biases axis-2 unique counts downward when P-prompt is partial; could flip H3(ii) to fail.
Severity: HIGH if partial cells exist; defuse effort MED.

I did not find a DL algebra transcription error in `preregistration_decision_test.py:150-209`; the DL τ², Q, I², RE weights, and normal CI are formula-consistent.

**Confirmed Methodological Assumptions Violated**
1. DL random-effects meta with `N=4` cells is fragile. The prereg explicitly gates H1/H3 on DL meta over 4 cells at `docs/checkpoints/pre_run/preregistration.md:54-57,76-77,226`, but DL τ² and I² are unstable with k=4 and normal Wald RE CIs are commonly anti-conservative. A reviewer will ask for REML/Hartung-Knapp or a design-level justification. Severity HIGH.

2. The power appendix does not support the current design. It still analyzes `K_h1=12/16` and `K_h3=11/16` at `docs/analysis/cross_sites/power_analysis.md:36-49`, says DL meta is on cells with `N≥10` at line 55, and discusses pooled `N=910` TOST across cls+red+shop at lines 56 and 62. Current Phase 1a is 4 cells and no shop. Severity HIGH for prereg coherence.

3. Heterogeneity override is preregistered but not implemented. The prereg says I² > 75% means “do NOT pool” and maps to per-cell direction consistency at `docs/checkpoints/pre_run/preregistration.md:152,231`. The canonical script still gates on pooled meta at `scripts/analysis/preregistration_decision_test.py:361-365` and `:443-448`, and `apply_framing_rule()` at `:522-542` has no heterogeneity branch. Severity HIGH.

4. H2 is not the preregistered 4-fold drop-in gate in the canonical script. The prereg requires cost, latency, signal AUROC, and folded drop-one at `docs/checkpoints/pre_run/preregistration.md:61-68` and R1 requires H2(a)(b)(c) at line 144. Code only checks median cost point estimates at `scripts/analysis/preregistration_decision_test.py:477-515`, then maps that single boolean to H2 at `:524-525`. Severity HIGH.

5. H2 “equivalence” is a point-estimate band, not an equivalence test. Cost passes if median ratio is within ±10% at `scripts/analysis/preregistration_decision_test.py:493-503`; no uncertainty, bootstrap CI, or TOST-style decision is used. A top reviewer will not accept “median inside band” as statistical equivalence. Severity MED-HIGH.

6. H3 pools raw unique counts, not rates. The prereg gates on mean `|P-text \ P-SoM|` and `|P-prompt \ P-SoM|` counts at `docs/checkpoints/pre_run/preregistration.md:76-78`; code pools counts directly at `scripts/analysis/preregistration_decision_test.py:416-444`. This is tolerable only because cls/red N are close; it becomes indefensible when shop N=466 enters Phase 1b. Severity MED.

7. H3 axis multiplicity is under-declared. Axis 1 and axis 2 are each `m=1` at `docs/checkpoints/pre_run/preregistration.md:164-168`. For R2, “only one axis holds” creates a two-test opportunity for a single-axis claim without correcting across axes. Severity MED.

8. Missing-data edge cases are undefined in the canonical gate. The script only requires `len(cells_by_id) >= 2` at `scripts/analysis/preregistration_decision_test.py:665-667`; it does not enforce exactly the four preregistered cells, all six mode columns, or nonzero tasks per cell. `_paired_bootstrap()` will fail on zero-task cells at `:134-138`. Severity MED.

9. Prereg prose still contradicts the 2026-05-13 superiority-test revision. H1 says superiority at `docs/checkpoints/pre_run/preregistration.md:55`, but R5 and family declaration still mention TOST rejection at `:148,160,232`; code treats TOST as informational at `scripts/analysis/preregistration_decision_test.py:357-359`. Severity MED.

**Reviewer Ammunition**
- “Why DerSimonian-Laird with only four cells, and why not REML plus Hartung-Knapp?” Current answer: no adequate answer prepared; prereg says DL is locked.
- “Why does I² > 75% cancel pooling in prose but not in the canonical implementation?” Current answer: no answer prepared.
- “Why is H2 called equivalence when you only check median point estimates?” Current answer: no answer prepared.
- “Why are H3 axes separate m=1 families when R2 lets either axis support a structural hook?” Current answer: structural family is claimed weaker/separate, but this does not address familywise error for R2.
- “Why raw unique counts instead of per-task rates or paired binomial/logistic models?” Current answer: noise floor is task-count based; weak for unequal N.
- “How does the power appendix justify N=4 when it still discusses 16 cells, N≥10, and shop?” Current answer: stale appendix, not current support.
- “How are missing cells or partial mode runs handled?” Current answer: prereg says all 6 modes per cell; canonical code does not enforce it.

**Framing Rule Edge Cases**
- H1/H3 pass with I² > 75%: prereg says do not pool and downgrade to per-cell consistency; code can still emit R1/R2/R3 from pooled gates.
- Cost passes but latency or AUROC fails: code can emit R1 because H2 means cost-only.
- Only one H3 axis passes at raw α=0.05: R2 can fire without two-axis family correction.
- Three cells loaded instead of four: script still runs and may frame claims outside the preregistered N=4 design.
- All four cells have one unique H3 task each: pooled count may look positive, but per-cell noise floor says one task is noise; primary handling is ambiguous.

**Verdict on Next Steps**
Before advisor meeting: fix prereg prose contradictions around superiority vs TOST, update the power appendix to the 4-cell design, and explicitly state whether H2 is cost-only or the full 4-fold gate.

Advisor-sync required: decide whether DL with k=4 is acceptable as the primary gate, or whether the gate needs REML/Hartung-Knapp, permutation/randomization, or a fixed-effect/stratified paired model sensitivity.

Appendix sensitivity is sufficient for: DL vs REML/HK, H3 count vs rate, one-sided vs two-sided H1 meta p, and leave-one-cell-out influence. The heterogeneity branch and H2 gate mismatch are not appendix-only issues; they affect the primary framing rule.
