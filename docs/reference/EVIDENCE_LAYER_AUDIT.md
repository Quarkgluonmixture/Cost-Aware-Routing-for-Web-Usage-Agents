---
type: reference
status: rolling
created: 2026-05-03
updated: 2026-05-20
purpose: pre-rerun methodology + visualization gap registry; lock analysis design before 6-cell data lands
audience: self + advisor
---

# Evidence-Layer Audit & Action Queue

> **Why this doc exists**: 6-cell paper-grade rerun lands ~1 week wallclock. 数据进来前必须 lock 死分析方案 (multiple-comparison family / effect-size convention / pre-registered hypothesis / pooled estimate strategy)，否则 reviewer 一句 "post-hoc analysis decisions" 折损 paper rigor。
>
> **What this doc is NOT**: 不是 finding registry / 不是 paper hook / 不是 result narrative。**纯方法论 + viz 装备清单**, 跟数据 direction agnostic。
>
> **How to use**: Tier T0 = rerun launch 前必须 ready；T1 = advisor sync + paper §5/§7 commit 前 ready；T2 = paper end-stage prose 前 ready. 顺序 strict — T0 先全 done, 再动 T1, 再 T2.

---

## §0 Audit scope (where evidence lives now)

**Stats artifacts** (`results/phantom_paper/`):
- `phantom_lift.{md,csv}` — Outcome 0c/0d (3→5/6-mode oracle lift, drop-one, Jaccard)
- `auroc_cross_condition.{md,csv,_summary.md}` — Macro routing-signal AUROC (5 mode × 3 cell × top-3 signal)
- `fig0c_drop_one_bootstrap_ci.csv` — drop-one CI registry
- `cost_per_mode.{md,json}` — Efficiency cost
- `run_summary_collect.json` — collected episode summary (~509KB)
- `run_manifest.yaml` — cell config registry

**Aggregator scripts** (`scripts/analysis/`):
- `aggregate_phantom_lift.py` (572 lines) — phantom_lift + Jaccard, has Wilcoxon/McNemar/bootstrap CI infra
- `aggregate_routing_auroc.py` (163 lines) — AUROC top-3 per (cell × mode)
- `aggregate_cross_site.py` (419 lines) — cross-site narrative findings
- `aggregate_cost_electricity.py` — cost + electricity
- `aggregate_sr_fp_per_mode.py` — FP-adjusted SR

**Cross-site / mechanism**: `docs/analysis/cross_sites/` (`axis1_microbehavior` / `axis_effect_size` / `mechanism_per_task`)

**Figures** (`results/phantom_paper/figures/`, 13 active):
- fig0c (drop-one + phantom_lift_bars), fig0d (jaccard), fig0e (cat × mode), fig0f (overlap), fig0g (AUROC heatmap)
- fig1ab (cascade), fig1c (strategy gradient)
- fig2 (micro_divergence) + fig2b/2c/2d/2e/2f
- fig3a (token_cost), fig3c (latency), fig3d (Pareto), fig3 (regional_carbon)
- fig_capability_b0_b1

---

## §1 Methodology + visualization gap registry

按 4 evidence type × 4 cross-X axis 组织 (paper §2 Zoom 1-4 evidence layer skeleton). 每行: stats gap + paired viz gap + tier + ETA.

### A. Cross-cutting methodology (优先级最高)

| # | Stats gap | Paired viz gap | Tier | ETA |
|---|---|---|---|---|
| **A1** | **Multiple-comparison correction columns** in `phantom_lift.md`: Bonferroni p / Holm-Bonferroni p / BH FDR q. Comparison family explicit. | **Forest plot per phantom arm** showing raw 95% CI vs Holm-adjusted CI (`fig_forest_drop_one.py`) | **T0** | 3h stats + 1h viz |
| **A2** | **Pre-registration doc** `docs/checkpoints/pre_run/preregistration.md` — primary H1 / secondary H2-Hn / disconfirmation conditions / multiple-comparison family / decision rule. Git commit timestamp = registration time. | **Hypothesis × outcome confirmation matrix** (filled post-rerun): hypothesis row × cell column × {pass/fail/inconclusive} cell coloring (`fig_hypothesis_matrix.py`) | **T0** | 2h doc (need user-decided H list) + 1h viz scaffold |
| **A3** | **Cross-cell meta-analysis**: random-effect (DerSimonian-Laird) pooled drop-one per arm + I² heterogeneity statistic. New `aggregate_phantom_meta.py` + `meta_phantom_lift.md`. | **Forest plot pooled estimate** (`fig_meta_forest.py`) — per arm: cells listed vertically + pooled diamond at bottom + I² annotation | **T0** | 4h stats + 2h viz |
| **A4** | **TOST equivalence test** for "effect ≈ 0" reverse-claim ability — per-arm vs equivalence margin δ=0.5pp, two one-sided tests. | **Equivalence bound viz** (`fig_tost_bounds.py`) — CI bar with ±δ shaded region overlay, per arm | T1 | 2h stats + 1h viz |
| **A5** | **Effect-size standardization 跨连续 outcome**: Cohen's d (cost / latency 连续) + Cliff's δ (AUROC non-parametric) + CI for Cohen's h (currently point estimate only) | **Effect-size CI panel** (`fig_effect_size_panel.py`) — h / d / δ as 3-panel forest, per arm | T1 | 2h stats + 1h viz |
| **A6** | **Negative result registry** in `paper_planning §3` — pre-list "hypothesized but failed" candidate explanations | (no viz) | T1 | 1h manual |

### B. Outcome (SR) — cross-mode primary

| # | Stats gap | Paired viz gap | Tier | ETA |
|---|---|---|---|---|
| B1 | **Friedman test for within-cell 5-mode global ranking** (current pairwise 9 tests) + Kendall's W rank concordance across cells | **Mode-rank line plot per task** (`fig_friedman_ranks.py`) — N=234 tasks × 5 modes, per-task rank trajectory | T1 | 2h stats + 2h viz |
| B2 | **CI for Cohen's h** (bootstrap), currently point estimate | (folded into A5 panel) | T1 | 1h stats |
| B3 | **Per-task SR variance decomposition** — heterogeneity 来源 quantify | **Per-task SR violin** (`fig_per_task_violin.py`) — per cell × mode, distribution of within-task SR resamples | T2 | 2h stats + 2h viz |
| B4 | **FP-filter sensitivity master table** — raw / +na_fp / +eval_fp / +visual_fp 4 variant 下 ranking + sig stable check (现散在 `docs/analysis/`) | **FP-filter tornado plot** (`fig_fp_sensitivity_tornado.py`) — per (cell × arm), 4 variant ranking diff | T1 | 2h stats + 1h viz |
| B5 | **Achievable-vs-ceiling SR gap** — oracle ceiling vs single-best-mode achievable | (folded into existing fig0c phantom_lift_bars with new layer) | T2 | 1h stats |

### C. Macro (action freq, episode length, finish rate) — cross-mode

| # | Stats gap | Paired viz gap | Tier | ETA |
|---|---|---|---|---|
| C1 | **Per-action-type breakdown CSV** (click / type / scroll / goto) — currently 散在 figs, 无 master | **Per-action stacked bar cross mode** (`fig_action_stacked.py`) — per cell, 5 mode × 5 action stack | T2 | 2h stats + 1h viz |
| C2 | **Censored-data Kaplan-Meier** for episode length (early-stop = right-censored event) | **Episode-length survival curves** (`fig_km_episode_length.py`) — per cell, 5 mode survival curves with CI bands | T2 | 3h stats + 1h viz (待 early-stop A/B/C decision) |
| C3 | **Cross-cell pooled AUROC** + I² heterogeneity (类似 A3 但 for AUROC) | (folded into fig0g heatmap with pooled column) | T1 | 2h stats |
| C4 | **DeLong test on AUROC paired difference** — phantom AUROC ≥ DOM AUROC paired sig (currently visual inspect) | **AUROC delta forest** (`fig_auroc_delta_forest.py`) — per (cell × signal), phantom_AUROC - DOM_AUROC with CI + sig stars | T1 | 3h stats + 1h viz |

### D. Micro (per-step) — cross-mode

| # | Stats gap | Paired viz gap | Tier | ETA |
|---|---|---|---|---|
| D1 | **Step-aligned trajectory comparison** (DTW or edit distance) cross mode | **DTW alignment heatmap** (`fig_dtw_alignment.py`) — per (cell × mode-pair), task × step distance matrix | T2 | 5h stats + 2h viz |
| D2 | **Recovery probability after first divergence** rate cross mode | **Recovery prob heatmap** (`fig_recovery_prob.py`) — per cell, mode × first-divergence-step-bin | T2 | 3h stats + 2h viz |
| D3 | **`mechanism_per_task` cross-cell aggregation** — 现 per-cell 散开 | **Mechanism per-task cross-cell heatmap** (`fig_mechanism_per_task_cross.py`) | T2 | 2h stats + 2h viz |
| D4 | **Q-test for axis-effect heterogeneity** across task categories | (folded into existing fig0e with sig overlay) | T2 | 2h stats |

### E. Efficiency (cost / latency / carbon) — cross-mode

| # | Stats gap | Paired viz gap | Tier | ETA |
|---|---|---|---|---|
| E1 | **Cost median + bootstrap CI** in `cost_per_mode.md` (currently point estimate only) | **Cost errorbars in fig3a** — currently point, add CI whiskers | T1 | 2h stats + 0.5h viz |
| E2 | **Latency-vs-cost ratio per arm with CI** | (folded into fig3c with CI) | T2 | 1h stats |
| E3 | **Cost-effectiveness ratio (lift / cost) with CI** — paper §6 routing decision 用 | **CE ratio forest** (`fig_ce_ratio_forest.py`) — per arm, lift÷cost with CI | T1 | 2h stats + 1h viz |
| E4 | **Pareto frontier with confidence band** — currently fig3d 是 point-only Pareto | **CI-shaded Pareto** (`fig3d` augment) — bootstrap envelope around frontier | T2 | 3h stats + 1h viz |
| E5 | **Multi-metric Pareto (cost + latency + carbon)** — already flagged in next_steps §5 | **Trellis or 3D Pareto** (`fig_multi_pareto.py`) | T2 | 2h stats + 2h viz |

### F. Cross-X axis (interaction tests)

| # | Stats gap | Paired viz gap | Tier | ETA |
|---|---|---|---|---|
| F1 | **Mode × site interaction test** (fixed-effect ANOVA / GLM logistic) — 现 cross-site 是 narrative compare | **Site × mode interaction line plot** (`fig_site_mode_interaction.py`) — y=SR, x=mode, lines colored by site, sig star on interaction term | **T1** (advisor sync 用) | 3h stats + 1h viz |
| F2 | **B0 × B1 cross-model interaction test** for "capability-modulated reversal" claim — currently narrative | **B0×B1 interaction crossed line plot** (`fig_capability_interaction.py`) — y=drop-one, x=axis (text/image), lines for B0/B1, crossover visible | **T1** (advisor sync 用) | 3h stats + 1h viz |
| F3 | **Per-task heterogeneity violin per cell** — task-as-random-effect | (folded into B3 violin) | T2 | (folded) |
| F4 | **Cross-task variance (bootstrap by task)** — separate from within-task variance | (folded into A3 forest with cell-level CI from task-bootstrap) | T0 | (folded into A3) |

### G. Reproducibility / determinism reporting

| # | Stats gap | Paired viz gap | Tier | ETA |
|---|---|---|---|---|
| G1 | **B0 5-call probe table 进 main paper supplement** (现仅 `probe_b37_api_determinism.md`) | **B0 5-call action consistency stacked bar** (`fig_b0_determinism.py`) — per task, 5 calls × {byte_diff, action_same, action_diff} | T1 | 0.5h stats (already done) + 1h viz |
| G2 | **B1 determinism spot-check** — N=5 同 task 同 seed byte-exact replication probe | **B1 byte-level vs decision-level panel** (`fig_b1_determinism.py`) | T1 | 1h script + 1h compute + 1h viz |
| G3 | **Per-cell config diff manifest** — Phase A pre/post / SoM-prompt v1/v2 — unified manifest replacing git-log scrape | (no viz, table only in run_manifest) | T1 | 1h manual |

---

## §2 Pre-registration template (T0e, blocks rerun launch)

> **创建** `docs/checkpoints/pre_run/preregistration.md` **with this skeleton**, advisor sync 时 lock + sign. Git commit SHA + timestamp = registration time. OSF DOI (optional, paper submission 前 1 周上传) = third-party witness.
>
> **Epistemic 结构** (核心 design move 2026-05-03 reframe):
> - **Hero claim** (P-SoM as deployment routing arm) — pre-registered strict
> - **4-fold drop-in property** — pre-registered strict (4 sub-claims a/b/c/d)
> - **2-axis structural claim** (phantom space is multi-region, not collapsed point) — pre-registered with low-threshold non-overlap evidence requirement
> - **Framing decision rule** — pre-registered, data-conditional (paper hook 升降级 mapping)
> - **Theory predictions (别扭, capability-reversal)** — marked post-hoc explanatory, no gating

```
---
type: preregistration
status: locked
registered_at: <yyyy-mm-dd HH:MM BST>
registered_git_sha: <40-char>
witnessed_by: <advisor name>
osf_doi: <optional>
data_lock_until: <6-cell rerun completion timestamp>
---

# Phantom-SoM Pre-Registration

## Hypotheses

### PRIMARY (gates paper claim)

H1 (Hero deployment claim — P-SoM is hidden routing arm):
  P-SoM drop-one > 0 across cells, satisfying ALL three sub-conditions:
    (i)   Pooled DerSimonian-Laird random-effect meta sig at Holm α=0.05
          (PRIMARY family m = 1 test, no correction needed within family)
    (ii)  ≥ K_h1 of N_cells individually Holm-sig at α=0.05
          within SECONDARY family m = N_cells (per-cell P-SoM tests)
          where K_h1 = 0.75 (commit-locked, see Commit #1)
    (iii) Pooled magnitude θ_RE ≥ 1.0pp; TOST equivalence rejected
          at margin δ = 1.0pp (commit-locked, see Commit #2)

H2 (4-fold drop-in property — P-SoM specifically):
  All four sub-claims hold per cell, replicated in ≥ K_h1 cells:
    (a) median cost(P-SoM) within ±10% of median cost(DOM)
    (b) median latency(P-SoM) ≤ 0.6 × median latency(SoM)
    (c) top-1 signal AUROC(P-SoM) ≥ AUROC(DOM) − 0.05
    (d) P-SoM drop-one magnitude ≥ 1.0pp (=H1 (iii); folded)

H3 (2-axis empirical structural claim — phantom space is not collapsed point):
  Each phantom-space axis (axis 1 = text-payload via P-text;
  axis 2 = SoM-prompt via P-prompt) contributes tasks NOT solved by P-SoM,
  evidencing axis decomposition is empirically non-trivial:
    (i)   axis 1: P-text ∖ P-SoM unique-task count > 0 with bootstrap
          95% CI excluding 0, in ≥ K_h3 of N_cells
          (lower threshold than H1: structural claim, NOT deployment)
    (ii)  axis 2: P-prompt ∖ P-SoM unique-task count > 0 with bootstrap
          95% CI excluding 0, in ≥ K_h3 of N_cells
          where K_h3 = 0.67 (commit-locked, lower than K_h1 because
          structural is weaker commit than deployment)
    (iii) Per-cell unique-count ≥ 2 tasks (≈1pp at N=234); 1 task is noise
          floor. Tested via exact binomial / paired McNemar one-sided.

### EXPLORATORY (post-data, no pre-commit threshold)

H4 (P-text / P-prompt drop-one magnitude):
  Reported per cell + meta-pooled. No pre-registered ranking commitment.
  Disclosed as exploratory (paper §4 prose explicit "exploratory analysis").

H5 (别扭 framework predictions, 笔记 §108.16):
  4 distinguishing predictions tested against 6-cell data. POST-HOC because
  framework was developed after observing N=4 pre-Phase-A cells.
  Reported irrespective of direction. Paper §5 prose explicit "post-hoc
  theoretical framework, validated on same data motivating it; no formal
  significance gating."

H6 (Capability-modulated reversal):
  B0 vs B1 axis-preference ranking (text-axis drop-one vs image-axis
  drop-one) tested via B0 × B1 × axis GLM logistic interaction. POST-HOC
  exploratory; paper §7 prose explicit "post-hoc finding."

### FRAMING DECISION RULE (pre-registered, data-conditional)

R1 IF (H1 holds AND H2 holds AND H3 (i) AND (ii) hold):
   → Hook framing: "Phantom routing space (M1/M2 2-axis empirical structure);
                    P-SoM as deployment hero, P-text/P-prompt as structural
                    ablation arms validating axis decomposition."
   → Paper §1 hook: STRONGEST.

R2 IF (H1 holds AND H2 holds AND only one of H3 (i)/(ii) holds):
   → Hook framing: "Phantom routing space (single-axis empirical structure)
                    with P-SoM as deployment hero; remaining axis decomp
                    theoretical (Zoom 1 architectural argument only)."
   → Paper §1 hook: MODERATE-STRONG.

R3 IF (H1 holds AND H2 holds AND neither H3 (i)/(ii) holds):
   → Hook framing: "Phantom-SoM is hidden 4th routing arm; M1/M2 axis
                    decomposition supported by Zoom 1 architectural argument
                    only, not empirically validated by ablation."
   → Paper §1 hook: MODERATE (= 04-30 fallback framing).

R4 IF (H1 holds AND H2 partially fails — e.g., (a) cost or (b) latency
       fails on some site):
   → Hook framing: "Phantom-SoM partial drop-in, with site-specific
                    deployment limitations" + §4 disclosure of failed sub-claim.
   → Paper §1 hook: WEAK; substantial revision needed.

R5 IF (H1 fails: pooled meta sig fails Holm OR < K_h1 cells individually sig):
   → Paper death scenario. Reframe:
      Option A: pivot to VWA bug audit paper (§107 4-cluster fix as primary)
      Option B: abandon and merge findings into other paper
   → Decision deferred to advisor sync at fail time.

## Multiple-Comparison Family

  PRIMARY family (gating, m varies):
    H1 (i) pooled meta:    m = 1 (no correction within family)
    H1 (ii) per-cell P-SoM: m = N_cells
    H2 sub-claims (a)(b)(c)(d): m = 4 × N_cells
    Method: Holm-Bonferroni step-down per H-sub-family (Holm 1979).

  STRUCTURAL family (gating, m varies):
    H3 (i) axis 1 per-cell:  m = N_cells
    H3 (ii) axis 2 per-cell: m = N_cells
    Method: Holm-Bonferroni step-down per axis sub-family.
    Rationale: structural claim is weaker than deployment, separate family
    avoids inflating PRIMARY family m count.

  EXPLORATORY family (NOT gating, reported only):
    H4 P-text/P-prompt drop-one: m = 2 × N_cells
    H5/H6 post-hoc: uncorrected, explicitly marked post-hoc.
    BH FDR q-value reported for transparency, not used for paper claim gating.

## Locked Analysis Choices (pre-data)

  Primary metric: oracle ceiling SR pp lift (binary, paired)
  CI method: 1000-resample task-level paired bootstrap
  Sig threshold: Holm α=0.05 within respective family
  Effect size: Cohen's h with bootstrap CI for binary; Cohen's d for continuous
  TOST equivalence margin: δ = 1.0pp (≈ 2 tasks in N=234, bootstrap noise floor)
  H1 K_h1 cell-pass threshold: 0.75 (75% of cells must Holm-sig)
  H3 K_h3 cell-pass threshold: 0.67 (67%, lower because structural < deployment)
  H3 unique-count floor: ≥ 2 tasks per cell
  Cell inclusion: Phase A post-fix only (commit ≥ 3c15cd7) for main analysis;
                  archived pre-fix data → Appendix D robustness check only
  N inclusion: ≥ 100 ep per (cell × mode), else partial-cell intersection
  FP filter primary: na_fp + eval_fp + visual_fp combined
  FP filter sensitivity: 4 variants reported (raw / na_fp / +eval / +visual)

## Exploratory (NOT pre-registered, paper must explicitly flag)

  - Per-task category × mode heatmap exploration (fig0e)
  - Mechanism per-task qualitative analysis (mechanism_per_task)
  - 8-channel image-axis decomposition (axis 3 framework)
  - Any post-hoc cell subsetting beyond H1-H6 family scope
  - 别扭 / capability-reversal explanations (H5/H6) — post-hoc theory, NOT validation

## Witness Mechanism

  (a) Internal: Git SHA + advisor email confirmation (single line:
      "I witness pre-registration of phantom-SoM hypotheses as of
      <SHA> <date>") archived in `.witness/` (gitignored).
  (b) External (optional, paper-time): OSF DOI generated by uploading this
      file to a public OSF project; paper §1 footnote cites the DOI.
```

---

## §3 Action queue (ordered, T0 → T1 → T2)

### T0 — Pre-rerun launch (blocks 6-cell start)

- [x] **T0a — A1 + A4 stats columns** (2026-05-03): 改 `aggregate_phantom_lift.py` 加 Bonferroni / Holm / BH q / TOST col + comparison family declaration block. Output `phantom_lift.md` 包含 augmented PRIMARY table + new SECONDARY-family adjusted per-arm table.
- [x] **T0b — A1 paired viz** (2026-05-03): `scripts/analysis/figures/fig_forest_drop_one.py` 3-panel forest (P-text / P-SoM / P-prompt) × per-cell point + raw 95% CI errorbar + Holm-sig marker fill + TOST equivalence band ±0.5pp. Output `fig_forest_drop_one.png`.
- [x] **T0c — A3 cross-cell meta** (2026-05-03): `scripts/analysis/aggregate_phantom_meta.py` (DerSimonian-Laird random-effect) per arm × all cells. SE_i derived from bootstrap CI. Reports k / FE / RE / 95% CI / Cochran's Q / df / p_Q / τ² / I² + per-family Holm-Bonferroni gating. Output `meta_phantom_lift.{md,csv}`.
- [x] **T0d — A3 paired viz** (2026-05-03): `fig_meta_forest.py` classical forest with weight-sized squares per cell + pooled diamond + I²/Q/τ² annotation + TOST band. Output `fig_meta_forest.png`.
- [ ] **T0e — A2 pre-registration doc**: 写 `docs/checkpoints/pre_run/preregistration.md` (上方 §2 template). 需要 user lock H1-H5 specifics + advisor 见证. _ETA 2h_
- [ ] **T0f — A2 paired viz scaffold**: `fig_hypothesis_matrix.py` 骨架 (post-rerun fill). _ETA 1h_

**T0 total: ~13h work** (≈ 2 个 focused day) — **T0a-T0d done (4/6); T0e blocked on user H-list decision**

### T1 — Advisor sync + paper §5/§7 commit ready

- [ ] T1a — A4 TOST stats + viz (`fig_tost_bounds.py`)
- [ ] T1b — A5 effect-size CI standardization + `fig_effect_size_panel.py`
- [ ] T1c — A6 negative result registry in paper_planning §3
- [ ] T1d — B1 Friedman + Kendall's W
- [ ] T1e — B4 FP-filter sensitivity master table + tornado plot
- [ ] T1f — C3 cross-cell pooled AUROC + I²
- [ ] T1g — C4 DeLong AUROC delta + `fig_auroc_delta_forest.py`
- [ ] **T1h — F1 Site × mode interaction test + viz** ⭐ (advisor sync)
- [ ] **T1i — F2 B0 × B1 interaction test + viz** ⭐ (advisor sync, capability-modulated reversal claim)
- [ ] T1j — E1 cost CI + E3 CE ratio forest
- [ ] T1k — G1 B0 5-call viz + G2 B1 determinism probe + G3 config manifest

**T1 total: ~25h** (~ 3-4 days)

### T2 — Paper end-stage prose

- [ ] T2a — B3 per-task SR violin + B5 achievable-vs-ceiling
- [ ] T2b — C1 per-action stacked bar + C2 KM episode survival (waits early-stop A/B/C)
- [ ] T2c — D1 DTW alignment + D2 recovery prob + D3 mechanism cross-cell + D4 Q-test
- [ ] T2d — E2 latency-vs-cost ratio + E4 Pareto CI + E5 multi-metric Pareto

**T2 total: ~25h**

---

## §4 Tracking

**Status**: T0 in progress (started 2026-05-03)

| Tier | Items done | Items total | % |
|---|---|---|---|
| T0 | 4 | 6 | 67% |
| T1 | 0 | 11 | 0% |
| T2 | 0 | ~10 | 0% |

**Decision log**:
- 2026-05-03: Doc created. T0 ordering decided: A1 stats → A1 viz → A3 stats → A3 viz → A2 doc → A2 viz scaffold. Rationale: A1+A3 stats are mechanical (existing aggregator infra), A2 needs user H-list decision.
- 2026-05-03: T0a done. Added 4 utility functions (`bootstrap_tost_p` / `bonferroni_adjust` / `holm_bonferroni_adjust` / `bh_fdr_adjust`) + per-arm TOST p computation + post-collection family-based correction in `aggregate_phantom_lift.py`. md output gains comparison-family declaration block + Holm/BH/Bonf/TOST cols in PRIMARY table + new SECONDARY-family adjusted per-arm subsection.
- 2026-05-03: T0b done. New `fig_forest_drop_one.py` reads `phantom_lift.csv` directly (no recomputation) and renders 3-panel forest with TOST equivalence band shaded ±0.5pp. Design move: raw CI as visual primary, Holm-sig as marker fill — separates "estimate uncertainty" from "after-correction sig" so reader doesn't misread CI width.
- 2026-05-03: T0c done. New `aggregate_phantom_meta.py` implements DerSimonian-Laird random-effect pooling with per-family Holm gating + Cochran's Q heterogeneity test + I²/τ². Decision: SE_i from bootstrap CI as `(CI_hi - CI_lo)/(2×1.96)` (standard normal approximation, valid at N=210-234). RE chosen over FE because paper §7 "site-modulated + capability-modulated" framing already implies between-cell true-effect heterogeneity.
- 2026-05-03: T0d done. New `fig_meta_forest.py` uses classical forest convention — weight-sized squares per cell + pooled diamond. Design choice: keep 3-panel structure even when k=1 (P-prompt) so the layout auto-upgrades when 6-cell rerun lands without redesign.
- 2026-05-03: T0a-d Makefile integration done. `phantom-meta` target added to `_aggregate`; `fig_forest_drop_one.py` + `fig_meta_forest.py` added to both `_figures` (full analysis pipeline) and `figures:` (quick regen). `make analysis [FAST=1]` and `make figures` end-to-end confirmed working.

---

## §5 References

- `docs/checkpoints/paper_planning.md` §3 Findings + §4 Section status (evidence framework)
- `docs/checkpoints/实验笔记.md` §108.6 evidence/explanation separation
- `docs/checkpoints/phase1_plan.md` (VWA bug + 6-cell rerun decision context; replaces ADVISOR_SYNC.md retired 2026-05-15)
- Bonferroni: Bonferroni 1936, conservative FWER control
- Holm-Bonferroni: Holm 1979, step-down FWER control (less conservative, same α)
- BH FDR: Benjamini & Hochberg 1995, false discovery rate control
- DerSimonian-Laird random-effect meta: 1986
- TOST equivalence: Schuirmann 1987, Lakens 2017 review
- DeLong AUROC test: DeLong et al. 1988
- Cliff's δ non-parametric effect size: Cliff 1993
