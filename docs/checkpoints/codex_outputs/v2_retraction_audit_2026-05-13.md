# Cross-AI hostile review — P79 v2 NPZ retraction audit

## Verdict line

当前 v2 retraction 的“v1 三轴层级作废、v2 cosine 数字可作为新基线”基本站得住；但“cosine-causal disjoint / lm_head amplification / Method 4.4 preserved”作为 paper-grade hero claim 还不 defensible。现在更像 workshop-strong / mid-tier-borderline 的机制证据栈，离 NeurIPS/ICML top-tier 还差关键控制。

## Strong claims that survive attack

1. **V1 three-axis hierarchy is invalidated.**  
   `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:7-11` 说明 v1 SOM regex produced near-empty payloads and v2 uses production-like extraction; `:15-31` 给出 side-by-side collapse: DOM↔P-text peak 0.0254 -> 0.0047, P-prompt↔P-SoM 0.0292 -> 0.0048, while image pairs mostly preserve scale. 这足以 retract old §5.7 hierarchy。

2. **Method 4.2 metric is mean-then-cosine, not per-task cosine average.**  
   `scripts/analysis/stage4_pca_cosine_gap.py:70-88` builds `states`, then `means`, then computes `cosine_gap(c1, c2)` per layer. PCA is **per-mode**, not pooled: `:135-141` fits PCA on `X = states[mode]`. 这回答 Q2：cosine gap = `1 - cos(mean_A, mean_B)`.

3. **Stage 2/3 patching does not directly consume Stage 4 NPZ.**  
   `scripts/mechanistic/run_stage2b_continuation_pilot.py:121-126` imports `p79.experiment.som._extract_text_marks`; `:325-375` reads archived `observation_dom.txt` and builds source/target inputs directly. 所以 “Stage 2/3 not invalidated by Stage 4 NPZ replacement” directionally correct. Caveat: it still shares formatter assumptions and omits production option-line augmentation.

## Weak claims that would tank under attack

### 1. Claim: “V2 NPZ uses production `_extract_text_marks` (full 72-line `[id=N] {label}` payload).”

Source: `docs/checkpoints/mechanism/plan.md:13`.

**Attack.** This is over-specific and partly false. The v2 NPZ stores hidden states and labels, not text payloads or mark counts. Provenance stores formatter module/hash (`results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.provenance.json:71-82`; reddit `:94-105`) but no per-task payload audit. Reconstructing selected step-2 local archives gives:

- cls selected tasks: new extractor mark counts min/median/max = **30 / 66 / 73**, not 72/72; old regex counts = **1 / 1 / 1**, old payload chars 35-38.
- reddit selected tasks: new counts **2 / 46 / 111**, old regex counts = **1 / 1 / 1**.

So v2 clearly fixes catastrophic 1-line extraction, but “72-line per task” is not proven and not true globally. There is also no unit test; `rg test_extract` only finds unrelated external-module tests.

**Defuse evidence.** Add a payload audit sidecar: selected task, step, mark_count, payload_char_count, sha256(payload), old_regex_count, old_regex_chars. Add regression tests for `_extract_text_marks`.  
**Effort:** 3-5h.

### 2. Claim: “New hero claim: cosine-causal disjoint … reviewer-defensible.”

Source: `docs/checkpoints/mechanism/plan.md:24`, `:64-69`, `:186-193`.

**Attack.** Reviewer-3 will call this a category error. Cosine gap is a mean residual-vector angular distance; KL is distributional divergence after `norm + lm_head`; patching Δoverlap is text-continuation displacement under intervention. “0.5-1% vs 20-30%” compares different units, scales, and estimands. “Orders of magnitude” sounds precise but has no shared denominator.

**Defuse evidence.** Replace with: “small mean-geometry but large intervention effect under our patching metric.” If comparing, normalize within each metric against controls: task-shuffled patching, same-layer random directions, image-axis positive controls, and per-task standardized effect sizes. Report z-scores or percentile ranks, not raw cross-unit ratios.  
**Effort:** 1 day wording + 2-3 days controls/plots.

### 3. Claim: “lm_head amplifies cosine→KL by 8-44×.”

Source: `docs/checkpoints/mechanism/plan.md:59-67`; result table at `axis2_logit_lens_v2.md:14-22`, `:30-38`.

**Attack.** The logit lens script decodes **per-mode averaged hidden states**, not actual forward-pass hidden states: `scripts/analysis/stage4_logit_lens_axis2.py:109-127`. KL between softmaxes of averaged vectors is not the mean KL of model states. Since `softmax(lm_head(norm(mean h)))` is nonlinear, “amplification” may be a mean-collapse artifact. The result file itself admits “per-layer per-mode mean hidden states” (`axis2_logit_lens_v2.md:3-6`).

**Defuse evidence.** Decode each actual hidden state per task/mode/layer, compute paired KL per task/step, then average and bootstrap. Compare `KL(mean decoded distributions)` vs `KL(decoded mean hidden states)` to quantify averaging distortion.  
**Effort:** 1-2 days.

### 4. Claim: “Cross-site disagreement is real” and “site-specific mechanism.”

Source: `docs/checkpoints/mechanism/plan.md:71-97`.

**Attack.** The raw v2 tables are weaker than the interpretation. Classifieds `layer_axis_emergence_v2_cls.md:7-14` shows all Vision pairs peak L04 and all SoM pairs peak L36; grouped by no-image text, both AXTree and `[SOM_MARKS]` have mean peak L20 (`:18-30`). Reddit `layer_axis_emergence_v2_reddit.md:7-14` shows 7/8 image pairs peak L04. Yet both result files still contain stale prose saying `[SOM_MARKS]` shifts to L17-L36 (`cls :34-42`, reddit `:34-42`), contradicted by their own tables.

Strongest reviewer attack: peak layer is an argmax over 37 noisy layers at cosine magnitude ~0.04, N=24 tasks, no bootstrap over tasks/steps, no CI over layer argmax. “Site-specific mechanism” could be sampling noise or pipeline artifact.

**Defuse evidence.** Bootstrap peak-layer distributions per pair/site; report P(peak=L04), P(peak=L36), and CI for top-two layer gap. Add paired cross-site test for “SoM-side late vs early.”  
**Effort:** 1 day.

### 5. Claim: “Method 4.4 steering unchanged by v2.”

Source: `docs/checkpoints/mechanism/plan.md:21`, `method42_v1_vs_v2_comparison.md:88-90`.

**Attack.** Current code contradicts this. `scripts/mechanistic/run_stage4_method44_v2_sweep.py:55` points to `hidden_states.npz`, not `hidden_states_v2_fixed.npz`; local inspection shows `hidden_states.npz` is 288 examples with steps [2,5], while v2 fixed is 144 examples with step [2]. The script’s `build_som_marks` is a crude line filter (`:65-67`), not production extractor. Method 4.4 is not proven isolated from the v1 payload bug.

**Defuse evidence.** Patch Method 4.4 to use `hidden_states_v2_fixed.npz`, production-aligned marks, pinned revision, and rerun sweep. Compare old vs new direction cosine/norm per layer and top cells.  
**Effort:** 1 day GPU + 2-4h analysis.

### 6. Claim: “5/5 robustness pass.”

Source: `docs/checkpoints/mechanism/plan.md:151-156`.

**Attack.** `scripts/analysis/stage4_robustness.py:24-27` defaults to legacy `hidden_states.npz`, not v2. It assumes step 2 vs step 5 (`:130-148`, `:217-220`), while v2 provenance says `steps: [2]` and shape 144, not 288. This is not a v2 robustness pass.

**Defuse evidence.** Rerun robustness on v2, or label the existing robustness suite v1-only. If step invariance is desired, re-extract v2 step 5.  
**Effort:** 4-8h if data exists; 1 day if extraction needed.

### 7. Claim: “V2 data = 288 examples.”

Source: `docs/checkpoints/mechanism/plan.md:224`; §5 draft also says 288 at `section5_mechanism.md:29`.

**Attack.** v2 result files say 144 examples and per-mode n=24 (`stage4_method42_v2_cls.md:3-4`; reddit same `:3-4`). Provenance records `hidden_state_shape: [144, 37, 2560]`. The 288 claim is inherited from v1 step2+step5 and silently doubles N.

**Defuse evidence.** Correct prose to “24 tasks × 1 step × 6 modes = 144 examples” or re-extract step 5.  
**Effort:** 1h prose; 1 day extraction.

## Code bug findings

- **Method 4.4 v2 sweep uses v1 NPZ and non-production marks.** `run_stage4_method44_v2_sweep.py:55`, `:65-67`. Highest-impact code bug because it attacks a “preserved” claim.
- **Logit lens computes KL of decoded means, not average per-forward KL.** `stage4_logit_lens_axis2.py:109-127`. Acceptable exploratory visualization, not proof of “lm_head amplification.”
- **Robustness suite is v1-default and step5-dependent.** `stage4_robustness.py:25`, `:130-148`. Not valid v2 evidence unless rerun.
- **Stage4 and Stage2B mark builders omit production option-line augmentation.** Production `_build_som_result` recovers `[OPTIONS]` / `[DROPDOWN OPTIONS]` (`p79/experiment/som.py:222-249`), while Stage4 only formats `_extract_text_marks` output (`run_stage4_multimode_extract.py:59-63`). Not identical to full production SoM text on dropdown tasks.
- **Format variation extraction uses old regex-like parsers.** `run_stage4_format_variation_extract.py:59-84` extracts only lines matching `[N] role 'label'` or simplistic startswith logic. Fine for controlled variants, not “production-aligned” evidence.
- **Per-task fragility files have stale labels/verdict text.** L17 file says top-task section “@ L23” (`axis2_per_task_fragility_v2_L17.md:26`), and both L17/L23 verdicts compare to fixed `% > 0.010`, not the reported mean (`:68-75`).

## Honest gaps

- **Plan §0 “preserved” list triage.** AUROC 1.000 and image-axis magnitude are verifiable from `stage4_method42_v2_{cls,reddit}.md`, though AUROC still needs payload provenance. Stage 2/3 patching is mostly verifiable as archive-direct, not NPZ-direct. Method 4.4 is **not** verifiably preserved because current code uses legacy NPZ. Exp 5 axis-2 causal patching may be archive-direct, but it needs the same formatter/path audit before being called unaffected. Drop-one CI strict-positive is outside Stage 4 NPZ and likely unaffected, but this audit did not re-read the behavioral bootstrap path; in plan wording it should be “not touched by this migration,” not “mechanistically preserved.”
- No bootstrap CI for peak-layer argmax or cross-site layer differences.
- No v2 payload sidecar proving exact payload counts/hashes used for each hidden-state row.
- No unit test for `_extract_text_marks` and no regression test for the old regex failure.
- No per-forward-pass logit-lens KL; current KL is decoded-mean only.
- No causal bridge from 50-token continuation patching to task success / drop-one oracle value.
- No completed task-shuffled content-specificity control in the cited stack; Gaussian controls are weaker.
- Sample-size statement inconsistent: v2 is 144 examples, while plan/draft still say 288.
- Cross-family generalization remains untested.

## Distance to top-tier

今天的状态：**workshop-strong / mid-tier-borderline**, not top-tier. The empirical retraction itself is honest and valuable, but the replacement hero claim is over-framed.

Top-tier blockers:

- v2 provenance must be auditable at payload level, not “trust-by-rename.”
- Method 4.4 must be rerun on v2; otherwise “preserved steering” is vulnerable.
- Logit lens must move from decoded means to per-example paired KL.
- Cross-site peak-layer claims need bootstrap distributions or must be downgraded to descriptive.
- The paper needs language discipline: stop comparing raw cosine, KL, and overlap as if they share units.

Unblock estimate: 2-4 days for minimum defensible revision; 1-2 weeks for a top-tier-grade mechanism appendix with all controls.

## Top 5 must-defuse-before-commit items

1. **Fix Method 4.4 v2 path and rerun.**  
   Tonight: patch `NPZ` to `hidden_states_v2_fixed.npz`, replace `build_som_marks` with production-aligned helper, rerun reported best layers. ETA: 1 day.

2. **Generate v2 payload audit sidecars.**  
   Tonight: script mark counts/chars/hashes for every selected task/step and compare old regex counts. Add one unit test. ETA: 3-5h.

3. **Rewrite “cosine-causal disjoint.”**  
   Tonight: replace “orders of magnitude / reviewer-defensible” with “small mean-geometry, large intervention effect under separate metrics.” ETA: 1-2h.

4. **Rerun or retract v2 robustness claims.**  
   Tonight: mark current `stage4_robustness.py` outputs as v1-only; either re-extract v2 step5 or remove step-invariance from v2 plan. ETA: 4-8h analysis, 1 day if extraction.

5. **Replace logit-lens amplification claim with per-example KL or downgrade.**  
   Tonight: add explicit caveat that current KL is decoded mean-hidden-state KL. ETA: 1h prose; 1-2 days for GPU-backed per-forward KL.
