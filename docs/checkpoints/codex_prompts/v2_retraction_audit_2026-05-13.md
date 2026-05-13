# Cross-AI hostile review — P79 v2 NPZ retraction audit

You are an independent NeurIPS / ICML reviewer reading paper drafts + mechanism analysis cold. You have NOT seen any prior Claude analysis. **Do NOT read** `docs/checkpoints/codex_outputs/*`, `docs/checkpoints/process/stress_skill_replica.md`, or any file containing prior reviews. Read paper drafts, mechanism plan, evidence files, and code independently and write your own claim-by-claim attack.

## Context (minimal orientation)

P79 = "Cost-Aware Routing for Web Usage Agents" paper-1 (毕设/Master's thesis). Model under analysis: Qwen3-VL-4B local B1 baseline. 6 observation modes: dom / som / vision / phantom_dom / phantom_som / phantom_text / phantom_prompt. Sites: classifieds + reddit (paper main claim); shopping out of scope. Stage 4 = mechanism analyses (Method 4.2 = PCA cosine gap; Method 4.4 = activation steering).

V2 NPZ migration happened 2026-05-12 late after a prior audit found Bug 2 (SOM_MARKS regex extraction was dropping 71/72 marks). The author has now rewritten `docs/checkpoints/mechanism/plan.md` to reflect v2 numbers and is asking for cross-AI verification.

## Files to read (in this order)

1. `docs/checkpoints/mechanism/plan.md` (just rewritten with v2 retraction)
2. `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md` (v1 vs v2 canonical diff)
3. `docs/checkpoints/mechanism/results/axis2_layer_profile_v2.md`
4. `docs/checkpoints/mechanism/results/axis2_logit_lens_v2.md`
5. `docs/checkpoints/mechanism/results/axis2_per_task_fragility_v2_L17.md`
6. `docs/checkpoints/mechanism/results/axis2_per_task_fragility_v2_L23.md`
7. `docs/checkpoints/mechanism/results/layer_axis_emergence_v2_cls.md`
8. `docs/checkpoints/mechanism/results/layer_axis_emergence_v2_reddit.md`
9. `docs/checkpoints/mechanism/results/stage4_method42_v2_cls.md`
10. `docs/checkpoints/mechanism/results/stage4_method42_v2_reddit.md`
11. `docs/checkpoints/paper_drafts/section5_mechanism.md` (current §5 prose, may be stale vs v2 plan)
12. `scripts/analysis/stage4_pca_cosine_gap.py` (analysis script for v2 metrics)
13. `scripts/mechanistic/run_stage4_multimode_extract.py` if exists (NPZ extraction)
14. `p79/agents/qwen3vl_agent.py` (production `_extract_text_marks`, grep for the function)
15. `scripts/mechanistic/run_stage4_format_variation_extract.py` if relevant

## Code-level questions

### Q1. Is v2 NPZ extraction provably correct?

- v1 buggy regex was `^\[\d+\]\s+\w+` which extracted only 38 chars / 3 lines per task (dropped 71/72 SOM_MARKS).
- v2 uses production `_extract_text_marks`.
- Has anyone verified the SOM_MARKS counts in v2 NPZ went from 1/72 to 72/72 per task?
- Is there a unit test? Or just trust-by-rename? Grep for `test_extract` or similar.
- Read the production extractor and judge: does it actually extract all marks, or does it have its own subtle bug?

### Q2. Method 4.2 cosine gap computation

- Is PCA done per-mode or pooled across modes?
- Is the cosine gap formula `1 - cos(mean_A, mean_B)` (mean-then-cosine) or per-task average of cosines?
- Is the v2 script the same code as v1 with different input NPZ, or is the code also changed? Git log + diff.

### Q3. Logit lens calculation

The author applies `model.model.language_model.norm + model.lm_head` to **PER-LAYER PER-MODE MEAN hidden states**, then computes KL between mode pairs.

- Per-mode means are averages that don't correspond to any actual model forward pass.
- Is KL between two averaged distributions reviewer-defensible?
- Alternative: per-forward-pass KL (decode each task's actual hidden state with logit lens), then average task-level KLs. Which does the script do?
- If the script decodes the AVERAGE, the resulting "amplification" might be an artifact of averaging dampening one distribution more than another, not a real lm_head amplification.

### Q4. Stage 2/3 patching unaffected by v2 NPZ migration?

- Author claims Stage 2/3 patching uses `archive_subset` not Stage 4 NPZ, so unaffected.
- Stage 2/3 builds SoM marks via `build_som_marks`. Does this function call the production extractor or the buggy regex?
- Grep `build_som_marks` to verify. If it uses the production extractor, claim holds. If not, Stage 2/3 results need re-running.

## Logical questions

### Q5. "Cosine-causal disjoint" framing

Plan.md §1.2 calls cosine 0.5-1% vs causal Δoverlap 20-30% a "disjoint" or "geometry underestimates causal".

- Is this terminology defensible? Cosine measures mean-distance in residual stream; KL measures distributional divergence after lm_head decoding; patching measures behavioral output change after intervention.
- These are SUPPOSED to have different magnitudes (different units, different physical meanings).
- Reviewer-3 will say "calling 0.5% mean-distance vs 20% output-change a 'disjoint' is category error". Is this attack defusable?
- What would be the proper terminology / normalization for comparing these?

### Q6. Cross-site dichotomy DIVERGENT (cls vs reddit)

Plan.md §1.3:
- cls v2: clean image-side-based dichotomy (Vision→L04 all 4 / SoM→L36 all 4)
- reddit v2: peak layer mostly L04 (7/8 pairs); only P-text↔SoM at L17

Author interprets this as "honest paper-grade nuance, site-specific mechanism". Reviewer alternative interpretations:

- Sampling noise: 288 examples per site at 0.04 cosine magnitude, what's the bootstrap CI on peak layer?
- Site-specific true mechanism: would need bootstrap CI proving the cross-site difference is statistically significant.
- Pipeline / extraction artifact: re-verify v2 NPZ on reddit didn't have a different bug.

What's the strongest reviewer-3 attack? What evidence would defuse?

### Q7. Things "preserved" by v2 — verifiable or asserted?

Plan.md §0 lists 6 things "preserved" by v2 migration:
1. AUROC linear-readability 1.000 cross-site
2. Image-axis cosine peaks (~0.04-0.07)
3. Stage 2/3 patching (uses archive_subset, not Stage 4 NPZ)
4. Method 4.4 steering (separate pipeline)
5. Exp 5 axis-2 causal patching
6. Drop-one CI strict-positive

For each, is the "unaffected" claim verifiable? Or just asserted? Spot-check 1-2 by reading the relevant code path.

### Q8. v1 → v2 magnitude collapse interpretation

V1 had text-format axis cosine 0.025 at L23, v2 has 0.005 at L36 (-81%). Author interprets v1 number as "NPZ artifact". 

- Alternative: v2 properly includes SOM_MARKS, which HOMOGENIZES the text payload across `som / phantom_som / phantom_text` modes (they all carry the same `[SOM_MARKS]` block). v1 happened to expose true text-format signal because the regex selectively dropped marks in some modes more than others.
- This would mean v2 is OVER-correcting (masking real signal), not v1 being artifact.
- Which interpretation is right? What evidence would distinguish them?

## Deliverable

Output structured markdown, ~1500-2500 words:

### Verdict line (one sentence)
Current state of v2 retraction defensibility.

### Strong claims (survive attack)
1-3 things in plan.md v2 that hold up under hostile reading. Be specific (cite file + line + number).

### Weak claims (would tank under attack)
For each:
- Quote exact claim
- State attack
- State what evidence would defuse
- Effort estimate (h / day / week)

### Code bug findings
Anything you find in the actual analysis scripts: incorrect math, incorrect averaging, missing controls, unit-test absence.

### Honest gaps
Things not in plan.md or §5 prose that a reviewer expects (bootstrap CI on peak layer, sample size justification, etc.)

### Distance to top-tier
Where does v2 plan + §5 prose stand today (workshop / mid-tier / top-tier)? Specific blockers, what would unblock, effort estimate.

### Top 3-5 must-defuse-before-commit items
Ranked list. For each: what to do tonight, ETA.

## Calibration

Hostile-but-fair NeurIPS reviewer-3 mode. User explicitly skeptical of the v2 retraction framing — be brutal on weak terminology, hand-waving, or under-evidenced claims. Specifics > generalities. Quote exact lines. Output 中文为主双语 (criticism specifics in technical English, framework + recommendations in Chinese).

Write to: `docs/checkpoints/codex_outputs/v2_retraction_audit_2026-05-13.md`
