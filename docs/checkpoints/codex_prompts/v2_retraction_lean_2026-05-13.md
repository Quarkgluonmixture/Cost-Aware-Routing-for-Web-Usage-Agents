# Hostile review — P79 v2 NPZ retraction

You are an independent NeurIPS / ICML reviewer with 200+ papers read in mechinterp + multimodal agents. The author rewrote `docs/checkpoints/mechanism/plan.md` on 2026-05-13 after discovering Stage 4 NPZ Bug 2 (regex was dropping 71/72 SOM_MARKS). User wants brutal honest audit.

Read whatever you need, but at minimum:
- `docs/checkpoints/mechanism/plan.md` §0-§1.3 + §5.1 + §7.3.0a
- `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md`
- `scripts/analysis/stage4_logit_lens_axis2.py` (lines 1-150)
- `scripts/analysis/stage4_pca_cosine_gap.py` (lines 1-180)

Find weakness independently. Don't follow any pre-baked attack list. Use your reviewer experience.

Output ≤ 800 words, structured however makes sense. Last line literal: `=== END ===`
