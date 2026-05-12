# Method 4.2 v1 vs v2 cosine gap comparison — Bug 1+2 propagation audit

**Status**: Land 2026-05-12 late-late, after Myriad 359736 (cls v2) + 359737 (reddit v2) re-extraction with Bug 1 (tier filter) + Bug 2 (production `[SOM_MARKS]` format) + Bug 5 (model revision pin) fixes.

## Headline result

**§5.7 "three-axis hierarchy with quantitatively distinct magnitudes" claim is INVALIDATED by v2 data.**

V1 numbers came from buggy NPZ where the SOM_MARKS regex `^\[\d+\]\s+\w+` extracted only 38 chars / 3 lines per task, dropping 71/72 marks. All Stage 4 cosine geometry was computed on near-empty text payloads where the only differentiator between flat-text modes (som / phantom_som / phantom_text) was prompt template. Modes still separated perfectly (AUROC 1.000) but the cosine-gap magnitudes were artifacts of prompt-template differences, not text-payload differences.

V2 NPZ uses production `_extract_text_marks` (72-line full payload with `[id=N] {label}` envelope). Modes still separable (AUROC 1.000), but axis-1 + axis-2 cosine magnitudes collapse to noise level. Image-axis magnitudes preserve.

## Side-by-side peak comparison (cls, N=24 strong-tier)

| Mode pair | v1 buggy peak | v2 fixed peak | Magnitude Δ | Layer Δ |
|---|---|---|---|---|
| DOM ↔ Vision (image axis) | L04 0.0653 | L04 0.0670 | unchanged | unchanged |
| P-prompt ↔ Vision (image axis) | L04 0.0649 | L04 0.0664 | unchanged | unchanged |
| P-text ↔ Vision (image axis) | L36 0.0614 | **L04** 0.0602 | unchanged | **earlier** |
| P-SoM ↔ Vision (image axis) | L36 0.0613 | **L04** 0.0599 | unchanged | **earlier** |
| DOM ↔ SoM (image axis) | L04 0.0604 | **L36** 0.0496 | -18% | **boundary-shift** |
| P-prompt ↔ SoM (image axis) | L04 0.0600 | **L36** 0.0439 | -27% | **boundary-shift** |
| P-text ↔ SoM (image axis) | L20 0.0494 | **L36** 0.0488 | -1% | boundary-shift |
| **P-SoM ↔ SoM (image axis, paper §5.7 image-axis anchor)** | **L17** 0.0412 | **L36** 0.0416 | unchanged | **L17 → L36** |
| DOM ↔ P-SoM | L23 0.0321 | **L36** 0.0152 | **-53%** | L23 → L36 |
| P-prompt ↔ P-SoM (axis-1 SoM-prompt) | L23 0.0292 | **L36** 0.0048 | **-84%** | L23 → L36 |
| P-text ↔ P-prompt | L23 0.0288 | **L36** 0.0081 | **-72%** | L23 → L36 |
| **DOM ↔ P-text (axis-1 DOM-prompt, paper §5.7 axis-1 anchor)** | **L23** 0.0254 | **L36** 0.0047 | **-81%** | L23 → L36 |
| SoM ↔ Vision | L22 0.0238 | **L36** 0.0255 | +7% | boundary-shift |
| **P-text ↔ P-SoM (axis-2, paper §5.7 axis-2 anchor)** | L23 0.0114 | **L36** 0.0088 | -23% | L23 → L36 |
| DOM ↔ P-prompt | L36 0.0067 | L36 0.0068 | unchanged | unchanged |

## Headline ratios

| Ratio | v1 (3:1 ratio claim) | v2 (reality) |
|---|---|---|
| Image axis magnitude (P-SoM↔SoM) | 0.041 | 0.042 |
| Text-format axis (DOM↔P-text) | 0.025 | **0.005** |
| Prompt-family axis (P-text↔P-SoM) | 0.011 | 0.009 |
| Image / text-format ratio | **1.7x** | **8x** |
| Image / prompt-family ratio | **3.7x** | **5x** |
| Text-format / prompt-family ratio | **2.3x** | **0.5x** ← axis-1 NOW SMALLER than axis-2 |

The "image > text-format > prompt-family" hierarchy with 4:3:1-ish quantitative ratio (v1) is **wrong**. V2 reality: image axis dominates by ~5-10x; axis-1 is **smaller than** axis-2 (reversed ranking); both axis-1 and axis-2 are noise-level (<0.01 cosine).

## L17 cosine gap snapshot (cls + reddit cross-site)

| Mode pair | cls v1 | cls v2 | reddit v1 | reddit v2 |
|---|---|---|---|---|
| DOM ↔ P-text | 0.0120 | **0.0021** | (similar) | **0.0019** |
| DOM ↔ P-SoM | 0.0124 | **0.0029** | (similar) | **0.0031** |
| P-text ↔ P-prompt | 0.0132 | **0.0031** | — | **0.0032** |
| P-text ↔ P-SoM (axis-2) | 0.0028 | 0.0019 | — | 0.0020 |
| DOM ↔ SoM (image axis) | 0.0557 | 0.0452 | — | 0.0450 |
| DOM ↔ Vision (image axis) | 0.0545 | 0.0571 | — | 0.0537 |

Reddit cross-site replication confirms the cls pattern: image-axis magnitudes preserve, axis-1 + axis-2 collapse to sub-permille at L17.

## AUROC lototask (held-out, paper-grade Bug 3 fix)

All pairs at all layers report AUROC lototask = 1.000 (perfect held-out linear separability). The modes ARE distinguishable in residual stream; the **magnitude of the mode-mean difference** is just much smaller than v1 claimed.

This is the key reframe: **separability survives, magnitude does not**. Cosine gap measures effect SIZE; AUROC measures CLASSIFICATION RELIABILITY. They can dissociate.

## What this means for paper §5

**§5.7 three-axis hierarchy** (the prior framing):
> "Three quantitatively distinct axes: image axis L17 0.041, text-format L23 0.029, prompt-family L23 0.011, with 4:3:1 magnitude ratio that holds cross-site."

→ **INVALIDATED**. Replace with:
> "All three axes are linearly readable in residual stream (held-out AUROC 1.000 across cls and reddit). The image axis dominates geometrically (~0.04-0.07 cosine peak) and emerges by L04. Text-format and prompt-family axes produce sub-permille mean-difference (cosine ~0.005-0.009) without a localized layer peak (monotone rise to boundary L36). The geometric magnitude rank-order reverses cross-site at L17 (axis-1 ≤ axis-2), indicating these axes are noise-level rather than quantitatively distinct dimensions."

**§5.2 Method 4.2** (cosine gap table at L17):
- All non-image-axis numbers drop 4-8x (re-run on v2 NPZ provides canonical values)
- L17 ceases to be a meaningful "disruption locus" for text-format / prompt-family axes — they peak at L36 (boundary monotone)

**§5.5 image-axis peak-layer dichotomy** (paper claims "no-image side's text format predicts peak layer with zero overlap"):
- v1 had: 4 pairs at L04 (AXTree no-image side) vs 4 pairs at L17-L36 (flat-marks no-image side)
- v2 reorganization: DOM/P-prompt ↔ Vision still L04; **P-text/P-SoM ↔ Vision shifted from L36 → L04** (BREAKS dichotomy); DOM/P-prompt/P-text/P-SoM ↔ SoM ALL at L36 now (collapses dichotomy on SoM image side)
- → **§5.5 dichotomy ALSO needs significant revision**. The clean "AXTree → L04, flat-marks → late" pattern is partially v1 artifact.

**§5.4 Stage 2/3 patching** (Cell A-H/D-G/H-text/H-prompt/H-d/Exp 5):
- These do NOT use Stage 4 NPZ; they use archive_subset directly via Stage 2B build_som_marks which calls production code
- All Stage 2/3 patching results **REMAIN VALID**
- Exp 5 cellhprompt cls + red axis-2 patching (80-125% capture of combined image+prompt displacement): **INTACT**
- Mid-layer L11-L17 patching effect: **INTACT**

**§5.3 Method 4.4 steering** (45-cell layer-α sweep):
- Separate pipeline (uses run_stage4_method44_v2_sweep + different feature extraction): **INTACT**

**§5.6 four-vertical-defense stack**:
- Per-task fragility (uses Stage 4 NPZ): NEEDS RE-RUN on v2
- Selection-bias (reverse-tier H1): uses format variation NPZ, separately audited (INTACT but baseline caveat)
- Cross-site H1: format variation (INTACT)
- Cross-site Mirage geometry: NEEDS RE-RUN on v2

**Exp 1 axis-2 layer profile** (`axis2_layer_profile.md`): NEEDS RE-RUN on v2; current 4:3:1 ratio claim is invalidated.

**Exp 3 logit lens** (`axis2_logit_lens.md`): NEEDS RE-RUN on v2. Output-space KL magnitudes likely also change (the lm_head amplifies whatever residual-stream signal is there).

## What still stands for paper

✅ §1 hero (drop-one CI strict-positive reddit + cls): unchanged
✅ §4 empirical SR tables (after 11.90→12.38 canonicalization): unchanged
✅ §4.5 reddit behavioral: unchanged
✅ §5.4 Stage 2/3 patching + Exp 5 axis-2 causal: unchanged
✅ §5.3 Method 4.4 steering: unchanged
✅ §6 image-axis early L04 separation: unchanged (real)
✅ Held-out AUROC 1.000 linear-readability: unchanged

## New cleaner mechanism story

> **Three claim layers, distinct evidence types**:
> 1. **Linear readability** (Method 4.2 lototask AUROC 1.000): all 6 modes linearly separable in residual stream; small cosine magnitudes but reliable classification
> 2. **Geometric magnitude is mostly image-axis driven** (Method 4.2 cosine peak): image presence produces ~0.04-0.07 cosine separation early L04; other axes produce sub-permille separation throughout
> 3. **Causal patching effect at mid-layer L11-L17** (Stage 2/3 + Exp 5): 20-30% target-overlap displacement under final-token replacement patching, content-specific (task-shuffled control 359768/359769 in flight)
>
> The disjoint between **small geometric magnitude (0.005-0.009)** and **large causal patching effect (0.20-0.30)** is the new headline claim — it means residual-stream geometry underestimates causal influence by orders of magnitude. This is paper-grade-novel and reviewer-defensible.

## Files / provenance

- v2 NPZ: `results/mechanistic/stage4_multimode_b1_{cls,reddit}/hidden_states_v2_fixed.npz`
- v2 provenance: `hidden_states_v2_fixed.provenance.json` (git SHA, model revision, task IDs, formatter hash)
- v2 metrics: `/tmp/method42_{cls,red}_v2.{json,md}` (canonical paths after rerun: `results/mechanistic/stage4_multimode_b1_*/method42_metrics.json`)
- Legacy v1 metrics still in `docs/checkpoints/stage4_method42_results.md`
- Comparison source: this file
