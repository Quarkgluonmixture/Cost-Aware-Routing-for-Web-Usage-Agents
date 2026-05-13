Reading prompt from stdin...
OpenAI Codex v0.128.0 (research preview)
--------
workdir: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: high
reasoning summaries: none
session id: 019e204f-d79d-7760-be09-b5fedd10da4f
--------
user
# Hostile review — P79 v2 NPZ retraction

You are an independent NeurIPS / ICML reviewer with 200+ papers read in mechinterp + multimodal agents. The author rewrote `docs/checkpoints/mechanism/plan.md` on 2026-05-13 after discovering Stage 4 NPZ Bug 2 (regex was dropping 71/72 SOM_MARKS). User wants brutal honest audit.

Read whatever you need, but at minimum:
- `docs/checkpoints/mechanism/plan.md` §0-§1.3 + §5.1 + §7.3.0a
- `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md`
- `scripts/analysis/stage4_logit_lens_axis2.py` (lines 1-150)
- `scripts/analysis/stage4_pca_cosine_gap.py` (lines 1-180)

Find weakness independently. Don't follow any pre-baked attack list. Use your reviewer experience.

Output ≤ 800 words, structured however makes sense. Last line literal: `=== END ===`

codex
I’ll audit the stated sections first, then inspect the Stage 4 scripts around the parsing and tensor-shape assumptions. I’ll keep the review in a hostile-review frame: claims, evidence chain, and what would fail under scrutiny.
exec
/bin/bash -lc "sed -n '1,220p' docs/checkpoints/mechanism/plan.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
---
name: mechanism plan
description: Full mechanism workspace — theory, lit anchor stack, methods, identification protocol, current findings, open questions, advisor sync, roadmap. Specialized companion to paper_planning §2; not a duplicate.
type: workspace_plan
last_substantive_update: 2026-05-13
v2_retraction: 2026-05-13 — Stage 4 v1→v2 NPZ migration (Bug 1 tier filter + Bug 2 SOM_MARKS regex + Bug 5 model revision pin) invalidated several v1 quantitative claims. See §0 below.
---

# Mechanism Plan — paper §5

## 0. v2 retraction summary (2026-05-13)

V1 Stage 4 NPZ regex `^\[\d+\]\s+\w+` extracted only 38 chars / 3 lines per task, dropping 71/72 SOM_MARKS. Affected: Method 4.2 cosine geometry, Exp 1 axis-2 layer profile, Exp 3 logit lens, per-task fragility. V2 NPZ uses production `_extract_text_marks` (full 72-line `[id=N] {label}` payload). Re-extraction Myriad 359736 (cls) + 359737 (reddit) landed 2026-05-12 late, v2 metrics 2026-05-13 02:52.

**What changed**:
- ✗ V1 "three-axis hierarchy 4:3:1 magnitude ratio" → INVALIDATED. V2: image dominates ~5-10×; axis-1 and axis-2 both noise-level (cosine ~0.005-0.009); axis-1 magnitude is now ≤ axis-2 (reversed ranking).
- ✗ V1 "AXTree → L04 vs flat → L17-L36" no-image-side dichotomy → REORGANIZED. V2: dichotomy is image-side-based (Vision→L04, SoM→L36), not text-format-based.
- ✓ AUROC linear-readability 1.000 cross-site → preserved.
- ✓ Image-axis cosine peaks (~0.04-0.07) → preserved.
- ✓ Stage 2/3 patching (uses archive_subset, not Stage 4 NPZ) → unchanged.
- ✓ Method 4.4 steering (separate pipeline) → unchanged.
- ✓ Exp 5 axis-2 causal patching → unchanged.

**New hero claim** (replaces v1 three-axis hierarchy): **cosine-causal disjoint** — geometric magnitude is sub-permille (0.005-0.009) but causal patching displaces overlap 20-30% AND lm_head amplifies cosine→KL by 8-25×. Residual-stream geometry underestimates causal influence by orders of magnitude; cosine gap measures effect SIZE while AUROC measures CLASSIFICATION RELIABILITY and they dissociate. Paper-grade novel + reviewer-defensible.

Provenance: `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md` (canonical v1↔v2 diff). V2 NPZ at `results/mechanistic/stage4_multimode_b1_{cls,reddit}/hidden_states_v2_fixed.npz`.

## 1. Theory framework (1-screen summary, paper_planning §2 is canonical)

### 1.1 Zoom 1-4 hierarchy

| Zoom | Level | What our paper claims |
|---|---|---|
| **1** | Architectural | Phantom routing space = "skip annotated image" boundary contains 3 arms (P-text / P-prompt / P-SoM) sharing 4-fold drop-in property |
| **2** | Behavioral (axis effects) | Axis 1 (text payload: AXTree vs [SOM_MARKS]) + Axis 2 (prompt: SoM-prompt vs DOM-prompt) are both linearly readable (AUROC 1.0) but geometrically sub-permille; Axis 3 (image presence: in vs out) is gating + geometrically dominant |
| **3** | Named phenomena (lit-anchored) | Mirage Effect (Asadi 2026) / Scaffold Effect (Vu&Balloccu 2026) / Cross-modal flow (Kaduri) / Prompt-format sensitivity (Sclar 2024) |
| **4** | Model-internal | L11-L17 mid-layer window is BOTH causal patching disruption locus (Stage 2/3 Δoverlap -0.27 to -0.35) AND probe-decodable (AUROC 1.0 via lototask held-out). L23-L25 logit-lens window is where mode signal concentrates at output distribution. |

### 1.2 Cosine-causal disjoint (Method 4.2 v2 + Stage 2/3 + Exp 3 logit lens)

V2 NPZ-corrected geometry (paper-grade canonical, 2026-05-13):

| Axis | Pair | L17 cos gap | Peak L / gap | Notes |
|---|---|---:|---:|---|
| **Image-axis (Vision)** | DOM ↔ Vision | 0.057 | L04 0.067 | early visual encoder |
| **Image-axis (SoM)** | P-SoM ↔ SoM | 0.003 | L36 0.042 | late integration |
| Axis-1 text-format | DOM ↔ P-text | 0.002 | L36 0.005 | sub-permille, monotone-to-boundary |
| Axis-1 text-format | P-prompt ↔ P-SoM | 0.002 | L36 0.005 | sub-permille |
| Axis-2 prompt-family flat | P-text ↔ P-SoM | 0.002 | L36 0.009 | sub-permille |
| Axis-2 prompt-family hier | DOM ↔ P-prompt | 0.001 | L36 0.007 | sub-permille |

**Geometric magnitudes** v2: image 0.04-0.07 / text-format 0.005 / prompt-family 0.009 → image dominates **5-10×**, axis-1 ≤ axis-2 (sub-permille).

**Causal patching magnitudes** (Stage 2/3 mid-layer L11-L17 window, 6/6 cells cross-site):
- Δoverlap-to-target: -0.27 to -0.35 (cls + reddit, all SoM→{no-image-arm} forward cells)
- Random injection control (E + Er): null effect
- → **causal patching effect magnitude 20-30%** vs **geometric magnitude 0.5-1%**

**Logit lens amplification** (Exp 3 v2, Qwen3-VL-4B `norm + lm_head` on per-layer means):
- Axis-2 P-text↔P-SoM cosine 0.002 at L17 → KL **0.088 at L25** (cls), 0.057 at L25 (reddit)
- Cosine→KL amplification: **8-44× depending on pair**, peak amplification at L21-L25 decoding window
- KL collapses to ~0 at L36 (mean hidden collapses to common JSON-header prefix) → mode-distinct signal lives in **L23-L25 window**, not final embedding

**Interpretive disjoint**: residual-stream cosine geometry severely underestimates causal influence. Three converging numbers:
- Cosine gap 0.5-1% (geometric magnitude small)
- Δoverlap 20-30% (causal effect large)
- KL ~0.05-0.09 (output divergence intermediate, amplified 8-44× from cosine)

This is the new paper §5 hero claim. AUROC linear-readability 1.000 holds throughout — modes ARE distinguishable in residual stream; the magnitude of the mode-mean difference is just much smaller than v1 claimed.

### 1.3 Image-axis peak-layer signature (v2 — cross-site DIVERGENT, needs further work)

V2 NPZ data shows the dichotomy **does NOT replicate cleanly cross-site**. This is a v2-revealed paper-grade nuance not present in v1:

**Cls v2**: clean image-side-based dichotomy

| Image side | Peak layer | All 4 pairs cos gap |
|---|---|---:|
| Vision (naked) | **L04** | 0.060-0.067 |
| SoM (annotated) | **L36** | 0.042-0.050 |

**Reddit v2**: peak layer mostly L04 across the board (7/8 pairs), only P-text↔SoM at L17.

| Image side | Peak layer | Pairs |
|---|---|---|
| Vision (naked) | L04 (all 4) | DOM/P-text/P-prompt/P-SoM ↔ Vision |
| SoM (annotated) | L04 (3/4) | DOM↔SoM 0.046, P-prompt↔SoM 0.043, P-SoM↔SoM 0.039 |
| SoM (annotated) | **L17 (1/4)** | P-text↔SoM 0.043 |

**Cross-site disagreement is real**: cls SoM-image pairs all defer to L36 late integration; reddit SoM-image pairs mostly emerge at L04 with one exception. Possible explanations:
1. Reddit's smaller/sparser SoM overlay produces clearer early visual discrepancy regardless of text-payload format
2. Cls listing-heavy DOM trees push annotated SoM cosine peak past mid-layers; reddit comment-thread DOM doesn't
3. V2 NPZ sampling variance (288 ex each is borderline for layer-peak precision at 0.04 magnitude)

**v1 framing retraction**: v1 said the dichotomy was no-image-side-text-based (AXTree → L04 vs `[SOM_MARKS]` → L17-L36) and cross-site stable. V2 data on cls reorganizes to image-side-based; v2 data on reddit collapses to L04 dominant. Neither v1 nor a single v2 reorganized framing replicates cross-site.

**Paper §5 prose implication**: do NOT make a "peak-layer dichotomy is universal mechanism" claim. Honest framing: image-axis cosine peak structure varies by site (cls late-integration on SoM, reddit early-integration), with **AUROC linear-readability 1.000 preserved cross-site at all layers**. The "Mirage signature" claim must be reframed around AUROC + cosine magnitude rank-order (image > text-format ≈ prompt-family), not peak-layer location.

### 1.4 H1 test confirms broader: flat-list (not just indexed) triggers shortcut (2026-05-12)

Format variation extraction (Myriad job 352998, `stage4_format_variation_b1_cls`, 450 hidden states = 45 task-step × 10 modes). For each text format V, compute image-axis cosine gap V↔som per layer; peak layer reveals shortcut activation:

| Format | Peak layer | Verdict |
|---|---|---|
| **AXTree hierarchical (DOM)** | **L04** | **SOLE format defeating shortcut** |
| `"a, b, c, ..."` plain sentence | L17 | mid-level trigger |
| `[N] role 'label'` (SoM standard) | L36 | strong trigger |
| `@N label` (Browser Use) | L36 | strong trigger |
| `id_N: label` (AppAgent) | L36 | strong trigger |
| `[BN:r:l]` (Tarsier) | L36 | strong trigger |
| `N. label` (numbered) | L36 | strong trigger |
| `<el_N>label</el_N>` (XML) | L36 | strong trigger |
| `#hash label` (control: no integer) | L36 | **still triggers!** |

**Refined H1 verdict**: trigger is **flat element listing**, not "indexed list pattern". Even integer-free hash IDs and pure-sentence variants engage the shortcut. AXTree hierarchical depth is the **unique format** that defeats shortcut activation.

Paper §5 implication: SoM-family web agents (Browser Use, AppAgent, Tarsier, OmniParser, etc.) **all** implicitly exploit the same flat-list-element-grounding shortcut from VLM training distribution. P79 phantom routing space makes this systematic and routes accordingly.

## 2. Literature anchor stack (5 anchors, all 2026-05-08 except Sclar 2024)

| Anchor | Role | bib key | What it gives our paper §5 |
|---|---|---|---|
| **Wu et al. 2026** (UCL lab, our advisors) | Method backbone | `wu2026toolcalling` (2605.07990) | Mean-difference activation steering at second-to-last layer, 77–100% switch on tool selection (93–100% at 4B+). Our Method 4.2/4.4 port to multimodal Qwen3-VL-4B web agent |
| **Ma & Rui 2026** | Probe-vs-causal vocabulary | `maRui2026planning` (2605.07984) | "Planning-compatible representation" vs "causally active planning site". Qwen3-family pattern: probe works, causal patching weak (1% rhyme newline causal vs Gemma 67%). Our Method 4.4 v2 50% reliability is consistent with this family pattern |
| **HDMI / Khorasani et al. 2026** | Alt method + evaluation metric | `khorasani2026hdmi` (2605.07631) | Probe-free gradient-based steering. Critically: **completeness × selectivity → harmonic mean reliability** — what our Method 4.4 v2 reports (not raw shift rate) |
| **Lin & Liu 2026 Position paper** | Methodology protocol | `linLiu2026disclosure` (2605.08012) | 5-step identification disclosure norm: state claim / name strategy / enumerate assumptions / stress-test / separate validation. Paper §5 adopts as identification subsection structure |
| **Peale et al. 2026** | §6 routing theory | `peale2026flexibleRouting` (2605.07805) | Uncertainty decomposition (reducible + irreducible) with regret bound. Paper §6 theoretical anchor; 4-fold drop-in maps onto predict/route/abstain trichotomy |

## 3. Methods (Stage 4 + planned)

### 3.0 Curation pipeline (Stage 0 setup, shared by Stages 1-4)

`scripts/mechanistic/curate_mirage_tasks.py` builds a strong-tier task manifest from a B1 production run:
- Filter to tasks where Mirage Effect is well-defined: agent succeeds in `som` mode but fails in `dom` mode (real image-feature dependence, not random failure)
- Pick 24 cls + 24 reddit "strong-tier" tasks (clean Mirage); separate 15 cls reverse-tier (DOM > SoM, selection-bias defense)
- Archive raw observations (AXTree, screenshots, SoM-marks) per (task, step) → `archive_subset_b1_<site>/<site>_task_<tid>/step_<step>/` for offline patching/extraction (Stages 2-4 read from this archive, no live env required)

Outputs:
- `results/mechanistic/curate_mirage_b1_classifieds/manifest.json` — cls strong/reverse tier task list
- `results/mechanistic/curate_mirage_b1_reddit/manifest.json` — reddit strong tier
- `results/mechanistic/archive_subset_b1_cls/` (17 MB, 144 files, 24 tasks × 6 steps)
- `results/mechanistic/archive_subset_b1_reddit/` (35 MB, 356 files, 24 tasks × ~15 steps)

### 3.1 Method 4.2 — PCA cosine gap (DONE)

`scripts/analysis/stage4_pca_cosine_gap.py` + `stage4_robustness.py`. Three metrics per (mode_pair, layer):
- A. Cosine gap = 1 − cos(mean_A, mean_B)
- B. AUROC via (mean_A − mean_B) projection
- C. Per-(mode, layer) PCA top-10 variance explained

**5/5 robustness pass**:
- Test A label perm: 9.8σ above noise (real 1.000 vs perm 0.629)
- Test B per-task: 100% of 24 tasks positive
- Test C per-step (step 2 vs step 5): invariant
- Test D silhouette ≥ 0.5 at L23 (strong clustering)
- Test E bootstrap 95% CI tight (4-15% of mean)

### 3.2 Method 4.4 — mean-diff activation steering (v2 in flight)

`scripts/mechanistic/run_stage4_method44_v2_sweep.py`. Layer × α sweep:
- Layers: [11, 17, 23, 29, 33, 34] — covers mid (Stage 2 disruption locus) → late (Wu et al. second-to-last)
- α: [1, 2, 5, 10, 20] — Wu et al. typical α=1, our diag found ≥5 needed for multi-step JSON
- 24 cls strong-tier tasks × 2 steps × 30 cells = 1440 generations (~2h)

**HDMI reliability metric**: completeness × selectivity → harmonic mean (Khorasani et al. 2026):
- Completeness = % tasks where overlap_psom > overlap_dom
- Selectivity = % tasks where JSON envelope preserved (starts with `{`)
- Reliability = 2 · c · s / (c + s)

**Current smoke (8/48 cells)**: L17 α=5 = **0.44** sweet spot (29% shift + 100% JSON valid). L33 α=10 = 0.23 (57% shift but JSON breaks).

### 3.3 Method 4.5 — LA-HDMI / SAE (future work, paper §8)

Two alternative paths:
- **LA-HDMI**: probe-free gradient steering (Khorasani 2026 method). Per-input optimization replaces fixed mean-diff direction. May overcome Qwen3-family causal patching weakness
- **SAE feature steering** (Zekun-recommended in advisor recording, paper_planning §108): train SAE on Qwen3-VL-4B residual stream (1-2 week cost, no public SAE exists), find mirage/format feature, steer directly. Differentiates from Wu et al. mean-diff path

Decision pending Method 4.4 v2 full sweep + Zekun sync.

## 4. Identification protocol (Lin & Liu 2026 disclosure norm)

Following Lin & Liu Position paper, paper §5 must explicitly state:

### 4.1 Causal claim (revised after Stage 4 v2 NPZ migration 2026-05-13)

> The patch-sensitive continuation window L11-L17 (block-output index convention) at the last-input-token position is causally consequential for phantom routing space mode selection in Qwen3-VL-4B web agents, under final-token-replacement activation patching. Mode-distinguishing signal is linearly readable in residual stream throughout (AUROC 1.000 lototask cross-site), with sub-permille cosine geometry for prompt-family / text-format axes (peak L36 monotone-to-boundary, magnitude ≤ 0.009) and ~5-10× larger image-axis geometry. The lm_head decoding amplifies sub-permille cosine into measurable KL (~0.05-0.09 at L21-L25 window, 7-10× amplification), and final-token patching at L11-L17 produces 20-30% Δoverlap-to-target. Residual-stream cosine geometry severely underestimates causal influence; signature layer ≠ decision layer ≠ amplification layer (mechanistic-interpretability standard finding cf. Wang et al. 2023 IOI).

Three stale framings retracted with v2 evidence:
- ❌ "L17 singular planning site" → corrected to **L11-L17 window** (Stage 2/3 6/6 cells)
- ❌ "Three-axis hierarchy 4:3:1 magnitude ratio" → corrected to **image-dominant ~5-10× + axis-1 ≤ axis-2 both sub-permille** (v2 NPZ §1.2/§5.1/§7.3.0)
- ❌ "L17 α=5 H-mean 0.44 sweet spot" (Method 4.4 smoke) → corrected to **L33 α=10 H-mean 0.33** (full 45-cell sweep, probe-causal dissociation §5.3)

New core paper §5 hero claim: **cosine-causal disjoint** — three converging numbers (cosine 0.5-1% / KL 5-9% / patching Δoverlap 20-30%) anchor "geometry underestimates causal" framing. Not a single layer / single magnitude claim.

### 4.2 Identification strategy

Triangulation of 3 evidence types:
1. **Probe-level** (Method 4.2 PCA cosine gap; AUROC reported as both `in_sample` and held-out `leave-one-task-out` after 2026-05-12 Bug 3 fix; v1 buggy NPZ data invalidated, v2 NPZ in flight: Myriad 359736 cls + 359737 reddit)
2. **Replacement patching** (Stage 2/3 Cell A-H, L11-L17 window disruption, Holm-significant per layer; baseline empirically equals unpatched at L35 final-block patching position since overlap→target ≈ 1.00 at L35 across all forward cells)
3. **Additive steering** (Method 4.4 v2 full sweep 45 cells: layer-α tradeoff; mid-layer L11-L17 preserves JSON envelope but low completeness, late-layer L33 produces largest output shifts but over-steers — H-mean ceiling 0.33 indicates probe-causal dissociation, not a single sweet-spot validation)

### 4.3 Identification assumptions

| # | Assumption | Stress-test |
|---|---|---|
| A1 | L17 last-token hidden state mediates action selection (not earlier obs token positions) | Stage 2/3 swept all layers, L17 is peak |
| A2 | Mean-difference direction approximates causal axis (Wu et al. hypothesis) | Method 4.4 v2 H-mean 0.44 partial — assumption holds weakly; LA-HDMI would test |
| A3 | 24 strong-tier tasks generalize to broader VWA distribution | Stage 4 robustness Test B: 100% per-task positive, but tier-selection bias possible. Reverse-tier 15 tasks pending |
| A4 | Qwen3-VL-4B mechanism transfers to other VLM sizes/architectures | Not tested. Wu et al. shows family generality on tool-only; multimodal+multi-step unknown |
| A5 | Replacement patching faithfully simulates "natural" model read of the representation | Cell E random-injection control rules out non-specific disruption — content-specific causation confirmed |

### 4.4 Stress-test result

Cell E random-injection control: replacing source hidden with Gaussian noise (same μ, σ) yields **null L17 disruption effect**. Confirms our patching effect is source-content-specific, not noise-driven. Most directly stresses A5.

### 4.5 Validation ≠ identification (Lin & Liu §5)

- Method 4.2 AUROC 1.000 = validation (decodability)
- Stage 2/3 + Method 4.4 v2 = identification attempts (causal use)
- These are reported SEPARATELY in paper §5; reviewer should not conflate

exec
/bin/bash -lc "sed -n '1,260p' docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
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

exec
/bin/bash -lc "sed -n '1,190p' scripts/analysis/stage4_logit_lens_axis2.py" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
#!/usr/bin/env python3
"""Exp 3: Logit lens at late layers — does axis-2 prompt-family signal
re-emerge in output distribution even though mid-layer residual stream
shows only weak (~0.011) signal at L23?

Method: Apply Qwen3-VL-4B's lm_head + final_norm to each per-layer hidden
state mean, get a token distribution per (mode, layer). For each axis-2
pair (P-text vs P-SoM at same task) compute:
  - top-1 token disagreement rate per layer
  - KL divergence (P-text || P-SoM) per layer
  - log-prob gap on canonical SoM-prompt vs DOM-prompt action tokens
    (e.g., "click" vs "search", "_pick_", json keys)

This is Wu et al. tool-calling "knows but says differently" mirror: if
axis-2 cosine gap is 0.011 at L23 but output KL is large at L30-L36,
prompt prior is amplified by late-layer decoding into different output.

Inputs:
  results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
  results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz

Outputs:
  docs/checkpoints/mechanism/results/axis2_logit_lens.md
  results/phantom_paper/figures/fig_axis2_logit_lens.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLS_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz"
DEFAULT_RED_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_reddit/hidden_states_v2_fixed.npz"
DEFAULT_MD = ROOT / "docs/checkpoints/mechanism/results/axis2_logit_lens.md"
DEFAULT_FIG = ROOT / "results/phantom_paper/figures/fig_axis2_logit_lens.png"
MODEL_PATH = "Qwen/Qwen3-VL-4B-Instruct"
# Bug 5 fix (/codex-stress methodology audit 2026-05-12): pin HF revision
# to match HiddenStateExtractor + Stage 2B / Stage 4 v2 extraction. Previously
# unpinned, so logit lens KL applied `norm + lm_head` from an arbitrary cached
# revision to hidden states extracted under a pinned revision — making KL
# magnitudes non-reproducible across machines or cache states.
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"

AXIS_2_PAIRS = [
    ("phantom_text", "phantom_som", "P-text vs P-SoM  (axis-2 flat-text)"),
    ("dom",          "phantom_prompt", "DOM vs P-prompt  (axis-2 hierarchical)"),
]
AXIS_1_PAIRS = [
    ("dom",           "phantom_text",   "DOM vs P-text    (axis-1 DOM-prompt)"),
    ("phantom_prompt","phantom_som",    "P-prompt vs P-SoM (axis-1 SoM-prompt)"),
]


def load_lm_head_and_norm(device="cuda"):
    """Load Qwen3-VL-4B lm_head + final_norm from HF cache (offline)."""
    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, trust_remote_code=True
    )
    print(f"  loading Qwen3VLForConditionalGeneration (lm_head + norm only, revision={MODEL_REVISION[:12]}...)")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, revision=MODEL_REVISION, dtype=torch.bfloat16,
        device_map=device, trust_remote_code=True,
    )
    # Qwen3-VL structure (verified via p79/mechanistic/activation_patching.py):
    #   model.model.language_model.layers  (36 decoder layers, no embedding included)
    #   model.model.language_model.norm    (final RMSNorm, sibling of layers)
    #   model.lm_head                       (top-level projection)
    norm = model.model.language_model.norm
    lm_head = model.lm_head
    print(f"  norm: {type(norm).__name__}, lm_head: {type(lm_head).__name__}")
    return tokenizer, lm_head, norm, model


@torch.no_grad()
def logits_at_layer(hidden: torch.Tensor, lm_head, norm) -> torch.Tensor:
    """hidden: (D,) → logits (V,) after final_norm + lm_head."""
    h = hidden.unsqueeze(0).to(lm_head.weight.device).to(lm_head.weight.dtype)
    h = norm(h)
    logits = lm_head(h).squeeze(0)
    return logits


def kl_divergence(p_logits, q_logits) -> float:
    """KL(P || Q) with softmax on logits."""
    log_p = torch.log_softmax(p_logits, dim=-1)
    log_q = torch.log_softmax(q_logits, dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum().item()
    return kl


def top1_agree(p_logits, q_logits) -> bool:
    return torch.argmax(p_logits).item() == torch.argmax(q_logits).item()


def compute_pair_logit_lens(npz: Path, pair_pairs: list, lm_head, norm, n_layers_use: int):
    d = np.load(npz, allow_pickle=True)
    H = d["hidden_states"]  # (N, L, D)
    ml = d["mode_labels_str"]
    means = {}
    for m in {p[0] for p in pair_pairs} | {p[1] for p in pair_pairs}:
        mask = ml == m
        if mask.sum() == 0:
            continue
        means[m] = H[mask].mean(axis=0)

    result = {}
    for a, b, label in pair_pairs:
        if a not in means or b not in means:
            continue
        layer_kl = []
        layer_disagree = []
        for L in range(n_layers_use):
            h_a = torch.tensor(means[a][L])
            h_b = torch.tensor(means[b][L])
            l_a = logits_at_layer(h_a, lm_head, norm)
            l_b = logits_at_layer(h_b, lm_head, norm)
            layer_kl.append(kl_divergence(l_a, l_b))
            layer_disagree.append(0.0 if top1_agree(l_a, l_b) else 1.0)
        result[label] = {
            "kl": np.array(layer_kl),
            "disagree": np.array(layer_disagree),
            "mode_a": a, "mode_b": b,
            "peak_kl_L": int(np.argmax(layer_kl)),
            "peak_kl": float(np.max(layer_kl)),
        }
    return result, n_layers_use


def write_md(cls_axis2, cls_axis1, red_axis2, red_axis1, n_layers, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)",
        "",
        "Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.",
        "For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement",
        "across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets",
        "amplified into output distribution divergence by late-layer decoding.",
        "",
        "## Classifieds site",
        "",
        "### Axis-2 (prompt-family) pairs:",
        "",
        "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, info in cls_axis2.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )
    lines += ["", "### Axis-1 (text-format) pairs:", "",
              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, info in cls_axis1.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )

    lines += ["", "## Reddit site", "",
              "### Axis-2 (prompt-family) pairs:", "",
              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, info in red_axis2.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )
    lines += ["", "### Axis-1 (text-format) pairs:", "",
              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, info in red_axis1.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )

exec
/bin/bash -lc "sed -n '1,210p' scripts/analysis/stage4_pca_cosine_gap.py" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
#!/usr/bin/env python3
"""Stage 4 Method 4.2: PCA cosine gap analysis of phantom routing space.

Ports Tool Calling Linear Steerable Circuit method (Anonymous 2026 ACL, validated
on Qwen3-4B) to Qwen3-VL-4B (B1). Tests whether phantom routing space modes are
mechanistically distinct in hidden state geometry layer-by-layer.

Three analyses per (mode pair, layer):
  A. Cosine gap between mean hidden states
  B. AUROC: project hidden states onto (mean_A - mean_B) direction, predict mode
  C. Per-(mode, layer) PCA top-10 variance explained

Outputs:
  - results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json
  - docs/checkpoints/stage4_method42_results.md
  - results/phantom_paper/figures/fig_stage4_pca_cosine_gap.png
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NPZ = ROOT / "results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz"
DEFAULT_OUT_JSON = ROOT / "results/mechanistic/stage4_multimode_b1_cls/method42_metrics.json"
DEFAULT_OUT_MD = ROOT / "docs/checkpoints/stage4_method42_results.md"
DEFAULT_OUT_FIG = ROOT / "results/phantom_paper/figures/fig_stage4_pca_cosine_gap.png"

MODES = ["dom", "phantom_text", "phantom_prompt", "phantom_som", "som", "vision"]
DISPLAY = {"dom": "DOM", "phantom_text": "P-text", "phantom_prompt": "P-prompt",
           "phantom_som": "P-SoM", "som": "SoM", "vision": "Vision"}


def cosine_gap(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(1.0 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))


def pair_key(a: str, b: str) -> str:
    """Canonical pair key using MODES index order (matches itertools.combinations output)."""
    i, j = MODES.index(a), MODES.index(b)
    return f"{MODES[min(i, j)]}_vs_{MODES[max(i, j)]}"


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument("--output-fig", type=Path, default=DEFAULT_OUT_FIG)
    args = parser.parse_args()
    NPZ = args.input
    OUT_JSON = args.output_json
    OUT_MD = args.output_md
    OUT_FIG = args.output_fig

    d = np.load(NPZ, allow_pickle=True)
    H = d["hidden_states"]
    mode_labels = d["mode_labels_str"]
    task_ids = d["task_ids"] if "task_ids" in d.files else None
    n_layers = H.shape[1]
    print(f"[stage4] loaded {H.shape[0]} examples × {n_layers} layers × {H.shape[2]} dim")

    states = {m: H[mode_labels == m] for m in MODES}
    means = {m: states[m].mean(axis=0) for m in MODES}  # each (37, 2560)
    print(f"[stage4] per-mode counts: " + ", ".join(f"{m}={len(states[m])}" for m in MODES))

    # Per-mode task_id mapping for leave-one-task-out (Bug 3 fix, codex
    # methodology audit 2026-05-12: previous AUROC fit direction on the
    # same examples used to evaluate → inflated, not held-out decodability).
    mode_task_ids = {m: task_ids[mode_labels == m] if task_ids is not None else None
                     for m in MODES}

    pairs = list(combinations(MODES, 2))
    cos_gap = np.zeros((len(pairs), n_layers))
    auroc_in_sample = np.zeros((len(pairs), n_layers))
    auroc_lototask = np.zeros((len(pairs), n_layers))  # leave-one-task-out CV
    for pi, (m1, m2) in enumerate(pairs):
        for L in range(n_layers):
            c1, c2 = means[m1][L], means[m2][L]
            cos_gap[pi, L] = cosine_gap(c1, c2)
            direction = (c1 - c2) / (np.linalg.norm(c1 - c2) + 1e-9)
            s1 = states[m1][:, L, :] @ direction
            s2 = states[m2][:, L, :] @ direction
            y = np.concatenate([np.ones(len(s1)), np.zeros(len(s2))])
            scores = np.concatenate([s1, s2])
            try:
                auroc_in_sample[pi, L] = roc_auc_score(y, scores)
            except Exception:
                auroc_in_sample[pi, L] = 0.5

            # Leave-one-task-out CV — only when task_ids are available
            tids_m1 = mode_task_ids[m1]
            tids_m2 = mode_task_ids[m2]
            if tids_m1 is None or tids_m2 is None:
                auroc_lototask[pi, L] = np.nan
                continue
            # Tasks that appear in BOTH modes (paper-grade design has all
            # tasks in all modes, so this is usually all 24)
            common_tasks = sorted(set(tids_m1.tolist()) & set(tids_m2.tolist()))
            if len(common_tasks) < 3:
                auroc_lototask[pi, L] = np.nan
                continue
            fold_aurocs = []
            for held_out_tid in common_tasks:
                # Train: all examples whose task_id != held_out_tid
                train_mask_m1 = tids_m1 != held_out_tid
                train_mask_m2 = tids_m2 != held_out_tid
                test_mask_m1 = tids_m1 == held_out_tid
                test_mask_m2 = tids_m2 == held_out_tid
                if (train_mask_m1.sum() == 0 or train_mask_m2.sum() == 0 or
                        test_mask_m1.sum() == 0 or test_mask_m2.sum() == 0):
                    continue
                train_c1 = states[m1][train_mask_m1, L, :].mean(0)
                train_c2 = states[m2][train_mask_m2, L, :].mean(0)
                train_dir = (train_c1 - train_c2) / (np.linalg.norm(train_c1 - train_c2) + 1e-9)
                test_s1 = states[m1][test_mask_m1, L, :] @ train_dir
                test_s2 = states[m2][test_mask_m2, L, :] @ train_dir
                test_y = np.concatenate([np.ones(len(test_s1)), np.zeros(len(test_s2))])
                test_scores = np.concatenate([test_s1, test_s2])
                if len(np.unique(test_y)) < 2:
                    continue
                try:
                    fold_aurocs.append(roc_auc_score(test_y, test_scores))
                except Exception:
                    pass
            auroc_lototask[pi, L] = float(np.mean(fold_aurocs)) if fold_aurocs else np.nan

    pca_var = np.zeros((len(MODES), n_layers))
    for mi, mode in enumerate(MODES):
        X = states[mode]  # (n, 37, 2560)
        for L in range(n_layers):
            if X.shape[0] >= 11:
                n_comp = min(10, X.shape[0] - 1)
                pca_var[mi, L] = PCA(n_components=n_comp).fit(X[:, L, :]).explained_variance_ratio_.sum()

    peak = {}
    for pi, (m1, m2) in enumerate(pairs):
        L = int(np.argmax(cos_gap[pi]))
        peak[f"{m1}_vs_{m2}"] = {
            "layer": L,
            "gap": float(cos_gap[pi, L]),
            "auroc_in_sample_at_peak": float(auroc_in_sample[pi, L]),
            "auroc_lototask_at_peak": (
                float(auroc_lototask[pi, L])
                if not np.isnan(auroc_lototask[pi, L]) else None
            ),
        }

    # Replace NaN with None for JSON serializability
    def _nan_to_none(arr):
        return [None if np.isnan(x) else float(x) for x in arr]

    metrics = {
        "n_examples": int(H.shape[0]), "n_layers": int(n_layers), "n_modes": len(MODES),
        "modes": MODES, "n_per_mode": {m: int(len(states[m])) for m in MODES},
        "pairwise_cosine_gap": {f"{m1}_vs_{m2}": cos_gap[pi].tolist()
                                  for pi, (m1, m2) in enumerate(pairs)},
        "pairwise_auroc_in_sample": {f"{m1}_vs_{m2}": auroc_in_sample[pi].tolist()
                                       for pi, (m1, m2) in enumerate(pairs)},
        "pairwise_auroc_lototask": {f"{m1}_vs_{m2}": _nan_to_none(auroc_lototask[pi])
                                      for pi, (m1, m2) in enumerate(pairs)},
        "pca_top10_var_ratio": {m: pca_var[mi].tolist() for mi, m in enumerate(MODES)},
        "peak_disruption_layers": peak,
        "auroc_protocol_note": (
            "auroc_in_sample fits mode-mean direction on all examples and scores those "
            "same examples (inflated, NOT held-out decodability). auroc_lototask is "
            "leave-one-task-out cross-validation: for each held-out task, fit direction "
            "on the remaining tasks' means, then score the held-out task's examples. "
            "Report lototask as the paper-grade linear-readability metric; in-sample is "
            "kept for descriptive comparison only. Bug 3 fix per codex methodology audit "
            "2026-05-12."
        ),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"[stage4] metrics → {OUT_JSON}")

    write_summary(metrics, OUT_MD)
    plot(cos_gap, auroc_lototask, pairs, pca_var, OUT_FIG)


def write_summary(m: dict, out: Path) -> None:
    sorted_pairs = sorted(m["peak_disruption_layers"].items(),
                           key=lambda x: -x[1]["gap"])
    lines = [
        "# Stage 4 Method 4.2: PCA Cosine Gap Analysis",
        "",
        f"**Data**: {m['n_examples']} examples × {m['n_layers']} layers × {m['n_modes']} modes (Qwen3-VL-4B B1 cls)",
        f"**Per-mode n**: " + ", ".join(f"{DISPLAY[k]}={v}" for k, v in m['n_per_mode'].items()),
        "",
        "**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade "
        "metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean "
        "direction on training tasks, score held-out task). `auroc_in_sample` (fit + score "
        "on same examples) is reported for descriptive comparison only; treat any in-sample "
        "≥0.95 as expected algebraic separability, NOT held-out linear-readability.",
        "",
        "## Peak disruption layer per mode pair",
        "",
        "Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):",
        "",
        "| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |",
        "|---|---|---|---|---|",
    ]

codex
The first pass already shows a few review-level red flags: the retraction text is more honest, but some old artifacts are still embedded in script docstrings and method labels. I’m going to locate the later plan sections now, because §5.1 and §7.3.0a likely contain the paper-facing risk.
exec
/bin/bash -lc 'rg -n "''^## 5|''^### 5'"\\.1|"'^## 7|''^### 7'"\\.3\\.0a|7\\.3\\.0a|5\\.1\" docs/checkpoints/mechanism/plan.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
190:- ❌ "Three-axis hierarchy 4:3:1 magnitude ratio" → corrected to **image-dominant ~5-10× + axis-1 ≤ axis-2 both sub-permille** (v2 NPZ §1.2/§5.1/§7.3.0)
222:## 5. Current findings dashboard
224:### 5.1 Stage 4 Method 4.2 v2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers, 2026-05-13 canonical)
329:## 7. Advisor sync state — Zekun (Wu et al. 2026 first author = lab member)
343:Updated after v2 full sweep + H1 test. Key revisions from §125.10 draft:
468:### 7.3.0a Exp 3 logit lens v2 — output-layer amplification (2026-05-13 01:55)

exec
/bin/bash -lc "sed -n '180,360p' scripts/analysis/stage4_pca_cosine_gap.py" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(metrics, indent=2))
    print(f"[stage4] metrics → {OUT_JSON}")

    write_summary(metrics, OUT_MD)
    plot(cos_gap, auroc_lototask, pairs, pca_var, OUT_FIG)


def write_summary(m: dict, out: Path) -> None:
    sorted_pairs = sorted(m["peak_disruption_layers"].items(),
                           key=lambda x: -x[1]["gap"])
    lines = [
        "# Stage 4 Method 4.2: PCA Cosine Gap Analysis",
        "",
        f"**Data**: {m['n_examples']} examples × {m['n_layers']} layers × {m['n_modes']} modes (Qwen3-VL-4B B1 cls)",
        f"**Per-mode n**: " + ", ".join(f"{DISPLAY[k]}={v}" for k, v in m['n_per_mode'].items()),
        "",
        "**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade "
        "metric is `auroc_lototask` = leave-one-task-out cross-validation (fit mode-mean "
        "direction on training tasks, score held-out task). `auroc_in_sample` (fit + score "
        "on same examples) is reported for descriptive comparison only; treat any in-sample "
        "≥0.95 as expected algebraic separability, NOT held-out linear-readability.",
        "",
        "## Peak disruption layer per mode pair",
        "",
        "Sorted by cosine gap magnitude (= geometric distance between mode means in hidden space):",
        "",
        "| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |",
        "|---|---|---|---|---|",
    ]
    for k, v in sorted_pairs:
        m1, m2 = k.split("_vs_")
        lototask_val = v.get("auroc_lototask_at_peak")
        lototask_str = f"{lototask_val:.3f}" if lototask_val is not None else "n/a"
        lines.append(
            f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | L{v['layer']:02d} | {v['gap']:.4f} | "
            f"{v['auroc_in_sample_at_peak']:.3f} | {lototask_str} |"
        )

    # Mid-layer (L17) snapshot — paper §5 disruption locus
    L17_section = ["", "## L17 cosine gap snapshot (paper §5 disruption locus)", ""]
    L17_section.append("| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |")
    L17_section.append("|---|---|---|---|")
    pairs = list(combinations(MODES, 2))
    for pi, (m1, m2) in enumerate(pairs):
        gap = m["pairwise_cosine_gap"][f"{m1}_vs_{m2}"][17]
        a_in = m["pairwise_auroc_in_sample"][f"{m1}_vs_{m2}"][17]
        a_lo = m["pairwise_auroc_lototask"][f"{m1}_vs_{m2}"][17]
        a_lo_str = f"{a_lo:.3f}" if a_lo is not None else "n/a"
        L17_section.append(f"| {DISPLAY[m1]} vs {DISPLAY[m2]} | {gap:.4f} | {a_in:.3f} | {a_lo_str} |")
    lines.extend(L17_section)

    # Phantom-arm specific anchor — P-SoM cosine to each baseline mode at L17
    psom_section = ["", "## P-SoM vs baseline modes (paper §5 HERO arm)", "",
                     "P-SoM identity test: is P-SoM closer to SoM (prompt-axis sibling) or DOM (text-axis sibling)?",
                     ""]
    psom_section.append("| L | P-SoM↔DOM | P-SoM↔SoM | P-SoM↔Vision | P-SoM↔P-text | P-SoM↔P-prompt |")
    psom_section.append("|---|---|---|---|---|---|")
    for L in [0, 8, 11, 17, 24, 30, 36]:
        row = [f"L{L:02d}"]
        for other in ["dom", "som", "vision", "phantom_text", "phantom_prompt"]:
            key = pair_key("phantom_som", other)
            row.append(f"{m['pairwise_cosine_gap'][key][L]:.4f}")
        psom_section.append("| " + " | ".join(row) + " |")
    lines.extend(psom_section)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"[stage4] summary → {out}")


def plot(cos_gap, auroc, pairs, pca_var, out):
    plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    pair_labels = [f"{DISPLAY[m1]}↔{DISPLAY[m2]}" for m1, m2 in pairs]

    ax = axes[0, 0]
    im = ax.imshow(cos_gap, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pair_labels, fontsize=7)
    ax.set_xlabel("Layer index")
    ax.set_title("(a) Pairwise cosine gap (geometric distance between mode means)")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(auroc, aspect="auto", cmap="RdYlGn", vmin=0.5, vmax=1.0)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(pair_labels, fontsize=7)
    ax.set_xlabel("Layer index")
    ax.set_title("(b) Pairwise AUROC (project onto Δ-mean direction, classify)")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    for mi, mode in enumerate(MODES):
        ax.plot(pca_var[mi], label=DISPLAY[mode], linewidth=1.5)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Top-10 PCA cumulative variance explained")
    ax.set_title("(c) Per-mode within-cluster dimensionality")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    psom_idx = {f"{m1}_vs_{m2}": i for i, (m1, m2) in enumerate(pairs)}
    for other in ["dom", "som", "vision", "phantom_text", "phantom_prompt"]:
        key = pair_key("phantom_som", other)
        ax.plot(cos_gap[psom_idx[key]], label=f"P-SoM ↔ {DISPLAY[other]}", linewidth=1.5)
    ax.axvline(17, color="red", linestyle=":", alpha=0.5, label="L17 (Stage 2 disruption locus)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Cosine gap to P-SoM")
    ax.set_title("(d) P-SoM identity — closest sibling per layer")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle("Stage 4 Method 4.2: Phantom routing space hidden state geometry (Qwen3-VL-4B B1 cls)",
                  fontsize=12, fontweight="bold")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[stage4] figure → {out}")


if __name__ == "__main__":
    main()

exec
/bin/bash -lc "sed -n '150,330p' scripts/analysis/stage4_logit_lens_axis2.py" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
        "",
        "### Axis-2 (prompt-family) pairs:",
        "",
        "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, info in cls_axis2.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )
    lines += ["", "### Axis-1 (text-format) pairs:", "",
              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, info in cls_axis1.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )

    lines += ["", "## Reddit site", "",
              "### Axis-2 (prompt-family) pairs:", "",
              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, info in red_axis2.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )
    lines += ["", "### Axis-1 (text-format) pairs:", "",
              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, info in red_axis1.items():
        kl = info["kl"]
        lines.append(
            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        "Three hypotheses tested:",
        "",
        "- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family",
        "  effect bypasses logit lens, only visible via attention heads or runtime decoding.",
        "- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →",
        "  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling",
        "  'knows but says differently' mirror).",
        "- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →",
        "  prompt prior signal proportional to mid-layer geometry, no amplification.",
        "",
        "Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to",
        "axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.",
    ]
    out.write_text("\n".join(lines) + "\n")
    print(f"summary → {out}")


def plot(cls_a2, cls_a1, red_a2, red_a1, n_layers, out: Path):
    plt.rcParams.update({"font.size": 10, "figure.dpi": 150})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    layers = np.arange(n_layers)
    for ax, a2, a1, site in [(axes[0], cls_a2, cls_a1, "classifieds"),
                              (axes[1], red_a2, red_a1, "reddit")]:
        for label, info in a2.items():
            ax.plot(layers, info["kl"], color="#d62728" if "DOM" in label else "#ff7f0e",
                    linewidth=2.5, label=label)
        for label, info in a1.items():
            ax.plot(layers, info["kl"], color="#1f77b4" if "DOM" in label else "#2ca02c",
                    linestyle="--", linewidth=1.5, alpha=0.7, label=label)
        ax.axvline(17, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.axvline(23, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Layer index")
        ax.set_title(f"{site}  (axis-2 solid, axis-1 dashed)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7, loc="upper left")
    axes[0].set_ylabel("KL divergence (logit lens)")
    fig.suptitle("Exp 3: Output distribution divergence per layer via logit lens", fontsize=11)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"figure → {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cls-npz", type=Path, default=DEFAULT_CLS_NPZ)
    p.add_argument("--red-npz", type=Path, default=DEFAULT_RED_NPZ)
    p.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    p.add_argument("--output-fig", type=Path, default=DEFAULT_FIG)
    args = p.parse_args()

    print("Loading model...")
    tokenizer, lm_head, norm, _ = load_lm_head_and_norm(device="cuda")

    print("\n[cls] axis-2 pairs:")
    cls_a2, n_L = compute_pair_logit_lens(args.cls_npz, AXIS_2_PAIRS, lm_head, norm, 37)
    print(f"  done, {len(cls_a2)} pairs")

    print("[cls] axis-1 pairs:")
    cls_a1, _ = compute_pair_logit_lens(args.cls_npz, AXIS_1_PAIRS, lm_head, norm, 37)
    print(f"  done, {len(cls_a1)} pairs")

    print("[reddit] axis-2 pairs:")
    red_a2, _ = compute_pair_logit_lens(args.red_npz, AXIS_2_PAIRS, lm_head, norm, 37)
    print("[reddit] axis-1 pairs:")
    red_a1, _ = compute_pair_logit_lens(args.red_npz, AXIS_1_PAIRS, lm_head, norm, 37)

    write_md(cls_a2, cls_a1, red_a2, red_a1, n_L, args.output_md)
    plot(cls_a2, cls_a1, red_a2, red_a1, n_L, args.output_fig)


if __name__ == "__main__":
    main()

exec
/bin/bash -lc "python - <<'PY'
import numpy as np, pathlib
for p in ['results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz','results/mechanistic/stage4_multimode_b1_reddit/hidden_states_v2_fixed.npz']:
    path=pathlib.Path(p)
    print(p, path.exists())
    if path.exists():
        d=np.load(path, allow_pickle=True)
        print(d.files)
        for k in ['hidden_states','mode_labels_str','task_ids','step_ids']:
            if k in d.files:
                arr=d[k]
                print(k, arr.shape, arr.dtype, arr[:5] if arr.ndim==1 else '')
        if 'mode_labels_str' in d.files:
            vals, counts=np.unique(d['mode_labels_str'], return_counts=True)
            print(dict(zip(vals.tolist(), counts.tolist())))
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 exited 127 in 0ms:
/bin/bash: line 1: python: command not found

exec
/bin/bash -lc "sed -n '222,310p' docs/checkpoints/mechanism/plan.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
## 5. Current findings dashboard

### 5.1 Stage 4 Method 4.2 v2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers, 2026-05-13 canonical)

V2 NPZ-corrected (Bug 1+2+5 fix, see §0). All AUROC lototask = 1.000 (held-out per-task fold) — modes linearly separable in residual stream. Cosine magnitudes:

| Pair | L17 cos gap (v2) | Peak L / gap (v2) | AUROC | Axis |
|---|---:|---:|---:|---|
| P-SoM ↔ P-text | 0.0019 | L36 0.009 | 1.000 | axis-2 prompt-family (flat) |
| DOM ↔ P-prompt | 0.0015 | L36 0.007 | 1.000 | axis-2 prompt-family (hier) |
| DOM ↔ P-text | 0.0021 | L36 0.005 | 1.000 | axis-1 text-format (DOM-prompt) |
| P-prompt ↔ P-SoM | 0.0017 | L36 0.005 | 1.000 | axis-1 text-format (SoM-prompt) |
| P-SoM ↔ SoM | 0.0029 | L36 0.042 | 1.000 | image-axis (annotated) |
| DOM ↔ Vision | 0.0571 | L04 0.067 | 1.000 | image-axis (naked) |

V1 (buggy) reference for diff: P-SoM↔P-text was L23 0.011 (now collapsed -23% to L36 0.009), DOM↔P-text was L23 0.025 (now collapsed -81% to L36 0.005). Image-axis pairs preserve magnitude (v1 0.041 vs v2 0.042 at peak). See `method42_v1_vs_v2_comparison.md` for full 15-pair diff.

Reddit cross-site v2 replicates: image-axis pairs preserve magnitude (~0.04-0.07 cross-pair); axis-1 + axis-2 magnitudes both sub-permille (0.002-0.009 L17, monotone-to-boundary L36 0.005-0.009). Rank-order axis-1 ≤ axis-2 holds cross-site.

### 5.2 Stage 2/3 patching disruption (14 cells, B1 cls + reddit)

**Stage 2 — P-SoM ↔ SoM patching (10 cells):**

| Cell | Site | Direction | L17 Δoverlap | Holm-sig |
|---|---|---|---|---|
| A | cls | SoM→P-SoM forward | -0.32 | ✓ |
| B | cls | P-SoM→SoM reverse | -0.16 | ✓ |
| C | cls | 2x2 reverse-tier fwd | -0.02 | ✗ (null) |
| D | cls | 2x2 strong-tier rev | -0.18 | ✓ |
| E | cls | random injection | -0.03 (uniform) | ✓ (negative control) |
| F | reddit | SoM→P-SoM forward | -0.21 | ✓ |
| G | reddit | P-SoM→SoM reverse | -0.18 | ✓ |
| Cr/Dr | reddit 2x2 | both directions | -0.15 to -0.18 | ✓ |
| Er | reddit | random injection | ~0 (uniform) | ✓ |

**Stage 3 — 2x2 mechanism additivity test (SoM → {DOM, P-text, P-prompt}, cls + reddit):**

| Cell | Site | Source→Target | Best-L overlap→src | L17 Δoverlap→tgt | Path |
|---|---|---|---|---|---|
| H-d-cls | cls | SoM → DOM | L10 (0.192) | -0.33 | `stage3_cellhd_cls_fwd_dom_myriad/` |
| H-p-cls | cls | SoM → P-prompt | L27 (0.219) | -0.22 | `stage3_cellhp_cls_fwd_prompt_myriad/` |
| H-t-cls | cls | SoM → P-text | L28 (0.164) | -0.25 | `stage3_cellht_cls_fwd_text_myriad/` |
| H-p-red | reddit | SoM → P-prompt | L20 (0.209) | -0.19 | `stage3_cellhp_red_fwd_prompt_myriad/` |
| H-t-red | reddit | SoM → P-text | L01 (0.194) | -0.24 | `stage3_cellht_red_fwd_text_myriad/` |
| **H-d-red** | reddit | SoM → DOM | L28 (0.204) | **L11 -0.33 / L17 -0.26** | `stage3_cellhd_red_fwd_dom_myriad/` ✅ done 2026-05-12 19:57 |

**Stage 3 interpretation (6/6 cells complete 2026-05-12)**: All forward SoM→{no-image-arm} patching cells show mid-layer L11-L17 disruption -0.19 to -0.33 Δoverlap→tgt. Magnitude > random injection control (Cell E -0.03) at all 6. **Mechanism additivity confirmed**: image-feature axis is shared substrate across DOM / P-text / P-prompt arms — single SoM→{any-no-image-arm} patching displaces target prediction toward source. Cross-site cls + reddit both replicate (paper §5 universal mid-layer fusion locus); reddit fusion locus slightly earlier (L11 vs cls L17), magnitude identical.

Stage 3 cross-site DOM-axis additivity table (paired-test Δoverlap-to-target from `patching_continuation_results.json`):

| Site | SoM→DOM | SoM→P-text | SoM→P-prompt | best-L Δ range |
|---|---|---|---|---|
| cls | H-d-cls L17 -0.309 / L18 **-0.352** best | H-t-cls L17 -0.255 / L12 **-0.270** best | H-p-cls L17 -0.223 / L13 **-0.273** best | [-0.273, -0.352] |
| reddit | H-d-red L11 -0.335 / L17 -0.255 / L14 **-0.338** best | H-t-red L11 -0.244 / L17 -0.236 / L15 **-0.330** best | H-p-red L11 -0.233 / L17 -0.191 / L14 **-0.322** best | [-0.322, -0.338] |

All 6 cells best layer 落在 **L12-L18 mid-layer 窗口** (tight 7-layer band), Δ range [-0.27, -0.35]. Cross-site / cross-arm 一致, mid-layer fusion locus 不是 single layer index 而是稳定窗口.

### 5.3 Stage 4 Method 4.4 v2 (FULL 45/48 cells, finalized 2026-05-11 22:00)

H-mean reliability (HDMI framework) per (layer, α). **L17 α=5 smoke claim REFUTED by full sweep**; actual sweet spot at L33 α=10:

| Layer \ α | α=1 | α=2 | α=5 | α=10 | α=20 |
|---|---|---|---|---|---|
| L11 | 0.04 | 0.09 | 0.20 | 0.12 | 0.12 |
| L17 | 0.00 | 0.12 | **0.16** (was 0.44 smoke) | 0.12 | 0.09 |
| L23 | 0.00 | 0.09 | 0.09 | 0.16 | 0.00 |
| L29 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 |
| **L33** | 0.04 | 0.00 | 0.00 | **0.33** ⭐ | 0.00 |
| L34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

**Layer-specialization** (probe-causal dissociation):
- Mid-layer (L11-L23): **selectivity 100%** at all α (JSON envelope preserved), but completeness 0-11% (modest shift)
- Late-layer (L33): completeness 38% (highest), but selectivity drops to 29% (over-steers JSON)
- L33 α=10 H-mean 0.33 = max reliability cell

**Smoke variance lesson** (笔记 §126 + §127): 4-cell smoke H-mean 0.44 on L17 was statistical artifact (1/4 hit = inflated rate). Full 45-cell H-mean 0.16 is true rate. Future mechanism findings require n ≥ 30 cells before "sweet spot" claims.

### 5.4 Image-axis peak-layer signature (Method 4.2 v2, 8 pairs — REORGANIZED 2026-05-13)

`docs/checkpoints/mechanism/results/layer_axis_emergence_v2_{cls,reddit}.md`. V2 NPZ reorganizes the dichotomy from text-format-based to **image-side-based**:

- **Image side = naked Vision** → peak L04 (all 4 pairs cls; cos 0.060-0.067)
- **Image side = annotated SoM** → peak L36 (all 4 pairs cls; cos 0.042-0.050)

Reddit cross-site preserves the same image-side-based split. V1 framing of "AXTree text → L04 vs flat-marks text → L17-L36 dichotomy" was partially NPZ-regex artifact (see §0 + §1.3). The Mirage mechanism signature in v2 is: annotated overlay matches text payload until late integration; naked image creates immediate visual-text mismatch detectable at L04.

### 5.5 H1 test: flat-list format variation (Method 4.2 extension, 2026-05-12)

`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.

exec
/bin/bash -lc "sed -n '468,540p' docs/checkpoints/mechanism/plan.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
### 7.3.0a Exp 3 logit lens v2 — output-layer amplification (2026-05-13 01:55)

`axis2_logit_lens_v2.md` + figure regen. Apply Qwen3-VL-4B `model.model.language_model.norm` + `model.lm_head` to per-layer per-mode mean hidden states on **v2 NPZ**, compute KL across 37 layers.

| Pair | Site | Peak L (KL) | Peak KL | Exp 1 v2 cos peak | Amplification |
|---|---|---|---|---|---|
| P-text↔P-SoM (axis-2 flat) | cls | **L25** | 0.0879 | 0.009 (L36) | **~10×** |
| DOM↔P-prompt (axis-2 hier) | cls | L21 | 0.0459 | 0.007 (L36) | **~7×** |
| DOM↔P-text (axis-1) | cls | L3 | 0.0425 | 0.005 (L36) | **~8×** (peak shift) |
| P-prompt↔P-SoM (axis-1) | cls | L3 | 0.0393 | 0.005 (L36) | **~8×** |
| P-text↔P-SoM (axis-2 flat) | reddit | L25 | 0.0574 | 0.007 (L36) | **~8×** |
| DOM↔P-prompt (axis-2 hier) | reddit | L25 | 0.0488 | 0.006 (L36) | **~8×** |

**v2 findings**:
1. **Axis-2 IS in output distribution** — KL ~0.05-0.09 at L25, NOT null. Despite sub-permille residual stream cosine (~0.002-0.009 L17), lm_head decoding amplifies into measurable output divergence.
2. **lm_head 7-10× amplification cosine → KL** (v2 amplification factor smaller than v1's 14-25× claim, but disjoint qualitatively unchanged). Amplification axis-agnostic (axis-1 and axis-2 both 7-10×).
3. **KL peak layer shift in v2**: axis-1 peaks at L3 (early) while axis-2 peaks at L25 (decoding). V1 had both at L23. The early-axis-1 peak suggests text-format prior dominates initial embeddings; prompt-family signal lives later. Cross-site reddit replicates the L25 axis-2 peak.
4. **KL @ L36 ≈ 0**: mean hidden at last layer collapses to common JSON-header prefix; mode-distinct signal concentrated in **L21-L25 decoding window**, not final embedding. "Knows but says differently" mirror of Wu et al. tool calling.

**Paper §5.7 v2 prose** (replaces v1 three-axis hierarchy prose): cosine-causal disjoint is the new hero — residual-stream cosine 0.002-0.009 expands to KL 0.04-0.09 via lm_head and to causal Δoverlap 0.20-0.30 via patching. Three converging numbers anchor "geometry underestimates causal", and the L21-L25 decoding window is the cheapest highest-signal feature for paper-2 deployment routing.

### 7.3.1 Reddit cross-site results (2026-05-12 16:30 — P5a + P5b analyses landed)

**P5a — Format variation H1 test on reddit** (`format_variation_h1_test_reddit.md`):

| Variant | Peak L (reddit) | Peak L (cls baseline) |
|---|---|---|
| som_standard / browser_use_at / tarsier_typed / xml_tagged | **L17** | L36 (last) |
| appagent_id / plain_numbered | **L04** | L36 |
| hash_id_control | **L04** ✓ (acts as control) | L36 (control failed) |
| plain_sentence | **L17** | L17 |
| dom (baseline) | **L04** ✓ | L04 ✓ |

**Reddit nuance — cleaner mid-layer fusion**: Reddit 上 marks-like 4/6 真 peak 在 L17 (mid-layer), cls 上 L36 是 monotonic increasing artifact (peak hit boundary). Reddit hash_id_control L04 acts as proper "no integer" control (cls 上失败). Reddit data supports Q5 mid-layer fusion hypothesis better than cls.

Caveats: small n (24×2=48/mode) makes 2/6 marks-like falling to L04 (appagent_id, plain_numbered) plausible as sampling noise; plain_sentence triggering L17 on reddit (not cls) suggests reddit narrative comments may pattern-match list semantics.

**P5b — Mirage signature on reddit** v2 NPZ (`stage4_method42_v2_reddit.md`, 2026-05-13):

| Test | v2 value at L17 | cls v2 baseline | v1 (buggy) reference |
|---|---|---|---|
| P-SoM ↔ DOM (text-axis sibling) | **0.0031** (sub-permille) | cls 0.0029 | v1 reddit 0.0098 (3× inflated) |
| P-SoM ↔ SoM (image-axis split) | **0.0367** | cls 0.0367 | v1 reddit 0.0423 (15% inflated) |
| P-SoM ↔ Vision | 0.0468 | cls 0.0468 | v1 0.0457 |
| DOM ↔ Vision peak L04 | 0.0658 (AUROC=1.0) | cls L04 0.067 | v1 0.0687 |

→ **Cross-site reddit replicates v2 cls magnitudes**: image-axis pairs preserve (~0.04-0.07), text-axis pairs collapse to sub-permille. The "P-SoM = text-axis sibling of DOM" claim still holds (L17 magnitude 0.003 for P-SoM↔DOM vs 0.037 for P-SoM↔SoM = 12× ratio) but the absolute magnitude is much smaller than v1 implied. Paper §5 4-fold (d) drop-one mechanism remains supported by AUROC linear-readability + image-axis dominance, not by cosine magnitude size.

**Paper §5 cross-site evidence stack v2 (post-NPZ fix)**:
1. P-SoM mid-layer causal mechanism (Stage 2/3 patching, 4-fold drop-one) — cls + reddit replicated ✓ (Stage 2/3 uses archive_subset not Stage 4 NPZ, unaffected by v2 migration)
2. Indexed-list format → shortcut activation — directional consistency cls ↔ reddit ✓ (format variation uses separate NPZ, unaffected)
3. Mirage signature geometric structure — cls + reddit AUROC 1.000 ✓, BUT image-axis peak-layer dichotomy DIVERGES cross-site in v2 (cls SoM-image side defers to L36, reddit SoM-image side mostly emerges at L04). See §1.3 honest framing.

**P4 selection-bias defense (2026-05-12 18:50)** — cls reverse-tier H1 (`format_variation_h1_test_cls_reverse.md`):

| Variant | strong-tier cls | reverse-tier cls | reddit |
|---|---|---|---|
| 6 marks-like | L36 monotonic | **L36 monotonic** ✓ same | L17 (4/6 真 peak) |
| hash_id_control | L36 (failed control) | **L36** ✓ same | L04 ✓ proper control |
| plain_sentence | L17 | **L22** close to L17 | L17 |
| dom baseline | L04 ✓ | **L04** ✓ | L04 ✓ |

H1 mechanism in cls is **not tier selection artifact** (strong vs reverse both replicate). Reddit data paradoxically cleaner reveal of true L17 mid-layer fusion locus (cls L36 is monotonic-boundary artifact).

### 7.4 Decisions pending

| Decision | Owner | Trigger |
|---|---|---|
| Co-author multimodal extension vs cite + independent framing | Zekun | After Zekun reply to message |
| Method 4.5 path: LA-HDMI vs SAE | Zekun + advisor sync | After v2 full sweep + Zekun reply |
| Paper §5 prose round | Codex + me | After v2 full + Zekun decision |

## 8. Roadmap (next 2-4 weeks)

exec
/bin/bash -lc "rg -n \"0\\.011|three-axis|3-4x|hidden_states\\.npz|L30|36 layers|37 layers|all layers|AUROC|sub-permille|orders of magnitude|8-44|8-25|0\\.088|0\\.057|0\\.005-0\\.009|noise\" docs/checkpoints/mechanism/plan.md docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md scripts/analysis/stage4_logit_lens_axis2.py scripts/analysis/stage4_pca_cosine_gap.py" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
scripts/analysis/stage4_logit_lens_axis2.py:4:shows only weak (~0.011) signal at L23?
scripts/analysis/stage4_logit_lens_axis2.py:15:axis-2 cosine gap is 0.011 at L23 but output KL is large at L30-L36,
scripts/analysis/stage4_logit_lens_axis2.py:19:  results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
scripts/analysis/stage4_logit_lens_axis2.py:20:  results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz
scripts/analysis/stage4_logit_lens_axis2.py:146:        "across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets",
scripts/analysis/stage4_logit_lens_axis2.py:198:        "- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family",
scripts/analysis/stage4_logit_lens_axis2.py:200:        "- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →",
scripts/analysis/stage4_logit_lens_axis2.py:207:        "axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.",
scripts/analysis/stage4_pca_cosine_gap.py:10:  B. AUROC: project hidden states onto (mean_A - mean_B) direction, predict mode
scripts/analysis/stage4_pca_cosine_gap.py:75:    # methodology audit 2026-05-12: previous AUROC fit direction on the
scripts/analysis/stage4_pca_cosine_gap.py:198:        "**AUROC protocol** (Bug 3 fix, codex methodology audit 2026-05-12): paper-grade "
scripts/analysis/stage4_pca_cosine_gap.py:208:        "| Mode pair | Peak layer | Cosine gap | AUROC (in-sample) | AUROC (lototask) |",
scripts/analysis/stage4_pca_cosine_gap.py:222:    L17_section.append("| Mode pair | L17 cosine gap | L17 AUROC in-sample | L17 AUROC lototask |")
scripts/analysis/stage4_pca_cosine_gap.py:271:    ax.set_title("(b) Pairwise AUROC (project onto Δ-mean direction, classify)")
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:7:**§5.7 "three-axis hierarchy with quantitatively distinct magnitudes" claim is INVALIDATED by v2 data.**
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:9:V1 numbers came from buggy NPZ where the SOM_MARKS regex `^\[\d+\]\s+\w+` extracted only 38 chars / 3 lines per task, dropping 71/72 marks. All Stage 4 cosine geometry was computed on near-empty text payloads where the only differentiator between flat-text modes (som / phantom_som / phantom_text) was prompt template. Modes still separated perfectly (AUROC 1.000) but the cosine-gap magnitudes were artifacts of prompt-template differences, not text-payload differences.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:11:V2 NPZ uses production `_extract_text_marks` (72-line full payload with `[id=N] {label}` envelope). Modes still separable (AUROC 1.000), but axis-1 + axis-2 cosine magnitudes collapse to noise level. Image-axis magnitudes preserve.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:30:| **P-text ↔ P-SoM (axis-2, paper §5.7 axis-2 anchor)** | L23 0.0114 | **L36** 0.0088 | -23% | L23 → L36 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:39:| Prompt-family axis (P-text↔P-SoM) | 0.011 | 0.009 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:44:The "image > text-format > prompt-family" hierarchy with 4:3:1-ish quantitative ratio (v1) is **wrong**. V2 reality: image axis dominates by ~5-10x; axis-1 is **smaller than** axis-2 (reversed ranking); both axis-1 and axis-2 are noise-level (<0.01 cosine).
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:55:| DOM ↔ Vision (image axis) | 0.0545 | 0.0571 | — | 0.0537 |
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:57:Reddit cross-site replication confirms the cls pattern: image-axis magnitudes preserve, axis-1 + axis-2 collapse to sub-permille at L17.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:59:## AUROC lototask (held-out, paper-grade Bug 3 fix)
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:61:All pairs at all layers report AUROC lototask = 1.000 (perfect held-out linear separability). The modes ARE distinguishable in residual stream; the **magnitude of the mode-mean difference** is just much smaller than v1 claimed.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:63:This is the key reframe: **separability survives, magnitude does not**. Cosine gap measures effect SIZE; AUROC measures CLASSIFICATION RELIABILITY. They can dissociate.
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:67:**§5.7 three-axis hierarchy** (the prior framing):
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:68:> "Three quantitatively distinct axes: image axis L17 0.041, text-format L23 0.029, prompt-family L23 0.011, with 4:3:1 magnitude ratio that holds cross-site."
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:71:> "All three axes are linearly readable in residual stream (held-out AUROC 1.000 across cls and reddit). The image axis dominates geometrically (~0.04-0.07 cosine peak) and emerges by L04. Text-format and prompt-family axes produce sub-permille mean-difference (cosine ~0.005-0.009) without a localized layer peak (monotone rise to boundary L36). The geometric magnitude rank-order reverses cross-site at L17 (axis-1 ≤ axis-2), indicating these axes are noise-level rather than quantitatively distinct dimensions."
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:109:✅ Held-out AUROC 1.000 linear-readability: unchanged
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:114:> 1. **Linear readability** (Method 4.2 lototask AUROC 1.000): all 6 modes linearly separable in residual stream; small cosine magnitudes but reliable classification
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:115:> 2. **Geometric magnitude is mostly image-axis driven** (Method 4.2 cosine peak): image presence produces ~0.04-0.07 cosine separation early L04; other axes produce sub-permille separation throughout
docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md:118:> The disjoint between **small geometric magnitude (0.005-0.009)** and **large causal patching effect (0.20-0.30)** is the new headline claim — it means residual-stream geometry underestimates causal influence by orders of magnitude. This is paper-grade-novel and reviewer-defensible.
docs/checkpoints/mechanism/plan.md:16:- ✗ V1 "three-axis hierarchy 4:3:1 magnitude ratio" → INVALIDATED. V2: image dominates ~5-10×; axis-1 and axis-2 both noise-level (cosine ~0.005-0.009); axis-1 magnitude is now ≤ axis-2 (reversed ranking).
docs/checkpoints/mechanism/plan.md:18:- ✓ AUROC linear-readability 1.000 cross-site → preserved.
docs/checkpoints/mechanism/plan.md:24:**New hero claim** (replaces v1 three-axis hierarchy): **cosine-causal disjoint** — geometric magnitude is sub-permille (0.005-0.009) but causal patching displaces overlap 20-30% AND lm_head amplifies cosine→KL by 8-25×. Residual-stream geometry underestimates causal influence by orders of magnitude; cosine gap measures effect SIZE while AUROC measures CLASSIFICATION RELIABILITY and they dissociate. Paper-grade novel + reviewer-defensible.
docs/checkpoints/mechanism/plan.md:35:| **2** | Behavioral (axis effects) | Axis 1 (text payload: AXTree vs [SOM_MARKS]) + Axis 2 (prompt: SoM-prompt vs DOM-prompt) are both linearly readable (AUROC 1.0) but geometrically sub-permille; Axis 3 (image presence: in vs out) is gating + geometrically dominant |
docs/checkpoints/mechanism/plan.md:37:| **4** | Model-internal | L11-L17 mid-layer window is BOTH causal patching disruption locus (Stage 2/3 Δoverlap -0.27 to -0.35) AND probe-decodable (AUROC 1.0 via lototask held-out). L23-L25 logit-lens window is where mode signal concentrates at output distribution. |
docs/checkpoints/mechanism/plan.md:45:| **Image-axis (Vision)** | DOM ↔ Vision | 0.057 | L04 0.067 | early visual encoder |
docs/checkpoints/mechanism/plan.md:47:| Axis-1 text-format | DOM ↔ P-text | 0.002 | L36 0.005 | sub-permille, monotone-to-boundary |
docs/checkpoints/mechanism/plan.md:48:| Axis-1 text-format | P-prompt ↔ P-SoM | 0.002 | L36 0.005 | sub-permille |
docs/checkpoints/mechanism/plan.md:49:| Axis-2 prompt-family flat | P-text ↔ P-SoM | 0.002 | L36 0.009 | sub-permille |
docs/checkpoints/mechanism/plan.md:50:| Axis-2 prompt-family hier | DOM ↔ P-prompt | 0.001 | L36 0.007 | sub-permille |
docs/checkpoints/mechanism/plan.md:52:**Geometric magnitudes** v2: image 0.04-0.07 / text-format 0.005 / prompt-family 0.009 → image dominates **5-10×**, axis-1 ≤ axis-2 (sub-permille).
docs/checkpoints/mechanism/plan.md:60:- Axis-2 P-text↔P-SoM cosine 0.002 at L17 → KL **0.088 at L25** (cls), 0.057 at L25 (reddit)
docs/checkpoints/mechanism/plan.md:61:- Cosine→KL amplification: **8-44× depending on pair**, peak amplification at L21-L25 decoding window
docs/checkpoints/mechanism/plan.md:67:- KL ~0.05-0.09 (output divergence intermediate, amplified 8-44× from cosine)
docs/checkpoints/mechanism/plan.md:69:This is the new paper §5 hero claim. AUROC linear-readability 1.000 holds throughout — modes ARE distinguishable in residual stream; the magnitude of the mode-mean difference is just much smaller than v1 claimed.
docs/checkpoints/mechanism/plan.md:97:**Paper §5 prose implication**: do NOT make a "peak-layer dichotomy is universal mechanism" claim. Honest framing: image-axis cosine peak structure varies by site (cls late-integration on SoM, reddit early-integration), with **AUROC linear-readability 1.000 preserved cross-site at all layers**. The "Mirage signature" claim must be reframed around AUROC + cosine magnitude rank-order (image > text-format ≈ prompt-family), not peak-layer location.
docs/checkpoints/mechanism/plan.md:148:- B. AUROC via (mean_A − mean_B) projection
docs/checkpoints/mechanism/plan.md:152:- Test A label perm: 9.8σ above noise (real 1.000 vs perm 0.629)
docs/checkpoints/mechanism/plan.md:186:> The patch-sensitive continuation window L11-L17 (block-output index convention) at the last-input-token position is causally consequential for phantom routing space mode selection in Qwen3-VL-4B web agents, under final-token-replacement activation patching. Mode-distinguishing signal is linearly readable in residual stream throughout (AUROC 1.000 lototask cross-site), with sub-permille cosine geometry for prompt-family / text-format axes (peak L36 monotone-to-boundary, magnitude ≤ 0.009) and ~5-10× larger image-axis geometry. The lm_head decoding amplifies sub-permille cosine into measurable KL (~0.05-0.09 at L21-L25 window, 7-10× amplification), and final-token patching at L11-L17 produces 20-30% Δoverlap-to-target. Residual-stream cosine geometry severely underestimates causal influence; signature layer ≠ decision layer ≠ amplification layer (mechanistic-interpretability standard finding cf. Wang et al. 2023 IOI).
docs/checkpoints/mechanism/plan.md:190:- ❌ "Three-axis hierarchy 4:3:1 magnitude ratio" → corrected to **image-dominant ~5-10× + axis-1 ≤ axis-2 both sub-permille** (v2 NPZ §1.2/§5.1/§7.3.0)
docs/checkpoints/mechanism/plan.md:198:1. **Probe-level** (Method 4.2 PCA cosine gap; AUROC reported as both `in_sample` and held-out `leave-one-task-out` after 2026-05-12 Bug 3 fix; v1 buggy NPZ data invalidated, v2 NPZ in flight: Myriad 359736 cls + 359737 reddit)
docs/checkpoints/mechanism/plan.md:206:| A1 | L17 last-token hidden state mediates action selection (not earlier obs token positions) | Stage 2/3 swept all layers, L17 is peak |
docs/checkpoints/mechanism/plan.md:214:Cell E random-injection control: replacing source hidden with Gaussian noise (same μ, σ) yields **null L17 disruption effect**. Confirms our patching effect is source-content-specific, not noise-driven. Most directly stresses A5.
docs/checkpoints/mechanism/plan.md:218:- Method 4.2 AUROC 1.000 = validation (decodability)
docs/checkpoints/mechanism/plan.md:224:### 5.1 Stage 4 Method 4.2 v2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers, 2026-05-13 canonical)
docs/checkpoints/mechanism/plan.md:226:V2 NPZ-corrected (Bug 1+2+5 fix, see §0). All AUROC lototask = 1.000 (held-out per-task fold) — modes linearly separable in residual stream. Cosine magnitudes:
docs/checkpoints/mechanism/plan.md:228:| Pair | L17 cos gap (v2) | Peak L / gap (v2) | AUROC | Axis |
docs/checkpoints/mechanism/plan.md:235:| DOM ↔ Vision | 0.0571 | L04 0.067 | 1.000 | image-axis (naked) |
docs/checkpoints/mechanism/plan.md:237:V1 (buggy) reference for diff: P-SoM↔P-text was L23 0.011 (now collapsed -23% to L36 0.009), DOM↔P-text was L23 0.025 (now collapsed -81% to L36 0.005). Image-axis pairs preserve magnitude (v1 0.041 vs v2 0.042 at peak). See `method42_v1_vs_v2_comparison.md` for full 15-pair diff.
docs/checkpoints/mechanism/plan.md:239:Reddit cross-site v2 replicates: image-axis pairs preserve magnitude (~0.04-0.07 cross-pair); axis-1 + axis-2 magnitudes both sub-permille (0.002-0.009 L17, monotone-to-boundary L36 0.005-0.009). Rank-order axis-1 ≤ axis-2 holds cross-site.
docs/checkpoints/mechanism/plan.md:318:| ✅ Stage 4 NPZ Bug 1+2+5 — does v2 invalidate paper claims? | **Closed 2026-05-13 02:52**: §5.7 three-axis hierarchy magnitude claim INVALIDATED. AUROC + Stage 2/3 patching + Method 4.4 + Exp 5 axis-2 patching INTACT. New hero claim cosine-causal disjoint replaces magnitude hierarchy | — |
docs/checkpoints/mechanism/plan.md:319:| ✅ Cross-site Method 4.2 — does cls finding replicate on reddit? | **Closed 2026-05-13 (v2 cls+reddit)**: image-axis L04/L36 peak preserved cross-site; axis-1 + axis-2 sub-permille cross-site; rank-order axis-1 ≤ axis-2 < image preserved | — |
docs/checkpoints/mechanism/plan.md:357:> 24 cls strong-tier × 2 step × 6 mode = 288 hidden states, 37 layer × 2560 dim。全 540 pair × layer AUROC = 1.000 (perm baseline 0.629, real 9.8σ above). 你方法在 multimodal Qwen 上 readable transfer 干净。
docs/checkpoints/mechanism/plan.md:392:| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | ✅ **done 18:50:46** — shape (260, 37, 2560), 10 modes, 46 MB pulled. Same pattern as cls strong-tier (L36 marks-like + L04 dom). Selection-bias defended | `stage4_format_variation_b1_cls_reverse/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:393:| **P5a** | reddit format variation (cross-site H1) | Myriad **354382** (3rd attempt) | ✅ **done 08:09:38** — shape (430, 37, 2560), 10 modes, 76 MB pulled | `stage4_format_variation_b1_reddit/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:394:| **P5b** | reddit Method 4.2 multimode (cross-site Mirage) | Myriad 353890 | ✅ **done 07:31:14** — 288 examples, 6 modes, 51 MB pulled | `stage4_multimode_b1_reddit/hidden_states.npz` |
docs/checkpoints/mechanism/plan.md:418:### 7.3.0 Exp 1 axis-2 layer profile v2 (2026-05-13 01:52 — three-axis hierarchy retracted)
docs/checkpoints/mechanism/plan.md:436:2. **Magnitudes collapse**: axis-1 ~0.005 (v1 said 0.029), axis-2 ~0.009 (v1 said 0.011); image preserves ~0.04. New ratio image:text:prompt ≈ **8:1:1** (not 4:3:1).
docs/checkpoints/mechanism/plan.md:437:3. **Reversed ranking**: v2 axis-2 magnitude (0.009) ≥ axis-1 magnitude (0.005). Both sub-permille and near noise-floor.
docs/checkpoints/mechanism/plan.md:440:**Reframe**: "Three-axis hierarchy with distinct quantitative magnitudes 4:3:1" is RETRACTED. New framing: axis-1 + axis-2 are sub-permille in residual stream but probe-decodable (AUROC 1.0) AND lm_head-amplified (Exp 3 logit lens). Paper §5.7 prose updates to **cosine-causal disjoint** as hero, not magnitude hierarchy.
docs/checkpoints/mechanism/plan.md:462:3. **Magnitudes all collapse to v2 sub-permille range** but per-task uniformity preserved. Axis-2 flat 100% > 0.005 cross-site (vs v1 100% > 0.010).
docs/checkpoints/mechanism/plan.md:464:**/stress W2 attack defused on v2 data**: axis-2 cosine gap is uniform per-task signature at the new sub-permille level. Distribution is tight, not 2-3 outlier-driven. **What v2 changed**: the mean is smaller (0.007 vs v1 0.013), but the per-task uniformity argument holds — every task contributes to the sub-permille signal, not 2-3 outliers.
docs/checkpoints/mechanism/plan.md:466:**Paper §5.7 v2 prose addendum**: per-task fragility argument is preserved at L23. Cross-site uniformity holds (0.0073 cls vs 0.0070 reddit, < 5% diff). Combined with logit lens 7-10× amplification, the sub-permille residual signal becomes the L21-L25 KL signal — both uniform per-task.
docs/checkpoints/mechanism/plan.md:470:`axis2_logit_lens_v2.md` + figure regen. Apply Qwen3-VL-4B `model.model.language_model.norm` + `model.lm_head` to per-layer per-mode mean hidden states on **v2 NPZ**, compute KL across 37 layers.
docs/checkpoints/mechanism/plan.md:478:| P-text↔P-SoM (axis-2 flat) | reddit | L25 | 0.0574 | 0.007 (L36) | **~8×** |
docs/checkpoints/mechanism/plan.md:482:1. **Axis-2 IS in output distribution** — KL ~0.05-0.09 at L25, NOT null. Despite sub-permille residual stream cosine (~0.002-0.009 L17), lm_head decoding amplifies into measurable output divergence.
docs/checkpoints/mechanism/plan.md:487:**Paper §5.7 v2 prose** (replaces v1 three-axis hierarchy prose): cosine-causal disjoint is the new hero — residual-stream cosine 0.002-0.009 expands to KL 0.04-0.09 via lm_head and to causal Δoverlap 0.20-0.30 via patching. Three converging numbers anchor "geometry underestimates causal", and the L21-L25 decoding window is the cheapest highest-signal feature for paper-2 deployment routing.
docs/checkpoints/mechanism/plan.md:503:Caveats: small n (24×2=48/mode) makes 2/6 marks-like falling to L04 (appagent_id, plain_numbered) plausible as sampling noise; plain_sentence triggering L17 on reddit (not cls) suggests reddit narrative comments may pattern-match list semantics.
docs/checkpoints/mechanism/plan.md:509:| P-SoM ↔ DOM (text-axis sibling) | **0.0031** (sub-permille) | cls 0.0029 | v1 reddit 0.0098 (3× inflated) |
docs/checkpoints/mechanism/plan.md:512:| DOM ↔ Vision peak L04 | 0.0658 (AUROC=1.0) | cls L04 0.067 | v1 0.0687 |
docs/checkpoints/mechanism/plan.md:514:→ **Cross-site reddit replicates v2 cls magnitudes**: image-axis pairs preserve (~0.04-0.07), text-axis pairs collapse to sub-permille. The "P-SoM = text-axis sibling of DOM" claim still holds (L17 magnitude 0.003 for P-SoM↔DOM vs 0.037 for P-SoM↔SoM = 12× ratio) but the absolute magnitude is much smaller than v1 implied. Paper §5 4-fold (d) drop-one mechanism remains supported by AUROC linear-readability + image-axis dominance, not by cosine magnitude size.
docs/checkpoints/mechanism/plan.md:519:3. Mirage signature geometric structure — cls + reddit AUROC 1.000 ✓, BUT image-axis peak-layer dichotomy DIVERGES cross-site in v2 (cls SoM-image side defers to L36, reddit SoM-image side mostly emerges at L04). See §1.3 honest framing.
docs/checkpoints/mechanism/plan.md:552:- **§6 cost-aware routing** — Peale et al. 2026 uncertainty decomposition anchor adds theoretical layer to phantom routing space's empirical AUROC. Method 4.2 cosine gap could serve as "reducible uncertainty" signal in deployment

exec
/bin/bash -lc "python3 - <<'PY'
import numpy as np, pathlib
for p in ['results/mechanistic/stage4_multimode_b1_cls/hidden_states_v2_fixed.npz','results/mechanistic/stage4_multimode_b1_reddit/hidden_states_v2_fixed.npz']:
    path=pathlib.Path(p)
    print('\\n',p, path.exists())
    if path.exists():
        d=np.load(path, allow_pickle=True)
        print(d.files)
        for k in ['hidden_states','mode_labels_str','task_ids','step_ids']:
            if k in d.files:
                arr=d[k]
                print(k, arr.shape, arr.dtype, arr[:8] if arr.ndim==1 else '')
        if 'mode_labels_str' in d.files:
            vals, counts=np.unique(d['mode_labels_str'], return_counts=True)
            print('counts', dict(zip(vals.tolist(), counts.tolist())))
        if 'task_ids' in d.files and 'mode_labels_str' in d.files:
            ml=d['mode_labels_str']; tids=d['task_ids']
            for m in np.unique(ml):
                print(m, 'tasks', len(set(tids[ml==m].tolist())), 'examples', (ml==m).sum())
PY" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 exited 1 in 0ms:
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named 'numpy'

codex
The later sections add a more serious inconsistency: §1.3 admits the image peak-layer story diverges cross-site, while §5.4 says it is preserved cross-site. I’m checking the generated metric artifacts directly so the critique distinguishes documentation drift from data.
exec
/bin/bash -lc "rg --files | rg 'method42_metrics|axis2_logit_lens|layer_axis_emergence_v2|stage4_method42_v2_reddit|axis2_layer|per_task'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/analysis/cross_sites/mechanism_per_task.json
docs/analysis/cross_sites/mechanism_per_task_report.md
scripts/analysis/stage4_axis2_layer_profile.py
scripts/analysis/stage4_axis2_per_task_fragility.py
scripts/analysis/stage4_h1_per_task_fragility.py
scripts/analysis/mechanism_per_task.py
docs/checkpoints/codex_prompts/mechanism_per_task_explanation.md
docs/checkpoints/codex_prompts/mechanism_per_task_case_studies.md
docs/checkpoints/mechanism/results/layer_axis_emergence_v2_cls.md
docs/checkpoints/mechanism/results/layer_axis_emergence_v2_reddit.md
docs/checkpoints/mechanism/results/axis2_layer_profile_v2.md
docs/checkpoints/mechanism/results/axis2_logit_lens.md
docs/checkpoints/mechanism/results/axis2_per_task_fragility_v2_L17.md
docs/checkpoints/mechanism/results/axis2_layer_profile.md
docs/checkpoints/mechanism/results/stage4_method42_v2_reddit.md
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md
docs/checkpoints/mechanism/results/axis2_per_task_fragility_v2_L23.md
docs/checkpoints/mechanism/results/axis2_logit_lens_v2.md
docs/checkpoints/mechanism/results/h1_per_task_fragility.md

exec
/bin/bash -lc "sed -n '1,220p' docs/checkpoints/mechanism/results/axis2_logit_lens_v2.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)

Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement
across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
amplified into output distribution divergence by late-layer decoding.

## Classifieds site

### Axis-2 (prompt-family) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L25** | 0.0879 | 0.0134 | 0.0520 | 0.0000 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L21** | 0.0459 | 0.0026 | 0.0240 | 0.0000 |

### Axis-1 (text-format) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L3** | 0.0425 | 0.0164 | 0.0096 | 0.0000 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L3** | 0.0393 | 0.0167 | 0.0242 | 0.0000 |

## Reddit site

### Axis-2 (prompt-family) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| P-text vs P-SoM  (axis-2 flat-text) | **L25** | 0.0574 | 0.0192 | 0.0391 | 0.0000 |
| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0488 | 0.0106 | 0.0415 | 0.0000 |

### Axis-1 (text-format) pairs:

| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
|---|---:|---:|---:|---:|---:|
| DOM vs P-text    (axis-1 DOM-prompt) | **L2** | 0.0330 | 0.0126 | 0.0322 | 0.0000 |
| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L3** | 0.0391 | 0.0136 | 0.0214 | 0.0000 |

## Interpretation

Three hypotheses tested:

- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family
  effect bypasses logit lens, only visible via attention heads or runtime decoding.
- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →
  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling
  'knows but says differently' mirror).
- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →
  prompt prior signal proportional to mid-layer geometry, no amplification.

Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to
axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.

exec
/bin/bash -lc "sed -n '1,220p' docs/checkpoints/mechanism/results/layer_axis_emergence_v2_cls.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
# Stage 4: image-axis peak-layer split — Mirage Effect signature

Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:

| no-image side | image side | no-img text | peak layer | peak cosine gap |
|---|---|---|---|---|
| DOM | Vision | AXTree | **L04** | 0.0670 |
| P-text | Vision | [SOM_MARKS] | **L04** | 0.0602 |
| P-prompt | Vision | AXTree | **L04** | 0.0664 |
| P-SoM | Vision | [SOM_MARKS] | **L04** | 0.0599 |
| DOM | SoM | AXTree | **L36** | 0.0496 |
| P-text | SoM | [SOM_MARKS] | **L36** | 0.0488 |
| P-prompt | SoM | AXTree | **L36** | 0.0439 |
| P-SoM | SoM | [SOM_MARKS] | **L36** | 0.0416 |

## Grouped by no-image side text format

### no-image text = `AXTree` (mean peak L20)

- DOM ↔ SoM: peak **L36** = 0.0496
- DOM ↔ Vision: peak **L04** = 0.0670
- P-prompt ↔ SoM: peak **L36** = 0.0439
- P-prompt ↔ Vision: peak **L04** = 0.0664

### no-image text = `[SOM_MARKS]` (mean peak L20)

- P-text ↔ SoM: peak **L36** = 0.0488
- P-text ↔ Vision: peak **L04** = 0.0602
- P-SoM ↔ SoM: peak **L36** = 0.0416
- P-SoM ↔ Vision: peak **L04** = 0.0599

## Mechanism interpretation (paper §5 v3 Mirage anchor)

When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).

When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.

**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.

**Paper §5 prose** (suggested):

> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*

exec
/bin/bash -lc "sed -n '1,220p' docs/checkpoints/mechanism/results/layer_axis_emergence_v2_reddit.md" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
# Stage 4: image-axis peak-layer split — Mirage Effect signature

Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:

| no-image side | image side | no-img text | peak layer | peak cosine gap |
|---|---|---|---|---|
| DOM | SoM | AXTree | **L04** | 0.0455 |
| DOM | Vision | AXTree | **L04** | 0.0658 |
| P-text | Vision | [SOM_MARKS] | **L04** | 0.0590 |
| P-prompt | SoM | AXTree | **L04** | 0.0434 |
| P-prompt | Vision | AXTree | **L04** | 0.0634 |
| P-SoM | SoM | [SOM_MARKS] | **L04** | 0.0386 |
| P-SoM | Vision | [SOM_MARKS] | **L04** | 0.0586 |
| P-text | SoM | [SOM_MARKS] | **L17** | 0.0433 |

## Grouped by no-image side text format

### no-image text = `AXTree` (mean peak L4)

- DOM ↔ SoM: peak **L04** = 0.0455
- DOM ↔ Vision: peak **L04** = 0.0658
- P-prompt ↔ SoM: peak **L04** = 0.0434
- P-prompt ↔ Vision: peak **L04** = 0.0634

### no-image text = `[SOM_MARKS]` (mean peak L7)

- P-text ↔ SoM: peak **L17** = 0.0433
- P-text ↔ Vision: peak **L04** = 0.0590
- P-SoM ↔ SoM: peak **L04** = 0.0386
- P-SoM ↔ Vision: peak **L04** = 0.0586

## Mechanism interpretation (paper §5 v3 Mirage anchor)

When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).

When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.

**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.

**Paper §5 prose** (suggested):

> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/plan.md | sed -n '1,110p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	---
     2	name: mechanism plan
     3	description: Full mechanism workspace — theory, lit anchor stack, methods, identification protocol, current findings, open questions, advisor sync, roadmap. Specialized companion to paper_planning §2; not a duplicate.
     4	type: workspace_plan
     5	last_substantive_update: 2026-05-13
     6	v2_retraction: 2026-05-13 — Stage 4 v1→v2 NPZ migration (Bug 1 tier filter + Bug 2 SOM_MARKS regex + Bug 5 model revision pin) invalidated several v1 quantitative claims. See §0 below.
     7	---
     8	
     9	# Mechanism Plan — paper §5
    10	
    11	## 0. v2 retraction summary (2026-05-13)
    12	
    13	V1 Stage 4 NPZ regex `^\[\d+\]\s+\w+` extracted only 38 chars / 3 lines per task, dropping 71/72 SOM_MARKS. Affected: Method 4.2 cosine geometry, Exp 1 axis-2 layer profile, Exp 3 logit lens, per-task fragility. V2 NPZ uses production `_extract_text_marks` (full 72-line `[id=N] {label}` payload). Re-extraction Myriad 359736 (cls) + 359737 (reddit) landed 2026-05-12 late, v2 metrics 2026-05-13 02:52.
    14	
    15	**What changed**:
    16	- ✗ V1 "three-axis hierarchy 4:3:1 magnitude ratio" → INVALIDATED. V2: image dominates ~5-10×; axis-1 and axis-2 both noise-level (cosine ~0.005-0.009); axis-1 magnitude is now ≤ axis-2 (reversed ranking).
    17	- ✗ V1 "AXTree → L04 vs flat → L17-L36" no-image-side dichotomy → REORGANIZED. V2: dichotomy is image-side-based (Vision→L04, SoM→L36), not text-format-based.
    18	- ✓ AUROC linear-readability 1.000 cross-site → preserved.
    19	- ✓ Image-axis cosine peaks (~0.04-0.07) → preserved.
    20	- ✓ Stage 2/3 patching (uses archive_subset, not Stage 4 NPZ) → unchanged.
    21	- ✓ Method 4.4 steering (separate pipeline) → unchanged.
    22	- ✓ Exp 5 axis-2 causal patching → unchanged.
    23	
    24	**New hero claim** (replaces v1 three-axis hierarchy): **cosine-causal disjoint** — geometric magnitude is sub-permille (0.005-0.009) but causal patching displaces overlap 20-30% AND lm_head amplifies cosine→KL by 8-25×. Residual-stream geometry underestimates causal influence by orders of magnitude; cosine gap measures effect SIZE while AUROC measures CLASSIFICATION RELIABILITY and they dissociate. Paper-grade novel + reviewer-defensible.
    25	
    26	Provenance: `docs/checkpoints/mechanism/results/method42_v1_vs_v2_comparison.md` (canonical v1↔v2 diff). V2 NPZ at `results/mechanistic/stage4_multimode_b1_{cls,reddit}/hidden_states_v2_fixed.npz`.
    27	
    28	## 1. Theory framework (1-screen summary, paper_planning §2 is canonical)
    29	
    30	### 1.1 Zoom 1-4 hierarchy
    31	
    32	| Zoom | Level | What our paper claims |
    33	|---|---|---|
    34	| **1** | Architectural | Phantom routing space = "skip annotated image" boundary contains 3 arms (P-text / P-prompt / P-SoM) sharing 4-fold drop-in property |
    35	| **2** | Behavioral (axis effects) | Axis 1 (text payload: AXTree vs [SOM_MARKS]) + Axis 2 (prompt: SoM-prompt vs DOM-prompt) are both linearly readable (AUROC 1.0) but geometrically sub-permille; Axis 3 (image presence: in vs out) is gating + geometrically dominant |
    36	| **3** | Named phenomena (lit-anchored) | Mirage Effect (Asadi 2026) / Scaffold Effect (Vu&Balloccu 2026) / Cross-modal flow (Kaduri) / Prompt-format sensitivity (Sclar 2024) |
    37	| **4** | Model-internal | L11-L17 mid-layer window is BOTH causal patching disruption locus (Stage 2/3 Δoverlap -0.27 to -0.35) AND probe-decodable (AUROC 1.0 via lototask held-out). L23-L25 logit-lens window is where mode signal concentrates at output distribution. |
    38	
    39	### 1.2 Cosine-causal disjoint (Method 4.2 v2 + Stage 2/3 + Exp 3 logit lens)
    40	
    41	V2 NPZ-corrected geometry (paper-grade canonical, 2026-05-13):
    42	
    43	| Axis | Pair | L17 cos gap | Peak L / gap | Notes |
    44	|---|---|---:|---:|---|
    45	| **Image-axis (Vision)** | DOM ↔ Vision | 0.057 | L04 0.067 | early visual encoder |
    46	| **Image-axis (SoM)** | P-SoM ↔ SoM | 0.003 | L36 0.042 | late integration |
    47	| Axis-1 text-format | DOM ↔ P-text | 0.002 | L36 0.005 | sub-permille, monotone-to-boundary |
    48	| Axis-1 text-format | P-prompt ↔ P-SoM | 0.002 | L36 0.005 | sub-permille |
    49	| Axis-2 prompt-family flat | P-text ↔ P-SoM | 0.002 | L36 0.009 | sub-permille |
    50	| Axis-2 prompt-family hier | DOM ↔ P-prompt | 0.001 | L36 0.007 | sub-permille |
    51	
    52	**Geometric magnitudes** v2: image 0.04-0.07 / text-format 0.005 / prompt-family 0.009 → image dominates **5-10×**, axis-1 ≤ axis-2 (sub-permille).
    53	
    54	**Causal patching magnitudes** (Stage 2/3 mid-layer L11-L17 window, 6/6 cells cross-site):
    55	- Δoverlap-to-target: -0.27 to -0.35 (cls + reddit, all SoM→{no-image-arm} forward cells)
    56	- Random injection control (E + Er): null effect
    57	- → **causal patching effect magnitude 20-30%** vs **geometric magnitude 0.5-1%**
    58	
    59	**Logit lens amplification** (Exp 3 v2, Qwen3-VL-4B `norm + lm_head` on per-layer means):
    60	- Axis-2 P-text↔P-SoM cosine 0.002 at L17 → KL **0.088 at L25** (cls), 0.057 at L25 (reddit)
    61	- Cosine→KL amplification: **8-44× depending on pair**, peak amplification at L21-L25 decoding window
    62	- KL collapses to ~0 at L36 (mean hidden collapses to common JSON-header prefix) → mode-distinct signal lives in **L23-L25 window**, not final embedding
    63	
    64	**Interpretive disjoint**: residual-stream cosine geometry severely underestimates causal influence. Three converging numbers:
    65	- Cosine gap 0.5-1% (geometric magnitude small)
    66	- Δoverlap 20-30% (causal effect large)
    67	- KL ~0.05-0.09 (output divergence intermediate, amplified 8-44× from cosine)
    68	
    69	This is the new paper §5 hero claim. AUROC linear-readability 1.000 holds throughout — modes ARE distinguishable in residual stream; the magnitude of the mode-mean difference is just much smaller than v1 claimed.
    70	
    71	### 1.3 Image-axis peak-layer signature (v2 — cross-site DIVERGENT, needs further work)
    72	
    73	V2 NPZ data shows the dichotomy **does NOT replicate cleanly cross-site**. This is a v2-revealed paper-grade nuance not present in v1:
    74	
    75	**Cls v2**: clean image-side-based dichotomy
    76	
    77	| Image side | Peak layer | All 4 pairs cos gap |
    78	|---|---|---:|
    79	| Vision (naked) | **L04** | 0.060-0.067 |
    80	| SoM (annotated) | **L36** | 0.042-0.050 |
    81	
    82	**Reddit v2**: peak layer mostly L04 across the board (7/8 pairs), only P-text↔SoM at L17.
    83	
    84	| Image side | Peak layer | Pairs |
    85	|---|---|---|
    86	| Vision (naked) | L04 (all 4) | DOM/P-text/P-prompt/P-SoM ↔ Vision |
    87	| SoM (annotated) | L04 (3/4) | DOM↔SoM 0.046, P-prompt↔SoM 0.043, P-SoM↔SoM 0.039 |
    88	| SoM (annotated) | **L17 (1/4)** | P-text↔SoM 0.043 |
    89	
    90	**Cross-site disagreement is real**: cls SoM-image pairs all defer to L36 late integration; reddit SoM-image pairs mostly emerge at L04 with one exception. Possible explanations:
    91	1. Reddit's smaller/sparser SoM overlay produces clearer early visual discrepancy regardless of text-payload format
    92	2. Cls listing-heavy DOM trees push annotated SoM cosine peak past mid-layers; reddit comment-thread DOM doesn't
    93	3. V2 NPZ sampling variance (288 ex each is borderline for layer-peak precision at 0.04 magnitude)
    94	
    95	**v1 framing retraction**: v1 said the dichotomy was no-image-side-text-based (AXTree → L04 vs `[SOM_MARKS]` → L17-L36) and cross-site stable. V2 data on cls reorganizes to image-side-based; v2 data on reddit collapses to L04 dominant. Neither v1 nor a single v2 reorganized framing replicates cross-site.
    96	
    97	**Paper §5 prose implication**: do NOT make a "peak-layer dichotomy is universal mechanism" claim. Honest framing: image-axis cosine peak structure varies by site (cls late-integration on SoM, reddit early-integration), with **AUROC linear-readability 1.000 preserved cross-site at all layers**. The "Mirage signature" claim must be reframed around AUROC + cosine magnitude rank-order (image > text-format ≈ prompt-family), not peak-layer location.
    98	
    99	### 1.4 H1 test confirms broader: flat-list (not just indexed) triggers shortcut (2026-05-12)
   100	
   101	Format variation extraction (Myriad job 352998, `stage4_format_variation_b1_cls`, 450 hidden states = 45 task-step × 10 modes). For each text format V, compute image-axis cosine gap V↔som per layer; peak layer reveals shortcut activation:
   102	
   103	| Format | Peak layer | Verdict |
   104	|---|---|---|
   105	| **AXTree hierarchical (DOM)** | **L04** | **SOLE format defeating shortcut** |
   106	| `"a, b, c, ..."` plain sentence | L17 | mid-level trigger |
   107	| `[N] role 'label'` (SoM standard) | L36 | strong trigger |
   108	| `@N label` (Browser Use) | L36 | strong trigger |
   109	| `id_N: label` (AppAgent) | L36 | strong trigger |
   110	| `[BN:r:l]` (Tarsier) | L36 | strong trigger |

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/plan.md | sed -n '224,242p;468,488p;514,520p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
   224	### 5.1 Stage 4 Method 4.2 v2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers, 2026-05-13 canonical)
   225	
   226	V2 NPZ-corrected (Bug 1+2+5 fix, see §0). All AUROC lototask = 1.000 (held-out per-task fold) — modes linearly separable in residual stream. Cosine magnitudes:
   227	
   228	| Pair | L17 cos gap (v2) | Peak L / gap (v2) | AUROC | Axis |
   229	|---|---:|---:|---:|---|
   230	| P-SoM ↔ P-text | 0.0019 | L36 0.009 | 1.000 | axis-2 prompt-family (flat) |
   231	| DOM ↔ P-prompt | 0.0015 | L36 0.007 | 1.000 | axis-2 prompt-family (hier) |
   232	| DOM ↔ P-text | 0.0021 | L36 0.005 | 1.000 | axis-1 text-format (DOM-prompt) |
   233	| P-prompt ↔ P-SoM | 0.0017 | L36 0.005 | 1.000 | axis-1 text-format (SoM-prompt) |
   234	| P-SoM ↔ SoM | 0.0029 | L36 0.042 | 1.000 | image-axis (annotated) |
   235	| DOM ↔ Vision | 0.0571 | L04 0.067 | 1.000 | image-axis (naked) |
   236	
   237	V1 (buggy) reference for diff: P-SoM↔P-text was L23 0.011 (now collapsed -23% to L36 0.009), DOM↔P-text was L23 0.025 (now collapsed -81% to L36 0.005). Image-axis pairs preserve magnitude (v1 0.041 vs v2 0.042 at peak). See `method42_v1_vs_v2_comparison.md` for full 15-pair diff.
   238	
   239	Reddit cross-site v2 replicates: image-axis pairs preserve magnitude (~0.04-0.07 cross-pair); axis-1 + axis-2 magnitudes both sub-permille (0.002-0.009 L17, monotone-to-boundary L36 0.005-0.009). Rank-order axis-1 ≤ axis-2 holds cross-site.
   240	
   241	### 5.2 Stage 2/3 patching disruption (14 cells, B1 cls + reddit)
   242	
   468	### 7.3.0a Exp 3 logit lens v2 — output-layer amplification (2026-05-13 01:55)
   469	
   470	`axis2_logit_lens_v2.md` + figure regen. Apply Qwen3-VL-4B `model.model.language_model.norm` + `model.lm_head` to per-layer per-mode mean hidden states on **v2 NPZ**, compute KL across 37 layers.
   471	
   472	| Pair | Site | Peak L (KL) | Peak KL | Exp 1 v2 cos peak | Amplification |
   473	|---|---|---|---|---|---|
   474	| P-text↔P-SoM (axis-2 flat) | cls | **L25** | 0.0879 | 0.009 (L36) | **~10×** |
   475	| DOM↔P-prompt (axis-2 hier) | cls | L21 | 0.0459 | 0.007 (L36) | **~7×** |
   476	| DOM↔P-text (axis-1) | cls | L3 | 0.0425 | 0.005 (L36) | **~8×** (peak shift) |
   477	| P-prompt↔P-SoM (axis-1) | cls | L3 | 0.0393 | 0.005 (L36) | **~8×** |
   478	| P-text↔P-SoM (axis-2 flat) | reddit | L25 | 0.0574 | 0.007 (L36) | **~8×** |
   479	| DOM↔P-prompt (axis-2 hier) | reddit | L25 | 0.0488 | 0.006 (L36) | **~8×** |
   480	
   481	**v2 findings**:
   482	1. **Axis-2 IS in output distribution** — KL ~0.05-0.09 at L25, NOT null. Despite sub-permille residual stream cosine (~0.002-0.009 L17), lm_head decoding amplifies into measurable output divergence.
   483	2. **lm_head 7-10× amplification cosine → KL** (v2 amplification factor smaller than v1's 14-25× claim, but disjoint qualitatively unchanged). Amplification axis-agnostic (axis-1 and axis-2 both 7-10×).
   484	3. **KL peak layer shift in v2**: axis-1 peaks at L3 (early) while axis-2 peaks at L25 (decoding). V1 had both at L23. The early-axis-1 peak suggests text-format prior dominates initial embeddings; prompt-family signal lives later. Cross-site reddit replicates the L25 axis-2 peak.
   485	4. **KL @ L36 ≈ 0**: mean hidden at last layer collapses to common JSON-header prefix; mode-distinct signal concentrated in **L21-L25 decoding window**, not final embedding. "Knows but says differently" mirror of Wu et al. tool calling.
   486	
   487	**Paper §5.7 v2 prose** (replaces v1 three-axis hierarchy prose): cosine-causal disjoint is the new hero — residual-stream cosine 0.002-0.009 expands to KL 0.04-0.09 via lm_head and to causal Δoverlap 0.20-0.30 via patching. Three converging numbers anchor "geometry underestimates causal", and the L21-L25 decoding window is the cheapest highest-signal feature for paper-2 deployment routing.
   488	
   514	→ **Cross-site reddit replicates v2 cls magnitudes**: image-axis pairs preserve (~0.04-0.07), text-axis pairs collapse to sub-permille. The "P-SoM = text-axis sibling of DOM" claim still holds (L17 magnitude 0.003 for P-SoM↔DOM vs 0.037 for P-SoM↔SoM = 12× ratio) but the absolute magnitude is much smaller than v1 implied. Paper §5 4-fold (d) drop-one mechanism remains supported by AUROC linear-readability + image-axis dominance, not by cosine magnitude size.
   515	
   516	**Paper §5 cross-site evidence stack v2 (post-NPZ fix)**:
   517	1. P-SoM mid-layer causal mechanism (Stage 2/3 patching, 4-fold drop-one) — cls + reddit replicated ✓ (Stage 2/3 uses archive_subset not Stage 4 NPZ, unaffected by v2 migration)
   518	2. Indexed-list format → shortcut activation — directional consistency cls ↔ reddit ✓ (format variation uses separate NPZ, unaffected)
   519	3. Mirage signature geometric structure — cls + reddit AUROC 1.000 ✓, BUT image-axis peak-layer dichotomy DIVERGES cross-site in v2 (cls SoM-image side defers to L36, reddit SoM-image side mostly emerges at L04). See §1.3 honest framing.
   520	

exec
/bin/bash -lc "nl -ba scripts/analysis/stage4_logit_lens_axis2.py | sed -n '1,35p;140,210p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	#!/usr/bin/env python3
     2	"""Exp 3: Logit lens at late layers — does axis-2 prompt-family signal
     3	re-emerge in output distribution even though mid-layer residual stream
     4	shows only weak (~0.011) signal at L23?
     5	
     6	Method: Apply Qwen3-VL-4B's lm_head + final_norm to each per-layer hidden
     7	state mean, get a token distribution per (mode, layer). For each axis-2
     8	pair (P-text vs P-SoM at same task) compute:
     9	  - top-1 token disagreement rate per layer
    10	  - KL divergence (P-text || P-SoM) per layer
    11	  - log-prob gap on canonical SoM-prompt vs DOM-prompt action tokens
    12	    (e.g., "click" vs "search", "_pick_", json keys)
    13	
    14	This is Wu et al. tool-calling "knows but says differently" mirror: if
    15	axis-2 cosine gap is 0.011 at L23 but output KL is large at L30-L36,
    16	prompt prior is amplified by late-layer decoding into different output.
    17	
    18	Inputs:
    19	  results/mechanistic/stage4_multimode_b1_cls/hidden_states.npz
    20	  results/mechanistic/stage4_multimode_b1_reddit/hidden_states.npz
    21	
    22	Outputs:
    23	  docs/checkpoints/mechanism/results/axis2_logit_lens.md
    24	  results/phantom_paper/figures/fig_axis2_logit_lens.png
    25	"""
    26	from __future__ import annotations
    27	
    28	import argparse
    29	import json
    30	from pathlib import Path
    31	
    32	import matplotlib.pyplot as plt
    33	import numpy as np
    34	import torch
    35	from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
   140	    out.parent.mkdir(parents=True, exist_ok=True)
   141	    lines = [
   142	        "# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)",
   143	        "",
   144	        "Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.",
   145	        "For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement",
   146	        "across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets",
   147	        "amplified into output distribution divergence by late-layer decoding.",
   148	        "",
   149	        "## Classifieds site",
   150	        "",
   151	        "### Axis-2 (prompt-family) pairs:",
   152	        "",
   153	        "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
   154	        "|---|---:|---:|---:|---:|---:|",
   155	    ]
   156	    for label, info in cls_axis2.items():
   157	        kl = info["kl"]
   158	        lines.append(
   159	            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
   160	            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
   161	        )
   162	    lines += ["", "### Axis-1 (text-format) pairs:", "",
   163	              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
   164	              "|---|---:|---:|---:|---:|---:|"]
   165	    for label, info in cls_axis1.items():
   166	        kl = info["kl"]
   167	        lines.append(
   168	            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
   169	            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
   170	        )
   171	
   172	    lines += ["", "## Reddit site", "",
   173	              "### Axis-2 (prompt-family) pairs:", "",
   174	              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
   175	              "|---|---:|---:|---:|---:|---:|"]
   176	    for label, info in red_axis2.items():
   177	        kl = info["kl"]
   178	        lines.append(
   179	            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
   180	            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
   181	        )
   182	    lines += ["", "### Axis-1 (text-format) pairs:", "",
   183	              "| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |",
   184	              "|---|---:|---:|---:|---:|---:|"]
   185	    for label, info in red_axis1.items():
   186	        kl = info["kl"]
   187	        lines.append(
   188	            f"| {label} | **L{info['peak_kl_L']}** | {info['peak_kl']:.4f} | "
   189	            f"{kl[17]:.4f} | {kl[23]:.4f} | {kl[-1]:.4f} |"
   190	        )
   191	
   192	    lines += [
   193	        "",
   194	        "## Interpretation",
   195	        "",
   196	        "Three hypotheses tested:",
   197	        "",
   198	        "- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family",
   199	        "  effect bypasses logit lens, only visible via attention heads or runtime decoding.",
   200	        "- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →",
   201	        "  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling",
   202	        "  'knows but says differently' mirror).",
   203	        "- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →",
   204	        "  prompt prior signal proportional to mid-layer geometry, no amplification.",
   205	        "",
   206	        "Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to",
   207	        "axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.",
   208	    ]
   209	    out.write_text("\n".join(lines) + "\n")
   210	    print(f"summary → {out}")

exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/results/layer_axis_emergence_v2_cls.md | sed -n '1,60p'; nl -ba docs/checkpoints/mechanism/results/layer_axis_emergence_v2_reddit.md | sed -n '1,60p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4: image-axis peak-layer split — Mirage Effect signature
     2	
     3	Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:
     4	
     5	| no-image side | image side | no-img text | peak layer | peak cosine gap |
     6	|---|---|---|---|---|
     7	| DOM | Vision | AXTree | **L04** | 0.0670 |
     8	| P-text | Vision | [SOM_MARKS] | **L04** | 0.0602 |
     9	| P-prompt | Vision | AXTree | **L04** | 0.0664 |
    10	| P-SoM | Vision | [SOM_MARKS] | **L04** | 0.0599 |
    11	| DOM | SoM | AXTree | **L36** | 0.0496 |
    12	| P-text | SoM | [SOM_MARKS] | **L36** | 0.0488 |
    13	| P-prompt | SoM | AXTree | **L36** | 0.0439 |
    14	| P-SoM | SoM | [SOM_MARKS] | **L36** | 0.0416 |
    15	
    16	## Grouped by no-image side text format
    17	
    18	### no-image text = `AXTree` (mean peak L20)
    19	
    20	- DOM ↔ SoM: peak **L36** = 0.0496
    21	- DOM ↔ Vision: peak **L04** = 0.0670
    22	- P-prompt ↔ SoM: peak **L36** = 0.0439
    23	- P-prompt ↔ Vision: peak **L04** = 0.0664
    24	
    25	### no-image text = `[SOM_MARKS]` (mean peak L20)
    26	
    27	- P-text ↔ SoM: peak **L36** = 0.0488
    28	- P-text ↔ Vision: peak **L04** = 0.0602
    29	- P-SoM ↔ SoM: peak **L36** = 0.0416
    30	- P-SoM ↔ Vision: peak **L04** = 0.0599
    31	
    32	## Mechanism interpretation (paper §5 v3 Mirage anchor)
    33	
    34	When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).
    35	
    36	When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
    37	
    38	**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.
    39	
    40	**Paper §5 prose** (suggested):
    41	
    42	> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*
     1	# Stage 4: image-axis peak-layer split — Mirage Effect signature
     2	
     3	Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:
     4	
     5	| no-image side | image side | no-img text | peak layer | peak cosine gap |
     6	|---|---|---|---|---|
     7	| DOM | SoM | AXTree | **L04** | 0.0455 |
     8	| DOM | Vision | AXTree | **L04** | 0.0658 |
     9	| P-text | Vision | [SOM_MARKS] | **L04** | 0.0590 |
    10	| P-prompt | SoM | AXTree | **L04** | 0.0434 |
    11	| P-prompt | Vision | AXTree | **L04** | 0.0634 |
    12	| P-SoM | SoM | [SOM_MARKS] | **L04** | 0.0386 |
    13	| P-SoM | Vision | [SOM_MARKS] | **L04** | 0.0586 |
    14	| P-text | SoM | [SOM_MARKS] | **L17** | 0.0433 |
    15	
    16	## Grouped by no-image side text format
    17	
    18	### no-image text = `AXTree` (mean peak L4)
    19	
    20	- DOM ↔ SoM: peak **L04** = 0.0455
    21	- DOM ↔ Vision: peak **L04** = 0.0658
    22	- P-prompt ↔ SoM: peak **L04** = 0.0434
    23	- P-prompt ↔ Vision: peak **L04** = 0.0634
    24	
    25	### no-image text = `[SOM_MARKS]` (mean peak L7)
    26	
    27	- P-text ↔ SoM: peak **L17** = 0.0433
    28	- P-text ↔ Vision: peak **L04** = 0.0590
    29	- P-SoM ↔ SoM: peak **L04** = 0.0386
    30	- P-SoM ↔ Vision: peak **L04** = 0.0586
    31	
    32	## Mechanism interpretation (paper §5 v3 Mirage anchor)
    33	
    34	When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).
    35	
    36	When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
    37	
    38	**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.
    39	
    40	**Paper §5 prose** (suggested):
    41	
    42	> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*

codex
**Hostile Audit**

The retraction is directionally honest, but the replacement claim still overreaches.

Major weakness: the new “cosine-causal disjoint” claim compares incommensurable quantities. A cosine gap of 0.005, KL of 0.05, and overlap displacement of 20-30% are not the same unit, not the same intervention, and not measured on the same object. Calling this “orders of magnitude” evidence that geometry underestimates causality is rhetorically strong but statistically weak. At most it says three probes give different scales. A reviewer will ask for a calibrated effect-size bridge or ablation showing small-cosine directions are specifically causal, not merely correlated with a large patching operation.

The “sub-permille” language is also wrong or at least sloppy. Cosine gaps of 0.005-0.009 are 0.5-0.9%, not sub-permille if interpreted literally. This matters because the whole reframing leans on scale rhetoric.

The AUROC=1.000 result is not as impressive as written. Leave-one-task-out still lets the classifier exploit mode-template regularities, fixed formatting, token-position artifacts, and shared extraction conventions. It proves “mode labels are linearly decodable under this construction,” not “the model has a routing-axis representation.” Perfect pairwise AUROC with tiny cosine is exactly what I would expect from systematic prompt scaffolding. You need harder controls: train on one site/test on another, remove static prompt/template tokens, equalize payload length/position, classify only within semantically matched payloads, and report multiclass mode decoding, not only pairwise projections.

There is serious documentation drift. [plan.md](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/mechanism/plan.md:71) correctly says the image-axis peak-layer signature does not replicate cleanly cross-site. But [plan.md](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/mechanism/plan.md:239) says reddit replicates, and §5.4 apparently claims the same split is preserved. The generated layer-axis files are worse: their tables show cls is image-side-based while reddit is mostly L04, but the interpretation still says `[SOM_MARKS]` shifts peaks to L17-L36. That is false for reddit and not even cleanly true for cls as written. Any reviewer catching this will downgrade confidence in the whole mechanism narrative.

The logit-lens evidence is fragile. [stage4_logit_lens_axis2.py](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/scripts/analysis/stage4_logit_lens_axis2.py:2) still contains v1 assumptions: L23 cosine 0.011, L30-L36 amplification, old NPZ paths, and the generated markdown repeats the stale framing. Re-running the script can regenerate a v2-looking file with v1 prose. More substantively, it applies norm+lm_head to per-mode mean hidden states. KL of logits from an averaged hidden vector is not the expected KL over examples, and nonlinearity makes this a questionable surrogate. The “L36 collapses to common JSON-header prefix” explanation is asserted, not demonstrated by token-level or example-level analysis.

The causal patching story is under-identified relative to the paper claims. Final-token replacement at L11-L17 causing overlap shifts does not establish a natural decision locus, especially if the same representations are strongly mode-template decodable everywhere. You need patching controls that preserve mode but swap payload, preserve payload but swap prompt, and patch earlier observation-token positions. Right now the claim “mode selection mechanism” outruns the intervention.

The scripts themselves encourage overclaiming. [stage4_pca_cosine_gap.py](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/scripts/analysis/stage4_pca_cosine_gap.py:198) labels cosine peaks as “Peak disruption layer,” conflating descriptive geometry with causal disruption. Raw residual mean cosine is uncentered and unwhitened; no isotropic/noise-floor calibration is visible in the core method. Calling <0.01 “noise-level” while still using it as a hero mechanism is unstable.

Bottom line: v2 retracts the broken 4:3:1 result, but the replacement is still too narrative-first. The defensible paper claim is narrower: mode labels remain perfectly decodable under current extraction; image/no-image geometry is much larger than text/prompt geometry; patching changes output overlap in selected Mirage tasks. The current “cosine-causal disjoint” mechanism needs stronger controls before it is NeurIPS/ICML-safe.

=== END ===
tokens used
105,747
**Hostile Audit**

The retraction is directionally honest, but the replacement claim still overreaches.

Major weakness: the new “cosine-causal disjoint” claim compares incommensurable quantities. A cosine gap of 0.005, KL of 0.05, and overlap displacement of 20-30% are not the same unit, not the same intervention, and not measured on the same object. Calling this “orders of magnitude” evidence that geometry underestimates causality is rhetorically strong but statistically weak. At most it says three probes give different scales. A reviewer will ask for a calibrated effect-size bridge or ablation showing small-cosine directions are specifically causal, not merely correlated with a large patching operation.

The “sub-permille” language is also wrong or at least sloppy. Cosine gaps of 0.005-0.009 are 0.5-0.9%, not sub-permille if interpreted literally. This matters because the whole reframing leans on scale rhetoric.

The AUROC=1.000 result is not as impressive as written. Leave-one-task-out still lets the classifier exploit mode-template regularities, fixed formatting, token-position artifacts, and shared extraction conventions. It proves “mode labels are linearly decodable under this construction,” not “the model has a routing-axis representation.” Perfect pairwise AUROC with tiny cosine is exactly what I would expect from systematic prompt scaffolding. You need harder controls: train on one site/test on another, remove static prompt/template tokens, equalize payload length/position, classify only within semantically matched payloads, and report multiclass mode decoding, not only pairwise projections.

There is serious documentation drift. [plan.md](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/mechanism/plan.md:71) correctly says the image-axis peak-layer signature does not replicate cleanly cross-site. But [plan.md](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/docs/checkpoints/mechanism/plan.md:239) says reddit replicates, and §5.4 apparently claims the same split is preserved. The generated layer-axis files are worse: their tables show cls is image-side-based while reddit is mostly L04, but the interpretation still says `[SOM_MARKS]` shifts peaks to L17-L36. That is false for reddit and not even cleanly true for cls as written. Any reviewer catching this will downgrade confidence in the whole mechanism narrative.

The logit-lens evidence is fragile. [stage4_logit_lens_axis2.py](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/scripts/analysis/stage4_logit_lens_axis2.py:2) still contains v1 assumptions: L23 cosine 0.011, L30-L36 amplification, old NPZ paths, and the generated markdown repeats the stale framing. Re-running the script can regenerate a v2-looking file with v1 prose. More substantively, it applies norm+lm_head to per-mode mean hidden states. KL of logits from an averaged hidden vector is not the expected KL over examples, and nonlinearity makes this a questionable surrogate. The “L36 collapses to common JSON-header prefix” explanation is asserted, not demonstrated by token-level or example-level analysis.

The causal patching story is under-identified relative to the paper claims. Final-token replacement at L11-L17 causing overlap shifts does not establish a natural decision locus, especially if the same representations are strongly mode-template decodable everywhere. You need patching controls that preserve mode but swap payload, preserve payload but swap prompt, and patch earlier observation-token positions. Right now the claim “mode selection mechanism” outruns the intervention.

The scripts themselves encourage overclaiming. [stage4_pca_cosine_gap.py](/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents/scripts/analysis/stage4_pca_cosine_gap.py:198) labels cosine peaks as “Peak disruption layer,” conflating descriptive geometry with causal disruption. Raw residual mean cosine is uncentered and unwhitened; no isotropic/noise-floor calibration is visible in the core method. Calling <0.01 “noise-level” while still using it as a hero mechanism is unstable.

Bottom line: v2 retracts the broken 4:3:1 result, but the replacement is still too narrative-first. The defensible paper claim is narrower: mode labels remain perfectly decodable under current extraction; image/no-image geometry is much larger than text/prompt geometry; patching changes output overlap in selected Mirage tasks. The current “cosine-causal disjoint” mechanism needs stronger controls before it is NeurIPS/ICML-safe.

=== END ===
