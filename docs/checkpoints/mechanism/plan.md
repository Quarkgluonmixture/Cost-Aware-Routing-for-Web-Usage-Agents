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

## 6. Open questions (paper-grade gaps)

| Q | Status | Next action |
|---|---|---|
| ✅ Method 4.4 v2 full 48-cell sweep — sweet spot stable? | **Closed 2026-05-11 22:00**: L17 α=5 smoke 0.44 → full 0.16 (smoke variance artifact). **Real sweet spot L33 α=10 H-mean 0.33** | — |
| ✅ H1 test: do all flat-list formats trigger shortcut? | **Closed 2026-05-12 00:00**: YES, including hash-ID + plain-sentence controls. AXTree-DOM is sole defeating format | — |
| ✅ Stage 4 NPZ Bug 1+2+5 — does v2 invalidate paper claims? | **Closed 2026-05-13 02:52**: §5.7 three-axis hierarchy magnitude claim INVALIDATED. AUROC + Stage 2/3 patching + Method 4.4 + Exp 5 axis-2 patching INTACT. New hero claim cosine-causal disjoint replaces magnitude hierarchy | — |
| ✅ Cross-site Method 4.2 — does cls finding replicate on reddit? | **Closed 2026-05-13 (v2 cls+reddit)**: image-axis L04/L36 peak preserved cross-site; axis-1 + axis-2 sub-permille cross-site; rank-order axis-1 ≤ axis-2 < image preserved | — |
| ✅ Stage 3 reddit 2x2 closure — H-d-red | **Closed 2026-05-12 19:57** (Myriad 358831). L11 Δ=-0.33 / L17 Δ=-0.26. Cross-site additivity confirmed — see §5.2 Stage 3 table | — |
| §5.7 v2 prose surgery (#52) | HIGH | Re-write paper §5 to use cosine-causal disjoint hero claim, retract 4:3:1 ratio, anchor image-side-based dichotomy |
| Reverse-tier 15 tasks vs strong-tier 24 — does L33 + H1 finding generalize beyond selection bias? | Med-High | qsub Stage 4 multimode + format variation with --tier reverse on v2 NPZ |
| LA-HDMI vs mean-diff — does gradient steering beat 0.33 ceiling? | Med | Pending Zekun reply + attribution decision |
| SAE feature steering feasibility — is 1-2 week self-training Qwen3-VL-4B SAE worth it? | Low-Med | Depends on Zekun reply + paper §8 prose direction |
| Cross-family (Phi-3.5-Vision + Qwen2-VL-7B) — does cosine-causal disjoint hold? | Med (#45) | HF cache landed 2026-05-13; pending GPU + adapter code |
| B0 (proxy API) — paper §5 Qwen-specific or generalizable? | Low | Cannot test on B0; cite Wu et al. cross-family generality as proxy |
| AXTree-defeats-shortcut mechanism — *why* hierarchy beats flat? Cross-modal attention specific to indentation tokens? | High (paper §5 supplement) | Activation patching at L4 with hierarchical-text vs flat-text → see which attention heads pre-disrupt image embedding |

## 7. Advisor sync state — Zekun (Wu et al. 2026 first author = lab member)

### 7.1 Timeline confirmed (not scoop)

- 2026-04-09 笔记 §19: I first grok the paper (then "Anonymous 2026 ACL"), record cosine gap method + L23+ steering 80-93%
- 2026-05-01 笔记 §108.19: upgraded to Zoom 4 anchor stack
- 2026-05-02 commit `6662b91`: anchored into paper_planning §2 + paper.bib placeholder
- 2026-05-09 advisor recording: Zekun explicitly recommended "SAE feature steering — 前所未有 inference time steering, 单独发 paper" — directed me to differentiating path
- 2026-05-11: arxiv landed publicly; identity confirmed as lab paper

**Net**: Zekun explicitly invited mechanism extension. Method 4.4 multimodal port is on his recommendation; SAE Method 4.5 is his next-step suggestion.

### 7.2 Message draft (v3, paste-ready 2026-05-12)

Updated after v2 full sweep + H1 test. Key revisions from §125.10 draft:
- ❌ Removed: "L17 α=5 H-mean 0.44 mid-layer sweet spot" (smoke variance artifact, full data refutes)
- ✓ Added: **L33 α=10 H-mean 0.33** = matches your second-to-last-layer choice; multi-step JSON selectivity drop explains 38% vs your 93% gap
- ✓ Added: H1 test finding — flat-list format universally triggers shortcut (8/8 variants), only AXTree hierarchical defeats; implication for industry SoM-family agents
- ✓ Three asks: (a) attribution co-author vs cite + independent; (b) your ablation on mid- vs late-layer (we see selectivity tradeoff); (c) SAE direction priority given mean-diff ceiling

Final message (Chinese, casual WeChat tone):

> Zekun 早, 你那篇 Tool Calling 上 arxiv 我看了, 恭喜! 我前几天按你说的开始 mechanism work, 跑出来一些东西想跟你 sync 一下, 顺便问几个方向问题。
>
> # Context
> P79 paper 在做 VisualWebArena 的 phantom routing space — agent 6 种 obs mode (DOM 文本/SoM 标注图/Vision 裸图 + 3 个 phantom 变体). 模型 Qwen3-VL-4B, 你 Qwen 3 4B 同 base LM。
>
> # 1. Method 4.2 PCA cosine gap port 到 6 modes
> 24 cls strong-tier × 2 step × 6 mode = 288 hidden states, 37 layer × 2560 dim。全 540 pair × layer AUROC = 1.000 (perm baseline 0.629, real 9.8σ above). 你方法在 multimodal Qwen 上 readable transfer 干净。
>
> # 2. Method 4.4 mean-diff steering (HDMI metric)
> 45 task-step × 6 layer × 5 α full sweep. 用 HDMI completeness×selectivity → H-mean 评估:
>
>   - **L33 α=10 H-mean 0.33** (sweet spot, c=38% s=29%) ← matches 你 paper second-to-last-layer
>   - Mid-layer (L11-L23) selectivity 100% 但 completeness 0-11% — readable but not effectively steerable
>   - 你 paper Qwen 3 4B 93% switch vs 我 38% — 我猜原因是 multi-step JSON gen 的 selectivity 是真约束 (你 single-token tool decision selectivity 自动 1.0)
>
> # 3. H1 test: flat-list format variation (Myriad)
> 测了 8 个 industry-relevant text format (Browser Use @, AppAgent id_, Tarsier typed, numbered, XML, hash-ID, plain-sentence + SoM baseline) vs AXTree-DOM:
>
>   - 全 8 flat variants peak L17/L36 (= 都触发 shortcut)
>   - **AXTree hierarchical 是唯一保留 L04 peak 的 format**
>   - 包括 hash-ID (no integer) + plain-sentence (no list) 都触发
>   - = SoM-family agents 全 implicit exploit 同一 VLM shortcut, AXTree 是 sole exception
>
> # 三个 ask
> (1) Attribution: paper §5 mechanism 这块 — cite 你 + 我独立 framing 比较合理, 还是 co-author 一篇 multimodal extension 比较好? 都 OK, 想听你意见。
>
> (2) 你 ablation 里有跑过 mid- vs late-layer 对比吗? 我 mid-layer selectivity 100% 但 shift 弱, late-layer shift 强但 envelope 破 — 不知道你 tool calling 上是不是也有这种 tradeoff。
>
> (3) 你之前 advisor 录音里建议 SAE feature steering, 我也写进 future work 了。现在 mean-diff ceiling ~0.33, 是不是 SAE 这条路更有差异化? Qwen3-VL-4B SAE 没公开, 自训成本 1-2 周, 你觉得值得 commit GPU 吗?
>
> 不急, 你忙完回我就行. paper 写得真漂亮.

### 7.3 H1 generalization in-flight (2026-05-12 night)

After per-task fragility revealed 11% strict dichotomy (aggregate statistical, not deterministic), launched 5-priority defense matrix to triangulate H1 across **(tier × site × family/size)**:

| Pri | Test | Where | Status @ 06:25 | Sentinel |
|---|---|---|---|---|
| **P1** | Per-task fragility audit (24 cls strong) | DGX | ✅ done | `results/h1_per_task_fragility.md` |
| **P2** | Cross-family (Phi-3.5-Vision 4.2B) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_phi35_cls/pilot_summary.md` |
| **P3** | Within-family bigger (Qwen2-VL-7B, H1' capacity test) | DGX | ❌ deferred (HF cas-bridge throttling) | `stage4_h1_qwen2vl7b_cls/pilot_summary.md` |
| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | ✅ **done 18:50:46** — shape (260, 37, 2560), 10 modes, 46 MB pulled. Same pattern as cls strong-tier (L36 marks-like + L04 dom). Selection-bias defended | `stage4_format_variation_b1_cls_reverse/hidden_states.npz` |
| **P5a** | reddit format variation (cross-site H1) | Myriad **354382** (3rd attempt) | ✅ **done 08:09:38** — shape (430, 37, 2560), 10 modes, 76 MB pulled | `stage4_format_variation_b1_reddit/hidden_states.npz` |
| **P5b** | reddit Method 4.2 multimode (cross-site Mirage) | Myriad 353890 | ✅ **done 07:31:14** — 288 examples, 6 modes, 51 MB pulled | `stage4_multimode_b1_reddit/hidden_states.npz` |

**P5a bug history** (3 attempts):
1. Myriad 353764 (00:48) — `no hidden states extracted` after 105 task skips. Root cause: hardcoded `classifieds_task_{tid}` prefix in `run_stage4_format_variation_extract.py:177`, archive uses `reddit_task_*`
2. Myriad 353889 (06:26) — same failure, same root cause
3. Myriad **354382** (07:26) — fixed via commit 3d41953 (add `--site reddit` arg, default classifieds for backcompat)

**P2/P3 deferred** (2026-05-12 00:31 → 06:30, 3 attempts each):
- `snapshot_download` `thread_map` 8-worker concurrent download hits cas-bridge throttling/timeout
- Each attempt: get `HTTP 206 Partial Content` then concurrent.futures `result_iterator` raises (underlying worker exception masked)
- Cleanup 4×2.3G incomplete blobs to reclaim disk
- **Recovery plan**: tomorrow morning, single-thread CLI:
  ```bash
  HF_HUB_DOWNLOAD_TIMEOUT=600 huggingface-cli download Qwen/Qwen2-VL-7B-Instruct --max-workers 1
  HF_HUB_DOWNLOAD_TIMEOUT=600 huggingface-cli download microsoft/Phi-3.5-vision-instruct --max-workers 1
  ```
- Paper §5 generalization claim still defensible via P4 (selection-bias) + P5a/P5b (cross-site). P2/P3 are nice-to-have (family/size triangulation), not paper-critical.

**Expected verdict matrix** (most paper-grade interesting):
- P3 7B per-task variability < 4B per-task variability → H1' capacity-limit partially confirmed (training-distribution still creates shortcut, but consistency increases with size)
- P2 cross-family dichotomy holds → H1 is cross-family universal training prior
- P4 reverse-tier holds → not tier-selection-bias
- P5a reddit holds → cross-site universal

### 7.3.0 Exp 1 axis-2 layer profile v2 (2026-05-13 01:52 — three-axis hierarchy retracted)

`axis2_layer_profile_v2.md` + figure regen. Re-examine residual stream geometry per axis-isolated pair on **v2 NPZ** (Bug 1+2+5 corrected), full 37-layer cosine curves.

Cls site peak layers + magnitudes (v2):

| Pair | Group | L4 | L17 | L23 | L36 | Peak L | Peak gap |
|---|---|---:|---:|---:|---:|---:|---:|
| P-SoM↔SoM (image-axis ref) | image | 0.0375 | 0.0386 | — | 0.0416 | **L36** | 0.0416 |
| DOM↔P-text (text fmt) | axis-1 | 0.0035 | 0.0021 | — | 0.0047 | **L36** | 0.0047 |
| P-prompt↔P-SoM (text fmt) | axis-1 | 0.0034 | 0.0017 | — | 0.0048 | **L36** | 0.0048 |
| P-text↔P-SoM (prompt fam, flat) | axis-2 | 0.0002 | 0.0019 | — | 0.0088 | **L36** | 0.0088 |
| DOM↔P-prompt (prompt fam, hier) | axis-2 | 0.0002 | 0.0013 | — | 0.0068 | **L36** | 0.0068 |

Reddit cross-site replicates: P-text↔P-SoM L36 = 0.0069 (vs cls 0.0088), same rank-order, all axis-1 + axis-2 pairs peak L36 monotone-to-boundary.

**v2 retraction notes**:
1. **Single peak layer** (L36 monotone), not v1's "distinct peaks per axis" (image L17, text-format L23, prompt-family L23). The L23 peaks were v1 NPZ artifact.
2. **Magnitudes collapse**: axis-1 ~0.005 (v1 said 0.029), axis-2 ~0.009 (v1 said 0.011); image preserves ~0.04. New ratio image:text:prompt ≈ **8:1:1** (not 4:3:1).
3. **Reversed ranking**: v2 axis-2 magnitude (0.009) ≥ axis-1 magnitude (0.005). Both sub-permille and near noise-floor.
4. **Cross-site rank-order stable** (axis-1 ≤ axis-2 < image both sites). Pattern is real but tiny.

**Reframe**: "Three-axis hierarchy with distinct quantitative magnitudes 4:3:1" is RETRACTED. New framing: axis-1 + axis-2 are sub-permille in residual stream but probe-decodable (AUROC 1.0) AND lm_head-amplified (Exp 3 logit lens). Paper §5.7 prose updates to **cosine-causal disjoint** as hero, not magnitude hierarchy.

→ Paper §5.7 needs re-write (commit pending §52).

### 7.3.0b Axis-2 per-task fragility check v2 (2026-05-13 01:52 — /stress W2 defuse on v2 NPZ)

`axis2_per_task_fragility_v2_L17.md` + `axis2_per_task_fragility_v2_L23.md`. Re-run on v2 NPZ at both L17 (Exp 1 v2 image-axis peak) and L23 (v1 historical axis-2 peak reference).

**L23 v2 numbers** (closer to historical axis-2 peak):

| Pair | Site | Mean | Median | IQR | % > 0.005 | % > 0.010 |
|---|---|---:|---:|---:|---:|---:|
| **Axis-2 flat (P-text↔P-SoM)** | cls | 0.0073 | 0.0075 | [0.007, 0.008] | **100%** | 0% |
| **Axis-2 flat (P-text↔P-SoM)** | reddit | 0.0070 | 0.0071 | [0.006, 0.008] | **100%** | 0% |
| Axis-2 hier (DOM↔P-prompt) | cls | 0.0055 | 0.0055 | [0.005, 0.006] | 83% | 0% |
| Axis-2 hier (DOM↔P-prompt) | reddit | 0.0057 | 0.0056 | [0.005, 0.006] | 92% | 0% |
| Axis-1 ref (DOM↔P-text) | cls | 0.0039 | 0.0036 | [0.003, 0.004] | 8% | 0% |
| Axis-3 image (P-SoM↔SoM) | cls | 0.0324 | 0.0316 | [0.029, 0.035] | 100% | 100% |

**3 v2 findings**:
1. **Mean ≈ median** cross-site, distribution **NOT right-skewed**, not outlier-driven.
2. **IQR 极窄** (~0.001-0.002 wide, 5×narrower than v1 reported). All 24 tasks within tight band.
3. **Magnitudes all collapse to v2 sub-permille range** but per-task uniformity preserved. Axis-2 flat 100% > 0.005 cross-site (vs v1 100% > 0.010).

**/stress W2 attack defused on v2 data**: axis-2 cosine gap is uniform per-task signature at the new sub-permille level. Distribution is tight, not 2-3 outlier-driven. **What v2 changed**: the mean is smaller (0.007 vs v1 0.013), but the per-task uniformity argument holds — every task contributes to the sub-permille signal, not 2-3 outliers.

**Paper §5.7 v2 prose addendum**: per-task fragility argument is preserved at L23. Cross-site uniformity holds (0.0073 cls vs 0.0070 reddit, < 5% diff). Combined with logit lens 7-10× amplification, the sub-permille residual signal becomes the L21-L25 KL signal — both uniform per-task.

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

| Week | Milestone | Deliverable |
|---|---|---|
| **Week 1** (now → 2026-05-18) | §5 v2 prose surgery (#52) + Zekun sync + cross-AI audit | section5_mechanism.md v2 (cosine-causal disjoint hero, image-side dichotomy) + /stress + /codex-stress passes + Zekun reply |
| **Week 2** (2026-05-19 → 25) | Cross-family pilot (Phi-3.5-Vision + Qwen2-VL-7B Method 4.2 v2) + reverse-tier Method 4.4 | Cross-family disjoint preserved? + paper §5 supplement evidence |
| **Week 3** (2026-05-26 → 06-01) | Method 4.5 launch (LA-HDMI or SAE per Zekun decision) | Pilot results + paper §5 §6-7 prose |
| **Week 4** (2026-06-02 → 08) | Paper §5 codex round + advisor review | Submission-ready paper §5 |

## 9. Connection to paper §1 + §6

- **§1 phantom routing space + 4-fold drop-in property** — completely independent of mechanism work, anchors Outcome / Macro / Efficiency dimensions. NOT in this folder; see `paper_planning.md` §1
- **§6 cost-aware routing** — Peale et al. 2026 uncertainty decomposition anchor adds theoretical layer to phantom routing space's empirical AUROC. Method 4.2 cosine gap could serve as "reducible uncertainty" signal in deployment

These two stay outside mechanism folder. Mechanism workspace is paper §5-specific.
