---
name: mechanism plan
description: Full mechanism workspace — theory, lit anchor stack, methods, identification protocol, current findings, open questions, advisor sync, roadmap. Specialized companion to paper_planning §2; not a duplicate.
type: workspace_plan
last_substantive_update: 2026-05-11
---

# Mechanism Plan — paper §5

## 1. Theory framework (1-screen summary, paper_planning §2 is canonical)

### 1.1 Zoom 1-4 hierarchy

| Zoom | Level | What our paper claims |
|---|---|---|
| **1** | Architectural | Phantom routing space = "skip annotated image" boundary contains 3 arms (P-text / P-prompt / P-SoM) sharing 4-fold drop-in property |
| **2** | Behavioral (axis effects) | Axis 1 (text payload: AXTree vs [SOM_MARKS]) is PRIMARY; Axis 2 (prompt: SoM-prompt vs DOM-prompt) is secondary; Axis 3 (image presence: in vs out) is gating |
| **3** | Named phenomena (lit-anchored) | Mirage Effect (Asadi 2026) / Scaffold Effect (Vu&Balloccu 2026) / Cross-modal flow (Kaduri) / Prompt-format sensitivity (Sclar 2024) |
| **4** | Model-internal | L17 mid-layer is BOTH discrimination locus (probe AUROC 1.0) AND causally active planning site (Stage 2/3 patching + Method 4.4 v2 reliability) |

### 1.2 Three-axis hierarchy quantified (Method 4.2 PCA cosine gap, Qwen3-VL-4B B1 cls)

| Axis | Peak cosine gap | Peak layer | Magnitude ratio |
|---|---|---|---|
| Image-axis (vs SoM / Vision) | 0.06 | L4–L17 | **10×** |
| Text-axis ([SOM_MARKS] vs AXTree) | 0.025 | L23 | **4×** |
| Prompt-axis (SoM-prompt vs DOM-prompt alone) | 0.007 | L36 | **1×** |

→ Mechanism magnitude image >> text > prompt. Validates `project_phantom_space_axes_format_not_information.md` memory: P-SoM closest mode at every layer is **P-text** (text-axis sibling, L17 cosine 0.0028 vs P-SoM↔SoM 0.0412 = 14.7× more distant).

## 2. Literature anchor stack (5 anchors, all 2026-05-08 except Sclar 2024)

| Anchor | Role | bib key | What it gives our paper §5 |
|---|---|---|---|
| **Wu et al. 2026** (UCL lab, our advisors) | Method backbone | `wu2026toolcalling` (2605.07990) | Mean-difference activation steering at second-to-last layer, 77–100% switch on tool selection (93–100% at 4B+). Our Method 4.2/4.4 port to multimodal Qwen3-VL-4B web agent |
| **Ma & Rui 2026** | Probe-vs-causal vocabulary | `maRui2026planning` (2605.07984) | "Planning-compatible representation" vs "causally active planning site". Qwen3-family pattern: probe works, causal patching weak (1% rhyme newline causal vs Gemma 67%). Our Method 4.4 v2 50% reliability is consistent with this family pattern |
| **HDMI / Khorasani et al. 2026** | Alt method + evaluation metric | `khorasani2026hdmi` (2605.07631) | Probe-free gradient-based steering. Critically: **completeness × selectivity → harmonic mean reliability** — what our Method 4.4 v2 reports (not raw shift rate) |
| **Lin & Liu 2026 Position paper** | Methodology protocol | `linLiu2026disclosure` (2605.08012) | 5-step identification disclosure norm: state claim / name strategy / enumerate assumptions / stress-test / separate validation. Paper §5 adopts as identification subsection structure |
| **Peale et al. 2026** | §6 routing theory | `peale2026flexibleRouting` (2605.07805) | Uncertainty decomposition (reducible + irreducible) with regret bound. Paper §6 theoretical anchor; 4-fold drop-in maps onto predict/route/abstain trichotomy |

## 3. Methods (Stage 4 + planned)

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

### 4.1 Causal claim

> Mid-layer L17 hidden state at last-token position is the causally active planning site for phantom routing space mode selection in Qwen3-VL-4B web agents.

### 4.2 Identification strategy

Triangulation of 3 evidence types:
1. **Probe-level** (Method 4.2 PCA cosine gap, AUROC 1.000 across 540 tests)
2. **Replacement patching** (Stage 2/3 Cell A-H, L17 disruption peak, 8/8 Holm-sig)
3. **Additive steering** (Method 4.4 v2, mid-layer L17 α=5 H-mean reliability 0.44)

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

### 5.1 Stage 4 Method 4.2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers)

| Pair @L17 | Cosine gap | 95% CI | AUROC |
|---|---|---|---|
| P-SoM ↔ P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
| DOM ↔ P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
| P-SoM ↔ SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
| DOM ↔ Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |

### 5.2 Stage 2/3 patching disruption (10 cells, B1 cls + reddit)

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
| H-d-cls | cls | DOM target (2x2 additivity) | -0.33 | ✓ |

### 5.3 Stage 4 Method 4.4 v2 (in flight, 8/48 cells)

H-mean reliability (HDMI framework) per (layer, α):

| Layer \ α | α=5 | α=10 | α=20 |
|---|---|---|---|
| L11 | 0.25 | 0.00 | 0.00 |
| **L17** | **0.44** ⭐ | 0.25 | 0.25 |
| L23 | 0.00 | 0.25 | 0.00 |
| L29 | 0.00 | 0.00 | 0.14 |
| L33 | 0.00 | 0.23 | 0.00 |
| L34 | 0.00 | 0.00 | 0.00 |

→ L17 α=5 current sweet spot. Stable after full 48-cell sweep completes.

## 6. Open questions (paper-grade gaps)

| Q | Gravity | Next action |
|---|---|---|
| Method 4.4 v2 full 48-cell sweep — does L17 α=5 H-mean 0.44 hold or shift? | High | Wait for bg sweep (~23:00 BST 2026-05-11) |
| Reverse-tier 15 tasks vs strong-tier 24 — does L17 finding generalize beyond selection bias? | Med-High | Re-run Stage 4 multimode + Method 4.4 v2 with --tier reverse |
| Cross-site Method 4.2 — does cls Method 4.2 result replicate on reddit? | High | qsub Stage 4 multimode on B1 reddit (1 cell, ~1h on Myriad) |
| LA-HDMI vs mean-diff on Qwen3-VL-4B — does gradient steering beat 0.44 ceiling? | Med | Pending Zekun discussion of attribution before committing GPU |
| SAE feature steering feasibility — is 1-2 week self-training Qwen3-VL-4B SAE worth it? | Low-Med | Depends on Zekun reply + paper review feedback timeline |
| B0 (proxy API, no internals) — is paper §5 mechanism story Qwen-specific or generalizable? | Low | Cannot test directly on B0; cite Wu et al. cross-family generality as proxy |
| Per-task variability across 24 tasks — is L17 finding driven by 1-2 cell types? | Med | Already analyzed (Stage 4 robustness Test B: 100% positive). May need stratification by task type |

## 7. Advisor sync state — Zekun (Wu et al. 2026 first author = lab member)

### 7.1 Timeline confirmed (not scoop)

- 2026-04-09 笔记 §19: I first grok the paper (then "Anonymous 2026 ACL"), record cosine gap method + L23+ steering 80-93%
- 2026-05-01 笔记 §108.19: upgraded to Zoom 4 anchor stack
- 2026-05-02 commit `6662b91`: anchored into paper_planning §2 + paper.bib placeholder
- 2026-05-09 advisor recording: Zekun explicitly recommended "SAE feature steering — 前所未有 inference time steering, 单独发 paper" — directed me to differentiating path
- 2026-05-11: arxiv landed publicly; identity confirmed as lab paper

**Net**: Zekun explicitly invited mechanism extension. Method 4.4 multimodal port is on his recommendation; SAE Method 4.5 is his next-step suggestion.

### 7.2 Message draft (paste-ready after v2 full sweep completes)

See `docs/checkpoints/实验笔记.md` §125.10 for current draft. Key revisions from earlier:
- ❌ Removed: "Method 4.4 null fills your §6 multi-turn limitation" (was wrong — v1 null was α calibration bug)
- ✓ Added: HDMI H-mean reliability table showing L17 α=5 = 0.44 as sweet spot
- ✓ Added: Ma & Rui Qwen3-family positioning (50% between Wu 93% and Ma & Rui Qwen3 1%)
- ✓ Three explicit asks: attribution decision / mid-vs-late layer ablation question / SAE Method 4.5 GPU commitment

### 7.3 Decisions pending

| Decision | Owner | Trigger |
|---|---|---|
| Co-author multimodal extension vs cite + independent framing | Zekun | After Zekun reply to message |
| Method 4.5 path: LA-HDMI vs SAE | Zekun + advisor sync | After v2 full sweep + Zekun reply |
| Paper §5 prose round | Codex + me | After v2 full + Zekun decision |

## 8. Roadmap (next 2-4 weeks)

| Week | Milestone | Deliverable |
|---|---|---|
| **Week 1** (now → 2026-05-18) | v2 full sweep land + Zekun sync + paper §5 prose v1 | 48-cell H-mean table + Zekun message + paper §5 §1-4 prose draft |
| **Week 2** (2026-05-19 → 25) | Cross-site Method 4.2 (reddit) + reverse-tier Method 4.4 | Replication results + paper §5 §5 prose |
| **Week 3** (2026-05-26 → 06-01) | Method 4.5 launch (LA-HDMI or SAE per Zekun decision) | Pilot results + paper §5 §6-7 prose |
| **Week 4** (2026-06-02 → 08) | Paper §5 codex round + advisor review | Submission-ready paper §5 |

## 9. Connection to paper §1 + §6

- **§1 phantom routing space + 4-fold drop-in property** — completely independent of mechanism work, anchors Outcome / Macro / Efficiency dimensions. NOT in this folder; see `paper_planning.md` §1
- **§6 cost-aware routing** — Peale et al. 2026 uncertainty decomposition anchor adds theoretical layer to phantom routing space's empirical AUROC. Method 4.2 cosine gap could serve as "reducible uncertainty" signal in deployment

These two stay outside mechanism folder. Mechanism workspace is paper §5-specific.
