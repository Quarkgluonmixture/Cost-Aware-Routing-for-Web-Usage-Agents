---
name: mechanism plan
description: Full mechanism workspace — theory, lit anchor stack, methods, identification protocol, current findings, open questions, advisor sync, roadmap. Specialized companion to paper_planning §2; not a duplicate.
type: workspace_plan
last_substantive_update: 2026-05-12
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

### 1.3 Image-axis peak-layer dichotomy (Mirage mechanism signature)

Method 4.2 reveals image-axis cosine-gap peak shifts based on text format of the no-image side. Clean dichotomy, zero overlap across 8 image-axis pairs:

| No-image side text | Peak layer | Pairs |
|---|---|---|
| AXTree (hierarchical) | **L04** | DOM↔Vision, DOM↔SoM, P-prompt↔Vision, P-prompt↔SoM |
| [SOM_MARKS] / flat | **L17–L36** | P-text↔Vision, P-text↔SoM, P-SoM↔Vision, P-SoM↔SoM |

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

### 5.4 Image-axis peak-layer dichotomy (Method 4.2, 8 pairs)

`docs/checkpoints/mechanism/results/layer_axis_emergence.md`. AXTree-no-image side → L04 peak (4/4); [SOM_MARKS]-no-image side → L17–L36 peak (4/4). Zero overlap. Mirage Effect mechanism signature.

### 5.5 H1 test: flat-list format variation (Method 4.2 extension, 2026-05-12)

`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.

## 6. Open questions (paper-grade gaps)

| Q | Status | Next action |
|---|---|---|
| ✅ Method 4.4 v2 full 48-cell sweep — sweet spot stable? | **Closed 2026-05-11 22:00**: L17 α=5 smoke 0.44 → full 0.16 (smoke variance artifact). **Real sweet spot L33 α=10 H-mean 0.33** | — |
| ✅ H1 test: do all flat-list formats trigger shortcut? | **Closed 2026-05-12 00:00**: YES, including hash-ID + plain-sentence controls. AXTree-DOM is sole defeating format | — |
| Reverse-tier 15 tasks vs strong-tier 24 — does L33 + H1 finding generalize beyond selection bias? | Med-High | qsub Stage 4 multimode + format variation with --tier reverse |
| Cross-site Method 4.2 — does cls finding replicate on reddit? | High | qsub Stage 4 multimode on B1 reddit (1 cell, ~1h on Myriad) |
| LA-HDMI vs mean-diff — does gradient steering beat 0.33 ceiling? | Med | Pending Zekun reply + attribution decision |
| SAE feature steering feasibility — is 1-2 week self-training Qwen3-VL-4B SAE worth it? | Low-Med | Depends on Zekun reply + paper §8 prose direction |
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
| **P4** | cls reverse-tier (selection-bias defense) | Myriad 353763 | qw 16h+ | `stage4_format_variation_b1_cls_reverse/` |
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

### 7.4 Decisions pending

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
