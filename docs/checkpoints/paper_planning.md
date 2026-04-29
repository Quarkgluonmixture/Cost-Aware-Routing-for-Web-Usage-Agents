# Paper 1 Strategy & Notes (Phantom-SoM)

> **Paper-level strategy notebook** for the毕设 paper.
> 含 theory framework / findings 列表 / risks / cascade / router design /
> advisor align checklist. 内容会随 paper writing 逐步落到 paper drafts.
>
> **职能分工**:
> - **paper_planning.md** (此文档): paper strategy, theory, findings, risks
> - **next_steps.md**: action ledger (active processes, codex queue, next 3 actions)
> - **paper drafts** (`docs/analysis/paper_drafts/`): final paper prose
> - **实验笔记** (`docs/checkpoints/实验笔记.md`): time-order chronicle (历史 record)
>
> **Last updated**: 2026-04-29

---

## §1 Paper Hook + Tagline

**Core finding**: Phantom-SoM (SoM prompt + `[SOM_MARKS]` text + no image) is a **hidden 4th routing arm** for web agents with **4-fold drop-in property**:

| Drop-in property | Evidence |
|---|---|
| (a) **Cost ≈ DOM** | `[SOM_MARKS]` 是 AXTree regex filter, 不需 bbox/image (验 `som.py::_extract_text_marks` line 24); text token ±7% (3437 vs 3661 reddit / 3008 vs 2948 cls) |
| (b) **Latency ~50% lower** | cls SoM p95 74s vs Phantom-SoM 18.2s = **4× faster** (no image encoding stage) |
| (c) **Signal AUROC ≥ baseline** | 5-mode 全 `overall_usable=True`; red P-text verbalized 0.793 是 5-mode 最高 (超 baseline 0.766) |
| (d) **Drop-one oracle 1.7-3.3pp** | red Phantom-SoM 3.33pp drop-one (≥ SoM 1.90pp); cls 2.56pp |

**Paper one-liner (for advisor pitch)**:
> "Phantom-SoM identifies a hidden text-only routing arm in SoM-style web agents that achieves DOM-level cost and ~50% lower latency while contributing 1.7-3.3pp drop-one oracle value. The arm is created by skipping the marked-image draw and image-token inference path — no model retraining, no prompt change, no infrastructure overhead. We characterize its mechanism via 3-axis ablation (text-payload structure × system prompt × image), explain its site-modulated effect (cls visual-rich win for SoM, red text-dominated win for Phantom), and demonstrate routing infrastructure drop-in (signal AUROC ≥ baseline)."

### Cascade design (token-monotonic, paper Section 6)

```
DOM       (~3K AXTree text + DOM prompt + 无图)        ← cheapest text
  ↓ axis 1: AXTree → [SOM_MARKS] flat (text 结构 swap, ~3K both, prompt 不变)
P-text     ([SOM_MARKS] flat + DOM prompt + 无图)
  ↓ axis 2: dom_prompt → som_prompt (system prompt swap, 0 data token)
P-SoM     ([SOM_MARKS] flat + SoM prompt + 无图)
  ↓ axis 3: + image (~1.5K image embedding tokens)
SoM       ([SOM_MARKS] flat + SoM prompt + 有图)        ← highest text+image
```

**Order rationale**: Step 1+2 都 **0 增量 token**，第 3 步才付 image embedding tax — token-monotonic cascade，trigger router 不需要"先加再删"。Vision 是另条独立路径（image-only, no text），适合纯 visual task。

---

## §2 Theory Framework — 3-axis Hierarchical (validated)

### 5-mode 选择基于 2×2×2 design cube 对角路径

3 axes 形成 2×2×2 = 8 hypothetical modes，paper 5-mode 是其中**沿对角 axis-by-axis 路径**的 5 个 self-consistent 节点 (DOM → P-text → P-SoM → SoM + Vision 独立 image-only arm)。

**4 个 mismatched hybrid 故意排除** (e.g. AXTree + SoM-prompt + 无图)：
- 没 LLM 机制 — SoM prompt 指 `[SOM_MARKS] N` 但 AXTree 用 DOM accessibility ID（不同 ID 系统 → mismatched parsing）
- Confuse agent — `click [42]` 解析歧义，action selection 必错
- 不 clean axis 2 ablation — confound prompt-effect with ID-system-parsing-effect
- 跳出 token-monotonic cascade — token 数与 P-text 同但 SR 必差（mismatched parsing tax）

Paper Section 3 footnote 用此 defense reviewer "why 5 not 8".

### Axis 1: Text payload structure (PRIMARY, first-order SR effect)

```
AXTree (hierarchical) vs [SOM_MARKS] (flat indexed) → action surface + trajectory basin
→ Phantom modes 获得 routing arm (因为 [SOM_MARKS] obs structure)
```

注意：这个 axis 是 **text payload 结构**（agent 看到的 obs 文本），不是抽象的"模型表征"。Token 数大致不变（~3K both），但 layout / parsing pattern 不同。

**LLM mechanism**:
- Token distribution shift (hierarchical metadata vs flat indexed list)
- In-context learning bias (pretraining data context)
- Long-context attention degradation (Liu 2023 "Lost in the Middle")
- Output format priming

**Evidence**: fig3 strategy gradient (reddit verified §103 N=48, cls live extension), fig5 category × mode

### Axis 2: Prompt (multi-dimensional task-conditional decision prior)

**Falsified hypothesis**: "prompt 只影响 commitment confidence" (codex `5821387` falsifies)

**Replaced theory**: Prompt acts as task-conditional decision prior over:
- (a) search phrasing
- (b) candidate selection (ranking/disambiguation)
- (c) backtracking strategy
- (d) commitment confidence (FP gap evidence, **subeffect not唯一**)

**Evidence**:
- P-text ∩ Phantom-SoM Jaccard 0.45-0.54 (task pool 显著 disjoint despite same SR)
- 6 case studies (codex `5821387` phantom_dom_vs_som_diagnostic.md)
- N=48 verified anchor: Phantom-SoM FP gap 2.08pp vs P-text 6.25pp

**Lit support**: Persona priming (Salemi 2024), in-context learning (Min 2022), Sclar 2024 prompt format sensitivity, Mishra 2022. Task-pool divergence Jaccard <0.5 是 paper unique empirical finding.

### Axis 3: Image (8-channel multi-dimensional, codex `7106d2e` validated)

**Falsified hypothesis**: "Visual-hijack 是唯一 image effect" (codex `7106d2e` falsifies)

**Replaced theory**: Image is **bidirectional modality fusion** with multiple sub-channels:

**Helping channels (4)**:
- 3a Spatial grounding (cls 5 tasks)
- 3b Visual context disambiguation (cls 13 / red 2)
- 3c Element disambiguation (cls 10)
- 3d State/action recognition (cls 1 / red 6)

**Harming channels (6, refined per user critique 04-28)**:
- **3e False visual confidence (image-over-text)** [MAIN red 60%, 9/15 failures]
  - Mechanism: image-text alignment pretraining bias → premature commit
  - Lit: Li POPE 2023, Yang 2024, Yu 2024 (object hallucination)
- **3f Text-over-vision fallback fail** (反向 modality bias) [cls task 24 verified]
  - Mechanism: language prior dominance → ignore actual image content
  - Lit: ⭐ Tong 2024 "Eyes wide shut" (NeurIPS), Bitton-Guetta 2023
- **3g Visual saliency hijack on image content** [red task 0/167 verified]
  - 实测: 15× cycles same image link, element_id +2846 each cycle (page reload 不脱)
  - 与 SoM density 不直接相关 (mark_count 117 ≠ outlier vs P95 127)
- **3h SoM occlusion** [§100 ground truth, B0/B1 都 affected]
  - 量化: B0/B1 reddit_task_6 mode-SoM 18%/15% vs NoMarks 78%/75% → **-60pp OCR**
- **3i SoM numeric label attention hijack** [§100 verified, density-dependent]
  - 量化: B1 num_ids 0→**446** at 128 marks; mode-WithText 立即降 0
- **3j Visual misdirection** (visual saliency drift)

### Site-modulated framing (LLM-level explanation)

```
Task 需 vision (cls visual-rich):  3f text-over-vision 主导失败 → Phantom-SoM 失败
Task 不需 vision (red text-dom):    3e image-over-text 主导失败 → SoM 失败

Net effect:
  cls: helping 29 vs harming 13 → image win +6.84pp adj SR
  red: helping 8 vs harming 15 → image lose -3.33pp adj SR
```

### Site mechanical substrate (full characterization, 2026-04-29)

Each VWA/WA site has distinct **mechanical substrate** that determines which axis dominates. Section 5 narrative organized as `site × axis × LLM-mechanism` 3-way table, not just "axis effect on aggregate". Source for detailed per-site failure analyses: `docs/analysis/vwa_<site>/B*_{DOM,SoM,Vision}_digest.md` (9 files per site, manually + codex curated).

#### reddit (Postmill, N=210)

| Aspect | Detail |
|---|---|
| Information structure | Forum hierarchy (forum → posts → comments) |
| Navigation affordance | Sidebar `f/<forum>` links + search box (text-rich) |
| Image role | Content (post attachments), NOT navigation affordance |
| Mechanically dominant axis | **Axis 1 (text payload structure)** |
| Mechanism | AXTree hierarchical embeds sidebar in deep tree → search-box becomes shortcut → search-loop pathology. [SOM_MARKS] flat list makes sidebar `f/<forum>` directly clickable → DOM digest §2.1 search-loop 29.6% failure → P-text reduces |
| Image axis sub-effects | Layer 1b axis 3 small (5/5 reddit metrics show image effect d_z<0.16) — image is content not navigation, helping/harming roughly balanced |
| Site-specific failures | `fail_max_steps_search_repeat` (DOM 13.8%), `fail_no_progress` (22.4%), `fail_finish_eval_mismatch` (23.8%, read-and-report tasks) |
| Source digest | `docs/analysis/vwa_reddit/B0_DOM_digest.md` etc. |

#### classifieds (OSClass, N=234)

| Aspect | Detail |
|---|---|
| Information structure | Product listings + categories + search results |
| Navigation affordance | Category dropdown (`select_option`) + search box (intrinsic — most cls tasks are search) |
| Image role | **Product identity** (visual disambiguation critical for color/style) |
| Mechanically dominant axis | **Axis 3 (image)** — Layer 1b cls image axis dominates 5/8 metrics (h=+0.57 finish rate, d=−0.42 action repeat) |
| Mechanism | OSClass query routing (`/index.php?page=item&id=N`) means URL-path is uninformative — visual product comparison required for "find blue motorcycle" tasks. Image absence → P-SoM cls collapses toward DOM (Layer 1a 6/8 cells DOM-like). Image axis recovers at SoM. |
| Axis 1 sub-effects | Smaller; axis 1 (DOM↔P-text) on cls path-only Jaccard 0.904 (path-level same), 0.66 with query (semantic page-id divergence). Reveals macro-vs-micro mismatch: aggregate macro DOM-like but per-task page selection differs |
| Site-specific failures | Latent visual attribute (e.g. "red blanket" without ref image, A2 64% per `codex_audit_shopping_A_refined.json`), aggregation (least/most/cheapest, A3 35%), category navigation (case study task 12) |
| Source digest | `docs/analysis/vwa_classifieds/B0_findings.md`, `_DOM_digest.md` etc. |

#### shopping (Magento, N=466)

| Aspect | Detail |
|---|---|
| Information structure | Product pages + cart + checkout + admin (largest, most complex) |
| Navigation affordance | Product browsing + form interactions (custom-options swatch / qty / cart actions) + admin panel |
| Image role | **Product identification + visual variant selection** (color swatches partially DOM-readable, partially visual) |
| Mechanically dominant axis | **Axis 1 + Axis 3 mixed**, plus axis 2 prompt matters for form-action vs retrieval task split |
| Mechanism | Magento custom-options form interactions (swatch radio / select_option / qty) require precise element selection. §105 swatch state-change bug discovered 04-29 affects 2.4% tasks (DOM/SoM, not Vision). Visual-rich product variants need image axis for color/style disambiguation |
| Axis 1 sub-effects | Form action tasks need select_option for dropdowns; product retrieval needs visual ID |
| Site-specific failures | Aggregation (>50%, A3 dominant per `codex_audit_shopping_A_refined.json`), latent visual attribute (A2 41%), form-stall (swatch loop §105), admin-flow tasks |
| Site-specific quirks | Magento FPC cache full-page-cache requires hook + post-restart curl; custom-option radio swatch bug; review form ratings same bug pattern; long product comparison (12 items × 10 fields per Magento aggregation tasks) |
| Source digest | `docs/analysis/vwa_shopping/` (sparse, 跑中) + `codex_audit_shopping_A_refined.json` (A1/A2/A3/A4 sub-classification) + §105 swatch_form_change_audit.md |

#### Mechanism three-way table (Section 5 narrative scaffold)

```
                    reddit          classifieds         shopping
                    ─────────       ────────────        ──────────
Axis 1 (text)       PRIMARY         secondary           secondary (form-action)
                    sidebar→loop    page-id semantic    select_option matters
                                    divergence

Axis 2 (prompt)     macro driver    type/selfcorr only  prompt × text task split
                    of search/type  (cls aggregate masked)

Axis 3 (image)      weak/balanced   PRIMARY (5/8)       PRIMARY (visual variant)
                    image=content   image=affordance    image=ID + variant select

Site failure mode   search-loop     latent visual /     aggregation /
                    eval-mismatch   aggregation         form-stall / swatch
```

Section 5 prose 用此 3-way table 组织: per-site axis-by-axis mechanism, citing 4-Layer evidence + per-site digests + LLM-level mechanism (axis 1 attention shift / axis 2 task-conditional decision prior / axis 3 bidirectional fusion).

### Capability layer (B0 vs B1, lazy minimization §101.九)

**Lazy Minimization Hypothesis** (4B small VLM signal selection):
```
优先级: 数字标签 (高对比) > 文本 (结构化 token) > 截图内容文字 (低对比 + 遮挡)

物理解释 (capability-aware routing 的 mechanism):
  对 4B small VLM, 视觉处理成本/收益比更差
  → text-over-vision bias 在 small VLM 更强 (与 Asadi 2026 anchor 一致)

支持证据 (probe 全 align):
  B1 SoM 高密度 → num_ids 446 (数字 = easy)
  B1 WithText num_ids → 0 (文本可用就忽略截图)
  B1 NoMarks ≈ B0 NoMarks (视觉本身可用, 只是不优先)
```

**B0→B1 cross-site shift** (cross_site_pattern_consolidation `ab86019`):
- SoM hijack flip cross-site: cls **+50.0pp**, red **+33.3pp** (vs aggregate +43.7pp)
- Capability-modulated amplification of harming channels

### Cross-axis interaction LLM mechanism

```
Repr × Prompt: 同 obs token, 不同 prompt prefix → attention(obs|prompt) 不同
Repr × Image: 同 prompt, image vs no-image → cross-modal vs pure self-attention
Prompt × Image: SoM prompt + no image (Phantom-SoM) = "phantom prompt" mismatch
Site × axis: cls visual-bound → image helping dominate; red text-dom → harming dominate
```

### Paper contribution position (Section 5 framing)

**不是发现新 LLM mechanism** (每 axis 都有 prior literature). Contribution 是:
1. **Systematic isolation** (2x2 ablation matrix)
2. **Joint quantification** (Jaccard 0.33-0.55 disjoint task pools)
3. **Site-modulated framework** (cls vs red natural ablation)
4. **Drop-in deployment claim** (4-fold drop-in property)
5. **Paper-grade clean re-run protocol** (watchdog auto-clean + 重跑 verify)

→ Section 5 prose 应当 frame 成 "we don't propose new LLM mechanisms; we systematically decompose them in web-agent multi-step setting + provide drop-in deployment evidence".

### Literature gap 5-dimension (§103 anchor)

| Dimension | 是否有人 isolate marks-text? | 关键 papers |
|---|---|---|
| A. SoM lit | **No** — 全 bundle text+image | Yang 2023, Magma, Ferret-UI 2 |
| B. Representation routing | **No** — 现 routing model-level/modality-level | RouteLLM, Avenir-Web |
| C. AXTree vs flat list | **No** — head-to-head 缺失 | FOCUSAGENT, VWA baseline |
| D. Prompt format sensitivity | **Yes (theory anchor)** — 但无 web agent 应用 | Sclar 2023, Mishra 2022 |
| E. Cost-aware web agent | **No** — focus prune 不 reformat | FOCUSAGENT pruning, ModServe |

**Closest prior** = FOCUSAGENT (text 压缩) + Yang 2023 (SoM with image). 本工作 = unprecedented synthesis.

完整 deep research: `docs/literature/The Novelty and Efficacy of Set-of-Mark Text as an Independent Representation Routing Arm in Web Agents.md`

---

## §3 Findings — 4-Layer Evidence + Mechanism Framework (重组 2026-04-29)

> **重组动因 (§105)**：之前 10 条 finding 是 flat list，paper 写作时不好定位"哪个证据支持哪个 claim"。重组为 **4-layer framework** —— 每个证据进自己的层，每个 paper claim 引用层 (e.g. "Layer 0d Jaccard 0.447 supports routing-arm complementarity")。**所有原 10 条 finding 都映射到对应 layer，未删除**（见末尾索引）。

### Layered framework 概览

```
Layer 0  Outcome           哪些 task 成功 / 哪些 mode 在哪些 task 上 win
Layer 1  Macro Behavior    agent 平均怎么 act（action-type 频率）
Layer 2  Micro Behavior    agent per-step 决策（点哪个元素 / 走哪些页 / 搜什么词）
Layer 3  Efficiency        cost / latency / carbon (4-fold drop-in property)
```

每 layer 内部 sub-evidence 都标注 source artifact + current 数字（实时 live）。

---

### Layer 0 — Outcome（task 成功 / 路由 arm 证据）

| Sub | 内容 | Source artifact | 当前数字（B0, FRESH 04-29） |
|---|---|---|---|
| **0a** Aggregate raw + adjusted SR per mode | summary_v2.json live | live | red P-SoM **adj 13.81%** > all baseline; red P-text **12.38%** > DOM **9.52%**; cls SoM **21.37%** (best); cls P-text/P-SoM **adj 14.53%** ≈ DOM 14.10% |
| **0b** FP rate per mode | summary_v2.json (raw_succ 与 adj_succ 之差) | live | **red P-SoM 0.48%** (lowest, "honest commit"); cls P-SoM 1.28%; **§3-legacy finding 4 prompt-as-decision-prior 的核心证据** |
| **0c** Routing oracle uplift (3-mode → 4/5-mode) + drop-one | `phantom_lift.{md,csv}` | red 3→5: **+5.24pp** [2.38, 8.11] Wilcoxon p=0.0009 McNemar p=0.0005 ✅; cls +4.70pp [2.14, 7.69] p=0.0009 ✅. red drop-one P-text +3.81pp / P-SoM +3.33pp; cls P-text +3.42pp / P-SoM +2.56pp |
| **0d** Task-pool Jaccard (Scenario C sentinel) | `phantom_lift.md` | red P-SoM↔P-text **0.571** (≤0.7 safe ✅); cls 0.447 ✅. **核心 routing-arm 证据**：cls aggregate SR 同 (P-SoM≈DOM) 但 task-pool 0.53 disjoint —— same SR ≠ same routing pool |
| **0e** Per-category SR (4 cat × 5 mode heatmap) | `fig0e_category_mode_heatmap.png` ✅ live (rebuilt 04-29) | DOM-only Cat B (ref-image) systematic fail; cls Cat A counter-intuitive 8.54% < B 21.30% (codex audit refined → A1/A2/A3/A4 04-29) |
| **0f** Overlap depth (5-mode solve-pool depth distribution) | `fig0f_overlap_stacked_bar.png` ✅ live | red P-SoM 30 succ: 3 unique / 8 d2 / 7 d3 / 9 d4 / 2 d5; **§3-legacy finding 8** |
| **0g** Routing AUROC per mode (signal quality) | `auroc_cross_condition_summary.md` | red P-text 0.793 (5-mode max), P-SoM 0.720; cls P-text 0.737, P-SoM 0.728. **5/5 phantom `overall_usable=True`** —— paper 4-fold drop-in property (c) ✅ |

---

### Layer 1 — Macro Behavior（agent 平均怎么 act）

| Sub | 内容 | Source | 当前数字 |
|---|---|---|---|
| **1a** Tier 1 hook (3-mode coarse: DOM/P-SoM/SoM × 8 metric) | `axis_effect_size.py` (FRESH 04-29) + `axis_effect_size_report.md` | P-SoM "fully independent" cells: **red 4/8 vs cls 1/8**. cls P-SoM 主要"瘫向 DOM" (6/8 DOM-like) —— image axis 决定性, **印证 0d 的 task-pool 复杂性** |
| **1b** Tier 2a Mechanism cascade (3 axes × 8 metric) | `axis_effect_size.py` | **6 antagonistic pairs** (red scroll/text↔prompt 反向相消 / cls finish/prompt↔image 反向); cls **image axis 5/8 dominant** (h=+0.57 finish, d=−0.42 repeat); axis 1 在 macro 0/8 dominant (但 outcome 层 primary, 见 0c) |
| **1c** Strategy gradient (search-loop / type / scroll / selfcorr) | `fig1c_strategy_gradient.png` ✅ FRESH 04-29 全数据 | red DOM **search-loop 51.9%** → P-SoM 35.7% → SoM 31.4% (§3-legacy finding 3 升级版：从 §103 N=48 → N=210 全数据，原"5/5 metrics P-text=P-SoM"已 falsify) |

---

### Layer 2 — Micro Behavior（per-step 决策）

| Sub | 内容 | Source | 当前数字 |
|---|---|---|---|
| **2a** URL signature divergence | `axis1_microbehavior.{py,json,md}` (FRESH 04-29 codex + 我补 compound) | **axis 1 alone**: red Jaccard 0.573 / cls 0.904 (path-only). **compound DOM↔P-SoM**: red 0.481 / cls 0.885 path-only (但 cls path+query 0.66 —— OSClass 用 query routing). **决策真改了，aggregate macro 在 cls 上掩盖** |
| **2b** Target-page hit rate | `axis1_microbehavior.json` | red axis 1 +3.47pp; cls axis 1 +2.33pp; compound red −0.69pp / cls +1.74pp |
| **2c** Search-keyword reuse / repeat | `axis1_microbehavior.json` | red P-text vs DOM 重复 −0.633 (axis 1 减少死循环); cls P-text +0.077 (无 site 损失) |
| **2d** First-action divergence | `axis1_microbehavior.json` | red 21% / cls 14% tasks first-action type differ (axis 1) |
| **2e** Cross-site validity ratio | `axis1_microbehavior.json` `cross_site_validity` | **verdict: generalizes** (red 2.28, cls 1.02). cls 边界 —— 单独 axis 1 在 cls 上 micro≈macro，但 compound DOM↔P-SoM 在 cls 上 path+query Jaccard 0.66 强 divergence |

---

### Layer 3 — Efficiency（4-fold drop-in property）

| Sub | 内容 | Source | 当前数字 |
|---|---|---|---|
| **3a** Token cost per step (input) | `condition_summary_v2.json` | P-SoM ≈ DOM (~3K both); SoM +image embedding tax. **4-fold drop-in (a) cost ≈ DOM ✅** |
| **3b** Image embedding tokens (per step median) | `run_summary_collect.json` | red 733 / cls 1064 tokens; **P-SoM 省去这部分**, **§3-legacy finding 6** |
| **3c** Latency per step | `condition_summary_v2.json` | P-SoM ~50% of SoM latency (无 image inference). **4-fold drop-in (b) latency ~50% ✅** |
| **3d** B0 (API) vs B1 (local) deployment-class cost gap | `cost_per_mode.{json,md}` (FRESH 04-29) + `fig3d_cost_sr_frontier.png` | **B0 API ~$0.04/ep (Qwen3-VL-235B-A22B token cost)**; **B1 electricity-equivalent ~$0.0004/ep** (DGX Spark `avg_total_energy_kwh × $0.12/kWh` UK industrial rate). **Ratio ~100×** (red 98× / cls 105×) — **deployment-class gap, NOT capability/parameter ratio**. ⚠️ §103 / §3-legacy "30×" claim **superseded** by FRESH data. Paper presents both classes side-by-side, not a single multiplier. |

---

### Cross-layer Mechanism Chain（每个 axis 在哪些 layer 上 first-order）

| Axis | Layer 0 outcome 贡献 | Layer 1 macro signature | Layer 2 micro signature | Layer 3 cost |
|---|---|---|---|---|
| **Axis 1 (text payload)** | **PRIMARY** (red P-text +3.81pp drop-one over 3-mode; cls +3.42pp) | 0/8 dominant (但**红 scroll/selfcorr 是 antagonist canceller**) | red URL Jaccard 0.57 / cls 0.90 (axis 1 alone) — 在 reddit 改 WHERE 强 | 0 (text swap 不改 token 量) |
| **Axis 2 (prompt)** | secondary (red P-SoM 加在 P-text 上 +3.33pp; cls +2.56pp) | red 3/8 cascade dominant (search/type/scroll); cls 3/8 (type/selfcorr/click) | URL Jaccard 0.55 (axis 2 alone) | 0 (prompt swap 不改 token 量) |
| **Axis 3 (image)** | secondary (cls SoM 21.37% > P-SoM 14.53%, image 决定性 cls 上) | **cls 5/8 dominant** (finish h=+0.57 medium-effect 最强信号); red 3/8 dominant (efficiency cluster) | image 加上 = URL Jaccard 0.46-0.60 minor change | **+700-1100 image tokens** (Layer 3a 主要 cost source) |
| **Compound Axis 1+2 (P-SoM vs DOM)** | red SR delta +2.86pp aggregate; **cls SR delta 0.85pp 但 task-pool Jaccard 0.53** = routing-arm 证据 | cls macro 60-70% DOM-like 但 task-pool 0.53 disjoint —— aggregate 误导 | **path+query Jaccard cls 0.66 / red 0.48** —— per-step decision quality 真改了 | 0 |

---

### Evidence chain — paper claims → layer support

每个 paper claim 直接 cite layer + 数字：

| Paper claim | Layer support |
|---|---|
| **C1**: P-SoM is independent routing arm | 0a (red SR best), 0c (drop-one 3.33pp red / 2.56pp cls), 0d (Jaccard ≤ 0.6), 0g (AUROC ≥ baseline), 1a (red 4/8 macro independent), 2a (red URL Jaccard 0.48 micro divergence) |
| **C2**: 4-fold drop-in property (cost / latency / signal / drop-one) | (a) Layer 3a, (b) Layer 3c, (c) Layer 0g, (d) Layer 0c |
| **C3**: 3-axis hierarchical theory | Layer 1b (cascade decomposition), Layer 2 (axis-by-axis micro), Cross-layer table |
| **C4**: aggregate macro can mislead about routing potential (cls case) | Layer 1a (cls 6/8 DOM-like macro) + Layer 0d (cls task-pool Jaccard 0.53) + Layer 2a (cls path+query Jaccard 0.66) |
| **C5**: prompt as task-conditional decision prior (not commit-only) | Layer 0b (FP rate), Layer 0d (Jaccard 0.45-0.55 same-SR-different-pool), Layer 1b (cascade axis 2 dominant on red strategy metrics) |
| **C6**: image is bidirectional 8-channel modality fusion | Layer 1b (cls image axis 5/8 dominant), Layer 0e (codex audit category × mode), codex `7106d2e` channel decomposition |

---

### Mechanism chain — 三个机制阶段

```
Stage 1 (Outcome 层):    P-SoM 的 routing arm 价值在 task-pool complementarity (0d), 不在 aggregate SR
                         ↓ 为什么 P-SoM 拿到 unique tasks?
Stage 2 (Micro 层):       因为 axis 1+2 swap 改变了 per-step 决策 (2a 0.48-0.66 URL Jaccard)
                          ↓ 这些决策具体改了什么?
Stage 3 (Mechanism 层):  axis 1 改 text payload 结构 → 改 in-context attention pattern → 改 element selection 决策
                          axis 2 改 prompt 描述 → 改 task-conditional decision prior → 改 commit / search / 导航策略
                          axis 3 加 image → 改 visual disambiguation → 决定 cls 上的 finish rate
```

**关键 insight**: Layer 1 macro (action-type 频率) 是 downstream signal，单独看会误导 (cls case)。真正的 mechanism chain 是 Layer 2 micro (decision quality) + Layer 0 outcome (task-pool complementarity) 闭环。

---

### Honest framing (avoid over-claim)

- Phantom-SoM red **adj 13.81% > SoM adj 10.48%** —— 这次有数据，是 site-specific dominance（reddit 上）
- cls SoM **adj 21.37% 显著领先 P-SoM 14.53% (+6.84pp)** —— 反例必须明示, image 在 cls 是决定性 axis (Layer 1b 5/8 dominant 印证)
- 主 narrative: **site-modulated representation × prompt × image effects**, 不是 "Phantom #1 universal routing arm"
- Layer 1 macro 单独 weak on cls (1/8 fully independent) —— paper 必须用 Layer 0 task-pool + Layer 2 micro 一起讲，不能只 cite macro
- §103 N=48 "5/5 metrics P-text = P-SoM" 已 **superseded** by N=210 (FRESH 04-29 Layer 1c) — 早期 small-sample artifact

---

### Legacy index (原 10 条 finding 映射)

Naming traceability (04-29): completed filesystem run dirs now distinguish paper-facing text from SoM phantom arms:
`B0_phantom_*` completed runs became `B0_phantom_som_*`, and completed `B0_phantom_dom_*` runs became `B0_phantom_text_*`. Internal mode IDs and condition dirs remain unchanged (`phantom_dom` / `phase1_phantom_dom_router_0`, `phantom_som` / `phase1_phantom_som_router_0`) for backward compatibility with recorded JSONL.

| 原 finding | 映射到 layer |
|---|---|
| 1 DOM Cat B 系统性失败 | **0e** per-category heatmap |
| 2 Image 8-channel cross-capability | **1b axis 3 + 0e cls cat × image** |
| 3 DOM reddit search-loop 22.7% | **1c strategy gradient** (升级为 N=210 51.9%) |
| 4 P-text cls adj 14.53% ≈ DOM 14.10% | **0a + 0d** (same-SR-different-pool) |
| 5 P-SoM cost ≈ DOM cost | **3a** (4-fold drop-in (a)) |
| 6 Image tokens per step (red 733 / cls 1064) | **3b** |
| 7 B0 vs B1 cost gap | **3d** (修正 04-29: ~100× deployment-class gap, NOT 30× — see `cost_per_mode.md`) |
| 8 Phantom unique tasks (fig8) | **0f** overlap depth |
| 9 AUROC ≥ baseline | **0g** (4-fold drop-in (c)) |
| 10 Watchdog 100% pure | (data quality precondition, 不是 finding) |

新增 finding（§105 04-29）：
| 新 finding | Layer |
|---|---|
| **N1**: P-prompt 模式必要性（symmetric ablation, AXTree+SoM-prompt+无图） | 设计层 (§2 cube), 数据 pending B0 reddit 跑中 |
| **N2**: Tier 1 hook macro: red 4/8 cells fully independent / cls 1/8 (cls 主要 DOM-like) | **1a** |
| **N3**: 6 antagonistic mechanism pairs（4-level cascade vs 2-endpoint 比较的核心 paper value） | **1b** |
| **N4**: cls compound DOM↔P-SoM micro path+query Jaccard 0.66 | **2a** |
| **N5**: P-SoM cls aggregate SR ≈ DOM 但 task-pool 0.53 (12 unique successes) | **0d** (reframes "cls Phantom-SoM 失败" 为 "complementary not dominant") |
| **N6**: red P-SoM FP=0.48% lowest（最 honest commit） | **0b** |

---

### Evidence vs Explanation: framework 的真实定位（2026-04-29 反思）

4-Layer framework **不是 paper Section 4/5 的 narrative 结构**，是**分析 scaffold + future-data drop-in 索引**。明确两个层次：

#### 4-Layer = Evidence layer（paper Section 4）

观测 evidence: "在 mode/axis swap 下我们 observe 到什么 shift"
- Layer 0: 哪些 task 成功（SR / oracle / Jaccard / category / overlap / AUROC）
- Layer 1: agent 平均怎么 act（action-type 频率 cascade）
- Layer 2: per-step 决策怎么变（URL / target / keyword）
- Layer 3: 资源 footprint（cost / latency / carbon）

四个**正交 dimensions**（不是 hierarchical layer），从宏观 outcome 到微观 decision。Paper Section 4 是 evidence catalog，每个 sub-finding 引用一层数据 + figure。

#### LLM mechanism = Explanation layer（paper Section 5）

解释 evidence: "为什么 axis swap 产生这个 shift"——必须 **跨 layer** 同时 **site × axis × LLM-mechanism** 三阶交互：

```
观测 (evidence): reddit axis 1 swap → search-loop 51.9 → 35.7 (Layer 1c) +
                                       URL Jaccard 0.57 (Layer 2a) +
                                       SR uplift 4.76pp drop-one (Layer 0c)
解释 (LLM mechanism):
  AXTree (hierarchical, sidebar embedded in tree) → [SOM_MARKS] (flat indexed list)
  ⇒ attention pattern shift: sidebar forum link 在 flat list 显著
  ⇒ agent 直接 click forum link 而非 search-loop
  ⇒ trajectory 变短 + 决策准 + SR up
  ⇒ 横跨 Layer 1+2+0 evidence 的 single mechanism
```

不同 site 触发不同 mechanism (site × axis × LLM):
- **reddit text-heavy forum**: axis 1 主要影响 attention pattern (sidebar visibility)；image axis 几乎冗余
- **cls visual-rich product browsing**: axis 3 image 是 affordance（finish-rate h=+0.57 决定性）；axis 1 主要影响 ID-system parsing efficiency

**paper Section 5 narrative 由 mechanism 驱动**, layered evidence 作 underlying support — 不是按层组织 narrative。

#### Axis decomposition（diamond 完整后的 final form）

```
total observed effect (DOM → SoM endpoint)
  = main(axis 1, P-text alone via DOM↔P-text)
  + main(axis 2, P-prompt alone via DOM↔P-prompt)              [DIAMOND ENABLES]
  + main(axis 3, image alone via P-SoM↔SoM)
  + interaction(axis 1, axis 2)                                 [DIAMOND ENABLES]
  + ...higher-order interactions usually 0
```

P-prompt 是必需的，因为它是 **axis 2 在 AXTree-text context 下的唯一测量点**。如果 interaction term ≈ 0 → paper 写 "axis additive, independent first-order"；如果 interaction term ≠ 0 → honest disclose "axis 1 effect is modulated by prompt context"。任一 verdict 都比 cascade-only 多一个 quantitative claim。

#### Framework 的 future-data 弹性

所有 cells 自动落到 layer × site × axis × baseline 索引：

| Future data | drop-in 到 |
|---|---|
| B1 phantom cls/red 4-cell (P-SoM + P-text × cls + red) | Layer 0/1/2/3 × cls/red × B1 cells |
| B1 P-prompt cls/red (Tier 2) | Diamond axis 2 in B1 capability — Section 7 cross-capability |
| B0/B1 shopping 6-mode | Layer 0e per-category（shopping-rich audit categories）+ all layers shopping cells |
| WA B0/B1 6 sites × 5 modes | Cross-benchmark generalization (Section 7 main) |
| Claude Opus 4.7 5-mode | Cross-model boundary check (Section 7) |
| 其他 benchmark | Same scaffold, no rework |

`make analyze-layered` 是 idempotent 的——新数据 commit 后跑一遍 `layered_status.py` 自动 regenerate `layered_evidence_status.md` + 所有 figures。

#### Caveats / honest framing

- **N=234 cls underpower**: cls Layer 1 macro 弱信号可能是 statistical power not enough（needs ~800 task to detect d=0.2 small effect with α=0.05, β=0.2）。后续 shopping 466 + WA 480 数据可补强
- **"Layer" 命名不严格**：4 层是 orthogonal dimensions, 不是 hierarchical。命名沿用 "Layer" 是因 codebase 已 lock-down (Makefile / scripts), 不改回 "dimension"
- **不是所有 13 figures 进 paper**：Section 4 只 cherry-pick 5 个代表 figure (e.g. fig0c + fig0d + fig0g + fig1c + fig3d)，其他 supplementary
- **paper Section 5 可能简化**：若 codex prose 过分 layered cataloging，必须 mechanism-first restructure

---

### Mechanism Tier 1/2/3 escalation plan (Section 5 explanation methodology, 2026-04-29)

4-Layer evidence catalogs *what shifts*; Section 5 mechanism explains *why*. Three escalating tiers, only Tier 1 is currently feasible; Tier 2/3 execute on existing 实验笔记 §19 future-work plan once B1 GPU frees up.

#### Tier 1 — Behavioral mechanism (paper-ready now, B0+B1 data, no GPU work)

Per-task per-step decision-quality metrics, mode-invariant, computable from existing step JSONL:

| Metric | What it measures | Layer | Status |
|---|---|---|---|
| **E1** click-target Jaccard | per-task `(pre_url, post_url)` transition signature, mode-invariant + step-invariant | Layer 2 micro | 🟢 codex prompt ready (`mechanism_per_task_explanation.md`) |
| **E2** trajectory boundary | for symmetric-diff success tasks, first divergent step | Layer 2 micro | 🟢 prompt ready |
| **E3** confidence calibration cross-condition | ECE/MCE/Brier/AUROC per (model, site, mode), aggregating existing `analyze_confidence_calibration.py` per-run output | Layer 0b + Layer 1 | 🟢 prompt ready |
| **E4** action vocabulary distribution | full action_type × subtype frequency per cell (extends axis_effect_size's 4 metrics) | Layer 1 macro | 🟢 prompt ready |

Tier 1 deliverables: `scripts/analysis/mechanism_per_task.py` + `docs/analysis/cross_sites/mechanism_per_task.{json,md}`. Adds `make analyze-mechanism` target. ~80K codex tokens. **Trigger anytime**.

#### Tier 2 — Mechanistic interpretability (B1-only, executes 实验笔记 §19 future-work)

实验笔记 §19 已 documented "Tool Calling is a Linear, Steerable Circuit" (ACL 2026, Qwen3 4B verified) 适用于 P79: action selection 是线性电路, cosine gap 预测 92% 错误, L23+ steering 可 80-93% 准确率切换 tool。Section 5 paper-strongest mechanism evidence 走这条路:

| Metric | What it measures | Tooling | Status |
|---|---|---|---|
| **M1** B1 attention pattern probe | feed same task obs through B1 in DOM/P-text/P-SoM modes; extract attention to "forum sidebar link" / "search box" / "post title" tokens; measure shift across modes | `output_attentions=True` forward pass, ~2300 forwards | 🟡 blocked B1 GPU contention |
| **M2** B1 hidden state probing | layer L hidden state → probe "task will succeed"; PCA cosine gap (per §19) → AUROC vs logprob | `output_hidden_states=True` forward pass; PCA + LR | 🟡 blocked B1 GPU |
| **M3** Token-level decision attribution | next-action token distribution per mode; quantify "axis 1 改 token-level decision prior" claim | forward inference, no training | 🟡 blocked B1 GPU |

**Trigger condition**: B1 GPU 空 (~B1 phantom 4-cell chain done, ~30-40d ETA). 不需要重跑 environment — `~2300 task × ~12 steps = ~28K forward passes`, 离线 inference 单 GPU 可在 ~1-2 天 batch 跑完。Code 已部分存在: `analyze_confidence_calibration.py` 处理 logprob, 可扩展 `output_hidden_states/attentions` 提取。

**Paper value**: 比 Tier 1 行为分析更 mechanistic, reviewer 期望顶刊看到。直接对应 ACL 2026 Tool Calling lit。

#### Tier 3 — Causal mechanistic intervention (heavy, may be future paper)

| Metric | What it measures | Tooling | Status |
|---|---|---|---|
| **H1** Activation patching | DOM forward pass at (layer L, step S) → patch hidden state into P-text run → does behavior become DOM-like? | causal scrubbing infrastructure | 🔴 blocked B1 GPU + 1-2 weeks impl |
| **H2** Steering vectors | train PCA / linear probe to find "mode direction" in activation space; add steering vector at inference to induce mode-like behavior without obs/prompt swap | per §19 future work "L23 steering 修正 'know-but-cant-say'" | 🔴 blocked B1 GPU + advanced technique |
| **H3** Attention head ablation | systematic zero-out specific heads; find "axis 1 head" / "axis 2 head" responsible for mode-specific behavior | head-by-head intervention scaffold | 🔴 heaviest, possible split paper |

**Trigger condition**: 顶刊投稿 reviewer 要求 mechanistic 强化 OR 时间允许提前做。可能的 split: H1+H2 进 Section 5, H3 留 future work / paper 2.

**Paper value**: causal claim, 比 correlation-based mechanism (Tier 2) 更强. ACL/NeurIPS mechanistic interpretability track 期望.

#### 总体 Section 5 mechanism narrative cascade

```
Section 5 (顶刊版) 期望证据 stack:
  Tier 1 behavioral (E1-E4)  ← Section 5 fast-write, 现在 ready
  Tier 2 mechanistic (M1-M3) ← Section 5 strengthening, ~30-40d
  Tier 3 causal (H1-H3)      ← Section 5 顶刊 differentiator, optional
```

如果 deadline 紧, Tier 1 + Tier 2 already make Section 5 paper-grade. Tier 3 是 nice-to-have / split-paper option.

---

## §4 Paper Section Status (2026-04-29, 8 sections final scope)

| Section | evidence 质量 | 状态 | Hard blocker |
|---|---|---|---|
| 1 Intro | ✅ 已写 (786w + 4-fold drop-in framing + conservative framing) | done `62c1380` `ef29add` | — |
| 2 Background + paper.bib | ✅ 已写 (1514w, 16 entries) | done `206cd93` | 待 codex #10 expand to ~38 |
| 3 Definition + Ablation | ✅ 已写 (863w, token re-estimate corrected) | done `13b9608` `4d63c9f` `48db047` | — |
| 4 Empirical Findings | 🟡 80% (figures FRESH ✅ + B0 5-mode FRESH, prose 待 update) | data ready | codex #11 fresh prose (~30K) |
| 5 Mechanism | 🟡 90% evidence (3-axis × 8-channel × bidirectional × §100) | data 完整 | codex #13 prose (~50K, 待 #10 lit) |
| **6 Routing (Tier 1+2)** ⭐ NEW | 🟡 40% (signal AUROC ≥ baseline `9d7e99f`, infra scaffold) | scaffold ready | Tier 1 prototype (~3 天) + Tier 2 first-step trigger (~7-10 天) |
| 7 Generalization | 🟡 40% (B1 capability profile done) | partial | shopping (跑中) + WA + cross-model (Claude) |
| 8 Discussion + Implications (含 sustainability + 4-fold drop-in summary) | ❌ 未写 | end-stage | 全部 data done |

**Section 1-3 总 prose 3163 words** (paper-ready). Section 4 1725w draft 待 fresh data update. Section 5/6/7/8 待写.

### Section 6 Routing — 详细 outline

paper Section 6 不是 Section 7 Generalization 的子部分, 是**独立 contribution**. 内容:

```
6.1 Routing problem formulation
  - 5-mode arms: DOM / SoM / Vision / Phantom-SoM / P-text
  - Per-task feature space (instruction + browser meta + step-1 trigger signals)
  - Target: max adjusted SR / cost-aware / Pareto

6.2 Tier 1 — task-level oracle router (offline supervised)
  - Feature: TF-IDF (task instruction) + binary {has_ref_image, has_finish_string_match}
  - Model: Logistic regression / random forest (interpretable + small data)
  - Train/eval split: 80/20 of cls + red task pool
  - Result: routing pool oracle bound vs learned router gap

6.3 Tier 2 — first-step-trigger router (online cascade)
  - First step on cheaper mode (e.g. DOM) → trigger signal at step 1 → escalate to expensive mode
  - Trigger features: response confidence / action_diversity / behavioral signals
  - No test-time leakage (step-1 obs ≠ task feature)
  - Cost vs SR Pareto improvement

6.4 Routing infra drop-in property (4-fold #c)
  - Phantom modes signal AUROC ≥ baseline (5/5 usable, red P-text 0.793 highest)
  - 不需要 retrain signal extraction infra
  - paper claim: "router trained on baseline 可 directly extend to Phantom"

6.5 Visualization (4-fig stack, see §10)
  - Fig A: 3-panel multi-metric Pareto
  - Fig B: Cumulative SR vs Budget curve ⭐ (cost-aware 顶刊套路)
  - Fig C: Routing decision Sankey
  - Fig D: Per-task savings histogram
```

### Section 8 Discussion — 详细 outline (含 sustainability + green AI)

```
8.1 4-fold drop-in property summary
  (a) Cost ≈ DOM
  (b) Latency ~50% lower (cls SoM 74s → Phantom 18s)
  (c) Signal AUROC ≥ baseline (router infra drop-in)
  (d) Drop-one oracle 1.7-3.3pp

8.2 Site/capability-modulated mechanism
  cls visual-rich → image helping channels dominate (3a-3d)
  red text-dominated → image harming channels dominate (3e-3j)
  B1 amplify failure dimensions (lazy minimization, density threshold)

8.3 Sustainability + Green AI implications ⭐
  - cls Phantom-SoM latency 4× improvement (production-relevant)
  - Regional carbon sensitivity (fig9): Phantom-SoM advantage region-dependent
    (large for India 632 g/kWh / Poland 773; small for France 85 / Norway 29)
  - Multi-metric Pareto: cost + latency + carbon 三向 drop-in
  - Lit anchor: Strubell ACL 2019, Patterson 2021

8.4 Limitations + future work
  - Single benchmark family (VWA + WA), single backbone model family (Qwen + Claude)
  - Tier 3 online learning router 留 future work
  - Cross-model meta-policy (cross model family routing) 留 future
```

---

## §5 Final Scope + 顶刊概率

### Final scope (paper 完整版)

```
Benchmark: VWA 3 站 (cls 234 + red 210 + shop 466) + WA 3 站 (red 106 + shop 192 + sa 182)
           = 6 sites, ~1390 task per condition
Models:    B0 (Qwen3-VL-235B proxy) + B1 (Qwen3-VL-4B local) + Claude Opus 4.7
           = 3 model families
Modes:     DOM / SoM / Vision / Phantom-SoM / P-text = 5 modes
Cells:     6 sites × 3 models × 5 modes = ~90 cells (~125K episode total)
+ Router:  Tier 1+2 (oracle + first-step trigger), 实际 deploy on agent
+ Multi-metric: cost / P95 latency / carbon (B1 measured + B0 estimate)
```

### 顶刊概率（final scope + multi-metric/green AI 加成 后）

| 投稿目标 | 概率 | 投稿优先级 |
|---|---:|---|
| **NeurIPS / ICLR main** | 45-60% | Tier 1 stretch |
| **ICML** | 40-55% | Tier 1 stretch |
| **ACL / EMNLP main** | 50-65% | Tier 1 |
| **MLSys** | **75-85%** ⭐ | **Tier 1 safe** (drop-in framing 完美 fit) |
| WWW / WSDM | 75-85% | Tier 2 |
| NeurIPS D&B | 70-80% | Tier 2 |
| **TMLR (journal)** | **75-85%** | **保底** |

→ Final scope 完成后, paper 顶刊出版几乎 100% (cascade NeurIPS → ACL/EMNLP → MLSys → TMLR)

### Multi-metric + Green AI axis 加成的 paper-level 价值

1. **Differentiator**: 现 web-agent paper (VWA/WebArena/SeeAct/SoM/FocusAgent) 几乎全不报 carbon
2. **Multi-metric Pareto** 在 ML 顶会近年是 expected
3. **三向 drop-in** (cost+latency+carbon) narrative 立体
4. **Green AI** 是顶会新兴 axis (Strubell ACL 2019, Patterson 2021)

**Caveat**: green AI 是 second-order, 不能抢主线 "hidden routing arm + drop-in deployment"

---

## §6 Critical Risks + Mitigation (4 risks, 决定接收 vs reject)

### Risk 1: Execution quality（顶刊成败 #1 因素 ⚠️⚠️⚠️）

90 cells × ~1390 task = ~125K episode. 任何 cell 跑 sloppy (auth bug / cross-contam / 数据污染 / FP 没处理) 都被 reviewer 抓出.

**Mitigation**:
- 维持 paper-grade re-run 协议: reset between conditions, exclusive same-site B0 XOR B1, watchdog auto-rederive
- 每 cell 完成后立刻 `make analyze` + manual audit gallery
- **不在 execution quality 妥协**

**Status (04-28)**: ✅ B0 cls + red 5-mode 100% paper-grade clean (watchdog auto-clean verified, 0% wasted task)

### Risk 2: Story discipline ⚠️⚠️

6×3×5 cells 容易让 paper 变 "data dump". 顶会 reviewer 反感 "everything but the kitchen sink".

**Single narrative**: "Phantom-SoM is hidden routing arm + we explain why + we route on it + here's the cost saving".

其他 finding (capability shift / category profile) 都是 supporting evidence. Section 4-5 each ≤4 pages, supplementary 装其余.

### Risk 3: Router design ⚠️⚠️

Router 只比 best-single-mode 提升 1-2pp 被 reviewer 说 "不值". Oracle features (test-time leak) 直接 reject.

**Router design tiers**:
- **Tier 1 (must-have)**: Oracle router — task feature → best mode lookup, train/test split
- **Tier 2 (great-to-have)**: First-step-trigger router — 看 step 1 obs 决定 mode, no test leak
- **Tier 3 (stretch)**: Online learning router — mid-trajectory escalation

Tier 1 + Tier 2 就够顶会 contribution; Tier 3 stretch goal.

**Realistic timeline**: ~3-4 周 (vs 之前估 2-3 周)

**Minimum viable router** (start ~3 天 prototype):
```
Feature:  task instruction TF-IDF + binary {has_ref_image, has_finish_string_match}
Target:   max adjusted SR
Model:    Logistic regression (interpretable + small-data friendly)
Train:    cls + red 6 mode, 80/20 split
Baseline: random / best-single-mode / rule-based ("if has_ref_image → SoM else → Phantom-SoM")
```

### Risk 4: Negative results 必须诚实报告 ⚠️

某些 cell 可能反 trend (e.g. Claude shopping Phantom-SoM 不 work). **绝不 cherry-pick**, reviewer 看出直接 reject.

**Mitigation**: 诚实报告反而强化 mechanism claim ("effect 是 task-type/capability bound, 不是 universal").

---

## §7 Investment Cascade Plan

```
Round 1 (T+12 周, paper done):
  → MLSys 2027 (deadline 通常 9-10 月) 或 NeurIPS workshop (Maria 推荐)
  → 75-85% expected outcome  ⭐ first paper friendly venue

Round 2 (rejection 或 timing 错过):
  → ACL / EMNLP main (industry track 友好)
  → 50-65% expected outcome

Round 3 (still rejected):
  → NeurIPS / ICLR main (大幅修改 narrative)
  → 45-60% expected outcome

Round 4 (保底):
  → TMLR (journal rolling review)
  → 75-85% expected outcome
```

**Modified strategy** (per first paper considerations):
- 不把 NeurIPS/ICLR 作 round 1 (lottery + first-paper baggage)
- MLSys 是 strategic safer bet (drop-in framing 完美 fit)
- Maria's 推荐 channel 在 sustainability workshop / AI4SD venue 最有效

期望出版 venue 链 ~99% (5 站 5 model deployed-router scope 没法被全拒).

---

## §8 Router Design (Tier 1+2)

### 5 个关键设计决策点 (each requires ablation)

| 维度 | 选项 | 难点 |
|---|---|---|
| **Feature** | task NLP / browser state / step-1 trigger / capability / audit cat | audit cat 是 leak; small data overfit |
| **Target** | max SR / SR-per-cost / Pareto / budget-constrained | multi-obj weight 选 |
| **Granularity** | task-level / step-level / confidence-triggered | step-level 重跑 2x cost |
| **Cascade** | 单 router / B1→B0 escalation / rule+ML hybrid | escalation 实验代价大 |
| **Baseline** | random / best-single-mode / oracle / rule-based | best-single-mode 是 hardest baseline |

### Realistic timeline (paper 真正最值钱的工作量)

```
Tier 1 (task-level oracle): ~5-7 天
  ├─ Feature engineering (task NLP + browser meta): 2-3 天
  ├─ Train/eval split + baseline 对比:               1-2 天
  └─ Ablation (各 feature 组的 contribution):         1-2 天

Tier 2 (first-step trigger / cascade): ~7-10 天
  ├─ 重新跑 step-1 切换实验:                         3-4 天
  ├─ Confidence threshold tuning:                  1-2 天
  └─ Cascade ablation:                             2-3 天

Total: ~3-4 周
```

### Routing infra 现状 (paper 1 直接用)

- 4 baselines + 5 phantom × `confidence_summary.json` (`overall_usable=True`)
- Behavioral signals AUROC 0.682-0.748 (cls behavioral 主导, red verbalized 主导)
- Verbalized signals AUROC 0.701-0.793 (red P-text 0.793 是 5-mode 最高)
- Router scaffold: `p79/experiment/router.py::RuleBasedRouter`
- **Phantom modes 直接复用 baseline signal infra** (drop-in routing claim 第 4 fold)

---

## §9 Advisor Align Checklist

### Meeting #1 (~Week 3, cls+red+shopping done)

| 决策 | Options | 推荐 | 影响 |
|---|---|---|---|
| Router scope | (a) Tier 1 only / (b) Tier 1+2 / (c) Tier 1+2+3 | (b) Tier 1+2 | paper main contribution 强度 |
| Cross-model | (a) Skip / (b) Claude Opus 4.7 only / (c) + GPT-4o/Gemini | (b) Claude only | $70 budget vs scope |
| 单 paper vs 双 paper | (a) Integrated (Paper 1 含 router) / (b) Split (Paper 2 router) | **(a) Integrated** (毕设决策) | publication count vs paper depth |
| Authorship 预期 | TBD with advisor + Zekun | — | first paper credit |
| Investment timing | NeurIPS 2026 ~5 月 / MLSys 2027 ~9 月 / ICLR 2027 ~9 月 | MLSys safer | timeline 紧或松 |

### Meeting #2 (~Week 6-7, WA + Claude done)

| 决策 | Options | 推荐 |
|---|---|---|
| Paper venue (Round 1) | NeurIPS / ICLR / ACL / **MLSys** | **MLSys** (drop-in framing 完美 fit) |
| Section 6 Generalization 范围 | VWA + WA + Claude / + Mind2Web | VWA + WA + Claude 够 |
| 投稿 timing | ASAP vs polish 1-2 周 | polish 后 stable submit |

### 关键 strategic 问题 (advisor align 时主动问)

1. Maria 是否能 referee NeurIPS workshop / Climate Change AI workshop?
2. Holistic AI Zekun 推荐 industry track?
3. Paper review timing: 投稿前 advisor read pass 1 周, 改完 submit
4. 是否要做 Mind2Web pilot (advisor 偏好)
5. Claude Opus 预算: $70 上限 OK?

---

## §10 Visualization Plan (cascade router viz)

**单纯 2D cost-SR Pareto 不够 striking**. 推荐 4-figure stack:

| Figure | 作用 | 设计 |
|---|---|---|
| **Fig A: 3-panel multi-metric Pareto** | 主 figure, fig7 升级 | 3 panel: cost-SR + latency-SR + CO2-SR |
| **Fig B: Cumulative SR vs Budget curve** ⭐ | 最 striking, cost-aware 顶刊套路 | x=budget per task, y=cumulative SR; lines: random/best-single/rule/learned/oracle |
| **Fig C: Routing decision Sankey** | Section 6 解释 router 学到什么 | task category → routed mode → outcome |
| **Fig D: Per-task savings histogram** | Appendix supplementary | distribution: cost saved by routing per task |

**Fig B 详细设计** (参考 RouteLLM ICML 2024 / FocusAgent EMNLP 2025):

```
x: cumulative cost budget per task ($)
y: cumulative SR achievable
lines:
  --- random
  ··· best-single-mode (DOM/SoM/Phantom-SoM 各一条)
  --- rule-based router (handcrafted)
  ▬▬▬ learned ML router (ours) ⭐
  ─── oracle router (upper bound)
fill area: ours vs best-single-mode gap; ours vs oracle gap
```

直观论证: 在 $0.04 budget per task → 我们 router 25% SR vs best-single-mode 21%; oracle 边界 ~30%, learned router 缩小 60% gap.

**反对 3D Pareto**: rotate 才看清, paper 印刷不友好, reviewer 抗拒.

CO2 维度单独 fig E (regional sensitivity, 见 §11).

---

## §11 Cost / Latency / Carbon Multi-metric Plan

### 已有数据状况 (per `condition_summary_v2.json`)

| Backend | Cost | P95 Latency | Energy (kWh) | CO2e (kg) |
|---|---|---|---|---|
| B0 (proxy 235B API) | ✅ | ✅ | ❌ NaN (远端 GPU 不可测) | ❌ NaN (token-estimate-able) |
| B1 (local 4B GPU NVML) | ✅ | ✅ | ✅ | ✅ |

### Carbon tracker 现状 (`p79/experiment/energy_tracker.py`)

- ✅ NVML GPU measurement + 45 region intensity table
- ❌ 未 port: 220+ country DB (CodeCarbon), token-based proxy estimator, cloud provider data
- Default region: UK 220 g/kWh

### Tier 化 paper 利用

| Tier | Metric | 在 paper |
|---|---|---|
| **Tier 1 (主体)** | adjusted SR, drop-one oracle, cost/task | Section 1 hook + Section 4 main + fig7 |
| **Tier 2 (主体辅)** | P95 latency, CO2/task | Section 4 cost-aware table + Section 7 sustainability |
| **Tier 3 (附录)** | wasted cost, energy kWh, cost_efficiency_ratio | supplementary |

### Striking findings 已 measured (paper 直接 cite)

1. **B0 cls SoM P95 lat 74s ≈ 2× DOM 38s** (image inference 拖慢) ⭐
2. **B1 cls SoM energy 0.0020 kWh < DOM 0.0052** (step count 主导, counterintuitive)
3. **B1 reddit SoM > DOM energy** (site-dependent)
4. **Phantom-SoM cls cost ≈ DOM** + latency 4× 改进 = triple win

### Regional Carbon Sensitivity (fig9 already done, codex `d3dfc8f`)

`scripts/analysis/figures/fig3_regional_carbon.py`:
- 45 region × B1 3-mode × cls + red
- Norway 29 g/kWh → South Africa 928 g/kWh (32x range)
- Phantom-SoM advantage region-dependent (large for India/Poland, small for France/Norway)

---

## §12 References / Doc Map

### Paper drafts (final prose, `docs/analysis/paper_drafts/`)

| File | Status | Words |
|---|---|---:|
| `section1_intro.md` | ✅ done | 786 |
| `section2_background.md` (+ paper.bib 16 entries) | ✅ done | 1514 |
| `section3_definition.md` | ✅ done | 863 |
| `section4_empirical_findings.md` | 🟡 stale, codex #11 待 update | 1725 |
| `section5_mechanism.md` | ❌ 未写, codex #13 待 (~50K, 等 #10 lit) | — |
| `section6_generalization.md` | ❌ 未写 (待 WA + Claude data) | — |
| `section7_discussion.md` | ❌ 未写 (paper end-stage) | — |

### paper.bib

`docs/analysis/paper_drafts/paper.bib` (16 entries, 待 codex #10 expand to ~38)

### Codex analyses (`docs/analysis/phantom_paper/`)

| File | Words | Commit |
|---|---:|---|
| `disagreement_clusters.md` (B0+B1 9-cat) | — | `ded0ef6` `c4b52c3` |
| `cross_site_pattern_consolidation.md` (cls vs red shift +50/+33pp) | 1596 | `ab86019` |
| `phantom_dom_vs_som_diagnostic.md` (axis 2 prompt diag) | — | `5821387` |
| `som_vs_phantom_som_diagnostic.md` (axis 3 image 8-ch) | — | `7106d2e` |

### Other analyses

- `docs/analysis/B1_capability_profile.md` (B1 6 sections, 2245w, `03ffb2f`)
- `docs/literature/The Novelty and Efficacy of Set-of-Mark Text...md` (deep research)

### Figures (`results/phantom_paper/figures/`, all FRESH 04-28)

```
fig1 4-mode venn (2x2 B0+B1 cls+red)
fig2 drop-one oracle (2x2)
fig3 strategy gradient (2x4 reddit + cls)
fig4 two-knob diagram schematic
fig5 category × mode heatmap (B0 cls+red)
fig6 capability contrast B0-vs-B1 +43.7pp aggregate
fig7 cost-SR Pareto + deployment callouts
fig8 overlap-depth stacked bar (5-mode)
fig9 regional carbon sensitivity (B1 only)
```

### 实验笔记 § index (key findings)

- §95 adjusted_success canonical + Pareto
- §97 audit ~17500 LOC + 13 YAML
- §99 Magento auth bug + Knockout
- **§100 SoM 截图视觉 probe (B0/B1 OCR + attention)** ⭐ ground truth
- **§101.九 Lazy minimization hypothesis** ⭐
- §102 Phantom-SoM 工程实施
- **§103 Phantom-SoM 4-mode routing arm + paper narrative** ⭐
- §104+ Daily chronicle (Day 1-2 progress moved here from next_steps)

### Recent key commits (~04-27 / 04-28)

```
ae0f8e7  next_steps Day 2 update
8dde2cb  watchdog auto-clean paper-grade 100% pure verified
139afb0  router framing fix (paper 1 not paper 2)
9d7e99f  phantom routing signal AUROC ≥ baseline (4-fold drop-in)
8263d26  axis 2/3 literature deep research plan
b4bbe75  axis 3 image 8-channel framework
81613e0  axis 3 sub-mechanism refinement
00124e4  3-axis hierarchical theory framework
7106d2e  som_vs_phantom_som diag
5821387  phantom_dom_vs_som diag
ab86019  cross-site pattern consolidation
03ffb2f  B1 capability profile
ef29add  drop-in deployment punchline
48db047  Phantom-SoM cost ≈ DOM
93e413f  3-layer cost decomposition
4d63c9f  Section 3.2 token re-estimate (1064/733 measured)
```

详 `git log --oneline --since="2026-04-27"` 看完整历史.

---

## §13 Pending TODO (paper-strategic, not action ledger)

### A. Codex prose tasks (跟踪 in next_steps §4 codex queue)

- [ ] codex #10 axis 2/3 literature deep research → expand paper.bib 16→~38 (~Wed)
- [ ] codex #11 Section 4 fresh-data prose update (~Wed)
- [ ] codex #13 Section 5 prose 写 (3-axis hierarchical + lit cite, ~Thu)
- [ ] codex #16 Section 6 Routing prose (Week 5-6, after Tier 1+2 prototype)
- [ ] codex #17 Section 7 Generalization 草稿 (~Week 6-7, after WA + Claude done)
- [ ] codex #18 Section 8 Discussion 草稿 (paper end-stage, 含 sustainability + lat 4× finding)
- [ ] codex #19 二次 deep research (Section 6/7/8 + 全 paper revisit, paper 终稿前 Week 8+)

### B. Data analysis pipeline (Python scripts, not codex tokens)

- [x] **统计显著性测试** ✅ done 04-28 — `fig0c_drop_one_oracle.py` 加 `bootstrap_drop_one_ci()` (1000 resample × 4 panel)，error bars + `fig0c_drop_one_bootstrap_ci.csv` 12 rows
  - Section 4 reviewer-grade rigor; codex #11 prose 可直接引用 95% CI
  - Pending: paired permutation test for cross-mode SR delta (lower priority)
- [x] **AUROC aggregation table** ✅ done 04-28 — `scripts/analysis/aggregate_routing_auroc.py` (~110 行)
  - Outputs: `results/phantom_paper/auroc_cross_condition.csv` (188 rows × 5 modes × 4 cells) + `_summary.md` (top-1 per cell, Section 6 claim 证据)
  - Section 6 "AUROC ≥ baseline" claim 部分支持: B0 red P-text 0.793 highest; B0 cls P-text 0.737 ≥ SoM 0.709 baseline; B1 cells 待 chain done
- [x] **Phantom routing lift** ✅ done 04-29 — `scripts/analysis/aggregate_phantom_lift.py` (~180 行)
  - Outputs: `results/phantom_paper/phantom_lift.{csv,md}` — 3-mode → 5-mode oracle ceiling lift + bootstrap CI + per-phantom decomposition
  - **Paper Section 1/4 hook 主 evidence**: B0 cls **+4.70pp [2.14, 7.69]** ✅, B0 red **+5.24pp [2.38, 8.11]** ✅ (CI 排除 0)
  - Decomposition: P-text adds 8 tasks / P-SoM adds 6-7; each phantom 有独家 + overlap 部分 → keep both phantoms in paper
  - B1 cells 待 chain done 自动 cover (script 检测 ep count, ≥50 ep 触发)
- [ ] **Multi-metric Pareto pipeline** (cost + latency + carbon)
  - Section 8 sustainability prose 前置；fig9 已有 carbon B1 only, 需 cost/latency 三向 join
  - Output: 3-panel Pareto figure + per-condition multi-metric table
  - Implementation: extend `scripts/analysis/figures/fig3d_cost_sr_frontier.py`
- [ ] **每 task 特征提取** (Section 6 Tier 1 oracle router 前置)
  - Features: TF-IDF (task instruction) + has_ref_image binary + has_finish_string_match binary + site / category metadata
  - Output: `task_features.parquet` per benchmark
  - Implementation: `scripts/analysis/extract_task_features.py`
- [ ] **B0 token-based carbon estimator** (Section 8 Tier 3 sustainability)
  - 当前 §3.6 marked "optional"; 需 minimum implementation (eu-west-2 default region, token × carbon factor)
  - Source: `condition_summary_v2.json` 含 token counts per condition
  - Implementation: ~20 行 helper in `p79/experiment/metrics.py`，paper Section 8 引用

### C. Paper end-stage tasks (Week 8+)

- [ ] Pre-submission checklist (paper_planning §17) execute
- [ ] LaTeX 转换（当前 markdown drafts → LaTeX template per venue）
- [ ] Bib 完整性 check (citations present, format correct)
- [ ] Reproducibility appendix（commit summaries + onboarding instructions）
- [ ] Router Tier 1 prototype (~3 天, baseline + phantom 全 done 后)
- [ ] Router Tier 2 first-step trigger (~7-10 天)
- [ ] Advisor align meeting #1 prep (~Week 3)
- [ ] Advisor align meeting #2 prep (~Week 6-7)
- [ ] paper writing + revisions (~Week 8-12)

---

## §14 Reviewer Attack Anticipation + Pre-Rebuttal

顶刊投稿 reviewer 常见攻击 + 我们的 response (paper integrity hardening):

| Attack | Likely Reviewer Concern | Our Response | Evidence |
|---|---|---|---|
| **Sample size too small** | "VWA cls 234 + red 210 = 444 task, single benchmark" | Final scope = 6 sites × 3 models × 5 modes ≈ 1390 task per condition. Cross-site (cls + red + shopping × VWA + WA), cross-model (Qwen 235B + 4B + Claude Opus 4.7) | §5 Final scope; §3.1 B0 5-mode SR table |
| **Single benchmark family** | "VWA only, no Mind2Web/WebVoyager validation" | + WA (Postmill / Magento / shopping_admin) cross-stack validation. Mind2Web out of scope per advisor align (Plan B) | §7 generalization + paper §6 |
| **Single model family (Qwen)** | "Effect Qwen-specific?" | + Claude Opus 4.7 cross-model after advisor align (~$70). B0 (235B) + B1 (4B) shows capability-dependent shift (+50/+33pp cross-site, §101.九 lazy minimization) | §2 capability layer + cross_site_pattern_consolidation.md |
| **Phantom is just a degraded SoM** | "Why not collapse to DOM if no image?" | Theory C (codex 5821387) verifies prompt knob: cls P-text = Phantom-SoM SR 14.53% but Jaccard 0.447 (task pool 显著 disjoint). Same SR ≠ same routing pool | paper §5; codex `5821387` |
| **Effect size small (drop-one 1.7-3.3pp)** | "Statistically marginal" | site-modulated effect, conservative framing (within 2σ noise floor). Paper claim 不是 "Phantom #1 routing arm" 而是 "site-modulated representation effect with 4-fold drop-in property" | §1 Paper hook conservative framing |
| **Latency claim cherry-picked** | "Just one P95 measurement" | §100 SoM probe ground truth (5 imgs × 3 mode × 2 model = 30 cells measured). cls SoM 74s vs Phantom 18s p95 = 4× slower. Across all conditions consistent | §11 + 实验笔记 §100 |
| **Carbon estimation rough** | "B0 carbon NaN, only B1 measured" | Transparent disclose: B1 NVML measured directly, B0 (proxy API) 远端 GPU 不可测 (per Strubell 2019 / Patterson 2021 estimation acknowledged). fig9 regional sensitivity 用 B1 measured + 45 region intensity table | §11 + fig9 footnote |
| **Router contribution toy** | "Tier 1 oracle is overfit" | Tier 1 train/test split, baseline 对比 (random, best-single-mode, rule-based, oracle, learned). Tier 2 first-step trigger no test leakage | §8 + Section 6 outline §4.6 |
| **No production deployment** | "Drop-in claim hypothetical" | 4-fold drop-in property: code-level verified (`som.py::_extract_text_marks` line 24 regex); routing signal AUROC ≥ baseline (5/5 `overall_usable=True`); 实证 cost+latency+CO2 measured | §1 + §3 finding #5 #9 |
| **Watchdog detection unreliable** | "FPC false alarm undermines paper-grade" | Site-specific audit: cls (real auth issue + auto-clean + 重跑 done), red (0 events), shopping (FPC false alarm fixed). Watchdog auto-clean protocol delete contaminated + runner resume → 0% wasted task. paper-grade 100% pure verified | §18 + 实验笔记 §104 |
| **Mechanism not novel** | "Each axis has prior literature" | Contribution = systematic decomposition + web-agent multi-step setting + drop-in deployment claim. NOT new LLM mechanism. Paper §5 framing 已 acknowledge | §2 paper contribution position |
| **Overfit to VWA visual specifics** | "Effect won't generalize to WA" | §103 falsifiable prediction: WA Phantom-SoM 5-mode oracle gain. WA pilot ≤50 task verify Jaccard ≤0.5 universal vs >0.7 VWA-specific | §103 generalization prediction; pending data |

**Pre-rebuttal strategy**:
- Section 4-5 prose 写时 inline cite this table (proactive defense)
- Section 7 Generalization 必须 explicit address WA + Claude (跨 stack + 跨 model)
- Section 8 Discussion 4.4 limitations 提前 acknowledge known weaknesses

---

## §15 Prior Work Comparison Table

paper Section 2 必备 explicit table (review 加分项):

| Aspect | Yang 2023 SoM (NeurIPS) | VWA Koh 2024 (ICLR) | SeeAct Zheng 2024 (ICML) | FocusAgent Kerboua 2025 (EMNLP) | RouteLLM Ong 2024 (ICML) | **Ours (Phantom-SoM)** |
|---|---|---|---|---|---|---|
| **Marks-text isolation** | ❌ bundled with image | ❌ bundled | ❌ bundled | n/a | n/a | ✅ Phantom-SoM ⭐ |
| **Routing arms** | 1 (single SoM) | 1 (per mode) | 1 (single SoM) | 1 (text prune) | model-level routing | **5-mode** (DOM/SoM/Vision/Phantom-SoM/P-text) ⭐ |
| **Cost-aware Pareto** | ❌ | ❌ | ❌ | ✅ token cost | ✅ model cost | ✅ **multi-metric** (cost+latency+carbon) ⭐ |
| **Cross-site validation** | 4 task domains | 3 sites (cls+red+shop) | 1 site | 2 sites | n/a | **6 sites** (VWA+WA) ⭐ |
| **Cross-model** | 4 models (multimodal) | 6 models (api+local) | 4 models | 1-2 | many (text-only LLM) | 3 models (Qwen 235B+4B + Claude Opus) |
| **Mechanism analysis** | ❌ effect-only | ❌ partial | ❌ baseline | partial (text size effect) | ❌ effect-only | ✅ **3-axis × 8-channel × bidirectional** ⭐ |
| **Drop-in deployment** | ❌ | ❌ | ❌ | partial | partial | ✅ **4-fold property** (cost/latency/signal/oracle) ⭐ |
| **Carbon report** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ **Differentiator** ⭐ |
| **Failure mode taxonomy** | none | 3 categories | none | none | none | **9 categories** + 8-channel image (codex diags) |
| **Sample size** | varies | 910 task total | 50 task subset | 3 sites partial | many | **1390 task** per condition (final scope) |

**Closest prior pairing**: FocusAgent (text 压缩, hierarchy 保持) + Yang 2023 SoM (visual marks). 本工作 = unprecedented synthesis + drop-in deployment claim + multi-metric Pareto + green AI differentiator.

详 deep research: `docs/literature/The Novelty and Efficacy of Set-of-Mark Text as an Independent Representation Routing Arm in Web Agents.md` (5-dimension gap confirmation, §103).

---

## §16 Authorship + Advisor Roles + First-Paper Strategy

### 毕设 paper authorship plan (TBD with advisor align meeting #1)

```
First author: jiaming (毕设 student, primary work)
Co-supervisor: Zekun (Holistic AI, industry collaboration)
Advisor: Maria Perez Ortiz (UCL, AI4SD program director)

Tentative authorship order: jiaming, Zekun, Maria
(final order pending advisor align meeting #1)
```

### Advisor / collaborator roles

| Person | Role | Paper contribution |
|---|---|---|
| **jiaming** | Implementation + experiments + writing + first-paper learning | Main author, all sections, codex orchestration, paper-grade execution |
| **Zekun** | Industry collab + drop-in deployment 视角 + MLSys positioning | Section 8 Discussion deployment angle + venue strategy + reference review |
| **Maria** | Theoretical guidance + AI4SD framing + sustainability + conference network | Section 1/2 background + Section 8 sustainability + referrer pipeline (NeurIPS/Climate Change AI workshop) |

### Personal context (毕设 backdrop, 本 paper 是 first paper)

- 西安交大 undergrad → UCL AI4SD master/PhD transition
- First paper, 经历: paper trajectory 从 "magical noise 怀疑" 到 "4-fold drop-in deployment claim"
- 多次 critique-driven theory refinement (4 rounds: prompt-only / visual-hijack-only / image-over-text / SoM density) — paper integrity discipline 体现
- Holistic AI industry collab 是 publication track signal (industry endorsement)

### First-paper psychology + strategic advice

```
Realistic outcome distribution (per §5 顶刊概率):
  Round 1 (MLSys 2027 ~9月): 75-85% accept (推荐 first paper friendly)
  Round 2 (ACL/EMNLP if rejected): 50-65%
  Round 3 (NeurIPS/ICLR main if still rejected): 45-60%
  Round 4 保底 (TMLR rolling): 75-85%
  
出版概率 cascade 累积: ~99% (基本 lock paper 出版)

Key first-paper considerations:
- Don't put NeurIPS/ICLR as round 1 (lottery + first-paper baggage if rejected)
- MLSys 推荐: drop-in framing 完美 fit + first-paper friendly review
- Maria's referrer pipeline (NeurIPS workshop / Climate Change AI / AI4SD venue) 是 strategic leverage
- Holistic AI industry endorsement → industry track 友好
- Rejection 是 norm, 不要把 rejection 等同于 paper 不行
```

### Acknowledgments draft (paper end-stage)

```
预 draft (Section 8 Acknowledgments):
- Compute: DGX Spark (UCL AI4SD) + remote VWA Docker (Tailscale) + Myriad GPU pending
- Data: VWA + WA benchmarks (open source, properly cited)
- API: Qwen3-VL-235B-A22B (proxy via internal infra), Claude Opus 4.7 (advisor budget)
- Discussions: advisor + co-supervisor + UCL AI4SD group
- COI: Holistic AI industry collab acknowledged
```

---

## §17 Pre-Submission Checklist (~Week 10-12 paper 终稿前)

### Content completeness

- [ ] All 8 sections prose done (Section 1-3 ✅ done, Section 4-8 待 codex)
- [ ] paper.bib expanded to ~38 entries (待 codex #10 deep research)
- [ ] All figures FRESH with latest data + paper-grade captions
- [ ] Section 2 含 prior work comparison table (paper_planning §15)
- [ ] Section 4-5 prose 含 reviewer attack pre-rebuttal (paper_planning §14)
- [ ] Section 5 mechanism 含 §100 SoM probe ground truth + 14 case studies + Tong 2024 cite
- [ ] Section 6 Routing 含 Tier 1+2 implementation + 4-fig stack
- [ ] Section 7 Generalization 含 cross-site (shopping + WA) + cross-model (Claude) data
- [ ] Section 8 Discussion 4 sub-sections (drop-in summary / mechanism / sustainability / limitations)
- [ ] Negative results explicit listed (paper integrity)
- [ ] Limitations section honest (no over-claim)

### Format / Style

- [ ] Page count check (NeurIPS/ICLR 9 page main + unlimited supp; MLSys 12-15 page)
- [ ] Reference style (BibTeX validation, all 38+ entries cite-resolved)
- [ ] Figure resolution 300 DPI (paper print)
- [ ] Code anonymized for review (if double-blind)
- [ ] Supplementary materials packed (data CSVs, configs, analysis dirs)
- [ ] Captions self-contained (figure 不依赖 text)

### Reproducibility

- [ ] Code release path: github / zenodo decided
- [ ] Data: VWA tasks + run results + figures input data (per condition_summary_v2.json)
- [ ] Configs: configs/exp_v2_*.yaml all referenced
- [ ] Reproducibility statement: `make figures` / `make analyze` workflow documented
- [ ] Replication recipe in supplementary

### Authorship + Submission

- [ ] Author order finalized (advisor align #1 + #2)
- [ ] Acknowledgments (compute resources + advisor + collaborators)
- [ ] Conflict of interest declaration (Holistic AI industry collab)
- [ ] Venue-specific format (MLSys vs NeurIPS template chosen)
- [ ] cover letter / abstract polished

### Pre-rebuttal preparedness

- [ ] §14 reviewer attack table integrated to prose (proactive)
- [ ] §15 prior work table integrated to Section 2
- [ ] Limitations explicit (Section 8.4)
- [ ] Cost/budget transparency (acknowledgments)

---

## §18 Watchdog Protocol + Paper-Grade Execution Discipline

> 这部分内容 paper 写时可作为 supplementary "paper-grade execution discipline"
> 引用; 也是 reviewer 信任度的 evidence。

### 6-layer Defense in Depth (per `experiment_watchdog.py`)

```
1. Detection: per-task DOM session check (5000 char window, _check_session_health)
   - Site-specific tab guard (cross-site task skip)
   - Logout / Sign In link regex
2. Alert: streak ≥3 → ntfy notification + ALERT log
3. Refresh: real Playwright sign-in subprocess (auth_refresh.py)
   - Per-site account credentials
   - host-resolver-rules MAP metis → IP (legacy)
   - Verify post-login URL ≠ login_path before storage_state write
4. Cleanup: delete contaminated episodes (auto-clean on login restored)
   - Delete summary_v2.json + steps_v2.jsonl
   - rmtree artifacts/{site}_task_{tid}/
   - Purge digest records
   - Remove from seen_keys
5. Resume: runner re-run with fresh logged-in storage_state
6. Verify: post-cleanup mtime + DOM check (paper-grade integrity audit)
   - State file persists across watchdog restart
```

### Magento history (3 复发 + final fix)

```
2026-04-X (initial): cookie domain split (PHPSESSID under IP, form_key under metis)
                     → fix `7150db8` (quark side base_url 改 IP)
2026-04-27: docker reset 后 base_url 退回 metis
                     → fix `f9cbebf` (DGX defensive curl + quark side scripts)
                     → 3-layer 持久化 (magento_baseurl_fix.sh + start_vwa_docker.sh
                        hook + reset_shopping.sh remove hardcode localhost)
2026-04-28: PowerShell reset chain 没集成 base_url fix → docker reset 仍 invalidate
                     → fix: PowerShell `C:\vwa\reset_vwa.ps1` 加 Configure-MagentoBaseUrl
                       (docker exec config:set + cache:flush, shopping 7770 +
                        shopping_admin 7780 都覆盖)
2026-04-28: Magento Full Page Cache (FPC) homepage cache guest page → false alarm
                     → fix: quark side `bin/magento cache:disable full_page` +
                       PowerShell hook 持久化 (reset 后 auto-disable FPC)
```

### Paper-grade clean re-run protocol

```
Before each new condition:
  1. reset_vwa_sites.sh → DGX SSH quark PowerShell
     PowerShell: docker stop + start vwa-{site} container
     PowerShell: Configure-MagentoBaseUrl (config:set + cache:flush + cache:disable full_page)
     PowerShell: site-specific health check (HTTP 200)
  2. DGX defensive curl: verify redirect ≠ metis (commit f9cbebf)
  3. Refresh storage_state (auth_refresh.py if streak ≥3)
  4. Launch runner with --resume flag
  
During run:
  - Watchdog poll 30s, NOT LOGGED IN streak detection
  - Auto-clean on streak ≥3 + login restored
  - Runner resume picks up missing tasks → fresh re-run
  
Post run:
  - rederive episode summaries (re-compute adjusted_success per FP rules)
  - Auto figures regen (`make figures`)
  - Cross-rep / reason_diag / cross_run analysis (per analyze pipeline)
```

### Paper integrity 论证 (Section 4 / supplementary)

- **0% wasted task data** (Day 2 audit verified): all NOT LOGGED IN events auto-cleaned + 重跑 done. Final episode summaries 全 fresh logged-in.
- **Site-specific noise sources**:
  - cls (OSClass): real auth issue, watchdog auto-clean + 重跑 (~2% early tasks affected per condition)
  - red (Postmill): 0 NOT LOGGED IN events
  - shopping (Magento): FPC false alarm fixed, B0 NEW launch with FPC disabled
- **Cross-mode comparison preserved**: 5 modes 受同一 protocol, drop-one oracle / Jaccard / cost-SR Pareto 都不被 ~2% noise bias
- **Paper-grade discipline**: self-healing data pipeline, 6-layer defense in depth → reviewer 信任 paper data integrity

---

## §19 Decision Log (paper-strategic decisions audit trail)

| Date | Decision | Rationale | Status |
|---|---|---|---|
| 2026-04-27 | Final scope: 6 sites × 3 models × 5 modes + deployed router + multi-metric + green AI | NeurIPS/顶刊 viable scope (paper_planning §5) | ✅ in plan |
| 2026-04-27 | P-text scope 缩减 18→5 cells (mechanism only) | P-text 是 ablation 不是 routing arm 候选 | ✅ in plan |
| 2026-04-27 | Future paper 2 转向 Phase 3 modules (router 整合 paper 1) | 毕设决策, paper 1 含完整 contribution | ✅ in plan |
| 2026-04-27 | First paper 投稿 cascade: round 1 → MLSys (不 NeurIPS) | first-paper friendly + drop-in framing 完美 fit | ✅ in plan |
| 2026-04-27 | Paper hook 升级到 "drop-in deployment intervention" | Phantom-SoM cost ≈ DOM (regex filter), 4-fold property | ✅ commits 48db047 + ef29add |
| 2026-04-28 | B1 shopping DOM 466 ep clear+rerun (paper-grade 协议一致性) | pre-Magento-bug 跑期间, cookie domain split risk | ⏳ 等 Myriad GPU |
| 2026-04-28 | Magento FPC disabled (server-wide) | FPC homepage cache guest false alarm + persistent fix | ✅ done |
| 2026-04-28 | Theory C: prompt as task-conditional decision prior (NOT commit-only) | codex `5821387` Jaccard 0.45-0.54 disjoint task pool | ✅ paper_planning §2 |
| 2026-04-28 | Image axis 8-channel taxonomy (NOT visual-hijack only) | codex `7106d2e` 4 helping + 4 harming, false visual confidence MAIN red 60% | ✅ paper_planning §2 |
| 2026-04-28 | Bidirectional modality framing (image-over-text vs text-over-vision) | user Q3 critique + Tong 2024 "Eyes wide shut" anchor | ✅ paper_planning §2 |
| 2026-04-28 | 4-doc structure (next_steps + paper_planning + drafts + 笔记) | original 1102-line next_steps too dense, separation of concerns | ✅ commit 97cc4ac |
| 2026-04-28 | 8 sections paper structure (含 Section 6 Routing 独立) | router 是 paper independent contribution, not Section 7 sub | ✅ commit 4ca9f66 |
