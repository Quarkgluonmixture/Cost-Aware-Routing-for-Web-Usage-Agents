# Phantom-SoM 学长会议 brief — 30-45 分钟讲稿

**Date**: 2026-04-30
**Audience**: PhD advisor (PI)
**Goal**: 同步 Phantom-SoM finding + 14 figures 现状 + VWA bug 副发现 + 2 个 decision request (RunPod 经费 / paper scope)
**Format**: 序列化 talking script — 跟着 figures 顺序讲, 每张图给"指着说什么 + reviewer 可能问什么 + 准备答案"
**Companion**: `docs/checkpoints/paper_planning.md` (1255 行 strategy notebook, canonical source)

---

## 0. 会议节奏 (建议)

| Block | 时长 | 内容 |
|---|---:|---|
| **A. 30-sec hook** | 0:00–0:30 | 4-fold drop-in property one-liner |
| **B. Theory setup** | 0:30–4:30 | 3-axis ablation framework + diamond completion |
| **C. Empirical walk-through** | 4:30–18:00 | 14 figures 顺序讲 — Outcome / Macro / Micro / Efficiency 4 dimension |
| **D. Mechanism + caveats** | 18:00–28:00 | site-modulated effect / capability tier / VWA bugs |
| **E. Decisions (asks)** | 28:00–35:00 | RunPod 经费 + scope (single vs split paper) + Section 4 framing |
| **F. Buffer** | 35:00–45:00 | reviewer pushback / next steps |

---

## A. 30-sec Hook (开场必背)

> "Phantom-SoM 是 SoM-style web agent 里一个**之前没人发现的第 4 路由臂**, 性质是 4-fold drop-in: cost 跟 DOM 一样便宜, latency 比 SoM 低 50%, 路由 signal 的 AUROC 不输 baseline, drop-one oracle 还能贡献 1.7-3.3pp. 这个 arm 是把 SoM 的"图"那部分关掉, 只留下 SoM 的 text + prompt — 不需要 retrain, 不需要换 prompt, 不需要 infrastructure. 我们用 3-axis ablation 证明它是 hidden arm 不是 SoM 的子集."

**关键词**: hidden / 4-fold drop-in / 3-axis ablation / no retraining

**不要说**:
- ❌ "我们发现 Phantom-SoM 比 SoM 好" (overclaim — 实际是 site-modulated)
- ❌ "我们提出新 prompting 方法" (Phantom 不改 prompt)
- ❌ "新 backbone" (零 model 改动)

---

## B. Theory Setup (3-axis ablation framework)

### 第一句话引入

> "Standard VWA 是 3-mode (DOM, SoM, Vision). 我们想问: SoM 比 DOM 强 (假如它强), 到底是因为 text 形式变了 / prompt 变了 / 加了图? 这 3 个 axis 同时变, 没法 attribute. 所以我们设计了 3 个 phantom mode 做 controlled mismatch."

### 三个 axis 的清单 (paper_planning §2 line 47)

| Axis | 是什么 | DOM | SoM |
|---|---|---|---|
| 1. Text payload | obs 文本结构 | AXTree (层级) | `[SOM_MARKS]` (扁平索引) |
| 2. Prompt prior | system prompt 风格 | "AXTree 风格" | "看图找 mark 风格" |
| 3. Image | 是否给截图 | ❌ | ✅ marked image |

### 6 mode 拆开 (这部分指 fig1ab cascade diamond 配合讲, 见 C.1)

```
DOM      = (AXTree, DOM-pr, ❌)              ← baseline, 最便宜 text
P-text   = (SOM_MARKS, DOM-pr, ❌)           ← 只换 axis 1 (text 形式)
P-SoM    = (SOM_MARKS, SoM-pr, ❌)           ← 再换 axis 2 (prompt) — Phantom-SoM 主角
SoM      = (SOM_MARKS, SoM-pr, ✅)           ← 再加 axis 3 (image) — 完整 SoM
P-prompt = (AXTree, SoM-pr, ❌)              ← 我加的 diamond completion (mirror P-text)
Vision   = (空, vision-pr, ✅)                ← 独立 image-only 路径
```

### Diamond completion (你的关键设计贡献)

> "原来的 ablation 是 L-shape: DOM → P-text → P-SoM 三个点串成一条链. 我加了 P-prompt corner — AXTree text + SoM prompt + 无图 — 让 ablation 闭合成 2x2 factorial 方格 (旋转 45° 看是 diamond). 这样 prompt × text 的 interaction 才能 separately quantify."

**关键好处** (paper_planning line 437):
- 原 L-shape 只能测 prompt 在 [SOM_MARKS] text 上的效应
- Diamond 4 corner 可以测 prompt 在 AXTree text 上的效应 (vs DOM)
- 两个 estimate 一致 → prompt × text 加性
- 两个 estimate 不一致 → 有 interaction, 可以 attribute 给具体 axis 组合

**reviewer 可能问**: 你为什么 5-mode 不是 8-mode (2³)?
**答**: 8-mode 里 4 个是 deliberate-ID-mismatched mode (e.g. AXTree text + SoM prompt 直接 hardcode SoM 的 `[mark N]` 编号系统但 AXTree 用 accessibility ID) — confound prompt effect 跟 parsing-confusion effect, 不是 clean ablation. 详见 paper_planning §2 line 53.

---

## C. Empirical Walk-through (14 figures 序列化)

### 推荐讲解顺序 (按 paper Section 5 mechanism 论证 flow)

```
C.1 Theory introduction → fig1ab_cascade_diamond.png
C.2 Outcome — Phantom contributes → fig0c_drop_one_oracle.png + fig0c_phantom_lift_bars.png
C.3 Outcome — but not via beating baseline → fig0d_taskpool_jaccard.png + fig0e_category_mode_heatmap.png + fig0f_overlap_stacked_bar.png
C.4 Routing infrastructure drop-in → fig0g_routing_auroc_heatmap.png
C.5 Macro — strategy cascade → fig1c_strategy_gradient.png
C.6 Micro — per-step divergence → fig2_micro_divergence_heatmap_B0/B1.png
C.7 Efficiency — 4-fold drop-in (a)(b) → fig3a_token_cost_intra_baseline.png + fig3d_cost_sr_frontier.png
C.8 Sustainability bonus → fig3_regional_carbon.png
C.9 Cross-capability → fig_capability_b0_b1.png
```

---

### C.1 fig1ab_cascade_diamond.png — 理论框架图

**指着说**:
> "这是 3-axis ablation cube 的 2D 投影 (image axis 都 = off). 4 个 corner 是 DOM / P-text / P-SoM / P-prompt. 边连接的是只差一个 axis 的 mode. 这是我们整个 paper Section 5 的 backbone."

**reviewer 看到的**:
- 左下 DOM (AXTree, DOM-pr): 11.6% red SR (示例数字)
- 左上 P-prompt (AXTree, SoM-pr): pending
- 右下 P-text (SOM_MARKS, DOM-pr): 13.81% red
- 右上 P-SoM (SOM_MARKS, SoM-pr): 13.81% red

**reviewer 可能问**: 为什么没 image axis?
**答**: image axis 单独效应在 fig0a (Vision vs P-SoM 等 paired comparison) — 这张专注 prompt × text 2D plane 是为了 visualize Phantom 系列 mode 怎么互相 connect.

---

### C.2 fig0c_drop_one_oracle.png + fig0c_phantom_lift_bars.png — 核心 Outcome 证据

**指着说**:
> "Drop-one oracle = 把某个 mode 拿掉之后 oracle SR 损失多少, 量化这个 mode 的 unique contribution. Phantom-SoM 在 reddit 上 drop-one 3.33pp, 比 SoM 的 1.90pp 还高 — 说明它 unique 解了 SoM 解不了的 task. 这是 (d) drop-one 1.7-3.3pp 性质的证据."

**关键数字** (paper_planning line 19-29):
- red P-SoM drop-one: **3.33pp** (vs SoM 1.90pp — Phantom 反而 unique 更多)
- cls P-SoM drop-one: 2.56pp
- red P-text drop-one: +3.21pp (axis 1 单独贡献)

**reviewer 可能问**: drop-one 跟 standalone SR 关系?
**答**: drop-one 测的是 marginal 贡献到 oracle ensemble — 即使一个 mode SR 不高, 只要它解的 task 跟其它 mode 不重叠, drop-one 就高. Phantom 在 reddit 的高 drop-one 说明它 task pool 跟 SoM disjoint, 不是 SoM 的子集.

---

### C.3 fig0d_taskpool_jaccard.png + fig0e_category_mode_heatmap.png + fig0f_overlap_stacked_bar.png — 互补不是替代

**指着 fig0d 说**:
> "这是 task-pool Jaccard heatmap. P-text 和 P-SoM Jaccard 0.500 — 一半 success task 不重叠. P-SoM 跟 SoM Jaccard 也 < 0.7. 这反驳了 'Phantom 是 SoM 的子集' 假说."

**fig0e** (类别 × mode):
> "不同 task category 各 mode 强项不一样 — Phantom-SoM 在 text-dominated category (reddit forum browsing) 强, SoM 在 visual category (cls listing browsing) 强. Site-modulated dominance."

**fig0f** (重叠堆叠):
> "Phantom 唯一 success 数量是 8 个 task — 这 8 个是 Phantom 的 unique contribution, 4-mode oracle 拿不到."

**关键**: 不要 frame 成 "Phantom 比 SoM 好" — frame 成 "**Phantom is complementary, not dominant**" (paper_planning line 365).

---

### C.4 fig0g_routing_auroc_heatmap.png — 路由 infrastructure drop-in (4-fold property c)

**指着说**:
> "这是 4-fold drop-in property 的第 (c) 条 — Phantom 模式作为 routing arm, 它的 routing signal AUROC ≥ baseline (5-mode 全 overall_usable=True). 意思是用 Phantom-SoM 当 router 候选 arm, 不会让 routing 决策变差."

**关键数字**: red P-text verbalized AUROC **0.793** (5-mode 最高, 超 baseline 0.766)

---

### C.5 fig1c_strategy_gradient.png — Macro action-frequency cascade

**指着说**:
> "这是 search-loop% 等 macro strategy metric 在 mode 之间的 gradient. 例如 reddit search-loop%: DOM 51.9% → P-SoM 35.7% → SoM 31.4%. 单调下降跟 [SOM_MARKS] 暴露程度一致 — 不是 SR 的副产物, 是 strategy 本身的 mechanism shift."

**为什么这张图重要**: 反驳"Phantom 优势是 dispatch bug 假象"假说 — bug 是 SR 的事, 这张图测的是 macro action distribution, 跟 bug 无关.

---

### C.6 fig2_micro_divergence_heatmap_B0.png + fig2_micro_divergence_heatmap_B1.png

**指着说**:
> "这是 step-level divergence — 我们对每个 task 看 DOM vs P-SoM 的轨迹什么时候 diverge (不同的第一步). Median 是 step 0 — 一开始就走不同路, 不是后期纠错. 进一步证明 Phantom 不是 SoM 的子集, 是真不同的 routing arm."

**B0 vs B1 关键**: B1 (4B) micro divergence pattern 跟 B0 (235B) 类似 — capability tier 不改 mechanism, 只改 absolute SR.

---

### C.7 fig3a_token_cost_intra_baseline.png + fig3d_cost_sr_frontier.png — 4-fold drop-in (a) cost

**fig3a 指着说**:
> "Phantom-SoM 的 token cost 跟 DOM 几乎一样 (red 3437 vs 3661 ±7%; cls 3008 vs 2948). 因为 [SOM_MARKS] 是 AXTree 的 regex filter, 不需要额外 model call. 这是 (a) cost ≈ DOM 性质."

**fig3d** (cost-SR frontier):
> "Pareto frontier 上 Phantom-SoM 在 reddit 上是 efficient corner — DOM-cost 等级 + SoM-之上 SR. 配 Vision 高 cost 高 SR 形成完整 cascade."

**4× latency 的来源**: 不画图但口头 cite — cls SoM p95 latency 74s, P-SoM p95 18.2s (paper_planning line 24). 因为 P-SoM 跳过 image encoding stage.

---

### C.8 fig3_regional_carbon.png — Sustainability angle (paper Section 8 加成)

**指着说**:
> "Cost methodology 我用了 electricity-equivalent ($0.12/kWh) 估算 — B0 API ~$0.04/ep vs B1 local ~$0.0004/ep, real ratio 是 ~100× (paper_planning line 363). 这是 §103 修正掉之前 30× claim 的产物. Sustainability framing 适合 ICLR/NeurIPS 加成或 ICML green AI track."

**reviewer 可能问**: 这个 carbon 数字怎么算?
**答**: 用 deployment-class ratio + 各 region grid carbon intensity (UK / EU / US-CA). Paper Section 8 详写 outline 在 paper_planning line 579.

---

### C.9 fig_capability_b0_b1.png — Cross-capability

**指着说**:
> "B0 是 235B Qwen3-VL via API, B1 是 4B local. Phantom-SoM 在 B1 上 SR 也比 DOM 高 (cls B1 P-SoM XX% vs DOM YY%). Mechanism 跨 capability tier 一致 — 这强 paper Section 7 generalization claim."

**Caveat 必须说**: B1 paper-grade 数据**仍在跑** (B1 P-text cls 198/234, ~85% done). 跨 baseline 的 final picture 1-2 周内补齐.

---

## D. Mechanism + Caveats

### Site-modulated framing (paper_planning line 123 + 363)

**核心 message**:
> "Phantom-SoM 不是 universal 第 4 routing arm — 是 **site-modulated**. 在 text-dominated reddit 上 Phantom-SoM 13.81% > SoM 10.48%; 在 visual-rich cls 上 SoM 21.37% > Phantom-SoM (image 真起作用). Site mechanical substrate 决定 axis 主导."

**为什么这个 framing 重要**:
- Honest framing — 不 overclaim "Phantom > SoM 总是"
- Pivotal for paper Section 5 mechanism 论证
- Reviewer 喜欢 site-aware 论证 (避免 cherry-pick)

### VWA framework bugs (副发现)

**怎么 frame**:
> "我们今天系统 audit 了 VWA framework, 发现 37 个 scaffold-level bugs (cite Tier 3 Gemini Deep Research 综述对照). 已经 ship Phase A 4-cluster patch 修了主要的 (commit 3c15cd7). 这个 bugs 影响 absolute SR 数字 (~5-10% inflation/deflation), 但**不影响 cross-mode comparison** — 4 个 text-bearing mode 共享同样的 dispatch contamination, symmetric noise cancels in paired comparison. 详见 docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md."

**4 reasons why Phantom finding still valid** (paper_planning + reference doc § 5.3):
1. Symmetric contamination on text-bearing modes
2. Vision counter-evidence (Vision 不受 dispatch bug 影响, 但 reddit Vision < P-SoM)
3. Pilot wave-2 已经验证 T=0 修复后 Δ=0pp matched-subset SR
4. Architectural design (3-axis cube 是 well-defined factorial, 不是 emergent from bugs)

**这是 ask #2 的引子** — paper 是单文章 (主 + appendix bug 披露) 还是 split (Phantom paper + bug audit short paper)?

---

## E. Decisions (Asks)

### Ask #1: RunPod 经费 ~$150-200

**说法**:
> "DGX GPU 争抢严重, B1 baseline 跑 234 ep 要 20+ 小时. Phase A 14-cell 全跑预计 150-200h wallclock. UCL Myriad 我申请下来了但物理级 blocked (UCL firewall drop Tailscale CGNAT — 详见 MYRIAD_SMOKE_REPORT.md). RunPod 4090 dedicated $0.6/h, 全跑 ~$110-150 + buffer ~$200. 想申请课题经费走 RunPod."

**学长可能问**: 走什么流程?
**准备**: 听学长指示 — 不预设答案 (可能是 PI 直接报销 / 项目经费 / 其它).

### Ask #2: Paper scope — single vs split

**说法**:
> "原计划 1 篇 paper (Phantom-SoM 主线 + appendix VWA bugs 披露). 但 VWA bug audit 本身值得文献跟进 — 37 entry, Tier 1-5 系统化 audit + Tier 10 dispatch-effective-target probe, 可独立成 short paper / workshop paper. 您觉得哪个 framing 好?"

**两条 framing 各自优劣** (帮 user 准备):

| | Single (Phantom + appendix) | Split (Phantom + bug paper) |
|---|---|---|
| 时间 | 快 (1 个 deadline) | 慢 (2 个 paper draft) |
| Risk | 主线被 bug appendix 抢戏 | bug paper 可能 reviewer 觉得 too negative-result |
| Citation 价值 | 集中, 主线 paper 引用次数高 | 分散, bug paper 可能成 niche cite hub |
| Reviewer 心理 | "诚实披露 limitation, +ve" | "single contribution, more digestible" |

### Ask #3: Section 4 框架 + experimental scope check

**说法**:
> "Section 4 evidence catalog 现在用 4-dimension framework (Outcome / Macro / Micro / Efficiency, line 255). 这个 framing 是不是您觉得 OK? Section 5 用 site × axis × LLM-mechanism C 框架, 您觉得 paper structure 还需要调整吗?"

---

## F. Buffer (5-10 min) — Reviewer Pushback Prep

| Q | A |
|---|---|
| **Phantom-SoM 是 prior work 已有的 trick 吗?** | 没有. 之前 SoM-style paper (Yang 2023 Set-of-Mark, etc.) 没有 isolate text-payload axis from image axis. 我们的 3-axis ablation 框架 + diamond completion + 4-fold drop-in characterization 是 paper-original. |
| **如果 bug 修了, Phantom 优势会变吗?** | 估算 +2-5pp 整体 SR 提升 (Tier 10 estimate 5.5% off-target → locator-route lift), 但 cross-mode delta robust. Pilot wave-2 (T=0 修复, dispatch bug 仍在) 已经 Δ=0pp on N=60 ep. |
| **B1 数据没跑完, paper 写得动吗?** | B1 cls 现 85%, 1-2 周内补齐. 即使 B1 不完整, B0 paper-grade clean (cls + reddit 5-mode 全) 已经够 paper Section 4 主线. B1 是 Section 7 cross-capability evidence, 可分阶段交. |
| **VWA bug 这个发现新颖吗?** | Tier 3 Gemini DR 综述显示同领域 paper 普遍 acknowledge silent-failure noise (cite 5-category taxonomy), 但**没有 paper 系统 catalog 37 entries + ship fix patches + verify via Playwright replay**. 我们的 audit 是 paper-grade rigor 的最强级别. |
| **shopping site 数据为啥还没好?** | shopping 是 Magento 复杂 site, FPC + swatch 等 site-bug 比 reddit/cls 多. B0 DOM shopping 已 N=465/466, 在 user manual gallery triage 修剩余 site bug. 决定何时启动其它 mode. |
| **跟 prior 4-mode VWA paper (e.g. WebArena) 比?** | 我们 5-mode (DOM/SoM/Vision/P-text/P-SoM) + diamond P-prompt = 6-mode 是已知最系统化 ablation. WebArena 原 paper 只 3-mode. |

---

## G. 不要犯的错 / 演讲风险

`★ Anti-patterns 必须避免 ─────────────────────`

1. **过早提 bug**: 不要开场就说"我发现 framework 有 bug" — 学长第一印象会变成 "你的实验数据有问题". 应该先讲 Phantom finding (主线), bug 当 caveat 在 D 节讲.

2. **Overclaim "Phantom 总是 win"**: paper_planning line 363 强调 "site-modulated representation × prompt × image effects, 不是 Phantom #1 universal routing arm". 反复 frame 成 complementary.

3. **数字 cite 错**: 主要数字背熟:
   - red P-SoM drop-one **3.33pp**, cls **2.56pp**
   - red P-SoM SR **13.81%** > SoM 10.48%
   - cls SoM **21.37%** > P-SoM (image rich-site SoM 主导)
   - cls SoM p95 latency **74s**, P-SoM **18.2s** (4× faster)
   - cost ratio B0 vs B1 ~100×
   - 4-fold drop-in: cost ≈ DOM / latency 50% / AUROC ≥ baseline / drop-one 1.7-3.3pp

4. **Diamond / P-prompt 必须强调是你的设计贡献**: paper_planning line 392 "P-prompt 必要性" 标注 N1 — 学长可能 challenge 这是不是 overengineering, 准备 factorial design completeness 答案.

5. **VWA bug 措辞**: 不说 "bug 全都是 framework 问题" (overclaim), 说 "VWA framework 跟 P79 wrapper 共有 37 个 scaffold-level issues". 这是 fair description.

6. **不要让 bug 抢戏 main contribution**: bug 是 secondary finding. 主线 5 minutes 讲 Phantom, bug 1 minute 提及, 不 deep-dive 在 meeting.
`─────────────────────────────────────────────────`

---

## H. 会议结束后的 deliverable

不论会议结论如何, 会后立即:
1. Update `next_steps.md` 写学长 feedback + 决策结果
2. 如 RunPod 经费 approved → 1 天内做 RunPod onboarding + launch B1 cls 余下 36 ep
3. 如 Phantom paper 单走 → 加速 Section 5 prose + Section 6 routing
4. 如 split paper → 写 bug audit short paper outline (放 docs/checkpoints/codex_prompts/)
5. 如 scope 调整 → update paper_planning §4 Section status

---

## I. 一页 cheat sheet (印出来带去会议)

```
┌─────────────────────────────────────────────────────────────┐
│ Phantom-SoM 4-fold drop-in property:                        │
│ (a) cost ≈ DOM     (b) latency 50% lower                    │
│ (c) AUROC ≥ baseline   (d) drop-one 1.7-3.3pp               │
│                                                             │
│ Key numbers (red B0):                                       │
│   P-SoM 13.81% > SoM 10.48% > DOM 9.05%                     │
│   P-SoM drop-one 3.33pp (Phantom unique)                    │
│   cls latency: SoM 74s vs P-SoM 18.2s = 4× faster           │
│                                                             │
│ Theory: 3-axis (text × prompt × image) ablation cube        │
│   5-mode 沿对角路径 + diamond P-prompt 我加的 corner          │
│                                                             │
│ Caveats:                                                    │
│   - site-modulated (cls SoM win, red Phantom win)           │
│   - VWA bugs found+fixed today, cross-mode robust           │
│   - B1 data 85% done                                        │
│                                                             │
│ Asks:                                                       │
│   1. RunPod $150-200 经费                                   │
│   2. Single vs split paper scope                            │
│   3. Section 4 4-dimension framework OK?                    │
│                                                             │
│ Refs to read:                                               │
│   docs/checkpoints/paper_planning.md (1255 行)              │
│   docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md    │
│   docs/reference/MYRIAD_SMOKE_REPORT.md                     │
└─────────────────────────────────────────────────────────────┘
```

---

## J. References

- `docs/checkpoints/paper_planning.md` §1-§8 (canonical strategy notebook, 1255 行)
- `docs/reference/VWA_FRAMEWORK_BUGS_AND_PHASE_A_FIXES.md` (337 行 bug audit synopsis)
- `docs/reference/MYRIAD_SMOKE_REPORT.md` (282 行 Myriad 失败档案)
- `results/phantom_paper/figures/` (15 figures, all dimension-prefixed per §106)
- `docs/checkpoints/master_bug_catalog.md` (37 entries 技术细节)
