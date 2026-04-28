# P79 Next Steps — 行动汇总

> 中央 plan 文档。汇总所有 pending experiments / analyses / paper-writing tasks。
> 实验笔记 (`docs/checkpoints/实验笔记.md`) 顶部"行动规划"区是 quick reference，
> 这个文档是详细 owner / blocker / ETA 跟踪。
> 最后更新：**2026-04-27 23:30** (今晚 paper drafts 1/2/3 + 9 figures + 2 analyses 完整)

---

## 0.0 今晚 (2026-04-27) Progress Dashboard ⭐

### Infra 修复

| 修复 | 状态 | Commits / Notes |
|---|---|---|
| **Magento base_url 复发** (docker restart 后 redirect 退回 metis, shopping reset FAIL 3 次) | ✅ **fixed + 三层持久化** | quark side: `magento_baseurl_fix.sh` 维护脚本 + `start_vwa_docker.sh` hook + `reset_shopping.sh` 移除 hardcode localhost; DGX side: `reset_vwa_sites.sh` 加 defensive curl health check 验证 redirect ≠ metis |
| DGX→shopping 可达性 | ✅ HTTP 200 OK (no metis redirect, 04-27 23:00 verify) | shopping 7770 + shopping_admin 7780 都 OK |

### 论证 / Framing 升级（paper-level）

| 升级项 | Commit | 影响 |
|---|---|---|
| Phantom-SoM cost 论证从 "省 image" → "**cost ≈ DOM**" (regex filter same AXTree) | `48db047` | paper hook 上一档；fig7 callout 已落地 |
| **Drop-in deployment intervention** punchline (零模型/prompt 改动 backend 切两步即可) | `ef29add` | Section 1 contribution 1 + Section 3.2 cost 段 |
| 三层 cost decomposition (text / annotation / image-tokens) — deployment-time framing | `93e413f` | Section 3.2 + 实验笔记 §103 |
| Section 3.2 image-token estimate 从 total cost 反推 → step-level `tokens.input` median | `4d63c9f` | reddit 733 / cls 1064 tokens/step (measured n=145/234) |
| `[SOM_MARKS]` token claim 从 "~50% 少" → "AXTree 相当 ±7%" (实测) | `722a97a` `edfc256` | 4 处文档同步修正 |

### Paper Drafts（earlier today done）

| Section | Words | Commit |
|---|---:|---|
| 1 Intro | 786 | `62c1380` (+ `ef29add` deployment framing 强化) |
| 2 Background + paper.bib (16 entries) | 1514 | `206cd93` |
| 3 Definition + Ablation Setup | 863 | `13b9608` (+ `4d63c9f` token re-estimate) |
| 4 Empirical Findings | 1725 | `bfe0154` (待 phantom fresh 数据 prose update) |

### Figures (9 张全数据 ready, 远超 Week 0 末预期)

| Fig | 内容 | Commits |
|---|---|---|
| fig1 | 4-mode venn 2x2 panel (B0+B1 cls+red) | `5736fb4` |
| fig2 | drop-one oracle 2x2 | `5736fb4` |
| fig3 | strategy gradient 2x4 (reddit + cls) + cls footnote | `6960de0` `4d63c9f` |
| fig4 | two-knob diagram schematic | existing |
| fig5 | category × mode heatmap (B0 cls+red) | existing |
| fig6 | capability contrast B0-vs-B1 | `c4b52c3` |
| fig7 | cost-SR Pareto + deployment callouts | `bedcb31` `91367f3` |
| fig8 | overlap-depth stacked bar (5-mode independence) | `b3a5adc` |
| fig9 | regional carbon sensitivity (B1 only scope) | `d3dfc8f` `0cb26c5` |

### Analyses 文档

| 文档 | Words | Commit |
|---|---:|---|
| disagreement_clusters.md (B0+B1 cls+red, 9 categories) | — | `ded0ef6` `c4b52c3` |
| **cross_site_pattern_consolidation.md** ⭐ | 1596 | `ab86019` |
| **B1_capability_profile.md** (6 sections) ⭐ | 2245 | `03ffb2f` |

### 决策入库

| 决策 | 状态 | 来源 |
|---|---|---|
| Final scope: 6 sites × 3 models × 5 modes + deployed router + multi-metric + green AI | ✅ 入库 | §4.5.1 |
| 顶刊概率 final scope: NeurIPS/ICLR 45-60%, MLSys 75-85%, TMLR 75-85% | ✅ 入库 | §4.5.2 |
| Phantom-DOM scope 缩减 18→5 cells (mechanism only) | ✅ 入库 | §4.5.7 |
| Future paper 2 转向 Phase 3 modules (router 整合 paper 1) | ✅ 入库 | §5 |
| First paper 投稿 cascade 修正: round 1 → MLSys (不 NeurIPS, first paper friendly) | ✅ 入库 | §4.5.5 |
| Carbon: 现状 simplified port (45 region OK), Option A+D 推荐 | ✅ 入库 | §4.5.9 |
| Router 难点: 5 决策维度, timeline 修正 ~3-4 周 (vs ~2-3 周) | ✅ 入库 | §4.6.1 |
| Advisor align 时机: Week 3 + Week 6-7 两次 | ✅ 入库 | §4.6.2 |
| 级联 Router 可视化: 4-fig stack (Pareto + Cumulative + Sankey + Histogram) | ✅ 入库 | §4.6.3 |

---

## 当前 status snapshot

```
Active processes (paper-grade clean re-run chain):
  ├─ runner B0_phantom_som_classifieds  (re-run, 82/234, ~35%, ~6-7h ETA)
  ├─ runner B0_phantom_dom_reddit       (123/210, ~59%, ~3-4h ETA)
  ├─ watchdog × 4 (含 phantom_dom cls done watchdog)
  ├─ chain orchestrator B0 cls (waiting som re-run done)
  ├─ chain orchestrator B0 red (waiting dom done → reset → som re-run)
  └─ queue_b1_after_b0 sequencer (waiting all B0 chains)

Active data (paper-grade clean):
  ├─ B0 cls + red 3-mode baseline ✅
  ├─ B1 cls + red 3-mode baseline ✅
  ├─ B0 phantom_dom cls (FRESH 234/234) ✅
  └─ B0 phantom_dom red (123/210, ~59%) 🔄

Pending re-run / waiting:
  ├─ B0 phantom_som cls (82/234 跑中) 🔄
  ├─ B0 phantom_som red (chain auto-trigger after dom done) ⏳
  ├─ B1 phantom_reddit (queue_b1_after_b0) ⏳
  ├─ B1 phantom_classifieds (queue_b1_after_b0) ⏳
  └─ Shopping (B0+B1, all modes, cleared) — 待 Myriad GPU
```

### B0 cls 5-mode 现况（其中 phantom_som 是 stale, 其他 fresh）

| Mode | N | raw | adj | FP gap |
|---|---:|---:|---:|---:|
| DOM (baseline) | 234 | 14.96% | 14.10% | 0.85pp |
| **SoM** (baseline) | 234 | **23.08%** | **21.37%** | 1.71pp |
| Vision (baseline) | 234 | 15.81% | 13.68% | 2.14pp |
| **Phantom-DOM** (FRESH 04-27) | 234 | **16.67%** | **14.53%** | 2.14pp |
| Phantom-SoM (stale .bak, 重跑中) | 234 | — | — | — |

Phantom-DOM **adj 14.53% ≈ DOM 14.10%** (Phantom-SoM 期望 fresh 后类似 SoM 21%, 待验证)
Section 4 prose 待 phantom-SoM re-run done 后一起 fresh-data update.

### Wall-time ETA (2026-04-27 23:30 → 完整 cls+red 5-mode)

```
B0 phantom_som cls re-run    : 现 82/234 → 234 完成 ~Tue 06:00
B0 phantom_dom red continue  : 现 123/210 → 210 完成 ~Tue 02:00
B0 phantom_som red re-run    : chain auto, 完成 ~Tue 16:00
B1 cls chain (som→dom)       : queue_b1 trigger, 完成 ~Wed 08:00
B1 red chain (som→dom)       : 完成 ~Fri 02:00 ⚠️ reddit 慢 2.4x/step

Critical path A 完整 ETA: ~Friday 凌晨 (04-30 / 05-01)
```

⚠️ Reddit 比 cls 慢 2.4×/step (Postmill render 5.4× + AXTree 1.5×).

---

## 0. 完备性审计（2026-04-27 23:30）

### 0.1 数据状态矩阵

| Cell | DOM | SoM | Vision | Phantom-SoM | Phantom-DOM |
|---|---|---|---|---|---|
| **B0 cls** | ✅ stable | ✅ stable | ✅ stable | 🔄 重跑 82/234 | ✅ FRESH 234/234 |
| **B0 red** | ✅ stable | ✅ stable | ✅ stable | ⏳ 待 chain | 🔄 跑中 123/210 |
| **B1 cls** | ✅ stable | ✅ stable | ✅ stable | ⏳ 待 queue_b1 | ⏳ 待 queue_b1 |
| **B1 red** | ✅ stable | ✅ stable | ✅ stable | ⏳ 待 queue_b1 | ⏳ 待 queue_b1 |
| **B1 shopping** | 🟡 466 ep pre-Magento-fix (待 clear+重跑) | 🟡 4 ep partial | ❌ 0 ep | ❌ | ❌ |
| B0 shopping | ❌ cleared 待 Myriad | ❌ | ❌ | ❌ | ❌ |
| WA (3 sites × B0/B1) | ❌ 未跑 | ❌ | ❌ | ❌ | ❌ |

**paper-ready**: baseline 12/20 (60%) + phantom 1/8 (12.5%) (与昨晚相同, 但 phantom 重跑进度推进)
**Shopping 状态修正 (2026-04-27 23:40)**: B1 shopping DOM 466 ep 实际仍存（之前误标 cleared）, 跑期间 Magento bug 可能 active, 决策 clear+重跑 (timing ~Wed advisor align #1 后)

### 0.2 分析层完备性

| 分析 | 状态 | 备注 |
|---|---|---|
| Codex audit (cls/red, A/B/C/D 分类) | ✅ done | shopping, WA 未做 (待数据) |
| disagreement_clusters.md (B0+B1 cls+red, 9 categories) | ✅ done | aggregate cls+red |
| **cross_site_pattern_consolidation.md** (cls vs red 拆分) | ✅ **done** ⭐ `ab86019` | per-site 拆出 SoM hijack +50.0/+33.3pp |
| **B1_capability_profile.md** (6 sections, 2245w) | ✅ **done** ⭐ `03ffb2f` | Section 6 cross-model 准备 |
| Phantom-SoM step traces digest | ❌ 待 phantom-SoM 重跑 done | Section 5 prose 必备 |

### 0.3 核心 paper claim evidence 完备性

| Claim | 当前 evidence | 完备度 | 主要 blocker |
|---|---|---|---|
| **C1**: Phantom-SoM 是 hidden 4th routing arm | fig1/fig2 (stale) + §103 §95 N=48 | 50% ⚠️ | **B0 Phantom-SoM fresh 重跑 — paper 第一论点的根** |
| **C2**: Two-knob (prompt × obs format) | fig3 reddit + fig5 + B0 cls Phantom-DOM ≈ DOM | 60% | red Phantom-DOM 完整 + cls 4 metrics + B0 cls Phantom-SoM fresh |
| **C3**: Capability × representation interaction | fig6 +43.7pp aggregate | 65% | per-site 拆分 (cross-site consolidation 跑中) + B1 phantom |
| **C4**: Cost-SR routing value | fig7 B0+B1 baseline + B0 cls Phantom-DOM | 75% | B1 Phantom 4 cell + red Phantom-DOM |
| **C5**: Generalization (cross-site/benchmark/model) | cls+red 部分 | 30% | shopping + WA + Claude — Section 6 后期 |

### 0.4 9 张 Figure 当前讲故事能力 (今晚 fig3 扩展 + fig7 callouts + fig8/fig9 新增)

| Figure | 能看到的 ⭐ paper-relevant finding | 看不到 / 缺口 |
|---|---|---|
| **fig1** 4-mode venn (2x2) | B0 red Phantom-SoM 10.95% > DOM 9.52%, ≈ SoM 10.48% (反直觉) | B1 phantom panel 全 N/A; B0 phantom-SoM stale; 4 圆视觉拥挤 |
| **fig2** drop-one oracle (2x2) | Phantom-SoM cls **1.71pp** / red **2.38pp** drop-one ⭐; SoM cls 7.69pp 最高 | B1 panel N/A |
| **fig3** ⭐ strategy gradient **2x4 (red + cls)** | DOM search-loop 22.7%→SoM 12%→Phantom-SoM 10.8% **单调梯度** (red); cls Search-loop% 加 footnote 标 cross-site 不可比较 | cls Phantom-SoM n/a (stale .bak 没 step JSONL) |
| **fig4** two-knob diagram | schematic 2x2 ablation 解释图 | 无数据 |
| **fig5** category × mode heatmap | Cat A 全 mode 高; Cat B 仅 SoM/Vision 高 (视觉必需); Cat D 全低 (FP) ⭐ | Phantom-SoM stale; reddit Phantom-DOM partial (auto-fix on chain done) |
| **fig6** capability contrast | B0 SoM early-finish 53.3% → B1 SoM visual-hijack 70.4%, **+43.7pp flip** ⭐⭐ | aggregate cls+red — per-site 拆已在 cross_site_pattern_consolidation.md |
| **fig7** ⭐ cost-SR Pareto (cls+red) **+ deployment callouts** | B0 cls cost: SoM 0.041 ≈ Phantom-DOM 0.040; deployment callout "Phantom ≈ DOM cost" 已 visual emphasis | B1 Phantom 4 点 dotted placeholder |
| **fig8** ⭐ NEW overlap-depth stacked bar | 2x2 panel (B0+B1 cls+red), depth=1 (unique) 数字 = paper hidden-arm hook; B0 cls SoM unique=16, Phantom-DOM=5, Phantom-SoM=1\* | Phantom-SoM stale → unique=1 (期待 fresh 后 ↑); B1 phantom hatched N/A |
| **fig9** ⭐ NEW regional carbon sensitivity (B1 only) | 45 region × 3 modes per-task CO2; cls Vision ≪ DOM (steps 少); red SoM ≈ DOM | B0 proxy API 不可测 (transparent disclose); Phantom modes pending |

### 0.4.5 Theory Framework Refinement (2026-04-28 codex diag 后) ⭐⭐

Phantom-DOM vs Phantom-SoM mechanism diag (commit `5821387`,
`results/phantom_paper/analyses/phantom_dom_vs_som_diagnostic.md`) 显示:
- cls Phantom-DOM = Phantom-SoM SR 14.53% 但 Jaccard **0.447** (task pool 显著不同)
- red Phantom-DOM **11.90%** vs Phantom-SoM **13.81%** (我之前误用 condition_summary
  raw 13.81% = 13.81% 是错的, 实际 episode-level adj 不相同)
- Theory C (prompt 是 task-conditional decision prior) 胜出, 不是简单 commitment knob

**OLD framework (太简单)**:
- two-knob: representation (text format) × prompt (commit confidence)
- visual-hijack 单一 dimension cover image effect

**NEW framework (3-axis hierarchical, multi-dimensional)**:

```
Axis 1: Representation (PRIMARY, first-order SR)
  AXTree vs [SOM_MARKS] → action surface + trajectory basin
  → Phantom 模式获得 4th routing arm 因为 [SOM_MARKS] obs

Axis 2: Prompt (SECONDARY, multi-dimensional, task-conditional)
  DOM-family vs SoM-family within same [SOM_MARKS] obs
  Sub-effects (codex diag 验证):
    (a) search phrasing
    (b) candidate selection (ranking/disambiguation)
    (c) backtracking strategy
    (d) commitment confidence (FP gap evidence, NOT唯一)
  Effect 是 task-pool divergence (Jaccard 0.45-0.54), NOT uniform SR shift

Axis 3: Image (codex diag commit `7106d2e` validated)
  SoM image vs Phantom-SoM no-image, same SoM prompt + [SOM_MARKS]
  Multi-dimensional sub-effects (counted over adj-exclusive task sets):
    Helping dimensions (image-on win):
      cls: Visual context disambig 13, Element disambig 10, Spatial 5, State 1 (total 29)
      red: State/action recognition 6, Visual context 2 (total 8)
    Failure dimensions (image-off win):
      cls: False confidence 8, Image-induced detour 5 (total 13)
      red: ⭐ False visual confidence 9 (60% of red failures), Detour 3,
            Visual-hijack 2 (only 13%!), Misdirection 1 (total 15)
  Effect site-modulated:
    cls: helping 29 vs harming 13 → +6.84pp adj SR (image win 主导)
    red: helping 8 vs harming 15 → -3.33pp adj SR (image hurt 主导)
  Task-conditional (Jaccard adj 0.333 cls / 0.378 red, 不是 monotone)
  Visual-hijack 只是 minor: red 13%, NOT main image-induced failure
  ⭐ False visual confidence (premature commit before text verify) is main
  ⭐ Latency: cls SoM p95 74.0s vs Phantom-SoM 18.2s = 4x faster (no image encoding)
  Capability-modulated: B1 失败 amplify (cross_site +50/+33pp shift)

Cross-axis interaction:
  Site × representation × prompt × image 形成 complex routing landscape
  paper Section 5 mechanism 应当 hierarchical 不是 flat two-knob
```

**Paper Section 1/5 prose 修订要求**:
- ❌ 撤回 "prompt knob mainly changes commitment confidence" (codex `5821387` falsify)
- ✅ 改成 "prompt as task-conditional decision prior over search/select/backtrack/commit"
- ❌ 撤回 "visual-hijack is the main image effect" (codex `7106d2e` falsify)
- ✅ 改成 "image as 8-dimensional channel: 4 helping (spatial/context/element/state)
  + 4 harming (false confidence DOMINANT 60% red, detour, misdirection, hijack 13% red)"
- 保留: drop-in deployment / cost ≈ DOM / hidden 4th arm / cross-capability mild
- ⭐ NEW: latency 4x improvement (cls Phantom-SoM 18.2s vs SoM 74.0s) — paper Section 7
  sustainability framing 新维度

Codex prompt 9 (som_vs_phantom_som) 已 done (commit `7106d2e`), axis 3 evidence
完整. Paper Section 4/5 prose 修订可基于 3-axis hierarchical framework, 待 user
决定时机.

### 0.4.6 3-axis Framework LLM-level Mechanism Rationale (Section 5 写作 reference)

**Each axis 的 LLM 底层 mechanism 都有 prior literature support** (deep research 待 codex #10 verify + cite):

**Axis 1 Representation (text format)**:
- Token distribution shift → attention pattern → next-token logits 不同
- In-context learning bias (pretraining 中 AXTree-like vs numbered-list contexts 不同)
- Long-context attention degradation (lost-in-the-middle)
- Output format priming
- Lit support: ⭐⭐⭐ Sclar 2024 ICLR, Mishra 2022 ACL, Liu 2023 lost-in-the-middle

**Axis 2 Prompt (system prompt as decision prior)**:
- "Phantom prompt" effect: prompt 描述不存在 modality 仍 anchor behavior
- Task instruction reinterpretation under prompt prefix
- Action schema priming (click-by-id vs click-by-text bias)
- Latent state anchoring (prefix attention always-active)
- Lit support: ⭐⭐ persona priming (Salemi 2024), in-context learning (Min 2022),
  but task-pool divergence 0.45-0.54 是 paper 的 unique empirical finding

**Axis 3 Image (multi-dimensional, refined per user critique 04-28)**:

Helping channels:
- Vision encoder 提供 spatial / context / disambig / state recognition

Harming channels (sub-decomposed, **bidirectional modality failure + SoM-specific**):
- **3a. False visual confidence (image-over-text)** [MAIN red 60%, codex 7106d2e]:
  - Mechanism: image-text alignment pretraining bias → premature commit
  - Lit: Li POPE 2023, Yang 2024, Yu 2024 (object hallucination)
- **3b. Text-over-vision fallback fail** (反向 modality bias) [cls task 24, §100]:
  - Mechanism: language prior dominance → ignore actual image content
  - §100 直接量化: B1 WithText (96%) >> B0 WithText (50%) on reddit_164/14
    → B1 给文本几乎完全 ignore image, 比 B0 更彻底 text-over-vision
  - Lit: ⭐ Tong 2024 "Eyes wide shut" (NeurIPS), Bitton-Guetta 2023
- **3c. Visual saliency hijack on image content** [red task 0/167 实测 verified]:
  - Mechanism: same image link 是 attention attractor; page reload 后 element_id
    重新分配 (+2846/cycle), 但 model 仍 click 同一 image link, 15× cycle stuck
  - 与 SoM density 不直接相关 (mark_count 117 ≠ outlier vs P95 127)
  - 是 image content saliency, 不是 numeric mark saliency
- **3d. SoM occlusion** [§100 实测 verified, B0/B1 都 affected]:
  - Mechanism: SoM 标签实心 fill 覆盖元素文字前 3-6 字符 → OCR loss
  - 量化: B0/B1 reddit_task_6 mode-SoM 18%/15% vs NoMarks 78%/75% → -60pp
  - **这是 paper Section 5 直接 measured ground truth evidence**
- **3e. SoM numeric label attention hijack** [§100 实测 verified, density-dependent]:
  - Mechanism: high mark density (>100) → model attention 默认选数字 ID 而非内容
  - 量化: B1 num_ids 在 mode-SoM 下 0→446 at 128 marks; mode-WithText 立即降 0
  - "在标签 + 内容竞争中默认选标签"
- **3f. Text-over-vision rescue** [§100 实测 verified, paper helping channel 反向]:
  - Mechanism: 给 text fallback (mode-WithText), model 完全 recover OCR loss
  - 量化: B0 WithText 80% ≈ B0 NoMarks 78% on reddit_task_6
  - → Phantom-SoM no-image 实际是这个 mechanism 的 systematic deployment

**Bidirectional framing 关键 insight (新)**:
- Task 需 vision: 3b (text-over-vision) 主导 → Phantom-SoM 失败
- Task 不需 vision: 3a (image-over-text) 主导 → SoM 失败
- → site-modulated effect (cls visual-rich vs red text-dom) 的 **LLM-level bidirectional 解释**

Lit support: ⭐⭐⭐ bidirectional modality fusion failure literature
  - Image→text dominance: Li POPE 2023, Yang 2024, Yu 2024 (object hallucination)
  - Text→vision dominance: ⭐ Tong 2024 "Eyes wide shut" (NeurIPS), Bitton-Guetta 2023
  - Bidirectional modality bias: Wang 2024
  - SoM-specific: Yang 2023 (mark numeric label attention), 待 verify SoM density paper

**Cross-axis interaction LLM mechanism**:
- Repr × Prompt: 同 obs token, 不同 prompt prefix → attention(obs|prompt) 不同
- Repr × Image: 同 prompt, image vs no-image → cross-modal vs pure self-attention
- Prompt × Image: SoM prompt + no image (Phantom-SoM) = "phantom prompt" mismatch
- Site × axis: cls visual-bound → image helping dominate; red text-dom → harming dominate

**Paper contribution position** (Section 5 写作 framing):
不是发现新 LLM mechanism (每 axis 都有 prior). Contribution 是:
1. **Systematic isolation**: 2x2 ablation matrix isolate 三轴在同 setup 下 effect
2. **Joint quantification**: web agent multi-step setting jointly form routing pool
   disjoint structure (Jaccard 0.33-0.55)
3. **Site-modulated framework**: cls vs red 自然提供 task-type ablation
4. **Task-conditional task-pool divergence**: 同 SR 不同 task pool, routing 论点
   direct evidence
5. **Drop-in deployment**: LLM mechanism + cost analysis combined → systems-level
   claim

→ Section 5 prose 应当 frame 成 "we don't propose new LLM mechanisms, each axis
has prior literature; what this paper adds is systematic decomposition + web-
agent-setting empirical evidence + drop-in deployment claim". Reviewer 不会攻
"你没新 mechanism" — paper contribution 是 decomposition + framing.

### 0.4.7 §101-103 Additional Findings worth integrating into Section 5/6

(Earlier analyses 2026-04-26 → 04-27 含 valuable mechanism 论点, 现 integrate 入 framework)

**§101.九 Lazy Minimization Hypothesis** ⭐⭐⭐ (Section 5 capability-modulated 主轴):
```
4B small VLM 信号选择优先级:
  数字标签 (高对比) > 文本 (结构化) > 截图内容文字 (低对比 + 遮挡)
Mechanism: small VLM 视觉处理成本/收益比差 → text-over-vision bias 在 small
VLM 更强 (与 Asadi 2026 anchor 一致)
→ 给 capability shift +43.7pp 提供 LLM-level 解释:
  B1 不是"弱视觉" (NoMarks 75% ≈ B0 78%), 是 "lazy minimization 默认选 easy
  signal (numeric labels), 跳过 visual content"
```

**§103 Site-Modulated Universal Effect** ⭐⭐ (Section 5 cross-site mechanism):
```
Reddit: clear strategy gradient (search-loop 27→20→12%, 12.7→9.9→8.1 steps)
Classifieds: 弱 gradient (DOM ≈ Phantom_som macro metric) 但 Jaccard 0.40-0.48
→ Effect 在不同 layer manifest:
  reddit: macro strategy (search-loop, type rate)
  cls: micro element selection ordering (same macro stats, different task pool)
统一 mechanism: representation 改变 default exploration strategy + decision threshold
  → site-specific MAGNITUDE 但 universal EFFECT
```

**§101.1.3 Density-Threshold Capability Hijack** ⭐⭐ (Section 5 axis 3.e refinement):
```
B1 hijack 量化与 mark density 挂钩:
  cls B1 (33-41 marks): hijack 弱, num_ids 0-12, M2 是主要 cost
  red B1 (111-128 marks): hijack 强, num_ids 88-446, M1 主因 -1.9pp 反转
  阈值约 ≥111 marks B1 hijack onset
→ axis 3.e numeric label attention hijack 不是 binary, 是 density-dependent
```

**§103 Literature Gap 5-Dimension Confirmation** ⭐⭐⭐ (Section 2 + 6 backbone, codex #10 deep research anchor):
```
A. SoM lit: 全 bundle text+image, 没人 isolate marks-text
B. Representation routing: 现 routing model-level/modality-level, 非 text-format-level
C. AXTree vs flat list: head-to-head 对比缺失
D. Prompt format sensitivity: anchor (Sclar 2023, Mishra 2022) 但无 web agent 应用
E. Cost-aware web agent: token efficiency 焦于 prune, 不 reformat
Closest prior: FOCUSAGENT (text 压缩) + Yang 2023 (SoM with image)
本工作 = unprecedented synthesis

完整 deep research: docs/literature/The Novelty and Efficacy of SoM Text...md
codex prompt 10 应当 anchor 在这 5 dimensions
```

**§103 WA Generalization Falsifiable Prediction** ⭐ (Section 6 待数据 verify):
```
If H2' (representation 主导): Phantom-SoM 适用所有 text-only web benchmark
WA pilot ≤50 task verify (Jaccard ≤0.5 universal, >0.7 VWA-specific)
WA routing pool 扩展为 {DOM_AXTree, Phantom_SoM_marks_text}, oracle 增益理论 universal
```

**Section 5 prose codex #13 必须 read 这 5 entries (in addition to §100/Q2/diags).**

### 0.5 现在可下结论 vs 待数据 finding（2026-04-28 fresh phantom 完成 + audit 后）

**已稳定（可写进 paper 主文）**:
1. DOM-only mode 在 visual-required task 上系统性失败 (fig5 cat B)
2. **Image effect 是 cross-capability + site-modulated + 8-dimensional 现象** ⭐⭐ (codex `7106d2e` validated):
   - **8-channel multi-dimensional** (NOT 单一 visual-hijack):
     * 4 helping: spatial grounding (cls 5), visual context (cls 13 / red 2), element disambig (cls 10), state recognition (cls 1 / red 6)
     * 4 harming: **false visual confidence (cls 8 / red 9 ⭐ MAIN red failure 60%)**, image-induced detour (cls 5 / red 3), visual-hijack (cls 0 / red 2 only 13%), misdirection (red 1)
   - **Capability-modulated**: B0→B1 shift +50.0pp cls / +33.3pp red (B1 4B 模型 amplify failure dimensions)
   - **Capability-agnostic mild form**: 即便 B0 frontier (235B) 在 reddit 上 Phantom-SoM 13.81% ≥ SoM 10.48% (no image → harming dimensions 不出现)
   - **Site-modulated**: cls helping 29 vs harming 13 → image win +6.84pp; red helping 8 vs harming 15 → image lose -3.33pp
   - **Task-conditional**: Jaccard adj 0.333 cls / 0.378 red (not monotone "more visual" knob)
   - ⭐ **Latency 4x improvement**: cls SoM p95 74s vs Phantom-SoM 18.2s — production-relevant sustainability dimension
3. DOM 在 reddit search-loop 22.7% (vs SoM 12%) — cls 不直接对比 (search 是 OSClass 核心 workflow, fig3 已加 footnote)
4. **Phantom-DOM cls adj 14.53% ≈ DOM 14.10% (n=234, FRESH)** — prompt-as-commitment-knob 直接 evidence ⭐
5. **Phantom-SoM cost ≈ DOM cost (deployment-time)** — `[SOM_MARKS]` 是 AXTree regex filter, 不需 bbox/image 处理 ⭐⭐
6. **Image tokens per step (measured medians)**: red 733 / cls 1064 (SoM input - Phantom-DOM input, n=145/234) `4d63c9f`
7. Cost gap B0 vs B1 ~30× (Pareto frontier 形状清楚, fig7)
8. **Phantom-SoM unique tasks 验证**: red 7 task (3.33% drop-one), cls fresh 5 task (vs stale 1 — 验证 fresh 比 stale 更强)
10. **Watchdog NOT LOGGED IN site-specific + paper-grade 数据真正完全干净** (2026-04-28 final audit):
    - **cls (OSClass)**: 真实 auth issue. Clear+resume 后 storage_state stale → task 0-3 真 NOT LOGGED IN → watchdog ALERT + `auth_refresh` 真 sign in → task 4 restored → **auto-cleaned 4 NOT-LOGGED-IN episodes** (delete summary/steps/artifacts + remove from seen_keys) → runner resume **重跑 task 0-3** with fresh logged-in storage_state. 历史 cls phantom_som/phantom_dom 各 ~4-5 events 都 auto-cleaned + 重跑.
    - **red (Postmill)**: 0 events — Postmill cookies 持久, reset+resume 不 issue
    - **shopping (Magento)**: FPC false alarm — server session valid, 但 homepage `/` 是 FPC cached guest page → watchdog 误判. Fix: quark side `bin/magento cache:disable full_page` ✅ done 2026-04-28 + PowerShell hook 持久化
    - **paper-grade 数据真正完全 clean (0% wasted task)**:
      - cls 早期 NOT LOGGED IN tasks 被 watchdog auto-clean + 重跑 → final episode summary 都是 fresh logged-in (verified mtime 19:45-19:47 vs watchdog event timeline)
      - red 数据本来就 clean
      - shopping NEW launch FPC disabled, clean
      - **完全 NOT 之前 framing 的 "~2% noise" — 实际 0% wasted, all 5 mode fully fresh paper-grade**
    - Watchdog auto-clean protocol (commit history): delete summary/steps/artifacts + purge digest + remove seen_keys → resume 自动重跑 → 实现 paper-grade clean re-run guarantee
    - Section 4 prose 不需要 disclose noise — watchdog 协议保证 fresh logged-in episodes only

9. **⭐⭐⭐ Phantom 模式 routing signal 完整 + ≥ baseline** (新发现 2026-04-28):
   - 5/5 phantom condition `overall_usable=True`, signal extraction 直接复用 baseline infra
   - Behavioral AUROC 0.694-0.733, Verbalized AUROC 0.701-0.793 (与 baseline 同 tier)
   - **red Phantom-DOM verbalized AUROC 0.793 是 5-mode 最高** (超 baseline 0.766)
   - cls site behavioral 主导 (action_diversity), red site verbalized 主导 (ep_mean_verbalized) — 与 site-modulated effect 一致
   - Token-level signal 全 non-discriminative — paper 应避免 token-level routing claim
   - **paper 论证升级 (paper 1 router section 直接用)**: Phantom 是 4-fold drop-in property:
     (a) Cost ≈ DOM (deployment)
     (b) Latency 4x 改进 (cls 18s vs SoM 74s)
     (c) Signal AUROC ≥ baseline (router 复用 infra) ⭐ NEW
     (d) Drop-one oracle 1.7-3.3pp (routing value)
   - **paper 1 router section 论证 backbone**: router trained on baseline 可 directly extend to Phantom (signal infra drop-in), 不是 future work

**Honest framing 警告 (audit 04-28)**:
- Phantom-SoM red 13.81% > SoM 10.48% **不是 unconditional dominance** — fresh vs stale 同 condition 也有 ±5% task variance (16/210 task 不同), +3.33pp ≈ 2σ noise floor
- 主 narrative 应当是 "**site-modulated representation effect**" NOT "Phantom-SoM #1 routing arm"
- cls SoM 显著领先 Phantom-SoM (+6.84pp adj SR) 反例必须明示, 不要 cherry-pick reddit
- Section 1+4 prose 建议 "matches or modestly exceeds" / "comparable to top arms" 而不是 "clearly exceeds" / "first place"
- **prompt knob 撤回 commit-only claim** (Theory C codex 验证): 改 "task-conditional decision prior over search/select/backtrack/commit"
- **image effect 撤回 visual-hijack-only claim**: 改 "multi-dimensional capability + multi-dimensional failure dimensions" (待 codex #9 verify)
- paper drafts (Section 1, 4, 5) 要 update reflect 3-axis framework — 但 user 决定 "之后再做, 现阶段先收 evidence"

**待数据**:
- cross-site Phantom-DOM 一致性 (red 已 done, cls 已 done — 都 fresh ✅)
- B1 phantom 在 cost-SR 上的位置 (queue_b1_after_b0 trigger 后, B1 cls 跑中 GPU 共享 ETA ~7-10d)
- shopping / WA / cross-model (Claude Opus) 的 generalization

### 0.6 Paper Section 论证强度 + critical path（今晚 cross-site + B1 profile + 9 figures 后）

| Section | evidence 质量 | 状态 | Hard blocker |
|---|---|---|---|
| 1 Intro | ✅ 已写 (786w + drop-in framing 强化) | done | — |
| 2 Background + paper.bib | ✅ 已写 (1514w, 16 entries) | done | — |
| 3 Definition + Ablation | ✅ 已写 (863w, token re-estimate corrected) | done | — |
| 4 Empirical | 🟡 70% (figures done, prose 待 fresh phantom 数据) | drafted (1725w stale) | B0 phantom-SoM 重跑 + B1 phantom (~48h) |
| 5 Mechanism | 🟡 65% (fig3 cls+red + fig5 + cross-site +50/+33pp ⭐) | data 充分 | Phantom-SoM step traces digest |
| 6 Generalization | 🟡 40% (B1 capability profile done) | partial | shopping + WA + cross-model (~周-月) |
| 7 Discussion | ❌ 未写 | end-stage | 全部 data done |

**Critical path A** (Section 4-5 prose): B0 phantom-SoM cls (~6h) + red (~14h) + B0 phantom-DOM red (~3h) ≈ **~24h** to fresh data, +Phantom-SoM step traces digest = ~36h
**Critical path B** (Section 6 generalization): shopping + WA + cross-model = ~周-月，独立 critical path

### 0.7 下一步 codex 任务（按论证收益/token 比）

| # | Task | Tokens | 论证收益 | Blocker | Status |
|---|---|---|---|---|---|
| 1 | fig3 cls 扩展 (2x4 grid) | ~80K | ⭐⭐ | — | ✅ done `6960de0` |
| 2 | fig7 deployment annotation upgrade | ~30K | ⭐ | — | ✅ done `91367f3` |
| 3 | fig8 stacked bar overlap depth | ~120K | ⭐⭐ | — | ✅ done `b3a5adc` |
| 4 | fig9 regional carbon sensitivity | ~80K | ⭐⭐ | — | ✅ done `d3dfc8f` `0cb26c5` |
| 5 | B1 capability profile 单独文档 | ~300K | ⭐ | — | ✅ done `03ffb2f` |
| 6 | Cross-site pattern consolidation | ~200K | ⭐⭐ | — | ✅ done `ab86019` |
| 7 | fig3 cls metric footnote + Section 3.2 token re-estimate | ~50K | ⭐⭐ | — | ✅ done `4d63c9f` |
| 8 | **Phantom-DOM vs Phantom-SoM mechanism diag** (Axis 2 prompt) | ~300K | ⭐⭐ Theory C validate | — | ✅ done `5821387` |
| 9 | **SoM vs Phantom-SoM mechanism diag** (Axis 3 image multi-dim) | ~350K | ⭐⭐ Theory image multi-dim validated | — | ✅ done `7106d2e` |
| --- | --- | --- | --- | --- | --- |
| 10 | **Axis 2/3 literature deep research + paper.bib expansion** (16→~38 entries, 含 bidirectional modality + Tong 2024 Eyes Wide Shut) ⭐⭐⭐ | ~400-600K | ⭐⭐⭐ Section 5 grounding + reviewer resilience | — | 🟡 prompt ready, 发 ~Wed/Thu |
| 10.5 | ~~Axis 3 sub-mechanism 专项 audit~~ — **不需要** (§100 already measured + Q2 verified) | — | — | — | ✅ 已用 §100 SoM probe 测试 + red task 0/167 verify, 不需要重做 |
| 10.6 | **Existing codex docs refine 决策**: 不重做, lazy integrate via Section 5 prose | — | — | — | ✅ verdict: 5 docs 80-95% valid, prose 写时 codex 一次 integrate refined framework + §100 + Q2; 不重做 saves ~1M tokens |
| 11 | Section 4 fresh-data prose update (3-axis framework + cost/latency 4x finding) | ~30K | ⭐⭐ paper hook 升级 | 等 #10 一起发 | ⏳ ~Wed |
| 12 | Phantom-SoM step trace digest (extra Section 5 case studies, supplementary) | ~400K | ⭐ supplement #8/#9 已有 14 case studies | 14 case 已在 #8/#9 cover | ⏳ optional, Thu+ |
| 13 | Section 5 prose 写 (3-axis hierarchical + literature inline cite) | ~50K | ⭐⭐⭐ paper Section 5 main | 待 #10 + #11 done | ⏳ ~Thu-Fri |
| 14 | Codex audit shopping VWA (466) | ~500K | ⭐ fig5 扩 shopping panel | 待 shopping 数据 | ⏳ Myriad |
| 15 | Codex audit WA tasks (480) | ~500K | ⭐ fig5 扩 WA panel | 待 WA 数据 | ⏳ Week 4-5 |
| 16 | Section 6 Generalization 草稿 | ~50K | ⭐⭐ | 待 WA + Claude done + 二次 lit research | ⏳ Week 6-7 |
| 17 | Section 7 Discussion 草稿 (含 sustainability + latency 4x finding) | ~30K | ⭐ | 待全部 data done | ⏳ Week 8+ |
| 18 | 二次 deep research (Section 6/7 + 全 paper revisit, 终稿前) | ~300K | ⭐ paper 终稿前 review robustness | Week 8+ paper 终稿前 | ⏳ Week 8+ |

---

## 1. 实验队列（优先级排序）

### 1.1 立即跑中（不要动）

| Cell | Status | ETA | Owner |
|---|---|---|---|
| B0 phantom_dom cls (chain step 1) | 跑中 | ~3-6h | runner 自动 |
| B0 phantom_dom red (chain step 1) | 跑中 | ~3-6h | runner 自动 |
| Chain B0 cls (dom→som) | sleeping | ~6-12h total | chain script 自动 |
| Chain B0 red (dom→som) | sleeping | ~6-12h total | chain script 自动 |

### 1.2 自动 follow-up（chain done 后立即触发）

| Cell | Trigger | Owner |
|---|---|---|
| B0 phantom_som cls re-run (fresh state) | chain B0 cls dom 完成后 | chain 自动 |
| B0 phantom_som red re-run | chain B0 red dom 完成后 | chain 自动 |
| B1 phantom_som cls + B1 phantom_dom cls (sequential GPU) | queue_b1_after_b0: B0 全 chain 退出 | sequencer 自动 |
| B1 phantom_som red + B1 phantom_dom red | queue_b1_after_b0: B1 cls done | sequencer 自动 |

### 1.3 等资源（手动决定时机）

| Cell | Blocker | 启动命令 | Cost |
|---|---|---|---|
| WA × 3 sites × B0+B1 × som+dom (12 cells) | 学长 align + 当前 chain done | `make phantom-wa-all && make phantom-dom-wa-all` | ~$6 + 24h GPU |
| Cross-model Claude Opus 4.7 cls+red 4-mode | 学长 align + agent 适配 | (需先 add Anthropic agent) | ~$70 |
| B0_3mode_shopping (DOM/SoM/Vision baseline) | Myriad GPU 上线 | 改写 queue_b0_with_reset 的 site filter | ~$7 + 18h API |
| B0/B1 phantom shopping (4 cells) | Myriad GPU + B0 baseline first | sequential after baseline | ~$10 + 24h |
| B1 phantom shopping (2 cells) | Myriad GPU 独占 | `make phantom B=B1 M=som S=shopping` | ~12h GPU 独占 |

---

## 2. Codex 任务队列

### 2.1 已完成 (累计)

**Earlier (今晚之前)**:
| Task | Commit |
|---|---|
| Section 4 draft (1725w) + 4 figure scripts | `bfe0154` |
| Disagreement task cluster analysis (B0 cls+red) | `ded0ef6` |
| B1 disagreement capability contrast (+43.7pp aggregate) | `c4b52c3` |
| Section 2 Background + paper.bib (1514w, 16 entries) | `206cd93` |
| Section 1 Intro (786w) | `62c1380` |
| Section 3 Phantom-SoM Definition (863w) | `13b9608` |
| figs 1/2/5/6/7 update + scaffolding | `5736fb4` |

**今晚 2026-04-27 完成 (8 个 commits)**:
| Task | Commit |
|---|---|
| fig7 live Pareto frontier B0/B1 baseline + B0 Phantom-DOM cls | `bedcb31` |
| fig7 deployment annotation callouts ("Phantom ≈ DOM cost") | `91367f3` |
| fig3 cls extension (2x4 grid reddit + cls) | `6960de0` |
| fig9 regional carbon sensitivity (45 region × B1 measured) | `d3dfc8f` |
| fig8 overlap-depth stacked bar (5-mode independence) | `b3a5adc` |
| **cross_site_pattern_consolidation.md** (1596w, +50.0/+33.3pp shift) | `ab86019` |
| **B1_capability_profile.md** (2245w, 6 sections) | `03ffb2f` |
| fig3 cls footnote + Section 3.2 token re-estimate (1064/733 measured) | `4d63c9f` |
| fig9 B1-only scope clarification | `0cb26c5` |

**Paper-level framing 升级 (additional commits 今晚)**:
| Update | Commit |
|---|---|
| `[SOM_MARKS]` token claim 校正 (≈AXTree, 不是 ~50% 少) | `722a97a` `edfc256` |
| Phantom-SoM cost decomposition 三层 (deployment-time framing) | `93e413f` |
| Phantom-SoM cost ≈ DOM (regex filter same AXTree) | `48db047` |
| Drop-in deployment intervention punchline | `ef29add` |

### 2.2 已发包跑中

(空 — 全部今晚 prompts 完成)

### 2.3 Figures 当前数据完整度审计 (2026-04-27 23:30)

| Figure | 数据完整度 | 备注 |
|---|---|---|
| fig1 4-mode venn (2x2) | ~63% | B0 Phantom-SoM `.bak_pre_rederive` (stale) + B1 Phantom 全列 N/A |
| fig2 drop-one oracle (2x2) | ~63% | 同上 |
| **fig3 strategy gradient (2x4)** | ✅ 90% | reddit §103 anchors + cls live; cls Phantom-SoM n/a (stale .bak 没 step JSONL) |
| fig4 two-knob diagram | ✅ 100% | schematic |
| fig5 category × mode heatmap | ~70% | B0 Phantom-SoM stale + red Phantom-DOM partial (auto-fix on chain done) |
| fig6 capability contrast | ✅ 100% | parses disagreement_clusters.md aggregate |
| **fig7 cost-SR frontier + callouts** | ✅ 90% | B1 Phantom 4 点 dotted placeholder (auto-fix B1 chain done) |
| **fig8 overlap-depth stacked bar** | ~75% | B0 cls Phantom-SoM unique=1 (stale, expect ↑); B1 phantom hatched N/A |
| **fig9 regional carbon (B1 only)** | ✅ 90% | B1 phantom modes pending re-run (auto-fix on chain done); B0 不可测已 transparent disclose |

→ 9 figures 全数据 / scaffolding ready，剩余 25-30% 待 phantom 重跑后 `make figures` 自动 regen。

### 2.4 现在能发包（数据 ready, 不依赖 phantom 重跑）

(空 — 现阶段 paper-grade evidence work 已完成。下一波 codex prompts 等 phantom 重跑 done 后启动)

### 2.5 等数据后发（codex prompt 已 prep，等 trigger）

| Task | Blocker | 触发条件 |
|---|---|---|
| Phantom-SoM step traces 补 (disagreement analysis) | B0 phantom_som cls + red chain done | watchdog ntfy COMPLETE 后手动 trigger codex |
| WA disagreement analysis | WA × 3 sites × B0+B1 done | 同上 |
| Cross-model Claude disagreement | Claude Opus 4.7 run done | 同上 |
| Section 5 prose update with phantom traces | Phantom traces analyzed | 同上 |
| Section 6 Cross-Site/Model Generalization draft | WA + Claude data done | 同上 |
| Section 7 Discussion + Future Work | 全部 data + Section 1-6 done | 最后 |

---

## 3. 决策待 align（学长讨论）

| 决策 | Options | 推荐 | 影响 |
|---|---|---|---|
| Paper split | (a) Phantom-only paper + Routing follow-up, (b) Combined paper | **(a)** | 决定 Section 6 内容 |
| Cross-model 范围 | (a) 0 model, (b) 1 model (Opus 4.7), (c) 2 model (Opus + Nova) | **(b) Opus 4.7 同价 + Most Capable** | 决定 Section 6 / 是否 Cross-model 章节 |
| 跨 benchmark | (a) VWA+WA 够, (b) 加 Mind2Web | **(a) 够** | reviewer 接受度 |
| Routing experiment 时机 | (a) 这一篇就做, (b) follow-up paper | **(b) defer** | Paper structure |
| Shopping Myriad 时机 | 等 Myriad 上线（不可控） | wait | 时间线 |

---

## 4. Paper 1 (Phantom-only) Section 状态（2026-04-27 23:30 update）

| Section | Status | Words | Commits |
|---|---|---:|---|
| 1. Intro | ✅ Done (含 drop-in framing) | 786 | `62c1380` + `ef29add` |
| 2. Background + Related Work + paper.bib (16 entries) | ✅ Done | 1514 | `206cd93` |
| 3. Phantom-SoM Definition + Ablation Setup (含 token re-estimate corrected) | ✅ Done | 863 | `13b9608` `4d63c9f` `48db047` `93e413f` `722a97a` `edfc256` `ef29add` |
| **4. Empirical Findings** | 🟡 Draft done, 待 fresh data update | 1725 | `bfe0154` (待 phantom 重跑 prose update) |
| 5. Mechanism (Two-Knob) | 🟡 evidence 充分 (fig3 cls+red + cross-site +50/+33pp ⭐), prose 待写 | — | data ready |
| 6. Cross-Site/Model Generalization | 🟡 partial (B1 capability profile ✅ done, 待 WA + Claude) | — | `03ffb2f` (B1 profile) |
| 7. Discussion | ❌ 未写 (含 sustainability framing slot) | — | end-stage |

Section 1-3 总 prose **3163 words** (paper-ready)，Section 4 draft 1725w 待 fresh data prose update。
Section 5 evidence: fig3 (cls+red) + fig5 + fig6 + cross-site consolidation +50/+33pp + 9 figures stack 全 ready。
Section 6 evidence: B1 capability profile done, 等 WA + Claude 跑完后 prose。

---

## 4.5 Paper 1 顶刊 Execution Plan（2026-04-27 晚 final scope 定）

### 4.5.1 Final scope (paper 完整版)

```
Benchmark: VWA 3 站 (cls 234 + red 210 + shop 466) + WA 3 站 (red 106 + shop 192 + sa 182)
           = 6 sites, ~1390 task per condition
Models:    B0 (Qwen3-VL-235B proxy) + B1 (Qwen3-VL-4B local) + Claude Opus 4.7
           = 3 model families
Modes:     DOM / SoM / Vision / Phantom-SoM / Phantom-DOM = 5 modes
Cells:     6 sites × 3 models × 5 modes = ~90 cells (~125K episode total)
+ Router:  build router using disagreement-cluster / capability / category signals
+ Deploy:  实际 run agent with router, measure cost/SR/latency vs best-single-mode
```

### 4.5.2 顶刊概率（final scope + multi-metric/green AI 加成 后）

| 投稿目标 | 概率 (final scope only) | **+ multi-metric + green AI** | 上调 |
|---|---:|---:|---:|
| **NeurIPS / ICLR main** | 40-55% | **45-60%** | +5pp |
| **ICML** | 35-50% | 40-55% | +5pp |
| **ACL / EMNLP main** | 45-60% | 50-65% | +5pp |
| **MLSys** | 60-70% | **75-85%** ⭐ | **+15pp** multi-metric Pareto + drop-in + green AI 完美 fit |
| WWW / WSDM | 70-80% | 75-85% | +5pp |
| NeurIPS D&B | 60-70% | 70-80% | +10pp |
| **TMLR (journal)** | 65-80% | **75-85%** | +10pp |

→ Final scope + multi-metric + green AI 完成后:
  - **MLSys 几乎 lock (75-85%)**
  - NeurIPS/ICLR 命中 ~50%+
  - 顶刊出版几乎 100% (cascade 4 venue)

**Multi-metric/Green AI axis 加成的 paper-level 价值**:
1. Differentiator: 现有 web-agent paper (VWA/WebArena/SeeAct/SoM/FocusAgent) 几乎全部不报 carbon
2. Multi-metric Pareto 在 ML 顶会近年是 expected
3. 正好支撑 drop-in framing (cost+latency+carbon 三向论点, narrative 更立体)
4. Green AI 是 NeurIPS/ICLR 近 2 年新兴关注点 (Strubell ACL 2019, Patterson 2021)

**Caveat (重要)**: green AI 是 second-order 价值, **不能抢主线**.
- ❌ 不要把 paper 改成 "Green AI for Web Agents" (retrofit 痕迹)
- ✅ 保持主线 "Phantom-SoM hidden routing arm + drop-in deployment"
- ✅ Multi-metric / green AI 作为 Section 4 supporting + Section 7 implications

**Reviewer 风险 + 应对**:
- Risk: B0 carbon NaN reviewer 会问 "怎么能 claim green AI"
- 应对: 透明 disclose (B0 proxy API 远端 GPU 不可测), B1 measured + B0 token-based estimate

**对比 baseline**: VWA (Koh ICLR 2024) / WebArena (Zhou ICLR 2024) / SeeAct (Zheng ICML 2024) / SoM (Yang NeurIPS 2023) / FocusAgent (Kerboua EMNLP 2025) — 你 final scope 比上述任何一个 axis coverage 都全 (2 benchmark family × 3 model × deployed router).

### 4.5.3 Timeline 估算（execution-only, 不含 paper writing）

| 阶段 | 任务 | 时间 | 资源 |
|---|---|---|---|
| 1 | cls+red 5-mode B0+B1 完整 (current paper-grade clean re-run) | ~48h | 现有 GPU |
| 2 | shopping (B0+B1, 5 modes, 10 cells) | ~3-5 天 | Myriad GPU 待 |
| 3 | WA 三站 (B0+B1, 5 modes, 30 cells) | ~1-2 周 | 现有 GPU |
| 4 | Claude Opus 4.7 (6 sites × 5 modes, 30 cells) | ~3-5 天 | $70-150 API |
| 5 | Router design + train + offline eval + deploy on agent | ~2-3 周 | 现有 + API |
| 6 | Paper writing + figures + 投稿 | ~3-4 周 | — |
| **Total** | | **~3 月 (12 周)** | — |

### 4.5.4 4 个关键 Risks + Mitigation（按重要性排序）

#### Risk 1: Execution quality（顶刊成败 #1 因素 ⚠️⚠️⚠️）
90 cells × ~1390 task = ~125K episode。任何 cell 跑 sloppy（auth bug / cross-contam / 数据污染 / FP 没处理）都被 reviewer 抓出。

**Mitigation**:
- 维持现有 paper-grade re-run 协议: reset between conditions, exclusive same-site B0 XOR B1, watchdog auto-rederive
- 每 cell 完成后立刻 `make analyze` + manual audit gallery
- 不在 execution quality 妥协（哪怕慢也不要 cherry-pick / 跳过 reset）

#### Risk 2: Story discipline ⚠️⚠️
6×3×5 cells 容易让 paper 变 "data dump"。顶会 reviewer 反感 "everything but the kitchen sink"。

**Mitigation**: **Single narrative**:
> "Phantom-SoM is a hidden routing arm + we explain why + we route on it + here's the cost saving"

其他 finding (capability shift / category profile / etc) 都是 supporting evidence, 不是独立 contribution. Section 4-5 each ≤4 pages, supplementary 装其余.

#### Risk 3: Router design ⚠️⚠️
Router 只比 best-single-mode 提升 1-2pp 被 reviewer 说 "不值"。用 oracle features (test-time leak) 直接 reject.

**Router design tiers**:
- **Tier 1 (must-have)**: Oracle router — task feature → best mode lookup, train 在 train split, 测在 held-out test split
- **Tier 2 (great-to-have)**: First-step-trigger router — 看 step 1 obs 决定 mode, no test leak
- **Tier 3 (stretch)**: Online learning router — mid-trajectory escalation

Tier 1 + Tier 2 就够顶会 contribution; Tier 3 stretch goal.

#### Risk 4: Negative results 必须诚实报告 ⚠️
某些 cell 可能反 trend (e.g. Claude shopping Phantom-SoM 不 work)。**绝不 cherry-pick**, reviewer 看出直接 reject.

**Mitigation**: 诚实报告反而强化 mechanism claim ("effect 是 model-capability-bound, 不是 universal").

### 4.5.5 投稿 Cascade Plan

```
Round 1 (T+12 周, paper done):
  → NeurIPS 2026 (or ICLR 2027 if timing 错过 NeurIPS)
  → 50% expected outcome

Round 2 (rejection 后):
  → ACL / EMNLP (大幅修改 narrative for NLP venue)
  → 50% expected outcome

Round 3 (still rejected):
  → MLSys (强调 drop-in deployment 角度)
  → 65% expected outcome

Round 4 (保底):
  → TMLR (journal rolling review)
  → 70% expected outcome
```

期望出版 venue 链 ~99%（5 站 5 model deployed-router scope 没法被全拒）, NeurIPS/ICLR 命中 ~50%.

### 4.5.6 关键决策点

| 时间点 | 决策 | 影响 |
|---|---|---|
| 当前 (week 0) | 维持 paper-grade re-run 纪律 | execution quality risk ↓ |
| Week 4 (cls+red done) | Claude Opus 启动决定 | API budget vs scope |
| Week 6 (shopping done) | Router design 启动 | Tier 1/2/3 选择 |
| Week 8 (WA + Claude done) | Router 实际 deploy 时机 | paper 主 contribution 成型 |
| Week 10 (router done) | Paper writing 启动 | story discipline 决策 |
| Week 12 (paper done) | 投稿 venue 决策 | NeurIPS vs ICLR vs ACL timing |

### 4.5.7 Phantom-DOM scope 缩减 (2026-04-27 晚)

Phantom-DOM 是 **mechanism ablation** (two-knob: 同 obs 不同 prompt), 不是 routing arm 候选. 所以不需要 cross-site cross-model 全跑.

**Phantom-DOM 实际必需 scope (5 cells, 节省 70%)**:
- B0 cls + red (✅ done / 🔄 跑中)
- B1 cls + red (待 queue_b1_after_b0)
- Claude cls 1 site (cross-model frontier validation, ~$15 API)

**不需要的 Phantom-DOM cells**:
- B0/B1/Claude × shopping (mechanism 论证不需要)
- B0/B1/Claude × WA × 3 sites (mechanism 论证 VWA 已够)

**Phantom-SoM 仍需要全 scope** (18 cells: 3 model × 6 site) — routing arm 主体，drop-in deployment claim 需要 cross-site cross-model validation.

### 4.5.8 Week 0 行动清单 (2026-04-27 → 2026-05-03)

**🟢 已完成 (Day 1, 2026-04-27 晚)**:
- ✅ 维持现有 paper-grade chain (B0 phantom_dom red 跑中, queue_b1_after_b0 sequencer)
- ✅ 5 codex prompts 全部 done (fig3 cls / fig7 callouts / fig8 / fig9 / B1 profile)
- ✅ Cross-site cluster consolidation done (`ab86019`)
- ✅ Section 3.2 image-token 数字 measured corrected (`4d63c9f`)
- ✅ Paper-level framing 升级 (drop-in deployment intervention) 落地 Section 1/3 + 实验笔记

**🟢 Day 2-7 应做**:
1. Monitor chain progress (auto, 不需手动)
2. ~Tue PM (B0 phantom-SoM cls+red done) — 发 codex prompt 8: Phantom-SoM step trace digest (~400K)
3. ~Wed (B1 phantom cls done) — 发 codex prompt 9: Section 4 + figures fresh-data update (~30K)
4. ~Thu (B1 phantom red done) — 发 codex prompt 10: Section 5 prose draft (~30K)
5. Week 1 末: 第一次 advisor align checklist 准备 (router scope + Claude budget + 单/双 paper)

**🟡 准备 (不动手)**:
- Claude Opus API key + cost budget align (启动决定 ~Week 3, advisor align 后)
- Router design literature search (启动 ~Week 4-5, baseline 全 done 后)
- Shopping pipeline check (Magento auth, etc. — 等 Myriad GPU 上线时一起)

**🔴 不要做**:
- ❌ shopping (B0+B1, 等 Myriad GPU + advisor align)
- ❌ WA (任何 site, 等 cls+red 完整 + advisor align)
- ❌ Claude API run (等 cls+red 完整 + advisor align)
- ❌ Router design 实际写代码 (等 baseline + phantom 完整再启动 ~Week 4)
- ❌ Section 4-5 prose (等 fresh phantom data, ~Wed-Thu)
- ❌ Mind2Web (out of scope per Plan B)

**Week 0 末 (2026-05-03) 预期状态**:
- ✅ B0 phantom 5-mode cls + red 完整 (fresh paper-grade clean)
- ✅ B1 phantom 5-mode cls 完整 (B1 red 还在跑)
- ✅ 9 paper figures 全数据完整 (auto-regen 完成)
- ✅ Cross-site pattern consolidation done (今晚已 done)
- ✅ B1 capability profile done (今晚已 done)
- ✅ Section 1/2/3 paper drafts done (今晚已 done)
- 🟡 Section 4 fresh-data prose update (codex prompt 9 done)
- 🟡 Section 5 prose draft (codex prompt 10 done)
- ⏳ Phantom-SoM step trace digest done

### 4.5.9 Cost / Latency / Carbon Metrics — Paper 利用规划

**已有数据状况** (per condition_summary_v2.json):

| Backend | Cost | P95 Latency | Energy (kWh) | CO2e (kg) |
|---|---|---|---|---|
| B0 (proxy 235B API) | ✅ | ✅ | ❌ NaN | ❌ NaN (token-estimate-able) |
| B1 (local 4B GPU NVML) | ✅ | ✅ | ✅ | ✅ |

理由: B0 inference 在远端服务器, DGX local GPU 不动, NVML 测不到; B1 走 local GPU 有真实测量.

**Carbon tracker 现状 (`p79/experiment/energy_tracker.py`)**:
- ✅ 已 port: NVML GPU measurement + 45 region intensity table (REGION_INTENSITY_G_PER_KWH, 含 30+ country/region: world/USA/EU 各国/Asia-Pacific/Americas/Mid-East-Africa)
- ❌ 未 port (但 external_code 有, 数据可 import): 220+ country database (CodeCarbon/OWID), cloud provider GCP/AWS/Azure region data, network backbone emission, token-based proxy API estimator
- ⚠️ Default region: UK 220 g/kWh (configs/exp_v2_base.yaml:80-81)

**paper 完善 carbon evidence 的 3 个选项**:

| Option | 工作量 | Paper 影响 | 推荐 |
|---|---|---|---|
| A. 现状不动 (B1 measured + B0 不报) | 0 | 主线稳, 透明 disclose | ⭐ default |
| B. 集成完整 codecarbon (220+ country) | ~3 天 | 边际加分小 | ❌ overkill |
| C. Token-based B0 estimator (per-token energy × API server region) | ~1 天 | B0 carbon 也可 disclose | ⭐⭐ 加分明显 |
| D. **Regional sensitivity fig** (45 region 上 Phantom-SoM vs SoM CO2 savings) | ~2 小时 codex | reviewer 喜欢的 framing | ⭐⭐ low cost high value |

**推荐组合**: A + D (Week 2-3 codex 做), C 视情况后期补 (Week 4-5)

**Regional sensitivity fig 设计**:
- x 轴: 45 region 按 carbon intensity sorted (France 85 → Poland 773 → South Africa 928)
- y 轴: per-task CO2 savings (Phantom-SoM vs SoM, 用 B1 measured energy + region intensity)
- 论点: "Phantom-SoM 的 carbon 优势 region-dependent — carbon-heavy 部署 (India/Poland) saving > carbon-light 部署 (France/Norway)"

**Striking findings 已在数据里 (paper 直接可用)**:

1. **B0 cls SoM P95 latency 74s ≈ 2× DOM 38s** — image encoding/inference 拖慢
   → Phantom-SoM 预期 recover DOM latency ⭐ paper Section 1/4 hook
2. **B1 cls SoM energy (0.0020 kWh) < DOM (0.0052 kWh)** — step count 主导
   B1 reddit 反向: SoM 高于 DOM (132s vs 88s) — site-dependent
   → Phantom-SoM 预期 **任何 site 都接近 DOM 数据** (no image penalty)

**Phantom-SoM 三重 Win Hypothesis** (待 phantom 数据验证):
- Cost ≈ DOM (regex filter same AXTree)
- P95 latency ≈ DOM (no image inference stage, B0 省 ~50%)
- Energy ≤ DOM (no image processing, B1 直接验证)

**已有分析 infrastructure** (`p79/experiment/analysis.py:1379-1410`):
- `phase1_comparison_overview.png` — 6-panel multi-metric overview (success / steps / cost / p95_lat / energy / co2e)
- 每 run `make analyze` 自动 regen
- paper figure 不需新建 — 现有 pipeline 已 paper-ready

**Paper 利用 plan**:

| Section | Metric 用法 | 触发时机 |
|---|---|---|
| Section 1 hook | "SoM 2× DOM latency in production (B0 235B), Phantom-SoM 预期 recover" | 当前 prose 可加 (B0 数据已 stable) |
| Section 4 main table | Cost-aware snapshot: 5 mode × {SR, cost, p95 lat} 表 | Section 4 fresh-data update 时 (~Week 2-3) |
| Section 4 fig9 (新) | 2x2 panel (B0 cls/red + B1 cls/red), 每 panel 多指标 grouped bar | Week 2-3, codex prompt |
| Section 5 wasted-cost | DOM search-loop 12 step 高 wasted cost, Phantom-SoM quick decision 低 wasted cost | Section 5 prose 时 |
| Section 7 Sustainability | B1 CO2 数据 (B0 不可测) 论证 representation routing as green-AI lever | Section 7 写时 |
| Supplementary | B0+B1 完整 multi-metric table + per-condition cost/latency distribution | paper appendix |

**Tier 1 (paper main body)**: SR + drop-one + cost + P95 latency
**Tier 2 (Section 7)**: CO2 (B1 only, transparent disclosure)
**Tier 3 (supplementary)**: wasted cost / energy kWh / cost_efficiency_ratio

**Risk: 不要让 cost/latency/carbon overload 主 narrative.**
保持 single narrative ("Phantom-SoM is hidden routing arm + drop-in deployment"),
其他 metric 是 supporting evidence, 不抢主线.

---

## 4.6 Forward-Thinking — Router 难点 / Advisor Align / Visualization (2026-04-27 晚)

### 4.6.1 Router 是真难点 (修正 §4.5.3 timeline)

5 个关键设计决策点 (每个都要做 ablation):

| 维度 | 选项 | 难点 |
|---|---|---|
| Feature | task NLP / browser state / step-1 trigger / capability / audit cat | audit cat 是 leak; small data overfit |
| Target | max SR / SR-per-cost / Pareto / budget-constrained | multi-obj weight 选, single-obj 失论点 |
| Granularity | task-level / step-level / confidence-triggered | step-level 重跑实验 2x cost |
| Cascade | 单 router / B1→B0 escalation / rule+ML hybrid | escalation 实验代价大 |
| Baseline | random / best-single-mode / oracle / rule-based | best-single-mode 是 hardest baseline |

**Realistic timeline 修正** (vs §4.5.3 估 2-3 周):
- Tier 1 (task-level oracle): ~5-7 天
- Tier 2 (first-step trigger / cascade): ~7-10 天
- **Total: ~3-4 周** (paper 真正最值钱的工作量, 不应压缩)

**Minimum viable router** (start, ~3 天 prototype):
```
Feature:  task instruction TF-IDF + binary {has_ref_image, has_finish_string_match}
Target:   max adjusted SR
Model:    Logistic regression (interpretable + small-data friendly)
Train:    cls + red 6 mode, 80/20 split
Baseline: random / best-single-mode / rule-based ("if has_ref_image → SoM else → Phantom-SoM")
```
LR 都打过 best-single-mode → paper 已 honest minimum router 论证, Tier 2 再加 escalation.

### 4.6.2 Advisor / 学长 Align 时机 + 议题

**第一次 align (~Week 3, cls+red+shopping 5-mode B0+B1 完整)**:
- Router scope (Tier 1+2 vs Tier 1+2+3)
- Claude Opus 启动决定 (~$70 budget)
- 单 paper vs 双 paper 决策
- Authorship / 学长导师贡献预期

**第二次 align (~Week 6-7, WA + Claude done)**:
- Paper venue (NeurIPS / ICLR / ACL / MLSys)
- Section 6 generalization 范围 (是否加 Mind2Web)
- 投稿 timing (NeurIPS 2026 deadline ~5 月 / ICLR 2027 ~9 月)

**Align checklist** (每次带):
| 决策 | 现状 | 影响 |
|---|---|---|
| Paper venue | NeurIPS → ACL → MLSys cascade | narrative angle (discovery vs systems) |
| Router scope | Tier 1+2 (timeline ~3-4 周) | paper main contribution 强度 |
| Cross-model | Claude Opus 4.7 only | + GPT-4o/Gemini/Llama 加分但贵 |
| Section 6 范围 | VWA + WA + Claude | + Mind2Web 是 advisor 偏好 |
| Paper 数 | 单 paper (router integrated) | split 增 publication count 但每篇弱 |
| 投稿 timing | TBD | 紧 vs 松 决定 polish 程度 |

**关键**: 不要等所有数据 done 才 align, 提前 align 可避免方向错重做.

### 4.6.3 级联 Router 可视化方案

**单纯 2D cost-SR Pareto 不够 striking**, 推荐 4-figure stack:

| Figure | 作用 | 设计 |
|---|---|---|
| **Fig A: 3-panel multi-metric Pareto** | 主 figure, fig7 升级 | 3 panel: cost-SR + latency-SR + CO2-SR; router 点 above frontier |
| **Fig B: Cumulative SR vs Budget curve** ⭐ | 最 striking, cost-aware 顶刊套路 | x=budget per task, y=cumulative SR; lines: random/best-single/rule/learned/oracle |
| **Fig C: Routing decision Sankey** | Section 6 解释 router 学到什么 | task category → routed mode → outcome flow |
| **Fig D: Per-task savings histogram** | Appendix supplementary | distribution: cost saved by routing per task |

**Fig B 详细设计** (cost-aware paper 顶级 figure 套路, 参考 RouteLLM ICML 2024 / FocusAgent EMNLP 2025):

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

**反对 3D Pareto**: rotate 才看清, paper 印刷不友好, reviewer 抗拒. 用 2D multi-panel 替代.

CO2 维度单独 fig E (regional sensitivity, 见 §4.5.9 Option D), 不塞主 Pareto.

---

## 5. Future paper 2 — DEPRECATED (2026-04-28 晚)

⚠️ **最终决策**: Paper 1 = 毕设论文, **全部 contribution 含 router 都 integrated**.
Paper 2 现在非常边缘 — 仅作为远期 future work 占位, 不进入 active planning.

实际 Paper 1 (毕设) integrated contributions:
- Phantom-SoM hidden 4th routing arm + drop-in deployment
- 3-axis hierarchical mechanism (representation × prompt × image)
- Bidirectional modality + 6-channel image taxonomy
- Site/capability-modulated routing pool
- **Router (Tier 1+2)** 含 phantom routing signal (AUROC ≥ baseline) ⭐
- Cost-aware multi-metric Pareto + green AI sustainability framing

如果毕设答辩后未来真做 Paper 2 (可能性低), 主题候选:
- Phase 3 模块消融 (M1-M4 fallback) — 实验笔记 §19 提到的 future work
- Online learning / mid-trajectory escalation (Tier 3 stretch)
- Cross-model routing meta-policy (跨 model family 一致性)

→ 不进入 active planning, 仅 Phase 3 modules 作 placeholder.

Routing infra 现 paper 1 直接用 (不再是 future):
- 4 baselines + 5 phantom × confidence_summary.json (`overall_usable: True`)
- Behavioral signals AUROC 0.682-0.748 (cls behavioral 主导, red verbalized 主导)
- Verbalized signals AUROC 0.701-0.793 (red Phantom-DOM 0.793 是 5-mode 最高)
- Router infrastructure: `p79/experiment/router.py` RuleBasedRouter scaffold ready
- Phantom modes 直接复用 baseline signal infra (drop-in routing claim 第 4 fold)

---

## 6. Daily routine（routine ops 我可以 monitor）

- ✅ ntfy notifications: condition complete + idle alerts (working)
- ✅ Auto rederive after condition complete (watchdog)
- ✅ Auto figures regen after condition complete (watchdog hook)
- ✅ Auto unified gallery refresh after condition (watchdog)
- 🔔 Persistent error / session lost → high priority ntfy

需要人工 attention 的 trigger:
- ntfy "PERSISTENT ERROR" → 调试 + 决定如何 recover
- ntfy "session lost streak" → 检查 site auth
- ntfy "P79 IDLE" 30+ min → 检查 runner alive

---

## 7. Open issues / 跟踪中的 bug

| Issue | 状态 | Blocker |
|---|---|---|
| B0 phantom_classifieds 04-26 跑过 234 ep on 04-24 reset state, cleared for paper-grade | 待 chain re-run | none |
| Phantom-SoM step traces unavailable for cleared runs | 待 chain re-run done | ~30h |
| WA tasks 5 reddit 含"image" intent keyword (5/106) | 不 codex audit (verified 100% non-visual benchmark) | n/a |
| Magento auth bug (cookie domain split) | ✅ initially fix `7150db8` | quark side base_url 改 IP |
| **Magento base_url 复发 (04-27)** — docker restart 后 base_url 退回 metis, shopping reset FAIL 3 次 | ✅ **fixed + 持久化** (quark side 三层: `magento_baseurl_fix.sh` + `start_vwa_docker.sh` hook + `reset_shopping.sh` 不再 hardcode localhost; DGX side 加 post-reset redirect health check) | DGX-quark reset chain 走 PowerShell, 持久化在 quark linux side, 加 defensive curl check on DGX 验证 redirect target ≠ metis |
| **B1 shopping DOM 466 ep 跑期间 Magento bug 状态不明** (04-23 → 04-24, 在 fix 之前) | 🟡 **决策**: clear+重跑 (paper-grade 协议一致性) | timing: 等 Critical path A done (~Wed) + advisor align #1 后；推荐 rename 旧 run dir `_pre_magento_fix` 保留 reference |
| **Magento base_url 第三次复发 (04-28)** — DGX defensive reset 检测到 metis redirect after docker reset | ✅ **fixed + 持久化 (2026-04-28 13:43)** — PowerShell `C:\vwa\reset_vwa.ps1` 加 `Configure-MagentoBaseUrl` (docker exec config:set + cache:flush, shopping 7770 + shopping_admin 7780 都覆盖). DGX defensive curl check 验证 post-reset HTTP 200 no metis redirect. PowerShell parser check OK. B0 shopping unblocked. | (verified 04-28 13:48 launch B0 dom shopping success) |
| **Magento FPC (Full Page Cache) — homepage 缓存 guest page** (04-28 14:00 发现) — User question 触发: "所有 condition 最一开始都 session lost without login". 实验笔记 §99 已记录但未根因 fix | 🟡 **partial fix needed**: server-side session 完全 valid (sales/order/history accessible, customer endpoint returns Emma Lopez), 但 homepage `/` 是 Magento FPC cached guest page (server-rendered HTML 含 "Sign In" link). Watchdog SESSION detection 看 step_000 DOM 误判 NOT LOGGED IN (false alarm). 实际 task functional logged-in OK. **B0 dom shopping killed 14:10 to address before re-launch**. | **Fix path**: quark side `docker exec vwa-shopping bin/magento cache:disable full_page` + `cache:flush`. 或者改 watchdog SESSION detection 用 server-side check (e.g. /customer/account/ access) 替代 client-side DOM grep |

---

## 优先级总结（如果只能做一件事）— 修正 (2026-04-27 23:30)

**Now (Day 1 night, 全部 codex prompts 完成)**:
- ✅ 不打扰 active chain (cls phantom_som re-run + red phantom_dom)
- ✅ 不发新 codex prompt (今晚 5 个 prompts 全 done)
- 🟢 睡觉 — chain 自动跑, watchdog 自动 rederive + figures regen, ntfy 自动通知

**+6h (~Tue 06:00)**:
- B0 phantom_som cls re-run 完成 (~234/234)
- chain auto trigger B0 phantom_som red re-run
- watchdog auto regen fig8 (Phantom-SoM unique 数字应 ↑)

**+12h (~Tue 12:00)**:
- B0 phantom_dom red 完成 (~210/210)
- watchdog auto regen fig5 + fig8 (Phantom-DOM 完整 panel)

**+24h (~Wed)**:
- B0 phantom_som red 完成 → B0 5-mode 完整
- queue_b1_after_b0 trigger B1 cls chain
- 发 codex prompt 8: Phantom-SoM step trace digest (~400K)
- 发 codex prompt 9: Section 4 prose fresh-data update (~30K)

**+36-48h (~Thu)**:
- B1 cls chain done, B1 red chain start
- 发 codex prompt 10: Section 5 prose draft (~30K)

**+72h (~Fri 凌晨)**:
- B1 red chain done → cls+red 5-mode B0+B1 完整
- **Critical path A 完成** (Section 4-5 prose 全 ready)
- 启动 advisor align meeting #1: router scope + Claude budget + 单/双 paper

**Week 2 (~05-04 起)** Pending advisor align:
- 启 WA × 3 sites × B0+B1 (advisor approve 后)
- 启 Cross-model Claude Opus 4.7 (advisor approve 后)
- Shopping (Myriad GPU 上线后)

**Week 3-5 (paper Section 6 + Router)**:
- WA + Claude done → Section 6 generalization draft
- Router design + train + eval (~3-4 weeks)

**Week 6-7 第二次 advisor align**:
- Paper venue 决定 (MLSys vs NeurIPS workshop vs ACL)
- 投稿 timing decision

**Week 8-12 paper writing + revisions**

**Paper submit target**: ~Week 12 (12 周后, 即 ~2026-07-20).
