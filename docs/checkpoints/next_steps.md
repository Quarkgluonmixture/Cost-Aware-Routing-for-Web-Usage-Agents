# P79 Next Steps — 行动汇总

> 中央 plan 文档。汇总所有 pending experiments / analyses / paper-writing tasks。
> 实验笔记 (`docs/checkpoints/实验笔记.md`) 顶部"行动规划"区是 quick reference，
> 这个文档是详细 owner / blocker / ETA 跟踪。
> 最后更新：2026-04-27 ~20:00 (B0 phantom_dom_classifieds fresh 234/234 done)

---

## 当前 status snapshot

```
Active processes:
  ├─ runner B0_phantom_som_classifieds  (re-run, fresh 04-27 reset, 4 ep, ~3-6h ETA)
  ├─ runner B0_phantom_dom_reddit       (running, 65 ep, ~2-3h ETA)
  ├─ watchdog × 2
  ├─ chain orchestrator B0 cls (waiting som re-run done)
  ├─ chain orchestrator B0 red (waiting dom done → reset → som re-run)
  └─ queue_b1_after_b0 sequencer (waiting all B0 chains)

Active data (paper-grade clean):
  ├─ B0 cls + red 3-mode baseline (rederived 04-27, 数字 stable per §103)
  ├─ B1 cls + red 3-mode baseline (rederived 04-27)
  ├─ B0 phantom_dom cls (FRESH 04-27 reset, 234/234 ✅ DONE)
  └─ B0 phantom_dom red (跑中 65/210)

Cleared / pending re-run:
  ├─ B0 phantom_som cls (re-running 现在 fresh, 4 ep)
  ├─ B0 phantom_som red (待 chain re-run after dom)
  ├─ B1 phantom_reddit (待 queue_b1_after_b0 trigger)
  ├─ B1 phantom_classifieds (同上)
  └─ All shopping data (cleared, 等 Myriad)
```

### 04-27 fresh data: B0 phantom_dom_classifieds 234/234 完整 ⭐

| Mode | N | raw | adj | FP gap |
|---|---:|---:|---:|---:|
| DOM (baseline) | 234 | 14.96% | 14.10% | 0.85pp |
| **SoM** (baseline) | 234 | **23.08%** | **21.37%** | 1.71pp |
| Vision (baseline) | 234 | 15.81% | 13.68% | 2.14pp |
| **Phantom-DOM** (FRESH) | 234 | **16.67%** | **14.53%** | 2.14pp |

Note: Phantom-DOM **adj 14.53% ≈ DOM 14.10%** (基本持平), raw **16.67% > DOM 14.96%**.
Section 4 draft + figures 用的是 §103 之前 stale 数据 (Phantom-DOM 12.50%)，
**需要 update with fresh numbers** (待 phantom_som re-run done 后一起 update).

ETA paper-grade clean cls + red full 4-mode matrix: **~48h from now** (revised)

⚠️ Reddit 比 cls 慢 2.4×/step (Postmill render 5.4× slower + AXTree 1.5× larger).
B1 reddit chain (GPU sequential, 4B + slow site) = **~42h alone**.
B1 cls chain ~16h.

Wall-time breakdown:
- B0 phantom_som cls re-run: ~5h (4.6h × 234 task)
- B0 phantom_som red re-run: ~14h
- B0 phantom_dom red continue: ~9h remaining
- B1 cls chain (sequential GPU som + dom): ~16h
- B1 red chain (sequential GPU som + dom): ~42h ⚠️

---

## 0. 完备性审计（2026-04-27 22:00）

### 0.1 数据状态矩阵

| Cell | DOM | SoM | Vision | Phantom-SoM | Phantom-DOM |
|---|---|---|---|---|---|
| **B0 cls** | ✅ stable | ✅ stable | ✅ stable | 🔄 重跑 (4 ep) | ✅ FRESH 234/234 |
| **B0 red** | ✅ stable | ✅ stable | ✅ stable | ⏳ 待 chain | 🟡 partial 71/210 |
| **B1 cls** | ✅ stable | ✅ stable | ✅ stable | ⏳ 待 queue_b1 | ⏳ 待 queue_b1 |
| **B1 red** | ✅ stable | ✅ stable | ✅ stable | ⏳ 待 queue_b1 | ⏳ 待 queue_b1 |
| Shopping (B0/B1) | ❌ cleared 待 Myriad | ❌ | ❌ | ❌ | ❌ |
| WA (3 sites × B0/B1) | ❌ 未跑 | ❌ | ❌ | ❌ | ❌ |

**paper-ready**: baseline 12/20 (60%) + phantom 1/8 (12.5%)

### 0.2 分析层完备性

| 分析 | 状态 | 缺口 |
|---|---|---|
| Codex audit (cls/red) | ✅ A/B/C/D 已分类 | shopping, WA 未做 |
| disagreement_clusters.md (B0+B1 cls+red) | ✅ 9 categories | aggregate, 待 cross-site 拆 |
| Cross-site pattern consolidation | 🔄 codex 跑中 | — |
| B1 capability profile 单独文档 | ❌ 未做 | Section 6 cross-model 准备 |
| Phantom-SoM step traces digest | ❌ 待 phantom-SoM 重跑 done | Section 5 prose 必备 |

### 0.3 核心 paper claim evidence 完备性

| Claim | 当前 evidence | 完备度 | 主要 blocker |
|---|---|---|---|
| **C1**: Phantom-SoM 是 hidden 4th routing arm | fig1/fig2 (stale) + §103 §95 N=48 | 50% ⚠️ | **B0 Phantom-SoM fresh 重跑 — paper 第一论点的根** |
| **C2**: Two-knob (prompt × obs format) | fig3 reddit + fig5 + B0 cls Phantom-DOM ≈ DOM | 60% | red Phantom-DOM 完整 + cls 4 metrics + B0 cls Phantom-SoM fresh |
| **C3**: Capability × representation interaction | fig6 +43.7pp aggregate | 65% | per-site 拆分 (cross-site consolidation 跑中) + B1 phantom |
| **C4**: Cost-SR routing value | fig7 B0+B1 baseline + B0 cls Phantom-DOM | 75% | B1 Phantom 4 cell + red Phantom-DOM |
| **C5**: Generalization (cross-site/benchmark/model) | cls+red 部分 | 30% | shopping + WA + Claude — Section 6 后期 |

### 0.4 7 张 Figure 当前讲故事能力

| Figure | 能看到的 ⭐ paper-relevant finding | 看不到 / 缺口 |
|---|---|---|
| **fig1** 4-mode venn (2x2) | B0 red Phantom-SoM 10.95% > DOM 9.52%, ≈ SoM 10.48% (反直觉) | B1 phantom panel 全 N/A; B0 phantom-SoM stale; 4 圆视觉拥挤 unique 数字小 |
| **fig2** drop-one oracle (2x2) | Phantom-SoM cls **1.71pp** / red **2.38pp** drop-one (paper 标题数字 ⭐); SoM cls 7.69pp 最高 | B1 panel N/A |
| **fig3** strategy gradient (red only) | DOM search-loop 22.7%→SoM 12%→Phantom-SoM 10.8% **单调梯度** ⭐⭐; Phantom-DOM = Phantom-SoM (obs 决定 exploration, prompt 不决定) | cls 待加 (cherry-pick 风险) |
| **fig4** two-knob diagram | schematic, paper reader 一图理解 2x2 | 无数据 |
| **fig5** category × mode heatmap | Cat A 全 mode 高; Cat B 仅 SoM/Vision 高 (视觉必需); Cat C SoM 最优; Cat D 全低 (FP) ⭐ | Phantom-SoM stale; reddit Phantom-DOM partial |
| **fig6** capability contrast | B0 SoM early-finish 53.3% → B1 SoM visual-hijack 70.4%, **+43.7pp flip** ⭐⭐ | aggregate cls+red, per-site 拆分待 |
| **fig7** cost-SR Pareto (cls+red) | B1 ~0 cost 占 frontier; B0 SoM cls 21.37%/$0.04; Phantom-DOM cls 14.5%/$0.012 ≈ DOM (不是 routing 值) | B1 Phantom 4 点 dotted placeholder |

### 0.5 现在可下结论 vs 待数据 finding

**已稳定（不需等数据，可写进 paper 主文）**:
1. DOM-only mode 在 visual-required task 上系统性失败 (fig5 cat B)
2. B0→B1 同 mode 失败 pattern 显著 flip (+43.7pp aggregate)
3. DOM 在 reddit search-loop 22.7% (vs SoM 12%) — 单 site，cls 扩 fig3 后强化
4. **Phantom-DOM cls adj 14.53% ≈ DOM 14.10% (n=234, FRESH)** — prompt-as-commitment-knob 直接 evidence ⭐
5. Cost gap B0 vs B1 ~30× (Pareto frontier 形状清楚)

**待数据**:
- Phantom-SoM 真实 unique task pool (stale → fresh shift 风险)
- cross-site Phantom-DOM 一致性 (red partial)
- B1 phantom 在 cost-SR 上的位置 (全 N/A)
- shopping / WA / cross-model 的 generalization

### 0.6 Paper Section 论证强度 + critical path

| Section | evidence 质量 | 可立刻动笔 | Hard blocker |
|---|---|---|---|
| 1 Intro | ✅ 已写 | — | — |
| 2 Background | ✅ 已写 | — | — |
| 3 Definition | ✅ 已写 | — | — |
| 4 Empirical | 🟡 60% | DOM/SoM/Vision baseline + B0 cls Phantom-DOM 单点 | **B0 phantom-SoM 重跑 + B1 phantom 全跑** (~48h) |
| 5 Mechanism | 🟡 50% | fig3 reddit + fig5 cls/red | **fig3 cls 扩 + Phantom-SoM step traces** |
| 6 Generalization | ❌ 30% | cls vs red 一致性 | **shopping + WA + cross-model** (~周-月) |
| 7 Discussion | ❌ — | — | 全部 data done |

**Critical path A** (Section 4-5): B0 phantom-SoM 重跑 + B0 phantom-DOM red 完整 + cross-site consolidation = ~24-48h
**Critical path B** (Section 6): shopping + WA + cross-model = ~周-月，独立 critical path

### 0.7 下一步 codex 任务（按论证收益/token 比）

| # | Task | Tokens | 论证收益 | Blocker |
|---|---|---|---|---|
| 1 | **fig3 cls 扩展** (top reddit + bottom cls = 2x4 grid) | ~80K | ⭐⭐ Section 5 强化 cross-site mechanism | 无 |
| 2 | **B1 capability profile 单独文档** | ~300K | ⭐ Section 6 cross-model 准备 | 无 |
| 3 | **fig8 stacked bar by overlap depth** (fig1 替代/补充) | ~120K | ⭐ paper figure quality 提升 | 无 |
| 4 | **Cross-site pattern consolidation** | ~200K | ⭐⭐ Section 4/6 evidence | 已发包跑中 |
| 5 | Phantom-SoM step trace digest | ~400K | ⭐⭐ Section 5 prose 必备 | 待 phantom-SoM 重跑 done (~24h) |
| 6 | Section 4 + figures 全量 update with fresh data | ~30K | ⭐ data accuracy | 待 B0 phantom-SoM/DOM 全完成 (~48h) |
| 7 | Section 5 prose | ~30K | ⭐ paper end-stage | 待 #5 + #6 done |
| 8 | Codex audit shopping VWA tasks (466) | ~500K | ⭐ fig5 扩 shopping panel | 待 shopping 数据 (Myriad) |

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

### 2.1 已完成

| Task | Output | Commit |
|---|---|---|
| Section 4 draft (1725 words) + 4 figure scripts | docs/analysis/paper_drafts/section4_empirical_findings.md + scripts/analysis/figures/ | bfe0154 |
| Disagreement task cluster analysis (B0 cls+red) | results/phantom_paper/analyses/disagreement_clusters.md | ded0ef6 |
| **B1 disagreement capability contrast** (B0 vs B1 SoM hijack flip +43.7pp ⭐) | append to disagreement_clusters.md (90 fail pairs, 9 categories) | **c4b52c3** |
| **Section 2 Background + paper.bib** (1514 words, 16 bibtex entries) | docs/analysis/paper_drafts/section2_background.md + results/phantom_paper/paper.bib | **206cd93** |
| **Section 1 Introduction** (786 words) | docs/analysis/paper_drafts/section1_intro.md | **62c1380** |
| **Section 3 Phantom-SoM Definition + Ablation Setup** (863 words) | docs/analysis/paper_drafts/section3_definition.md | **13b9608** |
| **figs 5/6/7 + fig1/2 update** (2x2 panels with B0/B1, Codex audit heatmap, capability contrast, fig7 placeholder) | results/phantom_paper/figures/ | **5736fb4** |

### 2.2 已发包跑中

(空 — 等待 codex 接下来任务派发)

### 2.3 Figures 当前数据完整度审计 (2026-04-27)

| Figure | 数据完整度 | 备注 |
|---|---|---|
| fig1 4-mode venn (2x2) | ~63% | B0 Phantom-SoM `.bak_pre_rederive` (stale) + B1 Phantom-SoM 整列 N/A |
| fig2 drop-one oracle (2x2) | ~63% | 同上 |
| fig3 strategy gradient | ✅ 100% | §103 verified N=48 anchors |
| fig4 two-knob diagram | ✅ 100% | schematic, 无数据 |
| fig5 category × mode heatmap | ~70% | B0 Phantom-SoM stale + reddit Phantom-DOM partial |
| fig6 capability contrast | ✅ 100% | parses disagreement_clusters.md aggregate |
| fig7 cost-SR frontier | 0% | placeholder; **数据 ready，待 codex 填实** |

→ 数据完整后 `make figures` 一键 regen，scaffolding 已备。

### 2.4 现在能发包（不依赖 phantom 重跑，数据 ready）

| Task | Estimated tokens | 输出位置 | Paper value |
|---|---|---|---|
| **fig7 实数据填充** (cost vs adj-SR Pareto, B0/B1 baseline 6 cells + B0 Phantom-DOM cls) | ~80K | scripts/analysis/figures/fig7_cost_sr_frontier.py + .png | ⭐⭐ paper 核心 figure |
| **Cross-site cluster consolidation** (cls vs red site-agnostic vs site-specific patterns) | ~200K | results/phantom_paper/analyses/cross_site_pattern_consolidation.md | ⭐⭐ Section 4/6 evidence |
| **B1 capability profile 单独文档** (B1 cls/red 三模式 trajectory + capability vs B0) | ~300K | docs/analysis/B1_capability_profile.md | ⭐ Section 6 cross-model 准备 |
| **Codex audit shopping VWA tasks** (466 tasks, A/B/C/D 分类) | ~500K | docs/analysis/cross_sites/codex_audit_shopping.json | ⭐ fig5 扩 shopping panel |

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

## 4. Paper 1 (Phantom-only) Section 状态

| Section | Status | Source / Owner |
|---|---|---|
| 1. Intro | 未写 | codex prep prompt ready |
| 2. Background + Related Work | 未写 (deep research doc ready) | codex |
| 3. Phantom-SoM Definition + Ablation Setup | 未写 | codex |
| **4. Empirical Findings** | **✅ Codex draft 1725w (cls+red B0)** | codex (bfe0154) |
| 5. Mechanism (Two-Knob) | 部分 evidence ready (disagreement cluster) | codex 数据 ready, prose 待写 |
| 6. Cross-Site/Model Generalization | 等数据 (WA + Claude) | TBD |
| 7. Discussion | 未写 | codex (paper end-stage) |

Section 4-5 数据 + figures 是 paper 主体核心，~~已 ready~~ 待 chain done 后 phantom traces 补全后 100% 完整。

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

### 4.5.2 顶刊概率（final scope 完成后）

| 投稿目标 | 接收概率 | 投稿优先级 |
|---|---:|---|
| **NeurIPS / ICLR main** | **40-55%** | **Tier 1 stretch** |
| **ICML** | 35-50% | Tier 1 stretch |
| **ACL / EMNLP main** | 45-60% | Tier 1 |
| **MLSys** | **60-70%** | **Tier 1 safe** ⭐ drop-in framing 完美 fit |
| WWW / WSDM | 70-80% | Tier 2 |
| NeurIPS D&B | 60-70% | Tier 2 |
| **TMLR (journal)** | **65-80%** | **保底** ⭐ |

→ Final scope 完成后, paper 顶刊出版几乎 lock (cascade NeurIPS → ACL → MLSys → TMLR)。NeurIPS/ICLR 顶级命中率 ~50%。

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

**🟢 主动跑 (priority order)**:
1. ✅ 维持现有 paper-grade chain — 不打断 (B0 phantom_dom red 跑中, chain auto-trigger phantom_som re-run, queue_b1_after_b0 sequencer)
2. ✅ 22:06 codex 重置后发 4 prompts (按收益排序):
   - Prompt A (~80K): fig3 cls 扩展
   - Prompt C (~30K): fig7 deployment annotation
   - Prompt B (~120K): fig8 stacked bar overlap
   - Prompt D (~300K): B1 capability profile
3. ✅ Cross-site cluster consolidation (codex 已发跑中)
4. ✅ 每天 monitor watchdog ntfy + chain progress
5. ✅ 每天 quick health check (`make schedule-list` + check chain logs)

**🟡 准备 (不动手)**:
- Claude Opus API key + cost budget align (启动决定 ~Week 4)
- Router design literature search (启动 ~Week 5-6)

**🔴 不要做**:
- ❌ shopping (B0+B1, 等 Myriad GPU)
- ❌ WA (任何 site, 等 cls+red 完整)
- ❌ Claude API run (等 cls+red 完整)
- ❌ Router design 实际写代码 (等 baseline + phantom 完整再启动)
- ❌ Section 4-5 prose (等 fresh phantom data)
- ❌ Mind2Web (out of scope)

**Week 0 末预期状态**:
- B0 phantom 5-mode cls + red 完整 ✅
- B1 phantom 5-mode cls 完整 ✅ (red 还在跑)
- 4 paper figures (1/2/3/5/7/8) 数据完整
- Cross-site pattern consolidation done
- B1 capability profile draft done
- Section 1/2/3 paper drafts done (已 done 现在)

---

## 5. Future paper 2 — REVISED (2026-04-27 晚)

⚠️ **决策更新**: 原计划 "Phantom-only paper + Routing follow-up paper". 
**新决策 (final scope)**: Router (Tier 1+2) integrated into Paper 1 主 contribution, 
**not 独立 follow-up paper**.

如果未来真有 Paper 2, 主题应转向:
- Phase 3 模块消融 (M1-M4 fallback 机制)
- 或 routing online learning / mid-trajectory escalation (Tier 3 stretch)
- 或 cross-model routing meta-policy (跨 model family 的 routing 一致性)

记录 routing-relevant 已积累的 infra (供 paper 1 router section + future use):

- Routing signals: 4 baselines × confidence_summary.json (`overall_usable: True`)
  - Behavioral signals: action_diversity / max_repeat_streak / url_revisit (AUROC 0.62-0.77)
  - Verbalized signals: ep_mean_verbalized (0.69-0.77)
  - Cross-mode AUROC computed
- Router infrastructure: `p79/experiment/router.py` RuleBasedRouter scaffold ready
- Pending: routing rule design, threshold tuning, Phase 2 condition variants

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
| Magento auth bug (cookie domain split) | ✅ 已 fix (commit 7150db8) | quark side base_url 改 IP |

---

## 优先级总结（如果只能做一件事）

**Now (cls+red chain 跑中, 不打扰)**:
- 发包给 codex: B1 disagreement analysis (Tier 1A-B1)
- 不动 active runs
- 学长讨论 cross-model + paper split + WA 启动时机

**+6h**:
- B0 phantom_dom cls + red 完成 → chain auto-trigger som re-run
- 给 codex: Phantom traces 补 (after som re-run done)

**+12h**:
- B0 phantom_som re-run done → chain done → queue_b1_after_b0 trigger
- 给 codex: Section 1-3 paper drafts

**+24h**:
- B1 cls chain done
- 给 codex: cross-baseline disagreement compare

**+36h**:
- B1 red chain done → cls+red full matrix paper-grade clean
- 启 WA × 3 sites × B0+B1 chains
- 启 Cross-model Claude Opus 4.7 (parallel)

**+3 days**:
- WA done → Section 6 generalization draft
- Cross-model done → Section 6 model dimension

**+5 days (Myriad 上线)**:
- B0 + B1 shopping baseline + phantom 全部跑
- Final 24-cell matrix complete

**Paper writing**: ~1 week after data complete → submit
