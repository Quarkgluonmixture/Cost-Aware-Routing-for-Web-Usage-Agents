---
type: task
status: active
priority: P0
horizon: now
order: 1
blocker: ""
eta: "2026-08-05 (同 Paper A deadline); 证据 k-无关故可先写完"
detail: docs/checkpoints/实验笔记.md §383.4
created: 2026-07-22
updated: 2026-07-22
---

# Paper B — 路由阴性结果 + 标签供给诊断 (REALM @ EMNLP 2026, **archival**)

**主张**: 路由天花板很高**但学不到** —— 瓶颈是标签**产生率**, 不是假设类, 也不是标签**定义方式**。

**归档选择 = archival** (决策 2026-07-22, 笔记 §383.1): 本身不太可能单独撑起主会论文,
拿 ACL Anthology 记录更划算。

**⭐ 不阻塞于 k=6**: 核心证据是**结构性事实**, B2_reddit 进来只多一行, 论点一字不改 →
**现在就能整篇写完**。(反之 Paper A 的 hero 数字全随 k=6 移动。)

## 证据清单 (**2026-07-27 刷新为 k=6 实测**; 旧版是 07-22 的 k=5 数字, 已 superseded)

> ⚠️ 全部有可重跑来源。**不要再引 07-22 那批** —— 那三个 scratch 脚本
> (`label_supply_sweep` / `label_trainability` / `pooled_label_conflict`) 已丢失,
> 数字也随 B2_reddit 入池 + reddit 转 203 集而移动。

**产物**: `router_label_supply_diagnosis.{md,json}` (新, 本次重建) ·
`router_triage_learnability.{md,json}` · `router_objective_ordering.{md,json}` ·
`sr_per_mode.json` · `phase1_full_prereg_decision.json`

### A. 路由天花板存在 (oracle 远高于任何单模)

| cell | oracle triage SR | 最强单模 SR | 可解率 |
|---|---|---|---|
| cls·B0 | 27.23% | 27.23% | 43.3% |
| red·B0 | 14.78% | 14.78% | 26.1% |
| cls·B1 | 14.29% | 14.29% | 24.6% |

(oracle 与最强单模 SR 相同、cost 更低 —— 天花板体现在**成本**维度: cls·B0
0.06312 vs 0.07236, red·B0 0.09998 vs 0.11045。)

### B. which-mode 半: 败在标签**供给**

| cell | 计分集 | 可训练标签 | 可解率 | min-class 过滤后存活类数 | 可训练? |
|---|---|---|---|---|---|
| B0_cls | 224 | **97** | 43.3% | 3 (dom/pprompt/som) | yes |
| B0_red | 203 | **53** | 26.1% | 1 (dom) | **no** |
| B1_cls | 224 | **55** | 24.6% | 2 (dom/som) | yes |
| B1_red | 203 | **24** | 11.8% | 0 | **no** |
| B2_cls | 224 | **16** | 7.1% | 0 | **no** |
| B2_red | 203 | **15** | 7.4% | 0 | **no** |

**4/6 cell 无可训练分类器**; Stage3 终判 **1/6 cells fully trained** (仅 B0_cls;
B1_cls folds_ok=[0,1,2,3])。pooled **260** (原 249)。

### C. 换标签定义救不了 (三条路径)

- **连续标签**: VWA `score` 纯二值 {0,1} (7963 episodes: 7278/685), 无部分分 → 路堵死
- **池化解供给、破可识别性**: 特征全是任务的函数(14 intent 正则 + difficulty +
  has_ref_image, 无模型信息) → 同 X 矛盾 y。cls **57.41%** / red **56.0%** 矛盾率
  (07-22 red 记 45.5%, 是 2-cell 时代的数)
- **Bayes 上限**: which-mode cls **79.17%** / red **83.70%**
- **cost-tier 重切是唯一有收益的重标注**: 上限抬到 cls **89.88%** / red **96.74%**,
  且**不需要制造任何新 solve 事件**; tier 一致性 cls 68.52% / red 88.0%

### D. ⭐ triage 半: 标签够、AUROC 够, 仍然失败 (§387.16 / §392)

- 标签**充足** (203/224 全有), AUROC **0.651-0.717** 在 5/6 cell, 4/6 超最强单协变量
- **真嵌套 CV 下** (B-1903 修正后): **0/6 cell 能 Pareto 胜过平凡 always-cheapest**
- 唯一 Holm 通过的 red·B2 (p=0.0050) **AUROC 只有 0.483** → saving 来自**尾部富集**
  而非全局判别: 把 192/203=95% 扔给便宜模式, 与免费固定策略只差 5 个百分点的分配,
  那 11 个留守任务含 4 个成功 (§394)
- **`best_mode` 跨折不稳定**: red·B0 五折选 DOM/DOM/SoM/SoM/DOM → 用全量结局挑一个
  best mode 的管线, 报告的 mode 选择连自己的重采样都复现不出来

### E. 监督本身的任意性 (推翻旧提法)

- 旧说「~1/4 标签由 MODES 硬编码顺序 tie-break 决定」→ **不成立**: `true_tie` 在
  6 个 cell **全是 0** (cost 是连续浮点, 恰好相等不发生), tie-break 分支从未触发
- **真实缺陷更严重**: **12.5-54.64%** 的标签上 MODES 顺序返回了一个**严格更贵**的
  成功 mode, 而其 docstring 声称 "ascending prior cost" 并被当作 cheapest-successful
  的代理 → 那些标签连「最便宜的成功 mode」都不是

### 论点 (四条独立路径闭合)

瓶颈是标签的**产生率**, 不是假设类、不是标签**定义方式**、也不是 triage 侧的
**标签量或可预测性**。标签只在任务被解开时诞生; 成功率 2-27% 时无法凭重新切分制造事件。
而 triage 侧即便标签与 AUROC 都够, 仍然赢不过一条白送的固定策略。
