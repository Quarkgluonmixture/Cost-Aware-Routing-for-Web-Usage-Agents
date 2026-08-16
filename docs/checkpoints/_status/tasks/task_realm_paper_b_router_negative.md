---
type: task
status: done
priority: P0
horizon: now
order: 1
blocker: ""
eta: "✅ **已提交 2026-08-06** — OpenReview **Submission #192**, REALM @ EMNLP 2026, `Non-archival long`, Cross Submission To: `Plan to submit to ACL ARR 2026 August` (主会投稿权保住), 作者四人。最终稿 `d737f92`: 正文 8 页 / 35 页 A4 / 0 error / 0 undefined / 首页 Anonymous。**notif 2026-08-21** (user 08-16 更正, 旧记 09-07 作废) **/ camera-ready 09-14**。留给 camera-ready 的 4 条终审项见 next_steps §0"
detail: 笔记 §398.8(合并拍板) + §406(噪声审计) + **§407(三分转向 + 非归档轨)**. 新四步 = ①上限一半是假币 / ②默认答案最贵而没挣到 / ③该买哪条随模态翻号 / ④压不到 per-request(+51.1% 训练行重跑就翻). 新章节 §1问题陈述 / §2setup+2x2 / §3上限与假币 / §4该买什么 / §5为什么压不到per-request / §6discussion
created: 2026-07-22
updated: 2026-08-06
---

# REALM 稿 (合并 A+B) — 表征路由的上限真实存在, 但既不稳定也不可达

> ## ⚠️ 本卡自 2026-07-28 起 = **唯一的 REALM 稿**, 不再是「Paper B」
>
> **user 拍板 2026-07-28 (笔记 §398.8): 两篇 → 一篇。**
> 原 `task_realm_paper_a_phenomenon` 已 superseded, 其内容降级为本稿的 ①② 两步。
> **不是因为两篇都弱 —— 原 Paper B 强、原 Paper A 弱**: B 是完整自洽的负结果,
> A 则 H1 FAIL + H3 双轴低于噪声地板。
>
> ### 焊接点 (四步链, §398.8)
>
> | 步 | 内容 | 来源 | 依赖对账? |
> |---|---|---|---|
> | ① | ceiling 高 (+3.4~16.1pp, 省 13.7-35.3%) | 原 Paper A | **是** |
> | ② | 有结构基础 (H3 双轴独立) | 原 Paper A | **是** |
> | ③ | **但结构小于同模式重跑地板** | §398.2 噪声地板 (焊接的关键新增) | 否 |
> | ④ | 且学不到 (0/6 Pareto) | 本卡 A-E 证据 | 否 |
>
> 第 ③ 步是焊接的枢纽 —— 它**同时**关掉 A 的正面结果、补上 B「为什么不是估计器问题」
> 的另一半。两篇原本都没有这一步。
>
> **周末可动手的**: ③④ 两半 (本卡 A-E + §398.2 地板) 是 k-无关的结构性事实,
> 不依赖 08-03 对账。对账真正要定的是 ①② 那半怎么讲。

**主张 (合并后)**: 表征路由的上限真实存在, 但既不稳定也不可达 —— ceiling 高且有结构基础,
然而该结构小于同模式重跑的噪声地板, 且瓶颈是标签**产生率**而非假设类或标签定义方式。

**归档 = archival** (双盲 ACL, 8 页正文 + refs/appendix 不限; camera-ready +1)。

**从原 Paper A 继承的待办**
- [ ] AAAI→ACL 格式转换 (`aaai27/latex/` skeleton + convert.sh 改目标模板; 不等数据)
- [ ] ①② 两步的 prose (等 08-03 对账定骨架)
- [ ] k=6 后删 Protocol Note 06 两轨制披露 + "5 of 6" 免责段
- [ ] B-1284 cross-family modifier 解除后开放跨族复制主张
- [ ] /stress + Mode B/C chain (2026-07-31 已跑一轮, 见笔记 §405.12)

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
