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

## 证据清单 (全部已在手, 2026-07-22)

| 条 | 数字 |
|---|---|
| 路由天花板存在性 | oracle 43.3% vs 最强单模 27.2% (cls·B0); 97 可解任务中 36 个在最强单模之外 |
| 标签供给 | 每 cell **16-97** 个可训练标签; B2_cls 可解率 7.1% / B1_red 12.7% |
| 可训练性 | `N_MIN_CLASS_TRAIN=10` → **3/5 cell 零可训练折**; Stage 3 终判 **1/5 cells fully trained** |
| 连续标签路已堵 | VWA `score` 纯二值 {0,1} (7963 episodes: 7278/685), **无部分分** |
| tier 化不救 | 6 路→2 路类数减少, 但绝对标签量仍卡死 |
| 池化解供给 | 249 个标签, 六类全过 min-class 过滤器 |
| 池化的代价 | 特征是任务的函数 → 同 X 矛盾 y: **cls 57.4% / red 45.5% 矛盾** |
| Bayes 上限 | 纯任务特征分类器: **cls 79.2% / red 87.7%** |
| 唯一自洽组合 | **池化 + cost-tier 二分类** (tier 跨 cell 一致性 red **95.5%** / cls 68.5%) |
| tie-break 任意性 | **约 1/4 标签由 `MODES` 硬编码顺序决定而非数据** (26%/29%/25%/18%/15%) |

三条独立路径(假设类 15 格 sweep 已有 / 监督侧三种定义 / 池化换可识别性)都指向同一瓶颈 → 论证闭合。

## 待办

- [ ] 起稿 (8 页, ACL 2026 style, 双盲)
- [ ] **LOCO 池化+tier 实训** —— 唯一缺的一块: "它到底有没有用"。落在 prereg L447 已注册的
      LOCO cross-cell appendix sensitivity 槽位内
- [ ] 全部 exploratory 产物按 `post_hoc_exploratory=True / h10_eligible=False` stamp 归档
- [ ] /stress + Mode B/C chain

## 纪律

所有标签重定义分析均为 **post-hoc exploratory, 非 H10-eligible, 不入 gating family**
(沿用 `router_model_sweep_summary.csv` 既有 stamp 约定)。**绝不**用于事后挽救 H10 判定。
