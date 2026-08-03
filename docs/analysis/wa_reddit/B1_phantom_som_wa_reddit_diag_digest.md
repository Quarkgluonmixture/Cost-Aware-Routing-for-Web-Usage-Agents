# /diag digest — B1 × `phantom_som` / P-SoM × **WA** reddit

*生成 2026-08-02（Tier-1 重扫 + Tier-2 深挖 + Tier-3 整合）*

> **定位声明**：本 digest 是**单 condition** 的失败归因；其中 per-rule 分布只描述它自己。
> 跨 mode 共有的发现（scaffold-bug、benchmark-FP、规则候选、分母口径）统一收在
> **[[_cell_cross_mode_findings]]**，本文件不复述。
>
> ⚠️ **这是 WebArena，不是 VisualWebArena。** 不要与 `docs/analysis/vwa_reddit/` 下的
> 同名 mode digest 并表 —— 任务集不同（WA 104 scored / VWA 205 collected），
> WA reddit **0/104 任务带 image**。
>
> ℹ️ 本文数字来自 `results/diag_scans/v8_wa/` —— **2026-08-02 修复 `task_configs` 回传后
> 重扫**（B-1919）。此前那批扫描是在空 config 下跑的，44 条规则里 28 条读 config 的被静默
> 禁用；修复经过 + 与注入版的逐 hit 互证见 [[_cell_cross_mode_findings]] §0。

## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B1_phantom_som_wa_reddit_20260730_231304_547960004_3121337_R11421` |
| **Condition** | `phase1_phantom_som_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `phantom_som` / P-SoM（SoM 风格 prompt + `[SOM_MARKS]` 文本，**不给标注图**） / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False，无 N/A 排除） |
| **SR** | **11.54%**（12 success / 92 failed） |
| **ruleset_version** | `9-wa-p47p48`（正文成稿于 `8-reddit-p41p46-b1890fix`，v9 数字块见 §2b）|
| **Tier-1 三子集** | failed+hit 84 · **failed-NO-hit 8** · success+hit 0 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step 级命中 | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 503 | 72 |
| `P31` | budget 耗尽未完成 | 66 | 66 |
| `P5` | 感知缺失循环 | 89 | 56 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 85 | 53 |
| `P14` | URL 自环 | 56 | 37 |
| `P12` | 从不翻页 | 22 | 22 |
| `P4` | 根节点误操作 | 39 | 7 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT（中性标签） | 5 | 5 |
| `P33` | 导航至裸图片 URL 幻觉 | 5 | 5 |
| `P44` | HALLUCINATED_ELEMENT_REF | 1 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**：（无）

**failed-NO-hit episode（deterministic 盲区 → Tier-2 全覆盖）**：[409, 583, 600, 601, 645, 720, 733, 735]

**success episode**：[399, 400, 401, 402, 403, 595, 596, 597, 598, 599, 602, 650]


### 2b. v9 数字块（`9-wa-p47p48`，2026-08-03 补）

> 本 digest 正文成稿于 `8-reddit-p41p46-b1890fix`。v9 落码了 `P47`/`P48` 两条新规则
> （由本轮 Tier-2 的 R1/R3 提议而来），scan 已全量重扫，此处补齐数字使 **B0/B1 可并表**。
> 正文的 Tier-2 定性结论不受影响（v9 是纯 additive，未改动任何既有规则的正则或阈值）。

| 规则 | 含义 | step 级 | episode 级 | v8→v9 |
|---|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 503 | 72 | — |
| `P31` | budget耗尽未完成 | 66 | 66 | — |
| `P5` | 感知缺失循环 | 89 | 56 | — |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 85 | 53 | — |
| `P14` | URL 自环 | 56 | 37 | — |
| `P12` | 从不翻页 | 22 | 22 | — |
| `P4` | 根节点误操作 | 39 | 7 | — |
| `P47` | PREMATURE_FINISH_ON_FORM | 6 | 6 | **0 → 6** |
| `P33` | 导航至裸图片URL幻觉 | 5 | 5 | — |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 5 | 5 | — |
| `P44` | HALLUCINATED_ELEMENT_REF | 1 | 1 | — |
| `P13` | 搜索代替浏览 | 1 | 1 | — |

**三子集** v8 `failed+hit 84 · failed-NO-hit 8 · success+hit 0` → v9 `failed+hit 87 · failed-NO-hit 5 · success+hit 0`

**唯一变化**：`P47` 新增命中。`P48` 在本 cell **0 命中**（它由 B1 数据提议却在 B1 上不 fire —— 实现比提议窄，排查见 [[_benchmark_level_findings]]）。

**从 no-hit 转入 hit 的 episode**（v9 新规则接住的 deterministic 盲区）：[600, 645, 733]

## 3. Tier-2 深挖

**覆盖**：8 个 no-hit **全覆盖** · 1 个 sonnet sub-agent（另有跨 mode 的因果验证组与
`P40` FP 审计组，见 [[_cell_cross_mode_findings]] §3 §6）

**三分类**：scaffold-bug 1 · agent-limit 7 · benchmark-FP 0 · unclear 0

> 本 mode 是 locator 失败最严重的（`P36` 命中 72/92 失败 episode），但这 8 个 no-hit
> 恰恰一条规则都没中 —— 它们的失败全部发生在"动作都成功执行"的层面。

- **task 600 / 645 / 733 → 输完文本直接 finish，从未点提交**：`type` 之后紧跟 `finish`，
  中间无任何 `click`，`obs_url` 仍停在 `/submit/<forum>`。733 尤其说明问题：导航路径
  **完全正确**（`/f/television/135201/-/...` 与 eval 前缀吻合），只差保存那一下。
  → 规则候选 **R1**。
- **task 583 → 表单被顶栏导航链接带走**：填完 create_forum 三个字段后点了
  `element_id=10`（bbox `[852,0,103,52]`，与 task 600/645/601 中同一元素同一 bbox），
  跳到 `/submit`，表单数据丢弃，finish 时幻觉"论坛已创建"。
- **task 720 → 量词范围坍缩**：intent 是 "Like **all** submissions by CameronKelsey"，
  agent 3 步内（type→click→finish）认为"页面上已显示该用户的一个帖子，任务完成"，
  全程无任何投票动作。
- **task 735 → 编辑对象完全错**：把 "Edit biography"（用户简介页）当成帖子编辑入口，
  把目标文本输进了简介框。
- **task 409 → sub-agent 判 scaffold-bug，我标注为「未复核」**：回复成功落在
  `/comment/1`，而 eval 引用 `/comment/1235250`，agent 据此推断本地 docker 种子数据的评论
  ID 空间与评测引用不对齐。⚠️ 这是**间接推理**（未读取当时 DOM/DB），且只有 1 个样本。
  按本项目纪律，**不据此落码**，列为待查。

## 4. 与 cell 级发现的关联

本 mode 的失败质量主要落在下列 cell 级机制上，详见 [[_cell_cross_mode_findings]]：

- §5 **F1** `select_option` 在 reddit 99.6% 失败（reddit 站点特有，波及 VWA reddit 计分 cell）
- §5 **F2** `walk_fail` 的元素 100% 在观测里且在视口内 → `P36` 需按元素角色分叉
- §5 **F3** WA reddit task 66 reference 硬编码生产域名
- §5 **F4** phantom 系列 artifact 落盘 ≠ 模型输入
- §6 `P31`（本 mode 覆盖 66/92 失败）**裁定为风险标记而非死因指标**
- §7 新规则候选 R1–R4（R1/R3 建议落码，R2 需收窄，**R4 否决**）

## 5. 可 actionable 项

见 [[_cell_cross_mode_findings]] §8（A1–A9）。本 mode 无独有的 actionable 项。
