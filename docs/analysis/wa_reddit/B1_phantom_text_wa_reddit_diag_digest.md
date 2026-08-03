# /diag digest — B1 × `phantom_text` / P-text × **WA** reddit

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
| **Run** | `B1_phantom_text_wa_reddit_20260729_154551_859907467_2958275_R10542` |
| **Condition** | `phase1_phantom_text_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `phantom_text` / P-text（DOM 风格 prompt + `[SOM_MARKS]` 文本，无图） / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False，无 N/A 排除） |
| **SR** | **16.35%**（17 success / 87 failed） |
| **ruleset_version** | `9-wa-p47p48`（正文成稿于 `8-reddit-p41p46-b1890fix`，v9 数字块见 §2b）|
| **Tier-1 三子集** | failed+hit 75 · **failed-NO-hit 12** · success+hit 1 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step 级命中 | 命中 episode 数 |
|---|---|---|---|
| `P31` | budget 耗尽未完成 | 61 | 61 |
| `P36` | WALK_FAIL_DEGENERATE | 271 | 55 |
| `P5` | 感知缺失循环 | 50 | 36 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 43 | 32 |
| `P14` | URL 自环 | 23 | 20 |
| `P4` | 根节点误操作 | 27 | 15 |
| `P12` | 从不翻页 | 8 | 8 |
| `P33` | 导航至裸图片 URL 幻觉 | 6 | 6 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT（中性标签） | 5 | 5 |
| `P44` | HALLUCINATED_ELEMENT_REF | 9 | 4 |
| `P10` | 跨步数值记忆失败 | 2 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**：task 31（P40）

**failed-NO-hit episode（deterministic 盲区 → Tier-2 全覆盖）**：[67, 68, 581, 601, 610, 635, 641, 651, 729, 732, 733, 734]

**success episode**：[31, 69, 399, 400, 401, 402, 403, 595, 596, 597, 598, 599, 600, 605, 645, 649, 650]


### 2b. v9 数字块（`9-wa-p47p48`，2026-08-03 补）

> 本 digest 正文成稿于 `8-reddit-p41p46-b1890fix`。v9 落码了 `P47`/`P48` 两条新规则
> （由本轮 Tier-2 的 R1/R3 提议而来），scan 已全量重扫，此处补齐数字使 **B0/B1 可并表**。
> 正文的 Tier-2 定性结论不受影响（v9 是纯 additive，未改动任何既有规则的正则或阈值）。

| 规则 | 含义 | step 级 | episode 级 | v8→v9 |
|---|---|---|---|---|
| `P31` | budget耗尽未完成 | 61 | 61 | — |
| `P36` | WALK_FAIL_DEGENERATE | 271 | 55 | — |
| `P5` | 感知缺失循环 | 50 | 36 | — |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 43 | 32 | — |
| `P14` | URL 自环 | 23 | 20 | — |
| `P4` | 根节点误操作 | 27 | 15 | — |
| `P12` | 从不翻页 | 8 | 8 | — |
| `P33` | 导航至裸图片URL幻觉 | 6 | 6 | — |
| `P47` | PREMATURE_FINISH_ON_FORM | 5 | 5 | **0 → 5** |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 5 | 5 | — |
| `P44` | HALLUCINATED_ELEMENT_REF | 9 | 4 | — |
| `P10` | 跨步数值记忆失败 | 2 | 1 | — |
| `P13` | 搜索代替浏览 | 1 | 1 | — |

**三子集** v8 `failed+hit 75 · failed-NO-hit 12 · success+hit 1` → v9 `failed+hit 78 · failed-NO-hit 9 · success+hit 1`

**唯一变化**：`P47` 新增命中。`P48` 在本 cell **0 命中**（它由 B1 数据提议却在 B1 上不 fire —— 实现比提议窄，排查见 [[_benchmark_level_findings]]）。

**从 no-hit 转入 hit 的 episode**（v9 新规则接住的 deterministic 盲区）：[581, 732, 733]

## 3. Tier-2 深挖

**覆盖**：12 个 no-hit **全覆盖** · 1 个 sonnet sub-agent（另有跨 mode 的因果验证组与
`P40` FP 审计组，见 [[_cell_cross_mode_findings]] §3 §6）

**三分类**：scaffold-bug 1 · agent-limit 10 · benchmark-FP 0 · unclear 1

- **task 601 → scaffold-bug**：agent 已正确判断该发到 r/nyc（thought 明确提到下拉里看得见
  `nyc`），但 `select_option` 命中 `no_match_in_css_menus`，选择失败后放弃，仍发到默认的
  AskReddit。→ 这是 **F1 的一个实例**，全量数据见 [[_cell_cross_mode_findings]] §5 F1
  （本 mode 106 次尝试 **106 次全失败**）。
- **task 581 / 732 / 733 / 635 → 打完字直接 finish，从未点提交/保存**：finish 时 `obs_url`
  仍停在 `/create_forum`、`/-/edit`、`/submit/headphones`，thought 却明确幻觉"已提交成功"。
  732/733 尤其干净：目标帖定位**完全正确**（与 reference URL 精确匹配）、正文也已输入，
  只差最后一次点击。→ 规则候选 **R1**（24 failed / **0 success**，是本轮唯一完全干净的候选）。
- **task 67 / 68 → "top 10 post…recommend a single book" 模板 2/2 全败**：只滚动 1-2 屏就
  点进第一个看似符合的帖子即 finish，未系统核对候选集。
- **task 610 → 评论 vs 新帖语义错位**：evaluator 用 `.submission__inner`，agent 在别人的
  同名书帖下发了评论。
- **task 641 → unclear（按硬要求不强分）**：`select_option` 反复失败与"最终落在 `/f/books`
  而非新帖详情页"两条证据互相纠缠，且落盘观测不可用，无法判定 `click(32)` 点的是提交按钮
  还是"返回论坛"链接。

## 4. 与 cell 级发现的关联

本 mode 的失败质量主要落在下列 cell 级机制上，详见 [[_cell_cross_mode_findings]]：

- §5 **F1** `select_option` 在 reddit 99.6% 失败（reddit 站点特有，波及 VWA reddit 计分 cell）
- §5 **F2** `walk_fail` 的元素 100% 在观测里且在视口内 → `P36` 需按元素角色分叉
- §5 **F3** WA reddit task 66 reference 硬编码生产域名
- §5 **F4** phantom 系列 artifact 落盘 ≠ 模型输入
- §6 `P31`（本 mode 覆盖 61/87 失败）**裁定为风险标记而非死因指标**
- §7 新规则候选 R1–R4（R1/R3 建议落码，R2 需收窄，**R4 否决**）

## 5. 可 actionable 项

见 [[_cell_cross_mode_findings]] §8（A1–A9）。本 mode 无独有的 actionable 项。
