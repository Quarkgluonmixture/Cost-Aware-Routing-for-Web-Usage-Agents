# /diag digest — B1 × `vision` × **WA** reddit

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
| **Run** | `B1_vision_wa_reddit_20260729_002545_844006252_2860757_R20074` |
| **Condition** | `phase1_vision_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `vision`（纯截图，无文本树） / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False，无 N/A 排除） |
| **SR** | **9.62%**（10 success / 94 failed） |
| **ruleset_version** | `9-wa-p47p48`（正文成稿于 `8-reddit-p41p46-b1890fix`，v9 数字块见 §2b）|
| **Tier-1 三子集** | failed+hit 81 · **failed-NO-hit 13** · success+hit 1 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step 级命中 | 命中 episode 数 |
|---|---|---|---|
| `P5` | 感知缺失循环 | 100 | 63 |
| `P31` | budget 耗尽未完成 | 62 | 62 |
| `P14` | URL 自环 | 50 | 41 |
| `P12` | 从不翻页 | 13 | 13 |
| `P33` | 导航至裸图片 URL 幻觉 | 7 | 7 |
| `P36` | WALK_FAIL_DEGENERATE | 4 | 3 |
| `P27` | 找不到即放弃 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**：task 30（P40）

**failed-NO-hit episode（deterministic 盲区 → Tier-2 全覆盖）**：[66, 68, 69, 600, 612, 628, 641, 715, 719, 725, 730, 732, 733]

**success episode**：[30, 67, 399, 400, 401, 402, 403, 596, 650, 652]


### 2b. v9 数字块（`9-wa-p47p48`，2026-08-03 补）

> 本 digest 正文成稿于 `8-reddit-p41p46-b1890fix`。v9 落码了 `P47`/`P48` 两条新规则
> （由本轮 Tier-2 的 R1/R3 提议而来），scan 已全量重扫，此处补齐数字使 **B0/B1 可并表**。
> 正文的 Tier-2 定性结论不受影响（v9 是纯 additive，未改动任何既有规则的正则或阈值）。

| 规则 | 含义 | step 级 | episode 级 | v8→v9 |
|---|---|---|---|---|
| `P5` | 感知缺失循环 | 100 | 63 | — |
| `P31` | budget耗尽未完成 | 62 | 62 | — |
| `P14` | URL 自环 | 50 | 41 | — |
| `P12` | 从不翻页 | 13 | 13 | — |
| `P33` | 导航至裸图片URL幻觉 | 7 | 7 | — |
| `P36` | WALK_FAIL_DEGENERATE | 4 | 3 | — |
| `P47` | PREMATURE_FINISH_ON_FORM | 2 | 2 | **0 → 2** |
| `P27` | 找不到即放弃 | 1 | 1 | — |

**三子集** v8 `failed+hit 81 · failed-NO-hit 13 · success+hit 1` → v9 `failed+hit 81 · failed-NO-hit 13 · success+hit 1`

**唯一变化**：`P47` 新增命中。`P48` 在本 cell **0 命中**（它由 B1 数据提议却在 B1 上不 fire —— 实现比提议窄，排查见 [[_benchmark_level_findings]]）。

## 3. Tier-2 深挖

**覆盖**：13 个 no-hit **全覆盖** · 1 个 sonnet sub-agent（另有跨 mode 的因果验证组与
`P40` FP 审计组，见 [[_cell_cross_mode_findings]] §3 §6）

**三分类**：agent-limit 13 · scaffold-bug 0 · benchmark-FP 0 · unclear 0

> vision 是 6 个 mode 里 no-hit 比例最高的（13/94 失败），且规则画像与其余 mode 截然不同：
> `P36` 只 3 个 episode、`P45` **0 个**。原因见下方"失败落点"。

- **失败发生在更晚的环节，不是"没找到页面"**：13 个里 11 个成功导航到了语义正确或接近正确的
  页面／表单。具体落点：
  - **表单内点错文本框**（612 / 628）：帖子创建后，必需文本被打进了新帖的**评论框**而不是
    submission body —— URL 序列 `submit/<forum>` → `f/<forum>/<id>/<slug>` →
    `.../comment/<n>`，而 `program_html` 检查的是 `.submission__inner`。
  - **到达目标帖后从未点投票控件**（715 / 719 / 725 / 730）：把"到达"误当"完成"。
  - **profile 列表页选错点击对象**（732 / 733）：两个不同任务收敛到**完全相同**的错误 URL
    `/f/technology/134852/-/comment/14/edit`，暗示是位置性点击偏好而非读标题匹配。
  - **点帖子标题的站外链接而非站内评论区入口**（69 / 715 / 730）。
- **`P36`/`P45` 在 vision 上几乎不 fire 不代表 vision 没有定位失败** —— vision 全程走
  `coord_mouse_click`，完全绕开 element_id / DOM walk 这条检测通路，这类失败**在坐标体系里
  无法被这两条规则观测到**。补位的是 `P5`（63 episode，本 mode 最高）。
- sub-agent 已排除三个替代假设：`must_include` 大小写（读 evaluator 源码确认不敏感）、
  截图管线故障（`screenshot_timeout_recovered_count` / `image_encode_error_step_count` 均 0）、
  缺少 vote 专属 action type（reddit 投票在所有 mode 下都是普通 click）。

## 4. 与 cell 级发现的关联

本 mode 的失败质量主要落在下列 cell 级机制上，详见 [[_cell_cross_mode_findings]]：

- §5 **F1** `select_option` 在 reddit 99.6% 失败（reddit 站点特有，波及 VWA reddit 计分 cell）
- §5 **F2** `walk_fail` 的元素 100% 在观测里且在视口内 → `P36` 需按元素角色分叉
- §5 **F3** WA reddit task 66 reference 硬编码生产域名
- §5 **F4** phantom 系列 artifact 落盘 ≠ 模型输入
- §6 `P31`（本 mode 覆盖 62/94 失败）**裁定为风险标记而非死因指标**
- §7 新规则候选 R1–R4（R1/R3 建议落码，R2 需收窄，**R4 否决**）

## 5. 可 actionable 项

见 [[_cell_cross_mode_findings]] §8（A1–A9）。本 mode 无独有的 actionable 项。
