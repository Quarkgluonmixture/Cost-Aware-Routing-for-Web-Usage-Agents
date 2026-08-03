# /diag digest — B1 × `dom` × **WA** reddit

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
| **Run** | `B1_dom_wa_reddit_20260727_180024_017253388_2658596_R13217` |
| **Condition** | `phase1_dom_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `dom`（AXTree 文本，无图） / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False，无 N/A 排除） |
| **SR** | **16.35%**（17 success / 87 failed） |
| **ruleset_version** | `9-wa-p47p48`（正文成稿于 `8-reddit-p41p46-b1890fix`，v9 数字块见 §2b）|
| **Tier-1 三子集** | failed+hit 80 · **failed-NO-hit 7** · success+hit 2 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step 级命中 | 命中 episode 数 |
|---|---|---|---|
| `P31` | budget 耗尽未完成 | 66 | 66 |
| `P36` | WALK_FAIL_DEGENERATE | 355 | 61 |
| `P5` | 感知缺失循环 | 44 | 31 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 39 | 26 |
| `P14` | URL 自环 | 15 | 15 |
| `P12` | 从不翻页 | 13 | 13 |
| `P44` | HALLUCINATED_ELEMENT_REF | 13 | 6 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT（中性标签） | 5 | 5 |
| `P33` | 导航至裸图片 URL 幻觉 | 4 | 4 |
| `P46` | COMMENT_INTENT_NO_TYPE | 1 | 1 |
| `P13` | 搜索代替浏览 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**：task 27（P40）、task 28（P40）

**failed-NO-hit episode（deterministic 盲区 → Tier-2 全覆盖）**：[66, 621, 641, 645, 651, 728, 729]

**success episode**：[27, 28, 69, 399, 400, 401, 402, 403, 404, 407, 595, 597, 598, 599, 605, 626, 650]


### 2b. v9 数字块（`9-wa-p47p48`，2026-08-03 补）

> 本 digest 正文成稿于 `8-reddit-p41p46-b1890fix`。v9 落码了 `P47`/`P48` 两条新规则
> （由本轮 Tier-2 的 R1/R3 提议而来），scan 已全量重扫，此处补齐数字使 **B0/B1 可并表**。
> 正文的 Tier-2 定性结论不受影响（v9 是纯 additive，未改动任何既有规则的正则或阈值）。

| 规则 | 含义 | step 级 | episode 级 | v8→v9 |
|---|---|---|---|---|
| `P31` | budget耗尽未完成 | 66 | 66 | — |
| `P36` | WALK_FAIL_DEGENERATE | 355 | 61 | — |
| `P5` | 感知缺失循环 | 44 | 31 | — |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 39 | 26 | — |
| `P14` | URL 自环 | 15 | 15 | — |
| `P12` | 从不翻页 | 13 | 13 | — |
| `P44` | HALLUCINATED_ELEMENT_REF | 13 | 6 | — |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 5 | 5 | — |
| `P33` | 导航至裸图片URL幻觉 | 4 | 4 | — |
| `P47` | PREMATURE_FINISH_ON_FORM | 4 | 4 | **0 → 4** |
| `P46` | COMMENT_INTENT_NO_TYPE | 1 | 1 | — |
| `P13` | 搜索代替浏览 | 1 | 1 | — |

**三子集** v8 `failed+hit 80 · failed-NO-hit 7 · success+hit 2` → v9 `failed+hit 80 · failed-NO-hit 7 · success+hit 2`

**唯一变化**：`P47` 新增命中。`P48` 在本 cell **0 命中**（它由 B1 数据提议却在 B1 上不 fire —— 实现比提议窄，排查见 [[_benchmark_level_findings]]）。

## 3. Tier-2 深挖

**覆盖**：7 个 no-hit **全覆盖** · 1 个 sonnet sub-agent（另有跨 mode 的因果验证组与
`P40` FP 审计组，见 [[_cell_cross_mode_findings]] §3 §6）

**三分类**：benchmark-FP 1 · agent-limit 6 · scaffold-bug 0 · unclear 0

- **task 66 → benchmark-FP（本轮最值得升级处理的一条）**：`reference_answers` 硬编码
  `http://www.reddit.com/f/books/...`，部署域名是 `localhost:9999`，`must_include` 纯子串比对
  → 结构性不可解。**已全量复核 7 份 raw config**：仅此 1 例，VWA 三站 0 命中。详见
  [[_cell_cross_mode_findings]] §5 F3。
- **task 728 / 729 → 复合搜索空结果后草率 finish**：`type('sirbarani sports')` →
  `/search?q=...` → 2 步内 finish 判"无此类帖子"，从未访问 `/user/<name>` 或 `/f/<forum>`。
  站内搜索只索引标题/正文，不索引作者。→ 全量复核后成为规则候选 **R3**（9 failed / 0 success）。
- **task 651 → 30 步原地死锁**：连续点击同一 "87 comments" 锚点，thought 逐字重复，
  `dom_complexity` 恒为 55、`obs_url` 全程不变，直到预算耗尽。
- **task 641 → 措辞改写导致 must_include 失配**：required `'virtual meetup'`，agent 打的是
  `'virtual Harry Potter enthusiasts meetup'` —— 中间被插入词打断，不构成连续子串。**与中途
  `select_option` 失败无关**：那个子串从头到尾就没被打出来过。
- **task 621 / 645**：把创建型 intent 当成检索任务（全程无 `/submit/`）／表单字段疑似被重置后
  只补打了 title。645 的具体机制**无法确认**（该 episode 的落盘观测已不可用），root_cause 保留了
  这层不确定性。

## 4. 与 cell 级发现的关联

本 mode 的失败质量主要落在下列 cell 级机制上，详见 [[_cell_cross_mode_findings]]：

- §5 **F1** `select_option` 在 reddit 99.6% 失败（reddit 站点特有，波及 VWA reddit 计分 cell）
- §5 **F2** `walk_fail` 的元素 100% 在观测里且在视口内 → `P36` 需按元素角色分叉
- §5 **F3** WA reddit task 66 reference 硬编码生产域名
- §5 **F4** phantom 系列 artifact 落盘 ≠ 模型输入
- §6 `P31`（本 mode 覆盖 66/87 失败）**裁定为风险标记而非死因指标**
- §7 新规则候选 R1–R4（R1/R3 建议落码，R2 需收窄，**R4 否决**）

## 5. 可 actionable 项

见 [[_cell_cross_mode_findings]] §8（A1–A9）。本 mode 无独有的 actionable 项。
