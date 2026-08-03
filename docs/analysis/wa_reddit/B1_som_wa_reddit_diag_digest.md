# /diag digest — B1 × `som` × **WA** reddit

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
| **Run** | `B1_som_wa_reddit_20260728_090436_011011933_2760426_R301` |
| **Condition** | `phase1_som_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `som`（Set-of-Marks：标注图 + `[SOM_MARKS]` 文本） / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False，无 N/A 排除） |
| **SR** | **13.46%**（14 success / 90 failed） |
| **ruleset_version** | `9-wa-p47p48`（正文成稿于 `8-reddit-p41p46-b1890fix`，v9 数字块见 §2b）|
| **Tier-1 三子集** | failed+hit 80 · **failed-NO-hit 10** · success+hit 1 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step 级命中 | 命中 episode 数 |
|---|---|---|---|
| `P31` | budget 耗尽未完成 | 66 | 66 |
| `P36` | WALK_FAIL_DEGENERATE | 353 | 46 |
| `P5` | 感知缺失循环 | 60 | 45 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 53 | 42 |
| `P12` | 从不翻页 | 27 | 27 |
| `P14` | URL 自环 | 32 | 25 |
| `P4` | 根节点误操作 | 26 | 6 |
| `P33` | 导航至裸图片 URL 幻觉 | 3 | 3 |
| `P10` | 跨步数值记忆失败 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**：task 27（P40）

**failed-NO-hit episode（deterministic 盲区 → Tier-2 全覆盖）**：[29, 69, 581, 610, 651, 721, 724, 727, 730, 732]

**success episode**：[27, 67, 399, 400, 402, 403, 597, 598, 599, 600, 605, 609, 650, 652]


### 2b. v9 数字块（`9-wa-p47p48`，2026-08-03 补）

> 本 digest 正文成稿于 `8-reddit-p41p46-b1890fix`。v9 落码了 `P47`/`P48` 两条新规则
> （由本轮 Tier-2 的 R1/R3 提议而来），scan 已全量重扫，此处补齐数字使 **B0/B1 可并表**。
> 正文的 Tier-2 定性结论不受影响（v9 是纯 additive，未改动任何既有规则的正则或阈值）。

| 规则 | 含义 | step 级 | episode 级 | v8→v9 |
|---|---|---|---|---|
| `P31` | budget耗尽未完成 | 66 | 66 | — |
| `P36` | WALK_FAIL_DEGENERATE | 353 | 46 | — |
| `P5` | 感知缺失循环 | 60 | 45 | — |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 53 | 42 | — |
| `P12` | 从不翻页 | 27 | 27 | — |
| `P14` | URL 自环 | 32 | 25 | — |
| `P4` | 根节点误操作 | 26 | 6 | — |
| `P33` | 导航至裸图片URL幻觉 | 3 | 3 | — |
| `P47` | PREMATURE_FINISH_ON_FORM | 3 | 3 | **0 → 3** |
| `P10` | 跨步数值记忆失败 | 1 | 1 | — |

**三子集** v8 `failed+hit 80 · failed-NO-hit 10 · success+hit 1` → v9 `failed+hit 81 · failed-NO-hit 9 · success+hit 1`

**唯一变化**：`P47` 新增命中。`P48` 在本 cell **0 命中**（它由 B1 数据提议却在 B1 上不 fire —— 实现比提议窄，排查见 [[_benchmark_level_findings]]）。

**从 no-hit 转入 hit 的 episode**（v9 新规则接住的 deterministic 盲区）：[732]

## 3. Tier-2 深挖

**覆盖**：10 个 no-hit **全覆盖** · 1 个 sonnet sub-agent（另有跨 mode 的因果验证组与
`P40` FP 审计组，见 [[_cell_cross_mode_findings]] §3 §6）

**三分类**：scaffold-bug 1 · agent-limit 9 · benchmark-FP 0 · unclear 0

- **task 581 → scaffold-bug**：agent 认定的"表单提交按钮" `element_id=10` 实际是站点常驻顶栏的
  `Submit` 导航链接（`locator_route_meta.target_tag='A'`，无条件跳 `/submit`），不是
  `create_forum` 表单自己的提交控件。三次 fill→click(10)→back 循环后放弃。
  ⚠️ 这条与 phantom_prompt task 580/583（bbox 恒为 `[852,0,103,52]` 的顶栏链接）是同一机制，
  但**尚未做全量复核**，见 [[_cell_cross_mode_findings]] §7 末。
- **task 721 / 727 / 730 → 用户名搜索代替 profile 导航，过早否定**：把用户名塞进全站搜索框，
  零命中即断言"该用户无投稿"。**硬反证**：同批 task 724 证明同一用户 Hrekires 在同一版
  确实有投稿（经 `/user/Hrekires/submissions` 找到）→ 730 的否定结论是错的，不是评测过严。
  → 规则候选 **R3**。
- **task 29 / 69 → 搜索代替论坛浏览**：到了 `/forums` 不滚动就改用全站关键词搜索，
  把搜索命中的无关帖当成目标论坛最新帖。
- **task 610 → 发成了评论而非帖子正文**：evaluator 检查 `.submission__inner`，agent 把书评
  发在了帖子下的独立回复里（最终 URL 落在 `/comment/2`）。
- **task 651 → 同 dom**：对 `id=28` 连续点击 30 步，`state_digest.text_length` 从 step_9 起
  锁死在 2843。sub-agent 还顺带查到 runner 里已实现 `_detect_action_cycle`／
  `_anti_repeat_control`，但开关 `diagnostic_controls.anti_repeat.enabled` 默认 False
  且全库无 yaml 打开过 —— 该断言我**未复核**，列此备查。

## 4. 与 cell 级发现的关联

本 mode 的失败质量主要落在下列 cell 级机制上，详见 [[_cell_cross_mode_findings]]：

- §5 **F1** `select_option` 在 reddit 99.6% 失败（reddit 站点特有，波及 VWA reddit 计分 cell）
- §5 **F2** `walk_fail` 的元素 100% 在观测里且在视口内 → `P36` 需按元素角色分叉
- §5 **F3** WA reddit task 66 reference 硬编码生产域名
- §5 **F4** phantom 系列 artifact 落盘 ≠ 模型输入
- §6 `P31`（本 mode 覆盖 66/90 失败）**裁定为风险标记而非死因指标**
- §7 新规则候选 R1–R4（R1/R3 建议落码，R2 需收窄，**R4 否决**）

## 5. 可 actionable 项

见 [[_cell_cross_mode_findings]] §8（A1–A9）。本 mode 无独有的 actionable 项。

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B1_som_wa_reddit_20260728_090436_011011933_2760426_R301` |
| Episodes | 104（success 14 · SR 13.46%） |
| 三子集 | failed+hit 81 · failed-NO-hit 9 · success+hit 1 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P31` | budget耗尽未完成 | 66 | 66 |
| `P5` | 感知缺失循环 | 60 | 45 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 53 | 42 |
| `P36` | WALK_FAIL_DEGENERATE | 280 | 30 |
| `P12` | 从不翻页 | 27 | 27 |
| `P14` | URL 自环 | 32 | 25 |
| `P49` | SUBMIT_PAGE_ANCHOR_MISCLICK | 9 | 9 |
| `P4` | 根节点误操作 | 26 | 6 |
| `P48` | PREMATURE_NEGATIVE_AFTER_SEARCH | 5 | 5 |
| `P33` | 导航至裸图片URL幻觉 | 3 | 3 |
| `P47` | PREMATURE_FINISH_ON_FORM | 3 | 3 |
| `P10` | 跨步数值记忆失败 | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
