# /diag digest — B1 × `phantom_prompt` / P-prompt × **WA** reddit

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
| **Run** | `B1_phantom_prompt_wa_reddit_20260730_073250_892705973_3033575_R21734` |
| **Condition** | `phase1_phantom_prompt_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `phantom_prompt` / P-prompt（SoM 风格 prompt + 原生 AXTree 文本，无图） / B1 = Qwen3-VL-4B (local) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False，无 N/A 排除） |
| **SR** | **16.35%**（17 success / 87 failed） |
| **ruleset_version** | `8-reddit-p41p46-b1890fix`（config 注入重扫） |
| **Tier-1 三子集** | failed+hit 81 · **failed-NO-hit 6** · success+hit 0 |

## 2. Tier-1 规则分布（failed 侧）

| 规则 | 含义 | step 级命中 | 命中 episode 数 |
|---|---|---|---|
| `P36` | WALK_FAIL_DEGENERATE | 516 | 73 |
| `P31` | budget 耗尽未完成 | 69 | 69 |
| `P5` | 感知缺失循环 | 78 | 53 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 69 | 50 |
| `P14` | URL 自环 | 37 | 32 |
| `P12` | 从不翻页 | 17 | 17 |
| `P44` | HALLUCINATED_ELEMENT_REF | 28 | 9 |
| `P33` | 导航至裸图片 URL 幻觉 | 6 | 6 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT（中性标签） | 5 | 5 |
| `P27` | 找不到即放弃 | 1 | 1 |

**success 侧 fire 的规则（presence-only 误报审计对象）**：（无）

**failed-NO-hit episode（deterministic 盲区 → Tier-2 全覆盖）**：[27, 29, 580, 583, 729, 731]

**success episode**：[69, 399, 400, 401, 402, 403, 404, 595, 597, 598, 599, 607, 623, 629, 641, 650, 651]

## 3. Tier-2 深挖

**覆盖**：6 个 no-hit **全覆盖** · 1 个 sonnet sub-agent（另有跨 mode 的因果验证组与
`P40` FP 审计组，见 [[_cell_cross_mode_findings]] §3 §6）

**三分类**：scaffold-bug 1 · agent-limit 5 · benchmark-FP 0 · unclear 0

- **task 731 → scaffold-bug（代码级断言）**：`type()` 的 prompt 文案写的是 "clicks the target
  to focus it, then types the text"（暗示插入/追加），而实际派发走 Playwright `.fill()` =
  **清空后整体替换**。task 731 要求"给正文追加一行"，模型只 type 了新增行、没有重建
  "原文+新增"，原正文被静默清空 → `program_html` 要求两个字符串同时存在，必败。
  ⚠️ 该断言引用了 `locator_dispatch.py` 与 `_shared_vl_utils.py` 的具体行号，**我未逐行复核**，
  列为待查（见 [[_cell_cross_mode_findings]] §7 末）。
- **task 27 / 29 → 自我身份混淆**：intent 里 "the user who made the latest post on the
  `<X>` forum"（未知第三方）被模型直接等同于**自己**（已登录账号 MarvelsGrantMan136），
  全程在自己的 profile/trash/submissions 页打转，从未访问 `/f/Showerthoughts` 或 `/f/DIY`。
  ⚠️ sub-agent 由此提出 "forum-slug 从未访问" 规则并列为首选；**全量复核后否决** ——
  它在 31/87 = 36% 的 **success** episode 上 fire，是 presence-only。见 §7 R4。
- **task 580 / 583 → 顶栏 Submit 链接吞掉表单**：两次点击 bbox 完全一致
  `[852.0, 0.0, 103.0, 52.0]`（y=0 = 页头），与 phantom_som task 583 / som task 581 同机制。
- **task 729 → 搜索两次即放弃**（同 dom 728/729）。
- **P-prompt 专属问题的回答**：这 6 个 episode 里**没有观察到** "模型报小整数 SoM 风格 id 但
  页面是稀疏原生 id" 的格式错配 —— 用到的 id 全是大稀疏数字，与观测格式一致。但发现了一个
  相关的 mode 代价：反复选中顶栏同名 `Submit` 而非表单内按钮，很可能是因为纯 AXTree 文本
  只剩同名 accessible name、缺乏空间上下文去消歧（SoM/vision 下这个差异肉眼可辨）。
  倾向"去图像后的固有代价"（agent-limit / paper finding）而非派发层 bug —— 因为
  `element_id_locator_route` 显示派发本身正确。**仅 6 样本，不外推。**

## 4. 与 cell 级发现的关联

本 mode 的失败质量主要落在下列 cell 级机制上，详见 [[_cell_cross_mode_findings]]：

- §5 **F1** `select_option` 在 reddit 99.6% 失败（reddit 站点特有，波及 VWA reddit 计分 cell）
- §5 **F2** `walk_fail` 的元素 100% 在观测里且在视口内 → `P36` 需按元素角色分叉
- §5 **F3** WA reddit task 66 reference 硬编码生产域名
- §5 **F4** phantom 系列 artifact 落盘 ≠ 模型输入
- §6 `P31`（本 mode 覆盖 69/87 失败）**裁定为风险标记而非死因指标**
- §7 新规则候选 R1–R4（R1/R3 建议落码，R2 需收窄，**R4 否决**）

## 5. 可 actionable 项

见 [[_cell_cross_mode_findings]] §8（A1–A9）。本 mode 无独有的 actionable 项。
