# /diag digest — B0 × `phantom_text` × **WA** reddit

*生成 2026-08-03（Tier-1 全扫 + Tier-2 深挖 + Tier-3 整合）*

> **定位声明**：本 digest 是**单 condition** 的失败归因；per-rule 分布只描述它自己。
> 跨 model / 跨 mode 的结构性问题（发帖限流、evaluator 假阴性、框架 bug、计分口径）
> 统一收在 **[[_benchmark_level_findings]]**，本文件不复述，只引用编号（B1–B7）。
> B1 model 的 cell 级发现见 [[_cell_cross_mode_findings]]。
>
> ⚠️ **这是 WebArena，不是 VisualWebArena。** 不要与 `docs/analysis/vwa_reddit/` 下的同名
> mode digest 并表 —— 任务集不同（WA 104 scored / VWA 205 collected），WA reddit **0/104** 带 image。

## 1. Header

| 字段 | 值 |
|---|---|
| **Run** | `B0_phantom_text_wa_reddit_20260802_005017_553273903_3453420_R26435` |
| **Condition** | `phase1_phantom_text_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `phantom_text`（DOM 风格 prompt + [SOM_MARKS] 文本，无图） / B0 = Qwen3-VL-235B-A22B (via AWS proxy) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False） |
| **SR** | **35.58%**（37 success / 67 failed） |
| **ruleset_version** | `9-wa-p47p48`（`config_missing=0`） |
| **Tier-1 三子集** | failed+hit 58 · **failed-NO-hit 9** · success+hit 3 |

## 2. Tier-1 规则分布（failed 侧，episode 计数）

| 规则 | 命中 | 占 failed |
|---|---:|---:|
| `P36` | 45 | 67.2% |
| `P31` | 40 | 59.7% |
| `P5` | 38 | 56.7% |
| `P45` | 36 | 53.7% |
| `P12` | 26 | 38.8% |
| `P14` | 8 | 11.9% |
| `P43` | 5 | 7.5% |
| `P33` | 4 | 6.0% |
| `P27` | 3 | 4.5% |
| `P47` | 2 | 3.0% |
| `P4` | 2 | 3.0% |
| `P44` | 1 | 1.5% |
| `P10` | 1 | 1.5% |
| `P48` | 1 | 1.5% |

**success 侧 fire 的规则**（presence ≠ causation，见 B6）：

- task 28: `P40`
- task 30: `P40`
- task 595: `P33`

## 3. Tier-2 深挖 — 9 个 no-hit failed 全覆盖

**裁决后三分类**：agent-limit **8** · benchmark-FP **0** · scaffold-bug **1** · unclear **0**

| task | 分类 | 根因 |
|---|---|---|
| 68 | agent-limit | top-10 枚举只访问 1 个帖子，且把 reddit 用户名当书籍作者 |
| 409 | agent-limit | 未定位 manager 评论（**裁决**：可通过，反例 task 410） |
| 647 | agent-limit | 内容语义对但绕开字面关键词（写 'aid' 而 must_include 要 'help'） |
| 651 | scaffold-bug | INERT_CLICK_LOOP — 点击 action_success=True 但 url/scroll_y/dom_complexity 三者全程零变化，恰落在 P31 与 P45 的夹缝 |
| 722 | agent-limit | search-only 放弃，never browse |
| 725 | agent-limit | intent_template_id=1510 族 |
| 728 | agent-limit | 同族，1 次搜索即投降 |
| 729 | agent-limit | 同族，3 次重复同一失效策略 |
| 730 | agent-limit | 同族，自造 `site:` 搜索算子 |

## 4. 与 benchmark 级发现的关联

- **B1 发帖限流**：本 condition no-hit 子集里 0 个坐实站点横幅；全 condition 口径见 [[_benchmark_level_findings]] §B1 表
- **B2 tokenize 假阴性**：本 condition no-hit 子集里 0 个
- **B4 结构性不可通过**：task 66 / 646 —— 与 agent 表现无关

## 5. 可 actionable 项

见 [[_benchmark_level_findings]] §8（C1–C8）。本 condition 无独有的 actionable 项。
