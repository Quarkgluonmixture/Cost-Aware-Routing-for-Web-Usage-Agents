# /diag digest — B0 × `som` × **WA** reddit

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
| **Run** | `B0_som_wa_reddit_20260801_050840_717420611_3312794_R28517` |
| **Condition** | `phase1_som_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `som`（Set-of-Marks 标注截图 + [SOM_MARKS] 文本） / B0 = Qwen3-VL-235B-A22B (via AWS proxy) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False） |
| **SR** | **22.12%**（23 success / 81 failed） |
| **ruleset_version** | `9-wa-p47p48`（`config_missing=0`） |
| **Tier-1 三子集** | failed+hit 65 · **failed-NO-hit 16** · success+hit 2 |

## 2. Tier-1 规则分布（failed 侧，episode 计数）

| 规则 | 命中 | 占 failed |
|---|---:|---:|
| `P31` | 38 | 46.9% |
| `P36` | 33 | 40.7% |
| `P5` | 29 | 35.8% |
| `P45` | 28 | 34.6% |
| `P12` | 20 | 24.7% |
| `P48` | 8 | 9.9% |
| `P14` | 7 | 8.6% |
| `P33` | 4 | 4.9% |
| `P4` | 3 | 3.7% |
| `P47` | 3 | 3.7% |
| `P27` | 3 | 3.7% |
| `P44` | 1 | 1.2% |

**success 侧 fire 的规则**（presence ≠ causation，见 B6）：

- task 28: `P40`
- task 31: `P40`

## 3. Tier-2 深挖 — 16 个 no-hit failed 全覆盖

**裁决后三分类**：agent-limit **6** · benchmark-FP **1** · scaffold-bug **9** · unclear **0**

| task | 分类 | 根因 |
|---|---|---|
| 29 | agent-limit | 到 /user/Sorkill/comments 后 0 次 scroll 就 finish，漏计 |
| 67 | agent-limit | 多项枚举只点 1 个帖子 |
| 409 | agent-limit | 直接往顶层评论框打字，未定位 manager 的评论 |
| 607 | scaffold-bug | 站点真实发帖限流 |
| 608 | scaffold-bug | 站点真实发帖限流（同账号级联） |
| 609 | scaffold-bug | 站点真实发帖限流 |
| 621 | benchmark-FP | `cheat` 单 token vs agent 写 'cheating'（B2） |
| 632 | scaffold-bug | 站点真实发帖限流 |
| 634 | agent-limit | 自述'posting restriction'但页面无站点横幅 → **幻觉限流**（裁决） |
| 635 | agent-limit | 发到 r/AskReddit 而非 eval 要求的 r/headphones |
| 641 | scaffold-bug | 站点真实发帖限流 |
| 642 | scaffold-bug | 站点真实发帖限流，5 轮重试烧掉 30 步 |
| 644 | scaffold-bug | 站点真实发帖限流 |
| 645 | scaffold-bug | 站点真实发帖限流 |
| 651 | scaffold-bug | 同一 element_id 连点 30 次、url 全程不变，但 CSRF nonce 抖动骗过 page_changed → loop detector 盲区 |
| 714 | agent-limit | 用全局搜索代替 /f/gadgets 排序，踩了无关帖 |

## 4. 与 benchmark 级发现的关联

- **B1 发帖限流**：本 condition no-hit 子集里 8 个坐实站点横幅；全 condition 口径见 [[_benchmark_level_findings]] §B1 表
- **B2 tokenize 假阴性**：本 condition no-hit 子集里 1 个
- **B4 结构性不可通过**：task 66 / 646 —— 与 agent 表现无关

## 5. 可 actionable 项

见 [[_benchmark_level_findings]] §8（C1–C8）。本 condition 无独有的 actionable 项。
