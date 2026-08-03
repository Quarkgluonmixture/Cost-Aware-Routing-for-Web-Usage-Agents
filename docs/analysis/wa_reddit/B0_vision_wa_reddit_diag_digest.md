# /diag digest — B0 × `vision` × **WA** reddit

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
| **Run** | `B0_vision_wa_reddit_20260801_140334_349425772_3381851_R10604` |
| **Condition** | `phase1_vision_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `vision`（纯截图，坐标 grounding，无 AXTree） / B0 = Qwen3-VL-235B-A22B (via AWS proxy) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False） |
| **SR** | **19.23%**（20 success / 84 failed） |
| **ruleset_version** | `9-wa-p47p48`（`config_missing=0`） |
| **Tier-1 三子集** | failed+hit 67 · **failed-NO-hit 17** · success+hit 3 |

## 2. Tier-1 规则分布（failed 侧，episode 计数）

| 规则 | 命中 | 占 failed |
|---|---:|---:|
| `P31` | 58 | 69.0% |
| `P5` | 29 | 34.5% |
| `P14` | 19 | 22.6% |
| `P33` | 11 | 13.1% |
| `P36` | 9 | 10.7% |
| `P12` | 6 | 7.1% |
| `P47` | 3 | 3.6% |
| `P27` | 2 | 2.4% |
| `P48` | 1 | 1.2% |

**success 侧 fire 的规则**（presence ≠ causation，见 B6）：

- task 27: `P40`
- task 30: `P33` + `P40`
- task 597: `P33`

## 3. Tier-2 深挖 — 17 个 no-hit failed 全覆盖

**裁决后三分类**：agent-limit **8** · benchmark-FP **4** · scaffold-bug **5** · unclear **0**

| task | 分类 | 根因 |
|---|---|---|
| 66 | agent-limit | top-10 只探索 1 个候选且语义误判 |
| 67 | scaffold-bug | step_8 已记下 The Hobbit，finish 时消失 — `_format_history` 丢 thought（B5） |
| 68 | scaffold-bug | 同 67，step_11 确认的 Christmas Carol 在 finish 时丢失（B5） |
| 69 | agent-limit | 被帖内外链带出站到 crimereads.com 后未返回即 finish |
| 409 | agent-limit | 未定位 manager 评论 |
| 583 | agent-limit | 误点全局 header 的 Submit，自我纠错后未回退重试，仍宣称成功 |
| 600 | benchmark-FP | 开放式 intent（'a subreddit where I'm likely to get an answer'）被压成单一 golden subreddit |
| 604 | scaffold-bug | `select_option` CSS 菜单匹配器只认隐藏菜单，对 reddit 可见展开 `<ul>` 100% 失败（= B1 已坐实的 F1） |
| 606 | scaffold-bug | 站点真实发帖限流 |
| 607 | agent-limit | 把'新建帖子'误认成'在已有帖下评论' |
| 608 | scaffold-bug | 站点真实发帖限流 |
| 621 | benchmark-FP | `cheat` vs 'cheating'（B2） |
| 622 | agent-limit | 把发帖任务当成搜索阅读任务，全程未访问 /submit/ |
| 624 | benchmark-FP | `break` 单 token vs agent 写 'break-up'（B2） |
| 629 | agent-limit | 把'创建讨论帖'做成'在已有帖下评论' |
| 635 | agent-limit | 内容正确但发错 subreddit（technology 而非 headphones） |
| 646 | benchmark-FP | **结构性不可通过**（B4） |

## 4. 与 benchmark 级发现的关联

- **B1 发帖限流**：本 condition no-hit 子集里 2 个坐实站点横幅；全 condition 口径见 [[_benchmark_level_findings]] §B1 表
- **B2 tokenize 假阴性**：本 condition no-hit 子集里 2 个
- **B4 结构性不可通过**：task 66 / 646 —— 与 agent 表现无关

## 5. 可 actionable 项

见 [[_benchmark_level_findings]] §8（C1–C8）。本 condition 无独有的 actionable 项。
