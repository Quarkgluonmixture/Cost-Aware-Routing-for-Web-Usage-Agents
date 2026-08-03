# /diag digest — B0 × `dom` × **WA** reddit

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
| **Run** | `B0_dom_wa_reddit_20260731_195425_442316725_3242503_R10765` |
| **Condition** | `phase1_dom_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `dom`（AXTree 纯文本，无图） / B0 = Qwen3-VL-235B-A22B (via AWS proxy) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False） |
| **SR** | **26.92%**（28 success / 76 failed） |
| **ruleset_version** | `9-wa-p47p48`（`config_missing=0`） |
| **Tier-1 三子集** | failed+hit 62 · **failed-NO-hit 14** · success+hit 1 |

## 2. Tier-1 规则分布（failed 侧，episode 计数）

| 规则 | 命中 | 占 failed |
|---|---:|---:|
| `P36` | 49 | 64.5% |
| `P5` | 31 | 40.8% |
| `P31` | 30 | 39.5% |
| `P45` | 30 | 39.5% |
| `P12` | 15 | 19.7% |
| `P27` | 9 | 11.8% |
| `P14` | 5 | 6.6% |
| `P43` | 5 | 6.6% |
| `P33` | 4 | 5.3% |
| `P44` | 3 | 3.9% |
| `P48` | 2 | 2.6% |
| `P47` | 1 | 1.3% |

**success 侧 fire 的规则**（presence ≠ causation，见 B6）：

- task 31: `P40`

## 3. Tier-2 深挖 — 14 个 no-hit failed 全覆盖

**裁决后三分类**：agent-limit **10** · benchmark-FP **3** · scaffold-bug **1** · unclear **0**

| task | 分类 | 根因 |
|---|---|---|
| 29 | agent-limit | DOM 缺票数符号线索，同一个 '12' 在 4 步内被反号重解读，最终数错 |
| 67 | agent-limit | top-10 枚举只点开 1 个帖子就 finish，漏掉第二个必需书名 |
| 68 | agent-limit | 同 67 同族（intent_template_id=17），只查 1 个候选帖 |
| 408 | benchmark-FP | '最新帖' 的静态 reference id 在长期不重置环境里已漂移，agent 点的是执行时真实最新帖 |
| 409 | agent-limit | 未先辨认哪条评论是 manager 发的，直接用顶层评论框 → 回复落在错误位置（**裁决**：非结构性不可通过，见 B4/反例 task 410） |
| 584 | agent-limit | 建版时把 sidebar 内容填进 'Tags' 输入框，且 finish 前未回访核实 |
| 604 | agent-limit | /forums 只滚 1 次就选定 technology，落在 eval 接受集合外 |
| 607 | agent-limit | 误点导航栏 Submit 链接致表单重置，把自身失败**幻觉**成限流（页面无站点横幅，见 B1 幻觉口径） |
| 608 | scaffold-bug | 站点真实发帖限流横幅（observation 侧坐实） |
| 620 | benchmark-FP | `long`/`relation` 单 token 精确匹配 vs agent 写 'long-distance relationship'（B2） |
| 646 | benchmark-FP | **结构性不可通过** — ref `/f/diy` vs 站点只渲染 `/f/DIY`（B4） |
| 722 | agent-limit | 用全文搜索框当'按作者查找'，0 命中即断言无投稿 |
| 725 | agent-limit | 同 722 同族（intent_template_id=1510） |
| 729 | agent-limit | 同族，且自造 `author:` 检索算子（Postmill 不支持） |

## 4. 与 benchmark 级发现的关联

- **B1 发帖限流**：本 condition no-hit 子集里 1 个坐实站点横幅；全 condition 口径见 [[_benchmark_level_findings]] §B1 表
- **B2 tokenize 假阴性**：本 condition no-hit 子集里 1 个
- **B4 结构性不可通过**：task 66 / 646 —— 与 agent 表现无关

## 5. 可 actionable 项

见 [[_benchmark_level_findings]] §8（C1–C8）。本 condition 无独有的 actionable 项。
