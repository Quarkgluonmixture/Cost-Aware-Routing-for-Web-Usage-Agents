# /diag digest — B0 × `phantom_prompt` × **WA** reddit

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
| **Run** | `B0_phantom_prompt_wa_reddit_20260802_112513_879977727_3523969_R4739` |
| **Condition** | `phase1_phantom_prompt_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `phantom_prompt`（SoM 风格 prompt + AXTree 文本，无图） / B0 = Qwen3-VL-235B-A22B (via AWS proxy) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False） |
| **SR** | **25.96%**（27 success / 77 failed） |
| **ruleset_version** | `9-wa-p47p48`（`config_missing=0`） |
| **Tier-1 三子集** | failed+hit 68 · **failed-NO-hit 9** · success+hit 2 |

## 2. Tier-1 规则分布（failed 侧，episode 计数）

| 规则 | 命中 | 占 failed |
|---|---:|---:|
| `P36` | 53 | 68.8% |
| `P5` | 32 | 41.6% |
| `P31` | 30 | 39.0% |
| `P45` | 27 | 35.1% |
| `P12` | 19 | 24.7% |
| `P27` | 9 | 11.7% |
| `P14` | 9 | 11.7% |
| `P43` | 5 | 6.5% |
| `P33` | 4 | 5.2% |
| `P44` | 2 | 2.6% |
| `P47` | 2 | 2.6% |
| `P48` | 2 | 2.6% |
| `P10` | 1 | 1.3% |

**success 侧 fire 的规则**（presence ≠ causation，见 B6）：

- task 28: `P40`
- task 30: `P40`

## 3. Tier-2 深挖 — 9 个 no-hit failed 全覆盖

**裁决后三分类**：agent-limit **6** · benchmark-FP **1** · scaffold-bug **2** · unclear **0**

| task | 分类 | 根因 |
|---|---|---|
| 409 | agent-limit | 未定位 manager 评论（裁决） |
| 611 | scaffold-bug | 站点真实发帖限流 |
| 631 | scaffold-bug | 站点真实发帖限流，两轮重填重试 |
| 641 | agent-limit | 帖子建成但正文遗漏 must_include 要求的 'virtual' |
| 652 | benchmark-FP | eval 复用 agent 末页 DOM + `querySelector` 单取首个 `.comment__body` → 很可能取到既存旧评论 |
| 718 | agent-limit | 只完成 2/5 且有失败点击未重试，即宣称 5 个全完成 |
| 722 | agent-limit | 反复猜错 slug（真实是 nyc），9 步语义循环后误判无投稿 |
| 728 | agent-limit | 1510 族，误把全文搜索当版面导航 |
| 733 | agent-limit | 点到锚点链接而非保存按钮，beforeunload 静默丢弃编辑，未察觉即宣称成功 |

## 4. 与 benchmark 级发现的关联

- **B1 发帖限流**：本 condition no-hit 子集里 2 个坐实站点横幅；全 condition 口径见 [[_benchmark_level_findings]] §B1 表
- **B2 tokenize 假阴性**：本 condition no-hit 子集里 0 个
- **B4 结构性不可通过**：task 66 / 646 —— 与 agent 表现无关

## 5. 可 actionable 项

见 [[_benchmark_level_findings]] §8（C1–C8）。本 condition 无独有的 actionable 项。

---

### v11 数字块（`11-intent-text-fallback`，2026-08-03 补）

> 本 digest 正文成稿于更早的 ruleset。v10 落了 **+P49 / P36 carve-out / P14 carve-out**，
> v11 给 **P34/P48 换用 `_finish_intent_text()`**（answer 为空时 fallback 读 `thought`——
> B0 惯于把结论写进 `answer`，B1 留在 `thought`，旧口径因此变成了模型行为检测器）。
> 全部 48 个 canonical condition 已在 v11 下重扫，**cross-mode / cross-model 聚合以本块为准**。

| 字段 | 值 |
|---|---|
| Run | `B0_phantom_prompt_wa_reddit_20260802_112513_879977727_3523969_R4739` |
| Episodes | 104（success 27 · SR 25.96%） |
| 三子集 | failed+hit 63 · failed-NO-hit 14 · success+hit 2 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P36` | WALK_FAIL_DEGENERATE | 210 | 37 |
| `P5` | 感知缺失循环 | 60 | 32 |
| `P31` | budget耗尽未完成 | 30 | 30 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 52 | 27 |
| `P12` | 从不翻页 | 19 | 19 |
| `P14` | URL 自环 | 9 | 9 |
| `P27` | 找不到即放弃 | 9 | 9 |
| `P49` | SUBMIT_PAGE_ANCHOR_MISCLICK | 5 | 5 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 5 | 5 |
| `P33` | 导航至裸图片URL幻觉 | 4 | 4 |
| `P44` | HALLUCINATED_ELEMENT_REF | 2 | 2 |
| `P47` | PREMATURE_FINISH_ON_FORM | 2 | 2 |
| `P48` | PREMATURE_NEGATIVE_AFTER_SEARCH | 2 | 2 |
| `P10` | 跨步数值记忆失败 | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
