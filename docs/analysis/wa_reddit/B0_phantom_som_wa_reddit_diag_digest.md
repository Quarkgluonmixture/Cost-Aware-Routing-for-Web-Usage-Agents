# /diag digest — B0 × `phantom_som` × **WA** reddit

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
| **Run** | `B0_phantom_som_wa_reddit_20260802_200105_448982698_3591110_R14533` |
| **Condition** | `phase1_phantom_som_router_0` |
| **Benchmark / Site / Mode / Model** | WebArena / reddit / `phantom_som`（SoM prompt + [SOM_MARKS] 文本，**跳过标注图**；部署代表臂） / B0 = Qwen3-VL-235B-A22B (via AWS proxy) |
| **Episodes** | 104 collected = **104 scored**（`sr_excluded` 全 False） |
| **SR** | **25.00%**（26 success / 78 failed） |
| **ruleset_version** | `9-wa-p47p48`（`config_missing=0`） |
| **Tier-1 三子集** | failed+hit 63 · **failed-NO-hit 15** · success+hit 1 |

## 2. Tier-1 规则分布（failed 侧，episode 计数）

| 规则 | 命中 | 占 failed |
|---|---:|---:|
| `P36` | 47 | 60.3% |
| `P31` | 35 | 44.9% |
| `P5` | 34 | 43.6% |
| `P45` | 33 | 42.3% |
| `P12` | 21 | 26.9% |
| `P14` | 10 | 12.8% |
| `P27` | 5 | 6.4% |
| `P43` | 5 | 6.4% |
| `P33` | 4 | 5.1% |
| `P47` | 2 | 2.6% |
| `P10` | 2 | 2.6% |
| `P4` | 1 | 1.3% |
| `P48` | 1 | 1.3% |

**success 侧 fire 的规则**（presence ≠ causation，见 B6）：

- task 595: `P33`

## 3. Tier-2 深挖 — 15 个 no-hit failed 全覆盖

**裁决后三分类**：agent-limit **9** · benchmark-FP **2** · scaffold-bug **4** · unclear **0**

| task | 分类 | 根因 |
|---|---|---|
| 68 | agent-limit | top-10 只覆盖约 5 个帖子且语义误判 |
| 408 | agent-limit | 用搜索代替按 new 排序，落在非最新帖，误读 'Retract upvote' 为已完成 |
| 409 | agent-limit | 未定位 manager 评论（裁决） |
| 607 | scaffold-bug | 站点真实发帖限流 |
| 608 | scaffold-bug | 站点真实发帖限流 |
| 609 | scaffold-bug | 站点真实发帖限流 |
| 620 | benchmark-FP | `relation` vs 'relationship'（B2） |
| 632 | benchmark-FP | `shoes` vs agent 写 'shoe'（B2） |
| 644 | agent-limit | 发到 f/gaming 而非 f/games，且正文无 'virtual meetup' |
| 651 | scaffold-bug | INERT_CLICK_LOOP，30/30 步完全相同的 (action, element_id, url) 元组 |
| 652 | agent-limit | 模型输出字面 `\n` 污染 exact_match（**裁决**：非转义 bug，见 B7） |
| 718 | agent-limit | 'top 5' 量词坍缩成 'top 1'，finish 自曝只踩了 1 个 |
| 727 | agent-limit | 1510 族 |
| 728 | agent-limit | 1510 族 + 重复点进错误 subreddit 3 次 |
| 730 | agent-limit | 1510 族，仅滚 2 次就宣称 'all' 完成（eval 需 10 个 permalink） |

## 4. 与 benchmark 级发现的关联

- **B1 发帖限流**：本 condition no-hit 子集里 3 个坐实站点横幅；全 condition 口径见 [[_benchmark_level_findings]] §B1 表
- **B2 tokenize 假阴性**：本 condition no-hit 子集里 2 个
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
| Run | `B0_phantom_som_wa_reddit_20260802_200105_448982698_3591110_R14533` |
| Episodes | 104（success 26 · SR 25.00%） |
| 三子集 | failed+hit 58 · failed-NO-hit 20 · success+hit 1 |
| config_missing | 0 |

| 规则 | 含义 | step 级 | episode 级 |
|---|---|---:|---:|
| `P31` | budget耗尽未完成 | 35 | 35 |
| `P5` | 感知缺失循环 | 66 | 34 |
| `P45` | IDENTICAL_FAILED_ACTION_STREAK | 65 | 33 |
| `P36` | WALK_FAIL_DEGENERATE | 88 | 23 |
| `P12` | 从不翻页 | 21 | 21 |
| `P14` | URL 自环 | 12 | 10 |
| `P49` | SUBMIT_PAGE_ANCHOR_MISCLICK | 9 | 9 |
| `P27` | 找不到即放弃 | 5 | 5 |
| `P43` | PAGE_EMBEDDED_VISUAL_NO_SCREENSHOT | 5 | 5 |
| `P33` | 导航至裸图片URL幻觉 | 4 | 4 |
| `P47` | PREMATURE_FINISH_ON_FORM | 2 | 2 |
| `P10` | 跨步数值记忆失败 | 2 | 2 |
| `P4` | 根节点误操作 | 2 | 1 |
| `P48` | PREMATURE_NEGATIVE_AFTER_SEARCH | 1 | 1 |

> ⚠️ **解读约束**（`docs/analysis/_data_quality_audit.md`）：
> ① 本表是**症状分布，不是死因分布** —— P36/P31 经 10 例跨 benchmark 因果验证均判为 risk-marker；
> ② `P2`/`P4` 依赖 `element_bbox`，在 **vision 上结构性为 0（假 0）**；
> ③ `P36` 在 vision 上只覆盖 `type` 步（click 无 `locator_route_meta`）→ **分母与 dom/som 不同**。
