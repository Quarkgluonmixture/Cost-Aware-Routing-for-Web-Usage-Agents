# B0 Shopping 单模式实验报告

> **实验不完整：仅 DOM 模式有数据，无跨模式交叉分析。**
> Run: `B0_3mode_shopping_20260421`
> 模型: Qwen3-VL-235B-A22B（proxy API，MoE 22B 活跃参数）
> 站点: Shopping (Magento), 466 episodes, 仅 phase1_dom_router_0 (DOM, router_off)
> 分析管线默认使用 **adjusted labels**（扣除 N/A FP + visual FP + eval FP）
> DOM 模式详细分析见 `B0_DOM_digest.md`
> **本版数据基于 2026-04-23 分析管线输出**

---

## 1. 成功率

### 1.1 主指标

| 指标 | 值 | 说明 |
|------|----|------|
| Raw SR | 11.80% (55/466) | 未经调整 |
| Reason diagnostics adjusted SR | 6.24% (29/465) | FP 26 个从分子移除 |
| Condition metrics adjusted SR | 6.44% (30/466) | condition_metrics 管线 |
| N/A adjusted SR | 4.37% (19/435) | 移除 31 个 N/A task（30 个为 FP） |
| Visual adjusted SR | 6.22% | 移除 12 个 visual lucky hits |
| Non-visual SR | 6.09% | 仅非视觉任务 |

> 注：不同管线的 adjusted SR 存在差异，源于调整方法不同（见方法论说明）。主报告采用 condition_metrics 管线的 6.44% 作为主指标。

### 1.2 FP 机制说明

**N/A FP（30/31）**：31 个 N/A reference tasks 中 30 个被标记为 FP。Agent prompt 无 N/A 出口（Rule 4: "NEVER give up"），空答案或循环到截断后评测器误判。

**Visual FP（12 个 visual lucky hits）**：DOM 无截图，269 个 visual tasks 中 12 个碰巧通过 url_match/string_match，标记为 visual lucky hits。

**Eval FP**：reason diagnostics 检测到 26 个 FP（含 N/A FP 与 eval FP），从 55 个 raw success 中移除后剩 29 个 adjusted success。

### 1.3 Bootstrap 95% CI

| SR | CI 下界 | CI 上界 | N |
|----|---------|---------|---|
| 6.44% | 4.29% | 8.80% | 466 |

> 基于 condition_metrics 管线 adjusted labels 计算。

### 1.4 统计显著性

仅 1 个 condition，无法进行 McNemar/Wilcoxon 成对检验。统计检验跳过（`notes: "Fewer than 2 conditions -- pairwise tests skipped"`）。

---

## 2. 效率指标

### 2.1 全 episode 效率

| 指标 | DOM |
|------|-----|
| 平均成本 ($/ep) | $0.0424 |
| 平均输入成本 | $0.0371 |
| 平均输出成本 | $0.0053 |
| 平均步数 | 10.20 |
| P95 步延迟 | 16,695 ms (16.7s) |
| No-op rate | 18.6% |
| Page unchanged rate | 26.9% |
| Cost efficiency ratio | 0.1177 |
| 平均浪费成本 | $0.0374/ep |
| 平均重试次数 | 0.0 |

**成本结构**：输入成本占总成本 87.5%（$0.0371/$0.0424），DOM 模式纯文本请求以 AXTree token 为主。

**延迟**：P95 步延迟 16.7s，远低于 classifieds B0 DOM 的 37.5s（shopping AXTree 通常比 classifieds 短）。

**浪费成本**：平均 $0.0374/ep，占总成本 88.2%，反映了大多数 episode 以失败告终（raw SR 仅 11.80%）。

### 2.2 早停触发分布

| 触发原因 | 次数 |
|---------|------|
| action_failed | 664 |
| page_unchanged_streak | 261 |
| no_progress_streak | 261 |

`action_failed` 触发最多（664 次），说明 DOM 模式下大量操作未能改变页面状态。page_unchanged 和 no_progress 触发次数相同（261），两者在同一 episode 中往往同步触发。

### 2.3 状态变化分布

| 变化类型 | 次数 |
|---------|------|
| content_changed | 3370 |
| interactive_elements_changed | 3295 |
| form_fields_changed | 2804 |
| form_value_changed | 2309 |
| url_changed | 1987 |
| scroll_changed | 1920 |
| dom_complexity_changed | 1751 |
| title_changed | 1657 |
| modal_state_changed | 1373 |
| text_length_changed | 535 |
| about_blank_recovery | 3 |

4751 步中 content_changed 最频繁（3370 次），scroll_changed（1920 次）反映 agent 大量使用滚动浏览商品列表。

---

## 3. 失败模式分析

### 3.1 失败原因分布（reason_bucket）

| 失败原因 | 数量 | 占比 |
|----------|------|------|
| fail_finish_eval_mismatch | 157 | 33.8% |
| fail_no_progress | 156 | 33.5% |
| success | 55 | 11.8% |
| fail_finish_wrong_url_not_found | 22 | 4.7% |
| fail_incomplete_or_stuck | 21 | 4.5% |
| fail_max_steps_click_back_loop | 13 | 2.8% |
| fail_early_finish | 12 | 2.6% |
| fail_max_steps_search_repeat | 12 | 2.6% |
| fail_max_steps | 6 | 1.3% |
| fail_finish_empty_answer | 6 | 1.3% |
| fail_finish_claim_missing | 4 | 0.9% |
| fail_parse_error | 1 | 0.2% |

> 数据来源：`condition_reason_summary.csv`，基于 465 episodes（1 episode 差异为管线边界）。

**两大主导失败**：`fail_finish_eval_mismatch`（33.8%）和 `fail_no_progress`（33.5%）合计占 67.3%。

- **fail_finish_eval_mismatch**：Agent 完成了 finish 动作，但答案与评测标准不匹配。在 shopping 站点，这通常源于选错商品、价格计算错误、或遗漏多条件约束。
- **fail_no_progress**：Agent 陷入循环/停滞，步数耗尽仍未取得进展。

### 3.2 Digest 失败类别分布（213 failure episodes）

| 类别 | 数量 | 占比 | 主要表现 |
|------|------|------|---------|
| 答案对齐错误 | 57 | 26.8% | 选错商品、遗漏约束、价格计算错误 |
| 目标不可达 | 43 | 20.2% | 视觉属性（颜色/形状）在 DOM 中不可获取 |
| 执行停滞 | 37 | 17.4% | 重复操作、元素点击失败 |
| 搜索循环 | 24 | 11.3% | 误输入订阅框、重复点击同一结果 |
| 事实推理错误 | 23 | 10.8% | 排序方向误判、价格计算错误 |
| 导航循环 | 10 | 4.7% | click-back 死循环 |
| 过早结束 | 10 | 4.7% | 未完成任务即 finish |
| 导航失败 | 7 | 3.3% | 目标页面无法到达 |
| 流程超时 | 1 | 0.5% | 步数耗尽 |
| 综合失败 | 1 | 0.5% | 多因素叠加 |

### 3.3 DOM 特有缺陷分布

| DOM 缺陷类型 | 数量 | 占失败比 |
|-------------|------|---------|
| 视觉信息缺失 | 73 | 34.3% |
| 空间感知缺失 | 17 | 8.0% |
| element_id 失效 | 16 | 7.5% |
| AXTree 截断 | 2 | 0.9% |
| 不适用（非 DOM 缺陷） | 105 | 49.3% |

**视觉信息缺失是 DOM 模式的核心瓶颈**：73/213（34.3%）的失败直接归因于 DOM 无法提供颜色、形状、布局等视觉属性。Shopping 站点 466 个 task 中有 269 个（57.7%）涉及视觉属性，DOM 模式在这些 task 上存在结构性劣势。

### 3.4 脚手架 vs 模型归因

| 归因 | 数量 | 占比 | 主要类别 |
|------|------|------|---------|
| 脚手架/表征缺陷 | 103 | 48.4% | 目标不可达(43)、执行停滞(21)、答案对齐错误(15) |
| 模型能力问题 | 110 | 51.6% | 答案对齐错误(42)、搜索循环(21)、执行停滞(16) |

两类归因几乎对半。**脚手架缺陷**以视觉信息缺失为主（DOM 模式的固有限制）；**模型能力问题**以答案对齐错误为主（选错商品、约束遗漏等推理错误）。

### 3.5 动作执行统计（失败 episodes）

| 动作类型 | 总次数 | 失败次数 | 失败率 |
|---------|--------|---------|--------|
| click | 518 | 128 | 24.7% |
| type | 148 | 13 | 8.8% |
| scroll | 363 | - | - |

Click 失败率 24.7%——DOM 模式下 agent 通过 element ID 点击，但部分元素中心坐标越界或为容器节点，导致点击无效。

---

## 4. 跨模式交叉分析

> **不适用**：仅 DOM 一个 condition，无法进行跨模式交叉分析。
> 需要至少 SoM 或 Vision 的 shopping 数据才能计算 oracle ceiling、独占任务集、模式间互补性等指标。

---

## 5. 路由信号分析（Confidence & Behavioral Signals）

> B0 为 API 调用（proxy），无 token-level logprobs；仅有 verbalized confidence 和 behavioral signals。
> 数据来源：`signals/combined/confidence_summary.json`

### 5.1 信号覆盖率

| 指标 | 值 |
|------|-----|
| Episodes | 465 |
| Verbalized episode 覆盖率 | 100% |
| Verbalized step 覆盖率 | 99.92% (4747/4751) |
| Token-level | 不可用（API 模式） |

### 5.2 AUROC 区分力

| 信号 | 类型 | AUROC | 95% CI | 判定 |
|------|------|-------|--------|------|
| **action_diversity** | behavioral | **0.6856** | [0.5947, 0.7763] | 可用 |
| **ep_mean_verbalized** | verbalized | **0.6808** | [0.5676, 0.7860] | 可用 |
| action_unique_types | behavioral | 0.6374 | [0.5106, 0.7563] | 边缘 |
| max_repeat_streak | behavioral | 0.6351 | [0.5210, 0.7394] | 边缘 |
| ep_min_verbalized | verbalized | 0.6350 | [0.5386, 0.7323] | 边缘 |
| url_revisit_count | behavioral | 0.5817 | [0.4701, 0.6986] | 不可用 |
| url_revisit_max | behavioral | 0.5481 | [0.4305, 0.6684] | 不可用 |
| url_unique_count | behavioral | 0.4229 | [0.3162, 0.5311] | 不可用 |
| ep_mean_logprob | token_level | N/A | N/A | 不可用 |
| ep_min_logprob | token_level | N/A | N/A | 不可用 |
| ep_mean_entropy | token_level | N/A | N/A | 不可用 |
| ep_max_entropy | token_level | N/A | N/A | 不可用 |

**最强信号**：action_diversity（AUROC=0.6856）和 ep_mean_verbalized（0.6808），均超过 0.6 路由阈值但低于 classifieds B0 的 0.74-0.76 水平。

**与 classifieds B0 对比**：Shopping DOM 单模式的信号区分力明显弱于 classifieds 三模式合并数据（classifieds ep_mean_verbalized=0.755, action_diversity=0.741）。原因：(1) 单模式样本量更小（465 vs 702）；(2) 类别不平衡更严重（adjusted success 仅 19 个，n_success=19 vs n_failure=446）。

### 5.3 Mann-Whitney 检验

| 信号 | U | p 值 | rank_biserial | 显著性 |
|------|---|------|---------------|--------|
| ep_mean_verbalized | 5769 | 0.0076 | -0.3616 | ** |
| action_diversity | 5810 | 0.0060 | -0.3711 | ** |
| action_unique_types | 5401 | 0.0333 | -0.2747 | * |
| max_repeat_streak | 3092 | 0.0353 | +0.2702 | * |
| ep_min_verbalized | 5381 | 0.0355 | -0.2700 | * |

成功 episode 的 verbalized confidence 显著高于失败（p=0.008），action_diversity 显著低于失败（p=0.006，成功 episode 动作类型更聚焦）。

### 5.4 Routing Readiness 判定

| 维度 | 结果 |
|------|------|
| Token-level 区分力 | 不可用（API 模式） |
| Entropy 区分力 | 不可用 |
| Verbalized 区分力 | 可用（AUROC=0.6808） |
| Behavioral 区分力 | 可用（AUROC=0.6856, action_diversity） |
| 信号覆盖率 | 充足（100%） |
| 校准 | 不可评估 |
| 模式不变性 | 不可评估（单模式） |
| **Overall** | **可用于路由（overall_usable=true）** |

> 信号区分力达标但 headroom 有限（AUROC <0.7），实际路由收益需多模式数据验证。

---

## 6. 共性脚手架缺陷

### 6.1 N/A 任务 False Positive（30/31）

31 个 N/A reference tasks 中 30 个被判为 FP。机制与 classifieds 相同：Agent prompt 无 N/A 出口 + evaluator ua_match bug，空答案或循环到截断后被误判为成功。

### 6.2 Visual Tasks 在 DOM 模式下的结构性劣势

466 个 task 中 269 个（57.7%）涉及视觉属性。DOM 模式下这些 task 依赖 AXTree 文本中的间接线索（如商品名称含颜色词），但大量视觉属性（颜色、形状、布局位置）无法从文本获取。73/213（34.3%）的失败直接归因于视觉信息缺失。

### 6.3 搜索框/订阅框混淆

Shopping 站点页面底部有 Newsletter 订阅框，AXTree 中与搜索栏类型相同（input text）。Agent 在长 AXTree 滚动后容易将搜索词输入订阅框（37/213 失败存在 element_id 问题，其中部分为此类混淆）。

### 6.4 搜索策略局限

- 搜索关键词过于具体：将多个约束全部拼入搜索词（如 "red ps4 controller under $200"），Magento 搜索引擎无法正确处理
- 极少使用价格排序/筛选器：DOM 模式下 B0 对 Magento 的 Sort By 和 Price Filter 使用率较低
- 翻页行为有限：相比 classifieds B0 DOM 的 33+ task 翻页，shopping 站点翻页频率更低

---

## 与 Classifieds B0 DOM 的初步对比

| 维度 | Shopping DOM | Classifieds DOM | 说明 |
|------|-------------|----------------|------|
| Raw SR | 11.80% (55/466) | 14.96% (35/234) | Shopping 任务量更大但 SR 更低 |
| Adjusted SR (reason diag) | 6.24% (29/465) | 8.48% (19/224) | 两站调整后差距缩小 |
| 平均步数 | 10.20 | 11.52 | Shopping 略快完成（或更快放弃） |
| 平均成本 | $0.0424/ep | $0.0425/ep | 几乎相同 |
| P95 延迟 | 16,695 ms | 37,513 ms | Shopping 延迟更低（AXTree 更短） |
| No-op rate | 18.6% | 14.4% | Shopping 无效操作更多 |
| 视觉任务占比 | 57.7% (269/466) | ~50% | Shopping 视觉任务更多，DOM 劣势更大 |
| 最强路由信号 | action_diversity 0.686 | ep_mean_verbalized 0.782 | Shopping 信号弱于 classifieds |

> 注：Classifieds 数据来自三模式完整实验，部分指标不完全可比。

---

## 方法论说明

- **Adjusted labels 多套方案**：
  - `analysis_summary`：从分母移除 N/A task（466→435），从分子移除对应 FP → 4.37%
  - `condition_metrics`（Bootstrap CI 使用）：adjusted SR = 6.44%，保持分母 466
  - `reason_diagnostics`：从分子移除 FP（55→29），分母 465 → 6.24%
  - 三套方案差异源于调整策略不同，本报告各节标注数据来源
- **单 condition 限制**：无法进行 McNemar（成对 SR 检验）、Wilcoxon（成对效率检验）、cross_representation（跨模式交叉分析）
- **Digest 覆盖**：213 failure episodes（不含 55 个 success），覆盖全部失败案例
- **Confidence 信号**：B0 为 API 模式，仅 verbalized confidence + behavioral signals（无 token-level logprobs）
- **数据时间**：本报告基于 2026-04-23 分析管线输出

---

*更新时间：2026-04-23*
*数据来源：B0_3mode_shopping_20260421 analysis/ 目录*
*DOM 模式详情：B0_DOM_digest.md*
