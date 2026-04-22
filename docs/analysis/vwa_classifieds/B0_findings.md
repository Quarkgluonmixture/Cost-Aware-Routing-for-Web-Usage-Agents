# B0 Classifieds 三模式实验报告

> Run: `B0_3mode_classifieds_20260413`
> 模型: Qwen3-VL-235B-A22B（proxy API，MoE 22B 活跃参数）
> 站点: Classifieds (OSClass), 234 tasks × 3 modes (DOM / SoM / Vision)
> 分析管线默认使用 **adjusted labels**（扣除 N/A FP + visual FP）
> 各模式专有分析见 `B0_DOM_digest.md` / `B0_SOM_digest.md` / `B0_Vision_digest.md`
> B0 vs B1 跨模型对比见 `B0_B1_findings.md`
> **本版数据含 parse_error 修复后重跑结果及最新 FP 检测（2026-04-21）**

---

## 1. 成功率

### 1.1 主指标（Adjusted SR）

| 模式 | Raw SR | Adjusted SR | 成功数（adjusted） | FP 分解 |
|------|--------|------------|------------------|---------|
| DOM | 14.96% (35/234) | 8.48% (19/224) | 19 | N/A FP: 6, Visual FP: 14 (overlap 4) |
| **SoM** | **23.50%** (55/234) | **20.98%** (47/224) | 47 | N/A FP: 8 |
| Vision | 15.81% (37/234) | 12.05% (27/224) | 27 | N/A FP: 10 |

**三模式排序：SoM (20.98%) >> Vision (12.05%) > DOM (8.48%)**

SoM 显著领先其他两种模式。Vision 与 DOM 的差距现在具有统计显著性（§1.3）。

### 1.2 FP 机制说明

**N/A FP**（24/30）：Agent prompt 无 N/A 出口（Rule 4: "NEVER give up"）→ 空答案或循环到截断 → 评测器误判。10 个 N/A reference tasks 从分母移除（234→224），对应的 FP success 从分子移除。DOM 6/10, SoM 8/10, Vision 10/10 被标记为 na_fp。

**Visual FP（DOM 14 个）**：DOM 无截图，14 个 visual task 碰巧通过 url_match/string_match（kwd_only 过滤确认），其中 4 个与 N/A FP 重叠，净新增 10 个。SoM/Vision 有截图，无 visual FP。

**Eval FP**：三模式均为 0（无 eval_fp 检测到）。

### 1.3 统计显著性（McNemar 精确检验）

| 对比 | 不一致对 (A-only / B-only) | p 值 | 显著性 |
|------|--------------------------|------|--------|
| SoM vs DOM | 34 / 6 | **8.4e-6** | ★★★ |
| Vision vs DOM | 22 / 8 | **0.016** | ★ |
| SoM vs Vision | 31 / 17 | 0.059 | — (marginal) |

**SoM 显著优于 DOM**（p<0.001）。**Vision 显著优于 DOM**（p=0.016）。SoM vs Vision 差异为边缘水平（p=0.059），不满足 α=0.05 显著性阈值。

> 注：McNemar 使用 condition_metrics 管线的 adjusted labels（含 FP 检测），与 analysis_summary 的调整方案存在微小差异。

### 1.4 Bootstrap 95% CI

| 模式 | SR | CI 下界 | CI 上界 |
|------|-----|---------|---------|
| SoM | 21.79% | 16.67% | 27.35% |
| Vision | 15.81% | 11.11% | 20.51% |
| DOM | 9.83% | 6.40% | 13.68% |

> 注：Bootstrap CI 基于 condition_metrics 管线 SR 计算。SoM CI 下界（16.67%）低于 Vision CI 上界（20.51%），与 McNemar 边缘显著一致。

---

## 2. 效率指标

### 2.1 全 episode 效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| 平均成本 ($/ep) | 0.0425 | 0.0417 | **0.0248** |
| 平均步数 | 11.52 | 8.62 | **7.85** |
| P95 延迟 (ms) | **37,513** | 75,932 | 46,361 |
| 平均输入成本 | 0.0362 | 0.0358 | **0.0202** |
| 平均输出成本 | 0.0063 | 0.0059 | **0.0046** |
| No-op rate | 14.4% | **6.0%** | 30.0% |
| Page unchanged rate | **25.0%** | 23.9% | 37.9% |

**Vision 成本最低**（$0.0248/ep）：无 AXTree 文字、无 SoM 标注图像，token 量最少。

**SoM 和 DOM 成本接近**（$0.0417 vs $0.0425，Wilcoxon 不显著 p=0.522）：SoM 步数少（8.62 vs 11.52）但每步成本更高（图文混合），两者抵消后总成本无显著差异。

**DOM 延迟最低**（37,513ms P95）：纯文本请求在 proxy API 上最快，但仍比旧分析（10,537ms）高——数据更新后包含了更多高延迟 episode。

### 2.2 Wilcoxon 效率对比

| 对比 | 指标 | p 值 | 方向 |
|------|------|------|------|
| Vision vs SoM | total_cost | **2.8e-9** | Vision 更便宜 ★★★ |
| Vision vs SoM | p95_step_latency | **3.0e-14** | Vision 延迟更低 ★★★ |
| Vision vs DOM | total_cost | **5.5e-12** | Vision 更便宜 ★★★ |
| Vision vs DOM | p95_step_latency | **1.5e-4** | Vision 延迟更低 ★★★ |
| SoM vs DOM | total_cost | 0.522 | **无显著差异** |
| SoM vs DOM | p95_step_latency | **5.4e-12** | SoM 延迟更高 ★★★ |

**成本排序**：Vision << SoM ≈ DOM（Vision 显著更便宜；SoM 与 DOM 成本无显著差异）。
**延迟排序**：DOM < Vision < SoM（DOM 最快，SoM 最慢，均显著）。

### 2.3 早停触发分布

| 触发原因 | DOM | SoM | Vision |
|---------|-----|-----|--------|
| action_failed | 301 | 108 | **430** |
| page_unchanged_streak | 131 | 36 | **239** |
| no_progress_streak | 131 | 36 | **239** |
| dom_size_exceeds_threshold | 5 | — | — |
| text_length_high | 4 | — | — |

Vision 所有早停维度均最高——坐标点击的固有不稳定性导致大量无效动作积累。SoM 的早停触发远低于其他两种模式，说明 SoM 标注帮助 agent 更精准地定位交互元素。

---

## 3. 失败模式对比

### 3.1 三模式失败原因对比

| 失败原因 | DOM | SoM | Vision |
|----------|-----|-----|--------|
| fail_no_progress | 62 (26.5%) | 19 (8.1%) | **92 (39.3%)** |
| fail_finish_wrong_url_not_found | 43 (18.4%) | **54 (23.1%)** | 20 (8.5%) |
| fail_finish_eval_mismatch | 32 (13.7%) | **31 (13.2%)** | 18 (7.7%) |
| fail_early_finish | 10 (4.3%) | **30 (12.8%)** | 19 (8.1%) |
| fail_incomplete_or_stuck | 7 (3.0%) | 2 (0.9%) | **21 (9.0%)** |
| fail_max_steps_target_unreachable | 14 (6.0%) | **15 (6.4%)** | 5 (2.1%) |
| fail_max_steps_click_back_loop | **11 (4.7%)** | 4 (1.7%) | — |
| fail_finish_claim_missing | 6 (2.6%) | **9 (3.8%)** | 7 (3.0%) |
| fail_finish_empty_answer | 7 (3.0%) | 3 (1.3%) | **9 (3.8%)** |
| fail_parse_error | 1 (0.4%) | **6 (2.6%)** | 2 (0.9%) |
| fail_max_steps | 2 (0.9%) | **5 (2.1%)** | 3 (1.3%) |
| fail_max_steps_search_repeat | 3 (1.3%) | — | 1 (0.4%) |

**三模式各有主导失败原因**：
- **DOM**：`no_progress`（26.5%）+ `wrong_url`（18.4%）— 无截图辅助，agent 在长 AXTree 中反复选错元素且无法得到视觉反馈纠正
- **SoM**：`wrong_url`（23.1%）+ `early_finish`（12.8%）— SoM 截图提供"视觉确认"锚点，agent 更果断但也更容易在相似页面上过早 finish
- **Vision**：`no_progress`（39.3%）— 纯坐标点击的固有不稳定性，大量无效操作累积

### 3.2 脚手架 vs 模型归因

| 模式 | 脚手架/表征缺陷 | 模型能力问题 | 主要特征 |
|------|--------------|------------|---------|
| DOM | 高（visual 不可达、visual FP 14 个） | 低（235B 文字推理强） | 信息瓶颈 |
| SoM | 低（截图可见） | 中（early_finish + parse_error） | 过度自信 + 格式问题 |
| Vision | 低（截图可见） | 高（坐标精度、no_progress 39.3%） | 执行失败主导 |

---

## 4. 跨模式交叉分析

> 三模式完整 cross_representation 分析（DOM + SoM + Vision），234 tasks common set。
> 注：以下数据使用 cross_rep 管线的三重 adjusted labels（na_fp + visual_fp + eval_fp，/234 分母）。

### 4.1 Oracle 分析

| 指标 | Raw | Adjusted |
|------|-----|----------|
| Oracle ceiling | 32.05% (75 tasks) | **29.06%** (68 tasks) |
| 最佳单模式 (SoM) | 22.22% | 20.51% |
| Routing headroom | 9.83pp | **8.55pp** |
| Oracle 选择分布 | SoM:26, Vision:28, DOM:21 | **SoM:29, Vision:26, DOM:13** |

**Adjusted routing headroom 8.55pp**：理论最优路由可将 SR 从 20.51% 提升到 29.06%。三模式在 oracle 中均有显著贡献——Vision 26/68（38.2%）、DOM 13/68（19.1%）。

### 4.2 Oracle 选择按任务类型（raw）

| 任务类型 | DOM | SoM | Vision | 总计 |
|---------|-----|-----|--------|------|
| page_reading | 9 | 12 | **14** | 35 |
| single_navigation | 11 | **13** | 13 | 37 |
| action_on_item | — | — | 1 | 1 |
| date_count | 1 | — | — | 1 |
| grid_position | — | 1 | — | 1 |

**page_reading 类型 Vision 主导**（14/35 = 40%），**single_navigation 类型三模式均衡**（SoM 和 Vision 并列 13）。

### 4.3 任务类型成功率（A5）

| 任务类型 | n | DOM raw | DOM adj | SoM raw | SoM adj | Vision raw | Vision adj |
|---------|---|---------|---------|---------|---------|------------|-----------|
| single_navigation | 148 | 10.1% | 6.8% | 17.3% | **17.3%** | 14.2% | 12.5% |
| page_reading | 62 | 29.0% | 16.1% | 44.3% | **37.7%** | 29.0% | 24.2% |
| action_on_item | 9 | 11.1% | 0% | 11.1% | 11.1% | 11.1% | 11.1% |
| collection | 7 | 0% | 0% | 0% | 0% | 0% | 0% |
| grid_position | 5 | 0% | 0% | 20% | **20%** | 0% | 0% |
| date_count | 3 | 33.3% | 0% | 0% | 0% | 0% | 0% |

SoM 在 page_reading 类型 adjusted SR 最高（37.7%），Vision 次之（24.2%）。collection 类型全败。

### 4.4 集合分析（Adjusted，三模式）

| 集合 | 数量 | 占比 | 任务类型分布 |
|------|------|------|-------------|
| all_fail | 166 | 70.9% | single_nav:112, page_reading:32, others:22 |
| only_som | 27 | 11.5% | single_nav:16, page_reading:10, grid:1 |
| **only_vision** | **15** | **6.4%** | single_nav:10, page_reading:5 |
| all_success | 9 | 3.9% | page_reading:4, single_nav:5 |
| dom_and_som (not vision) | 6 | 2.6% | page_reading:4, single_nav:2 |
| som_and_vision (not dom) | 6 | 2.6% | page_reading:5, action:1 |
| only_dom | 4 | 1.7% | single_nav:3, page_reading:1 |
| dom_and_vision (not som) | 1 | 0.4% | page_reading:1 |

**关键发现**：
- **Vision 独占 15 个 task**（6.4%），远高于 DOM 独占（4 个），说明 Vision 有大量不可替代的独特贡献
- **SoM 独占 27 个**（11.5%）仍是三模式中最多——SoM 是成功率和独占性最强的单模式
- **SoM+Vision 共享 6 个**，相比 SoM+DOM 共享 6 个，两对互补性相当
- **All fail 70.9%** 的 task 三模式全败，以 single_navigation 类型为主（112/166）

### 4.5 交集成功成本对比（9 tasks，三模式均成功）

| 模式 | 平均成本 | 中位成本 | 平均步数 | 最便宜次数 |
|------|---------|---------|---------|-----------|
| **DOM** | **$0.0134** | **$0.0130** | **3.56** | **5/9** |
| SoM | $0.0198 | $0.0208 | 4.22 | 0/9 |
| Vision | $0.0241 | $0.0154 | 6.89 | 4/9 |

**DOM 在交集任务上最便宜**（平均 $0.0134），步数最少（3.56）。Wilcoxon DOM vs SoM p=0.004 ★★（DOM 显著更便宜）。SoM vs Vision 无显著差异（p=0.734）。DOM 在这 9 个共同成功的简单任务上，纯文本请求 token 少、延迟低，成本优势明显。

---

## 5. 路由方向分析

### 5.1 Headroom 评估

| 路由场景 | Adjusted headroom | 备注 |
|---------|------------------|------|
| 三模式 oracle | **8.55pp** | 完整三模式 cross_representation 数据 |

### 5.2 各模式路由角色

**SoM（29/68 oracle, 42.6%）**— 主力模式：
- single_navigation 类型主导
- adjusted SR 最高（20.51%），独占 27 个 task

**Vision（26/68 oracle, 38.2%）**— 成本优化 + 独占贡献：
- page_reading 类型主导
- 成本最低（$0.0248/ep），独占 15 个 task（不可被其他模式替代）
- 与 SoM 共享 6 个 task 可降级到 Vision

**DOM（13/68 oracle, 19.1%）**— 延迟/精度补充：
- P95 延迟最低（37,513ms vs SoM 75,932ms）
- 交集任务上最便宜（$0.0134/ep，Wilcoxon p=0.004 vs SoM）
- 独占 4 个 task，AXTree 文字搜索精度在特定任务上优于视觉扫描

### 5.3 与 B1 路由格局对比

| 维度 | B1 | B0 |
|------|----|----|
| 最优单模式 SR | SoM 16.24% | **SoM 20.51%** |
| Oracle ceiling (adj) | 19.66% | **29.06%** |
| Routing headroom | 3.42pp | **8.55pp** |
| DOM 路由价值 | 极低（adjusted SR 0.85%） | **有价值**（8.55%, 13 oracle） |
| Vision oracle 占比 | — | **38.2%** |
| 主路由方向 | SoM ↔ Vision | **三模式均有价值** |

---

## 6. 共性脚手架缺陷

### 6.1 `<select>` 下拉菜单不可用（VWA 框架级）

同 B1。B0 额外问题：**capability-environment gap 更严重**——235B 模型更精准识别 `<select>` 为正确入口，反而更执着地反复点击同一元素，cycle detection 更快截断。见 B0_DOM_digest 详细分析。

### 6.2 N/A 任务 False Positive（24/30）

三模式各 6/10、8/10、10/10 误判。机制同 B1：Agent prompt 无 N/A 出口 + evaluator ua_match bug。

### 6.3 confirm 弹窗不可交互

Delete 操作全部失败，三模式均受 VWA Playwright 限制。

### 6.4 搜索策略局限

- 极少翻页：B0 DOM 中翻页能力改善（33+ task），但 SoM/Vision 仍较少翻页
- 搜索关键词过于具体：将约束全部拼入搜索词，应用宽泛词+筛选器策略
- 235B 的策略规划稍有改善（价格筛选、表单聚焦等），见 B0_DOM_digest

### 6.5 Prompt 对 select_option 与 click 的一刀切指令

三模式 prompt 均包含 **"Clicking a dropdown does NOT open it. Use select_option instead."**，阻止模型对下拉框使用 click。

**问题**：Classifieds (OsClass) 的 Sort By 是 **CSS 自定义下拉框**（非原生 `<select>`），click 其实可以展开它。Prompt 的指令对原生 `<select>` 是正确的（click 打开浏览器原生 UI，截图不可捕获），但对 CSS dropdown 反而有害——尤其在 Vision 模式下，模型看不到闭合下拉框的选项文字，被迫猜测 label（如猜 "Price: Low to High" 实际为 "Lower price first"），精确匹配失败后陷入循环。

**不修的理由**：(1) 实验已跑完，改 prompt 破坏对比一致性；(2) 即使允许 click 展开，部分任务所需选项不存在（如 "Oldest first"），收益有限；(3) Vision 模式无法从截图区分原生 `<select>` 与 CSS dropdown，修改 prompt 可能引入新问题

---

## 方法论说明

- **Adjusted labels**：N/A FP 优先（重叠时标为 na_fp），visual FP 次之（DOM kwd_only 过滤确认），eval_fp 第三层
- **两套 adjusted SR**：analysis_summary 使用 /224 分母（移除 N/A task）；cross_representation 使用 /234 分母（三重修正）。本报告主指标用 /224 分母。
- **McNemar / Bootstrap CI**：使用 condition_metrics 管线的 adjusted labels（与 analysis_summary 存在微小差异）
- **统计检验**：McNemar 精确检验（SR）+ Wilcoxon 有符号秩检验（成本/延迟）+ Bootstrap CI
- **数据时间**：本报告全部基于 2026-04-21 最新分析数据

---

*更新时间：2026-04-21*
*数据来源：B0_3mode_classifieds_20260413 analysis/ 目录（最新分析数据）*
*各模式详情：B0_DOM_digest.md / B0_SOM_digest.md / B0_Vision_digest.md*
