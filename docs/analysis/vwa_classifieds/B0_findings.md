# B0 Classifieds 三模式实验报告

> Run: `B0_3mode_classifieds_20260413`
> 模型: Qwen3-VL-235B-A22B（proxy API，MoE 22B 活跃参数）
> 站点: Classifieds (OSClass), 234 tasks x 3 modes (DOM / SoM / Vision)
> 分析管线默认使用 **adjusted labels**（扣除 N/A FP + eval FP）
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**
> 各模式专有分析见 `B0_DOM_digest.md` / `B0_SOM_digest.md` / `B0_Vision_digest.md`
> B0 vs B1 跨模型对比见 `B0_B1_findings.md`

---

## 1. 成功率

### 1.1 主指标（Adjusted SR）

| 模式 | Raw SR | Adjusted SR | 成功数（adjusted） | FP 分解 |
|------|--------|------------|------------------|---------|
| DOM | 14.96% (35/234) | **12.95%** (29/224) | 29 | N/A FP: 6, eval FP: 0 |
| **SoM** | **23.08%** (54/234) | **20.54%** (46/224) | 46 | N/A FP: 8, eval FP: 0 |
| Vision | 15.81% (37/234) | 12.05% (27/224) | 27 | N/A FP: 10, eval FP: 0 |

**三模式排序：SoM (20.54%) > DOM (12.95%) ~ Vision (12.05%)**

SoM 显著领先其他两种模式。DOM 与 Vision 差异不显著。

> §95 变更：DOM adjusted SR 从此前的 8.48%（含 visual_fp 扣除）上升至 12.95%。DOM 不再是最弱模式——与 Vision（12.05%）几乎持平。

### 1.2 FP 机制说明

**N/A FP**（24/30）：Agent prompt 无 N/A 出口（Rule 4: "NEVER give up"）→ 空答案或循环到截断 → 评测器误判。10 个 N/A reference tasks 从分母移除（234→224），对应的 FP success 从分子移除。DOM 6/10, SoM 8/10, Vision 10/10 被标记为 na_fp。

**Eval FP**：三模式均为 0。

**Visual FP**：**§95 中废弃**。此前 DOM 有 14 个 visual_fp（其中 4 个与 na_fp 重叠），现在这些 DOM 成功保留为有效——DOM 虽无截图，但部分视觉任务可通过 AXTree 文本推理间接完成（如标题含颜色关键词、url_match 不验证视觉理解）。

### 1.3 统计显著性（McNemar 精确检验）

| 对比 | 不一致对 (A-only / B-only) | p 值 | 显著性 |
|------|--------------------------|------|--------|
| SoM vs DOM | 29 / 12 | **0.0115** | ★ |
| Vision vs DOM | 17 / 21 | 0.627 | — (n.s.) |
| SoM vs Vision | 31 / 18 | 0.085 | — (n.s.) |

**SoM 显著优于 DOM**（p=0.012）。Vision 与 DOM 差异不显著（p=0.627）。SoM vs Vision 差异 marginal（p=0.085）。

> §95 变更：SoM vs DOM p 值从 1.4e-5 上升至 0.012（仍显著但幅度缩小）。Vision vs DOM 从 p=0.016（显著）变为 p=0.627（不显著）。

### 1.4 Bootstrap 95% CI（cross_rep adjusted labels, /234 分母）

| 模式 | SR | CI 下界 | CI 上界 |
|------|-----|---------|---------|
| SoM | 21.37% | 16.24% | 26.92% |
| Vision | 15.81% | 11.11% | 20.51% |
| DOM | 14.10% | 9.83% | 18.80% |

> 注：Bootstrap CI 基于 cross_rep adjusted labels（/234 分母）。DOM CI [9.83%, 18.80%] 与 Vision CI [11.11%, 20.51%] 大幅重叠，与 McNemar 不显著一致。

---

## 2. 效率指标

### 2.1 全 episode 效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| 平均成本 ($/ep) | 0.0427 | 0.0415 | **0.0248** |
| 平均步数 | 11.56 | 8.60 | **7.85** |
| P95 延迟 (ms) | **37,513** | 74,004 | 44,984 |

**Vision 成本最低**（$0.0248/ep）：无 AXTree 文字、无 SoM 标注图像，token 量最少。

**SoM 和 DOM 成本接近**（$0.0415 vs $0.0427，Wilcoxon 不显著 p=0.496）：SoM 步数少（8.60 vs 11.56）但每步成本更高（图文混合），两者抵消后总成本无显著差异。

**DOM 延迟最低**（37,513ms P95）：纯文本请求在 proxy API 上最快。

### 2.2 Wilcoxon 效率对比

| 对比 | 指标 | p 值 | 方向 |
|------|------|------|------|
| Vision vs SoM | total_cost | **2.9e-9** | Vision 更便宜 ★★★ |
| Vision vs SoM | p95_step_latency | **1.9e-14** | Vision 延迟更低 ★★★ |
| Vision vs DOM | total_cost | **4.5e-12** | Vision 更便宜 ★★★ |
| Vision vs DOM | p95_step_latency | **3.1e-5** | Vision 延迟更低 ★★★ |
| SoM vs DOM | total_cost | 0.496 | **无显著差异** |
| SoM vs DOM | p95_step_latency | **1.4e-11** | SoM 延迟更高 ★★★ |

**成本排序**：Vision << SoM ~ DOM。**延迟排序**：DOM < Vision < SoM。

### 2.3 成本分解

| 模式 | 总成本 | 有效成本 | no-op 成本 | 循环成本 |
|------|--------|---------|-----------|---------|
| DOM | $0.0427 | $0.0300 | $0.0056 | $0.0074 |
| SoM | $0.0415 | $0.0340 | $0.0027 | $0.0049 |
| Vision | $0.0248 | $0.0128 | $0.0071 | $0.0055 |

Vision no-op 成本占比最高（28.7%），反映坐标 misclick 的浪费。SoM 有效成本占比最高（81.9%），最高效利用 token。

### 2.4 Action 执行效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| click_fail_rate (mean) | **12.2%** | 7.0% | 45.9% |
| type_fail_rate (mean) | 6.8% | **3.6%** | 16.7% |
| pixel_coordinate_leak | 0% | 0% | **34.6%** |

Vision click_fail_rate 达 45.9%（近半数点击失败），解释其 no-op 成本占比高。SoM click_fail_rate 最低（7.0%），element_id 定位的精度优于坐标。DOM 居中（12.2%），AXTree element_id 偶有定位偏差。Vision 34.6% 的 episode 出现 pixel_coordinate_leak。

---

## 3. 失败模式对比

### 3.1 三模式失败原因对比

| 失败原因 | DOM | SoM | Vision |
|----------|-----|-----|--------|
| fail_no_progress | 62 (26.5%) | 19 (8.1%) | **93 (39.7%)** |
| fail_finish_wrong_url_not_found | 44 (18.8%) | **57 (24.4%)** | 21 (9.0%) |
| fail_finish_eval_mismatch | 32 (13.7%) | **32 (13.7%)** | 18 (7.7%) |
| fail_early_finish | 10 (4.3%) | **32 (13.7%)** | 19 (8.1%) |
| fail_incomplete_or_stuck | 7 (3.0%) | 2 (0.9%) | **21 (9.0%)** |
| fail_max_steps_target_unreachable | 14 (6.0%) | **15 (6.4%)** | 5 (2.1%) |
| fail_max_steps_click_back_loop | **11 (4.7%)** | 4 (1.7%) | — |
| fail_finish_claim_missing | 6 (2.6%) | **9 (3.8%)** | 7 (3.0%) |
| fail_finish_empty_answer | 7 (3.0%) | 4 (1.7%) | **9 (3.8%)** |
| fail_max_steps | 2 (0.9%) | **5 (2.1%)** | 3 (1.3%) |
| fail_max_steps_search_repeat | 3 (1.3%) | — | 1 (0.4%) |

**三模式各有主导失败原因**：
- **DOM**：`no_progress`（26.5%）+ `wrong_url`（18.8%）
- **SoM**：`wrong_url`（24.4%）+ `early_finish`（13.7%）
- **Vision**：`no_progress`（39.7%）

### 3.2 脚手架 vs 模型归因

| 模式 | 脚手架/表征缺陷 | 模型能力问题 | 主要特征 |
|------|--------------|------------|---------|
| DOM | 中（信息瓶颈，但无 visual_fp 后缓解） | 低（235B 文字推理强） | 信息瓶颈 |
| SoM | 低（截图可见） | 中（early_finish + parse_error） | 过度自信 |
| Vision | 低（截图可见） | 高（坐标精度、no_progress 39.7%） | 执行失败主导 |

### 3.3 Reason Stability（跨模式失败一致性）

同一 task 在三模式下是否落入相同 failure bucket：
- **Mean stability**: 0.412（1.0=三模式完全一致，0.0=全不同）
- **完全一致（stability=1.0）**: 34/234 tasks (14.5%)
- **高度不一致（stability<0.5）**: 75/234 tasks (32.1%)

大部分 task 在不同模式下的失败原因不同，进一步支持路由的价值——不同模式各有擅长领域。

---

## 4. 跨模式交叉分析

> 三模式完整 cross_representation 分析（DOM + SoM + Vision），234 tasks common set。
> 注：以下数据使用 cross_rep 管线的 adjusted labels（na_fp + eval_fp，/234 分母）。

### 4.1 Oracle 分析

| 指标 | Raw | Adjusted |
|------|-----|----------|
| Oracle ceiling | 32.05% (75 tasks) | **31.20%** (73 tasks) |
| 最佳单模式 (SoM) | 23.08% | 21.37% |
| Routing headroom | 8.97pp | **9.83pp** |
| Oracle 选择分布 (raw) | SoM:27, Vision:27, DOM:21 | — |
| Oracle 选择分布 (adj) | **SoM:25, Vision:25, DOM:23** | — |

**Adjusted routing headroom 9.83pp**：理论最优路由可将 SR 从 21.37% 提升到 31.20%。三模式在 oracle 中贡献均衡——Vision 25/73（34.2%）、DOM 23/73（31.5%）、SoM 25/73（34.2%）。

> §95 变更：DOM oracle 贡献从 13/68（19.1%）大幅上升至 23/73（31.5%），因为此前被 visual_fp 扣除的 DOM 成功现在保留。三模式 oracle 贡献趋于均衡。

### 4.2 Oracle 选择按任务类型（raw）

| 任务类型 | DOM | SoM | Vision | 总计 |
|---------|-----|-----|--------|------|
| page_reading | 9 | 12 | **14** | 35 |
| single_navigation | 11 | **14** | 12 | 37 |
| action_on_item | — | — | 1 | 1 |
| date_count | 1 | — | — | 1 |
| grid_position | — | 1 | — | 1 |

### 4.3 集合分析（Adjusted，/234 分母）

| 集合 | 数量 | 占比 | 任务类型分布 |
|------|------|------|-------------|
| all_fail | 161 | 68.8% | single_nav:111, page_reading:29, others:21 |
| only_som | 21 | 9.0% | single_nav:12, page_reading:8, grid:1 |
| dom_and_som (not vision) | 12 | 5.1% | page_reading:6, single_nav:6 |
| **only_vision** | **11** | **4.7%** | single_nav:7, page_reading:4 |
| all_success | 9 | 3.9% | page_reading:4, single_nav:5 |
| **only_dom** | **8** | **3.4%** | page_reading:4, single_nav:3, date:1 |
| som_and_vision (not dom) | 8 | 3.4% | page_reading:5, single_nav:3 |
| dom_and_vision (not som) | 4 | 1.7% | page_reading:2, single_nav:1, action:1 |

**关键发现**：
- **DOM 独占 8 个 task**（3.4%），远高于旧数据的 4 个——visual_fp 废弃后 DOM 独占成功增加
- **Vision 独占 11 个**（4.7%）
- **SoM 独占 21 个**（9.0%）仍是三模式中最多
- **All fail 68.8%** 的 task 三模式全败
- **三模式 oracle 贡献趋于均衡**，routing 价值最大化

### 4.4 Task type × mode SR 矩阵（Adjusted）

| Task type | N | DOM SR | SoM SR | Vision SR |
|-----------|---|--------|--------|-----------|
| single_navigation | 148 | 10.1% | **17.6%** | 10.8% |
| page_reading | 62 | 25.8% | **37.1%** | 24.2% |
| action_on_item | 9 | 11.1% | 0% | 11.1% |
| collection | 7 | 0% | 0% | 0% |
| grid_position | 5 | 0% | **20.0%** | 0% |
| date_count | 3 | **33.3%** | 0% | 0% |

SoM 在主要类型（single_nav/page_reading）上均领先。DOM 在 date_count（需精确文本计数）上独占。collection 类型三模式全败。

### 4.5 按失败原因的成本分解

高成本失败模式（avg cost per episode）：
- `fail_max_steps`: $0.140（30 步耗尽，成本最高）
- `fail_max_steps_click_back_loop`: ~$0.13
- `fail_max_steps_search_repeat`: ~$0.12

路由若能提前识别这些高成本失败模式并切换到低成本 Vision 模式，可节省显著成本。

---

## 5. 路由方向分析

### 5.1 Headroom 评估

| 路由场景 | Adjusted headroom | 备注 |
|---------|------------------|------|
| 三模式 oracle | **9.83pp** | 完整三模式 cross_representation 数据 |

### 5.2 各模式路由角色

**SoM（25/73 oracle, 34.2%）**— 主力模式：
- adjusted SR 最高（20.54%），独占 21 个 task
- single_navigation 类型主导

**Vision（25/73 oracle, 34.2%）**— 成本优化 + 独占贡献：
- 成本最低（$0.0248/ep），独占 11 个 task
- page_reading 类型主导

**DOM（23/73 oracle, 31.5%）**— 文本推理 + 延迟优势：
- P95 延迟最低（37,513ms vs SoM 74,004ms）
- 独占 8 个 task，AXTree 文字推理在特定任务上优于视觉扫描
- §95 后路由价值大幅提升（oracle 占比从 19.1% 升至 31.5%）

### 5.3 与 B1 路由格局对比

| 维度 | B1 | B0 |
|------|----|----|
| 最优单模式 SR | SoM 13.84% | **SoM 20.54%** |
| Oracle ceiling (adj) | 21.37% | **31.20%** |
| Routing headroom | 8.12pp | **9.83pp** |
| DOM oracle 占比 | 32.0% | **31.5%** |
| Vision oracle 占比 | 34.0% | **34.2%** |
| 主路由方向 | **三模式均有价值** | **三模式均有价值** |

### 5.4 State Change × Outcome

| 模式 | 成功 page_change_rate | 失败 page_change_rate | 成功 avg_steps | 失败 avg_steps |
|------|---------------------|---------------------|---------------|---------------|
| DOM | 0.758 | 0.749 | 7.4 | 12.3 |
| SoM | 0.726 | 0.774 | 4.6 | 9.7 |
| Vision | 0.666 | 0.611 | 5.7 | 8.2 |

成功 episode 步数显著少于失败（DOM: 7.4 vs 12.3），但 page_change_rate 在成功与失败间无显著差异。成功取决于**精准的少步操作**，而非页面变化频率。

### 5.5 Temporal SR 趋势

| 模式 | Q1 (earliest) | Q5 (latest) | 趋势 |
|------|--------------|-------------|------|
| DOM | 15.2% | 12.0% | ↓ 轻微下降 |
| SoM | 17.4% | 18.0% | → 稳定 |
| Vision | 10.9% | 16.0% | ↑ 上升 |

DOM 呈现轻微的 temporal degradation（C23 WARN），可能因晚期 task 难度更高。SoM 和 Vision 无显著时序退化。

---

## 6. 共性脚手架缺陷

### 6.1 `<select>` 下拉菜单不可用（VWA 框架级）

同 B1。B0 额外问题：**capability-environment gap 更严重**——235B 模型更精准识别 `<select>` 为正确入口，反而更执着地反复点击同一元素，cycle detection 更快截断。

### 6.2 N/A 任务 False Positive（24/30）

三模式各 6/10、8/10、10/10 误判。机制同 B1：Agent prompt 无 N/A 出口 + evaluator ua_match bug。

### 6.3 confirm 弹窗不可交互

Delete 操作全部失败，三模式均受 VWA Playwright 限制。

### 6.4 搜索策略局限

- 极少翻页：B0 DOM 中翻页能力改善（33+ task），但 SoM/Vision 仍较少翻页
- 搜索关键词过于具体
- 235B 的策略规划稍有改善（价格筛选、表单聚焦等），见 B0_DOM_digest

---

## 7. 路由信号分析（Confidence & Behavioral Signals）

> B0 为 API 调用（proxy），无 token-level logprobs；仅有 verbalized confidence 和 behavioral signals。

### 7.1 AUROC 区分力（全局，702 episodes）

| 信号 | 类型 | AUROC | 95% CI |
|------|------|-------|--------|
| **ep_mean_verbalized** | verbalized | **0.756** | [0.709, 0.802] |
| **action_diversity** | behavioral | **0.744** | [0.696, 0.789] |
| url_revisit_max | behavioral | 0.710 | [0.661, 0.757] |
| max_repeat_streak | behavioral | 0.685 | [0.644, 0.724] |
| url_revisit_count | behavioral | 0.682 | [0.630, 0.731] |
| ep_min_verbalized | verbalized | 0.642 | [0.590, 0.694] |
| action_unique_types | behavioral | 0.581 | [0.523, 0.636] |
| url_unique_count | behavioral | 0.488 | [0.434, 0.541] |

### 7.2 跨模式 AUROC 稳定性

| 信号 | DOM | SoM | Vision | 模式间差异 |
|------|-----|-----|--------|-----------|
| ep_mean_verbalized | 0.782 | 0.709 | **0.763** | 0.073 |
| action_diversity | 0.747 | 0.697 | **0.747** | 0.050 |
| url_revisit_max | 0.694 | 0.683 | **0.707** | 0.024 |
| max_repeat_streak | 0.683 | 0.593 | **0.773** | 0.180 |

**ep_mean_verbalized** 和 **action_diversity** 跨模式相对稳定（模式间最大差异 <0.08）。

### 7.3 Routing Readiness 判定

| 维度 | 结果 |
|------|------|
| Token-level 区分力 | 不可用（API 模式） |
| Verbalized 区分力 | 有（AUROC=0.756） |
| Behavioral 区分力 | 有（AUROC=0.744, action_diversity） |
| 信号覆盖率 | 100% |
| **Overall: 可用于路由** | **是** |

---

## 方法论说明

- **Adjusted labels**：仅扣除 N/A FP + eval FP。visual_fp 已在 §95 中废弃
- **两套 adjusted SR**：analysis_summary 使用 /224 分母（移除 N/A task）；cross_representation 使用 /234 分母（FP 标为失败）
- **McNemar / Bootstrap CI**：使用 cross_rep adjusted labels
- **统计检验**：McNemar 精确检验（SR）+ Wilcoxon 有符号秩检验（成本/延迟）+ Bootstrap CI
- **数据时间**：本报告基于 2026-04-25 最新分析数据（§95 FP 重构后）

---

*更新时间：2026-04-25（§95 FP 重构：废弃 visual_fp，DOM adjusted SR 从 8.48% 升至 12.95%；三模式 oracle 贡献趋于均衡；McNemar 检验更新）*
*数据来源：B0_3mode_classifieds_20260413 analysis/ 目录*
*各模式详情：B0_DOM_digest.md / B0_SOM_digest.md / B0_Vision_digest.md*
