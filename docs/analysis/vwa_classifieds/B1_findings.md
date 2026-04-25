# B1 Classifieds 三模式实验报告

> Run: `B1_3mode_classifieds_20260413`
> 模型: Qwen3-VL-4B bf16
> 站点: Classifieds (OSClass), 234 tasks x 3 modes (DOM / SoM / Vision)
> 分析管线使用 **adjusted labels**（扣除 N/A FP + eval FP）
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**
> 各模式专有分析见 `B1_DOM_digest.md` / `B1_SOM_digest.md` / `B1_Vision_digest.md`
> 文档更新：2026-04-25

---

## 1. 成功率

### 1.1 主指标

| 模式 | Raw SR | Adjusted SR | 95% CI (Bootstrap, /234) | 成功数 (adj/raw) |
|------|--------|-------------|--------------------------|-----------------|
| DOM | 11.11% (26/234) | **7.59%** | [5.13%, 12.39%] | 17 / 26 |
| SoM | 17.52% (41/234) | **13.84%** | [9.40%, 18.38%] | 31 / 41 |
| Vision | 11.11% (26/234) | **7.14%** | [4.27%, 10.68%] | 16 / 26 |

- Adjusted SR 使用 N/A FP + eval FP 修正（§95），分母 224（扣除 10 个 N/A reference task）
- Bootstrap CI 使用 cross_rep adjusted labels（/234 分母）

> §95 变更：DOM adjusted SR 从 4.91%（含 visual_fp）上升至 7.59%（仅 N/A FP + eval FP）。DOM 不再是最弱模式，与 Vision（7.14%）几乎持平。

### 1.2 McNemar 检验（adjusted labels）

| 对比 | p-value | 显著? | 不一致对 (A-only / B-only) |
|------|---------|-------|--------------------------|
| SoM vs DOM | 0.065 | **否** (marginal) | 24 / 12 |
| Vision vs DOM | 0.728 | **否** | 15 / 18 |
| Vision vs SoM | 0.006 | **是** | 6 / 21 |

**排序：SoM > DOM ~ Vision**。SoM vs DOM 仅 marginal（p=0.065），DOM 与 Vision 无显著差异。

> §95 变更：SoM vs DOM 从 p=0.004（显著）变为 p=0.065（marginal）——因为 DOM adjusted SR 上升，差距缩小。

### 1.3 Raw SR 与 FP 分解

| 模式 | Raw SR | N/A FP | Eval FP | Adjusted SR |
|------|--------|--------|---------|------------|
| DOM | 11.11% (26) | 9 | 0 | 7.59% (17/224) |
| SoM | 17.52% (41) | 10 | 1 | 13.84% (31/224) |
| Vision | 11.11% (26) | 10 | 0 | 7.14% (16/224) |

- **N/A FP**（29 个）：10 个 N/A reference task x 3 modes（DOM 9/10, SoM 10/10, Vision 10/10）
- **Eval FP**（1 个，仅 SoM）：§95 简化后的 eval_fp 规则检出
- **Visual FP**：**§95 废弃**。此前 DOM 有 12 个 visual_fp（与 na_fp 重叠 6 个），现在保留为有效成功

---

## 2. 效率指标

### 2.1 全 episode 效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| 平均步数 | 13.83 | 9.90 | **6.73** |
| 平均成本 ($/ep) | 0.0399 | 0.0347 | **0.0133** |
| p95 步延迟 (s) | 43.2 | **30.2** | 64.5 |
| 平均能耗 (kWh) | 0.00522 | 0.00199 | **0.00194** |

### 2.2 成本统计检验（Wilcoxon signed-rank）

| 对比 | cost p | latency p |
|------|--------|-----------|
| Vision vs SoM | <1e-23 *** | 0.172 (n.s.) |
| Vision vs DOM | <1e-28 *** | 3.7e-6 *** |
| SoM vs DOM | 0.050 (边界) | <1e-28 *** |

- Vision 成本仅为 DOM 的 **33%**、SoM 的 **38%**
- SoM 与 DOM 成本差异仅边界显著（p=0.050）

### 2.3 成本分解

| 模式 | 总成本 | 有效成本 | no-op 成本 | 循环成本 |
|------|--------|---------|-----------|---------|
| DOM | $0.0399 | $0.0189 | $0.0060 | $0.0159 |
| SoM | $0.0347 | $0.0223 | $0.0072 | $0.0059 |
| Vision | $0.0133 | $0.0078 | $0.0050 | $0.0005 |

DOM 循环成本最高（$0.0159，占 39.8%），反映搜索/导航循环的严重浪费。Vision 循环成本最低（$0.0005），因为坐标失败直接触发早停而非循环。

### 2.4 Action 执行效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| click_fail_rate (mean) | 17.8% | **33.3%** | **45.7%** |
| type_fail_rate (mean) | **9.2%** | 6.5% | 5.8% |
| pixel_coordinate_leak | 0% | 0% | **20.9%** |

B1 的 click_fail_rate 全面高于 B0——4B 模型定位精度更差。SoM click_fail_rate（33.3%）远高于 B0 SoM（7.0%），这是 SoM 标注精度在小模型上衰减的体现。Vision 45.7% 的点击失败率解释了其 58.1% no_progress 失败。

### 2.5 路由信号校准质量（per-mode ECE）

| 模式 | Token ECE | Verbalized ECE | Verbalized Brier |
|------|-----------|---------------|------------------|
| DOM | 0.837 | 0.635 | — |
| SoM | 0.781 | 0.606 | — |
| Vision | 0.839 | 0.602 | — |

Token-level ECE 极高（~0.8），几乎无校准价值。Verbalized ECE（0.60-0.64）虽然也偏高，但跨模式差异较小（模式间最大差 0.033），说明 verbalized confidence **虽不精确，但模式间可比**——可作为路由决策的相对排序信号。

---

## 3. 跨模式交叉分析

### 3.1 Venn 集合（Adjusted, /234 分母）

| 区域 | 数量 | 占比 |
|------|------|------|
| 三模式均失败 | 184 | 78.6% |
| 仅 SoM | 15 | 6.4% |
| **仅 DOM** | **13** | **5.6%** |
| SoM + Vision（非 DOM） | 9 | 3.9% |
| 仅 Vision | 6 | 2.6% |
| DOM + SoM（非 Vision） | 5 | 2.1% |
| 三模式均成功 | 2 | 0.9% |

> §95 变更：DOM 独占成功从 7 个升至 13 个，是三模式中第二大独占集（仅次于 SoM 15 个）。DOM 路由价值大幅提升。

### 3.2 Oracle Ceiling

| 指标 | Raw | Adjusted (/234) |
|------|-----|------------------|
| Best single (SoM) | 17.52% | 13.25% |
| Oracle ceiling | 23.50% | **21.37%** |
| Routing headroom | 5.98pp | **8.12pp** |

Oracle 选择分布（adjusted）：DOM 16 (32.0%) / Vision 17 (34.0%) / SoM 17 (34.0%)。三模式贡献均衡。

> §95 变更：DOM oracle 从 7/44（15.9%）大幅上升至 16/50（32.0%），routing headroom 从 5.13pp 升至 8.12pp。

### 3.3 模式转换矩阵（Adjusted labels, 234 tasks）

**SoM vs DOM**：

| | DOM 成功 | DOM 失败 |
|---|---|---|
| SoM 成功 | 8 | 24 |
| SoM 失败 | 12 | 190 |

SoM 净优势 +12（24-12），但不显著（p=0.065）。

**Vision vs DOM**：

| | DOM 成功 | DOM 失败 |
|---|---|---|
| Vision 成功 | 2 | 15 |
| Vision 失败 | 18 | 199 |

净差异 -3（15-18），不显著（p=0.728）。

### 3.4 Task type × mode SR 矩阵（Adjusted）

| Task type | N | DOM SR | SoM SR | Vision SR |
|-----------|---|--------|--------|-----------|
| single_navigation | 148 | 6.1% | **12.2%** | 6.1% |
| page_reading | 62 | 14.5% | **21.0%** | 12.9% |
| action_on_item | 9 | **11.1%** | 0% | 0% |
| collection | 7 | 0% | 0% | 0% |
| grid_position | 5 | 0% | 0% | 0% |
| date_count | 3 | **33.3%** | 0% | 0% |

SoM 在主要类型上领先。DOM 在 date_count 和 action_on_item 上有独占成功——需精确文本交互的任务。grid_position 和 collection 三模式全败。

### 3.5 Reason Stability（跨模式失败一致性）

- **Mean stability**: 0.476（B0: 0.412）
- **完全一致**: 48/234 tasks (20.5%)
- **高度不一致（<0.5）**: 59/234 tasks (25.2%)

B1 的 reason stability 略高于 B0（0.476 vs 0.412），说明 4B 模型在不同模式下更倾向于以相同方式失败——能力上限更低时失败模式更趋一致。

### 3.6 按失败原因的成本分解

高成本失败模式：
- `fail_max_steps_target_unreachable`: $0.097/ep（30 步耗尽，45 episodes）
- `fail_max_steps_click_back_loop`: ~$0.09/ep
- `fail_no_progress`: $0.019/ep（单步成本低但数量最多 308 episodes，总成本 $5.82）

`fail_no_progress` 虽然单个 episode 成本低（快速失败），但其 308 episodes 的总量使其成为最大成本来源。

---

## 4. 失败模式

### 4.1 失败原因分布

| 失败原因 | DOM | SoM | Vision |
|----------|-----|-----|--------|
| fail_no_progress | **86** (36.8%) | **86** (36.8%) | **136** (58.1%) |
| fail_early_finish | 14 (6.0%) | 28 (12.0%) | 33 (14.1%) |
| fail_finish_eval_mismatch | 25 (10.7%) | 15 (6.4%) | 5 (2.1%) |
| fail_max_steps_target_unreachable | 25 (10.7%) | 17 (7.3%) | 3 (1.3%) |
| fail_finish_wrong_url_not_found | 16 (6.8%) | 19 (8.1%) | 10 (4.3%) |
| fail_max_steps_click_back_loop | 11 (4.7%) | 6 (2.6%) | 1 (0.4%) |
| fail_max_steps_search_repeat | 9 (3.8%) | 1 (0.4%) | 0 |
| fail_finish_empty_answer | 8 (3.4%) | 9 (3.8%) | 3 (1.3%) |
| fail_incomplete_or_stuck | 5 (2.1%) | 6 (2.6%) | 16 (6.8%) |
| fail_finish_claim_missing | 6 (2.6%) | 3 (1.3%) | 1 (0.4%) |

### 4.2 DOM vs SoM 失败模式对比

**DOM 特有高发**：
- `fail_finish_eval_mismatch`（10.7% vs 6.4%）
- `fail_max_steps_click_back_loop`（4.7% vs 2.6%）
- `fail_max_steps_search_repeat`（3.8% vs 0.4%）

**SoM 特有高发**：
- `fail_early_finish`（12.0% vs 6.0%）
- `fail_finish_wrong_url_not_found`（8.1% vs 6.8%）

### 4.3 Vision 主要失败路径

- **fail_no_progress 支配**（58.1%）：坐标 misclick → 无效动作 → 早停
- **fail_early_finish**（14.1%）：缺 AXTree 结构化导航信息
- **fail_incomplete_or_stuck**（6.8%）：坐标不精确的低效循环

### 4.4 DOM 与 SoM 差距根因（§18/§23）

**Mirage Effect**（§18）：相同文本信息 + 图片存在触发质变推理路径。

§95 后 DOM 独占成功增至 13 个（此前 7 个），说明 DOM 文本信息在更多 task 上有不可替代的价值。SoM vs DOM 差距从显著（p=0.004）变为 marginal（p=0.065）。

---

## 5. 路由信号评估

> 三模式 702 episodes，adjusted labels

### 5.1 信号区分力（Combined AUROC）

| 信号类型 | 最佳指标 | AUROC | 95% CI |
|---------|---------|-------|--------|
| Verbalized | **ep_mean_verbalized** | **0.769** | [0.704, 0.832] |
| 行为信号 | url_revisit_max | **0.767** | [0.718, 0.814] |
| 行为信号 | action_diversity | 0.749 | [0.688, 0.810] |
| 行为信号 | url_revisit_count | 0.747 | [0.691, 0.798] |
| 行为信号 | max_repeat_streak | 0.673 | [0.607, 0.735] |
| Token-level | ep_max_entropy | 0.594 | [0.523, 0.665] |

### 5.2 跨模式一致性

| 信号 | DOM | SoM | Vision |
|------|-----|-----|--------|
| ep_mean_verbalized | 0.753 | 0.755 | **0.757** |
| url_revisit_max | 0.755 | 0.727 | **0.816** |
| action_diversity | 0.738 | 0.706 | **0.809** |
| max_repeat_streak | **0.761** | 0.604 | 0.744 |

**Verbalized 信号三模式 AUROC 几乎相同（0.753-0.757），最适合跨模式路由。**

### 5.3 路由就绪度

| 维度 | 结论 |
|------|------|
| Token-level | **弱**（AUROC 0.49-0.59） |
| 行为信号 | **有**（url_revisit_max 0.77，跨模式一致） |
| Verbalized | **有**（0.77，三模式高度一致） |
| **整体** | **行为信号 + verbalized 均可用于路由** |

### 5.4 State Change × Outcome

| 模式 | 成功 page_change_rate | 失败 page_change_rate | 成功 avg_steps | 失败 avg_steps |
|------|---------------------|---------------------|---------------|---------------|
| DOM | 0.733 | 0.723 | 8.2 | 14.4 |
| SoM | 0.603 | 0.668 | 4.9 | 10.7 |
| Vision | 0.619 | 0.574 | 3.7 | 7.0 |

成功 episode 步数显著少于失败（Vision: 3.7 vs 7.0），但 page_change_rate 差异不大。与 B0 一致：成功取决于精准少步操作。

### 5.5 Temporal SR 趋势

| 模式 | Q1 (earliest) | Q5 (latest) | 趋势 |
|------|--------------|-------------|------|
| DOM | 10.9% | 6.0% | ↓ 下降 |
| SoM | 8.7% | 10.0% | → 稳定 |
| Vision | 4.4% | 8.0% | ↑ 上升 |

DOM 呈现 temporal degradation 趋势。SoM 和 Vision 保持稳定或略升。

---

## 6. 共性脚手架缺陷

### 6.1 地点过滤困难（3 例）

Classifieds 站点的地点筛选依赖搜索结果页的 City 文本输入框。

### 6.2 `<select>` 下拉菜单三层不可达（VWA 框架级缺陷）

`<select>` 在 VWA 默认配置下对所有 agent 实质不可用。

### 6.3 Type 操作导致页面全选变蓝

VWA 框架内置 `Meta+A` 作为 type 前置步骤。

### 6.4 极少翻页

模型几乎只会反复 scroll，极少点击分页控件。

### 6.5 confirm 弹窗不可交互（VWA 框架级缺陷）

Classifieds "Delete" 触发浏览器原生 `confirm()` 弹窗，VWA Playwright 默认不自动接受。

### 6.6 N/A 任务 False Positive（10 例）

10 个 N/A reference task，三模式全部误判为 success=1.0。

### 6.7 搜索关键词过于具体

模型将任务描述全部约束拼接为搜索词。正确策略：宽泛品类词搜索 + 筛选器/排序/翻页。

---

## 7. Phase 2 路由方向

### 7.1 Headroom 评估

| 路由场景 | Adjusted headroom | 独占成功 | 可行性 |
|---------|------------------|---------|--------|
| SoM ↔ DOM | 8.12pp − ? | DOM 13, SoM 15 | **高价值**（DOM 贡献 13 个独占成功） |
| SoM ↔ Vision | 8.12pp − ? | Vision 6, SoM 15 | **有价值** |
| **三模式 Oracle** | **8.12pp** | DOM 13, Vision 6, SoM 15 | **最大化利用** |

> §95 变更：DOM 独占成功从 7 个升至 13 个，headroom 从 5.13pp 升至 8.12pp。DOM 路由价值大幅提升。

### 7.2 推荐路由设计

**三模式路由**：
- SoM 作为默认（Adjusted SR 最高，13.84%）
- DOM 作为特定 task 类型的替代（13 个独占成功）
- Vision 作为低成本替代（$0.0133/ep）
- 路由信号：ep_mean_verbalized（三模式高度一致 0.75+）

### 7.3 Pareto 分析

| 策略 | Adjusted SR | 平均成本 |
|------|-------------|---------|
| 全部 SoM | 13.84% | $0.0347 |
| 全部 DOM | 7.59% | $0.0399 |
| 全部 Vision | 7.14% | $0.0133 |
| Oracle 三模式 | **21.37%** | ~$0.030 |

---

## 方法论说明

- **Adjusted SR**：仅扣除 N/A FP + eval FP（§95），不再扣除 visual FP。分母 224（移除 10 个 N/A reference task）
- **cross_rep adjusted labels**：扣除 na_fp + eval_fp，分母保持 234
- **统计检验**：McNemar exact test（成功率），Wilcoxon signed-rank（成本/延迟），Bootstrap 10K resamples（CI）
- **路由信号**：AUROC 使用 adjusted labels

---

*数据目录：`results/visualwebarena/phase1/B1_3mode_classifieds_20260413/analysis/`*
*文档更新：2026-04-25（§95 FP 重构：废弃 visual_fp，DOM adjusted SR 从 4.91% 升至 7.59%；SoM vs DOM 从显著变为 marginal；DOM 独占成功增至 13 个；routing headroom 升至 8.12pp）*
