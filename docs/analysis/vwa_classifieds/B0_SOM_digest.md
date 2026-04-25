# B0 SoM Baseline 分析报告（Classifieds）

> B0 = Qwen3-VL-235B-A22B（proxy API），SoM 模式，classifieds 站点
> 对应 B1 分析见 `B1_SOM_digest.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**
>
> 数据来源：`phase1_som_router_0`，classifieds 全部 234 tasks
> 分析方法：自动化 post analysis（condition_summary_v2 + reason_diagnostics + cross_representation）
> GLM Digest 覆盖：74/179 failed episodes 有完整 GLM 根因分析（2026-04-23 补跑）
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**

---

## 一、总体概况

| 指标 | 数值 |
|------|------|
| SoM condition 总 episode 数 | 234 |
| 成功（raw） | 54（23.08%） |
| N/A FP | 8 |
| Eval FP | 0 |
| **Adjusted SR** | **20.54%**（46/224） |
| 平均步数 | 8.60 步 |
| 平均成本 | $0.0415 / episode |
| P95 延迟 | 74,004ms |

### 与 DOM / Vision 对比（B0 三模式）

| 指标 | DOM | **SoM** | Vision |
|------|-----|---------|--------|
| Raw SR | 14.96% | **23.08%** | 15.81% |
| Adjusted SR | 12.95% | **20.54%** | 12.05% |
| 平均步数 | 11.56 | **8.60** | 7.85 |
| 平均成本 | $0.0427 | $0.0415 | **$0.0248** |
| SoM vs DOM McNemar p | — | **0.0115 ★** | — |
| SoM vs Vision McNemar p | — | **0.085** (n.s.) | — |
| Vision vs DOM McNemar p | — | — | **0.627** (n.s.) |

B0 三模式中，**SoM adjusted SR 最高（20.54%），显著优于 DOM**（McNemar p=0.012）。SoM vs Vision 差异 marginal（p=0.085），Vision vs DOM 不显著（p=0.627）。

> §95 变更：DOM adjusted SR 从 8.48% 上升至 12.95%（visual_fp 废弃），SoM vs DOM 差距缩小（从 12.06pp 缩至 7.59pp），但仍显著。

### 与 B1 SoM 对比

| 模型 | Raw SR | Adjusted SR |
|------|--------|-------------|
| **B0 (235B)** | **23.08%** | **20.54%** |
| B1 (4B) | 17.52% | 13.84% |

**B0 SoM 以 +6.70pp 领先 B1 SoM**。235B 模型在 SoM 多模态输入下确实更强。

---

## 二、失败原因分布

| 失败原因 | 数量 | 比例 | 备注 |
|---------|------|------|------|
| **fail_finish_wrong_url_not_found** | **57** | **24.4%** | 最大失败源 |
| success | 54 | 23.1% | (raw) |
| fail_early_finish | 32 | 13.7% | SoM 过度自信早停 |
| fail_finish_eval_mismatch | 32 | 13.7% | 评测不一致 |
| fail_no_progress | 19 | 8.1% | 连续无进展 |
| fail_max_steps_target_unreachable | 15 | 6.4% | 目标不可达 |
| fail_finish_claim_missing | 9 | 3.8% | 声称缺失 |
| fail_max_steps | 5 | 2.1% | 步数耗尽 |
| fail_finish_empty_answer | 4 | 1.7% | 空答案 |
| fail_max_steps_click_back_loop | 4 | 1.7% | click-back 循环 |
| fail_incomplete_or_stuck | 2 | 0.9% | 不完整/卡住 |
| fail_finish_wrong_url_left_target | 1 | 0.4% | 离开目标页 |

---

## 三、核心失败模式详析

### 3.1 fail_finish_wrong_url_not_found（57 例，24.4%）

最大失败源。Agent 执行了操作但 finish 时停在错误的页面/URL。常见于 `single_navigation + url_match` 任务：agent 找到了类似商品但非 eval 要求的精确目标。SoM 截图给 agent "视觉确认"感，使其更容易在相似页面上 finish。

### 3.2 fail_early_finish / SoM 过度自信早停（32 例，13.7%）

SoM 模式下 agent 过早 finish，且 confidence 往往很高（0.8-0.95）。核心机制是**"信息充分幻觉"**：SoM 截图+标注给 agent "已看到所有信息"的错觉。

**两种子模式**：

1. **"看了没找到"型早停**：搜索结果截图中看不到目标视觉属性 → 直接判定不存在，不尝试更多搜索策略
2. **"看到了就是了"型早停**：截图中看到似乎匹配的商品 → 不验证约束条件就 finish

这是 SoM 的 **text-over-vision 反转**：截图不是帮助 agent 更好决策，而是提供了一个"快速放弃"或"快速确认"的锚点。

### 3.3 fail_no_progress（19 例，8.1%）

对比 DOM（62 例，26.5%）：SoM 远低，因为 SoM 截图帮助 agent 更快定位交互元素。

### 3.4 "自信声明找不到" 跨模式定量分析

> 注：本节数据来自早期人工分析 pass，定量数字基于当时数据快照。定性结论和比例特征保持一致。

| 模式 | 失败数 | 声明找不到 | 占失败比 |
|------|--------|-----------|---------|
| DOM | ~199 | 9 | **5%** |
| **SoM** | ~177 | **49** | **28%** |
| Vision | ~198 | 29 | 15% |

SoM "错误放弃"率是 DOM 的 5.6 倍，Vision 的 1.9 倍。

### 3.5 GLM Digest 根因分类（N=74 full GLM）

| 排名 | GLM 根因类别 | 数量 | 占比 |
|------|------------|------|------|
| 1 | **目标不可达** | 37 | 50.0% |
| 2 | 过早结束 | 12 | 16.2% |
| 3 | 答案对齐错误 | 9 | 12.2% |
| 4 | 执行停滞 | 7 | 9.5% |
| 5 | 导航循环 / 搜索循环 | 4 + 4 | 10.8% |

**关键发现**：SoM 标注在多数失败 episode 中被 agent 忽视（57% 未使用视觉标注，53% 表现为 text_over_vision）。SoM 标注的实际价值主要体现在成功 episode 的精准元素定位上。

---

## 四、交叉分析（三模式完整）

### 4.1 集合分析（Adjusted，/234 分母）

| 集合 | 数量 | 占比 |
|------|------|------|
| all_fail | 161 | 68.8% |
| only_som | 21 | 9.0% |
| dom_and_som (not vision) | 12 | 5.1% |
| only_vision | 11 | 4.7% |
| all_success | 9 | 3.9% |
| only_dom | 8 | 3.4% |
| som_and_vision (not dom) | 8 | 3.4% |
| dom_and_vision (not som) | 4 | 1.7% |

SoM adjusted (/234): 50/234 = 21.37%。SoM 独有成功 21 tasks。

### 4.2 Oracle 路由分析

| 指标 | Raw | Adjusted |
|------|-----|----------|
| Oracle ceiling | 32.05% (75 tasks) | **31.20%** (73 tasks) |
| 最佳单模式 (SoM) | 23.08% | 21.37% |
| Routing headroom | 8.97pp | **9.83pp** |
| Oracle 选择分布 (raw) | SoM:27, Vision:27, DOM:21 | — |
| Oracle 选择分布 (adj) | SoM:25, Vision:25, DOM:23 | — |

**Adjusted routing headroom 9.83pp**：理论最优路由可将 SR 从 21.37% 提升到 31.20%。三模式在 oracle 中均有均衡贡献。

### 4.3 成本效率

| 模式 | Adjusted SR (/224) | 平均成本 | cost_efficiency_ratio |
|------|------------|---------|----------------------|
| DOM | 12.95% | $0.0427 | 3.03 |
| **SoM** | **20.54%** | $0.0415 | **4.95** |
| Vision | 12.05% | $0.0248 | 4.86 |

SoM 的 cost_efficiency_ratio（SR/cost）是三模式中最高的（4.95）。

---

## 五、共性脚手架缺陷（与 B1 SoM 相同）

- **`<select>` 下拉菜单三层不可达**（VWA 框架级缺陷）
- **confirm 弹窗不可交互**：Delete 操作被取消
- **N/A 任务 False Positive（8/10）**：机制与 B1 相同（Agent prompt 无 N/A 出口）
- **极少翻页**：SoM 模式下 agent 仍以 scroll 为主

---

## 六、B0 SoM 效率分析

| 指标 | B0 SoM | B0 DOM | B0 Vision |
|------|--------|--------|-----------|
| 平均步数 | 8.60 | 11.56 | 7.85 |
| 平均成本/ep | $0.0415 | $0.0427 | **$0.0248** |
| cost_efficiency_ratio | **4.95** | 3.03 | 4.86 |
| P95 延迟 | 74,004ms | **37,513ms** | 44,984ms |

SoM 步数少于 DOM（8.60 vs 11.56），略多于 Vision（7.85）。SoM 的 cost_efficiency_ratio（SR/cost）最高。但 P95 延迟显著高于 DOM（SoM 图文混合请求在 proxy API 上更耗时）。

---

*更新时间：2026-04-25（§95 FP 重构：废弃 visual_fp，更新全部定量数据及 McNemar 检验；DOM adjusted SR 上升导致 SoM vs DOM 差距缩小但仍显著）*
*数据来源：B0_3mode_classifieds_20260413 phase1_som_router_0*
*B0 三模式定量对比见 `B0_findings.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`*
