# B0 vs B1 Reddit 跨模型对比报告

> B0: Qwen3-VL-235B-A22B（proxy API，temperature=0.1，max_tokens=4096）
> B1: Qwen3-VL-4B bf16（本地推理，do_sample=False，max_new_tokens=384）
> 站点: Reddit (Postmill), 210 tasks × 3 modes
> 本报告关注模型规模（4B vs 235B）对三种观测模式的差异化影响
> B0 run: `B0_3mode_reddit_20260422` | B1 run: `B1_3mode_reddit_20260413`
> **v1 (2026-04-24): 三模式完整数据首版**

---

## 1. 核心对比表

### 1.1 Adjusted SR 对比

| 模式 | B0 (235B) | B1 (4B) | 差值 | 方向 |
|------|-----------|---------|------|------|
| DOM | **8.78%** | 6.83% | **+1.95pp** | B0 > B1 |
| SoM | **11.71%** | 5.85% | **+5.86pp** | B0 >> B1 |
| Vision | **6.34%** | 2.44% | **+3.90pp** | B0 > B1 |

**235B 模型在全部三种模式上优于 4B**，符合模型规模假设。SoM 差距最大（+5.86pp），与 Classifieds 模式一致。

### 1.2 Raw SR 对比

| 模式 | B0 Raw | B1 Raw | 差值 |
|------|--------|--------|------|
| DOM | 11.43% | 10.00% | +1.43pp |
| SoM | 13.81% | 8.10% | +5.71pp |
| Vision | 8.57% | 4.76% | +3.81pp |

Raw SR 方向与 adjusted SR 一致。SoM 差距最大（+5.71pp），DOM 差距最小（+1.43pp）。

### 1.3 效率对比

| 指标 | B0 DOM | B1 DOM | B0 SoM | B1 SoM | B0 Vision | B1 Vision |
|------|--------|--------|--------|--------|-----------|-----------|
| 平均步数 | **12.70** | 16.64 | **8.09** | 11.70 | 6.87 | **6.45** |
| 平均成本/ep | $0.0516 | $0.0536 | $0.0387 | $0.0410 | $0.0227 | **$0.0137** |

> 注：B0 成本为 API 实际调用费用；B1 成本为本地 GPU 推理的 API 等价估算。两者成本体系不同，直接比较需谨慎。

B0 步数在 DOM/SoM 上显著少于 B1（DOM: 12.70 vs 16.64；SoM: 8.09 vs 11.70）。235B 模型更高效——更快收敛到成功或失败。Vision B1 步数略少（6.45 vs 6.87）但 SR 更低——快速失败（premature finish）压低了平均步数。

---

## 2. 模式排序反转：DOM 在 B1 中的异常领先

### 2.1 现象

B0 和 B1 在 Reddit 上的模式排序**不同**：

| 排序 | B0 (235B) | B1 (4B) |
|------|-----------|---------|
| 第一 | **SoM** (11.71%) | **DOM** (6.83%) |
| 第二 | DOM (8.78%) | SoM (5.85%) |
| 第三 | Vision (6.34%) | Vision (2.44%) |

**B1 Reddit 是所有 site × baseline 组合中唯一 DOM 领先 SoM 的场景**（Classifieds B0/B1 均为 SoM 领先）。

### 2.2 原因分析

1. **SoM ID 幻觉在 Reddit 上更严重**：Reddit 论坛页面链接密集，SoM 标注 ID 对 4B 模型产生更多幻觉干扰。B1 SoM `no_progress` 率 38.6%，而 B0 SoM 仅 34.8% — 235B 模型能更好地处理 SoM 标注。

2. **DOM 的搜索能力在低能力模型上更有价值**：DOM 提供精确 element_id，4B 模型虽然搜索策略差（search_repeat 22.9%），但至少能执行有效的搜索操作。SoM/Vision 的坐标/ID 点击在 4B 模型上更不可靠。

3. **DOM 步数多 = 更多探索机会**：DOM 16.64 步 vs SoM 11.70 步，更多步数给了 4B 模型更多"试错"机会。

### 2.3 Mirage Effect 在 B1 Reddit 上的弱化

| 模型 | SoM SR | DOM SR | Mirage Gap |
|------|--------|--------|-----------|
| B0 | 11.71% | 8.78% | **+2.93pp** |
| B1 | 5.85% | 6.83% | **-0.98pp** |

**B1 Reddit Mirage Gap 为负值**（-0.98pp），即 SoM 反而不如 DOM。这与 Classifieds 的 Mirage Effect 形成对比：

| 站点 | B0 Mirage Gap | B1 Mirage Gap |
|------|--------------|--------------|
| Classifieds | +12.06pp | +8.93pp |
| Reddit | +2.93pp | **-0.98pp** |

Reddit 站点对 Mirage Effect 的抑制作用在 4B 模型上最为显著。原因：Reddit 页面结构复杂（密集链接、多层导航），SoM 标注的视觉信息反而给 4B 模型带来更多干扰而非帮助。

---

## 3. 各模式详细对比

### 3.1 DOM 模式

| 指标 | B0 DOM | B1 DOM | 分析 |
|------|--------|--------|------|
| Adjusted SR | 8.78% | 6.83% | B0 领先 +1.95pp（三模式中差距最小） |
| 平均步数 | 12.70 | **16.64** | B1 步数更多（4B 循环更多） |
| search_repeat | 13.8% | **22.9%** | B1 搜索循环更严重 |
| click_back_loop | 4.3% | **9.5%** | B1 导航循环更严重 |
| eval_mismatch | **23.8%** | 13.8% | B0 答案对齐问题更突出 |

**DOM 是 B0/B1 差距最小的模式**（+1.95pp）。DOM 提供的结构化文本信息在两个模型上都有一定效果，但 4B 模型的策略缺陷更严重（search_repeat 22.9% vs 13.8%）。

B0 DOM 的独特问题：`eval_mismatch` 更高（23.8% vs 13.8%），说明 235B 模型更倾向于给出答案（即使错误），而 4B 更倾向于陷入循环。

### 3.2 SoM 模式

| 指标 | B0 SoM | B1 SoM | 分析 |
|------|--------|--------|------|
| Adjusted SR | **11.71%** | 5.85% | B0 领先 +5.86pp（差距最大） |
| 平均步数 | 8.09 | **11.70** | B1 步数多 45%（效率差） |
| no_progress | 34.8% | **38.6%** | B1 SoM 交互失败更严重 |
| early_finish | **14.8%** | 9.5% | B0 更频繁过早完成 |
| click_back_loop | 1.0% | **6.2%** | B1 导航循环是 B0 的 6 倍 |

**SoM 是 B0/B1 差距最大的模式**（+5.86pp）。235B 模型能更好地利用 SoM 标注信息（结合截图布局和元素 ID），而 4B 模型在 SoM 上受 ID 幻觉拖累严重。

B0 SoM 的 `early_finish`（14.8%）高于 B1（9.5%） — 235B 模型的视觉确认过度自信问题在 SoM 上更突出。

### 3.3 Vision 模式

| 指标 | B0 Vision | B1 Vision | 分析 |
|------|-----------|-----------|------|
| Adjusted SR | **6.34%** | 2.44% | B0 领先 +3.90pp |
| 平均步数 | 6.87 | **6.45** | 接近（两者都快速失败） |
| no_progress | 40.0% | **56.2%** | B1 坐标 misclick 更频繁 |
| incomplete_or_stuck | 12.9% | **22.9%** | B1 更容易陷入低效循环 |

Vision 模式差距适中（+3.90pp）。两个模型步数接近（~6.5 步），但 B1 的 no_progress 率高达 56.2%（B0: 40.0%），说明 4B 模型的坐标精度在 Reddit 密集布局上更差。

---

## 4. Oracle 路由格局对比

### 4.1 Oracle Ceiling 对比

| 指标 | B0 | B1 |
|------|----|----|
| 最优单模式 SR | **SoM 11.71%** | DOM 6.83% |
| Oracle ceiling (adj) | **16.19%** | 8.57% |
| Routing headroom | **+5.24pp** | +2.86pp |
| Oracle DOM 贡献 | 10 (29.4%) | **12 (66.7%)** |
| Oracle SoM 贡献 | **13 (38.2%)** | 3 (16.7%) |
| Oracle Vision 贡献 | **11 (32.4%)** | 3 (16.7%) |

B0 routing headroom（5.24pp）是 B1（2.86pp）的 1.83 倍。

**关键差异：Oracle 选择分布的极化**：
- B0: 三模式均衡（29-38%），SoM 主导
- B1: 高度偏向 DOM（66.7%），SoM 和 Vision 各仅 16.7%

这进一步验证了 Capability-Aware Routing 的核心论点：**最优表征是模型能力的函数**。4B 模型在 Reddit 上需要 DOM 的结构化文本来弥补视觉推理能力的不足。

### 4.2 Exclusive Sets 对比

| 集合 | B0 | B1 |
|------|----|----|
| all_fail | 176 (83.8%) | **192 (91.4%)** |
| only_som | **9 (4.3%)** | 3 (1.4%) |
| only_dom | 4 (1.9%) | **7 (3.3%)** |
| only_vision | **5 (2.4%)** | 2 (1.0%) |
| all_success | 2 (1.0%) | **0 (0%)** |

B1 all_fail 比例（91.4%）远高于 B0（83.8%），B1 在 Reddit 上几乎所有 task 全模式失败。B1 无任何 all_success task。

---

## 5. 行为差异

### 5.1 搜索行为

| 指标 | B0 | B1 |
|------|----|----|
| 搜索率（DOM）| 62.9% | **68.8%** |
| search_repeat 失败率（DOM）| 13.8% | **22.9%** |
| 搜索后切换为浏览 | ✓ （较常见）| ✗ （极少） |

B1 搜索倾向更强（68.8% vs 62.9%），且搜索失败后不会切换策略，导致 search_repeat 成为 B1 DOM 的最大失败模式。B0 在搜索失败后更快切换为 scroll 浏览。

### 5.2 坐标自纠正（Vision）

| 行为 | B0 | B1 |
|------|----|----|
| Misclick 后策略 | 修改坐标重试 | **重复相同坐标** |
| no_progress 率 | 40.0% | **56.2%** |

B0 在 Vision misclick 后会尝试调整坐标（虽然方向不一定正确），B1 则以相同坐标和 confidence 重复点击，这反映了 235B 更强的状态感知能力。

### 5.3 过早完成（SoM）

| 行为 | B0 | B1 |
|------|----|----|
| early_finish 率（SoM）| **14.8%** | 9.5% |
| eval_mismatch 率（SoM）| **12.9%** | 8.1% |

B0 SoM 的 early_finish 和 eval_mismatch 更高 — 235B 模型更果断但也更容易过早给出（错误）答案。4B 模型在 SoM 上更多是执行失败（no_progress 38.6%）而非判断错误。

---

## 6. Mirage Effect 跨模型对比

### 6.1 Mirage Gap

| 模型 | SoM SR (adj) | DOM SR (adj) | Mirage Gap |
|------|-------------|-------------|-----------|
| B0 | 11.71% | 8.78% | **+2.93pp** |
| B1 | 5.85% | 6.83% | **-0.98pp** |

### 6.2 跨站 Mirage Effect 对比

| 站点 × 模型 | Mirage Gap | SoM 优势? |
|-------------|-----------|-----------|
| Classifieds B0 | +12.06pp | **强** |
| Classifieds B1 | +8.93pp | **强** |
| Reddit B0 | +2.93pp | **弱** |
| Reddit B1 | -0.98pp | **无（反转）** |

**Mirage Effect 存在站点×模型交互**：
- Classifieds（结构化列表页）上两个模型都展现强 Mirage Effect
- Reddit（密集论坛页）上 Mirage Effect 大幅减弱，B1 甚至反转

原因：Reddit 的页面复杂度（密集链接、多层导航）超过了 4B 模型从 SoM 标注中获益的能力上限。SoM 标注反而增加了信息噪声，4B 模型更适合 DOM 的纯文本结构。

---

## 7. 路由信号跨模型对比

### 7.1 最佳信号 AUROC

| 信号 | B0 | B1 |
|------|----|----|
| ep_mean_verbalized | **0.769** | 0.719 |
| max_repeat_streak | **0.670** | 0.629 |
| url_revisit_max | 0.585 | 0.575 |
| action_diversity | **0.610** | 0.521 |

B0 的信号区分力全面优于 B1，尤其 action_diversity（0.610 vs 0.521）。235B 模型的行为信号更有区分力，可能因为 235B 在成功/失败 episode 间的行为差异更大。

### 7.2 Token-level 信号（仅 B1）

B1 有 token-level 信号但全部 AUROC ≈ 0.5（无区分力）。4B 模型在 Reddit 上的 logprob/entropy 无法区分成功与失败。

---

## 8. 关键发现汇总

1. **B0 在全部三种模式上优于 B1**：SoM 差距最大（+5.86pp），DOM 差距最小（+1.95pp），Vision 居中（+3.90pp）。

2. **模式排序反转**：B1 Reddit 唯一出现 DOM > SoM 的场景（6.83% vs 5.85%），原因是 SoM 标注 ID 幻觉在 Reddit 密集布局上对 4B 模型干扰更大。

3. **Mirage Effect 站点×模型交互**：Classifieds 上 Mirage Effect 强（B0 +12.06pp, B1 +8.93pp），Reddit 上大幅减弱（B0 +2.93pp, B1 -0.98pp）。Reddit 页面复杂度超过 4B 模型从 SoM 标注获益的上限。

4. **Oracle 选择分布极化**：B0 三模式均衡（29-38%），B1 高度偏向 DOM（66.7%）。验证了 Capability-Aware Routing 的核心论点：最优表征是模型能力的函数。

5. **B0 routing headroom 是 B1 的 1.83 倍**（5.24pp vs 2.86pp）：B0 oracle ceiling 16.19%，B1 仅 8.57%。更大模型在 Reddit 上有更大的路由收益空间。

6. **Reddit 整体难度远高于 Classifieds**：B1 all_fail 91.4%（Classifieds: 81.2%），B0 all_fail 83.8%（Classifieds: 70.9%）。两站差异主要来自搜索交互难度和 Postmill UI 陷阱。

7. **搜索策略差异**：B1 搜索倾向更强（68.8% vs 62.9%），search_repeat 更严重（22.9% vs 13.8%），搜索失败后不切换策略。

8. **设计不对称需注意**：B0/B1 存在温度/max_tokens/scroll 等不对称（同 Classifieds），SR 差异无法完全归因于模型规模。

---

## 方法论说明

- **比较局限**：B0 和 B1 存在多项设计不对称（温度/max_tokens/scroll），SR 差异无法完全归因于模型规模
- **Adjusted SR**：使用 analysis_summary 方案（/205 分母），cross_representation 使用 /210 分母
- **Mirage Gap**：使用 adjusted SR 计算（SoM adj SR - DOM adj SR）
- **成本比较**：B0（API 费用）与 B1（本地推理 GPU 等价成本）成本体系不同
- **数据时间**：B0 run 日期 2026-04-22，B1 run 日期 2026-04-13，两者非同日运行但 Reddit 站点状态一致
- **三模式完整**：B0/B1 均为 210/210 × 3 完整数据（此前版本 B0 Vision 不完整、B1 SoM 极少）

---

*v1 (2026-04-24): 三模式完整数据首版*
*数据来源：B0_3mode_reddit_20260422 + B1_3mode_reddit_20260413*
*B0 三模式详情：B0_findings.md；B1 三模式详情：B1_findings.md*
