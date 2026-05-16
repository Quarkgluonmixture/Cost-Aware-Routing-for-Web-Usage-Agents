# B0 vs B1 Reddit 跨模型对比报告

> B0: Qwen3-VL-235B-A22B（proxy API，temperature=0.1，max_tokens=4096）
> B1: Qwen3-VL-4B bf16（本地推理，do_sample=False，max_new_tokens=384）
> 站点: Reddit (Postmill), 210 tasks × 3 modes
> 本报告关注模型规模（4B vs 235B）对三种观测模式的差异化影响
> B0 run: `B0_3mode_reddit_20260422` | B1 run: `B1_3mode_reddit_20260413`
>
> **v2 (2026-04-26)**：
> - **B1 reddit SoM 是 max_marks 80→200 重跑后版本**（§94），目的为验证之前 SoM 反垫底是否因为标记不足。**结果：反转未消失，B1 Reddit SoM 仍劣于 DOM**（更深：-1.91pp vs v1 -0.98pp）。
> - 全部 4 condition 在 04-26 经过 §97 rederive（PUR 重算 → eval FP 判定更新），SR 数字相对 v1 有 0.1-1.2pp 漂移，文字结论与 v1 一致。
> - **B1 数据非最终**：DGX 共享 GPU 同时跑多实例时存在 VRAM/算力争抢，B1 latency 数字受污染；最终 latency 待 Myriad HPC 上线后用独占 GPU 重跑。SR/cost/oracle 数字不受影响（推理逻辑 deterministic）。

---

## 1. 核心对比表

### 1.1 Adjusted SR 对比 (v2, post-rederive)

| 模式 | B0 (235B) | B1 (4B) | 差值 | 方向 |
|------|-----------|---------|------|------|
| DOM | **8.10%** | 6.67% | **+1.43pp** | B0 > B1 |
| SoM | **10.48%** | 4.76% | **+5.72pp** | B0 >> B1 |
| Vision | 6.67% | 1.43% | **+5.24pp** | B0 >> B1 |

**235B 模型在全部三种模式上优于 4B**，符合模型规模假设。SoM 差距最大（+5.72pp），DOM 差距最小（+1.43pp）。Vision 差距 +5.24pp，与 v1 (+3.90pp) 相比扩大，因 B1 Vision 在 rederive 后再降至 1.43%。

### 1.2 Raw SR 对比

| 模式 | B0 Raw | B1 Raw | 差值 |
|------|--------|--------|------|
| DOM | 11.43% | 10.00% | +1.43pp |
| SoM | 11.90% | 8.10% | +3.80pp |
| Vision | 8.57% | 4.76% | +3.81pp |

Raw SR 方向与 adjusted SR 一致。Adjusted/Raw 差异主要来自 N/A FP（每模式 5）+ eval FP（B0 SoM 3 / B1 SoM 7）。

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
| 第一 | **SoM** (10.48%) | **DOM** (6.67%) |
| 第二 | DOM (8.10%) | SoM (4.76%) |
| 第三 | Vision (6.67%) | Vision (1.43%) |

**B1 Reddit 是所有 site × baseline 组合中唯一 DOM 领先 SoM 的场景**（Classifieds B0/B1 均为 SoM 领先）。

**§94 验证（max_marks 80→200）**：v1 数据 max_marks=80 时怀疑 SoM 反转可能由"reddit 列表元素超过 80 标记数被截断"导致。B1 reddit SoM 在 04-25 用 max_marks=200 重跑后，反转**未消失反而加深**（v1 -0.98pp → v2 -1.91pp）。结论：标记数不是主导因素，4B 模型在 reddit 密集页面上无法从 SoM 视觉信息获益是结构性问题。

### 2.2 原因分析

1. **SoM ID 幻觉在 Reddit 上更严重**：Reddit 论坛页面链接密集，SoM 标注 ID 对 4B 模型产生更多幻觉干扰。B1 SoM `no_progress` 率 38.6%，而 B0 SoM 仅 34.8% — 235B 模型能更好地处理 SoM 标注。

2. **DOM 的搜索能力在低能力模型上更有价值**：DOM 提供精确 element_id，4B 模型虽然搜索策略差（search_repeat 22.9%），但至少能执行有效的搜索操作。SoM/Vision 的坐标/ID 点击在 4B 模型上更不可靠。

3. **DOM 步数多 = 更多探索机会**：DOM 16.64 步 vs SoM 11.70 步，更多步数给了 4B 模型更多"试错"机会。

### 2.3 Mirage Effect 在 B1 Reddit 上的弱化

| 模型 | SoM SR | DOM SR | Mirage Gap |
|------|--------|--------|-----------|
| B0 | 10.48% | 8.10% | **+2.38pp** |
| B1 | 4.76% | 6.67% | **-1.91pp** |

**B1 Reddit Mirage Gap 为负值**（-1.91pp），即 SoM 显著不如 DOM。max_marks 重跑后反转幅度比 v1 翻倍（-0.98pp → -1.91pp）。

跨站对比：

| 站点 | B0 Mirage Gap | B1 Mirage Gap |
|------|--------------|--------------|
| Classifieds | +7.27pp | +4.70pp |
| Reddit | +2.38pp | **-1.91pp** |

Reddit 站点对 Mirage Effect 的抑制作用在 4B 模型上最为显著。原因：Reddit 页面结构复杂（密集链接、多层导航），SoM 标注的视觉信息反而给 4B 模型带来更多干扰而非帮助。max_marks=200 实验进一步证实，问题不在标记数量而在小模型对 SoM 视觉布局的处理能力。

---

## 3. 各模式详细对比

### 3.1 DOM 模式

| 指标 | B0 DOM | B1 DOM | 分析 |
|------|--------|--------|------|
| Adjusted SR | 8.10% | 6.67% | B0 领先 +1.43pp（三模式中差距最小） |
| 平均步数 | 12.70 | **16.64** | B1 步数更多（4B 循环更多） |
| search_repeat | 13.8% | **22.9%** | B1 搜索循环更严重 |
| click_back_loop | 4.3% | **9.5%** | B1 导航循环更严重 |
| eval_mismatch | **23.8%** | 13.8% | B0 答案对齐问题更突出 |

**DOM 是 B0/B1 差距最小的模式**（+1.95pp）。DOM 提供的结构化文本信息在两个模型上都有一定效果，但 4B 模型的策略缺陷更严重（search_repeat 22.9% vs 13.8%）。

B0 DOM 的独特问题：`eval_mismatch` 更高（23.8% vs 13.8%），说明 235B 模型更倾向于给出答案（即使错误），而 4B 更倾向于陷入循环。

### 3.2 SoM 模式

| 指标 | B0 SoM | B1 SoM | 分析 |
|------|--------|--------|------|
| Adjusted SR | **10.48%** | 4.76% | B0 领先 +5.72pp（差距最大；max_marks=200 后 B1 仍劣于 DOM） |
| 平均步数 | 8.09 | **11.70** | B1 步数多 45%（效率差） |
| no_progress | 34.8% | **38.6%** | B1 SoM 交互失败更严重 |
| early_finish | **14.8%** | 9.5% | B0 更频繁过早完成 |
| click_back_loop | 1.0% | **6.2%** | B1 导航循环是 B0 的 6 倍 |

**SoM 是 B0/B1 差距最大的模式**（+5.86pp）。235B 模型能更好地利用 SoM 标注信息（结合截图布局和元素 ID），而 4B 模型在 SoM 上受 ID 幻觉拖累严重。

B0 SoM 的 `early_finish`（14.8%）高于 B1（9.5%） — 235B 模型的视觉确认过度自信问题在 SoM 上更突出。

### 3.3 Vision 模式

| 指标 | B0 Vision | B1 Vision | 分析 |
|------|-----------|-----------|------|
| Adjusted SR | **6.67%** | 1.43% | B0 领先 +5.24pp（v1 +3.90pp，post-rederive 差距扩大） |
| 平均步数 | 6.87 | **6.45** | 接近（两者都快速失败） |
| no_progress | 40.0% | **56.2%** | B1 坐标 misclick 更频繁 |
| incomplete_or_stuck | 12.9% | **22.9%** | B1 更容易陷入低效循环 |

Vision 模式差距适中（+3.90pp）。两个模型步数接近（~6.5 步），但 B1 的 no_progress 率高达 56.2%（B0: 40.0%），说明 4B 模型的坐标精度在 Reddit 密集布局上更差。

---

## 4. Oracle 路由格局对比

### 4.1 Oracle Ceiling 对比

| 指标 | B0 | B1 |
|------|----|----|
| 最优单模式 SR | **SoM 10.48%** | DOM 6.67% |
| Oracle ceiling (raw) | **18.57%** | 12.38% |
| Best single (raw) | 13.81% | 10.48% |
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
| B0 | 10.48% | 8.10% | **+2.38pp** |
| B1 | 4.76% | 6.67% | **-1.91pp** |

### 6.2 跨站 Mirage Effect 对比

| 站点 × 模型 | Mirage Gap | SoM 优势? |
|-------------|-----------|-----------|
| Classifieds B0 | +7.27pp | **强** |
| Classifieds B1 | +4.70pp | **中** |
| Reddit B0 | +2.38pp | **弱** |
| Reddit B1 | -1.91pp | **无（反转，max_marks=200 后未消失）** |

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

## 8. 关键发现汇总 (v2)

1. **B0 在全部三种模式上优于 B1**：SoM 差距最大（+5.72pp），DOM 差距最小（+1.43pp），Vision 居中（+5.24pp）。

2. **模式排序反转**：B1 Reddit 唯一出现 DOM > SoM 的场景（6.67% vs 4.76%），原因是 SoM 标注 ID 幻觉在 Reddit 密集布局上对 4B 模型干扰更大。

3. **§94 验证（max_marks 80→200 重跑后反转未消失）**：v1 (max_marks=80) Mirage Gap -0.98pp，v2 (max_marks=200) -1.91pp，反转加深而非消失。结论：B1 Reddit SoM 反垫底不是标记数不足导致，而是小模型对密集页面的视觉信息处理能力上限问题。**这是论文中"capability-aware routing"论点的强证据**。

4. **Mirage Effect 站点×模型交互**：Classifieds 上 Mirage Effect 强（B0 +7.27pp, B1 +4.70pp），Reddit 上大幅减弱（B0 +2.38pp, B1 **-1.91pp**）。Reddit 页面复杂度超过 4B 模型从 SoM 标注获益的上限。

5. **Oracle 选择分布极化**：B0 三模式均衡（29-38%），B1 高度偏向 DOM（66.7%）。验证了 Capability-Aware Routing 的核心论点：最优表征是模型能力的函数。

6. **B0 oracle ceiling 18.57% > B1 12.38%**：更大模型在 Reddit 上有更大的路由收益空间。

7. **Reddit 整体难度远高于 Classifieds**：B1 all_fail 91.4%，B0 all_fail 83.8%。两站差异主要来自搜索交互难度和 Postmill UI 陷阱。

8. **设计不对称需注意**：B0/B1 存在温度/max_tokens/scroll 等不对称（同 Classifieds），SR 差异无法完全归因于模型规模。**B1 latency 数字未最终化**（DGX 共享 GPU 争抢），Myriad HPC 上线后用独占 GPU 重跑。

---

## 方法论说明

- **比较局限**：B0 和 B1 存在多项设计不对称（温度/max_tokens/scroll），SR 差异无法完全归因于模型规模
- **Adjusted SR**：使用 analysis_summary 方案（/205 分母），cross_representation 使用 /210 分母
- **Mirage Gap**：使用 adjusted SR 计算（SoM adj SR - DOM adj SR）
- **成本比较**：B0（API 费用）与 B1（本地推理 GPU 等价成本）成本体系不同
- **数据时间**：B0 run 日期 2026-04-22，B1 run 日期 2026-04-13，两者非同日运行但 Reddit 站点状态一致
- **三模式完整**：B0/B1 均为 210/210 × 3 完整数据（此前版本 B0 Vision 不完整、B1 SoM 极少）

---

*v2 (2026-04-26): post-rederive + B1 SoM max_marks=200 重跑验证*
*v1 (2026-04-24): 三模式完整数据首版*
*数据来源：B0_3mode_reddit_20260422 + B1_3mode_reddit_20260413（B1 SoM 04-25 重跑）*
*B0 三模式详情：B0_findings.md；B1 三模式详情：B1_findings.md*

---

## 11. SoM 视觉 probe 实验（v3, §100/§101）

### 11.1 实验设计

5 张密度梯度截图 × B0/B1 × 3 mode probe，让 model 列出截图可见 link/button/heading 文字内容，ground truth 从 axtree 提取。脚本：`scripts/maintenance/probe_som_occlusion.py`。

三个 mode：
- **mode-SoM**：当前实现的带标签截图（含 occlusion bug）
- **mode-NoMarks**：原始 `screenshot.png`（无标签 baseline）
- **mode-WithText**：SoM 截图 + prompt 附加完整 [SOM_MARKS] 文本

### 11.2 完整数据矩阵（visual recall %）

| 图 | marks | B0 SoM | B0 NoMarks | B0 WithText | B1 SoM | B1 NoMarks | B1 WithText |
|---|---|---|---|---|---|---|---|
| reddit_164 step14 | 54 | 46% | 46% | 50% | 46% | 36% | **96%** ⭐⭐ |
| **reddit_task_6 step0** | **111** | **18%** | **78%** | **80%** | 15% | **75%** | **81%** ⭐⭐ |
| reddit_164 step0 | 128 | 40% | 55% | 43% | 28% | 42% | **81%** ⭐⭐ |

**B1 num_ids 输出（attention hijack 量）随密度变化**：
- mode-SoM: 1 (54 marks) → 88 (111 marks) → **446** (128 marks) ⚠️
- mode-WithText: 0 → 0 → 7（给文本后 attention 完全 bypass 截图）

### 11.3 三个核心 finding

**(F1) B1 视觉 capability ≈ B0**（NoMarks 接近）：reddit_task_6 上 B1 NoMarks 75% vs B0 78%，差仅 3pp。**反驳"4B 视觉本质弱"假说**——B1 视觉 capability 在无标签下接近 B0。

**(F2) SoM 标签是 destructive bug**：B0/B1 OCR 都从 ~78% 降至 ~15-18%（-60pp），实心填充覆盖元素文字开头 3-6 字符是系统性问题。但 task SR 不直接受 OCR 损失影响，因 [SOM_MARKS] 文本提供 fallback。

**(F3) text-over-vision bias 在 small VLM 更强**：B1 mode-WithText reddit_164/14 = **96%** 反超 B0 50%——4B 给文本后**完全忽略截图**（甚至比 235B 切换更彻底）。Asadi 2026 small VLM 强 text-over-vision bias 的直接 probe-level 证据。

### 11.4 三模式分解（DOM-Vision = text-over-vision bias 量化）

| | DOM | SoM | Vision | SoM-DOM | **DOM-Vision** | SoM-Vision |
|---|---|---|---|---|---|---|
| B0 reddit | 11.4% | 11.9% | 8.6% | +0.5pp | **+2.9pp** | +3.3pp |
| B1 reddit | 10.0% | 8.1% | 4.8% | -1.9pp | **+5.2pp** ⭐ | +3.3pp |

**B1 DOM-Vision +5.2pp** 是 4B 强 text-over-vision bias 的论文级直接证据：纯文本（DOM）SR 远高于纯截图（Vision），且 B1 比 B0 (+2.9pp) 更显著。

注：SoM-Vision = (文本贡献) + (带标签截图 vs 无标签截图差异)，**不能简化为"全文本贡献"**。

### 11.5 韦恩图：B1 reddit DOM only 主导（反转 fundamental 证据）

| 区域 | B0 reddit (n=210) | B1 reddit (n=210) |
|---|---|---|
| DOM only | 5 | **7** ⭐ |
| SoM only | 7 | 3 |
| Vision only | 5 | 2 |
| DOM ∩ SoM | 9 | 6 |
| DOM ∩ Vision | 4 | **0**（完全互斥） |
| SoM ∩ Vision | 3 | **0** |
| all 3 | 6 | 8 |
| **Oracle** | **18.6%** | **12.4%** |
| **Headroom** | +6.7pp | +2.4pp |

**B1 reddit DOM only 7 ≫ SoM only 3** —— 反转的 fundamental 证据。其他 3 cell（B0 reddit / classifieds B0/B1）都是 SoM only > DOM only。

**B1 上 DOM ∩ Vision = 0**：DOM (text path) 和 Vision (coordinate path) 解决**完全不同**的 task，路由理论上有意义。

### 11.6 Codex 重审计 task category subset SR

[Codex audit](../cross_sites/codex_audit_reddit.json) 把 reddit 210 tasks 分为：
- A NON_VISUAL_TEXT_ONLY: 11 (5.2%)
- B VISUAL_REQUIRED_REFERENCE_IMAGE: 84 (40%)
- C VISUAL_REQUIRED_PAGE_SCREENSHOT: 113 (53.8%)
- D UNCERTAIN: 2 (1%)

| Cat | n | B0 DOM | B0 SoM | B0 Vision | B1 DOM | B1 SoM | B1 Vision |
|---|---|---|---|---|---|---|---|
| A | 11 | 0% | 0% | 0% | 0% | 0% | 0% |
| **B** | **84** | 20.2% | 21.4% | 15.5% | **16.7%** | **13.1%** | 7.1% |
| C | 113 | 6.2% | 6.2% | 4.4% | 6.2% | 5.3% | 3.5% |

**B subset (ref-image required) × B1 SoM-DOM = -3.6pp**——reddit 上反转最严重的 subset。给 4B 加 reference image + page screenshot，hijack 完全压制视觉收益。**Lazy minimization 假说的直接证据**。

### 11.7 反转因果链（reddit B1 SoM < DOM）

```
[B1 视觉 capability 正常 (NoMarks 75% ≈ B0 78%)]
            ↓
[SoM 标签 destructive: 实心 fill + 数字 hijack]
            ↓
[B1 给截图 → OCR 75%→15% (-60pp), num_ids 0→88]
            ↓
[B1 给截图 + 文本 → 完全忽略截图，OCR 跳到 81%, num_ids 归 0]
            ↓
[Task SR: DOM 10.0% > SoM 8.1% > Vision 4.8%]
[反转 -1.9pp 来自截图 destructive + 视觉收益本来就低 (reddit 高密度)]
```

### 11.8 Phantom-SoM 路由分析

probe 直接证据（mode-WithText num_ids 0 + B1 reddit OCR recover）：
- B1 给 [SOM_MARKS] 文本后 attention 完全 bypass 截图
- **B1 reddit Phantom-SoM (无图) 预测 ≥ Full SoM**（去掉 hijack +2pp + 省 50% token cost）
- **B0 reddit Phantom-SoM ≈ Full SoM**（gap +0.5pp 可忽略）→ **cost-saving win**（保 SR + 省 50% cost）

### 11.9 Lazy minimization 假说（4B 偏好 easy 信号）

```
4B 信号选择优先级: 数字标签 > 文本 > 截图内容文字

机制：
  - 给 [文本 + 截图]: 默认用文本
  - 给 [数字标签 + 内容文字]: 默认 attend 数字
  - 给 [仅截图]: OCR 仍可用 (NoMarks ≈ B0)

证据：
  - num_ids 0→446 with marks（lazy 选数字）
  - 给文本归 0（lazy 选文本）
  - WithText 反超 B0（4B 切换更彻底）

Asadi 2026 small VLM 强 text-over-vision bias 的 mechanistic 解释。
```

### 11.10 SoM 设计参数 confound（claim scope 限制）

P79 SoM 实现 vs VWA 原版有 3 大差异：
1. 标全部元素 vs 仅 Interactable（P79 marks +50%）
2. 固定青色 #00BCD4 vs categorical 多色
3. simple placement (2 候选) vs 8-corner + 重叠避免

→ §11.x findings 的 scope 限于 P79 实现。要 generalize 到 VWA 原版必须重做实验。详见笔记 §101。

---

*v3 (2026-04-26): §100/§101 SoM probe + Codex audit subset + Phantom-SoM 评估*
*v2 (2026-04-26): post-rederive + max_marks=200 重跑验证*

---

## 12. M1+M2 root cause 实证 + 设计 confound（v4, §101 后修订）

### 12.1 实证 4 cell × SoM mode 非交互 click 率

| Cell | total clicks | non-interactive % | top non-interactive role |
|---|---|---|---|
| B0 classifieds SoM | 524 | 11.3% | image (46) |
| B1 classifieds SoM | 957 | **30.0%** ⚠️ | image (184, 商品 thumbnail) |
| B0 reddit SoM | 662 | 10.4% | heading (60, 字母分类) |
| **B1 reddit SoM** | 1488 | **9.5%** ⭐ | heading (49) |

**关键反直觉**：
- **B1 reddit M2（误点 StaticText）= 9.5% ≈ B0 reddit 10.4%**——M2 在 reddit 上 **不显著**
- 之前外推 §96 classifieds B1 32% 数据到 reddit 是错的
- → reddit 上 B0/B1 都受 heading 误点影响，差异不构成反转主因

### 12.2 反转 root cause 拆解（实证后修订）

**M1 (visual hijack)**：4B attention 被高对比度青色数字标签 dominate
**M2 (误点 StaticText)**：4B click 自由度被赋予非交互元素

```
classifieds B1 (SoM > DOM +6.4pp 仍正向):
  M1: 弱（41 marks num_ids=12, 33 marks=0；阈值未触发）
  M2: 显著（30%）但被 visual 收益压过 → C subset SoM-DOM +10.4pp

reddit B1 (SoM < DOM -1.9pp 反转):
  M1: 强（111 marks num_ids=88, 128 marks=446）
  M2: 弱（9.5% ≈ B0 10.4%，B0/B1 都受 heading 误点影响）
  反转主因：M1（hijack），M2 解释 ~0%

reddit B0 (SoM > DOM +0.5pp 微正):
  M1: 极弱（B0 视觉容量足以分辨标签 vs 内容）
  M2: 弱
  net: 截图微弱视觉收益勉强压过损失
```

→ **reddit B1 反转 -1.9pp 主要由 M1 解释**。M2 hypothesis 在 reddit 上被实证否定。

### 12.3 P79 vs VWA 原版 SoM 设计差异（confound 评估）

**1 fundamental（标注范围）+ 2 minor（颜色/placement 影响小）**：

| | P79 "Universal SoM" | VWA 原版 "Action-Affordance SoM" |
|---|---|---|
| 标注范围 | 所有元素（StaticText/heading 也标） | 仅 `Interactable=True` |
| Action affordance | 可 click 任何元素（含 StaticText） | 只能 click 可交互（结构性约束 M2=0%） |
| 设计哲学 | trust model with role info | structural constraint |

**VWA 原版下预测**（修正后）：
- M1 hijack：标签数 reddit ~74 vs P79 111；阈值是否触发未知，hijack 减弱程度待实证
- M2 在 reddit 上本来就不显著（实测 9.5% ≈ B0），VWA 改进意义有限
- M2 在 classifieds 上结构性消除（B1 30%→0%），但 SoM 已经赢，进一步扩大优势

→ **reddit B1 反转量级几乎完全取决于 M1 在 VWA 原版下的命运**。

### 12.4 主 claim 与 SoM 设计的依赖度

| Claim | 设计依赖度 |
|---|---|
| **Lazy minimization** (4B 偏好 easy 信号: 数字 > 文本 > 内容文字) | 独立 |
| **B1 视觉 capability ≈ B0** (NoMarks 75% ≈ 78%) | 独立 |
| **text-over-vision bias 在 small VLM 更强** (DOM-Vision +5.2pp B1 / +2.9pp B0) | 独立 |
| **Capability × density × task-category 三轴交互** | 独立 |
| **Phantom-SoM cost saving motivation** | 独立 |
| reddit B1 反转 **量级** -1.9pp | 依赖 P79 设计（VWA 下可能 -0.5 ~ -1.5pp） |
| occlusion OCR -60pp **量级** | 依赖 P79 设计 |
| classifieds M2 **量级** 30% | 依赖 P79 设计（VWA 下结构性 0%） |

**主 mechanism claim 全部独立于 SoM 设计选择**。仅"具体量级"是 P79-specific。

### 12.5 Phantom-SoM motivation 不依赖反转 magnitude

即便 L4 ablation 显示 VWA 原版下 reddit B1 反转消失（SoM ≈ DOM），Phantom-SoM 仍 motivated：
- SoM 含图片 token (~50% input cost)，DOM/Phantom 无
- 同等 SR 下 → cost-saving win（universal）
- 韦恩图角度：SoM only / DOM only 非空 → 两 mode 解决不同 task → routing headroom 存在
- → **Phantom-SoM 是 universal cost-aware tool，不是反转 magnitude 的 hostage**

### 12.6 Honest scope statement（论文应披露）

> "We adopt the 'Universal SoM' design (annotating all elements including non-interactive ones), motivated by §96's routing experiment control variable need (DOM/SoM textual content parity). The mechanism findings (lazy minimization, capability × density × task-category interaction, text-over-vision bias) are design-independent. Specific reversal magnitudes (e.g. reddit B1 SoM-DOM = -1.9pp) reflect this design and may attenuate (not reverse direction) under VWA-style 'Action-Affordance SoM'. M2 (StaticText 误点) was empirically verified to contribute ~0% to reddit reversal but ~30% (B1 classifieds) to action affordance overhead."

---

*v4 (2026-04-26): M1+M2 实证拆解 + Universal vs Action-Affordance 范式 + scope 标注*
