# B1 Reddit Findings

> Reddit B1 三模式跨模式对比分析（DOM vs Vision，SoM 仅 3/210 不计入）
> 数据来源：`B1_3mode_reddit_20260413`

---

## 一、总体 SR 对比

### analysis_summary 口径（主指标）

| 模式 | Raw SR | Adjusted SR | FP 扣除 | Adjusted 分母 |
|------|--------|------------|---------|-------------|
| DOM | 10.00%（21/210） | **5.85%（12/205）** | 9（NA 5 + Visual 2 + Eval 2） | 205 |
| Vision | 5.78%（10/173） | 2.98%（5/168） | 5（NA 5） | 168 |

**DOM 领先 Vision +2.87pp（adjusted）**。但 McNemar p=0.648，差异不显著。

### cross-rep 口径（/204，用于 oracle 分析）

| 模式 | Raw SR | Adjusted SR |
|------|--------|------------|
| DOM | 10.00%（21/210） | 5.71%（12/210） |
| Vision | 4.76%（10/210） | 1.43%（3/210） |

### FP 分解

| FP 类型 | DOM | Vision |
|---------|-----|--------|
| N/A FP | 4† | 5 |
| Visual FP | 1† | 0 |
| Eval FP | 2 | 2 |
| **Total** | **7** | **7** |

†cross-rep 口径。analysis_summary 口径 DOM NA FP=5, Visual FP=2。DOM Eval FP 2 个为 task 69/72（§88 program_html 补充规则）。

---

## 二、统计检验（173 共同 task）

| 检验 | 统计量 | p 值 | 显著? |
|------|--------|------|-------|
| McNemar（SR） | 8.0 | 0.648 | 否 |
| Wilcoxon（成本） | 1,026 | 6.7×10⁻²³ | **是** |
| Wilcoxon（延迟） | 4,657 | 1.4×10⁻⁵ | **是** |

McNemar 配对矩阵：

|  | Vision ✓ | Vision ✗ |
|--|----------|----------|
| **DOM ✓** | 2 | 11 |
| **DOM ✗** | 8 | 152 |

**结论**：SR 差异不显著，但 DOM 成本和延迟显著更高。

---

## 三、效率对比

| 指标 | DOM | Vision | 倍数 |
|------|-----|--------|------|
| 平均步数 | 16.64 | 6.59 | 2.5× |
| 平均成本 | $0.054 | $0.014 | 3.9× |
| p95 延迟 | 87,926ms | 46,378ms | 1.9× |
| 平均能耗 | 3.76 mWh | 1.39 mWh | 2.7× |
| No-op rate | 17.1% | 38.9% | — |
| Page unchanged rate | 20.9% | 39.3% | — |

DOM 更贵的原因：步数多（AXTree 提供有效 action，不易触发早停），更多 episode 跑满 30 步（search_repeat 48 + click_back_loop 20 + max_steps 3 = 71 个 episode）。

---

## 四、失败模式对比

| 原因 | DOM | DOM % | Vision | Vision % |
|------|-----|-------|--------|----------|
| fail_no_progress | 50 | 23.8% | 94 | **54.3%** |
| fail_max_steps_search_repeat | **48** | **22.9%** | 1 | 0.6% |
| fail_finish_eval_mismatch | **29** | **13.8%** | 2 | 1.2% |
| fail_incomplete_or_stuck | 20 | 9.5% | 39 | 22.5% |
| fail_max_steps_click_back_loop | **20** | **9.5%** | 1 | 0.6% |
| fail_early_finish | 9 | 4.3% | **19** | **11.0%** |
| fail_finish_empty_answer | 9 | 4.3% | 4 | 2.3% |
| fail_max_steps | 3 | 1.4% | 2 | 1.2% |
| success | 21 | 10.0% | 10 | 5.8% |

**DOM 特征**：search_repeat（48）和 click_back_loop（20）是 DOM 独有的大类——AXTree 使 agent 能持续产出语法正确 action，不触发早停，但陷入无效循环。

**Vision 特征**：no_progress（54.3%）和 early_finish（11.0%）主导——action 执行失败率高，快速停滞或直接放弃。

---

## 五、Oracle 路由分析

### Exclusive sets（adjusted, /204）

| 集合 | 数量 | 占比 |
|------|------|------|
| all_fail | 191 | 93.6% |
| only_dom | 10 | 4.9% |
| only_vision | 3 | 1.5% |
| both_success | **0** | **0%** |

**Adjusted 后零交集**——DOM 和 Vision 的 adjusted 成功完全不重叠。

### Oracle headroom

| 指标 | Raw | Adjusted |
|------|-----|----------|
| Best single（DOM） | 10.00% | 5.71% |
| Oracle ceiling | — | 7.14%（15） |
| Headroom | — | **1.43pp** |

Oracle 选择（adjusted）：DOM 12 / Vision 3。DOM 占 80.0%。

### 与 Classifieds 的 oracle 对比

| 指标 | Classifieds B1 | Reddit B1 |
|------|---------------|-----------|
| Oracle ceiling（adjusted） | — | 7.14% |
| Routing headroom（adjusted） | — | 1.43pp |
| Intersection（adjusted） | — | 0 |

Reddit B1 的 headroom（1.43pp）远小于 Classifieds B0（8.55pp），说明 Reddit 的两种模式互补性更弱——绝大多数 task 两种模式都失败（92.9%）。

---

## 六、Reddit 特有问题

### Postmill UI 陷阱

1. **Comment 自链接死循环**（DOM F5）：帖子页面的 "N comments" 链接指向自身，agent 不理解"已到达目标"，反复点击。
2. **Image link trap**（Vision F2）：帖子标题和缩略图链接指向原图而非讨论页，只有 "N comments" 小字链接通向帖子页面。Vision 模式受此影响更大（18 tasks）。
3. **密集分类列表坐标偏移**（Vision F2a）：`/forums/all` 多列字母序列表，行间距 ~20px < 4B Y 轴误差 ~40-60px，Vision 在此布局上命中率仅 18%。

### Reddit vs Classifieds 站点差异

| 维度 | Classifieds | Reddit |
|------|------------|--------|
| 站点结构 | 分类→列表→详情 | 论坛→帖子→评论 |
| 搜索难度 | 低（结构化标题） | **高（自由标题）** |
| 链接结构 | 简单（标题→详情） | **复杂（标题→原图 vs comments→帖子）** |
| Visual task 比例 | 67%（162/234） | ~84%（177/210）† |
| DOM adjusted SR | 8.48% | **5.85%** |
| Vision adjusted SR | — | 2.98% |

†DOM condition 中 visual_tasks=177/210。

---

## 七、关键发现

1. **DOM 是 Reddit B1 最优模式**（adjusted 5.85% vs 2.98%），但差异不显著（p=0.648），且 DOM 成本 3.9× 于 Vision
2. **Reddit B1 整体 SR 极低**：92.9% 的 task 两种模式均失败，oracle ceiling 仅 7.14%（adjusted）
3. **Adjusted 后零交集**：DOM 和 Vision 的成功完全不重叠，oracle headroom 仅 1.43pp
4. **搜索循环是 DOM 的标志性问题**（48/210=22.9%），Vision 的标志性问题是 no_progress（54.3%）和 click-not-type（41.2%）
5. **Reddit 对 4B 模型极具挑战**：相比 Classifieds（DOM adjusted 8.48%），Reddit DOM 5.85% 更低，两站差异主要来自搜索交互难度和 Postmill UI 陷阱
6. **模式互补性弱**：Reddit 的 oracle headroom（1.43pp）远小于 Classifieds B0（8.55pp），routing 潜力有限

---

*生成时间：2026-04-21*
*数据来源：B1_3mode_reddit_20260413，DOM 210/210，Vision 173/210（`_synthesized`），SoM 3/210（不计入）*
