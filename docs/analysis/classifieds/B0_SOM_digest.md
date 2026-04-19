# B0 SoM Baseline 分析报告（Classifieds）

> B0 = Qwen3-VL-235B-A22B（proxy API），SoM 模式，classifieds 站点
> 对应 B1 分析见 `B1_SOM_digest.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`
>
> 数据来源：`phase1_som_router_0`，classifieds 全部 234 tasks
> 分析方法：自动化 post analysis（condition_summary_v2 + reason_diagnostics + cross_representation）
> **本版数据含 parse_error 修复后重跑结果（2026-04-17）**

---

## 一、总体概况

| 指标 | 数值 |
|------|------|
| SoM condition 总 episode 数 | 234 |
| 成功（raw） | 57（24.36%） |
| N/A FP | 9（10 个 N/A reference task 中 9 个误判） |
| Visual FP | 0（SoM 有截图，无 visual lucky hits） |
| **Adjusted SR** | **21.43%**（48/224） |
| 平均步数 | 7.25 步 |
| 平均成本 | $0.0355 / episode |
| 平均无效步率（no-op） | 7.7% |
| 平均页面无变化率 | 26.5% |
| 早停触发分布 | action_failed: 113, page_unchanged_streak: 40, no_progress_streak: 40 |
| Bootstrap 95% CI | [15.38%, 25.64%]（raw SR） |

### 与 DOM / Vision 对比（B0 三模式）

| 指标 | DOM | **SoM** | Vision |
|------|-----|---------|--------|
| Raw SR | 14.96% | **24.36%** | 15.38% |
| Adjusted SR | 8.04% | **21.43%** | 11.61% |
| 平均步数 | 12.25 | **7.25** | 7.67 |
| 平均成本 | $0.0467 | $0.0355 | **$0.0241** |
| parse_error 数 | 1 | 12 | 2 |
| SoM vs DOM McNemar p | — | **5.3e-6 ★** | — |
| SoM vs Vision McNemar p | — | **1.1e-4 ★** | — |

B0 三模式中，**SoM adjusted SR 最高（21.43%），显著优于 DOM 和 Vision**（McNemar p<0.001）。Vision vs DOM 差异不显著（p=0.152）。

### 与 B1 SoM 对比

| 模型 | Raw SR | Adjusted SR |
|------|--------|-------------|
| **B0 (235B)** | **24.36%** | **21.43%** |
| B1 (4B) | 20.51% | 16.24% |

**parse_error 修复后，B0 SoM 以 +5.19pp 领先 B1 SoM**。之前观察到的"SoM 反转"（B0 < B1）完全由 parse_error 导致（旧数据 45 例 20.1%，修复后仅 12 例 5.17%），证实了 §五 假说 A。235B 模型在 SoM 多模态输入下确实更强。

---

## 二、失败原因分布

| 失败原因 | 数量 | 比例 | 备注 |
|---------|------|------|------|
| **fail_finish_wrong_url_not_found** | **54** | **23.3%** | ★ 最大失败源 |
| fail_early_finish | 34 | 14.7% | SoM 过度自信早停 |
| success | 57 | 24.6% | (raw) |
| fail_finish_eval_mismatch | 26 | 11.2% | 评测不一致 |
| fail_no_progress | 24 | 10.3% | 连续无进展 |
| fail_parse_error | 12 | 5.2% | 残留 parse error |
| fail_max_steps_target_unreachable | 8 | 3.4% | 目标不可达 |
| fail_finish_claim_missing | 5 | 2.2% | 声称缺失 |
| fail_finish_empty_answer | 5 | 2.2% | 空答案 |
| fail_max_steps_click_back_loop | 3 | 1.3% | click-back 循环 |
| 其他 | 4 | 1.7% | search_repeat / max_steps / summary_error / left_target |

### 与旧数据（parse_error 修复前）对比

| 失败原因 | 旧数据 | 新数据 | 变化 |
|---------|--------|--------|------|
| fail_parse_error | 45（20.1%） | 12（5.2%） | **↓14.9pp** |
| success | 37（15.8%） | 57（24.4%） | **↑8.6pp** |
| fail_early_finish | 12（5.4%） | 34（14.7%） | ↑9.3pp（暴露出来的真实失败） |
| fail_finish_wrong_url | 29（12.9%） | 54（23.3%） | ↑10.4pp |

parse_error 大幅下降后，之前被掩盖的 **fail_early_finish 和 fail_finish_wrong_url** 成为新的头号问题。这些不是新增的失败，而是之前被 parse_error 截断的 episode 现在能够正常执行、暴露出 agent 层面的真实失败模式。

---

## 三、核心失败模式详析

### 3.1 fail_finish_wrong_url_not_found（54 例，23.3%）

最大失败源。Agent 执行了操作但 finish 时停在错误的页面/URL。

- 常见于 `single_navigation + url_match` 任务：agent 找到了类似商品但非 eval 要求的精确目标
- SoM 截图给 agent "视觉确认"感，使其更容易在相似页面上 finish
- 对比 DOM（46 例，19.8%）：SoM 稍高，因为 agent 更果断、更少探索

### 3.2 fail_early_finish / SoM 过度自信早停（34 例，14.7%）

SoM 模式下 agent 过早 finish，且 confidence 往往很高（0.8-0.95）。核心机制是**"信息充分幻觉"**：SoM 截图+标注给 agent "已看到所有信息"的错觉，在视觉属性查询任务中尤为严重。

**典型案例：**

**Task 17**（cheapest bike with **red handlebars**, $900-950，eval=url_match）：
- Step 0 搜索 `bike`（太宽泛），Step 1-2 设价格区间，Step 3 scroll，Step 4 点进一个商品看了看，Step 5 back，Step 6 又点进**同一个商品**（id=68328），Step 7 back
- Step 9：conf=0.8，finish 声称 "No bike with red handlebars is available"
- **对比 DOM 模式**：搜索 `red handlebars bike`（更精准的关键词）→ 直接找到 Cannondale Six13（id=79747, $900）→ **成功**
- **失败根因**：SoM 截图让 agent 产生"我已看到了所有商品图片，没有红把手"的幻觉，仅查看了 2 个商品就放弃。DOM 模式无图可看，反而用更精确的文字搜索找到了目标。

**Task 18**（most recently posted **blue** iPhone，eval=url_match）：
- Step 0 搜索 `blue iPhone`，结果显示无关商品（沙发、灯具等）
- **仅 scroll 2 次**（Step 1-2），Step 3 即 conf=0.9 finish："No blue iPhone listings found"
- 未尝试：分类筛选（Cell phones）、换关键词（`iPhone blue`）、翻页
- **三模式全败**（DOM 5 步 / Vision 5 步也失败），但 SoM 放弃最快（4 步 vs DOM/Vision 5-6 步）

**Task 19**（most recently posted painting showing the sea, Arts+crafts, $80-100，eval=url_match）：
- Step 0 选 Arts+crafts 分类，Step 1 搜索 `sea painting`
- **Step 2 即 conf=0.95 finish**："No painting showing the sea within the $80-100 price range was found"
- 仅 3 步、**零次 scroll**、**零次翻页**就声称搜索穷尽
- DOM 模式 7 步、Vision 6 步后才放弃（虽然也失败，但至少做了更多探索）

**Task 21**（newest RVs+campers listing in a **dark color**，eval=url_match）：
- Step 0 选 "Rvs + campers" 分类，Step 1 搜索，Step 2 直接点第一个结果（1997 Ford Airstream RV，白色车身）
- Step 3：conf=0.95 finish，声称"I have successfully navigated to the newest RVs + campers listing"
- **完全忽略"dark color"条件**：截图中 Airstream 明显是白色/银色，agent 看到了图但没有核验颜色约束就结束
- 三模式全败（DOM 11 步 / Vision 5 步），但 SoM 最激进（4 步，conf=0.95）

**Task 22**（How many miles does the **red car** in the **second row** have?，eval=string_match）：
- start_url 直接进入 Cars+trucks gallery 视图（3×N 网格缩略图）
- **Step 0 即 conf=0.9 finish**：thought 称"已成功导航到红色车辆详情页"，answer="120,000 miles"
- 实际上 agent **从未点进任何商品**，仍在 gallery 列表页。agent **在缩略图列表页就编造了里程数**，从未打开详情页查看
- 这是**零步探索即 finish 的极端案例**：SoM 截图 gallery 视图看到红色车，agent 幻觉自己"已导航到详情页"

**共同模式**：

| 维度 | SoM 早停 | DOM/Vision |
|------|---------|------------|
| finish confidence | 0.8-0.95（极高） | 较低或不 finish |
| 探索步数 | 0-4 步 | 5-11 步 |
| 搜索策略多样性 | 低（1 种搜索词） | 略高（Task 17 DOM 用更精准词） |
| 翻页 | 0-1 次 | 1-2 次 |
| 视觉约束核验 | **跳过**（Task 21/22 未检查颜色） | DOM 无图不会幻觉确认 |

**根因分析**：SoM 模式同时提供截图和元素标注，agent 形成了"视觉确认完成"的捷径。具体表现为两种子模式：

1. **"看了没找到"型早停**（Task 17/18/19）：搜索结果截图中看不到目标视觉属性 → 直接判定不存在，不尝试更多搜索策略。DOM 模式无图可看，反而用更精确的文字搜索。
2. **"看到了就是了"型早停**（Task 21/22）：截图中看到似乎匹配的商品 → 不验证约束条件就 finish，甚至幻觉自己已进入详情页。Task 22 的 0 步 finish 是最极端案例。

这是 SoM 的 **text-over-vision 反转**：截图不是帮助 agent 更好决策，而是提供了一个"快速放弃"或"快速确认"的锚点。

### 3.3 fail_no_progress（24 例，10.3%）

Agent 连续多步无进展被 cycle detection 截断。

- 对比 DOM（61 例，26.3%）：SoM 远低，因为 SoM 截图帮助 agent 更快定位交互元素
- 典型表现：反复点击 combobox、重复搜索相同关键词、select_option 循环（见 §3.5）

### 3.4 fail_parse_error（12 例，5.2%）

修复后的残留 parse error。仍是 SoM 独有的偏高现象（DOM 1 例，Vision 2 例）。

- **根因**：SoM 是唯一同时传**长文本（SOM_MARKS 列表）+ 图像**的模式
- temperature=0.1 在复杂多模态输入下偶尔导致 JSON 格式出错
- 12 例中大部分可通过 GLM 后处理修复，但仍有少数无法恢复

### 3.5 select_option 重复选择循环

**Task 9**（发布 iPhone 13 mini pink listing）：
- Step 7：`select_option` eid=3152 "Cell phones" → **成功**，DOM 已更新 `currently selected="Cell phones"`
- Step 8-9：对同一 combobox 重复 `select_option` "Cell phones" → **失败**（值已是 "Cell phones"，DOM 无变化）→ cycle early stop

**Task 8**（发布 Nintendo Switch listing）：
- Step 4：`select_option` eid=3134 "Video gaming" → **成功**
- Step 6：`click` eid=3152 → 失败（combobox 不响应 click）
- Step 10/12/13：反复 `click` eid=3134 → 失败（想离开发布页搜索价格，但错误地点击 category 下拉）→ cycle early stop

**共同根因**：

1. **Agent 不理解 `currently selected` 的语义**。SoM 观察中明确显示 `[OPTIONS: currently selected="Cell phones"]`，agent 读到了但不理解"已选中 = 不需要再选"。
2. **History 中 `select_option` 不显示 option_label**。`_format_history()` 对 select_option 没有特殊处理，输出为 `Step 7: select_option [id=3152] -> OK`，缺少 `"Cell phones"` 这个关键信息。agent 无法从 history 确认"选了什么"。
3. **规划能力缺失**。Task 8 中 agent 完成 category 选择后需要去搜索类似商品定价，但发布页 SoM marks 中没有搜索栏/导航入口，agent 不知道用 `back` 返回首页再搜索。

**与 P8（§54 select 反馈缺失）的关系**：P8 描述的是修复前 DOM 不反映 `currently selected` 变化。这里 DOM 已正确反映（§54 已修），但模型仍然无法利用该信息。属于**模型能力缺陷**而非脚手架 bug。

**潜在改进**：`_format_history()` 中为 `select_option` 添加 option_label 显示（`Step 7: select_option [id=3152] "Cell phones" -> OK`），可增强反馈信号，但不保证 4B 模型能利用。

### 3.6 "自信声明找不到" 跨模式定量分析（49 例，27.1% of failed）

对所有 SoM 失败 episode 的最终 finish answer/thought 进行正则匹配（`not found`/`no results`/`does not exist`/`cannot be completed` 等 20+ 模式），统计 agent **主动声明目标不存在**后放弃的比例。

#### 三模式对比

| 模式 | 失败数 | 声明找不到 | 占失败比 | 占全部比 |
|------|--------|-----------|---------|---------|
| DOM | 199 | 9 | **5%** | 3.8% |
| **SoM** | 177 | **49** | **28%** | **20.9%** |
| Vision | 198 | 29 | 15% | 12.4% |

SoM "错误放弃"率是 DOM 的 **5.6 倍**，Vision 的 **1.9 倍**。49 例中 **0 例是 N/A 任务**（N/A 任务的正确 finish 不匹配此模式），全部为模型错误判断。

#### 放弃时机

- **早期放弃（≤3 步）：28/49 = 57%** — 多数连搜索策略都没换就认输
- 晚期放弃（>3 步）：21/49 = 43% — 搜索了一圈、换了 1-2 次关键词后认输

#### Confidence 分布

- 高自信（≥0.8）：**38/49 = 78%**
- 中自信（0.5-0.8）：7/49 = 14%
- 低自信（<0.5）：3/49 = 6%

绝大多数放弃时 agent 非常自信（0.8-0.95）——不是"不确定所以放弃"，而是"确信不存在所以放弃"。

#### 其他模式能否救回？

| 救回方式 | 数量 | 比例 |
|---------|------|------|
| DOM 成功 | 3/49 | 6% |
| Vision 成功 | 1/49 | 2% |
| 至少一个救回 | 4/49 | 8% |
| **三模式全败** | **45/49** | **92%** |

92% 的任务三模式全部失败——这些任务本身就很难（多涉及视觉属性+搜索筛选组合），SoM 不是唯一的失败点，但 **SoM 放弃最快、最自信**。

#### 机制解释

SoM 截图上叠加了大量彩色标注框，**挤压了模型对页面内容（尤其是缩略图）的视觉理解能力**。在需要判断"搜索结果中是否有匹配视觉条件的物品"时（颜色、图片内容等），SoM 标注反而干扰了模型对缩略图的识别，导致更快得出"找不到"的结论。DOM 模式靠纯文本信息不涉及视觉判断，错误放弃率最低。

此模式与 §3.2 的"看了没找到型早停"机制相同，本节提供了跨模式的定量证据。

---

## 四、交叉分析（三模式完整）

### 4.1 SoM 独有成功（24 tasks）

三模式 adjusted 集合分析：

| 集合 | 数量 | 占比 |
|------|------|------|
| only_som | 24 | 10.3% |
| som+vision (not dom) | 13 | 5.6% |
| all_success | 8 | 3.4% |
| dom+som (not vision) | 3 | 1.3% |

SoM 独有成功 24 tasks（single_nav:17, page_reading:6, grid:1）。另有 13 个 task 与 Vision 共享成功（可降级到更便宜的 Vision）。

### 4.2 Oracle 路由分析

| 指标 | Raw | Adjusted |
|------|-----|----------|
| Oracle ceiling | 32.05% (75 tasks) | **24.79%** (58 tasks) |
| 最佳单模式 (SoM) | 24.36% | 20.51% |
| Routing headroom | 7.69pp | **4.27pp** |
| Oracle 选择分布 | SoM:28, Vision:28, DOM:19 | **SoM:27, Vision:21, DOM:10** |

Oracle 中 SoM 贡献 27/58 adjusted 成功（46.6%），Vision 21/58（36.2%），DOM 10/58（17.2%）。SoM 在 single_navigation 类型主导（20/42），Vision 在 page_reading 类型主导（14/31）。

### 4.3 成本对比（8 tasks 三模式均成功）

| 模式 | 平均成本 | 中位成本 | 平均步数 | 最便宜次数 |
|------|---------|---------|---------|-----------|
| DOM | $0.028 | $0.014 | 7.38 | 2/8 |
| SoM | **$0.016** | $0.017 | **3.38** | 3/8 |
| Vision | $0.025 | $0.018 | 7.13 | 3/8 |

SoM 步数最少（3.38），三模式成本差异均不显著（Wilcoxon p>0.4）。

---

## 五、共性脚手架缺陷（与 B1 SoM 相同）

以下问题在 B1 SoM 中已记录，B0 再次确认：

- **`<select>` 下拉菜单三层不可达**（VWA 框架级缺陷，见 §3.5 案例分析）
- **confirm 弹窗不可交互**：Delete 操作被取消，所有删除任务均失败
- **N/A 任务 False Positive（9/10）**：B0 N/A 任务 9 个误判 success=1.0，机制与 B1 相同（Agent prompt 无 N/A 出口）
- **极少翻页**：SoM 模式下 agent 仍以 scroll 为主，分页导航依赖程度低

---

## 六、B0 SoM 效率分析

| 指标 | B0 SoM | B0 DOM | B0 Vision |
|------|--------|--------|-----------|
| 平均步数 | 7.25 | 12.25 | 7.67 |
| 平均成本/ep | $0.0355 | $0.0467 | **$0.0241** |
| cost_efficiency_ratio | **0.186** | 0.102 | 0.140 |
| P95 延迟 | 74,042ms | 10,537ms | 50,540ms |

SoM 步数（7.25）远少于 DOM（12.25），接近 Vision（7.67）——SoM 截图使 agent 决策更快。SoM 的 cost_efficiency_ratio（SR/cost）是三模式中最高的。但 P95 延迟显著高于 DOM（SoM 图文混合请求在 proxy API 上更耗时）。

---

## 七、parse_error 修复前后对比总结

| 指标 | 修复前 | 修复后 | 变化 |
|------|--------|--------|------|
| parse_error 率 | 20.1%（45 例） | 5.2%（12 例） | **↓14.9pp** |
| Raw SR | 15.81% | **24.36%** | **↑8.6pp** |
| Adjusted SR | 12.05% | **21.43%** | **↑9.4pp** |
| B0 vs B1 SoM | B0 < B1（-4.19pp） | **B0 > B1（+5.19pp）** | 反转 |
| SoM 在三模式中排名 | 第一（但优势微弱） | **第一（显著领先）** | McNemar p<0.001 |

**结论**：之前报告的"SoM 反转"完全由 parse_error 导致。修复后，B0 235B 在所有三种观测模式下均优于 B1 4B，符合模型规模预期。SoM 是 B0 Classifieds 的最优表征。

---

*更新时间：2026-04-18*
*数据来源：B0_3mode_classifieds_20260413 phase1_som_router_0（parse_error 修复后重跑数据）*
*B0 三模式定量对比见 `B0_findings.md`；B0 vs B1 跨模型对比见 `B0_B1_findings.md`*
