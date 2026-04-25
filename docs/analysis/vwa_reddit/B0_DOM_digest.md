# B0 Reddit -- DOM 模式分析

> B0 = Qwen3-VL-235B-A22B（proxy API），DOM 模式，Reddit 站点
> 210 episodes, 2667 steps, adjusted SR = 7.62% (16/210)
> digest 数据来源：`digest_dom.jsonl`（186 行失败 episode 定性分析）
> 跨模式对比见 `B0_findings.md`

---

## 1. 总览

### 1.1 核心指标

| 指标 | 值 |
|------|-----|
| Raw SR | 11.43% (24/210) |
| Adjusted SR | 7.62% (16/210) |
| FP: N/A | 4 |
| FP: Visual | 1 |
| FP: Eval | 3 |
| 平均步数 | 12.70 |
| 平均成本 ($/ep) | $0.0516 |
| P95 延迟 | 73,618 ms |
| No-op rate | 12.3% |
| Page unchanged rate | 21.1% |
| Cost efficiency ratio | 0.0855 |

### 1.2 失败原因分布

| 失败原因 | 数量 | 占比 |
|----------|------|------|
| fail_finish_eval_mismatch | 50 | 23.8% |
| fail_no_progress | 47 | 22.4% |
| fail_max_steps_search_repeat | 29 | 13.8% |
| fail_incomplete_or_stuck | 14 | 6.7% |
| fail_early_finish | 13 | 6.2% |
| fail_max_steps_click_back_loop | 9 | 4.3% |
| fail_finish_wrong_url_not_found | 9 | 4.3% |
| fail_finish_claim_missing | 7 | 3.3% |
| fail_finish_empty_answer | 6 | 2.9% |
| fail_parse_error | 2 | 1.0% |

**前三大失败原因占 60.0%**：eval_mismatch (23.8%) + no_progress (22.4%) + search_repeat (13.8%)。

---

## 2. Digest 定性分析分类

> 基于 186 行 digest_dom.jsonl 的 category 字段统计。

| 类别 | 数量 | 占比 | 说明 |
|------|------|------|------|
| 搜索循环 | 55 | 29.6% | 重复使用同一搜索词无效 |
| 执行停滞 | 40 | 21.5% | 关键交互未完成，任务流程卡住 |
| 事实推理错误 | 27 | 14.5% | 找错帖子、选错排序、误判内容 |
| 过早结束 | 18 | 9.7% | 未完成关键步骤即 finish |
| 目标不可达 | 17 | 9.1% | 任务本身无法通过 DOM 完成 |
| 答案对齐错误 | 16 | 8.6% | 找到正确内容但答案格式/内容不匹配 |
| 导航循环 | 8 | 4.3% | click-back 循环 |
| 导航失败 | 4 | 2.2% | 未进入目标页面 |
| 综合失败 | 1 | 0.5% | 多种原因叠加 |

### 2.1 搜索循环（55 个，29.6%）-- DOM 最大失败模式

**核心问题**：DOM 模式看不到图片内容，agent 只能通过文字描述搜索目标帖子。但 Reddit 大量任务以图片为线索（"Find this post" + 参考图片），DOM 模式的搜索词只能依赖 agent 对图片 alt text 的推测，导致搜索词不精确。

**典型模式**：
- 重复搜索同一关键词 5-15 次不变（如 task 23 "pumpkin robot" x15, task 30 "colmscomics" x15）
- 搜索关键词过于具体或不准确（如 task 4 "wheat field city skyline" x7）
- DOM 步数长（平均 12.70），更多步骤被浪费在无效搜索上

**与 Classifieds 的差异**：Classifieds 的搜索循环主要是筛选条件不精确（价格区间、类别），而 Reddit 的搜索循环根本原因是**视觉信息缺失** -- DOM 模式从根本上无法解决"根据图片找帖子"的任务。

### 2.2 执行停滞（40 个，21.5%）

Agent 在某个步骤后无法推进任务：
- **click 目标存在但不生效**：连续点击同一元素，页面无变化
- **进入错误子论坛后无法返回**：如 task 28 进入提交页面后陷入 tab_focus 循环
- **缺少空间导航感知**：DOM 模式下 agent 不知道还有多少内容可滚动

### 2.3 事实推理错误（27 个，14.5%）

Agent 对页面内容的理解或推理错误：
- **未切换排序**（如 task 34/35）：任务要求"all time top"或"latest"帖子，agent 停留在默认 Hot 排序
- **选错帖子**：多个帖子中选择了标题相似但不是目标的帖子
- **评论计数错误**：如 task 5，页面显示"13 comments"但参考答案是 121

### 2.4 Element ID 问题

| element_id_issue | 数量 | 占比 |
|-----------------|------|------|
| 否 | 167 | 89.8% |
| 是 | 19 | 10.2% |

19 个 episode 存在 element_id 问题（10.2%），其中：
- element_id 失效：13 个（点击的 element_id 不存在或指向错误元素）
- 视觉信息缺失：5 个（DOM 无法提供图片相关信息）
- 不适用：1 个

### 2.5 脚手架归因

| is_scaffolding_issue | 数量 | 占比 |
|---------------------|------|------|
| 否 | 122 | 65.6% |
| 是 | 64 | 34.4% |

64 个 episode（34.4%）被归因为脚手架/表征缺陷而非纯模型能力问题。主要包括：
- click-back 循环（导航循环 8 个）
- DOM 模式视觉信息缺失导致的目标不可达
- VWA 框架限制（如文件上传不可达）

---

## 3. 高成本失败模式

> 来源：A4b_fail_reason_cost_stats.csv（两模式合并数据）

| 失败原因 | 平均步数 | 平均成本 | P95 延迟 |
|----------|---------|---------|----------|
| fail_max_steps | 30.0 | $0.163 | 367,217ms |
| fail_max_steps_search_repeat | 30.0 | $0.122 | 741,675ms |
| fail_max_steps_click_back_loop | 30.0 | $0.114 | 451,747ms |

**最昂贵的失败**都是达到 max_steps（30步）的情况，平均成本 $0.11-0.16/ep。DOM 模式中 search_repeat（29 个）和 click_back_loop（9 个）合计 38 个 episode 消耗了大量无效成本。

**Wasted cost**：DOM 平均 wasted cost = $0.0472/ep（即失败 episode 的平均成本），占总成本的 91.4%。

---

## 4. Reddit DOM 特色行为

### 4.1 图片描述推测

Reddit 大量任务提供参考图片作为线索。DOM 模式看不到图片，agent 只能根据 alt text 或 page context 推测图片内容。例如：
- task 4: 从 alt text "wheat field city skyline" 推测搜索词
- task 17: 从 page context 推测 "money cycle meme"
- task 22: 从任务描述推测 "mountain dew xbox limited"

这种推测经常不准确，导致搜索循环。

### 4.2 DOM 的排序/筛选能力

与 Classifieds B0 DOM 类似，Reddit DOM agent 也展现了一定的排序切换能力，但使用率较低：
- task 34: 尝试点击 Sort by 但未成功切换到 Top
- 多数任务停留在默认 Hot 排序，未尝试切换

### 4.3 DOM 长步骤倾向

DOM 平均步数 12.70，远高于 SoM 的 8.01。DOM agent 倾向于：
- 更多的搜索尝试（搜索循环达 30 步上限）
- 更长的浏览链（scroll + click 交替）
- 更少的 early_finish（6.2% vs SoM 14.3%） -- DOM agent 不容易过早结束，但也不容易快速找到答案

### 4.4 Comment 自链接死循环（28/210 tasks）

> 分析脚本：`scripts/analysis/analyze_reddit_selflink_cycle.py`，跨 B0/B1 完整数据见 `B1_DOM_digest.md F5`。

Postmill 帖子页面的 "N comments" 链接指向当前页面自身，Agent 反复点击但 URL 不变。B0 DOM 中 **28/210 tasks（13.3%）** 存在此模式，少于 B1 的 36 个。

**逃出方式对比（B0 vs B1）**：

| 逃出方式 | B0 (235B) | B1 (4B) |
|---------|-----------|---------|
| **scroll** | **5** | 3 |
| type | 3 | 4 |
| finish | 4 | 3 |
| navigate_away | 3 | 6 |
| **截断（未逃出）** | 12 (42.9%) | **19 (52.8%)** |

**B0 的关键优势在于 scroll 逃出能力**：5 次 scroll 逃出中 1 次成功（task 72：scroll → type comment → finish），B1 的 3 次 scroll 逃出全部失败。B0 循环后平均 4.5 步即切换策略，B1 需要 5.2 步。

**典型对比**（task 72）：B0 前 4 步与 B1 行为相同（反复点击 "6 comments"），但 B0 在 step 4 主动 scroll down（confidence 降到 0.7，thought: "scroll down to locate the comment input area"），成功找到 textarea 并输入评论。B1 则 9 步全部点击，始终未 scroll。

---

## 5. 与 SoM 对比

### 5.1 模式对比总结

| 维度 | DOM | SoM |
|------|-----|-----|
| Adjusted SR | 7.62% | **10.48%** |
| 平均步数 | 12.70 | **8.01** |
| 平均成本 | $0.0516 | **$0.0384** |
| 主要失败模式 | 搜索循环 (29.6%) | 执行停滞 (41.8%) |
| 视觉任务能力 | 弱（无图片信息） | 中（有截图但匹配有限） |
| 搜索循环率 | **29.6%** | 11.5% |
| 过早结束率 | 9.7% | **22.0%** |

### 5.2 DOM 独占成功任务

DOM 独占 6 个 task（adjusted），全部为 single_navigation 类型。这些任务可能是：
- 不依赖图片的纯文字导航任务
- DOM AXTree 的文字搜索精度在特定任务上优于 SoM 的视觉定位

---

*更新时间：2026-04-23*
*数据来源：B0_3mode_reddit_20260422 analysis/digest/digest_dom.jsonl*
