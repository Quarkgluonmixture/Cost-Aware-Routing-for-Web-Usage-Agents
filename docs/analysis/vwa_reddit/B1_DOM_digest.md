# B1 Reddit DOM Digest

> 数据来源：`phase1_dom_router_0`，Reddit 全部 210 tasks
> 分析方法：自动化 post analysis + 逐 episode 轨迹审阅（关键 case 深读）
> 本报告**仅分析 DOM 模式**（纯 AXTree，无截图 / SoM 标注）。
> 三模式共性缺陷与定量对比待 SoM 完成后见 `B1_findings.md`。

---

## 一、总体概况

| 指标 | 数值 |
|------|------|
| 已完成 episode | 210 / 210 |
| Raw SR | 10.00%（21/210） |
| Adjusted SR | 5.85%（12/205） |
| 全量平均步数 | 16.64 步（总 3,495 步） |
| 平均成本 | $0.054 / episode |
| 平均 token | 56,037 / episode |
| p95 步延迟 | 87,926ms（~88s） |
| 平均能耗 | 3.76 mWh / episode |
| Bootstrap 95% CI | [3.33%, 10.00%]（raw） |
| Evaluator error | 3（benchmark noise，详见第六节） |

### Adjusted SR 方法

| 扣除类型 | 数量 | 涉及 task |
|---------|------|----------|
| N/A FP | 5 | 7, 26, 31, 39, 182 |
| Visual FP | 2 | 36, 160 |
| Eval FP | 2 | 69, 72（program_html, click-only, 旧评论碰巧匹配） |
| **合计** | **9** | Raw 21 → Adjusted 12 |

**分母调整**：移除 5 个 N/A reference task → 210 − 5 = 205。

### 早停触发分布

| 触发条件 | 触发次数 |
|---------|---------|
| action_failed | 418 |
| page_unchanged_streak | 133 |
| no_progress_streak | 133 |
| dom_size_exceeds_threshold | 20 |
| text_length_high | 19 |

`action_failed` 是最主要的早停信号。`page_unchanged_streak` 和 `no_progress_streak` 数量一致（133），说明两者几乎同时触发——DOM 模式中页面不变 ≈ 无进展。

---

## 二、成功案例分析（21 raw / 12 adjusted）

### Oracle 成功 task 列表（cross-rep 口径，15/204）

| task_id | eval_type | visual/reasoning | 步数 | 成本 | succeeded_modes |
|---------|-----------|-----------------|------|------|----------------|
| 0 | url_match | medium/easy | 6 | $0.021 | dom |
| 6 | string_match | medium/hard | 3 | $0.012 | dom |
| 7 | string_match | easy/hard | 4 | $0.009 | dom\|vision |
| 18 | url_match | easy/medium | 28 | $0.105 | dom |
| 26 | string_match | medium/hard | 4 | $0.009 | dom\|vision |
| 31 | string_match | medium/hard | 5 | $0.012 | dom\|vision |
| 36 | url_match | medium/medium | 7 | $0.022 | dom |
| 39 | string_match | easy/medium | 5 | $0.020 | dom\|vision |
| 40 | url_match | medium/medium | 2 | $0.010 | dom |
| 42 | url_match | medium/medium | 4 | $0.013 | dom |
| 58 | string_match | medium/medium | 6 | $0.015 | dom |
| 69 | program_html | medium/easy | 3 | $0.005 | dom\|vision |
| 72 | program_html | medium/easy | 3 | $0.005 | dom\|vision |
| 100 | page_image_query | easy/easy | 30 | $0.075 | dom |
| 129 | program_html | easy/easy | 7 | $0.025 | dom |

**注**：此为 cross-rep 口径（15/204），condition_overview 的 21/210 中还包含 6 个仅在 DOM 完整 run 中存在的成功 task。

### SR 按 task 类型分解（cross-rep 口径，/204）

| 类型 | DOM raw | DOM adjusted | Vision raw | Vision adjusted |
|------|---------|-------------|-----------|----------------|
| single_navigation | 10.95%（15/137） | 7.30% | 5.73%（9/157） | 1.27% |
| page_reading | 0%（0/5） | 0% | 6.25%（1/16） | 6.25% |
| action_on_item | —（0/0） | — | 0%（0/3） | 0% |

DOM 优势集中在 single_navigation。page_reading 仅 Vision 有 1 例成功（task 201, url_match）。

### 成功特征

- **url_match 和 string_match 主导**：15 个成功中 url_match 5 个、string_match 6 个、program_html 3 个、page_image_query 1 个。
- **步数两极分化**：多数成功在 2-7 步完成，但 task 18（28 步）和 task 100（30 步，跑满）是低效成功。
- **6/15 与 Vision 重叠**：tasks 7/26/31/39/69/72 两种模式均成功。

---

## 三、失败模式详解

### 失败原因分布

| 原因 | 数量 | 占比 |
|------|------|------|
| fail_no_progress | 50 | 23.8% |
| fail_max_steps_search_repeat | 48 | 22.9% |
| fail_finish_eval_mismatch | 29 | 13.8% |
| fail_incomplete_or_stuck | 20 | 9.5% |
| fail_max_steps_click_back_loop | 20 | 9.5% |
| fail_early_finish | 9 | 4.3% |
| fail_finish_empty_answer | 9 | 4.3% |
| fail_max_steps | 3 | 1.4% |
| fail_finish_wrong_url_not_found | 1 | 0.5% |

**与 Vision 对比**：DOM 的 search_repeat（48 vs 1）和 click_back_loop（20 vs 1）远多于 Vision。DOM 更容易产出有效 action（解析成功率高），"坚持更久"导致更多 30 步跑满的昂贵失败。Vision 的 no_progress（94 vs 50）和 early_finish（19 vs 9）更多——action 执行成功率低导致快速停滞。

### F1. 搜索循环 search_repeat（48 tasks, 22.9%）

Agent 反复使用相同或极相似的搜索词，30 步全部用完。Reddit 帖子标题长且非结构化，搜索结果杂乱，4B 模型难以从失败搜索中提取有效新关键词。

**典型模式**：Task 2 — 搜索 "fbi mobile command c" 11 次，每次得到相同结果页但不改变策略。

**与 Classifieds 的巨大差异**：Classifieds 仅 3 个 search_repeat（1.3%），Reddit 高达 48 个（22.9%）。Classifieds 的商品标题短且标准化（如 "2019 Toyota Camry"），搜索更容易命中；Reddit 的帖子标题更自由、更长，搜索需要更精准的关键词提取能力。

### F2. No progress（50 tasks, 23.8%）

最大失败类别。Agent 执行了多步操作但未产生有效进展，被 no_progress_streak / page_unchanged_streak 触发早停。平均约 7-8 步。包含：

- 点击无效元素导致页面不变
- 在同一页面反复尝试不同操作但都未推进任务
- 与 F3（click_back_loop）不同之处：无明显的"前进→后退"循环，而是"原地打转"

### F3. Click-back loop（20 tasks, 9.5%）

Agent 点击某链接 → 发现不对 → back → 再点同一链接的循环。30 步全部用完，是最昂贵的失败模式。

### F4. Finish-eval mismatch（29 tasks, 13.8%）

Agent 调用 finish 并提交答案，但答案不正确。说明 agent 自认为完成了任务但实际答错或导航到了错误页面。29 个中包含：

- 答案内容错误（如计数错误、选错帖子）
- 导航到了相似但不正确的目标页面
- 正确导航但提取了错误信息

### F5. Comment 自链接死循环（task 0/1/3）

**现象**：Agent 成功导航到帖子页面后，反复点击 "N comments" 链接（如 "45 comments" / "171 comments"），但 URL 始终不变。每步的 element_id 不同（DOM 重新渲染所致），但 bbox 完全一致。Agent thought 每步几乎相同（"clicking will navigate to the comment section"），confidence 保持 0.95。

**受影响 task**：

| Task | Intent | 目标链接 | 循环步数 | 结果 |
|------|--------|---------|---------|------|
| 0 | Navigate to comment section of [homemade] Pumpkin Loaf | "45 comments" | step 1-5 (5 次) | 30 步用完，score=0 |
| 1 | 同上（重复 task） | "45 comments" | step 1-5 (5 次) | 30 步用完，score=0 |
| 3 | Count comments mentioning 'spicy' in Beef Noods post | "171 comments" | step 1-5 (5 次) | 30 步用完，score=0 |

**action 序列示例（task 0）**：

```
step 0: click [401] → food 列表页 → 帖子页 ✓（正确导航）
step 1: click [3187] "45 comments" → URL 不变 ✗
step 2: click [9481] "45 comments" → URL 不变 ✗
step 3: click [15775] "45 comments" → URL 不变 ✗
step 4: click [22069] "45 comments" → URL 不变 ✗
step 5: click [28363] "45 comments" → URL 不变 ✗
... (重复至 step 29)
```

**根因分析**：

1. **Agent 已在目标页但不自知**：Reddit（Postmill）的帖子页面本身就是 comment section，URL `f/food/18838/homemade-...` 已包含评论。Agent 期望点击 "45 comments" 后跳转到不同 URL，但实际上该链接指向当前页面自身（锚点或自链接）。
2. **零自纠正**：连续 5 次点击同一位置（bbox `[152, 705, 81, 14]`）后 URL 不变，Agent 不调整策略，不尝试 scroll down 查看评论，不 finish。
3. **Postmill 特有问题**：主流 Reddit 的帖子页和评论区是同一页面，但 "N comments" 链接是自引用锚点。Agent 缺乏"已到达目标"的判断能力。

### F6. Early finish + Empty answer（18 tasks, 8.6%）

- **Early finish**（9 tasks）：Agent 在 1-2 步内 finish，未进行有效探索。平均步数仅 ~1.3。
- **Empty answer**（9 tasks）：Agent 调用 finish 但提交空 answer。对于 `string_match` 评测类型，空答案必然失败。

**代表 case：Task 2**——正确导航但空 answer 提交：

```
step 0: click [297] "Comments" → /f/movies/comments ✓
step 1: click [3283] permalink → /f/movies/128396/-/comment/2561509 ✓
step 2: finish (empty answer) → score=0
```

Agent 成功到达目标但提交空 answer。对于 `url_match` 评测类型，即使不 finish 也能得分——问题在于 Agent 选择了 finish 而非继续浏览。

### 失败成本分布

| 失败类别 | 平均步数 | 平均成本 | 说明 |
|---------|---------|---------|------|
| click_back_loop | 30.0 | $0.105 | 最贵，全部跑满 |
| search_repeat | 30.0 | $0.096 | 次贵，全部跑满 |
| max_steps | 30.0 | $0.091 | 跑满但无明确循环模式 |
| finish_eval_mismatch | 10.9 | $0.038 | 有 finish 但答错 |
| success | 8.0 | $0.024 | — |
| no_progress | 7.8 | $0.021 | 中程停滞 |
| incomplete_or_stuck | 6.5 | $0.016 | 快速停滞 |
| early_finish | 1.3 | $0.002 | 最便宜 |

**注**：上表来自 A4b_fail_reason_cost_stats（跨 DOM+Vision 全条件统计），DOM 条件内的 search_repeat / click_back_loop 成本可能略高。

---

## 四、效率指标

### DOM vs Vision 对比

| 指标 | DOM | Vision |
|------|-----|--------|
| 平均步数 | 16.64 | 6.59 |
| p95 步延迟 | 87,926ms | 46,378ms |
| 平均成本 | $0.054 | $0.014 |
| 平均能耗 | 3.76 mWh | 1.39 mWh |
| 平均 token | 56,037 | — |
| No-op rate | 17.1% | 38.9% |
| Page unchanged rate | 20.9% | 39.3% |

**DOM 成本是 Vision 的 3.9×**（$0.054 vs $0.014），主要因为 DOM 步数多（16.64 vs 6.59，2.5×）。DOM 更容易产出语法正确的 action（AXTree 提供 element_id），不容易触发早停，因此"坚持更久"——但也导致 search_repeat（48 tasks × 30 步）和 click_back_loop（20 tasks × 30 步）等昂贵失败。

**Vision 的 no-op 和 page_unchanged 率远高**：说明 Vision 的 action 执行成功率低，更快触发早停，反而"省钱"。

### 状态变化分布（DOM, top 5）

| 变化类型 | 触发次数 |
|---------|---------|
| content_changed | 2,902 |
| interactive_elements_changed | 2,554 |
| url_changed | 2,380 |
| form_value_changed | 2,277 |
| title_changed | 2,161 |

---

## 五、与 Vision 的跨模式对比

### 统计检验（基于 173 共同 task）

| 检验 | 统计量 | p 值 | 显著? |
|------|--------|------|-------|
| McNemar（SR） | 8.0 | 0.648 | 否 |
| Wilcoxon（成本） | 1,026 | 6.7×10⁻²³ | **是** |
| Wilcoxon（延迟） | 4,657 | 1.4×10⁻⁵ | **是** |

**McNemar 配对**：n11=2（双赢），n10=11（仅 DOM），n01=8（仅 Vision），n00=152（双败）。

**结论**：DOM vs Vision 的 SR 差异**不显著**（p=0.648），但 DOM 的成本（p < 10⁻²²）和延迟（p < 10⁻⁵）**显著更高**。

### Oracle 路由分析（cross-rep 口径，/210）

| 指标 | Raw | Adjusted |
|------|-----|----------|
| DOM SR | 10.00%（21） | 5.71%（12） |
| Vision SR | 4.76%（10） | 1.43%（3） |
| Union（oracle ceiling） | — | 7.14%（15） |
| Intersection | — | **0%（0）** |
| Routing headroom | — | 1.43pp |

### Exclusive sets（adjusted）

| 集合 | 数量 | 占比 |
|------|------|------|
| all_fail | 195 | 92.9% |
| only_dom | 12 | 5.7% |
| only_vision | 3 | 1.4% |
| both_success | 0 | 0% |

**Adjusted 后零交集**：DOM 和 Vision 的 adjusted 成功完全不重叠。每个 adjusted 成功都是 mode-exclusive 的。这意味着理想 oracle 路由可将 SR 从 5.71%（best single = DOM）提升到 7.14%（+1.43pp），但 headroom 很小。

### Oracle 选择分布（adjusted）

| 模式 | Oracle 选择 | 占比 |
|------|-----------|------|
| DOM | 12 | 80.0% |
| Vision | 3 | 20.0% |

DOM 在 oracle 中占主导地位。Vision 的 3 个 exclusive 成功中 2 个是 single_navigation、1 个是 page_reading。

---

## 六、Evaluator Error（Benchmark Noise）

210/210 task 全部完成。其中 3 个 task 存在持久性 `evaluator_error:Page.goto net::ERR_ABORTED`，重跑后仍一致复现，属于 benchmark 环境缺陷（reddit Docker 特定页面 Playwright 全量加载失败，curl 200 但浏览器 load 事件中断）。

| Task | 评测 URL | 评测类型 | Agent steps | 说明 |
|------|---------|---------|-------------|------|
| 72 | `f/memes/127531` | program_html | 7 | 检查 comment 内容 |
| 146 | `f/wallstreetbets/50335` | program_html | 5 | 检查 comment 内容 |
| 172 | `f/jerseycity/62526` | program_html | 8 | 检查 comment 内容 |

**处理**：2026-04-17 清除旧 stub summary 后重跑全部 5 个 error task，其中 task 149 和 151 修复成功（评测器正常运行，score=0），上述 3 个仍失败。评测器导航至 reference URL 时 `ERR_ABORTED`，Agent 侧执行正常。最终 score=0 合理（均为 comment-posting 任务，Agent 均未成功发表评论）。

**与 Classifieds 先例对比**（见 `classifieds/B1_DOM_digest.md`）：Classifieds 的 3 个 evaluator_error 来自 OpenAI API key 缺失或 program_html 超时，可离线重评修复；Reddit 的 3 个来自 Docker 页面加载缺陷，无法通过重评或重跑修复，属于 benchmark noise 的一种。

---

## 七、与 Classifieds DOM 的跨站对比

| 指标 | Classifieds DOM | Reddit DOM |
|------|----------------|------------|
| Raw SR | 10.26%（24/234） | 10.00%（21/210） |
| Adjusted SR | 8.48%（19/224） | **5.85%（12/205）** |
| FP 扣除数 | 5 | 9 |
| 平均步数 | 11.52 | **16.64** |
| 平均成本 | $0.043 | **$0.054** |
| p95 延迟 | 37,513ms | **87,926ms** |
| 搜索循环 | 3（1.3%） | **48（22.9%）** |
| Click-back loop | 11（4.7%） | **20（9.5%）** |
| No progress | 72（30.7%） | 50（23.8%） |

**Raw SR 几乎相同**（10.26% vs 10.00%），但 adjusted 后 Reddit 更低（5.85% vs 8.48%），因为 Reddit 的 FP 更多（9 vs 5，含 2 个 program_html eval_fp）。

**Reddit DOM 步数和成本显著更高**：搜索循环（48 vs 3）是最大差异。Reddit 帖子标题非结构化、搜索结果杂乱，4B 模型难以从失败搜索中提炼有效新关键词；Classifieds 商品标题短且标准化，搜索命中率高。

**Reddit P95 延迟翻倍**（88s vs 38s）：DOM 长 episode 更多（search_repeat + click_back_loop 共 68 个跑满 30 步），长 episode 中后期 DOM 变大导致 token 增多、推理变慢。

---

## 八、关键发现

1. **DOM 是 Reddit B1 最优模式**：Adjusted SR 5.85%（12/205），高于 Vision 2.98%（5/168），但 McNemar 不显著（p=0.648）
2. **DOM 成本是 Vision 的 3.9×**：$0.054 vs $0.014，DOM 步数多（16.64 vs 6.59）导致更多昂贵失败
3. **搜索循环是 Reddit DOM 的标志性失败**（48/210=22.9%）：远超 Classifieds（1.3%），是 Reddit 搜索交互难度的直接体现
4. **Adjusted 后零交集**：DOM 和 Vision 的 adjusted 成功完全不重叠，oracle headroom 仅 1.43pp
5. **FP 扣除比例高**：Raw 21 → Adjusted 12（扣 9，43%），NA FP（5）+ Visual FP（2）+ Eval FP（2, §88 program_html 补充规则）
6. **Comment 自链接死循环**（F5）：Postmill 特有问题，Agent 不理解"已到达目标"
7. **30 步跑满 episode 集中且昂贵**：search_repeat（48）+ click_back_loop（20）+ max_steps（3）= 71 个 episode（33.8%）跑满 30 步，平均成本 $0.095-0.105
8. **Reddit 对 DOM 模式更难**：相比 Classifieds DOM（adjusted 8.48%），Reddit 5.85% 更低，搜索循环和自链接问题是主因

---

*生成时间：2026-04-21*
*数据来源：B1_3mode_reddit_20260413 phase1_dom_router_0，210 tasks*
*Vision 当前 173/210 tasks（`_synthesized`），SoM 仅 3/210 tasks，跨模式分析为 DOM vs Vision 二模式*
