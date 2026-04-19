# B0 WA Shopping — DOM 模式分析

> B0 = Qwen3-VL-235B-A22B（proxy API），DOM 模式，WA shopping 站点（192 tasks）
> RUN_ID: `B0_wa_3mode_shopping_20260417`

---

## 持久性 Evaluator 错误

2 个 task 存在持久性 `evaluator_error:n/a`，重跑 3 次均一致复现。

| Task | Intent | 评测类型 | Reference | Agent steps | 根因 |
|------|--------|---------|-----------|-------------|------|
| 334 | Tell me when I last ordered my muffin cornbread mix? | string_match (fuzzy_match) | March 11th 2023 | 30 | GPT evaluator 返回 "N/A" |
| 335 | Tell me when I last ordered my body butter? | string_match (fuzzy_match) | January 16th 2023 | 30 | GPT evaluator 返回 "N/A" |

**根因**：`llm_fuzzy_match` 调用 GPT-4o-mini 判断 agent answer 与 reference 的语义一致性，期望回复 "correct"/"incorrect"/"partially correct"。GPT 返回 "N/A"（可能因 agent 提交空 answer 且 reference 是具体日期，GPT 认为无法判断），触发 `assert "correct" in response` → `AssertionError("n/a")` → `evaluator_error:n/a`。

**修复**：已 patch `external/visualwebarena/evaluation_harness/helper_functions.py` 的 `llm_fuzzy_match` 和 `llm_ua_match`，将意外 GPT 响应当作 score=0 处理（不再抛异常）。后续 SoM/Vision condition 不会再遇到此问题。

**影响**：2/192 tasks (1.0%)。Agent 在这两个 task 上均未提交正确答案（30 步耗尽，click/back 循环），即使评测正常运行 score 也为 0。最终 score=0 合理。

**与先例对比**：
- Classifieds B1 §10.4：OpenAI API key 缺失导致 evaluator_error，通过离线重评修复
- Reddit B1 DOM：ERR_ABORTED 页面加载缺陷，无法修复
- 本例：GPT 返回意外响应，已通过代码 patch 根治（assert→graceful fallback）

---

## 失败模式：不充分探索即自信 Finish

Order history 类任务中反复出现的模式：agent 看到 My Orders 第一页就认定"最新订单"并 finish，没有 scroll 或翻页确认所有订单后再比较日期。

| Task | Intent | Agent 选中 | 正确答案 | 根因 |
|------|--------|-----------|---------|------|
| 51 | （order status 类） | 首页某订单 | 另一个更新的订单 | 日期比较不充分 |
| 96 | Tell me the status of my latest order and when will it arrive | #189 (5/2/23, Pending) | #170 (5/17/23, Canceled) | 首页 #189 排在上方，agent 未检查 #170 日期更新 |
| 117 | （order status 类） | 首页某订单 | 另一个更新的订单 | 同上 |

**对称模式**：
- **不翻页就 finish**（本节）：agent 信息不足时过早提交 → 答案错误
- **一直翻页不 finish**（task 47/48）：agent 翻完所有页面但答案是"零"，不知如何报告 → 被 early-stop

**根因**：模型能力缺陷——信息在页面中可得（日期、状态均在 AXTree 内），agent 未充分利用。Prompt 无"确保看完所有数据再回答"的引导。SOM 模式在 task 96 上正确识别了最新订单（5/17 > 5/2），说明问题不在信息可见性而在模型推理。

---

## 评测 False Negative：program_html must_include 过严

`program_html` + `must_include` 评测器要求最终页面包含精确短语，导致 agent 行为正确但被判失败。

| Task | Intent | Agent 行为 | 评测要求 | 失败原因 |
|------|--------|-----------|---------|---------|
| 118 | I have jaw bruxism problem, show me something that could alleviate | 搜索 "bruxism mouth guard" → 点击相关产品（Mouth Guard, Stops Bruxism, TMJ） | `must_include: ["jaw bruxism", "mouth guard"]` | 产品页含 "Bruxism" 但不含 "jaw bruxism" 精确短语 |

**性质**：DOM/SOM 均失败（3 步完成），agent 找到了完全相关的产品，任务是开放式 "show me something"，任何 bruxism mouth guard 都应算对。评测器要求精确短语匹配过于严格。

**处置**：WA/VWA `program_html` 评测的已知局限，论文中作为评测噪声讨论。FN 无法像 FP 那样系统性自动纠正（需人工判断 agent 回答是否合理）。

---

## Viewport-Only 观测与 Scroll 决策机制

### 核心发现：`current_viewport_only=True`

VWA/WA 框架的 `TextObervationProcessor` 默认 `current_viewport_only=True`（`p79/envs/vwa_wrapper.py:76`），AXTree 只包含当前 viewport 内的元素。

代码路径：`external/visualwebarena/browser_env/processors.py:307` — 遍历 DOM 节点，`in_viewport_ratio < IN_VIEWPORT_RATIO_THRESHOLD` 的节点被 `remove_node_in_graph` 移除。

### `in_viewport_ratio` 计算 Bug（§80）

`processors.py:218` 的面积比计算存在运算符优先级错误：

```python
# Bug：ratio = overlap_width * overlap_height / width * height   # ((ow*oh)/w)*h → 远超 1.0
# 修正：ratio = (overlap_width * overlap_height) / (width * height)   # 面积比 0~1
```

`IN_VIEWPORT_RATIO_THRESHOLD=0.6`（constants.py:324）形同虚设：任何有 ≥1px 在 viewport 内的元素，ratio 都远超 0.6，全部被保留。

**DOM 和 SoM 对称受影响**：SoM 的 `[SOM_MARKS]` 文本列表从同一份 filtered AXTree 提取（`som.py:172 _extract_text_marks(obs_text)`），viewport 边缘元素在 SoM marks 中也有完整标签。Vision 不受影响（纯截图）。

**双层影响**：
1. **功能性失败**：viewport 边缘元素暴露 element_id → agent 点击 → 元素中心在 viewport 外 → click 失败 → 早停。WA Shopping DOM 14 tasks（7.5pp），Classifieds DOM 3 tasks（1.3pp）
2. **语义误导**：部分可见元素给完整文字 → agent 以为信息齐全不 scroll → 漏掉 viewport 外信息。WA Shopping order history 无分页提示（对比 Classifieds 有 "1-12 of N"），语义惩罚尤重

**实证**：
- Task 145 step 17：Orangina 产品行在截图底部只露出一行产品名，DOM AXTree 包含完整 170 字符描述 + 价格 $29.99
- WA Shopping DOM vs SoM SR（93 task 重叠）：17.2% vs 16.1%（gap -1.1pp），几乎持平——验证两者受语义误导等量影响

**修正后阈值 0.6 的数学保证**：若 overlap_h/h ≥ 0.6，则 center_y = y+h/2 ≤ 720-0.1h < 720，元素中心必在 viewport 内。功能性失败完全消除，语义误导大幅缓解（<60% 可见元素被排除）。

**决策：修复 + 重跑所有 DOM/SoM condition**（B0+B1，所有站点）。理由：
1. 功能性失败（7.5pp）和语义误导不可接受，公平 baseline 优先
2. 修正后阈值 0.6 是上游已有常量，数学保证功能安全
3. 残余不公平（60-100% 可见元素 DOM 给完整文字）是 DOM 元素级粒度的结构性限制，无法进一步解决
4. 论文注明：修正了上游 ratio bug，使用原始阈值 0.6

### P79 viewport_height=720 vs VWA 官方 2048

VWA 官方 `run.py` 默认 `viewport_height=2048`（并硬编码 `current_viewport_only=True`）。P79 使用 `viewport_height=720`（`vwa_wrapper.py:78`）。

| viewport | 典型 order detail 页 | ratio bug 影响 |
|----------|---------------------|---------------|
| 2048px | 整页可见（~1140px < 2048） | 几乎无感（元素本就 100% 可见） |
| 720px | 只看到前 ~4 个 order / 1 个商品 + 半个 | 放大（更多元素半露出边缘） |

**720px 是正确选择**：(1) 标准浏览器窗口高度，截图与真实用户视角一致；(2) 三模式共享同一 viewport，内部对比公平；(3) 2048px 本质上回避了 scroll/探索决策的挑战，对 web agent 研究不利。论文中需注明此差异，跨论文 SR 绝对值不可直接比较。

### Agent 何时 scroll？

**人类判断需要 scroll 的三个信号**：

| 信号 | DOM | 截图 | Agent 可用？ |
|------|-----|------|-------------|
| 浏览器滚动条 | 不在 AXTree | 不在 Playwright 截图（截的是页面内容，不含浏览器 chrome） | **三模式都不可用** |
| 内容被截断 | 元素级全有/全无，无法表达截断 | 像素级可见截断 | 仅 Vision/SoM 截图部分可用 |
| 常识（"不可能只有2个产品"） | LLM 推理 | LLM 推理 | 不可靠 |

Agent 实际依赖的线索：

1. **页面级文字提示**：如 `"1 - 12 of 2537 listings"`（Classifieds 有，WA Shopping order history 无，Amazon/Google 等真实网站通常也无）
2. **任务级推理**：目标信息是否已找到（"blue bike 还没出现" → scroll；"看到两个食品的价格" → 认为齐全 → finish）
3. **视觉截断**（仅截图模式）：看到文字/图片被切 → 暗示有更多内容。但若内容完全在 viewport 外（如 Reddit 评论区在帖子正文下方），截图也无截断线索

**更深层的 scroll 问题**（超出 ratio fix 范围）：Reddit comment 页面中，评论区完全在 viewport 外，post 正文在 viewport 内完整显示，无任何截断 → 三种模式都无法判断需要 scroll。Agent 反复点击已在当前页面的 "comment" 链接。这需要 agent 架构层面的探索机制或网站先验知识注入（参见 M5 EIP 方案 §31）。

### 典型案例：Task 145

- **Intent**: "How much I spent on cooking and food shopping during March 2022"
- **Ref**: $52.35（Order Grand Total，含 tax/shipping）
- **Agent 提交**: $42.35（$12.36 + $29.99，手动加两个商品价格）
- **根因**: 订单详情页的 "Items Ordered" 表格在 viewport 内完整显示了 2 个商品，但 **Order Totals 区域（Grand Total $52.35）在 viewport 外被过滤**。AXTree 中没有任何信号暗示页面下方还有内容。Agent 认为信息齐全就 finish 了。

### 对比：Classifieds Task 11 为何能 scroll

- **Intent**: "What is the size of the wheels in inches of the first blue bike on this page?"
- DOM 中明确显示 `'1 - 12 of 2537 listings'` → agent 知道还有很多未显示的内容
- 当前列表无任何 listing 提及 "blue" → 任务目标未达成
- 两个信号叠加 → agent 持续 scroll（3 次成功 scroll 直到页底）

### 典型案例：Task 146

- **Intent**: "What is the size configuration of the picture frame I bought Sep 2022"
- **Ref**: 16x24
- **Agent 行为**: 进入 order #175（9/1/22）详情页，viewport 内只看到第一个商品（Disney Mickey Mouse T-Shirt），picture frame 在 viewport 外。Agent 认为"这个订单没有 picture frame"→ 返回。反复进出 #175 共 4 次，30 步耗尽。
- **根因**: 与 task 145 对称——订单有多个商品，viewport 只能显示第一个，后续商品被过滤。SoM 也失败（11 步），确认是三模式共享的 viewport 信息盲区。

### 影响

- **ratio bug 修复前**：DOM/SoM 部分可见元素完整文本被保留 → "信息齐全"假象 → 功能性失败（click 边缘元素）+ 语义误导（不 scroll）
- **ratio bug 修复后**：<60% 可见元素被排除，功能性失败消除，语义误导大幅缓解。但 DOM 仍无法表达"60% 可见"（元素级全有/全无限制）
- **超出 ratio fix 的 scroll 盲区**：完全不在 viewport 的内容（Reddit 评论区、长页面底部信息），三模式都无法判断需要 scroll
- 无分页提示的页面（订单详情、产品页、Amazon/Google 等真实网站）→ agent 容易过早 finish
- 有分页提示的页面（Classifieds 列表页 "1-12 of N"）→ agent 知道该 scroll/翻页

---

## 待填充

- [ ] DOM 全量完成后：SR/cost/latency 总体统计
- [ ] 失败模式分类
- [ ] Auth refresh 效果验证（step_000 URL 是否 logged-in）
