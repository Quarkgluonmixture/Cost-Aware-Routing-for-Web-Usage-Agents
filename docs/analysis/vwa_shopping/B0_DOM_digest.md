# B0 Shopping -- DOM 模式分析

> B0 = Qwen3-VL-235B-A22B（proxy API），DOM 模式，shopping 站点（Magento）
> Run: `B0_3mode_shopping_20260421`，466 episodes，仅此一个 condition
> Digest 覆盖 213 个 failure episodes（不含 55 个 raw success）
> 综合指标见 `B0_findings.md`

---

## 概要

B0 Shopping DOM raw SR 11.80%（55/466），adjusted SR 6.24%（29/465，reason diagnostics 管线）。213 个失败 episode 中，**答案对齐错误（26.8%）** 和 **目标不可达（20.2%）** 是两大主导失败类别。34.3% 的失败直接归因于 DOM 模式视觉信息缺失，是 shopping 站点 DOM 模式的核心瓶颈。

---

## 1. 失败类别总览

| 类别 | 数量 | 占比 | 平均步数 | 中位步数 | 步数范围 |
|------|------|------|---------|---------|---------|
| 答案对齐错误 | 57 | 26.8% | 6.9 | 5 | [2, 18] |
| 目标不可达 | 43 | 20.2% | 8.9 | 6 | [3, 30] |
| 执行停滞 | 37 | 17.4% | 7.0 | 6 | [4, 18] |
| 搜索循环 | 24 | 11.3% | 17.9 | 16 | [7, 30] |
| 事实推理错误 | 23 | 10.8% | 5.8 | 6 | [2, 14] |
| 导航循环 | 10 | 4.7% | 28.8 | 30 | [19, 30] |
| 过早结束 | 10 | 4.7% | 3.0 | 2 | [1, 12] |
| 导航失败 | 7 | 3.3% | 6.7 | 6 | [4, 14] |
| 流程超时 | 1 | 0.5% | - | - | - |
| 综合失败 | 1 | 0.5% | - | - | - |

**步数特征**：搜索循环（17.9 步）和导航循环（28.8 步）消耗步数最多，agent 陷入死循环直到 max_steps 截断。过早结束（3.0 步）步数最少，agent 在信息不足时快速但错误地给出答案。

---

## 2. 脚手架/表征缺陷 vs 模型能力问题

| 归因 | 数量 | 占比 | 主要类别 |
|------|------|------|---------|
| 脚手架/表征缺陷 | 103 | 48.4% | 目标不可达(43)、执行停滞(21)、答案对齐错误(15)、事实推理(9)、导航循环(8) |
| 模型能力问题 | 110 | 51.6% | 答案对齐错误(42)、搜索循环(21)、执行停滞(16)、事实推理(14)、导航失败(7) |

脚手架缺陷以**视觉信息缺失**为绝对主因（73/103 = 70.9%）。模型能力问题更分散，答案对齐错误（42/110）是最大单项。

---

## 3. DOM 特有缺陷分析

### 3.1 视觉信息缺失（73 episodes, 34.3%）

**核心问题**：Shopping 站点 466 个 task 中 269 个（57.7%）涉及视觉属性（颜色、形状、图案、布局位置），DOM 模式无法从 AXTree 文本获取这些信息。

**典型表现**：

| 视觉属性 | 示例 | 后果 |
|---------|------|------|
| 颜色识别 | Task 4: 黑白商品识别；Task 6: 红色拖鞋筛选 | 仅能通过名称含颜色词猜测，遗漏无颜色词的目标商品 |
| 形状识别 | Task 11: 圆形饼干；Task 36: 圆形奶酪 | DOM 中 image alt 通常为空（"Image"），无形状描述 |
| 包装外观 | Task 8: 红色包装薯片 | 将所有搜索结果视为符合，错误纳入非目标商品 |
| 颜色筛选 | Task 25: 蓝色雨衣；Task 39: 色彩丰富的商品 | 反复查看同一商品的颜色选项（仅 Black/Gray/Green），形成 click-back 循环 |

**Task 6 实例**（红色 Nike 拖鞋筛选）：

- 搜索 "nike slide slippers"，结果页含多款拖鞋
- 仅名称含 "Red" 的 1 款被识别（Nike Jordan AR6374-602）
- 参考答案要求 4 款红色产品，其余 3 款名称无颜色信息，需通过产品图片视觉判断
- DOM 模式下该任务必然失败

**Task 11 实例**（圆形饼干）：

- 搜索 "ice cream sandwiches"，正确商品 Skinny Cow 出现在搜索结果首项
- DOM 文本无饼干形状描述（image alt = "Image"），agent 无法判断哪个是"圆形"
- 3 对 click-back 循环后选错商品，19 步消耗

**与 classifieds 的差异**：Classifieds 视觉任务主要涉及商品照片（车/房/电子产品），特征通常在标题中部分体现。Shopping 视觉任务更多涉及颜色、形状等纯视觉属性，image alt 字段几乎为空，DOM 信息瓶颈更严重。

### 3.2 空间感知缺失（17 episodes, 8.0%）

**核心问题**：Magento 商品列表在浏览器中呈网格布局（4-5 列），但 AXTree 将其线性化为一维列表，丢失行列位置信息。

**典型表现**：

**Task 10**（冷冻披萨第二行第三项）：Agent 搜索后反复点击正确商品 Portobello Arancini Bites（4 次），但因 DOM 无法验证"第二行第三项"位置而每次返回，形成 9 对 click-back 循环，30 步耗尽。

**Task 13**（产品展示第一行价格范围）：Agent 无法判断哪些产品属于"第一行"，点击第一个产品查看价格后返回，形成 15 对 click-back 循环。

**Task 15/17**（网格布局行列定位）：Agent 将线性列表中第 N 个商品等同于"第 N 行"，步数仅 1-2 步即过早结束，答案完全错误。

**影响范围**：主要集中在 `grid_position` 类型任务（需要定位特定行列的商品），以及部分需要判断"最后一行/列"的 `page_reading` 任务。

### 3.3 Element ID 失效（16 episodes, 7.5%）

**表现类型**：

1. **下拉菜单子项无独立 element_id**（6 episodes）：Magento 顶部导航的子分类（如 "Lamps & Shades"、"Basic Cases"）在 AXTree 中仅作为 `[DROPDOWN OPTIONS]` 文本出现，无可点击的 element_id。Agent 反复点击父级菜单项，无法进入子分类。

   - Task 18: 点击 "Cell Phones & Accessories" 6 次，无法进入 "Basic Cases" 子分类
   - Task 19: 点击 "Tools & Home Improvement" 19 次，无法进入 "Lamps & Shades"

2. **Newsletter 订阅框与搜索框混淆**（4 episodes）：页面底部 Newsletter 订阅框与顶部搜索栏同为 `input text` 类型，长 AXTree 滚动后 agent 将搜索词输入订阅框。

   - Task 3: 搜索词 "red ps4 controller" 4 次输入 Newsletter 框（element 37541）
   - Task 44: AXTree 截断后搜索栏丢失，查询输入 Newsletter 框（element 10839）

3. **容器节点中心越界**（6 episodes）：元素中心 y 坐标超出 viewport 高度（>720px），click 静默失败。

### 3.4 AXTree 截断（2 episodes, 0.9%）

**Task 12**：搜索 "van gogh" 后，AXTree 仅显示导航/筛选区，12 条产品 item 的链接均未出现，agent 无法点击任何商品。

**Task 44**：搜索词返回无关结果后，后续步骤 AXTree 截断导致搜索栏丢失。

AXTree 截断在 shopping 站点不常见（仅 2/213 = 0.9%），Magento 页面结构相对规整。

---

## 4. 主要失败类别详解

### 4.1 答案对齐错误（57 episodes, 26.8%）

最大的失败类别。Agent 完成了 finish 动作但答案与评测标准不匹配。

**子类型分解**：

| 子类型 | 描述 | 典型 task |
|--------|------|----------|
| 约束遗漏 | 多条件任务中遗漏部分约束（分类、颜色、价格范围） | Task 0（未验证 "red"）、Task 2（错误分类） |
| 扫描不完整 | 未遍历全部候选结果，遗漏符合条件的商品 | Task 7（遗漏 Two Wolves 产品） |
| 视觉属性猜测 | DOM 无视觉信息，凭名称含色词猜测颜色 | Task 8（包装颜色猜错） |
| 分类误导航 | 搜索词被 Magento 过滤到错误分类 | Task 2（Posters & Prints 不存在） |

**Task 0 实例**（最便宜的红色毯子）：
- Agent 从 Home & Kitchen > Blankets & Throws 导航，按价格排序，找到 $24.97 的 BOLDROLE 红色毛毯
- 返回翻页寻找更低价选项，改用搜索 "red blanket"
- 最终选中 $11.99 的 Christmas Fleece Blanket，但该商品标题不含 "red" 且未验证分类归属
- 评测拒收（fail_finish_eval_mismatch）

**Confidence 分布**：57 个答案对齐错误中 37 个 confidence=medium、20 个 confidence=high。Agent 对错误答案的自信程度不一，medium 居多说明部分情况下 agent 意识到不确定性但仍提交了错误答案。

### 4.2 目标不可达（43 episodes, 20.2%）

全部 43 个 episode 的 `is_scaffolding_issue=是`。主因为视觉信息缺失（40/43 = 93%），少数为空间感知缺失或 element_id 失效。

这些 task 在 DOM 模式下**结构性不可能成功**——所需信息（颜色、形状等）不存在于 AXTree 文本中。Agent 行为表现为：反复搜索/滚动试图找到文本线索 → 失败 → 循环到截断。

### 4.3 执行停滞（37 episodes, 17.4%）

Agent 操作未能改变页面状态，步数积累但无进展。

**主要原因**：
- **Click 连续失败**：Element 中心越界或元素不可交互，连续多次点击无效（Task 5: 3 次、Task 22: 3 次）
- **重复操作无果**：相同搜索词/操作重复执行，结果不变
- **视觉+执行双重失败**：视觉任务本身不可达，叠加操作失败（Task 5: 视觉需求 + click 失败）

### 4.4 搜索循环（24 episodes, 11.3%）

Agent 在搜索流程中陷入循环。平均步数 17.9（所有类别中仅次于导航循环），大量步数被浪费。

**两种主要模式**：
1. **搜索框错用**（4 episodes）：将查询输入 Newsletter 订阅框而非搜索栏
2. **搜索-点击循环**（20 episodes）：搜索→点击首个结果→发现不对→重新搜索相同词，从不尝试修改搜索策略或探索其他结果

**Task 34 实例**（人造花）：搜索 "artificial plants & flowers" 后 SELECT_OPTION(Price) → 点击首个图片 → 跳转至 Xerox 打印机页面。此循环精确重复 10 次，30 步全部浪费。Agent 从未尝试点击文字链接或修改搜索词。

### 4.5 事实推理错误（23 episodes, 10.8%）

Agent 对页面信息的推理/计算错误。

**子类型**：
- **排序方向误判**：URL 显示 asc 但 agent 认为 desc（Task 9）
- **价格计算错误**：$500-$10=$490 而非应算 $395-$10=$385
- **视觉属性猜测**：无依据假设第一个商品为圆形（Task 36）、搜索 "colorful thing" 选中 Dr. Seuss T-Shirt（Task 39）
- **搜索词语义误解**：将 "Chocolate category" 当搜索词而非分类导航（Task 40）

### 4.6 导航循环（10 episodes, 4.7%）

Agent 反复 click 某个元素后 back，形成 click-back 死循环。平均 28.8 步，中位 30 步（达到 max_steps 截断）。

**主因**：DOM 无法提供空间布局验证（8/10 为 grid_position 或需空间定位的任务），agent 反复查看同一商品但无法确认其网格位置，循环直到截断。

### 4.7 过早结束（10 episodes, 4.7%）

Agent 在信息不足时快速给出错误答案。平均仅 3.0 步。

**主因**：
- **空间布局误判**：将线性列表第 N 项等同于网格第 N 行（Task 15/17）
- **筛选器范围误用**：将 Shop By 面板的价格筛选范围当作产品实际价格范围（Task 16/17）
- **翻页不足**：仅滚动一次即判断已到底部（Task 14）

---

## 5. 动作执行分析

### 5.1 动作类型统计（213 failure episodes）

| 动作类型 | 总次数 | 失败次数 | 失败率 |
|---------|--------|---------|--------|
| click | 518 | 128 | 24.7% |
| type | 148 | 13 | 8.8% |
| scroll | 363 | - | - |

**Click 失败率 24.7%** 远高于 type 的 8.8%。主要原因：
1. 元素中心坐标越界（y > 720px viewport 高度）
2. 容器节点中心落在非可交互区域
3. 下拉菜单子项无独立 element_id

### 5.2 早停触发（全 466 episodes）

| 触发原因 | 次数 |
|---------|------|
| action_failed | 664 |
| page_unchanged_streak | 261 |
| no_progress_streak | 261 |

action_failed 触发 664 次，说明大量操作未能改变页面状态。这与 click 24.7% 的失败率一致。

---

## 6. B0 Shopping DOM 特有行为

### 6.1 Magento 搜索引擎的语义局限

Magento 内置搜索基于关键词匹配，不支持复杂查询。Agent 常将多约束拼入搜索词（如 "red ps4 controller under $200"、"canvas print with grapes category:posters & prints"），搜索引擎无法正确解析，返回无关结果或被过滤到错误分类。

与 classifieds（OSClass）的差异：OSClass 搜索相对简单（品名 + 价格区间分离输入），Magento 的分类体系更深（3-4 级导航），Agent 更容易迷失在分类导航中。

### 6.2 下拉导航菜单 capability-environment gap

与 classifieds B0 的 `<select>` 问题类似但机制不同：Magento 使用 CSS hover 下拉菜单展示子分类，AXTree 将子项暴露为 `[DROPDOWN OPTIONS]` 文本但无独立 element_id。Agent 正确识别子分类名称（如 "Lamps & Shades"），但只能反复点击父级菜单项，无法进入子分类。

| 站点 | 控件类型 | Agent 行为 | 后果 |
|------|---------|-----------|------|
| Classifieds | 原生 `<select>` | 反复 click 同一 eid | Cycle detection 截断 |
| Shopping | CSS hover dropdown | 反复 click 父级菜单 | 步数耗尽或 incomplete_or_stuck |

### 6.3 价格筛选器误读

Shopping 站点左侧 "Shop By" 面板包含价格范围筛选器（如 "$0.00 - $9,999.99"），AXTree 中以文本形式呈现。Agent 有时将筛选器范围值误当作商品实际价格范围，在 1-2 步内给出错误答案后 finish（Task 16/17）。

这在 classifieds 中未观测到——OSClass 没有类似的价格范围面板。

### 6.4 Scroll 行为

B0（235B）在 shopping 中的 scroll 行为与 classifieds 一致：dy 符号约定不稳定（+0.5 和 -0.5 混用），为模型固有行为（跨 API 实验已确认，见 classifieds B0_DOM_digest 详细分析）。

全局 scroll_changed 1920 次 / 4751 步 = 40.4% 步骤触发了滚动变化，说明 agent 频繁使用滚动浏览商品列表。

---

## 7. 与 Classifieds B0 DOM 的行为对比

| 行为 | Classifieds B0 DOM | Shopping B0 DOM |
|------|-------------------|----------------|
| 翻页（paginate） | 33+ task 翻到 page 2/3/4 | 较少观测到 |
| 价格区间筛选 | 21+ task 使用 sPriceMin/sPriceMax | 极少使用 Magento 价格 filter |
| 多 Tab 切换 | 4 个 task 使用 tab_focus | 未观测到 |
| 表单字段精准聚焦 | click-by-eid 逐字段填写 | 类似，但 Magento 表单场景更少 |
| 搜索框/订阅框混淆 | 未观测到 | 4 episodes 误输入 Newsletter 框 |
| 价格筛选器误读 | 未观测到 | 2+ episodes 将筛选范围当答案 |
| Scroll dy 混用 | 确认为模型行为 | 同样存在 |

**关键差异**：B0 在 classifieds 展现的翻页、价格筛选等高级导航能力在 shopping 站点显著减弱，可能原因：(1) Magento 站点结构更复杂（更深的分类层级、更多 UI 控件），增加了导航难度；(2) Shopping 任务类型分布不同（更多视觉属性任务，减少了导航类任务的比例）。

---

## 8. Reason Bucket 与 Digest 类别对应关系

| Digest 类别 | 主要 reason_bucket | 数量 |
|------------|-------------------|------|
| 答案对齐错误 | fail_finish_eval_mismatch | 54 |
| 执行停滞 | fail_no_progress | 31 |
| 目标不可达 | fail_no_progress | 24 |
| 事实推理错误 | fail_finish_eval_mismatch | 20 |
| 搜索循环 | fail_no_progress | 13 |
| 目标不可达 | fail_finish_eval_mismatch | 9 |
| 导航循环 | fail_max_steps_click_back_loop | 8 |
| 过早结束 | fail_early_finish | 8 |
| 导航失败 | fail_finish_wrong_url_not_found | 7 |

**fail_finish_eval_mismatch** 横跨答案对齐错误（54）和事实推理错误（20），二者区别在于：答案对齐错误是约束遗漏/扫描不完整，事实推理错误是对已获取信息的推理/计算错误。

---

## 总结

B0 Shopping DOM 模式的核心瓶颈是 **视觉信息缺失**（34.3% 的失败）和 **答案对齐错误**（26.8%）。前者是 DOM 模式的结构性限制，后者是 235B 模型在复杂多约束任务上的推理不足。

**可改善空间**（需多模式数据验证）：
- 视觉信息缺失的 73 个 task 中，相当比例在 SoM/Vision 模式下可能成功
- 空间感知缺失的 17 个 task，SoM 模式（带布局标注）可能部分解决
- 搜索循环和导航循环消耗大量步数但不产出，早期检测+模式切换可节约成本

**不可改善的固有限制**：
- 答案对齐错误中的纯推理问题（42/57 归因为模型能力）需更强的模型而非模式切换
- Magento 下拉导航的 capability-environment gap 在所有模式中均存在
- N/A task 的 FP 问题需修改 agent prompt 或评测器

---

*更新时间：2026-04-23*
*数据来源：B0_3mode_shopping_20260421 analysis/digest/digest_dom.jsonl（213 failure episodes）*
