# B1 三模式共性脚手架缺陷分析

> 本报告记录 DOM / SoM / Vision 三种观测模式**共同受影响**的脚手架缺陷。
> 这些缺陷与观测表征无关，属于站点功能限制或 VWA 环境的结构性问题。
> 各模式专有缺陷见 `AI_B1_DOM_analysis.md` / `AI_B1_SOM_analysis.md`。

---

## 1. 地点过滤不可达（3 例：task_58, 72, 74）

Classifieds 站点没有有效的地点筛选 UI。Agent 只能将地名作为搜索关键词输入搜索框，但搜索框按商品名匹配，不按地点过滤。

**表现**：
- task_58：搜索"blue chair Washington DC"无结果，无法筛选特定地区商品
- task_72：重复搜索"delaware"三次，搜索框无法理解地点约束
- task_74：将"Ohio"作为搜索关键词，结果无法限定到目标州

**影响范围**：三种模式均失败，与观测方式无关。

---

## 2. 编辑页面字段不可达（2 例：task_4, 75）

商品编辑页面的价格/描述输入框不在当前 viewport 中，Agent 持续滚动但无法定位目标字段。

**表现**：
- task_4：编辑页面中价格字段未在可见区域，Agent 持续滚动未找到
- task_75：错误编辑了冰箱列表而非白色花瓶，且在编辑页面无法定位价格字段

**影响范围**：三种模式均受限于 viewport 尺寸，与观测方式无关。

---

## 3. `<select>` 下拉菜单三层不可达（VWA 框架级缺陷）

`<select>` 下拉菜单在 VWA 默认配置下对**所有 agent**（不限于本实验）实质不可用。问题不在 Chrome AXTree 本身（CDP `Accessibility.getFullAXTree` 返回完整 option 列表），而在 VWA 框架的三层过滤和操作限制：

### 第一层：关闭状态 → option 元素 bbox=0 被过滤

VWA 的 `TextObervationProcessor`（`external/visualwebarena/browser_env/processors.py`）在 `current_viewport_only=True`（默认）下，会过滤掉 `width==0` 或 `height==0` 的节点。未展开的 `<select>` 的 `<option>` 元素在 DOM 中存在，但浏览器不为其分配可见 bounding box（宽高为 0），因此**全部被过滤掉**。

AXTree 中只显示 `combobox "Category"` 本身，不列出任何可选项。

### 第二层：展开状态 → viewport 交集过滤

即使 Agent 成功点击 `<select>` 展开了下拉列表，VWA 的 viewport 交集过滤（`IN_VIEWPORT_RATIO_THRESHOLD = 0.6`，即节点面积需≥60% 在 viewport 内才保留）也会过滤掉滚动到 viewport 外的选项。

实际观察（task_58 SoM step_001）：Agent 展开了分类下拉菜单，AXTree 显示 `combobox '' expanded: True`，但**没有任何 option 子节点**——Furniture 选项在下拉列表底部，不在 viewport 可见区域内。

### 第三层：scroll 只滚动页面，不滚动下拉列表

VWA 的 `scroll [x, y]` action 通过 `window.scrollBy()` 实现，只能滚动页面 viewport，无法滚动 `<select>` 下拉列表的内部滚动区域。Agent 识别到目标选项不可见后执行 scroll，实际效果是页面整体滚动，下拉列表反而可能被滚出可见区域。

### 各模式表现

- **DOM 模式**：AXTree 不含 option → Agent 不知道有哪些选项可选
- **SoM 模式**：SoM 标注基于 AXTree 生成 → option 无 mark 标签；截图中可见的选项也无法被文本索引引用
- **Vision 模式**：截图可见当前展开的部分选项，但无法滚动下拉列表查看全部

**模型的自适应绕路**：虽然 `<select>` 下拉菜单不可用，但模型在多个 task 中（如 task_38）能自行发现页面上的 "All categories" 链接作为替代入口，通过链接导航绕过下拉菜单完成分类选择。这表明模型具有一定的 recovery 能力，但绕路过程通常消耗 2-3 个额外步骤（点击下拉失败 → 重试 → 发现替代链接 → 点击进入），在 max_steps 有限的情况下压缩了后续任务执行的步数预算。

**结论**：这是 VWA 框架 `current_viewport_only=True` 配置下的结构性限制，非本实验的设计缺陷。所有基于 VWA 默认配置的 agent 均受影响。

---

## 4. Type 操作导致页面全选变蓝（VWA/Playwright 环境问题）

`type` 操作（如搜索框输入）偶尔导致页面文本被全选高亮（蓝色覆盖），影响后续截图的视觉可读性。

**机制**：VWA 底层通过 Playwright 的 `type` 方法执行输入，该方法会先全选输入框内容再键入新文本。在某些页面状态下，选区从输入框扩散到整个页面，导致后续截图中所有文本呈蓝色高亮状态。

**实例**：task_17 step 8 执行 `type "red handlebars"` 后，step 9 截图中整个搜索结果页面被蓝色选中框覆盖。

**影响**：
- **DOM 模式**：无影响（Agent 读 AXTree 不看截图）
- **SoM 模式**：蓝色高亮覆盖 SoM mark 标注，干扰视觉定位
- **Vision 模式**：严重干扰视觉理解，Agent 可能无法识别页面内容

**结论**：VWA/Playwright 环境层面的已知行为，非本实验脚手架缺陷。

---

## 5. 模型不会翻页（模型能力缺陷）

模型在需要浏览多页结果时，只会反复 scroll 向下滚动，从不尝试点击分页控件（"Next"、页码按钮等），即使 AXTree 中分页元素可见可点击。

**实例**：task_43 要求在 Cars + trucks 分类中找到所有红色车辆及价格，Agent 起始于搜索结果第 4 页（URL 含 `iPage=4`）。Agent 连续执行 5 次 `scroll [0, 300]`，其中后 3 次页面完全未变化（已到底部），最终因 `page_unchanged_streak` 触发结束，全程未尝试翻页。

**影响**：涉及多页结果的任务（如"列出所有满足条件的商品"）基本无法完成。模型缺乏"当前页已无更多内容 → 应点击下一页"的策略意识，这是模型自身的能力局限，非脚手架或环境问题。

---

## 方法论说明

- 以上缺陷通过 DOM 和 SoM 的 digest 数据交叉验证确认
- 判定标准：同一 task 在不同模式下因相同原因失败，或缺陷机制与观测表征逻辑无关
- 后续如发现更多共性缺陷，将持续补充

---

*生成时间：2026-04-07*
