# B1 三模式共性脚手架缺陷分析

> 本报告记录 DOM / SoM / Vision 三种观测模式**共同受影响**的脚手架缺陷。
> 这些缺陷与观测表征无关，属于站点功能限制或 VWA 环境的结构性问题。
> 各模式专有缺陷见 `B1_DOM_digest.md` / `B1_SOM_digest.md`。

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

## 6. JavaScript confirm 弹窗不可交互（VWA 框架级缺陷）

Classifieds 站点的"Delete"操作会触发浏览器原生 `confirm()` 对话框（"Are you sure you want to delete this listing?"）。VWA 的 Playwright 默认不自动接受 `confirm()` 弹窗，导致删除操作实际被取消。

**机制**：
1. Agent 点击 Delete 链接 → 浏览器弹出 `confirm()` 对话框
2. Playwright 没有注册 `page.on('dialog', dialog => dialog.accept())` 处理器
3. 对话框被自动 dismiss（默认行为 = 取消）→ 删除未执行
4. 页面保持不变，agent 再次点击 Delete，陷入循环

**典型案例**：task_5 要求删除白色车 listing。SoM 模式下 agent 正确导航到 My account → 找到 listing → 点击 Delete（eid=1983），但连续 3 次点击页面均无变化（`page_changed=False`），最终因 cycle 停止。DOM 模式同样失败，表现完全一致。

**影响范围**：所有涉及删除操作的任务。三种模式均受影响，与观测方式无关。Agent 的操作逻辑完全正确，失败完全归因于环境限制。

**与 §3 `<select>` 缺陷的关系**：同属 VWA Playwright 环境对浏览器原生 UI 元素（下拉菜单、confirm 弹窗）的支持不完整。

---

## 7. VWA `ua_match` 评测器噪声导致 N/A 任务 False True（10 例）

VWA 对"不可行任务"（reference answer = "N/A"）的评测使用两级判断：
1. **Exact match**：agent 答案是否精确等于 "N/A"
2. **`ua_match` fallback**：调用 GPT-4o-mini 判断 agent 的回答是否与 `string_note`（不可行原因）"语义一致，即使是隐式的"

Classifieds 站有 **10 个 N/A reference task**（24, 135, 164, 167, 189, 191, 194, 195, 196, 220），DOM 模式下**全部 10 个被误判为 success=1.0**。SoM 模式已完成的 N/A task（24, 189, 195, 196）**同样全部 score=1.0**——ua_match FP 与观测模式无关。

### 误判机制

分两类：

**类型 A：空 answer 匹配（7 例）**
Agent 未调用 finish（达到 max_steps 或 cycle 早停），runner 自动追加 `create_stop_action("")`。空字符串送入 `ua_match`，GPT-4o-mini 将空回答解读为"agent 无法完成任务"，与 `string_note` 的不可行原因判定为 "same"。

**类型 B：错误答案被误判（3 例：167, 189, 196）**
Agent 提交了具体商品名作为答案（如 task_196 提交了一辆黑色卡车 "2022 GMC Sierra 1500 AT4X"，但任务要求红色卡车）。`ua_match` 的 prompt 将 agent 答案标记为 `reported unachievable reason`，GPT-4o-mini 将"找到了非红色车"脑补为"隐式说明不存在红色车"，判定 "same"。

### 根因：`ua_match` prompt 设计缺陷

```
actual unachievable reason: There are no red trucks from Maryland worth at least 50000 dollars.
reported unachievable reason: 2022 GMC Sierra 1500 AT4X Frederick, MD $70,826
→ GPT-4o-mini 判定 "same"（错误）
```

1. **字段名误导**：agent 的答案被标记为 "reported unachievable reason"，但 agent 实际是在提交成功答案，不是报告失败原因
2. **"even if implicitly" 过于宽松**：给 LLM 过度解读空间
3. **缺前提验证**：未先判断 agent 是否意识到任务不可行

### 影响

| 指标 | Raw | 排除 N/A FP 后 |
|------|-----|----------------|
| DOM 成功数 | 21 | 11 |
| DOM SR | 8.97% | 4.70% |

结合 visual FP（19 例），DOM 的 21 个 success 中：10 个 N/A FP + 部分 visual FP（两类有重叠），真实成功极少。

### 模式对比

| 模式 | N/A task 完成数 | 其中 FP | 原因 |
|------|---------------|---------|------|
| DOM | 10/10 | **10** | 提交错误答案（Type B）或耗尽步数空 answer（Type A）→ ua_match 误判 |
| SoM | 4/10（进行中） | **4** | 同上——搜索/scroll 循环→早停或 max_steps→空 answer→ua_match 误判 |

### 根因：Agent prompt 无 N/A 出口（脚手架缺陷）

ua_match FP 与观测模式无关，根因是 **agent prompt 没有 N/A 提交指引**：

- Rule 1: *"Do NOT answer or finish immediately. You MUST navigate to find the item."*
- Rule 4: *"NEVER give up early. If you don't see the item, SEARCH for it."*
- Finish action 定义中无 "如果任务不可行，answer 填 N/A" 的说明

结果：agent 在 N/A task 上必然陷入搜索/scroll 循环直到被截断，runner 兜底填空 answer，ua_match 将空 answer 当作"隐式报告不可行"。DOM 和 SoM 走的是同一条路径，FP 机制完全一致。

**结论**：N/A FP 是评测器缺陷（ua_match 过于宽松）叠加脚手架缺陷（prompt 无 N/A 出口）的共同产物，与观测模式无关。所有模式的 N/A task 成功均应视为假阳性。

---

## 8. 任务参考图片未传递给模型（脚手架缺陷）

部分 VWA 任务在 config 中包含 `"image"` 字段，指向参考图片（如 `input_images/classifieds/task_44/input_0.png`），任务 intent 通过代词引用该图片（"I recall seeing **this exact item** on the site, help me find the most recent post of it."）。

**问题**：`runner.py:924` 只传递 `task.intent`（纯文本），**从未将 `task.raw_task["image"]` 传给模型**。模型唯一收到的图像是当前网页截图（`obs.image`），不是任务参考图。

**表现**：
- task_44：Agent 在搜索框输入字面量 `"item name"`（不知道搜什么），随机点了一个 item 后 finish → score=0
- task_45：同样输入 `"item name"`，反复点击和回退 listing，最终提交错误 item（id=54140，目标 id=45196）→ score=0

Agent 的思维链中**完全没有关于参考图片内容的描述**（如物品外观、颜色、类别），证实模型确实未收到图片。

**影响范围**：所有含 `"image"` 字段的任务。三种模式（DOM/SoM/Vision）均受影响——即使 Vision 模式能看到网页截图，也看不到任务参考图。这不是模型能力问题，而是脚手架未将任务图片传递到 prompt 中。

**修复方向**：在 runner 构建 instruction 时，检测 `task.raw_task.get("image")`，将参考图片以 base64/PIL 形式追加到多模态 prompt 中。但这会改变所有条件的 prompt schema，需在 B1 baseline 完成后的 B2 或下一轮实验中统一修复。

---

## 方法论说明

- 以上缺陷通过 DOM 和 SoM 的 digest 数据交叉验证确认
- 判定标准：同一 task 在不同模式下因相同原因失败，或缺陷机制与观测表征逻辑无关
- 后续如发现更多共性缺陷，将持续补充

---

*生成时间：2026-04-07*
*更新时间：2026-04-10，追加 §8 任务参考图片未传递缺陷*
*更新时间：2026-04-10，修正 §7：SoM 同样存在 ua_match FP（原 "SoM 0/10" 有误），补充 prompt 无 N/A 出口的脚手架缺陷分析*
