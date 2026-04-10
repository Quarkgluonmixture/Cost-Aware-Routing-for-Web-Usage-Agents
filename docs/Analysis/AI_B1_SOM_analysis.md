# B1 SoM Baseline 失败原因分析报告

> 数据来源：`digest_som.jsonl`（仅含 SoM 模式数据，持续增长中）
> 归因方法：GLM-5.1 + SoM 专属归因规则（S1-S5）
> 本报告**仅分析 SoM 模式**，不引用 DOM / vision 模式数据。
> 三模式共性缺陷见 `AI_B1_Overall_analysis.md`。
>
> **状态：digest 尚未全部完成，以下为已有数据的中间分析。**

---

## 一、非表征方法的脚手架缺陷

以下缺陷与 SoM 表征无关，属于站点功能或 VWA 环境的结构性限制。

### 1.1 地点过滤不可达（3 例：task_58, 72, 74）

Classifieds 站点没有有效的地点筛选 UI。Agent 只能将地名（"Delaware"/"Ohio"/"Washington DC"）作为搜索关键词输入搜索框，但搜索框按商品名匹配，不按地点过滤。

**表现**：
- task_58：搜索"blue chair Washington DC"无结果，无法筛选特定地区商品
- task_72：重复搜索"delaware"三次，搜索框无法理解地点约束
- task_74：将"Ohio"作为搜索关键词，结果无法限定到目标州

**归因**：站点功能缺失，与观测模式无关。DOM 模式下同样失败（已在 DOM digest 中确认）。

### 1.2 编辑页面字段不可达（2 例：task_4, 75）

商品编辑页面的价格/描述输入框不在当前 viewport 中，Agent 持续滚动但无法定位目标字段。

**表现**：
- task_4：编辑页面中价格字段未在可见区域，Agent 持续滚动未找到
- task_75：错误编辑了冰箱列表而非白色花瓶，且在编辑页面无法定位价格字段

**归因**：页面布局 + viewport 限制，与观测模式无关。DOM 模式下同样失败。

---

## 二、SoM 表征结构性缺陷（待 digest 完成后补全）

> 以下为已有 70 条 digest 的中间统计，最终数据待 digest 全部完成后更新。

### 已识别的 SoM 特有失败类型

| som_failure_type | 数量 | 含义 |
|------------------|------|------|
| text_over_vision | 30 | 模型依赖 SOM_MARKS 文本而忽略截图视觉信息 |
| 标注遮挡 | 3 | 青色 mark 框遮挡关键文字/图片区域 |
| 空间布局丢失 | 2 | SoM 文本索引丢失了元素的空间位置关系 |
| ID幻觉 | 1 | 模型引用不存在的 element_id |
| location_filter | 1 | 地点过滤不可达（非表征问题，见 1.1） |

### text_over_vision（30 例，86% 的表征类脚手架问题）

4B 模型系统性忽略 SoM 截图中的视觉细节（颜色、图片内容），主要依赖 `[SOM_MARKS]` 文本索引进行决策。这导致 SoM 模式在视觉类任务上的表现与 DOM 模式接近——虽然视觉信息通过截图提供了，但模型不使用。

#### 典型案例

**task_14：行列混淆（空间布局丢失 + text_over_vision）**

Gallery 模式下 3×2 网格，任务要求找"second row"的画。截图清楚显示第一行 3 个、第二行 3 个，但 SoM marks 按 DOM 顺序线性排列。模型将列表中"第二个"item（Ocean View Oil Painting，第1行第2列）误认为"第二行"的画，与 DOM 模式犯完全相同的错误。模型没有利用截图中的网格空间布局来判断行列关系，退化为依赖文本序号。

**task_16：文本名称匹配替代视觉内容识别**

任务要求找"图片中有 coffee mug"的 item。页面有 Mimi fridge、cuisinart coffee grinder、cuisinart coffee maker 三个 listing。模型从 SoM 文本 "cuisinart coffee maker" 中看到 "coffee" 就做了文本级联想（coffee → coffee mug），没有检查缩略图中是否真的有咖啡杯。实际上 coffee maker 图片是一台咖啡机，不是咖啡杯。

### visual scope anchoring：截图锚定导致任务降级（模型能力）

4B 模型在 SoM 模式下将"当前截图可见内容"等同于"所有可操作范围"，无法推理出可以通过导航到达截图中不存在的页面/站点。当任务涉及跨站操作时，模型因为截图中看不到目标站点，就主动砍掉相关子任务。

**task_36 & task_37（跨站比价任务）**

两个 task 都要求在 Classifieds（OSClass）和 OneStopMarket（Shopping）之间比较价格。起始页是 Classifieds 首页，截图中只有 OSClass 的 Latest Listings。

- task_36 thought："the current screenshot **does not show** any reference to OneStopMarket, and **no item is specified** for comparison... I will **assume** the task is to find the cheapest listing on OSClass" → 直接放弃跨站比较，把任务降级为"在 OSClass 找最便宜的"
- task_37 thought："the screenshot **does not show** any item from OneStopMarket, and no search or navigation to OneStopMarket is **indicated**" → 同样因截图中不可见就判定不可达

错误归因链：**截图中没有 OneStopMarket → 认定不可达 → 主动砍掉跨站子任务**。截图反而成了思维的牢笼。对比之下，DOM 模式不会产生"截图中没看到"这种推理——虽然 DOM 也可能因其他原因失败，但不会因视觉锚定而主动放弃任务目标。

### Gallery 聚合类任务的 premature finish（模型能力）

4B 模型在需要扫描多个 item 才能回答的 gallery 聚合任务中，系统性地在 1 步内 premature finish，不做任何滚动或翻页来获取完整信息。

**task_41（找最贵的 item 并报价）**：Gallery 页面只显示当前 viewport 中的几个 listing。Agent 在 step 0 就直接从可见区域选了一个 item finish，没有向下滚动查看是否有更贵的。

**task_42（列出所有红色车辆及价格）**：需要遍历多页搜索结果。Agent 在 step 0 从当前可见结果中挑了几个 finish，完全未翻页。结果遗漏了大量 item。

**task_43（Cars + trucks 分类中找所有红色车辆）**：同上，Agent 起始于搜索结果某页，只看当前 viewport 可见内容就 finish，未尝试翻页或滚动到底部。

**共性模式**：
1. 三个 task 都在 **1 步内 finish**（premature finish）
2. Agent 将"当前 viewport 可见内容"等同于"全部数据"
3. 行列混淆（task_41）、颜色识别不完整（task_42/43）、不翻页（全部）

**与 visual scope anchoring 的关系**：这与 task_36/37 的截图锚定属于同一认知缺陷的不同表现——模型无法推理"当前看到的不是全部，需要主动探索更多"。区别在于 36/37 是跨站不可达，41/42/43 是同站内不遍历。

**归因**：模型能力问题（缺乏"数据不完整 → 需翻页/滚动"的元认知），非 SoM 表征缺陷。DOM 模式下也存在翻页缺陷（见 B1_Overall §5），但 SoM 模式因截图锚定可能加剧此倾向。

### 视觉自信加速错误 finish（SoM 特有加剧机制）

SoM 模式因能看到缩略图，模型反而"更有信心"基于不完整的首页信息做出（错误的）快速判断，比 DOM 模式更快地错误 finish。视觉信息在此成为过度自信的来源而非决策辅助。

**task_84（找 selfie 拍摄的最贵戒指，答案 $6000）**：

任务需要搜索 "ring" → 浏览结果 → 检查每个 listing 的图片是否为 selfie。模型在首页看到 4 个 Latest Listings，其中 Engagement Ring $2200 的缩略图是一张 IGI 证书文件照片。模型 thought："none of the visible listings have images that can be identified as selfies"，但仍然选了 Engagement Ring finish。**没有搜索 "ring"，没有翻页，1 步即退出**。

**task_85（找图片中不显示实物戒指的最贵戒指，答案 $3200）**：

同样需要搜索+遍历。模型在首页看到 Engagement Ring 缩略图是证书（不是实物戒指），认为符合条件，thought："its image shows a document or paperwork, not the physical ring"，直接 finish 回答 $2200。**1 步退出，答案错误**。

**SoM 特有的加剧机制**：

| 因素 | DOM 模式 | SoM 模式 |
|------|----------|----------|
| 能否看到缩略图内容 | 不能（纯 AXTree） | 能（截图+标注） |
| 视觉判断是否触发 | 不会——无图可判 | 会——看到图就做判断 |
| 1 步 finish 的"信心来源" | 只有文本名称 | 文本名称 + 视觉确认 |
| 是否搜索 "ring" | 同样不搜索（共性缺陷） | 同样不搜索，但视觉信息让模型更确信当前信息已足够 |

核心悖论：SoM 提供了更丰富的信息（截图），但 4B 模型缺乏"当前视野 ≠ 全部数据"的元认知，视觉信息反而强化了"已有足够信息做决策"的错觉。DOM 模式虽然也不会搜索，但至少不会因"看到了图片"而产生额外的虚假确信。

**与 visual scope anchoring 和 gallery premature finish 的关系**：三者共享同一底层缺陷（将当前视野等同于全部数据），但触发机制不同：

- visual scope anchoring（task_36/37）：截图中看不到目标 → 判定不可达
- gallery premature finish（task_41/42/43）：只看首屏 → 不翻页
- **视觉自信 premature finish（task_84/85）**：看到图片 → 视觉判断增强错误信心 → 跳过搜索

**归因**：模型能力 × SoM 表征交互效应。不搜索是共性缺陷，但 SoM 的视觉信息加剧了 premature finish 的倾向。

### 视觉误匹配：语义级匹配替代图片级匹配（模型能力）

"find this exact item" 类任务提供一张参考图片，要求 agent 在站点中找到**同一张图片对应的 listing**。4B 模型在 SoM 模式下只做语义类别匹配（"都是 eggs"/"都是 painting"），不做像素级或视觉细节匹配，导致选中同类但完全不同的 item。

**task_141（找鸡蛋，Farm + garden 分类）**：

- Intent 图片：纸盒装 12 枚棕色鸡蛋，厨房台面上
- 正确答案：item 41568
- 模型行为：搜索 "eggs" → 找到 "20 eggs for $5.00"（item 35914）→ 进入详情页 → finish
- 模型选的图片：**红色圆盘上的鸡蛋在孵化器里**——与参考图在容器、背景、数量、场景上完全不同
- 匹配逻辑：thought = "The listing for '20 eggs for $5.00'... This appears to be the item" — 纯语义匹配（eggs = eggs）

**task_142（找 $20 以下的画，回答所在州）**：

- Intent 图片：圆形树脂画，蓝绿色海浪+沙滩抽象图案，标价 $15
- 正确答案：item 22310，Pennsylvania
- 模型行为：搜索 "painting" → 找到 "matador painting on wooden plate" $15（item 14761）→ 进入详情页 → finish "Maryland"
- 模型选的图片：**斗牛士写实木盘画**——与参考图在主题、色调、风格上毫无相似度
- 匹配逻辑：关键词 "painting" + 价格 < $20 → 符合 → 不核对图片

**共性模式**：模型将"find this exact item"降级为"find any item in this category"。SoM 截图中 listing 缩略图清晰可见，但模型不将缩略图与 intent 参考图做视觉对比，只用文本关键词（eggs / painting）+ 约束条件（分类 / 价格）匹配。即使进入详情页看到大图后，thought 中也从未提及"这张图与任务给的图是否一致"。

**归因**：模型能力（缺乏图片-图片视觉对比能力）。4B 模型的视觉理解停留在"识别物体类别"层面，无法判断两张图是否为同一张照片。这不是 SoM 表征问题——SoM 已经提供了足够的视觉信息，模型不使用。

### Navigate-to 任务：到达目标但不会 finish（SoM 特有缺陷）

在 `url_match` 类型的 "Navigate to" 任务中，SoM 模式能利用视觉信息正确识别目标 item 并导航到详情页，但**进入详情页后不知道自己已完成任务**，继续探索直到离开正确页面或耗尽步数。

#### 统计

Classifieds 共 32 个 navigate-to + url_match 任务（几乎全是视觉条件：sunset/grass/human hand 等）。

| 指标 | DOM (32 tasks) | SoM (13 tasks, 进行中) |
|------|---------------|----------------------|
| 成功 | 1/32 (3.1%) | 3/13 (23.1%) |
| 曾到达正确 URL 但最终失败 | 0 | **2** (task_124, 152) |
| 失败原因 | 看不到图片，根本找不到目标 | 找到目标但不 finish |

DOM 几乎全败是因为这些任务需要**视觉判断**（"image is set on grass"），纯 AXTree 无法完成——唯一成功的 task_153 靠文本名称 "matador painting" 碰巧匹配。DOM 不存在"到达了但不 finish"的问题，因为它过不了第一关。

#### SoM 成功 vs 失败的行为对比

**唯一的主动 finish（task_153，2 步）**：
- Step 0: click 进入详情页
- Step 1: thought = "The painting in the screenshot **matches** the item title on the **current page**" → `finish` ✅

**运气型成功（task_130，9 步）**：
- Step 0: click 进入正确详情页（item 19604）
- Step 1-8: 继续在详情页探索，反复点击 Related Listings 缩略图，但**所有 click 都因元素滚出视野（bbox 负坐标）而未生效** → max_steps 时仍在正确 URL → `url_match` 判定成功
- 如果任何一次 click 生效，就会跳到错误 item 而失败

**运气型成功（task_151，7 步）**：
- Step 1: click 进入正确详情页（item 22560）
- Step 2-6: 反复 scroll + click 图片，始终未离开 → 同上

**失败（task_124，21 步）**：
- Step 1: click 进入正确详情页（item 10702，Standing Lamp on grass）✅
- Step 2: 在详情页点击图片
- Step 3-4: scroll → click Related Listings → **成功跳到 item 2364（couch and love seat）**
- Step 5-20: 在错误 item 的 Related Listings 里死循环，反复 click/scroll 无效
- 最终 URL = item 2364（错误）→ fail

**失败（task_152，18 步）**：
- Step 1: click 进入正确详情页（item 81346，Sony Camera with human hand）✅
- Step 4: click Related Listings → **跳到 item 43278（Kodak camera）**
- Step 5: `back` 回到正确页 → Step 6: 又跳走
- Step 8-17: 在 item 19663 上死循环，反复点击图片进裸图片 URL 再 back
- 最终 URL = item 19663（错误）→ fail

#### 根因：缺乏页面类型感知

模型的 thought 在详情页上始终重复 "I need to find the item whose image is..."——它把详情页当作列表页处理，把 Related Listings 当作"更多候选 item"。核心缺陷是 **4B 模型不具备列表页 vs 详情页的页面类型识别能力**：

| 页面状态 | 正确行为 | 模型实际行为 |
|---------|----------|-------------|
| 列表页，未找到目标 | 搜索/翻页 | ❌ 有时直接 finish（旧问题，Rule 6 已修） |
| 列表页，找到目标 | 点击进入详情页 | ✅ 能做到 |
| **详情页，已到达目标** | **finish** | ❌ 继续探索 Related Listings，不知道自己已完成 |

详情页有明确的区分信号（面包屑导航 "Classifieds > Category > Item Name"、单个 item 大图+描述+联系方式、URL 含 `page=item&id=`），但模型不利用这些线索判断页面类型。

**归因**：模型能力（页面类型感知缺失）。这是 SoM 特有的实际影响——DOM 模式在这类视觉 navigate-to 任务上因看不到图片而根本无法到达目标，所以不会暴露"到达但不 finish"的问题。SoM 3/13 的成功中，仅 1 个是主动 finish，另 2 个是**被困在正确页面上的运气结果**。

### 表单交互 action type 选择缺陷（模型能力）

4B 模型在需要向输入框填写内容时，反复生成 `click` 动作而非 `type` 动作，导致即使正确定位到目标输入框也无法完成输入。

**task_19（价格过滤失败）**：Agent 需要在 $80-100 价格范围内搜索海景画。Step 3 中 Agent 正确点击了 Min. 价格输入框（eid=2596，确认是 textbox 元素而非 StaticText 标签），但 action_type 为 `click` 而非 `type`。click 只让输入框获得焦点，不输入任何值。随后 retry scroll 触发，Agent 被推走后再未回来输入价格。后续 step 9、step 14 同样模式——反复 click 价格区域元素但从不 type。

Agent 最终只能靠手动翻页（第1→2→3页）逐页浏览，在 17 步后选了一个 $100 的灯塔画 finish。

**与 task_6/task_219 的关系**：task_6 和 task_219 是更基础的 grounding 失误（点击 StaticText 标签而非相邻 textbox）。task_19 更进一步——定位正确但 action type 错误。两者都反映 4B 模型对表单输入交互的 action decomposition 能力不足：无法将"设置价格过滤"分解为 `type [Min] "80"` → `type [Max] "100"` → `click [Apply]` 的动作序列。

---

## 三、方法论说明

- **Digest 文件**：`digest_som.jsonl`（与 DOM/vision 物理隔离）
- **SoM 专属归因规则**：S1(标注遮挡) / S2(颜色混淆) / S3(空间布局丢失) / S4(text_over_vision) / S5(ID幻觉)
- **额外输出字段**：`som_visual_used`、`som_mark_occlusion`、`som_failure_type`

---

## 四、基础设施噪声分析

> 以下问题在 2026-04-07 的 step 级数据审查中发现，属于实验运行环境引入的系统性噪声，已在下一次重跑前修复。

### 4.1 busy:1 页面加载中间态消耗步数预算（SoM 严重度 2× DOM）

**现象**：VWA 底层 `wait_for_load_state("networkidle", timeout=2000)` 对远程 Classifieds 站点不够长。SoM 模式因处理链更长（截图 → CDP bbox → 标注渲染），页面在 2s timeout 内更难加载完成。

**机制**：Runner 在 LLM 调用**之后**才检查 `busy: 1` 并覆盖 action 为 `wait`，导致 LLM 推理已浪费且 `wait` 占用 `max_steps` 步数名额。

**SoM 条件影响**：

| 指标 | SoM | DOM（对比） |
|------|-----|-------------|
| 受影响 task 数 | **137/171 (80.1%)** | 130/234 (55.6%) |
| busy:1 步数占总步数比 | **502/2052 (24.5%)** | 422/3584 (11.8%) |
| 浪费 tokens | 1,058,785 | 561,278 |
| 浪费延迟 | ~3.3 小时 | ~2.9 小时 |
| 到 max_steps 失败且有 busy:1 步的 task | 26 | 55 |
| 这些 task 平均浪费步数 | **9.8 步 (32.8% 预算)** | 4.7 步 (15.7%) |

**公平性问题**：SoM 的 busy:1 率是 DOM 的 **2 倍**（24.5% vs 11.8%），导致 SoM 的有效步数预算被大量侵蚀。DOM vs SoM 的成功率/效率对比被系统性地偏向 DOM。SoM 条件中到 max_steps 失败的 task 平均损失近 **1/3 步数预算**。

**修复**（已合入 runner.py）：将 `busy: 1` 检查提到 LLM 调用之前，命中时跳过推理并执行免费 wait（不消耗 step_idx），配置 `busy_wait_limit=5` 防无限循环。下次重跑将消除此噪声，两个条件的有效步数预算将恢复公平。

### 4.2 SoM 低 mark_count 加载中间态

与 busy:1 相关但角度不同：在 busy:1 页面上，SoM 标注器只能检测到极少的可交互元素。

| mark_count | 步数 | 占 SoM 总步数 | 特征 |
|------------|------|--------------|------|
| 1 | 433 | 21.1% | dom_complexity=3, text_length≈178, 页面加载中间态 |
| 2 | 21 | 1.0% | dom_complexity=4, agent 误导航到裸图片 URL |

mark_count=1 的 433 步中 94.5% 的 action 为 `wait`（busy:1 guard 覆盖或 agent 自行判断）。

mark_count=2 的 21 步分布在 4 个 task（121, 132, 135, 146），agent 导航到了 `/oc-content/uploads/.../*.png` 裸图片 URL，页面只有一个 image 元素。这属于模型导航错误，非 SoM 问题。

**资源浪费**：这些低 mark 步合计消耗 ~15% 的 SoM 条件总预算（tokens 14.5%, latency 17.5%, cost 14.6%）。busy:1 修复后，mark_count=1 的情况将大幅减少。

### 4.3 SoM 标注密度问题（非 bug，已知局限）

**现象**：部分页面截图出现大面积蓝色区域（如 task_0 step_5），看似"全选"但实际原因不同：

**原因 1：浏览器文本选中状态（task_0 等）**

Agent 错误地将 `type` action 发送到 RootWebArea 而非搜索输入框。VWA 的 type 实现会先执行 `Meta+A`（全选），导致页面文本被选中（蓝色高亮）。**这是模型错误，不是 SoM 问题**——原始截图中已有蓝色全选效果，SoM 标注只是叠加了矩形框。

SoM 条件中 13 步 / 11 task 受影响（DOM 条件为 24 步 / 18 task）。SoM 受影响更少，可能因为视觉标注帮助模型更好地识别了正确的输入框。

**原因 2：SoM 矩形框密集重叠**

我们的 `som.py` 对 AXTree 中所有带 `[id]` 的元素画 `#00BCD4` 矩形框（包括 StaticText、LayoutTable、LayoutTableRow 等不可交互元素），而 VWA 原版 SoM 只标注 `Interactable=True` 的元素且使用多色。在密集列表页（一个 listing 有 link + image + table + row + 7 个 StaticText ≈ 10 个元素），大量重叠的单色矩形框形成视觉噪音。

**影响**：对 4B 模型的视觉理解可能产生干扰（text_over_vision 问题的加剧因素），但这是当前 baseline 的一部分，属于 SoM 实现差异的如实记录。

### 4.4 JSONL 重启噪声

SoM 条件中 1 个 task 受影响（DOM 为 12 个）。已通过 dedup 逻辑修复，所有 JSONL 读取器在分析时自动跳过 stale lines。

### 4.5 Evaluator 错误

3 个 SoM episode 因 evaluator 基础设施问题失败：
- **task 24, 135**：OpenAI API key 缺失导致 `evaluator_error:401`。已通过 `reeval_phase1.py` 离线重评 → 仍为失败（agent 提交空答案，正确答案为 "N/A"）
- **task 160**：`program_html` 评测超时。需 live browser，无法离线重评

---

*生成时间：2026-04-07，基于 digest_som.jsonl 中间数据（70/91 条）*
*更新时间：2026-04-07，追加第四节基础设施噪声分析*
*更新时间：2026-04-10，追加 gallery 聚合 premature finish（41/42/43）、visual scope anchoring（36/37）、action type 缺陷（19）、text_over_vision 案例（14/16）*
*更新时间：2026-04-10，追加视觉自信加速错误 finish（84/85）——SoM 截图信息反而强化 premature finish*
*更新时间：2026-04-10，追加 navigate-to 页面类型感知缺失（124/130/151/152/153）——到达目标详情页但不 finish*
*更新时间：2026-04-10，追加视觉误匹配（141/142）——语义类别匹配替代图片级匹配*
