# B1 SoM Baseline 分析报告（Classifieds）

> **[DATA STALE]** 本报告数据基于 §33/§34/§36 修复前的运行结果。参考图片传递修复后 adjusted 指标可能变化，待重跑后更新。

> 数据来源：`digest_som.jsonl`（186 条，仅含 SoM 模式失败 episode）
> 归因方法：GLM-5.1 + SoM 专属归因规则（S1-S5）+ 人工定性分析交叉验证
> 置信度：57.0% high / 43.0% medium
> 本报告**仅分析 SoM 模式**，不引用 DOM / vision 模式数据。
> 三模式共性缺陷与定量对比见 `B1_findings.md`。

---

## 总体概况

| 指标 | 数值 |
|------|------|
| SoM condition 总 episode 数 | 234 |
| 成功 | 48 (20.51%) |
| 失败 | 186 (79.49%) |
| Digest 覆盖率 | 186/186 (100%) |
| N/A Adjusted SR | 16.96%（38/224，扣除 10 个 N/A FP） |
| Visual Adjusted SR | 20.51%（无 visual FP） |
| 失败 episode 平均步数 | 11.8 步 |
| 全 episode 平均步数 | 11.8 步 |
| max_steps 命中率 | 19.4%（36/186 failures） |

### 失败原因分布

| 失败原因 | 数量 | 占失败比 |
|----------|------|---------|
| fail_incomplete_or_stuck | 56 | 30.1% |
| fail_early_finish | 29 | 15.6% |
| fail_finish_wrong_url_not_found | 28 | 15.1% |
| fail_max_steps_target_unreachable | 28 | 15.1% |
| fail_finish_eval_mismatch | 15 | 8.1% |
| fail_finish_empty_answer | 9 | 4.8% |
| fail_no_progress | 7 | 3.8% |
| fail_max_steps_search_repeat | 3 | 1.6% |
| fail_parse_error | 3 | 1.6% |
| fail_max_steps_click_back_loop | 3 | 1.6% |
| fail_finish_claim_missing | 2 | 1.1% |
| fail_max_steps | 2 | 1.1% |

### 脚手架 vs 模型能力归因

| 归因类型 | 数量 | 占比 |
|----------|------|------|
| 脚手架/表征结构性缺陷 | 63 | 33.9% |
| 模型能力问题 | 123 | 66.1% |

对比 DOM 模式（脚手架 44.1% / 模型 55.9%），SoM 的脚手架问题占比显著下降（-10.2pp），说明 SoM 截图有效缓解了 DOM 的信息瓶颈。

### SoM 特有字段统计

**som_failure_type 分布**（归因可用的 failure episodes）：

| som_failure_type | 数量 |
|------------------|------|
| text_over_vision | 56 |
| 不适用（非 SoM 表征问题） | 46 |
| 标注遮挡 | 4 |

**som_visual_used 分布**：

| 模型是否实际使用了视觉信息 | 数量 |
|---------------------------|------|
| 否 | 67 |
| 是 | 41 |

**som_mark_occlusion 分布**：

| SoM 标注是否遮挡关键信息 | 数量 |
|--------------------------|------|
| 否 | 91 |
| 是 | 9 |

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

## 二、SoM 表征结构性缺陷

### 已识别的 SoM 特有失败类型

| som_failure_type | 数量 | 含义 |
|------------------|------|------|
| text_over_vision | 56 | 模型依赖 SOM_MARKS 文本而忽略截图视觉信息 |
| 标注遮挡 | 4 | 青色 mark 框遮挡关键文字/图片区域 |
| 空间布局丢失 | 2 | SoM 文本索引丢失了元素的空间位置关系 |
| 扁平化容器误点 | — | 扁平 marks 丢失层级，模型 click 非交互容器节点 |
| label/textbox 混淆链 | — | 扁平 marks 混淆表单 label 与 textbox → scroll → marks 丢失 → P4 |
| 首页视觉干扰 | — | 首页缩略图截断搜索流程，agent 点击无关 item |
| ID幻觉 | 1 | 模型引用不存在的 element_id |
| location_filter | 1 | 地点过滤不可达（非表征问题，见 1.1） |

### text_over_vision（56 例，87.5% 的表征类脚手架问题）

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

### 首页视觉干扰 → 放弃搜索流程（视觉自信的行为变体）

与 task_84/85 的"首页即 finish"同源，但行为路径不同：agent **正确执行了 category select**，然后被首页 Latest Listings 的缩略图吸引，**放弃搜索流程直接点进 homepage 上的无关 item**，最终在错误的商品上 finish。

**task_65/66/67（三个同模板任务："Find the cheapest video game item where I can roleplay the situation in the image"）**：

三个 task 意图图片不同（足球赛/橄榄球/其他），正确答案分别是 item 28239/6175/7114，但 SoM agent 的轨迹**几乎完全相同**：

1. step 0：`select_option` "Video gaming" ✓（类目选对）
2. step 1：**不是**点 Search 或输入关键词，而是点了 `[id=282] link 'xbox series x / with extras'`——这是首页 Latest Listings 的第二张缩略图（$350）
3. step 2-3：从 xbox 详情页导航到 Video Gaming 搜索结果，找到 "tom clancy R6 lockdown"（$5.00，item 83357）
4. step 4：finish "tom clancy R6 lockdown for $5.00" ← 三个 task 给出**完全相同的错误答案**

**三个不同意图图片 → 同一个错误答案**：agent 完全忽略了 intent 图片中的场景差异（足球/橄榄球/...），只关注"cheapest video game"，被首页 xbox 缩略图的视觉吸引力捕获。

**DOM 模式对比**：
- task_65 DOM：3 步 finish（也失败，DOM 同样不够好）
- task_66 DOM：**23 步**，触发 action_failed/page_unchanged，未 finish（至少尝试搜索）
- task_67 DOM：**30 步**，触发 action_failed，未 finish（持续探索）

DOM 模式下 task_66/67 不会被首页图片干扰（看不到缩略图），因此持续搜索直到超时。SoM 模式的截图让 agent "看到" xbox 后立即行动，跳过了本应执行的搜索流程。

**与已有 task_84/85 的统一视角**：

| 变体 | 首页行为 | 搜索流程 | 失败机制 |
|------|---------|---------|---------|
| task_84/85 | 看到 Ring → 立即 finish | **完全跳过** | 1 步，premature finish |
| **task_65/66/67** | 选对类目 → 看到 xbox → 点击 | **中断后偏移** | 5 步，沿错误路径 finish |

共性：首页 Latest Listings 的**视觉吸引力**截断了正常的"select → search → browse → compare"工作流。4B 模型将"视觉上看起来相关的 item"等同于"搜索结果"，跳过了关键的过滤/比较步骤。

**归因**：模型能力（intent 图片理解不足 + 缺乏"首页 ≠ 搜索结果"的元认知）× SoM 表征交互（截图让首页 item 视觉可及，触发"捷径"行为）。

#### 量化：首页 select 后走偏的系统性规模

对全部 153 个首页 task 统计"前 3 步含 select_option → 后续行为"：

| 指标 | SoM | DOM |
|------|-----|-----|
| 前 3 步含 select_option | 61 (40%) | 72 (47%) |
| select 后走偏/早停 | **37/61 = 61%** | 30/72 = 42% |
| → select 循环（重复 select 同一 dropdown 不搜索） | 10 | 5 |
| → 直接点击首页 listing（SoM 独有） | **7** | **0** |
| → type 搜索但仍走偏 | 20 | 25 |
| select 后正常 type 搜索 | 25 (41%) | 43 (60%) |

**三层结论**：

1. **共性底层**（模型能力）：4B 模型缺乏"select → type keyword → click Search"的完整 action sequence 元认知。select 循环（重复选同一 dropdown）在两模式中都存在（SoM 10 / DOM 5），重叠 task（58/154/155）证实这是模型共性缺陷。

2. **SoM 独有加剧——视觉捷径**：7 个 task 在 select 后**直接 click 首页 Latest Listings 缩略图**，DOM 为 0。截图让 agent "看到可点的 item"后绕过搜索流程。

3. **总体搜索启动率差距**：SoM select 后 type 搜索仅 41%，DOM 达 60%。截图提供了"已经看到了信息"的错觉，降低了 agent 发起主动搜索的意愿。

### 视觉非确认 → 过早否定 finish（视觉自信的镜像变体）

与 task_84/85 的"视觉假阳性"对称，task_25/26/27 展示了"视觉假阴性"：模型看到缩略图但**无法视觉确认**目标属性（颜色/内饰），于是过早 finish 回答"0"。DOM 模式不存在此问题——没有视觉证据可以"不确认"，反而基于文本推理更灵活。

**task_25（红船计数，答案 1）——DOM 成功 vs SoM 失败**：

最干净的反转案例。两个模式搜 "red boats" 得到相同结果列表：

- **DOM agent（4 步，✅）**：select "Boats" 类目 → 搜 "red boats" → 点进详情页 → thought: "this is the only listing visible and it matches the date" → finish "1"。不需要视觉颜色确认，基于"搜 red boats 返回了结果"的文本推断即可
- **SoM agent（3 步，❌）**：搜 "red boats" → 看缩略图 → thought: "the listing is for a 1967 FIBRA/FISHING GEAR, **not a red boat**"、"**no visual indicator of the boat being red**" → finish "0"。从小缩略图无法辨识红色 → 否定匹配

**task_26（黄/蓝摩托计数，答案 2）——DOM stuck vs SoM 过早否定**：

- SoM（6 步）：搜 "motorcycles" → 列表日期均为 November，看不到 October 25 → scroll 一圈 → finish "0"（confidence=0.0）
- DOM（5 步）：同样搜索，陷入 stuck/page_unchanged_streak，未 finish → 两者都失败，但 DOM 至少不会给出错误断言

**task_27（RV 内饰计数，答案 3）——搜索策略错误 + 过早否定**：

- SoM（9 步）：搜 "car interior" 作为关键词（错误——任务是看图片是否展示内饰，不是搜索标题含"interior"的 listing）→ 找到的 listing 标题不含 interior → finish "0"（confidence=0.95）
- DOM（12 步）：类似困境但更多探索步骤

**机制对比**：

| 变体 | 视觉信息角色 | 结果 | 例子 |
|------|------------|------|------|
| 视觉假阳性（84/85） | 看到图 → 错误确认 → 过早肯定 | 选错 item | "缩略图看起来匹配 → finish" |
| **视觉假阴性（25/26/27）** | 看到图 → 无法确认 → 过早否定 | 答 "0" | "缩略图看不出红色 → 不存在" |

**不 scroll 是关键行为标记**：SoM agent 在搜索结果页上几乎不 scroll——task_25 仅 1 次 scroll(300px) 就 finish，task_26 scroll 3 次但只是在同一区间上下踏步，task_27 在结果页仅 1 次 scroll。与 DOM 模式形成鲜明对比：DOM agent 在同类计数任务中典型行为是反复 scroll、翻页、陷入循环——信息不足导致不敢判断。SoM 的截图让模型"看了一眼就觉得全看到了"，跳过了本应有的遍历步骤。

**与 B0 过度自信的对应**：B0(235B) 信息充足时快速判断（高 confidence finish），B1 DOM(4B) 信息不足时无限循环（不敢判断）。B1 SoM 的截图给 4B 模型"类 B0 的信息完备感"——"我看到了画面 → 如果红色存在我应该能看到"——于是产生和 B0 类似的过早 finish，但 verbalized confidence=0.0 暴露了本质差异：不是"我确信"而是"我看不到所以放弃"。

**归因**：模型能力（视觉分辨力不足以从小缩略图识别颜色）× SoM 表征交互。截图提供了虚假完备感，4B 模型将"视觉未确认"等同于"不存在"。

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

| 指标 | DOM (32 tasks) | SoM (32 tasks) |
|------|---------------|----------------|
| 成功 | 1/32 (3.1%) | 9/32 (28.1%) |
| 曾到达正确 URL 但最终失败 | 0 | **多例** (task_124, 152 等) |
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

**Paired case 佐证（task_183 vs 184）**：两个任务行为完全一致——都是 step 7-8 就找到正确 item，然后反复进出详情页 10+ 次直到 max_steps。唯一区别是截断时机：task_183 的 step 29 恰好停在详情页（`url_match` pass），task_184 的 step 29 恰好 back 回了列表页（fail）。同一行为模式，成败纯粹取决于奇偶步数。

**归因**：模型能力（页面类型感知缺失）。这是 SoM 特有的实际影响——DOM 模式在这类视觉 navigate-to 任务上因看不到图片而根本无法到达目标，所以不会暴露"到达但不 finish"的问题。SoM 的 "lucky success" 中，成功与否取决于 max_steps 截断时恰好停在详情页还是列表页，不反映真实能力。

### 表单交互 action type 选择缺陷（模型能力）

4B 模型在需要向输入框填写内容时，反复生成 `click` 动作而非 `type` 动作，导致即使正确定位到目标输入框也无法完成输入。

**task_19（价格过滤失败）**：Agent 需要在 $80-100 价格范围内搜索海景画。Step 3 中 Agent 正确点击了 Min. 价格输入框（eid=2596，确认是 textbox 元素而非 StaticText 标签），但 action_type 为 `click` 而非 `type`。click 只让输入框获得焦点，不输入任何值。随后 retry scroll 触发，Agent 被推走后再未回来输入价格。后续 step 9、step 14 同样模式——反复 click 价格区域元素但从不 type。

Agent 最终只能靠手动翻页（第1→2→3页）逐页浏览，在 17 步后选了一个 $100 的灯塔画 finish。

**与 task_6/task_219 的关系**：task_6 和 task_219 是更基础的 grounding 失误（点击 StaticText 标签而非相邻 textbox）。task_19 更进一步——定位正确但 action type 错误。两者都反映 4B 模型对表单输入交互的 action decomposition 能力不足：无法将"设置价格过滤"分解为 `type [Min] "80"` → `type [Max] "100"` → `click [Apply]` 的动作序列。

### 扁平化 marks 丢层级 → 容器/文本节点误点（SoM 特有）

SoM agent 在搜索结果列表页上反复 click 非交互节点（StaticText、LayoutTable），导致 `action_success=false` 卡死。**DOM 模式完全不存在此问题。**

#### 机制分析

DOM 与 SoM 的文本格式差异是根因：

**DOM 模式**——带缩进的层级树，角色 + 父子关系一目了然：
```
[2913] link 'Trek - Classic Rebuild' url: http://...     ← 可点击，有 url
    [2914] image 'Trek - Classic Rebuild'
[2918] LayoutTable ''                                     ← 不可点击，容器
    [2920] LayoutTableRow ''
        [2948] StaticText 'Rebuilt bike...(International Red)...'  ← 不可点击，纯文本
```

**SoM 模式**——`_extract_text_marks()` 扫描同一 AXTree，提取所有 `[N]` 行生成**扁平列表**（`som.py:24-35`），丢失缩进：
```
[SOM_MARKS]
[id=2913] link 'Trek - Classic Rebuild' url: http://...
[id=2914] image 'Trek - Classic Rebuild'
[id=2918] LayoutTable ''
[id=2948] StaticText 'Rebuilt bike...(International Red)...'
[/SOM_MARKS]
```

类型标签（`link`/`StaticText`/`LayoutTable`）**保留了**，但两个信息丢失：
1. **层级关联**：DOM 树缩进告诉你 2913(link) 和 2948(StaticText) 属于同一个 listing 卡片。扁平列表中它们只是相邻行，模型很难将 "link 'Trek - Classic Rebuild'" 与 "StaticText '...(International Red)...'" 关联为同一 listing
2. **交互性暗示**：DOM 树中 `link` 带 `url:` 属性是强信号；扁平列表中所有元素视觉权重相同，加上截图中整张卡片看起来是一个可点击整体，模型倾向**按文本内容匹配选 ID**而忽略类型标签

#### 案例

**task_39（红框自行车，step 4/5/6）**：
- Agent 想点开含 "International Red" 的 listing
- 选了 `[id=2948] StaticText '...(International Red)...'`（描述段落，bbox `[507,52,623,60]`）
- 应该选 `[id=2913] link 'Trek - Classic Rebuild'`（标题链接），但标题不含 "red"
- 连续 3 次 click 全部 `action_success=false`，最终 page_unchanged_streak 终止

**task_51（Arts+crafts 画作，step 4/5/6）**：
- Agent 想点开 Serigraph 画作
- 选了 `[id=2185] LayoutTable ''`（容器节点，bbox `[507,633,623,139]`）
- 应该选 `[id=2180] link 'Thou Shalt Not Covet...'`（标题链接）
- 同样连续 3 次失败卡死

**DOM 模式对比**：
- task_39 DOM：**30 步**（无 click 卡死），自由进出详情页
- task_51 DOM：**8 步**（无 click 卡死），正常点击链接导航

#### 与已有模式的关系

此缺陷与 P2（容器节点误点，bbox `[507,*,623,*]`）共享相同的失败特征（click 大面积非交互区域），但根因不同：

- **P2**：模型在 DOM 模式下也可能点到容器（但 DOM 中极少发生，因为树结构区分清晰）
- **本模式**：SoM 扁平化 + 截图视觉整体感 **共同导致**模型按内容相关性而非元素角色选 ID，是 SoM 表征结构的固有缺陷

**归因**：SoM 表征结构缺陷（`_extract_text_marks` 扁平化丢失层级）× 模型能力（4B 不能从类型标签推断交互性）。截图中卡片的视觉整体感进一步加剧了误选倾向。

### label/textbox 混淆 → viewport 滚动 → marks 丢失 → P4 链（SoM 特有）

SoM 扁平化 marks 将表单 label（StaticText）和相邻 textbox 并列，4B 模型无法从扁平格式中区分"标签"与"输入框"。当 type 动作命中 label 时，浏览器行为不可预期，可能触发 viewport 滚动，导致表单元素滚出视野、从 SoM marks 中**永久消失**，agent 失去导航能力。

**task_52（Arts+crafts 最新画作，url_match）——三重叠加故障**：

step 0：正确 `select_option` Arts + crafts ✓

step 1：type "painting\n" 到 `[id=138] StaticText 'Keyword'`，而非 `[id=140] textbox 'e.g., a blue used car'`。SoM marks 中两者并列：
```
[id=138] StaticText 'Keyword'
[id=140] textbox 'e.g., a blue used car' required: False
```
Agent 选了 label 而非 textbox。`\n`（Enter）触发 `scroll_changed` + `modal_state_changed`（可能是 autocomplete），页面**向下滚动**。

step 2 观测（step_002/observation_som.txt）：搜索表单**完全消失**——无 textbox、无 Search button、无 Category dropdown，只剩 `[id=2] RootWebArea` + 一堆 listing 卡片。Agent 想再搜 "painting" 但 marks 中无输入框 → type 到 `[id=2] RootWebArea`（bbox `[0,0,10,10]`，P4 根节点误操作）→ 浏览器 Ctrl+A 全选蓝 + 页面进一步下滚。

后续（step 3-29）：agent 恢复导航但陷入 click-back 循环，反复进出同一个 "matador painting" 详情页（item 14761），最终 30 步截断。

**故障链**：label 误选 → Enter 触发 scroll → 表单 marks 丢失 → P4 根节点 → 全选蓝 → 循环

**与容器误点的区别**：容器误点（task_39/51）是模型按内容选 ID 忽略类型标签；label 误选是模型无法区分表单中"标签"和"输入框"的功能角色。两者根因同源（扁平化丢失层级关联），但后者引发的滚动→marks 丢失→P4 连锁反应更具破坏性，因为 agent 直接失去了返回表单的途径。

---

## 二B、SoM 正向效应：视觉信息触发高阶推理策略（Mirage Effect 案例）

> 以下案例证实 Mirage Effect（§18）在 4B 模型上的正向表现：相同 DOM 文本信息下，SoM 截图的存在触发了质变推理路径——agent 使用了 DOM 模式从未尝试过的高阶 UI 交互。

### 价格过滤器使用（task_17，成功 vs DOM 失败）

**最干净的 Mirage Effect 正向证据。**

- Intent："Show me the cheapest bike with red handlebars between $900-950"
- **两种模式的 DOM observation 完全一致**：`heading 'Price'` → `StaticText 'Min.'` → `textbox` → `StaticText 'Max.'` → `textbox`，价格过滤 UI 在文本层面已完整暴露
- **DOM agent（失败，10 步）**：将 `bike with red handlebars between $900-950\n` 作为自然语言整串输入搜索框，价格约束被当作搜索关键词，搜索无结果后反复 scroll，最终未找到目标
- **SoM agent（成功，5 步）**：搜索 `bike with red handlebars\n` → type `900` 到 Min 字段 → type `900` 到 Max 字段 → 从结果中点击正确 item → finish

**关键差异**：SoM agent 将价格约束**分解**为独立的过滤器操作（搜索 + Min/Max 字段填写），而 DOM agent 将整个 intent 作为单一搜索字符串。这不是信息差异——两者看到了完全相同的 Price filter 结构化字段——而是**视觉信息改变了 action decomposition 策略**。截图中 Price 区域在侧边栏视觉上突出，触发了"这是一个独立操作入口"的认知。

### Sort By 使用（task_18，失败但行为值得记录）

- Intent："Show me blue iPhones that were listed most recently"
- SoM agent 在 step 5 使用了 `select_option` 选择 "Newly listed"（Sort By 下拉）。这在 4B 模型中极为罕见——B1 classifieds 全站几乎没有其他 episode 主动使用 Sort By
- 虽然最终失败（27 步，未找到实际的蓝色 iPhone），但 Sort By 的使用表明 SoM 截图促使模型识别了"排序"这一 UI affordance

### 搜索自纠正 + 价格过滤 + 多页浏览（task_19，成功）

- Intent："Find a sea painting that is listed for over $80"
- **三个高阶行为在单个 episode 中同时出现**：
  1. **搜索自纠正**：先搜 "sea" → 无结果 → 改搜 "sea painting" → 出结果。跨步自纠正在 B1(4B) 中极为罕见
  2. **价格过滤器**：在 Min 字段输入 `80`，与 task_17 相同的 action decomposition 策略
  3. **多页浏览**：翻了 3 页（第 1→2→3 页），逐页比较后选择正确 item
- 最终在 17 步内成功 finish

### 小结

这三个 task 集中体现了 SoM 视觉信息对 4B 模型的正向效应：

| 高阶行为 | B1 DOM 出现频率 | B1 SoM 出现 |
|---------|---------------|------------|
| 价格过滤器 Min/Max 分解使用 | 极低（task_19 DOM 尝试但 click 代替 type） | task_17 ✅, task_19 ✅ |
| Sort By 主动使用 | 未观测到 | task_18 |
| 跨步搜索自纠正 | 未观测到 | task_19 ✅ |

**与负向效应的关系**：本节案例与 §二 中的 visual scope anchoring、premature finish 形成对照——同一机制（视觉信息改变推理策略）在不同场景下产生正/负效果。当任务需要利用 UI 控件（过滤器、排序）时，SoM 截图让模型"看到"这些控件并主动使用；当任务需要超出当前视野推理时，截图锚定反而限制了模型。

---

## 三、方法论说明

- **Digest 文件**：`digest_som.jsonl`（与 DOM/vision 物理隔离）
- **归因管线**：
  1. 人工定性分析 — 逐 episode 审查，建立 SoM 表征缺陷分类体系
  2. GLM-5.1 batch digest + SoM 专属归因规则：S1(标注遮挡) / S2(颜色混淆) / S3(空间布局丢失) / S4(text_over_vision) / S5(ID幻觉)
  3. 交叉验证 — 人工分类与 GLM 归因结论一致
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

### 4.5 Evaluator 错误（已解决）

~~3 个 SoM episode 因 evaluator 基础设施问题失败。~~ 已全部解决（2026-04-12 确认）：
- **task 24, 135**：评测已正常运行，`score=1.0`。但这两个是 N/A 任务（`fuzzy_match="N/A"`），score=1.0 属于 `ua_match` false positive（见 §5）
- **task 160**：`program_html` 评测已正常运行，`score=0.0, error=None`，结果为失败

## 五、N/A 任务 ua_match 假阳性（SoM 模式）

Classifieds 10 个 N/A reference task 中，SoM **10/10 全部 score=1.0**（ua_match FP），与 DOM 模式一致。

| Task | 步数 | 结束方式 | FP 类型 |
|------|------|---------|---------|
| 24 | 15 | page_unchanged_streak | Type A（空 answer） |
| 135 | 24 | finish("Light Pink Depression Glass Candy Dish") | Type B（离开起始页） |
| 164 | 8 | finish(item 5636 URL) | Type B（搜索 "yellow car"） |
| 167 | 30 | max_steps | Type A |
| 189 | 6 | page_unchanged_streak | Type A |
| 191 | 5 | page_unchanged_streak | Type A |
| 194 | 6 | finish("Mickey Mouse item not found") | 特殊（正确理解不可行性） |
| 195 | 12 | page_unchanged_streak | Type A |
| 196 | 30 | max_steps | Type A |
| 220 | — | — | ua_match FP |

**根因**：ua_match 评测器 prompt 缺陷（将 agent answer 包装为 "reported unachievable reason"，"even if implicitly" 过于宽松）叠加 agent prompt 无 N/A 出口。详见 `B1_findings.md` §6.7。

**对 adjusted SR 的影响**：10 个 N/A task 全部是 visual task，已被 visual FP 过滤覆盖，不影响 adjusted 数字。

### Task 192 评测严格性边界案例

非 N/A task。Agent 正确看到两辆车（red + white），thought 中明确提到两种颜色，但 answer 只写 "red"（理解 "primary color" 为主色）。Reference 要求 must_include ["red", "white"]，score=0。视觉理解正确，answer 格式不完整。

---

## 六、Benchmark 答案歧义（人工审核，无法脚本批处理）

以下 task 模型选择了合理的答案，但因 `url_match` 只认唯一 reference URL 而被判失败。此类歧义需要**人工查看 intent 图片 + 模型选择 + reference 答案**才能判定，无法通过脚本自动检测。

| Task | Intent 关键信息 | 模型选择 | Reference 答案 | 歧义说明 |
|------|----------------|---------|---------------|---------|
| 166 | 找图片中包含与蓝莓同色乐器的 listing | Jackson Charvel 电吉他（蓝绿色, id=26772） | 蓝色木吉他 (id=40109) | 两把吉他都是蓝色系，都与蓝莓颜色匹配 |

**与现有 benchmark_noise 的区别**：
- 现有 `benchmark_noise`（§4）= 环境/基础设施错误（超时、API 失败），可脚本批量检测
- 本节 = **语义歧义**（多个合理答案但评测只认一个），只能人工逐个审核
- 等 SoM 全部跑完后统一审一遍，补充此表

---

*生成时间：2026-04-07，基于 digest_som.jsonl 中间数据（70/91 条）*
*更新时间：2026-04-07，追加第四节基础设施噪声分析*
*更新时间：2026-04-10，追加 gallery 聚合 premature finish（41/42/43）、visual scope anchoring（36/37）、action type 缺陷（19）、text_over_vision 案例（14/16）*
*更新时间：2026-04-10，追加视觉自信加速错误 finish（84/85）——SoM 截图信息反而强化 premature finish*
*更新时间：2026-04-10，追加 navigate-to 页面类型感知缺失（124/130/151/152/153）——到达目标详情页但不 finish*
*更新时间：2026-04-10，追加视觉误匹配（141/142）——语义类别匹配替代图片级匹配*
*更新时间：2026-04-10，追加 §5 Benchmark 答案歧义（166）——人工审核，无法脚本批处理*
*更新时间：2026-04-10，合并人工定性分析（原 B1_SOM_manual.md），更新方法论溯源*
*更新时间：2026-04-10，追加 §5 N/A ua_match FP 分析（SoM 9/9 全部 FP）+ task_192 评测边界案例*
*更新时间：2026-04-11，digest 全部完成（186/186），追加总体概况、更新 SoM 特有失败类型统计（text_over_vision 30→56，标注遮挡 3→4）、更新 N/A FP 为 10/10、移除"尚未完成"标注*
*更新时间：2026-04-21，追加 §二B SoM 正向效应（task_17/18/19）——价格过滤器、Sort By、搜索自纠正，Mirage Effect 正向案例*
*更新时间：2026-04-21，追加"视觉非确认→过早否定"（task_25/26/27）——视觉自信 premature finish 的镜像变体，task_25 DOM 成功 vs SoM 失败反转*
*更新时间：2026-04-21，追加"扁平化 marks 丢层级→容器/文本节点误点"（task_39/51）——SoM 特有 click-stuck，含 DOM vs SoM 格式对比分析*
*更新时间：2026-04-21，追加"label/textbox 混淆链"（task_52）——扁平 marks 表单 label 误选→scroll→marks 丢失→P4 三重叠加*
*更新时间：2026-04-21，追加"首页视觉干扰"（task_65/66/67）——与 task_84/85 统一为首页缩略图截断搜索流程*
*更新时间：2026-04-21，追加首页 select 后走偏量化（153 task 全量统计：SoM 61% vs DOM 42%，含三层归因）*
