# B1 Reddit Vision Digest

> 数据来源：`phase1_vision_router_0`，Reddit 全部 210 tasks
> 分析方法：自动化 post analysis + 逐 episode 轨迹审阅（关键 case 深读）
> 本报告**仅分析 Vision 模式**（纯截图，无 AXTree / SoM 标注）。
> 三模式共性缺陷与定量对比待 DOM/SoM 完成后见 `B1_findings.md`。

---

## 总体概况

| 指标 | 数值 |
|------|------|
| 已完成 episode | 210 / 210 |
| Raw SR | 4.76%（10/210）|
| 成功 episode 平均步数 | 4.2 步 |
| 全量平均步数 | 6.42 步（中位 4.0，范围 [1, 30]） |
| 平均成本 | $0.014 / episode |
| 平均 token | 13,997 / episode |
| p95 步延迟 | 27.3s |
| 平均能耗 | 1.39 mWh / episode |
| Evaluator error | 4（benchmark noise，`net::ERR_ABORTED`） |

### 早停触发分布

| 触发条件 | 触发次数 | 覆盖 episode 数 |
|---------|---------|---------------|
| action_failed | 408 | 148 / 210（70.5%） |
| no_progress_streak | 196 | 102（48.6%） |
| page_unchanged_streak | 193 | 99（47.1%） |

**注**：`action_failed` 是最主要的早停信号，远超 Classifieds Vision 的 `page_unchanged_streak` 主导模式。这与 Reddit 的 UI 结构有关——搜索栏和帖子链接更难通过坐标精准命中。

### Action 类型分布（1,349 步）

| Action | 次数 | 占比 | 成功率 |
|--------|------|------|--------|
| click | 631 | 46.8% | 54.8% |
| scroll | 381 | 28.2% | 66.4% |
| back | 116 | 8.6% | 85.3% |
| type | 101 | 7.5% | 50.5% |
| tab_focus | 87 | 6.4% | **28.7%** |
| finish | 33 | 2.4% | 100% |

**click 成功率仅 54.8%**——接近一半的 click 无法命中目标，是 Vision 模式在 Reddit 上的核心瓶颈。

---

## 成功案例分析（10 tasks）

| task_id | 评测方式 | 有参考图 | 步数 | 成本 | Intent 摘要 |
|---------|---------|---------|------|------|------------|
| 7 | string_match | Yes | 4 | $0.009 | 找帖子中 OP 发的 recipe comment URL |
| 26 | string_match | Yes | 4 | $0.009 | 找帖子，看 top comment |
| 31 | string_match | Yes | 5 | $0.012 | 找用户，导航到其另一个 post 的 comments |
| 39 | string_match | Yes | 15 | $0.037 | 找照片拍摄城市的 subreddit |
| 69 | program_html | No | 3 | $0.005 | 评论解释图片内容 |
| 72 | program_html | No | 3 | $0.005 | 评论解释图片内容 |
| 120 | string_match | No | 1 | $0.002 | 哪个国家铁路最差 |
| 160 | program_html | No | 1 | $0.002 | 订阅以 'i' 开头的特定 subreddit |
| 182 | string_match | Yes | 3 | $0.005 | 在学校所在城市发帖问最佳印度食物 |
| 201 | url_match | Yes | 3 | $0.007 | 导航到与图片最相关的帖子 |

### SR 按维度分解

| 维度 | SR |
|------|-----|
| 有参考图的 task | 6/84 = **7.14%** |
| 无参考图的 task | 4/126 = **3.17%** |
| string_match | 6/70 = 8.57% |
| program_html | 3/85 = 3.53% |
| url_match | 1/51 = 1.96% |
| page_image_query | **0/28 = 0.00%** |

**page_image_query 全军覆没**：这类 task 需要先导航到目标帖子的图片页再分析图片内容。Vision 模式的导航能力不足以完成多步链式导航。

---

## 失败模式详解

### F1. Click-not-type 系统性缺陷（73 tasks, 41.2%——**最大失败类别**）

全量扫描显示，177 个 task 中 **73 个（41.2%）** 存在"反复 click 同一坐标但从不 type"的模式（≥3 次连续同坐标 click 且 click 占全部 action ≥60%）。其中 **70/73 整个 episode 没有产生过一次 type 动作**。

此前 F1 仅统计了搜索栏场景（7 tasks），实际该模式远不限于搜索栏——评论框、输入表单、论坛导航链接等场景均大量出现，是 Vision 模式最主要的系统性失败原因。

**典型场景分布**：
- 搜索栏反复 click（如 task 6/13/15/78/79/99）
- 评论框/输入区 click 不 type（如 task 88/92/93）
- 帖子链接/论坛链接反复 click（如 task 1/8/33/100/101）

**几乎全部 3 步内被 cycle detection 终止**，平均步数 3.7，远低于其他失败模式。

**代表 case：Task 6**（找 cooked pork 帖子）

| Step | Action | 坐标 | 成功? |
|------|--------|------|-------|
| 0 | click | [0.49, 0.03] | title_changed only |
| 1 | click | [0.49, 0.03] | **失败** |
| 2 | click | [0.49, 0.03] | **失败** |

**对比 Task 4**：同样起初 click 搜索栏失败，但 **step 2 成功切换到 type("wheat field\n")**，完成搜索。然而 step 9-11 回到主页后又陷入纯 click 死循环，说明自纠正不稳定。

**根因**：Vision 模式下 agent 看到搜索框/输入区但缺乏 AXTree 的 `<input>`/`textbox`/`searchbox` 语义提示，无法将"点击输入框→输入文字"拆分成 click + type 两步操作序列。DOM 模式天然暴露元素类型，agent 会组合 click→type 两步操作。**属于 4B 模型在 vision-only 下的固有能力缺陷，不修。**

**注**：此 73 tasks 与下文 F3（click 死循环 48 tasks）有大量重叠。F3 从"同坐标循环"角度归类，F1 从"缺少 type"角度归类。两者合并后去重约 85 tasks，覆盖 48% 的失败 episode。

### F2. 图片链接陷阱（18 tasks, 9.0%，99 步浪费）

Postmill（Reddit 克隆）的 UI 设计导致这一陷阱：**帖子标题和缩略图都链接到原始图片，而非帖子讨论页**。只有标题下方的小字 "N comments" 链接才指向帖子页面。

**Postmill 帖子列表中每个条目的链接结构**（以 DOM 第 27/33 行为证）：

| UI 元素 | 视觉位置 | 链接指向 |
|---------|---------|---------|
| 缩略图 | 左侧方形图 | `/submission_images/...jpg`（原图） |
| **帖子标题** | **标题行（最显眼）** | **`/submission_images/...jpg`（原图）** |
| "N comments" | 标题下方灰色小字 | `/f/food/18838/...`（帖子讨论页） |

因此 agent 点击"标题"（视觉上最自然的入口）实际上跳转到一张充满整个页面的大图，该页面无任何导航元素，只能 `back` 返回。

**代表 case：Task 10**（找 Pumpkin Loaf 帖子中 OP 的 recipe comment）

```
step 0: click [0.42, 0.28] "Pumpkin Loaf" 标题
        → /submission_images/66a1b3d1...jpg （大图页面，非帖子页）
        Agent thought: "clicking on the post to access its comment section"
        实际：标题 <a> 的 href 就是原图 URL
step 1: scroll（无效，大图页面没有可滚动内容）
step 2: back → 回到 /f/food
```

Agent 准确命中了标题文字区域——**坐标没有偏移**。问题在于 Postmill 的帖子标题本身就链接到图片而非讨论页，这与主流 Reddit 的 UX 惯例相悖（Reddit 标题链接到帖子页面）。

**Task 4**（wheat field）同样中招：
```
step 2: type "wheat field\n" → 进入搜索结果 ✓
step 3: click [0.28, 0.41] → /submission_images/e5b89... ✗（点了标题，不是缩略图）
step 4: back → 回到搜索结果
step 5: click [0.28, 0.41] → /submission_images/... ✗（同一坐标，重复）
step 6: back → 回到搜索结果
```

**Task 20 最严重**：19 步中几乎每步都导航到 `/submission_images/` URL，步数全部浪费。

**受影响 task**：1, 4, 9, 10, 14, 20, 52, 53, 61, 77, 80, 85, 155, 173, 174, 193, 202, 206

**根因**：
1. **视觉上无法区分**：截图中帖子标题看起来就是"进入帖子"的主要入口，agent 的点击意图完全合理
2. **Postmill 违反常见 UX 惯例**：主流 Reddit 的标题链接到讨论页，Postmill 链接到原图——即使是人类用户也可能中招
3. **正确入口不直观**："45 comments" 小字链接才是通往讨论页的唯一路径，但它在视觉层级上远不如标题显眼
4. **DOM 模式可部分规避**：AXTree 暴露了链接 URL 结构（`/submission_images/` vs `/f/food/18838/`），理论上 agent 可通过 URL 辨别，但 B1(4B) 未必利用此信息

### F2a. 分类页面坐标偏移——密集文本列表中的系统性 misclick（11 tasks）

37 个 task 从 `/forums/all`（字母序分类页）出发，其中 11 个 task 在 step 0 尝试点击特定分类链接，**仅 2/11（18%）命中正确分类**。

**分类页面布局**：Postmill 的 `/forums/all` 是一个多列字母序列表（A-Z 分栏，每列约 6-8 个链接），行间距约 20px。这种密集布局对坐标精度要求极高。

| Task | 目标分类 | 实际落点 | 坐标 | 错误类型 |
|------|---------|---------|------|---------|
| **21** | **OldSchoolCool** | **OldSchoolCool ✓** | [0.65, 0.38] | — |
| **36** | **nyc** | **nyc ✓** | [0.607, 0.278] | — |
| 19 | GetMotivated | **gaming** | [0.28, 0.68] | Y 上偏 |
| 35 | washingtondc | **wallstreetbets** | [0.78, 0.84] | Y 上偏 |
| 40 | pittsburgh | **photoshopbattles** | [0.63, 0.64] | Y 上偏 |
| 18 | MechanicalKeyboards | 未命中（留原页） | [0.48, 0.38] | 近似 miss |
| 38 | nyc | 未命中（留原页） | [0.61, 0.28] | 近似 miss |
| 34 | boston | 未命中（留原页） | [97, 647] | 像素坐标（已归一化） |
| 37 | Art | 未命中（留原页） | [95, 395] | 像素坐标（已归一化） |
| 41 | Newark | 未命中（留原页） | [0.433, 980] | 混合坐标（已归一化） |
| 42 | MechanicalKeyboards | 未命中（留原页） | [476, 725] | 像素坐标（已归一化） |

**三种错误类型**：

**(A) Y 轴系统性上偏（3 tasks: 19/35/40）**：Agent 使用有效归一化坐标，**X 轴（列定位）完全正确**，但 Y 值始终偏高，导致点中目标上方 1-2 行的链接。精确偏移量如下：

| Task | 目标 | 目标 Y (px) | 点击 Y (px) | 偏移 | 跳过行数 |
|------|------|-----------|-----------|------|---------|
| 19 | GetMotivated | 506 | 489 | +17px | 1 行 |
| 40 | pittsburgh | 504 | 460 | +44px | 2 行 |
| 35 | washingtondc | 665 | 604 | +61px | 2 行+标题 |

```
Task 19: GetMotivated → gaming          （G 组内，上偏 1 行）
Task 40: pittsburgh → photoshopbattles   （P 组内，上偏 2 行）
Task 35: washingtondc → wallstreetbets   （W 组内，上偏 2 行 + "W" 标题）
```

这不是固定的"一行偏移"，而是 **4B 模型的 Y 坐标分辨率（~40-60px）大于行间距（~20px）**。偏移量在 17-61px 之间变化，但方向始终向上。在行间距更大的 UI 中（如帖子列表，间距 ~80-100px）同等误差不会造成跨行。

**(B) 像素坐标格式（4 tasks: 34/37/41/42）**：模型输出原始像素值（如 `[97, 647]`）或混合格式（如 `[0.433, 980]`）而非归一化坐标。`vwa_wrapper.py:231-234` 的防御逻辑会自动归一化（>1.0 则除以 viewport 尺寸），所以不会崩溃，但归一化后的坐标未必落在目标链接上（如 `[97/1280, 647/720]=[0.076, 0.899]` 落在页面左下角空白处）。这说明 4B 模型的**坐标格式遵循不稳定**，但不影响系统健壮性。

**(C) 近似 miss（2 tasks: 18/38）**：坐标合理但差几个像素。Task 38 点 nyc 的坐标 `[0.610, 0.280]` 与 task 36 成功的 `[0.607, 0.278]` 仅差 0.003——约 2-4 像素。

**零恢复**：9 个失败 task 中无一在后续步骤到达正确分类。典型模式是在错误 subreddit 上重复 click 2-3 次直到早停。

**与 Classifieds 的对比**：Classifieds 的分类导航通过下拉菜单（`<select>`）实现，DOM 模式可精确选择；Vision 模式在 Classifieds 上也有 `<select>` 失败（见 Classifieds Vision digest F4a），但 Classifieds 有替代路径（分类图标）。Reddit 的 `/forums/all` 密集文本列表没有替代入口，坐标精度是唯一路径。**核心矛盾**：4B 模型的 Y 坐标误差（~40-60px）> 列表行间距（~20px），在密集文本布局中无法精准定位。

**DOM/SoM 模式可完全规避此问题**：AXTree 中每个分类链接有独立 element_id（如 `[460] link 'GetMotivated'`），通过 `click [460]` 精确命中，不依赖坐标。

---

### F3. Click 死循环（48 tasks, 24.0%——**最主要失败模式**）

Agent 在相同或相近坐标连续 click 3+ 次，`page_changed=false`。

**两种变体**：

**(a) 相同坐标循环（37 tasks）**：坐标几乎一致，每步 thought 重复

| Task | 循环坐标 | 循环次数 | Thought |
|------|---------|---------|---------|
| 4 | [0.49, 0.05] | 4 次 | "click search bar" |
| 19 | [0.49, 0.12] | 3 次 | "navigate to /f/GetMotivated" |
| 100 | [0.33, 0.28] | 3 次 | "click on post" |

**(b) 同区域漂移循环（11 tasks）**：坐标略有变化但目标相同，仍无法命中

**根因**：无 DOM 反馈，agent 无法判断 click 是否命中了目标元素。screenshot 不变时只能重试相同坐标。

### F4. 过早 finish / 直接作答（25 tasks, 12.5%）

**17/25 在 step 0 就 finish**，大多是 `page_image_query` 类型——agent 试图从起始页截图直接回答视觉问题。

| Task | 步数 | Agent 回答 | 正确? |
|------|------|-----------|-------|
| 89 | 0 | "It has 3 Jupiter!!" | 错 |
| 90 | 0 | "It has 10 teeth :)" | 错 |
| 102 | 0 | "It has 2 $%@" | 错 |
| 95 | 0 | "The task cannot be completed" | 放弃 |

**两种子类型**：
- **错误视觉计数**（多数）：从首页截图数对象，数错
- **跳过导航直接答题**：task 要求先导航到特定帖子再分析图片，agent 在首页就答了

### F5. 参考图片感知存在但 grounding 失败

**Token 分析证实参考图片已加载**：

| Task | 参考图片原尺寸 | resize 后 | input_image tokens |
|------|-------------|-----------|-------------------|
| 2 | 1440×960 | ~1024×682 | 1248 |
| 4 | 1698×1126 | ~1024×679 | 1248 |
| 10 | 3024×4032 | ~768×1024 | 1344 |

三个 task 的 screenshot 均为 1280×720（同一 viewport），token 差异只能来自参考图片大小不同，证明参考图片确实进入了模型。

**Agent 的感知 vs 行动差距**：

78 个有参考图的失败 task 中，66 个（85%）的 thought 包含超出 intent 文本的视觉描述（如 "sushi platter"、"pumpkin loaf"、"colorful keyboard"），证明 agent **看到了参考图片内容**。

| Task | Intent | Agent 视觉描述 | 行为 |
|------|--------|--------------|------|
| 4 | "post with this image" | "wheat image" ✓ | 搜索 "wheat field" → 成功找到搜索结果 |
| 10 | "post with this image" | 无描述，直接选了第一个帖子 ✗ | 选了 "Pumpkin Loaf"（错误帖子） |
| 6 | "post with this image" | "cooked pork" ✓ | 识别了内容但搜索栏 click 失败 |

**根因**：不是感知问题而是 **grounding-to-action gap**——agent 知道图片是什么，但无法在页面上精准定位匹配的帖子并点击正确的链接。

### F6. 自纠正能力极弱

| 指标 | 数值 |
|------|------|
| 失败 action 总数 | 397 |
| 切换了 action_type 的次数 | 106（**26.7%**） |
| 重复相同 action_type | 291（**73.3%**） |

**Top 纠正路径**：
- click → scroll：25 次
- click → type：22 次（搜索场景）
- scroll → click：17 次
- scroll → back：10 次

**73.3% 的时间，agent 在失败后重复相同策略**。对比 Task 4 的表现：step 2 成功从 click 切换到 type（少数自纠正案例），但 step 9-11 又退回纯 click 循环——说明自纠正即使发生也不稳定。

---

## 失败模式分布总览

| 模式 | 数量 | 占失败 episode % |
|------|------|----------------|
| F1 Click-not-type 系统性缺陷 | 73 | **43.7%** |
| F3 Click 死循环（与 F1 大量重叠） | 48 | 28.7% |
| F4 过早 finish | 25 | 15.0% |
| F2 Image link trap | 18 | 10.8% |
| F1+F3 去重合并 | ~85 | ~50.9% |
| 至少命中一种模式 | ~120 | ~71.9% |
| 其他（多样化失败） | ~47 | ~28.1% |

其余 89 个未归类 episode 包含：scroll 探索但未找到目标（18）、短 episode 无明显模式（45）、坐标 miss 但不形成循环（26）。

### 失败 episode 步数分布

| 范围 | 数量 | 占比 |
|------|------|------|
| 1-3 步（快速失败） | 73 | 36.5% |
| 4-10 步（中程） | 95 | 47.5% |
| 11-29 步 | 28 | 14.0% |
| 30 步（跑满） | 4 | 2.0% |

**36.5% 的失败在 3 步内结束**——agent 要么快速 finish 交了错误答案，要么连续 click 失败触发早停。

---

## Evaluator Error（Benchmark Noise）

4 个 task 存在 `net::ERR_ABORTED` evaluator 错误，与 DOM digest 中报告的 Reddit Docker 页面加载缺陷一致：

| Task | 评测 URL | 评测类型 |
|------|---------|---------|
| 143 | `f/movies/128615` | url_match |
| 146 | `f/wallstreetbets/50335` | program_html |
| 147 | `f/food/18831` | program_html |
| 148 | `f/food/60745` | program_html |

全部标记 `benchmark_noise=True`，评测器侧失败，非 agent 问题。

---

## Reddit vs Classifieds Vision 对比

| 指标 | Classifieds Vision | Reddit Vision |
|------|-------------------|---------------|
| Raw SR | 12.39%（29/234） | **4.76%（10/210）** |
| 平均步数 | 8.0 | 6.42 |
| 平均成本 | $0.029 | **$0.014** |
| click 成功率 | ~65%（估） | **54.8%** |
| 主要早停信号 | page_unchanged_streak | **action_failed** |
| 过早 finish | 3 tasks（1.3%） | **25 tasks（12.5%）** |
| Image link trap | 无 | **18 tasks（9.0%）** |

**Reddit 更难的原因**：

1. **Reddit UI 对 Vision 更不友好**：帖子列表中缩略图和标题链接混杂，容易误点图片进入 `/submission_images/` 死胡同。Classifieds 列表页结构更简单。
2. **搜索交互更复杂**：Reddit 搜索栏需要 click→type 两步，且搜索栏坐标不易定位。Classifieds 首页直接有分类导航，可绕过搜索。
3. **多步导航需求更高**：Reddit 的成功路径通常是 搜索→找帖→进评论→找目标评论，链路更长。Classifieds 多为单步导航（搜索→点 item）。
4. **page_image_query 类型是 Reddit 特有难题**：需要 OCR 或视觉计数，4B 模型能力不足。

---

## 关键发现

1. **Reddit Vision SR 极低（4.76%）**：显著低于 Classifieds Vision（12.39%），Reddit 的 UI 结构对纯视觉模式更不友好
2. **Click 成功率仅 54.8%**：近半数 click 未命中目标，是 Reddit 上 Vision 失败的首要原因
3. **参考图片已加载且被感知，但 grounding 失败**：85% 的参考图失败 task 中 agent 识别了图片内容，但无法将视觉识别转化为正确的页面操作
4. **Image link trap 是 Reddit 特有问题**（18 tasks, 99 步浪费）：Postmill 的缩略图链接结构让 Vision agent 频繁误入原图页面
5. **Click-not-type 是最大失败类别（73 tasks, 41.2%）**：远超此前统计的 7 tasks，涵盖搜索栏、评论框、导航链接等全部需要 type 的场景。70/73 整个 episode 无一次 type，属 4B+Vision 固有缺陷
6. **过早 finish 占 12.5%**：远高于 Classifieds（1.3%），agent 倾向于从首页截图直接答题而非导航
7. **自纠正率仅 26.7%**：73.3% 的失败后重复相同策略，B1(4B) 缺乏适应性
8. **page_image_query 全军覆没**（0/28）：视觉计数/识别类任务需要先导航后分析，Vision 模式两步都做不好
9. **action_failed 是主要早停信号**（覆盖 70.5% episode），与 Classifieds 的 `page_unchanged_streak` 主导模式不同——Reddit 的 UI 元素更难通过坐标命中

---

*生成时间：2026-04-19*
*数据来源：B1_3mode_reddit_20260413 phase1_vision_router_0，210 tasks*
*DOM/SoM 尚未完成，跨模式 Venn 分析待补充*
