# B1 Vision Baseline 分析报告（Classifieds，完整 234 tasks）

> **[DATA STALE]** 本报告数据基于 §33/§34/§36 修复前的运行结果。参考图片传递修复后 adjusted 指标可能变化，待重跑后更新。

> 数据来源：`phase1_vision_router_0`，classifieds 全部 234 tasks
> 分析方法：自动化 post analysis + 逐 episode 轨迹人工审阅（前 18 tasks 精读 + 关键 case 深读）
> 本报告**仅分析 Vision 模式**（纯截图，无 AXTree / SoM 标注）。
> 三模式共性缺陷与定量对比见 `B1_findings.md`。

---

## 总体概况

| 指标 | 数值 |
|------|------|
| 已完成 episode | 234 / 234 |
| Raw SR | 12.39%（29/234）|
| Adjusted SR（扣 visual FP + N/A FP） | 8.12%（19/234） |
| Visual FP | 0（Vision 模式本身有图，无 visual FP） |
| N/A FP | 10（与 DOM/SoM 相同，评测器缺陷） |
| 平均步数 | 8.0 步 |
| 平均成本 | $0.029 / episode |
| 平均 token | 15,225 / episode |
| p95 步延迟 | 53.2s |
| 平均能耗 | 3.84 mWh / episode |
| 早停触发分布 | page_unchanged_streak: 164, action_failed: 9, no_progress_streak: 2 |

### 与 DOM / SoM 全量对比

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| Raw SR | 8.97% (21) | 20.51% (48) | **12.39% (29)** |
| Adjusted SR | 0.85% (2) | 16.24% (38) | **8.12% (19)** |
| 平均步数 | 14.9 | 11.8 | **8.0** |
| 平均成本 | $0.074 | $0.077 | **$0.029** |
| 平均 token | 39,575 | 41,259 | **15,225** |
| p95 步延迟 | 45.5s | 82.6s | **53.2s** |
| 平均能耗 | 5.34 mWh | 6.49 mWh | **3.84 mWh** |

**Vision vs SoM**：McNemar p=0.0013（**显著**），Vision SR 显著低于 SoM；成本显著低于 SoM（Wilcoxon p<1e-26）。
**Vision vs DOM**：McNemar p=0.0002（**显著**），Vision SR 显著高于 DOM（adjusted）；成本显著低（Wilcoxon p<1e-25）。

---

## 成功案例分析

### Vision-Only 成功（7 tasks，adjusted 同）

这 7 个 task 只有 Vision 成功，DOM 和 SoM 均失败——全部是**纯视觉识别任务**：

| Task | 任务类型 | 评测方式 | 描述 |
|------|---------|---------|------|
| 14 | grid_position | string_match | 找第二行绘画的卖家邮箱（需 grid 空间布局理解） |
| 40 | single_navigation | url_match | 找不锈钢洗碗机（需视觉确认材质） |
| 97 | single_navigation | url_match | 找动物形状的物品（需视觉形状识别） |
| 124 | page_reading | string_match | 找图片背景是草地的 item |
| 152 | page_reading | string_match | 找图片中包含人手的 item |
| 187 | single_navigation | url_match | 找图片中有 Lightning McQueen 的 item |
| 188 | single_navigation | url_match | 找封面有婴儿的书 |

**共性**：全部需要从截图中识别视觉属性（颜色/形状/人物/背景），DOM 纯文本无法获取这些信息，SoM 虽有截图但 text-over-vision 效应导致文字推理覆盖了图片信息。

### SoM+Vision 共同成功（DOM 失败）— 9 tasks (raw), 12 tasks (adjusted)

| Task | 描述 | 共性 |
|------|------|------|
| 13 | 最贵 Boats 的颜色 | 视觉颜色识别 |
| 78, 79, 151 | 图片中引用城市/人物/粉色便签 | 图片内容匹配 |
| 87 | 找与参考图最相似的 item | 视觉相似度 |
| 106, 112 | 有动物/穿西装男人的图片 | 图片人物/动物识别 |
| 110 | Mario Kart 图中有几个游戏 | 图片内容计数 |
| 120 | 紫色自行车车架上写了什么 | 图片文字识别（OCR） |

### 三模式均成功 — 12 tasks (raw), 0 tasks (adjusted)

Raw 12 tasks 中大部分是 page_reading + single_navigation 类型。Adjusted 后因 DOM visual FP 过滤，三模式交集降为 0。

### Task 14 — Grid 空间布局理解（三模式对比关键 case）

- 2 步完成。起始页已在 Art gallery page 2（grid 布局）→ 点击坐标 [0.595, 600] 命中第二行 painting → 详情页截图中 email 清晰可见
- **DOM 和 SoM 均在此任务失败**，Vision 是唯一成功的模式

**三模式对比**：

| 模式 | "第二行"识别方式 | 点击目标 | 结果 |
|------|---------------|---------|------|
| DOM | AXTree 线性顺序数行 → item 60133 | eid=401 | 错误 email |
| SoM | 截图有空间信息，但 thought 与 DOM 完全一致 → item 60133 | eid=401 | 错误 email |
| Vision | 从截图视觉布局直接定位第二行区域 | coord=[0.595, 600] → item 32385 | 正确 email |

- **DOM 失败根因**：AXTree 是线性文本，无法表达 gallery 的 grid（行×列）布局。模型按元素出现顺序数"第二行"，将错误的 item 认定为目标。这类空间布局任务是 DOM 模式的**结构性盲区**（见 DOM digest 4.3.2）。
- **SoM 失败根因**：Text-over-vision。SoM 截图包含完整的 grid 空间信息，但模型的 thought 与 DOM 完全相同——文字推理先于图像处理锁定了错误答案（"第二行 = Ocean View"），截图的空间信息未被利用。这正是 SoM digest 中 text-over-vision 模式的典型表现。
- **Vision 成功根因**：纯视觉模式下没有 AXTree 文本干扰，模型直接从截图的视觉布局中感知 grid 结构，坐标落在了正确的第二行区域。

**与 Mirage Effect 的关系**：此 case 是 Mirage Effect 的正面案例——相同的 4B 模型，去掉文本信息（AXTree）后反而做出了更准确的空间判断。文本不仅没有帮助，反而误导了空间推理。SoM 作为中间态最能说明问题：图文同时存在时，文字推理主导了决策，图像的空间信息被忽略。

---

## 失败模式详解

### F1. 坐标定位失败（Coordinate Misclick）

Vision 模式使用归一化坐标 `[x, y] ∈ [0,1]` 点击，但 Qwen3-VL-4B 存在两个问题：

**(a) 坐标精度不足，无法命中目标元素**

| Task | Step | 目标 | 坐标 | 结果 |
|------|------|------|------|------|
| 2 | 7 | "Cocktail Ring" listing 链接 | [0.499, 0.283] | 未命中，page_changed=false |
| 3 | 7 | $1499 相机 listing | [0.438, 0.037] | y=0.037 在页面顶部 header 区域 |
| 9 | 4-6 | iPhone 13 mini listing | [0.49, 0.23] ×3 | 连续 3 次未命中同一链接 |
| 11 | 3-5 | 蓝色自行车 listing | [0.44, 0.28] ×3 | 连续 3 次未命中 |

Agent 识别了正确的目标，但生成的坐标偏移了目标元素的可点击区域。**失败后不会调整坐标**——连续 3-4 步重复几乎相同的坐标，形成死循环。

**(c) 系统性坐标偏移：task 100/101/102（Art+crafts → Books）**

三个独立 episode（同一任务模板，不同 seed）呈现完全一致的坐标偏移：

| Task | Thought | 目标分类 | 实际点击坐标 | 实际落点 |
|------|---------|---------|------------|---------|
| 100 | "navigate to Art+crafts" | Arts + crafts | [0.37, 0.15] | Books (sCategory=9) |
| 101 | "navigate to Art+crafts" | Arts + crafts | [0.37, 0.15] | Books (sCategory=9) |
| 102 | "navigate to Art+crafts" | Arts + crafts | [0.37, 0.15] | Books (sCategory=9) |

- **Thought 层正确**（知道要去 Art+crafts）
- **Action type 正确**（click）
- **Argument（坐标）固定偏移**，三次完全一致 → 偏差发生在 argument generation 层（非线性）
- **零自纠正**：back → click 相同坐标，循环 3 次不调整

DOM 模式通过 element_id 精确命中，无此问题——UI 元素本身没有错。这是 §19 "argument generation non-linearity"（知道要做什么，但生成的参数始终错）的坐标参数版本。三 seed 一致性排除了随机性，确认为系统性偏移。

**(b) 坐标格式混乱（混合归一化/像素值）**

| Task | Step | 坐标 | 格式问题 |
|------|------|------|---------|
| 4 | 0 | [0.683, 55] | x 归一化，y 像素 |
| 4 | 2 | [0.333, 28] | x 归一化，y 像素 |
| 4 | 3 | [0.783, 238] | x 归一化，y 像素 |
| 5 | 2 | [472, 180] | 全像素 |
| 5 | 5 | [480, 268] | 全像素 |

Prompt 要求 `[0.0-1.0]` 归一化坐标，但模型频繁输出像素值或混合格式。`vwa_wrapper.py` 有防御性归一化（>1.0 自动除以 viewport 尺寸），实际点击位置合理，但说明 **4B 模型的指令遵循不稳定**。

### F2. Scroll-Down 到底早停（不会翻页）

**代表 case：Task 6**（找 3 个 $1000-$2000 摩托车）

| Step | Action | page_changed | 备注 |
|------|--------|-------------|------|
| 0 | type "motorcycles\n" | true | 进入搜索结果 |
| 1 | scroll down | true | 看到 Harley $8990, $20495 |
| 2 | scroll down | true | |
| 3 | scroll down | false | 到底了 |
| 4 | scroll down | false | 重复相同 thought |
| 5 | scroll down | false | 重复相同 thought |
| 6 | scroll down | false | **page_unchanged_streak → 早停** |

Agent 到达页面底部后，**连续 3 步 scroll down 无变化**触发早停。AXTree 中分页链接可见但模型从不点击——这在 DOM/SoM ���式中同样存在（见 `B1_findings.md` §6.5），但 vision 模式更严重：没有 AXTree 暴露分页链接的文本/ID，agent 只能从截图中视觉识别分页控件，而 4B 模型做不到。

### F3. 过早结束（Premature Finish，Vision 特有高发）

Vision 模式过早 finish 的频率远高于 DOM/SoM，呈现两种变体：

**(a) 首页看不到目标即放弃**

| Task | 步数 | Agent 行为 |
|------|------|-----------|
| 10 | 1 | 首页显示 speaker/Xbox/camera/ring，看不到家具 → 直接 finish 空答案 |
| 12 | 1 | 首页无摩托车 → finish "No motorcycle is listed" |
| 16 | 1 | Appliances 分类页无咖啡杯 → finish 空答案 |

三个 task 均仅 1 步，verbalized confidence = 0.0，未尝试搜索或类别导航。

**(b) 找到目标但不进详情页即 finish**（见 F5 信息充分幻觉）

**根因**：Vision 模式下 agent 缺乏 AXTree 提供的结构化导航信息（搜索框、类别链接、item link 类型标注），加上 4B 模型的规划能力有限，导致：
1. 不知道可以搜索/浏览类别（变体 a）
2. 不知道列表页 item 可以点进去看到更多信息（变体 b）

**DOM/SoM 对比**：DOM/SoM 在这些 task 上**从不出现 1 步 finish**——AXTree 中的搜索框、类别链接、item link 元素给 agent 提供了明确的"还可以做什么"的行动线索。

**全量数据验证**：早停触发分布（page_unchanged_streak=164）远超 action_failed(9)，说明 Vision 的主要失败路径是"滚动到底/重复点击 → 无变化 → 早停"，而非 action 解析失败。

### F4. 不可交互元素误点（UI 理解失败）

**代表 case：Task 4**（修改白车价格到 $25000）

| Step | Action | Agent Thought | 实际情况 |
|------|--------|--------------|---------|
| 3 | click [0.783, 238] | "点击价格来编辑它" | 价格是只读文本，不可点击 |
| 4 | click [0.45, 0.38] | "激活编辑字段" | 详情页无内联编辑功能 |
| 5-7 | scroll ×3 | "找编辑字段" | 详情页无编辑入口，page_unchanged → 早停 |

Agent 试图在只读详情页上直接点击价格文本来编辑，不理解需要先点 "Edit item" 链接进入编辑表单。**Vision 模式下 agent 无法区分可交互与不可交互元素**——DOM 模式的 AXTree 会标注元素类型（link/button/text），但纯截图没有这些信息。

### F4a. click vs type 混淆（三模式共性，Prompt 缺陷）

Task 7 step 4 的价格筛选失败暴露了一个更深层的问题：agent 用 `click` 试图"输入"价格，而非使用 `type` action。

**机制分析**：`type` action 在 wrapper 中已封装了完整的 click→focus→清空→输入 流程（`vwa_wrapper.py:183-210`），agent 只需发一个 `type` action 带坐标+文本即可一步完成输入。但 Vision prompt 对此描述不清。三个模式的 agent 都会犯这个错误，但 SoM 模式偶尔能正确使用 `type`（有 AXTree 标注 textbox 类型辅助判断），Vision 模式从未成功。

### F5. 信息充分幻觉——列表页直接 finish 不进详情页（Vision 特有）

**代表 case：Task 15**（找红色 case 中吉他的卖家邮箱）

Agent 在列表页的截图中看到了正确的物品（Gibson SG Tribute），也看到了描述摘要中的 "Send contact number"，就直接判断"没有 email"并 finish——**始终停在列表页，从未点进详情页**。

**本质**：Vision 模式的截图给 agent 一种"我已经看到了所有信息"的错觉。DOM/SoM 的 AXTree link 结构隐式传达了"点进去还有更多内容"的信号，Vision 没有这个信号。

### F6. 幻觉与推理错误

| Task | 错误 |
|------|------|
| 3 | 将 $1499 判断为"在 $1000-$1200 区间内" |
| 5 | 反复声称"第一个物品是 Toyota 86"但实际点进了自行车详情页，循环 3 次不自知 |
| 6 | 将 $20,495 描述为"within the $2000 price range" |
| 11 | 每步对"第一个蓝色 bike"的判断不同（3 步 3 个不同物品） |

Task 5 最严重：agent 点击坐标 [480, 268] 反复进入自行车页面，但 thought 每次都说"第一个物品是 Toyota 86"——**视觉内容与语言推理完全脱节**。

---

## Vision 模式特有的结构性劣势

| 问题 | DOM/SoM 有而 Vision 无 | 影响 |
|------|----------------------|------|
| 元素 ID 点击 | AXTree 提供 `[id]` 精确点击 | Vision 只能用坐标，misclick 率高 |
| 元素类型标注 | `link` / `button` / `textbox` | Vision 无法区分可交互 vs 只读 |
| 结构化导航 | 搜索框、分类链接、分页控件有文本标注 | Vision 需视觉识别 UI 控件 |
| 文本信息 | AXTree 提供精确数值/标签 | Vision 依赖 OCR，易出现数值误读 |

---

## 与 DOM / SoM 的对比定位（全量）

| 维度 | DOM | SoM | Vision |
|------|-----|-----|--------|
| Raw SR | 8.97% | **20.51%** | 12.39% |
| Adjusted SR | 0.85% | **16.24%** | 8.12% |
| 平均步数 | 14.9 | 11.8 | **8.0** |
| 平均成本 | $0.074 | $0.077 | **$0.029** |
| 坐标 misclick | N/A（用 element_id） | 罕见（有 id fallback） | 高频 |
| 翻页 | 从未 | 偶发（task 19） | 偶发（task 58） |
| 视觉任务优势 | 无 | 有（图+标注） | 有（纯图，独占 7 tasks） |

**成本效率**：Vision 成本仅为 SoM 的 38%、DOM 的 40%。每成功 episode 的成本（adjusted）：Vision $0.029/0.0812 ≈ $0.36，SoM $0.077/0.1624 ≈ $0.47。Vision 在成本效率上有优势。

### 成功 task 成本对比（12 个三模式均成功 task）

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| 平均成本 | $0.055 | $0.085 | **$0.025** |
| 平均步数 | 10.3 | 12.3 | **6.8** |
| 平均 token | 29,028 | 45,300 | **12,850** |

SoM vs Vision 成本差异显著（Wilcoxon p=0.003）。在成功 task 上 Vision 最便宜且最快。

---

## Venn 图集合分析（Adjusted）

| 区域 | 数量 | 占比 |
|------|------|------|
| 三模式均失败 | 188 | 80.3% |
| 仅 SoM 成功 | 25 | 10.7% |
| SoM + Vision（非 DOM） | 12 | 5.1% |
| **仅 Vision 成功** | **7** | **3.0%** |
| DOM + SoM（非 Vision） | 1 | 0.4% |
| 仅 DOM 成功 | 1 | 0.4% |
| 三模式均成功 | 0 | 0% |
| DOM + Vision（非 SoM） | 0 | 0% |

**关键发现**：
- Adjusted 后三模式交集从 12（raw）降到 0（DOM visual FP 全部被过滤）
- Vision 独占 7 个成功，全部是纯视觉任务——这些 task 不可能通过 DOM/SoM 完成
- DOM + Vision 交集为 0，说明两者的成功 task 完全互补
- SoM + Vision = 12 个共同成功（非 DOM），视觉信息是共同因素

### Oracle Routing 分解（Adjusted）

| 模式 | 被 Oracle 选中次数 |
|------|------------------|
| SoM | ~25 |
| Vision | ~19 |
| DOM | 2 |

Oracle ceiling (adjusted) = 19.66%（46/234），headroom = 3.42%（在 SoM best 16.24% 之上）。12 个 SoM+Vision 共同成功 task 中，Vision 成本更低（$0.029 vs $0.077/ep）通常被 oracle 选中。Vision 贡献了 oracle 中约 41% 的选择——如果只有 SoM，ceiling 会低 ~3.4pp。

---

## 补充案例：Task 19 — `<select>` 缺陷与 auto-scroll 交互

Task 19（找 Arts + crafts 分类中 $80-100 海景画）展示了 auto-scroll 与 `<select>` 脚手���缺陷（`B1_findings.md` §6.3）的意外交互。

**轨迹**：
| Step | Action | 结果 |
|------|--------|------|
| 0 | click [0.598, 0.418] | 打开分类下拉菜单 |
| 1 | click [0.554, 0.138] "Arts + crafts" | option 点击失败 → **auto-scroll 触发 → 关闭 dropdown** |
| 2 | click [0.192, 0.904] | click 失败 → auto-scroll |
| 3 | click [0.362, 0.077] 分类图标 | **成功导航到 Arts + crafts** |
| 4-9 | scroll ×6 | 浏览 listings → 到底 → 早停（最终失败：未找到目标画作） |

**与 DOM/SoM 对比**：DOM/SoM 在 `<select>` 失败后也需要寻找替代路径，但 AXTree 中可见 "All categories" 链接文本，更容易发现替代方案。Vision 纯靠视觉识别分类图标，auto-scroll 的"逃逸效果"对 Vision 模式相对更重要。

### Task 22 — auto-scroll 主动移走已识别目标（有害干预）

Task 22（找第二行红色车的里程数）是 auto-scroll **直接有害**的典型 case。

- Agent 的视觉判断**完全正确**（thought: "The red car in the second row is visible"），仅坐标精度不足（像素值超出 viewport）
- Auto-scroll 把已识别的红色车从视野移走 → agent 下一步误点白色 Civic → 后续 5 步全浪费在错误目标上
- 模型缺乏跨步意图记忆（Task 48 验证：完全相同坐标连续 4 次），auto-scroll 的危害纯粹在效率层面（8 步 vs 4 步），不影响成功率

---

## Task 58 — 翻页 + City 过滤：Vision 独有的"聪明"行为

Task 58（在 Washington, D.C. 的 Furniture 分类中找最近的蓝色椅子）是 Vision 模式展现**最高策略水平**的 case，尽管最终失败。

**两个罕见行为**：

1. **翻页**（step 7）：Agent thought 写道"Since the page has pagination, I should check the next page"。成功翻页到 iPage=3

2. **City 过滤**（step 8）：正确找到 City 文本框并输入 "Washington, D.C."，URL 参数完全正确。DOM/SoM 在同一任务中 thought **分别提到 City 过滤 4 次和 5 次**但始终未能执行——典型的"认知-执行鸿沟"

**三模式对比**：
| | DOM | SoM | Vision |
|---|---|---|---|
| City 过滤 | thought 提到 4 次，未执行 | thought 提到 5 次，未执行 | **step 8 成功执行** |
| 翻页 | 30 步从未尝试 | 30 步从未尝试 | **step 7 成功翻页** |
| 步数 / Token | 30 / 78.8K | 30 / 105.4K | **16 / 28.6K** |

**与 Mirage Effect 的关系**：Task 58 是 Mirage Effect（相同语义信息+图片 → 质变推理路径）的强证据。Vision 的图片使 UI 控件的空间布局和可交互性更直观，触发了执行。

---

## Scroll 交替死循环（Vision-only）

Vision 模式 3 个 task（71/107/136）出现 scroll up/down 交替死循环，无法被现有 cycle detection 捕获，全部跑满 max_steps。DOM/SoM 为 0。

**根因：cycle detection 双重豁免**：
1. **Strict 签名**：`is_scroll and page_changed` 时跳过不记录 → 每次 scroll 都触发视口变化，strict 签名列表中无 scroll 条目
2. **Soft 签名**：`page_changed` 时清空列表 → 每步 scroll 都触发页面变化，soft 签名不断被清空

---

## 路由信号质量（Vision 模式）

### 信号区分力（AUROC，adjusted labels，n=115 有 confidence 的 episodes）

| 信号类型 | 最佳指标 | AUROC | 95% CI | 区分力 |
|---------|---------|-------|--------|-------|
| 行为信号 | action_diversity | **0.810** | [0.706, 0.902] | 强 |
| 行为信号 | url_revisit_max | 0.774 | [0.620, 0.892] | 强 |
| 行为信号 | url_revisit_count | 0.730 | [0.562, 0.880] | 中 |
| 行为信号 | max_repeat_streak | 0.708 | [0.564, 0.833] | 中 |
| Verbalized | ep_mean_verbalized | **0.710** | [0.551, 0.853] | 中 |
| Verbalized | ep_min_verbalized | 0.618 | [0.448, 0.786] | 弱 |
| Token-level | ep_mean_logprob | 0.479 | [0.305, 0.657] | 无 |
| Token-level | ep_mean_entropy | 0.494 | [0.342, 0.657] | 无 |

### 路由就绪度评估

| 维度 | 结果 |
|------|------|
| Token-level 区分力 | 无（全部 AUROC ≈ 0.5） |
| Entropy 区分力 | 无 |
| 行为信号区分力 | **有**（action_diversity AUROC=0.81） |
| Verbalized 区分力 | **有**（ep_mean_verbalized AUROC=0.71） |
| 信号校准 | 未校准（ECE=0.81） |
| 覆盖率 | 充分（100% episode 有 confidence） |
| **整体可用** | **是** |

### Verbalized Confidence 校准

| 指标 | Token-level | Verbalized |
|------|------------|------------|
| ECE | 0.808 | **0.563** |
| MCE | 0.817 | **0.690** |
| Brier | 0.747 | **0.429** |
| AUROC | 0.479 | **0.710** |

Vision 模式的 verbalized confidence 覆盖率 100%（远高于 DOM 0.85%、SoM 36.75%），且 AUROC=0.71 具有实用价值。这是 Vision 模式在路由方面的独特优势：**Vision 模式 prompt 结构天然要求输出 confidence**，是唯一能全覆盖 verbalized confidence 的模式。

---

## 关键发现

1. **Vision 定位在 DOM 与 SoM 之间**：Raw SR 12.39%（DOM 9.0% < Vision < SoM 20.5%），Adjusted SR 8.12%（DOM 0.85% < Vision < SoM 16.24%）。显著高于 DOM（McNemar p=0.0002），显著低于 SoM（p=0.0013）
2. **成本效率最优**：Vision 成本仅为其他模式的 38-40%，token 量约 1/3。同一 task 成功时 Vision 最快最便宜
3. **7 个 Vision-only 成功全是纯视觉任务**：图片内容识别（颜色/形状/人物/背景）是 Vision 独有且不可替代的能力
4. **DOM + Vision 交集为 0（adjusted）**：两者成功 task 完全互补，强化了路由的理论价值
5. **坐标精度是最大瓶颈**：misclick 后不自纠正（Task 48 验证），直接导致大量失败
6. **过早放弃是 Vision 特有模式**：缺乏结构化导航信息导致 agent 不知道如何探索
7. **认知-执行鸿沟弥合**（Task 58）：DOM/SoM 的 thought 多次提到操作意图但无法执行，Vision 成功执行——视觉信息触发执行
8. **路由信号可用**：行为信号（action_diversity AUROC=0.81）和 verbalized confidence（AUROC=0.71）具有区分力，且 Vision 是唯一全覆盖 verbalized 的模式
9. **Scroll 交替死循环**：Vision-only（3 task），因缺乏结构化文本定位导致空间迷失，现有 cycle detection 无法捕获

---

*生成时间：2026-04-12*
*更新时间：2026-04-12，全量 234 tasks 完成，数值更新为 adjusted（19/234 = 8.12%），McNemar Vision vs DOM 改为显著（p=0.0002）*
*数据来源：B1_3mode_classifieds_20260404_141103 完整三模式运行*
