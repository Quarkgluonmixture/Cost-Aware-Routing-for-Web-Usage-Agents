# B0 Classifieds — DOM 模式分析

> B0 = Qwen3-VL-235B-A22B（proxy API），DOM 模式，classifieds 站点
> 对应 B1 分析见 `B1_DOM_digest.md`；B0 的核心分析价值是 235B vs 4B 的能力差异

---

## 已知共性缺陷（与 B1 DOM 相同，不重复分析）

以下问题在 B1 DOM 已详细记录，B0 重新确认，结论引用 B1：

- **未 scroll up 直接 type**：agent 根据 AXTree 操作，无空间感知
- **type 时误触全选**：`select_all` 副作用，界面变蓝
- **未使用右侧价格筛选工具**：filter/sort 控件识别但不使用

---

## B0 vs B1 能力差异：三项 B0 特有正向行为

以下三项行为在 B1（4B）中**从未或极少**出现，B0（235B）稳定使用，是模型规模带来的实质性能力提升。

### 1. 翻页导航（paginate）

B0 DOM 中 **33+ 个 task** 出现了 `iPage=2/3/4` 的 URL 翻页，agent 主动点击"下一页"按钮导航到第 2、3、4 甚至更后面的页面：

- Task 201 step 6/11/15：搜索结果翻到第 2/3/4 页找 snare drum
- Task 80 step 1：直接从 `iPage=4` 开始浏览
- Task 149/150：从已在高页码的 gallery 视图开始任务

B1 DOM 在 classifieds 几乎从不翻页（MEMORY §31：DOM 0/646 步骤含翻页行为）。**B0 的翻页能力使其能遍历更多候选结果，是 raw SR 高于 B1 的重要原因之一。**

### 2. 价格上下限筛选（sPriceMin / sPriceMax）

B0 DOM 中 **21+ 个 task** 正确使用了价格区间筛选，**独立填写 sPriceMin 和 sPriceMax 两个字段**（不同元素 ID），并通过 Enter 提交：

```
step 0: type eid=140  text='oval table\n'   → 搜索关键词
step 1: type eid=1382 text='420\n'          → 价格下限字段
step 2: type eid=2595 text='430\n'          → 价格上限字段
step 3: click → 目标 listing
```

Task 217（真实成功）、Task 216、Task 2、Task 6、Task 7 等均有此行为。B1 DOM 极少使用价格筛选控件（见 B1_DOM_digest §4：filter/sort 控件识别但不使用）。**B0 的价格筛选能力使区间类任务的成功率大幅提升。**

### 3. 多 Tab 切换（tab_focus）

B0 prompt 明确包含 `tab_focus` 指令（`proxy_api_agent.py:76/116/151/184`），agent 能识别多 tab 任务并调用 `tab_focus {page_number}` 切换浏览器 tab。classifieds B0 DOM 中有 4 个 task 触发了 `tab_focus`（148/150/163/229）。

**Task 229 实例**（三枚戒指跨 tab 比较设计，`program_html` eval）：step 0 正确推断 Emerald 18K + Emerald 14K 最相似（items 47824+42770），但 step 2 推理翻转改口 Ruby+Emerald 最像，全程仅 tab_focus 未发评论。

**Cycle detection 误杀 bug（已修复）**：`_action_signature` 未包含 `page_number`，tab_focus 三步产生同一签名，触发 cycle-1×3reps 早停。修复（`runner.py`）：签名加入 `page_number`，使 1→0→1 不触发 cycle，1→1→1 仍正确触发。Task 229（pn=1→0→1）和 150（pn=1→1→1，真 cycle）均被误杀，229 修复后需重跑。

### 4. 表单字段精准聚焦（form field focus）

B0 通过 AXTree element ID 精准定位并逐一聚焦表单字段（搜索框 → 价格下限框 → 价格上限框），每个字段独立 type + Enter，不混用字段。**未观测到显式 Tab 键使用**（全局搜索 `key=Tab` 和 `text 含 \t` 均为空），聚焦机制是 click-by-eid。

B1 有时在 type 时选错字段或丢失焦点（见 B0_DOM_digest §3/§5：越界 type 导致全选蓝）。B0 的字段聚焦更稳定，是价格筛选能正确执行的前提。

---

## B0 特有行为：scroll dy 约定随机混用

### 现象
235B 模型对 scroll `delta=[dx, dy]` 的符号约定**不稳定**，在同一个 episode 内混用两种约定：

| 约定 | dy 符号 | 来源 | B0 中的表现 |
|------|---------|------|------------|
| Prompt 约定 | dy>0 = 向下 | prompt 明确说明 | 部分 step 遵循 |
| 训练先验 | dy<0 = 向下 | 数学坐标/部分框架 | 部分 step 回退 |

**实例（task_0 新跑，修复后观测到的模型原始输出）**：
- step 1：模型输出 dy=+0.5，thought 不明确 → 遵循 prompt（down）
- step 2：模型输出 dy=-0.5，thought 不明确 → 用了先验（wrong）
- step 3：模型输出 dy=+0.5，thought "I need to scroll **further down** the page" → 遵循 prompt（down）✓
- step 4-5：模型输出 dy=+0.5，thought 同上 → 遵循 prompt（down）✓

早期观测（task_4/18/28/29）均为 dy=-0.5，当时误判为"始终用反约定"，并尝试对 dy 取反修复 → 反而更混乱。

### 为什么 235B 会这样？

**根因：大模型训练数据中存在多种 scroll 约定，temperature=0.1 的轻度随机允许采样切换**

1. **多约定竞争**：235B 的训练语料涵盖大量代码库，不同框架 scroll 约定不同：
   - CSS/数学坐标：y 轴向下为正，scroll down → dy>0 ✓
   - 部分游戏引擎/图形库：屏幕坐标翻转，dy<0 = 向下
   - 某些 UI 框架：scroll event 的 deltaY 以内容移动方向定义，down = 负
   235B 同时"记住"了多种约定，形成多个竞争性的吸引子（attractors）

2. **小模型更易遵循指令**：B1（4B）始终输出 dy=+500（大正值），完全遵循 prompt。训练数据少意味着竞争先验少，对 prompt 的约束更敏感。这是"指令遵循 vs 训练先验"的规模悖论（scaling paradox）：**更大的模型反而更难覆盖特定格式约定**。

3. **思维链影响采样**：thought 生成时触发的语义框架（"scroll down"）理论上应强化 dy>0，但 temperature 采样噪音有时让模型切换到另一个约定吸引子。这解释了为什么同一 thought 内容（"scroll down"）有时输出 +0.5 有时 -0.5。

4. **magnitude 差异也印证了这一点**：B0 用 dy=±0.5（归一化，来自某类 web 自动化框架训练数据），B1 用 dy=+500（像素级，来自 JavaScript API 训练数据）——两模型连数值范围都习得自不同的训练分布。

### 影响评估

每次 scroll 方向错误消耗 1 步，并可能触发 page_unchanged 计数（若方向相反导致已在顶部/底部无变化）。由于约定混用是随机的，影响无法系统性量化，只能作为噪声记录。

### 处置

- **不修复**：取反修复已验证失败（使混乱更严重）；intent-based 修复（解析 thought 文字）过于脆弱
- **分析时**：将 scroll 方向错误导致的 page_unchanged 视为模型随机噪声，非系统性脚手架缺陷
- **论文披露**：B0 scroll 行为存在约定不稳定性，为 235B 模型的已知局限，不影响 B0 vs B1 主要结论（SR 对比）

---

## B0 特有行为：`<select>` 下拉菜单 capability-environment gap

### 现象
B0（235B）识别出 classifieds 首页的 category `<select>` 元素是正确入口，反复 click 同一 eid（task_2: eid=147，task_3: eid=148），每个 task 只走 3 步即被 cycle detection 截断，未进入任何备选路径。

### 与 B1 对比

| | B1 (4B) | B0 (235B) |
|--|---------|-----------|
| 对 select 的认知 | 不确定，随机探索 | 正确识别为目标元素 |
| 失败后行为 | scroll 到侧边栏分类链接（绕路成功） | cycle 截断，不进入备选 |
| 结果 | 有时成功（偶然发现 workaround） | 快速失败 |

### 根因：capability-environment gap
Playwright 无法正常打开原生 `<select>` 下拉框（点击无效）。B0 能力更强反而被卡死：**认知越准确，越执着于"正确"路径，cycle detection 触发越快，探索越少**。B1 因不确定性而随机游走，偶然找到侧边栏链接。

这类任务（需要分类导航 + url_match 评测）在 classifieds 共有多个，**B0 DOM 的系统性劣势**可能部分来自此 gap，而非模型能力不足。

### 影响
- 受影响 task 类型：需要 category filter 的 url_match 任务
- 实际 task 数量待统计（task_2、task_3 已确认，其他需分析）
- 论文披露：B0 vs B1 SR 差距在分类导航类任务上可能倒置（B1 > B0），为环境限制导致，非能力差距

---

## B0 DOM 典型失败模式（逐案例分析，2026-04-14）

### 1. 视觉任务在 DOM 模式下必然失败

**Task 6**（3 most recent motorcycles $1000-2000, not orange）：
- Agent 返回 1 个链接（id=22796），eval 要求 3 个（id=4208/65357/11549）
- 根因：颜色过滤不可能（无图），且搜索结果截断只返回第一条

**Task 7**（2 most recent white cell phones $300-600）：
- Agent 返回 [38125（正确）, 26581（错误）]，eval 要求 [40404, 38125]
- 根因：DOM 无法验证颜色，第二条结果随机猜错

**Task 24**（How many miles does the black truck on this page have）：
- start_url 是 gallery 视图（`sShowAs=gallery&iPage=5`），颜色只在缩略图里
- Step 1 agent 用 `eid=1`（AXTree 根节点，bbox=[0,0,10,10]）执行 TYPE → 点击 (5,5) → 焦点落 body → Meta+A 全选蓝
- 同时为 N/A 任务（answer="Miles not listed"），双重失败

以上均为 **visual task + DOM 模式** 的必然失败，已被 `_mark_visual_false_positives()` 正确归类。

---

### 2. 跨步数值记忆失败

**Task 31**（Find latest white Google Pixel, comment offering $10 less）：
- Step 9 thought 明确读到价格 "$260.00"；step 13 输入报价 "$140"（应为 $250）
- 中间经过 3 步 scroll + click（step 10-12），模型在长轨迹中丢失了关键数值
- 同时选错了商品（去了 id=70602，eval 检查 id=62633）

**Task 32**（Find cheapest item $300-500, comment offering $10 less）：
- Agent 搜索后找到正确商品（价格 $395），在 thought 中正确识别了价格
- 但 finish 时输入报价 "$490"（应为 $385）——agent 似乎将上限 $500 误记为商品价格，然后 $500-$10=$490
- **P10 模式命中**：跨步数值记忆失败，价格被搜索区间上限替代

**Task 33**（Find latest white fridge, comment offering $10 less）：
- 同类任务，选错了商品（id=70602 vs eval 要求 id=62633，eval 期望评论含 $115）
- Step 6 额外触发全选变蓝（见下节）

---

### 3. 元素中心越界→点击/输入失败

**通用机制**：VWA 用坐标直接点击 `page.mouse.click(cx, cy)`，不自动 scroll-to-center。当元素中心 y > 720px（viewport 高度）时，click 静默失败或 TYPE 失焦→全选蓝。

| Task | Step | bbox | center_y | 后果 | 不修复理由 |
|------|------|------|----------|------|-----------|
| Task 33 | Step 6 | [350, 697, 378, 120] | **757 > 720** | TYPE 失焦→全选蓝 | agent 应多 scroll |
| Task 52 | Step 2-4 | [415, 687, 110, 120] | **747 > 720** | click 静默失败×3 | 同上 |

Task 52 额外注：元素可见比例仅 27.5%（<60% 阈值），理论上应被 AXTree 过滤但仍出现，属 VWA 框架边界行为。

---

### 4. 容器节点误点→导航失败

**通用机制**：classifieds 搜索结果每条 item 的文字区域（title+price+date）对应一个大容器 div（宽 623px，从 x=507 起），AXTree 把整个容器暴露为单一节点。其中心 x≈818.5 落在 price/date 文字上（非 `<a>` 子节点），click 无导航效果。

| Task | Steps | bbox | center | pc | 对比成功案例 |
|------|-------|------|--------|----|------------|
| Task 10 | 7/8/9 | [507, 261, 623, 159] | (818.5, 340.5) | False×3 | Step 4 点图片缩略图 [402,193,95,79]→True |
| Task 20 | 3/4/5 | [507, 333, 623, 119] | (818.5, 392.5) | False×3 | Step 2 点分类筛选链接 [162,501,86,19]→True |

**规律**：图片缩略图（`<a>` 实体，~95×79px，x≈400）→ 成功；文字容器 div（~623px，x=507）→ 失败。DOM 模式下 agent 偏向选大 bbox 节点（信息更丰富），反而更易选中容器节点。不修复（agent 元素选择能力问题）。

---

### 5. AXTree 节点类型误认→链式全选蓝

**Task 3, Step 15**（全选变蓝的不典型路径）：

- Step 14 把 eid=5748（bbox 高度仅 16px，link/label）当搜索框 TYPE → click 触发导航
- 新页面加载后，sPattern 搜索框位于 y=-19（部分滚出视口上方）
- Step 15 TYPE 点击 center y=(-19+40/2)=-19+20=1px（视口顶边缘）→ 实际 focus 失败 → Meta+A 全选蓝

高度 16px 的节点是诊断关键：classifieds 正常 input 高度 ≈ 40-50px，16px 表明是非 input 节点（link/label），TYPE 会导致意外导航而非文本输入。

---

### 6. Delete 操作：多删商品 + 感知失败

**Task 5**（Navigate to white car listing and delete it）：

- Step 1-2 连续删了"Pristine 2021 Toyota 86"和"Toyota 86 - Low Miles"（非目标）
- Step 3-17 不断 scroll/search 找"白色车"，无法从"listing 消失"推断"删除成功"
- Step 18 偶然看到 flash 消息"Your listing has been deleted"才 finish，最终 score=1.0

**根因**：§53 confirm 自动接受后重定向极快，PHP session flash 消息在下次 obs 采集时已消失。Agent 无法从负向证据（"列表里没有"）推断删除成功，只能等到偶然捕获 flash。

**不修复理由（§55）**：非结构性缺失，DOM 模式下识别"白色车"本身就依赖名称猜色，注入 success 信号无法解决识别错误根因，且超出标准 VWA 边界。

**对后续任务的影响**：Task 5 有 `require_reset=True`，本任务前全站重置，多余删除不影响其他 `require_reset` 任务；后续 no-reset 任务（Task 6/7）涉及摩托车/手机，与被删的车辆无关。

---

### 7. 视觉幻觉（DOM 模式下 235B 幻觉视觉内容）

**Task 222**（卷尺测量直径是否正确，`string_match: must_include: ['yes']`）：

- 1 步 finish，thought 开头写道：*"The image shows a measuring tape next to the microwave glass tray..."*
- 问题：DOM 模式根本看不到任何图片。Agent **幻觉了图像内容**，然后提交 `answer="Yes, the stated diameter of 16½" is correct"`
- 答案里含 "yes" → 碰巧命中 `must_include: ['yes']` → score=1.0

这是 B0 DOM 失败模式中最危险的一类：**幻觉 + 碰巧答对**。Agent 未感知到自己处于 DOM 模式（无图），直接对不存在的视觉内容生成描述并作出判断。与"沉默失败"（循环等待）相比，幻觉成功在 raw SR 里占一个真实位置，调试时极难发现。

**Task 233**（Reddit 图中角色是否出现在 classifieds listing，说出电影名，`must_include: ['lion king']`）：

- 1 步 finish。Agent 在正确 item 页面（id=28914，儿童书 Disney+Seuss 角色）看到文字描述，推断"Disney characters"并在答案中提到了 Lion King
- 问题：任务要求先**跨站**识别 Reddit 图中角色（DOM 无法看 Reddit 图），再回到 classifieds 验证
- Agent 完全跳过了跨站视觉识别步骤，从 listing 文字猜出电影名——**答案路径与任务意图完全不同，结果碰巧正确**

**分类**：两者均为 `kwd_only visual_fp`，FP 无争议。与 Task 201 对比：Task 201（snare drum）的文本推理是主动的搜索+标题语义推断，属于"合法文本推理被保守过滤"；Task 222/233 是被动幻觉/跳步，不存在争议。

---

## FP 分类统计（B0 DOM，234 episodes）

**成功总数：35**（raw SR 14.96%，234 tasks）

| 类别 | 数量 | 代表任务 | 机制 |
|------|------|---------|------|
| na_fp | 7 | 24,135,167,189,191,194,196 | `fuzzy_match("","N/A")=1.0`，evaluator bug |
| visual_fp（kwd_only） | 10 | 2,5,15,17,50,94,174,201,222,233 | DOM 无 listing 图，url_match/string_match 不验证理解 |
| §56 盲区（has_image 豁免） | 17 | 44-46,48,53-55,60,87,101,103,126,140,153,170,183,184 | has_image 豁免，可能部分为真实成功 |
| 真实成功 | 1 | 217 | 纯文本操作 |

**Adjusted SR**：8.04%（18/224，现行过滤）；严格下界 0.4%（1/234）；报告使用 8.04%，注明区间。

**§56 盲区内部分级**：
- 确定 FP（2个）：87/153，需对比参考图与 listing 图，DOM 看不到 listing 图
- 高风险 FP（5个）：44/45/46/48/101，"最贵/最新+分类"文本路径唯一可达，参考图与答案解耦（task 101 已验证）
- 不确定（10个）：53/54/55/60/103/126/140/170/183/184，需识别参考图中角色/场景，235B 可能真正利用了图像

**kwd_only 轻微过滤问题**：Task 201（snare drum black red）属于主动文本推理（搜关键词+标题语义推断）被归为 FP，是保守处理。论文注明 kwd_only 检测为启发式方法，adjusted SR 为下界估计。

---

## 失败原因分布

| 失败原因 | 数量 | 比例 | 备注 |
|---------|------|------|------|
| **fail_no_progress** | **61** | **26.3%** | ★ 最大失败源 |
| fail_finish_wrong_url_not_found | 46 | 19.8% | URL 不匹配 |
| success | 35 | 15.1% | (raw) |
| fail_finish_eval_mismatch | 33 | 14.2% | 评测不一致 |
| fail_max_steps_target_unreachable | 19 | 8.2% | 目标不可达 |
| fail_max_steps_search_repeat | 9 | 3.9% | 搜索循环 |
| fail_early_finish | 9 | 3.9% | 过早结束 |
| fail_incomplete_or_stuck | 7 | 3.0% | 页面卡住 |
| fail_max_steps_click_back_loop | 4 | 1.7% | click-back 循环 |
| fail_max_steps | 3 | 1.3% | 达到最大步数 |
| fail_finish_empty_answer | 3 | 1.3% | 空答案 |
| fail_finish_claim_missing | 2 | 0.9% | 声称缺失 |
| fail_parse_error | 1 | 0.4% | JSON 解析错误 |

DOM 模式的 `fail_no_progress`（26.3%）显著高于 SoM（10.3%），因为 DOM 纯文本无截图辅助，agent 在长 AXTree 中反复选择相同（错误的）元素但无法得到视觉反馈进行纠正。

---

## 定量结果

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| Raw SR | 14.96% (35/234) | **24.36%** (57/234) | 15.38% (36/234) |
| Adjusted SR | 8.04% (18/224) | **21.43%** (48/224) | 11.61% (26/224) |
| FP 分解 | 7 N/A + 10 visual | 9 N/A | 10 N/A |
| avg steps | 12.25 | 7.25 | 7.67 |
| cost/ep | $0.0467 | $0.0355 | **$0.0241** |
| P95 latency | **10,537ms** | 74,042ms | 50,540ms |
| no-op rate | 15.7% | **7.7%** | 31.7% |
| page_unchanged rate | 25.2% | 26.5% | **39.6%** |

### 统计显著性（McNemar 精确检验，adjusted labels）

| 对比 | 不一致对 (A-only / B-only) | p 值 | 显著性 |
|------|--------------------------|------|--------|
| SoM vs DOM | 37 / 7 | **5.3e-6** | ★★★ |
| Vision vs DOM | 16 / 8 | 0.152 | — |
| SoM vs Vision | 27 / 5 | **1.1e-4** | ★★★ |

> DOM adjusted SR 注：严格下界 0.4%（1/234 真实成功 task_217）；现行过滤 8.04%（扣除 7 N/A FP + 10 kwd visual FP，§56 盲区 17 tasks 暂计为成功）；报告使用 8.04%，注明为上界估计。
> 与 B1 对比：DOM 8.04% vs B1 0.85%（9.5×）；SoM 21.43% vs B1 16.24%（+5.19pp）；Vision 11.61% vs B1 8.12%。B0 235B 在所有三种模式下均优于 B1 4B。

---

*最后更新：2026-04-18（更新：全部定量数据对齐 parse_error 修复后最新分析；新增 Task 32 案例；新增失败原因分布表；新增统计显著性检验）*
