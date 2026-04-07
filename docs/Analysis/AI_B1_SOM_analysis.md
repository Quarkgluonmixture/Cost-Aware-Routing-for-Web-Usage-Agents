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
