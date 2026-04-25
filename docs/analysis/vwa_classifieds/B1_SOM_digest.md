# B1 SoM Baseline 分析报告（Classifieds）

> **数据更新说明**：定量指标已更新至 `B1_3mode_classifieds_20260413` 运行。定性案例分析基于原始运行的逐 episode 审查，行为模式在结构上仍然有效。
> **注：visual_fp 层已在 §95 中废弃，adjusted SR 仅扣除 N/A FP + eval FP**

> 数据来源：`digest_som.jsonl`（193 条，仅含 SoM 模式失败 episode）
> 归因方法：GLM-5.1 + SoM 专属归因规则（S1-S5）+ 人工定性分析交叉验证
> 置信度：57.0% high / 43.0% medium
> 本报告**仅分析 SoM 模式**，不引用 DOM / vision 模式数据。
> 三模式共性缺陷与定量对比见 `B1_findings.md`。

---

## 总体概况

| 指标 | 数值 |
|------|------|
| SoM condition 总 episode 数 | 234 |
| 成功 | 41 (17.52%) |
| 失败 | 193 (82.48%) |
| Digest 覆盖率 | 193/193 (100%) |
| N/A FP | 10 |
| Eval FP | 1 |
| **Adjusted SR** | **13.84%**（31/224） |
| 失败 episode 平均步数 | 11.8 步 |
| 全 episode 平均步数 | 9.90 步 |

> §95 变更：eval FP 增加 1 个（此前为 0），adjusted SR 维持 13.84%（31/224）不变。SoM 无 visual_fp，§95 对 SoM 影响极小。

### 失败原因分布

| 失败原因 | 数量 | 占失败比 |
|----------|------|---------|
| fail_no_progress | 86 | 44.6% |
| fail_early_finish | 28 | 14.5% |
| fail_finish_wrong_url_not_found | 19 | 9.8% |
| fail_max_steps_target_unreachable | 17 | 8.8% |
| fail_finish_eval_mismatch | 15 | 7.8% |
| fail_finish_empty_answer | 9 | 4.7% |
| fail_incomplete_or_stuck | 6 | 3.1% |
| fail_max_steps_click_back_loop | 6 | 3.1% |
| fail_finish_claim_missing | 3 | 1.6% |
| fail_max_steps | 2 | 1.0% |
| fail_finish_wrong_url_price_mismatch | 1 | 0.5% |
| fail_max_steps_search_repeat | 1 | 0.5% |

### 脚手架 vs 模型能力归因

| 归因类型 | 数量 | 占比 |
|----------|------|------|
| 脚手架/表征结构性缺陷 | 63 | 33.9% |
| 模型能力问题 | 123 | 66.1% |

对比 DOM 模式（脚手架 51.2% / 模型 48.8%），SoM 的脚手架问题占比显著下降（-17.3pp），说明 SoM 截图有效缓解了 DOM 的信息瓶颈。

### SoM 特有字段统计

| som_failure_type | 数量 |
|------------------|------|
| text_over_vision | 56 |
| 不适用 | 46 |
| 标注遮挡 | 4 |

| som_visual_used | 数量 |
|-----------------|------|
| 否 | 67 |
| 是 | 41 |

---

## 一、非表征方法的脚手架缺陷

### 1.1 地点过滤不可达（3 例：task_58, 72, 74）

Classifieds 站点没有有效的地点筛选 UI，搜索框按商品名匹配不按地点过滤。与观测模式无关。

### 1.2 编辑页面字段不可达（2 例：task_4, 75）

商品编辑页面的价格/描述输入框不在当前 viewport 中。与观测模式无关。

---

## 二、SoM 表征结构性缺陷

### text_over_vision（56 例，87.5% 的表征类脚手架问题）

4B 模型系统性忽略 SoM 截图中的视觉细节（颜色、图片内容），主要依赖 `[SOM_MARKS]` 文本索引进行决策。

### visual scope anchoring：截图锚定导致任务降级

4B 模型将"当前截图可见内容"等同于"所有可操作范围"，无法推理出可以通过导航到达截图中不存在的页面/站点。

### Gallery 聚合类任务的 premature finish

4B 模型在需要扫描多个 item 才能回答的 gallery 聚合任务中，系统性地在 1 步内 premature finish。

### 视觉自信加速错误 finish（SoM 特有加剧机制）

SoM 模式因能看到缩略图，模型反而"更有信心"基于不完整的首页信息做出（错误的）快速判断。

### 视觉非确认 → 过早否定 finish（视觉自信的镜像变体）

模型看到缩略图但无法视觉确认目标属性，于是过早 finish 回答"0"。task_25 DOM 成功 vs SoM 失败反转是最干净的案例。

### 扁平化 marks 丢层级 → 容器/文本节点误点（SoM 特有）

SoM 扁平化 marks 将文本按线性列表排列，丢失了层级关联和交互性暗示。DOM 模式完全不存在此问题。

### label/textbox 混淆 → viewport 滚动 → marks 丢失 → P4 链

SoM 扁平化 marks 将表单 label 和相邻 textbox 并列，4B 模型无法区分。

---

## 二B、SoM 正向效应：视觉信息触发高阶推理策略（Mirage Effect 案例）

### 价格过滤器使用（task_17，成功 vs DOM 失败）

最干净的 Mirage Effect 正向证据。SoM agent 将价格约束分解为独立的过滤器操作，而 DOM agent 将整个 intent 作为单一搜索字符串。

### Sort By 使用（task_18）

SoM agent 在 step 5 使用了 `select_option` 选择 "Newly listed"——在 4B 模型中极为罕见。

### 搜索自纠正 + 价格过滤 + 多页浏览（task_19，成功）

三个高阶行为在单个 episode 中同时出现。跨步自纠正在 B1(4B) 中极为罕见。

---

## 三、N/A 任务 ua_match 假阳性

Classifieds 10 个 N/A reference task 中，SoM **10/10 全部 score=1.0**（ua_match FP）。

---

## 四、方法论说明

- **Adjusted SR**：仅扣除 N/A FP + eval FP（§95），不再扣除 visual FP
- **归因管线**：GLM-5.1 + SoM 专属归因规则（S1-S5）+ 人工交叉验证
- **额外输出字段**：`som_visual_used`、`som_mark_occlusion`、`som_failure_type`

---

*生成时间：2026-04-07*
*更新时间：2026-04-25（§95 FP 重构：废弃 visual_fp 层，新增 eval_fp=1；定量指标基于 B1_3mode_classifieds_20260413 运行）*
