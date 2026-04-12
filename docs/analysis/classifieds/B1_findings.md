# B1 Classifieds 三模式实验报告

> **[DATA STALE]** 本报告数据基于 §33/§34/§36 修复前的运行结果。参考图片传递 + has_image FP 过滤修复后，DOM adjusted SR 和交叉分析数据将发生变化，待重跑后更新。

> Run: `B1_3mode_classifieds_20260404_141103`
> 模型: Qwen3-VL-4B bf16
> 站点: Classifieds (OSClass), 234 tasks × 3 modes (DOM / SoM / Vision)
> 分析管线默认使用 **adjusted labels**（扣除 visual FP + N/A FP）
> 各模式专有分析见 `B1_DOM_digest.md` / `B1_SOM_digest.md` / `B1_Vision_digest.md`

---

## 1. 成功率

### 1.1 主指标（Adjusted SR）

| 模式 | Adjusted SR | 95% CI | 成功数 |
|------|------------|--------|--------|
| DOM | **0.85%** | [0.00%, 2.14%] | 2 / 234 |
| SoM | **16.24%** | [11.97%, 20.94%] | 38 / 234 |
| Vision | **8.12%** | [4.70%, 11.54%] | 19 / 234 |

三模式 CI 均无重叠。

### 1.2 McNemar 检验（adjusted labels）

| 对比 | p-value | 显著? | 不一致对 (A-only / B-only) |
|------|---------|-------|--------------------------|
| SoM vs DOM | 2.84e-10 | **是** | 37 / 1 |
| Vision vs DOM | 2.21e-4 | **是** | 19 / 2 |
| Vision vs SoM | 0.0013 | **是** | 7 / 26 |

**三组比较全部显著**，排序：SoM > Vision > DOM。

> 注：raw labels 时 Vision vs DOM p=0.169（不显著）。切换到 adjusted 后 DOM 从 21 降到 2，差异变显著。

### 1.3 Raw SR 与 FP 分解

| 模式 | Raw SR | N/A FP | Visual FP | Adjusted SR |
|------|--------|--------|-----------|------------|
| DOM | 8.97% (21) | 10 | 9 | 0.85% (2) |
| SoM | 20.51% (48) | 10 | 0 | 16.24% (38) |
| Vision | 12.39% (29) | 10 | 0 | 8.12% (19) |

- **N/A FP**（30 个）：10 个 N/A reference task × 3 模式，`ua_match` 评测器 bug 全部误判为 success。根因：Agent prompt 无 N/A 出口（Rule 4: "NEVER give up"）→ 循环截断 → 空 answer → 评测器误判
- **Visual FP**（9 个，仅 DOM）：无图像但 `url_match` 碰巧通过。10 个 N/A task 同时也是 visual task，被 N/A 优先捕获
- DOM 21 个 raw 中仅 2 个是真正的能力成功

---

## 2. 效率指标

### 2.1 全 episode 效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| 平均成本 ($/ep) | 0.074 | 0.077 | **0.029** |
| 平均步数 | 14.9 | 11.8 | **8.0** |
| 平均 token | 39,575 | 41,259 | **15,225** |
| p95 步延迟 (s) | 45.5 | 82.6 | 53.2 |
| 平均能耗 (mWh) | 5.34 | 6.49 | **3.84** |
| 平均 CO₂e (g) | 1.17 | 1.43 | **0.84** |
| No-op rate | 4.3% | 2.0% | **0.6%** |
| Page unchanged rate | 22.7% | 23.4% | 30.1% |

### 2.2 成本统计检验（Wilcoxon signed-rank）

| 对比 | cost p | latency p |
|------|--------|-----------|
| Vision vs SoM | <1e-26 *** | 8.6e-7 *** |
| Vision vs DOM | <1e-25 *** | 5.2e-28 *** |
| SoM vs DOM | 0.88 (n.s.) | 3.6e-23 *** |

Vision 成本仅为其他模式的 **38%**。SoM 与 DOM 成本无差异，但 SoM 延迟显著更高（截图+标注开销）。

### 2.3 早停触发分布

| 触发原因 | DOM | SoM | Vision |
|---------|-----|-----|--------|
| page_unchanged_streak | 188 | 123 | 164 |
| action_failed | 100 | 49 | 9 |
| no_progress_streak | 33 | 9 | 2 |

Vision 的 `action_failed` 极低（不用 element_id），但 `page_unchanged` 占比最高——坐标 miss / scroll 到底 → 无变化 → 早停。

---

## 3. 跨模式交叉分析

### 3.1 Venn 集合（Adjusted）

| 区域 | 数量 | 占比 |
|------|------|------|
| 三模式均失败 | 188 | 80.3% |
| 仅 SoM | 25 | 10.7% |
| SoM + Vision（非 DOM） | 12 | 5.1% |
| 仅 Vision | 7 | 3.0% |
| DOM + SoM（非 Vision） | 1 | 0.4% |
| 仅 DOM | 1 | 0.4% |
| 三模式均成功 | 0 | 0% |
| DOM + Vision（非 SoM） | 0 | 0% |

- DOM + Vision 交集 = 0 — 两者成功 task **完全互补**
- Vision-only 7 个全是纯视觉任务（图片内容/颜色/形状/背景），不可被其他模式替代
- SoM + Vision 12 个，视觉信息是共同成功因素
- 三模式交集从 12 (raw) 降到 0 (adjusted)——DOM 的 visual FP 全部剔除

### 3.2 Oracle Ceiling

| 指标 | Raw | Adjusted |
|------|-----|----------|
| Best single (SoM) | 20.51% | 16.24% |
| Oracle ceiling | 25.64% | **19.66%** |
| Routing headroom | 5.13% | **3.42%** |

Headroom 3.42pp ≈ 8 个 task 的理论改进空间。

**Oracle 选择分布（Adjusted, 46 tasks）**：SoM 26 (56.5%) / Vision 18 (39.1%) / DOM 2 (4.3%)

### 3.3 任务类型分解（Adjusted SR）

| 任务类型 | n | DOM | SoM | Vision |
|---------|---|-----|-----|--------|
| single_navigation | 148 | 0.68% | **14.86%** | 6.76% |
| page_reading | 62 | 1.61% | **22.58%** | 12.90% |
| grid_position | 5 | 0% | 20% | 20% |
| date_count | 3 | 0% | **33.3%** | 0% |
| action_on_item | 9 | 0% | 0% | 0% |
| collection | 7 | 0% | 0% | 0% |

SoM 在所有有成功的类型上均最优或并列。action_on_item、collection 三模式全败。

### 3.4 模式转换矩阵（Adjusted, 234 tasks）

**Vision vs DOM**：

| | DOM 成功 | DOM 失败 |
|---|---|---|
| Vision 成功 | 0 | 19 |
| Vision 失败 | 2 | 213 |

净改善 +17。两者成功 task 完全不重叠。

**Vision vs SoM**：

| | SoM 成功 | SoM 失败 |
|---|---|---|
| Vision 成功 | 12 | 7 |
| Vision 失败 | 26 | 189 |

SoM 净优势 +19。

**SoM vs DOM**：

| | DOM 成功 | DOM 失败 |
|---|---|---|
| SoM 成功 | 1 | 37 |
| SoM 失败 | 1 | 195 |

SoM 净优势 +36。

---

## 4. 失败模式

### 4.1 DOM vs SoM 失败原因对比

| 失败原因 | DOM | SoM | 差值 |
|----------|-----|-----|------|
| fail_incomplete_or_stuck | 29.1% | 30.1% | +1.0pp |
| fail_max_steps_target_unreachable | 16.9% | 15.1% | -1.8pp |
| fail_no_progress | **9.9%** | **3.8%** | -6.1pp |
| fail_finish_wrong_url_not_found | 9.4% | **15.1%** | +5.7pp |
| fail_early_finish | 7.0% | **15.6%** | +8.6pp |
| fail_max_steps_click_back_loop | **6.1%** | 1.6% | -4.5pp |

**DOM 特有高发**：`fail_no_progress` + `click_back_loop` + `search_repeat` — 信息瓶颈导致反复尝试。
**SoM 特有高发**：`fail_early_finish` + `wrong_url` — 视觉信息加速过早（错误）决策。
**共性**：`fail_incomplete_or_stuck` 两模式占比接近（~30%），是最大失败类别。

### 4.2 Vision 主要失败路径

- **坐标 misclick**：4B 模型坐标精度不足，misclick 后不自纠正（重复相同坐标连续 3-4 步）
- **过早 finish**：缺乏 AXTree 结构化导航信息，首页看不到目标即放弃（1 步 finish）
- **信息充分幻觉**：列表页截图给 agent "已看到所有信息" 的错觉，不进详情页
- **Scroll 交替死循环**：3 个 task 出现 scroll up/down 交替，现有 cycle detection 无法捕获

### 4.3 脚手架 vs 模型归因

| 归因类型 | DOM | SoM |
|----------|-----|-----|
| 脚手架/表征缺陷 | 97/220 (44.1%) | 63/186 (33.9%) |
| 模型能力问题 | 123/220 (55.9%) | 123/186 (66.1%) |

SoM 脚手架问题占比下降 10pp：截图缓解信息瓶颈。但更多失败归因于模型——拿到足够信息但未能利用（text_over_vision 56 例）。

---

## 5. 路由信号评估

> 三模式 702 episodes，adjusted labels

### 5.1 信号区分力（Combined AUROC）

| 信号类型 | 最佳指标 | AUROC | 95% CI | 覆盖 |
|---------|---------|-------|--------|------|
| 行为信号 | **action_diversity** | **0.741** | [0.681, 0.797] | 100% |
| 行为信号 | url_revisit_max | 0.723 | [0.663, 0.781] | 100% |
| 行为信号 | url_revisit_count | 0.704 | [0.635, 0.767] | 100% |
| 行为信号 | max_repeat_streak | 0.673 | [0.612, 0.731] | 100% |
| Verbalized | ep_mean_verbalized | **0.695** | [0.591, 0.788] | 46% |
| Entropy | ep_max_entropy | 0.612 | [0.519, 0.710] | 53% |
| Token-level | ep_min_margin | 0.541 | — | 100% |
| Token-level | ep_mean_logprob | 0.530 | — | 100% |

### 5.2 跨模式一致性

| 信号 | DOM | SoM | Vision | 跨模式一致? |
|------|-----|-----|--------|-----------|
| action_diversity | 0.942 | 0.701 | 0.739 | 方向一致，DOM 异常高\* |
| url_revisit_max | 0.848 | 0.682 | 0.734 | 一致 |
| max_repeat_streak | 0.804 | 0.665 | 0.667 | 一致 |
| ep_mean_verbalized | — | 0.717 | 0.674 | 一致（DOM 无数据） |
| ep_max_entropy | — | 0.725 | 0.488 | **不一致** |

\* DOM adjusted 仅 2 个成功，AUROC 膨胀。

**行为信号跨模式最稳定**。Token-level 全部 AUROC ≈ 0.5（无用）。Verbalized 仅 SoM+Vision 覆盖（Vision 100%，SoM 37%，DOM ~1%）。

### 5.3 路由就绪度

| 维度 | 结论 |
|------|------|
| Token-level | **无**（AUROC ≈ 0.5） |
| 行为信号 | **有**（action_diversity 0.74，跨模式一致） |
| Verbalized | **有**（0.69，仅 SoM+Vision 覆盖） |
| 校准 | 未校准（ECE=0.82 token，0.56 verbalized） |
| **整体** | **行为信号可用于路由，verbalized 辅助 SoM↔Vision** |

---

## 6. 共性脚手架缺陷

以下缺陷与观测模式无关，三种模式均受影响。

### 6.1 地点过滤困难（3 例：task_58, 72, 74）

Classifieds 站点的地点筛选依赖搜索结果页的 City 文本输入框，而非主搜索框。模型普遍把地名塞入搜索框（按商品名匹配，不按地点过滤），导致搜索失败。

- task_58（DOM/SoM）：搜索"blue chair Washington DC"无结果。DOM/SoM thought 多次提到要用 City 过滤（DOM 4 次、SoM 5 次），但**始终无法执行**——"认知-执行鸿沟"
- task_58（**Vision 例外**）：step 8 成功找到 City 输入框并输入 "Washington, D.C."，视觉信息弥合了执行鸿沟（详见 Vision digest task 58）
- task_72/74：重复搜索地名，搜索框无法理解地点约束

归因：模型+UI。

### 6.2 编辑页面字段不可达（2 例：task_4, 75）

商品编辑页面的价格/描述输入框不在当前 viewport 中，Agent 持续滚动但无法定位目标字段。三种模式均受限于 viewport 尺寸。归因：框架。

### 6.3 `<select>` 下拉菜单三层不可达（VWA 框架级缺陷）

`<select>` 在 VWA 默认配置下对所有 agent 实质不可用。三层过滤：

1. **关闭状态**：option 元素 bbox=0，被 `TextObervationProcessor` 的 `width==0` 过滤
2. **展开状态**：`IN_VIEWPORT_RATIO_THRESHOLD = 0.6` 过滤 viewport 外选项
3. **scroll 限制**：`window.scrollBy()` 只滚页面，不滚 `<select>` 内部

模型能自行发现"All categories"链接绕路，但消耗 2-3 额外步骤。归因：VWA 框架。

### 6.4 Type 操作导致页面全选变蓝

`type` 操作偶尔导致页面文本被全选高亮（蓝色覆盖），影响 SoM/Vision 截图可读性。DOM 无影响。归因：Playwright。

### 6.5 极少翻页（模型能力缺陷）

模型几乎只会反复 scroll，极少点击分页控件。已知反例：SoM task_19（逐页翻页 1→2→3）、Vision task_58（识别 pagination 并翻页到 iPage=3）。DOM 未观察到翻页。视觉信息使分页控件更显著，但远未泛化为稳定策略。归因：模型。

### 6.6 confirm 弹窗不可交互（VWA 框架级缺陷）

Classifieds "Delete" 触发浏览器原生 `confirm()` 弹窗，VWA Playwright 默认不自动接受，导致删除操作被取消。所有删除任务三模式均失败。归因：VWA 框架。

### 6.7 N/A 任务 False Positive（10 例）

10 个 N/A reference task（24, 135, 164, 167, 189, 191, 194, 195, 196, 220），**三模式全部误判为 success=1.0**。

**误判机制**：
- **Type A（7 例）**：Agent 未 finish → runner 兜底填空 answer → `ua_match` 解读为"agent 无法完成"
- **Type B（3 例：167, 189, 196）**：Agent 提交错误答案 → `ua_match` 脑补为"隐式不可行"

**根因**：Agent prompt 无 N/A 出口（Rule 4: "NEVER give up"），三模式路径完全一致。归因：Prompt + 评测器。

### 6.8 ~~任务参考图片未传递给模型~~ [已修复 §33/§34/§36]

~~部分 VWA 任务 config 含 `"image"` 字段，但 `runner.py:924` 只传 `task.intent` 纯文本，**从未将参考图传给模型**。三种模式均受影响。修复方向：runner 构建 instruction 时追加参考图。归因：脚手架。~~

**修复说明**：§33 runner 构建 instruction 时追加参考图（三模式均支持）；§34 `analysis.py` 新增 `_load_has_image_task_ids()` + `compute_adjusted_success` 排除 has_image FP；§36 digest pipeline 新增 `visual_has_ref_image` subtype 区分"有参考图但模型能力不足"和"纯视觉属性 DOM 不可达"。DOM 模式现在可以看到任务参考图片（但仍无页面截图）。

### 6.9 搜索关键词过于具体

模型将任务描述全部约束拼接为搜索词，但 OSClass 仅做标题/描述简单文本匹配（且要求 ≥4 字符）。正确策略：宽泛品类词搜索 + 筛选器/排序/翻页缩小范围。潜在改善：EIP 站点先验（M5）。归因：模型策略。

---

## 7. Phase 2 路由方向

### 7.1 Headroom 评估

| 路由场景 | Adjusted headroom | 可行性 |
|---------|------------------|--------|
| DOM ↔ SoM | 0.4% | **无意义**（DOM adjusted 仅 2 个成功） |
| **SoM ↔ Vision** | **3.42%** | **有价值**（Oracle 选 Vision 18 次） |
| 三模式 Oracle | 3.42% | 与 SoM↔Vision 相同（DOM 贡献极小） |

### 7.2 推荐路由设计

**SoM ↔ Vision 路由**：
- SoM 作为默认（Adjusted SR 最高，16.24%）
- Vision 作为低成本替代（成本 38%，7 个独占成功）
- 路由信号：action_diversity（行为，0.74）+ verbalized（SoM+Vision 均有）

**DOM 不纳入路由**：Adjusted SR 0.85%，独占成功仅 1 个。

### 7.3 Pareto 分析

| 策略 | Adjusted SR | 平均成本 |
|------|-------------|---------|
| 全部 SoM | 16.24% | $0.077 |
| 全部 Vision | 8.12% | $0.029 |
| Oracle SoM↔Vision | **19.66%** | ~$0.057 |

Oracle routing 在 SoM 基础上提升 3.42pp SR 且可能降低成本。

---

## 方法论说明

- **Adjusted labels**：分析管线默认扣除 visual FP（DOM + visual task + raw success → False）和 N/A FP（N/A task + raw success → False）。所有图表、统计检验、CSV 均使用 adjusted
- **FP 优先级**：N/A FP 优先于 Visual FP（重叠时标记为 na_fp）
- **统计检验**：McNemar exact test（成功率），Wilcoxon signed-rank（成本/延迟），Bootstrap 10K resamples（CI）
- **路由信号**：AUROC 使用 adjusted labels，Mann-Whitney U 检验显著性
- **Benchmark 噪声检测**：Visual task 基于关键词 + config `image` 字段；N/A 基于 `reference_answers.fuzzy_match == "N/A"`

---

*生成时间：2026-04-12*
*数据目录：`results/visualwebarena/phase1/B1_3mode_classifieds_20260404_141103/analysis/`*
*合并自原 `B1_findings.md` + `B1_overall.md`*
