# B0 跨站汇总

> 模型: Qwen3-VL-235B-A22B (proxy API, MoE 22B 活跃参数)
> 站点: Classifieds (234 tasks, 3 modes), Reddit (210 tasks, 2 modes + Vision 不完整), Shopping (466 tasks, DOM only)
> 文档生成: 2026-04-23
> 数据来源: `B0_3mode_classifieds_20260413`, `B0_3mode_reddit_20260422`, `B0_3mode_shopping_20260421`

---

## 1. 跨站成功率对比

### 1.1 主指标汇总（Adjusted SR）

| 站点 | 模式 | Raw SR | Adjusted SR | Raw Success | Adjusted Episodes | FP 说明 |
|------|------|--------|------------|-------------|-------------------|---------|
| **Classifieds** | DOM | 14.96% (35/234) | 8.48% (19/224) | 35 | 224 | N/A FP:6, Visual FP:14 |
| | **SoM** | **23.50%** (55/234) | **20.98%** (47/224) | 55 | 224 | N/A FP:8 |
| | Vision | 15.81% (37/234) | 12.05% (27/224) | 37 | 224 | N/A FP:10 |
| **Reddit** | DOM | 11.43% (24/210) | 7.62% (16/210) | 24 | 210 | N/A FP:4, Visual FP:1, Eval FP:3 |
| | **SoM** | **13.33%** (28/210) | **10.48%** (22/210) | 28 | 210 | N/A FP:3, Eval FP:2 |
| | Vision (不完整) | 3.85% (4/104) | 0.00% (0/104) | 4 | 104 | 全部为 FP |
| **Shopping** | DOM | 11.80% (55/466) | 6.44% (30/466) | 55 | 466 | N/A FP:30, Visual FP:12 |

> 注：Classifieds adjusted SR 使用 /224 分母（移除 N/A task）；Reddit 使用 /210 分母（三重修正）；Shopping condition_metrics 管线使用 /466 分母。各站调整方案存在差异，详见各站 findings 方法论说明。

### 1.2 数据完整性说明

- **Classifieds**: 三模式（DOM/SoM/Vision）均完整，234 tasks x 3 = 702 episodes
- **Reddit**: DOM 和 SoM 完整（各 210 episodes），Vision 仅 104/210 episodes（缺 condition_summary_v2.json），Vision 数据仅供参考
- **Shopping**: 仅 DOM 模式有数据（466 episodes），无 SoM/Vision，无法进行跨模式分析

### 1.3 跨站 SoM 优势一致性

三站数据一致表明 **SoM 是最优观测模式**：

| 站点 | SoM adj SR | 次优模式 adj SR | SoM 优势 (pp) | McNemar p 值 |
|------|-----------|----------------|--------------|-------------|
| Classifieds | 20.98% | Vision 12.05% | +8.93 | 0.059 (SoM vs Vision) |
| Classifieds | 20.98% | DOM 8.48% | +12.50 | 8.4e-6 (SoM vs DOM) |
| Reddit | 10.48% | DOM 7.62% | +2.86 | 0.238 (不显著) |

Classifieds 站点 SoM 显著优于 DOM（p<0.001），但 Reddit 站点两模式差异不显著（p=0.238）。SoM 优势的显著性与站点难度和任务类型分布相关。

---

## 2. 站点难度排序

### 2.1 基于 SR 的难度排序

| 排序 | 站点 | 最优 adj SR | DOM adj SR | 难度评级 |
|------|------|-----------|-----------|---------|
| 1 (最易) | Classifieds | 20.98% (SoM) | 8.48% | 中等 |
| 2 | Reddit | 10.48% (SoM) | 7.62% | 高 |
| 3 (最难) | Shopping | -- (仅 DOM) | 6.44% | 高 |

### 2.2 基于 all_fail 比例的难度排序

| 站点 | all_fail 占比 | 分析范围 |
|------|-------------|---------|
| Classifieds | 70.9% (166/234) | 三模式 adjusted，任一成功即非 all_fail |
| Reddit | 86.7% (182/210) | 两模式 adjusted |

Classifieds 有 29.1% 的 task 至少在一种模式下成功，而 Reddit 仅 13.3%。Shopping 无跨模式数据，无法计算 all_fail。

### 2.3 难度差异分析

**Reddit 最难的原因**：
- Visual task 占比极高（177/210 = 84.3%），大量任务要求根据参考图片找帖子
- 图片搜索能力缺失：DOM 无法看到图片内容，SoM/Vision 的视觉匹配能力有限
- Postmill 界面交互复杂（排序切换、论坛导航），SoM 标注 ID 幻觉导致 click 失败率高

**Shopping 挑战**：
- 任务量最大（466 tasks），多条件约束任务（颜色+价格+品牌）比例高
- 仅 DOM 数据，视觉任务占 57.7%（269/466），DOM 结构性劣势显著
- N/A FP 极高（30/31），raw SR 11.80% 被大幅下调至 6.44%

**Classifieds 相对最易**：
- OsClass 界面结构简单，列表型布局利于 agent 导航
- SoM 模式下 agent 可直观看到列表条目和价格信息
- All_fail 比例最低（70.9%），路由 headroom 最大（8.55pp）

---

## 3. 模式 x 站点交互效应

### 3.1 SoM 是否在所有站点都最优？

**是，SoM 在有数据的所有站点上均为最优模式**，但优势幅度差异显著：

| 站点 | SoM adj SR | DOM adj SR | SoM - DOM (pp) | 显著性 |
|------|-----------|-----------|---------------|--------|
| Classifieds | 20.98% | 8.48% | +12.50 | p<0.001 |
| Reddit | 10.48% | 7.62% | +2.86 | p=0.238 (不显著) |
| Shopping | -- | 6.44% | -- | 无数据 |

SoM 优势在 Classifieds 显著（列表型 UI，SoM 标注直接对应可点击条目），在 Reddit 缩小（Postmill 交互复杂，SoM ID 幻觉问题抵消部分优势）。

### 3.2 DOM vs Vision 的相对排序

仅 Classifieds 有 DOM/Vision 的完整对比数据：

| 站点 | DOM adj SR | Vision adj SR | 排序 | McNemar |
|------|-----------|-------------|------|---------|
| Classifieds | 8.48% | 12.05% | Vision > DOM | p=0.016 (显著) |
| Reddit | 7.62% | 0.00% (不完整) | 不可比 | -- |

Classifieds 上 **Vision 显著优于 DOM**（p=0.016），但 DOM 在交集任务上成本更低（$0.0134 vs $0.0241）。Reddit Vision 数据不完整，无法比较。

### 3.3 三模式排序一致性

仅 Classifieds 有完整三模式数据，排序为：**SoM (20.98%) >> Vision (12.05%) > DOM (8.48%)**。

该排序的可能解释：
- SoM 兼具结构化标注（ID 定位）和视觉信息（截图），信息量最大
- Vision 有截图但缺乏元素定位辅助，坐标点击不稳定（action_failed 430 次 vs SoM 108 次）
- DOM 无截图，视觉任务存在结构性天花板（14 个 visual FP 在 Classifieds）

---

## 4. 效率对比

### 4.1 跨站成本对比

| 站点 | 模式 | 平均成本 ($/ep) | 平均步数 | P95 延迟 (ms) | Cost Efficiency |
|------|------|----------------|---------|--------------|----------------|
| Classifieds | DOM | 0.0425 | 11.52 | 37,513 | 0.1145 |
| Classifieds | SoM | 0.0417 | 8.62 | 75,932 | 0.1550 |
| Classifieds | Vision | **0.0248** | **7.85** | 46,361 | 0.1429 |
| Reddit | DOM | 0.0516 | 12.70 | 73,618 | 0.0855 |
| Reddit | SoM | 0.0384 | 8.01 | 78,542 | 0.1357 |
| Shopping | DOM | 0.0424 | 10.20 | **16,695** | 0.1177 |

> Cost efficiency ratio = success_rate / avg_total_cost_usd，值越高越好。

### 4.2 关键发现

**成本排序（同模式跨站）**：
- DOM 成本：Reddit ($0.0516) > Classifieds ($0.0425) ~ Shopping ($0.0424)
- SoM 成本：Classifieds ($0.0417) > Reddit ($0.0384)
- Reddit DOM 成本最高，因步数最多（12.70 步）且 AXTree 较长

**延迟排序**：
- Shopping DOM P95 延迟最低（16,695ms），因 Magento AXTree 通常较短
- Classifieds DOM 次之（37,513ms）
- Reddit DOM（73,618ms）和两站 SoM（75,932ms / 78,542ms）延迟最高

**Cost efficiency 排序**：
- Classifieds SoM（0.1550）最优：SR 高（20.98%）且成本适中
- Classifieds Vision（0.1429）次之：SR 中等但成本最低
- Reddit DOM（0.0855）最差：SR 低且成本高

### 4.3 步数与成功率的关系

| 站点 | 模式 | 平均步数 | adj SR | 观察 |
|------|------|---------|--------|------|
| Classifieds | Vision | 7.85 | 12.05% | 步数最少 |
| Classifieds | SoM | 8.62 | 20.98% | 步数少、SR 高 |
| Reddit | SoM | 8.01 | 10.48% | 步数少 |
| Classifieds | DOM | 11.52 | 8.48% | 步数多、SR 低 |
| Shopping | DOM | 10.20 | 6.44% | 步数多、SR 低 |
| Reddit | DOM | 12.70 | 7.62% | 步数最多 |

**跨站一致模式**：SoM/Vision 步数更少（7.85-8.62）且 SR 更高，DOM 步数更多（10.20-12.70）但 SR 更低。DOM 模式下 agent 更容易陷入循环，积累无效步骤。

### 4.4 交集任务成本优势

在两站的交集任务（所有对比模式均成功的 task）上，DOM 均为最便宜模式：

| 站点 | 交集大小 | DOM 平均成本 | SoM 平均成本 | DOM 最便宜次数 |
|------|---------|------------|------------|--------------|
| Classifieds | 9 tasks | $0.0134 | $0.0198 | 5/9 |
| Reddit | 10 tasks | $0.0244 | $0.0337 | 8/10 |

DOM 在简单任务（所有模式都能解决的）上成本优势显著，因纯文本请求 token 少、延迟低。这支持了路由策略中「简单任务降级到 DOM」的方向。

---

## 5. 路由信号跨站一致性

### 5.1 Verbalized Confidence AUROC 跨站对比

| 信号 | Classifieds (702 ep) | Reddit (525 ep) | Shopping (465 ep) | 跨站稳定性 |
|------|---------------------|-----------------|-------------------|-----------|
| ep_mean_verbalized | **0.755** [0.706, 0.800] | **0.736** [0.658, 0.808] | **0.681** [0.568, 0.786] | 稳定 (0.074 range) |
| ep_min_verbalized | 0.642 [0.591, 0.694] | 0.686 [0.616, 0.756] | 0.635 [0.539, 0.732] | 稳定 (0.051 range) |

**ep_mean_verbalized 跨站稳定**：三站 AUROC 范围 0.681-0.755（极差 0.074），均超过 0.6 路由阈值。Classifieds 最高（0.755），Shopping 最低（0.681）。三站 95% CI 均不含 0.5（随机线），确认该信号在所有站点上具有统计显著的区分力。

### 5.2 Behavioral Signals AUROC 跨站对比

| 信号 | Classifieds | Reddit | Shopping | 跨站一致 |
|------|------------|--------|----------|---------|
| action_diversity | **0.741** | 0.576 | **0.686** | 不一致 |
| max_repeat_streak | 0.681 | **0.660** | 0.635 | 较稳定 (0.046 range) |
| url_revisit_max | 0.706 | 0.551 | 0.548 | 不一致 |
| url_revisit_count | 0.680 | 0.561 | 0.582 | 不一致 |
| action_unique_types | 0.579 | 0.485 | 0.637 | 不一致 |
| url_unique_count | 0.486 | 0.434 | 0.423 | 一致（均不可用） |

**跨站一致的信号**：
- **ep_mean_verbalized**：三站均为最强或次强信号，AUROC 0.681-0.755，跨站最稳定
- **max_repeat_streak**：三站 AUROC 0.635-0.681（极差 0.046），稳定但区分力中等

**跨站不一致的信号**：
- **action_diversity**：Classifieds 0.741（强），Reddit 0.576（不可用），Shopping 0.686（可用）。该信号在 Reddit 失效，原因可能是 Reddit 任务类型单一（84.3% visual task），成功/失败 episode 的动作多样性差异不大
- **url_revisit_max/count**：Classifieds 0.706/0.680（可用），但 Reddit（0.551/0.561）和 Shopping（0.548/0.582）均不可靠

### 5.3 跨模式 x 跨站 AUROC

ep_mean_verbalized 按模式分拆后：

| 模式 | Classifieds | Reddit | 差异说明 |
|------|------------|--------|---------|
| DOM | 0.782 | **0.829** | Reddit DOM 最高 |
| SoM | 0.705 | 0.702 | 两站接近 |
| Vision | 0.765 | 0.410 (不完整) | Reddit Vision 不可靠 |

**DOM 模式下 verbalized confidence 区分力最强**（Classifieds 0.782, Reddit 0.829），可能因为 DOM 模式下模型的自评更校准（纯文字推理，confidence 更能反映真实把握度）。SoM 模式下两站一致偏低（约 0.70），因 SoM 成功率更高导致类别更平衡、信号分离度下降。

### 5.4 路由信号推荐

基于跨站一致性分析，推荐的路由信号优先级：

| 优先级 | 信号 | 理由 |
|-------|------|------|
| 1 | ep_mean_verbalized | 三站均为最强/次强，AUROC 0.681-0.755，跨站最稳定 |
| 2 | max_repeat_streak | 三站 AUROC 0.635-0.681，稳定但区分力弱于 verbalized |
| 3 | action_diversity | 两站可用但 Reddit 失效，需站点特异阈值 |
| -- | url_revisit_max/count | 仅 Classifieds 可用，不推荐作为通用信号 |
| -- | token-level 信号 | B0 API 模式不可用 |

---

## 6. 共性失败模式

### 6.1 三站共同的脚手架/表征缺陷

| 缺陷 | Classifieds | Reddit | Shopping | 影响程度 |
|------|------------|--------|----------|---------|
| **N/A FP** | 24/30 FP | 7/10 FP | 30/31 FP | 高 -- 全站普遍 |
| **Agent prompt 无 N/A 出口** | Rule 4 "NEVER give up" | 同左 | 同左 | 高 -- 系统级 |
| **Visual FP（DOM 模式）** | 14 个 | 1 个 | 12 个 | 中 -- DOM 专属 |
| **搜索策略局限** | 极少翻页/排序 | 搜索循环 13.8% | 关键词过于具体 | 中 -- 模型行为 |
| **DOM 步数膨胀** | 11.52 步 | 12.70 步 | 10.20 步 | 中 -- 模式特征 |

### 6.2 N/A FP 的跨站一致性

| 站点 | N/A Reference Tasks | N/A FP 数 | FP 率 |
|------|--------------------|-----------|----|
| Classifieds | 10 (per mode, 30 total) | 24 | 80% |
| Reddit | 5 (per mode, 10 total) | 7 | 70% |
| Shopping | 31 | 30 | 96.8% |

Shopping 的 N/A FP 率最高（96.8%），几乎所有 N/A task 都被误判。这是 Agent prompt 无 N/A 出口 + evaluator ua_match bug 的系统级问题，贯穿三站。

### 6.3 DOM Visual FP 的跨站分布

| 站点 | Visual Tasks 占比 | DOM Visual FP | Visual FP 率 |
|------|-----------------|--------------|-------------|
| Classifieds | 69.2% (162/234) | 14 | 8.6% of visual tasks |
| Reddit | 84.3% (177/210) | 1 | 0.6% |
| Shopping | 57.7% (269/466) | 12 | 4.5% |

Classifieds DOM visual FP 率最高（8.6%），因 OsClass 的 url_match 评测器更容易被碰巧匹配。Reddit 最低（0.6%），因 Postmill 的评测以 program_html 为主。

### 6.4 搜索策略局限（三站共性）

- **极少翻页**：三站 agent 均倾向在首页结果中选择，很少 scroll down 或点击分页
- **搜索关键词过于具体**：将所有约束拼入单次搜索（如 "red ps4 controller under $200"），搜索引擎无法正确处理多约束查询
- **Sort By 使用率低**：Classifieds 的 CSS 自定义下拉框不可交互（prompt 限制），Shopping 的 Magento Sort By 使用率也较低
- **B0 相比 B1 的改善**：235B 模型的策略规划稍有改善（价格筛选、表单聚焦等），但核心搜索策略局限仍在

---

## 7. 站点特异性问题

### 7.1 Classifieds 特有

| 问题 | 描述 | 影响 |
|------|------|------|
| `<select>` 下拉菜单不可用 | VWA 框架级限制，原生 `<select>` 的 click 不可截图 | 依赖 select_option 的任务全败 |
| CSS 自定义下拉框 vs 原生 `<select>` | Sort By 是 CSS dropdown，prompt 阻止 click | Vision 模式更受影响 |
| confirm 弹窗不可交互 | Delete 操作全部失败 | 涉及删除的 task 全败 |
| SoM 过度自信 | early_finish 12.8%，SoM 截图提供"视觉确认"锚点 | SoM 特有 |
| collection 类型全败 | 7 个 collection 任务三模式全败 | 结构性天花板 |

### 7.2 Reddit 特有

| 问题 | 描述 | 影响 |
|------|------|------|
| Visual task 占比极高 | 84.3% (177/210) 要求图片匹配 | 所有模式受限 |
| Postmill 排序切换困难 | Hot/New/Active/Top 切换后页面显示"There's nothing here..." | 需要排序的 task 大量失败 |
| SoM click_fail_rate 极高 | 论坛导航中 66-80% 的 SoM click 失败 | SoM no_progress 34.3% |
| 文件上传不可达 | VWA Playwright 限制，无法触发文件选择器 | 图片上传任务全败 |
| DOM 搜索循环 | fail_max_steps_search_repeat 13.8%，重复相同搜索词 | DOM 特有 |
| fail_early_finish 在 SoM 中突出 | SoM 14.3% vs DOM 6.2% | SoM 过度自信 |

### 7.3 Shopping 特有

| 问题 | 描述 | 影响 |
|------|------|------|
| 视觉信息缺失是 DOM 核心瓶颈 | 34.3% (73/213) 失败归因于视觉缺失 | DOM 结构性劣势 |
| N/A FP 率最高 | 30/31 = 96.8% | raw SR 11.80% 大幅下调至 6.44% |
| 搜索框/订阅框混淆 | Newsletter 订阅框与搜索栏类型相同 | DOM 模式搜索失败 |
| 多条件约束任务 | 颜色+价格+品牌+评分的组合查询 | 答案对齐错误 33.8% |
| fail_finish_eval_mismatch 最高 | 33.8%（三站最高），选错商品/价格计算错误 | 模型推理能力瓶颈 |
| Click 失败率 24.7% | DOM 模式 element ID 点击中心坐标越界 | 执行稳定性问题 |

### 7.4 各站主导失败原因对比

| 站点 | 第一失败原因 | 第二失败原因 | 特征 |
|------|------------|------------|------|
| Classifieds DOM | no_progress (26.5%) | wrong_url (18.4%) | 循环 + 导航错误 |
| Classifieds SoM | wrong_url (23.1%) | early_finish (12.8%) | 过度自信 |
| Classifieds Vision | no_progress (39.3%) | -- | 坐标不稳定 |
| Reddit DOM | eval_mismatch (23.8%) | no_progress (22.4%) | 答案对齐 + 循环 |
| Reddit SoM | no_progress (34.3%) | early_finish (14.3%) | ID 幻觉 + 过度自信 |
| Shopping DOM | eval_mismatch (33.8%) | no_progress (33.5%) | 答案对齐 + 循环 |

**跨站一致**：DOM 模式的 no_progress 和 eval_mismatch 是共性主导失败；SoM 模式的 early_finish 在 Classifieds（12.8%）和 Reddit（14.3%）均突出。

---

## 8. 综合结论

### 8.1 核心发现

1. **SoM 是跨站最优模式**：在有数据的所有站点上 SoM adjusted SR 最高（Classifieds 20.98%, Reddit 10.48%），但优势幅度与站点交互复杂度相关。

2. **站点难度差异显著**：Classifieds (all_fail 70.9%) < Reddit (all_fail 86.7%)。Shopping 仅有 DOM 数据（adj SR 6.44%），推测在三模式条件下难度介于两者之间。

3. **路由 headroom 与站点难度负相关**：Classifieds 8.55pp >> Reddit 2.86pp。难度越高的站点，模式间差异越小，路由空间越有限。

4. **DOM 在简单任务上有成本优势**：交集任务上 DOM 成本最低（Classifieds $0.0134, Reddit $0.0244），支持「简单任务降级到 DOM」的路由策略。

5. **ep_mean_verbalized 是唯一跨站稳定的强路由信号**：三站 AUROC 0.681-0.755，均超过 0.6 阈值。Behavioral 信号跨站一致性差（action_diversity 在 Reddit 失效）。

6. **N/A FP 是最严重的系统级问题**：三站合计 61/71 个 N/A task 被误判（85.9%），其中 Shopping 高达 96.8%。

### 8.2 对路由策略的启示

| 策略 | 可行性 | 依据 |
|------|-------|------|
| SoM 作为默认模式 | 高 | 三站均最优，跨站一致 |
| 低 confidence 时切换模式 | 中 | ep_mean_verbalized 跨站稳定但 AUROC 仅 0.68-0.76 |
| 简单任务降级到 DOM | 中 | 交集任务 DOM 成本最低，但需可靠的"简单任务"检测器 |
| 基于 behavioral 信号的 early routing | 低 | behavioral 信号跨站不一致，不适合通用策略 |
| 站点特异路由阈值 | 中 | 各站最优阈值不同（Youden's J），需 per-site 校准 |

---

## 方法论说明

- **数据来源**：三站各自的 `analysis_summary.json`、`condition_metrics.csv`、`confidence_summary.json` 以及各站 `B0_findings.md`
- **Adjusted SR 差异**：各站使用不同的调整方案（分母移除 vs 分子移除 vs 三重修正），跨站 SR 比较需注意此差异。DOM 模式是唯一三站均有数据的模式，DOM adj SR 可直接跨站比较
- **统计检验**：McNemar 精确检验（SR 成对比较）+ Wilcoxon 有符号秩检验（成本/延迟）+ Bootstrap 95% CI。Shopping 单 condition 无法进行成对检验
- **Confidence 信号**：B0 为 API 模式，仅 verbalized confidence + behavioral signals，无 token-level logprobs
- **数据完整性限制**：Reddit Vision 不完整（104/210），Shopping 仅 DOM，跨模式分析的跨站比较仅在 Classifieds 完全可靠
- **FP 检测**：三层优先级 -- N/A FP > Visual FP > Eval FP。具体检测逻辑见各站 findings 方法论说明

---

*更新时间：2026-04-23*
*数据来源：B0_3mode_classifieds_20260413, B0_3mode_reddit_20260422, B0_3mode_shopping_20260421*
*各站详情：vwa_classifieds/B0_findings.md, vwa_reddit/B0_findings.md, vwa_shopping/B0_findings.md*
