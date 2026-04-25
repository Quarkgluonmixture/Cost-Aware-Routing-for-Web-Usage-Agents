# B0 Reddit 三模式实验报告

> Run: `B0_3mode_reddit_20260422`
> 模型: Qwen3-VL-235B-A22B（proxy API，MoE 22B 活跃参数）
> 站点: Reddit (Postmill), 210 tasks × 3 modes (DOM / SoM / Vision)
> 分析管线默认使用 **adjusted labels**（扣除 N/A FP + eval FP）
> **注：visual_fp 层已在 §95 中废弃，本文档中的 visual FP 数据为历史记录**
> 各模式专有分析见 `B0_DOM_digest.md` / `B0_SOM_digest.md` / `B0_Vision_digest.md`
> B0 vs B1 跨模型对比见 `B0_B1_findings.md`
> **本版数据三模式全部完整（210/210 × 3），2026-04-24 更新**

---

## 1. 成功率

### 1.1 主指标（Adjusted SR）

| 模式 | Raw SR | Adjusted SR | 成功数（adjusted） | FP 分解 |
|------|--------|------------|------------------|---------|
| DOM | 11.43% (24/210) | 8.78% (18/205) | 18 | N/A FP: 5, Visual FP: 1, Eval FP: 3 |
| **SoM** | **13.81%** (29/210) | **11.71%** (24/205) | 24 | N/A FP: 5, Eval FP: 2 |
| Vision | 8.57% (18/210) | 6.34% (13/205) | 13 | N/A FP: 5, Eval FP: 1 |

**三模式排序：SoM (11.71%) > DOM (8.78%) > Vision (6.34%)**

SoM 领先 DOM 约 2.93pp，领先 Vision 约 5.37pp。Reddit 的模式排序与 Classifieds 一致（SoM 最优），但整体 SR 低得多（Classifieds SoM adj: 20.54%）。

### 1.2 FP 机制说明

**N/A FP**（15 个 reference tasks）：5 个 N/A task_id（7/26/31/39/182），三模式各 5 个 FP。Agent prompt 无 N/A 出口（Rule 4: "NEVER give up"）→ 空答案或循环到截断 → 评测器误判。分母从 210→205。

**Visual FP（DOM 1 个）**：DOM 无截图，1 个 visual task（task 160）碰巧通过评测。SoM/Vision 有截图，无 visual FP。

**Eval FP**：DOM 3 个、SoM 2 个、Vision 1 个。program_html/string_match 类型 success 但 agent 未真正 finish 的情况。

### 1.3 统计显著性（McNemar 精确检验）

| 对比 | 不一致对 (A-only / B-only) | p 值 | 显著性 |
|------|--------------------------|------|--------|
| SoM vs Vision | 20 / 8 | **0.036** | ★ |
| SoM vs DOM | 13 / 6 | 0.167 | — (n.s.) |
| Vision vs DOM | 9 / 14 | 0.405 | — (n.s.) |

**SoM 显著优于 Vision**（p=0.036）。SoM vs DOM 差异不显著（p=0.167），Vision vs DOM 亦不显著（p=0.405）。Reddit 站点模式间差异不如 Classifieds 明显——Classifieds 中 SoM 显著优于 DOM（p<0.001）。

### 1.4 Bootstrap 95% CI

| 模式 | SR | CI 下界 | CI 上界 |
|------|-----|---------|---------|
| SoM | 12.38% | 8.10% | 17.14% |
| DOM | 9.05% | 5.24% | 13.33% |
| Vision | 6.67% | 3.81% | 10.48% |

SoM CI 与 DOM CI 大幅重叠（8.10%-13.33%），与 McNemar 不显著结果一致。

---

## 2. 效率指标

### 2.1 全 episode 效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| 平均成本 ($/ep) | 0.0516 | 0.0387 | **0.0227** |
| 平均步数 | 12.70 | 8.09 | **6.87** |
| P95 延迟 (ms) | **73,618** | 74,101 | 55,568 |

**Vision 成本最低**（$0.0227/ep）：无 AXTree 文字、无 SoM 标注图像，token 量最少。

**SoM 成本介于中间**（$0.0387/ep）：步数少于 DOM（8.09 vs 12.70），但每步图文混合 token 量高。

**DOM 成本最高**（$0.0516/ep）：步数最多（12.70），AXTree 冗长的 Reddit 论坛页面产生大量 token。

### 2.2 Wilcoxon 效率对比

| 对比 | 指标 | p 值 | 方向 |
|------|------|------|------|
| Vision vs SoM | total_cost | **1.1e-11** | Vision 更便宜 ★★★ |
| Vision vs SoM | p95_step_latency | **5.3e-7** | Vision 延迟更低 ★★★ |
| Vision vs DOM | total_cost | **9.9e-21** | Vision 更便宜 ★★★ |
| Vision vs DOM | p95_step_latency | **1.8e-5** | Vision 延迟更低 ★★★ |
| SoM vs DOM | total_cost | **3.9e-4** | SoM 更便宜 ★★★ |
| SoM vs DOM | p95_step_latency | 0.751 | **无显著差异** |

**成本排序**：Vision << SoM < DOM（均显著）。
**延迟排序**：Vision < DOM ≈ SoM（SoM 和 DOM 延迟无显著差异，Vision 显著更快）。

### 2.3 成本分解

| 模式 | 总成本 | 有效成本 | No-op 成本 | 循环成本 |
|------|--------|---------|-----------|---------|
| DOM | $0.0516 | $0.0319 (61.8%) | $0.0055 (10.6%) | $0.0143 (27.7%) |
| SoM | $0.0387 | $0.0243 (62.8%) | $0.0063 (16.3%) | $0.0082 (21.2%) |
| Vision | $0.0227 | $0.0145 (63.9%) | $0.0050 (22.0%) | $0.0032 (14.1%) |

三模式有效成本占比接近（61-64%），但 DOM 循环成本占比最高（27.7%），Vision 的 no-op 成本占比最高（22.0%）。

---

## 3. 失败模式对比

### 3.1 三模式失败原因对比

| 失败原因 | DOM | SoM | Vision |
|----------|-----|-----|--------|
| fail_no_progress | 48 (22.9%) | **73 (34.8%)** | **84 (40.0%)** |
| fail_finish_eval_mismatch | **50 (23.8%)** | 27 (12.9%) | 31 (14.8%) |
| fail_early_finish | 14 (6.7%) | **31 (14.8%)** | 21 (10.0%) |
| fail_max_steps_search_repeat | **29 (13.8%)** | 13 (6.2%) | 1 (0.5%) |
| fail_incomplete_or_stuck | 14 (6.7%) | 10 (4.8%) | **27 (12.9%)** |
| fail_finish_wrong_url_not_found | 9 (4.3%) | **12 (5.7%)** | 11 (5.2%) |
| fail_finish_claim_missing | 7 (3.3%) | 7 (3.3%) | **14 (6.7%)** |
| fail_max_steps_click_back_loop | **9 (4.3%)** | 2 (1.0%) | — |
| fail_finish_empty_answer | 6 (2.9%) | 4 (1.9%) | 3 (1.4%) |
| success | 24 (11.4%) | **29 (13.8%)** | 18 (8.6%) |

**三模式各有主导失败原因**：
- **DOM**：`fail_finish_eval_mismatch`（23.8%）+ `fail_max_steps_search_repeat`（13.8%）— 答案对齐问题突出，搜索循环严重
- **SoM**：`fail_no_progress`（34.8%）+ `fail_early_finish`（14.8%）— SoM 标注 ID 幻觉导致 click 失败；视觉确认加速过早（错误）决策
- **Vision**：`fail_no_progress`（40.0%）+ `fail_incomplete_or_stuck`（12.9%）— 坐标点击固有不稳定性，大量无效操作累积

### 3.2 脚手架 vs 模型归因

| 模式 | 脚手架/表征缺陷 | 模型能力问题 | 主要特征 |
|------|--------------|------------|---------|
| DOM | 中（visual 信息缺失、visual FP 1 个） | 中（搜索循环 13.8%、答案对齐 23.8%） | 搜索循环 + 信息瓶颈 |
| SoM | 高（ID 幻觉、click 失败导致 no_progress） | 中（early_finish 14.8%、eval_mismatch 12.9%） | 交互失败 + 过度自信 |
| Vision | 低（截图可见） | 高（坐标精度、no_progress 40.0%） | 执行失败主导 |

---

## 4. 跨模式交叉分析

> 三模式完整 cross_representation 分析（DOM + SoM + Vision），210 tasks common set。
> 注：以下数据使用 cross_rep 管线的三重 adjusted labels（na_fp + visual_fp + eval_fp，/210 分母）。

### 4.1 Oracle 分析

| 指标 | Raw | Adjusted |
|------|-----|----------|
| Oracle ceiling | 18.10% (38 tasks) | **16.19%** (34 tasks) |
| 最佳单模式 (SoM) | 13.33% | 10.95% |
| Routing headroom | 4.76pp | **5.24pp** |
| Oracle 选择分布 | DOM:12, SoM:14, Vision:12 | **DOM:10, SoM:13, Vision:11** |

**Adjusted routing headroom 5.24pp**：理论最优路由可将 SR 从 10.95% 提升到 16.19%。三模式在 oracle 中贡献均衡——DOM 10/34（29.4%）、SoM 13/34（38.2%）、Vision 11/34（32.4%）。

### 4.2 Oracle 选择按任务类型

| 任务类型 | DOM | SoM | Vision | 总计 |
|---------|-----|-----|--------|------|
| single_navigation | 11 | **13** | 10 | 34 |
| page_reading | 1 | 1 | **2** | 4 |

single_navigation 类型中 SoM 略占优（13/34），三模式分布较均衡。

### 4.3 集合分析（Adjusted，三模式）

| 集合 | 数量 | 占比 | 任务类型分布 |
|------|------|------|-------------|
| all_fail | 176 | 83.8% | single_nav:160, page_reading:13, action_on_item:3 |
| only_som | 9 | 4.3% | single_nav:8, page_reading:1 |
| dom_and_som (not vision) | 8 | 3.8% | single_nav:8 |
| only_vision | 5 | 2.4% | single_nav:5 |
| only_dom | 4 | 1.9% | single_nav:4 |
| som_and_vision (not dom) | 4 | 1.9% | single_nav:3, page_reading:1 |
| all_success | 2 | 1.0% | single_nav:1, page_reading:1 |
| dom_and_vision (not som) | 2 | 1.0% | single_nav:2 |

**关键发现**：
- **all_fail 高达 83.8%**，远高于 Classifieds 的 70.9% — Reddit 站点整体难度更高
- **SoM 独占 9 个 task**（4.3%）是三模式中最多
- **dom_and_som 共享 8 个**（不含 vision）— DOM 和 SoM 的互补性较强
- **所有三种模式都有独占成功**，路由有意义
- **交集仅 2 个 task** — 三模式成功集重叠极低

### 4.4 交集成功成本对比（2 tasks，三模式均成功）

样本量过小（仅 2 个 task），无法做有意义的统计分析。

---

## 5. 路由方向分析

### 5.1 Headroom 评估

| 路由场景 | Adjusted headroom | 备注 |
|---------|------------------|------|
| 三模式 oracle | **5.24pp** | 完整三模式 cross_representation |

### 5.2 各模式路由角色

**SoM（13/34 adjusted oracle, 38.2%）**— 主力模式：
- Adjusted SR 最高（10.95% cross-rep / 11.71% analysis_summary）
- 独占 9 个 task，成本效率优于 DOM

**DOM（10/34 oracle, 29.4%）**— 文本精度补充：
- 独占 4 个 task + 与 SoM 共享 8 个
- 搜索能力在 Reddit 上被搜索循环拖累，但精确文本匹配在特定导航任务上有优势

**Vision（11/34 oracle, 32.4%）**— 成本优化 + 独占贡献：
- 成本最低（$0.0227/ep），仅为 DOM 的 44%
- 独占 5 个 task，不可被其他模式替代

### 5.3 Reddit vs Classifieds 路由格局对比

| 维度 | Classifieds B0 | Reddit B0 |
|------|---------------|-----------|
| 最优单模式 SR (adj) | **SoM 20.54%** | SoM 11.71% |
| Oracle ceiling (adj) | **29.06%** | 16.19% |
| Routing headroom | 7.69pp | **5.24pp** |
| DOM oracle 占比 | 19.1% | **29.4%** |
| all_fail 比例 | 70.9% | **83.8%** |
| McNemar SoM vs DOM | p<0.001 ★★★ | p=0.167 (n.s.) |

Reddit 站点整体难度更高（all_fail 83.8%），但三模式路由 headroom（5.24pp）仍然可观——Oracle 选择分布更均衡（三模式各约 30%），意味着路由有较大空间。

---

## 6. 共性脚手架缺陷

### 6.1 Reddit Postmill 特有问题

**排序切换困难**：Reddit (Postmill) 的排序按钮（Hot/New/Active/Top）在 agent 操作中经常出现问题。多个任务要求"all time top"或"latest"帖子，但 agent 未切换排序或切换后页面显示"There's nothing here..."。

**论坛导航**：Reddit 的 All Forums 页面按字母排列 subreddit，agent 需要先定位到正确的 subreddit。SoM 模式下多次出现点击 subreddit 链接但页面无响应的情况（SoM 标注 ID 幻觉问题）。

**文件上传不可达**：涉及图片上传的任务中，agent 点击"Choose File"按钮但无法触发文件选择器（VWA Playwright 限制），所有此类任务三模式均失败。

**Comment 自链接死循环**：帖子页面的 "N comments" 链接指向当前页面自身（自链接），评论内容在页面下方需 scroll 才能看到。三模式均受影响。

### 6.2 N/A 任务 False Positive

15 个 N/A reference tasks（5 个 task_id），三模式各 5 个 FP。机制同其他站点：Agent prompt 无 N/A 出口 + evaluator 误判。

### 6.3 搜索策略局限

- **搜索循环是 DOM 最大问题**：DOM 模式 `fail_max_steps_search_repeat` 占 13.8%
- **图片搜索能力缺失**：Reddit 大量任务涉及"找到包含此图片的帖子"，DOM 模式无法看到图片内容
- **SoM text_over_vision**：SoM 模式下 agent 仍倾向于用文字搜索而非视觉匹配

### 6.4 Search-over-Browse 偏差（跨 B0/B1 分析）

> 分析脚本：`scripts/analysis/analyze_search_over_browse.py`，170 Reddit tasks。

Agent 系统性地过度依赖搜索框，忽视页面已有的分类导航和列表浏览。

- **74.1%** 的 Reddit task 中 agent 访问了 `/search` 页面
- **45.3%**（77/170）存在明确的 search-over-browse 偏差
- 首次搜索中位数在 **step 0** — agent 第一反应就是搜索

**Subreddit 起始 task 中搜索 = 有害**：B0 搜索组仅 6.2%（1/16），不搜索组 11.4%（5/44）。

### 6.5 Visual Task 的跨模式挑战

Reddit 的 210 个 task 中 177 个是 visual task（84.3%）。这些任务通常要求 agent 根据参考图片找到对应帖子。DOM 模式看不到图片（只有 alt text），SoM/Vision 模式可以看到但视觉匹配能力有限。

---

## 7. 路由信号分析（Confidence & Behavioral Signals）

> B0 为 API 调用（proxy），无 token-level logprobs；仅有 verbalized confidence 和 behavioral signals。

### 7.1 AUROC 区分力（全局，628-630 episodes）

| 信号 | 类型 | AUROC | 95% CI | 显著性 |
|------|------|-------|--------|--------|
| **ep_mean_verbalized** | verbalized | **0.769** | [0.708, 0.823] | ★★★ |
| ep_min_verbalized | verbalized | 0.697 | [0.634, 0.759] | ★★ |
| **max_repeat_streak** | behavioral | **0.670** | [0.613, 0.722] | ★★ |
| action_diversity | behavioral | 0.610 | [0.532, 0.684] | — |
| url_revisit_count | behavioral | 0.584 | [0.511, 0.655] | — |
| url_revisit_max | behavioral | 0.585 | [0.514, 0.654] | — |
| action_unique_types | behavioral | 0.510 | [0.433, 0.584] | — |
| url_unique_count | behavioral | 0.460 | [0.386, 0.535] | — |

**最强信号**：ep_mean_verbalized（AUROC=0.769）超过 0.6 路由阈值。

**Behavioral 信号较弱**：max_repeat_streak（0.670）勉强超过阈值，其余 behavioral 信号 CI 均跨越 0.5（随机线），不可靠。相比 Classifieds（action_diversity AUROC=0.741），Reddit 的 behavioral 信号区分力明显下降。

### 7.2 跨模式 AUROC 稳定性

| 信号 | DOM | SoM | Vision | 模式间差异 |
|------|-----|-----|--------|-----------|
| ep_mean_verbalized | **0.830** | 0.714 | 0.778 | 0.116 |
| ep_min_verbalized | **0.769** | 0.683 | 0.617 | 0.152 |
| max_repeat_streak | **0.678** | 0.636 | 0.709 | 0.073 |
| action_diversity | 0.581 | 0.603 | **0.689** | 0.108 |

- **ep_mean_verbalized 在 DOM 模式下区分力最强**（AUROC=0.830），Vision 次之（0.778）
- **max_repeat_streak 跨模式最稳定**（差异 0.073）

### 7.3 Routing Readiness 判定

| 维度 | 结果 |
|------|------|
| Token-level 区分力 | 不可用（API 模式） |
| Verbalized 区分力 | ✅ AUROC=0.769 |
| Behavioral 区分力 | ⚠️ AUROC=0.670 (max_repeat_streak，勉强可用) |
| 信号覆盖率 | ✅ >98% |
| **Overall: 可用于路由** | **✅（但信号弱于 Classifieds）** |

---

## 方法论说明

- **Adjusted labels**：N/A FP 优先（重叠时标为 na_fp），visual FP 次之（DOM kwd_only 过滤确认），eval_fp 第三层
- **两套 adjusted SR**：analysis_summary 使用 /205 分母（移除 N/A task）；cross_representation 使用 /210 分母（三重修正）。本报告主指标用 /205 分母
- **McNemar / Bootstrap CI**：使用 condition_metrics 管线的 adjusted labels
- **统计检验**：McNemar 精确检验（SR）+ Wilcoxon 有符号秩检验（成本/延迟）+ Bootstrap CI
- **Confidence 信号**：B0 为 API 模式，仅 verbalized confidence + behavioral signals（无 token-level logprobs）
- **数据时间**：本报告基于 2026-04-24 三模式完整数据（此前版本 Vision 仅 104/210）
- ⚠️ C23 WARN：SoM 存在时序 SR 退化（20.0% → 11.4% → 10.0%）

---

*更新时间：2026-04-24*
*数据来源：B0_3mode_reddit_20260422 analysis/ 目录（三模式全部 210/210 完整）*
*各模式详情：B0_DOM_digest.md / B0_SOM_digest.md / B0_Vision_digest.md*
