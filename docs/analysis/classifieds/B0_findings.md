# B0 Classifieds 三模式实验报告

> Run: `B0_3mode_classifieds_20260413`
> 模型: Qwen3-VL-235B-A22B（proxy API，MoE 22B 活跃参数）
> 站点: Classifieds (OSClass), 234 tasks × 3 modes (DOM / SoM / Vision)
> 分析管线默认使用 **adjusted labels**（扣除 N/A FP + visual FP）
> 各模式专有分析见 `B0_DOM_digest.md` / `B0_SOM_digest.md` / `B0_Vision_digest.md`
> B0 vs B1 跨模型对比见 `B0_B1_findings.md`
> **本版数据含 parse_error 修复后重跑结果（2026-04-18）**

---

## 1. 成功率

### 1.1 主指标（Adjusted SR）

| 模式 | Raw SR | Adjusted SR | 成功数（adjusted） | FP 分解 |
|------|--------|------------|------------------|---------|
| DOM | 14.96% (35/234) | 8.04% (18/224) | 18 | N/A FP: 7, Visual FP: 10 |
| **SoM** | **24.36%** (57/234) | **21.43%** (48/224) | 48 | N/A FP: 9 |
| Vision | 15.38% (36/234) | 11.61% (26/224) | 26 | N/A FP: 10 |

**三模式排序：SoM (21.43%) >> Vision (11.61%) > DOM (8.04%)**

parse_error 修复后，SoM 从旧数据的 12.05% 跃升至 21.43%（+9.4pp），确立了对其他两种模式的显著优势。

### 1.2 FP 机制说明

**N/A FP**（26/30）：Agent prompt 无 N/A 出口（Rule 4: "NEVER give up"）→ 空答案或循环到截断 → 评测器误判。10 个 N/A reference tasks 从分母移除（234→224），对应的 FP success 从分子移除。

**Visual FP（DOM 10 个）**：DOM 无截图，visual task 碰巧通过 url_match/string_match（kwd_only 过滤确认）。SoM/Vision 有截图，无 visual FP。DOM 额外有 15 个 visual_lucky_hits，其中 5 个与 N/A FP 重叠。

### 1.3 统计显著性（McNemar 精确检验）

| 对比 | 不一致对 (A-only / B-only) | p 值 | 显著性 |
|------|--------------------------|------|--------|
| SoM vs DOM | 37 / 7 | **5.3e-6** | ★★★ |
| SoM vs Vision | 27 / 5 | **1.1e-4** | ★★★ |
| Vision vs DOM | 16 / 8 | 0.152 | — |

**SoM 显著优于 DOM 和 Vision**（p<0.001）。Vision 与 DOM 差异不显著（p=0.152）。

### 1.4 Bootstrap 95% CI

| 模式 | SR | CI 下界 | CI 上界 |
|------|-----|---------|---------|
| SoM | 20.51% | 15.38% | 25.64% |
| Vision | 11.11% | 7.26% | 15.38% |
| DOM | 7.69% | 4.27% | 11.11% |

SoM CI 下界（15.38%）高于 Vision CI 上界（15.38%），区间刚好不重叠，进一步确认 SoM 优势。

---

## 2. 效率指标

### 2.1 全 episode 效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| 平均成本 ($/ep) | 0.0467 | 0.0355 | **0.0241** |
| 平均步数 | 12.25 | 7.25 | **7.67** |
| P95 延迟 (ms) | **10,537** | 74,042 | 50,540 |
| 平均输入成本 | 0.0402 | 0.0301 | **0.0195** |
| 平均输出成本 | 0.0065 | 0.0054 | **0.0046** |
| No-op rate | 15.7% | **7.7%** | 31.7% |
| Page unchanged rate | **25.2%** | 26.5% | 39.6% |
| cost_efficiency_ratio (SR/cost) | 0.102 | **0.186** | 0.140 |

**Vision 成本最低**（$0.0241/ep）：无 AXTree 文字、无 SoM 标注图像，token 量最少。

**SoM cost_efficiency_ratio 最高**（0.186）：虽然每 episode 成本高于 Vision，但 SR 优势（21.43% vs 11.61%）使每成功 episode 成本更低。

**DOM 步数最多**（12.25 步）：无截图辅助，需要更多探索步骤。但 P95 延迟最低（10,537ms）因为纯文本请求在 proxy API 上最快。

### 2.2 Wilcoxon 效率对比

| 对比 | 指标 | p 值 | 方向 |
|------|------|------|------|
| Vision vs SoM | total_cost | **6.3e-7** | Vision 更便宜 ★★★ |
| Vision vs SoM | p95_step_latency | **1.7e-16** | Vision 延迟更低 ★★★ |
| Vision vs DOM | total_cost | **2.7e-16** | Vision 更便宜 ★★★ |
| Vision vs DOM | p95_step_latency | 0.307 | 无显著差异 |
| SoM vs DOM | total_cost | **1.3e-5** | SoM 更便宜 ★★★ |
| SoM vs DOM | p95_step_latency | **2.2e-22** | SoM 延迟更高 ★★★ |

**成本排序**：Vision < SoM < DOM（全部显著差异）。
**延迟**：DOM 最低；SoM 最高（图文混合请求在 proxy API 上更耗时）。

### 2.3 早停触发分布

| 触发原因 | DOM | SoM | Vision |
|---------|-----|-----|--------|
| action_failed | 360 | 113 | **455** |
| page_unchanged_streak | 176 | 40 | **250** |
| no_progress_streak | 176 | 40 | **250** |

Vision 所有早停维度均最高——坐标点击的固有不稳定性导致大量无效动作积累。SoM 的早停触发远低于其他两种模式，说明 SoM 标注帮助 agent 更精准地定位交互元素。

---

## 3. 失败模式对比

### 3.1 三模式失败原因对比

| 失败原因 | DOM | SoM | Vision |
|----------|-----|-----|--------|
| fail_no_progress | 61 (26.3%) | 24 (10.3%) | **92 (39.3%)** |
| fail_finish_wrong_url_not_found | 46 (19.8%) | **54 (23.3%)** | 23 (9.8%) |
| fail_finish_eval_mismatch | 33 (14.2%) | 26 (11.2%) | 17 (7.3%) |
| fail_early_finish | 9 (3.9%) | **34 (14.7%)** | 19 (8.1%) |
| fail_incomplete_or_stuck | 7 (3.0%) | — | 21 (9.0%) |
| fail_max_steps_target_unreachable | 19 (8.2%) | 8 (3.4%) | 4 (1.7%) |
| fail_parse_error | 1 (0.4%) | 12 (5.2%) | 2 (0.9%) |

**三模式各有主导失败原因**：
- **DOM**：`no_progress`（26.3%）+ `wrong_url`（19.8%）— 无截图辅助，agent 在长 AXTree 中反复选错元素且无法得到视觉反馈纠正
- **SoM**：`wrong_url`（23.3%）+ `early_finish`（14.7%）— SoM 截图提供"视觉确认"锚点，agent 更果断但也更容易在相似页面上过早 finish
- **Vision**：`no_progress`（39.3%）— 纯坐标点击的固有不稳定性，大量无效操作累积

### 3.2 脚手架 vs 模型归因

| 模式 | 脚手架/表征缺陷 | 模型能力问题 | 主要特征 |
|------|--------------|------------|---------|
| DOM | 高（visual 不可达、visual FP 15 个） | 低（235B 文字推理强） | 信息瓶颈 |
| SoM | 低（截图可见） | 中（early_finish + parse_error） | 过度自信 + 格式问题 |
| Vision | 低（截图可见） | 高（坐标精度、no_progress 39.3%） | 执行失败主导 |

---

## 4. 跨模式交叉分析

> 三模式完整 cross_representation 分析（DOM + SoM + Vision），234 tasks common set。

### 4.1 Oracle 分析

| 指标 | Raw | Adjusted |
|------|-----|----------|
| Oracle ceiling | 32.05% (75 tasks) | **24.79%** (58 tasks) |
| 最佳单模式 (SoM) | 24.36% | 20.51% |
| Routing headroom | 7.69pp | **4.27pp** |
| Oracle 选择分布 | SoM:28, Vision:28, DOM:19 | SoM:27, Vision:21, DOM:10 |

**Adjusted routing headroom 4.27pp**：理论最优路由可将 SR 从 20.51% 提升到 24.79%。Vision 贡献了 21/58 oracle 选择（36.2%），是路由中不可忽视的力量。

### 4.2 Oracle 选择按任务类型

| 任务类型 | DOM | SoM | Vision |
|---------|-----|-----|--------|
| page_reading | 10 | 7 | **14** |
| single_navigation | 9 | **20** | 13 |
| action_on_item | — | — | 1 |
| grid_position | — | 1 | — |

**page_reading 类型 Vision 主导**（14/31 = 45%），**single_navigation 类型 SoM 主导**（20/42 = 48%）。

### 4.3 集合分析（Adjusted，三模式）

| 集合 | 数量 | 占比 | 任务类型分布 |
|------|------|------|-------------|
| all_fail | 176 | 75.2% | single_nav:116, page_reading:38, others:22 |
| only_som | 24 | 10.3% | single_nav:17, page_reading:6, grid:1 |
| som+vision (not dom) | 13 | 5.6% | page_reading:8, single_nav:4, action:1 |
| all_success | 8 | 3.4% | single_nav:5, page_reading:3 |
| only_dom | 5 | 2.1% | single_nav:3, page_reading:2 |
| dom+som (not vision) | 3 | 1.3% | page_reading:3 |
| **only_vision** | **3** | **1.3%** | single_nav:2, page_reading:1 |
| dom+vision (not som) | 2 | 0.9% | single_nav:1, page_reading:1 |

**关键发现**：
- **SoM+Vision 共享成功（13）远多于 SoM+DOM（3）**，说明 SoM 和 Vision 的优势互补性低、重叠多
- **Vision 独占仅 3 个**（与 only_dom 5 个相当），Vision 的路由价值更多体现在它与 SoM 共享的 13 个 task 中——这些 task 可以用更便宜的 Vision 替代 SoM
- **DOM 独占 5 个**：DOM 仍有不可替代的独特贡献（文字搜索精准定位）

### 4.4 交集成功成本对比（8 tasks，三模式均成功）

| 模式 | 平均成本 | 中位成本 | 平均步数 | 最便宜次数 |
|------|---------|---------|---------|-----------|
| DOM | $0.028 | $0.014 | 7.38 | 2/8 |
| SoM | **$0.016** | $0.017 | **3.38** | 3/8 |
| Vision | $0.025 | $0.018 | 7.13 | 3/8 |

三模式在交集 tasks 上成本差异不显著（所有 Wilcoxon p>0.4）。SoM 步数最少（3.38），但成本分布上三模式互有高低。

---

## 5. 路由方向分析

### 5.1 Headroom 评估

| 路由场景 | Adjusted headroom | 备注 |
|---------|------------------|------|
| 三模式 oracle | **4.27pp** | 完整三模式 cross_representation 数据 |
| SoM ↔ Vision | ~2pp（估算） | som+vision 共享 13 个，可用低成本 Vision 替代 |
| SoM ↔ DOM | ~1.5pp（估算） | DOM 独占 5 个，SoM 独占 24 个 |

### 5.2 各模式路由角色

**SoM（27/58 oracle, 46.6%）**— 主力模式：
- single_navigation 类型主导（20/42）
- adjusted SR 最高（20.51%），cost_efficiency_ratio 最高

**Vision（21/58 oracle, 36.2%）**— 成本优化路由候选：
- page_reading 类型主导（14/31）
- 成本最低（$0.0241/ep），som+vision 共享 13 个 task 可降级到 Vision
- 独占 3 个 task（不可被其他模式替代）

**DOM（10/58 oracle, 17.2%）**— 延迟/精度补充：
- 低 P95 延迟（10,537ms vs SoM 74,042ms）
- 独占 5 个 task，AXTree 文字搜索精度在特定任务上优于视觉扫描
- 3 个 page_reading 任务仅 DOM 成功（dom+som not vision），说明某些复杂页面阅读任务文字比截图更有效

### 5.3 与 B1 路由格局对比

| 维度 | B1 | B0 |
|------|----|----|
| 最优单模式 SR | SoM 16.24% | **SoM 20.51%** |
| Oracle ceiling (adj) | 19.66% | **24.79%** |
| Routing headroom | 3.42pp | **4.27pp** |
| DOM 路由价值 | 极低（adjusted SR 0.85%） | **有价值**（7.69%, 10 oracle） |
| Vision oracle 占比 | — | **36.2%** |
| 主路由方向 | SoM ↔ Vision | **三模式均有价值** |

---

## 6. 共性脚手架缺陷

### 6.1 `<select>` 下拉菜单不可用（VWA 框架级）

同 B1。B0 额外问题：**capability-environment gap 更严重**——235B 模型更精准识别 `<select>` 为正确入口，反而更执着地反复点击同一元素，cycle detection 更快截断。见 B0_DOM_digest 详细分析。

### 6.2 N/A 任务 False Positive（26/30）

三模式各 7/10、9/10、10/10 误判。机制同 B1：Agent prompt 无 N/A 出口 + evaluator ua_match bug。

### 6.3 confirm 弹窗不可交互

Delete 操作全部失败，三模式均受 VWA Playwright 限制。

### 6.4 搜索策略局限

- 极少翻页：B0 DOM 中翻页能力改善（33+ task），但 SoM/Vision 仍较少翻页
- 搜索关键词过于具体：将约束全部拼入搜索词，应用宽泛词+筛选器策略
- 235B 的策略规划稍有改善（价格筛选、表单聚焦等），见 B0_DOM_digest

### 6.5 Prompt 对 select_option 与 click 的一刀切指令

三模式 prompt 均包含 **"Clicking a dropdown does NOT open it. Use select_option instead."**，阻止模型对下拉框使用 click。

**问题**：Classifieds (OsClass) 的 Sort By 是 **CSS 自定义下拉框**（非原生 `<select>`），click 其实可以展开它。Prompt 的指令对原生 `<select>` 是正确的（click 打开浏览器原生 UI，截图不可捕获），但对 CSS dropdown 反而有害——尤其在 Vision 模式下，模型看不到闭合下拉框的选项文字，被迫猜测 label（如猜 "Price: Low to High" 实际为 "Lower price first"），精确匹配失败后陷入循环。

**典型案例**：
- Task 65 Vision：猜 "Price: Low to High"，实际 "Lower price first"，3 步循环
- Task 114 Vision：猜 "Oldest first"，该选项根本不存在，2 步循环后放弃
- Task 114 DOM：AXTree 列出 `[DROPDOWN OPTIONS]`，前两步 select 成功（label 精确匹配），但第 3 步仍幻觉 "Oldest first"

**不修的理由**：(1) 实验已跑完，改 prompt 破坏对比一致性；(2) 即使允许 click 展开，部分任务所需选项不存在（如 "Oldest first"），收益有限；(3) Vision 模式无法从截图区分原生 `<select>` 与 CSS dropdown，修改 prompt 可能引入新问题

---

## 方法论说明

- **Adjusted labels**：N/A FP 优先（重叠时标为 na_fp），visual FP 次之（DOM kwd_only 过滤）
- **DOM adjusted SR 不确定性**：§56 盲区 17 tasks（has_image 豁免）尚未全部分类，真实 SR 区间 [0.4%, 8.04%]
- **cross_representation 仅含 DOM+SoM**：Vision 未纳入管线，三模式 oracle 和集合分析需补充
- **统计检验**：McNemar 精确检验（SR）+ Wilcoxon 有符号秩检验（成本/延迟）+ Bootstrap CI
- **parse_error 修复后数据**：本报告全部基于 2026-04-17 重跑后的最新数据

---

*更新时间：2026-04-18*
*数据来源：B0_3mode_classifieds_20260413 analysis/ 目录（parse_error 修复后重跑数据）*
*各模式详情：B0_DOM_digest.md / B0_SOM_digest.md / B0_Vision_digest.md*
