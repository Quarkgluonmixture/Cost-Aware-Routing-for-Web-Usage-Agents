# B1 Reddit 三模式实验报告

> Run: `B1_3mode_reddit_20260413`
> 模型: Qwen3-VL-4B bf16（本地推理，do_sample=False，max_new_tokens=384）
> 站点: Reddit (Postmill), 210 tasks × 3 modes (DOM / SoM / Vision)
> 分析管线默认使用 **adjusted labels**（扣除 N/A FP + eval FP）
> **注：visual_fp 层已在 §95 中废弃，本文档中的 visual FP 数据为历史记录**
> 各模式专有分析见 `B1_DOM_digest.md` / `B1_SOM_digest.md` / `B1_Vision_digest.md`
> B0 vs B1 跨模型对比见 `B0_B1_findings.md`
>
> **v2 (2026-04-26)**：
> - **SoM 是 max_marks 80→200 重跑后版本**（§94），04-25 启动，04-26 16:36 完成 210/210。目的：验证之前 SoM 反垫底是否因为标记不足。**结果：反转未消失**（DOM 6.67% > SoM 4.76% > Vision 1.43%）。
> - 04-26 全部 condition rederive（PUR 重算 → eval FP 判定更新），SR 数字相对 v1 (2026-04-24) 有 0.1-1.0pp 漂移。
> - **B1 数据非最终**：DGX 共享 GPU 同时跑多实例时存在 VRAM/算力争抢，B1 latency 数字受污染；最终 latency 待 Myriad HPC 上线后用独占 GPU 重跑。SR/cost/oracle 数字不受影响。
> - v1 (2026-04-24)：三模式完整数据首版，max_marks=80。

---

## 1. 成功率

### 1.1 主指标（Adjusted SR）

| 模式 | Raw SR | Adjusted SR | 成功数（adjusted） | FP 分解 |
|------|--------|------------|------------------|---------|
| **DOM** | **10.00%** (21/210) | **6.67%** (14/210) | 14 | N/A FP: 5, Eval FP: 2 |
| SoM | 8.10% (17/210) | 4.76% (10/210) | 10 | N/A FP: 5, Eval FP: 7 |
| Vision | 4.76% (10/210) | 1.43% (3/210) | 3 | N/A FP: 5, Eval FP: 2 |

**三模式排序：DOM (6.67%) > SoM (4.76%) > Vision (1.43%)**

B1 Reddit 是唯一 **DOM 领先 SoM** 的场景（Classifieds 中 SoM 始终最优）。**§94 验证（max_marks=200）**：v1 max_marks=80 时假设 reddit 列表元素超出标记上限导致 SoM 反垫底，重跑 max_marks=200 后反转**未消失反而加深**（v1 -0.98pp → v2 -1.91pp）。结论：标记数不是主导因素，4B 模型在 reddit 密集页面上无法从 SoM 视觉信息获益是结构性问题。

> 注：v2 起 cross_representation 对 reddit 用 /210 分母（不再扣除 N/A reference tasks），与 §95 canonical 一致。

### 1.2 FP 机制说明

**N/A FP**（15 个 reference tasks）：5 个 N/A task_id（7/26/31/39/182），三模式各 5 个 FP。分母从 210→205。

**Visual FP（DOM 2 个）**：DOM 无截图，2 个 visual task（task 36, 160）碰巧通过评测。SoM/Vision 有截图，无 visual FP。

**Eval FP**：DOM 2 个、SoM 3 个、Vision 2 个。program_html/string_match 类型 success 但 agent 未真正 finish。

### 1.3 统计显著性（McNemar 精确检验）

| 对比 | 不一致对 (A-only / B-only) | p 值 | 显著性 |
|------|--------------------------|------|--------|
| Vision vs SoM | 7 / 9 | 0.804 | — (n.s.) |
| Vision vs DOM | 8 / 12 | 0.503 | — (n.s.) |
| SoM vs DOM | 5 / 7 | 0.774 | — (n.s.) |

**三模式间差异均不显著**。B1 (4B) 在 Reddit 站点上能力极低，三模式差异被大量共同失败淹没。

### 1.4 Bootstrap 95% CI（v1 数据，post-rederive 待重跑）

| 模式 | SR (v2) | CI 下界 (v1) | CI 上界 (v1) |
|------|-----|---------|---------|
| DOM | 6.67% | 3.33% | 10.48% |
| SoM | 4.76% | 2.86% | 9.05% |
| Vision | 1.43% | 1.90% | 7.62% |

三模式 CI 大幅重叠，与 McNemar 全不显著一致。CI 数字基于 v1（max_marks=80），post-rederive 后 SoM/Vision SR 略低于 CI 下界，需重跑 bootstrap。

---

## 2. 效率指标

### 2.1 全 episode 效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| 平均成本 ($/ep) | 0.0536 | 0.0410 | **0.0137** |
| 平均步数 | 16.64 | 11.70 | **6.45** |
| P95 延迟 (ms) | 87,926 | **61,569** | 53,526 |

**Vision 成本最低**（$0.0137/ep），仅为 DOM 的 25.6%、SoM 的 33.4%。

**DOM 步数极高**（16.64）：AXTree 提供有效 action 候选，不易触发早停，agent 跑满 30 步的比例高。

**SoM 步数介于中间**（11.70）。

### 2.2 Wilcoxon 效率对比

| 对比 | 指标 | p 值 | 方向 |
|------|------|------|------|
| Vision vs SoM | total_cost | **1.7e-24** | Vision 更便宜 ★★★ |
| Vision vs SoM | p95_step_latency | 0.632 | **无显著差异** |
| Vision vs DOM | total_cost | **8.6e-29** | Vision 更便宜 ★★★ |
| Vision vs DOM | p95_step_latency | **2.2e-8** | Vision 延迟更低 ★★★ |
| SoM vs DOM | total_cost | **1.2e-3** | SoM 更便宜 ★★★ |
| SoM vs DOM | p95_step_latency | **1.0e-5** | SoM 延迟更低 ★★★ |

**成本排序**：Vision << SoM < DOM（均显著）。
**延迟排序**：Vision ≈ SoM < DOM（Vision 和 SoM 延迟无显著差异，两者均显著低于 DOM）。

### 2.3 成本分解

| 模式 | 总成本 | 有效成本 | No-op 成本 | 循环成本 |
|------|--------|---------|-----------|---------|
| DOM | $0.0536 | $0.0301 (56.2%) | $0.0074 (13.8%) | $0.0164 (30.6%) |
| SoM | $0.0410 | $0.0284 (69.3%) | $0.0084 (20.5%) | $0.0043 (10.5%) |
| Vision | $0.0137 | $0.0075 (54.7%) | $0.0055 (40.1%) | $0.0010 (7.3%) |

**DOM 循环成本占比最高（30.6%）**：search_repeat + click_back_loop 消耗大量步数。Vision no-op 成本占比最高（40.1%）：坐标 misclick 导致大量无效步骤。

---

## 3. 失败模式对比

### 3.1 三模式失败原因对比

| 失败原因 | DOM | SoM | Vision |
|----------|-----|-----|--------|
| fail_no_progress | 50 (23.8%) | 81 (38.6%) | **118 (56.2%)** |
| fail_max_steps_search_repeat | **48 (22.9%)** | 10 (4.8%) | 1 (0.5%) |
| fail_incomplete_or_stuck | 20 (9.5%) | **33 (15.7%)** | **48 (22.9%)** |
| fail_finish_eval_mismatch | **29 (13.8%)** | 17 (8.1%) | 2 (1.0%) |
| fail_max_steps_click_back_loop | **20 (9.5%)** | **13 (6.2%)** | 1 (0.5%) |
| fail_early_finish | 9 (4.3%) | **20 (9.5%)** | **22 (10.5%)** |
| fail_finish_empty_answer | 9 (4.3%) | 5 (2.4%) | 4 (1.9%) |
| fail_max_steps | 3 (1.4%) | **9 (4.3%)** | 2 (1.0%) |
| fail_finish_claim_missing | — | 4 (1.9%) | 1 (0.5%) |
| fail_finish_wrong_url_not_found | 1 (0.5%) | 1 (0.5%) | 1 (0.5%) |
| success | **21 (10.0%)** | 17 (8.1%) | 10 (4.8%) |

**三模式各有主导失败原因**：
- **DOM**：`search_repeat`（22.9%）+ `eval_mismatch`（13.8%）+ `click_back_loop`（9.5%）— AXTree 使 agent 持续产出语法正确 action，不触发早停，但陷入搜索/导航循环
- **SoM**：`no_progress`（38.6%）+ `incomplete_or_stuck`（15.7%）+ `click_back_loop`（6.2%）— SoM 标注 ID 幻觉导致 click 失败严重
- **Vision**：`no_progress`（56.2%）+ `incomplete_or_stuck`（22.9%）— 坐标 misclick 主导，79.1% 的失败属于执行失败类

### 3.2 脚手架 vs 模型归因

| 模式 | 脚手架/表征缺陷 | 模型能力问题 | 主要特征 |
|------|--------------|------------|---------|
| DOM | 低（AXTree 信息丰富） | 高（搜索循环 22.9%、答案错误 13.8%） | 循环陷阱 |
| SoM | 高（ID 幻觉严重） | 高（stuck 15.7%、loop 6.2%） | 执行 + 推理双重失败 |
| Vision | 低（截图可见） | 极高（no_progress 56.2%） | 坐标精度不足 |

---

## 4. 跨模式交叉分析

> 三模式完整 cross_representation 分析（DOM + SoM + Vision），210 tasks common set。
> 注：以下数据使用 cross_rep 管线的三重 adjusted labels（na_fp + visual_fp + eval_fp，/210 分母）。

### 4.1 Oracle 分析

| 指标 | Raw | Adjusted |
|------|-----|----------|
| Oracle ceiling | 12.38% (26 tasks) | **8.57%** (18 tasks) |
| 最佳单模式 (DOM) | 10.00% | 5.71% |
| Routing headroom | 2.38pp | **2.86pp** |
| Oracle 选择分布 | DOM:14, Vision:9, SoM:3 | **DOM:12, SoM:3, Vision:3** |

**Adjusted routing headroom 2.86pp**：理论最优路由可将 SR 从 5.71% 提升到 8.57%。Oracle 选择分布高度偏向 DOM（12/18 = 66.7%）——4B 模型在 Reddit 上，AXTree 文本信息比截图更有效。

### 4.2 集合分析（Adjusted，三模式）

| 集合 | 数量 | 占比 | 任务类型分布 |
|------|------|------|-------------|
| all_fail | 192 | 91.4% | single_nav:175, page_reading:14, action_on_item:3 |
| only_dom | 7 | 3.3% | single_nav:7 |
| dom_and_som (not vision) | 5 | 2.4% | single_nav:5 |
| only_som | 3 | 1.4% | single_nav:2, page_reading:1 |
| only_vision | 2 | 1.0% | single_nav:1, page_reading:1 |
| som_and_vision (not dom) | 1 | 0.5% | single_nav:1 |

**关键发现**：
- **all_fail 高达 91.4%** — B1 在 Reddit 上极度困难
- **DOM 独占 7 个 task**（3.3%），远高于 SoM（3 个）和 Vision（2 个）
- **无 all_success**（0 个 task 三模式全部成功）
- **DOM 主导 oracle 选择**（12/18），B1 在 Reddit 上 DOM 是最有价值的模式

### 4.3 B1 Reddit 的异常模式：DOM 领先

B1 Reddit 是所有 site×baseline 组合中**唯一 DOM adjusted SR 高于 SoM 的场景**。可能原因：

1. **Reddit 搜索重度**：210 tasks 中大量需要搜索/导航，DOM 的 AXTree 提供精确元素 ID（如搜索框、导航链接），4B 模型在 SoM 上频繁遭遇 ID 幻觉导致 click 失败
2. **SoM 在 Reddit 上 click_fail_rate 更高**：论坛页面链接密集，SoM 标注 ID 对 4B 模型的干扰更大
3. **DOM 步数多 = 探索更充分**：DOM 16.64 步 vs SoM 11.70 步，更多步数给了 DOM 更多找到目标的机会

---

## 5. 路由方向分析

### 5.1 Headroom 评估

| 路由场景 | Adjusted headroom | 备注 |
|---------|------------------|------|
| 三模式 oracle | **2.86pp** | Oracle 高度偏向 DOM（66.7%） |

### 5.2 各模式路由角色

**DOM（12/18 adjusted oracle, 66.7%）**— B1 Reddit 主力模式：
- Adjusted SR 最高（6.83%），独占 7 个 task
- 搜索循环虽严重（22.9%），但 AXTree 的精确元素交互在 Reddit 上仍是 4B 模型的最佳策略

**SoM（3/18 oracle, 16.7%）**— 补充角色：
- 独占 3 个 task，与 DOM 共享 5 个
- SoM 在 Reddit 上的 ID 幻觉问题比 Classifieds 更严重

**Vision（3/18 oracle, 16.7%）**— 低成本替代：
- 成本仅为 DOM 的 25.6%
- 独占 2 个 task，但 adjusted SR 极低（2.44%）

### 5.3 Pareto 分析

| 策略 | Adjusted SR | 平均成本 |
|------|-------------|---------|
| 全部 DOM | 6.83% | $0.0536 |
| 全部 SoM | 5.85% | $0.0410 |
| 全部 Vision | 2.44% | $0.0137 |
| Oracle 三模式 | **8.57%** | ~$0.045 |

---

## 6. 共性脚手架缺陷

### 6.1 Reddit Postmill 特有问题

**Comment 自链接死循环**：帖子页面的 "N comments" 链接指向当前页面自身，agent 不理解"已到达目标"，反复点击。B1 中 36/210 tasks（17.1%）存在此模式。

**Image link trap**（Vision）：帖子标题和缩略图链接指向原图而非讨论页，只有 "N comments" 小字链接通向帖子页面。Vision 模式受此影响更大。

**密集分类列表坐标偏移**（Vision）：`/forums/all` 多列字母序列表，行间距 ~20px < 4B Y 轴误差 ~40-60px，Vision 在此布局上命中率极低。

### 6.2 N/A 任务 False Positive

15 个 N/A reference tasks（5 个 task_id），三模式各 5 个 FP。

### 6.3 Search-over-Browse 偏差

B1 搜索倾向更强（68.8% vs B0 62.9%）。19 个 task 仅 B1 搜索而 B0 未搜索（反向仅 9 个），小模型更依赖搜索作为默认策略。

- **subreddit 起始 + 搜索**：0/25 (0.0%) — 搜索在已定位场景下有害
- **subreddit 起始 + 不搜索**：3/35 (8.6%)

### 6.4 搜索循环（DOM 特有）

DOM `search_repeat` 占 22.9%（48/210），是 B1 Reddit 最突出的单一失败模式。Agent 反复使用相同搜索词（如 "pumpkin robot" x15），AXTree 的搜索框持续可用使 agent 不触发早停。

---

## 7. 路由信号分析

### 7.1 AUROC 区分力（全局，630 episodes）

| 信号 | 类型 | AUROC | 95% CI | 显著性 |
|------|------|-------|--------|--------|
| **ep_mean_verbalized** | verbalized | **0.719** | [0.620, 0.804] | ★★★ |
| ep_min_verbalized | verbalized | 0.651 | [0.540, 0.751] | ★ |
| **max_repeat_streak** | behavioral | **0.629** | [0.541, 0.706] | ★ |
| url_revisit_max | behavioral | 0.575 | [0.466, 0.681] | — |
| url_revisit_count | behavioral | 0.566 | [0.455, 0.674] | — |
| ep_min_margin | token_level | 0.536 | [0.488, 0.600] | — |
| url_unique_count | behavioral | 0.528 | [0.429, 0.630] | — |
| action_diversity | behavioral | 0.521 | [0.413, 0.632] | — |
| ep_max_entropy | token_level | 0.518 | [0.393, 0.642] | — |
| ep_mean_logprob | token_level | 0.513 | [0.392, 0.625] | — |

**最强信号**：ep_mean_verbalized（AUROC=0.719），仍超过 0.6 路由阈值但低于 Classifieds（0.769）。

**Token-level 信号**：B1 有 token-level 信号但全部接近 0.5（无区分力）——4B 模型在 Reddit 上的 logprob/entropy 无法区分成功与失败。

**Behavioral 信号**：仅 max_repeat_streak（0.629）勉强可用，其余不可靠。

### 7.2 跨模式 AUROC 稳定性

| 信号 | DOM | SoM | Vision | 模式间差异 |
|------|-----|-----|--------|-----------|
| ep_mean_verbalized | **0.730** | 0.708 | 0.698 | 0.032 |
| max_repeat_streak | 0.540 | 0.576 | **0.725** | 0.185 |
| url_revisit_max | 0.612 | 0.528 | **0.857** | 0.329 |

- **ep_mean_verbalized 跨模式最稳定**（差异 0.032）
- **url_revisit_max 在 Vision 模式下极高**（0.857），但 DOM 仅 0.612 — 模式依赖性强，不适合跨模式路由
- B1 Reddit Vision 下的 behavioral 信号 AUROC 虽高，但基于极少成功样本（仅 10 raw / 5 adj），CI 极宽

### 7.3 Routing Readiness 判定

| 维度 | 结果 |
|------|------|
| Token-level 区分力 | ❌ AUROC ≈ 0.5（无用） |
| Verbalized 区分力 | ✅ AUROC=0.719 |
| Behavioral 区分力 | ⚠️ AUROC=0.629 (max_repeat_streak，勉强可用) |
| 信号覆盖率 | ✅ 100% |
| **Overall: 可用于路由** | **⚠️（仅 verbalized 可靠，behavioral 弱）** |

---

## 方法论说明

- **Adjusted labels**：N/A FP 优先 → visual FP → eval FP，分母 205（移除 5 个 N/A reference task）
- **McNemar / Bootstrap CI**：使用 condition_metrics 管线的 adjusted labels
- **统计检验**：McNemar 精确检验（SR）+ Wilcoxon 有符号秩检验（成本/延迟）+ Bootstrap CI
- **Token-level 信号**：B1 为本地推理，有 logprob/entropy/margin 但全部 AUROC ≈ 0.5
- **数据时间**：本报告基于 2026-04-24 三模式完整数据（此前版本 SoM 仅 3/210）
- ⚠️ C23 WARN：三模式均存在时序 SR 退化（dom: 17.1%→4.3%→8.6%, som: 14.3%→4.3%→5.7%, vision: 7.1%→2.9%→4.3%）

---

*更新时间：2026-04-24*
*数据来源：B1_3mode_reddit_20260413 analysis/ 目录（三模式全部 210/210 完整）*
*各模式详情：B1_DOM_digest.md / B1_SOM_digest.md / B1_Vision_digest.md*
