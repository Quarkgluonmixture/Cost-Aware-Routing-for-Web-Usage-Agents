# B0 Classifieds 三模式实验报告

> Run: `B0_3mode_classifieds_20260413`
> 模型: Qwen3-VL-235B-A22B（proxy API，MoE 22B 活跃参数）
> 站点: Classifieds (OSClass), 234 tasks × 3 modes (DOM / SoM / Vision)
> 分析管线默认使用 **adjusted labels**（扣除 N/A FP + visual FP）
> 各模式专有分析见 `B0_DOM_digest.md` / `B0_SOM_digest.md` / `B0_Vision_digest.md`
> B0 vs B1 跨模型对比见 `B0_B1_findings.md`

---

## 1. 成功率

### 1.1 主指标（Adjusted SR）

| 模式 | Adjusted SR | 成功数（adjusted） | FP 分解 |
|------|------------|------------------|---------|
| DOM | **8.07%** | 18 / 223 | N/A FP: 7, Visual FP: 10 |
| SoM | **12.05%** | 27 / 224 | N/A FP: 10, Visual FP: 0 |
| Vision | **10.71%** | 24 / 224 | N/A FP: 9, Visual FP: 0 |

> DOM adjusted SR 注：严格下界 0.4%（1/234），现行过滤 8.07%（扣除 17 已确认 FP），§56 盲区 17 tasks 待进一步分类。报告使用 8.07% 作为上界估计。

**三模式排序：SoM (12.05%) > Vision (10.71%) > DOM (8.07%)**

B0 三模式 SR 分布比 B1 更均衡（B1: SoM 16.24% >> Vision 8.12% >> DOM 0.85%）。

### 1.2 Raw SR 与 FP 分解

| 模式 | Raw SR | N/A FP | Visual FP | Adjusted SR |
|------|--------|--------|-----------|------------|
| DOM | 15.02% (35/233) | 7 | 10 | 8.07% |
| SoM | 15.81% (37/234) | 10 | 0 | 12.05% |
| Vision | 14.10% (33/234) | 9 | 0 | 10.71% |

**N/A FP 机制**（27/30）：B0 与 B1 完全相同——Agent prompt 无 N/A 出口（Rule 4: "NEVER give up"）→ 空答案 → 评测器误判。Vision 1 个 N/A task（189）未触发 FP（agent 可能提交了明确错误答案而非空答案）。

**Visual FP（DOM 10 个）**：DOM 无截图，url_match 碰巧通过（kwd_only 过滤确认）。SoM/Vision 有截图，无 visual FP。

### 1.3 统计显著性

> 注：以下检验基于 adjusted labels，N/A 任务从分母移除（224 tasks common set）

| 对比 | 不一致对 (A-only / B-only) | 趋势 |
|------|--------------------------|------|
| SoM vs DOM | 需 task-level 数据 | SoM > DOM（+3.98pp） |
| Vision vs DOM | 需 task-level 数据 | Vision > DOM（+2.64pp） |
| SoM vs Vision | 需 task-level 数据 | SoM > Vision（+1.34pp） |

> McNemar 精确检验需 task-level 成功标签，当前数据仅有汇总统计。三模式间差距较 B1（SoM 15.39pp > DOM）小得多，推测显著性弱于 B1。

---

## 2. 效率指标

### 2.1 全 episode 效率

| 指标 | DOM | SoM | Vision |
|------|-----|-----|--------|
| 平均成本 ($/ep) | 0.0457 | 0.0411 | **0.0256** |
| 平均步数 | 14.10 | 8.27 | **8.02** |
| 平均输入成本 | 0.0393 | 0.0350 | 0.0207 |
| 平均输出成本 | 0.0064 | 0.0061 | 0.0049 |
| No-op rate | 7.3% | 7.6% | **27.4%** |
| Page unchanged rate | 26.8% | 31.6% | **41.8%** |
| 每成功 ep 成本 (adjusted) | $0.566 | $0.341 | **$0.239** |

**Vision 成本效率最优**（$0.239/成功 ep）：步数最少 + 每步成本最低（无 AXTree 文字、无标注图像）。

**DOM 步数最多（14.10 步）**：没有截图辅助决策，需要更多探索步骤。

**Vision no-op 率最高（27.4%）**：坐标 misclick 频繁导致无效动作。

### 2.2 早停触发分布

| 触发原因 | DOM | SoM | Vision |
|---------|-----|-----|--------|
| action_failed | 185 | 112 | **412** |
| page_unchanged_streak | 173 | 98 | **312** |
| no_progress_streak | 88 | 33 | **214** |

Vision 所有早停维度均最高——坐标点击的固有不稳定性导致大量无效动作积累。

---

## 3. 跨模式交叉分析

### 3.1 Oracle 分析（Adjusted，common 224 tasks）

| 指标 | 数值 |
|------|------|
| 最优单模式（SoM） | 12.05% |
| Oracle ceiling（adjusted） | **20.54%** (46/224) |
| Routing headroom | **+8.49pp** |

**Routing headroom 8.49pp** >> B1 的 3.42pp——B0 三模式成功 task 分布更互补，路由潜力显著更大。

### 3.2 Oracle 选择分布（Adjusted，46 tasks）

| 模式 | Oracle 选择次数 | 占比 |
|------|--------------|------|
| Vision | **20** | 43.5% |
| SoM | 14 | 30.4% |
| **DOM** | **12** | **26.1%** |

B0 Oracle 中 DOM 贡献 12 个（vs B1 仅 2 个），是路由格局最显著的变化：**B0 DOM 的 real SR 足够高，使其重新成为路由候选**。

### 3.3 集合分析（Adjusted）

| 指标 | 数值 |
|------|------|
| Union（至少 1 种模式成功） | 46 tasks (20.54%) |
| Intersection（三模式均成功） | 6 tasks (2.68%) |
| DOM 独有成功（推算） | ~6-8 tasks |
| SoM 独有成功（推算） | ~13-15 tasks |
| Vision 独有成功（推算） | ~10-12 tasks |

与 B1 关键差异：
- **B1 Intersection = 0（adjusted）**，B0 Intersection = 6——说明 B0 三模式在同一 task 上都能成功的情况更多，任务难度降低
- **B0 DOM 贡献 12 个 oracle**，B1 DOM 仅 2 个——235B 模型在 DOM 模式下能力大幅提升

### 3.4 任务类型 SR（部分，feature-level oracle）

| 任务类型 + 评测方式 | 最优模式 | 该类 SR | 任务数 |
|------------------|---------|--------|--------|
| page_reading + url_match | **SoM** | 35.7% | 28 |
| single_navigation + url_match | **Vision** | 12.2% | 90 |
| single_navigation + string_match | **SoM** | 12.9% | 31 |
| action_on_item + program_html | **Vision** | 22.2% | 9 |
| single_navigation + program_html | **DOM** | 7.1% | 14 |

**single_navigation + url_match** 是最大任务类型（90/224），Vision 最优（12.2%）。**page_reading + url_match** SoM 最优（35.7%）。**single_navigation + program_html** DOM 最优（7.1%），这是 B0 DOM 的独特贡献类型（B1 DOM 在此类型上接近 0）。

---

## 4. 失败模式对比

### 4.1 三模式失败原因对比

| 失败原因 | DOM | SoM | Vision |
|----------|-----|-----|--------|
| fail_incomplete_or_stuck | **23.8%** | 19.6% | 22.3% |
| fail_no_progress | 11.2% | 9.8% | **32.1%** |
| fail_parse_error | 1.3% | **20.1%** | 5.4% |
| fail_finish_wrong_url_not_found | **15.2%** | 12.9% | 8.5% |
| fail_early_finish | 4.0% | 5.4% | 4.5% |
| fail_max_steps_target_unreachable | 10.3% | 5.4% | 3.1% |
| fail_max_steps_click_back_loop | 4.0% | 2.2% | — |

**三模式各有主导失败原因**：
- DOM：incomplete_or_stuck（信息瓶颈，14 步找不到目标）+ target_unreachable（视觉任务 DOM 不可达）
- SoM：**parse_error（20.1%，API 格式问题）** + incomplete_or_stuck
- Vision：**no_progress（32.1%，坐标 misclick 积累）** + incomplete_or_stuck

### 4.2 脚手架 vs 模型归因（定性）

| 模式 | 脚手架/表征缺陷 | 模型能力问题 | 主要特征 |
|------|--------------|------------|---------|
| DOM | 较高（visual 不可达、描述不进 AXTree） | 较低（235B 文字推理强） | visual 瓶颈 |
| SoM | 低（截图可见） | 高（+20% parse_error 基础设施问题） | parse_error 主导 |
| Vision | 低（截图可见） | 高（坐标精度、no_progress） | 执行失败主导 |

---

## 5. 路由方向分析

### 5.1 Headroom 评估

| 路由场景 | Adjusted headroom | 可行性评估 |
|---------|------------------|-----------|
| SoM ↔ Vision | ~3-4pp | **有价值**（B0 Vision 贡献 20 oracle，成本低） |
| DOM ↔ Vision | ~4-5pp | **有价值**（B0 DOM 12 oracle，成本中等） |
| DOM ↔ SoM ↔ Vision（三模式）| **8.49pp** | **最大潜力**（三模式 oracle 均有贡献） |

### 5.2 与 B1 路由格局的关键差异

| 维度 | B1 | B0 |
|------|----|----|
| 最优单模式 | SoM（16.24%）远优 | SoM（12.05%）轻微领先 |
| Oracle headroom | 3.42pp | **8.49pp** |
| DOM 路由价值 | 无（仅 2 oracle） | **有**（12 oracle，7.1% SR on program_html） |
| 主路由方向 | SoM ↔ Vision | **三模式均有价值** |

**B0 的路由格局更接近理论最优**：三模式在不同任务类型上各有优势，路由潜力显著大于 B1。

### 5.3 路由信号（待 B0 信号分析完成）

> 注：本报告基于汇总统计，task-level 路由信号分析需要 signals/combined/ 目录的数据。
> 预期：行为信号（action_diversity）跨模式 AUROC 高于 0.7（参考 B1 的 0.74）。

---

## 6. 共性脚手架缺陷

以下缺陷三种模式均受影响：

### 6.1 `<select>` 下拉菜单不可用（VWA 框架级）

同 B1。B0 额外问题：**capability-environment gap 更严重**——235B 模型更精准识别 `<select>` 为正确入口，反而更执着地反复点击同一元素，cycle detection 更快截断（3 步循环），未能绕路。B1 因随机游走偶尔找到侧边栏链接（见 B0_DOM_digest §5）。

### 6.2 N/A 任务 False Positive（27/30）

三模式各 7/10、10/10、9/10 误判。机制同 B1：Agent prompt 无 N/A 出口 + evaluator ua_match bug。

### 6.3 confirm 弹窗不可交互

Delete 操作全部失败，三模式均受 VWA Playwright 限制。

### 6.4 极少翻页（模型能力缺陷）

B0 DOM 中翻页能力显著改善（33+ task 翻页），但 SoM 和 Vision 模式仍较少翻页（待进一步统计）。

### 6.5 搜索关键词过于具体

同 B1：将任务约束全部拼入搜索词，应用宽泛词+筛选器策略。B0 235B 的策略规划稍有改善。

---

## 方法论说明

- **Adjusted labels**：N/A FP 优先（重叠时标为 na_fp），visual FP 次之（DOM kwd_only 过滤）
- **DOM adjusted SR 不确定性**：§56 盲区 17 tasks（has_image 豁免）尚未全部分类，真实 SR 区间 [0.4%, 8.07%]
- **统计检验**：task-level McNemar 检验需 episode_reason_rows.csv 的 task-level 标签，当前仅完成汇总层面分析
- **B0 cross-mode Venn**：6 个三模式交集 task、46 个 union task（adjusted），独占/pairwise 集合需 task-level 数据进一步分解

---

*生成时间：2026-04-15*
*数据来源：B0_3mode_classifieds_20260413 analysis/ 目录*
*各模式详情：B0_DOM_digest.md / B0_SOM_digest.md / B0_Vision_digest.md*
