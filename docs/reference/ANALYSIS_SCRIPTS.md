# P79 分析脚本能力说明

本文档描述项目中所有分析脚本的功能、输入输出和运行方式。

> **最后更新**: 2026-04-10（目录重组 + verbalized confidence + watchdog 管线集中化）

---

## 输出目录总览

```
<run_dir>/analysis/
├── analysis_summary.json              # 主实验汇总（analyze_experiment）
│
├── results/                           # 主实验分析结果
│   ├── <condition_id>/                #   per-condition 指标/图表
│   │   ├── tables/
│   │   ├── plots/
│   │   └── session_summary.json
│   ├── _overview/                     #   跨 condition 对比（McNemar、Wilcoxon、Pareto）
│   │   ├── tables/
│   │   ├── plots/
│   │   └── reports/
│   └── cross_representation/          #   跨表征交叉分析
│       ├── tables/
│       ├── plots/
│       └── cross_representation_summary.json
│
├── signals/                           # 信号分析（confidence + behavioral）
│   ├── combined/                      #   全模式聚合
│   │   ├── tables/
│   │   ├── plots/
│   │   └── confidence_summary.json
│   ├── dom/                           #   DOM 单模式
│   ├── som/                           #   SoM 单模式
│   └── vision/                        #   Vision 单模式
│
├── benchmark_noise/                   # 基准噪声过滤
│   ├── na_reference_tasks.csv
│   └── visual_lucky_hits.csv
│
├── reason_diagnostics/                # 失败原因诊断
│   └── episode_reason_rows.csv
│
└── digest/                            # GLM 批量消化
    └── digest.jsonl
```

---

## 1. `analyze_experiment.py` — 主实验分析

**路径**: `scripts/analysis/analyze_experiment.py` → `p79.experiment.analysis.analyze_run()`

### 触发方式

- **自动**: `runner.py:_run_post_condition_analysis()` 每个 condition 跑完后 subprocess 调用
- **自动**: `experiment_watchdog.py:_run_post_condition_analysis()` 每检测到新 condition 完成时调用
- **手动**: `python3 scripts/analysis/analyze_experiment.py --run_dir <run_dir>`

### 输入

- `<run_dir>/*/episodes/*_summary_v2.json` — episode 级汇总
- `<run_dir>/*/episodes/*_steps_v2.jsonl` — step 级记录
- `<run_dir>/*/condition_summary_v2.json` — condition 级聚合

### 分析能力

| 能力 | 说明 |
|---|---|
| 每 condition 指标汇总 | 成功率、平均步数、cost、latency、energy、no-op rate、page_unchanged_rate |
| 成功率累计曲线 | 按 episode 序号的滚动成功率 |
| Cost/Latency/Energy 分布 | 直方图 |
| No-op / Retry 分布 | 直方图 |
| State change reason 分布 | 各类页面变化原因的计数 |
| 跨 condition McNemar 检验 | 按 task_id 配对的成功率显著性检验（exact test） |
| 跨 condition Wilcoxon 检验 | 按 task_id 配对的 cost/latency 显著性检验 |
| Per-site 分层 | 按 benchmark_site 分组统计 |
| Pareto 前沿 | 成功率 vs cost 的 Pareto 分析 |
| NA reference tasks 过滤 | 识别并输出不可行参考任务 |
| Visual lucky hits 过滤 | 识别 DOM 模式视觉任务假阳性 |

### 输出

```
<run_dir>/analysis/
├── results/
│   ├── <condition_id>/
│   │   ├── tables/           # CSV: episode_metrics, step_metrics, cumulative_success_rate 等
│   │   ├── plots/            # PNG: 分布图、累计曲线
│   │   └── session_summary.json
│   └── _overview/            # 跨 condition 对比（McNemar、Wilcoxon、Pareto）
│       ├── tables/
│       ├── plots/
│       └── reports/
├── benchmark_noise/
│   ├── na_reference_tasks.csv
│   └── visual_lucky_hits.csv
└── analysis_summary.json
```

---

## 2. `analyze_confidence_calibration.py` — 信号分析（Token + Verbalized + Behavioral）

**路径**: `scripts/analysis/analyze_confidence_calibration.py`

### 定位

评估 routing 信号的判别力和校准度，为 Phase 2 routing go/no-go 决策提供量化依据。分析三类信号：

| 信号类型 | 来源 | 指标 |
|----------|------|------|
| **Token-level** | API logprobs | `ep_mean_logprob`, `ep_min_logprob`, `ep_mean_margin`, `ep_min_margin`, `ep_mean_entropy`, `ep_max_entropy`, `ep_prob` |
| **Verbalized** | Agent JSON 自我报告 | `ep_mean_verbalized`, `ep_min_verbalized` |
| **Behavioral** | Step 记录统计 | `url_revisit_count`, `url_revisit_max`, `action_diversity`, `max_repeat_streak` |

### 触发方式

- **自动**: `experiment_watchdog.py:_run_post_condition_analysis()` 每检测到新 condition 完成时调用（combined 模式）
- **手动**: `python3 scripts/analysis/analyze_confidence_calibration.py --run-dir <run_dir> [--mode dom|som|vision]`

### CLI

```bash
python3 scripts/analysis/analyze_confidence_calibration.py \
    --run-dir <run_dir>            # 必需
    [--output-dir <path>]          # 默认 <run_dir>/analysis/signals/<mode_name>
    [--mode dom|som|vision]        # 过滤单模式（默认 combined）
```

### 分析模块（C0—C9 + Routing Verdict）

#### C0 — Coverage Report

- 每 condition 的 token-level + verbalized 覆盖率统计
- **输出**: `tables/confidence_coverage.csv`
- 列: `episodes`, `episodes_with_confidence`, `episode_coverage`, `confidence_steps`, `step_coverage`, `episodes_with_verbalized`, `verbalized_episode_coverage`, `verbalized_steps`, `verbalized_step_coverage`

#### C1 — Success vs Failure Distribution

- 所有信号指标的 success/failure violin 对比 + Mann-Whitney U 检验 + rank-biserial 效应量
- 动态包含有足够数据（≥4 episodes）的指标（token-level + entropy + verbalized）
- **输出**: `tables/confidence_by_outcome.csv`, `tables/wilcoxon_test.csv`, `plots/C1_confidence_violin.png`

#### C2 — Reliability Diagram + Calibration Metrics

**Token-level（ep_prob）**:
- 10-bin reliability diagram + ECE/MCE/Brier/AUROC
- **输出**: `tables/calibration_bins.csv`, `tables/calibration_metrics.csv`, `plots/C2_reliability_diagram.png`

**Verbalized（ep_mean_verbalized）**:
- 独立 reliability diagram（verbalized 已在 [0,1] 范围，无需 exp 变换）
- **输出**: `tables/verbalized_calibration_bins.csv`, `tables/verbalized_calibration_metrics.csv`, `plots/C2_verbalized_reliability_diagram.png`

**全指标 AUROC 表**:
- 涵盖 token-level + verbalized + behavioral，每项标注 `signal_type`
- Entropy 取反（低 entropy = 高置信）
- **输出**: `tables/auroc_all_metrics.csv`

#### C3 — Per-Step Trajectory

三种轨迹图（按 success/failure 分组，含 ±1σ 带）：

| 轨迹 | Y 轴 | 输出 |
|------|------|------|
| Log-prob | `mean_logprob` | `plots/C3_confidence_trajectory.png` |
| Entropy | `mean_entropy` | `plots/C3_entropy_trajectory.png` |
| Verbalized | `verbalized` | `plots/C3_verbalized_trajectory.png` |

- Step×Confidence position heatmap: `tables/step_position_stats.csv`, `plots/C3_step_position_heatmap.png`

#### C4 — Per-Mode Comparison

- Per-mode summary 表（token-level + verbalized 统计）
- **Token-level**: violin + per-mode reliability diagram
- **Verbalized**: violin + per-mode reliability diagram
- 需要 ≥2 个 observation mode
- **输出**: `tables/per_mode_summary.csv`, `plots/C4_per_mode_violin.png`, `plots/C4_per_mode_reliability.png`, `plots/C4_per_mode_verbalized_violin.png`, `plots/C4_per_mode_verbalized_reliability.png`

#### C5 — Mode × Outcome Cross-Analysis

- 使用 `ep_mean_logprob` 进行 mode×outcome Kruskal-Wallis + pairwise Mann-Whitney
- 判定信号是否 mode-invariant（相同 outcome 在不同 mode 下 logprob 分布一致）
- 刻意只测 logprob（避免混淆 verbalized 可用性差异）
- **输出**: `tables/mode_outcome_cross.csv`, `tables/mode_outcome_tests.csv`, `plots/C5_mode_outcome_violin.png`, `plots/C5_mode_outcome_ridge.png`

#### C6 — Behavioral Signals

- Behavioral 信号的 success/failure violin + Mann-Whitney U
- AUROC bar chart：token-level（蓝）vs verbalized（紫）vs behavioral（橙），含 random baseline 线
- **输出**: `tables/behavioral_by_outcome.csv`, `tables/behavioral_wilcoxon.csv`, `plots/C6_behavioral_violin.png`, `plots/C6_auroc_comparison.png`

#### C7 — Cross-Mode AUROC Comparison

- Grouped bar chart：每个信号在不同 mode 下的 AUROC，含 token-level / verbalized / behavioral 分区线
- 需要 ≥2 个 observation mode
- **输出**: `tables/cross_mode_auroc.csv`, `plots/C7_cross_mode_auroc.png`

#### C8 — Behavioral Signal Accumulation

- 逐步 cutoff 累积 behavioral 信号的 AUROC，回答"第几步可以开始 route"
- 专注 behavioral（零成本信号），不含 token-level 或 verbalized
- **输出**: `tables/signal_accumulation.csv`, `plots/C8_signal_accumulation.png`

#### C9 — Token vs Verbalized Comparison

- Scatter plot：`ep_prob` vs `ep_mean_verbalized`，按 success/failure 着色 + y=x 参考线
- Spearman 相关（全体 + 分 outcome 子相关）
- AUROC 对比 bar chart：ep_prob vs ep_mean_verbalized vs ep_min_verbalized
- **输出**: `tables/token_vs_verbalized_corr.csv`, `tables/token_vs_verbalized_auroc.csv`, `plots/C9_token_vs_verbalized_scatter.png`, `plots/C9_token_vs_verbalized_auroc.png`

#### Routing Readiness Verdict

综合判定 routing 信号是否可用：

| 判据 | 阈值 | 来源 |
|------|------|------|
| `token_discriminative` | Wilcoxon p < 0.05 且 \|rank_biserial\| > 0.2 | C1 |
| `behavioral_discriminative` | best AUROC > 0.6 | C2 |
| `verbalized_discriminative` | best AUROC > 0.6 | C2 |
| `signal_calibrated` | ECE < 0.15 | C2 |
| `signal_sufficient_coverage` | max episode coverage > 50% | C0 |
| `signal_mode_invariant` | 同 outcome 跨 mode 无显著差异 | C5 |
| **`overall_usable`** | (token ∨ behavioral ∨ verbalized) ∧ coverage | 综合 |

- **输出**: `confidence_summary.json`（含所有上述字段 + coverage/calibration/discrimination/AUROC 详情）

### 输出目录结构

```
signals/<mode_name>/           # combined | dom | som | vision
├── tables/
│   ├── confidence_coverage.csv
│   ├── confidence_by_outcome.csv
│   ├── wilcoxon_test.csv
│   ├── calibration_bins.csv
│   ├── calibration_metrics.csv
│   ├── verbalized_calibration_bins.csv
│   ├── verbalized_calibration_metrics.csv
│   ├── auroc_all_metrics.csv
│   ├── step_position_stats.csv
│   ├── per_mode_summary.csv
│   ├── mode_outcome_cross.csv
│   ├── mode_outcome_tests.csv
│   ├── behavioral_by_outcome.csv
│   ├── behavioral_wilcoxon.csv
│   ├── cross_mode_auroc.csv
│   ├── signal_accumulation.csv
│   ├── token_vs_verbalized_corr.csv
│   └── token_vs_verbalized_auroc.csv
├── plots/
│   ├── C1_confidence_violin.png
│   ├── C2_reliability_diagram.png
│   ├── C2_verbalized_reliability_diagram.png
│   ├── C3_confidence_trajectory.png
│   ├── C3_entropy_trajectory.png
│   ├── C3_verbalized_trajectory.png
│   ├── C3_step_position_heatmap.png
│   ├── C4_per_mode_violin.png
│   ├── C4_per_mode_reliability.png
│   ├── C4_per_mode_verbalized_violin.png
│   ├── C4_per_mode_verbalized_reliability.png
│   ├── C5_mode_outcome_violin.png
│   ├── C5_mode_outcome_ridge.png
│   ├── C6_behavioral_violin.png
│   ├── C6_auroc_comparison.png
│   ├── C7_cross_mode_auroc.png
│   ├── C8_signal_accumulation.png
│   ├── C9_token_vs_verbalized_scatter.png
│   └── C9_token_vs_verbalized_auroc.png
└── confidence_summary.json
```

### 数据不足时的降级行为

| 条件 | 行为 |
|------|------|
| < 4 episodes with metric | 该指标从 violin/Wilcoxon 中排除 |
| < 4 episodes with ep_prob | C2 token reliability 跳过 |
| < 4 episodes with ep_mean_verbalized | C2 verbalized reliability 跳过 |
| < 10 steps with verbalized | C3 verbalized trajectory 跳过 |
| < 10 episodes with both token + verbalized | C9 全部跳过 |
| < 2 observation modes | C4/C5/C7 跳过 |
| 单 class（全成功或全失败） | AUROC 返回 NaN |
| 0 verbalized data（旧 run） | 所有 verbalized 相关分析优雅跳过 |

---

## 3. `analyze_reason_diagnostics.py` — 失败原因深度诊断

**路径**: `scripts/analysis/analyze_reason_diagnostics.py`

### 触发方式

- **自动 (sidecar)**: `glm_diagnosis_sidecar.py` 每 N 个新 episode 轮询触发
- **自动 (digest)**: `glm_batch_digest.py` 在消化前调用生成最新 CSV
- **手动**: `python3 scripts/analysis/analyze_reason_diagnostics.py --run-dir <run_dir> [--skip-similarity]`

### 输入

- `<run_dir>/*/episodes/*_summary_v2.json`
- `<run_dir>/*/episodes/*_steps_v2.jsonl`

### 分析能力

| 能力 | 说明 |
|---|---|
| Reason bucket 分类 | 将每个 episode 归入 18 种 reason bucket（success, fail_max_steps_target_unreachable, fail_incomplete_or_stuck 等） |
| Task type 分类 | 从 intent 文本识别 6 种 task type（grid_position, date_count, page_reading, collection, action_on_item, single_navigation） |
| Stuck subtype 分类 | 识别卡住的具体子类型（account_loop, target_unreachable, scroll_static, search_no_result 等） |
| Unreachable subtype 分类 | 目标不可达的具体原因 |
| Page unchanged 信号 | stuck_first_step、最长连续未变化步数及位置 |
| Loop 检测 | click_back 循环、搜索重复、action 序列分析 |
| Thought 多样性分析 | 思考内容的去重率、相邻相似度、重复模板（需 `--skip-similarity` 关闭） |
| Select 事件追踪 | 下拉操作的成功/失败统计 |
| 每 bucket 典型样本 | 每个 reason bucket 输出 N 个代表 episode 的详细信息 |

### 核心输出

```
<run_dir>/analysis/reason_diagnostics/
└── episode_reason_rows.csv    # 64 列，每行一个 episode，包含全部诊断字段
```

此 CSV 是 `analyze_cross_representation.py` 的主数据源。

---

## 4. `analyze_cross_representation.py` — 跨表征交叉分析

**路径**: `scripts/analysis/analyze_cross_representation.py`

### 定位

在 task 粒度对比不同观测模式（dom / som / vision）的表现差异。回答：
- Oracle router 理论天花板多高？routing headroom 有多少？
- 哪些 task 只在某个模式下成功？有什么共性？
- 同一 task 在不同模式下失败原因是否一致？
- Router 的特征线索在哪？

### 触发方式

- **自动**: `experiment_watchdog.py:_run_post_condition_analysis()` 检测到 ≥2 个 condition 完成时以 `--priority all` 运行
- **手动**: `python3 scripts/analysis/analyze_cross_representation.py --run-dir <run_dir> [--priority all]`

### 站点隔离

**绝不跨站交叉**。两层保证：

1. **Watchdog 层**: 检测 `condition_summary_v2.json` 存在的目录数 ≥2 才触发
2. **脚本层**: 加载数据后按 `site` 列分组，每个站点独立运行全部分析。单站时不加 site 子目录；多站时输出到 `<out>/<site>/`

### CLI

```bash
python3 scripts/analysis/analyze_cross_representation.py \
    --run-dir <run_dir>            # 必需
    [--reason-diag-dir <path>]     # 显式指定 reason diagnostics 目录或 CSV
    [--output-dir <path>]          # 默认 <run_dir>/analysis/results/cross_representation/
    [--skip-plots]                 # 跳过 PNG 生成
    [--priority p0|p1|p2|all]      # 分析范围，默认 p0
```

### Reason diagnostics 自动查找顺序

1. `--reason-diag-dir` 显式指定
2. `<run_dir>/analysis/reason_diagnostics/episode_reason_rows.csv`
3. `<run_dir>/analysis/reason_diagnostics_live/*/episode_reason_rows.csv`（多条件自动合并去重）
4. 自动调 `analyze_reason_diagnostics.py --skip-similarity` 生成

### 依赖

| 包 | 用途 | 是否必须 |
|---|---|---|
| pandas, numpy | 数据处理 | 必须 |
| matplotlib | 图表生成 | 可选（`--skip-plots` 可跳过） |
| matplotlib_venn | A6 Venn 图 | 可选（fallback 到 CSV 表格） |
| scipy | A4 Wilcoxon 配对检验 | 可选（缺失时跳过统计检验） |

### 输入数据流

```
episode_reason_rows.csv ─┐
  (64列: task_id, success,│  build_task_pivot()     ┌─ P0: A1, A2, A3
   reason_bucket,          │  pivot 到 (site,task_id)├─ P1: A4, A5, A6, B1, B2
   observation_mode 等)    ├─────────────────────────├─ P2: R1, R2, R3
                           │                         └─ cross_representation_summary.json
*_summary_v2.json ─────────┤  (cost/latency/tokens)
  (episode summaries)      │
                           │
task_configs/*.json ───────┘  (difficulty/eval_type/image)
```

### 核心数据结构：task pivot

从 `episode_reason_rows.csv` 按 `(site, task_id)` pivot，每个观测模式展开为列：

| 列 | 说明 |
|---|---|
| `site`, `task_id` | 主键 |
| `task_type`, `eval_type`, `task_intent` | 任务元信息 |
| `{mode}_success` | 该模式下是否成功（NaN = 该模式未测试该 task） |
| `{mode}_reason_bucket` | 失败原因分类 |
| `{mode}_steps` | 步数 |

---

### P0: 核心交叉对比（默认运行）

#### A1 — Task x Condition 结果矩阵

- **输出**: `tables/A1_task_result_matrix.csv`
- 完整的 pivot 表，每行一个 task，包含所有模式的 success / reason_bucket / steps
- 用途：后续分析的基础数据，也可直接人工浏览

#### A2 — 集合分析 + 双层 Oracle Ceiling

- **输出**: `A2_set_analysis_summary.json` + `tables/A2_set_analysis.csv`
- 核心指标：

| 指标 | 含义 |
|---|---|
| `per_mode_sr` | 各模式成功率（分母 = 全部 task） |
| `per_mode_sr_tested` | 各模式成功率（分母 = 该模式实际测试的 task） |
| `per_mode_n_tested` | 各模式实际测试了多少 task |
| `union_sr` | 至少一个模式成功的 task 比例 |
| `intersection_sr` | 所有模式都成功的 task 比例 |
| `perfect_oracle_ceiling` | = `union_sr`，理论不可达上限（需事后知道哪个模式成功） |
| `feature_oracle_ceiling` | 按 `(task_type, eval_type)` 分组，每组选 SR 最高的模式，加权汇总；近似简单规则 router 的天花板 |
| `perfect_headroom` | = `perfect_ceiling - best_single`，理论最大 routing 收益 |
| `feature_headroom` | = `feature_ceiling - best_single`，简单规则 router 的预期收益 |
| `feature_gap` | = `perfect_ceiling - feature_ceiling`，gap 小说明简单规则够用，gap 大说明需要更细粒度信号 |
| `feature_oracle_choices` | 每个 `(task_type, eval_type)` 分组的最佳模式及各模式 SR |

- CSV 在 pivot 上追加 `in_union`, `in_intersection` 布尔列

#### A3 — 排他集列表 + task type 分布

- **输出**: `tables/A3_exclusive_sets_summary.csv` + `tables/A3_exclusive_sets_detail.csv`
- 将所有 task 按成功模式组合分入互斥集合：
  - `only_dom` / `only_som` / `only_vision`: 仅某模式成功
  - `dom_and_som_not_vision`: 两模式成功但第三个失败
  - `all_success`: 所有模式都成功
  - `all_fail`: 所有模式都失败
- Summary 表：每个集的 count、占比、task_type 分布（JSON）
- Detail 表：每个 task 所属集及各模式 reason_bucket

---

### P1: 深度分析（`--priority p1` 或 `all`）

#### A4 — Cost-at-Success（交集 task 成本对比）

- **输出**: `tables/A4_cost_at_success.csv` + `A4_cost_at_success_summary.json`
- 过滤到**所有模式都成功**的 task（intersection set）
- Join episode summary 取各模式的 `total_cost_usd`, `total_latency_ms`, `total_tokens`, `total_energy_kwh`, `steps`
- 每 task 标注 `cheapest_mode`
- Summary：per-mode mean/median + cheapest_mode 分布
- 可选 Wilcoxon 配对检验（scipy 可用且样本 >= 5 时）

#### A5 — Task Type 分层成功率

- **输出**: `tables/A5_task_type_success_rate.csv` + `plots/A5_task_type_success_rate.png`
- 按 `task_type` 分组，计算各模式成功率（分母 = 该模式在该 type 中有数据的 task 数）
- Grouped bar chart 可视化

#### A6 — 成功集 Venn 图

- **输出**: `tables/A6_venn_table.csv` + `plots/A6_venn_diagram.png`
- 2-way（2 模式）或 3-way（3 模式）成功集 Venn
- `matplotlib_venn` 可用时画 Venn 图，否则 fallback 到纯数值表
- 需要至少 2 个模式有成功 task

#### B1 — Reason Bucket 转移矩阵

- **输出**: 每对模式 `tables/B1_transition_{a}_to_{b}.csv` + `plots/B1_transition_{a}_to_{b}.png`
- `pd.crosstab` 生成失败原因迁移矩阵，仅对两个模式都有数据的 task 计算
- Heatmap 标注每格数值
- 用途：看同一 task 在不同模式下失败原因是否一致（模型能力 vs 表征敏感）

#### B2 — Reason Stability Score

- **输出**: `tables/B2_reason_stability.csv` + `plots/B2_reason_stability_histogram.png`
- 公式：`stability = 1 - (n_unique_buckets - 1) / (n_modes_present - 1)`

| 情况 | stability |
|---|---|
| 所有模式同一 bucket | 1.0 |
| 三模式中两同一异 | 0.5 |
| 所有模式不同 bucket | 0.0 |
| 只有 1 个模式有数据 | NaN（不参与统计） |

- score 接近 1 → 失败原因跨模式稳定（模型能力问题，routing 帮不了）
- score 接近 0 → 表征敏感（routing 有价值）
- 直方图显示分布 + 均值线

---

### P2: Router 设计支持（`--priority p2` 或 `all`）

#### R1 — Task Feature Extraction

- **输出**: `tables/R1_task_features.csv`
- 从三个来源提取 per-task 特征：

| 来源 | 特征 |
|---|---|
| intent 文本 | `has_color_word`, `has_visual_description`, `has_numeric_comparison`, `has_navigation_verb`, `intent_length` |
| task config JSON | `eval_type`, `visual_difficulty`, `reasoning_difficulty`, `overall_difficulty`, `has_image` |
| episode 结果 | `succeeded_in_modes`, `n_modes_succeeded`, `best_mode` |

- 用途：为 router 提供候选特征，后续可做特征-模式关联分析

#### R2 — Step-Level Escalation Signal + Counterfactual

- **输出**: `tables/R2_escalation_signals.csv` + `plots/R2_divergence_step_distribution.png`
- 复用 `stuck_first_step` 字段作为 divergence step（agent 开始卡住的步号）
- **Counterfactual**: 对每个有 divergence step 的失败 episode，检查同一 task 在其他模式下是否成功
  - `escalation_would_help = True` → 如果在 divergence step 切换到另一模式，理论上能成功
  - 总体比率 = step-level router 的理论收益率上限
- Divergence step 分布直方图按模式分层

#### R3 — Oracle Router Decomposition

- **输出**: `tables/R3_oracle_decomposition.csv` + `R3_oracle_decomposition.json`
- 对 union set 中每个成功 task：
  - 多模式都成功时，选 `total_cost_usd` 最低的模式（fallback by steps）
  - 只有一个模式成功时，选该模式
- 交叉 task features（task_type、difficulty）分析 oracle 选择的模式分布
- 用途：理解"理想 router 会怎么选"以及选择是否有可学习的模式

---

### 输出目录结构

```
results/cross_representation/           # 单站时直接输出
├── tables/                              # CSV 数据表
│   ├── A1_task_result_matrix.csv        # P0
│   ├── A2_set_analysis.csv              # P0
│   ├── A3_exclusive_sets_summary.csv    # P0
│   ├── A3_exclusive_sets_detail.csv     # P0
│   ├── A4_cost_at_success.csv           # P1
│   ├── A5_task_type_success_rate.csv    # P1
│   ├── A6_venn_table.csv                # P1
│   ├── B1_transition_{a}_to_{b}.csv     # P1
│   ├── B2_reason_stability.csv          # P1
│   ├── R1_task_features.csv             # P2
│   ├── R2_escalation_signals.csv        # P2
│   └── R3_oracle_decomposition.csv      # P2
├── plots/                               # PNG 可视化
│   ├── A5_task_type_success_rate.png    # P1
│   ├── A6_venn_diagram.png              # P1
│   ├── B1_transition_{a}_to_{b}.png     # P1
│   ├── B2_reason_stability_histogram.png# P1
│   └── R2_divergence_step_distribution.png # P2
├── A2_set_analysis_summary.json         # P0
├── A4_cost_at_success_summary.json      # P1
├── R3_oracle_decomposition.json         # P2
└── cross_representation_summary.json    # 总汇总

results/cross_representation/           # 多站时按站点分子目录
├── classifieds/
│   ├── tables/ ...
│   ├── plots/ ...
│   └── *.json
├── shopping/
│   ├── tables/ ...
│   ├── plots/ ...
│   └── *.json
└── cross_representation_summary.json    # 全局汇总（含 per_site 摘要）
```

### Stale Output Cleanup

脚本在 `OutputDirs.ensure()` 中自动清理 `tables/*.csv`、`plots/*.png`、`*.json`，防止旧 run 的残留文件干扰。

---

### Edge Cases

| 场景 | 行为 |
|---|---|
| 只有 2 个模式（如 vision 未跑） | Venn 退化为 2-way，其余功能正常 |
| 某站点只有 1 个模式 | 该站点跳过分析（print WARN），不输出文件 |
| 部分 task 缺模式 | pivot 中对应列为 NaN，不计入该模式的成功/失败集 |
| `per_mode_sr` vs `per_mode_sr_tested` | 前者分母为全部 task 数，后者分母为该模式实际测试的 task 数 |
| Multi-seed | 当前单 seed；多 seed 时取 `(site, task_id, seed)` 分组需手动扩展 |
| 无 episode summary JSON | A4 / R3 跳过（cost 数据不可用） |
| 无 task_configs 目录 | R1 的 config 特征列为空，intent 特征正常 |

---

## 5. `glm_diagnosis_sidecar.py` — 实时诊断 Sidecar

**路径**: `scripts/maintenance/glm_diagnosis_sidecar.py`

### 触发方式

- **外部启动**: `queue_b1_serial.sh` 中 `start_live_reason_watch()` 以 `setsid nohup` 启动，实验结束后 kill
- 不由 `runner.py` 管理

### 工作方式

- 长驻轮询进程，每 `--poll-secs`（默认 60s）扫描 run 目录
- 每积累 `--interval-episodes`（默认 5）个新 episode 后触发一轮分析：
  1. 调 `analyze_reason_diagnostics.py` 生成增量 `episode_reason_rows.csv`
  2. 调 GLM API 生成高层失败模式结论 + 逐 episode 失败归因
  3. 通过 ntfy.sh 推送通知
- 使用 `*.state.json` 持久化进度，支持重启续跑
- 使用 flock 防止重复实例

### 分析能力

| 能力 | 说明 |
|---|---|
| 增量 reason diagnostics | 每 N 个 episode 生成一次失败原因 CSV |
| GLM 高层结论 | LLM 总结当前失败模式趋势 |
| GLM 逐 episode 归因 | 对新增失败 episode 逐个做 LLM 归因分析 |
| 失败重试队列 | GLM API 失败时自动重试（5 分钟间隔） |
| ntfy 推送 | 实时推送诊断结果到手机/桌面 |

---

## 6. `glm_batch_digest.py` — 批量失败 Episode 预消化

**路径**: `scripts/maintenance/glm_batch_digest.py`

### 定位

用 GLM 视觉模型将失败 episode 的截图 + step 日志压缩为结构化 JSONL digest，供 Claude 快速归因（无需再看截图）。

### 触发方式

- **自动 (sidecar)**: `glm_diagnosis_sidecar.py` 在每次诊断后自动运行（需 `--digest-output` 参数）
- **手动**: `python3 scripts/maintenance/glm_batch_digest.py --run-dir <run_dir> --output <path> --glm-config .auth/glm`

### CLI

```bash
python3 scripts/maintenance/glm_batch_digest.py \
    --run-dir results/.../B1_3mode_classifieds_20260404_141103 \
    --output analysis/digest.jsonl \
    --glm-config .auth/glm \
    --condition phase1_dom_router_0 \  # 可选，过滤特定 condition
    --delay-secs 2 \                   # GLM 调用间隔
    --max-images 3 \                   # 每 episode 最多截图数
    --max-cases 0 \                    # 0=不限
    --site classifieds \               # 可选，覆盖 site 字段
    --dry-run                          # 不调 GLM，仅用确定性 fallback
```

### Sidecar 集成

在 `glm_diagnosis_sidecar.py` 启动时加入 `--digest-output` 即可激活自动增量消化：

```bash
python3 scripts/maintenance/glm_diagnosis_sidecar.py \
    ... \
    --digest-output analysis/digest/digest.jsonl \
    --digest-max-images 3 \
    --digest-delay-secs 1.0
```

流程：每次 sidecar 触发诊断 → episode diagnosis → **batch digest ALL 新失败 case** → ntfy 推送。

digest 与 episode diagnosis 的区别：
- episode diagnosis 受 `--episode-diagnosis-max-cases` 限制（默认 5），输出到 ntfy
- batch digest 处理**所有**新增失败 case，输出到 JSONL 文件，供后续离线分析

### 分析能力

| 能力 | 说明 |
|---|---|
| 关键步骤选取 | 自动选 step 0 / stuck 步 / 中间步 / 最后步（最多 max_images 张） |
| 截图描述 | GLM 视觉模型对关键步骤截图生成 30-50 字描述 |
| 思维链压缩 | 全程 thought 压缩为 2-3 句话摘要 |
| 动作序列压缩 | 完整 action 序列压缩为 ≤15 个语义块 |
| 失败分类 | category / root_cause / is_scaffolding_issue / evidence |
| 断点续跑 | 通过 `(condition_id, task_id)` 去重，自动跳过已消化 |
| 视觉模型 fallback | GLM-5V-Turbo → GLM-4.6V 自动降级 |

### 输出格式

每行一个 JSON 对象（JSONL），核心字段：

```json
{
  "task_id": 123,
  "condition_id": "phase1_dom_router_0",
  "task_intent": "...",
  "observation_mode": "dom",
  "reason_bucket": "fail_incomplete_or_stuck",
  "screenshot_descriptions": {"0": "...", "15": "..."},
  "thought_summary": "2-3句话压缩",
  "key_actions_compressed": "SEARCH(...)→CLICK(...)→BACK→FINISH",
  "category": "搜索循环",
  "root_cause": "≤60字根因",
  "is_scaffolding_issue": "否",
  "confidence": "high",
  "evidence": "具体证据"
}
```

### Hot-restart 注意

`scripts/maintenance/restart_watchdog.sh` 从 `/proc/PID/cmdline` 读取原始参数原样重启（也兼容 legacy `monitor_glm_sidecar.py` 进程）。`--digest-output` 等参数只需在首次启动时传入，后续 hot-restart 自动保留。

也可通过 `--append-args` 在 hot-restart 时注入新参数：

```bash
bash scripts/maintenance/restart_watchdog.sh --append-args "--digest-output analysis/digest.jsonl"
```

---

## 7. `experiment_watchdog.py` — 实验健康监控 + 自动分析

**路径**: `scripts/maintenance/experiment_watchdog.py`

### 触发方式

- **外部启动**: `queue_b1_serial.sh` 中 `start_watchdog()` 启动

### 监控能力

| 能力 | 说明 |
|---|---|
| 滚动窗口指标 | success rate、wrong_url rate、no_progress rate、max_steps rate、avg step latency |
| 阈值告警 | 指标超阈值时 ntfy 推送 |
| Idle 检测 | N 分钟无新 episode 时告警 |
| 分析状态追踪 | 检查 `_ANALYSIS_MARKERS` 判断哪些分析已完成 |

### 自动分析管线

Watchdog 在检测到新 condition 完成时，通过 `_run_post_condition_analysis()` 顺序调用三个分析脚本：

| 顺序 | 脚本 | 参数 | 条件 |
|------|------|------|------|
| 1 | `analyze_experiment.py` | `--run_dir` | 总是运行 |
| 2 | `analyze_confidence_calibration.py` | `--run-dir`（combined 模式） | 总是运行 |
| 3 | `analyze_cross_representation.py` | `--run-dir --priority all` | ≥2 个 condition 完成时才运行 |

每个脚本独立 try/catch（timeout 300s），返回 `experiment:ok; confidence:ok; cross_rep:ok` 格式的状态字符串，ntfy 推送完成通知。

### Analysis Markers

```python
_ANALYSIS_MARKERS = {
    "condition_analysis":      "analysis/results/_overview/tables/condition_metrics.csv",
    "reason_diagnostics":      "analysis/reason_diagnostics/reason_diagnostics_summary.json",
    "cross_representation":    "analysis/results/cross_representation/cross_representation_summary.json",
    "confidence_calibration":  "analysis/signals/combined/confidence_summary.json",
}
```

---

## 多脚本协作关系

```
实验运行中
  │
  ├── runner.py 每个 condition 跑完后
  │     └── 调 analyze_experiment.py           → analysis/results/<cid>/{tables,plots}/
  │
  ├── experiment_watchdog.py 检测到新 condition 完成后
  │     ├── 调 analyze_experiment.py           → analysis/results/_overview/
  │     ├── 调 analyze_confidence_calibration  → analysis/signals/combined/
  │     └── 调 analyze_cross_representation    → analysis/results/cross_representation/
  │         （需 >=2 conditions，--priority all）
  │
  ├── glm_diagnosis_sidecar.py 每 N 个 episode
  │     ├── 调 analyze_reason_diagnostics.py   → analysis/reason_diagnostics_live/
  │     ├── GLM episode diagnosis              → ntfy 推送
  │     └── glm_batch_digest (自动)            → analysis/digest/digest.jsonl
  │
  └── experiment_watchdog.py 持续轮询          → ntfy 告警

实验跑完后（手动）
  │
  ├── analyze_reason_diagnostics.py 最终版     → analysis/reason_diagnostics/
  ├── glm_batch_digest.py 最终版               → analysis/digest/digest.jsonl
  ├── analyze_confidence_calibration.py        → analysis/signals/{combined,dom,som,vision}/
  │     （手动可跑 --mode dom/som/vision 单模式）
  └── analyze_cross_representation.py          → analysis/results/cross_representation/
        （手动 --priority all 获取完整分析）
```
