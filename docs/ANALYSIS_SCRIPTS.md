# P79 分析脚本能力说明

本文档描述项目中所有分析脚本的功能、输入输出和运行方式。

---

## 1. `analyze_experiment.py` — 单 Condition 基础分析

**路径**: `scripts/analysis/analyze_experiment.py` -> `p79.experiment.analysis.analyze_run()`

### 触发方式

- **自动**: `runner.py:_run_post_condition_analysis()` 在每个 condition 跑完后 subprocess 调用
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

### 输出

```
<run_dir>/analysis/
├── <condition_id>/
│   ├── tables/           # CSV: episode_metrics, step_metrics, cumulative_success_rate 等
│   ├── plots/            # PNG: 分布图、累计曲线
│   └── session_summary.json
├── _overview/            # 跨 condition 对比（McNemar、Wilcoxon、Pareto）
└── analysis_summary.json
```

---

## 2. `analyze_reason_diagnostics.py` — 失败原因深度诊断

**路径**: `scripts/analysis/analyze_reason_diagnostics.py`

### 触发方式

- **自动 (sidecar)**: `glm_diagnosis_sidecar.py` 每 N 个新 episode 轮询触发
- **自动 (queue)**: `queue_b1_serial.sh` 每站跑完后调用
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

## 3. `analyze_cross_representation.py` — 跨表征交叉分析

**路径**: `scripts/analysis/analyze_cross_representation.py`

### 定位

在 task 粒度对比不同观测模式（dom / som / vision）的表现差异。回答：
- Oracle router 理论天花板多高？routing headroom 有多少？
- 哪些 task 只在某个模式下成功？有什么共性？
- 同一 task 在不同模式下失败原因是否一致？
- Router 的特征线索在哪？

### 触发方式

- **自动**: `runner.py:_run_cross_representation_analysis()` 在每个 condition 跑完后检测同一站点是否已有 >=2 个 condition 有数据，满足则以 `--priority p0 --skip-plots` 运行
- **手动**: `python3 scripts/analysis/analyze_cross_representation.py --run-dir <run_dir> [--priority all]`

### 站点隔离

**绝不跨站交叉**。两层保证：

1. **Runner 层**: 从 episode 文件名解析站点，按 `site -> set(condition_ids)` 分组，仅同站点内有 >=2 个 condition 时才触发
2. **脚本层**: 加载数据后按 `site` 列分组，每个站点独立运行全部分析。单站时不加 site 子目录；多站时输出到 `<out>/<site>/`

### CLI

```bash
python3 scripts/analysis/analyze_cross_representation.py \
    --run-dir <run_dir>            # 必需
    [--reason-diag-dir <path>]     # 显式指定 reason diagnostics 目录或 CSV
    [--output-dir <path>]          # 默认 <run_dir>/analysis/cross_representation/
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
cross_representation/                    # 单站时直接输出
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

cross_representation/                    # 多站时按站点分子目录
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

## 4. `glm_diagnosis_sidecar.py` — 实时诊断 Sidecar

**路径**: `scripts/glm_diagnosis_sidecar.py`

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

## 5. `experiment_watchdog.py` — 实验健康监控

**路径**: `scripts/experiment_watchdog.py`

### 触发方式

- **外部启动**: `queue_b1_serial.sh` 中 `start_watchdog()` 启动

### 分析能力

| 能力 | 说明 |
|---|---|
| 滚动窗口指标 | success rate、wrong_url rate、no_progress rate、max_steps rate、avg step latency |
| 阈值告警 | 指标超阈值时 ntfy 推送 |
| Idle 检测 | N 分钟无新 episode 时告警 |

不运行任何分析脚本，纯监控/告警。

---

## 三脚本协作关系

```
实验运行中
  │
  ├── runner.py 每个 condition 跑完后
  │     ├── 调 analyze_experiment.py          → analysis/<cid>/{tables,plots}/
  │     └── 检测同站 >=2 conditions 后
  │           └── 调 analyze_cross_representation.py --priority p0
  │                                            → analysis/cross_representation/{tables,plots}/
  │
  ├── glm_diagnosis_sidecar.py 每 N 个 episode
  │     └── 调 analyze_reason_diagnostics.py  → analysis/reason_diagnostics_live/
  │
  └── experiment_watchdog.py 持续轮询         → ntfy 告警（不写文件）

实验跑完后（手动）
  │
  ├── analyze_reason_diagnostics.py 最终版    → analysis/reason_diagnostics/
  └── analyze_cross_representation.py --priority all
                                               → analysis/cross_representation/{tables,plots}/
```
