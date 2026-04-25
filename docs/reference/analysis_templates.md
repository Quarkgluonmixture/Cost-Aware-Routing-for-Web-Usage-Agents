# Analysis Document Templates

write-analysis SKILL Step 7 引用的文档模板。按需 Read 对应模板，不必全部加载。

---

## Mode Digest (`B{x}_{Mode}_digest.md`)

```markdown
# B{x} {Site} — {Mode} 模式分析

> 数据来源：`{run_id}` / `{condition_id}`
> Episodes: {n_episodes} | Adjusted SR: {adj_sr}% | Raw SR: {raw_sr}%
> 文档生成/更新：{date}

## 1. 总体概况

- Episode 总数、成功/失败分布
- Adjusted SR（三套 FP 修正：N/A FP ×n, Visual FP ×n, Eval FP ×n）
- Digest 覆盖率（如不足 100% 则标注）

## 2. 失败类别分布

（引用 condition_reason_summary.csv 的聚合数据）
- 按 reason 分类的 episode 数和占比
- 脚手架 vs 模型归因

## 3. 脚手架缺陷

（从 digest JSONL 提取定性案例 + 从 reason_diagnostics 提取统计）

## 4. 模型能力问题

（搜索循环、导航循环、推理错误、过早终止等）

## 5. 步数消耗模式

（引用 step_metrics.csv 或 episode_metrics.csv）

## 6. FP 分析

（引用 benchmark_noise/ 下的 CSV，标注各类 FP 对 SR 的影响）

## 7. 基础设施噪声

（评测器错误、JSONL 重启重复、auth 失效等，如有）

## 8. 方法论

数据来源、digest 工具、已知局限。
```

---

## Findings (`B{x}_findings.md`)

```markdown
# B{x} {Site} — 跨模式发现

> Baselines: {modes_list} | 数据来源：{run_id}
> 文档生成/更新：{date}

## 1. 成功率

- 主表：raw SR / adjusted SR（按 mode 分列）
- FP breakdown
- 显著性检验（McNemar p-values、Bootstrap CI）

## 2. 效率指标

- 成本、延迟、步数（按 mode 对比）

### 2.4 能耗分析

（如有 energy_kwh/co2e_kg 数据，引用 condition_metrics）

## 3. 失败模式对比

- 各 mode 特有的失败类别
- 信息可达性差异

### 3.X Reason Stability（跨模式失败一致性）

（来源：`reason_stability` 字段，引用 B2_reason_stability.csv）
- 同一 task 在不同模式下是否落入相同 failure bucket
- mean stability pct 和 per-bucket 拆分

## 4. 跨模式交叉分析

（引用 cross_representation/ 输出）
- Oracle analysis（任一模式成功的上界）
- 独占成功集分析
- Task type breakdowns

### 4.3 Task type × mode SR 矩阵

（来源：`task_type_mode_sr`，引用 A5_task_type_success_rate.csv）

### 4.4 Feature-based Oracle 分解

（来源：`oracle_decomposition`，引用 R3 按 task_type/eval_type 拆分）

## 5. 路由方向分析

- Headroom 估算
- 各 mode 角色定位

### 5.X 路由信号校准质量

（来源：`per_mode_calibration`，ECE/MCE per mode）

### 5.Y 按失败原因的成本分解

（来源：`fail_reason_cost_stats`，引用 A4b_fail_reason_cost_stats.csv）

### 5.Z Action 执行效率

（来源：`action_execution_stats`）
- click_fail_rate / type_fail_rate per mode
- pixel_coordinate_leak 比例
- max_consecutive_fail_streak 分布

## 6. 共性脚手架缺陷

- 跨 mode 共享的问题
```

---

## 跨 Baseline Findings (`B0_B1_findings.md`)

```markdown
# B0 vs B1 {Site} — 跨模型对比

> B0: {b0_run_id} | B1: {b1_run_id}
> 文档生成/更新：{date}

## 1. 核心对比表

（引用 compare_b0_b1.py 输出）

## 2. 各模式详细对比

（DOM / SoM / Vision 分节）

## 3. 行为差异

（模型级别的策略/能力差异）

## 4. Mirage Effect

（SoM-DOM gap 对比）

## 5. Action 执行效率对比

（B0 vs B1 click_fail_rate per mode，来源：`action_execution_stats`）

## 6. 方法论
```

---

## 跨站 Findings (`cross_sites/B{x}_cross_site_findings.md`)

```markdown
# B{x} 跨站汇总

> Sites: {sites_list} | Benchmarks: {benchmarks_list}
> 文档生成/更新：{date}

## 1. 跨站成功率对比

（引用 aggregate_cross_site.py 输出）

## 2. 站点难度排序

## 3. 模式 × 站点交互效应

## 4. 共性失败模式

## 5. 站点特异性问题
```
