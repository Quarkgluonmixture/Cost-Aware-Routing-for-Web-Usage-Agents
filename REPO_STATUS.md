# Repository Status

**Last Updated:** 2026-03-21  
**Project:** Cost-Aware Routing for Web Usage Agents (P79)

## P79 对齐完成度面板
以 `P79_experimental_scope_rq_variables.md` 为基线，当前状态（估算）：

1. 实验内核与变量层级（Phase1/2/3, SoM, router, M1-M4）：**90%**
2. RQ2/RQ3 开销核算与可解释性：**82%**
3. 日志与分析闭环（trigger/reason/checklist）：**85%**
4. 安装依赖、Docker 实践、一键运行：**75%**
5. 文档与发布就绪度：**80%**

## 当前主线（v2）
### 已完成
- 统一实验入口：`run-experiment` / `analyze-experiment`（CLI + scripts wrapper）。
- 条件矩阵自动生成：
  - Phase1: 2x2 SoM/observation screening
  - Phase2: `phase2_fixed_best` vs `phase2_routed`
  - Phase3: `phase3_none/m1/m2/m3/m4` 单模块消融
- SoM 接入与降级策略：无 bbox 时降级文本 SoM，记录 `degraded_som=true`。
- Router 开销拆账：decision/dom parse/screenshot/extra model/retry。
- 成本口径修复：
  - `total_model_cost_usd`
  - `total_router_overhead_cost_usd`
  - `total_cost_usd = model + overhead`
  - Phase2 `NetSaving` 计算避免 double-count overhead。
- Step schema v2 + 可解释字段：`page_change_reasons`、`text_similarity`、`checklist`、`state_digest`。
- 错误分类标准化：`parse_error` / `invalid_action` / `no_progress` / `env_error` / `benchmark_noise`。
- 分析产物增强：
  - `phase1_representation_screening.csv/png`
  - `phase2_pareto_metrics.csv/png`
  - `phase2_net_saving_decomposition.csv/json`
  - `phase3_module_ablation.csv/png`
  - `phase3_module_gain_vs_base.csv`
  - `trigger_distribution.csv/png`
  - `state_change_reason_distribution.csv/png`
  - `checklist_progress_curve.csv/png`
  - `checklist_failure_distribution.csv/png`
  - `benchmark_noise_report.csv`

### 结构精简（Balanced）
- 主入口保留：
  - `scripts/run_experiment.py`
  - `scripts/analyze_experiment.py`
- 脚本分层归档：
  - `scripts/dev/`
  - `scripts/cloud/`
  - `scripts/dgx/`
- legacy 配置迁移：`legacy/configs/exp_*.yaml`（已标注 deprecated）。
- `.auth/` 停止跟踪，改为本地准备。

## 仍待完成（剩余 gap）
1. 端到端真实环境验收
- 需要在完整 VWA + Playwright + auth 条件下跑通 Phase1/2/3 全矩阵并留档。

2. 测试环境完善
- 当前环境缺少 `pytest`，尚未在本机执行自动化测试回归。

3. 运行手册进一步固化
- 可继续补充多机型部署细节（仅通用内容放 README，机器特化仍放 DGX 文档）。

## 运行建议
1. 安装依赖：
   - 最小：`pip install -e .`
   - 全功能：`pip install -e ".[analysis,dev]"`
2. 预检查：`bash scripts/preflight_v2.sh`
3. 实验运行：`python3 scripts/run_experiment.py --config configs/exp_v2_phase*.yaml`
4. 分析：`python3 scripts/analyze_experiment.py --run_dir <run_dir>`

## 机器特化说明
`DGX_SPARK_MACHINE_QUIRKS.md` 仅用于本机特化，不作为通用默认流程。
