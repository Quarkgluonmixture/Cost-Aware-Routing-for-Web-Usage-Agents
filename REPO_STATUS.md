# Repository Status

**Last Updated:** 2026-03-26
**Project:** Cost-Aware Routing for Web Usage Agents (P79)

## P79 对齐完成度面板
以 `P79_experimental_scope_rq_variables.md` 为基线，当前状态（估算）：

1. 实验内核与变量层级（Phase1/2/3, SoM, router, M1-M4）：**95%**
2. RQ2/RQ3 开销核算与可解释性：**93%**
3. 日志与分析闭环（trigger/reason/checklist/Pareto）：**95%**
4. 安装依赖、Docker 实践、一键运行：**75%**
5. 文档与发布就绪度：**85%**

## 当前主线（v2）
### 已完成
- 统一实验入口：`run-experiment` / `analyze-experiment`（CLI + scripts wrapper）。
- 条件矩阵自动生成：
  - Phase1: 2x2 SoM/observation screening
  - Phase2: `phase2_fixed_best` vs `phase2_routed`
  - Phase3: `phase3_none/m1/m2/m3/m4` 单模块消融
- SoM 接入与降级策略：无 bbox 时降级文本 SoM，记录 `degraded_som=true`。
- SoM 绘制不再污染原始 observation（在副本上绘制）。
- Router 开销拆账：decision/dom parse/screenshot/extra model/retry。
- Router 升级到 hybrid 时实际计量 `extra_screenshot_ms` 开销。
- 成本口径修复：
  - `total_model_cost_usd`
  - `total_router_overhead_cost_usd`
  - `total_cost_usd = model + overhead`
  - Phase2 `NetSaving` 计算避免 double-count overhead。
  - **新增：** latency net saving 分解 (`phase2_net_saving_latency.json`)
  - **新增：** energy net saving 分解 (`phase2_net_saving_energy.json`)
- **成本参数已填入实际估计值**（本地 GPU 摊销 + API 定价 + overhead 费率）。
- Step schema v2 + 可解释字段：`page_change_reasons`、`text_similarity`、`checklist`、`state_digest`。
- 错误分类标准化：`parse_error` / `invalid_action` / `no_progress` / `env_error` / `benchmark_noise`。
- **benchmark noise 检测扩展**：增加 timeout、Playwright error、connection error、Docker service error、navigation error 类别。
- 分析产物增强：
  - `phase1_representation_screening.csv/png`
  - `phase2_pareto_metrics.csv/png` — **含 Pareto front 标注**
  - `phase2_pareto_latency.png` — success vs latency Pareto
  - `phase2_pareto_energy.png` — success vs energy Pareto
  - `phase2_net_saving_decomposition.csv/json`
  - `phase2_net_saving_latency.json`
  - `phase2_net_saving_energy.json`
  - `phase3_module_ablation.csv/png`
  - `phase3_module_gain_vs_base.csv`
  - `trigger_distribution.csv/png`
  - `state_change_reason_distribution.csv/png`
  - `checklist_progress_curve.csv/png`
  - `checklist_failure_distribution.csv/png`
  - `benchmark_noise_report.csv`

### 本轮修复（2026-03-26）

| 修复 | 文件 | 说明 |
|------|------|------|
| dom_only 不再绕过 LLM | `backends/local_qwen.py`, `api_qwen.py` | dom_only 现在仍走 LLM 推理（text-only），heuristic 仅在显式 `dom_mode: "heuristic_only"` 时使用 |
| 成本参数实际化 | `configs/exp_v2_base.yaml` | 填入 GPU 摊销估计、API 定价、overhead 费率 |
| SoM 绘制副本 | `experiment/som.py` | 在 `image.copy()` 上绘制，不再污染原始 obs |
| M4 真正的两阶段分离 | `backends/*.py`, `runner.py`, `base.py` | planner 输出 sub-goal → 传递给 grounder；新增 `planner_sub_goal` 字段 |
| Pareto front 计算 | `experiment/analysis.py` | 非支配解识别 + 三轴 Pareto 图（cost/latency/energy） |
| 能源追踪默认开启 | `energy_tracker.py`, base config | 新增 `dgx_spark` / `gb200` 硬件 profile，默认启用 |
| 多维 net saving | `metrics.py`, `analysis.py` | 新增 `net_saving_latency()` / `net_saving_energy()` |
| screenshot overhead 计量 | `runner.py` | router 升级时实际计时 image prep |
| M3 智能重试 | `modules.py` | 根据失败的 action type 选择不同重试策略 |
| 多 seed 支持 | `runner.py` | `seed` 可为列表，自动扩展 condition×seed |
| noise 检测扩展 | `metrics.py` | 7 类噪声模式（原 2 类） |

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

2. 多 seed 统计显著性
   - 框架已支持 `seed: [42, 123, 456]`，分析侧可进一步输出 mean +/- std。

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
