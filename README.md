# P79: Cost-Aware Routing for Web Usage Agents

## 目标
本仓库对齐 `P79_experimental_scope_rq_variables.md`，用于构建可复现、可扩展、可核算开销的 WebAgent 实验系统，直接支持：

- RQ1: SoM / observation mode / router 的实验筛选
- RQ2: best-fixed vs routed 的净收益评估（含 router overhead）
- RQ3: M1-M4 单模块消融与可解释性分析

当前主线默认覆盖 VisualWebArena 四站点：`shopping` / `reddit` / `wikipedia` / `classifieds`。

## 快速开始
### 1) 安装
最小运行依赖：

```bash
pip install -e .
```

全功能（分析+测试）：

```bash
pip install -e ".[analysis,dev]"
```

### 2) 启动 VWA 环境（通用）
先检查环境：

```bash
bash scripts/preflight_v2.sh
```

按需拉取并启动 VWA Docker：

```bash
bash scripts/setup_vwa.sh --target-dataset all
bash scripts/start_vwa_docker.sh --sites all
source scripts/vwa_env.sh
```

说明：`scripts/start_vwa_docker.sh` / `scripts/setup_vwa.sh` 都支持非交互参数化运行。

### 3) 准备认证文件（本地）
`.auth/` 已停止跟踪，不随 git 提交。请在本机自行生成或复制到仓库根目录 `.auth/`。

## Phase 命令
统一入口仅保留：

- `scripts/run_experiment.py`
- `scripts/analyze_experiment.py`

运行 Phase 1/2/3：

```bash
python3 scripts/run_experiment.py --config configs/exp_v2_phase1.yaml
python3 scripts/run_experiment.py --config configs/exp_v2_phase2.yaml
python3 scripts/run_experiment.py --config configs/exp_v2_phase3.yaml
```

分析单次运行目录：

```bash
python3 scripts/analyze_experiment.py --run_dir results/visualwebarena/phase2/<RUN_ID>
```

## 结果目录
统一目录层级：

```text
results/<benchmark>/<phase>/<run_id>/<condition_id>/
```

关键产物（按 phase 自动输出）包括：

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

## 常见问题
1. 为什么 `preflight_v2.sh` 报缺少环境变量？
   先执行 `source scripts/vwa_env.sh`，并确认本机 `.auth/` 已就绪。

2. 为什么分析命令报错缺少 pandas/matplotlib？
   安装扩展依赖：`pip install -e ".[analysis]"`。

3. 为什么不再推荐旧脚本/旧配置？
   v2 主线只维护统一实验入口。旧批跑配置和脚本已移到 `legacy/` 或 `scripts/dev|cloud|dgx/`。

4. DGX 本机特化怎么跑？
   DGX 特化不作为通用默认，请看：
   - `DGX_SPARK_MACHINE_QUIRKS.md`
   - `scripts/dgx/`

## 额外文档
- 可选字段语义：`docs/STEP_SCHEMA_V2_OPTIONAL_FIELDS.md`
- 第三方代码引用：`docs/THIRD_PARTY_CODE.md`
