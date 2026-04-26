# P79: Cost-Aware Routing for Web Usage Agents

研究 cost-aware routing 能否改善 web agent 的成功率-效率权衡。基于 VisualWebArena 三站点（shopping / reddit / classifieds），使用 Qwen3-VL-4B 作为 baseline 模型。

对齐文档：`P79_experimental_scope_rq_variables.md`

## 快速开始

```bash
# 安装
pip install -e .                    # 最小依赖
pip install -e ".[analysis,dev]"    # 全功能

# 环境预检
bash scripts/preflight_v2.sh

# 运行实验
python3 scripts/run_experiment.py --config configs/exp_v2_phase1.yaml

# 分析结果
python3 scripts/analysis/analyze_experiment.py --run_dir results/visualwebarena/phase1/<RUN_ID>

# 测试
pytest tests/
```

## 三阶段实验设计

- **Phase 1** — 表征筛选：2×2 grid (SoM on/off × dom_only/hybrid)
- **Phase 2** — 路由研究：fixed best vs rule-based router
- **Phase 3** — 模块消融：M1(select fallback) / M2(input fallback) / M3(retry) / M4(two-stage)

## 代码结构

```
p79/
├── agents/           # LLM 推理（qwen3vl_agent=本地, qwen_api_agent=API）
├── backends/         # Backend 抽象层（local_qwen/api_qwen/heuristic + action_utils）
├── envs/             # VWA 环境封装（P79Observation 标准化）
├── experiment/       # 核心实验引擎
│   ├── runner.py     # 主编排器 condition→seed→task→step
│   ├── router.py     # 规则路由器（dom_only↔hybrid 切换）
│   ├── conditions.py # Phase1/2/3 条件生成
│   ├── modules.py    # M1-M4 辅助模块
│   ├── som.py        # Set-of-Marks 标注
│   ├── state_change.py # 页面状态变化检测
│   ├── metrics.py    # 成本/延迟/能耗聚合
│   ├── logger_v2.py  # JSONL 结构化日志（写入带 fsync 持久化）
│   ├── io_utils.py   # JSONL 读取 + restart dedup（统一入口）
│   ├── analysis.py   # 后分析与可视化
│   └── config.py     # 配置加载与默认值合并
├── utils/            # 工具（CUDA workaround, asyncio, log cleanup）
└── cli/              # CLI 入口
configs/              # YAML 实验配置
scripts/              # 运行/部署/分析脚本（见下方）
tests/                # 单元测试
docs/                 # 分析报告、schema 文档、周报
```

## 脚本说明

### 实验运行

| 脚本 | 用途 |
|------|------|
| `scripts/run_experiment.py` | 统一实验入口（Phase 1/2/3） |
| `scripts/preflight_v2.sh` | 环境预检（CUDA、VWA 站点、认证） |
| `scripts/vwa_env.sh` | 本地 VWA 环境变量 |
| `scripts/vwa_env_remote.sh` | 远程 VWA 环境变量（DGX 模式） |

### 实验队列（`scripts/queues/`）

| 脚本 | 用途 |
|------|------|
| `scripts/queues/queue_b0_with_reset.sh` / `queue_b1_with_reset.sh` | B0/B1 三模式 VWA 队列（classifieds→reddit→shopping，condition 间 reset） |
| `scripts/queues/queue_b0_wa_with_reset.sh` / `queue_b1_wa_with_reset.sh` | B0/B1 三模式 WA 队列（shopping→shopping_admin→reddit） |
| `scripts/maintenance/restart_watchdog.sh` | 热重启 watchdog（自动恢复所有参数，支持 `--append-args`） |
| `scripts/maintenance/wait_for_reddit_then_rederive.sh` | reddit 跑完→自动 rederive→自动启 shopping queue（用 `make watch-reddit`） |

### 监控与诊断

| 脚本 | 用途 |
|------|------|
| `scripts/maintenance/experiment_watchdog.py` | 实验守护进程：进度推送、idle 告警、自动 digest/标注/gallery |
| `scripts/maintenance/glm_batch_digest.py` | GLM sidecar 批量诊断（自动归因每个 episode） |
| `scripts/maintenance/glm_diagnosis_sidecar.py` | GLM 实时诊断 sidecar |
| `scripts/maintenance/digest_enrich.py` | Digest 后处理与富化 |

### 可视化

| 脚本 | 用途 |
|------|------|
| `scripts/maintenance/annotate_screenshots.py` | 截图标注（动作 banner + thought + 元素高亮） |
| `scripts/maintenance/generate_gallery.py` | HTML 画廊（键盘导航、自动刷新、远程访问） |

### 分析

| 脚本 | 用途 |
|------|------|
| `scripts/analysis/analyze_experiment.py` | 主分析入口（condition 对比、成功率、成本） |
| `scripts/analysis/analyze_reason_diagnostics.py` | 失败模式聚类与归因分析 |
| `scripts/analysis/analyze_confidence_calibration.py` | Logprobs 置信度校准分析 |
| `scripts/analysis/analyze_cross_representation.py` | 跨表征对比分析 |

## 结果目录

```
results/<benchmark>/<phase>/<run_id>/
├── <condition_id>/
│   ├── episodes/         # 每 episode 的 steps JSONL + summary JSON
│   ├── artifacts/        # 截图、DOM、SoM 图、标注截图
│   ├── condition_meta.json
│   └── condition_summary_v2.json
├── analysis/             # 分析输出（reason_diagnostics、digest 等）
└── gallery.html          # 自动生成的可视化画廊
```

## DGX Spark 注意事项

- 用 `python3` 或 `.venv/bin/python`，不要用 `python`
- 必须设置 `PYTORCH_NVML_BASED_CUDA_CHECK=1`（脚本已自动处理）
- GB10 `sm_121` 架构可能触发 nvrtc 错误，仓库内置 fallback 自动兜底
- 远程站点配置见 `scripts/vwa_env_remote.sh`
- 详细机器特化见 `DGX_SPARK_MACHINE_QUIRKS.md`

## 文档

| 文件 | 内容 |
|------|------|
| `P79_experimental_scope_rq_variables.md` | 实验范围与研究问题 |
| `DGX_SPARK_MACHINE_QUIRKS.md` | DGX 机器特化 |
| `docs/reference/STEP_SCHEMA_V2_OPTIONAL_FIELDS.md` | Step schema 可选字段 |
| `docs/reference/ANALYSIS_SCRIPTS.md` | 分析脚本详细说明 |
| `docs/analysis/<site>/` | B1 baseline 分析报告（按站点分目录，`*_manual` 人工定性 / `*_digest` GLM 定量） |
| `docs/literature/` | 文献综述与 P79 映射 |
| `docs/checkpoints/周报/` | 实验进展周报 |
