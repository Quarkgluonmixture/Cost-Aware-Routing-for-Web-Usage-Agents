# Scripts Index

## 主入口 (Daily Use)
| Script | Description |
|--------|-------------|
| `run_experiment.py` | 运行实验，配合 `--config configs/XXX.yaml` |
| `analysis/analyze_experiment.py` | 分析实验结果，配合 `--run_dir results/...` |
| `analysis/analyze_reason_diagnostics.py` | 失败原因诊断分析，生成 reason_buckets + 报告 |
| `analysis/analyze_cross_representation.py` | 跨表征交叉分析，生成 oracle ceiling + 排他集 + router 信号 |
| `preflight_v2.sh` | 实验前环境预检（Python/CUDA/Docker/站点连通性） |
| `vwa_env.sh` | VWA 站点本地 URL 环境变量（`source scripts/vwa_env.sh`） |
| `vwa_env_remote.sh.example` | 远程 VWA 站点 URL 模板（复制为 `vwa_env_remote.sh`） |

## Sidecar / Watchdog
| Script | Description |
|--------|-------------|
| `glm_diagnosis_sidecar.py` | GLM 深度归因 sidecar，每 N episode 触发诊断+GLM总结+ntfy推送 |
| `experiment_watchdog.py` | 实验健康 watchdog，滚动窗口指标告警+idle 检测（无 GLM 依赖） |

## vwa/ — VWA 环境管理
| Script | Description |
|--------|-------------|
| `vwa/setup_vwa.sh` | 下载 VWA Docker 镜像和数据集 |
| `vwa/start_vwa_docker.sh` | 启动指定 VWA 站点的 Docker 容器 |
| `vwa/import_vwa_assets.sh` | 从其他机器离线导入 VWA 资产（镜像+数据集） |

## utils/ — 偶发工具
| Script | Description |
|--------|-------------|
| `utils/cleanup_logs.py` | 清理旧 log/result 文件（按时间/大小/数量） |
| `utils/reeval_phase1.py` | 离线重评估 phase1 中 evaluator_error 的任务 |

## dgx/ — DGX Spark 专用
| Script | Description |
|--------|-------------|
| `dgx/queue_b1_with_reset.sh` | B1 三模式队列（dom→som→vision，含 condition 间站点 reset+auth 刷新，带 watchdog+自动 resume） |
| `dgx/run_b0_3mode_classifieds.sh` | B0 三模式 classifieds（dom→reset→som→reset→vision，api_proxy 235B） |
| `dgx/restart_watchdog.sh` | 热重启 experiment_watchdog（保留 state） |
| `dgx/refresh_gallery.sh` | 刷新 gallery HTML |
| `dgx/trigger_watchdog_status.sh` | 向 watchdog 发送 SIGUSR1，立即触发状态报告 |

## dev/ — 开发/调试工具
| Script | Description |
|--------|-------------|
| `dev/run_one_vwa_episode.py` | 运行单个 VWA episode（测试 agent/观测流水线） |
| `dev/smoke_test_vwa.py` | VWA 环境快速 smoke test |
| `dev/check_disk_usage.sh` | 查看磁盘用量（top 目录/Docker/大文件） |

## cloud/ — 云部署
| Script | Description |
|--------|-------------|
| `cloud/sagemaker_setup.sh` | SageMaker 环境初始化 |
| `cloud/sagemaker_run.sh` | SageMaker 实验启动 |
