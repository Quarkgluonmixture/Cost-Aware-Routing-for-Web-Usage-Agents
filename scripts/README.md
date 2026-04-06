# Scripts Index

## 主入口 (Daily Use)
| Script | Description |
|--------|-------------|
| `run_experiment.py` | 运行实验，配合 `--config configs/XXX.yaml` |
| `analyze_experiment.py` | 分析实验结果，配合 `--run_dir results/...` |
| `analyze_reason_diagnostics.py` | 失败原因诊断分析，生成 reason_buckets + 报告 |
| `preflight_v2.sh` | 实验前环境预检（Python/CUDA/Docker/站点连通性） |
| `vwa_env.sh` | VWA 站点本地 URL 环境变量（`source scripts/vwa_env.sh`） |
| `vwa_env_remote.sh.example` | 远程 VWA 站点 URL 模板（复制为 `vwa_env_remote.sh`） |

## Sidecar
| Script | Description |
|--------|-------------|
| `reason_diag_live_sidecar.py` | 实时增量归因 sidecar，每 N episode 触发诊断+GLM总结+ntfy推送 |

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
| `dgx/queue_b1_serial.sh` | B1 baseline 串行队列（classifieds→reddit→shopping），带 watchdog+自动 resume |
| `dgx/restart_queue_b1_serial.sh` | 从最新状态重启 B1 队列 |
| `dgx/restart_sidecar.sh` | 热重启 reason_diag_live_sidecar（保留 state，可追加参数） |
| `dgx/qwen3vl4b_status.sh` | 查看 Qwen3-VL-4B baseline 当前状态（进程/日志/GPU/完成度） |
| `dgx/run_qwen3vl4b_baseline.sh` | DGX baseline 执行脚本（含站点健康检查、CUDA 环境） |
| `dgx/start_qwen3vl4b_baseline_site.sh` | 单站点 baseline 启动器（细粒度环境变量控制） |
| `dgx/start_qwen3vl4b_baseline_when_gpu_idle.sh` | GPU 空闲检测，自动触发 baseline |

## dev/ — 开发/调试工具
| Script | Description |
|--------|-------------|
| `dev/run_one_vwa_episode.py` | 运行单个 VWA episode（测试 agent/观测流水线） |
| `dev/smoke_test_vwa.py` | VWA 环境快速 smoke test |
| `dev/check_disk_usage.sh` | 查看磁盘用量（top 目录/Docker/大文件） |
| `dev/monitor_glm_sidecar.py` | 旧版 GLM sidecar 监控（实验性，已由 reason_diag_live_sidecar 取代） |

## cloud/ — 云部署
| Script | Description |
|--------|-------------|
| `cloud/sagemaker_setup.sh` | SageMaker 环境初始化 |
| `cloud/sagemaker_run.sh` | SageMaker 实验启动 |
