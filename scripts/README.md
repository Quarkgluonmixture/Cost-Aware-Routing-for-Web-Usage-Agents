# Scripts Index

> 顶层只放入口与环境文件，其余按职责分到 `analysis/` / `queues/` / `maintenance/` / `vwa/`。
> Daily 命令优先用 `make help`（仓库根 `Makefile`），本文档仅列脚本职责。

## 顶层入口

| Script | Description |
|--------|-------------|
| `run_experiment.py` | 实验入口，配合 `--config configs/XXX.yaml` |
| `preflight_v2.sh` | 实验前环境预检（Python/CUDA/Docker/站点连通性） |
| `vwa_env.sh` | 本地 VWA 站点 URL 环境变量（`source scripts/vwa_env.sh`） |
| `vwa_env_remote.sh.example` | 远程 VWA 站点 URL 模板（复制为 `vwa_env_remote.sh`，不入版本管理） |

## `analysis/` — 数据分析

| Script | Description |
|--------|-------------|
| `analyze_experiment.py` | 主分析入口（condition 对比、SR、cost） |
| `analyze_reason_diagnostics.py` | 失败模式聚类与归因（reason_buckets + 报告） |
| `analyze_cross_representation.py` | 跨表征交叉分析（oracle ceiling + 排他集 + router 信号） |
| `analyze_confidence_calibration.py` | Logprobs 置信度校准 |
| `compare_b0_b1.py` | B0 vs B1 对比 |
| `validate_run.py` | run 目录数据完整性校验 |
| `aggregate_cross_site.py` | 跨站点汇总 |
| `collect_analysis_summary.py` | 收集 analysis 输出 |
| `diag_pattern_match.py` | 诊断 pattern 匹配 |
| `analyze_noninteractive_click_earlystop.py` | 非交互点击早停分析 |
| `analyze_search_over_browse.py` | search vs browse 行为分析 |
| `analyze_reddit_selflink_cycle.py` | Reddit self-link 循环分析 |
| `analyze_comment_selflink_loop.py` / `_v2.py` | Comment self-link loop 模式分析（基础 + 后续） |
| `b0_vision_coordinate_errors.py` | B0 vision 坐标错误定量统计 |

## `queues/` — 实验队列

| Script | Description |
|--------|-------------|
| `queue_b0_with_reset.sh` | B0 三模式 VWA classifieds→reddit→shopping，每 condition 间 reset |
| `queue_b0_wa_with_reset.sh` | B0 三模式 WA shopping→shopping_admin→reddit |
| `queue_b1_with_reset.sh` | B1 三模式 VWA（同 B0_VWA 结构，本地 4B 模型，无 API key） |
| `queue_b1_wa_with_reset.sh` | B1 三模式 WA |
| `run_scroll_comparison.sh` | Scroll error 一次性交叉验证（Claude / DashScope） |

> 4 个 queue 脚本结构相近但有 ~50% 真实差异（API key、config 来源、dataset、auth 逻辑），暂未参数化合并——Phase 1 三模式跑完后再处理。

## `maintenance/` — 数据维护、守护、调试

### 数据修复
| Script | Description |
|--------|-------------|
| `rederive_episode_summary.py` | 修补 episode summary（adjusted_success/cost/etc.，§95 canonical） |
| `clear_tasks.py` | 清 task summary/steps/artifacts/digest 记录（统一入口，**不要手动 rm**） |
| `create_b1_classifieds_stub.py` | 一次性恢复 §34 误删的 classifieds B1 stub |
| `split_wa_tasks.py` | 一次性拆分 WA test_webarena.raw.json → per-site |
| `reeval_phase1.py` | 离线重评估 phase1 中 evaluator_error 的任务 |

### 守护与监控
| Script | Description |
|--------|-------------|
| `experiment_watchdog.py` | 实验守护进程：进度推送、idle 告警、自动 digest/标注/gallery |
| `restart_watchdog.sh` | 热重启 watchdog（保留 state 与原始参数，支持 `--append-args`） |
| `trigger_watchdog_status.sh` | 向 watchdog 发 SIGUSR1 立即触发状态报告 |
| `wait_for_reddit_then_rederive.sh` | reddit 跑完→自动 rederive→自动启 shopping queue（用 `make watch-reddit`） |

### 诊断与可视化
| Script | Description |
|--------|-------------|
| `glm_diagnosis_sidecar.py` | GLM 实时诊断 sidecar |
| `glm_batch_digest.py` | GLM sidecar 批量诊断（自动归因每个 episode） |
| `digest_enrich.py` | Digest 后处理与富化 |
| `annotate_screenshots.py` | 截图标注（动作 banner + thought + 元素高亮） |
| `generate_gallery.py` | HTML 画廊（键盘导航、自动刷新、远程访问） |
| `refresh_gallery.sh` | 一键刷新：`annotate_screenshots` + `generate_gallery` |

### 站点与 episode 维护
| Script | Description |
|--------|-------------|
| `reset_vwa_sites.sh` | 重置 VWA 站点（用于 condition 间清理） |
| `retry_b1_single_task.sh` | 重跑 B1 单 task 三模式 |

### 偶发工具
| Script | Description |
|--------|-------------|
| `cleanup_logs.py` | 清旧 log/result 文件（按时间/大小/数量） |
| `check_disk_usage.sh` | 磁盘用量快速查看 |
| `smoke_test_vwa.py` | VWA 环境 smoke test |
| `run_one_vwa_episode.py` | 跑单个 VWA episode（测试 agent/观测流水线） |

## `vwa/` — VWA 站点 setup

| Script | Description |
|--------|-------------|
| `setup_vwa.sh` | 下载 VWA Docker 镜像与数据集 |
| `start_vwa_docker.sh` | 启动指定 VWA 站点容器 |
| `import_vwa_assets.sh` | 离线导入 VWA 资产（镜像 + 数据集） |
