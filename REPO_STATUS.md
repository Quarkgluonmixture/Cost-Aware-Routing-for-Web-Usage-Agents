# Repository Status

**Last Updated:** 2026-03-27  
**Project:** Cost-Aware Routing for Web Usage Agents (P79)

## P79 对齐完成度（当前估计）
1. 实验内核与变量层级（Phase1/2/3、SoM、router、M1-M4）：**95%**
2. 开销核算与可解释分析闭环（RQ2/RQ3）：**92%**
3. 日志/分析产物（schema v2 + phase 报告）：**94%**
4. 一键运行与跨机实践（WSL 起站、DGX 跑实验）：**85%**
5. 文档/仓库发布就绪：**88%**

## 本轮关键变更（2026-03-27）

### 运行稳定性（DGX）
- 新增 GB10 NVRTC 兼容层，自动兜底 `torch.prod` 触发的
  `invalid value for --gpu-architecture`。
  - 文件：`p79/utils/torch_cuda_workarounds.py`
  - 接入点：`p79/agents/qwen3vl_agent.py`
- VWA wrapper 在 `reset/step` 异常后主动 `close()`，避免后续 episode 被脏状态污染。
  - 文件：`p79/envs/vwa_wrapper.py`
- episode 失败日志增强，定位失败任务更快。
  - 文件：`p79/experiment/runner.py`

### 脚本与预检查
- `scripts/dgx/run_qwen3vl4b_baseline.sh`
  - 自动创建 `logs/`
  - 打印内部日志路径
  - 若不存在 `p79_ai` conda 环境，不再刷报错
  - 默认注入占位 `OPENAI_API_KEY`（仅用于非 LLM eval 的导入链）
- `scripts/preflight_v2.sh`
  - 支持 `--help`、`--no-strict-ports`、`--require-cuda`、`--allow-missing-evaluator`
  - 增加 Playwright runtime 检查、Torch CUDA 检查、VWA evaluator import 检查
  - evaluator 导入检查时自动注入占位 key，避免无关失败

### VWA 远程站点联动（WSL -> DGX）
- `scripts/start_vwa_docker.sh` 增强：
  - 容器名兼容：`shopping`/`vwa-shopping`、`forum`/`vwa-reddit`、`wikipedia`/`vwa-wikipedia`
  - 即使容器已在运行，也会重写 shopping base_url（避免 302 回 `localhost`）
  - classifieds compose 中 `CLASSIFIEDS=` 会按 `--hostname` 重写
  - homepage 模板中的 `localhost:*` 链接会按 `--hostname` 重写

### 文档与忽略规则
- `DGX_SPARK_MACHINE_QUIRKS.md` 更新：
  - CPU-only torch 识别与修复
  - GB10 NVRTC 问题与 fallback
  - 远程站点 `localhost` 重定向问题
  - homepage `:4399` 端口代理注意事项
- `.gitignore` 更新：
  - `venv/`、`.venv/`
  - `scripts/vwa_env_remote.sh`（主机本地私有配置）

## 当前运行现状（与你本次联调一致）
1. 远程四站核心链路：
   - shopping/reddit/wikipedia/classifieds：DGX 可达
2. shopping 重定向：
   - 已从 `Location: http://localhost:7770/...` 修复为对外可用地址
3. homepage:
   - 仍可能因 Windows/WSL 端口代理导致 DGX 不可达（preflight 为 WARN）
4. baseline：
   - 建议前台先确认首个 task 进入执行，再切后台长跑

## 仍待完成项
1. homepage `4399` 远程可达性固化为稳定 PASS（减少 run 间歇性波动）
2. Phase1/2/3 在远程站点模式下做一次完整可复现实验留档
3. 多 seed 统计汇总（mean/std/置信区间）纳入默认分析报告

## 说明
- `DGX_SPARK_MACHINE_QUIRKS.md` 是机器特化文档，不作为通用默认流程。
- `scripts/vwa_env_remote.sh` 是本机私有配置文件，不纳入版本管理。
