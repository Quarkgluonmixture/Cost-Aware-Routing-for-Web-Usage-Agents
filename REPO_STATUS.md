# Repository Status

**Last Updated:** 2026-03-27 (evening)
**Project:** Cost-Aware Routing for Web Usage Agents (P79)

## P79 对齐完成度（当前估计）
1. 实验内核与变量层级（Phase1/2/3、SoM、router、M1-M4）：**95%**
2. 开销核算与可解释分析闭环（RQ2/RQ3）：**92%**
3. 日志/分析产物（schema v2 + phase 报告）：**95%** ↑
4. 一键运行与跨机实践（WSL 起站、DGX 跑实验）：**88%** ↑
5. 文档/仓库发布就绪：**88%**

## 本轮关键变更（2026-03-27 evening）

### Baseline 配置变更
- **Qwen3-VL-4B 从 4-bit 量化改为 full bf16**
  - 原因：4-bit 量化在共享 GPU 环境下 VRAM 分配不稳定（transformers `device_map="auto"` + bnb 量化对碎片显存敏感），且 full precision 是更干净的 baseline 条件。
  - 文件：`configs/exp_v2_qwen3vl4b_baseline.yaml` (`quantization: "none"`)
  - 预估显存：~10 GB（bf16）vs 原 ~3-4 GB（4-bit）

### VRAM 预检查与自动等待
- 新增 `_wait_for_vram()` 函数：模型加载前检查可用显存，不足时每 30s 轮询等待，满足后自动继续。
  - 适配共享 DGX 场景（多用户共用 GPU）。
  - 可配置：`min_free_vram_gb: 12`（yaml 配置项），`timeout: 0`（无限等待）。
  - 文件：`p79/agents/qwen3vl_agent.py`
  - 透传：`p79/backends/local_qwen.py`

### state_digest 日志修复
- **问题：** step JSONL 中 `state_digest.url_before/url_after/title_before/title_after` 全为空字符串。
- **根因：** VWA `ScriptBrowserEnv.step()` 返回的 `info` dict 不含 `info["url"]`，URL 藏在 `info["page"].url`（`DetachedPage` dataclass）。
- **修复：**
  - `p79/envs/vwa_wrapper.py`：`_to_p79_obs()` 增加 `info["page"].url` fallback。
  - `p79/experiment/state_change.py`：`build_page_state()` 从 `info["page"]` 提取 URL，从 `info["page"].content` (HTML) 解析 `<title>`。
- **影响：** 纯日志改进，不改变 agent 行为。`detect_page_state_change()` 现在能检测到 `url_changed` 和 `title_changed` 事件。

### 循环检测 early stop
- **问题：** 4B 模型在低温度（T=0.1）下容易陷入行动循环（如反复搜索同一关键词），浪费 GPU 时间跑注定失败的 episode。
- **方案：** 两层检测，不改变 agent prompt 或决策（纯基础设施优化，等同于更精确的 max_steps timeout）。
  - **Strict：** `action_type + element_id + text` 完全匹配，3 轮重复触发。
  - **Soft：** `action_type + text` 匹配（忽略 element_id），3 轮重复触发。捕捉 DOM 重渲染后 element_id 变化但语义动作相同的循环。
- 实测 task 0：soft cycle 在 step 5 触发（原本跑到 step 11），节省 ~162s。
- 文件：`p79/experiment/runner.py`（`_action_signature()`, `_action_signature_soft()`, `_detect_action_cycle()`）

### 此前变更（2026-03-27 早间，保留记录）

#### 运行稳定性（DGX）
- 新增 GB10 NVRTC 兼容层，自动兜底 `torch.prod` 触发的 `invalid value for --gpu-architecture`。
  - 文件：`p79/utils/torch_cuda_workarounds.py`
  - 接入点：`p79/agents/qwen3vl_agent.py`
- VWA wrapper 在 `reset/step` 异常后主动 `close()`，避免后续 episode 被脏状态污染。
  - 文件：`p79/envs/vwa_wrapper.py`
- episode 失败日志增强，定位失败任务更快。
  - 文件：`p79/experiment/runner.py`

#### 脚本与预检查
- `scripts/dgx/run_qwen3vl4b_baseline.sh`
  - 自动创建 `logs/`
  - 打印内部日志路径
  - 若不存在 `p79_ai` conda 环境，不再刷报错
  - 默认注入占位 `OPENAI_API_KEY`（仅用于非 LLM eval 的导入链）
- `scripts/preflight_v2.sh`
  - 支持 `--help`、`--no-strict-ports`、`--require-cuda`、`--allow-missing-evaluator`
  - 增加 Playwright runtime 检查、Torch CUDA 检查、VWA evaluator import 检查
  - evaluator 导入检查时自动注入占位 key，避免无关失败

#### VWA 远程站点联动（WSL -> DGX）
- `scripts/start_vwa_docker.sh` 增强：
  - 容器名兼容：`shopping`/`vwa-shopping`、`forum`/`vwa-reddit`
  - 即使容器已在运行，也会重写 shopping base_url（避免 302 回 `localhost`）
  - classifieds compose 中 `CLASSIFIEDS=` 会按 `--hostname` 重写
  - homepage 模板中的 `localhost:*` 链接会按 `--hostname` 重写

#### 文档与忽略规则
- `DGX_SPARK_MACHINE_QUIRKS.md` 更新
- `.gitignore` 更新：`venv/`、`.venv/`、`scripts/vwa_env_remote.sh`

## 当前运行现状
1. **Baseline（B1）正在等待 VRAM：**
   - PID 878187，配置 `Qwen3-VL-4B bf16`，`min_free_vram_gb: 12`
   - 当前 GPU 空闲 ~6.6 GB / 121.7 GB（其他用户占用大量显存）
   - 进程每 30s 自动检查，满足后自动加载模型并开始跑 shopping 466 tasks
   - 日志：`logs/baseline_qwen3vl4b_shopping_2026-03-27_151540.log`
2. 远程三站核心链路：shopping/reddit/classifieds DGX 可达
3. homepage 仍可能因 Windows/WSL 端口代理导致 DGX 不可达（preflight 为 WARN）

## 已知 Bug（B1 跑完后修复）

### Cycle detection 在 SoM（及 DOM）模式下实质失效
- **发现日期：** 2026-04-06
- **影响范围：** 所有 Phase 1 条件（dom / som / vision）
- **Bug 1 — Strict 签名用了 element_id：** VWA 每步重渲染 AXTree，element_id 在步与步之间不稳定。Strict 签名 `atype|eid=X|t=...|c=...|d=...` 无法匹配语义相同但 eid 不同的动作。
- **Bug 2 — Soft buffer 被 page_changed 清空：** SoM 模式下几乎每步都触发 `content_changed`（SoM 标注本身是动态内容），导致 soft buffer 每步清空，永远无法积累 4 次重复。
- **实际影响：** task=6 (classifieds, motorcycle 搜索) 在 SoM 下跑满 30 步，应在 ~8 步 early stop。
- **处理策略：** 不在实验中途修复（避免 within-condition 不一致）。B1 全部跑完后修复。分析阶段用离线 soft signature 重标注哪些 episode 本应被 early stop。
- **修复方案：** (1) Strict 签名去掉 eid，改用 `atype|t=text|c=coord|d=delta`；(2) Soft buffer 改用 URL path 变化触发清空，而非 `page_changed`。

## 仍待完成项
1. homepage `4399` 远程可达性固化为稳定 PASS（减少 run 间歇性波动）
2. Phase1/2/3 在远程站点模式下做一次完整可复现实验留档
3. 多 seed 统计汇总（mean/std/置信区间）纳入默认分析报告
4. 旧 4-bit run (`run_1774622012`) 数据可归档或删除，不作为正式 baseline
5. B1 跑完后修复 cycle detection bug（见上方）

## 说明
- `DGX_SPARK_MACHINE_QUIRKS.md` 是机器特化文档，不作为通用默认流程。
- `scripts/vwa_env_remote.sh` 是本机私有配置文件，不纳入版本管理。
