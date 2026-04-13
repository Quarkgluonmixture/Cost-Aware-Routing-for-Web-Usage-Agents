# B0 / B1 实际调用链与实验设置

> 生成日期：2026-04-13（经代码深度核查补全）

---

## 一、运行时环境概览：三个并行进程

B1 实验（B0 同理）运行时，系统里同时存在三个独立进程，各司其职、通过文件系统共享状态：

```
┌─────────────────────────────────────────────────────────────────────────┐
│  进程 1：queue_b1_with_reset.sh                                          │
│  角色：总调度员（Orchestrator）                                            │
│  职责：决定"跑什么、跑几次、什么时候重启"                                    │
│  生命周期：整个实验从头到尾（reddit + shopping 全程）                        │
└──────────────────────────┬──────────────────────────────────────────────┘
                           │ 启动 / kill / 重启
          ┌────────────────▼────────────────┐
          │  进程 2：run_experiment.py        │
          │  角色：实际执行者（Runner）         │
          │  职责：跑 tasks，写结果文件         │
          │  生命周期：一个 condition 一个进程   │
          │  （完成或卡死后被 kill，再重新启动） │
          └─────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│  进程 3：experiment_watchdog.py                                          │
│  角色：旁观监控员（Monitor）                                               │
│  职责：监控进展、运行分析、发通知、处理异常                                   │
│  生命周期：一个站点的三模式全程（不随 runner 重启而重启）                      │
└─────────────────────────────────────────────────────────────────────────┘
```

**核心设计原则**：三个进程的唯一共享介质是**文件系统**。runner 写结果文件，watchdog 读结果文件，queue 检查文件是否存在来判断完成状态。三者之间没有直接 IPC（无管道/socket），完全松耦合。queue 是唯一可以 kill/restart runner 的进程；watchdog 只读文件、写分析结果，永远不控制 runner 进程的生死。

---

## 二、各进程详解

### 进程 1：queue（总调度员）

**脚本**：`scripts/dgx/queue_b1_with_reset.sh`

**职责**：
- 决定实验的整体顺序：reddit 全部三模式完成后，再跑 shopping
- 对每个站点，按固定顺序跑三个 condition：DOM → reset → SoM → reset → Vision
- 负责站点 reset（SSH 到 Windows 执行 PowerShell 脚本）和重新登录（Playwright 刷新 auth cookies）
- 启动和停止 watchdog（每个站点一个 watchdog 实例）
- **内层进度 watchdog**（shell 循环，不是独立进程）：每 60s 检查一次新 episode 数量，如果 35min 没有任何新 episode 产生，就认为 runner 卡死，kill 掉它并等 15s 后重新启动（resume 模式）
- 最多允许 10 次自动 resume，超过则发 ntfy urgent 告警并终止
- 实验全部完成后，调用 `analyze_reason_diagnostics.py --report --report-language zh` 做最终归因报告

**关键参数**（可通过环境变量覆盖）：
```
WATCHDOG_TIMEOUT_MINS=35      内层 watchdog 超时（分钟）
WATCHDOG_CHECK_SECS=60        内层 watchdog 检查间隔（秒）
MAX_RESUME_ATTEMPTS=10        最多自动重启次数
NTFY_MINIMAL_MODE=1           只发关键通知（减少噪音）
EXP_WATCHDOG_ENABLE=1         是否启动外层 Python watchdog
EXP_WATCHDOG_POLL_SECS=30     外层 watchdog 轮询间隔
EXP_WATCHDOG_IDLE_ALERT_MINS=20  外层 watchdog idle 告警阈值
P79_DISABLE_STALE_CLEANUP=1   禁止 runner 启动时清理旧结果
```

---

### 进程 2：runner（实际执行者）

**脚本**：`scripts/run_experiment.py` → `p79/cli/run_experiment.py` → `p79/experiment/runner.py`

**CLI 参数**：
```
--config     YAML 配置文件路径（必填）
--run_id     覆盖自动生成的 run_id
--phase      覆盖 phase（phase1/phase2/phase3）
--max_steps  覆盖每 episode 最大步数
--log_path   日志路径，存入 run_meta.json
```

**启动流程**（`ExperimentRunner.run()`）：
1. `_cleanup_stale_runs()`：如果 `P79_DISABLE_STALE_CLEANUP` 未设置，清理旧的不完整 run（不影响已有 summary）
2. `_write_run_meta()`：写入 `run_meta.json`（含配置快照、开始时间、log_path）
3. 遍历 conditions（每次 queue 传入的 config 只有一个 condition）：
   - 创建 `LoggerV2`，写 `condition_meta.json`
   - **Resume 逻辑**：检查每个 task 的 `_summary_v2.json` 是否已存在；已有 summary 且无 error 则跳过
   - 调用 `_run_and_record_episode()` 跑每个 task
   - **Retry pass**：runner 自己也会做一轮检查——watchdog 可能在 runner 运行中删除了某些 error episode 的 summary，runner 检测到 summary 缺失后会重新跑那些 task
   - 全部 task 完成后调用 `aggregate_condition_metrics()`，写 `condition_summary_v2.json`
   - 立即调用 `_run_post_condition_analysis()`（subprocess 调用 `analyze_experiment.py`，timeout=300s）
4. 写 `run_summary_v2.json`（含 phase2 净节省估算、trigger_distribution 等）
5. 创建 `latest_{site}` 符号链接（方便快速访问最新结果）
6. 关闭 VWA 环境和 energy_tracker

**B1 和 B0 的核心差异在这里**（backend → agent → 模型）：
```
B1: local_4b  → LocalQwenBackend(local_qwen.py)  → Qwen3VLAgent(qwen3vl_agent.py)  → 本地 GPU 推理
B0: api_strong → ApiProxyBackend(api_proxy.py)    → ProxyApiAgent(proxy_api_agent.py) → HTTP → 235B
```

**模型加载时机**：LocalQwenBackend 在 `__init__` 时立即加载模型（非 lazy），ApiProxyBackend 在 `__init__` 时初始化 HTTP 客户端，两者都不是 per-step lazy load。

---

### 进程 3：watchdog（旁观监控员）

**脚本**：`scripts/experiment_watchdog.py`

watchdog **不控制** runner 的生死（那是 queue 的工作），它只旁观、汇报、做后处理。

**启动参数**（由 queue 传入）：
```
--run-dir         结果目录（必填）
--poll-secs 30    轮询间隔
--idle-alert-mins 20  idle 告警阈值
--ntfy-topic      ntfy 推送 topic
--state-file      持久化状态文件路径（.state.json）
--glm-config      GLM config 路径（有此参数才启用 digest）
--digest-dir      digest 输出目录
--aggregate-prefix B1_3mode  aggregate gallery 前缀
--notify-completion  是否发 condition 完成通知
```

**主循环**（每 `poll_secs` 秒执行）：

**① 每轮必做：扫描新 episode**
- glob 所有 `*_summary_v2.json`，过滤未见过的 key
- 对每个新 episode：打印进度日志、会话健康检查、error 自动重试
- 将已见 key 持久化到 `.state.json`（fsync 保证崩溃安全）

**② 每 30min：周期状态报告**
- 运行 `analyze_reason_diagnostics.py --skip-similarity`（更新归因 CSV）
- 运行 `glm_batch_digest.py`（对失败 episode 做 GLM 解读，写 `digest_{mode}.jsonl`，max-images=3，delay=3s）
- 运行 `annotate_screenshots.py`（给截图加标注）
- 运行 `generate_gallery.py`（刷新单 run gallery + aggregate gallery）
- 发 ntfy 推送：进度摘要 + 最近新完成 tasks + pipeline 状态

**③ 每轮检查：condition 完成**
- 检测 `condition_summary_v2.json` 是否出现（由 runner 写入）
- 新出现时触发完整分析流水线（见第十一节）
- 发 ntfy POST-ANALYSIS 通知

**④ 每轮检查：idle 告警**
- 20min 无新 episode → 发 ntfy `high` 优先级告警

**异常处理能力**：
- **error episode 自动重试**：检测到 summary 含 error 字段 → 删除 summary + steps + artifacts → runner resume 时重跑（code_bug 最多 2 次，benchmark_noise 无限次）
- **session 丢失检测**：连续 3 个 task 的 step_000 DOM 无 logout 链接 → 发 urgent 告警 + 自动 Playwright 重登录
- **session 恢复后自动清理**：登录恢复后，删除所有"未登录状态下跑的"污染 episode，让 runner 重跑
- **持久化状态**：所有状态写入 `.state.json`（seen_keys、seen_completions、analysis mtimes 等），watchdog 重启后恢复，不重复触发

**手动触发立即报告**：`kill -USR1 <watchdog_pid>`（SIGUSR1 信号）

---

## 三、三个进程的交互关系

```
                    ┌──────────────────────────────────────────┐
                    │         queue_b1_with_reset.sh           │
                    │                                          │
                    │  内层 shell watchdog 每 60s:             │
                    │  count_episode_summaries(run_dir)        │
                    │  35min 无增长 → kill runner → restart     │
                    └────────┬─────────────────┬──────────────┘
                             │ nohup 启动        │ nohup 启动
                             │ （每 condition）  │ （每 site 一次）
                             ▼                  ▼
              ┌──────────────────────┐  ┌──────────────────────┐
              │  run_experiment.py   │  │ experiment_watchdog  │
              │                     │  │         .py          │
              │  写文件 ─────────────┼─▶│  读文件（只读扫描）    │
              │  steps_v2.jsonl      │  │  每 30s poll         │
              │  summary_v2.json     │  │                      │
              │                     │◀─┼─ 删除 error episode   │
              │  （runner 在 retry   │  │  （watchdog 删除后    │
              │   pass 中重跑被删的） │  │   runner 下次重跑）   │
              │  condition_summary   │  │  触发 analysis 脚本   │
              └──────────────────────┘  └──────────────────────┘
                        │  写                        │  写（分析结果）
                        ▼                            ▼
              ┌─────────────────────────────────────────────────┐
              │              文件系统（唯一共享状态）               │
              │                                                 │
              │  results/<run_id>/                              │
              │    <cid>/episodes/*_summary_v2.json  ← 进度标志  │
              │    <cid>/condition_summary_v2.json   ← 完成标志  │
              │    analysis/results/                 ← 分析输出  │
              │    analysis/signals/                 ← 路由信号  │
              │    analysis/digest/digest_*.jsonl    ← GLM 解读  │
              │    analysis/reason_diagnostics/      ← 归因      │
              └─────────────────────────────────────────────────┘
```

**watchdog ↔ runner 的隐式协作**：watchdog 删除 error episode 的文件 → runner 的 retry pass 检测到 summary 缺失 → runner 在同一次运行内重跑该 task。这是两个进程唯一的"写-读"双向交互。

---

## 四、一个 condition 的完整生命周期

以 B1 reddit DOM 为例：

```
queue 开始跑 DOM condition
  │
  ├─ 生成 /tmp/b1_3mode_reddit_dom_$$.yaml（include_sites=["reddit"], observation_mode=["dom"]）
  ├─ start_exp_watchdog(reddit) 已在运行，跳过
  ├─ is_condition_complete? → NO（condition_summary_v2.json 不存在）
  │
  ├─ attempt=1: nohup run_experiment.py --config ... --run_id B1_3mode_reddit_20260413 &
  │              job_pid=XXXXX
  │
  │   ── runner 进程开始 ──────────────────────────────────────────────────
  │   runner: load_experiment_config → normalize_config → 合并默认值
  │   runner: generate_conditions() → [ConditionSpec(phase1_dom_router_0)]
  │   runner: _cleanup_stale_runs()（P79_DISABLE_STALE_CLEANUP=1 时跳过）
  │   runner: _write_run_meta()
  │   runner: _get_backend("local_4b") → LocalQwenBackend.__init__（加载模型）
  │   runner: resume 扫描：哪些 task 已有 summary → 跳过
  │   runner: [task_1] _run_and_record_episode()
  │             └─ _run_episode()（见第五节）
  │             └─ 写 reddit_task_1_steps_v2.jsonl（逐行 fsync）
  │             └─ 写 reddit_task_1_summary_v2.json        ← watchdog 发现
  │   runner: [task_2..N] ...
  │   runner: retry pass：检查有无 summary 被 watchdog 删除 → 重跑
  │   runner: aggregate_condition_metrics() → 写 condition_summary_v2.json ← watchdog 发现
  │   runner: _run_post_condition_analysis() → subprocess analyze_experiment.py
  │   runner: 写 run_summary_v2.json
  │   runner: 进程正常退出
  │   ── runner 进程结束 ──────────────────────────────────────────────────
  │
  │   ── queue 内层 shell watchdog 同时在跑 ────────────────────────────────
  │   每 60s: cur = count_episode_summaries(run_dir)
  │   有增长 → stale_secs=0（正常不触发 kill）
  │   ── ─────────────────────────────────────────────────────────────────
  │
  │   ── watchdog 进程同时在跑 ─────────────────────────────────────────────
  │   每 30s: 扫描 summary → [dom] task=X OK succ=Y/Z
  │   每 30min: digest + gallery + annotate → ntfy 状态报告
  │   发现 condition_summary → 触发 analyze_experiment + cross_rep + confidence
  │                          → 发 ntfy POST-ANALYSIS 通知
  │   ── ─────────────────────────────────────────────────────────────────
  │
  ├─ queue: wait job_pid → 退出
  ├─ queue: is_condition_complete? → YES → 跳出 while 循环
  │
  ├─ reset_vwa_sites "reddit" → SSH → quark@100.95.81.103 → PowerShell reset_vwa.ps1
  ├─ refresh_site_auth "reddit" → Playwright 重登录 → 写 .auth/reddit_state.json
  │
  └─ 进入下一个 condition（SOM）...
```

---

## 五、Runner 内部：单个 Episode 的执行流程

`_run_episode()`（`p79/experiment/runner.py:737~`）是最核心的方法，完整流程如下：

### 5.1 初始化

- 清理旧 artifacts 目录（如果是 resume，先删旧的再重跑）
- 清理旧 steps JSONL（通过 `io_utils.read_jsonl_dedup` 支持 restart dedup：JSONL 被追加时，检测 `step_idx==0` 重置点，只保留最后一次 run 的步骤）
- env reset，初始化 trajectory、router_state、action_signatures

### 5.2 Step 循环（最多 max_steps=30 步）

每一步按以下顺序执行：

**① Busy page 保护**
- 检测 obs_text 中是否包含 `"busy: 1"`（VWA 页面正在加载）
- 如果是，发送免费的 `wait` action，**不消耗 step budget**，最多 `busy_wait_limit=5` 次

**② 观测准备 + Router 决策**
- 用当前 observation_mode 调用 `prepare_observation_for_mode()`，生成 SomResult
- SomResult 包含：`som_text`（DOM/SoM 标注文本）、`marked_image`（裸截图或带框截图）
- Router.decide()：根据 DOM 大小、停滞步数等信号决定是否 escalate
  - dom_size_threshold=12000 chars → 超过则 escalate 到 hybrid 模式
  - unchanged_steps_trigger=2 → 连续 2 步页面不变则 escalate
- 如果 router 触发 escalation，重新准备 som 观测
- 保存 artifacts（DOM text、SoM 图、截图）

**③ Backend 调用**
- 构建 `BackendStepContext`：
  ```
  observation_mode, som_enabled, som_text, stage,
  planner_sub_goal, history（最近 8 steps）,
  module_flags, reference_images
  ```
- 如果 M4 启用：先调用 planner stage（注入 `[Stage: planner]` prefix），再调用 grounder stage
- 否则：单次调用 single stage
- 返回 `(action, meta)`，meta 含 token 数、推理耗时、logprob 信号等

**④ Action 执行 + 页面状态检测**
- 执行 action，等待 `sleep_after_execution=0.5s`
- `detect_page_state_change()`：计算新旧页面 similarity（阈值 0.95）
- 如果检测到 `about:blank`（页面跳转异常）：自动执行 `go_back` 恢复

**⑤ M3 重试**
- `should_trigger_m3_retry()`：检测 action 无效（如点击未找到元素）
- 如果触发，执行 retry_action，采用新状态继续

**⑥ Cycle / 卡死检测（会导致 early stop）**
- **Strict cycle**：最近 N steps action_signature 出现 ≥3 次完整循环 → early stop
- **Soft cycle**：忽略 scroll_up/down 后，action_signature 循环 ≥4 次 → early stop
- **Scroll alternation**：连续 6 次 up/down 交替 → early stop
- **URL stuck**：连续 5 次 click 同 URL → early stop

**⑦ 成本计算 + 日志写入**
- `token_cost = compute_token_cost(input_tokens, output_tokens, cost_cfg)`
- `router_overhead_cost = router_decision_ms × overhead_cost_per_ms + extra_model_calls × extra_call_cost + retries × retry_cost`
- `total_cost = token_cost + router_overhead_cost`
- 构建 StepRecordV2，写入 JSONL（**每行 fsync**，保证崩溃安全）

### 5.3 Episode 收尾

- 在 trajectory 末尾自动补一个 `stop("")` action（VWA evaluator 需要）
- 调用 `evaluator.evaluate(trajectory, config_file, env)`
- **Reward override**：如果 evaluator 返回 score=0，但 env 自身 reward>0 且 agent 确实提交了 finish/stop，覆盖为 score=1.0（处理评估器与环境不一致的情况）
- 汇总 episode_summary（steps、success、reason、cost、tokens、wasted_cost、component_breakdown）

---

## 六、Condition 生成规则（conditions.py）

`generate_conditions(cfg)` 根据 `experiment.phase` 字段生成 condition 列表：

### Phase1（当前 B0/B1 使用的）
每种 `observation_mode` 生成一个 condition，固定 router_on=False：
```
phase1_dom_router_0    observation_mode=dom,    som_on=False, router_on=False
phase1_som_router_0    observation_mode=som,    som_on=True,  router_on=False
phase1_vision_router_0 observation_mode=vision, som_on=False, router_on=False
```
> **注意**：Phase1 是 3-mode flat design（dom/som/vision），router 固定关闭。不是 2×2 grid。

### Phase2（路由研究，待跑）
- `phase2_fixed_best`：从 phase1 的 `run_summary_v2.json` 自动读取最佳 condition（按 success→cost→latency 排序）
- `phase2_routed`：用 `cheap_default_mode=dom`，`router_on=True`

### Phase3（模块消融，待跑）
```
phase3_none / phase3_m1 / phase3_m2 / phase3_m3 / phase3_m4
```
每个 condition 只开对应模块，其余关闭。

### B0 Baseline（由 conditions.py 生成，非独立脚本）
当 `baselines.run_b0=true` 时，额外生成：
```
b0_strong_upper_bound   backend_id=api_strong（235B）
```
> **注意**：当前 B0/B1 的 config 均设 `run_b0: false`，即不走此路径。B0 实验实际上是用 phase1 的条件（`phase1_{mode}_router_0`）+ api_proxy backend，而非 b0_strong_upper_bound 条件。

---

## 七、Config 加载流程（config.py）

```
load_experiment_config(config_path)
  ├─ 加载 YAML
  ├─ 递归加载 defaults[] 列表（先加载 base，再用覆盖层 merge）
  ├─ _merge_dict()：深度合并，子实验 config 的值覆盖 base 的值
  └─ normalize_config()：补全所有字段默认值
```

重要默认值（normalize_config 中补全）：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `runtime.max_steps` | 40 | B1 config 中覆盖为 30 |
| `runtime.busy_wait_limit` | 5 | busy page 保护最大等待次数 |
| `runtime.baseline_retry_on_no_progress` | False | M3 相关，默认不重试 |
| `state_change.similarity_threshold` | 0.95 | 页面变化检测阈值 |
| `router.dom_size_threshold` | 12000 | DOM 文本超长触发 escalation |
| `metrics.energy.enabled` | True（base）/ False（B0 override） | 能耗追踪 |

VWA 站点 URL 通过占位符注入（`tasks.py` 中替换）：
```
__REDDIT__       → env var REDDIT       或 http://localhost:9999
__SHOPPING__     → env var SHOPPING     或 http://localhost:7770
__CLASSIFIEDS__  → env var CLASSIFIEDS  或 http://localhost:9980
```
`vwa_env_remote.sh` 设置这些环境变量为 `http://100.95.81.103:{port}`。

---

## 八、Backend 与 Agent 对应关系

`p79/backends/factory.py` 根据 config 中的 `type` 字段分发：

| `type` 值 | Backend 类 | Agent 类 | 用途 |
|-----------|-----------|---------|------|
| `local_qwen` | `LocalQwenBackend` | `Qwen3VLAgent` | B1：本地 4B GPU 推理 |
| `api_proxy` | `ApiProxyBackend` | `ProxyApiAgent` | B0：HTTP 代理 235B API |
| `heuristic_dom` | `HeuristicDomBackend` | 无（规则 action） | 测试/baseline |
| `mock` | `MockBackend` | 无 | 单元测试 |

**LocalQwenBackend 特殊行为**：
- `mock_mode=True` 时跳过模型加载（用于测试）
- `dom_mode="heuristic_only"` 时跳过 LLM，直接用启发式规则生成 action

**ProxyApiAgent 与 Qwen3VLAgent 差异**：
- ProxyApiAgent 无 M4 两阶段支持（只有 single stage）
- ProxyApiAgent 通过环境变量 `PROXY_API_KEY` 认证，endpoint 在 config 的 `base_url` 字段
- Qwen3VLAgent 支持 M4 planner/grounder 两阶段

---

## 九、日志文件与写入机制（LoggerV2）

`p79/experiment/logger_v2.py` 写入的文件：

| 文件 | 写入时机 | 内容 |
|------|---------|------|
| `condition_meta.json` | condition 开始时 | 配置快照（mode、backend、modules 等） |
| `episodes/<site>_task_<id>_steps_v2.jsonl` | 每步完成后 append | StepRecordV2（含 action、obs、cost、tokens、logprob 等） |
| `episodes/<site>_task_<id>_summary_v2.json` | episode 完成时 | 汇总（success、steps、cost、reason、wasted_cost 等） |
| `condition_summary_v2.json` | 所有 task 完成后 | 聚合指标（SR、平均 cost/steps/tokens 等）|

**fsync 机制**：JSONL 每行写入后都执行 fsync，保证进程崩溃时已写的步骤不丢失。这是 resume 能从断点继续的基础。

**restart dedup**（`io_utils.py`）：JSONL 被多次 append 时（每次 resume 都 append），通过检测 `step_idx==0` 重置点识别新 run，`read_jsonl_dedup` 只返回最后一次 run 的步骤，避免重复计数。

---

## 十、异常恢复机制

### 10.1 Runner 卡死（最常见）

```
情况：task 卡在某步（网络超时、VWA env 无响应）
检测：queue 内层 shell watchdog，35min 无新 episode
处理：queue kill runner → sleep 15s → 重新启动 runner（resume=true 跳过已有 summary）
```

### 10.2 Error 类 episode

```
情况：runner 写出一个 error summary（code_bug 或 benchmark_noise）
检测：watchdog 每 30s 扫描，见到 summary.error 字段
处理：watchdog 删除 summary + steps + artifacts（+ 清理 digest 记录）
      runner retry pass：发现 summary 缺失 → 同次运行内重跑
      code_bug 最多重试 2 次，benchmark_noise 无限重试
      超过重试上限 → watchdog 发 ntfy PERSISTENT_ERROR 告警，保留 episode
```

### 10.3 Session 丢失（站点登录状态失效）

```
情况：VWA session 过期，agent 在未登录状态下跑 task
检测：watchdog 检查 step_000 的 DOM，连续 3 个 task 无 logout 链接
处理：发 ntfy urgent 告警 + 自动 Playwright 重登录刷新 cookies
      登录恢复后，自动删除所有"未登录状态下跑的"污染 episode → runner 重跑
```

### 10.4 Watchdog 崩溃重启

```
情况：watchdog 进程异常退出
处理：queue 重新启动时会 kill 同 run_dir 的遗留 watchdog 再启动新实例
      或手动 bash scripts/dgx/restart_watchdog.sh（热重启，保留 .state.json）
      watchdog 通过 .state.json 恢复：不重复触发已做过的分析，不重复发告警
```

### 10.5 Runner 早停（cycle/卡死检测）

```
情况：agent 陷入重复动作循环（见第五节 5.2.⑥）
检测：runner 内部 cycle detector（action_signature 比较）
处理：early stop，写 reason="cycle_early_stop" 或 "scroll_alternation" 等到 summary
      不触发 watchdog 重试（非 error，是正常 fail）
```

---

## 十一、B0 实际调用链

```
run_b0_3mode_classifieds.sh
  ├─ 加载 .auth/qwen_api → export PROXY_API_KEY/QWEN_API_KEY/DASHSCOPE_API_KEY
  ├─ 读 configs/exp_v2_B0_3mode_classifieds.yaml
  │    └─ backends.default_backend = "api_strong"
  │         type: "api_proxy", model: qwen3-vl-235b-a22b
  │         max_new_tokens: 4096, image_max_size: 1280
  │
  ├─ 生成三份临时 config（/tmp/b0_3mode_{dom|som|vision}_$$.yaml）
  │    通过 Python regex 只替换 observation_mode 字段为单模式
  │
  ├─ start_watchdog() → nohup experiment_watchdog.py
  │    --run-dir OUTPUT_DIR --poll-secs 30 --idle-alert-mins 20
  │    --glm-config .auth/glm --digest-dir OUTPUT_DIR/analysis/digest
  │    --aggregate-prefix B0_3mode --notify-completion
  │    （单实例监控三个 condition，B0 三模式共享同一 OUTPUT_DIR）
  │
  ├─ [1/3] run_until_complete "dom" → cid=phase1_dom_router_0
  │    └─ nohup run_experiment.py --config /tmp/dom.yaml --run_id B0_3mode_classifieds_20260413
  │         └─ ExperimentRunner
  │              ├─ generate_conditions() → [phase1_dom_router_0]
  │              ├─ _get_backend("api_strong")
  │              │    └─ factory → ApiProxyBackend (backends/api_proxy.py)
  │              │         └─ ProxyApiAgent (agents/proxy_api_agent.py) [HTTP → 235B]
  │              ├─ runner 主循环: condition→task→step（见第五节）
  │              └─ condition 完成 → 写 condition_summary + 调用 analyze_experiment.py
  │
  ├─ reset_vwa_sites "classifieds" + refresh_classifieds_auth（Playwright）
  ├─ [2/3] run_until_complete "som"    → cid=phase1_som_router_0
  ├─ reset_vwa_sites "classifieds" + refresh_classifieds_auth
  ├─ [3/3] run_until_complete "vision" → cid=phase1_vision_router_0
  ├─ stop_watchdog()
  └─ run_reason_diagnostics → analyze_reason_diagnostics.py --report --report-language zh
```

---

## 十二、B1 实际调用链

```
queue_b1_with_reset.sh
  ├─ 读 configs/exp_v2_qwen3vl4b_B1_baseline.yaml
  │    └─ backends.default_backend = "local_4b"
  │         type: "local_qwen", model: Qwen/Qwen3-VL-4B-Instruct (本地 GPU)
  │         max_new_tokens: 384（B1 覆盖值，base 默认 512）
  │
  ├─ start_gallery_server() → python3 -m http.server 8765 --directory results/…/phase1
  │
  ├─ [站点1] run_site_3mode_with_reset "reddit" RUN_ID_REDDIT
  │    ├─ 生成三份临时 config（Python regex 替换 include_sites + observation_mode）
  │    ├─ start_exp_watchdog(reddit) → nohup experiment_watchdog.py（同 B0 逻辑）
  │    ├─ [1/3] run_condition_until_complete dom → phase1_dom_router_0
  │    │    └─ nohup run_experiment.py --config tmp_dom.yaml --run_id B1_3mode_reddit_20260413
  │    │         └─ ExperimentRunner
  │    │              ├─ generate_conditions() → [phase1_dom_router_0]
  │    │              ├─ _get_backend("local_4b")
  │    │              │    └─ factory → LocalQwenBackend (backends/local_qwen.py)
  │    │              │         └─ Qwen3VLAgent (agents/qwen3vl_agent.py) [本地 4B]
  │    │              └─ runner 主循环（同 B0，见第五节）
  │    ├─ reset_vwa_sites "reddit" + refresh_site_auth "reddit"（Playwright）
  │    ├─ [2/3] run_condition_until_complete som    → phase1_som_router_0
  │    ├─ reset_vwa_sites "reddit" + refresh_site_auth "reddit"
  │    ├─ [3/3] run_condition_until_complete vision → phase1_vision_router_0
  │    ├─ stop_exp_watchdog()
  │    └─ run_reason_diagnostics(reddit_run_dir) → analyze_reason_diagnostics.py --report
  │
  ├─ sleep 15
  │
  └─ [站点2] run_site_3mode_with_reset "shopping" RUN_ID_SHOPPING
       └─ （同 reddit，site 换为 shopping）
```

---

## 十三、B0 vs B1 关键差异对比

| 维度 | B0 | B1 |
|------|----|----|
| **模型** | Qwen3-VL-235B（API，HTTP） | Qwen3-VL-4B（本地 GPU） |
| **Backend 链路** | `ApiProxyBackend` → `ProxyApiAgent` | `LocalQwenBackend` → `Qwen3VLAgent` |
| **max_new_tokens** | 4096 | 384（B1 config 覆盖）|
| **image_max_size** | 1280px | 默认（base config） |
| **站点** | classifieds（单站） | reddit → shopping（两站，串行） |
| **RUN_ID** | 三模式共享一个 RUN_ID | 每站独立 RUN_ID |
| **条件间 reset** | dom→som→vision 各 reset 一次 | 同上，每站内各 reset |
| **Watchdog 实例** | 单实例（三模式共享） | 每站一个（reddit/shopping 分开） |
| **结果目录** | `results/…/B0_3mode_classifieds_20260413/` | `results/…/B1_3mode_{reddit\|shopping}_*/` |
| **Gallery prefix** | `B0_3mode` | `B1_3mode` |
| **API key** | `.auth/qwen_api`（rp_ 开头） | 无需（本地 GPU） |
| **能耗追踪** | 关闭（config 覆盖 `energy.enabled: false`） | 开启（`use_pynvml: true`，dgx_spark） |
| **M4 支持** | 否（ProxyApiAgent 无两阶段） | 是（Qwen3VLAgent 支持 planner/grounder） |

---

## 十四、相关脚本文件索引

| 脚本 | 路径 | 作用 |
|------|------|------|
| B1 总调度 | `scripts/dgx/queue_b1_with_reset.sh` | B1 实验入口，管理站点顺序和 condition 重启 |
| B0 总调度 | `scripts/dgx/run_b0_3mode_classifieds.sh` | B0 实验入口 |
| 实验执行 | `scripts/run_experiment.py` | Runner 入口，委托给 `p79/cli/run_experiment.py` |
| Runner 核心 | `p79/experiment/runner.py` | 主编排器：condition→task→step 循环，含 cycle 检测、reward override 等 |
| Watchdog | `scripts/experiment_watchdog.py` | 监控、分析触发、error 重试、session 检测 |
| Watchdog 重启 | `scripts/dgx/restart_watchdog.sh` | 热重启 watchdog（保留 .state.json） |
| 站点重置 | `scripts/reset_vwa_sites.sh` | SSH → quark@100.95.81.103 → PowerShell C:\vwa\reset_vwa.ps1 |
| Backend 工厂 | `p79/backends/factory.py` | 根据 `type` 字段创建 LocalQwenBackend / ApiProxyBackend 等 |
| B1 Backend | `p79/backends/local_qwen.py` | 本地 4B，__init__ 时加载模型 |
| B1 Agent | `p79/agents/qwen3vl_agent.py` | 4B 推理，支持 M4 两阶段 |
| B0 Backend | `p79/backends/api_proxy.py` | HTTP 代理 235B API |
| B0 Agent | `p79/agents/proxy_api_agent.py` | HTTP 调用，无 M4 两阶段 |
| 条件生成 | `p79/experiment/conditions.py` | generate_conditions()，phase1/2/3 + b0_baseline |
| Config 加载 | `p79/experiment/config.py` | YAML merge + normalize_config，补全所有默认值 |
| 日志写入 | `p79/experiment/logger_v2.py` | JSONL fsync + summary JSON，4 类文件 |
| IO 工具 | `p79/experiment/io_utils.py` | read_jsonl_dedup（restart dedup + corrupt 行处理） |
| 主分析 | `scripts/analysis/analyze_experiment.py` | 每 condition 完成后，输出 `analysis/results/` |
| 置信度分析 | `scripts/analysis/analyze_confidence_calibration.py` | 路由信号，输出 `analysis/signals/` |
| 交叉分析 | `scripts/analysis/analyze_cross_representation.py` | ≥2 conditions，含 visual FP 过滤 |
| 归因分析 | `scripts/analysis/analyze_reason_diagnostics.py` | 失败分类，`--report` 生成中文报告 |
| GLM Digest | `scripts/glm_batch_digest.py` | 对失败 episode 做 GLM 批量解读 |
| Gallery 生成 | `scripts/generate_gallery.py` | 生成 HTML gallery（`--run-dir` 单 run 或 `--phase-dir` aggregate） |
| 截图标注 | `scripts/annotate_screenshots.py` | 给 artifacts 截图加标注 |
| Gallery 刷新 | `scripts/dgx/refresh_gallery.sh` | 手动触发 gallery 重新生成 |

---

## 十五、Analysis 输出目录结构（B0/B1 相同）

```
results/visualwebarena/phase1/<run_id>/
  ├─ run_meta.json                          ← 实验元信息（配置快照、开始时间、log_path）
  ├─ run_summary_v2.json                    ← 全局汇总（SR、phase2 净节省估算等）
  ├─ task_configs/                          ← 本次实验的 task JSON 配置
  ├─ latest_<site> → <run_id>              ← 符号链接，指向最新 run
  │
  ├─ <cid>/                                 ← 例如 phase1_dom_router_0/
  │    ├─ condition_meta.json              ← condition 配置（observation_mode 等）
  │    ├─ condition_summary_v2.json        ← 完成标志，由 runner 写入
  │    ├─ episodes/
  │    │    ├─ <site>_task_<id>_steps_v2.jsonl   ← 每步记录（逐行 fsync）
  │    │    └─ <site>_task_<id>_summary_v2.json  ← episode 摘要（成功/失败/cost）
  │    └─ artifacts/
  │         └─ <site>_task_<id>/
  │              ├─ step_000/
  │              │    ├─ observation_dom.txt
  │              │    ├─ observation_som.txt
  │              │    ├─ screenshot.png
  │              │    └─ screenshot_som.png
  │              └─ step_001/ ...
  │
  └─ analysis/
       ├─ results/                          ← analyze_experiment.py（CSV/JSON 表格）
       │    ├─ _overview/tables/condition_metrics.csv
       │    └─ cross_representation/cross_representation_summary.json
       ├─ signals/                          ← analyze_confidence_calibration.py
       │    └─ combined/confidence_summary.json
       ├─ reason_diagnostics/              ← analyze_reason_diagnostics.py
       │    └─ reason_diagnostics_summary.json
       └─ digest/
            ├─ digest_dom.jsonl            ← GLM 对 DOM 失败 episode 的解读
            ├─ digest_som.jsonl
            └─ digest_vision.jsonl
```

---

## 十六、Watchdog 触发的 analysis 脚本一览

| 触发时机 | 脚本 | 输出位置 |
|---------|------|---------|
| 每 30min 周期 | `analyze_reason_diagnostics.py --skip-similarity` | `analysis/reason_diagnostics/` |
| 每 30min 周期 | `glm_batch_digest.py --max-images 3 --delay-secs 3.0` | `analysis/digest/digest_*.jsonl` |
| 每 30min 周期 | `annotate_screenshots.py` | `<cid>/artifacts/` |
| 每 30min 周期 | `generate_gallery.py`（单 run） | `<run_id>/gallery.html` |
| 每 30min 周期 | `generate_gallery.py --phase-dir --prefix` | `B{0\|1}_3mode/gallery.html` |
| condition 完成 | `analyze_experiment.py --run_dir` | `analysis/results/` |
| condition 完成 | `analyze_confidence_calibration.py --run-dir` | `analysis/signals/` |
| condition 完成（≥2） | `analyze_cross_representation.py --priority all` | `analysis/results/cross_representation/` |
| condition 完成 | `annotate_screenshots.py` + `generate_gallery.py` | 同上 |
| 实验全部完成（shell） | `analyze_reason_diagnostics.py --report --report-language zh --samples-per-bucket 5` | `analysis/reason_diagnostics/` |

**注意**：`analyze_experiment.py` 会被**双重触发**——runner 完成 condition 时调用一次（subprocess），watchdog 检测到 condition_summary 时再调用一次。两次调用均幂等（覆盖写），不产生冲突。
