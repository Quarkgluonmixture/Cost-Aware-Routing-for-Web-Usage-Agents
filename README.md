# P79: Cost-Aware Routing for Web Usage Agents

研究 cost-aware routing 能否改善 web agent 的成功率-效率权衡。基于 VisualWebArena 三站点（shopping / reddit / classifieds），使用 Qwen3-VL-4B (B1 local) + Qwen3-VL-235B (B0 API proxy) 双 baseline。

**Paper hook (毕设)**: Phantom-SoM 是 SoM-style web agent 的隐藏第 4 个 routing arm，具备 **4-fold drop-in property**：
- (a) cost ≈ DOM（regex filter 同一 AXTree, 无需 bbox / image）
- (b) latency ~50% 更低（cls SoM p95 74s → Phantom-SoM 18.2s = 4× 更快）
- (c) signal AUROC ≥ baseline（5-mode 全 `overall_usable=True`）
- (d) drop-one oracle 1.7-3.3pp（red Phantom-SoM 3.33pp ≥ SoM 1.90pp）

完整 paper 论证 + theory framework (3-axis × 8-channel) + section status 见 `docs/checkpoints/paper_planning.md`。

## 快速开始

```bash
# Clone（含 visualwebarena fork submodule）
git clone --recursive <repo-url> Cost-Aware-Routing-for-Web-Usage-Agents
cd Cost-Aware-Routing-for-Web-Usage-Agents
# 或已 clone 过：
git submodule update --init --recursive

# 安装
pip install -e .                    # 最小依赖
pip install -e ".[analysis,dev]"    # 全功能（含 scipy/pandas/matplotlib）

# 环境预检（CUDA + VWA 站点 + 认证）
bash scripts/preflight_v2.sh

# 实验入口（推荐 Makefile）
make help                                          # 列所有 targets
make smoke                                         # smoke + integration 测试
make phantom B=B0 M=dom S=red                      # 单 condition phantom 实验
make analyze RUN=results/visualwebarena/phase1/<RUN_ID>  # 全分析管线
make compare B0=<b0_run> B1=<b1_run> SITE=classifieds    # B0 vs B1 对比

# 底层命令
python3 scripts/run_experiment.py --config configs/exp_v2_B0_dom_classifieds.yaml   # per-condition yaml (master phase1.yaml retired B-232 2026-05-16)
python3 scripts/analysis/analyze_experiment.py --run_dir results/visualwebarena/phase1/<RUN_ID>
pytest tests/                                      # 81 测试
```

## 文档分层（4-doc separation of concerns）

| 文档 | 用途 | 更新频率 |
|---|---|---|
| `docs/checkpoints/next_steps.md` | **Action ledger** — active processes / next 3 actions / codex queue / paper section status | Daily |
| `docs/checkpoints/paper_planning.md` | **Paper strategy notebook** — theory framework / findings 列表 / risks / cascade / router / advisor align / reviewer 预案 / decision log（19 sections） | Weekly |
| `docs/analysis/paper_drafts/` | **Final paper prose** — `section1_intro.md` ... `section8_discussion.md` + `paper.bib` | 每 codex round |
| `docs/checkpoints/实验笔记.md` | **Time-order chronicle** — §1-§104 append-only history | Append-only |

新数据/结果该更新哪些文档详见 `next_steps.md` §10 (Doc Update Workflow)。

## 三阶段实验设计

- **Phase 1** — 表征筛选：5-mode flat (`dom` / `som` / `vision` / `phantom_text` / `phantom_som`)，per site per model
- **Phase 2** — 路由研究：Tier 1 oracle router (TF-IDF + LR) + Tier 2 first-step trigger router（Phantom-SoM 是 router 第 4 arm，同一篇 paper）
- **Phase 3** — 模块消融：M1(select fallback) / M2(input fallback) / M3(retry) / M4(two-stage)（未启动）

## 当前实验状态（2026-04-28）

| Cell | Status |
|---|---|
| B0 VWA cls + red 5-mode FRESH paper-grade clean | ✅ done (Critical path A B0 部分) |
| B0 VWA shopping DOM pilot | 🟡 跑中 ~9h ETA |
| B0 VWA shopping {SoM, Vision, P-text, Phantom-SoM} | ⏳ 待 DOM pilot 验证 |
| B1 VWA cls phantom_som | 🟡 跑中 ~7-10d ETA (GPU contention) |
| B1 VWA red phantom + B1 shopping 5-mode | ⏳ chain after B1 cls done |
| B0/B1 WA 480 tasks | ⏳ Week 4-5 cross-bench generalization |

详细 active processes / codex queue / pending cells 见 `next_steps.md`。

## 关键变量

- **A1 Observation Mode** (5-mode flat):
  - `dom` — viewport-only AXTree
  - `som` — `[SOM_MARKS]` text + 带框截图
  - `vision` — 裸截图
  - `phantom_text` (P-text; legacy mode value `phantom_dom` still accepted) — DOM prompt + `[SOM_MARKS]` text, 无图（control for prompt vs image effect）
  - `phantom_som` — SoM prompt + `[SOM_MARKS]` text, 无图（hidden 4th routing arm）
- **B1 Router**: off (固定策略) / on (规则路由 / oracle / learned) — Phase 2
- **M1-M4**: 二级模块，Phase 3 逐一消融

## 代码结构

```
p79/
├── agents/           # LLM 推理（qwen3vl_agent=B1 本地, proxy_api_agent=B0 API）
├── backends/         # Backend 抽象层（local_qwen / api_proxy / heuristic + factory）
├── envs/             # VWA 环境封装（P79Observation 标准化 + viewport 过滤）
├── experiment/       # 核心实验引擎
│   ├── runner/       # 主编排器 condition→seed→task→step
│   ├── router.py     # 规则路由器
│   ├── conditions.py # Phase1/2/3 条件生成（5-mode 含 phantom）
│   ├── modules.py    # M1-M4 辅助模块
│   ├── som.py        # Set-of-Marks 标注 + `[SOM_MARKS]` 文本提取
│   ├── metrics.py    # 成本 / 延迟 / 能耗聚合
│   ├── logger_v2.py  # JSONL 结构化日志（fsync 持久化）
│   ├── io_utils.py   # JSONL 读取 + restart dedup
│   ├── analysis.py   # adjusted_success canonical + Pareto + analyze_run
│   └── ...
├── utils/            # auth_refresh / CUDA workaround / asyncio
└── cli/              # CLI 入口
configs/              # YAML 实验配置（base + phase1-3 + B0/B1/WA per-site + phantom 5-mode）
external/visualwebarena/  # ⭐ Submodule → P79 fork (Quarkgluonmixture/visualwebarena, p79-patches branch)
                          # 含 viewport ratio bug fix / DGX host-resolver / NumPy 2.0 float casts
scripts/
├── analysis/         # 数据分析（analyze_*, compare_b0_b1, validate_run）
├── analysis/figures/ # 论文 figure 生成（fig1-9，输出到 results/phantom_paper/figures/）
├── queues/           # 实验队列（queue_b{0,1}{,_wa}_with_reset.sh, queue_phantom_pair.sh）
├── maintenance/      # experiment_watchdog / rederive / clear_tasks / annotate / gallery
├── vwa/              # VWA 站点 setup
└── *.sh / run_experiment.py / preflight_v2.sh
tests/                # 81 测试（含 test_runner_smoke invariants）
```

## 结果目录

```
results/<benchmark>/<phase>/<run_id>/
├── <condition_id>/
│   ├── episodes/         # 每 episode 的 steps JSONL + summary JSON
│   ├── artifacts/        # 截图、DOM、SoM 图、标注截图
│   ├── condition_meta.json
│   └── condition_summary_v2.json
├── analysis/             # 分析输出（reason_diagnostics、digest、confidence）
└── gallery.html          # HTML 画廊
results/phantom_paper/figures/   # paper figure 输出（gitignored, 由 scripts/analysis/figures/*.py 生成）
```

## DGX Spark 注意事项

- 用 `python3` 或 `.venv/bin/python`，不要用 `python`（可能不存在）
- 必须设置 `PYTORCH_NVML_BASED_CUDA_CHECK=1` + `CUDA_MPS_PIPE_DIRECTORY=""`（脚本已自动处理）
- GB10 `sm_121` 架构可能触发 nvrtc 错误，仓库内置 `torch.prod` fallback (`p79/utils/torch_cuda_workarounds.py`)
- 远程站点配置见 `scripts/vwa_env_remote.sh`（不入版本管理；`VWA_REMOTE_HOST=100.95.81.103`）
- Shopping base_url 必须配为 DGX 可达 IP（非 localhost），否则 Magento 302 回环
- 详细见 `DGX_SPARK_MACHINE_QUIRKS.md`

## Onboarding new host (Myriad / future GPU machines)

`git clone --recursive` + `pip install -e .` 之后还要本地准备 6 项（不入 git，per-host）：

1. **`.env`** — API keys (`OPENAI_API_KEY` for VWA eval / `DASHSCOPE_API_KEY` for B0 proxy / `P79_GLM_KEY` optional for digest)
2. **`scripts/vwa_env_remote.sh`** — 15 行 BASE_URL（template 见 `DGX_SPARK_MACHINE_QUIRKS.md`）
   ```bash
   export VWA_REMOTE_HOST=100.95.81.103   # quark Tailscale IP
   export CLASSIFIEDS=http://${VWA_REMOTE_HOST}:9980
   export REDDIT=http://${VWA_REMOTE_HOST}:9999
   export SHOPPING=http://${VWA_REMOTE_HOST}:7770
   export SHOPPING_ADMIN=http://${VWA_REMOTE_HOST}:7780
   export WIKIPEDIA=http://${VWA_REMOTE_HOST}:8888
   export HOMEPAGE=http://${VWA_REMOTE_HOST}:4399
   ```
3. **VWA 站点 access** — 选 A：Tailscale 加入网内（reach quark `100.95.81.103`）；选 B：本机起 docker compose（需 sudo + ~3GB Wikipedia zim + ~81MB classifieds compose，按 VWA upstream README）
4. **`.auth/`** — Playwright session state，`cd external/visualwebarena && bash prepare.sh` 生成（依赖 VWA 可达）
5. **GPU torch CUDA build**：
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu128 \
       torch==2.11.0+cu128 torchvision==0.26.0+cu128
   python3 -c "import torch; print(torch.cuda.is_available())"   # must be True
   ```
6. **Shell env** — 加到 `~/.bashrc` 或 conda activate hook：
   ```bash
   export PYTORCH_NVML_BASED_CUDA_CHECK=1
   export CUDA_MPS_PIPE_DIRECTORY=""
   export CUDA_MPS_LOG_DIRECTORY=""
   ```

**已知 hardcoded `100.95.81.103` (quark)** — 9 处 .py/.sh 直写。Myriad 若直接 reach 此 IP（Tailscale）则无需改。否则临时 `sed` 替换或等 IP env-var-ize 重构（next_steps §5 backlog）。

## Cross-host results sync (hub-spoke)

数据画像：单 condition run **~1.7GB**（artifacts ~100MB / **JSONL 仅 ~15MB**）。3 层 sync 策略（DGX = hub）：

| Tier | 内容 | 大小/cell | 策略 |
|---|---|---|---|
| A. Summary | `condition_summary_v2.json` + `run_meta.json` | KB-MB | 重要数据，可 commit candidate（待评估，大概率不） |
| B. Episodes JSONL + analysis | `episodes/*.jsonl` + `analysis/**` | ~15MB | 跑完即 rsync 到 hub（默认行为） |
| C. Artifacts | screenshots / SoM 图 | 100MB-1GB | 留本地，paper figure 重生时按需拉 |

```bash
# Spoke (Myriad) → Hub (DGX): 推 Tier B（默认无 artifacts）
make rsync-to-hub                                 # HOST=spark-9ea3 (default)
DRY=1 make rsync-to-hub HOST=jiaming@spark-9ea3   # 预演

# Hub (DGX) → 拉别 host 跑出来的数据
make rsync-from-hub                               # 默认 Tier B
make rsync-from-hub RUN=B1_phantom_classifieds_20260428          # narrow 1 run
make rsync-artifacts-from-hub RUN=... COND=phase1_phantom_som_router_0   # 包含 artifacts
```

底层脚本 `scripts/maintenance/rsync_results_{to,from}_hub.sh` 支持 `HOST` / `HUB_PATH` / `RUN` / `COND` / `ARTIFACTS` / `DRY` env vars。

## 实验启动 hard rules（paper-grade 不可违反）

1. **同 site 同时只能跑一个 baseline (B0 XOR B1)** — 否则共享 user account / cart / session → cross-contam
2. **跑实验必须 reset 站点** — 用 `RESET_BEFORE=1 bash scripts/queues/queue_baseline.sh ...` 或 `queue_phantom_{som,text}.sh`
3. **禁止裸用 `python scripts/run_experiment.py`** — 必须走 queue script (`queue_baseline.sh` / `queue_phantom_som.sh` / `queue_phantom_text.sh`)。Queue 处理 reset (race-safe)、env loading、watchdog 启动、idempotent skip。裸 runner 实证导致 contamination（04-28 audit）。

## 参考文档

- `docs/checkpoints/next_steps.md` — daily action ledger ⭐
- `docs/checkpoints/paper_planning.md` — paper strategy notebook ⭐
- `docs/analysis/paper_drafts/` — paper section1-8 prose + paper.bib
- `docs/checkpoints/实验笔记.md` — chronicle §1-§104
- `docs/runs_index.md` — results/ 目录索引
- `P79_experimental_scope_rq_variables.md` — 实验范围与研究问题
- `DGX_SPARK_MACHINE_QUIRKS.md` — DGX 机器特化
- `docs/reference/STEP_SCHEMA_V2_OPTIONAL_FIELDS.md` — step schema 可选字段
- `Makefile` — daily 命令一键化（`make help`）
