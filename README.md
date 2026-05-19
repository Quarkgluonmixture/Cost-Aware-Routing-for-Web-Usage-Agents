<div align="center">

# P79 · Cost-Aware Routing for Web Usage Agents

**研究 cost-aware routing 能否改善 web agent 的 success-rate / efficiency 权衡**
基于 VisualWebArena 三站点（shopping / reddit / classifieds） · 3 baseline × 6 mode × 2 site

</div>

---

## ✨ Paper hook

在 *"skip annotated image"* 边界上存在一个 **phantom routing space**，由 3 个 sibling arm 组成 — **P-text** / **P-prompt** / **P-SoM**。其中 P-SoM (deployment representative) 展现 **4-fold drop-in property**：

| 维度 | 性质 | 数据 |
|---|---|---|
| (a) **Cost** | 与 DOM 持平 | regex filter 同一 AXTree，无 bbox / image |
| (b) **Latency** | 比 full SoM 低约 50% | cls SoM p95 74s → P-SoM 18.2s (~4×) |
| (c) **Signal** | AUROC ≥ baseline | 5-mode 全 `overall_usable=True` |
| (d) **Drop-one oracle** | 1.7-3.3 pp | red P-SoM 3.33 pp ≥ SoM 1.90 pp |

> 完整论证、theory framework、router 设计、section status → [`docs/checkpoints/paper_planning.md`](docs/checkpoints/paper_planning.md)

---

## 📑 Contents

- [Quick Start](#-quick-start)
- [Repository Layout](#-repository-layout)
- [Experiment Design](#-experiment-design)
- [Documentation](#-documentation)
- [Runtime Environments](#-runtime-environments)
- [Onboarding a New Host](#-onboarding-a-new-host)
- [Cross-Host Results Sync](#-cross-host-results-sync)
- [Hard Rules (Paper-Grade)](#-hard-rules-paper-grade)
- [References](#-references)

---

## 🚀 Quick Start

```bash
# 1. Clone (含 VisualWebArena fork submodule)
git clone --recursive <repo-url> Cost-Aware-Routing-for-Web-Usage-Agents
cd Cost-Aware-Routing-for-Web-Usage-Agents
# 或已 clone：
git submodule update --init --recursive

# 2. Install
pip install -e .                    # 最小依赖
pip install -e ".[analysis,dev]"    # 全功能 (scipy / pandas / matplotlib)
pip install -e ".[test]"            # 推荐 — make test 需要 pandas/scipy/matplotlib

# 3. Preflight (CUDA + VWA + auth)
bash scripts/preflight_v2.sh

# 4. Run via Makefile (preferred)
make help                                                  # 列所有 targets
make active                                                # 实时 process scan
make smoke                                                 # smoke + integration 测试
make launch BASELINE=B0 SITE=reddit MODE=som               # 一键启动 (reset + watchdog)
make analyze RUN=results/visualwebarena/phase1/<RUN_ID>    # 单 run 分析管线
make analysis                                              # 全管线 (per-run + cross-condition + figures)
make compare B0=<b0_run> B1=<b1_run> SITE=classifieds      # B0 vs B1 对比
```

底层入口（按需使用）：

```bash
python3 scripts/run_experiment.py --config configs/exp_v2_B0_dom_classifieds.yaml
python3 scripts/analysis/analyze_experiment.py --run_dir results/visualwebarena/phase1/<RUN_ID>
pytest tests/                       # 929 test functions / 68 files
```

> ⚠️ **禁止裸用 `python scripts/run_experiment.py`** — 必须走 `scripts/queues/queue_*.sh` 或 `make launch`。Queue 处理 reset (race-safe) / env / watchdog / idempotent skip；裸 runner 已实证导致 paper-grade contamination。

---

## 🗂 Repository Layout

```
p79/
├── agents/              # LLM inference (qwen3vl_agent=B1 local, proxy_api_agent=B0 API)
├── backends/            # Backend 抽象 (local_qwen / api_proxy / heuristic + factory)
├── envs/                # VWA 环境封装 (P79Observation, viewport 过滤)
├── experiment/
│   ├── runner/          # 主编排器 condition → seed → task → step
│   ├── router.py        # 规则路由器 (DOM ↔ hybrid)
│   ├── conditions.py    # Phase 1/2/3 condition 生成
│   ├── modules.py       # M1-M4 辅助模块
│   ├── som.py           # Set-of-Marks 标注 + [SOM_MARKS] 提取
│   ├── metrics.py       # cost / latency / energy 聚合
│   ├── logger_v2.py     # JSONL 结构化日志 (fsync)
│   ├── io_utils.py      # read_jsonl_dedup — 统一入口
│   ├── analysis.py      # adjusted_success canonical + Pareto + analyze_run
│   └── ...
├── mechanistic/         # paper §5 (deferred) — activation patching, layer probes
├── policies/            # routing policies (rule-based + learned, paper-2 substrate)
├── utils/               # auth_refresh / CUDA workaround / asyncio
└── cli/                 # CLI 入口

configs/                 # 119 YAML configs (base + phase1-3 × B0/B1/B2 × 6 mode × site)
external/visualwebarena/ # ⭐ submodule — fork (Quarkgluonmixture/visualwebarena, p79-patches)
                         #    含 viewport ratio bug fix / DGX host resolver / NumPy 2.0 cast
scripts/
├── analysis/            # analyze_*, compare_b0_b1, validate_run, ...
│   └── figures/         # paper-grade figures → results/phantom_paper/figures/
├── queues/              # queue_baseline.sh / queue_phantom_{som,dom,text,prompt}.sh
│                        # queue_chain.sh (3-baseline collision check)
│                        # queue_phase1_paper_grade.sh (Phase 1a 全 42 condition 编排器)
├── maintenance/         # watchdog / rederive / clear_tasks / gallery / ...
│   └── glm/             # GLM Phase 2 cron sidecars
├── mechanistic/         # paper §5 (deferred)
└── vwa/                 # VWA 站点 setup

tests/                   # 929 test functions / 68 files
docs/                    # Obsidian vault root (Git plugin auto-pull 10min)
```

---

## 🧪 Experiment Design

### 三阶段实验（项目分类法）

| Phase | 目标 | 状态 |
|---|---|---|
| **Phase 1** — 表征筛选 | 多 mode flat per (site, model) | Phase 1a (workshop) + 1b (main paper) |
| **Phase 2** — 路由研究 | rule-based + learned router (paper §6) | Pass-2 同 Phase 1a 串行 |
| **Phase 3** — 模块消融 | M1/M2/M3/M4 ablation (paper-2 deferred) | — |

### Phase 1a scope（workshop-targeted）

**42 conditions / 6 cells** = Pass-1 baseline 36 + Pass-2 learned router 6
- Pass-1 = `{cls, red} × {B0, B1, B2} × {DOM, SoM, Vision, P-text, P-prompt, P-SoM}`
- Pass-2 = `{cls, red} × {B0, B1, B2} × {learned router}` （`obs_mode="learned"` sentinel）
- 统计单元 = **6 cells** = (site, model) 分层；router pass-2 = 每 cell 额外 condition，不增 cell

### Phase 1b scope（main paper expansion, deferred）

+18 conditions = `shop × {B0, B1, B2} × 6 modes`，post-workshop fire。

### 关键变量

**A1 Observation mode** (6-mode flat)：

| Mode | 文本 | 图像 |
|---|---|---|
| `dom` | viewport-only AXTree | — |
| `som` | `[SOM_MARKS]` text | 带框截图 |
| `vision` | — | 裸截图 |
| `phantom_text` (P-text, legacy `phantom_dom`) | DOM prompt + `[SOM_MARKS]` text | — |
| `phantom_prompt` (P-prompt) | SoM prompt + AXTree text | — |
| `phantom_som` (P-SoM, hero arm) | SoM prompt + `[SOM_MARKS]` text | — |

**Baselines (3 模型, advisor discussion 2026-05-14)**：

| ID | Model | 备注 |
|---|---|---|
| **B0** | Qwen3-VL-235B-A22B | via AWS proxy (hybrid shim — Anthropic URL + OpenAI tools schema) |
| **B1** | Qwen3-VL-4B | local, bf16 ~10 GB VRAM |
| **B2** | `google/gemma-3-4b-it` (Gemma3-VL) | 跨族 control，4B 量级对齐 B1，bf16 装 A100 40 GB |

**Statistical gate (preregistration §2.5)**：
- **Primary** = one-sided FE inverse-variance pooled superiority test (H₀: θ_FE ≤ +1.0 pp, α=0.05)
- **Appendix sensitivity** = DerSimonian-Laird RE / HKSJ + TOST equivalence
- **K-of-N** = transparency-only count，**非 gate**

> 当前 Phase 1a active cells / GPU 状态 / done counts → `make active` 或 `cells.base` (Obsidian Bases) · **不在 README 里硬编码** (会 stale)

---

## 📚 Documentation

仓库采用 **6-doc + Obsidian data layer** 架构：

| 文档 | 用途 | 更新 |
|---|---|---|
| `docs/checkpoints/next_steps.md` | **Action ledger** — live + future only (embed Bases views) | Daily |
| `docs/checkpoints/paper_planning.md` | **Strategy notebook** — theory / findings / risks / router / advisor / reviewer 预案 / decision log | Weekly |
| `docs/checkpoints/paper_drafts/` | **Final prose** — `section1_intro.md` ... `section8_limitations.md` + `paper.bib` | 每 codex round |
| `docs/checkpoints/实验笔记.md` | **Chronicle** — append-only，247+ sections，Obsidian-tagged (`#finding #literature #bug #infra #design`) | Append-only |
| `docs/checkpoints/phase1_plan.md` | **Phase 1 canonical execution** — §A1/A2 audit + §B clean run + §C router + §D evidence + §E milestones | Per-milestone |
| `docs/checkpoints/PLAYBOOK.md` | **Operating manual** + GLM-managed §1+§2 live status (今日瓶颈 / cron 健康 / cell changelog) | Rolling (GLM @daily 8AM) |

**Obsidian data layer**：
- Vault root = `docs/`；Obsidian Git plugin auto-pull 10 min (Windows side)
- Single-source frontmatter at `_status/{section,cells,codex,issues}/*.md`
- 4 Bases views at vault root: `status.base` / `cells.base` / `codex.base` / `issues.base`
- Canvases: `phantom_space.canvas` / `paper_section2_framework.canvas` / `experiment_matrix.canvas`

---

## 🖥 Runtime Environments

三层算力（按需调用 — 详见 [`docs/reference/COMPUTE_INFRASTRUCTURE.md`](docs/reference/COMPUTE_INFRASTRUCTURE.md)）：

| Tier | 主机 | 特性 | 适合 |
|---|---|---|---|
| **DGX Spark** | `spark-9ea3` aarch64 GB10 ~128 GB | 共享 GPU 有争抢、无 sudo；可经 Tailscale 到 quark VWA Docker (dev only) | dev session / curation / archived 数据源 |
| **Condenser A100** ⭐ | VM `a100-jiaming-test` @ `10.134.51.2`，A100-PCIE-40GB | 独占无争抢、无 wallclock；**VWA docker self-hosted on VM** | **paper-grade fire** (Phase 1a/1b/Pass-2) |
| **Myriad HPC** | `myriad.rc.ucl.ac.uk` (`ucab352`) V/U-type 4×A100 80GB | SGE qsub、有 wallclock (72h/48h)、terminal-only；CGNAT 不能连 VWA | 大模型 / SAE 训练 / 4-GPU 并行 / CPU 分析批处理 |

跨集群一切经 **quark** (Windows home `100.95.81.103`) — 唯一同时能到 lab Tailscale + UCL VPN 的机器。

### DGX Spark 必须遵循

- 用 `python3` 或 `.venv/bin/python`（不要用 `python`）
- 必须设环境变量（脚本已自动处理）：
  ```bash
  export PYTORCH_NVML_BASED_CUDA_CHECK=1
  export CUDA_MPS_PIPE_DIRECTORY=""
  export CUDA_MPS_LOG_DIRECTORY=""
  ```
- GB10 `sm_121` 架构可能触发 nvrtc 错误 → 仓库内置 `torch.prod` fallback (`p79/utils/torch_cuda_workarounds.py`)
- torch 必须 CUDA 版本：`pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.11.0+cu128 torchvision==0.26.0+cu128`
- Shopping `base_url` 必须配为 DGX 可达 IP（非 localhost），否则 Magento 302 回环
- 详见 [`docs/reference/DGX_SPARK_MACHINE_QUIRKS.md`](docs/reference/DGX_SPARK_MACHINE_QUIRKS.md)

---

## 🔧 Onboarding a New Host

`git clone --recursive` + `pip install -e .` 之后，本地准备 6 项（不入 git，per-host）：

1. **`.env`** — API keys
   - `OPENAI_API_KEY`（VWA eval）
   - `DASHSCOPE_API_KEY`（B0 proxy）
   - `P79_GLM_KEY`（可选，digest 用）

2. **`scripts/vwa_env_remote.sh`** — VWA base URL
   ```bash
   export VWA_REMOTE_HOST=100.95.81.103   # quark Tailscale IP
   export CLASSIFIEDS=http://${VWA_REMOTE_HOST}:9980
   export REDDIT=http://${VWA_REMOTE_HOST}:9999
   export SHOPPING=http://${VWA_REMOTE_HOST}:7770
   export SHOPPING_ADMIN=http://${VWA_REMOTE_HOST}:7780
   export WIKIPEDIA=http://${VWA_REMOTE_HOST}:8888
   export HOMEPAGE=http://${VWA_REMOTE_HOST}:4399
   ```

3. **VWA 站点 access**
   - A — Tailscale 加入网内（reach quark `100.95.81.103`），或
   - B — 本机起 docker compose（需 sudo + ~3 GB Wikipedia zim，按 VWA upstream README）

4. **`.auth/`** — Playwright session state，`cd external/visualwebarena && bash prepare.sh` 生成

5. **GPU torch CUDA build**
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu128 \
       torch==2.11.0+cu128 torchvision==0.26.0+cu128
   python3 -c "import torch; print(torch.cuda.is_available())"   # must be True
   ```

6. **Shell env** — 加到 `~/.bashrc` 或 conda activate hook
   ```bash
   export PYTORCH_NVML_BASED_CUDA_CHECK=1
   export CUDA_MPS_PIPE_DIRECTORY=""
   export CUDA_MPS_LOG_DIRECTORY=""
   ```

> **Known hardcoded `100.95.81.103`** (quark) — 9 处 .py/.sh 直写。Myriad 若直接 reach 此 IP（Tailscale）则无需改；否则临时 `sed` 替换或等 IP env-var-ize 重构（next_steps §5 backlog）。

---

## 🔄 Cross-Host Results Sync

数据画像：单 condition run **~1.7 GB**（artifacts ~100 MB / **JSONL 仅 ~15 MB**）。3-tier sync 策略（DGX = hub）：

| Tier | 内容 | 大小/cell | 策略 |
|---|---|---|---|
| **A. Summary** | `condition_summary_v2.json` + `run_meta.json` | KB-MB | 重要数据，commit candidate (待评估) |
| **B. Episodes + analysis** | `episodes/*.jsonl` + `analysis/**` | ~15 MB | 跑完即 rsync 到 hub（默认行为） |
| **C. Artifacts** | screenshots / SoM 图 | 100 MB-1 GB | 留本地，paper figure 重生时按需拉 |

```bash
# Spoke → Hub: 推 Tier B (默认无 artifacts)
make rsync-to-hub                                                # HOST=spark-9ea3 (default)
DRY=1 make rsync-to-hub HOST=jiaming@spark-9ea3                  # 预演

# Hub → 拉别 host 跑出来的数据
make rsync-from-hub                                              # 默认 Tier B
make rsync-from-hub RUN=B1_phantom_classifieds_20260428          # narrow 1 run
make rsync-artifacts-from-hub RUN=... COND=phase1_phantom_som_router_0   # 含 artifacts
```

底层脚本 `scripts/maintenance/rsync_results_{to,from}_hub.sh` 支持 `HOST` / `HUB_PATH` / `RUN` / `COND` / `ARTIFACTS` / `DRY` env vars。

---

## 🛡 Hard Rules (Paper-Grade)

1. **同 site 同时只能跑一个 baseline (B0 XOR B1 XOR B2)** — 共享 docker container + 同 user account login (reddit `MarvelsGrantMan136` / cls `blake.sullivan` / shop `emma.lopez`)。多 runner 同 site → session race + watchdog auth_refresh 互相 invalidate + cart 串污染。3-way collision check 在 `queue_chain.sh`。
   - 启动前 check：`pgrep -f "run_experiment.*${SITE}"` 必须空
   - 跨 baseline 同 site → 用 `queue_chain.sh` 自动 sequential

2. **跑实验必须 reset 站点** — 之前 condition 累积 site state (cart, posted listing, subscribed forum) 破坏 paper-grade ablation fairness。

3. **paper-grade fire 同一物理 host 同时只能跑一条 site chain (cls XOR red XOR shop)** — cls + red 共享 A100 docker bridge + Postgres/Redis underlay + B0 AWS proxy quota；实证 2026-05-18 fire 出现 red 99s busy-wait + B-1581 asyncio race。`queue_phase1_paper_grade.sh launch` 默认 sequential cls→red (`PHASE1A_PARALLEL=1` 显式 opt-in dev parallel，非 paper-grade safe)。

4. **禁止裸用 `python scripts/run_experiment.py`** — 五选一 queue script：
   - `queue_baseline.sh <baseline> <mode:dom|som|vision> <site> [benchmark]`
   - `queue_phantom_som.sh <baseline> <site> [benchmark]`
   - `queue_phantom_dom.sh <baseline> <site> [benchmark]`
   - `queue_phantom_text.sh <baseline> <site> [benchmark]`
   - `queue_phantom_prompt.sh <baseline> <site> [benchmark]`

---

## 📖 References

**Live state**

- `make active` — 真实时 process scan
- `cells.base` / `status.base` / `codex.base` / `issues.base` — Obsidian Bases 视图（~10 min snapshot）
- `docs/checkpoints/PLAYBOOK.md` §1+§2 — 今日瓶颈 + cron 健康度（GLM @daily 8AM）

**Paper trajectory (6-doc)**

- [`docs/checkpoints/next_steps.md`](docs/checkpoints/next_steps.md) — daily action ledger ⭐
- [`docs/checkpoints/paper_planning.md`](docs/checkpoints/paper_planning.md) — paper strategy ⭐
- [`docs/checkpoints/paper_drafts/`](docs/checkpoints/paper_drafts/) — section1-8 prose + paper.bib
- [`docs/checkpoints/实验笔记.md`](docs/checkpoints/实验笔记.md) — chronicle (247+ §)
- [`docs/checkpoints/phase1_plan.md`](docs/checkpoints/phase1_plan.md) — Phase 1 canonical execution
- [`docs/checkpoints/PLAYBOOK.md`](docs/checkpoints/PLAYBOOK.md) — operating manual

**Operational reference (`docs/reference/`)**

- `DGX_SPARK_MACHINE_QUIRKS.md` — DGX 机器特化 (sm_121 fallback / MPS / Tailscale)
- `COMPUTE_INFRASTRUCTURE.md` — DGX ↔ quark ↔ Myriad SSH chain + VPN
- `automation_overview.md` — 6-layer automation architecture
- `launch_checklist.md` — paper-grade rerun protocol
- `EVIDENCE_LAYER_AUDIT.md` — figure provenance (paper §1-§7)
- `master_bug_catalog.md` — bug taxonomy + fix history
- `PHANTOM_SOM_CODE_TOUR.md` — paper §3 backing implementation walkthrough
- `analysis_templates.md` — analysis pattern library
- `condition_map.md` — condition_id → benchmark / mode / phase mapping
- `glm_quark_myriad_sync.md` — GLM Phase 2 cron sync mechanics

**Top-level entry points**

- [`Makefile`](Makefile) — daily 命令一键化（`make help`）
- `results/` 目录索引 — `ls -lt results/visualwebarena/phase1/` 或 `_status/cells/cell_*.md` frontmatter
