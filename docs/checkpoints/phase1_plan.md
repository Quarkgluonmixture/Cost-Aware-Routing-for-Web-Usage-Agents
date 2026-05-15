---
type: phase-plan
status: active
phase: 1
updated: 2026-05-15
---

# Phase 1 执行计划 — VWA cls+red × 3 模型 × 6 模式

> **统领性 checklist**。三条 workstream A/B/C + 下游 D,每条是 audit/execute checklist。
> Bug 全表 / 实验细节 → `master_bug_catalog.md` + [[实验笔记]] (不在这)。
>
> Live state 去处: `make active` / `cells.base` / [[PLAYBOOK]] §1+§2 / [[next_steps]] / [[issue_advisor_sync_2026-05-14]]

## /stress 用法约定

- **单点调用**: `/stress <item-id>` (e.g. `A1.7` / `A2.3`) — /stress 自读 item 名下的 artifact
- **条目不写 invariant** — 让 /stress 从 code/prose 反推断言,避免 list-shaped blind spot (memory `feedback_lean_audit_prompts`)
- **§A1 (实现层)** = /stress **code-audit mode** — read SCRIPTS first, 找 code↔prose mismatch + pipeline bug
- **§A2 (设计层)** = /stress **claim-audit mode** — read paper_drafts + preregistration + paper_planning, 不读 SCRIPTS, 攻击 claim / control rigor / methodology
- **Mode B 默认链** `/codex-stress` 同 scope (memory `feedback_cross_ai_audit`)

---

## §0 总览

**Phase 1** = VWA **classifieds + reddit** × **3 baseline 模型** × **6 observation mode** paper-grade clean run + 下游 4 层 evidence + router 双路线。

- **3 模型**: B0 Qwen3-VL-235B-A22B (proxy API) / B1 Qwen3-VL-4B (local) / B2 Gemma3-VL `google/gemma-3-4b-it` (跨族 matched-capability control)
- **6 mode**: DOM / SoM / Vision / P-text / P-prompt / P-SoM
- **Phase 1a** (本文档主体) = cls + red ‖ **Phase 1b** (post-workshop deferred) = + shopping

### Condition / cell 规模 (terminology hard rule)

- **condition** = 1 (site, model, mode) launch unit ‖ **cell** = 1 (site, model) 统计 stratification unit
- **Phase 1a = 36 conditions / 6 cells** = (cls + red) × {B0, B1, B2} × 6 modes
- Phase 1b = + 18 conditions = shop × {B0, B1, B2} × 6 modes

### Critical path

```
A1 实现层 audit ──┐                                      ┌──→ D 四层 evidence ──→ paper §1/§4 hero
                  │                                      │
A2 设计层 audit ──┼──→ B clean run (gated on A1+A2+B0) ──┤
                  │                                      │
B0 infra prereq ──┘                                      └──→ C router 双路线
```

---

## §A 审查 — 两层 stress target

> **§A1 = 实现层** (代码 / 管线 / 数据流正确性) — /stress code-audit mode。
> **§A2 = 设计层** (research question / control / statistics / framing) — /stress claim-audit mode。
> 两层是 **正交** 的: A1 干净 ≠ A2 干净;A2 干净 ≠ A1 干净。

### §A1 实现层 audit surfaces

> 用法: 挑一个 `[ ]` → `/stress A1.x` → /stress 自读 named scope。审完勾掉,发现的 bug 进 `master_bug_catalog.md`,**不回写这里**。

#### A1-代码 (`p79/`)

- [ ] **A1.1** `p79/agents/` — agent 层 (`proxy_api_agent` / `qwen3vl_agent` / `gemma3vl_agent` + step contract)
- [ ] **A1.2** `p79/backends/` — backend 层 (`api_proxy` / `local_qwen` / `local_gemma` + `factory` + `base` + `action_utils` + `image_utils` + `heuristic`)
- [ ] **A1.3** `p79/envs/` — VWA wrapper + locator dispatch + observation pipeline
- [ ] **A1.4** `p79/experiment/som.py` + mark 抽取链 — SoM + phantom mark layer
- [ ] **A1.5** `p79/experiment/runner/` — main orchestrator (`main.py` + `helpers.py`)
- [ ] **A1.6** `p79/experiment/analysis.py` — FP architecture + `scored_task_count` + adjusted_success 退役痕迹
- [ ] **A1.7** `p79/experiment/conditions.py` + `configs/exp_v2_*.yaml` — condition matrix 生成 + 36-cond config 全家
- [ ] **A1.8** `p79/experiment/{types.py, schema_migrations/, io_utils.py, logger_v2.py}` — schema + JSONL + dedup
- [ ] **A1.9** `p79/experiment/{metrics.py, energy_tracker.py, environment.py}` — cost / energy / env wrapper
- [ ] **A1.10** `p79/experiment/{router.py, modules.py, state_change.py, checklist_module.py, tasks.py, config.py}` — utility 模块 cluster
- [ ] **A1.11** `p79/{utils/, cli/, logging/}` — 辅助 (CUDA workaround / CLI 入口 / structured logging)
- [ ] **A1.12** `tests/` — pytest 覆盖度 + invariant 测试是否对得上当前实现

#### A1-管线 (`scripts/`)

- [ ] **A1.13** `scripts/queues/queue_{baseline,phantom_*,chain}.sh` — launch + 3-way collision
- [ ] **A1.14** `scripts/queues/queue_phase1_paper_grade.sh` + `scripts/preflight_v2.sh` — orchestrator + pre-launch gates
- [ ] **A1.15** `scripts/maintenance/experiment_watchdog.py` + `scripts/maintenance/glm/*.py` — watchdog auto-clean + cron sidecars (glm_cell_autoupdate / myriad_watcher / glm_pre_launch_check / batch_digest)
- [ ] **A1.16** `scripts/provenance/snapshot_*` — env + VWA fingerprint
- [ ] **A1.17** `scripts/vwa/` + `RESET_BEFORE` protocol — site setup + reset race-safety

#### A1-外部 (VWA submodule)

- [ ] **A1.18** VWA submodule `p79-patches` branch — evaluator + helper_functions + LLM-judge guard

#### A1-分析管线 (clean-run 下游, 但 code 本身是 pre-data audit)

- [ ] **A1.19** `scripts/analysis/aggregate_*.py` — aggregator 层 (sr_fp / phantom_lift / cross_site / routing_auroc / failure_modes / cost_electricity)
- [ ] **A1.20** `scripts/analysis/figures/*.py` — figure 脚本全家 (fig0a-3d + mechanism + venn)
- [ ] **A1.21** `scripts/analysis/preregistration_decision_test.py` + `scripts/analysis/lib/run_registry.py` + `results/phantom_paper/run_manifest.yaml` — decision test + registry + paper-grade promotion

#### §A1 已知未结项 (pointer — 不复制)

- ⏳ B-86 B0 GLM parse-error fallback scaffold — 待学长回应
- 🟢 FP-restructure piece 4d cosmetic — 低优
- ⏳ #10 analysis 层 3-model 改造 — gate §D 不 gate launch
- 详细 9-bug 表 → `master_bug_catalog.md` §139 / [[实验笔记]] §139

---

### §A2 设计层 audit surfaces

> 用法: 挑一个 `[ ]` → `/stress A2.x` → /stress 读对应 paper_drafts / preregistration / paper_planning section,**不读 SCRIPTS**。攻击 claim / framing / methodology,不查 bug。

- [ ] **A2.1** Research question framing — "phantom routing space" 假设是否良构 + falsifiable; "4-fold drop-in property" 是 1 个 property 还是 4 个独立 claim 包成一个 (artifacts: `paper_drafts/section1` + [[paper_planning]] §1)
- [ ] **A2.2** Comparison rigor / control design — B0 / B1 / B2 是哪条轴的 control (capability / family / deployment-class)? B2 与 B1 "matched-capability cross-family" 在 4B 参数对齐外是否还有别的对齐要求 (训练数据 / alignment 配方 / instruction tuning)? (artifacts: [[paper_planning]] §15 prior-work table + [[实验笔记]] §138)
- [ ] **A2.3** Statistical design — N=6 cells + 观测 effect size 1-3pp 的 power; DL meta vs REML+HK at k=6; K-of-N transparency-only 重分类合理性; TOST δ=1.0pp 来源 + 与文献对齐; Bonferroni / Holm 是否够 (artifacts: `pre_run/preregistration.md` §2.4 / §3 / §4 + `power_analysis.py` 注释)
- [ ] **A2.4** Evidence-claim coupling — 4-dim × 4 cross-X = 16 sub-cell 是否真支撑 paper §1 hero?cross-site axis 只 cls+red 2 site 够不够支撑 generalization claim?cross-model 3 baseline 中 B0 是 API 异类、B1/B2 同 deployment-class — 这对 cross-model claim 意味着什么 (artifacts: [[paper_planning]] §3 + §21)
- [ ] **A2.5** Operationalization — rule-based router 的 "task 属性" 定义边界 + leak risk (用 task description 训练 → test 时 leak);learned classifier 的 feature set (TF-IDF + binary + browser meta) 信号源 vs leak 边界;5-fold site-stratified CV 是否解决 site leak (artifacts: [[paper_planning]] §8 + `p79/experiment/router.py`)
- [ ] **A2.6** Scope / external validity — Phase 1a 只有 cls + red → R3 framing risk;"phantom space" 概念领域 (VWA-specific / web-agent-general / LLM-general?);Phase 1b shop 推迟 + WA 缺席的下游影响 (artifacts: [[paper_planning]] §6 R1-R5 + `pre_run/preregistration.md` §7 reproducibility scope)
- [ ] **A2.7** Confound register / known asymmetries — B0 (API) vs B1/B2 (local) deployment 异类;B0 max_new_tokens / GLM parse fallback / quantization 非对称;A100 docker stack vs DGX→quark stack 切换的实验环境变化 (artifacts: CLAUDE.md Guard Rails + [[实验笔记]] §139 B-86 + memory `project_paper_hook.md`)
- [ ] **A2.8** Pre-registration completeness — H1-H8 primary/exploratory/post-hoc/deferred 族 declaration 边界;§4 locked analysis choices 全覆盖;§6 witness mechanism (Git commit + advisor email + OSF DOI) 防 post-hoc 修改的强度 (artifacts: `pre_run/preregistration.md` 全文 + `pre_run/osf_lock_manifest.md`)

#### §A2 已知未结项 (pointer)

- ⏳ advisor 确认 K_h1 / K_h3 / TOST δ + DL vs REML+HK meta 方法 (sync 后才能 preregistration lock)
- ⏳ Gemma3-VL matched-capability 论证 (vs B1) — 当前只有 4B 参数对齐,是否够 (A2.2 直接攻击)
- 详细 framing decision log → [[paper_planning]] §19 + [[ADVISOR_SYNC]]

---

### §A 边界 — stress 完之后还剩什么 (本文档故意不覆盖)

1. **Post-data 解释纪律** — clean run 数据 land 之后,"X pp effect → claim Y" 的推断链是否成立。这是 §D 之后的 mode,届时另起 /stress (claim-audit mode 跑在 actual data + paper drafts 上)。
2. **Reviewer rehearsal** — submit 前模拟 R1-R5 reviewer 攻击。框架已在 [[paper_planning]] §14,但应在 submit 前再走一次,**不在 phase1_plan 责任范围**。

→ 这两项**不要**塞进 §A;留作独立 stress phase,触发条件分别是 "M2 clean run done" 与 "M5 paper draft ready"。

---

## §B — cls+red × 3 模型 × 6 模式 clean run checklist

### B0. Infra prereq (launch 前必须)

- [x] Gemma3-VL Tier 1-3 接入 — agent / backend / factory / 12 configs / queue / orchestration / A100 smoke ALL PASS ([[实验笔记]] §140)
- [x] A100 venv 全栈 dep
- [ ] **#11 A100 VM VWA docker bring-up** — ⭐ gates launch
- [ ] A100 playwright install
- [ ] **#10 analysis 层 3-model 改造** — gates §D,不 gate launch

### B1. Pre-run lock 文档 (= launch gate,引 `pre_run/` 不复制)

- [ ] `preregistration.md` status `draft → locked` (待 advisor 确认 A2.3)
- [ ] `locked_versions.md` / `model_card.md` / `dataset_card.md` — B2 三模型对齐
- [ ] env + vwa provenance snapshot on A100 host
- [ ] `pre_rerun_audit.md` + `reeval_audit_protocol.md` 走查
- [ ] `osf_lock_manifest.md` 8-step (可 launch 后并行)

### B2. Launch

- [ ] `bash scripts/queues/queue_phase1_paper_grade.sh dry-run`
- [ ] `bash scripts/queues/queue_phase1_paper_grade.sh launch`
- [ ] watchdog + done-monitor 配好

### B3. 完成判定

- [ ] 36 conditions 全 done
- [ ] 每 cell `validate_run.py` pass
- [ ] watchdog auto-clean 0 污染确认
- [ ] 输出 → per-task SR + `confidence_summary.json` → 喂 §C + §D

### Hard rules (paper-grade)

- 同 site 单 baseline (B0 XOR B1 XOR B2) — `queue_chain.sh` 自动 sequential
- `RESET_BEFORE=1` 或 chain auto-reset
- 禁止裸用 `run_experiment.py` — 必须 queue script

---

## §C — router 双路线 checklist

> Gated on §B done。设计细节 [[paper_planning]] §8。

### C1. 路线 (a) rule-based — 按 task 属性

- [ ] per-task SR ready
- [ ] task 属性 feature 定义
- [ ] `p79/experiment/router.py::RuleBasedRouter` 扩展
- [ ] baseline 对比 (random / best-single-mode / oracle / rule)

### C2. 路线 (b) learned classifier

- [ ] feature extraction (TF-IDF + binary + browser meta)
- [ ] 5-fold site-stratified CV split
- [ ] classifier 训练 + feature ablation
- [ ] vs rule-based + vs baseline

### C3. 未来扩展 (本 Phase 不做)

- behavior-level route (按 mode 行为模式而非 task 属性) — 留 follow-up

### C4. 完成判定

- [ ] 两路线各出 vs-baseline 对比 + ablation
- [ ] 输出 → paper §6 routing section

---

## §D — 下游 4 层 evidence 分析 checklist

> 框架 [[paper_planning]] §3。Gated on §B done + B0 #10。

### D1. 4 层 × 4 cross-X = 16 evidence sub-cell

- [ ] Outcome 层
- [ ] Macro 层
- [ ] Micro 层
- [ ] Efficiency 层
- [ ] cross-task / cross-mode⭐ / cross-site / cross-model 四 axis 全覆盖

### D2. Pipeline

- [ ] `make analysis`
- [ ] `preregistration_decision_test.py` → H1/H3/TOST verdict

### D3. 完成判定

- [ ] 16 sub-cell 全填
- [ ] decision test verdict JSON
- [ ] 输出 → paper §1 hero / §4 P-prompt column / §5 Table

---

## §E 里程碑

| M | 判定 | gated on |
|---|---|---|
| M0 infra ready | §B0 全勾 (重点 #11) | — |
| M1a 实现层 audit 放行 | §A1 全勾 (或 open item disclosed) | — |
| M1b 设计层 audit 放行 | §A2 全勾 + preregistration locked | advisor sync |
| M2 clean run done | §B3 全勾 | M0 + M1a + M1b |
| M3 evidence 分析 done | §D3 全勾 | M2 + #10 |
| M4 router done | §C4 全勾 | M2 |
| M5 workshop-ready | M3 + M4 | — |

**Phase 1b** = shop × 3 × 6 = 18 cond, post-workshop deferred。

---

## §F Refs

- **Scope**: [[issue_advisor_sync_2026-05-14]]
- **Bug 全表**: `master_bug_catalog.md` §139 + [[实验笔记]] §139 + §139.8 + [[issue_phase1_audit_2026-05-13]]
- **Framework**: [[paper_planning]] §3 (evidence) + §8 (router) + §14 (reviewer attack) + §19 (decision log)
- **Pre-run lock**: `docs/checkpoints/pre_run/` 全 folder
- **Launch**: `scripts/queues/queue_phase1_paper_grade.sh` + `docs/reference/launch_checklist.md`
- **Compute**: `docs/reference/COMPUTE_INFRASTRUCTURE.md`
- **/stress + /codex-stress 流程**: `.claude/skills/stress/SKILL.md` + `docs/checkpoints/process/stress_skill_replica.md`
</content>
