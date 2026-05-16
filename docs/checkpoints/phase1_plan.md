---
type: phase-plan
status: active
phase: 1
updated: 2026-05-16
router_design: v7 learned-only (cascade deferred paper-2 per Q3 decision 2026-05-16)
phase1a_conditions: 42 = 36 baseline + 6 learned router (sequential 2-pass A100 launch)
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

### Condition / cell 规模 (terminology hard rule, v7 2026-05-16 walk-back update)

- **condition** = 1 (site, model, mode, router_kind) launch unit ‖ **cell** = 1 (site, model) 统计 stratification unit
- **Phase 1a total = 42 conditions / 6 cells**, 分两个 pass sequential A100 launch (per `proposals_v7.md` D3 + user Q1/Q2 confirmation 2026-05-16):
  - **Pass 1 (baseline)**: 36 conditions = (cls + red) × {B0, B1, B2} × 6 modes — paper §1 hook data source. A100 wallclock ~1-2 weeks
  - **Pass 2 (router)**: 6 conditions = (cls + red) × {B0, B1, B2} × **1 learned router/cell** (LR over phantom-augmented mode set, obs_mode="learned" sentinel; per-task mode 由 LR runtime decide) — paper §6 H10 Pareto data source. A100 wallclock ~3-5 days
  - Cells (statistical strata) unchanged at 6 across both passes
- Phase 1b = + 21 conditions = shop × {B0, B1, B2} × (6 baseline + 1 router) modes (deferred to post-workshop, mirrors Phase 1a 2-pass protocol)
- **Cascade router (v6) 6 router-cond/cell variant** = paper-2 forward stub, `phase1.router_kind: cascade` enum value 保留 但 paper-1 不用

### Critical path (v7 2026-05-16 sequential 2-pass)

```
A1 实现层 audit ──┐                                                                            ┌──→ D 4-layer evidence ──→ paper §1 hook
                  │                                                                            │
A2 设计层 audit ──┼──→ §B-baseline launch (36 cond, 1-2 weeks A100) ──→ §B-router launch ──┤
                  │   (gated on A1+A2+B0)                                (6 cond, 3-5 days)    │
B0 infra prereq ──┘                                                                            └──→ §C learned router → paper §6 H10
```

**Pass-1 (baseline) → Pass-2 (router) sequential**: baseline pass data must land first (paper §1 hook + §C learned router train fold labels via per-task oracle), router pass runs second on same A100 host. v7 walk-back drops cascade ↔ paper-2 — paper-1 §C 只 learned classifier 路线。

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
- [x] **A1.6** `p79/experiment/analysis.py` — FP architecture + `scored_task_count` + adjusted_success 退役痕迹 (2026-05-16: 17 findings B-237~B-253; hard-delete sweep retract selective-retain-for-output-schema-stability; chronicle §158)
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
- [x] **A1.17** `scripts/vwa/` + `RESET_BEFORE` protocol — Chunk 1 ✅ 2026-05-16 (B-298~B-306, §162: 5 P0 + glm-absorb-P1 + P2-2); Chunk 2 ✅ 2026-05-16 late (B-307~B-314, §163: 6 P1 quality + Option K Trajectory Event Log schema/API/hooks)

#### A1-外部 (VWA submodule)

- [x] **A1.18** VWA submodule `p79-patches` branch — evaluator + helper_functions + LLM-judge guard (2026-05-16: 15 findings B-254~B-268; gemini OOB P0-1 viewport paradox catch + codex F10 IP 794-hit deepening; full clean: 913 task configs IP→placeholder + paper §3.5/§4.X.11/§4.X.12 disclosure + 3-layer SBOM lock + 5 P1 code fixes; chronicle §159; memory `reference_vwa_submodule_p79_patches.md`)

#### A1-分析管线 (clean-run 下游, 但 code 本身是 pre-data audit)

- [ ] **A1.19** `scripts/analysis/aggregate_*.py` — aggregator 层 (sr_fp / phantom_lift / cross_site / routing_auroc / failure_modes / cost_electricity)
- [ ] **A1.20** `scripts/analysis/figures/*.py` — figure 脚本全家 (fig0a-3d + mechanism + venn)
- [ ] **A1.21** `scripts/analysis/preregistration_decision_test.py` + `scripts/analysis/lib/run_registry.py` + `results/phantom_paper/run_manifest.yaml` — decision test + registry + paper-grade promotion

#### §A1 已知未结项 (pointer — 不复制)

- ⏳ B-86 B0 GLM parse-error fallback scaffold — 待学长回应
- 🟢 FP-restructure piece 4d cosmetic — 低优
- ⏳ #10 analysis 层 3-model 改造 — gate §D 不 gate launch
- ⏸ **A1.16-mechanism subset deferred per paper-2 scope** (set 2026-05-16, advisor 2026-05-14 mechanism defer): `scripts/provenance/numerical_determinism_check.py` 内 5 bugs (D-1 TF32 matmul blindness gemini-OOB P0 / D-2 dtype non-determinism gemini-OOB P0 / D-3 `external_code` typo P1 / D-4 threshold 1e-2 vs 1e-3 mismatch P1 / D-5 input not SHA-pinned codex-OOB P1). Paper-1 不依赖 (mechanism quote 暂搁); paper-2 mechanism resumes 时是 hard gate. Full bug list → [[master_bug_catalog]] A1.16 batch tail (B-273~B-279 + DEFER list)。
- ⏸ **B-275 runner-side enforce** — `from_pretrained(model_id, revision=<pinned_sha>)` 强制拒绝 stale cache 跟 `pre_run/locked_versions.md` 联动。B-275 snapshot-side capture 已 land (双字段 `loaded_revision` + `registry_head` + `divergence`); runner-side enforce 是 downstream change (touches `p79/backends/local_qwen.py` + `local_gemma.py`), 单独 fix slice 处理。
- ⏸ **A1.17 Chunk 2 — paper-grade quality + Option K Trajectory Event Log** (set 2026-05-16, ~11h investment): P1-1 cls reset sentinel multi-table (3-AI overlap) / P1-2 a100_self_host_vwa.sh deploy_reddit+shopping bad paths / P1-6 reddit reset missing `-e TZ` (gemini OOB) / P1-7 REQ_GB 130→250 (gemini OOB) / P1-8 Magento `indexer:reindex` async poll (gemini OOB) / P1-12 cls DB seed `|| true` strip BUG-5 sibling propagation。**Option K Trajectory Event Log** (user cross-talk insight 2026-05-16): unified `trajectory_events: [{event_type, task_index, wallclock_ts, metadata}]` schema 在 `p79/experiment/logger_v2.py` 加 `log_trajectory_event()` API + hooks in `experiment_watchdog.py` (auth-clear) + `_lib_paper_grade_gates.sh` (reset event) — generalizes P1-5-B Tier 1 stack 到 auth-loss/auto-clear class 同时 cover, ~2-3h additional schema work。Full bug list → [[master_bug_catalog]] A1.17 Chunk 1 batch tail (B-298~B-306) Deferred section。
- ⏸ **P1-5-B advisor sync items** (set 2026-05-16, post-§162 Tier 1 stack): (2-gemini) Paper §3 "Multi-Epoch Sequential Benchmark Protocol" reframe (1h prose, codex round 时加) / (1-gemini) GLMM with `is_after_reset + had_auth_clear` covariate (4h, Phase 1a 数据落地后跑 via `scripts/analysis/aggregate_sr_mixed_effects.py`) / (4-gemini) Fisher's exact homogeneity rebuttal (3h, Phase 1a 数据落地后)。Tier 2 unconventional alternatives 跨 paper-1/paper-2: (D-codex) Prefix Action Replay 2-4d / (E-codex) Coarse checkpoint K=25 2-3d / (F-codex) Per-cell disposable site namespace 1-2w (paper-2 quality)。Code fix (B) `--no-reset` on resume **已 Chunk 1 land** as B-304。
- 详细 9-bug 表 → `master_bug_catalog.md` §139 / [[实验笔记]] §139

---

### §A2 设计层 audit surfaces

> 用法: 挑一个 `[ ]` → `/stress A2.x` → /stress 读对应 paper_drafts / preregistration / paper_planning section,**不读 SCRIPTS**。攻击 claim / framing / methodology,不查 bug。

- [ ] **A2.1** Research question framing — "phantom routing space" 假设是否良构 + falsifiable; "4-fold drop-in property" 是 1 个 property 还是 4 个独立 claim 包成一个 (artifacts: `paper_drafts/section1` + [[paper_planning]] §1)
- [ ] **A2.2** Comparison rigor / control design — B0 / B1 / B2 是哪条轴的 control (capability / family / deployment-class)? B2 与 B1 "matched-capability cross-family" 在 4B 参数对齐外是否还有别的对齐要求 (训练数据 / alignment 配方 / instruction tuning)? (artifacts: [[paper_planning]] §15 prior-work table + [[实验笔记]] §138)
- [ ] **A2.3** Statistical design — N=6 cells + 观测 effect size 1-3pp 的 power; DL meta vs REML+HK at k=6; K-of-N transparency-only 重分类合理性; TOST δ=1.0pp 来源 + 与文献对齐; Bonferroni / Holm 是否够 (artifacts: `pre_run/preregistration.md` §2.4 / §3 / §4 + `power_analysis.py` 注释)
- [ ] **A2.4** Evidence-claim coupling — 4-dim × 4 cross-X = 16 sub-cell 是否真支撑 paper §1 hero?cross-site axis 只 cls+red 2 site 够不够支撑 generalization claim?cross-model 3 baseline 中 B0 是 API 异类、B1/B2 同 deployment-class — 这对 cross-model claim 意味着什么 (artifacts: [[paper_planning]] §3 + §21)
- [ ] **A2.5** Operationalization — rule-based router 的 "task 属性" 定义边界 + leak risk (用 task description 训练 → test 时 leak);learned classifier 的 feature set (TF-IDF + binary + browser meta) 信号源 vs leak 边界;5-fold site-stratified CV 是否解决 site leak (artifacts: [[paper_planning]] §8 + `p79/experiment/router.py`)
- [ ] **A2.6** Scope / external validity — Phase 1a 只有 cls + red → R3 framing risk;"phantom space" 概念领域 (VWA-specific / web-agent-general / LLM-general?);Phase 1b shop 推迟 + WA 缺席的下游影响 (artifacts: [[paper_planning]] **§5 顶刊概率 R1-R5 conditional tree** + [[paper_planning]] §6 Critical Risks + `pre_run/preregistration.md` §2 framing decision rule + §7 reproducibility scope) — pointer corrected 2026-05-15: R1-R5 framing rule lives in paper_planning §5, NOT §6 (§6 is critical risks)
- [ ] **A2.7** Confound register / known asymmetries — B0 (API) vs B1/B2 (local) deployment 异类;B0 max_new_tokens / GLM parse fallback / quantization 非对称;A100 docker stack vs DGX→quark stack 切换的实验环境变化 (artifacts: CLAUDE.md Guard Rails + [[实验笔记]] §139 B-86 + memory `project_paper_hook.md`)
- [ ] **A2.8** Pre-registration completeness — H1-H8 primary/exploratory/post-hoc/deferred 族 declaration 边界;§4 locked analysis choices 全覆盖;§6 witness mechanism (Git commit + advisor email + OSF DOI) 防 post-hoc 修改的强度 (artifacts: `pre_run/preregistration.md` 全文 + `pre_run/osf_lock_manifest.md`)

#### §A2 已知未结项 (pointer)

- ⏳ advisor 确认 K_h1 / K_h3 / TOST δ + DL vs REML+HK meta 方法 (sync 后才能 preregistration lock)
- ⏳ Gemma3-VL matched-capability 论证 (vs B1) — 当前只有 4B 参数对齐,是否够 (A2.2 直接攻击)
- 详细 framing decision log → [[paper_planning]] §19 + [[issue_advisor_sync_2026-05-14]] (ADVISOR_SYNC.md retired 2026-05-15, commit `f64bc9d`)

---

### §A 边界 — stress 完之后还剩什么 (本文档故意不覆盖)

1. **Post-data 解释纪律** — clean run 数据 land 之后,"X pp effect → claim Y" 的推断链是否成立。这是 §D 之后的 mode,届时另起 /stress (claim-audit mode 跑在 actual data + paper drafts 上)。
2. **Reviewer rehearsal** — submit 前模拟 R1-R5 reviewer 攻击。框架已在 [[paper_planning]] §14,但应在 submit 前再走一次,**不在 phase1_plan 责任范围**。

→ 这两项**不要**塞进 §A;留作独立 stress phase,触发条件分别是 "M2 clean run done" 与 "M5 paper draft ready"。

---

## §B — cls+red × 3 模型 × 6 模式 clean run checklist (v7 2-pass split)

### B0. Infra prereq (Pass-1 launch 前必须)

- [x] Gemma3-VL Tier 1-3 接入 — agent / backend / factory / 12 configs / queue / orchestration / A100 smoke ALL PASS ([[实验笔记]] §140)
- [x] A100 venv 全栈 dep
- [ ] **#11 A100 VM VWA docker bring-up** — ⭐ gates Pass-1 launch
- [ ] A100 playwright install
- [ ] **#10 analysis 层 3-model 改造** — gates §D, 不 gate launch

### B1. Pre-run lock 文档 (= launch gate, 引 `pre_run/` 不复制)

- [ ] `preregistration.md` status `draft → locked` (待 advisor 确认 A2.3 + v7 walk-back H9/H11 DEFER + Pareto reformulation)
- [ ] `locked_versions.md` / `model_card.md` / `dataset_card.md` — B2 三模型对齐
- [ ] env + vwa provenance snapshot on A100 host
- [ ] `pre_rerun_audit.md` + `reeval_audit_protocol.md` 走查
- [ ] `osf_lock_manifest.md` 8-step (可 launch 后并行)

### B-baseline. Pass-1 Launch (36 cond, paper §1 hook 数据源)

- [ ] `bash scripts/queues/queue_phase1_paper_grade.sh dry-run`
- [ ] `bash scripts/queues/queue_phase1_paper_grade.sh launch`
  - Default `phase1.variant: baseline` (per `conditions.py` v7 enum) → emit 6 cond/cell × 6 cells = 36 cond
- [ ] watchdog + done-monitor 配好
- A100 wallclock estimate ~1-2 weeks (`B0 → B1 → B2` per-site sequential, same-site one-baseline-only rule enforced by `queue_chain.sh`)

### B-baseline. 完成判定 (= gate for Pass-2)

- [ ] 36 baseline conditions 全 done
- [ ] 每 cell `validate_run.py` pass
- [ ] watchdog auto-clean 0 污染确认
- [ ] 输出 → per-task SR + `condition_summary_v2.json` (`success` 字段 canonical post-§139.8) → 喂 §C learned router train-fold + §D evidence
- [ ] **Mid-pass paper §1 prose round candidate** — Pass-1 数据够写 §1 hook (phantom 4-fold drop-in), Pass-2 期间可平行 prose round

### B-router. Pass-2 Launch (6 cond, paper §6 H10 数据源, gated on Pass-1 done)

- [ ] **新建 queue script** OR env override path: `bash scripts/queues/queue_phase1_router_paper_grade.sh launch` OR `PHASE1_VARIANT=router PHASE1_ROUTER_KIND=learned bash queue_phase1_paper_grade.sh launch`
  - Config 设 `variables.phase1.variant: router` + `variables.phase1.router_kind: learned` → emit 1 cond/cell × 6 cells = 6 cond (obs_mode="learned" sentinel, LR runtime predict)
- [ ] LR train pipeline: per-task oracle-best label assignment from Pass-1 → 5-fold site-stratified CV (within-cell) → balanced class weight LR → cell-stratified predictions
- [ ] LOCO (Leave-One-Cell-Out) train protocol implementation: train on 5 cells, test on 6th, repeat 6 times — paper §6 main number source per Q4 decision
- [ ] watchdog + done-monitor for Pass-2
- A100 wallclock estimate ~3-5 days

### B-router. 完成判定

- [ ] 6 router conditions 全 done
- [ ] 每 router cell `condition_summary_v2.json` 含 (Cost, SR, Latency) 三 metric
- [ ] Pareto non-dominance H10 paired bootstrap per cell (vs 5 single-mode baselines from Pass-1)
- [ ] LOCO 6-fold cross-cell SR + (Cost, SR) Pareto frontier
- [ ] 输出 → paper §6 H10 verdict + figure (per-cell Pareto scatter)

### Hard rules (paper-grade) — applies to BOTH passes

- 同 site 单 baseline (B0 XOR B1 XOR B2) — `queue_chain.sh` 自动 sequential
- `RESET_BEFORE=1` 或 chain auto-reset (含 router conditions)
- 禁止裸用 `run_experiment.py` — 必须 queue script
- Pass-1 + Pass-2 **不能并行** 在同 site (A100 共享 VWA Docker, account contamination 风险) — 必须 Pass-1 完全 done 之后 Pass-2 launch

---

## §C — learned router single-path checklist (v7 walk-back 2026-05-16)

> Gated on §B-baseline done (per-task oracle labels needed for LR train fold). 设计细节 [[proposals_v7]] §1 + [[paper_planning]] §8.
>
> **v7 walk-back**: §C 从 "双路线 (rule-based + learned)" 缩到 **单路线 learned**. Rule-based router 经 P1 archive sim 实证 degenerate (P1 v3 ≡ always_phantom_som); 跟 cascade L2 + H9 + H11 一起 DEFERRED paper-2. Paper-1 §6 contribution = single learned router with Pareto non-dominance H10 gate + site-asymmetric viability finding.

### C1. 路线 (DEFERRED to paper-2)

- ❌ ~~Rule-based router (route by task attribute) — 缩到 paper-2 cascade router scope~~ (v7 walk-back)
- ❌ ~~Cascade L1 + L2 (cycle + phantom-verbose) — 缩到 paper-2~~ (v7 walk-back)

### C2. Learned classifier (paper-1 §6 sole router)

- [ ] Feature extraction: 8-dim (site one-hot, capability_tier one-hot B0/B1/B2, has_reference_image, intent_color_regex, intent_compare_regex, intent_search_regex, intent_token_count, axtree_element_count from step-0)
- [ ] 5-fold site-stratified CV split (within-cell, seed=42 per preregistration §354)
- [ ] LR training: multinomial 6-class, balanced class_weight, in-fold StandardScaler on numeric features (per `l1_archive_simulation.py:117-129` Pipeline pattern, P1-11 fix)
- [ ] **LOCO (Leave-One-Cell-Out) cross-cell validation**: train on 5 cells (~1000 tasks), test on 6th cell (~200 tasks), repeat 6 times — paper §6 main number source per Q4 user decision 2026-05-16
- [ ] Pareto non-dominance H10 paired bootstrap on (Cost, SR) per cell vs 5 single-mode baselines from §B-baseline
- [ ] Latency dominance secondary check (router latency ≤ 1.10 × best-single-mode latency per cell)
- [ ] Random baseline comparison: Tier-0a uniform / Tier-0b train-fold-frequency-weighted / Tier-0c top-3-modes-per-cell uniform (paper §6 disclosure rows, not gating)

### C3. Archive sim development sanity (supplementary, NOT paper-grade)

- [x] `scripts/analysis/l1_archive_simulation.py` repeated stratified 5-fold × 10 repeats (Q4 fix landed 2026-05-16)
- Archive cls Variant B: 17.84% [16.30, 19.42] (+2.02pp vs always_phantom_som 15.81%) — robust 50-pair estimate
- Archive red Variant B: 10.33% [9.05, 11.67] (-3.95pp vs always_phantom_som 14.29%) — collapsed to majority on text-dominated cell
- 用途: development sanity check only; paper §6 main number 等 Phase 1a LOCO

### C4. 未来扩展 (paper-2 forward stub)

- Rule-based router with cascade L2 (cycle + phantom-verbose) — v6 design preserved at [[proposals_v6]] for paper-2
- Behavior-level route (按 mode 行为模式而非 task 属性) — 留 paper-2 advanced router
- Cross-VLM-family routing transfer (B0 ↔ B1 ↔ B2 capability tier interactions) — paper-2 advanced

### C5. 完成判定 (paper-1 §6 lock)

- [ ] Learned router LOCO 6-fold + within-cell 5-fold × 10 repeats reported
- [ ] H10 Pareto non-dominance verdict (cells pass / total cells)
- [ ] Site-asymmetric viability empirical finding written up (paper §6 main narrative — cls visual-rich vs red text-dominated routing behavior contrast)
- [ ] 输出 → paper §6 routing section + Pareto scatter figure (per-cell with 95% confidence regions)

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

## §E 里程碑 (v7 2-pass split 2026-05-16)

| M | 判定 | gated on | A100 wallclock |
|---|---|---|---|
| M0 infra ready | §B0 全勾 (重点 #11) | — | — |
| M1a 实现层 audit 放行 | §A1 全勾 (或 open item disclosed) | — | — |
| M1b 设计层 audit 放行 | §A2 全勾 + preregistration locked (incl. v7 H9/H11 DEFER + Pareto reformulation) | advisor sync | — |
| **M2-baseline clean run done** | §B-baseline 全勾 (36 cond, paper §1 hook 数据) | M0 + M1a + M1b | **~1-2 weeks** |
| M3 evidence 分析 done (paper §1 hook) | §D3 全勾 | M2-baseline + #10 | mostly parallel to M2-router |
| **M2-router clean run done** | §B-router 全勾 (6 cond, paper §6 H10 数据) | M2-baseline (per-task oracle labels needed for LR train fold) | **~3-5 days** |
| M4 learned router done | §C5 全勾 (LOCO + within-cell CV + Pareto H10 verdict) | M2-router + #10 | post-M2-router |
| M5 workshop-ready | M3 + M4 | — | ~1.5-2.5 weeks total |

**Wallclock savings vs v6 cascade**: v6 2-pass 72-cond = 2-4 weeks A100; v7 2-pass 42-cond = **1.5-2.5 weeks A100** (~5-7 days saved). Critical path 缩 mostly on M2-router (cascade 6 cond/cell × 6 cells = 36 router-cond → learned 1 cond/cell × 6 cells = 6 router-cond).

**Phase 1b** = shop × {B0, B1, B2} × (6 baseline + 1 learned router) = **21 cond**, post-workshop deferred. Mirrors Phase 1a 2-pass protocol.

---

## §F Refs

- **Scope**: [[issue_advisor_sync_2026-05-14]]
- **Bug 全表**: `master_bug_catalog.md` §139 + [[实验笔记]] §139 + §139.8 + §153 (v6 cross-AI audit) + §154 (v7 walk-back) + [[issue_phase1_audit_2026-05-13]]
- **Framework**: [[paper_planning]] §3 (evidence) + §8 (router) + §14 (reviewer attack) + §19 (decision log)
- **Router design trajectory** (v3 → v7 chronicle, latest LOCKED v7 2026-05-16):
  - [[proposals_v7]] — **paper-1 §6 LOCKED** (learned-only)
  - [[proposals_v6]] — paper-2 forward stub (cascade L1+L2 design preserved)
  - [[proposals_v3]] / [[proposals_v4]] / [[proposals_v5]] — historical (v3 rule-based, v4 Option C, v5 cascade 2-layer)
  - [[l1_archive_simulation_2026-05-16]] / [[l2_partial_traj_auroc_2026-05-16]] — empirical sims
- **Pre-run lock**: `docs/checkpoints/pre_run/` 全 folder + Appendix A 2026-05-16 v7 entry
- **Launch**: `scripts/queues/queue_phase1_paper_grade.sh` (Pass-1 baseline) + **新 queue or env override for Pass-2 router** (TODO) + `docs/reference/launch_checklist.md`
- **Compute**: `docs/reference/COMPUTE_INFRASTRUCTURE.md` (A100 Condenser VM)
- **/stress + /codex-stress 流程**: `.claude/skills/stress/SKILL.md` + `docs/checkpoints/process/stress_skill_replica.md`
</content>
