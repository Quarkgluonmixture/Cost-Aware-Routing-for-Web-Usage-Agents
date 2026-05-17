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

- [x] **A1.1** `p79/agents/` — agent 层 (`proxy_api_agent` / `qwen3vl_agent` / `gemma3vl_agent` + step contract) (2026-05-16: 3-AI cross-audit Mode A+B+C, 16 findings → 10 fixed + 1 defused-by-existing-disclosure + 4 Mode C prose deferred next codex round + 3 paper-2 scope; commits `0f3a7c2`+`d765dbf`; B-395~B-405; key fixes = B-395 paper_grade flag end-to-end wire (3-AI overlap P0-1) + B-396 39 yaml use_glm_fallback flip (paper-grade hard-off defense-in-depth) + B-397 image_meta_recorded backend-aware (2-AI A+B P0-2) + B-398 glm_fallback_attempted unconditional persistence (2-AI A+B P0-3) + B-399 total_minus_retry failed-attempt elapsed + B-400 image_payload_bytes_total ref+screenshot (2-AI A+C) + B-401 B0 latency split None + B-402 B0 _shared_vl_utils direct import + B-403 image_encode_error symmetric-exclude aggregator + B-405 adjusted_sr archive warning; user Phase A directive "永远最 clean paper grade, GLM rescue paper-grade 全禁" 满足)
- [x] **A1.2** `p79/backends/` — backend 层 (`api_proxy` / `local_qwen` / `local_gemma` + `factory` + `base` + `action_utils` + `image_utils` + `heuristic`) (2026-05-16 late-night: 3-AI cross-audit Mode A+B+C, 16 findings → 11 fixed + 5 deferred (4 prose + 1 P2 infra); commits `4c559d2`+`abb0900`; B-406~B-416; key fixes = B-406 coord-type strict 851 row 2-AI A+B P0 OOB + B-407 type no-target 23 row 1-AI Mode B P0 OOB + B-408 dom_mode 3-AI overlap enum + B2 fail-loud P1 OOB + B-409 multiple_actions full-field sig per Q2=A + B-410 yaml temp dead-config warning + B-411 paper_grade local wire defense-in-depth + B-412 naked scroll strict + B-413 repair path detailed reason + B-414 first_element_id role-anchored 7-callsite sibling fix + B-415 mock_<id> canonical naming + B-416 b64.encode redundant; pytest 414/414 GREEN; user Q&A all-recommended Q1/Q2/Q3=A + auto-default take-all; **P0-3 + P0-4 + P1-7 + P2-2 paper §3.5 + §4 disclose DEFER next codex prose round** + **P2-5 mock_strategy DEFER test infra round**)
- [x] **A1.3** `p79/envs/` — VWA wrapper + locator dispatch + observation pipeline (2026-05-17 v9 third-pass after §147 v8: 3-AI cycle + 1 deeper round (heuristic family + scaffold completeness); commits `5799fda`+`5fa579f`; B-417 (defer) + B-418~B-425; key landings = D1=A delete HeuristicDomBackend family (B-425, 0/53924 paper-grade usage confirmed by 3-AI deeper) + B-418 type+Enter tab-switch parity sibling to B-157 + B-419 snapshot_form_fields error sentinel + B-420 select_option_meta env_dispatch_meta 195/738 archive empirical + B-421 locator_route_meta regression test + B-422 named injection-distance constants + B-423 beforeunload accept policy + B-424 form snapshot value_djb2 hash closes §147 P2-B5; pytest 417/417 GREEN; **B-417 iframe descent DEFER follow-up** (2h+ scope low Phase 1a ROI) + **D2 paper §3 GRL framing DEFER to user's parallel session** (`docs/checkpoints/stress_grl_audit_2026-05-17.md`))
- [x] **A1.4** `p79/experiment/som.py` + mark 抽取链 — SoM + phantom mark layer (2026-05-17 deep night: 3-AI cycle 22+ findings → user-triaged 8 fixes B-449~B-458 + 2 deferred to A1.25 GRL session B-453/B-455 + 2 dropped P0-1-B/P0-4-C* per user reframing, §174; key fixes = B-449 delete `degraded_som` schema bool (A+C overlap, paper §3.5 line 109 commitment + schema commitment 同时收口 via `mark_count == 0` aggregator) + B-450 `select_option_meta_primary/retry` ghost field schema add + B-451 `_shared_vl_utils.build_mode_prompt_dispatch_table()` canonical factory (4-consumer drift surface consolidation, paper-2 mechanism §138 unfreeze prep) + B-452 coord_type pixel/normalized inference (codex unique OOB, audit trail honesty) + B-454 bbox unit contract docstring + B-456 p95 opt-in `strict=True` for figure renderers + B-457 paper §3.5 regex anchoring prose precision + B-458 `condition_map.md` add 4 phantom condition_ids; pytest 410→412 GREEN; **B-453 select_option JS contract + B-455 CSS dropdown multi-menu DEFER** to user parallel A1.25 GRL session per user instruction 2026-05-17; 4-chunk commits `3a2d204`/`e5af0e7`/`901956d`/this)
- [ ] **A1.5** `p79/utils/` + `p79/cli/` — 9-fix utility + CLI substrate (2026-05-16, commit `2d3286c`, §156, B-211~B-229; 21 findings → 9 fixed + 5 deferred + 5 disclosed + 2 companion paper). **Scope clarification (per A1.15 Q4 finding 2026-05-16)**: ladder label was originally `p79/experiment/runner/` but actual audit covered utils + CLI + watchdog hardening; `p79/experiment/runner/` remains un-audited — split out as new A1.5b slot below.
- **A1.5b** `p79/experiment/runner/` — **2-pass split** (single-file >1500 LOC trigger per `feedback_split_large_scope`; `main.py` = 2320 LOC, one /stress invocation = uneven coverage). Dedicated slot post-A1.15 ladder bookkeeping fix (2026-05-16).
  - [x] **A1.5b Phase 1** control plane (main.py L1-984 + helpers.py + cross-pipeline substrate logger_v2 / types / schema_migrations) — 2026-05-17 deep night, **3-AI cycle 27 findings → user-triaged 21 fixes B-485~B-505** (codex scheduled fire pattern @03:24:30 限额 reset preserved cross-AI precision; renumber sweep B-460~B-480→B-485~B-505 post mid-session collision discovery with A1.20 + A1.25 GRL Chunk 2 catalog reservations). §175. **Key fixes**: B-485 resume fingerprint sha256[:16] of cfg+model+prompt+transformers (3-AI overlap) / B-486 quarantine-rerun for exception-path success=False (gemini OOB) / B-487 Option K covariate substrate (codex OOB, closes §168 Pre-fire 闭环 last-mile gap with EpisodeSummaryV2.wallclock_start/end + condition_summary.episode_summaries) / B-488 in-progress-aware archive (gemini OOB, watchdog companion patch excludes `.stale_*` from orphan cleanup) / B-489 run_meta atomic (B-331 sibling-prop gap) / B-490 escalation_count recompute (paper §4 router fire-rate headline defense) / B-491 staging merge hash-based dedup (codex OOB, closes resume+reset event-loss) / B-492 JSONL first-create dir fsync / B-493 partial recovery B-180 guard / B-495 artifacts atomic / B-497 control_intervention schema slot (runtime write path = Phase 2 deferred) / B-498 timeout 300s→1800s + ntfy / B-499 multi-site symlink / B-500 try/finally cleanup / P2 batch B-501~B-505. **Disclosure-only**: B-494 SOM-as-DOM naming (paper §3.5 footnote, file rename + 14-consumer prop deferred to Phase 2) + B-496 site-uniform diagnostic control thresholds (paper §3 footnote). Tests 20/20 PASS at every chunk boundary. 6-commit chain `eac0d2b` (Chunk 1 schema+write) → `528d81d` (Chunk 2 resume+atomic) → `3cd175e` (Chunk 3 quarantine+archive) → `00df3e4` (Chunk 4a substrate hardening) → `73037c7` (Chunk 4b schema+UX) → `89cb2bf` (Chunk 5a P2) → this (Chunk 5b docs+catalog+renumber). Phase 1a fire green-light **on** post-A1.5b Phase 1.
  - [ ] **A1.5b Phase 2** data plane (main.py L984-2320 `_run_episode` body + io_utils.read_jsonl_dedup full audit + experiment_watchdog race surface) — carries B-494 mode-aware artifact filename rename + 14-consumer propagation, B-497 `_run_episode` runtime write path for control_intervention, aggregate_trajectory_covariates.py sync to consume new wallclock_start/end fields (B-487 split deferred half); P0-4 finalize race full audit (aggregator covariate route per B-385 already partial-closed)
- **A1.6** `p79/experiment/analysis.py` — **2-pass split** (single-file >1500 LOC trigger per `feedback_split_large_scope`; 1842 LOC, A1.6 已有 17 findings B-237~B-253 §158 但未 close, 说明需要 deeper round).
  - [ ] **A1.6a** FP architecture half — `compute_adjusted_success` retirement traces + `scored_task_count` canonical + N/A task-load exclude pathway + benchmark_noise removal (lines ~1-900)
  - [ ] **A1.6b** Pareto + decision-test half — `analyze_run` template + Pareto frontier + decision-test surface + adjusted_sr archive warning (lines ~900-1842)
- [ ] **A1.7** `p79/experiment/conditions.py` + `configs/exp_v2_*.yaml` — condition matrix 生成 + 36-cond config 全家 (2026-05-16: 3-AI cycle 17 findings, 12 fixes B-261~B-269 + B-262 advisor-pending, §159; key fixes = B-261 phantom_dom legacy alias retire (resume:true × shared condition_id silent overwrite) + B-263 dead modes retire (`dom_only`/`hybrid` v1 artifact) + B-264 三头案 N_conditions 统一 42 + B-265 vision require_image guard + B-266 pilot T0 abandon + B-268 router_learned doc↔code drift + B-269 baselines.run_b0 retire)
- [ ] **A1.8** `p79/experiment/{types.py, schema_migrations/, io_utils.py, logger_v2.py}` — schema + JSONL + dedup (2026-05-16: 3-AI cycle 18 fixes B-280~B-297, §161; key fixes = B-283 string-truthy `bool("false")=True` paper §1 SR inflation guard + B-289 episodes_dir fsync at construction + B-291 image_meta_recorded schema separator + B-288 errors='replace' RAPL race + B-313 Option K Trajectory Event Log (A1.17 cross-talk insight); pytest 281→316 GREEN)
- [ ] **A1.9** `p79/experiment/{metrics.py, energy_tracker.py, environment.py}` — cost / energy / env wrapper (2026-05-16 late: 3-AI cycle 22 fixes B-320~B-341, §165; key fixes = B-320 HARDWARE_PROFILES `a100_pcie_40gb` alias + fail-loud + B-321 `_average_measured_power` step boundary fix + `window_sample_count` + B-322 strict-aggregator-types entry guard (B-283 sibling) + B-326 paper §1 `B=10000→1000` mismatch + B-327 `clean_success_rate` excluding benchmark_noise + B-329 program_html eval skip retry + B-330 H3 axis universe → universe_6 + B-336 kwh_per_step deprecation raise + B-338 cost_usd nested-key validator + B-340 paper_grade flag propagation hard-block GLM rescue; merged via worktree `../p79-a1.9` → master `3a2fc70`)
- [x] **A1.10** `p79/experiment/{router.py, modules.py, state_change.py, checklist_module.py, tasks.py, config.py}` — utility 模块 cluster (2026-05-16 late: 3-AI cycle 28 findings, 17 fixes B-359~B-376 + 5 deferred P2 B-379~B-383 + 8 prose fixes, §167; key fixes = B-359 router 3 numeric thresholds dead empirically (3-AI overlap; dom_size 0.14% / dom_complexity 0.00% / text_length 0.14% fire rate per 5001-step empirical) + paper §3.5/§4.X.5 disclosure + aggregator `--audit-fire-rate` gate + B-360 B-09 split propagation to router input + 5 analyzer sites (2-AI overlap) + B-361 DEFAULT_CONFIG 3→6-mode canonical + B-362 sibling regex anchored 7 callsites (2-AI overlap, A1.4 SOM fix lineage) + B-368 learned-router skip rule router (codex unique P0 OOB) + B-372/373/374 paper §1 + §3.4.1 + §3.4.2 prose hallucinations (gemini unique P0 OOB) + B-377 §4.X.12 env var name fix; pytest 398→406 GREEN; **B-369 retry schema bump DEFER** paper-2 per Q3=B + **B-370 state_digest after-fields DEFER** gated on schema bump)
- [ ] **A1.11** `p79/{utils/, cli/, logging/}` — 辅助 (CUDA workaround / CLI 入口 / structured logging)
- [ ] **A1.12** `tests/` — pytest 覆盖度 + invariant 测试是否对得上当前实现 (2026-05-16: 17 findings B-342~B-358, §166; key fixes = B-342 prereg gate fixture schema_version + B-343 cell topology Phase 1a drift + B-344 io_utils invariant net + B-345 shell-script smoke + B-346 B-91 evaluator guard test + B-347 zero-SE floor stale + B-348 source-grep behavioral retrofit + B-349 backends parity + B-350 pytest config hygiene + B-351 registry fixture split + B-353 external smoke + B-354 [test] extras; suite 333 → 398 GREEN; **B-352 learned router test DEFER** to Pass-2 fire − 1 week per T1-4=B)

#### A1-管线 (`scripts/`)

- [ ] **A1.13** `scripts/queues/queue_{baseline,phantom_*,chain}.sh` — launch + 3-way collision (2026-05-16: combined batch with A1.14, commit `3cd23f2`, §157, B-230~B-236; key fixes = B-224 auth gate hard-fail propagation + Bug 2 URL-locality preflight + RUN_ID nanos+PID+RANDOM collision defense + sibling-propagation defect class closed via `_lib_paper_grade_gates.sh` extraction)
- [ ] **A1.14** `scripts/queues/queue_phase1_paper_grade.sh` + `scripts/preflight_v2.sh` — orchestrator + pre-launch gates (2026-05-16: combined commit with A1.13 `3cd23f2`, §157; B-91 VWA submodule lock check + preflight 触发 paper-grade gates)
- [ ] **A1.15** `scripts/maintenance/experiment_watchdog.py` + 3 control scripts — watchdog stack + 6-layer auto-clean protocol full audit + Pre-fire 闭环 batch (2026-05-16 late-night, §168, B-384~B-394; 22 findings 3-AI cycle → 11 fixes + Option K critical path closure). Key fixes = B-384 Hook C session-cleanup trajectory emit (codex unique P0 OOB) + B-385 schema doc + P0-4 reframe via aggregator covariate route + B-386 race-window disclose §4.X.13 stub + B-387 reddit DOM regex inert per-site dispatch (Claude unique P0 OOB, empirically verified 5/5 logged-in 0→5 detect) + B-388 runner staging pickup helper + B-389 NEW `aggregate_trajectory_covariates.py` Option K covariate aggregator + B-390 end-to-end smoke + B-391 `_run_auto_digest` silent dead path (codex unique P1 OOB) + B-392 `_purge_digest_records_batch` O(N·M)→O(1) + fsync + B-393 `_load_state` fail-closed corrupt state + B-394 `wait_for_reddit_then_rederive.sh` retired (T6=(a)). Remaining un-audited slice (deferred to A1.15b): `glm_cell_autoupdate.py` / `myriad_watcher.py` / `batch_digest.py` GLM sidecars + full glm/ 5-sidecar cron stack audit.
- ⏸ **A1.15b** GLM sidecars cluster — **DEFERRED post-workshop** (set 2026-05-17): 6 文件 / 4826 LOC, 含 `glm_diagnosis_sidecar 1996` + `glm_batch_digest 1204` 单文件破 1500 阈值, 重新 split 需 3 sub-chunk (critical cron / batch jobs / read-only). **Operational layer, not paper-grade blocker** — GLM sidecar bug 不影响 paper §1 hero number / phantom 4-fold / drop-one / AUROC. Resume condition: post-workshop submission OR cron incident 触发 (whichever first). Files: `scripts/maintenance/glm/{glm_cell_autoupdate, glm_playbook_refresh, glm_batch_digest, glm_diagnosis_sidecar, myriad_watcher, error_scan}.py` + cron schedule integrity.
- [ ] **A1.16** `scripts/provenance/snapshot_*` — env + VWA fingerprint (2026-05-16: commit `5e11721`, §160, B-273~B-279; 7 fixes incl. probe URL static-asset / HF loaded SHA double-field / evaluator hash canonical + Gemma3-VL gated hard-fail / scope expansion + docker source SHA; mechanism-script subset 5 bugs D-1..D-5 deferred per paper-2 scope)
- [ ] **A1.17** `scripts/vwa/` + `RESET_BEFORE` protocol

#### A1-外部 (VWA submodule)

- [ ] **A1.18** VWA submodule `p79-patches` branch — evaluator + helper_functions + LLM-judge guard (2026-05-16: 15 findings B-254~B-268; gemini OOB P0-1 viewport paradox catch + codex F10 IP 794-hit deepening; full clean: 913 task configs IP→placeholder + paper §3.5/§4.X.11/§4.X.12 disclosure + 3-layer SBOM lock + 5 P1 code fixes; chronicle §159; memory `reference_vwa_submodule_p79_patches.md`)

#### A1-分析管线 (clean-run 下游, 但 code 本身是 pre-data audit)

- [x] **A1.19** `scripts/analysis/aggregate_*.py` — aggregator 层 (sr_fp / phantom_lift / cross_site / routing_auroc / failure_modes / cost_electricity / phase1_prereg_gate / phantom_meta / trajectory_covariates / lib/run_registry) (2026-05-17 deep night: 3-AI cross-audit Mode A+B+C, 11 artifacts pre-fire scope, 20 unique findings → 13 fixed (4 P0 + 9 P1) + 4 P1 deferred (P1-2/10/11/12 per Q&A bottom-tier) + 3 P2 backlog; commits `<pending>`; B-426~B-438; key fixes = B-426 SE floor 1.0pp prereg-disclosed Agresti-Coull-style archive-median-anchored (2-AI A+C OOB) + B-427 use_adjusted default flip §139.8 retired layer (3-AI A+B+C overlap) + B-428 evaluate_h2_cost margin 10→20 + K-of-N→ALL-pass strict (gemini OOB reframe) + B-429 mixed-universe lift L664-670 per-comparison universe (codex P0-1-B*) + B-430 hashlib.sha256 swap PYTHONHASHSEED reproducibility (codex P1-1-B*) + B-431 HKSJ row at k=6 anti-conservative-DL-Wald fix (codex P1-3-B*) + B-432 7-bucket docstring (A+C) + B-433 datetime parsing fragile ts-comparison (Claude OOB) + B-434 LEGACY_MODE_ALIAS cross-tier collision warn + B-435 cond_dirs first-match fail-loud + B-436 multi-rerun seen_runs dedup + B-437 APPENDIX demote stale "Hero" wording + B-438 Superiority vs TOST disjoint-hypotheses warning; pytest 413/414 GREEN; user Q1=C 完整P0+P1 + Q3 P0-1 archive-grounded 1.0pp const + Q4=A code only paper §1 prose 留 codex round + Q5=A await user push confirm)
- [x] **A1.20** `scripts/analysis/figures/*.py` — figure 脚本全家 (fig0a-3d + mechanism + venn + shared lib/panels.py infra) (2026-05-17 02:00 deep night: 3-AI cross-audit Mode A+B+C, 16 artifacts (10 Claude + 6 codex 互补 + 5 gemini prose/caption), 25 unique findings → 17 fixed (10 P0 + 7 P1) + 2 deferred P0 (B-463 fig0e per Q3=B + B-465 fig1ab/fig1c full refactor) + 1 deferred P1 (B-470 with A1.19 P1-2 advisor batch) + 5 P2 backlog; commits `<pending>`; B-459~B-477 (B-numbers collided with parallel GRL A1.4+A1.25 batch consuming B-449~B-458, renumbered); key fixes = B-459 HKSJ schema CSV col bump (A1.19 B-431 figure-layer propagation gap) + B-460 fig0c_phantom_lift_bars mixed-universe bar rendering (A1.19 B-429 propagation) + B-461 B2 silent missing 12 figures via shared `scripts/analysis/figures/lib/panels.py` infra (3-AI A+B+C overlap, closes sibling propagation reservoir) + B-462 fig_meta_forest HERO label drift APPENDIX demote (A1.19 B-437 propagation) + B-463 fig0e archive-only source defer per Q3=B (Makefile commented out) + B-464 fig0f uniqueness inflation direction-bias gate (codex OOB) + B-465 fig1ab/fig1c minimal patch latest-glob→run_registry + strict success (full refactor defer dedicated session) + B-466 section1_intro "Zero image tokens" prose vs prereg §2.6 footnote correction (gemini OOB, paper §1 hero defense critical) + B-467 H1 estimand schizophrenia 4-mode vs 6-mode prereg gate footnote (gemini OOB) + B-468 hardcoded 234/210 → scored_task_count 224/205 canonical + B-469 strict `is True` 6 figures (B-283 sibling propagation) + B-471 SECTION103_LOSS retired (Q13=c) + B-472 B2 baseline_order canonical + B-473 footer 7-bucket sync + B-474 fig2 global vmin/vmax cross-baseline + B2 PNG output + B-475 fig1c masked None bars + fixed y-limit + B-476 fig0c caption N denominator transparency; pytest 417/418 GREEN (1 pre-existing GRL session fail unchanged); user Q1=B 完整 P0+P1 + Q2=A shared lib helper + Q3=B defer fig0e + Q4=A code+prose-stub only + Q5=A await user push + bottom-tier 全 推荐)
- [x] **A1.21** `scripts/analysis/preregistration_decision_test.py` + `scripts/analysis/lib/run_registry.py` + `results/phantom_paper/run_manifest.yaml` — decision test + registry + paper-grade promotion (2026-05-17 morning: 3-AI cross-audit Mode A+B+C, 10 artifacts, 29 unique findings → 21 fixed (11 P0 + 10 P1) + 5 deferred (P0-10 SE floor reframed user-confirm option A retain + P1-6 4-vs-6-mode re-aggregate post-data + P1-14 P-prompt parse_valid off-scope + P1-15 B0 GLM disclosure paper §8 round + P2 backlog); commits `18fd8b7` + `c4c033d` + `5b08d79` renumber + `<chunk3 pending>`; B-513~B-533 (sed renumber from B-478~B-502 mid-session due to parallel A1.25 GRL Chunks 2-3 + A1.5b collision B-479~B-512); key fixes = B-515 NEW `aggregate_phase1_full_prereg_decision.py` canonical full R1-R5 producer (codex P0-3 OOB; replaces retired-DL decision_test as paper §1 framing producer) + B-513 retire DL meta + magnitude + superiority 3-test compound from `evaluate_h1` (codex P0-1 OOB; prereg §1 L68-86 locks SINGLE FE superiority) + B-514 retire `_effective_gate_pass` heterogeneity rescue branch (codex P0-2 OOB; cap-only NOT rescue per prereg L323+L340-342) + B-525 NEW `validate_run_manifest.py` 7-check paper-grade promotion validator (Claude P0-8; surfaces 22 errors current pre-fire state) + B-526 NEW `canonical_cells.py` single-source PHASE_1A_PLANNED_CELLS (Claude P0-7 OOB; closes triple-source-of-truth) + B-524 `--run-manifest` arg propagation `get_all_cells(manifest_path=...)` (codex P0-4 OOB; was provenance theater) + B-518 H2(a) per-task ratio paired estimand replacing median-of-marginals (Claude P0-9 + prereg §2 H2(a) prose lock amend 2026-05-17) + B-527 cost integrity 0.0 short-circuit + cross-baseline unit basis disclosure (3-AI A+B+C overlap, B0 telemetry hole gemini OOB reframed) + B-517 percentile bootstrap p-value (3-AI overlap; was percentile CI + normal-approx p mixed method) + B-522 `expected_n` enforce canonical (Claude+codex 2-AI; archived backwards-compat preserved) + B-530 `get_aggregator_cells()` lazy fn (2-AI; module-import freeze) + B-531 cross-site baseline collapse (codex OOB; B0/B1/B2 silent misattribution) + B-532 B2 cost markdown (codex F7 sibling propagation to A1.20 B-461) + B-533 OSF lock 36→42 align prereg §4 L438 (codex F10) + prereg §2 H1 SE floor anchor prose expand (P0-10 user Q2=A retain 1.0pp Agresti-Coull + archive-median 0.98pp + sensitivity table); pytest 420/420 GREEN (1 pre-existing GRL session VWA submodule SHA drift unchanged — A1.25 Chunk 1 bump `5d8fc2f` outside scope); user Q1=A full P0+P1 batch + Q2=A SE floor retain option A + Q3=A per-task ratio + prereg amend "我自己判断" + Q4=A new canonical file + Q5=A wait-fix-all + Q6=A wait push + bottom-tier 全 推荐)

#### A1-横向 (cross-cutting — 跨文件 class / contract 层)

> 这些 surface **不属于** 单文件 audit 范畴, 用 /stress code-audit mode 时显式指定 "spans multiple files, audit contract not implementation"。Memory `feedback_split_large_scope` 不适用 (因为 cross-cutting 才是它们的 scope 本身)。

- [ ] **A1.22** Cross-baseline 输出契约 (B0 / B1 / B2 step_record + condition_summary 对称性) — 跑同 condition 1 task on B0/B1/B2, diff `step_record` 字段集 + value type + value range, 找 silent asymmetry。Memory 2026-05-15 retract "B0/B1 设计不对称 disclose 即可"作废, 全 align by default → paper §1 hero number 跨 3 model 平均的安全性 gate。Artifacts: `p79/agents/{proxy_api_agent,qwen3vl_agent,gemma3vl_agent}.py` + `p79/backends/{api_proxy,local_qwen,local_gemma}.py` + `p79/experiment/types.py`
- [ ] **A1.23** Concurrency + race contract (launch × watchdog × cron sidecar interaction) — "同 site 单 baseline" hard rule **runtime enforcement** (queue_chain.sh launch-time check 不够, runner mid-flight 多 PID 起来怎么办); watchdog `auth_refresh` × runner page navigation race; A1.17 Chunk 2 Option K Trajectory Event Log 已收一部分但 cross-cutting "concurrency contract test" 没单独 audit。Artifacts: `scripts/queues/queue_chain.sh` + `scripts/maintenance/experiment_watchdog.py` + `scripts/maintenance/glm/myriad_watcher.py` + `p79/experiment/runner/main.py` + `p79/experiment/logger_v2.py:log_trajectory_event`
- [ ] **A1.24** `scripts/maintenance/clear_tasks.py` + 半删 recovery 协议 — CLAUDE.md hard rule "删除 task 结果统一用 clear_tasks.py" 的官方入口, partial-clean bug (summary 删了 steps 没删 / artifacts 残留) → rerun 拉到污染数据 silent 风险; A1.15 partial 边缘覆盖但 single-purpose surface 重要性高
- [/] **A1.25** GRL (Generated Runtime Layer) — `p79/envs/locator_dispatch.py` + `p79/envs/vwa_wrapper.py` + `external/visualwebarena/browser_env/actions.py` (submodule) — user-invoked /stress on full GRL surface (P79 runtime 容错层 on top of VWA upstream `89f5af2`). 5-chunk decomposition: Chunk 0 discovery (6 net-new GRL items beyond user 15-item list — N1/N2/N3 hover/clear/upload locator route DEAD CODE / N4 CSS dropdown asymmetry / N5 min_free_vram OOM gate / N6 get_all_tab_titles helper). **Chunk 1 ✅ COMPLETE** (2026-05-17 deep night, §173, B-439~B-448 = 10 fixes; 3-AI cycle = Claude 10 / codex 8 / gemini 7 = 25 unique findings; user Q1=A 全 P0 + low-risk P1, Q5=B wait-fix-all). **Chunks 2-4 PENDING** = observation enrichment / action policy + cross-baseline / VWA upstream patches. Tracker: `docs/checkpoints/stress_grl_audit_2026-05-17.md`. Phase 1a launch blocked on Chunks 2-4 per Q5=B.

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

- [ ] **A2.1** Research question framing — "phantom routing space" 假设是否良构 + falsifiable; "4-fold drop-in property" 是 1 个 property 还是 4 个独立 claim 包成一个; phantom 概念与文献 (cascade / routing-mix / mixture-of-experts) boundary 在哪 (reviewer R1 originality 攻击) (artifacts: `paper_drafts/section1` + [[paper_planning]] §1 + memory `project_phantom_space_axes_format_not_information`)
- [ ] **A2.2** Comparison rigor / control design — B0 / B1 / B2 是哪条轴的 control (capability / family / deployment-class)? B2 与 B1 "matched-capability cross-family" 在 4B 参数对齐外是否还有别的对齐要求 (训练数据 / alignment 配方 / instruction tuning)? (artifacts: [[paper_planning]] §15 prior-work table + [[实验笔记]] §138)
- [ ] **A2.3a** Statistical design — **power + sample** — N=6 cells × 观测 effect size 1-3pp 的 power 计算 (per-cell + pooled meta); MDE (min detectable effect) at α=0.05 / 80% power; cell-level vs task-level unit-of-analysis 边界 (artifacts: `pre_run/preregistration.md` §2.4 + `pre_run/power_analysis.py` 注释)
- [ ] **A2.3b** Statistical design — **meta method** — DerSimonian-Laird vs REML + Hartung-Knapp at k=6 cells; small-k DL bias well-known, REML+HK 是否更稳; per-site fixed-effect alternative (artifacts: `pre_run/preregistration.md` §3 + advisor sync confirm)
- [ ] **A2.3c** Statistical design — **multi-test correction + equivalence** — H1-H8 family-wise Bonferroni / Holm / BH-FDR 选择依据; TOST δ=1.0pp 来源 + 文献对齐 (Lakens 2017 / Wellek 2010); K-of-N transparency-only 重分类的 prereg trail 强度 (artifacts: `pre_run/preregistration.md` §4 + Appendix A 2026-05-13 K-of-N entry)
- [ ] **A2.4a** Evidence-claim coupling — **within-axis** — 4 层 (Outcome / Macro / Micro / Efficiency) × 4 cross-X (task / mode / site / model) = 16 sub-cell 是否真支撑 paper §1 hero; 单 sub-cell NaN 时 hero 数字 fallback 协议 (artifacts: [[paper_planning]] §3 + `analyze_run` template)
- [ ] **A2.4b** Evidence-claim coupling — **cross-axis generalization** — cross-site 2 site (cls+red) 够不够撑 generalization claim (R3 sensitivity); cross-model 3 baseline 中 B0 API 异类 vs B1/B2 同 deployment-class 对 cross-model claim 的影响 (artifacts: [[paper_planning]] §3 + §21 + §5 R1-R5 framing)
- [ ] **A2.5** Operationalization — rule-based router 的 "task 属性" 定义边界 + leak risk (用 task description 训练 → test 时 leak);learned classifier 的 feature set (TF-IDF + binary + browser meta) 信号源 vs leak 边界;5-fold site-stratified CV 是否解决 site leak (artifacts: [[paper_planning]] §8 + `p79/experiment/router.py`)
- [ ] **A2.6a** Scope / external validity — **Phase 1a 集**: 只有 cls + red → R3 framing risk; Phase 1b shop 推迟 + WA 缺席的下游影响; sequential 2-pass 不能并行 site 是否削弱 fairness claim (artifacts: [[paper_planning]] §5 R1-R5 conditional tree + §6 Critical Risks + `pre_run/preregistration.md` §2 framing decision rule)
- [ ] **A2.6b** Scope / external validity — **phantom 概念域**: phantom space 是 VWA-specific / web-agent-general / LLM-general? Reviewer R5 跨族外推攻击边界; "无 annotated image" boundary 在 LLM-general 上的可外推性 (artifacts: [[paper_planning]] §5 + memory `project_phantom_space_axes_format_not_information`)
- [ ] **A2.7** Confound register / known asymmetries — B0 (API) vs B1/B2 (local) deployment 异类;B0 max_new_tokens / GLM parse fallback / quantization 非对称;A100 docker stack vs DGX→quark stack 切换的实验环境变化 (artifacts: CLAUDE.md Guard Rails + [[实验笔记]] §139 B-86 + memory `project_paper_hook.md`)
- [ ] **A2.8** Pre-registration completeness — H1-H8 primary/exploratory/post-hoc/deferred 族 declaration 边界;§4 locked analysis choices 全覆盖;§6 witness mechanism (Git commit + advisor email + OSF DOI) 防 post-hoc 修改的强度; **§C5 "site-asymmetric viability empirical finding" 是 H10 之内的 pre-registered subhypothesis 还是 post-hoc descriptive finding** (cross-link C5, archive sim pre-fire 已知 cls+2.02pp / red-3.95pp → narrative pre-registered 边界必须明确) (artifacts: `pre_run/preregistration.md` 全文 + `pre_run/osf_lock_manifest.md`)
- [ ] **A2.9** Reporting + ethics — **NeurIPS / ICML 2025+ requirement**: compute cost statement (A100 wallclock × 3 model × 6 mode × cls+red + Phase 1b); broader impact + ethics statement; data-rights disclosure (VWA submodule data 来源 / Google `gemma-3-4b-it` 用法 / GPT-4o-mini judge usage). Submission-ready scope 不卡 Pass-1 launch (artifacts: NeurIPS 2025 checklist + paper_drafts 末尾 reporting section TBD)

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

- [ ] Gemma3-VL Tier 1-3 接入 — agent / backend / factory / 12 configs / queue / orchestration / A100 smoke ALL PASS ([[实验笔记]] §140)
- [ ] A100 venv 全栈 dep
- [ ] **#11 A100 VM VWA docker bring-up** — ⭐ gates Pass-1 launch
- [ ] A100 playwright install
- [ ] **#10 analysis 层 3-model 改造** — gates §D, 不 gate launch

### B1. Pre-run lock 文档 (= launch gate, 引 `pre_run/` 不复制)

- [ ] `preregistration.md` status `draft → locked` (待 advisor 确认 A2.3a/b/c + v7 walk-back H9/H11 DEFER + Pareto reformulation)
- [ ] `locked_versions.md` / `model_card.md` / `dataset_card.md` — B2 三模型对齐
- [ ] env + vwa provenance snapshot on A100 host
- [ ] `pre_rerun_audit.md` + `reeval_audit_protocol.md` 走查
- [ ] `osf_lock_manifest.md` 8-step (可 launch 后并行)
- [ ] **§C LR feature step-0 schema dry-run** — 跑 1 task per (3 baseline × 6 mode) = 18 mini-runs, 抽 `step_record[0]` verify §C2 8-dim LR feature 全字段 capture (site_one_hot / capability_tier_one_hot / has_reference_image / intent_color_regex / intent_compare_regex / intent_search_regex / intent_token_count / axtree_element_count). **Why**: §B-baseline 完成判定只 verify `condition_summary_v2.json` per-cond aggregate, 不 verify step-record-level fields — 如果 B0 step-0 没 `axtree_element_count` 或字段名 drift 或 vision mode step-0 null, Pass-1 跑完 1-2 周才发现 → Pass-2 LR pipeline 整 cell drop。Gates **§B-baseline launch**。Artifacts: `p79/experiment/types.py:StepRecordV2` + `p79/experiment/logger_v2.py` write path + `scripts/analysis/feature_extract_*.py` (TODO if missing)

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

- [ ] `scripts/analysis/l1_archive_simulation.py` repeated stratified 5-fold × 10 repeats (Q4 fix landed 2026-05-16)
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
- [ ] Site-asymmetric viability empirical finding written up (paper §6 main narrative — cls visual-rich vs red text-dominated routing behavior contrast) — **⚠️ pre/post-hoc framing status 必须在 A2.8 prereg lock 中明确**: archive sim 2026-05-16 pre-fire 已知 cls Variant B +2.02pp / red -3.95pp → 此 narrative 是 H10 之内 pre-registered subhypothesis (confirmatory) 还是 post-hoc descriptive (exploratory)? Reviewer R3 经典攻击点, prereg.md 必须二选一并 Git/OSF witness lock
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

| M | 判定 | gated on | A100 wallclock (happy / realistic) |
|---|---|---|---|
| M0 infra ready | §B0 全勾 (重点 #11) | — | — |
| M1a 实现层 audit 放行 | §A1 全勾 (或 open item disclosed) | — | — |
| M1b 设计层 audit 放行 | §A2 全勾 + preregistration locked (incl. v7 H9/H11 DEFER + Pareto reformulation) | advisor sync (hard 卡 A2.2 + A2.3a/b/c + A2.8; A2.1/A2.4/A2.5/A2.6/A2.7/A2.9 可 Claude self-audit + advisor batch confirm) | — |
| **M2-baseline clean run done** | §B-baseline 全勾 (36 cond, paper §1 hook 数据) | M0 + M1a + M1b | **~1-2w happy / 2-3w realistic** |
| M3 evidence 分析 done (paper §1 hook) | §D3 全勾 | M2-baseline + #10 | mostly parallel to M2-router |
| **M2-router clean run done** | §B-router 全勾 (6 cond, paper §6 H10 数据) | M2-baseline (per-task oracle labels needed for LR train fold) | **~3-5d happy / 5-7d realistic** |
| M4 learned router done | §C5 全勾 (LOCO + within-cell CV + Pareto H10 verdict) | M2-router + #10 | post-M2-router |
| M5 workshop-ready | M3 + M4 | — | **~1.5-2.5w happy / 2.5-4w realistic** |

**Wallclock 估计协议** — `happy` = 36-cond × per-cond happy-path × sequential 3-baseline; `realistic` = happy × **1.3-1.5×** historical overhead band 含: (a) watchdog auto-clean 重跑 ~5-15% episode (Bug-9 stack 实证), (b) GLM fallback parse-error 重跑 (B-86 open, B0 cond 影响最大), (c) sequential 3-baseline 中任一 baseline lag 顺延全 site, (d) A100 docker stack random hiccup, (e) Phase 1a 数据 land 后发现 schema drift → partial re-extract。Real-world fire 中位 = `realistic`, M5 user-facing ETA 报 `realistic` 边界对齐心理预期。

**Wallclock savings vs v6 cascade**: v6 2-pass 72-cond = 2-4 weeks A100; v7 2-pass 42-cond = **1.5-2.5 weeks A100 happy / 2.5-4 weeks realistic** (~5-7 days saved on happy-path). Critical path 缩 mostly on M2-router (cascade 6 cond/cell × 6 cells = 36 router-cond → learned 1 cond/cell × 6 cells = 6 router-cond).

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
