---
type: action-ledger
status: rolling
updated: 2026-05-22
---

# Next Steps — Forward Action Ledger

> **Future-only**. **Roadmap = §1 `tasks.base`** (dynamic — edit `_status/tasks/*.md` frontmatter). Live state 不在这里:
> - Today / 瓶颈 / cron health → [[PLAYBOOK#§1]] + [[PLAYBOOK#§2]] (🤖 GLM @daily)
> - Real-time active runs / GPU → `make active` CLI (DGX only; fire 在 A100)
> - Paper-grade fire verdict → `paper_grade_check.py` (一条命令, §0) + 每 6h cron
> - Cell snapshot (active / pending / done) → `cells.base`
> - Paper section progress → `status.base`
> - 过去 chronicle → [[实验笔记]] (latest §271, 2026-05-22)
> - Strategy / theory → [[paper_planning]]
> - **Phase 1 执行计划 + audit checklist** → [[phase1_plan]] ⭐ canonical
> - OSF DOI lock workflow → [[osf_lock_manifest]] · Compute infra → [[COMPUTE_INFRASTRUCTURE]]
>
> 🔧 新数据 → `make analysis` (~5-10min). Cron 每 10min 自动 sync cell frontmatter.

---

## §0 SESSION HANDOFF — 新 session 接手 ⭐ 先读这个

> [!important] **掌握现状 = 跑命令拿 LIVE 状态, 别信本文/笔记的硬编码数字 (秒级 stale)。** 笔记 §265-271 = WHY; 本节 = HOW (拿 live + 盯什么)。

**① paper-grade verdict (一条命令; 每 6h cron 也自动跑 + ntfy)**:
```
ssh condense-a100 'cd /home/ubuntu/workspace/p79 && .venv/bin/python3 scripts/maintenance/paper_grade_check.py'
#  → VERDICT: OK completed_ok=N inprog=[Rxxxx mode ep=N img=N errflood=0]   /   ISSUES=...
tail -40 logs/cron/paper_grade_check.log     # 每 6h 自动检查历史 (00:30/06:30/12:30/18:30)
```

**② fire 死活 (fire 在 A100; `make active` 只扫 DGX → 空是正常)**:
```
ssh condense-a100 'pgrep -af "queue_phase1_paper_grade|run_experiment\.py" | grep -v bash'
ssh condense-a100 'cd /home/ubuntu/workspace/p79 && .venv/bin/python3 scripts/analysis/validate_fire_manifest.py'
```

**③ 写时状态 (2026-05-23 ~17:40 — 用①验证, 勿照搬)**: ✅ **Gate 3 fire 跑通续链** — cls chain **第 2/18**: **R31194** (B0 dom) DONE (224 ep, manifest bound) → **B0 som R2815** 续链跑中 (orchestrator `queue_chain.sh` 自动衔接 = 18-chain 自动化验证通过)。compact 后又修 2 个 binding-层 bug (§276): **B-1840** fire6_monitor 误报 (`_orch_up`/FIRELOG 命名没跟 `queue_chain.sh` 迁移 → 每 tick 假报 orchestrator GONE+FATAL; `d55e587`) + **B-1841** fire_manifest.json 是 Fire-6 stale bind R9755 → R31194 误判 ghost (Gate3 启动漏 reset manifest; 已 reset+bind R31194, validate exit 0, `da1f771`)。监控现可信 (fire6 不误报 + manifest Gate3-consistent)。前序全链 (launch 前) 见 §275: reboot A100 (NVML fix + fresh docker) → archive R9755+R11315 → **B-1839 per-condition docker restart** + 3-AI /stress (commit `06a303d`) → e2e 实测 → homepage fix → launch。**~12-21 天** (实测 B0 cls dom ~9.5h/mode)。Amendment 02 witnessed (`e338cb4`; §274)。B-1839/B-1841 OSF 不动 (衬底/binding 非 estimand)。本 session 全链 (§275): reboot A100 (NVML 580.159 fix → 60s 回来 driver 580.159.03 + nvidia-smi OK; fresh docker) → archive R9755+R11315 (probe → `_archive_pre_gate3_20260523`, A100+DGX, 都 non-paper-grade 进 36-fresh) → **B-1839 per-condition docker restart** + 3-AI /stress (gemini 独占 prereg-witness / codex 独占 sibling+timeout+app-layer / 全修 P1-1~3+P2-1~4, commit `06a303d`) → e2e 实测 (restart+token+完整 reset+seed pk≤84154 保留, 75s/cond) → homepage fix (reboot 遗漏, 见 §0⑤下 checklist) → launch。监控全自动: A100 fire6 (re-armed) + DGX paper_grade_check 6h + sync 15min。**~6-10 天** (B1/B2 latency cls 阶段实测; 实测 B0 cls ~8.3h/mode, 编排器估偏低 2× → 36-cond 重估 12-21 天)。canary/R9755 archive 不进 paper。Amendment 02 witnessed (`e338cb4`+tag+OSF, H1-strict 6-mode gate 不动; §274)。B-1839 OSF 不动 (衬底非 estimand, git tag `b1839-substrate-prefire` 凭证)。

**④ PENDING — 盯**:
> - **🔥 fire 跑中** (cls 18-cond, **第 2/18 = B0 som R2815**; 每 cond **B-1839 docker restart** fresh): 监控全自动 → **新 session 入口 = 上方 ① `paper_grade_check` verdict** (别照搬本快照)。Gate 3 步骤全 done: reboot✓/archive✓/B-1839✓/3-AI stress✓(06a303d)/wallclock✓/re-arm fire6✓/launch✓/**续链验证✓ (R31194→R2815)**/**B-1840 fire6 误报修✓(d55e587)**/**B-1841 manifest reset✓(da1f771, validate exit 0)**。**剩 (fire 进行中/完成后)**: ① **R31194 (B0 dom) 已 DONE → 可跑 `/diag` per-condition** (B0 dom cls; 见下 bullet) ② **cls 18 完 → red 前**: scratch redirect (results→/mnt/scratch; 全 36≈25G 需 scratch; 数据落 DGX 后做) + prune + **launch red 18** ③ B-1837 measure (见下) ④ Phase 1b shopping reset 含 docker restart (B-1839-fu, §4)。续链 monitor `bx856r7z0` 已 fire DONE (R31194→R2815 验证, exit 0); 坏 monitor `bqrj94mpl` 已 kill (B-1840 同源 eval bug)。
> - **✅ Amendment 02 witnessed + B-1839 substrate (pre-fire prereg DONE, 2026-05-23)** — Amendment 02: git `e338cb4` + tag + OSF, pre-data 封存; H1-strict 6-mode gate **不动**; (a) 修 power-table label (4-mode ADD ≠ 6-mode gate) (b) 拓宽 post-R5 reporting (C'-S/C'-R/F)。B-1839 docker-restart substrate: **OSF 不动** (衬底非 estimand, power 只更稳, 对称无偏), git tag `b1839-substrate-prefire` + paper methods disclose。**re-fire 不再 gated** (code+witness 全 done, fire 已 launch)。**post_r5_pivot 代码** 跨 producer, H1 失败后才需 (数据落地后写, 不卡)。详 §274/§275 + `AMENDMENT_02_GATE_LADDER_20260523.md`。
> - **⚠️ Gate 3 disk + "A100 算力 / DGX 数据" 架构** (2026-05-22 深查): A100 `/` 449G/485G **93%**, **docker 镜像 390G = 地板**(VWA 三站 essential, sudo du 实证; deleted-open=0 **非泄漏 → reboot 不释放磁盘**, reboot 只修 NVML+退化), 37G free; `/mnt/scratch` **366G free 闲置**。**正解 = A100 纯算力 / DGX(3.7T)纯数据**: ① results→`/mnt/scratch` symlink(canary 跑完后; 366G≈8-10 fire)② sync→DGX 永久(artifacts 实测**已同步**, R11558 两端 byte-identical; sync=mirror `--delete-after` 非 archive)③ verify-on-DGX 后 prune → A100 占用由 active-run 决定非累积。**Gate 3 go/no-go 第三因子**(除 wallclock+abort)。非 clean run 已归档 `_archive_non_clean_20260522/`(26 个), top-level 只剩 R9755 clean + canary。
> - **gallery / 实时 agent 操作 = on-demand (B-1828)**: `make gallery RUN=<run>`(annotate agent 行为 overlay + HTML; som_image fallback 已修 §15721)/`make gallery-all`(聚合)→ `http.server 8765 --directory results`。⚠️ **不自动刷新**(watchdog auto-refresh 默认 OFF 保 paper-grade 干净): resync 保证**原图随时在 DGX 可生成**, 但 gallery HTML 需手动 `make gallery`; DGX ≤15min sync 滞后, 近实时设 `P79_WATCHDOG_GALLERY=1`(run 期 overhead, paper-grade 不建议)。
> - **⚠️ Gate 2→3 transition = reboot A100**(2026-05-22 `make ntfy` 挖出): NVML `Driver/library version mismatch (lib 580.159)` 确认**仍坏** → **B1/B2 本地模型 Gate 3 blocker**(B0 proxy 不吃 GPU 故 canary 没事)。reboot 一举两得: 修 NVML + 重置 6 天 docker 退化(fresh 衬底)。**时序硬约束**: canary 必须**先跑完**再 reboot(否则清掉 canary 正测的退化窗口)→ canary 扛退化 = worst-case 下界, fresh Gate 3 严格更易。sync-a100 14:00 transient fail 已自愈(19:31 ✓, 每 15min 同步 A100→DGX 正常, results 有 off-box 副本)。
> - **canary 落地 → 跑 `/diag` 错因分析** (per-condition, B0×som×cls): `/diag` skill 3-tier (Tier-1 0-token) → `docs/analysis/vwa_classifieds/B0_som_classifieds_diag_digest.md`。拆失败结构 (早期 task=0/1 success=False / N/A 循环截断 / eval-timeout 分布) → 喂 Gate 2/3 评估 + paper-grade per-condition 错因表。**标准步**: 每个 condition 落地都跑 (per-condition 命名, 勿用 run_id)。
> - **monitor `b7z09re1f`** (DGX background, done/abort + retry 触发计数 → ntfy `p79-exp-dgx-spark`)。compact 后查 monitor: `make schedule-list` 或读 `/tmp/.../tasks/b7z09re1f.output`。
> - **B-1837** (eval 5-retry vs agent-step 0-retry → differential baseline rescue confound) = measure-then-decide: canary+Pass-1 量化 per-baseline eval-rescue rate, 再定 disclose (paper §3.5/§8) vs symmetric retry。NOT code change now (master_bug_catalog B-1837)。
> - **Gate 框架 (user 2026-05-22) — 全 done**: Gate1✓(B-1836) Gate1.5✓(/stress) Gate2✓(canary R11315 clean) **Gate3✓ fire 跑通续链 (R31194 done → B0 som R2815, 18-chain 自动化验证)**。下一战略节点 = cls 18 完 → red 18 → Pass-1 36 齐 → `make analysis` + H1 gate。**fire6 已 re-armed + B-1840 修命名漂移误报 (d55e587)**; **fire_manifest.json B-1841 reset 为 Gate3 (da1f771, validate exit 0)** — 监控+aggregation binding 全 Gate3-consistent。
> - **canary 期 ntfy = 2 层**: 每 6h paper-grade cron (✅/🔴 VERDICT+manifest) + monitor `b7z09re1f` (DONE/RUNNER_GONE/retry/evalerr)。⚠️ **A100 `fire6_monitor` healthcheck cron 已静音** (2026-05-22, canary=`queue_baseline` 非 orchestrator → `_orch_up` false + FIRELOG pin 死 fire6_relaunch log → 每 30min 误报 "orchestrator GONE + FATAL"; daily heartbeat 保留; 备份 A100 `/tmp/cron_fire6.bak`)。**Gate 3 转 full Fire 必须 re-arm**: `ssh condense-a100 "crontab -l | sed 's|^# \[Gate2-canary-quiet[^]]*\] ||' | crontab -"`。
> - **🔴 / abort → 直接看 A100 runner log 新 traceback** (`ssh condense-a100 'tail -40 /home/ubuntu/workspace/p79/logs/B0_*_R*_runner.log'`), **别 isolation 复现** (B-1832→1836 教训: 运行期 bug 只在生产暴露)。

**⑤ WHY (按需读)**: §265 (B-1832 .tmp) / §266 (substrate + 版本误判) / §267 (3-AI /stress 6 修) / §268 (B-1835 os shadow) / **§269 (B-1836 eval-timeout 统一根因 + B-1803 RCA 修正)** / §271 (Gate 1.5 pre-fire /stress, B-1837) + master_bug_catalog B-1832~B-1837。

**⚙️ reboot A100 后服务恢复 checklist (reusable — Gate 3 fire 2026-05-23 踩坑)**: reboot 修 NVML/退化是一举两得, 但 reboot 后**2 个服务必须手动恢复**, 否则 paper-grade fire preflight **Gate 4 fail-closed 拦** (`homepage endpoint not reachable`):
> 1. **`vwa-reddit`** (docker restart policy = `no`, 不自动起): `ssh condense-a100 'docker start vwa-reddit'`
> 2. **homepage :4399** (是 **flask 进程不是 docker 容器**, reboot 杀进程): `ssh condense-a100 'cd /home/ubuntu/workspace/p79/external/visualwebarena/environment_docker/webarena-homepage && setsid nohup /home/ubuntu/workspace/p79/.venv/bin/flask run --host=0.0.0.0 --port=4399 >/tmp/vwa_homepage.log 2>&1 </dev/null &'`
>
> **验证**: `docker ps` 见 `vwa-reddit Up` + `curl -sf localhost:4399` HTTP 200。**自动回来无需手动** (restart policy unless-stopped/always): classifieds / classifieds_db / vwa-shopping / vwa-wikipedia。**WHY**: Gate 3 fire 第一次 launch 因 homepage down 被 Gate 4 拦 (B-1839 无关, 纯 reboot 恢复遗漏)；reddit `restart=no` + flask homepage 是 reboot 恢复的 2 个盲区。fail-closed 拦住 = preflight 工作正常。

---

## §1 ROADMAP — dynamic (`tasks.base` ← edit `_status/tasks/*.md` frontmatter)

> 7 里程碑 = `_status/tasks/*.md` 小 frontmatter (`status`/`priority`/`horizon`/`blocker`/`eta`/`order`) + `tasks.base` 视图。改一字段即更新 (同 cells/issues 模式; 也自动进 `status.base`)。**roadmap canonical = 这里**。

![[tasks.base#🔴 NOW]]
![[tasks.base#📋 NEXT]]
![[tasks.base#🧊 BACKLOG]]

---

## §2 EXPERIMENTS — live (`cells.base`)

![[cells.base#Active 跑中]]
![[cells.base#Pending / Queued / Blocked]]

---

## §3 Router — ⭐ Phase 1 并行核心线 (advisor 2026-05-14) + Pass-1-gated 细项

> ⭐ router = Phase 1 并行核心 contribution (非 paper-2 deferred)。**执行 checklist + blockers + 完成判定 → [[phase1_plan]] §C**。

**路线 (a) rule-based** — 按 task 属性 route。 **路线 (b) learned** — 训练 classifier route (paper-1 §6 = learned-only per v7 amendment)。 **未来扩展** — 按 mode 行为模式 route。
Routing signal infra ✅: `confidence_summary.json` per-condition。Train/test → 5-fold site-stratified CV。设计 → [[paper_planning#§8]]。

> [!todo] Deferred follow-ups — **gated on Pass-1 data landing** (router code 现跑空数据 `n_pooled_total:0`)。详 [[实验笔记]] §255 + [[master_bug_catalog]] B-1805~B-1818。
> 1. **Pass-1 run manifest** (C2/B-1810) — `results/phantom_paper/l1_router/pass1_run_manifest.json` 36 paper-grade run IDs → `discover_runs` strict whitelist。然后 `aggregate_h10_pareto.py --require-full-coverage` (C8/B-1811) 出 paper-grade H10 verdict (fail-closed on incomplete)。
> 2. **τ objective sensitivity** (F3/G2 B-1814) — Pass-1 后跑 τ 对比: (a) accuracy-τ vs (b) fixed-τ vs (c) outcome-matrix Pareto-τ; 仅当 (c) 实质移动 frontier 才采用。**不重构 Stage 1-3** (user 2026-05-21)。
> 3. **inner-CV MI leak** (C3/B-1816) — inner-CV 复用 Stage-2 outer-pool MI selector (mild 2nd-order)。Pass-1 后判断是否实质移 τ; 否则 disclosure 足够。
> 4. **§6 disclosure block** (写进 paper §6/§3.5): self-oracle noise ceiling (G1/B-1809, N=1 oracle) · deployment realism (no-success tasks routed) · MODES cost order (F2/B-1806, 实测 tie-break) · router_strictly_better (F7/B-1815, θ CI>0) · sklearn/numpy version metadata (C6/B-1813)。
> 5. **F9 (P2)** — 删/guard deprecated 8-dim `predict_mode` (`learned_router.py:425-494`) 确认无 caller 后。

---

## §4 Audit follow-ups (DGX-side, independent)

| Pri | Item | Effort | Status |
|---|---|---|---|
| 🟡 R1 | Preflight v2 extension (B0 XOR B1 conflict / archive_subset checks) | 45 min | partially done §134 |
| 🟡 R4 | Stage 2B `--resume` flag for reboot recovery | 10 min | independent |
| 🟡 R6 | `check_evaluator_consistency.py` (Gate 7 evaluator_code_sha == lock-time SHA) | 30 min | OSF lock prep |
| 🟡 **B-1839-fu** | **Phase 1b shopping reset 实现必须含 per-condition docker restart** (B-1839 cls 同款)。per-condition restart 覆盖现状: **reddit ✓** (reset=`docker rm+run` 天然 fresh) / **cls ✓** (B-1839 加 `docker restart classifieds_db classifieds`) / **shopping ✗ 最后缺口** (`_reset_vwa_local_shopping` 现 placeholder `return 78`；Phase 1b 实现时 = HTTP/SQL reset + `docker restart vwa-shopping` + db-ready/http-200 warmup, 否则 shopping 同 cls 退化 + cross-condition latency confound)。 | 1h (随 shopping reset impl) | deferred — Phase 1b launch 前 |
| 🟡 **B-1760** | **DOM mode `screenshot.png` regression — `obs.image=None` for accessibility_tree across 91/91 step records on Fire-3 cls B0 DOM.** Archive 2026-05-15 had it; logic byte-identical archive↔HEAD; runtime instrument needed. Trigger: post cls B0 SoM cell land. Acceptance: re-fire smoke6 / 10-task pilot, verify `screenshot.png` per step + `annotate_screenshots.py` produces `screenshot_annotated.png`. Paper §3 evidence layer NOT blocked (DOM trajectory + schema-v2 fields present); screenshot is audit-layer only. | 2 h | deferred — post cls B0 SoM land |
| 🟢 **R2-P2-10-C** | **Appendix E.3 temporal language fix** — `preregistration.md` Appendix E.3 "witnessed alongside DOI 1 anchor" but artifacts timestamped 2026-05-19, DOI 1 minted 2026-05-18T23:10:06Z. Fix: rephrase "post-DOI-1 forward disclosures, appending to the DOI 1 anchor without modifying its locked estimands". Gemini Mode C F5. Honesty surface, NOT re-witness. | 5 min | deferred — next /stress or paper finalize |
| 🟢 **R2-P2-11-B** | **Schema 4-place sync test enumeration** — `test_schema_4place_sync.py:test_phase2_intervention_fields_present` only enumerates 4 step + 2 episode fields; other 18 Phase 2 fields covered indirectly. Add `test_phase2_attempt_lineage_fields_present` + `test_phase2_footprint_fields_present`. codex Mode B F6. | 15 min | deferred — next /stress or schema v3 |
| 🟢 **C-1** | `aggregate_phantom_lift.py` denominator inconsistency (sr_3 universe_5 vs u_psom) — archived-data analysis, 影响 Appendix D | — | post-Phase-1a |
| 🟢 **C-2** | `aggregate_phantom_lift.py` H3 axis-2 universe — 可能 flip H3(ii) false negative | — | post-Phase-1a |
| 🟢 N1 | Bonferroni / Holm correction paper §3 paragraph | 10 min | paper write phase |
| 🟢 N3 | Phantom variant FP rules | 1 h | post Phase 1a rerun |

---

## §5 Sustainability / Green AI (Section 8 end-stage)

| Item | Status |
|---|---|
| fig regional carbon sensitivity (B1, 45 region) | ✅ done |
| B1 measured energy (cls + red × modes) | ✅ ready |
| Multi-metric Pareto (cost + lat + carbon) | ⏳ Section 8 前置 (~2h) |
| B0 token-based carbon estimator | ❌ optional Tier 3 future |
| Section 8 prose | ❌ paper end-stage |

---

## §6 Codex task queue

![[codex.base#Ready to send (now)]]

![[codex.base#Running / In flight]]

![[codex.base#Blocked / Queued]]

**Pending Python scripts (非 codex)**:
- ⏳ Multi-metric Pareto (cost + lat + carbon) — Section 8 前置 (~2h)
- ⏳ TF-IDF + binary feature extraction — Section 6 Tier 1 router 前置 (~1h)

---

## §7 Open issues

![[issues.base#Active blockers]]

![[issues.base#Backlog]]

---

## §8 Advisor align

详 [[issue_advisor_sync_2026-05-14]] (2026-05-14 sync — Part 1 novelty + Part 2 决策点). Sync 后 decision log → [[paper_planning]] §19; framing register → issue status open → discussed (ADVISOR_SYNC.md retired 2026-05-15).

---

## §9 References + quick links

### Phase 1 canonical
- [[phase1_plan]] — 统领性 audit/execute checklist (§A1 实现层 stress + §A2 设计层 stress + §B clean run + §C router + §D evidence + §E milestones)

### Paper drafts (final prose)
```
docs/checkpoints/paper_drafts/
  section1_intro.md          ✅
  section2_background.md     ✅ + paper.bib
  section3_definition.md     ✅
  section4_findings.md       🟡 待 P-prompt column + Phase 1a hero numbers
  section5_mechanism.md      ✅ v2 (probe-causal-steering trichotomy)
  section6_routing.md        ❌ paper-2 / Tier 1+2 prototype
  section7_generalization.md ❌ 待 WA
  section8_discussion.md     ❌ paper end-stage
```

### Figures
`results/phantom_paper/figures/` — regenerate via `make analysis` / `make figures`:
- §1 hook: fig0a_sr_per_mode_heatmap / fig0b_fp_rate_per_mode / fig0c_drop_one_oracle / fig0g_routing_auroc_heatmap / fig3b_image_token_gap
- §5 mechanism: fig_stage4_method42_v2_{cls,reddit} / fig_axis2_logit_lens_v2 / fig_axis2_layer_profile_v2 / fig_mech_8cell_l17_forest / fig_mech_real_vs_random / fig_layer_axis_emergence_v2_{cls,reddit} / fig_phantom_structure_venn

### Key infra paths
```
configs/exp_v2_*.yaml                              per-site experiment configs (含 12 baseline configs §133 A4)
scripts/queues/queue_phase1_paper_grade.sh         Phase 1a 36-cond orchestrator (§134 harden)
scripts/queues/queue_chain.sh                      sequential chain (§134 C3 crash-detect)
scripts/queues/queue_{baseline,phantom_*}.sh       per-condition launch (§134 FORCE_NEW)
scripts/maintenance/paper_grade_check.py           paper-grade integrity check (cron + on-demand)
scripts/analysis/validate_fire_manifest.py         manifest ghost/over-complete gate (B-1825/1834)
scripts/analysis/preregistration_decision_test.py  H1/H3/TOST canonical (§133 T3 heterogeneity)
scripts/provenance/snapshot_{env,vwa}.*            provenance fingerprint
```

### Provenance artifacts (paper-cite-able)
```
results/provenance/env_dgx_baseline.json           DGX baseline (HF Qwen3-VL-4B SHA ebb281ec...)
docs/checkpoints/pre_run/osf_lock_manifest.md      8-step DOI workflow (DOI-1 minted 10.17605/OSF.IO/9QCWU)
docs/checkpoints/pre_run/preregistration.md        R1-R5 framing + AMENDMENT 01 Protocol Reset
```

---

## §10 独立 bug 研究 paper (workshop-targeted)

> advisor 2026-05-14 收口: bug 部分可**单独再发一篇 paper** 投 workshop — 独立于主 paper, **不替换**主 paper workshop 节点.

**方向**: cross-benchmark bug 聚合研究, 针对现有 web agent benchmark. **参考**: agisdk (https://github.com/agi-inc/agisdk). **素材**: dual-track environment / VWA bug fix ([[实验笔记]] §109 / `master_bug_catalog.md` 37+ bugs). **状态**: 方向 locked; scope + benchmark 选型 + 时间线待 planning. 详 [[workshop_subpaper_plan]]。

---

> 📕 **Historical / superseded** (从本 ledger 移除, 仍在 chronicle/canonical docs): Fire-3/4/5/6 RCA 叙事 → [[实验笔记]] §233/§237/§252/§269 · OSF DOI-1 8-step (已 mint) → [[osf_lock_manifest]] · stats methodology (FE estimand resolved) → [[preregistration]] §2.5 · mechanism §5 暂搁 (2026-05-14) → [[实验笔记]] §138 · fire-event-sequence (Fire-6 已 fire) → [[phase1_plan]] §E。
> 📖 **Doc update workflow** (when X → update which docs) → [[paper_planning#§20]]
