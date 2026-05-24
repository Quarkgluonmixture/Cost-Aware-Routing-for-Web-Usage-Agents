---
type: action-ledger
status: rolling
updated: 2026-05-23
---

# Next Steps — Forward Action Ledger

> **Future-only**. **Roadmap = §1 `tasks.base`** (dynamic — edit `_status/tasks/*.md` frontmatter). Live state 不在这里:
> - cron health / 错误扫描 / ntfy 历史 → `make ntfy` + `logs/cron/*.log` (PLAYBOOK retired 2026-05-23, §279)
> - Real-time active runs / GPU → `make active` CLI (DGX only; fire 在 A100)
> - Paper-grade fire verdict → `paper_grade_check.py` (一条命令, §0) + 每 6h cron
> - Cell snapshot (active / pending / done) → `cells.base`
> - Paper section progress → `status.base`
> - 过去 chronicle → [[实验笔记]] (latest §278, 2026-05-23)
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
ssh condense-a100 'pgrep -af "queue_phase1_paper_grade|queue_chain\.sh|run_experiment\.py" | grep -v "bash -c"'
ssh condense-a100 'cd /home/ubuntu/workspace/p79 && .venv/bin/python3 scripts/analysis/validate_fire_manifest.py'
```
> ⚠️ 滤词用 `grep -v "bash -c"` **不是** `grep -v bash`: 链编排器本体 = `bash scripts/queues/queue_chain.sh …` (它本身就是个 bash 进程), `grep -v bash` 会把它一起滤掉 → 误判"编排器死了" → 诱发**有害的手动 re-fire** (同-site 双起 baseline → 违反 hard rule → cross-contam)。读数: `run_experiment.py` 活 = 当前 condition 在跑; `queue_chain.sh` 活 = 下一 condition 会自动续链; **后者空但前者活 = 当前 condition 跑完会静默 stall** (链不会自动续, 需手动 re-arm)。

**③ 当前阶段 (慢变语境 — live 进度跑①拿, 勿读快照数字)**: **Gate 3 = cls 18-cond chain 跑中** (B0/B1/B2 × 6 mode 顺序: dom / som / vision / P-text / P-SoM / P-prompt), orchestrator `queue_chain.sh` 自动续链, 每 cond **B-1839 per-condition docker restart** = fresh substrate。监控全自动: A100 `fire6_monitor` (re-armed) + DGX `paper_grade_check` 6h cron + sync 15min。**cls 18 完 → red 18** (sequential, cross-site contention 规避)。**ETA ~12-21 天** (B0 cls ~8.3h/mode 实测; 编排器原估偏低 2×)。Amendment 02 (`e338cb4` + tag + OSF, H1-strict 6-mode gate **不动**) + B-1839 substrate (git tag `b1839-substrate-prefire`) 均 **pre-fire witnessed, OSF 不动** (衬底/binding 非 estimand)。WHY 全链 chronicle → §274/§275 (launch 前序) · §276 (B-1840/1841 binding bug) · §277 (R31194 /diag) · §278 (parse-error /stress disclosure)。

**④ PENDING — forward-only (live 进度靠 ①, 不在此冻快照)**:
> - **每 condition 落地跑 `/diag`** (per-condition 命名, 勿用 run_id): 3-tier (Tier-1 0-token) → `docs/analysis/vwa_<site>/<model>_<mode>_<site>_diag_digest.md`, 拆失败结构喂 paper-grade 错因表。R31194 (B0 dom) 已跑 (§277)。
> - **cls 18 完 → red 前**: scratch redirect (results→`/mnt/scratch` symlink; 全 36≈25G; 数据落 DGX verify 后做) + prune + launch red 18。详见下 disk 架构 bullet。
> - **B-1837** (eval 5-retry vs agent-step 0-retry → differential rescue confound) = measure-then-decide: Pass-1 量化 per-baseline eval-rescue rate, 再定 disclose (§3.5/§8) vs symmetric retry。NOT code change now (master_bug_catalog B-1837)。
> - **B-1842~1847** (parse-error rescue accounting, 本 session 3-AI /stress): disclosure 已 land (`section4_limitations_disclosure §4.X.19`); **code remediation defer post-fire** (fire 跑中 `p79/` code immutable — 续链 spawn import)。见 §4。
> - **Phase 1b shopping reset 必须含 per-condition docker restart** (B-1839-fu, §4): shopping `_reset_vwa_local_shopping` 现 placeholder `return 78`, Phase 1b launch 前补 (否则 shopping 同 cls 退化 + cross-condition latency confound)。
> - **⚠️ disk 架构 "A100 算力 / DGX 数据"** (慢变 reference): A100 `/` ~93% (docker 镜像 390G = 地板, VWA 三站 essential), `/mnt/scratch` 366G 闲置。正解 = results→scratch symlink (366G≈8-10 fire) + sync→DGX 永久 (mirror `--delete-after`) + verify-on-DGX 后 prune → A100 占用由 active-run 决定非累积。**Gate go/no-go 第三因子** (除 wallclock+abort)。
> - **gallery on-demand** (B-1828): `make gallery RUN=<run>` (annotate overlay + HTML) / `make gallery-all` → `http.server 8765 --directory results`。不自动刷新 (paper-grade 保干净); 原图随时 DGX 可生成, HTML 手动 `make gallery`; 近实时设 `P79_WATCHDOG_GALLERY=1` (run 期 overhead, 不建议)。
> - **🔴 / abort → 直接看 A100 runner log 新 traceback** (`ssh condense-a100 'tail -40 /home/ubuntu/workspace/p79/logs/B0_*_R*_runner.log'`), **别 isolation 复现** (B-1832→1836 教训: 运行期 bug 只在生产暴露)。

**⑤ WHY (按需读)**: §265 (B-1832 .tmp) / §266 (substrate + 版本误判) / §267 (3-AI /stress 6 修) / §268 (B-1835 os shadow) / **§269 (B-1836 eval-timeout 统一根因 + B-1803 RCA 修正)** / §271 (Gate 1.5 pre-fire /stress, B-1837) / §274 (Amendment 02 gate ladder) / §275 (Gate 3 launch 全链) / §276 (B-1840/1841 binding 修) / §277 (R31194 /diag) / §278 (parse-error rescue 3-AI /stress disclosure) / **§280 (B-1848 som Playwright driver-wedge hang → py-spy RCA → Option-A 全新重跑; R2815 archived, som re-fired R9725)** / **§281 (analysis-layer 3-AI /stress → AMENDMENT_04 implementation-alignment; B-1849~1857; gate 算术 faithful, drift 在 producer/figure/prose; tag + OSF kv9sf witness 链闭合)** + master_bug_catalog B-1832~B-1857。

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
| 🟡 **B-1842~1847** | **parse-error rescue accounting symmetry (3-AI /stress 2026-05-23, disclosure 已 land `section4_limitations_disclosure.md §4.X.19`)** — forward code remediations: canonical-latency 加 `−parse_error_injected_wait_ms` 对称扣除 (B-1842) / rename `parse_error_rate`→`injected_wait_rate` + `parse_valid_before_rescue` flag 区分 rescue-wait vs model-wait (B-1843) / `no_progress_rate` per-cell covariate (B-1844) / `termination_reason` episode 字段 (B-1846) / B0 `tool_call_emit_rate≥0.95` condition gate in `paper_grade_check.py` (B-1847). 实测 sink rate B0≈0/B1 0/B2 0.7% → 量级 negligible, disclosure 已充分。 | 2-3h | **deferred post-fire** (fire 跑中 `p79/` code immutable — 续链 spawn import) |
| 🟡 **B-1848** | **Playwright driver-wedge hang — silent infinite block 绕过 operation timeout + runner M1 + watchdog (Gate3 cls B0 som task190 production incident 2026-05-23, §280)** — runner MainThread 阻塞 Playwright sync 事件循环 `select` 等 wedged node driver (py-spy 定位); Playwright op-timeout 是 driver 侧强制 (driver wedge 同死) + python 客户端无 IPC wall-clock deadline + M1 需 exception (silent block 不抛) → 永久 hang, watchdog 仅 alert-only。Fix: Playwright driver IPC / sync page-op 加**客户端 wall-clock deadline** (SIGALRM 或 watchdog-thread → page/context.close 超时) → wedge raise 进 runner M1 而非 silent block; 一并 reap B0 proxy CLOSE-WAIT pool 连接 (次要)。 | 2-3h | **deferred post-fire** (fire 跑中 `p79/` immutable) |
| 🟡 **Repro-replicate** | **Run-to-run SR 方差: 机制确定 + 量级 OPEN (深挖闭环, 笔记 §282 / B-1858)**。**机制 (实证)**: VWA element-ID 非确定 (`processors.py:532` CDP/树序号) → 同页面 obs byte-diff (**仅 `[id=N]` 变**, content-diff 证结构稳) → temp=0 模型对 ID token 敏感 → action churn → 轨迹分叉; 截图=视觉锚 (SOM step-0 action 90%>DOM 73%)。**量级 OPEN — 无干净 replicate** (两 pair 都 confound): DOM (R31194 vs R9755)=regime (fresh vs pre-B-1839-stale), 0.4pp 对称 = regime≈0 proxy **非纯 replicate**; SOM (R9725 vs R2815)=R2815 是 **B-1848 wedge run**, +7pp **非对称** (19/6 p=0.015)。判别 (telemetry + **full 25-flip 轨迹扫**: 全 clean agent failure, t151 minSoM=2=agent 点图非 substrate, errors=`no_progress` symmetric) **排除 A/A' → B=真 agent run-to-run 方差 confirmed across all 19 + directional** (n=1 可能 fluke / ① model drift)。**撤回**早前 "0.4pp clean floor / 对称 / gate robust"。**gate 风险 OPEN**: som run-to-run 可能几 pp → 接近 phantom 1.7-3.3pp → 可能威胁 gate; dom<1pp 不背书 som。**Forward**: (a) **🔴 clean 3rd som replicate (两健康 B0 som) = gate 前置** (估真 run-to-run std + 是否 directional/复现); (b) §4 disclosure 用 replicate 数非 R2815; (c) post-fire replay 切 ①/② + positional-ID fix (B-1858 根除 ②); (d) phantom 落地测 step-0 分叉。 | (a) **gate-blocking** post-fire/idle · (b)-(d) post-fire | **gate 前置 (a)** / post-fire |
| 🟢 **R2-P2-10-C** | **Appendix E.3 temporal language fix** — `preregistration.md` Appendix E.3 "witnessed alongside DOI 1 anchor" but artifacts timestamped 2026-05-19, DOI 1 minted 2026-05-18T23:10:06Z. Fix: rephrase "post-DOI-1 forward disclosures, appending to the DOI 1 anchor without modifying its locked estimands". Gemini Mode C F5. Honesty surface, NOT re-witness. | 5 min | deferred — next /stress or paper finalize |
| 🟢 **A03-fu** | **prereg L103-111 honesty-surface sync** — degenerate-SE-floor 段仍是 pre-B-1003 "SE = 0 exactly" 措辞 + implementation 指针指 `aggregate_phase1_prereg_gate.py:185-187`, 与 L98/L718 锁定的 0.68pp Agresti-Coull threshold 矛盾。Fix: "SE = 0 exactly" → "SE < 0.68pp threshold" + repoint 到 canonical `aggregate_phase1_full_prereg_decision` (legacy gate 已 transparency-only)。纯措辞 sync, **0.68 estimand 值不变** (已 recorded `AMENDMENT_03 §3` + tag `prereg-amendment-03-implementation-alignment-20260524`)。与 R2-P2-10-C 同批。 | 5 min | deferred — next /stress or paper finalize |
| 🟢 **AMENDMENT_04-fu** | **analysis-alignment 收尾 (3 项, 均 deferred, 非阻塞)** — AMENDMENT_04 (B-1849~1857) tag + OSF kv9sf 已 witness; 剩: **(a) prereg prose sync** — `AMENDMENT_04 §3` supersession table 记 prereg §4 latency row → scaffold-adjusted (B-1854) + §H10 LR `class_weight balanced→None` (B-995/P1-7), 与 A03-fu + R2-P2-10-C **同批 fold at paper-finalize** (DOI anchor 不改, estimand 值不变); **(b) stale artifacts regenerate** (P1-5) — fire 后 `make analysis` 重生 `h10_pareto_verdict.json` (旧 `h0_rejected` schema) + `cross_site_*` (缺 baseline/billed/basis 列), 可加 CI reject-old-schema; **(c) /stress retro** — data land 后对照实际 θ_FE/H10 verdict 回看 B-1849~1857 哪些真拦 paper-grade 错 (P1-4 latency-misattribution 已当场证). | (a) paper-finalize · (b) 随 fire · (c) data-land 后 | deferred — 非阻塞 |
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
- §1 hook (脚本已存在): fig0a_sr_per_mode_heatmap / fig0b_extra_confidence_calibration / fig0c_drop_one_oracle / fig0g_routing_auroc_heatmap / fig3b_image_token_gap / fig_phantom_structure_venn / fig_meta_forest / fig_forest_drop_one
- §5 mechanism (§5 暂搁 — 脚本未落地/已 frozen): fig_stage4_method42_v2_{cls,reddit} / fig_axis2_logit_lens_v2 / fig_axis2_layer_profile_v2 / fig_mech_8cell_l17_forest / fig_mech_real_vs_random / fig_layer_axis_emergence_v2_{cls,reddit}
- §5 mechanism (脚本存在但 §5 暂搁): fig_mechanism_pilot

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
