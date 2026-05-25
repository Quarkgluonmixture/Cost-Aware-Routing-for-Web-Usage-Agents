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
>
> **📊 §1/§2/§4-§7 = `![[base#view]]` embed — Obsidian 端渲染成活表; CLI/Claude session `Read` 只见空壳指针**。拿数据: `make status` (列全 base + 视图 + 计数) → `make status V='<base>#<视图名子串, 勿含空格>'` 渲染单视图 (= Obsidian 等价)。改字段: `make status-set N=<note> SET='status=done'`。数据源 = `_status/*.md` frontmatter (cron ~10min sync; 落后实际 fire 时以 ① live 为准)。

**① paper-grade verdict (一条命令; 每 6h cron 也自动跑 + ntfy; ⚠️ 无 A100 shell 的受限 session → 跳过 ssh, 直接读下方 `tail` 本地 cron log = 等价 verdict)**:
```
ssh condense-a100 'cd /home/ubuntu/workspace/p79 && .venv/bin/python3 scripts/maintenance/paper_grade_check.py'
#  → VERDICT: OK completed_ok=N inprog=[Rxxxx mode ep=N img=N errflood=0]   /   ISSUES=...
tail -40 logs/cron/paper_grade_check.log     # ⚠️ ssh 拿不到时 (受限 session) 这是等价 live 源, 滞后 ≤6h; cron 落 DGX 本地 00:30/06:30/12:30/18:30
```

**② fire 死活 (fire 在 A100; `make active` 只扫 DGX → 空是正常)**:
```
ssh condense-a100 'pgrep -af "queue_phase1_paper_grade|queue_chain\.sh|run_experiment\.py" | grep -v "bash -c"'
ssh condense-a100 'cd /home/ubuntu/workspace/p79 && .venv/bin/python3 scripts/analysis/validate_fire_manifest.py'
```
> ⚠️ 滤词用 `grep -v "bash -c"` **不是** `grep -v bash`: 链编排器本体 = `bash scripts/queues/queue_chain.sh …` (它本身就是个 bash 进程), `grep -v bash` 会把它一起滤掉 → 误判"编排器死了" → 诱发**有害的手动 re-fire** (同-site 双起 baseline → 违反 hard rule → cross-contam)。读数: `run_experiment.py` 活 = 当前 condition 在跑; `queue_chain.sh` 活 = 下一 condition 会自动续链; **后者空但前者活 = 当前 condition 跑完会静默 stall** (链不会自动续, 需手动 re-arm)。

**③ 当前阶段 (慢变语境 — live 进度跑①拿, 勿读快照数字)**: **Gate 3 = cls 18-cond chain 跑中** (B0/B1/B2 × 6 mode 顺序: dom / som / vision / P-text / P-SoM / P-prompt), orchestrator `queue_chain.sh` 自动续链, 每 cond **B-1839 per-condition docker restart** = fresh substrate。监控全自动: A100 `fire6_monitor` (re-armed) + DGX `paper_grade_check` 6h cron + sync 15min。**cls 18 完 → red 18** (sequential, cross-site contention 规避)。**ETA ~12-21 天** (B0 cls ~8.3h/mode 实测; 编排器原估偏低 2×)。Amendment 02 (`e338cb4` + tag + OSF, H1-strict 6-mode gate **不动**) + B-1839 substrate (git tag `b1839-substrate-prefire`) 均 **pre-fire witnessed, OSF 不动** (衬底/binding 非 estimand)。
>
> **🔄 2026-05-25**: B-1860 Qwen 0-1000 coord contract **APPLIED** (merge `d977006` + amendment 05 witness git tag + OSF kv9sf) + **B-1861** watchdog ntfy fail-safe。cls chain **从 vision restart** (手动 `queue_chain` 16 cells: B0 vision→ptext→psom→pprompt + B1×6 + B2×6; B0 dom/som 保留旧 coord-invariant 代码; ⚠️ NOT `queue_phase1 launch` = FORCE_NEW 重跑全部)。详情 → `make status V='audit#Done'` + 笔记 §285-288。**live run_id / 进度跑 ①**。

**④ PENDING — forward-only (live 进度靠 ①, 不在此冻快照)**:
> - **每 condition 落地跑 `/diag`** (per-condition 命名, 勿用 run_id): 3-tier (Tier-1 0-token) → `docs/analysis/vwa_<site>/<model>_<mode>_<site>_diag_digest.md`, 拆失败结构喂 paper-grade 错因表。**已跑**: R31194 (B0 dom, §277) + R9725 (B0 som, §283 — SR 30.4%/scaffold=0/P14 presence-only/≥3 benchmark-FP[task96/42 复查完成=**B-21** string_match 货币 tokenize, P28 覆盖, 非 scoring bug])。**规则库已 self-evolve** (§284, ruleset 2-dom→**3-domsom**: P24-P30 落码 + P14 修正 succ-fire 15→0 + 全量重扫 dom+som)。**vision (R3671) /diag 完成** → 挖出 **B-1860 coordinate 根因** (Qwen 0-1000 vs P79 [0,1]/viewport 栈不匹配; 笔记 §285)。**✅ B-1860 contract APPLIED 2026-05-25** (merge `d977006` + amendment 05 witness git+OSF + B-1861 watchdog fix; §288)。R3671 archived; cls chain **从 vision restart** (手动 queue_chain 16 cells, R24792 vision **完成 + /diag done §290**; B0 dom/som 保留旧代码 coord-invariant)。**vision diag 全闭环 done (§290)**: ① B-1860 验证 ✅ (parse 13.6%→0.027%=1/3662; 0 scaffold; 0 coord-residual/59; fail-loud no-op `action_executed=None`); ② 失败全 agent-limit (SR 24.1%, vision 视觉天花板 = paper §3-§4 evidence); ③ **benchmark-FP forensic** (artifacts 截图, 不连 fire-中 docker): task40 翻案 agent-limit (ref LG 实际更 recent, agent 没用 vision 判材质; log-only sub-agent 误判→截图纠正) + task132 倾向 agent-limit → **净 FP≈0**; ④ **落码 self-evolve**: P31 budget耗尽(trajectory_incomplete; **三 mode 头号失败维度** vis73/dom58/som49) + P32 text-in-price + P27 扩展(措辞+ref-carveout); bump **ruleset 统一 `4-domsomvis-b1860coord`**, 全量重扫 dom/som/vision success-fire 全 0, SKILL.md(30条/相位)同步 → digest `vwa_classifieds/B0_vision_classifieds_diag_digest.md`。**仍 open**: P14 v3 scroll_changed 豁免复核 · failed-hit causal verify (P19/P5/P14) · cross-mode 仍禁 (已统一版本但只 3/6 mode)。**下一个 diag**: **B0 phantom_text (R2647 = chain 续到的下一 mode, 完成后跑)**。
> - **cls 18 完 → red 前**: scratch redirect (results→`/mnt/scratch` symlink; 全 36≈25G; 数据落 DGX verify 后做) + prune + launch red 18。详见下 disk 架构 bullet。
> - **B-1837** (eval 5-retry vs agent-step 0-retry → differential rescue confound) = measure-then-decide: Pass-1 量化 per-baseline eval-rescue rate, 再定 disclose (§3.5/§8) vs symmetric retry。NOT code change now (master_bug_catalog B-1837)。
> - **post-fire / Phase-1b code remediation** (B-1842~1847 parse-error rescue · B-1839-fu shopping reset · B-1848 Playwright wedge · B-1760 screenshot): 全迁 §4 → `make status V='audit#Deferred'`。
> - **⚠️ disk 架构 "A100 算力 / DGX 数据"** (慢变 reference): A100 `/` ~93% (docker 镜像 390G = 地板, VWA 三站 essential), `/mnt/scratch` 366G 闲置。正解 = results→scratch symlink (366G≈8-10 fire) + sync→DGX 永久 (mirror `--delete-after`) + verify-on-DGX 后 prune → A100 占用由 active-run 决定非累积。**Gate go/no-go 第三因子** (除 wallclock+abort)。
> - **gallery on-demand** (B-1828): `make gallery RUN=<run>` (annotate overlay + HTML) / `make gallery-all` → `http.server 8765 --directory results`。不自动刷新 (paper-grade 保干净); 原图随时 DGX 可生成, HTML 手动 `make gallery`; 近实时设 `P79_WATCHDOG_GALLERY=1` (run 期 overhead, 不建议)。
> - **🔴 / abort → 直接看 A100 runner log 新 traceback** (`ssh condense-a100 'tail -40 /home/ubuntu/workspace/p79/logs/B0_*_R*_runner.log'`), **别 isolation 复现** (B-1832→1836 教训: 运行期 bug 只在生产暴露)。

**⑤ WHY (按需读)**: 全链 chronicle → 笔记 §265-290 + `master_bug_catalog` B-1832~B-1861。关键锚: §269 (B-1836 eval-timeout) · §280 (B-1848 wedge) · §281 (AMENDMENT_04) · §285-288 (B-1860 coord 全链) · §289 (status 动态层) · §290 (vision /diag: B-1860 验证 + forensic + 落码 P31/P32/P27) · §291 (cross-mode 失败 taxonomy 框架 `cross_mode_failure_taxonomy.py` O(mode) + routing=缩略图识别梯度 provisional)。

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

**Cross-mode 失败 taxonomy** (`scripts/analysis/cross_mode_failure_taxonomy.py`, §291) — routing 证据/feature 来源: B0 cls 3-mode 实证 routing 价值 = **列表缩略图识别 (THUMBNAIL) 梯度** dom>som>vision (非纯导航 SEARCH-NAV 小 / 非详情大图 IMG 极小)。**6-mode 数据齐复用** (`--run` 加列, O(mode) 不爆炸) → 定 routable task 类型 → router feature。Open: UNCLEAR-NAV 拆 (需 listing-level obs, ROI 低) + cross-site (red/shop)。⚠️ provisional 3/6 mode, 禁 cross-mode 定量直至 freeze。

> [!todo] Deferred follow-ups — **gated on Pass-1 data landing** (router code 现跑空数据 `n_pooled_total:0`)。详 [[实验笔记]] §255 + [[master_bug_catalog]] B-1805~B-1818。
> 1. **Pass-1 run manifest** (C2/B-1810) — `results/phantom_paper/l1_router/pass1_run_manifest.json` 36 paper-grade run IDs → `discover_runs` strict whitelist。然后 `aggregate_h10_pareto.py --require-full-coverage` (C8/B-1811) 出 paper-grade H10 verdict (fail-closed on incomplete)。
> 2. **τ objective sensitivity** (F3/G2 B-1814) — Pass-1 后跑 τ 对比: (a) accuracy-τ vs (b) fixed-τ vs (c) outcome-matrix Pareto-τ; 仅当 (c) 实质移动 frontier 才采用。**不重构 Stage 1-3** (user 2026-05-21)。
> 3. **inner-CV MI leak** (C3/B-1816) — inner-CV 复用 Stage-2 outer-pool MI selector (mild 2nd-order)。Pass-1 后判断是否实质移 τ; 否则 disclosure 足够。
> 4. **§6 disclosure block** (写进 paper §6/§3.5): self-oracle noise ceiling (G1/B-1809, N=1 oracle) · deployment realism (no-success tasks routed) · MODES cost order (F2/B-1806, 实测 tie-break) · router_strictly_better (F7/B-1815, θ CI>0) · sklearn/numpy version metadata (C6/B-1813)。
> 5. **F9 (P2)** — 删/guard deprecated 8-dim `predict_mode` (`learned_router.py:425-494`) 确认无 caller 后。

---

## §4 Audit follow-ups → audit.base (`_status/audit/*.md`)

> 22 条 audit follow-ups 迁入 **audit.base** (2026-05-25 commit). 改 `_status/audit/<note>.md`
> frontmatter (`status` / `phase` / `priority`) 即更新视图; CLI `make status V='audit#All'` /
> `make status-set N=audit_<x> SET='status=done'`. §3 router follow-ups 仍 prose (见 §3);
> §5 sustainability 已并入 (phase==section8).

![[audit.base#🔴 Gate-blocking / now]]
![[audit.base#Deferred (post-fire)]]
![[audit.base#Paper-finalize]]
![[audit.base#All (by priority)]]

---

## §5 Sustainability / Green AI → audit.base (phase: section8)

![[audit.base#Section 8 (sustainability)]]

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
