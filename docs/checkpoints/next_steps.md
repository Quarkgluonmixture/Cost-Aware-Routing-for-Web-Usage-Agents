---
type: action-ledger
status: rolling
updated: 2026-05-15
---

# Next Steps — Forward Action Ledger

> **Future-only**. Live state 不在这里:
> - Today / 瓶颈 / cron health → [[PLAYBOOK#§1]] + [[PLAYBOOK#§2]] (🤖 GLM @daily)
> - Real-time active runs / GPU → `make active` CLI
> - Cell snapshot (active 跑中 / pending / done) → `cells.base`
> - Paper section progress → `status.base`
> - 过去 chronicle → [[实验笔记]] (latest §140)
> - Strategy / theory → [[paper_planning]]
> - **Phase 1 执行计划 + audit checklist** → [[phase1_plan]] ⭐ canonical
> - Advisor sync prep → [[issue_advisor_sync_2026-05-14]] + [[followup]]
> - OSF DOI lock workflow → [[osf_lock_manifest]]
> - Compute infrastructure → [[COMPUTE_INFRASTRUCTURE]]
>
> 🔧 新数据 → `make analysis` (~5-10min). Cron 每 10min 自动 sync cell frontmatter.

---

## §0 Direction

**Paper hook**: → [[paper_planning#§1]] (canonical, phantom routing space 3 arms / 4-fold drop-in)

> [!todo] Top forward actions (priority order, 2026-05-14 收口 — advisor discussion done; mechanism 暂搁; router 升 Phase 1 核心)
> 1. **审查 bug + pipeline (两层 stress)** ⭐⭐⭐ — §A1 实现层 + §A2 设计层 audit, clean run 前置. 详 [[phase1_plan]] §A1+§A2.
> 2. **cls + red baseline 干净 clean run** ⭐⭐⭐ — 3 模型 (B0/B1/B2=Gemma3-VL) = **36 conditions / 6 cells**. Gemma3-VL pipeline 已 land ([[实验笔记]] §140). 详 [[phase1_plan]] §B.
> 3. **同步做 router (双路线)** ⭐⭐⭐ — (a) rule-based 按 task 属性区分; (b) learned classifier routing. 从 paper-2 deferred 升为 Phase 1 核心. 详 [[phase1_plan]] §C + §5.
> 4. **独立 bug 研究 paper** ⭐ — cross-benchmark bug 聚合研究 (e.g. agisdk), 可单独投 workshop. 详 §11.
>
> **重大变化 (2026-05-14 收口, 见 [[实验笔记]] §138)**: mechanism (§5/§0a) 整个暂搁; Gemma3-VL 正式入 baseline; venue cascade = EMNLP (5/25) → workshop → NeurIPS. 学长: 论文写作交 advisor, 学生 focus = experiment execution.

---

## §0a Mechanism (§5) — ⏸️ 暂搁 (advisor discussion 2026-05-14)

> ⏸️ **2026-05-14 收口**: 学长 "mechanism 部分先不要管了". 整个 §5 (activation patching / layer probe / logit lens / SAE) 暂搁; 下面 forward items 全部冻结, **不进当前 paper scope**. §133/§136 已 land 的 mechanism v2 工作存档保留 (见 [[实验笔记]] §138.3). 以下内容保留作未来 paper-2 / 解冻参考.

**DONE (2026-05-11 → 05-14, 见 [[实验笔记]] §125-§133)**: Stage 4 全部 4 方法 land —
Method 4.2 PCA cosine gap (AUROC 1.000) / activation patching Exp 5 cellhprompt (L11-L17 displacement 0.20-0.30) /
Exp 3 logit lens (per-task KL, axis-2 ratio 1.1-3.95×) / Method 4.4 mean-diff steering (v2 train/eval split,
held-out 0.12 vs in-sample 0.29 → A5 counter-claim succeeds). §5 prose v2 reframe = probe–causal–steering trichotomy.
v2 NPZ re-extraction done. Pipeline audit + 5 paper-grade fix commits.

**Forward**:

| Pri | Item | Effort | Gating |
|---|---|---|---|
| ⭐⭐ | **Cross-family P2/P3 fire** — Phi-3.5-Vision + Qwen2-VL-7B extraction (scripts 已修 Bug 2/5, paper-grade safe). H1' capacity-limit test: 4B shortcut 是容量限制还是训练分布先验 | ~1-2h GPU/model | advisor mechanistic scope 决策 (B1-only → 不跑; cross-arch → 跑) |
| ⭐ | **SAE feature steering** — 把 "steering 不 transfer" 翻成 positive intervention. 当前倾向: 不做, 留 paper-2 (三分结论已自洽, SAE 引入新举证负担) | weeks | advisor 决策 (SAE 进 paper-1?) |
| 🟡 | **format_variation `fmt_som_standard` v1-ish 修复** — codex C1 P0, data-altering, 需 re-extract. 当前 documented NOT patched | 30-60min + extraction | 决定是否影响 H1/format-variation baseline 可比性 |
| 🟢 | `run_stage1_pilot.py` NPZ schema gap (older pipeline) | low | 非阻塞 |

---

## §1 Phase 1a paper-grade rerun → [[phase1_plan]]

**Scope** (2026-05-14 advisor 收口, [[实验笔记]] §138): **36 conditions / 6 cells** = (cls + red) × {B0, B1, B2 = Gemma3-VL} × 6 modes。旧 24/4 已废 (B2 入 baseline)。
**Phase 1b** (post-workshop deferred) = + shop × 3 × 6 = 18 cond, feeds R3 → R1 / Option D framing decision。

**Terminology hard rule**: "condition" = 1 (site, model, mode) launch unit; "cell" = 1 (site, model) stratification unit. **不要混用**。

**Canonical 执行 checklist + critical path + milestones + pre-launch gates + post-completion**: → [[phase1_plan]] §0 + §A + §B + §E

**当前 launch 主 blockers (snapshot)**:
- ⏳ #11 A100 VM VWA docker bring-up (gates launch, [[phase1_plan]] §B0)
- ⏳ #10 analysis 层 3-model 改造 (gates §D 不 gate launch)
- ⏳ `preregistration.md` status `draft → locked` (gates §B1, 待 advisor)
- ⏳ B-86 GLM parse fallback scaffold (B0-only paper-grade disclosure or fix, 待学长)

**Post-data-lands doc updates** (笔记 §133.6): §4 P-prompt column / §1 hero numbers / §8 limitations "two sites + Phase 1b deferred" + model name fix "Qwen3-VL-235B" / paper.bib DL+Higgins refs / §6§7 section files.

---

## §2 OSF DOI 8-step lock workflow (post advisor sync)

**Trigger**: Advisor 确认 K_h1 / K_h3 / TOST δ + DL meta 方法 (REML+HK vs DL).

**8 steps** (详 [[osf_lock_manifest]]):
1. Save advisor confirmation → `docs/reference/advisor_email_<date>.pdf` (或 sync notes)
2. Update `preregistration.md` (replace `TBD` with confirmed numbers; propagate TOST→superiority 已在 §133 T1 完成)
3. `python3 scripts/provenance/snapshot_env.py` on DGX + Myriad
4. `bash scripts/provenance/snapshot_vwa.sh` on each VWA host
5. `cp -r paper_drafts paper_drafts_locked` + commit
6. `git tag -a preregistration-locked` + push
7. Mint OSF DOI at https://osf.io/registries/ (link GitHub tag URL)
8. Backfill `osf_lock_manifest.md` with SHAs + DOI + timestamp

**Artifacts ready**: ✅ `env_dgx_baseline.json` / `vwa_dgx_via_quark.json` / `osf_lock_manifest.md` / provenance scripts / `preregistration_decision_test.py` (smoke 4 scenarios route correct, §133.5) / HF revision pin `ebb281ec...` in `exp_v2_base.yaml` (§134 C8).

---

## §3 Statistical methodology — advisor sync questions (deferred)

From §133 codex Round C + §134 /stress v6:

| Item | Question | 倾向 |
|---|---|---|
| C-M1 / F1 | DL meta τ² biased at k<10 (Veroniki 2016) — preregistration `decision 3A` 2026-05-14 retired DL in favor of FE estimand (no τ² needed). k=4 → k=6 per B2 addition 2026-05-14 eases but does not eliminate k<10 fragility under any RE estimator | RESOLVED via FE estimand (no advisor decision needed for DL replacement); k=6 power numbers ⏳ pending advisor lock |
| C-M2 / F2 | Wald 1.96 CI anti-conservative at k<10 (IntHout 2014) — FE estimand side-steps (FE Wald is sound at any k under CLT on per-cell θ_i). No Hartung-Knapp needed since no RE estimator in primary gate | RESOLVED via FE estimand 2026-05-14 |
| C-M2 | `power_analysis.md` rewrite for 4-cell Phase 1a scope (现仍 12/16 + N≥10 mismatch) | post-sync |
| C-1 / F4 | `aggregate_phantom_lift.py` denominator inconsistency (sr_3 universe_5 vs u_psom) — archived-data analysis, 影响 Appendix D | post-Phase-1a |
| C-2 | `aggregate_phantom_lift.py` H3 axis-2 universe — 可能 flip H3(ii) false negative | post-Phase-1a |

---

## §4 Audit follow-ups (DGX-side, independent)

| Pri | Item | Effort | Status |
|---|---|---|---|
| 🟡 R1 | Preflight v2 extension (B0 XOR B1 conflict / archive_subset checks) | 45 min | partially done §134 |
| 🟡 R4 | Stage 2B `--resume` flag for reboot recovery | 10 min | independent |
| 🟡 R6 | `check_evaluator_consistency.py` (Gate 7 evaluator_code_sha == lock-time SHA) | 30 min | OSF lock prep |
| 🟢 N1 | Bonferroni / Holm correction paper §3 paragraph | 10 min | paper write phase |
| 🟢 N3 | Phantom variant FP rules | 1 h | post Phase 1a rerun |

---

## §5 Router — ⭐ Phase 1 并行核心线 (advisor 2026-05-14 收口)

> ⭐ **2026-05-14 收口**: router 从 "Section 6 / paper-2 deferred" 升为 **Phase 1 并行核心 contribution**. 两条基础路线并行做, 与 cls+red baseline clean run 同步.
>
> **执行 checklist + blockers + 完成判定** → [[phase1_plan]] §C

**路线 (a) rule-based router** — 根据 task 属性 / 任务区分来 route。
**路线 (b) learned router** — 训练一个 classifier 做 routing。
**未来扩展** — 根据不同 mode 的行为模式 route。

Routing signal infra (✅ ready): `confidence_summary.json` per-condition。
Train/test split protocol → 倾向 5-fold site-stratified CV (vs LOSO)。设计细节 → [[paper_planning#§8]]。

---

## §6 Sustainability / Green AI (Section 8 end-stage)

| Item | Status |
|---|---|
| fig regional carbon sensitivity (B1, 45 region) | ✅ done |
| B1 measured energy (cls + red × modes) | ✅ ready |
| Multi-metric Pareto (cost + lat + carbon) | ⏳ Section 8 前置 (~2h) |
| B0 token-based carbon estimator | ❌ optional Tier 3 future |
| Section 8 prose | ❌ paper end-stage |

---

## §7 Codex task queue

![[codex.base#Ready to send (now)]]

![[codex.base#Running / In flight]]

![[codex.base#Blocked / Queued]]

**Pending Python scripts (非 codex)**:
- ⏳ Multi-metric Pareto (cost + lat + carbon) — Section 8 前置 (~2h)
- ⏳ TF-IDF + binary feature extraction — Section 6 Tier 1 router 前置 (~1h)

---

## §8 Open issues

![[issues.base#Active blockers]]

![[issues.base#Backlog]]

---

## §9 Advisor align

详 [[issue_advisor_sync_2026-05-14]] + [[followup]] (2026-05-14 sync — Part 1 novelty + Part 2 决策点). Sync 后:
decision log 写 [[paper_planning]] + framing decisions register → [[issue_advisor_sync_2026-05-14]] status open → discussed (ADVISOR_SYNC.md retired 2026-05-15).

---

## §10 References + quick links

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
scripts/queues/queue_phase1_paper_grade.sh         Phase 1a 36-cond orchestrator (§134 harden; expanded 24 → 36 per B2 addition 2026-05-14)
scripts/queues/queue_chain.sh                      sequential chain (§134 C3 crash-detect)
scripts/queues/queue_{baseline,phantom_*}.sh       per-condition launch (§134 FORCE_NEW)
scripts/analysis/preregistration_decision_test.py  H1/H3/TOST canonical (§133 T3 heterogeneity branch)
scripts/mechanistic/run_stage4_*.py                Stage 4 mechanism pipeline (v2 post-fix)
scripts/mechanistic/run_stage4_h1_{phi35,qwen2vl}.py  cross-family extraction (Bug 2/5 fixed §133)
scripts/provenance/snapshot_{env,vwa}.*            provenance fingerprint
p79/mechanistic/activation_patching.py             patching infra (layer-index convention documented)
```

### Provenance artifacts (paper-cite-able)
```
results/provenance/env_dgx_baseline.json           DGX baseline (HF Qwen3-VL-4B SHA ebb281ec...)
results/provenance/vwa_dgx_via_quark.json          VWA stack fingerprint
docs/checkpoints/pre_run/osf_lock_manifest.md      8-step DOI workflow
docs/checkpoints/pre_run/preregistration.md        R1-R5 framing rule + K-of-N transparency-only
```

---

## §11 独立 bug 研究 paper (workshop-targeted)

> advisor 2026-05-14 收口: bug 部分可**单独再发一篇 paper** 投 workshop — 独立于主 paper, **不替换**主 paper 的 workshop 节点.

**方向**: cross-benchmark bug 聚合研究, 针对现有 web agent benchmark.
**参考**: agisdk (https://github.com/agi-inc/agisdk).
**素材基础**: 项目 dual-track environment / VWA bug fix 工作 ([[实验笔记]] §109 dual-track 9-cell taxonomy / B-82 / `master_bug_catalog.md` 37+ bugs).
**状态**: 方向已 locked; 具体 scope + benchmark 选型 + 时间线待 planning.

---

> 📖 **Doc update workflow** (when X happens, update which docs) → [[paper_planning#§20]]
