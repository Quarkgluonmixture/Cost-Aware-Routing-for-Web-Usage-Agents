---
type: action-ledger
status: rolling
updated: 2026-05-14
---

# Next Steps — Forward Action Ledger

> **Future-only**. Live state 不在这里:
> - Today / 瓶颈 / cron health → [[PLAYBOOK#§1]] + [[PLAYBOOK#§2]] (🤖 GLM @daily)
> - Real-time active runs / GPU → `make active` CLI
> - Cell snapshot (active 跑中 / pending / done) → `cells.base`
> - Paper section progress → `status.base`
> - 过去 chronicle → [[实验笔记]] (latest §134)
> - Strategy / theory → [[paper_planning]]
> - Advisor sync prep → [[ADVISOR_SYNC]] + [[followup]]
> - OSF DOI lock workflow → [[osf_lock_manifest]]
> - Compute infrastructure → [[COMPUTE_INFRASTRUCTURE]]
>
> 🔧 新数据 → `make analysis` (~5-10min). Cron 每 10min 自动 sync cell frontmatter.

---

## §0 Direction

**Paper hook**: → [[paper_planning#§1]] (canonical, phantom routing space 3 arms / 4-fold drop-in)

> [!todo] Top 3 forward actions (priority order, 2026-05-14 — advisor sync day; mechanism v2 landed; Phase 1a infra audited)
> 1. **Advisor sync 今天** ⭐⭐⭐ — [[followup]] 已备 (Part 1 novelty 重列 + Part 2 决策点). 需拿到的决策: (a) paper 分配 (几篇 / 以哪篇为主), (b) SAE 是否进 paper-1, (c) pre-reg 三阈值 K_h1 / K_h3 / TOST δ, (d) routing train/test split (5-fold vs LOSO), (e) mechanistic scope (B1-only vs cross-arch), (f) DL meta k=4 → REML+Hartung-Knapp vs DL+disclose. 周报 dashboard 已重建 (`周报_5.14.md` + weekly-dashboard/dist).
> 2. **Fire Phase 1a 24-condition 干净重跑** ⭐⭐⭐ — §134 已 audit+fix 全部 queue infra (13 fix: C1 FORCE_NEW / C2 preflight gate / C3 chain crash detect / C6 active-run fatal 等). Gated on: advisor pre-reg lock → OSF DOI lock → launch. 详 §1.
> 3. **OSF DOI 8-step lock** ⭐⭐ — advisor 确认阈值后立即跑. 详 §3.

---

## §0a Mechanism (§5) — v2 landed, remaining forward items

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

## §1 Phase 1a paper-grade rerun launch sequence

**Scope** (2026-05-13 codex stress audit 定): 24 operational conditions / 4 statistical cells.
"condition" = 1 (site, model, mode) launch unit; "cell" = 1 (site, model) stratification unit. **不要混用**.

**Phase 1a (workshop-targeted, immediate)** — 24 conditions:
- {B0, B1} × {cls, red} × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 24 conditions, 4 cells

**Phase 1b (main paper expansion, deferred post-workshop)** — 12 conditions:
- {B0, B1} × shop × 6 modes. Feeds R3 → R1 / Option D framing decision.

**Orchestrator**: `bash scripts/queues/queue_phase1_paper_grade.sh dry-run` → `... launch` (Phase 1a = cls + red parallel chains). `FORCE_NEW=1` 自动 export (§134 C1 fix — 不复用 pre-fix archived dirs).

**Pre-launch gates** (orchestrator auto-checks, §134 已 harden):
1. `preregistration.md` status `locked` + no `TBD` (含 K-of-N transparency-only 2026-05-13 reclassification)
2. `results/provenance/env_<host>_baseline.json` + `vwa_<host>_baseline.json` committed
3. Gate 4: `preflight_v2.sh` exit code captured (§134 C2 fix — 不再 theatrical)
4. Gate 5: GPU/CUDA available (blocking)
5. Gate 6: 无 conflicting active runs — fatal unless `ALLOW_ACTIVE_RUNS=1` (§134 C6 fix)
6. Gate 7: config-existence check 全 Phase 1a + 1b configs (§134 C9 fix)

**ETA (A100 40GB, Phase 1a parallel)**: ~72h ≈ 3 days (cls 12 cond 72h / red 12 cond 60h). Phase 1b shop +96h post-workshop.

**Post-completion**:
```bash
make analysis                                    # rerun all aggregators + figures
python3 scripts/analysis/preregistration_decision_test.py \
    --per-task-csv results/phantom_paper/per_task_sr.csv \
    --primary-gate drop_one_pooled_meta_superiority \
    --H1-magnitude-pp 1.0 --TOST-delta-pp 1.0 \
    --transparency-K_h1 3 --transparency-K_h3 3 \
    --out results/phantom_paper/preregistration_test_results.json
```
Output → paper §1 hero numbers replace 现有 `(provisional)` archived 数 + paper §5 Table quotable JSON.

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
| C-M1 / F1 | DL meta τ² biased at k=4 (Veroniki 2016) → REML / Paule-Mandel? | advisor 拍板 |
| C-M2 / F2 | Wald 1.96 CI anti-conservative at k=4 → Hartung-Knapp t_{k-1}=3.18? | advisor 拍板 |
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

## §5 Router experiments (Section 6 / paper-2, post Phase 1a)

| Cell | Blocker | Implementation |
|---|---|---|
| **Tier 1 oracle router** (TF-IDF + LR, ~3 d) | Phase 1a rerun done | `p79/experiment/router.py::RuleBasedRouter` 扩展 |
| **Tier 2 first-step trigger** (~7-10 d) | Tier 1 done + step-1 trigger features | 新增 cascade runner config |
| Routing signal infra | ✅ ready | `confidence_summary.json` per-condition |

Train/test split protocol → advisor sync 决策 (5-fold site-stratified CV vs LOSO; 倾向 5-fold). 详 [[paper_planning#§8]].

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

详 [[ADVISOR_SYNC]] + [[followup]] (2026-05-14 sync — Part 1 novelty + Part 2 决策点). Sync 后:
decision log 写 [[paper_planning]] + framing decisions register → ADVISOR_SYNC status open → discussed.

---

## §10 References + quick links

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
scripts/queues/queue_phase1_paper_grade.sh         Phase 1a 24-cond orchestrator (§134 harden)
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

> 📖 **Doc update workflow** (when X happens, update which docs) → [[paper_planning#§20]]
