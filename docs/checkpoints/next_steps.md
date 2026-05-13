---
type: action-ledger
status: rolling
updated: 2026-05-10
---

# Next Steps — Forward Action Ledger

> **Future-only**. Live state 不在这里:
> - Today / 瓶颈 / cron health → [[PLAYBOOK#§1]] + [[PLAYBOOK#§2]] (🤖 GLM @daily)
> - Real-time active runs / GPU → `make active` CLI
> - Cell snapshot (active 跑中 / pending / done) → `cells.base`
> - Paper section progress → `status.base`
> - 过去 chronicle → [[实验笔记]] §1-§114
> - Strategy / theory → [[paper_planning]]
> - Advisor sync prep → [[ADVISOR_SYNC]]
> - OSF DOI lock workflow → [[osf_lock_manifest]]
> - Compute infrastructure → [[COMPUTE_INFRASTRUCTURE]]
>
> 🔧 新数据 → `make analysis` (~5-10min). Cron 每 10min 自动 sync cell frontmatter.

---

## §0 Direction

**Paper hook**: → [[paper_planning#§1]] (canonical, 3 arms / 4-fold drop-in)

> [!todo] Top 3 forward actions (priority order, 2026-05-10 evening update — Stage 3 mechanism attribution + new methods unlocked)
> 1. **Stage 4 mechanism methods queue** ⭐⭐⭐ — paper §5 升级路径; 详 §0a 新加. Cell H-d-cls (DOM target 2x2 closure, job 344623 qw on Myriad) 数据回来后 trigger §124 笔记 + decide Method 1/3/4 顺序. Method 1 (PCA cosine gap, Tool Calling Linear Circuit replicate on B1=Qwen3-VL-4B) is highest-leverage Zoom 4 self-probe — **可立即跑 partial** on existing Stage 1 hidden state cache (3 modes × 26 tasks). Full 6-mode requires Myriad hidden state extraction (DGX 96% util currently).
> 2. **Quark SSH cert → A100 SSH verify** ⭐⭐ — needed for 16-cell rerun (VWA self-host on A100). Portal cert (id_arc + id_arc.signed) + ~/.ssh/config. ETA 10 min once user has time.
> 3. **Advisor email reply wait** (~2-5d, passive) — Q1-Q11 in [[advisor_sync_5_5_followup]]. K_h1=12 / K_h3=11 / TOST δ=1.0pp threshold lock + paper split 3v4. Reply triggers OSF DOI 8-step lock + 16-cell launch gate clearance.

---

## §0a Stage 4 mechanism methods queue (added 2026-05-10 after Q1/Q2/Q3 deep critique)

**Trigger**: post-Stage 3 attribution (H-text + H-prompt cells) showed mid-layer L11/L17 disruption locus is real but no robust fusion under Spearman; user critique forced reframe of paper §5 from "fusion" → "disruption + attribution". To strengthen mechanism story beyond disruption-only, queue 4 methods (Zoom 4 model-internal probes, all feasible on B1=Qwen3-VL-4B).

**Existing methods used**: linear probe (§111 trivial), activation patching with token-overlap + LD (Stage 2/3, 12 valid cells; Spearman robust check shows no clean transfer).

### Stage 4.1: Cell H-d-cls (2x2 additivity closure) ⏳ in flight

- Job 344623 qw on Myriad, cls fwd × strong × source=som × target=dom × N=24
- Pre-registered prediction: Δ_to_target @ L11 ≈ +10.74 (= Ht_cls + Hp_cls − Cell A = 9.04 + 5.62 − 3.92)
- Falsifies if observed outside ±2pp of prediction → prompt × text interaction at mid-layer
- Bg monitor `bh702x73i` auto-computes observed Δ + ntfy on completion
- ETA 30-90 min A100 / 1.5-2.5h V100 once it leaves qw

### Stage 4.2: PCA cosine gap (Tool Calling Linear Circuit replicate, B1=Qwen3-VL-4B) ⭐ next priority

**Method**: at L11/L17/L23, PCA on hidden states across 6 modes (DOM/P-text/P-prompt/P-SoM/SoM/Vision), measure (a) cosine gap between mode-mean vectors, (b) AUROC for binary mode classification via cosine to mode mean, (c) % variance captured in top-k PCA dims (Tool Calling found 15 tools → 10 PCA dims = 90.2% var on Qwen3-4B).

**Why this answers "is it just prompt engineering"**: linear probe trivial ≠ PCA gap trivial. Even when classifier can't separate, mode means may differ on low-rank subspace (Tool Calling Linear Circuit demonstrated this on architectural cousin Qwen3-4B). If AUROC ≥ 0.8 at L11/L17 → phantom space is real representational structure; if ≈ 0.5 → paper §5 stays disruption-only.

**Existing data (already on disk, no new compute)**:
- `results/mechanistic/stage1B_archived_b1_classifieds_pilot/hidden_states.npz` — P-prompt + P-SoM, 96 examples × 37 layers × 2560 dim (cls, 26 tasks × 2 steps)
- `results/mechanistic/stage1C_image_axis_b1_cls_pilot/hidden_states.npz` — SoM + P-SoM, 96 examples × 37 layers × 2560 dim

**Immediate (today)**: 3-mode partial PCA cosine gap on existing cache (CPU-only, ~5 min) — answers "is there ANY mid-layer mode-specific structure?" using SoM/P-prompt/P-SoM.

**Full 6-mode (next 1-2 days)**: extract DOM + P-text + Vision hidden states for same 26 cls tasks. DGX is at 96% util (don't run there) → **launch as Myriad qsub** parallel to Cell H-d-cls. ~1-2h forward pass on A100, ~3-5h on V100.

**Decision tree post Method 4.2**:
- AUROC ≥ 0.8 + clean cosine gap → §5 upgrade to "phantom-mode-specific subspace at L11-L17" with figure (cosine heatmap × layer)
- AUROC ≈ 0.5 → §5 stays "disruption-only" honest framing; pivot to Method 4.3 (logit-level KL during patching) for transfer evidence

### Stage 4.3: Logit-level KL during patching ⭐ (paper §5 transfer hypothesis decisive)

**Method**: modify `p79/mechanistic/activation_patching.py` to dump first-token logit distribution at each patched layer position. Compute KL(patched ‖ source) and KL(patched ‖ target) per layer per task. Bypasses greedy decoding lock-in issue that masked transfer in token-overlap metric.

**Why**: greedy decoding can lock first-token deterministically even if logit distribution shifted toward source. Token-overlap metric misses this. Logit KL is direct distribution-level measure.

**Effort**: ~half day infra mod + 1 cell re-run on Myriad to verify. Then post-hoc on existing 12 patching cells if infra captures all needed data.

### Stage 4.4: Counterfactual activation steering (Causal proof of phantom direction)

**Method**: from Method 4.2 PCA, extract "phantom direction" = h_PSoM_mean - h_DOM_mean. During DOM forward pass, ADD this direction to L17 hidden state. Does output switch from DOM behavior to P-SoM behavior?

**Why**: tool calling circuit showed L23+ steering 80-93% accuracy switch. If our phantom direction has similar steering effect → causal proof phantom space is mechanism-level (not just correlation).

**Effort**: 1 day (re-use patching infra with vector add instead of full replace). Requires Method 4.2 to find the direction first.

### Stage 4.5: Path patching (lower priority, paper §8 future work)

**Method**: patch attention head OR MLP output specifically (not full layer). Identify sub-component carrying phantom info.

**Effort**: 2-3 days infra. Reserve for paper-2 follow-up unless Method 4.2-4.4 leave open questions.

### Routing decision (DGX vs Myriad for Stage 4 work)

- **DGX 96% GPU util currently** (other user 31GB, seonglae 5GB; 96% compute) → **don't run new GPU work on DGX**
- **Myriad available** but queue wait variable (3-9h observed today on V100/A100 mix)
- **Method 4.2 partial (3-mode existing data)**: run on DGX CPU NOW (5 min, no GPU)
- **Method 4.2 full (6-mode extraction)**: launch Myriad qsub parallel to Cell H-d-cls
- **Method 4.3/4.4**: launch Myriad qsub when ready (no DGX competition)

---

## §1 Phase 1 paper-grade rerun launch sequence (post advisor email + A100 SSH)

**Scope revised 2026-05-13 post codex stress audit** (replaces prior 16-cell phantom-only scope):

**Phase 1a (workshop-targeted, immediate launch)** — 24 operational conditions across 4 statistical cells:
- B0 × cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
- B0 × red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
- B1 × cls × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
- B1 × red × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
- **Total: 24 conditions = 2 sites × 2 models × 6 modes, 4 statistical (site, model) cells**

**Phase 1b (main paper expansion, deferred to post-workshop)** — 12 conditions:
- B0 × shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
- B1 × shop × {DOM, SoM, Vision, P-text, P-SoM, P-prompt} = 6
- Feeds R3 → R1 / Option D framing decision for main paper

**Orchestrator**: `bash scripts/queues/queue_phase1_paper_grade.sh dry-run` (preview) → `... launch` (Phase 1a default = cls + red parallel chains). Phase 1b launches via `launch phase1b shop`.

**Pre-launch gates** (orchestrator auto-checks):
1. `preregistration.md` status `locked` + no `TBD` in threshold lines (incl. 2026-05-13 K-of-N transparency reclassification propagated)
2. `results/provenance/env_<host>_baseline.json` committed
3. `results/provenance/vwa_<host>_baseline.json` committed
4. `bash scripts/preflight_v2.sh` passes
5. GPU CUDA available (smoke `python3 -c "import torch; print(torch.cuda.is_available())"`)
6. No conflicting active runs (`pgrep -f run_experiment` ≤ existing approved chains)

**ETA on A100 40GB** (Phase 1a, post-advisor lock):
| Chain | Conditions | ETA |
|---|---|---|
| cls | 12 (B0 24h → B1 48h) | 72h ≈ 3 days |
| red | 12 (B0 20h → B1 40h) | 60h ≈ 2.5 days |
| **Phase 1a wallclock (parallel)** | 24 | **~72h ≈ 3 days** |
| Phase 1b shop (post-workshop) | 12 (B0 32h → B1 64h) | 96h ≈ 4 days |

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
Output → paper §5 Table 5 quotable JSON.

---

## §2 Mechanistic Stage 2B + Stage 2C scale-up (Myriad NOW, A100 backup)

**Pre-curated dataset** (笔记 §113, commit `cd50c34`): `results/mechanistic/archive_subset_b1_cls/` (24 strong + 15 reverse, 16.5MB).

### §2a Myriad path ⭐ ACTIVE (no A100 dependency, ssh ready)

```bash
# Phase A — one-shot bootstrap (~30-45 min, login node only):
ssh myriad
bash ~/Scratch/p79/scripts/setup/myriad_bootstrap.sh
# 自动: git pull + venv + torch + p79 install + HF model pre-download (revision-pinned) + env_snapshot

# Phase B — submit batch jobs (parallel, different GPUs):
cd ~/Scratch/p79
qsub scripts/queues/qsub_stage2b_myriad.sh    # forward 24 task ~24h
qsub scripts/queues/qsub_stage2c_myriad.sh    # reverse 15 task ~12h
qstat -u $USER                                 # check status

# Phase C — monitor + retrieve:
ssh myriad tail -f ~/Scratch/p79/logs/qsub_stage2b_b1_cls.*.out
# 完成后:
scp -r myriad:~/Scratch/p79/results/mechanistic/stage2b_curated_b1_cls_myriad/ ./results/mechanistic/
```

**Why Myriad first** (vs waiting for A100 SSH):
- ✅ SSH passwordless ready 5/7 evening (`ssh myriad`)
- ✅ Stage 2B/2C 跑 frozen archive_subset, **不依赖 VWA Docker** (Myriad CGNAT block irrelevant)
- ✅ L-type 4× A100 40GB single GPU = Qwen3-VL-4B 装得下
- ✅ V/U-type 4× A100 80GB enables future Llama-4 cross-arch (A100 Condense 40GB can't)
- ✅ 不需要 IDE (terminal+qsub batch native)

### §2b A100 path (backup if Myriad queue too slow)

```bash
# 等 A100 SSH cert 之后:
ssh condense-a100
cd ~/workspace/p79  # post git clone + venv setup
.venv/bin/python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site classifieds --n-tasks 24 --step 2 --max-new-tokens 50 \
    --output-dir results/mechanistic/stage2b_curated_b1_cls_a100
```

**Expected output (either path)**:
- L11 forward causal layer confirmation (笔记 §111 task 0 → N=24)
- Reverse null effect cross-task → paper §5 strongest mechanism evidence
- Token overlap per-layer distribution

**Followup paper-grade artifact**: `run_manifest.json` aggregate field → paper Table 6 / Figure mechanism panel.

---

## §3 OSF DOI 8-step lock workflow (post advisor email)

**Trigger**: Advisor email reply with confirmed K_h1 / K_h3 / TOST δ.

**8 steps** (详 [[osf_lock_manifest]]):
1. Save advisor email PDF → `docs/reference/advisor_email_<date>.pdf`
2. Update `preregistration.md` (replace `TBD` with confirmed numbers)
3. Run `python3 scripts/provenance/snapshot_env.py` on DGX + A100 (+ Myriad if used)
4. Run `bash scripts/provenance/snapshot_vwa.sh` on each VWA host
5. `cp -r paper_drafts paper_drafts_locked` + commit
6. `git tag -a preregistration-locked -m "OSF DOI mint $(date)"` + push
7. Mint OSF DOI at https://osf.io/registries/ (link to GitHub tag URL)
8. Backfill `osf_lock_manifest.md` with all SHAs + DOI + timestamp

**Artifacts already ready** (committed 5/7):
- ✅ `env_dgx_baseline.json` (HF Qwen3-VL-4B SHA `ebb281ec...`)
- ✅ `vwa_dgx_via_quark.json` (10 containers fingerprinted)
- ✅ `osf_lock_manifest.md` (8-step checklist)
- ✅ `scripts/provenance/snapshot_env.py` + `snapshot_vwa.sh` + `numerical_determinism_check.py`
- ✅ `scripts/analysis/preregistration_decision_test.py` (smoke-tested with 3 synthetic scenarios)
- ✅ HF revision pin in `qwen3vl_agent.py` + `extract_hidden_states.py` (commit TBD)

---

## §4 Audit follow-ups (deferred to A100 SSH)

From 2026-05-07 pipeline audit (笔记 §114 follow-up):

| Pri | Item | Effort | Gating |
|---|---|---|---|
| 🔴 C3 | A100 memory + wallclock smoke (Stage 2B 1 task forward) | 30 min | A100 SSH |
| 🟡 R1 | Preflight v2 extension (B0 XOR B1 conflict / archive_subset / archived_run_dir checks) | 45 min | Independent (can do on DGX now) |
| 🟡 R2 | A100 cron / live status setup (cells.base / PLAYBOOK) | 1 h | A100 SSH + crontab dump |
| 🟡 R3 | Energy tracking pynvml test on A100 | 15 min | A100 SSH |
| 🟡 R4 | Stage 2B `--resume` flag for reboot recovery | 10 min | Independent (can do on DGX) |
| 🟡 R6 | `check_evaluator_consistency.py` (Gate 7 in `queue_phase1_paper_grade.sh`) — verify all conditions' most-recent `rederive_metadata.evaluator_code_sha` == lock-time SHA | 30 min | OSF DOI lock prep (笔记 §115 Protocol B §6) |
| 🟢 N1 | Bonferroni correction paper §3 paragraph | 10 min | Paper write phase |
| 🟢 N2 | Power analysis script | 30 min | Paper write phase |
| 🟢 N3 | Phantom variant FP rules | 1 h | Post 16-cell rerun |

**R1 + R4 可以现在做** (DGX-side, 不依赖 A100). 评估是否抢在 advisor email 来之前 fix.

---

## §5 Router experiments (Section 6, ~Week 4-5 post 16-cell)

| Cell | Blocker | Implementation |
|---|---|---|
| **Tier 1 oracle router** (TF-IDF + LR, ~3 d) | 16-cell rerun done | `p79/experiment/router.py::RuleBasedRouter` 扩展 |
| **Tier 2 first-step trigger** (~7-10 d) | Tier 1 done + step-1 trigger features | 新增 cascade runner config |
| Routing signal infra | ✅ ready (`9d7e99f`) | `confidence_summary.json` per-condition |

详 [[paper_planning#§8]] (5 决策维度: feature / target / granularity / cascade / baseline).

---

## §6 Sustainability / Green AI (Section 8 end-stage)

| Item | Status |
|---|---|
| fig9 regional carbon sensitivity (B1, 45 region) | ✅ done |
| B1 measured energy (cls + red × DOM/SoM/Vision) | ✅ ready |
| B0 token-based carbon estimator | ❌ optional Tier 3 future |
| Section 8 prose | ❌ 待 codex #17 (paper end-stage) |

---

## §7 Codex task queue

![[codex.base#Ready to send (now)]]

![[codex.base#Running / In flight]]

![[codex.base#Blocked / Queued]]

**Pending Python scripts (非 codex)**:
- ⏳ Multi-metric Pareto (cost + lat + carbon) — Section 8 前置 (~2h)
- ⏳ TF-IDF + binary feature extraction — Section 6 Tier 1 router 前置 (~1h)
- ⏳ B0 token-based carbon estimator — Section 8 Tier 3 (~20 行 helper)

---

## §8 Open issues

![[issues.base#Active blockers]]

![[issues.base#Backlog]]

---

## §9 Advisor align

详 [[ADVISOR_SYNC]] (sync prep + 5 framing decisions register + Q1-Q11 follow-up email pending reply).

---

## §10 References + quick links

### Paper drafts (final prose)
```
docs/checkpoints/paper_drafts/
  section1_intro.md          ✅ 786w
  section2_background.md     ✅ 1514w + paper.bib (57 entries)
  section3_definition.md     ✅ 863w
  section4_findings.md       🟡 1725w stale (待 codex #11)
  section5_mechanism.md      ❌ 待 codex #13 (post Stage 2B scale-up)
  section6_routing.md        ❌ 待 Tier 1+2 prototype
  section7_generalization.md ❌ 待 WA + Claude
  section8_discussion.md     ❌ paper end-stage
```

### Codex analysis docs
```
docs/analysis/phantom_paper/disagreement_clusters.md
docs/analysis/phantom_paper/cross_site_pattern_consolidation.md
docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md
docs/analysis/phantom_paper/som_vs_phantom_som_diagnostic.md
docs/analysis/B1_capability_profile.md
```

### Figures
`results/phantom_paper/figures/` (FRESH 04-28 per `make figures`):
- fig1 4-mode venn / fig2 drop-one oracle / fig3 strategy gradient / fig4 two-knob
- fig5 category × mode heatmap / fig6 capability B0-vs-B1 / fig7 cost-SR Pareto
- fig8 overlap-depth / fig9 regional carbon / fig10 phantom_lift_bars / fig11 routing_auroc_heatmap

### Key infra paths
```
configs/exp_v2_*.yaml                              per-site experiment configs
scripts/queues/queue_phase1_paper_grade.sh         🆕 Phase 1 paper-grade orchestrator (Phase 1a 24-cond default + Phase 1b shop deferred)
scripts/queues/queue_chain.sh                      sequential chain wrapper
scripts/queues/queue_phantom_*.sh                  per-cell launch
scripts/maintenance/reset_vwa_sites.sh             DGX→quark PowerShell reset
scripts/maintenance/experiment_watchdog.py         auto-clean + post-condition
scripts/provenance/snapshot_env.py                 🆕 env fingerprint (Gap 1)
scripts/provenance/snapshot_vwa.sh                 🆕 VWA Docker fingerprint (Gap 2)
scripts/provenance/numerical_determinism_check.py  🆕 cross-machine drift (Gap 5)
scripts/analysis/preregistration_decision_test.py  🆕 H1/H3/TOST canonical (Gap C2)
p79/utils/auth_refresh.py                          Playwright sign-in subprocess
p79/experiment/router.py                           RuleBasedRouter scaffold
p79/agents/qwen3vl_agent.py                        🆕 HF revision pinned (Gap C1)
p79/mechanistic/extract_hidden_states.py           🆕 HF revision pinned (Gap C1)
```

### Provenance artifacts (paper-cite-able)
```
results/provenance/env_dgx_baseline.json           DGX baseline lock value
results/provenance/vwa_dgx_via_quark.json          VWA stack fingerprint
results/provenance/preregistration_smoke_*.json    decision rule smoke tests
docs/checkpoints/pre_run/osf_lock_manifest.md              8-step DOI workflow
```

---

> 📖 **Doc update workflow** (when X happens, update which docs) → [[paper_planning#§20]]
