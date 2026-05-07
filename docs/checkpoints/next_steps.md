---
type: action-ledger
status: rolling
updated: 2026-05-07
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

> [!todo] Top 3 forward actions (priority order, 2026-05-07 update post-§114 provenance)
> 1. **Quark SSH cert + A100 SSH verify** ⭐⭐⭐ — UCL Condense A100 40GB (10.52.6.89, ProxyJump via `ssh.condenser.arc.ucl.ac.uk`). User-side: portal cert (id_arc + id_arc.signed) + ~/.ssh/config. ETA 10 min. **Unblocks**: 16-cell rerun launch (R5 orchestrator) + Mechanistic Stage 2B scale-up + provenance A100 baseline (§5 below).
> 2. **Advisor email reply wait** (~2-5d, passive) — Q1-Q11 in [[advisor_sync_5_5_followup]]. Critical: K_h1=12 / K_h3=11 / TOST δ=1.0pp threshold lock + paper split 3v4. Reply triggers OSF DOI 8-step lock + 16-cell launch gate clearance.
> 3. **Mechanistic Stage 2B curated scale-up** (post 1+2 parallel) — 24 strong + 15 reverse mirage tasks (`results/mechanistic/archive_subset_b1_cls/`). A100 wallclock: ~24h forward + ~12h reverse. Paper §5 mechanism upgrade from N=3 pilot to N=24 paper-grade.

---

## §1 16-cell rerun launch sequence (post advisor email + A100 SSH)

**Scope** (post-5/5 sync, student-decided 16 cells):
- B0×{cls, red}×3 phantom (P-text / P-SoM / P-prompt) = 6
- B1×{cls, red}×3 phantom = 6
- B0×shop×{P-text, P-SoM} = 2 (P-prompt scope cut, advisor confirmed)
- B1×shop×{P-text, P-SoM} = 2
- **Total: 16 cells**

**Orchestrator**: `bash scripts/queues/queue_16cell_paper_grade.sh dry-run` (preview) → `... launch` (3 parallel chains: cls / red / shop).

**Pre-launch gates** (orchestrator auto-checks):
1. `preregistration.md` no `TBD` in K_h1 / K_h3 / TOST_delta lines
2. `results/provenance/env_<host>_baseline.json` committed
3. `results/provenance/vwa_<host>_baseline.json` committed
4. `bash scripts/preflight_v2.sh` passes
5. GPU CUDA available (smoke `python3 -c "import torch; print(torch.cuda.is_available())"`)
6. No conflicting active runs (`pgrep -f run_experiment` ≤ existing approved chains)

**ETA on A100 40GB** (post-advisor lock):
| Chain | Cells | ETA |
|---|---|---|
| cls | 6 (B0 12h → B1 24h) | 36h |
| red | 6 (B0 10h → B1 20h) | 30h |
| shop | 4 (B0 16h → B1 32h) | 48h |
| **Total wallclock (parallel)** | 16 | **~48h ≈ 2 days** |

**Post-completion**:
```bash
make analysis                                    # rerun all aggregators + figures
python3 scripts/analysis/preregistration_decision_test.py \
    --cells-csv results/phantom_paper/cells_aggregated.csv \
    --K_h1 12 --K_h3 11 --TOST-delta 1.0 \
    --out results/phantom_paper/preregistration_test_results.json
```
Output → paper §5 Table 5 quotable JSON.

---

## §2 Mechanistic Stage 2B + Stage 2C scale-up (A100 parallel)

**Pre-curated dataset** (笔记 §113, commit `cd50c34`): `results/mechanistic/archive_subset_b1_cls/` (24 strong + 15 reverse, 16.5MB).

**Launch commands** (A100 SSH 通后):
```bash
# Stage 2B forward direction (24 task × 36 layer × 50 max_new_tokens)
.venv/bin/python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --site classifieds --n-tasks 24 --step 2 --max-new-tokens 50 \
    --output-dir results/mechanistic/stage2b_curated_b1_cls
    # Auto-emits: env_snapshot.json + run_manifest.json (Gap 3 §114)

# Stage 2C reverse direction (15 task asymmetry confirm)
.venv/bin/python3 scripts/mechanistic/run_stage2b_continuation_pilot.py \
    --reverse --site classifieds --n-tasks 15 --step 2 \
    --output-dir results/mechanistic/stage2c_reverse_curated_b1_cls
```

**Expected output**:
- L11 forward causal layer confirmation (笔记 §111 task 0 finding extended to N=24)
- Reverse null effect cross-task confirm → paper §5 strongest mechanism evidence
- Token overlap distribution histogram per layer

**Wallclock A100**: forward ~24h + reverse ~12h (parallel-able if memory allows, sequential safer)

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
scripts/queues/queue_16cell_paper_grade.sh         🆕 16-cell orchestrator (post-advisor)
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
docs/checkpoints/osf_lock_manifest.md              8-step DOI workflow
```

---

> 📖 **Doc update workflow** (when X happens, update which docs) → [[paper_planning#§20]]
