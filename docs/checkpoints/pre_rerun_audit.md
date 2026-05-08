# Pre-Rerun Audit Checklist — 16-Cell Phantom Routing Rerun

**Purpose**: Comprehensive paper-grade gate review before launching 16-cell rerun
on A100 (post advisor email + SSH cert). Designed to catch spec/code drift,
provenance gaps, operational issues, and methodology gaps BEFORE 48h of compute
is spent on contaminated data.

**Triggered by**: User audit prompts 2026-05-08 — sequential refinement:
(1) "整体 audit before rerun" → §A-§L process gates
(2) "paper grade 还缺少哪些" → §M-§T scientific gates
(3) "watchdog 和数据纯洁度自动检查" → §U watchdog + data purity
(4) "从头梳理: 设置 → run → 结果 → 分析 整个过程" → **this restructure (lifecycle-based)**

**Source docs**: ADVISOR_SYNC.md, advisor_sync_5_5_outcomes.md / followup.md,
preregistration.md, osf_lock_manifest.md, evaluator_change_protocol.md,
reeval_audit_protocol.md, 实验笔记 §107-§116, master_bug_catalog.md (~80 entries),
PAPER_STRATEGY_OPEN_QUESTIONS.md.

**Status**: 🟡 Active — populate as items verified. Block rerun until all 🔴 cleared.

---

## Lifecycle Overview — 4 Phases

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 实验设置 (Setup)        [pre-launch, before any cell] │
│  └─ §1.1 Code & Bug Catalog                                     │
│  └─ §1.2 Config & Hyperparameters                               │
│  └─ §1.3 Pre-Registration & Witness                             │
│  └─ §1.4 Methodology Pre-Spec (statistical / robustness)        │
│  └─ §1.5 Inter-Rater Reliability Prep                           │
│  └─ §1.6 Advisor Sync Outcomes & Open Questions                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 实验 run 过程 (Run)       [during 16-cell + Stage 2B] │
│  └─ §2.1 Pre-Launch Sanity Gate                                 │
│  └─ §2.2 Provenance Capture per Cell                            │
│  └─ §2.3 Watchdog 6-Layer Auto-Clean Protocol                   │
│  └─ §2.4 Watchdog Operational Gates                             │
│  └─ §2.5 Mid-Run Automatic Safeguards                           │
│  └─ §2.6 Cross-Cell Isolation                                   │
│  └─ §2.7 Failure-Mode Contingency / Resume Protocols            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 3: 实验结果 (Results)         [post-cell, pre-aggregate] │
│  └─ §3.1 Output Schema Conformance                              │
│  └─ §3.2 Data Quality Gates                                     │
│  └─ §3.3 Data Purity Automatic Checks                           │
│  └─ §3.4 Cost / Sustainability Tracking                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 4: 实验分析 (Analysis)        [aggregation, paper draft] │
│  └─ §4.1 Statistical Methodology Execution                      │
│  └─ §4.2 Robustness / Sensitivity Analyses                      │
│  └─ §4.3 Inter-Rater Reliability Execution (κ)                  │
│  └─ §4.4 Section 4 Limitations Disclosure Prose                 │
│  └─ §4.5 Evaluator Independence Verification                    │
│  └─ §4.6 Audit Trail & Reproducibility                          │
│  └─ §4.7 OSF DOI Lock (8-Step Workflow)                         │
└─────────────────────────────────────────────────────────────────┘
```

---

# Phase 1 — 实验设置 (Setup) — pre-launch, before any cell

## §1.1 Code & Bug Catalog

| # | Item | Status | Verify |
|---|---|---|---|
| 1.1.1 | **Early-stop disabled** (advisor 5/5 Option A cancel) | ✅ FIXED commit `3de6d95` | `grep -c "_early_stop_enabled" p79/experiment/runner/main.py` ≥ 4 |
| 1.1.2 | Phase A 4-cluster fix active (`3c15cd7`) — dispatch / cycle / RNG / page_changed | ✅ commit ≥ `3c15cd7` | `git log --oneline 3c15cd7..HEAD --stat \| head` |
| 1.1.3 | HF revision pinned in `qwen3vl_agent.py` + `extract_hidden_states.py` | ✅ commit `3b25438` | `grep "ebb281ec" p79/agents/*.py p79/mechanistic/*.py` |
| 1.1.4 | Evaluator code SHA captured in env_snapshot | ✅ commit `1304f59`+`1fefd39` | `python3 scripts/provenance/snapshot_env.py /tmp/test.json && grep evaluator_code /tmp/test.json` |
| 1.1.5 | FP filter primary `na_fp + eval_fp` (visual_fp removed §95) | ✅ commit `1fefd39` | `grep "fp_reason" p79/experiment/analysis.py` |
| 1.1.6 | rederive_metadata audit trail enabled | ✅ commit `1fefd39` | `grep "rederive_metadata" scripts/maintenance/rederive_episode_summary.py` |
| 1.1.7 | **B-35 time-based auth refresh** | ✅ FIXED today (笔记 §116.9) | `grep "seconds_since_refresh" p79/utils/auth_refresh.py` |
| 1.1.8 | Bug catalog status backfilled — 13 Phase A entries CONFIRMED → FIXED | ✅ commit `780f6c9` | `grep -c "FIXED commit" docs/reference/master_bug_catalog.md` ≥ 45 |
| 1.1.9 | Phase 0 historical bugs catalogued (§5-§90 atomic + umbrella sub-entries) | ✅ commit `49e128a` | catalog 1318+ lines, ~80 atomic entries |
| 1.1.10 | All 🔄 UNVERIFIED entries triaged to ✅ CONFIRMED or ❌ NOT_A_BUG | ✅ post-§116 | catalog Status counts: 0 UNVERIFIED |

## §1.2 Config & Hyperparameters

| # | Item | Status | Verify |
|---|---|---|---|
| 1.2.1 | 16-cell scope per `preregistration.md §4` | ✅ | `grep "N_cells" preregistration.md` = 16 |
| 1.2.2 | K_h1=12 / K_h3=11 / TOST δ=1.0pp values present (no TBD) | 🟡 pending advisor email | `grep -c "TBD" preregistration.md` after email |
| 1.2.3 | Mode operational definitions (6 modes) stipulative | ✅ | `preregistration.md` line 199 |
| 1.2.4 | `max_steps: 30` consistent across all 35 per-site YAMLs | ✅ B4 audit | `for f in configs/exp_v2_*.yaml; do grep "max_steps:" "$f"; done \| sort -u` |
| 1.2.5 | RNG seeds explicit + deterministic (B-37 partially fixed) | ✅ B5 audit | `seed: 42` in base + plumbing in `runner/main.py:81-94` |
| 1.2.6 | run_manifest.yaml grades reflect rerun plan | ✅ archived | `grep "grade:" results/phantom_paper/run_manifest.yaml \| sort \| uniq -c` |
| 1.2.7 | `time_interval_seconds: 1200` in auth_refresh config (B-35 fix) | ✅ today | `grep "time_interval_seconds" p79/experiment/config.py` |
| 1.2.8 | `state_change.form_snapshot_enabled: true` (B-67 fix verified) | ✅ §68 | DEFAULT_CONFIG check |

## §1.3 Pre-Registration & Witness

| # | Item | Status | Action when ready |
|---|---|---|---|
| 1.3.1 | Advisor email reply with K_h1/K_h3/TOST δ confirmation | 🟡 | Update `preregistration.md` § Decision log + flip `status: draft → locked` |
| 1.3.2 | `preregistration.md` `registered_at` + `registered_git_sha` | 🟡 | Fill at lock moment |
| 1.3.3 | `preregistration.md` `witnessed_by` advisor name + date | 🟡 | Fill at lock moment |
| 1.3.4 | git tag `preregistration-locked` | 🟡 | `git tag -a preregistration-locked` at lock moment |
| 1.3.5 | OSF DOI prep — see Phase 4 §4.7 8-step workflow | 🟡 | Triggered by 1.3.1 |
| 1.3.6 | Advisor email follow-up Q1-Q11 status tracking | 🟡 | `advisor_sync_5_5_followup.md` |

## §1.4 Methodology Pre-Spec (statistical / robustness)

### §1.4.1 Statistical methodology gates

| # | Item | Status | Notes |
|---|---|---|---|
| 1.4.1 | Multiple comparison correction for H1+H3+TOST family | ✅ in preregistration.md §3 | Holm-Bonferroni step-down per H-sub-family |
| 1.4.2 | Bootstrap CI procedure spec (N resamples, RNG seed, BCa vs percentile) | 🟡 partial | Add for H1/H3 oracle lift CI |
| 1.4.3 | Power analysis / MDE | ✅ today | `scripts/analysis/power_analysis.py` + `docs/analysis/cross_sites/power_analysis.md` |
| 1.4.4 | Effect size reporting (Cohen's h alongside p-values) | 🟡 partial | Add for paper §5 Table 5 |
| 1.4.5 | Outlier / extreme value handling rule | 🔴 TBD | Add system-crash exclusion explicit |
| 1.4.6 | FP filter sensitivity ladder (3 variants) | ✅ in preregistration.md | aggregate_sr_fp_per_mode.py |
| 1.4.7 | Reporting precision standards (2dp pp / 1dp diff / int counts) | 🟡 implicit | Make explicit in paper §3 |

### §1.4.2 Robustness pre-spec (paper §3 commit before lock)

| # | Item | Status | Notes |
|---|---|---|---|
| 1.4.8 | Non-visual subset robustness (43 VWA + 480 WA = 523 manually-audited) | ✅ in preregistration.md | Replaces deprecated visual_fp |
| 1.4.9 | Pre-Phase-A archive robustness (Appendix D) | ✅ in preregistration.md | Symmetric contamination disclosure |
| 1.4.10 | K_h1 / K_h3 threshold sensitivity (also report ±1) | 🔴 TBD | Show threshold gradient |
| 1.4.11 | Per-difficulty bucket analysis | 🔴 TBD | 3 difficulty terciles |
| 1.4.12 | Hold-out site validation (LOSO if advisor confirms) | 🟡 advisor email | `preregistration.md` mentions LOSO as alternative |
| 1.4.13 | Cross-machine reproducibility (DGX/A100/Myriad) | 🟡 | numerical_determinism_check post-rerun |

## §1.5 Inter-Rater Reliability Prep (κ ≥ 0.7 targets)

| # | Item | Status | Target |
|---|---|---|---|
| 1.5.1 | FP labeling reliability — 30-task pilot, 2 raters | 🔴 TBD pre-rerun pilot | Cohen κ ≥ 0.7 per preregistration.md |
| 1.5.2 | Failure-mode 5-bucket rubric reliability | 🔴 TBD | κ ≥ 0.7 target |
| 1.5.3 | Codex-as-rater calibration spot-check | 🔴 TBD | Disagreement >30% triggers prompt revision |
| 1.5.4 | Visual subset audit | ✅ exists `docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py` | Used as Appendix D |

## §1.6 Advisor Sync Outcomes & Open Questions

### §1.6.1 Advisor 5/5 sync outcomes (advisor_sync_5_5_outcomes.md §A)

| # | Item | Status |
|---|---|---|
| 1.6.1 | A.1 Early-stop A 全 cancel | ✅ code fixed (1.1.1) |
| 1.6.2 | A.2 Manifest 全 archive + 16-cell rerun | ✅ run_manifest.yaml grade=archived |
| 1.6.3 | A.3 Paper 拆开发 (split direction) | 🟡 exact count Q1 advisor email |
| 1.6.4 | A.4 VWA bug → ACL position paper | ✅ accepted, out of immediate scope |
| 1.6.5 | A.5 Routing benchmark 独立成文 | ✅ accepted |
| 1.6.6 | A.6 Mechanistic interpretability publication-worthy | ✅ Stage 2B/2C running on Myriad |
| 1.6.7 | A.7 Workshop submission 节奏 | ✅ accepted |
| 1.6.8 | A.8 Compute paths (A100 / Myriad / advisor 5090) | A100 🟡 SSH cert / Myriad ✅ |
| 1.6.9 | A.9 Pre-reg witness mechanism (git+email+OSF) | 🟡 advisor email |
| 1.6.10 | A.10 Environment 3-layer framework | ✅ accepted |

### §1.6.2 Open questions resolution (PAPER_STRATEGY_OPEN_QUESTIONS.md)

| Q | Title | Status | Notes |
|---|---|---|---|
| Q1 🔴 | Early-stop bias | ✅ A1 + 1.1.1 cancel | This audit |
| Q2 🟡 | B0 pre/post Phase A asymmetry | ✅ 16-cell rerun handles | |
| Q3 🟢 | Env non-determinism | ✅ via snapshot_vwa.sh | Paper §3 |
| Q4 ❌ RETRACTED | Cross-site SR comparability | — | |
| Q5 ❌ RETRACTED | FP filter asymmetry | — | |
| Q6 🟢 | Diamond completion | 🟡 | depends on rerun |
| Q7 🟡 | B0 vs B1 sampling regime | 🟡 | paper §3 limit |
| Q8 🟢 | Drop-one oracle observed-mode-set | ✅ | |
| Q9 🟢 | Routing AUROC in-sample | 🟡 | depends on H7-H8 |

---

# Phase 2 — 实验 run 过程 (Run) — during 16-cell + Stage 2B/2C

## §2.1 Pre-Launch Sanity Gate (`glm_pre_launch_check.py`)

| # | Hard rule | Exit code on violation |
|---|---|---|
| 2.1.1 | RESET_BEFORE=1 enforced for paper-grade cell | 2 (BLOCK) |
| 2.1.2 | Same-site B0 XOR B1 (no parallel) | 2 (BLOCK) |
| 2.1.3 | Queue script ↔ baseline ↔ site ↔ mode arg consistency | 1 (WARN) |
| 2.1.4 | Config `benchmark` matches site (vwa vs wa) | 1 (WARN) |
| 2.1.5 | No conflicting `pgrep -f run_experiment.*<site>` | 2 (BLOCK) |
| **Verify** | `bash scripts/maintenance/launch.sh ... DRY=1` | Exit 0 = OK |

## §2.2 Provenance Capture per Cell

| # | Item | Status | Verify |
|---|---|---|---|
| 2.2.1 | env_snapshot.py auto-runs at run_experiment.py post-runner.run() | ✅ commit `1304f59` | inspect `<run_dir>/env_snapshot.json` after first cell |
| 2.2.2 | snapshot_vwa.sh (DGX baseline + A100 self-host) | DGX ✅ / A100 🟡 | `ls results/provenance/vwa_*.json` |
| 2.2.3 | numerical_determinism_check ready | 🟡 needs A100/Myriad SSH | script exists per §114 Gap 5 |
| 2.2.4 | sitecustomize.py shim (Myriad-only RHEL 7) committed | ✅ | `myriad_bootstrap.sh` |
| 2.2.5 | constraints.txt (urllib3<2 / numpy<2) | ✅ Myriad only | `myriad_constraints.txt` |
| 2.2.6 | env_snapshot includes `evaluator_code.combined_sha256` | ✅ commit `1304f59` | `jq .evaluator_code.combined_sha256 env_snapshot.json` |

## §2.3 Watchdog 6-Layer Auto-Clean Protocol (笔记 §95 + §107)

| # | Layer | Status | Verify |
|---|---|---|---|
| 2.3.1 | **Detect** — step_000 DOM login marker scan | ✅ B-41 §14 | `experiment_watchdog.py::_check_session_health` |
| 2.3.2 | **Alert** — ntfy push priority=high to `$NTFY_TOPIC` | ✅ | `_post_ntfy()` line 63 |
| 2.3.3 | **Refresh** — auth_refresh.py subprocess | ✅ B-41/B-67 | `_auto_refresh_auth()` line 111 |
| 2.3.4 | **Cleanup** — purge contaminated episodes (10-min mtime guard) | ✅ B-49/B-51 | `_purge_digest_records()` line 212 + orphan prune line 1188 |
| 2.3.5 | **Resume** — runner re-attempts via dedup | ✅ B-46f | `runner/main.py` resume protocol |
| 2.3.6 | **Verify** — automated post-rerun spot-check | 🟡 manual | TBD: `verify_auth_signatures.py` |

## §2.4 Watchdog Operational Gates

| # | Item | Status | Verify |
|---|---|---|---|
| 2.4.1 | Watchdog process alive during cell | 🟡 per-launch | `pgrep -f experiment_watchdog` non-empty |
| 2.4.2 | `--reset-state` flag clears stale state.json | ✅ B-54i | restart_watchdog.sh handles |
| 2.4.3 | ntfy topic configured per cell | 🟡 | Verify `$NTFY_TOPIC` env in queue script |
| 2.4.4 | Watchdog log rotation | 🟡 | TBD: log size cap or logrotate |
| 2.4.5 | Watchdog self-restart on crash | ✅ | restart_watchdog.sh |
| 2.4.6 | Watchdog idle self-exit | ✅ | line 1698 |
| 2.4.7 | Cross-site NOT-LOGGED-IN false-positive guard (B-67/B-74) | ✅ | site-specific marker matching |
| 2.4.8 | Post-condition analysis trigger | ✅ | `_run_post_condition_analysis` line 595 |
| 2.4.9 | Auto-runs `make rederive` post-condition | 🟡 | Verify pipeline integration |

## §2.5 Mid-Run Automatic Safeguards

| # | Trigger | Action | Status |
|---|---|---|---|
| 2.5.1 | ≥3 consecutive auth failures | ntfy + auto-refresh | ✅ |
| 2.5.2 | ≥5 consecutive episode failures | ntfy + halt cell | 🔴 TBD |
| 2.5.3 | API 503 cascade (B-50d) | exponential backoff 3 attempts | ✅ |
| 2.5.4 | GPU OOM | watchdog kill + restart | 🟡 partial |
| 2.5.5 | VWA Docker container down | ntfy + halt | 🟡 manual |
| 2.5.6 | Disk >95% full | ntfy + halt | 🔴 TBD |
| 2.5.7 | env_snapshot evaluator_code SHA mismatch | ntfy (drift detection) | 🔴 TBD (R6) |
| 2.5.8 | Episode wallclock >30 min | ntfy + kill | 🟡 manual |

## §2.6 Cross-Cell Isolation

| # | Item | Status | Verify |
|---|---|---|---|
| 2.6.1 | Each cell unique RUN_ID | ✅ | `<baseline>_<mode>_<site>_YYYYMMDD` pattern |
| 2.6.2 | Cell directory isolation (no shared episodes/) | ✅ | run_dir convention |
| 2.6.3 | RESET_BEFORE=1 between cells of same site | ✅ enforced via 2.1.1 | hard rule |
| 2.6.4 | Auth state regeneration per-site per-cell | ✅ B-66 | per-site files |
| 2.6.5 | Watchdog state.json reset between cells | ✅ B-54i | restart handles |
| 2.6.6 | env_snapshot.json dumped at each cell launch | ✅ | run_experiment.py hook |

## §2.7 Failure-Mode Contingency / Resume Protocols

| # | Scenario | Status | Protocol |
|---|---|---|---|
| 2.7.1 | A100 GPU OOM mid-cell | 🟡 partial | Watchdog auto-clean; verify resume |
| 2.7.2 | Myriad qsub job killed (wallclock) | 🟡 partial | run_stage2b incremental save; --resume flag (R4 pending) |
| 2.7.3 | VWA Docker restart mid-rerun | 🟡 | RESET_BEFORE=1 + auth_refresh; ntfy on >3 fails |
| 2.7.4 | B0 proxy API rate-limit | ✅ B-50d | Exponential backoff 10/20/40s |
| 2.7.5 | Phase A locator-route regression | 🔴 TBD | Halt-on-N-consecutive-fail detection |
| 2.7.6 | Disk full mid-cell | 🟡 | Pre-launch verify (1.6); mid-launch monitor TBD |
| 2.7.7 | Network partition (DGX↔quark Tailscale OR A100↔bastion) | 🟡 | A100 self-host VWA solves |
| 2.7.8 | Cell completes with `expected_n - actual_n > 5` | 🔴 TBD | Halt subsequent cells until investigated |

---

# Phase 3 — 实验结果 (Results) — post-cell, pre-aggregate

## §3.1 Output Schema Conformance

| # | Item | Verify |
|---|---|---|
| 3.1.1 | step JSONL schema v2 catalog (笔记 §97) | `p79/experiment/schema_migrations/` |
| 3.1.2 | episode_summary `adjusted_success` + `fp_reason` populate | post-cell `make rederive` then `head episodes/*.json` |
| 3.1.3 | condition_summary_v2.json aggregation | post-cell `cat condition_summary_v2.json` |
| 3.1.4 | Logs format consistent (JSON-L) | `head <run_dir>/log.jsonl` |
| 3.1.5 | env_snapshot.json `evaluator_code.combined_sha256` matches lock SHA | `jq .evaluator_code.combined_sha256` |

## §3.2 Data Quality Gates

| # | Item | Status | Verify |
|---|---|---|---|
| 3.2.1 | Episode completeness (no silent skips) | 🟡 | Per cell: `ls episodes/ \| wc -l == expected_n` |
| 3.2.2 | N balance ≥100 per cell | ✅ specified | `aggregate_phantom_lift.py` excludes <100 |
| 3.2.3 | Site state contamination check (cart / listing diff between cells) | 🔴 TBD | `scripts/maintenance/site_state_snapshot.sh` |
| 3.2.4 | Auth state freshness per cell | 🟡 | mtime within 1h of launch |
| 3.2.5 | Cross-cell shared task pool (paired comparisons) | ✅ | `aggregate_phantom_lift.py` common universe |
| 3.2.6 | Step JSONL no corruption | 🟡 | `read_jsonl_dedup` corrupt counter zero |
| 3.2.7 | Schema v2 conformance | ✅ | `tests/test_step_schema_v2.py` |
| 3.2.8 | Wall-clock outliers per task (>3σ flag) | 🔴 TBD | Add to analyze_run pipeline |

## §3.3 Data Purity Automatic Checks (笔记 §107 Phase A wave verified)

| # | Item | Status | Implementation |
|---|---|---|---|
| 3.3.1 | JSONL dedup on every step file read | ✅ | `p79.experiment.io_utils.read_jsonl_dedup` |
| 3.3.2 | Orphan artifact prune (10-min mtime guard) | ✅ B-51 | `experiment_watchdog.py:1188` |
| 3.3.3 | Stale summary detection (`done < total` → re-run) | ✅ B-61b | `is_condition_complete` |
| 3.3.4 | Per-episode auth refresh (Magento 302 + B-35 time-based) | ✅ B-70/B-35 | per-cell + 1200s fallback |
| 3.3.5 | Backup before re-derive (.bak_pre_rederive) | ✅ §97 | one-shot, never overwrite |
| 3.3.6 | rederive_metadata audit trail | ✅ §115 Protocol B | append-only |
| 3.3.7 | Phase A scoring fix verified (13 catalog entries) | ✅ §116.9 | catalog updated 2026-05-08 |
| 3.3.8 | Site state snapshot pre/post-cell | 🔴 TBD | `site_state_snapshot.sh` |
| 3.3.9 | Disk space monitor mid-cell | 🔴 TBD | cron `df -h ~/Scratch` + ntfy |
| 3.3.10 | GPU memory leak detection | 🟡 partial | watchdog has B-62 fix per BLIP-2 |

## §3.4 Cost / Sustainability Tracking

| # | Item | Status | Verify |
|---|---|---|---|
| 3.4.1 | Per-cell GPU-hours estimate | ✅ `condition_summary_v2.json` | aggregate post-rerun |
| 3.4.2 | Per-cell USD cost (B0 API) | ✅ `cost_usd.model` per step | aggregate post-rerun |
| 3.4.3 | Carbon footprint per cell (45-region) | ✅ `aggregate_cost_electricity.py` | run post-rerun |
| 3.4.4 | Total compute budget tracking | 🟡 | PLAYBOOK §1 GLM-managed |
| 3.4.5 | Cross-platform GPU power profile | 🔴 TBD | NVML probe per cell start |
| 3.4.6 | Section 8 prose draft | 🔴 TBD post-rerun | references R1-R5 |

---

# Phase 4 — 实验分析 (Analysis) — aggregation, paper draft

## §4.1 Statistical Methodology Execution

| # | Item | Status | Verify post-rerun |
|---|---|---|---|
| 4.1.1 | Run `preregistration_decision_test.py` with locked thresholds | 🟡 | `K_h1=12 --K_h3=11 --TOST-delta=1.0` (after advisor email) |
| 4.1.2 | Holm-Bonferroni step-down per H-sub-family | ✅ pre-spec'd | preregistration.md §3 |
| 4.1.3 | Bootstrap CI (BCa) on H1/H3 oracle lift | 🟡 partial | Spec N=1000 resamples + RNG seed=42 |
| 4.1.4 | Cohen's h effect size alongside p-values | 🟡 partial | Add to paper §5 Table 5 |
| 4.1.5 | Post-rerun power analysis re-run with observed SR | 🔴 TBD | Update `power_analysis.py --baseline-sr <observed>` |

## §4.2 Robustness / Sensitivity Analyses (run all)

| # | Analysis | Status | Verify |
|---|---|---|---|
| 4.2.1 | Non-visual subset (43 VWA + 480 WA) | ✅ pre-spec'd | `docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py` |
| 4.2.2 | Pre-Phase-A archive Appendix D | ✅ pre-spec'd | re-run analysis on archived cells |
| 4.2.3 | FP filter sensitivity (raw / +na_fp / +na_fp+eval_fp) | ✅ pre-spec'd | `aggregate_sr_fp_per_mode.py` 3 variants |
| 4.2.4 | K_h1 / K_h3 ±1 threshold gradient | 🔴 TBD | re-run preregistration_decision_test with K±1 |
| 4.2.5 | Per-difficulty bucket (intent length / N actions / has_ref_image) | 🔴 TBD | Add bucketing in aggregate scripts |
| 4.2.6 | Hold-out site validation (LOSO if locked) | 🟡 advisor email | `router_split.py` LOSO mode |
| 4.2.7 | Cross-machine numerical agreement (DGX/A100/Myriad) | 🟡 needs A100/Myriad SSH | `numerical_determinism_check.py compare` |

## §4.3 Inter-Rater Reliability Execution (κ ≥ 0.7)

| # | Item | Status | Output |
|---|---|---|---|
| 4.3.1 | FP labeling 30-task pilot (2 raters) | 🔴 TBD pre-rerun | Cohen κ report |
| 4.3.2 | Failure-mode 5-bucket rubric reliability | 🔴 TBD | κ report |
| 4.3.3 | Codex-as-rater calibration spot-check | 🔴 TBD | Disagreement <30% threshold |

## §4.4 Section 4 Limitations Disclosure Prose

**Source**: `docs/checkpoints/paper_drafts/section4_limitations_disclosure.md` (created today, ~10 subsections)

| # | Item | Status | Source |
|---|---|---|---|
| 4.4.1 | B-20 ua_match GPT-judge drift prose | ✅ today | Section 4.X.1 |
| 4.4.2 | B-21 string_match fuzzy_threshold misnomer | ✅ today | Section 4.X.2 |
| 4.4.3 | B-22 program_html selector brittleness | ✅ today | Section 4.X.3 |
| 4.4.4 | B-15 finish_wrong_state (handled by §95 FP) | ✅ today | Section 4.X.4 |
| 4.4.5 | B-26 in_viewport_ratio operator precedence | ✅ today | Section 4.X.5 |
| 4.4.6 | B-28 scroll direction (mitigated via §67) | ✅ today | Section 4.X.6 |
| 4.4.7 | A1/A3 baseline-design asymmetries (B-56) | ✅ today | Section 4.X.7 |
| 4.4.8 | Cross-machine numerical drift | ✅ template | Section 4.X.8 (post-rerun fill numbers) |
| 4.4.9 | Pre-Phase-A vs post-Phase-A asymmetry | ✅ today | Section 4.X.9 |
| 4.4.10 | Stage 2B input vintage independence | ✅ today | Section 4.X.10 |

## §4.5 Evaluator Independence Verification

| # | Item | Status | Verify |
|---|---|---|---|
| 4.5.1 | VWA evaluator code unchanged from upstream | ✅ | `git diff upstream/main -- external/visualwebarena/evaluation_harness/` |
| 4.5.2 | GPT-4o-mini judge prompt template pinned | 🟡 | `helper_functions.py:llm_fuzzy_match` no edits |
| 4.5.3 | Judge model temperature explicit (=0) | 🟡 | Verify; if non-zero disclose |
| 4.5.4 | Episode-level eval reproducibility (N=20 spot-check) | 🔴 TBD | Add `eval_reproducibility_check.py` |
| 4.5.5 | Cross-evaluator-version sensitivity (Protocol B) | ✅ §115 | reeval_audit_protocol.md |

## §4.6 Audit Trail & Reproducibility (reviewer-defensible chain)

After rerun, the following chain reconstructs any cell's adjusted_SR:

1. `git show <commit-at-lock>:p79/experiment/analysis.py` (canonical FP rules)
2. `git show <commit-at-lock>:scripts/provenance/snapshot_env.py` (env capture spec)
3. `<run_dir>/env_snapshot.json` (machine + HF + evaluator SHA at run time)
4. `<condition>/episodes/*.json` `rederive_metadata` (per-episode audit trail)
5. `<run_dir>/run_manifest.yaml` cell entries with grade=paper-grade
6. OSF DOI page citing git SHA + advisor email message-id
7. `master_bug_catalog.md` Status fields with commit refs (post-§116.9 backfill)

## §4.7 OSF DOI Lock (8-Step Workflow)

**Trigger**: Advisor email reply with confirmed K_h1 / K_h3 / TOST δ.

8 steps from `osf_lock_manifest.md`:
1. Save advisor email PDF → `docs/reference/advisor_email_<date>.pdf`
2. Update `preregistration.md` (replace TBD with confirmed numbers)
3. Run `python3 scripts/provenance/snapshot_env.py` on DGX + A100 + Myriad
4. Run `bash scripts/provenance/snapshot_vwa.sh` on each VWA host
5. `cp -r paper_drafts paper_drafts_locked` + commit
6. `git tag -a preregistration-locked` + push
7. Mint OSF DOI at https://osf.io/registries/ (link to GitHub tag URL)
8. Backfill `osf_lock_manifest.md` with all SHAs + DOI + timestamp

---

## Decision Flow (use this before launching)

```
Phase 1 verify ALL ✅:
  All §1.1 code-state fixes deployed
  §1.2 config items match preregistration.md scope
  §1.3.1 advisor email arrived → unlock 1.4-1.6 lock checklist
  §1.4 statistical pre-spec ✅
  §1.5 inter-rater pilots done (or scheduled before lock)
  §1.6 advisor outcomes acted on
  ↓
Phase 2 launch checklist:
  §2.1 pre-launch sanity gate passes (RC 0)
  §2.2 provenance script runs successfully on target machine
  §2.4 watchdog alive
  §2.6 cross-cell isolation verified per cell launch
  ↓
Phase 3 mid-rerun monitor:
  cells.base shows progress (cron 10min)
  PLAYBOOK §1+§2 GLM real-time
  Per-cell env_snapshot SHA matches lock SHA — drift halts
  ↓
Phase 4 post-rerun analysis:
  make analysis (full pipeline)
  python3 preregistration_decision_test.py with locked thresholds
  Run all §4.2 robustness analyses
  Execute §4.3 inter-rater κ pilots
  Fill §4.4 limitations prose with post-rerun numbers
  Verify §4.5 evaluator independence
  §4.7 mint OSF DOI
```

---

## §References

- `docs/checkpoints/preregistration.md` (canonical commitment)
- `docs/checkpoints/ADVISOR_SYNC.md` / `advisor_sync_5_5_outcomes.md` / `advisor_sync_5_5_followup.md`
- `docs/checkpoints/osf_lock_manifest.md` (8-step DOI workflow)
- `docs/checkpoints/evaluator_change_protocol.md` (Protocol A — Tier classification)
- `docs/checkpoints/reeval_audit_protocol.md` (Protocol B — episode audit trail)
- `docs/checkpoints/paper_drafts/section4_limitations_disclosure.md` (created 5/8 — 10 prose drafts)
- `docs/reference/master_bug_catalog.md` (~80 catalog entries, post §116.9 backfill)
- `docs/reference/PAPER_STRATEGY_OPEN_QUESTIONS.md` (Q1-Q9)
- `docs/analysis/cross_sites/power_analysis.md` (created 5/8 — paper §3 cite-ready)
- `docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py` (visual subset audit)
- 实验笔记 §107 (Phase A) / §110 (5/5 sync) / §114 (provenance) / §115 (Protocol A+B) / §116 (audit + restructure)

---

**Last restructure**: 2026-05-08, 笔记 §116.12 — lifecycle-based reorganization (4 phases × 18 sections × ~150 gate items).
