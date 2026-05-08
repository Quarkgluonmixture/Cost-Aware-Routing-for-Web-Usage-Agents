# Pre-Rerun Audit Checklist — 16-Cell Phantom Routing Rerun

**Purpose**: Comprehensive paper-grade gate review before launching 16-cell rerun
on A100 (post advisor email + SSH cert). Designed to catch spec/code drift,
provenance gaps, operational issues, and methodology gaps BEFORE 48h of compute
is spent on contaminated data.

**Triggered by**: User audit prompts 2026-05-08 — sequential refinement:
(1) "整体 audit before rerun" → §A-§L process gates
(2) "paper grade 还缺少哪些" → §M-§T scientific gates
(3) "watchdog 和数据纯洁度自动检查" → §U watchdog + data purity
(4) "从头梳理: 设置 → run → 结果 → 分析 整个过程" → lifecycle-based restructure
(5) "validate_run + 笔记 cross-references" → §1.2.9-10 / §3.2.9-14 / §4.4.b expansion
(6) **"continue audit — repo scripts + docs sweep"** → this round (probe scripts §2.5b / mechanistic Stage 1+2A §4.9 / experiment_matrix §3.1 / EVIDENCE_LAYER_AUDIT §1.4)

**Source docs**: ADVISOR_SYNC.md, advisor_sync_5_5_outcomes.md / followup.md,
preregistration.md, osf_lock_manifest.md, evaluator_change_protocol.md,
reeval_audit_protocol.md, 实验笔记 §107-§116, master_bug_catalog.md (~80 entries),
PAPER_STRATEGY_OPEN_QUESTIONS.md.

**Status**: 🟡 Active — populate as items verified. Block rerun until all 🔴 cleared.

---

## Lifecycle Overview — 5 Phases

```
┌─────────────────────────────────────────────────────────────────┐
│  Phase 1: 实验设置 (Setup)        [pre-launch, before any cell] │
│  └─ §1.1 Code & Bug Catalog                                     │
│  └─ §1.2 Config & Hyperparameters                               │
│  └─ §1.3 Pre-Registration & Witness                             │
│  └─ §1.4 Methodology Pre-Spec (statistical+robustness+stopping) │
│  └─ §1.5 Inter-Rater Reliability Prep                           │
│  └─ §1.6 Advisor Sync Outcomes & Open Questions                 │
│  └─ §1.7 Pre-experimental scope & data lineage 🆕               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 2: 实验 run 过程 (Run)       [during 16-cell + Stage 2B] │
│  └─ §2.1 Pre-Launch Sanity Gate                                 │
│  └─ §2.2 Provenance Capture per Cell                            │
│  └─ §2.3 Watchdog 6-Layer Auto-Clean Protocol                   │
│  └─ §2.4 Watchdog Operational Gates                             │
│  └─ §2.5 Mid-Run Automatic Safeguards                           │
│  └─ §2.5b Bug self-verification probes 🆕                       │
│  └─ §2.6 Cross-Cell Isolation                                   │
│  └─ §2.7 Failure-Mode Contingency / Resume Protocols            │
│  └─ §2.8 Pre-launch end-to-end smoke test 🆕                    │
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
│  └─ §4.8 Falsification & counterfactual robustness 🆕           │
│  └─ §4.9 Mechanistic Stage 2B/2C reproducibility 🆕             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  Phase 5: 出版与持续性 (Publication & Continuity) 🆕            │
│  └─ §5.1 Replication package contents catalog                   │
│  └─ §5.2 Operational continuity & hand-off plan                 │
│  └─ §5.3 Compliance & disclosure                                │
│  └─ §5.4 Long-term maintenance                                  │
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
| 1.2.9 | **SoM `max_marks=200`** (笔记 §94 reform — was 80, caused B1 Reddit DOM>SoM mode reversal) | ✅ §94 fix | `grep "max_marks" p79/experiment/som.py` should be 200 not 80 |
| 1.2.10 | **`current_viewport_only=True`** (paper §3 mode operational definition) | ✅ | `grep "current_viewport_only" p79/envs/vwa_wrapper.py` |

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
| 1.4.7b | **EVIDENCE_LAYER_AUDIT.md** — methodology + visualization gap registry (5 evidence types × 4 cross-axes) | ✅ exists | `docs/reference/EVIDENCE_LAYER_AUDIT.md` linked from paper_planning §3 |

### §1.4.2 Robustness pre-spec (paper §3 commit before lock)

| # | Item | Status | Notes |
|---|---|---|---|
| 1.4.8 | Non-visual subset robustness (43 VWA + 480 WA = 523 manually-audited) | ✅ in preregistration.md | Replaces deprecated visual_fp |
| 1.4.9 | Pre-Phase-A archive robustness (Appendix D) | ✅ in preregistration.md | Symmetric contamination disclosure |
| 1.4.10 | K_h1 / K_h3 threshold sensitivity (also report ±1) | 🔴 TBD | Show threshold gradient |
| 1.4.11 | Per-difficulty bucket analysis | 🔴 TBD | 3 difficulty terciles |
| 1.4.12 | Hold-out site validation (LOSO if advisor confirms) | 🟡 advisor email | `preregistration.md` mentions LOSO as alternative |
| 1.4.13 | Cross-machine reproducibility (DGX/A100/Myriad) | 🟡 | numerical_determinism_check post-rerun |

### §1.4.3 Stopping rules + missing data policy (NEW per §116.13)

| # | Item | Status | Notes |
|---|---|---|---|
| 1.4.14 | **Stopping rules** — when to halt a cell mid-run | 🔴 TBD | Pre-spec: e.g., if first 30 episodes have <5% SR or >50% eval errors, halt + investigate (do NOT continue contaminated cell to N=234) |
| 1.4.15 | **Imputation policy for crashed episodes** | 🔴 TBD | Pre-spec: episode with `error: env_crashed` excluded from N (paired bootstrap auto-handles); no imputation. Paper §3 cite |
| 1.4.16 | **Heterogeneity analysis pre-spec** — site differences in H1/H3 | 🔴 TBD | Pre-spec: report per-site SR + 95% CI; if >5pp site-difference, note as "site-modulated" rather than retract |
| 1.4.17 | **Bootstrap clustering decision** — by task or IID | 🔴 TBD | Pre-spec: cluster bootstrap by task (since same task observed 6× across modes); not IID |
| 1.4.18 | **Falsification criteria pre-spec** — what data outcome retracts hero claim | 🔴 TBD | Per paper_planning R5: <X cells pass H1+H3 → pivot to VWA bug paper; formalize threshold here |

## §1.5 Inter-Rater Reliability Prep (κ ≥ 0.7 targets)

| # | Item | Status | Target |
|---|---|---|---|
| 1.5.1 | FP labeling reliability — 30-task pilot, 2 raters | 🔴 TBD pre-rerun pilot | Cohen κ ≥ 0.7 per preregistration.md |
| 1.5.2 | Failure-mode 5-bucket rubric reliability | 🔴 TBD | κ ≥ 0.7 target |
| 1.5.3 | Codex-as-rater calibration spot-check | 🔴 TBD | Disagreement >30% triggers prompt revision |
| 1.5.4 | Visual subset audit | ✅ exists `docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py` | Used as Appendix D |

## §1.7 Pre-experimental scope & data lineage (NEW per §116.13 deep audit)

| # | Item | Status | Verify |
|---|---|---|---|
| 1.7.1 | **VWA submodule git SHA pin** at lock moment | 🔴 TBD | `cd external/visualwebarena && git rev-parse HEAD` recorded in osf_lock_manifest.md |
| 1.7.2 | **Playwright Chromium version pin** | 🔴 TBD | `python3 -c "import playwright; print(playwright.__version__)"` + `playwright --version` recorded |
| 1.7.3 | **Tokenizer SHA verification** matches model HF revision | 🔴 TBD | `cat ~/.cache/huggingface/.../tokenizer_config.json` SHA |
| 1.7.4 | **Reference image hash registry** — per-task ref image SHA-256 stable | 🔴 TBD | `find external/visualwebarena/config_files/vwa -name "*.json" -exec jq -r '.image' {} \;` then SHA all referenced images |
| 1.7.5 | **Task pool freezing explicit** — task IDs locked per cell | 🟡 implicit | Add `task_id_pool_sha256` in run_manifest.yaml per cell (sorted task IDs hashed) |
| 1.7.6 | **License compliance explicit** — VWA Apache 2.0 / Qwen3-VL license / dependencies | 🔴 TBD | Add to paper §3 footnote + `LICENSE` file at repo root |
| 1.7.7 | **Conflicts of interest** / IRB statement (likely N/A but explicit) | 🔴 TBD | Paper §1 footnote: "no IRB needed (synthetic web tasks); no COI" |
| 1.7.8 | **Cross-paper data lineage map** — phantom 16-cell ↔ mechanistic Stage 2B/2C ↔ VWA bug catalog | 🟡 partial | Document data flow in osf_lock_manifest.md §1.5 |
| 1.7.9 | **`COMPUTE_INFRASTRUCTURE.md`** — live infra landscape (UCL Condense A100 / Myriad HPC / DGX Spark, SSH paths, allocation sources) | ✅ exists | `docs/reference/COMPUTE_INFRASTRUCTURE.md` |
| 1.7.10 | **`condition_map.md`** — condition→path mapping (run_id patterns, VWA/WA sites, fast inference rules) | ✅ exists | `docs/reference/condition_map.md` |
| 1.7.11 | **Obsidian Bases data layer** (`cells.base` / `codex.base` / `issues.base` / `status.base` at vault root) — frontmatter source-of-truth for run state | ✅ deployed | 4 base files at `docs/` |
| 1.7.12 | **`_status/section{1..8}*.md`** frontmatter — per-section paper prose progress + word-count + blocker registry | ✅ deployed | `docs/checkpoints/_status/section/*.md` |
| 1.7.13 | **笔记 §99 scripts restructure lineage** — `scripts/queues/` + `scripts/maintenance/` + `scripts/analysis/` separation rationale | ✅ documented | 笔记 §99 (rationale: queue ≠ maintenance ≠ analysis script roles) |

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
| 2.1.6 | **`bash scripts/preflight_v2.sh`** — comprehensive sanity (docker daemon / site ports strict-or-warn / CUDA / evaluator imports / VWA configs) | 1 (WARN if site down) | exit 0 = launch OK |
| 2.1.7 | **`scripts/queues/queue_16cell_paper_grade.sh`** — master orchestrator (B0×{cls,red}×3 + B1×{cls,red}×3 + shop×4) | 🟡 lock at advisor email | replaces ad-hoc `queue_phantom_pair` chains for the 16-cell rerun |
| 2.1.8 | **`scripts/setup/a100_self_host_vwa.sh`** — A100 self-host VWA runbook (replaces Tailscale↔quark dependency) | 🟡 needs A100 SSH | one-time ~1-2h setup; eliminates Tailscale single-point failure |
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
| 2.3.7 | **GLM auto-analysis pipeline** (笔记 §6) — daily digest generation + failure taxonomy | ✅ deployed | cron `glm-update-cells` + `glm-refresh-playbook`; PLAYBOOK §1+§2 GLM-managed |

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

### §2.5b Bug self-verification probes (paper-grade scaffold safety chain)

> Every CONFIRMED catalog entry has a **probe script** that re-validates the bug exists before/after a fix. Reviewer-defensible: "we don't just claim B-XX is fixed, we re-run the probe and show the symptom is gone." Probes are paper §4 limitations cite-anchors.

| # | Probe | Bug coverage | Status |
|---|---|---|---|
| 2.5.9 | `probe_b01_b13_self_verify.py` | B-01 TYPE 100% scaffold + B-13 action_fail_page_changed | ✅ exists; re-run pre-rerun on smoke cell |
| 2.5.10 | `probe_b08_b06_self_replay.py` | B-08/B-06 self-replay mechanistic validation (mark hallucination + scroll loop) | ✅ exists |
| 2.5.11 | `probe_b37_api_determinism.py` | B-37 RNG + auth_refresh determinism (T=0 + seed reproducibility) | ✅ exists |
| 2.5.12 | `probe_som_occlusion.py` | SoM occlusion consistency (text visibility ↔ rendered SoM marks) | ✅ exists |
| 2.5.13 | `probe_tier10_dispatch_target.py` | Phase A Tier10 locator-route dispatch verification | ✅ exists; commit `3c15cd7` |
| 2.5.14 | `compare_pilot_t0_vs_paper_grade.py` | B-37 pilot gate: T=0 pilot SR ↔ paper-grade baseline (no regression) | ✅ exists; pre-rerun gate |
| 2.5.15 | **Pre-rerun probe re-run protocol** — fire all 6 probes on smoke cell, all exit 0 before launching 16-cell | 🔴 TBD add to launch protocol | scripts above + smoke 2-task cell |

## §2.6 Cross-Cell Isolation

| # | Item | Status | Verify |
|---|---|---|---|
| 2.6.1 | Each cell unique RUN_ID | ✅ | `<baseline>_<mode>_<site>_YYYYMMDD` pattern |
| 2.6.2 | Cell directory isolation (no shared episodes/) | ✅ | run_dir convention |
| 2.6.3 | RESET_BEFORE=1 between cells of same site | ✅ enforced via 2.1.1 | hard rule |
| 2.6.4 | Auth state regeneration per-site per-cell | ✅ B-66 | per-site files |
| 2.6.5 | Watchdog state.json reset between cells | ✅ B-54i | restart handles |
| 2.6.6 | env_snapshot.json dumped at each cell launch | ✅ | run_experiment.py hook |

## §2.8 Pre-launch end-to-end smoke test (NEW per §116.13)

**Run before launching all 16 cells** — verify full pipeline works on a minimal subset.

| # | Item | Status | Verify |
|---|---|---|---|
| 2.8.1 | **End-to-end smoke**: 1 cell × 2 tasks × all 6 modes (DOM/SoM/Vision/P-text/P-prompt/P-SoM) | 🔴 TBD | `make smoke` or custom 2-task launch; verify all 12 (2×6) episodes produce summary.json with non-error status |
| 2.8.2 | **ntfy delivery test** — auth fail trigger → alert reaches phone | 🔴 TBD | Manual: kill `.auth/cls_state.json`; observe ntfy notification arrives |
| 2.8.3 | **Watchdog 6-layer artificial trigger** — fake auth fail → verify cleanup→resume on test data | 🔴 TBD | Add `scripts/maintenance/watchdog_smoke_test.sh` |
| 2.8.4 | **Disk I/O speed baseline** for target machine | 🔴 TBD | `dd if=/dev/zero of=test bs=1M count=1000 conv=fdatasync` on Scratch — paper §3 disclose if Lustre |
| 2.8.5 | **Auth refresh subprocess smoke** — invoke `auth_refresh.py` standalone, verify completes <30s | 🔴 TBD | `python3 -m p79.utils.auth_refresh refresh classifieds` |
| 2.8.6 | **GPU forward pass smoke** — load model + 1 forward pass on each target machine | 🔴 TBD | A100/Myriad/DGX: `python3 -c "import torch; from p79.agents.qwen3vl_agent import Qwen3VLAgent; ..."` |
| 2.8.7 | **`scripts/smoke_test_vwa.py`** — single VWA episode × 6 mode variants | ✅ exists | run on smoke cell pre-launch |
| 2.8.8 | **`scripts/setup/smoke_login.sh`** — Myriad SSH + compute node connectivity smoke | ✅ exists | run pre-Myriad qsub |

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
| 2.7.9 | **Quark host watchdog** — VWA Docker on quark, alert if quark off / Docker daemon dead | 🔴 TBD | Cron on DGX or laptop: `curl -sI http://100.95.81.103:9980` every 10 min, ntfy on fail |
| 2.7.10 | **Backup restore protocol** — Scratch corrupt recovery | 🔴 TBD | Document: archive subset is in git (16.5MB safe); HF model cache re-downloadable; experiment results need DGX backup if A100 Scratch lost |

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
| 3.1.6 | **`docs/analysis/experiment_matrix.md`** — Phase 1 baseline progress tracker (B0/B1 × 3 sites × 3 modes, raw/adjusted SR, post-§97 rederive version) | check after each cell completes |
| 3.1.7 | **`docs/analysis/B1_capability_profile.md`** — B1 (Qwen3-VL-4B) snapshot: 6-cell SR + cost/latency/energy + failure patterns vs B0 | update post-rerun |

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
| 3.2.9 | **`validate_run.py` 27-check suite per cell** (笔记 §91 + §93 expanded) | ✅ exists | `python3 scripts/analysis/validate_run.py --run-dir <run> --strict` post-cell. **10 check groups, 27 checks**: G1 file existence (C01-02) / G2 structure (C03-06) / G3 coverage (C07-08) / G4 episode integrity (C09-11) / G5 step integrity (C12-15) / G6 scaffold safety (C16-18) / G7 artifact integrity (C19-20) / G8 analysis freshness (C21-22) / **G9 Temporal Analysis (C23-25)** — SR-over-time degradation, auth drift, reset contamination / **G10 Data Consistency (C26-27)** — summary.steps vs JSONL line count, zero-cost episode detection. Exit 0/1/2 |
| 3.2.10 | **Cross-baseline comparison check (C18)** — `validate_run.py --compare-dir <other_baseline_run>` | ✅ exists | Run when comparing B0 vs B1 paired data integrity |
| 3.2.11 | `validate_run.py --strict` exit code = 0 BEFORE moving cell to grade=paper-grade | 🟡 add to launch protocol | Per-cell pipeline gate |
| 3.2.12 | **Watchdog cross-run analysis auto-trigger** (笔记 §98) — `compare_b0_b1` + `aggregate_cross_site` fire automatically when sibling/cross-site conditions complete | ✅ deployed | `experiment_watchdog.py::_run_cross_run_analysis` |
| 3.2.13 | **Intent feature + cost attribution columns** (笔记 §93) — `analyze_reason_diagnostics.py` outputs +16 columns (10 `intent_has_*` booleans + 6 cost cols) + 7 plots | ✅ deployed | invoked by `make reason-diag` per cell |
| 3.2.14 | **Waste breakdown** (`metrics.py::compute_waste_breakdown`) — no_op / page_unchanged / total / wasted cost dimensions per cell | ✅ §93 | aggregated post-rerun |
| 3.2.15 | **`scripts/analysis/b0_vision_coordinate_errors.py`** — per-run B0 vision coordinate error detector (paper §4 B0 vision asymmetry) | ✅ exists | run post-cell on B0 vision cells |

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
| 4.1.6 | **`scripts/analysis/aggregate_phantom_meta.py`** — random-effect pooled meta-analysis with drop-one forest plots (paper §3 Figure) | ✅ exists | post-rerun |
| 4.1.7 | **`scripts/analysis/collect_analysis_summary.py`** — run-level metadata collector + efficiency dimension evidence aggregator | ✅ exists | post-rerun |
| 4.1.8 | **`scripts/analysis/reeval_phase1.py`** — re-evaluation of Phase 1 with locked FP filter / na_fp / eval_fp rules | ✅ exists | post-rerun (or post advisor email if FP rule changes) |
| 4.1.9 | **笔记 §109 dual-track reframe** — methodology pivot to 9-cell taxonomy + 3-round DR | ✅ documented | paper_planning §3 + paper §1 hook |

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
| 4.2.8 | **`analyze_cross_representation.py`** — cross-rep behavior diagnostic (DOM↔SoM↔Vision token attention) | ✅ exists | post-rerun |
| 4.2.9 | **`analyze_search_over_browse.py`** — search vs browse behavior diagnostic (Reddit) | ✅ exists | post-rerun |
| 4.2.10 | **`analyze_comment_selflink_loop.py`** + **`analyze_reddit_selflink_cycle.py`** — Reddit cycle / self-link diagnostics | ✅ exists | post-rerun on red cells |
| 4.2.11 | **`analyze_noninteractive_click_earlystop.py`** — non-interactive click + early-stop pattern (paper §4) | ✅ exists | post-rerun |
| 4.2.12 | **`analyze_confidence_calibration.py`** — model confidence calibration diagnostic (logprob/entropy/margin) | ✅ exists | feeds Phase 2 router signal selection (per `routing_signals.md` lit) |

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

## §4.4.b Failure-mode pattern detection (`diag_pattern_match.py` + analysis pipeline)

**Source**: 笔记 §92 (P1-P14 rule scripting), §93 (analysis pipeline 27-check + intent + cost), §94 finding (max_marks reversal), §97 (cross_representation audit), §98 (watchdog cross-run automation).

| # | Item | Status | Verify |
|---|---|---|---|
| 4.4.b.1 | `diag_pattern_match.py` P1-P14 rules implemented (P9 deferred) | ✅ | `python3 scripts/analysis/diag_pattern_match.py --run-dir <run>` |
| 4.4.b.2 | Per-cell failure-mode breakdown | 🟡 TBD post-rerun | Add to `make analysis` pipeline |
| 4.4.b.3 | Cross-cell pattern comparison (e.g. P14 URL self-loop %) | 🟡 | Aggregate across 16 cells |
| 4.4.b.4 | Pattern-rule κ for 5-bucket failure-mode mapping (per §1.5.2) | 🔴 TBD | Spot-check P-rule output vs human label |
| 4.4.b.5 | **Analysis pipeline 4-dimension Evidence Framework** (笔记 §106) | ✅ pre-spec'd | `scripts/analysis/aggregate_phantom_lift.py` (Outcome 0c/0d) + `aggregate_routing_auroc.py` (0g) + `axis_effect_size.py` (1a/1b) + `axis1_microbehavior.py` (2a-2e) + `aggregate_cross_site.py` (3a-3c) + `figures/` per-outcome scripts |
| 4.4.b.6 | **`layered_status.py`** — live evidence layer status | ✅ exists | `docs/analysis/layered_evidence_status.md` snapshot |
| 4.4.b.7 | `compare_b0_b1.py` + `aggregate_cross_site.py` triggered automatically post-condition | ✅ §98 | watchdog `_run_cross_run_analysis` |

## §4.4.c Reference framework integration (笔记 §106 4-dim Evidence Framework)

**Source**: 笔记 §106 (4-dimension Evidence Framework), §108 (Phantom space refinement, evidence/explanation separation).

| # | Item | Status | Verify |
|---|---|---|---|
| 4.4.c.1 | 4-dimension Evidence Framework applied to paper §1+§4 organization | ✅ pre-spec'd | paper_planning §3 + paper drafts |
| 4.4.c.2 | Phantom space evidence/explanation separation (Zoom 1-4) | ✅ pre-spec'd | paper_planning §1 + phantom_space.canvas |
| 4.4.c.3 | §100 SoM screenshot OCR ground truth probe — used for §5 mechanism evidence | ✅ data exists | `docs/analysis/.../som_ocr_probe.md` |
| 4.4.c.4 | §103 4-mode routing arm finding (paper §1 hook) | ✅ pre-spec'd | paper §1 narrative |
| 4.4.c.5 | §94 max_marks reversal as B1 capability-modulated finding (paper §7) | ✅ documented | section 4 limitations or section 7 cross-capability |

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

## §4.8 Falsification & counterfactual robustness (NEW per §116.13)

| # | Item | Status | Notes |
|---|---|---|---|
| 4.8.1 | **K_h1 ±1 sensitivity** — pass at K=12 but fail at K=13: paper hook intact? | 🔴 TBD | Pre-spec: K=12 gates HERO; K-12 to K-15 gradient reported transparently |
| 4.8.2 | **Counterfactual cell removal** — removing any one cell changes decision? | 🔴 TBD | Re-run preregistration_decision_test with each cell excluded; report N_unstable |
| 4.8.3 | **Outlier task spot-check** — top 5 / bottom 5 task-level effects, manual review | 🔴 TBD | Verify no single task drives the result |
| 4.8.4 | **Falsification hierarchy** (per paper_planning §5 R-rules R1-R5) | 🟡 partial | Formalize threshold: if H1<10/16 AND H3<8/16 → R5 retract+pivot to VWA bug paper |

## §4.9 Mechanistic Stage 2B/2C-specific reproducibility (NEW per §116.13)

| # | Item | Status | Verify |
|---|---|---|---|
| 4.9.1 | Same input → same L11 hidden state (within 1e-3) cross-machine | 🟡 needs A100/Myriad SSH | `numerical_determinism_check.py compare` on stage2b_curated subset |
| 4.9.2 | Hook fire_count protocol verified — first-forward-only patching | ✅ commit `1304f59` | activation_patching.py `_fire_count_tracker` |
| 4.9.3 | Layer indexing — `model.config.num_hidden_layers` matches L0-L35 used in script | 🟡 | Verify on each machine: Qwen3-VL-4B has 36 layers |
| 4.9.4 | Token alignment — input/output tokens correspondence per task | 🟡 | Spot-check N=5 task pairs, output length sanity |
| 4.9.5 | Stage 2B post-rerun spot-check — re-run 3 tasks, verify L11 finding stable | 🔴 TBD | After full Stage 2B finishes, re-run task 0/100/233 to confirm reproducibility |
| 4.9.6 | **`scripts/mechanistic/run_stage1_pilot.py`** — Stage 1 per-layer linear probe baseline (predates Stage 2A patching) | ✅ exists | establishes which layers are causally relevant before patching |
| 4.9.7 | **`scripts/mechanistic/run_stage2_patching_pilot.py`** — Stage 2A activation patching pilot (preceded Stage 2B continuation) | ✅ exists | mirage layer hypothesis generation |
| 4.9.8 | **`scripts/mechanistic/curate_mirage_tasks.py`** — paper §5 candidate scoring (24 strong + 11 reverse mirage) | ✅ commit `4425fa6` | curated subset committed |
| 4.9.9 | **`scripts/mechanistic/extract_archive_subset.py`** — archive subset extraction → 16.5MB committed to git | ✅ commit `cd50c34` | replication artifact (paper §5 cite) |
| 4.9.10 | **`scripts/analysis/mechanism_per_task.py`** — Section 5 per-task mechanism evidence aggregator | ✅ exists | feeds paper §5 figures |
| 4.9.11 | **`docs/reference/PHANTOM_SOM_CODE_TOUR.md`** — mechanism extraction code walkthrough (activation patching / linear probe layer assignments) | ✅ exists | paper §5 reproducibility appendix |
| 4.9.12 | **笔记 §111 Stage 1+2 mechanistic pilot** — initial L11 mirage hypothesis derivation | ✅ documented | predates Stage 2B; cite for theory provenance |
| 4.9.13 | **笔记 §113 Mirage task curation** — 24 strong + 11 reverse selection rationale | ✅ documented | paper §5 candidate selection methodology |

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

# Phase 5 — 出版与持续性 (Publication & Continuity) — NEW per §116.13

> Often-overlooked: paper-grade also covers **what gets released to public** + **how the work survives author availability gaps**.

## §5.1 Replication package contents catalog

| # | Item | Goes into | Status |
|---|---|---|---|
| 5.1.1 | preregistration.md (locked version, post advisor email) | OSF DOI deposit | 🟡 awaiting email |
| 5.1.2 | paper_drafts_locked/ (immutable copy at lock moment) | OSF + GitHub tag | 🔴 TBD post-rerun |
| 5.1.3 | paper.bib (57 entries) | OSF + GitHub | ✅ |
| 5.1.4 | run_manifest.yaml (post-rerun, grade=paper-grade) | OSF + GitHub | 🟡 post-rerun |
| 5.1.5 | env_snapshot.json (per host: DGX + A100 + Myriad) | OSF + GitHub `results/provenance/` | DGX ✅ / others 🟡 |
| 5.1.6 | snapshot_vwa.json (per VWA host) | OSF + GitHub | DGX ✅ / others 🟡 |
| 5.1.7 | numerical_determinism_check output | OSF + GitHub | 🔴 TBD post-rerun |
| 5.1.8 | master_bug_catalog.md (~80 entries) | OSF + GitHub | ✅ |
| 5.1.9 | section4_limitations_disclosure.md | Paper §4 + OSF | ✅ |
| 5.1.10 | mechanistic archive_subset_b1_cls/ (16.5MB) | GitHub (already committed); link from paper §5 | ✅ |
| 5.1.11 | Stage 2B/2C results — full per-task .json + curves.png + run_manifest.json | OSF + GitHub `results/mechanistic/` | 🟡 post-Myriad |
| 5.1.12 | Code release — current master pinned at `preregistration-locked` tag | GitHub release | 🟡 post-lock |
| 5.1.13 | License files (Apache 2.0 root + per-dependency) | GitHub root | 🔴 TBD verify |
| 5.1.14 | README for replication (step-by-step from clone to figures) | Paper Appendix + GitHub | 🔴 TBD |
| 5.1.15 | **`scripts/maintenance/rsync_results_to_hub.sh`** + `rsync_results_from_hub.sh` — A100↔central hub artifact sync | OSF deposit prep | ✅ exists |
| 5.1.16 | **`scripts/generate_gallery.py`** + `scripts/maintenance/refresh_gallery.sh` — HTML annotated screenshot gallery | Paper appendix figures + OSF | ✅ exists |
| 5.1.17 | **`docs/reference/analysis_templates.md`** — `B{x}_{Mode}_digest.md` template for replication digest | OSF + GitHub | ✅ exists |
| 5.1.18 | **`docs/literature/literature_insights.md`** + `routing_signals.md` + `logprob_signals.md` + `phantom_som.md` — 26-paper synthesis + signal calibration lit | Paper §1+§2 + OSF | ✅ exists |

## §5.2 Operational continuity & hand-off plan

| # | Scenario | Status | Plan |
|---|---|---|---|
| 5.2.1 | **48h rerun monitoring** — who responds at 3 AM if cell halts | 🔴 TBD | Pre-rerun: define on-call schedule (you full coverage? Or family member as escalation contact?) |
| 5.2.2 | **You unavailable >12h** (exam / illness / network out) — can advisor act? | 🔴 TBD | Document minimum-viable action list ("if PID dies, 2 commands to restart"); share with advisor or trusted peer |
| 5.2.3 | **Hand-off documentation** — if someone else needs to continue, where do they start | 🔴 TBD | Add `docs/HANDOFF_PLAYBOOK.md` — top-3 immediate-action commands per emergency type |
| 5.2.4 | **Calendar integration** — exam dates 5/12-6/1 vs rerun launch timing | 🟡 partial | next_steps.md mentions; ensure rerun launches AFTER exams or has reduced monitoring expectation |
| 5.2.5 | **PLAYBOOK §1+§2 GLM cron health** — what if GLM API itself goes down | 🟡 | Cron + ntfy fail alerts already in place per §116 audit |

## §5.3 Compliance & disclosure

| # | Item | Status | Where |
|---|---|---|---|
| 5.3.1 | License compliance — VWA Apache 2.0 / Qwen3-VL Apache 2.0 / transformers Apache 2.0 | 🔴 TBD verify | Paper §3 footnote + repo `LICENSE` |
| 5.3.2 | Conflicts of interest statement | 🔴 TBD | Paper §1 footnote (likely "none — student MSc work") |
| 5.3.3 | IRB / ethics review statement | 🔴 TBD | Paper §3 ("no IRB needed — synthetic web tasks, no human subjects") |
| 5.3.4 | Data sharing policy (what gets released, what doesn't) | 🔴 TBD | Section 9 / Appendix: code public, run_manifest+env_snapshots public, .auth/ private (gitignored) |
| 5.3.5 | Compute cost transparency | 🟡 partial | Paper §3 / §8 — total GPU-hours + USD + carbon |
| 5.3.6 | API keys / credentials redacted from any released artifacts | ✅ | `.auth/` gitignored; verify env_snapshot doesn't expose tokens |

## §5.4 Long-term maintenance

| # | Item | Status | Notes |
|---|---|---|---|
| 5.4.1 | Tag freezing protocol — `preregistration-locked` git tag at OSF DOI mint | 🟡 | osf_lock_manifest.md §3 step 6 |
| 5.4.2 | Branch policy — main = clean post-lock, `post-paper` branch for revisions | 🔴 TBD | Decide policy now |
| 5.4.3 | Paper §3 reproducibility statement (1 paragraph: "code at <SHA>, data at <DOI>, env at...") | 🔴 TBD post-rerun | Boilerplate template ready in paper drafts |
| 5.4.4 | Bug-fix discipline going forward — Protocol A T0/T1/T2 classification | ✅ | evaluator_change_protocol.md |
| 5.4.5 | Paper revision policy — if reviewer asks for new analysis, classify as preregistered or post-hoc explicit | 🔴 TBD | Pre-spec: any new analysis post-DOI is post-hoc, must be flagged in revision response |

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
- `docs/reference/EVIDENCE_LAYER_AUDIT.md` (5 evidence types × 4 axes)
- `docs/reference/COMPUTE_INFRASTRUCTURE.md` (live infra landscape)
- `docs/reference/condition_map.md` (condition→path mapping)
- `docs/reference/PHANTOM_SOM_CODE_TOUR.md` (mechanism extraction code walkthrough)
- `docs/reference/analysis_templates.md` (digest template)
- `docs/analysis/experiment_matrix.md` (Phase 1 progress) / `B1_capability_profile.md`
- `docs/literature/literature_insights.md` / `routing_signals.md` / `logprob_signals.md` / `phantom_som.md`
- `docs/{cells,codex,issues,status}.base` (Obsidian Bases data layer)
- `docs/checkpoints/_status/{section,cells,codex,issues}/*.md` (frontmatter source-of-truth)
- 实验笔记 §6 (watchdog/GLM pipeline) / §99 (scripts restructure lineage) / §107 (Phase A) / §109 (dual-track reframe) / §110 (5/5 sync) / §111 (Stage 1+2 mechanistic pilot) / §113 (mirage curation) / §114 (provenance) / §115 (Protocol A+B) / §116 (audit + restructure)

---

**Last restructure**: 2026-05-08, 笔记 §116.12 — lifecycle-based reorganization (4 phases × 18 sections × ~150 gate items).

**Last expansion**: 2026-05-08, 笔记 §116.15 — repo-wide scripts/docs/笔记 sweep (5 phases × 25 sections × ~245 gate items): §1.4.7b EVIDENCE_LAYER_AUDIT / §1.7.9-13 infrastructure & data layer / §2.1.6-8 preflight + 16-cell orchestrator + A100 self-host / §2.3.7 GLM pipeline / **§2.5b 7-probe bug self-verification chain** / §2.8.7-8 smoke scripts / §3.1.6-7 progress trackers / §3.2.15 B0 vision coord errors / §4.1.6-9 meta-analysis + reeval + dual-track reframe / §4.2.8-12 5 behavior diagnostics / **§4.9.6-13 Stage 1+2A mechanistic pipeline** / §5.1.15-18 replication artifacts.
