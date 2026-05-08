# Pre-Rerun Audit Checklist — 16-Cell Phantom Routing Rerun

**Purpose**: Comprehensive paper-grade gate review before launching 16-cell rerun
on A100 (post advisor email + SSH cert). Designed to catch spec/code drift,
provenance gaps, and operational issues BEFORE 48h of compute is spent on
contaminated data.

**Triggered by**: User audit prompt 2026-05-08 — "整体 audit before rerun".
**Source docs**: ADVISOR_SYNC.md, advisor_sync_5_5_outcomes.md / followup.md,
preregistration.md, osf_lock_manifest.md, evaluator_change_protocol.md,
reeval_audit_protocol.md, 实验笔记 §107-§116, master_bug_catalog.md,
PAPER_STRATEGY_OPEN_QUESTIONS.md.

**Status**: 🟡 Active — populate as items verified. Block rerun until all 🔴 cleared.

---

## §A — Code-level paper-grade gates

| # | Item | Status | Owner | Verify command |
|---|---|---|---|---|
| A1 | **Early-stop disabled** (advisor 5/5 Option A cancel) | ✅ FIXED commit `<TBD>` | code | `grep -c "_early_stop_enabled" p79/experiment/runner/main.py` ≥ 4 |
| A2 | Phase A 4-cluster fix active (`3c15cd7`) — dispatch / cycle / RNG / page_changed | ✅ commit ≥ `3c15cd7` | code | `git log --oneline 3c15cd7..HEAD --stat \| head` |
| A3 | HF revision pinned in `qwen3vl_agent.py` + `extract_hidden_states.py` | ✅ commit `3b25438` | code | `grep "ebb281ec" p79/agents/*.py p79/mechanistic/*.py` |
| A4 | Evaluator code SHA captured in env_snapshot | ✅ commit `1304f59`+`1fefd39` | code | `python3 scripts/provenance/snapshot_env.py /tmp/test.json && grep evaluator_code /tmp/test.json` |
| A5 | FP filter primary `na_fp + eval_fp` (visual_fp removed §95) | ✅ commit `1fefd39` | code | `grep "fp_reason" p79/experiment/analysis.py` |
| A6 | rederive_metadata audit trail enabled | ✅ commit `1fefd39` | code | `grep "rederive_metadata" scripts/maintenance/rederive_episode_summary.py` |
| A7 | Watchdog auto-clean 6-layer protocol | ✅ stable | infra | `pgrep -f experiment_watchdog` (during rerun) |

## §B — Config alignment with preregistration.md

| # | Item | Status | Verify |
|---|---|---|---|
| B1 | 16-cell scope per `preregistration.md §4` | ✅ | `grep "N_cells" preregistration.md` = 16 |
| B2 | K_h1=12 / K_h3=11 / TOST δ=1.0pp values present (no TBD) | 🟡 pending advisor email | `grep -c "TBD" preregistration.md` after email |
| B3 | Mode operational definitions (6 modes) stipulative | ✅ | `preregistration.md` line 199 |
| B4 | Per-site YAML configs not overriding `max_steps` artificially | 🔴 TODO verify | `grep -l "max_steps" configs/exp_v2_*.yaml` |
| B5 | RNG seeds explicit + deterministic per cell | 🔴 TODO verify | `grep "seed" configs/*.yaml` |
| B6 | run_manifest.yaml grades reflect rerun plan | ✅ archived | `grep "grade:" results/phantom_paper/run_manifest.yaml \| sort \| uniq -c` |

## §C — Provenance chain (笔记 §114 + §115)

| # | Item | Status | Verify |
|---|---|---|---|
| C1 | env_snapshot.py works on each target machine | DGX ✅ / A100 🟡 / Myriad 🟡 | `ls results/provenance/env_*_baseline.json` |
| C2 | snapshot_vwa.sh works (DGX baseline + A100 self-host) | DGX ✅ / A100 🟡 | `ls results/provenance/vwa_*.json` |
| C3 | numerical_determinism cross-machine check ready | 🟡 needs A100 SSH | `scripts/provenance/numerical_determinism_check.py` exists |
| C4 | sitecustomize.py shim (Myriad-only) committed | ✅ | `git show:scripts/setup/myriad_bootstrap.sh \| grep sitecustomize` |
| C5 | constraints.txt patterns (urllib3<2 / numpy<2) | ✅ Myriad bootstrap | `bootstrap script writes` |

## §D — Pre-registration witness state

| # | Item | Pending? | Action when ready |
|---|---|---|---|
| D1 | Advisor email reply with K_h1/K_h3/TOST δ confirmation | 🟡 | Update `preregistration.md` § Decision log + flip `status: draft → locked` |
| D2 | `preregistration.md` `registered_at` + `registered_git_sha` | 🟡 | Fill at lock moment |
| D3 | `preregistration.md` `witnessed_by` advisor name + date | 🟡 | Fill at lock moment |
| D4 | OSF DOI minted + paper §1 footnote | 🟡 | 8-step `osf_lock_manifest.md` after email |
| D5 | git tag `preregistration-locked` | 🟡 | `git tag -a preregistration-locked` at lock moment |

## §E — Open questions resolution (PAPER_STRATEGY_OPEN_QUESTIONS.md)

| Q | Title | Status | Notes |
|---|---|---|---|
| Q1 🔴 | Early-stop bias on micro metrics | ✅ A1 cancel locked + code fixed | This audit |
| Q2 🟡 | B0 pre/post Phase A sampling asymmetry | ✅ handled by 16-cell rerun (post-fix only) | preregistration.md cell inclusion |
| Q3 🟢 | Environment non-determinism | ✅ accepted via `snapshot_vwa.sh` fingerprint | Paper §3 disclose |
| Q4 ❌ RETRACTED | Cross-site SR comparability | — | Already handled per-site bootstrap |
| Q5 ❌ RETRACTED | FP filter asymmetry | — | Feature not bug per §95 |
| Q6 🟢 | Diamond completion partial | 🟡 | depends on rerun output |
| Q7 🟡 | B0 vs B1 cross-baseline sampling regime | 🟡 | discuss in paper §3 limitations |
| Q8 🟢 | Drop-one oracle observed-mode-set dependence | ✅ documented | preregistration.md routing signal universe |
| Q9 🟢 | Routing AUROC in-sample evaluation | 🟡 | depends on H7-H8 router family |

## §F — Bug catalog status (master_bug_catalog.md, 37 entries)

| Tier | Count | Action |
|---|---|---|
| ✅ CONFIRMED | TBD | Verify root-cause traced + in fix scope |
| ⚠️ DISPUTED | TBD | Re-replay or downgrade if probe evidence weak |
| ❌ NOT_A_BUG | 3 (B-12 / B-13 / B-14) | No action |
| 🔄 UNVERIFIED | TBD | Decide retain / downgrade |
| 🛠️ FIXED | 1 (B-10 §105 Magento) + Phase A 4-cluster | Verify post-fix in current code |

**Pre-rerun rule**: Any 🛠️ FIXED bug in catalog must have its fix in code at HEAD.
Any ⚠️ DISPUTED or 🔄 UNVERIFIED MUST be triaged to either ✅ CONFIRMED or ❌ NOT_A_BUG before lock.

## §G — Advisor sync 5/5 outcomes (advisor_sync_5_5_outcomes.md §A)

| # | Item | Status |
|---|---|---|
| A.1 | Early-stop A 全 cancel | ✅ code fixed this audit |
| A.2 | Manifest 全 archive + 16-cell rerun | ✅ run_manifest.yaml grade=archived |
| A.3 | Paper 拆开发 (split direction) | 🟡 exact count Q1 advisor email |
| A.4 | VWA bug → ACL position paper | ✅ accepted, out of immediate scope |
| A.5 | Routing benchmark 独立成文 | ✅ accepted |
| A.6 | Mechanistic interpretability publication-worthy | ✅ Stage 2B/2C running on Myriad |
| A.7 | Workshop submission 节奏 | ✅ accepted |
| A.8 | Compute paths (A100 / Myriad / advisor 5090) | A100 🟡 SSH cert / Myriad ✅ |
| A.9 | Pre-reg witness mechanism (git+email+OSF) | 🟡 advisor email |
| A.10 | Environment 3-layer framework | ✅ accepted |

## §H — Operational gates (rerun-time)

| # | Item | Verify |
|---|---|---|
| H1 | No conflicting active runs (B0 XOR B1 same site) | `pgrep -af "run_experiment.*<site>"` empty pre-launch |
| H2 | `RESET_BEFORE=1` enforced via queue scripts | `grep RESET_BEFORE scripts/queues/queue_*.sh` |
| H3 | Auth files fresh per site | `ls -la .auth/*.session.json` not stale |
| H4 | Disk space ~50GB/cell × 16 = 800GB+ on Scratch | `df -h ~/Scratch` |
| H5 | Watchdog running | `pgrep -f experiment_watchdog` |
| H6 | NTFY notification setup | curl smoke test |
| H7 | env_snapshot auto-dumps at run start (笔记 §114 hook) | inspect `<run_dir>/env_snapshot.json` after first cell |

## §I — Output schema gates

| # | Item | Verify |
|---|---|---|
| I1 | step JSONL schema v2 catalog (笔记 §97) | `p79/experiment/schema_migrations/` |
| I2 | episode_summary `adjusted_success` + `fp_reason` fields populate | post-cell `make rederive` then `head episodes/*.json` |
| I3 | condition_summary_v2.json aggregation | post-cell `cat condition_summary_v2.json` |
| I4 | Logs format consistent (JSON-L lines) | `head <run_dir>/log.jsonl` |
| I5 | env_snapshot.json `evaluator_code.combined_sha256` matches lock SHA | `jq .evaluator_code.combined_sha256 env_snapshot.json` |

## §J — Decision flow (use this before launching)

> Updated 2026-05-08 (笔记 §116.10): now includes §M-§T paper-grade scientific gates.

```
Pre-launch checklist run:
  All §A items ✅ → continue
  All §B items ✅ (B2/B4/B5 may be 🟡 pending email/verify) → continue
  All §C items ✅ for target machine → continue
  §D items 🟡 OK pre-rerun (locks at OSF DOI mint, post-rerun analysis) → continue
  All §E Q1-Q9 with green/locked status → continue
  §F bug catalog cleaned (no UNVERIFIED outstanding) → continue
  §G advisor 5/5 outcomes A.1-A.10 acted on → continue
  §H operational gates ✅ at launch time → LAUNCH

Mid-rerun monitoring:
  cells.base shows progress
  PLAYBOOK §1+§2 GLM cron (live snapshot)
  Per-cell env_snapshot SHA matches lock SHA — if drift, halt + investigate

Post-rerun:
  make analysis (full pipeline)
  python3 scripts/analysis/preregistration_decision_test.py --K_h1 12 --K_h3 11 --TOST-delta 1.0
  Update preregistration.md §6 + osf_lock_manifest.md
  git tag preregistration-locked
  Mint OSF DOI
```

## §K — Reviewer-defensible audit trail

After rerun, the following chain should reconstruct any cell's adjusted_SR:

1. `git show <commit-at-lock>:p79/experiment/analysis.py` (canonical FP rules)
2. `git show <commit-at-lock>:scripts/provenance/snapshot_env.py` (env capture spec)
3. `<run_dir>/env_snapshot.json` (machine + HF + evaluator SHA at run time)
4. `<condition>/episodes/*.json` `rederive_metadata` (per-episode audit trail)
5. `<run_dir>/run_manifest.yaml` cell entries with grade=paper-grade
6. OSF DOI page citing git SHA + advisor email message-id

---

## §M — Statistical methodology gates (paper-grade rigor)

| # | Item | Status | Verify |
|---|---|---|---|
| M1 | **Multiple comparison correction** for H1+H3+TOST family | ✅ already in preregistration.md §3 | Holm-Bonferroni step-down per H-sub-family (PRIMARY: H1/H2; STRUCTURAL: H3 axes; ROUTER: H7/H8; EXPLORATORY: H4 + best-signal-per-mode; POST-HOC: H5/H6 disclosed) |
| M2 | **Bootstrap CI procedure** spec — N resamples, RNG seed, BCa vs percentile | 🟡 partial | preregistration.md routing_signal section mentions bootstrap; add for H1/H3 oracle lift CI |
| M3 | **Power analysis** — minimum detectable effect (MDE) at observed N=234/210/466 with α=0.05, β=0.20 | 🔴 TBD | run `scripts/analysis/power_analysis.py` (笔记 §116 R5 audit followup); paper §3 cite MDE |
| M4 | **Effect size reporting** — pp lift + Cohen's h for binary SR, alongside p-values | 🟡 partial | preregistration.md mentions H3 lift_pp but no Cohen's h; add for paper §5 Table 5 |
| M5 | **Outlier / extreme value handling** — pre-spec rule for tasks with anomalous trajectories (e.g., env crashed mid-task) | 🔴 TBD | preregistration.md FP filter handles eval_fp + na_fp; add "system crash" exclusion rule explicit |
| M6 | **Sensitivity ladder** for FP filter (raw / +na_fp only / +na_fp+eval_fp) reported in Appendix D | ✅ in preregistration.md `FP filter sensitivity` row | Verify `aggregate_sr_fp_per_mode.py` outputs all 3 |
| M7 | **Reporting precision** — 2 decimal pp for SR, 1 decimal pp for differences, full integer for episode counts | 🟡 implicit | Make explicit in paper §3 prose to prevent over-precision |

## §N — Data quality gates (post-collection, pre-analysis)

| # | Item | Status | Verify |
|---|---|---|---|
| N1 | **Episode completeness** — every task attempted, no silent skips | 🟡 | Per cell: `ls episodes/ \| wc -l == expected_n` (234/210/466) |
| N2 | **N balance across cells** — no cell <100 episodes (preregistration.md `N inclusion floor`) | ✅ specified | `aggregate_phantom_lift.py` excludes cells <100 |
| N3 | **Site state contamination check** — pre/post-cell snapshot of mutable state (cart, posted listings, subscribed forums) | 🔴 TBD | Add `scripts/maintenance/site_state_snapshot.sh` — diff cart count / listing count / etc. between consecutive cells |
| N4 | **Auth state freshness per cell** — auth file timestamp ≤ cell launch time | 🟡 | RESET_BEFORE=1 protocol handles, but verify each cell's `.auth/<site>_state.json` mtime within 1h of launch |
| N5 | **Cross-cell shared task pool** — H1/H3 require paired comparisons on same task universe | ✅ task IDs identical across modes within cell (run_manifest.yaml expected_n) | `aggregate_phantom_lift.py` uses common observed-task universe |
| N6 | **Step JSONL no corruption** — all lines parseable JSON, no truncation | 🟡 | `read_jsonl_dedup` handles corrupt lines; verify post-cell zero `corrupt_lines_skipped` counter |
| N7 | **Schema v2 conformance** — every step record has required fields | ✅ | `tests/test_step_schema_v2.py` (must pass post-rerun) |
| N8 | **Wall-clock outliers per task** — flag tasks with >3σ latency (likely crashed env) | 🔴 TBD | Post-rerun: add to analyze_run pipeline |

## §O — Paper-grade disclosure prep (Section 4 limitations prose)

| # | Item | Status | Owner |
|---|---|---|---|
| O1 | **B-20 ua_match GPT-judge drift** prose paragraph | 🔴 TBD | ~3-5 sentences acknowledging GPT-4o-mini judge variance + impact bound |
| O2 | **B-21 string_match GPT-judged binary** prose | 🔴 TBD | ~2 sentences clarifying fuzzy_threshold=1.0 misnomer |
| O3 | **B-22 program_html selector brittleness** (562/1598 = 35%) prose + impact bound | 🔴 TBD | ~3 sentences + future-work pointer |
| O4 | **B-15 finish_wrong_state** as agent error not scaffold (handled by §95 FP filter) | 🟡 partial | preregistration.md FP filter already cites; add Section 4 sentence |
| O5 | **B-26 in_viewport_ratio operator precedence** (CLAUDE.md NOT_FIXED note) | 🟡 partial | Section 4 cite as known DOM advantage source |
| O6 | **B-28 scroll direction confusion** (mitigated via §67 schema) | 🟡 partial | Section 4 cite |
| O7 | **A1/A3 design asymmetries** (B-56 — temperature 0.0/0.1, max_new_tokens 4096/384) | ✅ partial | preregistration.md mentions; Section 4 needs full prose |
| O8 | **Cross-machine numerical drift** (DGX vs Myriad vs A100 sm_121/sm_80/sm_70) | 🔴 TBD | Run `scripts/provenance/numerical_determinism_check.py compare` post-rerun + cite max \|Δh\| |
| O9 | **Pre-Phase-A vs post-Phase-A asymmetry** (preregistration.md cell inclusion main + Appendix D) | ✅ specified | Verify Appendix D robustness check executes |
| O10 | **Stage 2B/2C input from pre-Phase-A archive** (mechanism findings unaffected per 笔记 §116 user Q on "旧数据") | 🟡 partial | Add Section 5 footnote on data vintage independence |

## §P — Inter-rater reliability gates (κ requirements)

| # | Item | Status | Target |
|---|---|---|---|
| P1 | **FP labeling reliability** — 30-task pilot, 2 raters | 🔴 TBD | Cohen κ ≥ 0.7 per preregistration.md `Failure-mode classification rubric` |
| P2 | **Failure-mode 5-bucket rubric reliability** (early_finish / wrong_commit / visual_hijack / click_loop / persistent_error) | 🔴 TBD | κ ≥ 0.7 (preregistration.md target) |
| P3 | **Codex-as-rater calibration** — when codex labels failure modes, spot-check 30 examples manually | 🔴 TBD | Disagreement >30% triggers prompt revision before scaling |
| P4 | **Visual subset audit** (43 VWA non-visual + manual review) | ✅ exists `docs/analysis/cross_sites/vwa_manual_non_visual_task_ids.py` | Used as Appendix D robustness check |

## §Q — Robustness / sensitivity analysis pre-spec (paper §3 must commit before lock)

| # | Item | Status | Notes |
|---|---|---|---|
| Q1 | **Non-visual subset robustness** — H1/H3 also evaluated on 43 VWA + 480 WA non-visual tasks | ✅ in preregistration.md `Non-visual subset robustness` row | Replaces deprecated visual_fp |
| Q2 | **Pre-Phase-A archive robustness** — Appendix D shows H1/H3 with archived data | ✅ in preregistration.md `Cell inclusion (Appendix D)` | Symmetric contamination disclosure |
| Q3 | **FP filter sensitivity** — 3 variants reported (raw / +na_fp / +na_fp+eval_fp) | ✅ in preregistration.md (post-§95 reform per §116 fix) | aggregate_sr_fp_per_mode.py outputs all |
| Q4 | **K_h1 / K_h3 threshold sensitivity** — also report decision at K_h1±1 / K_h3±1 | 🔴 TBD | Per-cell pass count is robust, but show threshold gradient |
| Q5 | **Per-difficulty bucket** — split tasks by intent length / N actions / has_reference_image | 🔴 TBD | Show H1/H3 hold across 3 difficulty terciles |
| Q6 | **Hold-out site validation** (LOSO if advisor confirms) — train router on red+shop, test cls | 🟡 advisor email | preregistration.md mentions LOSO as alternative to k-fold |
| Q7 | **Reproducibility on different machine** — DGX vs A100 vs Myriad cross-validation | 🟡 | numerical_determinism_check post-rerun |

## §R — Cost / sustainability tracking (Section 8 prep)

| # | Item | Status | Verify |
|---|---|---|---|
| R1 | Per-cell GPU-hours estimate logged | ✅ `condition_summary_v2.json` `total_latency_ms` + GPU type | Aggregate post-rerun |
| R2 | Per-cell USD cost (B0 API) | ✅ `cost_usd.model` per step | Aggregate post-rerun |
| R3 | Carbon footprint per cell (45-region table) | ✅ `aggregate_cost_electricity.py` | Run post-rerun |
| R4 | Total experiment compute budget tracking | 🟡 | Add running total in PLAYBOOK §1 GLM-managed |
| R5 | Cross-platform GPU power profile (sm_121 vs sm_80 W draw) | 🔴 TBD | NVML probe per cell start (笔记 §114 Gap 5 + audit R3) |
| R6 | Section 8 prose draft references R1-R5 | 🔴 TBD | After 16-cell + mechanistic complete |

## §S — Failure mode contingency (resume / recovery protocols)

| # | Scenario | Status | Protocol |
|---|---|---|---|
| S1 | A100 GPU OOM mid-cell | 🟡 partial | Watchdog auto-clean handles; verify resume from last checkpoint |
| S2 | Myriad qsub job killed (wallclock) | 🟡 partial | run_stage2b script writes `patching_continuation_results.json` incrementally — resume via `--resume` flag (R4 pending) |
| S3 | VWA Docker container restart mid-rerun | 🟡 | RESET_BEFORE=1 + auth_refresh; verify ntfy notifies on >3 consecutive auth failures |
| S4 | B0 proxy API rate-limit / 503 cascade | ✅ B-50d fix | Exponential backoff 3 attempts (10/20/40s); mitigation in catalog B-50d |
| S5 | Phase A locator-route regression on edge case | 🔴 TBD | Add halt-on-N-consecutive-failures detection (e.g., 5 in a row → ntfy + halt cell) |
| S6 | Disk full mid-cell (Scratch / archive) | 🟡 | Pre-launch H4 verify; mid-launch monitoring TBD |
| S7 | Network partition (DGX → quark Tailscale OR A100 → bastion) | 🟡 | A100 self-host VWA solves; Tailscale-dependent path risky |
| S8 | Cell completes with `expected_n - actual_n > 5` (silent skips) | 🔴 TBD | Add post-cell gate: if mismatch, halt subsequent cells until investigated |

## §T — Evaluator independence verification (paper §3 reviewer-defensible)

| # | Item | Status | Verify |
|---|---|---|---|
| T1 | VWA evaluator code unchanged from upstream | ✅ | `git diff upstream/main -- external/visualwebarena/evaluation_harness/` empty (apart from documented patches in 笔记) |
| T2 | GPT-4o-mini judge prompt template pinned | 🟡 | `evaluation_harness/helper_functions.py:llm_fuzzy_match` prompt; verify no edits |
| T3 | Judge model temperature explicit (=0 ideally for determinism) | 🟡 | Verify; if non-zero, paper §3 disclose |
| T4 | Episode-level eval reproducibility — re-run evaluator on N=20 spot-check episodes, verify SR stable | 🔴 TBD | Add `scripts/provenance/eval_reproducibility_check.py` |
| T5 | Cross-evaluator-version sensitivity — if evaluator code changes between cells (shouldn't), `rederive_metadata` per Protocol B captures | ✅ via §115 Protocol B | reeval_audit_protocol.md |

---

## §L — References

- `docs/checkpoints/preregistration.md` (canonical commitment)
- `docs/checkpoints/ADVISOR_SYNC.md` (sync prep)
- `docs/checkpoints/advisor_sync_5_5_outcomes.md` (decision register)
- `docs/checkpoints/advisor_sync_5_5_followup.md` (Q1-Q11 pending email)
- `docs/checkpoints/osf_lock_manifest.md` (8-step DOI workflow)
- `docs/checkpoints/evaluator_change_protocol.md` (Protocol A — 4-tier classification)
- `docs/checkpoints/reeval_audit_protocol.md` (Protocol B — episode audit trail)
- `docs/reference/master_bug_catalog.md` (37 catalogued bugs)
- `docs/reference/PAPER_STRATEGY_OPEN_QUESTIONS.md` (Q1-Q9 strategic questions)
- 笔记 §107 (Phase A bug fix wave) / §110 (5/5 sync) / §114 (provenance) / §115 (Protocol A+B) / §116 (this audit)
