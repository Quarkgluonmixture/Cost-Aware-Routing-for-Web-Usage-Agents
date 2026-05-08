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
