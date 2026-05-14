# Pre-Rerun Audit — Index (slimmed 2026-05-14)

**Status**: 🟢 Index only. The prior 678-line standalone checklist was a stale
aggregator (titled "16-Cell", referenced the since-deleted
`PAPER_STRATEGY_OPEN_QUESTIONS.md`, predated the queue Gate 1-7 hardening + the
/stress v6 pre-fire audit). Its content is now enforced by — and lives in — the
canonical sources below. This file is kept as a **pointer index** so a reviewer or
future-self has one place to find "where is the X gate".

**Why slimmed** (codex prereg-structure review + folder-redundancy review,
2026-05-14): a 678-line doc that *copies* spec content from ~6 other docs goes
stale the moment any source changes — exactly the failure this project's doc
philosophy warns against (CLAUDE.md "don't hardcode what changes; reference it").

---

## Where each audit layer now lives

| Audit layer | Canonical source | Enforced by |
|---|---|---|
| **Pre-launch gates** (prereg locked, env/vwa snapshots committed, preflight, GPU, no-conflict, all chain configs exist) | — | `scripts/queues/queue_phase1_paper_grade.sh` **Gate 1-7** (`check_gates()`) — blocking, codex /stress v6 hardened 2026-05-14 |
| **Statistical methodology pre-spec** (estimand, FE pooling, superiority test, H1-H3 family, K-of-N transparency, FP filter, bootstrap unit, missing-data policy, stopping rules) | `preregistration.md` §2 (H1-H3 + framing rule + §2.4 power + §2.5 decision flow), §3 (family declaration), §4 (locked choices) | prereg lock + `preregistration_decision_test.py` |
| **Pre-registration & witness** | `preregistration.md` §6 + `osf_lock_manifest.md` (8-step DOI workflow + artifact-freeze registry) | advisor email + git tag + OSF DOI |
| **Evaluator change discipline** | `evaluator_change_protocol.md` (Protocol A — 4-tier classification) | commit-message prefix discipline + `evaluator_code_sha` |
| **Re-derive audit trail** | `reeval_audit_protocol.md` (Protocol B — `rederive_metadata` per-episode log) | `rederive_episode_summary.py` |
| **Run-process safeguards** (reset-before, auth refresh, watchdog 6-layer auto-clean, cross-cell isolation, mid-run halt criteria) | `docs/reference/reference_watchdog_protocol.md` (replica) + `preregistration.md` §4 "Stopping rules" row | `experiment_watchdog.py` + queue scripts (FORCE_NEW + completion sentinel, codex /stress v6 C1/C3) |
| **Version / dependency pinning** | `locked_versions.md` (single source of truth — SHA / hash / commit) | `snapshot_env.py` + `snapshot_vwa.sh` |
| **Model + dataset descriptors** | `model_card.md` + `dataset_card.md` (human-readable cards; pinned values should reference `locked_versions.md`) | — |
| **Launch protocol / checklist** | `docs/reference/launch_checklist.md` | `make launch` wrapper |
| **Top-venue compliance scoreboard** | `topvenue_constraints.md` (auto-generated — regenerate, don't hand-edit) | — |
| **Negative-results discipline** | `negative_results_registry.md` | — |
| **Release hygiene** | `release_redaction_checklist.md` + `ethics_license_coi_statements.md` | pre-OSF-deposit manual review |
| **Cross-AI pre-fire audit** | `docs/checkpoints/codex_outputs/stress_v6_pre_fire_*` + `codex_stress_16cell_design_2026-05-13.md` | /stress v6 skill |

## Pre-fire checklist (the actual gate — run this)

```bash
bash scripts/queues/queue_phase1_paper_grade.sh dry-run    # preview 24-condition chains
bash scripts/queues/queue_phase1_paper_grade.sh launch     # Gate 1-7 block on any failure
```

Gate 1-7 (in `check_gates()`) is the executable replacement for what this file's
prior Phase-1/2/3 lifecycle prose described by hand:
1. `preregistration.md` status `locked` + no TBD thresholds
2. `env_*_baseline.json` committed
3. `vwa_*.json` snapshot committed
4. `preflight_v2.sh` passes (blocking — codex /stress v6 C2)
5. GPU/CUDA available (blocking — C2 sibling)
6. No conflicting active runs unless `ALLOW_ACTIVE_RUNS=1` (blocking — C6)
7. All chain configs exist (blocking — C9)

## References

- `docs/checkpoints/实验笔记.md` §132-§135 — the audit + fix chronicle this index supersedes
- Prior full version: `git log --follow docs/checkpoints/pre_run/pre_rerun_audit.md` (678-line version at commit before 2026-05-14)
