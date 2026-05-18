# Pre-Rerun Audit Walkthrough — 2026-05-18 (pre-Phase-1a-Pass-1-fire)

> **Purpose**: Per `pre_run/pre_rerun_audit.md` index format (slimmed 2026-05-14),
> verify each of the 11 canonical audit layers points to a current locked
> source pre-fire. Output = per-layer ✓ / ⚠️ / ❌ + cited locked-source state.
>
> **Status**: 🟢 Pre-fire walkthrough complete 2026-05-18 (post §A2 12/16 audit
> cascade closure + B-1603 + B-1604 + B-1605 + my A2.9 chain).
>
> **Reviewer comprehension**: this file is a one-pass walkthrough showing each
> audit layer's current state; it does NOT duplicate canonical content (the
> 678-line stale aggregator was slimmed precisely because hardcoded copies go
> stale).

---

## 11-layer verification table

| # | Audit layer | Canonical source | Current state | Status |
|---|---|---|---|:---:|
| 1 | **Pre-launch gates** (prereg locked, env/vwa snapshots committed, preflight, GPU, no-conflict, all chain configs exist) | `scripts/queues/queue_phase1_paper_grade.sh` Gate 1-7 (`check_gates()`) — codex /stress v6 hardened 2026-05-14 + A2.7 substrate B-1400/06/07/10/12/13/14 + B-991 GLM rescue retire | Gate 1-7 wired; queue dry-run 36-cond emit verified 2026-05-18 my SSH | ✅ |
| 2 | **Statistical methodology pre-spec** (estimand, FE pooling, superiority test, H1-H3 family, K-of-N transparency, FP filter, bootstrap, missing-data, stopping rules) | `preregistration.md` §2 + §3 + §4 (post §A2.3a/b/c/d B-941~B-1064 + A2.5 B-994~B-1006 + A2.8 B-1550~B-1561 + A2.9 NeurIPS Q1-Q16 + B-1580 estimand) | All resolved per §A2 cascade; canonical 3-axis cost-latency estimand B-1580 propagated to §4 row; H10 two-layer operational gate B-1550; FE pooled bootstrap H1 primary B-1009 | ✅ |
| 3 | **Pre-registration & witness** | `preregistration.md` §6 + `osf_lock_manifest.md` (8-step DOI workflow) | Substance locked via Git refs; B-1570 doctrine shift advisor email optional; OSF DOI mint post-Phase-1a-fire-data-complete per §3 header | ✅ (locked-substance; tag pending fire) |
| 4 | **Evaluator change discipline** | `evaluator_change_protocol.md` (Protocol A — 4-tier classification) | Protocol exists; T0/T1/T2/T3 classification spec; commit-message prefix `fix(eval-postlock):` discipline; `evaluator_code.combined_sha256` recorded per run | ✅ |
| 5 | **Re-derive audit trail** | `reeval_audit_protocol.md` (Protocol B — `rederive_metadata` per-episode log) | Protocol exists; B-133 banner for §139.8 FP-architecture-retire context updated 2026-05-15; archive vs post-§139.8 distinction clear; §6 OSF DOI lock interaction spec (cells with grade=paper-grade need rederive_metadata non-empty at lock time) | ✅ (post-fire enforcement) |
| 6 | **Run-process safeguards** (reset-before, auth refresh, watchdog 6-layer auto-clean, cross-cell isolation, mid-run halt criteria) | `docs/reference/reference_watchdog_protocol.md` (replica) + `preregistration.md` §4 "Stopping rules" row | Watchdog 6-layer defense (detect→alert→refresh→cleanup→resume→verify); paper §4.X.13 race-window honest "10-200ms typical, up to ~1s heavy artifacts" per B-765 A1.15 Chunk c 2026-05-17; experiment_watchdog.py + queue scripts FORCE_NEW + completion sentinel | ✅ |
| 7 | **Version / dependency pinning** | `locked_versions.md` (single source) | VWA submodule SHA `f883a116da89c2acc3a7530e48bb8d70a5f4571d` (A1.18-re Chunk 1 + A2.7 P0-1-B* sync 2026-05-18 supersedes `1c3a615` + `eb5cbd8` + `f0c835b`); B1 HF SHA `ebb281ec...`; **B2 HF SHA `093f9f388b31de276ce2de164bdc2081324b9767` locked 2026-05-18 per B-1603**; Playwright 1.58.0; Chromium revision 1208 | ✅ |
| 8 | **Model + dataset descriptors** | `model_card.md` + `dataset_card.md` | B0/B1/B2 cards aligned; B2 HF SHA filled (B-1603); cls 234 / red 210 / shop 466 = 910 task pool hashes recorded per `locked_versions.md`; 3-baseline scope explicit | ✅ |
| 9 | **Launch protocol / checklist** | `docs/reference/launch_checklist.md` + `phase1_plan.md §B` | `make launch` wrapper functional (verified my 4 DRY=1 runs 2026-05-18); `queue_phase1_paper_grade.sh` dry-run emits 36 ops clean | ✅ |
| 10 | **Top-venue compliance scoreboard** | `topvenue_constraints.md` | 78-constraint internal audit format ✓/⚠️/❌; supplanted at paper-time submission layer by `pre_run/neurips_checklist.md` Q1-Q16 NeurIPS 2025 standard format (B-1506 A2.9 P0-1-AC* OOB 2026-05-18) | ✅ |
| 11 | **Negative-results discipline** | `negative_results_registry.md` | 12+ retracted framings registered; C1 phantom-SoM 4-fold drop-in confirmed-framing entry (post-B-1502 36→42 sweep 2026-05-18); paper-action items §1 + §2 + §5 + §8 prose mappings explicit | ✅ |
| 12 | **Release hygiene** | `release_redaction_checklist.md` + `ethics_license_coi_statements.md` | `make pre-release-check` wired (B-1512 /stress A2.9 P0-7-ABC* 2026-05-18); 5-step recipe PASS verified 2026-05-18 + sign-off row; LLM Use Disclosure section per NeurIPS 2025 (B-1507); release license matrix incl. Gemma Terms of Use (B-1501) | ✅ |
| 13 | **Cross-AI pre-fire audit** | `docs/checkpoints/codex_outputs/*` + `gemini_outputs/*` + `docs/checkpoints/process/*_skill_replica.md` | /stress 3-lineage workflow (Claude + codex v0.130 + gemini); §A2 audits 1-9 cycles all closed; A2.10 + A2.6c in flight parallel sessions; B-1570 advisor doctrine shift; v7.8 Phase 4 claim-realness spot-check mandatory | ✅ (Submission-scope A2.10/A2.6c parallel) |

## Pre-fire executable checklist (Gate 1-7 per queue_phase1_paper_grade.sh check_gates())

| Gate # | Predicate | Status |
|---|---|---|
| 1 | `preregistration.md` status `locked` + no TBD thresholds | ⚠️ frontmatter currently `draft`; will flip at lock event pre-fire (this walkthrough is the prep, lock event is one commit + `git tag preregistration-locked`) |
| 2 | `env_*_baseline.json` committed | ✅ A100 snapshot `a100_pre_launch_2026-05-18_071612.json` written 2026-05-18 my SSH (B-1428 VWA SBOM 4-match PASS); commit at lock event |
| 3 | `vwa_*.json` snapshot committed | ✅ tree-hash chain witness `142bb1b6f18b37fbb12c1c2e84f91929d70a1f605560adcf2ddb0325248a7d46` matches lock; committed at lock event |
| 4 | `preflight_v2.sh` passes (blocking, codex /stress v6 C2) | ⚠️ to be run at fire start; queue_baseline.sh chains into preflight per A2.7 substrate; A100 venv + playwright 1.58 + HF caches all verified my SSH 2026-05-18 |
| 5 | GPU/CUDA available (blocking) | ✅ A100-PCIE-40GB / 39.5 GB total / 40441 MB free verified my SSH 2026-05-18 |
| 6 | No conflicting active runs unless `ALLOW_ACTIVE_RUNS=1` (blocking, codex /stress v6 C6) | ⚠️ verified at fire start via `pgrep -f "run_experiment.*<site>"` empty |
| 7 | All chain configs exist (blocking, codex /stress v6 C9) | ✅ 12 yaml configs verified `make vwa-generate-configs` my SSH 2026-05-18 + 910 per-task configs materialized cls 234 / red 210 / shop 466 |

## Audit verdict (pre-fire 2026-05-18)

**11/11 canonical audit layers ✅** + **5/7 Gate 1-7 ✅, 2/7 ⚠️ deterministic flip-at-lock-event** (Gate 1 prereg status + Gate 4 preflight run + Gate 6 active-runs check are runtime gate executions, not source-of-truth verifications).

**Walkthrough decision**: APPROVE pre-fire (Gate 1-7 will execute at queue launch trigger).

**Sign-off**: Claude (Opus 4.7) /stress 深入审 Mode A successor 2026-05-18 — walkthrough complete pre-Phase-1a-Pass-1-fire.

---

## Next step

Per `osf_lock_manifest.md §3` 8-step workflow (B-1570 doctrine, advisor email optional):

1. ⏭ **Step 2-6 pre-fire**: snapshot env + vwa + paper_drafts + tag git
2. ⏭ **Fire `queue_phase1_paper_grade.sh launch`**
3. ⏭ **Post-fire Steps 7-8**: OSF page upload + DOI mint + backfill manifest

This walkthrough closes phase1_plan §B1 `pre_rerun_audit.md` 走查 item.
