---
protocol_note_id: 01
title: Watchdog session-loss cleanup → paper_grade-preserve (B-1868) — implementation-alignment with B-1777 invariant, NOT estimand change
date: 2026-05-27
status: DRAFT — witness for code change + master_bug_catalog entry + tests; live-verification pending fresh re-launch
parent_prereg: docs/checkpoints/pre_run/preregistration.md (status: locked)
parent_doi: 10.17605/OSF.IO/9QCWU
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_01a_SCHEMA_VALIDATOR_20260521
  - AMENDMENT_02_GATE_LADDER_20260523
  - AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524
  - AMENDMENT_04_ANALYSIS_ALIGNMENT_20260524
  - AMENDMENT_05_COORDINATE_CONTRACT_20260525
  - AMENDMENT_06_REPRODUCIBILITY_SENSITIVITY_20260525
  - AMENDMENT_07_SOM_IDENTIFIER_CONTRACT_20260525
witness_tag: protocol-note-01-session-lost-paper-grade-20260527   # to be created at the finalizing commit
relation: >
  NOT estimand-affecting. PROTOCOL_NOTE_01 differs structurally from AMENDMENT_##
  (01-07): no H1/H3/H10 estimand changes, no scored_task_count change, no
  observation-id contract change, no eval-context change. This is recovery-
  semantics alignment: B-1777 (笔记 §247, 2026-05-20) established the invariant
  "paper_grade=True ⇒ never delete+retry" for the watchdog's error-retry path
  (`_can_auto_retry`). B-1868 extends that invariant to the sibling session-
  cleanup path, which was the same denominator-surgery class of bug but
  outside B-1777's literal scope.
provenance: >
  R14849 (B0 P-SoM cls, P79_PAPER_GRADE=1 fire) 2026-05-27 13:47Z lost
  classifieds session task 142→143; pre-fix watchdog auto-cleaned 3 episodes
  (task 143/144/145). User noted catalog B-1777 invariant was scope-restricted
  to error-retry path while session-cleanup path silently performed the same
  denominator surgery class of operation. Code fix written; cross-AI audit
  Mode B (codex) + Mode C (gemini) completed 2026-05-27 with 4 P0 + 7 P1 + 3
  P2 unified bug list; user confirmed fix scope (1A schema add, 2A prose +
  aggregator + codex round, 3C downgrade to PROTOCOL_NOTE_01, 4A full fix
  scope + bottom auto-default). PROTOCOL_NOTE_01 witnesses the
  implementation-alignment commit + tests + catalog entry; OSF deposit NOT
  required for this tier (recovery alignment vs estimand revision — per
  AMENDMENT_07 OSF kv9sf precedent which was an estimand contract change).
---

# PROTOCOL_NOTE_01 — Watchdog session-loss paper_grade preserve (B-1868)

## 0. Scope clarification — why "PROTOCOL_NOTE" not "AMENDMENT_08"

User Q3=C decision 2026-05-27: B-1868 is **implementation-alignment**, NOT
estimand change. AMENDMENT_## namespace is reserved for changes that move
H1/H3/H10 estimand definitions, scored_task_count, observation-id contracts,
eval-context, or sample-pool composition (AMENDMENT_07 changed how SoM-family
modes serialize element identifiers — agent observation surface changed).
B-1868 only changes **runtime recovery semantics**:

- Pre-B-1868: paper_grade=1 + session lost → silent `clear_task_files` →
  RESUME_MISSING re-runs the deleted task at a different server state.
  Effect: replaces a failed-by-infra attempt with a new attempt that may
  succeed; cross-baseline mode/wallclock-correlated denominator surgery
  asymmetry (B-1777 same class).
- Post-B-1868: paper_grade=1 + session lost → preserve the contaminated
  episode in canonical denominator + emit `session_lost_preserved` covariate
  for paper §3.5 non-gating sensitivity.

The H1/H3/H10 estimand definitions are UNCHANGED. The H1 drop-one oracle,
H3 axis-decomposition, H10 learned-router Pareto are all defined at the
canonical episode-outcome level; B-1868 preserves the canonical episode
outcome where pre-fix would have silently rewritten it. No prereg estimand
text needs amending. Hence PROTOCOL_NOTE_##, not AMENDMENT_##.

## 1. What changes

### 1.1 Recovery-semantics alignment (B-1777 invariant scope extension)

`scripts/maintenance/experiment_watchdog.py` session-restored branch
(~L2641-2790) now branches on `_watchdog_paper_grade`:

- **`P79_PAPER_GRADE=1`**: PRESERVE contaminated episodes in canonical
  denominator. Emit `session_lost_paper_grade_preserved` trajectory event
  (deterministic event_key for aggregator dedup) + atomic-patch episode
  summary `infra_covariates += ["session_lost_preserved"]` for canonical-
  artifact dual-path read.
- **`P79_PAPER_GRADE=0`** (dev mode): existing `clear_task_files` +
  RESUME_MISSING-style retry path unchanged.

Detection-time event (`session_lost_contaminated_detected`) ALSO gated on
paper_grade — dev mode already has `task_auto_cleared` from `clear_task_files`
as the existing covariate channel.

### 1.2 Persist-state ordering (B-1584 FATAL race close)

Persist `_persist_state()`:
- IMMEDIATELY after `session_contaminated[site].append(...)` — so subsequent
  B-1584 FATAL `raise` does NOT lose the contamination tuple.
- Best-effort in B-1584 FATAL `except` block before `raise`.
- After preserve loop completes (existing batch-end persist retained).

Peek-then-pop at restore: `session_contaminated.get(site, [])` not `.pop`
upfront; pop only after preserve loop completes. Mid-loop crash → state
file still carries original list → restart replays idempotently via
`event_key`.

### 1.3 Canonical summary schema (P1-3-B*)

`p79/experiment/types.py:610` adds `infra_covariates: List[str] = []` field
to `EpisodeSummaryV2` + entries in `PAPER_GRADE_EPISODE_OPTIONAL_KEYS` +
`_EPISODE_OPTIONAL_FIELD_TYPES`. Default `[]` = "no infrastructure
contamination". List enum so multiple infra contaminations can co-occur on
the same episode (forward-compat for future markers).

This is the canonical-artifact dual-path defense. Pre-fix, all preserve
semantics lived in trajectory_events.jsonl event-log path only; any gap
(logger fail + fallback miss + replay miss) → reviewer reading canonical
episode summary directly sees `success=False/score=0` and CANNOT distinguish
"agent clean failure" from "infrastructure preserved outcome". B-543
`needs_reevaluation` is the precedent for summary-level non-exclusionary
flag pattern.

### 1.4 Aggregator updates

`scripts/analysis/aggregate_trajectory_covariates.py` adds:
- **Event_key dedup** before lookup-table build (watchdog restart replay
  events collapse to one).
- **Fallback file replay** from `watchdog_session_preserved_failures.jsonl`
  per condition_dir.
- **`session_lost_preserved`** (bool) + **`session_lost_preserved_wave_size`**
  (Optional[int]) covariate columns. Dual-path OR semantics: event-log OR
  summary `infra_covariates` → True.

### 1.5 Helpers (testability)

`scripts/maintenance/experiment_watchdog.py` adds module-level pure
functions:
- `_build_session_lost_event_key(*, run_id, condition_id, task_id, condition_key, phase)`
- `_build_session_lost_detected_metadata(...)`
- `_build_session_lost_preserved_metadata(...)`
- `_append_watchdog_audit_fallback(condition_dir, entry)` (shared fallback writer with fcntl flock + fsync, mirrors `logger_v2.py:251-260`)
- `_mark_episode_infra_covariate(condition_dir, task_id, site, covariate)` (atomic summary patch)

Tests (`tests/test_b1868_session_lost_paper_grade_guard.py`, 14 invariants):
metadata required keys + forbids-`is_noise` + event_key determinism + source-
grep forward-guard (inline call sites can't bypass helpers) + fallback
atomic round-trip + summary-marker idempotency + aggregator dual-path round-
trip + aggregator event_key dedup.

## 2. What does NOT change

- H1 drop-one oracle estimand definition
- H3 axis-decomposition estimand definition
- H10 learned-router Pareto estimand definition
- `scored_task_count` (cls=224, red=205, shop=435 per §139.8 + B-91 N/A
  exclusion)
- Observation-id contracts (AMENDMENT_05 coord, AMENDMENT_07 SoM ids)
- Eval-context (AMENDMENT_04 / Fire-6 C1 isolation)
- Sample-pool composition (Phase 1a 36 conditions / 6 cells)
- The watchdog detection logic (`_check_session_health` `_SITE_AUTH_REGEX`)
- `_auto_refresh_auth` recovery path (logged-in tasks resume normally)
- B-1584 FATAL SIGTERM-runner path (only adds best-effort pre-raise persist)
- B-742 `auth_refresh_no_clear` event emit
- B-880 `clear_task_files` single source in dev-mode path

## 3. Provenance — cross-AI audit chain

- Mode A (Claude /stress): 2 P0 + 3 P1 + 2 P2 = 7 findings (initial)
- Mode B (codex /codex-stress): 5 findings, 4 OOB — NEW catches: event_key
  vaporware, B-1584 FATAL race, fallback-JSONL dead-end, canonical summary
  schema gap (Finding 5 = P1-3-B*)
- Mode C (gemini /gemini-stress): 4 findings — NEW catches: paper §3.5 prose
  vaporware (P0-4-C* OOB), AMENDMENT directory placement (was
  `docs/checkpoints/pre_run/` should be `docs/prereg_amendments/`), R14849
  forensic gap explicit disclosure, B-1777 cross-link §253→§247 fix
- User fix-scope decision 2026-05-27: 1A (P1-3 schema), 2A (P0-4 prose +
  aggregator same PR + 1 codex round), **3C (PROTOCOL_NOTE_01 not
  AMENDMENT_08)**, 4A (full P0 + selected P1) + bottom auto-default

Cross-link prose: `master_bug_catalog.md` B-1868 entry (full Fix steps
1-8 + Behavior preserved + Condition-level integrity threshold + Pre-fix
archive disclosure).

## 4. Pre-fix R14849 archive plan

R14849 (B0 P-SoM cls, 2026-05-27 08:35-14:32Z, killed @ task 168 during fix
implementation) is **archived non-canonical**:
- Tag: `pre_B-1868_session_cleanup_artifact`
- Path: `_archive_b1868_session_cleanup_R14849/` (planned, post-Phase-B
  step on A100 side)
- Contents: surviving episodes task 0-142 + 146-168 (sans N/A excludes)
- Forensic gap: episodes 143/144/145 are PHYSICALLY GONE (B-863 reaper
  purged pending_delete markers ~13:52Z, ~5min after watchdog cleanup);
  catalog gemini Finding 3 disclosure makes this gap explicit.
- Use: RCA evidence of pre-fix watchdog behavior, NOT analysis input.

Fresh `queue_phase1_paper_grade.sh launch` (FORCE_NEW, NOT RESUME_MISSING)
for the canonical Phase 1a P-SoM cls cell post-PROTOCOL_NOTE_01 commit.

## 5. Witness chain (PROTOCOL_NOTE_01 pattern)

Required artifacts:
- ✅ `scripts/maintenance/experiment_watchdog.py` patch (3 helpers + restore-
  time paper_grade branch + detection-time event + persist-state ordering +
  shared fallback + summary marker)
- ✅ `p79/experiment/types.py` schema (`infra_covariates` field + OPTIONAL_KEYS
  + FIELD_TYPES entries)
- ✅ `scripts/analysis/aggregate_trajectory_covariates.py` updates (event_key
  dedup + fallback replay + 2 new covariate columns)
- ✅ `tests/test_b1868_session_lost_paper_grade_guard.py` (14 tests, all pass)
- ✅ `docs/reference/master_bug_catalog.md` B-1868 entry (full Fix steps +
  Behavior preserved + threshold table + forensic gap disclosure)
- ✅ This PROTOCOL_NOTE_01 witness doc (`docs/prereg_amendments/PROTOCOL_NOTE_01_*.md`)
- ⏳ Paper prose: `paper_drafts/section3_definition.md` §3.5 non-gating
  sensitivity disclosure + `section4_findings.md` covariate column reference
  (1 codex round on prose, pending)
- ⏳ git tag `protocol-note-01-session-lost-paper-grade-20260527` (at
  finalizing commit, pending)
- ⏳ 笔记 chronicle §303 append `[bug][design][infra]` (after fresh-launch
  live-verification, pending)

NOT required (recovery alignment vs estimand revision distinction):
- OSF deposit (only AMENDMENT_##-grade estimand changes require, per
  AMENDMENT_07 OSF kv9sf precedent which was the SoM sequential-id contract
  estimand change).
- Prereg §2.5 H1 claim-tier gate update (estimand unchanged).
- DOI 2 schedule adjustment (final OSF DOI 2 mint timeline unchanged;
  PROTOCOL_NOTE_01 is in the audit trail for that DOI but not a gate).

## 6. Live-verification plan

Fresh re-launch step ordering:
1. Commit PROTOCOL_NOTE_01 changes (this doc + code + tests + catalog)
2. Push to remote (paper-grade gate verification)
3. Tag `protocol-note-01-session-lost-paper-grade-20260527`
4. A100-side: `git pull` + sync to fresh-launch script path
5. `bash scripts/queues/queue_phase1_paper_grade.sh launch` (FORCE_NEW)
6. Monitor via `fire6_monitor` cron + `paper_grade_check.py` 6h cron
7. Live test of preserve path triggers when classifieds session loss recurs
   (probabilistic — may not occur in first re-launch wave); if it occurs,
   verify:
   - episode summary contains `infra_covariates: ["session_lost_preserved"]`
   - trajectory_events.jsonl contains `session_lost_paper_grade_preserved`
     event with deterministic event_key
   - aggregator output has `session_lost_preserved=True` for affected episode
   - canonical denominator unchanged (episode count for condition = expected
     scored_task_count)
8. If preserve fires successfully → append 笔记 §303 "PROTOCOL_NOTE_01 land
   + live-verified" + remove the `live-verification pending` status from
   this doc.
9. If preserve does NOT fire in first wave → live-verification deferred; the
   14 unit tests + cross-AI audit + source-grep forward-guards provide
   sufficient witness for paper-grade Phase 1a continuation.
