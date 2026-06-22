---
title: Pre-flight transient-substrate episode-retry (B-1881) — recovery-alignment, NOT estimand change
status: DRAFT — witness for code change + master_bug_catalog B-1881 + tests; live-verification pending re-launch
parent_doi: 10.17605/OSF.IO/9QCWU
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_02_GATE_LADDER_20260523
  - AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524
  - AMENDMENT_04_ANALYSIS_ALIGNMENT_20260524
  - AMENDMENT_05_COORDINATE_CONTRACT_20260525
  - AMENDMENT_06_REPRODUCIBILITY_SENSITIVITY_20260525
  - AMENDMENT_07_SOM_IDENTIFIER_CONTRACT_20260525
  - PROTOCOL_NOTE_01_SESSION_LOST_PAPER_GRADE_20260527
witness_tag: protocol-note-02-transient-preflight-retry-20260621   # created at finalizing commit
osf_deposit: NOT REQUIRED — recovery-alignment, estimand UNCHANGED (see §0); mirrors PROTOCOL_NOTE_01 tier
cross_ai_audit: /stress 3-AI (Claude Mode A + codex Mode B + gemini Mode C) 2026-06-21
---

# PROTOCOL_NOTE_02 — Pre-flight transient-substrate episode-retry (B-1881)

## 0. Scope clarification — why "PROTOCOL_NOTE" not "AMENDMENT_08"

The H1/H3/H10 estimand definitions are **UNCHANGED**. SR is still "one recorded
outcome per (task, seed), on valid substrate; infra failures are NOT absorbed as
agent score=0" (the B-486 / B-783 paper-grade principle). B-1881 changes only
*recovery behavior* for a strict subset of failures: a **PRE-FLIGHT** quarantine
(`steps == 0` — the agent took NO browser action) of class `auth` or `network`
now triggers a bounded episode-level retry on fresh substrate instead of
aborting the whole condition.

**Why this leaves the estimand invariant**: at `steps == 0` the episode never
ran — the failure is at the auth gate / reset-goto / first model call, *before*
any agent action. Retrying is "the episode finally starting on valid substrate",
mathematically identical to the counterfactual where the auth blip had been a few
seconds shorter. There is (i) no site mutation to carry over, (ii) no partial
stochastic rollout to re-draw, (iii) no possibility the agent caused the failure.
The recorded outcome is still the *first valid* single rollout. No H1 drop-one
oracle, H3, or H10 router quantity moves; `scored_task_count` (cls 224 / red 205
/ shop 435) is unchanged; no observation-id / eval-context / sample-pool change.
Hence PROTOCOL_NOTE_## (recovery alignment), not AMENDMENT_## (estimand change) —
same tier as PROTOCOL_NOTE_01.

**Cross-AI dissent recorded**: gemini Mode C (2026-06-21) returned a verdict of
"MANDATORY OSF AMENDMENT". That verdict was predicated on the *original* B-1881
draft which retried **mid-episode** transients too (re-drawing a partial rollout
on B0's ~14pp-nondeterministic proxy = a conditional multi-shot estimand). The
3-AI consensus fix — restricting retry to `steps == 0` pre-flight — *removes* the
re-draw entirely, which is precisely what collapses gemini's amendment argument
back to a protocol note. codex Mode B independently reached the same boundary
("retry only `steps==0` / `model_call_attempt_count==0`; otherwise abort"). The
narrowing is the load-bearing reason this is NOT an estimand change. Advisor may
upgrade to AMENDMENT_08 at next sync if they disagree; until then the disclosure
in §3 + the `transient_retry_count` telemetry make the behavior fully auditable.

## 1. What changes

### 1.1 Pre-flight transient retry (the contract change)
`p79/experiment/runner/main.py` — the original `_run_and_record_episode` was
renamed `_run_and_record_episode_once` (its B-168 / B-486 / B-488 / B-323
invariants are byte-unchanged); a thin `_run_and_record_episode` wrapper now
catches `PaperGradeAbortError` and retries iff ALL hold:
- `paper_grade=True` AND NOT `diagnostic_replay`
- `exc.transient_class ∈ {auth, network}` (structured provenance on the exception,
  not message re-parsing)
- `exc.steps == 0` (pre-flight master gate)
- `attempt < transient_episode_max_retries` (default 3, yaml-exposed)

`proxy_5xx` is **excluded** from episode-retry: the proxy agent already retries
5xx internally for ~11min (B-1880 capped backoff); a proxy_5xx that reaches the
episode level = that budget exhausted = sustained outage ⇒ legitimate abort (also
avoids a B-1880×B-1881 ~4×11min worst-case time-to-abort). Mid-episode (`steps>0`)
failures of any class abort (mutation / re-draw risk). Non-transient quarantines
(agent Playwright timeout / benchmark / evaluator) and retry exhaustion abort —
the fail-closed safety net is preserved, not removed.

### 1.2 Structured failure provenance
`p79/experiment/environment.py` — `PaperGradeAbortError.__init__` now accepts
`transient_class` + `steps` (default None/0, back-compat). Set at the raise site
in `_run_and_record_episode_once` so the gate reads structured fields, not the
truncated message (codex F3 — string classification is fragile).

### 1.3 Transparency (P1-6, 3-AI overlap)
- Each retry → `trajectory_events.jsonl` event `transient_substrate_retry`
  (transient_class / retry_attempt / max_retries / steps_at_failure / site).
- A retry-rescued success stamps the **canonical** episode summary:
  `transient_retry_count`, `transient_retry_classes`, `is_retry_attempt`,
  `attempt_index`, `retry_trigger` (populates the EpisodeSummaryV2 attempt-lineage
  reservation, types.py). A reviewer can audit retry frequency from episode
  summaries, not only the side-channel log.
- ntfy `transient-retry` per event.
- `transient_episode_max_retries: 3` added explicitly to `configs/exp_v2_base.yaml`
  (reproducible from yaml + commit SHA; set 0 to restore legacy single-attempt).

### 1.4 Watchdog stale-ingest race close (P1-4, codex OOB)
The failed canonical summary is `unlink`-ed before the recovery backoff so the
watchdog (tracks by episode-key, not mtime — `experiment_watchdog.py:2151`)
cannot ingest the `needs_reevaluation=True` summary during the sleep window,
emit a false quarantine alert, and key-lock against the clean retry overwrite.
`steps==0` ⇒ no forensic step data is lost; the trajectory event preserves the
retry record.

## 2. What does NOT change
- H1 drop-one oracle / H3 / H10 estimand math; `scored_task_count`; observation-id
  / SoM contracts (AMENDMENT_07); eval-context; gate ladder (AMENDMENT_02).
- `_run_and_record_episode_once` body (B-168 partial recovery, B-486 quarantine,
  B-488 stale-archive, B-323 fail-loud write, B-487 covariate anchors) — unchanged.
- Non-paper-grade + diagnostic_replay paths — legacy single-attempt fail-closed.

## 3. Disclosure owed to reviewers (paper §3.5 / §8)
- State the pre-flight transient-retry policy + bound (3) + the steps==0 gate.
- Report per-cell "N episodes required a pre-flight transient retry (auth/network)".
- Note the per-model asymmetry (gemini F2): the path fires almost only for B0
  (proxy/auth-bearing); B1/B2 local rarely trigger it. Pre-flight retry is
  symmetric infra recovery (not policy-rollout resampling), but disclose it and —
  if any B0 cell has retries — report a sensitivity "zero-retry SR" (treat the
  retried episodes as the legacy abort) to show retries are not the source of any
  B0 vs B1/B2 delta.

## 4. Tests + verification
- `tests/test_b1881_transient_episode_retry.py` (18 cases): pre-flight auth/network
  retry→success; proxy_5xx never retried; mid-episode (steps>0) any class aborts;
  exhaustion / non-transient / diagnostic / dev / max=0 abort; lineage stamp;
  failed-summary unlink; classifier conservatism; legacy-no-provenance back-compat.
- Full suite: 1420 passed (incl. 18 new), 0 new regressions (1 pre-existing
  `test_section1_jaccard_null_reference_present` paper-prose failure, unrelated).
- A100 sync verified (6 files md5 match + import + provenance smoke).

## 5. Live-verification pending
The PRESERVE/retry path is live-verified on the next transient blip during the
re-launched reddit chain (auth refresh fires ~every 25min on B0 reddit → a
transient auth blip is likely within the multi-day run; expect a `transient-retry`
ntfy + a `transient_substrate_retry` trajectory event + the condition continuing
past the blip instead of aborting).

## 6. Cross-link
master_bug_catalog B-1881 · B-1880 (proxy 503 capped-backoff, the layer below) ·
B-486 / B-783 (infra-not-agent-score principle) · B-488 (stale-archive) ·
Fire-4 RCA Wave 1 M1 (the fail-closed rule this refines) · PROTOCOL_NOTE_01
(same recovery-alignment tier precedent) · 笔记 §350 · /stress 3-AI outputs
`docs/checkpoints/{codex,gemini}_outputs/b1881_transient_retry*`.

## 7. Addendum (2026-06-22, B-1883): budget retune 3 → 6 — §5 live-verified
§5's predicted live event occurred. On the resumed reddit chain (run R819) an
auth blip hit **task 143** at 2026-06-22T21:00–21:03Z (`auth_refresh
outcome=cred_wrong LOGIN_FAILED still_on_login`). The B-1881 path **engaged
exactly as designed** — runner log: `B-1881 PRE-FLIGHT transient quarantine
(class=auth steps=0) — episode-level retry 3/3 on fresh substrate` — confirming
the PRESERVE/retry mechanism fires on the right class at the right boundary.

But it **exhausted**: the 30/60/120s capped backoff at n=3 spans only a ~3.5min
absorption window, and this blip ran ~4min → overflowed by seconds → abort (the
8th reddit-chain abort; cf. 笔记 §349/§350/§352). Post-hoc the substrate was
healthy (vwa-reddit Up 26h no restart, HTTP 200, login 302, proxy 401); task 143
was classified `transient_drift` (registry, commit-synced) and re-ran clean on
resume.

**Retune**: `transient_episode_max_retries: 3 → 6` (`configs/exp_v2_base.yaml`;
supersedes the `3` recorded in §1.1 / §1.3 / §1.4). Backoff caps at 120s, so n=6
= 30+60+120+120+120+120 ≈ **9.5min** window — mirroring B-1880's proxy "wait-out"
grade for the auth class. **Estimand-safety UNCHANGED**: still `steps==0`
pre-flight only (zero contamination, agent took no action), still same-rollout
no-redraw, still proxy_5xx-excluded and mid-episode-excluded. The 3-AI /stress
consensus that cleared the *mechanism* (§2–§4) covers this *count* bump — only
the wait window lengthens; no new failure-mode surface. No test asserts the
literal `3` (the suite passes the budget as a parameter), so §4 stays green.
Recovery-alignment tier → NO OSF deposit (same as the parent note). Cross-link:
master_bug_catalog B-1883 · 笔记 §353.
