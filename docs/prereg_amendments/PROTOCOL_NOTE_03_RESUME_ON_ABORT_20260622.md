---
title: Resume-on-abort for independent-task sites (B-1882 / B-304 exception) — reddit resume is estimand-NEUTRAL
status: WITNESS — mechanism B-1882 (commit 42c1bbd) + this note; live-verified 3× (R819 abort#7/#8 resume, R11344 dom, R32139 phantom_text task-80 resume_rerun_clean 2026-06-30). Estimand UNCHANGED (recovery-alignment tier, like NOTE_01/02).
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
  - PROTOCOL_NOTE_02_TRANSIENT_PREFLIGHT_RETRY_20260621
witness_tag: protocol-note-03-resume-on-abort-20260622   # set at finalizing commit
osf_deposit: NOT required (recovery-alignment tier, estimand-neutral; same as NOTE_01/02). Advisor may note at next sync.
decided_by: user 2026-06-22 (resume-on-abort over chunking, after Explore confirmed reddit task independence); re-affirmed 2026-06-30 (R32139 resume)
cross_ai_audit: B-1882 mechanism unblock = latent dead-code fix (no separate /stress). The resume-on-abort POLICY rests on the empirical task-independence premise (Explore on test_reddit.raw.json, 0 cross-task post-ID collisions) — this note IS the witness PROTOCOL_NOTE_02 §7 / B-1882 deferred.
---

# PROTOCOL_NOTE_03 — Resume-on-abort for independent-task sites (reddit)

## 0. Scope — why "PROTOCOL_NOTE", and why estimand-NEUTRAL

When a paper-grade reddit condition aborts mid-run (proxy 503 outage, auth blip),
the recovery question is: **discard the whole condition and re-run from episode 0
(FORCE_NEW), or resume from the breakpoint keeping the already-completed tasks?**

This note witnesses the policy that **reddit resumes** rather than FORCE_NEW. Like
PROTOCOL_NOTE_01/02 (and *unlike* NOTE_04, which *defines* reddit's estimand),
this is **recovery-alignment tier: the measured quantity is UNCHANGED**. For a
site whose tasks are independent, resume and FORCE_NEW measure the same per-task
quantity — resume only changes *which run_id* carries the data and *how much
machine-time is wasted*, not *what is measured*. The fix is therefore a
non-OSF-deposit note, auditable via the quarantine registry + the resume markers
in the run directory.

## 1. The default rule and its premise (B-304)

The project default is **FORCE_NEW** on any abort: the interrupted condition is
archived whole and re-run from ep0 with a fresh run_id. The rationale (B-304):

> Resuming a condition mid-stream mixes two reset baselines — the first half of
> the tasks ran on "the site state after the condition-opening reset", the second
> half on "the accumulated state at the abort point". If tasks depend on each
> other, that baseline discontinuity contaminates per-task measurement → not
> paper-grade.

**B-304's premise** is that **tasks within a condition have state dependencies** —
a later task consumes state a earlier task created. Only under that premise does
"two mixed baselines" actually corrupt the measurement.

## 2. Why reddit is the witnessed exception (empirical task-independence)

The premise is **falsified for reddit** by direct inspection of the task
definitions (`external/visualwebarena/.../test_reddit.raw.json`, 210 tasks,
Explore audit 2026-06-22):

- **0 cross-task post-ID collisions** — no task references an object ID created
  by another task.
- Every reddit task is **find-only** (locate an already-existing post / comment /
  user) or **create-only** (submit a post / comment / DM). There is **no "find
  the post I just submitted"** chain.

i.e. **reddit tasks do not consume each other's created state — they are
independent.** Therefore:

- reset granularity / accumulated site state has **no effect** on per-task success;
- the "two mixed baselines" discontinuity B-304 guards against **cannot arise**
  (there is no cross-task dependency to make it matter);
- resume (keep completed tasks, re-run only the breakpoint) is **statistically
  equivalent** to FORCE_NEW-from-ep0 for reddit.

Independent corroboration: VWA's per-task `require_reset` is **only implemented
for classifieds** (`browser_env/envs.py:172` `TODO(jykoh)` for reddit/shopping).
So reddit tasks *already* run on accumulated state as their **normal regime** —
resume introduces no new inconsistency that wasn't there in every reddit run.

Conversely, forcing FORCE_NEW on reddit is **pure waste** (discards dozens of
completed episodes — worst case 135 ep / ~16h, R819) and is **no cleaner**,
because there is no cross-task dependency to purify.

## 3. Mechanism + safety — resume is not naked continuation

| Mechanism | Role |
|---|---|
| **B-1882** (mint v2-marker) | `mint_run_id` with FORCE_NEW=0 correctly recognizes the aborted run as a resume candidate via the v2-filename marker (`condition_summary_v2.json` / `episodes/*_summary_v2.json`), instead of minting a fresh id and losing progress. *Was latent dead-code* (the canonical chain always exports FORCE_NEW=1) until reddit resume surfaced it. |
| **B-486** (force-rerun breakpoint) | The task interrupted by the abort has an untrustworthy partial trajectory → its `needs_reevaluation` flag forces a clean re-run, the half-episode is never reused. |
| **B-488** (forensic archive) | That **one** task's stale partial is `mv`'d to `.stale_*` (never deleted). The already-completed tasks' summaries are untouched — this is the literal "don't archive the whole run" property. |
| **Fix 4 / NOTE_04** (identity restore) | Each reddit task restores the seed username at episode start, so a resume that crosses task 138 (the destructive username-rename) does not re-trigger the auth self-destruct cascade. |

So resume re-runs exactly the breakpoint task on clean substrate, preserves all
prior completed tasks, and is protected from the one reddit-specific cross-task
hazard (task 138) by an orthogonal mechanism.

## 4. Scope boundary + live-verification

**Boundary (do not over-generalize):** this exception is **reddit-only**.
**classifieds and shopping must still FORCE_NEW** — cls has per-task `require_reset`
and potential task dependencies; shopping accumulates cart/wishlist state. Their
B-304 premise is **not** falsified. B-1882 only *unblocks* the resume path; this
note + the task-independence evidence decide *when* it is licensed (reddit only).

**Live-verification (3 events, all clean):**
1. **R819** (B0 dom, 2026-06-22) — abort#7 (proxy 503 @task139) + abort#8 (auth
   blip @task143): resume from breakpoint, B-486 re-ran the breakpoint task to a
   legitimate episode each time; ~135 ep / ~16h preserved.
2. **R11344** (B0 dom, 2026-06-25→27) — reddit's first-ever bound-clean
   paper-grade condition; crossed task 138/151 with no abort (Fix 4 + resume both
   in force).
3. **R32139** (B0 phantom_text, 2026-06-30) — abort (proxy 503 mid-episode
   @task80) → resume → B-486 re-ran task 80 to a legitimate episode
   (`error=None`, `needs_reevaluation` cleared) = **`resume_rerun_clean`**, the
   strongest in-situ falsification of a task/evaluator/substrate fault. 76 ep
   preserved; task 80 classified `transient_drift` via that evidence; Gate G8
   re-opened (0 unclassified). See quarantine_registry.jsonl + 笔记 §358-359.

## Cross-link

B-1882 (mint resume mechanism, commit `42c1bbd`); B-304 (FORCE_NEW default — this
is its independent-task exception); B-486 (needs_reevaluation force-rerun); B-488
(forensic archive of breakpoint partial); B-1880/B-1881/B-1883 (the abort-causing
transient stack resume makes cheap); PROTOCOL_NOTE_04 / Fix 4 / B-1884 (identity
restore — the orthogonal task-138 protection that makes crossing-138 resumes
safe); PROTOCOL_NOTE_01/02 (recovery-alignment siblings, estimand-unchanged like
this one); 笔记 §352/§353 (resume-on-abort strategic turn + abort#7/#8 chronicle),
§358 (R11344 land + first reddit /diag); R32139 resume + this note = finalizing
event (笔记 §359 chronicle pending).
