---
title: Reddit per-task identity restore (B-1884 / Fix 4) — reddit estimand DEFINED = clean per-task
status: WITNESS — code change + master_bug_catalog B-1884 + tests; A100 live-verification PASSED 2026-06-25 (two-column restore + fresh login OK). Fire launch awaits user go.
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
  # PROTOCOL_NOTE_03 reserved (resume-on-abort policy witness, B-1882, pending)
witness_tag: protocol-note-04-reddit-identity-reset-20260625   # set at finalizing commit
osf_deposit: RECOMMENDED at next advisor sync — this DEFINES the reddit measurement (Option a); see §0
decided_by: user 2026-06-25 ("不问学长了，直接按照之前推荐的做吧" — proceed with the recommended option without an advisor gate)
cross_ai_audit: prior fix-space audit = codex 2-round (ungrounded → file-grounded, 笔记 §357.3/§357.4); claim-by-claim /stress NOT re-run (mechanical substrate-restore, not a new claim)
---

# PROTOCOL_NOTE_04 — Reddit per-task identity restore (B-1884 / Fix 4)

## 0. Scope — why "PROTOCOL_NOTE", and the honest estimand caveat

Unlike PROTOCOL_NOTE_01/02 (which were strictly *recovery-alignment, estimand
UNCHANGED*), this note **defines how reddit is measured**. The root cause
(B-1884): reddit **task 138** ("Change my username to …") is a destructive task
— a capable model renames the shared test account (postmill `users.id=13915`,
`MarvelsGrantMan136` → e.g. "Patrick"). Since the username IS the login
credential, P79's periodic fresh re-login (`auth_refresh.py`, every ~5 episodes,
to survive the 24-min `gc_maxlifetime`) then fails and the whole reddit condition
fail-closes. Within a condition, every task after 138 inherits a corrupted
account.

**The choice.** There were two defensible estimands (笔记 §356.3): **(a)** reset
the corruption between tasks → measure *clean per-task* capability (matches the
project's per-task representation/routing comparison), vs **(b)** tolerate it →
measure capability under self-mutated state (deployment realism, ≈ what upstream
VWA effectively does by reusing one cookie). The user chose **(a)** on
2026-06-25.

**Why this is still PROTOCOL_NOTE tier, not AMENDMENT_08:**
1. **No bound data changes.** Reddit has produced **zero** clean paper-grade
   runs (the abort saga = never finished). We are *defining* reddit's substrate
   handling for its first run, not re-defining a measured quantity. cls (the
   only site with bound data) is **untouched** — the hook is gated on
   `task.site == "reddit"`; cls's already-bound DOM/SoM runs keep their exact
   behavior (cls auth was clean, 44/44 OK, R21557).
2. **Brings reddit to parity with the registered protocol intent.** The prereg
   registered "per-condition site reset"; upstream VWA additionally gives cls a
   *per-task* `require_reset` but left reddit's reset unimplemented
   (`browser_env/envs.py:172` `TODO(jykoh)`). Fix 4 supplies the missing
   per-task substrate restore for reddit — the same *clean-per-task* semantics
   cls already had — rather than introducing a novel measurement regime.
3. **Setup phase, not measured execution.** The restore runs at the **start** of
   each reddit task, *before* the auth-refresh check and *before* the agent's
   trajectory (`runner/main.py:_run_episode`, ahead of the `should_refresh`
   block). It mutates no in-trajectory state; latency/cost accounting is
   untouched.

**Honest caveat (do not hide):** option (a) **neutralizes** a real confound —
*capability-modulated self-contamination* (only a model strong enough to finish
task 138 corrupts its own downstream tasks; weaker models fail it and run on
unharmed). This is reported as a **separate standalone finding** (§8), not
silently erased. Advisor may upgrade this to AMENDMENT_08 at next sync if they
prefer option (b) or want the choice formally deposited; until then the
disclosure below + the idempotent-restore telemetry make it fully auditable.

## 1. The change (code)

- `p79/utils/reddit_identity.py` (new): `restore_reddit_identity(cfg)` runs an
  **idempotent** `UPDATE users SET username='MarvelsGrantMan136' WHERE id=13915
  AND username<>'MarvelsGrantMan136'` (no-op when already correct) via the
  **verified** DB path (笔记 §354 cross-system audit): `docker exec vwa-reddit
  bash -lc "su - postgres -c 'psql -d postmill -c <sql>'"` (unix-socket peer
  auth; the password `-U` path is not provisioned in the image). All targets
  (container/db/table/column/id/seed-username/SQL-override/timeout/fail_closed)
  are config-overridable.
- `p79/experiment/runner/main.py`: calls it at the top of `_run_episode` for
  `task.site == "reddit"`, **before** the auth-refresh check — so a renamed
  account is healed before the next fresh re-login (timing-critical: auth-refresh
  is at the same method's `should_refresh` block, which executes *after* this
  call).
- `p79/experiment/config.py`: `reddit_identity_reset` DEFAULT_CONFIG block.
- `tests/test_reddit_identity.py`: 12 tests (idempotent SQL, injection-safe
  quoting, verified docker argv shape, enabled gate, soft-fail vs fail-closed).
  Full suite: runner smoke + new = 19 passed, no regression.

Quoting verified byte-exact: a `shlex.split` simulation of both shell layers
(`bash -lc` → `su -c`) confirms psql receives the SQL string unchanged.

## 2. Failure semantics

Default `fail_closed=False`: a transient `docker exec` hiccup logs a warning and
continues (a one-off failed restore at worst resurfaces the old symptom, which
the existing abort-recovery stack handles — it does NOT silently corrupt a
scored outcome, because the restore is pre-trajectory). Set
`reddit_identity_reset.fail_closed=true` to make a failed restore raise. On a dev
box without the container the restore soft-fails (set `enabled=false` to
silence).

## 3. Disclosure (paper §3.5 / §8)

- §3.5: "Reddit's shared test account is restored to its seed username at each
  task boundary (idempotent DB UPDATE), supplying the per-task clean-state that
  the upstream harness implements for classifieds but not reddit. This isolates
  per-task capability from the destructive task 138 (username rename)."
- §8: report (i) the deviation from pure cookie-reuse VWA behavior; (ii) the
  capability-modulated self-contamination finding that option (a) controls for;
  (iii) that the restore is idempotent and pre-trajectory.

## 4. Live-verification — PASSED on A100 (2026-06-25)

Ran `scripts/maintenance/verify_reddit_identity_fix.sh` on A100 (`a100-jiaming-test`,
repo `/home/ubuntu/workspace/p79`). **The verification caught a real bug in the
first draft and forced a fix:** postmill's `users` table carries a lowercase
canonical `normalized_username` column and **login matches against IT, not
`username`**. A `username`-only restore therefore would NOT have restored login
— the exact "canonical column" risk flagged here. (A first run gave a misleading
LOGIN_OK because the *simulation* only renamed `username`; correcting the
simulation to rename BOTH columns — as postmill's real `setUsername()` does —
exposed the gap.) `restore_reddit_identity` + the default SQL now restore BOTH
columns.

End-to-end result (deployed code, realistic two-column rename):
- baseline `13915|MarvelsGrantMan136|marvelsgrantman136`
- simulate real task-138 rename → `13915|Patrick|patrick`
- `restore_reddit_identity({})` → `True`; both columns back to seed
- **fresh login as MarvelsGrantMan136 → LOGIN_OK**

14 unit tests pass on A100; full local suite (runner smoke + new) = 19 pass, no
regression. **Remaining: operator `launch red`** (FORCE_NEW from ep0, B-304;
same-site-collision check) — awaits user go, not a code gate.

## Cross-link

B-1884 (root cause + fix); 笔记 §354/§355/§356/§357 (diagnosis + fix-space →
Fix 4 convergence); `envs.py:172` (`TODO(jykoh)` reddit reset gap); B-1839
(per-condition docker restart — the per-condition layer this complements);
PROTOCOL_NOTE_01/02 (recovery-alignment siblings; this one differs by defining
the reddit estimand); cls R21557 (44/44 auth OK = cls account never renamed →
cls untouched by this).
