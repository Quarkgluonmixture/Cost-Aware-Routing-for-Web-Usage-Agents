# Resume prompt: Fire-6 RCA Stage C2 (diagnostic replay) + C3 (full Fire-6)

> ✅ **C1 (3c767a2) + C2 (9d46134) 已 land**（2026-05-20）。本文档仅 C3（full Fire-6 launch）仍 pending = live-run gate，非文档任务；forward action 见 [next_steps.md](next_steps.md)。下方 C1/C2 段落仅留作 RCA 记录。

## Where we are (2026-05-20, post-compact resume)
Fire-6 RCA gamma-staged plan (user-approved). **C1 DONE** (commit `3c767a2`
on origin/master): L2 program_html evaluator isolation + timeout
instrumentation + 6 eval-context metadata fields + 11 tests. **C2 + C3
remain.**

## The 3-fire root cause (settled)
Fire-3 (cls task 75 edit), Fire-4 (cls task 75 edit), Fire-5 (cls task 4 edit)
all died: agent enters a stateful EDIT form (154KB DOM, 2582 inline city/state
`<option>`), then the evaluator's `program_html` `page.goto(item_url)` reused
the agent's 25-min-cumulative-state runner page → 30s × 3 timeout →
EvaluatorUnavailableError → condition abort. Path Z (Playwright MCP, authed)
proved a FRESH page does the same goto in 639ms. C1 fixes this: program_html-
safe tasks now run eval on a fresh isolated page from the start.

## C2 — targeted diagnostic replay (NEXT, user-approved design)
Goal: prove C1 fixes the evaluator timeout on task 4 + 75 WITHOUT a blind full
Fire-6. This also provides the **matched-temporal-context reproduce** that
Gate 8 cross-fire-recurrence Rule 2 requires to unblock task 75.

Design (user constraints — STRICT):
- New runner mode `--diagnostic-replay --tasks 4,75` (cls).
- Output to **non-canonical dir** `results/diagnostic_replay/<ts>/` (NOT
  `results/visualwebarena/phase1/`).
- Every episode stamped `diagnostic_replay=True, sr_excluded=True` (never
  enters paper §1 SR).
- Full forensic logging (the C1 `_dump_eval_timeout_forensic` + eval-context
  metadata fields).
- **Gate 8 override ONLY for diagnostic replay**: a flag like
  `QUARANTINE_DIAGNOSTIC_REPLAY=1` that works ONLY in combination with
  `--diagnostic-replay` + explicit `--tasks` + non-canonical output path +
  `sr_excluded=True`. It must NOT become a canonical Gate-8 bypass (user
  decision: never disable Gate 8 in canonical mode).
- Implementation layers: runner CLI flag (`scripts/run_experiment.py` /
  `p79/cli/run_experiment.py` / `p79/experiment/runner/main.py`), a thin
  diagnostic-replay queue wrapper, Gate 8 override in
  `scripts/queues/_lib_paper_grade_gates.sh` / `quarantine_registry.py`
  preflight gated on the diagnostic env flag.

C2 run: launch diagnostic replay of cls task 4 + 75 on A100. Expected: C1
isolation → eval Page.goto succeeds (~639ms, no timeout). If it succeeds,
classify task 4 + 75 via `matched-temporal-context reproduce` in the
quarantine registry (this satisfies Gate 8 Rule 2). If it STILL times out,
the C1b forensic dump captures the mid-fire mechanism → iterate.

## C3 — full Fire-6 (ONLY after C2 proves fix)
After C2 diagnostic replay shows task 4/75 evaluator timeout is fixed AND
task 75 is classified via matched-temporal-context (Gate 8 Rule 2 cleared):
- Full canonical `bash scripts/queues/queue_phase1_paper_grade.sh launch`
  (sequential cls→red, 16h B0 budget, all Wave 1-5 + C1 fixes active).
- Pre-fire: A100 `git pull`, clean `.locks/`, verify 8 preflight gates + Gate 8.

## Current infra state (verify on resume — may have changed)
- **A100** (`condense-a100` via SSH): Fire-5 aborted + fully cleaned (0 P79
  procs, .locks empty as of 2026-05-20 ~01:00). cls docker healthy (Up 3 days).
  Disk 92% (structural — VWA docker images 403GB, not cleanable, fine).
- **DGX→A100 SSH port forward** likely still up (`pgrep -af "9980:localhost:9980"`):
  `ssh -L 9980:localhost:9980 -L 9999:localhost:9999 -L 7770:localhost:7770 -N -f condense-a100`
- **Playwright MCP** available (chromium-1223, `--headless --isolated` in
  `~/.claude.json`). cls creds: `blake.sullivan@gmail.com` / `Password.123`.
  Login URL `http://localhost:9980/index.php?page=login`.
- **Gate 8 status**: task 75 [cross_fire_recurrence] HALTs Fire-6 (2 fires,
  needs matched-temporal-context reproduce — C2 provides this). task 4
  classified `unreproducible_in_isolation` (1 fire, not recurrence-blocking).

## Parallel-session test-debt (DO NOT block on it)
31 pre-existing red tests (fixture drift from 8d2a327 step fields + e20c6ef
latency field) are being fixed in a SEPARATE session — see
`docs/checkpoints/handoff_fix_31_red_tests.md`. NOT C1-caused. Don't fix here
unless that session didn't finish; if touching, only `tests/` files.

## User's standing constraints (gamma-staged)
- Do NOT launch full Fire-6 until C2 diagnostic replay proves evaluator
  timeout fixed.
- Gate 8 override is diagnostic-replay-scoped ONLY (non-canonical, task-scoped,
  logged, sr_excluded). Never a canonical bypass.
- Push is authorized session-wide (user said "整个 session 都允许 push").
- /stress cross-AI before milestone commits (per CLAUDE.md auto-trigger).

## Immediate next action on resume
Implement C2 diagnostic-replay mode (runner flag + non-canonical output +
sr_excluded + diagnostic-scoped Gate 8 override), then run it on cls task 4+75
on A100, confirm C1 isolation eliminates the eval timeout.
