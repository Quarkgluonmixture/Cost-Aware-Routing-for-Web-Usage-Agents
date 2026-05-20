# Separate-session task: fix 31 pre-existing red tests (fixture drift)

## Context
P79 paper-grade VWA cost-aware-routing project. Multiple Claude Code sessions
have been running concurrently doing Fire-4/5/6 RCA + paper-grade hardening.
A main session just landed Fire-6 RCA Stage C1 (evaluator isolation, commit
`3c767a2` on origin/master). While verifying C1 it discovered a **pre-existing
test-debt of 31 red tests** that is orthogonal to C1 and needs a focused cleanup.

## The problem
`pytest tests/` → **31 failed, 1040 passed, 10 skipped**. All pre-existing
(NOT caused by C1 — C1's own 26 tests pass). Two root causes, both from
already-pushed commits:

1. **(25 of 31) step-record fixture drift** — `StepRecordV2 missing paper-grade
   critical optional keys: ['counted_as_agent_action', 'intervention_from_url',
   'intervention_recovery_url', 'intervention_type']`. These 4 Phase-2 step
   intervention fields were added to `PAPER_GRADE_STEP_OPTIONAL_KEYS` in commit
   `8d2a327` but ~25 test step-record fixtures were never backfilled. The
   validator (`p79/experiment/types.py:validate_step_record_v2`) requires KEY
   presence (value may be None) for paper-grade evidence contract (B-732/B-280).

2. **(~6 of 31) episode-summary fixture drift** — `_avg('total_latency_minus_retry_ms'):
   no episode populates field` from commit `e20c6ef` (`_avg require_present`
   in `p79/experiment/metrics.py`). Episode-summary fixtures passed to
   `aggregate_condition_metrics` lack `total_latency_minus_retry_ms`.

## The 12 affected test files
```
tests/test_phase1_prereg_gate.py                 (1)
tests/test_runner_integration.py                 (1)
tests/test_step_record_validation.py             (1)
tests/test_stress_a1_3_fixes.py                  (1)
tests/test_stress_a1_4b_i_g2_fixes.py            (1)
tests/test_stress_a1_4b_ii_g3_fixes.py           (1)
tests/test_stress_a1_4b_ii_g4_fixes.py           (3)
tests/test_stress_a1_6b_fixes.py                 (6)
tests/test_stress_a1_7_fixes.py                  (1)
tests/test_stress_a1_8_cold_start_fixes.py       (9)
tests/test_stress_a1_9_fixes.py                  (3)
tests/test_stress_a2_5_runtime_h10.py            (3)
```

## The fix (DO NOT touch source schema — it is correct)
The schema requirement is intentional (paper-grade evidence contract). ONLY
fix the test fixtures. **Robust pattern (already applied in 5 fixtures by the
main session — use as reference):** derive fixtures from the canonical
defaults so they auto-sync with future schema additions instead of hardcoding
field lists that drift.

Reference the already-fixed pattern in:
- `tests/test_stress_a1_8_cold_start_fixes.py::_minimal_valid_episode_summary`
  (now `base = dict(EPISODE_SUMMARY_V2_DEFAULTS); base.update({...})`)
- `tests/test_stress_a1_9_cold_start_fixes.py::_valid_episode` (same pattern)
- `tests/test_som_and_schema.py` (step-record fixtures: added the 4
  intervention fields explicitly)

**Recommended approach (best — kills the whole class of drift):**
1. Create `tests/conftest.py` (or a `tests/_fixtures.py` helper module) with:
   ```python
   def complete_step_record(**overrides):
       from p79.experiment.schema_migrations.v2 import STEP_RECORD_V2_DEFAULTS
       rec = dict(STEP_RECORD_V2_DEFAULTS)
       rec.update(overrides)
       return rec

   def complete_episode_summary(**overrides):
       from p79.experiment.schema_migrations.v2 import EPISODE_SUMMARY_V2_DEFAULTS
       ep = dict(EPISODE_SUMMARY_V2_DEFAULTS)
       ep.update(overrides)
       return ep
   ```
2. Retrofit the 12 files' step-record / episode-summary fixtures to start from
   these helpers + `.update()` the test-specific values. This guarantees ALL
   paper-grade keys present + never drifts again.

**Minimal alternative (faster but drift-prone):** add the 4 step fields
(`intervention_type`, `counted_as_agent_action`, `intervention_from_url`,
`intervention_recovery_url` — all `None`) to each failing step-record fixture,
and `total_latency_minus_retry_ms` to each failing episode-summary fixture.

## Verification
```bash
cd /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
.venv/bin/python3 -m pytest tests/ -q   # must be 0 failed
```
Also confirm the schema 4-place sync invariant still holds:
```bash
.venv/bin/python3 -m pytest tests/test_schema_4place_sync.py -q
```

## Commit + push
Single focused commit, e.g.:
`test(fixture-drift) backfill 31 step/episode fixtures via derive-from-DEFAULTS`
Then `git push origin master`. Coordinate: main session is on the same repo;
only touch `tests/` files (main session owns `p79/experiment/` source + is
working on C2/C3 runner diagnostic-replay). Pull before push.

## Scope guard
- Touch ONLY `tests/` files (+ optionally new `tests/conftest.py`).
- Do NOT modify `p79/experiment/types.py`, `metrics.py`, `schema_migrations/v2.py`
  (schema is correct; main session may also be editing source).
- ~30-45 min mechanical work.
