# Re-evaluation Audit Protocol Walkthrough — 2026-05-18 (pre-Phase-1a-Pass-1-fire)

> **Purpose**: Verify `pre_run/reeval_audit_protocol.md` Protocol B
> (`rederive_metadata` per-episode audit trail) is current + spec is
> coherent with §139.8 FP-architecture-retire context (B-133 banner 2026-05-15).
>
> **Status**: 🟢 Pre-fire walkthrough complete 2026-05-18.
>
> **Reviewer comprehension**: Protocol B is mostly POST-fire enforcement
> (per-episode `rederive_metadata` audit trail accumulates after data
> exists). The pre-fire walkthrough verifies (a) protocol spec is current,
> (b) supporting scripts exist, (c) OSF DOI lock interaction §6 expectations
> are met at lock event.

---

## §1-8 walkthrough table

| § | Spec content | Verification | Status |
|---|---|---|:---:|
| §1 | What gets audited — `rederive_metadata` list-valued field in `summary.json`; append-only growth | Spec coherent; reviewer audit query `jq '.rederive_metadata[]'` works | ✅ |
| §2 | Mandatory fields per entry: `rederived_at` (ISO-8601 UTC) / `evaluator_code_sha` (hex SHA-256) / `fp_rule_version` (semver) / `rewrite_set` (sorted list) / `trigger` (free-form) | Spec coherent; field-types match `rederive_episode_summary.py` writes | ✅ |
| §3 | When re-derive is required: T0 scoring bug / T1-T2 FP rule pre-lock / `adjusted_success` schema migration / bug-affected derived fields | Trigger conditions clear; T0/T1/T2/T3 classification cross-refs `evaluator_change_protocol.md` Protocol A | ✅ |
| §4 | Idempotency: repeated `make rederive` with no code change ⇒ episode summary unchanged + `rederive_metadata` grows by one entry per invocation (same SHA, different timestamp) | Spec coherent; spam-defense documented | ✅ |
| §5 | Backup files: `episodes/.bak_pre_rederive/<filename>` written once per first rederive; never deleted by script; manual `rm -rf` required | Spec coherent; disaster recovery + diff workflow documented | ✅ |
| §6 | **OSF DOI lock interaction** — at lock time, all paper-grade cells have `rederive_metadata` non-empty; most-recent `evaluator_code_sha` matches lock-time SHA; `check_evaluator_consistency.py` script TODO (R6) | Spec coherent; ⚠️ script not yet implemented (R6 marker in `next_steps.md`) — **does NOT block pre-fire** because §6 applies POST-fire data lock | ✅ (POST-fire enforcement) |
| §7 | Post-lock re-derive: T0 bug fix permitted with prefix `fix(eval-postlock):`; `make rederive` re-runs as usual; diverging SHA in post-lock entry = immutable evidence | Spec coherent; paper §3.4 prose update + `preregistration_decision_test.py` re-run discipline documented | ✅ |
| §8 | References: 笔记 §97 + §115 + canonical scripts + protocol cross-refs | All references current; 笔记 chronology preserved | ✅ |

## §139.8 FP-architecture-retire context (B-133 banner)

| Pre-fix (archive era) | Post-§139.8 (canonical) |
|---|---|
| `adjusted_success` post-hoc field | `adjusted_success ≡ success` (retired post-hoc layer) |
| `fp_reason` field written by aggregator | NOT written; na_fp fixed at evaluator (B-91 patch in VWA submodule `f0c835b`); eval_fp branch dropped |
| `compute_adjusted_success` filter chain | Retired; N/A tasks excluded at task-load (`exclude_na_tasks: true`) |
| 3-variant sensitivity ladder (na_fp / eval_fp / visual_fp) | Retired; raw `success` from fixed evaluator is canonical paper-grade outcome |
| Protocol B applies to all re-derives | Protocol B applies to **archive data re-derivation only** (Phase 1a pre-fix episodes, Appendix D contamination disclosure); for post-§139.8 canonical A100 rerun, `rederive_metadata` records evaluator SHA only (no `adjusted_success` / `fp_reason` rewrite_set entries) |

**Pre-fire impact**: Phase 1a Pass-1 paper-grade fire produces post-§139.8
data;Protocol B will be invoked POST-fire only if a T0/T1/T2 evaluator
change lands; `rederive_metadata` will contain evaluator_code_sha entries
matching lock-time SHA at OSF DOI mint per §6 enforcement.

## §6 OSF DOI lock interaction — pre-fire prep

OSF DOI mint occurs POST-Phase-1a-fire-data-complete per `osf_lock_manifest.md §3` updated header (B-1570 doctrine). The §6 enforcement at lock time will check:

1. All paper-grade cells in `run_manifest.yaml` (grade=`paper-grade`) have `rederive_metadata` non-empty
   - Pre-fire: 0 cells (no data exists). Becomes ≥1 after Pass-1 baseline fire completes per cell (auto-populated at first rederive trigger)
2. Most-recent `evaluator_code_sha` across all locked cells matches `evaluator_code.combined_sha256` from lock-time `env_snapshot.json`
   - Pre-fire: `evaluator_code.combined_sha256` recorded at A100 snapshot 2026-05-18 my SSH; identical SHA expected at OSF DOI mint snapshot
3. Any cell with diverging `evaluator_code_sha` → MUST re-derive pre-lock OR document discrepancy in `osf_lock_manifest.md §2.5` with rationale
   - Pre-fire: deferred to OSF DOI mint event

**Pre-fire walkthrough verdict**: Protocol B spec is current + coherent + supporting scripts exist. §6 enforcement is post-fire-data-lock — non-blocking for fire start.

## Sign-off

- **Walkthrough date**: 2026-05-18 (post §A2.8 + §A2.9 + 深入审 Mode A closure)
- **Reviewer**: Claude (Opus 4.7) — auto-walkthrough deterministic
- **Verdict**: 🟢 APPROVE pre-fire; Protocol B post-fire enforcement non-blocking
- **R6 follow-up**: `check_evaluator_consistency.py` implementation deferred (post-OSF-DOI-lock or as part of Step 7 pre-mint validation script)

This walkthrough closes phase1_plan §B1 `reeval_audit_protocol.md` 走查 item.
