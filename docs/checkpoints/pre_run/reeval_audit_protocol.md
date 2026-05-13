# Re-evaluation Audit Protocol

**Purpose**: Codify the audit trail every `make rederive` invocation must leave
behind, so a reviewer asking "this episode's adjusted_SR was re-derived how many
times, by which evaluator code SHA, when, and why?" has a deterministic answer
from `summary.json` alone.

**Tied to**: `evaluator_change_protocol.md` (Protocol A — Tier classification);
`osf_lock_manifest.md` (lock-time SHA capture); `preregistration.md` (FP filter
primary spec).

**Status**: 🟢 Active (effective from 笔记 §115, 2026-05-07).

---

## §1 What gets audited

Every successful re-derive of an episode summary writes one entry to a
**list-valued** field `rederive_metadata` in `summary.json`. The list grows
monotonically — old entries are never removed, so the full re-derive history
is preserved.

```json
{
  "task_id": 0,
  "site": "classifieds",
  "raw_success": true,
  "adjusted_success": false,
  "fp_reason": "eval_fp",
  "rederive_metadata": [
    {
      "rederived_at": "2026-04-25T10:23:01Z",
      "evaluator_code_sha": "<sha256 of analysis.py + environment.py + metrics.py>",
      "fp_rule_version": "§95_v2.0_na_eval",
      "rewrite_set": ["adjusted_success", "fp_reason", "page_unchanged_rate"],
      "trigger": "rederive_episode_summary.py"
    },
    {
      "rederived_at": "2026-04-29T19:12:44Z",
      "evaluator_code_sha": "<different SHA — Magento swatch fix between>",
      "fp_rule_version": "§95_v2.0_na_eval",
      "rewrite_set": ["adjusted_success", "fp_reason", "has_effective_action"],
      "trigger": "rederive_episode_summary.py"
    }
  ]
}
```

Reviewer audit query:
```bash
jq '.rederive_metadata[] | "\(.rederived_at) [\(.evaluator_code_sha[0:8])] \(.trigger) \(.rewrite_set | join(\",\"))"' \
  results/visualwebarena/phase1/<run>/<cond>/episodes/*_summary_v2.json
```

---

## §2 Mandatory fields per entry

| Field | Type | Source | Why |
|---|---|---|---|
| `rederived_at` | ISO-8601 UTC string | `datetime.now(timezone.utc)` at write time | Temporal ordering across re-derives |
| `evaluator_code_sha` | hex SHA-256 | `hashlib.sha256(...)` over EVALUATOR_SOURCE_FILES | Identifies WHICH evaluator code wrote this re-derive |
| `fp_rule_version` | semver-like string | Hard-coded in `rederive_episode_summary.py` | Identifies WHICH FP rule lineage (e.g., `§95_v2.0_na_eval`) |
| `rewrite_set` | sorted list of strings | Caller-passed `rewrite_set` argument | Identifies WHICH fields the re-derive touched (e.g., not all re-derives touch `adjusted_success`) |
| `trigger` | string | Caller-passed | Free-form trigger identifier (`rederive_episode_summary.py`, `manual`, `migration_v3`, etc) |

**Write semantics**: append-only. If `rederive_metadata` already exists as a
dict (legacy single-entry format), promote to a list with the legacy entry as
element 0, then append. Never overwrite or truncate.

---

## §3 When re-derive is required

Re-derive is mandatory whenever any of these conditions hold post-data-collection:

| Condition | Trigger | Affected scope |
|---|---|---|
| **Scoring code change (T0 bug)** per Protocol A | Bug fix commit | All cells whose data was collected pre-fix |
| **FP rule change (T1/T2)** pre-lock | Rule amendment commit | All cells, all sites |
| **`adjusted_success` field missing** | Schema migration | Cells from before §95 reform |
| **Bug-affected derived field** (e.g., `page_unchanged_rate`, `energy_partial`) | §97 audit findings | Cells from before audit |

Re-derive is NOT done for:
- Adding new derived fields that don't change SR (e.g., adding `rederive_metadata` itself — bootstrapped lazily on next legitimate re-derive)
- Cosmetic refactors (renaming, type hints)
- Documentation-only commits

---

## §4 Re-derive idempotency

The current `rederive_episode_summary.py` writes to `episodes/.bak_pre_rederive/<orig_name>` once (one-shot, never overwrites existing backup). This Protocol B layer adds a second idempotency invariant:

**Repeated `make rederive RUN=<R>` invocations with no code change between them**:
- ✅ Episode summary unchanged (fast-path: §95 fast-path skips if `adjusted_success` already present and FP rules unchanged)
- ⚠️ `rederive_metadata` list **does grow by one entry per invocation** (each entry has the same SHA but different `rederived_at`)

If you do not want spam entries, use `make rederive` only when there's an actual reason. The Protocol A workflow makes this explicit.

---

## §5 Backup files

Original (pre-first-rederive) `summary_v2.json` → backed up to `episodes/.bak_pre_rederive/<filename>` once. Used for:
1. Disaster recovery (revert to known-good pre-rederive state)
2. Diff inspection ("what did rederive change?") via:
   ```bash
   diff <(jq -S . episodes/.bak_pre_rederive/cls_task_0_summary_v2.json) \
        <(jq -S . episodes/cls_task_0_summary_v2.json)
   ```

Backups are NEVER deleted by `rederive_episode_summary.py`. If the user wants to start fresh, manual `rm -rf .bak_pre_rederive/` is required (and not advised).

---

## §6 OSF DOI lock interaction

At OSF DOI lock (Protocol A §3, `osf_lock_manifest.md` 8-step workflow):
1. All cells in `run_manifest.yaml` with `grade=paper-grade` must have `rederive_metadata` non-empty
2. The most recent `evaluator_code_sha` across all locked cells must match the lock-time `evaluator_code.combined_sha256` from `env_snapshot.json`
3. If any cell has an `evaluator_code_sha` that differs from the lock-time SHA → MUST re-derive that cell pre-lock, OR document the discrepancy in `osf_lock_manifest.md` §2.5 with rationale

**Lock-time check script** (paper-grade gate, to be added to `queue_phase1_paper_grade.sh` Gate 7):
```bash
python3 scripts/provenance/check_evaluator_consistency.py --manifest run_manifest.yaml \
  --lock-snapshot results/provenance/env_<host>_lock.json
# Exits 0 if all cells' most-recent evaluator_code_sha == lock SHA, 1 otherwise
```

(This script is **not yet implemented** — flagged in `next_steps.md` §4 audit
follow-ups as R6.)

---

## §7 Post-lock re-derive (rare, requires Protocol A T0 disclosure)

Per Protocol A §3, T0 bug fixes ARE permitted post-lock with proper disclosure.
The re-derive workflow then becomes:

1. T0 bug fix committed with prefix `fix(eval-postlock):`
2. `make rederive RUN=<each affected run>` runs as usual; new entry in
   `rederive_metadata` shows new SHA
3. `osf_lock_manifest.md` §4 records: "Post-lock T0 fix: `fix(eval-postlock):
   <commit-msg>` on YYYY-MM-DD; affected cells: <list>; magnitude:
   ΔadjSR_max=X.Xpp"
4. Paper §3.4 prose updated to disclose
5. `preregistration_decision_test.py` re-run; both pre-fix and post-fix decision
   reported in paper §5

**The audit trail in `rederive_metadata` is what makes this defensible** — the
diverging SHA in the post-lock entry is the immutable evidence.

---

## §8 References

- 笔记 §97 (rederive_episode_summary.py introduction + `.bak_pre_rederive`)
- 笔记 §115 (this protocol introduction)
- `scripts/maintenance/rederive_episode_summary.py` (the canonical re-derive)
- `scripts/provenance/snapshot_env.py` (`evaluator_code` field in env snapshot)
- `docs/checkpoints/pre_run/evaluator_change_protocol.md` (Protocol A — Tier classification)
- `docs/checkpoints/pre_run/osf_lock_manifest.md` (lock-time SHA capture)
- `docs/checkpoints/pre_run/preregistration.md` (FP filter primary commitment)
- Pre-launch gate: `scripts/queues/queue_phase1_paper_grade.sh` Gate 7 (TODO)
