# Evaluator Change Protocol

**Purpose**: Codify how to handle changes to scoring code (VWA evaluator wrappers,
FP rules, source-level evaluator patches like B-91). Without an explicit policy,
post-lock evaluator edits become indistinguishable from p-hacking. With this
policy, every change is classified into a tier with a corresponding workflow.

**Tied to**: `preregistration.md` §4 "FP filter architecture" row (REVISED
2026-05-14 — §139.8 source-level fix supersedes the post-hoc `na_fp + eval_fp`
ladder); `osf_lock_manifest.md` (8-step DOI lock); `reeval_audit_protocol.md`
(per-episode audit trail).

**Status**: 🟢 Active (effective from 笔记 §115, 2026-05-07; B-134 banner update
2026-05-15 for §139.8 FP-architecture retire context).

⚠️ **B-134 update 2026-05-15 — current FP framework is source-level, not post-hoc**:
The Tier classification (T0/T1/T2/T3) framework below remains valid. **However**:
the protocol's body still references the OBSOLETE post-hoc `compute_adjusted_success`
layer + `na_fp + eval_fp + visual_fp` 3-layer ladder — per §139.8 (preregistration.md
§4 + Appendix A 2026-05-14) all three are retired:
- na_fp fixed at the **VWA evaluator boundary** (B-91 empty-prediction guard
  on `llm_fuzzy_match` / `llm_ua_match`, submodule `p79-patches` branch `f0c835b`)
- eval_fp `program_html` branch dropped (no scalable boundary; contamination
  prevented upstream by `RESET_BEFORE`)
- visual_fp retired earlier (2026-05-09, boundary-undecidable)
- N/A tasks excluded at task-load (`task.exclude_na_tasks: true` default)
- `compute_adjusted_success` retired from `p79/experiment/analysis.py` — raw
  `success` is canonical post-fix

Tier definitions below (§2) and the §95 reform precedent are retained for
**historical context + future evaluator changes**. Any new evaluator-code
modification still goes through this Tier classification — e.g., the B-91
patch itself was a **T0** post-§95-reform change applied pre-lock.

---

## §1 4-tier classification

Every change to scoring code (or rules-as-code) falls into exactly one tier.
Pre-lock and post-lock workflows differ; **post-lock = after OSF DOI mint**.

| Tier | Definition | Examples (real or hypothetical) |
|---|---|---|
| **T0 — Bug fix** | Evaluator logic incorrect vs. its declared spec; spec itself unchanged | §105 Magento radio swatch detection (program_html missed swatch element); evaluator string normalization fix; race-condition retry that was meant to be there |
| **T1 — FP rule expansion** | Adds a new FP category that subsumes more cases as false positive | Hypothetical: discover "url-fragment-only-match" pattern systematically inflating SR — propose new `url_fragment_fp` category |
| **T2 — FP rule simplification / merge** | Removes or merges existing FP categories | §95 deprecating `visual_fp` (no lit precedent + boundary-undecidable + over-filter); §95 simplifying eval_fp from 3-layer to 2-rule |
| **T3 — Definition change** | Redefines what "success" means at the metric level (not the rule level) | Hypothetical: change SR to "weighted SR" with task-difficulty weights; change adjusted_SR formula from `raw ∧ ¬fp` to `raw - α × fp_density` |

---

## §2 Pre-lock workflow (advisor email reply NOT yet received)

Before OSF DOI lock the project still has full flexibility. Discipline:

| Tier | Action |
|---|---|
| T0 | Fix code → `make rederive RUN=...` on affected runs → 笔记 chronicle entry tagged `[bug]` → commit message prefix `fix(eval):` |
| T1 | Add new FP category in code → run sensitivity check on archived data → discuss with advisor in next sync → preregistration.md amend (single-version, since not yet locked) → 笔记 chronicle `[design]` |
| T2 | Same as T1 (simplification is symmetric to expansion in pre-lock phase) |
| T3 | Treat as "design change" → paper_planning §3 + §19 decision log entry → advisor sync agenda item → preregistration.md substantial rewrite |

**Common requirement (all tiers)**: every change MUST appear in 笔记 with date, commit SHA, and rationale paragraph. The diff between two consecutive `evaluator_code_sha` values must be 100% explainable from chronicle entries.

---

## §3 Post-lock workflow (OSF DOI minted)

After lock, the witness chain (git tag + email + OSF DOI) freezes the spec.
Rule of thumb: **scoring code may not change in ways that change cell-level
adjusted_SR for any cell in the locked manifest**, except:

| Tier | Action |
|---|---|
| **T0** | Permitted, BUT: (a) commit message prefix `fix(eval-postlock):`; (b) paper §3 must disclose the bug + which cells affected + magnitude (raw_SR Δ + adjusted_SR Δ); (c) 笔记 entry tagged `[bug][post-lock]`; (d) `make rederive` on all affected cells; (e) re-run `preregistration_decision_test.py` and report BOTH pre-fix and post-fix decision in paper §5 (e.g., "H1: 12/16 pre-fix, 13/16 post-fix; both PASS") |
| **T1** | **Forbidden in same paper.** Either (a) drop the new FP category from this paper and report it as future work, or (b) mint OSF v2 DOI for "preregistration_v2" with the new FP category, cite both v1 and v2 in paper §3, and report decision under both rule sets |
| **T2** | **Forbidden in same paper.** Same as T1 — OSF v2 with new simplified rules, both reported |
| **T3** | **Forbidden in same paper.** Definition change is a new paper. Cite the locked v1 paper, treat T3 as future work / v2 study |

The commit-message prefix discipline (`fix(eval-postlock):` vs `feat(eval-v2):` vs `paper(metric-redefine):`) lets `git log --grep="eval"` reconstruct the change history at audit time.

---

## §4 The "evaluator_code_sha" mechanic

Every `env_snapshot.json` (auto-dumped at run start since 笔记 §114) carries:
- `evaluator_code` field with `combined_sha256` (SHA256 over concatenated content of files in `EVALUATOR_SOURCE_FILES`)
- `per_file_sha256` (so we can identify which file changed when SHAs diverge)

Lock-time procedure:
1. `python3 scripts/provenance/snapshot_env.py results/provenance/env_<host>_lock.json` (post-advisor-email)
2. Record `evaluator_code.combined_sha256` in `osf_lock_manifest.md` §2.1
3. After lock, `git diff <commit-of-lock>:p79/experiment/analysis.py HEAD:p79/experiment/analysis.py` should be **empty** unless a T0 fix was made (with proper disclosure).

Reviewer-defensible claim: "Our scoring code at OSF DOI lock had SHA `abc123...`. Three T0 fixes made post-lock are documented in paper §3.4 with magnitudes."

---

## §5 The fp_rule_version string

Embedded in every `rederive_metadata` entry (per-episode, see Protocol B):
- `§95_v2.0_na_eval` = current (post-§95 reform: na_fp + eval_fp; visual_fp deprecated)
- `§95_v2.0_na_eval_pre_§105` = pre Magento swatch fix (any rederives done with this version had T0 bug present)
- Future hypothetical: `v2.1_na_eval_url_frag` (T1 expansion adding url-fragment FP — only post-lock if OSF v2)

Bumping `fp_rule_version` in code is itself a Tier classification:
- Patch bump (v2.0 → v2.0.1) = T0 bug fix
- Minor bump (v2.0 → v2.1) = T1 expansion
- Major bump (v2.0 → v3.0) = T2 simplification or T3 redefinition

---

## §6 Decision flow

```
Found a scoring discrepancy → classify it:
├─ Spec & code agree, but real-world behavior unexpected? → T0 bug
│  └─ Pre-lock: rederive + chronicle
│  └─ Post-lock: rederive + chronicle + paper §3 disclose + double-decision report
│
├─ Need a NEW FP category to capture more FPs? → T1 expansion
│  └─ Pre-lock: amend preregistration + advisor confirm
│  └─ Post-lock: OSF v2 DOI, cite both versions
│
├─ Existing FP category broken / inflated? → T2 simplification
│  └─ Pre-lock: amend preregistration + advisor confirm + sensitivity check
│  └─ Post-lock: OSF v2 DOI, cite both versions
│
└─ Want to redefine SR / adjusted_SR formula? → T3 redefinition
   └─ Pre-lock: paper_planning major reframe + advisor sync agenda item
   └─ Post-lock: future work / v2 paper
```

---

## §7 Historical retroactive classifications

Apply this protocol retroactively to past changes for paper §3 audit trail:

| Change | Date | Tier | Notes |
|---|---|---|---|
| Initial 3-layer FP framework (§78a / §83 / §88) | 2026-04-01 to 2026-04-15 | T1 expansion (pre-lock, no advisor commitment yet) | Original na_fp + eval_fp + visual_fp design |
| §95 reform — drop visual_fp + simplify eval_fp | 2026-04-24 | T2 simplification (pre-lock) | Lit-grounded; preregistration.md updated 2026-05-07 (笔记 §115) |
| §105 Magento radio swatch detection fix | 2026-04-29 | T0 bug fix (pre-lock) | program_html eval was missing swatch elements |
| Phase A 4-cluster fix (commit `3c15cd7`) | 2026-04-30 | T0 bug fix (pre-lock) | runner dispatch / cycle / RNG; not strictly evaluator but score-affecting |
| §114 evaluator_code_sha tracking introduced | 2026-05-07 | (Infra, not eval change) | Enabling this protocol |

---

## §8 References

- 笔记 §78a (na_fp introduction)
- 笔记 §95 (FP reform — visual_fp deprecation rationale + lit anchors)
- 笔记 §105 (Magento radio swatch T0 bug)
- 笔记 §114 (provenance hardening)
- 笔记 §115 (this protocol introduction)
- `p79/experiment/analysis.py:52` (`compute_adjusted_success` canonical impl)
- `scripts/provenance/snapshot_env.py` (`evaluator_code_sha` capture)
- `docs/checkpoints/pre_run/preregistration.md` (FP filter primary spec)
- `docs/checkpoints/pre_run/reeval_audit_protocol.md` (Protocol B — episode-level audit trail)
- `docs/checkpoints/pre_run/osf_lock_manifest.md` (lock-time SHA capture)
