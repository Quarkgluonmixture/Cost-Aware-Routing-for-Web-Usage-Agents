---
title: Analysis-layer estimand conformance restoration (toolchain audit P0-1/2/3/4 + P1-4) — estimand UNCHANGED, implementation corrected to locked text
status: WITNESS — outcome-blind at k=3 interim (reddit verdict cells not landed); prereg text UNCHANGED; canonical artifacts NOT regenerated in this change (dry-run delta disclosed below)
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
  - PROTOCOL_NOTE_03_RESUME_ON_ABORT_20260622
  - PROTOCOL_NOTE_04_REDDIT_IDENTITY_RESET_20260625
witness_tag: protocol-note-05-analysis-estimand-conformance-20260714   # set at finalizing commit
osf_deposit: not required (recovery-alignment class, estimand UNCHANGED; disclose at next advisor sync)
decided_by: autonomous codex-loop session 2026-07-14 (user standing directive "codex 不停针对 repo 做所有有意义的事"); user review pending before push
cross_ai_audit: source = codex (gpt-5.6-sol xhigh) verdict-toolchain correctness audit 2026-07-14 (9 P0 / 5 P1 / 2 P2, Stop-ship); 4 highest-impact P0 independently code-verified by Claude before fixes; prereg text read as normative anchor for fix direction
---

# PROTOCOL_NOTE_05 — Analysis-layer estimand conformance restoration

## 0. Scope — strictly recovery-alignment, estimand UNCHANGED

Like PROTOCOL_NOTE_01/02 (and unlike PROTOCOL_NOTE_04), this note changes **no
estimand**. It corrects analysis-producer implementations that had *deviated
from the already-locked preregistration text*, discovered by a hostile
correctness audit of the verdict-day toolchain
(`docs/checkpoints/codex_outputs/verdict_toolchain_review_2026-07-14.md`).
`scripts/analysis/` is not on the fire import path (fire-immutability memo
2026-05-24); the running A100 fire is unaffected.

**Timing / outcome-blindness.** At the time of this change the H1/H3 pools are
k=3 interim (classifieds only); no reddit verdict cell has landed and no final
verdict has been computed. Fixing now is the outcome-blind window — deferring
until after verdict data lands would make any correction outcome-dependent.

## 1. Deviations corrected (normative anchor = preregistration.md locked text)

| # | Audit ID | Deviation (implementation) | Locked text (prereg) | Correction |
|---|---|---|---|---|
| 1 | P0-4 | `_h3_axis_pooled_fe` removed cells with `n_unique < 2` from the FE pool input, then pooled the remainder (outcome-dependent deletion; live in current artifact: axis-1 pooled k=2 of 3) | H3(i)/(ii): FE pooled mean "**over the 6 planned cells**"; H3(iii) ≥2-task floor applies to "**cell-level pass**" only; A1.21 degenerate rule = SE floor 1.0pp on SE≤0 | Pool ALL data-bearing planned cells; `n_unique<2` → `cell_pass=false` label + transparency counts only; SE floor unchanged (fires only on SE≤0, `n_zero_se_floored_cells` emitted); axis verdict evaluated ONLY at `k_cells_input == 6`, else `NOT_EVALUATED` |
| 2 | P0-2 | H1 kernel accepted any ≥50-task six-mode intersection as a "complete cell" and used the data-dependent intersection as denominator | Operational scored universe = classifieds **224** / reddit **205** exact (prereg §4; `scored_task_count` single source) | Per-mode `observed_ids == expected_scored_ids(site)` enforced fail-closed; `missing_ids`/`extra_ids`/`task_set_sha256` persisted; bootstrap universe = expected scored set |
| 3 | P0-3 | SR producer `complete = n_total >= expected_n`; SR = `n_success/n_total` (vintage 210-task dirs pass as complete with wrong denominator) | Same canonical scored universe | Exact-set equality; SR denominator fixed to expected scored N; set diffs + sha persisted; schema `v3-2026-07-14-exact-task-set` |
| 4 | P0-1 | Producers emit `gate_status ∈ {PASS, FAIL, PARTIAL_DATA, INSUFFICIENT_DATA}` while consumers (slotsheet/F2/runbook) test `== "COMPLETE"` — final artifacts would forever read as interim; partial sheets still printed branch advice | (schema hygiene; no estimand content) | Orthogonal fields added: `analysis_status ∈ {COMPLETE, PARTIAL, INSUFFICIENT}` + `h1_verdict ∈ {PASS, FAIL, NOT_EVALUATED}`; legacy `gate_status` retained verbatim for back-compat until consumers migrate (Chunk 2) |
| 5 | P1-4 | H3-insufficient wrote fake `0` numerics to CSV; MD writer KeyError on skip-dict; CSV→JSON→MD sequential overwrite could leave half-written artifacts | (robustness) | Blank/null numerics for unavailable values; explicit status-union MD branch; all three outputs staged + atomically `os.replace`d as a transaction |

## 2. Disclosed interim-number impact (dry-run, canonical artifacts NOT overwritten)

Recomputation of the current k=3 interim artifact under the restored estimand
(`/tmp/p79_toolchain_chunk1_dryrun/`, vs `results/phantom_paper/phase1_full_prereg_decision.json`
captured 2026-07-13):

- **H3 axis-1 (P-text)**: pre-fix pool k=2 (B2-cls `n_unique=1`, SE=0.454pp cell
  deleted) θ_FE=3.20pp, boot CI [1.76, 4.88], `passed=true` → post-fix pool k=3
  θ_FE=**1.08pp**, boot CI [0.47, 1.98], **PARTIAL / NOT_EVALUATED / passed=null**.
  The re-included low-unique cell carries high FE weight (small SE) and pulls the
  pooled estimate down; interim CI still excludes 0.
- **H3 axis-2 (P-prompt)**: numerics unchanged (θ_FE=2.26pp); verdict semantics
  corrected to PARTIAL / NOT_EVALUATED (no cell had been deleted on this axis).
- **H1**: point/CI machinery unchanged by this note (universe enforcement only);
  k=3 interim status now also carries `analysis_status=PARTIAL`, `h1_verdict=NOT_EVALUATED`.

Any draft slot currently displaying the pre-fix interim axis-1 numbers is
stale-by-correction and must be refreshed from the regenerated artifact via
`verdict_day_slotsheet.py` (post-Chunk-2) before any further use. Interim
numbers were never verdicts (draft marks them interim-only).

## 3. Verification

- Baseline pytest before change: 1433 passed / 2 failed (both pre-existing
  prose-assertion failures, root-caused separately: one stale since 7b0f456,
  one a Round-2 prose regression; both fixed in the sibling commit).
- After change + sibling fixes: **1442 passed / 0 failed** (7 new regression
  tests: status-schema orthogonality, exact-set fail-closed with persisted diff,
  canonical helper vs locked counts, SR fixed denominator, H3 six-cell pool +
  floor semantics, blank-numeric/no-crash writers, three-output atomicity).
- Fix report with before/after hunks:
  `docs/checkpoints/codex_outputs/toolchain_fixes_chunk1_2026-07-14.md` (gitignored, local).

## 4. Out of scope (deferred to Chunk 2, same audit)

fig0c bootstrap universe (P0-5), fig0c/F2/slotsheet strict six-mode + provenance
join (P0-6, P1-1, P1-5), router covariate CLI rehearsal default + majority-baseline
OOF leakage (P0-7, P0-8, P1-2, P1-3), draft-number bypass closure (P0-9), and
consumer migration to `analysis_status`. Until Chunk 2 lands, slotsheet/F2 keep
reading legacy `gate_status` and keep treating everything as interim — the
conservative failure direction.
