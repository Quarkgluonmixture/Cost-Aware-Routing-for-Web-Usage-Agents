# Cross-check: is the preregistration's overall structure and content optimal?

## Context

P79 paper-1 preregistration (`docs/checkpoints/pre_run/preregistration.md`) has been
edited ~12 times over the last 11 days, including a major 2026-05-13 reframe (16-cell
phantom-only → 24-condition / 4-cell scope, K-of-N gate → transparency-only). The
student is about to lock it with the advisor before firing the 24-condition Phase 1a
rerun (workshop submission target).

**Your job**: assess whether the preregistration's STRUCTURE and CONTENT is optimal —
NOT a bug hunt (that's /stress), but a "is this the right shape for a preregistration
that will be OSF-DOI'd and cited in paper §1" review. You have NOT seen Claude's
assessment.

## What to read

- `docs/checkpoints/pre_run/preregistration.md` — the full document (§1 epistemic
  structure → §2 hypotheses H1-H8 → §3 family declaration → §4 locked choices →
  §5 exploratory → §6 witness → §7 reproducibility → Appendix A decision log)
- `docs/checkpoints/followup.md` Part 2 — the student's advisor-sync questions, which
  reveal what the student thinks the open decisions are
- `docs/analysis/cross_sites/power_analysis.md` + `results/phantom_paper/meta_phantom_lift.md`
  — the archive data the prereg's thresholds are (or should be) calibrated against

## Questions to answer

1. **Epistemic structure** — is the "Hero + Drop-in + Structural + Framing-Rule"
   hierarchy (§1) the right shape? Does it actually compress garden-of-forking-paths,
   or does the R1-R5 data-conditional framing rule re-introduce researcher
   degrees-of-freedom through the back door?

2. **Hypothesis coherence** — H1 has two sub-conditions H1(i) pooled-meta-significant
   + H1(ii) magnitude≥1pp AND superiority-test. Are these non-redundant? (Hint: if the
   one-sided superiority test rejects H0: θ ≤ 1pp, what does that already imply about
   H1(i) and the magnitude check?) Is H1 over-specified?

3. **Scope/role staleness** — the doc was reframed but is the reframe COMPLETE? Grep
   for stale "16-cell", "H1-H6", "8 lock decisions" vs "9", model-name inconsistencies
   (Qwen3-Omni vs Qwen3-VL), reading-order claims that no longer match content.

4. **K_h1 vs K_h3 at N=4** — §4 locks K_h1=0.75 and K_h3=0.67 as two separate
   "transparency ratios". At N=4, ⌈0.75×4⌉ = ⌈0.67×4⌉ = 3. Are these two locked
   choices actually distinct? Should the doc keep two rows for them?

5. **Router family H7/H8** — the prereg is for a WORKSHOP submission on the
   phantom-space phenomenon (Phase 1a = §1-§4). H7/H8 router needs Phase 2 data that
   does not exist. Should the full H7/H8 spec + ROUTER family declaration be IN this
   prereg, or bracketed/moved? Does keeping it create a "preregistered but untested"
   liability?

6. **DerSimonian-Laird lock** — §4 audit B8 row LOCKS "Primary estimator:
   random-effects DerSimonian-Laird". Given k=4 statistical cells, is locking DL
   (rather than REML / Hartung-Knapp, or leaving it as an explicit advisor-decision)
   the right call? Should a preregistration lock an estimator known to be biased at
   the sample size it will be applied to?

7. **Missing content** — is anything a preregistration SHOULD have absent? (e.g.,
   explicit power/sample-size acknowledgment in the doc itself; a crisp H1
   PASS/FAIL decision flow; how the I²>75% "do not pool" branch interacts with the
   superiority test which needs a pooled estimate.)

8. **Over-engineering** — §4 has 20+ locked choices. Are some of them operational
   protocol (e.g., "stopping rules / contamination halt criteria") that belong in a
   companion ops doc rather than a hypothesis-gating preregistration? Is the prereg
   doing double-duty?

## Output format

### One-line verdict
Is the prereg structure optimal / good-with-fixes / needs-restructure?

### Structural strengths (what to keep)
2-4 things that are genuinely well-designed.

### Structural problems (ordered by severity)
For each: what's wrong, why it matters for an OSF-DOI'd cited document, the fix.

### Missing content
What a preregistration of this kind should have but doesn't.

### Over-engineering / scope creep
What's IN the doc that shouldn't be (belongs elsewhere or shouldn't be locked).

### Recommended restructure (if any)
If you'd reorganize: the proposed section order + what moves where.

## Calibration
- This is a structure/content review, not a bug hunt and not a stats-methodology
  audit (both already done separately).
- Bilingual OK (Chinese prose + English section refs / file:line).
- Be specific: quote section numbers / line content.
- Time budget: 30-45 min.
