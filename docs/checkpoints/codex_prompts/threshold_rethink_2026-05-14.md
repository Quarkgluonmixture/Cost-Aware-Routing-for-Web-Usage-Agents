# Cross-think: pre-registration threshold calibration grounded in bug-fix-pre archive data

## Context

P79 paper-1 pre-registration has 3 thresholds the student wants to lock with the advisor:
- **K_h1** (hero claim per-cell pass ratio) — currently drafted 0.75
- **K_h3** (structural claim per-cell pass ratio) — currently drafted 0.67
- **δ** (effect-size margin) — currently drafted 1.0pp

The student's draft (`docs/checkpoints/followup.md` Part 2 §1) still frames these for a
**16-cell** scope and as **gating** thresholds. But two things changed in the last 2 days:

1. **Scope reframed** to Phase 1a = 24 operational conditions across **4 statistical
   cells** (= (site, model) tuples: cls×B0, cls×B1, red×B0, red×B1). Shop deferred.
2. **K-of-N reclassified** from gate → transparency-only consistency check. Primary
   gate is now pooled DerSimonian-Laird random-effects meta + one-sided superiority
   test (H1(ii) was changed TOST→superiority).

**Your job**: independently re-derive what these 3 thresholds SHOULD be, grounded in
the actual bug-fix-pre archive data. Do NOT anchor on the student's 0.75/0.67/1.0pp
draft — derive from the archive evidence. You have NOT seen Claude's analysis.

## The archive data (read these cold)

- `results/phantom_paper/meta_phantom_lift.md` — per-cell drop-one lifts + pooled
  random-effects meta on the k=3 archive cells (B0 cls, B0 red, B1 cls). This is the
  bug-fix-pre data the thresholds must be calibrated against.
- `docs/analysis/cross_sites/power_analysis.md` — observed SR ranges, per-cell MDE,
  family-wise power at observed effect sizes. Note: written for 16-cell scope.
- `docs/analysis/cross_sites/axis_effect_size_report.md` — axis decomposition
  (text/prompt/image) effect sizes, relevant to H3 structural claim.
- `docs/analysis/cross_sites/sensitivity_loo_meta.md` — leave-one-out meta sensitivity.
- `docs/checkpoints/followup.md` Part 2 §1 — the student's current draft + reasoning.
- `docs/checkpoints/pre_run/preregistration.md` §2 (H1/H2/H3) + §4 (locked choices) —
  current locked statistical framework (post 2026-05-13 revisions).

## Questions to answer

For EACH of K_h1, K_h3, δ:

1. **What does the archive data actually say?** Quote the specific per-cell lifts,
   per-cell SEs, I², pooled estimates. What per-cell power do the observed effect
   sizes imply?
2. **Does the 16-cell framing survive the reframe to 4 statistical cells?** At N=4,
   ⌈0.75×4⌉ = ⌈0.67×4⌉ = 3 — the K_h1 vs K_h3 distinction collapses. Is the
   percentage framing even meaningful at N=4? What should replace it?
3. **Given K-of-N is now transparency-only**, what is the threshold actually FOR?
   What's a defensible value (or "report count, no threshold")?
4. **For δ**: the student's draft frames δ=1.0pp as a "cost equivalence margin" but
   H1(ii) was changed TOST→one-sided superiority. Is δ now the superiority threshold?
   Does the archive pooled drop-one (+2.34pp per meta_phantom_lift.md) clear δ=1.0pp
   with adequate headroom? Would δ=0.5pp or δ=2.0pp be better/worse and why?

Also flag: is there a P0-level concern the student's draft is missing? (e.g., the
known DerSimonian-Laird-at-k=4 fragility, or B1-reddit being the untested 4th cell,
or P-text's I²=71% making pooling questionable.)

## Output format

### One-line verdict
Is the student's 0.75 / 0.67 / 1.0pp draft defensible as-is, needs-revision, or
needs-replacement-of-framing?

### Per-threshold recommendation
For K_h1, K_h3, δ each:
- **Archive evidence** — the specific numbers
- **Recommended value** — your number, with the archive-grounded justification
- **Framing fix** — if the draft's framing is stale (gate vs transparency, 16-cell
  vs 4-cell, cost-equiv vs superiority), state the corrected framing

### The advisor question that actually matters
The student is going to ask the advisor about K_h1/K_h3/δ. Is that the right question?
If the real methodology decision is something else (e.g., DL vs REML, or per-cell vs
pooled primary), say so.

## Calibration

- Ground every recommendation in a quoted archive number. No first-principles hand-waving.
- Bilingual OK (Chinese prose + English stats terms / file:line).
- If the archive data is insufficient to calibrate a threshold, say so explicitly —
  "insufficient archive evidence, must be a conservative default" is a valid answer.
- Time budget: 30-45 min.
