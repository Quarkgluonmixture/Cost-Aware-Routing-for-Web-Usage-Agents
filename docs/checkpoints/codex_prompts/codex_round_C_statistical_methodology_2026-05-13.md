# Round C — Statistical methodology assumptions audit

## Context

P79 paper-1 statistical framework (post 2026-05-13 codex stress audit revisions):

- **N = 4 statistical cells** (Phase 1a scope: 2 sites × 2 models)
- **Primary gate (H1)**: pooled DerSimonian-Laird random-effects meta on per-cell
  P-SoM drop-one oracle ceiling lift, Holm α=0.05 sig (m=1) + pooled magnitude
  θ_RE ≥ 1.0pp + one-sided superiority test (H0: θ ≤ +1.0pp) rejected at α=0.05
- **Primary gate (H3)**: pooled DL meta on axis-1 |P-text \ P-SoM| unique count +
  axis-2 |P-prompt \ P-SoM| (two separate sub-families, each m=1)
- **Primary gate (H2)**: per-cell median cost equivalence within ±10%, replicated
  in ≥ 3 of 4 cells (K_h2 transparency)
- **Transparency-only (NOT gating)**: K-of-N consistency checks per H sub-family.
  K_h1 = 3 of 4 cells; K_h3 = 3 of 4 per axis. Reclassified pre-data 2026-05-13
  from gating threshold to transparency consistency per power analysis showing
  K family power < 10% at observed 1-3pp effect sizes
- **Bootstrap**: 1000-resample paired task-level bootstrap, percentile CI,
  single-level (no nested cluster), seed=42
- **Framing rule R1-R5**: maps (H1, H2, H3_axis1, H3_axis2) primary gate decisions
  to hook power (STRONGEST / MODERATE-STRONG / MODERATE / WEAK / paper-death)

**Your job**: Audit this framework as a hostile **statistical reviewer**. NOT a
methodology blessing. Find anything that:
- (a) A top statistical reviewer (e.g., NeurIPS statistics-area chair) would
      challenge as "the assumption doesn't hold here"
- (b) Has a math error (DL formula transcribed wrong, bootstrap variance wrong,
      Holm step-down wrong, K-of-N power calculation wrong)
- (c) Is statistically valid in isolation but problematic in combination (e.g.,
      separate sub-families with Holm correction but undeclared shared cells)
- (d) Has a edge case where R1-R5 framing rule mapping gives the wrong answer
- (e) Is undefined for the observed data (e.g., what if a cell has 0 tasks, or
      one mode is missing, or I² > 75% triggers "do not pool")

## Input files (read cold)

### Primary statistical method docs

- `docs/checkpoints/pre_run/preregistration.md` §2 H1/H2/H3 (statements + drop-one
  formula + superiority test wording) + §3 family declaration + §4 locked analysis
  choices (TOST δ / K-of-N / heterogeneity / bootstrap / missing data / stopping
  rules) + §5.X Stage 2 layer selection disclosure + Appendix A 2026-05-13 entries
- `docs/analysis/cross_sites/power_analysis.md` — K-of-N power calculation,
  underwriting K-of-N transparency-only reclassification

### Statistical implementation code

- `scripts/analysis/preregistration_decision_test.py` — Round 2 rewrite, the
  canonical implementation of the framework:
  - `dersimonian_laird_meta()` — pooled meta math (Higgins & Thompson 2002 /
    DerSimonian & Laird 1986)
  - `superiority_test()` — one-sided z-test for θ > threshold
  - `tost_equivalence()` — TOST (Schuirmann 1987), now informational only
  - `holm_correct()` — Holm-Bonferroni step-down
  - `_paired_bootstrap()` — task-level resampling
  - `apply_framing_rule()` — R1-R5 mapper
- `scripts/analysis/aggregate_phantom_meta.py` (if exists) — random-effects meta
  on existing pre-Phase-A archived data
- `scripts/analysis/aggregate_phantom_lift.py` (if exists) — TOST on pooled tasks
- `scripts/analysis/aggregate_routing_auroc.py` (if exists) — routing-signal AUROC
  family
- `scripts/analysis/sensitivity_loo_meta.py` (if exists) — leave-one-out
  robustness meta

### Companion sanity references

- `docs/checkpoints/paper_planning.md` §2 (theory framework) — does the
  statistical framework match the theoretical scaffold the paper claims?
- `docs/reference/EVIDENCE_LAYER_AUDIT.md` — each figure traces to a gated
  H1/H3 sub-claim OR exploratory; check consistency

## Output format

### One-sentence statistical verdict

Pick one:
- "Statistical framework is methodologically sound — safe to gate paper claims"
- "Statistical framework has N issue(s) that would be flagged by a top stat reviewer"
- "Statistical framework has reviewer-defensible interpretation choices but no math error"
- "Insufficient time to verify — partial audit only"

### Confirmed math errors

For each: location (file:line, method name), the correct formula vs the implemented
formula, magnitude of inference error, severity (HIGH / MED / LOW), defuse effort.

### Confirmed methodological assumptions violated

E.g., DL random-effects meta typically requires N ≥ 10 cells for stable τ²
estimation — N = 4 is small. Is the prereg disclosed this? Is the I² > 75%
"do not pool" branch in R1-R5 statistically coherent? Is paired bootstrap
single-cluster appropriate (vs nested cell × task two-level)?

### Reviewer ammunition (statistical angle)

What questions would a top-tier statistical reviewer ask? E.g.: "Why DerSimonian-Laird
and not REML for τ²?" / "Why one-sided superiority and not two-sided 90% CI?" /
"Why Holm and not FDR for the transparency sub-families?" / "Why N=4 not N≥10?" /
"Did you correct for the multiple K-of-N transparency reports as multiple comparisons?"

For each, list the question + the user's current answer (or "no answer prepared").

### Framing rule R1-R5 edge cases

Are there observed-data outcomes where R1-R5 mapping is ambiguous or gives the
wrong hook framing? E.g., what if H1 passes but I² > 75% triggers "do not pool"
branch — does the heterogeneity-conditional rule resolve cleanly?

### Verdict on next steps

If framework holds: user can confidently lock prereg + push commit (already done).
If framework has flaws: prioritized list — which can be fixed in prereg prose
before advisor meeting tomorrow vs which require advisor sync to repropose vs
which can be defused as appendix sensitivity check after Phase 1a data lands.

## Calibration

- Paper-grade audit, statistical reviewer mode. Not a textbook intro.
- Don't propose code fixes; identify the suspect, impact, defuse cost
- Negative result valid: if framework holds after 60 min, write verdict and stop
- Don't fabricate. File paths + line numbers must be real
- Set your own attack vectors based on what the docs + code show. Do NOT follow
  any enumerated list — read the actual statistical content cold

## Time budget

Up to 60 min. Tier 3 PID monitor fires when codex exits.
