# Round B — Paper §1-§8 prose claim ↔ evidence cross-section audit

## Context

User is preparing paper-1 for workshop submission (Phase 1a 24-condition rerun
launches this week). Paper drafts are in `docs/checkpoints/paper_drafts/`:
section1_intro / section2_background / section3_definition / section4_empirical_findings /
section4_limitations_disclosure / section5_mechanism / section8_limitations + paper.bib.

Prior codex audits this week focused on **code/data pipeline** (`codex_full_prerun_audit_2026-05-13.md`,
`codex_phase_a_audit_2026-05-13.md`), **viz pipeline** (`viz_pipeline_audit_FINAL_2026-05-13.md`),
**v2 retraction** (`v2_retraction_audit_2026-05-13.md`), and **16-cell design**
(`codex_stress_16cell_design_2026-05-13.md`). They did NOT systematically reviewer-pass
the **paper prose itself** for cross-section claim ↔ evidence consistency.

Today's commit `e9ddbe3` revised paper §3.4 (P-prompt re-inclusion as 4th cell of
complete 2×2) + revised preregistration H1/H3 + scope reframe to 24/4. The paper §1
hook + §4 empirical findings prose may now be **stale** relative to revised §3 and
prereg. Cross-section coherence post-revision needs check.

**Your job**: hostile reviewer cold-read of paper §1-§8 prose. NOT a copyedit — a
claim-vs-evidence audit. Find anything that, if shipped to a top-tier reviewer
(NeurIPS / ICML / ACL / ICLR), would:
- (a) Get flagged as "claim ambition > evidence ambition" (stretched claim)
- (b) Get flagged as "post-revision drift" (§3 says X but §1 still says X')
- (c) Get flagged as "where's evidence for this" (claim without supporting §N pointer)
- (d) Get flagged as "citation does not actually say what you claim it says"
- (e) Get flagged as "scope statement contradicts what was measured"

## Input files (read cold)

### Primary targets — paper prose

- `docs/checkpoints/paper_drafts/section1_intro.md` — paper hook + R1-R5 framing rule
- `docs/checkpoints/paper_drafts/section2_background.md` — related work + theoretical framing
- `docs/checkpoints/paper_drafts/section3_definition.md` — mode definitions + 2×2 ablation
  (revised today: §3.4 P-prompt re-inclusion)
- `docs/checkpoints/paper_drafts/section4_empirical_findings.md` — hero tables, drop-one
- `docs/checkpoints/paper_drafts/section4_limitations_disclosure.md` — disclosure prose
- `docs/checkpoints/paper_drafts/section5_mechanism.md` — Stage 2-4 mechanism evidence
- `docs/checkpoints/paper_drafts/section8_limitations.md` — limitations statement
- `docs/checkpoints/paper_drafts/paper.bib` — citation accuracy

### Cross-reference against (does prose match these?)

- `docs/checkpoints/pre_run/preregistration.md` — H1/H2/H3 hypothesis statements +
  R1-R5 framing rule + §4 locked analysis choices + §7 reproducibility scope
- `docs/checkpoints/paper_planning.md` — theory framework + paper hook canonical
- `docs/reference/EVIDENCE_LAYER_AUDIT.md` — figure registry: each paper figure should
  trace to a gated H1/H3 sub-claim OR be flagged exploratory

### Recent state context

- Commit `e9ddbe3` — today's scope reframe (16 conditions phantom-only → 24
  conditions/4 cells; P-prompt re-inclusion; K-of-N transparency)
- `docs/checkpoints/实验笔记.md` §132 — chronicle of today's changes

## Output format

### One-sentence prose verdict

Pick one:
- "Paper prose is reviewer-ready (workshop-grade)"
- "Paper prose has N claim ↔ evidence inconsistency / drift / overclaim"
- "Paper prose has scope-honesty concerns but no proven overclaim"
- "Insufficient time to verify — partial audit only"

### Claim ↔ Evidence inconsistencies

For each: claim location (file:line + quote claim), evidence pointer (where claim
is supposed to be supported), gap (claim says X, evidence shows Y), severity
(HIGH = invalidates hook framing / MED = forces disclosure / LOW = reviewer would
ask question but answerable).

### Cross-section drift (post-revision)

If §3 was revised today to include P-prompt as 4th cell of 2×2, do §1 hook, §4
table, §7 limitations, §8 disclosure all reflect 6-mode framework? If not, report
exact lines that need updating.

### Citation accuracy

Spot-check 5-10 high-leverage cites in paper.bib (e.g., Yang et al. 2023 for SoM,
DerSimonian & Laird 1986 for meta-analysis, Higgins & Thompson 2002 for I², Holm
1979, Wu et al. for Stage 2/4 layer locus). For each, does the prose claim match
what the cited paper actually argues?

### Scope-honesty / overclaim audit

For each claim in §1 hook + §1 contributions, ask: "does the Phase 1a evidence
(cls + red × B0 + B1 × 6 modes, 4 statistical cells) actually support this claim,
or is it stretched?" Specifically check generalization claims (cross-site,
cross-model, cross-family) — are they appropriately scoped to the data?

### Reviewer ammunition

What questions would a hostile NeurIPS / ICML / ACL / ICLR reviewer ask after
reading paper §1 + §3 + §4 + §5 + §8 + the supplement preregistration appendix?
List 5-10 specific reviewer questions with the answer the user currently has
(or "no answer prepared" if gap).

Calibrate to reviewer-3-skeptical-3/10 (peer-lab top-tier, not friendly).

### Verdict on next steps

If prose holds: tell user paper §1-§8 is workshop-submission ready post-Phase-1a-data-land.
If prose has flaws: prioritized list of which to fix tonight (1-3h) vs which to
fix after Phase 1a data lands (when §4 numbers are real) vs which to defer to
extended-version main paper.

## Calibration

- Paper-grade audit, not copyedit. Don't flag typos or style. Don't propose new
  citations or new prose. Identify the suspect, the impact, the defuse cost.
- Calibrate severity by paper-grade impact, not aesthetic ugliness
- Negative result valid: if prose holds after 60 min, write verdict and stop
- Don't fabricate. File paths + line numbers must be real
- Set your own attack vectors based on what the prose shows

## Time budget

Up to 60 min. Tier 3 PID monitor fires when codex exits.
