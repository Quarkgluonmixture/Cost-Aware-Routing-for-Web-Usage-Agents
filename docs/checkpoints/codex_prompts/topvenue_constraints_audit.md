# Codex Prompt — Top-Venue Web Agent Paper Compliance Audit

## Goal

Produce `docs/checkpoints/pre_run/topvenue_constraints.md` — a literature-anchored
hard-constraint checklist that captures **what a top-venue (NeurIPS / ICML /
ACL / EMNLP / TMLR) web-agent + mechanistic-interpretability paper is expected
to satisfy**, then audit our repository's current state against each constraint
with **✓ done / ⚠️ partial / ❌ missing** plus per-gap remediation note +
reviewer-rebuttal-defense one-liner.

This is an **outside-view audit**. Do NOT just describe what our repo
already does — derive constraints from external standards (top-venue
reproducibility checklists, methodology lit, web-agent benchmark conventions,
mechanistic-interpretability best practices), then map TO our compliance.
The point is to surface design gaps that an inside-view repo audit would miss.

## Repository context to read FIRST (in order)

1. `docs/checkpoints/paper_planning.md` — paper strategy, hook, theory, decisions
2. `docs/checkpoints/paper_drafts/section1_intro.md`, `section2_*.md`, `section3_*.md` — completed prose
3. `docs/checkpoints/paper_drafts/paper.bib` — 57 lit references already in scope
4. `docs/checkpoints/pre_run/preregistration.md` — H1+H3 hypotheses + R1-R5 framing rules
5. `docs/checkpoints/pre_run/pre_rerun_audit.md` — 281-item internal audit
6. `docs/checkpoints/pre_run/evaluator_change_protocol.md` — Tier 1/2/3 classification
7. `docs/checkpoints/pre_run/reeval_audit_protocol.md` — Protocol A+B for re-evaluation
8. `docs/checkpoints/pre_run/osf_lock_manifest.md` — DOI workflow
9. `docs/reference/master_bug_catalog.md` — known bug catalog (50+ entries, B-81 8-class HPC)
10. `docs/checkpoints/实验笔记.md` §117 — Stage 2 mechanistic findings (skim §117.1-117.8)

## External sources to derive constraints from

Cite from the existing `paper.bib` where applicable. Where a key constraint
needs an anchor not in `paper.bib`, propose the canonical reference and flag
"NEEDS_BIB_ENTRY".

### Reproducibility / methodology (general ML)

- **Pineau et al. 2018** "Reproducibility checklist" NeurIPS
- **NeurIPS 2024 Reproducibility Checklist** (current submission requirement)
- **Gebru et al. 2018** "Datasheets for Datasets"
- **Mitchell et al. 2019** "Model Cards for Model Reporting"
- **Breck et al. 2017** "ML Test Score" Google
- **Sculley et al. 2014** "Hidden Technical Debt in ML Systems"
- **Lipton & Steinhardt 2018** "Troubling Trends in ML Scholarship"
- **CONSORT-AI** (Liu et al. 2020 BMJ) — clinical AI reporting

### Web-agent benchmark methodology

- **VisualWebArena** (Koh+ 2024) — eval criteria + visual FP handling
- **WebArena** (Zhou+ 2024) — site reset + url_match / program_html / string_match / ua_match
- **WebShop / Mind2Web / GAIA / OSWorld** — cross-bench standards
- **AgentBench** (Liu+ 2024) — cross-model eval
- **BrowserGym / WorkArena** (ServiceNow) — task curation conventions

### Mechanistic interpretability

- **Wang et al. 2023** "Interpretability in the Wild" (IOI circuit) — patching standards
- **Zhang et al. 2024** "How to use and interpret activation patching" — methodology guide
- **Heimersheim & Janiak** "Best practices for activation patching" (alignment forum / paper)
- **Conmy et al. 2023** ACDC — circuit discovery
- **Geva et al. 2023** key-value memory
- **Q5 Gemini DR 6/6** lit anchors already in our paper.bib

## Output structure

Single markdown file at `docs/checkpoints/codex_outputs/topvenue_constraints_audit_YYYY-MM-DD.md`
following this exact structure:

```markdown
# Top-Venue Web Agent Paper — Compliance Audit (auto-generated)

> Compliance status against literature-anchored hard constraints.
> Generated: <date>
> Sources surveyed: <N papers> across <M categories>

## TL;DR scoreboard

- ✓ Done: NN constraints
- ⚠️ Partial: NN constraints
- ❌ Missing: NN constraints
- Total: NN constraints across 8 categories

## Top 10 highest-leverage gaps (ranked)

| # | Constraint | Severity | Cost to fix |
|---|---|---|---|
| 1 | ... | High | 1 day |
| ... |

## Category A — Reproducibility (NeurIPS / Pineau)

| # | Constraint | Lit anchor | Compliance | Gap | Reviewer rebuttal |
|---|---|---|---|---|---|
| A1 | Random seeds documented + multi-seed averages | Pineau 2018 §3.2; NeurIPS Q14 | ✓ | seed=42 default in `configs/exp_v2_base.yaml` + `_seed_global_rng()` per (cond, seed); | "All experiments use seed=42 unless stated; multi-seed [42,123,456] available via `seeds` config field" |
| A2 | ... | ... | ... | ... | ... |

## Category B — Statistical Methodology (Cohen / Holm / CONSORT-AI)
...

## Category C — Evaluation Rigor (web-agent specific: VWA / WA convention)
...

## Category D — Methodology Hygiene (Lipton 2018 troubling trends)
...

## Category E — Web-Agent Specific (cross-bench / cost / latency)
...

## Category F — Threats to Validity (Cook & Campbell 1979)
...

## Category G — Mechanism / Interpretability (Wang 2023 / Zhang 2024)
...

## Category H — Limitations + Negative-Result Framing (open science)
...

## Action items (consolidated)

1. ⚠️ A3 — Add seed disclosure to figure captions (5 min, before paper §4 ship)
2. ❌ E1 — Cross-bench WA 480 task untested → scope to "VWA only" or future work
3. ...

## Notes / caveats
- Constraints not in our paper.bib that need entries: ...
- Areas where paper precedent is unclear or contested: ...
```

## Required behaviors

1. **Each constraint MUST have a lit anchor** (paper or community standard). Don't invent constraints with no precedent.
2. **Compliance evaluation MUST cite specific repo evidence** (file path or doc reference).
3. **Reviewer-rebuttal column MUST be 1-2 sentences max** — short answer ready for §8 limitations.
4. **Gaps marked ❌ MUST have remediation note** (specific action + cost estimate).
5. **At least 50 constraints, target 60-80, max 100** across all 8 categories.
6. **Categories A-H rough N target**: A=15, B=10, C=15, D=8, E=10, F=8, G=12, H=5.
7. **Top-10 gap ranking** at top of doc — for paper-grade prioritization.
8. **Use ✓ / ⚠️ / ❌ symbols verbatim** for compliance.
9. **Date the output file** with today's YYYY-MM-DD.
10. **Cite from existing `paper.bib`** by bibkey where the entry exists; use "NEEDS_BIB_ENTRY: <author year>" otherwise.

## Domain context for our paper

- **Hook**: Phantom routing space (3 arms: P-text / P-prompt / P-SoM) with 4-fold drop-in property (cost ≈ DOM / latency ~50% / signal AUROC ≥ baseline / drop-one 1.7-3.8pp). PROVISIONAL pending data confirmation.
- **Benchmark**: VisualWebArena (cls/reddit/shopping = 234/210/466 tasks), planned WA expansion (480 tasks).
- **Models**: B0 = Qwen3-Omni-235B-Thinking via proxy API. B1 = Qwen3-VL-4B local greedy decoding.
- **Modes**: 5-mode (DOM / SoM / Vision / phantom_som / phantom_dom).
- **Mechanistic Stage 2**: activation patching forward + reverse, 4 cells 2x2 (direction × tier) + cell E random-inject control + reddit cross-site pending.
- **Pre-registration**: H1+H3 hypotheses + R1-R5 data-conditional framing rules.
- **Critical concerns we already know about**: pattern-based mirage curation brittleness, N=15-24 mechanistic limited power, single-model B1, single-site partially mitigated, post-hoc layer selection.

## Constraints out of scope (do NOT audit)

- Internal code style (PEP 8 etc.) — not paper-grade
- Documentation prose quality — separate concern
- File naming conventions — separate concern
- Git workflow — separate concern

Focus is **paper-grade scientific defensibility ONLY**.

## When done

Write output to `docs/checkpoints/codex_outputs/topvenue_constraints_audit_<today>.md`
and print "DONE: wrote <path>" as final line. Do NOT also write to
`docs/checkpoints/pre_run/topvenue_constraints.md` (human reviewer will integrate).
