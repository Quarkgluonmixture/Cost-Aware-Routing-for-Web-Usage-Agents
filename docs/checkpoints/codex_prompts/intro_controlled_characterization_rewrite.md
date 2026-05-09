# Codex Prompt — Paper §1 Intro Rewrite to "Controlled Characterization"

## Goal

Revise `docs/checkpoints/paper_drafts/section1_intro.md` (and `section2_background.md`'s
related-work paragraphs) so that **novelty claims are grounded in
"controlled scientific characterization"** rather than "first-deployment"
language. Address audit constraints **D8** (avoid novelty overclaim against
industry/prior systems) and **H5** (document where precedent is contested).

## Why

Industry artifacts predate our deployment claim:

- `yang2023som` SoM-Mark — uses textual Set-of-Marks observation, this is closer
  to our P-text/P-SoM than commonly recognized.
- `zheng2024seeact` — SeeAct already explored marked-screenshot agent paradigms.
- `yang2025magma` — magma's omni-modal action grounding overlaps phantom design.

Our claim should NOT be **"first to deploy text-only / marked observations"**.
Our actual contribution is:

1. **Controlled scientific characterization** of the phantom routing space
   boundary (skip annotated image) and its 4-fold drop-in property
2. **Mid-layer (L11-L17) mechanistic evidence** for why phantom modes work
3. **2x2 + random-injection control infrastructure** that establishes specificity
4. **Pre-registered framing decision rules R1-R5** for data-conditional claims

Read `docs/checkpoints/pre_run/negative_results_registry.md` entry #8
("First inference-time substitution / first deployment of text-only or
marked observations" novelty claim **retracted**) for the full context.

## Repository context to read FIRST

1. `docs/checkpoints/paper_drafts/section1_intro.md` — current intro prose
2. `docs/checkpoints/paper_drafts/section2_background.md` — current related work
3. `docs/checkpoints/paper_planning.md` §21-§22 — earlier industry-precedent retraction discussion
4. `docs/checkpoints/pre_run/negative_results_registry.md` entry #8
5. `docs/checkpoints/paper_drafts/paper.bib` keys: `yang2023som`, `zheng2024seeact`, `yang2025magma`,
   `koh2024visualwebarena`, `zhou2024webarena`
6. `memory/project_paper_hook.md` retract list
7. `docs/checkpoints/pre_run/preregistration.md` §7 (reproducibility/external-validity scope)

## Task

1. **Read** the 7 input docs (2-3 of them are large; quote evidence).
2. **Audit** the current intro / related work for instances of:
   - "First [to/use of] X" claims where X has industry precedent
   - "Novel routing arm" where the arm itself has artifact precedent (we control novelty)
   - "Hidden 4th routing arm" (already retracted to "phantom routing space (3 arms)" but final prose may still have stale wording)
3. **Rewrite** intro paragraphs to "controlled characterization" framing:
   - Specifically: replace "we propose X" with "we characterize the phantom configuration X under controlled scientific evaluation"
   - Replace "we discover" with "we identify and quantitatively characterize"
   - Replace "first" with "the first **controlled scientific evaluation** of" where applicable
   - Add explicit acknowledgment to industry precedents (yang2023som SoM-Mark, zheng2024seeact, yang2025magma) at the relevant intro paragraph
4. **Rewrite** related work to add 1-2 paragraph "Industry artifact precedents vs scientific characterization" subsection clarifying what's new.
5. **Preserve** the 4-fold drop-in property as the paper's empirical finding (not retract).
6. **Preserve** the phantom routing space framework (3 arms) and R1-R5 framing decision rules (already aligned).

## Output

Write **two files** (use apply_patch to update existing files):
- `docs/checkpoints/paper_drafts/section1_intro.md` — updated intro
- `docs/checkpoints/paper_drafts/section2_background.md` — updated related-work paragraph

**Each rewrite must**:
- Cite specific bibkeys from `paper.bib` (use `\citep{yang2023som}` etc.)
- Be 1-1.5 pages worth of prose updates each (don't rewrite whole section, only the affected paragraphs)
- Preserve existing structure / equation numbering / figure refs
- Diff cleanly against pre-rewrite

**Do NOT**:
- Add new sections (just modify existing paragraphs)
- Remove the 4-fold drop-in claim (it's the empirical finding, not the novelty claim)
- Change H1-H8 or R1-R5 (those are pre-registered)
- Touch section3 or later (out of scope for this rewrite)

## When done

Print "DONE: rewrote section1_intro.md and section2_background.md" as final line,
plus a 5-bullet diff summary of what changed (e.g. "removed 'first inference-time
substitution' from §1.2", "added 1-paragraph industry precedent subsection in §2.3", etc.).

## Constraints

- Output 2 files modified, no new files
- Maintain reviewer-defensible tone: don't oversell, don't undersell
- Reading the 7 input docs is mandatory before any rewrite — quote evidence in your reasoning
