# Codex Prompt — Paper §8 Limitations Consolidation (D6 + H1)

## Goal

Write the **final consolidated §8 Limitations** for the paper as a new file `docs/checkpoints/paper_drafts/section8_limitations.md` (do NOT overwrite the existing piecewise `section4_limitations_disclosure.md` — that stays as appendix-level long-form prose).

§8 is the **camera-ready limitations section** that goes in the body of the paper (~1.5 pages, ~900-1100 words). It should be reviewer-rebuttal-ready: every concession is paired with a *blast-radius bound* (how the limitation affects which claim, and why the affected claim is still defensible).

Addresses audit constraints **D6** ("State failed assumptions and limitations in a dedicated section") and **H1** ("Maintain a dedicated limitations section with strong assumptions and robustness failures"). Both ⚠️ → ✓ on completion.

## Inputs to read FIRST (in order, ~15 min)

1. `docs/checkpoints/pre_run/topvenue_constraints.md` — locate D6, H1, F2 (construct), F3 (external), F4 (internal validity), C9, A14 (external validity bound), G5 (post-hoc disclosure), B6 (missing-data), and read the "reviewer one-liner" column for each. These one-liners are the *spine* of §8.
2. `docs/checkpoints/paper_drafts/section4_limitations_disclosure.md` — 10 sub-disclosures (§4.X.1-§4.X.10) on specific bugs (ua_match drift, string_match fuzzy, program_html brittleness, finish_wrong_state, viewport bug, scroll direction, B0/B1 asymmetry, cross-machine drift, pre/post Phase A asymmetry, Stage 2B input vintage). These are inputs but §8 should *summarize and cluster*, not duplicate. §4_limitations_disclosure stays as appendix.
3. `docs/checkpoints/pre_run/pre_rerun_audit.md` §4.4 — final enumeration list for §8 prose.
4. `docs/checkpoints/pre_run/negative_results_registry.md` — 12 retracted framings + 2 confirmed framings table; the 5 paper-§ action items at the end. §8 should reference this registry, not duplicate it.
5. `docs/checkpoints/pre_run/preregistration.md` — H1-H8 hypothesis lock, §5.X post-hoc layer disclosure, §7 reproducibility scope. §8 references the H4 (exploratory) and H5-H6 (post-hoc) labels.
6. `docs/checkpoints/paper_drafts/section1_intro.md` — read first 100 lines for the R1-R5 outcome-conditional framing rules (H4 audit).
7. `docs/checkpoints/paper_drafts/section5_mechanism_reddit.md` §5.1 method box (just written, addresses G1) — §8 should cite §5.1 when discussing mechanism limitations (e.g., 36-layer Qwen3-VL only, no cross-architecture).
8. `docs/checkpoints/实验笔记.md` §111 (task-0 over-interpretation retracted), §111.5b (reverse-null overturn), §117 (Stage 2 cell A-E), §118 (selection-bias control rejected). Quote-paraphrase, do NOT bulk-import.
9. `docs/checkpoints/paper_drafts/paper.bib` — check existing bibkeys; flag NEEDS_BIB_ENTRY for any new cites (likely Lipton & Steinhardt 2018 "Troubling Trends", Pineau 2021 reproducibility, NeurIPS checklist).

## Required structure for §8 Limitations (camera-ready prose)

Use these subsection headers (LaTeX `\subsection*{...}` style at section level, no auto-numbering since §8 is "Limitations"):

### 8.1 Scope & external-validity bounds (~200 words)
- 3 sites (cls/red/shop), 1 benchmark family (VWA + WA-mini), 2 model classes (Qwen3-VL-4B / Qwen3-Omni-235B-Thinking proxy). Mechanism §5 evidence is from **B1 only** (open-weight); B0 has no activation access.
- Cross-architecture (e.g., GPT-4o family) untested → claim is "phantom routing space exists in the Qwen3-VL family on VWA-style tasks", not "in all VLMs everywhere".
- Cite A14 + F3 reviewer-rebuttal language from constraint table.

### 8.2 Construct validity & evaluator threats (~200 words)
- ua_match GPT-judge drift (B-20), string_match fuzzy_threshold misnomer, program_html selector brittleness, finish_wrong_state. Cluster as 4 evaluator-class threats; cite §4_limitations_disclosure §4.X.1-§4.X.4 for full prose.
- FP filter (§95 eval_fp/visual_fp + §78a na_fp) addresses these *measurement-side*, not *task-side*; report raw + adjusted SR.
- Cite F2 reviewer-rebuttal: "We report raw and adjusted success and isolate evaluator-class threats in limitations."

### 8.3 Internal-validity threats: known scaffold bugs (~150 words)
- in_viewport_ratio operator precedence (B-26 — affects all DOM modes uniformly, doesn't bias inter-mode contrast), scroll direction confusion (B-28), Stage 2B input vintage independence (笔记 §116). All three: blast radius is **uniform across modes**, so does not explain inter-mode SR gaps.
- Cite C9 reviewer-rebuttal: "We treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims."

### 8.4 Pre-vs-post-hoc analyses & retracted framings (~200 words)
- Per preregistration §5.X, the L11/L17 mechanism layer choice was **not pre-registered**; it emerged from Stage 2A pilot (hypothesis-generating). Confirmed by 4 independent paths (logit_shift, forward overlap, reverse overlap, cross-tier).
- Reference negative_results_registry: 12 framings retracted (e.g., "task-0 over-interpretation" §111, "reverse-null" §111.5b, "selection bias" §118 Welch p≥0.5 NS), 2 confirmed framings (4-fold drop-in property, sparse mechanism).
- Cite G5 reviewer-rebuttal: "Layer-set choice is disclosed as exploratory in §5 and pre-registered in the deposited registry."

### 8.5 Statistical & methodological limits (~150 words)
- Holm-Bonferroni across 6 layers chosen post-hoc (not 36); disclose. Bootstrap clustering policy locked at task-pair (B2). Random-effects meta-analysis only on cells with N≥10 to avoid τ² instability (per B8 lock).
- Power analysis (B9) shows minimum-detectable effect for cells with N=15 ~0.65 Cohen's d at α=0.05 — not powered for small mid-layer effects.
- Missing-data policy: complete-case (B6 lock); ≤5% per cell, no multiple-imputation needed.

### 8.6 Sparse-mechanism caveat (~100 words)
- G8 finding: median Levenshtein-distance shift = 0 for 4 of 5 cells at L17, IQR includes zero. Mechanism activates on *task subsets* (~25% of strong-tier), not uniformly — reframes the §5 claim as "task-conditional sparse mechanism" rather than "universal mid-layer circuit". Cite §5 fig and §117 笔记 entry.

### 8.7 Compute, cost, and sustainability bounds (~80 words)
- Per-cell GPU-hours / USD / kg-CO₂ table per A10 reviewer-rebuttal. Cross-machine power profile (DGX Spark vs Myriad V100) is one limitation source for absolute energy numbers; relative comparisons within the same cell are unaffected.

## Constraints

- One file created: `docs/checkpoints/paper_drafts/section8_limitations.md`
- Plain markdown (compatible with codex / pandoc), use `\citep{...}` for bibkeys
- Length: 900-1100 words total (sum of 8.1-8.7). Do NOT exceed 1200.
- Each subsection: lead with the *concession* (1 sentence), then the *blast-radius bound* (1-2 sentences), then *paper-impact* (1 sentence: which §/claim this affects, why still defensible).
- Reference but do NOT duplicate `section4_limitations_disclosure.md`. §4 stays as long-form appendix; §8 is paper-body summary.
- Flag NEEDS_BIB_ENTRY for any cite not in `paper.bib`.
- Do NOT recommend new experiments; this is a *bounding* document, not an action list.
- Use \citep{} not [X] format throughout (per D8 audit upgrade earlier this session).

## Output

Print exactly:
```
DONE: wrote section8_limitations.md (W words across 7 subsections)
```
plus 5-bullet diff summary listing the new file's subsection headers + word counts + any NEEDS_BIB_ENTRY flags.

Stop. Do not modify any other file.
