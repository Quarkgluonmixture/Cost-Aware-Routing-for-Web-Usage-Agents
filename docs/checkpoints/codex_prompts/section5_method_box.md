# Codex Prompt — Paper §5 Method Box (G1)

## Goal

Add a formal **§5.1 Method** subsection to `docs/checkpoints/paper_drafts/section5_mechanism_reddit.md`
defining the activation patching protocol. Addresses audit constraint **G1**
(define clean/corrupt/source/target prompts and behavioral metric for
activation patching, per Wang et al. 2023 IOI / Zhang et al. 2024 patching guide).

## Inputs to read FIRST

1. `docs/checkpoints/paper_drafts/section5_mechanism_reddit.md` — current §5 prose, identify where Method box should go
2. `scripts/mechanistic/run_stage2b_continuation_pilot.py` — the patching script (especially `patching_grid_continuation` call site at line ~248)
3. `p79/mechanistic/activation_patching.py` — `patching_grid_continuation()` function definition
4. `scripts/analysis/stage2_layer_significance.py` — stat test methodology (paired t / Wilcoxon / Holm / bootstrap CI)
5. `docs/checkpoints/pre_run/preregistration.md` §4 (bootstrap policy lock) + §5.X (post-hoc layer disclosure)
6. `docs/checkpoints/实验笔记.md` §117 — Stage 2 cell A/B/C/D/E results
7. `results/mechanistic/archive_subset_b1_cls/manifest.json` — task subset definitions

## Required content for §5.1 Method box

The new subsection should be ~1 page (LaTeX-equivalent), with these **explicit definitions**:

### 1. Source / Target prompts
- "Source" = som mode (with annotated screenshot + `[SOM_MARKS]` text + SoM prompt)
- "Target" = phantom_som mode (no image, `[SOM_MARKS]` text only, SoM prompt)
- Same task, same step (step 2), same model (Qwen3-VL-4B-Instruct revision `ebb281ec70b0...`)

### 2. Patching protocol
- For each task, generate source greedy continuation (50 tokens) + target greedy continuation (50 tokens)
- Cache source's per-layer hidden states (36 layers L0-L35 = embedding output through final post-block)
- For each layer L, run target's forward with source's L_th hidden state injected at the last input position, then greedy-generate 50 tokens
- "Forward direction" = source(som)→target(phantom_som) injection
- "Reverse direction" = phantom_som hidden injected into som run (`--reverse` flag swaps)

### 3. Behavioral metrics (per layer per task)
- `token_overlap_to_source` = fraction of positions where patched output matches source token (0-1)
- `token_overlap_to_target` = same vs target (0-1)
- `ld_to_source` = Levenshtein distance, patched vs source (0-50)
- `ld_to_target` = same vs target

Higher overlap_to_target / lower LD_to_target = "patch had no effect". Disruption signal = drop in overlap or rise in LD.

### 4. Statistical procedure (cite preregistration §4 lock)
- Paired t-test layer L vs L35 baseline, **alternative='less'** for overlap, **'greater'** for LD
- Holm-Bonferroni correction across 6 tested layers (L0/5/11/17/23/29 vs L35)
- 1000-sample task-paired percentile bootstrap 95% CI on per-task (L_n - L35) difference
- Wilcoxon signed-rank as non-parametric backup

### 5. Task curation (cite curate_mirage_tasks.py)
- Composite score = `(src_neg - tgt_neg) + (tgt_aff - src_aff)` × `(1 + divergence)`
- Strong-tier: composite ≥ 1.0, overlap < 0.5
- Reverse-tier: composite ≤ -1.5
- Task counts: cls 24 strong + 15 reverse, reddit 47 strong + 48 reverse (per `dataset_card.md`)

### 6. Cell design (2x2 + random control)
- Cell A: forward × cls-strong (N=24)
- Cell B: reverse × cls-reverse (N=15)
- Cell C: forward × cls-reverse (N=15) — selection-bias control
- Cell D: reverse × cls-strong (N=24) — selection-bias control
- Cell E: forward × cls-strong + random Gaussian source hidden (N=24, seed=42) — content-specificity control
- (Cells F/G: cross-site reddit, pending)

### 7. Random-injection control specifics
- Replace cached source hidden with `randn_like(h) * h.std() + h.mean()` per layer
- Matched mean+std preserves activation magnitude, destroys task-specific structure
- Specificity ratio = random LD / real LD (paper-grade signal of content-specific causal effect)

### 8. Pre-vs-post-hoc layer selection disclosure
- Cite preregistration §5.X
- Layers L11/L17 not pre-registered; emerged from Stage 2A pilot (hypothesis-generating); confirmed by Stage 2B/2C/Cell-D (3 confirmatory)
- Convergence of 4 independent paths (logit_shift, forward overlap, reverse overlap, cross-tier) constitutes confirmatory evidence

## Output

Modify `docs/checkpoints/paper_drafts/section5_mechanism_reddit.md` via `apply_patch`:
- Insert new `## 5.1 Method` subsection at top of §5 (before existing prose)
- Renumber subsequent subsections accordingly
- Cite paper.bib bibkeys (`wang2023interpretability` IOI, `zhang2024patching`, etc.)
- Use `\citep{}` LaTeX format

Print "DONE: rewrote section5_mechanism_reddit.md with §5.1 Method box" as final line plus 5-bullet diff summary.

## Constraints

- Output ONE file modified (section5_mechanism_reddit.md), no new files
- Length: ~1 page LaTeX equivalent (~600-800 words for §5.1)
- Cite specific bibkeys; flag NEEDS_BIB_ENTRY for any not in paper.bib
- Reproducibility-detail bias: include enough detail that a replicator can reproduce setup without code, but defer code reference to repo
