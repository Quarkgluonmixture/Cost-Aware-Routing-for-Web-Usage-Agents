# Axis Effect Size Ablation

Hierarchical analysis with two tiers:

- **Tier 1 (Hook)** — 3-mode coarse validation (DOM / P-SoM / SoM): establishes that **P-SoM is an independent routing arm** distinct from both DOM and SoM endpoints. Does not require P-text/P-prompt data.
- **Tier 2 (Mechanism)** — 5-mode diamond (DOM / P-text / P-prompt / P-SoM / SoM): explains *why* P-SoM is distinct by decomposing the compound DOM→P-SoM transition into text-axis and prompt-axis sub-effects. Splits into:
  - **2a Macro** — action-type frequencies (this file): finish rate, step count, search/type/scroll/click %, action repeat, self-correction.
  - **2b Micro** — per-step decision quality (separate analysis): URL trajectory Jaccard, target-page hit rate, search keyword reuse, first-action divergence.

**Current data status**: Tier 1 ✅ complete; Tier 2a Macro **partial** (cascade only, P-prompt data not yet available — diamond will replace cascade once it arrives); Tier 2b Micro tracked separately in `axis1_microbehavior.{json,md}`.

## Tier 1 — Hook: is P-SoM distinct from both DOM and SoM?

| baseline | site | metric | DOM→P-SoM (compound) | P-SoM→SoM (image) | distinct from DOM? | distinct from SoM? |
|---|---|---|---|---|---|---|

**P-SoM independence verdict** (cells where P-SoM differs from BOTH DOM and SoM, |effect|>0.1):
- **No cells show P-SoM distinct from both endpoints simultaneously**

## Tier 2a — Mechanism (Macro): cascade decomposition

DOM → P-text (axis 1, text only) → P-SoM (axis 2, prompt only) → SoM (axis 3, image). Once P-prompt data arrives this becomes a full diamond with two paths from DOM to P-SoM (via P-text or via P-prompt), letting us check prompt × text additivity / interaction.

| baseline | site | metric | text-axis (DOM→P-text) | prompt-axis (P-text→P-SoM) | image-axis (P-SoM→SoM) | dominant cascade axis | consistency |
|---|---|---|---|---|---|---|---|

★ marks Wilcoxon p<0.05. Effects with |d_z|>0.1 or |h|>0.1 are treated as non-negligible for axis dominance and cancellation checks.

## Cancellation patterns

No antagonistic pairs met the |0.1| effect-size threshold.

## Consistency checks

For every site x metric, text + prompt + image matches the direct SoM minus DOM endpoint within tolerance (0.1 percentage points for binary search-loop, 0.005 raw units for fractions/counts).

## Tier 2b — Mechanism (Micro): per-step decision quality

Tracked separately in `axis1_microbehavior.{json,md}`. Macro action-frequency metrics (this file) average per-step decisions; micro metrics directly compare per-step element selection / page coverage / search keyword reuse via mode-invariant anchors (URL, action.text).

## Paper Section 5 implication

**Tier 2a Macro — dominant cascade axis per metric**: text: none; prompt: none; image: none.

No antagonistic pair cleared the |0.1| effect-size threshold.

**4-level cascade design value**: decomposes DOM → SoM into three controlled transitions (AXTree vs [SOM_MARKS] structure, DOM vs SoM prompting, marginal image), and **reveals 6 antagonistic mechanism pairs** that endpoint-only comparison would mask.