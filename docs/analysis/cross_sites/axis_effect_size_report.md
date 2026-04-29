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
| B0 | reddit | search loop | h=-0.33★; -16.19 pp; [-23.33, -8.57] | h=-0.09; -4.29 pp; [-11.43, +2.86] | ✅ | — |
| B0 | reddit | type fraction | d_z=-0.41★; -8.92 pp; [-11.73, -6.12] | d_z=-0.11; -2.50 pp; [-5.42, +0.33] | ✅ | ✅ |
| B0 | reddit | scroll fraction | d_z=-0.01; -0.35 pp; [-3.83, +3.16] | d_z=-0.14; -3.41 pp; [-6.80, -0.27] | — | ✅ |
| B0 | reddit | self-correction | d_z=-0.02; -0.06; [-0.38, +0.27] | d_z=+0.02; +0.05; [-0.26, +0.34] | — | — |
| B0 | reddit | click fraction | d_z=-0.05; -1.18 pp; [-4.23, +1.81] | d_z=-0.11★; -3.08 pp; [-6.67, +0.94] | — | ✅ |
| B0 | reddit | finish rate | h=-0.10; -4.76 pp; [-12.38, +2.86] | h=+0.22★; +10.95 pp; [+3.33, +18.11] | — | ✅ |
| B0 | reddit | step count | d_z=-0.25★; -2.80; [-4.28, -1.34] | d_z=-0.16★; -1.85; [-3.38, -0.33] | ✅ | ✅ |
| B0 | reddit | action repeat | d_z=+0.10; +4.00 pp; [-1.36, +9.46] | d_z=-0.05; -1.84 pp; [-7.45, +3.92] | ✅ | — |
| B0 | classifieds | search loop | h=+0.00; +0.00 pp; [-4.70, +4.70] | h=-0.27★; -11.97 pp; [-17.52, -6.83] | — | ✅ |
| B0 | classifieds | type fraction | d_z=-0.10; -2.24 pp; [-5.22, +0.56] | d_z=+0.03; +0.65 pp; [-2.14, +3.60] | — | — |
| B0 | classifieds | scroll fraction | d_z=+0.11; +2.67 pp; [-0.63, +5.91] | d_z=-0.31★; -9.55 pp; [-13.58, -5.55] | ✅ | ✅ |
| B0 | classifieds | self-correction | d_z=+0.02; +0.05; [-0.22, +0.35] | d_z=-0.03; -0.08; [-0.41, +0.27] | — | — |
| B0 | classifieds | click fraction | d_z=+0.07; +1.41 pp; [-1.09, +4.03] | d_z=-0.07; -1.59 pp; [-4.83, +1.72] | — | — |
| B0 | classifieds | finish rate | h=-0.09; -4.27 pp; [-12.39, +3.85] | h=+0.57★; +26.50 pp; [+18.80, +34.19] | — | ✅ |
| B0 | classifieds | step count | d_z=+0.05; +0.48; [-0.75, +1.74] | d_z=-0.33★; -3.45; [-4.83, -2.12] | — | ✅ |
| B0 | classifieds | action repeat | d_z=+0.12★; +4.26 pp; [-0.32, +8.66] | d_z=-0.42★; -17.08 pp; [-22.40, -11.90] | ✅ | ✅ |
| B1 | classifieds | search loop | h=-0.18★; -7.83 pp; [-13.48, -2.17] | h=-0.17★; -8.26 pp; [-14.35, -2.17] | ✅ | ✅ |
| B1 | classifieds | type fraction | d_z=-0.42★; -10.65 pp; [-14.24, -7.27] | d_z=-0.10; -2.40 pp; [-5.32, +0.55] | ✅ | ✅ |
| B1 | classifieds | scroll fraction | d_z=-0.13; -3.44 pp; [-6.68, -0.24] | d_z=-0.01; -0.20 pp; [-3.84, +3.29] | ✅ | — |
| B1 | classifieds | self-correction | d_z=-0.12; -0.17; [-0.37, +0.00] | d_z=+0.14; +0.19; [+0.02, +0.37] | ✅ | ✅ |
| B1 | classifieds | click fraction | d_z=+0.57★; +16.94 pp; [+13.05, +20.84] | d_z=-0.11★; -3.62 pp; [-8.04, +0.86] | ✅ | ✅ |
| B1 | classifieds | finish rate | h=-0.14; -6.52 pp; [-13.48, +0.43] | h=+0.19★; +9.13 pp; [+2.17, +16.10] | ✅ | ✅ |
| B1 | classifieds | step count | d_z=-0.18★; -1.98; [-3.38, -0.60] | d_z=-0.17★; -1.97; [-3.51, -0.48] | ✅ | ✅ |
| B1 | classifieds | action repeat | d_z=+0.16★; +5.38 pp; [+1.05, +9.58] | d_z=-0.04; -1.62 pp; [-6.43, +2.96] | ✅ | — |

**P-SoM independence verdict** (cells where P-SoM differs from BOTH DOM and SoM, |effect|>0.1):
- **Independent on**: type_frac@B0/reddit, n_steps@B0/reddit, scroll_frac@B0/classifieds, action_repeat_frac@B0/classifieds, search_loop@B1/classifieds, type_frac@B1/classifieds, selfcorr_count@B1/classifieds, click_frac@B1/classifieds, finish_rate@B1/classifieds, n_steps@B1/classifieds
- Distinct from DOM only (≈ SoM-like): search_loop@B0/reddit, action_repeat_frac@B0/reddit, scroll_frac@B1/classifieds, action_repeat_frac@B1/classifieds
- Distinct from SoM only (≈ DOM-like): scroll_frac@B0/reddit, click_frac@B0/reddit, finish_rate@B0/reddit, search_loop@B0/classifieds, finish_rate@B0/classifieds, n_steps@B0/classifieds
- Indistinct from both endpoints: selfcorr_count@B0/reddit, type_frac@B0/classifieds, selfcorr_count@B0/classifieds, click_frac@B0/classifieds

## Tier 2a — Mechanism (Macro): cascade decomposition

DOM → P-text (axis 1, text only) → P-SoM (axis 2, prompt only) → SoM (axis 3, image). Once P-prompt data arrives this becomes a full diamond with two paths from DOM to P-SoM (via P-text or via P-prompt), letting us check prompt × text additivity / interaction.

| baseline | site | metric | text-axis (DOM→P-text) | prompt-axis (P-text→P-SoM) | image-axis (P-SoM→SoM) | dominant cascade axis | consistency |
|---|---|---|---|---|---|---|---|
| B0 | reddit | search loop | h=-0.05; -2.38 pp; [-8.57, +3.33] | h=-0.28★; -13.81 pp; [-20.48, -7.14] | h=-0.09; -4.29 pp; [-11.43, +2.86] | prompt | pass |
| B0 | reddit | type fraction | d_z=-0.12; -2.34 pp; [-4.99, +0.22] | d_z=-0.36★; -6.58 pp; [-9.10, -4.08] | d_z=-0.11; -2.50 pp; [-5.42, +0.33] | prompt | pass |
| B0 | reddit | scroll fraction | d_z=+0.15; +3.44 pp; [+0.30, +6.63] | d_z=-0.15★; -3.79 pp; [-7.15, -0.36] | d_z=-0.14; -3.41 pp; [-6.80, -0.27] | prompt | pass |
| B0 | reddit | self-correction | d_z=-0.08; -0.17; [-0.45, +0.09] | d_z=+0.05; +0.11; [-0.14, +0.40] | d_z=+0.02; +0.05; [-0.26, +0.34] | neither (all small) | pass |
| B0 | reddit | click fraction | d_z=-0.05; -1.04 pp; [-4.18, +2.08] | d_z=-0.01; -0.14 pp; [-3.62, +3.14] | d_z=-0.11★; -3.08 pp; [-6.67, +0.94] | image | pass |
| B0 | reddit | finish rate | h=-0.05; -2.38 pp; [-9.06, +3.81] | h=-0.05; -2.38 pp; [-9.52, +4.76] | h=+0.22★; +10.95 pp; [+3.33, +18.11] | image | pass |
| B0 | reddit | step count | d_z=-0.11; -1.25; [-2.71, +0.19] | d_z=-0.15★; -1.55; [-3.06, -0.12] | d_z=-0.16★; -1.85; [-3.38, -0.33] | image | pass |
| B0 | reddit | action repeat | d_z=+0.14; +4.64 pp; [+0.13, +9.15] | d_z=-0.02; -0.64 pp; [-5.76, +4.50] | d_z=-0.05; -1.84 pp; [-7.45, +3.92] | text | pass |
| B0 | classifieds | search loop | h=-0.09; -3.85 pp; [-8.55, +0.85] | h=+0.09; +3.85 pp; [-0.43, +7.69] | h=-0.27★; -11.97 pp; [-17.52, -6.83] | image | pass |
| B0 | classifieds | type fraction | d_z=+0.06; +1.49 pp; [-1.67, +4.37] | d_z=-0.20★; -3.73 pp; [-5.92, -1.43] | d_z=+0.03; +0.65 pp; [-2.14, +3.60] | prompt | pass |
| B0 | classifieds | scroll fraction | d_z=+0.04; +0.85 pp; [-1.92, +3.77] | d_z=+0.08; +1.82 pp; [-0.96, +4.69] | d_z=-0.31★; -9.55 pp; [-13.58, -5.55] | image | pass |
| B0 | classifieds | self-correction | d_z=-0.14★; -0.29; [-0.54, -0.03] | d_z=+0.17★; +0.34; [+0.09, +0.60] | d_z=-0.03; -0.08; [-0.41, +0.27] | prompt | pass |
| B0 | classifieds | click fraction | d_z=-0.04; -0.74 pp; [-3.08, +1.59] | d_z=+0.11; +2.14 pp; [-0.41, +4.78] | d_z=-0.07; -1.59 pp; [-4.83, +1.72] | prompt | pass |
| B0 | classifieds | finish rate | h=+0.04; +2.14 pp; [-4.70, +8.97] | h=-0.13; -6.41 pp; [-12.82, +0.00] | h=+0.57★; +26.50 pp; [+18.80, +34.19] | image | pass |
| B0 | classifieds | step count | d_z=-0.04; -0.40; [-1.68, +0.81] | d_z=+0.11; +0.88; [-0.05, +1.82] | d_z=-0.33★; -3.45; [-4.83, -2.12] | image | pass |
| B0 | classifieds | action repeat | d_z=+0.01; +0.24 pp; [-3.77, +3.97] | d_z=+0.13; +4.02 pp; [+0.35, +8.10] | d_z=-0.42★; -17.08 pp; [-22.40, -11.90] | image | pass |

★ marks Wilcoxon p<0.05. Effects with |d_z|>0.1 or |h|>0.1 are treated as non-negligible for axis dominance and cancellation checks.

## Cancellation patterns

The following site/metric pairs are antagonistic: two cascade axes have opposite-signed effects and both exceed |0.1| effect size. These are exactly the cases where a DOM-vs-SoM endpoint comparison can mask the internal mechanism.

- B0 reddit / scroll fraction: text vs prompt (d_z=+0.15 vs -0.15) -> antagonistic
- B0 reddit / scroll fraction: text vs image (d_z=+0.15 vs -0.14) -> antagonistic
- B0 classifieds / self-correction: text vs prompt (d_z=-0.14 vs +0.17) -> antagonistic
- B0 classifieds / finish rate: prompt vs image (h=-0.13 vs +0.57) -> antagonistic
- B0 classifieds / step count: prompt vs image (d_z=+0.11 vs -0.33) -> antagonistic
- B0 classifieds / action repeat: prompt vs image (d_z=+0.13 vs -0.42) -> antagonistic

## Consistency checks

For every site x metric, text + prompt + image matches the direct SoM minus DOM endpoint within tolerance (0.1 percentage points for binary search-loop, 0.005 raw units for fractions/counts).

## Tier 2b — Mechanism (Micro): per-step decision quality

Tracked separately in `axis1_microbehavior.{json,md}`. Macro action-frequency metrics (this file) average per-step decisions; micro metrics directly compare per-step element selection / page coverage / search keyword reuse via mode-invariant anchors (URL, action.text).

## Paper Section 5 implication

**Tier 2a Macro — dominant cascade axis per metric**: text: action_repeat_frac@B0/reddit; prompt: search_loop@B0/reddit, type_frac@B0/reddit, scroll_frac@B0/reddit, type_frac@B0/classifieds, selfcorr_count@B0/classifieds, click_frac@B0/classifieds; image: click_frac@B0/reddit, finish_rate@B0/reddit, n_steps@B0/reddit, search_loop@B0/classifieds, scroll_frac@B0/classifieds, finish_rate@B0/classifieds, n_steps@B0/classifieds, action_repeat_frac@B0/classifieds, search_loop@B1/classifieds, type_frac@B1/classifieds, selfcorr_count@B1/classifieds, click_frac@B1/classifieds, finish_rate@B1/classifieds, n_steps@B1/classifieds.

**Antagonistic pairs** (axes pulling opposite directions, hidden by DOM↔SoM endpoint comparison): text_vs_prompt@scroll_frac@B0/reddit; text_vs_image@scroll_frac@B0/reddit; text_vs_prompt@selfcorr_count@B0/classifieds; prompt_vs_image@finish_rate@B0/classifieds; prompt_vs_image@n_steps@B0/classifieds; prompt_vs_image@action_repeat_frac@B0/classifieds.

**4-level cascade design value**: decomposes DOM → SoM into three controlled transitions (AXTree vs [SOM_MARKS] structure, DOM vs SoM prompting, marginal image), and **reveals 6 antagonistic mechanism pairs** that endpoint-only comparison would mask.