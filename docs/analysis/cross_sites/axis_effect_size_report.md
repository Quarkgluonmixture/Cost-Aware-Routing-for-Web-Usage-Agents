# Axis Effect Size Ablation

Hierarchical analysis with two tiers:

- **Tier 1 (Hook)** — 3-mode coarse validation (DOM / P-SoM / SoM): establishes that **P-SoM is an independent routing arm** distinct from both DOM and SoM endpoints. Does not require P-text/P-prompt data.
- **Tier 2 (Mechanism)** — 5-mode diamond (DOM / P-text / P-prompt / P-SoM / SoM): explains *why* P-SoM is distinct by decomposing the compound DOM→P-SoM transition into text-axis and prompt-axis sub-effects. Splits into:
  - **2a Macro** — action-type frequencies (this file): finish rate, step count, search/type/scroll/click %, action repeat, self-correction.
  - **2b Micro** — per-step decision quality (separate analysis): URL trajectory Jaccard, target-page hit rate, search keyword reuse, first-action divergence.

**Current data status**: computed from the run registry at grade `paper-grade`; Tier 2b Micro is tracked separately in `axis1_microbehavior.{json,md}`.

> **Episodes dropped for steps↔summary identity mismatch**: B0/reddit/Phantom-SoM: tasks [87, 149]. Contrasts touching those arms pair on the intersection and so lose the task on both legs.

## Tier 1 — Hook: is P-SoM distinct from both DOM and SoM?

| baseline | site | metric | DOM→P-SoM (compound) | P-SoM→SoM (image) | distinct from DOM? | distinct from SoM? |
|---|---|---|---|---|---|---|
| B0 | reddit | search loop | h=-0.27★; -13.43 pp; [-20.40, -5.97] | h=+0.05; +2.49 pp; [-4.48, +9.45] | ✅ | — |
| B0 | reddit | type fraction | d_z=-0.07; -1.12 pp; [-3.41, +1.19] | d_z=-0.08; -1.43 pp; [-3.80, +0.78] | — | — |
| B0 | reddit | scroll fraction | d_z=-0.18★; -4.20 pp; [-7.31, -0.97] | d_z=-0.15★; -2.49 pp; [-4.79, -0.06] | ✅ | ✅ |
| B0 | reddit | self-correction | d_z=+0.22★; +0.76; [+0.32, +1.24] | d_z=-0.26★; -0.69; [-1.08, -0.34] | ✅ | ✅ |
| B0 | reddit | click fraction | d_z=-0.01; -0.23 pp; [-3.69, +3.19] | d_z=+0.08; +2.11 pp; [-1.40, +5.64] | — | — |
| B0 | reddit | finish rate | h=-0.20★; -9.95 pp; [-17.91, -2.49] | h=+0.16★; +7.96 pp; [+0.00, +16.42] | ✅ | ✅ |
| B0 | reddit | step count | d_z=+0.25★; +2.72; [+1.20, +4.21] | d_z=-0.25★; -2.82; [-4.42, -1.31] | ✅ | ✅ |
| B0 | reddit | action repeat | d_z=+0.20★; +7.35 pp; [+2.40, +12.14] | d_z=-0.03; -1.08 pp; [-5.68, +3.37] | ✅ | — |
| B0 | classifieds | search loop | h=-0.06; -2.23 pp; [-7.14, +2.24] | h=-0.23★; -9.82 pp; [-15.18, -4.46] | — | ✅ |
| B0 | classifieds | type fraction | d_z=-0.14; -2.38 pp; [-4.61, -0.10] | d_z=+0.03; +0.45 pp; [-1.47, +2.43] | ✅ | — |
| B0 | classifieds | scroll fraction | d_z=+0.13; +2.50 pp; [-0.01, +4.97] | d_z=-0.25★; -5.39 pp; [-8.09, -2.61] | ✅ | ✅ |
| B0 | classifieds | self-correction | d_z=-0.02; -0.04; [-0.27, +0.20] | d_z=-0.09; -0.20; [-0.47, +0.09] | — | — |
| B0 | classifieds | click fraction | d_z=-0.06; -0.96 pp; [-3.07, +1.32] | d_z=+0.04; +0.69 pp; [-1.87, +3.33] | — | — |
| B0 | classifieds | finish rate | h=-0.11; -4.91 pp; [-10.71, +0.45] | h=+0.12; +5.36 pp; [-0.89, +11.61] | ✅ | ✅ |
| B0 | classifieds | step count | d_z=+0.07; +0.62; [-0.61, +1.88] | d_z=-0.25★; -2.57; [-3.93, -1.21] | — | ✅ |
| B0 | classifieds | action repeat | d_z=+0.03; +0.76 pp; [-2.52, +4.02] | d_z=-0.12; -3.18 pp; [-6.73, +0.41] | — | ✅ |
| B1 | reddit | search loop | h=-0.08; -3.94 pp; [-9.85, +1.97] | h=-0.30★; -14.78 pp; [-21.67, -7.39] | — | ✅ |
| B1 | reddit | type fraction | d_z=-0.30★; -7.25 pp; [-10.60, -3.95] | d_z=-0.16★; -3.68 pp; [-6.77, -0.45] | ✅ | ✅ |
| B1 | reddit | scroll fraction | d_z=-0.07; -1.03 pp; [-2.94, +0.83] | d_z=+0.16; +3.94 pp; [+0.76, +7.41] | — | ✅ |
| B1 | reddit | self-correction | d_z=-0.12; -0.26; [-0.56, +0.03] | d_z=-0.06; -0.10; [-0.31, +0.12] | ✅ | — |
| B1 | reddit | click fraction | d_z=+0.37★; +10.90 pp; [+6.89, +14.92] | d_z=-0.02; -0.63 pp; [-5.90, +4.84] | ✅ | — |
| B1 | reddit | finish rate | h=-0.21★; -9.36 pp; [-16.26, -2.96] | h=+0.22★; +9.85 pp; [+2.96, +16.75] | ✅ | ✅ |
| B1 | reddit | step count | d_z=+0.17★; +1.73; [+0.35, +3.13] | d_z=-0.23★; -2.79; [-4.45, -1.17] | ✅ | ✅ |
| B1 | reddit | action repeat | d_z=+0.25★; +7.92 pp; [+3.35, +12.36] | d_z=+0.02; +0.77 pp; [-4.38, +6.02] | ✅ | — |
| B1 | classifieds | search loop | h=-0.09; -4.02 pp; [-9.38, +1.79] | h=-0.20★; -9.38 pp; [-15.62, -3.57] | — | ✅ |
| B1 | classifieds | type fraction | d_z=-0.45★; -13.27 pp; [-17.15, -9.43] | d_z=-0.01; -0.14 pp; [-3.66, +3.44] | ✅ | — |
| B1 | classifieds | scroll fraction | d_z=-0.06; -1.70 pp; [-5.12, +1.62] | d_z=-0.06; -1.70 pp; [-5.22, +1.92] | — | — |
| B1 | classifieds | self-correction | d_z=-0.01; -0.01; [-0.18, +0.16] | d_z=+0.15★; +0.22; [+0.03, +0.42] | — | ✅ |
| B1 | classifieds | click fraction | d_z=+0.61★; +19.03 pp; [+14.96, +23.20] | d_z=-0.17★; -6.75 pp; [-12.16, -1.75] | ✅ | ✅ |
| B1 | classifieds | finish rate | h=+0.03; +1.34 pp; [-6.25, +8.49] | h=+0.17★; +8.48 pp; [+0.89, +15.62] | — | ✅ |
| B1 | classifieds | step count | d_z=-0.01; -0.12; [-1.63, +1.39] | d_z=-0.25★; -3.25; [-4.92, -1.60] | — | ✅ |
| B1 | classifieds | action repeat | d_z=+0.05; +1.87 pp; [-3.22, +7.00] | d_z=-0.17★; -7.10 pp; [-12.65, -1.81] | — | ✅ |
| B2 | reddit | search loop | h=-0.02; -0.99 pp; [-7.88, +5.91] | h=-0.18★; -6.90 pp; [-13.79, +0.00] | — | ✅ |
| B2 | reddit | type fraction | d_z=-0.07; -1.47 pp; [-4.16, +1.48] | d_z=+0.04; +0.91 pp; [-1.86, +3.64] | — | — |
| B2 | reddit | scroll fraction | d_z=+0.05; +0.62 pp; [-0.98, +2.25] | d_z=-0.04; -0.46 pp; [-2.02, +1.12] | — | — |
| B2 | reddit | self-correction | d_z=+0.02; +0.02; [-0.15, +0.20] | d_z=+0.02; +0.02; [-0.15, +0.21] | — | — |
| B2 | reddit | click fraction | d_z=-0.05; -1.56 pp; [-6.07, +3.08] | d_z=-0.02; -0.59 pp; [-5.72, +4.04] | — | — |
| B2 | reddit | finish rate | h=-0.10; -2.46 pp; [-6.90, +1.48] | h=+0.38★; +11.33 pp; [+5.91, +16.75] | ✅ | ✅ |
| B2 | reddit | step count | d_z=-0.05; -0.52; [-1.82, +0.82] | d_z=-0.14★; -1.53; [-3.04, -0.05] | — | ✅ |
| B2 | reddit | action repeat | d_z=-0.17★; -4.51 pp; [-8.17, -0.79] | d_z=+0.12★; +3.47 pp; [-0.63, +7.29] | ✅ | ✅ |
| B2 | classifieds | search loop | h=+0.08; +3.57 pp; [-4.02, +10.71] | h=-0.14; -6.70 pp; [-14.73, +1.34] | — | ✅ |
| B2 | classifieds | type fraction | d_z=-0.29★; -6.04 pp; [-8.73, -3.29] | d_z=+0.30★; +6.11 pp; [+3.45, +8.88] | ✅ | ✅ |
| B2 | classifieds | scroll fraction | d_z=+0.03; +0.22 pp; [-0.83, +1.38] | d_z=+0.00; +0.02 pp; [-0.95, +0.99] | — | — |
| B2 | classifieds | self-correction | d_z=-0.02; -0.02; [-0.12, +0.08] | d_z=-0.11; -0.07; [-0.16, +0.01] | — | ✅ |
| B2 | classifieds | click fraction | d_z=+0.01; +0.34 pp; [-3.64, +4.41] | d_z=-0.10; -3.53 pp; [-8.52, +1.15] | — | — |
| B2 | classifieds | finish rate | h=-0.26★; -7.14 pp; [-11.61, -2.68] | h=+0.33★; +9.82 pp; [+5.80, +14.73] | ✅ | ✅ |
| B2 | classifieds | step count | d_z=+0.11★; +1.00; [-0.16, +2.14] | d_z=-0.36★; -4.02; [-5.50, -2.64] | ✅ | ✅ |
| B2 | classifieds | action repeat | d_z=+0.06; +1.89 pp; [-2.04, +5.77] | d_z=-0.07; -2.16 pp; [-6.46, +2.06] | — | — |

**P-SoM independence verdict** (cells where P-SoM differs from BOTH DOM and SoM, |effect|>0.1):
- **Independent on**: scroll_frac@B0/reddit, selfcorr_count@B0/reddit, finish_rate@B0/reddit, n_steps@B0/reddit, scroll_frac@B0/classifieds, finish_rate@B0/classifieds, type_frac@B1/reddit, finish_rate@B1/reddit, n_steps@B1/reddit, click_frac@B1/classifieds, finish_rate@B2/reddit, action_repeat_frac@B2/reddit, type_frac@B2/classifieds, finish_rate@B2/classifieds, n_steps@B2/classifieds

> **Multiplicity.** That count asks only |effect| > 0.1, across 48 (cell, metric) combinations and 96 Wilcoxon tests. Requiring **both** legs to also clear a correction applied jointly over all legs: **7 survive Benjamini-Hochberg** (FDR 0.05) and **2 survive Holm** (FWER 0.05), against 15 on effect size alone. The BH set spans 4 of the 6 cells, so it is not one cell's accident: selfcorr_count@B0/reddit, n_steps@B0/reddit, type_frac@B1/reddit, finish_rate@B1/reddit, click_frac@B1/classifieds, type_frac@B2/classifieds, finish_rate@B2/classifieds. Report the corrected count, not the bare one.

> **Distinct from both endpoints is not the same as independent.** A mode that *interpolates* between DOM and SoM also differs from both. P-SoM is off the DOM–SoM segment — an extremum rather than a midpoint — exactly when the two legs disagree in sign. Of the 7 BH survivors, **6 are off the segment** and 1 interpolate (type_frac@B1/reddit). The off-segment count is the one that supports an independent arm; on `finish_rate@B1/reddit` P-SoM sits about 9pp below *both* endpoints while the endpoints differ from each other by 0.5pp.
- Distinct from DOM only (≈ SoM-like): search_loop@B0/reddit, action_repeat_frac@B0/reddit, type_frac@B0/classifieds, selfcorr_count@B1/reddit, click_frac@B1/reddit, action_repeat_frac@B1/reddit, type_frac@B1/classifieds
- Distinct from SoM only (≈ DOM-like): search_loop@B0/classifieds, n_steps@B0/classifieds, action_repeat_frac@B0/classifieds, search_loop@B1/reddit, scroll_frac@B1/reddit, search_loop@B1/classifieds, selfcorr_count@B1/classifieds, finish_rate@B1/classifieds, n_steps@B1/classifieds, action_repeat_frac@B1/classifieds, search_loop@B2/reddit, n_steps@B2/reddit, search_loop@B2/classifieds, selfcorr_count@B2/classifieds
- Indistinct from both endpoints: type_frac@B0/reddit, click_frac@B0/reddit, selfcorr_count@B0/classifieds, click_frac@B0/classifieds, scroll_frac@B1/classifieds, type_frac@B2/reddit, scroll_frac@B2/reddit, selfcorr_count@B2/reddit, click_frac@B2/reddit, scroll_frac@B2/classifieds, click_frac@B2/classifieds, action_repeat_frac@B2/classifieds

## Tier 2a — Mechanism (Macro): cascade decomposition

DOM → P-text (axis 1, text only) → P-SoM (axis 2, prompt only) → SoM (axis 3, image). The second route, through P-prompt, closes this into a full diamond; the two paths and their additivity are reported in Tier 2b below.

| baseline | site | metric | text-axis (DOM→P-text) | prompt-axis (P-text→P-SoM) | image-axis (P-SoM→SoM) | dominant cascade axis | consistency |
|---|---|---|---|---|---|---|---|
| B0 | reddit | search loop | h=-0.17★; -8.37 pp; [-14.78, -1.97] | h=-0.10; -4.98 pp; [-10.45, +0.00] | h=+0.05; +2.49 pp; [-4.48, +9.45] | text | pass |
| B0 | reddit | type fraction | d_z=+0.03; +0.49 pp; [-1.84, +2.98] | d_z=-0.10; -1.60 pp; [-3.92, +0.49] | d_z=-0.08; -1.43 pp; [-3.80, +0.78] | neither (all small) | pass |
| B0 | reddit | scroll fraction | d_z=-0.21★; -4.88 pp; [-8.25, -1.63] | d_z=+0.04; +0.77 pp; [-1.83, +3.25] | d_z=-0.15★; -2.49 pp; [-4.79, -0.06] | text | pass |
| B0 | reddit | self-correction | d_z=+0.28★; +0.90; [+0.48, +1.36] | d_z=-0.07; -0.16; [-0.53, +0.17] | d_z=-0.26★; -0.69; [-1.08, -0.34] | text | fail |
| B0 | reddit | click fraction | d_z=-0.06; -1.41 pp; [-4.70, +1.77] | d_z=+0.05; +1.09 pp; [-1.64, +3.88] | d_z=+0.08; +2.11 pp; [-1.40, +5.64] | neither (all small) | pass |
| B0 | reddit | finish rate | h=-0.30★; -14.78 pp; [-22.17, -6.90] | h=+0.10; +4.98 pp; [-1.99, +12.44] | h=+0.16★; +7.96 pp; [+0.00, +16.42] | text | fail |
| B0 | reddit | step count | d_z=+0.28★; +3.00; [+1.49, +4.47] | d_z=-0.03; -0.32; [-1.69, +1.05] | d_z=-0.25★; -2.82; [-4.42, -1.31] | text | fail |
| B0 | reddit | action repeat | d_z=+0.23★; +8.76 pp; [+3.33, +14.21] | d_z=-0.06; -1.62 pp; [-5.71, +2.38] | d_z=-0.03; -1.08 pp; [-5.68, +3.37] | text | pass |
| B0 | classifieds | search loop | h=-0.01; -0.45 pp; [-4.91, +4.02] | h=-0.04; -1.79 pp; [-6.25, +2.68] | h=-0.23★; -9.82 pp; [-15.18, -4.46] | image | pass |
| B0 | classifieds | type fraction | d_z=-0.17★; -3.07 pp; [-5.42, -0.76] | d_z=+0.05; +0.69 pp; [-1.09, +2.60] | d_z=+0.03; +0.45 pp; [-1.47, +2.43] | text | pass |
| B0 | classifieds | scroll fraction | d_z=+0.07; +1.26 pp; [-1.14, +3.65] | d_z=+0.06; +1.25 pp; [-1.30, +3.86] | d_z=-0.25★; -5.39 pp; [-8.09, -2.61] | image | pass |
| B0 | classifieds | self-correction | d_z=+0.09; +0.20; [-0.09, +0.51] | d_z=-0.12; -0.24; [-0.50, +0.02] | d_z=-0.09; -0.20; [-0.47, +0.09] | prompt | pass |
| B0 | classifieds | click fraction | d_z=-0.00; -0.02 pp; [-2.10, +2.29] | d_z=-0.06; -0.93 pp; [-2.97, +1.05] | d_z=+0.04; +0.69 pp; [-1.87, +3.33] | neither (all small) | pass |
| B0 | classifieds | finish rate | h=-0.07; -3.12 pp; [-8.93, +2.68] | h=-0.04; -1.79 pp; [-8.04, +5.36] | h=+0.12; +5.36 pp; [-0.89, +11.61] | image | pass |
| B0 | classifieds | step count | d_z=+0.02; +0.22; [-1.03, +1.57] | d_z=+0.04; +0.40; [-1.02, +1.75] | d_z=-0.25★; -2.57; [-3.93, -1.21] | image | pass |
| B0 | classifieds | action repeat | d_z=-0.05; -1.40 pp; [-4.87, +2.25] | d_z=+0.09; +2.16 pp; [-1.08, +5.45] | d_z=-0.12; -3.18 pp; [-6.73, +0.41] | image | pass |
| B1 | reddit | search loop | h=+0.02; +0.99 pp; [-4.43, +6.90] | h=-0.10; -4.93 pp; [-11.33, +1.48] | h=-0.30★; -14.78 pp; [-21.67, -7.39] | image | pass |
| B1 | reddit | type fraction | d_z=+0.03; +0.73 pp; [-2.30, +3.90] | d_z=-0.33★; -7.98 pp; [-11.24, -4.80] | d_z=-0.16★; -3.68 pp; [-6.77, -0.45] | prompt | pass |
| B1 | reddit | scroll fraction | d_z=+0.01; +0.11 pp; [-1.86, +2.02] | d_z=-0.07; -1.14 pp; [-3.32, +1.07] | d_z=+0.16; +3.94 pp; [+0.76, +7.41] | image | pass |
| B1 | reddit | self-correction | d_z=-0.05; -0.12; [-0.46, +0.20] | d_z=-0.07; -0.13; [-0.38, +0.10] | d_z=-0.06; -0.10; [-0.31, +0.12] | neither (all small) | pass |
| B1 | reddit | click fraction | d_z=+0.02; +0.51 pp; [-3.24, +4.33] | d_z=+0.33★; +10.39 pp; [+6.12, +14.77] | d_z=-0.02; -0.63 pp; [-5.90, +4.84] | prompt | pass |
| B1 | reddit | finish rate | h=-0.22★; -9.85 pp; [-16.75, -2.96] | h=+0.01; +0.49 pp; [-5.42, +6.90] | h=+0.22★; +9.85 pp; [+2.96, +16.75] | text | pass |
| B1 | reddit | step count | d_z=+0.20★; +2.11; [+0.65, +3.56] | d_z=-0.04; -0.38; [-1.56, +0.74] | d_z=-0.23★; -2.79; [-4.45, -1.17] | image | pass |
| B1 | reddit | action repeat | d_z=+0.13★; +4.55 pp; [-0.11, +9.05] | d_z=+0.11; +3.37 pp; [-0.96, +7.64] | d_z=+0.02; +0.77 pp; [-4.38, +6.02] | text | pass |
| B1 | classifieds | search loop | h=+0.07; +2.68 pp; [-2.23, +8.04] | h=-0.16★; -6.70 pp; [-11.61, -1.79] | h=-0.20★; -9.38 pp; [-15.62, -3.57] | image | pass |
| B1 | classifieds | type fraction | d_z=+0.09; +2.22 pp; [-1.02, +5.44] | d_z=-0.55★; -15.49 pp; [-19.22, -11.83] | d_z=-0.01; -0.14 pp; [-3.66, +3.44] | prompt | pass |
| B1 | classifieds | scroll fraction | d_z=-0.10; -2.32 pp; [-5.43, +0.80] | d_z=+0.03; +0.62 pp; [-2.46, +3.68] | d_z=-0.06; -1.70 pp; [-5.22, +1.92] | neither (all small) | pass |
| B1 | classifieds | self-correction | d_z=-0.02; -0.02; [-0.18, +0.13] | d_z=+0.01; +0.01; [-0.14, +0.17] | d_z=+0.15★; +0.22; [+0.03, +0.42] | image | pass |
| B1 | classifieds | click fraction | d_z=+0.14; +3.50 pp; [+0.36, +6.72] | d_z=+0.53★; +15.52 pp; [+11.65, +19.50] | d_z=-0.17★; -6.75 pp; [-12.16, -1.75] | prompt | pass |
| B1 | classifieds | finish rate | h=-0.05; -2.68 pp; [-9.38, +3.57] | h=+0.08; +4.02 pp; [-3.12, +11.16] | h=+0.17★; +8.48 pp; [+0.89, +15.62] | image | pass |
| B1 | classifieds | step count | d_z=+0.10; +1.07; [-0.29, +2.47] | d_z=-0.11; -1.20; [-2.70, +0.23] | d_z=-0.25★; -3.25; [-4.92, -1.60] | image | pass |
| B1 | classifieds | action repeat | d_z=+0.05; +1.80 pp; [-2.71, +6.41] | d_z=+0.00; +0.07 pp; [-4.58, +4.94] | d_z=-0.17★; -7.10 pp; [-12.65, -1.81] | image | pass |
| B2 | reddit | search loop | h=+0.09; +3.94 pp; [-3.94, +11.34] | h=-0.12; -4.93 pp; [-12.32, +2.96] | h=-0.18★; -6.90 pp; [-13.79, +0.00] | image | pass |
| B2 | reddit | type fraction | d_z=-0.05; -0.97 pp; [-3.83, +1.95] | d_z=-0.02; -0.50 pp; [-3.59, +2.48] | d_z=+0.04; +0.91 pp; [-1.86, +3.64] | neither (all small) | pass |
| B2 | reddit | scroll fraction | d_z=-0.04; -0.37 pp; [-1.77, +1.05] | d_z=+0.09; +0.99 pp; [-0.43, +2.45] | d_z=-0.04; -0.46 pp; [-2.02, +1.12] | neither (all small) | pass |
| B2 | reddit | self-correction | d_z=+0.05; +0.06; [-0.09, +0.22] | d_z=-0.03; -0.03; [-0.24, +0.15] | d_z=+0.02; +0.02; [-0.15, +0.21] | neither (all small) | pass |
| B2 | reddit | click fraction | d_z=-0.19★; -6.34 pp; [-10.90, -1.85] | d_z=+0.14; +4.78 pp; [+0.26, +9.37] | d_z=-0.02; -0.59 pp; [-5.72, +4.04] | text | pass |
| B2 | reddit | finish rate | h=-0.06; -1.48 pp; [-5.42, +2.46] | h=-0.04; -0.99 pp; [-4.93, +2.46] | h=+0.38★; +11.33 pp; [+5.91, +16.75] | image | pass |
| B2 | reddit | step count | d_z=-0.13; -1.12; [-2.35, +0.09] | d_z=+0.06; +0.61; [-0.84, +2.11] | d_z=-0.14★; -1.53; [-3.04, -0.05] | image | pass |
| B2 | reddit | action repeat | d_z=-0.16; -4.23 pp; [-8.18, -0.56] | d_z=-0.01; -0.27 pp; [-4.19, +3.67] | d_z=+0.12★; +3.47 pp; [-0.63, +7.29] | text | pass |
| B2 | classifieds | search loop | h=-0.04; -1.79 pp; [-9.38, +5.80] | h=+0.11; +5.36 pp; [-1.79, +12.05] | h=-0.14; -6.70 pp; [-14.73, +1.34] | image | pass |
| B2 | classifieds | type fraction | d_z=+0.12; +2.59 pp; [-0.28, +5.50] | d_z=-0.41★; -8.63 pp; [-11.54, -5.92] | d_z=+0.30★; +6.11 pp; [+3.45, +8.88] | prompt | pass |
| B2 | classifieds | scroll fraction | d_z=+0.15★; +1.23 pp; [+0.21, +2.29] | d_z=-0.11; -1.00 pp; [-2.17, +0.15] | d_z=+0.00; +0.02 pp; [-0.95, +0.99] | text | pass |
| B2 | classifieds | self-correction | d_z=+0.10; +0.10; [-0.02, +0.25] | d_z=-0.11; -0.12; [-0.28, +0.02] | d_z=-0.11; -0.07; [-0.16, +0.01] | prompt | pass |
| B2 | classifieds | click fraction | d_z=-0.48★; -13.57 pp; [-17.40, -9.91] | d_z=+0.49★; +13.91 pp; [+10.26, +17.76] | d_z=-0.10; -3.53 pp; [-8.52, +1.15] | prompt | pass |
| B2 | classifieds | finish rate | h=-0.22★; -6.25 pp; [-11.16, -1.34] | h=-0.04; -0.89 pp; [-5.36, +3.12] | h=+0.33★; +9.82 pp; [+5.80, +14.73] | image | pass |
| B2 | classifieds | step count | d_z=-0.05; -0.54; [-1.91, +0.80] | d_z=+0.16; +1.54; [+0.29, +2.79] | d_z=-0.36★; -4.02; [-5.50, -2.64] | image | pass |
| B2 | classifieds | action repeat | d_z=-0.09; -2.79 pp; [-6.67, +1.06] | d_z=+0.16★; +4.68 pp; [+0.88, +8.50] | d_z=-0.07; -2.16 pp; [-6.46, +2.06] | prompt | pass |

★ marks Wilcoxon p<0.05. Effects with |d_z|>0.1 or |h|>0.1 are treated as non-negligible for axis dominance and cancellation checks.

## Tier 2b — Diamond: base-set consistency across the two routes

Two routes lead from DOM to P-SoM:

- **path A** DOM →(text)→ P-text →(prompt)→ P-SoM
- **path B** DOM →(prompt)→ P-prompt →(text)→ P-SoM

⚠️ **This is not an interaction test.** On mean differences the agreement is an algebraic identity — mean(P-text−DOM) + mean(P-SoM−P-text) = mean(P-SoM−DOM) — whenever the legs are averaged over the same tasks. A zero residual is therefore arithmetic and carries no evidence about text × prompt interaction. What a **non**-zero residual does carry is that the legs were averaged over *different* task sets, which is why the table is kept: it is a base-set consistency check that fires automatically. Testing for an interaction would require comparing effect sizes or fitting an interaction term, and nothing on this page does that.

| baseline | site | metric | path A | path B | compound | A−comp | B−comp | same base set? |
|---|---|---|---|---|---|---|---|---|
| B0 | reddit | search loop | -13.35 | -13.39 | -13.43 | +0.083 | +0.044 | ✅ |
| B0 | reddit | type fraction | -0.01 | -0.01 | -0.01 | +0.000 | -0.000 | ✅ |
| B0 | reddit | scroll fraction | -0.04 | -0.04 | -0.04 | +0.001 | +0.002 | ✅ |
| B0 | reddit | self-correction | +0.74 | +0.76 | +0.76 | -0.019 | +0.002 | ⚠️ n differs across legs |
| B0 | reddit | click fraction | -0.00 | -0.00 | -0.00 | -0.001 | -0.002 | ✅ |
| B0 | reddit | finish rate | -9.80 | -9.98 | -9.95 | +0.147 | -0.029 | ⚠️ n differs across legs |
| B0 | reddit | step count | +2.68 | +2.73 | +2.72 | -0.035 | +0.013 | ⚠️ n differs across legs |
| B0 | reddit | action repeat | +0.07 | +0.07 | +0.07 | -0.002 | +0.000 | ✅ |
| B0 | classifieds | search loop | -2.23 | -2.23 | -2.23 | +0.000 | +0.000 | ✅ |
| B0 | classifieds | type fraction | -0.02 | -0.02 | -0.02 | +0.000 | +0.000 | ✅ |
| B0 | classifieds | scroll fraction | +0.03 | +0.03 | +0.03 | +0.000 | -0.000 | ✅ |
| B0 | classifieds | self-correction | -0.04 | -0.04 | -0.04 | +0.000 | -0.000 | ✅ |
| B0 | classifieds | click fraction | -0.01 | -0.01 | -0.01 | +0.000 | +0.000 | ✅ |
| B0 | classifieds | finish rate | -4.91 | -4.91 | -4.91 | +0.000 | +0.000 | ✅ |
| B0 | classifieds | step count | +0.62 | +0.62 | +0.62 | +0.000 | -0.000 | ✅ |
| B0 | classifieds | action repeat | +0.01 | +0.01 | +0.01 | +0.000 | +0.000 | ✅ |
| B1 | reddit | search loop | -3.94 | -3.94 | -3.94 | -0.000 | +0.000 | ✅ |
| B1 | reddit | type fraction | -0.07 | -0.07 | -0.07 | +0.000 | -0.000 | ✅ |
| B1 | reddit | scroll fraction | -0.01 | -0.01 | -0.01 | +0.000 | +0.000 | ✅ |
| B1 | reddit | self-correction | -0.26 | -0.26 | -0.26 | +0.000 | +0.000 | ✅ |
| B1 | reddit | click fraction | +0.11 | +0.11 | +0.11 | +0.000 | +0.000 | ✅ |
| B1 | reddit | finish rate | -9.36 | -9.36 | -9.36 | +0.000 | +0.000 | ✅ |
| B1 | reddit | step count | +1.73 | +1.73 | +1.73 | +0.000 | +0.000 | ✅ |
| B1 | reddit | action repeat | +0.08 | +0.08 | +0.08 | -0.000 | +0.000 | ✅ |
| B1 | classifieds | search loop | -4.02 | -4.02 | -4.02 | +0.000 | +0.000 | ✅ |
| B1 | classifieds | type fraction | -0.13 | -0.13 | -0.13 | +0.000 | +0.000 | ✅ |
| B1 | classifieds | scroll fraction | -0.02 | -0.02 | -0.02 | +0.000 | +0.000 | ✅ |
| B1 | classifieds | self-correction | -0.01 | -0.01 | -0.01 | +0.000 | +0.000 | ✅ |
| B1 | classifieds | click fraction | +0.19 | +0.19 | +0.19 | +0.000 | +0.000 | ✅ |
| B1 | classifieds | finish rate | +1.34 | +1.34 | +1.34 | +0.000 | +0.000 | ✅ |
| B1 | classifieds | step count | -0.12 | -0.12 | -0.12 | +0.000 | +0.000 | ✅ |
| B1 | classifieds | action repeat | +0.02 | +0.02 | +0.02 | +0.000 | -0.000 | ✅ |
| B2 | reddit | search loop | -0.99 | -0.99 | -0.99 | +0.000 | +0.000 | ✅ |
| B2 | reddit | type fraction | -0.01 | -0.01 | -0.01 | +0.000 | -0.000 | ✅ |
| B2 | reddit | scroll fraction | +0.01 | +0.01 | +0.01 | -0.000 | +0.000 | ✅ |
| B2 | reddit | self-correction | +0.02 | +0.02 | +0.02 | -0.000 | -0.000 | ✅ |
| B2 | reddit | click fraction | -0.02 | -0.02 | -0.02 | +0.000 | +0.000 | ✅ |
| B2 | reddit | finish rate | -2.46 | -2.46 | -2.46 | +0.000 | +0.000 | ✅ |
| B2 | reddit | step count | -0.52 | -0.52 | -0.52 | +0.000 | +0.000 | ✅ |
| B2 | reddit | action repeat | -0.05 | -0.05 | -0.05 | +0.000 | +0.000 | ✅ |
| B2 | classifieds | search loop | +3.57 | +3.57 | +3.57 | +0.000 | +0.000 | ✅ |
| B2 | classifieds | type fraction | -0.06 | -0.06 | -0.06 | +0.000 | +0.000 | ✅ |
| B2 | classifieds | scroll fraction | +0.00 | +0.00 | +0.00 | +0.000 | +0.000 | ✅ |
| B2 | classifieds | self-correction | -0.02 | -0.02 | -0.02 | -0.000 | +0.000 | ✅ |
| B2 | classifieds | click fraction | +0.00 | +0.00 | +0.00 | -0.000 | +0.000 | ✅ |
| B2 | classifieds | finish rate | -7.14 | -7.14 | -7.14 | -0.000 | -0.000 | ✅ |
| B2 | classifieds | step count | +1.00 | +1.00 | +1.00 | +0.000 | +0.000 | ✅ |
| B2 | classifieds | action repeat | +0.02 | +0.02 | +0.02 | -0.000 | +0.000 | ✅ |

**The identity holds in 45 of 48 (cell × metric) combinations.** Where it does not, the legs were averaged over different task sets, not over a world containing an interaction.
The rows that miss are exactly the ones on the B0·reddit P-SoM arm, whose legs are summed over 201 tasks against 203 on the others (the two identity-mismatched episodes). The residual there is the base-set difference and nothing else.

## Cancellation patterns

The following site/metric pairs are antagonistic: two cascade axes have opposite-signed effects and both exceed |0.1| effect size. These are exactly the cases where a DOM-vs-SoM endpoint comparison can mask the internal mechanism.

- B0 reddit / self-correction: text vs image (d_z=+0.28 vs -0.26) -> antagonistic
- B0 reddit / finish rate: text vs prompt (h=-0.30 vs +0.10) -> antagonistic
- B0 reddit / finish rate: text vs image (h=-0.30 vs +0.16) -> antagonistic
- B0 reddit / step count: text vs image (d_z=+0.28 vs -0.25) -> antagonistic
- B1 reddit / finish rate: text vs image (h=-0.22 vs +0.22) -> antagonistic
- B1 reddit / step count: text vs image (d_z=+0.20 vs -0.23) -> antagonistic
- B1 classifieds / click fraction: text vs image (d_z=+0.14 vs -0.17) -> antagonistic
- B1 classifieds / click fraction: prompt vs image (d_z=+0.53 vs -0.17) -> antagonistic
- B1 classifieds / step count: text vs prompt (d_z=+0.10 vs -0.11) -> antagonistic
- B1 classifieds / step count: text vs image (d_z=+0.10 vs -0.25) -> antagonistic
- B2 reddit / click fraction: text vs prompt (d_z=-0.19 vs +0.14) -> antagonistic
- B2 reddit / action repeat: text vs image (d_z=-0.16 vs +0.12) -> antagonistic
- B2 classifieds / search loop: prompt vs image (h=+0.11 vs -0.14) -> antagonistic
- B2 classifieds / type fraction: text vs prompt (d_z=+0.12 vs -0.41) -> antagonistic
- B2 classifieds / type fraction: prompt vs image (d_z=-0.41 vs +0.30) -> antagonistic
- B2 classifieds / scroll fraction: text vs prompt (d_z=+0.15 vs -0.11) -> antagonistic
- B2 classifieds / self-correction: text vs prompt (d_z=+0.10 vs -0.11) -> antagonistic
- B2 classifieds / self-correction: text vs image (d_z=+0.10 vs -0.11) -> antagonistic
- B2 classifieds / click fraction: text vs prompt (d_z=-0.48 vs +0.49) -> antagonistic
- B2 classifieds / finish rate: text vs image (h=-0.22 vs +0.33) -> antagonistic
- B2 classifieds / step count: prompt vs image (d_z=+0.16 vs -0.36) -> antagonistic

## Consistency checks

text + prompt + image recovers the direct SoM − DOM endpoint in **45 of 48** (site × metric) combinations, at tolerance 0.1 pp for the binary metric and 0.005 raw units for fractions and counts.

Read this the same way as Tier 2b: on mean differences the three axes summing to the endpoint is an **algebraic identity**, so passing is arithmetic rather than evidence that the cascade decomposes cleanly. A failure means the legs were averaged over different task sets. This sentence used to be a fixed claim that every combination passed, which was true of the empty table it was printed under. (§F audit, 2026-08-02)

## Tier 2c — Mechanism (Micro): per-step decision quality

Tracked separately in `axis1_microbehavior.{json,md}`. Macro action-frequency metrics (this file) average per-step decisions; micro metrics directly compare per-step element selection / page coverage / search keyword reuse via mode-invariant anchors (URL, action.text).

## Paper Section 5 implication

**Tier 2a Macro — dominant cascade axis per metric**: text: search_loop@B0/reddit, scroll_frac@B0/reddit, selfcorr_count@B0/reddit, finish_rate@B0/reddit, n_steps@B0/reddit, action_repeat_frac@B0/reddit, type_frac@B0/classifieds, finish_rate@B1/reddit, action_repeat_frac@B1/reddit, click_frac@B2/reddit, action_repeat_frac@B2/reddit, scroll_frac@B2/classifieds; prompt: selfcorr_count@B0/classifieds, type_frac@B1/reddit, click_frac@B1/reddit, type_frac@B1/classifieds, click_frac@B1/classifieds, type_frac@B2/classifieds, selfcorr_count@B2/classifieds, click_frac@B2/classifieds, action_repeat_frac@B2/classifieds; image: search_loop@B0/classifieds, scroll_frac@B0/classifieds, finish_rate@B0/classifieds, n_steps@B0/classifieds, action_repeat_frac@B0/classifieds, search_loop@B1/reddit, scroll_frac@B1/reddit, n_steps@B1/reddit, search_loop@B1/classifieds, selfcorr_count@B1/classifieds, finish_rate@B1/classifieds, n_steps@B1/classifieds, action_repeat_frac@B1/classifieds, search_loop@B2/reddit, finish_rate@B2/reddit, n_steps@B2/reddit, search_loop@B2/classifieds, finish_rate@B2/classifieds, n_steps@B2/classifieds.

**Antagonistic pairs** (axes pulling opposite directions, hidden by DOM↔SoM endpoint comparison): text_vs_image@selfcorr_count@B0/reddit; text_vs_prompt@finish_rate@B0/reddit; text_vs_image@finish_rate@B0/reddit; text_vs_image@n_steps@B0/reddit; text_vs_image@finish_rate@B1/reddit; text_vs_image@n_steps@B1/reddit; text_vs_image@click_frac@B1/classifieds; prompt_vs_image@click_frac@B1/classifieds; text_vs_prompt@n_steps@B1/classifieds; text_vs_image@n_steps@B1/classifieds; text_vs_prompt@click_frac@B2/reddit; text_vs_image@action_repeat_frac@B2/reddit; prompt_vs_image@search_loop@B2/classifieds; text_vs_prompt@type_frac@B2/classifieds; prompt_vs_image@type_frac@B2/classifieds; text_vs_prompt@scroll_frac@B2/classifieds; text_vs_prompt@selfcorr_count@B2/classifieds; text_vs_image@selfcorr_count@B2/classifieds; text_vs_prompt@click_frac@B2/classifieds; text_vs_image@finish_rate@B2/classifieds; prompt_vs_image@n_steps@B2/classifieds.

**4-level cascade design value**: decomposes DOM → SoM into three controlled transitions (AXTree vs [SOM_MARKS] structure, DOM vs SoM prompting, marginal image). This run finds **21 antagonistic mechanism pair(s)** that endpoint-only comparison would mask.