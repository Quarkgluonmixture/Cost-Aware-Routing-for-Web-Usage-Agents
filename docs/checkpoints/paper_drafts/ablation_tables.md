---
type: paper-draft
status: generated
purpose: every evidence product as an ablation table, for the Overleaf draft
producer: scripts/analysis/export_ablation_tables.py
---

## Ablation tables

<!-- GENERATED 2026-08-05T00:06:06+00:00 — do not hand-edit between the table markers; rerun the producer instead. -->

Every number below is read from a product JSON at render time. None is typed by hand: six hand-copied numbers were found decoupled from their products on 2026-08-03, one of them wrong on the fact rather than only on the denominator.

Cells are (site × backbone). `cls`/`red` are VisualWebArena classifieds/reddit; `WA` is WebArena reddit. `B0` = Qwen3-VL-235B-A22B, `B1` = Qwen3-VL-4B, `B2` = Gemma-3-4B. **`WA·B2` does not exist** — B2 never ran WebArena.

<!-- BEGIN table:sr -->

| cell | DOM | SoM | Vision | DOM+stext | DOM+sprompt | SoM-image | best |
|---|---|---|---|---|---|---|---|
| cls·B0 | 17.41 | **27.23** | 25.00 | 15.62 | 19.64 | 15.62 | SoM |
| cls·B1 | 6.25 | **14.29** | 12.50 | 7.59 | 6.70 | 6.70 | SoM |
| cls·B2 | 1.34 | **2.23** | 2.23 | 0.45 | 1.79 | 0.89 | SoM |
| red·B0 | 14.29 | **14.78** | 7.39 | 13.30 | 12.32 | 10.84 | SoM |
| red·B1 | 5.91 | **7.39** | 2.46 | 5.91 | 5.42 | 5.91 | SoM |
| red·B2 | **3.94** | 0.99 | 1.97 | 1.97 | 0.00 | 0.49 | DOM |
| WA·B0 | 26.92 | 22.12 | 19.23 | **35.58** | 25.96 | 25.00 | DOM+stext |
| WA·B1 | **16.35** | 13.46 | 9.62 | 16.35 | 16.35 | 11.54 | DOM |

*Table 1: Success rate per mode. Success rate (%) per observation mode. Denominator is the canonical scored set (cls 224 / red 203) and, for WebArena, the six-mode task intersection (104). Bold = best mode in the cell. Source: `representation_class_comparison.json`.*

<!-- END table:sr -->

<!-- BEGIN table:class -->

| cell | no-image (4 arms) | vision-only | hybrid | sole best |
|---|---|---|---|---|
| cls·B0 | 19.64 (DOM+sprompt) | 25.00 | 27.23 | hybrid |
| cls·B1 | 7.59 (DOM+stext) | 12.50 | 14.29 | hybrid |
| cls·B2 | 1.79 (DOM+sprompt) | 2.23 | 2.23 | tie: hybrid+vision-only |
| red·B0 | 14.29 (DOM) | 7.39 | 14.78 | hybrid |
| red·B1 | 5.91 (DOM) | 2.46 | 7.39 | hybrid |
| red·B2 | 3.94 (DOM) | 1.97 | 0.99 | no-image |
| WA·B0 | 35.58 (DOM+stext) | 19.23 | 22.12 | no-image |
| WA·B1 | 16.35 (DOM) | 9.62 | 13.46 | no-image |

*Table 2: Best arm per deployment class. Best arm within each deployment class (%). Classes: **no-image** = {DOM, DOM+stext, DOM+sprompt, SoM-image}, **vision-only** = {Vision}, **hybrid** = {SoM}. Grouping the four is licensed by the non-separability result (Table 4): they clear the ≥83% consistency bar on none of 26 metrics. Source: `representation_class_comparison.json`.*

<!-- END table:class -->

<!-- BEGIN table:class-1arm -->

| cell | no-image (DOM) | vision-only (Vision) | hybrid (SoM) | sole best | gap vs hybrid | Table 2 gap |
|---|---|---|---|---|---|---|
| cls·B0 | 17.41 | 25.00 | 27.23 | hybrid | -9.82 | -7.59 |
| cls·B1 | 6.25 | 12.50 | 14.29 | hybrid | -8.04 | -6.70 |
| cls·B2 | 1.34 | 2.23 | 2.23 | tie: hybrid+vision-only | -0.89 | -0.45 |
| red·B0 | 14.29 | 7.39 | 14.78 | hybrid | -0.49 | -0.49 |
| red·B1 | 5.91 | 2.46 | 7.39 | hybrid | -1.48 | -1.48 |
| red·B2 | 3.94 | 1.97 | 0.99 | no-image | +2.96 | +2.96 |
| WA·B0 | 26.92 | 19.23 | 22.12 | no-image | +4.81 | +13.46 |
| WA·B1 | 16.35 | 9.62 | 13.46 | no-image | +2.88 | +2.88 |

*Table 3: Deployment classes at one arm each. The same comparison at **one arm per class** (%). Table 2's no-image column is a maximum over four arms while the other two are single arms, so it is biased up; this panel uses the arm of each class that exists outside this study. **The ordering does not move** — hybrid 4, no-image 3, one tie, and vision-only is never a sole best in either version — which is the robustness statement Table 2 cannot make. **The gaps do move**: on `WA·B0` the no-image lead is +4.81pp here against +13.46pp there, because Table 2's figure is carried by DOM+stext. Quote this table for any class gap; Table 2 only for the ordering. Source: `representation_class_comparison.json`.*

<!-- END table:class-1arm -->

<!-- BEGIN table:class-ablate -->

| cell | all six | −no-image | −vision-only | −hybrid | +1 no-image | +1 vision-only | +1 hybrid |
|---|---|---|---|---|---|---|---|
| cls·B0 | 43.30 | −9.38 | −4.02 | −2.68 | +7.14pp | +6.70pp | — |
| cls·B1 | 24.55 | −5.36 | −4.02 | −4.46 | +3.57pp | +4.91pp | — |
| cls·B2 | 7.14 | −2.68 | −2.23 | −2.23 | +1.79pp | +2.23pp | — |
| red·B0 | 26.11 | −7.88 | −1.97 | −1.97 | +4.93pp | +3.45pp | — |
| red·B1 | 11.82 | −3.45 | −0.99 | −0.49 | +1.97pp | +0.99pp | — |
| red·B2 | 7.39 | −4.43 | −1.48 | −0.49 | +0.99pp | +1.97pp | +0.49pp |
| WA·B0 | 51.92 | −22.12 | −3.85 | −1.92 | +5.77pp | +5.77pp | +4.81pp |
| WA·B1 | 30.77 | −14.42 | −0.96 | −0.96 | +4.81pp | +3.85pp | +3.85pp |

*Table 4: Class ablation, unmatched and arm-matched. Class ablation. Columns 2–4: oracle coverage lost when a whole class is unavailable — **not arm-matched**, the no-image class has four arms and the others one each, so most of the gap is arm count. Columns 5–7: the matched comparison — gain from adding ONE arm of that class to the cell's best single arm (— = that class already supplies the starting arm). The matched panel shows no systematic difference between classes. Source: `representation_class_comparison.json`.*

<!-- END table:class-ablate -->

<!-- BEGIN table:nonsep -->

| mode | metrics reaching the bar |
|---|---|
| Vision | 9 |
| SoM | 5 |
| DOM | 0 |
| DOM+stext | 0 |
| DOM+sprompt | 0 |
| SoM-image | 0 |

*Table 5: Absence of repeated extrema among image-free modes. **Absence of repeated extrema** — read the name literally. A mode 'reaches the bar' on a metric when it is the extreme (highest or lowest) in ≥7 of 8 cells (87.5%). Over 26 metrics the four image-free modes reach it on **none**. **Caution:** **This is not a separability test.** `rank_consistency()` only counts which mode attains each metric's max/min; a mode that is consistently *second* — and strongly distinguishable — never reaches the bar. Establishing non-separability would need pairwise equivalence margins with task-clustered intervals, which this does not do. **Caution:** **The threshold got stricter when cells grew, and an earlier caption said otherwise**: ≥7/8 is 87.5%, not the 83% it claimed, and the six-cell ≥5/6 it says it matches is 83.3% — a +4.2pp shift, chosen after the cell count changed. Carrying the literal numerator (≥5/8 = 62.5%) instead would let DOM+stext clear two metrics and this negative would appear to break, so the choice is load-bearing and is disclosed rather than defended. Source: `per_mode_four_dimension_profile_with_wa.json`.*

<!-- END table:nonsep -->

<!-- BEGIN table:prof-outcome -->

| cell | mode | SR % | solves | unique |
|---|---|---|---|---|
| cls·B0 | DOM | 17.41 | 39 | 4 |
| cls·B0 | SoM | 27.23 | 61 | 6 |
| cls·B0 | Vision | 25.00 | 56 | 9 |
| cls·B0 | DOM+stext | 15.62 | 35 | 2 |
| cls·B0 | DOM+sprompt | 19.64 | 44 | 6 |
| cls·B0 | SoM-image | 15.62 | 35 | 2 |
| cls·B1 | DOM | 6.25 | 14 | 1 |
| cls·B1 | SoM | 14.29 | 32 | 10 |
| cls·B1 | Vision | 12.50 | 28 | 9 |
| cls·B1 | DOM+stext | 7.59 | 17 | 1 |
| cls·B1 | DOM+sprompt | 6.70 | 15 | 2 |
| cls·B1 | SoM-image | 6.70 | 15 | 3 |
| cls·B2 | DOM | 1.34 | 3 | 1 |
| cls·B2 | SoM | 2.23 | 5 | 5 |
| cls·B2 | Vision | 2.23 | 5 | 5 |
| cls·B2 | DOM+stext | 0.45 | 1 | 0 |
| cls·B2 | DOM+sprompt | 1.79 | 4 | 0 |
| cls·B2 | SoM-image | 0.89 | 2 | 1 |
| red·B0 | DOM | 14.29 | 29 | 3 |
| red·B0 | SoM | 14.78 | 30 | 4 |
| red·B0 | Vision | 7.39 | 15 | 4 |
| red·B0 | DOM+stext | 13.30 | 27 | 2 |
| red·B0 | DOM+sprompt | 12.32 | 25 | 2 |
| red·B0 | SoM-image | 10.84 | 22 | 2 |
| red·B1 | DOM | 5.91 | 12 | 1 |
| red·B1 | SoM | 7.39 | 15 | 1 |
| red·B1 | Vision | 2.46 | 5 | 2 |
| red·B1 | DOM+stext | 5.91 | 12 | 1 |
| red·B1 | DOM+sprompt | 5.42 | 11 | 2 |
| red·B1 | SoM-image | 5.91 | 12 | 0 |
| red·B2 | DOM | 3.94 | 8 | 6 |
| red·B2 | SoM | 0.99 | 2 | 1 |
| red·B2 | Vision | 1.97 | 4 | 3 |
| red·B2 | DOM+stext | 1.97 | 4 | 1 |
| red·B2 | DOM+sprompt | 0.00 | 0 | 0 |
| red·B2 | SoM-image | 0.49 | 1 | 1 |
| WA·B0 | DOM | 26.92 | 28 | 2 |
| WA·B0 | SoM | 22.12 | 23 | 2 |
| WA·B0 | Vision | 19.23 | 20 | 4 |
| WA·B0 | DOM+stext | 35.58 | 37 | 7 |
| WA·B0 | DOM+sprompt | 25.96 | 27 | 2 |
| WA·B0 | SoM-image | 25.00 | 26 | 1 |
| WA·B1 | DOM | 16.35 | 17 | 3 |
| WA·B1 | SoM | 13.46 | 14 | 1 |
| WA·B1 | Vision | 9.62 | 10 | 1 |
| WA·B1 | DOM+stext | 16.35 | 17 | 3 |
| WA·B1 | DOM+sprompt | 16.35 | 17 | 5 |
| WA·B1 | SoM-image | 11.54 | 12 | 1 |

*Table 6: Full matrix — Outcome dimension. Outcome dimension, every cell × every mode. `unique` counts tasks **no other mode in that cell solved**. Denominators: cls 224 / red 203 / WA 104. Source: `per_mode_four_dimension_profile_with_wa.json`.*

<!-- END table:prof-outcome -->

<!-- BEGIN table:prof-macro -->

| cell | mode | steps/ep | cap-hit | click | type | scroll | search-loop | URL-revisit |
|---|---|---|---|---|---|---|---|---|
| cls·B0 | DOM | 15.61 | 0.268 | 0.322 | 0.224 | 0.183 | 0.812 | 0.606 |
| cls·B0 | SoM | 13.67 | 0.259 | 0.319 | 0.205 | 0.155 | 0.692 | 0.558 |
| cls·B0 | Vision | 15.88 | 0.295 | 0.353 | 0.146 | 0.260 | 0.728 | 0.639 |
| cls·B0 | DOM+stext | 15.83 | 0.295 | 0.322 | 0.194 | 0.196 | 0.808 | 0.603 |
| cls·B0 | DOM+sprompt | 14.96 | 0.290 | 0.328 | 0.208 | 0.174 | 0.759 | 0.583 |
| cls·B0 | SoM-image | 16.23 | 0.321 | 0.312 | 0.201 | 0.208 | 0.790 | 0.600 |
| cls·B1 | DOM | 21.38 | 0.585 | 0.246 | 0.345 | 0.171 | 0.777 | 0.698 |
| cls·B1 | SoM | 18.01 | 0.491 | 0.368 | 0.211 | 0.137 | 0.643 | 0.655 |
| cls·B1 | Vision | 20.17 | 0.598 | 0.330 | 0.071 | 0.360 | 0.603 | 0.745 |
| cls·B1 | DOM+stext | 22.46 | 0.625 | 0.281 | 0.367 | 0.148 | 0.804 | 0.716 |
| cls·B1 | DOM+sprompt | 21.40 | 0.594 | 0.375 | 0.209 | 0.163 | 0.723 | 0.710 |
| cls·B1 | SoM-image | 21.26 | 0.576 | 0.436 | 0.212 | 0.154 | 0.737 | 0.715 |
| cls·B2 | DOM | 27.38 | 0.817 | 0.520 | 0.142 | 0.031 | 0.647 | 0.823 |
| cls·B2 | SoM | 24.37 | 0.688 | 0.488 | 0.143 | 0.033 | 0.616 | 0.828 |
| cls·B2 | Vision | 28.25 | 0.888 | 0.336 | 0.128 | 0.323 | 0.710 | 0.896 |
| cls·B2 | DOM+stext | 26.85 | 0.804 | 0.384 | 0.168 | 0.043 | 0.629 | 0.832 |
| cls·B2 | DOM+sprompt | 27.84 | 0.830 | 0.514 | 0.078 | 0.048 | 0.629 | 0.857 |
| cls·B2 | SoM-image | 28.38 | 0.853 | 0.523 | 0.082 | 0.033 | 0.683 | 0.858 |
| red·B0 | DOM | 20.18 | 0.498 | 0.455 | 0.148 | 0.166 | 0.473 | 0.721 |
| red·B0 | SoM | 20.08 | 0.512 | 0.474 | 0.123 | 0.099 | 0.363 | 0.731 |
| red·B0 | Vision | 23.55 | 0.672 | 0.339 | 0.079 | 0.343 | 0.239 | 0.817 |
| red·B0 | DOM+stext | 23.22 | 0.647 | 0.442 | 0.153 | 0.117 | 0.388 | 0.775 |
| red·B0 | DOM+sprompt | 19.87 | 0.463 | 0.475 | 0.142 | 0.126 | 0.428 | 0.704 |
| red·B0 | SoM-image | 22.90 | 0.592 | 0.453 | 0.137 | 0.124 | 0.338 | 0.775 |
| red·B1 | DOM | 23.44 | 0.680 | 0.471 | 0.238 | 0.066 | 0.596 | 0.754 |
| red·B1 | SoM | 22.38 | 0.670 | 0.574 | 0.129 | 0.095 | 0.409 | 0.747 |
| red·B1 | Vision | 23.24 | 0.685 | 0.434 | 0.029 | 0.270 | 0.148 | 0.812 |
| red·B1 | DOM+stext | 25.56 | 0.783 | 0.476 | 0.246 | 0.067 | 0.606 | 0.793 |
| red·B1 | DOM+sprompt | 23.72 | 0.695 | 0.545 | 0.149 | 0.052 | 0.567 | 0.757 |
| red·B1 | SoM-image | 25.17 | 0.773 | 0.580 | 0.166 | 0.056 | 0.557 | 0.778 |
| red·B2 | DOM | 28.39 | 0.897 | 0.724 | 0.108 | 0.042 | 0.227 | 0.870 |
| red·B2 | SoM | 26.34 | 0.783 | 0.702 | 0.103 | 0.043 | 0.148 | 0.857 |
| red·B2 | Vision | 26.91 | 0.828 | 0.349 | 0.081 | 0.344 | 0.138 | 0.911 |
| red·B2 | DOM+stext | 27.27 | 0.818 | 0.660 | 0.099 | 0.038 | 0.266 | 0.885 |
| red·B2 | DOM+sprompt | 27.68 | 0.842 | 0.703 | 0.081 | 0.052 | 0.271 | 0.873 |
| red·B2 | SoM-image | 27.87 | 0.882 | 0.708 | 0.094 | 0.048 | 0.217 | 0.882 |
| WA·B0 | DOM | 16.88 | 0.298 | 0.544 | 0.208 | 0.072 | 0.308 | 0.709 |
| WA·B0 | SoM | 17.45 | 0.365 | 0.465 | 0.278 | 0.029 | 0.327 | 0.716 |
| WA·B0 | Vision | 22.38 | 0.567 | 0.409 | 0.200 | 0.241 | 0.240 | 0.783 |
| WA·B0 | DOM+stext | 19.77 | 0.471 | 0.478 | 0.323 | 0.045 | 0.288 | 0.741 |
| WA·B0 | DOM+sprompt | 17.08 | 0.317 | 0.538 | 0.198 | 0.068 | 0.260 | 0.714 |
| WA·B0 | SoM-image | 18.97 | 0.385 | 0.502 | 0.284 | 0.056 | 0.221 | 0.729 |
| WA·B1 | DOM | 22.64 | 0.625 | 0.459 | 0.319 | 0.023 | 0.644 | 0.746 |
| WA·B1 | SoM | 23.91 | 0.702 | 0.499 | 0.218 | 0.035 | 0.452 | 0.793 |
| WA·B1 | Vision | 23.12 | 0.606 | 0.450 | 0.097 | 0.249 | 0.260 | 0.807 |
| WA·B1 | DOM+stext | 23.33 | 0.615 | 0.495 | 0.316 | 0.028 | 0.683 | 0.733 |
| WA·B1 | DOM+sprompt | 24.50 | 0.702 | 0.551 | 0.231 | 0.033 | 0.519 | 0.776 |
| WA·B1 | SoM-image | 23.79 | 0.683 | 0.569 | 0.249 | 0.017 | 0.587 | 0.772 |

*Table 7: Full matrix — Macro dimension. Macro dimension — what the agent did, per step, aggregated per episode. Fractions are over agent actions. `cap-hit` = share of episodes that exhausted the 30-step budget. Source: `per_mode_four_dimension_profile_with_wa.json`.*

<!-- END table:prof-macro -->

<!-- BEGIN table:prof-micro -->

| cell | mode | parse-fail | act-fail | act-fail|click | act-fail|type | no-op | scroll-inert | no-op|success | vis-gap | loc-fallback | act-repeat | finish |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cls·B0 | DOM | 0.0007 | 0.133 | 0.163 | 0.035 | 0.241 | 0.096 | 0.108 | 0.046 | 0.078 | 0.324 | 0.737 |
| cls·B0 | SoM | 0.0021 | 0.082 | 0.102 | 0.007 | 0.216 | 0.018 | 0.134 | 0.051 | 0.027 | 0.300 | 0.741 |
| cls·B0 | Vision | 0.0000 | 0.150 | 0.153 | 0.005 | 0.263 | 0.072 | 0.113 | 0.058 | 0.002 | 0.396 | 0.705 |
| cls·B0 | DOM+stext | 0.0000 | 0.101 | 0.100 | 0.022 | 0.209 | 0.071 | 0.108 | 0.042 | 0.046 | 0.310 | 0.705 |
| cls·B0 | DOM+sprompt | 0.0009 | 0.133 | 0.159 | 0.026 | 0.246 | 0.078 | 0.114 | 0.049 | 0.079 | 0.322 | 0.719 |
| cls·B0 | SoM-image | 0.0005 | 0.112 | 0.130 | 0.016 | 0.221 | 0.078 | 0.109 | 0.042 | 0.037 | 0.332 | 0.688 |
| cls·B1 | DOM | 0.0031 | 0.285 | 0.155 | 0.153 | 0.359 | 0.219 | 0.074 | 0.011 | 0.175 | 0.443 | 0.411 |
| cls·B1 | SoM | 0.0016 | 0.249 | 0.203 | 0.068 | 0.353 | 0.149 | 0.104 | 0.013 | 0.118 | 0.390 | 0.509 |
| cls·B1 | Vision | 0.0000 | 0.454 | 0.345 | 0.031 | 0.548 | 0.278 | 0.094 | 0.018 | 0.011 | 0.530 | 0.402 |
| cls·B1 | DOM+stext | 0.0004 | 0.259 | 0.182 | 0.140 | 0.317 | 0.159 | 0.058 | 0.010 | 0.202 | 0.461 | 0.384 |
| cls·B1 | DOM+sprompt | 0.0044 | 0.309 | 0.272 | 0.081 | 0.380 | 0.178 | 0.070 | 0.012 | 0.174 | 0.409 | 0.406 |
| cls·B1 | SoM-image | 0.0010 | 0.322 | 0.274 | 0.136 | 0.381 | 0.169 | 0.059 | 0.015 | 0.203 | 0.461 | 0.424 |
| cls·B2 | DOM | 0.0259 | 0.509 | 0.555 | 0.207 | 0.526 | 0.099 | 0.018 | 0.030 | 0.339 | 0.630 | 0.125 |
| cls·B2 | SoM | 0.0876 | 0.559 | 0.475 | 0.110 | 0.584 | 0.067 | 0.025 | 0.026 | 0.327 | 0.627 | 0.152 |
| cls·B2 | Vision | 0.0067 | 0.670 | 0.773 | 0.355 | 0.685 | 0.443 | 0.015 | 0.017 | 0.124 | 0.518 | 0.103 |
| cls·B2 | DOM+stext | 0.0761 | 0.441 | 0.438 | 0.227 | 0.453 | 0.130 | 0.012 | 0.029 | 0.292 | 0.602 | 0.062 |
| cls·B2 | DOM+sprompt | 0.0518 | 0.495 | 0.569 | 0.167 | 0.504 | 0.137 | 0.009 | 0.028 | 0.312 | 0.633 | 0.071 |
| cls·B2 | SoM-image | 0.0522 | 0.425 | 0.507 | 0.109 | 0.433 | 0.099 | 0.008 | 0.027 | 0.273 | 0.649 | 0.054 |
| red·B0 | DOM | 0.0002 | 0.222 | 0.194 | 0.051 | 0.294 | 0.161 | 0.072 | 0.075 | 0.119 | 0.428 | 0.507 |
| red·B0 | SoM | 0.0008 | 0.252 | 0.127 | 0.022 | 0.328 | 0.085 | 0.077 | 0.092 | 0.068 | 0.491 | 0.488 |
| red·B0 | Vision | 0.0000 | 0.382 | 0.193 | 0.018 | 0.430 | 0.268 | 0.049 | 0.064 | 0.006 | 0.556 | 0.328 |
| red·B0 | DOM+stext | 0.0005 | 0.292 | 0.126 | 0.062 | 0.339 | 0.146 | 0.047 | 0.087 | 0.064 | 0.518 | 0.358 |
| red·B0 | DOM+sprompt | 0.0026 | 0.194 | 0.146 | 0.046 | 0.270 | 0.149 | 0.076 | 0.093 | 0.083 | 0.421 | 0.537 |
| red·B0 | SoM-image | 0.0027 | 0.285 | 0.118 | 0.034 | 0.335 | 0.160 | 0.050 | 0.097 | 0.070 | 0.502 | 0.408 |
| red·B1 | DOM | 0.0021 | 0.227 | 0.210 | 0.049 | 0.274 | 0.090 | 0.047 | 0.050 | 0.143 | 0.465 | 0.315 |
| red·B1 | SoM | 0.0063 | 0.297 | 0.228 | 0.062 | 0.357 | 0.111 | 0.061 | 0.061 | 0.176 | 0.552 | 0.320 |
| red·B1 | Vision | 0.0141 | 0.532 | 0.334 | 0.010 | 0.582 | 0.302 | 0.050 | 0.022 | 0.004 | 0.632 | 0.291 |
| red·B1 | DOM+stext | 0.0018 | 0.283 | 0.195 | 0.107 | 0.319 | 0.156 | 0.036 | 0.042 | 0.164 | 0.511 | 0.217 |
| red·B1 | DOM+sprompt | 0.0010 | 0.295 | 0.278 | 0.056 | 0.340 | 0.114 | 0.045 | 0.055 | 0.201 | 0.494 | 0.305 |
| red·B1 | SoM-image | 0.0019 | 0.332 | 0.297 | 0.077 | 0.368 | 0.146 | 0.036 | 0.059 | 0.208 | 0.544 | 0.222 |
| red·B2 | DOM | 0.0224 | 0.492 | 0.502 | 0.111 | 0.501 | 0.156 | 0.009 | 0.044 | 0.443 | 0.770 | 0.074 |
| red·B2 | SoM | 0.0431 | 0.384 | 0.344 | 0.090 | 0.403 | 0.053 | 0.019 | 0.052 | 0.305 | 0.760 | 0.163 |
| red·B2 | Vision | 0.0163 | 0.640 | 0.613 | 0.276 | 0.669 | 0.547 | 0.029 | 0.005 | 0.078 | 0.597 | 0.153 |
| red·B2 | DOM+stext | 0.0786 | 0.285 | 0.263 | 0.081 | 0.292 | 0.094 | 0.007 | 0.041 | 0.202 | 0.728 | 0.059 |
| red·B2 | DOM+sprompt | 0.0660 | 0.604 | 0.621 | 0.126 | 0.609 | 0.115 | 0.005 | 0.042 | 0.503 | 0.724 | 0.044 |
| red·B2 | SoM-image | 0.0501 | 0.316 | 0.297 | 0.123 | 0.325 | 0.137 | 0.009 | 0.034 | 0.240 | 0.725 | 0.049 |
| WA·B0 | DOM | 0.0008 | 0.277 | 0.286 | 0.083 | 0.363 | 0.046 | 0.085 | 0.226 | 0.165 | 0.455 | 0.702 |
| WA·B0 | SoM | 0.0063 | 0.249 | 0.139 | 0.079 | 0.345 | 0.026 | 0.096 | 0.219 | 0.045 | 0.433 | 0.625 |
| WA·B0 | Vision | 0.0000 | 0.245 | 0.178 | 0.042 | 0.295 | 0.165 | 0.050 | 0.152 | 0.007 | 0.478 | 0.433 |
| WA·B0 | DOM+stext | 0.0010 | 0.347 | 0.292 | 0.200 | 0.421 | 0.057 | 0.074 | 0.247 | 0.099 | 0.497 | 0.538 |
| WA·B0 | DOM+sprompt | 0.0018 | 0.295 | 0.278 | 0.070 | 0.379 | 0.045 | 0.085 | 0.223 | 0.157 | 0.469 | 0.692 |
| WA·B0 | SoM-image | 0.0033 | 0.302 | 0.268 | 0.170 | 0.377 | 0.040 | 0.075 | 0.268 | 0.092 | 0.500 | 0.625 |
| WA·B1 | DOM | 0.0227 | 0.244 | 0.201 | 0.038 | 0.289 | 0.026 | 0.045 | 0.135 | 0.139 | 0.512 | 0.327 |
| WA·B1 | SoM | 0.0052 | 0.363 | 0.214 | 0.084 | 0.415 | 0.024 | 0.052 | 0.166 | 0.149 | 0.573 | 0.279 |
| WA·B1 | Vision | 0.0072 | 0.416 | 0.326 | 0.023 | 0.456 | 0.215 | 0.041 | 0.102 | 0.002 | 0.571 | 0.365 |
| WA·B1 | DOM+stext | 0.0102 | 0.206 | 0.197 | 0.076 | 0.247 | 0.045 | 0.041 | 0.113 | 0.105 | 0.503 | 0.385 |
| WA·B1 | DOM+sprompt | 0.0068 | 0.343 | 0.302 | 0.070 | 0.379 | 0.066 | 0.036 | 0.098 | 0.197 | 0.522 | 0.298 |
| WA·B1 | SoM-image | 0.0144 | 0.343 | 0.315 | 0.076 | 0.382 | 0.036 | 0.039 | 0.099 | 0.180 | 0.562 | 0.288 |

*Table 8: Full matrix — Micro dimension. Micro dimension — per-step execution quality. `act-fail|click` and `act-fail|type` are conditional on the action type, so they are not comparable to the unconditional `act-fail` column. **Caution:** **They are also not pure conditionals.** An episode containing no click at all has no click-failure rate to report, and the producer stores `0.0` there — an undefined rate encoded as a perfect one — which is then averaged over every task. The zero-denominator share is large: **25–35% of episodes never type**, and a few percent never click. The product now carries `*_fail_rate_complete_case` (averaged only over episodes where the action occurred) and `*_fail_rate_denom_zero_frac` beside these columns; the complete-case values run **higher**, and any statement about which mode fails most on clicks or types should be read from those, not from this column. **Caution:** `loc-fallback` is near-zero for Vision **by construction** (no element ids to fall back from), not as a finding. Source: `per_mode_four_dimension_profile_with_wa.json`.*

<!-- END table:prof-micro -->

<!-- BEGIN table:prof-eff -->

| cell | mode | cost/ep | cost rel DOM | latency s | latency canon s | tokens/ep |
|---|---|---|---|---|---|---|
| cls·B0 | DOM | 0.06962 | 1.000 | 115.0 | 114.1 | 63045 |
| cls·B0 | SoM | 0.07236 | 1.039 | 106.7 | 106.0 | 67000 |
| cls·B0 | Vision | 0.06481 | 0.931 | 126.3 | 125.2 | 58302 |
| cls·B0 | DOM+stext | 0.06919 | 0.994 | 123.8 | 120.4 | 62154 |
| cls·B0 | DOM+sprompt | 0.06853 | 0.984 | 109.4 | 107.8 | 62555 |
| cls·B0 | SoM-image | 0.07206 | 1.035 | 121.1 | 117.9 | 65334 |
| cls·B1 | DOM | 0.05951 | 1.000 | 308.9 | 308.9 | 61904 |
| cls·B1 | SoM | 0.06028 | 1.013 | 262.0 | 262.0 | 62953 |
| cls·B1 | Vision | 0.04316 | 0.725 | 269.8 | 269.8 | 44524 |
| cls·B1 | DOM+stext | 0.05879 | 0.988 | 313.5 | 313.5 | 61074 |
| cls·B1 | DOM+sprompt | 0.06304 | 1.059 | 301.4 | 301.4 | 65628 |
| cls·B1 | SoM-image | 0.05970 | 1.003 | 311.7 | 311.7 | 61985 |
| cls·B2 | DOM | 0.07676 | 1.000 | 402.3 | 402.3 | 79653 |
| cls·B2 | SoM | 0.09075 | 1.182 | 374.3 | 374.3 | 95081 |
| cls·B2 | Vision | 0.07065 | 0.920 | 417.8 | 417.8 | 73126 |
| cls·B2 | DOM+stext | 0.07320 | 0.954 | 399.4 | 399.4 | 75946 |
| cls·B2 | DOM+sprompt | 0.08453 | 1.101 | 396.4 | 396.4 | 87948 |
| cls·B2 | SoM-image | 0.08456 | 1.102 | 411.0 | 411.0 | 87931 |
| red·B0 | DOM | 0.10147 | 1.000 | 572.0 | 552.5 | 93303 |
| red·B0 | SoM | 0.11045 | 1.089 | 461.4 | 451.6 | 103031 |
| red·B0 | Vision | 0.09807 | 0.966 | 449.8 | 418.5 | 88872 |
| red·B0 | DOM+stext | 0.10577 | 1.042 | 631.4 | 562.1 | 96214 |
| red·B0 | DOM+sprompt | 0.10163 | 1.002 | 498.4 | 447.7 | 93817 |
| red·B0 | SoM-image | 0.10814 | 1.066 | 562.0 | 532.0 | 99005 |
| red·B1 | DOM | 0.07330 | 1.000 | 602.6 | 602.6 | 76462 |
| red·B1 | SoM | 0.08000 | 1.091 | 609.1 | 609.1 | 83517 |
| red·B1 | Vision | 0.05240 | 0.715 | 456.6 | 456.6 | 53994 |
| red·B1 | DOM+stext | 0.06948 | 0.948 | 598.6 | 598.6 | 72222 |
| red·B1 | DOM+sprompt | 0.07656 | 1.044 | 614.0 | 614.0 | 79896 |
| red·B1 | SoM-image | 0.07480 | 1.020 | 616.6 | 616.6 | 77722 |
| red·B2 | DOM | 0.09479 | 1.000 | 669.9 | 669.9 | 99015 |
| red·B2 | SoM | 0.11160 | 1.177 | 623.0 | 623.0 | 117379 |
| red·B2 | Vision | 0.06833 | 0.721 | 550.0 | 550.0 | 70826 |
| red·B2 | DOM+stext | 0.08852 | 0.934 | 677.6 | 677.6 | 92346 |
| red·B2 | DOM+sprompt | 0.09940 | 1.049 | 599.3 | 599.3 | 104021 |
| red·B2 | SoM-image | 0.09451 | 0.997 | 640.0 | 640.0 | 98699 |
| WA·B0 | DOM | 0.07531 | 1.000 | 284.1 | 282.7 | 68616 |
| WA·B0 | SoM | 0.09110 | 1.210 | 272.1 | 271.7 | 84854 |
| WA·B0 | Vision | 0.08640 | 1.147 | 337.9 | 337.5 | 77786 |
| WA·B0 | DOM+stext | 0.08478 | 1.126 | 330.3 | 328.4 | 76591 |
| WA·B0 | DOM+sprompt | 0.07747 | 1.029 | 262.4 | 260.9 | 71176 |
| WA·B0 | SoM-image | 0.08498 | 1.128 | 324.0 | 320.9 | 77689 |
| WA·B1 | DOM | 0.06579 | 1.000 | 485.2 | 485.2 | 68491 |
| WA·B1 | SoM | 0.07944 | 1.208 | 494.8 | 494.8 | 82940 |
| WA·B1 | Vision | 0.04468 | 0.679 | 490.7 | 490.7 | 45878 |
| WA·B1 | DOM+stext | 0.06151 | 0.935 | 509.9 | 509.9 | 63727 |
| WA·B1 | DOM+sprompt | 0.07386 | 1.123 | 506.3 | 506.3 | 76839 |
| WA·B1 | SoM-image | 0.06659 | 1.012 | 510.6 | 510.6 | 68819 |

*Table 9: Full matrix — Efficiency dimension. Efficiency dimension. **Cost is comparable within a cell only** — B0 bills a proxy API, B1/B2 are electricity-derived from a per-token constant calibrated for a different accelerator, so absolute dollars for B1/B2 are uncalibrated and only `cost rel DOM` is safe across backbones. `latency canon` removes retry, busy-wait and recovered-screenshot time; it differs from raw only on the API-served arm. Source: `per_mode_four_dimension_profile_with_wa.json`.*

<!-- END table:prof-eff -->

<!-- BEGIN table:pareto -->

| cell | cost span | latency span | cheapest | fastest | same? | ρ(cost,lat) | frontier (success × cost) |
|---|---|---|---|---|---|---|---|
| cls·B0 | 1.116× | 1.183× | Vision | SoM | **no** | -0.60 | SoM, Vision |
| red·B0 | 1.126× | 1.404× | Vision | Vision | yes | +0.14 | DOM, SoM, Vision |
| cls·B1 | 1.461× | 1.197× | Vision | SoM | **no** | -0.26 | SoM, Vision |
| red·B1 | 1.527× | 1.350× | Vision | Vision | yes | +0.77 | SoM, Vision, DOM+stext |
| cls·B2 | 1.285× | 1.116× | Vision | SoM | **no** | -0.60 | Vision |
| red·B2 | 1.633× | 1.232× | Vision | Vision | yes | -0.03 | DOM, Vision |
| WA·B1 | 1.778× | 1.052× | Vision | DOM | **no** | +0.20 | Vision, DOM+stext |
| WA·B0 | 1.210× | 1.288× | DOM | DOM+sprompt | **no** | +0.26 | DOM, DOM+stext |

*Table 10: Multi-metric Pareto. Multi-metric Pareto. `span` = dearest/cheapest (or slowest/fastest) ratio within the cell. **Caution:** Per-cell exact permutation p-values on ρ are **not significant** — six modes give a Spearman test almost no power — so the cross-cell pattern carries this, not any single ρ. Source: `multimetric_pareto_with_wa.json`.*

<!-- END table:pareto -->

<!-- BEGIN table:latency-split -->

| cell | mean step (ms) | model call (ms) | model share | fastest by total | fastest by model only | same? |
|---|---|---|---|---|---|---|
| B0·classifieds | 7,622 | 2,140 | **28.1%** | DOM+sprompt | DOM+sprompt | yes |
| B0·reddit | 24,601 | 5,603 | **22.8%** | Vision | SoM-image | **no** |
| B0·wa_reddit | 16,112 | 3,551 | **22.0%** | Vision | DOM+sprompt | **no** |
| B1·classifieds | 14,180 | 8,213 | **57.9%** | Vision | Vision | yes |
| B1·reddit | 24,376 | 7,916 | **32.5%** | Vision | DOM+stext | **no** |
| B1·wa_reddit | 21,221 | 7,747 | **36.5%** | DOM+sprompt | Vision | **no** |
| B2·classifieds | 14,739 | 9,908 | **67.2%** | DOM+sprompt | DOM+sprompt | yes |
| B2·reddit | 22,863 | 9,131 | **39.9%** | Vision | Vision | yes |

*Table 11: What a latency number contains. What a latency number contains. Every latency figure elsewhere in this paper is the whole step; `backend_infer` isolates the model call and was read by no analysis script until 2026-08-03. **The model is 22–67% of the measured time**; the rest is the browser and the container, which `offsite_navigation_audit` measures at 1.69× between the two sites. Removing it **changes which mode is fastest in 4 of 8 cells**, and not at random: 4 of 5 reddit-family cells flip against 0 of 3 classifieds cells — the flips land where the container is slowest. A sentence naming the fastest mode is therefore partly a sentence about this deployment. What survives estimand choice is only that the two orderings disagree. Source: `latency_decomposition.json`.*

<!-- END table:latency-split -->

<!-- BEGIN table:estimands -->

| quantity | what the reported number is | what changes under the alternative |
|---|---|---|
| latency | whole step | model call is only 22–67% of it; removing the container **changes the fastest mode in 4 of 8 cells** |
| local cost | price per token | the constant assumes 60 tok/s against 248–551 measured; pricing by GPU-time **changes the cheapest mode in 2 of 4 local cells** |
| energy / carbon | kWh from a CPU estimate | r(energy, latency) = 0.966–1.000 at 66 W — it **is** elapsed time; and it does not exist for B0 at all |

*Table 12: Three efficiency quantities, three estimand choices. Three efficiency quantities, three estimand choices, none of them previously stated. Each row's right-hand column is what a defensible alternative definition does to the per-mode ordering. The pattern is the finding: **efficiency claims in this setting are estimand-dependent, and the estimand is usually left implicit.** The local-cost constant was additionally derived for a DGX Spark while every run was served on an A100 — the same config file migrated its energy profile and not its cost block. Sources: `latency_decomposition.json`, `local_cost_estimand_audit.json`, `energy_carbon_audit.json`.*

<!-- END table:estimands -->

<!-- BEGIN table:dispatch -->

| delivery path | actions | action success |
|---|---|---|
| id locator | 8,857 | **88.9%** |
| other | 1,530 | **65.9%** |
| coord | 2,138 | **38.6%** |
| id framework | 4,564 | **16.1%** |

*Table 13: What actually delivered the click. How each action reached the browser. **`Vision` is on the coordinate path by construction** — it emits no element ids — so its action success is capped by this harness's coordinate implementation (39%) rather than by the 89% the element-id path achieves. That is not a confound to remove (it is what screenshot-only *is*), but the Vision arm measures our grounding code as much as the representation. Separately the element-id fallback share rises with backbone weakness — B0 12% · B1 35% · B2 37% on the text arms: *how often* a run falls back is a model property, the fallback's own 16% success is ours. No success rate elsewhere is adjusted by this; it bounds external validity. Source: `dispatch_path_audit.json`.*

<!-- END table:dispatch -->

<!-- BEGIN table:metric-noise -->

| dimension | metric | cross-mode spread | rerun band | ratio | > a rerun? |
|---|---|---|---|---|---|
| Outcome | `sr_pct` | 11.607 | 2.232 | 5.20x | **yes** |
| Outcome | `n_success` | 26.000 | 5.000 | 5.20x | **yes** |
| Outcome | `n_unique_solves` | — | — | — | *cross-mode by construction* |
| Macro | `n_steps` | 2.567 | 0.469 | 5.48x | **yes** |
| Macro | `cap_hit_rate` | 0.062 | 0.045 | 1.40x | **yes** |
| Macro | `click_frac` | 0.041 | 0.014 | 2.87x | **yes** |
| Macro | `type_frac` | 0.078 | 0.019 | 4.21x | **yes** |
| Macro | `scroll_frac` | 0.106 | 0.006 | 16.85x | **yes** |
| Macro | `search_loop_rate` | 0.121 | 0.018 | 6.75x | **yes** |
| Macro | `url_revisit_rate` | 0.081 | 0.022 | 3.66x | **yes** |
| Micro | `parse_fail_rate` | 0.002 | 0.002 | 1.00x | no |
| Micro | `action_fail_rate` | 0.068 | 0.014 | 4.87x | **yes** |
| Micro | `click_fail_rate` | 0.063 | 0.025 | 2.48x | **yes** |
| Micro | `type_fail_rate` | 0.030 | 0.006 | 5.28x | **yes** |
| Micro | `no_change_rate` | 0.053 | 0.009 | 5.82x | **yes** |
| Micro | `scroll_inert_rate` | 0.078 | 0.017 | 4.68x | **yes** |
| Micro | `noop_inert_rate` | 0.026 | 0.007 | 3.92x | **yes** |
| Micro | `visibility_gap_rate` | 0.016 | 0.011 | 1.56x | **yes** |
| Micro | `locator_fallback_rate` | 0.076 | 0.006 | 11.81x | **yes** |
| Micro | `action_repeat_frac` | 0.096 | 0.004 | 21.88x | **yes** |
| Micro | `finish_rate` | 0.054 | 0.036 | 1.50x | **yes** |
| Efficiency | `mean_cost_usd` | 0.008 | 0.002 | 3.24x | **yes** |
| Efficiency | `cost_rel_dom` | 0.108 | 0.026 | 4.22x | **yes** |
| Efficiency | `mean_latency_s` | 19.562 | 22.488 | 0.87x | no |
| Efficiency | `mean_latency_canonical_s` | 19.204 | 22.846 | 0.84x | no |
| Efficiency | `mean_tokens` | 8697.978 | 2186.054 | 3.98x | **yes** |

*Table 14: Behavioural metrics against run-to-run noise. Behavioural metrics against run-to-run movement, `B0 x classifieds`, three replicated arms (dom, vision, som). `rerun band` is the largest |metric(run A) - metric(run B)| over those arms; `cross-mode spread` is max-min over the six modes. **22 of 25 metrics exceed the band**, several by 5-22x. The exceptions are `mean_latency_canonical_s` (0.84x), `mean_latency_s` (0.87x), `parse_fail_rate` (1.00x) — i.e. **both latency metrics**, which `latency_decomposition` reaches independently by decomposing the step into model and container. Every other efficiency and behavioural claim in this paper is judged against a rerun band; these 26 metrics were not, until this table. One cell, one rerun per arm: a point estimate, not a threshold. Source: `replicate_metric_noise.json`.*

<!-- END table:metric-noise -->

<!-- BEGIN table:per-success -->

| cell | content? | cheapest/attempt | cheapest/success | fastest/attempt | fastest/success | max solves |
|---|---|---|---|---|---|---|
| cls·B0 | yes | Vision | Vision | SoM | SoM | 61 |
| red·B0 | yes | Vision | DOM | Vision | SoM | 30 |
| cls·B1 | yes | Vision | Vision | SoM | SoM | 32 |
| red·B1 | yes | Vision | SoM | Vision | SoM | 15 |
| cls·B2 | **no** | Vision | Vision | SoM | SoM | 5 |
| red·B2 | **no** | Vision | DOM | Vision | DOM | 8 |
| WA·B1 | yes | Vision | DOM+stext | DOM | DOM | 17 |
| WA·B0 | yes | DOM | DOM+stext | DOM+sprompt | DOM+stext | 37 |

*Table 15: Per-attempt versus per-success. Per-attempt versus per-success denominators. `content? = no` marks cells whose best mode has fewer than 10 successes — their ratios are directions at best and both B2 cells are in that state. Among the 6 cells with content the cheapest mode changes under the success denominator in 4 and the fastest in 3. **Caution:** **Every pairwise CI overlaps**, so this supports the methodological point that the denominator must be declared, not any ranking. Source: `outcome_efficiency.json`.*

<!-- END table:per-success -->

<!-- BEGIN table:fusion -->

| cell | n | SoM − Vision | 95% CI | SoM − DOM | 95% CI |
|---|---|---|---|---|---|
| cls·B0 | 224 | +2.23 | [-2.68, +7.59] | +9.82 | [+3.57, +16.07] |
| cls·B1 | 224 | +1.79 | [-2.68, +6.25] | +8.04 | [+3.57, +12.95] |
| cls·B2 | 224 | +0.00 | [-2.68, +2.68] | +0.89 | [-1.34, +3.57] |
| red·B0 | 203 | +7.39 | [+2.46, +12.32] | +0.49 | [-3.94, +4.93] |
| red·B1 | 203 | +4.93 | [+1.48, +8.87] | +1.48 | [-1.48, +4.43] |
| red·B2 | 203 | -0.99 | [-3.45, +1.48] | -2.96 | [-5.91, -0.49] |
| WA·B1 | 104 | +3.85 | [-1.92, +9.62] | -2.88 | [-9.62, +2.88] |
| WA·B0 | 104 | +2.88 | [-5.77, +11.54] | -4.81 | [-12.50, +2.88] |

*Table 16: Fusion premium against the rerun band. Fusion premium (pp). Comparators are fixed a priori, not per-cell maxima. Paired bootstrap over tasks, 10,000 resamples. Read against the measured rerun band **0.89–2.23pp**, not against zero: a premium must beat what repetition delivers for the same money. No cell clears the band; `cls_B0`'s +2.23 *equals* its upper edge (both are 5/224). Source: `fusion_premium.json`.*

<!-- END table:fusion -->

<!-- BEGIN table:exante -->

| cell | flagged | arm | Δ vs DOM on flagged | 95% CI | Δ on the rest | 95% CI |
|---|---|---|---|---|---|---|
| cls·B0 | 71 | vision | +22.54 | [+9.86, +33.80] | +0.65 | [-5.88, +7.84] |
| cls·B0 | 71 | som | +19.72 | [+7.04, +32.39] | +5.23 | [-1.31, +12.42] |
| cls·B1 | 71 | vision | +16.90 | [+8.45, +25.35] | +1.31 | [-3.92, +6.54] |
| cls·B1 | 71 | som | +12.68 | [+5.63, +21.13] | +5.88 | [+0.00, +11.76] |
| cls·B2 | 71 | vision | +1.41 | [-4.23, +7.04] | +0.65 | [-1.31, +3.27] |
| cls·B2 | 71 | som | +0.00 | [-5.63, +5.63] | +1.31 | [-1.31, +3.92] |
| red·B0 | 63 | vision | -3.17 | [-14.29, +7.94] | -8.57 | [-14.29, -2.86] |
| red·B0 | 63 | som | +0.00 | [-9.52, +9.52] | +0.71 | [-4.29, +5.71] |
| red·B1 | 63 | vision | -3.17 | [-9.52, +3.17] | -3.57 | [-7.86, +0.71] |
| red·B1 | 63 | som | +1.59 | [-3.17, +7.94] | +1.43 | [-2.14, +5.00] |
| red·B2 | 63 | vision | +4.76 | [-1.59, +12.70] | -5.00 | [-8.57, -1.43] |
| red·B2 | 63 | som | -1.59 | [-4.76, +0.00] | -3.57 | [-7.14, +0.00] |

*Table 17: Ex-ante visual-intent partition. Ex-ante partition. The predicate is a regex over the task intent plus 'carries no reference image' — both read the task config, so it costs no tokens and needs no episode. On classifieds the screenshot is worth an order of magnitude more on the flagged tasks than on the rest, and the flagged/rest split is significant/not respectively on the two capable backbones. **Caution:** WebArena is omitted: the predicate fires on only 5 of 104 tasks there and none is solved by any mode, so the cells are degenerate rather than null. Source: `visual_intent_routing.json`.*

<!-- END table:exante -->

<!-- BEGIN table:floor -->

| cell | best single | +1 distinct arm | +1 rerun (measured floor) | verdict |
|---|---|---|---|---|
| cls·B0 | SoM  at 27.23 | +7.14 (DOM) | 4.91 – 7.59pp | **inside the rerun band** |
| cls·B1 | SoM  at 14.29 | +4.91 (Vision) | — | no floor on this cell |
| cls·B2 | SoM  at 2.23 | +2.23 (Vision) | — | no floor on this cell |
| red·B0 | SoM  at 14.78 | +4.93 (DOM) | — | no floor on this cell |
| red·B1 | SoM  at 7.39 | +1.97 (DOM+sprompt) | — | no floor on this cell |
| red·B2 | DOM  at 3.94 | +1.97 (Vision) | — | no floor on this cell |
| WA·B0 | DOM+stext  at 35.58 | +5.77 (DOM) | — | no floor on this cell |
| WA·B1 | DOM  at 16.35 | +4.81 (DOM+stext) | 0.00 – 10.00pp *(pooled would read 2.00–4.00)* | **inside the rerun band** |

*Table 18: New representation versus a rerun. Is a new representation worth more than a rerun? Both middle columns are the same functional at the same arm count — `|{added} ∖ {baseline}| / n` — so they are directly comparable; only the *source* of the extra arm differs. **The band is 3 rerun pairs on one cell** (`B0 × classifieds`, n=224), one each for **DOM, SoM, Vision** — so the rows without a band have no comparator at all, and the band itself is 3 draws rather than a bound. Since the SoM replicate landed 2026-08-03 the band is **no longer extrapolated onto an unreplicated arm**: both the fused mode this table's best-single column keeps selecting and the arm the comparison adds now carry their own measured floor, and adding the third pair left the band unmoved. Source: `noise_floor_inventory.json`.*

<!-- END table:floor -->

<!-- BEGIN table:routing -->

| cell | policy | SR | cost | latency | on frontier |
|---|---|---|---|---|---|
| cls·B0 | always-SoM | 27.23 | 0.07236 | 106.0s | yes |
| cls·B0 | always-Vision | 25.00 | 0.06481 | 125.2s | yes |
| cls·B0 | rule: flag→Vision else DOM | 24.55 | 0.06809 | 117.6s | yes |
| cls·B0 | rule: flag→SoM else DOM+stext | 24.11 | 0.07019 | 115.8s | yes |
| cls·B0 | rule: flag→SoM else DOM | 23.66 | 0.07049 | 111.5s | yes |
| cls·B0 | always-DOM | 17.41 | 0.06962 | 114.1s | no |
| cls·B1 | always-SoM | 14.29 | 0.06028 | 262.0s | yes |
| cls·B1 | always-Vision | 12.50 | 0.04316 | 269.8s | yes |
| cls·B1 | rule: flag→Vision else DOM | 11.61 | 0.05433 | 296.5s | no |
| cls·B1 | rule: flag→SoM else DOM+stext | 11.16 | 0.05926 | 297.2s | no |
| cls·B1 | rule: flag→SoM else DOM | 10.27 | 0.05976 | 294.0s | no |
| cls·B1 | always-DOM | 6.25 | 0.05951 | 308.9s | no |
| cls·B2 | always-SoM | 2.23 | 0.09075 | 374.3s | yes |
| cls·B2 | always-Vision | 2.23 | 0.07065 | 417.8s | yes |
| cls·B2 | rule: flag→Vision else DOM | 1.79 | 0.07483 | 407.2s | yes |
| cls·B2 | always-DOM | 1.34 | 0.07676 | 402.3s | yes |
| cls·B2 | rule: flag→SoM else DOM | 1.34 | 0.08120 | 393.4s | no |
| cls·B2 | rule: flag→SoM else DOM+stext | 1.34 | 0.07876 | 391.5s | yes |
| red·B0 | always-SoM | 14.78 | 0.11045 | 451.6s | yes |
| red·B0 | always-DOM | 14.29 | 0.10147 | 552.5s | yes |
| red·B0 | rule: flag→SoM else DOM | 14.29 | 0.10426 | 521.2s | yes |
| red·B0 | rule: flag→Vision else DOM | 13.30 | 0.10041 | 510.9s | yes |
| red·B0 | rule: flag→SoM else DOM+stext | 13.30 | 0.10722 | 527.8s | no |
| red·B0 | always-Vision | 7.39 | 0.09807 | 418.5s | yes |
| red·B1 | always-SoM | 7.39 | 0.08000 | 609.1s | yes |
| red·B1 | rule: flag→SoM else DOM+stext | 6.90 | 0.07275 | 601.8s | yes |
| red·B1 | rule: flag→SoM else DOM | 6.40 | 0.07538 | 604.6s | no |
| red·B1 | always-DOM | 5.91 | 0.07330 | 602.6s | no |
| red·B1 | rule: flag→Vision else DOM | 4.93 | 0.06682 | 557.3s | yes |
| red·B1 | always-Vision | 2.46 | 0.05240 | 456.6s | yes |
| red·B2 | rule: flag→Vision else DOM | 5.42 | 0.08658 | 632.7s | yes |
| red·B2 | always-DOM | 3.94 | 0.09479 | 669.9s | no |
| red·B2 | rule: flag→SoM else DOM | 3.45 | 0.10001 | 655.4s | no |
| red·B2 | always-Vision | 1.97 | 0.06833 | 550.0s | yes |
| red·B2 | rule: flag→SoM else DOM+stext | 1.48 | 0.09568 | 660.7s | no |
| red·B2 | always-SoM | 0.99 | 0.11160 | 623.0s | no |

*Table 19: Routing policies on the 3-axis frontier. Is routing worth it? `rule` policies send the ex-ante-flagged tasks to one arm and the rest to another; the partition is a regex over the task intent, so nothing is learned and there is no in-sample optimism. 'On frontier' means **nothing dominates it**, not that it is preferable — on `cls·B0` all three rule policies sit between always-SoM and always-Vision, worse on every axis than one or the other. Cost/latency are per-attempt cell means, **within-cell comparable only**. Source: `rule_routing_pareto.json`.*

<!-- END table:routing -->

<!-- BEGIN table:cascade -->

| cell | n | cheap SR | rich SR | always-rich cost | oracle SR | comparable? | beats always-rich — strictly / SR-tied | signals dropped |
|---|---|---|---|---|---|---|---|---|
| cls·B0 | 224 | 25.00 | 27.23 | 1.116× | 33.93 | yes | 0 / 0 | 2 |
| cls·B1 | 224 | 12.50 | 14.29 | 1.397× | 19.20 | yes | 0 / 0 | 0 |
| cls·B2 | 224 | 2.23 | 2.23 | 1.284× | 4.46 | — *(rich is worse)* | **11** / 22 | 0 |
| red·B0 | 203 | 7.39 | 14.78 | 1.126× | 18.23 | yes | 0 / 0 | 0 |
| red·B1 | 203 | 2.46 | 7.39 | 1.527× | 8.37 | yes | 0 / 0 | 2 |
| red·B2 | 203 | 1.97 | 0.99 | 1.633× | 2.96 | — *(rich is worse)* | **41** / 5 | 1 |
| WA·B0 | 104 | 19.23 | 22.12 | 1.054× | 29.81 | yes | 0 / 0 | 0 |
| WA·B1 | 104 | 9.62 | 13.46 | 1.778× | 16.35 | yes | 0 / 3 | 1 |

*Table 20: Confidence-triggered cascade. Confidence-triggered cascade, vision → som. The escalation decision sees only the cheap run's own episode — no outcome, no rich-run information. **In 0 of the 6 comparable cells does any operating point beat always-rich on success rate**; the cells marked *rich is worse* are excluded because the cascade's premise fails there, and that exclusion rule (`cheap_sr >= rich_sr`) is **outcome-dependent**. **Caution:** **The two counts are split for a reason.** A point also 'wins' by matching always-rich's SR at lower cost, and every such point here is a **tie artefact**: on `WA·B1` all of them sit at exactly the rich arm's SR while 44-60 of 104 episodes are tied at the cutoff, so the ranking falls through to task id and the reachable SR spans 8.65-14.42. Counting the two together is what let this table print a WA win for months while the prose said the WA exception was withdrawn — the rows disagreed with the sentence and nobody could see it, because the WA rows were dropped before render until 2026-08-03. **Caution:** **Every number is an offline splice**: an escalated task takes its outcome from a standalone rich run, whereas a real cascade would start the rich episode after the cheap one had already acted on a stateful site. That sequential outcome is unobserved in this project. Source: `confidence_cascade_with_wa.json`.*

<!-- END table:cascade -->

<!-- BEGIN table:cascade-control -->

| cell | signals | **best** margin (10/20/30%) | **median over all signals** (10/20/30%) | signals >0 | oracle headroom captured (10/20/30%) |
|---|---|---|---|---|---|
| cls·B0 | 8 | +1.12 / +1.78 / +2.01 | +1.12 / +1.11 / +0.67 | 19/24 | 15% / 25% / 30% |
| cls·B1 | 10 | +1.16 / +1.43 / +2.14 | +0.27 / +0.53 / +0.81 | 19/30 | 20% / 27% / 40% |
| cls·B2 | 10 | +0.45 / +0.89 / +0.45 | +0.00 / +0.00 / +0.00 | 11/30 | 20% / 40% / 20% |
| red·B0 | 10 | +1.74 / +1.46 / +0.74 | +0.01 / +0.48 / -0.25 | 15/30 | 23% / 27% / 27% |
| red·B1 | 8 | +1.49 / +1.96 / +1.48 | +0.50 / +1.22 / +1.23 | 23/24 | 33% / 50% / 50% |
| red·B2 | 9 | +0.10 / +0.20 / +0.30 | +0.10 / +0.20 / +0.30 | 20/27 | 0% / 0% / 0% |
| WA·B0 | 10 | +3.57 / +3.26 / +3.95 | +1.16 / +1.82 / +1.54 | 24/30 | 36% / 36% / 45% |
| WA·B1 | 9 | +2.51 / +2.11 / +2.70 | -0.37 / -0.78 / -0.18 | 12/27 | 43% / 43% / 57% |

*Table 21: Cascade signal against a random-escalation control. Is the cascade's signal doing anything? Table 20 answers a deployment question — no operating point beats always-rich on success rate — which on its own cannot distinguish a signal that carries nothing from one that carries something insufficient. This is the comparator that separates them: the same escalation budget spent at random. **Caution:** **Read the median column, not the best column.** The best margin is positive in 24 of 24 (cell × fraction) combinations, but each of those is a **maximum over that cell's 8–10 candidate signals against a constant** — the random comparator does not depend on which signal is used — so positivity there is partly arithmetic. Over **all 222 (signal × cell × fraction) points, 64.4% are positive with a median of +0.327pp**: that is the unselected statement, and it is the one to quote. On cls·B2, red·B0, WA·B1 the median is negative at one or more fractions while the best is positive — there the apparent win is entirely selection. The right-hand column is how much of the gap to a per-task oracle the best signal recovers, and inherits the same caveat. The offline-splice caveat on Table 20 applies unchanged. Source: `confidence_cascade_with_wa.json`.*

<!-- END table:cascade-control -->

<!-- BEGIN table:triage -->

| cell | n | solvable | AUROC without | AUROC with visual_difficulty | Δ | visual_difficulty alone |
|---|---|---|---|---|---|---|
| cls·B0 | 224 | 97 | 0.726 | 0.726 | -0.000 | 0.534 |
| cls·B1 | 224 | 55 | 0.732 | 0.726 | -0.006 | 0.553 |
| cls·B2 | 224 | 16 | 0.642 | 0.630 | -0.013 | 0.570 |
| red·B0 | 203 | 53 | 0.780 | 0.776 | -0.005 | 0.630 |
| red·B1 | 203 | 24 | 0.864 | 0.863 | -0.002 | 0.620 |
| red·B2 | 203 | 15 | 0.615 | 0.655 | +0.040 | 0.718 |

*Table 22: Triage learnability and the visual-difficulty feature. Can the triage label be predicted, and does the benchmark's own visual-difficulty annotation help? Task-held-out 5-fold CV, L2 logistic regression, seed 42. Mean ΔAUROC = **+0.0024** over 6 cells, improving 1 — inside fold-split noise. `solvable` is the positive count: the label exists for every task, unlike the which-mode label. **Caution:** **Six cells, not eight, and the reason is the finding.** `visual_difficulty` is a VisualWebArena task-config annotation; WebArena's 106 reddit configs carry **none of** `visual_difficulty`, `reasoning_difficulty`, `overall_difficulty` or `image` — the field simply is not there, so the contrast cannot be computed rather than computing to zero. That absence is what makes WA the clean test elsewhere: where a router can read the benchmark's own difficulty annotation it looks learnable, and WA is the setting where it cannot. Source: `visual_difficulty_router.json`.*

<!-- END table:triage -->

<!-- BEGIN table:feature -->

| cell | tasks with ref image | without | best mode WITH image | SR | best mode WITHOUT | SR |
|---|---|---|---|---|---|---|
| cls·B0 | 65 | 159 | SoM | 40.00 | Vision | 22.64 |
| cls·B1 | 65 | 159 | DOM+stext | 16.92 | SoM | 13.84 |
| cls·B2 | 65 | 159 | SoM | 4.62 | DOM | 1.89 |
| red·B0 | 79 | 124 | SoM | 26.58 | DOM | 7.26 |
| red·B1 | 79 | 124 | SoM | 13.92 | SoM | 3.23 |
| red·B2 | 79 | 124 | DOM | 8.86 | Vision | 3.23 |

*Table 23: The intuitive routing feature. The intuitive routing feature. A task shipping a reference image ought to route to a mode that can see images — the table shows which mode is actually best on each side of that split. **Caution:** Reference images are delivered in **every** mode, so this feature does not separate what it appears to. **WebArena cannot arbitrate**: its 106 reddit configs carry no `image` field at all — nor any of the three difficulty annotations — so both of this table's stratifiers are undefined there. Six cells is the whole population for this question, not a coverage gap. Source: `routing_feature_diagnostics.json`.*

<!-- END table:feature -->

<!-- BEGIN table:cond-text -->

| rule | name | on disagreement | baseline | enrichment | hits |
|---|---|---|---|---|---|
| `P49` | submit-page anchor misclick | 3.7% | 1.0% | **3.61x** | 8 |
| `P17` | click-back oscillation | 4.6% | 3.9% | **1.17x** | 10 |
| `P12` | never paginates | 13.8% | 14.8% | **0.93x** | 30 |
| `P31` | budget exhausted, unfinished | 49.5% | 54.2% | **0.91x** | 108 |
| `P36` | degenerate walk failure | 27.1% | 31.2% | **0.87x** | 59 |
| `P5` | perception-gap loop | 40.8% | 51.5% | **0.79x** | 89 |
| `P14` | URL self-loop | 25.7% | 32.5% | **0.79x** | 56 |
| `P45` | identical failed-action streak | 13.3% | 17.1% | **0.78x** | 29 |

*Table 24: Paired failure attribution: text wins. Only the TEXT channel solved it: how the IMAGE channel failed. Pooled over 8 cells at ruleset v11: 109 disagreement tasks, 218 losing-channel failure episodes against 2653 of that channel's failures overall. Enrichment = hit rate on the disagreement set over hit rate across all that channel's failures; approximately 1x means the loser failed there the way it fails everywhere. Rules with fewer than 8 pooled conditional hits are omitted. **Caution:** TEXT is four arms against IMAGE's two, so the two panels' task counts are not comparable to each other. Source: `conditional_failure_attribution.json`.*

<!-- END table:cond-text -->

<!-- BEGIN table:cond-image -->

| rule | name | on disagreement | baseline | enrichment | hits |
|---|---|---|---|---|---|
| `P27` | gives up when not found | 3.2% | 1.4% | **2.31x** | 13 |
| `P17` | click-back oscillation | 15.1% | 6.7% | **2.25x** | 61 |
| `P16` | visual task, DOM cannot see | 6.2% | 2.8% | **2.24x** | 25 |
| `P43` | page visual, no screenshot | 48.5% | 29.3% | **1.65x** | 196 |
| `P19` | P19 | 2.0% | 1.3% | **1.55x** | 8 |
| `P30` | P30 | 2.0% | 1.4% | **1.39x** | 8 |
| `P4` | root-node misfire | 8.7% | 8.6% | **1.01x** | 35 |
| `P12` | never paginates | 14.4% | 14.6% | **0.99x** | 58 |

*Table 25: Paired failure attribution: image wins. Only the IMAGE channel solved it: how the TEXT channel failed. Pooled over 8 cells at ruleset v11: 101 disagreement tasks, 404 losing-channel failure episodes against 5388 of that channel's failures overall. Enrichment = hit rate on the disagreement set over hit rate across all that channel's failures; approximately 1x means the loser failed there the way it fails everywhere. Rules with fewer than 8 pooled conditional hits are omitted. **Caution:** TEXT is four arms against IMAGE's two, so the two panels' task counts are not comparable to each other. Source: `conditional_failure_attribution.json`.*

<!-- END table:cond-image -->

<!-- BEGIN table:cond-probes -->

| candidate mechanism | on the disagreement set | that channel's baseline | enrichment |
|---|---|---|---|
| finished in five steps or fewer | 0.162 | 0.141 | **1.15×** |
| any parse failure | 0.092 | 0.109 | **0.84×** |
| never searched | 0.387 | 0.468 | **0.83×** |
| ran out of budget (>=30 steps) | 0.535 | 0.658 | **0.81×** |
| page unchanged on over half the steps | 0.331 | 0.431 | **0.77×** |
| action failure on over half the steps | 0.310 | 0.416 | **0.74×** |

*Table 26: Vocabulary-free probes on the text-wins residual. Is the text-wins residual real, or an artefact of the rule vocabulary? Tables 24–25 count rule hits, and the ruleset was discovered on VisualWebArena — so an absent signature there could be a property of the vocabulary rather than of the world. Each probe here is computed from raw step fields and never from a rule hit, over 142 disagreement episodes against 2304 baseline failures. **The largest enrichment is 1.15×** and 5 of 6 sit *below* 1: on the tasks the text channel uniquely solves, the image channel fails **more blandly** than it fails elsewhere — it did not arrive, rather than breaking somewhere nameable. **Caution:** Six candidates chosen by us, so this cannot show that no mechanism exists; what it closes is the specific objection that the residual is an artefact of a VWA-shaped vocabulary. Source: `conditional_failure_attribution.json`.*

<!-- END table:cond-probes -->

<!-- BEGIN table:instability -->

| stratum | tasks | share of cell | flipped | flip rate | share of all flips | enrichment vs complement |
|---|---|---|---|---|---|---|
| which-mode label rows (any mode solved) | 97 | 43.3% | 47 | 48.45% | 95.9% | **16.47x** |
| …of those, the arms DISAGREE (the choice matters) | 88 | 39.3% | 45 | 51.14% | 91.8% | **17.39x** |
| three-way channel decision is contested | 74 | 33.0% | 36 | 48.65% | 73.5% | **16.54x** |
| exactly one mode solved it (label unambiguous) | 29 | 12.9% | 15 | 51.72% | 30.6% | **17.59x** |
| COMPLEMENT: no mode solved, or all did | 136 | 60.7% | 4 | 2.94% | 8.2% | **1.00x** |
| whole cell | 224 | 100.0% | 49 | 21.88% | 100.0% | **7.44x** |

*Table 27: Per-task label instability. Per-task label instability on `cls_B0` (n=224): 49 tasks change outcome between two runs of the same condition. The rows a which-mode router could learn from are exactly the contested ones, and they carry almost all of the instability. **Caution: this table is computed on 2 replicated arms of one cell (B0.cls.dom, B0.cls.vision), rerun once.** The project's whole replicate inventory is three arms of that cell — the SoM pair landed after this product was generated and is not folded in here — plus five modes of `wa_red_B1` on a registered ten-task draw; no VWA-reddit cell and no B2 cell carries one at all, so every stability figure elsewhere is imported rather than measured. The headline enrichment has two defensible definitions and **neither may be quoted alone**: 17.4x defined over all 6 arms is correct for the claim (a router chooses among 6) but the flips are produced by rerunning 2 of them, so the same arms decide both membership and outcome; rebuilding the difficulty proxy from the other 4 breaks that circle and gives 3.95x. Source: `label_instability.json`.*

<!-- END table:instability -->

<!-- BEGIN table:leakage -->

| cell | contrast | before | 95% CI | after | 95% CI | verdict |
|---|---|---|---|---|---|---|
| red·B0 | SoM − Vision | +7.39 | [+2.46, +12.32] | +7.88 | [+2.96, +12.81] | unchanged |
| red·B0 | SoM − DOM | +0.49 | [-3.94, +4.93] | +0.99 | [-3.45, +5.42] | unchanged |
| red·B1 | SoM − Vision | +4.93 | [+1.48, +8.87] | +4.43 | [+0.99, +7.88] | unchanged |
| red·B1 | SoM − DOM | +1.48 | [-1.48, +4.43] | +0.99 | [-1.48, +3.94] | unchanged |
| red·B2 | SoM − Vision | -0.99 | [-3.45, +1.48] | -0.99 | [-3.45, +1.48] | unchanged |
| red·B2 | SoM − DOM | -2.96 | [-5.91, -0.49] | -1.48 | [-3.45, +0.49] | **flips** |

*Table 28: Leaked-success sensitivity. Sensitivity to environmentally-credited successes. `require_reset` is a no-op on reddit, so subscriptions accumulate across a run's episodes and a later task can be scored on state an earlier one created. 6 such successes are set to 0 here — the denominator is unchanged, because an attempted-and-unaccomplished task is a 0, not a missing row. 4 of the leaks are on DOM, so removing them **helps** the fused arm: the direction that disfavours this project's own caution. **Caution:** The WA cells are **unaudited** for the same defect. Source: `leakage_sensitivity.json`.*

<!-- END table:leakage -->

<!-- BEGIN table:offsite -->

| cell | off-site steps | off-site episodes | median env_step on-site | off-site | ratio |
|---|---|---|---|---|---|
| B0·VWA-cla | 0/20646 (0.00%) | 0/1344 (0.0%) | 4,493 ms | — | — |
| B0·VWA-red | 478/26425 (1.81%) | 36/1230 (2.9%) | 11,349 ms | 10,007 ms | 0.88× |
| B1·VWA-cla | 0/27927 (0.00%) | 0/1344 (0.0%) | 5,750 ms | — | — |
| B1·VWA-red | 501/29309 (1.71%) | 54/1230 (4.4%) | 7,848 ms | 6,349 ms | 0.81× |
| B2·VWA-cla | 58/36529 (0.16%) | 4/1344 (0.3%) | 4,656 ms | 14,884 ms | 3.20× |
| B2·VWA-red | 719/33749 (2.13%) | 49/1230 (4.0%) | 9,324 ms | 4,850 ms | 0.52× |
| B1·WA-red | 278/14695 (1.89%) | 40/624 (6.4%) | 6,809 ms | 6,105 ms | 0.90× |
| B0·WA-red | 123/11703 (1.05%) | 21/624 (3.4%) | 6,613 ms | 11,405 ms | 1.72× |

*Table 29: Off-site navigation and container latency. Off-site navigation. Postmill is a link aggregator, so an agent opening a trending thread can walk onto the live public internet; classifieds is self-contained. **Caution:** Off-site steps are **faster**, not slower — commercial CDNs beat a Postmill container sharing a host with the agent — so the distortion runs opposite to the intuition. The larger asymmetry is in the last two columns of the on-site medians: reddit's container costs ~1.69× what classifieds' does before any agent behaviour enters, which is why no between-site latency number is quotable bare. Source: `offsite_navigation_audit.json`.*

<!-- END table:offsite -->

<!-- BEGIN table:axis -->

| cell | metric | text axis | prompt axis | image axis | DOM->SoM-image |
|---|---|---|---|---|---|
| cls·B0 | search_loop | -0.011 | -0.045 | -0.225 | -0.056 |
| cls·B0 | type_frac | -0.169 | +0.049 | +0.030 | -0.140 |
| cls·B0 | scroll_frac | +0.072 | +0.063 | -0.253 | +0.134 |
| cls·B0 | selfcorr_count | +0.088 | -0.115 | -0.091 | -0.020 |
| cls·B0 | click_frac | -0.001 | -0.059 | +0.035 | -0.056 |
| cls·B0 | finish_rate | -0.070 | -0.039 | +0.119 | -0.109 |
| cls·B0 | n_steps | +0.023 | +0.039 | -0.246 | +0.066 |
| cls·B0 | action_repeat_frac | -0.051 | +0.086 | -0.117 | +0.030 |
| cls·B1 | search_loop | +0.066 | -0.160 | -0.203 | -0.094 |
| cls·B1 | type_frac | +0.088 | -0.546 | -0.005 | -0.452 |
| cls·B1 | scroll_frac | -0.097 | +0.025 | -0.062 | -0.065 |
| cls·B1 | selfcorr_count | -0.015 | +0.007 | +0.149 | -0.007 |
| cls·B1 | click_frac | +0.139 | +0.526 | -0.169 | +0.614 |
| cls·B1 | finish_rate | -0.055 | +0.082 | +0.170 | +0.027 |
| cls·B1 | n_steps | +0.101 | -0.108 | -0.253 | -0.011 |
| cls·B1 | action_repeat_frac | +0.053 | +0.002 | -0.174 | +0.048 |
| cls·B2 | search_loop | -0.037 | +0.113 | -0.141 | +0.076 |
| cls·B2 | type_frac | +0.116 | -0.410 | +0.297 | -0.290 |
| cls·B2 | scroll_frac | +0.154 | -0.112 | +0.003 | +0.026 |
| cls·B2 | selfcorr_count | +0.100 | -0.111 | -0.105 | -0.024 |
| cls·B2 | click_frac | -0.476 | +0.485 | -0.097 | +0.011 |
| cls·B2 | finish_rate | -0.217 | -0.038 | +0.333 | -0.256 |
| cls·B2 | n_steps | -0.052 | +0.160 | -0.363 | +0.115 |
| cls·B2 | action_repeat_frac | -0.093 | +0.159 | -0.067 | +0.064 |
| red·B0 | search_loop | -0.170 | -0.104 | +0.052 | -0.275 |
| red·B0 | type_frac | +0.029 | -0.098 | -0.084 | -0.067 |
| red·B0 | scroll_frac | -0.206 | +0.041 | -0.148 | -0.182 |
| red·B0 | selfcorr_count | +0.279 | -0.065 | -0.258 | +0.223 |
| red·B0 | click_frac | -0.057 | +0.053 | +0.081 | -0.009 |
| red·B0 | finish_rate | -0.299 | +0.102 | +0.160 | -0.200 |
| red·B0 | n_steps | +0.276 | -0.032 | -0.252 | +0.245 |
| red·B0 | action_repeat_frac | +0.226 | -0.056 | -0.033 | +0.202 |
| red·B1 | search_loop | +0.020 | -0.100 | -0.297 | -0.080 |
| red·B1 | type_frac | +0.032 | -0.329 | -0.159 | -0.301 |
| red·B1 | scroll_frac | +0.008 | -0.071 | +0.157 | -0.074 |
| red·B1 | selfcorr_count | -0.053 | -0.075 | -0.062 | -0.120 |
| red·B1 | click_frac | +0.019 | +0.326 | -0.016 | +0.370 |
| red·B1 | finish_rate | -0.224 | +0.012 | +0.223 | -0.212 |
| red·B1 | n_steps | +0.202 | -0.044 | -0.231 | +0.165 |
| red·B1 | action_repeat_frac | +0.133 | +0.108 | +0.020 | +0.249 |
| red·B2 | search_loop | +0.092 | -0.115 | -0.179 | -0.024 |
| red·B2 | type_frac | -0.045 | -0.023 | +0.045 | -0.070 |
| red·B2 | scroll_frac | -0.037 | +0.093 | -0.040 | +0.054 |
| red·B2 | selfcorr_count | +0.052 | -0.026 | +0.015 | +0.019 |
| red·B2 | click_frac | -0.193 | +0.143 | -0.017 | -0.047 |
| red·B2 | finish_rate | -0.059 | -0.044 | +0.382 | -0.103 |
| red·B2 | n_steps | -0.126 | +0.057 | -0.137 | -0.054 |
| red·B2 | action_repeat_frac | -0.157 | -0.010 | +0.121 | -0.173 |
| WA·B0 | search_loop | -0.042 | -0.155 | +0.238 | -0.197 |
| WA·B0 | type_frac | +0.476 | -0.197 | -0.031 | +0.336 |
| WA·B0 | scroll_frac | -0.247 | +0.155 | -0.191 | -0.135 |
| WA·B0 | selfcorr_count | +0.106 | -0.053 | +0.088 | +0.078 |
| WA·B0 | click_frac | -0.255 | +0.111 | -0.139 | -0.170 |
| WA·B0 | finish_rate | -0.339 | +0.176 | +0.000 | -0.163 |
| WA·B0 | n_steps | +0.260 | -0.084 | -0.128 | +0.200 |
| WA·B0 | action_repeat_frac | +0.157 | +0.012 | -0.217 | +0.170 |
| WA·B1 | search_loop | +0.081 | -0.200 | -0.270 | -0.119 |
| WA·B1 | type_frac | -0.016 | -0.271 | -0.109 | -0.275 |
| WA·B1 | scroll_frac | +0.072 | -0.133 | +0.155 | -0.064 |
| WA·B1 | selfcorr_count | +0.148 | -0.136 | +0.108 | -0.006 |
| WA·B1 | click_frac | +0.153 | +0.244 | -0.182 | +0.353 |
| WA·B1 | finish_rate | +0.121 | -0.204 | -0.021 | -0.083 |
| WA·B1 | n_steps | +0.071 | +0.058 | +0.010 | +0.121 |
| WA·B1 | action_repeat_frac | -0.034 | +0.218 | +0.029 | +0.168 |

*Table 30: 2x2 axis decomposition. 2×2 axis decomposition. Effect sizes are Cohen's h (binary metrics) or d_z (paired continuous), signed right-minus-left. The compound DOM->SoM-image transition decomposes into a text-payload axis and a prompt-style axis; the image axis is SoM-image->SoM. **Caution:** On mean differences the two decomposition routes agreeing is an **algebraic identity**, so a zero residual is arithmetic and not evidence about an interaction. `B2 × wa_reddit` is absent because B2 never ran WebArena. Source: `axis_effect_size_with_wa.json`.*

<!-- END table:axis -->

<!-- BEGIN table:axis1 -->

| cell | decision effect (mean abs) | macro effect (mean abs) | ratio | >1? |
|---|---|---|---|---|
| cls·B0 | 0.1530 | 0.0606 | **2.52** | yes |
| cls·B1 | 0.1024 | 0.0766 | **1.34** | yes |
| cls·B2 | 0.3159 | 0.1556 | **2.03** | yes |
| red·B0 | 0.2761 | 0.1926 | **1.43** | yes |
| red·B1 | 0.2450 | 0.0863 | **2.84** | yes |
| red·B2 | 0.3877 | 0.0952 | **4.07** | yes |
| WA·B0 | 0.2284 | 0.2351 | **0.97** | **no** |
| WA·B1 | 0.2588 | 0.0869 | **2.98** | yes |

*Table 31: Decision quality versus macro frequency. Does the text axis change per-step decisions more than it changes macro action frequencies? Ratio >1 means yes. Verdict: **generalizes** (site_ok = {'reddit': True, 'classifieds': True, 'wa_reddit': True}). **Caution:** `_site_ok` passes a site if **any** backbone clears 1.0, which is a loose bar — `WA·B0` is 0.97 and the WA site passes on B1's 2.98. Until 2026-08-03 the verdict function named only the two VWA sites literally and did not consult WA at all. Source: `axis1_microbehavior_with_wa.json`.*

<!-- END table:axis1 -->

<!-- BEGIN table:halluc -->

| cell | mode | episodes | failed | with hallucinated ref | rate of failed |
|---|---|---|---|---|---|
| classifieds·B0 | DOM | 224 | 185 | 15 | 8.1% |
| classifieds·B0 | DOM+sprompt | 224 | 180 | 11 | 6.1% |
| classifieds·B0 | SoM-image | 224 | 189 | 2 | 1.1% |
| classifieds·B0 | DOM+stext | 224 | 189 | 9 | 4.8% |
| classifieds·B0 | SoM | 187 | 129 | 2 | 1.6% |
| classifieds·B0 | Vision | 224 | 168 | 0 | 0.0% |
| reddit·B0 | DOM | 203 | 174 | 8 | 4.6% |
| reddit·B0 | DOM+sprompt | 203 | 178 | 8 | 4.5% |
| reddit·B0 | SoM-image | 203 | 181 | 1 | 0.6% |
| reddit·B0 | DOM+stext | 203 | 176 | 6 | 3.4% |
| reddit·B0 | SoM | 203 | 173 | 2 | 1.2% |
| reddit·B0 | Vision | 203 | 188 | 0 | 0.0% |
| classifieds·B1 | DOM | 224 | 210 | 20 | 9.5% |
| classifieds·B1 | DOM+sprompt | 224 | 209 | 33 | 15.8% |
| classifieds·B1 | SoM-image | 224 | 209 | 7 | 3.3% |
| classifieds·B1 | DOM+stext | 224 | 207 | 45 | 21.7% |
| classifieds·B1 | SoM | 224 | 192 | 0 | 0.0% |
| classifieds·B1 | Vision | 224 | 196 | 0 | 0.0% |
| reddit·B1 | DOM | 203 | 191 | 25 | 13.1% |
| reddit·B1 | DOM+sprompt | 203 | 192 | 33 | 17.2% |
| reddit·B1 | SoM-image | 203 | 191 | 3 | 1.6% |
| reddit·B1 | DOM+stext | 203 | 191 | 24 | 12.6% |
| reddit·B1 | SoM | 203 | 188 | 5 | 2.7% |
| reddit·B1 | Vision | 203 | 198 | 0 | 0.0% |
| classifieds·B2 | DOM | 224 | 221 | 82 | 37.1% |
| classifieds·B2 | DOM+sprompt | 224 | 220 | 114 | 51.8% |
| classifieds·B2 | SoM-image | 224 | 222 | 43 | 19.4% |
| classifieds·B2 | DOM+stext | 224 | 223 | 75 | 33.6% |
| classifieds·B2 | SoM | 224 | 219 | 39 | 17.8% |
| classifieds·B2 | Vision | 224 | 219 | 0 | 0.0% |
| reddit·B2 | DOM | 203 | 195 | 127 | 65.1% |
| reddit·B2 | DOM+sprompt | 203 | 203 | 155 | 76.4% |
| reddit·B2 | SoM-image | 203 | 202 | 62 | 30.7% |
| reddit·B2 | DOM+stext | 203 | 199 | 51 | 25.6% |
| reddit·B2 | SoM | 203 | 201 | 59 | 29.4% |
| reddit·B2 | Vision | 203 | 199 | 0 | 0.0% |

*Table 32: Hallucinated element references. Hallucinated element references, ruleset `11-intent-text-fallback` over 36 conditions. An action naming an element id that is not in the observation. **Caution:** `vision` carries no element-id list at all, so this rule is **structurally inapplicable** there rather than measuring zero — the same gate-versus-measurement confusion flagged for P2/P4. Source: `cross_mode_failure_signatures.json`.*

<!-- END table:halluc -->

<!-- BEGIN table:pooled -->

| pool | cell | router SR | cheapest SR | Δ SR | Δ cost | non-dom. |
|---|---|---|---|---|---|---|
| same-family | `cls·B0` | 25.00% | 25.00% | +0.00pp | +13.9% | 46.8% |
| same-family | `cls·B1` | 12.05% | 12.50% | -0.45pp | +33.0% | 33.6% |
| all-three | `cls·B0` | 24.11% | 25.00% | -0.89pp | +14.6% | 34.5% |
| all-three | `cls·B1` | 11.16% | 12.50% | -1.34pp | +36.6% | 21.2% |
| all-three | `cls·B2` | 1.79% | 2.23% | -0.45pp | +4.8% | 18.0% |
| same-family | `red·B0` | 14.29% | 7.39% | +6.90pp | +10.2% | 99.2% |
| same-family | `red·B1` | 4.43% | 2.46% | +1.97pp | +38.2% | 86.8% |
| all-three | `red·B0` | 13.30% | 7.39% | +5.91pp | +2.7% | 98.2% |
| all-three | `red·B1` | 4.43% | 2.46% | +1.97pp | +35.8% | 86.8% |
| all-three | `red·B2` | 1.97% | 1.97% | +0.00pp | +7.4% | 42.5% |
| same-family | `wa_red·B0` | 30.77% | 26.92% | +3.85pp | +6.3% | 83.3% |
| same-family | `wa_red·B1` | 15.38% | 9.62% | +5.77pp | +55.7% | 94.1% |
| all-three ‡ | `wa_red·B0` | 30.77% | 26.92% | +3.85pp | +6.3% | 83.3% |
| all-three ‡ | `wa_red·B1` | 15.38% | 9.62% | +5.77pp | +55.7% | 94.1% |

*Table 33: Pooled tier router. **Pooling backbones restores label supply and does not buy a router.** Labels are scarce per cell, so this arm pools backbones that share a task and routes by the coarsest label available — which of two cost tiers to spend — giving classifieds 152/reddit 77/wa_reddit 86 labelled rows against the 15--97 a single cell affords. H-pool NOT supported — the same-family × cost-tier router dominates always-cheapest in 0/6 cells and is dominated by the fixed-mode menu in every cell. The most favourable corner (agreeing backbones, coarse label, highest ceiling) does not change the negative result. **Δ SR is not the story: the cost column is.** Where the router does raise success it also raises cost by 2.7--55.7\%, so it is buying success at a price rather than routing to a better arm, and it strictly dominates always-cheapest in 0 of 14 arms. **Caution: non-dominance is not dominance.** The `non-dominated` column is an admissibility criterion — a policy buying success for more money passes it — and it must never be quoted as a win rate. **Caution: ‡ marks rows that are identical by construction, not by result** — WebArena has no cross-family backbone, so its all-three pool and its same-family pool are the same pool, reported twice to keep the layout parallel. **Caution:** the pooled task overlap is 50 tasks on classifieds and 20 on reddit, so every per-cell contrast is underpowered; costs are never compared across backbones (an API bill against an electricity estimate); and this probe is post-hoc and exploratory — it is not the preregistered gate and must not be cited as one. Source: `router_pooled_tier_learnability.json`.*

<!-- END table:pooled -->

<!-- BEGIN table:pagechange -->

| mode | no-change rate observed | corrected | cosmetic FP steps | steps |
|---|---|---|---|---|
| DOM | 0.3846 | 0.4332 | 1618 | 33277 |
| phantom_prompt | 0.4222 | 0.4681 | 1528 | 33283 |
| phantom_som | 0.3683 | 0.4169 | 1688 | 34741 |
| phantom_text | 0.3331 | 0.3770 | 1518 | 34610 |
| SoM | 0.4071 | 0.4675 | 1870 | 30921 |
| Vision | 0.5675 | 0.6134 | 1569 | 34151 |

*Table 34: page_changed false positives. `page_changed` false positives. A step can register a page change that is purely cosmetic; correcting for it **raises** every mode's no-change rate. The Micro conclusion is unaffected — Vision remains highest in every cell either way — but a router firing on a 2-step no-change streak would trigger 5321 → 5910 times (+11.1%). Source: `page_change_corrected.json`.*

<!-- END table:pagechange -->

<!-- BEGIN table:evaluator -->

| quantity | value |
|---|---|
| conditions resolved | 36 |
| conditions unresolved | 0 |
| episodes scored | 7686 |
| episodes with no score | 0 |
| episodes with a non-numeric score | 0 |
| distinct score values | [0.0, 1.0] |
| count per value | {"0.0": 7041, "1.0": 645} |

*Table 35: Evaluator granularity. Evaluator granularity over the paper-grade set. Across **7,686 scored episodes in 36 conditions**, the evaluator emits **2 distinct values** (0.0, 1.0) — 645 of them positive — with 0 missing and 0 non-numeric. **There is no graded quality target to regress on.** That is a property of the benchmark's design rather than of this pipeline, and it is a precondition of every routing negative in this document: a router's training signal can only be as fine-grained as the evaluator, so a partially-completed task is indistinguishable from one the agent never started. Source: `evaluator_score_granularity.json`.*

<!-- END table:evaluator -->

<!-- BEGIN table:costclass -->

| site | B0 API dollars/ep | B1 electricity dollars/ep | ratio |
|---|---|---|---|
| reddit | 0.10376 | 0.0012670 | **81.9×** |
| classifieds | 0.06943 | 0.0006480 | **107.1×** |

*Table 36: Two cost estimands. Two cost estimands that are **not the same quantity**. B0 pays a per-token API bill; B1/B2 pay electricity. The ratio is reported to show the scale of the category error, not as a comparison — a paper that divides one by the other is comparing a price to a physical cost. Within a cell, mode-to-mode ratios are safe; across deployment classes only the ordering is. Source: `cost_per_mode.json`.*

<!-- END table:costclass -->

<!-- BEGIN table:cost-protocol -->

| backbone | conditions | billed | canonical action | protocol repair | share of billed | worst parse-error waits/ep |
|---|---|---|---|---|---|---|
| B0 | 12 | $1.0391 | $1.0336 | $0.00553 | **0.53%** | 0.08 (red·DOM+sprompt) |
| B1 | 12 | $0.7690 | $0.7504 | $0.01856 | **2.41%** | 0.16 (red·Vision) |
| B2 | 12 | $1.0373 | $0.9919 | $0.04541 | **4.38%** | 1.23 (cla·SoM) |

*Table 37: What a cost number contains. Every cost figure elsewhere is `cost/ep`, a single number; the schema splits it additively into what bought a canonical action and what was burned re-parsing the model's output. Summed over each backbone's conditions, protocol repair is 0.53% of B0's bill and 4.38% of B2's — a **8.2× spread across model families for the same task set**, and the worst per-episode parse-error wait counts all sit on B2. This is small in absolute terms and is **not** a correction to any reported cost. It bounds something else: a cross-family cost comparison is not quite comparing like with like, because one family pays a protocol tax the others do not. Source: `results/phantom_paper/cross_site/cross_site_aggregation.csv` (via `aggregate_cross_site.py`).*

<!-- END table:cost-protocol -->

<!-- BEGIN table:mech-read -->

| site | modes | examples | pairs | pairs at AUROC 1.000 | worst pair | image gap | text-format gap | prompt-family gap |
|---|---|---|---|---|---|---|---|---|
| classifieds | 6 | 144 | 15 | **15/15** | 1.000 | 0.0416 (L36) | 0.0047 (L36) | 0.0088 (L36) |
| reddit | 6 | 144 | 15 | **15/15** | 1.000 | 0.0386 (L04) | 0.0037 (L36) | 0.0069 (L36) |

*Table 38: Linear readability against geometric magnitude. Are the six representations distinguishable inside the model? Leave-one-task-out AUROC over the residual stream separates **every** mode pair perfectly on both sites, at best layer. The gap columns are why that is worth stating rather than assuming: the image axis peaks at 0.042 cosine while the text-format axis peaks at 0.0047 — **a factor of ~9** — so perfect separability coexists with geometric differences in the third decimal place. **Caution:** Scope: one backbone (B1), 144 examples per site, a single decode step; and AUROC 1.000 on 24 tasks × 6 modes is a **ceiling effect**, not a calibrated effect size. This licenses 'the representation is present and readable', nothing about whether it is used. Source: `mechanism_evidence.json` ← `method42_metrics_v2.json`.*

<!-- END table:mech-read -->

<!-- BEGIN table:mech-patch -->

| site | arm | n | displacement (peak) | convergence to source (peak) | peak at |
|---|---|---|---|---|---|
| classifieds | **real** | 24 | 0.230 | 0.188 | L14 |
| classifieds | random injection | 24 | 0.992 | 0.085 | L02 |
| classifieds | task-shuffled source | 24 | 0.296 | 0.157 | L00 |
| reddit | **real** | 24 | 0.338 | 0.188 | L09 |
| reddit | random injection | 24 | 0.992 | 0.100 | L03 |
| reddit | task-shuffled source | 24 | 0.312 | 0.159 | L20 |

*Table 39: Causal patching against both of its controls. Does the prompt-family signature *do* anything, or is it only visible? Source hidden states from `phantom_som` are patched into a `phantom_text` run, holding image and text-format constant. **Displacement is the wrong column to read**: random injection scores 0.992 there, which is destruction, not steering — only *convergence to the source* speaks to direction. Against random injection the real arm wins clearly (0.188 vs 0.085). **Caution:** **Against the task-shuffled control it barely wins** (0.188 vs 0.157, +20%): a source drawn from an *unrelated task* moves the output almost as far toward itself. What does separate them is **where**: the real arm's convergence peaks mid-stack (L14) and the shuffled arm's collapses to the boundary layer (L00). So the content-specific claim rests on the **layer profile**, not on the magnitude — which is weaker than the write-ups in `docs/checkpoints/mechanism/results/` read, and is the reason this table reports both controls rather than the headline. n = 24 tasks per arm, one backbone, one decode step. Source: `mechanism_evidence.json` ← `patching_continuation_results.json`.*

<!-- END table:mech-patch -->

<!-- BEGIN table:leakaudit -->

| benchmark | cell · mode | scored successes | of which LEAKED | share |
|---|---|---|---|---|
| VWA | B0 · DOM | 5 | 1 | 20.0% |
| VWA | B0 · SoM-image | 2 | 0 | 0.0% |
| VWA | B0 · DOM+sprompt | 3 | 0 | 0.0% |
| VWA | B0 · DOM+stext | 3 | 0 | 0.0% |
| VWA | B0 · SoM | 3 | 0 | 0.0% |
| VWA | B0 · Vision | 2 | 1 | 50.0% |
| VWA | B1 · DOM | 2 | 0 | 0.0% |
| VWA | B1 · SoM-image | 4 | 0 | 0.0% |
| VWA | B1 · DOM+sprompt | 2 | 0 | 0.0% |
| VWA | B1 · DOM+stext | 4 | 0 | 0.0% |
| VWA | B1 · SoM | 3 | 1 | 33.3% |
| VWA | B2 · DOM | 3 | 3 | 100.0% |
| VWA | B2 · SoM | 1 | 0 | 0.0% |
| WA | B0 · DOM | 3 | 0 | 0.0% |
| WA | B0 · SoM-image | 5 | 0 | 0.0% |
| WA | B0 · DOM+stext | 5 | 0 | 0.0% |
| WA | B0 · SoM | 3 | 0 | 0.0% |
| WA | B0 · Vision | 3 | 0 | 0.0% |
| WA | B1 · DOM | 4 | 0 | 0.0% |
| WA | B1 · SoM-image | 5 | 0 | 0.0% |
| WA | B1 · DOM+stext | 5 | 0 | 0.0% |
| WA | B1 · SoM | 3 | 0 | 0.0% |
| WA | B1 · Vision | 1 | 0 | 0.0% |

*Table 40: Earned versus leaked successes. Which successes were earned. `#sidebar > section > ul` is read by 9 VWA reddit tasks; `require_reset` is a no-op on reddit so subscriptions accumulate. **LEAKED** = scored success by an episode that never visited the required forum. On VWA: 6 leaked, 68 earned; Table 28 recomputes every contrast with the leaked ones zeroed. **WebArena audited 2026-08-03** (first time): 50 scored episodes over its 5 sidebar tasks, **0 leaked**. **Caution:** That zero is a *lower bound*, not a clearance — the test asks whether the episode reached the forum, and an episode can arrive at a forum an earlier one subscribed to, read `Unsubscribe`, and finish without acting. One such case is hand-confirmed (`B1`/DOM task 597) and scores `earned` here; a text heuristic for the pattern was tried and rejected because model self-report cannot separate deliberating from acting. Source: `reddit_sidebar_leakage_audit_with_wa.json`.*

<!-- END table:leakaudit -->

<!-- BEGIN table:failmode -->

| | rule | how it failed | on disagreement | baseline | enrichment | hits |
|---|---|---|---|---|---|---|
| **A. image wins** | | *how the text channel failed* | (101 tasks) | | | |
| | `P27` | gives up when not found | 3.2% | 1.4% | **2.31x** | 13 |
| | `P17` | click-back oscillation | 15.1% | 6.7% | **2.25x** | 61 |
| | `P16` | visual task, DOM cannot see | 6.2% | 2.8% | **2.24x** | 25 |
| | `P43` | page visual, no screenshot | 48.5% | 29.3% | **1.65x** | 196 |
| | | *15 further rules* | | | all <= 1.55x | |
| **B. text wins** | | *how the image channel failed* | (109 tasks) | | | |
| | `P49` | submit-page anchor misclick | 3.7% | 1.0% | **3.61x** | 8 |
| | `P17` | click-back oscillation | 4.6% | 3.9% | **1.17x** | 10 |
| | `P12` | never paginates | 13.8% | 14.8% | **0.93x** | 30 |
| | `P31` | budget exhausted, unfinished | 49.5% | 54.2% | **0.91x** | 108 |
| | | *6 further rules* | | | all <= 0.87x | |

*Table 41: Failure modes are asymmetric across channels. **The two channels do not fail the same way.** On tasks only one channel solved, how did the other fail? Pooled over 8 cells at ruleset v11. Enrichment = hit rate on the disagreement set over that channel's hit rate across all its failures, so about 1x means it failed there the way it fails everywhere. Rules with fewer than 8 pooled conditional hits are omitted, and the largest omitted enrichment is stated per block rather than left to the reader. **Block A has named death causes** (top 2.31x, 404 losing-channel episodes against 5388 of that channel's failures overall). **Block B does not**: its top row rests on 8 hits, exactly the reporting floor, and all 8 of them fall in the 2 WebArena cells, none in the other 6; every other rule in that block sits at or below the everywhere-baseline. On the tasks the text channel uniquely solves, the image channel did not break somewhere nameable -- it did not arrive. **Caution:** TEXT is **four arms** and IMAGE is **two**, so the two blocks' task counts are not comparable to each other; read each block against its own baseline column, never across blocks. **Caution:** a per-rule frequency is a distribution of symptoms, not of causes -- the largest rows in most cells are risk markers, not death causes, and only rules whose docstrings record a causal check are verified as such. Source: `conditional_failure_attribution.json`.*

<!-- END table:failmode -->

<!-- BEGIN table:ceiling -->

| cell | n | best single mode | ceiling: any mode solves | headroom | same tasks, lower cost | +1 arm | rerun once |
|---|---|---|---|---|---|---|---|
| WA·B0 | 104 | DOM+stext 35.58% | **51.92%** | +16.35pp | **-10.7%** | +5.77 | 2.00-4.00 |
| cls·B0 | 224 | SoM 27.23% | **43.30%** | +16.07pp | **-12.8%** | +7.14 | 4.91-7.59 |
| WA·B1 | 104 | DOM 16.35% | **30.77%** | +14.42pp | **-26.8%** | +4.81 | 2.00-4.00 |
| red·B0 | 203 | SoM 14.78% | **26.11%** | +11.33pp | **-9.5%** | +4.93 | -- |
| cls·B1 | 224 | SoM 14.29% | **24.55%** | +10.27pp | **-19.4%** | +4.91 | -- |
| red·B1 | 203 | SoM 7.39% | **11.82%** | +4.43pp | **-30.6%** | +1.97 | -- |
| red·B2 | 203 | DOM 3.94% | **7.39%** | +3.45pp | **-26.4%** | +1.97 | -- |
| cls·B2 | 224 | SoM 2.23% | **7.14%** | +4.91pp | **-21.3%** | +2.23 | -- |

*Table 42: What a perfect per-task choice could buy. **Two ceilings, and only one of them survives its own control.** *Ceiling: any mode solves it* is what a perfect per-task choice could reach; it runs 7.1-51.9% against a best single mode of 2.2-35.6%. **Caution:** **That column is a six-arm union against a one-arm baseline.** The arm-matched comparison is the last two columns: adding the single best distinct arm buys +1.97 to +7.14pp, and rerunning an arm already in hand buys a draw in the same range wherever a replicate exists -- so the headroom cannot be attributed to representation diversity rather than to resampling. The union of *five* reruns has never been measured, so the split at higher arm counts is unknown, not estimated. *Same tasks, lower cost* keeps the best mode everywhere and sends only the tasks no mode solves to the cheapest one: success is unchanged **by construction** and cost falls 9.5-30.6% in **8 of 8** cells. That ceiling is immune to the arm-count objection because it adds no arms. Why both are hard to reach: no mode solves 48.1-92.9% of tasks, and the set where a per-task choice even exists is 1.5-34.6% of the cell. **Leaked successes are kept here**, as in every other table; 6 scored successes on reddit were credited without the episode ever visiting the forum the evaluator reads, and zeroing them (denominator unchanged) is reported as a sensitivity analysis rather than folded in — the detection criterion is a lower bound, so it cannot license a corrected point estimate. Both sets of figures are in the product. Source: `routing_ceiling.json`.*

<!-- END table:ceiling -->

<!-- BEGIN table:divergence -->

| site | contrast | tasks | median first divergent step | diverged by step 3 | diverged at step 10+ |
|---|---|---|---|---|---|
| classifieds | DOM vs SoM-image | 22 | 1 | 95.5% | 0.0% |
| classifieds | DOM vs DOM+sprompt | 33 | 1 | 84.8% | 3.0% |
| classifieds | DOM vs DOM+stext | 22 | 1 | 95.5% | 0.0% |
| classifieds | SoM-image vs SoM | 46 | 1 | 93.5% | 0.0% |
| classifieds | DOM+sprompt vs SoM-image | 23 | 1 | 91.3% | 0.0% |
| classifieds | DOM+stext vs SoM-image | 18 | 2 | 72.2% | 0.0% |
| reddit | DOM vs SoM-image | 19 | 0 | 100.0% | 0.0% |
| reddit | DOM vs DOM+sprompt | 18 | 1 | 88.9% | 0.0% |
| reddit | DOM vs DOM+stext | 16 | 0 | 93.8% | 0.0% |
| reddit | SoM-image vs SoM | 20 | 0 | 94.7% | 0.0% |
| reddit | DOM+sprompt vs SoM-image | 17 | 0 | 100.0% | 0.0% |
| reddit | DOM+stext vs SoM-image | 19 | 0 | 100.0% | 0.0% |
| wa_reddit | DOM vs SoM-image | 16 | 1 | 75.0% | 6.2% |
| wa_reddit | DOM vs DOM+sprompt | 19 | 1 | 89.5% | 5.3% |
| wa_reddit | DOM vs DOM+stext | 21 | 1 | 66.7% | 9.5% |
| wa_reddit | SoM-image vs SoM | 19 | 1 | 73.7% | 0.0% |
| wa_reddit | DOM+sprompt vs SoM-image | 13 | 0.5 | 83.3% | 8.3% |
| wa_reddit | DOM+stext vs SoM-image | 19 | 4 | 50.0% | 22.2% |

*Table 43: Where two representations part company. Where two representations part company. Restricted to the tasks whose outcome the two modes **disagree** on — exactly one of them succeeds — so these are the trajectories that produced the difference, not all trajectories. `first divergent step` is the first step at which the two URL signatures differ. **Drift would look different**: two runs that separate through accumulated perturbation share a prefix and part late, whereas on VisualWebArena 72--100% of these tasks have already diverged by step 3 and at most 3.0% diverge at step 10 or later. **Caution: this is markedly weaker on WebArena** (50--89% early, up to 22.2% late), which is why the table is per site and not pooled. **Caution:** each row rests on 13--46 tasks, and element ids are deliberately not used as the anchor — they are neither step-invariant nor mode-invariant, so URL signatures are the only stable comparison. Source: `mechanism_per_task.json` E2.*

<!-- END table:divergence -->

<!-- BEGIN table:coupling -->

| cell | n | best single mode | best SR | routable set (>1 solver) | solvable (any mode) | routable / solvable |
|---|---|---|---|---|---|---|
| `wa_red_B0` | 104 | DOM+stext | 35.58% | 34.6% (36) | 51.9% | 0.67 |
| `cls_B0` | 224 | SoM | 27.23% | 30.4% (68) | 43.3% | 0.70 |
| `wa_red_B1` | 104 | DOM | 16.35% | 17.3% (18) | 30.8% | 0.56 |
| `red_B0` | 203 | SoM | 14.78% | 17.7% (36) | 26.1% | 0.68 |
| `cls_B1` | 224 | SoM | 14.29% | 12.9% (29) | 24.6% | 0.53 |
| `red_B1` | 203 | SoM | 7.39% | 8.4% (17) | 11.8% | 0.71 |
| `red_B2` | 203 | DOM | 3.94% | 1.5% (3) | 7.4% | 0.20 |
| `cls_B2` | 224 | SoM | 2.23% | 1.8% (4) | 7.1% | 0.25 |

*Table 44: Label supply and routing value are one quantity. **Label supply and routing value are the same set.** The rows a which-mode router can be trained on exist only where something succeeded; the rows where a per-task choice exists at all are those with more than one solver. Both are subsets of the solvable set, and across all 8 cells they track the best single mode's success rate with Spearman rho = **0.952** (exact permutation p = 0.0011), mean |best SR - routable share| = **1.65pp**. So the two obstructions the lower bound reports are not independent walls but one wall, whose height is set by how much the agent can do. **Caution: part of this association is structural, not empirical** — both columns are functionals of one solve matrix. The non-structural content is the **near-linearity**: under mode independence the routable share would grow near-quadratically in the per-mode rate at these success levels, and it does not, which says task difficulty dominates mode-task matching. **Caution: the ratio is not constant** — 0.53-0.71 in the 6 cells above the floor, but half that (`red_B2`, `cls_B2`) on 3 and 4 routable tasks, where it is one task wide. **Caution: the cells are not independent** — they share three sites and three backbones, so the permutation null asks whether the pairing is arbitrary, not whether eight systems were sampled. Source: `supply_value_coupling.json`.*

<!-- END table:coupling -->
