# Stage 4 Robustness Suite (Method 4.2 caveat coverage)

## Test A: Label Permutation Negative Control

P-SoM↔SoM at L17 — does AUROC=1.000 survive random label shuffles?

- **Real AUROC** (true labels): **1.0000**
- **Permuted AUROC** (n=200 random shuffles): mean = 0.6294 ± 0.0376
- **95% CI of perm**: [0.6085, 0.7401]
- **p-value**: 0.0050

→ Real signal is **9.8σ above permutation baseline**. Cosine-gap AUROC is NOT achievable from random label noise.

## Test B: Per-Task Cosine Gap Variance

Mean (cosine gap) computed separately per (task × step pair) and aggregated over 24 tasks at L17:

| Mode pair | n tasks | Mean gap | Std | Range | % tasks with positive gap |
|---|---|---|---|---|---|
| P-SoM vs SoM | 24 | 0.0426 | 0.0022 | [0.0386, 0.0458] | 100% |
| P-SoM vs P-text | 24 | 0.0031 | 0.0003 | [0.0027, 0.0039] | 100% |
| P-SoM vs P-prompt | 24 | 0.0124 | 0.0016 | [0.0075, 0.0150] | 100% |
| DOM vs P-prompt | 24 | 0.0020 | 0.0002 | [0.0017, 0.0027] | 100% |
| P-SoM vs DOM | 24 | 0.0138 | 0.0016 | [0.0085, 0.0168] | 100% |
| P-text vs SoM | 24 | 0.0481 | 0.0025 | [0.0441, 0.0526] | 100% |

## Test C: Per-Step Comparison (step 2 vs step 5)

| Mode pair | Step 2 gap | Step 5 gap |
|---|---|---|
| P-SoM vs SoM | 0.0414 | 0.0411 |
| P-SoM vs P-text | 0.0028 | 0.0028 |
| P-SoM vs P-prompt | 0.0120 | 0.0107 |

## Test D: Silhouette Score Across Layers

Silhouette = (between-cluster - within-cluster) / max, range [-1, 1]. Higher = cleaner mode separation.

| Layer | Silhouette |
|---|---|
| L04 | 0.3852 |
| L11 | 0.4260 |
| L17 | 0.4710 |
| L23 | 0.5166 |
| L30 | 0.5168 |
| L36 | 0.5088 |

## Test E: Bootstrap 95% CI (n=1000, task-level resample)

| Mode pair | Mean | 95% CI |
|---|---|---|
| P-SoM vs SoM | 0.0413 | [0.0403, 0.0422] |
| P-SoM vs P-text | 0.0028 | [0.0027, 0.0029] |
| P-SoM vs P-prompt | 0.0113 | [0.0105, 0.0119] |
| DOM vs P-prompt | 0.0013 | [0.0012, 0.0014] |
| DOM vs Vision | 0.0547 | [0.0531, 0.0563] |

