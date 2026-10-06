---
type: analysis
status: complete
created: 2026-08-02
purpose: does the efficiency ordering survive switching from per-attempt to per-success
post_hoc_exploratory: true
scope_warning: within-cell only (B0 bills an API, B1/B2 are electricity-derived). Ratios at low success counts are directions, not measurements — read the CI.
producer: scripts/analysis/aggregate_outcome_efficiency.py
---

# Per attempt is not per success

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_outcome_efficiency.py`

Every efficiency figure elsewhere in this project is **per attempt**. A deployment buys completed tasks, not attempts, and the two orderings differ. Estimand: `sum(cost) / sum(success)` over the cell's scored tasks, with a paired bootstrap over tasks so the CI carries the success rate's own sampling noise.

## 1. Who wins, under each denominator

| cell | max successes | cheapest / attempt | cheapest / **success** | fastest / attempt | fastest / **success** |
|---|---|---|---|---|---|
| `cls_B0` | 61 | Vision | **Vision** | SoM | **SoM** |
| `red_B0` | 30 | Vision | **DOM** ←flips | Vision | **SoM** ←flips |
| `cls_B1` | 32 | Vision | **Vision** | SoM | **SoM** |
| `red_B1` | 15 | Vision | **SoM** ←flips | Vision | **SoM** ←flips |
| `cls_B2` ⚠️ | 5 | Vision | **Vision** | SoM | **SoM** |
| `red_B2` ⚠️ | 8 | Vision | **DOM** ←flips | Vision | **DOM** ←flips |
| `wa_B1` | 17 | Vision | **P-text** ←flips | DOM | **DOM** |
| `wa_B0` | 37 | DOM | **P-text** ←flips | P-prompt | **P-text** ←flips |

⚠️ marks cells whose best mode has fewer than 10 successes; their ratios are directions at best. The 6 unmarked cells are where this has content.

Among those 6: the cheapest-per-attempt mode stops being cheapest-per-success in **4**, and the fastest-per-attempt mode stops being fastest-per-success in **3**.

## 2. The three channels, side by side

| cell | mode | cost/attempt | SR% | **cost/success** | 95% CI | **latency/success (s)** | 95% CI |
|---|---|---|---|---|---|---|---|
| `cls_B0` | DOM | 0.0696 | 17.41 | **0.400** | [0.300, 0.566] | **660** | [493, 937] |
| `cls_B0` | SoM | 0.0724 | 27.23 | **0.266** | [0.205, 0.356] | **392** | [300, 529] |
| `cls_B0` | Vision | 0.0648 | 25.00 | **0.259** | [0.201, 0.347] | **505** | [387, 685] |
| `red_B0` | DOM | 0.1015 | 14.29 | **0.710** | [0.512, 1.064] | **4004** | [2714, 6264] |
| `red_B0` | SoM | 0.1105 | 14.78 | **0.747** | [0.543, 1.115] | **3122** | [2200, 4736] |
| `red_B0` | Vision | 0.0981 | 7.39 | **1.327** | [0.855, 2.496] | **6087** | [3794, 11444] |
| `cls_B1` | DOM | 0.0595 | 6.25 | **0.952** | [0.604, 1.815] | **4943** | [3130, 9454] |
| `cls_B1` | SoM | 0.0603 | 14.29 | **0.422** | [0.308, 0.616] | **1834** | [1331, 2688] |
| `cls_B1` | Vision | 0.0432 | 12.50 | **0.345** | [0.244, 0.534] | **2158** | [1525, 3339] |
| `red_B1` | DOM | 0.0733 | 5.91 | **1.240** | [0.775, 2.514] | **10194** | [6312, 20955] |
| `red_B1` | SoM | 0.0800 | 7.39 | **1.083** | [0.718, 1.985] | **8242** | [5450, 14857] |
| `red_B1` | Vision | 0.0524 | 2.46 | **2.128** | [1.067, 10.303] | **18537** | [9409, 84929] |
| `cls_B2` | DOM | 0.0768 | 1.34 | **5.732** | [2.459, 17.656] | **30040** | [12913, 92461] |
| `cls_B2` | SoM | 0.0908 | 2.23 | **4.066** | [2.063, 19.867] | **16767** | [8513, 82120] |
| `cls_B2` | Vision | 0.0707 | 2.23 | **3.165** | [1.584, 15.724] | **18716** | [9336, 92744] |
| `red_B2` | DOM | 0.0948 | 3.94 | **2.405** | [1.389, 6.431] | **16999** | [9715, 45442] |
| `red_B2` | SoM | 0.1116 | 0.99 | **11.327** | [4.378, 23.512] | **63235** | [23720, 135761] |
| `red_B2` | Vision | 0.0683 | 1.97 | **3.468** | [1.680, 14.085] | **27914** | [13295, 113367] |
| `wa_B1` | DOM | 0.0658 | 16.35 | **0.402** | [0.262, 0.703] | **2968** | [1895, 5219] |
| `wa_B1` | SoM | 0.0794 | 13.46 | **0.590** | [0.374, 1.124] | **3676** | [2283, 7098] |
| `wa_B1` | Vision | 0.0447 | 9.62 | **0.465** | [0.272, 1.024] | **5104** | [2905, 12027] |
| `wa_B0` | DOM | 0.0753 | 26.92 | **0.280** | [0.196, 0.432] | **1055** | [719, 1644] |
| `wa_B0` | SoM | 0.0911 | 22.12 | **0.412** | [0.280, 0.659] | **1231** | [819, 1975] |
| `wa_B0` | Vision | 0.0864 | 19.23 | **0.449** | [0.299, 0.763] | **1757** | [1158, 2994] |

## 3. What this does and does not license

**Licensed.** The efficiency ordering is denominator-dependent, and the denominator the field reports is not the one a deployment pays. The screenshot channel's per-attempt lead is universal and by construction; its per-success lead is not universal. The fused channel's per-attempt cost penalty is real and its per-success latency position is much better than that penalty suggests.

**Not licensed.** Any statement of the form "mode X is more efficient" without a denominator. Also any cross-cell comparison of these ratios: B0 bills an API and B1/B2 are electricity-derived, so only within-cell ordering is meaningful.

⚠️ These ratios inherit the success rate's noise twice over — once in the estimate and once in the fact that success itself moves 0.89–2.23pp between identical reruns (`noise_floor_inventory`). The CIs above capture the first, not the second.

## Extension cells — B5 x classifieds, B0/B1 x shopping

Added 2026-10-07 (实验笔记 §538); not in any count above. Within-cell only (B0 is API-billed, B1 electricity-derived). Shopping mode comparisons carry B-2002 and the catalog-state drift (§534).

| cell | cheapest per attempt | cheapest per success | same? |
|---|---|---|---|
| `shopping_B0` | Vision | Vision | yes |
| `shopping_B1` | Vision | Vision | yes |
| `classifieds_B5` | DOM | SoM | **no** |

| cell | mode | successes | cost / attempt | cost / success [95% CI] |
|---|---|---:|---:|---|
| `shopping_B0` | DOM | 49 | 0.1199 | 1.0570 [0.8123, 1.4534] |
| `shopping_B0` | SoM | 64 | 0.0977 | 0.6595 [0.5211, 0.8671] |
| `shopping_B0` | Vision | 62 | 0.0748 | 0.5215 [0.4107, 0.6872] |
| `shopping_B1` | DOM | 21 | 0.1111 | 2.2850 [1.5617, 3.8260] |
| `shopping_B1` | SoM | 33 | 0.0827 | 1.0824 [0.8017, 1.6007] |
| `shopping_B1` | Vision | 24 | 0.0494 | 0.8897 [0.6154, 1.4231] |
| `shopping_B1` | P-text | 29 | 0.0787 | 1.1725 [0.8506, 1.7834] |
| `shopping_B1` | P-prompt | 23 | 0.1142 | 2.1450 [1.5008, 3.4634] |
| `shopping_B1` | P-SoM | 20 | 0.0813 | 1.7550 [1.2002, 3.0055] |
| `classifieds_B5` | DOM | 53 | 0.1374 | 0.5808 [0.4499, 0.7758] |
| `classifieds_B5` | SoM | 83 | 0.1671 | 0.4510 [0.3581, 0.5763] |
| `classifieds_B5` | P-text | 54 | 0.1468 | 0.6089 [0.4713, 0.8121] |
| `classifieds_B5` | P-prompt | 49 | 0.1529 | 0.6990 [0.5352, 0.9492] |
| `classifieds_B5` | P-SoM | 51 | 0.1402 | 0.6156 [0.4708, 0.8361] |
