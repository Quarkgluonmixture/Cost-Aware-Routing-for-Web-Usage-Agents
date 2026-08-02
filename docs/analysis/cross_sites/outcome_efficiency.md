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

⚠️ marks cells whose best mode has fewer than 10 successes; their ratios are directions at best. The 4 unmarked cells are where this has content.

Among those 4: the cheapest-per-attempt mode stops being cheapest-per-success in **2**, and the fastest-per-attempt mode stops being fastest-per-success in **2**.

## 2. The three channels, side by side

| cell | mode | cost/attempt | SR% | **cost/success** | 95% CI | **latency/success (s)** | 95% CI |
|---|---|---|---|---|---|---|---|
| `cls_B0` | DOM | 0.0696 | 17.41 | **0.400** | [0.299, 0.569] | **660** | [493, 937] |
| `cls_B0` | SoM | 0.0724 | 27.23 | **0.266** | [0.205, 0.355] | **392** | [299, 528] |
| `cls_B0` | Vision | 0.0648 | 25.00 | **0.259** | [0.200, 0.348] | **505** | [386, 685] |
| `red_B0` | DOM | 0.1015 | 14.29 | **0.710** | [0.513, 1.077] | **4004** | [2744, 6286] |
| `red_B0` | SoM | 0.1105 | 14.78 | **0.747** | [0.540, 1.127] | **3122** | [2199, 4754] |
| `red_B0` | Vision | 0.0981 | 7.39 | **1.327** | [0.851, 2.514] | **6087** | [3789, 11502] |
| `cls_B1` | DOM | 0.0595 | 6.25 | **0.952** | [0.600, 1.868] | **4943** | [3103, 9657] |
| `cls_B1` | SoM | 0.0603 | 14.29 | **0.422** | [0.308, 0.626] | **1834** | [1335, 2727] |
| `cls_B1` | Vision | 0.0432 | 12.50 | **0.345** | [0.244, 0.530] | **2158** | [1529, 3324] |
| `red_B1` | DOM | 0.0733 | 5.91 | **1.240** | [0.773, 2.551] | **10194** | [6279, 21313] |
| `red_B1` | SoM | 0.0800 | 7.39 | **1.083** | [0.723, 1.965] | **8242** | [5461, 14851] |
| `red_B1` | Vision | 0.0524 | 2.46 | **2.128** | [1.082, 10.486] | **18537** | [9486, 88866] |
| `cls_B2` | DOM | 0.0768 | 1.34 | **5.732** | [2.441, 17.641] | **30040** | [12790, 92348] |
| `cls_B2` | SoM | 0.0908 | 2.23 | **4.066** | [2.020, 20.017] | **16767** | [8315, 82537] |
| `cls_B2` | Vision | 0.0707 | 2.23 | **3.165** | [1.591, 15.703] | **18716** | [9365, 92388] |
| `red_B2` | DOM | 0.0948 | 3.94 | **2.405** | [1.395, 6.424] | **16999** | [9718, 45674] |
| `red_B2` | SoM | 0.1116 | 0.99 | **11.327** | [4.451, 23.478] | **63235** | [24222, 135165] |
| `red_B2` | Vision | 0.0683 | 1.97 | **3.468** | [1.683, 14.061] | **27914** | [13260, 114243] |

## 3. What this does and does not license

**Licensed.** The efficiency ordering is denominator-dependent, and the denominator the field reports is not the one a deployment pays. The screenshot channel's per-attempt lead is universal and by construction; its per-success lead is not universal. The fused channel's per-attempt cost penalty is real and its per-success latency position is much better than that penalty suggests.

**Not licensed.** Any statement of the form "mode X is more efficient" without a denominator. Also any cross-cell comparison of these ratios: B0 bills an API and B1/B2 are electricity-derived, so only within-cell ordering is meaningful.

⚠️ These ratios inherit the success rate's noise twice over — once in the estimate and once in the fact that success itself moves 0.89–2.23pp between identical reruns (`noise_floor_inventory`). The CIs above capture the first, not the second.
