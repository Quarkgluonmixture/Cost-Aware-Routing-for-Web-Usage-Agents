# Is the triage half of routing learnable?

`post_hoc_exploratory=True`, `h10_eligible=False`. Task-held-out 5-fold CV per cell, seed 42, L2 LR on the 20 raw features, fold-local standardisation.

Triage policy: predicted-hopeless → cheapest mode, otherwise → best-SR mode. The oracle row knows the true label; the learned rows use out-of-fold scores.


## 1. Can the label be predicted at all?

| cell | n | solvable % | AUROC LR | AUROC best single feature | Δ | that feature |
|---|---|---|---|---|---|---|
| classifieds·B0 | 224 | 43.3 | **0.676** | 0.607 | +0.069 | `dom_complexity` |
| reddit·B0 | 203 | 26.1 | **0.666** | 0.612 | +0.054 | `intent_nav` |
| classifieds·B1 | 224 | 24.6 | **0.717** | 0.627 | +0.090 | `intent_compare` |
| reddit·B1 | 203 | 11.8 | **0.685** | 0.637 | +0.048 | `intent_account_action` |
| classifieds·B2 | 224 | 7.1 | **0.651** | 0.655 | -0.005 | `intent_compare` |
| reddit·B2 | 203 | 7.4 | **0.483** | 0.711 | -0.228 | `dom_complexity` |

A prior-only predictor scores 0.500 by construction. The single-feature column is the §367 check: if the LR does not clear it, 'learnable' means 'learnable by one covariate', which is not a router.


## 2. What does the policy actually buy?

| cell | policy | SR % | mean cost | ΔSR | Δcost | sent to cheapest |
|---|---|---|---|---|---|---|
| classifieds·B0 | best-single (`SoM`) | 27.23 | 0.07236 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 25.00 | 0.06481 | -2.23pp | -10.4% | 224/224 |
| | oracle triage | 27.23 | 0.06312 | +0.00pp | -12.8% | 127/224 |
| | **learned, nested threshold (honest)** | 26.79 | 0.07197 | -0.45pp | -0.5% | 14/224 |
| | learned, SR-lossless (in-sample threshold) | 27.23 | 0.07236 | +0.00pp | -0.0% | 0/224 |
| | learned, ≤1pp give-back (in-sample threshold) | 26.34 | 0.06988 | -0.89pp | -3.4% | 78/224 |
| reddit·B0 | best-single (`SoM`) | 14.78 | 0.11045 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 7.39 | 0.09807 | -7.39pp | -11.2% | 203/203 |
| | oracle triage | 14.78 | 0.09998 | +0.00pp | -9.5% | 150/203 |
| | **learned, nested threshold (honest)** | 12.81 | 0.09836 | -1.97pp | -10.9% | 55/203 |
| | learned, SR-lossless (in-sample threshold) | 14.78 | 0.11045 | +0.00pp | -0.0% | 0/203 |
| | learned, ≤1pp give-back (in-sample threshold) | 14.78 | 0.11045 | +0.00pp | -0.0% | 0/203 |
| classifieds·B1 | best-single (`SoM`) | 14.29 | 0.06028 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 12.50 | 0.04316 | -1.79pp | -28.4% | 224/224 |
| | oracle triage | 14.29 | 0.04858 | +0.00pp | -19.4% | 169/224 |
| | **learned, nested threshold (honest)** | 14.29 | 0.05757 | +0.00pp | -4.5% | 26/224 |
| | learned, SR-lossless (in-sample threshold) | 14.29 | 0.05388 | +0.00pp | -10.6% | 53/224 |
| | learned, ≤1pp give-back (in-sample threshold) | 13.39 | 0.05106 | -0.89pp | -15.3% | 101/224 |
| reddit·B1 | best-single (`SoM`) | 7.39 | 0.08000 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 2.46 | 0.05240 | -4.93pp | -34.5% | 203/203 |
| | oracle triage | 7.39 | 0.05554 | +0.00pp | -30.6% | 179/203 |
| | **learned, nested threshold (honest)** | 5.91 | 0.06970 | -1.48pp | -12.9% | 81/203 |
| | learned, SR-lossless (in-sample threshold) | 7.39 | 0.06847 | +0.00pp | -14.4% | 91/203 |
| | learned, ≤1pp give-back (in-sample threshold) | 6.40 | 0.06369 | -0.99pp | -20.4% | 122/203 |
| classifieds·B2 | best-single (`SoM`) | 2.23 | 0.09075 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 2.23 | 0.07065 | +0.00pp | -22.1% | 224/224 |
| | oracle triage | 2.23 | 0.07145 | +0.00pp | -21.3% | 208/224 |
| | **learned, nested threshold (honest)** | 1.34 | 0.07247 | -0.89pp | -20.1% | 201/224 |
| | learned, SR-lossless (in-sample threshold) | 2.23 | 0.07190 | +0.00pp | -20.8% | 212/224 |
| | learned, ≤1pp give-back (in-sample threshold) | 2.23 | 0.07190 | +0.00pp | -20.8% | 212/224 |
| reddit·B2 | best-single (`DOM`) | 3.94 | 0.09479 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 1.97 | 0.06833 | -1.97pp | -27.9% | 203/203 |
| | oracle triage | 3.94 | 0.06974 | +0.00pp | -26.4% | 188/203 |
| | **learned, nested threshold (honest)** | 3.94 | 0.06964 | +0.00pp | -26.5% | 192/203 |
| | learned, SR-lossless (in-sample threshold) | 3.94 | 0.06964 | +0.00pp | -26.5% | 192/203 |
| | learned, ≤1pp give-back (in-sample threshold) | 3.94 | 0.06964 | +0.00pp | -26.5% | 192/203 |

## 3. Is the saving real, or manufactured by the threshold sweep?

| cell | observed SR-lossless saving | median under shuffled labels | p |
|---|---|---|---|
| classifieds·B0 | 0.0% | 3.3% | 1.000 |
| reddit·B0 | 0.0% | 0.0% | 1.000 |
| classifieds·B1 | 10.6% | 10.1% | 0.483 |
| reddit·B1 | 14.4% | 1.0% | 0.015 |
| classifieds·B2 | 20.8% | 20.8% | 0.515 |
| reddit·B2 | 26.5% | 5.5% | 0.0005 |

Smallest reportable p at B=10000 is 1/(B+1) = 1.00e-04; Holm's tightest threshold over six cells is 0.05/6 = 8.33e-3. B is therefore not what decides any cell's verdict (it was at B=200, where the floor 4.98e-3 sat inside the threshold and the surviving cell reported exactly it).


10000 permutations per cell. The permutation unit is the whole task bundle (y, succ, cost) against X — permuting only `y` leaves the label disconnected from the outcomes that define it, and its error is not one-directional (measured at B=200: cls/B1 0.478→0.503 but red/B2 0.040→0.005; both figures are from that era, not from the current B). p is the plus-one Monte Carlo estimator (k+1)/(B+1). The sweep still picks its operating point post hoc, so this column is how much of the observed saving a signal-free pipeline reproduces.


## 4. Verdict

Holm at α=0.05 over the m=6 cells tested (the sweep was run once per cell, so the family is the six cells) — **1 of 6 reject**:

- reddit·B2: p=0.0005 vs 0.0083 → reject null
- reddit·B1: p=0.015 vs 0.0100 → **stop — this and all larger p unrejected**

Cells where the learned triage Pareto-beats the trivial always-cheapest fixed policy: **0 of 6**.

Read together — and note this is a **narrower** negative than an earlier draft of this file claimed. In five of six cells the label is predictable (AUROC 0.651-0.717, and unlike the which-mode task it clears the best single covariate in 4/6). Two cells yield no SR-lossless saving at all; two more yield savings a signal-free pipeline reproduces (p ~= 0.50). **One cell, reddit·B2 (p=0.0005 vs 0.0083), has a saving that survives Holm at m=6** — under the corrected bundle-permutation null; the earlier y-only null reported 0.040 for reddit/B2 and supported a blanket 'nothing survives' claim, which was wrong.

⚠️ **The sixth cell is the significant one, and its AUROC is 0.483** — below chance, and below its own best single covariate (0.711). That is not a contradiction: the two quantities measure different things, and on this data they come apart. AUROC scores the GLOBAL ranking; the saving comes from the TAIL. reddit/B2 sends 192 of 203 tasks (95%) to the cheap mode with no SR loss — in a cell where only 7.4% of tasks are solvable at all, almost nothing in that 95% was ever going to succeed. It differs from the free always-cheapest policy by five percent of the task allocation, and those 11 retained tasks happen to hold 4 successes (8 vs 4). The permutation null is detecting that tail enrichment, not a globally ordered score.
So the honest phrasing is NOT 'the label is predictable, yet triage fails'. It is: **at 2-27% base SR, a high AUROC is neither necessary nor sufficient — what decides whether triage saves anything is whether a handful of tail tasks land on the right side, and at n=203 that handful is 4 successes.**

What still holds, and is the load-bearing statement: **no cell's learned triage Pareto-beats the trivial always-cheapest fixed policy** (0 of 6). In reddit/B2 specifically the learned policy keeps 1.97pp more SR than always-cheapest but pays ~2.4% more cost — a genuine trade-off point, not a dominating one, and not something a deployment would prefer without a stated SR price. So: a detectable signal in one of six cells, worth less than the policy you get for free.

⚠️ What is and is not out-of-sample. The **nested** row is now FULLY nested (B-1903, 2026-07-27): per outer fold the modes are re-selected from training-row SR/cost, the threshold is chosen against inner-CV out-of-fold scores over the training rows only, and the outer-test rows are scored by an LR fitted on training rows alone — nothing that touches an outer test fold has seen it. An earlier revision of this file claimed a nested operating point while reusing the GLOBAL out-of-fold scores (whose folds include the outer test rows) and a whole-cell choice of `best_mode`/`cheap_mode`; codex caught that, and the numbers here supersede it. The remaining caveat is that the threshold **sweep** rows are still post hoc by construction, which is why the permutation null shares that same selection step — the null is what keeps the swept saving honest, and the nested row is what an actual deployment would get.

One thing the nested design exposes that the whole-cell version hid: `best_mode` is **not stable across folds**. In reddit·B0 the five outer folds select DOM, DOM, SoM, SoM, DOM. A pipeline that picks one best mode from all realized outcomes is therefore not merely optimistic about the threshold — it is reporting a mode choice that its own resampling does not reproduce.

Contrast with the which-mode half: that one fails on label SUPPLY (16-97 labels per cell, 笔记 §383.4). Triage has the labels and the AUROC and still does not beat a fixed policy — a different failure mode, at 2-27% base SR where almost every task is hopeless and 'always take the cheap one' is already close to optimal.

