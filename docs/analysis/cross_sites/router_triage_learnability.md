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
| | **learned, nested threshold (honest)** | 25.45 | 0.07172 | -1.79pp | -0.9% | 47/224 |
| | learned, SR-lossless (in-sample threshold) | 27.23 | 0.07236 | +0.00pp | -0.0% | 0/224 |
| | learned, ≤1pp give-back (in-sample threshold) | 26.34 | 0.06988 | -0.89pp | -3.4% | 78/224 |
| reddit·B0 | best-single (`SoM`) | 14.78 | 0.11045 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 7.39 | 0.09807 | -7.39pp | -11.2% | 203/203 |
| | oracle triage | 14.78 | 0.09998 | +0.00pp | -9.5% | 150/203 |
| | **learned, nested threshold (honest)** | 13.79 | 0.11001 | -0.99pp | -0.4% | 12/203 |
| | learned, SR-lossless (in-sample threshold) | 14.78 | 0.11045 | +0.00pp | -0.0% | 0/203 |
| | learned, ≤1pp give-back (in-sample threshold) | 14.78 | 0.11045 | +0.00pp | -0.0% | 0/203 |
| classifieds·B1 | best-single (`SoM`) | 14.29 | 0.06028 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 12.50 | 0.04316 | -1.79pp | -28.4% | 224/224 |
| | oracle triage | 14.29 | 0.04858 | +0.00pp | -19.4% | 169/224 |
| | **learned, nested threshold (honest)** | 13.84 | 0.05392 | -0.45pp | -10.5% | 59/224 |
| | learned, SR-lossless (in-sample threshold) | 14.29 | 0.05388 | +0.00pp | -10.6% | 53/224 |
| | learned, ≤1pp give-back (in-sample threshold) | 13.39 | 0.05106 | -0.89pp | -15.3% | 101/224 |
| reddit·B1 | best-single (`SoM`) | 7.39 | 0.08000 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 2.46 | 0.05240 | -4.93pp | -34.5% | 203/203 |
| | oracle triage | 7.39 | 0.05554 | +0.00pp | -30.6% | 179/203 |
| | **learned, nested threshold (honest)** | 6.40 | 0.06779 | -0.99pp | -15.3% | 93/203 |
| | learned, SR-lossless (in-sample threshold) | 7.39 | 0.06847 | +0.00pp | -14.4% | 91/203 |
| | learned, ≤1pp give-back (in-sample threshold) | 6.40 | 0.06369 | -0.99pp | -20.4% | 122/203 |
| classifieds·B2 | best-single (`SoM`) | 2.23 | 0.09075 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 2.23 | 0.07065 | +0.00pp | -22.1% | 224/224 |
| | oracle triage | 2.23 | 0.07145 | +0.00pp | -21.3% | 208/224 |
| | **learned, nested threshold (honest)** | 1.34 | 0.07326 | -0.89pp | -19.3% | 199/224 |
| | learned, SR-lossless (in-sample threshold) | 2.23 | 0.07190 | +0.00pp | -20.8% | 212/224 |
| | learned, ≤1pp give-back (in-sample threshold) | 2.23 | 0.07190 | +0.00pp | -20.8% | 212/224 |
| reddit·B2 | best-single (`DOM`) | 3.94 | 0.09479 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 1.97 | 0.06833 | -1.97pp | -27.9% | 203/203 |
| | oracle triage | 3.94 | 0.06974 | +0.00pp | -26.4% | 188/203 |
| | **learned, nested threshold (honest)** | 3.94 | 0.06996 | +0.00pp | -26.2% | 189/203 |
| | learned, SR-lossless (in-sample threshold) | 3.94 | 0.06964 | +0.00pp | -26.5% | 192/203 |
| | learned, ≤1pp give-back (in-sample threshold) | 3.94 | 0.06964 | +0.00pp | -26.5% | 192/203 |

## 3. Is the saving real, or manufactured by the threshold sweep?

| cell | observed SR-lossless saving | median under shuffled labels | p |
|---|---|---|---|
| classifieds·B0 | 0.0% | 2.9% | 1.000 |
| reddit·B0 | 0.0% | 0.0% | 1.000 |
| classifieds·B1 | 10.6% | 10.6% | 0.502 |
| reddit·B1 | 14.4% | 0.5% | 0.030 |
| classifieds·B2 | 20.8% | 20.7% | 0.463 |
| reddit·B2 | 26.5% | 6.3% | 0.005 |

200 permutations per cell. The permutation unit is the whole task bundle (y, succ, cost) against X — permuting only `y` leaves the label disconnected from the outcomes that define it, and its error is not one-directional (measured cls/B1 0.478→0.503 but red/B2 0.040→**0.005**). p is the plus-one Monte Carlo estimator (k+1)/(B+1). The sweep still picks its operating point post hoc, so this column is how much of the observed saving a signal-free pipeline reproduces.


## 4. Verdict

Holm at α=0.05 over the m=6 cells tested (the sweep was run once per cell, so the family is the six cells) — **1 of 6 reject**:

- reddit·B2: p=0.005 vs 0.0083 → reject null
- reddit·B1: p=0.030 vs 0.0100 → **stop — this and all larger p unrejected**

Cells where the learned triage Pareto-beats the trivial always-cheapest fixed policy: **0 of 6**.

Read together — and note this is a **narrower** negative than an earlier draft of this file claimed. The label is predictable (AUROC 0.65-0.72 in 5/6 cells, and unlike the which-mode task it clears the best single covariate in 4/6). Two cells yield no SR-lossless saving at all; two more yield savings a signal-free pipeline reproduces (p ~= 0.50). **One cell, reddit/B2, has a saving that survives Holm at m=6 (p=0.005 vs 0.0083)** — under the corrected bundle-permutation null; the earlier y-only null reported 0.040 for that cell and supported a blanket 'nothing survives' claim, which was wrong.

What still holds, and is the load-bearing statement: **no cell's learned triage Pareto-beats the trivial always-cheapest fixed policy** (0 of 6). In reddit/B2 specifically the learned policy keeps 1.97pp more SR than always-cheapest but pays ~2.4% more cost — a genuine trade-off point, not a dominating one, and not something a deployment would prefer without a stated SR price. So: a detectable signal in one of six cells, worth less than the policy you get for free.

⚠️ Two caveats that cap how far even that reading can go. (1) The operating point is still not fully out-of-sample: `best_mode` / `cheap_mode` are chosen on all six cells' realized outcomes, and the `nested` row's training-side scores come from models whose folds include the outer test rows — a true nested design needs inner-CV scores and per-outer-fold mode selection. (2) The threshold sweep remains post hoc, so the null (which shares it) is the only thing keeping the reported saving honest.

Contrast with the which-mode half: that one fails on label SUPPLY (16-97 labels per cell, 笔记 §383.4). Triage has the labels and the AUROC and still does not beat a fixed policy — a different failure mode, at 2-27% base SR where almost every task is hopeless and 'always take the cheap one' is already close to optimal.

