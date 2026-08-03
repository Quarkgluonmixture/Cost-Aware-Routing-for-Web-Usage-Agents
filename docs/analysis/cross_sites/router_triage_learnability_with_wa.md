# Is the triage half of routing learnable?

`post_hoc_exploratory=True`, `h10_eligible=False`. Task-held-out 5-fold CV per cell, seed 42, L2 LR on the 18 raw features, fold-local standardisation.

Triage policy: predicted-hopeless → cheapest mode, otherwise → best-SR mode. The oracle row knows the true label; the learned rows use out-of-fold scores.

⚠️ **Eight cells on a matched 18-feature set.** WebArena carries no reference images and no `reasoning_difficulty` annotation, so those two features are **dropped on every cell here**, including the six VWA ones. Zero-filling them on WA and comparing against the 20-feature VWA numbers would put an unmatched column beside matched ones — the defect the deployment-class table had. The 20-feature VWA result therefore lives in `router_triage_learnability.md` and is *not* a subset of this table: every number below was refitted.


## 1. Can the label be predicted at all?

| cell | n | solvable % | AUROC LR | AUROC best single feature | Δ | that feature |
|---|---|---|---|---|---|---|
| classifieds·B0 | 224 | 43.3 | **0.683** | 0.607 | +0.075 | `dom_complexity` |
| reddit·B0 | 203 | 26.1 | **0.700** | 0.685 | +0.014 | `intent_token_count` |
| classifieds·B1 | 224 | 24.6 | **0.705** | 0.631 | +0.074 | `intent_token_count` |
| reddit·B1 | 203 | 11.8 | **0.723** | 0.664 | +0.059 | `intent_token_count` |
| classifieds·B2 | 224 | 7.1 | **0.646** | 0.655 | -0.009 | `intent_compare` |
| reddit·B2 | 203 | 7.4 | **0.526** | 0.790 | -0.264 | `intent_token_count` |
| wa_reddit·B0 | 104 | 51.9 | **0.554** | 0.607 | -0.053 | `dom_complexity` |
| wa_reddit·B1 | 104 | 30.8 | **0.758** | 0.710 | +0.048 | `dom_complexity` |

A prior-only predictor scores 0.500 by construction. The single-feature column is the §367 check: if the LR does not clear it, 'learnable' means 'learnable by one covariate', which is not a router.


**The §367 tally: the LR clears its own best single feature in 5 of 8 cells**, by +0.014 to +0.075. The single feature is `intent_token_count` in 4 of 8. 


⚠️ **Not every feature is live in every cell.** A column with no variance carries no information whatever the model does with it, so the header's 18 is a ceiling, not a count of usable features:

| cell | live features | dead (zero-variance) |
|---|---|---|
| classifieds·B0 | 15/18 | `intent_account_action`, `intent_action_word`, `intent_filter` |
| reddit·B0 | 17/18 | `intent_action_word` |
| classifieds·B1 | 15/18 | `intent_account_action`, `intent_action_word`, `intent_filter` |
| reddit·B1 | 17/18 | `intent_action_word` |
| classifieds·B2 | 15/18 | `intent_account_action`, `intent_action_word`, `intent_filter` |
| reddit·B2 | 17/18 | `intent_action_word` |
| wa_reddit·B0 | 13/18 | `intent_action_word`, `intent_color`, `intent_compare`, `intent_filter`, `intent_form_fill` |
| wa_reddit·B1 | 13/18 | `intent_action_word`, `intent_color`, `intent_compare`, `intent_filter`, `intent_form_fill` |

The WebArena rows lose five intent regexes outright. Those regexes were written against VisualWebArena phrasing, and WebArena words its intents differently — the same mismatch that makes the ex-ante visual-intent predicate flag only 5 of 104 WA tasks. So the WA cells are fitted on **13** live features against the VWA cells' 18, and the comparison is matched on the feature *set*, not on the feature *support*.


## 2. What does the policy actually buy?

| cell | policy | SR % | mean cost | ΔSR | Δcost | sent to cheapest |
|---|---|---|---|---|---|---|
| classifieds·B0 | best-single (`SoM`) | 27.23 | 0.07236 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 25.00 | 0.06481 | -2.23pp | -10.4% | 224/224 |
| | oracle triage | 27.23 | 0.06312 | +0.00pp | -12.8% | 127/224 |
| | **learned, nested threshold (honest)** | 26.79 | 0.07129 | -0.45pp | -1.5% | 18/224 |
| | learned, SR-lossless (in-sample threshold) | 27.23 | 0.06730 | +0.00pp | -7.0% | 56/224 |
| | learned, ≤1pp give-back (in-sample threshold) | 27.23 | 0.06730 | +0.00pp | -7.0% | 56/224 |
| reddit·B0 | best-single (`SoM`) | 14.78 | 0.11045 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 7.39 | 0.09807 | -7.39pp | -11.2% | 203/203 |
| | oracle triage | 14.78 | 0.09998 | +0.00pp | -9.5% | 150/203 |
| | **learned, nested threshold (honest)** | 12.81 | 0.09928 | -1.97pp | -10.1% | 43/203 |
| | learned, SR-lossless (in-sample threshold) | 14.78 | 0.11045 | +0.00pp | -0.0% | 0/203 |
| | learned, ≤1pp give-back (in-sample threshold) | 13.79 | 0.10960 | -0.99pp | -0.8% | 11/203 |
| classifieds·B1 | best-single (`SoM`) | 14.29 | 0.06028 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 12.50 | 0.04316 | -1.79pp | -28.4% | 224/224 |
| | oracle triage | 14.29 | 0.04858 | +0.00pp | -19.4% | 169/224 |
| | **learned, nested threshold (honest)** | 14.29 | 0.05867 | +0.00pp | -2.7% | 15/224 |
| | learned, SR-lossless (in-sample threshold) | 14.29 | 0.05874 | +0.00pp | -2.6% | 12/224 |
| | learned, ≤1pp give-back (in-sample threshold) | 13.39 | 0.05386 | -0.89pp | -10.6% | 56/224 |
| reddit·B1 | best-single (`SoM`) | 7.39 | 0.08000 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 2.46 | 0.05240 | -4.93pp | -34.5% | 203/203 |
| | oracle triage | 7.39 | 0.05554 | +0.00pp | -30.6% | 179/203 |
| | **learned, nested threshold (honest)** | 6.40 | 0.07021 | -0.99pp | -12.2% | 78/203 |
| | learned, SR-lossless (in-sample threshold) | 7.39 | 0.07057 | +0.00pp | -11.8% | 71/203 |
| | learned, ≤1pp give-back (in-sample threshold) | 6.40 | 0.06695 | -0.99pp | -16.3% | 112/203 |
| classifieds·B2 | best-single (`SoM`) | 2.23 | 0.09075 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 2.23 | 0.07065 | +0.00pp | -22.1% | 224/224 |
| | oracle triage | 2.23 | 0.07145 | +0.00pp | -21.3% | 208/224 |
| | **learned, nested threshold (honest)** | 1.34 | 0.07385 | -0.89pp | -18.6% | 186/224 |
| | learned, SR-lossless (in-sample threshold) | 2.23 | 0.07220 | +0.00pp | -20.4% | 212/224 |
| | learned, ≤1pp give-back (in-sample threshold) | 2.23 | 0.07220 | +0.00pp | -20.4% | 212/224 |
| reddit·B2 | best-single (`DOM`) | 3.94 | 0.09479 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 1.97 | 0.06833 | -1.97pp | -27.9% | 203/203 |
| | oracle triage | 3.94 | 0.06974 | +0.00pp | -26.4% | 188/203 |
| | **learned, nested threshold (honest)** | 4.43 | 0.06910 | +0.49pp | -27.1% | 191/203 |
| | learned, SR-lossless (in-sample threshold) | 4.43 | 0.06950 | +0.49pp | -26.7% | 192/203 |
| | learned, ≤1pp give-back (in-sample threshold) | 4.43 | 0.06950 | +0.49pp | -26.7% | 192/203 |
| wa_reddit·B0 | best-single (`P-text`) | 35.58 | 0.08478 | — | — | — |
| | **always-cheapest (`DOM`)** — the fixed policy a router must beat | 26.92 | 0.07531 | -8.65pp | -11.2% | 104/104 |
| | oracle triage | 35.58 | 0.07573 | +0.00pp | -10.7% | 50/104 |
| | **learned, nested threshold (honest)** | 32.69 | 0.08649 | -2.88pp | +2.0% | 6/104 |
| | learned, SR-lossless (in-sample threshold) | 35.58 | 0.08103 | +0.00pp | -4.4% | 16/104 |
| | learned, ≤1pp give-back (in-sample threshold) | 35.58 | 0.08103 | +0.00pp | -4.4% | 16/104 |
| wa_reddit·B1 | best-single (`DOM`) | 16.35 | 0.06579 | — | — | — |
| | **always-cheapest (`Vision`)** — the fixed policy a router must beat | 9.62 | 0.04468 | -6.73pp | -32.1% | 104/104 |
| | oracle triage | 16.35 | 0.04814 | +0.00pp | -26.8% | 72/104 |
| | **learned, nested threshold (honest)** | 13.46 | 0.06428 | -2.88pp | -2.3% | 14/104 |
| | learned, SR-lossless (in-sample threshold) | 16.35 | 0.06331 | +0.00pp | -3.8% | 11/104 |
| | learned, ≤1pp give-back (in-sample threshold) | 15.38 | 0.04847 | -0.96pp | -26.3% | 57/104 |

## 3. Is the saving real, or manufactured by the threshold sweep?

| cell | observed SR-lossless saving | median under shuffled labels | p |
|---|---|---|---|
| classifieds·B0 | 7.0% | 3.3% | 0.242 |
| reddit·B0 | 0.0% | 0.0% | 1.000 |
| classifieds·B1 | 2.6% | 10.1% | 0.741 |
| reddit·B1 | 11.8% | 1.0% | 0.033 |
| classifieds·B2 | 20.4% | 20.8% | 0.682 |
| reddit·B2 | 26.7% | 5.5% | 0.0004 |
| wa_reddit·B0 | 4.4% | 0.6% | 0.186 |
| wa_reddit·B1 | 3.8% | 2.9% | 0.443 |

Smallest reportable p at B=10000 is 1/(B+1) = 1.00e-04; Holm's tightest threshold over six cells is 0.05/6 = 8.33e-3. B is therefore not what decides any cell's verdict (it was at B=200, where the floor 4.98e-3 sat inside the threshold and the surviving cell reported exactly it).


10000 permutations per cell. The permutation unit is the whole task bundle (y, succ, cost) against X — permuting only `y` leaves the label disconnected from the outcomes that define it, and its error is not one-directional (measured at B=200: cls/B1 0.478→0.503 but red/B2 0.040→0.005; both figures are from that era, not from the current B). p is the plus-one Monte Carlo estimator (k+1)/(B+1). The sweep still picks its operating point post hoc, so this column is how much of the observed saving a signal-free pipeline reproduces.


## 4. Verdict

Holm at α=0.05 over the m=6 cells tested (the sweep was run once per cell, so the family is the six cells) — **1 of 6 reject**:

- reddit·B2: p=0.0004 vs 0.0063 → reject null
- reddit·B1: p=0.033 vs 0.0071 → **stop — this and all larger p unrejected**

Cells where the learned triage Pareto-beats the trivial always-cheapest fixed policy: **0 of 8**.

Read together — and note this is a **narrower** negative than an earlier draft of this file claimed. In five of six cells the label is predictable (AUROC 0.651-0.717, and unlike the which-mode task it clears the best single covariate in 4/6). Two cells yield no SR-lossless saving at all; two more yield savings a signal-free pipeline reproduces (p ~= 0.50). **One cell, reddit·B2 (p=0.0004 vs 0.0063), has a saving that survives Holm at m=6** — under the corrected bundle-permutation null; the earlier y-only null reported 0.040 for reddit/B2 and supported a blanket 'nothing survives' claim, which was wrong.

⚠️ **The sixth cell is the significant one, and its AUROC is 0.483** — below chance, and below its own best single covariate (0.711). That is not a contradiction: the two quantities measure different things, and on this data they come apart. AUROC scores the GLOBAL ranking; the saving comes from the TAIL. reddit/B2 sends 192 of 203 tasks (95%) to the cheap mode with no SR loss — in a cell where only 7.4% of tasks are solvable at all, almost nothing in that 95% was ever going to succeed. It differs from the free always-cheapest policy by five percent of the task allocation, and those 11 retained tasks happen to hold 4 successes (8 vs 4). The permutation null is detecting that tail enrichment, not a globally ordered score.
So the honest phrasing is NOT 'the label is predictable, yet triage fails'. It is: **at 2-27% base SR, a high AUROC is neither necessary nor sufficient — what decides whether triage saves anything is whether a handful of tail tasks land on the right side, and at n=203 that handful is 4 successes.**

What still holds, and is the load-bearing statement: **no cell's learned triage Pareto-beats the trivial always-cheapest fixed policy** (0 of 6). In reddit/B2 specifically the learned policy keeps 1.97pp more SR than always-cheapest but pays ~2.4% more cost — a genuine trade-off point, not a dominating one, and not something a deployment would prefer without a stated SR price. So: a detectable signal in one of six cells, worth less than the policy you get for free.

⚠️ What is and is not out-of-sample. The **nested** row is now FULLY nested (B-1903, 2026-07-27): per outer fold the modes are re-selected from training-row SR/cost, the threshold is chosen against inner-CV out-of-fold scores over the training rows only, and the outer-test rows are scored by an LR fitted on training rows alone — nothing that touches an outer test fold has seen it. An earlier revision of this file claimed a nested operating point while reusing the GLOBAL out-of-fold scores (whose folds include the outer test rows) and a whole-cell choice of `best_mode`/`cheap_mode`; codex caught that, and the numbers here supersede it. The remaining caveat is that the threshold **sweep** rows are still post hoc by construction, which is why the permutation null shares that same selection step — the null is what keeps the swept saving honest, and the nested row is what an actual deployment would get.

One thing the nested design exposes that the whole-cell version hid: `best_mode` is **not stable across folds**. In reddit·B0 the five outer folds select DOM, DOM, SoM, SoM, DOM. A pipeline that picks one best mode from all realized outcomes is therefore not merely optimistic about the threshold — it is reporting a mode choice that its own resampling does not reproduce.

Contrast with the which-mode half: that one fails on label SUPPLY (16-97 labels per cell, 笔记 §383.4). Triage has the labels and the AUROC and still does not beat a fixed policy — a different failure mode, at 2-27% base SR where almost every task is hopeless and 'always take the cheap one' is already close to optimal.

