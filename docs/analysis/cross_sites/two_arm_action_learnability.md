# Is the ACTION half of routing learnable?

`post_hoc_exploratory=True`, `h10_eligible=False`. Task-held-out 5-fold CV per cell, seed 42, L2 LR on the 18 raw features (same feature set as `router_triage_learnability.py`), fold-local standardisation.

Label z: oracle's per-task pick between {best_mode, cheap_mode} — 1 iff best_mode succeeds AND cheap_mode fails, else 0 (both-succeed/both-fail default to cheap_mode, not a realized-cost tie-break — see `derive_two_arm_label`). Defined for EVERY task, but its POSITIVE class is thin: 2.2-14.4% base rate, comparable to or thinner than the six-way which-mode label's per-cell supply (笔记 §490.4).


## 1. Per-task vs per-class assignment over the same two arms

| cell | n | oracle_triage SR/cost | oracle_two_arm SR/cost | ΔSR | Δcost | ΔSR vs rerun band |
|---|---|---|---|---|---|---|
| classifieds·B0 | 224 | 27.23/0.06312 | 33.93/0.06257 | +6.70pp | -0.9% | clear |
| reddit·B0 | 203 | 14.78/0.09998 | 18.23/0.09719 | +3.45pp | -2.8% | clear |
| classifieds·B1 | 224 | 14.29/0.04858 | 19.20/0.04284 | +4.91pp | -11.8% | clear |
| reddit·B1 | 203 | 7.39/0.05554 | 8.37/0.05503 | +0.99pp | -0.9% | **within band** |
| classifieds·B2 | 224 | 2.23/0.07145 | 4.46/0.07037 | +2.23pp | -1.5% | **knife-edge** |
| reddit·B2 | 203 | 3.94/0.06974 | 5.91/0.06919 | +1.97pp | -0.8% | **within band** |
| wa_reddit·B0 | 104 | 35.58/0.07573 | 41.35/0.07100 | +5.77pp | -6.2% | clear |
| wa_reddit·B1 | 104 | 16.35/0.04814 | 20.19/0.04498 | +3.85pp | -6.6% | clear |

**Two statements, and they are not the same statement.** (a) *Guaranteed by construction*: SR(two_arm) ≥ SR(triage) in every cell — any success the class-level pick achieves is reproducible by the per-task pick. (b) *Measured on these logs, not guaranteed*: the per-task pick is also cheaper, giving full Pareto dominance in 8 of 8 cells. **Cost dominance is NOT implied by the construction** — with a strong-but-dear arm and a weak-but-cheap one, recovering a success by switching arms can cost more, not less (worked counterexample in `evaluate()`). It holds here because a failed episode on these benchmarks usually burns the whole step budget.

**Of the 8 cells, 6 carry a ΔSR above the 2.23pp rerun band**; the rest move by less than re-running one unchanged condition would. The direction is safe in all 8 (it is guaranteed), but the magnitude is only separable from noise in 5--6. ⚠️ **classifieds·B2 clears it by only 0.0021pp** — less than one task's worth of SR at this n, i.e. inside the rounding of the threshold itself. Counting it as cleared is an artefact; the defensible count is 5.


### 1b. Is the selected pair even on the two-arm frontier?

| cell | selected (best, cheap) | its oracle SR/cost | dominating pairs | best alternative |
|---|---|---|---|---|
| classifieds·B0 | (SoM, Vision) | 33.93/0.06257 | 1/30 | (P-prompt, Vision) 36.16/0.06221 |
| reddit·B0 | (SoM, Vision) | 18.23/0.09719 | 1/30 | (DOM, Vision) 18.72/0.09650 |
| classifieds·B1 | (SoM, Vision) | 19.20/0.04284 | 0/30 | — |
| reddit·B1 | (SoM, Vision) | 8.37/0.05503 | 0/30 | — |
| classifieds·B2 | (SoM, Vision) | 4.46/0.07037 | 0/30 | — |
| reddit·B2 | (DOM, Vision) | 5.91/0.06919 | 0/30 | — |
| wa_reddit·B0 | (P-text, DOM) | 41.35/0.07100 | 0/30 | — |
| wa_reddit·B1 | (DOM, Vision) | 20.19/0.04498 | 0/30 | — |

⚠️ **The selected pair is dominated by some other pair in 2 of 8 cells.** `best_mode` and `cheap_mode` are two independent extrema, not a jointly optimal pair. That is the correct thing to hold fixed here — it is exactly the action space `oracle_triage` operates over, and the comparison in §1 is only meaningful on the same two arms — but it means **`oracle_two_arm` is a conditional oracle over the triage-selected pair, not a two-arm ceiling.** Any claim of the form 'this is the best a two-arm policy could do' is unsupported.


## 2. Is z more or less predictable than the triage label y?

| cell | n | z base rate % (sent to best_mode) | solvable % (y) | AUROC(z) | AUROC(y) | AUROC(z) best single feat | that feature |
|---|---|---|---|---|---|---|---|
| classifieds·B0 | 224 | 8.9 | 43.3 | **0.405** | 0.683 | 0.676 | `intent_compose` |
| reddit·B0 | 203 | 10.8 | 26.1 | **0.562** | 0.700 | 0.625 | `intent_token_count` |
| classifieds·B1 | 224 | 6.7 | 24.6 | **0.454** | 0.705 | 0.637 | `intent_color` |
| reddit·B1 | 203 | 5.9 | 11.8 | **0.733** | 0.723 | 0.686 | `intent_token_count` |
| classifieds·B2 | 224 | 2.2 | 7.1 | **0.712** | 0.646 | 0.747 | `dom_complexity` |
| reddit·B2 | 203 | 3.9 | 7.4 | **0.821** | 0.526 | 0.837 | `intent_token_count` |
| wa_reddit·B0 | 104 | 14.4 | 51.9 | **0.406** | 0.554 | 0.664 | `intent_token_count` |
| wa_reddit·B1 | 104 | 10.6 | 30.8 | **0.761** | 0.758 | 0.756 | `dom_complexity` |

The z-LR clears its own best single covariate in 2 of 8 cells. AUROC(z) < AUROC(y) in 4 of 8, and falls below chance in 3.

⚠️ **Do not read this as 'z is harder because its positive class is smaller'.** AUROC is a ranking statistic and is insensitive to class balance in expectation; a thin positive class inflates the VARIANCE of the estimate (5-22 positives per cell), it does not depress its expected value. What a low AUROC(z) says is that these features do not separate the specific margin z encodes — *this task needs the dear arm and the cheap one will not do* — which is a strictly finer distinction than y's *something solves this*. Scarcity and feature inadequacy are different diagnoses and this table cannot separate them; report both as open.


## 3. What does the LEARNED (honest, fully nested) two-arm policy buy?

The nested router pays for arm-selection uncertainty out of its own training fold. Comparing it against a whole-cell always-cheapest — which was handed the min-cost arm by full-data hindsight — puts the two on different information sets. The Δ columns below are therefore against the **cross-fitted** comparator: always-cheapest evaluated on the same outer-test rows with the same fold-local arm. Whole-cell rows are kept as descriptive oracles, marked *(whole-cell)*.

| cell | policy | SR % | mean cost | ΔSR vs xfit-cheap | Δcost vs xfit-cheap | sent to best_mode |
|---|---|---|---|---|---|---|
| classifieds·B0 | **always-cheapest (cross-fitted)** | 25.00 | 0.06481 | ref. | ref. | 0/224 |
| | best-single (cross-fitted) | 27.23 | 0.07236 | +2.23pp | +11.6% | 224/224 |
| | **learned two-arm, nested (honest)** | 25.00 | 0.06507 | +0.00pp | +0.4% | 9/224 sent best |
| | oracle triage *(whole-cell)* | 27.23 | 0.06312 | +2.23pp | -2.6% | 127/224 sent cheap |
| | oracle two-arm, conditional on the pair *(whole-cell)* | 33.93 | 0.06257 | +8.93pp | -3.5% | 204/224 sent cheap |
| reddit·B0 | **always-cheapest (cross-fitted)** | 7.39 | 0.09807 | ref. | ref. | 0/203 |
| | best-single (cross-fitted) | 11.33 | 0.10171 | +3.94pp | +3.7% | 203/203 |
| | **learned two-arm, nested (honest)** | 9.36 | 0.10042 | +1.97pp | +2.4% | 45/203 sent best |
| | oracle triage *(whole-cell)* | 14.78 | 0.09998 | +7.39pp | +1.9% | 150/203 sent cheap |
| | oracle two-arm, conditional on the pair *(whole-cell)* | 18.23 | 0.09719 | +10.84pp | -0.9% | 181/203 sent cheap |
| classifieds·B1 | **always-cheapest (cross-fitted)** | 12.50 | 0.04316 | ref. | ref. | 0/224 |
| | best-single (cross-fitted) | 14.29 | 0.06028 | +1.79pp | +39.7% | 224/224 |
| | **learned two-arm, nested (honest)** | 12.50 | 0.04316 | +0.00pp | +0.0% | 0/224 sent best |
| | oracle triage *(whole-cell)* | 14.29 | 0.04858 | +1.79pp | +12.6% | 169/224 sent cheap |
| | oracle two-arm, conditional on the pair *(whole-cell)* | 19.20 | 0.04284 | +6.70pp | -0.7% | 209/224 sent cheap |
| reddit·B1 | **always-cheapest (cross-fitted)** | 2.46 | 0.05240 | ref. | ref. | 0/203 |
| | best-single (cross-fitted) | 7.39 | 0.08000 | +4.93pp | +52.7% | 203/203 |
| | **learned two-arm, nested (honest)** | 2.46 | 0.05240 | +0.00pp | +0.0% | 0/203 sent best |
| | oracle triage *(whole-cell)* | 7.39 | 0.05554 | +4.93pp | +6.0% | 179/203 sent cheap |
| | oracle two-arm, conditional on the pair *(whole-cell)* | 8.37 | 0.05503 | +5.91pp | +5.0% | 191/203 sent cheap |
| classifieds·B2 | **always-cheapest (cross-fitted)** | 2.23 | 0.07065 | ref. | ref. | 0/224 |
| | best-single (cross-fitted) | 0.89 | 0.08333 | -1.34pp | +17.9% | 224/224 |
| | **learned two-arm, nested (honest)** | 2.23 | 0.07057 | +0.00pp | -0.1% | 47/224 sent best |
| | oracle triage *(whole-cell)* | 2.23 | 0.07145 | +0.00pp | +1.1% | 208/224 sent cheap |
| | oracle two-arm, conditional on the pair *(whole-cell)* | 4.46 | 0.07037 | +2.23pp | -0.4% | 219/224 sent cheap |
| reddit·B2 | **always-cheapest (cross-fitted)** | 1.97 | 0.06833 | ref. | ref. | 0/203 |
| | best-single (cross-fitted) | 3.94 | 0.09479 | +1.97pp | +38.7% | 203/203 |
| | **learned two-arm, nested (honest)** | 1.97 | 0.06833 | +0.00pp | +0.0% | 0/203 sent best |
| | oracle triage *(whole-cell)* | 3.94 | 0.06974 | +1.97pp | +2.1% | 188/203 sent cheap |
| | oracle two-arm, conditional on the pair *(whole-cell)* | 5.91 | 0.06919 | +3.94pp | +1.3% | 195/203 sent cheap |
| wa_reddit·B0 | **always-cheapest (cross-fitted)** | 26.92 | 0.07860 | ref. | ref. | 0/104 |
| | best-single (cross-fitted) | 35.58 | 0.08478 | +8.65pp | +7.9% | 104/104 |
| | **learned two-arm, nested (honest)** | 25.96 | 0.07908 | -0.96pp | +0.6% | 4/104 sent best |
| | oracle triage *(whole-cell)* | 35.58 | 0.07573 | +8.65pp | +0.6% | 50/104 sent cheap |
| | oracle two-arm, conditional on the pair *(whole-cell)* | 41.35 | 0.07100 | +14.42pp | -5.7% | 89/104 sent cheap |
| wa_reddit·B1 | **always-cheapest (cross-fitted)** | 9.62 | 0.04468 | ref. | ref. | 0/104 |
| | best-single (cross-fitted) | 13.46 | 0.06752 | +3.85pp | +51.1% | 104/104 |
| | **learned two-arm, nested (honest)** | 12.50 | 0.04486 | +2.88pp | +0.4% | 5/104 sent best |
| | oracle triage *(whole-cell)* | 16.35 | 0.04814 | +6.73pp | +7.7% | 72/104 sent cheap |
| | oracle two-arm, conditional on the pair *(whole-cell)* | 20.19 | 0.04498 | +10.58pp | +0.7% | 93/104 sent cheap |

**Learned nested two-arm policy Pareto-beats the cross-fitted always-cheapest in 1 of 8 cells** (against the whole-cell comparator it is 3 of 8). The comparator choice moves the effect size, not the verdict.


⚠️ Read that 1 closely. **classifieds·B2**: +0.00pp SR and -0.12% cost — an identical success rate and a cost difference of $0.000087 per episode, on a cell whose base success is 2.23% (5 of 224 tasks). It satisfies the Pareto definition and it is not a result anyone would deploy on.


## 4. Is the sweep saving real, or manufactured?

| cell | observed SR-floor-preserving saving vs always-cheap | median under shuffled z | p |
|---|---|---|---|
| classifieds·B0 | 1.1% | 0.0% | 0.099 |
| reddit·B0 | 0.0% | 0.0% | 1.000 |
| classifieds·B1 | -0.0% | 0.0% | 1.000 |
| reddit·B1 | 0.0% | 0.0% | 1.000 |
| classifieds·B2 | -0.0% | 0.0% | 0.999 |
| reddit·B2 | 0.0% | 0.0% | 1.000 |
| wa_reddit·B0 | -0.0% | 0.0% | 1.000 |
| wa_reddit·B1 | 2.0% | 0.0% | 0.003 |

Holm at α=0.05 over m=8 cells — **1 of 8 reject**:

- wa_reddit·B1: p=0.003 vs 0.0063 → reject null
- classifieds·B0: p=0.099 vs 0.0071 → **stop — this and all larger p unrejected**
