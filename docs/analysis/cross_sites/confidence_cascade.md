---
type: analysis
status: complete
created: 2026-08-01
purpose: post-hoc two-tier cascade escalating vision -> som on the cheap run's own decoder confidence; the one routing formulation the paper never tested
scope_warning: cost is within-cell only (B0 = API bill, B1/B2 = electricity-derived). The escalation threshold is swept, NOT selected out-of-fold, so every operating point below is in-sample and is an UPPER bound on deployable performance.
producer: scripts/analysis/aggregate_confidence_cascade.py
---

# Confidence-triggered cascade

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_confidence_cascade.py`

Cheap tier = **vision**, rich tier = **som** — fixed **a priori** from the six-cell cost ordering in `multimetric_pareto`, not chosen per cell (choosing per cell would make the cells incomparable).

⚠️ **The tiers are not cost-ordered in every cell.** On `wa_B0` the cheapest mode is `dom`, not `vision`. On that cell this is still a fixed-pair escalation and the SR arithmetic is unaffected, but it is not a cheap→rich escalation in the cost sense, and its cost column should not be read as one.

The escalation decision sees only the cheap run's own episode — no outcome, no rich-run information. Two nulls accompany every point: **random** escalates the same number of tasks signal-free (exact expectation, not sampled), and **oracle** escalates exactly the tasks the rich mode would fix.

## 1. Endpoints

Cost is relative to running the cheap mode on every task.

| cell | n | cheap SR | **always-rich SR / cost** | oracle SR | oracle escalates | oracle cost |
|---|---|---|---|---|---|---|
| `cls_B0` | 224 | 25.00% | **27.23% / 1.12x** | 33.93% | 20 tasks | 1.06x |
| `cls_B1` | 224 | 12.50% | **14.29% / 1.40x** | 19.20% | 15 tasks | 1.07x |
| `cls_B2` | 224 | 2.23% | **2.23% / 1.28x** | 4.46% | 5 tasks | 1.02x |
| `red_B0` | 203 | 7.39% | **14.78% / 1.13x** | 18.23% | 22 tasks | 1.10x |
| `red_B1` | 203 | 2.46% | **7.39% / 1.53x** | 8.37% | 12 tasks | 1.12x |
| `red_B2` | 203 | 1.97% | **0.99% / 1.63x** | 2.96% | 2 tasks | 1.02x |

The **oracle cascade is the attractive operating point in this table**: it pays double only on the 2–22 tasks that need it, so it buys +2.2 to +10.8pp for +2% to +12% cost. Everything below asks how much of that a deployable signal recovers.

> ⚠️ **Every number below is an offline splice.** An escalated task takes its outcome from a standalone rich-mode run, but a real cascade would start the rich episode *after* the cheap one had already acted on a stateful site. That sequential outcome is unobserved in this project, so the bias can run either way — this is a limitation of the design, not of the estimator.

## 1b. THE VERDICT — does any operating point Pareto-beat *always-rich*?

Always running the rich mode is a fixed policy: no signal, no threshold, no fitting. A cascade that does not beat it on both axes has bought nothing.

| cell | always-rich SR / cost | operating points that Pareto-beat it |
|---|---|---|
| `cls_B0` | 27.23% / 1.12x | **none** |
| `cls_B1` | 14.29% / 1.40x | **none** |
| `cls_B2` | 2.23% / 1.28x | `mean_logprob_mean`@5%, `mean_logprob_mean`@10%, `mean_logprob_mean`@15%, `mean_logprob_mean`@20% · ⚠️ rich mode is *worse than or equal to* cheap here, so the cascade question is moot |
| `red_B0` | 14.78% / 1.13x | **none** |
| `red_B1` | 7.39% / 1.53x | **none** |
| `red_B2` | 0.99% / 1.63x | `mean_logprob_mean`@5%, `mean_logprob_mean`@10%, `mean_logprob_mean`@15%, `mean_logprob_mean`@20% · ⚠️ rich mode is *worse than or equal to* cheap here, so the cascade question is moot |

⚠️ **Read the cell count, not the combination count.** Of 6 cells, **4 pose the cascade question at all** — in the other 2 the rich mode is no better than the cheap one, so there is nothing to escalate *to* and any 'win' is an artefact of that. Among the comparable cells, **0 show Pareto-beating operating points**.

For completeness the raw search tally is **79 of 495 (cell, signal, operating point) combinations, in 2 of 6 cells** — but 2 of those 2 cells are the degenerate ones just named. `frac=0` is excluded throughout — it is the always-cheap fixed policy, not a cascade. The denominator counts only signals a cell can actually rank with; where a signal was dropped for having no variance it is not part of the search space.

## 1c. Fraction of the oracle's headroom the best signal recovers

| cell | 10% | 20% | 30% |
|---|---|---|---|
| `cls_B0` | 15% | 25% | 30% |
| `cls_B1` | 20% | 27% | 40% |
| `cls_B2` | 20% | 40% | 20% |
| `red_B0` | 23% | 27% | 27% |
| `red_B1` | 33% | 50% | 50% |
| `red_B2` | 0% | 0% | 0% |

## 2. Does the confidence signal beat a signal-free escalation of the same size?

For each cell, the best signal at each escalation fraction, and the margin over the random-escalation expectation. **A positive margin is the entire claim** — without it the cascade is just paying more.

| cell | frac | best signal | SR | gain vs cheap | random gain | **margin** |
|---|---|---|---|---|---|---|
| `cls_B0` | 10% | `mean_logprob_mean` | 26.34% | +1.34pp | +0.22pp | **+1.12pp** ✅ |
| `cls_B0` | 20% | `neg_steps` | 27.23% | +2.23pp | +0.45pp | **+1.78pp** ✅ |
| `cls_B0` | 30% | `neg_steps` | 27.68% | +2.68pp | +0.67pp | **+2.01pp** ✅ |
| `cls_B1` | 10% | `min_margin_min` | 13.84% | +1.34pp | +0.18pp | **+1.16pp** ✅ |
| `cls_B1` | 20% | `neg_steps` | 14.29% | +1.79pp | +0.36pp | **+1.43pp** ✅ |
| `cls_B1` | 30% | `neg_steps` | 15.18% | +2.68pp | +0.53pp | **+2.14pp** ✅ |
| `cls_B2` | 10% | `mean_logprob_mean` | 2.68% | +0.45pp | +0.00pp | **+0.45pp** ✅ |
| `cls_B2` | 20% | `neg_noop_rate` | 3.12% | +0.89pp | +0.00pp | **+0.89pp** ✅ |
| `cls_B2` | 30% | `mean_logprob_min` | 2.68% | +0.45pp | +0.00pp | **+0.45pp** ✅ |
| `red_B0` | 10% | `mean_logprob_min` | 9.85% | +2.46pp | +0.73pp | **+1.74pp** ✅ |
| `red_B0` | 20% | `mean_logprob_min` | 10.34% | +2.96pp | +1.49pp | **+1.46pp** ✅ |
| `red_B0` | 30% | `mean_logprob_mean` | 10.34% | +2.96pp | +2.22pp | **+0.74pp** ✅ |
| `red_B1` | 10% | `min_logprob_min` | 4.43% | +1.97pp | +0.49pp | **+1.49pp** ✅ |
| `red_B1` | 20% | `min_logprob_min` | 5.42% | +2.96pp | +0.99pp | **+1.96pp** ✅ |
| `red_B1` | 30% | `mean_logprob_min` | 5.42% | +2.96pp | +1.48pp | **+1.48pp** ✅ |
| `red_B2` | 10% | `mean_logprob_mean` | 1.97% | +0.00pp | -0.10pp | **+0.10pp** ✅ |
| `red_B2` | 20% | `mean_logprob_min` | 1.97% | +0.00pp | -0.20pp | **+0.20pp** ✅ |
| `red_B2` | 30% | `mean_margin_mean` | 1.97% | +0.00pp | -0.30pp | **+0.30pp** ✅ |

⚠️ The best signal is picked per (cell, fraction) from 10 candidates against realised outcomes, so these margins are in-sample maxima over a signal menu. Treat them as an upper bound on what an out-of-fold selection could deliver.

## 3. Per-signal margin over random, averaged across cells

| signal | 10% | 20% | 30% |
|---|---|---|---|
| `mean_logprob_mean` | +0.29pp | +0.27pp | +0.26pp |
| `mean_logprob_min` | +0.16pp | +0.39pp | +0.35pp |
| `min_logprob_min` | +0.32pp | +0.36pp | -0.04pp |
| `mean_margin_mean` | +0.05pp | +0.11pp | -0.13pp |
| `min_margin_min` ⚠️ 5/6 cells | +0.53pp | +0.49pp | +0.07pp |
| `verbalized_mean` ⚠️ 4/6 cells | +0.03pp | +0.30pp | +0.79pp |
| `verbalized_min` ⚠️ 4/6 cells | +0.17pp | +0.30pp | -0.02pp |
| `neg_steps` | +0.53pp | +0.97pp | +0.95pp |
| `neg_noop_rate` | +0.22pp | +0.50pp | +0.56pp |
| `neg_actfail_rate` | +0.29pp | +0.73pp | +0.65pp |

**Signals dropped before ranking** — a score with no variance cannot rank anything, and `sorted()` then falls through to task id, so the resulting "operating point" is a set of task ids wearing a threshold's name:
- `cls_B0` / `verbalized_mean`: not populated on 2/224 episodes
- `cls_B0` / `verbalized_min`: not populated on 2/224 episodes
- `red_B1` / `verbalized_mean`: not populated on 1/203 episodes
- `red_B1` / `verbalized_min`: not populated on 1/203 episodes
- `red_B2` / `min_margin_min`: no variance: all 203 episodes share the value 0.0, so ranking falls through to task id

## 4. Full curves

### `cls_B0` (n=224, cheap 25.00%, oracle 33.93%)

| frac | k | SR | cost | SR gain | random gain | margin |
|---|---|---|---|---|---|---|
| 0% | 0 | 25.00% | 1.00x | +0.00pp | +0.00pp | +0.00pp |
| 5% | 11 | 25.45% | 1.10x | +0.45pp | +0.11pp | +0.34pp |
| 10% | 22 | 26.34% | 1.19x | +1.34pp | +0.22pp | +1.12pp |
| 15% | 34 | 26.79% | 1.29x | +1.79pp | +0.34pp | +1.45pp |
| 20% | 45 | 27.23% | 1.36x | +2.23pp | +0.45pp | +1.78pp |
| 30% | 67 | 27.68% | 1.55x | +2.68pp | +0.67pp | +2.01pp |
| 40% | 90 | 28.12% | 1.68x | +3.12pp | +0.90pp | +2.23pp |
| 50% | 112 | 28.57% | 1.81x | +3.57pp | +1.12pp | +2.46pp |
| 75% | 168 | 28.12% | 2.02x | +3.12pp | +1.67pp | +1.45pp |
| 100% | 224 | 27.23% | 2.12x | +2.23pp | +2.23pp | +0.00pp |

_Signal shown: `neg_steps` (best at 20% for this cell)._

### `cls_B1` (n=224, cheap 12.50%, oracle 19.20%)

| frac | k | SR | cost | SR gain | random gain | margin |
|---|---|---|---|---|---|---|
| 0% | 0 | 12.50% | 1.00x | +0.00pp | +0.00pp | +0.00pp |
| 5% | 11 | 13.39% | 1.08x | +0.89pp | +0.09pp | +0.81pp |
| 10% | 22 | 13.39% | 1.17x | +0.89pp | +0.18pp | +0.72pp |
| 15% | 34 | 14.29% | 1.25x | +1.79pp | +0.27pp | +1.51pp |
| 20% | 45 | 14.29% | 1.35x | +1.79pp | +0.36pp | +1.43pp |
| 30% | 67 | 15.18% | 1.51x | +2.68pp | +0.53pp | +2.14pp |
| 40% | 90 | 14.73% | 1.69x | +2.23pp | +0.72pp | +1.51pp |
| 50% | 112 | 15.18% | 1.90x | +2.68pp | +0.89pp | +1.79pp |
| 75% | 168 | 15.62% | 2.21x | +3.12pp | +1.34pp | +1.79pp |
| 100% | 224 | 14.29% | 2.40x | +1.79pp | +1.79pp | +0.00pp |

_Signal shown: `neg_steps` (best at 20% for this cell)._

### `cls_B2` (n=224, cheap 2.23%, oracle 4.46%)

| frac | k | SR | cost | SR gain | random gain | margin |
|---|---|---|---|---|---|---|
| 0% | 0 | 2.23% | 1.00x | +0.00pp | +0.00pp | +0.00pp |
| 5% | 11 | 2.23% | 1.06x | +0.00pp | +0.00pp | +0.00pp |
| 10% | 22 | 2.23% | 1.12x | +0.00pp | +0.00pp | +0.00pp |
| 15% | 34 | 2.68% | 1.19x | +0.45pp | +0.00pp | +0.45pp |
| 20% | 45 | 3.12% | 1.27x | +0.89pp | +0.00pp | +0.89pp |
| 30% | 67 | 2.68% | 1.38x | +0.45pp | +0.00pp | +0.45pp |
| 40% | 90 | 2.68% | 1.51x | +0.45pp | +0.00pp | +0.45pp |
| 50% | 112 | 2.68% | 1.64x | +0.45pp | +0.00pp | +0.45pp |
| 75% | 168 | 3.12% | 1.97x | +0.89pp | +0.00pp | +0.89pp |
| 100% | 224 | 2.23% | 2.28x | +0.00pp | +0.00pp | +0.00pp |

_Signal shown: `neg_noop_rate` (best at 20% for this cell)._

### `red_B0` (n=203, cheap 7.39%, oracle 18.23%)

| frac | k | SR | cost | SR gain | random gain | margin |
|---|---|---|---|---|---|---|
| 0% | 0 | 7.39% | 1.00x | +0.00pp | +0.00pp | +0.00pp |
| 5% | 10 | 8.37% | 1.05x | +0.99pp | +0.36pp | +0.62pp |
| 10% | 20 | 9.85% | 1.11x | +2.46pp | +0.73pp | +1.74pp |
| 15% | 30 | 9.85% | 1.17x | +2.46pp | +1.09pp | +1.37pp |
| 20% | 41 | 10.34% | 1.24x | +2.96pp | +1.49pp | +1.46pp |
| 30% | 61 | 10.34% | 1.36x | +2.96pp | +2.22pp | +0.74pp |
| 40% | 81 | 11.33% | 1.46x | +3.94pp | +2.95pp | +0.99pp |
| 50% | 102 | 12.32% | 1.59x | +4.93pp | +3.71pp | +1.21pp |
| 75% | 152 | 14.29% | 1.84x | +6.90pp | +5.53pp | +1.36pp |
| 100% | 203 | 14.78% | 2.13x | +7.39pp | +7.39pp | +0.00pp |

_Signal shown: `mean_logprob_min` (best at 20% for this cell)._

### `red_B1` (n=203, cheap 2.46%, oracle 8.37%)

| frac | k | SR | cost | SR gain | random gain | margin |
|---|---|---|---|---|---|---|
| 0% | 0 | 2.46% | 1.00x | +0.00pp | +0.00pp | +0.00pp |
| 5% | 10 | 2.46% | 1.07x | +0.00pp | +0.24pp | -0.24pp |
| 10% | 20 | 4.43% | 1.18x | +1.97pp | +0.49pp | +1.49pp |
| 15% | 30 | 4.43% | 1.27x | +1.97pp | +0.73pp | +1.24pp |
| 20% | 41 | 5.42% | 1.37x | +2.96pp | +0.99pp | +1.96pp |
| 30% | 61 | 5.42% | 1.55x | +2.96pp | +1.48pp | +1.48pp |
| 40% | 81 | 5.91% | 1.72x | +3.45pp | +1.97pp | +1.48pp |
| 50% | 102 | 5.91% | 1.87x | +3.45pp | +2.48pp | +0.97pp |
| 75% | 152 | 7.39% | 2.23x | +4.93pp | +3.69pp | +1.24pp |
| 100% | 203 | 7.39% | 2.53x | +4.93pp | +4.93pp | +0.00pp |

_Signal shown: `min_logprob_min` (best at 20% for this cell)._

### `red_B2` (n=203, cheap 1.97%, oracle 2.96%)

| frac | k | SR | cost | SR gain | random gain | margin |
|---|---|---|---|---|---|---|
| 0% | 0 | 1.97% | 1.00x | +0.00pp | +0.00pp | +0.00pp |
| 5% | 10 | 1.97% | 1.05x | +0.00pp | -0.05pp | +0.05pp |
| 10% | 20 | 1.97% | 1.13x | +0.00pp | -0.10pp | +0.10pp |
| 15% | 30 | 1.97% | 1.23x | +0.00pp | -0.15pp | +0.15pp |
| 20% | 41 | 1.97% | 1.30x | +0.00pp | -0.20pp | +0.20pp |
| 30% | 61 | 0.99% | 1.47x | -0.99pp | -0.30pp | -0.69pp |
| 40% | 81 | 0.99% | 1.64x | -0.99pp | -0.39pp | -0.59pp |
| 50% | 102 | 0.49% | 1.82x | -1.48pp | -0.50pp | -0.98pp |
| 75% | 152 | 0.99% | 2.25x | -0.99pp | -0.74pp | -0.25pp |
| 100% | 203 | 0.99% | 2.63x | -0.99pp | -0.99pp | +0.00pp |

_Signal shown: `mean_logprob_min` (best at 20% for this cell)._

