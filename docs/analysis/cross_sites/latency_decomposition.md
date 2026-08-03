---
type: analysis
status: complete
purpose: how much of the reported latency is the model and how much is the environment
post_hoc_exploratory: true
scope_warning: this is a validity audit of the latency estimand, not a new behavioural metric. It deliberately does not enter per_mode_four_dimension_profile, because adding a 27th metric moves the >=7/8 consistency denominators claim 6 rests on.
producer: scripts/analysis/latency_decomposition.py
---

# What is inside a latency number

Regenerate: `.venv/bin/python3 scripts/analysis/latency_decomposition.py`

Every latency figure in this project is `latency_ms.total`, or the canonical estimand, which subtracts retry / busy-wait / recovered-screenshot time from it. Neither subtracts the environment. `latency_ms.backend_infer` isolates the model call; it has been written on 100% of steps since the beginning and **was read by no analysis script until 2026-08-03**.

## 1. The split, per cell

| cell | mean total (ms) | mean model call (ms) | model share | obs prepare (ms) | runtime sleep (ms) |
|---|---|---|---|---|---|
| `B0·classifieds` | 7,622 | 2,140 | **28.1%** | 2.9 | 44 |
| `B0·reddit` | 24,601 | 5,603 | **22.8%** | 3.8 | 17 |
| `B0·wa_reddit` | 16,112 | 3,551 | **22.0%** | 3.8 | 46 |
| `B1·classifieds` | 14,180 | 8,213 | **57.9%** | 3.1 | 69 |
| `B1·reddit` | 24,376 | 7,916 | **32.5%** | 3.8 | 17 |
| `B1·wa_reddit` | 21,221 | 7,747 | **36.5%** | 4.1 | 59 |
| `B2·classifieds` | 14,739 | 9,908 | **67.2%** | 3.0 | 78 |
| `B2·reddit` | 22,863 | 9,131 | **39.9%** | 4.2 | 27 |

**The model is 22–67% of the time we report.** The remainder is the browser and the container. `offsite_navigation_audit` already measured the reddit container at 1.69x the classifieds one before any agent behaviour enters, so that remainder is not a constant either.

## 2. Does the fastest mode change when the container is removed?

| cell | fastest by total | fastest by model call alone | same? |
|---|---|---|---|
| `B0·classifieds` | P-prompt | P-prompt | yes |
| `B0·reddit` | Vision | P-SoM | **no** |
| `B0·wa_reddit` | Vision | P-prompt | **no** |
| `B1·classifieds` | Vision | Vision | yes |
| `B1·reddit` | Vision | P-text | **no** |
| `B1·wa_reddit` | P-prompt | Vision | **no** |
| `B2·classifieds` | P-prompt | P-prompt | yes |
| `B2·reddit` | Vision | Vision | yes |

⚠️ **The fastest mode changes in 4 of 8 cells** (`B0·reddit`, `B0·wa_reddit`, `B1·reddit`, `B1·wa_reddit`). **They are not scattered: 4 of 5 reddit-family cells flip and 0 of 3 classifieds cells do** — i.e. the flips land exactly where the container is slowest and the model is the smallest share of the step. That is the pattern a container effect produces, not the pattern noise produces, and it is why this is reported as an estimand problem rather than as a finding about modes. Any sentence naming *which* mode is fastest is therefore a statement about this deployment's browser and container as much as about the mode. The claim that survives is the weaker, estimand-independent one: **cost ordering and latency ordering are not the same ordering** — that is a statement about two rankings disagreeing, and it does not depend on which latency you rank by.

## 3. Two things this rules out

- **The SoM annotation step is not the cost.** `obs_prepare` — the marking pass that turns a screenshot into a numbered one — runs at 15–21 ms on SoM arms and ~0.1 ms elsewhere. Against a 6,000–37,000 ms step it is nothing. If SoM is slower it is because of what it makes the agent *do*, not because annotating costs time.
- **Runtime sleeps are not the cost either.** They run 0–211 ms per step, i.e. under 2% of a step even at their worst.

## 4. What this does not license

Model-only latency is **not** a better estimand for a deployment claim — a user waits for the whole step, container included. It is the right estimand for a claim about *the representation*, and the wrong one for a claim about *the system*. Both are reported here so a sentence can pick the one it means, which is the same discipline `outcome_efficiency` applies to the cost denominator.
