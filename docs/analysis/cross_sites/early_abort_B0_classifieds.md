---
type: analysis
status: complete
purpose: whether the first k steps of a trajectory predict that the episode will fail, i.e. whether an abort is learnable
scope_warning: B0 x classifieds only. Every row is CONDITIONAL ON SURVIVING TO STEP k -- episodes shorter than k are excluded, and those are exactly the easy-to-call ones (early success, early hard failure), so the discrimination reported here is on the residual hard subset, not on all episodes.
producer: scripts/analysis/early_abort_learnability.py
---

# Is an abort learnable from the first k steps?

Regenerate: `.venv/bin/python3 scripts/analysis/early_abort_learnability.py --baseline B0 --site classifieds`

The layered-evidence 0b row reports routing AUROC up to 0.877 for the per-step confidence signal, but that is aggregated over the **whole** episode -- for an abort decision at step k it is looking at the future. Everything below uses the first k steps only.

| mode | k | episodes | AUROC | shuffle null | gap | truncate-at-k | learned @ that same loss |
|---|---:|---:|---:|---:|---:|---|---|
| DOM | 3 | 217 | 0.563 | 0.396 | +0.167 | 81.3% steps, −34 succ | 70.5% steps, −34 succ |
| DOM | 5 | 196 | 0.528 | 0.490 | +0.038 | 71.3% steps, −25 succ | 63.2% steps, −25 succ |
| DOM | 10 | 130 | 0.667 | 0.423 | +0.244 | 56.2% steps, −12 succ | 54.0% steps, −12 succ |
| SoM | 3 | 211 | 0.497 | 0.493 | +0.004 | 79.2% steps, −45 succ | 65.4% steps, −45 succ |
| SoM | 5 | 176 | 0.505 | 0.476 | +0.029 | 69.8% steps, −27 succ | 55.6% steps, −27 succ |
| SoM | 10 | 101 | 0.516 | 0.622 | -0.105 | 58.7% steps, −9 succ | 57.4% steps, −9 succ |
| Vision | 3 | 209 | 0.505 | 0.547 | -0.042 | 82.2% steps, −40 succ | 68.2% steps, −40 succ |
| Vision | 5 | 185 | 0.526 | 0.467 | +0.059 | 73.2% steps, −26 succ | 65.7% steps, −25 succ |
| Vision | 10 | 123 | 0.611 | 0.462 | +0.150 | 59.3% steps, −13 succ | 58.7% steps, −13 succ |
| P-text | 3 | 218 | 0.568 | 0.519 | +0.049 | 81.5% steps, −30 succ | 76.1% steps, −29 succ |
| P-text | 5 | 195 | 0.528 | 0.480 | +0.048 | 71.8% steps, −19 succ | 63.0% steps, −19 succ |
| P-text | 10 | 130 | 0.336 | 0.368 | -0.032 | 57.0% steps, −11 succ | 38.2% steps, −11 succ |
| P-prompt | 3 | 216 | 0.448 | 0.474 | -0.026 | 80.6% steps, −40 succ | 68.4% steps, −40 succ |
| P-prompt | 5 | 184 | 0.494 | 0.415 | +0.079 | 71.4% steps, −27 succ | 65.6% steps, −27 succ |
| P-prompt | 10 | 122 | 0.526 | 0.373 | +0.153 | 56.5% steps, −13 succ | 53.3% steps, −13 succ |
| P-SoM | 3 | 215 | 0.355 | 0.522 | -0.167 | 82.2% steps, −29 succ | 70.1% steps, −29 succ |
| P-SoM | 5 | 191 | 0.584 | 0.382 | +0.202 | 73.0% steps, −15 succ | 55.5% steps, −15 succ |
| P-SoM | 10 | 130 | 0.502 | 0.593 | -0.091 | 58.8% steps, −7 succ | 50.9% steps, −7 succ |

## What this says

**The signal is not there.** AUROC runs 0.336-0.667 against a label-shuffle null of 0.368-0.622; several rows sit **below** their own null. The best row is DOM at k=10 (0.667 vs 0.423).

**And it loses to the trivial policy.** At the loss level truncate-at-k happens to sit at, the learned policy saves fewer steps in 18 of 18 rows.

⚠️ **That matched-loss point is not a deployable one.** Truncating at k=3 discards most of the cell's successes, so both policies are being compared at an operating point nobody would ship. The deployable comparison is the zero-loss column below — and there the fixed policy has **no knob at all**: truncating at k loses its `fixed_lost` successes by construction.

| mode | k | aborted | steps saved (0 successes lost) |
|---|---:|---:|---:|
| DOM | 3 | 17 | 7.5% |
| DOM | 5 | 0 | 0.0% |
| DOM | 10 | 13 | 5.1% |
| SoM | 3 | 1 | 0.3% |
| SoM | 5 | 7 | 3.4% |
| SoM | 10 | 4 | 2.0% |
| Vision | 3 | 0 | 0.0% |
| Vision | 5 | 4 | 1.3% |
| Vision | 10 | 6 | 3.0% |
| P-text | 5 | 9 | 2.5% |
| P-text | 10 | 4 | 0.5% |
| P-prompt | 3 | 3 | 1.7% |
| P-prompt | 5 | 0 | 0.0% |
| P-prompt | 10 | 10 | 3.9% |
| P-SoM | 3 | 0 | 0.0% |
| P-SoM | 5 | 2 | 1.4% |
| P-SoM | 10 | 2 | 1.3% |

So at zero success loss a learned abort reclaims a few percent of steps where the fixed policy reclaims none — a real but small effect resting on an AUROC that is mostly indistinguishable from noise.

## Why this matters for the paper

This is a **third** routing question on the same data, and the three fail (or not) for different reasons:

| question | label supply | signal | outcome |
|---|---|---|---|
| which mode (§6) | **starved** — 4/6 cells admit no classifier | — | fails |
| retry vs switch (§455) | adequate but mostly preference-free | — | no gain over fixed |
| **abort at step k** (this) | **every episode has one** | **absent (AUROC≈null)** | fails |
| **abstain up front** (§457) | **every task has one** | **present (0.615-0.864)** | **works** |

The circularity §7 names therefore has two distinguishable failure modes, not one: a label that does not exist, and a label that exists with no signal behind it. Only the pre-flight question has both.
