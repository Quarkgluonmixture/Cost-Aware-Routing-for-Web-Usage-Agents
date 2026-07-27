## 5. Triage routing has the labels, the signal, and no value

Section 4's failure is about supply. The obvious response is to pick a target whose labels
are not conditional on success. Triage does that: the label is whether *any* mode solves
the task, defined for every task in the universe. Supply goes from 15–97 examples to 203
or 224, and the problem becomes binary.

This section shows that it still fails, that the failure is not a supply failure, and that
the reason is the baseline.

### 5.1 The label is predictable

| cell | tasks | solvable | AUROC (logistic) | best single covariate | margin |
|---|---|---|---|---|---|
| classifieds · B0 | 224 | 43.3% | 0.676 | 0.607 | +0.069 |
| reddit · B0 | 203 | 26.1% | 0.666 | 0.612 | +0.054 |
| classifieds · B1 | 224 | 24.6% | **0.717** | 0.627 | +0.090 |
| reddit · B1 | 203 | 11.8% | 0.685 | 0.637 | +0.048 |
| classifieds · B2 | 224 | 7.1% | 0.651 | 0.655 | −0.005 |
| reddit · B2 | 203 | 7.4% | **0.483** | 0.711 | −0.228 |

*Table 5: Predictability of the triage label. AUROC is cross-validated; the comparison column
is the best single covariate used alone, so the margin isolates what the multivariate model
adds. The reddit · B2 row is below chance and is the subject of §5.4.*

In five of six cells cross-validated AUROC is 0.651–0.717, and in four it beats the best
single covariate used alone. Whether a task is solvable by anything is genuinely
predictable from task features, noticeably more so than which mode will solve it.

The sixth cell is discussed in §5.4, because it turns out to be the only cell with a
statistically detectable saving.

### 5.2 Nesting the operating point

Turning a score into a policy requires a threshold: below it, send to the cheapest mode;
above it, send to the strongest. Selecting that threshold against realised outcomes on the
whole cell makes the resulting operating point in-sample, even when the scores themselves
are out-of-fold [@cawley2010overfitting]. So does selecting *which* mode is "strongest" and
"cheapest" from whole-cell outcomes.

We therefore report a fully nested design. Within each outer fold, and using only that
fold's training rows: the best-success and cheapest modes are re-selected; an inner
cross-validation produces out-of-fold scores; the threshold is chosen against those inner
scores; and a model refitted on the training rows scores the held-out rows. Nothing that
touches an outer test fold has seen it.

Nesting moves the result in both directions, from **−0.99pp** on reddit · B0 to **+1.34pp**
on classifieds · B0 (Appendix A.3), which is worth stating plainly because it is not the usual
story of honest evaluation being uniformly worse. The two classifieds gains come from
re-selecting the mode per fold: the fold-local choice is sometimes better adapted than the
global one, and because that selection uses only training rows it is a legitimate gain rather
than leakage.

The nested design also exposes something the whole-cell version conceals. **The
best-success mode is not stable across folds.** In reddit · B0 the five outer folds select
DOM, DOM, SoM, SoM, DOM. A pipeline that picks one best mode from all realised outcomes is
not merely optimistic about its threshold; it reports a mode choice its own resampling does
not reproduce.

### 5.3 The comparison that matters

Against the best single mode, the nested triage policy looks reasonable: it holds accuracy
within two percentage points and cuts cost by 0.5% to 26.5%.

Against always-cheapest, it does not:

| cell | nested SR / cost | always-cheapest SR / cost | Pareto-dominates? |
|---|---|---|---|
| classifieds · B0 | 26.79% / 0.07197 | 25.00% / 0.06481 | no (+1.79pp SR, +11.0% cost) |
| reddit · B0 | 12.81% / 0.09836 | 7.39% / 0.09807 | no (+5.42pp SR, +0.3% cost) |
| classifieds · B1 | 14.29% / 0.05757 | 12.50% / 0.04316 | no (+1.79pp SR, +33.4% cost) |
| reddit · B1 | 5.91% / 0.06970 | 2.46% / 0.05240 | no (+3.45pp SR, +33.0% cost) |
| classifieds · B2 | 1.34% / 0.07247 | 2.23% / 0.07065 | no (−0.89pp SR, +2.6% cost) |
| reddit · B2 | 3.94% / 0.06964 | 1.97% / 0.06833 | no (+1.97pp SR, +1.9% cost) |

*Table 6: The nested triage policy against the always-cheapest fixed policy. Pareto
dominance requires no worse on both axes; no cell achieves it. The parenthetical gives the
router's deltas relative to the fixed policy.*

**No cell Pareto-dominates the fixed policy.** Every cell where the router retains more
accuracy pays more to do it, and in classifieds · B2 it is dominated outright, worse on both
axes than taking the cheapest mode everywhere.

These are trade-off points, not wins. A deployment could rationally choose the reddit · B0 row,
where 5.42 points of accuracy cost 0.3% more, but that is a judgement about how much accuracy
is worth rather than a demonstration that the router beats the free alternative, and it is
available only in the cell with the highest solvable rate on reddit.

### 5.4 The one significant cell, and why it does not help

We test each cell's cost saving against a permutation null in which the task bundle
(label, per-mode outcomes, per-mode costs) is shuffled relative to the features, holding
the selection procedure fixed so that the null and the observation share it. Under
Holm correction [@holm1979sequentially] across the six cells, one saving survives: reddit · B2, at p = 0.0050
against a threshold of 0.0083.

That cell has AUROC 0.483, below chance and below its own best single covariate at 0.711. The
two facts are not in conflict: AUROC scores the *global* ranking and the saving comes from the
*tail*. Reddit · B2 sends 192 of 203 tasks (95%) to the cheap mode with no accuracy loss, which
is unsurprising where only 7.4% of tasks are solvable at all. It therefore differs from the
free fixed policy by five percent of the allocation, and the 11 tasks it holds back carry four
successes the fixed policy does not collect. The permutation test detects that enrichment; a
globally ordered score is not required to produce it, and AUROC does not see it.

So the honest reading of §5 is not "the label is predictable, yet triage fails". It is:

> At 2–27% base success rate, a high AUROC is neither necessary nor sufficient for a triage
> policy to save money. What decides the outcome is whether a handful of tail tasks fall on
> the correct side of the threshold, and at n = 203 that handful is four successes.

This is why we regard the negative result as robust rather than underpowered. What would have
to improve is not the score, which already beats the strongest single feature in four cells,
but the number of solve events in the tail, which returns to §4's constraint.
