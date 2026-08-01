## 4. Which-mode routing fails on label supply

### 4.1 Labels are produced at the success rate

A which-mode label names the cheapest mode that solved the task (§4.3 makes "cheapest"
precise), so it exists only for tasks something solved. That couples the size of the training
set to the very quantity the router is supposed to improve.

| cell | scored tasks | labelled | solvable | classes present |
|---|---|---|---|---|
| classifieds · B0 | 224 | **97** | 43.3% | 6/6 |
| reddit · B0 | 203 | **53** | 26.1% | 6/6 |
| classifieds · B1 | 224 | **55** | 24.6% | 6/6 |
| reddit · B1 | 203 | **24** | 11.8% | 6/6 |
| classifieds · B2 | 224 | **16** | 7.1% | 5/6 |
| reddit · B2 | 203 | **15** | 7.4% | 5/6 |

*Table 3: Which-mode label supply. A label exists only where some mode solved the task, so
the labelled column equals the number of solved tasks. Six classes have to be discriminated
from between 15 and 97 examples.*

The largest cell has 97 examples for a six-class problem, the smallest 15, and in the two B2
cells one class never appears. Pooled across cells the total is 260. These are small not
because we sampled a small benchmark but because the benchmark is hard for these agents and
the label is a by-product of succeeding at it: labelled examples equal successes, so improving
the router requires the successes the router was supposed to help produce.

### 4.2 The supply is below the threshold for fitting anything

Counting examples understates the problem, because a multiclass fit needs examples *per
class per fold*. Applying a minimum of ten training rows per class in a five-fold split:
the threshold used by our pipeline, and unremarkable as practice, leaves:

| cell | labels | classes | classes surviving the filter | trainable |
|---|---|---|---|---|
| classifieds · B0 | 97 | 6 | 3 (DOM, P-prompt, SoM) | yes |
| reddit · B0 | 53 | 6 | 1 (DOM) | **no** |
| classifieds · B1 | 55 | 6 | 2 (DOM, SoM) | yes |
| reddit · B1 | 24 | 6 | 0 | **no** |
| classifieds · B2 | 16 | 5 | 0 | **no** |
| reddit · B2 | 15 | 5 | 0 | **no** |

*Table 4: Classes surviving a minimum of ten training rows per class in a five-fold split.
Fewer than two surviving classes leaves nothing to discriminate between, which is the state
of four of the six cells.*

**Four of six cells admit no classifier**: fewer than two classes survive, so there is nothing
to discriminate between. Of the two that survive, one retains three of six classes and the
other two. Training the pipeline end to end confirms the arithmetic: one cell trains across all
five folds, one in four of five, and the remaining four produce no usable model.

Lowering the threshold would not create examples. It would only permit fitting classes
represented by a handful of rows, converting an honest refusal into an overfitted model whose
cross-validated performance we would then discount anyway.

Nor does the count turn on the label definition. §4.3 gives a second defensible one;
relabelling under it leaves supply identical by construction and moves the untrainable count
from four cells to **five** (Appendix B.5), the flipped cell losing a class rather than gaining
one.

### 4.3 Two defensible labels, and which one we report

Most labelled tasks were solved by more than one mode, 20% to 71% of rows, so on most rows
something must choose among the winners. Two rules are available and they disagree on 12.5% to
54.6% of labels (Appendix A.2). The one we report walks a fixed mode list in **prior** cost
order, text-only ahead of image-bearing, and takes the first success; the alternative takes the
cheapest success by **measured** per-episode cost. §2.1 says why they differ: measured cost is
dominated by how many steps an episode took, not by whether it carried an image.

We report the prior-order rule for three reasons, each checked on the landed data. The measured
per-mode order is not stable across cells, so "cheapest" is not a property of a mode: P-prompt
is second-cheapest on classifieds · B0 and dearest on classifieds · B1. Measured episode cost is
endogenous to the outcome, so a label defined by it smuggles outcome information into a target a
pre-action router must predict from pre-action features. And per-mode success counts of 14 to 61
are too few to pin an order. Appendix B.5 reports both.

Two successful modes never cost exactly the same: the exact-tie case occurs in **zero** rows
across all six cells, billed cost being continuous. Earlier drafts framed the disagreement as
tiebreak arbitrariness, which was wrong twice over. It is not a tie, and it is not
arbitrariness, but two rules answering slightly different questions.

### 4.4 Scope of the supply argument

At these success rates which-mode supervision is unavailable in the quantity a six-class model
needs, under either label definition. This does not show mode selection is unlearnable in
principle: at 60% success a cell would produce several hundred labels. Within the 2–43% regime
current web agents occupy the constraint is structural, because supervision arrives at the rate
the agent succeeds and that rate is what needed improving.

One check outside this benchmark, with the same six modes and B1 on the reddit split of
WebArena [@zhou2024webarena], whose tasks specify their goal in text alone where 33.7% of the ones
above attach a reference image. It behaves as predicted from a *more* favourable start: 32 of
104 tasks are solvable, a higher rate than any B1 cell in Table 3, and all six classes appear,
yet only one clears the per-class threshold (DOM, 17 of 32), leaving the cell untrainable for the
same reason reddit · B0 is. A higher solve rate does not by itself buy a trainable label set,
because labels concentrate on whichever mode is strongest. This run is exploratory and outside
the pre-registered design; we report it as corroboration, not as a seventh cell, and do not
pool it.
