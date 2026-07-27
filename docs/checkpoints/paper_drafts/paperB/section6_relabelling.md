## 6. Relabelling closes three routes and opens one

Sections 4 and 5 leave an obvious question: if the supervision is the problem, change the
supervision. Three redefinitions are available without collecting new data, and this
section reports what each buys. Two are closed by properties of the benchmark; the third
trades supply for identifiability, and its accounting points at the one target that works.

### 6.1 Continuous labels: closed by the benchmark

The cleanest fix would be to regress on a graded quality signal rather than classify a
discrete winner, partial credit turns every episode into a training example regardless of
whether it succeeded.

VisualWebArena does not provide one [@koh2024visualwebarena]. Across 7,963 episodes the
evaluator emits exactly two
values, 0.0 and 1.0, in a 7,278 / 685 split. There is no partial credit to regress on. This
is a property of the benchmark's evaluation design, not of our pipeline, and it forecloses
the route entirely: any continuous target would have to be a surrogate we invented, at
which point it is no longer supervision about task success.

### 6.2 Coarser classes: closed by the count, not the class structure

The second fix is to reduce the number of classes, and it does not work when the collapse is
only about class count. §4.2's obstruction is that four of six cells have fewer than ten
labelled rows in more than one class, and merging classes does not add rows: a cell with 15
labels over five classes still has 15 after merging them into two. Coarsening changes the
ratio of examples to parameters, not the absolute supply, and the absolute supply is what
fails. §6.4's binary collapse does help, but for a different reason, having to do with
agreement across backbones rather than with class count.

### 6.3 Pooling across backbones: supply for identifiability

The third fix is to pool labels across cells. Six cells at 15–97 labels become 260 pooled
examples, and every class clears the minimum-count filter. Supply is solved.

Identifiability is not. As noted in §2.3, the features carry no model identity: two
backbones facing the same task on the same site produce the same feature vector. When their
oracle labels differ, a task-feature classifier is being asked to emit two different
answers for one input.

Of the tasks labelled in two or more cells, **57.4%** on classifieds (31 of 54) and **56.0%**
on reddit (14 of 25) carry conflicting labels (Appendix A.4). More than half of the shared
tasks therefore carry contradictory supervision. The Bayes ceiling is the accuracy of the best
possible rule on the pooled set, emit the modal label for each distinct feature vector, and it
caps any task-feature classifier at **79.2%** and **83.7%** respectively.

The interpretation is not that pooling is a mistake. It is that pooling changes the
question being asked. A model trained on the pooled set is estimating "which mode is best
for this task, marginalising over backbones", which is not the quantity a deployment needs;
a deployment has one backbone and needs the answer for that backbone. Pooling buys examples
by discarding the conditioning variable that makes the label well-defined.

### 6.4 Screenshot tier: the one target that is both supplied and identified

The conflicts in §6.3 are about *which mode*. They are much rarer when the question is
*whether the task needs the screenshot*. Collapsing the six modes into two tiers,
image-bearing (SoM, Vision) and text-only (DOM and the three P-modes), and repeating the
measurement, the ceiling rises from 79.2% to **89.9%** on classifieds and from 83.7% to
**96.7%** on reddit, with tier agreement across backbones of 68.5% and 88.0% (Appendix A.5).

We call this a screenshot tier and not a cost tier deliberately. As §2.1 records, the
image-bearing tier contains both the most expensive mode (SoM) and the cheapest (Vision), so
the partition separates modalities and not prices, and nothing in this subsection supports a
cost claim.

The ceiling rises by 10.7 and 13.0 points **without a single new solve event**. Nothing was
collected; the same successes were re-described. Backbones disagree sharply about which of
six modes is best and agree substantially about whether the image is needed. 88% of shared
reddit tasks receive the same tier from every backbone that solved them.

This is the one place in our results where a relabelling is genuinely free. Two things it is
not. It is not a trained router: we measure its ceiling and never fit a classifier to it, so
we make no claim about whether the tier bit is predictable from task features. And it is not
the bit §5 tested. §5's bit asks whether a task is solvable at all; this one asks whether a
solved task needed the screenshot, and it is defined only on solved tasks. The identifiability
analysis explains *why* a one-bit target is more tractable than a six-class one, and §5
explains why tractable was not enough for the one bit we did train.

### 6.5 Summary of the five supervision targets

| route | supply | identifiability | outcome |
|---|---|---|---|
| which-mode, per cell | **fails** (15–97 labels, 4/6 cells untrainable) | fine | closed by §4 |
| continuous target | would fix | fine | closed: benchmark score is binary |
| pooled which-mode | fixed (260) | **fails** (56–57% conflict, ceiling 79–84%) | wrong estimand |
| triage (binary) | fine (203–224, every task) | fine | learnable, **and beaten by a fixed policy** (§5) |
| screenshot tier | fine, but only on solved tasks | fine (ceiling 90–97%, over solved tasks) | ceiling measured; no tier classifier trained |

*Table 7: The five supervision targets against the two requirements a trainable router
needs. Each closed route is closed for a different reason, which is what makes the negative
result closed rather than provisional. Supply and identifiability are judged on that target's
own denominator, which differs between the last two rows.*

The four closed routes fail in three distinct ways, which is the argument for regarding the
negative result as closed rather than provisional: it is not one obstruction that a better
method might route around, but supply, estimand, and value failing separately. The fifth row
is not closed; it is unattempted, and §6.4 says why we regard it as the one target worth
attempting next.
