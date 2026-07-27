## 6. Relabelling closes three routes and opens one

Sections 4 and 5 leave an obvious question: if the supervision is the problem, change the
supervision. Three redefinitions are available without collecting new data, and this
section reports what each buys. Two are closed by properties of the benchmark; the third
trades supply for identifiability, and its accounting points at the one target that works.

### 6.1 Continuous labels: closed by the benchmark

The cleanest fix would be to regress on a graded quality signal rather than classify a
discrete winner, partial credit turns every episode into a training example regardless of
whether it succeeded.

VisualWebArena does not provide one. Across 7,963 episodes the evaluator emits exactly two
values, 0.0 and 1.0, in a 7,278 / 685 split. There is no partial credit to regress on. This
is a property of the benchmark's evaluation design, not of our pipeline, and it forecloses
the route entirely: any continuous target would have to be a surrogate we invented, at
which point it is no longer supervision about task success.

### 6.2 Coarser classes: closed by the count, not the class structure

The second fix is to reduce the number of classes. Six modes is a lot to discriminate with
97 examples; collapsing to two should help.

It does not, when the collapse is only about class count. §4.2's obstruction is that four
of six cells have fewer than ten labelled rows in more than one class, and merging classes
does not add rows. A cell with 15 labels spread over five classes still has 15 labels after
merging them into two. Coarsening changes the ratio of examples to parameters; it does not
change the absolute supply, and the absolute supply is what fails.

This is worth separating from §6.4, where a specific binary collapse *does* help, for a
different reason, having to do with agreement across backbones rather than with class count.

### 6.3 Pooling across backbones: supply for identifiability

The third fix is to pool labels across cells. Six cells at 15–97 labels become 260 pooled
examples, and every class clears the minimum-count filter. Supply is solved.

Identifiability is not. As noted in §2.3, the features carry no model identity: two
backbones facing the same task on the same site produce the same feature vector. When their
oracle labels differ, a task-feature classifier is being asked to emit two different
answers for one input.

| site | tasks labelled in ≥2 cells | conflicting | conflict rate | Bayes ceiling |
|---|---|---|---|---|
| classifieds | 54 | 31 | **57.4%** | **79.2%** |
| reddit | 25 | 14 | **56.0%** | **83.7%** |

More than half of the shared tasks carry contradictory supervision. The Bayes ceiling is
the accuracy of the best possible rule on the pooled set, emit the modal label for each
distinct feature vector, and it caps any task-feature classifier at 79.2% and 83.7%.

The interpretation is not that pooling is a mistake. It is that pooling changes the
question being asked. A model trained on the pooled set is estimating "which mode is best
for this task, marginalising over backbones", which is not the quantity a deployment needs;
a deployment has one backbone and needs the answer for that backbone. Pooling buys examples
by discarding the conditioning variable that makes the label well-defined.

### 6.4 Cost tier: the one target that is both supplied and identified

The conflicts in §6.3 are about *which mode*. They are much rarer when the question is
*whether the task needs the screenshot*. Collapsing the six modes into two tiers:
image-bearing (SoM, Vision) and text-only (DOM and the three P-modes), and repeating the
measurement:

| site | which-mode ceiling | cost-tier ceiling | tier agreement across backbones |
|---|---|---|---|
| classifieds | 79.2% | **89.9%** | 68.5% |
| reddit | 83.7% | **96.7%** | 88.0% |

The ceiling rises by 10.7 and 13.0 points **without a single new solve event**. Nothing was
collected; the same successes were re-described. Backbones disagree sharply about which of
six modes is best and agree substantially about whether the image is needed. 88% of shared
reddit tasks receive the same tier from every backbone that solved them.

This is the one place in our results where a relabelling is genuinely free. It is also, we
note, a considerably less interesting router: it decides one bit, and §5 has already shown
that a one-bit decision does not beat always-cheapest at these success rates. The
identifiability analysis explains *why* triage was the more tractable of our two targets,
and §5 explains why tractable was not enough.

### 6.5 Summary of the four routes

| route | supply | identifiability | outcome |
|---|---|---|---|
| which-mode, per cell | **fails** (15–97 labels, 4/6 cells untrainable) | fine | closed by §4 |
| continuous target | would fix | fine | closed: benchmark score is binary |
| pooled which-mode | fixed (260) | **fails** (56–57% conflict, ceiling 79–84%) | wrong estimand |
| triage / cost tier | fine (203–224) | fine (ceiling 90–97%) | learnable, **and beaten by a fixed policy** |

The four routes fail in three distinct ways, which is the argument for regarding the
negative result as closed rather than provisional: it is not one obstruction that a better
method might route around, but supply, estimand, and value failing separately.
