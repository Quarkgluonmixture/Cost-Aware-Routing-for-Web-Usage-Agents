## 6. Relabelling closes three routes and opens one

Sections 4 and 5 leave an obvious question: if the supervision is the problem, change the
supervision. Four redefinitions are available without collecting new data. Appendix B derives
each; the outcomes are what matter here.

**Continuous labels are closed by the benchmark.** Regressing on a graded quality signal would
turn every episode into a training example regardless of whether it succeeded, but
VisualWebArena emits only 0.0 and 1.0 [@koh2024visualwebarena]. Any continuous target would be
a surrogate we invented, at which point it is no longer supervision about task success.

**Coarser classes are closed by the count.** Merging classes does not add rows: a cell with 15
labels over five classes still has 15 after merging them into two. Coarsening changes the ratio
of examples to parameters, not the absolute supply, and the absolute supply is what fails.

**Pooling across backbones buys supply and spends identifiability.** Six cells become 260
pooled examples and every class clears the minimum-count filter, but the features carry no
model identity (§2.3), so a pooled row barely says which backbone it describes. Of the tasks
labelled in two or more cells, **57.4%** on classifieds and **56.0%** on reddit carry
conflicting labels, capping any classifier on these features at **83.9%** and **89.1%**
(Appendix A.4, which also reports a stricter grouping). Pooling is less a mistake than a
change of question: it estimates which mode is best marginalising over backbones, and a
deployment has one backbone and needs the answer for that one.

**The screenshot tier is the one target both supplied and identified.** Those conflicts are
about *which mode*. They are much rarer when the question is *whether the task needs the
screenshot*: collapsing the six modes into image-bearing (SoM, Vision) and text-only (DOM and
the three P-modes) raises the ceiling to **92.3%** and **97.8%**, by 8.3 and 8.7 points,
**without a single new solve event** (Appendix A.5). Backbones disagree sharply about which of
six modes is best and agree substantially about whether the image is needed at all.

We call it a screenshot tier and not a cost tier deliberately: as §2.1 records, the
image-bearing tier holds both the most expensive mode and the cheapest, so the partition
separates modalities and not prices. Two further things it is not. It is not a trained router,
since we measure its ceiling and never fit a classifier to it. And it is not the bit §5 tested:
§5's bit asks whether a task is solvable at all, this one asks whether a solved task needed the
screenshot, and it is defined only on solved tasks.

| route | supply | identifiability | outcome |
|---|---|---|---|
| which-mode, per cell | **fails** (15–97 labels, 4/6 cells untrainable) | fine | closed by §4 |
| continuous target | would fix | fine | closed: benchmark score is binary |
| pooled which-mode | fixed (260) | **fails** (56–57% conflict, ceiling 84–89%) | wrong estimand |
| triage (binary) | fine (203–224, every task) | fine | learnable, **and beaten by a fixed policy** (§5) |
| screenshot tier | fine, but only on solved tasks | fine (ceiling 92–98%, over solved tasks) | ceiling measured; no tier classifier trained |

*Table 7: The five supervision targets against the two requirements a trainable router needs.
Each closed route is closed for a different reason, which is what makes the negative result
closed rather than provisional. Supply and identifiability are judged on that target's own
denominator, which differs between the last two rows.*

The four closed routes fail in three distinct ways, which is the argument for regarding the
negative result as closed rather than provisional: it is not one obstruction that a better
method might route around, but supply, estimand, and value failing separately. The fifth row is
not closed but unattempted, and it is the one target we regard as worth attempting next.
