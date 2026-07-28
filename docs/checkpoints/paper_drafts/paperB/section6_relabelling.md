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
conflicting labels. Emitting each task's modal pooled label is right only **79.2%** and
**83.7%** of the time, which is an optimistic in-sample bound (Appendix A.4). Pooling is less a mistake than a
change of question: it estimates which mode is best marginalising over backbones, and a
deployment has one backbone and needs the answer for that one.

**The screenshot tier is the one target both supplied and identified.** Those conflicts are
about *which mode*. They are much rarer when the question is *whether the task needs the
screenshot*: collapsing the six modes into image-bearing (SoM, Vision) and text-only (DOM and
the three P-modes), backbones that both label a task agree on its tier **68.5%** and **88.0%**
of the time, against 42.6% and 44.0% for the six-way label, **without a single new solve event**
(Appendix A.5). A binary label has a high chance baseline, so we call this suggestive.

We call it a screenshot tier and not a cost tier deliberately: as §2.1 records, the
image-bearing tier holds both the most expensive mode and the cheapest, so the partition
separates modalities and not prices. Two further things it is not. It is not a trained router,
since we measure only agreement and never fit a classifier to it. And it is not the bit §5 tested:
§5's bit asks whether a task is solvable at all, this one asks whether a solved task needed the
screenshot, and it is defined only on solved tasks.

| route | supply | identifiability | outcome |
|---|---|---|---|
| which-mode, per cell | **fails** (15–97 labels, 4/6 cells untrainable) | fine | closed by §4 |
| continuous target | would fix | fine | closed: benchmark score is binary |
| pooled which-mode | fixed (260) | **fails** (56–57% conflict; in-sample modal agreement 79–84%) | wrong estimand |
| triage (binary) | fine (203–224, every task) | fine | learnable, **and beaten by a fixed policy** (§5) |
| screenshot tier | fine, but only on solved tasks | fine (68–88% cross-backbone agreement, over solved tasks) | agreement measured; no tier classifier trained |

*Table 7: The five supervision targets against the two requirements a trainable router needs.
Each closed route is closed for a different reason, which is what makes the negative result
closed rather than provisional. Supply and identifiability are judged on that target's own
denominator, which differs between the last two rows.*

The four closed routes fail in three distinct ways, namely supply, estimand and value failing
separately, rather than one obstruction a better method might route around. The fifth row is not
closed but unattempted, and it is the one we regard as worth attempting next.
