## 6. Relabelling closes three routes and opens two

Sections 4 and 5 leave an obvious question: if the supervision is the problem, change the
supervision. Five redefinitions are available without collecting new data. Appendix B derives
each; the outcomes are what matter here.

**Per-mode binary success drops the winner label entirely, and it is the strongest of the
five.** §4's scarcity is a property of one target (*the identity of the winning mode*), not of
the outcome matrix. A task no mode solved is not unlabelled; under a per-mode target it is six
negative examples. Every cell therefore holds 1,218 or 1,344 observed (task, mode) binary
outcomes, not 15–97. Fitting one out-of-fold success head $\hat p_m(x)$ per mode and taking the
cheapest mode with $\hat p_m(x) \ge \tau$ needs neither a graded score nor a unique winner.

We ran it, and it is the only formulation we tested that beats a fixed policy on both axes at a
point estimate: at $\tau = 0.10$ on classifieds · B0 it reaches **29.91% at \$0.0705**, against
the best single mode's 27.23% at \$0.0724, or **+2.68pp for 2.5% less**. Two things keep us from
reporting that as a positive result. Against the baseline §3.2 argues is the one that matters,
always-cheapest, it is **not** dominant but merely on the frontier (+4.91pp for +8.8%). And the
threshold is post-hoc: twenty-one values were swept on one out-of-fold replay, the curve peaks
at $\tau = 0.10$ and declines on both sides, and a frozen $\tau = 0.10$ transferred to reddit
**failed** at 11.22% against DOM's 14.63% ($\Delta = -3.41$pp, 95% CI $[-7.80, +0.98]$).

We therefore report it as the open route with the strongest prior, not as a result, and we
withdraw any reading of §4 on which no estimator could exist. What §4 shows is that *the
hindsight-winner target* is not produced at these success rates. A fully nested, pre-specified
per-mode success model across all six cells is the experiment that would settle whether the
routing failure is about supervision at all; we have not run it.

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
| **per-mode binary** | **fine (1,218–1,344 (task, mode) outcomes)** | fine | **open**: dominates best-single at a post-hoc τ, not always-cheapest; frozen transfer failed |
| screenshot tier | fine, but only on solved tasks | fine (68–88% cross-backbone agreement, over solved tasks) | agreement measured; no tier classifier trained |

*Table 7: The six supervision targets against the two requirements a trainable router needs.
Each closed route is closed for a different reason, which is what makes the negative result
closed rather than provisional. Supply and identifiability are judged on that target's own
denominator, which differs across the last three rows.*

**The three closed routes fail in three distinct ways**, supply and estimand and value failing
separately, rather than one obstruction a better method might route around. That is what makes
the negative result closed rather than provisional: there is no single repair to attempt.

The two open rows are open for different reasons, the per-mode target having been attempted and
left unresolved and the screenshot tier not attempted at all. Both are worth doing next, and it
is worth noting where they point. The per-mode target has the most supply of anything we tested
and still failed to transfer a frozen threshold across sites, which is the same lesson as §5's:
supply stops being the binding constraint well before a deployable policy appears. And the
screenshot tier is a decision about which channel to carry, taken once rather than per task,
which is the shape of the conclusion §1 reaches from the other end.
