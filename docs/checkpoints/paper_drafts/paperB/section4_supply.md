## 4. Which-mode routing fails on label supply

### 4.1 Labels are produced at the success rate

A which-mode label is the identity of the cheapest successful mode. It therefore exists
only for tasks that were solved by something. This couples the size of the training set
to the very quantity the router is supposed to improve.

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

The largest cell has 97 examples for a six-class problem. The smallest has 15, and in the
two B2 cells one class never appears at all. Pooled across cells the total is 260.

These are not small numbers because we sampled a small benchmark. They are small because
the benchmark is hard for these agents, and the label is a by-product of succeeding at it.
The relationship is mechanical: labelled examples equal successes, so improving the router
requires the successes the router was supposed to help produce.

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

**Four of six cells admit no classifier**: fewer than two classes survive, so there is
nothing to discriminate between. Of the two that survive, one retains three of six classes
and the other two. Training the full pipeline end to end confirms the arithmetic: one cell
trains completely across all five folds, one trains in four of five, and the remaining four
produce no usable model.

A reviewer might reasonably ask whether the threshold is itself the problem. Lowering it
would not create examples. It would only permit fitting classes represented by a handful of
rows, converting an honest refusal into an overfitted model whose cross-validated
performance we would then have to discount anyway. The binding constraint is the count of
labelled examples, and no threshold changes that.

### 4.3 The supervision is also arbitrary where it exists

Two facts about the labels we *do* have are worth recording, because they bound how much
signal the surviving cells could carry even in principle.

First, most labelled tasks were solved by more than one mode: 20% to 71% of labelled rows
across cells. For those tasks the label is whichever successful mode a hardcoded list
reaches first. The list is documented as being in ascending cost order, which would make
it a sound proxy for "cheapest successful".

Second, that documentation is wrong often enough to matter, and §2.1 says why: the order was
built on the assumption that image-bearing modes are the expensive ones, and measured cost
does not agree. Comparing the list's pick against measured per-task cost, per cell
(Appendix A.2):

On 12.5% to 54.6% of labels, the mode recorded as "cheapest successful" was not the
cheapest successful mode. Cost varies per task, the same mode is not uniformly cheaper
across a site, so a static priority order cannot track it.

We note that the exact-tie case, where two successful modes cost the same and the list
order is the only tiebreaker, occurs in **zero** rows across all six cells. Billed cost is
a continuous quantity; two modes costing precisely the same does not happen. Framing this
as tiebreak arbitrariness, as we did in earlier drafts of this analysis, understates it.
The defect is not that ties are broken by a list; it is that the list is used *instead of*
the cost measurement it claims to approximate.

### 4.4 Scope of the supply argument

The argument shows that at these success rates which-mode supervision is unavailable in the
quantity a six-class model needs, and that where it is available it is noisy in a way
traceable to the label definition rather than to the environment. It does not show that mode
selection is unlearnable in principle: on a benchmark where agents succeed at 60% a cell would
produce several hundred labels, and §4.1's mechanism predicts the router becomes trainable
there. Within the 2–43% regime current web agents occupy, the constraint is structural. The
supervision arrives at the rate the agent succeeds, and that rate is what needed improving.
