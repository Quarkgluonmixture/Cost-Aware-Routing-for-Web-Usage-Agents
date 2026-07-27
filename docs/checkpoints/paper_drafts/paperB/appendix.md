## A. Supporting tables

Every number in these tables is quoted in the body section that references it. They are
placed here because the body has an eight-page limit and these tables document rather than
carry the argument.

### A.1 Observation modes

The grid §2.1 refers to. The last column is the modality split §6.4 uses. It is not a cost
split, for the reason §2.1 gives: the image-bearing tier holds both the dearest mode and the
cheapest.

| mode | text payload | prompt family | annotated screenshot |
|---|---|---|---|
| DOM | accessibility tree | DOM | no |
| SoM | mark legend | SoM | **yes** |
| Vision | none | vision | **yes** (unannotated) |
| P-text | mark legend | DOM | no |
| P-prompt | accessibility tree | SoM | no |
| P-SoM | mark legend | SoM | no |

*Table 8: The six observation modes (§2.1). The three P-modes keep either the mark legend,
the SoM prompt, or both, while removing the per-step screenshot.*

### A.2 Where the which-mode label disagrees with measured cost

Per-cell detail for the 12.5–54.6% range quoted in §4.3, and for the claim that the exact-tie
case never occurs.

| cell | labels | multi-success | list picked a strictly pricier mode |
|---|---|---|---|
| classifieds · B0 | 97 | 68 (70.1%) | **53 (54.6%)** |
| reddit · B0 | 53 | 36 (67.9%) | **23 (43.4%)** |
| classifieds · B1 | 55 | 29 (52.7%) | **26 (47.3%)** |
| reddit · B1 | 24 | 17 (70.8%) | **9 (37.5%)** |
| classifieds · B2 | 16 | 4 (25.0%) | **2 (12.5%)** |
| reddit · B2 | 15 | 3 (20.0%) | **2 (13.3%)** |

*Table 9: Per-cell detail behind §4.3. The last column counts labelled tasks on which the
fixed priority list selected a mode strictly more expensive than another mode that also
succeeded on that task. The exact-tie case, where the list order is the only tiebreaker,
occurs in zero rows in every cell.*

### A.3 Effect of nesting the operating-point selection

Per-cell detail for the −0.99pp to +1.34pp range quoted in §5.2, and for the observation that
nesting moves the result in both directions.

| cell | naive nesting SR | fully nested SR | Δ |
|---|---|---|---|
| classifieds · B0 | 25.45% | 26.79% | **+1.34pp** |
| classifieds · B1 | 13.84% | 14.29% | +0.45pp |
| classifieds · B2 | 1.34% | 1.34% | ±0.00pp |
| reddit · B0 | 13.79% | 12.81% | **−0.99pp** |
| reddit · B1 | 6.40% | 5.91% | −0.49pp |
| reddit · B2 | 3.94% | 3.94% | ±0.00pp |

*Table 10: Per-cell detail behind §5.2. The naive column selects the threshold and the
strongest and cheapest modes from whole-cell outcomes; the nested column re-derives all three
inside each outer fold. Nesting moves the result in both directions.*

### A.4 Identifiability of pooled which-mode labels

Per-site detail for the conflict rates and Bayes ceilings quoted in §6.3.

| site | tasks labelled in ≥2 cells | conflicting | conflict rate | Bayes ceiling |
|---|---|---|---|---|
| classifieds | 54 | 31 | **57.4%** | **79.2%** |
| reddit | 25 | 14 | **56.0%** | **83.7%** |

*Table 11: Per-site detail behind §6.3. A conflict is one task on which two cells recorded
different oracle modes; since the features carry no model identity, both rows present the same
input. The Bayes ceiling is the accuracy of emitting the modal label per distinct feature
vector.*

### A.5 The screenshot-modality tier

Per-site detail for the ceilings quoted in §6.4. Both columns are measured over tasks solved
by at least two backbones, which is the only set on which a tier label exists for more than
one backbone.

| site | which-mode ceiling | tier ceiling | tier agreement across backbones |
|---|---|---|---|
| classifieds | 79.2% | **89.9%** | 68.5% |
| reddit | 83.7% | **96.7%** | 88.0% |

*Table 12: Per-site detail behind §6.4, over tasks solved by at least two backbones. The
ceiling rises on the same solve events, because backbones that disagree about which mode is
best still agree about whether the screenshot is needed. No classifier is fitted to this
target anywhere in the paper; only its ceiling is measured.*

## B. Derivations for the four relabelling routes

§6 states the outcome of each route. This appendix gives the derivation.

### B.1 Continuous labels

The cleanest fix would be to regress on a graded quality signal rather than classify a discrete
winner: partial credit turns every episode into a training example regardless of whether it
succeeded. VisualWebArena does not provide one. Across 7,963 episodes the evaluator emits
exactly two values, 0.0 and 1.0, in a 7,278 / 685 split. This is a property of the benchmark's
evaluation design rather than of our pipeline, and it forecloses the route entirely.

### B.2 Coarser classes

§4.2's obstruction is that four of six cells have fewer than ten labelled rows in more than one
class. Merging classes does not add rows. The binary collapse of §6 does help, but for a
different reason, having to do with agreement across backbones rather than with class count.

### B.3 Pooling across backbones

Six cells at 15–97 labels become 260 pooled examples and every class clears the minimum-count
filter, so supply is solved. Identifiability is not. The features carry no model identity, so
two backbones facing the same task on the same site produce the same feature vector, and when
their oracle labels differ a task-feature classifier is being asked to emit two different
answers for one input. The Bayes ceiling reported in A.4 is the accuracy of the best possible
rule on the pooled set, which is to emit the modal label for each distinct feature vector.

### B.4 Screenshot tier

The tier label is derived from the same solve events as the which-mode label, by mapping each
oracle mode to image-bearing (SoM, Vision) or text-only (DOM, P-text, P-prompt, P-SoM). No new
episodes are involved, which is why the ceiling rises without a single new solve event. The
tier is defined only on tasks that some mode solved, so its denominator is the solved set and
not the full task universe; A.5 reports it over tasks solved by at least two backbones, the
only set on which cross-backbone agreement is defined.

## C. The reddit · B2 saving in detail

§5.4 reports that the one cell whose cost saving survives Holm correction has an AUROC below
chance. The mechanism is tail enrichment rather than a globally ordered score.

Reddit · B2 sends 192 of 203 tasks (95%) to the cheap mode with no accuracy loss, which is
unsurprising in a cell where only 7.4% of tasks are solvable at all: almost nothing in that 95%
was going to succeed under any mode. The policy therefore differs from the free always-cheapest
policy by five percent of the allocation. The 11 tasks it holds back for the strong mode carry
four successes that the fixed policy does not collect, against four collected by the fixed
policy overall. The permutation null detects that enrichment. A globally ordered score is not
required to produce it, which is why the cell's AUROC of 0.483, below both chance and its own
best single covariate at 0.711, is consistent with a real saving.

### B.5 Supply and trainability under both label definitions

§4.3 keeps the prior-order label and reports the measured-cost rule as a sensitivity. Supply is
identical under both by construction, since each labels exactly the tasks some mode solved, so
only the class distribution and through it trainability can move. The relabelled column equals
the "order picked a strictly pricier mode" column of A.2 exactly, which is the consistency check
one wants: a label moves if and only if the order's pick was not the measured cheapest.

| cell | labels | relabelled | prior order: surviving classes | measured cost: surviving classes |
|---|---|---|---|---|
| classifieds · B0 | 97 | 53 | 3 (DOM, P-prompt, SoM) | 2 (SoM, Vision) |
| reddit · B0 | 53 | 23 | 1 (DOM) — **no** | 1 (P-text) — **no** |
| classifieds · B1 | 55 | 26 | 2 (DOM, SoM) | 1 (Vision) — **no** |
| reddit · B1 | 24 | 9 | 0 — **no** | 0 — **no** |
| classifieds · B2 | 16 | 2 | 0 — **no** | 0 — **no** |
| reddit · B2 | 15 | 2 | 0 — **no** | 0 — **no** |

*Table 13: Trainability under the two label definitions. "Surviving" counts classes clearing ten
training rows in a five-fold split; **no** marks a cell with fewer than two, where no classifier
exists. Four of six cells are untrainable under the reported label and five of six under the
measured-cost alternative, so the supply argument does not depend on the choice.*

The single cell that changes, classifieds · B1, loses a class rather than gaining one: the
prior-order label keeps DOM and SoM above the threshold, the measured-cost label concentrates
enough of those rows onto Vision that only Vision survives. The alternative definition therefore
strengthens the negative result, which is a reason to report it and not a reason to adopt it.

### C.1 The best-success mode is not stable across folds

The nested design of §5.2 re-selects the best-success mode inside every outer fold, which
exposes something the whole-cell version conceals. In reddit · B0 the five outer folds select
DOM, DOM, SoM, SoM, DOM. A pipeline that picks one best mode from all realised outcomes is
therefore not merely optimistic about its threshold; it reports a mode choice that its own
resampling does not reproduce.
