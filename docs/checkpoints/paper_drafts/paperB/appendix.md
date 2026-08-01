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

Per-site detail for the conflict rates and modal-agreement figures quoted in §6.3.

| site | tasks labelled in ≥2 cells | conflicting | conflict rate | in-sample modal agreement | same, on shared tasks only | same, exact-vector grouping |
|---|---|---|---|---|---|---|
| classifieds | 54 | 31 | **57.4%** | **79.2%** | 70.3% (n=118) | 83.9% |
| reddit | 25 | 14 | **56.0%** | **83.7%** | 74.1% (n=58) | 89.1% |

*Table 11: Per-site detail behind §6.3. A conflict is one task on which two cells recorded
different oracle modes. In-sample modal agreement is the accuracy of emitting the modal label
per task, over all pooled labelled rows (168 on classifieds, 92 on reddit), a row being one
(task, backbone) pair on which that backbone solved the task.*

*We call it agreement and not a Bayes ceiling because it is a resubstitution estimate: it scores
the same rows it took the modal label from, so a task labelled by only one backbone is correct
by construction. That describes 50 of 168 classifieds rows (29.8%) and 34 of 92 reddit rows
(37.0%); restricted to tasks two or more backbones label, agreement falls to 70.3% and 74.1%.
An out-of-sample bound would need leave-one-backbone-out prediction or a shrinkage estimator,
and would be lower still. Every number here is therefore an optimistic bound on what a pooled
classifier could reach, which is the direction the argument needs.*

*The last column groups by the exact feature vector instead of by task. Rows of one task are not
always identical, because three of the five numeric features are read from that backbone's own
step-0 observation, so they differ somewhere on 31.5% of shared classifieds tasks and 80.0% of
shared reddit tasks. We report but do not use that grouping: it leaves 74 of 117 classifieds
groups and 69 of 78 reddit groups with a single member, covering 44% and 75% of rows, so it is
even more inflated than the headline, and a router serving one backbone could not recover
backbone identity from that jitter anyway.*

### A.5 The screenshot-modality tier

Per-site detail for §6.4, on the same pooled labelled rows and the same grouping as A.4. The
agreement column is the exception: it is defined only on tasks labelled by two or more
backbones, since agreement needs two labels to compare.

| site | which-mode modal agreement | tier modal agreement | tier agreement across backbones | six-way agreement across backbones |
|---|---|---|---|---|
| classifieds | 79.2% | **89.9%** | **68.5%** | 42.6% |
| reddit | 83.7% | **96.7%** | **88.0%** | 44.0% |

*Table 12: Per-site detail behind §6.4, over the same solve events. Columns two and three carry
the same resubstitution caveat as A.4, and column three additionally rises for an arithmetic
reason: merging six classes into two can only increase a modal share. The claim that backbones
agree about the screenshot therefore rests on the last two columns, which measure agreement
between two backbones' labels directly rather than against a modal label. Under A.4's
exact-vector grouping the tier figures are 92.3% and 97.8%. No classifier is fitted to this
target anywhere in the paper.*

### A.6 The rerun reference of §3.3

Three same-condition replicate pairs exist. Two are end-to-end reruns of landed classifieds
conditions under B0, produced by a reproducibility exercise. The third was not run for this
purpose: on the reddit split of WebArena a 10-task pilot draw and the subsequent full run use
the *same* condition — the full configuration layer only removes the pilot's task-id filter —
so their overlap on the pilot's tasks is a same-condition rerun, and we read it as one.

| pair | n | $\lvert A \setminus B \rvert$ | $\lvert B \setminus A \rvert$ | discordance |
|---|---|---|---|---|
| cls · B0 · DOM | 224 | 7.14 | 4.91 | 12.05% |
| cls · B0 · Vision | 224 | 7.59 | 6.70 | 14.29% |
| WA-red · B1, 5 modes pooled | 50 | 2.00 | 4.00 | 6.00% |

*Table 13: Run-to-run instability of a single fixed condition, in percentage points of the task
universe. On both classifieds pairs every flip is trajectory divergence and step-0 start URLs
agree on all 224 tasks, so none of it is reset contamination. On WebArena the flips fall
entirely in Vision (2) and P-prompt (1); DOM, SoM and P-text are stable across all 30 of their
paired tasks. P-SoM is excluded from that row because its early directory is a restarted
partial of the full run rather than the pilot draw, and is reported alone: 3 one-directional
flips over 26 shared tasks, a pattern that reads as state drift rather than symmetric noise.*

Two properties of this reference matter for how §3.3 uses it. It is an **upper bound**: none of
the six cells in Table 1 has a replicate of its own, so the band is transferred across cells,
and locally served cells may sit lower than the API-served pairs. And it is run-to-run variation
**including environment drift**, not decoding stochasticity — replicates are separated by days,
and one sits on a site whose per-task reset is a no-op, so accumulated site state contributes
alongside the model. For the §3.3 comparison that is the right quantity, because the conditions
in Table 1 were also collected over days on the same infrastructure. It also means the locally
served row is not evidence that a deterministic decoder yields deterministic episodes: greedy
decoding is bit-reproducible at the step level in our own checks, and the episode is not,
because the episode is not only the decoder.

## B. Derivations for the four relabelling routes

§6 states the outcome of each route. This appendix gives the derivation.

### B.1 Continuous labels

The cleanest fix would be to regress on a graded quality signal rather than classify a discrete
winner: partial credit turns every episode into a training example regardless of whether it
succeeded. VisualWebArena does not provide one. Across the 7,686 scored episodes of our 36 landed
conditions the evaluator emits exactly two values, 0.0 and 1.0, in a 7,041 / 645 split, as do
the 36 protocol-excluded episodes we do not score. This is a property of the benchmark's
evaluation design rather than of our pipeline, and it forecloses the route entirely.

### B.2 Coarser classes

§4.2's obstruction is that four of six cells have fewer than ten labelled rows in more than one
class. Merging classes does not add rows. The binary collapse of §6 does help, but for a
different reason, having to do with agreement across backbones rather than with class count.

### B.3 Pooling across backbones

Six cells at 15–97 labels become 260 pooled examples and every class clears the minimum-count
filter, so supply is solved. Identifiability is not. The features carry no model identity, so
two backbones facing the same task on the same site produce near-identical feature vectors, and
where they are identical and the oracle labels differ, a classifier is being asked to emit two
different answers for one input. The figure reported in A.4 is the accuracy of emitting the modal
label per group, scored on the rows the label came from. It is an in-sample bound on what a
pooled classifier could reach, not a Bayes ceiling, and A.4 gives the resubstitution caveat.

They are only near-identical, not identical, because `dom_complexity`, `text_length` and
`tokens_input_text` are read from the backbone's own step-0 observation rather than from the
task config. On 31.5% of shared classifieds tasks and 80.0% of shared reddit tasks the rows
therefore differ somewhere. Grouping by the exact vector rather than by task raises the figure
(A.4, last column), and we report but do not adopt that number: it leaves most groups with one
member, and a group of one is scored perfectly whatever the labels do, so it inflates with
feature sparsity rather than tracking identifiability. Every version of the number is an
optimistic bound and all of them are far below what a deployable which-mode router needs, which
is the only thing the argument turns on.

### B.4 Screenshot tier

The tier label is derived from the same solve events as the which-mode label, by mapping each
oracle mode to image-bearing (SoM, Vision) or text-only (DOM, P-text, P-prompt, P-SoM). No new
episodes are involved, which is why the figure rises without a single new solve event. Part of
that rise is arithmetic — two classes admit a larger modal share than six — so the agreement
columns rather than the modal-agreement columns carry the claim. The tier is defined only on
tasks that some mode solved, so its denominator is the solved set and not the full task
universe. A.5's modal-agreement columns run over the whole pooled labelled set, as in A.4; the
two agreement columns are restricted to tasks labelled by two or more backbones, that being the
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

Two properties of that test are worth recording. It runs 10,000 draws and reports the plus-one
Monte Carlo estimator (k+1)/(B+1), whose floor is therefore 1.0 × 10⁻⁴, two orders below the
tightest Holm threshold of 8.3 × 10⁻³. That matters because at the 200 draws we first used, the
floor was 1/201 = 4.98 × 10⁻³, this cell reported exactly it, and whether it could clear the threshold at
all was a function of the draw count rather than of the data; at 10,000 draws four of the draws
match or beat the observed saving, so p = 5.0 × 10⁻⁴ is measured. Second, the quantity tested
is the saving at an operating point selected against whole-cell outcomes, which is not the
nested policy of Table 6. Null and observation select that point the same way, so the null
absorbs the selection optimism and the comparison is fair, but the point is not one a
deployment could occupy, and §5.3's conclusion rests on the nested numbers rather than on this
test.

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

*Table 14: Trainability under the two label definitions. "Surviving" counts classes clearing ten
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
