## 4. What each knob changes

§3.2 establishes that the two axes are separately consequential. It does not say what they
do. This section reports the behavioural correlates we can measure from trajectories, and
is explicit that they are correlates.

### 4.1 A two-knob account

The 2×2 lets each knob be read against a matched control. Across cells the pattern is:

**Text representation appears to shape exploration.** Swapping the accessibility tree for
the compact mark legend changes how much of the page the agent visits before committing.
The legend is a flattened, renumbered view of the same elements, so the information is
nominally preserved while its presentation is not.

**Prompt family appears to modulate commitment timing.** Keeping the text fixed and moving
from the DOM prompt to the SoM prompt changes when the agent decides it is finished, which
shows up in finish-step index and in the rate of finishing before evidence is gathered.

We state both with hedges because the trajectory metrics that back them are descriptive.
The behavioural claim is that the two knobs have distinguishable behavioural signatures,
which is consistent with §3.2 finding them separately consequential; it is not a mechanism.

### 4.2 One measurement, reported descriptively

Hallucinated element references (actions naming an element id absent from the observation)
are a recognised bottleneck for multimodal web agents [@zheng2024seeact; @zheng2024uground],
and the 2×2 lets us see how they distribute over the four image-free arms. We report that
distribution and deliberately stop short of calling it an interaction.

P-SoM carries the lowest rate of the four arms in all six cells when the rate is taken per
action-step, and P-prompt the highest in five of six. Substituting the mark legend for the
accessibility tree lowers the rate in all six cells under the SoM prompt, and by more than the
same substitution achieves under the DOM prompt: at reddit · B2, 25.4 against 9.4 points.

Two reasons not to promote that to an interaction claim. First, most of it is forced. Whenever
P-SoM is the minimum arm and P-prompt the maximum, their difference is the largest of any pair
by construction, so "the reduction is larger under the SoM prompt" follows from the ordering
rather than from any joint effect; that configuration holds in five of six cells. Second, the
ordering itself is denominator-dependent. A rate per action-step weights an episode by how many
actions it took, so an episode deadlocking on one invalid id for thirty steps outweighs one
that misfires once. Counting instead the share of episodes with at least one such reference,
P-SoM is lowest in five of six cells and P-prompt highest in three of six. The comparison still
favours the SoM prompt in six of six under that denominator, and there only two of the six are
forced by the ordering, which is the strongest form in which we can put it.

Descriptively, the arms in which the prompt's advertised id scheme matches the one the text
supplies carry fewer such references, and part of that is mechanical: a prompt advertising ids
1..K induces references in that range, which lie outside an accessibility tree. We read it as a
behavioural correlate of representational mismatch, not as an account of what drives coverage.
The axis magnitudes of §3.2 cannot supply that account either, because each axis differences a
single-displacement arm against P-SoM, leaving the *other* knob as the factor that varies inside
the difference: axis-1 holds the mark legend fixed across both its arms, and axis-2 the SoM
prompt.

### 4.3 The outer boundary of what representation can fix

We ran a deterministic failure-pattern scan over all 36 landed conditions, at one frozen
ruleset version, and aggregated the signatures cross-mode. The four highest-frequency
signatures (budget exhausted without finishing, degenerate element resolution,
perception-loss loops, and URL self-loops) are **similar after pooling**: pooled over the six
cells, each one's episode-level rate differs across the five text-bearing modes by no more than
7.4 to 13.7 percentage points, depending on the signature.

That is a weaker statement than mode-invariance and we do not make the stronger one. Pooling
sums numerators and denominators across cells before the rate is formed, so cell-level
variation cancels: within a single cell the same four signatures span up to 15.1, 27.3, 48.3
and 36.1 points respectively. What survives is that no mode escapes these four signatures at
the pooled level, not that any individual cell shows them at equal rates. The largest single
excursion is degenerate element resolution under Vision, 20.7% against 59.6% under DOM, where a
coordinate action space has no element ids to fail to resolve; the category lapses rather than
the failure being fixed.

Two companion observations bound the space from the same direction. The rate at which agents
fail on comment-and-reply tasks is capped at a level no mode improves on. And adding the
marked screenshot back yields near-zero marginal gain in the cells where the text arms
already succeed, which matches reports that vision-language models often ignore much of the
image content [@tong2024eyes; @kaduri2024whatsintheimage; @zhou2026visualignorance].

Read together with §3.2, this gives the honest shape of the result. The phantom space is
structured, and its axes are separately real, but the region of agent failure that any
representation choice can reach is narrower than a drop-one oracle number suggests on its
own. We state this here rather than in the limitations section because a reader who takes
the oracle figures without it will over-read what routing could recover.

### 4.4 Router: designed, not run

The preregistered design included a second pass in which a learned router selects a mode per
task at run time, gated by a Pareto criterion against single-mode baselines, in the manner
of cost-aware routing for single-turn serving [@chen2023frugalgpt; @ong2025routellm].
**That pass was never executed.** Zero Pass-2 runs exist across all six cells.

We disclose this because the design document, the gating code, and the analysis pipeline for
it all exist and are cited in our preregistration, so a reader could reasonably assume
results followed. They did not. The offline analysis we can report is that the mode-selection
label is produced only where some mode succeeds, which at these success rates yields 15 to
97 labelled examples per cell, too few for a six-class model in four of the six cells. That
analysis, and the negative routing result it supports, is the subject of a companion paper;
we mention it here only to close the loop on a preregistered pass that produced no data.
