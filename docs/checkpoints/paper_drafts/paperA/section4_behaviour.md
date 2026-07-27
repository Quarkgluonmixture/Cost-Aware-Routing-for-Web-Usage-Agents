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

### 4.2 One measurement that constrains the story

The compact renumbering of the mark legend produces a **2.3–24.8× lower rate of hallucinated
element references** than the native accessibility-tree numbering. The ratio is per backbone,
not per cell: 0.04% against 0.39% at B0, 0.12% against 2.98% at B1, and 7.84% against 18.21%
at B2, so the reduction is largest at B1 and smallest at the weakest backbone.
Element-reference failures are a recognised bottleneck for multimodal web agents
[@zheng2024seeact; @zheng2024uground]. Native ids are sparse and non-semantic; the legend's ids run
1..K. Whatever else the text axis does, it makes the element-reference space smaller and
denser, and models refer outside it far less often.

This cuts against a natural reading of §3.2's asymmetry. If compact ids were the dominant
mechanism, axis-1 (the text axis) should carry more unique coverage than axis-2. It carries
less. So reference hallucination is real and measurable but is not what the text axis is
mostly doing for coverage.

### 4.3 The outer boundary of what representation can fix

We ran a deterministic failure-pattern scan over all 36 landed conditions and aggregated the
signatures cross-mode. The four highest-frequency failure signatures are **mode-invariant**:
they occur at comparable rates under every one of the six modes. Changing the representation
does not address them.

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
