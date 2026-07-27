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

### 4.2 One measurement, and which knob owns it

Hallucinated element references (actions naming an element id absent from the observation)
are a recognised bottleneck for multimodal web agents [@zheng2024seeact; @zheng2024uground].
Because the 2×2 was run, we can ask which knob moves them instead of assuming.

Neither knob owns the effect. P-SoM has the lowest rate in all six cells and P-prompt the
highest in five of six; at reddit · B2 the two differ by 33.06% of action-steps against 7.78%.
Substituting the mark legend for the accessibility tree lowers the rate in six of six cells
under the SoM prompt, but in only three of six under the DOM prompt. Moving from the DOM prompt
to the SoM prompt lowers it in six of six cells when the text is the legend, and raises it in
five of six when the text is the accessibility tree. Each knob's sign depends on the other
knob's setting, so this is an interaction and not a property of either payload alone.

What the pattern tracks is agreement between the id scheme the prompt advertises and the one
the text supplies. The two matched arms sit low and the mismatched arms high, worst at
P-prompt, where the prompt announces numbered marks and the text carries sparse native ids.
Part of that is mechanical: a prompt advertising ids 1..K induces references in that range,
which lie outside an accessibility tree. We therefore read it as a behavioural cost of
representational mismatch, not as an account of what drives coverage. The axis magnitudes of
§3.2 cannot supply that account either, because each axis differences a single-displacement arm
against P-SoM, leaving the *other* knob as the factor that varies inside the difference: axis-1
holds the mark legend fixed across both its arms, and axis-2 the SoM prompt.

### 4.3 The outer boundary of what representation can fix

We ran a deterministic failure-pattern scan over all 36 landed conditions, at one frozen
ruleset version, and aggregated the signatures cross-mode. The four highest-frequency
signatures (budget exhausted without finishing, degenerate element resolution,
perception-loss loops, and URL self-loops) are **near mode-invariant**: across the five
text-bearing modes their episode-level rates span 7.4 to 13.7 percentage points. Changing the
text representation does not address them. The single large excursion is degenerate element
resolution under Vision, 20.7% against 59.5% under DOM, where a coordinate action space has no
element ids to fail to resolve; the category lapses rather than the failure being fixed.

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
