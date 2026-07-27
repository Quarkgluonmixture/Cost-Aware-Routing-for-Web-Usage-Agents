**Two sites, one benchmark.** Site-level and benchmark-level effects are not separable at
N = 2 sites. The cross-site asymmetry in §3.3 maps where phantom arms help; it does not
identify why, and alternative explanations (page density, task-type mix, baseline success
variance) are uncontrolled.

**Serialization dependency.** The mark legend is a regex filter over accessibility-tree text
that already carries numeric element ids. Environments without pre-assigned ids need an
adapter, and the drop-in property does not transfer to them unexamined.

**Backbone contrasts are not clean ablations.** B0 against B1 mixes capability scale with
deployment class (API against local). B1 against B2 matches parameter count but not
capability. The 4B cross-family cell operates at 0.5–4% success, near the floor at which
mode comparisons still discriminate; we report it and scope around it rather than dropping it.

**The floor rule flatters the hypothesis it constrains.** As §3.1 notes, the degenerate-cell
SE floor moved θ_FE upward by de-weighting the smallest-effect cells. Both the preregistered
rule and its superseded predecessor fail the gate, so no decision turns on this, but a
reader recomputing from the per-cell table should know which rule produced the pooled number.

**No same-mode null for the structural test.** Each H3 axis is a set difference between two
arms, and a set difference between stochastic runs is non-zero even when both runs come from
one policy. Our interval is a paired bootstrap over tasks, so it reflects which tasks were
sampled and not how one task's outcome varies across repeats of the same condition. Every
condition here is a single rollout at temperature 0, which removes decoding variance but not
the environment's [@he2025nondeterminism]. No same-mode replicate exists in our data to
calibrate against, so both axes are separated from zero by the preregistered gate rather than
from a measured noise floor, and axis-1 at 1.35 pp has the less comfortable margin of the two.
The nearest repeat in our archive is a truncated pre-fix reddit DOM run: on its 148-task
overlap with the canonical run the two disagree on ten tasks, but nine fall the same way and
the success count rises from 12 to 20, so it measures a repaired defect rather than run-to-run
variation. Repeating one cell in one mode is the first experiment we would run.

**Mechanism is absent.** §4 reports behavioural correlates, not mechanisms. Activation-level
analysis of the two-knob account is not in this paper.

**A preregistered pass produced no data.** The learned-router second pass was designed and
never run (§4.4). We disclose it rather than let the preregistration imply results.

**A post-hoc exclusion.** Two reddit tasks were removed from the scored set after all six
modes had run in all six cells and after we had inspected the outcomes (§2.3). We label it as
post-hoc rather than preregistered. It moves the H1 estimand against this paper's own
hypothesis, and the sensitivity to including both tasks is in Appendix D.
