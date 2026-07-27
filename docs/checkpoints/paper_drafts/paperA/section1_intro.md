## Abstract

Multimodal web agents bundle three things into what gets called an observation mode: the
text payload, the prompt family, and whether a marked screenshot is attached. We unbundle
them. Starting from a DOM baseline, varying the text payload and the prompt family
independently while holding the screenshot off produces three arms we call the phantom
routing space: P-text (mark legend under a DOM prompt), P-prompt (accessibility tree under
a SoM prompt), and P-SoM (both). Across six preregistered cells (two VisualWebArena sites
by three backbones spanning 4B to 235B parameters and two model families), we test whether
this space contains a superior deployment arm and whether it has internal structure.

It does not contain a superior arm. The preregistered superiority test for P-SoM's
drop-one oracle contribution fails: pooled θ_FE = 0.7897 pp against a margin of 1.0 pp,
one-sided p = 0.807. The per-cell effects are consistently small rather than dispersed
(I² = 0.0%, Cochran Q = 1.43 on 5 df), so the failure is not a power problem that more
cells would fix.

It does have structure, and that is the paper's result. Both preregistered decomposition
axes pass: the tasks solved by P-text but not P-SoM pool at θ_FE = 1.3528 pp (bootstrap
CI [0.799, 2.026]), and the tasks solved by P-prompt but not P-SoM pool at 2.0877 pp
(CI [1.399, 2.919]). The compound arm does not absorb either single-axis arm. Text format
and prompt family are therefore separately consequential rather than two names for one
intervention, which is what makes the region a space rather than a point. P-SoM in turn
uniquely solves 6 classifieds and 3 reddit tasks that none of the other five modes solves.
Cost stays within the preregistered band of the DOM baseline in all six cells across 1,281
paired tasks, since the mark legend is a regex filter over text the DOM agent already
consumes.

We report the failed superiority test as the primary Phase 1a outcome rather than as a
caveat, because the preregistered decision rule makes it one, and because the structural
result stands without it.

## 1. Introduction

A web agent's observation mode is usually named as one thing. In practice it is at least
three: what text the model receives, how the prompt describes that text, and whether a
screenshot with numbered marks is attached. Systems fix all three together, so a comparison
between modes is a comparison between bundles.

This paper takes the bundle apart on one boundary: the screenshot stays off. From a DOM
baseline (accessibility tree, DOM prompt, no image) two knobs remain. Swapping the text
payload for the Set-of-Mark legend while keeping the DOM prompt gives **P-text**. Keeping
the accessibility tree while switching to the SoM prompt gives **P-prompt**. Doing both
gives **P-SoM**. We call these three arms the **phantom routing space**: they use the
vocabulary of marked-screenshot prompting without the marked screenshot.

P-SoM began as a control we expected to be broken. The SoM prompt tells the model it will
receive an annotated image; withholding the image while keeping that prompt should either
collapse the arm into a worse DOM or produce incoherent behaviour. Archived runs did not
behave that way, which is why the configuration became an object of study rather than a
discarded ablation.

Two questions follow, and they have different answers.

**Is there a better arm here?** No. The preregistered hypothesis was that P-SoM
contributes drop-one oracle value above a 1.0 pp margin when pooled across the six cells.
It contributes **0.7897 pp** (paired-bootstrap pooled median 0.7490 pp, 95% CI [0.2858,
1.4471]) and the one-sided test returns **p = 0.807**. The gate fails. What matters for
interpretation is the shape of the failure: **I² = 0.0%** with Cochran Q = 1.43 on 5
degrees of freedom, so the six cells agree that the effect is small. This is not a case of
one cell dragging a pool, and it is not a case of wide intervals hiding an effect.
Additional cells would tighten the interval around a value below the margin.

**Does the space have internal structure?** Yes, and both axes clear their gates by a wide
margin. The preregistered decomposition asks whether either single-axis arm solves tasks
the compound arm misses. Axis-1 counts tasks in P-text but not P-SoM and pools at
**θ_FE = 1.3528 pp** with bootstrap CI **[0.799, 2.026]**; axis-2 counts tasks in P-prompt
but not P-SoM and pools at **2.0877 pp** with CI **[1.399, 2.919]**. Both survive Holm
correction over the two-axis family on the legacy Wald channel (p = 1.19 × 10⁻⁵ and
7.52 × 10⁻⁷). Five of six cells exceed the preregistered noise floor of two unique tasks
on each axis; the number of individually Holm-significant cells is three of six on axis-1
and four of six on axis-2.

The interpretation is the paper's main claim. **The compound arm does not absorb the
single-axis arms.** Switching the text format alone, and switching the prompt family alone,
each leaves behind tasks that switching both does not recover. Text payload and prompt
family are separately consequential. That is what makes this a space with axes rather than
a single configuration with a name, and it is a claim about structure that survives the
failure of the superiority claim about P-SoM specifically.

Complementarity remains measurable at the task level: **P-SoM uniquely solves 6 classifieds
and 3 reddit tasks** that no other of the six modes solves. We do not claim P-SoM replaces
full SoM. On classifieds, full SoM is clearly stronger as a single arm (27.23% vs 15.62%
at the 235B backbone), which is the sanity check one wants when marked screenshots carry
real visual information.

The cost profile is a consequence of the construction rather than a finding. The mark
legend is produced by a regex filter over the same accessibility-tree text the DOM agent
already consumes, followed by a sequential renumber; there is no bounding-box pipeline and
no per-step image encoding. Per-task cost stays inside the preregistered band around the
DOM baseline in **all six cells over 1,281 paired tasks**. We state this as an
architectural property, not as an empirical discovery, and we do not present it as a
cheaper substitute for SoM.

Our contributions:

1. An unbundling of the observation-mode construct into text payload and prompt family on
   the image-off boundary, with the 2×2 ablation that makes each separately testable (§2).
2. A preregistered negative result on P-SoM superiority, reported with the heterogeneity
   statistics that distinguish a small effect from an underpowered one (§3.1).
3. A preregistered positive result on structure: neither single-axis arm is absorbed by the
   compound arm, which establishes the region as decomposable (§3.2).
4. A behavioural account of what each knob changes, and a disclosure of what we did not
   measure, including a routing pass that was designed and never run (§4, §5).
