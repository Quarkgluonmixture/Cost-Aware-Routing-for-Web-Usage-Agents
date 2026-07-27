## 5. Related work

**Observation modes in web agents.** Benchmarks and agent systems in this area fix the
observation mode per system: DOM-derived text [@zhou2024webarena; @deng2023mind2web;
@agentoccam2025], screenshots [@he2024webvoyager], or Set-of-Mark annotated screenshots with
a textual legend [@koh2024visualwebarena; @zheng2024seeact]. Comparisons across modes are
therefore comparisons across systems, which bundles the text payload, the prompt family, and
the image together. Work on reducing the observation itself
[@enomoto2026observation; @schiepanski2025d2snap] varies how much text is sent while holding
the prompt family and the image fixed, which is one of our two axes rather than both. Our
contribution is to unbundle two of the three on a fixed image setting, and §3.2 is the
measurement that shows the unbundling was not vacuous.

**Set-of-Mark prompting.** SoM was introduced for the multimodal setting [@yang2023som],
pairing a marked image with a textual legend so that visual objects can be named by number.
Later multimodal-agent systems explored marked-screenshot and omni-modal grounding as system
designs [@zheng2024uground; @yang2025magma; @li2025ferretui2]. None treats the legend text as
a variable separable from the image, which is the configuration our P-text and P-SoM arms
occupy. That the arms function at all is less surprising in the light of evidence that
vision-language models often decide with limited reliance on the supplied image
[@tong2024eyes; @kaduri2024whatsintheimage; @zhou2026visualignorance;
@asadi2026mirageillusionvisualunderstanding], and that visually prompted evaluations are
fragile to exactly this kind of perturbation [@feng2025visually].

**Prompt-format sensitivity.** That language models are sensitive to prompt formatting
independent of content is established for single-turn tasks
[@sclar2024promptformat; @mishra2022reframing]. §3.2's axis-2 result is a multi-step agentic
instance: holding the text payload fixed and changing only how the prompt describes it leaves
behind a measurable set of tasks the compound change does not recover.

**Negative results with preregistration.** We report a failed preregistered superiority test
alongside a passed preregistered structural test, in a literature where progress claims for
web agents have been shown to rest on fragile evaluation
[@xue2025illusion; @elhattami2025webarenaverified; @lu2025agentrewardbench] and where the
distinction between a measured effect and an argued one is a recurring concern
[@lipton2018troubling]. The heterogeneity statistics in §3.1 [@higgins2002quantifying] are
what let us distinguish "small effect" from "insufficient power", and we regard reporting
them as the minimum for a negative result to be usable by anyone else.

## 6. Discussion

### 6.1 What we claim, and at what strength

The phantom routing space is structured: two axes, each separately consequential, neither
absorbed by their compound. This is a preregistered result at p = 1.19 × 10⁻⁵ and
7.52 × 10⁻⁷ over six cells with bootstrap CIs excluding zero.

P-SoM is not a superior deployment arm. This is a preregistered negative at p = 0.807 with
I² = 0.0%, which we report as the primary Phase 1a outcome because the decision rule makes
it one.

P-SoM's coverage is complementary but small: 6 classifieds and 3 reddit tasks solved by no
other mode. We claim complementarity, not superiority, and the two are not interchangeable.

### 6.2 Scope and limitations

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

**Mechanism is absent.** §4 reports behavioural correlates, not mechanisms. Activation-level
analysis of the two-knob account is not in this paper.

**A preregistered pass produced no data.** The learned-router second pass was designed and
never run (§4.4). We disclose it rather than let the preregistration imply results.

### 6.3 What the structural result is good for

If the two axes were redundant, the design space for image-free web agents would be a line
from DOM to P-SoM and the only question would be how far along it to sit. §3.2 says it is not a line: text payload and prompt
family can be chosen separately, and each choice reaches tasks the other does not.

What §4.3 adds is the boundary: the failure modes that dominate at these success rates are
mode-invariant, so representation choice operates on a narrower slice of the failure
distribution than the oracle arithmetic suggests. Both statements are needed. The space is
real and decomposable, and it is small.
