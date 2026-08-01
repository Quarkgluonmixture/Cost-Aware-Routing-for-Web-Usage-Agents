<!--
CANDIDATE §1 UNDER THE 2026-08-01 REFRAME. Not in convert.sh's SECTIONS list.
Read side by side with `section1_intro.md` (the current draft) and pick one on 08-03.

WHAT CHANGED AND WHY
  The current draft's subject is six observation modes and whether a router can pick among
  them. Practitioners do not distinguish the four image-free variants; they distinguish
  vision from text, and they pay for the fused mode. Treating the four variants as separate
  objects of study answers a question nobody asks. So the six modes become the instrument
  and the three deployment channels become the subject:

      TEXT     dom, P-text, P-prompt, P-SoM   (four formalisations of one channel)
      VISION   vision                          (pixels, coordinate-addressed, no ids)
      FUSED    som                             (both, plus a mark-to-id correspondence)

  Consequences: (a) the H3 two-axis claim stops being a claim and becomes a construct-validity
  check on "text", which is the one framing under which its numbers survive; (b) the phantom
  taxonomy exposition is no longer needed, which is roughly the two pages of overflow;
  (c) WA stops being corroboration and becomes the contrasting end of the modality axis.

STILL OPEN AT TIME OF WRITING
  - No SoM replicate exists; the rerun floor is measured on DOM and Vision. Queued on A100
    behind the B0 x WA chain, expected 2026-08-03 (logs/som_replicate/).
  - The negative half of the fusion claim rests mainly on WA. See the hedge in 1.2.
  - Two workloads. Two points do not test a rule, and nothing queued changes that.
-->

## Abstract

A deployed web agent has a budget of model calls per task, and more than one way to spend it:
different observation channels, or the same channel more than once. We measure both in the same
units on VisualWebArena and WebArena, across three backbones, six observation modes and 8,310
scored episodes, and report a negative result about routing together with a positive one about how to
spend. Three findings. First, the ceiling that motivates representation routing is partly
counterfeit: adding a second, different representation raises the per-task oracle by 1.97 to
8.65 points, and simply rerunning the representation already in hand raises it by 2.0 to 7.6,
so on classifieds with the strongest backbone 69% to 106% of the apparent gain is reproducible
without changing representation at all. Second, the mode the field defaults to does not earn
its price: the fused mode carrying both an annotated screenshot and a mark legend is the
dearest of the six in five of six cells, its accuracy advantage over the better of the two
single-channel alternatives never clearly exceeds the measured rerun floor, it is negative in
two cells, and at matched arm count its uniquely solved tasks do not outnumber those of the
unannotated screenshot alone. Third, which channel to add reverses with the workload: on
image-specified tasks a second visual view beats adding text, while on text-specified tasks
adding text beats a second visual view by 8.65 against 2.88 points, clear of that benchmark's
rerun floor. None of this is reachable per request. Four routing formulations fail, including a
cascade escalating on the cheap run's own decoder confidence, a post-action signal strictly
richer than the pre-action features the other three use, and no operating point of it
Pareto-beats the fixed policy of always paying for the expensive mode. We give the reason that
does not depend on our sample size: on the cell where replicates exist, 51% of the tasks on
which the routing choice is even contested are tasks whose outcome we measured to change when
the same configuration is rerun. Representation choice is a deployment-time configuration
decision whose payoff must be netted against repetition, not a per-request routing opportunity.

## 1. Introduction

### 1.1 The decision a deployment actually faces

A multimodal web agent can observe a browser as text, as pixels, or as both
[@zhou2024webarena; @koh2024visualwebarena; @deng2023mind2web]. The accessibility tree can be
read as text. A screenshot can be looked at directly [@he2024webvoyager]. Between them sits a
screenshot annotated with numbered marks, paired with a legend naming them [@yang2023som], and
this fused representation is what most current systems ship.

The question a deployment asks is not which of these is best in the abstract. It has a budget
of model calls per task and has to spend it. It can spend one call on one channel. It can spend
two on two different channels. It can spend two on the *same* channel, which costs the same and
is almost never reported. Cost-aware model selection treats the first two options as the whole
menu [@chen2023frugalgpt; @ding2024hybridllm; @ong2025routellm; @gupta2024cascades;
@moslem2026routingsurvey], and its transfer to multi-step agents inherits that framing
[@wang2026boundaryrouter; @li2026dmr].

Putting the third option on the menu is not our idea. @hajimiri2026budgetmatched give three
online augmentation methods a token-matched vanilla baseline that simply spends the budget on
more actor steps, and find that the vanilla baseline matches or surpasses all three, concluding
that run-to-run variance should be reported as a core evaluation criterion. We ask their
question of a different axis. Augmentation modules are added to an agent; a representation is
what the agent *is*, it is the axis the routing literature is built on, and it is the one place
where the matched alternative is not "more steps" but the identical configuration run again.

Throughout we treat the six observation modes we run as instruments for three deployment
channels. **Text** is the accessibility tree or a mark legend, with either prompt family, four
combinations in total. **Vision** is an unannotated screenshot addressed by coordinates, with no
element identifiers. **Fused** is the annotated screenshot with its legend, the only condition
in which a mark in the image and an identifier in the text refer to the same object. Section 2.3
reports the four text formalisations separately and shows that the conclusions below do not turn
on which is chosen; we group them because practitioners do not distinguish them.

### 1.2 Three findings about how to spend

**The ceiling is real and partly counterfeit.** An oracle picking the best channel per task
beats the best single mode by 3.45 to 16.07 points. That number motivates every routing paper,
including the one we set out to write, and it is the wrong baseline. A union over arms grows
whenever any arm is added, including an arm that adds no capability, such as a second run of a
mode already on the menu. That repetition is not a small effect in this literature: a
screenshot-only web agent reports pass@4 of 94.7% against pass@1 of 78.2% on WebVoyager
[@gupta2026molmoweb], and 23 repeats of one tool-calling evaluation span 18.9 points, enough to
reorder a leaderboard [@bhat2026benchmarkingbenchmarks]. Those numbers are usually presented as
test-time scaling or as a reliability problem. We read them as the baseline a representation
gain has to clear, and measure both sides in the same units. Measured against same-condition replicates at matched arm count, one
extra representation buys 1.97 to 8.65 points and one extra *rerun* buys 2.0 to 7.6 (§3.3). On
classifieds with the strongest backbone the two are not separable: 7.14 against a 4.91 to 7.59
band. On WebArena they are, 8.65 against 2.00 to 4.00. The correct statement of the routing
opportunity is therefore a difference of differences, and it is much smaller than the headline.

**The default is the expensive answer, and it does not earn the difference.** The fused mode is
dearest in five of six cells, because the bill is driven by tokens per step and it carries both
payloads. Evidence against its necessity is accumulating from the modelling side: an 8B
screenshot-only policy with no HTML or accessibility tree now surpasses set-of-mark agents built
on much larger closed models [@gupta2026molmoweb]. We come at it from the accounting side, with
the same backbone on both arms. Its accuracy advantage over the better of the two single-channel alternatives is
+2.23, +1.79, +1.48, +0.49, +0.00, -2.96 points across the six VWA cells and -2.88 on
WebArena. The largest of these exactly equals the upper end of the measured rerun band, so
**no cell shows a fusion advantage that clearly clears the floor**, while the premium ranges
from +2.5% to +17.7% (§4.2). Nor does fusion appear to contribute a distinct kind of coverage:
at matched arm count its uniquely solved tasks number 1 to 10 against the unannotated
screenshot's 1 to 9, one cell ahead, three level, three behind. We state this as a bound rather
than a null. The counts are small, no interval accompanies them, the two cells in which fusion
is outright worse are one WebArena cell and one cell whose absolute success rate is 0.99%, and
the rerun floor itself is measured on DOM and Vision rather than on the fused mode.

**Which channel to add reverses with the workload.** Adding one arm on top of the strongest
single visual mode, against the reference of adding a *second* visual arm: on classifieds the
second visual arm wins for three of four text formalisations in every cell, and on WebArena
reddit the text arm wins for four of four, by 8.65 against 2.88 points, clear of that
benchmark's rerun floor by 4.65 (§4.3). This is one rule seen twice, not a claim that failed on
one benchmark. That the best representation is conditional rather than fixed is itself known:
@enomoto2026readmore show compact accessibility trees suit lower-capability models while verbose
HTML suits higher-capability ones, and recommend selecting adaptively. Their two arms are both
text and their conditioning variable is the model; ours are text against pixels against their
fusion, and the conditioning variable is the workload. VWA specifies 40.0% of its goals with a reference image and WebArena specifies
none, so the two span a task-modality axis, and the sign of the comparison follows the axis. The
mechanism is visible in the setup: the task's reference image is delivered to all six modes, so
what the text-only modes lack is the *page* screenshot, and a task carrying its own reference
image is precisely the one that does not need one.

### 1.3 Why none of it is reachable per request

Four formulations, three of them reported here for the first time in this setting.

**Choosing the channel is not learnable because labels are produced at the success rate.** The
identity of the cheapest successful mode exists only for solved tasks, so the six cells yield
15 to 97 labels each over six classes, and four of six admit no trainable classifier under a
minimum-class filter (§5.1). Collapsing six classes to the binary text-or-vision decision does
not repair it, because supply was never limited by class count: only 4 to 33 tasks per cell are
ones on which the two channels disagree.

**Choosing whether to spend is learnable and loses to a fixed policy.** The triage label is
defined everywhere and predictable at cross-validated AUROC 0.651 to 0.717 in five of six
cells, yet under fully nested cross-validation no cell's learned policy Pareto-dominates always
taking the cheapest mode, and the accuracy gain's interval covers zero in every cell (§5.2).

**Escalating on the agent's own confidence also loses to a fixed policy.** Every step record
carries the decoder's mean and minimum log-probability and margin. A cascade that runs the
cheap channel and escalates the least confident fraction to the fused one uses a strictly
larger information set than any pre-action router, which forecloses the reply that our features
were weak. The oracle version of this cascade is the most attractive operating point we
measure, paying double on only 2 to 22 tasks to buy 2.2 to 10.8 points for 2% to 12% more.
A deployable signal recovers 0% to 50% of that headroom, and **no operating point in any of the
four non-degenerate cells Pareto-beats simply running the fused mode on everything** (§5.3).
The best fixed signal is not a confidence statistic at all but the episode's step count, at
+0.5 to +1.0 points over a size-matched random escalation; the log-probabilities manage +0.1 to
+0.4. Because the threshold is swept rather than selected out of fold, and the signal chosen per
cell against realised outcomes, these are upper bounds, which only strengthens the reading.

**Half the contested labels are not stable.** The three failures above are about supply,
predictability and operating points, and each invites the answer that a larger benchmark or a
better estimator would fix it. One obstruction does not. On the cell where two same-condition
replicates exist, 49 of 224 tasks change outcome between runs, and of the 88 tasks on which the
channels disagree, and which are therefore the only tasks a router can learn from, **45 are
tasks we measured to flip** (§5.4). This is a lower bound: two of six arms were replicated, once
each. A router asked to predict which channel wins is being fitted on labels that a rerun
rewrites, and no amount of data repairs a target that is not reproducible.

### 1.4 Contributions

1. **The repetition-netted comparison, carried to the representation axis** (§3). The
   budget-matched critique is @hajimiri2026budgetmatched's and we do not restate it as new. What
   is new here is the axis and the matched arm: for representations the honest alternative is the
   identical configuration rerun, which lets us net the two at matched arm count under one
   functional, and the netted quantity is what §4 then reports.
2. **A price audit of the field's default representation** (§4.2). Fusion is the dearest of six
   modes and we cannot show it clearing the repetition floor on any cell.
3. **A workload-dependence result with a sign change** (§4.3), and the design that exposes it:
   two benchmarks that differ in whether goals are specified visually.
4. **A four-way negative on per-request routing** (§5), including the confidence cascade, and a
   reproducibility-derived upper bound on any per-request router that does not depend on our
   sample size.

Two methodological points transfer. A ranking metric can be high while the policy built on it
saves nothing, because at a 2% to 27% base rate the outcome is decided by a handful of tail
tasks; this has been observed in adjacent agent settings [@li2026aucnotenough]. And the baseline
that decides whether a router is worth deploying is never the best single mode. It is whichever
fixed policy the router is trying to replace, and we find two of them, always-cheapest and
always-fused, each beating a different learned or heuristic router.

### 1.5 Scope

Six modes, three backbones, two sites of VisualWebArena and the reddit split of WebArena,
8,310 scored episodes (7,686 + 624). Cost is comparable within a cell only: the proxy-served backbone reports
a bill and the locally served ones an electricity-derived figure [@strubell2019energy], which is
also why we do not attempt to route across backbones. Two workloads is enough to show a sign
change and not enough to characterise the axis along which it changes. The rerun floors come
from replicates separated by days on a site whose per-task reset is a no-op, so the quantity is
run-to-run variation including environment drift rather than decoding stochasticity; §3.3 argues
this is the right quantity for the comparison and names it accordingly. Greedy decoding is
bit-reproducible at the step level in our checks and the episode is still not reproducible,
because the episode is not only the decoder.
