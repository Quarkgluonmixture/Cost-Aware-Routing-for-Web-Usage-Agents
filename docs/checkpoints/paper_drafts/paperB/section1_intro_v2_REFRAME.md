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

A deployed web agent has a budget of model calls per task and more than one way to spend it:
different observation channels, or the same channel more than once. We measure both in the same
units across two benchmarks, three backbones, six observation modes and 8,310 scored episodes.
Three findings. The ceiling that motivates representation routing is partly counterfeit: a
second, different representation raises the per-task oracle by 1.97 to 8.65 points and a rerun
of the representation already in hand raises it by 2.0 to 7.6, so in our largest cell 69% to
106% of the apparent gain needs no change of representation. The mode the field defaults to
does not earn its price: the fused representation is dearest in five of six cells, yet its
advantage over the better single-channel alternative clears the measured rerun floor in none of
them and is negative in two. And which channel to add reverses with the workload, from a second
visual view on image-specified tasks to the text channel on text-specified ones, there by 8.65
against 2.88 points and clear of that benchmark's floor. None of it is reachable per request:
four routing formulations fail, including a cascade escalating on the cheap run's own decoder
confidence, a post-action signal strictly richer than the pre-action features the others use,
whose every operating point still loses to always paying for the expensive mode. The reason
does not depend on our sample size. Where replicates exist, 51% of the tasks on which the
routing choice is contested are tasks we measured to change outcome when the same configuration
is rerun. Representation choice is a deployment-time configuration decision whose payoff must be
netted against repetition, not a per-request routing opportunity.

## 1. Introduction

### 1.1 The decision a deployment actually faces

A multimodal web agent can observe a browser as text, as pixels, or as both
[@zhou2024webarena; @koh2024visualwebarena; @deng2023mind2web]. The accessibility tree can be
read as text. A screenshot can be looked at directly [@he2024webvoyager]. Between them sits a
screenshot annotated with numbered marks, paired with a legend naming them [@yang2023som], and
this fused representation is what most current systems ship.

The question a deployment asks is not which of these is best in the abstract. It has a budget
of model calls per task and has to spend it. It can spend one call on one channel, or two on two
different channels. It can also spend two on the *same* channel. That third option costs what
the second one costs, and it is almost never reported alongside it. Cost-aware model selection
treats the first two as the whole menu [@chen2023frugalgpt; @ding2024hybridllm; @ong2025routellm; @gupta2024cascades;
@moslem2026routingsurvey], and its transfer to multi-step agents inherits that framing
[@wang2026boundaryrouter; @li2026dmr].

Putting the third option on the menu is not our idea: a token-matched baseline that simply buys
more actor steps already erases the gains of three online augmentation methods
[@hajimiri2026budgetmatched]. We ask that question of a different axis. Augmentation is added to
an agent, whereas a representation is what the agent *is*, and it is the one axis where the
matched alternative is not more steps but the identical configuration run again.

Throughout we treat the six observation modes we run as instruments for three deployment
channels. **Text** is the accessibility tree or a mark legend, with either prompt family, four
combinations in total. **Vision** is an unannotated screenshot addressed by coordinates, with no
element identifiers. **Fused** is the annotated screenshot with its legend, the only condition
in which a mark in the image and an identifier in the text refer to the same object. Section 2.5
shows that the four text formalisations are behaviourally non-separable across cells, which is
why we group them, and Appendix A.1 keeps them apart wherever a number would change; we group them because practitioners do not distinguish them.

### 1.2 Three findings about how to spend

**The ceiling is real and partly counterfeit.** An oracle picking the best channel per task
beats the best single mode by 3.45 to 16.07 points. That gap is what motivates a router, and it
is what motivated ours, but it is measured against the wrong thing. A union over arms grows
whenever any arm is added, including an arm that adds no capability at all, such as a second run
of a mode already on the menu. Repetition is not a small effect here: one
screenshot-only agent reports pass@4 of 94.7% against pass@1 of 78.2% [@gupta2026molmoweb], a
number offered as test-time scaling that reads equally well as the bar a representation gain
must clear. Measured against same-condition replicates at matched arm count, one
extra representation buys 1.97 to 8.65 points and one extra *rerun* buys 2.0 to 7.6 (§3.3). On
classifieds with the strongest backbone the two are not separable: 7.14 against a 4.91 to 7.59
band. On WebArena they are, 8.65 against 2.00 to 4.00. What a router could win is therefore the gap
between those two gains, not the headline gap, and on one of our two measured cells that
difference is indistinguishable from zero.

**The default is the expensive answer, and it does not earn the difference.** The fused mode is
dearest in five of six cells, because the bill is driven by tokens per step and it carries both
payloads, and it charges between +2.5% and +17.7% for them. Against comparators fixed in
advance, one arm each and paired over tasks, the fixed-effect pool across seven cells puts its
advantage at **+1.43pp [+0.12, +2.75] over the unannotated screenshot and +0.89pp [-0.40, +2.18]
over the accessibility tree**. The second interval does not exclude zero and neither clears the
measured rerun band of 0.89 to 2.23 points, so on this evidence **the premium buys less than a
repetition of what the deployment already has** (§4.2).

The per-cell structure says something the pool cannot. In each cell one single channel is the
stronger, the visual one on all three classifieds cells and the text one on all four reddit
splits, and **against that channel fusion's interval includes zero in six of seven cells and is
negative in the seventh**. It beats the channel that does not suit the workload, by 8 to 10
points over the accessibility tree on classifieds and 5 to 7 over the screenshot on reddit. Read
together with the reversal below, this is what fusion appears to be worth: not a capability
neither channel has, but insurance against picking the wrong one. Nor does it behave like an
added capability on coverage either. Its selling point is a correspondence between a mark in the
image and an identifier in the text that neither single channel has, and a mechanism of that
kind predicts tasks the others cannot reach; at matched arm count the tasks fusion alone solves
outnumber those the screenshot alone solves in one cell, tie in three and lose in three, on
counts that never exceed ten. That is a shape mismatch rather than a refutation, but it is not
the shape a distinct capability would produce. We state this as a bound rather
than a null, and §4.2 gives the reasons, the sharpest being that the rerun floor is so far
measured on the other two channels. One caveat we do *not* enter is the identity of the two
cells where fusion is negative. One of them scores 0.99% and carries no information either way.
The other is WebArena, which is the cell the next finding predicts fusion should lose.

**Which channel to add reverses with the workload.** We add one arm on top of the strongest
single visual mode and ask whether a text arm beats a *second* visual one. On classifieds the
visual arm wins, for three or four of the four text formalisations in every cell. On WebArena
reddit the text arm wins, for all four, by 8.65 points against 2.88, clearing that benchmark's
rerun floor by 4.65 (§4.3). This is one rule seen twice, not a claim that failed on
one benchmark. That the best representation is conditional rather than fixed is known
[@enomoto2026readmore], but there the conditioning variable is the model and both arms are text;
here it is the workload, across text, pixels and their fusion. VWA attaches a reference image to 33.7% of the goals we score and WebArena to
none, so the two span a task-modality axis, and the sign of the comparison follows the axis. The
mechanism is visible in the setup: the task's reference image is delivered to all six modes, so
what the text-only modes lack is the *page* screenshot, and a task carrying its own reference
image is precisely the one that does not need one.

### 1.3 Why none of it is reachable per request

Four formulations, and the fourth obstruction is of a different kind from the first three.

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
A deployable signal recovers 0% to 50% of that headroom, and how much it recovers turns out not
to be the binding quantity: **no operating point in any of the four non-degenerate cells
Pareto-beats simply running the fused mode on everything** (§5.3), so even perfect recovery of
the signal would be competing against a policy that needs no signal at all.
The best fixed signal is not a confidence statistic at all but the episode's step count, at
+0.5 to +1.0 points over a size-matched random escalation; the log-probabilities manage +0.1 to
+0.4. Because the threshold is swept rather than selected out of fold, and the signal chosen per
cell against realised outcomes, these are upper bounds, which only strengthens the reading.

**Half the contested labels are not stable.** The three failures above are about supply,
predictability and operating points, and each invites the answer that a larger benchmark or a
better estimator would fix it. One obstruction does not. On the cell where two same-condition
replicates exist, 49 of 224 tasks change outcome between runs, and of the 88 tasks on which the
channels disagree, and which are therefore the only tasks a router can learn from, **45 are
tasks we measured to flip** (§5.4). This is a lower bound, and the bound runs the
favourable way: two of six arms were replicated, once each, so replicating more of them can only
raise the fraction. A router asked to predict which channel wins is being fitted on labels that
a rerun rewrites, and no amount of data repairs a target that is not reproducible.

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
8,310 scored episodes. Cost is comparable within a cell only, the proxy-served backbone
reporting a bill and the locally served ones an electricity-derived figure
[@strubell2019energy], which is also why we do not route across backbones. Two workloads cannot characterise the axis the
reversal turns on, and two workloads with opposite signs are what it takes to establish that
there is a reversal at all: fitting the dependence needs many points, showing the quantity is
not a constant needs exactly two that disagree. We claim the second and not the first. §3.3 names the rerun
quantity precisely: it is run-to-run variation including environment drift, not decoding
stochasticity.
