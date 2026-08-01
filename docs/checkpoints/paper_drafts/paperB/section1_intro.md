## Abstract

Cost-aware routing promises to cut the price of multimodal web agents: send each task
to the cheapest observation mode that can solve it. We test that promise on
VisualWebArena with six observation modes, three model backbones, and two sites, and
report a negative result with a diagnosis. The ceiling is real and splits in two: an oracle
that picks the cheapest successful mode per task gains 3.4 to 16.1 points of success rate over
the best single mode while spending 13.7% to 35.3% less, and the accuracy-neutral half of that
saving needs only a binary label where the accuracy-bearing half needs a per-mode one. The
accuracy half is also less solid than it looks: measured against same-condition replicate runs,
one extra representation buys what one extra rerun of the same mode buys, on two backbones and
two benchmarks. Neither half survives supervision. Choosing *which* mode fails on supply: a label exists only where
some mode succeeded, so at 7–43% solvable rates the six cells yield 15–97 labels each, and
four of six admit no trainable classifier under a minimum-class filter. Choosing *whether* to
spend has the supply and fails anyway: the label is defined everywhere and predictable at
AUROC 0.65–0.72 in five of six cells, yet under fully nested cross-validation no cell's
learned triage Pareto-dominates always taking the cheapest mode. Relabelling rescues neither.
The benchmark's score is binary, so there is no partial credit to regress on, and pooling
across backbones restores supply while breaking identifiability, with 56–57% of shared tasks
carrying contradictory labels. Asking only whether a task needs the screenshot does help,
raising the attainable ceiling from 79–84% to 90–97% while manufacturing no new solve events, and so does dropping the winner label for a per-mode success target, though that one has so far won only at a post-hoc threshold and failed on transfer. What binds is the production rate of *winner* labels, a property of the benchmark regime and of that target; we do not show that no estimator can route at this operating point, and §6 gives the one we could not close.

## 1. Introduction

A multimodal web agent can observe a browser in more than one way
[@zhou2024webarena; @koh2024visualwebarena; @deng2023mind2web]. The accessibility
tree can be read as text. A screenshot can be looked at directly [@he2024webvoyager]. Between
those two sits a screenshot annotated with numbered marks, paired with a textual legend that
names them [@yang2023som].

These modes differ in what they cost, and not in the direction the design suggests. What
dominates the bill is how many tokens a mode sends per step, which is why the mode carrying an
unannotated screenshot and no accessibility tree turns out to be the cheapest of the six in
every cell we measure, below every text-only mode. Cost is worth routing on; "does it carry an
image" is not the axis that predicts it.

That cost difference invites an obvious idea. If modes differ in price and each solves a
somewhat different set of tasks, then a router that predicts the right mode per task
should buy accuracy at a discount. Cost-aware model selection is an active area for
single-turn LLM serving [@chen2023frugalgpt; @ding2024hybridllm; @ong2025routellm;
@gupta2024cascades; @moslem2026routingsurvey], and its transfer to multi-step agents is
beginning to be studied [@wang2026boundaryrouter; @li2026dmr]. We set out to build such a
router for web agents and to measure what it bought.

It bought nothing, and we can say precisely why.

The first thing to establish is that the opportunity is real, because a negative routing
result is uninteresting if the modes are redundant. They are not. An oracle that sees the
outcome and picks the cheapest successful mode per task beats the strongest single mode on
both axes at once, by 3.45 to 16.07 points of success rate and 13.7% to 35.3% of cost (§3).

The accuracy side needs one calibration, which we supply rather than assume. A union over arms
grows when any arm is added, including one that adds nothing, such as a rerun of a mode already
in the menu. Against same-condition replicates, one rerun raises the same ceiling by 2.0 to 7.6 points
and the best *different* representation raises it by 4.8 to 7.1 (§3.3): at the margin, not
separable. The cost half is untouched, and the negative result is only reinforced.

What matters for the rest of the paper is that this ceiling decomposes, and the two pieces
are not reachable by the same supervision. Deciding only whether to spend, without changing
which mode solves anything, recovers the cost and needs one bit per task. Choosing among the
modes that actually solved a task recovers the accuracy and needs to name a mode. One
obstruction closes each piece.

**Choosing which mode is not learnable, because labels are not produced.** The natural
supervision is the identity of the cheapest mode that solved each task, and it exists only
for tasks that were solved. At the success rates these agents reach that yields tens of
examples per cell over six classes, and under the minimum-class threshold any practitioner
would apply before fitting a multiclass model, four of six cells contain no trainable
classifier at all (§4). This is a property of that target: the winning mode's identity comes
into existence only when the agent succeeds, and no re-slicing of it changes how many successes
the benchmark produced. A per-mode binary target escapes the scarcity and is the one route we
leave open (§6).

**Choosing whether to spend is learnable and still loses to a fixed policy.** A weaker router only decides,
per task, whether to pay for the expensive mode. Its label asks whether the task is solvable
by anything, so it is defined everywhere, and it is predictable above the best single
covariate in most cells. Under fully nested cross-validation, where the mode choice, the
threshold, and the scoring model are all re-derived inside each outer fold, **no cell's
learned triage Pareto-dominates the trivial fixed policy of always taking the cheapest
mode** (§5). The one cell whose saving survives multiple-testing correction has an AUROC
below chance, and its saving comes from tail enrichment rather than from a globally ordered
score.

**Relabelling does not rescue either half.** Regressing on a graded quality signal is
impossible because the benchmark's score is binary. Pooling labels across backbones fixes
supply and breaks identifiability in exchange, since the features carry no model identity and
more than half of the tasks shared between cells then carry contradictory labels. Two re-slicings survive: asking per mode whether it will
succeed, which escapes the scarcity but has so far won only at a post-hoc threshold; and asking
only whether the task needs the screenshot, which buys a materially higher ceiling from the same
features and the same solve events (§6).

Our contribution is therefore not a router. It is an account of why this router is
unavailable at this operating point, stated in terms that transfer:

1. A ceiling measurement that separates the accuracy-neutral, cost-bearing part of the
   routing opportunity from the accuracy-bearing part, and shows that the two parts need
   different supervision (§3).
2. A supply argument showing that mode-selection supervision is produced at the success
   rate, and a demonstration that this is binding rather than incidental (§4).
3. A triage result showing that adequate supply and adequate predictability are jointly
   insufficient, with a fixed policy as the baseline that matters (§5).
4. A relabelling analysis that closes three obvious escape routes and identifies the two that
   remain open, together with the identifiability cost of pooling (§6).

Two methodological findings generalise beyond this study. First, in a low-success regime a
high AUROC is neither necessary nor sufficient for a triage policy to save money; what decides
the outcome is whether a handful of tail tasks are routed to the expensive mode, and at
n = 203 that handful is four successes. A ranking metric that does not license the downstream operational
claim has been reported in adjacent agent settings as well [@li2026aucnotenough]. Second, the
comparison that decides whether a learned router is worth deploying is not against the best
single mode but against the cheapest one, and switching baselines removes every saving we
measured.
