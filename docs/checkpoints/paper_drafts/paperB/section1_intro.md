## Abstract

Cost-aware routing promises to cut the price of multimodal web agents: send each task
to the cheapest observation mode that can solve it. We test that promise on
VisualWebArena with six observation modes, three model backbones, and two sites, and
report a negative result with a diagnosis. The routing ceiling is real — an oracle that
picks the cheapest successful mode per task matches the best single mode's success rate
at 13–22% lower cost. But the router cannot be learned, and it fails twice for two
different reasons. Choosing *which* mode fails on label supply: a training label exists
only where some mode succeeded, so at 7–43% solvable rates the six cells yield 15–97
labels each, and four of six cells have no trainable classifier at all under a
minimum-class filter. Choosing *whether* to spend fails despite having supply: the
solvable/hopeless label is available for every task and predictable at AUROC 0.65–0.72,
yet under fully nested cross-validation no cell's learned triage Pareto-dominates the
trivial policy of always taking the cheapest mode. We then show that relabelling cannot
rescue either half. The benchmark's score is binary, so there is no partial credit to
regress on; pooling across backbones restores supply but breaks identifiability, because
the routing features are functions of the task alone and 56–57% of shared tasks carry
contradictory labels. The one re-slicing that helps — collapsing six modes to a binary
"does this need the screenshot" tier — raises the attainable ceiling from 79–84% to
90–97% without manufacturing a single new solve event. We argue the binding constraint
on learned routing for web agents is the *production rate* of supervision, not the
hypothesis class, and that this constraint is a property of the benchmark regime rather
than of our estimator.

## 1. Introduction

A multimodal web agent can observe a browser in several ways. It can read the
accessibility tree as text, look at a screenshot, or receive a screenshot annotated
with numbered marks paired with a textual legend. These modes differ in what they cost:
rendering and encoding an annotated screenshot at every step is the dominant expense in
a vision-language agent loop, and a text-only mode avoids it entirely.

That cost difference invites an obvious idea. If modes differ in price and each solves a
somewhat different set of tasks, then a router that predicts the right mode per task
should buy accuracy at a discount. The idea is attractive enough that it appears as
future work in many agent papers, and cost-aware model selection is an active area for
single-turn LLM serving. We set out to build such a router for web agents and to measure
what it bought.

It bought nothing, and we can say precisely why.

The first thing to establish is that the opportunity is real, because a negative routing
result is uninteresting if the modes are redundant. They are not. An oracle that sees the
outcome and picks the cheapest successful mode per task matches the strongest single
mode's success rate while spending 13% less on classifieds and 22% less on reddit for
our largest backbone. The gap is entirely in cost rather than accuracy, which already
tells us something about where the achievable value lies: not in solving more tasks, but
in not overpaying for the ones that get solved.

The second thing is that this opportunity does not survive contact with supervision, and
it fails in two distinguishable ways.

**Choosing which mode is not learnable, because labels are not produced.** The natural
supervision for a mode-selection router is the identity of the cheapest mode that solved
each task. Such a label exists only for tasks that *were* solved. At the solvable rates
we observe — 43.3% down to 7.1% across six (site, backbone) cells — this yields between
97 and 15 labelled examples per cell, spread over six classes. Applying the same
minimum-class threshold any practitioner would apply before fitting a multiclass model,
four of the six cells contain no trainable classifier at all: fewer than two classes
survive. This is not a modelling failure that a better estimator repairs. Labels come
into existence only when the agent succeeds, and no re-slicing of the supervision changes
how many successes the benchmark produced.

**Choosing whether to spend is learnable and still worthless.** A weaker router only
decides, per task, whether to pay for the expensive mode or send the task to the cheap
one. Its label — is this task solvable by anything — is defined for *every* task, so the
supply problem disappears: 203 or 224 labels per cell instead of 15 to 97. And it is
predictable: cross-validated AUROC reaches 0.651–0.717 in five of six cells, clearing the
best single covariate in four. Under fully nested cross-validation, where the mode
choice, the decision threshold, and the scoring model are all re-derived inside each
outer fold, **no cell's learned triage Pareto-dominates the trivial fixed policy of
always taking the cheapest mode**. One cell yields a saving that survives multiple-testing
correction, and inspecting it is instructive: it sends 95% of its tasks to the cheap mode,
differing from the free fixed policy by five percent of the allocation, in a cell where
only 7.4% of tasks are solvable at all. Its AUROC is 0.483 — below chance. The saving
comes from tail enrichment, not from a globally ordered score.

**Relabelling does not rescue either half.** Three routes are available and all are
closed. Regressing on a continuous quality signal is impossible because the benchmark's
score is binary: across 7,963 episodes we observe only 0 and 1, with no partial credit.
Pooling labels across backbones does fix supply, and it breaks identifiability in exchange:
the routing features are computed from the task and the first observation, carrying no
model identity, so two backbones facing the same task present the same feature vector.
On tasks covered by more than one cell, 57.4% (classifieds) and 56.0% (reddit) carry
contradictory labels, capping any task-feature classifier at 79.2% and 83.7% accuracy
respectively. The one re-slicing that pays is to stop asking which of six modes and ask
only whether the task needs the screenshot: the same features, the same solve events, and
a ceiling of 89.9% and 96.7%. Backbones disagree sharply about *which* mode is best and
agree substantially about *whether the image is needed*.

Our contribution is therefore not a router. It is a characterisation of why this
particular router is not available at this operating point, stated in terms that transfer:

1. A ceiling measurement that separates the accuracy-neutral, cost-bearing part of the
   routing opportunity from the accuracy-bearing part (§3).
2. A supply argument showing that mode-selection supervision is produced at the success
   rate, and a demonstration that this is binding rather than incidental (§4).
3. A triage result showing that adequate supply and adequate predictability are jointly
   insufficient, with a fixed policy as the baseline that matters (§5).
4. A relabelling analysis that closes the obvious escape routes and identifies the one
   that is open, together with the identifiability cost of pooling (§6).

We also report two methodological findings that we believe generalise beyond this study.
First, in a low-success regime, a high AUROC is neither necessary nor sufficient for a
triage policy to save money; what decides the outcome is whether a handful of tail tasks
land on the correct side, and at n=203 that handful is four successes. Second, the
comparison that determines whether a learned router is worth deploying is not against the
best single mode but against the cheapest single mode, and switching to that baseline
removes every saving we measured.
