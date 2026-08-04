# Threats to validity

**Leaked successes.** `require_reset` is a no-op on reddit, so subscriptions
accumulate across a run's 205 episodes and a later task can be scored against state
an earlier one created. Six scored successes were credited without the episode ever
visiting the forum the evaluator reads. We set those to 0 with the denominator
unchanged --- an attempted and unaccomplished task is a 0, not a missing row.
Doing so moves `red`$\cdot$`B2`'s SoM$-$DOM contrast from $-2.96$pp [$-5.91$,
$-0.49$] to $-1.48$pp [$-3.45$, $+0.49$], i.e. across zero: this was the only cell
in which the fused mode was significantly beaten. Four of the six leaks are on the
text arm, so removing them *helps* the fused arm --- the direction that disfavours
our own caution, which is why it is reported rather than quietly adopted. The
criterion is a lower bound: an episode can reach a forum an earlier episode
subscribed to and finish without acting, and that case is confirmed to occur but
cannot be counted from the recorded state.

**Action delivery is an unreported mediator.** Actions reach the browser by three
paths with 88.9\%, 38.6\% and 16.1\% action success. `Vision` is on the coordinate
path **by construction** --- it emits no element identifiers --- so its arm
measures this harness's grounding code as much as it measures the representation.
This is not a confound to remove, since coordinate addressing is part of what
screenshot-only *is*, but every per-mode gap that runs through it is an
external-validity limit.

**Latency is mostly not the model.** The model call is 22--67\% of a measured step;
the rest is the browser and the container. Removing the environment changes which
mode is fastest in 4 of 8 cells, and the flips are concentrated where the container
is slowest (4 of 5 reddit-family cells, 0 of 3 classifieds). Any sentence naming a
fastest mode is partly a sentence about this deployment.

**Cost ordering is estimand-dependent.** Pricing the same episodes by GPU-time
rather than by token changes which mode is cheapest in 2 of 4 locally-served cells.

**Off-site steps.** Postmill is a link aggregator, so 1.05--2.13\% of reddit steps
load pages on the public internet against 0.00--0.16\% of classifieds steps. Those
steps are *faster*, not slower. Separately, reddit's container is 1.69$\times$
slower than classifieds' before any agent behaviour enters, so no between-site
latency number is quotable bare.

**Rule frequencies are symptoms, not causes.** The two largest rows in most cells
are risk markers rather than death causes, established by causal verification on 10
cases. Only rules whose docstrings record such a check are verified as causal.

**Multiplicity.** The table set is not one inferential family and is not corrected
as one. \todo{Fix the confirmatory list with the advisor --- currently only the
axis-independence set is corrected within its own family, and which further tables
join it is a framing decision.}
