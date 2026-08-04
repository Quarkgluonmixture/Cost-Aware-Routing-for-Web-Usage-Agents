**Two workloads.** VisualWebArena shopping has zero landed runs, so the
site-dependent reversal is stated between classifieds and reddit and cannot be
attributed to modality, task set, or benchmark --- two points do not identify a
moderator.

**One application on the reddit axis.** WebArena reddit is the same Postmill
deployment as VisualWebArena reddit. Agreement across the two is agreement across
task sets, not across applications.

**No cross-family control on WebArena.** Both WebArena cells are Qwen backbones, so
no statement holds cross-benchmark and cross-family simultaneously.

**Every cascade number is an offline splice.** An escalated task takes its outcome
from a standalone rich run, whereas a real cascade would start the rich episode
after the cheap one had already acted on a stateful site. That sequential outcome
is unobserved here, and the instrumentation that would bound the splice bias ---
counters for state-mutating actions --- reads zero on every episode, so the bias
cannot be bounded either.

**Replicates are thin.** Three arms on one cell carry a same-condition rerun. Every
instability figure is a lower bound from arms replicated once.

**The ex-ante predicate is ours.** It demonstrates that a zero-token ex-ante signal
exists; it is not a claim that this is the best one. It also degenerates on
WebArena, flagging 5 of 104 tasks, none solved by any mode.
