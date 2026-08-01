**The rerun reference of §3.3 is thin, and is an upper bound.** Three replicate pairs, two of
them on one (backbone, site) and the third pooled over 50 paired tasks on a second benchmark.
None of the six cells in Table 1 has a replicate of its own, so the rerun band is transferred
across cells rather than measured within them, and cells served locally may sit lower than the
API-served pairs. The band is therefore an upper bound on instability, which is the
conservative direction for the *cost* half and the anti-conservative direction for our reading
of the *accuracy* half — we state the comparison as "not separable at the margin" rather than
as a subtraction, and we perform no arithmetic between the two quantities. A within-cell
replicate on each of the six cells would settle it; that is the experiment we did not run.

**We report no five-rerun ceiling.** §3.3 compares one added arm against one added arm. Whether
five reruns would reach the five-mode union is unmeasured, and nothing in this paper should be
read as claiming that they would.

**Two sites on the main benchmark, one split of a second.** The six cells of Table 1 are two
VisualWebArena sites; WebArena contributes one reddit split under one backbone, used for the
rerun floor of §3.3 and the corroboration of §4.4 and never pooled with the six. Site-level
effects and benchmark-level evaluation design (notably the binary score of §6) are not
separable at this scale, and two workloads are enough to show that the best channel changes
with the workload but not to characterise what it changes with. A benchmark with graded
outcomes could reopen that route without changing §4.

**Offline replay, no live routing.** We never served a router. All results exclude router
inference cost and latency, which flatters the router and therefore strengthens a negative
result, but it means we have not measured a deployment. A mode chosen at step 0 is used for
the whole episode in our replay, so the interaction between routing decisions and multi-step
trajectory dynamics, including adaptive mid-episode switching, is unexamined.

**The cascade of §5.5 is a two-tier simulation, not a served system.** It replays completed
episodes, so an escalated task is charged the full cost of both runs and never benefits from
stopping the cheap run early, which understates a real cascade's cost advantage. It also uses
one fixed cheap tier and one fixed rich tier rather than choosing them per cell.

**Cost is not pooled across backbones.** API pricing and electricity-derived cost differ by
roughly three orders of magnitude per token, so we compare ratios within a backbone only.
Cross-backbone cost statements would be unit collisions.

**One label definition, reported against another.** The which-mode numbers in §4 and §6 use the
prior-order label. §4.3 gives the reasons and Appendix B.5 recomputes supply and trainability
under the measured-cost alternative, which strengthens the result rather than weakening it. We
have not repeated the identifiability analysis of §6 under the alternative label.
