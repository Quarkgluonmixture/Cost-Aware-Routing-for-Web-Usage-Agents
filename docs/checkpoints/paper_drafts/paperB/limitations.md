**Two sites, one benchmark.** Site-level effects and benchmark-level evaluation design
(notably the binary score of §6) are not separable at this scale. A benchmark with graded
outcomes could reopen that route without changing §4.

**Offline replay, no live routing.** We never served a router. All results exclude router
inference cost and latency, which flatters the router and therefore strengthens a negative
result, but it means we have not measured a deployment. A mode chosen at step 0 is used for
the whole episode in our replay, so the interaction between routing decisions and multi-step
trajectory dynamics, including adaptive mid-episode switching, is unexamined.

**Cost is not pooled across backbones.** API pricing and electricity-derived cost differ by
roughly three orders of magnitude per token, so we compare ratios within a backbone only.
Cross-backbone cost statements would be unit collisions.

**One label definition, measured against another.** The which-mode label our pipeline produces
walks a fixed mode list and takes the first success, and §4.3 measures how often that disagrees
with the measured cost it is meant to encode. We report the disagreement rather than
regenerating the label, so the supply and identifiability numbers in §4 and §6 are computed on
the list-order label. The direction of the resulting bias is not obvious a priori; a
recomputation under measured cost is the first thing we would run next.
