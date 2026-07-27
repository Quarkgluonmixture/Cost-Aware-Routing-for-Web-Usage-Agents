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

**One label definition, reported against another.** The which-mode numbers in §4 and §6 use the
prior-order label. §4.3 gives the reasons and Appendix B.5 recomputes supply and trainability
under the measured-cost alternative, which strengthens the result rather than weakening it. We
have not repeated the identifiability analysis of §6 under the alternative label.
