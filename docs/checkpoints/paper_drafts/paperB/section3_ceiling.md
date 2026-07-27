## 3. The routing ceiling is real, and it is a cost ceiling

Before asking whether a router can be learned, we measure what a perfect one would win.
The oracle sees each task's outcome under all six modes and picks the cheapest mode that
succeeded, falling back to the cheapest mode when nothing succeeded.

| cell | tasks | solvable | oracle SR | best single SR | oracle cost | best single cost | cost saved |
|---|---|---|---|---|---|---|---|
| classifieds · B0 | 224 | 43.3% | 27.23% | 27.23% | 0.06312 | 0.07236 | **−12.8%** |
| reddit · B0 | 203 | 26.1% | 14.78% | 14.78% | 0.09998 | 0.11045 | **−9.5%** |
| classifieds · B1 | 224 | 24.6% | 14.29% | 14.29% | 0.04858 | 0.06028 | **−19.4%** |
| reddit · B1 | 203 | 11.8% | 7.39% | 7.39% | 0.05554 | 0.08000 | **−30.6%** |
| classifieds · B2 | 224 | 7.1% | 2.23% | 2.23% | 0.07145 | 0.09075 | **−21.3%** |
| reddit · B2 | 203 | 7.4% | 3.94% | 3.94% | 0.06974 | 0.09479 | **−26.4%** |

*Table 2: The oracle ceiling. The oracle picks the cheapest mode that solved each task.
Success rate matches the best single mode in every cell; the entire ceiling is cost. Cost is
per-episode billed cost and is never compared across backbones.*

Two features of this table shape everything that follows.

**The ceiling is entirely in cost.** Oracle success rate equals best-single success rate
in every cell, to the task. This is not a coincidence of rounding: the oracle is
constrained to pick a mode that *succeeded*, so it can never exceed the union of
per-mode successes, and here the strongest single mode already covers that union's
success count. What the oracle adds is the ability to reach each success through a
cheaper route. A router built on this ceiling is a cost optimiser, not an accuracy
optimiser, and it should be evaluated as one.

**The ceiling is largest where the agent is weakest.** The cost saving grows as solvable
rate falls: 12.8% at 43.3% solvable, 30.6% at 11.8%. The mechanism is unflattering. When
almost every task is hopeless, almost every task can be sent to the cheapest mode without
losing anything, so the oracle's advantage over "use the best mode everywhere" is large.
The apparent opportunity is mostly an artifact of failure being cheap to reach.

This second observation is why §2.5 insists on the always-cheapest baseline. If the
oracle's saving comes largely from *not paying for hopeless tasks*, then a policy that
never pays for anything captures much of it for free. Measuring the router against
best-single credits it with a saving that a one-line policy already delivers.

We can quantify that directly. Substituting always-cheapest for the oracle in the table
above recovers a substantial share of the available saving in every cell, at the price of
accuracy. That trade is what the rest of the paper is about:

| cell | always-cheapest SR | vs best single | always-cheapest cost | vs best single |
|---|---|---|---|---|
| classifieds · B0 | 25.00% | −2.23pp | 0.06481 | −10.4% |
| reddit · B0 | 7.39% | −7.39pp | 0.09807 | −11.2% |
| classifieds · B1 | 12.50% | −1.79pp | 0.04316 | −28.4% |
| reddit · B1 | 2.46% | −4.93pp | 0.05240 | −34.5% |
| classifieds · B2 | 2.23% | ±0.00pp | 0.07065 | −22.1% |
| reddit · B2 | 1.97% | −1.97pp | 0.06833 | −27.9% |

*Table 3: The always-cheapest fixed policy against the best single mode. This policy needs
no model, no features, and no inference, and it already captures much of Table 2's cost
saving. It is the baseline §5.3 measures the learned router against.*

Note the classifieds · B2 row: always-cheapest matches the best single mode's success rate
exactly while costing 22% less. In that cell the routing problem is already solved by a
fixed policy, and any learned router can at best tie it.

The remaining cells define the actual target. A useful router must land somewhere the
oracle reaches and always-cheapest does not: recover more of the accuracy than
always-cheapest gives up, without giving back the cost saving. Sections 4 and 5 show that
neither of the two natural supervision targets gets there.
