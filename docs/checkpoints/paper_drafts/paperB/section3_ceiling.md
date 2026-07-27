## 3. The routing ceiling has two halves, and they need different labels

Before asking whether a router can be learned, we measure what a perfect one would win.
The oracle sees each task's outcome under all six modes and picks the cheapest mode that
succeeded, by measured per-episode cost, falling back to the cheapest mode when nothing
succeeded. Its success rate is therefore the union of per-mode successes by construction,
which is the `solvable` column: the two agree to the task in every cell, and that agreement
is the check that the oracle is implemented as defined rather than a self-standing result.

| cell | tasks | solvable | best single SR / cost | always-cheapest SR / cost | oracle SR / cost |
|---|---|---|---|---|---|
| classifieds · B0 | 224 | 43.3% | 27.23% / 0.07236 | 25.00% / 0.06481 | **43.30%** / **0.05777** |
| reddit · B0 | 203 | 26.1% | 14.78% / 0.11045 | 7.39% / 0.09807 | **26.11%** / **0.09534** |
| classifieds · B1 | 224 | 24.6% | 14.29% / 0.06028 | 12.50% / 0.04316 | **24.55%** / **0.04171** |
| reddit · B1 | 203 | 11.8% | 7.39% / 0.08000 | 2.46% / 0.05240 | **11.82%** / **0.05178** |
| classifieds · B2 | 224 | 7.1% | 2.23% / 0.09075 | 2.23% / 0.07065 | **7.14%** / **0.06953** |
| reddit · B2 | 203 | 7.4% | 3.94% / 0.09479 | 1.97% / 0.06833 | **7.39%** / **0.06958** |

*Table 2: The routing ceiling. The oracle picks the cheapest mode that solved each task, so
its success rate equals the solvable column by construction. Always-cheapest is the
lowest-mean-cost mode applied everywhere. Cost is per-episode billed cost and is never
compared across backbones.*

The opportunity is large on both axes. Against the best single mode the oracle gains **3.45 to
16.07 percentage points** of success rate and spends **13.7% to 35.3% less**. A router built on
this ceiling would be neither purely a cost optimiser nor purely an accuracy optimiser.

### 3.1 Splitting the ceiling by the label each half requires

The two axes are not reachable by the same supervision, and separating them decides what the
rest of the paper has to test. We therefore evaluate each half of the oracle on its own.

**The triage half** keeps the best-success mode on every task the oracle would have solved and
sends the rest to the cheapest mode. It never changes which mode solves anything, so its
success rate is pinned to the best single mode. Its label is binary: is this task solvable by
anything. That label is defined for every task in the universe.

**The route half** chooses among the modes that solved a task and uses the best-success mode
everywhere else. It captures the accuracy gain. Its label is the identity of a mode, and that
identity exists only for tasks something solved.

| cell | triage half: ΔSR / Δcost | route half: ΔSR / Δcost |
|---|---|---|
| classifieds · B0 | ±0.00pp / **−12.8%** | **+16.07pp** / −7.4% |
| reddit · B0 | ±0.00pp / **−9.5%** | **+11.33pp** / −4.2% |
| classifieds · B1 | ±0.00pp / **−19.4%** | **+10.27pp** / −11.4% |
| reddit · B1 | ±0.00pp / **−30.6%** | **+4.43pp** / −4.7% |
| classifieds · B2 | ±0.00pp / **−21.3%** | **+4.91pp** / −2.1% |
| reddit · B2 | ±0.00pp / **−26.4%** | **+3.45pp** / −0.2% |

*Table 3: The ceiling decomposed. Both columns are deltas against the best single mode. The
triage half is accuracy-neutral by construction and carries 9.5–30.6% of cost saving; the
route half carries 3.45–16.07pp of success rate and much less cost. The two halves require
different labels, and the rest of the paper fails on one obstruction per half.*

The split maps directly onto the two failures we report. The route half needs the which-mode
label, which §4 shows is produced only at the success rate and is unavailable in the quantity
a six-class model needs. The triage half needs the binary label, which §5 shows is plentiful
and predictable and still buys nothing a fixed policy does not already deliver. Neither half
of the ceiling survives, and each fails for its own reason.

### 3.2 The cost half is largest where the agent is weakest

Within classifieds the triage saving grows monotonically as the solvable rate falls: 12.8% at
43.3% solvable, 19.4% at 24.6%, 21.3% at 7.1%. Reddit shows the same direction without being
monotone. The mechanism is unflattering. When almost every task is hopeless, almost every task
can be sent to the cheapest mode without losing anything, so the advantage over "use the best
mode everywhere" is large precisely because failure is cheap to reach.

This is why §2.5 insists on the always-cheapest baseline. If much of the cost saving comes from
not paying for hopeless tasks, then a policy that never pays for anything captures much of it
for free. Table 2 shows always-cheapest already spending 10.4% to 34.5% less than the best
single mode, at a success-rate cost of 0.00 to 7.39 percentage points. Measuring a router
against best-single credits it with a saving a one-line policy already delivers.

Note the classifieds · B2 row: always-cheapest matches the best single mode's success rate
exactly while costing 22.1% less. In that cell the cost half of the routing problem is already
solved by a fixed policy, and any learned router can at best tie it.

The remaining cells define the actual target. A useful router must land somewhere the oracle
reaches and always-cheapest does not: recover accuracy that always-cheapest gives up, or cost
that it does not, without giving back the other. Sections 4 and 5 show that neither of the two
natural supervision targets gets there.
