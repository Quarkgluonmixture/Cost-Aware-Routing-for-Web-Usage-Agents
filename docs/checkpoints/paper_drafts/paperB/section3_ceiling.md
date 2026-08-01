## 3. The routing ceiling has two halves, and they need different labels

Before asking whether a router can be learned, we measure what a perfect one would win. The
oracle sees each task's outcome under all six modes and picks the cheapest mode that succeeded,
by measured per-episode cost, falling back to the cheapest when nothing succeeded. Its success
rate is therefore the union of per-mode successes by construction, which is the `solvable`
column; the two agree to the task in every cell, and that agreement is a check on the
implementation rather than a result.

| cell | n | solvable | best single | cheapest | oracle |
|---|---|---|---|---|---|
| cls · B0 | 224 | 43.3% | 27.23 / 0.07236 | 25.00 / 0.06481 | **43.30** / **0.05777** |
| red · B0 | 203 | 26.1% | 14.78 / 0.11045 | 7.39 / 0.09807 | **26.11** / **0.09534** |
| cls · B1 | 224 | 24.6% | 14.29 / 0.06028 | 12.50 / 0.04316 | **24.55** / **0.04171** |
| red · B1 | 203 | 11.8% | 7.39 / 0.08000 | 2.46 / 0.05240 | **11.82** / **0.05178** |
| cls · B2 | 224 | 7.1% | 2.23 / 0.09075 | 2.23 / 0.07065 | **7.14** / **0.06953** |
| red · B2 | 203 | 7.4% | 3.94 / 0.09479 | 1.97 / 0.06833 | **7.39** / **0.06958** |

*Table 1: The routing ceiling, over classifieds (cls) and reddit (red). The last three columns
give success rate (%) and mean per-episode billed cost (USD). The oracle picks the cheapest
mode that solved each task, so its success rate equals the solvable column by construction;
"cheapest" is the lowest-mean-cost mode applied everywhere. Cost is never compared across
backbones.*

The opportunity is large on both axes. Against the best single mode the oracle gains **3.45 to
16.07 percentage points** of success rate and spends **13.7% to 35.3% less**. A router built on
this ceiling would be neither purely a cost optimiser nor purely an accuracy optimiser.

One property of this oracle matters later. It breaks ties by *measured* episode cost, which is
realised after the fact and dominated by step count, so the cost column is an upper bound in a
stronger sense than the success column: unreachable not only because the outcome is unknown
before acting but because the cost is. §4.3 returns to this.

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

*Table 2: The ceiling decomposed. Both columns are deltas against the best single mode. The
triage half is accuracy-neutral by construction and carries 9.5–30.6% of cost saving; the
route half carries 3.45–16.07pp of success rate and much less cost. The two halves require
different labels, and the rest of the paper fails on one obstruction per half.*

The split maps directly onto the two failures we report. The route half needs the which-mode
label, which §4 shows is produced only at the success rate and is unavailable in the quantity
a six-class model needs. The triage half needs the binary label, which §5 shows is plentiful
and predictable and still buys nothing a fixed policy does not already deliver. Neither half
of the ceiling survives, and each fails for its own reason.

### 3.2 Why the fixed baseline is the one that matters

The cost half is largest where the agent is weakest: within classifieds the triage saving grows
from 12.8% at 43.3% solvable to 21.3% at 7.1%. The mechanism is unflattering. When almost every
task is hopeless, almost every task can be sent to the cheapest mode without losing anything,
so a policy that never pays for anything captures much of that saving for free. Table 1 shows
always-cheapest already spending 10.4% to 34.5% less than the best single mode while giving up
0.00 to 7.39 points of success rate, and on classifieds · B2 giving up nothing at all.

That is why §2.5 measures against it. A useful router must land somewhere the oracle reaches
and always-cheapest does not: recover accuracy that always-cheapest gives up, or cost that it
does not, without giving back the other. Sections 4 and 5 show that neither supervision target
gets there.

### 3.3 How much of the accuracy half is reproducible by rerunning?

The route half is a union over arms, so any added arm inflates it, including one that adds no
capability such as a rerun of a mode already in the menu. For runs $A, B$ of one (model, site,
mode) the *rerun drop* $|A \setminus B| / n$ is that same functional, and it is the reference
the accuracy half has to clear. We measure it on the three same-condition replicate pairs we
hold, two on classifieds under B0 and one under B1 on the reddit split of WebArena
(Appendix A.6).

The comparison must be made at equal arm count: the route half adds five arms, a rerun adds one.
At one arm the two are not separable. On classifieds · B0 the best *different* mode added to SoM
buys **7.14pp**, inside that cell's own **4.91–7.59** rerun band; on WebArena · B1 the best
different mode added to DOM buys **4.81pp** against **2.00–4.00**, clearing it by 0.81pp on a
floor estimated from 50 paired tasks. The two cells that carry a floor differ in backbone
family, in serving path (a commercial API against a local GPU) and in benchmark, and they agree:
one arm of representation is worth about one arm of repetition.

Union-over-reruns is measurable on the pairs we hold, and it yields the most directly usable
result in this section. On classifieds · B0, **running the cheapest mode twice reaches 31.70%,
against the best single mode's 27.23%**. Repetition alone closes 27.8% of the 16.07pp routing
gap, with no labels, no router and no second representation, and a practitioner can act on that
today. It is mode-dependent: two runs of DOM reach only 22.32%, and Vision is the arm with the
highest run-to-run discordance, so the recipe is to repeat the mode that is least stable rather
than the one that is best. The obvious next question is whether the ceiling is then just
$\text{pass}@k$.

The cost axis is what separates them, and it separates them decisively. $\text{pass}@k$ runs
every task $k$ times; the oracle runs each task once. Two runs of Vision cost \$0.12962 against
the oracle's \$0.05777, so the oracle **strictly dominates** it, taking 11.60 points more success at
45% of the price. The ceiling is not a resampling artefact, because it is a ceiling *under a
single-episode budget*. What repetition does undercut is the reading of the accuracy half as
complementarity, which is the claim we withdraw.

The claim we make is narrow and exact: **at the one-arm margin a router actually operates on,
the premium for representational diversity is not resolvable against repetition.** It does
**not** follow that the +16.07pp is noise, and we do not argue that: we hold one rerun arm
rather than five, reruns saturate, and any $k>1$ policy pays $k\times$. The cost half is untouched, being accuracy-neutral and not a union. It also sharpens
§§4–6: a router failing to recover the accuracy half is failing to recover a partly artefactual
quantity.

The same reference retires a finer claim we had pre-registered. Decomposing the three
image-free arms along their two knobs, text payload and prompt family, gives two set
differences against the compound arm that pool at **1.35pp** and **2.09pp**, both with
bootstrap intervals clear of zero. They are the same functional as the rerun drop, so they are
directly comparable to it, and axis-1 falls below even the lowest floor we measured while
axis-2 sits inside its lower half. Clearing a "different from zero" gate is weak evidence when
two runs of one fixed configuration also clear it, so we report the decomposition as measured
and withdraw the structural reading it was meant to license. What survives is a bound rather
than a null: the sign is positive in five of six cells on both axes, and the magnitude of each
knob is at most the run-to-run floor, which for a practitioner choosing among the image-free
modes means the choice is worth less than one repetition.

**What the rerun drop is and is not.** Replicate runs are separated by days, and one of the
three pairs sits on a site whose per-task reset is a no-op, so accumulated site state
contributes alongside decoding non-determinism. The quantity is therefore run-to-run
variation *including environment drift*, not decoding stochasticity, and we name it that
way. For the comparison above that is the right quantity: the conditions in Table 1 were also
collected over days, on the same infrastructure, and carry the same drift. It also means the
locally served row is not evidence that a deterministic decoder yields deterministic episodes.
Greedy decoding is bit-reproducible at the step level in our own checks; the episode is not,
because the episode is not only the decoder.
