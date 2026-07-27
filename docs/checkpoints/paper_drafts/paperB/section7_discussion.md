## 7. Related work

**Cost-aware routing for LLM serving.** A body of work routes queries among models of
differing price, typically learning a predictor of whether the cheap model suffices
[@chen2023frugalgpt; @ding2024hybridllm; @ong2025routellm; @gupta2024cascades;
@webrouter2025], with uncertainty and calibration as the usual routing signal
[@guo2017calibration; @kadavath2022know; @peale2026flexibleRouting], and recent surveys
covering the design space [@moslem2026routingsurvey]. Serving-system work routes by modality
and stage rather than by query [@qiu2025modserve]. These systems operate in a regime unlike
ours in two respects that our results suggest are decisive: the cheap model succeeds often, so
supervision is plentiful, and the task is single-turn, so the outcome is observed immediately
and cheaply. Multi-step agent routing is more recent and closer to our setting
[@wang2026boundaryrouter; @li2026dmr]; it has not, to our knowledge, reported the supply
accounting of §4. Our §4 argument is that the first property is what makes the serving
literature's supervision available, and our §5 argument is that even when it is available,
the baseline the router must beat is the fixed cheap policy rather than the expensive one.

**Web agents and observation modes.** Benchmarks and agents in this area treat DOM-derived
text [@zhou2024webarena; @deng2023mind2web; @agentoccam2025], screenshots
[@he2024webvoyager], and annotated screenshots [@koh2024visualwebarena; @zheng2024seeact] as
design choices fixed per system. Where multiple modes are compared, the comparison is usually
between systems rather than within a task set, so the per-task complementarity that a router
would exploit is not measured; the exception is work that varies observation content while
holding the system fixed [@enomoto2026observation; @schiepanski2025d2snap], which measures
cost and accuracy but not per-task disagreement. Our §3 ceiling is a measurement of that
complementarity, and it is the reason we consider the routing question worth asking despite
answering it negatively.

**Negative results and evaluation practice.** Our methodological findings, that nesting
the operating-point selection changes conclusions [@cawley2010overfitting], and that the
choice of baseline decides whether a saving exists, are instances of concerns raised
repeatedly in evaluation methodology work [@lipton2018troubling], and specifically in web-agent
evaluation, where reported progress has proved sensitive to harness and judging details
[@xue2025illusion; @elhattami2025webarenaverified; @lu2025agentrewardbench;
@he2025nondeterminism]. We report them as concrete measurements rather than as
recommendations: §5.2 quantifies the nesting effect at −0.99 to +1.34 percentage points, and
§5.3 shows the baseline switch removing every saving.

## 8. Discussion

### 8.1 What we claim

Our claim is scoped to the regime we studied, and within it a learned per-task mode router is
not available, with the obstruction lying in the production rate of supervision rather than in
the hypothesis class, the label definition, or the estimator. The three components carry
different weights. The supply obstruction (§4) is arithmetic: labels equal successes, and four
of six cells fall below any usable threshold. The value obstruction (§5) is empirical and
rests on a baseline choice we argue is the correct one. The closure argument (§6) is that the
escape routes fail for three different reasons, which is what distinguishes a closed negative
result from an unfinished search.

### 8.2 What we do not claim

We do not claim that mode routing is unlearnable in general. §4.1's mechanism predicts the
opposite as agents improve: at 60% success a cell would yield several hundred labels and
become trainable. The prediction is falsifiable and we would welcome its falsification.

We do not claim the ceiling is illusory. §3 measures a real one on both axes. Our result is
about which parts of it supervision can reach: the accuracy-neutral part is largely also
reachable by a fixed policy, so the *learned* component adds little there, and the
accuracy-bearing part is not reachable at all because its label is not produced.

We do not claim our router is the best possible one. The triage model is a logistic model over
20 numeric and binary features with no text features at all, and a stronger model might extract
more signal. But §5.1 shows the signal is already present and §5.3 shows that converting it
into value fails at the policy-comparison step, not at the prediction step. A better predictor
does not address that.

### 8.3 Limitations

**Two sites, one benchmark.** Site-level effects and benchmark-level evaluation design
(notably the binary score of §6.1) are not separable at this scale. A benchmark with graded
outcomes could reopen §6.1 without changing §4.

**Offline replay, no live routing.** We never served a router. All results exclude router
inference cost and latency, which flatters the router and therefore strengthens a negative
result, but it means we have not measured a deployment. A mode chosen at step 0 is used for
the whole episode in our replay, so the interaction between routing decisions and multi-step
trajectory dynamics, including adaptive mid-episode switching, is unexamined.

**Cost is not pooled across backbones.** API pricing and electricity-derived cost differ by
roughly three orders of magnitude per token, so we compare ratios within a backbone only.
Cross-backbone cost statements would be unit collisions.

### 8.4 Implications

For practitioners, the operational recommendation is uncomfortable but concrete: at current
web-agent success rates, compare any proposed router against always-taking-the-cheapest-mode
before building it. In none of our six cells did anything we learned Pareto-dominate that
fixed policy, and in one the fixed policy dominates the learned router outright.

For benchmark designers, §6.1 identifies a low-cost intervention with disproportionate
value. A graded per-task score (even three levels) would convert every episode into a
training signal instead of only the successful ones, and would reopen a route that is
currently closed by a design decision rather than by anything intrinsic to the task. Work on
step-level and reward-model evaluation for web agents [@lu2025agentrewardbench;
@pan2024webcanvas] already supplies machinery a graded score could be built on.

For research on cost-aware agents, our results suggest the sequencing is backwards. Routing
supervision is a by-product of capability. Until agents succeed often enough to produce it,
effort spent on routing machinery is spent on a component whose training signal does not
yet exist.
