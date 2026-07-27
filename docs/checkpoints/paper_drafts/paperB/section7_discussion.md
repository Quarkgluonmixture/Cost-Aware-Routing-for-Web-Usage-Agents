## 7. Related work

**Cost-aware routing for LLM serving.** A body of work routes queries among models of
differing price, typically learning a predictor of whether the cheap model suffices
[@chen2023frugalgpt; @ding2024hybridllm; @ong2025routellm; @gupta2024cascades;
@webrouter2025], usually on an uncertainty or calibration signal
[@guo2017calibration; @kadavath2022know; @peale2026flexibleRouting] and surveyed by
[@moslem2026routingsurvey]; serving systems route by modality and stage rather than by query
[@qiu2025modserve]. That regime differs from ours in two respects our results suggest are
decisive: the cheap model succeeds often, so supervision is plentiful, and the task is
single-turn, so the outcome is observed immediately. Multi-step agent routing is closer to our
setting [@wang2026boundaryrouter; @li2026dmr] but has not, to our knowledge, reported the
supply accounting of §4. Our §5 adds that even where supervision is available, the baseline to
beat is the fixed cheap policy rather than the expensive one.

**Web agents and observation modes.** Benchmarks and agents treat DOM-derived text
[@zhou2024webarena; @deng2023mind2web; @agentoccam2025], screenshots [@he2024webvoyager], and
annotated screenshots [@koh2024visualwebarena; @zheng2024seeact] as design choices fixed per
system, so cross-mode comparisons are between systems rather than within a task set and the
per-task complementarity a router would exploit goes unmeasured. Work that varies observation
content while holding the system fixed [@enomoto2026observation; @schiepanski2025d2snap]
measures cost and accuracy but not per-task disagreement. Our §3 ceiling measures exactly that,
and it is why we consider the routing question worth asking despite answering it negatively.

**Negative results and evaluation practice.** That nesting the operating-point selection
changes conclusions [@cawley2010overfitting] and that the choice of baseline decides whether a
saving exists are instances of concerns raised repeatedly in evaluation methodology
[@lipton2018troubling], and specifically in web-agent evaluation, where reported progress has
proved sensitive to harness and judging details [@xue2025illusion;
@elhattami2025webarenaverified; @lu2025agentrewardbench; @he2025nondeterminism]. We report
them as measurements rather than recommendations: §5.2 quantifies the nesting effect and §5.3
shows the baseline switch removing every saving.

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

### 8.3 Implications

For practitioners, the operational recommendation is uncomfortable but concrete: at current
web-agent success rates, compare any proposed router against always-taking-the-cheapest-mode
before building it. In none of our six cells did anything we learned Pareto-dominate that
fixed policy, and in one the fixed policy dominates the learned router outright.

For benchmark designers, §6 identifies a low-cost intervention with disproportionate
value. A graded per-task score (even three levels) would convert every episode into a
training signal instead of only the successful ones, and would reopen a route that is
currently closed by a design decision rather than by anything intrinsic to the task. Work on
step-level and reward-model evaluation for web agents [@lu2025agentrewardbench;
@pan2024webcanvas] already supplies machinery a graded score could be built on.

For research on cost-aware agents, our results suggest the sequencing is backwards. Routing
supervision is a by-product of capability. Until agents succeed often enough to produce it,
effort spent on routing machinery is spent on a component whose training signal does not
yet exist.
