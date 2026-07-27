## 7. Related work

**Cost-aware routing for LLM serving.** A body of work routes queries among models of differing
price, learning a predictor of whether the cheap model suffices [@chen2023frugalgpt;
@ding2024hybridllm; @ong2025routellm; @gupta2024cascades; @webrouter2025], usually on an
uncertainty or calibration signal [@guo2017calibration; @kadavath2022know;
@peale2026flexibleRouting] and surveyed by [@moslem2026routingsurvey]; serving systems route by
modality and stage instead [@qiu2025modserve]. That regime differs from ours in two respects
our results suggest are decisive: the cheap model succeeds often, so supervision is plentiful,
and the task is single-turn, so the outcome is observed immediately. Multi-step agent routing
is closer [@wang2026boundaryrouter; @li2026dmr] but has not, to our knowledge, reported the
supply accounting of §4.

**Web agents and observation modes.** Benchmarks and agents treat DOM-derived text
[@zhou2024webarena; @deng2023mind2web; @agentoccam2025], screenshots [@he2024webvoyager], and
annotated screenshots [@koh2024visualwebarena; @zheng2024seeact] as design choices fixed per
system, so cross-mode comparisons are between systems rather than within a task set and the
per-task complementarity a router would exploit goes unmeasured. Work that varies observation
content while holding the system fixed [@enomoto2026observation; @schiepanski2025d2snap]
measures cost and accuracy but not per-task disagreement, which is what our §3 ceiling
measures.

**Negative results and evaluation practice.** That nesting the operating-point selection changes
conclusions [@cawley2010overfitting] and that the choice of baseline decides whether a saving
exists are instances of concerns raised repeatedly in evaluation methodology
[@lipton2018troubling], and specifically in web-agent evaluation, where reported progress has
proved sensitive to harness and judging details [@xue2025illusion;
@elhattami2025webarenaverified; @lu2025agentrewardbench; @he2025nondeterminism]. We report them
as measurements rather than recommendations.

## 8. Discussion

Within the regime we studied, a learned per-task mode router is not available, and the
obstruction is the production rate of supervision rather than the hypothesis class, the label
definition, or the estimator. The three components carry different weights: §4's supply
obstruction is arithmetic, §5's value obstruction rests on a baseline choice we argue is
correct, and §6's closure argument is that the escape routes fail for three different reasons.
We do not claim mode routing is unlearnable in general, that the ceiling is illusory, or that
our router is the best possible one: §4.1 predicts trainability as agents improve, §3 measures
a real ceiling on both axes, and §5.3 locates the failure at the policy-comparison step rather
than the prediction step.

Three implications follow. Practitioners should compare any proposed router against
always-taking-the-cheapest-mode before building it, because in none of our six cells did
anything we learned Pareto-dominate that fixed policy. Benchmark designers have a low-cost
intervention available in §6: a graded per-task score would convert every episode into a
training signal instead of only the successful ones [@lu2025agentrewardbench;
@pan2024webcanvas]. And routing supervision is a by-product of capability, so until agents
succeed often enough to produce it, effort spent on routing machinery is spent on a component
whose training signal does not yet exist.
