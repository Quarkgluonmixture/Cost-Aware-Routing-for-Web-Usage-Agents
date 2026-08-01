## 7. Related work

**Cost-aware routing for LLM serving.** A body of work routes queries among models of differing
price, learning a predictor of whether the cheap model suffices [@chen2023frugalgpt;
@ding2024hybridllm; @ong2025routellm; @gupta2024cascades; @webrouter2025], usually on an
uncertainty or calibration signal [@guo2017calibration; @kadavath2022know;
@peale2026flexibleRouting] and surveyed by [@moslem2026routingsurvey]; serving systems route by
modality and stage instead [@qiu2025modserve]. That regime differs from ours in two decisive
respects: the cheap model succeeds often, so supervision is plentiful, and the task is
single-turn, so the outcome is observed immediately. Multi-step agent routing is closer
[@wang2026boundaryrouter; @li2026dmr] but has not, to our knowledge, reported §4's supply
accounting.

**Web agents and observation modes.** Benchmarks and agents treat DOM-derived text
[@zhou2024webarena; @deng2023mind2web; @agentoccam2025], screenshots [@he2024webvoyager], and
annotated screenshots [@koh2024visualwebarena; @zheng2024seeact] as design choices fixed per
system, so cross-mode comparisons are between systems rather than within a task set and the
per-task complementarity a router would exploit goes unmeasured. Work that varies observation
content while holding the system fixed [@enomoto2026observation; @schiepanski2025d2snap]
measures cost and accuracy but not per-task disagreement, which is what our §3 ceiling
measures. Two results bracket our §4. @enomoto2026readmore show the best representation is
conditional rather than fixed, though on two text arms with model capability as the
conditioning variable; and @gupta2026molmoweb train an 8B screenshot-only policy that surpasses
set-of-mark agents built on much larger closed models, which is evidence against the fused
representation's necessity from the modelling side where ours is from the accounting side.

**Negative results and evaluation practice.** That nesting the operating-point selection changes
conclusions [@cawley2010overfitting] and that the choice of baseline decides whether a saving
exists are instances of concerns raised repeatedly in evaluation methodology
[@lipton2018troubling], and specifically in web-agent evaluation, where reported progress has
proved sensitive to harness and judging details [@xue2025illusion;
@elhattami2025webarenaverified; @lu2025agentrewardbench; @he2025nondeterminism], and where 23
repeats of a single tool-calling setup span 18.9 points [@bhat2026benchmarkingbenchmarks].
Closest to our §3 is @hajimiri2026budgetmatched, who net three online augmentation methods
against a token-matched baseline that simply buys more actor steps and find the gains largely
disappear. We apply that accounting to the representation axis, where the matched alternative is
not more steps but the same configuration run again. We report these as measurements rather than
recommendations.

## 8. Discussion

Within the regime we studied, no learned per-task mode router we built is available. The
obstructions differ in kind, which is what makes the result closed rather than provisional:
§4's is arithmetic, §5.3's rests on a baseline choice we argue is correct, §5.5's survives letting the router see the cheap run
before it decides, and §6's is that three escape routes fail for
three separate reasons while two stay open. Only one of them does not depend on our sample
size. Where we hold replicates, half the tasks on which the channels disagree are tasks whose
outcome changes when the same configuration is rerun, so the target itself is partly
irreproducible and no quantity of data repairs that. We do not claim mode routing is unlearnable in general, that the ceiling is
illusory, that our router is the best possible one, or that no target admits enough supervision.
§6's per-mode success target plainly does, and it is the one we could not close: it wins at a
post-hoc threshold and fails on transfer, which is a different verdict from the one §4 delivers.
§4.1 predicts trainability as agents improve, §3 measures a real ceiling, and §5.3 puts the
triage failure at the policy comparison, not the prediction.

Three implications follow. Practitioners should compare any proposed router against the fixed
policy it would replace before building it, in both directions: nothing we learned
Pareto-dominated always-taking-the-cheapest-mode in any of the six cells, and nothing we
escalated Pareto-dominated always-taking-the-dearest either. They should also net any
representation gain against simply running the representation they have twice, which on
classifieds · B0 outperforms the best single mode outright (§3.3). Benchmark designers have a low-cost
intervention in §6: a graded per-task score would make every episode a training signal
[@lu2025agentrewardbench; @pan2024webcanvas]. And routing supervision
is a by-product of capability, so until agents succeed more often there is nothing to fit.
