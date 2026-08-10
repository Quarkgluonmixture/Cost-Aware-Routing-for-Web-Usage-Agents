# CHAPTER_CHAIN.md — Thesis Argument Chain

**Project:** Cost-Aware Routing for Web Usage Agents  
**Status:** argument architecture, not final table of contents  
**Last updated:** 2026-08-09

> **Chapter-chain rule:** every chapter answers one question. Its final paragraph states what has been established, what remains unresolved, and why the next chapter is the necessary next move.

This file controls the **logic** of the dissertation. Page allocation, figure count, appendix placement, and chapter compression can change after the 2025/26 COMP0191 handbook is obtained without changing the argument chain.

---

## 0. Thesis spine in one paragraph

Web agents can receive the same page through cheap structured text, expensive visual input, or combinations of the two. A fixed policy pays for one representation everywhere even though the value of that information can vary across websites, tasks, and interaction states. The thesis therefore first establishes a controlled representation space and measures how its modes differ in task success and cost. It then asks whether those differences are complementary enough to create value for selection, whether the need for a richer representation can be predicted from signals available cheaply at serving time, and finally whether a deployable router can realise that value after its own overhead is counted. The scientific object is therefore **not merely a router implementation**: it is the empirical structure that makes selective multimodal inference useful, predictable, or impossible.

---

## 1. Working research questions

These are **working thesis RQs**, not yet frozen wording. They should be reconciled with the final results tables before the Introduction is locked.

### RQ1 — Representation value

**How do structured text, prompt framing, and image presence affect the success and efficiency of web agents, and how does their value vary across tasks and sites?**

Purpose:
- establish the empirical reason routing might be worthwhile;
- separate representation effects rather than comparing only two monolithic systems;
- expose complementarity and failure structure.

### RQ2 — Predictability

**Can the step- or state-dependent value of richer page representations be predicted from signals available without first paying their full inference cost?**

Purpose:
- turn representation heterogeneity into a learnable decision problem;
- test whether the relevant information exists in serving-time features;
- distinguish oracle opportunity from deployable predictability.

### RQ3 — Realised routing value

**Does cost-aware routing improve the success–efficiency trade-off over fixed observation policies once router overhead is included, and how robust is the result across sites and an external validation setting?**

Purpose:
- evaluate the actual system rather than proxy classifier metrics;
- account for routing cost;
- identify where routing helps, where it does not, and why.

### Optional framing question for Discussion

**What do these results imply about when multimodal perception should be invoked in interactive agents?**

This is an implication question, not another primary experiment unless the evidence directly supports it.

---

# 2. Full chapter chain

## Chapter 1 — Introduction

### The one question

**Why is selective page representation a research problem worth studying in web agents?**

### What the reader must know by the end

1. Web agents repeatedly pay for perception and inference across multi-step tasks.
2. Richer visual context can supply useful information but is not free.
3. Existing systems often make representation choices globally or architecturally, while the value of a representation can vary by interaction context.
4. Routing is only useful if three conditions hold:
   - different modes provide genuinely different value;
   - that value is predictable before paying for the expensive mode;
   - realised savings exceed router overhead.
5. The thesis tests those conditions in a controlled web-agent setting.

### Recommended internal sequence

**1.1 Problem**  
Web agents need information about a page to act, but different ways of representing that page carry different computational costs and information.

**1.2 Stakes**  
Multi-step agents repeatedly invoke perception and generation; unnecessary rich inputs increase latency/resource use and make efficient deployment harder.

**1.3 Gap / pivot**  
The central missing question is not simply whether text or vision is “better”, but **when each is worth its cost and whether that need can be predicted**.

**1.4 Approach**  
Controlled observation modes → representation characterisation → routing-signal study → deployable router → overhead-aware evaluation → external validation.

**1.5 Contributions**  
Write contributions as findings/methodological assets, not implementation chores. Example categories:
- controlled factorisation of page-input configurations;
- empirical characterisation of complementary representation value;
- leakage-controlled routing formulation/signals;
- overhead-aware routed-system evaluation;
- external validation / scope analysis.

**1.6 Thesis map**  
One sentence per chapter describing the question it answers, not merely its topic.

### Evidence allowed here

Headline results only, once frozen. No dense table dump.

### Do not overclaim

- Do not say all web agents waste vision cost.
- Do not say routing is inherently sustainable.
- Do not introduce Phantom-SoM as a grand theory before the empirical problem is clear.
- Do not make the thesis sound like “we built a router and benchmarked it”.

### Opening logic

> Web agents must repeatedly decide what to do from a representation of the current page. Richer representations can reveal information unavailable in text alone, but they also increase the cost of each interaction. This creates a selection problem: the agent should pay for richer perception only when the additional information is likely to change the outcome.

### Chapter-ending handoff

> This chapter motivates selective representation as an efficiency problem, but that motivation alone does not show that routing is either necessary or feasible. We first need to understand what prior work establishes about web-agent representations, routing, and efficiency accounting, and where the evidence remains insufficient for that decision. Chapter 2 therefore situates the problem in the existing literature.

---

## Chapter 2 — Background and Related Work

### The one question

**What is already known about web-agent representations and selective computation, and what remains unresolved enough to justify this study?**

### Required jobs

1. Give only the technical background needed to read the rest of the thesis.
2. Synthesize prior work by **decision problem**, not paper-by-paper chronology.
3. Reuse the structured-literature-review evidence where still valid.
4. End in a precise research gap that the experimental design can actually answer.

### Recommended thematic structure

**2.1 How a web agent acts**  
Observation → model inference/planning → grounding/action → browser transition → next observation.

**2.2 Ways to represent a web page**  
Structured text / accessibility information; screenshots; visual grounding; Set-of-Marks; hybrid representations.

**2.3 Why representation choice changes both capability and cost**  
Information availability, grounding, token/image processing, latency, context size.

**2.4 Routing and selective computation**  
Fallbacks, cascades, modular control, learned routing. Explain the mechanism first, taxonomy second.

**2.5 Efficiency and Green-AI-style accounting**  
Latency, calls, tokens, money, energy/CO₂e where defensible; overhead is part of system cost.

**2.6 Evidence gap**  
Controlled cross-representation evidence and complete overhead accounting are limited; the literature motivates a controlled web setting but does not tell us whether a step-level router can realise net benefit.

### Research-process material worth retaining

The broad literature review originally started from efficiency-aware agentic AI and narrowed toward web agents because comparable evidence was concentrated there. This is useful **once**, as evidence-led scope justification. Do not reproduce the full coursework protocol in the main thesis unless required.

### Evidence contract

By the end of Chapter 2, the reader should accept:

- web agents are a defensible controlled setting for the question;
- representation is a first-order system design choice;
- routing/selective computation exists as a design pattern;
- prior work does not settle the thesis's conditional, overhead-aware question.

### Do not overclaim

- “No prior work has routed modalities” is almost certainly too strong.
- Industry deployment of a representation is not equivalent to peer-reviewed controlled characterisation.
- Sparse overhead reporting is a gap in evidence, not proof that existing methods are inefficient.

### Chapter-ending handoff

> Prior work establishes that page representation can alter both agent capability and computational burden, while routing is a plausible way to manage that trade-off. It does not, however, provide a clean basis for comparing the relevant choices under one agent, benchmark, action space, and accounting boundary. Chapter 3 therefore defines a controlled experimental system in which representation and routing can be varied without changing the rest of the agent stack.

---

## Chapter 3 — Problem Formulation and Experimental System

### The one question

**How can representation choice and routing be compared fairly enough that later differences are interpretable?**

### This chapter is the experimental contract

It should make a hostile examiner able to reconstruct exactly what changes across conditions and what does not.

### Recommended structure

**3.1 Task and interaction loop**  
Define task, episode, step, action space, termination, benchmark evaluator.

**3.2 Controlled observation space**  
Define AXTree, `[SOM_MARKS]`, prompt framing, annotated screenshot, raw screenshot.

**3.3 Observation modes**  
Give the mode table: DOM / P-prompt / P-text / P-SoM / SoM / Vision. If the final thesis drops any mode from primary analysis, still explain the factorisation needed to understand the selected conditions.

**3.4 Router formulation**  
At step \(t\), router observes serving-time state/features and selects an observation mode; the agent then selects the browser action.

A generic formulation is enough:

\[
m_t = \pi_r(z_t), \qquad a_t = \pi_a(o_t^{(m_t)}, h_t)
\]

where:
- \(z_t\): router-available cheap signals;
- \(m_t\): selected observation mode;
- \(o_t^{(m_t)}\): observation generated under that mode;
- \(h_t\): interaction history;
- \(a_t\): browser action.

**3.5 Benchmarks and scope**  
VisualWebArena main study; sites/subsets; environment validity rules; external validation reserved for later chapter.

**3.6 Common model and inference configuration**  
Same model, system prompt policy where appropriate, decoding, max steps, action space, seeds; explain any unavoidable mode-specific prompt differences.

**3.7 Metrics and accounting boundary**  
Task success, steps, no-progress diagnostics, latency, tokens/cost, energy/CO₂e only if final measurement is defensible.

**3.8 Statistical design**  
Paired tasks, CIs/tests, grouping/site structure, seed policy, missing/invalid task adjudication.

**3.9 Reproducibility and logging**  
Step-level JSONL/artifacts; runner separate from visualisation; versioning and configuration traceability.

### The key figure this chapter eventually needs

Not a decorative architecture diagram. It should show the **causal comparison boundary**:

`same task + same agent` → router/selected mode → mode-specific observation construction → same action loop → benchmark outcome + measured cost.

FIGURE_PLAN is deliberately deferred until handbook limits are known.

### Research-process material worth retaining

The move from the original three monolithic modes (DOM / SoM / Vision) to a factorised set of text/prompt/image configurations is worth explaining **if** it was motivated by an observed ambiguity/confound and supports the final analysis. Phrase it as experimental refinement, not chronology.

### Do not overclaim

- The factorisation identifies controlled configuration differences; it does not automatically identify causal internal VLM mechanisms.
- `dom` is an implementation label for an AXTree-based condition; define it precisely.

### Chapter-ending handoff

> Chapter 3 fixes the comparison boundary: the task, base agent and browser loop remain controlled while the information supplied to the agent is varied explicitly. Routing would still be unnecessary if one mode dominated across tasks and sites. Chapter 4 therefore asks the prerequisite empirical question: do the candidate representations exhibit meaningful performance–cost differences and complementary successes?

---

## Chapter 4 — Representation Characterisation: When Does Each Input Help?

### The one question

**Is there enough heterogeneous and complementary value across observation modes to make routing scientifically worthwhile?**

This is the **empirical motivation chapter**. It should stand on its own even if the learned router later disappoints.

### Recommended analysis ladder

**4.1 Global success and efficiency**  
Where does each fixed mode sit on success versus measured cost/latency?

**4.2 Site-specific behaviour**  
Does the ordering change across Classifieds / Reddit / Shopping or the final analysed sites?

**4.3 Pairwise disagreement / overlap**  
Which tasks are solved by multiple modes, and which are mode-specific?

**4.4 Oracle and drop-one analysis**  
How much selection opportunity exists if an oracle could choose retrospectively? Which modes contribute unique successes?

**4.5 Factorised phantom-space analysis**  
If retained after final verification: separate effects associated with text payload, prompt framing, and image presence. P-text / P-prompt are structural ablations; P-SoM may be an operationally interesting compound configuration.

**4.6 Behavioural/failure examples**  
Use a few concrete task trajectories to show what aggregate numbers mean: visual attribute dependence, structured grounding, page-state issues, etc.

**4.7 What this does *not* establish**  
Complementarity is necessary for routing, but an oracle uses forbidden hindsight. The next question is predictability.

### Core evidence types

- paired per-task success matrix;
- TSR with uncertainty;
- latency/token/cost distributions;
- overlap matrix or UpSet-style summary;
- oracle and drop-one estimates;
- site stratification;
- carefully chosen trajectory examples.

### Claim calibration

Strong enough:

> The fixed-mode results show that no single observation mode captures all successful tasks under the tested setup, creating measurable headroom for selective routing.

Too strong unless directly supported:

> Different modes use fundamentally different reasoning mechanisms.

A mechanism hypothesis belongs in Discussion unless separately tested.

### Paper-vs-thesis guardrail

The Phantom-SoM finding can be an important result here, but the thesis should not collapse into “a Phantom-SoM paper plus extra chapters”. The broader thesis question is **conditional representation selection**. P-SoM is evidence about the representation space and potentially one valuable routing arm.

### Chapter-ending handoff

> The fixed-policy study establishes that representation value is heterogeneous and that retrospective selection has non-trivial headroom. An oracle, however, chooses with knowledge of outcomes that a deployed system cannot observe. The practical question is therefore whether the need for a different representation leaves a detectable signature *before* the expensive choice is made. Chapter 5 formulates and evaluates that prediction problem.

---

## Chapter 5 — Predicting Representation Need

### The one question

**Can a router infer when another representation is worth using from information available at serving time?**

This chapter is about **predictability and methodological validity**, not yet the final system win.

### Recommended structure

**5.1 Routing target**  
Define exactly what label/benefit is being predicted. Make the target operational and tied to a possible action.

**5.2 Serving-time feature contract**  
List only features actually available before the decision. Separate:
- task/instruction features;
- page/structure features;
- interaction-history/progress features;
- failure/repetition features;
- any model-derived confidence features.

**5.3 Learned router pipeline**  
Feature construction → preprocessing/vectorisation → feature selection if any → classifier → calibration → threshold \(\tau\) → mode choice.

**5.4 Train–serve symmetry and leakage controls**  
This should be unusually explicit. State:
- which transformations are fit within training folds;
- how feature selection is nested;
- how \(\tau\) is selected by inner validation/CV;
- grouping/site policy;
- what information is prohibited.

**5.5 Predictive performance**  
AUROC/AP/calibration with uncertainty and relevant baseline classifiers.

**5.6 Signal ablations**  
Which feature families carry useful information? Distinguish predictive association from mechanism.

**5.7 Generalisation stress tests**  
Site transfer / held-out site / temporal or benchmark split if present in the final experiment.

### Key methodological sentence

> Router-model performance is evaluated only as evidence that the routing target is predictable; the system-level value of acting on those predictions is evaluated separately in Chapter 6.

That one sentence prevents a very common thesis mistake.

### Research-process material worth retaining

If the learned router went through materially different formulations, include only changes that corrected a scientific validity issue, e.g. a leakage risk, train–serve asymmetry, or target mismatch. Do not narrate routine classifier tuning.

### Do not overclaim

- High AUROC does not mean net cost saving.
- Feature importance does not prove why the web agent fails.
- A site-specific classifier is not evidence of cross-site generalisation.
- Threshold selection must not use final test outcomes.

### Chapter-ending handoff

> Chapter 5 shows whether useful routing decisions are predictable under a deployment-faithful information boundary. Predictive accuracy alone is still insufficient: a router can classify well yet lose once false escalations, missed escalations, and its own computation are priced into the system. Chapter 6 therefore executes the router end to end and evaluates the realised success–efficiency trade-off.

---

## Chapter 6 — End-to-End Routing Evaluation

### The one question

**Does routing actually beat defensible fixed alternatives after all relevant overhead is counted?**

This is the chapter where the thesis either earns the “cost-aware routing” claim or narrows it.

### Recommended comparator ladder

1. fixed DOM;
2. fixed SoM;
3. fixed Vision if retained as a primary comparator;
4. best fixed phantom configuration / P-SoM if appropriate;
5. heuristic/fallback router;
6. learned router;
7. oracle upper bound — clearly retrospective, not deployable;
8. strong-model or external comparator only if methodologically comparable.

### Recommended analyses

**6.1 Primary end-to-end result**  
Task success + the primary cost metric, paired on the same tasks.

**6.2 Pareto analysis**  
Show non-dominated fixed and routed systems. Do not hide dominated routed configurations.

**6.3 Router overhead decomposition**  
Decision time, feature extraction, additional parsing/image generation, model calls, retries.

**6.4 Error economics**  
What does a false escalation cost? What does a missed escalation cost in success? The asymmetry helps explain threshold selection.

**6.5 Site heterogeneity**  
Where does routing help or fail? If the aggregate is driven by one site, say so.

**6.6 External validation**  
Online-Mind2Web or final external setting: test transfer of the key claim, not every ablation from VWA.

**6.7 Robustness**  
Alternative thresholds, seeds/splits where available, cost assumptions, exclusions/adjudication sensitivity.

### If the router does not dominate

That is still a thesis result. The chapter chain becomes:

> representation complementarity exists → prediction is possible → but realised gains are small/fragile because overhead, prevalence, errors, or domain shift erase the oracle headroom.

That is scientifically stronger than forcing a “router wins” story.

### End-of-chapter claim template

> Under **[benchmark / sites / model / accounting boundary]**, the routed policy **[improves / does not materially improve]** the **[named success–cost metric]** relative to **[named fixed comparator]**. The result is driven primarily by **[measured mechanism at the system level: escalation frequency, site mix, error asymmetry, overhead]** and does not establish **[broader unsupported claim]**.

### Chapter-ending handoff

> The end-to-end evaluation determines whether the opportunity identified in Chapter 4 and the predictability established in Chapter 5 translate into deployable benefit. The remaining task is interpretation: which lessons are specific to this benchmark and model, which concern representation design more broadly, and how far the measured efficiency gains can support sustainability claims? Chapter 7 addresses those boundaries explicitly.

---

## Chapter 7 — Discussion

### The one question

**What has actually been learned about selective multimodal perception in web agents, and how far can those conclusions travel?**

This chapter should not repeat Results in prose. It should reconcile the evidence across RQ1–RQ3.

### Recommended discussion axes

**7.1 Representation value is conditional**  
Task/site/state dependence; no universal rich-input rule if supported.

**7.2 What the phantom-space result means**  
Text payload, prompt framing, and image presence are separable experimental variables; discuss possible mechanisms cautiously.

**7.3 Oracle headroom versus learnable headroom**  
Why complementarity may exceed realised routing gains.

**7.4 The economics of routing**  
Routing works only when prevalence, value of escalation, prediction quality, and overhead line up.

A useful conceptual relation:

\[
\text{realised routing value}
\approx
\text{selection opportunity}
\times
\text{decision quality}
-\text{routing overhead}
-\text{error cost}.
\]

This is a conceptual decomposition unless the thesis estimates every term directly.

**7.5 Sustainability interpretation**  
What was directly measured? What is only a proxy? Under what assumptions could compute reduction imply lower environmental burden?

**7.6 External validity**  
Model scale/family, benchmark ecology, site mix, online-web instability, action space, local hardware/API pricing.

**7.7 Limitations**  
Write limitations as boundaries on conclusions, not ceremonial disclaimers.

**7.8 Future work generated by the evidence**  
Only future work that follows from a demonstrated bottleneck: better routing labels, richer cheap signals, learning under domain shift, adaptive cascades, improved energy measurement, etc.

### Defence calibration questions

For every major Discussion paragraph, be able to answer:

- Which result supports this?
- Could another mechanism explain it?
- Does the external validation agree?
- What population of agents/tasks does this claim cover?
- Is this measured, inferred, or hypothesised?

### Chapter-ending handoff

> Across the controlled and external analyses, the thesis identifies the conditions under which selective representation has value and the factors that limit its realisation. Chapter 8 closes by answering the research questions at exactly that level of evidence, separating established findings from broader implications.

---

## Chapter 8 — Conclusion

### The one question

**What can be stated defensibly after all experiments, caveats, and scope restrictions are taken into account?**

Do not introduce new evidence.

### Conclusion structure: reuse the 8-beat logic

1. **Problem** — always-on rich perception can waste computation when its information is unnecessary.
2. **Stakes** — repeated web-agent inference magnifies latency/resource costs.
3. **Pivot** — study conditional representation value rather than assuming one globally best mode.
4. **Method** — controlled modes + complementarity analysis + serving-time routing + overhead-aware evaluation.
5. **Finding** — insert only frozen headline results.
6. **Triangulation** — paired tasks, stratification, robustness, external validation.
7. **Scope** — model/benchmarks/sites/accounting boundary.
8. **Implication** — what future agent systems can now test/build more intelligently.

### Final sentence target

The last sentence should describe **what the work enables**, not claim victory:

> These results provide a measured basis for deciding when web agents should pay for richer perception, and a framework for evaluating whether such decisions remain worthwhile once prediction errors and routing overhead are included.

---

# 3. The cross-chapter evidence contract

Every major chapter should be reducible to one row of this table once results are frozen.

| Chapter | Question | Evidence needed | Claim ceiling | Next unresolved question |
|---|---|---|---|---|
| 1 | Why route representations? | motivation + headline gap | routing is worth investigating | what does prior work establish? |
| 2 | What is known/missing? | literature synthesis | controlled overhead-aware conditional selection remains unresolved | how do we compare fairly? |
| 3 | What is the valid comparison? | formalisation + experimental controls | modes/router are operationally defined | is there selection opportunity? |
| 4 | Do modes have complementary value? | fixed-mode paired outcomes + costs + oracle | routing has empirical headroom | can headroom be predicted? |
| 5 | Is need predictable cheaply? | leakage-controlled predictive evaluation | routing target is learnable under stated splits | does acting on it help? |
| 6 | Does routing create net value? | end-to-end paired evaluation + overhead | routed policy's actual trade-off under stated scope | what explains/bounds it? |
| 7 | What does it mean? | synthesis + robustness + external validity | scoped scientific interpretation | what is the final answer? |
| 8 | What can be defended? | no new evidence | exact final thesis contribution | — |

---

# 4. Research journey: where it belongs

Zekun's advice to “write the research process” should mean a **cleaned causal history of scientific decisions**, not a lab diary.

Keep a development episode in the main thesis only if it satisfies at least one of these:

1. an assumption failed and forced a better research question;
2. an apparent result was revealed to be a confound;
3. a new controlled condition was introduced to isolate a variable;
4. a validity threat changed the methodology;
5. a negative result materially bounded the conclusion.

### Good insertion points in this thesis

**Chapter 2 / opening of Chapter 3:**  
The project narrowed from broad efficiency-aware agentic AI to web agents because the literature offered the clearest controlled representation comparisons there.

**Chapter 3:**  
The initial DOM/SoM/Vision comparison bundled several changes together; the later factorisation separated text payload, prompt framing, and image presence. Include this only if it is the reason the six-mode experiment exists.

**Chapter 4:**  
If unexpected image-off performance motivated Phantom-space analysis, present the observation first and the follow-up controlled ablations second.

**Chapter 5:**  
If a potential leakage/train–serve mismatch was discovered and corrected, document the corrected validity rule and why it matters. Do not foreground the buggy implementation.

**Chapter 6:**  
If router overhead or an external benchmark erased an apparent gain, make that part of the scientific result rather than hiding it.

### Bad research-journey prose

> First I tried X. It did not work, so then I tried Y. After debugging for two weeks I changed Z.

### Better

> The initial three-mode comparison could not distinguish the effect of image presence from changes in textual grounding and prompt format. We therefore introduced controlled image-off variants that altered these factors separately.

The second version explains **why the experiment exists**.

---

# 5. Chapter opener/closer templates

Do not copy these mechanically; use them to test structure.

### Opening template

> The previous chapter established **X**, but it left **Y** unresolved. This chapter addresses that question by **Z**.

### Closing template

> The analysis shows **A**, supported by **B**, under **C scope**. It does not establish **D**. Resolving **D** requires **E**, which is the focus of the next chapter.

If a chapter cannot fill these blanks, it probably does not yet have a single job.

---

# 6. Paragraph-level mini-chain

The same logic should operate below chapter level.

For important subsections:

1. **Question / motivation** — why are we looking at this?
2. **Method / comparison** — what would answer it?
3. **Observation** — what happened?
4. **Interpretation** — what does that support?
5. **Boundary** — what does it not show?
6. **Next move** — what question follows?

This prevents Results from becoming a sequence of figures with captions disguised as prose.

---

# 7. Red flags specific to this thesis

### Red flag 1 — The thesis becomes the Phantom-SoM paper

Fix: keep the top-level story as **conditional representation value → predictability → routing**. Phantom-space is a major empirical finding/representation analysis inside that story.

### Red flag 2 — The learned router becomes the only contribution

Fix: Chapter 4 must establish a publishable scientific result even if Chapter 6 produces weak routing gains.

### Red flag 3 — AUROC is treated as the answer

Fix: Chapter 5 explicitly stops at predictability. Chapter 6 owns system benefit.

### Red flag 4 — “cost-aware” without accounting

Fix: name the cost variable and include router overhead. If only latency is measured robustly, say latency-aware rather than pretending to cover every resource.

### Red flag 5 — “sustainable” from token count alone

Fix: distinguish computational efficiency from environmental sustainability and state the bridge/assumptions.

### Red flag 6 — Too much jargon before the reader sees a web page

Fix: early worked example. Show one page/task and explain what DOM/SoM/Vision expose before introducing the full taxonomy.

### Red flag 7 — Background becomes a textbook

Fix: include a concept only if a later method/result relies on it. A neighbouring-field examiner needs enough mechanism to follow the argument, not a survey of transformers.

### Red flag 8 — Appendix becomes an evidence graveyard

Fix: every appendix item must have a main-text pointer and a reason it is not in the body. Exact allocation waits for the 2025/26 handbook.

---

# 8. What should probably live in the appendix — provisional only

**Do not finalise this until the COMP0191 2025/26 page/appendix rules are obtained.** This is only a logical classification.

Likely appendix candidates:

- full prompt templates;
- complete action schema;
- long hyperparameter/configuration tables;
- per-site/per-mode secondary result tables;
- extra seeds and sensitivity curves;
- full router feature dictionary;
- extended calibration plots;
- complete external-validation breakdown;
- detailed environment failure adjudication;
- representative raw trajectories/log excerpts;
- reproducibility/configuration manifests.

Likely **not** appendix material:

- the main observation-mode definition;
- the primary system diagram;
- headline fixed-mode comparison;
- the main oracle/complementarity result;
- the primary router result;
- the central limitation needed to interpret a headline claim.

Rule: if removing an item from the main text makes a headline claim hard to verify, it probably belongs in the body.

---

# 9. Argument-freeze checklist

Before turning this into the final table of contents:

- [ ] Freeze the final RQ wording against the canonical result tables.
- [ ] Decide whether six modes all remain primary or some become structural ablations.
- [ ] Lock the exact router target and serving-time feature set.
- [ ] Verify that Chapter 4's oracle unit matches Chapter 5's router decision unit, or explain the difference.
- [ ] Freeze primary cost metric and accounting boundary.
- [ ] Freeze benchmark-noise exclusion/adjudication policy.
- [ ] Determine which external-validation result is strong enough for the main text.
- [ ] Replace every provisional claim with a result + uncertainty + scope.
- [ ] Obtain the **2025/26 COMP0191 final-project handbook / Moodle brief** and only then allocate pages and appendices.
- [ ] Build `FIGURE_PLAN.md` from the frozen chapter questions and the verified page/appendix constraints.

---

# 10. The shortest possible thesis story

If an examiner asks for the thesis in 30 seconds, the chapters should collapse to this:

> **Ch1–2:** Rich page representations can help web agents but cost more, and prior work does not tell us when that cost is worth paying.  
> **Ch3:** We create a controlled way to vary representation while holding the rest of the agent fixed.  
> **Ch4:** The modes solve different subsets of tasks and occupy different cost–success points, so selection has measurable headroom.  
> **Ch5:** We test whether that headroom is predictable from cheap serving-time signals without leakage.  
> **Ch6:** We act on those predictions and ask whether the resulting router beats fixed policies after overhead.  
> **Ch7–8:** We explain where the result holds, why it is limited, and what it means for selective multimodal inference.

If the final thesis cannot be compressed to something this coherent, fix the chapter chain before polishing sentences.
