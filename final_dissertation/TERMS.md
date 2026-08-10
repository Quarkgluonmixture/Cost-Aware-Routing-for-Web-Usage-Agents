# TERMS.md — Thesis Term Lock

**Project:** Cost-Aware Routing for Web Usage Agents  
**Status:** working terminology contract  
**Last updated:** 2026-08-09

This file is not a glossary for the reader. It is a **writing constraint for the author**.

> **Term-lock rule:** one concept gets one canonical name. One canonical name gets one definition.  
> Explain the mechanism in plain language at first use, then introduce the technical term. Do not silently switch labels later because another phrase sounds nicer.

The operational definitions below follow the current P79 experimental framing and later Phantom-space analysis. If implementation names and paper-facing names differ, the paper-facing term below wins unless the code must be quoted literally.

---

## 1. Core research objects

| Canonical term | Thesis definition / intended use | Do not casually substitute |
|---|---|---|
| **web agent** | An AI system that observes a web page, chooses an action, executes it in a browser environment, and repeats this loop to complete a user task. Use **web agent** after first use. | browser AI, browsing model, GUI agent, autonomous browser — unless the distinction matters |
| **task** | The user-level objective evaluated by the benchmark, e.g. finding, selecting, entering, or changing information on a website. | query, prompt, episode |
| **episode** | One attempted execution of one benchmark task from initial state until success, termination, error, or the step limit. | task |
| **step** | One observe–decide–act iteration within an episode. | turn, action |
| **action** | The browser operation chosen at a step, such as click, type, select, scroll, back, forward, wait, or finish. | step |
| **observation** | The information exposed to the agent at one step before it selects an action. | page, state, representation |
| **page representation** | The encoded view of the current page supplied to the agent. Different representations expose different structural and/or visual information. | modality, observation mode |
| **observation mode** | A complete, executable configuration specifying what page representation and prompt format the agent receives at a step. | representation alone |
| **agent policy** | The mapping from the information available to the agent to its browser action. | router policy |
| **router** | A separate decision mechanism that chooses which observation mode the agent should receive for a step. It does **not** itself choose the browser action. | planner, agent, oracle |
| **routing decision** | The router's choice of observation mode at a particular step. | agent action |
| **fixed policy** | A non-routed condition that uses the same observation mode for every step. | baseline, static model — unless qualified |
| **routed policy** | A condition in which the observation mode may change across steps according to a routing rule or learned router. | adaptive agent — too broad |

### 1.1 The three sides — how the six modes group  (locked 2026-08-10)

The six observation modes are best introduced by **what is actually sent to the
model**, not by their historical names. This grouping is verified from the step
records, not asserted: the four text-side modes all record
`image_payload_bytes == 0`.

| Canonical term | Members | What is sent |
|---|---|---|
| **text side** | DOM · P-text · P-prompt · P-SoM | structured text only — **no image at all** |
| **combined side** | SoM | the same page as text **and** as an annotated screenshot |
| **visual side** | Vision | the screenshot only — **no structured text** |

The text side is itself a 2×2 of **text format** (AXTree vs `[SOM_MARKS]`) ×
**prompt style** (DOM-style vs SoM-style); the `mark_count` field encodes the
first axis directly (30 = `[SOM_MARKS]`, 0 = AXTree). So "the three sides" and
"the 2×2" are the same structure at two zoom levels, not competing framings.

**Why this ordering matters for the argument.** Grouping this way makes one
result legible that is otherwise puzzling: the three phantom arms have small
drop-one contributions (0.00–2.68pp) because they all live **inside one side**
and overlap heavily, while the larger unique contributions come from **crossing
sides**. See F9.

**Do not** call the text side "phantom" — *phantom* names the three text-side
arms **other than DOM** (P-text, P-prompt, P-SoM), and DOM is text-side too.

### First-use sentence pattern

Use this style in the thesis:

> At each browser step, the agent receives a **page representation**: a structured or visual encoding of the current page. An **observation mode** specifies the complete input configuration used to construct that representation and prompt the agent.

Then lock the terms.

---

## 2. Structured page representations

### 2.1 AXTree

**Canonical term:** **accessibility tree (AXTree)**

A structured textual representation derived from browser accessibility information, containing elements, roles, names, states, and identifiers useful for interaction.

**Critical lock:** in this project, the condition historically named `dom` is primarily based on **AXTree text**, not raw HTML DOM serialization. In prose:

- first use: **AXTree-based DOM mode (`dom`)** if the historical/code label must be preserved;
- afterwards: use **DOM mode** only after explicitly defining what that label means in this thesis;
- when discussing the actual information channel, say **AXTree**.

Do **not** write statements such as “the model reads the raw DOM” unless that is literally true for that analysis.

### 2.2 `[SOM_MARKS]`

**Canonical term:** **SoM mark text (`[SOM_MARKS]`)**

A flattened textual list of interactable elements produced from the page's structured representation and indexed in the style used for Set-of-Marks grounding.

It is **text**, not an image, and not equivalent to the complete SoM observation.

### 2.3 annotated screenshot

**Canonical term:** **annotated screenshot**

A screenshot on which candidate web elements are overlaid with visible indexed marks/bounding boxes for visual reference.

Use **annotated screenshot** when image presence matters. Do not use “SoM” to mean only the image overlay.

### 2.4 element identifier

**Canonical term:** **element ID**

An identifier exposed by a structured observation and used by structured action grounding. Treat IDs as step-local implementation handles unless evidence shows stability across renders.

Do not infer semantic continuity from identical-looking ID values across page renders.

---

## 3. Observation modes: canonical six-mode vocabulary

The six-mode terminology below should be treated as the current paper/thesis-facing vocabulary unless the final analysis freezes a smaller set.

| Mode | Text payload | Prompt framing | Image | Canonical description |
|---|---|---|---|---|
| **DOM** (`dom`) | AXTree | DOM-style | No | Structured-text baseline using AXTree information only |
| **P-prompt** | AXTree | SoM-style | No | AXTree text with SoM-style instructions but no image |
| **P-text** | `[SOM_MARKS]` | DOM-style | No | Flattened SoM mark text with DOM-style instructions and no image |
| **P-SoM** | `[SOM_MARKS]` | SoM-style | No | SoM mark text and SoM-style instructions, but no annotated image |
| **SoM** (`som`) | `[SOM_MARKS]` | SoM-style | Yes, annotated | Full Set-of-Marks condition with mark text plus annotated screenshot |
| **Vision** (`vision`) | None | vision-oriented | Yes, raw | Screenshot-only condition without AXTree/element-ID text |

### 3.1 Set-of-Marks (SoM)

**Canonical term:** **Set-of-Marks (SoM)**

In this thesis, **SoM** means the complete full observation mode that combines indexed element information with an **annotated screenshot** and the corresponding SoM-style prompt.

Do not use **SoM** as a synonym for:

- `[SOM_MARKS]` text alone;
- bounding boxes alone;
- any numbered UI representation;
- P-SoM.

### 3.2 phantom routing space

**Canonical term:** **phantom routing space**

The subset of the text/prompt/image configuration space in which the annotated image is omitted while structured textual information and prompt framing can still vary. In the current factorisation, the image-off configurations are **DOM, P-prompt, P-text, and P-SoM**.

**Critical lock:**

- **phantom routing space** = a family / region of configurations;
- **P-SoM** = one specific configuration inside that space.

Never use the two interchangeably.

### 3.3 phantom mode

Use **phantom mode** only when referring generically to one of the image-off configurations introduced by the factorised analysis. Prefer the exact mode name whenever possible.

### 3.4 representation axis

If the factorisation is retained in the final thesis, use these axis names consistently:

1. **text-payload axis**: AXTree vs `[SOM_MARKS]`;
2. **prompt-framing axis**: DOM-style vs SoM-style prompt;
3. **image-presence axis**: annotated image absent vs present.

**Vision** is an image-only condition and should not be forced into the same structured-text cube if doing so makes the factorisation misleading.

---

## 4. Routing vocabulary

| Canonical term | Definition | Important boundary |
|---|---|---|
| **routing signal** | A variable available to the router that may predict which observation mode will be useful. | A signal is not a routing decision. |
| **cheap routing signal** | A routing signal whose acquisition does not require paying the main cost of the richer candidate observation it is intended to gate. | “Cheap” must be justified by accounting, not assumed. |
| **pre-decision signal** | A signal available before the router chooses the current step's observation mode. | Do not train on information unavailable at serving time. |
| **failure signal** | Evidence that recent interaction has not produced the intended progress, e.g. execution error, repeated action, or no page change. | `page_changed = false` is not by itself action failure. |
| **progress signal** | Evidence about whether the interaction is advancing toward task completion. | Keep separate from browser execution success. |
| **escalation** | Switching from a cheaper/lower-input mode to a richer or more costly mode because the routing rule predicts extra information is worthwhile. | Not every mode switch is escalation unless the ordering is defined. |
| **fallback** | A corrective switch used after a specific failure or unsupported interaction. | Fallback is reactive; routing can also be proactive. |
| **heuristic router** | A router whose decisions are defined by manually specified rules or thresholds. | Do not call it learned. |
| **learned router** | A router whose decision function is estimated from training data. | Specify target, features, split, calibration, and threshold. |
| **routing threshold** \(\tau\) | The decision threshold applied to a router score/probability to convert it into a routing action. | State how \(\tau\) is selected; avoid tuning on test outcomes. |
| **router score** | A continuous model output estimating the routing target or benefit of escalation. | Not automatically a calibrated probability. |
| **routing target** | The supervised quantity the learned router is trained to predict. | Must correspond to a serving-time decision. |
| **train–serve symmetry** | The requirement that the features, transformations, and information available while training the router match what will be available when it is deployed. | Violations are leakage or deployment mismatch. |
| **data leakage** | Test/future/otherwise unavailable information influences fitting, feature selection, calibration, or threshold choice. | Includes subtle preprocessing leakage. |
| **site-aware split** | A data split or analysis that explicitly accounts for website/domain structure. | Distinguish from random step-level splitting. |

---

## 5. Oracle terminology

| Canonical term | Definition |
|---|---|
| **oracle** | A retrospective upper-bound selector that is allowed to use outcome knowledge unavailable to a real router to choose the best mode. |
| **mode oracle** | Oracle that selects among observation modes for each unit of analysis. State whether the unit is task, episode, or step. |
| **drop-one oracle analysis** | Recompute oracle performance after removing one candidate mode; the drop estimates that mode's unique contribution to the oracle set. |
| **oracle gain / oracle lift** | Difference between oracle performance and a specified non-oracle comparison set. Always name the comparator. |
| **routing ceiling** | The maximum benefit suggested by the oracle under the same candidate modes and evaluation unit. |

**Critical lock:** oracle performance demonstrates **complementarity / potential value of selection**. It does **not** demonstrate that a deployable router can realise that value.

---

## 6. Outcome metrics

### 6.1 task success

**Canonical term:** **task success**

Binary benchmark outcome indicating whether the user-level task was completed according to the benchmark evaluator.

### 6.2 task success rate (TSR)

**Canonical abbreviation:** **TSR** after first definition.

\[
\mathrm{TSR}=\frac{\text{successful tasks}}{\text{attempted evaluable tasks}}
\]

State explicitly how blocked, invalid, crashed, or benchmark-noise episodes enter the denominator.

### 6.3 action execution success

Whether the browser successfully executed the requested operation at a step.

**Never equate this with task success.** A click can execute successfully while moving the agent away from the goal.

### 6.4 page-change signal

**Canonical field:** `page_changed`

A diagnostic indicating whether the observable page state changed after an action under the project's change detector.

**Critical lock:** `page_changed = false` means **no detected page change**, not “action failed”. Some valid actions may not cause the detector to register a change.

### 6.5 no-op / no-progress

Use these only with explicit operational definitions.

- **no-op**: an action whose execution produces no relevant observable state change under the defined detector;
- **no-progress**: a broader task-level or trajectory-level judgement that the interaction has not advanced.

Do not use them interchangeably.

### 6.6 step count

Number of agent interaction steps used in an episode. State treatment of terminal `finish` and retries if those affect counting.

---

## 7. Efficiency and cost vocabulary

### 7.1 efficiency

**Canonical term:** **efficiency** is an umbrella concept, not a directly observed scalar unless the thesis defines one.

When possible, name the measured quantity instead:

- latency;
- token usage;
- monetary cost;
- energy;
- CO₂e;
- number of model calls;
- router overhead.

Avoid sentences such as “method A is 30% more efficient” unless the exact efficiency metric is stated.

### 7.2 latency

Keep these distinct:

- **step latency**: wall-clock time for one observe–decide–act step under the stated boundary;
- **episode latency**: total wall-clock time for one task attempt;
- **model inference latency**: time attributable specifically to model inference;
- **P95 latency**: 95th percentile of the explicitly named latency distribution.

### 7.3 token usage

Number of text and/or model-accounted input/output tokens under the named tokenizer/provider accounting. If image tokens are provider/model-specific, state the accounting convention.

### 7.4 monetary cost

Estimated or billed currency cost under a specified model/provider price schedule. Do not present local-model GPU time as “monetary cost” unless a pricing model is explicitly defined.

### 7.5 energy

Measured or estimated energy use under a stated measurement boundary and method. Keep device energy, system energy, and model-attributable energy separate where relevant.

### 7.6 CO₂e

**Canonical term:** **carbon dioxide equivalent (CO₂e)**

An estimated climate-impact quantity derived from energy and an explicitly stated carbon-intensity assumption/source.

Do not use **carbon**, **emissions**, **CO₂**, and **CO₂e** interchangeably.

### 7.7 router overhead

The additional computation and latency caused by making and acting on a routing decision, including any feature extraction, extra parsing, image preparation, auxiliary model calls, or retries that would not occur in the comparator.

### 7.8 net saving

Use only when the accounting boundary is explicit. Generic form:

\[
\mathrm{NetSaving} = C_{\text{fixed comparator}}-
\left(C_{\text{routed execution}}+C_{\text{router overhead}}\right)
\]

Name the cost variable \(C\) each time: latency, tokens, money, energy, etc.

---

## 8. Trade-off language

### 8.1 success–efficiency trade-off

**Preferred thesis phrase:** **success–efficiency trade-off**

Reason: the benchmark's primary end outcome is task **success**, not classification accuracy.

### 8.2 cost–success trade-off

Use when the x-axis is a concrete cost variable and the y-axis is task success.

### 8.3 cost–accuracy

Reserve **cost–accuracy** for:

- the historical project title;
- a source that literally measures accuracy;
- a local component whose outcome is accuracy.

Do not use it as the default description of the web-agent thesis.

### 8.4 Pareto frontier

A set of non-dominated system configurations for two or more explicitly defined objectives, e.g. task success versus latency or cost.

A point is **Pareto-dominated** only if another configuration is at least as good on every stated objective and strictly better on at least one.

Do not call a single scalar ranking a Pareto analysis.

---

## 9. Predictive/router evaluation

| Canonical term | Use |
|---|---|
| **AUROC** | Ranking quality across all classification thresholds. It does not encode the deployment cost asymmetry by itself. |
| **average precision (AP)** | Precision–recall summary useful under class imbalance. |
| **calibration** | Agreement between predicted probabilities and empirical outcome frequencies. |
| **cross-validation** | Repeated held-out estimation within the training/development process. State grouping/stratification. |
| **inner cross-validation** | Cross-validation used inside an outer training partition for model/threshold selection; test data remains untouched. |
| **feature ablation** | Controlled removal of a feature or feature family to assess its contribution. |
| **baseline router** | A specified non-proposed routing strategy used for comparison. Always name it, e.g. majority, heuristic, random, or phantom-prompt baseline. |

**Critical lock:** good AUROC means the routing target is **predictable to some degree**. It does not by itself mean the routed system improves task success, cost, or sustainability.

---

## 10. Statistical language

| Canonical term | Writing rule |
|---|---|
| **estimate** | The observed statistic calculated from the sample. |
| **confidence interval (CI)** | State level and resampling/analytic method. |
| **statistically significant** | Use only with a defined test, null, and threshold. Prefer effect size + CI over significance-only prose. |
| **trend** | A descriptive pattern that is not being asserted as a statistically established difference. |
| **robustness check** | An analysis testing whether a conclusion survives a plausible alternative specification, split, seed, metric, or subset. |
| **external validation** | Evaluation on data/benchmark not used to develop the main method. Do not call another split of the same benchmark “external”. |

Avoid “proves”, “confirms”, and “demonstrates causally” unless the design genuinely supports that strength.

---

## 11. Explanation and claim levels

These four levels must stay separate in prose.

### Level 1 — observation

What the measured data show.

> P-text has lower measured latency than SoM under this setup.

### Level 2 — empirical pattern

A repeated relationship across conditions/subsets.

> Image-off modes tend to be more competitive on the text-dominated site than on the visually demanding site.

### Level 3 — mechanistic interpretation

A proposed explanation for the pattern.

> One possible explanation is that flattening the structured text changes how the model frames element selection.

Use **may**, **is consistent with**, **suggests**, or **we hypothesise** unless directly tested.

### Level 4 — broader implication

What the result could mean outside the exact experiment.

> Selective multimodal inference may be preferable to always-on vision when information value varies by interaction state.

This requires the strongest scope calibration.

---

## 12. Terms for heterogeneity and complementarity

| Canonical term | Definition |
|---|---|
| **mode complementarity** | Different modes succeed on partially non-overlapping tasks/steps, creating potential value for selection. |
| **unique success** | A task/step solved by one mode and not by the specified comparison modes. Name the comparison set. |
| **site dependence** | An effect differs materially across websites/domains in the benchmark. |
| **state dependence** | The relative value of a representation varies with the current interaction state/step rather than only across whole tasks. |
| **task dependence** | The relative value of a representation varies across user-level tasks. |
| **representation value** | The incremental decision/task benefit attributable to making a representation available under a defined comparison. Avoid implying causal identification unless design supports it. |

---

## 13. Sustainability terminology

### 13.1 sustainable agentic AI

Use **sustainable agentic AI** as the broad project motivation, not automatically as a measured outcome.

The thesis directly studies efficiency-related quantities. A claim about environmental sustainability requires an explicit bridge from those quantities to environmental impact.

### 13.2 computational efficiency

Reduced computation or resource use for a defined level of task performance, measured using named proxies/metrics.

### 13.3 environmental sustainability

A broader concept than computational efficiency. Do not equate lower tokens or latency with lower environmental impact without stating the assumptions and system boundary.

### Preferred calibration

Good:

> The routed policy reduces measured inference cost under our accounting boundary, which can be relevant to environmental sustainability when lower computation translates into lower energy use.

Too strong:

> The router is more sustainable because it uses fewer tokens.

---

## 14. Benchmark and dataset terminology

| Canonical term | Definition / policy |
|---|---|
| **VisualWebArena (VWA)** | Primary controlled multimodal web-agent benchmark when referring specifically to that benchmark. |
| **WebArena** | Related benchmark/environment; do not use as a synonym for VisualWebArena. |
| **Online-Mind2Web** | External-validation setting when using the online benchmark/task formulation. Spell consistently. |
| **site** | A benchmark website/domain such as Classifieds, Reddit, or Shopping. |
| **benchmark noise** | Failures caused by environment availability, blocking, anti-bot behaviour, or infrastructure rather than the agent policy, under an explicit adjudication rule. |
| **evaluable task** | A task that passes the predetermined environment/benchmark validity criteria and enters the analysis denominator. |

Do not retroactively remove hard tasks under the label “benchmark noise”. The exclusion rule must be independent of system outcome.

---

## 15. Baseline terminology

Use the most specific name possible.

- **fixed DOM baseline**
- **fixed SoM baseline**
- **fixed Vision baseline**
- **best fixed policy**
- **heuristic router**
- **learned router**
- **oracle upper bound**
- **strong-model upper bound** — only if it is genuinely an upper-bound comparator for the stated purpose

**Do not use “baseline” to mean every non-proposed condition.** A structural ablation, oracle, and upper bound play different inferential roles.

---

## 16. Research-process terminology

Use these distinctions when narrating how the thesis evolved.

| Canonical term | Meaning |
|---|---|
| **screening experiment** | Early controlled comparison used to determine whether a research direction is empirically worth pursuing. |
| **ablation** | Controlled removal/change of one component to isolate its contribution. |
| **diagnostic analysis** | Analysis used to understand behaviour/failure, not to establish the primary headline effect. |
| **exploratory analysis** | Analysis not fixed as a primary test before inspecting the relevant data. Label it accordingly. |
| **confirmatory analysis** | Analysis whose hypothesis, metric, comparison, and decision rule were fixed before the relevant results were inspected. |
| **preregistration / pre-specified rule** | Use only for decisions actually recorded before seeing the relevant outcome. |

Do not turn the chronological development log into the thesis narrative. Keep only decisions that explain **why the final experiment exists or how an unexpected result generated a defensible next question**.

---

## 17. Hard forbidden substitutions

These are likely sources of examiner confusion. Treat them as lint rules.

1. **DOM ≠ raw HTML DOM ≠ AXTree.** Define historical `dom` mode precisely.
2. **SoM ≠ `[SOM_MARKS]` ≠ annotated screenshot.**
3. **P-SoM ≠ phantom routing space.** One is a mode; one is a family.
4. **observation mode ≠ modality.** Mode includes payload + prompt + image configuration.
5. **router action ≠ browser action.**
6. **action execution success ≠ task success.**
7. **`page_changed = false` ≠ action failure.**
8. **AUROC ≠ routed-system benefit.**
9. **oracle gain ≠ deployable router gain.**
10. **latency ≠ cost ≠ energy ≠ CO₂e.**
11. **computational efficiency ≠ environmental sustainability.**
12. **task success ≠ accuracy.** Prefer success–efficiency over cost–accuracy.
13. **site dependence ≠ generalisation.** A site-stratified effect is not external validity.
14. **external validation ≠ another in-domain split.**
15. **mechanistic hypothesis ≠ measured finding.**

---

## 18. First-use definitions worth drafting almost verbatim

These are intentionally plain-language-first.

### Router

> Richer page inputs can help a web agent but cost more to process. We therefore introduce a **router**, a separate decision mechanism that chooses which observation mode the agent receives at each interaction step.

### Observation mode

> A web page can be shown to the agent as structured text, an image, or a combination of both. We call each complete input configuration an **observation mode**.

### AXTree

> Rather than serialising the full HTML page, our structured-text condition exposes the browser's accessibility representation, which records interactable elements, their roles and names. We refer to this representation as the **accessibility tree (AXTree)**.

### Set-of-Marks

> A visual agent can be given an annotated screenshot in which candidate elements are assigned visible indices, making them easier to reference during action selection. We refer to this grounding scheme as **Set-of-Marks (SoM)**.

### Phantom routing space

> Removing the annotated image does not leave a single text-only condition: the textual payload and the prompt framing can still vary independently. We call this family of image-off configurations the **phantom routing space**.

### Pareto trade-off

> A routed system is useful only if any performance gain justifies its additional cost. We therefore compare configurations on a **Pareto frontier**, where a system is preferred only when no alternative is at least as good on all measured objectives and strictly better on one.

---

## 19. Open term locks — resolve before prose freeze

These are deliberately **not** silently fixed yet.

- [ ] Exact final name of the learned router and each training stage.
- [ ] Exact final supervised routing target: “rich-mode benefit”, “escalation need”, or another operational label.
- [ ] Whether the final thesis uses **Phantom-SoM** as a named method, a discovered condition, or only one point in the factorised representation space.
- [ ] Final unit for oracle/routing labels: task-level, step-level, or both.
- [ ] Final definition of “no-op” and the detector used.
- [ ] Final cost accounting boundary for local inference.
- [ ] Whether energy and CO₂e are primary metrics, secondary estimates, or omitted from headline claims.
- [ ] Final external-validation name and scope for Online-Mind2Web.
- [ ] Final statistical names after the analysis code/table is frozen.
- [ ] Exact terminology for any Stage-1 / Stage-2 router components; avoid generic stage labels without definitions.

---

## 20. Final term-lock lint before submission

Run a literal search over the thesis for:

`DOM`, `AXTree`, `SoM`, `SOM_MARKS`, `P-SoM`, `phantom`, `accuracy`, `success`, `failure`, `page_changed`, `cost`, `efficient`, `sustainable`, `carbon`, `CO2`, `CO₂e`, `oracle`, `router`, `baseline`, `generalise`, `prove`, `significant`.

For every hit, ask:

1. Is this the canonical concept?
2. Was it defined before first technical use?
3. Is the comparison set/boundary stated?
4. Does the evidence support this exact strength of claim?
5. Would an examiner interpret the same word the same way everywhere else in the thesis?

If not, fix the term rather than adding explanatory prose around an unstable vocabulary.
