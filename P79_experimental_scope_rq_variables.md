# P79 Experimental Scope (Supervisor / Research Version, v2)

## Project title
**Towards Sustainable Agentic AI: Implementing a Cost-Accuracy Router for LLMs**

---

## 0) What this document is for
This is the **research / supervisor-facing experimental framing** for P79.

It is narrower than the coursework scope and assumes that the literature review has already justified narrowing toward **web agents** as the most tractable setting for studying efficiency-aware routing.

This document is for:
- supervisor discussion,
- implementation planning,
- baseline design,
- ablation control,
- and later dissertation reporting.

---

## 1) Experimental scope

### Proposed experimental scope
**Cost-aware routing for multimodal web agents on VisualWebArena, with Set-of-Marks treated as a top-level representation variable and router overhead explicitly accounted for**

### Why this scope
This scope assumes:
- **VisualWebArena / WebArena** is the primary benchmark because it is controlled and reproducible,
- **Online Mind2Web** is optional and used only for small external validation,
- **blocked / anti-bot / geo-restricted cases** are treated as benchmark noise rather than agent failure,
- the contribution is **not** building the most complex agent stack,
- but studying whether routing improves the **success–efficiency trade-off** once overhead is included.

---

## 2) Experimental research questions

### Experimental RQ1
**When is a structured page representation sufficient, and when is screenshot-based vision necessary, for web-agent task completion under controlled benchmarks?**

Operational reading:
- compare DOM / AXTree / SoM / hybrid conditions,
- evaluate trade-offs in success, step count, latency, and efficiency.

### Experimental RQ2
**Can a cost-aware router that switches between page representations and lightweight helper modules improve the Pareto trade-off between task success and efficiency compared with fixed single-policy baselines?**

Operational reading:
- compare fixed policies vs routed policies,
- evaluate success–latency–cost–CO₂e trade-offs.

### Experimental RQ3
**Under what conditions does routing overhead negate its own savings, and which task- or step-level signals best predict whether routing will produce a net benefit?**

Operational reading:
- explicitly measure router overhead,
- test failure signals, DOM size, repetition, and progress signals,
- compute net saving.

---

## 3) Working hypotheses

### H1
Using richer representations (e.g. screenshot-based vision or Set-of-Marks-augmented representations) will often improve task success on visually ambiguous or interaction-heavy steps, but usually increases latency, cost, and energy / CO₂e.

### H2
A simple cost-aware router that defaults to cheaper representations and escalates only under failure or complexity signals will achieve a better Pareto trade-off than a fixed always-rich policy.

### H3
Routing is not always beneficial; on simpler tasks or short-horizon steps, router overhead can outweigh the savings from selective escalation.

---

## 4) Router modelling framework
Zekun’s recommendation is to formalise the router as an **abstract decision policy** rather than treating each routing trick separately.

### 4.1 State
At interaction step \(t\), define the router state as:
\[
s_t = (c_t, h_t, p_t, f_t)
\]
where:
- **\(c_t\)** = task context (instruction, current sub-goal),
- **\(h_t\)** = interaction history (recent actions, observations, retries),
- **\(p_t\)** = current page signals (DOM/AXTree size, SoM availability, screenshot availability, page complexity),
- **\(f_t\)** = failure or uncertainty signals (page unchanged, selector not found, repeated no-progress behaviour).

### 4.2 Action space

The full representation space is the Cartesian product of two orthogonal factors:

| Factor | Levels | Varied by |
|---|---|---|
| **SoM overlay** | OFF / ON | Fixed per experimental condition (top-level variable A1) |
| **Observation channel** | DOM-only / Hybrid (DOM + screenshot) | Switched dynamically by the router (variable B1) |

Because SoM is a **between-condition** factor fixed before each run, it is not part of the router's action space. The router operates **within** a given SoM setting and selects only the observation channel:

\[
a_t \in \mathcal{A} = \{\texttt{dom\_only},\; \texttt{hybrid}\}
\]

This yields a compact two-action routing problem per SoM condition, avoiding the combinatorial blow-up of a four-action space while still allowing Phase 1 to evaluate all four representation cells (SoM × observation) as fixed policies.

### 4.3 Policy
The router policy is:
\[
a_t = \pi(s_t)
\]

Within this framework, every experimental condition is a special case of the same policy family:
- **Fixed policy** \(\pi_{\text{fix}}\): always select the same observation channel (e.g., always DOM-only). The four Phase 1 cells are four such fixed policies.
- **Heuristic / threshold router** \(\pi_{\text{rule}}\): select observation channel based on page complexity, uncertainty, or progress signals derived from \(s_t\).
- **Fallback router** \(\pi_{\text{fall}}\): default to the cheaper channel (DOM-only) and escalate to Hybrid when failure signals exceed a threshold.

### Why this matters
This lets the dissertation compare all conditions — including fixed baselines — **within a common formal framework**, rather than treating routing as an unrelated engineering trick. The SoM factor is analysed as an exogenous moderator: Phase 1 tests whether SoM shifts the success–efficiency frontier, while Phase 2 tests whether routing improves the trade-off **conditional on** the chosen SoM setting.

---

## 5) Variable hierarchy
This hierarchy is designed specifically to avoid **combinatorial explosion**.

### 5.1 Primary variables (systematically varied)

#### A. Representation variables
These sit at the top of the hierarchy.

##### A1. Set-of-Marks (SoM)
- OFF
- ON

**Reason:** SoM changes the representation itself and must be treated as a top-level representation variable, not as a minor switch.

##### A2. Observation channel
- DOM / AXTree-only
- Hybrid (DOM / AXTree + screenshot)

**Reason:** this is the central representation comparison for the project.

#### B. Routing variable
##### B1. Router policy
- OFF (fixed representation policy)
- ON (rule-based routing)

**MVP rule-based router idea:**
- default to the cheaper representation action,
- escalate to richer representation if:
  - DOM size exceeds threshold,
  - action fails,
  - page state does not change,
  - repeated no-progress steps are detected.

---

### 5.2 Secondary variables (one-at-a-time ablations only)
These are **not** multiplied together early on.

#### M1. DOM-only Select fallback
Use DOM control for native select-like controls.

#### M2. DOM-first Input fallback
Search DOM for inputs first; fall back to vision-based interaction if needed.

#### M3. Failure trigger + one retry
Use Playwright status and page-change checks to trigger a single correction attempt.

#### M4. Two-stage Action Generation / Action Grounding
Split history-heavy planning from current-page grounding.

**Rule:** introduce these **one at a time**, only after the best primary representation condition is selected.

---

### 5.3 Deferred variables (not part of early-stage core study)
These are postponed to avoid variable multiplication:
- Checklist manager
- Memory summarisation
- EIP / web-search planning
- Large model sweep beyond what is needed for baseline
- Online Mind2Web full-scale study
- Learning-based router instead of rule-based router

---

## 6) Experimental phases

### Phase 1 — Representation screening
Purpose:
- choose the best representation setup before adding routing and modules.

#### Suggested 2×2 grid
- SoM: OFF / ON
- Observation: DOM-only / Hybrid

Output:
- success / TSR
- average steps
- no-op rate
- P95 step latency
- basic cost / energy logging

Decision:
- keep the best one or two conditions only.

---

### Phase 2 — Routing study
Purpose:
- evaluate whether routing improves the trade-off relative to fixed policy.

#### Comparison
- best fixed representation policy from Phase 1
- same setup + rule-based router

Output:
- Pareto comparison
- overhead-aware net saving
- failure-trigger distribution
- escalation frequency

---

### Phase 3 — Module ablation
Purpose:
- test whether lightweight stabilisers improve the routed or fixed policy enough to justify their overhead.

#### Order
1. M1 DOM-only Select
2. M2 DOM-first Input
3. M3 Failure trigger + one retry
4. M4 Two-stage Action Generation / Grounding

**Rule:** each module is tested against the same base condition; do not enable all modules together in the early stage. No interaction effects (e.g., M1+M3) are tested in the early stage; if time allows, at most one cumulative bundle is evaluated.

---

### Statistical design and power considerations

#### Sample sizes
VisualWebArena provides approximately 910 tasks across three standalone sites (shopping ~250, reddit ~230, classifieds ~450) plus cross-site Wikipedia references embedded in reddit/shopping configs. Each task is run once per condition per seed. With the Phase 1 2×2 grid this yields ~910 episodes per condition.

#### Seed strategy
The primary analysis uses a single seed (seed = 42). If Phase 1 success rates are close between conditions (e.g., within 3 pp), a confirmation run with two additional seeds (123, 456) is triggered to assess stability. Multi-seed runs are not the default because each seed triplicates wall-clock time on a single-GPU setup.

#### Expected effect sizes and detection limits
With a 4B local model, expected per-task success rates are in the range 5–15%. For a two-condition comparison at \(n \approx 910\) tasks with baseline success \(p_0 = 0.10\):
- a **5 pp** absolute difference (0.10 → 0.15) is detectable at 80% power, \(\alpha = 0.05\) (two-sided), via a two-proportion z-test or McNemar's test on paired tasks.
- a **3 pp** difference (0.10 → 0.13) is borderline; it requires multi-seed pooling or per-site stratification to reach significance.
- differences below 2 pp are unlikely to be reliably detected and will be reported as directional trends only.

#### Primary statistical tests
| Comparison | Test | Rationale |
|---|---|---|
| Success rate between conditions (same tasks) | McNemar's exact test | Paired binary outcomes on identical task set |
| Success rate confidence intervals | Bootstrap 95% CI (10 000 resamples) | Non-parametric, no distributional assumption |
| Cost / latency between conditions | Wilcoxon signed-rank test | Paired continuous, non-normal (heavy right tail) |
| Per-site heterogeneity | Cochran–Mantel–Haenszel or stratified bootstrap | Controls for site as a blocking factor |

#### Contingency: low baseline success
If Phase 1 reveals that the 4B model achieves < 3% success across all four conditions, the following mitigations apply in order:
1. **Per-site reporting**: report sites separately; classifieds (~450 tasks) has the most power.
2. **Softer outcome metric**: supplement binary success with a graded progress score (e.g., fraction of sub-goals achieved or steps-before-failure), which has higher variance utilisation.
3. **Scope reduction**: restrict the routing study to the highest-performing site(s) only, and note the limitation.
4. **Stronger base model**: replace or supplement the 4B baseline with the API model (Qwen3-VL-Plus) as the primary agent, using the 4B model as the cheap tier in the router.

---

## 7) Baselines

### B0 — Strong upper bound
A strong API model or strong VLM baseline, to confirm solvability and upper-bound performance.

### B1 — Small-only baseline
Qwen3-VL-4B , fixed policy.

### B2 — Fixed best representation
Best non-routed representation configuration from Phase 1.

### B3 — Routed system
Rule-based representation router with explicit overhead accounting.

### B4 — Routed + one stabiliser
Only after B3 is understood.

---

## 8) Metrics and accounting

### Core outcome metrics
- Success / TSR
- Average steps
- P95 latency
- token usage / monetary cost (where applicable)
- energy / CO₂e
- retry count
- no-op / page-unchanged rate

### Overhead-aware accounting
\[
NetSaving = Cost_{baseline} - (Cost_{routed} + Cost_{router\_overhead})
\]

### Router overhead should include
- router decision logic,
- extra screenshot or resize work,
- extra DOM parsing,
- extra model calls for routing or staging,
- retries triggered by routing logic.

---

## 9) Logging requirements
The experiment runner should remain clean and reproducible.

### Required step-level fields
- run_id
- task_id
- seed
- step_idx
- SoM status
- observation mode
- router decision
- chosen representation action
- trigger reason
- module flags
- action type
- action success
- page_changed signal
- latency
- tokens / cost if available
- energy / CO₂e if available
- retry count
- error category
- artifact paths (screenshot / DOM / trace)

### Engineering rule
The visualisation layer must remain separate from the core experiment runner:
- runner writes JSONL and artifacts,
- analysis and visualisation scripts read from logs,
- no heavy demo coupling inside the main runner.

---

## 10) Benchmark policy

### Primary benchmark
**VisualWebArena / WebArena**

Reason:
- more stable,
- controlled,
- suitable for repeated comparable measurement,
- appropriate for step-level cost and energy analysis.

### Secondary benchmark
**Online Mind2Web (small subset only)**

Policy:
- use only as external validation,
- report blocked / anti-bot / geo-restricted tasks separately as benchmark noise.

### Avenir-Web position
- **VWA main study:** use as **reference + module library**, not as the main full-stack baseline.
- **Optional external validation:** can act as a strong open-source comparator on a small Online Mind2Web subset.

---

## 11) Immediate implementation priorities
1. Lock the primary variable hierarchy:
   - SoM at representation top-level,
   - observation mode,
   - router on/off.
2. Run a clean **4B baseline** first:
   - fixed config,
   - simple task subset,
   - repeatable logging.
3. Produce first-stage outputs:
   - representation screening results,
   - first Pareto plot,
   - first net-saving decomposition draft.
4. Only then introduce lightweight Aiden-inspired stabilisers.

---

## 12) Summary
This experimental framing keeps the project aligned with its strongest contribution:
- not maximum raw success,
- but a controlled and defensible study of whether routing improves the efficiency–success trade-off once representation choice, overhead, and benchmark conditions are treated carefully.
