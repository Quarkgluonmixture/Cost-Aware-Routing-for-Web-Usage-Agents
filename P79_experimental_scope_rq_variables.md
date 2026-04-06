# P79 Experimental Scope (Supervisor / Research Version, v3)

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
**Cost-aware routing for multimodal web agents on VisualWebArena, with observation mode as the primary representation variable and router overhead explicitly accounted for**

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
- compare dom / som / vision observation modes,
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
Zekun's recommendation is to formalise the router as an **abstract decision policy** rather than treating each routing trick separately.

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

Three observation modes are implemented as distinct representations:

| Mode | Observation content | Cost proxy |
|---|---|---|
| **dom** | Accessibility tree (AXTree) only — no screenshot | cheapest |
| **som** | AXTree with SoM mark index + screenshot with bounding boxes overlaid | richest |
| **vision** | Raw screenshot only — no AXTree, no element IDs | mid (no token-heavy AXTree) |

The router selects among these three at each step:
\[
a_t \in \mathcal{A} = \{\texttt{dom},\; \texttt{som},\; \texttt{vision}\}
\]

**Design rationale for 3 modes over a 2×2 grid:**
The original 2×2 design (SoM ON/OFF × DOM/Hybrid) yields four cells. Two of these collapse in practice:
- *DOM + SoM marks without screenshot* has no clear advantage over plain DOM;
- *DOM + plain screenshot without marks* adds image tokens without actionable element grounding.

The three implemented modes represent the three practically meaningful points on the representation spectrum.

**Router default policy (Phase 2):** default to `dom`; escalate to `som` or `vision` under failure or visual-signal triggers.

### 4.3 Policy
The router policy is:
\[
a_t = \pi(s_t)
\]

Within this framework, every experimental condition is a special case of the same policy family:
- **Fixed policy** \(\pi_{\text{fix}}\): always select the same observation mode (e.g., always `dom`). The three Phase 1 conditions are three such fixed policies.
- **Heuristic / threshold router** \(\pi_{\text{rule}}\): select observation mode based on page complexity, uncertainty, or progress signals derived from \(s_t\).
- **Fallback router** \(\pi_{\text{fall}}\): default to `dom` and escalate to `som` when failure signals exceed a threshold.

### Why this matters
This lets the dissertation compare all conditions — including fixed baselines — **within a common formal framework**, rather than treating routing as an unrelated engineering trick. Phase 1 tests where each fixed mode sits on the success–efficiency frontier, while Phase 2 tests whether routing improves the trade-off over the best fixed policy.

---

## 5) Variable hierarchy
This hierarchy is designed specifically to avoid **combinatorial explosion**.

### 5.1 Primary variables (systematically varied)

#### A. Representation variable

##### A1. Observation mode
Three levels, each defining a distinct information channel for the agent:

| Level | AXTree | Screenshot | SoM marks |
|---|---|---|---|
| **dom** | ✓ | ✗ | ✗ |
| **som** | ✓ (as SoM index) | ✓ | ✓ |
| **vision** | ✗ | ✓ | ✗ |

**Reason:** SoM is no longer a separate top-level factor — it is structurally embedded in the `som` mode. This avoids testing the degenerate cells (AXTree-only with marks, or screenshot-without-marks) and keeps the factor space interpretable.

#### B. Routing variable
##### B1. Router policy
- OFF (fixed observation mode)
- ON (rule-based routing)

**MVP rule-based router idea:**
- default to `dom` (cheapest),
- escalate to `som` if:
  - DOM size exceeds threshold,
  - action fails or selector not found,
  - page state does not change for N consecutive steps,
  - repeated no-progress steps detected,
  - task requires visual attribute matching (detected heuristically).
- escalate to `vision` for pages with minimal accessible DOM structure.

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

#### M_hist. History enrichment
Replace element IDs in the interaction history with element text labels.

**Rationale for deferral:** The baseline records `click [id=5528]` in history. Element IDs are reassigned on every page render and are opaque to the agent across steps; the agent cannot use them to reason about what it previously interacted with. Replacing them with text labels (e.g., `click "Sort by Price" dropdown`) would give the agent semantic context for multi-step intent maintenance. However, this changes the information available to the agent and is therefore not part of the baseline — it is a testable ablation module.

**Rule:** introduce these **one at a time**, only after the best primary observation mode is selected.

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
- choose the best observation mode before adding routing and modules.

#### Three fixed-policy conditions
| Condition | Observation mode | Router |
|---|---|---|
| B1-dom | dom | OFF |
| B1-som | som | OFF |
| B1-vision | vision | OFF |

All three share the same model (Qwen3-VL-4B), seed (42), and task set.

Output:
- success / TSR per condition
- average steps
- no-op rate
- P95 step latency
- token usage and basic cost / energy

Decision:
- keep the best one or two conditions only.

---

### Phase 2 — Routing study
Purpose:
- evaluate whether routing improves the trade-off relative to fixed policy.

#### Comparison
- best fixed observation policy from Phase 1 (B2)
- same setup + rule-based router (B3)

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
5. M_hist History enrichment (element text labels in history)

**Rule:** each module is tested against the same base condition; do not enable all modules together in the early stage. No interaction effects (e.g., M1+M3) are tested in the early stage; if time allows, at most one cumulative bundle is evaluated.

---

### Statistical design and power considerations

#### Sample sizes
VisualWebArena provides approximately 910 tasks across three standalone sites (shopping ~250, reddit ~230, classifieds ~450) plus cross-site Wikipedia references embedded in reddit/shopping configs. Each task is run once per condition per seed. With the Phase 1 three-condition design this yields ~910 episodes per condition.

#### Seed strategy
The primary analysis uses a single seed (seed = 42). If Phase 1 success rates are close between conditions (e.g., within 3 pp), a confirmation run with two additional seeds (123, 456) is triggered to assess stability. Multi-seed runs are not the default because each seed triplicates wall-clock time on a single-GPU setup.

#### Expected effect sizes and detection limits
With a 4B local model on classifieds, observed Phase 1 (dom-only) success rate is approximately **7–8%** (early estimate, 107 episodes). For a two-condition comparison at \(n \approx 450\) tasks (classifieds) with baseline \(p_0 = 0.08\):
- a **5 pp** absolute difference (0.08 → 0.13) is detectable at 80% power, \(\alpha = 0.05\) (two-sided), via McNemar's test on paired tasks.
- a **3 pp** difference is borderline and requires multi-seed pooling or per-site stratification.
- differences below 2 pp are reported as directional trends only.

#### Primary statistical tests
| Comparison | Test | Rationale |
|---|---|---|
| Success rate between conditions (same tasks) | McNemar's exact test | Paired binary outcomes on identical task set |
| Success rate confidence intervals | Bootstrap 95% CI (10 000 resamples) | Non-parametric, no distributional assumption |
| Cost / latency between conditions | Wilcoxon signed-rank test | Paired continuous, non-normal (heavy right tail) |
| Per-site heterogeneity | Cochran–Mantel–Haenszel or stratified bootstrap | Controls for site as a blocking factor |

#### Contingency: low baseline success
Phase 1 early data (dom-only on classifieds) shows ~7.5% success, which is **above the critical < 3% threshold**. The following mitigations remain on standby if full-run results fall significantly:
1. **Per-site reporting**: report sites separately; classifieds (~450 tasks) has the most power.
2. **Softer outcome metric**: supplement binary success with a graded progress score (e.g., fraction of sub-goals achieved or steps-before-failure), which has higher variance utilisation.
3. **Scope reduction**: restrict the routing study to the highest-performing site(s) only, and note the limitation.
4. **Stronger base model**: replace or supplement the 4B baseline with the API model (Qwen3-VL-Plus) as the primary agent, using the 4B model as the cheap tier in the router.

---

## 7) Baselines

### B0 — Strong upper bound
A strong API model or strong VLM baseline, to confirm solvability and upper-bound performance.

### B1 — Small-only baseline (current)
**Qwen3-VL-4B bf16**, fixed policy, single observation mode per condition.

Implementation notes:
- Model: Qwen/Qwen3-VL-4B-Instruct, bf16, eager attention (SDPA/Flash Attention not available on GB10/SM121 architecture without NVRTC compilation overhead)
- Inference: ~22.7 s/step mean on DGX Spark GB10 under shared GPU load; ~87% of step time is model inference
- History format: records `action_type` and `element_id`; element IDs are ephemeral (reassigned each render) and opaque across steps — this is the baseline design; text-label enrichment is deferred to M_hist
- Max steps per episode: 30
- Observation mode for current run: `dom` (Phase 1 first condition)
- Early success rate (classifieds, dom): ~7.5% (107 episodes)

### B2 — Fixed best representation
Best non-routed observation mode configuration from Phase 1.

### B3 — Routed system
Rule-based observation-mode router with explicit overhead accounting.

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
- observation mode
- SoM status (embedded in observation mode)
- router decision
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
- artifact paths (screenshot / DOM / SoM / trace)

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
1. Complete Phase 1 three-condition run (dom / som / vision) on all sites.
2. Produce representation screening results and first Pareto plot.
3. Select best fixed-policy condition(s) for Phase 2.
4. Run routing study (B3) and compute overhead-aware net saving.
5. Only then introduce Phase 3 modules (M1–M4, M_hist) one at a time.

---

## 12) Summary
This experimental framing keeps the project aligned with its strongest contribution:
- not maximum raw success,
- but a controlled and defensible study of whether routing improves the efficiency–success trade-off once representation choice, overhead, and benchmark conditions are treated carefully.

The observation mode variable (dom / som / vision) is the central axis of the study. SoM is embedded in the `som` mode rather than treated as a separate orthogonal factor, eliminating degenerate cells from the design. History enrichment (element text labels) is deferred to Phase 3 (M_hist) to preserve the integrity of the baseline.
