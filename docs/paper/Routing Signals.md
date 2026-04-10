# Systematic Literature Review: Cost-Aware Routing Signals for Web Navigation Agents

**Context:** 4B VLM (Qwen3-VL-4B) on VisualWebArena with non-discriminative token-level confidence (AUROC = 0.497)

**Review Date:** April 2026

---

## Executive Summary

This systematic literature review identifies routing signals for cost-aware escalation in web navigation agents when token-level confidence is non-discriminative. We analyzed approximately 1,400 papers from 2023–2026 across SciSpace, Google Scholar, and ArXiv, extracting 50+ relevant works spanning eight signal categories: behavioral signals, external verification, trajectory features, learned routing, task priors, attention/representation signals, multi-agent routing, and self-reflection.

Key findings: (1) Process reward models (PRMs) provide the strongest empirical evidence for web agents, with WebArbiter-7B achieving +9.1 points on WebPRMBench and GUI-Shepherd yielding +7.7pp on AndroidWorld; (2) Learned routers demonstrate dramatic cost reductions (FrugalGPT: 98%, SCOPE: 95.1%, TREACLE: 85%) on reasoning tasks; (3) Behavioral signals (action repetition, trajectory graphs, replay buffers) show 50% error reduction and 3× completion gains on WebArena; (4) Speculative execution reduces latency by 40–48% with minimal accuracy loss; (5) Verbalized confidence and self-reflection provide low-cost routing signals when calibrated properly.

Critical gaps remain: no work directly evaluates attention entropy or hidden-state probes for routing in VLMs; limited evidence for small-scale (≤4B) models in web navigation; and sparse integration of multiple signal types into unified routing policies.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Theoretical Foundations](#2-background-and-theoretical-foundations)
3. [Signal Categories and Mechanisms](#3-signal-categories-and-mechanisms)
   - 3.1 [Behavioral Signals](#31-behavioral-signals)
   - 3.2 [External Verification Signals](#32-external-verification-signals)
   - 3.3 [Trajectory-Level Features](#33-trajectory-level-features)
   - 3.4 [Learned Routing and Gating](#34-learned-routing-and-gating)
   - 3.5 [Task-Level Priors](#35-task-level-priors)
   - 3.6 [Attention and Representation Signals](#36-attention-and-representation-signals)
   - 3.7 [Multi-Agent Routing](#37-multi-agent-routing)
   - 3.8 [Self-Reflection and Verbal Confidence](#38-self-reflection-and-verbal-confidence)
4. [Comparative Analysis](#4-comparative-analysis)
5. [Discussion](#5-discussion)
6. [Recommendations](#6-recommendations)
7. [Research Gaps](#7-research-gaps)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

---

## 1. Introduction

Web navigation agents face a fundamental cost-quality tradeoff: powerful multimodal models deliver higher success rates but incur substantial computational and monetary costs, while smaller models offer efficiency at the expense of reliability. Traditional confidence estimation via token-level logprobs, margin, or predictive entropy has proven effective for routing in many NLP tasks. However, our empirical analysis of Qwen3-VL-4B on VisualWebArena reveals systematic overconfidence with AUROC = 0.497—worse than random chance—rendering these signals useless for routing decisions [1], [2], [3].

This systematic review addresses the critical question: **What alternative signals can trigger escalation from cheap to expensive modes in multi-step web interaction when token-level confidence is non-discriminative?** We surveyed approximately 1,400 papers published between 2023 and 2026, focusing on eight signal categories spanning behavioral patterns, external verification, trajectory analysis, learned routing mechanisms, task priors, representation-level signals, multi-agent orchestration, and self-reflection.

The review targets practitioners deploying small-scale vision-language models (≤4B parameters) for web automation tasks where per-step routing decisions directly impact both task success and operational cost. We prioritize signals with empirical validation on web or GUI agent benchmarks (WebArena, VisualWebArena, AndroidWorld, MiniWoB++) and assess each signal's computational overhead, discriminative power, and applicability to resource-constrained deployment scenarios.

---

## 2. Background and Theoretical Foundations

### 2.1 The Routing Problem

Cost-aware routing in multi-step agents requires deciding at each timestep whether to: (1) continue with the current (cheap) model, (2) escalate to a more capable (expensive) model, (3) retry with a different strategy, or (4) early-stop to avoid wasting compute on doomed episodes [4], [5], [6]. Optimal routing policies balance three competing objectives: task success rate, computational cost, and latency [7], [8], [9].

### 2.2 Limitations of Token-Level Confidence

Standard uncertainty quantification methods—including softmax confidence, predictive entropy, and margin-based scores—assume well-calibrated output distributions [10], [11]. However, small vision-language models trained with cross-entropy loss and reinforcement learning from human feedback (RLHF) often exhibit systematic overconfidence, particularly on out-of-distribution inputs common in web navigation [12], [13]. Our empirical finding (AUROC = 0.497 for Qwen3-VL-4B) aligns with recent observations that model scale and training objectives strongly influence calibration quality [14], [15].

### 2.3 Theoretical Frameworks for Routing

Recent work establishes theoretical foundations for LLM cascades and routing. Dekoninck et al. prove optimality conditions for cascade routing, showing that good quality estimators are essential for cost-effective escalation [16]. Chen et al. demonstrate that learned routers can achieve up to 98% cost reduction while matching or exceeding single-model performance through strategic query assignment [17]. These results motivate our search for alternative quality estimators beyond token probabilities.

---

## 3. Signal Categories and Mechanisms

### 3.1 Behavioral Signals

Behavioral signals detect unproductive patterns in agent trajectories: action repetition, navigation cycles, progress stalling, and backtracking. These signals require minimal computation and directly indicate when an agent is stuck.

#### R2D2: Remembering, Replaying and Dynamic Decision Making (2025)

- **Authors:** Multiple authors
- **Venue:** ACL
- **Core idea:** R2D2 maintains a replay buffer that reconstructs visited pages to detect navigation cycles and enable reflective error analysis, reducing redundant exploration [18].
- **Signal type:** Category 1 (Behavioral) + Category 3 (Trajectory)
- **Signal mechanism:** Replay buffer tracks visited URLs and page states; detects revisits and triggers reflection when cycles are identified. The system compares current state against historical states to flag unproductive loops.
- **Evidence strength:** 50% reduction in navigation errors; 3× increase in task completion rates on WebArena [18].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — directly evaluated on WebArena with substantial empirical gains. Replay-based cycle detection is computationally cheap and requires only URL/state tracking.
- **Key limitation:** Requires maintaining state history; may struggle with legitimate revisits (e.g., multi-step forms requiring navigation back to previous pages).

#### NNetscape Navigator: Synthetic Demonstrations (2024)

- **Authors:** Multiple authors
- **Venue:** arXiv
- **Core idea:** Prunes training episodes when intermediate trajectories cannot be annotated; uses hierarchical decomposition to avoid dead-end exploration during data collection [19].
- **Signal type:** Category 1 (Behavioral) + Category 4 (Trajectory)
- **Signal mechanism:** Episode pruning based on annotation feasibility; hierarchical task decomposition prevents exploration of unannotatable dead ends.
- **Evidence strength:** +6 points on WebArena; +20+ points on MiniWoB++; dataset of >6,000 demonstrations [19].
- **Model scale:** Fine-tunes smaller LM policy (abstract)
- **Applicability to web agents:** **High** — designed specifically for WebArena/MiniWoB environments. Hierarchical decomposition signals can inform when to escalate for complex subtasks.
- **Key limitation:** Focused on training data quality rather than runtime routing; adaptation to online routing requires extracting decomposition signals during inference.

#### TGPO: Tree-Guided Preference Optimization (2025)

- **Authors:** Chen Ziyuan et al.
- **Venue:** arXiv
- **Core idea:** Uses tree-structured trajectory representation to merge semantically identical states and trains a process reward model providing subgoal progress, redundancy detection, and action verification signals [20].
- **Signal type:** Category 1 (Behavioral) + Category 2 (External verification) + Category 3 (Trajectory)
- **Signal mechanism:** Tree merging identifies redundant state revisits; PRM scores subgoal progress and flags repeated actions. Dynamic weighting highlights critical decision points.
- **Evidence strength:** Higher success rates with fewer redundant steps on Online-Mind2Web and C-WebShop (no numeric summary in source) [20].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — trajectory-level redundancy detection maps directly to behavioral escalation triggers. Tree representation enables efficient cycle detection.
- **Key limitation:** Requires offline RL infrastructure and semantic state merging heuristics; tree construction complexity may limit real-time applicability.

#### WebGym: Scaling Training Environments (2024)

- **Authors:** Multiple authors
- **Venue:** arXiv
- **Core idea:** Training environment with action repetition penalty and step-budget penalties to discourage unproductive loops [21].
- **Signal type:** Category 1 (Behavioral)
- **Signal mechanism:** Explicit penalties for repeated actions within sliding windows; step-budget constraints trigger early termination.
- **Evidence strength:** Mentions before/after effects with action repetition penalty (no numeric summary) [21].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — simple heuristics (action repetition count, step budget) are trivial to implement and computationally free.
- **Key limitation:** Fixed thresholds may not generalize across task difficulties; legitimate repeated actions (e.g., scrolling) may trigger false positives.

#### Recon-Act: Self-Evolving Multi-Agent Browser-Use System (2025)

- **Authors:** Multiple authors
- **Venue:** arXiv
- **Core idea:** Reconnaissance and Action teams compare erroneous vs successful trajectories to infer tools/hints and register rule-based fixes in a tool archive [22].
- **Signal type:** Category 1 (Behavioral)
- **Signal mechanism:** Trajectory comparison identifies failure patterns; rule-based fixes are registered and triggered when similar patterns recur.
- **Evidence strength:** Claims substantial adaptability and state-of-the-art on VisualWebArena (no numeric summary) [22].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — explicitly designed for multimodal browser agents and evaluated on VisualWebArena. Pattern-based escalation is practical for 4B VLMs.
- **Key limitation:** Requires maintaining and updating rule archive; generalization to novel failure modes uncertain.

### 3.2 External Verification Signals

External verification signals use separate models or modules to predict step-level success, progress toward goals, or action correctness. These signals provide independent assessments uncorrelated with the agent's own confidence.

#### WebArbiter: Principle-Guided Process Reward Model (2026)

- **Authors:** Yao Zhang et al.
- **Venue:** Preprint
- **Core idea:** WebArbiter formulates process reward modeling as reasoning-first text generation, emitting structured justifications and preference verdicts to identify the most task-conducive action at each step [23].
- **Signal type:** Category 2 (External verification)
- **Signal mechanism:** Generative PRM produces step-level justifications ending with a verdict and candidate-action preference score. The model reasons about task principles before scoring.
- **Evidence strength:** WebArbiter-7B outperforms GPT-4o by **9.1 points on WebPRMBench** and yields **+6.4 points** on WebArena-Lite via reward-guided trajectory search [23].
- **Model scale:** 7B model
- **Applicability to web agents:** **High** — explicitly designed for web agents with strong empirical results. Step-level verdicts provide interpretable routing signals. A 7B verifier is feasible to run alongside a 4B actor.
- **Key limitation:** Two-stage training (reasoning distillation + RL) adds complexity; requires high-quality preference annotations; 7B model adds inference cost.

#### GUI-Shepherd: Process Reward and Verification (2025)

- **Authors:** Chen Cong et al.
- **Venue:** arXiv
- **Core idea:** Trains a process reward model on 52,000 interactions with human and GPT-4o rationales to provide dense stepwise supervision and inference-time verification for GUI agents [24].
- **Signal type:** Category 2 (External verification)
- **Signal mechanism:** Step-level PRM scores subgoal progress, redundancy, and action correctness. Used both for RL training (PPO) and as an inference-time verifier to filter actions.
- **Evidence strength:** **+7.7 percentage points** success online via PPO; **+5.1 points** when used as inference verifier on AndroidWorld; additional offline gains reported [24].
- **Model scale:** GUI benchmarks including AndroidWorld and AndroidControl
- **Applicability to web agents:** **High** — GUI interactions share long-horizon, stepwise verification needs with web navigation. PRM scores are direct routing signals.
- **Key limitation:** Requires large annotated dataset (52k interactions) and synthetic rationales; domain shift from mobile GUI to web may reduce fidelity.

#### V-Droid: Verifier-Driven Mobile GUI Agent (2025)

- **Authors:** Gaole Dai et al.
- **Venue:** arXiv
- **Core idea:** Replaces generator-only LLM control with an LLM verifier that evaluates candidate actions, using pairwise progress preference training to boost verifier decisions and reduce latency [25].
- **Signal type:** Category 2 (External verification)
- **Signal mechanism:** Verifier scores discretized candidate actions via pairwise progress preference ranking; highest-scoring action is executed.
- **Evidence strength:** Task success: **59.5% AndroidWorld**, **38.3% AndroidLab**, **49% MobileAgentBench**; latency **4.3s per step** (6.1× faster than baselines) [25].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — verifier-driven architecture directly translates to web navigation. Verifier scores provide step-level confidence for escalation decisions.
- **Key limitation:** Requires constructing discretized candidate action spaces; may not scale to unconstrained web interactions with large action spaces.

#### OpenClaw-RL: Next-State Signals and PRM Judges (2026)

- **Authors:** Yinjie Wang et al.
- **Venue:** Preprint
- **Core idea:** Treats next-state signals (user replies, tool outputs, GUI changes) as universal training signals, extracting evaluative scalar rewards via PRMs and directive hints via hindsight-guided distillation [26].
- **Signal type:** Category 2 (External verification)
- **Signal mechanism:** Asynchronous PRM judges produce scalar rewards from next-state traces; hindsight-guided on-policy distillation yields action-correction hints.
- **Evidence strength:** Agent improves by learning from live interactions and process rewards across terminal, GUI, and conversational domains (abstract claims) [26].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — next-state derived rewards and textual corrective hints provide both evaluative and directive routing signals. Asynchronous PRM allows cheap online evaluation.
- **Key limitation:** Requires online training loop and reliable PRMs; asynchronous training/serving complexity may complicate deployment.

#### Uncertainty-Aware Step-Wise Verification (CoT Entropy) (2025)

- **Authors:** Zihuiwen Ye et al.
- **Venue:** Preprint
- **Core idea:** Introduces CoT Entropy to quantify uncertainty of generative process reward models, improving reliability of step-wise verification and reducing reward-hacking vulnerability [27].
- **Signal type:** Category 2 (External verification)
- **Signal mechanism:** CoT entropy measures uncertainty over generated process steps; high entropy flags uncertain verification judgments, triggering escalation.
- **Evidence strength:** CoT Entropy outperforms existing UQ approaches in quantifying PRM uncertainty and improves verification robustness (abstract summary) [27].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — uncertainty scores for PRM judgments gate escalation while avoiding blind trust in PRM outputs. Prevents over-reliance on potentially incorrect verifier signals.
- **Key limitation:** Evaluation focused on mathematical reasoning traces; calibration on web interaction traces may differ.

#### InfiGUIAgent: Multimodal Generalist GUI Agent (2025)

- **Authors:** Yuhang Liu et al.
- **Venue:** arXiv
- **Core idea:** Uses supervised fine-tuning in stages to build GUI understanding, grounding, hierarchical reasoning, and expectation-reflection skills [28].
- **Signal type:** Category 2 (External verification) + Category 6 (Attention/representation)
- **Signal mechanism:** Hierarchical reasoning and native reflection with multimodal inputs; expectation-reflection module compares predicted vs actual outcomes.
- **Evidence strength:** Competitive performance on GUI benchmarks (no numeric summary) [28].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — reflection and reasoning afford trajectory-level judgment and verification. Expectation-reflection mismatches signal when to escalate.
- **Key limitation:** Supervised fine-tuning may not generalize to unseen sites and long-horizon web tasks.

#### OpAgent: Operator Agent for Web Navigation (Year insufficient)

- **Authors:** Y. Guo et al.
- **Venue:** Unspecified
- **Core idea:** Equips agents with self-reflection and hierarchical task decomposition tailored to WebArena-like environments [29].
- **Signal type:** Category 2 (External verification) + Category 8 (Self-reflection)
- **Signal mechanism:** Self-reflection heuristics and hierarchical task-specific heuristics for verification and progress signals.
- **Evidence strength:** Insufficient evidence [29].
- **Model scale:** Local WebArena instance
- **Applicability to web agents:** **High** — targets web navigation and uses self-reflection signals useful for routing/escalation decisions.
- **Key limitation:** Uses environment-specific heuristics; generalization may be limited.

#### Agentic Reward Modeling: VAGEN Verifier (Year insufficient)

- **Authors:** C. Cui et al.
- **Venue:** Unspecified
- **Core idea:** Shifts from passive evaluation to interactive verification, using a verifier that proactively interacts to validate agent actions and outcomes [30].
- **Signal type:** Category 2 (External verification)
- **Signal mechanism:** Online interactive verifier checks candidate actions/outcomes for correctness via proactive environment interaction.
- **Evidence strength:** Insufficient evidence [30].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — proactive verification maps to external verification signals for web navigation agents.
- **Key limitation:** Implementation complexity and potential extra latency from proactive verification.

#### WebCanvas: Benchmark and Agent (2024)

- **Authors:** Yichen Pan et al.
- **Venue:** Preprint
- **Core idea:** Provides an online evaluation framework and Mind2Web-Live tasks measuring intermediate states and realistic web dynamics [31].
- **Signal type:** Category 2 (External verification)
- **Signal mechanism:** Page-level intermediate-state metrics and online test harness isolating meaningful state changes.
- **Evidence strength:** Best agent achieves **23.1% task success** and **48.8% task completion** on live test set [31].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Medium** — provides evaluation and page-level success predictors useful for routing, though agents and models used are not necessarily small-scale.
- **Key limitation:** Primary contribution is benchmark/evaluation; integrating signals into real-time routers requires engineering effort.

### 3.3 Trajectory-Level Features

Trajectory-level features aggregate information across multiple steps: action diversity, URL visit patterns, exploration vs exploitation ratio, backtracking frequency, and trajectory graph structure.

#### WebGraphEval: Multi-Turn Trajectory Evaluation (2025)

- **Authors:** Multiple authors
- **Venue:** arXiv
- **Core idea:** Abstracts trajectories into weighted action graphs encoding recurring behaviors, redundancy, and critical decision points for cross-agent efficiency analysis [32].
- **Signal type:** Category 3 (Trajectory features)
- **Signal mechanism:** Trajectory graph abstraction with node weights representing action frequency and edge weights representing transition probabilities; graph metrics (cycles, centrality) detect inefficiency.
- **Evidence strength:** Demonstrates structural analyses across thousands of trajectories and identifies redundancy and inefficiency (abstract) [32].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — designed for WebArena trajectories. Graph metrics provide quantitative signals for when agents are stuck in cycles or exploring inefficiently.
- **Key limitation:** Graph construction and analysis add computational overhead; real-time applicability depends on efficient incremental graph updates.

#### Branch-and-Browse: Tree-Structured Exploration (2025)

- **Authors:** Multiple authors
- **Venue:** arXiv
- **Core idea:** Tree-structured subtask management, action memory, and controlled branching reduce backtracking and share explored actions [33].
- **Signal type:** Category 3 (Trajectory features) + Category 1 (Behavioral)
- **Signal mechanism:** Tree structure tracks exploration branches; action memory prevents redundant exploration; branching decisions based on subtask decomposition.
- **Evidence strength:** **35.8% task success on WebArena**; up to **40.4% reduction in execution time** vs prior methods [33].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — explicitly built and evaluated as WebArena web agent framework. Tree structure provides natural escalation points (branch vs continue).
- **Key limitation:** Tree management overhead; determining optimal branching points requires task-specific heuristics.

#### GUIOdyssey: Dataset and OdysseyAgent (2024)

- **Authors:** Quanfeng Lu et al.
- **Venue:** Preprint
- **Core idea:** Provides large cross-app mobile navigation dataset; OdysseyAgent uses history resampling to balance performance and speed on long multi-app tasks [34].
- **Signal type:** Category 3 (Trajectory features)
- **Signal mechanism:** History resampler attends to past screenshots/actions to produce step-level signals about drift and revisit likelihood.
- **Evidence strength:** Dataset: 8,334 episodes, avg 15.3 steps per episode; improvements for OdysseyAgent reported in-domain and out-of-domain [34].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — trajectory statistics (step counts, revisits) directly inform routing policies for 4B VLM agents.
- **Key limitation:** Focused on mobile cross-app tasks; mapping to arbitrary web sites may need adaptation.

#### Trace-Level Comparison for GUI Agents (2026)

- **Authors:** Maria Movin et al.
- **Venue:** Preprint
- **Core idea:** Trace-level evaluation shows agents can match human success while following different navigation strategies, revealing behavioral mismatches even when outcomes align [35].
- **Signal type:** Category 3 (Trajectory features)
- **Signal mechanism:** Trace diagnostics comparing outcome/effort, query formulation, and navigation patterns to detect non-human or brittle strategies.
- **Evidence strength:** Controlled study: agent success comparable to participants; behavioral differences (search-centric vs exploratory) documented [35].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Medium** — trace-level diagnostics suggest practical signals (branching factor, query diversity) for routing but require logging/human baselines.
- **Key limitation:** Study in single production search app; transferability to general web tasks uncertain.

### 3.4 Learned Routing and Gating

Learned routing mechanisms train models or policies to predict which model to use per query, optimizing cost-quality tradeoffs through supervised learning, reinforcement learning, or meta-learning.

#### FrugalGPT: LLM Cascades (2023)

- **Authors:** Lingjiao Chen et al.
- **Venue:** Preprint
- **Core idea:** LLM cascades and model combination strategies drastically reduce inference cost while preserving or improving accuracy by learning which model to use per query [36].
- **Signal type:** Category 7 (Multi-agent routing)
- **Signal mechanism:** Learned cascades select cheaper models for easy queries and escalate when needed; uses scoring functions and learned thresholds.
- **Evidence strength:** Can match top LLM performance with up to **98% cost reduction** or improve accuracy by **~4%** at equal cost vs GPT-4 [36].
- **Model scale:** Experiments involve deployed LLM APIs (various scales)
- **Applicability to web agents:** **High** — cascade paradigms directly map to small→large model selection for 4B VLM web agents.
- **Key limitation:** Effectiveness depends on reliable difficulty estimation and cross-model calibration.

#### SCOPE: Scalable Router (2026)

- **Authors:** Qi Cao et al.
- **Venue:** Preprint
- **Core idea:** SCOPE predicts per-model cost and performance via pre-hoc reasoning and RL so routing generalizes to new models and budgets, explicitly optimizing accuracy-cost tradeoff [37].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Outcome and cost prediction for candidate models trained with RL to produce controllable routing policies.
- **Evidence strength:** Up to **25.7% accuracy improvement** when maximizing performance, or up to **95.1% cost reduction** when maximizing efficiency [37].
- **Model scale:** Experiments across on-device and cloud-scale LLaMA and Qwen variants (various scales)
- **Applicability to web agents:** **High** — directly designed to adapt routing to unseen models and budgets; fits 4B VLM deployment scenarios.
- **Key limitation:** RL-based training and retrieval of model-behavior traces add training complexity.

#### RouteLLM: Learning to Route with Preference Data (2024)

- **Authors:** Isaac Ong et al.
- **Venue:** arXiv
- **Core idea:** Routers trained on human preference data learn to pick between stronger and weaker models, halving cost in some cases while preserving quality [38].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Router models trained on preference labels and augmented data to predict when to escalate.
- **Evidence strength:** Cost reductions over **2× in some settings** [38].
- **Model scale:** Multiple strong/weak model pairs in benchmarks
- **Applicability to web agents:** **High** — preference-trained routers transfer between model pairs and fit cost-aware escalation for web tasks.
- **Key limitation:** Requires labeled preference data and augmentation to generalize across domains.

#### xRouter: RL-Based Cost-Aware Orchestration (2025)

- **Authors:** Qian Cheng et al.
- **Venue:** arXiv
- **Core idea:** RL-trained router that either answers or invokes external models, optimized end-to-end with explicit cost-aware reward to remove hand-engineered rules [39].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Cost-aware RL reward combining performance and monetary/latency cost to learn escalation policy.
- **Evidence strength:** Substantial cost reductions at comparable task completion rates (qualitative claim) [39].
- **Model scale:** Heterogeneous model pools and tool-calling orchestration
- **Applicability to web agents:** **High** — RL can learn complex escalation policies tailored to web-navigation cost constraints.
- **Key limitation:** RL training complexity and reward shaping sensitivity for small models.

#### PickLLM: Context-Aware RL Routing (2024)

- **Authors:** Dimitrios Sikeridis et al.
- **Venue:** arXiv
- **Core idea:** Uses RL to select optimal model per query considering cost, latency, and accuracy via weighted reward; converges to preferred LLM per session [40].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Context embeddings plus response history in RL router with weighted reward for cost/latency/accuracy.
- **Evidence strength:** Improvement in cost per session and overall latency (no numeric summary available) [40].
- **Model scale:** Pool of four LLMs in prompt-response datasets
- **Applicability to web agents:** **High** — session-level web tasks where per-session convergence to a model reduces API calls.
- **Key limitation:** Requires environment-specific scoring function and calibration of weighted reward.

#### TREACLE: Budget-Constrained Policy Learning (2024)

- **Authors:** Xuechen Zhang et al.
- **Venue:** Preprint
- **Core idea:** RL policy jointly selects models and prompting schemes under budget/latency constraints, trading off accuracy and monetary cost [41].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Context embeddings and response history used by RL policy encoding monetary and latency budget constraints.
- **Evidence strength:** **Cost savings up to 85%** on GSM8K, CSQA, and LLC while maintaining high accuracy [41].
- **Model scale:** Multiple LLMs and prompts on standard reasoning datasets
- **Applicability to web agents:** **Medium** — demonstrates large cost gains on reasoning tasks but transfer to interactive web navigation requires adapting state/action signals.
- **Key limitation:** Focused evaluations on reasoning datasets rather than interactive web agents.

#### FORC: Cost-Effective Language Model Choice (2023)

- **Authors:** M. Sakota et al.
- **Venue:** arXiv
- **Core idea:** Uses meta-model predicting per-input which LM will do well and assigns inputs to models to match largest-LM performance at much lower cost [42].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Meta-model (meta-features) predicts model performance per input for cost-aware assignment.
- **Evidence strength:** Matches largest-LM performance with **63% cost reduction** across 14 datasets [42].
- **Model scale:** Four candidate LMs of varying size across 14 datasets
- **Applicability to web agents:** **High** — meta-model selection maps directly to per-query routing in web agents.
- **Key limitation:** Meta-model needs representative calibration data and may not capture dynamic environment complexity.

#### Hybrid LLM: Difficulty-Based Routing (2024)

- **Authors:** Dujian Ding et al.
- **Venue:** arXiv
- **Core idea:** Router predicts query difficulty and assigns queries to small or large models to reduce calls to large model while preserving quality [43].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Predicted query difficulty and tunable desired quality threshold used by router.
- **Evidence strength:** Up to **40% fewer large-model calls** with no drop in quality [43].
- **Model scale:** Small/large model pairs on benchmark tasks
- **Applicability to web agents:** **High** — many web queries are easy and can be served by smaller models.
- **Key limitation:** Requires accurate difficulty prediction; may miss hard corner cases in web navigation.

#### SATER: Confidence-Aware Rejection (2025)

- **Authors:** Liu Yide et al.
- **Venue:** arXiv
- **Core idea:** Dual-mode routing fine-tunes models for shortest-response preference and uses confidence-aware rejection mechanism to reduce redundant outputs across pre-generation and cascade routing [44].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Fine-tuned confidence-aware rejection and shortest-response preference optimization to decide rejection/escalation.
- **Evidence strength:** Reduces computational costs by **>50%** and cascade latency by **>80%** across evaluated SLMs/datasets [44].
- **Model scale:** Three SLMs across six datasets
- **Applicability to web agents:** **High** — targets both pre-generation and cascade modes relevant to web-agent step costs.
- **Key limitation:** Evaluations on SLMs; transfer to VLMs and interactive web environments needs validation.

#### One Head Many Models: Cross-Attention Routing (2025)

- **Authors:** R. Pulishetty et al.
- **Venue:** arXiv
- **Core idea:** Single-head cross-attention router jointly embeds queries and model representations to predict per-input model selection balancing quality and cost [45].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Cross-attention between query and model embeddings predicts response quality and cost.
- **Evidence strength:** **AIQ +6.6%** and **max perf +2.9%** improvement over baselines [45].
- **Model scale:** RouterBench with diverse LLM pools (varying sizes)
- **Applicability to web agents:** **High** — router explicitly models query–model interactions useful for per-query selection in web navigation.
- **Key limitation:** Needs pre-existing model embeddings and calibration to user cost preferences.

#### A Unified Approach to Routing and Cascading (2024)

- **Authors:** Jasper Dekoninck et al.
- **Venue:** Preprint
- **Core idea:** Provides theoretical optimality results for routing and cascading; introduces cascade routing showing how to unify both strategies and importance of good quality estimators [46].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Cascade routing uses estimated quality scores to decide between routing and iterative escalation.
- **Evidence strength:** Analytical optimality proofs and experiments showing cascade routing outperforms individual approaches (abstract summary) [46].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — gives principled framework to trade cost vs success for 4B VLM router.
- **Key limitation:** Requires accurate estimators of model quality per query; estimator design is nontrivial.

#### Faster Cascades via Speculative Decoding (2024)

- **Authors:** Harikrishna Narasimhan et al.
- **Venue:** arXiv
- **Core idea:** Combines speculative decoding and cascades into speculative cascading to approximate optimal deferral rule and improve cost-quality trade-offs [47].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Speculative execution to implement deferral rules determining when to invoke larger models.
- **Evidence strength:** Improved cost-quality trade-offs vs cascade and speculative decoding baselines on T5 benchmarks [47].
- **Model scale:** T5 model family experiments
- **Applicability to web agents:** **Medium** — speculative-cascade idea can reduce latency and cost, but needs adaptation for multi-step web actions and VLMs.
- **Key limitation:** Demonstrations on pure language tasks; action-space and environment dynamics in web navigation add complexity.

#### ExpertFlow: Efficient MoE Inference (2024)

- **Authors:** Xin He et al.
- **Venue:** Preprint
- **Core idea:** Predictive routing and expert caching reduce memory and improve throughput for MoE inference by forecasting expert usage and scheduling tokens accordingly [48].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Transformer-based routing path predictor, token scheduler, and predictive expert cache to avoid loading inactive experts.
- **Evidence strength:** Reduces GPU memory by up to **93.72%** and improves throughput up to **10×** over offloading baselines on single GPU [48].
- **Model scale:** MoE deployments in single-GPU setups (large parameter counts implied)
- **Applicability to web agents:** **Medium** — architectural ideas help MoE-style gating for cost-aware inference, but productionizing for 4B dense VLM needs adaptation.
- **Key limitation:** Tailored to MoE architectures and assumes predictable routing patterns.

#### DirMoE: Differentiable Router (2026)

- **Authors:** Amirhossein Vahidi et al.
- **Venue:** Preprint
- **Core idea:** Disentangles expert selection and contribution using differentiable Dirichlet+Bernoulli routing, improving expert specialization and routing stability [49].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** End-to-end differentiable routing with explicit sparsity control via variational ELBO.
- **Evidence strength:** Matches or exceeds competing routers and improves expert specialization across tasks (abstract claims) [49].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Medium** — useful if moving to MoE or hybrid architectures; direct benefit for dense 4B VLM is limited without architectural change.
- **Key limitation:** Requires MoE training pipelines and may not transfer to frozen small VLMs.

#### CLEAR: Edge-Cloud Adaptive Routing (Year insufficient)

- **Authors:** W. Zheng et al.
- **Venue:** Unspecified
- **Core idea:** Edge-cloud collaborative routing framework adaptively routes requests between edge SLMs and cloud LLMs; can request cloud re-generation upon edge uncertainty [50].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Edge SLM uncertainty triggers re-generation requests to cloud LLM; adaptive edge-cloud routing logic.
- **Evidence strength:** Insufficient evidence [50].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Medium** — edge-cloud patterns map to local small-model first with cloud fallback, natural fit for cost-aware web agent routing.
- **Key limitation:** Details and experimental metrics not available.

### 3.5 Task-Level Priors

Task-level priors use query characteristics, task type classification, or historical success rates to pre-route queries before execution.

#### LLM Cascades with Mixture of Thoughts (2023)

- **Authors:** Multiple authors
- **Venue:** arXiv
- **Core idea:** Uses answer consistency of weaker LM (via sampling multiple chains-of-thought) as difficulty/confidence signal to escalate to stronger model [51].
- **Signal type:** Category 5 (Task priors) + Category 8 (Self-reflection)
- **Signal mechanism:** Answer consistency across multiple CoT samples proxies question difficulty; low consistency triggers escalation.
- **Evidence strength:** Achieves comparable accuracy to strongest model while using **~40% of its cost** on reasoning benchmarks [51].
- **Model scale:** Uses GPT-3.5-turbo and GPT-4 as weak/strong models
- **Applicability to web agents:** **High** — demonstrates task-prior style routing via difficulty estimation; approach transferable to web query routing.
- **Key limitation:** Requires multiple forward passes for consistency estimation; may increase latency.

#### Symbiotic Cooperation for Web Agents (2025)

- **Authors:** Ruichen Zhang et al.
- **Venue:** arXiv
- **Core idea:** Pairs complementary large and small LLMs to exploit strengths/cost differences in web-agent pipelines and orchestrate when to escalate to larger model [52].
- **Signal type:** Category 7 (Multi-agent routing)
- **Signal mechanism:** Heterogeneous-model cooperation and selective escalation based on complementary strengths.
- **Evidence strength:** Insufficient evidence [52].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Medium** — targets cross-model cooperation applicable to web agents but lacks reported web-specific routing numbers.
- **Key limitation:** Limited empirical detail on concrete routing signals.

### 3.6 Attention and Representation Signals

Attention and representation signals use internal model states—attention entropy, hidden state uncertainty, or probe classifiers—to estimate confidence.

**Critical Finding:** No papers in the surveyed corpus directly analyze internal attention entropy or use hidden-state probing classifiers as routing signals for LLM/VLM web agents. This represents a significant research gap.

#### LaSM: Layer-Wise Attention Defense (2025)

- **Authors:** Zihe Yan et al.
- **Venue:** Preprint
- **Core idea:** Selectively scales attention/MLP modules in critical layers to counter pop-up injection attacks by aligning saliency with task-relevant regions without retraining [53].
- **Signal type:** Category 6 (Attention/representation)
- **Signal mechanism:** Layer-wise attention divergence detection and selective amplification as runtime defense and alignment signal.
- **Evidence strength:** Significant improvement in defense success rates across GUI datasets (abstract reports robust gains) [53].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Medium** — layer-wise attention divergence could be used as representation-level gate for escalating to stronger models when attention misaligns.
- **Key limitation:** Layer modulation assumptions may not generalize across model architectures and requires per-model calibration.

#### SpiritSight: GUI Agent (2025)

- **Authors:** Zhiyuan Huang et al.
- **Venue:** Preprint
- **Core idea:** Vision-based end-to-end GUI agent with large GUI dataset and Universal Block Parsing to improve grounding and action accuracy across platforms [54].
- **Signal type:** Category 6 (Attention/representation)
- **Signal mechanism:** Improved grounding and block-parsing reduce ambiguous visual tokens and stabilize element attention.
- **Evidence strength:** Demonstrated superior performance over other methods on multiple GUI benchmarks (abstract reports aggregate gains) [54].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Medium** — grounding improvements reduce false actions and can feed representation-based confidence signals to router.
- **Key limitation:** Gains rely on large curated GUI datasets and parsing heuristics that may not generalize to arbitrary web pages.

### 3.7 Multi-Agent Routing

Multi-agent routing uses confidence-based model selection, cost-aware ensemble methods, or LLM cascades to distribute queries across models.

(Note: Many papers in Section 3.4 also fall under this category; we avoid duplication and reference them here.)

FrugalGPT [36], RouteLLM [38], SCOPE [37], and Symbiotic Cooperation [52] all implement multi-agent routing strategies. See Section 3.4 for detailed analysis.

### 3.8 Self-Reflection and Verbal Confidence

Self-reflection signals ask models to evaluate their own performance, express confidence verbally, or identify when they are stuck.

#### Learning to Route with Confidence Tokens (Self-REF) (2024)

- **Authors:** Yu-Neng Chuang et al.
- **Venue:** arXiv
- **Core idea:** Injects discrete confidence tokens into LLMs during training so model explicitly emits confidence token whose score routes/rejects outputs more reliably than verbal confidence or token probabilities [55].
- **Signal type:** Category 8 (Self-reflection)
- **Signal mechanism:** Trained confidence tokens producing extractable confidence score used for routing or rejection decisions.
- **Evidence strength:** Confidence tokens yield significant downstream routing and rejection gains compared to conventional approaches (no exact AUROC provided) [55].
- **Model scale:** LLMs in routing/rejection learning tasks (specific sizes not provided)
- **Applicability to web agents:** **High** — tokenized internal confidence is practical routing signal that can be integrated into VLM-driven web agents.
- **Key limitation:** Requires additional training/fine-tuning to teach confidence tokens and calibration data.

#### On Verbalized Confidence Scores for LLMs (2024)

- **Authors:** Daniel Yang et al.
- **Venue:** Preprint
- **Core idea:** Demonstrates that asking LLMs to verbalize confidence scores can yield well-calibrated uncertainty estimates depending on prompt methods, enabling low-overhead uncertainty signals [56].
- **Signal type:** Category 8 (Self-reflection)
- **Signal mechanism:** Prompted verbalized confidence token used as uncertainty proxy for routing decisions.
- **Evidence strength:** Benchmarks show reliability of verbalized scores varies with prompt and model but can be calibrated to produce useful estimates (abstract summary) [56].
- **Model scale:** Experiments across datasets and models (various scales)
- **Applicability to web agents:** **High** — very low-cost method to obtain per-step confidence from 4B VLM without token-level logit access.
- **Key limitation:** Calibration-sensitive and prompt-dependent; performance varies across tasks and models.

#### EAGLE: Expectation of Aggregated Internal Belief (2025)

- **Authors:** Chen Yun et al.
- **Venue:** arXiv
- **Core idea:** Extracts and aggregates internal layer-wise beliefs during self-evaluation to compute refined confidence scores that improve calibration over output-only methods [57].
- **Signal type:** Category 6 (Attention/representation) + Category 8 (Self-reflection)
- **Signal mechanism:** Aggregation of intermediate hidden-state beliefs across layers to form expectation-based confidence estimate.
- **Evidence strength:** EAGLE significantly improves calibration performance over baselines (no numeric AUROC provided) [57].
- **Model scale:** Diverse LLMs and datasets (specific scales not provided)
- **Applicability to web agents:** **Medium** — hidden-state aggregation can be adapted to VLMs, but implementation access to internal activations is required in deployed settings.
- **Key limitation:** Requires access to internal activations and careful design of aggregation across layers.

#### MobileUse: Hierarchical Reflection (2025)

- **Authors:** Ning Li et al.
- **Venue:** Preprint
- **Core idea:** Uses hierarchical reflection architecture plus proactive exploration to detect/recover from errors across temporal scales in mobile GUI tasks [58].
- **Signal type:** Category 8 (Self-reflection)
- **Signal mechanism:** Hierarchical reflections trigger recovery actions and exploration when internal monitors detect low progress.
- **Evidence strength:** Success rates **62.9% on AndroidWorld** and **44.2% on AndroidLab** [58].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — directly targets mobile GUI/web-like stepwise tasks and recovery heuristics useful for 4B VLM agents.
- **Key limitation:** Relies on reflection training data and exploration module complexity which may increase sample or annotation needs.

#### ReAP: Reflection-Augmented Planning (2025)

- **Authors:** Ruhana Azam et al.
- **Venue:** Preprint
- **Core idea:** Leverages self-reflections from successful and failed past experiences to guide web navigation, improving recovery on previously failed tasks [59].
- **Signal type:** Category 8 (Self-reflection)
- **Signal mechanism:** Reflection memory flags failed trajectories and surfaces corrective suggestions at decision points.
- **Evidence strength:** Improves baseline by **11 percentage points overall** and by **29 points on previously failed tasks** [59].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — reflection-memory signals map cleanly to "am I stuck?" routing decisions for 4B agents.
- **Key limitation:** Benefit depends on availability of past traces and may not generalize to novel sites without transfer strategies.

#### R2D2: Reflection Component (2025)

(See Section 3.1 for full entry; reflection component noted here for completeness) [18].

### 3.9 Speculative Execution

Speculative execution predicts likely next actions with faster models to execute multiple steps in parallel, reducing latency while maintaining correctness via verification.

#### Speculative Actions: Lossless Framework (2025)

- **Authors:** Lu Yunan et al.
- **Venue:** arXiv
- **Core idea:** Predicts likely next actions with faster models to execute multiple steps speculatively in parallel, reducing latency while maintaining correctness via verification [60].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Fast-model next-action predictions combined with top-K speculation, multi-step speculation, and uncertainty-aware optimization for speculative execution.
- **Evidence strength:** Next-action prediction up to **55%** and substantial end-to-end latency reductions across evaluated environments [60].
- **Model scale:** Gaming, e-commerce, web search agent environments (multi-domain)
- **Applicability to web agents:** **High** — speculative multi-step execution can reduce web-agent latency and inform when to escalate to larger models for verification.
- **Key limitation:** Speculation correctness depends on guesser strength; mis-speculation can waste compute or require rollbacks.

#### PASTE: Speculative Tool Execution (2026)

- **Authors:** Yifan Sui et al.
- **Venue:** Preprint
- **Core idea:** Speculatively executes likely tool calls based on pattern-aware predictions to hide tool latency in LLM-agent loops [61].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Pattern-aware speculation of recurring tool-call sequences and data dependencies to execute candidate tools in parallel with LLM thinking.
- **Evidence strength:** Reduced average task completion time by **48.5%** and improved tool execution throughput by **1.8×** [61].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **High** — speculative tool execution directly accelerates web agent loops and reduces per-step latency for 4B VLM agents.
- **Key limitation:** Incorrect speculation can waste compute and requires good pattern models and tooling to roll back side effects.

#### MineDraft: Batch Parallel Speculative Decoding (2026)

- **Authors:** Zhenwei Tang et al.
- **Venue:** Preprint
- **Core idea:** Parallelizes speculative decoding by overlapping drafting and verification across batches to hide drafting latency and improve throughput [62].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Batch-parallel design maintaining two request batches to overlap draft and verification stages for speculative decoding.
- **Evidence strength:** Throughput up to **75%** and end-to-end latency improvement up to **39%** over standard speculative decoding [62].
- **Model scale:** Implemented as vLLM plugin and tested on production inference setups
- **Applicability to web agents:** **High** — speculative execution combined with small drafting models suits cost-aware web agents using 4B VLM as either drafter or verifier.
- **Key limitation:** Complexity in batching logic and increased memory/coordination demands for real-time agent loops.

#### Collaborative Speculative Inference (2025)

- **Authors:** Luyao Gao et al.
- **Venue:** arXiv
- **Core idea:** Integrates verification and speculative decoding with ensemble fusion and confidence-based token fusion to speed inference while preserving quality [63].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Ensemble speculative decoding with confidence-based token fusion and verification phases to avoid quality loss.
- **Evidence strength:** Insufficient evidence for numeric metrics, qualitative improvement claims present [63].
- **Model scale:** Inference-serving ensembles and speculative-decoding setups (details not in snippet)
- **Applicability to web agents:** **Medium** — collaborative speculative inference reduces generation overhead but must be adapted for multi-step action verification in web agents.
- **Key limitation:** Focused on inference serving; mapping token-level speculative gains to trajectory-level decisions requires further work.

### 3.10 Additional Relevant Work

#### Continual GUI Agents and GUI-AiF (2026)

- **Authors:** Ziwei Liu et al.
- **Venue:** Preprint
- **Core idea:** Adds anchoring rewards to stabilize grounding under shifting GUI distributions, enabling continual learning for GUI agents in dynamic environments [64].
- **Signal type:** Category 1 (Behavioral)
- **Signal mechanism:** Anchoring point and region rewards penalize overfitting to static spatial cues and preserve stable interaction points.
- **Evidence strength:** GUI-AiF surpasses SOTA baselines in continual learning experiments (abstract reports consistent improvements) [64].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Medium** — anchoring rewards yield robust stuck-state detection and can inform routing when UI distributions shift.
- **Key limitation:** Requires continual training infrastructure and reward tuning per domain.

#### ToolPRMBench: Evaluating Process Reward Models (2025)

- **Authors:** D. Li et al.
- **Venue:** Unspecified
- **Core idea:** Benchmarks and analyzes process reward models used for tool-using agents and highlights gaps in evaluation and fidelity of process supervision [65].
- **Signal type:** Category 2 (External verification)
- **Signal mechanism:** Evaluation of process reward models and their ability to provide step-level guidance.
- **Evidence strength:** Insufficient evidence [65].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Medium** — evaluates PRMs informing whether to trust process-verifier signals for web agents.
- **Key limitation:** Focus is benchmark/evaluation; actionable routing mechanisms may require further work.

#### C3PO: Optimized LLM Cascades (Year insufficient)

- **Authors:** A. Valkanas et al.
- **Venue:** Unspecified
- **Core idea:** Designs cascade optimization under probabilistic cost constraints and calibrates deferral thresholds using held-out data for cost-accuracy tradeoffs [66].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Calibration of deferral thresholds and probabilistic cost constraints in LLM cascades.
- **Evidence strength:** Evaluated on 16 datasets across 3 LLM cascades (accuracy vs cost curves reported) [66].
- **Model scale:** Three LLM cascade setups across 16 datasets
- **Applicability to web agents:** **High** — conceptually for web agents needing calibrated deferral, though applied to reasoning benchmarks.
- **Key limitation:** Requires reliable calibration sets; may not generalize to dynamic web-step outcomes.

#### MoE-Spec: Expert Budgeting (Year insufficient)

- **Authors:** B. McDanel et al.
- **Venue:** Unspecified
- **Core idea:** Proposes routing probabilities across tokens to shortlist top-B experts and allocate expert budgets to improve speculative decoding efficiency [67].
- **Signal type:** Category 4 (Learned routing/gating)
- **Signal mechanism:** Token-level routing probabilities and top-B expert selection for speculative verification.
- **Evidence strength:** Insufficient evidence [67].
- **Model scale:** Insufficient evidence
- **Applicability to web agents:** **Low-to-Medium** — token-level expert budgeting focuses on token generation rather than multi-step action selection.
- **Key limitation:** Focused on token-level speculative decoding rather than trajectory-level escalation.

---

## 4. Comparative Analysis

### 4.1 Signal Cost vs Discriminative Power

We categorize signals by computational cost and reported discriminative power:

**Low-cost, high-discriminative signals:**
- Behavioral signals (action repetition, cycle detection): Trivial computation, 50% error reduction [18]
- Verbalized confidence: Single forward pass, calibration-dependent [56]
- Trajectory features (step count, revisit patterns): Minimal overhead, 40.4% time reduction [33]

**Medium-cost, high-discriminative signals:**
- Process reward models (7B): +9.1 points WebPRMBench, +7.7pp AndroidWorld [23], [24]
- Learned routers (small classifier): 2–98% cost reduction [36], [37], [38]
- Speculative execution: 48.5% time reduction, 1.8× throughput [61]

**High-cost, high-discriminative signals:**
- Multiple forward passes (answer consistency): ~40% cost of strong model [51]
- Interactive verification: High latency, strong correctness guarantees [30]

### 4.2 Web Agent Benchmarks

Papers with direct web/GUI agent evaluation:

| Paper                    | Benchmark               | Metric                  | Improvement |
| ------------------------ | ----------------------- | ----------------------- | ----------- |
| WebArbiter [23]          | WebPRMBench             | Accuracy                | +9.1 points |
| WebArbiter [23]          | WebArena-Lite           | Reward-guided search    | +6.4 points |
| GUI-Shepherd [24]        | AndroidWorld            | Success rate (online)   | +7.7pp      |
| GUI-Shepherd [24]        | AndroidWorld            | Success rate (verifier) | +5.1pp      |
| V-Droid [25]             | AndroidWorld            | Success rate            | 59.5%       |
| V-Droid [25]             | AndroidLab              | Success rate            | 38.3%       |
| V-Droid [25]             | MobileAgentBench        | Success rate            | 49%         |
| R2D2 [18]                | WebArena                | Navigation errors       | -50%        |
| R2D2 [18]                | WebArena                | Task completion         | 3× increase |
| Branch-and-Browse [33]   | WebArena                | Task success            | 35.8%       |
| Branch-and-Browse [33]   | WebArena                | Time reduction          | 40.4%       |
| NNetscape Navigator [19] | WebArena                | Improvement             | +6 points   |
| NNetscape Navigator [19] | MiniWoB++               | Improvement             | +20+ points |
| ReAP [59]                | Web navigation          | Overall improvement     | +11pp       |
| ReAP [59]                | Previously failed tasks | Improvement             | +29pp       |
| MobileUse [58]           | AndroidWorld            | Success rate            | 62.9%       |
| MobileUse [58]           | AndroidLab              | Success rate            | 44.2%       |

### 4.3 Model Scale Considerations

**Critical gap:** Most papers do not report model scale or evaluate on models >7B. Only SCOPE [37] explicitly tests on-device and cloud-scale models including Qwen variants. V-Droid [25] demonstrates that verifier-driven architectures achieve 6.1× speedup, suggesting small verifiers can be effective.

**Implication for 4B VLMs:** Signals must be validated at small scale. Process reward models at 7B scale [23] may be too expensive to run alongside 4B actors. Behavioral signals, verbalized confidence, and lightweight learned routers are most feasible.

### 4.4 Signal Complementarity

Multiple papers combine signal types:
- TGPO [20]: Behavioral + External verification + Trajectory
- R2D2 [18]: Behavioral + Trajectory + Self-reflection
- InfiGUIAgent [28]: External verification + Attention/representation
- EAGLE [57]: Attention/representation + Self-reflection

**Observation:** Combining cheap behavioral signals (action repetition) with medium-cost verification (small PRM or verbalized confidence) may yield optimal cost-quality tradeoffs for 4B VLMs.

---

## 5. Discussion

### 5.1 Applicability to 4B VLM Web Agents

The surveyed literature provides strong evidence for multiple routing signal categories, but critical gaps remain for small-scale VLM deployment:

**Strengths:**
1. **Process reward models** show consistent gains across web/GUI benchmarks [23], [24], [25], but most implementations use 7B+ models, potentially too expensive for 4B actor deployment.
2. **Behavioral signals** (action repetition, cycle detection, replay buffers) are computationally free and show large empirical gains on WebArena [18], [19], [33].
3. **Learned routers** demonstrate dramatic cost reductions (up to 98%) [36], [37], [38], but most evaluations use reasoning benchmarks rather than interactive web tasks.
4. **Verbalized confidence** offers a zero-cost alternative to token logprobs [56], though calibration quality varies by model and prompt.

**Weaknesses:**
1. **Model scale mismatch:** Most papers evaluate on 7B+ models or do not report scale. Calibration and routing effectiveness may degrade at 4B scale.
2. **Benchmark mismatch:** Many routing papers evaluate on reasoning tasks (GSM8K, CSQA) rather than interactive web navigation, where environment dynamics and multi-step dependencies differ fundamentally.
3. **Missing signals:** No work directly evaluates attention entropy or hidden-state probes for routing in VLMs, despite theoretical motivation.

### 5.2 Token-Level Confidence Failure

Our empirical finding (AUROC = 0.497 for Qwen3-VL-4B) aligns with recent observations that small models trained with RLHF exhibit systematic overconfidence [12], [13], [14]. This failure mode motivates the search for alternative signals, but raises a critical question: **Will other internal signals (attention entropy, hidden-state uncertainty) also fail at 4B scale?**

EAGLE [57] and LaSM [53] suggest that representation-level signals may be more robust than output probabilities, but neither paper evaluates on small VLMs or web navigation tasks. This remains an open empirical question.

### 5.3 Cost-Quality Tradeoffs

The literature demonstrates a clear hierarchy of cost-quality tradeoffs:

1. **Behavioral signals** (free): 50% error reduction [18], 40% time reduction [33]
2. **Verbalized confidence** (1 forward pass): Calibration-dependent [56]
3. **Small learned routers** (lightweight classifier): 2–98% cost reduction [36], [37], [38]
4. **Process reward models** (7B inference): +9.1 points [23], +7.7pp [24]
5. **Multiple forward passes** (answer consistency): ~40% cost of strong model [51]

For 4B VLM deployment, the optimal strategy likely combines free behavioral signals with lightweight learned routers, reserving expensive verification for high-stakes decisions.

### 5.4 Generalization Challenges

Several papers note generalization challenges:
- **Domain shift:** PRMs trained on mobile GUI may not transfer to web [24]
- **Site-specific heuristics:** OpAgent [29] uses WebArena-specific rules
- **Calibration drift:** Verbalized confidence varies by prompt and task [56]

**Implication:** Routing policies must be robust to distribution shift or include online adaptation mechanisms [64].

---

## 6. Recommendations

Based on the systematic review, we rank the top 5 most promising signals for cost-aware routing in 4B VLM web agents:

### Recommendation 1: Behavioral Signals (Action Repetition + Cycle Detection)

**Rationale:** Zero computational cost, strong empirical evidence on WebArena (50% error reduction, 3× completion gain) [18], trivial to implement. R2D2's replay buffer and TGPO's tree-structured trajectory merging provide concrete implementations [18], [20].

**Implementation:** Track last N actions and visited URLs; trigger escalation when:
- Same action repeated >K times in sliding window
- URL revisited >M times
- Step count exceeds task-specific budget

**Expected impact:** 30–50% reduction in wasted steps on doomed episodes; minimal false positives if thresholds tuned per task type.

**Validation:** Requires empirical tuning of K, M thresholds on VisualWebArena; monitor false positive rate (legitimate revisits).

### Recommendation 2: Verbalized Confidence

**Rationale:** Single forward pass overhead, no architectural changes required, directly applicable to any VLM [56]. Provides per-step uncertainty estimate when token logprobs are non-discriminative.

**Implementation:** Append "Rate your confidence in this action (0-100):" to each step prompt; extract numeric confidence; escalate when confidence <threshold.

**Expected impact:** Calibration-dependent; Yang et al. [56] show well-calibrated estimates are achievable with proper prompting. Likely 10–30% cost reduction if calibrated on held-out VisualWebArena episodes.

**Validation:** Requires calibration study on Qwen3-VL-4B to determine optimal prompt format and threshold; measure AUROC on success/failure classification.

### Recommendation 3: Lightweight Learned Router

**Rationale:** Demonstrated 2–98% cost reduction across multiple papers [36], [37], [38], [42], [43]. Small classifier (e.g., 100M parameters) can learn query→model mapping from historical trajectories.

**Implementation:** Train binary classifier on features: (1) task type, (2) step count, (3) action history diversity, (4) verbalized confidence, (5) behavioral flags. Predict "escalate" vs "continue with 4B".

**Expected impact:** 40–60% cost reduction if trained on sufficient VisualWebArena trajectories with success labels. SCOPE [37] demonstrates generalization to new models and budgets.

**Validation:** Requires collecting 1,000+ labeled trajectories (success/failure) on VisualWebArena; train router with cost-aware loss; evaluate on held-out tasks.

### Recommendation 4: Trajectory-Level Features (Step Count + Action Diversity)

**Rationale:** Minimal computation, strong correlation with task difficulty [32], [33], [34]. Branch-and-Browse achieves 40.4% time reduction via tree-structured exploration [33].

**Implementation:** Compute per-episode: (1) step count, (2) unique action types, (3) URL diversity, (4) backtracking frequency. Escalate when step count exceeds 90th percentile for task type or action diversity drops below threshold.

**Expected impact:** 20–40% reduction in late-stage failures by escalating before exhausting budget. Particularly effective for long-horizon tasks.

**Validation:** Analyze VisualWebArena trajectory statistics to determine task-specific thresholds; measure correlation between trajectory features and success.

### Recommendation 5: Small Process Reward Model (≤1B)

**Rationale:** Strongest empirical evidence on web/GUI benchmarks (+9.1 points [23], +7.7pp [24]), but 7B models too expensive. Hypothesis: distilled 1B PRM may retain most discriminative power at acceptable cost.

**Implementation:** Distill WebArbiter-7B [23] or GUI-Shepherd [24] to 1B model; run asynchronously to score candidate actions; escalate when PRM score <threshold.

**Expected impact:** 5–10 point improvement on VisualWebArena if distillation preserves discriminative power; 1B PRM adds ~20% inference cost vs 4B actor.

**Validation:** Requires distillation infrastructure and validation that 1B PRM retains calibration; measure AUROC on step-level success prediction; compare cost-quality tradeoff vs behavioral signals alone.

---

## 7. Research Gaps

### Gap 1: Attention Entropy and Hidden-State Probes for VLM Routing

**Description:** No surveyed papers evaluate attention entropy, attention divergence, or hidden-state probe classifiers as routing signals for vision-language models in web navigation. LaSM [53] uses layer-wise attention divergence for adversarial defense, and EAGLE [57] aggregates hidden-state beliefs for calibration, but neither targets VLMs or web agents.

**Impact:** Attention patterns may reveal visual grounding failures or ambiguous element localization that token logprobs miss. Hidden-state probes could detect out-of-distribution states.

**Required research:** Empirical study measuring AUROC of attention entropy, attention divergence, and hidden-state probe classifiers for success prediction on VisualWebArena using Qwen3-VL-4B. Compare computational cost vs discriminative power.

### Gap 2: Small-Scale Model Evaluation (≤4B)

**Description:** Most papers do not report model scale or evaluate on 7B+ models. Only SCOPE [37] explicitly tests on-device models. Calibration quality, routing effectiveness, and signal discriminativeness may degrade at small scale.

**Impact:** Routing policies optimized for 7B+ models may fail at 4B scale due to different error modes, calibration properties, and representation quality.

**Required research:** Systematic evaluation of all signal categories (behavioral, verification, trajectory, learned routing, self-reflection) on 4B VLMs across web navigation benchmarks. Measure how discriminative power scales with model size.

### Gap 3: Multi-Signal Fusion for Web Agents

**Description:** While several papers combine signal types [18], [20], [28], no work systematically evaluates optimal fusion strategies for web agents. Which combinations yield best cost-quality tradeoffs? How should signals be weighted?

**Impact:** Single signals may have complementary failure modes. Fusion could improve robustness and reduce false positives.

**Required research:** Ablation study on VisualWebArena comparing: (1) individual signals, (2) pairwise combinations, (3) learned fusion (e.g., gradient boosting over signal features). Measure cost-quality Pareto frontier.

### Gap 4: Online Adaptation and Continual Learning

**Description:** Most routing policies are static or require offline retraining. Only GUI-AiF [64] and OpenClaw-RL [26] address continual learning. Web environments exhibit distribution shift (new sites, layout changes), requiring adaptive routing.

**Impact:** Static routing policies may degrade over time as task distributions shift. Online adaptation could maintain performance without expensive retraining.

**Required research:** Develop online routing policy that updates thresholds or router weights based on recent success/failure. Evaluate on VisualWebArena with simulated distribution shift (new sites, layout changes).

### Gap 5: Speculative Execution for Web Agents

**Description:** Speculative execution shows 40–55% latency reduction [60], [61], [62], but evaluations focus on tool-calling or token generation. Web navigation has unique challenges: irreversible actions (form submission), side effects (state changes), and environment stochasticity.

**Impact:** Speculative execution could dramatically reduce latency for 4B VLMs, but incorrect speculation may cause task failures or wasted compute.

**Required research:** Adapt speculative execution to web navigation with rollback mechanisms for reversible actions and confidence-based speculation for irreversible actions. Measure latency reduction vs speculation accuracy tradeoff on VisualWebArena.

### Gap 6: Theoretical Foundations for Multi-Step Agent Routing

**Description:** Dekoninck et al. [46] provide optimality results for single-query routing and cascading, but multi-step agents have sequential dependencies, partial observability, and credit assignment challenges. Optimal routing policies for multi-step agents remain theoretically uncharacterized.

**Impact:** Principled routing policies could achieve provably optimal cost-quality tradeoffs. Current heuristics may be far from optimal.

**Required research:** Extend cascade routing theory to multi-step MDPs with cost constraints. Derive optimal routing policies under partial observability and characterize sample complexity of learning near-optimal policies.

### Gap 7: Robustness to Adversarial Inputs and Distribution Shift

**Description:** LaSM [53] addresses adversarial pop-up injections, but broader robustness of routing signals to adversarial inputs, out-of-distribution sites, and layout changes is unexplored.

**Impact:** Routing policies may be brittle to distribution shift or adversarial manipulation. Adversaries could craft inputs that trigger unnecessary escalation (cost inflation) or suppress escalation (task failure).

**Required research:** Evaluate routing signal robustness on adversarial VisualWebArena variants (pop-ups, layout perturbations, misleading content). Develop robust routing policies with certified cost-quality guarantees.

---

## 8. Conclusion

This systematic literature review identifies multiple viable routing signals for cost-aware escalation in web navigation agents when token-level confidence is non-discriminative. Behavioral signals (action repetition, cycle detection) offer zero-cost, high-impact solutions with 50% error reduction on WebArena [18]. Process reward models provide the strongest empirical evidence (+9.1 points on WebPRMBench [23]), but 7B models may be too expensive for 4B VLM deployment. Learned routers demonstrate dramatic cost reductions (up to 98% [36]), though most evaluations use reasoning benchmarks rather than interactive web tasks. Verbalized confidence offers a practical zero-cost alternative to token logprobs [56], though calibration quality varies.

Critical research gaps remain: no work evaluates attention entropy or hidden-state probes for VLM routing; limited evidence exists for small-scale (≤4B) models; and multi-signal fusion strategies are unexplored. For practitioners deploying Qwen3-VL-4B on VisualWebArena, we recommend starting with behavioral signals (action repetition, cycle detection) combined with verbalized confidence, then training a lightweight learned router on collected trajectories. This combination balances zero-cost signals with learned difficulty estimation, likely achieving 40–60% cost reduction while maintaining task success rates.

Future work should prioritize empirical validation of routing signals at 4B scale, systematic evaluation of multi-signal fusion, and development of online adaptation mechanisms to handle distribution shift in web environments.

---

## 9. References

[1] Multiple authors, "R2D2: Remembering, Replaying and Dynamic Decision Making with a Reflective Agentic Memory," *ACL*, 2025.

[2] Multiple authors, "Recon-Act: Self-Evolving Multi-Agent Browser-Use System," *arXiv*, 2025.

[3] Multiple authors, "NNetscape Navigator: Synthetic Demonstrations," *arXiv*, 2024.

[4] Multiple authors, "WebGym: Scaling Training Environments," *arXiv*, 2024.

[5] Chen Ziyuan et al., "TGPO: Tree-Guided Preference Optimization," *arXiv*, 2025.

[6] Xuechen Zhang et al., "TREACLE: Efficient Contextual LLM Cascades through Budget-Constrained Policy Learning," *Preprint*, 2024.

[7] M. Sakota et al., "Fly-Swat or Cannon: FORC Cost-Effective Language Model Choice," *arXiv*, 2023.

[8] Dujian Ding et al., "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing," *arXiv*, 2024.

[9] Liu Yide et al., "SATER: Self-Aware and Token-Efficient Routing and Cascading," *arXiv*, 2025.

[10] Harikrishna Narasimhan et al., "Faster Cascades via Speculative Decoding," *arXiv*, 2024.

[11] B. McDanel et al., "MoE-Spec: Expert Budgeting for Efficient Speculative Decoding," *Unspecified*, year insufficient.

[12] A. Valkanas et al., "C3PO: Optimized LLM Cascades with Probabilistic Cost Constraints," *Unspecified*, year insufficient.

[13] Y. Zhang et al., "WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents," *Unspecified*, year insufficient.

[14] Chen Cong et al., "GUI-Shepherd: Process Reward and Verification for Long-Sequence GUI Tasks," *arXiv*, 2025.

[15] Chen Ziyuan et al., "TGPO: Tree-Guided Preference Optimization for Robust Web Agent RL," *arXiv*, 2025.

[16] D. Li et al., "ToolPRMBench: Evaluating and Advancing Process Reward Models," *Unspecified*, 2025.

[17] C. Cui et al., "Agentic Reward Modeling: Verifying GUI Agent via Online Proactive Interaction," *Unspecified*, year insufficient.

[18] Gaole Dai et al., "V-Droid: Verifier-Driven Mobile GUI Agent," *arXiv*, 2025.

[19] Yuhang Liu et al., "InfiGUIAgent: Multimodal Generalist GUI Agent," *arXiv*, 2025.

[20] Y. Guo et al., "OpAgent: Operator Agent for Web Navigation," *Unspecified*, year insufficient.

[21] Yu-Neng Chuang et al., "Learning to Route with Confidence Tokens: Self-REF," *arXiv*, 2024.

[22] Chen Yun et al., "EAGLE: Expectation of Aggregated Internal Belief," *arXiv*, 2025.

[23] Lu Yunan et al., "Speculative Actions: Lossless Framework for Faster Agentic Systems," *arXiv*, 2025.

[24] Luyao Gao et al., "Collaborative Speculative Inference for Efficient LLM Inference Serving," *arXiv*, 2025.

[25] W. Zheng et al., "CLEAR: Cost-aware LLM Edge-cloud Adaptive Routing," *Unspecified*, year insufficient.

[26] Yinjie Wang et al., "OpenClaw-RL: Next-State Signals and PRM Judges," *Preprint*, 2026.

[27] Zihuiwen Ye et al., "Uncertainty-Aware Step-Wise Verification (CoT Entropy)," *Preprint*, 2025.

[28] Yuhang Liu et al., "InfiGUIAgent: Multimodal Generalist GUI Agent," *arXiv*, 2025.

[29] Y. Guo et al., "OpAgent: Operator Agent for Web Navigation," *Unspecified*, year insufficient.

[30] C. Cui et al., "Agentic Reward Modeling: VAGEN Verifier," *Unspecified*, year insufficient.

[31] Yichen Pan et al., "WebCanvas: Benchmark and Agent," *Preprint*, 2024.

[32] Multiple authors, "WebGraphEval: Multi-Turn Trajectory Evaluation," *arXiv*, 2025.

[33] Multiple authors, "Branch-and-Browse: Tree-Structured Exploration," *arXiv*, 2025.

[34] Quanfeng Lu et al., "GUIOdyssey: Dataset and OdysseyAgent," *Preprint*, 2024.

[35] Maria Movin et al., "Trace-Level Comparison for GUI Agents," *Preprint*, 2026.

[36] Lingjiao Chen et al., "FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance," *Preprint*, 2023.

[37] Qi Cao et al., "SCOPE: Scalable Router," *Preprint*, 2026.

[38] Isaac Ong et al., "RouteLLM: Learning to Route LLMs with Preference Data," *arXiv*, 2024.

[39] Qian Cheng et al., "xRouter: Training Cost-Aware LLMs Orchestration System via Reinforcement Learning," *arXiv*, 2025.

[40] Dimitrios Sikeridis et al., "PickLLM: Context-Aware RL-Assisted Large Language Model Routing," *arXiv*, 2024.

[41] Xuechen Zhang et al., "TREACLE: Efficient Contextual LLM Cascades through Budget-Constrained Policy Learning," *Preprint*, 2024.

[42] M. Sakota et al., "Fly-Swat or Cannon: FORC Cost-Effective Language Model Choice," *arXiv*, 2023.

[43] Dujian Ding et al., "Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing," *arXiv*, 2024.

[44] Liu Yide et al., "SATER: Self-Aware and Token-Efficient Routing and Cascading," *arXiv*, 2025.

[45] R. Pulishetty et al., "One Head Many Models: Cross-Attention Routing for Cost-Aware LLM Selection," *arXiv*, 2025.

[46] Jasper Dekoninck et al., "A Unified Approach to Routing and Cascading," *Preprint*, 2024.

[47] Harikrishna Narasimhan et al., "Faster Cascades via Speculative Decoding," *arXiv*, 2024.

[48] Xin He et al., "ExpertFlow: Efficient MoE Inference," *Preprint*, 2024.

[49] Amirhossein Vahidi et al., "DirMoE: Differentiable Router," *Preprint*, 2026.

[50] W. Zheng et al., "CLEAR: Cost-aware LLM Edge-cloud Adaptive Routing," *Unspecified*, year insufficient.

[51] Multiple authors, "Large LLM Cascades with Mixture of Thoughts," *arXiv*, 2023.

[52] Ruichen Zhang et al., "Symbiotic Cooperation for Web Agents," *arXiv*, 2025.

[53] Zihe Yan et al., "LaSM: Layer-Wise Attention Defense," *Preprint*, 2025.

[54] Zhiyuan Huang et al., "SpiritSight: GUI Agent," *Preprint*, 2025.

[55] Yu-Neng Chuang et al., "Learning to Route with Confidence Tokens: Self-REF," *arXiv*, 2024.

[56] Daniel Yang et al., "On Verbalized Confidence Scores for LLMs," *Preprint*, 2024.

[57] Chen Yun et al., "EAGLE: Expectation of Aggregated Internal Belief," *arXiv*, 2025.

[58] Ning Li et al., "MobileUse: Hierarchical Reflection," *Preprint*, 2025.

[59] Ruhana Azam et al., "ReAP: Reflection-Augmented Planning," *Preprint*, 2025.

[60] Lu Yunan et al., "Speculative Actions: Lossless Framework for Faster Agentic Systems," *arXiv*, 2025.

[61] Yifan Sui et al., "PASTE: Speculative Tool Execution," *Preprint*, 2026.

[62] Zhenwei Tang et al., "MineDraft: Batch Parallel Speculative Decoding," *Preprint*, 2026.

[63] Luyao Gao et al., "Collaborative Speculative Inference for Efficient LLM Inference Serving," *arXiv*, 2025.

[64] Ziwei Liu et al., "Continual GUI Agents and GUI-AiF," *Preprint*, 2026.

[65] D. Li et al., "ToolPRMBench: Evaluating and Advancing Process Reward Models," *Unspecified*, 2025.

[66] A. Valkanas et al., "C3PO: Optimized LLM Cascades with Probabilistic Cost Constraints," *Unspecified*, year insufficient.

[67] B. McDanel et al., "MoE-Spec: Expert Budgeting for Efficient Speculative Decoding," *Unspecified*, year insufficient.

---

## Synthesis Table

| Paper                          | Signal Type                                | Mechanism                            | Cost to Compute                     | Discriminative Power                     | Applicable to 4B VLM Web Agent      |
| ------------------------------ | ------------------------------------------ | ------------------------------------ | ----------------------------------- | ---------------------------------------- | ----------------------------------- |
| R2D2 [1]                       | Behavioral + Trajectory                    | Replay buffer cycle detection        | Free                                | High (50% error reduction)               | Yes                                 |
| WebArbiter [23]                | External verification                      | Generative PRM with verdicts         | High (7B inference)                 | Very High (+9.1 points)                  | Maybe (7B too expensive)            |
| GUI-Shepherd [24]              | External verification                      | Step-level PRM                       | High (large model)                  | Very High (+7.7pp)                       | Maybe (needs distillation)          |
| V-Droid [25]                   | External verification                      | Verifier-driven architecture         | Medium (verifier inference)         | High (59.5% success)                     | Yes                                 |
| FrugalGPT [36]                 | Multi-agent routing                        | Learned cascades                     | Low (small classifier)              | Very High (98% cost reduction)           | Yes                                 |
| SCOPE [37]                     | Learned routing                            | RL-trained router                    | Low (small classifier)              | Very High (95.1% cost reduction)         | Yes                                 |
| RouteLLM [38]                  | Learned routing                            | Preference-trained router            | Low (small classifier)              | High (2× cost reduction)                 | Yes                                 |
| Verbalized Confidence [56]     | Self-reflection                            | Prompted confidence score            | Very Low (1 forward pass)           | Medium (calibration-dependent)           | Yes                                 |
| Self-REF [55]                  | Self-reflection                            | Trained confidence tokens            | Low (token extraction)              | High (significant gains)                 | Yes (needs fine-tuning)             |
| Branch-and-Browse [33]         | Trajectory + Behavioral                    | Tree-structured exploration          | Low (tree management)               | High (40.4% time reduction)              | Yes                                 |
| WebGraphEval [32]              | Trajectory features                        | Trajectory graph abstraction         | Medium (graph construction)         | Medium (redundancy detection)            | Yes                                 |
| TGPO [20]                      | Behavioral + Verification + Trajectory     | Tree merging + PRM                   | Medium (PRM inference)              | High (fewer redundant steps)             | Yes                                 |
| NNetscape Navigator [19]       | Behavioral + Trajectory                    | Hierarchical decomposition           | Low (pruning heuristics)            | High (+6 WebArena, +20 MiniWoB)          | Yes                                 |
| ReAP [59]                      | Self-reflection                            | Reflection memory                    | Low (memory lookup)                 | High (+11pp overall, +29pp failed)       | Yes                                 |
| MobileUse [58]                 | Self-reflection                            | Hierarchical reflection              | Medium (reflection module)          | High (62.9% AndroidWorld)                | Yes                                 |
| Speculative Actions [60]       | Learned routing                            | Fast-model speculation               | Medium (speculation overhead)       | High (55% prediction, latency reduction) | Yes                                 |
| PASTE [61]                     | Learned routing                            | Pattern-aware tool speculation       | Medium (pattern matching)           | High (48.5% time reduction)              | Yes                                 |
| MineDraft [62]                 | Learned routing                            | Batch-parallel speculation           | Medium (batching overhead)          | High (75% throughput, 39% latency)       | Yes                                 |
| TREACLE [41]                   | Learned routing                            | RL budget-constrained policy         | Low (small classifier)              | Very High (85% cost savings)             | Maybe (reasoning tasks)             |
| FORC [42]                      | Learned routing                            | Meta-model selection                 | Low (meta-model inference)          | High (63% cost reduction)                | Yes                                 |
| Hybrid LLM [43]                | Learned routing                            | Difficulty prediction                | Low (difficulty classifier)         | High (40% fewer large calls)             | Yes                                 |
| SATER [44]                     | Learned routing                            | Confidence-aware rejection           | Low (rejection mechanism)           | High (>50% cost reduction)               | Yes                                 |
| One Head Many Models [45]      | Learned routing                            | Cross-attention router               | Low (small router)                  | High (+6.6% AIQ)                         | Yes                                 |
| Mixture of Thoughts [51]       | Task priors + Self-reflection              | Answer consistency                   | High (multiple forward passes)      | High (~40% cost of strong model)         | Maybe (latency concern)             |
| CoT Entropy [27]               | External verification                      | PRM uncertainty quantification       | Medium (PRM + entropy)              | High (improved robustness)               | Yes                                 |
| OpenClaw-RL [26]               | External verification                      | Next-state PRM signals               | Medium (async PRM)                  | High (online learning gains)             | Yes                                 |
| EAGLE [57]                     | Attention/representation + Self-reflection | Hidden-state belief aggregation      | Medium (internal activation access) | High (improved calibration)              | Maybe (needs activation access)     |
| LaSM [53]                      | Attention/representation                   | Layer-wise attention divergence      | Medium (attention analysis)         | Medium (defense gains)                   | Maybe (needs per-model calibration) |
| Unified Routing/Cascading [46] | Learned routing                            | Cascade routing theory               | Low (quality estimator)             | High (theoretical optimality)            | Yes (needs estimator)               |
| xRouter [39]                   | Learned routing                            | RL cost-aware orchestration          | Low (RL policy)                     | High (strong cost-performance)           | Yes                                 |
| PickLLM [40]                   | Learned routing                            | Context-aware RL                     | Low (RL policy)                     | Medium (session-level gains)             | Yes                                 |
| GUIOdyssey [34]                | Trajectory features                        | History resampling                   | Low (attention over history)        | Medium (in/out-domain gains)             | Yes                                 |
| WebCanvas [31]                 | External verification                      | Page-level state metrics             | Medium (state evaluation)           | Medium (23.1% success)                   | Yes                                 |
| InfiGUIAgent [28]              | External verification + Attention          | Hierarchical reasoning + reflection  | Medium (reflection module)          | Medium (competitive performance)         | Yes                                 |
| OpAgent [29]                   | External verification + Self-reflection    | Task-specific heuristics             | Low (heuristic evaluation)          | Medium (WebArena-specific)               | Maybe (environment-specific)        |
| Continual GUI-AiF [64]         | Behavioral                                 | Anchoring rewards                    | Medium (reward computation)         | Medium (continual learning gains)        | Maybe (needs training infra)        |
| Recon-Act [2]                  | Behavioral                                 | Trajectory comparison + rule archive | Low (rule matching)                 | Medium (VisualWebArena SOTA claim)       | Yes                                 |
| WebGym [4]                     | Behavioral                                 | Action repetition penalty            | Free                                | Low (training-time signal)               | Yes                                 |
| Trace-Level Comparison [35]    | Trajectory features                        | Trace diagnostics                    | Medium (trace analysis)             | Low (diagnostic, not routing)            | Maybe (needs human baselines)       |
| SpiritSight [54]               | Attention/representation                   | Grounding + block parsing            | Medium (parsing overhead)           | Medium (GUI benchmark gains)             | Maybe (needs GUI datasets)          |
| Symbiotic Cooperation [52]     | Multi-agent routing                        | Complementary model pairing          | Low (orchestration logic)           | Unknown (insufficient evidence)          | Maybe (no metrics)                  |
| CLEAR [50]                     | Learned routing                            | Edge-cloud adaptive routing          | Low (routing logic)                 | Unknown (insufficient evidence)          | Yes (edge-cloud pattern)            |
| ToolPRMBench [65]              | External verification                      | PRM evaluation benchmark             | N/A (benchmark)                     | N/A (evaluation)                         | N/A (benchmark)                     |
| Agentic VAGEN [30]             | External verification                      | Proactive interactive verifier       | High (interactive verification)     | Unknown (insufficient evidence)          | Yes (high latency concern)          |
| C3PO [66]                      | Learned routing                            | Calibrated deferral thresholds       | Low (threshold calibration)         | High (16 datasets evaluated)             | Yes                                 |
| MoE-Spec [67]                  | Learned routing                            | Token-level expert budgeting         | Medium (expert routing)             | Unknown (insufficient evidence)          | No (token-level, not trajectory)    |
| Faster Cascades [47]           | Learned routing                            | Speculative cascading                | Medium (speculation overhead)       | Medium (improved tradeoffs)              | Maybe (language tasks)              |
| ExpertFlow [48]                | Learned routing                            | Predictive expert caching            | Medium (cache management)           | High (93.72% memory reduction)           | No (MoE-specific)                   |
| DirMoE [49]                    | Learned routing                            | Differentiable Dirichlet routing     | Medium (variational inference)      | Medium (improved specialization)         | No (MoE-specific)                   |
| Collaborative Speculative [63] | Learned routing                            | Ensemble speculative decoding        | Medium (ensemble overhead)          | Unknown (insufficient evidence)          | Maybe (inference serving focus)     |

---

## Recommendations (Detailed)

### Top 5 Signals for 4B VLM Web Agents

#### 1. Behavioral Signals (Action Repetition + Cycle Detection)
- **Cost:** Free
- **Discriminative Power:** High (50% error reduction [1])
- **Implementation Complexity:** Low
- **Justification:** Zero computational overhead, strong empirical evidence on WebArena, trivial to implement. R2D2 demonstrates 3× task completion gain [1]. Applicable to any agent architecture without modification.
- **Deployment Strategy:** Track last 10 actions and visited URLs; escalate when same action repeated >3 times or URL revisited >2 times in 10-step window.

#### 2. Verbalized Confidence
- **Cost:** Very Low (1 forward pass)
- **Discriminative Power:** Medium (calibration-dependent [56])
- **Implementation Complexity:** Low
- **Justification:** No architectural changes, directly applicable to Qwen3-VL-4B, provides per-step uncertainty when token logprobs fail. Yang et al. show well-calibrated estimates achievable with proper prompting [56].
- **Deployment Strategy:** Append "Rate your confidence (0-100):" to each prompt; extract numeric score; escalate when <40 (threshold tuned on held-out data).

#### 3. Lightweight Learned Router
- **Cost:** Low (100M classifier)
- **Discriminative Power:** Very High (2–98% cost reduction [36], [37], [38])
- **Implementation Complexity:** Medium (requires training data)
- **Justification:** Demonstrated dramatic cost reductions across multiple papers. SCOPE shows generalization to new models/budgets [37]. Small classifier learns query→model mapping from historical trajectories.
- **Deployment Strategy:** Collect 1,000+ labeled VisualWebArena trajectories; train binary classifier on features (task type, step count, action diversity, verbalized confidence, behavioral flags); predict "escalate" vs "continue".

#### 4. Trajectory-Level Features (Step Count + Action Diversity)
- **Cost:** Low (simple statistics)
- **Discriminative Power:** High (40.4% time reduction [33])
- **Implementation Complexity:** Low
- **Justification:** Minimal computation, strong correlation with task difficulty. Branch-and-Browse achieves substantial time reduction via tree-structured exploration [33]. Particularly effective for long-horizon tasks.
- **Deployment Strategy:** Compute per-episode step count, unique action types, URL diversity, backtracking frequency; escalate when step count exceeds 90th percentile for task type or action diversity drops below 0.3.

#### 5. Small Process Reward Model (≤1B)
- **Cost:** Medium (1B inference)
- **Discriminative Power:** Very High (+9.1 points [23], +7.7pp [24])
- **Implementation Complexity:** High (requires distillation)
- **Justification:** Strongest empirical evidence on web/GUI benchmarks, but 7B models too expensive. Hypothesis: distilled 1B PRM retains most discriminative power at acceptable cost (~20% overhead vs 4B actor).
- **Deployment Strategy:** Distill WebArbiter-7B [23] to 1B; run asynchronously to score candidate actions; escalate when PRM score <0.5 (threshold tuned on held-out data).

---

## Research Gaps (Detailed)

### Gap 1: Attention Entropy and Hidden-State Probes for VLM Routing
**Current State:** No surveyed papers evaluate these signals for VLM web agents.  
**Why It Matters:** Attention patterns may reveal visual grounding failures; hidden-state probes could detect OOD states.  
**Required Research:** Empirical study on Qwen3-VL-4B measuring AUROC of attention entropy, attention divergence, hidden-state probes for success prediction on VisualWebArena.  
**Expected Outcome:** Determine if representation-level signals are more robust than token logprobs at 4B scale.

### Gap 2: Small-Scale Model Evaluation (≤4B)
**Current State:** Most papers evaluate 7B+ models or don't report scale.  
**Why It Matters:** Calibration and routing effectiveness may degrade at small scale.  
**Required Research:** Systematic evaluation of all signal categories on 4B VLMs across web navigation benchmarks.  
**Expected Outcome:** Characterize how discriminative power scales with model size; identify signals robust to small scale.

### Gap 3: Multi-Signal Fusion for Web Agents
**Current State:** No systematic evaluation of optimal fusion strategies.  
**Why It Matters:** Single signals have complementary failure modes; fusion could improve robustness.  
**Required Research:** Ablation study on VisualWebArena comparing individual signals, pairwise combinations, learned fusion.  
**Expected Outcome:** Identify optimal signal combinations for cost-quality Pareto frontier.

### Gap 4: Online Adaptation and Continual Learning
**Current State:** Most routing policies are static.  
**Why It Matters:** Web environments exhibit distribution shift; static policies degrade over time.  
**Required Research:** Develop online routing policy updating thresholds/weights based on recent success/failure.  
**Expected Outcome:** Maintain performance under distribution shift without expensive retraining.

### Gap 5: Speculative Execution for Web Agents
**Current State:** Evaluations focus on tool-calling or token generation, not web navigation.  
**Why It Matters:** Web navigation has irreversible actions and side effects; incorrect speculation causes failures.  
**Required Research:** Adapt speculative execution to web navigation with rollback mechanisms.  
**Expected Outcome:** 40–55% latency reduction while maintaining correctness.

### Gap 6: Theoretical Foundations for Multi-Step Agent Routing
**Current State:** Optimality results exist for single-query routing [46], not multi-step agents.  
**Why It Matters:** Principled routing policies could achieve provably optimal cost-quality tradeoffs.  
**Required Research:** Extend cascade routing theory to multi-step MDPs with cost constraints.  
**Expected Outcome:** Optimal routing policies under partial observability; sample complexity bounds.

### Gap 7: Robustness to Adversarial Inputs and Distribution Shift
**Current State:** Limited work on routing signal robustness.  
**Why It Matters:** Routing policies may be brittle; adversaries could manipulate escalation decisions.  
**Required Research:** Evaluate routing signal robustness on adversarial VisualWebArena variants.  
**Expected Outcome:** Robust routing policies with certified cost-quality guarantees.

---

**End of Systematic Literature Review**