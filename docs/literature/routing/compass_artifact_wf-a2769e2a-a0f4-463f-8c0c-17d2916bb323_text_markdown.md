# Routing for Cost/Efficiency in LLM and Multimodal Web/GUI Agents — Full Landscape Map, with a Deep Dive on Input-Representation Routing

## TL;DR
- **Routing for efficiency spans at least ten distinct axes** — model selection, cascades, early-exit/adaptive-depth, speculative decoding, token/visual-token pruning, prompt/context compression, modality selection, reasoning-effort/test-time-compute routing, MoE/encoder routing, and tool/retrieval routing — but the specific axis of **input-representation routing (choosing HOW to serialize the same observation for a fixed model) is the least developed, especially for web/GUI agents.**
- For web/GUI agents, the dominant practice is to **pick one representation (AXTree, SoM, or screenshot) as a static deployment default**, and the small body of work that *compares* representations (VisualWebArena, SeeAct, "Read More, Think More") does so as **one-time ablations**, not as a per-instance runtime router on a fixed model.
- **There is no peer-reviewed systematic characterization of per-task input-representation routing for web agents on a fixed model.** The nearest neighbor is a dual-agent binary DOM↔vision switch (V-GEMS, unverified 2026 preprint); the strongest motivation is "Read More, Think More" (which shows the optimal representation depends on model capability/budget but only recommends a static, offline choice). This is a genuine, defensible research gap.

## Key Findings
1. The cost/efficiency-routing literature is mature on **model routing** (RouteLLM, FrugalGPT, Hybrid LLM, CSCR) and **inference-level efficiency** (speculative decoding, early-exit, token pruning, prompt compression), almost all of which is general-LLM work rather than agent-specific.
2. Web/GUI agents inherit these methods but add agent-specific axes: **observation pruning/compression** (FocusAgent, SimpAgent), **visual-token reduction** (ShowUI, GUI-KV), and **adaptive perception effort** (iSHIFT).
3. The distinction the user cares about — **(a) static ablation/design choice of representation vs. (b) dynamic per-instance routing among representations on a fixed model** — is real and sharp: essentially all web-agent work is type (a) or does within-representation pruning, not type (b).

---

## PART 1 — Full Taxonomy of Cost/Efficiency Routing Axes

### Axis 1 — Model routing / selection across a pool
- **RouteLLM: Learning to Route LLMs with Preference Data** — ICLR 2025; arXiv:2406.18665 (VERIFIED). Routes WHAT: each query to a stronger-vs-weaker model based on a learned win-rate predictor. Routing GPT-4 vs Mixtral-8x7B, the matrix-factorization router achieved 95% of GPT-4 performance using only 26% of GPT-4 calls, yielding over 85% cost reduction on MT Bench (45% on MMLU, 35% on GSM8K) at 95% of GPT-4 performance.
- **FrugalGPT: How to Use LLMs While Reducing Cost and Improving Performance** — arXiv preprint 2023; arXiv:2305.05176 (VERIFIED). Routes WHAT: queries through a learned LLM cascade/combination to minimize cost.
- **Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing** — ICLR 2024 (OpenReview id 02f3mUtqnM confirmed; arXiv id NOT independently confirmed — UNVERIFIED). Routes WHAT: each query to a small (edge) vs. large (cloud) model by predicted difficulty, with a tunable test-time quality level.
- **Cost-Aware Contrastive Routing for LLMs (CSCR)** — 2025 preprint; arXiv:2508.12491 (VERIFIED). Routes WHAT: prompts to the cheapest accurate model in a dynamic pool via a shared embedding space + single k-NN (FAISS) lookup; improves the accuracy-cost tradeoff by up to 25% and generalizes to unseen LLMs.

### Axis 2 — Cascades (cheap→expensive escalation)
- **FrugalGPT** (above; arXiv:2305.05176, VERIFIED) is the canonical cascade. Routes WHAT: escalates from a cheap to an expensive LLM when the cheap model's answer is judged insufficient. It can match GPT-4 with up to 98% cost reduction or improve accuracy over GPT-4 by 4% at the same cost; on the HEADLINES task it sends only 16.6% of queries to GPT-4.
- **Is Escalation Worth It? A Decision-Theoretic Characterization of LLM Cascades** — 2026 preprint; arXiv:2605.06350 (UNVERIFIED — future-dated ID seen in search, not independently confirmed). Routes WHAT: the deferral threshold between cheap and expensive models; characterizes the cost-quality frontier as an envelope over pairwise cascades.
- Mixture-of-Thought / answer-consistency cascades (ICLR 2024 reasoning cascade) escalate based on weak-model answer consistency.

### Axis 3 — Early-exit / halting / adaptive-depth / dynamic computation
- **ADEPT: Adaptive Dynamic Early-Exit Process for Transformers** — 2026 preprint; arXiv:2601.03700 (UNVERIFIED — future-dated). Routes WHAT: per-token computation depth (when to halt layers) in both prefill and generation.
- **You Need Multiple Exiting: Dynamic Early Exiting for Accelerating Unified Vision Language Model (MuE)** — CVPR 2023; arXiv:2211.11152 (VERIFIED via search). Routes WHAT: per-input layer-skipping in encoder+decoder of a VLM by modality (up to 50%/40% inference-time reduction on SNLI-VE/MS COCO).
- Classic lineage: Adaptive Computation Time (Graves 2016), PonderNet, Mixture-of-Depths. Routes WHAT: how much depth/compute per token (context only).

### Axis 4 — Speculative / draft-verify decoding
- **Speculative decoding** (Leviathan et al. 2023; Chen et al. 2023 speculative sampling). Routes WHAT: drafts K tokens with a small model, verifies them in parallel with the target model, accepting the longest agreeing prefix — accelerates inference (~2-3×) without changing the output distribution.
- **Medusa: Simple LLM Inference Acceleration with Multiple Decoding Heads** — ICML 2024; arXiv:2401.10774 (UNVERIFIED). Routes WHAT: parallel draft heads verified by the base model.
- **SDSAT: Accelerating LLM Inference through Speculative Decoding with Semantic Adaptive Tokens** — arXiv:2403.18647 (VERIFIED via search). Routes WHAT: semantic adaptive tokens that produce higher-quality drafts without a separate draft model.

### Axis 5 — Token & visual-token pruning, KV-cache compression
- **ShowUI: One Vision-Language-Action Model for GUI Visual Agent** — CVPR 2025; arXiv:2411.17465 (VERIFIED via search). Routes WHAT: UI-guided selection of which visual tokens to keep — reduces redundant visual tokens by 33% and accelerates training by 1.4×, cutting tokens from 1296 to as few as 291 in sparse areas like Google search pages, while the 2B model reaches 75.1% zero-shot grounding.
- **GUI-KV** (referenced in "Less is More"). Routes WHAT: KV-cache entries to keep across GUI screenshots — in a 5-screenshot setting on AgentNetBench it cut decoding FLOPs by 38.9% while increasing step accuracy by 4.1% over the full-cache baseline.
- **PruneVid** (arXiv:2412.16117, VERIFIED via search) and **SCOPE: Saliency-Coverage Oriented Token Pruning** (arXiv:2510.24214, VERIFIED via search). Route WHAT: which visual tokens to retain for efficient MLLM/video inference (PruneVid prunes >80% of tokens).

### Axis 6 — Prompt / context compression
- **LLMLingua: Compressing Prompts for Accelerated Inference** — EMNLP 2023; arXiv:2310.05736 (VERIFIED). Routes WHAT: which tokens to drop from the prompt by perplexity via a budget controller — up to 20× compression with only a ~1.5-point performance drop; on GSM8K it reaches 77.33 EM at 20× compression vs 48.75 EM zero-shot.
- **LongLLMLingua** — arXiv:2310.06839 (VERIFIED). Routes WHAT: key-information-aware compression in long-context (up to ~21.4% performance boost with ~4× fewer tokens on NaturalQuestions).
- **LLMLingua-2** — arXiv:2403.12968 (VERIFIED). Routes WHAT: task-agnostic token classification for faithful compression via a small encoder.

### Axis 7 — Modality selection (vision vs text)
- **WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models** — ACL 2024; arXiv:2401.13919 (VERIFIED via search). Routes WHAT: a design-level choice to use screenshot+SoM vs text-only; WebVoyager achieves a 59.1% task success rate on its 643-task / 15-website benchmark, significantly outperforming GPT-4 (All Tools) at 30.8% and the text-only (WebArena accessibility-tree) setting at 40.1%. This is a static design comparison, not per-instance routing.
- **SCOPE: Selective Cross-modal Orchestration of Visual Perception Experts** — 2025 preprint; arXiv:2510.12974 (VERIFIED via search). Routes WHAT: per image-text instance, selects one vision encoder (instance-level routing), cutting compute 24-49% while beating brute-force multi-encoder aggregation.

### Axis 8 — Reasoning-effort / test-time-compute routing
- **Reasoning on a Budget: A Survey of Adaptive and Controllable Test-Time Compute in LLMs** — 2025; arXiv:2507.02076 (VERIFIED via search). Routes WHAT: thinking-token budget per query (L1 controllability under a fixed budget, L2 adaptiveness by input difficulty); documents over-thinking on easy problems and under-thinking on hard ones.
- OpenAI o-series "reasoning effort" (low/medium/high) and Claude "thinking budget" are productized instances. Routes WHAT: amount of chain-of-thought per prompt (context).

### Axis 9 — Mixture-of-experts / encoder routing
- **SCOPE (MoEnc)** (arXiv:2510.12974, VERIFIED via search). Routes WHAT: instance-level expert (vision encoder) selection, in contrast to token-level MoE.
- Classic sparse MoE (Shazeer et al. 2017) routes WHAT: tokens to experts (width routing). Included as context.

### Axis 10 — Tool-use / retrieval / agent-vs-no-agent routing
- **Dynamic Tool Routing (OptiRoute)** — per-task model/tool selection via lightweight task analysis (a quantized FLAN-T5). Routes WHAT: which tool/model per task under multi-objective (accuracy/latency/cost) constraints.
- **Router-R1: Teaching LLMs Multi-Round Routing and Aggregation via RL** — arXiv:2506.09033 (UNVERIFIED). Routes WHAT: multi-round routing and aggregation across a model pool via reinforcement learning.
- Retrieval routing and agent-vs-no-agent routing are emerging but thin in the agent literature; included for completeness.

---

## PART 2 — Deep Dive: Representation/Observation Routing for Web/GUI Agents

The key analytic distinction: **(a)** papers that ablate/compare representations as a *static design choice* vs. **(b)** papers that *dynamically route per-task/per-instance* among representations on a fixed model. Almost everything is (a) or within-representation pruning; (b) is essentially absent.

| Paper | Venue/Year | arXiv ID (verified status) | Representations routed among | Controlled comparison vs deployment default | How it differs from per-task DOM/SoM/Vision routing on a fixed agent |
|---|---|---|---|---|---|
| VisualWebArena | ACL 2024 | 2401.13649 (VERIFIED; true title is "…Realistic **Visual** Web Tasks") | AXTree+captions; screenshot+captions; SoM; HTML | Controlled comparison (systematic ablation across observation types on same models) | Static benchmark ablation; default is `image_som`. No per-instance routing — each agent config is fixed for the whole run. |
| SeeAct (GPT-4V is a Generalist Web Agent, if Grounded) | ICML 2024 | 2401.01614 (VERIFIED) | Screenshot + textual choices; SoM-style image annotation; HTML grounding | Controlled comparison of grounding/representation strategies | Compares grounding methods as static options: with oracle grounding GPT-4V completes 51.1% of tasks (vs GPT-4 13.3%, FLAN-T5 8.9%), but "set-of-mark prompting turns out to be not effective for web agents," and the best practical grounding still leaves a 20-30% gap to oracle. No dynamic routing. |
| WebVoyager | ACL 2024 | 2401.13919 (VERIFIED via search) | Screenshot+SoM (multimodal) vs text-only (AXTree) | Controlled comparison (multimodal 59.1% vs text-only 40.1% vs GPT-4 All-Tools 30.8%) | Static modality choice for the whole agent; not per-task representation routing. |
| Read More, Think More: Revisiting Observation Reduction for Web Agents | 2026 preprint | 2604.01535 (VERIFIED by subagent; future-dated, treat as unrefereed preprint) | HTML vs accessibility tree (and diff-based history) | Controlled comparison/ablation; derives a *guideline* | Closest motivation: shows the optimal representation depends on model capability + thinking-token budget (compact AXTree better for weaker models, HTML better for stronger models / larger budgets) and recommends "adaptively selecting observation representations" — but this is a static, offline design guideline chosen per model/budget, NOT a per-instance runtime router. |
| FocusAgent | 2025 (OpenReview) | 2510.03204 (VERIFIED) | Within AXTree only (line-level LLM retrieval/pruning) | N/A (within-representation reduction) | Prunes one representation (AXTree, reducing observation size >50%); does not select among DOM/SoM/screenshot. Not representation routing. |
| Less is More (SimpAgent) | ICCV 2025 | 2507.03730 (VERIFIED by subagent) | Within screenshot (visual-token / history compression) | N/A (within-representation reduction) | Compresses visual tokens/history on a fixed visual representation; no cross-representation routing. |
| iSHIFT | 2025 preprint | 2512.22009 (VERIFIED by subagent; future-dated) | Within screenshot: slow (fine-grained grounding) vs fast (global cues) perception modes | Dynamic per-step perception-EFFORT routing on a fixed model | Routes perception *depth/effort* within one modality (screenshots) via "perception tokens"; does NOT choose among HTML/DOM/AXTree/SoM. Architecturally analogous but a different axis. |
| Set-of-Mark Prompting | arXiv 2023 (used widely 2024-26) | 2310.11441 (VERIFIED) | Introduces SoM as a representation (marks overlaid on image) | N/A (proposes a representation) | Defines one representation; adopted as a static default by VWA/WebVoyager. Not a router. |
| ShowUI | CVPR 2025 | 2411.17465 (VERIFIED via search) | Within screenshot (UI-guided visual-token selection) | N/A (within-representation reduction) | Selects visual tokens inside a fixed screenshot representation; not cross-representation. |
| V-GEMS / "See and Remember" (nearest neighbor) | 2026 preprint | 2603.02626 (UNVERIFIED — reported by subagent, not independently confirmed; verify before citing) | DOM-text (LLM) ↔ screenshot/visual (VLM), per page | Dynamic per-instance switch (a "US Calculator" scores page content to pick text vs vision) | Closest to (b), BUT it is a dual-agent (Explorer/Critic) architecture with a binary DOM↔vision switch, not a single fixed model routing across the full HTML/DOM/AXTree/SoM spectrum. |
| RecAgent (Uncertainty-Aware GUI Agent) | 2025 preprint | 2508.04025 (UNVERIFIED — reported by subagent) | AXTree vs SoM (a11y+screenshot), plus component filtering | Two modalities on a fixed GPT-4o base + element selection | Filters/ranks UI elements (content pruning) rather than dynamically selecting the representation type per task. |
| WebRouter | 2025 preprint | 2510.11221 (UNVERIFIED — reported by subagent) | N/A — routes among LLM backbones, not representations | Model routing | Despite the name, routes among *models* (cost-performance via a Variational Information Bottleneck), not input representations. Contrast case only. |

### Synthesis of Part 2
- **Type (a) — static ablation/design choice:** VisualWebArena, SeeAct, WebVoyager, "Read More, Think More." These compare representations rigorously but fix the choice for an entire run/deployment.
- **Within-representation pruning (a different thing entirely):** FocusAgent, SimpAgent, ShowUI, GUI-KV, PruneVid. These trim a single representation; they do not select among representations.
- **Adaptive effort within a fixed modality:** iSHIFT, RecAgent. Architecturally similar to routing but operate on perception depth/content selection, not representation *type*.
- **Type (b) — dynamic per-instance representation routing on a fixed model:** essentially absent. The only approximation is V-GEMS (unverified preprint), and it is a binary DOM↔vision switch in a multi-agent system, not a single fixed agent routing across the full HTML/DOM/AXTree/SoM/screenshot space.

---

## PART 3 — Gap Statement (3 sentences)
There is no peer-reviewed, systematic characterization of per-task (or per-instance) input-representation routing for web agents on a single fixed model: the existing rigorous studies (VisualWebArena, SeeAct, WebVoyager, and the preprint "Read More, Think More") treat the choice among HTML, DOM, accessibility tree, Set-of-Marks, and screenshot as a static deployment default or an offline ablation/guideline rather than a runtime decision. The closest dynamic approaches either route among *models* (WebRouter, RouteLLM, CSCR), route perception *effort* within one fixed modality (iSHIFT, RecAgent), prune *within* a single representation (FocusAgent, SimpAgent, ShowUI), or — in the single nearest case (V-GEMS, an unverified 2026 preprint) — perform only a binary DOM↔vision switch inside a multi-agent architecture. Therefore, **per-task routing across DOM/SoM/vision (and "phantom" observation modes) on one fixed web agent is a genuine, under-studied gap**, well-motivated by "Read More, Think More" (which shows the optimal representation is instance- and budget-dependent) yet not implemented or systematically characterized by any current peer-reviewed work.

## Recommendations
1. **Frame the contribution as the first systematic study of type-(b) per-instance representation routing on a fixed web agent**, explicitly positioning against type-(a) ablations (VWA, SeeAct, "Read More, Think More") and within-representation pruning (FocusAgent, SimpAgent). Benchmark on VisualWebArena and WebArena (which already expose HTML/AXTree/SoM/screenshot natively) plus a GUI benchmark (AndroidWorld) for cross-domain validation.
2. **Cite V-GEMS as the nearest neighbor and clearly differentiate** (binary DOM↔vision, multi-agent) — but only after independently verifying arXiv:2603.02626, which is currently unconfirmed.
3. **Verify every future-dated arXiv ID** (2603.02626, 2604.01535, 2605.06350, 2601.03700, 2508.04025, 2510.11221) against the live abstract page before publication; mark any that cannot be confirmed as unverified.
4. **Threshold that would change the recommendation:** if a peer-reviewed paper is found that (i) keeps the model fixed, (ii) routes per-instance among ≥3 representation types, and (iii) reports a cost-accuracy frontier, then the "gap" claim must be downgraded to "under-explored" and that paper cited as direct prior art.

## Caveats
- Several IDs are **future-dated 2026 preprints** flagged by search/subagent; they appear live on arXiv but are unrefereed, and a few (V-GEMS 2603.02626, RecAgent 2508.04025, WebRouter 2510.11221, Hybrid LLM arXiv id, Router-R1 2506.09033, "Is Escalation" 2605.06350, ADEPT 2601.03700, Medusa 2401.10774) were **not independently verified by me** and are marked UNVERIFIED. The core web-agent papers (VisualWebArena 2401.13649, SeeAct 2401.01614, WebVoyager 2401.13919, Set-of-Mark 2310.11441, FocusAgent 2510.03204, SimpAgent/Less-is-More 2507.03730, iSHIFT 2512.22009, Read-More-Think-More 2604.01535, ShowUI 2411.17465) and the general-LLM anchors (RouteLLM 2406.18665, FrugalGPT 2305.05176, CSCR 2508.12491, LLMLingua 2310.05736 / 2310.06839 / 2403.12968) are VERIFIED.
- The VisualWebArena title is correctly "Evaluating Multimodal Agents on Realistic **Visual** Web Tasks" (not "Visually Grounded Web Tasks").
- Much commentary on accessibility-tree vs vision tradeoffs comes from **industry blogs** (Playwright MCP, OpenAI Atlas, web.dev), which are useful context but not peer-reviewed; the CHI 2026 UC Berkeley/Michigan accessibility study (task success reportedly dropping 78%→42% on degraded accessibility trees) is cited secondhand and should be verified against the primary paper.
- "Routes WHAT" lines summarize each method's routing target; several general-LLM entries (Axes 3-6, 8-9) are context only, per the task's instruction to prioritize web/GUI/multimodal agent work.