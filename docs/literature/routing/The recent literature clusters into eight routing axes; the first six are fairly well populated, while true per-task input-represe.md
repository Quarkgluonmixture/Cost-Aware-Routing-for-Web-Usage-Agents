The recent literature clusters into eight routing axes; the first six are fairly well populated, while true per-task input-representation routing for fixed web agents is still thin.

| Axis | Routes what | Representative papers |
|---|---|---|
| Model routing / selection across a pool | Chooses the cheapest acceptable whole-model answerer for the query. [^1][^2][^3][^4] | RouteLLM (2024, arXiv:2406.18665); Hybrid LLM (2024, arXiv:2404.14618); CSCR (2025, arXiv:2508.12491); LightRouter (2025, arXiv:2505.16221) |
| Cascade / model+strategy routing | Chooses when to escalate and, in some systems, which reasoning strategy to pair with the model. [^5][^6][^7] | A Unified Approach to Routing and Cascading for LLMs (ICML 2024, arXiv:2410.10347); Route-To-Reason / RTR (2025, arXiv:2505.19435); EMAFusion (2025, arXiv:2504.10681); Adaptive Reasoning Executor (2025, arXiv:2510.13214) |
| Early-exit / halting / adaptive depth | Chooses how deep the same model should compute per token or example. [^8][^9] | CALM (NeurIPS 2022, arXiv:2207.07061); EESD (ACL 2024, arXiv:2406.03853); Dr.LLM (2025, arXiv:2510.12773); AdaPonderLM (2025/2026, arXiv:2509.24238) |
| Speculative / draft-verify decoding | Chooses which draft tokens to propose and how much verification the target model does. [^10] | Speculative Decoding via Early-exiting (ACL 2024, arXiv:2406.03853); SpecRouter (2025, arXiv:2505.07680); Fast and Cost-effective Speculative Edge-Cloud Decoding with Early Exits (TMLR 2025, arXiv:2505.21594) |
| Token / visual-token pruning + KV-cache compression | Chooses which tokens, frames, experts, or cache entries survive.  | FastMMoE (2025, arXiv:2511.17885); GUI-KV (2025, arXiv:2510.00536); KVCrush (2025, arXiv:2503.00022); KVReviver (2025, arXiv:2512.17917); RAP (2025, arXiv:2505.17138) |
| Prompt / context compression | Chooses which observation lines or history to keep in the prompt. [^11][^12] | FocusAgent (2025, arXiv:2510.03204); LineRetriever (2025, arXiv:2507.00210); Active Context Compression (2026, arXiv:2601.07190) |
| Modality selection | In the web-agent papers I found, this is mostly a fixed design choice, not a learned per-task router: screenshot-only, no-HTML, or text-plus-image systems. [^13][^14][^15][^16] | OmniParser (2024, arXiv:2408.00203); WebSight (2025, arXiv:2508.16987); MolmoWeb (2026, arXiv not surfaced in this pass) |
| Input / observation / representation routing | Chooses the serialization of the same state: HTML vs DOM vs AxTree vs grounded GUI snapshot vs SoM-style visual prompting. [^13][^14][^15][^16] | Read More, Think More (2026, preprint; arXiv not surfaced); D2Snap (2025, arXiv:2508.04412); FocusAgent (2025, arXiv:2510.03204); LineRetriever (2025, arXiv:2507.00210); OmniParser (2024, arXiv:2408.00203) |

| Paper | Which representations it routes among | Controlled comparison or deployment default? | What is it, relative to per-task routing on one fixed web agent? |
|---|---|---|---|
| Read More, Think More | HTML vs accessibility tree, with the choice conditioned on model capability and thinking-token budget. [^13] | Controlled comparison and recommendation. | This is the closest thing I found to a representation-aware policy, but it is still framed as an empirical rule of thumb, not a learned per-task router over DOM/AxTree/SoM/screenshot modes. |
| D2Snap | DOM snapshot downsampling versus a grounded GUI snapshot baseline.  | Controlled comparison. | It compresses one chosen representation rather than deciding which representation to use for each task. |
| FocusAgent | AxTree lines, keeping only task-relevant lines. [^11] | Controlled retriever/pruner. | This is within-representation pruning, not observation-mode selection. |
| LineRetriever | DOM/AxTree lines most relevant to future navigation steps. [^12] | Controlled retriever. | Planning-aware reduction, but still not a policy that switches among DOM, AxTree, screenshot, or SoM. |
| OmniParser | Screenshot-only input, compared against GPT-4V baselines that use extra information. [^14] | Deployment default with comparative baselines. | It fixes the visual channel and improves parsing; it does not choose among alternative input serializations per task. |
| WebSight / MolmoWeb | Screenshot-only, with no HTML, accessibility-tree, or DOM dependency. [^15][^16] | Deployment default. | These are fixed-modality designs; SoM appears mostly as a benchmark representation in comparisons, not as a routed choice. |

I did not find a peer-reviewed systematic characterization of per-task input-representation routing for fixed web agents. The closest papers are point solutions or benchmark comparisons: Read More, Think More argues that compact accessibility trees suit weaker models while detailed HTML helps stronger ones, and D2Snap, FocusAgent, and LineRetriever all prune or downsample within a chosen representation rather than learning which representation to use. [^13][^11][^12] So the precise gap is a learned or formally evaluated policy that, for the same web agent, chooses among DOM, accessibility tree, screenshot, Set-of-Marks, or hybrid serializations on a per-task basis and reports the cost-accuracy frontier. [^14][^15][^16]

This is an initial scan, so a deeper pass could still surface older accessibility work or differently named representation-routing papers that did not rank highly in these searches.

[^1]: Ong et al., 2024. RouteLLM: Learning to Route LLMs with Preference Data. arXiv.org.

[^2]: Ding et al., 2024. Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. International Conference on Learning Representations.

[^3]: Shirkavand et al., 2025. Cost-Aware Contrastive Routing for LLMs. arXiv.org.

[^4]: Zhang et al., 2025. LightRouter: Towards Efficient LLM Collaboration with Minimal Overhead. arXiv.org.

[^5]: Dekoninck et al., 2024. A Unified Approach to Routing and Cascading for LLMs. International Conference on Machine Learning.

[^6]: Pan et al., 2026. Adaptive Model and Strategy Routing for Cost-Efficient LLM Services. Proceedings of the ACM Web Conference 2026.

[^7]: Shah et al., 2025. EMAFusion: A Self-Optimizing System for Seamless LLM Selection and Integration. arXiv.org.

[^8]: Schuster et al., 2022. Confident Adaptive Language Modeling. Neural Information Processing Systems.

[^9]: Heakl et al., 2025. Dr.LLM: Dynamic Layer Routing in LLMs. arXiv.org.

[^10]: Liu et al., 2024. Speculative Decoding via Early-exiting for Faster LLM Inference with Thompson Sampling Control Mechanism. Annual Meeting of the Association for Computational Linguistics.

[^11]: Kerboua et al., 2025. FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents. arXiv.org.

[^12]: Kerboua et al., 2025. LineRetriever: Planning-Aware Observation Reduction for Web Agents. arXiv.org.

[^13]: Enomoto et al., 2026. Read More, Think More: Revisiting Observation Reduction for Web Agents.

[^14]: Lu et al., 2024. OmniParser for Pure Vision Based GUI Agent. arXiv.org.

[^15]: Gupta et al., 2026. MolmoWeb: Open Visual Web Agent and Open Data for the Open Web.

[^16]: Bhathal & Gupta, 2025. WebSight: A Vision-First Architecture for Robust Web Agents. arXiv.org.