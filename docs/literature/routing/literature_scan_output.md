(1) TAXONOMY TABLE

| Paper | Year | Domain | Core Technique | Efficiency Metric | Routes WHAT |
|---|---|---|---|---|---|
| **WebRouter** [1, 2, 3] | 2025 | Web Agents | Cost-aware Variational Information Bottleneck (ca-VIB) | 87.8% Cost Reduction | Compressed prompt representations |
| **MTRouter** [4, 5] | 2026 | Multi-turn Agents | History-Model Joint Embeddings | 58.7% Cost Reduction | Sequential model selection |
| **UIFormer** [6, 7] | 2025 | GUI/Web Agents | DSL-based UI Transformation | 48.7%–55.8% Token Reduction | UI representations (DOM/Trees) for token efficiency |
| **Surfer-H** [8] | 2025 | Web Agents | Specialized Open-Weight VLMs (Holo1) | Pareto-optimal Efficiency | Modular policy/localizer/validator pipeline |
| **Topaz** [9, 10] | 2026 | Agentic Workflows | Skill-based Profiling & Budgeting | Auditable Trade-offs | Routing decisions based on budget and skill-match |
| **CASTER** [11] | 2026 | Multi-Agent (MAS) | Dual-Signal Router (Semantic + Structural) | Task-efficient Orchestration | Sub-tasks in graph-based workflows |
| **ECCOS** [12] | 2025 | Multi-LLM Serving | Two-stage Scheduling (Predictor + Optimizer) | 10.1% Cost / 6.3% Success | Heterogeneous model pools |
| **Route-To-Reason** | 2025 | Reasoning/Web | Joint Model-Strategy Selection | 60% Cost Reduction | Optimal LLM and reasoning strategy per query |
| **FrugalGPT** | 2023 | General LLM | LLM Cascade & Routing | Cost-Performance Frontier | Model cascades for query escalation |

(2) GAP VERDICT

**NO**, there is no peer-reviewed, systematic characterization of per-task input-representation routing for web agents on a single fixed model, where the agent dynamically selects among DOM / HTML / accessibility-tree / screenshot / Set-of-Marks serializations of the same page and reports a cost-accuracy frontier.

**Closest Works and Why They Fall Short:**

*   **UIFormer** [6, 7]: This work focuses on optimizing UI representations for token efficiency by synthesizing programs to transform DOM/Trees. While it deals with UI representations and efficiency, it does not involve *dynamic, per-task selection* among *multiple, distinct input representations* (like DOM vs. screenshot) of the *same page* for a *single fixed model*. Instead, it transforms a single representation type.
*   **WebRouter** [1, 2, 3]: This paper aims for cost reduction through compressed prompt representations. It addresses cost efficiency for web agents, but its focus is on compressing the *prompt*, not dynamically routing between different *input representations of the webpage itself* (e.g., DOM vs. screenshot) for a single model.
*   **Surfer-H** [8]: This work uses specialized open-weight VLMs and a modular pipeline for web agents to achieve Pareto-optimal efficiency. While it touches upon efficiency in web agents and might implicitly use different modalities, it does not systematically characterize dynamic, per-task routing between explicitly defined input representations (DOM, screenshot, etc.) of the same page with a single fixed model to explore a cost-accuracy frontier. It uses specialized VLMs rather than routing representations to a single, fixed model.

(3) arXiv VERIFICATION TABLE

| arXiv ID | Real Title or NOT FOUND / cannot confirm | Confidence |
|---|---|---|
| 2603.02626 | NOT FOUND / cannot confirm | High |
| 2604.01535 | NOT FOUND / cannot confirm | High |
| 2401.01614 | NOT FOUND / cannot confirm | High |
| 2508.04412 | NOT FOUND / cannot confirm | High |
| 2605.00551 | NOT FOUND / cannot confirm | High |
