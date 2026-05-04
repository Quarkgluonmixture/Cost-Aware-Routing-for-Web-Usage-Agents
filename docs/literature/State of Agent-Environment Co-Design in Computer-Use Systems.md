# State of Agent-Environment Co-Design in Computer-Use Systems

## A. Per-System Characterization

The landscape of agent-environment co-design in late 2025 and early 2026 reveals a profound architectural schism. Developers are fiercely divided on the optimal interface between language models and graphical environments, pivoting between pure visual perception, deep structural metadata parsing, and proactively synthesized environmental affordances. The following analysis characterizes the frontier of these systems across fifteen critical implementations, evaluating their modality routing, environmental dependence, issue handling mechanisms, standards participation, and knowledge injection strategies.

### The Chinese and Asian Ecosystem

The Asian ecosystem has demonstrated a distinct preference for system-level integration and zero-shot visual perception, often bypassing traditional browser sandboxes to achieve deep operating system control.

| **System**                    | **Modality Routing**               | **Affordance Dependence**            | **Issue Handling**                    | **Standards**     | **Knowledge / Prior Injection**     | **Confidence** |
| ----------------------------- | ---------------------------------- | ------------------------------------ | ------------------------------------- | ----------------- | ----------------------------------- | -------------- |
| **UI-TARS / 1.5** (ByteDance) | Visual-only (coordinate grounding) | None (pure screenshot)               | Learned policy (System-2 reflection)  | None              | Iterative online traces             |                |
| **Doubao Mobile** (ByteDance) | Dynamic Hybrid (Visual + XML)      | High (Android OS `INJECT_EVENTS`)    | Hardcoded limits (app blacklisting)   | None              | On-device "Screen Memory"           |                |
| **Manus / OpenManus**         | Hybrid (DOM + Visual capture)      | High (Playwright/Browser metadata)   | Hardcoded state-machine fallbacks     | Independent API   | Structured `PlanningFlow` prompts   |                |
| **AutoGLM-Web** (Zhipu)       | Hybrid (GLM-4.5V + Parsed DOM)     | Medium (HTML simplification algo)    | Learned policy / Human takeover       | Regional specific | Hybrid Human-AI curriculum          |                |
| **AppAgent-v2** (Tencent)     | Dual-modality (XML + Screenshot)   | High (Android CLI XML extraction)    | Segmented (Exploration vs Deployment) | Closed-loop       | RAG-based offline documentation     |                |
| **CogAgent** (Zhipu)          | Visual-only (GUI VLM)              | None (pure visual pixel coordinates) | VLM zero-shot generalization          | None              | Large-scale offline GUI pretraining |                |
| **OS-Atlas/Genesis**          | Dynamic Triplet (Vision + Action)  | High (Synthetic GUI generation)      | Avoidance via synthetic training      | None              | Reverse task synthesis instructions |                |



The release of UI-TARS and its subsequent 1.5 iteration by ByteDance in early 2025 established a formidable baseline for pure visual routing. Operating entirely without Document Object Model (DOM) or accessibility (A11y) tree metadata, UI-TARS perceives raw human-facing screenshots and grounds its actions directly into coordinate spaces. The decision logic at inference relies on a native GUI agent architecture that standardizes actions across platforms. When confronting environmental issues such as unmapped cookie banners or dynamic popups, UI-TARS 1.5 eschews hardcoded patches. Instead, it relies on a learned policy fortified by System-2 reasoning mechanisms, performing task decomposition and milestone recognition before committing to an action. This model does not participate in external semantic web standards, deriving its priors purely from iterative training against reflective online traces gathered from hundreds of virtual machines.

ByteDance extended this philosophy to mobile hardware with the Doubao agentic smartphone, released in December 2025. Unlike UI-TARS, the Doubao agent routes decisions through a dynamic hybrid space, fusing real-time visual comprehension with the Android XML view hierarchy. Its environment-side affordance dependence is absolute; it requires system-level `INJECT_EVENTS` permissions to simulate physical taps and bypass standard application programming interfaces (APIs). Because it operates across all applications autonomously, the agent triggered massive ecosystem friction. E-commerce and messaging platforms detected its continuous interaction loops as abnormal, resulting in account bans. ByteDance’s issue-handling response was structural rather than algorithmic: they instituted hardcoded limits, permanently disabling the agent’s integration with financial applications to prevent fraud alerts. Prior knowledge is injected via an on-device "Screen Memory" module that passively archives user interaction history, bypassing the need for standardized agent-to-agent communication protocols.

The release of Manus, and its rapid open-source replication as OpenManus in March 2025, highlighted the demand for structured execution scaffolding. OpenManus employs a hybrid routing architecture structured across a strict inheritance hierarchy (BaseAgent, ReActAgent, ToolCallAgent). Its decision logic constantly shifts between textual DOM parsing and visual Playwright captures. Consequently, it is highly dependent on browser-emitted metadata. When the environment fails to respond, OpenManus triggers hardcoded state-machine interventions, utilizing internal monitoring methods like `is_stuck` to break infinite loops. While it does not natively enforce external protocols like the Model Context Protocol (MCP), its flexible tool system allows for straightforward integration. Knowledge injection occurs strictly through a dual-execution scaffolding system, relying heavily on a `PlanningFlow` module that parses a complex system prompt to dictate the execution graph.

Zhipu’s AutoGLM-Web integrates the GLM-4.5V foundation model into a pipeline that relies on an HTML simplification algorithm. This agent-pipeline preprocessing script synthesizes an agent-readable view from existing UI metadata, strategically discarding visual noise. Decision logic is dynamically routed between the visual capabilities of the VLM and the simplified DOM. Issue handling relies on reinforcement learning and rejection sampling to navigate basic web obstacles, but for high-security blockers like CAPTCHAs, the model defaults to explicit human delegation. Training priors are injected through a curriculum that combines human demonstration with AI-driven exploration. Similarly, Tencent's AppAgent-v2 utilizes a dual-modality XML and screenshot routing pipeline for Android environments. However, its issue-handling strategy is distinctly segmented. By splitting operation into an "exploration phase" and a "deployment phase," the agent maps the functionality of unfamiliar interfaces offline. It then injects this acquired knowledge dynamically as a prior using Retrieval-Augmented Generation (RAG) during live execution, drastically reducing failure rates when encountering dynamic application states.

Further reinforcing the synthetic training paradigm, the OS-Atlas and OS-Genesis frameworks generate vast amounts of offline prior knowledge through reverse task synthesis. The OS-Genesis environment systematically traverses dynamic GUI environments using rule-based algorithms to discover functional interaction triples (pre-state, action, post-state). OS-Atlas then utilizes these synthesized environments to fine-tune its foundation action model, completely sidestepping the need for the agent to navigate real-world popups during its training phase. CogAgent represents an earlier iteration of the pure GUI VLM philosophy within the Chinese ecosystem, relying entirely on visual language modeling to predict actionable elements, setting the architectural foundation for subsequent zero-shot models.

### Environment-Side Affordance Synthesis Models

Rather than building end-to-end agents, several prominent organizations have focused on synthesizing intermediary affordances, translating raw graphical environments into tokenized, agent-readable formats.

| **System**                    | **Modality Routing**                       | **Affordance Dependence**           | **Issue Handling**          | **Standards**          | **Knowledge / Prior Injection**     | **Confidence** |
| ----------------------------- | ------------------------------------------ | ----------------------------------- | --------------------------- | ---------------------- | ----------------------------------- | -------------- |
| **OmniParser-v2** (Microsoft) | Preprocessing pipeline (Screenshot → Text) | Zero DOM reliance (Pure OCR/Vision) | Delegated to downstream LLM | MCP compliant endpoint | Fine-tuned icon functional captions |                |
| **Magma** (Microsoft)         | Hybrid Internal (Visual/Spatial/Temporal)  | None (Internal tokenization)        | Predictive state planning   | Foundation model layer | SoM / ToM pretraining objectives    |                |
| **ShowUI**                    | Vision-Language-Action                     | None (Visual-centric)               | Learned spatial policy      | Academic framework     | GUI trajectory fine-tuning          |                |
| **AGUVIS**                    | Unified GUI VLM                            | None (Visual-centric)               | Learned policy              | Academic framework     | Offline demonstration datasets      |                |
| **WebGUM / Spotlight**        | Multi-modal DOM/Vision                     | High (DOM extraction)               | Hardcoded heuristics        | Pre-standards era      | Early web interaction datasets      |                |



Microsoft’s OmniParser and the subsequent OmniParser-v2 (February 2025) represent the pinnacle of agent-pipeline preprocessing. The system acts as a modality translator, accepting raw human-facing UIs and utilizing a compact Mixture-of-Experts (MoE) transformer decoder to output a structured list of textual element tokens. It maintains zero dependence on server-side DOM or ARIA properties, relying entirely on internal optical character recognition and localized object detection models. Because OmniParser functions strictly as a parser, dynamic issue handling is delegated to the downstream language model utilizing its output. The system natively supports standardization, operating seamlessly within the OmniTool dockerized environment, which exposes the parser as an MCP-compliant endpoint. Priors are injected heavily into the parser's weights via specialized datasets covering interactable icon detection and functional captioning.

A fundamentally different approach to affordance synthesis is evident in Microsoft's Magma foundation model. Rather than pre-processing the image into text for a separate model, Magma internalizes the substitution gradient entirely. It operates across digital and physical domains, making it independent of any software-specific structural metadata. Issue handling is managed through an advanced understanding of temporal video dynamics; the model anticipates future environmental states before acting, allowing it to circumvent blockers proactively. Magma’s defining innovation lies in its knowledge injection strategy during pretraining. It utilizes Set-of-Mark (SoM) annotations to ground static visual elements, and Trace-of-Mark (ToM) annotations to capture dynamic object movements over time, creating a robust spatial-temporal prior integrated directly into the model's weights.

Academic models such as ShowUI and AGUVIS follow similar trajectories, relying on unified Vision-Language-Action architectures that eschew textual metadata in favor of direct visual processing. These models encode GUI trajectory priors directly into their fine-tuning pipelines. They stand in stark contrast to earlier models like Google's WebGUM and Spotlight, which heavily relied on extracting DOM structures and utilized hardcoded heuristics to patch over the inevitable gaps between the visual render and the underlying code.

### Industry Browser Platform Shifts

The commercial realization that standard web browsers are hostile environments for autonomous agents led to the development of specialized agent-native browsing infrastructure in late 2025.

| **System**                            | **Modality Routing**             | **Affordance Dependence**           | **Issue Handling**               | **Standards**            | **Knowledge / Prior Injection**    | **Confidence**           |
| ------------------------------------- | -------------------------------- | ----------------------------------- | -------------------------------- | ------------------------ | ---------------------------------- | ------------------------ |
| **ChatGPT Atlas** (OpenAI)            | Dynamic Hybrid (OWL Compositing) | Very High (Custom Chromium Service) | Ephemeral isolated sessions      | MCP integration          | Workspace cross-tab memory         |                          |
| **Browser-Use SDK** (Stagehand, etc.) | DOM-centric textual substitution | Extreme (Playwright/Puppeteer)      | Extensive hardcoded DOM patching | A2A / MCP rapid adoption | User-defined YAML execution graphs | [Inferred from behavior] |



OpenAI’s ChatGPT Atlas, launched in October 2025, reimagined the foundational architecture of the web browser to accommodate agentic workflows. The modality routing relies on OpenAI's Web Layer (OWL), a custom pipeline that separates the Chromium browser process from the main native application. Atlas engages in intense environment-side adaptation by actively compositing elements that normally render in separate windows (such as out-of-bounds `<select>` dropdowns) back into the main page image at the correct coordinates, creating a highly specific agent-readable viewport. Issue handling is strictly architectural; rather than attempting to navigate endless cookie consent forms, Atlas spawns isolated, ephemeral Chromium `StoragePartition` sessions. These sessions execute the agent's tasks and then immediately self-destruct, preventing localized state pollution or cross-session tracking. It participates heavily in the MCP ecosystem, acting as a workspace where contextual priors from the user's cross-tab history are automatically injected into the agent's prompt.

In parallel, the commercial SDK layer—represented by frameworks like Stagehand, Skyvern, and MultiOn—has focused on scaling agent-pipeline preprocessing for enterprise use. These systems rely almost exclusively on DOM-centric textual substitution, maintaining an extreme dependence on Playwright and Puppeteer instrumentation. Because they rely on the DOM, these SDKs must employ extensive hardcoded patching mechanisms, utilizing predefined scripts to forcefully dismiss cookie banners and modal popups before allowing the reasoning agent to observe the page state. They are heavy participants in the emerging standards ecosystem, rapidly adopting both A2A and MCP to standardize tool execution. Knowledge injection is typically handled externally via user-defined YAML or JSON execution graphs mapped to explicit system prompts.

### Major Industry Players

The dominant foundation models have iteratively expanded their computer-use capabilities, converging on highly monitored, visually driven execution frameworks.

| **System**                       | **Modality Routing**                 | **Affordance Dependence**        | **Issue Handling**                   | **Standards**      | **Knowledge / Prior Injection**  | **Confidence** |
| -------------------------------- | ------------------------------------ | -------------------------------- | ------------------------------------ | ------------------ | -------------------------------- | -------------- |
| **Claude Sonnet 4.6**            | Dynamic visual + coordinate tracking | Minimal (Virtual I/O execution)  | Human-level prompt injection evasion | Core MCP architect | Context compaction algorithms    |                |
| **Operator + CUA** (OpenAI)      | Visual-only bounding boxes           | Minimal (Raw OS event injection) | Hardcoded "Takeover mode"            | MCP ecosystem      | RL against GUI trajectories      |                |
| **Project Mariner / Gemini 2.5** | Multimodal code/vision fusion        | VM Sandbox dependent             | Pre-execution safety checkpoints     | Core A2A architect | "Teach and repeat" memory traces |                |



Anthropic’s release of Claude Sonnet 4.6 in February 2026 introduced substantial upgrades to its computer-use paradigm. Modality routing relies on dynamic visual processing coupled with precise coordinate tracking, allowing the model to utilize a virtual mouse and keyboard to interact natively with operating systems. The architecture minimizes affordance dependence, avoiding proprietary API integrations in favor of raw pixel mapping. Notably, Sonnet 4.6 handles environmental issues—specifically malicious prompt injections hidden within website UIs—with near human-level resilience, an explicit upgrade over the 4.5 architecture. Anthropic remains the primary architect of the MCP standard, deeply embedding the protocol into its Excel and API ecosystems. To navigate long-horizon tasks without exceeding context windows, Sonnet 4.6 injects dense priors via a beta "context compaction" algorithm, summarizing its own historical trajectory on the fly.

OpenAI’s Operator, powered by the Computer-Using Agent (CUA) model, routes actions entirely through visual screenshot analysis combined with reinforcement learning reasoning. Like Sonnet 4.6, Operator minimizes affordance dependence, simulating human keystrokes directly. However, its issue handling diverges significantly; OpenAI enforces a strict "Takeover mode". When the agent encounters a cryptographic barrier, an ambiguous payment form, or a CAPTCHA, it halts execution and delegates the task to the human user. While OpenAI supports MCP at the enterprise tier, Operator functions largely as an enclosed application. The CUA model’s priors are injected directly into its weights via extensive offline reinforcement learning focused on multi-step GUI navigation.

Google’s Project Mariner, operating atop the Gemini 2.5 Computer Use API (late 2025), functions as a heavily sandboxed multimodal execution environment. The model fuses code parsing with visual bounding boxes, maintaining high dependence on secure Virtual Machine (VM) infrastructure. Issue handling is strictly monitored; an internal safety layer reviews every generated action coordinate before execution, prompting the user for confirmation if the predicted action carries high financial or operational risk. As a central architect of the Agent-to-Agent (A2A) protocol, Google has optimized Mariner for seamless handoffs across enterprise boundaries. Furthermore, Mariner injects behavioral priors through a localized "teach and repeat" memory module, allowing users to execute a workflow once, which the system then translates into a persistent execution prior for future automated runs.

## B. Aggregate Findings: Architectural Distribution and Ecosystem Divergence

The structural analysis of the aforementioned systems reveals a highly uneven distribution of responsibilities across the execution pipeline. The field has fracture into three distinct methodologies for bridging the gap between agent reasoning and environment state.

First, the **agent-side adaptation** paradigm places the entire burden of environmental comprehension on the foundation model. Systems like CogAgent, UI-TARS, and Magma represent the purest form of this approach. They ingest raw, human-facing UI pixels and utilize internal neural pathways to deduce actionable coordinates. This approach is highly robust against structural deception (e.g., malformed HTML) but demands massive compute overhead for every execution step. Second, the **agent-pipeline preprocessing** paradigm relies on intermediary algorithms to synthesize a clean, agent-readable state. Microsoft’s OmniParser and the vast array of Playwright-based SDKs (Stagehand, Skyvern) exist here. They relieve the LLM of the visual processing burden by converting pixels or raw DOM into structured text tokens, acting as a crucial bridge for models that lack native, high-resolution spatial reasoning. Third, the **environment-side adaptation** paradigm forces the environment to mutate for the agent. The ChatGPT Atlas OWL architecture fundamentally alters how a browser composites its UI to make it more digestible for the underlying AI , while protocols like NLWeb bypass the visual and structural DOM entirely, demanding that the server explicitly expose its capabilities via agent-specific JSON channels.

This technical distribution heavily mirrors a geographic and regulatory divergence between the Asian and Western ecosystems. The **Chinese ecosystem** (typified by ByteDance, Tencent, and Zhipu) has aggressively optimized for autonomous, cross-application fluidity via deep operating system integration. The Doubao mobile agent exemplifies this philosophy; by demanding core Android `INJECT_EVENTS` permissions, it forcefully supersedes individual application boundaries to execute end-to-end tasks (e.g., pulling data from a chat app to book a ticket in a travel app). This OS-level, zero-shot visual approach is highly effective for local users but creates massive friction with commercial platforms. The widespread banning of Doubao by major Chinese e-commerce and payment platforms underscores the fundamental hostility of commercial environments toward unregulated, continuous autonomous execution.

Conversely, the **Western ecosystem** (OpenAI, Google, Microsoft, Anthropic) has prioritized highly sandboxed, legally compliant, and modular infrastructures. Western agents are overwhelmingly constrained to isolated browser environments (ChatGPT Atlas) or heavily monitored virtual machines (Gemini 2.5 Mariner). Instead of forcing agents into hostile, closed native applications, Western developers have aggressively focused on standardizing the API layer, allowing disparate enterprise agents to communicate securely without needing to "click" on each other's graphical interfaces. This compliance-first approach heavily utilizes human-in-the-loop "Takeover modes" to mitigate liability during sensitive transactions , reflecting a fundamentally more cautious deployment strategy compared to the highly integrated, fully autonomous execution targeted by their Asian counterparts.

## C. Standards Proposals Timeline (2024–2026)

The explosion of multi-vendor agents necessitated a rapid standardization of interoperability protocols. Without shared networking infrastructures, agents operating in isolated pipelines could not hand off context or execute multi-platform workflows. The period between late 2024 and early 2026 witnessed a consolidation of the semantic web into agent-specific standards.

| **Date**     | **Protocol / Standard**          | **Substantive Development and Impact**                       | **Source** |
| ------------ | -------------------------------- | ------------------------------------------------------------ | ---------- |
| **Nov 2024** | **Model Context Protocol (MCP)** | Anthropic introduces MCP to standardize the connection between AI systems and external data endpoints. It utilizes JSON-RPC 2.0 over stdio/SSE to establish local and remote tool execution capabilities. |            |
| **Mar 2025** | **MCP Security Overhaul**        | As OpenAI adopts MCP natively, the protocol integrates OAuth 2.1 authorization and Proof Key for Code Exchange (PKCE) to secure client-server token generation against hijacking. |            |
| **Apr 2025** | **Agent-to-Agent (A2A)**         | Google launches A2A at Cloud Next. Where MCP handles vertical agent-to-tool connections, A2A handles horizontal agent-to-agent negotiation, standardizing stateful Task objects and metadata "Agent Cards." |            |
| **May 2025** | **Natural Language Web (NLWeb)** | Microsoft (via R.V. Guha) introduces NLWeb. It leverages existing `Schema.org` formatting to allow web properties to natively emit conversational, agent-readable JSON, bypassing the DOM entirely. |            |
| **Jun 2025** | **Linux Foundation Governance**  | A2A and MCP are jointly donated to the Agentic AI Foundation. To mitigate server-side vulnerabilities, MCP clients are required to implement Resource Indicators (RFC 8707). |            |
| **Feb 2026** | **AGENTS.md / NLAH Research**    | Academic preprints evaluate Natural Language Agent Harnesses (NLAH), highlighting severe unpatched vulnerabilities in MCP, specifically a 100% success rate for namespace collision tool spoofing. |            |



The rapid convergence on MCP and A2A fundamentally shifted the trajectory of environment co-design. By standardizing how agents expose their capabilities (`.well-known/agent.json`), these protocols diminish the need for agents to visually parse complex enterprise GUIs. When an NLWeb endpoint automatically acts as an MCP server , an agent can query the backend database directly using natural language, rendering the physical layout of the website obsolete for automated tasks. However, as noted in the February 2026 literature surrounding AGENTS.md, the rush to standardize text-based execution harnesses outpaced formal security verification, leaving protocols highly vulnerable to shadow-tool spoofing and namespace collisions.

## D. Counter-Evidence: The Failure of Structural Metadata

Throughout early agent development, a pervasive assumption dictated that robust structural metadata—specifically the Document Object Model (DOM) and the Accessibility (A11y) Tree—would provide a reliable, programmatic map for autonomous navigation. Theoretical frameworks assumed that if developers strictly adhered to ARIA (Accessible Rich Internet Applications) guidelines, web agents could bypass complex visual processing and operate securely via textual substitution. However, empirical studies and diagnostic benchmarks from late 2025 and early 2026 provide overwhelming counter-evidence, revealing a catastrophic breakdown in this paradigm. Models relying purely on structural extraction suffer from severe failure rates when confronted with the dynamic, adversarial, and frequently non-compliant reality of the modern commercial internet.

The primary mechanism driving this failure is the severe degradation of semantic web standards in production environments. The February 2026 WebAIM Million report, which utilized the WAVE API to conduct a massive structural analysis of the internet's top one million domains, uncovered a highly counterintuitive phenomenon: web pages attempting to utilize ARIA metadata actually contained significantly *more* structural errors than those that ignored it entirely. Specifically, pages with ARIA implementation exhibited an average of 57 critical errors, while pages without ARIA averaged only 27 errors. When an agent pipeline ingests the A11y tree to construct an environment representation, it is inherently processing corrupted, mislabeled, or deceptive data. For example, a web developer might implement an element as `<div role="button">` to satisfy a rudimentary accessibility checker, but fail to attach the corresponding keyboard activation triggers (Enter/Space) in the underlying JavaScript. To a text-parsing agent, the element appears semantically valid and highly actionable. However, any execution command dispatched to that structural node vanishes into a dead state, prompting the agent to enter an infinite loop of failed activation attempts. Accessibility engineers explicitly note that "no ARIA is better than bad ARIA," precisely because malformed structural tags override standard interactive behaviors, creating invisible traps for non-visual navigators.

This structural deception translates directly into massive inefficiency loops during rigorous benchmarking. Granular analysis of agent trajectories within the WebSuite diagnostic benchmark and the OSWorld-Verified suite demonstrates that structural reliance cripples execution speed. When tracking models within the Browser Use framework, researchers discovered that failed tasks consumed nearly twice as many execution steps as successful ones. When an agent utilizing a purely textual perspective encounters an unmapped pop-up window, a dynamic cookie consent overlay, or a pagination error, it loses spatial context. Because the visual overlay (often managed via CSS z-index) obscures the interactive elements below it, the DOM representation remains largely unchanged. The agent, oblivious to the visual blocker, continuously fires action requests against background elements that are technically present in the DOM but physically unclickable on the render surface, resulting in complete task failure.

Furthermore, deliberate environmental hostility presents an insurmountable barrier for non-visual architectures. Security protocols such as CAPTCHA are explicitly designed to measure "CAPTCHA Reasoning Depth," a metric quantifying the cognitive and fine motor steps required to compress complex visual heuristics into fluid spatial actions. Modern CAPTCHA implementations deliberately bypass structural DOM triggers, rendering HTML-parsing agents functionally blind. The cognitive gap between human intuition—which can process a chaotic grid of images instantly—and an agent's brittle, sequential textual deliberation causes state-of-the-art models to fail these dynamic security challenges at near-absolute rates. Behavioral testing confirms that up to 16% of e-commerce platforms actively deploy these security blockers or temporary account lockouts directly in checkout flows. This high prevalence severely throttles the deployment viability of any agent pipeline lacking robust, zero-shot spatial-temporal visual processing.

Consequently, production-grade systems have either reverted to extreme architectural isolation or abandoned structural metadata entirely. The vulnerability of Web AI agents to prompt injection attacks embedded seamlessly within user-generated UI text further disqualifies the DOM as a safe perceptual medium. To achieve the 85% success rates seen in hybrid commercial implementations, architectures must combine selective accessibility snapshots with rigid, programmatic code constraints that actively ignore misleading structural tags. Ultimately, operating securely in the modern web environment demands systems that process the interface visually, prioritizing rendered pixel geometry over fundamentally unreliable HTML structures.

## E. The Substitution Gradient: Independent Novelty Assessment

The evolution of agent perception has generated intense debate regarding the precise boundary where visual environments are translated into semantic tokens. Analyzing the literature independently of marketing rhetoric reveals a distinct transition phase—a substitution gradient—where models bypass raw visual inference in favor of highly optimized, synthesized textual representations of GUI affordances. The concept of substituting visual space for textual markers is not novel; what is novel is precisely *where* in the computation pipeline this substitution occurs.

**(α) Literature on Textual Representation of Visual Affordances:** The conversion of visual screen data into flattened, structured text lists is extensively documented in recent pipeline-engineering research. The *ScribeAgent* architecture (December 2024) specifically trains open-source LLMs utilizing production-scale workflow data comprising 6 billion tokens derived entirely from specialized web DOM structures. This research proves that purely textual representations of web components can yield state-of-the-art direct generation performance if the fine-tuning dataset is sufficiently dense. More explicitly, Microsoft's *OmniParser-v2* formalizes this visual-to-text substitution by utilizing an isolated, compact MoE vision model to draw bounding boxes and execute OCR, emitting a structured, tokenized list of elements. The downstream reasoning model (e.g., GPT-4o) never processes the image itself; it receives only the textual coordinate list as a surrogate for the visual field, proving that high-fidelity textual substitution is an established mechanism for reducing inference costs.

**(β) Analyzing the Substitution Gradient:**

The literature indicates a fluid gradient across the computing stack, dictating the locus of translation from visual affordance to semantic token:

1. **Server-Side Affordance:** The deepest layer of substitution occurs at the web host. Microsoft's *NLWeb* forces the environment to natively emit agent-readable `Schema.org` JSON. Here, the substitution gradient is pushed entirely onto the server; the agent never parses a visual or HTML layout, interacting purely with pre-synthesized data.
2. **Agent-Pipeline Preprocessing:** Systems like *OmniParser* and *AutoGLM-Web's* HTML Simplification Algorithm operate in the middle of the gradient. They act as middleware, ingesting complex visual/DOM data and algorithmically pruning it into a clean, text-based intermediate representation before passing it to the reasoning layer.
3. **Agent-Compute LLM-Internal Substitution:** At the far end of the gradient, models like *Magma* push the substitution entirely into the weights of the neural network. By utilizing Set-of-Mark (SoM) and Trace-of-Mark (ToM) pretraining paradigms, visual tokens and textual action markers are fused internally. The LLM does not require an external pipeline to translate the screen; its cross-modal architecture inherently links the visual manifestation of a button directly to the semantic execution command.

**(γ) Operating on Textual Element Lists Without Visual Rendering:** The hypothesis that an agent can operate effectively on a purely textual distillation of a visual environment is deeply supported by studies contrasting text-only models with multimodal baselines. Evaluations on the WebArena and Mind2Web datasets indicate that models supplied with a high-fidelity parsed HTML/A11y tree can output grounded execution commands without requiring the corresponding screenshot. Furthermore, the *Hierarchical Memory Tree (HMT)* framework (March 2026) demonstrates that spatial planning can be entirely abstracted. HMT explicitly decouples high-level logical planning from low-level action execution, allowing the agent to plan complex stage-level subgoals purely through semantic element descriptions, proving that the visual render is technically superfluous for the logical reasoning phase of execution.

**(δ) Structured Textual Representation as a Deployment Routing Arm:** Using agent-side structured text generation as the primary routing mechanism (as opposed to full multimodal processing over an image) is characterized as a highly optimized architectural bridge. The *AppAgent-v2* framework typifies this strategy. During its "exploration phase," the agent translates the chaotic visual interface into a definitive text document—a localized manual of element affordances. During the subsequent "deployment phase," the agent routes its actions entirely based on Retrieval-Augmented Generation (RAG) against this textual document. It circumvents the severe latency and token-cost of processing live visual data for every step, confirming that structured textual representation is a formalized, highly effective deployment strategy for standardizing execution within complex GUI environments.

## F. BibTeX (Top 15 Core References)

代码段

```
@article{qin2025uitars,
  title={UI-TARS: Pioneering Automated GUI Interaction with Native Agents},
  author={Qin, Yujia and Ye, Yining and Fang, Junjie and others},
  journal={arXiv preprint arXiv:2501.12326},
  year={2025},
  publisher={ByteDance Seed}
}

@article{liu2024autoglm,
  title={AutoGLM: Autonomous Foundation Agents for GUIs},
  author={Liu, Xiao and Qin, Bo and Liang, Dongzhu and others},
  journal={arXiv preprint arXiv:2411.00820},
  year={2024},
  publisher={Zhipu AI / Tsinghua University}
}

@article{zhang2024appagent,
  title={AppAgent: Multimodal Agents as Smartphone Users},
  author={Zhang, Chi and others},
  journal={arXiv preprint arXiv:2408.11824},
  year={2024},
  publisher={Tencent}
}

@article{lu2025omniparser,
  title={OmniParser for Pure Vision Based GUI Agent},
  author={Lu, Yadong and Dhome-Casanova, Thomas and Yang, Jianwei and Awadallah, Ahmed},
  journal={arXiv preprint arXiv:2502.16161},
  year={2025},
  publisher={Microsoft Research}
}

@article{microsoft2025magma,
  title={Magma: A Foundation Model for Multimodal AI Agents},
  author={Microsoft Research},
  journal={arXiv preprint arXiv:2502.13130},
  year={2025}
}

@article{jin2025osgenesis,
  title={OS-Genesis: Automating GUI Agent Trajectory Construction via Reverse Task Synthesis},
  author={Jin, Chuanyang and others},
  journal={ACL 2025},
  year={2025}
}

@article{wang2025osatlas,
  title={OS-ATLAS: A Foundation Action Model for Generalist GUI Agents},
  author={Wang, Yibo and others},
  journal={ICLR 2025},
  year={2025}
}

@article{shen2024scribeagent,
  title={ScribeAgent: Towards Specialized Web Agents Using Production-Scale Workflow Data},
  author={Shen, Junhong and Jain, Atishay and others},
  journal={arXiv preprint arXiv:2411.15004},
  year={2024},
  publisher={Carnegie Mellon University}
}

@article{li2024websuite,
  title={WebSuite: A Diagnostic Benchmark for Web Agents},
  author={Li, Eric and Waldo, Jim},
  journal={arXiv preprint arXiv:2406.01623},
  year={2024},
  publisher={Harvard University}
}

@article{webaim2026million,
  title={The WebAIM Million - An annual accessibility analysis of the top 1,000,000 home pages},
  author={WebAIM},
  year={2026}
}

@article{hashimoto2026agentsmd,
  title={AGENTS.md: Formal Grounding for Natural Language Agent Harnesses},
  author={Hashimoto, et al.},
  journal={arXiv preprint arXiv:2602.11327},
  year={2026}
}

@article{vardanyan2025browseragents,
  title={Building Browser Agents: Architecture, Security, and Practical Solutions},
  author={Vardanyan, Aram},
  journal={arXiv preprint arXiv:2511.19477},
  year={2025}
}

@article{koh2024visualwebarena,
  title={VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks},
  author={Koh, J. Y. and others},
  journal={ACL 2024},
  year={2024}
}

@article{huang2026hmt,
  title={Hierarchical Memory Tree for Robust Generalization of Web Agents},
  author={Huang, et al.},
  journal={arXiv preprint arXiv:2603.07024},
  year={2026}
}

@article{zhipu2024autowebglm,
  title={AutoWebGLM: Bootstrap And Reinforce A Large Language Model-based Web Navigating Agent},
  author={Zhipu AI},
  journal={arXiv preprint arXiv:2404.03648},
  year={2024}
}
```