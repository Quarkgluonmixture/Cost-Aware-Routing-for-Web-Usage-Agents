# Dual-Track Agent and Environment Routing: A Comprehensive Analysis of State-of-the-Art Computer-Use Systems

The transition from deterministic software architectures to agentic artificial intelligence represents one of the most significant paradigm shifts in computational logic and human-computer interaction. Historically, software engineering has been predicated on predictability, where explicit instructions yield known states. However, as autonomous systems are increasingly entrusted with manipulating graphical user interfaces (GUIs), the friction between stochastic web environments and agentic planners has emerged as the primary bottleneck to reliability and scalability.

Current state-of-the-art (SOTA) systems attempt to resolve this friction almost exclusively through agent-side adaptations. Industry leaders are leveraging advanced vision-language models (VLMs) to parse raw pixels, infer Document Object Model (DOM) states, and self-correct when confronted with dynamic web elements such as cookie banners, lazy-loaded content, and modal popups. While this unilateral "visual-first" approach bypasses the complexities of obfuscated DOM structures, it places an immense computational and reasoning burden entirely on the agent.

This exhaustive research report investigates how contemporary computer-use and web-agent systems handle environment-side adaptation and observation routing within their deployment pipelines. By systematically characterizing the capabilities, environmental assumptions, and failure modes of industry and academic SOTA agents, the analysis delineates the critical necessity for a "dual-track" architecture—a framework where web environments proactively route structured, semantic affordances to agents while simultaneously serving rich visual interfaces to human users.

## Section A — Per-System Characterization

The following section evaluates six distinct categories of computer-use systems, assessing their handling of multi-mode observation routing, environment-side affordances, environmental issue mitigation, adherence to agent-readable standards, and mechanisms for knowledge injection.

### 1. Anthropic Claude Computer Use (Sonnet 3.5/3.6/4.5/4.6, Claude Code)

Anthropic’s Computer Use feature fundamentally operates as a "visual-bot," designed to simulate human interaction by observing a virtual screen and executing precise mouse and keyboard commands. **

**(a) Multi-mode observation routing:** Claude operates predominantly via the visual modality. The core interaction loop involves capturing a screenshot of the current display, analyzing the visual state, deciding on specific Cartesian coordinates, and executing commands such as `left_click`, `type`, or `mouse_move`. The system does not natively route between observation modalities (e.g., dynamically switching to the Accessibility Tree or raw DOM for hybrid parsing). Instead, recent API updates (`computer_20251124` available in Claude Opus 4.7/4.6 and Sonnet 4.6) have expanded the visual-action space by introducing explicit programmatic controls for viewport management. These include the `scroll` action (allowing directional scrolling with precise amount control) and the `zoom` action (which requires `enable_zoom: true` and a region parameter `[x1, y1, x2, y2]` to inspect a specific quadrant at full resolution without degrading the model's context window). *[Confidence: Verified from Anthropic API references]*

**(b) Environment-side affordances:** The Claude agent makes minimal architectural assumptions about the underlying web pages. Because it acts as a visual layer, it bypasses the need for structured ARIA roles or `schema.org` tags. However, the agent enforces a strict environment-side affordance regarding the coordinate space: the host implementation must provide the exact structural bounds via `display_width_px` and `display_height_px` parameters during the tool definition. It does not natively preprocess screenshots with Set-of-Marks (SoM) bounding boxes; it relies entirely on its internal multimodal parameters to translate pixels into actionable coordinates. **

**(c) Environmental issue handling:** Claude mitigates environmental issues—such as cookie banners, unexpected modal dialogs, and infinite scrolls—reactively through a tightly coupled visual-programmatic feedback loop. Because Claude reads the actual current state of the screen at each step, it theoretically adapts to real-world variances and unexpected UI changes that would break a static DOM parser. However, this pixel-dependent approach introduces significant vulnerabilities. The agent is highly susceptible to "stochastic UI entropy." For example, Claude has been observed hallucinating scroll-fade animations as intentional, static design choices rather than dynamic content loading, leading to stalled execution. Furthermore, Anthropic explicitly acknowledges that tasks humans perform effortlessly—such as dragging or complex zooming—remain highly cumbersome and error-prone for the agent. *[Confidence: Inferred from behavior and confirmed by official release notes]*

**(d) Agent-readable web standards proposals:** While Claude does not natively seek out `agent.txt` on websites, Anthropic is the primary architect of the Model Context Protocol (MCP). MCP has rapidly evolved into an open-source industry standard (now managed by the Linux Foundation) that acts as the "USB-C of AI integration". For Claude Desktop and Claude Code, MCP provides a standardized client/server integration layer. When an MCP server is present, Claude can bypass visual GUI scraping entirely, routing directly to the server to fetch structured context or execute side-effects. *[Confidence: Verified from official MCP documentation and Linux Foundation announcements]*

**(e) Knowledge injection / task-specific priors:** For its terminal-based implementation (Claude Code), Anthropic utilizes explicit environment-side contracts for knowledge injection. Task-specific priors, architectural constraints, and procedural rules are injected via a persistent `CLAUDE.md` file located in the project root. As the agent processes long-horizon tasks and the context window approaches its 92% capacity threshold, Claude initiates an automated compaction pass. This process summarizes older conversation chunks into a mid-term memory block, refreshes key project facts within `CLAUDE.md`, and trims the live context. This prevents the hard-truncation of instructions and maintains the agent's strategic alignment over multi-hour workflows. *[Confidence: Verified from official engineering blogs]*

### 2. OpenAI Operator / Computer Use Agent (CUA)

OpenAI’s Operator, powered by the Computer-Using Agent (CUA) model (a derivative of GPT-4o optimized via reinforcement learning), represents a highly autonomous, cloud-hosted browsing agent. *[Confidence: Verified from official OpenAI product announcements and technical reports]*

**(a) Multi-mode observation routing:** Similar to Anthropic, Operator routes observations exclusively through the visual modality. The CUA model predicts thoughts and actions grounded entirely in raw pixel data derived from screenshots. It does not access the DOM or the Accessibility Tree (A-Tree). The system is designed to operate flexibly across various "harness shapes," including built-in Responses API loops, custom Playwright/Selenium frameworks, and code-execution environments. **

**(b) Environment-side affordances:** Operator relies heavily on its external execution environment to manage viewports. In custom deployments, developers must explicitly define the viewport resolution (e.g., 1280x720) to provide a consistent Cartesian coordinate space for the model to inspect and interact with. Operator does not utilize environment-side semantic structures like ARIA roles. It treats the web purely as a human-centric visual canvas, meaning every pixel must be interpreted by AI vision, which becomes exponentially more difficult in remote desktop environments (VDI/DaaS) where the entire UI is streamed via a `<canvas>` element devoid of DOM context. **

**(c) Environmental issue handling:** Operator handles environmental stochasticity through an advanced "event-driven observation schedule." Rather than utilizing rigid, fixed-interval screenshots (as seen in earlier agent loops like AutoGPT), Operator adapts its observation frequency to interface entropy. Before executing any click, it generates a "pre-commit verification" step, comparing the current screenshot against its expected state (a grounding check) to account for variable network latency or asynchronous popups. However, for severe environmental roadblocks—such as CAPTCHAs, complex cookie walls, age-verification prompts, or high-stakes authentication—Operator is hardcoded with proactive refusals and a "takeover mode". In this mode, the agent pauses execution, shields its visual processing to prevent capturing sensitive user input, and delegates the resolution back to the human operator before resuming the task. **

**(d) Agent-readable web standards proposals:** Operator does not natively seek out `agent.txt` or structured semantic markers on the web. The underlying philosophy of the CUA model is to operate seamlessly in environments built strictly for humans, foregoing the need for specialized agent-page contracts. *[Confidence: Inferred from product architecture]*

**(e) Knowledge injection / task-specific priors:** Operator operates primarily via natural language instructions provided dynamically at the start of a session. Unlike Codex, the default Operator deployment lacks a built-in mechanism for persistent, repository-level markdown injection, though developers building custom CUA harnesses can implement retrieval-augmented generation (RAG) or system prompt injections manually. *[Confidence: Verified from official documentation]*

### 3. OpenAI Codex CLI / Codex

The Codex CLI acts as a terminal-based software agent, bypassing the visual GUI entirely to interact directly with the underlying execution and file systems. *[Confidence: Verified from official OpenAI engineering blogs]*

**(a) Multi-mode observation routing:** Codex CLI operates exclusively in a programmatic and textual modality. It routes observations through standard Unix shell commands (`cat`, `ls`, `grep`), reading file contents, running test suites, and emitting unified diffs. The agent loop intercepts these diffs, applying patched changes directly to the file system, thereby avoiding the token-heavy overhead of processing visual screenshots. **

**(b) Environment-side affordances:** Codex relies on incredibly strict environment-side contracts, primarily through the enforcement of JSON Schema via the Structured Outputs API. This ensures that the agent's actions—such as generating automated code review comments for continuous integration (CI) pipelines—strictly conform to the machine-readable schemas expected by the host environment. The integration of `strict: true` guarantees that the semantic structure of the output is predictable, allowing for programmatic rather than heuristic parsing. *[Confidence: Verified from official OpenAI API documentation]*

**(c) Environmental issue handling:** Environmental issues (such as failed test executions, missing package dependencies, or incorrect file paths) are handled programmatically. The agent receives `stderr` and `stdout` streams directly into its context window and self-corrects through its ReAct-style planning loop. It is highly sensitive to file system conventions, requiring specific instruction tuning (e.g., GPT-5.1-Codex-Max) to manage the nuances between Windows PowerShell environments and Unix-centric bash terminals. *[Confidence: Verified from technical deep-dives and release notes]*

**(d) Agent-readable web standards proposals:** While Codex does not browse the web, it fully embraces structured agent contracts by exposing itself as an MCP Server. This allows external orchestration frameworks (like LangChain or custom orchestrators) to securely bind to Codex's capabilities (e.g., executing a local search or running a linter) using a standardized protocol rather than brittle CLI scraping. *[Confidence: Verified from GitHub repositories and official examples]*

**(e) Knowledge injection / task-specific priors:** Codex relies on an elegant, hierarchical file-system approach for knowledge injection. Before executing any work, the CLI walks the directory tree from the project root down to the current working directory, searching for `AGENTS.override.md` and `AGENTS.md` files. These files establish global guidance (e.g., "Always run `npm test` after modifying files") and project-specific overrides. Codex concatenates these files into the system prompt, stopping once the combined size reaches the `project_doc_max_bytes` limit (default 32 KiB). This acts as a robust, deterministic environment-side contract. *[Confidence: Verified from official documentation]*

### 4. Google Gemini Computer Use / Mariner

Project Mariner, powered by the Gemini 2.5 Computer Use model, represents Google DeepMind’s approach to multi-tasking agentic control across web and mobile app interfaces. **

**(a) Multi-mode observation routing:** Mariner utilizes a highly multimodal architecture. While its API predominantly requests a screenshot, a history of actions, and a text prompt to initiate the loop , the underlying Gemini infrastructure natively parses text, code, images, and forms simultaneously without relying on modular adapters. This allows Mariner to operate effectively in a hybrid mode. When deployed as a Chrome extension, it can seamlessly process pixel data alongside underlying DOM or Accessibility Tree structures, allowing for greater spatial grounding than pure-vision models. *[Confidence: Verified from technical reports and API guides]*

**(b) Environment-side affordances:** Mariner introduces dynamic environment-side shaping. Developers can explicitly define the agent's environment by providing a list of custom functions or excluding specific UI actions from the API request. This allows the host application to artificially constrain the agent's action space, preventing it from attempting unauthorized interactions based purely on visual affordances. *[Confidence: Verified from official API documentation]*

**(c) Environmental issue handling:** Mariner addresses environmental unpredictability through a combination of visual planning and rigid safety architecture. The system includes an out-of-model "Per-step safety service" that evaluates each proposed action at inference time before it is executed on the UI. For high-risk environmental blockers—such as CAPTCHAs, terms of service agreements, or payment gateways—Mariner is designed to trigger a "Take over" mode, pausing operations and requiring explicit human intervention to navigate the obstacle. *[Confidence: Verified from official system cards and documentation]*

**(d) Agent-readable web standards proposals:** There is no explicit documentation indicating that Mariner natively supports `agent.txt` or specific semantic web extensions. Like Operator, it primarily attempts to traverse the web by emulating human perception. *[Confidence: Inferred from lack of documentation in the Gemini technical report]*

**(e) Knowledge injection / task-specific priors:** Similar to Codex's `AGENTS.md`, the Gemini CLI ecosystem utilizes `gemini.md` files placed throughout the project hierarchy to establish high-level context (e.g., product vision, user personas, decision logs). This allows the agent to maintain strategic alignment without overwhelming its context window, progressively loading specific technical details only when necessary. **

### 5. Avenir-Web (Online Mind2Web SOTA)

Avenir-Web is an advanced open-source agent framework specifically designed to imitate human experience to overcome the reliability bottlenecks inherent in live, dynamic web navigation. *[Confidence: Verified from peer-reviewed academic literature]*

**(a) Multi-mode observation routing:** Avenir-Web implements a sophisticated "Mixture of Grounding Experts" (MoGE) architecture. It prioritizes a "visual-first" grounding path using a Multimodal Large Language Model (MLLM) to interact with the interface as a unified visual canvas. However, it intelligently routes to a semantic structural reasoner (parsing the DOM and A-Tree) as a robust fallback for edge cases requiring extreme precision or when the visual grounding point is unresponsive. *[Confidence: Verified from peer-reviewed paper]*

**(b) Environment-side affordances:** By treating the GUI primarily as a visual canvas, Avenir-Web bypasses the strict necessity for clean HTML or properly configured ARIA roles. This allows it to navigate highly complex DOM structures—including shadow DOMs, dynamically injected canvas elements, and nested iframes—that consistently paralyze traditional, DOM-centric web agents. *[Confidence: Verified from peer-reviewed paper]*

**(c) Environmental issue handling:** Avenir-Web exhibits one of the most explicit issue-handling methodologies in the academic sphere. It employs Action-Specific Checks and State-Change Verification. If an agent executes a click but the subsequent screenshot reveals zero visual or structural changes (e.g., because a transparent cookie banner intercepted the click), the action is immediately flagged as a failure. Avenir-Web utilizes an Adaptive Memory module with Failure Reflection to analyze these stalling patterns. The agent's learned policy explicitly directs it to "close/accept blocking modals, overlays, cookie banners first" before attempting to interact with underlying elements. *[Confidence: Verified from peer-reviewed paper and system prompts]*

**(d) Agent-readable web standards proposals:**

Avenir-Web operates on the assumption of a hostile, human-centric web and does not rely on emerging agent-readable standards. *[Confidence: Inferred from paper methodology]*

**(e) Knowledge injection / task-specific priors:** Avenir-Web mitigates the absence of site-specific procedural knowledge through "Experience-Imitation Planning" (EIP). This module retrieves and comprehends human-authored online guides (procedural priors) to produce high-level, site-specific roadmaps. This injection of external knowledge prevents the agent from engaging in expensive, trial-and-error exploration and reduces its reliance on transient parametric memory. *[Confidence: Verified from peer-reviewed paper]*

### 6. Academic / Open-Source Agents (Browser-Use, WebVoyager, SeeAct, WebArena)

This broad category encompasses various academic frameworks designed primarily for benchmark testing and open-source deployment, exposing a wide variance in architectural philosophies. *[Confidence: Verified from comprehensive review of academic literature]*

**(a) Multi-mode observation routing:** The routing mechanisms within academic agents are highly fragmented. WebVoyager is almost entirely vision-based, extracting elements via screenshots processed by models like GPT-4V. Conversely, SeeAct relies heavily on textual HTML and A-Tree representations, requiring a fine-tuned cross-encoder model to select candidate elements. Agents built for the WebArena benchmark typically rely heavily on the Accessibility Tree for precise programmatic navigation. *[Confidence: Verified from respective peer-reviewed papers]*

**(b) Environment-side affordances:** Due to the token constraints and precision issues of raw VLMs, many academic frameworks utilize heavy environment-side preprocessing. Frameworks like WebVoyager utilize external scripts (e.g., GPT-4-ACT) to parse the DOM, extract interactive elements, and inject explicit bounding box coordinates and Set-of-Marks (SoM) identifiers directly into the screenshot before passing it to the LLM. *[Confidence: Verified from peer-reviewed papers]*

**(c) Environmental issue handling:** Academic agents generally struggle profoundly with real-world environmental issues. Pure-DOM agents break when confronting obfuscated code, while vision agents struggle with occlusion. Evaluators utilizing proxy-based frameworks like WAREX (which deliberately injects network delays, JavaScript loading failures, and popups) have demonstrated severe behavioral fragility in these agents. When confronted with environmental blockers, agents typically exhibit one of two failure modes: "Premature Termination" (where the agent hallucinates task completion or gives up immediately upon encountering an error) or "Ineffective Persistence" (where the agent burns tokens in endless, repetitive loops attempting to click elements occluded by un-rendered modals). *[Confidence: Verified from empirical adversarial benchmark studies]*

**(d) Agent-readable web standards proposals:**

These architectures are designed to solve the human-web problem and do not interact with emerging semantic agent standards. *[Confidence: Inferred from architectural focus]*

**(e) Knowledge injection / task-specific priors:** Knowledge is injected primarily through massive, heavily engineered system prompts or few-shot demonstration examples. Emerging frameworks like SkillWeaver attempt to solve this by autonomously synthesizing successful task trajectories into reusable Python skills, creating an expanding library of plug-and-play APIs that act as dynamic procedural priors. *[Confidence: Verified from peer-reviewed papers]*

### Summary Characterization Table

| **System**           | **Multi-mode routing**                                       | **Env-side affordance**                                      | **Issue handling**                                           | **Agent-readable standards**                      | **Knowledge injection**                                   |
| -------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------- | --------------------------------------------------------- |
| **Anthropic Claude** | Visual only (screenshot + coordinates); explicit scroll/zoom parameters. | Requires predefined display bounds; minimal DOM reliance.    | Handled via visual-programmatic loop; susceptible to animation hallucinations. | Native MCP support (Linux Foundation standard).   | Persistent, auto-compacting `CLAUDE.md`.                  |
| **OpenAI Operator**  | Visual only (screenshot + coordinates); no DOM access.       | Relies on Playwright/VNC viewports and coordinate normalization. | Event-driven observation; falls back to human "takeover" for blockers. | Non-compliant with explicit semantic web markers. | Handled via natural language instruction/prompting.       |
| **OpenAI Codex**     | Programmatic/Textual via CLI (AST, JSON, Shell).             | Strict adherence to JSON schema / Structured Outputs API.    | Programmatic error handling via `stderr`/`stdout` ReAct loops. | Exposes itself as a standardized MCP Server.      | Hierarchical `AGENTS.md` and `.override.md` traversal.    |
| **Google Mariner**   | Multimodal (Vision + A-Tree/DOM hybrid capability).          | Function exclusion lists define action space dynamically.    | Out-of-model step safety checks; explicit human "takeover" mode. | None explicitly documented.                       | Hierarchical `gemini.md` context files.                   |
| **Avenir-Web**       | Visual-first (MoGE) with Semantic DOM fallback reasoner.     | Treats GUI as a unified canvas (bypasses shadow DOM/iframes). | State-Change Verification; adaptive memory for failure reflection. | None explicitly documented.                       | Experience-Imitation Planning (EIP) via retrieved guides. |
| **Academic SOTA**    | Highly fragmented (SeeAct: HTML; WebVoyager: Vision).        | Heavy reliance on preprocessors (SoM, bounding boxes).       | Severe fragility (Premature Termination or Ineffective Persistence). | None explicitly documented.                       | Prompt engineering / synthetic skill libraries.           |

------

## Section B — Aggregate Findings

An aggregate analysis of the deployment pipelines and benchmark performances reveals distinct patterns in how industry and academia approach agentic navigation. This analysis highlights a fundamental, unresolved misalignment between advancing agent capabilities and the inherent hostility of human-centric web environments.

### Industry SOTA Pattern: The Monolithic Vision Paradigm

The industry—represented by OpenAI’s CUA, Anthropic’s Computer Use, and Google’s Mariner—has overwhelmingly coalesced around "pixel-only" or "visual-first" foundational models. This shift represents a deliberate architectural retreat from reliance on the Document Object Model (DOM) and the Accessibility Tree (A-Tree).

The rationale driving this paradigm is pragmatic: the modern web's DOM is deeply convoluted, populated with shadow DOMs, dynamically injected iframes, and deliberately obfuscated elements (such as randomized CSS class names) designed specifically to thwart traditional scrapers and bots. By processing the UI exactly as a human does—via rendered pixels—industry agents bypass DOM instability.

However, this monolithic vision approach dictates that **environment-side fixes are largely uncoordinated and actively ignored by the ecosystem**. The burden of adaptation is placed entirely on the agent's cognitive capabilities. To interact with a simple webpage, the agent must capture a megapixel screenshot, visually identify a cookie banner, compute the Cartesian coordinates of the "Accept" button, generate a JSON object containing a `left_click` command, and execute a motor function. This paradigm is highly computationally expensive and introduces severe temporal latency (the screenshot $\rightarrow$ interpret $\rightarrow$ click $\rightarrow$ wait cycle).

Furthermore, the visual paradigm leaves agents acutely vulnerable to "stochastic UI entropy"—where asynchronous network loading, scroll-fade animations, or delayed popups invalidate the agent's spatial grounding between the moment of perception and the moment of execution. The environment remains passively hostile, and the industry's solution is merely to build a more resilient, albeit slower, agent.

### Academic SOTA Pattern: The Brittle DOM Reliance

In contrast, many academic SOTA agents (e.g., traditional WebArena agents, SeeAct) maintain a heavy reliance on the DOM and A-Tree. This allows for precise programmatic interactions and avoids the exorbitant token costs associated with passing high-resolution images to VLMs. To bridge the gap between structure and vision, researchers heavily utilize environment-side preprocessing—specifically, injecting Set-of-Marks (SoM) bounding boxes into screenshots to map visual elements back to their DOM nodes before inference.

However, this disposition results in profound behavioral fragility in real-world deployments. When deployed outside of sanitized, locally-hosted sandboxes (like the WebArena environment), academic agents fail catastrophically. Evaluators utilizing adversarial frameworks like WAREX demonstrate that when confronted with real-world failure injections (e.g., server-side 4xx errors, lazy-loading failures), these agents engage in "Ineffective Persistence"—burning tokens in endless loops attempting to click DOM elements that exist in the HTML but are occluded by un-rendered visual modals.

### Cross-Paper Consensus on Critical Environmental Issues

Across both industry technical reports and academic adversarial analyses, a definitive consensus emerges regarding the specific environment-side issues that present the highest probability of task failure. These elements consistently disrupt agentic loops:

1. **Modal Dialogs and Cookie Banners:** Unsolicited popups break the agent's planned action sequence. Visual models struggle to understand the z-index depth of elements, while DOM models frequently attempt to interact with obfuscated underlying elements. The analysis of BrowserArena benchmarks identifies pop-up banner removal as a primary, consistent failure mode across multiple LLMs.
2. **Scroll Heuristics and Viewport Culling:** Infinite scroll mechanisms and dynamic content loading break static screenshot analysis. Because elements existing in the DOM may not be rendered in the viewport, industry agents have had to hardcode specific `scroll`, `page_down`, and `zoom` tools, forcing the agent to waste inference steps simply trying to render the required data into the visual field.
3. **CAPTCHAs and High-Stakes Authentication:** These are universally recognized as insurmountable by current autonomous loops. The inability of VLMs to reliably solve or bypass these human-verification gates has forced both Operator and Mariner to implement hardcoded "takeover" or "watch" modes, effectively breaking the autonomous loop and requiring human intervention.
4. **Complex State Components:** Elements such as select dropdowns, calendar pickers, and multi-state toggles require sequential, temporally-linked interactions. These multi-step state changes often exceed the planning capacity of purely reactive agents, leading to high failure rates when the UI state does not perfectly match the agent's pre-computed expectation.

------

## Section C — Standards / Framework Proposals

To mitigate the profound friction between highly capable agents and unstructured, visually-optimized web environments, several "agent-readable" standards are emerging. However, the ecosystem remains highly fragmented, with competing philosophies regarding whether to standardize the agent's tools or the environment's structure.

**1. Model Context Protocol (MCP)**

- **Maturity Level:** **High / Emerging Industry Standard** (Linux Foundation / Anthropic).
- **Description:** MCP provides a standardized, open-source integration layer (a client/server model) that allows AI applications (such as Claude Desktop or various IDEs) to connect securely to external data sources and tools without writing custom integration code for every permutation. It acts essentially as a universal API wrapper. By March 2025, OpenAI officially adopted MCP clients in its Agents SDK, solidifying MCP's position as the leading protocol for agent-tool interaction. In this paradigm, environments expose themselves as MCP Servers, allowing the agent to fetch structured context or execute side-effects natively, completely bypassing the need for a GUI.

**2. `agent.txt` / `agents.json` (Robots.txt for Agents)**

- **Maturity Level:** **Idea / Draft RFC**.
- **Description:** A proposed standard (frequently discussed as being hosted at `/.well-known/agentbridge.json` or `agent.txt`) designed to replace or supplement the legacy `robots.txt` protocol. While `robots.txt` primarily dictates scraping permissions for foundational model training crawlers (e.g., GPTBot), `agent.txt` is conceived to provide actionable routing hints for autonomous task-completion agents. It allows domains to explicitly define which APIs an agent can hit, what schemas to use, and how to authenticate, effectively allowing the agent to bypass the visual GUI entirely.

**3. Repository-Level Context Contracts (`CLAUDE.md`, `AGENTS.md`)**

- **Maturity Level:** **De Facto Developer Standard** (Tool-Specific).
- **Description:** While not formal web standards, these markdown files represent the primary, highly mature method of environment-side knowledge injection for coding and CLI agents. `AGENTS.md` (OpenAI Codex) and `CLAUDE.md` (Anthropic) act as persistent state files where the host environment dictates its own constraints, testing heuristics, and file system conventions to the agent before execution begins. They represent a successful model of the environment dynamically shaping the agent's behavior.

------

## Section D — Counter-Evidence and Gaps

The prevailing narrative surrounding the efficacy of SOTA computer-use agents relies heavily on benchmark scores that obscure the severity of environment-side failures. A critical examination of the literature reveals systemic evaluation flaws and significantly higher real-world failure rates than commonly advertised.

### Systematic Ignorance of Environmental Issues in Evaluation

Many prominent academic evaluations systemically exclude realistic web obstacles to create "clean" testing environments. Benchmarks like **WebArena** are hosted locally and structurally sanitize the web experience. These generated environments inherently lack the stochastic noise of the open web—meaning they are devoid of pop-ups, cookie dialogs, dynamically injected CAPTCHAs, floating ads, and variable network latency. Consequently, agents tested in these sterile environments appear highly capable of complex reasoning and flawless execution, as the structural hostility of the environment has been artificially removed.

Furthermore, even in benchmarks that utilize the live web, evaluation metrics are often flawed by "Search-Time Contamination." For instance, testing on the **WebVoyager** benchmark traditionally allows agents to use Google Search. This permits agents to entirely bypass broken, obfuscated, or complex website domains by finding alternative routes to task completion via search engine aggregations. When this "search shortcut" is disabled to force the agent to actually navigate the target environment, the performance of SOTA models like Browser-Use drops drastically (from 31% to 26%), exposing their inability to navigate specific domain architectures.

### Effective Failure Rates When Issues are NOT Excluded

When evaluated on rigorous, live-web benchmarks that do not exclude environmental stochasticity—such as the **Online-Mind2Web** benchmark—the true capability gap becomes glaringly apparent.

- **Avenir-Web**, despite representing the open-source SOTA by employing a highly complex Mixture of Grounding Experts (MoGE) designed specifically to handle complex DOMs, achieved a task success rate of only **53.7%** when using the massive Gemini 3 Pro backbone. This translates to an effective failure rate of **46.3%** on real-world tasks.
- When Avenir-Web was restricted to lightweight, open-source models like Qwen-3-VL-8B, the failure rate spiked to **74.3%**.
- Proprietary industry leaders similarly falter in un-sanitized environments: the academic **Browser-Use** agent exhibits a **74% failure rate**, **SeeAct** demonstrates a **70% failure rate**, and even OpenAI's highly touted **Operator** (CUA) fails **41.7%** of the time on these un-sanitized, live-web tasks.

This counter-evidence heavily implies that relying purely on agent-side cognitive scaling (building larger, more expensive VLMs to parse pixels) yields diminishing returns. When the underlying environmental architecture remains actively hostile to programmatic navigation, the failure rate will persistently hover near the 40-50% threshold in real-world deployment.

------

## Section E — Where Our Paper Fits (Synthesis)

Given the landscape detailed above, the industry is locked in a monolithic approach: forcing hyper-advanced Vision-Language Models to visually decode a web architecture built exclusively for humans. The explicit framing of **"dual-track agent + environment routing"** is highly novel because it fundamentally shifts the burden of adaptation away from the agent's cognitive overhead and onto the host environment's routing architecture.

**Is the "Dual-Track" Framing Novel?**

Yes. Current discourse (e.g., Anthropic's Computer Use, OpenAI's Operator) frames the computer-use problem almost exclusively as: *"How do we make the agent better at seeing and understanding the web?"* The proposed dual-track routing research subverts this by asking: *"How do we make the web proactively present a semantic, agent-optimized interface alongside the visual human interface?"* This re-centers the environment as an active participant in task completion rather than a passive obstacle.

**Closest Precedents in Literature:**

1. **TRACE (Capability-Targeted Agentic Training):** TRACE establishes a strong precedent for the concept of "environment routing," but applies it strictly within a synthetic training context. It synthesizes targeted training environments based on agent failures and routes the agent to specific LoRA adapters at inference time. However, this constitutes routing *within the agent's internal neural architecture*, not routing the agent to a different semantic layer of the actual web environment.
2. **K3 Mariner (Community Edition) and ActionEngine:** The open-source K3 Mariner project introduces an "Architectural Decoupling" that separates the "Senses" (Tool Use) from the "Brain" (Reasoning). Similarly, the paper *ActionEngine* advocates shifting GUI agents from reactive visual execution to programmatic state-machine planning. Both hint at the necessity of separating the visual from the semantic.
3. **The "Agent-Fee" / Machine Economy Discourse:** Early industry literature is beginning to theorize a "Dual-Track Monetisation" ecosystem where websites must serve human-seeking-experiences alongside agent-seeking-utility, exchanging value via machine-to-machine (M2M) automated clearance.

**The Cleanest Gap: The Phantom Routing Space** The critical missing component in current literature is the architectural implementation of a **Phantom Routing Space**—an environment-side pilot. Currently, if an autonomous agent visits a domain, it is bombarded with a cookie banner designed for human visual compliance, paralyzing the VLM and contributing to a 40-70% failure rate.

The proposed dual-track routing research fills this gap by postulating that environments should feature a routing mechanism that detects agentic User-Agents and seamlessly diverts them to a "phantom space." This space would bypass visual DOM rendering entirely. Instead, it would offer a structured, semantic interface (akin to a native MCP connection or an `agent.txt` defined API endpoint) that explicitly maps site affordances without the stochastic noise of popups, dynamic CSS, or scroll-fade animations. Implementing this dual-track architecture dramatically reduces inference compute, eliminates the need for fragile VLM pixel-parsing, and effectively drops the environmental failure rate toward zero by treating the agent as a first-class, natively supported citizen of the web architecture rather than an unwanted visual scraper.

------

## Section F — BibTeX Entries for Top 10 Most-Relevant References

*(Note: Prioritization reflects official SDK documentation, peer-reviewed academic papers, and highly relevant technical disclosures as requested).*

代码段

```
@misc{anthropic2024computeruse,
  author = {Anthropic},
  title = {Computer Use (Beta) - Claude API Documentation},
  year = {2024},
  url = {https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool},
  note = {Official SDK Documentation. Details visual-only routing, display_width_px affordances, and explicit motor commands (scroll, zoom).}
}

@misc{openai2025operator,
  author = {OpenAI},
  title = {Introducing Operator: A Computer-Using Agent},
  year = {2025},
  url = {https://openai.com/index/introducing-operator/},
  note = {Official Engineering Blog. Details the CUA model, screenshot-based perception, and 'takeover mode' for handling complex environmental blockers.}
}

@article{li2026avenirweb,
  title={Avenir-Web: Human-Experience-Imitating Multimodal Web Agents with Mixture of Grounding Experts},
  author={Li, Aiden Yiliu and others},
  journal={arXiv preprint arXiv:2602.02468},
  year={2026},
  url={https://arxiv.org/abs/2602.02468},
  note={Academic SOTA. Details Mixture of Grounding Experts (MoGE), Experience-Imitation Planning, and high failure rates (46.3%) on live Online-Mind2Web.}
}

@misc{openai2026codexcli,
  author = {OpenAI},
  title = {Unrolling the Codex Agent Loop},
  year = {2026},
  url = {https://openai.com/index/unrolling-the-codex-agent-loop/},
  note = {Official Documentation. Details programmatic terminal interactions, structured outputs, and AGENTS.md hierarchical environment contracts.}
}

@misc{google2025mariner,
  author = {Google DeepMind},
  title = {Gemini 2.5 Computer Use Model and Project Mariner},
  year = {2025},
  url = {https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-computer-use-model/},
  note = {Technical Release. Details multi-modal Chrome browser interaction, exclusion APIs, and per-step out-of-model safety routing.}
}

@article{kang2026trace,
  title={TRACE: Turning Recurrent Agent failures into Capability-targeted training Environments},
  author={Kang, Hangoo and others},
  journal={arXiv preprint arXiv:2604.05336},
  year={2026},
  url={https://arxiv.org/abs/2604.05336},
  note={Precedent for 'environment routing'. Details automated synthesis of training environments and dynamic routing to LoRA adapters.}
}

@article{zhou2023webarena,
  title={WebArena: A Realistic Web Environment for Building Autonomous Agents},
  author={Zhou, Shuyan and others},
  journal={arXiv preprint arXiv:2307.13854},
  year={2023},
  note={Benchmark architecture. Heavily critiqued in subsequent literature for its 'clean' synthetic environments lacking real-world popups and cookie banners.}
}

@article{he2024webvoyager,
  title={WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models},
  author={He, Hongliang and others},
  journal={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2024},
  note={Vision-only web agent methodology. Highlights the use of external preprocessors for bounding box injection and visual state-action pairs.}
}

@misc{linuxfoundation2025mcp,
  author = {Linux Foundation / Anthropic},
  title = {Model Context Protocol (MCP) Anniversary Report},
  year = {2025},
  url = {https://blog.modelcontextprotocol.io/posts/2025-11-25-first-mcp-anniversary/},
  note = {Industry Standard. Details the rapid transition of MCP from an Anthropic experiment to a universal 'USB-C' standard for environment-agent data routing.}
}

@article{chen2025warex,
  title={WAREX: Proxy-based Framework for Evaluating Reliability of Browser-based LLM Web Agents},
  author={Chen, Yurun and others},
  journal={OpenReview},
  year={2025},
  url={https://openreview.net/forum?id=LS5A21bKmA},
  note={Failure mode analysis. Details 'Ineffective Persistence' and 'Premature Termination' behaviors when agents face real-world injected UI errors.}
}
```