You are a research assistant with web search. I am submitting a paper to ACL Rolling Review (ARR) for COLING 2027. I need three things researched: (A) every formatting and policy requirement, (B) related literature, (C) verification of the references I already cite. Search the web for each item; do not answer from memory. If you cannot open a source, say "not verified" — never guess, never invent a paper, author, venue, URL or BibTeX entry.

## The paper (for judging relevance — it is under double-blind review, do not search for the authors)

Working title: "Text, Screenshots, or Both? Representation Value and Task Routing in Web Agents".
Web-browsing agents (VisualWebArena classifieds / reddit / shopping, WebArena reddit) with four vision-language backbones (Qwen3-VL-235B via API, Qwen3-VL-4B and Gemma-3-4B served locally, GPT-5.6 via API) observe pages as text (accessibility tree), a screenshot, or set-of-marks (screenshot + marked elements as text); screenshot-free variants serve as controls. Identical conditions were rerun. Claims:
1. No observation choice is best everywhere.
2. A matched ablation (set-of-marks with vs without the screenshot, same text and prompt) shows the screenshot's value; a zero-cost regex over the task intent predicts where it pays off on one site, and the effect reproduces on a rerun.
3. Per-task representation preferences are real beyond task difficulty (a no-interaction Rasch null conditioned on margins, sampled with curveball swaps, is rejected), but single-run labels have reliability 0.31–0.50 (generalizability-theory decision study).
4. Routers choosing a representation per task, with the operating point chosen on training data, do not beat the fixed representations and their random mixtures (the convex-hull frontier); the only room above that frontier comes from task difficulty (sending hopeless tasks to the cheapest mode).
5. A protocol: rerun per arm, matched ablations, a difficulty-conditioned null for per-task structure, deployable policies scored against fixed arms plus random mixtures.

Earlier versions were presented as NON-ARCHIVAL posters at two workshops: REALM @ EMNLP 2026 and VLM4RWD @ NeurIPS 2026. The paper has never been reviewed in ARR before.

---

## A. Format and policy (quote the exact rule, give the URL and the date you read it)

1. **COLING 2027 via ARR**: confirm the ARR cycle COLING 2027 takes submissions from, the ARR submission deadline (date, time, time zone), the COLING 2027 commitment deadline, notification date, and conference dates/location. Is COLING 2027 the first COLING to use ARR? Any COLING-specific rules on top of ARR (tracks, areas, theme track, commitment form contents)?
2. **Page limits**: long and short paper limits at submission; what is excluded from the count (references, Limitations, Ethics statement, appendices, acknowledgements); extra page allowed for camera-ready.
3. **Template**: the current official ACL style files (link to the acl-style-files repository, current version/date); any COLING 2027 deviation; required options for anonymous review (`[review]`), line numbers, paper size, font rules; whether pdfLaTeX is required.
4. **Mandatory sections**: Limitations section (required? placement? counted?), Ethics statement (required or optional?), anything else mandatory.
5. **Responsible NLP Research checklist**: current version for ARR, every question it asks, where answers must point into the paper, what typically triggers desk rejection.
6. **Anonymity**: rules for self-citation, links to code/data, mentioning prior workshop presentations, anonymised repositories (e.g., anonymous GitHub), acknowledgements.
7. **Preprints**: current ACL / ARR policy on arXiv posting before or during review (is there still an anonymity period? since when was it removed or changed?).
8. **Prior non-archival presentation**: is a paper previously presented at non-archival workshops (REALM @ EMNLP 2026, VLM4RWD @ NeurIPS 2026) allowed? Must it be disclosed, and where (submission form, paper)? How much overlap is permitted? Cite the ACL / ARR dual-submission and non-archival policy text.
9. **Submission mechanics**: OpenReview profile requirements for all authors, reviewer-nomination / reviewing obligation for authors, maximum submissions per author, required submission-form fields (e.g., software/data, previous submissions, preferred venues), supplementary material (separate zip? size limits? code allowed?).
10. **AI assistance policy**: the current ACL policy on using AI writing / coding assistants — what must be disclosed and where.
11. **Desk-reject triggers**: list the common ARR desk-reject reasons (format, anonymity, missing Limitations, checklist).

## B. Literature (2023 – October 2026; peer-reviewed first, then arXiv)

For every paper: full citation, URL, one or two sentences on what it does, and how it relates to my paper — **supports / competes with / contradicts / prior art for my claim N** (use the claim numbers above). Then a BibTeX entry copied from the publisher, ACL Anthology, DBLP or arXiv (say which).

1. **Observation representations for web / GUI agents**: studies comparing accessibility tree / HTML / DOM vs screenshots vs set-of-marks or other fused inputs; ablations that remove only the image; analyses of *when* screenshots help (task types, visual questions). I most need any paper that already isolates the screenshot with a matched ablation — prior art for claim 2.
2. **Per-task or adaptive selection of observation modality / representation for agents**: any work routing between text and vision observations per task or per step. This is the novelty check for my central question — search hard, including GUI agents, mobile agents, OS agents, and multimodal LLM routing (choosing whether to use the image per query).
3. **LLM routing and cascades evaluated against convex-hull / random-mixture baselines**: which routing benchmarks or papers compare routers to the convex hull of fixed models or to random mixtures (e.g., RouterBench and successors), and which report only dominance over the best single model. Also routing for agents specifically (web agents, tool agents), 2024–2026.
4. **Run-to-run variance and reproducibility in agent evaluation**: repeated-run metrics (e.g., pass^k), reports of outcome flips between identical runs, nondeterminism of LLM inference and its effect on agent benchmarks, recommended numbers of runs.
5. **Reliability of evaluation labels in NLP / ML**: generalizability theory, variance-component decompositions, decision studies, test–retest reliability, item response theory applied to benchmarks — anything that measures how reliable a per-item label is and how many repeats are needed.
6. **Learning routers from noisy or single-sample outcome labels**: routers trained on preference or outcome data, label noise in router training, how many samples per query are used.
7. **Early exit, abstention and budget-aware agents** (2025–2026): stopping or downgrading agents on tasks likely to fail; budget-constrained comparisons of agent components. Relevant to claim 4's "difficulty lever" — I need to state clearly that this lever is not my novelty.
8. **Benchmark and harness defects** in WebArena / VisualWebArena and agent benchmarks generally (evaluator errors, state leakage between tasks, task invalidity), e.g., WebArena Verified and validity audits.
9. **Matched-mixture and fair-comparison baselines** in multi-arm / multi-model evaluation outside routing (e.g., ensembles compared against randomised mixtures, cost–quality Pareto reporting conventions).
10. **Anything that would contradict my claims**: papers reporting that per-task observation or model routing for web agents beats the best fixed choice by a clear margin, especially with pre-execution features — give their setup in enough detail that I can explain the difference.

Return at least the 5 most relevant papers per topic if they exist, ranked by relevance, and say explicitly when a topic seems to have no prior work after searching.

## C. Verify my current references

For each entry below, find the authoritative version (ACL Anthology, publisher, OpenReview, DBLP, arXiv) and report: correct full author list, exact title, year, venue (the peer-reviewed venue if it was published after the arXiv version), DOI/URL, and whether my title/year/venue below is wrong. Give a corrected BibTeX entry with the SAME key. Protect capitalisation in titles with braces (e.g., {GPT-4V}, {AgentOccam}, {LLM}).

| key | my title | my year | my venue |
|---|---|---|---|
| agentoccam2025 | AgentOccam: A Simple Yet Strong Baseline for LLM-Based Web Agents | 2025 | ICLR 2025 |
| bhat2026benchmarkingbenchmarks | Benchmarking the Benchmarks: A Validity Audit of Tool-Calling Evaluation | 2026 | ? |
| chen2023frugalgpt | FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance | 2023 | arXiv 2305.05176 (published version?) |
| deng2023mind2web | Mind2Web: Towards a Generalist Agent for the Web | 2023 | NeurIPS |
| ding2024hybridllm | Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing | 2024 | ? |
| elhattami2025webarenaverified | WebArena Verified: Reliable Evaluation for Web Agents | 2025 | ? |
| enomoto2026observation | Revisiting Observation Reduction for Web Agents: Comprehensive Evaluation with a Lightweight Framework | 2026 | ? |
| gupta2024cascades | Language Model Cascades: Token-Level Uncertainty and Beyond | 2024 | ? |
| hajimiri2026budgetmatched | Are Online Skill and Memory Modules Always Worth Their Tokens? A Budget-Constrained Study of Web Agents | 2026 | ? |
| he2024webvoyager | WebVoyager: Building an End-to-End Web Agent with Large Multimodal Models | 2024 | ? |
| he2025nondeterminism | Defeating Nondeterminism in LLM Inference | 2025 | Thinking Machines Lab blog |
| kerboua2025focusagent | FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents | 2025 | arXiv 2510.03204 |
| koh2024visualwebarena | VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks | 2024 | ACL 2024 |
| li2026avenirweb | Avenir-Web: Human-Experience-Imitating Multimodal Web Agents with Mixture of Grounding Experts | 2026 | arXiv 2602.02468 |
| lu2025earlyexit | Runaway is Ashamed, But Helpful: On the Early-Exit Behavior of Large Language Model-based Agents in Embodied Environments | 2025 | arXiv 2505.17616 |
| moslem2026routingsurvey | Dynamic Model Routing and Cascading for Efficient LLM Inference: A Survey | 2026 | arXiv 2603.04445 |
| ong2025routellm | RouteLLM: Learning to Route LLMs with Preference Data | 2025 | ICLR 2025 |
| peale2026flexibleRouting | Flexible Routing via Uncertainty Decomposition | 2026 | arXiv 2605.07805 |
| schiepanski2025d2snap | Beyond Pixels: Exploring DOM Downsampling for LLM-Based Web Agents | 2025 | arXiv 2508.04412 |
| sclar2024promptformat | Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting | 2024 | ICLR 2024 |
| webrouter2025 | WebRouter: Query-specific Router via Variational Information Bottleneck for Cost-sensitive Web Agent | 2025 | arXiv 2510.11221 |
| yang2023som | Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V | 2023 | arXiv 2310.11441 |
| yuan2025numerical | Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference | 2025 | arXiv 2506.09501 (published version?) |
| zheng2024seeact | GPT-4V(ision) is a Generalist Web Agent, if Grounded | 2024 | ICML 2024 |
| zheng2024uground | Navigating the Digital World as Humans Do: Universal Visual Grounding for GUI Agents | 2025 | ICLR 2025 |
| zhou2024webarena | WebArena: A Realistic Web Environment for Building Autonomous Agents | 2024 | ICLR 2024 |

Also tell me which of these are weak or out-of-place citations for the claims above, and which stronger, more standard references I should cite instead.

## Output format

1. **Section A** as a table: requirement | exact rule (quoted) | source URL | date read | what I must do.
2. **Section B** grouped by topic: ranked list with citation, URL, 1–2 sentence summary, relation to my claim N, BibTeX (and its source).
3. **Section C** as a table of corrections, then all corrected BibTeX entries in one block.
4. A final list of anything you could not verify.
