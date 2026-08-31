16 Sep
Exhibition
Talks
Programme
Awards
Committee
Register
Holistic AI
UCL Centre for Digital Innovation
HAI TAHR Showcase

How AI Thinks, Acts & Helps,
Responsibly.
Interpretability, agents, multimodal systems, and the evaluation that catches them failing. A day of Holistic AI × UCL CDI research, presented by the people who did it.

We aim to understand how responsible intelligence emerges across scales, from latent neural mechanisms to reliable agentic orchestration in real-world deployment.

Register, free
→
Wed 16 Sep 2026
/
UCL Centre for Artificial Intelligence
Date
Wed 16 Sep 2026
Posters all day · programme 10:30 to 16:25
Venue
UCL Centre for AI
London · all day
Participants
To be announced
One shared poster session, no tracks
Attendance
To be announced
Open registration and invited guests
01 · Exhibition
To be announced
Poster exhibition
All work is exhibited for the full day in a single session, without parallel tracks. Titles and abstracts will be published here once finalised.

02 · Invited talks
From thesis to publication
Three researchers from earlier cohorts of the Holistic AI and UCL master's programme, which UCL CDI joined this year. Each presents published work from ICML, ICLR, NAACL, NeurIPS and EMNLP, alongside an account of the route from thesis to peer review.

Seonglae Cho
Seonglae Cho

Senior AI Engineer · Engineering, Holistic AI

UCL MSc Computer Science · 2023/24 Holistic AI cohort

Invited talk
CorrSteer: Generation-Time LLM Steering via Correlated Sparse Autoencoder Features
ICML 2026
Sparse autoencoders (SAEs) decompose a transformer's residual stream into thousands of monosemantic features, but picking which features to steer at generation time has remained ad-hoc. CorrSteer correlates per-sample SAE activations with downstream task outcomes during generation, ranks features by that signal, and validates the top candidates with targeted causal intervention. The result is a fully automatic steering pipeline that requires no manual feature labelling, no fine-tuning, and no gradient updates to the base model.

Key results

+27.2 percentage points on HarmBench refusals (Gemma-2 2B)
+3.3 points on MMLU reasoning (LLaMA-3.1 8B)
Lower side-effect ratio than supervised fine-tuning across 6 benchmarks
Generalises across model families (Gemma-2, LLaMA-3.1) with no per-model re-training
At the workshop

Talk covers the SAE-correlation method, a live demo of feature discovery on Gemma-2, and a discussion of where activation steering does and does not generalise, including the failure modes observed on Pythia.

with Zekun Wu, Adriano Koshiyama

Read paper
↗
Ilham Wicaksono
Ilham Wicaksono

AI Engineer · AI Safety & Governance, Holistic AI

UCL MSc Artificial Intelligence · 2023/24 Holistic AI cohort

Invited talk
Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs
OpenAI Red-Teaming Challenge · Winner 2025
ICLR 2026 Agents in the Wild (Spotlight & Oral)
ICLR 2026 AFAA
NeurIPS 2025 LLMEval
Standard LLM evaluations score a model's responses to isolated prompts, but production agents fail in ways no prompt-level benchmark sees: bad tool selection, corrupted intermediate memory, and chained reasoning errors. AgentSeer is an observability layer that decomposes a live agent trajectory into an action-component graph (perceive → plan → act → observe), instruments each node, and lets you compare the same model evaluated in isolation against the same model embedded inside an agent.

Key results

Tool-calling attack-success rate runs 24 to 60 percentage points higher than the underlying model's ASR, invisible at prompt level
Reproduces the gap across GPT-4o, Claude 3.5, and Llama-3 agents
Action-graph framing won the OpenAI Red-Teaming Challenge (2025)
Released as open-source observability tooling
At the workshop

Talk walks through three live agentic failure modes captured with AgentSeer, contrasts them with what the underlying model's HELM / MMLU scores predict, and shows the action-graph framing that lets researchers reproduce these failures in an hour rather than a week.

with Zekun Wu, Rahul Patel, Theo King, Adriano Koshiyama, Philip Treleaven

Read paper
↗
arXiv:2509.04802
Rishi Kalra
Rishi Kalra

AI Engineer, Holistic AI

UCL MSc Financial Computing · 2022/23 Holistic AI cohort

Invited talk
HyPA-RAG: A Hybrid Parameter-Adaptive Retrieval-Augmented Generation System for AI Legal & Policy Applications
NAACL 2025 Industry Track
EMNLP 2024 CustomNLP4U
Legal and policy text is heterogeneous: a single corpus mixes statute, case law, regulatory guidance, and definitional clauses, each with different retrieval needs. HyPA-RAG classifies query complexity at runtime, then routes between dense, sparse, and knowledge-graph retrievers with parameters tuned per query type. The system was built and evaluated end-to-end on NYC Local Law 144 (the algorithmic-hiring audit law), one of the first AI-specific regulations in the US.

Key results

Higher retrieval accuracy and response fidelity than fixed-parameter RAG baselines on the NYC LL144 benchmark
Contextual-precision gains on definitional and compliance-checking queries
Deployed inside Holistic AI's audit tooling for client-facing compliance work
Released benchmark dataset and evaluation harness with the paper
At the workshop

Talk traces the route from MSc thesis (Aug 2023) → EMNLP 2024 workshop paper → NAACL 2025 Industry Track → production deployment, and discusses the engineering compromises needed to take an academic RAG system into a regulated compliance setting.

with Zekun Wu, Ayesha Gulley, Airlie Hilliard, Xin Guan, Adriano Koshiyama, Philip Treleaven

Read paper
↗
arXiv:2409.09046
03 · Doctoral study
The part-time route
Kleyton da Costa
Kleyton da Costa

AI Engineer · Holistic AI

Part-time PhD in Computer Science (AI & Robotics), UCL 2026–2030 · advised by Prof Philip Treleaven and Prof Dimitrios Kanoulas

What a part-time PhD actually costs, and what it buys

Most people finishing an MSc believe the choice is binary: carry on into a PhD, or go and get a job. There is a third option that almost nobody explains, and Kleyton is living it, a part-time PhD at UCL in AI and robotics while working full time as an AI engineer. This session is the practical version of that decision rather than the inspirational one: how the week divides, what genuinely has to give, how long it really takes, and who it suits.

His supervisor is Prof Philip Treleaven, who supervises most of this cohort, so this route is not hypothetical. It is directly open to you
Day job: research engineer on an AI governance platform deployed across S&P 500 enterprises, plus the open-source holisticai library
Research that travelled outside the lab: co-author on a Bank of England working paper on deep-learning model fragility
Came to UCL via an MSc at PUC-Rio and anomaly-detection research for offshore oil-well monitoring, a non-linear route, not a straight line
The honest trade-offs: why part-time rather than full-time or neither, what you give up and what you keep, how industry work sharpens the research and how the research changes the engineering, and the questions worth asking yourself before committing four years to it.

kleytoncosta.com
04 · Programme
To be confirmed
Schedule
A working draft. Timings and speakers remain subject to confirmation. The structure is settled: exhibition throughout the day, judging after lunch, awards to close.

Morning
09:00
Poster set-up
Programme

30m
09:30
Registration & refreshments
Break

1h
09:45
AV check & speaker briefing
Programme

30m
10:00
Exhibition opens: viewing and sticker voting, open from now on
Poster Session

all day
10:30
Opening remarks: Holistic AI
Programme

15m
10:45
Opening remarks: UCL Centre for Digital Innovation
Programme

15m
11:00
Inside the collaboration: Holistic AI × UCL IXN
Programme

20m
11:20
Break: stretch your legs and see the boards
Break

15m
11:35
Alumni spotlight: research and the road here
Alumni Talk

45m
12:20
The PhD route: doing both at once
Alumni Talk

15m
12:35
Lunch, posters and social: the long open stretch of the day
Break

2h
13:15 · within
Authors at their boards
Poster Session

1h 20m
14:35
Break: afternoon sweets, voting closes
Break

10m
14:45
Student presentations: selected projects from the cohort
Spotlights

45m
15:30
From Policy to Production: What AI Governance Actually Looks Like Inside the Enterprise
Keynote

1h
16:30
Awards & certificates
Awards

15m
16:45
Closing remarks & networking
Reception

30m
17:15
Poster take-down & venue clear
Programme

30m
05 · Awards
Decided by the room
Prizes
01
£300
1st Place

Most votes across the day

Cover of the showcase proceedings · Mentorship to write up as a workshop paper

02
£200
2nd Place

Second most votes

Featured in the showcase proceedings

03
£100
3rd Place

Third most votes

06 · Committee
Organising committee
Workshop chair

Dr Emre Kazim

Co-Founder & Co-CEO, Holistic AI

Workshop co-chair

Dr Adriano Koshiyama

Co-Founder & Co-CEO, Holistic AI

Programme chair

Zekun Wu

Research, Holistic AI · PhD candidate, UCL · OECD.AI

UCL CDI liaison

Graça Carvalho

Director, UCL Centre for Digital Innovation

07 · Attendance
To be announced
Invited guests
Invitations are currently being issued. Names will be listed here as acceptances are confirmed; an invitation is not an attendance, and it is not ours to announce on a guest's behalf.

Wed 16 September 2026

Come and ask the authors what they actually found.
Apply to attend
→
Free · 70 places
Reviewed
See the programme
→
Draft
09:30 to 16:30
Enquiries

Programme & logistics

zekun.wu@holisticai.com
Catering partner

To be announced

© 2026 Holistic AI · UCL Centre for Digital Innovation