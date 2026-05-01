# Gemini Deep Research Queries — Zoom 3 lit anchor expansion

**Date**: 2026-05-01
**Scope**: 5 parallel queries to expand `paper.bib` Zoom 3 (named cross-model phenomena) coverage.
**Context**: Phantom-SoM paper investigating cost-aware routing for VLM web agents. Found "phantom routing space" — 3 routing arms (P-text / P-prompt / P-SoM) achieving SoM-style benefits without annotated image. Need cross-model behavioral lit anchors to support paper §5 mechanism prose.

**How to use**: Copy each query block (Q1-Q5) as a separate Gemini Deep Research prompt. Run all 5 in parallel — they're independent. Save outputs to `docs/literature/zoom3_dr_<query_name>.md`.

**Existing anchors (don't redo)**: Asadi 2026 Mirage / Vu & Balloccu 2026 Scaffold / Kaduri Cross-modal Flow / Sclar 2024 prompt-format sensitivity / Mishra 2022. Already in `docs/literature/phantom_som.md` deep research (~1635 papers prior).

---

## Q1: Visual prompting without image — quantitative recovery rates across VLMs

**Context** (paste into Gemini):

I'm writing an academic paper on cost-aware routing for vision-language web agents. I have an empirical finding: when a Set-of-Mark (SoM) prompt is given to a VLM along with text observation but **no actual screenshot**, the agent still achieves substantial task completion — suggesting language priors fill in for absent visual modality.

Existing anchors I have:
- **Asadi et al. 2026 "Mirage Effect"** (arXiv:2603.21687, Stanford): VLM 无图准确率达有图的 70-80% (mirage-mode > guess-mode), cross-model
- **Kaduri et al.** layerwise attention analysis: middle-layer cross-modal flows store image info in query token representations
- **Liu et al.** on "text inertia": VLM outputs persist even without images

**Goal**: Find 5-10 additional papers (2023-2026) that quantify VLM accuracy retention without images, ideally with specific percentages or effect sizes. Cross-model coverage preferred (LLaVA / GPT-4V / Claude / Qwen-VL / Gemini families).

**Specific questions**:
1. What is the typical accuracy drop when a VLM is given a question that requires image understanding but no image is provided (text-only fallback)?
2. Are there papers explicitly measuring "language prior substitution" or "modality dropout" with quantitative recovery rates?
3. How does VLM size correlate with text-only recovery rate (Lazy Minimization Hypothesis: smaller VLMs = stronger text-over-vision bias)?
4. Are there specific benchmark datasets (e.g. VQA / GQA / MMBench / Mind2Web) where VLM no-image performance has been measured systematically?

**Output format**:
- Section 1: 5-10 paper summaries with year + authors + key finding + quantitative recovery rate (if reported)
- Section 2: BibTeX entries ready to paste into paper.bib
- Section 3: One-paragraph synthesis of how this body of evidence supports a "70-80% retention without image" general claim

---

## Q2: Prompt format sensitivity in multi-step / agentic LLM tasks

**Context** (paste into Gemini):

I'm investigating how observation text format (hierarchical accessibility tree vs flat indexed list) affects LLM web agent behavior. Existing anchors:
- **Sclar et al. 2024**: LLMs are highly sensitive to spurious formatting features; minor prompt changes → major performance shifts (single-task setting)
- **Mishra et al. 2022**: Reframing instructions to match LLM's preferred syntactic language alters generative output

**Gap**: most prompt-format-sensitivity literature is on single-turn tasks (classification, QA, summarization). I need evidence on **multi-step agent settings** where format changes affect not just a single output but a cumulative trajectory of decisions.

**Goal**: Find 5-10 papers (2023-2026) on prompt format / observation format sensitivity in multi-step agent contexts (web navigation, embodied agents, tool-use agents).

**Specific questions**:
1. Do hierarchical text representations (XML/JSON/tree) vs flat list representations cause measurable differences in agent task success?
2. Is there empirical work showing that the same underlying information presented as different formats leads to **different exploration policies** in agents?
3. How does observation-format sensitivity scale with model size? Do smaller models suffer more from format-induced policy shifts?
4. Specifically for web agents on benchmarks (WebArena / VisualWebArena / Mind2Web / WebShop): how do AXTree / DOM / SoM observation formats compare empirically?

**Output format**:
- Section 1: 5-10 paper summaries with year + authors + key finding + experimental setup
- Section 2: BibTeX entries
- Section 3: One-paragraph synthesis: "Prompt format sensitivity in multi-step settings — what is established vs unexplored"

---

## Q3: AXTree vs flat element list — head-to-head empirical comparison in web agents

**Context** (paste into Gemini):

In web agent literature, observations are typically given as either:
- **Accessibility Tree (AXTree)**: hierarchical text representation with parent-child relationships, used by WebArena baseline, FOCUSAGENT, etc.
- **Set-of-Mark (SoM) text**: flat indexed list `[id=N] role 'label'` extracted from interactive elements

Existing anchor (from my prior deep research): the SoM literature consistently bundles SoM-text with marked images — there's no head-to-head empirical comparison of AXTree vs SoM-text alone (without image).

**Goal**: Find 5-10 papers (2023-2026) that empirically compare hierarchical text observations vs flat indexed observations in autonomous web agents, with task success rates.

**Specific questions**:
1. Has anyone benchmarked AXTree vs flat element list as observation formats holding everything else constant?
2. What are the empirical task-success differences (% delta) between hierarchical and flat representations in WebArena / VWA / Mind2Web?
3. Is there mechanistic work explaining WHY format affects agent trajectory (attention patterns, in-context learning, etc.)?
4. Are there papers on **token-budget-controlled** comparisons (where tree vs flat tokens are matched)?
5. Is there a paper that frames this as "tree traversal vs sequential list scanning trajectory" cognitive operation?

**Output format**:
- Section 1: 5-10 paper summaries
- Section 2: BibTeX entries  
- Section 3: Synthesis paragraph: "AXTree vs flat list — what is empirical, what is assumed, what is novel territory"

---

## Q4: Scaffold Effect cross-domain validation

**Context** (paste into Gemini):

Existing anchor:
- **Vu & Balloccu 2026 "Scaffold Effect"**: in **clinical VLM** evaluation, merely mentioning "MRI is available" in the prompt accounts for **70-80% of apparent multimodal performance shifts**, independent of actual image presence.

This is one of the strongest signals that prompt mentioning a modality alone (without providing it) substantially affects VLM behavior. But the original paper is clinical-domain specific.

**Goal**: Find 5-10 papers (2023-2026) that replicate or contradict this "Scaffold Effect" outside clinical settings — ideally in web agents, embodied agents, or general VQA tasks.

**Specific questions**:
1. Has anyone replicated the Scaffold Effect in non-clinical domains?
2. Are there papers on "prompt-mention without modality" effects in agent settings?
3. What are quantitative effect sizes for this phenomenon across different VLMs?
4. Is the Scaffold Effect related to or distinct from "text inertia" / "language prior dominance" / "Mirage Effect"? How are these phenomena taxonomized in literature?
5. Are there debiasing methods that mitigate Scaffold Effect (training-free / inference-time interventions)?

**Output format**:
- Section 1: 5-10 paper summaries with quantitative findings
- Section 2: BibTeX entries
- Section 3: Synthesis: "Scaffold Effect — clinical origin, cross-domain status, taxonomic relations to Mirage/text-inertia"

---

## Q5: Bidirectional modality fusion — image-over-text vs text-over-vision dual nature

**Context** (paste into Gemini):

I'm framing VLM modality interaction as **bidirectional** rather than uni-directional. Existing anchors:
- **Tong et al. 2024 "Eyes Wide Shut"** (NeurIPS): VLMs over-rely on **language priors** even when correct visual evidence is available — text-over-vision bias
- **Li et al. POPE 2023**: object hallucination from **image-over-text** dominance — VLM commits to non-existent objects based on visual saliency
- **Bitton-Guetta et al. 2023** on visual hallucination

**Hypothesis**: VLMs exhibit DUAL failure modes:
- (a) image-over-text: visual saliency hijacks output even when text contains correct answer
- (b) text-over-vision: language prior dominates even when image contradicts text-implied content

These are NOT the same phenomenon — they manifest in opposite directions. Most prior work focuses on one direction at a time.

**Goal**: Find 5-10 papers (2023-2026) that explicitly characterize VLM modality interaction as bidirectional, or that compare image-over-text vs text-over-vision systematically.

**Specific questions**:
1. Is there a unified framework treating VLM hallucination as bidirectional modality fusion failure?
2. What empirical work measures both directions on the same model / benchmark?
3. Are there papers on "modality balance" / "cross-modal alignment" with quantitative dual-direction analysis?
4. How does VLM scale (small vs large) affect the relative magnitude of these two biases?
5. Are there decoding-time or training-time interventions that target one direction without amplifying the other?

**Output format**:
- Section 1: 5-10 paper summaries
- Section 2: BibTeX entries
- Section 3: Synthesis: "Bidirectional modality fusion — is this an explicit framework in lit, or are we synthesizing from one-directional studies?"

---

## After all 5 queries return

Save outputs to:
- `docs/literature/zoom3_dr_visual_prompting_no_image.md` (Q1)
- `docs/literature/zoom3_dr_prompt_format_agent.md` (Q2)
- `docs/literature/zoom3_dr_axtree_vs_flat.md` (Q3)
- `docs/literature/zoom3_dr_scaffold_effect.md` (Q4)
- `docs/literature/zoom3_dr_bidirectional_modality.md` (Q5)

Then update `docs/checkpoints/next_steps.md §4 Codex Task Queue` task #10:
- Old: "Axis 2/3 literature deep research + paper.bib expansion (16→~38) ~400-600K tokens"
- New: "**paper.bib formal expansion using Q1-Q5 Gemini DR outputs** ~50-100K tokens (just integrate, lit search done)"

Also update `docs/checkpoints/paper_planning.md §2` Zoom 3 lit anchor list to reference the 5 new DR docs alongside existing anchors.

---

## Why these 5 queries (not fewer or more)

- **Q1+Q4**: M1 axis (Image-mirage activation) lit anchors — Mirage + Scaffold are the two strongest anchors; deepening both gives M1 hypothesis cross-validation.
- **Q2+Q3**: M2 axis (Flat-list activation) lit anchors — prompt-format sensitivity is the general theory; AXTree-vs-flat is the web-agent-specific empirical anchor.
- **Q5**: Axis 3 (image-on extension, paper §2 8-channel framework) lit anchors — bidirectional modality fusion is the theoretical scaffold for paper §5 image-channel decomposition.

→ Each axis (M1 / M2 / image) gets at least 1 dedicated query; M1 gets 2 because it's the most contentious "vision-grounding without vision" claim that needs strongest cross-model evidence.

---

## Cost estimate

- Gemini DR: ~5 queries × 30-60 min each = ~3-5 hours wall time (async parallel, fire and forget)
- Gemini Pro / Advanced subscription cost: included in plan (no per-query charge)
- Total token output: ~50-100K text across 5 reports
- Codex integration after: ~30-50K tokens output (paper.bib formal entries + paper_planning §2 update)

Total cost: subscription only. ~$0 marginal compared to codex doing the same lit search.
