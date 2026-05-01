# Gemini Deep Research Queries — Zoom 3 lit anchor expansion (v2 — 2026-05-01 xhigh review)

**Scope**: 6 parallel queries to expand `paper.bib` Zoom 3 (named cross-model phenomena) coverage for Phantom-SoM paper §5 mechanism prose + §7 generalization narrative.

**v1 → v2 changes** (xhigh review):
- Added unified output template (5 sections, applied to all queries)
- Added negative-evidence + forward-citation-chain mandatory sections
- Tightened citation references (full Author Year Title)
- Q2 reframed as system-prompt-instruction-format (not observation format)
- Q3 added SoM-text isolation question (paper §1 hook anchor)
- Q5 reframed as "characterize current lit state" (uni vs bi vs task-dependent), added benchmark anchors (CHAIR/POPE/MMHal-Bench/BLINK/HallusionBench), added SoM-specific harming channel question
- Q6 NEW (Lazy Minimization Hypothesis cross-model VLM scaling)

**Existing anchors (don't redo)**: Asadi et al. 2026 "Mirage Effect" (arXiv:2603.21687, Stanford) / Vu & Balloccu 2026 "Scaffold Effect" / Kaduri et al. cross-modal flow / Sclar et al. 2024 prompt-format sensitivity / Mishra et al. 2022 / Tong et al. 2024 "Eyes Wide Shut" (NeurIPS) / Li et al. 2023 POPE. Already in `docs/literature/phantom_som.md` deep research (~1635 papers prior).

**How to use**: Copy each query block (Q1-Q6) as a separate Gemini Deep Research prompt. Run all 6 in parallel — independent. Save outputs to `docs/literature/zoom3_dr_<query_name>.md`.

---

## ⚠️ UNIFIED OUTPUT TEMPLATE (for ALL 6 queries)

When responding to any of these queries, structure output as:

```
SECTION 1 — Top 5-10 papers
  For each paper:
    • Citation: Author et al. Year, "Title", Venue (arXiv:XXXX.YYYYY)
    • Finding (one declarative sentence)
    • Quantitative result (specific number + unit, if reported; "n/a" otherwise)
    • Mapping to our paper claim (which axis: M1/M2/image; which channel/sub-claim)

SECTION 2 — BibTeX entries
  5-10 @article{...} or @inproceedings{...} blocks ready to paste into paper.bib

SECTION 3 — Synthesis paragraph (~150 words)
  • What is empirically established (with citation tags)
  • What is contested or methodologically uncertain
  • What is gap our paper fills (1-2 sentences)

SECTION 4 — Counter-evidence / negative findings (MANDATORY)
  • Papers that CONTRADICT or weaken the framing
  • Mark as "counter-anchor: <citation>"
  • If no counter-evidence found: state explicitly "No counter-evidence in 2023-2026 lit per this search"

SECTION 5 — Forward citation chain (MANDATORY)
  • Has the original anchor been cited by subsequent web agent / VLM agent / multimodal LLM papers?
  • List 3-5 forward-citing papers per primary anchor (with venue + year)
  • If anchor is too recent (2026 publication) for forward chain: state explicitly
```

**Date filter**: prefer 2023-2026 papers; include foundational pre-2023 work only if cited methodologically by 2023+ work.

---

## Q1: Visual prompting without image — quantitative recovery rates across VLMs

**Context** (paste into Gemini):

I'm writing an academic paper on cost-aware routing for vision-language web agents. Empirical finding: when a Set-of-Mark (SoM) prompt is given to a VLM along with text observation but **no actual screenshot**, the agent still completes substantial fraction of tasks — language priors fill in for absent visual modality.

Existing anchors:
- **Asadi et al. 2026 "The Mirage Effect: VLMs Hallucinate Visual Information"** (arXiv:2603.21687, Stanford): VLM 无图准确率达有图的 70-80% (mirage-mode > guess-mode), cross-model
- **Kaduri et al. 2024** (need year/title verification) layerwise attention analysis: middle-layer cross-modal flows store image info in query token representations
- **Liu et al. 2024 (text inertia / image attention amplification family)** — please find authoritative citation

**Goal**: Find 5-10 additional papers (2023-2026) quantifying VLM accuracy retention without images.

**Specific questions**:

1. **Task-success retention** (NOT description-fluency retention — distinguish): typical accuracy drop when VLM is given an image-requiring question with no image (text-only fallback). Looking for benchmark numbers (VQA / GQA / MMBench / VWA / WebArena task success delta), not generation perplexity.

2. **Cross-model size correlation**: does retention rate scale with VLM parameter count? E.g., 4B vs 70B vs 235B+. Looking for explicit comparative studies.

3. **Cross-VLM-family coverage**: LLaVA / GPT-4V / Claude / Qwen-VL / Gemini / InternVL — does the 70-80% Mirage finding generalize across families, or is it Asadi-specific?

4. **Mechanism papers**: Do any 2023-2026 papers explain WHY VLMs retain so much accuracy without image? (Cross-modal flow, language prior dominance, training data leakage, attention pattern, etc.)

5. **Web agent benchmarks specifically**: any 2024-2026 work measuring text-only fallback success on WebArena / VisualWebArena / Mind2Web / WebShop?

**Apply UNIFIED OUTPUT TEMPLATE.**

---

## Q2: System prompt instruction format effects on multi-step agent task success

**Context** (paste into Gemini):

I'm investigating how **system prompt instruction format** (not observation text format — that's a different query) affects LLM web agent multi-step task completion. Specifically: same observation, same task, different prompt instruction style → does it change agent task success?

Existing anchors:
- **Sclar et al. 2024 "Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design"** (ICLR or arXiv): LLMs highly sensitive to spurious formatting features; minor prompt changes → major performance shifts (single-task setting, classification/QA)
- **Mishra et al. 2022 "Reframing Instructional Prompts to GPTk's Language"** (Findings of ACL): reframing instructions to match LLM's preferred syntactic language alters generative output
- **Salemi et al. 2024** persona priming: persona / role specification in prompt affects output

**Gap**: most prompt-format-sensitivity literature is on **single-turn** tasks (classification, QA, summarization). I need evidence on **multi-step agentic** settings where prompt format affects cumulative trajectory of decisions over many steps, not a single output.

**Goal**: Find 5-10 papers (2023-2026) on system prompt instruction format effects in **multi-step agent contexts** (web navigation, embodied agents, tool-use agents, code agents).

**Specific questions**:

1. Is there empirical work where **same task + same observation but different system prompt instructions** are evaluated for agent task success on benchmarks (WebArena, VisualWebArena, Mind2Web, ScienceQA, ALFWorld, BabyAI, OSWorld, AgentBench)?

2. Specifically for **prompt instruction style variants**: zero-shot vs few-shot demonstrations / explicit Chain-of-Thought instruction vs implicit / Set-of-Mark referencing vs DOM-element-id referencing / persona-based vs neutral — which prompt-format axes have been benchmarked in agent settings?

3. Do prompt-format effects compound over multi-step trajectories (e.g. early-step format-induced decisions cascade into late-step failures)? Any longitudinal trace analysis?

4. Are there papers measuring **cycle / loop / no-progress rate** as a function of prompt format in agent settings (not task success rate)?

5. How does prompt-format effect magnitude scale with model size? Does GPT-4-tier vs Llama-3-8B-tier show same or different sensitivity?

**Apply UNIFIED OUTPUT TEMPLATE.**

---

## Q3: AXTree vs flat indexed list — head-to-head observation format comparison in web agents (+ SoM-text isolation gap)

**Context** (paste into Gemini):

In autonomous web agent literature, observations are typically given as either:
- **Accessibility Tree (AXTree)**: hierarchical text representation with parent-child relationships, indentation, role labels. Used by WebArena baseline, FOCUSAGENT, etc.
- **Set-of-Mark (SoM) text**: flat indexed list of interactive elements in `[id=N] role 'label'` format. Originally designed by Yang et al. 2023 to accompany a marked screenshot — text + image bundled.

**Research gap I want to verify**: I claim no prior work has empirically isolated **SoM text from marked image** (i.e., used the SoM-style flat indexed list WITHOUT the accompanying screenshot, as a standalone text-only observation). My contribution depends on this isolation being unprecedented.

Existing anchors:
- **Yang et al. 2023 "Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V"** (arXiv:2310.11441): SoM original, always bundled text + image
- **Zhou et al. 2024 "WebArena: A Realistic Web Environment for Building Autonomous Agents"** (ICLR): AXTree baseline
- **Koh et al. 2024 "VisualWebArena"**: SoM applied to web tasks, again bundled
- **Deng et al. 2023 "Mind2Web"**: structured page elements
- **FOCUSAGENT (citation needed)**: AXTree retrieval / pruning, never compares to flat list

**Goal**: Find 5-10 papers (2023-2026) that empirically compare hierarchical text vs flat indexed text observation formats in web agents, AND verify (or refute) my claim about SoM-text isolation gap.

**Specific questions**:

1. **Head-to-head comparison**: Is there any paper benchmarking AXTree vs flat element list (matched information content, different structural format) in WebArena / VWA / Mind2Web / WebShop?

2. **Token-budget-controlled comparison**: When AXTree is compressed to match flat list token count (e.g. via FOCUSAGENT-style pruning), does the structural difference (tree vs flat) still produce different agent behavior? Or is it just a context-length effect?

3. **SoM-text isolation gap (paper §1 hook anchor)**: Has ANY 2023-2026 paper empirically used SoM-style flat indexed text **without** the accompanying marked image as a standalone observation? If yes, list them prominently — they'd be our prior work. If no (which I suspect), state this gap explicitly.

4. **Mechanism work**: Are there papers explaining WHY format affects agent trajectory at attention / latent-state / exploration-policy level?

5. **Tree-traversal vs sequential-list-scanning**: Has anyone framed AXTree vs flat list cognitive operation as "tree-traversal vs sequential list scanning trajectory"? If so, who first articulated this? (Critical for paper §2 axis 1 framing attribution.)

**Apply UNIFIED OUTPUT TEMPLATE. Specifically Section 4 (counter-evidence): if any paper claims AXTree dominates flat list across all benchmarks, flag prominently.**

---

## Q4: Scaffold Effect cross-domain validation + forward citation chain

**Context** (paste into Gemini):

Existing anchor:
- **Vu & Balloccu 2026 "The Scaffold Effect: ..."** (need full title + venue): in **clinical VLM** evaluation, merely mentioning "MRI is available" in the prompt accounts for **70-80% of apparent multimodal performance shifts**, independent of actual image presence.

This is one of the strongest signals that prompt mentioning a modality alone (without providing it) substantially affects VLM behavior. Original paper is clinical-domain specific.

**Goal**: Find 5-10 papers (2023-2026) that replicate, contradict, or extend this Scaffold Effect outside clinical settings. ALSO: forward citation chain analysis on Vu & Balloccu 2026 (who has cited it, and how?).

**Specific questions**:

1. **Cross-domain replication**: Has Scaffold Effect been replicated in non-clinical VLM benchmarks (general VQA / web agents / embodied agents / OCR)? With what effect size?

2. **Forward citation chain on Vu & Balloccu 2026**: List ALL papers that cite this paper (especially 2026 publications). Categorize by domain (clinical replication vs cross-domain extension vs critique).

3. **Taxonomic relations** — distinguish or conflate?:
   - "Scaffold Effect" (Vu & Balloccu 2026): prompt-mention modality without providing it
   - "Mirage Effect" (Asadi et al. 2026): VLM no-image accuracy retention
   - "Text Inertia" (Liu et al.): outputs persist without images
   - "Language Prior Dominance" (Tong 2024 "Eyes Wide Shut"): VLM ignores image even when present
   
   Which 2023-2026 papers explicitly distinguish vs conflate these terms? Provide a Venn-diagram-like synthesis if possible.

4. **Debiasing methods**: Are there inference-time or training-time interventions that mitigate Scaffold Effect specifically (not Mirage Effect generally)? E.g., decoding strategies, attention re-weighting, prompt sanitization.

5. **Paper §6 routing implication**: Could Scaffold Effect be USED as a routing signal? E.g., if prompt mentions modality but agent's downstream behavior diverges from expected modality-fed behavior, that's a routing trigger.

**Apply UNIFIED OUTPUT TEMPLATE. Section 5 forward-citation-chain is critical for this query.**

---

## Q5: Current lit state on VLM modality interaction — uni-directional, bidirectional, or task-dependent?

**Context** (paste into Gemini):

I'm framing VLM modality interaction in our paper as **bidirectional** rather than uni-directional. Hypothesis: VLMs exhibit DUAL failure modes that act in opposite directions:

- (a) **Image-over-text dominance**: visual saliency hijacks output even when text contains correct answer (object hallucination from visual content; e.g. POPE benchmark errors)
- (b) **Text-over-vision dominance**: language prior dominates even when image contradicts text-implied content (e.g. CLIP-Blind pairs, "Eyes Wide Shut" failures)

Existing anchors:
- **Tong et al. 2024 "Eyes Wide Shut: Exploring the Visual Shortcomings of Multimodal LLMs"** (NeurIPS / CVPR): VLMs over-rely on language priors even when correct visual evidence is available
- **Li et al. 2023 "Evaluating Object Hallucination in Large Vision-Language Models"** (EMNLP, POPE benchmark): object hallucination from over-commitment to visual saliency
- **Bitton-Guetta et al. 2023** (need full title): visual hallucination

**Reframed goal** (instead of asking Gemini to confirm "bidirectional" exists in lit which it likely doesn't): characterize the **current lit state** of how VLM modality interaction is framed.

**Specific questions**:

1. **Survey current framings**: Does 2023-2026 lit treat VLM modality interaction as (a) uni-directional (always image-over-text or always text-over-vision), (b) bidirectional (both directions), or (c) task-dependent (direction depends on task)?

2. **Are these phenomena explicitly compared in same-paper experiments?** I.e., is there a paper that measures both image-over-text errors AND text-over-vision errors on the same model + benchmark, and reports relative magnitudes?

3. **Benchmark coverage**: Specifically — what do these benchmarks measure?
   - **CHAIR** (Caption Hallucination Assessment with Image Relevance)
   - **POPE** (Polling-based Object Probing Evaluation)
   - **MMHal-Bench** (Multimodal Hallucination Benchmark)
   - **BLINK** (Multimodal Language Models Can See but Not Perceive)
   - **HallusionBench** (Mu, Yu, Tay et al.)
   - **MM-Vet** (Yu et al.)
   
   Which benchmarks measure image-over-text? Text-over-vision? Both?

4. **VLM size correlation**: Does the relative magnitude of these two biases change with VLM scale? (Smaller model = more text-over-vision? Larger model = more image-over-text? Or symmetric?)

5. **SoM-specific harming channels**: Have any 2023-2026 papers reported on SoM-style annotation failure modes specifically? Specifically:
   - **SoM occlusion**: text labels covering image content reducing visual readability
   - **Numeric attention hijack**: model over-fixates on N labels at high mark density
   
   These are paper §100 ground truth findings I want to anchor in lit.

**Apply UNIFIED OUTPUT TEMPLATE. Section 3 synthesis should explicitly state whether bidirectional framing is novel synthesis or is already in lit.**

---

## Q6: Lazy Minimization Hypothesis — small VLM signal selection priorities (cross-model VLM scaling)

**Context** (paste into Gemini):

I have a paper-internal hypothesis (笔记 §101.九) called **Lazy Minimization Hypothesis** that connects to existing VLM scaling lit:

> Small VLM (4B param) signal-selection priorities (descending preference):
> 1. Numeric labels (high contrast, structured tokens, easy parse)
> 2. Structured text (AXTree element_id, JSON, hierarchies)
> 3. Screenshot text content (low contrast + occlusion + OCR-dependent)
>
> Physical interpretation: 4B small VLM has worse vision-processing cost-benefit ratio, so text-over-vision bias is amplified vs large VLMs (Asadi 2026 cross-model finding consistent).

This hypothesis predicts that small VLMs benefit MORE from "phantom" routing modes (no annotated image) than large VLMs do. Cross-capability claim for paper §7.

**Goal**: Find 5-10 papers (2023-2026) on **VLM parameter scaling effects on vision-vs-text signal selection priorities**.

**Specific questions**:

1. **VLM scaling laws**: How does VLM behavior change with parameter count, specifically for vision-vs-text balance? Are there VLM scaling-law papers analogous to Kaplan/Chinchilla for LLMs but for VLMs?

2. **Small-VLM-specific text-over-vision bias**: Does empirical evidence show that smaller VLMs (3B-10B) exhibit STRONGER text-over-vision bias than larger VLMs (30B+)? If so, by what magnitude?

3. **Signal-priority hierarchies**: Are there papers explicitly modeling/measuring VLM signal selection priorities (e.g., what does the model attend to first when given multi-modal input)? E.g., attention-pattern probing across model sizes.

4. **Numeric label salience**: Has anyone specifically studied how VLMs handle numeric labels (e.g., SoM's [N] markers, OCR-extracted numbers)? Do small VLMs over-focus on numeric tokens vs large VLMs?

5. **Implications for cost-aware routing**: Are there papers proposing routing strategies that route by VLM size to mode (e.g., small VLM → text-only mode, large VLM → multimodal mode)? This is paper §6 / §7 territory.

6. **Counter-evidence**: Is there empirical work showing small VLMs have STRONGER vision processing than large ones (against my hypothesis)? Critical for paper §7 honest framing.

**Apply UNIFIED OUTPUT TEMPLATE. Section 4 (counter-evidence) is critical for this query — the hypothesis must survive contradicting lit.**

---

## After all 6 queries return

Save outputs to:
- `docs/literature/zoom3_dr_visual_prompting_no_image.md` (Q1)
- `docs/literature/zoom3_dr_prompt_format_agent.md` (Q2)
- `docs/literature/zoom3_dr_axtree_vs_flat.md` (Q3)
- `docs/literature/zoom3_dr_scaffold_effect.md` (Q4)
- `docs/literature/zoom3_dr_modality_interaction_state.md` (Q5)
- `docs/literature/zoom3_dr_lazy_minimization_scaling.md` (Q6)

Then update `docs/checkpoints/next_steps.md §4 Codex Task Queue` task #10:
- Old: "Axis 2/3 literature deep research + paper.bib expansion (16→~38) ~400-600K tokens"
- New: "**paper.bib formal expansion using Q1-Q6 Gemini DR outputs** ~50-100K tokens (just integrate, lit search done)"

Update `docs/checkpoints/paper_planning.md §2` Zoom 3 lit anchor list to reference 6 new DR docs alongside existing anchors.

---

## Why these 6 queries (axis-balanced + risk-controlled)

| Query | Axis | Leverage | Risk if not done |
|---|---|---|---|
| Q1 (Mirage 70-80%) | M1 | ⭐⭐⭐ high | Paper §1 hook citation 单点 (Asadi only) — reviewer attack "single-paper anchor" |
| Q2 (system prompt format multi-step) | M2 (axis 2 prompt) | ⭐⭐ medium-high | Paper §2 axis 2 LLM mechanism description 弱 |
| Q3 (AXTree vs flat + SoM isolation gap) | M2 (axis 1 text) | ⭐⭐⭐ high | Paper §1 hook "first SoM-text isolation" claim 无 verification |
| Q4 (Scaffold cross-domain) | M1 + paper §6 | ⭐⭐⭐ high | Paper §5 axis 2 mechanism prose 弱化 (Vu&Balloccu 单 anchor 是 clinical, transfer 不 verified) |
| Q5 (modality interaction state) | image axis | ⭐⭐ medium | Paper §2 axis 3 8-channel framework 缺 lit anchor 厚度 |
| Q6 (Lazy Minimization scaling) | cross-capability §7 | ⭐⭐⭐ high | Paper §7 cross-capability 章节 (~40% currently) 主要 anchor |

→ **Q1+Q3+Q4+Q6 是 paper §1/§5/§7 的关键 lit anchor**, Q2+Q5 是 paper §2 mechanism prose 厚度。Worth all 6 fired in parallel。

---

## Cost estimate (revised)

- Gemini DR: 6 queries × 30-60 min each = ~3-5 hours wall time (async parallel, fire and forget)
- Gemini Pro/Advanced subscription cost: included in plan (no per-query charge)
- Total token output: ~60-120K text across 6 reports
- Codex integration after: ~30-50K tokens output (paper.bib formal entries + paper_planning §2 update + counter-evidence integration to Section 4 disclosure paragraphs)

Total marginal cost: $0 (Gemini subscription only). Saves ~300K tokens vs codex doing the lit search itself.
