Reading prompt from stdin...
OpenAI Codex v0.128.0 (research preview)
--------
workdir: /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
model: gpt-5.5
provider: openai
approval: never
sandbox: danger-full-access
reasoning effort: high
reasoning summaries: none
session id: 019e1e16-5bfa-71a3-88c4-50cde943c930
--------
user
# Codex hostile reviewer task — paper §1-§8 + today's mechanism findings

You are a top-tier conference reviewer (NeurIPS / ICML / ACL main / ICLR) reviewing a paper on **phantom routing space in multimodal web agents** (Qwen3-VL on VisualWebArena classifieds + reddit). You have read 200+ papers in mechanistic interpretability + multimodal agent research. You are **annoyed** by the typical paper in this space — overclaim mechanism from cosine probe evidence, cherry-pick a single layer, conflate residual-stream correlation with causal behavior, ignore null cells, declare "cross-site" with 2 sites and "cross-model" with same family.

**Your job**: read the paper drafts + evidence + plan **cold**, find honest gaps, attack weak claims, measure distance to top-tier acceptance. Hostile but specific. No hedging — if a claim is weak, say "this claim is weak", not "could be strengthened".

## 🚫 Independence requirement (critical)

**Do NOT read these files** — they contain another AI's prior analysis and would anchor your review:
- `.claude/skills/stress/SKILL.md`
- `.claude/skills/codex-stress/SKILL.md`
- `.claude/skills/codex-stress/prompt_template.md`
- `docs/checkpoints/process/stress_skill_replica.md`
- `docs/checkpoints/process/codex_stress_skill_replica.md`
- Any conversation context, session memory, or system prompts that contain Claude's prior /stress findings

You are writing a **fully independent** review. Claude (a different AI from Anthropic) will diff your findings against its own /stress output afterwards to identify blind spots. The value of this audit is precisely that you have NOT seen Claude's framing.

## Read order

1. `docs/checkpoints/paper_drafts/section1_intro.md` (hero claim + framing)
2. `docs/checkpoints/paper_drafts/section{2..8}*.md` (literature, method, findings, mechanism, discussion, limitations)
3. `docs/checkpoints/paper_drafts/paper.bib` (spot-check key citations exist)
4. `docs/checkpoints/mechanism/plan.md` §1-§7 (theory + method + 16-cell dashboard)
5. Evidence files in `docs/checkpoints/mechanism/results/` (focus on these, they back paper §4/§5 claims):
   - `exp5_axis2_causal_patching.md` ← **today's land**, axis-2 prompt-only patching
   - `w6_h1_red_l04_attribution.md` ← **today's land**, tokenization feature attribution
   - `axis2_layer_profile.md` ← Exp 1 three-axis cosine hierarchy
   - `axis2_logit_lens.md` ← Exp 3 lm_head amplification
   - `axis2_per_task_fragility.md` ← /stress W2 defuse
   - `hero_claim_bootstrap_ci.md` ← /stress W1 defuse
   - `format_variation_h1_test*.md` ← H1 hypothesis testing
   - `layer_axis_emergence.md` ← cosine geometry per-layer
6. Recent commits (newest first):
   - `3d61dde` fix(infra): myriad_watcher silent-miss bug + Exp 5 axis-2 causal evidence
   - `35784b9` analysis(stage1): hero claim bootstrap CI — /stress W1 partial defuse
   - `4cfc12f` analysis(stage4): axis-2 per-task fragility — /stress W2 defused
   - `5b6c5f0` exp3(stage4): logit lens — axis-2 IS in output distribution
   - `30e7488` exp1(stage4): Axis-2 prompt-family layer profile — three-axis hierarchy found
   - `55afbf3` docs(paper §5): mechanism prose v1 + axis-1/axis-2 dissociation finding
   - `9269d63` analysis(stage3): H-d-red done — Stage 3 reddit 2x2 mechanism closed
   - `03c4a22` analysis(stage4): P4 cls reverse-tier H1 done — selection-bias defended

If a claim in the paper cannot be traced to a specific file / line / number, that itself is a finding ("Claim X is unsourced").

## Output format

Write to stdout (will be captured to `docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md`).

Open with a one-sentence verdict on current paper state. Then these sections (use markdown headers):

### 强 claims (don't break under attack) / Strong claims
1-3 things that survive hostile reading. Quote specific paper lines + evidence files. Calibrates author so they don't over-correct toward weaker claims.

### 弱 claims (would tank under attack) / Weak claims
For each weak claim:
- Quote the **exact** paper line or claim (file:line)
- State the attack (what a reviewer would write in their review form)
- State what specific evidence would defuse the attack
- Effort estimate (hours / days / weeks of work)
- 中文为主双语 for attack reasoning, English for technical specifics

### 诚实缺口 (not weak, just missing) / Honest gaps
Things NOT in the paper that a reviewer would expect to see. Distinguish from weak claims — gaps are absences (no evidence shown), weakness is presence-but-fragile (evidence shown but insufficient).

### Distance to top-tier
- **Current tier**: what conference this would accept at today (workshop / mid-tier conf / top-tier conf / top-tier journal)
- **Specific blockers**: list 2-4 concrete missing pieces of evidence + which paper claims they block
- **Unblock plan**: per blocker, what experiment/analysis/prose would defuse + honest effort estimate
- **Submission-today probability**: if author submits today to NeurIPS / ICML / ACL main, your reviewer-confidence on accept (be specific, e.g., "0.1-0.25 — would reject with reviewer-3 score 4/10"). Don't be polite.

### One thing to fix tonight (1-3 hour leverage move)
If author is in death-march mode with 1-3 hours left, the **single highest-leverage move**. Be concrete (specific experiment, specific file, specific claim to rewrite).

## Mental backdrop — typical attack lines for this subfield

These are **starting points** for adversarial reading, not a checklist. Read the data and find new attack lines the evidence itself suggests:

- **Single-family illusion**: "you tested Qwen3-VL-4B + Qwen3-VL-235B-A22B — same family, same pretraining lineage. R5 framing says you don't claim family-independent generalization, fine, but then your mechanism findings (axis-1/axis-2/axis-image) are *Qwen-specific*. Why is this NeurIPS-worthy?"
- **Cherry-picked layer**: "you report L17 patching peak + L23 cosine peak — what about L8, L12, L20? Full-37-layer profile?"
- **Aggregate mean hides task-level chaos**: "24-task patching mean — per-task distribution?"
- **Mechanism necessary ≠ sufficient**: "axis-2 patching displaces output at L11-L17 — but is this the *only* feature that does so?"
- **Residual stream ≠ causal use**: "cosine separation at L23 is geometric. Patching at L11-L17 is causal. You frame this as a 'novel finding'. But it's just the difference between read-out and write-in — well-known in mechinterp"
- **Output amplification trivial**: "10-25x cosine-to-KL amplification by lm_head — bf16 linear projection geometrically must produce something for any small vector; what's your random baseline?"
- **Negative control too easy**: "Cell E random-injection -0.03 vs real -0.33 is 10x — Gaussian noise matched to source variance is a weak baseline. Where's content-matched-but-task-randomized?"
- **Sclar 2024 prompt-format absorbs**: "axis-1 'flat-text triggers shortcut' IS Sclar 2024 prompt format sensitivity on multimodal — what's novel?"
- **Wu et al. tool-calling absorbs method**: "method 4.2 / 4.4 IS Wu et al. tool-calling toolkit renamed — methodological contribution?"
- **Hero status confound**: "P-SoM (SoM-prompt + flat + no-image) vs P-text (DOM-prompt + flat + no-image) — drop-one difference could be ALL prompt — you haven't isolated"
- **48 → 45 cells**: "what failed in the 3?"
- **Boundary peak as 'mechanism'**: "cls marks-like peak at L36 (last decoder layer) — monotonic curve hitting endpoint, not L36 mechanism"
- **Task selection bias**: "strong-tier composite preselects tasks where SoM beats DOM — mechanism findings might be conditional on this filter; was reverse-tier replicated?"
- **Per-cell N for steering**: "Method 4.4 H-mean 0.33 from 45 cells × N tasks each — per-cell N + bootstrap CI?"
- **Hero CI**: "+3.33pp reddit P-SoM hero CI strictly above zero in per-seed bootstrap?"
- **Layer-disjoint claim**: "cosine peak L23 vs patching peak L11-L17 — is this 'two findings' (signature ≠ decision) or does it mean your mechanism story falls apart?"
- **3-axis stack double-counting**: "you have 3 axes (image / text-format / prompt-family) + 3 evidence layers (cosine / patching / logit lens) + 2 sites — but axis-image and axis-text-format share the patching protocol — is there really 3-axis evidence or only 2?"
- **Tokenization feature attribution post-hoc**: "W6 finds first-token markup-sigil predicts L17 peak (2/2 vs 4/4 on 6 marks-like variants). Six is a small sample. Two binary features (sigil-first + integer-marker) can perfectly separate 6 examples by chance. Where's the held-out variant test?"

Read adversarially — find new attack lines the data itself suggests.

## Voice

- Hostile but fair (peer-lab reviewer, not contemptuous)
- Specific — file paths, line numbers, exact numbers (no "could be improved")
- Acknowledge real strength — author needs calibration so they don't over-correct toward weaker claims
- 中文为主双语 — explanation in 中文, technical specifics in English
- No filler praise. No hedging on broken claims.

## What this audit is NOT

- Not a checklist runner
- Not a process audit (NEEDS_BIB, missing citations — separate tools)
- Not a writing coach
- Not a project manager
- Specifically: **adversarial scientific peer who reads the paper cold**

codex
I’ll read only the specified paper/evidence paths and avoid the excluded prior-analysis files. I’m going to anchor findings to exact line numbers, so I’ll first inventory the draft and mechanism files, then pull the specific claim/evidence lines.
exec
/bin/bash -lc 'git log --oneline -8' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
3d61dde fix(infra): myriad_watcher silent-miss bug + Exp 5 axis-2 causal evidence
6e5bdcb docs(chronicle): §127 today's burst + chronicle-on-milestone rule
35784b9 analysis(stage1): hero claim bootstrap CI — /stress W1 partial defuse
4cfc12f analysis(stage4): axis-2 per-task fragility — /stress W2 defused
8c67d8e infra(stress): bilingual Chinese+English output language rule
5546cf1 infra(stress): hostile reviewer audit skill (/stress) — replica + auto-trigger
df8ff72 docs(paper §5/§8): clear stale NEEDS_BIB markers after audit
be07296 docs(paper.bib): add 6 missing bibkeys + clear all NEEDS_BIB markers

exec
/bin/bash -lc 'pwd && rg --files docs/checkpoints/paper_drafts docs/checkpoints/mechanism | sort' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
/home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
docs/checkpoints/mechanism/README.md
docs/checkpoints/mechanism/plan.md
docs/checkpoints/mechanism/results/axis2_layer_profile.md
docs/checkpoints/mechanism/results/axis2_logit_lens.md
docs/checkpoints/mechanism/results/axis2_per_task_fragility.md
docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md
docs/checkpoints/mechanism/results/format_variation_h1_test.md
docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md
docs/checkpoints/mechanism/results/h1_per_task_fragility.md
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md
docs/checkpoints/mechanism/results/layer_axis_emergence.md
docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md
docs/checkpoints/paper_drafts/paper.bib
docs/checkpoints/paper_drafts/section1_intro.md
docs/checkpoints/paper_drafts/section2_background.md
docs/checkpoints/paper_drafts/section3_definition.md
docs/checkpoints/paper_drafts/section4_empirical_findings.md
docs/checkpoints/paper_drafts/section4_limitations_disclosure.md
docs/checkpoints/paper_drafts/section5_mechanism.md
docs/checkpoints/paper_drafts/section8_limitations.md

codex
The repo contains only sections 1-5 and 8 under `paper_drafts`; I don’t see section 6 or 7 draft files in the requested directory. I’ll treat that as an absence in the review rather than silently filling it in.
exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section2_background.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	## 2. Background and Related Work
     2	
     3	### 2.1 Web Agent Observation Modes
     4	
     5	Modern web agents differ less in the browser actions they expose than in the observation representation they give to the language model. Text-only agents typically serialize the Document Object Model or Accessibility Tree (AXTree) into a hierarchical text observation. WebArena uses this style of realistic browser environment to evaluate language-guided agents on shopping, forum, map, and software-development tasks \citep{zhou2024webarena}. Mind2Web similarly frames web interaction as selecting actions from structured page elements collected across real websites \citep{deng2023mind2web}. This line of work makes DOM-derived text the default low-cost representation: it is cheap, symbolic, and compatible with language-only models, but it can be verbose and blind to visual appearance.
     6	
     7	Multimodal web agents add screenshots to the observation. VisualWebArena extends WebArena with visually grounded tasks and evaluates agents that combine page text, screenshots, and visual grounding cues \citep{koh2024visualwebarena}. A common grounding device is Set-of-Mark prompting, introduced by Yang et al. as a way to overlay numbered or speakable marks on image regions so a multimodal model can refer to visual objects by discrete IDs \citep{yang2023som}. SeeAct likewise studies GPT-4V as a generalist web agent and finds that visual understanding must still be paired with reliable action grounding \citep{zheng2024seeact}. Magma pushes the same broad direction into an omni-modal agent foundation model with action grounding and multimodal pretraining \citep{yang2025magma}. Vision-only baselines remove the DOM/AXTree channel and ask the model to act from the screenshot alone; these baselines test whether visual perception can substitute for structured symbolic grounding.
     8	
     9	Across this literature, DOM, SoM, and Vision are treated as orthogonal observation modes. SoM in particular is treated as a multimodal bundle: a marked screenshot plus a text legend that maps mark IDs to elements. The `[SOM_MARKS]` text is normally an auxiliary index for the marked image, not a controlled standalone variable. This convention is the gap our paper targets. We ask what routing behavior emerges when the annotated image is skipped while the remaining factors are held apart: AXTree versus `[SOM_MARKS]` text, DOM versus SoM prompt family, and image-off versus image-on evaluation. The resulting object is not a claim that marked or text-only observations are new artifacts; it is a controlled characterization of the phantom routing space around **Phantom-SoM**.
    10	
    11	### 2.2 Routing in LLM Systems
    12	
    13	Routing has become a standard response to heterogeneous cost and capability. FrugalGPT frames inference as a cascade over multiple LLM APIs, learning when cheaper models can answer and when to escalate to stronger models [Chen et al. 2023]. RouteLLM similarly learns routers from preference data to choose between weaker and stronger LLMs under cost-quality tradeoffs [Ong et al. 2025]. These systems are important precedents for cost-aware inference, but their arms are models. The input representation is usually fixed while the backend model changes.
    14	
    15	Web-agent routing has begun to appear at the modality or expert level. Avenir-Web proposes a Mixture of Grounding Experts for multimodal web agents, using human-experience imitation and expert grounding components to improve long-horizon deployment [Li et al. 2026]. Systems work on multimodal serving also motivates modality-aware scheduling: image-heavy requests create different prefill, encoding, and time-to-first-token bottlenecks than text-only requests, so serving systems can separate resources by modality and stage [Qiu et al. 2025]. These works route over computational pathways, modalities, or grounding experts.
    16	
    17	What is missing is representation-level routing within a single model: selecting between different text formats generated from the same browser state. DOM/AXTree and `[SOM_MARKS]` can contain overlapping element semantics, but their token geometry is different. One is hierarchical, nested, and metadata-rich; the other is flat, indexed, and compact. Prior routing work does not ask whether a single model should see the same page as an AXTree for some tasks and as an isolated marks list for others. Phantom-SoM makes that missing routing axis explicit.
    18	
    19	This distinction also separates our setting from ordinary prompt selection. A representation arm is not just a different instruction template; it changes the observation object that enters the agent loop at every step. If two formats derived from the same browser state route the same model into different exploration policies, then representation becomes a deployable control surface. The router need not choose a larger model or a visual encoder first. It can choose a cheaper textual view of the page, observe whether the trajectory stalls, and escalate only when the cheap representation appears misaligned with the task.
    20	
    21	### 2.3 Prompt-Format Sensitivity in LLMs
    22	
    23	The plausibility of Phantom-SoM rests on a broader fact about language models: semantically equivalent prompts can induce different behavior when their surface form changes. Sclar et al. quantify language-model sensitivity to spurious prompt-format features and show that small formatting choices can produce large accuracy differences, even when the underlying task semantics are unchanged [Sclar et al. 2024]. Mishra et al. show a related effect for instructional prompting: reframing instructions into forms better aligned with a model's learned language can change few-shot performance [Mishra et al. 2022].
    24	
    25	These studies do not study web agents, but they explain why web observations should not be treated as neutral containers. A page serialized as AXTree text is not merely "the same information" as a page serialized as `[SOM_MARKS]`. The model receives different punctuation, ordering, indentation, repeated role tokens, ID patterns, and local neighborhoods. Those tokens prime different latent states and therefore different action distributions.
    26	
    27	For a web agent, prompt-format sensitivity matters at the trajectory level. The model is not producing a single label; it is choosing whether to search, click, scroll, revisit a page, or finish. Section 4 and Section 5 build on this theoretical anchor: the flat marks list tends to shift exploration toward quick element selection, while AXTree hierarchy tends to support sustained navigation and search. Prompt wording also matters, but our two-knob account separates the layers: text representation shapes how the agent explores, while prompt family tunes when it commits.
    28	
    29	### 2.4 Cost-Efficient Inference for Web Agents
    30	
    31	Cost-efficient web-agent inference has usually meant pruning or scheduling expensive context. AXTree observations are long, noisy, and security-sensitive. FocusAgent addresses this by using a lightweight retriever to trim AXTree observations before sending them to the main agent, reducing context while preserving the hierarchical representation [Kerboua et al. 2025]. This is a natural text-efficiency strategy: keep the DOM-derived tree, but remove irrelevant lines.
    32	
    33	Multimodal inference adds a second cost source: visual encoding. Image inputs increase prompt-processing time, memory pressure, and time-to-first-token. ModServe characterizes large multimodal model serving and shows that multimodal workloads have heterogeneous stages and resource requirements, motivating modality- and stage-aware resource disaggregation [Qiu et al. 2025]. In web agents, full SoM therefore has two costs: it prepares a marked screenshot and it sends image tokens to the model.
    34	
    35	Phantom-SoM explores a different kind of efficiency. It is not text pruning and it is not image scheduling. It is text reformatting. The `[SOM_MARKS]` list can be generated from the same browser/AXTree metadata already available to the agent, then sent without the marked screenshot. This removes image-token cost while preserving a discrete element index. In our runs the text observation is comparable in token length to the corresponding AXTree (within ±7% on reddit and classifieds, holding the system prompt fixed); the difference is in structure — flat indexed list versus nested hierarchy with url/tab metadata — rather than in length. The open question is whether such a representation is only a structural rewrite of DOM, or whether its format creates a distinct success pool. Our empirical sections answer the latter.
    36	
    37	This matters because many cost reductions trade away information: smaller models, shorter context windows, lower image resolution, or fewer retrieved lines. Phantom-SoM instead tests whether a cheap re-arrangement of already available text can expose a different reasoning path. If it succeeds on tasks missed by DOM, the gain is not merely compression; it is complementarity. That is why Section 4 reports both single-mode success and drop-one oracle value rather than treating token savings alone as the contribution.
    38	
    39	### 2.5 Position of This Work
    40	
    41	This paper positions Phantom-SoM at the intersection of four literatures that are usually studied separately. First, SoM and its descendants, including Magma and Ferret-UI 2, use marks as visual grounding devices and generally keep text tied to the marked image \citep{yang2023som,yang2025magma,li2025ferretui2}. Second, web-agent benchmarks compare DOM, SoM, and Vision modes, but do not use the mark-text-without-image condition as a controlled axis for routing characterization \citep{zhou2024webarena,koh2024visualwebarena,zheng2024seeact}. Third, routing systems optimize over models, modalities, or experts, not over text formats of the same browser state [Chen et al. 2023; Ong et al. 2025; Li et al. 2026]. Fourth, prompt-format work predicts sensitivity to representation syntax, but has not measured task-pool complementarity in interactive web agents \citep{sclar2024promptformat,mishra2022reframing}.
    42	
    43	There are important artifact precedents, and this paper treats them as context rather than as targets to out-claim. SoM-Mark already pairs textual mark references with visual marks \citep{yang2023som}; SeeAct explores marked-screenshot web-agent grounding \citep{zheng2024seeact}; and Magma incorporates related SoM-style and action-grounding ideas into an omni-modal agent model \citep{yang2025magma}. These systems show that marked observations, textual references, and multimodal action grounding are not new merely as artifacts. The contested point is different: whether the behavior of the image-skipped configurations has been scientifically isolated, compared against DOM/SoM/Vision on identical task pools, and explained mechanistically.
    44	
    45	The resulting gap is therefore a characterization gap, not a first-deployment gap. Published systems and benchmarks have not, to our knowledge, provided a controlled scientific evaluation of the phantom boundary: `[SOM_MARKS]` isolated from the screenshot, crossed with prompt family in a 2-by-2 control, compared through low-overlap success pools and drop-one oracle value, and checked against mechanistic controls including mid-layer L11-L17 evidence and random injection. This is also where the 4-fold drop-in property belongs. It is an empirical finding about a controlled configuration: zero image tokens, no new model, usable routing signal, and positive incremental oracle value. It is not a claim that text-only browser control or marked observations were first introduced here.
    46	
    47	Our contributions follow directly. We define **Phantom-SoM** as the deployment-relevant representative of the phantom routing space: marks text plus SoM prompt family, with the image removed. We show empirically that Phantom-SoM is not a degenerate DOM surrogate: it contributes independent oracle value and has substantial task-pool non-overlap with DOM, SoM, and Vision. The P-text and P-prompt controls establish specificity by separating text-payload flattening from prompt-family effects, rather than attributing all gains to a single novel arm. Finally, we provide mechanism evidence for a two-knob account: representation format shapes exploration, while prompt wording tunes commitment confidence. This motivates the experimental design in Section 3 and the controlled evidence in Section 4.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section1_intro.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	## 1. Introduction
     2	
     3	Web agents act through representations. A browser state can be serialized as a DOM or Accessibility Tree, shown as a screenshot, or annotated with Set-of-Mark (SoM) labels that connect visible regions to discrete element IDs. Existing benchmarks and agents treat these as different observation modes: WebArena and Mind2Web popularized DOM-derived text for realistic web tasks, while VisualWebArena and SeeAct introduced visually grounded settings where screenshots and action grounding become central \citep{zhou2024webarena,deng2023mind2web,koh2024visualwebarena,zheng2024seeact}. Set-of-Mark prompting was designed for this multimodal setting: a marked image is paired with a textual legend so the model can refer to visual objects by number \citep{yang2023som}. Later multimodal-agent systems, including SeeAct and Magma, further explored marked-screenshot and omni-modal action-grounding paradigms rather than treating mark text as an isolated scientific variable \citep{zheng2024seeact,yang2025magma}. These are important precedents. We therefore do not claim to be first to deploy text-only, marked, or SoM-style observations. Our claim is about controlled characterization: isolating what changes when the annotated image is skipped while the text payload and prompt family are varied under the same task, model, and evaluation protocol.
     4	
     5	This paper questions that bundling assumption as an experimental object. We characterize the **phantom routing space**: configurations on the "skip annotated image" boundary that retain some SoM-derived textual or prompt structure while removing the image. Its deployment-relevant representative is **Phantom-SoM**: the agent receives the SoM prompt and the `[SOM_MARKS]` textual element list, but no image. The structural controls are **P-text** (the `[SOM_MARKS]` text under the DOM prompt) and **P-prompt** (the SoM prompt over AXTree text). At the start of this project, Phantom-SoM looked like a broken ablation. The natural expectation was that removing the marked screenshot would collapse SoM into either a weak DOM surrogate or a nonsensical configuration: the prompt still suggests visual marks, but the visual substrate is absent. The data reject that expectation. Phantom-SoM solves tasks that DOM, full SoM, and Vision all miss, and on B0 reddit it matches or modestly exceeds full SoM by adjusted SR (**13.81% vs 10.48%, N=210**; the gap is within 2σ under the run-to-run variability we observe in same-condition repeats), while avoiding image-token cost. On classifieds, full SoM remains clearly stronger (**21.37% vs Phantom-SoM 14.53%, N=234**), the expected sanity check when marked screenshots carry real visual information.
     6	
     7	Our first contribution is a controlled scientific evaluation of this phantom boundary. Across completed B0 VisualWebArena classifieds and reddit runs, we compare DOM, full SoM, Vision, and Phantom-SoM on the same task sets (**N=234 classifieds; N=210 reddit; same-task adjusted SR**) and use the P-text/P-prompt controls to test whether the effect collapses to one prompt trick or one text-format swap. Phantom-SoM is not the best single arm on every site, and we do not claim that it replaces full SoM. Its value is complementarity. Its task-success pool has low overlap with the established modes, with Jaccard similarity in the roughly **0.29-0.49** range against other arms, and its removal reduces the oracle. The principal hero metric is therefore the **drop-one oracle**, not the single-mode SR difference: Phantom-SoM contributes **3.33 percentage points** of incremental oracle value on reddit with a per-task-bootstrap 95% CI of **[+0.95, +6.19]** strictly above zero (P(Δ>0)=0.998, B=10000 task resamples), comparable to full SoM at +1.90pp [+0.48, +3.81], and **2.56 percentage points** on classifieds with CI [+0.85, +4.70] strict positive. Phantom-SoM consistently sits within the top routing-value arms despite using no image. The same bootstrap on the head-to-head reddit single-mode comparison (Phantom-SoM 13.81% vs full SoM 10.48%) gives a marginal CI [-0.95, +7.62] that crosses zero (P(diff>0)=0.914), which is exactly the "within 2σ" caveat above; we therefore frame the head-to-head SR contrast as competitive parity, and let the strictly-positive drop-one oracle carry the deployment-relevant claim. Crucially, the cost of obtaining this configuration is essentially the cost of the DOM baseline: the `[SOM_MARKS]` block is produced by a regex pass over the same accessibility-tree text the DOM agent already consumes (interactive elements come pre-numbered as `[N] role 'label'`), so a deployment that can run DOM can run Phantom-SoM by changing what it forwards to the model: no bounding-box pipeline, no marked image, no extra inference modality. We therefore preserve the empirical **4-fold drop-in property** as the paper's practical finding: cost approximately DOM, lower image-stage latency, usable routing signal, and positive drop-one oracle value.
     8	
     9	The cross-site asymmetry is itself informative: full SoM clearly outperforms Phantom-SoM on classifieds, where visually rich product listings make layout and appearance important, but not on reddit, where post and comment threads are more text-dominated. This is consistent with marked screenshots being most useful when visual grounding is task-critical.
    10	
    11	Our second contribution is a mechanism account for why the ablation works. A 2-by-2 reddit ablation separates text representation from prompt family: DOM prompt versus SoM prompt, crossed with AXTree versus `[SOM_MARKS]` text. On the verified same-task subset (**N=48**), replacing AXTree with `[SOM_MARKS]` shifts exploration away from DOM-like search loops and toward Phantom-SoM-like quick decisions: the search-loop rate is **22.7% for DOM** but **10.8% for both Phantom-SoM and P-text**. The prompt knob appears elsewhere. DOM-prompt arms show the larger raw-to-adjusted false-positive gap (**6.25 pp; 3 N/A false positives**), while Phantom-SoM under the SoM prompt has a smaller gap (**2.08 pp; 1 N/A false positive**). Section 5 adds mid-layer mechanistic evidence: activation-patching effects concentrate in the L11-L17 region, and the random-injection control distinguishes content-specific phantom information from generic perturbation. The resulting two-knob view is simple: **text representation shapes how the agent explores; prompt wording tunes when it commits**. This aligns with prior evidence that language models are highly sensitive to prompt format \citep{sclar2024promptformat,mishra2022reframing}, but extends the claim from static prediction to multi-step web-agent trajectories.
    12	
    13	We evaluate on VisualWebArena classifieds and reddit with B0, an API-backed Qwen3-VL-235B agent, and use B1, a local Qwen3-VL-4B model, as a cross-capability robustness check rather than a separate scientific claim. Following the pre-registered R1-R5 framing rules, the scope of this paper is Qwen-family VisualWebArena characterization and explanation, not a claim of universal model-family generalization and not yet a learned deployment router. Routing exploitation is deferred to a follow-up paper. Section 2 situates the gap in web-agent representations, SoM, routing, and prompt-format sensitivity. Section 3 defines Phantom-SoM and the measurement protocol. Section 4 reports phantom-space and image-on baseline findings. Section 5 analyzes the mid-layer activation-patching mechanism. Section 6 discusses generalization, and Section 7 summarizes limitations and implications.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section3_definition.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	## 3. Phantom-SoM: Definition and Ablation Setup
     2	
     3	### 3.1 Set-of-Mark Bundle
     4	
     5	Set-of-Mark (SoM) prompting converts a screenshot into an indexed visual interface. The standard bundle has two synchronized parts: a marked image, where page regions are overlaid with bounding boxes and numeric IDs, and a text legend that maps those IDs to element descriptions [Yang et al. 2023]. We serialize the text component as:
     6	
     7	```text
     8	[SOM_MARKS]
     9	[id=N] role 'label'
    10	...
    11	[/SOM_MARKS]
    12	```
    13	
    14	Full SoM gives both pieces to the agent at the same step. The prompt says the `[SOM_MARKS]` list and annotated screenshot refer to one another, and the action schema asks the model to click, type, or select by `element_id` when possible. VisualWebArena and SeeAct use the same broad pattern: visual evidence is paired with grounding information so the model can convert perception into browser actions [Koh et al. 2024; Zheng et al. 2024].
    15	
    16	This bundle is the assumption Phantom-SoM ablates. The question is not whether marked screenshots are useful; Section 4 shows that they often are. The question is whether the text half of the bundle is only an image key, or itself a distinct text representation.
    17	
    18	### 3.2 Phantom-SoM
    19	
    20	We define **Phantom-SoM** as:
    21	
    22	```text
    23	Phantom-SoM(page) =
    24	  prompt = SoM prompt
    25	  text   = SOM_MARKS(page)
    26	  image  = None
    27	```
    28	
    29	Phantom-SoM uses the same SoM prompt family as full SoM and the same `[SOM_MARKS]` text, but removes the page screenshot passed to the model. In code, `p79/experiment/som.py::prepare_observation_for_mode` handles `mode in ("phantom_som", "phantom_dom", "phantom_text")` by calling `_build_som_result(...)`, then returning the generated `som_text` with `marked_image=None` (`phantom_dom` is the legacy mode value retained as alias for paper-grade run dirs; `phantom_text` is the current canonical name for P-text). The rendered screenshot path is retained for debugging; the model does not receive it.
    30	
    31	The critical property is that the prompt remains the SoM prompt. It still describes an annotated screenshot with numbered boxes, even though the observation channel contains no page screenshot. We call this the **mirage prompt** property: the behavioral scaffold of SoM is preserved while the visual substrate is removed.
    32	
    33	Phantom-SoM is a cost intervention, and the structure of the saving is best stated relative to two different baselines.
    34	
    35	**Relative to DOM**, Phantom-SoM is essentially free. The `[SOM_MARKS]` block is produced by a regex filter over the VisualWebArena accessibility-tree text that the DOM baseline already consumes. VWA serializes interactive elements with bracketed numeric IDs of the form `[N] role 'label'`; in our implementation `_extract_text_marks` (see `p79/experiment/som.py`) walks `obs_text` line by line, keeps the lines that match `\[\d+\]`, and returns `(id, label)` pairs that are wrapped in a `[SOM_MARKS] ... [/SOM_MARKS]` block. There is no bounding-box lookup and no image work in this path; bounding boxes are only used by full SoM when drawing numeric labels onto the screenshot. Empirically this leaves text length roughly unchanged: holding the system prompt fixed at the DOM family, median total input is 3437 tokens for DOM versus 3661 for P-text on reddit, and 3008 versus 2948 on classifieds — within ±7% on both sites. The two formats see the same accessibility content; what differs is the surface form (flat indexed list versus nested hierarchy with url/tab metadata). We treat this as a representation property and study its behavioral effect mechanistically in Section 5; for cost accounting the implication is that switching DOM → Phantom-SoM at deployment time costs at most a regex pass over the same observation.
    36	
    37	**Relative to full SoM**, Phantom-SoM saves two real layers of cost. (i) The on-server annotation step that draws numeric labels onto the page screenshot is unique to full SoM and is omitted in a Phantom-SoM deployment; in our research code we retain the marked image on disk for debugging, which is why both modes report ~30 ms median obs-prepare latency, but a production variant skips the draw entirely and recovers roughly 30 ms and on the order of $2e-5 per step. (ii) The marked screenshot is no longer encoded as image tokens at inference, removing the visual-encoding stage. Comparing step-level `tokens.input` medians between full SoM and P-text gives a same-prompt image-channel estimate of 733 input tokens per step on reddit (SoM 4275 versus P-text 3542; P-text partial live run, 145 episodes) and 1064 on classifieds (4034.5 versus 2970.5; 234 episodes). We attribute this median gap to the marked screenshot under our backend tokenization. These are the tokens that drive prompt-processing time, memory pressure, and time-to-first-token in multimodal serving (see Section 2.4); skipping them is the dominant component of the cost difference between full SoM and Phantom-SoM.
    38	
    39	The combined picture is that Phantom-SoM sits at roughly DOM cost (its observation is a text filter of the same AXTree) while replacing the visual-evidence half of SoM with nothing at all. This is also a deployment-level claim, not only an analytical one: an existing full-SoM agent can be converted into a Phantom-SoM agent by changing only what the server forwards to the model — keep the `[SOM_MARKS]` text that is already being produced from the accessibility tree, stop drawing labels onto the screenshot, and stop attaching the marked image to the inference request. The model interface, the prompt, the action schema, and the evaluator are unchanged. There is no retraining, no new data path, and no marks-side prompt edit; the only mutation is on the backend annotation pipeline, after the AXTree filter and before the model call. We use this property in Section 4 to interpret cost-versus-success comparisons as deployment-time tradeoffs rather than research-only configurations, and in Section 5 to argue that Phantom-SoM's behavior is a property of the format the model already saw inside SoM, not an emergent capability that requires new infrastructure.
    40	
    41	### 3.3 P-text
    42	
    43	**P-text** is the disambiguation ablation:
    44	
    45	```text
    46	P-text(page) =
    47	  prompt = DOM prompt
    48	  text   = SOM_MARKS(page)
    49	  image  = None
    50	```
    51	
    52	Its observation is identical to Phantom-SoM: `[SOM_MARKS]` text only, no page screenshot. The only intended change is the system prompt. In both B0 (`p79/agents/proxy_api_agent.py`) and B1 (`p79/agents/qwen3vl_agent.py`), `_system_prompts["phantom_som"]` maps to the SoM prompt, while `_system_prompts["phantom_dom"]` (and the alias `_system_prompts["phantom_text"]`) maps to the DOM prompt. For `som`, `phantom_som`, `phantom_dom`, and `phantom_text`, the agent passes through the `[SOM_MARKS]...[/SOM_MARKS]` text directly.
    53	
    54	This cell separates representation from prompt wording. If P-text behaves like Phantom-SoM, the flat marks text is driving behavior. If it behaves like DOM, the prompt is doing more of the work.
    55	
    56	### 3.4 The 2x2 Ablation Matrix and Excluded Hybrid
    57	
    58	The core ablation is a prompt-by-representation matrix:
    59	
    60	| | DOM prompt | SoM prompt |
    61	|---|---|---|
    62	| AXTree obs | DOM | *excluded — see below* |
    63	| `[SOM_MARKS]` obs | P-text | Phantom-SoM |
    64	
    65	Full SoM is adjacent to this 2x2: it uses the SoM prompt, the same `[SOM_MARKS]` text, and the marked screenshot. Vision is a separate screenshot-only baseline.
    66	
    67	The fourth cell — AXTree observation paired with the SoM prompt — is intentionally excluded from Paper 1 because it is not a self-consistent design point. The SoM system prompt instructs the agent to interact via `[SOM_MARKS]` IDs (e.g. `click [42]` referring to the SoM-marked element 42), but AXTree text uses an independent accessibility-tree ID space; an action like `click [42]` becomes parsing-ambiguous when the two ID systems do not match. This hybrid mode (i) has no clean LLM mechanism, (ii) confounds the prompt-effect ablation with mismatched-ID parsing failure, and (iii) does not reduce token cost relative to P-text. We treat the 5-mode set (DOM, P-text, Phantom-SoM, full SoM, plus Vision as a separate screenshot-only arm) as the diagonal axis-by-axis path through the 2×2×2 (text-payload-structure × prompt × image) design cube; the four mismatched-prompt-representation hybrids are excluded for the same reason.
    68	
    69	Each contrast isolates a different factor:
    70	
    71	- **DOM vs P-text** holds the prompt family fixed at DOM and changes the text-payload structure from AXTree to `[SOM_MARKS]`.
    72	- **Phantom-SoM vs P-text** holds the text observation fixed and changes only the prompt family.
    73	- **Full SoM vs Phantom-SoM** holds prompt and marks text fixed and adds the implemented marked-image channel.
    74	- **Full SoM vs DOM** measures the combined effect of SoM prompt, marks text, and marked screenshot relative to the standard text baseline.
    75	
    76	The 2x2 is not a routing policy by itself. It is a causal scaffold for Section 5: text-payload structure shapes exploration, while prompt wording tunes commitment confidence. Section 6 promotes this scaffold to a token-monotonic cascade — DOM → P-text (axis 1, text-payload swap, no token increase) → Phantom-SoM (axis 2, system-prompt swap, no data-token increase) → full SoM (axis 3, image embedding cost) — so a routing trigger never has to "add then remove" tokens.
    77	
    78	### 3.5 Implementation and Measurement Protocol
    79	
    80	All SoM-derived conditions share the same text-marks extractor. `_extract_text_marks` reads `obs_text` (the VisualWebArena accessibility-tree serialization the DOM baseline already uses) line by line, keeps each line whose label matches `\[\d+\]`, and produces `(id, label)` pairs up to a configured cap. `_build_som_result` then wraps those pairs in a `[SOM_MARKS] ... [/SOM_MARKS]` block. This text path **does not require bounding boxes**: the IDs come from the accessibility tree, not from a separate vision pipeline. Bounding boxes are only consulted by full SoM, which uses `obs_nodes_info` to draw numeric labels onto the page screenshot. Phantom-SoM and P-text reuse the exact `[SOM_MARKS]` text and drop the page screenshot; Marks are not re-filtered specifically for Phantom, and the source page state is unchanged.
    81	
    82	Reference images supplied by a task configuration are separate from the observation mode. These task-provided target images are passed to all modes as task input; Phantom-SoM removes only the current-page browser screenshot.
    83	
    84	Each episode starts from `environment.reset(task.config_file)`, and paper-grade condition comparisons use freshly reset site state to avoid cross-condition contamination. The April 27 Magento base-url/auth fix addressed an unrelated shopping-state reliability issue; this paper uses completed classifieds and reddit runs under the reset protocol.
    85	
    86	When comparing arms, we use same-task subsets: a task contributes only when the relevant conditions have completed it. We report **adjusted SR**, which starts from raw evaluator success and removes `na_fp` for not-applicable tasks that appear correct without agent-initiated finish, and `eval_fp` for evaluator matches caused by ineffective or non-finished trajectories. Section 4 reports results under these conventions; Section 5 uses the same traces for mechanism analysis.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section4_empirical_findings.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# 4. Empirical Findings
     2	
     3	This section reports empirical evidence that web-agent observation representations should be treated as routing arms, not only as fidelity levels. The key surprise is that **Set-of-Mark text alone**, with the marked screenshot removed, does not collapse to a DOM-like baseline. Instead, it behaves as a distinct text-only arm whose successes only partially overlap with DOM, full SoM, and vision-only observations. We refer to this arm as **Phantom-SoM**: the agent receives the `[SOM_MARKS]` textual element list and the SoM-style prompt, but no image.
     4	
     5	Throughout this section, we distinguish three measurement conventions. **Raw SR** is the evaluator success rate in `condition_summary_v2.json`. **Adjusted SR** subtracts confirmed false-positive terminal answers on not-applicable or evaluator-mismatch tasks. **Same-task adjusted SR** uses the same task set for all arms within a site. Unless otherwise noted, claims use same-task adjusted SR on completed B0 VisualWebArena classifieds and reddit runs. We also treat small cell-to-cell differences cautiously: under same-condition repeats, we observe roughly **+/-5% task-set variance**, so individual differences below about **2 pp** should be interpreted as noise-floor evidence rather than stable rankings.
     6	
     7	## 4.1 Setup
     8	
     9	We evaluate a single strong API-backed web agent, denoted **B0**, on two completed VisualWebArena sites: classifieds and reddit. The completed B0 pool contains **234 classifieds tasks** and **210 reddit tasks** for each reported observation condition:
    10	
    11	| Arm | Observation | Prompt family | Image input | Intended contrast |
    12	|---|---|---|---|---|
    13	| DOM | AXTree / DOM-derived text | DOM | No | Hierarchical text baseline |
    14	| SoM | `[SOM_MARKS]` text plus marked screenshot | SoM | Yes | Full Set-of-Mark baseline |
    15	| Vision | Screenshot without SoM marks | Vision | Yes | Visual-only baseline |
    16	| Phantom-SoM | `[SOM_MARKS]` text only | SoM | No | Isolated marks-text representation |
    17	| P-text | `[SOM_MARKS]` text only | DOM | No | Prompt-family control for marks text |
    18	
    19	The first three arms are the original Phase 1 representation baselines. Phantom-SoM is the new ablation arm. P-text is a prompt-family control: it receives the same marks-text-only observation as Phantom-SoM but uses the DOM prompt. We report all five modes for descriptive SR, cost, and latency. For the main routing-value claim, we keep the primary drop-one oracle on the four-arm comparison used throughout the paper: DOM, SoM, Vision, and Phantom-SoM.
    20	
    21	The original intuition was that Phantom-SoM should be either a broken SoM configuration or a weak DOM surrogate: it keeps a prompt that says the agent is operating with marked visual context, but removes the marked screenshot. The empirical results reject that collapse story. Phantom-SoM is lower than full SoM on classifieds, where marked screenshots carry clear visual grounding value, but it matches or modestly exceeds full SoM on reddit under adjusted SR.
    22	
    23	## 4.2 Single-Mode SR, Cost, and Latency
    24	
    25	The single-mode success rates show a site-modulated effect. On classifieds, full SoM remains the strongest individual representation. On reddit, Phantom-SoM is at least competitive with the strongest baselines, while using no image input. The table reports adjusted SR, because Figures 1, 2, 7, and 8 use episode-level `adjusted_success` for the paper comparisons. The latency column is p95 step latency from `condition_summary_v2.json`; cost is average total cost per task.
    26	
    27	| Site | Arm | Adjusted SR | Avg cost | p95 step latency | Metric |
    28	|---|---|---:|---:|---:|---|
    29	| Classifieds | DOM | 14.10 | $0.043 | 37.5s | N=234 |
    30	| Classifieds | SoM | **21.37** | $0.042 | 74.0s | N=234 |
    31	| Classifieds | Vision | 13.68 | $0.025 | 45.0s | N=234 |
    32	| Classifieds | P-text | 14.53 | $0.040 | 12.8s | N=234 |
    33	| Classifieds | Phantom-SoM | 14.53 | $0.044 | 18.2s | N=234 |
    34	| Reddit | DOM | 9.52 | $0.052 | 73.6s | N=210 |
    35	| Reddit | SoM | 10.48 | $0.041 | 58.9s | N=210 |
    36	| Reddit | Vision | 6.67 | $0.023 | 55.6s | N=210 |
    37	| Reddit | P-text | 11.90 | $0.046 | 58.1s | N=210 |
    38	| Reddit | Phantom-SoM | **13.81** | $0.038 | 51.4s | N=210 |
    39	
    40	The classifieds result is the expected sanity check: when tasks benefit from visual page layout and product imagery, the marked screenshot adds useful grounding and full SoM is clearly best (**SoM 21.37 vs Phantom-SoM 14.53; N=234; adjusted**). Phantom-SoM is close to DOM on classifieds (**14.53 vs 14.10**), but this is not a dominance claim; it is inside the noise floor and far below full SoM.
    41	
    42	The reddit result is the counterintuitive case. Removing the image does not eliminate the value of the SoM representation: Phantom-SoM matches or modestly exceeds full SoM and DOM on adjusted SR (**13.81 vs SoM 10.48 vs DOM 9.52; N=210; adjusted**). Given the variance we observe in repeats, the **+3.33 pp** gap over SoM is near the boundary of what should be treated as stable. We interpret this as evidence that Phantom-SoM is competitive on text-dominated reddit threads, not as an unconditional single-cell dominance claim. The more robust pattern is the cross-site asymmetry: **classifieds favors full SoM; reddit does not**. We treat that asymmetry as mechanism evidence rather than a setup bug: Section 5 shows a related site-modulated capability shift, with B0-to-B1 SoM visual-hijack/click-loop increasing by **+50.0 pp** on classifieds and **+33.3 pp** on reddit.
    43	
    44	This pattern suggests that the `[SOM_MARKS]` list is doing more than serving as a caption for a screenshot. It is a compact, flat, indexed text representation. Compared with AXTree-style DOM text, it removes much of the hierarchical nesting and metadata, and presents candidate actions as a linear set of marked elements. The outcome is not uniformly better, but it can push the agent toward a different solution basin.
    45	
    46	The cost and latency columns make the routing tradeoff concrete. On classifieds, Phantom-SoM's average cost is effectively in the same band as DOM and SoM (**$0.044 vs $0.043 vs $0.041**), but its p95 step latency is much lower than full SoM (**18.2s vs 74.0s**, roughly 4x faster). On reddit, Phantom-SoM is the cheapest of the main text/SoM-style arms (**$0.038 vs SoM $0.041 vs DOM $0.052**) and remains faster at p95 step latency than full SoM (**51.4s vs 58.9s**). These numbers support the cost-aware routing interpretation in Figures 7 and 9 without requiring Phantom-SoM to win every site.
    47	
    48	Raw SR tells the same high-level story but should not be mixed with adjusted SR. Some arms lose points after false-positive adjustment. Because the paper claim concerns deployable task success rather than answer attempts that only appear correct under a noisy evaluator, we use adjusted SR for the main empirical comparisons.
    49	
    50	## 4.3 Drop-One Oracle
    51	
    52	Single-mode SR can hide routing value. A representation may have modest average SR while still solving tasks that the other arms miss. We therefore compute a drop-one oracle: form the oracle union over the four primary arms, remove one arm, and measure how much oracle SR falls. This loss is the arm's incremental contribution to the routing pool.
    53	
    54	| Site | Largest loss | Second | Third | Fourth | Metric |
    55	|---|---:|---:|---:|---:|---|
    56	| Classifieds | SoM -8.55 pp | Vision -3.42 pp | Phantom-SoM -2.56 pp | DOM -2.14 pp | Drop-one oracle loss, N=234, adjusted |
    57	| Reddit | Phantom-SoM -3.33 pp | DOM -1.90 pp | SoM -1.90 pp | Vision -1.43 pp | Drop-one oracle loss, N=210, adjusted |
    58	
    59	The classifieds oracle is consistent with the single-mode story: full SoM contributes the most unique oracle value, followed by vision. Phantom-SoM still has a non-zero loss (**2.56 pp; N=234**), but the main effect on classifieds belongs to visual grounding.
    60	
    61	The reddit oracle is the stronger routing signal. Phantom-SoM has the largest nominal drop-one loss in the fresh four-arm oracle (**3.33 pp; N=210**), while DOM and SoM each contribute **1.90 pp** and Vision contributes **1.43 pp**. Because these are small absolute task counts, we do not read the ordering as a precise rank claim. The important point is that Phantom-SoM is comparable to the top routing-value arms and is not subsumed by DOM, SoM, or Vision.
    62	
    63	The overlap view supports the same conclusion. In the four-arm oracle, Phantom-SoM contributes a concrete reddit-only set of seven tasks (**7, 15, 36, 94, 157, 162, 167**) and a non-zero classifieds set as well. Two examples illustrate the kind of work this arm is doing. On reddit task 7, Phantom-SoM searched for the cake-recipe post and navigated directly to the OP recipe comment permalink. On reddit task 162, it searched within /f/wallstreetbets, scrolled hot posts, and returned the GIF URL for the retirement-account-versus-brokerage-account prompt. These are not proof of a universal mechanism by themselves, but they make the drop-one value concrete: the arm is adding recoverable successes, not only shifting aggregate percentages.
    64	
    65	The main empirical claim is therefore not that Phantom-SoM dominates the other modes. It does not. The claim is that it is an **independent routing arm**: it opens a distinct task pool at text-only cost, with the strongest relative benefit on the text-dominated reddit site and a clear visual-grounding disadvantage on classifieds.
    66	
    67	## 4.4 Two-Knob Ablation
    68	
    69	The five-mode result raises a confound: is Phantom-SoM useful because of the `[SOM_MARKS]` text representation, or because the SoM prompt changes the agent's confidence and behavior even without an image? P-text separates these factors. The full clean P-text runs are reported above for SR, cost, and latency; for behavioral mechanism, we use the verified same-task reddit subset of **N=48**, where all four cells of the prompt-by-representation ablation were manually checked.
    70	
    71	> **Text format shapes how the agent explores. Prompt wording tunes when the agent commits.**
    72	
    73	The first knob is exploration shape. On the same-task reddit ablation subset, replacing AXTree text with `[SOM_MARKS]` text shifts macro behavior away from DOM-like search loops and toward Phantom-SoM-like quick decisions. The verified search-loop rate is **22.7% for DOM** but **10.8% for Phantom-SoM and 10.8% for P-text** (**N=48; behavior metric; same-task subset**). The prompt change alone does not pull P-text back to DOM-like exploration. This supports the representation-driven part of the hypothesis: the flat marks list, not only the SoM prompt, changes the trajectory distribution.
    74	
    75	The second knob is commitment confidence. On the same N=48 subset, DOM and P-text have identical raw-to-adjusted SR gaps, while Phantom-SoM has a smaller gap:
    76	
    77	| Prompt family | Arm | Raw SR | Adjusted SR | FP gap | N/A FP | Metric |
    78	|---|---|---:|---:|---:|---:|---|
    79	| DOM prompt | DOM | 18.75 | 12.50 | 6.25 pp | 3 | N=48, raw/adjusted |
    80	| DOM prompt | P-text | 18.75 | 12.50 | 6.25 pp | 3 | N=48, raw/adjusted |
    81	| SoM prompt | Phantom-SoM | 18.75 | 16.67 | 2.08 pp | 1 | N=48, raw/adjusted |
    82	| SoM prompt | SoM | 22.92 | 16.67 | 6.25 pp | 3 | N=48, raw/adjusted |
    83	
    84	The aggregate SR equality should not be overread as task-level identity: equal counts such as 6/48 can occur with different solved-task sets. The robust signal is the false-positive pattern. DOM-prompt arms have the larger false-positive gap (**DOM and P-text: 3 N/A false positives, 6.25 pp gap; N=48**). The SoM-prompt Phantom arm has fewer N/A false positives (**1 N/A false positive, 2.08 pp gap; N=48**). This indicates that prompt wording affects terminal-action calibration: when the model decides it has enough evidence to `finish`.
    85	
    86	The two-knob account reconciles the apparent tension. The representation is the novel routing axis because it changes the agent's default exploration path. The prompt is a secondary but real tuning knob because it changes commitment confidence. Both are needed to explain the ablation. A representation-only story misses the FP gap, while a prompt-only story cannot explain why P-text follows Phantom-SoM rather than DOM on search-loop behavior.
    87	
    88	These findings explain why Phantom-SoM can be valuable despite not winning every single-mode comparison. Routing benefits depend on complementarity, not only average SR. A flat marks list can be worse for tasks that need hierarchy or visual layout, yet better for tasks where the same hierarchy induces over-searching. The practical implication is a cost-aware cascade: try cheap text representations first, use behavioral signals to detect when their exploration is unproductive, and escalate to full SoM when visual grounding is likely to matter.
    89	
    90	## 4.5 Reddit Substrate Behavioral Deep Dive
    91	
    92	The five-mode aggregate above conceals site-specific behavior. Section 4.5 fills in the reddit substrate. The Section 5 mechanism analysis explains *where in the model* the three axes appear; this subsection explains *what the agent does differently on reddit* under each axis swap, using outcome, macro, and micro behavioral evidence rather than residual-stream geometry.
    93	
    94	### 4.5.1 Reddit substrate
    95	
    96	The reddit environment in VisualWebArena is a Postmill-style forum rather than a visually organized marketplace. Its stable information structure is a hierarchy of forums, posts, and comments. The relevant navigation objects are therefore mostly textual: sidebar links to `f/<forum>` pages, post titles, comment-count links, comment permalinks, sort controls, and a global search box. The URL structure mirrors this hierarchy through path-based routes such as `/f/<forum>/<post>/<comment>`, so moving to the right page normally means choosing the right textual object in the forum tree rather than manipulating a visual layout.
    97	
    98	This substrate makes reddit an informative test case for separating the three axes. Images are frequent in the task prompts and in the posts themselves, but their role is usually evidential: an image can identify which post is being discussed, or disambiguate a content clue, but it is not the site's primary navigation affordance. The browser screenshot does not create the forum hierarchy; it only renders it. Conversely, the search box is prominent in the DOM and AXTree, but intrinsic search is not the intended substrate for many tasks. Repeated search is a failure basin: the agent can keep refining keywords while never taking the forum, post, or comment link that would satisfy the evaluator. The mechanism to explain is therefore not simply "text works better than vision." It is that each representation changes which textual affordances become salient enough for the model to commit to.
    99	
   100	### 4.5.2 Axis 1: Text Payload Structure
   101	
   102	Axis 1 changes the observation text from a hierarchical AXTree to a flat `[SOM_MARKS]` list while holding the DOM prompt fixed. On reddit this is the primary substrate-level mechanism. In the AXTree condition, the sidebar and post/comment links are embedded in a deep tree with many roles, containers, headings, and repeated page metadata. The search box is an easy high-level object, so the agent often converts the user intent into a query and then remains inside the search loop. In the flat marks condition, candidate links are serialized as a more uniform indexed action surface. This does not add image information and does not substantially change token budget; it changes the local attention pattern over action candidates. The forum link or comment permalink is no longer buried inside a nested accessibility structure, so the model is more likely to treat it as a clickable route rather than first translating the task into search terms.
   103	
   104	The evidence chain is consistent across dimensions. At the outcome level, adding P-text to the three-mode baseline contributes oracle value even without the SoM prompt or screenshot (Outcome 0c, +P-text +3.21pp single-phantom lift on the current oracle intersection). At the macro level, the whole-run strategy gradient shows the failure basin directly: reddit search-loop rate falls from DOM to Phantom-SoM and then to full SoM (Macro 1c, search-loop 51.90%->35.71%->31.43%). The axis-1-only macro effect is smaller than the compound prompt path, which is expected if flat text mainly changes which page objects are reachable rather than merely changing the action vocabulary. The micro evidence is sharper: DOM versus P-text has low path overlap for a text-only swap (Micro 2a, URL-path Jaccard 0.573), improves target-page reach (Micro 2b, target-hit +3.47pp), and reduces repeated keyword reuse (Micro 2c, max-keyword-repeat -0.633). The click-target view tells the same story: the two modes choose substantially different element sets even before images enter the system (Micro 2a-extra, click-target Jaccard 0.463).
   105	
   106	The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
   107	
   108	Efficiency further constrains the explanation. Because P-text is generated from the same AXTree-derived text source and does not attach a screenshot, the reddit improvement cannot be attributed to paying the visual-token tax (Efficiency 3a, DOM $0.0516/episode versus P-text $0.0459/episode in the site dictionary). Axis 1 is therefore a representation effect: the observation text is rearranged into an indexed list, not enriched with new visual evidence.
   109	
   110	The mechanism is not monotone in every individual task, which is useful because it identifies the boundary condition. Reddit task #81 asks the agent to upvote every PhotoshopBattles post on the current page whose picture contains a cat. DOM succeeds by using both title semantics and button-state feedback: after an upvote, the observation exposes enough state change for the agent to move on to the next cat post. P-text matches DOM through the early actions but then collapses onto the same marked upvote control after the state should have changed. The case is a negative example for a simplistic "flat marks always help" claim. Axis 1 helps when the bottleneck is finding the right navigation object, but flat serialization can remove or weaken action-state cues such as `Upvote` becoming `Retract upvote`. On reddit, the aggregate effect is positive because the dominant failure basin is route discovery through text, not per-button state tracking.
   111	
   112	### 4.5.3 Axis 2: Prompt as a Decision Prior
   113	
   114	Axis 2 holds the flat `[SOM_MARKS]` text fixed and changes only the prompt family from DOM-style interaction to SoM-style marked-element interaction. On reddit this axis is secondary to the substrate shift, but it is the strongest macro driver of search and typing behavior. The SoM prompt asks the model to point at marked elements, which changes the prior over when to keep querying and when to commit to a visible candidate. In practical terms, it makes the agent more conservative about long exact queries and more willing to use marked links, comment anchors, tab focus, or backtracking after a stagnant page.
   115	
   116	The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.
   117	
   118	The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
   119	
   120	The prompt contrast is also visible in the mode-invariant click-target metric. With the text payload held fixed, P-text and P-SoM still have low click-target overlap (Micro 2a-extra, P-text<->P-SoM click-target Jaccard 0.484). This matters because it rules out a purely verbal explanation in which the SoM prompt only changes confidence wording at `finish`. The prompt changes which marked objects are selected during navigation.
   121	
   122	Reddit task #7 is the cleanest case study. The task asks for the permalink to the original poster's recipe comment for an image post. DOM overfits the visual description into a long exact query about a cake with cranberries and rosemary, spends 30 steps cycling through empty or unhelpful search results, and never reaches the comment permalink. Phantom-SoM instead treats the task as finding an OP recipe comment, searches more broadly for `cake recipe`, and reaches the comment permalink in five steps. The important point is not that the image is absent; both traces must infer from the same task context. The prompt shifts the query breadth and the commitment target, and the marked-comment affordance gives the model a terminal object that a search-loop policy fails to use.
   123	
   124	### 4.5.4 Axis 3: Image as Weak and Bidirectional Evidence
   125	
   126	The image axis adds the marked screenshot to Phantom-SoM, yielding full SoM. On reddit the net effect is weak and bidirectional because images are mostly content clues, not navigation structure. When the task requires recognizing the depicted post, the screenshot can help identify a candidate. But the same screenshot can over-anchor the agent on image URLs or visually salient marked regions, especially when the evaluator requires a post page, comment page, or action state rather than the image asset itself.
   127	
   128	The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
   129	
   130	The image-axis micro contrast confirms that the screenshot is behaviorally strong even when it is not outcome-positive. P-SoM and SoM have low URL-path agreement and frequent immediate divergence (Micro 2a, image-axis URL-path Jaccard 0.456; Micro 2f, early divergence 95.24%). Thus "weak" should be read as weak net value on this substrate, not as weak causal force. The screenshot changes decisions; on reddit, those changed decisions often point to content assets rather than evaluator-relevant post or comment routes.
   131	
   132	Reddit task #0 illustrates the harmful channel. The task asks for the sushi-platter post and its comments section. Phantom-SoM initially clicks the sushi image URL several times, but it eventually recovers and selects the actual post URL `/f/food/82896/i-ate-sushi-platter`, then targets the comment link. Full SoM remains trapped for the full budget, alternating between the sushi image URL and the forum page. The screenshot correctly identifies the sushi platter; the failure is not visual ignorance. The failure is action-policy over-anchoring on the marked image element, where visual salience suppresses the neighboring post/comment route. This is exactly why reddit should be described as text-dominated rather than image-free: image evidence exists, but its marginal value depends on whether it guides the agent toward a forum route or into an asset-level loop.
   133	
   134	The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
   135	
   136	### 4.5.5 Compound Axis 1+2: Phantom-SoM Versus DOM
   137	
   138	The compound transition from DOM to Phantom-SoM combines the flat text payload with the marked-element prompt while still avoiding the image channel. On reddit this compound arm is best interpreted as a complementary routing arm rather than as a uniformly stronger baseline. Its adjusted SR is modestly higher than DOM on the full 210-task per-mode summaries (Outcome 0a, P-SoM 13.81% versus DOM 9.52%, +4.29pp), but the more robust mechanism evidence is the divergence in which routes it explores. DOM and P-SoM have low click-target overlap (Micro 2a-extra, compound click-target Jaccard 0.421) and low URL-path overlap (Micro 2a, compound URL-path Jaccard 0.481). The oracle result then follows naturally: P-SoM adds tasks that the original three-mode set misses (Outcome 0c, +P-SoM +2.56pp), and P-text/P-SoM are not redundant with each other (Outcome 0d, Jaccard 0.500).
   139	
   140	This interaction also explains why Section 4's evidence catalog should not be read as a single leaderboard. DOM, P-text, and P-SoM can have close aggregate success rates while visiting different pages, selecting different marked elements, and failing in different basins. Axis 1 exposes alternative textual routes; Axis 2 changes the policy's willingness to commit to those routes. Their combination is useful because it changes the task pool available to a router, not because it dominates every individual task.
   141	
   142	The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section5_mechanism.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# 5. Mechanism Interpretability
     2	
     3	## 5.1 Overview and Theoretical Framing
     4	
     5	Why does Phantom-SoM sometimes achieve DOM-like cost while retaining part of the SoM signal? The mechanism evidence points to a phantom routing space in the residual stream: when the model receives flat Set-of-Mark text without the annotated image, it does not simply collapse to DOM. Instead, it occupies a mode whose text-axis geometry is close to DOM/P-text and whose image-axis geometry remains separated from full SoM.
     6	
     7	This section is the Zoom-4 layer of the paper's four-level account. Zoom 1 defines the architectural intervention, "skip the annotated image"; Zoom 2 measures the behavioral axes of text payload, prompt family, and image presence; Zoom 3 links the observed behavior to Mirage-style no-image visual reasoning and prompt-format sensitivity; Zoom 4 asks where the resulting mode is represented and whether it is causally used by the model. We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs.
     8	
     9	The analysis builds on the linear-readable and steerable circuit framework of Wu et al., which uses mode means, PCA geometry, and mean-difference activation steering to study tool selection, and on work showing middle-layer cross-modal information flow in VLMs \citep{wu2026toolcalling,kaduri2024whatsintheimage}. Our setting is not a replication of those papers. It is a multimodal web-agent application of the same representation-level question: whether a behaviorally useful routing arm is linearly readable, partially steerable, and causally active inside the model.
    10	
    11	Four mechanism claims organize the evidence. First, observation modes are geometrically separable in the residual stream: the PCA cosine-gap analysis reaches AUROC 1.0 on the main mode contrasts. Second, Phantom-SoM is a mid-layer text-axis sibling of DOM/P-text, not an image-axis sibling of full SoM. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit. Fourth, the shortcut trigger is flat element-list format rather than a specific token pattern; AXTree hierarchy is the unique format that preserves the early L04 image-axis peak in the aggregate.
    12	
    13	The cross-site evidence stack is deliberately defensive. Per-task H1 fragility shows the dichotomy is an aggregate mechanism rather than a deterministic per-task law. Reverse-tier classifieds runs defend against strong-tier selection bias. Reddit format variation replicates the shortcut direction with cleaner mid-layer peaks. Reddit Method 4.2 replicates the Mirage signature: Phantom-SoM remains close to DOM on the text axis while separating from SoM on the image axis. Paper 1 uses these results for mechanism interpretation only; routing implementation is deferred to paper 2, consistent with the paper-planning scope split.
    14	
    15	## 5.2 Method 4.2: PCA Cosine Gap
    16	
    17	Method 4.2 extracts hidden states from Qwen3-VL-4B B1 runs and compares observation modes by layer. For each mode pair and layer, we compute the cosine gap between hidden-state means, evaluate AUROC by projecting examples onto the mean-difference direction, and summarize per-mode geometry through PCA top-10 variance. The classifieds baseline contains 288 examples, formed from 24 strong-tier tasks, two archived steps, and six modes, over 37 indexed layers.
    18	
    19	The robustness suite passes all five checks in the plan. Label permutation leaves the real AUROC 9.8 standard deviations above the permuted baseline. Per-task analysis is positive for all 24 tasks. Step 2 and step 5 curves are invariant at the mechanism level. The L23 silhouette score is at least 0.5, showing nontrivial clustering. Bootstrap 95% confidence intervals are tight, with widths of roughly 4-15% of the corresponding means.
    20	
    21	The key classifieds snapshot is the L17 cosine-gap table:
    22	
    23	| Pair at L17 | Cosine gap | 95% CI | AUROC |
    24	|---|---:|---:|---:|
    25	| P-SoM <-> P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
    26	| DOM <-> P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
    27	| P-SoM <-> SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
    28	| DOM <-> Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |
    29	
    30	The reddit replication lands the same qualitative geometry. At L17, P-SoM is close to DOM with cosine gap 0.0098 and close to P-text with gap 0.0027, while P-SoM-to-SoM remains much larger at 0.0423 and P-SoM-to-Vision at 0.0457. The DOM-to-Vision image-axis peak is L04 with cosine gap 0.0687 and AUROC 1.0.
    31	
    32	This is the Mirage signature in geometric form. Phantom-SoM is not represented as a weakened image mode. At the mid-layer disruption locus, it is a text-axis sibling of DOM/P-text, while the image-axis distance to full SoM remains large.
    33	
    34	## 5.3 Method 4.4: Mean-Difference Activation Steering
    35	
    36	Method 4.4 tests whether the readable mode direction can be used as a steering direction. For each layer, we form a mean-difference vector between Phantom-SoM-like and DOM-like hidden states, add it to each input at generation time with scaling factor $\alpha$, and evaluate whether the continuation moves toward the target mode while preserving the JSON action envelope. Following HDMI's evaluation vocabulary, reliability is the harmonic mean of completeness and selectivity, not a raw shift rate \citep{khorasani2026hdmi}.
    37	
    38	The v2 sweep covers layers [11, 17, 23, 29, 33, 34] and $\alpha \in [1,2,5,10,20]$, for 45 completed cells in the plan summary. The original L17, $\alpha=5$ smoke result reported H-mean 0.44, but the full sweep lowers that cell to 0.16. The plan records this as a smoke-variance artifact from notes 126/127: a 4-cell smoke was too small to support a sweet-spot claim.
    39	
    40	The strongest full-sweep cell is L33, $\alpha=10$, with H-mean 0.33. Its completeness is 38% and its selectivity is 29%. The layer profile is the important result: mid layers L11-L23 preserve the JSON envelope with 100% selectivity but have low completeness, while late-layer L33 produces the largest shifts but frequently over-steers the continuation out of the expected JSON format.
    41	
    42	This creates a probe-causal dissociation. The mid-layer geometry is cleanly readable and causally implicated by patching, but fixed mean-difference steering is only partially reliable. The 0.33 H-mean is therefore an evidence ceiling for Method 4.4, not a final control result. Section 8 should treat LA-HDMI and SAE feature steering as future work motivated by this ceiling, without claiming that either method has already improved it.
    43	
    44	## 5.4 Stage 2/3: Activation Patching for a Causal Mid-Layer Mechanism
    45	
    46	Activation patching provides the causal test. For each task, the clean/source run and corrupt/target run use the same archived browser step and deterministic 50-token continuation. In the core SoM-to-Phantom-SoM setup, the source prompt is `som`: task instruction, SoM prompt family, flat `[SOM_MARKS]` text, and annotated screenshot. The target prompt is `phantom_som`: the same instruction, same prompt family, and same `[SOM_MARKS]` text, but no image. Source hidden states are cached by layer, injected into the final input-token position of the target on the first forward pass, and subsequent decoding proceeds normally through the model cache.
    47	
    48	Each patched continuation is scored against the unpatched source and target continuations. The main disruption statistic is the drop in `token_overlap_to_target`; Levenshtein distance to target is the paired backup. Layer-wise tests compare each grid layer to the final-layer reference using task-paired differences and Holm-Bonferroni correction across the canonical grid. Random-injection controls replace source hidden states with Gaussian tensors matched to source activation mean and standard deviation.
    49	
    50	The Stage 2 P-SoM<->SoM dashboard now contains ten completed cells, including reddit F/G and the reddit random control:
    51	
    52	| Cell | Site | Direction | Mid-layer target-overlap drop | Holm status |
    53	|---|---|---|---:|---|
    54	| A | cls | SoM->P-SoM forward | -0.32 at L17 | significant |
    55	| B | cls | P-SoM->SoM reverse | -0.16 at L17 | significant |
    56	| C | cls | reverse-tier forward | -0.02 at L17 | null |
    57	| D | cls | strong-tier reverse | -0.18 at L17 | significant |
    58	| E | cls | random injection | -0.03 uniform | negative control |
    59	| F | reddit | SoM->P-SoM forward | -0.21 at L17 | significant |
    60	| G | reddit | P-SoM->SoM reverse | -0.18 at L17 | significant |
    61	| Cr/Dr | reddit | 2x2 controls | -0.15 to -0.18 | significant |
    62	| Er | reddit | random injection | approximately 0 uniform | negative control |
    63	
    64	Stage 3 extends this from P-SoM to the three no-image arms, testing whether the image-feature axis is shared across DOM, P-text, and P-prompt targets. The table below reports per-task-paired Δoverlap-to-target from the patching_continuation_results.json under each cell directory, with the layer at which the disruption peaks.
    65	
    66	| Site | SoM->DOM | SoM->P-text | SoM->P-prompt | best-L Δ range |
    67	|---|---:|---:|---:|---:|
    68	| cls | -0.309 at L17, -0.352 at L18 (best) | -0.255 at L17, -0.270 at L12 (best) | -0.223 at L17, -0.273 at L13 (best) | [-0.273, -0.352] |
    69	| reddit | -0.335 at L11, -0.255 at L17, -0.338 at L14 (best) | -0.244 at L11, -0.236 at L17, -0.330 at L15 (best) | -0.233 at L11, -0.191 at L17, -0.322 at L14 (best) | [-0.322, -0.338] |
    70	
    71	All six Stage 3 cells are now closed. Two observations carry the cross-site claim. First, every cell's best layer falls inside the L12-L18 mid-layer window, and every cell's best Δoverlap-to-target is between -0.27 and -0.35. The mid-layer fusion locus is therefore not a single layer index but a tight 7-layer window that transfers across cls and reddit. Second, the interpretation is additive rather than arm-specific: a SoM source state displaces DOM, P-text, and P-prompt targets toward the source with similar magnitude, implying a shared image-feature substrate across all three no-image arms. The negative controls, Cell E at -0.03 and Cell Er near zero, rule out a generic nonzero-injection explanation.
    72	
    73	## 5.5 Image-Axis Peak-Layer Dichotomy and H1 Format Variation
    74	
    75	The cleanest single-pair signature is the image-axis peak-layer dichotomy. Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap. If the no-image side is AXTree text, the image-axis cosine gap peaks at L04 in all four pairs: DOM<->Vision, DOM<->SoM, P-prompt<->Vision, and P-prompt<->SoM. If the no-image side is `[SOM_MARKS]` or another flat marks text, the peak shifts to L17-L36 in all four pairs: P-text<->Vision, P-text<->SoM, P-SoM<->Vision, and P-SoM<->SoM.
    76	
    77	The refined H1 is a pretraining co-occurrence shortcut: when the input contains a flat element-region list, the model activates a visual-grounding pathway even if the image is absent. Prompt-format sensitivity makes this plausible at the input level \citep{sclar2024promptformat}; Method 4.2 shows it as a layer-resolved internal signature.
    78	
    79	The format-variation grid contains ten modes: six marks-like variants, two controls, and DOM/SoM baselines. In the classifieds strong-tier baseline, all six marks-like variants peak at L36, the hash-ID control also peaks at L36, the plain-sentence control peaks at L17, and the DOM baseline preserves the L04 peak. Because L36 is the boundary layer, this is best read as a strong late/monotonic signature rather than as a precise late-layer mechanism.
    80	
    81	The classifieds reverse-tier run reproduces the strong-tier shape. The six marks-like variants and hash-ID control again peak at L36, the plain-sentence control moves to L22, and DOM remains at L04. This defends H1 against the selection-bias concern that strong-tier curation alone created the pattern.
    82	
    83	The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. The plan flags the two L04 marks-like reddit variants as small-n caveats rather than a reversal. Cross-site, the safe claim is directional: flat list formats tend to delay image-axis separation into mid/late layers, while AXTree hierarchy uniquely preserves the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.
    84	
    85	## 5.6 Convergent Four-Vertical-Defense Evidence Stack
    86	
    87	The first defense is per-task fragility. On 45 classifieds task-step pairs, only 11% satisfy the strict per-task dichotomy, even though aggregate marks-like peaks are later than AXTree peaks. This prevents over-claiming: H1 is a population-level mechanism signature with task variability, not a deterministic rule for every trajectory.
    88	
    89	The second defense is selection-bias robustness. The classifieds reverse-tier run replicates the strong-tier H1 pattern, including L36 marks-like peaks and L04 DOM baseline. The shortcut signature is therefore not an artifact of selecting tasks where SoM beats DOM.
    90	
    91	The third defense is cross-site H1. Reddit does not reproduce the exact boundary-layer shape, but it reproduces the direction of the indexed-list shortcut with a cleaner L17 mid-layer peak for four of six marks-like formats. The site changes the curve shape, not the basic interpretation.
    92	
    93	The fourth defense is cross-site Mirage geometry. Reddit Method 4.2 reproduces the central relation: P-SoM is close to DOM/P-text at L17 and far from SoM/Vision on the image axis, with AUROC 1.0 on the key contrasts. This supports cross-site generalization of the mechanism claim, not B0/B1 capability scaling.
    94	
    95	Two additional defenses remain deferred rather than folded into the claim: P2 cross-family Phi-3.5-Vision and P3 larger Qwen2-VL-7B. The current evidence is sufficient for the single-model, cross-site Qwen3-VL-4B mechanism section; family and capacity generalization belong in future work or Section 7.
    96	
    97	## 5.7 Layered Three-Axis Mechanism Hierarchy
    98	
    99	A naive reading of Method 4.2's L17 snapshot suggests the four phantom-boundary modes split into two text-format clusters with prompt-family making no geometric contribution. That reading is incomplete: it inspects the wrong layer. Computing full 37-layer cosine-gap profiles for axis-isolated pairs reveals a layered three-axis hierarchy in the residual stream.
   100	
   101	The pairs are constructed to isolate each axis. Axis-1 (text-format swap, prompt fixed) is measured by DOM<->P-text (both DOM prompts) and P-prompt<->P-SoM (both SoM prompts). Axis-2 (prompt-family swap, text fixed) is measured by DOM<->P-prompt (both hierarchical AXTree) and P-text<->P-SoM (both flat indexed list). Axis-3 (image-feature swap, mode otherwise fixed) is measured by the P-SoM<->SoM reference pair. All five curves are computed on `stage4_multimode_b1_cls/hidden_states.npz` (288 examples, 37 layers) and replicated cross-site on the matching reddit run.
   102	
   103	The peak-layer and magnitude table (cls site, reddit columns omitted but qualitatively identical):
   104	
   105	| Axis | Pair | L17 | L23 | L36 | Peak L | Peak gap |
   106	|---|---|---:|---:|---:|---:|---:|
   107	| Axis-3 image | P-SoM <-> SoM | 0.0412 | 0.0400 | 0.0411 | **L17** | 0.0412 |
   108	| Axis-1 text-format | DOM <-> P-text | 0.0120 | 0.0254 | 0.0201 | **L23** | 0.0254 |
   109	| Axis-1 text-format | P-prompt <-> P-SoM | 0.0113 | 0.0292 | 0.0201 | **L23** | 0.0292 |
   110	| Axis-2 prompt-family | P-text <-> P-SoM | 0.0028 | 0.0114 | 0.0089 | **L23** | 0.0114 |
   111	| Axis-2 prompt-family | DOM <-> P-prompt | 0.0013 | 0.0050 | 0.0067 | **L36** | 0.0067 |
   112	
   113	Three regularities organize the table. First, the three axes have distinct peak layers: image-axis at L17 (fast, sharp), text-format at L23 (slower late-mid build), prompt-family at L23 or L36 (same timing as text-format on the flat-text pair, boundary peak on hierarchical). Second, the three axes have distinct magnitudes: image axis approximately 0.04, text-format approximately 0.03, prompt-family approximately 0.01. Prompt-family is roughly 3 to 4 times smaller than text-format and 4 to 8 times smaller than image. Third, the magnitude rank holds cross-site: the reddit P-text<->P-SoM axis-2 peak is 0.0098 at L23 (versus cls 0.0114), the same rank-order and the same peak layer.
   114	
   115	The L17 snapshot exclusion of axis-2 is therefore a layer-selection artifact rather than a structural absence. Prompt-family is geometrically present in the residual stream; it simply emerges at L23 rather than L17, and at one-third the magnitude of text-format. The mid-layer fusion locus identified in Sections 5.2-5.5 is specifically the image-axis fusion locus (Mirage signature). The text-format and prompt-family axes share a separate late-mid build at L23 that runs in parallel.
   116	
   117	Per-task fragility check confirms the axis-2 signal is broad rather than outlier-driven. At L23 on the P-text<->P-SoM pair, the per-task cosine gap distribution is tight on both sites: classifieds median 0.0131 with IQR [0.0124, 0.0142] and reddit median 0.0120 with IQR [0.0113, 0.0127], with mean approximately equal to median in both cases (not right-skewed). 100% of 24 classifieds tasks and 100% of 24 reddit tasks exceed both the 0.005 noise threshold and the 0.010 cohort-mean magnitude. None of the 48 tasks falls outside the 0.010-0.018 band on either site. This contrasts with the per-task fragility of the binary H1 dichotomy in Section 5.5 (5 of 45 task-step pairs strict, 11%); the H1 dichotomy is fragile because it asks a per-task layer-comparison question, while the axis-2 cosine gap measures a continuous mode-pair distance that has low per-task variance even when the aggregate magnitude is small. The axis-2 claim is therefore not a 2-3 outlier mean artifact but a uniform 0.011-0.013 cosine-gap signature across every task in both sites.
   118	
   119	This layered hierarchy resolves the Phantom-SoM hero puzzle without requiring a non-mechanistic explanation. Phantom-SoM uniquely combines three contributions: residual-stream proximity to SoM on the image axis at L17 (the largest single signal, with image-feature reduction "as if image were present"), separation from P-text on the prompt-family axis at L23 (a small 0.011 signal but consistent across cls and reddit), and separation from P-prompt on the text-format axis at L23 (a medium 0.029 signal). P-text occupies only the text-format separation; P-prompt occupies only the prompt-family separation against P-SoM; DOM occupies none of the three. The drop-one hero status of Phantom-SoM in `fig_meta_forest.png` therefore corresponds to the only mode that satisfies all three axis criteria.
   120	
   121	This reframing is itself paper-grade contribution and not a downgrade of the original mechanism story. It strengthens Section 5 from "mid-layer image-feature axis explains text-format cluster" to "the residual stream carries three quantitatively distinct axes with image-axis dominant at L17 and text-format + prompt-family at L23". Section 8 inherits a sharper instruction for future single-axis steering: LA-HDMI and SAE residual-stream interventions can target the L23 prompt-family direction directly, but the signal-to-noise ratio is approximately 3-4 times worse than the image-axis intervention, so steering experiments on axis-2 require correspondingly more samples or alternative attribution methods.
   122	
   123	The three-axis hierarchy persists when we move from residual-stream geometry to output distribution. A logit lens test (Exp 3) applies Qwen3-VL-4B's final RMSNorm and lm_head to each per-layer per-mode mean hidden state, then computes KL divergence between mode pairs across all 37 layers. On classifieds the axis-2 prompt-family pair (P-text vs P-SoM) reaches peak KL 0.162 at L23, the axis-1 text-format pair (P-prompt vs P-SoM) reaches peak KL 0.695 at L23, and the reddit replication holds the same rank-order with peak KL 0.126 at L24 and 0.617 at L23 respectively. The output-level axis-1 to axis-2 magnitude ratio is therefore approximately 4.3 on classifieds and 4.9 on reddit, preserving the 3-4x residual-stream rank from Exp 1. The lm_head amplifies the residual-stream cosine signal into output divergence by roughly 14x on the axis-2 flat-text pair (cosine 0.011 to KL 0.16) and 24x on the axis-1 SoM-prompt pair (cosine 0.029 to KL 0.69), but it is axis-agnostic in the ratio it preserves.
   124	
   125	Two corollaries follow. First, the KL trajectory drops to approximately zero at L36 even though L23 KL is substantial. The mean hidden state at the final layer collapses to the shared JSON action-header tokens that every mode emits, so mode-distinct output signal is concentrated in the L23-L25 decoding window rather than at the final embedding. Second, the cosine-to-KL amplification factor is large enough that a deployment-time mode classifier built on output logprobs has strictly more signal than a classifier built on residual-stream geometry alone, even for the weak axis-2 pair. Section 6 routing (deferred to paper 2) should treat L23-L25 logit-lens features as the cheapest mode-axis discriminator with the highest already-amplified signal.
   126	
   127	## 5.8 Discussion and Limits
   128	
   129	The main limit is the Method 4.4 ceiling. The cosine-gap and patching evidence point to L11-L17 as the readable and causally active fusion region, while the best fixed mean-difference steering cell is late, L33 with $\alpha=10$, and has H-mean 0.33 because completeness and selectivity trade off. This supports a mechanism interpretation but not a strong deployment-time steering claim.
   130	
   131	The second limit is layer precision. Classifieds H1 peaks often hit L36, while reddit reveals cleaner L17 peaks. The robust claim is therefore an effect-direction claim: AXTree hierarchy preserves early image-axis separation, and flat element-list formats delay that separation into mid/late computation. We should not claim that every site or task has an identical peak layer.
   132	
   133	Literature positioning should stay modest. Section 5 applies the linear-readable, steerable, and mid/late-layer circuit framework to multimodal web-agent observation modes \citep{wu2026toolcalling,kaduri2024whatsintheimage,khorasani2026hdmi,fayyaz2026steermoe}. It should not claim novelty as the first such circuit or the first use of marked text. The contribution is controlled scientific characterization of the phantom boundary.
   134	
   135	Finally, AXTree hierarchy is the unique defeating format in the aggregate, but the reason hierarchy defeats the shortcut remains open. The plan records one attribution-pending hypothesis: hierarchy or indentation tokens may redirect cross-modal attention before the flat-list shortcut activates. That should be treated as a supplement question, not as a Section 5 finding.
   136	
   137	## NOTE FOR HUMAN
   138	
   139	Bibkeys audit (2026-05-12 21:18): all 5 core mechanism anchors verified present in `paper.bib` — `wu2026toolcalling`, `khorasani2026hdmi`, `kaduri2024whatsintheimage`, `sclar2024promptformat`, `fayyaz2026steermoe`. Plus 5 method/protocol references added: `wang2023interpretability` (IOI patching), `zhang2024patching` (patching survey, NEEDS_VERIFY exact paper), `holm1979sequentially` (multiple-comparison correction), `lipton2018troubling` (ML scholarship critique), `neurips2024checklist` (reproducibility standard). paper.bib total 67 entries / 638 lines.
   140	
   141	Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.
   142	
   143	Stage 3 numbers verified 2026-05-12 from full per-task paired-test computation on `patching_continuation_results.json` (each cell, 24 tasks × 36 layers). H-d-cls best L18 Δ=-0.352, H-d-red best L14 Δ=-0.338, H-t-cls best L12 -0.270, H-t-red best L15 -0.330, H-p-cls best L13 -0.273, H-p-red best L14 -0.322. All 6 cells' best layer lands in L12-L18 mid-layer window, Δ range [-0.27, -0.35]. The L17-only column previously cited in plan §5.2 reads -0.309/-0.255/-0.223 (cls) and -0.255/-0.236/-0.191 (reddit); plan §5.2 has been updated to record best-layer Δ instead of L17-only Δ.
   144	
   145	Pending items (post 2026-05-12 audit): (a) Method 4.4 sweep description should be "45 completed cells out of a 6x5 layer-alpha grid plus 3 placeholder cells that did not finish", not "45/48-cell sweep" (the 48-cell wording in plan §5.3 implies a 48-cell denominator that was never executed). (b) Bibkey `zhang2024patching` is marked NEEDS_VERIFY in `paper.bib` because the intended reference may be Heimersheim & Nanda 2024 [arXiv:2404.15255] rather than Zhang & Nanda 2024 [arXiv:2309.16042]; verify before submission. (c) Bibkey `fayyaz2026steermoe` is marked NEEDS_VERIFY pending deanon of the ICLR 2026 submission.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section8_limitations.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	\subsection*{8.1 Scope \& external-validity bounds}
     2	
     3	This study is deliberately narrow: three sites (classifieds, reddit, shopping), one benchmark family (VisualWebArena plus WA-mini-style task structure), and two Qwen-family model classes: Qwen3-VL-4B locally and Qwen3-Omni-235B-Thinking through a proxy API. The blast radius is external validity, not the internal comparison: our claim is that a phantom routing space exists for Qwen-family agents on VWA-style tasks, not that every VLM, browser benchmark, or production site will expose the same arm ordering \citep{zhou2024webarena,koh2024visualwebarena,deng2023mind2web,drouin2024workarena}. The mechanism evidence in Section 5 is narrower still: it is B1-only, because the open-weight 4B model exposes activations and the B0 proxy model does not. Shopping also has weaker intermediate mechanism coverage than classifieds/reddit, so site-level generalization is reported cell-by-cell rather than averaged into a universal web-agent claim. This affects Sections 1, 5, and 6 by bounding language about universality and reproducibility: B1 behavior and mechanistic patching are byte-reproducible from released artifacts, while B0 is verifiable from traces and replayable subject to API access; cross-architecture claims, including GPT-4o-family claims, remain outside the paper.
     4	
     5	\subsection*{8.2 Construct validity \& evaluator threats}
     6	
     7	VWA success labels are imperfect measurements of task completion, especially around `ua_match` GPT-judge drift, the `string_match` `fuzzy_threshold` misnomer, brittle `program_html` selectors, and `finish_wrong_state` episodes. The blast radius is measurement-side: these four evaluator-class threats can flip individual labels or inflate raw success, but they do not redefine the task universe or the mode definitions; full appendix prose is kept in Section 4.X.1--4.X.4 rather than duplicated here. We therefore report both raw and adjusted success and isolate `na_fp`, `eval_fp`, and `visual_fp` filters as sensitivity layers, including the audit-derived false-positive filters from Sections 78a and 95, following the constraint-table rule that evaluator artifacts should be disclosed rather than hidden \citep{lipton2018troubling,neurips2024checklist}. This affects Sections 3 and 4: adjusted SR is the headline metric, raw SR remains visible, and claims that survive the filter ladder are defensible as representation effects rather than judge artifacts.
     8	
     9	\subsection*{8.3 Internal-validity threats: known scaffold bugs}
    10	
    11	Several scaffold bugs were real: the `in_viewport_ratio` operator-precedence bug exposed clipped DOM text, early scroll actions suffered direction-convention confusion, and Stage 2B/2C mechanism inputs came from pre-Phase-A archived browser states. The blast radius is bounded because these failures are mode-uniform within the relevant comparisons: the viewport bug affects DOM-derived text and Phantom-SoM's `[SOM_MARKS]` source together; scroll-direction confusion is a trajectory-execution threat rather than an evaluator rule; and Stage 2 uses frozen prompt/screenshot inputs, so Phase-A dispatch bugs affect which step an agent reached, not the model's forward pass on that saved step. This affects cross-mode interpretation in Sections 3--5: we treat environment failures as measurement threats and disclose their blast radius rather than folding them into cognitive claims.
    12	
    13	\subsection*{8.4 Pre-vs-post-hoc analyses \& retracted framings}
    14	
    15	The exact L11/L17 mechanism-layer story was not preregistered, and several earlier framings were retracted. The blast radius is epistemic status: preregistered H1--H3 gate the deployment and structural claims, H4 is exploratory, and H5--H6 are post-hoc explanatory layers; Section 5.X states that L17 emerged from the Stage 2A pilot and L11 from an early single-task continuation, then converged across logit shift, forward overlap, reverse overlap, and cross-tier tests under Holm correction. The negative-results registry records 12 retracted framings, including task-0 over-interpretation, the reverse-null claim overturned by the N=15 reverse sample, and the rejected selection-bias explanation (cross-tier Welch tests non-significant, with reverse magnitude identical across tiers). It also records two framings that survived audit: the four-fold drop-in property and the sparse L11/L17 mechanism. This affects Sections 1 and 5: the paper can claim confirmed, registry-backed evidence for those two framings, but not a preregistered exact-layer prediction or a universal single-task circuit.
    16	
    17	\subsection*{8.5 Statistical \& methodological limits}
    18	
    19	The statistical design is adequate for medium effects but underpowered for small per-cell effects and exact-layer micro-effects. The blast radius is precision, not directionality: Holm-Bonferroni is applied across the six canonical tested layers (L0/5/11/17/23/29), not all 36 cached layers; this matches the disclosed post-hoc grid but should not be read as a full-layer search correction \citep{holm1979sequentially,wang2023interpretability,zhang2024patching}. Bootstrap intervals use task-paired resampling, random-effects meta-analysis is limited to cells with N>=10 to avoid unstable tau-squared estimates, and complete-case deletion handles crashes or missing artifacts without imputation. Exclusions are listwise only, at <=5% per cell under the B6 lock, so multiple imputation would add modeling assumptions without materially changing paired denominators. Power analysis shows N=15 mechanism cells are not powered for small mid-layer effects (roughly Cohen's d below 0.65 at alpha=0.05), while site-level SR cells mainly detect 4--7pp effects. This affects Sections 4--5: null or marginal cells are interpreted as low-power evidence, and pooled estimates are paired with per-cell uncertainty.
    20	
    21	\subsection*{8.6 Sparse-mechanism caveat}
    22	
    23	The activation-patching effect is sparse: at L17, four of five completed cells have median Levenshtein-distance shift equal to zero, with IQRs including zero. The blast radius is the mechanism claim: mean disruption and Holm significance are carried by task subsets, approximately the high-salience-image quarter of the strong-tier cases, while many tasks show no visible continuation change; Cell E random injection, by contrast, destroys outputs broadly. This affects Section 5 by reframing the finding as a task-conditional sparse mechanism rather than a universal mid-layer circuit. The claim remains defensible because Section 5.1 reports per-task scatter/violin views alongside mean bands and because Cell E separates content-specific sparse patching from generic injection damage.
    24	
    25	\subsection*{8.7 Compute, cost, and sustainability bounds}
    26	
    27	Compute and carbon accounting are approximate because runs span DGX Spark, UCL Condense A100, and UCL Myriad V100/A100 profiles. The blast radius is absolute energy reporting: the per-cell table reports GPU-hours, proxy-API USD cost, and kg-CO2 estimates with hardware provenance, but cross-machine power variation limits precision \citep{qiu2025modserve}. Relative comparisons inside a cell remain valid because modes share the same hardware, benchmark site, and evaluator path. This affects the cost part of the four-fold drop-in claim: token/API cost and latency comparisons are primary, while carbon totals are disclosed as bounded estimates rather than a fine-grained lifecycle assessment.
    28	
    29	<!-- Bibkey audit 2026-05-12: lipton2018troubling, neurips2024checklist, holm1979sequentially, wang2023interpretability, zhang2024patching all verified present in paper.bib. -->
    30	

exec
/bin/bash -lc 'nl -ba docs/checkpoints/paper_drafts/section4_limitations_disclosure.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Section 4 — Known Evaluator Limitations & Disclosure (Draft)
     2	
     3	**Status**: 🟡 Draft prose for paper §4 / §3 limitation table. Each subsection cites the
     4	master_bug_catalog.md entry by ID. Reviewer-defensible: bugs are CONFIRMED but blast radius
     5	bounded; mitigations or paper-§3 disclosure rather than retraction.
     6	
     7	**Source**: `docs/reference/master_bug_catalog.md` B-15 / B-20 / B-21 / B-22 / B-26 / B-28 +
     8	`docs/checkpoints/pre_run/preregistration.md` §A1/A3 design asymmetries.
     9	
    10	---
    11	
    12	## §4.X.1 ua_match GPT-judge drift (B-20)
    13	
    14	VWA's `ua_match` evaluator uses a GPT-4o-mini judge to rate the agent's terminating answer
    15	against the task's reference answer. The judge prompt template is fixed in
    16	`evaluation_harness/helper_functions.py` (`llm_fuzzy_match`) and not modified in this work.
    17	However, GPT-4o-mini is a stochastic API: the judge's output drifts across re-evaluations
    18	in 4 distinct modes (semantic equivalence vs strict literal match, spurious "partial credit",
    19	hallucinated rationale, and length-dependent confidence). Static audit of 87 N/A-task FP
    20	episodes (笔记 §95) showed the judge's binary verdict varies on ~12% of borderline cases when
    21	re-queried with temperature ≥0.
    22	
    23	**Mitigation in this work**: We pin judge `temperature=0` for all evaluations and report all
    24	ua_match-affected tasks as part of the `na_fp` exclusion class (preregistration.md §3 FP filter).
    25	Sensitivity analysis (Appendix D) shows our H1/H3 conclusions hold under three FP filter
    26	variants (raw / +na_fp / +na_fp+eval_fp), so judge drift cannot flip the paper's hero claim.
    27	
    28	**Residual concern**: If a future reviewer re-runs the evaluator with a newer GPT-4o-mini
    29	snapshot, single-task labels may flip. The aggregate per-cell SR is robust to this within
    30	±2pp by simulation. We make this explicit in our reproducibility statement (§3.X) rather
    31	than retract the SR claim.
    32	
    33	---
    34	
    35	## §4.X.2 string_match fuzzy_threshold misnomer (B-21)
    36	
    37	VWA's `string_match` evaluator exposes a `fuzzy_threshold` parameter that suggests a
    38	numerical similarity cutoff for string matching. In practice (catalog B-21 static audit),
    39	the parameter is **only honored when fuzzy_threshold=1.0** — under which the evaluator
    40	falls through to the same GPT-4o-mini fuzzy_match judge as `ua_match`. Threshold values
    41	strictly below 1.0 trigger a brittle exact-token-overlap path with no judge involvement.
    42	This is effectively binary GPT-judged matching, not a tunable similarity metric.
    43	
    44	**Mitigation**: We use `fuzzy_threshold=1.0` consistently across all conditions (verified
    45	via condition_meta.json `evaluator_config.fuzzy_threshold`), so the variability source is
    46	the same as B-20 ua_match drift and is jointly bounded by the same FP filter robustness.
    47	The mis-naming does not affect our results, but we flag it for readers attempting to
    48	interpret raw VWA evaluator parameters.
    49	
    50	---
    51	
    52	## §4.X.3 program_html selector brittleness (B-22)
    53	
    54	VWA's `program_html` evaluator scores tasks by goto'ing a target URL and querying DOM with
    55	CSS/XPath selectors authored in each task's reference config. Static audit (笔记 §107
    56	Tier 5) found 562 of 1598 (35.2%) selectors are class-only or attribute-only patterns
    57	(e.g., Magento's `.order-details-items.ordered`, classifieds' `.price` / `.desc`) that
    58	match site-skin-dependent layout. When the site's CSS skin updates between evaluator
    59	authoring time (2024) and our experimental deployment (2026), selectors can match the
    60	wrong DOM node or miss the intended element entirely.
    61	
    62	**Per-cell quantification**: We measure selector hit-rate parity in our archive — for each
    63	program_html task, we count post-action DOM nodes matching the reference selector. A pre/post
    64	ratio outside 0.95-1.05 across modes within the same task is flagged (~3% of program_html
    65	tasks); these are excluded from H1/H3 per the preregistered FP filter `eval_fp` rule.
    66	
    67	**Cannot-fix scope**: Patching all 562 brittle selectors requires authoring a parallel
    68	evaluator harness, which is out of scope for this paper. We retain VWA's evaluator unchanged
    69	(reviewer-defensible upstream parity per §3 evaluator independence) and bound the impact
    70	via the FP filter sensitivity ladder (Appendix D).
    71	
    72	---
    73	
    74	## §4.X.4 finish_wrong_state — agent error not scaffold (B-15)
    75	
    76	In Tier 2 silent-failure analysis (笔记 §107), 1552 of 4501 episodes (34.5%) had the agent
    77	emit `finish` while the page state did not match the task goal. Initial framing classified
    78	this as a scaffold bug; subsequent self-replay (笔记 §95 reform) showed it is an **agent
    79	reasoning error** — the agent decides to terminate prematurely or with partial completion,
    80	not a runner / dispatch / observation failure.
    81	
    82	**Treatment**: This is captured in our `eval_fp` filter rule (preregistration.md §3): if
    83	`agent_finished=True` but evaluator returns success and the agent has no effective action
    84	in the trajectory, we mark the episode as `eval_fp`. The agent error itself is not a paper
    85	limitation — different baselines and modes can succeed or fail at terminating decisions, and
    86	our paired-design comparison absorbs this into per-task variance.
    87	
    88	---
    89	
    90	## §4.X.5 in_viewport_ratio operator precedence (B-26)
    91	
    92	In `external/visualwebarena/browser_env/processors.py:218`, the `in_viewport_ratio`
    93	calculation `overlap_w * overlap_h / w * h` is parsed by Python as
    94	`((overlap_w * overlap_h) / w) * h` — multiplication-first then division — instead of the
    95	intended ratio `(overlap_w * overlap_h) / (w * h)`. The result is that the 0.6 viewport-overlap
    96	threshold (`current_viewport_only=True`) is effectively bypassed, allowing partially-visible
    97	elements to remain in the AXTree with their full text content even when they are visually
    98	truncated.
    99	
   100	**Implication for our claims**: This bug exists in upstream VWA and is documented in our
   101	CLAUDE.md as "DOM has structural information advantage." It systematically helps DOM mode
   102	relative to Vision/SoM modes by exposing element text that is visually clipped. We do **not**
   103	fix this bug because: (a) it's upstream code; (b) any threshold value would be debatable;
   104	(c) it does not affect our **paired** comparisons (P-SoM uses the same DOM-derived
   105	`[SOM_MARKS]` text), so our hero claims (P-SoM ≥ best of DOM/SoM/Vision) are invariant to
   106	this asymmetry. We disclose the asymmetry source for cross-mode interpretation.
   107	
   108	---
   109	
   110	## §4.X.6 scroll direction confusion (B-28)
   111	
   112	Early experiments (B0 cls/red, 笔记 §50) revealed inconsistent agent behavior for scroll
   113	direction conventions: Web CSS uses `dy>0 = scroll DOWN` (content moves up), but Win32 and
   114	macOS natural scrolling invert this convention. The 235B model occasionally chose the wrong
   115	direction sign, producing scroll-up-when-needed-down patterns counted as no-progress.
   116	
   117	**Mitigation**: §67 schema reform replaced `delta: [dx, dy]` with explicit
   118	`scroll_direction: enum("up", "down")` in the action schema (B0 only via tool-calling
   119	schema; B1 still uses delta in greedy decoding). This eliminates the symbol convention
   120	confound for B0 going forward but does not retroactively fix archived B0 data. We disclose
   121	this asymmetry in §3 evaluator-side fairness discussion.
   122	
   123	---
   124	
   125	## §4.X.7 A1/A3 baseline-design asymmetries (B-56)
   126	
   127	This work compares B0 (Qwen3-VL-235B-A22B via proxy API) against B1 (Qwen3-VL-4B-Instruct
   128	local). Two configuration asymmetries are intentional and documented:
   129	
   130	**A1 — Decoding strategy**: B0 uses `temperature=0.0` with `top_p=1.0` (B-37 fix
   131	post-§107); B1 uses `do_sample=False` (greedy top-1). Both target deterministic outputs,
   132	but B0 still inherits proxy-side stochasticity for which the API has no `seed` parameter.
   133	Cross-run trajectory variance for B0 is bounded by single-step branching at ties; aggregate
   134	SR is stable (laughs at our N=234+210+466 sampling).
   135	
   136	**A3 — Token budget**: B0 has `max_new_tokens=4096`; B1 has `max_new_tokens=384`. The
   137	asymmetry stems from B0's verbose thought + JSON output requirement; B1's parser is more
   138	robust to compact outputs. In rare cases (~0.15%), B1's compact budget causes truncated JSON →
   139	parse_failure → `wait` action. We retain this asymmetry as a B1-specific structural
   140	limitation rather than artificially inflate B1's budget; the impact is bounded and disclosed
   141	in §3 baseline configuration table.
   142	
   143	---
   144	
   145	## §4.X.8 Cross-machine numerical drift (笔记 §114 Gap 5)
   146	
   147	Our work runs across three GPU architectures: DGX Spark (NVIDIA GB10, sm_121), UCL Condense
   148	A100 (sm_80), and UCL Myriad (sm_70 V100 / sm_80 A100). Mechanistic Stage 2B/2C activation
   149	patching outputs are sensitive to floating-point matmul precision differences across CUDA
   150	generations (sm_70 vs sm_80 vs sm_121). We run `numerical_determinism_check.py` to quantify
   151	maximum absolute hidden-state drift |Δh| across machines on a fixed input.
   152	
   153	**Reproducibility statement**: Cross-machine numerical agreement on Qwen3-VL-4B between
   154	{DGX, A100, Myriad} layers L0-L35: max |Δh| < [TBD post-rerun, target <1e-2] at L11 (the
   155	mirage causal layer per §5). This bounds inter-machine reproducibility drift to a level that
   156	does not flip top-1 logit comparisons; aggregate SR claims are unaffected.
   157	
   158	---
   159	
   160	## §4.X.9 Pre-Phase-A vs post-Phase-A asymmetry (B-01 to B-37 family)
   161	
   162	The 16-cell rerun (preregistration.md §4 cell inclusion) uses post-Phase-A code only
   163	(commit ≥ `3c15cd7`, dispatch + page_changed + cycle + RNG fixes deployed). Pre-Phase-A
   164	data is retained as Appendix D robustness check (preregistration.md `Cell inclusion (Appendix D)`).
   165	For mechanistic Stage 2B/2C input artifacts, we use pre-Phase-A archived observations
   166	(`results/mechanistic/archive_subset_b1_cls/`); per 笔记 §116 user-prompt analysis, agent
   167	trajectory bugs (Phase A scaffold issues) do **not** affect the model's forward-pass
   168	input→output mapping at any frozen step. Mechanism findings (L11 causal layer, forward-vs-reverse
   169	asymmetry) are therefore unaffected by Phase A vintage; we make this independence explicit
   170	in §5.
   171	
   172	---
   173	
   174	## §4.X.10 Stage 2B input vintage independence (笔记 §116 user Q)
   175	
   176	Mechanistic Stage 2B (forward L11 mirage causal layer) and Stage 2C (reverse direction
   177	asymmetry) use frozen `observation_dom.txt` + `screenshot_annotated.png` artifacts from
   178	`B1_phantom_som_classifieds_20260428` archive (pre-Phase-A). Per 笔记 §116 user analysis:
   179	the mechanistic claim is about model forward-pass behavior given a fixed input, not about
   180	agent trajectory soundness. Phase A bugs in dispatch / cycle / RNG affect *which step* the
   181	agent reaches, not *what the model thinks* given a frozen step's observation. The L11
   182	mirage finding is therefore Phase-A-vintage-independent.
   183	
   184	For full robustness, we pre-specify a post-Phase-A spot-check (5-10 tasks from a clean
   185	post-`3c15cd7` cell) where we re-run Stage 2B and verify L11 causal layer holds. This
   186	sensitivity check is in §5 Appendix and does not gate the main mechanism claim.
   187	
   188	---
   189	
   190	## References
   191	
   192	- `docs/reference/master_bug_catalog.md` — full bug catalog (~80 entries)
   193	- `docs/checkpoints/pre_run/preregistration.md` §3-§4 — locked analysis choices including FP filter
   194	- `docs/checkpoints/pre_run/evaluator_change_protocol.md` — Protocol A Tier classification
   195	- 笔记 §95 (FP reform) / §107 (Phase A wave) / §114 (provenance) / §116 (audit) / §116.X user prompts

exec
/bin/bash -lc "nl -ba docs/checkpoints/paper_drafts/paper.bib | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	% Bibliography for Paper 1: "Phantom-SoM: Isolated Set-of-Mark Text as a Hidden Routing Arm in Web Agents".
     2	% Routing-paper references will be assembled separately.
     3	
     4	@misc{yang2023som,
     5	  title = {Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V},
     6	  author = {Yang, Jianwei and Zhang, Hao and Li, Feng and Zou, Xueyan and Li, Chunyuan and Gao, Jianfeng},
     7	  year = {2023},
     8	  eprint = {2310.11441},
     9	  archivePrefix = {arXiv},
    10	  primaryClass = {cs.CV},
    11	  doi = {10.48550/arXiv.2310.11441},
    12	  url = {https://arxiv.org/abs/2310.11441}
    13	}
    14	
    15	@inproceedings{koh2024visualwebarena,
    16	  title = {VisualWebArena: Evaluating Multimodal Agents on Realistic Visual Web Tasks},
    17	  author = {Koh, Jing Yu and Lo, Robert and Jang, Lawrence and Duvvur, Vikram and Lim, Ming Chong and Huang, Po-Yu and Neubig, Graham and Zhou, Shuyan and Salakhutdinov, Ruslan and Fried, Daniel},
    18	  booktitle = {Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics},
    19	  year = {2024},
    20	  eprint = {2401.13649},
    21	  archivePrefix = {arXiv},
    22	  primaryClass = {cs.LG},
    23	  doi = {10.48550/arXiv.2401.13649},
    24	  url = {https://arxiv.org/abs/2401.13649}
    25	}
    26	
    27	@inproceedings{zhou2024webarena,
    28	  title = {WebArena: A Realistic Web Environment for Building Autonomous Agents},
    29	  author = {Zhou, Shuyan and Xu, Frank F. and Zhu, Hao and Zhou, Xuhui and Lo, Robert and Sridhar, Abishek and Cheng, Xianyi and Ou, Tianyue and Bisk, Yonatan and Fried, Daniel and Alon, Uri and Neubig, Graham},
    30	  booktitle = {International Conference on Learning Representations},
    31	  year = {2024},
    32	  eprint = {2307.13854},
    33	  archivePrefix = {arXiv},
    34	  primaryClass = {cs.AI},
    35	  doi = {10.48550/arXiv.2307.13854},
    36	  url = {https://arxiv.org/abs/2307.13854}
    37	}
    38	
    39	@inproceedings{deng2023mind2web,
    40	  title = {Mind2Web: Towards a Generalist Agent for the Web},
    41	  author = {Deng, Xiang and Gu, Yu and Zheng, Boyuan and Chen, Shijie and Stevens, Samuel and Wang, Boshi and Sun, Huan and Su, Yu},
    42	  booktitle = {Advances in Neural Information Processing Systems, Datasets and Benchmarks Track},
    43	  year = {2023},
    44	  eprint = {2306.06070},
    45	  archivePrefix = {arXiv},
    46	  primaryClass = {cs.CL},
    47	  doi = {10.48550/arXiv.2306.06070},
    48	  url = {https://arxiv.org/abs/2306.06070}
    49	}
    50	
    51	@inproceedings{zheng2024seeact,
    52	  title = {GPT-4V(ision) is a Generalist Web Agent, if Grounded},
    53	  author = {Zheng, Boyuan and Gou, Boyu and Kil, Jihyung and Sun, Huan and Su, Yu},
    54	  booktitle = {Proceedings of the 41st International Conference on Machine Learning},
    55	  series = {Proceedings of Machine Learning Research},
    56	  volume = {235},
    57	  pages = {61349--61385},
    58	  publisher = {PMLR},
    59	  year = {2024},
    60	  eprint = {2401.01614},
    61	  archivePrefix = {arXiv},
    62	  primaryClass = {cs.IR},
    63	  doi = {10.48550/arXiv.2401.01614},
    64	  url = {https://proceedings.mlr.press/v235/zheng24e.html}
    65	}
    66	
    67	@inproceedings{yang2025magma,
    68	  title = {Magma: A Foundation Model for Multimodal AI Agents},
    69	  author = {Yang, Jianwei and Tan, Reuben and Wu, Qianhui and Zheng, Ruijie and Peng, Baolin and Liang, Yongyuan and Gu, Yu and Cai, Mu and Ye, Seonghyeon and Jang, Joel and Deng, Yuquan and Liden, Lars and Gao, Jianfeng},
    70	  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    71	  year = {2025},
    72	  eprint = {2502.13130},
    73	  archivePrefix = {arXiv},
    74	  primaryClass = {cs.CV},
    75	  doi = {10.48550/arXiv.2502.13130},
    76	  url = {https://arxiv.org/abs/2502.13130}
    77	}
    78	
    79	@inproceedings{li2025ferretui2,
    80	  title = {Ferret-UI 2: Mastering Universal User Interface Understanding Across Platforms},
    81	  author = {Li, Zhangheng and You, Keen and Zhang, Haotian and Feng, Di and Agrawal, Harsh and Li, Xiujun and Sathya Moorthy, Mohana Prasad and Nichols, Jeff and Yang, Yinfei and Gan, Zhe},
    82	  booktitle = {International Conference on Learning Representations},
    83	  year = {2025},
    84	  eprint = {2410.18967},
    85	  archivePrefix = {arXiv},
    86	  primaryClass = {cs.CV},
    87	  doi = {10.48550/arXiv.2410.18967},
    88	  url = {https://arxiv.org/abs/2410.18967}
    89	}
    90	
    91	@inproceedings{sclar2024promptformat,
    92	  title = {Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design or: How I learned to start worrying about prompt formatting},
    93	  author = {Sclar, Melanie and Choi, Yejin and Tsvetkov, Yulia and Suhr, Alane},
    94	  booktitle = {International Conference on Learning Representations},
    95	  year = {2024},
    96	  eprint = {2310.11324},
    97	  archivePrefix = {arXiv},
    98	  primaryClass = {cs.CL},
    99	  doi = {10.48550/arXiv.2310.11324},
   100	  url = {https://arxiv.org/abs/2310.11324}
   101	}
   102	
   103	@inproceedings{mishra2022reframing,
   104	  title = {Reframing Instructional Prompts to GPTk's Language},
   105	  author = {Mishra, Swaroop and Khashabi, Daniel and Baral, Chitta and Choi, Yejin and Hajishirzi, Hannaneh},
   106	  booktitle = {Findings of the Association for Computational Linguistics: ACL 2022},
   107	  pages = {589--612},
   108	  year = {2022},
   109	  doi = {10.18653/v1/2022.findings-acl.50},
   110	  url = {https://aclanthology.org/2022.findings-acl.50/}
   111	}
   112	
   113	@inproceedings{ong2025routellm,
   114	  title = {RouteLLM: Learning to Route LLMs with Preference Data},
   115	  author = {Ong, Isaac and Almahairi, Amjad and Wu, Vincent and Chiang, Wei-Lin and Wu, Tianhao and Gonzalez, Joseph E. and Kadous, M. Waleed and Stoica, Ion},
   116	  booktitle = {International Conference on Learning Representations},
   117	  year = {2025},
   118	  eprint = {2406.18665},
   119	  archivePrefix = {arXiv},
   120	  primaryClass = {cs.LG},
   121	  doi = {10.48550/arXiv.2406.18665},
   122	  url = {https://arxiv.org/abs/2406.18665}
   123	}
   124	
   125	@article{chen2023frugalgpt,
   126	  title = {FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance},
   127	  author = {Chen, Lingjiao and Zaharia, Matei and Zou, James},
   128	  journal = {arXiv preprint arXiv:2305.05176},
   129	  year = {2023},
   130	  eprint = {2305.05176},
   131	  archivePrefix = {arXiv},
   132	  primaryClass = {cs.LG},
   133	  doi = {10.48550/arXiv.2305.05176},
   134	  url = {https://arxiv.org/abs/2305.05176}
   135	}
   136	
   137	@misc{kerboua2025focusagent,
   138	  title = {FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents},
   139	  author = {Kerboua, Imene and Shayegan, Sahar Omidi and Thakkar, Megh and L{\`u}, Xing Han and Boisvert, L{\'e}o and Caccia, Massimo and Espinas, J{\'e}r{\'e}my and Aussem, Alexandre and Eglin, V{\'e}ronique and Lacoste, Alexandre},
   140	  year = {2025},
   141	  eprint = {2510.03204},
   142	  archivePrefix = {arXiv},
   143	  primaryClass = {cs.CL},
   144	  doi = {10.48550/arXiv.2510.03204},
   145	  url = {https://arxiv.org/abs/2510.03204}
   146	}
   147	
   148	@misc{li2026avenirweb,
   149	  title = {Avenir-Web: Human-Experience-Imitating Multimodal Web Agents with Mixture of Grounding Experts},
   150	  author = {Li, Aiden Yiliu and Hao, Xinyue and Liu, Shilong and Wang, Mengdi},
   151	  year = {2026},
   152	  eprint = {2602.02468},
   153	  archivePrefix = {arXiv},
   154	  primaryClass = {cs.AI},
   155	  doi = {10.48550/arXiv.2602.02468},
   156	  url = {https://arxiv.org/abs/2602.02468}
   157	}
   158	
   159	@misc{qiu2025modserve,
   160	  title = {ModServe: Modality- and Stage-Aware Resource Disaggregation for Scalable Multimodal Model Serving},
   161	  author = {Qiu, Haoran and Biswas, Anish and Zhao, Zihan and Mohan, Jayashree and Khare, Alind and Choukse, Esha and Goiri, {\'I}nigo and Zhang, Zeyu and Shen, Haiying and Bansal, Chetan and Ramjee, Ramachandran and Fonseca, Rodrigo},
   162	  year = {2025},
   163	  eprint = {2502.00937},
   164	  archivePrefix = {arXiv},
   165	  primaryClass = {cs.DC},
   166	  doi = {10.48550/arXiv.2502.00937},
   167	  url = {https://arxiv.org/abs/2502.00937}
   168	}
   169	
   170	@misc{yang2023dawnoflmms,
   171	  title = {The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)},
   172	  author = {Yang, Zhengyuan and Li, Linjie and Lin, Kevin and Wang, Jianfeng and Lin, Chung-Ching and Liu, Zicheng and Wang, Lijuan},
   173	  year = {2023},
   174	  eprint = {2309.17421},
   175	  archivePrefix = {arXiv},
   176	  primaryClass = {cs.CV},
   177	  doi = {10.48550/arXiv.2309.17421},
   178	  url = {https://arxiv.org/abs/2309.17421}
   179	}
   180	
   181	@misc{chen2026graph2eval,
   182	  title = {Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs},
   183	  author = {Chen, Yurun and Hu, Xavier and Liu, Yuhan and Wang, Ziqi and Liao, Zeyi and Chen, Lin and Wei, Feng and Qian, Yuxi and Zheng, Bo and Yin, Keting and Zhang, Shengyu},
   184	  year = {2026},
   185	  eprint = {2510.00507},
   186	  archivePrefix = {arXiv},
   187	  primaryClass = {cs.CL},
   188	  doi = {10.48550/arXiv.2510.00507},
   189	  url = {https://arxiv.org/abs/2510.00507}
   190	}
   191	
   192	% ---------------------------------------------------------
   193	% Gemini DR 2026-05-01 - Zoom 3 lit anchor expansion (5/6)
   194	% ---------------------------------------------------------
   195	
   196	% ---- Q1 Mirage / visual prompting w/o image ----
   197	@article{asadi2026mirageillusionvisualunderstanding,
   198	  title={MIRAGE: The Illusion of Visual Understanding},
   199	  author={Mohammad Asadi and Jack W. O'Sullivan and Fang Cao and Tahoura Nedaee and Kamyar Rajabalifardi and Fei-Fei Li and Ehsan Adeli and Euan Ashley},
   200	  year={2026},
   201	  journal={arXiv preprint arXiv:2603.21687},
   202	  primaryClass={cs.AI}
   203	}
   204	
   205	@inproceedings{kaduri2024whatsintheimage,
   206	  title={What's in the Image? A Deep-Dive into the Vision of Vision Language Models},
   207	  author={Omri Kaduri and Shai Bagon and Tali Dekel},
   208	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   209	  year={2025},
   210	  eprint={2411.17491},
   211	  archivePrefix={arXiv}
   212	}
   213	
   214	@article{liu2024paying,
   215	  title={Paying more attention to image: A training-free method for alleviating hallucination in lvlms},
   216	  author={Shi Liu and Kecheng Zheng and Wei Chen},
   217	  journal={arXiv preprint arXiv:2407.21771},
   218	  year={2024}
   219	}
   220	
   221	@inproceedings{wang2025xlrsbench,
   222	  title={XLRS-Bench: Could Your Multimodal LLMs Understand Extremely Large Ultra-High-Resolution Remote Sensing Imagery?},
   223	  author={Fengxiang Wang and Hongzhen Wang and Mingshuo Chen and Di Wang and Yulin Wang and Zonghao Guo and Qiang Ma and Long Lan and Wenjing Yang and Jing Zhang and Zhiyuan Liu and Maosong Sun},
   224	  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
   225	  year={2025},
   226	  eprint={2503.23771}
   227	}
   228	
   229	@article{lu2025agentrewardbench,
   230	  title={AGENTREWARDBENCH: Evaluating Reward Models for Web Agents},
   231	  author={L{\`u}, et al.},
   232	  journal={arXiv preprint arXiv:2604.04399},
   233	  year={2025}
   234	}
   235	
   236	@inproceedings{liu2025planandact,
   237	  title={Plan-and-Act: A Scalable Framework for Enhancing LLM-based Web Agents},
   238	  author={Liu, et al.},
   239	  booktitle={International Conference on Machine Learning (ICML)},
   240	  year={2025}
   241	}
   242	
   243	@inproceedings{anonymous2025vilp,
   244	  title={ViLP: A Benchmark for Evaluating Visual Language Priors in VLMs},
   245	  author={Anonymous},
   246	  booktitle={International Conference on Machine Learning (ICML)},
   247	  year={2025}
   248	}
   249	
   250	@inproceedings{zheng2024uground,
   251	  title={UGround: A Universal Visual Grounding Model for GUI Agents},
   252	  author={Boyuan Zheng and others},
   253	  booktitle={International Conference on Learning Representations (ICLR)},
   254	  year={2025}
   255	}
   256	
   257	@inproceedings{prabhu2026walt,
   258	  title={WALT: Web Agents that Learn Tools},
   259	  author={Viraj Prabhu and Yutong Dai and Matthew Fernandez and Jing Gu and Krithika Ramakrishnan and Yanqi Luo and Silvio Savarese and Caiming Xiong and Junnan Li and Zeyuan Chen and Ran Xu},
   260	  booktitle={International Conference on Learning Representations (ICLR)},

codex
The paper itself already contains some self-limiting language, but several mechanism statements still overrun the evidence. I’m now checking the plan and result files for whether the headline claims actually trace to numbers, controls, and denominators.
exec
/bin/bash -lc "nl -ba docs/checkpoints/mechanism/plan.md | sed -n '1,260p'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	---
     2	name: mechanism plan
     3	description: Full mechanism workspace — theory, lit anchor stack, methods, identification protocol, current findings, open questions, advisor sync, roadmap. Specialized companion to paper_planning §2; not a duplicate.
     4	type: workspace_plan
     5	last_substantive_update: 2026-05-12
     6	---
     7	
     8	# Mechanism Plan — paper §5
     9	
    10	## 1. Theory framework (1-screen summary, paper_planning §2 is canonical)
    11	
    12	### 1.1 Zoom 1-4 hierarchy
    13	
    14	| Zoom | Level | What our paper claims |
    15	|---|---|---|
    16	| **1** | Architectural | Phantom routing space = "skip annotated image" boundary contains 3 arms (P-text / P-prompt / P-SoM) sharing 4-fold drop-in property |
    17	| **2** | Behavioral (axis effects) | Axis 1 (text payload: AXTree vs [SOM_MARKS]) is PRIMARY; Axis 2 (prompt: SoM-prompt vs DOM-prompt) is secondary; Axis 3 (image presence: in vs out) is gating |
    18	| **3** | Named phenomena (lit-anchored) | Mirage Effect (Asadi 2026) / Scaffold Effect (Vu&Balloccu 2026) / Cross-modal flow (Kaduri) / Prompt-format sensitivity (Sclar 2024) |
    19	| **4** | Model-internal | L17 mid-layer is BOTH discrimination locus (probe AUROC 1.0) AND causally active planning site (Stage 2/3 patching + Method 4.4 v2 reliability) |
    20	
    21	### 1.2 Three-axis hierarchy quantified (Method 4.2 PCA cosine gap, Qwen3-VL-4B B1 cls)
    22	
    23	| Axis | Peak cosine gap | Peak layer | Magnitude ratio |
    24	|---|---|---|---|
    25	| Image-axis (vs SoM / Vision) | 0.06 | L4–L17 | **10×** |
    26	| Text-axis ([SOM_MARKS] vs AXTree) | 0.025 | L23 | **4×** |
    27	| Prompt-axis (SoM-prompt vs DOM-prompt alone) | 0.007 | L36 | **1×** |
    28	
    29	→ Mechanism magnitude image >> text > prompt. Validates `project_phantom_space_axes_format_not_information.md` memory: P-SoM closest mode at every layer is **P-text** (text-axis sibling, L17 cosine 0.0028 vs P-SoM↔SoM 0.0412 = 14.7× more distant).
    30	
    31	### 1.3 Image-axis peak-layer dichotomy (Mirage mechanism signature)
    32	
    33	Method 4.2 reveals image-axis cosine-gap peak shifts based on text format of the no-image side. Clean dichotomy, zero overlap across 8 image-axis pairs:
    34	
    35	| No-image side text | Peak layer | Pairs |
    36	|---|---|---|
    37	| AXTree (hierarchical) | **L04** | DOM↔Vision, DOM↔SoM, P-prompt↔Vision, P-prompt↔SoM |
    38	| [SOM_MARKS] / flat | **L17–L36** | P-text↔Vision, P-text↔SoM, P-SoM↔Vision, P-SoM↔SoM |
    39	
    40	### 1.4 H1 test confirms broader: flat-list (not just indexed) triggers shortcut (2026-05-12)
    41	
    42	Format variation extraction (Myriad job 352998, `stage4_format_variation_b1_cls`, 450 hidden states = 45 task-step × 10 modes). For each text format V, compute image-axis cosine gap V↔som per layer; peak layer reveals shortcut activation:
    43	
    44	| Format | Peak layer | Verdict |
    45	|---|---|---|
    46	| **AXTree hierarchical (DOM)** | **L04** | **SOLE format defeating shortcut** |
    47	| `"a, b, c, ..."` plain sentence | L17 | mid-level trigger |
    48	| `[N] role 'label'` (SoM standard) | L36 | strong trigger |
    49	| `@N label` (Browser Use) | L36 | strong trigger |
    50	| `id_N: label` (AppAgent) | L36 | strong trigger |
    51	| `[BN:r:l]` (Tarsier) | L36 | strong trigger |
    52	| `N. label` (numbered) | L36 | strong trigger |
    53	| `<el_N>label</el_N>` (XML) | L36 | strong trigger |
    54	| `#hash label` (control: no integer) | L36 | **still triggers!** |
    55	
    56	**Refined H1 verdict**: trigger is **flat element listing**, not "indexed list pattern". Even integer-free hash IDs and pure-sentence variants engage the shortcut. AXTree hierarchical depth is the **unique format** that defeats shortcut activation.
    57	
    58	Paper §5 implication: SoM-family web agents (Browser Use, AppAgent, Tarsier, OmniParser, etc.) **all** implicitly exploit the same flat-list-element-grounding shortcut from VLM training distribution. P79 phantom routing space makes this systematic and routes accordingly.
    59	
    60	## 2. Literature anchor stack (5 anchors, all 2026-05-08 except Sclar 2024)
    61	
    62	| Anchor | Role | bib key | What it gives our paper §5 |
    63	|---|---|---|---|
    64	| **Wu et al. 2026** (UCL lab, our advisors) | Method backbone | `wu2026toolcalling` (2605.07990) | Mean-difference activation steering at second-to-last layer, 77–100% switch on tool selection (93–100% at 4B+). Our Method 4.2/4.4 port to multimodal Qwen3-VL-4B web agent |
    65	| **Ma & Rui 2026** | Probe-vs-causal vocabulary | `maRui2026planning` (2605.07984) | "Planning-compatible representation" vs "causally active planning site". Qwen3-family pattern: probe works, causal patching weak (1% rhyme newline causal vs Gemma 67%). Our Method 4.4 v2 50% reliability is consistent with this family pattern |
    66	| **HDMI / Khorasani et al. 2026** | Alt method + evaluation metric | `khorasani2026hdmi` (2605.07631) | Probe-free gradient-based steering. Critically: **completeness × selectivity → harmonic mean reliability** — what our Method 4.4 v2 reports (not raw shift rate) |
    67	| **Lin & Liu 2026 Position paper** | Methodology protocol | `linLiu2026disclosure` (2605.08012) | 5-step identification disclosure norm: state claim / name strategy / enumerate assumptions / stress-test / separate validation. Paper §5 adopts as identification subsection structure |
    68	| **Peale et al. 2026** | §6 routing theory | `peale2026flexibleRouting` (2605.07805) | Uncertainty decomposition (reducible + irreducible) with regret bound. Paper §6 theoretical anchor; 4-fold drop-in maps onto predict/route/abstain trichotomy |
    69	
    70	## 3. Methods (Stage 4 + planned)
    71	
    72	### 3.0 Curation pipeline (Stage 0 setup, shared by Stages 1-4)
    73	
    74	`scripts/mechanistic/curate_mirage_tasks.py` builds a strong-tier task manifest from a B1 production run:
    75	- Filter to tasks where Mirage Effect is well-defined: agent succeeds in `som` mode but fails in `dom` mode (real image-feature dependence, not random failure)
    76	- Pick 24 cls + 24 reddit "strong-tier" tasks (clean Mirage); separate 15 cls reverse-tier (DOM > SoM, selection-bias defense)
    77	- Archive raw observations (AXTree, screenshots, SoM-marks) per (task, step) → `archive_subset_b1_<site>/<site>_task_<tid>/step_<step>/` for offline patching/extraction (Stages 2-4 read from this archive, no live env required)
    78	
    79	Outputs:
    80	- `results/mechanistic/curate_mirage_b1_classifieds/manifest.json` — cls strong/reverse tier task list
    81	- `results/mechanistic/curate_mirage_b1_reddit/manifest.json` — reddit strong tier
    82	- `results/mechanistic/archive_subset_b1_cls/` (17 MB, 144 files, 24 tasks × 6 steps)
    83	- `results/mechanistic/archive_subset_b1_reddit/` (35 MB, 356 files, 24 tasks × ~15 steps)
    84	
    85	### 3.1 Method 4.2 — PCA cosine gap (DONE)
    86	
    87	`scripts/analysis/stage4_pca_cosine_gap.py` + `stage4_robustness.py`. Three metrics per (mode_pair, layer):
    88	- A. Cosine gap = 1 − cos(mean_A, mean_B)
    89	- B. AUROC via (mean_A − mean_B) projection
    90	- C. Per-(mode, layer) PCA top-10 variance explained
    91	
    92	**5/5 robustness pass**:
    93	- Test A label perm: 9.8σ above noise (real 1.000 vs perm 0.629)
    94	- Test B per-task: 100% of 24 tasks positive
    95	- Test C per-step (step 2 vs step 5): invariant
    96	- Test D silhouette ≥ 0.5 at L23 (strong clustering)
    97	- Test E bootstrap 95% CI tight (4-15% of mean)
    98	
    99	### 3.2 Method 4.4 — mean-diff activation steering (v2 in flight)
   100	
   101	`scripts/mechanistic/run_stage4_method44_v2_sweep.py`. Layer × α sweep:
   102	- Layers: [11, 17, 23, 29, 33, 34] — covers mid (Stage 2 disruption locus) → late (Wu et al. second-to-last)
   103	- α: [1, 2, 5, 10, 20] — Wu et al. typical α=1, our diag found ≥5 needed for multi-step JSON
   104	- 24 cls strong-tier tasks × 2 steps × 30 cells = 1440 generations (~2h)
   105	
   106	**HDMI reliability metric**: completeness × selectivity → harmonic mean (Khorasani et al. 2026):
   107	- Completeness = % tasks where overlap_psom > overlap_dom
   108	- Selectivity = % tasks where JSON envelope preserved (starts with `{`)
   109	- Reliability = 2 · c · s / (c + s)
   110	
   111	**Current smoke (8/48 cells)**: L17 α=5 = **0.44** sweet spot (29% shift + 100% JSON valid). L33 α=10 = 0.23 (57% shift but JSON breaks).
   112	
   113	### 3.3 Method 4.5 — LA-HDMI / SAE (future work, paper §8)
   114	
   115	Two alternative paths:
   116	- **LA-HDMI**: probe-free gradient steering (Khorasani 2026 method). Per-input optimization replaces fixed mean-diff direction. May overcome Qwen3-family causal patching weakness
   117	- **SAE feature steering** (Zekun-recommended in advisor recording, paper_planning §108): train SAE on Qwen3-VL-4B residual stream (1-2 week cost, no public SAE exists), find mirage/format feature, steer directly. Differentiates from Wu et al. mean-diff path
   118	
   119	Decision pending Method 4.4 v2 full sweep + Zekun sync.
   120	
   121	## 4. Identification protocol (Lin & Liu 2026 disclosure norm)
   122	
   123	Following Lin & Liu Position paper, paper §5 must explicitly state:
   124	
   125	### 4.1 Causal claim
   126	
   127	> Mid-layer L17 hidden state at last-token position is the causally active planning site for phantom routing space mode selection in Qwen3-VL-4B web agents.
   128	
   129	### 4.2 Identification strategy
   130	
   131	Triangulation of 3 evidence types:
   132	1. **Probe-level** (Method 4.2 PCA cosine gap, AUROC 1.000 across 540 tests)
   133	2. **Replacement patching** (Stage 2/3 Cell A-H, L17 disruption peak, 8/8 Holm-sig)
   134	3. **Additive steering** (Method 4.4 v2, mid-layer L17 α=5 H-mean reliability 0.44)
   135	
   136	### 4.3 Identification assumptions
   137	
   138	| # | Assumption | Stress-test |
   139	|---|---|---|
   140	| A1 | L17 last-token hidden state mediates action selection (not earlier obs token positions) | Stage 2/3 swept all layers, L17 is peak |
   141	| A2 | Mean-difference direction approximates causal axis (Wu et al. hypothesis) | Method 4.4 v2 H-mean 0.44 partial — assumption holds weakly; LA-HDMI would test |
   142	| A3 | 24 strong-tier tasks generalize to broader VWA distribution | Stage 4 robustness Test B: 100% per-task positive, but tier-selection bias possible. Reverse-tier 15 tasks pending |
   143	| A4 | Qwen3-VL-4B mechanism transfers to other VLM sizes/architectures | Not tested. Wu et al. shows family generality on tool-only; multimodal+multi-step unknown |
   144	| A5 | Replacement patching faithfully simulates "natural" model read of the representation | Cell E random-injection control rules out non-specific disruption — content-specific causation confirmed |
   145	
   146	### 4.4 Stress-test result
   147	
   148	Cell E random-injection control: replacing source hidden with Gaussian noise (same μ, σ) yields **null L17 disruption effect**. Confirms our patching effect is source-content-specific, not noise-driven. Most directly stresses A5.
   149	
   150	### 4.5 Validation ≠ identification (Lin & Liu §5)
   151	
   152	- Method 4.2 AUROC 1.000 = validation (decodability)
   153	- Stage 2/3 + Method 4.4 v2 = identification attempts (causal use)
   154	- These are reported SEPARATELY in paper §5; reviewer should not conflate
   155	
   156	## 5. Current findings dashboard
   157	
   158	### 5.1 Stage 4 Method 4.2 (Qwen3-VL-4B B1 cls, 288 examples × 37 layers)
   159	
   160	| Pair @L17 | Cosine gap | 95% CI | AUROC |
   161	|---|---|---|---|
   162	| P-SoM ↔ P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
   163	| DOM ↔ P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
   164	| P-SoM ↔ SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
   165	| DOM ↔ Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |
   166	
   167	### 5.2 Stage 2/3 patching disruption (14 cells, B1 cls + reddit)
   168	
   169	**Stage 2 — P-SoM ↔ SoM patching (10 cells):**
   170	
   171	| Cell | Site | Direction | L17 Δoverlap | Holm-sig |
   172	|---|---|---|---|---|
   173	| A | cls | SoM→P-SoM forward | -0.32 | ✓ |
   174	| B | cls | P-SoM→SoM reverse | -0.16 | ✓ |
   175	| C | cls | 2x2 reverse-tier fwd | -0.02 | ✗ (null) |
   176	| D | cls | 2x2 strong-tier rev | -0.18 | ✓ |
   177	| E | cls | random injection | -0.03 (uniform) | ✓ (negative control) |
   178	| F | reddit | SoM→P-SoM forward | -0.21 | ✓ |
   179	| G | reddit | P-SoM→SoM reverse | -0.18 | ✓ |
   180	| Cr/Dr | reddit 2x2 | both directions | -0.15 to -0.18 | ✓ |
   181	| Er | reddit | random injection | ~0 (uniform) | ✓ |
   182	
   183	**Stage 3 — 2x2 mechanism additivity test (SoM → {DOM, P-text, P-prompt}, cls + reddit):**
   184	
   185	| Cell | Site | Source→Target | Best-L overlap→src | L17 Δoverlap→tgt | Path |
   186	|---|---|---|---|---|---|
   187	| H-d-cls | cls | SoM → DOM | L10 (0.192) | -0.33 | `stage3_cellhd_cls_fwd_dom_myriad/` |
   188	| H-p-cls | cls | SoM → P-prompt | L27 (0.219) | -0.22 | `stage3_cellhp_cls_fwd_prompt_myriad/` |
   189	| H-t-cls | cls | SoM → P-text | L28 (0.164) | -0.25 | `stage3_cellht_cls_fwd_text_myriad/` |
   190	| H-p-red | reddit | SoM → P-prompt | L20 (0.209) | -0.19 | `stage3_cellhp_red_fwd_prompt_myriad/` |
   191	| H-t-red | reddit | SoM → P-text | L01 (0.194) | -0.24 | `stage3_cellht_red_fwd_text_myriad/` |
   192	| **H-d-red** | reddit | SoM → DOM | L28 (0.204) | **L11 -0.33 / L17 -0.26** | `stage3_cellhd_red_fwd_dom_myriad/` ✅ done 2026-05-12 19:57 |
   193	
   194	**Stage 3 interpretation (6/6 cells complete 2026-05-12)**: All forward SoM→{no-image-arm} patching cells show mid-layer L11-L17 disruption -0.19 to -0.33 Δoverlap→tgt. Magnitude > random injection control (Cell E -0.03) at all 6. **Mechanism additivity confirmed**: image-feature axis is shared substrate across DOM / P-text / P-prompt arms — single SoM→{any-no-image-arm} patching displaces target prediction toward source. Cross-site cls + reddit both replicate (paper §5 universal mid-layer fusion locus); reddit fusion locus slightly earlier (L11 vs cls L17), magnitude identical.
   195	
   196	Stage 3 cross-site DOM-axis additivity table (paired-test Δoverlap-to-target from `patching_continuation_results.json`):
   197	
   198	| Site | SoM→DOM | SoM→P-text | SoM→P-prompt | best-L Δ range |
   199	|---|---|---|---|---|
   200	| cls | H-d-cls L17 -0.309 / L18 **-0.352** best | H-t-cls L17 -0.255 / L12 **-0.270** best | H-p-cls L17 -0.223 / L13 **-0.273** best | [-0.273, -0.352] |
   201	| reddit | H-d-red L11 -0.335 / L17 -0.255 / L14 **-0.338** best | H-t-red L11 -0.244 / L17 -0.236 / L15 **-0.330** best | H-p-red L11 -0.233 / L17 -0.191 / L14 **-0.322** best | [-0.322, -0.338] |
   202	
   203	All 6 cells best layer 落在 **L12-L18 mid-layer 窗口** (tight 7-layer band), Δ range [-0.27, -0.35]. Cross-site / cross-arm 一致, mid-layer fusion locus 不是 single layer index 而是稳定窗口.
   204	
   205	### 5.3 Stage 4 Method 4.4 v2 (FULL 45/48 cells, finalized 2026-05-11 22:00)
   206	
   207	H-mean reliability (HDMI framework) per (layer, α). **L17 α=5 smoke claim REFUTED by full sweep**; actual sweet spot at L33 α=10:
   208	
   209	| Layer \ α | α=1 | α=2 | α=5 | α=10 | α=20 |
   210	|---|---|---|---|---|---|
   211	| L11 | 0.04 | 0.09 | 0.20 | 0.12 | 0.12 |
   212	| L17 | 0.00 | 0.12 | **0.16** (was 0.44 smoke) | 0.12 | 0.09 |
   213	| L23 | 0.00 | 0.09 | 0.09 | 0.16 | 0.00 |
   214	| L29 | 0.00 | 0.00 | 0.00 | 0.04 | 0.04 |
   215	| **L33** | 0.04 | 0.00 | 0.00 | **0.33** ⭐ | 0.00 |
   216	| L34 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
   217	
   218	**Layer-specialization** (probe-causal dissociation):
   219	- Mid-layer (L11-L23): **selectivity 100%** at all α (JSON envelope preserved), but completeness 0-11% (modest shift)
   220	- Late-layer (L33): completeness 38% (highest), but selectivity drops to 29% (over-steers JSON)
   221	- L33 α=10 H-mean 0.33 = max reliability cell
   222	
   223	**Smoke variance lesson** (笔记 §126 + §127): 4-cell smoke H-mean 0.44 on L17 was statistical artifact (1/4 hit = inflated rate). Full 45-cell H-mean 0.16 is true rate. Future mechanism findings require n ≥ 30 cells before "sweet spot" claims.
   224	
   225	### 5.4 Image-axis peak-layer dichotomy (Method 4.2, 8 pairs)
   226	
   227	`docs/checkpoints/mechanism/results/layer_axis_emergence.md`. AXTree-no-image side → L04 peak (4/4); [SOM_MARKS]-no-image side → L17–L36 peak (4/4). Zero overlap. Mirage Effect mechanism signature.
   228	
   229	### 5.5 H1 test: flat-list format variation (Method 4.2 extension, 2026-05-12)
   230	
   231	`docs/checkpoints/mechanism/results/format_variation_h1_test.md`. 8 industry-relevant text formats + 2 controls. AXTree hierarchical (DOM) is **unique format** preserving L04 image-axis peak; all 8 flat-list variants (SoM standard, Browser Use @, AppAgent id_, Tarsier typed, plain numbered, XML tagged, hash-ID control, plain-sentence control) shift peak to L17–L36. Trigger is flat element listing, not specific token pattern.
   232	
   233	## 6. Open questions (paper-grade gaps)
   234	
   235	| Q | Status | Next action |
   236	|---|---|---|
   237	| ✅ Method 4.4 v2 full 48-cell sweep — sweet spot stable? | **Closed 2026-05-11 22:00**: L17 α=5 smoke 0.44 → full 0.16 (smoke variance artifact). **Real sweet spot L33 α=10 H-mean 0.33** | — |
   238	| ✅ H1 test: do all flat-list formats trigger shortcut? | **Closed 2026-05-12 00:00**: YES, including hash-ID + plain-sentence controls. AXTree-DOM is sole defeating format | — |
   239	| Reverse-tier 15 tasks vs strong-tier 24 — does L33 + H1 finding generalize beyond selection bias? | Med-High | qsub Stage 4 multimode + format variation with --tier reverse |
   240	| ✅ Cross-site Method 4.2 — does cls finding replicate on reddit? | **Closed 2026-05-12 16:30**: P-SoM↔DOM L17=0.0098 + P-SoM↔SoM L17=0.0423, AUROC 1.0 → Mirage signature replicated. See §7.3.1 | — |
   241	| ✅ Stage 3 reddit 2x2 closure — H-d-red | **Closed 2026-05-12 19:57** (Myriad 358831). L11 Δ=-0.33 / L17 Δ=-0.26. Cross-site additivity confirmed — see §5.2 Stage 3 table | — |
   242	| LA-HDMI vs mean-diff — does gradient steering beat 0.33 ceiling? | Med | Pending Zekun reply + attribution decision |
   243	| SAE feature steering feasibility — is 1-2 week self-training Qwen3-VL-4B SAE worth it? | Low-Med | Depends on Zekun reply + paper §8 prose direction |
   244	| B0 (proxy API) — paper §5 Qwen-specific or generalizable? | Low | Cannot test on B0; cite Wu et al. cross-family generality as proxy |
   245	| AXTree-defeats-shortcut mechanism — *why* hierarchy beats flat? Cross-modal attention specific to indentation tokens? | High (paper §5 supplement) | Activation patching at L4 with hierarchical-text vs flat-text → see which attention heads pre-disrupt image embedding |
   246	
   247	## 7. Advisor sync state — Zekun (Wu et al. 2026 first author = lab member)
   248	
   249	### 7.1 Timeline confirmed (not scoop)
   250	
   251	- 2026-04-09 笔记 §19: I first grok the paper (then "Anonymous 2026 ACL"), record cosine gap method + L23+ steering 80-93%
   252	- 2026-05-01 笔记 §108.19: upgraded to Zoom 4 anchor stack
   253	- 2026-05-02 commit `6662b91`: anchored into paper_planning §2 + paper.bib placeholder
   254	- 2026-05-09 advisor recording: Zekun explicitly recommended "SAE feature steering — 前所未有 inference time steering, 单独发 paper" — directed me to differentiating path
   255	- 2026-05-11: arxiv landed publicly; identity confirmed as lab paper
   256	
   257	**Net**: Zekun explicitly invited mechanism extension. Method 4.4 multimodal port is on his recommendation; SAE Method 4.5 is his next-step suggestion.
   258	
   259	### 7.2 Message draft (v3, paste-ready 2026-05-12)
   260	

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/exp5_axis2_causal_patching.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Exp 5 — Axis-2 Prompt-Family Causal Patching (cellhprompt cls + red)
     2	
     3	**Status**: Closed 2026-05-12 — cellhprompt_cls (359511) + cellhprompt_red (359512) landed via manual auto_pull after silent-miss bug in watcher.
     4	
     5	## Design
     6	
     7	| Variable | cellhprompt (this exp) | H-text baseline (prior) |
     8	|---|---|---|
     9	| Source | `phantom_som` (no image, flat `[SOM_MARKS]`, SoM prompt) | `som` (image, flat `[SOM_MARKS]`, SoM prompt) |
    10	| Target | `phantom_text` (no image, flat `[SOM_MARKS]`, DOM prompt) | `phantom_text` (same) |
    11	| Axes flipped src→tgt | **prompt-family only** | image axis + prompt-family |
    12	| N tasks | 24 (cls strong-tier) / 24 (red strong-tier) | matching |
    13	| Layers | 37 (L0-L36, Qwen3-VL-4B language decoder) | matching |
    14	
    15	**Test logic**: Holding both `image` and `text-format` constant (off + flat) and patching source hidden states from `phantom_som` into a `phantom_text` run isolates whether the residual-stream prompt-family signature has *causal* effect on token continuation, not just *geometric* magnitude (which Exp 1 already showed is small at 0.011 cosine gap @ L23).
    16	
    17	## Result — mid-layer (L11-L17) patching causal effect
    18	
    19	| Site | Cell (axes) | overlap→tgt L11 | overlap→tgt L17 | LD→tgt L11 | LD→tgt L17 |
    20	|---|---|---:|---:|---:|---:|
    21	| cls | H-text (image+prompt) | 0.74 | 0.75 | 9.0 | 9.2 |
    22	| cls | cellhprompt (**prompt only**) | 0.80 | 0.79 | 8.5 | 8.5 |
    23	| red | H-text (image+prompt) | 0.76 | 0.76 | 9.0 | 8.6 |
    24	| red | cellhprompt (**prompt only**) | 0.80 | 0.70 | 7.0 | 8.8 |
    25	
    26	(Baseline `overlap→tgt = 1.00` at L35 = full target preservation, no patching effect.)
    27	
    28	### Causal weight decomposition
    29	
    30	- Axis-2 (prompt) **alone** displaces target output by **0.20-0.30 overlap** units, mid-layer L11-L17 peak.
    31	- Combined image+prompt (H-text) displaces by **0.24-0.26** at same layers.
    32	- **Prompt-only captures ~77-100% of the combined effect** (cls 0.21/0.25 = 84%; red @ L17 0.30/0.24 = 125%, **prompt-only stronger on red**).
    33	- Therefore **image axis contributes a small residual** when prompt-family already differs; prompt-family is the dominant causal driver in this 2-axis subspace.
    34	
    35	### Cross-site replication
    36	Both cls + red show the same mid-layer L11-L17 peak. Reddit shows *stronger* axis-2 effect at L17 than cls (overlap→tgt 0.70 vs 0.79).
    37	
    38	## Geometric ⫨ causal disjoint (two disjoints: magnitude AND layer)
    39	
    40	### Disjoint 1 — magnitude
    41	
    42	Compared with Exp 1 cosine geometry, using best-layer values:
    43	
    44	| Axis | Cosine gap (best layer) | Patching displacement (best causal layer) |
    45	|---|---:|---:|
    46	| Image (SoM ↔ P-SoM) | 0.041 @ L17 | ~0.04-0.05 (inferred from H-text − cellhprompt diff) |
    47	| Text-format (DOM ↔ P-text) | 0.029 @ L23 | (Exp H-d-cls/red, not directly compared here) |
    48	| **Prompt-family (P-SoM ↔ P-text)** | **0.011 @ L23** | **~0.20-0.30 @ L11-L17** |
    49	
    50	**4:3:1 cosine geometry ratio does NOT translate to 4:3:1 causal patching ratio.** Prompt-family has the **smallest** geometric magnitude but the **largest** causal patching weight.
    51	
    52	### Disjoint 2 — layer
    53	
    54	Critically, the **layer at which cosine peaks ≠ the layer at which patching has maximal effect** for prompt-family:
    55	
    56	| Layer | cls overlap→tgt | red overlap→tgt | Interpretation |
    57	|---|---:|---:|---|
    58	| L0  | 0.86 | 0.92 | early, signal not yet routed |
    59	| L11 | 0.80 | 0.80 | **causal peak (cls)** — prompt-family begins routing decision |
    60	| L17 | 0.79 | **0.70** | **causal peak (red)** |
    61	| **L23** | 0.96 | 0.89 | **cosine geometry peak, but patching weak** — representation stabilized |
    62	| L29 | 0.92 | 0.95 | downstream re-encoding |
    63	| L35 | 1.00 | 1.00 | output convergence (baseline preserved) |
    64	
    65	At **L23** (the cosine peak), patching displaces target output by only **0.04-0.11 overlap units** — much smaller than the **0.20-0.30** displacement at L11-L17.
    66	
    67	### Interpretation: signature ≠ use
    68	
    69	This is the **second** geometric/causal disjoint, in addition to magnitude:
    70	
    71	- **L23 is the prompt-family "signature layer"**: representation has stabilized to its most discriminable form (highest cosine separation between P-SoM and P-text). It reflects *what prompt was given* — a state variable.
    72	- **L11-L17 is the prompt-family "decision routing layer"**: patching here changes upstream signal that downstream layers consume to drive token continuation. It reflects *how the model uses the prompt* — a causal variable.
    73	
    74	Activation patching is path-dependent: an upstream patch propagates into all downstream computations, while a downstream patch leaves upstream inputs unchanged so subsequent layers can re-encode the same signal. This is consistent with standard mechanistic-interpretability findings (cf. \citep{wang2023interpretability} IOI circuit: feature *encoded* ≠ feature *used*).
    75	
    76	### Three reads of the data
    77	
    78	1. Residual-stream cosine separation is a **necessary but not sufficient** signal of causal mechanism.
    79	2. Prompt-family information is **dispatchable** — small geometric perturbation at the decision layer produces large output displacement when patched.
    80	3. **Where a feature is most readable (L23) and where it is most consequential (L11-L17) are different layers** — paper-grade mechanism claims must report both, not collapse them.
    81	
    82	## Implications for paper §5
    83	
    84	**Strengthens 3-axis mechanism story**:
    85	- Axis-1 (text-format): Exp 1 cosine 0.029 + H-d cells causal patching (prior)
    86	- Axis-2 (prompt-family): Exp 1 cosine 0.011 + **Exp 5 cellhprompt causal patching (this)**
    87	- Axis-image: Exp 1 cosine 0.041 + indirect (H-text − cellhprompt residual ~0.04-0.05)
    88	
    89	**Defuses /stress critique** "you only have axis-1 mechanism":
    90	- Now have causal evidence for axis-2 separate from axis-1
    91	- 2-site cross-replication (cls + red), N=24 each, paired L0-L36 sweep
    92	
    93	**Reframes hero argument**: The paper §1 framing "text-format shapes exploration; prompt tunes commit" is now backed by:
    94	- Behavioral: exploration rate axis-1 dependent (Exp 1 cosine sigma + §4.5 reddit behavioral)
    95	- Causal mechanism: prompt-family mid-layer L11-L17 patching produces output displacement comparable to image-axis flip
    96	
    97	## Caveats
    98	
    99	- N=24 per cell — bootstrap CI on per-layer overlap means would tighten interpretation.
   100	- "phantom_som" archive vs "som" archive — the codebase uses same hidden-state extraction infrastructure but `--source-mode phantom_som` extracts without image. The pilot_summary template label "(with image — clean)" is a hardcoded artifact, not a runtime check. Manual verification of the archive contents (hidden state norms) would close a soundness gap.
   101	- Patching displacement is a token-level metric; doesn't directly translate to SR / drop-one oracle. Behavioral consequence (which paper §1 hero is about) operates on top of this causal signal.
   102	
   103	## Files
   104	
   105	- `pilot_summary.md`: per-site
   106	- `patching_continuation_results.json`: per-layer per-task continuation strings + metrics (~1.3 MB each)
   107	- `patching_continuation_curves.png`: visual layer profile
   108	
   109	## Provenance
   110	
   111	- Myriad jobs: 359511 (cls) + 359512 (red), submitted 2026-05-12, finished 21:42 + 21:54 UTC
   112	- Watcher missed GONE events due to silent-miss bug (PR same commit) — auto_pull dispatched manually
   113	- Bash invocation:
   114	  ```
   115	  bash scripts/maintenance/auto_pull_myriad_cell.sh 359511 cellhprm_cls stage3_cellhprompt_cls_fwd_ptext_myriad
   116	  bash scripts/maintenance/auto_pull_myriad_cell.sh 359512 cellhprm_red stage3_cellhprompt_red_fwd_ptext_myriad
   117	  ```

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/w6_h1_red_l04_attribution.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# W6 feature attribution — H1 reddit 2/6 marks-like L04 peak
     2	
     3	**Setup**: Qwen3-VL-4B tokenizer (Qwen/Qwen3-VL-4B-Instruct). Each marks-like format variant tokenized on a canonical single-element example (N=1, role=button, label=Submit). First-token character class + marker-fingerprint token count compared between L04-peak and L17-peak subgroups.
     4	
     5	## Per-variant tokenization
     6	
     7	| Variant | Peak | Example | n_tok | First token | First char class | Marker fp |
     8	|---|---|---|---:|---|---|---:|
     9	| appagent_id | L04 | `id_1: Submit` | 5 | `id` | alphanumeric | 4 (`id·_·1·:`) |
    10	| plain_numbered | L04 | `1. Submit` | 3 | `1` | alphanumeric | 2 (`1·.`) |
    11	| som_standard | L17 | `[1] button 'Submit'` | 7 | `[` | markup-sigil | 3 (`[·1·]`) |
    12	| browser_use_at | L17 | `@1 Submit` | 3 | `@` | markup-sigil | 2 (`@·1`) |
    13	| tarsier_typed | L17 | `[B1:button:Submit]` | 7 | `[B` | markup-sigil | 7 (`[B·1·:·button·:·Submit·]`) |
    14	| xml_tagged | L17 | `<el_1 role='button'>Submit</el_1>` | 14 | `<` | markup-sigil | 4 (`<·el·_·1`) |
    15	| hash_id_control | L04 | `#a3f7 Submit` | 5 | `#a` | markup-sigil | 4 (`#a·3·f·7`) |
    16	| plain_sentence | L17 | `Submit` | 1 | `Submit` | alphanumeric | 1 (`Submit`) |
    17	| dom | L04 | `button: Submit (AXTree)` | 7 | `button` | alphanumeric | 2 (`button·:`) |
    18	| som | L17 | `[1] button 'Submit' (+ image marks)` | 11 | `[` | markup-sigil | 3 (`[·1·]`) |
    19	
    20	## Subgroup first-char-class distribution (6 marks-like only)
    21	
    22	| Subgroup | alphanumeric | markup-sigil | punctuation | quote | other |
    23	|---|---:|---:|---:|---:|---:|
    24	| L04-peak (2) | 2 | 0 | 0 | 0 | 0 |
    25	| L17-peak (4) | 0 | 4 | 0 | 0 | 0 |
    26	
    27	## Hypothesis verdict
    28	
    29	✅ **Hypothesis supported (clean split)**: L04-peak variants both start with alphanumeric tokens (2/2); L17-peak variants start with markup-sigil tokens (4/4).
    30	
    31	## Secondary features
    32	
    33	- L04-peak mean marker-fp tokens: 3.00
    34	- L17-peak mean marker-fp tokens: 4.00
    35	- Δ (L17 − L04): +1.00
    36	
    37	## Full token sequence per variant (marks-like 6)
    38	
    39	- **appagent_id** (L04, `id_1: Submit`): 5 tokens: `id` · `_` · `1` · `:` · `ĠSubmit`
    40	- **plain_numbered** (L04, `1. Submit`): 3 tokens: `1` · `.` · `ĠSubmit`
    41	- **som_standard** (L17, `[1] button 'Submit'`): 7 tokens: `[` · `1` · `]` · `Ġbutton` · `Ġ'` · `Submit` · `'`
    42	- **browser_use_at** (L17, `@1 Submit`): 3 tokens: `@` · `1` · `ĠSubmit`
    43	- **tarsier_typed** (L17, `[B1:button:Submit]`): 7 tokens: `[B` · `1` · `:` · `button` · `:` · `Submit` · `]`
    44	- **xml_tagged** (L17, `<el_1 role='button'>Submit</el_1>`): 14 tokens: `<` · `el` · `_` · `1` · `Ġrole` · `='` · `button` · `'>` · `Submit` · `</` · `el` · `_` · `1` · `>`
    45	
    46	## Interpretation
    47	
    48	Within the 6 marks-like variants, the L17 vs L04 split corresponds to whether the variant's first tokens are **markup-sigil tokens** (`[`, `<`, `@`) — which co-occur with HTML / web-agent traces in pretraining and trigger the visual-grounding shortcut at mid layers — versus **plain alphanumeric tokens** (`id`, `1`) — which are common in prose / dictionary listings and behave like AXTree-DOM, peaking early at L04 where the image-axis divergence is freshly observable but not yet routed through the shortcut path.
    49	
    50	**Control variants (counterexamples that refine the rule)**:
    51	- `hash_id_control` (`#a3f7 Submit`): markup-sigil first token but L04 peak. The `#` sigil alone is not sufficient — the marker must contain an **integer index** (which `#a3f7` does not). This is consistent with prior H2 "integer is the trigger token" framing.
    52	- `plain_sentence` (`Submit`): alphanumeric first token but L17 peak. With no list/marker structure at all, the divergence path differs — possibly because the text observation drops to bare labels with no positional anchors, which the model handles via a different late-layer routing (likely commitment without grounding).
    53	
    54	Together these say: the L17 mid-layer shortcut requires **(a) integer-indexed marker + (b) markup-sigil-leading delimiter**. Either alone fails to trigger it.
    55	
    56	**Paper §5 implication**: H1's binary 'marks-like vs not' prediction is too coarse. The mechanism trigger is the **conjunction** of integer marker + markup-sigil first token, not the abstract concept of 'indexed list'. Variants like `id_N:` and `N.` are nominally indexed but lack the sigil; `hash_id_control` has the sigil but lacks an integer. Both fail to peak at L17. This refines H1 to **'integer marker + markup-sigil delimiter → triggers shortcut at L17'**, which is testable on additional variants and on a `bare_N` falsifier (drop the bracket from `[N]` and re-extract).
    57	
    58	**Falsifier (concrete next experiment)**: variant `bare_N` = `N button 'Submit'` (no brackets), which has integer + no sigil. Hypothesis predicts L04 peak. If it peaks L17, hypothesis fails.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/axis2_layer_profile.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Exp 1 — Axis-2 (prompt-family) layer profile
     2	
     3	**Question**: Method 4.2 at L17 shows prompt-family makes ~0 geometric contribution to residual stream
     4	(P-SoM↔P-text 0.0028, DOM↔P-prompt 0.0013). But forest plot drop-one places P-SoM as unique hero,
     5	implying axis-2 (prompt) contributes behaviorally. **Where in the model does axis-2 act?**
     6	
     7	**Method**: For each prompt-only pair (text format fixed, prompt swap), compute full 37-layer cosine gap.
     8	Overlay axis-1-only (text swap, prompt fixed) + image-axis P-SoM↔SoM reference curves to calibrate scale.
     9	
    10	## Results — classifieds site (stage4_multimode_b1_cls, 288 ex)
    11	
    12	| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
    13	|---|---|---:|---:|---:|---:|---:|---:|
    14	| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0013 | 0.0067 | **L36** | 0.0067 |
    15	| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0028 | 0.0089 | **L23** | 0.0114 |
    16	| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0134 | 0.0120 | 0.0201 | **L23** | 0.0254 |
    17	| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0127 | 0.0113 | 0.0201 | **L23** | 0.0292 |
    18	| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0394 | 0.0412 | 0.0411 | **L17** | 0.0412 |
    19	
    20	## Results — reddit site (stage4_multimode_b1_reddit, 288 ex)
    21	
    22	| Pair | Group | L0 | L4 | L17 | L36 | Peak L | Peak gap |
    23	|---|---|---:|---:|---:|---:|---:|---:|
    24	| DOM ↔ P-prompt  (axis-2 only, hierarchical) | axis-2 | 0.0000 | 0.0002 | 0.0012 | 0.0059 | **L36** | 0.0059 |
    25	| P-text ↔ P-SoM  (axis-2 only, flat) | axis-2 | 0.0000 | 0.0006 | 0.0027 | 0.0080 | **L23** | 0.0098 |
    26	| DOM ↔ P-text    (axis-1 only, DOM-prompt) | axis-1 | 0.0000 | 0.0125 | 0.0092 | 0.0183 | **L23** | 0.0217 |
    27	| P-prompt ↔ P-SoM (axis-1 only, SoM-prompt) | axis-1 | 0.0000 | 0.0115 | 0.0086 | 0.0176 | **L23** | 0.0240 |
    28	| P-SoM ↔ SoM     (image-axis reference) | image | 0.0000 | 0.0434 | 0.0423 | 0.0434 | **L4** | 0.0434 |
    29	
    30	## Interpretation
    31	
    32	Three hypotheses about axis-2 mechanism layer:
    33	
    34	1. **Truly null geometry** — axis-2 pair curves flat <0.01 at all layers. Prompt-family bypasses residual stream entirely (acts at attention pattern or output head). → Next: Exp 3 logit lens or Exp 4 attention probe.
    35	2. **Late-layer spike** — axis-2 pair curves spike at L25+ but flat at mid-layer. Prompt prior re-emerges at output decoding. → Next: Exp 5 late-layer patching.
    36	3. **Early-layer spike absorbed** — axis-2 pair curves spike at L0-L5 then collapse to ~0. Prompt embedding effect absorbed by mid-layer fusion. → Next: Exp 3 logit lens to verify if it re-emerges in output distribution.
    37	
    38	Compare peak layers above against axis-1 (text-format) pairs (the established mechanism with L17 peak) and image-axis reference (~0.04 magnitude). If axis-2 pair peak < 0.01 at all layers, hypothesis 1 holds.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/axis2_logit_lens.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Exp 3 — Logit lens at late layers (axis-2 vs axis-1)
     2	
     3	Apply Qwen3-VL-4B's final_norm + lm_head to per-layer per-mode mean hidden states.
     4	For each axis-isolated pair, compute KL(mode_a || mode_b) and top-1 token disagreement
     5	across all 37 layers. This probes whether axis-2 cosine signal (L23 peak 0.011) gets
     6	amplified into output distribution divergence by late-layer decoding.
     7	
     8	## Classifieds site
     9	
    10	### Axis-2 (prompt-family) pairs:
    11	
    12	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    13	|---|---:|---:|---:|---:|---:|
    14	| P-text vs P-SoM  (axis-2 flat-text) | **L23** | 0.1621 | 0.0215 | 0.1621 | 0.0003 |
    15	| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0444 | 0.0184 | 0.0234 | 0.0000 |
    16	
    17	### Axis-1 (text-format) pairs:
    18	
    19	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    20	|---|---:|---:|---:|---:|---:|
    21	| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5508 | 0.1299 | 0.5508 | 0.0001 |
    22	| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6953 | 0.1069 | 0.6953 | 0.0003 |
    23	
    24	## Reddit site
    25	
    26	### Axis-2 (prompt-family) pairs:
    27	
    28	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    29	|---|---:|---:|---:|---:|---:|
    30	| P-text vs P-SoM  (axis-2 flat-text) | **L24** | 0.1260 | 0.0371 | 0.1230 | 0.0002 |
    31	| DOM vs P-prompt  (axis-2 hierarchical) | **L25** | 0.0508 | 0.0228 | 0.0325 | 0.0000 |
    32	
    33	### Axis-1 (text-format) pairs:
    34	
    35	| Pair | Peak L (KL) | Peak KL | KL @ L17 | KL @ L23 | KL @ last |
    36	|---|---:|---:|---:|---:|---:|
    37	| DOM vs P-text    (axis-1 DOM-prompt) | **L23** | 0.5273 | 0.0898 | 0.5273 | 0.0000 |
    38	| P-prompt vs P-SoM (axis-1 SoM-prompt) | **L23** | 0.6172 | 0.0806 | 0.6172 | 0.0002 |
    39	
    40	## Interpretation
    41	
    42	Three hypotheses tested:
    43	
    44	- **H_A (axis-2 absent from output)**: axis-2 KL flat <0.1 at all layers → prompt-family
    45	  effect bypasses logit lens, only visible via attention heads or runtime decoding.
    46	- **H_B (axis-2 amplified at output)**: axis-2 KL peak at L30+ ≫ cosine 0.011 magnitude →
    47	  late-layer decoding amplifies prompt prior into output divergence (Wu et al. tool calling
    48	  'knows but says differently' mirror).
    49	- **H_C (axis-2 tracks residual stream)**: axis-2 KL peak at L23 same as cosine peak →
    50	  prompt prior signal proportional to mid-layer geometry, no amplification.
    51	
    52	Cross-site replication should hold for any of the three. Compare axis-2 KL magnitudes to
    53	axis-1 KL magnitudes to see whether 3-4x ratio in cosine space persists at output level.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/axis2_per_task_fragility.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Axis-2 per-task fragility check
     2	
     3	Per-task cosine gap distribution at L23 (axis-2 peak per §5.7 / Exp 1).
     4	Each task averaged across its 2 steps; cosine gap computed between mode pairs.
     5	
     6	**Defuse target**: /stress W2 attack — axis-2 mean 0.0114 might be dominated by 2-3 outlier tasks.
     7	
     8	## Classifieds (24 tasks)
     9	
    10	| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
    11	|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
    12	| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0132 | 0.0131 | [0.0124, 0.0142] | 0.0107 | 0.0174 | 100% | 100% | 0% |
    13	| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0048 | 0.0047 | [0.0044, 0.0052] | 0.0039 | 0.0065 | 33% | 0% | 0% |
    14	| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0287 | 0.0280 | [0.0250, 0.0312] | 0.0186 | 0.0456 | 100% | 100% | 92% |
    15	| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0407 | 0.0415 | [0.0353, 0.0438] | 0.0308 | 0.0597 | 100% | 100% | 100% |
    16	
    17	## Reddit (24 tasks)
    18	
    19	| Pair | Axis | Mean | Median | IQR | min | max | % > 0.005 | % > 0.010 | % > 0.020 |
    20	|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
    21	| P-text ↔ P-SoM   (axis-2 flat-text) | axis-2 | 0.0121 | 0.0120 | [0.0113, 0.0127] | 0.0102 | 0.0152 | 100% | 100% | 0% |
    22	| DOM ↔ P-prompt  (axis-2 hierarchical) | axis-2 | 0.0052 | 0.0051 | [0.0047, 0.0055] | 0.0039 | 0.0067 | 50% | 0% | 0% |
    23	| DOM ↔ P-text     (axis-1 reference) | axis-1 | 0.0260 | 0.0263 | [0.0226, 0.0305] | 0.0174 | 0.0344 | 100% | 100% | 83% |
    24	| P-SoM ↔ SoM     (axis-3 image ref) | axis-3 | 0.0436 | 0.0439 | [0.0409, 0.0453] | 0.0382 | 0.0535 | 100% | 100% | 100% |
    25	
    26	## Top 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)
    27	
    28	| Task ID | Cosine gap |
    29	|---|---:|
    30	| 109 | 0.0174 |
    31	| 211 | 0.0151 |
    32	| 181 | 0.0146 |
    33	| 108 | 0.0146 |
    34	| 191 | 0.0143 |
    35	
    36	## Bottom 5 axis-2 tasks (classifieds, P-text ↔ P-SoM @ L23)
    37	
    38	| Task ID | Cosine gap |
    39	|---|---:|
    40	| 10 | 0.0121 |
    41	| 228 | 0.0118 |
    42	| 116 | 0.0117 |
    43	| 32 | 0.0108 |
    44	| 161 | 0.0107 |
    45	
    46	## Top 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)
    47	
    48	| Task ID | Cosine gap |
    49	|---|---:|
    50	| 133 | 0.0152 |
    51	| 132 | 0.0145 |
    52	| 142 | 0.0135 |
    53	| 122 | 0.0135 |
    54	| 148 | 0.0131 |
    55	
    56	## Bottom 5 axis-2 tasks (reddit, P-text ↔ P-SoM @ L23)
    57	
    58	| Task ID | Cosine gap |
    59	|---|---:|
    60	| 137 | 0.0111 |
    61	| 107 | 0.0109 |
    62	| 115 | 0.0104 |
    63	| 116 | 0.0103 |
    64	| 135 | 0.0102 |
    65	
    66	## Verdict
    67	
    68	Read the `% > 0.010` column for the axis-2 P-text↔P-SoM pair:
    69	- cls: **100%** of 24 tasks above the L23 axis-2 mean magnitude
    70	- reddit: **100%** of 24 tasks above
    71	
    72	Interpretation tree:
    73	- If both ≥ 50% → axis-2 signal **broad**, /stress W2 attack defused, §5.7 framing OK
    74	- If both 25-50% → axis-2 signal **modest but present**, §5.7 needs to add 'task-conditional sparse' qualifier
    75	- If both < 25% → axis-2 signal **aggregate artifact**, §5.7 three-axis claim must downgrade to 'axis-1 + image-axis with axis-2 weak per-task'
    76	
    77	Median values: cls=0.0131, reddit=0.0120.
    78	Compare to mean: cls=0.0132, reddit=0.0121.
    79	If median << mean, the distribution is right-skewed → outlier-driven (consistent with /stress W2 attack).

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Hero-claim bootstrap CI (W1 defuse)
     2	
     3	Per-seed bootstrap 95% percentile CI on paired adjusted-SR diffs and drop-one oracle. B=10000, seed=42. Tasks resampled with replacement at task level.
     4	
     5	**Defuse target**: /stress W1 attack — paper §1 hero claim 'P-SoM 13.81% > SoM 10.48% reddit' is statistically marginal under author's own 2σ hedge.
     6	
     7	## reddit (N=210 same-task)
     8	
     9	**Per-mode adjusted SR (%)**:
    10	
    11	- dom: 9.52%
    12	- som: 10.48%
    13	- vision: 6.67%
    14	- phantom_som: 13.81%
    15	- phantom_text: 12.38%
    16	- phantom_prompt: 9.52%
    17	
    18	**Pairwise SR difference, bootstrap 95% CI:**
    19	
    20	| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
    21	|---|---:|---:|---:|---:|---:|
    22	| P-SoM vs SoM | +3.33 | +3.33 | [-0.95, +7.62] | 0.914 | 0.828 | 
    23	|  | | | ✗ crosses 0 | | |
    24	| P-SoM vs DOM | +4.29 | +4.29 | [+0.00, +8.57] | 0.963 | 0.914 | 
    25	|  | | | ✗ crosses 0 | | |
    26	| P-text vs DOM | +2.86 | +2.86 | [-0.95, +6.67] | 0.918 | 0.810 | 
    27	|  | | | ✗ crosses 0 | | |
    28	| P-SoM vs P-text | +1.43 | +1.43 | [-1.90, +5.24] | 0.739 | 0.548 | 
    29	|  | | | ✗ crosses 0 | | |
    30	
    31	**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
    32	
    33	| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
    34	|---|---:|---:|---:|---:|---:|
    35	| dom | +1.90 | +1.90 | [+0.48, +3.81] | 0.981 | 0.767 | 
    36	|  | | | ✓ strict-pos | | |
    37	| som | +1.90 | +1.90 | [+0.48, +3.81] | 0.980 | 0.762 | 
    38	|  | | | ✓ strict-pos | | |
    39	| vision | +1.43 | +1.43 | [+0.00, +3.33] | 0.949 | 0.574 | 
    40	|  | | | ✗ crosses 0 | | |
    41	| phantom_som | +3.33 | +3.33 | [+0.95, +6.19] | 0.998 | 0.969 | 
    42	|  | | | ✓ strict-pos | | |
    43	
    44	## classifieds (N=234 same-task)
    45	
    46	**Per-mode adjusted SR (%)**:
    47	
    48	- dom: 14.10%
    49	- som: 21.37%
    50	- vision: 13.68%
    51	- phantom_som: 14.53%
    52	- phantom_text: 14.53%
    53	
    54	**Pairwise SR difference, bootstrap 95% CI:**
    55	
    56	| Comparison | Point (pp) | Median | 95% CI | P(diff > 0) | P(diff > 1pp) |
    57	|---|---:|---:|---:|---:|---:|
    58	| P-SoM vs SoM | -6.84 | -6.84 | [-12.39, -1.28] | 0.005 | 0.001 | 
    59	|  | | | ✗ strict-neg | | |
    60	| P-SoM vs DOM | +0.43 | +0.43 | [-3.42, +4.70] | 0.538 | 0.374 | 
    61	|  | | | ✗ crosses 0 | | |
    62	| P-text vs DOM | +0.43 | +0.43 | [-3.42, +4.27] | 0.546 | 0.376 | 
    63	|  | | | ✗ crosses 0 | | |
    64	| P-SoM vs P-text | +0.00 | +0.00 | [-4.27, +4.27] | 0.464 | 0.317 | 
    65	|  | | | ✗ crosses 0 | | |
    66	
    67	**Drop-one oracle on 4-mode set (dom, som, vision, phantom_som), bootstrap 95% CI:**
    68	
    69	| Drop mode | Drop-one Δ (pp) | Median | 95% CI | P(Δ > 0) | P(Δ > 1pp) |
    70	|---|---:|---:|---:|---:|---:|
    71	| dom | +2.14 | +2.14 | [+0.43, +4.27] | 0.993 | 0.877 | 
    72	|  | | | ✓ strict-pos | | |
    73	| som | +8.55 | +8.55 | [+5.13, +12.39] | 1.000 | 1.000 | 
    74	|  | | | ✓ strict-pos | | |
    75	| vision | +3.42 | +3.42 | [+1.28, +5.98] | 1.000 | 0.988 | 
    76	|  | | | ✓ strict-pos | | |
    77	| phantom_som | +2.56 | +2.56 | [+0.85, +4.70] | 0.999 | 0.943 | 
    78	|  | | | ✓ strict-pos | | |
    79	
    80	## Verdict on /stress W1
    81	
    82	Read the **reddit P-SoM vs SoM** row + **reddit drop-one P-SoM** row:
    83	
    84	- If both CIs are strict-positive (ci_lo > 0) AND P(diff > 0) > 0.95 → **W1 attack defused**,   §1 hero claim is bootstrap-supported. Remove the '2σ hedge' from line 5, lead with the magnitude.
    85	- If CIs cross zero but P(diff > 0) > 0.80 → **W1 partially defused**, the claim is directional
    86	  but not strictly statistically significant. §1 hero must downgrade to 'competitive within 2σ' as
    87	  the author already wrote, but the complementarity (Jaccard / drop-one positive on N=7 tasks) carries
    88	  the structural weight.
    89	- If P(diff > 0) < 0.80 → **W1 sustained**, §1 hero claim must rewrite to 'parity / complementarity
    90	  rather than dominance'. The single-mode comparison is unsupported.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/format_variation_h1_test.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4 H1 test: indexed-list format variation
     2	
     3	Test refined H1 hypothesis (pretraining co-occurrence shortcut):
     4	*"input contains mark-like indexed region list → activates visual-grounding pathway"*
     5	
     6	**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
     7	- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
     8	- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut
     9	
    10	## Result table (sorted by peak layer)
    11	
    12	| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
    13	|---|---|---|---|---|
    14	| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0462 |
    15	| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L17** | 0.0544 |
    16	| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0434 |
    17	| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0502 |
    18	| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0506 |
    19	| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0468 |
    20	| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0502 |
    21	| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0426 |
    22	| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0514 |
    23	
    24	## Grouped by H1 prediction
    25	
    26	### marks-like  (mean peak L36)
    27	
    28	- `[N] role 'label' (SoM)`: peak **L36** = 0.0434
    29	- `@N label (Browser Use)`: peak **L36** = 0.0502
    30	- `id_N: label (AppAgent)`: peak **L36** = 0.0506
    31	- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0468
    32	- `N. label (numbered)`: peak **L36** = 0.0502
    33	- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0426
    34	
    35	### control (no integer)  (mean peak L36)
    36	
    37	- `#hash label (no integer)`: peak **L36** = 0.0514
    38	
    39	### control (no list)  (mean peak L17)
    40	
    41	- `'a, b, c, ...' (no list)`: peak **L17** = 0.0544
    42	
    43	### AXTree-baseline  (mean peak L4)
    44	
    45	- `AXTree (baseline DOM)`: peak **L04** = 0.0462
    46	
    47	## H1 verdict
    48	
    49	- **6 marks-like variants**: mean peak layer = 36, range L36-L36
    50	- **2 control variants** (no integer / no list): mean peak layer = 26, range L17-L36
    51	- **AXTree-DOM baseline**: peak L04
    52	
    53	→ **H1 PARTIAL**: marks-like AND controls all peak late — finding is broader than 'indexed list' (any text payload triggers).

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/format_variation_h1_test_cls_reverse.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4 H1 test: indexed-list format variation
     2	
     3	Test refined H1 hypothesis (pretraining co-occurrence shortcut):
     4	*"input contains mark-like indexed region list → activates visual-grounding pathway"*
     5	
     6	**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
     7	- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
     8	- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut
     9	
    10	## Result table (sorted by peak layer)
    11	
    12	| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
    13	|---|---|---|---|---|
    14	| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0434 |
    15	| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L22** | 0.0529 |
    16	| som_standard | `[N] role 'label' (SoM)` | marks-like | **L36** | 0.0429 |
    17	| browser_use_at | `@N label (Browser Use)` | marks-like | **L36** | 0.0520 |
    18	| appagent_id | `id_N: label (AppAgent)` | marks-like | **L36** | 0.0526 |
    19	| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L36** | 0.0475 |
    20	| plain_numbered | `N. label (numbered)` | marks-like | **L36** | 0.0518 |
    21	| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L36** | 0.0439 |
    22	| hash_id_control | `#hash label (no integer)` | control (no integer) | **L36** | 0.0516 |
    23	
    24	## Grouped by H1 prediction
    25	
    26	### marks-like  (mean peak L36)
    27	
    28	- `[N] role 'label' (SoM)`: peak **L36** = 0.0429
    29	- `@N label (Browser Use)`: peak **L36** = 0.0520
    30	- `id_N: label (AppAgent)`: peak **L36** = 0.0526
    31	- `[BN:role:label] (Tarsier)`: peak **L36** = 0.0475
    32	- `N. label (numbered)`: peak **L36** = 0.0518
    33	- `<el_N role='..'>label</el_N> (XML)`: peak **L36** = 0.0439
    34	
    35	### control (no integer)  (mean peak L36)
    36	
    37	- `#hash label (no integer)`: peak **L36** = 0.0516
    38	
    39	### control (no list)  (mean peak L22)
    40	
    41	- `'a, b, c, ...' (no list)`: peak **L22** = 0.0529
    42	
    43	### AXTree-baseline  (mean peak L4)
    44	
    45	- `AXTree (baseline DOM)`: peak **L04** = 0.0434
    46	
    47	## H1 verdict
    48	
    49	- **6 marks-like variants**: mean peak layer = 36, range L36-L36
    50	- **2 control variants** (no integer / no list): mean peak layer = 29, range L22-L36
    51	- **AXTree-DOM baseline**: peak L04
    52	
    53	→ **H1 PARTIAL**: marks-like AND controls all peak late — finding is broader than 'indexed list' (any text payload triggers).

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4 H1 test: indexed-list format variation
     2	
     3	Test refined H1 hypothesis (pretraining co-occurrence shortcut):
     4	*"input contains mark-like indexed region list → activates visual-grounding pathway"*
     5	
     6	**Method**: For each variant V (= different text format applied to same observation), compute per-layer cosine gap between V hidden state mean and SoM (marks+image) baseline hidden state mean. Peak layer indicates **when image-axis divergence emerges**:
     7	- Peak L04: image-presence detected freshly early → variant does NOT trigger marks-shortcut (behaves like AXTree-DOM)
     8	- Peak L17+: image-axis divergence delayed → variant DOES trigger marks-shortcut
     9	
    10	## Result table (sorted by peak layer)
    11	
    12	| Variant | Format example | H1 class | Peak layer | Peak cosine gap |
    13	|---|---|---|---|---|
    14	| appagent_id | `id_N: label (AppAgent)` | marks-like | **L04** | 0.0488 |
    15	| plain_numbered | `N. label (numbered)` | marks-like | **L04** | 0.0505 |
    16	| hash_id_control | `#hash label (no integer)` | control (no integer) | **L04** | 0.0508 |
    17	| dom | `AXTree (baseline DOM)` | AXTree-baseline | **L04** | 0.0495 |
    18	| som_standard | `[N] role 'label' (SoM)` | marks-like | **L17** | 0.0429 |
    19	| browser_use_at | `@N label (Browser Use)` | marks-like | **L17** | 0.0515 |
    20	| tarsier_typed | `[BN:role:label] (Tarsier)` | marks-like | **L17** | 0.0457 |
    21	| xml_tagged | `<el_N role='..'>label</el_N> (XML)` | marks-like | **L17** | 0.0431 |
    22	| plain_sentence | `'a, b, c, ...' (no list)` | control (no list) | **L17** | 0.0521 |
    23	
    24	## Grouped by H1 prediction
    25	
    26	### marks-like  (mean peak L13)
    27	
    28	- `[N] role 'label' (SoM)`: peak **L17** = 0.0429
    29	- `@N label (Browser Use)`: peak **L17** = 0.0515
    30	- `id_N: label (AppAgent)`: peak **L04** = 0.0488
    31	- `[BN:role:label] (Tarsier)`: peak **L17** = 0.0457
    32	- `N. label (numbered)`: peak **L04** = 0.0505
    33	- `<el_N role='..'>label</el_N> (XML)`: peak **L17** = 0.0431
    34	
    35	### control (no integer)  (mean peak L4)
    36	
    37	- `#hash label (no integer)`: peak **L04** = 0.0508
    38	
    39	### control (no list)  (mean peak L17)
    40	
    41	- `'a, b, c, ...' (no list)`: peak **L17** = 0.0521
    42	
    43	### AXTree-baseline  (mean peak L4)
    44	
    45	- `AXTree (baseline DOM)`: peak **L04** = 0.0495
    46	
    47	## H1 verdict
    48	
    49	- **6 marks-like variants**: mean peak layer = 13, range L04-L17
    50	- **2 control variants** (no integer / no list): mean peak layer = 10, range L04-L17
    51	- **AXTree-DOM baseline**: peak L04
    52	
    53	→ **H1 MIXED**: peak distribution doesn't fit simple binary prediction. Needs deeper analysis.

exec
/bin/bash -lc 'nl -ba docs/checkpoints/mechanism/results/layer_axis_emergence.md' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
     1	# Stage 4: image-axis peak-layer split — Mirage Effect signature
     2	
     3	Eight mode pairs differ in image presence (one side has image, the other doesn't). Peak cosine-gap layer reveals **when** image-axis mechanism emerges:
     4	
     5	| no-image side | image side | no-img text | peak layer | peak cosine gap |
     6	|---|---|---|---|---|
     7	| DOM | SoM | AXTree | **L04** | 0.0604 |
     8	| DOM | Vision | AXTree | **L04** | 0.0653 |
     9	| P-prompt | SoM | AXTree | **L04** | 0.0600 |
    10	| P-prompt | Vision | AXTree | **L04** | 0.0649 |
    11	| P-SoM | SoM | [SOM_MARKS] | **L17** | 0.0412 |
    12	| P-text | SoM | [SOM_MARKS] | **L20** | 0.0494 |
    13	| P-text | Vision | [SOM_MARKS] | **L36** | 0.0614 |
    14	| P-SoM | Vision | [SOM_MARKS] | **L36** | 0.0613 |
    15	
    16	## Grouped by no-image side text format
    17	
    18	### no-image text = `AXTree` (mean peak L4)
    19	
    20	- DOM ↔ SoM: peak **L04** = 0.0604
    21	- DOM ↔ Vision: peak **L04** = 0.0653
    22	- P-prompt ↔ SoM: peak **L04** = 0.0600
    23	- P-prompt ↔ Vision: peak **L04** = 0.0649
    24	
    25	### no-image text = `[SOM_MARKS]` (mean peak L27)
    26	
    27	- P-text ↔ SoM: peak **L20** = 0.0494
    28	- P-text ↔ Vision: peak **L36** = 0.0614
    29	- P-SoM ↔ SoM: peak **L17** = 0.0412
    30	- P-SoM ↔ Vision: peak **L36** = 0.0613
    31	
    32	## Mechanism interpretation (paper §5 v3 Mirage anchor)
    33	
    34	When the no-image side carries `AXTree` text (DOM, P-prompt), the image-axis cosine gap peaks at **L04** — early-layer fresh image-presence detection (vision encoder + cross-modal fusion).
    35	
    36	When the no-image side carries `[SOM_MARKS]` text (P-text, P-SoM), the image-axis cosine gap peak shifts to **L17–L36** — image yes/no divergence is deferred to mid/output layers.
    37	
    38	**Mechanism story**: `[SOM_MARKS]` text in input primes an indexed-parsing pathway through mid-layer computation. The model processes marks structurally regardless of whether image is provided, producing image-axis divergence only at late integration stages. This is the direct Method 4.2 empirical anchor for the **Mirage Effect** (Asadi et al. 2026, VLM ~70-80% no-image accuracy) and **Cross-modal flow** (Kaduri et al., middle-layer cross-modal flows store image info in query tokens): the marks-primed mid-layer computation runs *as if image were available*, with image grounding contributed only late.
    39	
    40	**Paper §5 prose** (suggested):
    41	
    42	> *Method 4.2 reveals a peak-layer shift signature for the Mirage Effect: image-axis cosine-gap peak transitions from L04 (when no-image side carries AXTree text) to L17–L36 (when no-image side carries [SOM_MARKS] text). The peak-layer shift quantifies how text-payload format primes mid-layer computation pathways — [SOM_MARKS] format triggers indexed-parsing through mid-layers regardless of image presence, with image-axis divergence deferred to late integration. This identifies [SOM_MARKS] as the mechanism trigger for the Mirage Effect, anchoring Asadi et al. 2026's behavioral finding (~70-80% no-image VLM accuracy) and Kaduri et al.'s middle-layer cross-modal flow hypothesis with layer-resolved empirical evidence.*

exec
/bin/bash -lc 'git show --stat --oneline --no-renames 3d61dde' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
3d61dde fix(infra): myriad_watcher silent-miss bug + Exp 5 axis-2 causal evidence
 .../issues/issue_myriad_watcher_silent_miss.md     | 35 ++++++++-
 .../results/exp5_axis2_causal_patching.md          | 89 ++++++++++++++++++++++
 ...256\236\351\252\214\347\254\224\350\256\260.md" | 60 +++++++++++++++
 scripts/maintenance/glm/myriad_watcher.py          | 43 ++++++++++-
 4 files changed, 225 insertions(+), 2 deletions(-)

exec
/bin/bash -lc 'git show --stat --oneline --no-renames 35784b9' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
35784b9 analysis(stage1): hero claim bootstrap CI — /stress W1 partial defuse
 .../mechanism/results/hero_claim_bootstrap_ci.md   |  90 ++++++++
 docs/checkpoints/paper_drafts/section1_intro.md    |   2 +-
 scripts/analysis/hero_claim_bootstrap.py           | 253 +++++++++++++++++++++
 3 files changed, 344 insertions(+), 1 deletion(-)

exec
/bin/bash -lc 'git show --stat --oneline --no-renames 4cfc12f' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
4cfc12f analysis(stage4): axis-2 per-task fragility — /stress W2 defused
 docs/checkpoints/mechanism/plan.md                 |  21 ++
 .../mechanism/results/axis2_per_task_fragility.md  |  79 ++++++
 .../checkpoints/paper_drafts/section5_mechanism.md |   2 +
 .../analysis/stage4_axis2_per_task_fragility.py    | 278 +++++++++++++++++++++
 4 files changed, 380 insertions(+)

exec
/bin/bash -lc 'git show --stat --oneline --no-renames 5b6c5f0' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
5b6c5f0 exp3(stage4): logit lens — axis-2 IS in output distribution
 docs/checkpoints/mechanism/plan.md                 | 19 ++++++++
 .../mechanism/results/axis2_logit_lens.md          | 53 ++++++++++++++++++++++
 .../checkpoints/paper_drafts/section5_mechanism.md |  4 ++
 scripts/analysis/stage4_logit_lens_axis2.py        | 21 ++++-----
 4 files changed, 86 insertions(+), 11 deletions(-)

exec
/bin/bash -lc 'git show --stat --oneline --no-renames 30e7488' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
30e7488 exp1(stage4): Axis-2 prompt-family layer profile — three-axis hierarchy found
 docs/checkpoints/mechanism/plan.md                 |  25 +++
 .../mechanism/results/axis2_layer_profile.md       |  38 +++++
 .../checkpoints/paper_drafts/section5_mechanism.md |  24 ++-
 scripts/analysis/stage4_axis2_layer_profile.py     | 190 +++++++++++++++++++++
 4 files changed, 271 insertions(+), 6 deletions(-)

exec
/bin/bash -lc 'git show --stat --oneline --no-renames 55afbf3' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
55afbf3 docs(paper §5): mechanism prose v1 + axis-1/axis-2 dissociation finding
 ...ection5_mechanism_prose_round_v1_2026-05-12.log | 3155 ++++++++++++++++++++
 .../section5_mechanism_prose_round_v1.md           |  139 +
 docs/checkpoints/mechanism/plan.md                 |   10 +-
 .../checkpoints/paper_drafts/section5_mechanism.md |  127 +
 4 files changed, 3427 insertions(+), 4 deletions(-)

exec
/bin/bash -lc 'git show --stat --oneline --no-renames 9269d63' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
9269d63 analysis(stage3): H-d-red done — Stage 3 reddit 2x2 mechanism closed
 docs/checkpoints/mechanism/plan.md | 13 ++++++++++---
 1 file changed, 10 insertions(+), 3 deletions(-)

exec
/bin/bash -lc 'git show --stat --oneline --no-renames 03c4a22' in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
03c4a22 analysis(stage4): P4 cls reverse-tier H1 done — selection-bias defended
 docs/checkpoints/mechanism/plan.md                 | 13 +++++-
 .../format_variation_h1_test_cls_reverse.md        | 53 ++++++++++++++++++++++
 2 files changed, 65 insertions(+), 1 deletion(-)

exec
/bin/bash -lc "rg -n \"Outcome 0c|Macro 1|Micro 2|Efficiency 3|fig_meta_forest|P-text raises|12\\.38|12\\.38|11\\.90|12\\.38|51\\.90|35\\.71|31\\.43|0\\.463|0\\.573|target-hit|keyword\" docs/checkpoints docs -g '"'!docs/checkpoints/process/**'"' -g '"'!*.log'"'" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/实验笔记.md:208:| 90 | keyword_finish 根除 + GLM Prompt 升级 | 04-24 | keyword_finish 是 scaffold confound（22 B0 + 1 B1 episode 被误杀）；移除 keyword_finish+GLM 可观测性+prompt 升级；23 episode 清除补跑 |
docs/checkpoints/实验笔记.md:229:| 63 | `<think>` 标签导致 parse_error → keyword_scroll | 04-15 | Qwen3-235B 输出 `<think>...</think>` extended thinking；regex DOTALL 贪婪捕获 think 块内容 → JSON 解析失败 → keyword_scroll（thought 丢失）；DOM 16 tasks / Vision 33 tasks；`action_utils.py` 解析前剥离 think 标签 |
docs/checkpoints/实验笔记.md:1000:## 63. `<think>` 标签导致 parse_error → keyword_scroll（2026-04-15） [bug] #bug
docs/checkpoints/实验笔记.md:1002:**症状**：B0 某些 steps `error_category=parse_error`，`thought=''`，`tokens.output` 明显偏大（300-600 vs 正常 100-150），action 退化为 keyword_scroll（delta=[0,0.8]）。DOM 18 steps/16 tasks，Vision 38 steps/33 tasks。
docs/checkpoints/实验笔记.md:1004:**根因**：Qwen3-235B-A22B 在某些步骤输出 `<think>...</think>` extended thinking 块，`parse_action_text` 先尝试直接 `json.loads`（失败），再用 `re.search(r"\{.*\}", text, re.DOTALL)` DOTALL 贪婪匹配——从 think 块内第一个 `{` 一直到文末最后一个 `}`，捕获到非法 JSON → 解析失败 → keyword_scroll。
docs/checkpoints/实验笔记.md:1045:**背景**：B0 SoM parse_error 率 ~20.1%（B0_SOM_digest），根因是 Bedrock 不支持 `enable_thinking` 字段分离，Qwen3-235B 的推理文本和 action JSON 混在同一 output 中，`parse_action_text` 的 keyword fallback 误判（"finish"/"scroll" 出现在 thought 自然语言中即触发）。`keyword_finish` 直接杀死 episode（task 4 SoM），`keyword_scroll` 蒙对方向但 thought 丢失。
docs/checkpoints/实验笔记.md:1053:**解析链路**：tool_use → text parse (json.loads→regex) → GLM extract → keyword fallback。
docs/checkpoints/实验笔记.md:1093:**方案 B（GLM fallback）验证通过**：冒烟测试 4/4 场景正确（think_block / markdown / scroll / keyword_trap）。在线验证 2/2 parse_error 成功救回：task_13 `keyword_scroll→finish`（32.5s）、task_16 `keyword_scroll→click eid=854`（49.5s，`reward=1.0`）。
docs/checkpoints/实验笔记.md:1478:## 90. keyword_finish 根除 + GLM Prompt 升级（2026-04-24） [bug] #bug
docs/checkpoints/实验笔记.md:1482:§67 通过 GLM fallback 将 parse_error 从 ~20% 降至 2-4%，但残留了一个致命路径：`action_utils.py` 的 keyword fallback 在 JSON 解析失败后扫描原始文本，若含 "finish"/"stop" 则直接触发 `keyword_finish`，绕过 GLM 恢复机会，episode 被脚手架提前终止（`fallback_finish=True`）。
docs/checkpoints/实验笔记.md:1497:**这是 scaffold confound**：B0（235B API）产生冗长输出命中 keyword_finish 的概率远高于 B1（4B local，输出结构化 JSON），导致 B0 vs B1 对比不公平。
docs/checkpoints/实验笔记.md:1501:**Fix 1：移除 keyword_finish（`action_utils.py`）**
docs/checkpoints/实验笔记.md:1503:删除 keyword fallback 中的 finish/stop 分支。不可解析的文本含 "finish" 时现在降级为 `wait`（parse_failed），episode 继续运行而非被终止。保留 `keyword_scroll` 和 `keyword_back` 作为非致命 heuristic。
docs/checkpoints/实验笔记.md:1506:# NOTE: keyword_finish removed (§67 follow-up).  "finish"/"stop" appearing
docs/checkpoints/实验笔记.md:2872:| B0 red P-text (FRESH) | 02:12 | 13.81% / 11.90% |
docs/checkpoints/实验笔记.md:3102:Micro       per-step decision (URL / target / keyword, fig2 micro)
docs/checkpoints/实验笔记.md:3146:发现：`condition_summary_v2.json` 的 `avg_total_cost_usd` 字段对 B0/B1 都用 Qwen3-VL-235B-A22B per-token rate ($0.001/1k input + $0.005/1k output) 算，导致 **B0/B1 ratio ~1.0×**（artifact）。但 paper §3 finding 7 / Efficiency 3d 旧 claim "30× cost gap" —— 实验笔记 grep 不到原始来源，凭空数字。
docs/checkpoints/实验笔记.md:3153:- **新建** `scripts/analysis/aggregate_cost_electricity.py` (Efficiency 3a + 3d standalone aggregator) → `docs/analysis/cross_sites/cost_per_mode.{json,md}`
docs/checkpoints/实验笔记.md:3161:`paper_planning §3` Efficiency 3d + Legacy index finding 7 都 update 标 "**~100×** deployment-class gap, NOT capability ratio. §103 / paper-planning legacy '30×' claim **superseded**".
docs/checkpoints/实验笔记.md:3163:`fig3d_cost_sr_frontier.png` 重做：x-axis log-scale + 顶部 "B0 API \$ vs B1 electricity-equivalent \$ (different cost classes)" + "B0/B1 ~100× deployment-class gap" annotation。同图同时 visual evidence Efficiency 3a (P-SoM ≈ DOM cost intra-baseline) + Efficiency 3d (deployment-class gap)。
docs/checkpoints/实验笔记.md:3172:- 12 scripts/figures 头部 docstring 加 dimension+sub-code 标签（e.g. `[Outcome 0c]`, `[Macro 1c]`, `[Micro 2a-2e]`, `[Efficiency 3a]`）
docs/checkpoints/实验笔记.md:3198:├── phantom_lift.{md,csv}            Outcome 0c+0d
docs/checkpoints/实验笔记.md:3207:├── fig2_micro_divergence_heatmap.png  URL/target/keyword 4-panel
docs/checkpoints/实验笔记.md:3213:├── cost_per_mode.{json,md}          Efficiency 3a + 3d standalone (electricity-equivalent)
docs/checkpoints/实验笔记.md:3214:└── run_summary_collect.json         Efficiency 3b token data
docs/checkpoints/实验笔记.md:3259:**paper Section 5 对 cls 的写法**: "macro action-type 频率 average 掩盖了 per-task decision divergence; routing arm 价值需 outcome (Outcome 0d) + micro (Micro 2a) 一起证，不能只 cite macro"。
docs/checkpoints/实验笔记.md:3281:- `scripts/analysis/aggregate_cost_electricity.py` (Efficiency 3a/3d)
docs/checkpoints/实验笔记.md:3282:- `scripts/analysis/axis_effect_size.py` (Macro 1a/1b cascade ablation)
docs/checkpoints/实验笔记.md:3283:- `scripts/analysis/axis1_microbehavior.py` (Micro 2a-2e)
docs/checkpoints/实验笔记.md:3295:- `docs/checkpoints/paper_planning.md` §3 重组 + Efficiency 3d 100× 修正
docs/checkpoints/实验笔记.md:3341:- **`make analyze-layered` Efficiency 3d cost ratio "1×" warning**：layered_status.py 已 fallback 到 cost_per_mode.json (~100×)，警告其实是从 condition_summary 直读的 artifact 信号，不影响 paper 数字
docs/checkpoints/实验笔记.md:3650:- Micro (per-step): denominator confound (Section 5 prose 已用 Micro 2f first-divergence patch 这层)
docs/checkpoints/实验笔记.md:5307:- `p79/utils/auth_refresh.py::should_refresh()` 加 `seconds_since_refresh` keyword arg + time-based check (default 1200s, below PHP `gc_maxlifetime=1440s`)
docs/checkpoints/实验笔记.md:5588:- 4-dimension Evidence Framework (笔记 §106) — Outcome 0c/0d (phantom_lift), 0g (auroc), 1a/1b (axis_effect_size), 2a-2e (axis1_microbehavior), 3a-3c (cross_site)
docs/checkpoints/实验笔记.md:5766:- 其他 phantom routing figures (fig_meta_forest, fig0c_*, fig_phantom_structure_venn) 仍是 priority — 它们 support 1+2 contribution
docs/checkpoints/实验笔记.md:6354:- Narrative mapping 严格: 每条都写 "P79 关联" 一句话, 没有 keyword-only 撞名词
docs/checkpoints/advisor_sync_5_5_outcomes.md:247:| `fig_meta_forest.png` | `results/phantom_paper/figures/` | Hero + Ablation forest plot |
docs/reference/master_bug_catalog.md:121:- **Fix**: forward `parsed_code[-1].arguments` and `.keywords` into `execute_playwright_select_option()` — 一行代码 + regression test (upstream cleanup, 不影响 P79 实际行为). Low priority (15 ep / 4501 ≈ 0.3% of all ep).
docs/reference/master_bug_catalog.md:288:  2. **Active-finish blank/generic N/A gap** — agent finish without "N/A" keyword but answer 是空 / 通用. **NOT covered**. Spot-check found adjusted-true N/A rows with active finish.
docs/reference/master_bug_catalog.md:1150:### B-65. <think> tag → parse_error → keyword_scroll (笔记 §63, 2026-04-15)
docs/reference/master_bug_catalog.md:1155:- **Bug**: Qwen3-235B-A22B sometimes output `<think>...</think>` extended thinking blocks. `parse_action_text` `re.search(r"\{.*\}", text, re.DOTALL)` greedy-matched from think block's first `{` to text's last `}` → invalid JSON → keyword_scroll fallback. DOM 18 steps/16 tasks; Vision 38 steps/33 tasks.
docs/reference/master_bug_catalog.md:1272:### B-79. keyword_finish 根除 + GLM Prompt 升级 (笔记 §90, 2026-04-24)
docs/reference/master_bug_catalog.md:1277:- **Bug**: Keyword fallback `keyword_finish` caused episodes to terminate prematurely on natural-language "finish" mention in thought.
docs/reference/master_bug_catalog.md:1278:- **Fix**: Removed keyword_finish entirely; rely on GLM extract fallback (§67/B).
docs/reference/master_bug_catalog.md:1318:- **Symptom**: stderr `TypeError: register_pytree_node() got an unexpected keyword argument 'serialized_type_name'`
docs/checkpoints/ADVISOR_SYNC.md:104:| `fig_meta_forest.png` | Hero + Ablation forest plot, P-SoM hero 顶部黑框 |
docs/checkpoints/canvas/paper_section2_framework.canvas:263:      "text": "### Outcome × mode ⭐\n\nadj SR per arm:\n• B0 red SoM 10.48 / P-SoM 13.81\n• B0 red P-text 11.90 / P-prompt 10.48\n\n6-mode oracle vs 3-mode +7.14pp [3.81, 10.48] sig"
docs/checkpoints/paper_drafts/section5_mechanism.md:119:This layered hierarchy resolves the Phantom-SoM hero puzzle without requiring a non-mechanistic explanation. Phantom-SoM uniquely combines three contributions: residual-stream proximity to SoM on the image axis at L17 (the largest single signal, with image-feature reduction "as if image were present"), separation from P-text on the prompt-family axis at L23 (a small 0.011 signal but consistent across cls and reddit), and separation from P-prompt on the text-format axis at L23 (a medium 0.029 signal). P-text occupies only the text-format separation; P-prompt occupies only the prompt-family separation against P-SoM; DOM occupies none of the three. The drop-one hero status of Phantom-SoM in `fig_meta_forest.png` therefore corresponds to the only mode that satisfies all three axis criteria.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:1:# Phase 3 — Evidence layer coverage figures (Micro 2b/2c/2d/2e/2f + Efficiency 3c)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:13:Paper §3 4-dim Evidence framework has 16 sub-codes (4 dim × 4 cross-X). Of these, **6 sub-codes have data but no dedicated figure**. This Phase fills the visualization gap so paper §5 mechanism prose can cite figures by name (currently `section5_mechanism_reddit.md` cites "Micro 2f, median first divergent step 0" but **no figure exists** — paper-writing blocker).
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:18:| Micro 2a URL signature divergence | ✅ axis1_microbehavior.json | ✅ fig2_micro_divergence_heatmap.png |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:19:| **Micro 2b Target-page hit rate** | ✅ axis1_microbehavior.json | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:20:| **Micro 2c Search-keyword reuse** | ✅ axis1_microbehavior.json | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:21:| **Micro 2d First-action divergence** | ✅ axis1_microbehavior.json | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:22:| **Micro 2e Cross-site validity ratio** | ✅ axis1_microbehavior.json (`cross_site_validity` field) | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:23:| **Micro 2f First-divergence step** | ✅ axis1_microbehavior.json (or computed live from steps JSONL) | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:24:| Efficiency 3a Token cost | ✅ condition_summary_v2.json | ✅ fig3a_token_cost_intra_baseline.png (bundled with 3c) |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:25:| **Efficiency 3c Latency per step** | ✅ condition_summary_v2.json (`avg_total_latency_ms` / `avg_steps`) | ❌ no dedicated figure (bundled in 3a) |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:26:| Efficiency 3d Cost-SR Pareto | ✅ paper_cost_usd | ✅ fig3d_cost_sr_frontier.png |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:36:**Sub-code**: Micro 2b — Target-page hit rate per mode × site (paired comparison)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:42:**Visualization**: 4-panel bar chart (B0 cls / B0 red / B1 cls / B1 red), x-axis = mode (DOM/P-text/P-prompt/P-SoM/SoM/Vision), y-axis = target-hit-rate %. Annotate axis-1 effect (DOM → P-text delta) and axis-2 effect (P-text → P-SoM delta) on top of bars.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:48:### 2. `scripts/analysis/figures/fig2c_keyword_repeat.py`
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:50:**Sub-code**: Micro 2c — Search-keyword reuse / max-keyword-repeat per trajectory
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:52:**Data source**: `docs/analysis/cross_sites/axis1_microbehavior.json` (`max_keyword_repeat` field per mode per site, distribution stats)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:54:**Visualization**: 4-panel box plot (B0/B1 × cls/red), x-axis = mode, y-axis = max keyword repeat count per task. Show median + IQR + outliers. Optional overlay: paired delta arrows (DOM → P-text, P-text → P-SoM) for axis effect direction.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:56:**Output**: `results/phantom_paper/figures/fig2c_keyword_repeat.png`
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:62:**Sub-code**: Micro 2d — First-action divergence between mode pairs (% tasks where mode A and mode B chose different first action_type)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:75:**Sub-code**: Micro 2e — Cross-site validity ratio (does axis effect generalize cls → red?)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:79:**Visualization**: Single panel (or 2-panel B0/B1 if scaling is asymmetric). x-axis = axis (axis 1 / axis 2 / compound). y-axis = effect ratio (red effect / cls effect on Micro 2a-2d metric). Horizontal target line at 1.0 (perfect generalization). Annotate with which Micro sub-metric is being aggregated. **If ratio > 1.0, effect is reddit-amplified; if < 1.0, cls-amplified; if ≈ 1.0, generalize-symmetric.**
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:87:**Sub-code**: Micro 2f — First-divergence step distribution per mode pair × site (paper §5 prose 已 cite, missing figure is paper-writing blocker)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:93:**This is paper-cited figure** (`section5_mechanism_reddit.md` line 27/39/43/51 references "Micro 2f"). Section 5 prose currently reads "median first divergent step 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%)" — this figure must visualize that.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:101:**Sub-code**: Efficiency 3c — Per-step latency separated from cost (paper §1 hook 4-fold drop-in property (b))
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:121:	$(PYTHON) scripts/analysis/figures/fig2c_keyword_repeat.py
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:136:1. **Module docstring**: `"""[Micro 2X] <dim> dimension — <description>.\n\nOutput:\n- results/phantom_paper/figures/figXX.png\n\n<sub-code description>.\n\nSee docs/checkpoints/paper_planning.md §3 <dim> dimension framework.\n"""`
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:187:5. **fig2f matches paper §5 prose claim** (`section5_mechanism_reddit.md` cites "Micro 2f, N=15, median first divergent step 0, early divergence 100%" for B0 reddit P-text↔P-SoM pair) — figure must show this finding for that pair specifically.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:194:- `docs/checkpoints/paper_planning.md` §3 Evidence framework (Micro 2a-2f / Efficiency 3a-3d sub-code definitions)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:195:- `docs/checkpoints/paper_drafts/section5_mechanism_reddit.md` (paper-cited Micro 2f references)
docs/checkpoints/paper_drafts/section4_empirical_findings.md:37:| Reddit | P-text | 11.90 | $0.046 | 58.1s | N=210 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:98:This substrate makes reddit an informative test case for separating the three axes. Images are frequent in the task prompts and in the posts themselves, but their role is usually evidential: an image can identify which post is being discussed, or disambiguate a content clue, but it is not the site's primary navigation affordance. The browser screenshot does not create the forum hierarchy; it only renders it. Conversely, the search box is prominent in the DOM and AXTree, but intrinsic search is not the intended substrate for many tasks. Repeated search is a failure basin: the agent can keep refining keywords while never taking the forum, post, or comment link that would satisfy the evaluator. The mechanism to explain is therefore not simply "text works better than vision." It is that each representation changes which textual affordances become salient enough for the model to commit to.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:104:The evidence chain is consistent across dimensions. At the outcome level, adding P-text to the three-mode baseline contributes oracle value even without the SoM prompt or screenshot (Outcome 0c, +P-text +3.21pp single-phantom lift on the current oracle intersection). At the macro level, the whole-run strategy gradient shows the failure basin directly: reddit search-loop rate falls from DOM to Phantom-SoM and then to full SoM (Macro 1c, search-loop 51.90%->35.71%->31.43%). The axis-1-only macro effect is smaller than the compound prompt path, which is expected if flat text mainly changes which page objects are reachable rather than merely changing the action vocabulary. The micro evidence is sharper: DOM versus P-text has low path overlap for a text-only swap (Micro 2a, URL-path Jaccard 0.573), improves target-page reach (Micro 2b, target-hit +3.47pp), and reduces repeated keyword reuse (Micro 2c, max-keyword-repeat -0.633). The click-target view tells the same story: the two modes choose substantially different element sets even before images enter the system (Micro 2a-extra, click-target Jaccard 0.463).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:106:The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:108:Efficiency further constrains the explanation. Because P-text is generated from the same AXTree-derived text source and does not attach a screenshot, the reddit improvement cannot be attributed to paying the visual-token tax (Efficiency 3a, DOM $0.0516/episode versus P-text $0.0459/episode in the site dictionary). Axis 1 is therefore a representation effect: the observation text is rearranged into an indexed list, not enriched with new visual evidence.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:116:The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:118:The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:120:The prompt contrast is also visible in the mode-invariant click-target metric. With the text payload held fixed, P-text and P-SoM still have low click-target overlap (Micro 2a-extra, P-text<->P-SoM click-target Jaccard 0.484). This matters because it rules out a purely verbal explanation in which the SoM prompt only changes confidence wording at `finish`. The prompt changes which marked objects are selected during navigation.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:128:The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:130:The image-axis micro contrast confirms that the screenshot is behaviorally strong even when it is not outcome-positive. P-SoM and SoM have low URL-path agreement and frequent immediate divergence (Micro 2a, image-axis URL-path Jaccard 0.456; Micro 2f, early divergence 95.24%). Thus "weak" should be read as weak net value on this substrate, not as weak causal force. The screenshot changes decisions; on reddit, those changed decisions often point to content assets rather than evaluator-relevant post or comment routes.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:134:The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:138:The compound transition from DOM to Phantom-SoM combines the flat text payload with the marked-element prompt while still avoiding the image channel. On reddit this compound arm is best interpreted as a complementary routing arm rather than as a uniformly stronger baseline. Its adjusted SR is modestly higher than DOM on the full 210-task per-mode summaries (Outcome 0a, P-SoM 13.81% versus DOM 9.52%, +4.29pp), but the more robust mechanism evidence is the divergence in which routes it explores. DOM and P-SoM have low click-target overlap (Micro 2a-extra, compound click-target Jaccard 0.421) and low URL-path overlap (Micro 2a, compound URL-path Jaccard 0.481). The oracle result then follows naturally: P-SoM adds tasks that the original three-mode set misses (Outcome 0c, +P-SoM +2.56pp), and P-text/P-SoM are not redundant with each other (Outcome 0d, Jaccard 0.500).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:142:The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/checkpoints/paper_planning.md:495:| Image axis sub-effects | Macro 1b axis 3 small (5/5 reddit metrics show image effect d_z<0.16) — image is content not navigation, helping/harming roughly balanced |
docs/checkpoints/paper_planning.md:506:| Mechanically dominant axis | **Axis 3 (image)** — Macro 1b cls image axis dominates 5/8 metrics (h=+0.57 finish rate, d=−0.42 action repeat) |
docs/checkpoints/paper_planning.md:507:| Mechanism | OSClass query routing (`/index.php?page=item&id=N`) means URL-path is uninformative — visual product comparison required for "find blue motorcycle" tasks. Image absence → P-SoM cls collapses toward DOM (Macro 1a 6/8 cells DOM-like). Image axis recovers at SoM. |
docs/checkpoints/paper_planning.md:636:**关键区分**: §3 4-dim 是 evidence layer 的**测量类型轴**, cross-X 是 evidence layer 的**比较 axis 轴**。两者**正交 organize 同一份数据**。Explanation layer 跟 evidence layer 严格分开 — explanation 是 hypothesis (Zoom 1-4), evidence 是 data。Paper writing 时 reviewer 最忌 evidence-explanation 混淆 ("Macro 1c search-loop 51.9→35.7%" 是 evidence, "M1 axis activates list-scanning trajectory" 是 explanation Zoom 2 — 两者必须分写然后 explicit link)。
docs/checkpoints/paper_planning.md:666:| **0a** Aggregate raw + adjusted SR per mode | summary_v2.json live | live | red P-SoM **adj 13.81%** > all baseline; red P-text **12.38%** > DOM **9.52%**; cls SoM **21.37%** (best); cls P-text/P-SoM **adj 14.53%** ≈ DOM 14.10% |
docs/checkpoints/paper_planning.md:690:| **2a** URL signature divergence | `axis1_microbehavior.{py,json,md}` (FRESH 04-29 codex + 我补 compound) | **axis 1 alone**: red Jaccard 0.573 / cls 0.904 (path-only). **compound DOM↔P-SoM**: red 0.481 / cls 0.885 path-only (但 cls path+query 0.66 —— OSClass 用 query routing). **决策真改了，aggregate macro 在 cls 上掩盖** |
docs/checkpoints/paper_planning.md:692:| **2c** Search-keyword reuse / repeat | `axis1_microbehavior.json` | red P-text vs DOM 重复 −0.633 (axis 1 减少死循环); cls P-text +0.077 (无 site 损失) |
docs/checkpoints/paper_planning.md:715:| **Axis 3 (image)** | secondary (cls SoM 21.37% > P-SoM 14.53%, image 决定性 cls 上) | **cls 5/8 dominant** (finish h=+0.57 medium-effect 最强信号); red 3/8 dominant (efficiency cluster) | image 加上 = URL Jaccard 0.46-0.60 minor change | **+700-1100 image tokens** (Efficiency 3a 主要 cost source) |
docs/checkpoints/paper_planning.md:727:| **C2**: 4-fold drop-in property (cost / latency / signal / drop-one) | (a) Efficiency 3a, (b) Efficiency 3c, (c) Outcome 0g, (d) Outcome 0c |
docs/checkpoints/paper_planning.md:728:| **C3**: 3-axis hierarchical theory | Macro 1b (cascade decomposition), Micro (axis-by-axis micro), Cross-dimension table |
docs/checkpoints/paper_planning.md:729:| **C4**: aggregate macro can mislead about routing potential (cls case) | Macro 1a (cls 6/8 DOM-like macro) + Outcome 0d (cls task-pool Jaccard 0.53) + Micro 2a (cls path+query Jaccard 0.66) |
docs/checkpoints/paper_planning.md:730:| **C5**: prompt as task-conditional decision prior (not commit-only) | Outcome 0b (FP rate), Outcome 0d (Jaccard 0.45-0.55 same-SR-different-pool), Macro 1b (cascade axis 2 dominant on red strategy metrics) |
docs/checkpoints/paper_planning.md:731:| **C6**: image is bidirectional 8-channel modality fusion | Macro 1b (cls image axis 5/8 dominant), Outcome 0e (codex audit category × mode), codex `7106d2e` channel decomposition |
docs/checkpoints/paper_planning.md:754:- cls SoM **adj 21.37% 显著领先 P-SoM 14.53% (+6.84pp)** —— 反例必须明示, image 在 cls 是决定性 axis (Macro 1b 5/8 dominant 印证)
docs/checkpoints/paper_planning.md:757:- §103 N=48 "5/5 metrics P-text = P-SoM" 已 **superseded** by N=210 (FRESH 04-29 Macro 1c) — 早期 small-sample artifact
docs/checkpoints/paper_planning.md:783:| **N2**: Tier 1 hook macro: red 4/8 cells fully independent / cls 1/8 (cls 主要 DOM-like) | **Macro 1a** |
docs/checkpoints/paper_planning.md:784:| **N3**: 6 antagonistic mechanism pairs（4-level cascade vs 2-endpoint 比较的核心 paper value） | **Macro 1b** |
docs/checkpoints/paper_planning.md:785:| **N4**: cls compound DOM↔P-SoM micro path+query Jaccard 0.66 | **Micro 2a** |
docs/checkpoints/paper_planning.md:800:- Micro: per-step 决策怎么变（URL / target / keyword）
docs/checkpoints/paper_planning.md:810:观测 (evidence): reddit axis 1 swap → search-loop 51.9 → 35.7 (Macro 1c) +
docs/checkpoints/paper_planning.md:811:                                       URL Jaccard 0.57 (Micro 2a) +
docs/checkpoints/paper_planning.md:812:                                       SR uplift 4.76pp drop-one (Outcome 0c)
docs/checkpoints/paper_planning.md:1435:- [x] **Forest plots (per-cell + meta)** ✅ done 2026-05-03 (T0b/T0d) — `fig_forest_drop_one.py` raw 95% CI + Holm-sig marker + TOST band; `fig_meta_forest.py` Hero+Ablation visual hierarchy with weight-sized squares + pooled diamond
docs/checkpoints/paper_planning.md:1723:| 2026-05-03 | Evidence layer + visualization audit infra (T0a-T0d done) | `aggregate_phantom_lift.py` Bonferroni/Holm/BH/TOST + H3 structural test cols; `aggregate_phantom_meta.py` DerSimonian-Laird random-effect; `fig_forest_drop_one.py` per-cell forest with Holm-sig markers; `fig_meta_forest.py` Hero+Ablation visual hierarchy; `fig_phantom_structure_venn.py` paper §1 centerpiece Venn; `make analysis [FAST=1]` end-to-end wired. | ✅ `docs/reference/EVIDENCE_LAYER_AUDIT.md` §3 T0 4/6 done |
docs/checkpoints/paper_planning.md:1924:| Runner reliability | §57 (tab_focus signature) / §58 (shell 孤儿 + stale summary) / §76 (atomic write + retry batch notification) / §85 (P0/P1/P2 batch 9 项) / §90 (keyword_finish 根除) / §87 (evaluator dirty page) / §97 (cross_rep 审计 RU/A/B 类) | Runner / pipeline 稳定性 |
docs/reference/EVIDENCE_LAYER_AUDIT.md:23:- `phantom_lift.{md,csv}` — Outcome 0c/0d (3→5/6-mode oracle lift, drop-one, Jaccard)
docs/reference/EVIDENCE_LAYER_AUDIT.md:58:| **A3** | **Cross-cell meta-analysis**: random-effect (DerSimonian-Laird) pooled drop-one per arm + I² heterogeneity statistic. New `aggregate_phantom_meta.py` + `meta_phantom_lift.md`. | **Forest plot pooled estimate** (`fig_meta_forest.py`) — per arm: cells listed vertically + pooled diamond at bottom + I² annotation | **T0** | 4h stats + 2h viz |
docs/reference/EVIDENCE_LAYER_AUDIT.md:291:- [x] **T0d — A3 paired viz** (2026-05-03): `fig_meta_forest.py` classical forest with weight-sized squares per cell + pooled diamond + I²/Q/τ² annotation + TOST band. Output `fig_meta_forest.png`.
docs/reference/EVIDENCE_LAYER_AUDIT.md:339:- 2026-05-03: T0d done. New `fig_meta_forest.py` uses classical forest convention — weight-sized squares per cell + pooled diamond. Design choice: keep 3-panel structure even when k=1 (P-prompt) so the layout auto-upgrades when 14-cell rerun lands without redesign.
docs/reference/EVIDENCE_LAYER_AUDIT.md:340:- 2026-05-03: T0a-d Makefile integration done. `phantom-meta` target added to `_aggregate`; `fig_forest_drop_one.py` + `fig_meta_forest.py` added to both `_figures` (full analysis pipeline) and `figures:` (quick regen). `make analysis [FAST=1]` and `make figures` end-to-end confirmed working.
docs/checkpoints/codex_prompts/refactor_phase1_run_registry.md:5:**Out of scope**: `make analysis` consolidation (Phase 2), new figures for Micro 2b-2f / Efficiency 3c (Phase 3)
docs/checkpoints/codex_prompts/refactor_phase1_run_registry.md:270:- **Phase 3** (new figures for Micro 2b/2c/2d/2e/2f + Efficiency 3c): no figure additions.
docs/checkpoints/codex_prompts/fix_figures_and_layer0_aggregator.md:84:如果觉得 Jaccard 一种太单薄，可以做 **2×N panel grid**：每 panel 一种 metric (URL Jaccard / target-hit diff / first-action divergence / keyword-repeat diff)，per metric 显示 5 contrast × 2 site 的 heatmap。但保持 figure 紧凑（不要超过 4 panel）。
docs/checkpoints/codex_prompts/section5_reddit_prose.md:35:    * Outcome 0c: +P-text adds +3.81pp drop-one oracle lift over 3-mode
docs/checkpoints/codex_prompts/section5_reddit_prose.md:36:    * Macro 1c: search-loop% DOM 51.9% → P-SoM 35.7% → SoM 31.4% (N=210 全数据, FRESH 04-29)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:37:    * Micro 2a: URL Jaccard 0.573 axis-1-alone (path-only signature)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:38:    * Micro 2b: target-page hit rate +3.47pp axis 1
docs/checkpoints/codex_prompts/section5_reddit_prose.md:39:    * Micro 2c: keyword-repeat -0.633 (axis 1 减少死循环)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:51:    * Outcome 0c: +P-SoM (axis 2 on top of P-text) adds another +3.33pp drop-one
docs/checkpoints/codex_prompts/section5_reddit_prose.md:53:    * Macro 1b: axis 2 cascade 3/8 dominant on red strategy metrics (search/type/scroll)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:66:    * Macro 1b: cls 5/8 axis 3 dominant vs red 3/8 (差异显著)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:67:    * Efficiency 3b: red token cost +733 image tokens, no SR benefit
docs/checkpoints/codex_prompts/section5_reddit_prose.md:108:docs/analysis/cross_sites/phantom_lift.csv (-> results/phantom_paper/)  ← Outcome 0c oracle lift
docs/checkpoints/codex_prompts/section5_reddit_prose.md:109:docs/analysis/cross_sites/axis_effect_size_report.md      ← Macro 1a/1b cascade
docs/checkpoints/codex_prompts/section5_reddit_prose.md:110:docs/analysis/cross_sites/axis1_microbehavior_report.md   ← Micro 2a-2e
docs/checkpoints/codex_prompts/section5_reddit_prose.md:131:- 每条 evidence cite 时用 `(Outcome 0c +3.81pp drop-one oracle lift)` 或 `(Macro 1c, search-loop 51.9%→35.7%)` 这种内联格式
docs/checkpoints/codex_prompts/refactor_layered_analysis.md:22:Layer 2  Micro Behavior    per-step decision (URL/keyword/element)
docs/checkpoints/codex_prompts/axis1_microbehavior.md:7:直接 element-id Jaccard 不可行（AXTree id 和 [SOM_MARKS] mark 不同 ID 系统，不可比）。改用 **mode-invariant** 的 anchor：URL trajectory、target-page hit、search keyword、first-action transition。
docs/checkpoints/codex_prompts/axis1_microbehavior.md:14:- cls 的 search 是 OSClass 任务**正常路径**（参考 `scripts/analysis/figures/fig1c_strategy_gradient.py` footnote: "OSClass tasks intrinsically use search pages"），所以 search-keyword 重复率 absolute 值都高——要看 **axis 间 differential**（DOM vs P-DOM 的差），不看绝对 level
docs/checkpoints/codex_prompts/axis1_microbehavior.md:48:target URL 从 `eval.reference_url` 或 `eval.program_html[].url` 提取（取第一个非 empty 的 URL）；如果没有就跳过该 task 的 target-hit metric（其他 metric 仍可计算）。
docs/checkpoints/codex_prompts/axis1_microbehavior.md:60:    "search_keywords": [action.text for type-actions],       # 所有打字动作 text（lowercased, stripped）
docs/checkpoints/codex_prompts/axis1_microbehavior.md:61:    "n_type_actions": len(search_keywords),
docs/checkpoints/codex_prompts/axis1_microbehavior.md:62:    "max_keyword_repeat": max(Counter(search_keywords).values() or [0]),  # 最重复的 keyword 出现次数（>5 = search loop)
docs/checkpoints/codex_prompts/axis1_microbehavior.md:63:    "distinct_keywords": len(set(search_keywords)),
docs/checkpoints/codex_prompts/axis1_microbehavior.md:77:| `DOM vs P-DOM` | axis 1 (text) | URL-set Jaccard / target-hit-rate diff / search-keyword diff |
docs/checkpoints/codex_prompts/axis1_microbehavior.md:84:- **Max keyword repeat diff**: per axis, paired diff in `max_keyword_repeat` (search-loop intensity)
docs/checkpoints/codex_prompts/axis1_microbehavior.md:85:- **Distinct keywords diff**: paired diff in `distinct_keywords` (是否多样 reformulate vs 单一关键词重复)
docs/checkpoints/codex_prompts/axis1_microbehavior.md:93:- "decision-quality bucket": URL-Jaccard, target-hit, first-action divergence
docs/checkpoints/codex_prompts/axis1_microbehavior.md:124:        "max_keyword_repeat_diff": -3.5, /* P-DOM has 3.5 less repeat */
docs/checkpoints/codex_prompts/axis1_microbehavior.md:154:- target-hit rate per task 计算时 task_config 里能 extract URL 的 task 子集报清楚 N
docs/checkpoints/pre_run/pre_rerun_audit.md:476:| 4.4.b.5 | **Analysis pipeline 4-dimension Evidence Framework** (笔记 §106) | ✅ pre-spec'd | `scripts/analysis/aggregate_phantom_lift.py` (Outcome 0c/0d) + `aggregate_routing_auroc.py` (0g) + `axis_effect_size.py` (1a/1b) + `axis1_microbehavior.py` (2a-2e) + `aggregate_cross_site.py` (3a-3c) + `figures/` per-outcome scripts |
docs/checkpoints/codex_prompts/B0_dom_shopping_A_refine.md:17:| **A3_AGGREGATION** | intent 含 比较 / 排序 / aggregation keyword: least/most expensive, cheapest, highest rated, lowest, most reviews, average, sum, count, top-N 等。需要遍历 list (Magento 12 items × ~10 fields) | 必败 (<5%) |
docs/checkpoints/codex_prompts/B0_dom_shopping_A_refine.md:38:    "primary_signals": ["least expensive (aggregation keyword)", "red (visual attribute, secondary)"],
docs/checkpoints/codex_prompts/mechanism_per_task_explanation.md:10:- **E4 action vocabulary distribution**：full action_type × subtype (click→link/button/textbox; type+keyword) distribution per (mode, site)
docs/literature/phantom_som.md:329:Kim et al. [23] introduced KiC (Keyword-inspired Cascade), which accepts cheaper-model outputs when representative keywords are semantically aligned across samples and otherwise escalates.
docs/literature/phantom_som.md:331:**Key Finding**: KiC accepts cheaper-model outputs when representative keywords are semantically aligned and otherwise escalates, achieving high accuracy with reduced API cost [23].
docs/literature/phantom_som.md:333:**Method Summary**: KiC identifies a representative response among multiple cheap-model outputs and evaluates semantic alignment via keyword-based checks [23].
docs/analysis/experiment_matrix.md:29:| | som | **210/210** | 11.90% (25) | **10.48%** (22) | ✅ 完成 |
docs/checkpoints/_status/issues/issue_14cell_phantom_rerun.md:57:- 13 figures including `fig_forest_drop_one.png` / `fig_meta_forest.png` (Hero+Ablation hierarchy) / `fig_phantom_structure_venn.png` (paper §1 centerpiece)
docs/literature/5.1/Cost-Aware Routing for Vision-Language Web Agents An Empirical Analysis of Text-Only Accuracy Retention.md:73:| GPT-4V (Multimodal)            | Image + Caps + Acc. Tree | 8.12                | 12.38          | 19.74            | 15.05              |
docs/checkpoints/advisor_sync_5_5_outcomes.md:247:| `fig_meta_forest.png` | `results/phantom_paper/figures/` | Hero + Ablation forest plot |
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:6:**Out of scope**: Phase 3 (new figures for Micro 2b/2c/2d/2e/2f + Efficiency 3c)
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:221:- **Phase 3** (new figures for Micro 2b/2c/2d/2e/2f + Efficiency 3c latency): not this phase.
docs/checkpoints/实验笔记.md:208:| 90 | keyword_finish 根除 + GLM Prompt 升级 | 04-24 | keyword_finish 是 scaffold confound（22 B0 + 1 B1 episode 被误杀）；移除 keyword_finish+GLM 可观测性+prompt 升级；23 episode 清除补跑 |
docs/checkpoints/实验笔记.md:229:| 63 | `<think>` 标签导致 parse_error → keyword_scroll | 04-15 | Qwen3-235B 输出 `<think>...</think>` extended thinking；regex DOTALL 贪婪捕获 think 块内容 → JSON 解析失败 → keyword_scroll（thought 丢失）；DOM 16 tasks / Vision 33 tasks；`action_utils.py` 解析前剥离 think 标签 |
docs/checkpoints/实验笔记.md:1000:## 63. `<think>` 标签导致 parse_error → keyword_scroll（2026-04-15） [bug] #bug
docs/checkpoints/实验笔记.md:1002:**症状**：B0 某些 steps `error_category=parse_error`，`thought=''`，`tokens.output` 明显偏大（300-600 vs 正常 100-150），action 退化为 keyword_scroll（delta=[0,0.8]）。DOM 18 steps/16 tasks，Vision 38 steps/33 tasks。
docs/checkpoints/实验笔记.md:1004:**根因**：Qwen3-235B-A22B 在某些步骤输出 `<think>...</think>` extended thinking 块，`parse_action_text` 先尝试直接 `json.loads`（失败），再用 `re.search(r"\{.*\}", text, re.DOTALL)` DOTALL 贪婪匹配——从 think 块内第一个 `{` 一直到文末最后一个 `}`，捕获到非法 JSON → 解析失败 → keyword_scroll。
docs/checkpoints/实验笔记.md:1045:**背景**：B0 SoM parse_error 率 ~20.1%（B0_SOM_digest），根因是 Bedrock 不支持 `enable_thinking` 字段分离，Qwen3-235B 的推理文本和 action JSON 混在同一 output 中，`parse_action_text` 的 keyword fallback 误判（"finish"/"scroll" 出现在 thought 自然语言中即触发）。`keyword_finish` 直接杀死 episode（task 4 SoM），`keyword_scroll` 蒙对方向但 thought 丢失。
docs/checkpoints/实验笔记.md:1053:**解析链路**：tool_use → text parse (json.loads→regex) → GLM extract → keyword fallback。
docs/checkpoints/实验笔记.md:1093:**方案 B（GLM fallback）验证通过**：冒烟测试 4/4 场景正确（think_block / markdown / scroll / keyword_trap）。在线验证 2/2 parse_error 成功救回：task_13 `keyword_scroll→finish`（32.5s）、task_16 `keyword_scroll→click eid=854`（49.5s，`reward=1.0`）。
docs/checkpoints/实验笔记.md:1478:## 90. keyword_finish 根除 + GLM Prompt 升级（2026-04-24） [bug] #bug
docs/checkpoints/实验笔记.md:1482:§67 通过 GLM fallback 将 parse_error 从 ~20% 降至 2-4%，但残留了一个致命路径：`action_utils.py` 的 keyword fallback 在 JSON 解析失败后扫描原始文本，若含 "finish"/"stop" 则直接触发 `keyword_finish`，绕过 GLM 恢复机会，episode 被脚手架提前终止（`fallback_finish=True`）。
docs/checkpoints/实验笔记.md:1497:**这是 scaffold confound**：B0（235B API）产生冗长输出命中 keyword_finish 的概率远高于 B1（4B local，输出结构化 JSON），导致 B0 vs B1 对比不公平。
docs/checkpoints/实验笔记.md:1501:**Fix 1：移除 keyword_finish（`action_utils.py`）**
docs/checkpoints/实验笔记.md:1503:删除 keyword fallback 中的 finish/stop 分支。不可解析的文本含 "finish" 时现在降级为 `wait`（parse_failed），episode 继续运行而非被终止。保留 `keyword_scroll` 和 `keyword_back` 作为非致命 heuristic。
docs/checkpoints/实验笔记.md:1506:# NOTE: keyword_finish removed (§67 follow-up).  "finish"/"stop" appearing
docs/checkpoints/实验笔记.md:2872:| B0 red P-text (FRESH) | 02:12 | 13.81% / 11.90% |
docs/checkpoints/实验笔记.md:3102:Micro       per-step decision (URL / target / keyword, fig2 micro)
docs/checkpoints/实验笔记.md:3146:发现：`condition_summary_v2.json` 的 `avg_total_cost_usd` 字段对 B0/B1 都用 Qwen3-VL-235B-A22B per-token rate ($0.001/1k input + $0.005/1k output) 算，导致 **B0/B1 ratio ~1.0×**（artifact）。但 paper §3 finding 7 / Efficiency 3d 旧 claim "30× cost gap" —— 实验笔记 grep 不到原始来源，凭空数字。
docs/checkpoints/实验笔记.md:3153:- **新建** `scripts/analysis/aggregate_cost_electricity.py` (Efficiency 3a + 3d standalone aggregator) → `docs/analysis/cross_sites/cost_per_mode.{json,md}`
docs/checkpoints/实验笔记.md:3161:`paper_planning §3` Efficiency 3d + Legacy index finding 7 都 update 标 "**~100×** deployment-class gap, NOT capability ratio. §103 / paper-planning legacy '30×' claim **superseded**".
docs/checkpoints/实验笔记.md:3163:`fig3d_cost_sr_frontier.png` 重做：x-axis log-scale + 顶部 "B0 API \$ vs B1 electricity-equivalent \$ (different cost classes)" + "B0/B1 ~100× deployment-class gap" annotation。同图同时 visual evidence Efficiency 3a (P-SoM ≈ DOM cost intra-baseline) + Efficiency 3d (deployment-class gap)。
docs/checkpoints/实验笔记.md:3172:- 12 scripts/figures 头部 docstring 加 dimension+sub-code 标签（e.g. `[Outcome 0c]`, `[Macro 1c]`, `[Micro 2a-2e]`, `[Efficiency 3a]`）
docs/checkpoints/实验笔记.md:3198:├── phantom_lift.{md,csv}            Outcome 0c+0d
docs/checkpoints/实验笔记.md:3207:├── fig2_micro_divergence_heatmap.png  URL/target/keyword 4-panel
docs/checkpoints/实验笔记.md:3213:├── cost_per_mode.{json,md}          Efficiency 3a + 3d standalone (electricity-equivalent)
docs/checkpoints/实验笔记.md:3214:└── run_summary_collect.json         Efficiency 3b token data
docs/checkpoints/实验笔记.md:3259:**paper Section 5 对 cls 的写法**: "macro action-type 频率 average 掩盖了 per-task decision divergence; routing arm 价值需 outcome (Outcome 0d) + micro (Micro 2a) 一起证，不能只 cite macro"。
docs/checkpoints/实验笔记.md:3281:- `scripts/analysis/aggregate_cost_electricity.py` (Efficiency 3a/3d)
docs/checkpoints/实验笔记.md:3282:- `scripts/analysis/axis_effect_size.py` (Macro 1a/1b cascade ablation)
docs/checkpoints/实验笔记.md:3283:- `scripts/analysis/axis1_microbehavior.py` (Micro 2a-2e)
docs/checkpoints/实验笔记.md:3295:- `docs/checkpoints/paper_planning.md` §3 重组 + Efficiency 3d 100× 修正
docs/checkpoints/实验笔记.md:3341:- **`make analyze-layered` Efficiency 3d cost ratio "1×" warning**：layered_status.py 已 fallback 到 cost_per_mode.json (~100×)，警告其实是从 condition_summary 直读的 artifact 信号，不影响 paper 数字
docs/checkpoints/实验笔记.md:3650:- Micro (per-step): denominator confound (Section 5 prose 已用 Micro 2f first-divergence patch 这层)
docs/checkpoints/实验笔记.md:5307:- `p79/utils/auth_refresh.py::should_refresh()` 加 `seconds_since_refresh` keyword arg + time-based check (default 1200s, below PHP `gc_maxlifetime=1440s`)
docs/checkpoints/实验笔记.md:5588:- 4-dimension Evidence Framework (笔记 §106) — Outcome 0c/0d (phantom_lift), 0g (auroc), 1a/1b (axis_effect_size), 2a-2e (axis1_microbehavior), 3a-3c (cross_site)
docs/checkpoints/实验笔记.md:5766:- 其他 phantom routing figures (fig_meta_forest, fig0c_*, fig_phantom_structure_venn) 仍是 priority — 它们 support 1+2 contribution
docs/checkpoints/实验笔记.md:6354:- Narrative mapping 严格: 每条都写 "P79 关联" 一句话, 没有 keyword-only 撞名词
docs/checkpoints/paper_planning.md:495:| Image axis sub-effects | Macro 1b axis 3 small (5/5 reddit metrics show image effect d_z<0.16) — image is content not navigation, helping/harming roughly balanced |
docs/checkpoints/paper_planning.md:506:| Mechanically dominant axis | **Axis 3 (image)** — Macro 1b cls image axis dominates 5/8 metrics (h=+0.57 finish rate, d=−0.42 action repeat) |
docs/checkpoints/paper_planning.md:507:| Mechanism | OSClass query routing (`/index.php?page=item&id=N`) means URL-path is uninformative — visual product comparison required for "find blue motorcycle" tasks. Image absence → P-SoM cls collapses toward DOM (Macro 1a 6/8 cells DOM-like). Image axis recovers at SoM. |
docs/checkpoints/paper_planning.md:636:**关键区分**: §3 4-dim 是 evidence layer 的**测量类型轴**, cross-X 是 evidence layer 的**比较 axis 轴**。两者**正交 organize 同一份数据**。Explanation layer 跟 evidence layer 严格分开 — explanation 是 hypothesis (Zoom 1-4), evidence 是 data。Paper writing 时 reviewer 最忌 evidence-explanation 混淆 ("Macro 1c search-loop 51.9→35.7%" 是 evidence, "M1 axis activates list-scanning trajectory" 是 explanation Zoom 2 — 两者必须分写然后 explicit link)。
docs/checkpoints/paper_planning.md:666:| **0a** Aggregate raw + adjusted SR per mode | summary_v2.json live | live | red P-SoM **adj 13.81%** > all baseline; red P-text **12.38%** > DOM **9.52%**; cls SoM **21.37%** (best); cls P-text/P-SoM **adj 14.53%** ≈ DOM 14.10% |
docs/checkpoints/paper_planning.md:690:| **2a** URL signature divergence | `axis1_microbehavior.{py,json,md}` (FRESH 04-29 codex + 我补 compound) | **axis 1 alone**: red Jaccard 0.573 / cls 0.904 (path-only). **compound DOM↔P-SoM**: red 0.481 / cls 0.885 path-only (但 cls path+query 0.66 —— OSClass 用 query routing). **决策真改了，aggregate macro 在 cls 上掩盖** |
docs/checkpoints/paper_planning.md:692:| **2c** Search-keyword reuse / repeat | `axis1_microbehavior.json` | red P-text vs DOM 重复 −0.633 (axis 1 减少死循环); cls P-text +0.077 (无 site 损失) |
docs/checkpoints/paper_planning.md:715:| **Axis 3 (image)** | secondary (cls SoM 21.37% > P-SoM 14.53%, image 决定性 cls 上) | **cls 5/8 dominant** (finish h=+0.57 medium-effect 最强信号); red 3/8 dominant (efficiency cluster) | image 加上 = URL Jaccard 0.46-0.60 minor change | **+700-1100 image tokens** (Efficiency 3a 主要 cost source) |
docs/checkpoints/paper_planning.md:727:| **C2**: 4-fold drop-in property (cost / latency / signal / drop-one) | (a) Efficiency 3a, (b) Efficiency 3c, (c) Outcome 0g, (d) Outcome 0c |
docs/checkpoints/paper_planning.md:728:| **C3**: 3-axis hierarchical theory | Macro 1b (cascade decomposition), Micro (axis-by-axis micro), Cross-dimension table |
docs/checkpoints/paper_planning.md:729:| **C4**: aggregate macro can mislead about routing potential (cls case) | Macro 1a (cls 6/8 DOM-like macro) + Outcome 0d (cls task-pool Jaccard 0.53) + Micro 2a (cls path+query Jaccard 0.66) |
docs/checkpoints/paper_planning.md:730:| **C5**: prompt as task-conditional decision prior (not commit-only) | Outcome 0b (FP rate), Outcome 0d (Jaccard 0.45-0.55 same-SR-different-pool), Macro 1b (cascade axis 2 dominant on red strategy metrics) |
docs/checkpoints/paper_planning.md:731:| **C6**: image is bidirectional 8-channel modality fusion | Macro 1b (cls image axis 5/8 dominant), Outcome 0e (codex audit category × mode), codex `7106d2e` channel decomposition |
docs/checkpoints/paper_planning.md:754:- cls SoM **adj 21.37% 显著领先 P-SoM 14.53% (+6.84pp)** —— 反例必须明示, image 在 cls 是决定性 axis (Macro 1b 5/8 dominant 印证)
docs/checkpoints/paper_planning.md:757:- §103 N=48 "5/5 metrics P-text = P-SoM" 已 **superseded** by N=210 (FRESH 04-29 Macro 1c) — 早期 small-sample artifact
docs/checkpoints/paper_planning.md:783:| **N2**: Tier 1 hook macro: red 4/8 cells fully independent / cls 1/8 (cls 主要 DOM-like) | **Macro 1a** |
docs/checkpoints/paper_planning.md:784:| **N3**: 6 antagonistic mechanism pairs（4-level cascade vs 2-endpoint 比较的核心 paper value） | **Macro 1b** |
docs/checkpoints/paper_planning.md:785:| **N4**: cls compound DOM↔P-SoM micro path+query Jaccard 0.66 | **Micro 2a** |
docs/checkpoints/paper_planning.md:800:- Micro: per-step 决策怎么变（URL / target / keyword）
docs/checkpoints/paper_planning.md:810:观测 (evidence): reddit axis 1 swap → search-loop 51.9 → 35.7 (Macro 1c) +
docs/checkpoints/paper_planning.md:811:                                       URL Jaccard 0.57 (Micro 2a) +
docs/checkpoints/paper_planning.md:812:                                       SR uplift 4.76pp drop-one (Outcome 0c)
docs/checkpoints/paper_planning.md:1435:- [x] **Forest plots (per-cell + meta)** ✅ done 2026-05-03 (T0b/T0d) — `fig_forest_drop_one.py` raw 95% CI + Holm-sig marker + TOST band; `fig_meta_forest.py` Hero+Ablation visual hierarchy with weight-sized squares + pooled diamond
docs/checkpoints/paper_planning.md:1723:| 2026-05-03 | Evidence layer + visualization audit infra (T0a-T0d done) | `aggregate_phantom_lift.py` Bonferroni/Holm/BH/TOST + H3 structural test cols; `aggregate_phantom_meta.py` DerSimonian-Laird random-effect; `fig_forest_drop_one.py` per-cell forest with Holm-sig markers; `fig_meta_forest.py` Hero+Ablation visual hierarchy; `fig_phantom_structure_venn.py` paper §1 centerpiece Venn; `make analysis [FAST=1]` end-to-end wired. | ✅ `docs/reference/EVIDENCE_LAYER_AUDIT.md` §3 T0 4/6 done |
docs/checkpoints/paper_planning.md:1924:| Runner reliability | §57 (tab_focus signature) / §58 (shell 孤儿 + stale summary) / §76 (atomic write + retry batch notification) / §85 (P0/P1/P2 batch 9 项) / §90 (keyword_finish 根除) / §87 (evaluator dirty page) / §97 (cross_rep 审计 RU/A/B 类) | Runner / pipeline 稳定性 |
docs/analysis/vwa_reddit/B0_findings.md:51:| SoM | 12.38% | 8.10% | 17.14% |
docs/checkpoints/codex_prompts/site_mechanism_dictionary.md:10:- `axis1_microbehavior.json` (Layer 2 micro) — URL/keyword signatures
docs/checkpoints/codex_prompts/site_mechanism_dictionary.md:45:          "evidence_layer_2": {"axis_1 URL Jaccard": 0.573, "click-target Jaccard": 0.463},
docs/checkpoints/codex_prompts/site_mechanism_dictionary.md:116:   - `docs/analysis/cross_sites/axis1_microbehavior.json` (Layer 2 URL Jaccard, target hit, keywords)
docs/checkpoints/_status/issues/issue_14cell_phantom_rerun.md:57:- 13 figures including `fig_forest_drop_one.png` / `fig_meta_forest.png` (Hero+Ablation hierarchy) / `fig_phantom_structure_venn.png` (paper §1 centerpiece)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:1:# Phase 3 — Evidence layer coverage figures (Micro 2b/2c/2d/2e/2f + Efficiency 3c)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:13:Paper §3 4-dim Evidence framework has 16 sub-codes (4 dim × 4 cross-X). Of these, **6 sub-codes have data but no dedicated figure**. This Phase fills the visualization gap so paper §5 mechanism prose can cite figures by name (currently `section5_mechanism_reddit.md` cites "Micro 2f, median first divergent step 0" but **no figure exists** — paper-writing blocker).
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:18:| Micro 2a URL signature divergence | ✅ axis1_microbehavior.json | ✅ fig2_micro_divergence_heatmap.png |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:19:| **Micro 2b Target-page hit rate** | ✅ axis1_microbehavior.json | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:20:| **Micro 2c Search-keyword reuse** | ✅ axis1_microbehavior.json | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:21:| **Micro 2d First-action divergence** | ✅ axis1_microbehavior.json | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:22:| **Micro 2e Cross-site validity ratio** | ✅ axis1_microbehavior.json (`cross_site_validity` field) | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:23:| **Micro 2f First-divergence step** | ✅ axis1_microbehavior.json (or computed live from steps JSONL) | ❌ no figure |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:24:| Efficiency 3a Token cost | ✅ condition_summary_v2.json | ✅ fig3a_token_cost_intra_baseline.png (bundled with 3c) |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:25:| **Efficiency 3c Latency per step** | ✅ condition_summary_v2.json (`avg_total_latency_ms` / `avg_steps`) | ❌ no dedicated figure (bundled in 3a) |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:26:| Efficiency 3d Cost-SR Pareto | ✅ paper_cost_usd | ✅ fig3d_cost_sr_frontier.png |
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:36:**Sub-code**: Micro 2b — Target-page hit rate per mode × site (paired comparison)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:42:**Visualization**: 4-panel bar chart (B0 cls / B0 red / B1 cls / B1 red), x-axis = mode (DOM/P-text/P-prompt/P-SoM/SoM/Vision), y-axis = target-hit-rate %. Annotate axis-1 effect (DOM → P-text delta) and axis-2 effect (P-text → P-SoM delta) on top of bars.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:48:### 2. `scripts/analysis/figures/fig2c_keyword_repeat.py`
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:50:**Sub-code**: Micro 2c — Search-keyword reuse / max-keyword-repeat per trajectory
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:52:**Data source**: `docs/analysis/cross_sites/axis1_microbehavior.json` (`max_keyword_repeat` field per mode per site, distribution stats)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:54:**Visualization**: 4-panel box plot (B0/B1 × cls/red), x-axis = mode, y-axis = max keyword repeat count per task. Show median + IQR + outliers. Optional overlay: paired delta arrows (DOM → P-text, P-text → P-SoM) for axis effect direction.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:56:**Output**: `results/phantom_paper/figures/fig2c_keyword_repeat.png`
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:62:**Sub-code**: Micro 2d — First-action divergence between mode pairs (% tasks where mode A and mode B chose different first action_type)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:75:**Sub-code**: Micro 2e — Cross-site validity ratio (does axis effect generalize cls → red?)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:79:**Visualization**: Single panel (or 2-panel B0/B1 if scaling is asymmetric). x-axis = axis (axis 1 / axis 2 / compound). y-axis = effect ratio (red effect / cls effect on Micro 2a-2d metric). Horizontal target line at 1.0 (perfect generalization). Annotate with which Micro sub-metric is being aggregated. **If ratio > 1.0, effect is reddit-amplified; if < 1.0, cls-amplified; if ≈ 1.0, generalize-symmetric.**
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:87:**Sub-code**: Micro 2f — First-divergence step distribution per mode pair × site (paper §5 prose 已 cite, missing figure is paper-writing blocker)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:93:**This is paper-cited figure** (`section5_mechanism_reddit.md` line 27/39/43/51 references "Micro 2f"). Section 5 prose currently reads "median first divergent step 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%)" — this figure must visualize that.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:101:**Sub-code**: Efficiency 3c — Per-step latency separated from cost (paper §1 hook 4-fold drop-in property (b))
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:121:	$(PYTHON) scripts/analysis/figures/fig2c_keyword_repeat.py
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:136:1. **Module docstring**: `"""[Micro 2X] <dim> dimension — <description>.\n\nOutput:\n- results/phantom_paper/figures/figXX.png\n\n<sub-code description>.\n\nSee docs/checkpoints/paper_planning.md §3 <dim> dimension framework.\n"""`
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:187:5. **fig2f matches paper §5 prose claim** (`section5_mechanism_reddit.md` cites "Micro 2f, N=15, median first divergent step 0, early divergence 100%" for B0 reddit P-text↔P-SoM pair) — figure must show this finding for that pair specifically.
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:194:- `docs/checkpoints/paper_planning.md` §3 Evidence framework (Micro 2a-2f / Efficiency 3a-3d sub-code definitions)
docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:195:- `docs/checkpoints/paper_drafts/section5_mechanism_reddit.md` (paper-cited Micro 2f references)
docs/analysis/cross_sites/axis_effect_size_report.md:8:  - **2b Micro** — per-step decision quality (separate analysis): URL trajectory Jaccard, target-page hit rate, search keyword reuse, first-action divergence.
docs/analysis/cross_sites/axis_effect_size_report.md:21:| B0 | reddit | finish rate | h=-0.10; -4.76 pp; [-12.38, +2.86] | h=+0.22★; +10.95 pp; [+3.33, +18.11] | — | ✅ |
docs/analysis/cross_sites/axis_effect_size_report.md:31:| B0 | classifieds | action repeat | d_z=+0.12★; +4.26 pp; [-0.32, +8.66] | d_z=-0.42★; -17.08 pp; [-22.40, -11.90] | ✅ | ✅ |
docs/analysis/cross_sites/axis_effect_size_report.md:68:| B0 | classifieds | action repeat | d_z=+0.01; +0.24 pp; [-3.77, +3.97] | d_z=+0.13; +4.02 pp; [+0.35, +8.10] | d_z=-0.42★; -17.08 pp; [-22.40, -11.90] | image | pass |
docs/analysis/cross_sites/axis_effect_size_report.md:89:Tracked separately in `axis1_microbehavior.{json,md}`. Macro action-frequency metrics (this file) average per-step decisions; micro metrics directly compare per-step element selection / page coverage / search keyword reuse via mode-invariant anchors (URL, action.text).
docs/checkpoints/codex_prompts/refactor_phase1_run_registry.md:5:**Out of scope**: `make analysis` consolidation (Phase 2), new figures for Micro 2b-2f / Efficiency 3c (Phase 3)
docs/checkpoints/codex_prompts/refactor_phase1_run_registry.md:270:- **Phase 3** (new figures for Micro 2b/2c/2d/2e/2f + Efficiency 3c): no figure additions.
docs/checkpoints/codex_prompts/refactor_layered_analysis.md:22:Layer 2  Micro Behavior    per-step decision (URL/keyword/element)
docs/checkpoints/codex_prompts/fix_figures_and_layer0_aggregator.md:84:如果觉得 Jaccard 一种太单薄，可以做 **2×N panel grid**：每 panel 一种 metric (URL Jaccard / target-hit diff / first-action divergence / keyword-repeat diff)，per metric 显示 5 contrast × 2 site 的 heatmap。但保持 figure 紧凑（不要超过 4 panel）。
docs/checkpoints/codex_prompts/mechanism_per_task_explanation.md:10:- **E4 action vocabulary distribution**：full action_type × subtype (click→link/button/textbox; type+keyword) distribution per (mode, site)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:35:    * Outcome 0c: +P-text adds +3.81pp drop-one oracle lift over 3-mode
docs/checkpoints/codex_prompts/section5_reddit_prose.md:36:    * Macro 1c: search-loop% DOM 51.9% → P-SoM 35.7% → SoM 31.4% (N=210 全数据, FRESH 04-29)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:37:    * Micro 2a: URL Jaccard 0.573 axis-1-alone (path-only signature)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:38:    * Micro 2b: target-page hit rate +3.47pp axis 1
docs/checkpoints/codex_prompts/section5_reddit_prose.md:39:    * Micro 2c: keyword-repeat -0.633 (axis 1 减少死循环)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:51:    * Outcome 0c: +P-SoM (axis 2 on top of P-text) adds another +3.33pp drop-one
docs/checkpoints/codex_prompts/section5_reddit_prose.md:53:    * Macro 1b: axis 2 cascade 3/8 dominant on red strategy metrics (search/type/scroll)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:66:    * Macro 1b: cls 5/8 axis 3 dominant vs red 3/8 (差异显著)
docs/checkpoints/codex_prompts/section5_reddit_prose.md:67:    * Efficiency 3b: red token cost +733 image tokens, no SR benefit
docs/checkpoints/codex_prompts/section5_reddit_prose.md:108:docs/analysis/cross_sites/phantom_lift.csv (-> results/phantom_paper/)  ← Outcome 0c oracle lift
docs/checkpoints/codex_prompts/section5_reddit_prose.md:109:docs/analysis/cross_sites/axis_effect_size_report.md      ← Macro 1a/1b cascade
docs/checkpoints/codex_prompts/section5_reddit_prose.md:110:docs/analysis/cross_sites/axis1_microbehavior_report.md   ← Micro 2a-2e
docs/checkpoints/codex_prompts/section5_reddit_prose.md:131:- 每条 evidence cite 时用 `(Outcome 0c +3.81pp drop-one oracle lift)` 或 `(Macro 1c, search-loop 51.9%→35.7%)` 这种内联格式
docs/checkpoints/codex_prompts/site_mechanism_dictionary.md:10:- `axis1_microbehavior.json` (Layer 2 micro) — URL/keyword signatures
docs/checkpoints/codex_prompts/site_mechanism_dictionary.md:45:          "evidence_layer_2": {"axis_1 URL Jaccard": 0.573, "click-target Jaccard": 0.463},
docs/checkpoints/codex_prompts/site_mechanism_dictionary.md:116:   - `docs/analysis/cross_sites/axis1_microbehavior.json` (Layer 2 URL Jaccard, target hit, keywords)
docs/checkpoints/codex_prompts/axis1_microbehavior.md:7:直接 element-id Jaccard 不可行（AXTree id 和 [SOM_MARKS] mark 不同 ID 系统，不可比）。改用 **mode-invariant** 的 anchor：URL trajectory、target-page hit、search keyword、first-action transition。
docs/checkpoints/codex_prompts/axis1_microbehavior.md:14:- cls 的 search 是 OSClass 任务**正常路径**（参考 `scripts/analysis/figures/fig1c_strategy_gradient.py` footnote: "OSClass tasks intrinsically use search pages"），所以 search-keyword 重复率 absolute 值都高——要看 **axis 间 differential**（DOM vs P-DOM 的差），不看绝对 level
docs/checkpoints/codex_prompts/axis1_microbehavior.md:48:target URL 从 `eval.reference_url` 或 `eval.program_html[].url` 提取（取第一个非 empty 的 URL）；如果没有就跳过该 task 的 target-hit metric（其他 metric 仍可计算）。
docs/checkpoints/codex_prompts/axis1_microbehavior.md:60:    "search_keywords": [action.text for type-actions],       # 所有打字动作 text（lowercased, stripped）
docs/checkpoints/codex_prompts/axis1_microbehavior.md:61:    "n_type_actions": len(search_keywords),
docs/checkpoints/codex_prompts/axis1_microbehavior.md:62:    "max_keyword_repeat": max(Counter(search_keywords).values() or [0]),  # 最重复的 keyword 出现次数（>5 = search loop)
docs/checkpoints/codex_prompts/axis1_microbehavior.md:63:    "distinct_keywords": len(set(search_keywords)),
docs/checkpoints/codex_prompts/axis1_microbehavior.md:77:| `DOM vs P-DOM` | axis 1 (text) | URL-set Jaccard / target-hit-rate diff / search-keyword diff |
docs/checkpoints/codex_prompts/axis1_microbehavior.md:84:- **Max keyword repeat diff**: per axis, paired diff in `max_keyword_repeat` (search-loop intensity)
docs/checkpoints/codex_prompts/axis1_microbehavior.md:85:- **Distinct keywords diff**: paired diff in `distinct_keywords` (是否多样 reformulate vs 单一关键词重复)
docs/checkpoints/codex_prompts/axis1_microbehavior.md:93:- "decision-quality bucket": URL-Jaccard, target-hit, first-action divergence
docs/checkpoints/codex_prompts/axis1_microbehavior.md:124:        "max_keyword_repeat_diff": -3.5, /* P-DOM has 3.5 less repeat */
docs/checkpoints/codex_prompts/axis1_microbehavior.md:154:- target-hit rate per task 计算时 task_config 里能 extract URL 的 task 子集报清楚 N
docs/checkpoints/codex_prompts/B0_dom_shopping_A_refine.md:17:| **A3_AGGREGATION** | intent 含 比较 / 排序 / aggregation keyword: least/most expensive, cheapest, highest rated, lowest, most reviews, average, sum, count, top-N 等。需要遍历 list (Magento 12 items × ~10 fields) | 必败 (<5%) |
docs/checkpoints/codex_prompts/B0_dom_shopping_A_refine.md:38:    "primary_signals": ["least expensive (aggregation keyword)", "red (visual attribute, secondary)"],
docs/checkpoints/pre_run/pre_rerun_audit.md:476:| 4.4.b.5 | **Analysis pipeline 4-dimension Evidence Framework** (笔记 §106) | ✅ pre-spec'd | `scripts/analysis/aggregate_phantom_lift.py` (Outcome 0c/0d) + `aggregate_routing_auroc.py` (0g) + `axis_effect_size.py` (1a/1b) + `axis1_microbehavior.py` (2a-2e) + `aggregate_cross_site.py` (3a-3c) + `figures/` per-outcome scripts |
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:6:**Out of scope**: Phase 3 (new figures for Micro 2b/2c/2d/2e/2f + Efficiency 3c)
docs/checkpoints/codex_prompts/refactor_phase2_make_analysis_consolidation.md:221:- **Phase 3** (new figures for Micro 2b/2c/2d/2e/2f + Efficiency 3c latency): not this phase.
docs/checkpoints/ADVISOR_SYNC.md:104:| `fig_meta_forest.png` | Hero + Ablation forest plot, P-SoM hero 顶部黑框 |
docs/checkpoints/paper_drafts/section5_mechanism.md:119:This layered hierarchy resolves the Phantom-SoM hero puzzle without requiring a non-mechanistic explanation. Phantom-SoM uniquely combines three contributions: residual-stream proximity to SoM on the image axis at L17 (the largest single signal, with image-feature reduction "as if image were present"), separation from P-text on the prompt-family axis at L23 (a small 0.011 signal but consistent across cls and reddit), and separation from P-prompt on the text-format axis at L23 (a medium 0.029 signal). P-text occupies only the text-format separation; P-prompt occupies only the prompt-family separation against P-SoM; DOM occupies none of the three. The drop-one hero status of Phantom-SoM in `fig_meta_forest.png` therefore corresponds to the only mode that satisfies all three axis criteria.
docs/checkpoints/canvas/paper_section2_framework.canvas:263:      "text": "### Outcome × mode ⭐\n\nadj SR per arm:\n• B0 red SoM 10.48 / P-SoM 13.81\n• B0 red P-text 11.90 / P-prompt 10.48\n\n6-mode oracle vs 3-mode +7.14pp [3.81, 10.48] sig"
docs/checkpoints/paper_drafts/section4_empirical_findings.md:37:| Reddit | P-text | 11.90 | $0.046 | 58.1s | N=210 |
docs/checkpoints/paper_drafts/section4_empirical_findings.md:98:This substrate makes reddit an informative test case for separating the three axes. Images are frequent in the task prompts and in the posts themselves, but their role is usually evidential: an image can identify which post is being discussed, or disambiguate a content clue, but it is not the site's primary navigation affordance. The browser screenshot does not create the forum hierarchy; it only renders it. Conversely, the search box is prominent in the DOM and AXTree, but intrinsic search is not the intended substrate for many tasks. Repeated search is a failure basin: the agent can keep refining keywords while never taking the forum, post, or comment link that would satisfy the evaluator. The mechanism to explain is therefore not simply "text works better than vision." It is that each representation changes which textual affordances become salient enough for the model to commit to.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:104:The evidence chain is consistent across dimensions. At the outcome level, adding P-text to the three-mode baseline contributes oracle value even without the SoM prompt or screenshot (Outcome 0c, +P-text +3.21pp single-phantom lift on the current oracle intersection). At the macro level, the whole-run strategy gradient shows the failure basin directly: reddit search-loop rate falls from DOM to Phantom-SoM and then to full SoM (Macro 1c, search-loop 51.90%->35.71%->31.43%). The axis-1-only macro effect is smaller than the compound prompt path, which is expected if flat text mainly changes which page objects are reachable rather than merely changing the action vocabulary. The micro evidence is sharper: DOM versus P-text has low path overlap for a text-only swap (Micro 2a, URL-path Jaccard 0.573), improves target-page reach (Micro 2b, target-hit +3.47pp), and reduces repeated keyword reuse (Micro 2c, max-keyword-repeat -0.633). The click-target view tells the same story: the two modes choose substantially different element sets even before images enter the system (Micro 2a-extra, click-target Jaccard 0.463).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:106:The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:108:Efficiency further constrains the explanation. Because P-text is generated from the same AXTree-derived text source and does not attach a screenshot, the reddit improvement cannot be attributed to paying the visual-token tax (Efficiency 3a, DOM $0.0516/episode versus P-text $0.0459/episode in the site dictionary). Axis 1 is therefore a representation effect: the observation text is rearranged into an indexed list, not enriched with new visual evidence.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:116:The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:118:The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:120:The prompt contrast is also visible in the mode-invariant click-target metric. With the text payload held fixed, P-text and P-SoM still have low click-target overlap (Micro 2a-extra, P-text<->P-SoM click-target Jaccard 0.484). This matters because it rules out a purely verbal explanation in which the SoM prompt only changes confidence wording at `finish`. The prompt changes which marked objects are selected during navigation.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:128:The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:130:The image-axis micro contrast confirms that the screenshot is behaviorally strong even when it is not outcome-positive. P-SoM and SoM have low URL-path agreement and frequent immediate divergence (Micro 2a, image-axis URL-path Jaccard 0.456; Micro 2f, early divergence 95.24%). Thus "weak" should be read as weak net value on this substrate, not as weak causal force. The screenshot changes decisions; on reddit, those changed decisions often point to content assets rather than evaluator-relevant post or comment routes.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:134:The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/paper_drafts/section4_empirical_findings.md:138:The compound transition from DOM to Phantom-SoM combines the flat text payload with the marked-element prompt while still avoiding the image channel. On reddit this compound arm is best interpreted as a complementary routing arm rather than as a uniformly stronger baseline. Its adjusted SR is modestly higher than DOM on the full 210-task per-mode summaries (Outcome 0a, P-SoM 13.81% versus DOM 9.52%, +4.29pp), but the more robust mechanism evidence is the divergence in which routes it explores. DOM and P-SoM have low click-target overlap (Micro 2a-extra, compound click-target Jaccard 0.421) and low URL-path overlap (Micro 2a, compound URL-path Jaccard 0.481). The oracle result then follows naturally: P-SoM adds tasks that the original three-mode set misses (Outcome 0c, +P-SoM +2.56pp), and P-text/P-SoM are not redundant with each other (Outcome 0d, Jaccard 0.500).
docs/checkpoints/paper_drafts/section4_empirical_findings.md:142:The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/analysis/layered_evidence_status.md:13:- reddit: DOM raw 11.43% / adj **9.52%**; P-text raw 13.81% / adj **12.38%**; P-prompt raw 10.48% / adj **9.52%**; P-SoM raw 14.29% / adj **13.81%**; SoM raw 11.90% / adj **10.48%**; Vision raw 8.57% / adj **6.67%**
docs/analysis/layered_evidence_status.md:91:- reddit: DOM search-loop 51.90% → P-SoM search-loop 35.71% → SoM search-loop 31.43%
docs/analysis/layered_evidence_status.md:112:- reddit: axis-1 click-transition Jaccard **0.463**; compound DOM↔P-SoM **0.421**
docs/analysis/layered_evidence_status.md:124:- reddit: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
docs/analysis/layered_evidence_status.md:125:- classifieds: axis-1 max-keyword-repeat diff **n/a**; compound **n/a**
docs/analysis/layered_evidence_status.md:189:| Axis 1 text payload | 0c single-phantom lift | 1b text-axis cells, 1d action shifts | 2a-2e URL/target/keyword shifts, E1 click transitions | no image tax |
docs/analysis/cross_sites/tier1_dispatch_audit.json:56:      "fix_sketch": "Forward parsed_code[-1].arguments and parsed_code[-1].keywords into execute_playwright_select_option(). Add a controlled select-option regression test.",
docs/analysis/vwa_reddit/B1_findings.md:149:| Oracle ceiling | 12.38% (26 tasks) | **8.57%** (18 tasks) |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:378:| 4.4.b.5 | **Analysis pipeline 4-dimension Evidence Framework** (笔记 §106) | ✅ pre-spec'd | `scripts/analysis/aggregate_phantom_lift.py` (Outcome 0c/0d) + `aggregate_routing_auroc.py` (0g) + `axis_effect_size.py` (1a/1b) + `axis1_microbehavior.py` (2a-2e) + `aggregate_cross_site.py` (3a-3c) + `figures/` per-outcome scripts |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:677:41:The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1002:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:31:The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1003:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:53:The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1140:docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:24:| Efficiency 3a Token cost | ✅ condition_summary_v2.json | ✅ fig3a_token_cost_intra_baseline.png (bundled with 3c) |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1141:docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:26:| Efficiency 3d Cost-SR Pareto | ✅ paper_cost_usd | ✅ fig3d_cost_sr_frontier.png |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1143:docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:101:**Sub-code**: Efficiency 3c — Per-step latency separated from cost (paper §1 hook 4-fold drop-in property (b))
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1299:docs/checkpoints/codex_prompts/section5_reddit_prose.md:67:    * Efficiency 3b: red token cost +733 image tokens, no SR benefit
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1348:docs/analysis/cross_sites/site_mechanism_dictionary.md:237:Reddit's forum hierarchy makes representation and prompt the leading mechanisms. AXTree depth plus image-referential tasks push DOM into repeated keyword search; flat marks and SoM-style prompt priors open a different, low-cost trajectory basin. The image channel is useful for some content tasks but is not the site navigation substrate, so full SoM is not a monotone improvement over P-SoM.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1481:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:43:The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1482:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:59:The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1483:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:67:The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1507:docs/analysis/cross_sites/site_mechanism_dictionary.md:669:| Axis 1 text | PRIMARY: URL Jaccard 0.573, click-target Jaccard 0.463, +P-text +3.81pp oracle | Secondary: URL Jaccard 0.904, click-target Jaccard 0.561, +P-text +3.42pp oracle | Mixed/form-action; sparse evidence: click failure 24.7%, swatch bug 11 episodes |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1531:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:43:The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1532:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:53:The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1533:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:59:The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1534:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:67:The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:378:| 4.4.b.5 | **Analysis pipeline 4-dimension Evidence Framework** (笔记 §106) | ✅ pre-spec'd | `scripts/analysis/aggregate_phantom_lift.py` (Outcome 0c/0d) + `aggregate_routing_auroc.py` (0g) + `axis_effect_size.py` (1a/1b) + `axis1_microbehavior.py` (2a-2e) + `aggregate_cross_site.py` (3a-3c) + `figures/` per-outcome scripts |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:677:41:The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1002:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:31:The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1003:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:53:The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1140:docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:24:| Efficiency 3a Token cost | ✅ condition_summary_v2.json | ✅ fig3a_token_cost_intra_baseline.png (bundled with 3c) |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1141:docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:26:| Efficiency 3d Cost-SR Pareto | ✅ paper_cost_usd | ✅ fig3d_cost_sr_frontier.png |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1143:docs/checkpoints/codex_prompts/refactor_phase3_evidence_coverage_figures.md:101:**Sub-code**: Efficiency 3c — Per-step latency separated from cost (paper §1 hook 4-fold drop-in property (b))
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1299:docs/checkpoints/codex_prompts/section5_reddit_prose.md:67:    * Efficiency 3b: red token cost +733 image tokens, no SR benefit
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1348:docs/analysis/cross_sites/site_mechanism_dictionary.md:237:Reddit's forum hierarchy makes representation and prompt the leading mechanisms. AXTree depth plus image-referential tasks push DOM into repeated keyword search; flat marks and SoM-style prompt priors open a different, low-cost trajectory basin. The image channel is useful for some content tasks but is not the site navigation substrate, so full SoM is not a monotone improvement over P-SoM.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1481:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:43:The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1482:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:59:The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1483:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:67:The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1507:docs/analysis/cross_sites/site_mechanism_dictionary.md:669:| Axis 1 text | PRIMARY: URL Jaccard 0.573, click-target Jaccard 0.463, +P-text +3.81pp oracle | Secondary: URL Jaccard 0.904, click-target Jaccard 0.561, +P-text +3.42pp oracle | Mixed/form-action; sparse evidence: click failure 24.7%, swatch bug 11 episodes |
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1531:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:43:The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1532:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:53:The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1533:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:59:The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/codex_outputs/section8_limitations_consolidation_2026-05-09.md:1534:docs/checkpoints/paper_drafts/section5_mechanism_reddit.md:67:The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/analysis/cross_sites/site_mechanism_dictionary.json:27:            "sr_fp": "B0 adjusted SR: DOM 9.52%, P-text 12.38%, P-SoM 13.81%, SoM 10.48%, Vision 6.67% [sr_fp_per_mode.json]"
docs/analysis/cross_sites/site_mechanism_dictionary.json:31:            "strategy_gradient": "reddit search-loop: DOM 51.90% -> P-SoM 35.71% -> SoM 31.43% [layered_evidence_status.md]"
docs/analysis/cross_sites/site_mechanism_dictionary.json:35:            "click_target_jaccard": 0.46345,
docs/analysis/cross_sites/site_mechanism_dictionary.json:37:            "max_keyword_repeat_diff": -0.633333,
docs/analysis/cross_sites/site_mechanism_dictionary.json:50:          "digest_quote": "DOM repeats the same keyword 5-15 times; examples include task 23 'pumpkin robot' x15 and task 4 'wheat field city skyline' x7 [B0_DOM_digest.md §2.1]."
docs/analysis/cross_sites/site_mechanism_dictionary.json:181:            "max_keyword_repeat_diff": 0.076923,
docs/analysis/cross_sites/site_mechanism_dictionary.json:426:        "reddit (PRIMARY: URL Jaccard 0.573, click-target Jaccard 0.463, +P-text +3.81pp oracle)",
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:15:- phantom_text: 12.38%
docs/checkpoints/mechanism/results/hero_claim_bootstrap_ci.md:15:- phantom_text: 12.38%
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:7:    "primary_signals": ["least expensive (aggregation keyword)", "red (visual attribute, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:15:    "primary_signals": ["least expensive (aggregation keyword)", "blue (visual attribute, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:23:    "primary_signals": ["least expensive (aggregation keyword)", "grapes (latent visual content, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:31:    "primary_signals": ["most expensive (aggregation keyword)", "red (visual attribute, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:39:    "primary_signals": ["most expensive (aggregation keyword)", "black and white (visual attribute, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:47:    "primary_signals": ["most expensive (aggregation keyword)", "yellow (visual attribute, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:63:    "primary_signals": ["most expensive (aggregation keyword)", "red (visual attribute, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:71:    "primary_signals": ["most expensive (aggregation keyword)", "feather lamp (latent design, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:79:    "primary_signals": ["most expensive (aggregation keyword)", "blue (visual attribute, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:87:    "primary_signals": ["most expensive (aggregation keyword)", "color readout (visual attribute downstream)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:95:    "primary_signals": ["second cheapest (aggregation/rank keyword)", "color readout (downstream)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:103:    "primary_signals": ["third most expensive (aggregation/rank keyword)", "color readout (downstream)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:112:    "rationale": "No aggregation keyword; agent must filter search results by color (a latent visual attribute) without a reference image. Color may appear in title/swatches in DOM but requires inference."
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:175:    "primary_signals": ["cheapest (aggregation keyword)", "color readout (downstream)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:183:    "primary_signals": ["cheapest (aggregation keyword)", "blue (color filter, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:191:    "primary_signals": ["cheapest (aggregation keyword)", "cupcake style (latent visual design, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:215:    "primary_signals": ["least expensive (aggregation keyword)", "opaque (visual material/style, secondary)", "iPhone X (text spec)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:343:    "primary_signals": ["most expensive (aggregation keyword)", "review form (secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:360:    "rationale": "White color and slide design are latent visual descriptors needing inference; the conditional checkout follows. No aggregation keyword."
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:495:    "primary_signals": ["cheapest (aggregation keyword)", "color readout (downstream)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:551:    "primary_signals": ["cheapest (aggregation keyword)", "in a box (latent packaging/visual, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:567:    "primary_signals": ["cheapest (aggregation keyword)", "box wine (text/category descriptor)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:615:    "primary_signals": ["cheapest (aggregation keyword)", "fully white (color, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:623:    "primary_signals": ["cheapest (aggregation keyword)", "white (color, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:631:    "primary_signals": ["cheapest (aggregation keyword)", "anime (latent visual style, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:639:    "primary_signals": ["cheapest (aggregation keyword)", "round (visual shape, secondary)"],
docs/analysis/cross_sites/codex_audit_shopping_A_refined.json:647:    "primary_signals": ["cheapest (aggregation keyword)", "standing (form factor, mostly textual)"],
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:368:    37	| Reddit | P-text | 11.90 | $0.046 | 58.1s | N=210 |
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:429:    98	This substrate makes reddit an informative test case for separating the three axes. Images are frequent in the task prompts and in the posts themselves, but their role is usually evidential: an image can identify which post is being discussed, or disambiguate a content clue, but it is not the site's primary navigation affordance. The browser screenshot does not create the forum hierarchy; it only renders it. Conversely, the search box is prominent in the DOM and AXTree, but intrinsic search is not the intended substrate for many tasks. Repeated search is a failure basin: the agent can keep refining keywords while never taking the forum, post, or comment link that would satisfy the evaluator. The mechanism to explain is therefore not simply "text works better than vision." It is that each representation changes which textual affordances become salient enough for the model to commit to.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:435:   104	The evidence chain is consistent across dimensions. At the outcome level, adding P-text to the three-mode baseline contributes oracle value even without the SoM prompt or screenshot (Outcome 0c, +P-text +3.21pp single-phantom lift on the current oracle intersection). At the macro level, the whole-run strategy gradient shows the failure basin directly: reddit search-loop rate falls from DOM to Phantom-SoM and then to full SoM (Macro 1c, search-loop 51.90%->35.71%->31.43%). The axis-1-only macro effect is smaller than the compound prompt path, which is expected if flat text mainly changes which page objects are reachable rather than merely changing the action vocabulary. The micro evidence is sharper: DOM versus P-text has low path overlap for a text-only swap (Micro 2a, URL-path Jaccard 0.573), improves target-page reach (Micro 2b, target-hit +3.47pp), and reduces repeated keyword reuse (Micro 2c, max-keyword-repeat -0.633). The click-target view tells the same story: the two modes choose substantially different element sets even before images enter the system (Micro 2a-extra, click-target Jaccard 0.463).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:437:   106	The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:439:   108	Efficiency further constrains the explanation. Because P-text is generated from the same AXTree-derived text source and does not attach a screenshot, the reddit improvement cannot be attributed to paying the visual-token tax (Efficiency 3a, DOM $0.0516/episode versus P-text $0.0459/episode in the site dictionary). Axis 1 is therefore a representation effect: the observation text is rearranged into an indexed list, not enriched with new visual evidence.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:447:   116	The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:449:   118	The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:451:   120	The prompt contrast is also visible in the mode-invariant click-target metric. With the text payload held fixed, P-text and P-SoM still have low click-target overlap (Micro 2a-extra, P-text<->P-SoM click-target Jaccard 0.484). This matters because it rules out a purely verbal explanation in which the SoM prompt only changes confidence wording at `finish`. The prompt changes which marked objects are selected during navigation.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:459:   128	The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:461:   130	The image-axis micro contrast confirms that the screenshot is behaviorally strong even when it is not outcome-positive. P-SoM and SoM have low URL-path agreement and frequent immediate divergence (Micro 2a, image-axis URL-path Jaccard 0.456; Micro 2f, early divergence 95.24%). Thus "weak" should be read as weak net value on this substrate, not as weak causal force. The screenshot changes decisions; on reddit, those changed decisions often point to content assets rather than evaluator-relevant post or comment routes.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:465:   134	The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:469:   138	The compound transition from DOM to Phantom-SoM combines the flat text payload with the marked-element prompt while still avoiding the image channel. On reddit this compound arm is best interpreted as a complementary routing arm rather than as a uniformly stronger baseline. Its adjusted SR is modestly higher than DOM on the full 210-task per-mode summaries (Outcome 0a, P-SoM 13.81% versus DOM 9.52%, +4.29pp), but the more robust mechanism evidence is the divergence in which routes it explores. DOM and P-SoM have low click-target overlap (Micro 2a-extra, compound click-target Jaccard 0.421) and low URL-path overlap (Micro 2a, compound URL-path Jaccard 0.481). The oracle result then follows naturally: P-SoM adds tasks that the original three-mode set misses (Outcome 0c, +P-SoM +2.56pp), and P-text/P-SoM are not redundant with each other (Outcome 0d, Jaccard 0.500).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:473:   142	The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:596:   119	This layered hierarchy resolves the Phantom-SoM hero puzzle without requiring a non-mechanistic explanation. Phantom-SoM uniquely combines three contributions: residual-stream proximity to SoM on the image axis at L17 (the largest single signal, with image-feature reduction "as if image were present"), separation from P-text on the prompt-family axis at L23 (a small 0.011 signal but consistent across cls and reddit), and separation from P-prompt on the text-format axis at L23 (a medium 0.029 signal). P-text occupies only the text-format separation; P-prompt occupies only the prompt-family separation against P-SoM; DOM occupies none of the three. The drop-one hero status of Phantom-SoM in `fig_meta_forest.png` therefore corresponds to the only mode that satisfies all three axis criteria.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:1769:    15	- phantom_text: 12.38%
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:368:    37	| Reddit | P-text | 11.90 | $0.046 | 58.1s | N=210 |
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:429:    98	This substrate makes reddit an informative test case for separating the three axes. Images are frequent in the task prompts and in the posts themselves, but their role is usually evidential: an image can identify which post is being discussed, or disambiguate a content clue, but it is not the site's primary navigation affordance. The browser screenshot does not create the forum hierarchy; it only renders it. Conversely, the search box is prominent in the DOM and AXTree, but intrinsic search is not the intended substrate for many tasks. Repeated search is a failure basin: the agent can keep refining keywords while never taking the forum, post, or comment link that would satisfy the evaluator. The mechanism to explain is therefore not simply "text works better than vision." It is that each representation changes which textual affordances become salient enough for the model to commit to.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:435:   104	The evidence chain is consistent across dimensions. At the outcome level, adding P-text to the three-mode baseline contributes oracle value even without the SoM prompt or screenshot (Outcome 0c, +P-text +3.21pp single-phantom lift on the current oracle intersection). At the macro level, the whole-run strategy gradient shows the failure basin directly: reddit search-loop rate falls from DOM to Phantom-SoM and then to full SoM (Macro 1c, search-loop 51.90%->35.71%->31.43%). The axis-1-only macro effect is smaller than the compound prompt path, which is expected if flat text mainly changes which page objects are reachable rather than merely changing the action vocabulary. The micro evidence is sharper: DOM versus P-text has low path overlap for a text-only swap (Micro 2a, URL-path Jaccard 0.573), improves target-page reach (Micro 2b, target-hit +3.47pp), and reduces repeated keyword reuse (Micro 2c, max-keyword-repeat -0.633). The click-target view tells the same story: the two modes choose substantially different element sets even before images enter the system (Micro 2a-extra, click-target Jaccard 0.463).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:437:   106	The standalone success table supports the same interpretation without requiring oracle selection. P-text raises adjusted SR over DOM on the full reddit set while preserving the same no-image deployment class (Outcome 0a, DOM 9.52% versus P-text 12.38%, N=210). The effect is not simply that P-text acts more: it uses fewer steps on average in the cost summary, and the cascade's largest text-axis macro shift is action-repeat rather than search success itself (Macro 1b, text-axis action-repeat +4.64pp). This is compatible with a routing-surface mechanism. The agent sometimes repeats a newly exposed marked control, but it is no longer confined to the same query-rewrite loop.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:439:   108	Efficiency further constrains the explanation. Because P-text is generated from the same AXTree-derived text source and does not attach a screenshot, the reddit improvement cannot be attributed to paying the visual-token tax (Efficiency 3a, DOM $0.0516/episode versus P-text $0.0459/episode in the site dictionary). Axis 1 is therefore a representation effect: the observation text is rearranged into an indexed list, not enriched with new visual evidence.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:447:   116	The outcome evidence suggests a calibration effect rather than a uniform success-rate increase. Phantom-SoM has the lowest false-positive rate among the B0 reddit modes (Outcome 0b, P-SoM FP rate 0.48%), and adding P-SoM as a single phantom arm contributes additional oracle tasks beyond the three standard modes (Outcome 0c, +P-SoM +2.56pp). The task-pool overlap also stays below the redundancy sentinel: P-text and P-SoM solve overlapping but not identical task sets (Outcome 0d, P-text<->P-SoM Jaccard 0.500). Thus the prompt is not just a cosmetic instruction layered on top of the same decisions; it changes which tasks enter the solved pool.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:449:   118	The macro and trajectory evidence explain why. In the cascade, the prompt axis dominates the reddit search-loop and typing shifts: search-loop falls by 13.81pp from P-text to P-SoM, type fraction falls by 6.58pp, and scroll fraction falls by 3.79pp (Macro 1b, prompt-axis dominant on search/type/scroll). The action-vocabulary decomposition shows the same policy change in local form: P-SoM reduces type by 0.0658 and increases tab focus by 0.0828 relative to P-text (Macro 1d, action-vocabulary shift). The per-task boundary analysis shows that this is often an early decision-prior change, not a late recovery: in the P-text versus P-SoM symmetric-difference set, the median first divergent step is 0 and all divergent cases are early (Micro 2f, N=15, median first divergent step 0, early divergence 100%).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:451:   120	The prompt contrast is also visible in the mode-invariant click-target metric. With the text payload held fixed, P-text and P-SoM still have low click-target overlap (Micro 2a-extra, P-text<->P-SoM click-target Jaccard 0.484). This matters because it rules out a purely verbal explanation in which the SoM prompt only changes confidence wording at `finish`. The prompt changes which marked objects are selected during navigation.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:459:   128	The outcome result rejects a naive monotone-vision story. Full SoM is below Phantom-SoM on adjusted reddit success (Outcome 0a, SoM 10.48% versus P-SoM 13.81%, a -3.33pp regression after adding the image). The image still changes behavior: the cascade shows an increase in finish rate and fewer steps when the image is added (Macro 1b, image-axis finish rate +10.95pp; step count -1.85), but on reddit this can mean earlier commitment from the wrong state rather than more correct routing. The click fraction also falls under the image axis (Macro 1b, click fraction -3.08pp), which is consistent with visual confidence substituting for link-following in some traces. The cost side makes the tradeoff concrete: adding the screenshot increases median token load without a reddit SR benefit (Efficiency 3b, SoM versus P-SoM observed gap 778 tokens/step).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:461:   130	The image-axis micro contrast confirms that the screenshot is behaviorally strong even when it is not outcome-positive. P-SoM and SoM have low URL-path agreement and frequent immediate divergence (Micro 2a, image-axis URL-path Jaccard 0.456; Micro 2f, early divergence 95.24%). Thus "weak" should be read as weak net value on this substrate, not as weak causal force. The screenshot changes decisions; on reddit, those changed decisions often point to content assets rather than evaluator-relevant post or comment routes.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:465:   134	The positive side should also be kept in scope. The macro image axis reduces search and typing and changes first actions substantially, and the symmetric-difference set between P-SoM and SoM contains successful SoM-only tasks (Micro 2f, P-SoM versus SoM N=23, median first divergent step 0). The point is not that screenshots are useless on reddit; it is that the site does not organize navigation through visual product cards or spatial filters. The image channel can disambiguate content, but it is not the dominant route planner.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:469:   138	The compound transition from DOM to Phantom-SoM combines the flat text payload with the marked-element prompt while still avoiding the image channel. On reddit this compound arm is best interpreted as a complementary routing arm rather than as a uniformly stronger baseline. Its adjusted SR is modestly higher than DOM on the full 210-task per-mode summaries (Outcome 0a, P-SoM 13.81% versus DOM 9.52%, +4.29pp), but the more robust mechanism evidence is the divergence in which routes it explores. DOM and P-SoM have low click-target overlap (Micro 2a-extra, compound click-target Jaccard 0.421) and low URL-path overlap (Micro 2a, compound URL-path Jaccard 0.481). The oracle result then follows naturally: P-SoM adds tasks that the original three-mode set misses (Outcome 0c, +P-SoM +2.56pp), and P-text/P-SoM are not redundant with each other (Outcome 0d, Jaccard 0.500).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:473:   142	The compound macro signature is exactly the one expected from this interaction. Relative to DOM, Phantom-SoM types less and uses more tab focus while leaving some coarse frequencies close to the endpoints (Macro 1d, compound type -0.089 and tab_focus +0.071). A router can exploit this complementarity because the behavioral difference appears before final success labels: the symmetric-difference set diverges early between DOM and P-SoM (Micro 2f, DOM<->P-SoM N=23, median first divergent step 0).
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:596:   119	This layered hierarchy resolves the Phantom-SoM hero puzzle without requiring a non-mechanistic explanation. Phantom-SoM uniquely combines three contributions: residual-stream proximity to SoM on the image axis at L17 (the largest single signal, with image-feature reduction "as if image were present"), separation from P-text on the prompt-family axis at L23 (a small 0.011 signal but consistent across cls and reddit), and separation from P-prompt on the text-format axis at L23 (a medium 0.029 signal). P-text occupies only the text-format separation; P-prompt occupies only the prompt-family separation against P-SoM; DOM occupies none of the three. The drop-one hero status of Phantom-SoM in `fig_meta_forest.png` therefore corresponds to the only mode that satisfies all three axis criteria.
docs/checkpoints/codex_outputs/codex_stress_2026-05-12.md:1769:    15	- phantom_text: 12.38%
docs/checkpoints/advisor_sync_5_5_followup.md:195:GitHub 链接我 push 完发您, 您扫 `preregistration.md` + 3 张 forest figure (`fig_meta_forest.png` / `fig_forest_drop_one.png` / `fig_phantom_structure_venn.png`), email 回:
docs/analysis/cross_sites/mechanism_per_task.json:179:        "mean_jaccard": 0.46345,
docs/checkpoints/advisor_sync_5_5_followup.md:195:GitHub 链接我 push 完发您, 您扫 `preregistration.md` + 3 张 forest figure (`fig_meta_forest.png` / `fig_forest_drop_one.png` / `fig_phantom_structure_venn.png`), email 回:
docs/analysis/vwa_reddit/B0_B1_findings.md:33:| SoM | 11.90% | 8.10% | +3.80pp |
docs/analysis/vwa_reddit/B0_B1_findings.md:145:| Oracle ceiling (raw) | **18.57%** | 12.38% |
docs/analysis/vwa_reddit/B0_B1_findings.md:262:6. **B0 oracle ceiling 18.57% > B1 12.38%**：更大模型在 Reddit 上有更大的路由收益空间。
docs/analysis/cross_sites/axis1_microbehavior_report.md:7:| baseline | site | N | URL-path Jaccard | URL divergence | target-hit diff | target N | keyword repeat diff | distinct keyword diff | first-action divergence | macro mean | ratio |
docs/analysis/cross_sites/axis1_microbehavior_report.md:9:| B0 | reddit | 210 | 0.573 | 0.427 | 3.47 pp | 144 | -0.633 | 0.100 | 0.210 | 0.093 | 2.42 |
docs/analysis/cross_sites/axis1_microbehavior_report.md:14:All signed differences are cascade-direction right-minus-left, so axis 1 is P-text minus DOM. Classifieds search-keyword levels should be read by axis differential, because OSClass tasks normally use search pages.
docs/analysis/cross_sites/axis1_microbehavior_report.md:20:| baseline | site | N | URL-path Jaccard (compound) | URL divergence | target-hit diff | first-action divergence |
docs/analysis/cross_sites/axis1_microbehavior_report.md:40:- DOM: steps=30, target_hit=False, reward=1.0, first=type -> /search, keywords=['pumpkin robot', 'pumpkin', 'pumpkin robot', 'pumpkin', 'pumpkin robot', 'pumpkin', 'pumpkin robot', 'pumpkin'], trajectory=/search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+
docs/analysis/cross_sites/axis1_microbehavior_report.md:41:- P-text: steps=10, target_hit=False, reward=1.0, first=type -> /search, keywords=['pumpkin robot', 'pumpkin', 'pumpkin robot', 'pumpkin'], trajectory=/search?q=pumpkin+robot+, /search?q=pumpkin+, /search?q=pumpkin+robot+, /search?q=pumpkin+, /submission_images/1b4655e4790380ba91d7eb446a736383e44dfcb3f41fdc7779c0f06610666ba4.jpg
docs/analysis/cross_sites/axis1_microbehavior_report.md:48:- DOM: steps=30, target_hit=False, reward=1.0, first=type -> /, keywords=['colmscomics', 'colmscomics', 'good lord! this man has handgun ammo in his torso!', 'colmscomics', 'good lord! this man has handgun ammo in his torso!', 'colmscomics', 'good lord! this man has handgun ammo in his torso!', 'colmscomics'], trajectory=/search?q=COLMSCOMICS+, /, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+, /search?q=COLMSCOMICS+, /search?q=Good+Lord%21+This+man+has+Handgun+ammo+in+his+torso%21+
docs/analysis/cross_sites/axis1_microbehavior_report.md:49:- P-text: steps=10, target_hit=False, reward=0.0, first=type -> /search, keywords=['colmscomics', 'colmscomics comic', 'colmscomics', 'colmscomics comic', 'colmscomics', 'colmscomics', 'user:colmscomics'], trajectory=/search?q=COLMSCOMICS+, /search?q=COLMSCOMICS+comic+, /search?q=COLMSCOMICS+, /search?q=COLMSCOMICS+comic+, /search?q=COLMSCOMICS+, /search?q=user%3ACOLMSCOMICS+
docs/analysis/cross_sites/axis1_microbehavior_report.md:56:- DOM: steps=21, target_hit=True, reward=1.0, first=type -> /search, keywords=['wheat field city skyline', 'woman in wheat field city skyline', 'wheat field city skyline', 'woman in wheat field city skyline', 'wheat field city skyline', 'woman in wheat field city skyline', 'wheat field city skyline', 'woman in wheat field city skyline'], trajectory=/search?q=wheat+field+city+skyline+, /search?q=woman+in+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+in+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+in+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+in+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+wheat+field+city+skyline+, /search?q=wheat+field+city+skyline+, /search?q=woman+wheat+field+city+skyline+
docs/analysis/cross_sites/axis1_microbehavior_report.md:57:- P-text: steps=4, target_hit=False, reward=1.0, first=type -> /search, keywords=['wheat field woman'], trajectory=/search?q=wheat+field+woman+
docs/analysis/cross_sites/axis1_microbehavior_report.md:64:- DOM: steps=9, target_hit=False, reward=0.0, first=type -> /index.php, keywords=['motorcycle'], trajectory=/index.php?page=search&sPattern=motorcycle+, /index.php?page=item&id=32241
docs/analysis/cross_sites/axis1_microbehavior_report.md:65:- P-text: steps=10, target_hit=True, reward=0.0, first=select_option -> /index.php, keywords=[], trajectory=/, /index.php?page=search&sCategory=20, /index.php?page=item&id=9068
docs/analysis/cross_sites/axis1_microbehavior_report.md:72:- DOM: steps=30, target_hit=False, reward=1.0, first=type -> /index.php, keywords=['foot brace', 'foot brace', 'foot brace', 'foot brace', 'foot brace', 'foot brace', 'foot brace'], trajectory=/index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190, /index.php?page=search&sPattern=foot+brace+, /index.php?page=item&id=67190
docs/analysis/cross_sites/axis1_microbehavior_report.md:73:- P-text: steps=5, target_hit=True, reward=0.0, first=type -> /index.php, keywords=['brace'], trajectory=/index.php?page=search&sPattern=brace+, /index.php?page=item&id=68133
docs/analysis/cross_sites/probe_audit_verification.json:124:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:364:            "thought": "I have exhausted the current page and still haven't found a guitar with a red velvet lined case. I should try searching for a different keyword or refine my search to find the correct item.",
docs/analysis/cross_sites/probe_audit_verification.json:442:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:760:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:1078:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:1397:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:1716:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:2035:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:2354:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:2651:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:2950:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:3271:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:3591:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:3912:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/probe_audit_verification.json:4233:              "error": "Keyboard.press() got an unexpected keyword argument 'timeout'",
docs/analysis/cross_sites/mechanism_per_task_report.md:11:| reddit | axis_1_text (DOM vs P-text) | 210 | 0.463 | 0.472 | 0.250 | 0.537 | 1.105 | 0.952 | 1.795 |
docs/analysis/cross_sites/mechanism_per_task_report.md:25:- axis_1_text: reddit Jaccard 0.463; classifieds Jaccard 0.561. Lower values indicate that the modes use different URL-changing click decisions.
docs/analysis/cross_sites/axis_effect_size.json:106:            "mean_diff": -11.904762,
docs/analysis/cross_sites/axis_effect_size.json:114:            "mean_diff_pct_pts": -11.904762,
docs/analysis/cross_sites/axis_effect_size.json:669:              -12.380952,
docs/analysis/cross_sites/axis_effect_size.json:916:            "wilcoxon_p": 0.57317,
docs/analysis/cross_sites/axis_effect_size.json:1915:              -11.90456
docs/analysis/cross_sites/axis1_microbehavior.json:2:  "method": "Mode-invariant micro-behavior analysis over reddit and classifieds. Per-task/per-mode metrics extract URL sets, URL path sets, target-page hits from task-config URLs, typed keywords, first action type, first post-action URL path, step count, and finish status. Cascade contrasts are right-minus-left: P-text minus DOM (axis 1 text), P-SoM minus P-text (axis 2 prompt), and SoM minus P-SoM (axis 3 image). URL path Jaccard is symmetric; lower values indicate stronger decision divergence. The cross-site claim ratio compares bounded decision effects (1 - URL-path Jaccard, absolute target-hit diff, first-action divergence) with the mean absolute axis-1 macro effect from axis_effect_size.json.",
docs/analysis/cross_sites/axis1_microbehavior.json:15:          "mean_max_keyword_repeat": 2.761905,
docs/analysis/cross_sites/axis1_microbehavior.json:16:          "mean_distinct_keywords": 1.895238,
docs/analysis/cross_sites/axis1_microbehavior.json:35:          "mean_max_keyword_repeat": 1.071429,
docs/analysis/cross_sites/axis1_microbehavior.json:36:          "mean_distinct_keywords": 1.028571,
docs/analysis/cross_sites/axis1_microbehavior.json:55:          "mean_max_keyword_repeat": 1.290476,
docs/analysis/cross_sites/axis1_microbehavior.json:56:          "mean_distinct_keywords": 0.795238,
docs/analysis/cross_sites/axis1_microbehavior.json:75:          "mean_max_keyword_repeat": 1.87619,
docs/analysis/cross_sites/axis1_microbehavior.json:76:          "mean_distinct_keywords": 1.171429,
docs/analysis/cross_sites/axis1_microbehavior.json:94:          "mean_max_keyword_repeat": 2.128571,
docs/analysis/cross_sites/axis1_microbehavior.json:95:          "mean_distinct_keywords": 1.995238,
docs/analysis/cross_sites/axis1_microbehavior.json:114:          "mean_max_keyword_repeat": 2.033333,
docs/analysis/cross_sites/axis1_microbehavior.json:115:          "mean_distinct_keywords": 1.680952,
docs/analysis/cross_sites/axis1_microbehavior.json:136:          "mean_max_keyword_repeat": 1.92735,
docs/analysis/cross_sites/axis1_microbehavior.json:137:          "mean_distinct_keywords": 1.547009,
docs/analysis/cross_sites/axis1_microbehavior.json:156:          "mean_max_keyword_repeat": 1.598291,
docs/analysis/cross_sites/axis1_microbehavior.json:157:          "mean_distinct_keywords": 1.418803,
docs/analysis/cross_sites/axis1_microbehavior.json:177:          "mean_max_keyword_repeat": 1.290598,
docs/analysis/cross_sites/axis1_microbehavior.json:178:          "mean_distinct_keywords": 1.213675,
docs/analysis/cross_sites/axis1_microbehavior.json:197:          "mean_max_keyword_repeat": 1.807692,
docs/analysis/cross_sites/axis1_microbehavior.json:198:          "mean_distinct_keywords": 1.461538,
docs/analysis/cross_sites/axis1_microbehavior.json:218:          "mean_max_keyword_repeat": 2.004274,
docs/analysis/cross_sites/axis1_microbehavior.json:219:          "mean_distinct_keywords": 1.551282,
docs/analysis/cross_sites/axis1_microbehavior.json:238:          "mean_max_keyword_repeat": 1.4,
docs/analysis/cross_sites/axis1_microbehavior.json:239:          "mean_distinct_keywords": 2.8,
docs/analysis/cross_sites/axis1_microbehavior.json:259:          "mean_max_keyword_repeat": 3.733333,
docs/analysis/cross_sites/axis1_microbehavior.json:260:          "mean_distinct_keywords": 1.604762,
docs/analysis/cross_sites/axis1_microbehavior.json:278:          "mean_max_keyword_repeat": 0.42381,
docs/analysis/cross_sites/axis1_microbehavior.json:279:          "mean_distinct_keywords": 0.180952,
docs/analysis/cross_sites/axis1_microbehavior.json:298:          "mean_max_keyword_repeat": 1.471429,
docs/analysis/cross_sites/axis1_microbehavior.json:299:          "mean_distinct_keywords": 0.77619,
docs/analysis/cross_sites/axis1_microbehavior.json:320:          "mean_max_keyword_repeat": 4.038462,
docs/analysis/cross_sites/axis1_microbehavior.json:321:          "mean_distinct_keywords": 1.435897,
docs/analysis/cross_sites/axis1_microbehavior.json:340:          "mean_max_keyword_repeat": 0.564103,
docs/analysis/cross_sites/axis1_microbehavior.json:341:          "mean_distinct_keywords": 0.470085,
docs/analysis/cross_sites/axis1_microbehavior.json:361:          "mean_max_keyword_repeat": 1.760684,
docs/analysis/cross_sites/axis1_microbehavior.json:362:          "mean_distinct_keywords": 0.952991,
docs/analysis/cross_sites/axis1_microbehavior.json:381:          "mean_max_keyword_repeat": 2.026087,
docs/analysis/cross_sites/axis1_microbehavior.json:382:          "mean_distinct_keywords": 1.152174,
docs/analysis/cross_sites/axis1_microbehavior.json:401:          "mean_max_keyword_repeat": 2.371795,
docs/analysis/cross_sites/axis1_microbehavior.json:402:          "mean_distinct_keywords": 1.226496,
docs/analysis/cross_sites/axis1_microbehavior.json:435:          "max_keyword_repeat_diff": -0.633333,
docs/analysis/cross_sites/axis1_microbehavior.json:436:          "distinct_keywords_diff": 0.1,
docs/analysis/cross_sites/axis1_microbehavior.json:462:          "max_keyword_repeat_diff": -0.252381,
docs/analysis/cross_sites/axis1_microbehavior.json:463:          "distinct_keywords_diff": -0.82381,
docs/analysis/cross_sites/axis1_microbehavior.json:489:          "max_keyword_repeat_diff": -0.585714,
docs/analysis/cross_sites/axis1_microbehavior.json:490:          "distinct_keywords_diff": -0.37619,
docs/analysis/cross_sites/axis1_microbehavior.json:516:          "max_keyword_repeat_diff": -0.885714,
docs/analysis/cross_sites/axis1_microbehavior.json:517:          "distinct_keywords_diff": -0.72381,
docs/analysis/cross_sites/axis1_microbehavior.json:543:          "max_keyword_repeat_diff": -0.728571,
docs/analysis/cross_sites/axis1_microbehavior.json:544:          "distinct_keywords_diff": -0.214286,
docs/analysis/cross_sites/axis1_microbehavior.json:570:          "max_keyword_repeat_diff": -0.157143,
docs/analysis/cross_sites/axis1_microbehavior.json:571:          "distinct_keywords_diff": -0.509524,
docs/analysis/cross_sites/axis1_microbehavior.json:597:          "max_keyword_repeat_diff": -1.471429,
docs/analysis/cross_sites/axis1_microbehavior.json:598:          "distinct_keywords_diff": -1.1,
docs/analysis/cross_sites/axis1_microbehavior.json:626:          "max_keyword_repeat_diff": 0.076923,
docs/analysis/cross_sites/axis1_microbehavior.json:627:          "distinct_keywords_diff": 0.004274,
docs/analysis/cross_sites/axis1_microbehavior.json:653:          "max_keyword_repeat_diff": -0.196581,
docs/analysis/cross_sites/axis1_microbehavior.json:654:          "distinct_keywords_diff": -0.089744,
docs/analysis/cross_sites/axis1_microbehavior.json:680:          "max_keyword_repeat_diff": -0.517094,
docs/analysis/cross_sites/axis1_microbehavior.json:681:          "distinct_keywords_diff": -0.247863,
docs/analysis/cross_sites/axis1_microbehavior.json:707:          "max_keyword_repeat_diff": -0.119658,
docs/analysis/cross_sites/axis1_microbehavior.json:708:          "distinct_keywords_diff": -0.08547,
docs/analysis/cross_sites/axis1_microbehavior.json:734:          "max_keyword_repeat_diff": 0,
docs/analysis/cross_sites/axis1_microbehavior.json:735:          "distinct_keywords_diff": 0.8,
docs/analysis/cross_sites/axis1_microbehavior.json:761:          "max_keyword_repeat_diff": 0.2,
docs/analysis/cross_sites/axis1_microbehavior.json:762:          "distinct_keywords_diff": -0.2,
docs/analysis/cross_sites/axis1_microbehavior.json:788:          "max_keyword_repeat_diff": -0.636752,
docs/analysis/cross_sites/axis1_microbehavior.json:789:          "distinct_keywords_diff": -0.333333,
docs/analysis/cross_sites/axis1_microbehavior.json:861:          "max_keyword_repeat_diff": -2.261905,
docs/analysis/cross_sites/axis1_microbehavior.json:862:          "distinct_keywords_diff": -0.828571,
docs/analysis/cross_sites/axis1_microbehavior.json:904:          "max_keyword_repeat_diff": -0.247826,
docs/analysis/cross_sites/axis1_microbehavior.json:905:          "distinct_keywords_diff": -0.204348,
docs/analysis/cross_sites/axis1_microbehavior.json:931:          "max_keyword_repeat_diff": -2.013043,
docs/analysis/cross_sites/axis1_microbehavior.json:932:          "distinct_keywords_diff": -0.291304,
docs/analysis/cross_sites/axis1_microbehavior.json:958:          "max_keyword_repeat_diff": -1.666667,
docs/analysis/cross_sites/axis1_microbehavior.json:959:          "distinct_keywords_diff": -0.209402,
docs/analysis/cross_sites/axis1_microbehavior.json:985:          "max_keyword_repeat_diff": -0.378261,
docs/analysis/cross_sites/axis1_microbehavior.json:986:          "distinct_keywords_diff": -0.082609,
docs/analysis/cross_sites/axis1_microbehavior.json:1012:          "max_keyword_repeat_diff": -2.277778,
docs/analysis/cross_sites/axis1_microbehavior.json:1013:          "distinct_keywords_diff": -0.482906,
docs/analysis/cross_sites/axis1_microbehavior.json:1040:        "search_keywords": [
docs/analysis/cross_sites/axis1_microbehavior.json:1050:        "max_keyword_repeat": 15,
docs/analysis/cross_sites/axis1_microbehavior.json:1051:        "distinct_keywords": 2,
docs/analysis/cross_sites/axis1_microbehavior.json:1077:        "search_keywords": [
docs/analysis/cross_sites/axis1_microbehavior.json:1083:        "max_keyword_repeat": 2,
docs/analysis/cross_sites/axis1_microbehavior.json:1084:        "distinct_keywords": 2,
docs/analysis/cross_sites/axis1_microbehavior.json:1111:        "search_keywords": [
docs/analysis/cross_sites/axis1_microbehavior.json:1121:        "max_keyword_repeat": 15,
docs/analysis/cross_sites/axis1_microbehavior.json:1122:        "distinct_keywords": 2,
docs/analysis/cross_sites/axis1_microbehavior.json:1150:        "search_keywords": [
docs/analysis/cross_sites/axis1_microbehavior.json:1159:        "max_keyword_repeat": 4,
docs/analysis/cross_sites/axis1_microbehavior.json:1160:        "distinct_keywords": 3,
docs/analysis/cross_sites/axis1_microbehavior.json:1187:        "search_keywords": [
docs/analysis/cross_sites/axis1_microbehavior.json:1197:        "max_keyword_repeat": 7,
docs/analysis/cross_sites/axis1_microbehavior.json:1198:        "distinct_keywords": 4,
docs/analysis/cross_sites/axis1_microbehavior.json:1225:        "search_keywords": [
docs/analysis/cross_sites/axis1_microbehavior.json:1228:        "max_keyword_repeat": 1,
docs/analysis/cross_sites/axis1_microbehavior.json:1229:        "distinct_keywords": 1,
docs/analysis/cross_sites/axis1_microbehavior.json:1251:        "search_keywords": [
docs/analysis/cross_sites/axis1_microbehavior.json:1254:        "max_keyword_repeat": 1,
docs/analysis/cross_sites/axis1_microbehavior.json:1255:        "distinct_keywords": 1,
docs/analysis/cross_sites/axis1_microbehavior.json:1271:        "search_keywords": [],
docs/analysis/cross_sites/axis1_microbehavior.json:1272:        "max_keyword_repeat": 0,
docs/analysis/cross_sites/axis1_microbehavior.json:1273:        "distinct_keywords": 0,
docs/analysis/cross_sites/axis1_microbehavior.json:1298:        "search_keywords": [
docs/analysis/cross_sites/axis1_microbehavior.json:1307:        "max_keyword_repeat": 7,
docs/analysis/cross_sites/axis1_microbehavior.json:1308:        "distinct_keywords": 1,
docs/analysis/cross_sites/axis1_microbehavior.json:1340:        "search_keywords": [
docs/analysis/cross_sites/axis1_microbehavior.json:1343:        "max_keyword_repeat": 1,
docs/analysis/cross_sites/axis1_microbehavior.json:1344:        "distinct_keywords": 1,
docs/analysis/phantom_paper/B0_dom_shopping_diagnostic.md:33:| `least/most expensive/cheapest` keyword | 27 | 0.00% | 10.67 | 10.67 | 0.272 | 3.11 | 40.3k | no progress |
docs/analysis/phantom_paper/som_vs_phantom_som_diagnostic.md:28:| reddit | SoM | 210 | 11.90 / 10.48 | 8.05 / 5 / 30 | 66 (31.4%) | 134 hits in 32 eps (15.2%) | 82.0% | 2.08 / 2 / 8 | $0.0409 | 58.9s |
docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md:5:This audit compares fresh B0 P-text and Phantom-SoM runs on classifieds and reddit. The old "prompt only changes commitment confidence" theory is too narrow. On classifieds, P-text and Phantom-SoM have identical adjusted SR (34/234, 14.53%) but solve substantially different task sets: only 21 adjusted successes overlap, giving Jaccard 0.447. On reddit, the premise that the two adjusted SRs are equal is not supported by episode-level `adjusted_success`: P-text has 25/210 adjusted successes (11.90%), while Phantom-SoM has 29/210 (13.81%), with Jaccard 0.543. The best update is: the prompt family acts as a task-conditional decision prior over search terms, clicks, backtracking, and finish timing; commitment calibration is a real second-order subeffect, but not the only prompt knob.
docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md:7:Important data note: `condition_summary_v2.json` stores raw `success_rate`. For reddit P-text it reports raw 29/210 = 13.81%, but the episode summaries contain 4 raw-success false positives, so adjusted SR is 25/210 = 11.90%. All overlap and oracle-style claims below use episode-level `adjusted_success` unless a row is explicitly marked raw.
docs/analysis/phantom_paper/phantom_dom_vs_som_diagnostic.md:28:| reddit | P-text | 210 | 13.81 / 11.90 | 11.45 / 7 / 30 | 104 (49.5%) | 103 hits in 37 eps (17.6%) | 84.3% | 3.70 / 3 / 21 |
docs/analysis/cross_sites/site_mechanism_dictionary.md:33:| P-text | 12.38% | 1.43% |
docs/analysis/cross_sites/site_mechanism_dictionary.md:38:The substrate claim is that reddit is dominated by text/repr structure and prompt prior, with a weak/balanced image axis. The site has many image-referential tasks, but images are content objects to be identified; they do not organize the navigation surface. AXTree therefore pushes the agent toward keyword search, while `[SOM_MARKS]` plus prompt changes where the agent commits.
docs/analysis/cross_sites/site_mechanism_dictionary.md:52:- B0 adjusted SR moves from DOM 9.52% to P-text 12.38%.
docs/analysis/cross_sites/site_mechanism_dictionary.md:60:- Whole-run search-loop gradient: DOM 51.90% -> P-SoM 35.71% -> SoM 31.43%.
docs/analysis/cross_sites/site_mechanism_dictionary.md:67:| Click-target Jaccard | 0.463450 |
docs/analysis/cross_sites/site_mechanism_dictionary.md:69:| Max keyword repeat diff | -0.633 |
docs/analysis/cross_sites/site_mechanism_dictionary.md:89:> DOM repeats the same keyword 5-15 times; examples include task 23 "pumpkin robot" x15 and task 4 "wheat field city skyline" x7.
docs/analysis/cross_sites/site_mechanism_dictionary.md:212:| DOM vs P-SoM target-hit diff | -0.69pp |
docs/analysis/cross_sites/site_mechanism_dictionary.md:237:Reddit's forum hierarchy makes representation and prompt the leading mechanisms. AXTree depth plus image-referential tasks push DOM into repeated keyword search; flat marks and SoM-style prompt priors open a different, low-cost trajectory basin. The image channel is useful for some content tasks but is not the site navigation substrate, so full SoM is not a monotone improvement over P-SoM.
docs/analysis/cross_sites/site_mechanism_dictionary.md:293:| Max keyword repeat diff | +0.077 |
docs/analysis/cross_sites/site_mechanism_dictionary.md:434:| DOM vs P-SoM target-hit diff | +1.74pp |
docs/analysis/cross_sites/site_mechanism_dictionary.md:669:| Axis 1 text | PRIMARY: URL Jaccard 0.573, click-target Jaccard 0.463, +P-text +3.81pp oracle | Secondary: URL Jaccard 0.904, click-target Jaccard 0.561, +P-text +3.42pp oracle | Mixed/form-action; sparse evidence: click failure 24.7%, swatch bug 11 episodes |
docs/analysis/cross_sites/site_mechanism_dictionary.md:701:Reddit is a forum-hierarchy site where image is content rather than navigation. AXTree depth and image-referential tasks push DOM into repeated keyword search, visible in the 51.90% -> 35.71% -> 31.43% search-loop gradient from DOM to P-SoM to SoM. The prompt axis then acts as a decision prior, reducing search/type behavior and changing trajectories at step 0 in symmetric-diff tasks. The image axis remains bidirectional: it helps some content tasks but also creates visual confidence, mark occlusion, and coordinate/ID failures.

exec
/bin/bash -lc "find docs/checkpoints -maxdepth 4 -type f | grep -E 'behavior|macro|micro|outcome|efficiency|forest|reddit' | sort" in /home/jiaming/workspace/Cost-Aware-Routing-for-Web-Usage-Agents
 succeeded in 0ms:
docs/checkpoints/advisor_sync_5_5_outcomes.md
docs/checkpoints/codex_prompts/axis1_microbehavior.md
docs/checkpoints/codex_prompts/section5_reddit_prose.md
docs/checkpoints/mechanism/results/format_variation_h1_test_reddit.md
docs/checkpoints/stage4_method42_results_reddit.md

codex
Verdict: 当前稿件有一个能站住的 empirical characterization，但 mechanism prose 明显过界；今天提交顶会主会会被当成“interesting artifact + overclaimed causal mechanism”拒掉。

### 强 claims (don't break under attack) / Strong claims

1. **Drop-one oracle 是比单臂胜率更稳的主 claim。**  
   Paper line: `section1_intro.md:7` says reddit Phantom-SoM drop-one is `+3.33pp` with CI `[+0.95,+6.19]`, classifieds `+2.56pp` with CI `[+0.85,+4.70]`. Evidence matches: `hero_claim_bootstrap_ci.md:31-42`, `:67-78`.  
   这个 claim 不怕 reviewer 攻击 “P-SoM vs SoM single-mode CI crosses zero”，因为你已经把 single-mode dominance 降级为 parity/complementarity。

2. **Cross-site asymmetry is real.**  
   `section4_empirical_findings.md:40-42` correctly states classifieds favors full SoM while reddit does not. Bootstrap evidence supports this: classifieds P-SoM vs SoM is strictly negative `[-12.39,-1.28]` (`hero_claim_bootstrap_ci.md:56-59`), while reddit P-SoM vs SoM crosses zero (`:20-23`).  
   这不是 universal win claim，而是 substrate-dependent representation effect；这个 framing 可以保留。

3. **Axis-2 cosine signal is broad, if framed only as geometry.**  
   `section5_mechanism.md:117` says P-text↔P-SoM L23 gap is not outlier-driven. Evidence supports that: classifieds median `0.0131`, reddit median `0.0120`, 100% tasks above `0.010` (`axis2_per_task_fragility.md:8-24`, `:66-79`).  
   Strong version: “prompt-family is linearly readable at L23 across curated B1 tasks.”  
   Do not upgrade this alone to causal planning.

### 弱 claims (would tank under attack) / Weak claims

1. **Claim: “Phantom-SoM uniquely combines three contributions... residual-stream proximity to SoM on the image axis”**  
   Exact line: `section5_mechanism.md:119`.  
   Attack: 这个 claim 是内部矛盾。Earlier the paper says P-SoM is far from SoM on image axis (`section5_mechanism.md:30-32`), and the table gives P-SoM↔SoM gap `0.0412`, the largest axis (`section5_mechanism.md:105-113`). Calling this “proximity to SoM” is wrong. A reviewer will write: “The authors cannot consistently state whether Phantom-SoM is image-like or separated from image modes.”  
   Defuse: rewrite as “P-SoM has a large image-axis separation from SoM, but the no-image flat marks condition delays/reshapes image-axis divergence.” Remove “as if image were present” unless you define it operationally.  
   Effort: 1 hour prose fix.

2. **Claim: “shortcut trigger is flat element-list format rather than a specific token pattern”**  
   Exact lines: `section5_mechanism.md:11`, `section5_mechanism.md:75-83`.  
   Attack: 这个 claim 已经被今天的 W6 部分推翻或至少严重细化。Reddit format test is mixed: `appagent_id`, `plain_numbered`, and `hash_id_control` peak at L04 (`format_variation_h1_test_reddit.md:14-17`), while only 4/6 marks-like variants peak L17 (`:18-23`, `:47-53`). W6 then says the trigger is likely “integer marker + markup-sigil delimiter,” not abstract flat list (`w6_h1_red_l04_attribution.md:50-56`). Six variants is tiny and post-hoc.  
   Defuse: run held-out falsifiers: `bare_N` integer without sigil, `bracket_no_int` sigil without integer, `id_N_bracketed`, and one shuffled flat-list control on both cls/reddit. Report before seeing results.  
   Effort: 0.5-1 day if pipeline is ready.

3. **Claim: “Prompt-only captures ~77-100% of combined effect; prompt-family is dominant causal driver”**  
   Exact lines: `exp5_axis2_causal_patching.md:28-34`.  
   Attack: 这是最危险的新 overclaim。You are subtracting patching effects from different source conditions and calling the residual “image axis.” That is not a valid causal decomposition unless additivity is tested directly. There is also no bootstrap CI (`exp5_axis2_causal_patching.md:97-101` admits this), no task-randomized source control, and no proof that the `phantom_som` archive truly had no image beyond manual intent (`:99-100`).  
   Defuse: add per-task bootstrap CIs, source-task-shuffled control, same-prompt random-source control, and a real 2x2 patching decomposition: source modes `{P-text, P-SoM, SoM}` into target P-text with interaction term.  
   Effort: 1-2 days.

4. **Claim: “output logprobs are the cheapest deployment-time mode-axis discriminator”**  
   Exact lines: `section5_mechanism.md:123-125`.  
   Attack: 这个 claim weak。The evidence is a logit lens on per-layer mean hidden states (`axis2_logit_lens.md:1-6`), not a deployed classifier, not API-visible for B0, and not benchmarked against random directions or simple surface-token classifiers. “Deployment-time” is unsupported.  
   Defuse: either delete deployment language or train/evaluate a held-out classifier using available features, with AUROC/CI and a random-direction baseline.  
   Effort: prose delete 30 min; real classifier 1 day.

5. **Claim: “L17 hidden state is the causally active planning site”**  
   Exact line: `plan.md:125-135`; paper softer version at `section5_mechanism.md:44-71`.  
   Attack: 计划文档还在写 L17 singular planning site, but new evidence says cosine peak L23 and prompt-family causal peak L11/L17, with L23 patching weak (`exp5_axis2_causal_patching.md:52-65`). Also Method 4.4 refuted L17 sweet spot: full sweep L17 α=5 H-mean `0.16`, best is L33 α=10 `0.33` with poor selectivity (`plan.md:205-223`; `section5_mechanism.md:34-42`).  
   Defuse: replace “planning site” with “patch-sensitive continuation window L11-L17 under final-token replacement patching.” Keep steering as failed/partial validation.  
   Effort: 1-2 hours.

6. **Claim/data inconsistency: reddit P-text adjusted SR.**  
   Exact lines: table says `11.90` (`section4_empirical_findings.md:37`), later prose says `12.38` (`section4_empirical_findings.md:106`), bootstrap evidence says `12.38` (`hero_claim_bootstrap_ci.md:15`).  
   Attack: Reviewer will treat this as evidence the analysis table is not locked. It directly touches the two-knob story.  
   Defuse: recompute one canonical adjusted-SR table from episode-level adjusted success and update every occurrence.  
   Effort: 1 hour.

### 诚实缺口 (not weak, just missing) / Honest gaps

- `section1_intro.md:13` promises Sections 6 and 7, but no `section6` or `section7` draft files are present. That is a manuscript completeness gap.
- No cross-family mechanism evidence. `section8_limitations.md:3` admits Qwen-family only; top-tier reviewers will still ask whether this is Qwen tokenizer/pretraining artifact.
- No behavioral causal bridge from patching to task success. Patching changes 50-token continuations; it does not show SR/drop-one changes.
- Negative controls are too narrow: Gaussian random injection is useful but not enough. Need content-matched wrong-task/source-mode controls.
- Null/missing cells are not cleanly narrated. “45 completed cells” appears (`section5_mechanism.md:38`), while plan says “FULL 45/48” (`plan.md:205`). Say exactly which 3 did not finish and why they do not matter.

### Distance to top-tier

- **Current tier**: strong workshop / borderline mid-tier conference. Not NeurIPS/ICML/ACL main in current form.
- **Specific blockers**:
  1. Mechanism overclaim: blocks §5 causal contribution.
  2. W6/H1 contradiction: blocks “flat-list shortcut trigger.”
  3. No cross-family or non-Qwen replication: blocks broad scientific significance.
  4. Data inconsistency in P-text SR: blocks trust in empirical tables.
- **Unblock plan**:
  1. Rewrite §5 to separate geometry, patching, steering, and speculation. 2-3 hours.
  2. Run W6 held-out format falsifiers on cls+reddit. 0.5-1 day.
  3. Add task-shuffled/content-matched patching controls and bootstrap CIs for Exp5. 1-2 days.
  4. Add one non-Qwen or at least Qwen2-VL replication for geometry-only claims. 2-5 days if infra works; weeks if not.
- **Submission-today probability**: NeurIPS/ICML/ACL main `0.10-0.20`. I would reject around reviewer score `4/10`: interesting empirical artifact, but causal mechanism and novelty claims are not yet controlled tightly enough.

### One thing to fix tonight (1-3 hour leverage move)

Rewrite `docs/checkpoints/paper_drafts/section5_mechanism.md` to remove the three worst overclaims: `flat element-list trigger` (`:11`, `:75-83`), `proximity to SoM / hero resolved` (`:119`), and `deployment-time logprob discriminator` (`:123-125`). Add a small “Evidence status” table: geometry = strong, patching = causal continuation evidence, steering = weak/partial, trigger attribution = exploratory after W6. This immediately prevents reviewer-3 from rejecting for causal overclaim before they even reach the good empirical result.
tokens used
94,501
Verdict: 当前稿件有一个能站住的 empirical characterization，但 mechanism prose 明显过界；今天提交顶会主会会被当成“interesting artifact + overclaimed causal mechanism”拒掉。

### 强 claims (don't break under attack) / Strong claims

1. **Drop-one oracle 是比单臂胜率更稳的主 claim。**  
   Paper line: `section1_intro.md:7` says reddit Phantom-SoM drop-one is `+3.33pp` with CI `[+0.95,+6.19]`, classifieds `+2.56pp` with CI `[+0.85,+4.70]`. Evidence matches: `hero_claim_bootstrap_ci.md:31-42`, `:67-78`.  
   这个 claim 不怕 reviewer 攻击 “P-SoM vs SoM single-mode CI crosses zero”，因为你已经把 single-mode dominance 降级为 parity/complementarity。

2. **Cross-site asymmetry is real.**  
   `section4_empirical_findings.md:40-42` correctly states classifieds favors full SoM while reddit does not. Bootstrap evidence supports this: classifieds P-SoM vs SoM is strictly negative `[-12.39,-1.28]` (`hero_claim_bootstrap_ci.md:56-59`), while reddit P-SoM vs SoM crosses zero (`:20-23`).  
   这不是 universal win claim，而是 substrate-dependent representation effect；这个 framing 可以保留。

3. **Axis-2 cosine signal is broad, if framed only as geometry.**  
   `section5_mechanism.md:117` says P-text↔P-SoM L23 gap is not outlier-driven. Evidence supports that: classifieds median `0.0131`, reddit median `0.0120`, 100% tasks above `0.010` (`axis2_per_task_fragility.md:8-24`, `:66-79`).  
   Strong version: “prompt-family is linearly readable at L23 across curated B1 tasks.”  
   Do not upgrade this alone to causal planning.

### 弱 claims (would tank under attack) / Weak claims

1. **Claim: “Phantom-SoM uniquely combines three contributions... residual-stream proximity to SoM on the image axis”**  
   Exact line: `section5_mechanism.md:119`.  
   Attack: 这个 claim 是内部矛盾。Earlier the paper says P-SoM is far from SoM on image axis (`section5_mechanism.md:30-32`), and the table gives P-SoM↔SoM gap `0.0412`, the largest axis (`section5_mechanism.md:105-113`). Calling this “proximity to SoM” is wrong. A reviewer will write: “The authors cannot consistently state whether Phantom-SoM is image-like or separated from image modes.”  
   Defuse: rewrite as “P-SoM has a large image-axis separation from SoM, but the no-image flat marks condition delays/reshapes image-axis divergence.” Remove “as if image were present” unless you define it operationally.  
   Effort: 1 hour prose fix.

2. **Claim: “shortcut trigger is flat element-list format rather than a specific token pattern”**  
   Exact lines: `section5_mechanism.md:11`, `section5_mechanism.md:75-83`.  
   Attack: 这个 claim 已经被今天的 W6 部分推翻或至少严重细化。Reddit format test is mixed: `appagent_id`, `plain_numbered`, and `hash_id_control` peak at L04 (`format_variation_h1_test_reddit.md:14-17`), while only 4/6 marks-like variants peak L17 (`:18-23`, `:47-53`). W6 then says the trigger is likely “integer marker + markup-sigil delimiter,” not abstract flat list (`w6_h1_red_l04_attribution.md:50-56`). Six variants is tiny and post-hoc.  
   Defuse: run held-out falsifiers: `bare_N` integer without sigil, `bracket_no_int` sigil without integer, `id_N_bracketed`, and one shuffled flat-list control on both cls/reddit. Report before seeing results.  
   Effort: 0.5-1 day if pipeline is ready.

3. **Claim: “Prompt-only captures ~77-100% of combined effect; prompt-family is dominant causal driver”**  
   Exact lines: `exp5_axis2_causal_patching.md:28-34`.  
   Attack: 这是最危险的新 overclaim。You are subtracting patching effects from different source conditions and calling the residual “image axis.” That is not a valid causal decomposition unless additivity is tested directly. There is also no bootstrap CI (`exp5_axis2_causal_patching.md:97-101` admits this), no task-randomized source control, and no proof that the `phantom_som` archive truly had no image beyond manual intent (`:99-100`).  
   Defuse: add per-task bootstrap CIs, source-task-shuffled control, same-prompt random-source control, and a real 2x2 patching decomposition: source modes `{P-text, P-SoM, SoM}` into target P-text with interaction term.  
   Effort: 1-2 days.

4. **Claim: “output logprobs are the cheapest deployment-time mode-axis discriminator”**  
   Exact lines: `section5_mechanism.md:123-125`.  
   Attack: 这个 claim weak。The evidence is a logit lens on per-layer mean hidden states (`axis2_logit_lens.md:1-6`), not a deployed classifier, not API-visible for B0, and not benchmarked against random directions or simple surface-token classifiers. “Deployment-time” is unsupported.  
   Defuse: either delete deployment language or train/evaluate a held-out classifier using available features, with AUROC/CI and a random-direction baseline.  
   Effort: prose delete 30 min; real classifier 1 day.

5. **Claim: “L17 hidden state is the causally active planning site”**  
   Exact line: `plan.md:125-135`; paper softer version at `section5_mechanism.md:44-71`.  
   Attack: 计划文档还在写 L17 singular planning site, but new evidence says cosine peak L23 and prompt-family causal peak L11/L17, with L23 patching weak (`exp5_axis2_causal_patching.md:52-65`). Also Method 4.4 refuted L17 sweet spot: full sweep L17 α=5 H-mean `0.16`, best is L33 α=10 `0.33` with poor selectivity (`plan.md:205-223`; `section5_mechanism.md:34-42`).  
   Defuse: replace “planning site” with “patch-sensitive continuation window L11-L17 under final-token replacement patching.” Keep steering as failed/partial validation.  
   Effort: 1-2 hours.

6. **Claim/data inconsistency: reddit P-text adjusted SR.**  
   Exact lines: table says `11.90` (`section4_empirical_findings.md:37`), later prose says `12.38` (`section4_empirical_findings.md:106`), bootstrap evidence says `12.38` (`hero_claim_bootstrap_ci.md:15`).  
   Attack: Reviewer will treat this as evidence the analysis table is not locked. It directly touches the two-knob story.  
   Defuse: recompute one canonical adjusted-SR table from episode-level adjusted success and update every occurrence.  
   Effort: 1 hour.

### 诚实缺口 (not weak, just missing) / Honest gaps

- `section1_intro.md:13` promises Sections 6 and 7, but no `section6` or `section7` draft files are present. That is a manuscript completeness gap.
- No cross-family mechanism evidence. `section8_limitations.md:3` admits Qwen-family only; top-tier reviewers will still ask whether this is Qwen tokenizer/pretraining artifact.
- No behavioral causal bridge from patching to task success. Patching changes 50-token continuations; it does not show SR/drop-one changes.
- Negative controls are too narrow: Gaussian random injection is useful but not enough. Need content-matched wrong-task/source-mode controls.
- Null/missing cells are not cleanly narrated. “45 completed cells” appears (`section5_mechanism.md:38`), while plan says “FULL 45/48” (`plan.md:205`). Say exactly which 3 did not finish and why they do not matter.

### Distance to top-tier

- **Current tier**: strong workshop / borderline mid-tier conference. Not NeurIPS/ICML/ACL main in current form.
- **Specific blockers**:
  1. Mechanism overclaim: blocks §5 causal contribution.
  2. W6/H1 contradiction: blocks “flat-list shortcut trigger.”
  3. No cross-family or non-Qwen replication: blocks broad scientific significance.
  4. Data inconsistency in P-text SR: blocks trust in empirical tables.
- **Unblock plan**:
  1. Rewrite §5 to separate geometry, patching, steering, and speculation. 2-3 hours.
  2. Run W6 held-out format falsifiers on cls+reddit. 0.5-1 day.
  3. Add task-shuffled/content-matched patching controls and bootstrap CIs for Exp5. 1-2 days.
  4. Add one non-Qwen or at least Qwen2-VL replication for geometry-only claims. 2-5 days if infra works; weeks if not.
- **Submission-today probability**: NeurIPS/ICML/ACL main `0.10-0.20`. I would reject around reviewer score `4/10`: interesting empirical artifact, but causal mechanism and novelty claims are not yet controlled tightly enough.

### One thing to fix tonight (1-3 hour leverage move)

Rewrite `docs/checkpoints/paper_drafts/section5_mechanism.md` to remove the three worst overclaims: `flat element-list trigger` (`:11`, `:75-83`), `proximity to SoM / hero resolved` (`:119`), and `deployment-time logprob discriminator` (`:123-125`). Add a small “Evidence status” table: geometry = strong, patching = causal continuation evidence, steering = weak/partial, trigger attribution = exploratory after W6. This immediately prevents reviewer-3 from rejecting for causal overclaim before they even reach the good empirical result.
