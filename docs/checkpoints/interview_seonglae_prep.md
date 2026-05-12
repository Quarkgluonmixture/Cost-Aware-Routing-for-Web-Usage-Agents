# Interview Prep — Seonglae Cho @ Holistic AI
**Date**: Friday 2026-05-15 (3 days out from this prep dated 2026-05-12)
**Position**: AI Engineer / Researcher internship (LLM interpretability focus)
**Prep doc lifecycle**: read before interview, archive after; update if new findings land Wed-Fri

---

## 1. Context summary (60-second opener)

Seonglae Cho is 3rd author on **"Tool Calling is Linearly Readable and Steerable in Language Models"** (UCL + Holistic AI + Imperial, author list: Zekun Wu, Ze Wang, **Seonglae Cho**, Yufei Yang, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz). This paper is one of the **6/6 Q5 lit anchors** for our paper §2 (Zoom 4 mechanism layer). The first author Zekun Wu is already an advisor sync contact (paper §7 ADVISOR_SYNC, v3 message paste-ready since 2026-05-12).

**This is not a cold interview**. We are already inside the academic loop of this Holistic AI group via Zekun. Seonglae will likely have heard our name; framing should be peer-level academic conversation rather than candidate-evaluator dynamic.

Seonglae's public LinkedIn stack:
- UCL 2024-2025 Distinction (same university as us)
- AI Engineer / Researcher @ Holistic AI
- LLM interpretability, **sparse autoencoder**, **LLM steering**, correctness representation
- AgentGraph (multi-agent), LibVulnWatch (LLM safety adjacent)
- Prior: LLaMA2 GPTQ quantization, Node.js + ONNX + vector DB, Kubernetes/GCP

---

## 2. Common ground (use these as conversational entry points)

### 2.1 Tool Calling Linear Circuit paper specifics

Key facts from the paper (`docs/literature/Tool Calling is Linearly Readable and Steerable in Language Models.md`):
- Method: PCA over 15 tool means → ~10 directions; cosine readout 69-82% on BFCL; mean-difference steering 77-100% switch accuracy across 12 instruction-tuned models (Gemma 3 / Qwen 3 / Qwen 2.5 / Llama 3.1, 270M-27B)
- Result: tool identity is **linearly readable and steerable** along the row of the output layer that produces the target tool's first token
- Localization: small set of **mid and late-layer attention heads**
- Striking detail: **base models encode the right tool before they can emit it** (cosine readout 69-82%, base generation 2-10%) — "knows but can't say"

### 2.2 UCL connection

Both you and Seonglae are UCL graduates. He just finished his MSc with Distinction in 2024-25. Brief acknowledgement of shared institutional context is natural opener — don't over-emphasize but worth noting.

### 2.3 Lab member relationship

Zekun Wu (1st author of Tool Calling paper) is **already an advisor sync contact** in paper §7. Your v3 message to Zekun is paste-ready. Seonglae may know Zekun is reviewing your paper §5 mechanism work; this is **already-collaborating, not cold-outreach**.

---

## 3. My paper §5 mechanism work — talking points

Use this if Seonglae asks "tell me about your research."

### 3.1 60-second pitch

"I'm working on a paper that uses phantom routing space — a 2x2 (text-format × prompt-family) ablation grid of Qwen3-VL-4B observation modes on VisualWebArena — to identify the mid-layer mechanism that explains why a deployment-class shortcut (skipping the annotated image) preserves SoM-like signal at DOM-like cost. Yesterday and today I closed cross-site replication and three-axis layered decomposition. The framework method-wise is descended from your Tool Calling work — same cosine gap + mean-diff steering toolkit — but the multimodal web-agent setting reveals a layered three-axis hierarchy that single-axis tool-calling doesn't expose."

### 3.2 Today's freshly-landed evidence (2026-05-12)

| Finding | Source | Magnitude |
|---|---|---|
| P5b reddit Mirage signature replication | `stage4_multimode_b1_reddit/method42_metrics.json` | P-SoM↔DOM L17 = 0.0098 (text-axis sibling), P-SoM↔SoM L17 = 0.0423 (image-axis split), AUROC=1.0 |
| P5a reddit format H1 cross-site | `format_variation_h1_test_reddit.md` | 4/6 marks-like at L17, hash_id_control at L04 (proper control), dom at L04 |
| P4 cls reverse-tier H1 (selection-bias defense) | `format_variation_h1_test_cls_reverse.md` | same pattern as strong-tier, H1 is not tier-selection artifact |
| Stage 3 6/6 reddit + cls 2x2 mechanism additivity | `stage3_cellh{d,t,p}_{cls,red}_*` | Δoverlap-to-tgt -0.19 to -0.35 mid-layer L11-L18 |
| Exp 1 three-axis residual stream hierarchy | `axis2_layer_profile.md` | image L17 = 0.041, text-format L23 = 0.029, prompt-family L23 = 0.011 (4:3:1 ratio) |
| Exp 3 logit lens output amplification | `axis2_logit_lens.md` | P-text↔P-SoM peak KL L23 = 0.162; lm_head amplifies cosine 10-25x; KL@L36 collapses (decoding window L23-L25) |

### 3.3 The three-axis hierarchy framework (paper §5.7)

This is the single strongest thing to lead with. Quote-able sentence:

> "We measure a layered three-axis mechanism in Qwen3-VL-4B residual stream: image-feature (L17, magnitude 0.041), text-format (L23, magnitude 0.029), and prompt-family (L23, magnitude 0.011). The 4:3:1 magnitude ratio is preserved both cross-site (classifieds and reddit replicate) and cross-layer (lm_head logit-lens decoding amplifies cosine 10-25x while preserving rank). Phantom-SoM uniquely earns drop-one hero status because it is the only observation mode that occupies all three axes simultaneously."

### 3.4 Method overlap with Wu et al. tool calling (1:1 mapping)

| Wu et al. Tool Calling method | Our paper §5 method | Difference |
|---|---|---|
| Cosine readout on per-tool means (69-82% accuracy, 15 tools) | Cosine gap on per-mode means (AUROC 1.0, 6 modes, 4 axis-isolated pairs) | Multimodal observation modes vs single-modal tool identity |
| PCA over 15 tool means → ~10 directions | PCA top-10 variance per (mode, layer); axis-isolated pair contrasts | Axis decomposition extension |
| Mean-difference steering 77-100% switch | Mean-difference steering H-mean 0.33 ceiling (45-cell sweep) | Multi-step JSON continuation harder than single-token tool name |
| "Knows but can't say" (cosine 69-82% vs base gen 2-10%) | "Weak residual signal amplified into strong output signal at L23-L25" (cosine 0.011 → KL 0.16, 14x via lm_head) | Direction of dissociation reversed but structurally homologous |
| Localized to mid + late-layer attention heads | Layer-resolved cosine + logit lens converge on L23-L25 decoding window | Compatible localization |

---

## 4. Anticipated questions from him + your answers

### Q1: "Why Qwen3-VL specifically?"
- Public availability + manageable size (4B parameters, fits single GPU bf16 ~10GB VRAM)
- VisualWebArena task benchmark is the substrate, Qwen3-VL-4B is one of few open multimodal models capable enough to run the agent loop end-to-end
- Today: closed cls + reddit cross-site. Tomorrow: P2 Phi-3.5-Vision-4.2B (cross-family) + P3 Qwen2-VL-7B (within-family capacity) HF downloads in progress

### Q2: "Method 4.4 0.33 H-mean ceiling — why?"
- L17 α=5 smoke = 0.44 (4-cell sample), full 45-cell sweep at L17 α=5 = 0.16 (smoke variance artifact, see `mechanism_notes §126/§127`)
- Real sweet spot L33 α=10 H-mean 0.33: completeness 38% (largest output shift), but selectivity 29% (over-steers JSON envelope)
- Mid-layer (L11-L23) preserves JSON 100% but completeness only 0-11%; late-layer (L33) flips the trade-off
- 0.33 is the **fixed mean-difference ceiling**, motivates LA-HDMI (Khorasani 2026 per-input gradient) or SAE feature steering in paper §8

### Q3: "What's novel beyond Wu et al. tool calling?"
- Three-axis hierarchy decomposition (not single-axis tool identity)
- Cross-site replication (cls + reddit, same magnitude ratio)
- Output amplification via logit lens (axis-2 cosine 0.011 → KL 0.16 at L23, 14x amplification, axis-agnostic in rank preservation)
- KL@L36 ≈ 0 paradox revealing L23-L25 decoding window (mode-distinct signal not in final embedding)
- Stage 3 6-cell mechanism additivity table: SoM source displaces DOM, P-text, P-prompt targets at -0.19 to -0.35 mid-layer disruption

### Q4: "Why is this paper-grade?"
- 4 vertical defenses populated: per-task fragility (P1) + selection-bias (P4 today) + cross-site H1 (P5a today) + cross-site Mirage (P5b today)
- 5/5 robustness tests passed for Method 4.2 (label perm 9.8σ, per-task 24/24 positive, per-step invariant, silhouette ≥0.5, bootstrap CI 4-15% of mean)
- 45-cell Method 4.4 full sweep (not smoke)
- 16-cell Stage 2/3 patching battery + negative controls (random injection -0.03 cls, ~0 reddit)

### Q5: "What's next?"
- Exp 5 (Myriad, qsub'd today): causal axis-2 mechanism via prompt-only patching cellhprompt_cls + cellhprompt_red — tests whether L23 axis-2 cosine peak is **causally** used
- Paper §8 / paper-2 future: **SAE feature steering on Qwen3-VL-4B**. This is where your Holistic AI SAE expertise becomes directly relevant.

---

## 5. Questions to ask him (research conversation, not interview-style)

Pick 2-3 max — leave room for natural flow.

### Q-Tier 1 (high-info, low-risk)
1. "In the Tool Calling paper you found mid + late-layer attention heads carry the circuit. Did you probe whether the magnitude was layer-monotone or had distinct peaks? Our 4:3:1 axis ratio shows that residual-stream geometry strength varies smoothly but lm_head decoding amplification is axis-agnostic. Curious if you saw analogous structure on tool calling."
2. "Your LinkedIn lists sparse autoencoder work. Are there Qwen-family SAEs publicly available that I should know about, or did Holistic AI train SAEs in-house? Paper §8 of our work is the LA-HDMI vs SAE feature-steering branch point and we'd want pretrained SAE if it exists."
3. "AgentGraph on your stack — does it overlap with the routing layer of paper-2 that we deferred from paper-1? Paper-1 §1 is cost-aware web-agent routing in phantom routing space; paper-2 will implement the actual learned router. Curious where AgentGraph sits."

### Q-Tier 2 (Holistic AI / role-fit, ask if natural)
4. "How does the interpretability work at Holistic AI translate into product or research output? Conferences / open-source / customer-facing?"
5. "What's the team structure for interpretability — solo researcher, or collaborative with the safety/correctness representation people?"

### Q-Tier 3 (DON'T ASK unless he raises)
- Salary / hours (let HR handle)
- Why Zekun is at Holistic AI vs full-time UCL (lab dynamics are off-limits)
- Their LibVulnWatch product details (probably proprietary)

---

## 6. Strategic asks (if interview goes well)

If the conversation reaches "tell me what you'd want from this internship":

- **Mentorship on SAE feature steering on Qwen3-VL-4B** — paper §8 explicit future work, would benefit enormously from Seonglae's SAE expertise
- **Co-author / advisor on paper §8 or paper-2** — if the SAE direction matures into an independent publication path
- **Compute / pretrained model access** — Holistic AI may have Qwen-family SAEs, evaluation infrastructure, or compute that DGX Spark single-GPU can't easily replicate
- **Connection to Zekun + Sahan + Maria** — already inside the lab loop, formalizing is mutually useful

---

## 7. Risk areas + handling

### Risk 1: Over-claiming methodological novelty
- DO NOT say "we improved on Wu et al." or "we extended your work" — frame as **method overlap, finding extension**
- Safer wording: "your linear-readable + steerable framework was the substrate; our finding is the three-axis hierarchy that wasn't visible in single-axis tool-calling setting"

### Risk 2: Single-model caveat
- Qwen3-VL-4B is the only model with complete evidence stack today; P2/P3 cross-family is deferred
- If he asks "have you tested other models?": "Phi-3.5-Vision-4.2B and Qwen2-VL-7B downloads are in progress; deferred because HF cas-bridge throttling on thread_map. Will run single-thread CLI tomorrow. Paper §5 cross-site (cls+reddit) is already replicated, cross-family is the 5th vertical defense, not paper-critical."

### Risk 3: Method 4.4 0.33 ceiling defensiveness
- Some interviewers will probe "why is steering only 0.33?" — own this as **evidence ceiling, paper-honest**, not a weakness
- Frame: "0.33 is what fixed mean-difference can do with selectivity preserved; LA-HDMI per-input gradient (Khorasani) or SAE features (your expertise) are the natural next-step ceiling-breakers"

### Risk 4: Paper-2 routing ambition
- DON'T pitch paper-2 routing implementation as a guaranteed deliverable — it's deferred for a reason (mechanism saturation on paper-1 first)
- If asked about routing application: "paper §6 routing is deferred to a follow-up; paper-1 establishes mechanism + cross-site evidence layer, paper-2 will use the L23-L25 decoding window features for cost-aware routing"

---

## 8. Tone and energy

- **Peer-level academic conversation**, not job interview
- Lead with intellectual curiosity about his SAE work, not your need for the role
- Show calibrated confidence: today's results are paper-grade, you know they are, but you also know the cross-family + LA-HDMI gaps
- Friday timing is **perfect** — yesterday's evidence is fresh, prose is in draft, you have something concrete to discuss not just plans

---

## 9. Reference materials to skim Wed-Thu

| File | Purpose |
|---|---|
| `docs/literature/Tool Calling is Linearly Readable and Steerable in Language Models.md` | Wu et al. paper — re-read Section 4 (linear readability) and Section 5 (attention head localization) |
| `docs/checkpoints/paper_drafts/section5_mechanism.md` | Your current paper §5 v1 — own all numbers |
| `docs/checkpoints/mechanism/results/axis2_layer_profile.md` | Exp 1 result table |
| `docs/checkpoints/mechanism/results/axis2_logit_lens.md` | Exp 3 result table |
| `docs/checkpoints/mechanism/plan.md` §1-§7 | Theory framework, methods, evidence dashboard — for "tell me about your research" 5-minute version |

---

## 10. Last 10 minutes before interview

- Open `axis2_layer_profile.md` + `axis2_logit_lens.md` + `paper_drafts/section5_mechanism.md` in tabs
- Pull up `results/phantom_paper/figures/fig_axis2_prompt_layer_profile.png` if screensharing is on the cards
- Have today's date (2026-05-15) and the four numbers memorized: **L17 image 0.041, L23 text 0.029, L23 prompt 0.011, ratio 4:3:1**
- Breathe. The work is real, evidence is fresh, framework is paper-grade. This is a peer conversation.
