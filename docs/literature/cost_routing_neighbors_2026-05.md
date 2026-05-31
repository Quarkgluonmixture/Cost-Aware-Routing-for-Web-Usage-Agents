# Paper Reading Notes — Four cost/efficiency-routing neighbors (May 2026)

**Source batch:** user lit-list 2026-05-31. Existence cross-verified via arXiv API (`id_list` + `ti:` title search) after Anthropic WebSearch false-negatived two of them (both published this month → consumer-index lag). Full chronicle: 实验笔记 §310. Verification lesson: memory `feedback_arxiv_api_for_verification`.
**Papers:** Judge Circuits (2605.16023) · PANDO (2605.24785) · ReVision (2605.11212) · DMR (2602.02711).
**Disposition:** all four added to `paper.bib` (keys `feldhus2026judgecircuits` / `li2026pando` / `abaskohi2026revision` / `li2026dmr`). Relevance = §6 related-work positioning + (Judge Circuits) §1 motivation + (DMR) learned-router method anchor. **None is a direct baseline.**

---

## 0. Why these four together — the routing-axis spectrum (slots into planning §103)

P79 already frames an efficiency-axis taxonomy (planning §103: model routing / early-exit halting / **representation routing ← P79's third axis**). These four sharpen it: three route *compute*, P79 alone routes *representation*, and Judge Circuits supplies the mechanistic reason representation/format is load-bearing.

| Paper | Routes WHAT (per step/task) | Axis | Bench / base |
|---|---|---|---|
| DMR | precision (bf16 vs W4/W3 quant) | compute | ALFWorld + WebShop / Qwen3-1.7B/4B |
| PANDO | model capacity (planner vs actor) | compute | VWA / Opus 4.6 + GPT-5.2 |
| ReVision | visual-token budget (drop redundant patches) | compute (input size) | OSWorld/WebTailBench/AgentNetBench / Qwen2.5-VL-7B |
| **P79** | **observation representation (DOM/SoM/phantom)** | **representation** | **VWA cls+red / Qwen3-VL 4B&235B + Gemma-3-4B** |
| Judge Circuits | — (mechanism, not routing) | — | text judges / Gemma-3, Qwen2.5, Llama-3 |

**One-sentence §6 synthesis:** *Prior efficiency routing acts on compute — precision [DMR], model capacity [PANDO], visual-token budget [ReVision]. We route the observation representation, the unfilled axis, which Judge Circuits shows carries separable, format-conditional computation.*

---

## 1. Judge Circuits (2605.16023) — mechanism SUPPORT, text-only

**Method.** PEAP (Position-aware Edge Attribution Patching) causal circuit analysis on **Gemma-3 / Qwen2.5 / Llama-3** text judges. Finds a shared sparse **Latent Evaluator** = rating-circuit ∩ classification-circuit, in mid-to-late MLPs; a continuous judgment signal computed there is mapped through **fragile, format-specific terminal branches** to a discrete score. Circuit is tiny (top-200 edges; k=5 on Gemma-3-27B/RewardBench).

**The finding that matters most for P79 — modularity is family×scale dependent.** Modular (judgment separable from world-knowledge): Qwen2.5-7B/14B, Llama-3.1-8B, Gemma-3-27B. **Entangled: Gemma-3-12B** (zero-ablation halves MMLU clinical 81→19%, physics 48→23%). FTI format-injection flips Qwen2.5-7B 100% (CoLA/STS-B) but 0.7% on Gemma-3-27B/MNLI. "Scale alone does not predict modularity."

**Mapping to P79.** ✅ §1/§6 motivation: format-conditional computation is mechanistically real and separable from content, on the *same backbones* we use (Qwen3-VL, Gemma-3-VL). The line "benchmark-level reliability comparisons across formats are partially measuring **formatter geometry** rather than evaluation quality" directly motivates the phantom format×prompt-style axis.

**Caveats (write both into prose).**
1. **Domain gap** — text-only judges, no multimodal/agent; "format" = judge *output* format, not our *input observation* format. Conceptual transfer only; do **not** cite as web-agent evidence.
2. **B2 risk** — our B2 = `google/gemma-3-4b-it` is *smaller* than the already-entangled Gemma-3-12B → the format/content separability the phantom axis assumes may be **weaker for B2** than for Qwen. A predicted cross-family confound to watch when B2 phantom results come in — not a settled result.
3. Maps to the **shelved §5 mechanism** scope; related-work/motivation only, not a mechanism claim of our own.

---

## 2. PANDO (2605.24785) — closest VWA-efficiency neighbor; differentiate hard

**Method.** Single-rollout online skill distillation on the full **910-task VWA**. Skill Library = rules (pattern guardrails) + routines (parameterized program-as-action like `apply_price_filter(min,max)`), retrieved by **deterministic keyword containment** (cacheable, not embedding). Plus progress reflection, confidence-based skill demotion (Beta), hierarchical routing, visual compression, cache-aware prompting.

**Numbers.** 58.3% SR vs SGV 54.0% (Self-Grounded Verification, Gemini-Flash 2-pass) vs WALT 45.2% reproduction (Web-Agents-that-Learn-Tools, offline tool discovery); 58%/61% fewer tokens. **Base = Claude Opus 4.6 (planner) + GPT-5.2 (grounding).** Ablation: skills/rules/routines 38.6→57.3% SR (+18.7pp); routing+compression+cache only +1.7pp SR but tokens 147K→117K.

**Mapping to P79.** Closest neighbor on "efficiency in VWA," but three precise differentiators:
- **Routing object ≠ ours.** PANDO "hierarchical routing" = model-capacity (planner vs actor); skill *selection* = keyword match. Neither is representation routing. **P79 routes the observation representation — the open axis.**
- **SR not comparable.** Frontier closed stack (Opus 4.6 + GPT-5.2) vs our 4B open models (B1/B2). Never put 58.3% next to our SR.
- **Cost lever is small** in their ablation (+1.7pp / ~20% tokens); skills do the SR work. So PANDO is a *narrative ally* for "routing → lower marginal cost," not evidence for our (definitional) cost≈DOM.

**Borrowable (more than a citation).** 3 trajectory-level efficiency metrics: **Action Repetition Rate** (PANDO 9.1 / SGV 14.2 / WALT 18.3%; aligns with our runner cycle detection), **Step Overhead Ratio** (1.8/2.3/2.6×), **Prompt Cache Utilization** (72.4/45.1/38.6%). Also: PANDO criticizes WALT for hiding pre-eval discovery cost → corroborates our total-billed canonical estimand ([[project_cost_latency_canonical_estimand]]).

---

## 3. ReVision (2605.11212) — visual-redundancy contrast; needs fine-tuning

**Method.** RTS (ReVision Token Selection) = 3-layer MLP over consecutive-frame patch-pair embeddings → binary keep-mask (labels via OmniParserV2 segmentation + IoU match). The MLLM is **fine-tuned** on filtered trajectories (first frame kept whole; later frames keep only non-redundant patches).

**Numbers (Qwen2.5-VL-7B, 5 history screenshots).** OSWorld 54%↓ tokens / +1.5pp (32.3→34.0); AgentNetBench 40%↓ / +1.3pp (72.5→73.8); WebTailBench 37%↓ / +3.4pp (36.0→40.2); avg ~46% / +3pp. History: no-drop saturates ~7 images then declines; ReVision improves to ~11; saturation ≈ total context (~23k tokens), not image count → "saturation reflects inefficient token representation, not limited usefulness of history."

**Mapping to P79.** ✅ §6 related-work for the visual-redundancy/efficiency framing. **Caveat:** (a) requires **fine-tuning the backbone** — not inference-time drop-in, unlike our training-free phantom modes; (b) it prunes **temporal** redundancy at **patch** level within the kept image stream; phantom drops the **entire** annotated image (**cross-modal** redundancy given the flattened [SOM_MARKS] text). Contrast point, not method transfer.

---

## 4. DMR (2602.02711) — per-step routing precedent + learned-router method anchor

**Method.** Per-step routing between **bf16 (high)** and **W4A16/W3A16-quantized (low)** variants of the *same* model; base = **Qwen3-1.7B / Qwen3-4B** (B1's backbone family). Router = lightweight Transformer encoder (~2–3% of routed-LLM params, 8 layers, trajectory input, softmax over precision). Two-stage: (1) **KL-ST** — label a step precision-sensitive if step-wise KL between low/high action distributions ≥ τ, class-weighted CE; (2) **GRPO** — reward = success − λ·cost.

**Numbers.** ALFWorld Qwen3-4B: bf16 94.3% / quant 85.4% / DMR@K=0.40 **95.1%** (beats bf16, faster). WebShop Qwen3-1.7B: bf16 59.0% / quant 40.9% / DMR@K=0.20 53.4% (1.17× speedup; recovers 93% of the quant gap on GPTQ).

**Mapping to P79.** ✅ §6 per-step routing precedent **+ method anchor** for the learned router ([[project_paper_hook]] §6 router contribution). Two borrowable ideas: (a) routing signal = **action-distribution KL between a cheap and an expensive variant** (vs our current per-task LR classifier) — a concrete signal-design idea; (b) **KL-supervised → GRPO-refine** as an upgrade recipe for the noted per-step phantom-routing extension. **Caveat:** routes *precision* not *representation*; router overhead 17–29 ms/call; WebShop forces high-KL search steps to bf16.

---

## 5. Quick disposition table

| Paper | bib key | Cite in | What to borrow | What NOT to do |
|---|---|---|---|---|
| Judge Circuits | `feldhus2026judgecircuits` | §1/§6 motivation | "formatter geometry" framing; B2 confound prediction | don't cite as web-agent evidence; don't claim mechanism |
| PANDO | `li2026pando` | §6 neighbor | 3 efficiency metrics; WALT total-cost critique | don't compare 58.3% to our SR; don't conflate its routing with ours |
| ReVision | `abaskohi2026revision` | §6 related-work | "visual saturation = token inefficiency" framing | don't imply we reuse RTS (it needs fine-tuning) |
| DMR | `li2026dmr` | §6 precedent + method | KL routing signal; KL→GRPO recipe | don't call it representation routing |

**Score-1 (parked):** Plan-Then-Execute (2605.14290, security framing, reviewer-defense) · Expert Strikes Back (2604.02178, ICML'26, MoE interp → paper-2) · SteerMoE (2509.09660, ICLR'26, MoE steering → paper-2).
