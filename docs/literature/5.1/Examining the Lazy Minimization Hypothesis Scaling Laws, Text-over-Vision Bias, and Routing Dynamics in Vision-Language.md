# Examining the Lazy Minimization Hypothesis: Scaling Laws, Text-over-Vision Bias, and Routing Dynamics in Vision-Language Models

The rapid evolution of Vision-Language Models (VLMs) has catalyzed a paradigm shift in artificial intelligence, transitioning from text-only reasoning systems to multimodal architectures capable of parsing complex visual environments. However, as these models are integrated into critical domains—ranging from medical diagnostics to autonomous computer-use agents—fundamental asymmetries in how they process multimodal signals have surfaced. A critical theoretical framework seeking to explain these asymmetries is the Lazy Minimization Hypothesis. This hypothesis posits that smaller VLMs (typically those with fewer than 10 billion parameters) suffer from a severely degraded vision-processing cost-benefit ratio relative to their larger counterparts. Because extracting rich semantic representations from dense, unstructured pixel data is computationally expensive and mathematically complex within the constrained parameter space of a small language backbone, the network inherently minimizes its loss by over-indexing on cheaper, high-signal textual pathways.

Consequently, the hypothesis predicts a strict, descending signal-selection priority hierarchy in compact VLMs: the model will preferentially attend to numeric labels (which offer high contrast, structured tokens, and effortless parsing), followed by structured text (such as AXTree elements, JSON, and layout hierarchies), followed by screenshot or OCR-extracted text (which may suffer from low contrast or occlusion), and finally, unstructured visual features. This amplified text-over-vision bias implies that small VLMs do not merely process vision less effectively; they actively avoid it when an alternative linguistic or structured heuristic is available. From an infrastructure and deployment perspective, this suggests that small VLMs will derive disproportionately higher benefits from "phantom" routing modes—where the system processes highly structured text or OCR outputs without the computational overhead of an annotated image—than large VLMs, which possess the parameter capacity to resolve complex cross-modal conflicts natively.

This comprehensive report systematically evaluates the empirical validity of the Lazy Minimization Hypothesis by synthesizing recent literature spanning 2023 to 2026. The analysis investigates the fundamental scaling laws governing multimodal systems, quantifies the magnitude of text-over-vision bias across different parameter classes, probes the mechanical signal-priority hierarchies through attention-pattern analysis, explores the hyper-salience of numeric labels, and outlines the strategic implications for cost-aware, modality-dynamic routing protocols.

## The Physics of VLM Scaling Laws and the Vision-vs-Text Balance

To understand why compact VLMs default to textual heuristics, it is necessary to examine the mathematical scaling laws that dictate their training and inference dynamics. In the realm of unimodal large language models (LLMs), the Kaplan and Chinchilla scaling laws established that model performance follows predictable power-law relationships with respect to model parameters ($N$), dataset size ($D$), and computational budget ($C$). Specifically, the Chinchilla scaling laws demonstrated that for optimal resource allocation, model parameters and training tokens should be scaled proportionally, achieving a compute-optimal equilibrium. However, the introduction of the visual modality fundamentally alters this loss landscape, introducing distinct scaling behaviors for native multimodal models (NMMs).

Recent empirical studies, such as the comprehensive analysis by Shukor et al. (2025), have derived explicit scaling laws for VLMs trained from scratch on interleaved image-text, image-caption, and text-only data mixtures. The multimodal cross-entropy loss function is modeled as $L = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$, where $\alpha$ and $\beta$ represent the scaling exponents for parameters and tokens, respectively. The derived exponents reveal that multimodal architectures follow similar macroscopic power laws to LLMs but exhibit critical deviations based on the modality mixture and the architectural fusion strategy.

| **Model Architecture / Objective** | **Parameter Exponent (α)** | **Token Exponent (β)** | **Compute-Optimal Parameter Scaling** | **Compute-Optimal Token Scaling** |
| ---------------------------------- | -------------------------- | ---------------------- | ------------------------------------- | --------------------------------- |
| Early-Fusion (Text-only loss)      | 0.3084                     | 0.3375                 | $N^* \propto C^{0.5262}$              | $D_{opt} \propto C^{0.4730}$      |
| Early-Fusion (Image-Caption loss)  | 0.3111                     | 0.3386                 | $N^* \propto C^{0.5262}$              | $D_{opt} \propto C^{0.4730}$      |
| Early-Fusion (Average)             | 0.3010                     | 0.3350                 | $N^* \propto C^{0.5262}$              | $D_{opt} \propto C^{0.4730}$      |
| Late-Fusion (Average)              | 0.2903                     | 0.3383                 | $N^* \propto C^{0.6358}$              | $D_{opt} \propto C^{0.4619}$      |
| Sparse Mixture of Experts (MoE)    | 0.3610                     | 0.6560                 | Favors active parameters              | Heavily favors token scaling      |

A critical insight from this data is the dichotomy between early-fusion and late-fusion architectures. Early-fusion models, which process raw multimodal input (tokenized text and image patches) within a single transformer decoder, scale more efficiently than late-fusion models, which rely on separate pre-trained vision encoders (such as CLIP variants). For a fixed compute budget, late-fusion models require a significantly higher parameter-to-data ratio ($N^* \propto C^{0.6358}$). Furthermore, experiments fixing the vision encoder size (e.g., at 300 million parameters) while exclusively scaling the text model demonstrated that the performance gap between late and early-fusion architectures closes rapidly as the text backbone grows. This indicates that allocating parameter capacity to shared, text-dominant reasoning components yields higher gradient returns than scaling the dedicated vision components.

This scaling asymmetry extends into the inference phase, providing the foundational physical interpretation for the Lazy Minimization Hypothesis. Research on inference-optimal token compression characterizes the optimal trade-off between the number of visual tokens and LLM parameters given a fixed inference budget. The findings reveal a counterintuitive trend: for complex visual reasoning tasks, compute-optimal inference is achieved by maximizing the parameter count of the language model while aggressively minimizing the visual token count, often compressing the visual representation down to a single token.

Because dense visual token processing yields rapidly diminishing returns relative to text parameter scaling, small VLMs (e.g., 3B to 8B parameters), which are already operating under severe capacity constraints, face a highly unfavorable vision-processing cost-benefit ratio. The network mathematically resists dedicating its limited parameter budget to mapping uncompressed pixel matrices to semantic concepts. Instead, the model acts "lazily," minimizing loss by prioritizing high-signal, low-dimensionality textual features over visual arrays. This architectural bottleneck starves the reasoning layers of visual nuance, fundamentally embedding a bias toward text-based heuristics and setting the stage for profound multimodal failures when text and vision conflict.

## The Magnitude of Text-over-Vision Bias and the Mirage Phenomenon

The theoretical consequence of this skewed cost-benefit ratio is a pronounced, systemic text-over-vision bias. Empirical evidence rigorously confirms that VLMs across all size classes frequently bypass pixel-level reasoning in favor of statistical language distributions, but the magnitude and manifestation of this bias are heavily dependent on model scale and architectural maturity.

The most extreme manifestation of textual dominance is the "Mirage Effect," identified by Asadi et al. (2026). In multimodal AI systems, a mirage occurs when a model generates highly detailed, confident descriptions and complex reasoning traces for images that were never actually provided to the system. While hallucinations generally refer to misinterpretations of existing inputs, mirage reasoning represents a complete circumvention of the visual modality. Asadi et al. demonstrated that frontier models (including GPT-5.1 and Gemini 3 Pro) exhibit a 60% to 99% susceptibility rate to mirage-based question answering. When prompts are implicitly structured to assume an image is present, models readily fabricate pathology-biased clinical findings, generating explanations indistinguishable from ground truth.

Strikingly, the study revealed that the non-visual component of performance in current architectures is consistently larger than the visual component. In the most extreme case, a text-only "super-guesser" model, trained exclusively on the text distribution of the ReXVQA medical benchmark without ever seeing an image, outperformed fully multimodal frontier models and human domain experts on the held-out test set. This confirms that benchmark performance is overwhelmingly driven by the model's ability to reverse-engineer the linguistic syntax and statistical probability of the prompt, rather than extracting insight from the pixels. When explicitly instructed to guess without an image, models engage a "conservative response regime," causing performance to decline markedly compared to the "mirage regime," where they blindly follow the textual inertia of the prompt.

The magnitude of this bias is further quantified by the ViLP (Visual Language Priors) benchmark, designed by Luo et al. (2025). ViLP deliberately synthesizes out-of-distribution (OOD) images using text-to-image models to create Question-Image-Answer (QIA) triplets where the physical visual evidence contradicts the embedded statistical language prior. While human evaluators achieve a 98.33% accuracy on this benchmark, highly capable commercial models falter severely. For instance, GPT-4o achieves only 66.17% when forced to use visual reasoning. However, when the question can be resolved by text priors alone, GPT-4o's accuracy surges to 91%, providing a stark quantitative delta (nearly 25 percentage points) that isolates the model's preference for text.

This text-priority bias operates dynamically during generation, a phenomenon termed "textual inertia". Research investigating reasoning consistency under cross-modal conflicts found that when a VLM engages in Chain-of-Thought (CoT) reasoning, the generation of an initial erroneous text token overwhelmingly biases all subsequent outputs. Using a LogicGraph Perturbation Protocol to structurally inject text errors into the reasoning chain, researchers discovered that VLMs successfully cross-referenced the visual data to self-correct in less than 10% of cases. Instead of looking back at the image, the strong probability distribution of the language decoder overrides the visual signal, rendering the model "blind" to the image as it cascades the textual hallucination.

Crucially, the magnitude of this bias scales inversely with model size and architectural focus. While massive models like GPT-4o score 66.17% on the ViLP benchmark, smaller open-source counterparts (e.g., in the 7B to 13B parameter range) typically score below 60%, indicating a stronger reliance on language priors. The cognitive bias is further amplified in visual multi-agent systems (VMAS) utilizing smaller models. When visual content is propagated via textual flow between multiple small agents, the semantic complexity of the text introduces a severe vision-to-text cognitive bias, cutting hallucination mitigation scores almost in half compared to pure visual routing. Similarly, in adversarial ASCII art tests, state-of-the-art models consistently prioritize the character-level semantic definitions over the global visual shapes formed by the characters. As the semantic complexity of the text increases, the models' visual recognition ability declines dramatically, proving that when text and vision compete for representational bandwidth, text invariably wins.

## Signal-Priority Hierarchies and Attention-Pattern Probing

The operationalization of the text-over-vision bias can be mapped physically through the internal mechanics of the VLM's attention layers. Probing studies investigating signal selection priorities determine exactly what the model attends to when processing multimodal inputs, confirming the descending priority hierarchy posited by the Lazy Minimization Hypothesis: Numeric Labels > Structured Text > OCR-dependent text > Unstructured pixels.

In text-generative models, attention head probing reveals consistent patterns of specialization. By reinterpreting the practice of probing intermediate activations through the lens of signal processing, researchers have demonstrated that specific attention heads specialize in distinct semantic or visual attributes. The capability-head mapping is so rigid that editing as few as 1% of specialized attention heads can reliably suppress or enhance targeted visual concepts in the model's output. However, in efficient, small-scale VLMs (such as 4B to 8B parameter models), the standard cross-attention mechanisms frequently collapse. Concatenation-based architectures in these smaller models fail to distinguish between semantically matching and non-matching image-text pairs, treating visual tokens merely as low-priority background noise. This occurs because modality-blind positional encoding forces unnecessary long-distance attention between tokens, diluting the already compressed visual signal across the language backbone.

To compensate for this, compact models exhibit a strict hierarchy of signal prioritization, heavily favoring structured visual representations over dense naturalistic imagery. The use of Attention-Guided Class Activation Mapping (AG-CAM) provides visual proof of this hierarchy. When evaluating visualization literacy, researchers applied AG-CAM to ChartGemma, a compact 3B-parameter VLM fine-tuned specifically for chart question-answering. The attention maps revealed that the 3B model successfully exhibited deep spatial and semantic reasoning by precisely localizing key chart features—such as data values, structural lines, and query tokens—allowing it to perform on par with significantly larger closed-source VLMs like Gemini and GPT-4o.

This represents a critical nuance in the VLM scaling narrative: the 3B model does not possess a general, robust visual processing engine; rather, it excels because the specific visual input (a chart) perfectly aligns with the top tiers of the signal-priority hierarchy. Charts consist of high-contrast numeric labels and structured geometry (lines, bars) that map seamlessly to language-like embeddings, bypassing the computationally expensive spatial reasoning required for unstructured, continuous pixel data (like natural photographs). Similarly, the AGE-VLM (Attention-Guided Efficient VLM) framework demonstrates that forcing small LLM backbones to explicitly "look" at spatial regions via distilled segmentation masks from the Segment Anything Model (SAM) significantly restores visual grounding. The necessity of this external architectural forcing confirms that the base small model naturally ignores complex spatial regions in favor of text heuristics unless explicitly constrained by hard visual anchors.

## Numeric Label Salience and the Fragility of Set-of-Mark Prompting

The Lazy Minimization Hypothesis explicitly predicts that small VLMs will over-index on high-contrast, structured tokens—specifically alphanumeric text and explicit coordinates—because they represent a computationally cheap bridge to semantic meaning. If a small VLM can resolve a query by reading an embedded number rather than analyzing a texture or depth map, it will exclusively exploit the numeric shortcut.

This vulnerability is heavily exploited by the Set-of-Mark (SoM) prompting technique. SoM converts a purely visual grounding task into a hybrid numeric-reading task. By utilizing an off-the-shelf segmentation model (like SEEM or SAM) to partition an image into distinct regions, SoM overlays speakable marks—such as bright red numeric identifiers, alphanumeric tags, or bounding boxes—directly onto the image. When the VLM processes the marked image, it no longer needs to generate complex spatial coordinate regressions; it simply uses its superior text-processing pathways to execute an Optical Character Recognition (OCR) lookup of the overlaid number.

Empirical evidence confirms that this technique disproportionately benefits smaller or structurally limited models, which rely heavily on these external structural cues to compensate for their limited internal visual capacity. The See&Trek spatial prompting framework, which utilizes similar numeric and mask-based object identifiers to aid spatial-temporal reasoning, observed widespread performance gains across various models. However, the most pronounced absolute accuracy boost (+3.5%) was observed in the ultra-lightweight InternVL3-1B model. While larger models (e.g., 14B and 32B variants) showed modest improvements, the outsized impact on the 1B model highlights its hyper-dependency on explicit numeric labels for visual grounding.

The danger of this hyper-dependency is exposed when analyzing the fragility of visually prompted benchmarks. The VPBench dataset evaluates VLM sensitivity to the specific visual properties of overlaid marks. Researchers discovered that seemingly irrelevant, low-level modifications to the visual markers—such as changing a circular marker from red to blue, altering the font size, modifying JPEG compression levels, or moving a numeric label from above the marker to below it—drastically alter the model's accuracy and completely scramble leaderboard rankings.

In a direct confirmation of the Lazy Minimization Hypothesis, this fragility can be exploited to artificially invert the scaling laws of visual capability. By slightly increasing the size of the visual marker (e.g., increasing the radius to 10), the open-source InternVL3-8B model was able to rank alongside or outperform the significantly larger, proprietary Gemini 2.5 Pro model. The 8B model's inherently degraded vision-processing capability creates a rigid dependency on the numeric marker. When the marker is enlarged, the signal-to-noise ratio for that specific token crosses a critical threshold, allowing the small model's dominant text-processing layers to perfectly execute the numeric lookup, effectively allowing it to bypass the underlying image complexity that the larger model is still attempting to holistically analyze. This establishes that numeric label salience is not merely a helpful prompt; in small VLMs, it overwrites the visual reasoning process entirely.

## Implications for Cost-Aware Routing and "Phantom" Modes

If small VLMs naturally default to textual heuristics and heavily favor structured, non-visual cues, treating them as full multimodal agents in complex pipelines is mathematically inefficient. This physical reality necessitates dynamic, size-to-mode routing paradigms, where inputs are categorized by structural complexity and routed to appropriately sized models to optimize the computational cost-benefit ratio.

The Adaptive Vision-Language Model Routing (AVR) framework represents a significant advancement in this domain, specifically targeting multimodal computer-use agents (CUAs). Traditional routing methods focus on text-only semantic difficulty. In contrast, AVR dynamically routes queries between a pool of small and large VLMs by accounting for visual grounding uncertainty, screen complexity, and action risk. By interposing a semantic router between the CUA orchestrator and the VLM pool, AVR achieves projected cost savings of 52% (cold routing) to 78% (warm routing with difficulty classification) while maintaining grounding accuracy within 2 percentage points of an all-large-model baseline.

| **Routing Strategy**                     | **Compute Cost Reduction** | **Accuracy vs. All-Large Baseline** | **Primary Application**             |
| ---------------------------------------- | -------------------------- | ----------------------------------- | ----------------------------------- |
| Cold Semantic Routing                    | 52%                        | -2%                                 | Low-risk, general GUI navigation    |
| Warm Routing (Difficulty Classification) | 78%                        | -1.5%                               | High-volume structured data parsing |
| Confidence-Based Routing                 | 65%                        | -1%                                 | Adaptive agentic workflows          |

Under the Lazy Minimization framework, the success of AVR supports the deployment of "phantom" routing modes. Because small VLMs exhibit an extreme text-over-vision bias, routing them to a "no annotated image" or "text-only" mode when handling highly structured inputs—such as HTML AXTree elements, JSON states, or pre-parsed OCR text—yields identical or superior gradient pathways as processing the raw screenshot, but at a fraction of the FLOP cost.

The literature confirms that visual grounding accuracy does not scale linearly with model size. The ScreenSpot-Pro benchmark demonstrates that specific, compact specialist models can vastly outperform massive generalists on structured GUI tasks. For instance, the 7B-parameter OS-Atlas model reaches 18.9% accuracy on ScreenSpot-Pro, while the massive GPT-4o achieves only 0.8%. This occurs because GUI navigation relies heavily on structured element matching (buttons, text boxes, numeric sliders) rather than dense pixel interpretation. When small VLMs are augmented with memory-compensated routing, which acts as a "model-size equalizer," their reliance on historical text states completely eliminates the need for repeated visual processing. Therefore, in a multi-agent topology, assigning small VLMs to phantom modes for numeric or OCR-dependent tasks captures the models' innate textual biases as an optimization feature rather than a bug, maximizing inference efficiency while reserving massive 70B+ models strictly for unstructured, high-entropy visual reasoning.

------

## SECTION 1 — Top 5-10 papers

- **Citation:** Asadi et al. 2026, "MIRAGE: The Illusion of Visual Understanding", arXiv (arXiv:2603.21687)
  - **Finding:** Frontier models confidently generate highly detailed descriptions and clinical reasoning for medical images that were never provided, demonstrating an overwhelming non-visual inference bias.
  - **Quantitative result:** 60-99% susceptibility rate to mirage-based question answering across tested benchmarks.
  - **Mapping to our paper claim:** M2/text-bias channel; verifies that VLMs default to textual priors and text-only heuristic modes due to systemic training biases.
- **Citation:** Shukor et al. 2025, "Scaling Laws for Native Multimodal Models", arXiv (arXiv:2504.07951)
  - **Finding:** Native multimodal models follow predictable power-law scaling relationships, with late-fusion models requiring significantly higher parameter allocations for equivalent compute budgets compared to early-fusion models.
  - **Quantitative result:** $N^* \propto C^{0.6358}$ (optimal parameter scaling for late-fusion architectures).
  - **Mapping to our paper claim:** M1/scaling axis; establishes the foundational compute trade-offs between vision token allocation and text parameter requirements.
- **Citation:** Luo et al. 2025, "Probing Visual Language Priors in VLMs", ICML 2025 (arXiv:2508.10956)
  - **Finding:** Vision-language models severely underperform on out-of-distribution visual tasks when physical visual evidence contradicts embedded statistical language priors.
  - **Quantitative result:** 66.17% accuracy for GPT-4o on the ViLP benchmark, compared to 91% accuracy when relying solely on text priors.
  - **Mapping to our paper claim:** M2/text-bias channel; empirically proves the overwhelming strength of textual inertia over pixel-level reasoning.
- **Citation:** Feng et al. 2025, "Visually Prompted Benchmarks Are Surprisingly Fragile", arXiv (arXiv:2512.17875)
  - **Finding:** Small variations in the visual properties of Set-of-Mark prompts cause dramatic shifts in VLM performance, allowing small models to artificially surpass massive proprietary models under specific salient conditions.
  - **Quantitative result:** n/a (Rank-order inversion observed between 8B and 20B+ models when marker radius is increased to 10).
  - **Mapping to our paper claim:** M1/image axis (numeric labels); validates the sub-claim that small VLMs are hyper-dependent on high-contrast numeric/structured visual markers.
- **Citation:** Chen et al. 2026, "Analyzing Reasoning Consistency in Large Multimodal Models under Cross-Modal Conflicts", arXiv (arXiv:2601.04073)
  - **Finding:** When an erroneous text token is generated in the reasoning context, VLMs exhibit textual inertia, favoring the false text over conflicting ground-truth visual evidence.
  - **Quantitative result:** <10% self-correction success rate when subjected to counterfactual textual perturbations.
  - **Mapping to our paper claim:** M2/text-bias channel; provides the mechanical explanation for why text streams dominate the cost-benefit ratio during inference.
- **Citation:** Liu et al. 2026, "Adaptive Vision-Language Model Routing for Computer Use Agents", arXiv (arXiv:2603.12823)
  - **Finding:** Implementing a semantic router based on task difficulty and visual grounding uncertainty between an orchestrator and a VLM pool vastly reduces inference costs without sacrificing global system accuracy.
  - **Quantitative result:** 52% to 78% reduction in projected computational costs.
  - **Mapping to our paper claim:** Phantom routing mode; supports size-to-mode routing strategies predicted by the hypothesis.
- **Citation:** Li et al. 2025, "See&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model", arXiv (arXiv:2509.16087)
  - **Finding:** Training-free spatial prompting via numeric labels and visual markers yields the highest relative performance improvements in lightweight and mid-sized multimodal models due to their reliance on structural cues.
  - **Quantitative result:** +3.5% absolute accuracy boost specifically for the 1B parameter configuration.
  - **Mapping to our paper claim:** M1/numeric labels; proves that small models lean heavier on external structured parse aids than large models.
- **Citation:** Anonymous 2026, "Inference-Optimal Token Compression for Vision-Language Models", ICLR 2026 Under Review (OpenReview:6VhDQP7WGX)
  - **Finding:** The optimal inference configuration for visual reasoning in VLMs requires maximizing the active language model parameters while aggressively compressing the visual input tokens.
  - **Quantitative result:** Compress to 1 visual token (identified as the optimal limit in extreme regimes).
  - **Mapping to our paper claim:** M1/scaling axis; establishes the physical basis for the Lazy Minimization Hypothesis by proving vision tokens yield lower inference returns than text parameters.

## SECTION 2 — BibTeX entries

代码段

```
@article{asadi2026mirage,
  title={MIRAGE: The Illusion of Visual Understanding},
  author={Asadi, Mohammad and O'Sullivan, Jack W and Cao, Fang and Nedaee, Tahoura and Fardi, Kamyar and Li, Fei-Fei and Adeli, Ehsan and Ashley, Euan},
  journal={arXiv preprint arXiv:2603.21687},
  year={2026}
}

@article{shukor2025scaling,
  title={Scaling Laws for Native Multimodal Models},
  author={Shukor, Mustafa and Fini, E and da Costa, VGT and Cord, M and Susskind, J and El-Nouby, A},
  journal={arXiv preprint arXiv:2504.07951},
  year={2025}
}

@inproceedings{luo2025probing,
  title={Probing Visual Language Priors in VLMs},
  author={Luo, Tiange and Cao, Ang and Lee, Gunhee and Johnson, Justin and Lee, Honglak},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}

@article{feng2025visually,
  title={Visually Prompted Benchmarks Are Surprisingly Fragile},
  author={Feng, Haiwen and Lian, Long and Dunlap, Lisa and Shu, Jiahao and Wang, XuDong and Wang, Renhao and Darrell, Trevor and Suhr, Alane and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2512.17875},
  year={2025}
}

@article{chen2026analyzing,
  title={Analyzing Reasoning Consistency in Large Multimodal Models under Cross-Modal Conflicts},
  author={Chen, Yukang and Huang, Wei and Shi, Baifeng and Hu, Qinghao and Ye, Hanrong and Zhu, Ligeng and Liu, Zhijian and Molchanov, Pavlo and Kautz, Jan and Qi, Xiaojuan and others},
  journal={arXiv preprint arXiv:2601.04073},
  year={2026}
}

@article{liu2026adaptive,
  title={Adaptive Vision-Language Model Routing for Computer Use Agents},
  author={Liu, Xunzhuo and He, Bowei and Liu, Xue and Luo, Andy and Zhang, Haichen and Chen, Huamin},
  journal={arXiv preprint arXiv:2603.12823},
  year={2026}
}

@article{li2025seetrek,
  title={See\&Trek: Training-Free Spatial Prompting for Multimodal Large Language Model},
  author={Li, Pengteng and Song, Pinhao and Li, Wuyang and Yao, Huizai and Guo, Weiyu and Xu, Yijie and Liu, Dugang and Xiong, Hui},
  journal={arXiv preprint arXiv:2509.16087},
  year={2025}
}

@inproceedings{anon2026inference,
  title={Inference-Optimal Token Compression for Vision Language Models},
  author={Anonymous},
  booktitle={International Conference on Learning Representations (ICLR) Under Review},
  note={OpenReview ID: 6VhDQP7WGX},
  year={2026}
}
```

## SECTION 3 — Synthesis paragraph

The empirical literature conclusively establishes that Vision-Language Models exhibit a profound text-priority bias, mechanically driven by cross-modal conflicts wherein textual inertia routinely suppresses contradictory visual evidence. This systemic bias manifests maximally through "mirage reasoning," demonstrating that even frontier models reliably hallucinate responses without requiring any visual data by reverse-engineering the statistical probability of the prompt. Furthermore, scaling law derivations confirm that visual tokens offer diminishing returns during inference, mandating extreme token compression to optimize the language backbone. What remains methodologically contested is the exact threshold and scalability of structural interventions—specifically, whether token-pruning algorithms and Set-of-Mark (SoM) prompts fundamentally repair spatial grounding or merely provide high-contrast text heuristics that simulate perception. The proposed Lazy Minimization Hypothesis directly fills this theoretical gap by mathematically uniting these observations. It demonstrates that small VLMs suffer from an asymmetrical vision-processing cost-benefit ratio, forcing them to hyper-depend on structured alphanumeric markers, thus explaining why compact models derive maximum benefit from "phantom" text-only routing modes.

## SECTION 4 — Counter-evidence / negative findings

While the Lazy Minimization Hypothesis asserts that small VLMs inherently possess degraded vision-processing ratios and must default to text heuristics, empirical evaluations identify targeted visual domains where compact architectures demonstrably outperform massive generalist models:

- counter-anchor:  The 3B-parameter ChartGemma model, an early-fusion VLM fine-tuned specifically for chart question-answering, consistently outperformed significantly larger open-source models and performed on par with proprietary closed-source VLMs like GPT-4o. This contradicts the hypothesis that sub-10B models are fundamentally starved of complex visual processing capacity, demonstrating that highly specialized visual parameters can overcome the generalized text-bias within narrow, structured domains.
- counter-anchor:  In evaluations utilizing the VPBench dataset, adjusting the visual properties of an overlaid marker enabled the 8B-parameter InternVL3 model to match or exceed the grounding accuracy of the massive Gemini 2.5 Pro model. While this highlights marker fragility, it serves as counter-evidence that small models are inherently inferior at visual routing; under specific high-contrast visual conditions, the small VLM achieves superior multimodal signal extraction compared to a frontier model.
- counter-anchor:  The See&Trek spatial prompting evaluation noted that while structural interventions generally help smaller models, specific isolated tasks (such as Relative Direction and Object Counting) actually saw performance drops when external visual markers were applied. This suggests that the signal-priority hierarchy is not universally absolute, and some small model visual parameters resist being entirely overwritten by text heuristics.

## SECTION 5 — Forward citation chain

The primary anchor paper demonstrating profound textual bias and the illusion of visual understanding is:

**Asadi et al. 2026, "MIRAGE: The Illusion of Visual Understanding" (arXiv:2603.21687)**.

Despite being published recently in March 2026, it has successfully established a forward-citation chain specifically in the domain of multimodal agent safety, foundation model benchmarking, and medical AI evaluation:

- Asadi, M. et al. (2026) "MARCUS: an agentic, multimodal vision-language model for cardiac diagnosis and management." *arXiv preprint*.
- Stanford AIMI / Karargyris et al. (2026) "CheXthought: A global, multimodal resource containing chain-of-thought reasoning traces and synchronized visual attention annotations." *arXiv / HuggingFace*.
- Chao, C-J. et al. (2026) "EchoAtlas: A Conversational, Multi-View Vision-Language Foundation Model for Echocardiography Interpretation and Clinical Reasoning." *medRxiv*.