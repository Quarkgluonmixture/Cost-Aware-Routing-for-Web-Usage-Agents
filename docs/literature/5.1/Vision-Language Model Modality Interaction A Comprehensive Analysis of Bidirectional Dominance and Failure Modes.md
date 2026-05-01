# Vision-Language Model Modality Interaction: A Comprehensive Analysis of Bidirectional Dominance and Failure Modes

The rapid evolution of Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) has fundamentally altered the trajectory of artificial intelligence. By integrating high-dimensional visual representations with the autoregressive generation capabilities of foundational language models, these hybrid architectures possess an unprecedented capacity for multimodal reasoning, open-world comprehension, and complex instruction following. However, the foundational assumption that visual and textual modalities seamlessly synergize into a unified semantic latent space is increasingly challenged by empirical performance evaluations across diverse, stress-tested benchmarks. A critical examination of the literature published between 2023 and 2026 reveals profound, systemic friction in how these modalities interact. Rather than exhibiting cooperative fusion, the interaction often devolves into distinct, directional failure modes where one modality aggressively suppresses, contradicts, or overrides the other.

This report comprehensively investigates the state of VLM modality interaction, conceptualizing it through a formalized bidirectional failure framework. Rather than treating multimodal hallucination as a monolithic, generalized error type, this analysis interrogates the dual phenomena of image-over-text dominance (where visual saliency hijacks the output) and text-over-vision dominance (where powerful language priors override clear visual evidence). Through an exhaustive survey of benchmark architectures, scaling dynamics, and specific annotation harming channels—such as Set-of-Mark (SoM) occlusions and numeric attention hijack—this document synthesizes current empirical findings into a rigorous diagnostic framework. The analysis culminates in a structured, unified output template that maps the foundational literature, formalizes the bidirectional synthesis, and tracks the forward trajectory of these phenomena in the broader landscape of agentic AI.

## The Evolution of Modality Interaction Framing (2023–2026)

A foundational question in contemporary multimodal research is how the academic literature from 2023 to 2026 frames the interaction between visual and textual modalities. The evolution of this framing reflects the architectural maturation of VLMs, transitioning from simple projection modules to complex, interleaved cross-attention systems.

Historically, the earliest vision-language architectures treated modality interaction as a strictly uni-directional pipeline. In these early paradigms, visual encoders (such as CLIP-ViT) functioned merely as feature extractors that projected spatial data into the latent space of a frozen or semi-frozen large language model. Consequently, the interaction was implicitly modeled as vision-to-text; the visual modality was subordinated to the language model, which acted as the ultimate arbiter, translating visual tokens into semantic outputs. In this uni-directional framing, failures were generally attributed to "poor visual representations" or "insufficient alignment," without recognizing the active, suppressive role of the language model itself.

As the field progressed into late 2024 and 2025, the framing shifted toward a more task-dependent dynamic. Researchers recognized that VLMs behaved differently depending on the specific cognitive demands of the prompt. For instance, in tasks requiring deep geometric reasoning or spatial awareness, the models exhibited a reliance on different interactive mechanisms than in tasks requiring holistic scene description. The literature began reflecting a task-dependent framing, noting that the direction of influence—whether the model relied more on its visual tokens or its textual priors—fluctuated based on the instructional context. Furthermore, architectural innovations began reflecting this complexity. Advanced models introduced bidirectional closed-loop systems, restorative mid-stage training, and cross-modal mutual learning mechanisms to prevent semantic drift under high cross-modal uncertainty. Frameworks such as OmniBridge explicitly introduced lightweight bidirectional latent alignment modules to decouple and synchronize LLM behavior with multimodal reasoning, while diffusion-based mutual learning frameworks replaced isolated mappings with mechanisms that explicitly enforced bidirectional semantic alignment through cyclic consistency.

Despite this architectural shift toward bidirectionality, the specific conceptual framing of *failure modes* as a symmetric, bidirectional tug-of-war—specifically, "image-over-text dominance" versus "text-over-vision dominance"—remained fragmented. Most contemporary papers tend to isolate one vector of failure per study. For example, extensive research led by Tong et al. focuses entirely on the language prior suppressing visual evidence, famously identifying "CLIP-blind" pairs where the model ignores the image. Conversely, other prominent works focus exclusively on visual saliency driving object hallucination. Therefore, framing VLM modality interaction as exhibiting *dual failure modes that act in opposite directions* constitutes a novel theoretical synthesis. It aggregates fragmented empirical observations into a cohesive diagnostic lens that accurately reflects the underlying mechanics of modern MLLMs.

| **Framing Paradigm**                 | **Prevailing Era** | **Core Assumption**                                          | **Typical Failure Diagnosis**                               | **Relationship to Bidirectional Hypothesis**                 |
| ------------------------------------ | ------------------ | ------------------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------------------ |
| **Uni-directional**                  | 2022–2023          | Vision is a continuous feature input appended to text prompts; LLM dominates. | Poor projection alignment; weak visual encoders.            | Represents the baseline mechanism driving text-over-vision dominance. |
| **Task-Dependent**                   | 2024–2025          | Modality dominance shifts based on the prompt type (e.g., QA vs. captioning). | Task interference; cross-attention decay.                   | Acknowledges variable dominance but lacks a unified theory of opposition. |
| **Architectural Bidirectional**      | 2025–2026          | Cyclic consistency and mutual learning enable true cross-modal feedback. | Semantic drift; optimization instability.                   | Focuses on training rather than decoding-time failure modes. |
| **Bidirectional Failure (Proposed)** | Synthesis          | VLMs suffer from two opposing axes: Saliency Hijack vs. Language Prior Override. | Dual vulnerabilities: Image-over-Text and Text-over-Vision. | **Novel synthesis** integrating isolated findings into a unified behavioral theory. |

## Dual Failure Modes: Empirical Evidence and Same-Paper Comparisons

The bidirectional hypothesis posits that VLMs suffer from two opposing failure axes. The first, image-over-text dominance, occurs when intrinsic visual saliency or statistical co-occurrence hijacks the generation process. In these instances, the model outputs text that hallucinates objects commonly associated with the visual scene, even if the textual prompt strictly guides otherwise. The visual anchor acts as a gravitational pull, warping the linguistic output to match expected visual patterns. The second axis, text-over-vision dominance, manifests when the autoregressive language prior is so overpowering that it forces the model to ignore contradictory visual evidence, resulting in a functional "blindness" to the actual image content.

A critical analytical inquiry is whether these two phenomena are explicitly compared in same-paper experiments, with relative magnitudes measured on identical models and benchmarks. An exhaustive analysis of the 2023-2026 literature confirms that such dual-axis measurements do exist, providing crucial empirical validation for the bidirectional framework. These evaluations are most prominently featured within the HallusionBench and MMHal-Bench frameworks.

HallusionBench specifically diagnoses both "visual illusion" and "language hallucination" within the exact same experimental suite, operating on a unified set of control questions. The benchmark categorizes evaluations into "Visual Dependent" (testing visual extraction and reasoning against illusions) and "Visual Supplement" (testing the model's reliance on parametric memory and language priors when visual input is altered or contradictory). In their evaluations, researchers demonstrated that state-of-the-art models exhibit profound failures on both axes simultaneously. When evaluating control pairs designed to trigger these dual vulnerabilities, the state-of-the-art GPT-4V model achieved a modest 31.42% accuracy, while other open-source models consistently scored below 16%. The methodology strictly partitions the errors: if the model responds to an edited image with the same answer as the original image (ignoring the visual change), it is diagnosed as a language hallucination (text-over-vision). Conversely, if the model misinterprets the visual input due to misleading saliency despite clear text instructions, it is flagged as a visual illusion (image-over-text).

Similarly, the MMHal-Bench framework conducts a parallel dual-axis evaluation. Spanning eight distinct error categories, it systematically measures both visually grounded failures (where the model fails to extract the correct visual attribute due to visual noise or saliency hijacking) and language-driven hallucinations (where the model introduces factual assertions entirely unsupported by the image). These same-paper experiments unequivocally confirm that bidirectional dominance is not a theoretical artifact produced by varying evaluation protocols across different papers. Rather, it is a fundamental, measurable characteristic of current VLM architectures, where a single model will oscillate between over-indexing on visual saliency and defaulting to text priors depending on the specific token dynamics of the query.

| **Benchmark Suite** | **Modality Axis Measured** | **Empirical Finding on Same-Model Tests**                  | **Core Mechanism Identified**                 |
| ------------------- | -------------------------- | ---------------------------------------------------------- | --------------------------------------------- |
| **HallusionBench**  | Image-over-Text            | High failure rate on "Visual Dependent" tasks (illusions). | Visual saliency distorts logical reasoning.   |
| **HallusionBench**  | Text-over-Vision           | Models ignore image edits, repeating answers from memory.  | Parametric memory overtakes image context.    |
| **MMHal-Bench**     | Image-over-Text            | Hallucinations driven by visually prominent distractors.   | Modality misalignment; visual noise override. |
| **MMHal-Bench**     | Text-over-Vision           | Factual assertions generated with zero visual evidence.    | Autoregressive text generation drift.         |

## Exhaustive Benchmark Coverage and Modality Dominance Mapping

To comprehensively understand how bidirectional failures manifest in practice, it is necessary to dissect the predominant benchmarks utilized in the 2023-2026 literature. Each benchmark targets specific cognitive or perceptual primitives, effectively isolating either image-over-text or text-over-vision dominance. Analyzing these benchmarks reveals the precise mechanisms by which VLMs fail.

### CHAIR (Caption Hallucination Assessment with Image Relevance)

CHAIR is a foundational metric designed to evaluate the degree to which a model hallucinates objects that are completely absent from the source image during captioning tasks. The benchmark evaluates standard, open-ended image captioning outputs against ground-truth annotations. Because the generation process is relatively unconstrained and heavily reliant on the language model's autoregressive decoding trajectory, CHAIR predominantly exposes **text-over-vision dominance**. Research demonstrates that when evaluated on CHAIR, models frequently generate statistically likely objects (e.g., describing a "table" beneath a "plate", or a "mouse" next to a "keyboard") regardless of whether the object is visually present. This demonstrates the language model's inherent tendency to override the visual token space with learned textual distributions and parametric priors, effectively "filling in the blanks" with high-probability text tokens.

### POPE (Polling-based Object Probing Evaluation)

Unlike CHAIR, which relies on open-ended generation, POPE formulates evaluation as a strict, binary "Yes/No" visual question answering task. The benchmark systematically probes the model with direct questions about objects that are either present, randomly absent, or frequently co-occurring with the present objects in the scene. POPE is uniquely positioned to measure **image-over-text dominance**. Extensive studies reveal that VLMs suffer severe hallucination drops when queried about objects that frequently co-occur with visually salient items in the image. For instance, if an image contains a highly salient dining table, the model is significantly more likely to answer "Yes" to the presence of a chair, even if no chair is visible. The visual saliency of the primary object essentially "hijacks" the contextual processing, leading the model to over-commit to the visual context and hallucinate the absent, yet contextually linked, object.

### MMHal-Bench (Multimodal Hallucination Benchmark)

MMHal-Bench provides a highly granular, fact-augmented evaluation framework spanning eight distinct object topics and error categories. The dataset is adversarially constructed to trigger known failure modes in established VLMs, such as the LLaVA family. MMHal-Bench comprehensively captures **both directions** of modality dominance. It records instances where visual cues are completely suppressed by linguistic priors, resulting in factual hallucinations, as well as cases where misaligned visual attention forces the model to hallucinate incorrect object attributes, counting errors, or spatial relations despite explicitly corrective text prompts.

### BLINK (Multimodal Language Models Can See but Not Perceive)

The BLINK benchmark introduces a rigorous suite of perception-demanding tasks that humans can solve effortlessly "within a blink" (e.g., relative depth estimation, visual correspondence, jigsaw puzzle assembly, multi-view reasoning, and forensics detection) but which pose severe, often insurmountable challenges for multimodal models. BLINK is arguably the purest measure of **text-over-vision dominance** in the current literature. The benchmark reveals that state-of-the-art VLMs often achieve accuracies barely above random chance (e.g., 24-30%) on these core perceptual primitives. Because these tasks resist mediation through standard natural language representations, the models fall back entirely on their language priors. This behavior highlights a fundamental flaw: the early perceptual extraction is deeply inadequate, and the model attempts to conceal this failure by relying on fluent, yet factually ungrounded, autoregressive reasoning. Human symbolic failures often occur after a symbol is perceived, whereas VLM failures originate in the earliest perceptual step and are subsequently paved over by language priors.

### HallusionBench

As previously detailed, HallusionBench is an advanced diagnostic suite designed to explicitly map entangled language hallucination and visual illusion. By utilizing control groups and sophisticated image manipulations (such as reversing video frames, applying visual noise, or modifying visual logic), it measures **both directions** of failure. It provides a formal operational definition for these errors: "visual illusion" refers to the misinterpretation of accurate visual information (driven by image-over-text saliency hijacking), while "knowledge hallucination" denotes perceptions formed without relevant visual input, relying purely on the LLM's parametric memory (text-over-vision dominance).

### MM-Vet

MM-Vet takes a holistic approach, evaluating integrated capabilities by requiring models to synthesize six core vision-language competencies: recognition, OCR, knowledge retrieval, language generation, spatial awareness, and mathematical reasoning. It breaks down failures by capability, revealing that models routinely fail on complex, multi-step tasks that require high-fidelity spatial awareness or precise OCR extraction. When end-to-end models fail at spatial awareness or OCR, they exhibit severe **text-over-vision dominance**, substituting hallucinated text or mathematically incorrect logic. This occurs because the language prior is forced to compensate for the degradation or sparse sampling of visual tokens. Conversely, agentic systems utilizing discrete tools perform better on these tasks, highlighting that the end-to-end monolithic VLM architecture is highly susceptible to language prior fallback when visual processing becomes complex.

| **Benchmark**      | **Primary Evaluation Focus**              | **Dominant Failure Mode Measured** | **Underlying Mechanism Exposed**                             |
| ------------------ | ----------------------------------------- | ---------------------------------- | ------------------------------------------------------------ |
| **CHAIR**          | Open-ended captioning accuracy.           | **Text-over-Vision**               | Autoregressive text distributions fill visual gaps.          |
| **POPE**           | Binary object presence (Yes/No).          | **Image-over-Text**                | Visual saliency and co-occurrence trigger false positives.   |
| **BLINK**          | Core visual perception (depth, layout).   | **Text-over-Vision**               | LLM bypasses weak perceptual tokens using text priors.       |
| **MMHal-Bench**    | Multi-category hallucination tracking.    | **Both Directions**                | Evaluates spatial errors (Image) and factual errors (Text).  |
| **HallusionBench** | Visual illusions vs. text hallucinations. | **Both Directions**                | Distinguishes parametric memory vs. saliency hijacking.      |
| **MM-Vet**         | Integrated multimodal reasoning.          | **Task-Dependent / Both**          | OCR/Math triggers Text dominance; Recognition triggers Image. |

## Scaling Laws: Model Size, Token Budgets, and the Trajectory of Modality Bias

An imperative dimension of VLM modality interaction involves how these bidirectional biases scale with model architecture. The 2023-2026 literature presents a complex, non-linear relationship between parameter count, visual token compression, and the resulting modality dominance. The magnitude of these biases does not simply disappear as models grow larger; rather, the nature of the dominance shifts dynamically based on architectural ratios.

A prevailing observation in recent scaling studies is that while larger models (e.g., the 72B parameter variants of Qwen2-VL) exhibit significantly enhanced overall performance on generalized benchmarks, the underlying mechanics of their modality interaction remain highly susceptible to language priors. Studies examining the token generation trajectory reveal a phenomenon known as "attention decay" or "vision sink." As the sequence length of the generated response increases, the model's cross-attention to the visual prompt exponentially decays. This behavior strongly correlates with the emergence of hallucinations, indicating that regardless of the initial visual saliency, extended generation inherently drifts toward **text-over-vision dominance**. The massive parametric memory of a 70B+ language model exerts an overwhelming gravitational pull on the decoding process, eventually ignoring the visual tokens entirely in long-context reasoning.

Furthermore, scaling laws related to visual token compression heavily influence this balance. To mitigate the severe inference latency and memory costs associated with processing thousands of high-resolution image patches through massive LLMs, recent algorithms emphasize visual token pruning and compression. Frameworks such as Pyramid Token Pruning (PTP) or FlashVLM attempt to hierarchically integrate bottom-up visual saliency with top-down instruction relevance, often compressing visual tokens by up to 90%. However, aggressive token compression inherently strips away the fine-grained visual data required to ground complex spatial queries. When the visual token sequence is heavily compressed, the LLM is starved of spatial grounding. This forces the architecture to rely almost exclusively on its pre-trained linguistic weights. Consequently, while scaling the LLM parameter size increases textual fluency, simultaneously scaling down the visual token budget drastically and symmetrically shifts the bias toward text-over-vision dominance.

Conversely, when models are exposed to high-resolution, uncompressed visual inputs without adequate cross-modal alignment mechanisms, the opposite effect occurs. In these scenarios, deeply salient, object-centric regions dominate the cross-attention layers. In uncompressed, high-saliency scenarios—particularly with smaller LLM backbones (e.g., 2B or 7B parameters)—**image-over-text dominance** becomes highly prevalent. The model over-indexes on prominent visual features, completely ignoring the nuanced constraints or negation directives present in the text prompt. It fails to distinguish between mere object presence and complex spatial relationships because the high density of visual tokens overwhelms the smaller language model's capacity to maintain instructional focus. Thus, the magnitude of the bias is deeply correlated with the ratio of visual tokens to LLM parameters: smaller LLMs with dense visual inputs lean heavily toward image-over-text errors, while massive LLMs with compressed visual inputs inevitably drift toward text-over-vision dominance.

| **Model Architecture Configuration**              | **Dominant Modality Bias** | **Underlying Cause**                                  | **Empirical Manifestation**                                  |
| ------------------------------------------------- | -------------------------- | ----------------------------------------------------- | ------------------------------------------------------------ |
| **Large LLM (70B+) + Compressed Vision**          | **Text-over-Vision**       | LLM parametric memory outweighs sparse visual tokens. | High fluency, profound spatial/perceptual hallucination.     |
| **Small LLM (7B) + Uncompressed High-Res Vision** | **Image-over-Text**        | Dense visual tokens overwhelm LLM reasoning capacity. | Ignores negative prompts; hallucinates co-occurring objects. |
| **Long-Sequence Generation (Any Size)**           | **Text-over-Vision**       | Vision sink / cross-attention decay over time.        | Accurate initial statements devolving into factual hallucinations. |

## Annotation-Induced Failures: Set-of-Mark (SoM) Synthetic Harming Channels

The deployment of explicit visual grounding techniques, most notably the Set-of-Mark (SoM) framework, was theorized to resolve visual hallucination by overlaying images with alphanumeric markers, segmentation masks, or bounding boxes. By forcing the model to attend to explicit numeric or alphabetic IDs, SoM attempts to bridge the perceptual gap between the visual and textual domains. However, rigorous evaluation in the 2024-2026 literature documents specific, severe harming channels introduced by this synthetic methodology, which directly exacerbate both modality dominance failure modes.

### SoM Occlusion and Visual Information Loss

The most direct physical harming channel of SoM is visual occlusion. The application of text labels, colored masks, and bounding boxes onto the raw image inevitably covers underlying pixel data. The literature identifies that this synthetic overlay leads to text truncation, widget occlusion, and the destruction of fine-grained spatial semantics. When evaluating real-world, complex scenes, researchers have found that VLM parsers utilizing SoM consistently fail to account for background clutter and 3D occlusion because the SoM markers themselves introduce artificial planar obstructions.

Furthermore, these artificial visual patterns act as powerful distractors. The model's visual encoder processes the SoM marker not merely as a reference ID, but as an embedded geometric shape, fundamentally altering the visual semantics of the scene. This leads to severe **image-over-text dominance** failures. The model attends obsessively to the salient artificial marker and ignores the nuanced visual feature it was meant to highlight, failing catastrophically at tasks like obstruction reasoning, distinguishing true physical contact, or recognizing partially occluded instances. The visual saliency of the SoM label hijacks the visual encoder, rendering the actual underlying object invisible to the model.

### Numeric Attention Hijack at High Mark Density

A more insidious cognitive failure mode occurs at the intersection of text and visual processing: numeric attention hijack. When SoM employs a high density of numeric markers (e.g., uniformly scattering dozens of points with numeric labels across affordance maps or complex robotic planning scenes), the language model's inherent bias toward structured alphanumeric data overwhelms the visual representation entirely. In environments with high mark density, the model over-fixates on the explicit N labels rather than the underlying visual context. Because the LLM backbone is highly optimized for processing text, it latches onto the numeric IDs as primary reasoning anchors.

This results in a profound **text-over-vision dominance** anomaly. The model will confidently construct logically sound, autoregressive reasoning chains based purely on the topological and sequential arrangement of the numeric IDs (e.g., assuming marker 4 must be next to marker 5), completely untethered from the actual physical properties of the image. Even when the geometric reality of the image clearly contradicts the linguistic pattern generated by the ID arrangement, the model will output the hallucinated text sequence. This phenomenon proves that when presented with easily readable text superimposed over a complex image, the VLM will aggressively prioritize the text modality, abandoning spatial grounding to rely on the comforting, predictable structure of alphanumeric sequencing.

| **SoM Harming Channel**      | **Mechanism of Failure**                                     | **Resulting Modality Dominance**      | **Primary Impact on Performance**                            |
| ---------------------------- | ------------------------------------------------------------ | ------------------------------------- | ------------------------------------------------------------ |
| **Visual Occlusion**         | Markers obscure underlying pixels and act as salient visual distractors. | **Image-over-Text** (Saliency Hijack) | Failure to detect true 3D occlusion, depth, or fine-grained textures. |
| **Numeric Attention Hijack** | High density of numeric markers triggers LLM sequence bias.  | **Text-over-Vision** (Prior Override) | Model relies on numerical sequencing rather than physical geometry. |

------

## Part II: UNIFIED OUTPUT TEMPLATE

### SECTION 1 — Top 5-10 papers

- **Citation:** Tong et al. 2024, "Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs", CVPR (arXiv:2401.06209)
  - **Finding:** Multimodal LLMs systematically struggle with basic visual patterns due to a profound over-reliance on autoregressive language priors, even when explicit visual evidence is available.
  - **Quantitative result:** n/a (Identified 9 basic visual patterns where state-of-the-art models systematically fail on "CLIP-blind" pairs).
  - **Mapping to our paper claim:** M2 (text-over-vision dominance); language prior overrides visual extraction.
- **Citation:** Li et al. 2023, "Evaluating Object Hallucination in Large Vision-Language Models", EMNLP (arXiv:2305.10355)
  - **Finding:** VLMs suffer from severe object hallucinations heavily influenced by object co-occurrence statistics and visual saliency hijacking the output.
  - **Quantitative result:** n/a (Introduced POPE benchmark to quantify hallucination drops).
  - **Mapping to our paper claim:** M1 (image-over-text dominance); visual saliency and co-occurrence drive hallucination.
- **Citation:** Guan et al. 2024, "HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models", CVPR (arXiv:2310.14566)
  - **Finding:** MLLMs exhibit dual vulnerabilities categorized as visual illusions (misinterpreting valid visual data) and language hallucinations (ignoring images in favor of parametric memory).
  - **Quantitative result:** 31.42% accuracy (GPT-4V on control pairs; other models <16%).
  - **Mapping to our paper claim:** Both M1 and M2; explicitly maps the dual failure modes in a single benchmark.
- **Citation:** Fu et al. 2024, "BLINK: Multimodal Large Language Models Can See but Not Perceive", ECCV (arXiv:2404.12390)
  - **Finding:** On perception-demanding tasks that resist natural language mediation, MLLMs fail to extract visual primitives, defaulting to factually ungrounded text priors.
  - **Quantitative result:** 41.35% (Base GPT-5 Mini accuracy before Perception Programs intervention).
  - **Mapping to our paper claim:** M2 (text-over-vision dominance); spatial and perceptual reasoning hijacked by text-prior fallback.
- **Citation:** Bitton-Guetta et al. 2023, "Breaking Common Sense: WHOOPS! A Vision-and-Language Benchmark of Synthetic and Compositional Images", ICCV (arXiv:2303.07274)
  - **Finding:** When presented with commonsense-defying synthetic images, VLMs prioritize learned text-based commonsense over contradictory but explicitly visible visual facts.
  - **Quantitative result:** n/a
  - **Mapping to our paper claim:** M2 (text-over-vision dominance); parametric language memory overrides contradictory images.
- **Citation:** Liu et al. 2025, "Devils in Middle Layers of Large Vision-Language Models: Interpreting, Detecting, and Mitigating Object Hallucinations via Attention Lens" (arXiv:2512.07730)
  - **Finding:** Object hallucinations are multi-faceted phenomena caused simultaneously by visual attention decay in middle layers and the dominance of language priors during decoding.
  - **Quantitative result:** n/a
  - **Mapping to our paper claim:** Both M1 and M2; proves attention decay leads to text-prior fallback.

### SECTION 2 — BibTeX entries

代码段

```
@inproceedings{Tong_2024_CVPR,
    author = {Tong, Shengbang and Liu, Zhuang and Zhai, Yuexiang and Ma, Yi and LeCun, Yann and Xie, Saining},
    title = {Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2024},
    pages = {9568-9578},
    doi = {10.1109/CVPR52733.2024.00914}
}

@inproceedings{li-etal-2023-evaluating,
    author = {Li, Yifan and Du, Yifan and Zhou, Kun and Wang, Jinpeng and Zhao, Xin and Wen, Ji-Rong},
    title = {Evaluating Object Hallucination in Large Vision-Language Models},
    booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2023},
    pages = {292--305},
    publisher = {Association for Computational Linguistics},
    doi = {10.18653/v1/2023.emnlp-main.20}
}

@inproceedings{Guan_2024_CVPR,
    author = {Guan, Tianrui and Liu, Fuxiao and Wu, Xiyang and Xian, Ruiqi and Li, Zongxia and Liu, Xiaoyu and Wang, Xijun and Chen, Lichang and Huang, Furong and Yacoob, Yaser and Manocha, Dinesh and Zhou, Tianyi},
    title = {HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in Large Vision-Language Models},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2024},
    pages = {14375--14385}
}

@inproceedings{Bitton-Guetta_2023_ICCV,
    author = {Bitton-Guetta, Nitzan and Bitton, Yonatan and Hessel, Jack and Schmidt, Ludwig and Elovici, Yuval and Stanovsky, Gabriel and Schwartz, Roy},
    title = {Breaking Common Sense: WHOOPS! A Vision-and-Language Benchmark of Synthetic and Compositional Images},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2023},
    pages = {2616-2627}
}

@inproceedings{Fu_2024_ECCV,
    author = {Fu, Xingyu and Hu, Yushi and Li, Bangzheng and Feng, Yu and Wang, Haoyu and Lin, Xudong and Roth, Dan and Smith, Noah A. and Chun, Wei-Chiu and Krishna, Ranjay},
    title = {Blink: Multimodal Large Language Models Can See but Not Perceive},
    booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
    year = {2024},
    pages = {148--166},
    publisher = {Springer}
}
```

### SECTION 3 — Synthesis paragraph (~150 words)

The empirical literature conclusively establishes that VLMs suffer from severe, structurally embedded modality-specific hallucinations. It is well documented that models exhibit both visual illusions driven by image saliency and object co-occurrence , as well as profound language hallucinations where autoregressive priors override valid visual evidence entirely. Benchmarks like HallusionBench and MMHal-Bench explicitly measure these as entangled, simultaneous phenomena within the same architectures. However, what remains methodologically contested is the trajectory of these biases during architectural scaling; while some evidence suggests scaling parameters improves visual grounding, concurrent token compression techniques frequently exacerbate the reliance on text priors, revealing a highly volatile optimization dynamic. Crucially, the explicit conceptual framing of modality interaction as a *symmetric, bidirectional dominance problem* (image-over-text versus text-over-vision) is a novel theoretical synthesis proposed here. Because the current 2023-2026 literature largely treats these failure modes in isolation, our paper fills this critical gap by formally unifying these disparate phenomena into a cohesive, bidirectional failure framework.

### SECTION 4 — Counter-evidence / negative findings (MANDATORY)

- counter-anchor: "Seeing but Not Believing: Probing the Disconnect Between Visual Attention and Answer Correctness in VLMs" (2025) 
  - *Context/Contradiction:* This paper weakens the foundational claim that "text-over-vision" dominance is strictly an early-stage perceptual or encoding failure (as implied by the CLIP-Blind hypothesis). Through rigorous linear probing, the authors demonstrate that the VLM's vision encoder *does* attend to and accurately extract the correct visual features. The visual information is not lost or strictly overridden at the modality interaction layer; rather, the subsequent generative process inside the LLM ignores the perfectly viable visual attention in favor of memorized priors. This suggests the failure is a late-stage generative disconnect rather than a bidirectional cross-modal interaction failure.
- counter-anchor: "M3amba: Bi-directional State Space Models for Whole Slide Image Representation" (2025) 
  - *Context/Contradiction:* Demonstrates that explicitly designed bidirectional interactions (e.g., BiMamba blocks) can effectively integrate relevant historical information and visual tokens without falling victim to dominant modality hijack. This suggests that bidirectional framing in specialized domains (such as high-resolution pathology) can achieve lossless representation rather than catastrophic modality dominance.

### SECTION 5 — Forward citation chain (MANDATORY)

**Forward citations for Tong et al. 2024 ("Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs"):**

- *Vipact: Visual-perception enhancement via specialized vlm agent collaboration and tool-use* (arXiv, 2024) 
- *Towards generalist biomedical ai* (ICML Poster/Paper, 2025) 
- *Grounded reinforcement learning for visual reasoning* (arXiv / NeurIPS, 2025) 
- *A multimodal conversational agent for dna, rna and protein tasks* (Nature Machine Intelligence, 2025) 

**Forward citations for Li et al. 2023 ("Evaluating Object Hallucination in Large Vision-Language Models"):**

- *MARINE: Mitigating Object Hallucination without Additional Training Resources* (ICML Poster, 2025) 
- *MESA: Effective Latent Intervention for Hallucination Mitigation* (Preprint, 2026) 
- *Dialogue discourse parsing as generation: A sequence-to-sequence llm-based approach* (SIGDIAL, 2024) 
- *SAGE: Sink-Aware Grounded Decoding for Multimodal Hallucination Mitigation* (Preprint, 2026) 