## TL;DR

Token log-prob (logprob) is a useful but unreliable routing signal for small VLM agents without calibration; it is often poorly calibrated and weak under distribution shift, multi-step agent uncertainty, and verbosity. Temperature scaling or learned calibration on a held-out set and combining predictive entropy with observation-adaptive rules are the best intrinsic strategies identified in the literature.

----

## Logprob reliability

Token log-prob measures the model’s instantaneous predictive confidence for generated tokens and is frequently used as a cheap, intrinsic routing metric in agent systems. Multiple studies report that raw token probability (or its average across tokens) is often miscalibrated for both language and vision-language models, and that one-step logprob does not reliably propagate through multi-step agent trajectories without explicit calibration or additional modeling of uncertainty [1] [2] [3].

- Evidence summary  
  - **Poor calibration in agents** Raw one-step output uncertainty (token logprob) is insufficient for multi-step agent reliability and can mislead routing unless uncertainty propagation is modeled explicitly [1].  
  - **VLM calibration issues** Vision–language models are not inherently well calibrated; simple post-hoc temperature scaling substantially reduces calibration error across tasks and domain shifts, indicating raw logits/logprobs need correction before use as a confidence proxy [3].  
  - **Small-model routing studies** Work that explicitly studies small-large routing finds that raw average token probability needs calibration to be useful for routing between small and large models; uncalibrated scores yield weak decision boundaries [4] [5].

- Quantitative evidence availability  
  - Several papers demonstrate statistically significant calibration improvements after temperature scaling or learned calibration, but specific global correlation coefficients between raw logprob and end-task success for small VLM web-agent benchmarks (WebArena / VisualWebArena) are not provided in the supplied literature; therefore precise correlation/AUROC numbers for Qwen3-VL-4B on those benchmarks are unavailable in the corpus [1] [3] [4].  

----

## Logprob failure modes

Logprob fails systematically in several settings that commonly arise in web agents and multimodal tasks; understanding these modes explains why raw logprob should not be used alone for routing.

- **Overconfidence under distribution shift**  
  - **Why** Models produce high-probability outputs on familiar-format but wrong answers when inputs shift out-of-distribution, leading to overconfident logprobs [3] [2].  
  - **Agent implication** This causes mistaken routing to light-weight (DOM-only) modes when the hybrid (DOM+screenshot) mode was needed.

- **Multi-step uncertainty accumulation**  
  - **Why** A good one-step logprob does not reflect downstream error propagation in multi-step plans; single-token confidence can be optimistic for long trajectories [1].  
  - **Agent implication** Routing based only on current-step logprob misses compound-plan failure risk [1].

- **Verbosity and input-length bias**  
  - **Why** Average-token logprob and token-entropy metrics can be biased by response length or verbose HTML inputs; low-capacity models can be misled by verbose HTML and show spurious confidence changes [6] [4].  
  - **Agent implication** Longer observation (full HTML) can increase apparent confidence for high-capacity models but decrease reliability for small models [6].

- **Instruction-following vs factual tasks**  
  - **Why** Models trained to follow instructions may assign high probability to compliant but factually wrong continuations (instructional fluency vs factual correctness mismatch) [2].  
  - **Agent implication** Routing on logprob could favor actions that look well-formed but are semantically incorrect for factual grounding tasks.

- **Modality mismatch (visual vs text inputs)**  
  - **Why** VLMs exhibit different calibration behavior in image-conditional vs pure-text tasks; raw logits from multimodal encoders are often mis-scaled relative to text-only counterparts [3].  
  - **Agent implication** Logprob computed on text tokens that are conditioned on image features may not reflect true visual grounding uncertainty [3].

- **Empirical note**  
  - Studies recommend explicit calibration (e.g., temperature scaling or learned mapping on held-out tasks) or richer uncertainty metrics because raw logprob frequently underperforms as a binary routing signal under the above failure modes [1] [3] [5].

----

## Intrinsic signals compared

This table compares intrinsic (no extra model call / no external verifier) signals that the literature proposes or analyzes as routing indicators for agents.

| Signal                    | Definition                                                   |                                               How to compute | Calibration / reliability evidence                           | Known failure modes                                          | Applicability to agent actions                               |
| ------------------------- | ------------------------------------------------------------ | -----------------------------------------------------------: | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Token predictive entropy  | Entropy of the model’s token distribution at a timestep: H = −∑ p(token) log p(token) | Compute per-token entropy and average across action tokens or aggregate by max/mean | Predictive entropy is a standard UQ proxy; surveys show it is informative but also requires calibration and can be dominated by distributional shifts [2] [3] | Overconfidence under shift; verbosity bias if averaged naively [2] | Good for deciding whether to probe more info before clicking/typing; used per-decision |
| Average token logprob     | Mean log probability across tokens in the generated action/response | Average of log p(token) across tokens in the candidate action | Frequently used in small-large routing work but reported as poorly calibrated raw; requires post-hoc scaling or learned mapping for reliable routing [4] [5] | Length/verbosity bias; miscalibration under OOD; multi-step accumulation problems [1] [6] | Simple, low-cost routing for click/type/navigation but should be calibrated first |
| Top-k mass / vocab spread | Sum of probability mass in top-k tokens (e.g., top-1 or top-5) or metrics like Gini on the distribution |     top_k_mass = ∑_{i≤k} p_i ; or compute mass concentration | Acts as a sharpness measure; literature treats it as complementary to entropy; needs calibration similar to logits [2] | Can be high when model is confidently wrong; sensitive to rank jitter [2] | Can flag confident single-action choices (e.g., single-button click) |
| Semantic entropy          | Entropy over semantic labels/intent classes derived from distribution in a semantic space or few-shot classifier | Map model outputs to semantic labels and compute label distribution entropy (requires mapping but no extra model) | Proposed refinement to predictive entropy for LLMs/VLMs; survey identifies it as promising but still nascent [2] [3] | Requires reliable mapping from tokens to semantic labels; can fail if mapping is noisy [3] | Useful for intent-level routing (choose screenshot when intent entropy high) |
| Attention-weight entropy  | Dispersion of attention weights (how concentrated attention is across inputs) | Compute entropy across attention weights in a chosen layer/head | Mentioned as exploratory UQ feature in surveys; empirical evidence is limited and mixed [2] | Attention does not always correlate with model certainty or correctness [2] | Potentially informative about whether model grounded on DOM vs image, but evidence sparse |
| Perplexity of the action  | Exponential of average negative logprob over action tokens   |                           perplexity = exp(−(1/N) ∑ log p_i) | Conceptually identical to average logprob; suffers same miscalibration issues and needs calibration [2] | Same as token logprob; length sensitivity                    | Same as average token logprob                                |
| Self-consistency variance | Variance across multiple sampled generations (diversity/consensus) | Generate k samples and compute variance or entropy across the set | Empirically effective to signal uncertainty but **requires extra model calls**, so not intrinsic per the user's constraint; surveys flag it as strong when allowed [2] | Costly; not intrinsic                                        | Useful if extra calls are allowed, but excluded from intrinsic-only routing |
| Verbalized confidence     | Model output phrase like “I am X% confident”                 |    Extract or prompt the model to state a numeric confidence | For well-calibrated, high-cap models verbal introspection can be useful; some studies on refusal prediction show high accuracy when restricted to high-confidence outputs [7] | Verbalized numbers are untrustworthy without calibration and can be gameable by instruction tuning [2] | Cheap to read, but often unreliable unless calibrated on the same domain |

Notes on table entries: surveys and empirical papers consistently emphasize that raw logits/logprobs and derived measures (entropy, perplexity, top-k mass) require post-hoc calibration to be reliable; semantic- and attention-based signals are promising but have limited, mixed empirical support [2] [3] [1] [4] [5].

----

## Threshold setting methods

Choosing thresholds for intrinsic signals is addressed by several papers; three practical families are discussed.

- **Fixed threshold**  
  - **Method** Pick an absolute value (e.g., average token prob < 0.2) and route when below it. This is simplest but brittle across domains and models [5].  
  - **Failure** Sensitive to model scale, dataset shift, and action-length biases.

- **Percentile-based threshold**  
  - **Method** Use a calibration dataset to compute the Nth percentile of the chosen signal and route on examples below (or above) that percentile. Percentile rules adapt to score distributions but may be unstable when calibration set is small [4].  
  - **When used** Small-large routing studies find percentile thresholds useful for preserving cost while capturing hard instances [4].

- **Task-adaptive learned mapping**  
  - **Method** Fit a small calibration model or temperature scaling (single scalar) on a held-out set to map raw signals to calibrated probabilities; some frameworks learn per-model µ,σ or scalar temperature and then apply cost-aware decision rules [3] [8] [5].  
  - **Evidence** Temperature scaling reliably reduces Expected Calibration Error (ECE) for VLMs across domain shifts [3]. ODAR-style approaches estimate per-model distributions (µ,σ) on calibration sets to form adaptive routing thresholds [8].  
  - **Tradeoffs** Requires a representative calibration set but yields the most robust routing in the literature.

- Practical recommendation from studies: use a small held-out calibration set to learn either a scalar temperature (for VLM logits) or a simple monotonic mapping from average token logprob to success probability, and then set a cost-aware decision rule (e.g., route to hybrid mode if calibrated success-probability < p_threshold) [3] [5] [8].

----

## Recommendation for 4B VLM

The supplied literature contains no study that reports calibration or routing metrics specifically for Qwen3-VL-4B on WebArena or VisualWebArena, so direct quantitative claims for that exact model are not available in the corpus; therefore a prescriptive recommendation must rely on general findings about small VLMs and small-LM routing experiments. Insufficient evidence for Qwen3-VL-4B exists in the provided papers.

- Practical signal choices given the literature and a locally-run 4B VLM (Qwen3-VL class behavior inferred from small-model studies)  
  - **Primary signal**: **Calibrated average token logprob** — compute the mean token logprob for the candidate action, fit a temperature or monotonic mapping on a held-out calibration set (task-specific examples), and use the calibrated value as the routing score [5] [3].  
  - **Complementary signal**: **Predictive entropy (average per-token)** — use alongside calibrated logprob; if both indicate low confidence, prefer DOM+screenshot hybrid observations [2] [3].  
  - **Observation-adaptive rule**: For a small VLM, prefer compact DOM-only observations by default and escalate to DOM+screenshot when the calibrated score falls below a learned task-adaptive threshold (percentile or calibrated probability) or when semantic/intent entropy is high [6] [4].

- How to implement (concrete steps)  
  1. **Collect a modest calibration set** of representative web-agent episodes labeled for success/failure under DOM-only and hybrid modes.  
  2. **Compute candidate intrinsic signals** per action: average token logprob, per-token predictive entropy, and top-1/top-5 mass.  
  3. **Fit calibration**: temperature-scale logits or fit a small isotonic/Platt-style monotonic mapper from mean logprob to empirical success probability on the calibration set [3] [5].  
  4. **Select threshold**: choose a calibrated success-probability cutoff (or a percentile of calibrated scores) tuned to your cost-accuracy trade-off; prefer task-adaptive mapping over fixed raw thresholds [8] [4].  
  5. **Fallback policy**: if intrinsic signals disagree (e.g., low logprob but low entropy), escalate to DOM+screenshot and/or a conservative hybrid policy.

- Why this aligns with the literature  
  - Temperature scaling reliably improves VLM calibration across distributional shifts, making calibrated logits a stronger routing basis than raw logprob [3].  
  - Small-large routing work demonstrates that uncalibrated average token probability is weak but becomes useful when explicitly calibrated, and percentile or learned decision boundaries perform better than naive fixed thresholds [4] [5].  
  - Observation-ablation work shows small models generally perform better with compact DOM inputs and benefit from hybrid observations only when intrinsic signals indicate uncertainty or failure risk [6].

- Quantitative anchors from related studies (where available)  
  - **Calibration helps** Temperature scaling and small calibration sets substantially reduce calibration error in VLMs in the cited study, improving reliability under shift [3].  
  - **High-confidence filtering works for introspection** In related introspection/refusal work, restricting to high-confidence model outputs yielded very high accuracy (e.g., 98.3% accuracy when selecting only high-confidence predictions for well-calibrated models) — illustrating the power of calibrated confidence for routing when calibration is effective [7].  
  - **No direct AUROC/ECE for Qwen3-VL-4B** The corpus does not contain per-signal AUROC, ECE, or correlation coefficients on WebArena/VisualWebArena for Qwen3-VL-4B, so exact thresholds must be derived empirically on a held-out calibration set for that model (insufficient evidence otherwise).

Summary recommendation: treat raw logprob as a starting signal but always calibrate it (temperature or learned mapping) on a small held-out set; combine it with predictive entropy and an observation-adaptive policy that uses DOM-only by default for a 4B VLM and escalates to DOM+screenshot when calibrated confidence is low or semantic entropy is high [3] [5] [6] [8].

----