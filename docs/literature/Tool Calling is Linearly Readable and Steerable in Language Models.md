Tool Calling is Linearly Readable and Steerable in Language Models   





**Zekun Wu, Ze Wang, Seonglae Cho, Yufei Yang, Adriano Koshiyama, Sahan Bulathwela, Maria Perez-Ortiz** *University College London, Holistic AI, Imperial College London*   



Abstract   



When a tool-calling agent picks the wrong tool, the failure is invisible until execution: the email gets sent, the meeting gets missed. Probing 12 instruction-tuned models across Gemma 3, Qwen 3, Qwen 2.5, and Llama 3.1 (270M to 27B), we find the identity of the chosen tool is linearly readable and steerable inside the model. Adding the mean-difference between two tools' average internal activations switches which tool the model selects at 77-100% accuracy on name-only single-turn prompts (93-100% at 4B+), and the JSON arguments that follow autoregressively match the new tool's schema, so flipping the name is enough. The same per-tool means also flag likely errors before they happen: on Gemma 3 12B and 27B, queries where the gap between the top-1 and top-2 tool is smallest produce 14-21x more wrong calls than queries with the largest gap. The causal effect concentrates along one direction, the row of the output layer that produces the target tool's first token: a unit vector along it at matched magnitude already reaches 93-100%, while what is left over leaves the choice almost untouched. Activation patching localises this to a small set of mid and late-layer attention heads, and a within-topic probe across 14 same-domain airline tools reaches top-1 61-89% across five 4B-14B models, ruling out the reading that we are just moving the model along a topic axis. Even base models encode the right tool before they can emit it: cosine readout from the internal state recovers 69-82% on BFCL while base generation reaches only 2-10%, suggesting pretraining forms the representation and instruction tuning later wires it to the output. We measure tool identity selection and JSON schema correctness in single-turn fixed-menu settings; multi-turn agentic transfer is more fragile and is discussed in Limitations.  



------

1 Introduction   



Imagine an LLM assistant is asked to "follow up with the client about tomorrow's meeting."  It has access to `send_email`, `schedule_meeting`, and `Contactss`. The model picks `send_email`, writes a plausible message, and fires it off. But the user wanted to reschedule, not send a reminder. The email goes out; the meeting is missed. This kind of silent failure is common: on the airline benchmark, even 4B-parameter models only succeed 25% of the time, and most failures come down to picking the wrong tool. As agents get access to actions that matter (running code, moving money, approving transactions), one bad tool call can do real damage.  



The tricky part is that we currently have no idea why a model picks one tool over another. It outputs a tool name and some JSON arguments, but there is no way to peek inside and catch a mistake before it happens. Recent work has started looking at this: Healy et al. (2026) detect tool-calling hallucinations from hidden states, and Wang et al. (2026) improve binary decisions about whether to call a tool at all (F1: 0.18 to 0.50). But neither of these traces the actual circuit responsible for selection, and neither shows how to control which specific tool gets chosen when there are many candidates.  



The question we ask is simple: how do language models internally pick among tools, and can we intervene on that process? We investigate this by combining five interpretability methods across three model families (Gemma 3, Qwen 3 / Qwen 2.5, and Llama 3.1), spanning 270M to 27B parameters across 12 instruction-tuned models.  



The core finding is that tool identity is linearly readable and intervenable inside the model. The model's internal state (called the residual stream) has 2,560 dimensions, and a single mean-difference vector along it suffices to switch which tool the model selects. PCA over 15 tool means fits in ~10 directions. Because the structure is so simple, two things become possible.  



1. First, we can switch which tool the model picks by adding a direction vector (the average internal state for the target tool minus the source tool). Using only tool names, this achieves 77-100% across 12 instruction-tuned models.  

   

   

2. Second, we can tell when the model is about to make a mistake: if the top two tools are nearly tied internally, it is much more likely to pick the wrong one.  

   

   

Base (pretrained-only) models extend this picture. On BFCL v3, a Gemma 3 4B base model generates the correct tool on only about 3% of queries, but reading the closest mean-activation direction from its residual stream recovers 75%. Instruction tuning mostly wires the existing internal tool signal into the output layer, rather than creating the signal.  



------

2 Related work   



Most work on tool use in LLMs focuses on making models better at calling tools, whether through specialized training, benchmarks, or prompt design. We ask a different question: not how to improve tool use, but how it works internally. Our work builds on the idea that language models represent high-level concepts as directions in activation space, sometimes called the linear representation hypothesis.  



On the mechanistic side, we combine three tools to trace how the model makes its decision. Activation patching tests which components matter. Sparse autoencoders identify individual features. Cross-layer transcoders decompose computation layer by layer. The superposition framework also predicts that discrete categories sit in nearly orthogonal directions, which fits with our ~10-dimensional subspace for 15 tools.  



------

3 Method: Mean-Difference Steering for Tool Selection   



The idea is straightforward. If different tools produce different average activation patterns, then the difference between two tool averages gives us a direction in activation space that points from one tool to another. Adding that direction during generation should switch the model's tool choice.  



The method has three steps:  



1. For each tool $t_{i}$, we collect $n=2-3$ queries that should trigger that tool and record the model's internal state $h_{i}^{(j)}\in\mathbb{R}^{d}$ at the final token position in the second-to-last layer $l$. The mean activation per tool is $\overline{h}_{i}=\frac{1}{n}\sum_{j}h_{i}^{(j)}$.  

   

   

2. The steering direction from tool $t_{a}$ to tool $t_{b}$ is the normalized mean difference:  

   

   

   $$d_{a\rightarrow b}=\frac{\overline{h}_{b}-\overline{h}_{a}}{||\overline{h}_{b}-\overline{h}_{a}||}$$

   We scale it by $\alpha\cdot sep_{a\rightarrow b}$, where $sep_{a\rightarrow b}=(\overline{h}_{b}-\overline{h}_{a})\cdot d_{a\rightarrow b}$ is the projection gap between the two tool means.  

   

   

3. During generation, we add this vector to the residual stream at layer $l$:  

   

   

   $$h_{l}^{\prime}=h_{l}+\alpha\cdot sep_{a\rightarrow b}\cdot d_{a\rightarrow b}$$

We only intervene at the final token position on the first forward pass; subsequent tokens are generated normally.  



------

4 Experiments   



4.1 Tool identity fits in a small space   



We start by looking at what tool selection looks like inside the model. Each tool has a mean activation vector (2,560 numbers describing its position in the model's internal space). With 15 tools, we want to know: how many independent directions do these 15 vectors actually span? To measure this, we use principal component analysis (PCA).  



Across all models, the first 10 principal components capture about 91% of the variance. We define $k_{90}$ as the smallest number of components needed to reach 90%; this is consistently 9-11 regardless of model family or scale.  





**Table 1: PCA scaling on Gemma 3 4B with real ToolBench APIs**   



| **K TOOLS** | **k90** | **MAX** | **RANDOM** | **COMPRESS** |
| ----------- | ------- | ------- | ---------- | ------------ |
| 50          | 17      | 49      | 43.0       | 35%          |
| 100         | 26      | 99      | 86.0       | 26%          |
| 200         | 36      | 199     | 167.0      | 18%          |
| 500         | 57      | 499     | 392.0      | 11%          |



**Table 2: $k_{90}$ at $K=200$, last layer**   



| **TOOLS**    | **MATCH** | **DIFF** | **READING**   |
| ------------ | --------- | -------- | ------------- |
| Gemma 3 4B   | 36        | +3       | 39 similar    |
| Qwen 3 4B    | 80        | 52-28    | tools spread  |
| Llama 3.1 8B | 72        | 83 +11   | tools cluster |
| Gemma 3 12B  | 84        | 11-73    | tools spread  |

4.2 Tracing the decision from input to output   



To trace the circuit, we decompose the model's computation using cross-layer transcoders. Applying this to Gemma 3 4B, a three-stage pipeline emerges:  



- 

  **Stage 1 (early layers, L0-3):** 38 features fire selectively for specific tools.  

  

  

- 

  **Stage 2 (mid layers, L16-30):** Attention heads lock onto tool-name and entity tokens.  

  

  

- 

  **Stage 3 (late layers, L30-33):** Features handle JSON output formatting.  

  

  

Using activation patching, we find causal results line up with this structure. On Gemma 3 4B, just two attention heads in the middle of the network (L17 H0 and H1) matter more than every other head combined.  





**Table 3: Attribution patching**   



| **MODEL**   | **Peak** | **Depth** | **IT** | **IT/Base** | **Base** |
| ----------- | -------- | --------- | ------ | ----------- | -------- |
| **Gemma 3** |          |           |        |             |          |
| 270M        | L17      | 94%       | 1.6    | 0.4         | 4.0x     |
| 1B          | L25      | 96%       | 7.4    | 12.3x       | 0.6      |
| 4B          | L33      | 97%       | 10.7   | 3.2x        | 3.4      |
| 12B         | L44      | 92%       | 8.5    | 1.6         | 5.3x     |
| 27B         | L61      | 98%       | 14.2   | 1.8         | 7.9x     |

4.3 Can we exploit this structure to control tool selection?   



Adding a direction vector switches which tool gets picked. In the simplest case of 5 tools, steering works perfectly: 60/60 (100%). Random Gaussian vectors at the same norm result in a 0% switch rate. With 15 tools spanning 8 domains, Qwen 3 4B steers at 93% and Gemma 3 4B at 80%.  



4.4 The model also rewrites its arguments   



When steering switches the tool name, the model also rewrites its entire argument payload to match the new tool's schema. This schema adaptation is driven by autoregressive generation. Once the model has committed to a tool name, the rest of the output is a continuation from that name.  





**Table 6: Schema-correct rate across three conditions**   



| **MODEL**       | **A: BASE (src schema)** | **B: PREFILL (tgt schema)** | **C: STEER (tgt schema)** |
| --------------- | ------------------------ | --------------------------- | ------------------------- |
| Gemma 3 4B IT   | 73%                      | 87%                         | 73%                       |
| Qwen 3 4B       | 60%                      | 27%                         | 27%                       |
| Llama 3.1 8B IT | 90%                      | 57%                         | 43%                       |

4.5 When does this structure appear?   



We run steering experiments across 15 models. At 270M, the model cannot tell tools apart at all. By 1B, 5-tool switching works perfectly (100%) but 15-tool is only partial (43%). At 4B and above, all three families reach $>83\%$ on 15 tools. Sparse autoencoders (SAEs) show that at sub-1B scale, tool-related patterns exist as statistical regularities without causal influence. By 1B-IT, features sharpen, suggesting instruction tuning organizes scattered features into functional pathways rather than creating new ones.  



------

5 Discussion   



5.1 Why should we expect linearity?   



The linearity conclusion rests on three independent lines of evidence: activation patching, random vector baselines, and within-topic probing. The linear representation hypothesis predicts that discrete categories end up as directions in activation space. The final layer of the model converts internal states to output probabilities through a linear projection, giving the model a natural incentive to keep tool names linearly separated.  





**Table 9: Decomposing the steering vector**   



| **MODEL**         | **FULL** | **PAR** | **PAR\*** | **ORTH** | **ORTH\*** | **ZERO** |
| ----------------- | -------- | ------- | --------- | -------- | ---------- | -------- |
| Gemma 3 4B IT     | 97%      | 3%      | 100%      | 7%       | 7%         | 0%       |
| Gemma 3 4B base   | 50%      | 3%      | 100%      | 7%       | 7%         | 0%       |
| Qwen 3 4B         | 93%      | 0%      | 100%      | 0%       | 0%         | 0%       |
| Llama 3.1 8B IT   | 77%      | 3%      | 93%       | 17%      | 17%        | 0%       |
| Llama 3.1 8B base | 77%      | 0%      | 100%      | 7%       | 7%         | 0%       |

5.2 Catching mistakes before they happen   



If the top two tools have very similar scores (a small gap), the model is torn between them and much more likely to make a mistake. On instruction-tuned models, the model's own generation is already accurate. On base models, reading the closest mean-activation direction from its residual stream recovers significant accuracy, such as jumping from 3% generation accuracy to 75% readout accuracy on Gemma 3 4B base.  





**Table 11: BFCL v3 tool-name accuracy**   



| **MODEL**                    | **GEN (%)** | **READOUT (%)** | **(PP)** |
| ---------------------------- | ----------- | --------------- | -------- |
| **Base models**              |             |                 |          |
| Gemma 3 1B base              | 2           | 69              | +67      |
| Gemma 3 4B base              | 3           | 75              | +72      |
| Gemma 3 12B base             | ~5          | 61              | +56      |
| Llama 3.1 8B base            | 10          | 82              | +72      |
| **Instruction-tuned models** |             |                 |          |
| Gemma 3 4B IT                | 92          | 74              | -17      |
| Gemma 3 12B IT               | 95          | 66              | -30      |
| Llama 3.1 8B IT              | 90          | 53              | -37      |

------

6 Conclusion   



Tool selection in language models turns out to be linearly readable and intervenable in single-turn, fixed-menu settings. Adding a mean-difference vector to the residual stream switches among 15 tools at 77-100% accuracy, and the JSON output matches the new tool's schema through ordinary autoregressive generation. Activation patching localizes the decision to specific attention heads, and base models encode tool identity well enough for a linear classifier to recover it, implying instruction tuning supplies the routing into structured output.  



The paper concludes that tool calling mechanisms are highly structured, linear, and predictable across multiple leading language model architectures.