# Paper Reading Note — *Where’s the Plan? Locating Latent Planning in Language Models with Lightweight Mechanistic Interventions*

**Source:** Ma & Rui, *Where’s the Plan? Locating Latent Planning in Language Models with Lightweight Mechanistic Interventions*, arXiv:2605.07984v1, 2026-05-08.  
**Use case for our project:** mechanistic evidence / latent planning / probe-vs-causality distinction / representation-site localization.

---

## 0. One-sentence summary

This paper shows a very important mechanistic lesson: **a model may encode future-relevant information in a hidden state, but that does not mean the model actually uses that hidden state causally during generation**. Linear probes find rhyme-planning information at the newline position in many models, but activation patching shows that only **Gemma-3-27B** causally relies on the newline as a planning site; Qwen3 and Llama mostly keep using the original rhyme word token.

---

## 1. Core research question

Autoregressive LMs generate one token at a time, but they can produce outputs that require long-range structure, such as rhyming couplets.

The paper asks:

> Where inside the model does future-relevant planning information live, and does the model actually use it causally?

The authors call this **planning site formation**.

They distinguish two levels of evidence:

| Level | Meaning | Method |
|---|---|---|
| **Planning-compatible representation** | Future-relevant information is decodable from a hidden state | Linear probing |
| **Causally active planning site** | Replacing that hidden state changes generation toward another target | Activation patching |

This distinction is the whole paper’s backbone.

---

## 2. Task setup: rhyming couplet completion

The task is deliberately clean and structured.

A prompt contains the first line of a rhyming couplet. The model must generate the second line so that its final word `r2` rhymes with the first line’s final word `r1`.

Example:

```text
A rhyming couplet:
She felt a sudden sense of fright,
```

Expected completion:

```text
and hoped that dawn would end the night.
```

Here:

- `r1 = fright`
- `r2 = night`
- `r2` must rhyme with `r1`

They center token positions around the newline after the first line:

| Position | Meaning |
|---|---|
| `i = 0` | newline token |
| `i < 0` | tokens before newline |
| `i > 0` | generated second-line tokens |
| last word token | `i = -1` for Qwen/Llama; `i = -2` for Gemma due to tokenization |

The key question:

> Does the model move rhyme-planning information from the original rhyme word `r1` into the newline token before it starts generating the second line?

---

## 3. Method 1 — Linear probing

### 3.1 What the probe does

For each layer `ℓ` and position `i`, the authors train a linear probe:

```math
f_{W,b}(h_{\ell,i}) = \mathrm{softmax}(Wh_{\ell,i} + b)
```

The probe tries to predict the future rhyme token `r2` from hidden state `h_{\ell,i}`.

If a hidden state contains future-rhyme information, the probe should predict `r2` better than chance.

### 3.2 Probe training details

- Dataset: 1,200 synthetic rhyming couplets generated with Claude Sonnet 4.6
  - 1,000 train
  - 200 validation
- Optimizer: AdamW
- Learning rate: `1e-4`
- Weight decay: `1e-3`
- Batch size: 32
- Epochs: 10
- Metrics:
  - Top-5 accuracy
  - Rhyme accuracy using CMU Pronouncing Dictionary
  - Wilson 95% confidence intervals

### 3.3 Negative control: general text

They also train probes on general text from The Pile.

Purpose:

> Check whether future-token information is generically decodable everywhere, or whether the signal is specific to structured rhyme generation.

Result:

- In general text, probe accuracy decays monotonically as lookahead distance `k` increases.
- At `k = 8`, probe performance overlaps the unigram baseline.
- This suggests the rhyme-probe signal is not just a generic property of residual streams.

---

## 4. Main probing result

Linear probes show that rhyme-relevant information is strongly decodable at:

1. the original rhyme word position, and
2. the newline position.

The newline result matters because the newline happens before the model starts generating the second line. So decodability at newline suggests the model may have formed a forward-looking representation.

### Key finding

Planning-compatible newline representations strengthen with model scale.

The maximum gap between newline probe accuracy and first-generated-position probe accuracy grows with scale:

| Family | Pattern |
|---|---|
| Qwen3 | Smaller models mostly overlap zero; largest Qwen3-32B shows nonzero gap |
| Gemma-3 | Positive at every scale; cleanest monotonic trend |
| Llama-3 | Smaller models mostly overlap zero; largest Llama-3.1-70B shows nonzero gap |

For Gemma-3, the top-5 accuracy gap rises from about **0.11 at 1B** to **0.38 at 27B**.

### Important interpretation

This is **encoding evidence**, not causal evidence.

The newline hidden state contains rhyme-relevant information, but the model might not actually read from that position during generation.

This is exactly why the paper then moves to activation patching.

---

## 5. Method 2 — Activation patching

### 5.1 Clean/corrupt setup

The authors construct prompt pairs that differ only in the rhyme word.

Example:

```text
Clean:
... sense of fright,

Corrupt:
... sense of fear,
```

Then they run the corrupt prompt, cache a hidden state, and insert that hidden state into the clean run at a chosen layer and token position.

If patching a position/layer causes the clean completion to rhyme with the corrupt word, then that hidden state is causally involved in rhyme planning.

Example:

```text
Clean target rhyme: fright → night
Corrupt target rhyme: fear → appear
```

If patching makes the clean prompt generate something like:

```text
and hoped that someone would appear.
```

then the patch successfully redirected the rhyme family.

### 5.2 Patching details

- Positions patched:
  - last word token
  - newline token `i = 0`
- Sweep across all layers
- 5 prompt pairs
- 20 stochastic samples per prompt pair
- Main per-layer result: `N = 100`
- Confidence intervals:
  - 95% cluster bootstrap over prompt pairs
  - pair is treated as the unit of independence

---

## 6. Main causal result

The causal result is much sharper than the probing result.

### 6.1 Gemma-3-27B shows a representational handoff

In **Gemma-3-27B**:

- early layers: patching the last word token works strongly
- around layer 30: last-word patching drops
- simultaneously: newline patching rises
- newline patching peaks at layer 33 with corrupt rhyme rate:

```text
0.63 [0.48, 0.78]
```

The authors call this a **representational handoff**:

> The causal planning site migrates from the original rhyme word token to the newline token.

### 6.2 Qwen3 and Llama do not show the handoff

In **Qwen3-32B** and **Llama-3.1-70B**:

- last-word patching remains strong across layers
- newline patching stays near zero

So these models may encode rhyme information at the newline, but they do not causally use the newline as the planning site.

### 6.3 All-layer patching table

When all layers are patched simultaneously:

| Model | Last word corrupt rhyme rate | Newline corrupt rhyme rate |
|---|---:|---:|
| Qwen3-32B | 76% `[67, 83]` | 1% `[0, 5]` |
| Gemma-3-27B | 85% `[77, 91]` | 67% `[57, 75]` |
| Llama-3.1-70B | 75% `[66, 82]` | 2% `[1, 7]` |

This is the cleanest evidence that Gemma-3-27B is special.

---

## 7. Sparse attention-head mechanism in Gemma-3-27B

After finding the newline handoff, the authors localize it.

They ask:

> Is the handoff implemented by many diffuse components, or by a small set of attention heads?

### 7.1 Attention-weight ranking

They inspect attention from newline token `i = 0` to the last word token `i = -2` in Gemma-3-27B.

The highest-attending heads are:

| Rank | Layer | Head |
|---:|---:|---:|
| 1 | 30 | 4 |
| 2 | 28 | 14 |
| 3 | 28 | 15 |
| 4 | 30 | 5 |
| 5 | 28 | 29 |

Some of these heads attend almost exclusively from newline to the last word:

- Layer 30 head 4: attention weight ≈ 0.99
- Layer 28 head 14: ≈ 0.97
- Layer 28 head 15: ≈ 0.95

### 7.2 Simple top-k head patching

They patch the top-k heads at the newline.

Result:

- `k = 1, 2, 3`: near-zero effect
- `k = 5`: corrupt rhyme rate jumps to **46%**
- This recovers about **73%** of the full-residual reference effect

### 7.3 Two-stage path patching

They then use stricter two-stage path patching:

1. Patch the residual at the last word token.
2. Cache the candidate heads’ outputs at the newline.
3. Insert those head outputs into the clean run.

Result:

- top-5 heads recover **57%** corrupt rhyme rate
- this is about **90%** of the full-residual reference
- random head sets and comma-control sets remain at zero
- MLP patches yield zero corrupt rhyme rate

Interpretation:

> The Gemma-3-27B handoff is primarily mediated by a sparse set of five attention heads, not by MLPs or diffuse residual-stream effects.

---

## 8. Why this paper matters mechanistically

The most important contribution is not just “Gemma plans at newline”.

The deeper lesson is:

> Linear separability and causal usage are different things.

A hidden state can contain highly decodable information while being causally irrelevant to the model’s actual output path.

This has direct consequences for mechanistic interpretability:

| Weak claim | Stronger claim |
|---|---|
| “Information is decodable from hidden states.” | “This hidden state causally drives behavior.” |
| “AUROC/probe accuracy is high.” | “Patching this site changes the output in the predicted direction.” |
| “Representations cluster by condition.” | “Intervening on the representation transfers the behavior.” |

For paper-grade mechanism evidence, the paper implicitly argues that probing alone is not enough.

---

## 9. Connection to our hidden-state geometry / Stage 4 Method 4.2

This paper is highly relevant to our current mechanism framing.

Our Stage 4 Method 4.2 uses metrics like:

- cosine gap between mode means
- AUROC along mean-difference direction
- PCA explained variance / representation compactness

These are strong **representation-geometry** metrics.

But this paper gives an important warning:

> Even perfect AUROC does not prove that the representation is causally used.

In their terms:

- AUROC / probe success = planning-compatible representation
- activation patching / behavioral redirection = causally active planning site

So for our own paper, the safest framing is:

```text
Our geometry results show that observation modes induce linearly separable and compact hidden-state regimes.
They establish representation-level evidence, not by themselves causal evidence.
A stronger causal claim would require activation patching, ablation, or mode-transfer intervention showing that swapping the representation shifts the downstream action / answer / routing decision.
```

This is especially relevant if we have an “AUROC = 1.000 across the board” result.

That result is strong, but the paper suggests we should not overclaim:

✅ Good claim:

```text
The model internally distinguishes the observation regimes with near-perfect linear separability.
```

⚠️ Too strong without intervention:

```text
The model uses this hidden-state direction to decide its next action.
```

To make the second claim, we would need causal intervention.

---

## 10. Possible extension for our P79 / VisualWebArena mechanism study

A direct analogue of this paper for our setting would be:

### 10.1 Clean/corrupt observation-mode patching

Construct paired runs:

| Clean run | Corrupt run |
|---|---|
| DOM observation | SOM observation |
| DOM observation | Vision observation |
| SOM observation | Vision observation |
| Successful trajectory | Failed trajectory |
| Low-cost mode | High-cost mode |

Then patch hidden states from corrupt into clean at selected layers/tokens.

Measure whether the next action shifts toward the corrupt run’s behavior.

Possible behavioral targets:

- selected element id
- action type
- coordinate distribution
- finish vs continue
- repeated-click loop vs recovery
- page-unchanged response
- route choice / mode choice

### 10.2 What would count as causal evidence?

A causal intervention would be strong if:

```text
Patching the hidden state from mode B into mode A shifts the model’s next action toward mode B, while controls do not.
```

For example:

```text
DOM clean run chooses wrong element.
SOM corrupt run chooses correct element.
Patch SOM hidden state into DOM run at layer L.
If DOM run now chooses the SOM-like/correct element, that layer-position is causally involved in observation-mode grounding.
```

### 10.3 Controls needed

Following the paper’s logic, we would want:

- zero-vector control
- unrelated-donor-prompt control
- same-mode donor control
- random layer/token patching
- patching after answer/action token as negative control
- bootstrap confidence intervals over tasks, not just samples
- report raw and adjusted success/action-shift rates

---

## 11. Useful language for our paper

### 11.1 Probe-vs-causality framing

```text
Following recent work on latent planning, we distinguish between representation availability and causal use. A feature may be linearly decodable from hidden states without serving as the causal read-out site for generation. We therefore interpret our geometry metrics as evidence of representation-level separation, and reserve stronger causal claims for intervention-based analyses.
```

### 11.2 For Method 4.2

```text
Our cosine-gap and AUROC analyses test whether the model forms separable hidden-state regimes under different observation channels. These metrics are analogous to probing evidence: they show that mode information is present and linearly accessible. They do not, by themselves, establish that the model uses this direction causally to select actions.
```

### 11.3 For future work / limitation

```text
A natural next step is activation patching across observation channels. By substituting hidden states from one observation mode into another and measuring shifts in action selection, we can test whether the identified mode-separating directions are merely diagnostic or causally active in web-agent decision making.
```

### 11.4 For contribution framing

```text
Our results should therefore be read as a representation-level mechanism map rather than a full causal circuit analysis. This distinction strengthens, rather than weakens, the claim: it prevents probe-style evidence from being overinterpreted while identifying precise candidate sites for future causal intervention.
```

---

## 12. Limitations of the paper

The paper is strong, but not unlimited.

### 12.1 Single task family

The task is rhyming couplet generation. This is clean, but narrow.

Open question:

> Does newline planning generalize to prose, code, math, web navigation, or multi-step reasoning?

### 12.2 Synthetic data

The couplets are generated by Claude. This may introduce distributional biases.

The authors acknowledge that probe accuracy could be inflated by dataset artifacts.

### 12.3 Rhyme metric limitations

They use CMU Pronouncing Dictionary. This can miss valid rhymes and unevenly affect rhyme families.

### 12.4 Small number of patching prompt pairs

Main patching uses 5 prompt pairs × 20 samples.

The cluster bootstrap is appropriate, but the confidence intervals are wide. The authors explicitly note that larger and more diverse prompt-pair sets would tighten estimates.

### 12.5 Activation patching is artificial

Patching shows that a site can redirect generation when manipulated, but future work is still needed to test whether the natural model reads the representation in exactly the proposed way.

---

## 13. Practical takeaway

For our work, the biggest takeaway is this:

> Representation geometry is valuable, but it should be framed as “where information is available,” not automatically as “where the model makes the decision.”

So if our Method 4.2 gives:

```text
AUROC = 1.000 across layers/modes
```

the strongest careful interpretation is:

```text
The model’s hidden states encode observation-mode identity in a highly linearly separable way.
```

The next-level causal interpretation would require:

```text
Swapping, ablating, or steering that representation changes the model’s action or outcome.
```

This paper gives us a clean citation and conceptual vocabulary for that distinction:

- **planning-compatible representation**
- **causally active planning site**
- **representational handoff**
- **probe-causality dissociation**
- **sparse attention-head localization**

---

## 14. How I would use this in our current pipeline

### Minimum use

Cite it in the mechanism-evidence section to justify why we separate:

1. geometry/probe-style evidence
2. causal intervention evidence

### Strong use

Add a small future-work subsection:

```text
Causal validation via mode-transfer patching
```

### Best use if we have time

Run one lightweight intervention:

```text
For a small set of matched tasks, patch hidden states between DOM/SOM/Vision runs at selected mid-layers and measure next-action transfer.
```

Even a small pilot would make the mechanism section much more paper-grade.

---

## 15. Bottom line

This is a very useful paper for our framing because it says exactly the thing reviewers often care about:

> Decodability is not causality.

It gives us a disciplined way to describe our own hidden-state geometry results without overclaiming, while also pointing to the natural next intervention that would upgrade the result from representation evidence to causal mechanism evidence.
