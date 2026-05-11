# Paper Reading Note — *Inference Time Causal Probing in LLMs (HDMI)*

**Source:** Khorasani, Salehkaleybar, Kiyavash, Grossglauser, *Inference Time Causal Probing in LLMs*, arXiv:2605.07631, 2026-05-08.
**Use case for our project:** Method 4.4 alternative method + paper §5 evaluation framework (completeness × selectivity → harmonic mean).

---

## 0. One-sentence summary

HDMI replaces the standard probe-classifier-then-add-direction recipe with a **probe-free gradient-based hidden-state steering** that directly maximizes a margin between target and source token probabilities using the model's native output. The paper also proposes a clean reliability metric — harmonic mean of completeness × selectivity — that is exactly what we should use to report our Method 4.4 v2 results.

---

## 1. Core research problem

Standard causal probing pipeline:

1. Train a linear probe to decode property `P` from hidden state `h`.
2. Add the probe weight vector (or mean-difference direction between classes) to `h` during inference.
3. Hope this shifts the model's output toward target class.

This approach has two failure modes:

- **Probe-output misalignment**: the probe learns a direction that is decodable but not actually the direction the model reads from. (= Ma & Rui 2026 probe-vs-causal dissociation.)
- **Class-set dependence**: probe is tied to a specific class set; new tasks require retraining.

HDMI is the probe-free alternative.

---

## 2. HDMI method

### 2.1 Standard mean-difference (Wu et al. 2026 family)

```
h' = h + α·(mean(h | class=target) − mean(h | class=source))
```

This is what we did in our Method 4.4 v1 + v2 sweep.

### 2.2 HDMI

Instead of pre-computing a fixed direction from class means, HDMI directly optimizes the hidden state per-input via gradient descent on a margin objective:

```
L(h) = log P(target_token | h) − log P(source_token | h)
h* = argmax_h L(h)   subject to ||h* − h||_2 ≤ ε
```

The optimization runs **inference-time per query**, not pretrained. No probe classifier involved.

### 2.3 LA-HDMI (Lookahead variant)

For multi-token continuations (relevant to our web agent JSON action generation), LA-HDMI:

- Backpropagates through the softmax embeddings of the generated continuation
- Modifies the current hidden state so that **user-specified target tokens become more likely in subsequent next-token generations**
- Preserves fluency by including a fluency-regularization term

This is the **direct analogue to what we tried in Method 4.4 v2** — except v2 uses fixed mean-difference direction, while LA-HDMI uses per-input gradient.

---

## 3. Evaluation framework — **the key contribution for our paper §5**

HDMI introduces a 2-axis reliability evaluation:

| Axis | Meaning | How to measure |
|---|---|---|
| **Completeness** | Does the target property change as intended? | Fraction of inputs where output shifts toward target |
| **Selectivity** | Do unrelated properties remain unchanged? | Fraction of inputs where output preserves non-target structure |

The overall reliability is the **harmonic mean**:

```
reliability = 2 · completeness · selectivity / (completeness + selectivity)
```

This **automatically penalizes the "shift target but break everything else" failure mode** — which is exactly what our Method 4.4 v2 sweep at L33 α=10 hit (50% shift toward P-SoM but only 25% JSON envelope valid).

---

## 4. Direct mapping onto our Method 4.4 v2 results

We re-cast our smoke results in HDMI metrics:

| Layer | α | Completeness (shift toward P-SoM) | Selectivity (JSON envelope valid) | Reliability (H-mean) |
|---|---|---|---|---|
| L11 | 5 | 0.25 | 1.00 | **0.40** |
| L17 | 5 | 0.25 | 1.00 | **0.40** |
| L23 | 5 | 0.00 | 1.00 | 0.00 |
| L29 | 5 | 0.00 | 1.00 | 0.00 |
| L33 | 10 | **0.50** | 0.25 | **0.33** |
| L34 | 10 | 0.25 | 0.00 | 0.00 |

**The "best" cell is not L33 α=10 (highest shift) but L11/L17 α=5 (H-mean 0.40, equal best)** — they shift moderately but preserve JSON structure. This is more paper-grade than our prior "50% shift at L33" headline.

Key insight: mid-layer mean-diff steering has **selectivity advantage** (output stays structured), late-layer has **completeness advantage** (output diverges). Wu et al.'s tool-calling 93% switch is at single-token decision where selectivity is trivially 1.0 (only 1 token differs). Our multi-step JSON has 15-token continuation where selectivity matters.

---

## 5. Comparison HDMI vs. mean-difference vs. our Method 4.4

| | Wu et al. 2026 (mean-diff) | HDMI (gradient margin) | Our Method 4.4 v2 |
|---|---|---|---|
| Pre-compute direction | Yes (from class means) | No (per-input optimization) | Yes (from class means) |
| Probe required | No (mean-diff) | No (direct margin) | No (mean-diff) |
| Per-query cost | 1 forward pass | k iterations of forward+backward | 1 forward pass |
| Generalization | Limited to fixed class set | Per-query, any target | Limited to fixed class set |
| LA variant for multi-token | Not in scope | LA-HDMI explicit | Not in scope |
| Tested models | 12 (Gemma, Qwen, Llama 270M-27B) | Llama-3-8B + Pythia-70M | Qwen3-VL-4B |
| Domain | Tool calling | LGD agreement, CausalGym | Web agent observation modes |

**Important**: HDMI is NOT tested on Qwen3 family. Cross-model generalization gap (Ma & Rui's Qwen3 1% rhyme newline causal weakness) may also affect HDMI on Qwen3-VL-4B.

---

## 6. Implications for our paper §5

### 6.1 Rewrite Method 4.4 v2 reporting using H-mean

Our current sweep table reports shift rate + JSON valid rate as separate columns. **Replace with reliability column** (harmonic mean). This is the paper-grade headline number.

### 6.2 Future work — Method 4.5 (LA-HDMI port)

If Method 4.4 v2 mean-diff steering reliability plateaus at ~0.4 H-mean, the natural next experiment is:

- **Method 4.5**: LA-HDMI port to Qwen3-VL-4B web agent
- Per-task gradient optimization on hidden state at L17 (or sweep)
- Target = first 5 tokens of P-SoM baseline continuation
- Selectivity = JSON envelope + step-action-name structure preservation

Expected outcome: HDMI may outperform mean-diff on Qwen3 family (Ma & Rui's caveat about Qwen3's probe-causal weakness applies to mean-diff but not necessarily to HDMI's direct gradient).

### 6.3 Citation usage

- **§5 Method**: Cite Khorasani et al. 2026 as "probe-free alternative to Wu et al. 2026's mean-difference steering"
- **§5 Evaluation**: Cite their completeness × selectivity → harmonic mean framework as the metric we adopt
- **§5 Limitations**: Acknowledge HDMI not tested on Qwen3 family, our Method 4.4 v2 reliability ceiling may also apply to LA-HDMI extension

---

## 7. Limitations of HDMI

- **No Qwen3 family validation**: only Llama-3-8B + Pythia-70M. The Ma & Rui Qwen3-rhyme-causal-1% finding suggests Qwen3 may not respond to gradient-based steering either.
- **Per-query optimization cost**: k iterations of forward+backward vs Wu et al.'s 1 forward. May be prohibitive for paper-scale ablations.
- **LA-HDMI fluency regularization knob**: paper does not give a recipe for setting it; cross-domain transfer may require tuning.
- **CausalGym + LGD benchmarks are linguistic agreement**: NOT multi-step JSON action generation. Transfer to web agent setting is open question.

---

## 8. Practical takeaway for our pipeline

| Action | Status |
|---|---|
| Adopt H-mean reliability metric in Method 4.4 v2 reporting | TODO (rewrite `stage4_method44_v2_results.md` aggregator) |
| Cite HDMI in paper §5 method + evaluation framework | TODO (after v2 full sweep completes) |
| Implement Method 4.5 (LA-HDMI port) | Future work, paper §8 |
| Compare HDMI vs. mean-diff on subset of 4 tasks | Optional, paper §5 supplement |
