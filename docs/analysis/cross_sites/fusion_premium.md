---
type: analysis
status: complete
created: 2026-08-02
purpose: turn §4.2's fusion-premium claim from a count over cells into a paired test against a priori comparators
post_hoc_exploratory: true
scope_warning: not the pre-registered H1; not gated. The rerun band it is read against (0.89-2.23pp, mean-difference scale) is measured on two conditions and extrapolated to the rest.
producer: scripts/analysis/aggregate_fusion_premium.py
---

# Does the fused mode earn its premium?

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_fusion_premium.py`

Effects are `SR(SoM) - SR(comparator)` in points, against **a priori fixed** comparators. An earlier version compared against the per-cell maximum of two alternatives; a maximum over noisy quantities biases the comparator up and so biased the fusion advantage down, in the direction that favoured the claim being made. Intervals are paired bootstrap over tasks, 10,000 resamples, seed 20260802.

## 1. Per cell

| cell | n | SoM − Vision | 95% CI | SoM − DOM | 95% CI |
|---|---|---|---|---|---|
| `cls_B0` | 224 | +2.23pp | [-2.23, +7.14] | +9.82pp | [+4.02, +16.07] |
| `cls_B1` | 224 | +1.79pp | [-2.23, +5.80] | +8.04pp | [+3.57, +12.50] |
| `cls_B2` | 224 | +0.00pp | [-2.68, +2.68] | +0.89pp | [-1.34, +3.12] |
| `red_B0` | 203 | +7.39pp | [+2.46, +12.81] | +0.49pp | [-3.94, +4.93] |
| `red_B1` | 203 | +4.93pp | [+1.48, +8.87] | +1.48pp | [-1.48, +4.43] |
| `red_B2` | 203 | -0.99pp | [-3.45, +1.48] | -2.96pp | [-5.91, -0.49] |
| `wa_red_B1` | 104 | +3.85pp | [-0.96, +9.62] | -2.88pp | [-8.65, +2.88] |

## 2. Fixed-effect pool

| comparator | k | pooled θ | 95% CI | clears 0? | clears the rerun band? |
|---|---|---|---|---|---|
| SoM − vision | 7 | **+1.43pp** | [+0.12, +2.75] | **yes** | no |
| SoM − dom | 7 | **+0.89pp** | [-0.40, +2.18] | no | no |

The band is the measured run-to-run mean-difference floor, 0.89 to 2.23pp. Reading the pooled estimate against it rather than against zero is the point: a premium has to beat what repetition delivers for the same money, not merely beat nothing.

## 3. Fusion against the channel that suits the workload

The two columns read together show something neither shows alone. In every cell one of the two single channels is the stronger, and it is the visual one on all three classifieds cells and the text one on all four reddit splits. Against **that** channel, fusion's interval includes zero everywhere but one, where it is significantly negative.

| cell | stronger single channel | SoM - that channel | 95% CI | excludes 0? |
|---|---|---|---|---|
| `cls_B0` | vision | +2.23pp | [-2.23, +7.14] | no |
| `cls_B1` | vision | +1.79pp | [-2.23, +5.80] | no |
| `cls_B2` | vision | +0.00pp | [-2.68, +2.68] | no |
| `red_B0` | dom | +0.49pp | [-3.94, +4.93] | no |
| `red_B1` | dom | +1.48pp | [-1.48, +4.43] | no |
| `red_B2` | dom | -2.96pp | [-5.91, -0.49] | **yes, negative** |
| `wa_red_B1` | dom | -2.88pp | [-8.65, +2.88] | no |

**6 of 7** intervals include zero and the remaining one is negative, so in no cell does fusion significantly beat the channel that suits the workload. It beats the channel that does not, by +8.04 and +9.82 points over DOM on classifieds and +4.93 and +7.39 over Vision on reddit. Which channel is stronger is read off each cell and is therefore post hoc, which is why both full columns appear in §1 and the pooled tests in §2 use comparators fixed in advance.

## 4. What this does and does not settle

It replaces a count over cells with an interval, and it removes the selection bias in the comparator. It does **not** supply a rerun floor for the fused mode itself, which is measured on DOM and Vision only; a same-condition SoM replicate is the experiment that would close that, and it is queued rather than done.
