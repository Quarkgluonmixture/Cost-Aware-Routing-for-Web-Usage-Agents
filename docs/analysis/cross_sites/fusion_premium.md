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
| `cls_B0` | 224 | +2.23pp | [-2.68, +7.59] | +9.82pp | [+3.57, +16.07] |
| `cls_B1` | 224 | +1.79pp | [-2.68, +6.25] | +8.04pp | [+3.57, +12.95] |
| `cls_B2` | 224 | +0.00pp | [-2.68, +2.68] | +0.89pp | [-1.34, +3.57] |
| `red_B0` | 203 | +7.39pp | [+2.46, +12.32] | +0.49pp | [-3.94, +4.93] |
| `red_B1` | 203 | +4.93pp | [+1.48, +8.87] | +1.48pp | [-1.48, +4.43] |
| `red_B2` | 203 | -0.99pp | [-3.45, +1.48] | -2.96pp | [-5.91, -0.49] |
| `wa_red_B1` | 104 | +3.85pp | [-1.92, +9.62] | -2.88pp | [-9.62, +2.88] |

## 2. Fixed-effect pool

| comparator | k | pooled θ | 95% CI | clears 0? | clears the rerun band? |
|---|---|---|---|---|---|
| SoM − vision | 7 | **+1.44pp** | [-0.01, +2.91] | no | no |
| SoM − dom | 7 | **+0.84pp** | [-0.49, +2.20] | no | no |

The band is the measured run-to-run mean-difference floor, 0.89 to 2.23pp. Reading the pooled estimate against it rather than against zero is the point: a premium has to beat what repetition delivers for the same money, not merely beat nothing.

**The interval above is the task-clustered one, and that choice changes an answer.** Within a site the three backbones are scored on the same task universe, so their effects share sampling noise; the textbook `sqrt(1/Σw)` treats them as independent and understates the pooled SE. Resampling tasks once per site and evaluating every backbone in that site on the same draw gives:

| comparator | independent-cells CI | task-clustered CI | SE |
|---|---|---|---|
| SoM − vision | [+0.09, +2.80] | **[-0.01, +2.91]** | 0.693 → 0.741 |
| SoM − dom | [-0.48, +2.17] | **[-0.49, +2.20]** | 0.674 → 0.688 |

The one interval that excluded zero (SoM − Vision) no longer does. (codex Mode B, §H stress 2026-08-02; its predicted clustered SE of 0.741 matched.)

⚠️ **And a fixed-effect pool is the wrong estimand here regardless.** Cochran's Q rejects a common effect for both comparators:

| comparator | Q | df | p | I² |
|---|---|---|---|---|
| SoM − vision | 14.72 | 6 | 2.26e-02 | 59% |
| SoM − dom | 26.23 | 6 | 2.02e-04 | 77% |

With I² at this level the pooled number describes no cell in particular. It is kept because the pre-registration names FE as the primary machinery, but the per-cell table in §1 and the workload split in §3 are what carry the finding — and the sign change across workloads in §3 is itself why a common-effect model cannot hold.

## 3. Fusion against the channel that suits the workload

The two columns read together show something neither shows alone. In every cell one of the two single channels is the stronger, and it is the visual one on all three classifieds cells and the text one on all four reddit splits. Against **that** channel, fusion's interval includes zero everywhere but one, where it is significantly negative.

| cell | stronger single channel | SoM - that channel | 95% CI | excludes 0? |
|---|---|---|---|---|
| `cls_B0` | vision | +2.23pp | [-2.68, +7.59] | no |
| `cls_B1` | vision | +1.79pp | [-2.68, +6.25] | no |
| `cls_B2` | vision | +0.00pp | [-2.68, +2.68] | no |
| `red_B0` | dom | +0.49pp | [-3.94, +4.93] | no |
| `red_B1` | dom | +1.48pp | [-1.48, +4.43] | no |
| `red_B2` | dom | -2.96pp | [-5.91, -0.49] | **yes, negative** |
| `wa_red_B1` | dom | -2.88pp | [-9.62, +2.88] | no |

**6 of 7** intervals include zero and the remaining one is negative, so in no cell does fusion beat the channel that suits the workload. ⚠️ The word *significantly* does not belong on that sentence and has been removed: the comparator is chosen per cell using the same observed success rates the interval is computed from, so these CIs do not retain nominal coverage. Restoring coverage needs either a site→channel mapping fixed in advance, or a bootstrap that re-selects the comparator inside every resample. Until then this row is descriptive. (§H stress P1-2.) It beats the channel that does not, by +8.04 and +9.82 points over DOM on classifieds and +4.93 and +7.39 over Vision on reddit. Which channel is stronger is read off each cell and is therefore post hoc, which is why both full columns appear in §1 and the pooled tests in §2 use comparators fixed in advance.

## 4. What this does and does not settle

It replaces a count over cells with an interval, and it removes the selection bias in the comparator. It does **not** supply a rerun floor for the fused mode itself, which is measured on DOM and Vision only; a same-condition SoM replicate is the experiment that would close that, and it is queued rather than done.
