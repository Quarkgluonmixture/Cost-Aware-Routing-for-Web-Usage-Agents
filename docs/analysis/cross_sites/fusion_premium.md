---
type: analysis
status: complete
created: 2026-08-02
purpose: turn §4.2's fusion-premium claim from a count over cells into a paired test against a priori comparators
post_hoc_exploratory: true
scope_warning: not the pre-registered H1; not gated. The rerun band it is read against (0.0-3.45pp, mean-difference scale) is measured on two conditions and extrapolated to the rest.
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
| `wa_red_B0` | 104 | +2.88pp | [-5.77, +11.54] | -4.81pp | [-12.50, +2.88] |

## 2. Fixed-effect pool

| comparator | k | pooled θ | 95% CI | clears 0? | clears the observed band? | clears the rerun **null**? |
|---|---|---|---|---|---|---|
| SoM − vision | 8 | **+1.48pp** | [+0.06, +2.93] | **yes** | no | no |
| SoM − dom | 8 | **+0.70pp** | [-0.62, +2.04] | no | no | no |

The band is the measured run-to-run mean-difference floor, 0.0 to 3.45pp. Reading the pooled estimate against it rather than against zero is the point: a premium has to beat what repetition delivers for the same money, not merely beat nothing.

⚠️ **Two bands, and the last column is the one that answers the question.** `0.0–3.45pp` is what 13 reruns *happened to* deliver — two draws from a random quantity, not a bound on it. That quantity's own spread is computable from the same pairs' discordant counts: `SD(ΔSR) = √d/n` gives **0.0–2.53pp**, i.e. the band's upper edge is about one standard deviation. An effect only becomes unlikely for a single rerun to manufacture at roughly **0.0–4.15pp** (one-sided 95%). Both are reported; nothing here should be read as clearing noise on the strength of the observed band alone. → `noise_floor_inventory` §1b.

**The interval above is the task-clustered one, and that choice changes an answer.** Within a site the three backbones are scored on the same task universe, so their effects share sampling noise; the textbook `sqrt(1/Σw)` treats them as independent and understates the pooled SE. Resampling tasks once per site and evaluating every backbone in that site on the same draw gives:

| comparator | independent-cells CI | task-clustered CI | SE |
|---|---|---|---|
| SoM − vision | [+0.14, +2.82] | **[+0.06, +2.93]** | 0.683 → 0.733 |
| SoM − dom | [-0.61, +2.00] | **[-0.62, +2.04]** | 0.665 → 0.678 |

Clustering widens every interval without changing a verdict: `SoM − vision` still excludes zero (lower bound +0.06pp). ⚠️ Read that against the band, not against zero: a lower bound of +0.06pp sits below the rerun band's floor of 0.0pp, so excluding zero here is **not** a premium claim — the `clears the rerun band?` column above is the one that answers the question.
(codex Mode B, §H stress 2026-08-02; its predicted clustered SE of 0.741 matched.)

⚠️ **And a fixed-effect pool is the wrong estimand here regardless.** Cochran's Q rejects a common effect for both comparators:

| comparator | Q | df | p | I² |
|---|---|---|---|---|
| SoM − vision | 14.83 | 7 | 3.82e-02 | 53% |
| SoM − dom | 28.05 | 7 | 2.15e-04 | 75% |

With I² at this level the pooled number describes no cell in particular. It is kept because the pre-registration names FE as the primary machinery, but the per-cell table in §1 and the workload split in §3 are what carry the finding — and the sign change across workloads in §3 is itself why a common-effect model cannot hold.

## 3. Fusion against the channel that suits the workload

The two columns read together show something neither shows alone. In every cell one of the two single channels is the stronger, and it is the visual one on all three classifieds cells and the text one on all four reddit splits. Against **that** channel, fusion's interval includes zero everywhere but one, where it is significantly negative.

| cell | stronger single channel | SoM - that channel | 95% CI | excludes 0? | \|effect\| > rerun null? |
|---|---|---|---|---|---|
| `cls_B0` | vision | +2.23pp | [-2.68, +7.59] | no | no |
| `cls_B1` | vision | +1.79pp | [-2.68, +6.25] | no | no |
| `cls_B2` | vision | +0.00pp | [-2.68, +2.68] | no | no |
| `red_B0` | dom | +0.49pp | [-3.94, +4.93] | no | no |
| `red_B1` | dom | +1.48pp | [-1.48, +4.43] | no | no |
| `red_B2` | dom | -2.96pp | [-5.91, -0.49] | **yes, negative** | no |
| `wa_red_B1` | dom | -2.88pp | [-9.62, +2.88] | no | no |
| `wa_red_B0` | dom | -4.81pp | [-12.50, +2.88] | no | yes |

**7 of 8** intervals include zero and the remaining one is negative, so in no cell does fusion beat the channel that suits the workload. ⚠️ The word *significantly* does not belong on that sentence and has been removed: the comparator is chosen per cell using the same observed success rates the interval is computed from, so these CIs do not retain nominal coverage. Restoring coverage needs either a site→channel mapping fixed in advance, or a bootstrap that re-selects the comparator inside every resample. Until then this row is descriptive. (§H stress P1-2.) Against the channel that does *not* suit the workload it does better — but the full set must be quoted, not its top end:

- `SoM − dom` over DOM, where the visual channel is stronger: cls_B2 +0.89pp, cls_B1 +8.04pp, cls_B0 +9.82pp  → range **+0.89 to +9.82pp**, 2 of 3 above the rerun null.
- `SoM − vision` over Vision, where the text channel is stronger: red_B2 -0.99pp, wa_red_B0 +2.88pp, wa_red_B1 +3.85pp, red_B1 +4.93pp, red_B0 +7.39pp  → range **-0.99 to +7.39pp**, 2 of 5 above the rerun null.

⚠️ **Derived from the table above, not typed.** Until 2026-08-03 this paragraph hardcoded four of these numbers and named only the two largest on each side — the source of the `4.93-7.39pp` string quoted downstream. The dropped cells are not cosmetic: on the text-stronger side the smallest is **negative**, so a range starting at +4.93 is a subrange with the sign change removed. Which channel is stronger is read off each cell and is therefore post hoc, which is why both full columns appear in §1 and the pooled tests in §2 use comparators fixed in advance.

## 4. What this does and does not settle

It replaces a count over cells with an interval, and it removes the selection bias in the comparator. It does **not** supply a rerun floor for the fused mode itself, which is measured on DOM and Vision only; a same-condition SoM replicate is the experiment that would close that, and it is queued rather than done.
