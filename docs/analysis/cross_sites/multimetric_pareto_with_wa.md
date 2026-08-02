---
type: analysis
status: complete
created: 2026-08-02
purpose: does the (success, cost) Pareto verdict survive adding latency as a third axis
scope_warning: within-cell only. B0 reports an API bill and B1/B2 an electricity-derived figure, so no quantity here is comparable across backbones.
producer: scripts/analysis/aggregate_multimetric_pareto.py (single-sources per_mode_four_dimension_profile.md)
---

# Multi-metric Pareto

Regenerate: `.venv/bin/python3 scripts/analysis/aggregate_multimetric_pareto.py`

## 1. Is latency an independent axis?

Two claims are separable here and only one needs our data. That a deployment may care about wall-clock rather than tokens is an industry fact, not something to demonstrate. What needs evidence is narrower: **in these runs, is the latency ordering something other than the cost ordering restated?**

| cell | cost span | latency span | cheapest | fastest | same? | ρ(cost, latency) |
|---|---|---|---|---|---|---|
| `B0_classifieds` | 1.12x | 1.18x | Vision | SoM | **no** | -0.600 |
| `B0_reddit` | 1.13x | 1.40x | Vision | Vision | yes | +0.143 |
| `B1_classifieds` | 1.46x | 1.20x | Vision | SoM | **no** | -0.257 |
| `B1_reddit` | 1.53x | 1.35x | Vision | Vision | yes | +0.771 |
| `B2_classifieds` | 1.28x | 1.12x | Vision | SoM | **no** | -0.600 |
| `B2_reddit` | 1.63x | 1.23x | Vision | Vision | yes | -0.029 |
| `B1_wa_reddit` | 1.78x | 1.05x | Vision | DOM | **no** | +0.200 |

Mean **ρ = -0.053** over 7 cells — the two orderings are close to uncorrelated, and on the classifieds cells they run *opposite*: the cheapest mode is the slowest. The cheapest mode is not the fastest in **4 of 7** cells, and those cells are `B0_classifieds`, `B1_classifieds`, `B1_wa_reddit`, `B2_classifieds` — the split follows the **site**, not the backbone.

**Under the canonical latency estimand** (retry, busy-wait and recovered-screenshot subtracted) the mean is ρ = -0.029. ⚠️ That agreement is weaker evidence than it looks: the two estimands are **identical by construction** on 5 of 7 cells (the locally-served ones have no retry, busy-wait or screenshot-timeout to subtract), so only the API-served cells test it at all.

⚠️ Per-cell exact permutation p-values on ρ are not significant (six modes give a Spearman test almost no power), so this is a descriptive structure and not a test. The cross-cell regularity is what carries it: three classifieds cells all put `Vision` cheapest and `SoM` fastest, three reddit cells all put `Vision` at both.

## 2. Why the frontier count is NOT the evidence for §1

An earlier version of this document argued §1 from frontier growth — the frontier widened in 3 of 6 cells when latency was added. **That argument is void**, and the control that kills it is exact rather than approximate. Adding an axis can only weakly enlarge a Pareto frontier, and six modes give five chances for a dominated mode to escape. Permuting latency across the modes within each cell (all 720 assignments) gives:

| cell | frontier widened? | P(widen) under permuted latency |
|---|---|---|
| `B0_classifieds` | yes | 0.800 |
| `B0_reddit` | yes | 0.750 |
| `B1_classifieds` | no | 0.792 |
| `B1_reddit` | no | 0.750 |
| `B2_classifieds` | yes | 0.833 |
| `B2_reddit` | no | 0.778 |
| `B1_wa_reddit` | yes | 0.800 |

Expected widened cells under the null: **5.50 of 7**. Observed: **4**. P(at least 4 widen | null) = **0.958**. The observed count is *below* chance, so frontier growth carries no information about whether latency is independent. It is reported here only so nobody reconstructs the retracted argument.

## 3. What the frontier count still legitimately says

| cell | success x cost | + latency | + tokens |
|---|---|---|---|
| `B0_classifieds` | SoM, Vision (2) | **SoM, Vision, P-prompt (3)** | 4 |
| `B0_reddit` | DOM, SoM, Vision (3) | **DOM, SoM, Vision, P-prompt (4)** | 4 |
| `B1_classifieds` | SoM, Vision (2) | **SoM, Vision (2)** | 2 |
| `B1_reddit` | SoM, Vision, P-text (3) | **SoM, Vision, P-text (3)** | 3 |
| `B2_classifieds` | Vision (1) | **DOM, SoM, Vision, P-text, P-prompt (5)** | 5 |
| `B2_reddit` | DOM, Vision (2) | **DOM, Vision (2)** | 2 |
| `B1_wa_reddit` | Vision, P-text (2) | **DOM, Vision, P-text (3)** | 3 |

Read as *width*, not as evidence: Pareto dominance is strictly harder to achieve against three axes, so §5.3's negative result (no learned policy dominates a fixed one) holds a fortiori. Non-dominance becomes correspondingly cheaper to satisfy, so wherever the paper treats non-dominance as informative it must be read against a frontier this wide.

Adding tokens enlarges the frontier further in **1** cell(s). The bill is computed from tokens, so this column is a consistency check rather than an axis — but the earlier claim that it 'changes nothing beyond latency' was false against this producer's own table.
