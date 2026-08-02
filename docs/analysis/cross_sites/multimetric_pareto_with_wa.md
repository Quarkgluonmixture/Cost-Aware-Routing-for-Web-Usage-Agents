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

| cell | cost span | latency span | cheapest mode | fastest mode | same? |
|---|---|---|---|---|---|
| `B0_classifieds` | 1.12x | 1.18x | Vision | SoM | **no** |
| `B0_reddit` | 1.13x | 1.40x | Vision | Vision | yes |
| `B1_classifieds` | 1.46x | 1.20x | Vision | SoM | **no** |
| `B1_reddit` | 1.53x | 1.35x | Vision | Vision | yes |
| `B2_classifieds` | 1.28x | 1.12x | Vision | SoM | **no** |
| `B2_reddit` | 1.63x | 1.23x | Vision | Vision | yes |
| `B1_wa_reddit` | 1.78x | 1.05x | Vision | DOM | **no** |

Latency spans the same order of magnitude as cost, and in **4 of 7** cells the cheapest mode is not the fastest. It is a second axis, not a restatement of the first.

## 2. What adding it does to the frontier

| cell | success x cost | + latency | + tokens |
|---|---|---|---|
| `B0_classifieds` | SoM, Vision (2) | **SoM, Vision, P-prompt (3)** | 4 |
| `B0_reddit` | DOM, SoM, Vision (3) | **DOM, SoM, Vision, P-prompt (4)** | 4 |
| `B1_classifieds` | SoM, Vision (2) | **SoM, Vision (2)** | 2 |
| `B1_reddit` | SoM, Vision, P-text (3) | **SoM, Vision, P-text (3)** | 3 |
| `B2_classifieds` | Vision (1) | **DOM, SoM, Vision, P-text, P-prompt (5)** | 5 |
| `B2_reddit` | DOM, Vision (2) | **DOM, Vision (2)** | 2 |
| `B1_wa_reddit` | Vision, P-text (2) | **DOM, Vision, P-text (3)** | 3 |

The frontier grows in **4 of 7** cells. This cuts both ways and the paper should say so. Pareto *dominance* becomes strictly harder to achieve with a third axis, so §5.3's negative result (no learned policy dominates a fixed one) holds a fortiori. But non-dominance becomes correspondingly cheaper to satisfy, so wherever the paper reports non-dominance as informative it should be read against a frontier this wide.

Adding tokens changes nothing beyond what latency already changed, which is expected: the bill is computed from tokens, so that column is a check rather than an axis.
