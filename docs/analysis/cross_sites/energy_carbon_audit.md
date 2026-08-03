---
type: analysis
status: complete
purpose: measure whether the energy/carbon column carries information beyond elapsed time
post_hoc_exploratory: true
producer: scripts/analysis/energy_carbon_audit.py
---

# Is carbon an axis?

Regenerate: `.venv/bin/python3 scripts/analysis/energy_carbon_audit.py`

`EVIDENCE_LAYER_SUMMARY` §4a has always said the energy column is "wall-clock in other units". It is — and until 2026-08-03 nobody had measured it. An unmeasured limitation is still a claim.

## 1. Energy against elapsed time

Over **24 conditions**, per-step energy correlates with per-step latency at **r = 0.9659 to 0.9998** (mean **0.9935**). Recorded power is **66.3 W** with a cross-condition SD of 0.40 W.

| cell | mode | steps | r(energy, latency) | mean W | SD W |
|---|---|---|---|---|---|
| `B1·classifieds` | DOM | 487 | 0.9945 | 65.7 | 1.80 |
| `B1·classifieds` | SoM | 342 | 0.9931 | 66.3 | 1.76 |
| `B1·classifieds` | Vision | 421 | 0.9659 | 66.2 | 1.90 |
| `B1·classifieds` | P-text | 520 | 0.9926 | 65.7 | 1.89 |
| `B1·classifieds` | P-prompt | 465 | 0.9882 | 66.0 | 1.90 |
| `B1·classifieds` | P-SoM | 466 | 0.9993 | 66.1 | 1.80 |
| `B1·reddit` | DOM | 584 | 0.9998 | 65.9 | 1.78 |
| `B1·reddit` | SoM | 588 | 0.9997 | 66.5 | 2.77 |
| `B1·reddit` | Vision | 634 | 0.9959 | 66.4 | 2.31 |
| `B1·reddit` | P-text | 604 | 0.9998 | 65.7 | 1.78 |
| `B1·reddit` | P-prompt | 561 | 0.9996 | 66.1 | 2.48 |
| `B1·reddit` | P-SoM | 630 | 0.9997 | 66.0 | 2.32 |
| `B2·classifieds` | DOM | 645 | 0.9917 | 66.3 | 1.85 |
| `B2·classifieds` | SoM | 666 | 0.9890 | 67.0 | 2.17 |
| `B2·classifieds` | Vision | 687 | 0.9862 | 67.0 | 1.94 |
| `B2·classifieds` | P-text | 622 | 0.9919 | 66.3 | 1.89 |
| `B2·classifieds` | P-prompt | 631 | 0.9902 | 66.3 | 1.90 |
| `B2·classifieds` | P-SoM | 718 | 0.9918 | 66.1 | 1.86 |
| `B2·reddit` | DOM | 685 | 0.9996 | 66.3 | 2.10 |
| `B2·reddit` | SoM | 683 | 0.9954 | 66.9 | 2.48 |
| `B2·reddit` | Vision | 679 | 0.9998 | 66.9 | 2.46 |
| `B2·reddit` | P-text | 696 | 0.9835 | 66.9 | 6.89 |
| `B2·reddit` | P-prompt | 690 | 0.9981 | 66.3 | 2.00 |
| `B2·reddit` | P-SoM | 636 | 0.9990 | 66.0 | 1.75 |

A near-constant wattage times elapsed time **is** elapsed time. The column should be reported as uninformative, never as a third axis beside cost and latency, and no mode-ordering may be read off it that is not already the latency ordering.

## 2. Two things the standing limitation does not say

**The configuration asked for the GPU counter and did not get it.** `configs/exp_v2_base.yaml:95` sets `use_pynvml: true`; every step records `source: psutil_profile`. That is a CPU-package estimate, which is why the number sits at ~66 W on an accelerator rated several times that. Nothing failed loudly because no product read the field — the same shape as `confidence.verbalized` and `latency_ms.backend_infer`.

**B0 has no energy data at all.** Per-backbone `energy.source` counts:

| backbone | source values |
|---|---|
| B0 | `disabled` × 5,740 |
| B1 | `psutil_profile` × 6,302 |
| B2 | `psutil_profile` × 8,038 |

B0 is served through an API, so there is no local draw to measure and `disabled` is the correct setting. The consequence still has to be stated: **energy exists for B1, B2 only**, and the backbone it is missing on is the strongest one and the one every headline number is computed on. A carbon comparison across the three backbones cannot be made at all — not merely 'uncalibrated', but absent.

## 3. What would make it an axis

A per-accelerator counter (`pynvml` actually engaged, or a wall meter) on the locally-served arms, plus a defensible figure for the API-served arm's remote draw — which the provider does not publish. The second half is not obtainable, so the honest position is that this project cannot report carbon as a comparable quantity across its backbones, and reporting it per-backbone adds nothing over reporting latency.
