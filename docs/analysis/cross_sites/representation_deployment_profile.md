---
type: analysis
status: complete
purpose: two deployment properties of an observation representation that success rate does not show -- whether its failures are diagnosable, and what its token tail costs
scope_warning: diagnosability is defined against THIS P-rule ruleset and is a floor, not a law -- a richer ruleset moves failures out of the generic bucket. Cross-mode comparison is valid only because every mode is scored by the same ruleset. The token tail is B0-only by default: per-step counts from a hosted endpoint are the provider's accounting, and B1/B2 count locally.
producer: scripts/analysis/representation_deployment_profile.py
---

# Deployment profile of a representation

Regenerate: `.venv/bin/python3 scripts/analysis/representation_deployment_profile.py`

## 1. When it fails, can you tell why?

A failure in a named bucket (committed early, search loop, misgrounded element, missing context) is triage an on-call engineer can act on. `max-steps-other` ("budget ran out, no rule fired") and `error/noise` name no mechanism. Diagnosability below is the share of failures that name one.

| cell | mode | failures | named mechanism | undiagnosed | **diagnosable** | biggest bucket |
|---|---|---:|---:|---:|---:|---|
| `B0/classifieds` | P-prompt | 180 | 167 | 13 | **92.8%** | early-finish/wrong-commit |
| `B0/classifieds` | P-text | 189 | 175 | 14 | **92.6%** | early-finish/wrong-commit |
| `B0/classifieds` | DOM | 185 | 171 | 14 | **92.4%** | early-finish/wrong-commit |
| `B0/classifieds` | SoM | 163 | 145 | 18 | **89.0%** | early-finish/wrong-commit |
| `B0/classifieds` | P-SoM | 189 | 165 | 24 | **87.3%** | early-finish/wrong-commit |
| `B0/classifieds` | Vision | 168 | 129 | 39 | **76.8%** | early-finish/wrong-commit |
| `B0/reddit` | P-prompt | 179 | 145 | 34 | **81.0%** | early-finish/wrong-commit |
| `B0/reddit` | DOM | 175 | 133 | 42 | **76.0%** | early-finish/wrong-commit |
| `B0/reddit` | P-text | 177 | 115 | 62 | **65.0%** | max-steps-other |
| `B0/reddit` | SoM | 175 | 113 | 62 | **64.6%** | early-finish/wrong-commit |
| `B0/reddit` | Vision | 189 | 88 | 101 | **46.6%** | max-steps-other |
| `B1/classifieds` | DOM | 210 | 187 | 23 | **89.0%** | search-loop |
| `B1/classifieds` | P-text | 207 | 183 | 24 | **88.4%** | search-loop |
| `B1/classifieds` | SoM | 192 | 165 | 27 | **85.9%** | early-finish/wrong-commit |
| `B1/classifieds` | P-SoM | 209 | 170 | 39 | **81.3%** | early-finish/wrong-commit |
| `B1/classifieds` | P-prompt | 209 | 169 | 40 | **80.9%** | early-finish/wrong-commit |
| `B1/classifieds` | Vision | 196 | 87 | 109 | **44.4%** | max-steps-other |
| `B1/reddit` | DOM | 191 | 148 | 43 | **77.5%** | search-loop |
| `B1/reddit` | P-prompt | 192 | 135 | 57 | **70.3%** | early-finish/wrong-commit |
| `B1/reddit` | P-text | 191 | 131 | 60 | **68.6%** | search-loop |
| `B1/reddit` | P-SoM | 191 | 127 | 64 | **66.5%** | search-loop |
| `B1/reddit` | SoM | 188 | 113 | 75 | **60.1%** | max-steps-other |
| `B1/reddit` | Vision | 199 | 75 | 124 | **37.7%** | max-steps-other |
| `B2/classifieds` | P-text | 223 | 136 | 87 | **61.0%** | search-loop |
| `B2/classifieds` | SoM | 219 | 127 | 92 | **58.0%** | max-steps-other |
| `B2/classifieds` | DOM | 221 | 120 | 101 | **54.3%** | max-steps-other |
| `B2/classifieds` | Vision | 219 | 103 | 116 | **47.0%** | max-steps-other |
| `B2/classifieds` | P-SoM | 222 | 89 | 133 | **40.1%** | max-steps-other |
| `B2/classifieds` | P-prompt | 220 | 80 | 140 | **36.4%** | max-steps-other |
| `B2/reddit` | SoM | 202 | 76 | 126 | **37.6%** | max-steps-other |
| `B2/reddit` | P-SoM | 202 | 72 | 130 | **35.6%** | max-steps-other |
| `B2/reddit` | P-prompt | 204 | 70 | 134 | **34.3%** | max-steps-other |
| `B2/reddit` | Vision | 200 | 65 | 135 | **32.5%** | max-steps-other |
| `B2/reddit` | P-text | 200 | 64 | 136 | **32.0%** | max-steps-other |
| `B2/reddit` | DOM | 197 | 56 | 141 | **28.4%** | max-steps-other |

Within a cell the spread across modes reaches 44.7 points, and across the whole table it runs from **28.4%** (`B2/reddit` DOM) to **92.8%** (`B0/classifieds` P-prompt). Two representations with the same success rate can therefore differ substantially in how much of their failure mass is actionable -- a cost that lands on the operator, not on the benchmark scoreboard.

## 2. What does the tail cost, not the mean?

Per-step `tokens.input` as the provider counted it. Context-window fit and the decision to truncate are set by the tail, and §449.1 already showed that episode cost is driven by step count rather than per-step volume -- so the per-step tail is an independent quantity, not a restatement of price.

| cell | mode | runs | steps | p50 | p95 | p99 | max | p99/p50 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `B0/classifieds` | P-prompt | 1 | 3350 | 3883 | 5096 | **6794** | 18058 | 1.75 |
| `B0/classifieds` | DOM | 1 | 3496 | 3769 | 4949 | **6495** | 17940 | 1.72 |
| `B0/classifieds` | SoM | 2 | 6099 | 4741 | 5689 | **6287** | 18914 | 1.33 |
| `B0/classifieds` | P-text | 1 | 3546 | 3663 | 4674 | **5861** | 17752 | 1.6 |
| `B0/classifieds` | P-SoM | 1 | 3636 | 3796 | 4788 | **5472** | 17916 | 1.44 |
| `B0/classifieds` | Vision | 1 | 3557 | 3495 | 4176 | **4265** | 4335 | 1.22 |
| `B0/reddit` | SoM | 1 | 4103 | 4754 | 6777 | **7129** | 8029 | 1.5 |
| `B0/reddit` | P-prompt | 1 | 4049 | 4649 | 6185 | **7008** | 10679 | 1.51 |
| `B0/reddit` | DOM | 2 | 6882 | 4537 | 6130 | **6913** | 10216 | 1.52 |
| `B0/reddit` | P-SoM | 1 | 4669 | 4018 | 5769 | **6136** | 7177 | 1.53 |
| `B0/reddit` | P-text | 1 | 4713 | 3812 | 5598 | **6020** | 6799 | 1.58 |
| `B0/reddit` | Vision | 1 | 4759 | 3518 | 4542 | **5253** | 5548 | 1.49 |

⚠️ **Pooled over runs where more than one exists.** `SoM` on `B0/classifieds` pools 2; `DOM` on `B0/reddit` pools 2 — a same-condition replicate lives under `phase1/` for those, so their step counts are correspondingly larger. Quantiles of one condition pooled across its own reruns are still quantiles of that condition, but the step counts are not comparable across rows without this column.

⚠️ `tokens.input` is the TOTAL input; the `input_text` / `input_image` split is null on B0 (the hosted endpoint does not itemise it), so a screenshot-bearing mode's tail cannot be attributed between text and image here. 台账 §260 estimated the image share from a som-vs-dom median difference instead.

## What this is for

Both columns are properties a single-mode deployment cannot measure about itself: diagnosability needs a shared ruleset applied across representations, and a token tail is only interpretable next to the alternatives'. They belong with the other deployment-facing numbers -- fusion's premium clearing the rerun threshold in 0 of 8 cells (`fusion_premium.md`), the cost of unstable element ids, and the abstention frontier (`abstention_learnability.md`).
