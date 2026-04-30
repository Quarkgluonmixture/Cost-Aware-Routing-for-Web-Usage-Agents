# B-37 Pilot T=0 Decision Gate

**Audit date**: 2026-04-30
**Aggregate verdict**: **YELLOW_INVESTIGATE**

## Per-site summary

| Site | Pilot run | Pilot SR (N) | Paper-grade matched SR (N) | Mode collapse % | Verdict |
|---|---|---:|---:|---:|---|
| classifieds | — | — | — | — | NOT_RUN_YET |
| reddit | `B0_dom_pilot_T0_reddit_20260430_121324` | 17.86% (28) | 17.86% (28) | 42.9% | **PASS** |
| shopping | `B0_dom_pilot_T0_shopping_20260430_121328` | 13.33% (30) | 13.33% (30) | 53.3% | **PASS** |

## Decision matrix

- **PASS** (within ±5pp): green-light Phase A full re-run with T=0 baseline
- **MARGINAL** (-5 to -15pp): investigate top_p / try T=0.05 / check first-action distribution
- **FAIL** (< -15pp or mode collapse): revert T=0→0.1, paper takes B-37 disclosure path

Mode collapse signature: ≥80% of episodes share same first action (action_type, element_id)