# B-37 Pilot T=0 Decision Gate

**Audit date**: 2026-04-30
**Aggregate verdict**: **YELLOW_INVESTIGATE**

## Per-site summary

| Site | Pilot run | Pilot SR (N) | Paper-grade matched SR (N) | Mode collapse % | Verdict |
|---|---|---:|---:|---:|---|
| classifieds | — | — | — | — | NOT_RUN_YET |
| reddit | `B0_dom_pilot_T0_reddit_20260430_193022` | 20.0% (30) | 16.67% (30) | 46.7% | **PASS** |
| shopping | `B0_dom_pilot_T0_shopping_20260430_193028` | 10.0% (30) | 13.33% (30) | 40.0% | **PASS** |

## Decision matrix

- **PASS** (within ±5pp): green-light Phase A full re-run with T=0 baseline
- **MARGINAL** (-5 to -15pp): investigate top_p / try T=0.05 / check first-action distribution
- **FAIL** (< -15pp or mode collapse): revert T=0→0.1, paper takes B-37 disclosure path

Mode collapse signature: ≥80% of episodes share same first action (action_type, element_id)