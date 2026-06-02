# Intelligent-baseline ladder — archive dev-sanity (paper §6.5, B-1006 R5)

> ⚠️ **SANITY-CHECK ONLY / DISCLOSURE, NOT GATING.** Same Option-C caveat as `l1_archive_simulation.py`: archive outcomes are pre-fire / directional and cannot enter the paper as SR claims. Paper §6 main data = Phase 1a Pass-2 fire. These four baselines bound the learned router; they do NOT feed the H10 operational deployment gate.

Generated: `2026-06-02T04:51:56.415891+00:00` · schema `ladder-2026-06-02-v1`

## Ladder definition

| Arm | Role | Bounds the router... |
|---|---|---|
| `always_cheapest_dom` (= always-DOM) | cost-only baseline | the cost-axis reference point |
| `decision_stump` (depth-1 tree) | feature floor | from below on feature-set value |
| `per_task_lookup` (oracle) | SR upper bound | from above (infinite-capacity ceiling) |
| `lr_dom_features_only` | text ablation | from the text-feature side |
| `learned_router_proxy` (8-dim LR) | bounded subject | (the archive stand-in being bracketed) |

## B0_classifieds

- n tasks: **224** (127 with no successful mode) · modes: `['dom', 'phantom_prompt', 'phantom_som', 'phantom_text', 'som', 'vision']`
- total-billed cost near-tie: single-mode cost spread **11.7%** (empirical cheapest = `vision` (DOM not strictly min)) — `{'dom': 0.069619, 'phantom_prompt': 0.068526, 'phantom_som': 0.072057, 'phantom_text': 0.069186, 'som': 0.072357, 'vision': 0.064807}`
- oracle label distribution: `{'dom': 39, 'som': 24, 'phantom_prompt': 13, 'phantom_som': 9, 'vision': 9, 'phantom_text': 3}`

| Arm | Role | SR % [95% CI] | Cost USD [95% CI] | Routed modes |
|---|---|---|---|---|
| `always_cheapest_dom` | cost_only_baseline | 17.41 [12.95, 22.32] | 0.06962 [0.06397, 0.07603] | dom=224 |
| `decision_stump` | feature_floor | 19.64 [14.73, 24.55] | 0.07118 [0.06471, 0.07770] | phantom_text=60, som=45, dom=45, phantom_prompt=35 |
| `lr_dom_features_only` | text_ablation | 22.32 [17.41, 28.12] | 0.06595 [0.05905, 0.07280] | phantom_prompt=58, som=57, phantom_text=45, vision=36 |
| `learned_router_proxy` | bounded_subject | 25.00 [19.64, 30.80] | 0.06887 [0.06137, 0.07538] | phantom_prompt=82, som=43, vision=34, phantom_som=27 |
| `per_task_lookup` | sr_upper_bound | 43.30 [37.05, 49.55] | 0.06411 [0.05843, 0.07004] | dom=166, som=24, phantom_prompt=13, phantom_som=9 |

**Ladder bounding** (R5 defense):

- SR ceiling holds (oracle ≥ every arm) — STRUCTURAL invariant: **True**
- router proxy beats single-feature stump: **True** (+5.36 pp)
- router proxy beats no-text ablation: **True** (+2.68 pp)
- generalisation headroom to memorisation ceiling (oracle − proxy): **+18.30 pp**
- router lift over always-DOM (proxy − always-DOM): **+7.59 pp**
- cost reference points (total-billed USD): always-DOM=0.06962 · proxy=0.06887 · oracle=0.06411

_Interpretation_: a learned router worth deploying must sit STRICTLY above the stump / no-text floor (positive feature-set + text-feature value) while leaving the ceiling gap as its generalisation headroom. On the small skewed archive the floor arms can tie the proxy — that itself is the honest pre-fire read; the Phase 1a fire router is the real test.

