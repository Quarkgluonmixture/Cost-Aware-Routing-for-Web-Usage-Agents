---
type: analysis
status: complete
purpose: per-metric run-to-run band for the 26 behavioural metrics, and which cross-mode differences survive it
post_hoc_exploratory: true
scope_warning: primary cell B0 x classifieds; also red_B0, wared_B1 (tables in the last section); one rerun per arm. A band from a single rerun is a point estimate, not a bound — the same caveat noise_floor_inventory carries for the SR-scale band.
producer: scripts/analysis/replicate_metric_noise.py
---

# Can a rerun produce the behavioural differences?

Regenerate: `.venv/bin/python3 scripts/analysis/replicate_metric_noise.py`

Every success-rate claim in this project is judged against the rerun band. The 26-metric behavioural claims never were — not for a reason, but because no per-metric band existed. 6 replicated arms on `B0·classifieds` (DOM, P-SoM, P-prompt, P-text, SoM, Vision) now allow one.

**20 of 25 metrics** have a cross-mode spread larger than the largest run-to-run movement of the same metric.

| dimension | metric | cross-mode spread | rerun band | ratio | bigger than a rerun? |
|---|---|---|---|---|---|
| Outcome | `sr_pct` | 11.607 | 2.679 | 4.33× | **yes** |
| Outcome | `n_success` | 26.000 | 6.000 | 4.33× | **yes** |
| Outcome | `n_unique_solves` | — | — | — | *cross-mode by construction* |
| Macro | `n_steps` | 2.567 | 1.692 | 1.52× | **yes** |
| Macro | `cap_hit_rate` | 0.062 | 0.045 | 1.40× | **yes** |
| Macro | `click_frac` | 0.041 | 0.014 | 2.87× | **yes** |
| Macro | `type_frac` | 0.078 | 0.023 | 3.47× | **yes** |
| Macro | `scroll_frac` | 0.106 | 0.020 | 5.16× | **yes** |
| Macro | `search_loop_rate` | 0.121 | 0.045 | 2.70× | **yes** |
| Macro | `url_revisit_rate` | 0.081 | 0.041 | 1.98× | **yes** |
| Micro | `parse_fail_rate` | 0.002 | 0.002 | 1.00× | no |
| Micro | `action_fail_rate` | 0.068 | 0.015 | 4.41× | **yes** |
| Micro | `click_fail_rate` | 0.063 | 0.025 | 2.48× | **yes** |
| Micro | `type_fail_rate` | 0.030 | 0.006 | 5.28× | **yes** |
| Micro | `no_change_rate` | 0.053 | 0.014 | 3.75× | **yes** |
| Micro | `scroll_inert_rate` | 0.078 | 0.035 | 2.23× | **yes** |
| Micro | `noop_inert_rate` | 0.026 | 0.014 | 1.87× | **yes** |
| Micro | `visibility_gap_rate` | 0.016 | 0.011 | 1.56× | **yes** |
| Micro | `locator_fallback_rate` | 0.076 | 0.009 | 8.06× | **yes** |
| Micro | `action_repeat_frac` | 0.096 | 0.024 | 4.01× | **yes** |
| Micro | `finish_rate` | 0.054 | 0.036 | 1.50× | **yes** |
| Efficiency | `mean_cost_usd` | 0.008 | 0.008 | 0.90× | no |
| Efficiency | `cost_rel_dom` | 0.108 | 0.120 | 0.90× | no |
| Efficiency | `mean_latency_s` | 19.562 | 22.488 | 0.87× | no |
| Efficiency | `mean_latency_canonical_s` | 19.204 | 22.846 | 0.84× | no |
| Efficiency | `mean_tokens` | 8697.978 | 7745.326 | 1.12× | **yes** |

## The metrics a rerun can reproduce

- **`mean_latency_canonical_s`** (latency canonical / episode (s)) — spread 19.204 against a band of 22.846, ratio **0.84×**.
- **`mean_latency_s`** (latency / episode (s)) — spread 19.562 against a band of 22.488, ratio **0.87×**.
- **`mean_cost_usd`** (billed cost / episode) — spread 0.008 against a band of 0.008, ratio **0.90×**.
- **`cost_rel_dom`** (cost relative to DOM (within cell)) — spread 0.108 against a band of 0.120, ratio **0.90×**.
- **`parse_fail_rate`** (parse-invalid step rate) — spread 0.002 against a band of 0.002, ratio **1.00×**.

⚠️ **Both latency metrics are in that list, and that is not a coincidence.** Independently of this table, `latency_decomposition` measured that only 22–67% of a step is the model call — the rest is the browser and the container — and that removing the container changes which mode is fastest in 4 of 8 cells. Two unrelated routes reach the same place: **on this cell the latency axis does not resolve modes above run-to-run movement.** Claim 9's safe form (*the cost ordering and the latency ordering disagree*) is a statement about two rankings and survives; any sentence naming a mode as fastest does not.

`cross-mode spread` = max − min of that metric over the six modes in the canonical cell. `rerun band` = the largest |metric(run A) − metric(run B)| over the replicated arms. A ratio near or below 1 means the differences the profile reports between modes are the size a rerun of one mode produces on its own.

## What this does and does not settle

**It is one cell and one rerun per arm.** The band is a point estimate of a random quantity, exactly as `noise_floor_inventory` §1b says of the SR-scale band — a second rerun would move it. Nothing here should be read as a threshold.

**It does not touch the non-separability result directly.** That claim is about which mode is *extreme* on a metric across 8 cells, not about the size of a gap in one cell. A metric can have a small spread and still put the same mode at the top in every cell — consistency and magnitude are different questions, and the ≥83% bar is a consistency bar. What this table adds is the magnitude the consistency is about, which the profile never printed.

## Other cells with every arm replicated

Same computation per cell (added 2026-10-07, 实验笔记 §537). The reading above is B0 x classifieds; each cell below is read from its own table.

### `red_B0` — 25 of 25 metrics exceed the rerun band

| dimension | metric | cross-mode spread | rerun band | ratio | bigger than a rerun? |
|---|---|---|---|---|---|
| Outcome | `sr_pct` | 7.389 | 3.448 | 2.14× | **yes** |
| Outcome | `n_success` | 15.000 | 7.000 | 2.14× | **yes** |
| Outcome | `n_unique_solves` | — | — | — | *cross-mode by construction* |
| Macro | `n_steps` | 3.488 | 1.493 | 2.34× | **yes** |
| Macro | `cap_hit_rate` | 0.202 | 0.084 | 2.41× | **yes** |
| Macro | `click_frac` | 0.137 | 0.030 | 4.59× | **yes** |
| Macro | `type_frac` | 0.073 | 0.012 | 6.29× | **yes** |
| Macro | `scroll_frac` | 0.243 | 0.037 | 6.59× | **yes** |
| Macro | `search_loop_rate` | 0.232 | 0.025 | 9.40× | **yes** |
| Macro | `url_revisit_rate` | 0.109 | 0.019 | 5.77× | **yes** |
| Micro | `parse_fail_rate` | 0.003 | 0.002 | 1.35× | **yes** |
| Micro | `action_fail_rate` | 0.184 | 0.052 | 3.52× | **yes** |
| Micro | `click_fail_rate` | 0.074 | 0.039 | 1.89× | **yes** |
| Micro | `type_fail_rate` | 0.044 | 0.026 | 1.71× | **yes** |
| Micro | `no_change_rate` | 0.160 | 0.044 | 3.60× | **yes** |
| Micro | `scroll_inert_rate` | 0.182 | 0.020 | 9.18× | **yes** |
| Micro | `noop_inert_rate` | 0.029 | 0.020 | 1.43× | **yes** |
| Micro | `visibility_gap_rate` | 0.033 | 0.023 | 1.45× | **yes** |
| Micro | `locator_fallback_rate` | 0.112 | 0.029 | 3.83× | **yes** |
| Micro | `action_repeat_frac` | 0.131 | 0.029 | 4.55× | **yes** |
| Micro | `finish_rate` | 0.202 | 0.089 | 2.28× | **yes** |
| Efficiency | `mean_cost_usd` | 0.012 | 0.006 | 1.95× | **yes** |
| Efficiency | `cost_rel_dom` | 0.122 | 0.063 | 1.95× | **yes** |
| Efficiency | `mean_latency_s` | 181.607 | 92.885 | 1.96× | **yes** |
| Efficiency | `mean_latency_canonical_s` | 143.615 | 75.249 | 1.91× | **yes** |
| Efficiency | `mean_tokens` | 14158.847 | 5832.685 | 2.43× | **yes** |

### `wared_B1` — 23 of 25 metrics exceed the rerun band

| dimension | metric | cross-mode spread | rerun band | ratio | bigger than a rerun? |
|---|---|---|---|---|---|
| Outcome | `sr_pct` | 6.731 | 5.769 | 1.17× | **yes** |
| Outcome | `n_success` | 7.000 | 6.000 | 1.17× | **yes** |
| Outcome | `n_unique_solves` | — | — | — | *cross-mode by construction* |
| Macro | `n_steps` | 1.856 | 0.962 | 1.93× | **yes** |
| Macro | `cap_hit_rate` | 0.096 | 0.058 | 1.67× | **yes** |
| Macro | `click_frac` | 0.119 | 0.027 | 4.44× | **yes** |
| Macro | `type_frac` | 0.222 | 0.019 | 11.91× | **yes** |
| Macro | `scroll_frac` | 0.232 | 0.015 | 15.41× | **yes** |
| Macro | `search_loop_rate` | 0.423 | 0.077 | 5.50× | **yes** |
| Macro | `url_revisit_rate` | 0.074 | 0.028 | 2.66× | **yes** |
| Micro | `parse_fail_rate` | 0.018 | 0.008 | 2.16× | **yes** |
| Micro | `action_fail_rate` | 0.210 | 0.049 | 4.24× | **yes** |
| Micro | `click_fail_rate` | 0.129 | 0.069 | 1.88× | **yes** |
| Micro | `type_fail_rate` | 0.062 | 0.009 | 6.79× | **yes** |
| Micro | `no_change_rate` | 0.209 | 0.043 | 4.85× | **yes** |
| Micro | `scroll_inert_rate` | 0.191 | 0.028 | 6.77× | **yes** |
| Micro | `noop_inert_rate` | 0.016 | 0.010 | 1.52× | **yes** |
| Micro | `visibility_gap_rate` | 0.068 | 0.025 | 2.76× | **yes** |
| Micro | `locator_fallback_rate` | 0.195 | 0.034 | 5.70× | **yes** |
| Micro | `action_repeat_frac` | 0.070 | 0.037 | 1.93× | **yes** |
| Micro | `finish_rate` | 0.106 | 0.096 | 1.10× | **yes** |
| Efficiency | `mean_cost_usd` | 0.035 | 0.005 | 6.88× | **yes** |
| Efficiency | `cost_rel_dom` | 0.528 | 0.057 | 9.29× | **yes** |
| Efficiency | `mean_latency_s` | 25.392 | 73.592 | 0.35× | no |
| Efficiency | `mean_latency_canonical_s` | 25.392 | 73.592 | 0.35× | no |
| Efficiency | `mean_tokens` | 37061.750 | 5318.817 | 6.97× | **yes** |
