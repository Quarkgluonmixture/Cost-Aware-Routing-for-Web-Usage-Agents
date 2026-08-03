---
type: analysis
status: complete
purpose: per-metric run-to-run band for the 26 behavioural metrics, and which cross-mode differences survive it
post_hoc_exploratory: true
scope_warning: one cell (B0 x classifieds) and one rerun per arm. A band from a single rerun is a point estimate, not a bound — the same caveat noise_floor_inventory carries for the SR-scale band.
producer: scripts/analysis/replicate_metric_noise.py
---

# Can a rerun produce the behavioural differences?

Regenerate: `.venv/bin/python3 scripts/analysis/replicate_metric_noise.py`

Every success-rate claim in this project is judged against the rerun band. The 26-metric behavioural claims never were — not for a reason, but because no per-metric band existed. Three replicated arms on `B0·classifieds` (dom, vision, **som**, the last landing 2026-08-03) now allow one.

**22 of 25 metrics** have a cross-mode spread larger than the largest run-to-run movement of the same metric.

| dimension | metric | cross-mode spread | rerun band | ratio | bigger than a rerun? |
|---|---|---|---|---|---|
| Outcome | `sr_pct` | 11.607 | 2.232 | 5.20× | **yes** |
| Outcome | `n_success` | 26.000 | 5.000 | 5.20× | **yes** |
| Outcome | `n_unique_solves` | — | — | — | *cross-mode by construction* |
| Macro | `n_steps` | 2.567 | 0.469 | 5.48× | **yes** |
| Macro | `cap_hit_rate` | 0.062 | 0.045 | 1.40× | **yes** |
| Macro | `click_frac` | 0.041 | 0.014 | 2.87× | **yes** |
| Macro | `type_frac` | 0.078 | 0.019 | 4.21× | **yes** |
| Macro | `scroll_frac` | 0.106 | 0.006 | 16.85× | **yes** |
| Macro | `search_loop_rate` | 0.121 | 0.018 | 6.75× | **yes** |
| Macro | `url_revisit_rate` | 0.081 | 0.022 | 3.66× | **yes** |
| Micro | `parse_fail_rate` | 0.002 | 0.002 | 1.00× | no |
| Micro | `action_fail_rate` | 0.068 | 0.014 | 4.87× | **yes** |
| Micro | `click_fail_rate` | 0.063 | 0.025 | 2.48× | **yes** |
| Micro | `type_fail_rate` | 0.030 | 0.006 | 5.28× | **yes** |
| Micro | `no_change_rate` | 0.053 | 0.009 | 5.82× | **yes** |
| Micro | `scroll_inert_rate` | 0.078 | 0.017 | 4.68× | **yes** |
| Micro | `noop_inert_rate` | 0.026 | 0.007 | 3.92× | **yes** |
| Micro | `visibility_gap_rate` | 0.016 | 0.011 | 1.56× | **yes** |
| Micro | `locator_fallback_rate` | 0.076 | 0.006 | 11.81× | **yes** |
| Micro | `action_repeat_frac` | 0.096 | 0.004 | 21.88× | **yes** |
| Micro | `finish_rate` | 0.054 | 0.036 | 1.50× | **yes** |
| Efficiency | `mean_cost_usd` | 0.008 | 0.002 | 3.24× | **yes** |
| Efficiency | `cost_rel_dom` | 0.108 | 0.026 | 4.22× | **yes** |
| Efficiency | `mean_latency_s` | 19.562 | 22.488 | 0.87× | no |
| Efficiency | `mean_latency_canonical_s` | 19.204 | 22.846 | 0.84× | no |
| Efficiency | `mean_tokens` | 8697.978 | 2186.054 | 3.98× | **yes** |

## The metrics a rerun can reproduce

- **`mean_latency_canonical_s`** (latency canonical / episode (s)) — spread 19.204 against a band of 22.846, ratio **0.84×**.
- **`mean_latency_s`** (latency / episode (s)) — spread 19.562 against a band of 22.488, ratio **0.87×**.
- **`parse_fail_rate`** (parse-invalid step rate) — spread 0.002 against a band of 0.002, ratio **1.00×**.

⚠️ **Both latency metrics are in that list, and that is not a coincidence.** Independently of this table, `latency_decomposition` measured that only 22–67% of a step is the model call — the rest is the browser and the container — and that removing the container changes which mode is fastest in 4 of 8 cells. Two unrelated routes reach the same place: **on this cell the latency axis does not resolve modes above run-to-run movement.** Claim 9's safe form (*the cost ordering and the latency ordering disagree*) is a statement about two rankings and survives; any sentence naming a mode as fastest does not.

`cross-mode spread` = max − min of that metric over the six modes in the canonical cell. `rerun band` = the largest |metric(run A) − metric(run B)| over the three replicated arms. A ratio near or below 1 means the differences the profile reports between modes are the size a rerun of one mode produces on its own.

## What this does and does not settle

**It is one cell and one rerun per arm.** The band is a point estimate of a random quantity, exactly as `noise_floor_inventory` §1b says of the SR-scale band — a second rerun would move it. Nothing here should be read as a threshold.

**It does not touch the non-separability result directly.** That claim is about which mode is *extreme* on a metric across 8 cells, not about the size of a gap in one cell. A metric can have a small spread and still put the same mode at the top in every cell — consistency and magnitude are different questions, and the ≥83% bar is a consistency bar. What this table adds is the magnitude the consistency is about, which the profile never printed.
