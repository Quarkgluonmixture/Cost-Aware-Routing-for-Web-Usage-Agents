# L1 (learned task-prior LR) archive simulation — SANITY-CHECK ONLY

> ⚠️ NOT preregistration lock substrate. Same Option C caveats as `p1_archive_simulation_findings_2026-05-16.md` + `archive_diagnostic_2026-05-16.md`.

Run date: `2026-05-16T11:07:07.786940Z`
Total tasks: 444 (cls + red, B0 only)

## Oracle label distribution

```
{'dom': 366, 'phantom_prompt': 4, 'phantom_som': 4, 'phantom_text': 15, 'som': 39, 'vision': 16}
```

## Variant comparison (5-fold site-stratified CV)

| Variant | Method | Overall SR | cls SR [95% CI] | red SR [95% CI] | vs always_phantom_som |
|---|---|---:|---|---|---|
| A: uniform LR (control) | LR | 13.29% [12.34, 14.35] | 14.96% [13.55, 16.41] | 11.43% [10.05, 12.76] | cls -0.85 / red -2.86 |
| B: balanced LR | LR | 14.29% [13.25, 15.34] | 17.84% [16.30, 19.42] | 10.33% [9.05, 11.67] | cls +2.02 / red -3.95 |
| C: binary + hand rule | LR | 13.04% [12.07, 14.12] | 14.70% [13.29, 16.11] | 11.19% [9.90, 12.52] | cls -1.11 / red -3.10 |

## Baselines (no router)

| Baseline | cls SR | red SR |
|---|---:|---:|
| always_dom | 14.96% | 11.43% |
| always_som | 23.08% | 11.90% |
| always_phantom_som | 15.81% | 14.29% |

## Prediction distribution (mode distribution from CV predictions)

| Variant | Distribution |
|---|---|
| A_uniform | dom=4440(100.0%) |
| B_balanced | phantom_text=950(21.4%), vision=808(18.2%), dom=797(18.0%), som=786(17.7%), phantom_prompt=735(16.6%), phantom_som=362(8.2%), invalid_phantom_prompt=2(0.0%) |
| C_binary | dom=2510(56.5%), phantom_som=1200(27.0%), som=730(16.4%) |

## Verdict

**Best variant**: A (min Δ across cells: -2.86pp vs always_phantom_som)

❌ **L1 not viable on archive** — all variants underperform `always_phantom_som` by > 0.5pp. proposals_v5 §2.4 path 3 (drop L1; paper §6 = L2 verbose reactive on phantom_som default).

Note: archive ≠ Phase 1a. L1 may behave differently on fresh post-fix data + with B1/B2 capability tier feature variance.