# Routing ceiling — what a perfect per-task choice could buy

- leak policy: **primary = leaked successes set to 0, denominator unchanged (user decision 2026-08-04); leak_kept retained for comparison**
- 6 leaked successes in the scored universe
- **The success ceiling is a six-arm union against a one-arm baseline.** The arm-matched columns are the honest comparison and are printed beside it.

## Primary (leaked successes zeroed)

| cell | n | best single mode | ceiling A: any mode | headroom | ceiling B': triage cost | +1 arm | rerun once |
|---|---|---|---|---|---|---|---|
| `wa_red_B0` | 104 | P-text 35.58% | **51.92%** | +16.35pp | SR unchanged, **-10.7%** | +5.77pp | 2.00–4.00pp |
| `cls_B0` | 224 | SoM 27.23% | **43.30%** | +16.07pp | SR unchanged, **-12.8%** | +7.14pp | 4.91–7.59pp |
| `wa_red_B1` | 104 | DOM 16.35% | **30.77%** | +14.42pp | SR unchanged, **-26.8%** | +4.81pp | 2.00–4.00pp |
| `red_B0` | 203 | SoM 14.78% | **26.11%** | +11.33pp | SR unchanged, **-9.5%** | +4.93pp | —pp |
| `cls_B1` | 224 | SoM 14.29% | **24.55%** | +10.27pp | SR unchanged, **-19.4%** | +4.91pp | —pp |
| `red_B1` | 203 | SoM 6.90% | **11.82%** | +4.93pp | SR unchanged, **-30.6%** | +1.97pp | —pp |
| `cls_B2` | 224 | SoM 2.23% | **7.14%** | +4.91pp | SR unchanged, **-21.3%** | +2.23pp | —pp |
| `red_B2` | 203 | DOM 2.46% | **5.91%** | +3.45pp | SR unchanged, **-26.7%** | +1.97pp | —pp |

## Effect of the leak policy

| cell | zeroed | best SR kept → zeroed | ceiling kept → zeroed |
|---|---|---|---|
| `red_B2` | 3 | 3.94% → 2.46% | 7.39% → 5.91% |
| `red_B0` | 2 | 14.78% → 14.78% | 26.11% → 26.11% |
| `red_B1` | 1 | 7.39% → 6.90% | 11.82% → 11.82% |
| `cls_B0` | 0 | 27.23% → 27.23% | 43.30% → 43.30% |
| `cls_B1` | 0 | 14.29% → 14.29% | 24.55% → 24.55% |
| `cls_B2` | 0 | 2.23% → 2.23% | 7.14% → 7.14% |
| `wa_red_B0` | 0 | 35.58% → 35.58% | 51.92% → 51.92% |
| `wa_red_B1` | 0 | 16.35% → 16.35% | 30.77% → 30.77% |

## Why the ceiling is hard to reach

| cell | no mode solves | >1 solver (the routable set) |
|---|---|---|
| `wa_red_B0` | 48.1% | 36/104 = 34.6% |
| `cls_B0` | 56.7% | 68/224 = 30.4% |
| `wa_red_B1` | 69.2% | 18/104 = 17.3% |
| `red_B0` | 73.9% | 35/203 = 17.2% |
| `cls_B1` | 75.4% | 29/224 = 12.9% |
| `red_B1` | 88.2% | 17/203 = 8.4% |
| `cls_B2` | 92.9% | 4/224 = 1.8% |
| `red_B2` | 94.1% | 3/203 = 1.5% |

