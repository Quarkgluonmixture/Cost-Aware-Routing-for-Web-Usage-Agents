# Efficiency — Cost Per Mode (deployment-class aware)

B0 reports avg_total_cost_usd from per-token API rates (Qwen3-VL-235B-A22B $0.001/1k input, $0.005/1k output). B1 reports avg_total_energy_kwh × $0.12/kWh as electricity-equivalent cost — local inference pays no API dollars; the per-token cost field in B1 condition_summary_v2.json is artifact (uses B0 rates) and is NOT comparable. B0 vs B1 dollar costs belong to different classes (API call cost vs electricity), so the paper presents both side-by-side, not a single ratio.

## B0 — API token dollars (paid)

| site | mode | avg_steps | avg_total_cost_usd ($/ep) |
|---|---|---:|---:|
| reddit | DOM | 12.7 | $0.0516 |
| reddit | SoM | 8.1 | $0.0409 |
| reddit | Vision | 6.9 | $0.0227 |
| reddit | P-text | 11.5 | $0.0459 |
| reddit | P-prompt | 11.9 | $0.0486 |
| reddit | P-SoM | 9.9 | $0.0381 |
| classifieds | DOM | 11.6 | $0.0427 |
| classifieds | SoM | 8.6 | $0.0415 |
| classifieds | Vision | 7.8 | $0.0248 |
| classifieds | P-text | 11.2 | $0.0397 |
| classifieds | P-SoM | 12.0 | $0.0441 |

## B1 — electricity equivalent ($/ep)

Computed as `avg_total_energy_kwh × $0.12/kWh` (DGX Spark, UK industrial rate per `metrics.energy.region: uk` in B1 yaml).

| site | mode | avg_steps | avg_energy_kwh | avg_co2e_kg | avg_electricity_usd ($/ep) |
|---|---|---:|---:|---:|---:|
| reddit | DOM | 16.6 | 0.00377 | 0.00083 | $0.000452 |
| reddit | SoM | 11.7 | 0.00512 | 0.00113 | $0.000614 |
| reddit | Vision | 6.4 | 0.00128 | 0.00028 | $0.000153 |
| classifieds | DOM | 13.8 | 0.00522 | 0.00115 | $0.000626 |
| classifieds | SoM | 9.9 | 0.00199 | 0.00044 | $0.000238 |
| classifieds | Vision | 6.7 | 0.00194 | 0.00043 | $0.000233 |
| classifieds | P-text | 13.1 | 0.00688 | 0.00151 | $0.000826 |
| classifieds | P-SoM | 11.6 | 0.00667 | 0.00147 | $0.000801 |

## Deployment-class ratio (informative, not a paper claim)

| site | avg B0 API ($/ep) | avg B1 electricity ($/ep) | ratio (B0/B1) |
|---|---:|---:|---:|
| reddit | $0.0413 | $0.000407 | 102× |
| classifieds | $0.0386 | $0.000545 | 71× |

The qualitative cost gap between API and local inference is large (2–3 orders of magnitude per these data) but is fundamentally a deployment-mode comparison, not a model-size ratio. Reporting a single multiplier (e.g. '30x') without specifying the cost class is misleading.

