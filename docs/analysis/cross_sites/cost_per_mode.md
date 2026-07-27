# Efficiency — Cost Per Mode (deployment-class aware)

B0 reports avg_total_cost_usd from per-token API rates (Qwen3-VL-235B-A22B $0.001/1k input, $0.005/1k output). B1 reports avg_total_energy_kwh × $0.12/kWh as electricity-equivalent cost — local inference pays no API dollars; the per-token cost field in B1 condition_summary_v2.json is artifact (uses B0 rates) and is NOT comparable. B0 vs B1 dollar costs belong to different classes (API call cost vs electricity), so the paper presents both side-by-side, not a single ratio.

## B0 — API token dollars (paid)

| site | mode | avg_steps | avg_total_billed_cost_usd ($/ep) |
|---|---|---:|---:|
| reddit | DOM | 20.2 | $0.1013 |
| reddit | SoM | 20.0 | $0.1100 |
| reddit | Vision | 23.2 | $0.0975 |
| reddit | P-text | 23.0 | $0.1050 |
| reddit | P-prompt | 19.8 | $0.1011 |
| reddit | P-SoM | 22.7 | $0.1078 |
| classifieds | DOM | 15.6 | $0.0696 |
| classifieds | SoM | 13.7 | $0.0724 |
| classifieds | Vision | 15.9 | $0.0648 |
| classifieds | P-text | 15.8 | $0.0692 |
| classifieds | P-prompt | 15.0 | $0.0685 |
| classifieds | P-SoM | 16.2 | $0.0721 |

## B1 — electricity equivalent ($/ep)

Computed as `avg_total_energy_kwh × $0.12/kWh` (DGX Spark, UK industrial rate per `metrics.energy.region: uk` in B1 yaml).

| site | mode | avg_steps | avg_energy_kwh | avg_co2e_kg | avg_electricity_usd ($/ep) |
|---|---|---:|---:|---:|---:|
| reddit | DOM | 23.4 | 0.01098 | 0.00241 | $0.001317 |
| reddit | SoM | 22.2 | 0.01103 | 0.00243 | $0.001324 |
| reddit | Vision | 23.1 | 0.00830 | 0.00183 | $0.000996 |
| reddit | P-text | 25.5 | 0.01078 | 0.00237 | $0.001294 |
| reddit | P-prompt | 23.7 | 0.01109 | 0.00244 | $0.001331 |
| reddit | P-SoM | 25.2 | 0.01116 | 0.00246 | $0.001340 |
| classifieds | DOM | 21.4 | 0.00564 | 0.00124 | $0.000677 |
| classifieds | SoM | 18.0 | 0.00482 | 0.00106 | $0.000578 |
| classifieds | Vision | 20.2 | 0.00497 | 0.00109 | $0.000596 |
| classifieds | P-text | 22.5 | 0.00575 | 0.00126 | $0.000689 |
| classifieds | P-prompt | 21.4 | 0.00552 | 0.00121 | $0.000663 |
| classifieds | P-SoM | 21.3 | 0.00571 | 0.00126 | $0.000685 |

## B2 — electricity equivalent ($/ep, Gemma3-VL local 4B)

Computed as `avg_total_energy_kwh × $0.12/kWh` (DGX Spark, UK industrial rate per `metrics.energy.region: uk` in B2 yaml). Same deployment class as B1 (per advisor §138 B2 ≈ B1 matched-capability lock).

| site | mode | avg_steps | avg_energy_kwh | avg_co2e_kg | avg_electricity_usd ($/ep) |
|---|---|---:|---:|---:|---:|
| reddit | DOM | 28.4 | 0.01224 | 0.00269 | $0.001468 |
| reddit | SoM | 26.4 | 0.01142 | 0.00251 | $0.001370 |
| reddit | Vision | 26.9 | 0.01018 | 0.00224 | $0.001222 |
| reddit | P-text | 27.3 | 0.01236 | 0.00272 | $0.001484 |
| reddit | P-prompt | 27.7 | 0.01102 | 0.00242 | $0.001322 |
| reddit | P-SoM | 27.9 | 0.01167 | 0.00257 | $0.001400 |
| classifieds | DOM | 27.4 | 0.00742 | 0.00163 | $0.000890 |
| classifieds | SoM | 24.4 | 0.00695 | 0.00153 | $0.000834 |
| classifieds | Vision | 28.2 | 0.00777 | 0.00171 | $0.000932 |
| classifieds | P-text | 26.8 | 0.00736 | 0.00162 | $0.000884 |
| classifieds | P-prompt | 27.8 | 0.00730 | 0.00161 | $0.000876 |
| classifieds | P-SoM | 28.4 | 0.00758 | 0.00167 | $0.000909 |

## Deployment-class ratio (informative, not a paper claim)

| site | avg B0 API ($/ep) | avg B1 electricity ($/ep) | ratio (B0/B1) |
|---|---:|---:|---:|
| reddit | $0.1038 | $0.001267 | 82× |
| classifieds | $0.0694 | $0.000648 | 107× |

The qualitative cost gap between API and local inference is large (2–3 orders of magnitude per these data) but is fundamentally a deployment-mode comparison, not a model-size ratio. Reporting a single multiplier (e.g. '30x') without specifying the cost class is misleading.

