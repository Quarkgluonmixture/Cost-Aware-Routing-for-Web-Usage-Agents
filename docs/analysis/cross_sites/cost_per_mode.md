# Efficiency — Cost Per Mode (deployment-class aware)

B0 reports avg_total_cost_usd from per-token API rates (Qwen3-VL-235B-A22B $0.001/1k input, $0.005/1k output). B1 reports avg_total_energy_kwh × $0.12/kWh as electricity-equivalent cost — local inference pays no API dollars; the per-token cost field in B1 condition_summary_v2.json is artifact (uses B0 rates) and is NOT comparable. B0 vs B1 dollar costs belong to different classes (API call cost vs electricity), so the paper presents both side-by-side, not a single ratio.

## B0 — API token dollars (paid)

| site | mode | avg_steps | avg_total_billed_cost_usd ($/ep) |
|---|---|---:|---:|
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

## B2 — electricity equivalent ($/ep, Gemma3-VL local 4B)

Computed as `avg_total_energy_kwh × $0.12/kWh` (DGX Spark, UK industrial rate per `metrics.energy.region: uk` in B2 yaml). Same deployment class as B1 (per advisor §138 B2 ≈ B1 matched-capability lock).

| site | mode | avg_steps | avg_energy_kwh | avg_co2e_kg | avg_electricity_usd ($/ep) |
|---|---|---:|---:|---:|---:|

## Deployment-class ratio (informative, not a paper claim)

| site | avg B0 API ($/ep) | avg B1 electricity ($/ep) | ratio (B0/B1) |
|---|---:|---:|---:|

The qualitative cost gap between API and local inference is large (2–3 orders of magnitude per these data) but is fundamentally a deployment-mode comparison, not a model-size ratio. Reporting a single multiplier (e.g. '30x') without specifying the cost class is misleading.

