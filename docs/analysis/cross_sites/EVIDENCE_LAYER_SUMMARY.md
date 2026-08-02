---
type: analysis
status: rolling
created: 2026-08-02
purpose: one page saying what the evidence layer can and cannot support, before any framing is chosen
audience: self + advisor 08-03
note: this is a coverage map, NOT a findings narrative. Numbers live in the products it points at.
---

# Evidence layer — what we hold

Written deliberately **before** choosing a frame, so the frame is chosen against coverage rather
than coverage assembled to fit a frame. Two candidate frames are live and both are listed in §5.

## 1. The coverage matrix

Rows are the seven evidence dimensions. Columns are the units they cover. **W** marks WebArena.

| dimension | product | VWA 6 cells | WA | 6 modes | status |
|---|---|---|---|---|---|
| **Outcome / SR** | `sr_per_mode` · `fusion_premium` | ✅ | ✅ | ✅ | complete |
| **Noise / rerun floor** | `noise_floor_inventory` · `phase0b_noise_floor` · `label_instability` | ⚠️ 1 of 6 cells | ✅ 1 pair | ⚠️ 2 of 6 arms | **SoM replicate queued** |
| **Four-layer profile** | `per_mode_four_dimension_profile` v2 (24 metrics) | ✅ | ❌ **impossible** | ✅ | complete on VWA |
| **Multi-metric Pareto** | `multimetric_pareto` | ✅ | ❌ not yet | ✅ | **WA is feasible, see §3** |
| **diag / failure attribution** | `cross_mode_failure_signatures` (marginal) · `conditional_failure_attribution` (paired) | ✅ | ❌ **impossible** | ✅ | complete on VWA |
| **2×2 ablation** | `axis_effect_size_report` · profile §2.5 non-separability | ✅ | ✅ | ✅ (the four) | complete |
| **Routing attempts** | `router_label_supply_diagnosis` · `router_triage_learnability` · `router_pooled_tier_learnability` · `confidence_cascade` | ✅ | ❌ | ✅ | complete on VWA |
| *(features)* | `routing_feature_diagnostics` | ✅ | ❌ n/a | ✅ | complete |

## 2. The one structural gap: WebArena has no step records

**All eight WA runs carry episode summaries and zero step JSONL files.** This is not an oversight
that can be repaired by analysis; the data was never written.

| WA can enter | because |
|---|---|
| Outcome | summaries carry `success` |
| Efficiency | summaries carry `total_billed_cost_usd`, `total_latency_ms`, `total_tokens`, `steps`, `total_energy_kwh` |

| WA cannot enter | because |
|---|---|
| Macro / Micro (four-layer) | every metric reads `*_steps_v2.jsonl` |
| Confidence cascade | escalation reads per-step log-probabilities |
| diag / conditional failure attribution | the rule scanner walks step records |

**Consequence for any claim.** WA is the workload that carries the modality reversal, and it is
**Outcome-and-Efficiency only**. Any claim that needs behavioural or failure-mode evidence is
VWA-only and cannot be shown to reverse. The reversal is established on success rates and on
arm-count-matched marginal gain, and on nothing else. Say so wherever it is claimed.

## 3. Gaps that are cheap to close

| gap | cost | what it buys |
|---|---|---|
| **WA into the Pareto** | ~1h. Summaries hold cost, latency and tokens; the producer currently single-sources the VWA profile and would read WA separately | the modality reversal on the efficiency axes, not only on SR. Turns a one-axis claim into two |
| **WA into Outcome + Efficiency of the four-layer** | ~1h, same data | makes the profile's own coverage gap explicit in the product rather than only here |
| **SoM replicate** | 7.8h + ~$17, queued on A100 behind the B0×WA chain | the rerun floor for the mode the fusion-premium claim is *about*. Currently borrowed from DOM and Vision |

## 4. Gaps that are not cheap, and are therefore limitations

- **A third workload.** Two workloads show a sign change and cannot characterise the axis it
  turns on. Nothing queued changes this; `shopping` has zero landed directories.
- **Replicates on the other four arms, and more than one rerun each.** Every instability figure
  is a lower bound from two arms replicated once.
- **Calibrated per-accelerator energy.** `use_pynvml: true` is configured but every step records
  `source: psutil_profile` at ~66W on a device rated several times that, so the carbon column is
  wall-clock in other units and is reported as uninformative rather than as an axis.
- **Absolute local cost.** The per-token constant for the locally-served backbones was derived
  for a different accelerator than the runs were served on. Within-cell ratios are unaffected
  because it is a single multiplier; absolute dollar figures for B1/B2 are uncalibrated.

## 5. What the layer supports, independent of frame

Stated as claims with their carrying product, so a frame can be chosen against them.

1. **Adding an arm buys 1.97–8.65pp; adding a *rerun* buys 2.0–7.6pp.** At the one-arm margin on
   `cls_B0` these are not separable. → `noise_floor_inventory`
2. **Instability is enriched 17.4× on the tasks where the routing choice is contested**, while
   aggregate SR between the same two runs moves under 2.3pp. → `label_instability`
3. **The fused mode is dearest in 5/6 cells and its pooled advantage clears neither the rerun
   band nor, against DOM, zero.** In 7/7 cells it fails to significantly beat the single channel
   that suits the workload. → `fusion_premium`
4. **Which channel to add reverses with workload modality** (VWA visual, WA text).
   → `noise_floor_inventory` §2 + `fusion_premium` §3
5. **Four routing formulations fail**, including a cascade on post-action confidence that no
   operating point makes Pareto-beat always-rich. → the four router products + `confidence_cascade`
6. **The four image-free modes are behaviourally non-separable** across 24 metrics × 6 cells,
   while the image-bearing pair is separable mostly by construction.
   → `per_mode_four_dimension_profile` v2
7. **The two channels' advantages are asymmetric in kind.** When the image channel wins, the text
   channel fails with a named signature (`P43` 1.66× on 196 hits). When the text channel wins,
   the image channel fails the way it fails everywhere, with no enrichment above 1.5×.
   → `conditional_failure_attribution`
8. **The obvious routing feature has the wrong sign**, and the right one was read and dropped.
   → `routing_feature_diagnostics`
9. **Latency is a second axis, not a restatement of cost**; adding it widens the frontier in 3/6
   cells. → `multimetric_pareto`

**Two frames are live and neither has been chosen.** One organises 1–2 as the conceptual
contribution (an oracle gap is not a routing opportunity). One organises 3–4 as it (representation
choice is a deployment-time configuration). Claims 5–9 serve either. The choice is a framing
decision to be taken with the evidence in front of it, which is the state this document exists to
create.
