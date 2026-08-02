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
| **Four-layer profile** | `per_mode_four_dimension_profile` v2 (24 metrics) | ✅ | ✅ `--with-wa` | ✅ | complete |
| **Multi-metric Pareto** | `multimetric_pareto` | ✅ | ✅ `--with-wa` | ✅ | complete |
| **diag / failure attribution** | `cross_mode_failure_signatures` (marginal) · `conditional_failure_attribution` (paired) | ✅ | ✅ **added 08-02** | ✅ | complete |
| **2×2 ablation** | `axis_effect_size_report` · profile §2.5 non-separability | ✅ | ✅ | ✅ (the four) | complete |
| **Routing attempts** | `router_label_supply_diagnosis` · `router_triage_learnability` · `router_pooled_tier_learnability` · `confidence_cascade` | ✅ | ✅ cascade `--with-wa`; the three router products stay VWA (they need the router feature table) | ✅ | complete |
| *(features)* | `routing_feature_diagnostics` | ✅ | ❌ n/a | ✅ | complete |

## 2. ~~The one structural gap~~ — RETRACTED 2026-08-02

An earlier version of this section reported that WebArena carried no step records and that the
Macro, Micro, cascade and diag layers were therefore **impossible** on it. **That was wrong on
both counts and the entry is kept, struck through, because the error is instructive.**

What was true: `find results/webarena -name '*steps*'` returned zero on this machine.
What was false: the inference. The paper-grade host is the source of truth and holds all 104 step
files for each of the six WA modes; `sync_a100_results.sh` had simply not mirrored them, and its
own comment says `keep step JSONLs`, so the omission was a sync gap and not a policy. 132.6 MB
pulled, every field the four layers need present, confidence populated on every step.

And the rules run on WA **unmodified**: same ruleset `8-reddit-p41p46-b1890fix`, 104 episodes per
mode, 76-84 with hits. WA reddit is the same Postmill application as VWA reddit. No code change
was needed beyond pointing `--run-dir` at the other tree.

**The lesson is the shape of the error, not the fact.** "Absent on this machine" was written down
as "does not exist", and then as "impossible", and then into a summary document as a structural
limitation of the study. The user's objection was that it did not sound reasonable, which was a
better instrument than the `find`.

WA now enters every layer. §1's matrix is updated accordingly.

## 3. WA wiring — done 2026-08-02, and what it changed

All three products take `--with-wa` and **write to `*_with_wa.*`**, because appending a cell
rewrites every consistency denominator from /6 to /7 and is not a superset of the six-cell
result. With the flag off, the six-cell outputs are byte-identical to before (verified by diff,
timestamp line excluded). Three things the seventh cell changed, none of them cosmetic:

1. **SoM's behavioural signature is VWA-only.** Fewest steps, least budget exhaustion, most
   explicit finishes are each 5/6 on the VWA grid and each 5/7 with WA: WA is the single cell
   that does not show it, and it is also the cell where SoM is not the strongest mode. Fifth
   appearance of the workload dependence, first in the behavioural layer.
2. **The load-bearing negative survives.** The four image-free modes reach the bar on **nothing**
   under either denominator, over 24 metrics. That is what licenses grouping them, and it does
   not move.
3. **"Latency is an independent axis" is VWA-only.** WA's latency span is **1.05×**, the smallest
   anywhere, against a cost span of **1.78×**, the largest anywhere. On WA the modes are within
   5% of each other on time.

And one exception to a claim made earlier in this document:

4. **WA is the only cell where a cascade operating point Pareto-beats always-rich**, which it does
   by *matching* its success rate exactly (13.46%) while escalating 30–40% of tasks rather than
   all of them, at 1.56–1.65× cost against 1.78×. Claim 5 in §5 is amended accordingly.
   ⚠️ Thresholds are swept rather than held out, 80 combinations are searched per cell, and at
   n = 104 one task is 0.96pp. Two of 80 points landing on an exact tie is what a search of that
   size produces by chance, so this is reported as an exception to state, not a result to build on.

| still open | cost | what it buys |
|---|---|---|
| **SoM replicate** | 7.8h + ~$17, armed on A100 behind the B0×WA chain | the rerun floor for the mode the fusion-premium claim is *about*. Currently borrowed from DOM and Vision |
| **`/diag` Tier-2 on WA** | a session | Tier-1 is done and its ruleset was discovered on VWA, so it finds only VWA-shaped failures. See `HANDOFF_evidence_layer_2026-08-02.md` §B |
| **§407 into the ledger** | a session | `ledger.jsonl` covers §1–§406 and has **zero** entries for today. See the handoff §C |

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
5. **Four routing formulations fail.** For the cascade on post-action confidence, no operating
   point Pareto-beats always-rich in any of the six VWA cells where the comparison is
   non-degenerate. **On WA two of 80 swept points do**, by matching its success rate exactly at
   lower cost; see §3.4 for why that is stated as an exception rather than a result.
   → the four router products + `confidence_cascade{,_with_wa}`
6. **The four image-free modes are behaviourally non-separable** across 24 metrics × 6 cells,
   while the image-bearing pair is separable mostly by construction.
   → `per_mode_four_dimension_profile` v2
7. **When the image channel uniquely wins, the text channel quits early rather than grinding.
   When the text channel uniquely wins, the image channel fails the way it fails everywhere.**
   The surviving, ungated, cross-site-comparable signal is `P27` gives-up-when-not-found at
   **2.98×** together with `P31` budget-exhausted at **0.47×**: on those tasks the text channel
   abandons instead of running out. On the other side nothing clears 1.5×.
   ⚠️ Three caveats, each of which cost an earlier version of this line.
   (a) `P6`, `P16` and `P17` are **gated off or absent on all reddit** (VWA reddit fires them at
   0.0% too), so any WA-vs-VWA-classifieds contrast on them compares a gate to a measurement.
   Retracted. (b) `P43` **is** ungated and its WA-vs-VWA-reddit contrast is real and clean,
   19.5% against 0.0% with the site held constant — **but P43 is a neutral (task × mode) label by
   its own definition**, and its docstring records a controlled dom→som test on exactly that task
   set measuring +0.00 / +1.56 / +0.00 pp from restoring the screenshot. It locates the image
   channel's advantage on visual-intent tasks; it does not explain it, and specifically does not
   license "the text channel failed because the screenshot was withheld".
   (c) WA's disagreement sets are 15 and 4 tasks. → `conditional_failure_attribution` §4
8. **The obvious routing feature has the wrong sign**, and the right one was read and dropped.
   → `routing_feature_diagnostics`
9. **Latency is a second axis, not a restatement of cost**; adding it widens the frontier in 3/6
   cells. → `multimetric_pareto`

**Two frames are live and neither has been chosen.** One organises 1–2 as the conceptual
contribution (an oracle gap is not a routing opportunity). One organises 3–4 as it (representation
choice is a deployment-time configuration). Claims 5–9 serve either. The choice is a framing
decision to be taken with the evidence in front of it, which is the state this document exists to
create.
