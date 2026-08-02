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
| **Four-layer profile** | `per_mode_four_dimension_profile` v2 (25 metrics) | ✅ | ✅ `--with-wa` | ✅ | complete |
| **Multi-metric Pareto** | `multimetric_pareto` | ✅ | ✅ `--with-wa` | ✅ | complete |
| **diag / failure attribution** | `cross_mode_failure_signatures` (marginal) · `conditional_failure_attribution` (paired) | ✅ | ✅ **added 08-02** | ✅ | complete |
| **2×2 ablation** | `axis_effect_size_report` · profile §2.5 non-separability | ✅ | ✅ | ✅ (the four) | **repaired 08-02** — see §2b |
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

## 2b. The 2×2 ablation row above was ✅ over an empty table — repaired 2026-08-02

The row said complete. Every contrast in the product behind it had **n = 0**, and had for weeks.

`axis_effect_size.py` imports the run registry as `scripts.analysis.lib.run_registry`, which
needs the repo root on `sys.path`; run from the command line, `sys.path[0]` is `scripts/analysis/`
and the import raises. The old code caught it, warned, and returned an empty directory map. The
script then ran to completion and wrote a full report in which every negative finding was
vacuously true over an empty set — including **"no cells show P-SoM distinct from both endpoints
simultaneously"**, which is a statement about the paper's hook. Exit status 0. Inside the JSON,
all 192 pair-count checks read `{observed: 0, expected: 203, pass: false}`: the self-check ran,
failed, and was connected to no exit — the Markdown a human reads never mentioned it.

With the input restored the finding **reverses**: P-SoM differs from **both** DOM and SoM on
**7 (metric, cell) combinations** spanning four of the six cells and all three backbones — that
is the multiplicity-corrected count (both legs clearing Benjamini-Hochberg at FDR 0.05 jointly
over 96 tests); on effect size alone it is 15, and 2 survive Holm. That is mechanism-layer
support for the independent-arm claim, and it was sitting behind a broken import.

Four things were repaired alongside it, each of the same family:

1. **The canonical scored universe was not applied.** reddit contributed 205 step files against
   a 203-task scored set, so the two AMENDMENT_08 exclusions were inside every effect size. On
   the P-SoM arm this produced a *passing* check over the wrong tasks: two identity-dropped
   episodes cancelled the two extra ones and n read 203. Compare sets, not counts.
2. **Identity mismatches now skip and report** instead of aborting the 36-cell run (2 episodes
   of 7,686, B0·reddit·P-SoM tasks 87 and 149; disclosed in a banner).
3. **The diamond's second path is rendered** — and labelled for what it is. On mean differences
   the two routes agreeing is an **algebraic identity**, so a zero residual is arithmetic, not
   evidence about a text × prompt interaction. A non-zero residual means the legs were averaged
   over different task sets; all three that miss are the 201-vs-203 P-SoM arm.
4. **`total_latency_canonical_ms` is consumed** (§G1 unconsumed-field sweep). It is
   `minus_retry − busy_wait − recovered` and `types.py:446` says it is meant to be reported
   beside the raw figure; no product read it. It matters only on the API-served arm and there
   **unevenly across modes** — B0·reddit P-text 0.890 and P-prompt 0.898 against DOM 0.966 and
   SoM 0.979 — enough to swap two modes' order. Claim 9 was re-tested against it and holds: 3/6
   either way, identical frontier membership, identical fastest mode; only the span moves
   (1.404× → 1.343×). The profile is now **25 metrics**, not 24.

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
   under either denominator, over 25 metrics. That is what licenses grouping them, and it does
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
6. **The four image-free modes are behaviourally non-separable** across 25 metrics × 6 cells,
   while the image-bearing pair is separable mostly by construction.
   → `per_mode_four_dimension_profile` v2
6b. **P-SoM is nonetheless distinct from both endpoints on 7 (metric, cell) combinations**
   spanning four of the six cells and all three backbones. That is the multiplicity-corrected
   figure (both legs clearing Benjamini-Hochberg at FDR 0.05, jointly over 96 Wilcoxon tests);
   the uncorrected effect-size-only count is 15 and **should not be quoted bare**, and only 2
   survive Holm. Non-separability among the four image-free modes and separability of P-SoM
   from DOM *and* SoM are different questions on different quantities (mode-vs-mode extremes
   versus paired contrasts).
   → `axis_effect_size_report` Tier 1 — **this replaces the "no cells" reading**, which was an
   artefact of an empty table; see §2b
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
   cells — **under either latency estimand**. Re-tested 08-02 against
   `total_latency_canonical_ms` (retry, busy-wait and recovered-screenshot subtracted): same
   3/6, identical frontier membership, identical fastest mode in every cell. Only the span
   narrows, and only on the API-served arm (B0·reddit 1.404× → 1.343×). The raw figure is what
   §3.3's cross-benchmark span comparison uses. → `multimetric_pareto`

**Two frames are live and neither has been chosen.** One organises 1–2 as the conceptual
contribution (an oracle gap is not a routing opportunity). One organises 3–4 as it (representation
choice is a deployment-time configuration). Claims 5–9 serve either. The choice is a framing
decision to be taken with the evidence in front of it, which is the state this document exists to
create.

## 6. What would refute each claim, and whether we hold it (§G3 sweep, 2026-08-02)

Written by starting from what the paper wants to say rather than from what the data happens to
contain — the only sweep of the three that finds missing *evidence* rather than missing metrics.
For each claim in §5: the measurement that would refute it, and whether we have that measurement.

| # | the measurement that would refute it | held? | consequence |
|---|---|---|---|
| 1 | the rerun floor measured **on the arm being added** — if SoM's own floor is far below SoM's marginal gain, the two are separable after all | ❌ floor is on DOM and Vision, one rerun each | this is exactly what the queued SoM replicate buys; until then the claim rests on a floor borrowed from other arms |
| 2 | (a) a second (cell, arm-pair) where the enrichment is ≈1×; (b) a **difficulty-matched** control, since "contested" is by construction a mid-difficulty band | (a) ❌ one cell, two arms · (b) ✅ **done 08-02** | (b) ran and cut both ways — see `label_instability` §"Is the enrichment just arithmetic?": the arithmetic null predicts *infinite* enrichment, so 17.4× is deflated not inflated; but inside the contested band the excess over the floor is only **1.37×** |
| 3 | one cell where fusion beats the workload-matched single channel significantly **and** by more than that cell's rerun band | ⚠️ 7 cells tested, but the band is the borrowed one from claim 1 | inherits claim 1's dependency; the 7/7 count is solid, the band is not |
| 4 | a third workload whose modality sits between the two, or contradicts the predicted sign | ❌ shopping has zero landed directories | stated limitation, not closable before submission |
| 5 | any routing formulation we did not try that wins — e.g. one using `visual_difficulty`, or the Tier-1 independence metrics | ⚠️ partial: `visual_difficulty` was diagnosed as read-but-dropped, never *fitted* | cheap and worth doing: fit one router with it and report that it does not rescue, rather than arguing it wouldn't |
| 6 | any metric on which one image-free mode reaches ≥5/6 | ✅ and it survived adding 6 metrics chosen to find differences | but all 25 metrics are ours; §G1 found **186 unread fields**, so the negative is only as strong as the metric pool |
| 6b | multiplicity correction removing the effect | ✅ **done 08-02** | 15 → **7** under BH, **2** under Holm; the 7 span four cells and all three backbones. §5 now quotes the corrected figure |
| 7 | a named mechanism on the text-wins side that the v8 ruleset cannot see — it was discovered on VWA, so it can only find VWA-shaped failures | ❌ | this is precisely what `/diag` Tier-2 on WA is for; the asymmetry may be a property of the ruleset rather than of the world |
| 8 | the feature carrying the intuitive sign in some other cell or benchmark | ✅ all six cells; WA ships no reference images so it cannot arbitrate | closed as far as this data goes |
| 9 | a different latency estimand changing the verdict; or the frontier widening simply because there are six modes | (a) ✅ **done 08-02** — canonical estimand, unchanged · (b) ❌ no permutation control | (b) is cheap: shuffle latency across modes and count how often 3/6 widening appears by chance |

**Three of these were closed on 08-02** (2b, 6b, 9a) and each was closed by *running* the control
rather than by arguing it was unnecessary. **Three remain cheap and open**: the `visual_difficulty`
router fit (5), the latency permutation control (9b), and new metrics from the unread-field
inventory (6). **Three are not cheap and are therefore limitations**: the third workload (4), the
SoM floor (1, queued), and the WA-native failure vocabulary (7, needs the Tier-2 session).
