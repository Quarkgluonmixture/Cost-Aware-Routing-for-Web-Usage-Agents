---
type: analysis
status: rolling
created: 2026-08-02
purpose: one page saying what the evidence layer can and cannot support, before any framing is chosen
audience: self + advisor 08-03
note: this is a coverage map, NOT a findings narrative. Numbers live in the products it points at.
---

# Evidence layer — what we hold

Written deliberately **before** choosing a frame, so the frame would be chosen against coverage
rather than coverage assembled to fit a frame. **The frame was chosen on 2026-08-02 and is in
§5b**, together with the two candidates that died on the way and what killed each.

## 1. The coverage matrix

Rows are the seven evidence dimensions. Columns are the units they cover. **W** marks WebArena.

| dimension | product | VWA 6 cells | WA | 6 modes | status |
|---|---|---|---|---|---|
| **Outcome / SR** | `sr_per_mode` · `fusion_premium` | ✅ | ✅ | ✅ | complete |
| **Noise / rerun floor** | `noise_floor_inventory` · `phase0b_noise_floor` · `label_instability` | ⚠️ 1 of 6 cells | ✅ 1 pair | ⚠️ 2 of 6 arms | **SoM replicate queued** |
| **Four-layer profile** | `per_mode_four_dimension_profile` v2 (26 metrics) | ✅ | ✅ `--with-wa` | ✅ | complete |
| **Multi-metric Pareto** | `multimetric_pareto` | ✅ | ✅ `--with-wa` | ✅ | complete |
| **diag / failure attribution** | `cross_mode_failure_signatures` (marginal) · `conditional_failure_attribution` (paired) | ✅ | ✅ **added 08-02** | ✅ | complete |
| **2×2 ablation** | `axis_effect_size_report` · profile §2.5 non-separability | ✅ | ✅ `--with-wa` **added 08-02** | ✅ (the four) | **repaired 08-02** — see §2b |
| **Routing attempts** | `router_label_supply_diagnosis` · `router_triage_learnability` · `router_pooled_tier_learnability` · `confidence_cascade` | ✅ | ✅ cascade `--with-wa`; the three router products stay VWA (they need the router feature table) | ✅ | complete |
| *(features)* | `routing_feature_diagnostics` · `visual_difficulty_router` | ✅ | ❌ n/a | ✅ | complete |
| *(efficiency denominator)* | `outcome_efficiency` | ⚠️ 4 of 6 cells carry enough successes | ❌ not wired | ✅ | complete, but see §5c |

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

4. ~~**WA is the only cell where a cascade operating point Pareto-beats always-rich**~~ —
   **WITHDRAWN 2026-08-02 (§H stress P0-2).** Both winning points were tie artefacts:
   * `min_margin_min`@40% — that signal has **one distinct value across all 104 episodes**, so
     the "confidence ranking" fell through to the stable sort's task-id order. The reported
     operating point was literally the first 42 task ids. The signal is now dropped before
     ranking, and the same defect exists on `red_B2` in VWA.
   * `neg_steps`@30% — **60 episodes tie at the cutoff** and 28 of them are chosen by task id.
     Across tie orders the SR spans **8.65–14.42%**, and the reported 13.46% "exact match" with
     always-rich sits inside that arbitrary span.

   Kept struck through rather than deleted: B0 × WA is still running, so when it lands this must
   be **re-judged with the fixed script**, not resurrected from the old text and not assumed dead
   either.

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

1. **Adding an arm buys 1.97–7.14pp; adding a *rerun* buys 2.0–7.6pp.** At the one-arm margin on
   `cls_B0` these are not separable. → `noise_floor_inventory`
   ⚠️ The upper end was previously written as **8.65pp**, which that product does not contain:
   its largest arm gain is 7.14pp. 8.65 is the WA figure measured from a *different baseline*
   (the strongest image-bearing mode, 笔记 §407.3), not from the best single mode this table
   uses. Two more limits belong on the sentence: "buys" is a post-hoc **two-arm oracle-ceiling
   increment**, not an operational gain; and only two cells carry a floor at all, neither of
   them on the arm being added.
2. **Instability is enriched 3.9×–17.4× on the tasks where the routing choice is contested**,
   while aggregate SR between the same two runs moves under 2.3pp. → `label_instability`
   The range is not imprecision, it is two defensible definitions. 17.4× defines "contested"
   over all six arms — correct for the *claim*, since a router chooses among six — but the flips
   are produced by rerunning two of those six, so the same arms decide both membership and
   outcome. Rebuilding the difficulty proxy from the other four breaks the circle and gives
   **3.95×**, with the complement rate rising 2.94% → 11.88%. Neither figure may be quoted
   alone. A binomial difficulty floor was also run: it predicts *infinite* enrichment (the
   complement's floor is exactly 0), so the arithmetic deflates this number rather than
   inflating it — but inside the contested band the excess over that floor is only 1.37×.
3. **The fused mode is dearest in 5/6 cells and its pooled advantage clears neither the rerun
   band nor zero — against either comparator.** In 7/7 cells it fails to beat the single channel
   that suits the workload. → `fusion_premium`
   Two corrections, both from §H. (a) The pool now resamples **tasks once per site**, because the
   three backbones inside a site are scored on the same universe and are not independent draws;
   that moves SoM − Vision from [+0.09, +2.80] to **[−0.01, +2.91]**, so the one interval that
   excluded zero no longer does. (b) The word *significantly* is gone from the 7/7 sentence: the
   comparator is picked per cell from the same observed success rates, so those CIs do not hold
   nominal coverage. Separately, Cochran's Q rejects a common effect (I² = 59% and 77%), so the
   fixed-effect pool describes no cell in particular and the per-cell table carries the finding.
4. **Which channel to add reverses with workload modality** (VWA visual, WA text).
   → `noise_floor_inventory` §2 + `fusion_premium` §3
5. **Four routing formulations fail.** For the cascade on post-action confidence, no operating
   point Pareto-beats always-rich in any of the four VWA cells where the comparison is
   non-degenerate. **The WA exception is withdrawn** — both of its winning points were tie
   artefacts, not thresholds; see §3.4. → the four router products + `confidence_cascade{,_with_wa}`
   **A fifth formulation was fitted rather than argued away**: adding `visual_difficulty` — the
   VWA-native annotation `extract_50_features` reads and drops — to the triage feature table
   moves out-of-fold AUROC by a mean of **+0.008** over six cells, improving three, which is
   inside fold-split noise. → `visual_difficulty_router`
   Three qualifications the earlier wording lacked: the non-degeneracy rule (`cheap_sr >=
   rich_sr`) is **outcome-dependent** and labelled an exact 2.23% = 2.23% tie as "rich worse";
   the search space is the signals a cell can actually rank with, not `len(SIGNALS) × len(fracs)`,
   so "2 of 80" was denominator drift; and every number is an **offline splice** — an escalated
   task takes its outcome from a standalone rich run, whereas a real cascade would start the rich
   episode after the cheap one had already acted on a stateful site. That sequential outcome is
   unobserved in this project.
6. **The four image-free modes are behaviourally non-separable** across 26 metrics × 6 cells
   (Vision reaches ≥5/6 on nine, SoM on eight, the other four on **none**), while the
   image-bearing pair is separable mostly by construction.
   → `per_mode_four_dimension_profile` v2
   The 26th metric came from the **unread-field inventory** rather than being chosen to find a
   difference: `scroll_inert_rate`, the share of scroll actions after which the viewport did not
   move. It lands on the image side too (Vision highest 5/6, SoM lowest 5/6). Surveying that
   inventory also shrank the objection behind this claim — of the fields §G1 listed,
   `retry_count`, `screenshot_timeout_recovered`, `destructive_action_count`,
   `partial_recovery_step_count` and `unknown_failure_reasons` are 0% populated and six more are
   never written at all.
6b. **P-SoM lies off the DOM–SoM segment on 6 (metric, cell) combinations**, spanning four of
   the six cells and all three backbones. Differing from both endpoints is *not* independence —
   a mode interpolating between them also differs from both — so the count that matters requires
   the two legs to disagree in sign, i.e. P-SoM is an extremum rather than a midpoint. On
   `finish_rate@B1/reddit` it sits ~9pp below **both** endpoints while the endpoints differ from
   each other by 0.5pp. Of the 7 that survive Benjamini-Hochberg (FDR 0.05, jointly over 96
   Wilcoxon tests) 6 are off-segment and 1 interpolates; the uncorrected effect-size-only count
   is 15 and **must not be quoted bare**; 2 survive Holm. Non-separability among the four image-free modes and separability of P-SoM
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
   **The "this is just the rule vocabulary" objection is now closed** (§5 of that product): six
   candidate mechanisms computed from raw step fields, using no rule hits at all, find nothing
   either — largest enrichment **1.15×**, most *below* 1. On the tasks the text channel uniquely
   solves, the image channel fails **more blandly** than it fails elsewhere: it did not arrive,
   rather than breaking somewhere nameable.
8. **The obvious routing feature has the wrong sign**, and the right one was read and dropped.
   → `routing_feature_diagnostics`
9. **The latency ordering is not the cost ordering restated.** Mean Spearman
   ρ(cost, latency) = **−0.095** over six cells, *negative* on all three classifieds cells, and
   the cheapest mode differs from the fastest in exactly those three — a split that follows the
   **site**, not the backbone. → `multimetric_pareto`
   ⚠️ **The frontier-count argument for this claim is retracted.** "Adding latency widens the
   frontier in 3/6 cells" is not evidence: adding an axis can only weakly enlarge a Pareto
   frontier, and permuting latency across the six modes (all 720 assignments per cell) widens it
   with probability 0.75–0.83, expected **4.70 of 6**, `P(≥3 of 6) = 0.978`. The observed 3 is
   *below* chance. ⚠️ Also note per-cell exact permutation p-values on ρ are not significant —
   six modes give a Spearman test almost no power — so this is a descriptive cross-cell
   regularity, not a test. Under the canonical estimand mean ρ = −0.067, but the two estimands
   are **identical by construction on 4 of 6 cells**, so only the API-served pair tests it.

## 5b. The frame, chosen 2026-08-02 against the evidence above

The document was written before a frame was picked, so that the frame would be chosen against
coverage rather than coverage assembled to fit a frame. It has now been picked, after three
independent passes (Claude, codex, Gemini) over the same numbers with the candidates withheld
from each. Two earlier candidates died in that process and both deaths are informative:

* **"The fusion default only beats the wrong channel"** — died on a *factual* premise. It assumed
  SoM is the field's default. It is not; the mainstream deployment choice is screenshot-only,
  because it matches OS-level computer-use / GUI-grounding stacks, with DOM as the cheap
  fallback. SoM is rare. A frame resting on a wrong picture of practice fails no matter how the
  numbers come out.
* **"The screenshot is cheapest everywhere and worst somewhere"** (weakness moves between axes) —
  died on the *latency* leg. Vision is not the slowest on visual workloads: on `cls_B1` it is
  second-fastest, and its episode-level slowness on the other two is not "because it takes more
  steps" — on both local backbones its per-step latency is *lower* and only the step count makes
  the episode longer. All three reviewers reached this independently.

**The frame that survives**, stated at the strength the evidence actually carries:

> **Which channel to add reverses between these two sites, and that choice cannot be pushed down
> to the individual task.** The reversal appears on four independent functionals — success rates,
> arm-matched marginal gains, per-arm rerun floors, and paired effect sizes — across three
> backbones and two benchmarks, at 4.93–7.39pp against a measured rerun band of 0.89–2.23pp.
> Pushing the decision per-task fails for three independent reasons: the supervision is enriched
> 3.9–17.4× in instability exactly on the contested rows, five formulations fail including one
> fitted afterwards on the benchmark's own difficulty annotation, and the intuitive feature is
> wrong-signed. At the one-arm margin a new representation buys about what a rerun of the
> existing one buys.

Wording discipline, from the codex pass: say **"classifieds versus reddit"** or "coarse
site-level selection in these benchmarks", **not** "workload law". Two sites cannot identify the
causal moderator, and the three backbones share a task set — they establish model robustness of
the site interaction, not six independent observations.

Claim 3 (fusion) states **"no detectable accuracy premium over the matched single channel"**,
never "fusion does not work": the interval crossing zero is not equivalence, the pooled effects
are heterogeneous (I² = 59% and 77%), and the rerun floor was measured on DOM and Vision rather
than on the fused arm.

## 5c. Efficiency needs a denominator — and this data can only say that much

→ `outcome_efficiency`

Every efficiency figure in this project, and in the literature it sits in, is **per attempt**. A
deployment buys completed tasks. Switching the denominator to `sum(cost) / sum(success)` moves
the point-estimate ordering in **2 of the 4 cells** where success counts are high enough for the
ratio to mean anything, and moves the latency ordering in the same 2. Screenshot-only is cheapest
per attempt in 6 of 6 cells — by construction, it carries no accessibility-tree text — but
cheapest per success in only the visual cells.

**What this licenses is the denominator, not a new ranking.** Every pairwise CI overlaps: on
`cls_B0`, SoM is 0.266 [0.205, 0.355] against Vision's 0.259 [0.200, 0.348]. Four cells and
overlapping intervals cannot adjudicate which channel is more efficient; they can show that the
answer depends on a choice the field does not currently state. That is the claim, and it is a
methodological one.

It does carry one concrete consequence worth a paragraph: **the fused channel is excluded from
deployments on a per-attempt cost argument, and its per-success latency point estimate is the
best in all four of those cells.** Not a recommendation to use it — the intervals forbid that —
but a demonstration that the exclusion rests on the denominator nobody declared.

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
| 5 | any routing formulation we did not try that wins — e.g. one using `visual_difficulty` | ✅ **done 08-02** | fitted, not argued: mean ΔAUROC **+0.008** over six cells, improves 3, inside fold-split noise. The binding constraint stays row count, which no feature changes → `visual_difficulty_router` |
| 6 | any metric on which one image-free mode reaches ≥5/6 | ✅ **strengthened 08-02** | a 26th metric was built from the unread-field inventory (`scroll_inert_rate`) and lands on the image side too. The pool objection also shrank: most of the 186 unread fields are dead schema (0% populated or never written) |
| 6b | multiplicity correction removing the effect | ✅ **done 08-02** | 15 → **7** under BH, **2** under Holm; the 7 span four cells and all three backbones. §5 now quotes the corrected figure |
| 7 | a named mechanism on the text-wins side that the ruleset cannot see — it was discovered on VWA, so it can only find VWA-shaped failures | ✅ **closed 08-02** | two routes. (a) `/diag` Tier-2 on WA (§410) produced two WA-native rules at ruleset `9-wa-p47p48`; the paired cut with them still tops out at `P17` 1.39×. (b) More decisively, six candidate mechanisms computed from **raw step fields with no rule hits at all** also find nothing — largest 1.15×, most below 1. The residual is not an artifact of a VWA-shaped vocabulary → `conditional_failure_attribution` §5 |
| 8 | the feature carrying the intuitive sign in some other cell or benchmark | ✅ all six cells; WA ships no reference images so it cannot arbitrate | closed as far as this data goes |
| 9 | a different latency estimand changing the verdict; or the frontier widening simply because there are six modes | (a) ✅ · (b) ✅ **done 08-02, and it refuted the claim** | the permutation control was run: expected 4.70/6 widened, `P(≥3)=0.978`. The frontier argument is retracted and claim 9 now rests on ρ(cost, latency) = −0.095 with a site-aligned cheapest≠fastest split |

**Eight of the ten were closed on 08-02**, each by *running* the control rather than arguing
it was unnecessary, and three of those runs **refuted what they tested**: 9b killed the frontier
argument, the circularity check cut claim 2's headline from 17.4× to a 3.9×–17.4× range, and the
`visual_difficulty` fit confirmed a negative the paper had only asserted. **Nothing cheap remains
open.** What is left is three structural gaps, and they are limitations rather than tasks: the
third workload (4), the SoM floor (1, queued on the paper-grade host), and the fusion band that
inherits it (3).

A sixth, added by the same round and not yet run: the cascade's outcome for an escalated task is
spliced from a standalone rich run, but a real cascade would start the rich episode *after* the
cheap one had acted on a stateful site. No run in this project observes that sequential outcome,
so claim 5 is about an offline splice and cannot be repaired by reanalysis.

## 7. Coverage holes per product (§G2 sweep, 2026-08-02)

Each product against the seven (site, backbone) units. **·** = not covered.

| product | cls_B0 | cls_B1 | cls_B2 | red_B0 | red_B1 | red_B2 | wa_red_B1 |
|---|---|---|---|---|---|---|---|
| `noise_floor_inventory` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `label_instability` | ✅ | · | · | · | · | · | · |
| `fusion_premium` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `confidence_cascade` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `_with_wa` |
| `multimetric_pareto` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `_with_wa` |
| `per_mode_four_dimension_profile` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `_with_wa` |
| `conditional_failure_attribution` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `routing_feature_diagnostics` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ n/a |
| `axis_effect_size` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `--with-wa` **closed 08-02** |
| `axis1_microbehavior` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · (shares the axis inputs; not wired) |

Three of these are **design**, one is an **omission nobody noticed**:

- `label_instability` at 1/7 is not a hole in the product, it is the replicate inventory: only
  `cls_B0` has any same-condition rerun at all. It widens when the SoM replicate lands.
- `routing_feature_diagnostics` cannot cover WA — WebArena ships no reference images and no
  `visual_difficulty` annotation, so the feature it diagnoses does not exist there.
- ~~**`axis_effect_size` and `axis1_microbehavior` have no WA cell**~~ — **closed 2026-08-02
  for `axis_effect_size`.** The omission is worth keeping on the record because of how it hid:
  WA step records had been on disk since 08-02 and the other four step-reading products were
  wired the same day; these two were left out because the handoff listed three products and this
  dimension *looked* ✅ complete while every contrast in it was n=0. **A product that is empty
  raises no question about its coverage, because it looks finished.** A bug concealed a hole.

What the seventh cell showed, and how strong it is:

| | VWA (6 cells) | WA (1 cell) |
|---|---|---|
| dominant cascade axis | text 12 · prompt 9 · **image 19** | text 1 · **prompt 4** · image 2 |

The image axis stops dominating on WA — consistent with it being the text workload, and the
**first appearance of the modality flip in paired effect sizes** rather than in success rates or
arm-matched marginal gains. ⚠️ It is weak in the ways that matter to state: WA contributes three
effect-only combinations and **none survives multiplicity control** (n=104 against 203/224, so
less power), and a dominant-axis tally over one cell is a description, not a test.

`axis1_microbehavior` still has no WA cell. It consumes the same `STEP_DIRS`, so the wiring is
mechanical, but its ratio machinery reads a VWA-shaped macro-effects file and would need its own
pass — left open deliberately rather than bolted on.
