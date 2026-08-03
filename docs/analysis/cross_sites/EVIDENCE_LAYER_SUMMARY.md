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
rather than coverage assembled to fit a frame.

⚠️ **The frame in §5b is a dead candidate as of 2026-08-02, and so is the one after it.** §5b was
picked on 08-02 and judged too weak the same evening; a fourth candidate ("everything but the
mismatch is noise") was written up on 08-03 and judged the same way. **No frame is currently
chosen.** §5b is kept because the *reasons* the candidates died are the most reusable thing in
this document — see the two autopsies inside it, and `deliverables/FRAME_2026-08-02.md` for the
fourth. Read §5's claims, not §5b's paragraph, as what the layer holds.

## 1. The coverage matrix

Rows are the seven evidence dimensions. Columns are the units they cover. **W** marks WebArena.

| dimension | product | VWA 6 cells | WA | 6 modes | status |
|---|---|---|---|---|---|
| **Outcome / SR** | `sr_per_mode` · `fusion_premium` | ✅ | ✅ | ✅ | complete |
| **Noise / rerun floor** | `noise_floor_inventory` · `phase0b_noise_floor` · `label_instability` | ⚠️ 1 of 6 cells | ✅ 1 pair | ⚠️ 2 of 6 arms | **SoM replicate queued** |
| **Four-layer profile** | `per_mode_four_dimension_profile` v2 (26 metrics) | ✅ | ✅ `--with-wa` | ✅ | complete |
| **Multi-metric Pareto** | `multimetric_pareto` | ✅ | ✅ `--with-wa` | ✅ | complete |
| **diag / failure attribution** | `cross_mode_failure_signatures` (marginal) · `conditional_failure_attribution` (paired) | ✅ | ✅ both WA cells, **v11 rescan 08-03** | ⚠️ **vision column not co-tabulable** — see §1b | complete |
| **2×2 ablation** | `axis_effect_size_report` · profile §2.5 non-separability | ✅ | ✅ `--with-wa` **added 08-02** | ✅ (the four) | **repaired 08-02** — see §2b |
| **Routing attempts** | `router_label_supply_diagnosis` · `router_triage_learnability` · `router_pooled_tier_learnability` · `confidence_cascade` | ✅ | ✅ cascade `--with-wa`; the three router products stay VWA (they need the router feature table) | ✅ | complete |
| *(features)* | `routing_feature_diagnostics` · `visual_difficulty_router` | ✅ | ❌ n/a | ✅ | complete |
| *(efficiency denominator)* | `outcome_efficiency` | ⚠️ **6 of 8** cells carry enough successes (both B2 cells do not) | ✅ both cells **wired 08-03** | ✅ | complete, but see §5c |
| *(validity)* | `reddit_sidebar_leakage_audit` · `leakage_sensitivity` · `offsite_navigation_audit` | ✅ | ⚠️ off-site ✅, leakage unaudited | ✅ | **added 08-03** — see §3b, §8b, claim 9 |
| *(ex-ante partition)* | `visual_intent_routing` | ✅ | ⚠️ degenerate (5/104 flagged, none solved) | ✅ (dom vs som/vision) | **new 08-03** — see claim 10 |
| *(deployment classes)* | `representation_class_comparison` | ✅ | ✅ | ✅ (3 classes) | **new 08-03** — see claim 12 |
| *(is routing worth it)* | `rule_routing_pareto` | ✅ | ❌ (needs the flagged set) | ✅ | **new 08-03** — see claim 5c |
| *(routing, the other half)* | `router_objective_ordering` · `router_triage_learnability` | ✅ | ❌ needs the router feature table | ✅ | **uncited until 08-03** — see §5b |

## 1b. How to read a diag per-rule number (ruleset v11, 2026-08-03)

Three constraints landed with the v9 → v11 rescan. They do not change any number in this
document; they change what a number is allowed to mean, so they sit here rather than in a
footnote.

1. **A per-rule table is a distribution of symptoms, not of causes.** `P36` (51%) and `P31`
   (50%) are the two largest rows in most cells and are **risk markers**, not death causes —
   established by causal verification on 10 cases across both benchmarks. Do not write "the
   dominant failure mode is X" from a rule frequency. The rules that *have* been causally
   verified say so in their docstrings (`P49` is one; see claim 7).
2. **The `vision` column cannot be co-tabulated with `dom`/`som`.** `P2` and `P4` read
   `element_bbox`, which vision's clicks carry no locator metadata for, so their vision cells
   are **structural zeros, not measurements**; `P36` on vision covers only `type` steps. A table
   that puts vision beside the other modes silently compares a gate against a measurement — the
   same defect §5 claim 7 caught with `P6`/`P16`/`P17` on reddit.
3. **`P43`'s "neutral label" framing holds only on reddit.** It was measured there (+0.00 /
   +1.56 / +0.00pp from restoring the screenshot) and the label was written from that. On
   classifieds the same rule's hits behave completely differently — see **claim 10**, which is
   what that discrepancy turned into.

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

Re-verified against disk 2026-08-03 rather than carried forward. Two entries are new and one of
them had never been written down anywhere.

**4a. Structural — cannot be closed before submission**

- **A third workload.** `results/visualwebarena/phase1/*shop*` is **zero directories** — not zero
  episodes, zero directories. Two workloads show a sign change and cannot characterise the axis
  it turns on. Claim 4 stays "between these two sites", permanently.
- **⚠️ NEW: the two benchmarks share one application.** `_lib_paper_grade_gates.sh:462-467`:
  WA reddit **is** the `vwa-reddit` container — same `postmill-populated-exposed-withimg`
  image, same port, same account (`storage_state` is byte-identical across both benchmarks'
  reddit task files). So "holds across two benchmarks" is, on the reddit axis, **one
  application with two task sets**. The task sets do differ substantively (104 vs 203 tasks,
  SR 2-3× apart), so this is not nothing — but it is not application-level generalisation,
  and it should never be written as though it were. A second application would need WA
  shopping or VWA shopping, both blocked on the same missing Magento DB restore
  (`wa_reset_supported()` returns 1 for everything but reddit; `reset_wa_sites.sh` is a
  scaffold whose roadmap still assumes a separate WA docker stack — an assumption §387.3
  retracted, which means the real remaining work is **one** restore implementation that
  would unlock both).
- **⚠️ NEW: no cross-family control on WebArena.** `results/webarena/phase1/B2_*` is **zero
  runs**. Both WA cells are Qwen (235B and 4B), so the *cross-family* control exists only on VWA.
  Any statement of the form "this holds across benchmarks" is, on the family axis, **broken** —
  it holds across benchmarks for one family and across families on one benchmark, never both.
  This left a trace in the code: `axis_effect_size` builds `B2 × wa_reddit` as a permanent
  all-`n=0` shell, and a `len()` over that grid reports 9 cells for an 8-cell study (fixed
  2026-08-03; the wrong count is how the gap surfaced).
- **The sequential cascade outcome.** Claim 5's escalated tasks take their outcome from a
  standalone rich run; a real cascade starts the rich episode *after* the cheap one has acted on
  a stateful site. No run in this project observes that, and no reanalysis can produce it.
- **Calibrated per-accelerator energy.** `use_pynvml: true` is configured but every step records
  `source: psutil_profile` at ~66W on a device rated several times that, so the carbon column is
  wall-clock in other units and is reported as uninformative rather than as an axis.
- **Absolute local cost.** The per-token constant for the locally-served backbones was derived
  for a different accelerator than the runs were served on. Within-cell ratios are unaffected
  because it is a single multiplier; absolute dollar figures for B1/B2 are uncalibrated.

**4a-bis. Two verdicts the ledger is holding for tonight's replicate**

`known.py` carries these as `CLAIM_UNVERIFIED`, both blocked on the same missing measurement,
both filed months before the replicate was queued. Recording them here so the data lands on a
question rather than on nobody:

- **§242** — *"drop-one oracle's 1.7–3.3pp must be shown to exceed the stochastic noise floor
  (B0 cls shows 12% per-task flip); run the same condition twice and measure the SR standard
  deviation."* `why_unverified: 重跑尚未做`. The SoM replicate **is** that rerun.
- **§293** — *conditional*: if H1 strict clears only 1–2pp **and** the replicate noise floor is
  also 1–2pp or more, the hero wording must be **downgraded** from "P-SoM has stable unique
  task-solving contribution" to "pre-registered single-run oracle evidence, with reproducibility
  caveat." `why_unverified: 依赖尚未做的 replicate-calibrated sensitivity`.

§293 is a **trigger, not a task**: when the replicate lands it either fires or it does not, and
the answer is mechanical. It should be evaluated before the number is quoted anywhere else.

**4b. Open and closable, with what each buys**

- **Replicates on the other four arms, and more than one rerun each.** Every instability figure
  is a lower bound from two arms replicated once. The SoM replicate (running 2026-08-03, ~8.6h +
  ~$17) closes the one that matters most: claims 1 and 3 currently rest on a floor **borrowed
  from DOM and Vision**, while claim 3 is *about* the SoM arm.
- **Wiring, hours each**: `axis1_microbehavior` and `cost_per_mode` have no WA cell;
  `conditional_failure_attribution` has no `wa_B0`. `mechanism_per_task_report` (425 lines) is
  computed and unintegrated — what it adds over `axis_effect_size` is a framing call, not wiring.

**4c. Validity decisions that are not ours to make unilaterally** (preregistration-level)

- **What to do about the 6 leaked reddit successes.** They flip one verdict (§3b). Exclude the
  tasks, reweight, or disclose and keep? Tasks 58 and 160 went through AMENDMENT_08 for less.
- **Whether to audit the WA cells for the same defect.** Two hand-traced episodes came back
  *earned* (§8b), which is evidence about two episodes.
- **Whether off-site steps stay in the latency estimand.** 1.05–2.13% of reddit steps, 0.00–0.16%
  of classifieds steps, and they are *faster* than on-site ones (claim 9).

**4d. A defect class, not a list of defects**

Five instances of "a conclusion hardcoded in the producer, next to a table that derives the same
quantity" were found and fixed on 2026-08-03: `aggregate_fusion_premium.py:317` (said an interval
no longer excluded zero while its own table said it did), `per_mode_four_dimension_profile.py`
(annotation column read `6/6` beside a count column reading `8/8`), the same file's frozen NOTES
registry (now disclosed rather than silently stale), `aggregate_confidence_cascade.py:430` (**wrong
denominator *and* wrong fact** — claimed Vision is cheapest in 6/6 cells; on `wa_B0` it is DOM),
and `axis_effect_size.py:1055`.

**The sweep that found them only covered the `n/6` shape.** Hardcoded *mode names*, *ratios* and
*directions* have not been swept, and the cascade instance proves those can be wrong on the fact
and not merely on the denominator. Also unresolved: `aggregate_h10_pareto.py`'s **5/6 deployment
threshold is a six-cell design** — whether it becomes 7/8 at eight cells is a preregistration
decision, deliberately not taken here (see claim 6 for why carrying the literal numerator across a
denominator change reverses a load-bearing negative).
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
3. **The fused mode is dearest in most cells and its pooled advantage clears the rerun band in
   neither comparator.** In 8/8 cells it fails to beat the single channel that suits the
   workload. → `fusion_premium`
   Three corrections. (a) The pool resamples **tasks once per site**, because the three
   backbones inside a site are scored on the same universe and are not independent draws.
   ⚠️ **An earlier version of this line said clustering removed the one zero-excluding
   interval. That is no longer true and the sentence was stale in a specific, instructive way:
   it was hardcoded prose inside `aggregate_fusion_premium.py` (line 317), so every rerun
   reprinted it unchanged while the data-driven table three lines above it said the opposite.**
   Fixed 2026-08-03 — the verdict is now derived. Current state: `SoM − Vision` clusters to
   **[+0.06, +2.93]** and does exclude zero, but the lower bound is +0.06pp against a rerun-band
   floor of 0.89pp, so **excluding zero here is not a premium claim** — the band column is the
   one that answers the question, and it reads *no*.
   (b) The word *significantly* is gone from the 8/8 sentence: the comparator is picked per cell
   from the same observed success rates, so those CIs do not hold nominal coverage.
   (c) Cochran's Q rejects a common effect (I² = 59% and 77%), so the fixed-effect pool describes
   no cell in particular and the per-cell table carries the finding.
   **No cell wins out of band.** `cls_B0`'s +2.23pp is the largest, and it is *exactly* the
   rerun band's upper edge — both quantities are 5/224, so this is an equality, not a margin.
3b. **The only cell that showed fusion *significantly beaten* rests on accumulated site state.**
   → `leakage_sensitivity` (new 2026-08-03)
   `red_B2` SoM − DOM = −2.96pp [−5.91, −0.49] was the one interval in the eight-cell table
   lying entirely on the negative side. `require_reset` is a no-op on reddit, so subscriptions
   accumulate across a run's 205 episodes; `reddit_sidebar_leakage_audit` finds **3 of the 8
   successes** behind `red_B2`·DOM were credited without the episode ever visiting the forum the
   evaluator reads — 37.5%, the highest share of any arm. Setting those three to 0 (denominator
   unchanged: an attempted-and-unaccomplished task is a 0, not a missing row) moves the contrast
   to **−1.48pp [−3.45, +0.49]**, which crosses zero. The modality reversal does **not** move:
   `red_B0` and `red_B1` SoM − Vision both still exclude zero, and `red_B0` moves further from
   it. 4 of the 6 leaks are on DOM, so removing them helps the fused arm — the direction that
   disfavours the paper's own caution, which is why it is reported rather than quietly adopted.
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
5b. ⚠️ **"Routing fails" is too strong, and the counter-evidence was already in the repo.**
   → `router_objective_ordering` (138 lines, never cited in this document until 2026-08-03) +
   `router_triage_learnability` (cited by name only, its result absorbed into the negative)
   The failures above are all **which-mode** routing — choose among six arms per task. There is a
   second half to the decision, and it behaves differently:
   * **Oracle triage** — keep the best-SR mode, but spend the cheapest on tasks nothing solves —
     gives **zero SR change** at **−9.5% to −30.6% cost in 6 of 6 cells**. Its label is binary
     (solvable / not) and is **defined on every task**, where the which-mode label exists only on
     the 16–97 tasks something solves. Claim 5's stated binding constraint is row count; triage
     is the formulation that constraint does not bind.
   * **Learned triage, honest out-of-fold** (nested threshold, task-held-out 5-fold): `cls_B1`
     reaches **+0.00pp SR at −4.5% cost** — a real win, not an oracle. The other three carrying
     cells give −0.45pp/−0.5%, −1.48pp/−12.9%, −1.97pp/−10.9%. AUROC 0.651–0.717 clears the
     best-single-feature baseline in 5 of 6 cells (`red_B2` is 0.483 and fails).
   So the honest statement is: **which-mode routing fails on this data; triage routing does not,
   and one cell shows it working out-of-fold.** The size of the prize is small and only one cell
   is lossless, but "four formulations fail" reads as a closed door over a result the repo holds.
5d. **The ceiling on any router is set by how few tasks anything solves.**
   → same product, never stated in this document
   Per cell, the share of tasks **no mode solves** is 56.7% / 73.9% / 75.4% / 88.2% / 92.9% /
   92.6%. And the share with **more than one** solver — the tasks where a per-task choice even
   exists — is 68/224, 36/203, 29/224, 17/203, 4/224, 3/203, i.e. **1.5% to 30%**. On `cls_B2`
   and `red_B2` the entire per-task routing space is 3–4 tasks. This is the most direct available
   measurement of why per-task routing has so little to work with, and it is a property of the
   difficulty of the benchmark rather than of any method.
   Adjacent, and also uncited: the evaluator emits **two distinct values** over all 7,686 scored
   episodes (0 and 1; 8.39% are 1). There is no graded target to regress on — a property of the
   benchmark's design, not of this pipeline. → `evaluator_score_granularity`
6. **The four image-free modes are behaviourally non-separable** across 26 metrics × **8 cells**
   (Vision reaches the bar on nine, SoM on five, the other four on **none**), while the
   image-bearing pair is separable mostly by construction.
   ⚠️ **The bar is ≥7/8, not ≥5/8** — 83%, the same proportion the six-cell version meant by
   ≥5/6. Carrying the literal numerator across the denominator change would set the bar at 63%,
   at which **P-text clears it on two metrics** and this load-bearing negative appears to break.
   It does not break; the threshold was misread. Same data, wrong denominator, opposite claim.
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
   each other by 0.5pp. **Recomputed on eight cells (2026-08-03):** of the **7** that survive
   Benjamini-Hochberg (FDR 0.05, jointly over **128** Wilcoxon tests) **6 are off-segment** and 1
   interpolates; the uncorrected effect-size-only count is **23** and **must not be quoted bare**;
   **1** survives Holm (was 2 on six cells — the extra tests cost it). All 7 BH survivors sit in
   VWA cells; **the two WA cells contribute none**, at n=104 against 203/224. Non-separability among the four image-free modes and separability of P-SoM
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
   ⚠️ **Updated 2026-08-03, ruleset v9 → v11: the residual now has a name on WA, and only on WA.**
   Rescanning at ruleset `11-intent-text-fallback` puts **`P49` SUBMIT_PAGE_ANCHOR_MISCLICK at
   3.61×** on this side — the first rule ever to clear 1.5× here, and it did not exist in v9.
   Postmill's navbar carries a *Submit* **link** (`target_tag='A'`, bbox y=0) that the AXTree
   cannot distinguish from the form's real submit button; clicking it reloads a blank form and
   silently discards what was typed, so the model retypes and reclicks in a self-reinforcing
   loop. It is **causally verified** (WA som tasks 610/614), which separates it from `P36`/`P31`
   — those are risk markers, not death causes.
   **But all 8 hits are in the two WA cells** (B0 5, B1 3); the six VWA cells contribute zero,
   and 8 is exactly the `MIN_HITS` floor. So: on WA the residual has a mechanism; **on VWA it is
   still unexplained**, and the sentence above stands there unchanged.
8. **The obvious routing feature has the wrong sign**, and the right one was read and dropped.
   → `routing_feature_diagnostics`
9. **The latency ordering is not the cost ordering restated.** Mean Spearman
   ρ(cost, latency) = **−0.014** over eight cells, *negative* on all three classifieds cells, and
   the cheapest mode differs from the fastest in **5 of 8**. → `multimetric_pareto`
   ⚠️ **The "follows the site" reading did not survive the eighth cell.** On the six-cell grid
   the split was clean: cheapest≠fastest in all three classifieds cells and in none of the three
   reddit cells. The two WA cells are *reddit* — the same Postmill application — and both land on
   the **≠** side. So the split is now cls + WA-red against VWA-red, which no longer separates by
   site. It may separate by benchmark, by task set, or by nothing; two benchmarks cannot tell
   these apart. State the observation (5/8, listed by name) and drop the explanation.
   ⚠️ Two further things the site-level framing was silently carrying → `offsite_navigation_audit`
   (new 2026-08-03). (a) **Reddit episodes leave the benchmark.** Postmill is a link aggregator,
   so 1.05–2.13% of reddit steps have an `obs_url` on the live public internet (imgur, news
   sites); classifieds is 0.00–0.16%. Those steps are *faster*, not slower — commercial CDNs beat
   a Postmill container sharing a host with the agent — so the distortion runs opposite to the
   intuition, at roughly 1% of one cell's environment time. Small, but undisclosed and one-sided.
   (b) **The containers differ more than the modes do.** Median on-site `env_step` is 4.5–5.8s on
   classifieds and 6.6–11.3s on reddit: **1.69×** before any agent behaviour enters. This does
   not threaten claim 9, which compares modes *within* a cell — but "follows the site" is
   carrying infrastructure as well as workload, and no between-site latency number should be
   quoted bare.
   ⚠️ **The frontier-count argument for this claim is retracted.** "Adding latency widens the
   frontier in 3/6 cells" is not evidence: adding an axis can only weakly enlarge a Pareto
   frontier, and permuting latency across the six modes (all 720 assignments per cell) widens it
   with probability 0.75–0.83, expected **4.70 of 6**, `P(≥3 of 6) = 0.978`. The observed 3 is
   *below* chance. ⚠️ Also note per-cell exact permutation p-values on ρ are not significant —
   six modes give a Spearman test almost no power — so this is a descriptive cross-cell
   regularity, not a test. Under the canonical estimand mean ρ = **+0.007**, but the two estimands
   are **identical by construction on 5 of 8 cells** (the locally-served ones have no retry,
   busy-wait or screenshot timeout to subtract), so only the API-served cells test it at all.

10. **A 0-token text rule says, in advance, where the screenshot pays — and where it does not.**
   → `visual_intent_routing` (new 2026-08-03)
   The predicate is a regex over the task intent plus "task carries no reference image". Both
   read the task config: no model call, no episode, no tokens. On classifieds it flags 71 of 224
   tasks, and the screenshot is worth **+22.54pp [+9.86, +33.80]** on them against **+0.65pp
   [−5.88, +7.84]** on the other 153 — **significant on the flagged set, not on the rest**, at
   `cls_B0`; `cls_B1` is +16.90 [+8.45, +25.35] against +1.31. That is a ~34× concentration of an
   effect whose selector costs nothing to evaluate.
   Three things make this stronger than it first looks. (a) **`vision` beats `som` on the flagged
   set** (+22.54 vs +19.72), so the rule identifies *needs a screenshot*, not *needs SoM
   annotation* — and `vision` is the cheapest-per-attempt arm in 7 of 8 cells, i.e. the rule
   points at the cheap solution. (b) **The classifieds numbers are out-of-sample**: the rule was
   written for reddit (its docstring cites "64 reddit tasks"), and its classifieds hits were
   incidental and never examined until now. (c) Unlike drop-one oracle, which needs all six arms
   run before it can say anything, this is decidable **before any episode starts**.
   ⚠️ Three limits, in order of cost. **It needs capability to cash in**: `cls_B2` gets +1.41pp on
   the same flagged set — the rule is right, the backbone cannot use it (2 of 3 backbones, not
   3 of 3). **It is site-specific and the sign flips**: `red_B0` gets **−3.17pp** on its flagged
   tasks, the screenshot *hurts* — the modality reversal appearing in a third functional.
   **The counts are small**: ~70 tasks, and the largest gap is 23 successes against 7.
   ⚠️ **This is not `P43` as shipped.** The production rule adds
   `if summary.get("success"): return []`, which makes its hit set outcome-dependent — tasks the
   text arms solved are excluded by construction, so an arm comparison *inside* that set measures
   the selection, not the screenshot. The product strips that filter and keeps only the ex-ante
   terms. The originally reported "dom 9.9% → som 29.6%" came from the outcome-filtered set; it
   survives the correction (identically, as it happens, because every flagged task fails in at
   least one text arm) but the two must not be conflated.
   ⚠️ It is also **not a router**: the partition is fixed and known in advance, which is exactly
   what makes it cheap, but nothing here learns or adapts it.
   ⚠️ **WA cannot test this** (wired 2026-08-03). The predicate flags 71/224 on classifieds and
   63/203 on VWA reddit but only **5/104** on WA — WebArena words its intents differently and the
   regex, written against VWA phrasing, mostly misses. Worse, **no mode solves any of those 5**,
   so the WA rows come out `+0.00pp` with a zero-width interval: that is *no information*, not a
   measured null, and the product labels it `degenerate` rather than printing a number that would
   be quoted as one. The WA cells are a coverage note; the result rests on classifieds.
11. **Macro action frequencies converge while per-step decisions keep diverging.**
   → `axis1_microbehavior` (in the repo since before 08-02, never cited in §5 until now)
   The decision-quality-over-macro-frequency ratio is **>1 in 6 of 6 cells** (B0 red 1.43 / cls
   2.52 · B1 red 2.84 / cls 1.34 · B2 red 4.07 / cls 2.03), verdict **generalizes**. It is the
   only product that directly tests the hook's weakest joint: even where the aggregate action mix
   of a text arm approaches DOM's — which happens most on classifieds — the *per-step* choices
   still differ, measured by URL-path Jaccard, target-hit differential, keyword repetition and
   first-action divergence rather than by outcome.
   ⚠️ Six cells only. `MACRO_JSON` points at the six-cell `axis_effect_size.json` and `STEP_DIRS`
   comes from the VWA-only run registry, so wiring WA needs its own pass (the WA macro
   denominators do exist and are healthy — `B0/wa_reddit` has the largest median |text effect| of
   any cell at 0.2549 — so this is worth doing, not blocked). **`B2 × wa_reddit` can never be
   filled**: B2 never ran WebArena.

12. **Three deployment classes, and the one the field is betting on never wins.**
   → `representation_class_comparison` (new 2026-08-03)
   Web agents ship in three shapes and this project's six modes map onto them cleanly:
   **no-image** (DOM, P-text, P-prompt, P-SoM), **vision-only** (Vision — the
   computer-use-aligned line), **hybrid** (SoM). Grouping the four is licensed rather than
   assumed: they clear the ≥83% consistency bar on **none** of 26 metrics over 8 cells.
   Two things hold at 8/8. (a) **`vision-only` is never the sole best class** — hybrid takes
   4 cells, no-image takes 3, and the remaining cell is a *tie* between vision and hybrid at
   2.23% (5/224), i.e. at the floor. (b) **Which class wins reverses with the workload**:
   hybrid takes VWA, no-image takes **both** WebArena cells, by **13.46pp** on `wa_B0`
   (P-text 35.58% against SoM 22.12%). That is claim 4 restated as the decision a deployment
   actually faces — *ship vision at all?* — rather than as a per-mode ranking.
   ⚠️ **The tempting number is arm-count, not evidence.** Dropping the whole no-image class
   costs +2.68 to +22.12pp against +0.49 to +4.46pp for the others — but it has four arms and
   they have one each. Arm-matched (add ONE arm to each cell's best), the largest gain lands
   on no-image 4 times and vision-only 3 times: **class membership does not predict marginal
   value**. The product computes the unmatched figure anyway, labelled, because a reader
   would otherwise compute it without the caveat.
   ⚠️ Everything here is an oracle over landed runs — what a perfect chooser could have had.
5c. **And routing on the best available signal still loses to not routing.**
   → `rule_routing_pareto` (new 2026-08-03)
   Claim 10 found a 0-token signal worth +22.54pp on the subset it flags. Turning it into a
   policy: flagged→Vision, else→DOM. On `cls_B0` that yields **24.55%**, exactly
   `always-DOM` + 7.14pp (= 71/224 × 22.54pp, the arithmetic closes) — **and still below
   `always-Vision`'s 25.00%**, because the screenshot does not *hurt* on the unflagged tasks
   (+0.65pp), so "use the image only when needed" is beaten by "just always use it".
   On the 3-axis (success, cost, latency) frontier a rule policy survives in 5 of 6 cells,
   but surviving means *nothing dominates*, not *preferable*: on `cls_B0` the three rule
   policies all sit between `always-SoM` and `always-Vision`, worse on every axis than one or
   the other, and `always-P-prompt` at 19.64% is equally "on the frontier".
   **This is a stronger negative than claim 5's.** There the diagnosis was label scarcity
   (15–97 trainable rows). Here the signal is free, ex ante, and large — and routing on it is
   still not worth doing, because the arm the rule routes *to* is already the right arm to
   route everything to. The one exception is `red_B2`, where the rule policy beats every
   fixed mode (5.42% vs 3.94%) at lower cost — on a cell whose successes are single digits.

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
ratio to mean anything, and moves the latency ordering in the same 2.

⚠️ **The eighth cell broke the constructional reading of the cheapness result.** This paragraph
used to say screenshot-only is cheapest per attempt in **6 of 6** cells *by construction* — "it
carries no accessibility-tree text". It is now **7 of 8**, and the exception is `wa_B0`, where
DOM is cheaper. A construction has no exceptions. What the exception shows is that a screenshot's
token cost can exceed a text serialisation's when the pages are text-light and the images are
not, so cheapness-by-construction was a contingent fact wearing an argument's clothes. Cheapest
**per success** is Vision in only the three classifieds cells; on both WA cells it is P-text, and
on `red_B0`/`red_B2` it is DOM.

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
| 3 | one cell where fusion beats the workload-matched single channel significantly **and** by more than that cell's rerun band | ⚠️ **8** cells tested, but the band is the borrowed one from claim 1 | inherits claim 1's dependency; the 8/8 count is solid, the band is not. **`cls_B0`'s +2.23pp is exactly the band's upper edge** (both are 5/224) — an equality, not a win |
| 4 | a third workload whose modality sits between the two, or contradicts the predicted sign | ❌ shopping has zero landed directories | stated limitation, not closable before submission |
| 5 | any routing formulation we did not try that wins — e.g. one using `visual_difficulty` | ✅ **done 08-02** | fitted, not argued: mean ΔAUROC **+0.008** over six cells, improves 3, inside fold-split noise. The binding constraint stays row count, which no feature changes → `visual_difficulty_router` |
| 6 | any metric on which one image-free mode reaches the ≥83% bar (≥5/6, now ≥7/8) | ✅ **strengthened 08-02** | a 26th metric was built from the unread-field inventory (`scroll_inert_rate`) and lands on the image side too. The pool objection also shrank: most of the 186 unread fields are dead schema (0% populated or never written) |
| 6b | multiplicity correction removing the effect | ✅ **done 08-02** | 15 → **7** under BH, **2** under Holm; the 7 span four cells and all three backbones. §5 now quotes the corrected figure |
| 7 | a named mechanism on the text-wins side that the ruleset cannot see — it was discovered on VWA, so it can only find VWA-shaped failures | ✅ **closed 08-02** | two routes. (a) `/diag` Tier-2 on WA (§410) produced two WA-native rules at ruleset `9-wa-p47p48`; the paired cut with them still tops out at `P17` 1.39×. (b) More decisively, six candidate mechanisms computed from **raw step fields with no rule hits at all** also find nothing — largest 1.15×, most below 1. The residual is not an artifact of a VWA-shaped vocabulary → `conditional_failure_attribution` §5 |
| 8 | the feature carrying the intuitive sign in some other cell or benchmark | ✅ all six VWA cells; neither WA cell ships reference images so they cannot arbitrate | closed as far as this data goes |
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

Each product against the **eight** (site, backbone) units — `wa_red_B0` landed 2026-08-03.
**·** = not covered.

| product | cls_B0 | cls_B1 | cls_B2 | red_B0 | red_B1 | red_B2 | wa_red_B1 | wa_red_B0 |
|---|---|---|---|---|---|---|---|---|
| `noise_floor_inventory` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ⚠️ margin only, no pilot → no floor |
| `label_instability` | ✅ | · | · | · | · | · | · | · |
| `fusion_premium` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `confidence_cascade` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `_with_wa` | ✅ `_with_wa` |
| `multimetric_pareto` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `_with_wa` | ✅ `_with_wa` |
| `per_mode_four_dimension_profile` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `_with_wa` | ✅ `_with_wa` |
| `outcome_efficiency` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `axis_effect_size` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ `--with-wa` | ✅ `--with-wa` |
| `offsite_navigation_audit` **new 08-03** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| `conditional_failure_attribution` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ **closed 08-03** (v11 rescan) |
| `visual_intent_routing` **new 08-03** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ n/a | ❌ n/a |
| `cost_per_mode` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · | · |
| `axis1_microbehavior` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | · | · |
| `leakage_sensitivity` **new 08-03** | n/a | n/a | n/a | ✅ | ✅ | ✅ | ⚠️ unaudited | ⚠️ unaudited |
| `routing_feature_diagnostics` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ n/a | ❌ n/a |
| `visual_difficulty_router` | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ n/a | ❌ n/a |

Three of these are **design**, one is an **omission nobody noticed**:

- `label_instability` at 1/8 is not a hole in the product, it is the replicate inventory: only
  `cls_B0` has any same-condition rerun at all. It widens when the SoM replicate lands.
- `routing_feature_diagnostics` cannot cover WA — WebArena ships no reference images and no
  `visual_difficulty` annotation, so the feature it diagnoses does not exist there.
- ~~**`axis_effect_size` and `axis1_microbehavior` have no WA cell**~~ — **closed 2026-08-02
  for `axis_effect_size`.** The omission is worth keeping on the record because of how it hid:
  WA step records had been on disk since 08-02 and the other four step-reading products were
  wired the same day; these two were left out because the handoff listed three products and this
  dimension *looked* ✅ complete while every contrast in it was n=0. **A product that is empty
  raises no question about its coverage, because it looks finished.** A bug concealed a hole.

What the WA cells showed, and how strong it is (the table below is the seventh cell; the eighth landed 2026-08-03 and contributes no BH survivor either):

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

## 8. Products this document was not citing (§8 sweep, 2026-08-03)

The three sweeps above (§G1 unread fields, §G2 coverage, §G3 refutation) all start from what is
*in* the document. None of them asks the complementary question: **what did the repo compute that
never reached this page?** Mechanically — every `docs/analysis/cross_sites/*.md` against the
names this document mentions — **11 products were uncited**. Most are minor. Four were not.

| product | lines | why it matters |
|---|---|---|
| `router_objective_ordering` | 138 | carries the oracle-triage result that **contradicts** claim 5's scope. Now §5b |
| `mechanism_per_task_report` | 425 | the largest uncited artifact. E1 click-target Jaccard, E2 trajectory divergence, E3 confidence, E4 action vocabulary, per task, all six modes, both sites. **Still not integrated** — see below |
| `evaluator_score_granularity` | 66 | the evaluator is binary over 7,686 episodes. A precondition of every routing negative. Now §5c-pre |
| `reddit_sidebar_leakage_audit` | 84 | 6 environmentally-credited successes; drove the new `leakage_sensitivity` and flipped one verdict. Now §3b |
| `amendment08_sensitivity` | 99 | classifieds' scored-universe SHA is byte-identical pre- and post-amendment, so **every classifieds number in the paper is provably untouched** by the exclusion protocol. A free robustness statement nobody was making |
| `cost_per_mode` | 68 | the second cost estimand (B0 API dollars vs B1/B2 electricity) and the argument that they are **different classes** presented side by side, never as one ratio |
| `failure_modes_per_cell` · `diag_digest_index` · `cross_mode_failure_taxonomy_B0_classifieds` · `power_analysis` · `router_covariate_baseline_2026-07-05` | — | superseded, index-only, or pre-dating the current cell set |

**The shape of this miss is the same one §2b recorded.** There, a product looked ✅ complete while
being empty, and *a finished-looking thing raises no questions*. Here, a product that was never
mentioned raises no questions either — a coverage table can only audit rows someone put in it.
Both failures are invisible to any sweep that starts from the document rather than from the disk.

**`mechanism_per_task` is deliberately still open.** It is 425 lines of per-task mechanism
evidence on the axis contrasts, computed in 2026-08-03's refresh, and integrating it properly
means deciding what it adds over `axis_effect_size` and `conditional_failure_attribution` —
which is a framing decision, not a wiring one. Flagged rather than absorbed.

### 8b. WA's state-carrying tasks — checked, and the suspicion did not hold

Ten of WebArena reddit's 104 tasks modify persistent site state: five subscribe to a forum and
are scored by reading the sidebar (identical in form to the VWA defect in §3b), five create a
forum. They carry **~25% of each WA cell's successes on 9.6% of its tasks** — B0 solves 65% of
them against a 25.8% overall rate, and tasks 596/597/598 are solved by **all six modes**, which
on a benchmark at that success rate normally means the environment supplied the answer.

It did not. Hand-tracing the step records: on task 597 the agent sees a `Subscribe` button
reading *"No subscribers"*, clicks it, and the next step observes `Unsubscribe` — it subscribed.
On task 582 it walks the full `/create_forum` form and lands on `/f/Cyberpunk`. **Earned, both.**

The plainer explanation holds: subscribing and creating are single-control operations with an
unambiguous UI target, and are simply easier than finding information. Part of WA's higher SR is
a task-mix property. ⚠️ This is a **two-episode hand check, not an audit** — `leakage_sensitivity`
is marked ⚠️ unaudited on both WA cells, and the mechanism (`require_reset` gated on classifieds)
applies to any Postmill site. What is now known is that the first two suspects were innocent.

One thing the trace did surface: on task 595 the agent left the benchmark entirely and spent 25
of 30 steps on `supercluster.com`. That generalised into `offsite_navigation_audit` — see claim 9.
