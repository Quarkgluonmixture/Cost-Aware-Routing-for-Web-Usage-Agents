---
type: handoff
status: active
created: 2026-08-02
purpose: what is left in the evidence layer, and the exact prompt for each session that should pick it up
note: prose is deliberately NOT in scope anywhere below. User decision 2026-08-02 — finish the
      evidence, then choose the frame. Two frames are live; see EVIDENCE_LAYER_SUMMARY §5.
---

# Evidence layer — what is left, and who should do it

## 0. Session split, and why

| work | where | why |
|---|---|---|
| **A. Wire WA into 3 products** | **the session that built them** | mechanical continuation; the producers were written in that context and splitting costs a re-read of ~2,000 lines |
| **B. `/diag` Tier-2 on WA** | **new session** | needs Claude sub-agents and a clean context; a genuinely different mode of work from aggregation |
| **C. Ledger + conclusions bookkeeping for §407** | **new session** | mechanical but requires reading a 26-subsection chronicle end to end, which a fresh context does better |
| **D. SoM replicate analysis** | whichever session is live when it lands (~08-03) | one command once the data exists |

Do **B** and **C** in either order; they do not touch each other. **A** is independent of both.

---

## A. Wire WebArena into the three products it is now eligible for

**Status of the enabler**: done. WA step records were absent from this machine and present on the
paper-grade host; 132.6 MB pulled, six modes × 104 episodes, every field the layers need, and
confidence populated on every step. Tier-1 diag scans for all six WA modes are at
`/tmp/diag_v8_wa/` at ruleset `8-reddit-p41p46-b1890fix`.

| product | what to add | note |
|---|---|---|
| `per_mode_four_dimension_profile.py` | a seventh cell, B1 × wa-reddit, all four dimensions | make it **additive** behind a flag: with the flag off the VWA output must be byte-identical. A baseline copy is at `/tmp/profile_before.md`. WA has no AMENDMENT_08, so its universe is the 104 tasks common to all six modes, not `expected_scored_ids` |
| `aggregate_multimetric_pareto.py` | the WA cell | it single-sources the profile, so doing the profile first may be enough. WA summaries carry `total_billed_cost_usd`, `total_latency_ms`, `total_tokens` |
| `aggregate_confidence_cascade.py` | the WA cell | B1, so confidence is 6/6 fields. Note WA's `som` SR (13.46%) is **below** `dom` (16.35%), so the cheap/rich tiers may need re-picking per cell rather than fixed globally |

**Registry debt to clear in the same pass** — `docs/reference/EVIDENCE_LAYER_AUDIT.md` §0 is missing
three entries: `multimetric_pareto`, `per_mode_four_dimension_profile` v2 (the 18→24 expansion and
the ◆ audit), and `EVIDENCE_LAYER_SUMMARY` itself.

---

## B. `/diag` Tier-2 on WebArena — prompt for the new session

> Run `/diag` Tier-2 and Tier-3 on the WebArena reddit cell (B1, six modes, 104 tasks each).
> Tier-1 is already done: scans are at `/tmp/diag_v8_wa/B1_<mode>_wa_reddit.json`, ruleset
> `8-reddit-p41p46-b1890fix`, 76–84 of 104 episodes carry hits per mode.
>
> **Why this cannot be skipped.** The v8 ruleset was discovered on VisualWebArena. Running it on
> WA finds only the failures VWA taught it to look for, so "81/104 have hits" is coverage and not
> understanding. Tier-2's job is the blind spot: the no-hit failures, whether existing hits are
> the actual cause of death, and the scaffold-bug / benchmark-FP classes the P-rules cover weakly.
>
> **Two traps this repo has already fallen into today, both recorded in 笔记 §407.25 and §407.26.**
> First, a rule returning 0.0% on a site may be a **site gate** rather than a measurement: `P6`,
> `P16` and `P17` fire at 0.0% on *all* reddit including VWA reddit, and a WA-versus-VWA-classifieds
> contrast on them compares a gate to a measurement. Before reading any zero, check the rule's
> firing rate on VWA reddit as the control. §4 of
> `docs/analysis/cross_sites/conditional_failure_attribution.md` has the seven-rule table.
> Second, `P43` is a **neutral (task × mode) label by its own docstring**, not a failure
> prediction, and 笔记 §387.10 measured restoring the screenshot on exactly its task set at
> +0.00 / +1.56 / +0.00 pp. It locates an effect; it does not explain one.
>
> **Deliverables.** The three-way triage digest the skill specifies, plus any proposed WA-native
> P-rules. Every new rule must report its firing rate on **all three** of VWA classifieds, VWA
> reddit and WA reddit before it is added, so a site gate can never again be mistaken for a
> finding. Bump `RULESET_VERSION` and rescan everything if `ALL_RULES` changes.

---

## C. Ledger and conclusions bookkeeping — prompt for the new session

**The debt.** `docs/reference/known/ledger.jsonl` holds 2,082 entries covering §1–§406.
**§407 has zero.** §407 is this session's entire output: roughly 26 subsections spanning six new
analysis products, three retractions, and a framing candidate from an external review.

> Read `docs/checkpoints/实验笔记.md` §407 end to end and register it in
> `docs/reference/known/ledger.jsonl` (chunk 9 or a new chunk 10), then update
> `docs/reference/KNOWN.md`'s count and coverage line, which currently says 2,082 and §1–§406 and
> will be stale the moment you add anything.
>
> Entry schema is one JSON object per line with `type`, `date`, `source_section`,
> `source_artifact`, `artifact_exists`, `superseded_by`, `_chunk`, plus type-specific fields
> (`decision` / `reasoning` for ADJUDICATED, `quantity` / `value` / `scope` / `caveats` for
> MEASURED, `why_dead` / `replaced_by` for RETRACTED). Match the existing style; read a dozen
> entries first.
>
> **§407 is unusually retraction-heavy and the retractions are the point.** Register at minimum:
> the three ◆ architecturally-downstream markings that were tested and did not survive as written
> (§407.22); the "WebArena has no step records, therefore impossible" claim and its refutation
> (§407.25); the three errors in that same refutation, two of them site gates (§407.26); the
> `visibility_gap` two-cell reading that reversed on the full grid (§407.22); and the corpus
> figure moving from 40% to 33.7% (§407.21 area). Each of these is a claim someone could
> otherwise rediscover and re-believe.
>
> Then annotate `docs/reference/known/conclusions/` where today supersedes something already
> written there, following the existing SUPERSEDED / closure convention.

---

## D. When the SoM replicate lands (~2026-08-03)

An armed chain on the paper-grade host fires `B0 × som × classifieds` once the B0 × WA chain
finishes, with three guards (WA complete 6/6, no classifieds runner active, site answering).
Logs and markers at `logs/som_replicate/`; completion pushes to ntfy `p79-claude`.

It closes the single largest hole in the fusion-premium claim: that claim is about the fused mode
and the rerun band it is judged against is measured on DOM and Vision. Once the data exists,
add the pair to `CLEAN_PAIRS` in `aggregate_noise_floor_inventory.py` and regenerate; then
`aggregate_fusion_premium.py` and `aggregate_label_instability.py` both pick up a floor measured
on the right arm, and `label_instability` gains a third replicated arm, which can only raise the
enrichment.

---

## F. Before anything else: the six checks that would have caught tonight

**Every error on 2026-08-02 had one shape.** A measurement instrument returned a value, and the
value was written down as a property of the world. Not one of them would have been caught by a
task list, which is what §A–§D above are. Run these as a protocol, not as advice.

| # | check | the error it would have caught |
|---|---|---|
| **1** | **A zero from a tool is not a zero in the world.** Before writing any 0.0%, name which of these you ruled out: the instrument is gated for this input · the data is not on this machine · the path or glob is wrong · the field is never populated for this arm. | `find` returned 0 for WA step files → written as "the data was never written" → written as "impossible" → written into a summary as a structural limit. The files were on the paper-grade host (§407.25). And `P6`/`P16`/`P17` return 0.0% on all reddit because they are **site-gated**, which was read as "the mechanism is absent" (§407.26). |
| **2** | **A direction from fewer than three cells is provisional.** Say "provisional" in the text, and run the full grid before it leaves a scratchpad. | `visibility_gap` read Vision as uniformly highest on two cells. On six, Vision is the **lowest in four** (§407.22). |
| **3** | **"Impossible" / "contaminated" / "unavailable" needs a measurement, not an absence.** | B0 latency was excluded from the Pareto as proxy-queue-contaminated. Measured, its per-step CV is 0.15–0.22 against the local backbone's 0.11–0.19, and it tracks tokens monotonically (§407.15). Carbon was called "not collected" from one B0 record; B1/B2 log it on every step (§407.20). |
| **4** | **State the arm count on both sides of any ratio.** | "6 representations buy 16.07pp, a rerun buys 7.6pp, 2.1×" compared five added arms to one. Arm-matched it is 7.14 against 4.91–7.59, indistinguishable (memory `same-name-not-comparable`). |
| **5** | **Never hand-edit a generated file.** Corrections go in the producer. | The ◆ audit was written into `per_mode_four_dimension_profile.md` and destroyed by the next regeneration (§407.22). |
| **6** | **A cross-site contrast needs the same-site control.** Before contrasting site A against site B on any rule or metric, report its value on the *other* cell of the same site type. | The WA-versus-VWA-classifieds contrast on `P43` was replaced by WA-reddit versus VWA-reddit, which holds the site type fixed and is the only version that survives (§407.26). |

**One more, specific to reading rules rather than metrics.** A rule can fire and still not explain
anything. `P43` carries a 1.66× enrichment and its own docstring says it is a **neutral label, not
a failure prediction**, with a controlled test measuring +0.00 / +1.56 / +0.00 pp from removing the
condition it names. Read the docstring of every rule before reading its number.

---

## G. Finding evidence nobody has thought of

Three sweeps, in this order. The first two are mechanical and the third is the one that finds
things.

**G1 — Unconsumed fields.** List every key in a step record and an episode summary, and mark which
analysis product reads it. Anything unread is a candidate metric nobody has looked at. This sweep
is how `agent_visible_changed`, `cap_hit`, `url_revisit` and `noop_inert` were found on 08-02,
after the four-dimension profile had already been called complete at 18 metrics. Known still
unread: `retry_count`, `fallback_finish`, `error_category`, `text_similarity`, `element_bbox`,
`select_option_meta`, `intervention_*`, `screenshot_timeout_recovered`.

**G2 — Coverage holes per product.** For each of the nine products, write down which cells, modes,
sites and backbones it does **not** cover, and whether that is a design choice or an omission
nobody noticed. `EVIDENCE_LAYER_SUMMARY` §1 is the start of this and it is a matrix, not a proof.

**G3 — Refutation targets.** For each of the nine claims in `EVIDENCE_LAYER_SUMMARY` §5, write the
measurement that would **refute** it, then check whether we hold that measurement. Where we do
not, that is either a limitation to state or an experiment to run. This is the sweep that finds
missing evidence rather than missing metrics, because it starts from what the paper wants to say
rather than from what the data happens to contain.

---

## H. The stress round, and what it can and cannot promise

Run `/stress` **after** §F and §G, never before: a hostile review of an incomplete evidence layer
spends its attacks on gaps you already know about.

Feed it `EVIDENCE_LAYER_SUMMARY.md` plus the nine products, and let it chain to the cross-AI
reviewers as the skill specifies. That chain earned its place on 08-02: a zero-preset external
review, given the numbers and denied the draft, computed the complement of a statistic we had
computed only half of. We had 51.1% of contested tasks flipping; it computed 2.9% on the
complement and therefore **17.4× enrichment**, which converts "benchmarks are noisy" into a
statement about structure. Zero tool calls, so it was reasoning we could have done and did not.

**What this cannot promise.** Complete and correct are different words and only one of them is
reachable here.

- **Reachable**: every number traceable to a producer that regenerates it; every claim carrying
  its scope, its arm count and its denominator; every retraction recorded where someone would
  otherwise rediscover the retracted version.
- **Not reachable**: that no result is wrong. Tonight, six products were built and three claims
  from earlier the same day were retracted, two of them within an hour of being written. The
  useful target is not zero errors, it is **that an error surfaces before it reaches the paper**,
  which is what §F is for and what the cross-AI chain is for.
- **Known to be open regardless**: two workloads cannot characterise the axis they disagree on;
  the rerun floor rests on two arms replicated once each; the fused mode has no replicate until
  the queued run lands; local absolute cost is uncalibrated. These are limitations to write, not
  gaps to close before 08-05.

---

## E. What is finished, so nobody redoes it

Nine products, each with a producer, a JSON, and a registry entry unless noted.

| product | one line |
|---|---|
| `noise_floor_inventory` | every measured rerun floor next to the arm-count-matched gain it must beat |
| `label_instability` | instability is 17.4× enriched on the tasks where the routing choice is contested |
| `fusion_premium` | the fused mode's premium as a paired test against fixed comparators, not a count |
| `confidence_cascade` | post-action escalation still loses to always-rich in every non-degenerate cell |
| `multimetric_pareto` | latency is a second axis; carbon was tested and rejected on instrument grounds |
| `per_mode_four_dimension_profile` v2 | 24 metrics; the four image-free modes reach ≥5/6 on nothing |
| `conditional_failure_attribution` | the paired diag cut, plus the site-gate table that makes zeros safe to read |
| `routing_feature_diagnostics` | the obvious feature is wrong-signed; the right one was read and dropped |
| `EVIDENCE_LAYER_SUMMARY` | the coverage map and the nine frame-independent claims |

**Prose is not in scope in any of the above.** The frame is chosen after the evidence is complete,
against `EVIDENCE_LAYER_SUMMARY` §5. Two candidates are live and neither has been picked.
