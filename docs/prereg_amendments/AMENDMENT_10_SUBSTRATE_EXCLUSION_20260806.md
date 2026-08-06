---
amendment_id: 10
title: Scored-set protocol exclusion — VWA shopping task 345 (tier E, substrate-unreachable) removed from the SCORING denominator; collection denominator unchanged
date: 2026-08-06
status: >
  NOT PRE-DATA — see §0. The rule and its 431-URL scope evidence were committed
  before task 345's only scored episode existed, but the author already knew from
  the 2026-04-29 fire that task 345 fails. No pre-data claim is made.
parent_prereg: docs/checkpoints/pre_run/preregistration.md (status: locked)
parent_doi: 10.17605/OSF.IO/9QCWU   # DOI 1, pre-canonical-outcome-creation witness, 2026-05-18
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_01a_SCHEMA_VALIDATOR_20260521
  - AMENDMENT_02_GATE_LADDER_20260523
  - AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524
  - AMENDMENT_04_ANALYSIS_ALIGNMENT_20260524
  - AMENDMENT_05_COORDINATE_CONTRACT_20260525
  - AMENDMENT_06_REPRODUCIBILITY_SENSITIVITY_20260525
  - AMENDMENT_07_SOM_IDENTIFIER_CONTRACT_20260525
  - AMENDMENT_08_SCORED_SET_EXCLUSIONS_20260727
  - AMENDMENT_09  # prereg amendment-log row 2026-08-03 + code only; no standalone doc. Witness tag was promised in the row but never created — retro-tagged by this amendment, see §7.
  - PROTOCOL_NOTE_01_SESSION_LOST_PAPER_GRADE_20260527
  - PROTOCOL_NOTE_02_TRANSIENT_PREFLIGHT_RETRY_20260621
  - PROTOCOL_NOTE_03_RESUME_ON_ABORT_20260622
  - PROTOCOL_NOTE_04_REDDIT_IDENTITY_RESET_20260625
  - PROTOCOL_NOTE_05_ANALYSIS_ESTIMAND_CONFORMANCE_20260714
  - PROTOCOL_NOTE_06_K5_EARLY_VERDICT_20260716
witness_tag: prereg-amendment-10-substrate-exclusion-20260806
provenance: >
  实验笔记 §436 (root cause + the 431-URL probe and its two refuted false-positive
  groups) · §439 (the fire that produced the scored episode) · B-1957 (runner-layer
  `benchmark_permanent` classification) · quarantine_registry.jsonl classification
  event ts=2026-08-05T22:42:21Z classified_via=substrate_probe_431_starturls_2026-08-05
---

# Preregistration Amendment 10 — scored-set protocol exclusion (substrate-unreachable task)

**VWA shopping scoring denominator: 433 → 432. Collection denominator: unchanged at 435.**

---

## §0 — Timing, and what this amendment does *not* claim

AMENDMENT_09 could claim **pre-data** because no VWA shopping run existed on disk when
it was written. This one cannot, and the difference is worth stating precisely rather
than glossed, because the effect direction here **favours the author** (§4).

| ts (UTC) | event | bearing on the warrant |
|---|---|---|
| 2026-04-29 | §81 follow-up adjudicates task 345: *"not a §81 regression; paper footnote = 1/466 task excluded due to upstream Wikipedia ZIM data drift independent of P79 setup"*. That fire ran 466/466 (SR 16.52%, 77/466) with 345 **among the failures** | ⚠️ **the author already knew 345 fails.** This is the fact that forecloses any pre-data claim |
| 2026-08-04 00:36 | `B0_dom_shopping_…_R3561` starts | — |
| 2026-08-05 02:24 | run aborts *at* task 345 (exception path, **no evaluator score produced**) | 345 had no scored episode yet |
| 2026-08-05 22:42:21 | the rule + 431-URL scope evidence are **written** — quarantine_registry classification event `ts` | content timestamp |
| **2026-08-05 22:52:11** | **that content is committed** — `61b60e6` commit timestamp | **the witness primitive.** The selection criterion is timestamped **before** the scored episode exists |
| 2026-08-06 08:41 | resumed runner re-runs 345 → `success=False`, `benchmark_noise=True` | 345's **only** scored episode in this fire |
| 2026-08-06 11:21 | run completes 435/435 | — |
| 2026-08-06 | this amendment | written after the number was visible |

**What is therefore true**: the *criterion* predates the *measurement it is applied to*
in this fire. **What is not true**: that the criterion was chosen blind to task 345's
outcome — it was not, and the 2026-04-29 fire is the reason. Classify this as
**criterion-preregistered, outcome-known** — strictly weaker than AMENDMENT_09's
pre-data status, strictly stronger than AMENDMENT_08's fully post-hoc status.

The defense of this amendment does **not** rest on timing. It rests on §2 (the rule's
selectivity is pinned by exhaustive probe, not by judgement) and §4 (the effect is
0.026pp, and the structural argument in §6 makes the hero-metric effect exactly zero).

---

## §1 — What is excluded

### VWA shopping task 345 — tier E (config-derivable target, environment-probe-confirmed, outcome-blind)

**Rule** (evaluated verbatim over every benchmark-site task set, see §2):

> A task whose `start_url` references a substrate resource that returns **404**, where
> the 404 is **not auth-induced** and **not repairable** by any action within the P79
> setup.

**Instantiation.** Task 345 is a `shopping |AND| wikipedia` cross-site task whose
wikipedia leg points at an **image asset**, `I/Country_calling_codes_map.svg.png.webp`.
The §81 ZIM-version rewrite (2022-05 → 2025-08) works correctly — `kiwix-serve` was
confirmed running `wikipedia_en_all_maxi_2025-08.zim`, matching the mounted file. The
asset simply is not in the 2025-08 dump:

| probed | result |
|---|---|
| `I/Country_calling_codes_map.svg.png.webp` | **404** |
| `I/Country_calling_codes_map.svg.png` | **404** |
| `I/Country_calling_codes_map.svg` | **404** |
| `A/List_of_country_calling_codes` (the *article*) | 302 |
| `A/Country_calling_code` (the *article*) | 302 |

Kiwix and the ZIM are healthy; only this one asset is gone. No substrate fix restores
it short of re-importing a 2022-era ZIM — which would silently change the wikipedia
substrate for every other cross-site task, i.e. the repair is worse than the defect.

**Why tier E and not tier A or B.** The tier letter records *where the warrant comes
from*, not how strong it is:

| tier | warrant source | needs |
|---|---|---|
| A (A08/A09) | task config alone | no data |
| B (A08) | task config + agent trajectories | trajectories — **outcome-adjacent** |
| **E (this)** | task config + **environment probe** | environment state — **outcome-blind** |

Tier E needs data that tier A does not (you must probe the substrate to learn the asset
is gone), but that data contains **no agent behaviour and no score**. On the axis that
actually matters for HARKing — whether the criterion can be tuned against an outcome —
tier E is *stronger* than tier B, not weaker. It is listed as a distinct letter rather
than folded into A precisely so that "needs no data at all" keeps meaning what it says.

---

## §2 — Why this is not a hand-pick, and the instrument that nearly lied

The rule was evaluated over **all six benchmark×site task sets** by expanding
`_placeholder_mapping()` — the *same* substitution pipeline the runner uses, verbatim —
across **1466 `start_url` references / 431 unique URLs**, probed on the A100.

**First-pass result: 7 × 404 + 18 × 000 — which, taken at face value, would have put
24 tasks into the exclusion set.** Both extra groups were false positives:

| group | how it was refuted | verdict |
|---|---|---|
| 6 × `:7780/catalog/product/edit/id/…` (WA shopping_admin) | the admin **login page itself** 404s under a session-less probe; ids 1/2 also 404; a direct DB query found all 6 products present with skus intact | **auth artefact** — the probe could not reach auth, and every one of these tasks is `require_login: true` |
| 18 × `:9980/index.php?…` (classifieds) | serial re-probe returned **200 at 0.13–0.19s** for all 18 | **6-way concurrency artefact** |

After refutation the rule selects **exactly 1 task out of 431 unique URLs**.

> The instrument was systematically wrong, and **wrong in the direction that enlarges
> the exclusion set** — a bare `curl` with no `storage_state` was being used to decide
> that resources "do not exist" for tasks that all require login. Recorded because the
> failure mode generalises: before reporting a set, ask whether the number is an upper
> bound, a measurement, or an instrument fault. Same lesson as §428.6 and memory
> `feedback-absence-of-evidence-vs-measured-zero`.

**Cross-benchmark check**: the same predicate selects **0** over WA shopping (192) and
**0** over WA shopping_admin (182) once the auth artefact is removed. Of the 3 VWA
shopping tasks that cross to wikipedia — 284 / 319 / 345 — task 284 targets the article
`A/Wagyu` (present; ran clean) and task 319 has no wikipedia URL in its `start_url` at
all. The failure surface is genuinely 1/466.

---

## §3 — Two denominators, deliberately kept apart

Identical to AMENDMENT_08 §3 and AMENDMENT_09:

- **Collection denominator stays 435.** The runner keeps collecting task 345. The
  B-1834 exact-episode-count fire-completeness check is unchanged, and both sensitivity
  arms stay computable from any landed run — including runs that predate this amendment.
- **Scoring denominator drops 433 → 432.**

This is also why the runner-layer fix (B-1957) and this analysis-layer exclusion are
**two separate decisions**. B-1957 downgrades the quarantine so the condition can
finalize; it explicitly leaves the scored-set question alone:

> *"DOWNGRADE, NOT SKIP: the runner still collects the episode as an ordinary failure
> … whether it leaves the SCORED set is a separate analysis-layer decision
> (PROTOCOL_EXCLUSIONS / a future amendment), deliberately not made here."*
> — quarantine_registry classification event, 2026-08-05T22:42:21Z

This amendment is that separate decision. The layering follows §428.7.

---

## §4 — The effect direction favours the author; here is the exposure

Task 345 is a **failure**. Removing a failure from the denominator **raises** SR. This
is the attack surface, so it is quantified rather than mentioned.

Measured on the landed `B0_dom_shopping_…_R3561` episodes (nothing re-run):

| exclusion arm | denom | successes | SR |
|---|---|---|---|
| none | 435 | 51 | 11.7241% |
| A09 (463, 465) | 433 | 49 | 11.3164% |
| **A09 + A10 (463, 465, 345)** | **432** | 49 | **11.3426%** |

**Δ from this amendment = +0.0262pp.**

Four things bound that exposure:

1. **Magnitude.** +0.026pp against an H1 gate of **δ = +1.0pp** and an SE floor of
   **1.0pp** — 38× smaller than either. It cannot move any pre-registered decision.
2. **Selectivity is pinned, not asserted.** §2's 431-URL sweep leaves no room to have
   selected 345 *because* it failed; the same rule applied to every other URL in the
   corpus returns the empty set.
3. **The cheaper routes were available and not taken.** If the goal were SR, the
   `A+B+X` style informational arms of AMENDMENT_08 (which drop whole task classes)
   move SR by ~1.5pp, two orders of magnitude more. No such arm is proposed here.
4. **Hero-metric effect is structurally zero** — §6.

**Incidental corroboration of AMENDMENT_09.** That amendment was written pre-data and
predicted that tasks 463 and 465 are passable by inaction. Both landed
`success=True` in this fire. The prediction held; the two tasks are exactly the
free-credit the amendment said they were.

---

## §5 — Sensitivity, and the falsifier this amendment carries

Only the **dom** arm has landed (cell 1 of 7). The remaining 5 modes + the replicate
arm are in flight. So §4's table is a one-mode measurement, and is labelled as such.

**The mode-independence argument is structural, not empirical.** The failure occurs
while loading `start_url` — *before* the observation is constructed, therefore before
the DOM / SoM / Vision / P-* branch point. Every mode receives the same 404, and the
information the evaluator requires lives in an asset that no mode can render. Task 345
should therefore fail under all six modes.

**Falsifier (pre-committed here):** if task 345 records `success=True` under **any**
mode in the remaining arms, the premise "unrepairable and mode-independent" is false,
this amendment is **withdrawn**, and shopping's scoring denominator returns to 433. The
check is one line over the completed chain and belongs in the post-fire analysis:

```bash
grep -l '"task_id": 345' results/visualwebarena/phase1/B0_*_shopping_*/*/episodes/*_summary_v2.json \
  | xargs -r grep -l '"success": true'   # must be empty
```

---

## §6 — Effect on the H1 hero metric: structurally zero

Paper §1's hero is **drop-one oracle** — per-arm irreplaceable routing coverage, i.e.
counts of tasks that exactly one arm solves. A task that **no** arm solves contributes
**0** to every arm's drop-one credit, both before and after exclusion.

By §5's structural argument task 345 is unsolvable in every arm. It is therefore worth
0 drop-one credit in the numerator under both the 433 and the 432 denominator, and the
only thing this amendment touches is the SR denominator.

This is the opposite of AMENDMENT_08's exposure, which is worth restating for contrast:
that amendment removed reddit task 58, which **was** a unique solve by `phantom_som` —
so it subtracted drop-one credit from the paper's own hero arm in the one cell where it
moved the hero metric at all. AMENDMENT_10 has no such exposure, because it removes a
task nothing solves.

**No gate threshold, δ, SE floor, bootstrap protocol, K-of-N transparency count, or
R1–R5 mapping changes.** H1 / H3 / H10 execute as locked over a one-task-smaller
shopping scored set.

---

## §7 — Witness chain

**This amendment.**

- Git tag `prereg-amendment-10-substrate-exclusion-20260806` on the commit landing this
  doc + the prereg amendment-log row + the `PROTOCOL_EXCLUSIONS` code entry.
- Criterion witness **predating the scored episode**: commit `61b60e6`, commit
  timestamp **2026-08-05T22:52:11Z** — the B-1957 landing, which carries the rule text
  and the 431-URL result in the `quarantine_registry.jsonl` classification event
  (whose own content `ts` is 22:42:21Z, ten minutes before the commit; the **commit**
  timestamp is the witness primitive, the event `ts` is only content)
  (`classified_via=substrate_probe_431_starturls_2026-08-05`). This is the artifact
  §0's table points at; it is what makes this "criterion-preregistered" rather than
  fully post-hoc.
- Adjudication predating everything: 实验笔记 §81 follow-up, 2026-04-29.

**A gap in AMENDMENT_09's witness chain, closed here.**

AMENDMENT_09's amendment-log row (2026-08-03) states its witness as *"git tag on the
commit landing this row"*. **That tag was never created** — `git tag | grep amendment-09`
returned empty when this amendment was written. The row's own witness claim was
therefore unsatisfied for three days.

Closed by retro-tagging the commit that actually landed AMENDMENT_09's code
(`568b27f`), with the tag message recording that the tag was created 2026-08-06 and
points at a 2026-08-03 commit. The **witness primitive is the commit SHA**, which is
content-addressed and hashes its own timestamp; the tag is only a named pointer, so a
late-created pointer to an early commit still witnesses the early state — provided the
lateness is disclosed, which is the purpose of this paragraph.

AMENDMENT_09's **pre-data** status is unaffected: `568b27f` predates the first VWA
shopping run (`R3561` started 2026-08-04 00:36Z), which is the property that claim rests
on.

> Process note: this is the second time a witness has been *described* in prose without
> being *created* — memory `feedback_pre_fire_protocol_witness` exists because of the
> first. The stated rule should be that an amendment row is not complete until
> `git rev-list -1 <its own tag>` resolves.
