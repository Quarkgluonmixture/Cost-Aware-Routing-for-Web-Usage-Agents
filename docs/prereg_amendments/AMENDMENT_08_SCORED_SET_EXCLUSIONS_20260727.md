---
amendment_id: 08
title: Scored-set protocol exclusions — reddit tasks 160 (tier A) + 58 (tier B) removed from the SCORING denominator; collection denominator unchanged
date: 2026-07-27
status: POST-HOC / OUTCOME-VISIBLE — see §0. Landed Pass-1 data existed and had been analysed before this amendment was written. Disclosed as such; no pre-data claim is made.
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
  - PROTOCOL_NOTE_01_SESSION_LOST_PAPER_GRADE_20260527
  - PROTOCOL_NOTE_02_TRANSIENT_PREFLIGHT_RETRY_20260621
  - PROTOCOL_NOTE_03_RESUME_ON_ABORT_20260622
  - PROTOCOL_NOTE_04_REDDIT_IDENTITY_RESET_20260625
  - PROTOCOL_NOTE_05_ANALYSIS_ESTIMAND_CONFORMANCE_20260714
  - PROTOCOL_NOTE_06_K5_EARLY_VERDICT_20260716
witness_tag: prereg-amendment-08-scored-set-exclusions-20260727
provenance: >
  /diag Tier-2 reddit deep-dive 2026-07-27 (实验笔记 §387.7-§387.9, bugs B-1889 + B-1892)
  surfaced task 160's positive-check-free eval and task 58's parametric-knowledge
  shortcut. User decision "(a)" 2026-07-27 selected pre-registered exclusion +
  disclosure over the alternatives (report as-is / report as FP transparency only).
relation: >
  CHANGES the scoring denominator and the sample-pool composition for the reddit
  cells: 205 -> 203. That is squarely inside the AMENDMENT_## namespace as defined
  by PROTOCOL_NOTE_01 §0 ("reserved for changes that move H1/H3/H10 estimand
  definitions, scored_task_count, observation-id contracts, eval-context, or
  sample-pool composition"), so this is AMENDMENT_08 and not PROTOCOL_NOTE_07.
  The classifieds cells are untouched and provably so (§4). The collection
  denominator (episodes a run owes) stays at 205 by design (§3).
---

# Preregistration Amendment 08 — scored-set protocol exclusions

> **One-line**: Two reddit tasks are removed from the set the paper computes success
> rates over, because their evals cannot distinguish the capability the task names
> from something else. The runner keeps collecting them, so every number in this
> amendment — with and without, at each warrant tier — comes from the same landed
> episodes.

## §0 — Post-hoc status (the honest anchor)

Amendments 01-07 were written before the data they governed existed. **This one is
not.** At witness time all 18 Pass-1 reddit conditions and all 18 classifieds
conditions had landed, `/diag` had been run over all 36, and the effect of these
exclusions on the pooled SR was known to the author before the criteria were
written down.

That is HARKing-adjacent and is not dressed up as anything else. What is offered
instead of a pre-data claim:

1. **Both criteria are properties of the task config, not of our results.** A
   reviewer can evaluate tier A with no data at all, and tier B from the VWA task
   file plus any agent's trajectories.
2. **Each criterion is applied uniformly over a pre-defined class**, and the class
   is stated before the selection: tier A over all 210 reddit tasks, tier B over
   all 40 cross-site reddit tasks. Neither is a per-task judgement call.
3. **Full sensitivity is reported** (§5), including the arm that reproduces the
   pre-amendment universe bit-for-bit.
4. **The effect runs against the paper's claim** where it has any effect at all on
   the H1 hero metric (§6). An exclusion reverse-engineered to help would not.

## §1 — What is excluded

### reddit task 160 — tier A (config-derivable, outcome-blind)

```
intent : Can you subscribe to all subreddits that start with the letter 'i' and
         have a female usb to male lightning connector image in their top 3 posts
         of all time?
eval   : program_html, single locator on the sidebar subscription list,
         required_contents = { must_exclude: [IAmA, InternetIsBeautiful, iphone] }
```

There is no `must_include` and no `reference_answers`. The eval asserts only that
three specific forums are **absent** from the sidebar. An agent that does nothing
scores 1; an agent that performs the task correctly also scores 1; the two are
indistinguishable. The intended subscription is never checked.

**Uniform rule (evaluated over all 210 reddit tasks)**: a `program_html` task whose
every `required_contents` block carries only `must_exclude` keys, hence is passable
by inaction. Reddit task 160 is the only task in the site that matches.

This is the §139.8 N/A defect with the sign flipped. §139.8 excluded tasks that are
un-passable under the no-N/A-exit prompt because they carry zero discriminative
signal; this task carries zero discriminative signal because it is trivially
passable. The pre-registered precedent is the same one.

Empirically the task is not even *consistently* passed: 13 of 18 conditions score 1
and 5 score 0 — the 5 failures are agents that over-subscribed into one of the
excluded forums. So the only capability the task measures is "did not take a wrong
action", which is not the capability its intent names.

### reddit task 58 — tier B (config-suggestive, trajectory-confirmed)

```
sites      : [wikipedia, reddit]
start_url  : __REDDIT__/f/dataisbeautiful/38990 |AND| __WIKIPEDIA__/.../Landing
intent     : Who is the author of the most popular novel adapted anime in year 2012?
eval       : string_match, exact_match = "Reki Kawahara"
```

The reference answer is a fact about Sword Art Online that a pretrained VLM can
emit without loading either page.

**Uniform rule (evaluated over all 40 cross-site reddit tasks)**: a task whose
successes reach the reference string without the trajectory ever loading a host
from the task's own `sites` list beyond the start site. Applied to the class, it
selects exactly one task:

| | cross-site tasks | with ≥1 success | successes | successes that loaded the 2nd site |
|---|---|---|---|---|
| all 40 | 40 | 3 | 11 | 2 |
| task 58 | — | ✓ | **9** | **0** |
| tasks 49 + 66 | — | ✓ | 2 | 2 |

Task 58 takes 9 of the 11 cross-site successes observed across the 18 conditions,
and none of those 9 ever loaded `localhost:8888`. The two successes on the other
solvable cross-site tasks both did. B1 dom reaches the reference answer in **3
steps**.

**This is not an environment gap.** Wikipedia was served and reachable throughout:
2265 steps across the cross-site episodes landed on `localhost:8888`. Agents can
and do reach the second site; on this task they do not need to.

**Why tier B is a weaker warrant than tier A, stated plainly.** Tier A needs no
data. Tier B needs trajectories to confirm — the config alone tells you the answer
is famous, not that models actually shortcut it. The rule is also outcome-*adjacent*
in that it quantifies over successes. It is reported as its own sensitivity arm for
exactly that reason (§5), and a reader who rejects tier B can read the `A`-only
column and lose nothing else.

## §2 — What is NOT excluded, and why

The cross-site class as a whole is **not** excluded, though it would be a defensible
scope decision: 40 of 205 reddit tasks (19.5%) declare a second site, and they run at
**1.91% SR (11/576 episodes) against 8.25% (245/2970) for the reddit-only tasks** —
a 4.3× gap; the 8 shopping-cross tasks are 0/144.

Excluding the class would raise reddit pooled SR from 6.94% to 7.86%. It is left in
because (a) the paper's site scope is a data-collection fact, not a task-eligibility
criterion, and the cross-site tasks are part of the benchmark's reddit split as
published; (b) the depressed SR is a *finding about cross-site grounding*, not an
eval defect — tasks 49 and 66 show the class is solvable as designed. The arm is
reported as informational in §5 so a reviewer can see that the tier-B single-task
exclusion is not standing in for a much larger unstated one.

## §3 — Two denominators, deliberately kept apart

| | reddit | classifieds | who uses it |
|---|---|---|---|
| **collection** — episodes a run owes | 205 | 224 | `validate_run`, `run_registry.is_complete`, `paper_grade_check`, `validate_fire_manifest`, `active_processes`, `clear_tasks`, `glm_cell_autoupdate` |
| **scoring** — what a rate divides by | **203** | 224 | every SR / lift / oracle / figure in the paper |

The excluded tasks are dropped at **analysis** time only. `load_tasks` is unchanged
and the runner still collects them. This is a choice, and the reasons are:

- **the fire-completeness contract stays identical across the amendment boundary.**
  Every landed run holds 205 reddit episodes. Had the exclusion gone in at task-load,
  future runs would produce 203 and the `episodes == scored_task_count` exact check
  (B-1834) would need to become version-aware — a new way for real contamination to
  hide.
- **both sensitivity arms stay computable from any run**, not only from the runs that
  predate the amendment. A load-time exclusion makes the "without" arm unrecoverable
  from new data.
- **nothing in the fire import path moves.** `p79/experiment/analysis.py` is imported
  only by `p79/cli/analyze_experiment.py`, never by the runner; `scripts/analysis/**`
  is not in the fire path at all.

Implementation:

- `p79/experiment/tasks.py` — `PROTOCOL_EXCLUSIONS` registry + `ProtocolExclusion`
  (task_id / tier / uniform rule / reason / amendment) + `protocol_excluded_task_ids`.
- `p79/experiment/analysis.py` — `paper_scored_task_count()` (scoring); existing
  `scored_task_count()` keeps its value and becomes explicitly the collection count.
- `scripts/analysis/lib/canonical_task_universe.py` — `collected_task_ids()` (was the
  body of `expected_scored_ids`) and `expected_scored_ids()` now returns the scored
  set. `protocol_excluded_in_universe()` lets aggregators tell an expected
  collected-but-unscored episode from real contamination.
- `scripts/analysis/aggregate_sr_fp_per_mode.py` — `exact_set` / `extra_ids` /
  `completeness_ratio` updated so a 205-episode reddit cell against a 203-task scored
  set still reads complete, and a genuinely unexpected task ID still does not.
- `tests/test_amendment08_scored_set_exclusions.py` (14 tests) +
  `tests/test_toolchain_chunk1_20260714.py` (3 tests, two SHAs pinned).

## §4 — Classifieds is provably untouched

`expected_scored_ids("classifieds")[1]`, old code vs new code on the same task files:

```
OLD (HEAD)            classifieds (224, b0f3b8b0b002843981cf12a8dc2db5479d73ac7ca03190bda37c27a26a508d0e)
NEW (working tree)    classifieds (224, b0f3b8b0b002843981cf12a8dc2db5479d73ac7ca03190bda37c27a26a508d0e)
OLD (HEAD)            reddit      (205, 41b1a918356a563c3b94c2db0b1d3c3d589a68340fca938617ef2b52b63f837b)
NEW (working tree)    reddit      (203, 1ce29c8b9fbee6a49cafb95eff8381fcb1b1aea566eea8fa444a65f1a6152c92)
```

The classifieds SHA is byte-identical, so no classifieds number in the paper can
move. Both SHAs are pinned in tests. `expected_scored_ids("reddit", tiers=())`
reproduces `41b1a918…` exactly, which is what makes the sensitivity arms a
comparison rather than a re-derivation.

**Stale-artifact detection is automatic.** The reddit universe SHA is recorded in
each artifact's `outcome_provenance.canonical_task_universe_sha256` and cross-checked
between artifacts (e.g. `router_prior_baselines.py:1563`). Any artifact generated
under `41b1a918…` now fails that check against a freshly generated one instead of
silently mixing denominators. Every reddit-touching artifact must be regenerated;
that regeneration is the k=6 reload step, not part of this amendment.

## §5 — Sensitivity (full table: `docs/analysis/cross_sites/amendment08_sensitivity.md`)

Regenerate with `.venv/bin/python3 scripts/analysis/amendment08_sensitivity.py`.

| arm | reddit universe | reddit pooled SR | classifieds pooled SR |
|---|---|---|---|
| `none` — pre-amendment | 205 | 6.94% | 10.19% |
| `A` — tier A only | 204 | 6.62% | 10.19% |
| **`A+B` — primary** | **203** | **6.40%** | **10.19%** |
| `A+B+X` — informational, also drops the cross-site class | 164 | 7.86% | 10.29% |

Pooled over both sites: 8.64% → **8.39%** (−0.25pp). Per-condition deltas run from
−0.97pp to +0.14pp; the sign is not uniform because removing a task the condition
passed lowers its SR while removing one it failed raises it.

**The one cell worth flagging**: `B2_phantom_prompt_reddit` had exactly one success
in 205 episodes, and it was task 160. Post-amendment that cell is **0.49% → 0.00%**.
Its nominal SR was entirely the artifact. This is disclosed rather than smoothed —
it is the sharpest single illustration of why the exclusion is warranted, and it is
also the least flattering number in the paper.

## §6 — Effect on the H1 hero metric

H1's hero is the drop-one oracle, not the SR. An excluded task can only move it if
that task was solved by exactly **one** mode inside a cell; otherwise it cancels
between the with-arm and the without-arm. Across the 6 (site, model) × 2 tasks
exposure points:

| cell | task | modes solving | uniquely solved? |
|---|---|---|---|
| reddit/B0 | 58 | 3/6 | no |
| reddit/B0 | 160 | 2/6 | no |
| reddit/B1 | 58 | 5/6 | no |
| reddit/B1 | 160 | 6/6 | no |
| reddit/B2 | 58 | **1/6 (phantom_som)** | **YES** |
| reddit/B2 | 160 | 5/6 | no |

**One** exposure point, and it removes drop-one credit from `phantom_som` — the
paper's hero arm — in the B2 reddit cell. The amendment moves the H1 hero metric
*against* the claim it supports, in the one cell where it moves it at all.

No gate threshold, δ, SE floor, bootstrap protocol, or R1-R5 mapping is changed by
this amendment. H1/H3/H10 execute exactly as locked, over a scored set that is two
reddit tasks smaller.

## §7 — Witness chain

- Amendment doc: this file.
- Witness note: `docs/prereg_amendments/git_witness_SCORED_SET_EXCLUSIONS_20260727.txt`.
- Git tag: `prereg-amendment-08-scored-set-exclusions-20260727`, created at the
  commit carrying this doc + the code + the tests + the sensitivity artifact.
- Prereg decision log: `preregistration.md` Appendix A, 2026-07-27 entry.
- Chronicle: 实验笔记 §387.14 (decision) + §387.15 (this land).
- Bugs: B-1889 (task 160 eval), B-1892 (task 58 shortcut) in `master_bug_catalog.md`.

The witness does **not** precede the data — it cannot, and §0 says so. It precedes
the regeneration of every reddit-touching analysis artifact under the new universe
SHA, which is the boundary this tag actually anchors.
