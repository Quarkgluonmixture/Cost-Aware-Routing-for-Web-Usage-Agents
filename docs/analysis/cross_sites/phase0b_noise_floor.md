---
type: analysis
status: complete
created: 2026-07-28
purpose: Phase 0b — self-oracle noise floor sweep + drop-one permutation null execution
scope_warning: every number below carries its own scope; do NOT do arithmetic across them (§302 category-error retraction, §300.2 cross-GPU drift)
---

# Phase 0b — noise floor sweep

Ran per `REBUILD_PLAN.md` Phase 0b. Offline only: reads cached artifacts, touched no
site, queued nothing on the A100.

Two things came out that the plan did not anticipate. Both are recorded before the
floor results because they change what the floor question even means:

1. the **pre-fix replicate data does not exist on disk** — the plan's "15 confounded
   pairs" are mostly empty directories, so the pre-fix arm of the sweep is not
   merely confounded, it is unavailable;
2. the **fixed-marginal permutation null had never been executed**, and running it
   returns a negative excess in **24 of 24** arm×cell combinations.

## 1. What same-mode replicate data actually exists

`run_manifest.yaml` (65 run entries, 40 distinct `(model, site, mode)`) has **19**
groups with ≥2 runs, not 15. But most second runs are **absent from disk**:

| group | second run | on disk | usable |
|---|---|---|---|
| B0·cls·{dom} | 4 × Protocol-Reset-#2 non-canonical | ✗ all four | no |
| B0·red·{dom,som,vision,p-text,p-prompt,p-som} | merged-name archives (`B0_3mode_reddit_20260422` etc.) | ✗ | no |
| B0·shop·dom | 3 × `_archive/…` | ✗ | no |
| B1·cls·{dom,som,vision} | `B1_3mode_classifieds_20260413` | ✓ dir exists | **no — 1 episode** |
| B1·cls·{p-text,p-prompt,p-som} | pre-Phase-A dirs | ✗ | no |
| B1·red·{dom,som,vision} | `B1_3mode_reddit_20260413` | ✗ | no |
| B2·{cls,red}·dom | 3-ep pilots | ✗ | no |

`B1_3mode_classifieds_20260413` is the trap: the directory **exists**, the manifest
declares `expected_n: 234`, and each of its three condition subdirs holds exactly
**one** episode (`classifieds_task_20`). A presence check on the directory passes; the
data is not there.

⇒ **No pre-fix upper bound is obtainable**, and no locally-served (B1/B2) replicate
exists at all. The plan's hope of bounding the B1 floor from pre-fix data is dead —
not because of confounding, but for want of episodes.

What does exist: **two** clean replicates in `results/repro_replicates/`, not the one
the plan lists.

## 2. Clean-pair self-oracle floor (both B0 · classifieds, n=224)

Tool: `compare_cross_run_same_condition.py` (unchanged, no new code).

| pair | SR archive → current | Δ SR | self_drop a→c | self_drop c→a | discordance | κ | flips: model-nondeterm / reset / unclassified |
|---|---|---|---|---|---|---|---|
| **dom** R31194 ↔ R21557 | 15.2% → 17.4% | +2.2pp | **4.9pp** (11/224) | **7.1pp** (16/224) | 12.1pp | 0.559 | 27 / 0 / 0 |
| **vision** R24792 ↔ R32024 | 24.1% → 25.0% | +0.9pp | **6.7pp** (15/224) | **7.6pp** (17/224) | 14.3pp | 0.614 | 30 / 0 / 0 |

Tool's own caveat, carried verbatim as required:

> ⚠️ instability proxy, NOT H1 drop-one bias correction (same-mode discordance !=
> P-SoM-vs-5-competitors false-unique; 小样本/可能混代码版本 = upper-bound risk trigger).

**The vision row reproduces §302.1 digit-for-digit** (6.7 / 7.6 / 14.3 / 0.614) — that
chronicle entry is confirmed accurate.

**The dom row is new.** `repro_replicates/README.md:52` recorded only a partial
snapshot at 88 tasks and asked for a canonical-N rerun; this is that rerun.

### Answer to the plan's open question

> *is the 6.7/7.6pp floor a B0-vision artifact, or general?*

**Not vision-specific.** Two different modes on the same (model, site) give
4.9–7.6pp self_drop. Both pairs show **0 reset/start-url contamination** — all flips
are trajectory divergence.

### Which comparison this licenses, and which it does not

Two different comparisons must be kept apart. An earlier draft of this document
conflated them and wrongly refused the first.

**Licensed — self_drop vs an H3 axis.** Both are the *same functional*, `|A ∖ B| / n`:

| quantity | form |
|---|---|
| `self_drop(run1, run2)` | \|{run1 solves} ∖ {run2 solves}\| / n |
| H3 axis-1 | \|{P-text solves} ∖ {P-SoM solves}\| / n |

Under the null that two arms are interchangeable, the axis should measure what two
runs of a *single* mode measure. That is exactly the floor. §397.10 (3) states it —
*"正是 H3 轴的估计量形式"* — and `paperA/limitations.md` already makes the comparison
in prose ("several times the axis magnitudes").

| quantity | scope | value |
|---|---|---|
| H3 axis-1 pooled θ_FE | 6 cells | **1.3528 pp** [0.799, 2.026] |
| H3 axis-2 pooled θ_FE | 6 cells | **2.0877 pp** [1.399, 2.919] |
| self_drop, **vision** pair | B0·cls, n=224 | **6.7 / 7.6 pp** |
| self_drop, **dom** pair (new) | B0·cls, n=224 | **4.9 / 7.1 pp** |

**Both axes sit below both floors.** The new dom pair matters because it shows the
floor is not an artifact of the screenshot-only arm: a text-based mode on the same
(model, site) also produces a spurious set difference several times the axis size.
`limitations.md` currently says *"We hold **one** same-condition replicate pair, on
the strongest backbone under the screenshot-only mode"* — there are **two**, and the
second is not screenshot-only. That sentence needs updating; the argument it supports
gets stronger, not weaker.

Scope discipline still applies: the floors are B0/classifieds only, the axes are
pooled over 6 cells including locally-served B1/B2 whose model-side floor may be far
lower. The floors are an **upper bound**, exactly as `limitations.md` says. No
subtraction is performed here.

**Not licensed — self_drop vs the drop-one oracle.** The tool's caveat
(`same-mode discordance != P-SoM-vs-5-competitors false-unique`) governs *this* pair,
not the one above: drop-one is a **joint** event over six arms
(`P-SoM` passes ∧ all five others fail), not a two-set difference. That comparison is
not made anywhere in this document.

The dom floor also came in slightly **below** vision despite dom carrying native
nodeId churn on top of the shared MoE component (README predicted the reverse).
13 of dom's 27 flips are step-0 element_id flips on a byte-identical bbox, i.e.
decision-harmless. No decomposition is offered here — that would be the retracted
arithmetic again.

## 3. Fixed-marginal permutation null — never executed until now

⚠️ **Scope first.** The claim below lives in `paper_drafts/section1_intro.md` (40 KB,
the older omnibus draft) and in `paper_drafts_locked/`. It does **not** appear in
`paper_drafts/paperA/` — grepping `permutation null` / `fixed-marginal` /
`drop-one oracle excess` across `paperA/` returns nothing. Current paperA reports the
drop-one superiority test as **failed** (θ_FE = 0.7897 pp, one-sided p = 0.807) and
rests its positive result on H3 instead. So this is **not** a live defect in the paper
being submitted; it is a defect in a still-edited older draft (last touched
2026-07-27) plus a process failure worth recording.

`section1_intro.md:23` states we *"empirically validate … (d) strictly-positive
drop-one oracle excess over fixed-marginal permutation null (§4.Y)"*, and footnote
`[^null-framing]` promises *"per-cell permutation excess values are reported in
§4.Y"*.

Search results before running anything:
- no `*_permutation_null.json` anywhere in `results/` or `docs/`
- neither `phantom_lift.csv` nor `meta_phantom_lift.csv` carries a `perm_null_*` column
- the chronicle mentions `permutation_drop_one_null` **only** in implementation
  records (§ around line 13298: functions added, 7 tests added) — no result anywhere
- there is no `section4.Y` / `§4.Y` in `paper_drafts/`

So B-893 (P0, 3-AI overlap) built the tool, wrote 12 invariant tests, and rewrote the
§1 prose to cite it — and the number behind the sentence was never computed.

### Executed 2026-07-28, `--permute-marginal-null --permutation-B 10000` (prereg-locked B, seed 42)

Observed drop-one reproduces `fig0c_drop_one_bootstrap_ci.csv` exactly (e.g. B0·cls
P-SoM 0.8929pp), and `marginal_counts` confirms the 6-mode universe.

| cell | arm | observed pp | null p50 | null p95 | excess over p95 | p (one-sided) |
|---|---|---|---|---|---|---|
| B0·cls | P-SoM | 0.893 | 4.911 | 6.696 | **−5.804** | 1.0000 |
| B0·red | P-SoM | 0.976 | 5.854 | 7.317 | **−6.341** | 1.0000 |
| B1·cls | P-SoM | 1.339 | 4.018 | 5.357 | **−4.018** | 0.9999 |
| B1·red | P-SoM | 0.000 | 4.878 | 6.341 | **−6.341** | 1.0000 |
| B2·cls | P-SoM | 0.446 | 0.893 | 0.893 | **−0.446** | 0.9928 |
| B2·red | P-SoM | 0.976 | 1.463 | 1.463 | **−0.488** | 0.9693 |

**Arms with positive excess: 0 of 24** (4 arms × 6 cells). The single non-significant
case is B2·cls SoM (excess 0.000, p=0.7065).

Full table: `<scratchpad>/phase0b_permnull_permutation_null.json`. Written to
scratchpad deliberately — canonical `results/phantom_paper/` artifacts were not
touched.

### This null is mis-specified for the claim it was built to test

Not a data problem — a direction problem:

- The null draws each arm's passes **independently** at its observed marginal.
- Given fixed marginals, **more overlap ⇒ fewer unique passes ⇒ smaller drop-one**.
  Shared task difficulty (easy tasks pass everywhere, hard tasks fail everywhere)
  guarantees that overlap.

**Orthogonal check, on the current data rather than on §1's prose.** `§1:21` quotes
Jaccard 0.29–0.49 vs E[J] ≈ 0.06–0.10, but from the *archive 4-mode* universe, so it
cannot carry the argument here. Recomputed on the 6-mode k=6 data, comparing observed
mean pairwise Jaccard against **the very same fixed-marginal independent shuffle the
null uses** (seed 42, B=2000 — script: `<scratchpad>/check_jaccard.py`):

| cell | n | J observed | J null p50 | J null p95 | ratio |
|---|---|---|---|---|---|
| B0·cls | 224 | 0.4100 | 0.1092 | 0.1240 | 3.75× |
| B0·red | 205 | 0.3857 | 0.0655 | 0.0810 | 5.89× |
| B1·cls | 224 | 0.3488 | 0.0448 | 0.0598 | 7.78× |
| B1·red | 205 | 0.4307 | 0.0319 | 0.0479 | 13.51× |
| B2·cls | 224 | 0.0567 | 0.0074 | 0.0262 | 7.65× |
| B2·red | 205 | 0.1707 | 0.0074 | 0.0284 | 23.04× |

**6 of 6 cells** sit above the null's *p95*, by 3.75× to 23.04× on the median. The
arms are massively more overlapping than independence. Fewer unique passes follow,
and a smaller drop-one follows from that — apples-to-apples, since the reference
distribution is constructed identically to the one the null test uses.

So observed drop-one lying *below* an independence null is the arithmetic consequence
of an overlap structure now measured on this very data. Claim (d) is not merely
unevidenced — under this null it is close to unachievable for any experiment whose
arms share a task-difficulty component.

This does **not** falsify complementarity. Observed drop-one stays positive in 22/24
arms. What it kills is *this reference distribution* as the test of it: independence
is the wrong foil, because correlated-difficulty arms are the normal case, not the
null-worthy case.

⚠️ **This judgement overturns a P0 adjudicated by 3-AI overlap (B-893) and should be
cross-AI reviewed before any prose acts on it.** What is not in doubt: the values had
never been computed, and 0/24 are positive.

## 4. AMENDMENT_07's −3.2pp against a same-estimand yardstick

§299.4 records B0 · **som** · classifieds SR 30.4% (R9725) → 27.2% (R5313),
**Δ −3.2pp**, read as evidence that sequential ids *"真消除 ID channel"*.

Two observations, no arithmetic:

**(a) Its stated attribution rests on retracted reasoning.** §299.4 justifies the
residual as *"符合 §298.3 dom 推断: id-dominant + MoE residual 10.5% + 1-2pp 拆解"* —
the linear decomposition §302 retracted as a category error. The measurement stands;
that explanatory sentence does not.

**(b) Δ −3.2pp is the same estimand as the clean pairs' Δ**, both being net SR
difference between two runs of one `(model, site, mode)`:

| | scope | Δ SR |
|---|---|---|
| AMENDMENT_07 (§299.4) | B0 · **som** · cls, n=224 | −3.2pp |
| clean pair, this sweep | B0 · **dom** · cls, n=224 | +2.2pp |
| clean pair, this sweep | B0 · **vision** · cls, n=224 | +0.9pp |

Same site, same model, same estimand — **different mode**, and the floor is not known
to be mode-invariant, so this is a flag, not a verdict. A single signed observation
cannot be separated from noise whose sign is random. The honest reading: −3.2pp is
**not comfortably outside** the run-to-run band measured on sibling modes, so it does
not on its own carry *"真消除 ID channel"*. Deciding it needs a SoM-mode replicate.

## 5. The axis-1 / axis-2 hypothesis is partly definitional

§397.10 (2) floats, explicitly unverified: axis-1 stays inside one id regime while
axis-2 crosses the AMENDMENT_07 boundary, which might explain axis-2 > axis-1.

Verified from source (`runner/main.py:2853-2860`, the authoritative dispatch): id
regime is decided **entirely by the text payload** — `[SOM_MARKS]` → seq-keyed 1..K,
AXTree → native nodeId. Therefore, from the prereg definitions:

- **axis-1** = |P-text ∖ P-SoM|: text identical (`[SOM_MARKS]`, both 1..K); the arms
  differ in **prompt style** → same regime **by construction**
- **axis-2** = |P-prompt ∖ P-SoM|: prompt identical (SoM); the arms differ in **text
  format** → crosses the regime boundary **by construction**

So "axis-2 crosses the id-regime boundary" is **not a finding — it is what axis-2 is**.
Axis-2 *is* the text-format axis, and text format *is* what sets the regime. The
hypothesis restates the design.

The non-trivial residue: is axis-2 > axis-1 driven by `[SOM_MARKS]`'s format
advantage, or mechanically by renumbering? Note that this cannot be settled by
comparing the axis magnitudes to §299.4's −3.2pp — those are unique-solve counts and
this is a net SR delta, two different estimands. It needs its own design (e.g. a
P-SoM variant carrying `[SOM_MARKS]` formatting with native ids). Also relevant is
the already-measured id channel, which §397.10 (2) correctly says should not be
re-derived: `b0_paired_idperturb_replay.py` reports id-shuffle changing the decision
in **B1 20.0% / B0 12.5%**, with within-group consistency **B1 1.000** / B0 0.867
(id-agnostic criterion: only landing on a *different physical element* counts).

## 6. Status against the plan

| plan item | outcome |
|---|---|
| clean vs pre-fix reported separately | ✅ clean ×2 done; **pre-fix arm impossible — data absent** |
| API-served B0 vs locally-served B1/B2 | ⚠️ **B0 only.** No B1/B2 replicate exists on disk |
| is 6.7/7.6pp a vision artifact? | ✅ **No** — dom gives 4.9/7.1pp on the same (model, site) |
| **"does H3 sit inside the noise?"** | ⚠️ **On the only cells where a floor exists, yes.** Both axes (1.35 / 2.09 pp pooled) fall below both B0·cls floors (4.9–7.6 pp). The floors are an upper bound and are B0-only; the locally-served cells are unmeasured. This does not overturn the *sign*, which is what `limitations.md` already claims — and no more |
| axis-1/axis-2 id-regime hypothesis | ✅ regime map verified in source; hypothesis is **definitional**, its empirical residue needs a new design |
| does this gate Phase 3? | ⚠️ **Yes — via the floor**, which is the plan's own gate. The permutation-null finding is real but scoped to a superseded draft (§3) |

## 7. What would actually be needed

1. **A locally-served (B1) same-mode replicate** — the only way to bound the floor for
   the deterministic backbone; every measurement above is B0/MoE. Queues behind the WA
   reddit run (~3 days, shared postmill container + `.auth/reddit_state.json`, B-647).
2. **A SoM-mode replicate** to decide §299.4's −3.2pp.
3. **A correctly-specified null for complementarity** — one preserving each task's
   cross-arm difficulty correlation (task-difficulty-stratified or copula-style),
   rather than independence.

Item 3 is offline and effort-bound; items 1–2 need the A100.
