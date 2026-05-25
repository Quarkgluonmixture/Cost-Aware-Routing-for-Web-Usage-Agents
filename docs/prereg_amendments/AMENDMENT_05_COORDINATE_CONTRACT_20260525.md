---
amendment_id: 05
title: Qwen 0-1000 coordinate serialization contract apply (B-1860) — ESTIMAND-AFFECTING change to vision/SoM coordinate format + vision-prompt instruction-strictness relaxation; vision conditions re-collected, R3671 archived non-canonical
date: 2026-05-25
status: pre-restart witness (git tag + push @ merge d977006 + OSF upload PENDING user 2026-05-25); vision re-collection NOT yet started (gated on this witness)
parent_prereg: docs/checkpoints/pre_run/preregistration.md (status: locked)
parent_doi: 10.17605/OSF.IO/9QCWU   # DOI 1, pre-canonical-outcome-creation witness, 2026-05-18
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_01a_SCHEMA_VALIDATOR_20260521
  - AMENDMENT_02_GATE_LADDER_20260523
  - AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524
  - AMENDMENT_04_ANALYSIS_ALIGNMENT_20260524
witness_tag: prereg-amendment-05-b1860-coord-contract-20260525
provenance: R3671 (B0×vision×classifieds) /diag failure attribution 2026-05-24 → B-1860 root cause; /stress 3-AI audit (Mode A Claude + Mode B codex + Mode C gemini) on the fix; codex verify on the fix implementation (4 fix-impl bugs补修 V-F1~V-F4a; V-F4b retracted = glm sidecar half-retired). B-1861 watchdog ntfy fail-safe co-disclosed (same restart cycle, NOT estimand-affecting).
relation: >
  UNLIKE Amendment 04 (which changed NO estimand), THIS amendment IS estimand-affecting:
  it changes the coordinate serialization contract for vision/SoM modes and relaxes the
  vision-prompt instruction strictness. Disclosed BEFORE re-collection of any vision data
  under the new contract (witness precedes re-fire). The buggy-contract run R3671 is
  archived non-canonical and excluded from all primary analyses. This is a post-hoc
  data-driven remediation (HARKing), disclosed transparently; analyses on re-collected
  vision data are exploratory with respect to this change. dom/som/P-text/P-prompt are
  coordinate-light (element_id-driven) and completed conditions are retained.
---

# Preregistration Amendment 05 — Coordinate serialization contract (B-1860), ESTIMAND-AFFECTING

> **One-line**: Qwen3-VL natively emits a 0–1000 coordinate system, but the locked stack
> assumed `[0,1]`-normalized / viewport-pixel coordinates. Discovered post-hoc via `/diag`
> on R3671, this mismatch turned 13.6% of vision steps into false parse errors and
> mis-located the rest. This amendment changes the coordinate contract (an estimand-
> affecting change), relaxes the vision-prompt strictness, archives R3671 as non-canonical,
> and re-collects vision conditions under the corrected contract. Witnessed **before**
> re-collection.

## §0 — Pre-restart status (the legitimacy anchor)

Witnessed **before any vision data is re-collected under the new contract**. At witness time:
- The Gate-3 classifieds chain aborted at B0 × phantom_text (R19776, 180/224) due to an
  unrelated watchdog crash (B-1861, see §5); no condition has yet run under the new
  coordinate contract.
- R3671 (B0 × vision × classifieds), the only vision condition collected so far, was run
  under the **buggy** contract and is archived non-canonical (excluded from all analyses).
- No paper-grade vision success rate, per-cell drop-one θ_i, or pooled θ_FE has been computed
  from any new-contract data. The H1-strict 6-mode gate is unchanged and cannot fire until
  6-mode data exists.

## §1 — Why this amendment (root cause B-1860)

During paper-grade collection, post-hoc failure attribution (`/diag`) on R3671 revealed a
coordinate-system mismatch:

- **Qwen3-VL natively emits a 0–1000 coordinate system**, but the experimental stack assumed
  `[0,1]`-normalized / viewport-pixel coordinates: the vision prompt demanded `[0,1]`, the
  action validator rejected values `>1`, and the environment wrapper divided by the viewport
  dimensions.
- Consequence: 13.6% of vision-mode steps were misclassified as parse errors (vs 0.06% in
  dom/som), ~48% of vision episodes hit a parse-error cap, and the remainder suffered
  mis-located clicks (a 0–1000 value such as 728 was divided by the 1280-px viewport → 0.57
  instead of the intended 0.728).
- The vision-mode success rate (13.84%) was therefore a **coordinate-scaffold artifact**, not
  a clean measure of model grounding capability.

Evidence is reproducible from R3671's 484 emitted coordinates (e.g. y_max = 972 > the 720-px
viewport but never exceeding 1000; the same target emitted as both `[422, 476]` and
`[0.422, 0.476]` across steps, i.e. `422 = 0.422 × 1000`).

## §2 — What changed (code)

1. **Coordinate contract → per-dimension by-value.** A coordinate dimension `≤ 1.1` is
   treated as `[0,1]`-normalized (kept); `> 1.1` is treated as Qwen 0–1000 (divided by 1000).
   This is **model-agnostic** — probes confirm Qwen emits 0–1000 and Gemma emits `[0,1]`, and
   the two value ranges do not overlap, so a single by-value rule serves all three baseline
   models with no per-model branch.
2. **Single-source normalizer.** Validator, environment wrapper (4 coordinate sites:
   click / type / select_option / hover), and downstream failure-attribution / analysis all
   call one `normalize_coordinate_pair()` so the contract can never diverge across the stack.
3. **Validator** no longer hard-rejects coordinates `>1` as schema violations; value-range
   judgment is delegated to the normalizer (the `coordinate_type` enum is still
   schema-guarded against garbage labels).
4. **Vision/SoM prompt** relaxed from "output `[0,1]`-normalized coordinates, otherwise the
   action fails" to a description of the 0–1000 coordinate system; the `coordinate_type`
   field is removed from prompt and agent outputs.
5. **Grounding boundary preserved (format layer only).** Only the save/serialization format
   is normalized. There is **no** target snapping, element nearest-correction, or grounding
   rescue. A genuinely out-of-bounds coordinate (a dimension `> 1000`) is recorded as a
   grounding miss and executed as a **no-op** (it is NOT clamped to the viewport edge and
   clicked — that would mutate page state).

## §3 — Disclosures (estimand-affecting + HARKing)

1. **Instruction-strictness relaxed.** The locked preregistration's vision prompt enforced
   strict `[0,1]` normalization ("or the action fails"). This amendment relaxes that to accept
   the model's native coordinate format, deliberately reducing an instruction-following
   confound in the coordinate-grounding measurement. This is a change to the measured
   protocol, hence estimand-affecting.
2. **HARKing (hypothesizing after results are known).** This contract change was **not**
   pre-registered. It was discovered post-hoc through failure analysis of collected data
   (R3671). We disclose it as a data-driven remediation, **not** as a confirmatory test of a
   pre-registered hypothesis. Any analysis using the re-collected vision data is exploratory
   with respect to this change.

## §4 — Data handling

- **R3671 (B0 × vision × classifieds)** is archived as **non-canonical** and excluded from
  all primary analyses.
- **Vision conditions** (B0/B1/B2 × classifieds, and downstream sites) are **re-collected**
  under the new contract.
- **The incomplete phantom_text condition** (R19776, 180/224, collected under the old code)
  is **not** resumed — a single condition must be code-homogeneous, so mixing 180 old-code
  episodes with 44 new-code episodes is disallowed; it is fully re-run.
- **dom / som / P-text / P-prompt conditions** are coordinate-light (driven by element IDs;
  coordinate is only a minor fallback). Conditions already completed under the old contract
  (B0 × dom = R31194, B0 × som = R9725) are **retained**; the restart resumes "from vision".

## §5 — Co-disclosed infrastructure fix (B-1861, NOT estimand-affecting)

The same restart cycle fixes the watchdog bug that aborted the fire. The watchdog's
notification helper (`_post_ntfy`) caught only `HTTPError` / `URLError`, but a transient read
timeout to the notification service (ntfy.sh) raised `socket.timeout` / `TimeoutError` — not
a `URLError` subclass — which crashed the watchdog. The chain's liveness check then saw
"watchdog dead, runner alive" and correctly fail-safe-aborted the runner (a watchdog-less
runner has no reactive auth refresh / auto-clean = paper-grade contamination risk). The fix
makes notification failures best-effort (swallowed, never propagated). This changes **no**
measured quantity; it only prevents a transient network hiccup from killing the fire.

## §6 — Witness

- Git tag: `prereg-amendment-05-b1860-coord-contract-20260525`
- Merge commit (applies the fix to the fire branch): `d977006` (merges
  `fix-coordinate-contract-b1860` @ `3ea598e` into `diag-discover-then-freeze`)
- Bug records: B-1860 (root cause + fix + codex-verify补修) and B-1861 (watchdog fail-safe)
  in `docs/reference/master_bug_catalog.md`
- Chronicle: 实验笔记 §285 (root cause) / §286 (fix) / §287 (codex-verify补修)
- See companion `git_witness_COORDINATE_CONTRACT_20260525.txt` for the content-addressed
  commit SHA + push/upload timestamps.
