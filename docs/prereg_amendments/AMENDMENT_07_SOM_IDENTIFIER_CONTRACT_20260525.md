---
amendment_id: 07
title: SoM-family element-identifier contract → deterministic sequential ids (replaces leaked CDP AXTree nodeId) — ESTIMAND-AFFECTING change to som / phantom_som / phantom_text observation ids; old-contract SoM-family runs archived non-canonical; cell re-collected
date: 2026-05-25
status: DRAFT — pre-restart witness (git tag + commit SHA + OSF upload PENDING); SoM-family re-collection NOT yet started (gated on this witness). Phase 1a paper-grade SoM-family outcome statistics NOT yet computed.
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
witness_tag: prereg-amendment-07-som-identifier-contract-20260525   # to be created at the finalizing commit
provenance: >
  ptext archive↔current run-to-run repro deep-dive (实验笔记 §292) → run-to-run noise threat to
  H1/H3 (§293, AMENDMENT_06) → SoM-family nodeId discovery (§294 + session checkpoint
  session_checkpoint_2026-05-25_runtorun_noise.md). 155-task empirical structure analysis
  (step-0, 4 runs). 4-round codex cross-AI hostile code review (Mode B, gpt-5.5 xhigh): round 1
  found the native-id fallback P0; round 2 found the fail-closed namespace P0 + recommended
  Option A over B; round 3 found a mock-protocol break (P1) + type-fallback telemetry mislabel
  (P2); round 4 final verification. GPT design-layer endorsement (axis-1 reframe, H3 caution).
relation: >
  ESTIMAND-AFFECTING, like AMENDMENT_05 (coordinate contract): it changes the element ids the
  model sees in SoM-family observations. It is the MITIGATION that AMENDMENT_06 (non-gating
  reproducibility sensitivity) left open, and it PARTIALLY SUPERSEDES AMENDMENT_06 §4: that
  section recorded "element-ID churn is NOT patched" on the premise that the churn "is part of
  real deployment". This amendment REVERSES that, because the premise turned out to be false —
  production / standard SoM interfaces use sequential selection ids, not raw browser nodeIds, so
  P79's nodeId churn is a P79-implementation artifact, not deployment-realistic noise (see §1).
  After this amendment, AMENDMENT_06's non-gating sensitivity covers the RESIDUAL stochasticity
  (provider/MoE nondeterminism, evaluator-judge), not element-ID churn (eliminated here for
  SoM-family). DOM / P-prompt retain native nodeId by design (the AXTree-native representation
  arm). Disclosed BEFORE re-collection of any SoM-family data under the new contract (witness
  precedes re-fire). Old-contract SoM-family runs are archived non-canonical.
---

# Preregistration Amendment 07 — SoM-family deterministic sequential identifier contract, ESTIMAND-AFFECTING

> **One-line**: P79's SoM-family modes (`som` / `phantom_som` / `phantom_text`) labelled each
> `[SOM_MARKS]` element with the raw Chromium CDP AXTree **nodeId**. That nodeId is a
> non-semantic, browser-internal AX-object counter that is **non-deterministic across resets**
> even when the page content and element order are byte-identical — a measurement artifact that
> injects run-to-run trajectory noise into the model's input. This amendment replaces it with a
> **deterministic 1..K sequential id** (by AXTree DFS / mark order), the standard
> SoM-interface operationalization. DOM / P-prompt keep native nodeId by design. Estimand-
> affecting; witnessed **before** re-collection.

## §0 — Pre-restart status (the legitimacy anchor)

Witnessed **before any SoM-family data is re-collected under the new contract**. At witness time:
- No SoM-family condition has run under the sequential-id contract. The Gate-3 classifieds chain
  is stopped (AMENDMENT_05/06 restart cycle), so no canonical SoM-family outcome statistic exists
  under either contract that would survive into Phase 1a primary analysis.
- The completed old-contract SoM-family runs (B0 × som = R9725, B0 × phantom_text = R2647 partial)
  are archived non-canonical and excluded from all primary analyses (§4).
- No per-cell drop-one θ_i, pooled θ_FE, or H1/H3/H10 verdict has been computed from any
  sequential-id data. The H1-strict 6-mode gate is unchanged and cannot fire until 6-mode
  sequential-id data exists.

## §1 — Why this amendment (root cause + why it is a correction, not an optimization)

**The leak.** P79 runs `observation_type = "accessibility_tree"` and builds its own SoM in
`p79/experiment/som.py` from the AXTree text; it never invokes VisualWebArena's native
`image_som` / `draw_bounding_boxes` path. The AXTree text emits each node as
`[obs_node_id] role 'label'`, where `obs_node_id` is the CDP `getFullAXTree` nodeId
(`external/visualwebarena/browser_env/processors.py:532`). P79's `_extract_text_marks` consumed
those nodeIds verbatim as the `[SOM_MARKS]` element ids and as the marked-image labels.

**The non-determinism (empirical).** Across 4 same-page runs of 155 classifieds tasks at step-0
(the reset landing page, before any agent action):
- raw nodeId byte-identical across all 4 runs: **4 / 155 (3%)** — i.e. 97% of tasks churn;
- id-stripped (content + order) identical: **155 / 155 (100%)**;
- after sequential renumbering: **155 / 155 (100%)** byte-identical.

So the page content and element **order** are fully deterministic; only the `[N]` **values**
churn. The mechanism is the Chromium AXObjectCache AXID counter: AX nodeIds are assigned at
AXObject-creation time from a per-document monotonic counter, so the same final page built across
two resets gets a near-uniform per-subtree offset plus per-element jitter for session/timing-
dependent nodes (e.g. the `Logout` link). The model is temperature-0 yet sensitive to id tokens,
so this non-semantic id churn perturbs activations and forks trajectories — a run-to-run noise
source feeding the H1/H3 oracle/set-difference estimands (the threat disclosed in AMENDMENT_06).

**Why it is a correction (not a result-driven optimization).** Standard and production SoM
interfaces use **sequential** selection ids, not raw browser nodeIds: Set-of-Mark prompting
(Yang et al. 2023) numbers regions 1..K; VisualWebArena's own `image_som` path uses
`unique_id = str(index + 1)` (`processors.py:947`); WebVoyager, SeeAct ("SeeAct Choice"),
AndroidControl, and browser-use all use flat sequentially-indexed element lists (see
`docs/literature/5.1/...Flat Indexed Lists.md`). P79's SoM-family nodeId is therefore **neither**
standard SoM **nor** deployment-realistic — it is an artifact of building SoM from the AXTree
path instead of a sequential selection interface. Sequential ids are the correct
operationalization of a SoM-style selection interface and, as a consequence, eliminate the
non-semantic id-churn noise source. This reverses the AMENDMENT_06 §4 "do not patch element-ID
churn" stance, whose premise ("the churn is part of real deployment") is falsified by the
production-convention evidence above.

## §2 — What changed (code; reviewed across 4 codex rounds)

1. **SoM-family text + image ids → deterministic sequential** (`p79/experiment/som.py`).
   `build_som_text_from_obs_text` emits `[id={seq}]` (`seq = 1..K` by mark / AXTree-DFS order)
   instead of the nodeId; the marked-image label uses the same seq. This is the single-source
   builder, so mechanistic extractors share the identical seq text. DOM and P-prompt observation
   text are unchanged (native nodeId, AXTree-native representation arm).
2. **Seq-keyed dispatch map** (`SomResult.obs_nodes_info_seq`). Each entry is a shallow copy of
   the original nodeId-keyed `obs_nodes_info` entry plus an embedded `native_element_id` (the
   original nodeId), built in the SAME single `_extract_text_marks` pass as the text (preserving
   the B-1828 phantom single-parse latency property). The original nodeId-keyed map is not
   mutated (AXTree modes still dispatch against it).
3. **Runner override** (`p79/experiment/runner/main.py`). After `prepare_observation_for_mode`,
   for SoM-family modes the runner pushes the seq map to the env's dispatch state
   (`set_dispatch_obs_nodes_info`), keyed on `obs_prep.obs_nodes_info_seq is not None` (an empty
   `{}` zero-marks map still overrides; `None` = AXTree mode, no override). Router-correct: keyed
   on the per-step `decision_mode`'s `obs_prep`.
4. **Explicit dispatch id-namespace + fail-closed translation** (`p79/envs/vwa_wrapper.py`).
   A `_dispatch_id_namespace` flag ("native" on every obs production; "seq" on the runner
   override). The bbox dispatch path uses the seq map directly (seq → union_bound → click by
   coordinate). The native-id dispatch paths (click locator-fallback, hover, and the id-based
   escape hatch that the type fallback routes through) translate `seq → native_element_id` via
   `_resolve_native_id`, which **fails closed (no-op)** when a seq is absent from the map in seq
   namespace — so a hallucinated `click [1]` (validators accept any positive id) can never be
   passed through as native AX node 1. select_option stays bbox-only (already fail-closed).
   Coordinate dispatch paths are unchanged.
5. **Guards / completeness**: `include_full_axtree=True` (a legacy path that would mix seq header
   ids with native AXTree ids) now fails loud; telemetry `element_bbox` resolves via the seq map
   when active; the `MockEnvironment` and dry-run paths implement the new env-protocol method /
   reset the namespace.

The two core correctness properties (seq consistency across text/image/dispatch; fail-closed
native translation) plus completeness of dispatch-site coverage were verified across 4
independent codex rounds; 3 new invariant unit tests pin them (cross-run determinism,
seq-dispatch identity + no native-map mutation, fail-closed resolver). Full suite green incl.
the mock integration path.

## §3 — Disclosures (estimand-affecting + HARKing + axis-1 reframe)

1. **Estimand-affecting.** The element ids the model sees in SoM-family observations change
   (nodeId → seq). This is a change to the measured input, hence estimand-affecting, parallel to
   AMENDMENT_05.
2. **HARKing (hypothesizing after results are known).** The nodeId-churn diagnosis and this
   correction were **not** pre-registered; they were discovered post-hoc through the ptext repro
   analysis (§292) and the run-to-run noise investigation (§293/§294). We disclose this as a
   data-driven contract correction, **not** as a confirmatory test. Analyses on re-collected
   SoM-family data are exploratory with respect to this change.
3. **Axis-1 is no longer "pure flattening".** The paper previously framed axis-1
   (DOM ↔ P-text, P-prompt ↔ P-SoM) as "the same AXTree content, only flattened". Under this
   amendment, the SoM-family text additionally uses a sequential selection id while DOM / P-prompt
   keep the native nodeId, so axis-1 is reframed as **AXTree-native representation → SoM-indexed
   selection representation** (a representation contract bundling structural format AND identifier
   contract), not pure flattening alone. Paper §2 / §3 prose is updated accordingly. This is the
   honest and, we argue, the more deployment-faithful framing (it matches production SoM).
4. **H3(ii) is not pure prompt-axis isolation.** P-prompt (native nodeId AXTree) vs P-SoM
   (sequential `[SOM_MARKS]`) now differ in both prompt family AND identifier/representation
   contract. The paper avoids strong causal "isolates the prompt effect" language for this contrast
   and reports it as a non-collapse / structural-region test instead.
5. **H1 asymmetric-noise disclosure.** After this amendment, SoM-family arms (P-text / P-SoM /
   som) are id-churn-stable while DOM / P-prompt retain native-nodeId churn. Part of any observed
   P-SoM drop-one "uniqueness" may therefore reflect its representation's greater stability rather
   than its format alone. We disclose this and treat it as a property of the SoM-style
   representation (consistent with the production-convention framing), with the residual
   stochasticity quantified by AMENDMENT_06's non-gating reproducibility sensitivity layer. It is
   not framed as "flat text alone" or "prompt mirage alone".

## §4 — Data handling

- **Old-contract SoM-family runs** — B0 × som (R9725) and B0 × phantom_text (R2647, partial) —
  are archived as **non-canonical** (diagnostic only) and excluded from all primary analyses.
- **SoM-family conditions** (som / phantom_som / phantom_text across B0/B1/B2 × sites) are
  **re-collected** under the sequential-id contract.
- **DOM / P-prompt / vision** are not SoM-family and their observation ids are unchanged by this
  amendment; however, for code-homogeneity of a cell's success matrix (a single cell's H1/H3/H10
  must come from one code SHA / substrate discipline), the affected cells are **re-run in full**
  on the post-amendment code rather than mixing pre/post-amendment runs within a cell.
- Replicate / sensitivity runs (AMENDMENT_06) live in the independent `results/repro_replicates/`
  namespace and never enter the canonical denominator.

## §5 — Relation to AMENDMENT_06 (the reversal, disclosed)

AMENDMENT_06 §4 recorded "element-ID churn is NOT patched" with two reasons: (a) patching would
destroy deployment realism, and (b) it would change the substrate / estimand. Reason (a) rested
on the premise that nodeId churn is part of real deployment. This amendment establishes that the
premise is false (production / standard SoM uses sequential ids — §1), so the churn is a P79
artifact and IS corrected. Reason (b) is acknowledged and handled the paper-grade way: the change
IS estimand-affecting, so it is witnessed pre-restart, the old data is archived non-canonical, and
the cell is re-collected (this amendment) — exactly the AMENDMENT_05 pattern, not a silent
in-place patch. AMENDMENT_06's non-gating sensitivity layer remains in force for the **residual**
run-to-run stochasticity (provider/MoE nondeterminism, evaluator-judge), which sequential ids do
**not** remove (they remove only the element-ID-churn component, the dominant source per §282).
The AMENDMENT_06 §4 replicate protocol item P3 (local deterministic isolation) now measures that
residual MoE/provider floor, since the element-ID component is eliminated upstream by this
amendment rather than only isolated.

## §6 — Witness

- Git tag: `prereg-amendment-07-som-identifier-contract-20260525`  (created at the docs commit; PENDING)
- Code commit SHA (the estimand change): `3a79196` on branch `fix-som-sequential-id`
- Bug record: `B-1862` in `docs/reference/master_bug_catalog.md` (cross-link B-1858 / B-12
  element-ID churn; the prior AMENDMENT_06 §4 "do not patch" decision is superseded here)
- Chronicle: 实验笔记 §295 (this session: 4-round codex review + sequential decision + impl)
- Cross-AI review artifacts: `docs/checkpoints/codex_outputs/som_sequential_review_FINAL_20260525.md`
  (round 1) · `som_optionA_review_FINAL_20260525.md` (round 2) ·
  `som_impl_verify_FINAL_20260525.md` (round 3) · `som_round4_FINAL_20260525.md` (round 4)
- Companion `git_witness_SOM_IDENTIFIER_CONTRACT_20260525.txt` for content-addressed SHA +
  push / OSF-upload timestamps (PENDING user, mirroring AMENDMENT_05/06).
