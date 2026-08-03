<!--
=====================================================================
AAAI-27 REPRODUCIBILITY SUPPLEMENT — BUILD STATUS (2026-07-14)
=====================================================================
Status: STRUCTURE COMPLETE; existing protocol material backfilled.
The remaining work is numerical population and submission packaging.

TODO BEFORE SUPPLEMENTARY DEADLINE:
  1. Replace every ⟨TBD: verdict-day slotsheet⟩ from the frozen
     verdict-day artifact; do not copy numbers manually from prose.
  2. Refresh the quarantine-registry snapshot, classify or disclose every
     outstanding G8 item, and populate the execution-audit table in S1.
  3. Populate the per-cell compute/energy/cost table in S4.
  4. Freeze final F1--F4 renders; retain interim/archive renders under
     distinct names with status and provenance metadata.
  5. Replace all conditional gate language with the realized H1/H3/H10
     branches, then rerun the anonymity checklist in S6.
  6. Strip this build block and all section-end source comments before
     submission; retain them in the internal source version.
=====================================================================
-->

---
title: "Supplementary Material for The Phantom Routing Space"
bibliography: ../paper.bib
---

# S1 Execution Protocol

## S1.1 Sequential execution and resets

The unit of execution is one (site, backbone, mode-or-router) condition. Conditions are run sequentially within each site; no two backbones share a site's mutable benchmark state concurrently. Pass 1 runs the six fixed observation modes, and Pass 2 runs the learned-router condition only after the fixed-mode data needed for training are complete. Every condition begins from a container reset to the locked benchmark snapshot, followed by authentication and site-health checks. Pass 2 uses the same reset boundary as Pass 1. A failed reset, authentication check, evaluator initialization, or provenance check stops the launch rather than producing a scored episode.

Tasks follow one fixed canonical order in every condition. This makes task identity and order position pairable across modes, but it does not make substrate state byte-identical across the sequence. We therefore retain task index, wall-clock anchors, reset/authentication events, and infrastructure covariates for order-position and footprint sensitivity analyses. The canonical comparison is paired by task identity; condition order is not re-randomized after outcomes are observed.

## S1.2 Reddit clean-per-task identity

Reddit contains a destructive username-change task that can alter the shared test account and thereby make later fresh logins fail. At the start of every Reddit task, before authentication refresh and before the agent trajectory, an idempotent database update restores both identity fields to their seeded values. The update is a no-op when the identity is already correct. This defines the Reddit estimand as clean per-task capability and prevents a capable model's success on the destructive task from selectively contaminating its later episodes. The intervention is pre-trajectory and is excluded from agent cost and latency accounting; its success/failure telemetry is retained.

## S1.3 Fail-closed quarantine and bounded recovery

Infrastructure failures are never silently converted into agent failures. An exception-path episode is marked `needs_reevaluation`, appended to the quarantine registry, and excluded from canonical aggregation until resolved by the registered recovery path.

Recovery is layered and bounded:

1. Provider 5xx responses receive capped transport-level backoff. Exhaustion propagates as a paper-grade abort; it is not recorded as score zero.
2. Episode-level retry is permitted only for structured `auth` or `network` failures with `steps == 0`, before any agent action. The current bound is six retries with capped backoff. Provider 5xx exhaustion, any mid-episode failure, evaluator failure, and non-transient failure remain aborts. Each rescued attempt records retry count, class, attempt index, and trigger in the canonical summary and trajectory-event log.
3. On abort, the interrupted partial episode is preserved under a forensic stale name. It cannot be consumed as the authoritative episode.

The `steps == 0` boundary is load-bearing: retrying a partially executed trajectory could replay site mutations and condition selection on a failed stochastic rollout. Pre-flight retry instead waits for the first valid rollout to begin and leaves the scientific estimand unchanged.

## S1.4 Resume after abort

The default after an interrupted condition is a fresh run. Reddit is the witnessed exception because direct inspection of all 210 upstream task definitions found no cross-task object dependencies: no task consumes an object created by another task. On Reddit only, a compatible aborted run may therefore resume from its breakpoint. Completed task summaries are retained; the breakpoint's `needs_reevaluation` episode is forcibly rerun after recovery; and its partial trajectory remains in the forensic archive. A clean resumed episode must clear the error and reevaluation flag before it becomes authoritative. Dependent-state sites retain the fresh-run default.

This resume protocol changes which run container holds an episode, not the per-task outcome being estimated. Compatibility checks reject empty, legacy-schema, or provenance-mismatched candidates. Section S3 describes the registry and the `resume_rerun_clean` evidence used to classify transient drift.

## S1.5 Verdict-day execution audit

| Audit quantity | Final report |
|---|---|
| Conditions with a valid condition-boundary reset witness | ⟨TBD: verdict-day slotsheet⟩ |
| Reddit task-boundary identity restores and failures | ⟨TBD: verdict-day slotsheet⟩ |
| Pre-flight transient retries, by cell and class | ⟨TBD: verdict-day slotsheet⟩ |
| Condition aborts, resumes, and clean breakpoint reruns | ⟨TBD: verdict-day slotsheet⟩ |
| Quarantined episodes excluded from canonical aggregates | ⟨TBD: verdict-day slotsheet⟩ |
| Order-position and reset/authentication sensitivity summary | ⟨TBD: verdict-day slotsheet⟩ |

<!-- Sources for S1 (internal; strip before submission):
git show 9acfa24^:docs/checkpoints/paper_drafts/aaai27/aaai27_main.md (§4 deleted sentence);
docs/prereg_amendments/PROTOCOL_NOTE_02_TRANSIENT_PREFLIGHT_RETRY_20260621.md;
docs/prereg_amendments/PROTOCOL_NOTE_03_RESUME_ON_ABORT_20260622.md;
docs/prereg_amendments/PROTOCOL_NOTE_04_REDDIT_IDENTITY_RESET_20260625.md;
docs/checkpoints/phase1_plan.md §B;
docs/reference/master_bug_catalog.md B-1880/B-1881/B-1882.
-->

# S2 Statistical Protocol Supplement

## S2.1 Confirmatory estimands and gates

**H1 (primary).** Within each of the six planned (site, backbone) cells, we form a task-level six-mode success matrix over DOM, SoM, Vision, P-text, P-prompt, and P-SoM. The cell effect is the six-mode oracle ceiling minus the ceiling after dropping P-SoM. We obtain a task-paired 1,000-resample bootstrap distribution and standard error per cell, then pool the six planned cells by fixed-effect inverse-variance weights. The cells are the complete registered design, not a random sample from a site/model superpopulation; no between-cell variance enters the primary estimand. H1 rejects `H0: θ_FE ≤ +1.0pp` when the one-sided pooled bootstrap-percentile probability `P(θ_FE* ≤ +1.0pp)` is below 0.05. Fixed point-estimate weights are used across pooled bootstrap iterations. Normal-approximation and random-effects results are sensitivity analyses only.

**H3 (structural).** Axis 1 measures tasks solved by P-text but not P-SoM; axis 2 analogously measures P-prompt relative to P-SoM. Each axis is pooled by the same design-based fixed-effect construction over all six planned cells and is evaluated against zero. A cell with fewer than two unique tasks fails the cell-level noise-floor label, but it remains in the pooled estimand. An H3 verdict is evaluated only when all six planned cells are present; smaller pools are explicitly interim.

**H10 (router).** H10 is an operational deployment criterion, not an across-cell significance test. Within each cell, paired task bootstrap replicates ask whether the router is non-dominated in (billed cost, success rate) by every registered fixed-mode baseline. A cell passes when the router is non-dominated in at least 95% of replicates, and deployability requires five of the six fixed cells. This engineering criterion is distinct from the H1/H3 K-of-N transparency counts below.

## S2.2 Standard-error floor and cell completeness

Flooring is fixed before the final data are read and never removes a planned cell. For H1, a per-cell bootstrap SE below the 0.68pp Agresti--Coull anchor is replaced by 1.0pp before inverse-variance weighting. For H3, the A1.21 degenerate-cell rule is narrower: only a non-positive SE is replaced by 1.0pp, while a low but positive SE remains unchanged. Outputs report both the number of non-positive SEs and the number of floor applications. Exact canonical task-set equality is required for every mode in a complete cell; a data-dependent intersection is not an eligible denominator.

## S2.3 K-of-N is transparency only

For H1 and each H3 axis, we report (i) the number of cells whose per-cell paired-bootstrap interval excludes zero and (ii) the number individually Holm-significant. Neither count is a gate and neither has a fixed K threshold. With six cells, the former 0.75 and 0.67 ratios both round to five cells, creating spurious precision; the fixed-effect pooled test carries the confirmatory decision. This retirement of K-of-N thresholds does not alter H10's separately registered five-of-six operational robustness criterion.

## S2.4 Abbreviated amendment log

| Date | Witness | Scope | Effect on estimand |
|---|---|---|---|
| 2026-05-21 | Amendment 01 | Restored upstream-core task semantics plus an execution-reliability layer; fixed action, budget, and billed-cost accounting contracts. | H1/H3/H10 structures unchanged; H10 cost axis clarified as total billed cost. |
| 2026-05-21 | Addendum 01a | Aligned the API tool schema with the runtime validator and completed accounting telemetry. | None. |
| 2026-05-23 | Amendment 02 | Corrected the strict-vs-additive H1 power label and clarified lower-claim reporting after H1 failure. | H1 gate and threshold unchanged. |
| 2026-05-24 | Amendment 03 | Aligned canonical producers, SE-floor code, and billed-cost consumers with earlier locks. | None; implementation conformance. |
| 2026-05-24 | Amendment 04 | Aligned analysis, figures, H10 entropy defer logic, and post-H1 reporting routes. | None; analysis conformance. |
| 2026-05-25 | Amendment 05 | Corrected the coordinate serialization contract and recollected affected visual conditions. | Estimand-affecting observation-contract change, witnessed before recollection. |
| 2026-05-25 | Amendment 06 | Added a non-gating run-to-run reproducibility sensitivity layer. | Primary gates unchanged. |
| 2026-05-25 | Amendment 07 | Replaced unstable internal node identifiers with deterministic sequential identifiers for SoM-family observations and recollected affected cells. | Estimand-affecting observation-contract change, witnessed before recollection. |
| 2026-05-27 | Protocol Note 01 | Made paper-grade session-loss handling non-deleting and auditable. | Recovery alignment only. |
| 2026-06-21 | Protocol Note 02 | Restricted bounded episode retry to pre-flight auth/network failures. | Recovery alignment only. |
| 2026-06-22 | Protocol Note 03 | Licensed breakpoint resume for the verified independent-task Reddit site. | Recovery alignment only. |
| 2026-06-25 | Protocol Note 04 | Restored Reddit's seeded identity before each task. | Defines the clean-per-task Reddit measurement before canonical Reddit completion. |
| 2026-07-14 | Protocol Note 05 | Restored exact task universes, all-cell H3 pooling, orthogonal status fields, and rollback-protected artifact replacement. | Locked estimands unchanged; producer conformance restored. |

## S2.5 Analysis-layer conformance

Protocol Note 05 was recorded while only an interim three-cell pool existed and before a final six-cell verdict. It corrected five implementation deviations without changing the registered analysis: all data-bearing planned cells enter H3 regardless of their cell-level noise-floor label; every mode must equal the exact canonical scored task set; single-mode SR uses that fixed denominator; completion status is separated from PASS/FAIL/NOT_EVALUATED; and unavailable values remain null rather than becoming numerical zeros. The JSON, CSV, and Markdown decision artifacts use render-before-replace with an inter-process lock, rollback backups, and parent-directory fsync. Final manuscript slots are populated only when the artifact reports complete analysis and an evaluated verdict.

<!-- Sources for S2 (internal; strip before submission):
docs/checkpoints/pre_run/preregistration.md §2 and §4;
docs/prereg_amendments/AMENDMENT_01_PROTOCOL_RESET_20260521.md;
docs/prereg_amendments/AMENDMENT_01a_SCHEMA_VALIDATOR_20260521.md;
docs/prereg_amendments/AMENDMENT_02_GATE_LADDER_20260523.md;
docs/prereg_amendments/AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524.md;
docs/prereg_amendments/AMENDMENT_04_ANALYSIS_ALIGNMENT_20260524.md;
docs/prereg_amendments/AMENDMENT_05_COORDINATE_CONTRACT_20260525.md;
docs/prereg_amendments/AMENDMENT_06_REPRODUCIBILITY_SENSITIVITY_20260525.md;
docs/prereg_amendments/AMENDMENT_07_SOM_IDENTIFIER_CONTRACT_20260525.md;
docs/prereg_amendments/PROTOCOL_NOTE_01_SESSION_LOST_PAPER_GRADE_20260527.md;
docs/prereg_amendments/PROTOCOL_NOTE_02_TRANSIENT_PREFLIGHT_RETRY_20260621.md;
docs/prereg_amendments/PROTOCOL_NOTE_03_RESUME_ON_ABORT_20260622.md;
docs/prereg_amendments/PROTOCOL_NOTE_04_REDDIT_IDENTITY_RESET_20260625.md;
docs/prereg_amendments/PROTOCOL_NOTE_05_ANALYSIS_ESTIMAND_CONFORMANCE_20260714.md.
-->

## S2.6 Per-cell drop-one estimates, five landed cells (Protocol Note 06)

Per-cell per-arm drop-one losses with 95% task-paired bootstrap CIs, regenerated from the canonical fig0c producer at the k=5 verdict (2026-07-16); the main text shows the cls-B0 panel (Table 3) and this table preserves full per-cell visibility as the protocol note requires. The B2-Reddit cell is absent (background completion track).

| panel | mode | drop-one pp | CI95 |
|---|---|---:|---:|
| B0 classifieds | DOM | +1.79 | [+0.00, +3.57] |
| B0 classifieds | SoM | +2.68 | [+0.89, +4.91] |
| B0 classifieds | Vision | +4.02 | [+1.79, +6.70] |
| B0 classifieds | P-text | +0.89 | [+0.00, +2.23] |
| B0 classifieds | P-prompt | +2.68 | [+0.89, +4.91] |
| B0 classifieds | P-SoM | +0.89 | [+0.00, +2.23] |
| B0 reddit | DOM | +1.46 | [+0.00, +2.93] |
| B0 reddit | SoM | +1.95 | [+0.49, +3.90] |
| B0 reddit | Vision | +1.95 | [+0.49, +3.90] |
| B0 reddit | P-text | +0.98 | [+0.00, +2.44] |
| B0 reddit | P-prompt | +0.98 | [+0.00, +2.44] |
| B0 reddit | P-SoM | +0.98 | [+0.00, +2.44] |
| B1 classifieds | DOM | +0.45 | [+0.00, +1.34] |
| B1 classifieds | SoM | +4.46 | [+1.79, +7.14] |
| B1 classifieds | Vision | +4.02 | [+1.77, +7.14] |
| B1 classifieds | P-text | +0.45 | [+0.00, +1.34] |
| B1 classifieds | P-prompt | +0.89 | [+0.00, +2.23] |
| B1 classifieds | P-SoM | +1.34 | [+0.00, +3.13] |
| B1 reddit | DOM | +0.49 | [+0.00, +1.46] |
| B1 reddit | SoM | +0.49 | [+0.00, +1.46] |
| B1 reddit | Vision | +0.98 | [+0.00, +2.44] |
| B1 reddit | P-text | +0.49 | [+0.00, +1.46] |
| B1 reddit | P-prompt | +0.98 | [+0.00, +2.44] |
| B1 reddit | P-SoM | +0.00 | [+0.00, +0.00] |
| B2 classifieds | DOM | +0.45 | [+0.00, +1.34] |
| B2 classifieds | SoM | +2.23 | [+0.45, +4.46] |
| B2 classifieds | Vision | +2.23 | [+0.45, +4.03] |
| B2 classifieds | P-text | +0.00 | [+0.00, +0.00] |
| B2 classifieds | P-prompt | +0.00 | [+0.00, +0.00] |
| B2 classifieds | P-SoM | +0.45 | [+0.00, +1.34] |

# S3 Quarantine and Abort Taxonomy

## S3.1 Append-only registry

The quarantine registry is an append-only event log. A `quarantine` event records the run, logical site/task key, failure class, call site, and reevaluation status. A `classification` event records one of `substrate`, `agent_induced`, `evaluator`, `transient_drift`, `unreproducible_in_isolation`, or `undecided`, together with the evidence path and rationale. Classifications are revised by appending another event; the latest timestamp defines the current label while earlier judgments remain auditable. A separate `resolution` event can close a recurrent error class only after its required evidence profile and code-provenance checks pass.

The registry is an investigation gate, not an exclusion list. Classification never automatically drops a benchmark task, changes a denominator, or imputes success. Any global exclusion would require a symmetric rule applied across all modes and a separate manuscript disclosure.

## S3.2 G8 preflight gate

G8 halts a canonical launch if any requested task has an unclassified quarantine occurrence. It also halts on a same-task recurrence across at least two fires even when classifications exist, unless every distinct recurrent error class has an evidence-gated resolution. A narrowly scoped diagnostic replay may bypass G8 only when all diagnostic guards hold; its episodes are explicitly non-canonical and success-rate excluded. Canonical launches have no such bypass.

## S3.3 `transient_drift` and `resume_rerun_clean`

`transient_drift` is reserved for failures whose evidence supports a transient execution event rather than a task, agent, evaluator, or persistent site defect. The strongest in-situ evidence path is `resume_rerun_clean`:

1. the failed partial trajectory is quarantined and forensically archived;
2. the same task is rerun after the transient service recovers, within the resumed condition context;
3. the authoritative episode completes with no infrastructure error and `needs_reevaluation=false`; and
4. the clean episode replaces only the breakpoint outcome, while prior completed tasks remain unchanged.

This path demonstrates that the task can execute cleanly on the registered substrate after recovery. It does not turn the first attempt into a success, resample completed merit outcomes, or license retries after arbitrary mid-episode failures.

## S3.4 Registry snapshot

As of 2026-07-14, the append-only file contains 47 events: 20 quarantines, 24 classification entries, and 3 resolution entries. After normalizing one legacy site alias, the quarantines affect 18 distinct (site, task) keys. Under latest-classification-wins, 16 keys are classified: 13 `transient_drift`, 2 `evaluator`, and 1 `substrate`; 2 remain unclassified and therefore remain G8 blockers. Six of the current `transient_drift` labels use the `resume_rerun_clean` evidence path. Counts describe registry state, not scientific outcomes, and will be refreshed at artifact freeze.

<!-- Sources for S3 (internal; strip before submission):
scripts/maintenance/quarantine_registry.py module docstring and G8/resolution logic;
docs/checkpoints/quarantine_registry.jsonl (47-event snapshot, aggregated 2026-07-14);
docs/prereg_amendments/PROTOCOL_NOTE_03_RESUME_ON_ABORT_20260622.md.
-->

# S4 Infrastructure and Compute

## S4.1 Execution architecture

The canonical benchmark sites run in a fresh, self-hosted Docker stack on a dedicated NVIDIA A100 environment. B0 inference is provided through a commercial large-model service behind a thin authenticated gateway; the experiment client adapts the request/response contract but does not expose the endpoint or infrastructure identity. B1 and B2 inference run locally on the A100 under the pinned reference inference path. All three backbones share the same benchmark containers, task order, evaluator, semantic action validator, and logging schema; serialization differs only where required by the backbone interface.

## S4.2 Energy, carbon, and cost accounting

GPU power is sampled with NVML over the bounded inference window. CPU package power is added from Linux RAPL when the interface is available; architecture and RAPL availability are logged so absence is not silently treated as zero. Low-density sampling windows carry an explicit partial-window flag. Per-step energy is integrated to kWh and aggregated by cell.

Carbon bounds use the registered grid intensity of 0.220 kg CO2e/kWh and PUE bounds of 1.0--1.5. Cost bases are not pooled across backbones: B0 reports billed API USD, whereas B1/B2 report electricity-derived USD. Judge calls and judge cost are counted separately. Any absolute cross-backbone USD comparison is therefore descriptive by basis, while scientific cost comparisons remain within backbone.

## S4.3 Aggregate compute table

| Cell | Inference placement | GPU hours | Energy (kWh) | CO2e lower--upper (kg) | Cost basis | Agent cost | Judge cost |
|---|---|---|---|---|---|---|---|
| cls·B0 | commercial service | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | API USD | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| cls·B1 | local A100 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | electricity-derived USD | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| cls·B2 | local A100 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | electricity-derived USD | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| red·B0 | commercial service | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | API USD | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| red·B1 | local A100 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | electricity-derived USD | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| red·B2 | local A100 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | electricity-derived USD | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| Phase 1a total | mixed | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | stratified | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |

<!-- Sources for S4 (internal; strip before submission):
docs/checkpoints/pre_run/compute_cost_carbon_table.md;
.claude/CLAUDE.md runtime-environment section;
p79/experiment/energy_tracker.py (NVML/RAPL availability and sampling-window telemetry).
-->

# S5 Additional Tables and Figures

## S5.1 Expansion of main-paper Table 2

The final table contains one row per cell and observation mode, with fixed scored denominator, success count, success rate, task-paired uncertainty, and provenance status.

| Cell | Mode | Scored n | Successful tasks | SR (%) | 95% task-bootstrap interval | Status |
|---|---|---|---|---|---|---|
| Six cells × six modes | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |

## S5.2 Expansion of main-paper Table 3

The final table expands the six-mode drop-one analysis to every complete cell and every arm. P-SoM rows additionally identify the H1 input and any SE-floor application; H3 axis rows are reported separately rather than conflated with full-portfolio drop-one.

| Cell | Arm | Six-mode oracle SR | Drop-one tasks | Drop-one loss (pp) | 95% paired-bootstrap interval | H1/H3 role |
|---|---|---|---|---|---|---|
| Six cells × six arms | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |

## S5.3 Expansion of main-paper Table 4

The final router table reports realized held-out operating points and the registered H10 checks, not training accuracy or an oracle proxy.

| Cell | Common n | Entropy status | Router SR | Router billed cost | Best feasible fixed arm | Fraction non-dominated | Strictly-better diagnostic | Cell pass |
|---|---|---|---|---|---|---|---|---|
| cls·B0 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| cls·B1 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| cls·B2 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| red·B0 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| red·B1 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |
| red·B2 | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ | ⟨TBD: verdict-day slotsheet⟩ |

## S5.4 Figure archive policy

F1 is the data-independent 2×2 construction diagram. F2 is the H1 forest and may be labeled final only when the canonical decision artifact contains the exact six planned cells, exact task universes, six unique modes per cell, and a completed H1 verdict. Any proper-subset F2 render carries an `INTERIM` watermark and is retained only as an archive snapshot. F3 is the realized per-cell H10 Pareto plot and is generated only from Pass-2 operating points; rehearsals are non-canonical. F4 is the optional unique-pass overlap visualization and is labeled archive or final according to its input manifest.

Final and archive renders use distinct filenames. The archive record retains render status, generating code revision, source-artifact digest, and creation timestamp; an archive render is never silently overwritten or cited as a final figure. Only final vector renders enter the submission package.

<!-- Sources for S5 (internal; strip before submission):
docs/checkpoints/paper_drafts/aaai27/aaai27_main.md Tables 2/3/4 and pre-submission figure checklist;
docs/checkpoints/paper_drafts/aaai27/NUMBERS_TODO.md;
scripts/analysis/figures/fig_f1_diamond_schematic.py;
scripts/analysis/figures/fig_f2_h1_forest.py.
-->

# S6 Anonymity Checklist

- [x] The current supplement contains no machine hostname, IP address, user/account name, proxy endpoint, or private service URL.
- [x] The infrastructure description uses only generic roles (self-hosted A100, commercial inference gateway, local backbones).
- [x] No institutional affiliation, funding acknowledgment, laboratory, administrator, or personal relationship is named.
- [x] No absolute internal filesystem path or private run location appears in submission-visible prose.
- [x] Benchmark seed-account values and database identifiers are omitted; only the idempotent restoration mechanism is described.
- [x] Artifact references are limited to an anonymous OSF review view in the submission version; persistent identifiers are de-anonymized only at camera-ready.
- [x] Internal source pointers occur only inside HTML comments marked for stripping.
- [ ] Repeat the scan after verdict-day numbers, figure captions, artifact links, and acknowledgments are inserted.
- [ ] Verify the compiled PDF metadata, embedded figure metadata, bibliography notes, and supplementary archive filenames.

<!-- Sources for S6 (internal; strip before submission):
docs/checkpoints/paper_drafts/aaai27/aaai27_main.md reproducibility statement and pre-submission checklist;
docs/checkpoints/pre_run/compute_cost_carbon_table.md;
.claude/CLAUDE.md runtime-environment section (used only to identify details that must be removed).
-->
