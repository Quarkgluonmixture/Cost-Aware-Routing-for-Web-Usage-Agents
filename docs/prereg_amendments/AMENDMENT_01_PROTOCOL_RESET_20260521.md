---
amendment_id: 01
title: Protocol Reset — Upstream-Core VWA semantics + P79-GRL reliability layer
date: 2026-05-21
status: pre-fire protocol witness (pending git tag + push)
parent_prereg: docs/checkpoints/pre_run/preregistration.md (status: locked)
parent_doi: 10.17605/OSF.IO/9QCWU  # DOI 1, pre-canonical-outcome-creation witness, 2026-05-18
parent_lock_tag: preregistration-locked @ ef609a3
witness_tag: prereg-amendment-01-protocol-reset-20260521  # to be created at the commit adding this file
relation: This amendment is the PRE-FIRE PROTOCOL WITNESS for Fire-6. DOI 1 (2026-05-18)
          witnessed the substance-locked prereg BEFORE any canonical outcome existed; the
          Protocol Reset (2026-05-20/21) changed the canonical protocol AFTER that lock but
          BEFORE Fire-6 creates canonical data. DOI 2 (post-data reproducibility bundle) is
          NOT a pre-fire protocol witness. This file + its git tag close that gap as a
          content-addressed, tamper-evident pre-fire Git witness: the protocol that Fire-6
          actually runs is anchored to a commit SHA + tag BEFORE the fire. OSF upload is an
          external visibility layer on top (do before Fire-6), not the witness primitive.
---

# Preregistration Amendment 01 — Protocol Reset

> **One-line**: Between the DOI-1 substance lock (2026-05-18) and Fire-6, the canonical
> Phase 1a protocol was redefined as **"Upstream-Core VWA semantics + P79-GRL reliability
> layer."** This amendment is the immutable pre-fire witness of that redefinition. It
> doubles as the **Protocol Reset Memo** (old protocol → problem → new protocol → policy).

## §1 — Why this amendment

**Trigger chain** (实验笔记 §242 → §249):
1. **§242** — A run-to-run SR variance investigation surfaced two confounds: B0 (Qwen3-VL-235B
   MoE) decode stochasticity, and **partial-run SR being doubly incomparable** across attempts.
2. **§243** — Root-cause deepened into a **P79-vs-upstream-VWA divergence audit**: P79 had
   silently diverged from upstream VisualWebArena in 12+ places (zero-shot custom prompt
   replacing upstream 5-shot CoT; an invented `WAIT` action used as a parse-failure sink;
   dropped `hover/press/new_tab/goto` from the action space; `max_steps` counting injected
   waits rather than agent decisions; B0 serialization bug B-991). These are protocol-level
   deviations that make absolute numbers non-comparable to the literature and create
   cross-baseline asymmetries.
3. **§244** — User locked a **12-point canonical decision**: redefine canonical Phase 1a as
   **Upstream-Core VWA semantics + a P79-GRL (execution-reliability) layer**, with the GRL
   boundary = *"make execution reliable, NOT change task policy."*
4. **§245–§249** — Execution: action-set restore, accounting reset (two-budget + three-column
   cost), B2 parse-health, vLLM/eager conclusion, GRL boundary audit, and a 3-AI pre-fire
   `/stress` on the accounting (B-1786).

The Protocol Reset is therefore a **deliberate restoration toward upstream**, not a bug-fix
patch. Because it post-dates the locked prereg, it requires its own pre-fire witness.

## §2 — What did NOT change (carried forward from the locked prereg)

The following are **unchanged** from `preregistration.md` (DOI 1) and are NOT amended:

- **Core research question** — does cost-aware routing improve the success-rate × efficiency
  trade-off for VWA web agents; the **phantom routing space** phenomenon (3 sibling arms:
  P-text / P-prompt / P-SoM) + 4-fold drop-in property.
- **Scope**: Phase 1a = **42 operational conditions** (Pass-1 baseline 36 + Pass-2 learned
  router 6) across **6 statistical cells** = {classifieds, reddit} × {B0, B1, B2}.
- **Baseline identities**: B0 = Qwen3-VL-235B-A22B (proxy API), B1 = Qwen3-VL-4B (local),
  B2 = Gemma-3-4B-it (local, cross-family 4B-parity robustness check). HF SHAs / proxy
  contract unchanged.
- **Observation modes**: the 6 modes (DOM / SoM / Vision / P-text / P-prompt / P-SoM).
- **Primary outcome (SR)**: `success = (VWA evaluator score ≥ 1.0)`, pure-evaluator authority
  (post-B-545, no reward override). **Unchanged** — the Protocol Reset does not touch the SR
  definition.
- **Gating hypotheses**: H1 (hero, phantom-SoM drop-in lift) + H2 + H3 gate the §1 R1-R5
  framing; H10 (learned router, (Cost, SR) Pareto non-dominance) gates §6. Estimand
  *structures* (bootstrap percentile p for H1; per-cell paired-bootstrap Pareto + 5/6 grid
  for H10) are **unchanged**. See §5 for the one operationalization clarification (which USD
  column is the Cost axis).
- **N/A task exclusion** at load (`exclude_na_tasks`), `scored_task_count`, K-of-N as
  transparency-only — unchanged.
- **DOI 1 substance** — this amendment ADDS to, does not retract, the DOI-1 witness.

## §3 — What changed (the Protocol Reset deltas)

Each delta carries its §244 canonical-point number, the implementing commit, and the chronicle §.

1. **Canonical protocol definition** (§244 #3) — canonical Phase 1a = **Upstream-Core VWA
   semantics + P79-GRL reliability layer**. GRL = evaluator isolation (C1/C1b), screenshot-
   timeout recovery, Gate-8 recurrent-failure registry, fail-closed quarantine,
   telemetry/provenance/SBOM, VWA evaluator bug fixes, wrapper reliability primitives, backend
   serialization adapters. GRL is **NOT** a prompt/action/task-policy layer. New framing prose:
   *"We evaluate VWA tasks under upstream-aligned prompt/action/termination semantics, with a
   P79-GRL runtime layer for execution reliability and auditability."* (§247 GRL audit
   B-1776~B-1783 verified the boundary held.)

2. **Backend-specific serialization, shared semantic schema** (§244 #1) — B0 = native
   tool-call (`tool_choice="required"`), B1/B2 = text JSON. Capability-forced split (B2 has no
   native tool-call ABI). The three **share**: semantic action schema + validator + max_steps
   semantics + failure accounting + cost accounting. S0–S5 serialization-health audit passed
   (B0 required ~5% / B1 0% / B2 0.7% parse_error). Commit: fix A `dbb1bda` (tool_choice
   auto→required, B0 emit 22.5%→~100%).

3. **WAIT is not a canonical valid agent action** (§244 #6) — `action_utils` rescues every
   parse/structural failure to a `{"action_type":"wait"}` sink; this is an internal recovery
   event (`valid_agent_action=False`, does not consume the agent-action budget). Commit
   `66860e5`.

4. **`max_steps` restored to upstream agent-action-budget semantics** (§244 #7) — two-budget:
   PRIMARY `max_agent_actions=30` (only parse-valid, budget-consuming steps); SAFETY budget
   `max_model_attempts` (LLM-call ceiling) + `max_consecutive_parse_errors=3` +
   `max_total_parse_errors=5`. Pre-reset `step_idx += 1` counted injected waits → a task could
   be 1 real decision + 29 waits. Commit `66860e5`.

5. **Three-column cost; §1 PRIMARY cost = `total_billed_cost`** (§244 #8 + /stress audit
   2026-05-21 Q1=A) — `total_billed_cost_usd` (every billed LLM call, incl. parse-error +
   policy-blocked + recovery) / `canonical_action_cost_usd` (valid-action steps only) /
   `protocol_wasted_cost_usd` (residual); `canonical + wasted ≡ billed`. **Paper §1 hero cost
   = `total_billed_cost` (honest "what you pay")**; `canonical` + `protocol_wasted` +
   `valid_action_step_count` + `parse_error_rate` + `model_call_attempt_count` are **§4
   efficiency decomposition / diagnostics**. This framing (Q1=A, /stress 2026-05-21) defuses
   the "canonical-only flatters the buggy-but-capable API model (B0)" reviewer attack:
   nothing is hidden from the headline cost; the decomposition is disclosed in §4. Commits
   `66860e5` + `38df50d`.

6. **Off-site `goto` is a policy-blocked action** (§244 #5 follow-up, B-1782) — restored to
   the action space but constrained to the VWA origin; an off-site goto is `valid_agent_action
   =False` + `consumes_agent_action_budget=True` + `error_category="policy_blocked_offsite"` +
   `action_success=False` (it spends the agent's turn but is not a permitted action — not a
   silent no-op). Commit `4141e0b`.

7. **Action set restored toward upstream** (§244 #5) — `hover / press / new_tab / close_tab /
   goto` restored across the full action stack: **prompt + shared semantic action schema +
   validator + wrapper/executor (dispatch + escape-hatch) + B0 tool-call schema + B1/B2 JSON
   serialization schema** — all layers kept consistent so an action is permitted only when
   every layer supports it. A targeted `/stress` first confirmed the cls+reddit task set is
   functionally served (cross-site reddit uses pre-opened tabs + `tab_focus`, not goto), then
   the restore landed across the stack. Commit `4141e0b` (+ cross-AI executable-contract fixes).

8. **Zero-shot controlled protocol disclosed** (§244 #4) — upstream 5-shot CoT is **NOT**
   restored; P79 runs a zero-shot controlled protocol. Consequence: **we do NOT claim absolute
   SR comparability to VWA literature.** Primary estimand = **within-protocol paired
   comparison across observation modes / routing** (the phantom routing space), not SOTA SR.

9. **B-991-window B0 data is non-canonical** (§244 #2) — all B0 data produced after the B-991
   serialization bug (2026-05-18) and before the tool_choice fix is non-canonical (Fire-3/4/5/6
   B0 candidates + R2987 = RCA/archive only). Marked in `results/phantom_paper/run_manifest.yaml`.
   Commit `1a829f5`. More broadly, **all pre-reset (pre-Amendment-01) data (B0/B1/B2) produced
   under the P79-custom-prompt / pre-two-budget / pre-action-set-restore protocol is
   pilot/archive only** and does not enter the canonical Phase 1a protocol. (The invalidation
   is protocol-wide — prompt framing, accounting semantics, AND action space all differ — not
   merely a prompt-text change.)

10. **C1/C1b/Gate-8 per-error-class resolution** (§244 #9) — eval-isolation + screenshot-
    recovery + Gate-8 recurrent-failure registry resolve per error-class, never task-level
    `resolved=true`. Commit `51dab0f` (cls-75 honest re-resolve, B-1779).

11. **B-651 Holm family reverted to `(test, metric)`** (§244 #10) — matches the locked prereg
    "m = N_cells across 6 cells" + paper §3. Commit `fd723b9`.

12. **B1/B2 inference engine = HF eager / reference path** (§244 #11) — canonical run forbids
    vLLM and behavior-changing `torch.compile` (both empirically change ~10–28% of step
    actions on 4B near-tie decisions, §246). Speedup may come only from hardware / scheduling /
    output-preserving I/O — never from a different numerical inference path. Commit `b4f93a3`
    (§246 conclusion).

## §4 — Data status

- **All pre-Amendment-01 data is non-canonical** (pilot / RCA / archive). This includes every
  B0/B1/B2 run produced under the P79-custom-prompt + pre-two-budget + pre-action-set-restore
  protocol.
- **Fire-6 is the FIRST canonical Phase 1a run under Amendment 01.** No prior run satisfies the
  canonical protocol.
- Diagnostic-replay episodes (`diagnostic_replay=True` / `sr_excluded=True`) are excluded from
  SR aggregation by construction.
- DOI 2 (post-data reproducibility bundle) will anchor the Fire-6 canonical data + the frozen
  analysis, and will cite this Amendment 01 as the pre-fire protocol witness.

## §5 — Estimand impact (the one operationalization clarification)

The Protocol Reset does **not** redefine any locked gating hypothesis. One **operationalization**
is clarified (it was under-specified in the locked prereg, which predates the three-column cost):

- **H10 (Cost, SR) Pareto** — the "Cost" axis is operationalized as **`total_billed_cost_usd`**
  (consistent with the §1 primary cost above). `total_billed_cost_usd` includes all billed
  model calls **and router/controller overhead where that overhead is itself billed** (e.g.,
  an extra routed model call). Where the routing/controller overhead is local / non-billed
  (CPU-side decision cost, no API charge), it is **not** folded into the billed cost — it is
  reported separately in §6 diagnostics / Net-Saving accounting so the routed-vs-fixed
  comparison stays apples-to-apples. H10 Pareto is computed **within each cell** (one model =
  one `cost_unit_basis`), so cross-baseline cost-unit mixing does not arise. The cost-axis
  sensitivity using `canonical_action_cost` is an Appendix decomposition, not the gate.
- **H1 (hero, SR)** — unchanged; SR definition untouched. H1 runs on Fire-6 canonical data.

No hypothesis is added, removed, or re-thresholded. No new analysis enters the gating family.

## §6 — Deferred / open (do NOT block Fire-6)

- **single-budget vs two-budget** — gemini Mode C (/stress 2026-05-21) argued that not charging
  the agent-action budget for parse-errors gives a buggy model "free looks." Decision: **keep
  two-budget** (it restores the upstream "30 valid agent actions" semantics) and report
  `total_billed_cost` + `protocol_wasted_cost` + `model_call_attempt_count` + `parse_error_rate`
  to expose, not hide, the surface. The single-budget question is referred to the advisor as a
  short note; it is **NOT a Fire-6 blocker** unless the advisor explicitly requires the change
  before data collection.
- **parse-cap sensitivity (cap 5 → 10)** — NOT a Fire-6 prerequisite. Fire-6 runs at the locked
  `max_total_parse_errors=5`; `parse_error_rate` / `model_call_attempt_count` /
  `protocol_wasted_cost` are recorded. A targeted cap=10 sensitivity is a **post-data**
  follow-up, run only if the observed parse-error rate (esp. B0) is non-trivial.

## §7 — Evidence (commits + audits + chronicle)

| Protocol Reset component | Commit(s) | Chronicle |
|---|---|---|
| 12-point canonical decision | (decision) | §243, §244 |
| Action-set restore (#5) + cross-AI fixes | `4141e0b` | §245 |
| B-991 data non-canonical (#2) | `1a829f5` | §244 |
| Accounting reset two-budget + three-column cost (#6/#7/#8) | `66860e5` | §248 |
| Accounting /stress 3-AI audit fixes (B-1786) | `38df50d` | §249 |
| fix A tool_choice required (#1 serialization) | `dbb1bda` | §247 |
| Holm family revert (#10) | `fd723b9` | (parallel) |
| GRL boundary audit (#9, B-1776~B-1783) | `f7bc44f` `dbb1bda` `526db4b` `54097d8` `51dab0f` | §247 |
| vLLM/eager canonical conclusion (#11) | `b4f93a3` | §246 |

Audit trail: `master_bug_catalog.md` (B-1776~B-1786 + Protocol Reset accounting section);
S0–S5 serialization-health table (§243); 3-AI /stress outputs (codex/gemini, gitignored —
condensed in catalog + chronicle).

## §8 — Witness mechanics

1. Commit this file (the amendment).
2. Tag the commit `prereg-amendment-01-protocol-reset-20260521`.
3. `git push origin master --tags` — the pushed commit + tag is the **content-addressed,
   tamper-evident pre-fire Git witness** (the SHA hashes the exact protocol state; the tag is
   an immovable named pointer to it). This Git witness is the primitive.
4. (Manual, user) Upload this amendment to the OSF project/component (`kv9sf` parent) as the
   **external visibility layer** on top of the Git witness — recommended before Fire-6 so the
   protocol change is OSF-visible; reference it in the DOI-2 reproducibility bundle.
5. Add a one-line pointer in `preregistration.md`'s amendment log → this file.

Fire-6 may proceed only after this witness is committed + tagged + pushed (and, if the advisor
sync requires it, the single-budget question resolved).
