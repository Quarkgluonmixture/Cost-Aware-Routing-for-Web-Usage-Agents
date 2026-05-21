---
amendment_id: 01a
title: Addendum to Amendment 01 — B0 schema≡validator contract + accounting-completeness fixes
date: 2026-05-21
status: pre-fire protocol witness (addendum to Amendment 01)
parent_amendment: docs/prereg_amendments/AMENDMENT_01_PROTOCOL_RESET_20260521.md
parent_amendment_tag: prereg-amendment-01-protocol-reset-20260521  # @ e1f86f4
parent_prereg: docs/checkpoints/pre_run/preregistration.md (status: locked)
parent_doi: 10.17605/OSF.IO/9QCWU  # DOI 1, pre-canonical-outcome-creation witness, 2026-05-18
witness_tag: prereg-amendment-01a-schema-validator-20260521  # to be created at the commit adding this file
relation: |
  Amendment 01 (tag e1f86f4) was committed + tagged + OSF-uploaded as the Fire-6
  pre-fire protocol witness. Between that tag and Fire-6, a pre-fire /stress (3-AI:
  Claude + codex + gemini, 2026-05-21) surfaced — and this session fixed — a set of
  backend-serialization and accounting-completeness defects that are part of the
  protocol Fire-6 actually runs. Per the project's pre-fire-witness doctrine (a
  canonical-protocol change AFTER the witness tag but BEFORE the canonical fire
  needs its own content-addressed witness), this addendum closes that gap: it is the
  immutable pre-fire witness for the B0 schema≡validator contract (B-1794 / 681b9cf)
  and the accounting-completeness fixes (B-1796 … B-1802). It ADDS to, does not
  retract, Amendment 01.
---

# Addendum 01a — B0 schema≡validator contract + accounting-completeness fixes

> **One-line**: The B-1794 schema≡validator fix (`681b9cf`) and the 2026-05-21
> pre-fire `/stress` accounting-completeness fixes (B-1796 … B-1802) landed AFTER
> the Amendment 01 witness tag (`e1f86f4`) but BEFORE Fire-6. They are within the
> GRL serialization-adapter + execution-reliability boundary that Amendment 01 §3
> items 1–2 already witnessed *in principle*; this addendum witnesses them
> *explicitly* because they materially shape what B0 emits and what the canonical
> accounting artifact contains.

## §1 — Why this addendum (the witness-timeline gap)

Amendment 01 §3 item 2 ("Backend-specific serialization, shared semantic schema")
cited only the `tool_choice` fix (`dbb1bda`) and a then-current "B0 ~5 % parse_error"
figure. It did **not** mention B-1794 / the schema≡validator contract — because the
B-1794 root cause (the `tool_choice="required"` *forcing* a minimal tool call that
dropped the OPTIONAL `element_id`) was only diagnosed AFTER the Amendment 01 tag.

`git merge-base --is-ancestor e1f86f4 681b9cf` is **true**: the witness tag is an
ancestor of the B-1794 fix. The pre-fire `/stress` (2026-05-21) flagged this as a
witness-completeness gap (finding P1-9). User decision (2026-05-21, Q2=B): record
the schema≡validator contract + the accounting fixes as a witnessed addendum, re-tag,
and re-upload to OSF.

The "~5 % parse_error" figure in Amendment 01 §3 item 2 is **superseded**: it was the
fix-A-era (`dbb1bda`, tool_choice only) measurement, pre-B-1794. Post-B-1794 two
30-step DOM smokes returned **0 invalid steps**.

## §2 — What this addendum witnesses (the deltas since e1f86f4)

All within the Amendment 01 §3 item 1 GRL boundary ("backend serialization adapters"
+ execution reliability + auditability), and the §3 item 2 "shared semantic schema +
validator" principle. None changes task policy, prompt, action-set, termination, or
the SR / cost estimand definitions.

| ID | Fix | Commit(s) | Why it is a Fire-6 protocol fact |
|---|---|---|---|
| **B-1794** | B0 tool schema per-action `required` ≡ `validate_action_detailed` (conditional `allOf` if/then; root cause = `tool_choice="required"` forcing) | `681b9cf` | Determines which action objects B0 can emit under forcing; pre-fix B0 dropped `element_id` on type/search → many invalid steps. Fire-6 B0 runs under this contract. |
| **B-1796** | `select_option` schema `anyOf(element_id, coordinate)` (was `required: element_id` only) + bidirectional invariant test | this round | VISION-mode B0 `select_option` was schema-impossible (no AXTree → no element_id) while B1/B2 free-gen could emit it — a cross-baseline asymmetry. Fixed so B0 is neither stricter nor looser than the validator. |
| **B-1797** | B1/B2 stamp `action_source="text_json"` + `text_parse_path`; `repaired_fenced` no longer written to `parse_failure_reason` | this round | Makes the cross-baseline "same `validate_action_detailed` gate" claim provable from the JSONL; keeps the parse-failure taxonomy clean (B2 Gemma's 30/31 fenced-repair rate was previously mislabelled as failures). Collection-time provenance (not backfillable). |
| **B-1798** | Episode-summary cost-basis accounting boundary: roll up `cost_unit_basis` (modal) + `cost_total_mixed_unit_warn` (any) step→episode→condition | this round | Without it, `cost_unit_basis` was `None` at episode/condition and `"unknown"` in the cross-site CSV → the §1 cross-baseline cost stratification (B0 api_usd vs B1/B2 electricity-derived) was unverifiable from the artifact. |
| **B-1799** | Exception-path `_aggregate_partial_steps` returns `total_obs_prepare_cost_usd` (component-breakdown closure on crashed episodes) | this round | Failed/partial episode cost breakdown no longer under-counts obs_prepare (component parts now sum to total). |
| **B-1800** | Crash-atomic billed-cost ledger: an exception between the billed model call and the step JSONL write no longer loses the in-flight step's billed cost + model-call attempt | this round | Protects the §1 PRIMARY cost (`total_billed_cost`) + `model_call_attempt_count` against mid-step crashes on the canonical run. |
| **B-1801** | `aggregate_cross_site.py` published-artifact fixes: emit `avg_total_latency_canonical_ms` + operands to CSV (P1-2); SR keyed per-(baseline, mode) not pooled (P1-3); suppress absolute-cost plot on mixed/unknown basis (P1-4); **fail-loud** when a paper-grade row has billed cost but unknown basis | this round | Makes the canonical-latency estimand reproducible from the CSV, prevents silent cross-baseline SR pooling, and refuses to silently emit a basis-less cost figure. |
| **B-1802** | Disclosure (P1-10): the B0 tool schema is a **soft constraint** (the proxy does not hard-enforce it); `validate_action_detailed` is the **hard runtime gate**. Code comment + paper §3.5.1. | this round | Corrects an over-claim (the proxy does NOT enforce the schema — B-1101); the runtime authority is the validator, not proxy-side enforcement. |

Full per-finding detail: `docs/reference/master_bug_catalog.md` (B-1794 + B-1796…B-1802)
and `docs/checkpoints/实验笔记.md` §252 (the 2026-05-21 pre-fire /stress chronicle).

## §3 — What did NOT change

- **No estimand change.** SR definition, the §1 cost = `total_billed_cost` primary
  (Amendment 01 §3 item 5 + §5), the H1/H10 gating structures — all unchanged.
- **No task / prompt / action-set / termination change.** These are serialization +
  accounting + disclosure fixes inside the GRL boundary.
- **No new hypothesis** enters the gating family.

## §4 — Data status (unchanged from Amendment 01 §4, reaffirmed)

- All pre-Amendment-01 data is non-canonical (pilot / RCA / archive).
- All pre-B-1794 B0 data (incl. the Fire-3/4/5 candidates) is non-canonical — it
  carried the `element_id`-omission artifact.
- **Fire-6 is the first canonical Phase 1a run under Amendment 01 + this addendum.**

## §5 — Witness mechanics

1. Commit this file (the addendum) together with the B-1796…B-1802 code/test/doc fixes.
2. Tag the commit `prereg-amendment-01a-schema-validator-20260521`.
3. `git push origin master --tags` — the pushed commit + tag is the content-addressed,
   tamper-evident pre-fire Git witness for the schema≡validator contract + accounting
   fixes (the SHA hashes the exact protocol state; the tag is the immovable pointer).
4. (Manual, user) Upload this addendum to the OSF project (`kv9sf` parent) alongside
   Amendment 01 as the external visibility layer; reference it in the DOI-2 bundle.
5. Add a one-line pointer in `preregistration.md`'s amendment log → this file.

Fire-6 may proceed once this addendum is committed + tagged + pushed (OSF upload is the
visibility layer on top of the Git witness primitive, recommended before Fire-6).
