---
amendment_id: 04
title: Implementation-alignment of analysis-layer producers + figures to locked estimands (ADD-label demotion / H10 entropy DEFER / cost total_billed / latency scaffold-adjusted / post-R5 route / B2 cross-family downgrade) — NO new estimand / gate / δ / R-ladder change
date: 2026-05-24
status: pre-fire witness (DRAFT — pending git tag + push + OSF upload); Phase 1a paper-grade outcome statistics NOT yet computed
parent_prereg: docs/checkpoints/pre_run/preregistration.md (status: locked)
parent_doi: 10.17605/OSF.IO/9QCWU   # DOI 1, pre-canonical-outcome-creation witness, 2026-05-18
parent_lock_tag: preregistration-locked @ ef609a3
prior_amendments:
  - AMENDMENT_01_PROTOCOL_RESET_20260521
  - AMENDMENT_01a_SCHEMA_VALIDATOR_20260521
  - AMENDMENT_02_GATE_LADDER_20260523
  - AMENDMENT_03_IMPLEMENTATION_ALIGNMENT_20260524
witness_tag: prereg-amendment-04-analysis-alignment-20260524   # to be created at the commit adding this file
provenance: 3-AI /stress audit 2026-05-24 (Mode A Claude stats-methodologist + Mode B codex reproducibility + Mode C gemini figure/prose); unified bug list user-confirmed fix scope 1A/2A/3A/4A.
relation: >
  Pure implementation-alignment of analysis-layer producers (aggregate_*, train_l1_router,
  power_analysis), figure scripts, and the analysis README to estimands ALREADY LOCKED by
  DOI-1 + AMENDMENT_01/02/03 + the preregistration §2/§4/§H10 prose. Recorded BEFORE any
  Phase 1a paper-grade outcome statistic exists (no per-cell drop-one θ_i, no pooled θ_FE,
  no H10 verdict computed on paper-grade data) and externally witnessed before any eligible
  H1/H3/H10 result is available. This amendment changes NO estimand, NO gate test, NO δ
  threshold, and NO R1-R5 framing ladder. Each item is "code/prose caught up to an
  already-witnessed estimand or a previously user-approved decision", not a new analytical
  choice. The analysis layer is NOT in the Gate-3 fire import path, so these edits are
  fire-safe; they are nonetheless witnessed because several touch estimand-adjacent
  surfaces (§1-hero label, H10 DEFER gate, cost/latency canonical column).
---

# Preregistration Amendment 04 — Analysis-layer implementation alignment (NO estimand change)

> **One-line**: The canonical statistical *gate* code (`aggregate_phase1_full_prereg_decision`
> H1 bootstrap-percentile + H2(a) ratio + H3 axes) was found faithful to the prereg, but a
> set of **producer / figure / README / prose** surfaces had drifted from already-locked
> estimands. This amendment lands those alignments in code + tests where stale
> implementations diverged. It does **not** introduce, weaken, or re-choose any estimand,
> gate, threshold, or framing tier.

## §0 — Pre-data status (the legitimacy anchor)

Witnessed **before any Phase 1a paper-grade outcome statistic exists**. At witness time:
- No per-cell drop-one θ_i and no pooled θ_FE computed on paper-grade data
  (`results/phantom_paper/phase1_full_prereg_decision.*` not yet produced from a complete
  6-mode cell; `results/phantom_paper/h10_pareto_verdict.*` = no Pass-2 router data).
- Pass-1 baseline (Gate-3 cls 18-condition chain) is in progress; Pass-2 router has not fired.
- The R5 post-pivot route, the B2 cross-family downgrade, and the H10 entropy DEFER gate
  cannot fire until 6-mode / Pass-2 data lands, so no gate statistic can be computed from
  the changed paths.

**Honest exposure note**: pre-fix *archive* outcomes are a correlated-population sanity
check, explicitly NON-substrate in DOI-1, known buggy, and were NOT used to motivate any
choice here. The motivation is purely consistency between code/prose and the already-locked
estimands + previously user-approved decisions (the drift was surfaced by a 3-AI code audit,
not by inspecting any outcome).

## §1 — Why this amendment

A 3-AI hostile `/stress` audit of the analysis + visualization layer (2026-05-24) found the
gate arithmetic sound but surfaced a vein of **estimand-label drift** + **prereg/amendment
logic that was locked in prose but never landed in code**. Left unaligned, a reviewer
comparing the paper §1/§2.4/§6 figures + tables to the preregistration would file a
bait-and-switch / unimplemented-gate objection. None of the items changes a locked estimand;
they make the producers/figures/prose say what the prereg already says.

## §2 — Items (each = alignment to an already-locked decision)

| # | Item | Already-locked by | Code/prose change |
|---|---|---|---|
| **B-1849** (P0-1, 3-AI) | ADD estimand (`4psom_vs_3`, +2.336pp, 3→{4,5}-mode incremental) was labeled "PRIMARY H1 gate / deployment hero" in `power_analysis.py`, `fig0c_phantom_lift_bars.py`, `README.md`. | AMENDMENT_02 §3 (+2.336pp = 4-mode ADD ≠ 6-mode-strict gate; strict ≤ ADD by construction). | Relabeled all three to "H1-deploy / Appendix-D sensitivity (4-mode ADD)"; 6-mode-strict H1 power/effect marked TBD post-Phase-1a. **Numbers unchanged**, labels only. Mirrors the ⚠️ disclaimer prereg §2.4 already carries + `fig_meta_forest.py:50-56` `APPENDIX_EXPLORATORY` precedent. |
| **B-1850** (P0-2, codex OOB) | H10 DEFER entropy gate (per-cell train-fold label entropy `H=−Σ p·log_2(p) < 1.0 bit` → §5 descriptive) was prose-only; `aggregate_h10_pareto` had no entropy branch. | prereg §H10 L238-240 (DEFER condition). | `train_l1_router.py` emits per-cell per-fold train-label entropy (RAW `y_train`, pre B-995 filter) → `h10_entropy_gate.json`; `aggregate_h10_pareto.run_h10_verdict` reads it + fail-closed DEFER (any cell min-over-folds < 1.0 bit OR artifact missing → `operational_gate_passed=False`, `h10_status=deferred_entropy/entropy_unavailable`, pre-entropy verdict retained for transparency). |
| **B-1851** (P1-1, Claude) | `_apply_framing` R5 branch hardcoded "advisor sync"; no `post_r5_pivot` field. | AMENDMENT_02 §4 L195-210 (post-R5 reporting route C'-S / C'-R / F, mechanically determined by H3/H10; regression test required). | `_apply_framing` accepts `h10_pass`; R5 emits `post_r5_pivot` ∈ {C_prime_structure (H3 pass), C_prime_router_only (H3 fail+H10 pass), F_failure (both fail), pending (H10 not yet computed)}. NO framing-tier rescue (R5 stays R5). Regression test asserts AMENDMENT_02 L209-210. |
| **B-1852** (P1-2, Claude) | B2 (Gemma) cross-family claim-tier downgrade unimplemented; `_apply_framing` did not receive per-cell B2 outcomes. | prereg §2.5 step-8 L408-413 (B-1284). | `_apply_b2_cross_family_downgrade`: Qwen-lineage {B0,B1}×{cls,red} any per-cell H1 fail → R5 (L412 load-bearing anchor); B2 {cls,red} any fail while Qwen all pass → R-tier downgrade one step R1→R2→R3 (L411); incomplete data → conservative no-change. |
| **B-1853** (P1-3, Claude+codex) | Canonical gate `_load_cell_per_task:191` (H2(a) cost-ratio source) read `total_cost_usd`, not `total_billed_cost_usd`. AMENDMENT_03 §3 migrated `aggregate_cost_electricity` + `aggregate_h10_pareto` but **missed this third producer** (sibling-propagation gap). | AMENDMENT_01 §1 + AMENDMENT_03 §3 (§1/H2a/H10 cost = total_billed). | `_load_cell_per_task` reads `total_billed_cost_usd` + fail-closed (legacy `total_cost_usd`/`total_model_cost_usd` only under `P79_ALLOW_LEGACY_COST=1`); missing-billed cell → H2(a) `cannot_evaluate`, not a wrong-basis ratio. |
| **B-1854** (P1-4, codex) **— A′ classification** | `aggregate_cross_site.py:309-316` canonical latency = `minus_retry − busy_wait − screenshot_recovered` (triple-subtract); prereg §4 + amendment-01 prose described retry-adjusted (`minus_retry`) only. | **User-approved B-1669 (Q6=A 2026-05-18, busy_wait subtraction治 red 99s污染) + B-1780 (Q3=A 2026-05-20, commit 526db4b, screenshot-recovery subtraction; 1162 tests passed)**. | **Code unchanged** (triple-subtract retained — it is the witnessed decision). Comment strengthened to name canonical = "scaffold-adjusted latency" + corrected one stale 2-subtract formula note. **The stale piece is the prereg §4 / amendment-01 prose, NOT the code** — see §3 supersession. raw / retry-adjusted (`minus_retry`) / scaffold-adjusted (`canonical`) all kept as separate columns; §1 hero cites scaffold-adjusted. |
| **B-1855** (P1-6, codex) | `train_l1_router.py` called `holdout_sr` "the H10 evaluation point"; it is CV mode-match accuracy (predicted == oracle-best label), not realized SR/cost. | True H10 = Pass-2 realized (Cost, SR) per prereg §H10 + `aggregate_h10_pareto.analyze_cell`. | Comment relabeled DIAGNOSTIC ONLY; added `cv_mode_match_acc` alias; "do NOT cite as H10 evidence". |
| **B-1856** (P2-3, codex) | `aggregate_h10_pareto --require-full-coverage` was opt-in; default run emitted a biased intersection-subset verdict. | prereg §4 router train/test + C8 (B-1811) paper-grade coverage. | Default now fail-closed full coverage; `--allow-partial-dev` opt-out for dev; Makefile dev target passes `--allow-partial-dev`. |
| **B-1857** (P2-4, Claude) | Stale "GLM fallback" comments in `_h2a_per_task_ratio` (GLM-rescue retired B-991 2026-05-17). | B-991 (GLM fallback physical removal). | Comments updated to "proxy edge / early-exit; GLM retired B-991". |

**Already resolved upstream (no action this amendment, recorded for completeness)**:
- **P2-2** (normal-Z transparency floor): AMENDMENT_03 already aligned `aggregate_phase1_prereg_gate._fe_pool` SE floor to 0.68/1.0 (lines 89-94), so the normal-Z transparency floor already matches the bootstrap-percentile primary floor. No drift remains.
- **P1-7** (LR `class_weight=None` vs prereg L224 `balanced`): code already carries the B-995 rationale (`train_l1_router.py` docstring L11 + cell_meta note L549). The prereg↔code tie is recorded here: **B-995 supersedes prereg L224 `balanced` — `class_weight=None` is canonical** (balanced reweighting produced ~15× minority-class hallucination). No code change.
- **P1-5** (stale on-disk artifacts `h10_pareto_verdict.json` / `cross_site_*`): these are gitignored, regenerated by `make analysis` post-fire. Tracked as a post-fire regeneration item (next_steps); no source change.

## §3 — Supersession table (prereg/amendment prose ← canonical)

Per the AMENDMENT_03 precedent, `preregistration.md` is the DOI-1 anchor and is **not edited
in place**; this table records where its prose lags the canonical (witnessed) estimand. These
are honesty-surface sync items (companion to A03-fu / R2-P2-10-C), safe to fold into the
prereg prose at paper-finalize without re-witnessing (the estimand values are unchanged).

| prereg / amendment locus | stale prose | canonical (this amendment) |
|---|---|---|
| `preregistration.md` §4 row "Canonical cost-and-latency estimand framework" (L511) | "(b) Latency canonical = retry-adjusted (`total_minus_retry_ms`)" | Latency canonical = **scaffold-adjusted** = `minus_retry − busy_wait − screenshot_recovered` per user-approved B-1669 (Q6=A) + B-1780 (Q3=A). `minus_retry` retained as a sensitivity column. |
| `AMENDMENT_01_PROTOCOL_RESET` latency disclosure | retry-adjusted-only framing | superseded by B-1669/B-1780 scaffold-adjusted canonical (busy_wait + screenshot-recovery are scaffold/instrumentation leakage, not mode-intrinsic latency). |
| `preregistration.md` §2 H1 L104-109 SE-floor protocol (= A03-fu) | "SE_i = 0 exactly" trigger + "SE_floor = 1.0pp" + impl pointer `aggregate_phase1_prereg_gate.py:185-187` | SE floor trigger = `SE < 0.68pp` (Agresti-Coull anchor), replace with 1.0pp; canonical producer = `aggregate_phase1_full_prereg_decision._pool_bootstrap_percentile_p` + `aggregate_phase1_prereg_gate._fe_pool` (both 0.68/1.0 per AMENDMENT_03). |
| `preregistration.md` §H10 router architecture L224 | `LogisticRegression(... balanced class_weight)` | `class_weight=None` per B-995 (balanced → ~15× minority hallucination). |

## §4 — Code-change manifest

Files (all `scripts/analysis/` — NOT in fire import path):
- `power_analysis.py` — P0-1 ADD-label disclaimer (docstring/header/table/footnote/interp/reviewer-claim).
- `figures/fig0c_phantom_lift_bars.py` — P0-1 docstring ADD/Appendix label.
- `README.md` — P0-1 Outcome-table ⚠️ estimand-label note.
- `aggregate_phase1_full_prereg_decision.py` — P1-1 post-R5 route + `_load_h10_operational_gate_passed`; P1-2 `_apply_b2_cross_family_downgrade`; P1-3 `_load_cell_per_task` total_billed; P2-4 GLM comment.
- `train_l1_router.py` — P0-2 `label_entropy_bits` + per-fold/per-cell entropy + `h10_entropy_gate.json` artifact; P1-6 holdout_sr diagnostic alias.
- `aggregate_h10_pareto.py` — P0-2 entropy DEFER read + fail-closed; P2-3 full-coverage default + `--allow-partial-dev`.
- `aggregate_cross_site.py` — P1-4 scaffold-adjusted canonical comment (code unchanged).
- `Makefile` — P2-3 dev target `--allow-partial-dev`.

Tests: `tests/test_amendment04_alignment.py` (20 cases: post-R5 route incl. AMENDMENT_02 L209-210 assertions, B2 downgrade, entropy bits). Full suite green at witness time: H10/router/prereg/framing/amendment04 = 174 passed.

## §5 — Witness statement

All changes are witnessed **before paper-grade H1/H2/H10 outputs are computed**. This
amendment introduces no new estimand, threshold, gate, R-ladder rule, or cost/latency
definition; it aligns analysis-layer code, figures, and prose to estimands already locked by
DOI-1 + AMENDMENT_01/02/03 and to the user-approved B-1669/B-1780 latency-canonicalization
decisions. Git tag `prereg-amendment-04-analysis-alignment-20260524` at the commit adding
this file is the internal witness; OSF upload appends to the DOI-1 anchor without modifying
its locked estimands.
