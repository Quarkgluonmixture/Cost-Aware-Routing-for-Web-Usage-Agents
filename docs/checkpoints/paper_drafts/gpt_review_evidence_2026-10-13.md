---
type: review-response-evidence
created: 2026-10-13
source: 3 searcher + 3 adversarial verifier agents over ledger.jsonl, conclusions/, 实验笔记 §528+, docs/analysis/cross_sites/, deliverables/coling2027
scope: GPT review of the COLING 2027 draft — three criticisms; "have we already done this?"
---

# GPT review: what we already have, what is genuinely missing

Verdict per criticism, then the evidence. Every number below was re-opened by a verifier in the cited
product / notebook section; items marked ⛔ must not be cited as they stand.

## C1 — "Only three cells were fully rerun"

**Verdict: true, and the paper already says so** (`limitations.tex`, Coverage of reruns). Existing work
does not raise the count of fully rerun cells; it shows the 3 cells agree with each other and that
no rerun quantity is transferred to unrerun cells.

What exists:
- 27 registered same-condition pairs (`noise_floor_inventory`, §557.2–557.3). Six-mode full: cls_B0,
  red_B0 (B0), wared_B1 (B1). Partial: cls_B1 3/6, red_B1 2/6, cls_B2 2/6, cls_B5 1/5, shop_B1 1/6.
  None: red_B2, shop_B0, WA_B0. Max k = 2 everywhere; no third replicate was ever registered.
- Why the gaps: paid reruns (B0 shop ×3 ≈ $135, WA red B0 ×6 ≈ $50, B5 ≈ $130) blocked at the
  $30 proxy floor (balance $30.18 on 09-15); the local chain was killed at 30 h on 09-24; A100
  unreachable 09-21..10-08 (`pre_run/local_replicate_chain_launch_intent_20260915.md`, §551, §528.3).
- Three-cell consistency: `rerun_union_extrapolation` (headroom share 52.8–168.6 / 51.5–167.5 /
  47.1–148.6 %), `label_instability` 7.95× / 11.0× / 11.4×, `replicate_metric_noise` 20/25, 25/25, 23/25.
- No-transfer discipline: `noise_floor_inventory` scope_warning, §470.5, §536.2, §529.4 (C1 retracted).

Closable without new runs:
1. **cls_B1 is a fourth complete cell under the three-mode test.** DOM, SoM and Vision all have
   registered reruns (3.12 / 0 / 0 %; † in `tab_sixmode`). It is missing from `routing_three_arm`
   null_cells only because `scripts/analysis/lib/replicate_pairs.py: full_paired_cells()` requires all
   six arms. §5's main test is over the three deployment modes, so it can be recomputed with 4 cells.
2. B2·red·dom (R5361) was running on the A100 (started 10-10 13:43Z). Pulling it needs the A100
   (condenser cert expires 10-15 01:15). It is one mode only: "no rerun" 3/11 → 2/11, full cells unchanged.
3. A per-cell coverage table (full / partial / none, d and power tier per pair, §469.7) and one sentence
   on why B0 shop / WA_B0 / B5 are missing (budget).
4. State the no-transfer rule in the text; state that the decision study's "2–6 runs" is projected from k = 2.

⛔ Do not use: "local backbones are near-deterministic on classifieds" — §557.2 explicitly declines this
(flip ceiling co-varies with SR). §470.5 "B1/B2 fully reproducible" — retracted by §529.4.
⚠ Wording: `2_setup.tex` says a rerun uses "the same tasks and code"; four of cls_B0's six B-runs are
~24 runtime commits after their A-run (§470, notebook L29728–29736). Code drift was judged non-dominant,
but the sentence is not literally true.

Needs new runs: k ≥ 3; B0 shopping, WA-reddit B0, B5 full reruns (API budget).

## C2 — "Rejecting the additive-difficulty null does not establish narrow task-level preference"

**Verdict: essentially right; the paper concedes it** (§5: mode-specific slopes would also reject the
null). The discriminating test was never run: §554.3 registers the 2PL null as "not done, left for the
author response"; nothing after §528 did it.

What exists (all only on the 3 full cells): Rasch + curveball null, three modes σ²_int
0.0333 / 0.0175 / 0.0303, p 0.0005 / 0.0020 / 0.0005; six modes 0.0252 / 0.0073 / 0.0267,
p 0.0005 / 0.0165 / 0.0005 (`task_mode_interaction_null`, `routing_three_arm`); stable strict preferences
54 / 24 / 19 vs null 41.5 / 17.1 / 11.0 (no multiplicity correction); decision study.

Closest evidence for a *narrow* preference: on classifieds B0 the intent-rule groups reverse difficulty
between text and screenshot modes (flagged vs other: DOM 7/71 vs 32/153, P-SoM 3/71 vs 32/153, but
SoM 21/71 vs 40/153, Vision 23/71 vs 33/153; rerun same direction) (`visual_intent_routing`). Task-class
level, one site, no interval, and a 2PL model can still produce group-mean reversals — "hard to explain",
not "ruled out".

Closable without new runs (per-task CSV in the supplement, 18,446 rows):
1. Fit a mode-specific-slope null (2PL / one-factor: logit = α_m θ_i + b_mr), parametric bootstrap,
   recompute σ²_int and strict-preference counts.
2. Opposite-direction preference counts within equal-total-success strata (X 2/2 & Y 0/2 vs Y 2/2 & X 0/2).
3. Report the intent-group difficulty reversal as task-class evidence, labelled descriptive.
4. Narrow the §5 title / "preferences" wording to "reproducible task-by-mode structure beyond an additive
   difficulty model" (abstract and intro (iii) already say "beyond an additive model").

⛔ Look relevant but do not answer this criticism:
- `routing_crossrun_template_validation` §3 persistence diagnostic — no null at all; routing never-solved
  tasks to the cheapest mode is itself a difficulty lever (§550.1).
- `conditional_failure_attribution` P16 / P43 enrichment — static config+mode rules (κ = 1.00, §555.3);
  P43 is the intent rule itself. Only P17 / P27 / P49 are behavioural, still without intervals.
- "First-step divergence" across modes — same-mode reruns already diverge at step 0 (shared-prefix
  median 0.03); `rerun_flip_failure_anatomy` pooled κ is now 0.599, verdict inconclusive (§557.2).
- `mechanism_per_task` VWA range is 0.306–0.406 (not 0.31–0.34).

## C3 — "Routers never reliably gained, and only two lightweight methods were tested"

**Verdict: first half true and is the paper's finding; second half true only of the COLING text.** The
project tested far more, but the COLING draft narrowed to two L2-LR routers, the abstract says "two
lightweight routers", and Limitations says "language-model routers may do better" — without disclosing
the LLM pilot that was run. `coling_frame_proposal.md` (L63, L84) planned an appendix of these pilots;
it was never executed. The VLM4RWD draft said "five routing constructions".

Other routing constructions already tested, all negative for which-representation routing:
- §368 classifier sweep on B0 classifieds: LR, GradientBoosting, RandomForest, LinearSVC+calibration,
  MLP × 3 feature pools incl. full TF-IDF — 15/15 below always-SoM 27.23 % (best LR 25.45 %, MLP 25.00 %).
  ⚠ product `docs/checkpoints/codex_outputs/router_model_sweep_2026-07-15.md` is missing on disk;
  regenerate from `scripts/analysis/router_model_sweep.py`.
- §505 GPT-5.6 prompt router pilot (v1–v3, 11 cells) — all within the rerun band; pilot only (position
  bias not ablated, B5 routes itself, no significance test, untracked product).
- Confidence cascade Vision→SoM: beats always-rich in 0 comparable cells; +0.10..+3.95 pp vs random,
  in-sample max over 10 signals.
- Pooled family × cost-tier: 0/6 dominate (0/26 at the most favourable corner, §399).
- kNN / FrugalGPT-style (§374, one cell, "-style"); ⚠ product missing, regenerate.
- Contextual bandit replay 11/11 lose to fixed-best; rule forward/mined 0/20 hold on rerun;
  one-step lookahead only +0.5 pp (inside band); two-arm action labels collapse to always-cheapest.
- Why they fail: interaction/noise 0.34 / 0.12, supply–value coupling ρ = 0.952.
- Difficulty decisions do pay: abstention saves 6.0–24.6 % at ≤ 5 % loss; step-budget saves 41 % cost,
  14/18 pairs right direction (but its look-ahead test crosses 0 with sampling uncertainty, §542).

Closable without new runs: an appendix "Other routing constructions" table from the above; disclose the
LLM pilot with its defects instead of "may do better"; restrict "no positive gain" explicitly to
which-representation routing.

⛔ Do not cite as current: the §544 "95 % upper bound −0.14" (dropped in §554); any §505 row containing
cls_B5 Vision (coordinate-contract defect, 85 % missed clicks; retracted_2 R2#144/#177) until B-1997 rerun;
B5 nested-half route-away counts (pending).

Genuinely missing: a non-pilot LLM router; learning-curve / training-error check on the largest cell
(R2#156); routers trained on multi-run-averaged labels; live / step ≥ 1 routing; simultaneous intervals
across policies and budgets; time-cost versions of §544 / §546.
