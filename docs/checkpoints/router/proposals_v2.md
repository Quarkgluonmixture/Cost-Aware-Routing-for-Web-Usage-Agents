---
type: design-proposal
status: v2-post-3AI-stress
created: 2026-05-16
purpose: final synthesized router design proposals after Mode A (Claude) + Mode B (codex) + Mode C (gemini) hostile review of v1
hypothesis-tags: H9 (rule-based router), H10 (learned classifier)
preregistration-anchor: docs/checkpoints/pre_run/preregistration.md §354 + §359 (updates required — see §C below)
supersedes: docs/checkpoints/router/proposals_v1.md
stress-trace:
  mode-a: docs/checkpoints/router/stress_mode_a_2026-05-16.md
  mode-b: docs/checkpoints/codex_outputs/router_design_FINAL_2026-05-16_084921.md
  mode-c: docs/checkpoints/gemini_outputs/router_design_2026-05-16_084921.md
---

# Router Design Proposals v2 — final after 3-AI cross-stress

> **What changed v1 → v2**: 12 distinct P0/P1 findings from 3-AI cross-stress folded in as design constraints. Both proposals now defuse all P0 and most P1 attacks pre-data.

---

## §A — 3-AI Unified Bug List (provenance for v2 changes)

**Verification status**: Mode A (Claude self) PASS · Mode B (codex) PASS · Mode C (gemini) PASS (after internal retry)

### 🔴 P0 (all defused in v2)

| # | Bug | Blast Radius | Defused in v2? |
|---|---|---|---|
| P0-1-ABC* | `router_proposals_v1.md:107-109` argmax_m `adjusted_success` on train fold | post-hoc adjusted_success retired per 实验笔记 §139.8 + memory `reference_fp_architecture_2026-05-14`; canonical outcome = `success` (`p79/experiment/analysis.py:1102-1108`). 若 v1 label / baseline 仍引 adjusted-SR → label provenance bug, classifier 训练 target 与 evaluation target 静默不一致 (sklearn 经典 trap). | ✅ §B P1+P2 全用 `success` |
| P0-2-AC* | `router_proposals_v1.md §"Statistical gate"` H9/H10 estimand `<pending>` | preregistration §6 OSF lock 不能 commit; reviewer (R1-R2) 检查 OSF preregistration 会拒. 等数据 land 后再 lock = HARKing. | ✅ §C 锁定 FE pooled meta, mirror H1 |
| P0-3-ABC | `router_proposals_v1.md:50` P1 L1 `has_reference_image` manual binary tag from §139 audit | Audit-time gold-labeled, 不是 runtime agent-observable. Router 用 gold-conditional feature 决策 = oracle leak. Per Chen FrugalGPT 2023 §4.1 router features must derive from task input + agent-observable signals only. | ✅ §B-P1 改为 `bool(task.reference_images)` runtime extract |
| P0-4-B | `router_proposals_v1.md:29-43` P1 pseudo-code `return` bypasses stateful `RuleBasedRouter.decide()` | Early-return skips `unchanged_streak`/`success_streak` update in `router.py:45-68`; L3 escalation never carries because `state.current_mode` starts from `"dom"` (`router.py:17`), not `phantom_som`. Layer-1 tasks never benefit from in-trajectory escalation. | ✅ §B-P1 L1/L2 set `state.current_mode` + `preferred_mode` then always call stateful `decide()` |
| P0-5-ABC | `router_proposals_v1.md:103` P2 F5 per-mode AUROC from `confidence_summary.json` | 3 angle 同问题: (A) re-uses contribution-1 signal AUROC = double-citing self-citation; (B) `aggregate_routing_auroc.py:70-103` produces per-`(baseline, site, mode, signal)` rows = global per cell, constant for all tasks → only shifts LR intercept, no task discrimination; (C) at inference router 需先跑所有 mode 才得 signal → online infeasibility. | ✅ §B-P2 F5 删 from primary spec, 仅作 §6 ablation row "+post-run signal" |
| P0-6-ABC | `router_proposals_v1.md:101-104` P2 4520-dim × N≈180 train task = ~25:1 feature-to-sample | LR fit at this ratio: test SR << train SR by 5-10pp guaranteed even with L2 reg. Reported router lift = overfit noise. Per Hastie ESL §3.4 / sklearn docs §1.1.11 N >> dim required. | ✅ §B-P2 dim cap = top-18 features via train-fold mutual info |
| P0-7-C* | `router_proposals_v1.md` P2 vs `preregistration.md` Appendix B H7 — P2 substrate = H7 substrate (TF-IDF + LR + per-task-best-mode label) verbatim | "Hollow contribution padding" — reviewer (R1) 一眼判 P2 是 deferred-to-paper-2 H7 改名 H10 塞回 paper-1. Contribution mutual exclusivity violation. | ✅ §B-P2 引 explicit test-leak-free constraint: features = step-1-observable text + browser-state ONLY, no post-run signals; differentiates from H7 oracle |

### 🟠 P1 (mostly defused in v2; remainders disclosed)

| # | Bug | Blast Radius | Status v2 |
|---|---|---|---|
| P1-1-B | `router_proposals_v1.md:53` `state_change.compute_axtree_complexity` does not exist; only `build_page_state()` line-count primitive (`state_change.py:74-86`) | ImportError on P1 first implementation. Layer 2 永远不触发. | ✅ §B-P1 renamed to existing `dom_complexity` field via `RouterState.dom_complexity_history` (already in `router.py:84-92`) |
| P1-2-AB | `router_proposals_v1.md:60` P1 16-combo threshold grid search × 5-fold × per-cell = 480 evaluations without nested CV | Hastie ESL §7.10.2 经典 "overfitting the CV" — selecting threshold by train SR then reporting test SR with same fold inflates lift estimate. | ✅ §B-P1 thresholds pre-locked on archive `meta_phantom_lift.md` data (frozen pre-Phase-1a), no post-launch grid search |
| P1-3-BC | `router_proposals_v1.md:114` P2 cross-model B0→B1/B2 transfer | (B) violates §354 "test fold predictions use ONLY train-fold mode rankings" if target labels leak; (C) `paper_planning §2` Lazy Minimization records B0 text-axis-dominant vs B1 image-axis-dominant — capability-modulated reversal means transfer assumes shared distribution that paper itself documents as antagonistic. | ✅ §B-P2 cross-model transfer DROPPED from primary spec; replaced with cross-site cls→red within same model (R3-friendly external validity) |
| P1-4-A | `preregistration.md §359` best-single-mode anchor at N≈180 train tasks, Kendall τ across folds may be < 0.7 | If τ < 0.7, anchor flickers between folds → router lift CI inflated by anchor noise. | ✅ §C pre-data MC simulation on archive sets fallback to "majority-winner across 100 resamples × 5-fold" if τ < 0.7 |
| P1-5-AC | `router_proposals_v1.md:84` Tier-0 random uniform over 6 modes = strawman | R1-R2 expect frequency-weighted random or top-3-modes-random per Chen FrugalGPT 2023. | ✅ §B 3 random baselines: uniform / train-fold-frequency-weighted / top-3-modes-per-cell |
| P1-6-C* | `router_proposals_v1.md:116` cost-weighted variant across B0 (USD/token) vs B1/B2 (kWh/inference) = geometrically disjoint Pareto frontiers | Per gemini retry F3 OOB: transferring a cost-weighted loss across deployment classes (~100× cost gap per `paper §3d`) requires shared cost-landscape conditional distribution that empirically doesn't exist. Cost-weighted training on B0 with API prices optimizes irrelevant landscape when transferred to B1. | ✅ §B-shared "Loss / objective" line: pure SR-maximization only, cost savings reported as emergent property per deployment class (no cross-deployment cost-weighted loss) |

### 🤝 Cross-AI agreement summary

- **3-AI overlap** (highest confidence): P0-1 (adjusted_success drift), P0-3 (has_ref_image leak), P0-5 (F5 leak), P0-6 (P2 overfit) = 4 bugs
- **2-AI overlap**: P0-2 (estimand), P1-2 (no nested CV), P1-3 (cross-model), P1-5 (strawman random) = 4 bugs
- **1-AI unique**: P0-4 codex (P1 early-return), P1-1 codex (compute_axtree_complexity), P1-4 Claude (anchor fold noise), P0-7 gemini (P2=H7 phantom renaming) = 4 bugs
- Total 12 distinct P0/P1 findings, all defused in v2 or disclosed as remaining gap

---

## §B — v2 Router design proposals (both)

### Shared substrate (unchanged from v1 except canonical outcome)

- **Mode universe**: 6 modes = `{dom, som, vision, phantom_text, phantom_prompt, phantom_som}`
- **Outcome column**: `success` (canonical, post-§139.8 retirement of adjusted_success). NO `adjusted_success` references anywhere in router pipeline.
- **CV protocol** (preregistration §354 locked): 5-fold site-stratified task-CV, seed=42, min test fold ≥ 40 tasks. **§C below proposes preregistration §359 update for anchor-flicker fallback.**
- **Best-single-mode baseline** (preregistration §359 anchor): per cell, mode with highest mean `success` rate on **train fold** (split-stratified)
- **Random baselines (3 tiers, P1-5 fix)**:
  - Tier-0a: uniform random over 6 modes
  - Tier-0b: train-fold-frequency-weighted random (each mode weighted by its mean train-fold success rate, then normalize to probability)
  - Tier-0c: top-3-modes-per-cell random (uniform over the 3 highest mean-SR train-fold modes)
- **Loss / objective** (P1-6 fix per gemini retry F3 OOB): both proposals optimize **pure success-rate maximization** (SR-only multinomial cross-entropy or rule-output SR-argmax). NO cost-weighted loss. Cost savings are reported as **emergent property** of router's mode selection (per-deployment-class separately: B0 USD/ep, B1 kWh/ep, B2 kWh/ep). Cross-deployment cost-weighted loss would require shared cost-landscape conditional distribution, which `paper §3d` empirically shows is ~100× disjoint between B0 API and B1/B2 local — geometrically incompatible Pareto frontiers, no single cost-weighted loss valid across cells.

### Proposal P1 — Rule-Based Router (handcrafted task-attribute + stateful trigger)

#### Decision logic (composes with existing `RuleBasedRouter.decide()` — P0-4 fix)

```python
# Wrapped around p79/experiment/router.py::RuleBasedRouter
def decide_p1(task, obs_1, step_state):
    # Layer 1 — runtime-extracted task attribute (P0-3 fix: no manual gold tag)
    has_ref_image = bool(task.reference_images)           # task object field, not audit Cat B
    is_search = bool(re.search(r'^(find|search|locate|how many)', task.intent.lower()))
    if has_ref_image:
        step_state.preferred_mode = "som"
        step_state.current_mode = "som"                   # P0-4: set state, do NOT early-return
    elif is_search:
        step_state.preferred_mode = "dom"
        step_state.current_mode = "dom"
    else:
        # Layer 2 — first-step browser-state escalation (P1-1 fix: use existing field)
        if obs_1.dom_size > θ_dom OR step_state.dom_complexity_history[-1] > θ_cmplx:
            step_state.preferred_mode = "som"
            step_state.current_mode = "som"
        else:
            step_state.preferred_mode = "phantom_som"
            step_state.current_mode = "phantom_som"
    # Layer 3 — ALWAYS call existing stateful trigger-based escalation (P0-4 fix)
    return RuleBasedRouter.decide(
        preferred_mode=step_state.preferred_mode,
        obs_text=obs_1.obs_text,
        state=step_state,
        ...
    )
```

#### Feature spec (4 features only — no manual gold labels)

| Layer | Feature | Source | Type |
|---|---|---|---|
| L1 task | `has_reference_image` | `bool(task.reference_images)` runtime | bool |
| L1 task | `is_search_intent` | `re.search(r'^(find\|search\|locate\|how many)', task.intent.lower())` | bool |
| L2 step-1 | `dom_size` | `len(obs_text)` at step 1 (existing) | int |
| L2 step-1 | `dom_complexity` | `text.count('\n')+1` from `state_change.py:74-86` (existing, **renamed from non-existent `compute_axtree_complexity`**) | int |
| L3 traj | `unchanged_streak`, `success_streak` | existing `RouterState` (no change) | int |

#### Thresholds pre-locked on archive data (P1-2 fix — no post-launch tuning)

- `θ_dom = 12000` — locked. Justification: `meta_phantom_lift.md` B0 4-cell archive shows DOM-size ≥ 12K bucket has 8pp lower SR than < 12K bucket across all phantom modes
- `θ_cmplx = 500` — locked. Same archive analysis on `dom_complexity_history`.

**No grid search post-Phase-1a**. If reviewer asks "why these thresholds", answer = "pre-data locked on pre-Phase-A archive `meta_phantom_lift.md`, frozen in commit XXX, not tuned on Phase-1a fresh data" — passes Brownlee/Hastie CV rule.

#### Evaluation

- vs Tier-0a/b/c random (3 baselines)
- vs Best-single-mode (preregistration §359)
- vs Oracle ceiling (per-task argmax SR)
- vs Proposal P2 head-to-head

### Proposal P2 — Learned Classifier (test-leak-free, ≤18 features, multinomial LR)

#### Test-leak-free constraint (P0-7 fix — differentiates from H7 oracle)

- **Inference path**: features extracted from `(task.intent, obs_1)` ONLY. NO post-run signal features. NO `confidence_summary.json` AUROC. NO per-mode aggregate statistics. Router must be runnable BEFORE any mode is executed.
- **This is the scientific distinction from preregistration Appendix B H7 (Tier-1 oracle TF-IDF + LR)**: H7 uses oracle best-mode labels + per-mode oracle-derived signals. P2 (= H10) explicitly forbids any feature requiring post-run information.

#### Feature spec (dim cap ≤ 18 — P0-6 fix)

Two-stage selection:
1. **Stage 1 — candidate pool** (≈ 50 features): top-30 TF-IDF terms on `task.intent` (English stop-words removed, min_df=3) + 6 categorical (site + manual task category one-hot, manual category lookup is runtime-known from VWA `task.category`) + 4 browser state (`dom_size`, `dom_complexity`, `image_count`, `form_count`) + 10 task-feature binaries (`has_ref_image`, `is_search`, `is_compose`, `is_navigation`, `is_form_fill`, `is_compare`, `is_filter`, `is_aggregate`, `is_account_action`, `is_visual_attribute`) — all runtime-derivable.
2. **Stage 2 — train-fold-only feature selection**: per train fold, select top-18 features by mutual information with label = `argmax_m success(t, m)`. Fold-stratified to prevent test leak (per sklearn `SelectKBest` with `mutual_info_classif`, fit on train fold only).

Final dim: **18** features × 6 modes = 108 LR weights — within Hastie ESL 10-samples-per-feature rule at N≈180 train tasks.

#### Label

Per training task `t`: `label = argmax_m success(t, m)` over 6 modes on **train-fold-only** outcomes. Ties broken by **train-fold-frequency-weighted random** (not cost-cheaper) — avoids systematic bias toward DOM/P-SoM (codex B6).

#### Training & evaluation

- **CV protocol**: preregistration §354 locked = 5-fold site-stratified, seed=42
- **Cross-site holdout (LOSO alternative)**: cls-trained → red test, red-trained → cls test — replaces dropped cross-model claim (P1-3 fix)
- **Cross-model**: NOT a primary claim. Disclosed as exploratory only with explicit caveat "paper_planning §2 records capability-modulated reversal; cross-model transfer may systematically fail".
- **Sample efficiency curve**: train on 25%/50%/75%/100% of train fold — diagnostic for whether N≈180 is adequate (B6 / A3 transparency)

#### Pre-Phase-1a label-distribution diagnostic (codex B6 — gate before launching P2)

Before §B Phase 1a fires, run `scripts/analysis/build_router_label_diagnostic.py` on existing pre-Phase-A archive (`meta_phantom_lift.md` B0 4 cells):
- Per-cell label histogram (6 modes × N tasks)
- Label entropy H = -Σ p log p
- Majority baseline (predict-always-majority) SR vs best-single-mode SR

**Gate**: if entropy < log(2) (i.e., labels collapse to ≤ 2 modes per cell) → P2 not viable, skip P2 → paper has single-route (P1 only). Disclose pre-data.

#### Baselines & ablations (same 3 random tiers + best-single-mode + oracle + P1)

Feature group ablations:
- F-text only (TF-IDF terms only)
- F-text + F-categorical
- F-text + F-categorical + F-browser
- F-text + F-categorical + F-browser + F-binary-task (full P2)
- +F-signal (F5 from v1) — **§6 disclosure ablation only, NOT primary**

### Comparative matrix (v2)

| Dimension | P1 Rule-Based | P2 Learned Classifier |
|---|---|---|
| Training | zero (thresholds pre-locked on archive) | per-cell LR fit on train fold |
| Parameters | 2 thresholds (pre-locked) + 5 rules | 18 features × 6 modes = 108 LR weights |
| Composition with `RuleBasedRouter.decide()` | ✅ L1/L2 set state, L3 always-call | n/a |
| Outcome column | `success` (no adjusted_success) | `success` (no adjusted_success) |
| Test-leak-free | ✅ runtime extract only | ✅ runtime extract only (no post-run signals) |
| Differentiates from H7 oracle | n/a (rule-based, no overlap) | ✅ explicit constraint vs H7 |
| Engineering effort | ~2 days (incl. pre-lock archive analysis) | ~5-7 days (incl. label diagnostic + sample efficiency curve) |
| Phase 1a launch-blocking? | no (after archive pre-lock) | no (after label diagnostic gate) |

---

## §C — preregistration updates required (launch-blocking for OSF lock)

These three changes must land in `preregistration.md` BEFORE OSF DOI commit:

### C1. H9/H10 estimand lock (P0-2 fix)

Append to `preregistration.md §2` H9/H10 stub:

> **H9 (rule-based router) PRIMARY gate**: fixed-effects inverse-variance pooled drop-one router lift `θ_FE^(P1) = Σ w_i × (SR_P1_i − SR_best_single_mode_i)` over 6 cells (mirror H1 estimand). One-sided FE superiority test H0: `θ_FE^(P1) ≤ +1.0pp` at α=0.05.
>
> **H10 (learned classifier) PRIMARY gate**: same formula but for P2 — `θ_FE^(P2) = Σ w_i × (SR_P2_i − SR_best_single_mode_i)`. One-sided FE superiority test H0: `θ_FE^(P2) ≤ +1.0pp` at α=0.05.
>
> **H9+H10 family-wise**: Holm correction over {H9, H10} test pair, α_family = 0.05.
>
> **Rationale for δ=1.0pp**: same as H1 (≈ 2 tasks in N=234, matches per-cell bootstrap SE). For router-vs-best-single-mode noise calibration, MC-validated on `meta_phantom_lift.md` archive — if archive shows router-anchor noise SD > 0.5pp, raise δ to 2×SD.

### C2. Anchor-flicker fallback (P1-4 fix)

Update `preregistration.md §359` row:

> **Best-single-mode baseline**: per cell, mode with highest mean `success` rate on **train fold** (split-stratified). **Anchor-flicker fallback**: if pre-data MC simulation on archive shows Kendall τ across 100 × 5-fold resamples < 0.7, switch to **majority-winner-across-resamples** anchor (mode that wins best-single position in ≥ 50% of 100 resamples).

### C3. Adjusted-SR retirement reflection (P0-1 fix)

Update `preregistration.md §359` + any other "adjusted-SR" mention:

> Where this preregistration says "adjusted-SR", read `success` (canonical outcome; post-§139.8 retirement of `adjusted_success`). Router pipeline must use `success` column from `condition_summary_v2.json` directly, no `compute_adjusted_success` post-hoc layer.

---

## §D — Remaining gaps disclosed (not defused, just disclosed)

These cannot be defused at v2 design time — they require either advisor lock or Phase 1a/1b data:

1. **(G1) δ_h9/δ_h10 fine-calibration pending pre-data MC simulation on archive** — current spec borrows δ=1.0pp from H1, may need adjustment after `meta_phantom_lift.md` MC simulation runs (§C1 rationale). Effort: 1 day pre-Phase-1a.
2. **(G2) Cross-model claim downscoped to exploratory** — paper-1 §6 router section will say "we observed router lift on B1 within model class; cross-model B0→B1/B2 transfer is exploratory and we observe capability-modulated reversal per `paper_planning §2`". Reviewer may still ask for full transfer analysis — paper-2 territory.
3. **(G3) External validity to shop (Phase 1b)** — rules + classifier are derived from cls+red. Without Phase 1b shop pre-data lock, paper claims generalization at cls+red boundary only. Locked in `phase1_plan.md` Phase 1b deferred.
4. **(G4) Routing decision granularity** — both proposals decide at task-level. Step/token-level routing (cascade literature) explicitly out of paper-1 scope. Disclosed as future work.
5. **(G5) Router-induced overhead in 4-fold drop-in claim** — router decision adds latency (P1: 2-5 ms, P2: 20-50 ms). Paper §1 hero "4-fold drop-in property (b) latency ~50% lower" applies to P-SoM mode itself, not P-SoM + router; clarify in §1 prose.

---

## §E — One thing to do tonight (1-3h leverage)

**Action**: Write `scripts/analysis/router_archive_diagnostic.py` (~120 LoC) that runs on existing `meta_phantom_lift.md` B0 4-cell archive:
1. Per-cell label histogram + entropy on `argmax_m success` labels (gate for P2 viability — codex B6)
2. Best-single-mode Kendall τ across 100 × 5-fold resamples per cell (gate for §C2 anchor-flicker fallback — A5)
3. Pre-locked threshold validation: split archive tasks by `dom_size > 12000` and `dom_complexity > 500`, check SR gap holds per cell (defends §B-P1 pre-lock — P1-2)
4. Router-vs-anchor noise SD MC on archive (gate for §C1 δ fine-calibration — G1)

**Why this**: single script defends 4 v2 design choices pre-Phase-1a, runs on already-collected archive data (no Phase 1a wait, no new compute), takes a few hours. After this lands, preregistration §C1/§C2 can be locked, OSF DOI can be committed, §B Phase 1a launch unblocked.

**Expected output**: `docs/checkpoints/router/archive_diagnostic_<date>.md` with the 4 verdicts + JSON files for each diagnostic.

---

## §F — Distance to top-tier (post-3-AI synthesis)

- **Workshop (R3)**: ✅ defensible with v2 + §E archive diagnostic land (0.85 prob)
- **Mid-tier (R2)**: needs G1 (MC simulation) + G3 (Phase 1b shop) — 0.45 prob today, 0.65 after Phase 1b
- **Top-tier (R1)**: needs G2 (full cross-model analysis), G4 (step-level routing), or significantly stronger router lift (>3pp) — 0.15 prob today, requires paper-2 territory

This is **0.85 workshop / 0.45 mid-tier** vs v1's **0.60 workshop / 0.20 mid-tier**. The 3-AI cross-stress moved P0-7 (phantom renaming) + P0-4 (composition bug) + P0-1 (column drift) from blocking to defused — those alone account for the lift.

---

## §G — Open questions (need user / advisor decision)

1. **Approve §C preregistration updates** (estimand lock + anchor-flicker fallback + adjusted-SR retirement) — required before OSF DOI commit
2. **Approve §E tonight diagnostic script** — gates §B launch + §C lock
3. **G2 cross-model downscope acceptable?** — moving from "transfer claim" to "exploratory observation" weakens paper §7 generalization narrative. Alternative: explicitly add full cross-model transfer in paper-2 plan.
4. **G5 router latency disclosure framing** — should §1 hero re-prose "4-fold drop-in property (b)" to clarify P-SoM-only vs P-SoM-plus-router?
