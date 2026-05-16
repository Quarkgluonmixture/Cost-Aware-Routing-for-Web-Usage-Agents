---
type: design-proposal
status: draft-v1
created: 2026-05-16
purpose: stress-target artifact for 3-AI router design synthesis (Mode A+B+C)
hypothesis-tags: H9 (rule-based router), H10 (learned classifier)
preregistration-anchor: docs/checkpoints/pre_run/preregistration.md §354 + §359
---

# Router Design Proposals v1 — 2 candidate designs

> Draft artifact for `/stress` adversarial review. **Not locked**. Estimand `<pending>` per preregistration §2 H9/H10 stub.
> Both designs share the same locked baselines (preregistration §359 best-single-mode + Tier-0 random) and the same locked CV protocol (preregistration §354 5-fold site-stratified, seed=42).
> Difference between P1 vs P2 = **feature → decision** mapping (rule-based handcrafted thresholds vs learned classifier).

## Shared substrate (both proposals)

- **Mode universe**: 6 modes = `{dom, som, vision, phantom_text, phantom_prompt, phantom_som}`
- **Data**: 36 conditions × per-task per-mode SR + step JSONL + `confidence_summary.json` per condition
- **CV protocol** (preregistration §354 locked): 5-fold site-stratified task-CV, seed=42, min test fold ≥ 40 tasks
- **Best-single-mode baseline** (preregistration §359 locked): per cell, mode with highest mean adjusted-SR on **train fold** (split-stratified to prevent test leak)
- **Statistical gate** (H9/H10 pending): router lift over best-single-mode ≥ 1.0pp, ≥ K cells Holm-significant — *K and δ pending advisor lock*
- **Cost axis**: paper §1 hero requires router preserves 4-fold drop-in property — i.e., router-induced extra-mode-switch cost (image-token tax if escalated to SoM/Vision) must be amortized

## Proposal P1 — Rule-Based Router (handcrafted task-attribute + online trigger hybrid)

### Decision logic (deterministic, zero-parameter beyond thresholds)

```
def decide(task_intent: str, obs_1: dict, step_state: dict) -> mode:
    # Layer 1 — task-attribute pre-routing (offline at task start, before step 1)
    if has_reference_image(task_intent):       # cls Cat B
        return "som"                            # full SoM (image + marks)
    if is_pure_search_intent(task_intent):     # red r-1 / cls c-3
        return "dom"                            # cheapest, no image needed
    # Layer 2 — first-step browser-state escalation (online at step 1)
    if obs_1.dom_size > θ_dom OR obs_1.axtree_complexity > θ_cmplx:
        return "som"                            # rich obs needed for dense pages
    # Layer 3 — default + in-trajectory escalation (extends existing scaffold)
    return "phantom_som"                        # deployment hero default
    # ... existing trigger-based escalation (unchanged_streak ≥ 2 → escalate up
    # one mode; success_streak ≥ 3 → de-escalate down one mode) carries from
    # p79/experiment/router.py::RuleBasedRouter
```

### Feature spec

| Layer | Feature | Source | Type |
|---|---|---|---|
| L1 task | `has_reference_image` | manual binary tag (already curated, §139 audit Cat B) | bool |
| L1 task | `is_pure_search_intent` | keyword regex on intent: `^(find\|search\|locate\|how many)` | bool |
| L2 step-1 | `dom_size` | `len(obs_text)` at step 1 | int |
| L2 step-1 | `axtree_complexity` | `state_change.compute_axtree_complexity(obs_1)` (existing primitive) | int |
| L3 traj | `unchanged_streak`, `success_streak` | existing `RouterState` (no change) | int |

### Threshold setup

- `θ_dom = 12000` (existing scaffold default; **stress target — uniform across 6 modes is suspect**)
- `θ_cmplx = 500` (existing scaffold default)
- All thresholds tuned via **train-fold-only grid search** over {6000, 12000, 18000, 24000} × {250, 500, 750, 1000} — picks max train-fold SR-per-cost frontier per (site, model) cell

### Baselines & evaluation

- **vs Tier-0 random** (uniform over 6 modes)
- **vs Best-single-mode** (preregistration §359 locked anchor; train-fold-stratified per cell)
- **vs Oracle ceiling** (per-task argmax SR upper bound)
- **vs Proposal P2** (learned classifier head-to-head)

### Strengths (per design intent)

1. Zero training → no overfitting risk at N=234 cls + 210 red per cell
2. Interpretable (rules grep-able for reviewer)
3. Extends existing `p79/experiment/router.py` scaffold (already in-tree, 149 LoC)
4. Threshold grid is finite — easily preregisterable

### Known weakness (acknowledged for stress target)

- Manual task-attribute taxonomy (Cat A/B/C/D from §139 audit) may not generalize to held-out site (Phase 1b shop)
- `has_reference_image` is a hand-labeled signal — counts as test-time leak if labels derived from gold answers
- Layer 1 rules dominate Layer 2/3 hierarchically — if L1 hit rate >50%, Layer 2/3 contribute marginal lift

---

## Proposal P2 — Learned Classifier Router (TF-IDF + multinomial LR, task-level)

### Architecture

```
input:  task_intent (str) + first-step obs (dict) + (site, model) categorical
output: predicted argmax-mode ∈ 6 modes
model:  Logistic Regression (multinomial, L2 reg, sklearn default)
training: per (site, model) cell, on train fold of 5-fold CV
```

### Feature spec

| Group | Feature | Dim | Source |
|---|---|---|---|
| **F1 task text** | TF-IDF on `task_intent` | 3000 (top features) | sklearn TfidfVectorizer |
| **F2 first-step text** | TF-IDF on first 500 chars of `obs_1.obs_text` | 1500 | sklearn TfidfVectorizer |
| **F3 categorical** | site one-hot (2), task category one-hot (4 from manual audit) | 6 | hand-curated |
| **F4 browser state** | `dom_size`, `axtree_complexity`, `image_count`, `form_count` | 4 (scaled) | step-1 obs |
| **F5 signal** | per-mode AUROC signal (verbalized + behavioral) from `confidence_summary.json` | 12 (6 modes × 2 signals) | existing |

Total: ~4520 dim, L2 regularized.

### Label: argmax_mode SR per task

Per training task `t`, label = `argmax_m SR(t, m)` over 6 modes evaluated on **train-fold-only data**. Ties broken by cost-cheaper mode (DOM > P-SoM > P-text > P-prompt > SoM > Vision cost order).

### Training & evaluation

- **CV protocol**: preregistration §354 locked = 5-fold site-stratified, seed=42, min test fold ≥ 40 tasks
- **Cross-model holdout**: train per cell, evaluate transfer to other models (B0 trained → B1/B2 test) for §7 cross-capability claim
- **Cross-site holdout (LOSO alternative)**: cls-trained → red test, red-trained → cls test (§354 lists this as pending alternative)
- **Loss**: standard multinomial cross-entropy; cost-weighted variant ablated (weight inversely proportional to mode's mean cost)

### Baselines & ablations

- **vs Tier-0 random**, **vs Best-single-mode**, **vs Oracle**, **vs Proposal P1** (head-to-head)
- **Feature ablations**: F1 only / F1+F3 / F1+F3+F4 / F1+F3+F4+F5 — measure marginal contribution of each group
- **Sample efficiency curve**: train on 25% / 50% / 75% / 100% of train fold, plot router SR vs train-N

### Strengths (per design intent)

1. End-to-end optimizable; cost-aware loss directly targets Pareto frontier
2. Multi-source feature fusion (text + categorical + browser + signal) — captures interactions hand rules miss
3. Sample efficiency curve directly addresses reviewer "is your data enough" concern
4. Cross-model transfer evaluation gives §7 generalization handle

### Known weakness (acknowledged for stress target)

- N_train per cell ≈ 180-190 tasks × 4 folds = small for 4520-dim feature → severe overfitting risk
- TF-IDF on 234 cls + 210 red tasks → vocabulary likely small; F1+F2 may collapse to ~50 effective dims
- F5 signal features include per-mode AUROC from same data being routed → potential leak
- Cross-model transfer assumes shared optimal-mode-per-task structure across capability tiers (B0 235B vs B2 4B)

---

## Comparative matrix (P1 vs P2)

| Dimension | P1 Rule-Based | P2 Learned Classifier |
|---|---|---|
| Training | zero (handcrafted) | per-cell LR fit |
| Parameters | ~8 thresholds | ~4520 × 6 LR weights = ~27K |
| Interpretability | high (grep-able rules) | medium (LR coefficient inspection) |
| Generalization risk | rule taxonomy leak (handlabel) | overfitting (low N / high dim) |
| Extends existing infra | yes (`router.py` 149 LoC) | no (new module `router_learned.py`) |
| Engineering effort | ~3 days | ~5-7 days |
| Reviewer attack surface | "rules picked post-hoc to fit data" | "27K params on 180 tasks = overfit" |
| Cost-aware target | implicit via threshold + de-escalation | explicit via cost-weighted loss |

## Open design questions (pre-stress)

1. **Estimand**: pooled FE meta over 6 cells (mirrors H1) vs per-cell paired bootstrap — *advisor pending*
2. **Sample size adequacy**: 5-fold × ~180 train tasks for 4520-dim P2 — power calc TBD
3. **Cross-site protocol**: 5-fold site-stratified (mixed) vs LOSO (clean) — preregistration §354 lists both, advisor lock pending
4. **Tie-break order**: P1 Layer 1 rules priority vs Layer 2 trigger priority — currently L1 dominates, justification needed
5. **Cost normalization**: USD/episode (B0 API) vs kWh/episode (B1/B2 local) — paper §3d shows ~100× gap, router target must handle both deployment classes

---

## /stress contract

This document is **stress target**. Reviewer attacks should:
- Quote specific section / row / threshold / dim count
- Identify design / methodology / leak vector
- Propose specific defuse (file:line OR equation OR experiment)
- Tag severity (P0 launch-blocking / P1 paper-grade / P2 defer)
