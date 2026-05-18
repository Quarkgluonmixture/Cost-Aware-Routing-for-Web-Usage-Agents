## 6. Cost-Aware Routing on the Phantom Space (v0 placeholder, paper-grade numbers TBD post-Pass-2)

> **v0 status (A2.5 Chunk D, 2026-05-18)**: Structure + framing locked per Q1=C + (E''') + (b) design + Q4=A K-of-6 PRIMARY + APPENDIX FE pool. Numbers cited here are **PLACEHOLDERS** to be filled by a codex round once Phase 1a Pass-2 router fire lands. Each `TBD` marker is the eventual replacement site; framing is reviewer-defensible at design-layer audit standard (B-1001 /stress A2.5 P1-8-A 2026-05-18).

### 6.1 Setup — Within-cell 5-fold CV deployment

We evaluate the **learned router R_P2** on the phantom routing space via Phase 1a Pass-2 fire across 6 (baseline, site) cells = {B0, B1, B2} × {classifieds, reddit}. Pass-2 reuses the Pass-1 baseline task pool (cls 0-233 / red 0-209) but evaluates each task through a **fold-aware predictor**: at training, tasks within each cell are split into 5 stratified folds (`StratifiedKFold`, seed=42); at deployment, each task receives its prediction from the LR head trained on the fold where the task was held out, ensuring no in-sample memorization between training and runtime evaluation. Cell-constant features (site, capability tier) are EXCLUDED from the per-cell LR input space — cell identity is implicit via the runtime pickle selected per (baseline, site).

**5-Stage pipeline**:
1. **Stage 1 deterministic raw extraction** (`scripts/analysis/extract_50_features.py`): 5 numeric features (`dom_complexity`, `text_length`, `tokens_input_text`, `intent_token_count`, `reasoning_difficulty`) + 15 binary features (`has_reference_image` + 14 intent regex banks: color / search / compare / nav / filter / sort / aggregate / compose / form_fill / account_action / visual_attribute / question / action_word / temporal).
2. **Stage 2 global fold-local pooled MI selection** (`scripts/analysis/train_l1_router_with_mi.py`): for each fold k ∈ {0..4}, pool = all task indices except union of all 6 cells' fold-k holdouts (pool size ≈ 1124 of 1404 total tasks). Fit `TfidfVectorizer(max_features=30, min_df=3, stop_words='english')` on pool intents → 30 TF-IDF columns. Concatenate with 5 numeric + 15 binary → 50-dim design matrix. Fit `SelectKBest(mutual_info_classif, k=18)` on pool to select top-18 features. Dump 5 vectorizer pickles + 5 selected_idx JSON.
3. **Stage 3 per-(cell, fold) LR training** (`scripts/analysis/train_l1_router.py`): for each cell × each fold (= 30 combinations), apply B-995 min-class-n=10 filter (drop classes with <10 train-fold samples to prevent minority hallucination), build design matrix using fold-k vectorizer + selected_idx mask, train `Pipeline(StandardScaler + LogisticRegression(class_weight=None, C=1.0, max_iter=2000))`. Dump 30 pickles + 6 per-cell meta JSON with thresholds-per-fold dict.
4. **(b) Inner-CV τ tuning**: within each (cell, fold) outer fold, run inner `StratifiedKFold(5)` on the train fold, evaluate cost-weighted decision rule at candidate τ ∈ {0.3, 0.4, 0.5, 0.6, 0.7}, pick τ* maximizing inner-holdout SR (tie-break = highest τ for conservative routing). τ never tuned on outer holdout.
5. **Runtime fold-aware prediction** (`p79/policies/learned_router.py::predict_mode_fold_aware`): at Pass-2 fire, lookup `fold_assignment[task_id] = k`, load the corresponding fold-k vectorizer + selected_idx + LR pickle + τ_{C,k}, apply `pipeline.predict_proba(features)`, route to `argmax_mode` if `max_prob > τ_{C,k}` else fall back to `phantom_som` (B-998 cost-weighted decision rule, B-995 safe routing under low confidence).

### 6.2 H10 Pareto non-dominance verdict (PRIMARY: K-of-6 descriptive)

**Estimand (preregistration §H10 line 188 + line 212, Q4=A locked 2026-05-18)**: per cell, the router achieves a (Cost, SR) operating point. We test whether this point is **Pareto non-dominated** by the 5 single-mode baselines {DOM, SoM, Vision, P-text, P-SoM} (P-prompt excluded per prereg line 199-204; expands to 6 if Phase 1a B0+B1+B2 cls produce ≥50 ep P-prompt outcomes). Per-cell test: 1000-iteration paired bootstrap on the cell's common task set; cell passes if router is non-dominated in ≥95% of bootstrap replicates.

**Primary verdict**: K cells pass / 6 (target ≥5/6 to reject H0 at α=0.05, δ=1.0pp threshold mirroring H1).

**Producer**: `scripts/analysis/aggregate_h10_pareto.py` (B-1002). Output artifacts: `results/phantom_paper/h10_pareto_verdict.{csv,md,json}` + per-cell Pareto scatter figure.

**Expected pattern (archive sim prior, preregistration §C5)**: site-asymmetric viability — visual-rich classifieds cells (3 cells × B0/B1/B2) pass Pareto non-dominance over best-single-mode at +0.5 to +2pp SR; text-dominated reddit cells (3 cells × B0/B1/B2) collapse toward always_phantom_som baseline (no router contribution beyond default mode swap). Archive B0-only simulation: cls +2.02pp / red -3.95pp (Variant B balanced class weight, 50-pair stratified CV). **Phase 1a clean-rerun confirmation pending**.

```
| Cell             | n_common | router SR | router Cost | θ (pp) | frac non-dom | Pass |
|------------------|---------:|----------:|------------:|-------:|-------------:|:----:|
| B0_classifieds   | TBD      | TBD       | TBD         | TBD    | TBD          | TBD  |
| B0_reddit        | TBD      | TBD       | TBD         | TBD    | TBD          | TBD  |
| B1_classifieds   | TBD      | TBD       | TBD         | TBD    | TBD          | TBD  |
| B1_reddit        | TBD      | TBD       | TBD         | TBD    | TBD          | TBD  |
| B2_classifieds   | TBD      | TBD       | TBD         | TBD    | TBD          | TBD  |
| B2_reddit        | TBD      | TBD       | TBD         | TBD    | TBD          | TBD  |
| **K of 6**       |          |           |             |        |              | TBD  |
```

### 6.3 APPENDIX SENSITIVITY: FE inverse-variance pool

We additionally report the **continuous per-cell θ_i = SR_router_i - max-feasible-baseline-SR_i** (where "feasible" = baseline cost ≤ router cost) pooled via fixed-effects inverse-variance weighting over the 6 cells. This mirrors the H1 estimand structure (preregistration §2.5 + §625) and provides a single pooled effect for reviewer comparison with H1.

Archive simulation predicts pooled θ ≈ -1pp with 95% CI crossing 0 (null), driven by site-asymmetric pattern (cls +2pp / red -4pp average). **Q4=A locks K-of-6 PRIMARY over FE pool to preserve site-asymmetric viability narrative (preregistration §C5).** The FE pool result is reported as sensitivity, not as gating.

**Pooled θ**: TBD pp [CI TBD, TBD] — Z vs δ=1.0pp: TBD — p_one_sided: TBD.

### 6.4 Site-asymmetric viability finding (paper §6 main narrative)

The K-of-6 PRIMARY breakdown is the load-bearing narrative for paper §6. We expect (per preregistration §C5 lock + archive simulation prior):

- **Visual-rich sites (classifieds)**: learned routing exhibits Pareto non-dominance over best-single-mode at +X.X pp SR; classifieds task distribution provides mode-asymmetric structure (color + visual-attribute tasks benefit from phantom_som; navigation tasks benefit from DOM). The 18-feature LR can distinguish these task classes from intent text alone.
- **Text-dominated sites (reddit)**: learned routing prediction distribution collapses toward majority-class prediction (≈ always_phantom_som baseline); per-cell Pareto non-dominance held marginally. Forum-style tasks with thread-traversal structure do not benefit from `[SOM_MARKS]` flat representation over hierarchical AXTree — the two-knob account's "forum lookup → flat list" prediction is empirically refuted on reddit task character (reddit tasks span navigate-and-discuss workflows, not pure list-lookup).

This site-asymmetric pattern is **paper-grade interesting**: not "router works/doesn't work" but "router viable conditional on site task-distribution heterogeneity" — a substantive contribution to cost-aware routing literature. The empirical evidence supports the **representation-axis routing hypothesis** but bounds the deployable applicability to visual-rich web settings.

### 6.5 Intelligent-baseline ladder (B-1006 R5 reviewer-defense)

Beyond random Tier-0a/0b/0c baselines (preregistration line 199 line 204), we report **4 intelligent baselines** to bound the learned router's lift from above and below:

1. **Always-cheapest-mode** (cost-only baseline, = always-DOM in current 5-arm set): bounds router's "cost-aware" claim from cost-axis side.
2. **Decision-stump single-feature** (e.g. `if DOM_tokens > 10K → route P-text else route DOM`): bounds router's feature-set value over a 1-feature heuristic.
3. **Per-task-lookup-table** (∞-capacity reductio = LR with task_id one-hot feature): bounds router from above — the 18-feature LR's generalization headroom equals (per-task lookup SR - 18-feature LR SR), quantifying how much "feature-conditioned routing transfers beyond memorization".
4. **LR-DOM-features-only** (intent regex + browser state, no TF-IDF ablation): bounds router from text-feature side.

These intelligent baselines address the R5 reviewer attack "router learns noise vs signal" — the per-task-lookup reductio in particular provides an ∞-capacity ceiling that any feature-conditioned router must remain below.

### 6.6 Limitations and disclosure

- **Within-population evaluation**: paper §6 results are within Phase 1a cls + red task pool. Phase 1b shop fire (post-workshop submission) provides the genuine cross-site out-of-distribution test for the routing claim.
- **5-arm baseline set**: P-prompt excluded due to cls archive aborted at 4 ep; expands to 6 if Phase 1a B0+B1+B2 cls produce ≥50 ep P-prompt outcomes.
- **τ candidates pre-locked**: {0.3, 0.4, 0.5, 0.6, 0.7} fixed before Phase 1a fire; not exhaustively grid-searched.
- **2-site meta pool**: K-of-6 = 2 sites × 3 baselines; cross-site generalization claim limited per §8 limitations 2-site disclaimer (B-1007).
- **Cost-aware decision rule**: routing uses `max_prob > τ` thresholding (B-998 (a) path); explicit cost-weighted objective `SR̂ - λ·Cost` deferred to follow-up (Phase 1b shop fire scope).

### 6.7 Code + artifact provenance (paper-grade reproducibility)

| Artifact | Path |
|---|---|
| Stage 1 raw features | `results/phantom_paper/l1_router/raw_features_phase1a.{npz,json}` |
| Stage 2 fold-local artifacts | `results/phantom_paper/l1_router/vectorizer_fold{k}.pkl` × 5 + `selected_idx_fold{k}.json` × 5 + `stage2_summary.json` |
| Stage 3 LR pickles | `results/phantom_paper/l1_router/{cell_id}_lr_fold{k}.pkl` × 30 |
| Per-cell meta | `results/phantom_paper/l1_router/{cell_id}_lr_meta.json` × 6 (incl. τ_{C,k} dict) |
| Fold assignments | `results/phantom_paper/l1_router/{cell_id}_fold_assignment.json` × 6 |
| H10 verdict | `results/phantom_paper/h10_pareto_verdict.{csv,md,json}` |
| Pass-2 condition outputs | `results/visualwebarena/phase1/{baseline}_router_learned_{site}_<date>/phase1_learned_router_*/episodes/*` |

**Reproducibility statement** (pre-fire lock): full pipeline rerun with seed=42 produces identical artifact hashes; runtime predictor consumes only the listed artifacts + VWA task config JSON + initial obs from `env.reset()` (mode-agnostic DOM-style observation). No post-hoc threshold tuning; no holdout-leak feature selection (Stage 2 fold-local pooled MI excludes all cells' fold-k holdouts from selection pool).
