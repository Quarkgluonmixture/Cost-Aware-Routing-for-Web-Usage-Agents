# Pooled × cost-tier router learnability (same-family corner)

- generated: `2026-08-03T16:40:05+00:00`
- schema: `2026-07-28-router-pooled-tier-learnability-v1`
- **post_hoc_exploratory=True / h10_eligible=False** — not the preregistered H10 gate.
- protocol: task-held-out 5-fold (§216.1), fully nested (§392.2), per-cell paired bootstrap B=1000 at 95% non-dominance (§150b.4 / B-1550)
- cost estimand: `total_billed_cost_usd`, comparable within a cell only

## H-pool

> A router trained on the same-family pool (B0+B1) with cost-tier labels Pareto-dominates always-cheapest on ≥1 site.

**Verdict: H-pool NOT supported — the same-family × cost-tier router dominates always-cheapest in 0/6 cells and is dominated by the fixed-mode menu in every cell. The most favourable corner (agreeing backbones, coarse label, highest ceiling) does not change the negative result.**

Three tallies, because the hypothesis and the locked test are not the same question. Non-dominance says the router is *admissible*; dominance says it is *better*. H-pool is worded as dominance (spec §2); the locked decision rule is non-dominance (spec §3).

| tally | same-family × cost-tier | all arms |
|---|---|---|
| non-dominated vs always-cheapest (locked §3 rule) | **1/6** | 7/36 |
| **dominates** always-cheapest (H-pool as worded, §2) | **0/6** | 0/36 |
| non-dominated vs all six fixed modes (on the front at all) | **0/6** | 0/36 |

### Attribution — what, if anything, caused a pass

- **classifieds label balance (same-family pool)**: {'text_only': 89, 'image': 63} — minority class is 41% of rows. Balanced enough that majority-class collapse is not the default explanation.
- **classifieds**: no cell clears the locked rule under any arm (7 arms tested).
- **reddit label balance (same-family pool)**: {'text_only': 63, 'image': 14} — minority class is 18% of rows. **Severely imbalanced**: a classifier can score well by collapsing to the majority class, so any statement about what the LABEL GRANULARITY contributed is confounded with the label barely varying.
- **reddit / B0_reddit** clears the locked rule in 5/7 arms: `same_family|cost_tier` 14.29%@0.10803, `same_family|which_mode` 15.27%@0.10415, `all_three|cost_tier` 13.30%@0.10071, `all_three|which_mode` 14.29%@0.10132, `per_cell|cost_tier|B0` 13.30%@0.10470.  Co-occurrence: clears with **and** without the cross-family cell; clears at **both** granularities; clears under **per-cell** training as well as pooled.
- **wa_reddit label balance (same-family pool)**: {'text_only': 76, 'image': 10} — minority class is 12% of rows. **Severely imbalanced**: a classifier can score well by collapsing to the majority class, so any statement about what the LABEL GRANULARITY contributed is confounded with the label barely varying.
- **wa_reddit / B1_wa_reddit** clears the locked rule in 2/6 arms: `same_family|which_mode` 16.35%@0.06108, `all_three|which_mode` 16.35%@0.06108.  Co-occurrence: clears with **and** without the cross-family cell.
- ⚠️ **These are co-occurrences, not causal attributions.** Two arms can both clear a binary admissibility bar while differing materially in (SR, cost) — the point estimates above show exactly that. Establishing that a factor did or did not move the operating point requires paired task-level contrasts (ΔSR and Δcost with paired bootstrap) or an explicit 2×2 interaction contrast, neither of which is computed here.

### How to read this

1. **Nothing reaches the front.** Across all 36 arm×cell combinations the router is non-dominated by the six-fixed-mode menu in 0 of them, and dominates always-cheapest in 0. The favourable corner the spec identified — same family, coarse label, highest ceiling — contributes 0/6 and 0/6 respectively.
2. **The coarse label did not buy a better operating point — and on the one passing cell it was slightly worse.** reddit·B0 same-family: which-mode 15.27% SR at 0.10415 vs cost-tier 14.29% at 0.10803, i.e. the 6-way label is better on both axes there. The §395.2 defect the tier label sidesteps is real, but sidestepping it did not help. ⚠️ Two qualifiers, both from cross-AI review 2026-07-29: (a) this is a point-estimate comparison, not a paired contrast, so it does not establish that granularity is causally inert; (b) on reddit the tier label is severely imbalanced (63 text_only vs 14 image), so a tier classifier can score by collapsing to the majority class — granularity and label-variance are confounded in exactly the cell that passes.
3. **The one pass is a property of the contrast, not of the router.** reddit·B0's always-cheapest is Vision at 7.39% SR against a best-single reference of 11.33% — an unusually weak baseline. Any routing policy that moves tasks off Vision buys SR there, which is why all five passing arms pass, including per-cell training with no pooling at all.
4. **The trade-off is genuine but priced.** reddit·B0 reaches 13.3-15.3% SR, above the best single mode, and pays 2.7-10.2% more per task for it. That is a legitimate operating point to report; it is not the dominance H-pool asked for, and the six-mode menu still dominates it in 35-71% of paired replicates.
5. **Direction for Paper B.** The negative result survives its most favourable test and can now be stated with the qualifier attached rather than as an unexamined generalisation: routing does not beat a fixed cheap policy *even when the pool agrees, the label is coarse and immune to the tie-break defect, and the plug-in ceiling is 97.5%*.

## classifieds

universe N=224 (sha `b0f3b8b0b002`), fold sizes {0: 45, 1: 45, 2: 45, 3: 45, 4: 44}

### Label supply

| pool | labelled rows | which-mode classes | cost-tier split |
|---|---|---|---|
| `same_family` (B0+B1) | 152 | dom=53, phantom_prompt=15, phantom_som=16, phantom_text=5, som=45, vision=18 | text_only=89, image=63 |
| `all_three` (B0+B1+B2) | 168 | dom=56, phantom_prompt=15, phantom_som=18, phantom_text=6, som=50, vision=23 | text_only=95, image=73 |

### Are the pooled features backbone-identifiable?

B0_classifieds vs B1_classifieds: 163/224 shared tasks have bit-identical feature rows (72.8%); out-of-fold AUROC for predicting the backbone from X = **0.500**.

> AUROC near 0.5 => the pooled features carry no usable backbone identity, so the conflict rate really is same-X-different-y.

### Is always-cheapest a cost floor? (No.)

A cross-AI reviewer argued H-pool is arithmetically unsatisfiable because the cheapest mode is cheapest by construction. That holds only if it is cheapest on every task. It is not — it has the lowest **mean**.

| cell | cheapest-on-mean | tasks where it is NOT the per-task floor | always-cheapest cost | per-task oracle | headroom |
|---|---|---|---|---|---|
| B0_classifieds | vision | 151/224 = 67.4% | 0.06481 | 0.03488 | **-46.2%** |
| B1_classifieds | vision | 109/224 = 48.7% | 0.04316 | 0.02545 | **-41.0%** |
| B2_classifieds | vision | 152/224 = 67.9% | 0.07065 | 0.04747 | **-32.8%** |

> So Pareto dominance is **not** excluded by definition; the failure below is empirical. The headroom column is also the cost-routing upper bound in its own right.

### Operating points

`best-SR ref` is the best single mode chosen per training fold — it shows how weak or strong the always-cheapest contrast is in that cell, which is what decides whether beating it means anything.

| arm | cell | router SR% | router cost | cheapest SR% | cheapest cost | best-SR ref% | ΔSR pp | Δcost % | ND vs cheapest | **dominates** | ND vs 6 fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `same_family|cost_tier` | B0_classifieds | 25.00 | 0.07382 | 25.00 | 0.06481 | 27.23 | +0.00 | +13.9 | 0.468 fail | 0.004 fail | 0.172 fail |
| `same_family|cost_tier` | B1_classifieds | 12.05 | 0.05739 | 12.50 | 0.04316 | 14.29 | -0.45 | +33.0 | 0.336 fail | 0.000 fail | 0.310 fail |
| `same_family|which_mode` | B0_classifieds | 22.77 | 0.06877 | 25.00 | 0.06481 | 27.23 | -2.23 | +6.1 | 0.254 fail | 0.046 fail | 0.223 fail |
| `same_family|which_mode` | B1_classifieds | 12.05 | 0.05607 | 12.50 | 0.04316 | 14.29 | -0.45 | +29.9 | 0.366 fail | 0.000 fail | 0.352 fail |
| `all_three|cost_tier` | B0_classifieds | 24.11 | 0.07427 | 25.00 | 0.06481 | 27.23 | -0.89 | +14.6 | 0.345 fail | 0.002 fail | 0.124 fail |
| `all_three|cost_tier` | B1_classifieds | 11.16 | 0.05896 | 12.50 | 0.04316 | 14.29 | -1.34 | +36.6 | 0.212 fail | 0.000 fail | 0.171 fail |
| `all_three|cost_tier` | B2_classifieds | 1.79 | 0.07402 | 2.23 | 0.07065 | 1.34 | -0.45 | +4.8 | 0.180 fail | 0.000 fail | 0.167 fail |
| `all_three|which_mode` | B0_classifieds | 25.45 | 0.06990 | 25.00 | 0.06481 | 27.23 | +0.45 | +7.9 | 0.555 fail | 0.046 fail | 0.487 fail |
| `all_three|which_mode` | B1_classifieds | 12.05 | 0.05732 | 12.50 | 0.04316 | 14.29 | -0.45 | +32.8 | 0.364 fail | 0.000 fail | 0.326 fail |
| `all_three|which_mode` | B2_classifieds | 0.89 | 0.08455 | 2.23 | 0.07065 | 1.34 | -1.34 | +19.7 | 0.000 fail | 0.000 fail | 0.000 fail |
| `per_cell|cost_tier|B0` | B0_classifieds | 24.11 | 0.07657 | 25.00 | 0.06481 | 27.23 | -0.89 | +18.2 | 0.335 fail | 0.000 fail | 0.023 fail |
| `per_cell|cost_tier|B1` | B1_classifieds | 10.71 | 0.05899 | 12.50 | 0.04316 | 14.29 | -1.79 | +36.7 | 0.176 fail | 0.000 fail | 0.148 fail |
| `per_cell|cost_tier|B2` | B2_classifieds | 1.79 | 0.07403 | 2.23 | 0.07065 | 1.34 | -0.45 | +4.8 | 0.180 fail | 0.000 fail | 0.170 fail |

### What the router actually selected

- `same_family|cost_tier` / B0_classifieds: routed {'phantom_prompt': 91, 'som': 133}; always-cheapest picked {'vision': 224}
- `same_family|cost_tier` / B1_classifieds: routed {'dom': 19, 'phantom_text': 76, 'som': 129}; always-cheapest picked {'vision': 224}
- `same_family|which_mode` / B0_classifieds: routed {'dom': 85, 'phantom_prompt': 34, 'phantom_som': 7, 'phantom_text': 3, 'som': 77, 'vision': 18}; always-cheapest picked {'vision': 224}
- `same_family|which_mode` / B1_classifieds: routed {'dom': 85, 'phantom_prompt': 34, 'phantom_som': 7, 'phantom_text': 3, 'som': 77, 'vision': 18}; always-cheapest picked {'vision': 224}
- `all_three|cost_tier` / B0_classifieds: routed {'phantom_prompt': 123, 'som': 101}; always-cheapest picked {'vision': 224}
- `all_three|cost_tier` / B1_classifieds: routed {'dom': 25, 'phantom_text': 82, 'som': 117}; always-cheapest picked {'vision': 224}
- `all_three|cost_tier` / B2_classifieds: routed {'dom': 8, 'phantom_prompt': 43, 'som': 3, 'vision': 170}; always-cheapest picked {'vision': 224}
- `all_three|which_mode` / B0_classifieds: routed {'dom': 90, 'phantom_prompt': 31, 'phantom_som': 6, 'phantom_text': 3, 'som': 71, 'vision': 23}; always-cheapest picked {'vision': 224}
- `all_three|which_mode` / B1_classifieds: routed {'dom': 90, 'phantom_prompt': 31, 'phantom_som': 6, 'phantom_text': 3, 'som': 71, 'vision': 23}; always-cheapest picked {'vision': 224}
- `all_three|which_mode` / B2_classifieds: routed {'dom': 90, 'phantom_prompt': 31, 'phantom_som': 6, 'phantom_text': 3, 'som': 71, 'vision': 23}; always-cheapest picked {'vision': 224}
- `per_cell|cost_tier|B0` / B0_classifieds: routed {'phantom_prompt': 90, 'som': 134}; always-cheapest picked {'vision': 224}
- `per_cell|cost_tier|B1` / B1_classifieds: routed {'dom': 35, 'phantom_text': 56, 'som': 133}; always-cheapest picked {'vision': 224}
- `per_cell|cost_tier|B2` / B2_classifieds: routed {'phantom_prompt': 33, 'som': 13, 'vision': 178}; always-cheapest picked {'vision': 224}

## reddit

universe N=203 (sha `1ce29c8b9fbe`), fold sizes {0: 41, 1: 41, 2: 41, 3: 40, 4: 40}

### Label supply

| pool | labelled rows | which-mode classes | cost-tier split |
|---|---|---|---|
| `same_family` (B0+B1) | 77 | dom=41, phantom_prompt=5, phantom_som=10, phantom_text=7, som=8, vision=6 | text_only=63, image=14 |
| `all_three` (B0+B1+B2) | 92 | dom=49, phantom_prompt=5, phantom_som=11, phantom_text=9, som=9, vision=9 | text_only=74, image=18 |

### Are the pooled features backbone-identifiable?

B0_reddit vs B1_reddit: 57/203 shared tasks have bit-identical feature rows (28.1%); out-of-fold AUROC for predicting the backbone from X = **0.526**.

> AUROC near 0.5 => the pooled features carry no usable backbone identity, so the conflict rate really is same-X-different-y.

### Is always-cheapest a cost floor? (No.)

A cross-AI reviewer argued H-pool is arithmetically unsatisfiable because the cheapest mode is cheapest by construction. That holds only if it is cheapest on every task. It is not — it has the lowest **mean**.

| cell | cheapest-on-mean | tasks where it is NOT the per-task floor | always-cheapest cost | per-task oracle | headroom |
|---|---|---|---|---|---|
| B0_reddit | vision | 144/203 = 70.9% | 0.09807 | 0.05682 | **-42.1%** |
| B1_reddit | vision | 102/203 = 50.2% | 0.05240 | 0.03733 | **-28.8%** |
| B2_reddit | vision | 96/203 = 47.3% | 0.06833 | 0.05314 | **-22.2%** |

> So Pareto dominance is **not** excluded by definition; the failure below is empirical. The headroom column is also the cost-routing upper bound in its own right.

### Operating points

`best-SR ref` is the best single mode chosen per training fold — it shows how weak or strong the always-cheapest contrast is in that cell, which is what decides whether beating it means anything.

| arm | cell | router SR% | router cost | cheapest SR% | cheapest cost | best-SR ref% | ΔSR pp | Δcost % | ND vs cheapest | **dominates** | ND vs 6 fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `same_family|cost_tier` | B0_reddit | 14.29 | 0.10803 | 7.39 | 0.09807 | 11.33 | +6.90 | +10.2 | 0.992 PASS | 0.010 fail | 0.293 fail |
| `same_family|cost_tier` | B1_reddit | 4.43 | 0.07243 | 2.46 | 0.05240 | 7.39 | +1.97 | +38.2 | 0.868 fail | 0.000 fail | 0.072 fail |
| `same_family|which_mode` | B0_reddit | 15.27 | 0.10415 | 7.39 | 0.09807 | 11.33 | +7.88 | +6.2 | 1.000 PASS | 0.062 fail | 0.652 fail |
| `same_family|which_mode` | B1_reddit | 4.43 | 0.07269 | 2.46 | 0.05240 | 7.39 | +1.97 | +38.7 | 0.856 fail | 0.000 fail | 0.140 fail |
| `all_three|cost_tier` | B0_reddit | 13.30 | 0.10071 | 7.39 | 0.09807 | 11.33 | +5.91 | +2.7 | 0.982 PASS | 0.250 fail | 0.567 fail |
| `all_three|cost_tier` | B1_reddit | 4.43 | 0.07119 | 2.46 | 0.05240 | 7.39 | +1.97 | +35.8 | 0.868 fail | 0.000 fail | 0.210 fail |
| `all_three|cost_tier` | B2_reddit | 1.97 | 0.07339 | 1.97 | 0.06833 | 3.94 | +0.00 | +7.4 | 0.425 fail | 0.000 fail | 0.425 fail |
| `all_three|which_mode` | B0_reddit | 14.29 | 0.10132 | 7.39 | 0.09807 | 11.33 | +6.90 | +3.3 | 0.998 PASS | 0.208 fail | 0.609 fail |
| `all_three|which_mode` | B1_reddit | 4.43 | 0.06930 | 2.46 | 0.05240 | 7.39 | +1.97 | +32.2 | 0.864 fail | 0.000 fail | 0.486 fail |
| `all_three|which_mode` | B2_reddit | 3.45 | 0.09369 | 1.97 | 0.06833 | 3.94 | +1.48 | +37.1 | 0.762 fail | 0.000 fail | 0.491 fail |
| `per_cell|cost_tier|B0` | B0_reddit | 13.30 | 0.10470 | 7.39 | 0.09807 | 11.33 | +5.91 | +6.8 | 0.981 PASS | 0.063 fail | 0.158 fail |
| `per_cell|cost_tier|B1` | B1_reddit | 4.43 | 0.07283 | 2.46 | 0.05240 | 7.39 | +1.97 | +39.0 | 0.868 fail | 0.000 fail | 0.051 fail |
| `per_cell|cost_tier|B2` | B2_reddit | 1.97 | 0.07315 | 1.97 | 0.06833 | 3.94 | +0.00 | +7.1 | 0.425 fail | 0.000 fail | 0.425 fail |

### What the router actually selected

- `same_family|cost_tier` / B0_reddit: routed {'dom': 124, 'phantom_text': 37, 'som': 42}; always-cheapest picked {'vision': 203}
- `same_family|cost_tier` / B1_reddit: routed {'dom': 70, 'phantom_som': 26, 'phantom_text': 70, 'som': 37}; always-cheapest picked {'vision': 203}
- `same_family|which_mode` / B0_reddit: routed {'dom': 126, 'phantom_prompt': 16, 'phantom_som': 8, 'phantom_text': 3, 'som': 38, 'vision': 12}; always-cheapest picked {'vision': 203}
- `same_family|which_mode` / B1_reddit: routed {'dom': 125, 'phantom_prompt': 16, 'phantom_som': 8, 'phantom_text': 4, 'som': 38, 'vision': 12}; always-cheapest picked {'vision': 203}
- `all_three|cost_tier` / B0_reddit: routed {'dom': 152, 'phantom_text': 33, 'som': 18}; always-cheapest picked {'vision': 203}
- `all_three|cost_tier` / B1_reddit: routed {'dom': 64, 'phantom_som': 15, 'phantom_text': 75, 'som': 49}; always-cheapest picked {'vision': 203}
- `all_three|cost_tier` / B2_reddit: routed {'dom': 43, 'vision': 160}; always-cheapest picked {'vision': 203}
- `all_three|which_mode` / B0_reddit: routed {'dom': 102, 'phantom_prompt': 20, 'phantom_som': 7, 'phantom_text': 8, 'som': 36, 'vision': 30}; always-cheapest picked {'vision': 203}
- `all_three|which_mode` / B1_reddit: routed {'dom': 101, 'phantom_prompt': 20, 'phantom_som': 7, 'phantom_text': 9, 'som': 36, 'vision': 30}; always-cheapest picked {'vision': 203}
- `all_three|which_mode` / B2_reddit: routed {'dom': 100, 'phantom_prompt': 20, 'phantom_som': 5, 'phantom_text': 8, 'som': 36, 'vision': 34}; always-cheapest picked {'vision': 203}
- `per_cell|cost_tier|B0` / B0_reddit: routed {'dom': 151, 'phantom_text': 33, 'som': 19}; always-cheapest picked {'vision': 203}
- `per_cell|cost_tier|B1` / B1_reddit: routed {'dom': 78, 'phantom_som': 29, 'phantom_text': 63, 'som': 33}; always-cheapest picked {'vision': 203}
- `per_cell|cost_tier|B2` / B2_reddit: routed {'dom': 40, 'som': 1, 'vision': 162}; always-cheapest picked {'vision': 203}

## wa_reddit

universe N=104 (sha `wa-six-mode-`), fold sizes {0: 21, 1: 21, 2: 21, 3: 21, 4: 20}

### Label supply

| pool | labelled rows | which-mode classes | cost-tier split |
|---|---|---|---|
| `same_family` (B0+B1) | 86 | dom=45, phantom_prompt=8, phantom_som=9, phantom_text=14, som=5, vision=5 | text_only=76, image=10 |
| `all_three` (B0+B1) | 86 | dom=45, phantom_prompt=8, phantom_som=9, phantom_text=14, som=5, vision=5 | text_only=76, image=10 |

### Are the pooled features backbone-identifiable?

B0_wa_reddit vs B1_wa_reddit: 19/104 shared tasks have bit-identical feature rows (18.3%); out-of-fold AUROC for predicting the backbone from X = **0.897**.

> AUROC near 0.5 => the pooled features carry no usable backbone identity, so the conflict rate really is same-X-different-y.

### Is always-cheapest a cost floor? (No.)

A cross-AI reviewer argued H-pool is arithmetically unsatisfiable because the cheapest mode is cheapest by construction. That holds only if it is cheapest on every task. It is not — it has the lowest **mean**.

| cell | cheapest-on-mean | tasks where it is NOT the per-task floor | always-cheapest cost | per-task oracle | headroom |
|---|---|---|---|---|---|
| B0_wa_reddit | dom | 91/104 = 87.5% | 0.07531 | 0.04286 | **-43.1%** |
| B1_wa_reddit | vision | 50/104 = 48.1% | 0.04468 | 0.03228 | **-27.8%** |

> So Pareto dominance is **not** excluded by definition; the failure below is empirical. The headroom column is also the cost-routing upper bound in its own right.

### Operating points

`best-SR ref` is the best single mode chosen per training fold — it shows how weak or strong the always-cheapest contrast is in that cell, which is what decides whether beating it means anything.

| arm | cell | router SR% | router cost | cheapest SR% | cheapest cost | best-SR ref% | ΔSR pp | Δcost % | ND vs cheapest | **dominates** | ND vs 6 fixed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `same_family|cost_tier` | B0_wa_reddit | 30.77 | 0.08355 | 26.92 | 0.07860 | 35.58 | +3.85 | +6.3 | 0.833 fail | 0.159 fail | 0.483 fail |
| `same_family|cost_tier` | B1_wa_reddit | 15.38 | 0.06959 | 9.62 | 0.04468 | 13.46 | +5.77 | +55.7 | 0.941 fail | 0.000 fail | 0.133 fail |
| `same_family|which_mode` | B0_wa_reddit | 25.96 | 0.07973 | 26.92 | 0.07860 | 35.58 | -0.96 | +1.4 | 0.523 fail | 0.213 fail | 0.176 fail |
| `same_family|which_mode` | B1_wa_reddit | 16.35 | 0.06108 | 9.62 | 0.04468 | 13.46 | +6.73 | +36.7 | 0.950 PASS | 0.000 fail | 0.709 fail |
| `all_three|cost_tier` | B0_wa_reddit | 30.77 | 0.08355 | 26.92 | 0.07860 | 35.58 | +3.85 | +6.3 | 0.833 fail | 0.159 fail | 0.483 fail |
| `all_three|cost_tier` | B1_wa_reddit | 15.38 | 0.06959 | 9.62 | 0.04468 | 13.46 | +5.77 | +55.7 | 0.941 fail | 0.000 fail | 0.133 fail |
| `all_three|which_mode` | B0_wa_reddit | 25.96 | 0.07973 | 26.92 | 0.07860 | 35.58 | -0.96 | +1.4 | 0.523 fail | 0.213 fail | 0.176 fail |
| `all_three|which_mode` | B1_wa_reddit | 16.35 | 0.06108 | 9.62 | 0.04468 | 13.46 | +6.73 | +36.7 | 0.950 PASS | 0.000 fail | 0.709 fail |
| `per_cell|cost_tier|B0` | B0_wa_reddit | 30.77 | 0.08759 | 26.92 | 0.07860 | 35.58 | +3.85 | +11.4 | 0.815 fail | 0.034 fail | 0.090 fail |
| `per_cell|cost_tier|B1` | B1_wa_reddit | 12.50 | 0.06613 | 9.62 | 0.04468 | 13.46 | +2.88 | +48.0 | 0.769 fail | 0.000 fail | 0.042 fail |

### What the router actually selected

- `same_family|cost_tier` / B0_wa_reddit: routed {'phantom_text': 79, 'som': 17, 'vision': 8}; always-cheapest picked {'dom': 84, 'phantom_prompt': 20}
- `same_family|cost_tier` / B1_wa_reddit: routed {'dom': 36, 'phantom_prompt': 1, 'phantom_text': 35, 'som': 32}; always-cheapest picked {'vision': 104}
- `same_family|which_mode` / B0_wa_reddit: routed {'dom': 68, 'phantom_prompt': 4, 'phantom_som': 5, 'phantom_text': 22, 'som': 1, 'vision': 4}; always-cheapest picked {'dom': 84, 'phantom_prompt': 20}
- `same_family|which_mode` / B1_wa_reddit: routed {'dom': 66, 'phantom_prompt': 7, 'phantom_som': 6, 'phantom_text': 20, 'som': 1, 'vision': 4}; always-cheapest picked {'vision': 104}
- `all_three|cost_tier` / B0_wa_reddit: routed {'phantom_text': 79, 'som': 17, 'vision': 8}; always-cheapest picked {'dom': 84, 'phantom_prompt': 20}
- `all_three|cost_tier` / B1_wa_reddit: routed {'dom': 36, 'phantom_prompt': 1, 'phantom_text': 35, 'som': 32}; always-cheapest picked {'vision': 104}
- `all_three|which_mode` / B0_wa_reddit: routed {'dom': 68, 'phantom_prompt': 4, 'phantom_som': 5, 'phantom_text': 22, 'som': 1, 'vision': 4}; always-cheapest picked {'dom': 84, 'phantom_prompt': 20}
- `all_three|which_mode` / B1_wa_reddit: routed {'dom': 66, 'phantom_prompt': 7, 'phantom_som': 6, 'phantom_text': 20, 'som': 1, 'vision': 4}; always-cheapest picked {'vision': 104}
- `per_cell|cost_tier|B0` / B0_wa_reddit: routed {'phantom_text': 79, 'som': 20, 'vision': 5}; always-cheapest picked {'dom': 84, 'phantom_prompt': 20}
- `per_cell|cost_tier|B1` / B1_wa_reddit: routed {'dom': 38, 'phantom_prompt': 20, 'phantom_text': 38, 'som': 8}; always-cheapest picked {'vision': 104}

## Known limitations (not optional)

1. WebArena (added 2026-08-03) carries neither `reasoning_difficulty` nor a reference image, so those two of the twenty features are zero-filled on its cells and cannot contribute there. That is tolerable in THIS product and not in router_triage_learnability, because nothing here compares a score across sites: every arm is judged inside one site against that site's own always-cheapest baseline. WebArena also has no cross-family backbone — B2 never ran it — so its `all_three` row is B0+B1 and is therefore IDENTICAL to `same_family` by construction, not by result. Read the two WA rows as one arm reported twice; the cross-family contrast this product is partly about cannot be formed there at all.
2. n is small: the same-family pool shares only 50 tasks on classifieds and 20 on reddit (labelled rows: cls 152, red 77). Every per-cell contrast below is underpowered and the fold-to-fold variation is correspondingly large.
3. A ceiling is not learnability (§394): red·B2 is the only cell of six whose triage signal survived Holm, and its AUROC was 0.483. A 97.5% tier ceiling bounds what a perfect classifier could do; it says nothing about whether this one does.
4. The 48% / 45% which-mode conflict may be a real backbone difference or may be noise; the present data cannot separate the two. Phase 0b measured a 4.9-7.6pp same-condition replicate floor, which is the scale the disagreement must clear before it can be read as a model property.
5. B0 and B1 costs are NOT comparable (B0 bills a proxy API; B1/B2 are electricity-derived). Pooling is used for labels and features only — every (SR, cost) pair and every Pareto verdict is computed inside a single cell.
6. post_hoc_exploratory=True / h10_eligible=False. This is not the preregistered H10 gate and must never be cited as one; it is an exploratory probe of a corner the negative result never covered, in the manner of router_objective_ordering.md.
7. Non-dominance is not dominance, and the two tallies must not be merged. The locked §3 rule (95% paired-bootstrap non-dominance vs always-cheapest) is an admissibility criterion: a policy buying +7pp SR for +10% cost passes it. H-pool as worded in §2 asks for dominance. Any reader quoting a pass rate must say which of the two it is.
8. A low tier-conflict rate can mean agreement or it can mean the label barely varies. The same-family reddit pool is 63 text_only vs 14 image, and B1·reddit alone is 20 vs 4 — so reddit's 5.0% conflict / 97.5% ceiling partly reflects a near-constant label rather than two backbones agreeing on a hard call. The class balance is reported per pool above for exactly this reason.
9. The which-mode arms carry no min-class filter. train_l1_router's Stage-3 N_MIN_CLASS_TRAIN=10 rule would leave several folds with fewer than two trainable classes and the control arm could not run at all; per-fold class counts are reported instead so the reader can see which classes were never learnable.
