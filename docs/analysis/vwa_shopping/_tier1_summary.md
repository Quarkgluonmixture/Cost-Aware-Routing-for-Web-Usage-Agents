---
type: analysis
status: tier-1 only
purpose: failure attribution coverage for VWA shopping, read against the other sites
producer: scripts/analysis/shopping_diag_tier1.py
---

# VWA shopping — Tier-1 failure attribution (ruleset v11)

Regenerate: `python scripts/analysis/shopping_diag_tier1.py`

Mechanical half only: rule hits from `diag_pattern_match.py` at the same ruleset as the classifieds / reddit / WA conditions, over the scored universe (432). No Tier-2 / Tier-3 reading yet except the benchmark-FP flags in §3.

## 1. How much of the failure mass the rule set explains, by site

Share of FAILED episodes on which no rule fires. The rules were discovered on classifieds and reddit; a higher share on a new site is the size of what Tier-1 cannot speak for there.

| site | conditions | median | range |
|---|---|---|---|
| classifieds | 18 | **6%** | 0–35% |
| reddit | 18 | **6%** | 0–17% |
| wa_reddit | 12 | **15%** | 9–26% |
| shopping | 9 | **23%** | 15–48% |

## 2. Per condition

| condition | SR | failed | no rule | only risk markers (P31/P36) | scaffold | top rules on failures |
|---|---|---|---|---|---|---|
| `B0_dom_shopping` | 11.34% (49/432) | 383 | 132 (34%) | 58 | 0 | P36 1697, P5 204, P31 173, P45 151, P44 124 |
| `B0_som_shopping` | 14.81% (64/432) | 368 | 176 (48%) | 68 | 0 | P36 1420, P5 143, P45 133, P31 127, P14 71 |
| `B0_vision_shopping` | 14.35% (62/432) | 370 | 158 (43%) | 89 | 0 | P31 167, P5 102, P14 66, P36 32, P12 14 |
| `B1_dom_shopping` | 4.86% (21/432) | 411 | 77 (19%) | 87 | 0 | P36 3520, P5 303, P31 281, P45 267, P14 103 |
| `B1_phantom_prompt_shopping` | 5.32% (23/432) | 409 | 83 (20%) | 76 | 0 | P36 2728, P5 303, P31 262, P45 231, P44 158 |
| `B1_phantom_som_shopping` | 4.63% (20/432) | 412 | 88 (21%) | 66 | 0 | P36 2712, P4 338, P5 310, P45 270, P31 266 |
| `B1_phantom_text_shopping` | 6.71% (29/432) | 403 | 62 (15%) | 101 | 0 | P36 3061, P31 270, P5 255, P45 217, P14 90 |
| `B1_som_shopping` | 7.64% (33/432) | 399 | 123 (31%) | 69 | 0 | P36 2067, P5 264, P31 230, P45 214, P4 190 |
| `B1_vision_shopping` | 5.56% (24/432) | 408 | 93 (23%) | 57 | 0 | P5 375, P31 272, P36 255, P14 167, P12 52 |

Scaffold-bug rules fire on **0** shopping failures across the nine conditions.

⚠️ Rule counts are symptoms, not causes: P31 (budget exhausted) and P36 (degenerate walk) fire on most long failures. The `only risk markers` column counts failures explained by nothing more specific than those two.

## 3. Benchmark-FP flags

| condition | task | outcome | rule | hand review |
|---|---|---|---|---|
| `B0_som_shopping` | 183 | success | P41 | eval HAS a positive check (`required_values: ['== 3']`, order quantity); P41 only recognises must_include / exact_match / fuzzy_match as positive |
| `B0_som_shopping` | 200 | success | P40 | agent opened the item page (…/spanish-cow-milk-cheese-mahon-1-pound.html) and read '1 pound' from the title; P40's item-page markers (page=item, product_id=, /product/) do not match Magento product URLs |
| `B0_vision_shopping` | 200 | success | P40 | same trajectory shape as B0_som_shopping task 200 |

**3 of 3** flags read by hand; all were rule misfires, so the benchmark-FP count on shopping is **0** at Tier-1. The misfires are queued for the v12 rule batch rather than patched here, because a rule change obliges a full rescan (discover-then-freeze).

## 4. What is still open

- The no-rule failures (§1) are unattributed. They need Tier-2 reading — a sample per condition is enough to say whether they hide a scaffold or evaluator class the rules miss. Task ids are in the JSON (`no_hit_task_ids`).
- B0 shopping has no phantom arms, so only three of its modes appear.
