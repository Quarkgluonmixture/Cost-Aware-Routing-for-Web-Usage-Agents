# B0 DOM Shopping Diagnostic

## 1. Headline Finding

The counter-intuitive result is real but not paradoxical: the `A_NON_VISUAL_TEXT_ONLY` bucket concentrates Magento comparison, category-disambiguation, and latent visual-attribute tasks that are especially brittle in DOM-only AXTree, while many `B_VISUAL_REQUIRED_REFERENCE_IMAGE` tasks become easier once the task-provided reference image is converted into a short textual search anchor.

## 2. SR Breakdown

Source set: `results/visualwebarena/phase1/B0_dom_shopping_20260428/analysis/reason_diagnostics/episode_reason_rows.csv` joined with `docs/analysis/cross_sites/codex_audit_shopping.json` and `external/visualwebarena/config_files/vwa/test_shopping.json`. The reason-diagnostics table covers 465 tasks; task 345 appears in the audit/config set but has no episode row in this diagnostic export.

| Visual taxonomy | n | raw success | adjusted success | FP | avg steps | fail avg steps | page-change rate | fail page-change | avg search attempts | avg tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A_NON_VISUAL_TEXT_ONLY | 82 | 8.54% | 8.54% | 0 | 9.73 | 9.91 | 0.717 | 0.711 | 1.96 | 34,510 |
| B_VISUAL_REQUIRED_REFERENCE_IMAGE | 169 | 24.26% | 21.30% | 5 | 9.91 | 10.48 | 0.745 | 0.731 | 2.94 | 42,056 |
| C_VISUAL_REQUIRED_PAGE_SCREENSHOT | 205 | 13.66% | 9.76% | 8 | 8.98 | 9.36 | 0.720 | 0.716 | 1.59 | 31,317 |
| D_UNCERTAIN | 9 | 11.11% | 0.00% | 1 | 8.67 | 8.67 | 0.658 | 0.658 | 1.67 | 30,634 |
| TOTAL | 465 | 16.56% | 13.55% | 14 | 9.44 | 9.82 | 0.726 | 0.719 | 2.15 | 36,415 |

The primary A failure buckets are `fail_no_progress` (42/82), `fail_finish_eval_mismatch` (17/82), `fail_incomplete_or_stuck` (6/82), and max-step loop variants (3 click-back loops, plus smaller max-step buckets). Successful A tasks were shorter and more diverse: A successes averaged 7.86 steps, 26.7k tokens, action diversity 0.682, and max repeat streak 1.57; A failures averaged 9.91 steps, 35.2k tokens, action diversity 0.415, and max repeat streak 2.81.

## 3. H1/H2/H3 Evidence

### H1. Aggregation and Sorting Cost

Verdict: strongly supported. A has the highest concentration of comparison and price-ordering intents, and those subgroups collapse.

Quantitative evidence:

| A subset | n | adjusted SR | avg steps | fail avg steps | avg page-unchanged | avg searches | avg tokens | dominant failure |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| all A | 82 | 8.54% | 9.73 | 9.91 | 0.219 | 1.96 | 34.5k | no progress |
| `intent_has_compare` | 23 | 0.00% | 11.13 | 11.13 | 0.264 | 3.39 | 42.2k | no progress |
| `least/most expensive/cheapest` keyword | 27 | 0.00% | 10.67 | 10.67 | 0.272 | 3.11 | 40.3k | no progress |
| quoted category / survey / first-item forms | 40 | 5.00% | 9.53 | 9.82 | 0.231 | 2.23 | 34.6k | no progress |

Case studies:

| task | intent excerpt | trajectory evidence | final thought / action | verdict |
|---:|---|---|---|---|
| 0 | "least expensive red blanket" from "Blankets & Throws" | Correctly navigates Home & Kitchen -> Bedding -> Blankets & Throws, sorts by price, then chooses the first item and tries to infer color from unlabeled `Style` options. | Final thought says Style 4 "may be red"; final action is another click on the product page, not completion. | Sorting succeeded, but DOM-only could not verify the visual/color attribute after list selection. |
| 4 | "most expensive black and white item" from "Posters & Prints" | Searches `Posters & Prints`, toggles price sort, then results drift to printers/copiers because category text and search query collide. | "search results are showing printers and copiers, not black and white posters or prints"; repeated type action. | Category-search ambiguity plus price ordering creates irrelevant result loops. |
| 425 | "cheapest round smartwatch" | Sorts `Smartwatches` by price, clicks/backs, scrolls, then picks a visible Skagen watch as cheapest round. | Finishes claiming the item was added, but adjusted evaluator marks mismatch. | The model uses "visible cheapest" instead of globally validating all sorted candidates and shape. |

The A successes reinforce this: the few successful A cases are mostly short lookup or direct-search cases, e.g. task 12 (`van gogh` first painting, 4 steps), task 201 (PS4 skull sticker weight, 6 steps), task 303 (calories per container, 4 steps). These do not require exhaustive comparison across a Magento list.

### H2. Complex Multi-Step Navigation

Verdict: partially supported. A failures are not uniquely low-navigation; their page-change rate is similar to B/C. The issue is more specific: DOM-only often changes pages successfully but changes to the wrong semantic state, then repeats search/click/back operations.

Quantitative evidence:

- A failures average 9.91 steps and page-change rate 0.711; B failures average 10.48 and 0.731; C failures average 9.36 and 0.716.
- A has elevated page-unchanged/repetition relative to A successes: failures page-unchanged 0.237 vs successes 0.032, URL revisits 7.17 vs 5.43, max repeat streak 2.81 vs 1.57.
- Behavioral signals agree with this failure mode: in `auroc_all_metrics.csv`, `action_diversity` AUROC is 0.718, `max_repeat_streak` is 0.699, and `url_revisit_max` is 0.697 for success/failure separation.

Case studies:

| task | intent excerpt | trajectory evidence | final thought / action | verdict |
|---:|---|---|---|---|
| 1 | "least expensive blue headset" from "Virtual Reality (VR) Headsets" | Clicks Electronics, searches exact category, sorts price, scrolls through irrelevant products, then repeats `blue VR headset` into the same field with no page change. | "current search results are not showing VR headsets but unrelated products"; repeated type action. | Navigation reaches a search page, not the category; DOM-only cannot recover category scope. |
| 18 | price of most expensive red product in "Basic Cases" | Six consecutive clicks while staying on `cell-phones-accessories.html`; thoughts repeatedly say Basic Cases is visible in a dropdown. | Final action is still click; no answer. | Menu/dropdown navigation is visible in AXTree but brittle as an action target. |
| 45 | cheapest item in "Children's Dental Care" with cupcake style | Runs to 30 steps, 11 searches, 116k tokens, click-back loop; results drift to unrelated beds/projectors. | "search results are not showing any relevant items"; repeated type/search recovery. | Multi-step search recovery becomes expensive looping rather than useful exploration. |

### H3. Classification Boundary / Pseudo-Non-Visual A Tasks

Verdict: strongly supported as a contributing factor, but not the whole story.

Quantitative evidence:

- 45/75 A failures (60.0%) contain explicit visual-ish attributes by intent review: color, theme, design, graphic, style, pattern, shape, or image-derived product semantics.
- The A color-flag subset is 32 tasks with 6.25% adjusted SR; A no-color is only slightly better at 10.0%, so visual attributes hurt but do not fully explain the 8.54% bucket-level SR.
- The hardest "pure text" A comparison subset also fails: A `intent_has_compare` is 23 tasks at 0.0% adjusted SR.
- `D_UNCERTAIN` is 9 tasks and 0.0% adjusted SR after FP adjustment, which confirms the audit boundary is meaningful.

Case studies:

| task | audit bucket | intent excerpt | why the boundary matters | verdict |
|---:|---|---|---|---|
| 25 | A | first raincoat that is blue | No reference image, but success depends on visual color verification; DOM titles/options do not reliably expose color. The agent loops click -> back -> scroll for 30 steps. | Pseudo-non-visual; actually visual-verification heavy. |
| 392 | A | purple and blue controllers with lightning and galaxy designs | The task encodes color/design in text, but verification depends on visual product art and exact SKU. Agent finishes on a purple/blue PS3 controller page while the category asks Legacy Systems > Playstation Systems. | Text-only wording still demands visual design matching. |
| 383 | A | item on the page with Rick and Morty themes | No reference image, but the target is a theme/design. Agent claims the cart action is complete after landing on a themed page; evaluator rejects. | Theme matching is weakly represented in DOM text and highly SKU-sensitive. |

Overall H3 interpretation: `A_NON_VISUAL_TEXT_ONLY` should be read as "no task-provided image and no current-page screenshot requirement," not as "semantically easy for DOM." Many A tasks still require color/design/theme inference, and the remaining A tasks include heavy list aggregation.

## 4. B_VISUAL_REF_IMAGE Mechanism

Section 3.5 notes that task-provided reference images are separate from observation mode and are passed to all modes; DOM removes only the current browser screenshot. The B traces show the model using those images before action step 0, converting them into concise search or product anchors. This often bypasses the Magento category/list traversal that hurts A.

Representative B successes:

| task | intent excerpt | first-step image-to-text anchor | outcome mechanism |
|---:|---|---|---|
| 93 | "something like the one in the bottom right" | "a pair of orange scissors"; searches `orange scissors` | Direct query lands on matching product; 4 steps. |
| 102 | "one of what its wearing for the winter" | "pink knitted hat"; searches `pink knitted hat` | Image caption becomes exact enough for search and wishlist action; 4 steps. |
| 111 | exact product in "Herbs, Spices & Seasonings" | "Chipotle Cinnamon Rub"; searches exact product name | Reference image supplies OCR/label-like anchor; 4 steps. |
| 121 | exact item cost | "SanDisk Extreme PRO 512GB CFast 2.0" | Exact product name from image yields price answer; 4 steps. |
| 155 | same brand as picture, silver headphones | "image shows a Sony camera"; searches `Sony headphones` | Reference image supplies brand; DOM supplies review/rating/price text. |
| 168 | drink flavor -> mochi | "boba tea / bubble tea"; searches `boba tea mochi` | Image supplies flavor concept; eventually finds Bubble Milk Tea mochi. |
| 296 | powder for beverage in picture | "matcha tea being prepared"; searches `matcha powder` | Image-to-text query solves the task in 3 steps. |

This is the main explanation for B > A: B has an extra semantic compression step in the task input. The reference image does not give DOM current-page vision, but it gives the model a target label, brand, flavor, color, object type, or OCR-like string. Many B tasks then reduce to exact or near-exact search. By contrast, A often starts with broad category text and must discover the target through long Magento result pages.

## 5. FP Source: Raw 16.56% -> Adjusted 13.55%

All 14 adjusted-out shopping false positives are `na_fp`.

| FP reason | count | taxonomy split | notes |
|---|---:|---|---|
| `na_fp` | 14 | B=5, C=8, D=1, A=0 | Evaluator credited cases where the trajectory had no effective finish / no applicable completion evidence. |
| `eval_fp` | 0 | none | No shopping adjusted-out case was labeled evaluator false positive. |
| `visual_fp` | 0 | none | No separate visual-FP label appears in this diagnostic export. |

There is no fallback-finish pattern: `fallback_finish_count=0` in `condition_overview.csv`, and the adjusted-out tasks mostly end with non-finish actions such as scroll/type/click while the evaluator had raw success. Examples include task 90 (`kite`, final type/no progress), task 125 (exact bag, final scroll/no progress), task 188 (4-seater car, max-step search repeat), and task 333 (exact red iPhone XR, max-step search repeat).

Cross-mode/site DOM FP comparison:

| run | DOM episodes | raw SR | adjusted SR | FP count | FP rate | FP reasons |
|---|---:|---:|---:|---:|---:|---|
| B0 classifieds DOM | 234 | 14.96% | 14.10% | 2 | 0.85% | 2 `na_fp` |
| B0 reddit DOM | 210 | 11.43% | 8.10% | 7 | 3.33% | 4 `na_fp`, 3 `eval_fp` |
| B0 shopping DOM | 465 | 16.56% | 13.55% | 14 | 3.01% | 14 `na_fp` |

Shopping's FP rate is close to reddit DOM, but the composition differs: shopping FPs are all non-applicable/no-effective-completion cases, not evaluator-string mismatches. This points to Magento task/evaluator noise around already-satisfied or non-finished trajectories rather than a systematic prompt template that calls `finish` too early.

## 6. Section 5 Mechanism Implications

For the representation axis, Magento shopping shows that DOM AXTree is not a simple "text is enough" baseline. On dense product-list pages, AXTree exposes many fields but does not preserve the visual layout and product-card grouping that a human uses to bind title, image, price, color swatch, rating, and action button. The model then treats the first visible or first sorted item as sufficient, loses global list context, or cannot verify visual attributes hidden behind image/style options. A text-payload swap such as SoM marks can help by making action targets and visible product units more compact, but the root problem is representation binding: comparison and visual-attribute verification require the observation to preserve which product row a price/color/action belongs to.

The successful B cases clarify that image input helps most when it compresses the target into a textual key before page interaction. This is different from full page vision. Reference images help DOM because the model can search "matcha powder," "Chipotle Cinnamon Rub," or "pink knitted hat"; current-page screenshots would help at the later binding stage, where the question is whether the visible product is actually blue, round, cupcake-style, or the cheapest among the displayed candidates.

## 7. Section 7 Generalization Implications

B0 DOM shopping is not simply weaker or stronger than classifieds/reddit; it shifts the capability profile. Adjusted DOM SR is 13.55% on shopping, 14.10% on classifieds, and 8.10% on reddit. Shopping's raw DOM SR is the highest of the three (16.56%), but its adjusted SR drops by 3.0 points due to 14 `na_fp` cases. The failure mix is also different: shopping is dominated by `fail_no_progress` (37.4%) and `fail_finish_eval_mismatch` (28.4%), classifieds has more wrong-url failures (18.8% DOM), and reddit has more search-repeat/max-step behavior (13.8% DOM search-repeat).

This means cross-site generalization should not be summarized as "DOM handles text tasks." DOM handles short exact lookup and form/state-update tasks reasonably, but it degrades on sites where textual accessibility output must stand in for product-card visual binding and ranked-list aggregation. Shopping also shows that reference images can improve a DOM run when they provide semantic target labels, so visual task taxonomy and observation modality should be separated in the paper: task image availability is not the same causal factor as page screenshot availability.

## 8. Paper-Ready Section 4 Summary

The shopping DOM run exposes a useful inversion: tasks audited as non-visual text-only had lower adjusted success than reference-image tasks (8.54% vs. 21.30%). The gap is explained by task composition rather than a DOM advantage on visual work. The text-only bucket concentrates Magento list aggregation and category-scoped comparison tasks; its comparison and cheapest/most-expensive subsets both achieved 0% adjusted success. Many also encode latent visual attributes such as color, shape, theme, or design without providing an image. In contrast, reference-image tasks often let the model translate the target image into a compact search phrase or product name, turning visual grounding into text retrieval before page interaction.
