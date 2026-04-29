# Site Mechanism Dictionary

Generated: 2026-04-29T17:37:59+01:00

Method: synthesis of site digests, `paper_planning.md` §2 substrate framing, Layer 0-3 aggregate files, shopping audit files, the swatch form-state audit, and the disagreement-cluster diagnostic.

Important caveats:

- Fresh Layer 0 SR and FP numbers come from `docs/analysis/cross_sites/sr_fp_per_mode.json`.
- Older per-mode digests are used for qualitative mechanisms and case studies; their older SR lines are not treated as authoritative when they differ from fresh Layer 0.
- Shopping is sparse: only `vwa_shopping/B0_findings.md` and `B0_DOM_digest.md` exist.
- `docs/analysis/phantom_paper/disagreement_clusters.md` is cited only as a 2026-04-27 stale snapshot / baseline-only diagnostic caveat, per queue instructions.

## Reddit

### Substrate

| Field | Mechanism |
|---|---|
| Site | Reddit / Postmill |
| N | 210 tasks |
| Information structure | Forum hierarchy: forum -> posts -> comments |
| Navigation affordance | Sidebar `f/<forum>` links, search box, post/comment links, sort controls |
| Image role | Content and target clue, especially post attachments/reference images; not a navigation affordance |
| Intrinsic search | False. Search-loop is usually a failure basin, not the intended substrate |
| URL routing | Path-based `/f/<forum>/<post>/<comment>` |

Layer 0 fresh outcome:

| Mode | Adjusted SR | FP rate |
|---|---:|---:|
| DOM | 9.52% | 1.90% |
| P-DOM | 12.38% | 1.43% |
| P-SoM | 13.81% | 0.48% |
| SoM | 10.48% | 1.43% |
| Vision | 6.67% | 1.90% |

The substrate claim is that reddit is dominated by text/repr structure and prompt prior, with a weak/balanced image axis. The site has many image-referential tasks, but images are content objects to be identified; they do not organize the navigation surface. AXTree therefore pushes the agent toward keyword search, while `[SOM_MARKS]` plus prompt changes where the agent commits.

### Axis 1 Text Payload

Dominance: PRIMARY.

Mechanism:

AXTree hierarchy exposes a deep forum/post/comment tree and makes the search box an attractive shortcut. `[SOM_MARKS]` flattens the action surface, exposes direct forum/post affordances, and changes which candidate links are attended to.

Layer 0 evidence:

- +P-DOM adds +3.81pp oracle lift over the 3-mode baseline.
- P-DOM adds 8 tasks; P-DOM-only tasks = 4.
- B0 adjusted SR moves from DOM 9.52% to P-DOM 12.38%.

Layer 1 evidence:

- Text-axis n_steps effect: -1.25 steps.
- Text-axis type_frac effect: -2.34pp.
- Text-axis action_repeat_frac effect: +4.64pp.
- Search-loop text-axis effect is only -2.38pp and not meaningful in Layer 1; the stronger search-loop change appears in the compound/prompt path.
- Whole-run search-loop gradient: DOM 51.90% -> P-SoM 35.71% -> SoM 31.43%.

Layer 2 evidence:

| Metric | Value |
|---|---:|
| URL-path Jaccard | 0.572555 |
| Click-target Jaccard | 0.463450 |
| Target-hit diff | +3.47pp |
| Max keyword repeat diff | -0.633 |
| First-action divergence | 20.95% |

Layer 3 evidence:

- DOM costs $0.0516/ep.
- P-DOM costs $0.0459/ep.
- DOM uses 12.70 steps; P-DOM uses 11.45 steps.
- This is a text-surface swap with no image tax.

Case studies:

- Task 4: DOM guesses from image/context and searches `"wheat field city skyline"` repeatedly.
- Task 23: `"pumpkin robot"` repeats 15 times in DOM and also appears in SoM search-loop examples.
- Task 30: `"colmscomics"` repeats 15 times in DOM.
- Task 72: B0 DOM escapes a Postmill comment self-link loop by scrolling to the textarea.
- E2 DOM vs P-SoM symmetric-diff cases: tasks 7, 15, 81.

Digest quote:

> DOM repeats the same keyword 5-15 times; examples include task 23 "pumpkin robot" x15 and task 4 "wheat field city skyline" x7.

Interpretation for Section 5:

Text payload structure is not merely a cheaper encoding. On reddit it changes the trajectory basin. The same user intent can become a 30-step search loop under hierarchical AXTree, while flat marks expose direct clickable candidates or lead the agent to commit earlier.

### Axis 2 Prompt

Dominance: PRIMARY_MACRO_DRIVER.

Mechanism:

The SoM prompt acts as a task-conditional decision prior. Holding flat text constant, it changes search phrasing, first actions, use of `tab_focus`, backtracking, and commitment. It reduces typing and shifts the agent out of repeated search.

Layer 0 evidence:

- +P-SoM adds +3.33pp oracle lift.
- P-SoM adds 7 tasks; P-SoM-only tasks = 3.
- P-DOM vs P-SoM task-pool Jaccard = 0.571, below the 0.7 redundancy sentinel.

Layer 1 evidence:

- Prompt axis dominates search_loop: -13.81pp, Cohen's h = -0.280.
- Prompt axis dominates type_frac: -6.58pp, d = -0.356.
- Prompt axis dominates scroll_frac: -3.79pp, d = -0.152.
- Prompt axis changes n_steps: -1.55.
- Action-vocabulary top shifts: `tab_focus` +0.0828, `type` -0.0658, `scroll` -0.0379.

Layer 2 evidence:

| Metric | Value |
|---|---:|
| URL-path Jaccard | 0.532997 |
| Click-target Jaccard | 0.484405 |
| Target-hit diff | -4.17pp |
| First-action divergence | 20.48% |
| P-DOM vs P-SoM symmetric-diff N | 15 |
| Median first divergent step | 0 |
| Early divergence | 100% |

Layer 3 evidence:

- P-DOM costs $0.0459/ep.
- P-SoM costs $0.0381/ep.
- The cost difference is driven by fewer steps, not image tokens.

Case studies:

- E2 P-DOM vs P-SoM cases: tasks 7, 26, 167.
- Task 124: Phantom-SoM-only in the disagreement snapshot; cite only with the 04-27 snapshot / baseline-only caveat.
- Task 28: SoM repeats `"baseball"` search, showing prompt helps but does not eliminate the failure basin.

Digest quote:

> SoM search-loop is lower than DOM but still includes task 23 "pumpkin robot" x15 and task 28 "baseball" repeated search.

Interpretation for Section 5:

Prompt is not only confidence wording. On reddit, it is the largest macro driver for search/type behavior and shifts trajectories at step 0 in the P-DOM/P-SoM symmetric-diff set. This supports the "prompt as task-conditional decision prior" claim.

### Axis 3 Image

Dominance: WEAK_BALANCED.

Mechanism:

Images help when the task is genuinely image-post matching or visual readout, but the screenshot/mark channel also introduces early visual confidence, mark occlusion, coordinate errors, and text-over-vision fallback. Since images are content rather than navigation affordances on reddit, the net effect is mixed.

Layer 0 evidence:

- P-SoM adjusted SR = 13.81%.
- SoM adjusted SR = 10.48%.
- Vision adjusted SR = 6.67%.
- Vision still contributes 11/34 adjusted oracle selections in the B0 three-mode digest.

Layer 1 evidence:

- Image axis improves finish_rate by +10.95pp.
- Image axis search_loop effect is -4.29pp, with CI crossing zero.
- Image axis n_steps effect is -1.85.
- Image axis type_frac effect is -2.50pp.
- `reddit_task_6` probe: OCR drops from roughly 75-78% without marks to 15-18% with SoM labels.

Layer 2 evidence:

| Metric | Value |
|---|---:|
| URL-path Jaccard | 0.456434 |
| Click-target Jaccard | 0.491667 |
| Target-hit diff | +4.86pp |
| First-action divergence | 37.62% |
| P-SoM vs SoM symmetric-diff N | 23 |
| Median first divergent step | 0 |
| Early divergence | 95.24% |

Layer 3 evidence:

- P-SoM costs $0.0381/ep.
- SoM costs $0.0409/ep.
- SoM has image-token cost, but also fewer steps than DOM.

Case studies:

- Task 0: P-SoM succeeds while SoM visual-hijack/click-loop fails in mechanism/disagreement evidence.
- Task 14: SoM-only success; Vision search-loops on cityscape query in the stale disagreement snapshot.
- Task 30: mark occlusion is noted in SoM digest.
- Task 6: SoM OCR probe shows label occlusion damage.

Digest quote:

> SoM failures include text_over_vision 83 cases (45.6%), ID hallucination 8 cases, and mark occlusion 2 cases.

Interpretation for Section 5:

The reddit image axis should be written as bidirectional. It is not irrelevant: Vision has oracle value and image can solve content tasks. But it is not the dominant navigation mechanism, and the marked-image implementation can actively damage grounding.

### Reddit Compound DOM To P-SoM

| Metric | Value |
|---|---:|
| P-DOM vs P-SoM task-pool Jaccard | 0.5714 |
| DOM vs P-SoM URL-path Jaccard | 0.481306 |
| DOM vs P-SoM click-target Jaccard | 0.420556 |
| DOM vs P-SoM target-hit diff | -0.69pp |
| P-SoM distinct from both endpoints | 2/8 Layer 1 cells |

Compound claim:

P-SoM is not redundant with DOM. Even where aggregate SR changes are modest, compound click-target Jaccard is 0.421 and the oracle adds +3.33pp through P-SoM alone.

### Reddit Failure Modes

| Failure mode | Evidence |
|---|---|
| Search repeat | DOM reason count 29/210; digest category 55/186; cases 4, 23, 30 |
| No progress | DOM 47/210; SoM 72/210; Vision 84/210 |
| Finish/eval mismatch | DOM 50/210; cases include 5, 22, 25 |
| Comment self-link loop | 28/210 DOM tasks; task 72 escape case |

### Reddit Quirks

- Postmill comment count links can self-link to the same page.
- B1 reddit SoM remains below DOM after max_marks 80 -> 200 rerun.
- P79 Universal SoM labels can damage OCR/attention; claims involving mark harm should be scoped to this implementation.
- Image-upstream removals or stale media should be kept out of fresh quantitative claims unless the source file directly supports them.

### Reddit Narrative Anchor

Reddit's forum hierarchy makes representation and prompt the leading mechanisms. AXTree depth plus image-referential tasks push DOM into repeated keyword search; flat marks and SoM-style prompt priors open a different, low-cost trajectory basin. The image channel is useful for some content tasks but is not the site navigation substrate, so full SoM is not a monotone improvement over P-SoM.

## Classifieds

### Substrate

| Field | Mechanism |
|---|---|
| Site | Classifieds / OSClass |
| N | 234 tasks |
| Information structure | Product listings, categories, search results, item pages |
| Navigation affordance | Category select/dropdown, search box, pagination, item links, price filters |
| Image role | Product identity and visual disambiguation |
| Intrinsic search | True |
| URL routing | Query-heavy OSClass routes, often `/index.php?page=item&id=N` |

Layer 0 fresh outcome:

| Mode | Adjusted SR | FP rate |
|---|---:|---:|
| DOM | 14.10% | 0.85% |
| P-DOM | 14.53% | 2.14% |
| P-SoM | 14.53% | 1.28% |
| SoM | 21.37% | 1.71% |
| Vision | 13.68% | 2.14% |

The substrate claim is that classifieds is image-dominant. The product identity often lives in the photo, not the path string. But the site also shows why aggregate SR can mislead: P-DOM and P-SoM have nearly identical aggregate SR while their task pools and click targets diverge.

### Axis 1 Text Payload

Dominance: SECONDARY.

Mechanism:

AXTree vs `[SOM_MARKS]` changes event-level grounding and click targets, but OSClass path structure compresses many semantic differences into similar URL paths. The text axis is therefore weaker at path-level macro metrics than on reddit, while still visible in click-target divergence and oracle lift.

Layer 0 evidence:

- +P-DOM adds +3.42pp oracle lift.
- P-DOM adds 8 tasks; P-DOM-only tasks = 5.
- B0 adjusted SR: DOM 14.10%, P-DOM 14.53%.

Layer 1 evidence:

- Text-axis finish_rate effect: +2.14pp.
- Text-axis search_loop effect: -3.85pp, not meaningful.
- Text axis is not the dominant classifieds macro axis.
- B0 DOM paginates in 33+ tasks and uses `sPriceMin`/`sPriceMax` in 21+ tasks.

Layer 2 evidence:

| Metric | Value |
|---|---:|
| URL-path Jaccard | 0.903684 |
| Click-target Jaccard | 0.561033 |
| Target-hit diff | +2.33pp |
| Max keyword repeat diff | +0.077 |
| First-action divergence | 13.68% |

Layer 3 evidence:

- DOM costs $0.0427/ep.
- P-DOM costs $0.0397/ep.
- DOM uses 11.56 steps; P-DOM uses 11.16 steps.

Case studies:

- Task 63: DOM succeeds and P-DOM fails with first divergent step 0.
- Task 201: P-DOM succeeds after late divergence from DOM.
- Task 222: DOM hallucinates visual content in the digest.

Digest quote:

> B0 DOM includes 33+ pagination tasks and 21+ price-range filter tasks.

Interpretation for Section 5:

Axis 1 on classifieds is best used to argue that aggregate path metrics understate per-event decision change. It is not the main outcome driver, but it helps establish that representation shifts behavior even on a site where image ultimately dominates.

### Axis 2 Prompt

Dominance: SECONDARY_TASK_CONDITIONAL.

Mechanism:

Prompt changes commitment and action vocabulary more than headline SR. P-DOM and P-SoM have equal adjusted SR in fresh Layer 0, but their task-pool Jaccard is only 0.447. This is the cleanest classifieds example of aggregate SR hiding routing complementarity.

Layer 0 evidence:

- +P-SoM adds +2.56pp oracle lift.
- P-SoM adds 6 tasks; P-SoM-only tasks = 3.
- P-DOM vs P-SoM task-pool Jaccard = 0.447.

Layer 1 evidence:

- Prompt axis dominates type_frac: -3.73pp.
- Prompt axis dominates selfcorr_count: +0.342.
- Prompt axis dominates click_frac: +2.14pp.
- Action shifts: `type` -0.0373, `click` +0.0214, `back` +0.0209, `scroll` +0.0182.

Layer 2 evidence:

| Metric | Value |
|---|---:|
| URL-path Jaccard | 0.885897 |
| Click-target Jaccard | 0.541814 |
| Target-hit diff | -0.58pp |
| First-action divergence | 17.95% |
| P-DOM vs P-SoM symmetric-diff N | 26 |
| Median first divergent step | 1 |
| Early divergence | 92.31% |

Layer 3 evidence:

- P-DOM costs $0.0397/ep.
- P-SoM costs $0.0441/ep.
- P-SoM takes more steps in this site, so prompt does not automatically reduce cost.

Case studies:

- Tasks 17, 79, 103: P-DOM vs P-SoM symmetric-diff examples.
- Task 60: Phantom-SoM-only in stale disagreement snapshot; DOM visual-missing and SoM wrong-commit counterparts.
- Task 93: Phantom-SoM-only in stale disagreement snapshot.

Digest quote:

> SoM screenshots/marks can create an "information sufficient" illusion and early finish on similar but wrong items.

Interpretation for Section 5:

Prompt should be framed as a decision-prior axis. It can preserve aggregate SR while changing which tasks are solved and which actions are attempted. This directly supports the routing-arm thesis.

### Axis 3 Image

Dominance: PRIMARY.

Mechanism:

Classifieds item identity is visual. The image axis supplies product identity for color/style/object matching and helps commit to item pages. It also reduces scrolling, repeated action, and search-loop behavior.

Layer 0 evidence:

- P-SoM adjusted SR = 14.53%.
- SoM adjusted SR = 21.37%.
- SoM is the best single mode.
- 3->5-mode oracle lift = +4.70pp.
- B0 digest: SoM has 21 SoM-only adjusted successes.

Layer 1 evidence:

- Image axis search_loop effect: -11.97pp.
- Image axis scroll_frac effect: -9.55pp.
- Image axis finish_rate effect: +26.50pp, Cohen's h = 0.567.
- Image axis n_steps effect: -3.45.
- Image axis action_repeat_frac effect: -17.08pp.
- Page-screen-required subset: SoM-DOM +13.5pp in B0 and +10.4pp in B1.

Layer 2 evidence:

| Metric | Value |
|---|---:|
| URL-path Jaccard | 0.904131 |
| Click-target Jaccard | 0.482123 |
| Target-hit diff | +1.74pp |
| First-action divergence | 22.65% |
| P-SoM vs SoM symmetric-diff N | 42 |
| Median first divergent step | 1 |
| Early divergence | 94.87% |

Layer 3 evidence:

- P-SoM costs $0.0441/ep.
- SoM costs $0.0415/ep.
- SoM has image-token cost, but fewer steps: 8.60 vs 12.05.

Case studies:

- Task 14: SoM succeeds while P-SoM fails at first divergent step 0.
- Task 49: blue LED item; DOM lacks visual product identity.
- Task 115: body-part image matching; DOM/Vision grounding failures, SoM succeeds in stale disagreement snapshot.
- Task 52: wave painting; Vision lacks stable text grounding in stale disagreement snapshot.

Digest quote:

> B0 SoM adjusted SR is highest and significantly better than DOM; McNemar p=0.012.

Interpretation for Section 5:

Classifieds is the strongest image-axis site. Full SoM wins because the site asks the agent to identify products visually. The important nuance is that P-SoM still has routing value: the image axis dominates aggregate improvement, but prompt/text arms remain task-pool complementary.

### Classifieds Compound DOM To P-SoM

| Metric | Value |
|---|---:|
| P-DOM vs P-SoM task-pool Jaccard | 0.4468 |
| DOM vs P-SoM URL-path Jaccard | 0.884577 |
| DOM vs P-SoM click-target Jaccard | 0.530912 |
| DOM vs P-SoM target-hit diff | +1.74pp |
| P-SoM distinct from both endpoints | 2/8 Layer 1 cells |

Compound claim:

P-DOM and P-SoM have the same fresh adjusted SR, 14.53%, but P-DOM/P-SoM task-pool Jaccard is 0.447. Aggregate SR alone would miss the routing signal.

### Classifieds Failure Modes

| Failure mode | Evidence |
|---|---|
| DOM no progress | 62/234, 26.5% |
| SoM wrong URL | 57/234, 24.4% |
| Vision no progress | 93/234, 39.7% |
| DOM visual missing in disagreement slice | 22 failure pairs, 04-27 snapshot caveat |

### Classifieds Quirks

- OSClass routes make path-level URL overlap high even when item IDs differ.
- Native category `<select>` can trigger a capability-environment gap.
- TinyMCE iframe description editing is a shared scaffold limitation.
- §95 removed the old visual_fp layer; current adjusted labels deduct N/A FP + eval FP only.

### Classifieds Narrative Anchor

Classifieds is the clean visual-substrate contrast to reddit. Image is not decoration; it is product identity. Yet the equal P-DOM/P-SoM adjusted SR and Jaccard 0.447 show why Section 5 should not infer mechanism from aggregate SR alone.

## Shopping

### Substrate

| Field | Mechanism |
|---|---|
| Site | Shopping / Magento |
| N | 466 tasks |
| Data status | SPARSE / mostly forward-looking |
| Information structure | Product catalog, product pages, cart/checkout, account/profile forms |
| Navigation affordance | Search, category hover/dropdowns, product grids, custom-option radios, quantity/cart forms |
| Image role | Product identity and visual variant selection |
| Intrinsic search | True |
| URL routing | Magento product/category/search/cart routes with product slugs |

Current data:

- Only B0 DOM digest and findings exist under `docs/analysis/vwa_shopping/`.
- No shopping cells exist in `axis_effect_size.json`, `axis1_microbehavior.json`, `mechanism_per_task.json`, `sr_fp_per_mode.json`, or `cost_per_mode.json`.
- Shopping claims below are therefore substrate/audit evidence, not measured axis effects.

### Axis 1 Text Payload

Dominance: MIXED_SECONDARY_FORM_ACTION.

Mechanism:

Text and element structure matter for form actions, dropdowns, radios, quantity fields, profile edits, cart actions, and category menus. DOM element IDs are useful for ordinary fields but fail when the AXTree lacks independent submenu IDs or when state-change bookkeeping collapses same-name radio options.

Layer 0 evidence:

- B0 DOM raw SR = 11.80% (55/466).
- Condition-metrics adjusted SR = 6.44%.
- Reason-diagnostics adjusted SR = 6.24%.
- No P-DOM/P-SoM/SoM/Vision cells are available.

Layer 1 evidence:

- Shopping macro axis effects are missing.
- DOM click failure rate is 24.7%.
- `action_failed` trigger count is 664.
- `form_value_changed` occurs 2309 times in the run.

Layer 2 evidence:

- URL Jaccard: missing.
- Click-target Jaccard: missing.
- Swatch form-state bug: 11/466 B0 DOM episodes match the same-name custom-option radio swatch-loop signature.
- 9/11 affected swatch episodes terminated before finish.

Layer 3 evidence:

- B0 DOM costs $0.0424/ep.
- B0 DOM uses 10.20 steps on average.
- Wasted cost is $0.0374/ep, 88.2% of total.
- No shopping cost cells exist in `cost_per_mode.json`.

Case studies:

- Task 0: red blanket aggregation plus custom-option radio swatch loop.
- Task 18: Basic Cases submenu lacks independent element ID.
- Task 19: Lamps & Shades submenu is unreachable through CSS hover dropdown.
- Tasks 236, 237, 345: profile/address/phone form-action tasks in A4 taxonomy.

Digest quote:

> Magento CSS hover subcategories appear as dropdown-option text without independent element IDs; the agent repeatedly clicks parent menu entries.

Interpretation for Section 5:

Shopping axis 1 should be written cautiously. The existing data show that element/form structure is central, but the missing five-mode cells mean this is not yet a measured text-axis effect. The swatch bug is especially important because it is not an LLM failure; it is a scaffold state-tracking failure.

### Axis 2 Prompt

Dominance: MIXED_FORWARD_LOOKING.

Mechanism:

Prompt effects are expected to split retrieval, aggregation, and form-action behaviors, but the current shopping evidence is taxonomy evidence rather than measured prompt-axis evidence.

Layer 0 evidence:

- A-class shopping audit contains 83 tasks.
- A2 latent visual: 34 tasks.
- A3 aggregation: 32 tasks.
- A1 pure text: 14 tasks.
- A4 form action: 3 tasks.

Layer 1 evidence:

- Shopping prompt macro effects are missing.
- Single-mode DOM routing signals are usable but modest:
- `action_diversity` AUROC = 0.6856.
- `ep_mean_verbalized` AUROC = 0.6808.

Layer 2 evidence:

- URL/click/trajectory prompt metrics are missing.
- A3 aggregation tasks require sorted/scanned candidate pools.
- A4 form tasks require deterministic form edits.

Layer 3 evidence:

- No prompt-axis cost cell exists.

Case studies:

- Task 0: least expensive red blanket; A3 aggregation with visual red secondary.
- Task 12: first Van Gogh search result by relevance; A1 pure text.
- Tasks 236 and 237: address updates; A4 form-action tasks.
- Task 345: profile phone country-code edit; A4 form-action task.

Digest quote:

> Shopping DOM fails through answer alignment and no-progress: 57 answer-alignment errors and 37 execution-stall failures in the 213-failure digest.

Interpretation for Section 5:

Shopping prompt claims should be kept as a data gap. The taxonomy tells us why prompt may matter, but no prompt-axis contrast has been measured yet.

### Axis 3 Image

Dominance: PRIMARY_FORWARD_LOOKING.

Mechanism:

Shopping product choice often depends on color, shape, pattern, packaging, and grid position. DOM can use product names and sparse alt text only; many task-critical attributes are absent from the AXTree. Therefore the image axis is expected to be decisive once SoM/Vision data exist.

Layer 0 evidence:

- 269/466 tasks, or 57.7%, involve visual attributes.
- DOM has 12 visual lucky hits.
- No SoM/Vision shopping SR exists in current aggregates.

Layer 1 evidence:

- Shopping image macro effects are missing.
- DOM bottleneck: 73/213 DOM failures, 34.3%, are directly attributed to missing visual information.
- Target-unreachable failures: 43; 40/43 are visual-missing.

Layer 2 evidence:

- URL/click image metrics are missing.
- A2 latent visual is the largest audited class: 34/83.
- A3 aggregation frequently includes secondary visual filters such as color or pattern.

Layer 3 evidence:

- No image-axis cost cell exists.

Case studies:

- Task 6: red Nike slides; title exposes only one of four red products.
- Task 11: round cookie/ice cream sandwich shape absent from DOM alt text.
- Task 25: blue raincoat; latent visual color.
- Task 54: item with waves; visual pattern plus wishlist action.
- Task 39: colorful product choice; DOM guesses from text.

Digest quote:

> Shopping has 269/466 visual-attribute tasks and 73/213 DOM failures are directly due to missing visual information.

Interpretation for Section 5:

Shopping is not ready for numeric axis claims, but its substrate is clear: the visual axis should matter because product variants and attributes are visual. This should be framed as a forward-looking mechanism supported by DOM failure audit, not as completed cross-mode evidence.

### Shopping Compound DOM To P-SoM

| Metric | Value |
|---|---|
| Task-pool Jaccard | missing |
| URL signature Jaccard | missing |
| Click-target Jaccard | missing |
| Macro independence cells | missing |
| Routing-arm status | unmeasured |

Compound claim:

Do not claim measured P-SoM complementarity for shopping yet. The current dictionary supports why shopping should be included in the next axis runs, not what those runs will show.

### Shopping Failure Modes

| Failure mode | Evidence |
|---|---|
| Visual information missing | 73/213 DOM digest failures, 34.3% |
| Answer alignment error | 57/213, 26.8% |
| Execution stall | 37/213, 17.4% |
| Swatch form-state bug | 11/466 episodes; 9/11 terminated before finish |
| Search loop | 24/213 digest failures, 11.3% |

### Shopping Quirks

- Sparse data: only B0 DOM is available in the site digest folder.
- Same-name radio groups in Magento custom options expose a runner-side `_form_fields_changed` key collision.
- CSS hover dropdown subcategory text lacks independent element IDs.
- Newsletter input can be confused with the search input.
- Product grids are spatial, while AXTree linearizes them.

### Shopping Narrative Anchor

Shopping combines form-interaction complexity with visual product identity. The current data do not support measured P-SoM claims, but they do show why both axis 1 and axis 3 should matter: Magento forms require precise element/state handling, while product and variant decisions often require visual evidence unavailable to DOM.

## Cross-Site Table

### Axis Dominance

| Axis | Reddit | Classifieds | Shopping |
|---|---|---|---|
| Axis 1 text | PRIMARY: URL Jaccard 0.573, click-target Jaccard 0.463, +P-DOM +3.81pp oracle | Secondary: URL Jaccard 0.904, click-target Jaccard 0.561, +P-DOM +3.42pp oracle | Mixed/form-action; sparse evidence: click failure 24.7%, swatch bug 11 episodes |
| Axis 2 prompt | Macro driver for search/type/scroll; prompt search_loop -13.81pp | Type/self-correction/action-vocabulary; P-DOM/P-SoM Jaccard 0.447 | Prompt x task split expected; taxonomy evidence only |
| Axis 3 image | Weak/balanced; image is content, P-SoM 13.81% vs SoM 10.48% | PRIMARY; image finish_rate +26.50pp, h=0.567, SoM 21.37% | PRIMARY forward-looking; 269/466 visual tasks, 73 visual-missing DOM failures |

### Routing Complementarity

| Site | P-DOM vs P-SoM Jaccard | Status |
|---|---:|---|
| reddit | 0.571 | <= 0.7, complementary |
| classifieds | 0.447 | <= 0.7, complementary |
| shopping | missing | data gap |

### Oracle Lift

| Site | 3->5-mode lift | +P-DOM lift | +P-SoM lift |
|---|---:|---:|---:|
| reddit | +5.24pp [2.38, 8.11] | +3.81pp | +3.33pp |
| classifieds | +4.70pp [2.14, 7.69] | +3.42pp | +2.56pp |
| shopping | missing | missing | missing |

### Cost Classes

| Site | B0 API avg | B1 electricity avg | Caveat |
|---|---:|---:|---|
| reddit | $0.0399/ep | $0.000407/ep | deployment classes differ |
| classifieds | $0.0386/ep | $0.000366/ep | deployment classes differ |
| shopping | missing in cost_per_mode | missing | B0 DOM digest reports $0.0424/ep |

## Section 5 Narrative Anchors

### Reddit Anchor

Reddit is a forum-hierarchy site where image is content rather than navigation. AXTree depth and image-referential tasks push DOM into repeated keyword search, visible in the 51.90% -> 35.71% -> 31.43% search-loop gradient from DOM to P-SoM to SoM. The prompt axis then acts as a decision prior, reducing search/type behavior and changing trajectories at step 0 in symmetric-diff tasks. The image axis remains bidirectional: it helps some content tasks but also creates visual confidence, mark occlusion, and coordinate/ID failures.

### Classifieds Anchor

Classifieds is the visual-substrate site. Product identity often lives in the image, so full SoM recovers finish behavior and dominates Layer 1 image metrics, including +26.50pp finish rate. But P-DOM and P-SoM have equal fresh adjusted SR and only 0.447 task-pool Jaccard, so aggregate SR would miss the routing arm. Section 5 should use classifieds to argue both that image can be the dominant axis and that routing complementarity is not reducible to single-mode SR.

### Shopping Anchor

Shopping is sparse, but the failure audit makes its substrate visible. Magento combines visual product/variant identification with brittle form and cart interactions. DOM fails because many product attributes are absent from text, while the swatch audit shows a separate scaffold problem in form-state tracking. Treat shopping as forward-looking until B0/B1 5-mode cells exist.

### Cross-Site Invariant

Where measured, P-SoM is task-pool complementary rather than redundant. P-DOM vs P-SoM Jaccard is 0.571 on reddit and 0.447 on classifieds, both below the 0.7 redundancy sentinel. This supports the routing-arm thesis even when aggregate SR is close to DOM or equal across phantom modes. Shopping is not measured yet and should be named as the data gap.

## Open Questions And Data Gaps

- Shopping needs B0/B1 5-mode data. Current mechanism statements are sparse and mostly forward-looking.
- WA cross-benchmark data are missing for Section 7 generalization.
- Claude/cross-model data are out of scope for this dictionary.
- `disagreement_clusters.md` should be refreshed after phantom traces are restored; current citations must carry the 04-27 snapshot / baseline-only caveat.
- Shopping swatch form-state detector should be fixed and rerun before paper-grade action_failed/no-progress claims.
- Shopping cost/frontier cells should be added to `cost_per_mode.json` after the rerun.
- Shopping axis microbehavior should be added to `axis1_microbehavior.json` or a shopping analogue once all modes exist.

## Self-Check Checklist

| Check | Status |
|---|---|
| 3 sites present | yes |
| 3 axes per site | yes |
| Mechanism fields per axis | dominance, mechanism, Layer 0-3 evidence, case studies, quote |
| At least 6 case IDs | yes: reddit 4/23/30/72/7/15/81, classifieds 14/49/63/115/201, shopping 0/6/11/18/19/25/54 |
| Shopping sparse caveat | explicit |
| Disagreement caveat | explicit wherever used |
| Cross-site invariant | explicit and verifiable |
| Missing data marked missing | yes |

