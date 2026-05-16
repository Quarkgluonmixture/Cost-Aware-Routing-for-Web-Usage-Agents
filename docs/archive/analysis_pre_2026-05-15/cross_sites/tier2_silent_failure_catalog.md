# Tier 2 Silent-Failure Signal Mining Catalog

Audit date: 2026-04-30

## Scope and Denominators

Scanned only the listed Phase 1 VisualWebArena run roots; archives, WA runs, and click reclassification were excluded. The listed roots currently contain 4,493 episode-mode traces, above the prompt's approximate 3,500 estimate.

- Episode-mode traces scanned: 4493
- Steps scanned: 46844
- Failed episode-mode traces used for failure-fraction denominator: 3992
- Skipped runs: 0

The mining pass does not redo the click taxonomy. It treats the existing click-probe result as the companion audit and focuses on TYPE, SCROLL, SELECT_OPTION, FINISH, and cross-step non-navigation anomalies.

The denominator is an episode-mode trace rather than a unique task ID. This is the right unit for this audit because a single task can be run in DOM, SoM, Vision, and phantom variants, and the question is whether a particular policy-observation stack silently loses action effects. The listed roots contain more traces than the prompt estimate; no additional roots were pulled in.

The signatures require state evidence, not only a bad final score. TYPE requires missing text/form/URL echo, repeated no-progress typing, or a stale/offscreen target. SCROLL requires repeated static viewport state or a success-marked static viewport. SELECT_OPTION requires a dropdown state that does not commit or a repeated no-progress option selection. FINISH is broader by design: any failed trace where the agent explicitly terminates is a wrong-state finish, then subclustered by the final URL, answer, and task evaluator target. Cross-step anomalies use large same-URL AXTree shifts after non-navigation actions.

## Cross-Action Summary

The overlap-adjusted estimate is **3052 episode-mode traces**, or **0.765** of all failed traces in this scan. The highest site concentration is **classifieds**; the highest mode concentration is **P-text**.

| Category | Episodes | Blast radius | Dominant site | Dominant mode |
|---|---:|---:|---|---|
| `type_silent_failure` | 549 | 12.22% | classifieds | DOM |
| `scroll_silent_failure` | 667 | 14.85% | classifieds | DOM |
| `select_option_silent_failure` | 149 | 3.32% | classifieds | DOM |
| `finish_wrong_state` | 1972 | 43.89% | classifieds | DOM |
| `cross_step_trajectory_anomaly` | 353 | 7.86% | classifieds | DOM |

## Category Findings

### TYPE Silent Failure

**Blast radius.** 549 episode-mode traces (12.22%). Mode breakdown: {'DOM': 224, 'P-SoM': 79, 'P-prompt': 12, 'P-text': 76, 'SoM': 78, 'Vision': 80}. Site breakdown: {'classifieds': 291, 'reddit': 172, 'shopping': 86}.

**Candidate root cause.** Typing frequently lands on stale, offscreen, or non-submitting elements: the runner may report success, but URL/form state and AXTree text do not echo the typed value, leaving the agent to repeat the same search or continue from an unchanged page.

**Interpretation.** The dominant evidence pattern is not a normal failed search. The agent emits text, often with a newline that should submit a search or fill a form, but the post-action URL, form flags, and AXTree text do not contain the intended value. Several cases also expose the runner targeting a 0,0,10,10 bounding box, which is a strong stale-element or hidden-element signature. This family is a direct runtime counterpart to static action-dispatch concerns: a syntactically valid TYPE action is accepted, but the state transition that the policy relies on is absent.

**Representative cases.**

- classifieds task 0 (B0_3mode_classifieds, DOM), steps [5, 6, 7]: repeated_same_element_type_without_echo;runner_failed_no_progress;same_url_no_form_or_text_echo;slow_env_step_without_text_echo;stale_or_offscreen_bbox, runner_failed_no_progress;same_url_no_form_or_text_echo;slow_env_step_without_text_echo;stale_or_offscreen_bbox. Evidence: text `blue kayak for sale`; url `http://100.95.81.103:9980/index.php?page=search&sPattern=blue+kayak+&sOrder=i_price&iOrderType=asc`.
- classifieds task 5 (B0_3mode_classifieds, DOM), steps [15]: runner_failed_no_progress;same_url_no_form_or_text_echo;slow_env_step_without_text_echo;stale_or_offscreen_bbox. Evidence: text `white car`; url `http://100.95.81.103:9980/index.php?page=user&action=items`.
- classifieds task 11 (B0_3mode_classifieds, DOM), steps [7]: runner_failed_no_progress;same_url_no_form_or_text_echo. Evidence: text `blue bike`; url `http://100.95.81.103:9980/index.php?page=item&id=71313`.
- classifieds task 29 (B0_3mode_classifieds, DOM), steps [7, 8, 9, 10, 11, 12, 13, 16]: repeated_same_element_type_without_echo;runner_failed_no_progress;same_url_no_form_or_text_echo;slow_env_step_without_text_echo;stale_or_offscreen_bbox, runner_failed_no_progress, runner_failed_no_progress;same_url_no_form_or_text_echo;slow_env_step_without_text_echo;stale_or_offscreen_bbox. Evidence: text `red car 2024`; url `http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOrderType=desc&sPattern=red+car+for+sale+`.

### SCROLL Silent Failure

**Blast radius.** 667 episode-mode traces (14.85%). Mode breakdown: {'DOM': 192, 'P-SoM': 116, 'P-prompt': 24, 'P-text': 90, 'SoM': 94, 'Vision': 151}. Site breakdown: {'classifieds': 373, 'reddit': 240, 'shopping': 54}.

**Candidate root cause.** Scroll no-ops are concentrated where the viewport is already pinned, a modal/overlay captures scroll, or the target page is not scrollable; the agent receives another near-identical AXTree and tends to spend more steps looking for hidden content.

**Interpretation.** The scroll family is mostly repeated no-op scrolling. A single no-op at the bottom of a page can be benign, but two or more consecutive scrolls with identical scroll_y and near-identical AXTree text is a silent progress failure for the policy: the agent has no new content and often keeps searching for items that are not reachable through the current viewport state. These are especially visible on classifieds and Vision traces, where the agent spends budget probing listing pages or image/detail pages that no longer move.

**Representative cases.**

- classifieds task 5 (B0_3mode_classifieds, DOM), steps [14]: consecutive_scrolls_no_viewport_move.
- classifieds task 6 (B0_3mode_classifieds, DOM), steps [11]: consecutive_scrolls_no_viewport_move.
- classifieds task 11 (B0_3mode_classifieds, DOM), steps [4, 10, 11]: consecutive_scrolls_no_viewport_move.
- classifieds task 12 (B0_3mode_classifieds, DOM), steps [5]: consecutive_scrolls_no_viewport_move.

### SELECT_OPTION Silent Failure

**Blast radius.** 149 episode-mode traces (3.32%). Mode breakdown: {'DOM': 35, 'P-SoM': 32, 'P-prompt': 4, 'P-text': 19, 'SoM': 32, 'Vision': 27}. Site breakdown: {'classifieds': 117, 'reddit': 27, 'shopping': 5}.

**Candidate root cause.** Dropdown failures mostly arise from selecting unavailable or DOM-stale options. The page keeps the previous sort/category state, often with action_success=false or no selected-option echo, so the agent continues under a false filter assumption.

**Interpretation.** SELECT_OPTION failures are smaller in absolute count but high value diagnostically. They concentrate on classifieds sort/category widgets, especially unavailable options such as `Oldest first` or stale category comboboxes. The page remains on the old URL or old selected value, yet the agent often reasons as if the filter changed. This is the cleanest non-click analogue of button/AJAX silent failure: the command targets a UI control whose state is supposed to commit into query parameters or selected text, but the next observation does not encode that commit.

**Representative cases.**

- classifieds task 60 (B0_3mode_classifieds, DOM), steps [2]: repeated_same_option_without_state_update. Evidence: option `Cars + trucks`; url `http://100.95.81.103:9980/`.
- classifieds task 74 (B0_3mode_classifieds, DOM), steps [1, 2]: runner_failed_no_progress. Evidence: option `Cars + trucks`; url `http://100.95.81.103:9980/`.
- classifieds task 81 (B0_3mode_classifieds, DOM), steps [1]: runner_failed_no_progress. Evidence: option `Books`; url `http://100.95.81.103:9980/`.
- classifieds task 114 (B0_3mode_classifieds, DOM), steps [14, 15, 16]: repeated_same_option_without_state_update;runner_failed_no_progress;same_url_dropdown_not_selected, runner_failed_no_progress;same_url_dropdown_not_selected. Evidence: option `Oldest first`; url `http://100.95.81.103:9980/index.php?page=search&sPattern=Virginia+&iPage=104&sOrder=dt_pub_date&iOrderType=desc`.

### FINISH Wrong-State Failure

**Blast radius.** 1972 episode-mode traces (43.89%). Mode breakdown: {'DOM': 604, 'P-SoM': 285, 'P-prompt': 87, 'P-text': 253, 'SoM': 447, 'Vision': 296}. Site breakdown: {'classifieds': 1033, 'reddit': 710, 'shopping': 229}.

**Candidate root cause.** Wrong-state finishes are dominated by agents completing from an answer-shaped local observation rather than the evaluator target state: search pages, image-only pages, partial form/message workflows, and confident but unevaluated claims.

**Interpretation.** FINISH dominates because it captures the policy-level endpoint of silent failure: the agent decides the task is complete while the evaluator state is false. This should not be read as an action-dispatch bug by itself. The subcategories separate direct hallucinated answers and image-blind guesses from search-result finishes, off-target URL finishes, and partial-work false confidence. This family is important for paper framing because many upstream silent failures only become measurable when the agent converts a misleading observation into a confident termination.

**Root-cause subcategories.**

- `agent_false_confidence_after_partial_work`: 646
- `wrong_state_unclear_from_trace`: 557
- `agent_finished_on_search_results_page_not_target`: 432
- `visual_perception_guess_or_image_blind_finish`: 158
- `finished_off_eval_target_url`: 156
- `agent_hallucinated_answer`: 23

**Representative cases.**

- classifieds task 1 (B0_3mode_classifieds, DOM), steps [3]: agent_false_confidence_after_partial_work. Evidence: answer `Found the cheapest red Toyota: 2007 Toyota Yaris priced at $3000 in Gainesville, Virginia.`; url `http://100.95.81.103:9980/index.php?page=item&id=7846`.
- classifieds task 2 (B0_3mode_classifieds, DOM), steps [6]: agent_false_confidence_after_partial_work. Evidence: answer `Found: Vintage Carved Jade Floral Ring 14kt (Reduced), $300.00, published 2023/11/01, Towson, Maryland.`; url `http://100.95.81.103:9980/index.php?page=item&id=69296`.
- classifieds task 3 (B0_3mode_classifieds, DOM), steps [16]: agent_false_confidence_after_partial_work. Evidence: answer `Nikon Z6 II, $1000, published 2023/11/15, not completely black, in Photo + video category.`; url `http://100.95.81.103:9980/index.php?page=item&id=60392`.
- classifieds task 4 (B0_3mode_classifieds, DOM), steps [7]: agent_false_confidence_after_partial_work. Evidence: answer `The price of the white car listing has been updated to $25,000 in both the price field and the description.`; url `http://100.95.81.103:9980/index.php?page=item&action=item_edit&id=84144`.

### Cross-Step Trajectory Anomaly

**Blast radius.** 353 episode-mode traces (7.86%). Mode breakdown: {'DOM': 183, 'P-SoM': 47, 'P-prompt': 5, 'P-text': 45, 'SoM': 62, 'Vision': 11}. Site breakdown: {'classifieds': 192, 'reddit': 95, 'shopping': 66}.

**Candidate root cause.** Large same-URL AXTree shifts without a navigation-like action point to frame-side async refreshes, form-only rerenders, modal state changes, and stale cache/observation replacements that change what the agent sees without an explicit navigation signal.

**Interpretation.** The cross-step family captures traces where the AXTree changes sharply without a navigation-like action or URL transition. Many overlap with TYPE because search pages rerender in place or stale observations are replaced after form-like actions; others are modal or async page refreshes. This is not necessarily a user-facing bug in isolation, but it is a benchmark-control risk: the agent cannot distinguish an expected in-place update from an observation-cache replacement unless the runner exposes a stronger transition reason.

**Representative cases.**

- classifieds task 0 (B0_3mode_classifieds, DOM), steps [4]: same_url_large_axtree_shift_non_navigation_action. Evidence: action `type`; url `http://100.95.81.103:9980/index.php?page=search&sPattern=blue+kayak+&sOrder=i_price&iOrderType=asc`.
- classifieds task 30 (B0_3mode_classifieds, DOM), steps [8]: same_url_large_axtree_shift_non_navigation_action. Evidence: action `type`; url `http://100.95.81.103:9980/index.php?page=search&sPattern=black+couch+&sOrder=i_price&iOrderType=desc&iPage=2`.
- classifieds task 68 (B0_3mode_classifieds, DOM), steps [3]: same_url_large_axtree_shift_non_navigation_action. Evidence: action `type`; url `http://100.95.81.103:9980/index.php?page=search&sPattern=Tiger+Woods+video+game+`.
- classifieds task 70 (B0_3mode_classifieds, DOM), steps [5]: same_url_large_axtree_shift_non_navigation_action. Evidence: action `type`; url `http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOrderType=desc&sPattern=green+tractor+&sCity=Delaware+`.

## Evidence Quality

These counts should be used as a blast-radius estimate, not as hand-adjudicated ground truth. The strongest evidence categories are SELECT_OPTION and TYPE because they have concrete expected postconditions: query text, form-value change, selected option text, or URL parameters. SCROLL is also strong when repeated, but a single static scroll at page bottom is not enough, so the miner only promotes repeated static viewport patterns or success-marked no movement. Cross-step anomalies are weaker as root-cause labels but useful as alerts for same-URL observation instability.

FINISH is deliberately framed differently. A wrong-state finish is not proof that the finish action failed; it is proof that the agent terminated from a state that did not satisfy the evaluator. The subcategory split is therefore the important paper signal. Search-page finishes and off-evaluator-URL finishes are high-confidence state mismatch. Image-blind guesses and hallucinated answers are policy/evaluator mismatch. The large `wrong_state_unclear_from_trace` bucket should be reserved for appendix-level examples or manual follow-up rather than headline claims.

The overlap-adjusted unique count is lower than the per-category sum because one trace can contain both an upstream action-state anomaly and a wrong-state finish. That overlap is expected and useful: it gives a trajectory-level story from dispatch loss or observation instability to final false confidence.

## Section 4 Wiring

Use this catalog as the non-click complement to the existing click-probe taxonomy. In Section 4, the clean wiring is to present click failures as the first family, then add these five non-click families as evidence that silent failure is not a single click-dispatch phenomenon. The strongest paper claims are:

- TYPE and SELECT_OPTION show dispatch/state-commit bugs: the action parser and runner accept a command, but DOM state, URL state, or selected value does not commit in a way visible to the next policy call.
- SCROLL and cross-step anomalies show observation-transition bugs: the policy sees either a repeated page after an accepted action or a large same-URL AXTree replacement that is not tied to a navigation action.
- FINISH wrong-state failures show policy/evaluator mismatch: the agent can confidently terminate on an answer-shaped local state while the programmatic evaluator requires a different URL, database side effect, or exact content.

For the paper table, report `n_episodes`, blast radius, dominant site/mode, and one task ID per family. For the narrative, pair this with Tier 1 static candidates: TYPE and SELECT_OPTION are the best confirmation targets for action-dispatch code paths; FINISH is the best evidence for policy-level false confidence and evaluator-state mismatch.

## Self-Check

- `mode_breakdowns_sum_to_n_episodes`: {'type_silent_failure': True, 'scroll_silent_failure': True, 'select_option_silent_failure': True, 'finish_wrong_state': True, 'cross_step_trajectory_anomaly': True}
- `case_study_count_at_least_3`: {'type_silent_failure': True, 'scroll_silent_failure': True, 'select_option_silent_failure': True, 'finish_wrong_state': True, 'cross_step_trajectory_anomaly': True}
- `category_episode_floor_at_least_5`: {'type_silent_failure': True, 'scroll_silent_failure': True, 'select_option_silent_failure': True, 'finish_wrong_state': True, 'cross_step_trajectory_anomaly': True}
