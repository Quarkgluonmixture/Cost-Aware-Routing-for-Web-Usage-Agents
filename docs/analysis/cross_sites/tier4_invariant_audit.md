# Tier 4 Invariant Audit: Runner/Page-State Contradictions

Audit date: 2026-04-30. Scope: the 12 requested Phase 1 VisualWebArena run roots, read with `read_jsonl_dedup`.

## 1. Scope and Method

The audit scanned 4501 deduplicated episode-mode traces and 46921 step records. This is above the prompt estimate because the listed roots currently contain additional completed mode/task traces. No click probe was rerun and no §106 relabeling was performed.

All ten invariants were implementable from recorded fields or explicit derived episode state. `page_changed`, `text_similarity`, `action_success`, `obs_url`, `state_digest`, per-step DOM artifacts, and summary success fields are present. For I8, the runner does not write a literal `truncated_at_max_step` boolean, so the script derives it from `len(steps)` reaching the observed cap, `agent_finished=False`, and `last.done=False`. For URL and AXTree adjacency checks, the script respects the recorder semantics: DOM artifacts are saved before each action, while `state_digest.url_before/url_after` bracket the action.

## 2. Ranked Violation Summary

| Rank | ID | Invariant | Violations | Step % | Dominant site | Dominant mode | Novelty |
|---:|---|---|---:|---:|---|---|---|
| 1 | I6 | `inv_axtree_drift_same_url` | 6002 | 12.7917% | classifieds | DOM | matches Tier 2 cross-step trajectory anomaly already-known family |
| 2 | I7 | `inv_finish_but_eval_reject` | 1552 | 3.3077% | classifieds | DOM | matches Tier 2 finish_wrong_state already-known family |
| 3 | I9 | `inv_element_id_role_drift` | 1127 | 2.4019% | reddit | DOM | mechanism anticipated by Tier 1 AXTree audit; empirical exposed-ID role-drift count is NEW |
| 4 | I4 | `inv_long_step_unexplained` | 828 | 1.7647% | reddit | DOM | partly anticipated by Tier 1 static timeout concerns; empirical long-step count is NEW |
| 5 | I3 | `inv_repeat_click_no_cycle_break` | 481 | 1.0251% | reddit | DOM | matches click-probe and phantom-paper click-loop already-known family |
| 6 | I10 | `inv_state_change_but_obs_same` | 288 | 0.6138% | reddit | Vision | NEW if nonzero; direct logger consistency check not covered by earlier tiers |
| 7 | I8 | `inv_max_step_truncate_at_click` | 201 | 0.4284% | reddit | DOM | NEW count relative to Tier 1/Tier 2/probe |
| 8 | I2 | `inv_action_fail_but_page_changed` | 25 | 0.0533% | reddit | Vision | NEW finding relative to Tier 1/Tier 2/probe counts |
| 9 | I1 | `inv_action_success_but_no_change` | 0 | 0.0000% | - | - | matches Tier 2 type/scroll silent-failure catalog and click probe no-progress family |
| 10 | I5 | `inv_unexplained_url_jump` | 0 | 0.0000% | - | - | NEW if nonzero; Tier 2 only covered same-URL drift, not between-record URL discontinuity |

The largest family is I6, the same-URL AXTree drift invariant. It is not a new mechanism: Tier 2 already found a cross-step trajectory anomaly family, and this stricter invariant confirms that the behavior is broad. I7 is likewise expected from Tier 2's finish-wrong-state catalog. The higher-value Tier 4 additions are the invariants that expose contradictions in runner bookkeeping rather than only policy failure: I2, I4, I8, I9, and I10.

## 3. Per-Invariant Findings

### I6 `inv_axtree_drift_same_url`

Violations: 6002 (12.7917% of steps; 38.3026% of episode-mode traces). Site breakdown: {'classifieds': 2946, 'reddit': 1938, 'shopping': 1118}. Mode breakdown: {'DOM': 2885, 'P-prompt': 358, 'P-text': 249, 'SoM': 1128, 'Vision': 1382}.

Interpretation: AJAX render, overlay/modal replacement, in-place search update, or observation-cache replacement not represented as navigation. Taxonomy match: Type 4 Evaluator State Drift.

Case study: classifieds task 0 step 2 (B0_3mode_classifieds, DOM): action=scroll; url_before=http://100.95.81.103:9980/index.php?page=search&sPattern=blue+kayak+&sOrder=i_price&iOrderType=asc; url_after=http://100.95.81.103:9980/index.php?page=search&sPattern=blue+kayak+&sOrder=i_price&iOrderType=asc; page_changed=True; action_success=True; text_similarity=0.088; adjacent_obs_shingle_similarity=0.037

Novelty assessment: matches Tier 2 cross-step trajectory anomaly already-known family.

### I7 `inv_finish_but_eval_reject`

Violations: 1552 (3.3077% of steps; 34.4812% of episode-mode traces). Site breakdown: {'classifieds': 774, 'reddit': 609, 'shopping': 169}. Mode breakdown: {'DOM': 474, 'P-SoM': 215, 'P-prompt': 80, 'P-text': 194, 'SoM': 353, 'Vision': 236}.

Interpretation: Agent terminates from an answer-shaped or partial state that does not satisfy evaluator URL/database/content checks. Taxonomy match: Type 4 Evaluator State Drift and false negatives/false positives.

Case study: classifieds task 100 step 3 (B0_3mode_classifieds, DOM): action=finish; answer=18000.00; url_before=http://100.95.81.103:9980/index.php?page=item&id=23386; url_after=http://100.95.81.103:9980/index.php?page=item&id=23386; page_changed=False; action_success=True; text_similarity=1.000; finish_attempted=True; raw_success=False

Novelty assessment: matches Tier 2 finish_wrong_state already-known family.

### I9 `inv_element_id_role_drift`

Violations: 1127 (2.4019% of steps; 2.9549% of episode-mode traces). Site breakdown: {'classifieds': 84, 'reddit': 663, 'shopping': 380}. Mode breakdown: {'DOM': 813, 'P-prompt': 192, 'SoM': 29, 'Vision': 93}.

Interpretation: Observation-local AX node IDs are reused across rerenders; treating element_id as stable across history risks stale-cache or wrong-role grounding. Taxonomy match: Type 1 Coordinate Dispatch Anomaly / Type 4 state drift.

Case study: classifieds task 207 step 1 (B0_3mode_classifieds, DOM): action=type; element_id=140; text=headphones; url_before=http://100.95.81.103:9980/; url_after=http://100.95.81.103:9980/index.php?page=search&sPattern=headphones+; page_changed=True; action_success=True; text_similarity=0.240; element_id=7; previous_role=button; previous_step=0; current_role=link

Novelty assessment: mechanism anticipated by Tier 1 AXTree audit; empirical exposed-ID role-drift count is NEW.

### I4 `inv_long_step_unexplained`

Violations: 828 (1.7647% of steps; 5.4432% of episode-mode traces). Site breakdown: {'classifieds': 7, 'reddit': 821}. Mode breakdown: {'DOM': 275, 'P-SoM': 70, 'P-prompt': 107, 'P-text': 116, 'SoM': 200, 'Vision': 60}.

Interpretation: Hidden Playwright timeout or slow actionability wait is collapsed into a generic step result rather than exposed as an environment error. Taxonomy match: Type 5 Actionability Check Masking and Timeout Swallowing.

Case study: classifieds task 232 step 0 (B0_3mode_classifieds, SoM): action=tab_focus; url_before=http://100.95.81.103:9999/f/consoles/124577/name-a-better-console-you-can-t; url_after=http://100.95.81.103:9999/f/consoles/124577/name-a-better-console-you-can-t; page_changed=True; action_success=True; text_similarity=1.000; env_step_ms=41041

Novelty assessment: partly anticipated by Tier 1 static timeout concerns; empirical long-step count is NEW.

### I3 `inv_repeat_click_no_cycle_break`

Violations: 481 (1.0251% of steps; 10.6865% of episode-mode traces). Site breakdown: {'classifieds': 181, 'reddit': 199, 'shopping': 101}. Mode breakdown: {'DOM': 194, 'P-SoM': 101, 'P-prompt': 20, 'P-text': 42, 'SoM': 124}.

Interpretation: Cycle detection did not halt a same-target click loop soon enough, or repeated clicks were treated as exploration despite no new target. Taxonomy match: Type 1 Coordinate Dispatch Anomaly, surfacing as a loop.

Case study: classifieds task 117 step 3 (B0_3mode_classifieds, DOM): action=click; element_id=2599; url_before=http://100.95.81.103:9980/index.php?page=search&sPattern=blue+bike+&sCategory=7&sOrder=i_price&iOrderType=asc; url_after=http://100.95.81.103:9980/index.php?page=search&sPattern=blue+bike+&sCategory=7&sOrder=i_price&iOrderType=asc; page_changed=False; action_success=False; text_similarity=1.000; repeat_click_element_id=2599; repeats=3

Novelty assessment: matches click-probe and phantom-paper click-loop already-known family.

### I10 `inv_state_change_but_obs_same`

Violations: 288 (0.6138% of steps; 5.3988% of episode-mode traces). Site breakdown: {'classifieds': 61, 'reddit': 218, 'shopping': 9}. Mode breakdown: {'DOM': 40, 'P-prompt': 4, 'P-text': 1, 'SoM': 58, 'Vision': 185}.

Interpretation: Step state-change bookkeeping and persisted observations disagree; likely logger/state digest desynchronization if nonzero. Taxonomy match: Type 4 Evaluator State Drift.

Case study: classifieds task 150 step 0 (B0_3mode_classifieds, DOM): action=tab_focus; url_before=http://100.95.81.103:9980/index.php?page=search&sCategory=16&sOrder=i_price&iOrderType=asc&iPage=331&sShowAs=gallery; url_after=http://100.95.81.103:9980/index.php?page=search&sCategory=16&sOrder=i_price&iOrderType=asc&iPage=331&sShowAs=gallery; page_changed=True; action_success=True; text_similarity=1.000; normalized_prev_and_next_obs_text_identical=True

Novelty assessment: NEW if nonzero; direct logger consistency check not covered by earlier tiers.

### I8 `inv_max_step_truncate_at_click`

Violations: 201 (0.4284% of steps; 4.4657% of episode-mode traces). Site breakdown: {'classifieds': 91, 'reddit': 96, 'shopping': 14}. Mode breakdown: {'DOM': 94, 'P-SoM': 32, 'P-prompt': 9, 'P-text': 20, 'SoM': 41, 'Vision': 5}.

Interpretation: Max-iteration masking hides the final failed click or unresolved click loop as a generic truncation. Taxonomy match: Type 5 Actionability Check Masking and Timeout Swallowing.

Case study: classifieds task 115 step 29 (B0_3mode_classifieds, DOM): action=click; element_id=21394; url_before=http://100.95.81.103:9980/index.php?page=search&sPattern=foot+brace+; url_after=http://100.95.81.103:9980/index.php?page=item&id=15342; page_changed=True; action_success=True; text_similarity=0.173; derived_truncated_at_max_step=True; max_steps_cap=30

Novelty assessment: NEW count relative to Tier 1/Tier 2/probe: explicit max-step-at-click masking slice.

### I2 `inv_action_fail_but_page_changed`

Violations: 25 (0.0533% of steps; 0.2666% of episode-mode traces). Site breakdown: {'classifieds': 6, 'reddit': 19}. Mode breakdown: {'P-SoM': 2, 'Vision': 23}.

Interpretation: Runner success flag is stricter or stale relative to actual state transition; this can cause unnecessary retries or wrong self-diagnosis. Taxonomy match: Type 4 Evaluator State Drift / Type 5 actionability masking.

Case study: classifieds task 226 step 7 (B0_phantom_classifieds, P-SoM): action=back; url_before=about:blank; url_after=about:blank; page_changed=True; action_success=False; text_similarity=1.000

Novelty assessment: NEW finding relative to Tier 1/Tier 2/probe counts: empirical runner-false-negative success flag.

### I1 `inv_action_success_but_no_change`

Violations: 0 (0.0000% of steps; 0.0000% of episode-mode traces). Site breakdown: {}. Mode breakdown: {}.

Interpretation: Runner accepted a no-op or swallowed an actionability failure; policy sees no state progress after a supposedly successful action. Taxonomy match: Type 5 Actionability Check Masking and Timeout Swallowing.

Case study: no violations were observed, so no task-level example is available.

Novelty assessment: matches Tier 2 type/scroll silent-failure catalog and click probe no-progress family.

### I5 `inv_unexplained_url_jump`

Violations: 0 (0.0000% of steps; 0.0000% of episode-mode traces). Site breakdown: {}. Mode breakdown: {}.

Interpretation: Popup, redirect, tab/frame switch, recovery path, or logger/environment state drift between recorded steps. Taxonomy match: Type 4 Evaluator State Drift.

Case study: no violations were observed, so no task-level example is available.

Novelty assessment: NEW if nonzero; Tier 2 only covered same-URL drift, not between-record URL discontinuity.

## 4. Section 4 Wiring

Use Tier 4 as the adversarial consistency layer on top of the earlier evidence. Tier 1 says which implementation surfaces are suspicious, Tier 2 mines action-specific silent-failure signatures, the click probe validates the §106 click-center mechanism, and Tier 4 asks whether the trace logs contradict themselves under simple invariants. The paper-ready claim should be framed as: state/action inconsistency is measurable without hand-authored bug signatures, and the largest contradictions concentrate in the same families predicted by the taxonomy.

For the main Section 4 table, report I6 and I7 as confirmation rows, then foreground the Tier 4-specific rows. I2 is the cleanest runner false-negative flag: the action is marked failed although page-state evidence changed. I4 is the timeout-swallowing row: long non-wait/non-type/non-scroll actions are likely hidden Playwright actionability waits. I8 isolates max-iteration masking at a terminal click, and I9 provides empirical support for the Tier 1 warning that AX element IDs are observation-local rather than stable cross-step object identities. I10 is a logger consistency guard; if nonzero, it should be described as direct state-change/observation desynchronization rather than an agent-policy failure.

## 5. New Findings and Follow-Up

New or newly quantified findings not covered as counts by Tier 1/Tier 2/probe: I9 (1127), I4 (828), I10 (288), I8 (201), I2 (25).

Recommended follow-up is surgical, not another broad probe: sample the top I2/I4/I8/I9 cases, inspect the raw environment logs around those steps, and decide which rows deserve a small manual adjudication table in the appendix. Do not merge I7 into action-dispatch evidence; it is endpoint evaluator drift. Do not overclaim I6 as a bug by itself; in-place AJAX updates are legitimate, but they are a benchmark-control risk when the runner does not label the transition clearly.

## Self-Check

- `implementable_invariants`: 10
- `at_least_7_of_10_implementable`: True
- `case_study_count_at_least_3_unless_zero`: {'I6': True, 'I7': True, 'I9': True, 'I4': True, 'I3': True, 'I10': True, 'I8': True, 'I2': True, 'I1': True, 'I5': True}
- `all_case_study_requirements_met`: True
- `invariants_sorted_by_violation_count`: True
- `max_steps_cap_used_for_i8`: 30
