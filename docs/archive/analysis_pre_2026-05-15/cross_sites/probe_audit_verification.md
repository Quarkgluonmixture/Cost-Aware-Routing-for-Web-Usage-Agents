# Probe Audit Verification

Audit date: 2026-04-30

This replay pass treats Tier 1/2/4/5 outputs as candidate signatures, not fix scope. It used Playwright with the recorded `.auth` storage state, task `start_url`, logged prior actions where feasible, and live DOM/scroll/form snapshots at the target step. No framework code was changed.

Sampling note: Hard cap enforced at 50 total probes: A=15, B=10, C=8, D=7, E=5, F=5. D and E are below the requested 10/6 because the requested per-category counts sum to 54, conflicting with the explicit do-not-sample-more-than-50 constraint. For Tier 2, the catalog exposes `case_study_task_ids` rather than a full `case_study_examples` array, so the sampler re-ran the same catalog predicates over the same 12 run roots and then diversified by site/mode.

## Confusion Matrix

| Category | Claimed N | Probed | Replay fail | Replay-backed scaffold fraction | Extrapolated blast radius | Breakdown |
|---|---:|---:|---:|---:|---:|---|
| `type_silent_failure` | 549 | 15 | 1 | 1.0 | 12.22% / 549 ep | `{'SCAFFOLD_TYPE_BUG': 14, 'REPLAY_FAIL': 1}` |
| `scroll_silent_failure` | 667 | 10 | 0 | 0.3 | 4.46% / 200 ep | `{'LEGIT_SCROLL_AT_BOTTOM': 7, 'SCAFFOLD_SCROLL_BUG': 3}` |
| `select_option_silent_failure` | 149 | 8 | 1 | 0.286 | 0.95% / 43 ep | `{'AGENT_BAD_OPTION_OR_STALE_TARGET': 3, 'OTHER': 2, 'REPLAY_FAIL': 1, 'SCAFFOLD_SELECT_ARG_DROP': 2}` |
| `i9_element_id_role_drift` | 1127 | 7 | 1 | 0.833 | 939 | `{'STALE_NODEID_REUSE': 5, 'SAME_ELEMENT_AXTree_RESHAPE': 1, 'REPLAY_FAIL': 1}` |
| `i10_state_change_obs_same` | 288 | 5 | 0 | 0.8 | 230 | `{'LOGGER_BUG': 3, 'OBS_CACHE_BUG': 1, 'INVISIBLE_CHANGE': 1}` |
| `i2_action_fail_page_changed` | 25 | 5 | 3 | 0.0 | 0 | `{'REPLAY_FAIL': 3, 'REPLAY_DID_NOT_CHANGE': 2}` |

## Category Notes

### `type_silent_failure`

The TYPE probe directly compared original center-click plus `Meta+A`/Backspace/type against a DOM locator fill and a JS focus/value/input path. Scaffold bugs were cases where the logged center hit a non-editable or stale coordinate while locator/JS could place text into a nearby intended editable field. A key limitation is platform behavior: on this Linux replay host, `Meta+A` does not consistently select the full page the way `Control+A` would, so the destructive full-page-select symptom is inferred from non-editable focus rather than always reproduced visually.

Case studies:
- `SCAFFOLD_TYPE_BUG`: classifieds task 0 step 5 (B0_3mode_classifieds, DOM). target=section#; ax=[2331] RootWebArea 'blue kayak - Classifieds' focused: True url: http://100.95.81.103:9980/index.php?page=search&sPattern=blue+kayak+&sOrder=i_price&iOrderType=asc
- `REPLAY_FAIL`: shopping task 107 step 4 (B0_dom_shopping, DOM). results/visualwebarena/phase1/B0_dom_shopping_20260428/phase1_dom_router_0/episodes/shopping_task_107_steps_v2.jsonl

### `scroll_silent_failure`

The SCROLL probe used `window.scrollBy(0, window.innerHeight)` at the target pre-state. Cases where the page was already at the bottom or had no overflow were classified as legitimate no-ops rather than bugs.

Case studies:
- `LEGIT_SCROLL_AT_BOTTOM`: classifieds task 107 step 8 (B0_3mode_classifieds, DOM). scroll 1541/2261 -> 1541
- `SCAFFOLD_SCROLL_BUG`: classifieds task 0 step 6 (B0_3mode_classifieds, Vision). scroll 576/2727 -> 1296

### `select_option_silent_failure`

The SELECT probe separated the static VWA `locator.select_option()` arg-drop path from custom dropdown and stale-target cases. Many logged actions target custom dropdown text rather than native `<select>`, so they are not evidence for the specific arg-drop bug even when JS can change the UI.

Case studies:
- `AGENT_BAD_OPTION_OR_STALE_TARGET`: classifieds task 114 step 14 (B0_3mode_classifieds, DOM). target=div#content; ax=[8731] StaticText 'Newly listed '
- `OTHER`: classifieds task 106 step 1 (B0_phantom_text_classifieds, P-text). target=select#sCategory; ax=
- `REPLAY_FAIL`: reddit task 187 step 9 (B1_3mode_reddit, DOM). results/visualwebarena/phase1/B1_3mode_reddit_20260413/phase1_dom_router_0/episodes/reddit_task_187_steps_v2.jsonl
- `SCAFFOLD_SELECT_ARG_DROP`: reddit task 169 step 1 (B0_3mode_reddit, SoM). target=span#; ax=[3630] combobox 'pics' hasPopup: menu required: False expanded: False

### `i9_element_id_role_drift`

I9 was verified against adjacent persisted AXTree artifacts and a live replay to the current step. Because arbitrary non-action AX ids do not have persisted union bounds, classification relies on the previous/current AX lines: same name with role change is reshaping; different role/name under the same id is treated as observation-local node-id reuse.

Case studies:
- `STALE_NODEID_REUSE`: classifieds task 207 step 1 (B0_3mode_classifieds, DOM). [7] button 'Search' disabled: True => [7] link 'Logout' url: http://100.95.81.103:9980/index.php?page=main&action=logout
- `SAME_ELEMENT_AXTree_RESHAPE`: classifieds task 229 step 1 (B0_3mode_classifieds, SoM). [99] image 'Classifieds' url: http://100.95.81.103:9980/oc-content/uploads/sigma_logo.png => [99] link 'Classifieds' url: http://100.95.81.103:9980/
- `REPLAY_FAIL`: reddit task 164 step 8 (B0_3mode_reddit, SoM). results/visualwebarena/phase1/B0_3mode_reddit_20260422/phase1_som_router_0/episodes/reddit_task_164_steps_v2.jsonl

### `i10_state_change_obs_same`

I10 replay executed the target action and compared live URL, title, body text, scroll, and form fields. No visible delta means a logger consistency bug; visible/form delta with identical persisted next observation means an observation cache bug.

Case studies:
- `LOGGER_BUG`: classifieds task 150 step 0 (B0_3mode_classifieds, DOM). delta={'url_changed': False, 'title_changed': False, 'body_changed': False, 'scroll_changed': False, 'fields_changed': False}; http://100.95.81.103:9980/index.php?page=search&sCategory=16&sOrder=i_price&iOrderType=asc&iPage=331&sShowAs=gallery -> http://100.95.81.103:9980/index.php?page=search&sCategory=16&sOrder=i_price&iOrderType=asc&iPage=331&sShowAs=gallery
- `OBS_CACHE_BUG`: classifieds task 106 step 0 (B0_3mode_classifieds, Vision). delta={'url_changed': False, 'title_changed': False, 'body_changed': False, 'scroll_changed': False, 'fields_changed': True}; http://100.95.81.103:9980/ -> http://100.95.81.103:9980/
- `INVISIBLE_CHANGE`: reddit task 143 step 5 (B0_3mode_reddit, DOM). delta={'url_changed': False, 'title_changed': False, 'body_changed': False, 'scroll_changed': True, 'fields_changed': False}; http://100.95.81.103:9999/submission_images/4deba2177751958fb88389e890368598158583571f6c619772e70b3f5c63ed6f.jpg -> http://100.95.81.103:9999/submission_images/4deba2177751958fb88389e890368598158583571f6c619772e70b3f5c63ed6f.jpg

### `i2_action_fail_page_changed`

I2 replay executed the target action despite the logged `action_success=False`. Live state deltas identify runner false negatives; unchanged replays are kept separate because several cases depend on volatile back-stack state.

Case studies:
- `REPLAY_FAIL`: classifieds task 226 step 7 (B0_phantom_som_classifieds, P-SoM). results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426/phase1_phantom_som_router_0/episodes/classifieds_task_226_steps_v2.jsonl
- `REPLAY_DID_NOT_CHANGE`: classifieds task 231 step 1 (B1_3mode_classifieds, Vision). delta={'url_changed': False, 'title_changed': False, 'body_changed': False, 'scroll_changed': False, 'fields_changed': False}; about:blank -> about:blank

## Blast-Radius Correction

`tier_audit_overestimate_factor`: type_silent_failure 1.0x; scroll_silent_failure 3.3x; select_option_silent_failure 3.5x; i9_element_id_role_drift 1.2x; i10_state_change_obs_same 1.2x; i2_action_fail_page_changed inf

Replay-backed true paper-relevant bugs:
- `type_silent_failure`: fraction 1.0, extrapolated 549
- `scroll_silent_failure`: fraction 0.3, extrapolated 200
- `select_option_silent_failure`: fraction 0.286, extrapolated 43
- `i9_element_id_role_drift`: fraction 0.833, extrapolated 939
- `i10_state_change_obs_same`: fraction 0.8, extrapolated 230

Fix-scope recommendation:
- Prioritize type_silent_failure: replay-backed scaffold fraction 1.0.
- Keep select_option_silent_failure in appendix/follow-up scope: replay-backed fraction 0.286, but audit overstates it.
- Keep scroll_silent_failure in appendix/follow-up scope: replay-backed fraction 0.3, but audit overstates it.
- Prioritize i9_element_id_role_drift: replay-backed scaffold fraction 0.833.
- Prioritize i10_state_change_obs_same: replay-backed scaffold fraction 0.8.
- Do not put i2_action_fail_page_changed in immediate fix scope from this sample; no replay-backed scaffold cases.

## Self-Check

- `categories_probed`: ['type_silent_failure', 'scroll_silent_failure', 'select_option_silent_failure', 'i9_element_id_role_drift', 'i10_state_change_obs_same', 'i2_action_fail_page_changed']
- `all_categories_at_least_5_cases`: True
- `total_cases_probed`: 50
