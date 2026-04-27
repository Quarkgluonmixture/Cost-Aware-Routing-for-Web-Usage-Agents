# Disagreement Task Cluster Analysis (B0 cls + red)

This analysis uses the `diag` workflow: run P1-P14 hard-rule matching, inspect the per-step action sequence, then cluster root causes into mechanism-level categories for Section 5. Data access was read-only; no shopping data, auth state, live experiments, or running processes were touched.

## Overview

- Total one-arm-only disagreement tasks analyzed: **54** (`classifieds` 36, `reddit` 18).
- Total `(task, mode)` pairs considered: **216** = 54 success-side pairs + 162 failure-side pairs.
- Failure-side pairs with step traces: **117 / 162**. The missing 45 are all Phantom-SoM pairs from cleared runs; they are counted from summaries but not per-step diagnosed.
- Modes covered: DOM, SoM, Vision, Phantom-SoM.
- Cluster categories identified: **9** observed categories.

Exclusive success sets used for the analysis:

| Site | DOM only | SoM only | Vision only | Phantom-SoM only |
|---|---:|---:|---:|---:|
| classifieds | 5 | 18 | 9 | 4 |
| reddit | 3 | 6 | 4 | 5 |

§103 consistency check: the exclusive sets derived from local summaries match the provided §103 lists exactly, including the complete classifieds SoM-only set `[14, 49, 52, 101, 106, 111, 115, 120, 127, 130, 132, 149, 160, 165, 166, 187, 209, 210]`.

## Failure Pattern Categories

### trace-unavailable

**Definition.** Phantom-SoM summary exists but steps/artifacts were cleared; only success/failure set membership can be used until the chain re-run restores traces.

**Prevalence.** 45/162 failure pairs; by mode: Phantom-SoM: 45.

**Representative diag excerpts.**
- `classifieds task 98 Phantom-SoM` (winner: DOM): summary-only: step trace unavailable after run clear; counted from adjusted_success summary.. Intent: How many hours are on the engine of the most recently listed red boat?
- `classifieds task 167 Phantom-SoM` (winner: DOM): summary-only: step trace unavailable after run clear; counted from adjusted_success summary.. Intent: Navigate to the listing on this page whose image includes an instrument the same color as the item in the image provided.

### visual-missing

**Definition.** DOM cannot observe necessary image/color/screen information. P6 is the dominant diagnostic; this is an observation limitation, not a scaffold bug.

**Prevalence.** 34/162 failure pairs; by mode: DOM: 34.

**Representative diag excerpts.**
- `classifieds task 60 DOM` (winner: Phantom-SoM): P6,P14; 3 steps; actions select_option→select_option→select_option; final_url=http://100.95.81.103:9980/; reason=fail_incomplete_or_stuck. Intent: Find the most expensive video game item where I can roleplay the situation in the image.
- `classifieds task 49 DOM` (winner: SoM): P6; 4 steps; actions type→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=678; reason=fail_finish_eval_mismatch. Intent: How much RAM (in GB) does the item with blue LED lights on this page have?

### early-finish/wrong-commit

**Definition.** The agent terminates or answers from insufficient evidence, producing wrong-url, eval-mismatch, claim-missing, or empty-answer outcomes.

**Prevalence.** 33/162 failure pairs; by mode: DOM: 6, SoM: 16, Vision: 11.

**Representative diag excerpts.**
- `reddit task 69 SoM` (winner: DOM): no P-rule; 4 steps; actions scroll→type→click→finish; final_url=http://100.95.81.103:9999/f/newhampshire/129011/new-hampshire-be-like/; reason=fail_finish_eval_mismatch. Intent: Make a comment in this post explaining what the picture is about.
- `classifieds task 14 DOM` (winner: SoM): no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=60133; reason=fail_early_finish. Intent: What is the email of the seller of the painting in the second row?

### element-misground

**Definition.** Element grounding failure: out-of-viewport target, container/root node, or mislocalized click target (P1/P2/P4).

**Prevalence.** 13/162 failure pairs; by mode: DOM: 1, Vision: 12.

**Representative diag excerpts.**
- `reddit task 81 Vision` (winner: DOM): P1; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9999/f/photoshopbattles; reason=fail_no_progress. Intent: Upvote all the posts that have a picture which contains cat on this page.
- `classifieds task 115 DOM` (winner: SoM): P2,P6,P14; 30 steps; actions type→scroll→scroll→click→back→click→back→click→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=15342; reason=fail_max_steps_target_unreachable. Intent: Search for "brace" and navigate to the most recently listed item that matches the body part in this image.

### click-loop/no-text-grounding

**Definition.** Vision-specific click-loop: the model can see the screen but lacks stable DOM/mark text grounding, so it clicks plausible visual regions and then loops on the same page.

**Prevalence.** 11/162 failure pairs; by mode: Vision: 11.

**Representative diag excerpts.**
- `classifieds task 184 Vision` (winner: DOM): P14; 30 steps; actions type→click→click→scroll→click→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=Charizard+Pok; reason=fail_max_steps. Intent: Navigate to the most expensive item in the "Collectibles" category with the same exact item from the image.
- `classifieds task 52 Vision` (winner: SoM): P14; 5 steps; actions type→click→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=wave+painting; reason=fail_incomplete_or_stuck. Intent: Find me the most recent painting in the "Arts + crafts" category of something that looks close to this.

### search-loop

**Definition.** The agent keeps using search or search-result scanning instead of committing to the relevant page. Operationally: P13 or repeated type/search actions, often with DOM hierarchy inducing continued exploration.

**Prevalence.** 10/162 failure pairs; by mode: DOM: 3, SoM: 3, Vision: 4.

**Representative diag excerpts.**
- `reddit task 14 Vision` (winner: SoM): P13; 12 steps; actions type→type→type→type→type→type→type→type→…; final_url=http://100.95.81.103:9999/search?q=cityscape+at+dusk+with+wet+pavement; reason=fail_incomplete_or_stuck. Intent: Can you help me find and navigate to the comments section of an image post that has something like this image?
- `reddit task 162 DOM` (winner: Phantom-SoM): P14; 30 steps; actions type→click→type→click→type→type→click→click→…; final_url=http://100.95.81.103:9999/; reason=fail_max_steps_search_repeat. Intent: Subscribe to the forum which is most likely to include discussions related to the image.

### visual-hijack/click-loop

**Definition.** SoM-specific click-loop: a marked visual/text element looks actionable, the model follows that mark, then repeats clicks or wrong-page actions. This is the dense-mark version of visual attention hijack.

**Prevalence.** 8/162 failure pairs; by mode: SoM: 8.

**Representative diag excerpts.**
- `reddit task 79 SoM` (winner: DOM): P14; 5 steps; actions type→click→click→click→finish; final_url=http://100.95.81.103:9999/f/television/92954/rick-and-morty-with-the-a; reason=fail_finish_eval_mismatch. Intent: Navigate to the comments section of any post that contains Rick and Morty in the picture.
- `classifieds task 184 SoM` (winner: DOM): P14; 30 steps; actions select_option→type→select_option→scroll→scroll→scroll→scroll→click→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=Charizard+Pok; reason=fail_max_steps. Intent: Navigate to the most expensive item in the "Collectibles" category with the same exact item from the image.

### abandon-after-N

**Definition.** The trajectory exhausts a long budget, usually max steps or repeated scroll/back/search, without converging on a target.

**Prevalence.** 7/162 failure pairs; by mode: DOM: 2, SoM: 2, Vision: 3.

**Representative diag excerpts.**
- `reddit task 179 SoM` (winner: Vision): no P-rule; 30 steps; actions type→type→click→type→type→type→type→type→…; final_url=http://100.95.81.103:9999/search?q=St.+Louis+forum+; reason=fail_max_steps_search_repeat. Intent: Can you take me to the page that shows the most controversial posts of the past month in the forum for a city located in the state pictured in the ima
- `classifieds task 167 SoM` (winner: DOM): no P-rule; 30 steps; actions type→click→back→click→back→click→back→click→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=72933; reason=raw_success_adjusted_false. Intent: Navigate to the listing on this page whose image includes an instrument the same color as the item in the image provided.

### other

**Definition.** No strong P-rule or cluster matched; mostly idiosyncratic failed trajectories.

**Prevalence.** 1/162 failure pairs; by mode: SoM: 1.

**Representative diag excerpts.**
- `reddit task 152 SoM` (winner: Vision): no P-rule; 4 steps; actions scroll→type→type→type; final_url=http://100.95.81.103:9999/f/OldSchoolCool/15059; reason=fail_no_progress. Intent: Leave a comment in this post with the text as the number of adults in the image.

## Per-Mode Failure Distribution Table

Counts are failure-side `(task, mode)` pairs among the 54 one-arm-only disagreement tasks. Percentages are within each failed mode. Phantom-SoM is reported separately as trace-unavailable because the completed runs were cleared. Note the denominator here is not the same as the §103 global behavior metric denominator: in this exclusive-task failure slice, DOM failures skew toward visual-missing because most non-DOM-only tasks are visually grounded; the broader §103 reddit search-loop gradient remains a whole-run trajectory statistic.

| Mode | N fail | search-loop | click-loop | early-finish/wrong-commit | abandon-after-N | hierarchy-confusion | visual-missing | visual-hijack/click-loop | click-loop/no-text-grounding | element-misground | trace-unavailable | other |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DOM | 46 | 3 (6.5%) | - | 6 (13.0%) | 2 (4.3%) | - | 34 (73.9%) | - | - | 1 (2.2%) | - | - |
| SoM | 30 | 3 (10.0%) | - | 16 (53.3%) | 2 (6.7%) | - | - | 8 (26.7%) | - | - | - | 1 (3.3%) |
| Vision | 41 | 4 (9.8%) | - | 11 (26.8%) | 3 (7.3%) | - | - | - | 11 (26.8%) | 12 (29.3%) | - | - |
| Phantom-SoM | 45 | - | - | - | - | - | - | - | - | - | 45 (100.0%) | - |

Site split:

| Site | Mode | search-loop | click-loop family | early-finish | visual-missing | element-misground | trace-unavailable | other |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| classifieds | DOM | 1 | 0 | 5 | 22 | 1 | 0 | 2 |
| classifieds | SoM | 1 | 4 | 12 | 0 | 0 | 0 | 1 |
| classifieds | Vision | 2 | 7 | 7 | 0 | 10 | 0 | 1 |
| classifieds | Phantom-SoM | 0 | 0 | 0 | 0 | 0 | 32 | 0 |
| reddit | DOM | 2 | 0 | 1 | 12 | 0 | 0 | 0 |
| reddit | SoM | 2 | 4 | 4 | 0 | 0 | 0 | 2 |
| reddit | Vision | 2 | 4 | 4 | 0 | 2 | 0 | 2 |
| reddit | Phantom-SoM | 0 | 0 | 0 | 0 | 0 | 13 | 0 |

## Mode-Pair Disagreement Insights

### DOM-only success tasks (8 tasks)

Tasks: reddit:69, reddit:79, reddit:81, classifieds:98, classifieds:167, classifieds:174, classifieds:184, classifieds:222.
Failure modes cluster as: early-finish/wrong-commit=8, trace-unavailable=8, element-misground=3, visual-hijack/click-loop=2, abandon-after-N=1, search-loop=1, click-loop/no-text-grounding=1.
Why DOM succeeded: these are cases where sustained AXTree/hierarchy exploration pays off. The failed visual/mark arms tend to click a plausible result too early or lose grounding after choosing a screen element; reddit tasks 69/79/81 are all page-screen style cases noted in §103.

### SoM-only success tasks (24 tasks)

Tasks: reddit:2, reddit:4, reddit:14, reddit:131, reddit:139, reddit:142, classifieds:14, classifieds:49, classifieds:52, classifieds:101, classifieds:106, classifieds:111, classifieds:115, classifieds:120, classifieds:127, classifieds:130, classifieds:132, classifieds:149, classifieds:160, classifieds:165, classifieds:166, classifieds:187, classifieds:209, classifieds:210.
Failure modes cluster as: trace-unavailable=24, visual-missing=18, early-finish/wrong-commit=11, element-misground=8, click-loop/no-text-grounding=8, abandon-after-N=2, search-loop=1.
Why SoM succeeded: the image plus marks supplies a locator that the text-only or unmarked visual arms lack. Failures in DOM are mostly visual-missing or hierarchy/search problems; failures in Vision often lack stable text grounding.

### Vision-only success tasks (13 tasks)

Tasks: reddit:148, reddit:150, reddit:152, reddit:179, classifieds:11, classifieds:16, classifieds:40, classifieds:61, classifieds:112, classifieds:152, classifieds:192, classifieds:194, classifieds:217.
Failure modes cluster as: trace-unavailable=13, visual-missing=10, early-finish/wrong-commit=8, visual-hijack/click-loop=3, abandon-after-N=2, search-loop=2, other=1.
Why Vision succeeded: the raw screenshot can solve some page-screen tasks without mark occlusion or AXTree over-search. The failed DOM/SoM arms either miss visual facts or over-commit to marked/textual affordances.

### Phantom-SoM-only success tasks (9 tasks)

Tasks: reddit:7, reddit:94, reddit:100, reddit:124, reddit:162, classifieds:60, classifieds:64, classifieds:93, classifieds:201.
Failure modes cluster as: visual-missing=6, early-finish/wrong-commit=6, search-loop=6, visual-hijack/click-loop=3, click-loop/no-text-grounding=2, element-misground=2, abandon-after-N=2.
Why Phantom-SoM succeeded: direct step traces are unavailable, but §103 classifies these as mixed and text-compact quick-decision cases. The counterpart failures are dominated by visual-missing, click-loop, and search-loop patterns, consistent with Phantom opening a distinct low-cost solution basin.

## Implications for Paper Section 5

1. **Representation effect: AXTree versus flat marks changes exploration shape.** DOM failures among non-DOM-only tasks include a visible search/hierarchy component, while SoM/Vision failures more often become click-loop or grounding failures after choosing a plausible visual target. This supports the Section 5 claim that representation changes the default trajectory, not only final accuracy.
2. **Visual channel effect: image access helps, but it also changes failure mode.** DOM has the cleanest `visual-missing` signature; SoM and Vision remove that bottleneck but introduce mark/visual grounding loops. This explains why full SoM is strong on classifieds but not a superset of DOM or Phantom-SoM.
3. **Phantom-SoM remains a distinct routing arm, but current per-step evidence is incomplete.** The exclusive sets match §103, yet Phantom-SoM traces are absent after run clearing. The mechanism evidence for Phantom therefore rests on §103 macro behavior metrics and counterpart-mode diagnostics until the chain re-run restores Phantom JSONL. This is the only material limitation relative to the requested 200-call diag sweep.

## Validation Notes

- `diag_pattern_match.py` was invoked through `.venv/bin/python3` via its Python API for each relevant run/condition; no experiments were started.
- Completed baseline scans: B0 classifieds DOM/SoM/Vision and B0 reddit DOM/SoM/Vision. Phantom-SoM scans returned zero step episodes because only summary backups are present.
- No §103 exclusive-set mismatch was found. The main missing item is not a claim contradiction but missing Phantom-SoM step-level traces.

## Appendix: Compact Pair Diagnostics

| Site | Task | Winner | Failed mode | Category | P-rules | Diag excerpt |
|---|---:|---|---|---|---|---|
| classifieds | 11 | Vision | DOM | visual-missing | P5,P6,P14 | P5,P6,P14; 12 steps; actions scroll→scroll→scroll→scroll→scroll→click→scroll→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=7; reason=fail_no_progress |
| classifieds | 11 | Vision | SoM | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=7; reason=fail_early_finish |
| classifieds | 11 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 14 | SoM | DOM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=60133; reason=fail_early_finish |
| classifieds | 14 | SoM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=4&iPage=2&sS; reason=fail_early_finish |
| classifieds | 14 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 16 | Vision | DOM | abandon-after-N | P5,P14 | P5,P14; 11 steps; actions type→scroll→scroll→scroll→click→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_no_progress |
| classifieds | 16 | Vision | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=45239; reason=fail_early_finish |
| classifieds | 16 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 40 | Vision | DOM | early-finish/wrong-commit | P14 | P14; 4 steps; actions type→click→scroll→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=60649; reason=fail_finish_eval_mismatch |
| classifieds | 40 | Vision | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions type→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=dishwasher+; reason=fail_early_finish |
| classifieds | 40 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 49 | SoM | DOM | visual-missing | P6 | P6; 4 steps; actions type→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=678; reason=fail_finish_eval_mismatch |
| classifieds | 49 | SoM | Vision | element-misground | P1 | P1; 6 steps; actions click→scroll→scroll→click→click→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=14; reason=fail_finish_eval_mismatch |
| classifieds | 49 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 52 | SoM | DOM | visual-missing | P5,P6,P14 | P5,P6,P14; 8 steps; actions select_option→click→scroll→scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=4; reason=fail_no_progress |
| classifieds | 52 | SoM | Vision | click-loop/no-text-grounding | P14 | P14; 5 steps; actions type→click→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=wave+painting; reason=fail_incomplete_or_stuck |
| classifieds | 52 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 60 | Phantom-SoM | DOM | visual-missing | P6,P14 | P6,P14; 3 steps; actions select_option→select_option→select_option; final_url=http://100.95.81.103:9980/; reason=fail_incomplete_or_stuck |
| classifieds | 60 | Phantom-SoM | SoM | early-finish/wrong-commit | - | no P-rule; 5 steps; actions select_option→click→select_option→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=17379; reason=fail_finish_wrong_url_not_found |
| classifieds | 60 | Phantom-SoM | Vision | click-loop/no-text-grounding | P14 | P14; 17 steps; actions type→type→type→type→type→type→type→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_no_progress |
| classifieds | 61 | Vision | DOM | visual-missing | P6 | P6; 30 steps; actions select_option→click→select_option→click→type→click→type→select_option→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=i_price&iOrderT; reason=fail_max_steps_target_unreachable |
| classifieds | 61 | Vision | SoM | search-loop | P13,P14 | P13,P14; 6 steps; actions type→scroll→type→type→type→type; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=video+game+; reason=fail_no_progress |
| classifieds | 61 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 64 | Phantom-SoM | DOM | visual-missing | P5,P6,P13,P14 | P5,P6,P13,P14; 9 steps; actions type→type→scroll→scroll→scroll→scroll→type→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_no_progress |
| classifieds | 64 | Phantom-SoM | SoM | visual-hijack/click-loop | P14 | P14; 30 steps; actions type→type→scroll→scroll→scroll→click→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=44459; reason=fail_max_steps_target_unreachable |
| classifieds | 64 | Phantom-SoM | Vision | search-loop | P5,P13,P14 | P5,P13,P14; 9 steps; actions type→scroll→scroll→scroll→type→type→type→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=driving+video; reason=fail_no_progress |
| classifieds | 93 | Phantom-SoM | DOM | visual-missing | P6,P10 | P6,P10; 6 steps; actions click→type→click→type→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=33622; reason=fail_finish_wrong_url_left_target |
| classifieds | 93 | Phantom-SoM | SoM | early-finish/wrong-commit | P10 | P10; 6 steps; actions click→type→select_option→type→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=33622; reason=fail_finish_wrong_url_left_target |
| classifieds | 93 | Phantom-SoM | Vision | element-misground | P1,P14 | P1,P14; 5 steps; actions type→scroll→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=39763; reason=fail_finish_wrong_url_not_found |
| classifieds | 98 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions type→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=red+boat+; reason=fail_early_finish |
| classifieds | 98 | DOM | Vision | element-misground | P1,P5,P14 | P1,P5,P14; 4 steps; actions type→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=red+boat; reason=fail_no_progress |
| classifieds | 98 | DOM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 101 | SoM | DOM | visual-missing | P6 | P6; 5 steps; actions select_option→click→select_option→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=60318; reason=fail_finish_wrong_url_not_found |
| classifieds | 101 | SoM | Vision | click-loop/no-text-grounding | P14 | P14; 5 steps; actions click→click→click→click→click; final_url=http://100.95.81.103:9980/; reason=fail_no_progress |
| classifieds | 101 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 106 | SoM | DOM | visual-missing | - | no P-rule; 4 steps; actions select_option→type→type→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_finish_eval_mismatch |
| classifieds | 106 | SoM | Vision | element-misground | P1,P5,P12,P14 | P1,P5,P12,P14; 11 steps; actions select_option→select_option→click→select_option→select_option→type→click→type→…; final_url=http://100.95.81.103:9980/; reason=fail_no_progress |
| classifieds | 106 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 111 | SoM | DOM | early-finish/wrong-commit | - | no P-rule; 4 steps; actions type→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=81897; reason=fail_finish_eval_mismatch |
| classifieds | 111 | SoM | Vision | early-finish/wrong-commit | - | no P-rule; 2 steps; actions type→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=hockey; reason=fail_early_finish |
| classifieds | 111 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 112 | Vision | DOM | visual-missing | P5,P14 | P5,P14; 23 steps; actions type→select_option→click→back→click→back→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=basketball+&s; reason=fail_no_progress |
| classifieds | 112 | Vision | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions type→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=basketball+; reason=fail_early_finish |
| classifieds | 112 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 115 | SoM | DOM | element-misground | P2,P6,P14 | P2,P6,P14; 30 steps; actions type→scroll→scroll→click→back→click→back→click→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=15342; reason=fail_max_steps_target_unreachable |
| classifieds | 115 | SoM | Vision | element-misground | P1,P14 | P1,P14; 5 steps; actions type→click→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=brace; reason=fail_no_progress |
| classifieds | 115 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 120 | SoM | DOM | visual-missing | P5,P6 | P5,P6; 5 steps; actions scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=7&sShowAs=ga; reason=fail_no_progress |
| classifieds | 120 | SoM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=7&sShowAs=ga; reason=fail_early_finish |
| classifieds | 120 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 127 | SoM | DOM | abandon-after-N | P10,P14 | P10,P14; 6 steps; actions type→scroll→scroll→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=68187; reason=fail_finish_wrong_url_not_found |
| classifieds | 127 | SoM | Vision | element-misground | P1 | P1; 4 steps; actions type→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=17593; reason=fail_finish_claim_missing |
| classifieds | 127 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 130 | SoM | DOM | visual-missing | - | no P-rule; 30 steps; actions type→scroll→click→back→click→back→click→back→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=58060; reason=fail_max_steps_click_back_loop |
| classifieds | 130 | SoM | Vision | element-misground | P1 | P1; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_no_progress |
| classifieds | 130 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 132 | SoM | DOM | visual-missing | P6 | P6; 30 steps; actions scroll→scroll→click→back→click→back→click→back→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=20660; reason=fail_max_steps_click_back_loop |
| classifieds | 132 | SoM | Vision | element-misground | P1 | P1; 4 steps; actions scroll→click→click→finish; final_url=http://100.95.81.103:9980/oc-content/uploads/21697/21697.png; reason=fail_finish_empty_answer |
| classifieds | 132 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 149 | SoM | DOM | visual-missing | P5 | P5; 5 steps; actions scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=24&sOrder=i_; reason=fail_no_progress |
| classifieds | 149 | SoM | Vision | early-finish/wrong-commit | - | no P-rule; 3 steps; actions scroll→scroll→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=24&sOrder=i_; reason=fail_finish_eval_mismatch |
| classifieds | 149 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 152 | Vision | DOM | visual-missing | P5 | P5; 5 steps; actions scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=22&sShowAs=g; reason=fail_no_progress |
| classifieds | 152 | Vision | SoM | visual-hijack/click-loop | P14 | P14; 18 steps; actions type→scroll→scroll→click→back→click→back→click→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_finish_wrong_url_not_found |
| classifieds | 152 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 160 | SoM | DOM | visual-missing | P14 | P14; 14 steps; actions scroll→scroll→type→scroll→click→type→type→click→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=41121; reason=fail_finish_eval_mismatch |
| classifieds | 160 | SoM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=5&sShowAs=ga; reason=fail_early_finish |
| classifieds | 160 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 165 | SoM | DOM | visual-missing | P6,P13 | P6,P13; 30 steps; actions type→type→type→type→type→type→type→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_max_steps_target_unreachable |
| classifieds | 165 | SoM | Vision | click-loop/no-text-grounding | P5 | P5; 5 steps; actions scroll→scroll→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=10&iPage=8; reason=fail_no_progress |
| classifieds | 165 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 166 | SoM | DOM | visual-missing | P5,P6 | P5,P6; 6 steps; actions scroll→scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=21&iPage=11; reason=fail_no_progress |
| classifieds | 166 | SoM | Vision | click-loop/no-text-grounding | P5 | P5; 4 steps; actions scroll→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=21&iPage=11; reason=fail_no_progress |
| classifieds | 166 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 167 | DOM | SoM | abandon-after-N | - | no P-rule; 30 steps; actions type→click→back→click→back→click→back→click→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=72933; reason=raw_success_adjusted_false |
| classifieds | 167 | DOM | Vision | search-loop | - | no P-rule; 13 steps; actions scroll→scroll→type→type→scroll→click→back→type→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=1465; reason=raw_success_adjusted_false |
| classifieds | 167 | DOM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 174 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions scroll→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=17&sOrder=i_; reason=fail_early_finish |
| classifieds | 174 | DOM | Vision | element-misground | P1,P5 | P1,P5; 4 steps; actions scroll→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=17&sOrder=i_; reason=fail_no_progress |
| classifieds | 174 | DOM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 184 | DOM | SoM | visual-hijack/click-loop | P14 | P14; 30 steps; actions select_option→type→select_option→scroll→scroll→scroll→scroll→click→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=Charizard+Pok; reason=fail_max_steps |
| classifieds | 184 | DOM | Vision | click-loop/no-text-grounding | P14 | P14; 30 steps; actions type→click→click→scroll→click→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=Charizard+Pok; reason=fail_max_steps |
| classifieds | 184 | DOM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 187 | SoM | DOM | visual-missing | - | no P-rule; 3 steps; actions type→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=76299; reason=fail_finish_wrong_url_not_found |
| classifieds | 187 | SoM | Vision | element-misground | P1 | P1; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=9&iPage=6&sS; reason=fail_no_progress |
| classifieds | 187 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 192 | Vision | DOM | visual-missing | P14 | P14; 7 steps; actions type→click→click→click→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=i_price&iOrderT; reason=fail_no_progress |
| classifieds | 192 | Vision | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=57831; reason=fail_early_finish |
| classifieds | 192 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 194 | Vision | DOM | visual-missing | - | no P-rule; 4 steps; actions type→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=43845; reason=fail_finish_eval_mismatch |
| classifieds | 194 | Vision | SoM | visual-hijack/click-loop | P14 | P14; 8 steps; actions type→click→type→type→scroll→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=80994; reason=fail_finish_eval_mismatch |
| classifieds | 194 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 201 | Phantom-SoM | DOM | visual-missing | P5,P6,P14 | P5,P6,P14; 12 steps; actions type→scroll→scroll→scroll→scroll→click→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=snare+drum+bl; reason=fail_no_progress |
| classifieds | 201 | Phantom-SoM | SoM | early-finish/wrong-commit | - | no P-rule; 3 steps; actions type→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=31068; reason=fail_finish_wrong_url_not_found |
| classifieds | 201 | Phantom-SoM | Vision | abandon-after-N | P14 | P14; 6 steps; actions type→scroll→scroll→type→scroll→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=snare+drum+bl; reason=fail_finish_wrong_url_not_found |
| classifieds | 209 | SoM | DOM | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=27156; reason=fail_early_finish |
| classifieds | 209 | SoM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=27156; reason=fail_early_finish |
| classifieds | 209 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 210 | SoM | DOM | early-finish/wrong-commit | - | no P-rule; 5 steps; actions select_option→type→select_option→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=32759; reason=fail_finish_wrong_url_not_found |
| classifieds | 210 | SoM | Vision | click-loop/no-text-grounding | P14 | P14; 4 steps; actions type→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=lamb; reason=fail_incomplete_or_stuck |
| classifieds | 210 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 217 | Vision | DOM | search-loop | P5,P14 | P5,P14; 18 steps; actions type→type→type→scroll→scroll→click→back→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=27617; reason=fail_no_progress |
| classifieds | 217 | Vision | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions type→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=Captain%27s+L; reason=fail_early_finish |
| classifieds | 217 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| classifieds | 222 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=34501; reason=fail_early_finish |
| classifieds | 222 | DOM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=34501; reason=fail_early_finish |
| classifieds | 222 | DOM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 2 | SoM | DOM | visual-missing | P6 | P6; 3 steps; actions type→click→finish; final_url=http://100.95.81.103:9999/f/washingtondc/136669/fbi-police-citing-driv; reason=fail_finish_wrong_url_not_found |
| reddit | 2 | SoM | Vision | click-loop/no-text-grounding | P14 | P14; 5 steps; actions click→click→click→click→click; final_url=https://deadline.com/2023/01/cindy-williams-dead-laverne-and-shirley-s; reason=fail_incomplete_or_stuck |
| reddit | 2 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 4 | SoM | DOM | visual-missing | P5,P6,P14 | P5,P6,P14; 21 steps; actions type→type→type→type→type→type→type→type→…; final_url=http://100.95.81.103:9999/f/OldSchoolCool/121626/in-1982-agnes-denes-c; reason=fail_no_progress |
| reddit | 4 | SoM | Vision | click-loop/no-text-grounding | P5,P14 | P5,P14; 12 steps; actions click→type→scroll→scroll→click→click→scroll→click→…; final_url=http://100.95.81.103:9999/search?q=wheat+field+woman; reason=fail_no_progress |
| reddit | 4 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 7 | Phantom-SoM | DOM | visual-missing | P6 | P6; 30 steps; actions type→click→type→click→type→click→type→click→…; final_url=http://100.95.81.103:9999/search?q=cranberry+rosemary+cake+recipe+; reason=raw_success_adjusted_false |
| reddit | 7 | Phantom-SoM | SoM | visual-hijack/click-loop | P14 | P14; 11 steps; actions click→scroll→scroll→click→back→click→click→click→…; final_url=http://100.95.81.103:9999/f/food/18811/homemade-peanut-butter-chocolat; reason=raw_success_adjusted_false |
| reddit | 7 | Phantom-SoM | Vision | element-misground | P1,P5,P14 | P1,P5,P14; 8 steps; actions type→click→click→scroll→click→click→click→click; final_url=http://100.95.81.103:9999/search?q=recipe+post+by+OP; reason=raw_success_adjusted_false |
| reddit | 14 | SoM | DOM | visual-missing | P6,P14 | P6,P14; 7 steps; actions type→click→click→click→click→click→click; final_url=http://100.95.81.103:9999/f/nyc/66043/development-v-historical-preserv; reason=fail_incomplete_or_stuck |
| reddit | 14 | SoM | Vision | search-loop | P13 | P13; 12 steps; actions type→type→type→type→type→type→type→type→…; final_url=http://100.95.81.103:9999/search?q=cityscape+at+dusk+with+wet+pavement; reason=fail_incomplete_or_stuck |
| reddit | 14 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 69 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 4 steps; actions scroll→type→click→finish; final_url=http://100.95.81.103:9999/f/newhampshire/129011/new-hampshire-be-like/; reason=fail_finish_eval_mismatch |
| reddit | 69 | DOM | Vision | early-finish/wrong-commit | - | no P-rule; 5 steps; actions scroll→click→type→click→finish; final_url=http://100.95.81.103:9999/f/newhampshire/129011/new-hampshire-be-like/; reason=fail_finish_eval_mismatch |
| reddit | 69 | DOM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 79 | DOM | SoM | visual-hijack/click-loop | P14 | P14; 5 steps; actions type→click→click→click→finish; final_url=http://100.95.81.103:9999/f/television/92954/rick-and-morty-with-the-a; reason=fail_finish_eval_mismatch |
| reddit | 79 | DOM | Vision | early-finish/wrong-commit | P14 | P14; 4 steps; actions type→click→scroll→finish; final_url=http://100.95.81.103:9999/f/television/92954/rick-and-morty-with-the-a; reason=fail_finish_empty_answer |
| reddit | 79 | DOM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 81 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 3 steps; actions click→click→finish; final_url=http://100.95.81.103:9999/f/photoshopbattles; reason=fail_finish_eval_mismatch |
| reddit | 81 | DOM | Vision | element-misground | P1 | P1; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9999/f/photoshopbattles; reason=fail_no_progress |
| reddit | 81 | DOM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 94 | Phantom-SoM | DOM | early-finish/wrong-commit | - | no P-rule; 4 steps; actions type→click→click→finish; final_url=http://100.95.81.103:9999/submission_images/e253bf35b1027ae2cb2664d5e1; reason=fail_finish_eval_mismatch |
| reddit | 94 | Phantom-SoM | SoM | early-finish/wrong-commit | - | no P-rule; 4 steps; actions click→type→click→finish; final_url=http://100.95.81.103:9999/submission_images/6a1e28e0f5710c55d1b653fd7f; reason=fail_finish_eval_mismatch |
| reddit | 94 | Phantom-SoM | Vision | early-finish/wrong-commit | - | no P-rule; 2 steps; actions type→finish; final_url=http://100.95.81.103:9999/search?q=f%2FEarthPorn; reason=fail_early_finish |
| reddit | 100 | Phantom-SoM | DOM | visual-missing | P14 | P14; 22 steps; actions type→click→type→click→type→click→type→click→…; final_url=http://100.95.81.103:9999/f/pics/110740; reason=fail_finish_eval_mismatch |
| reddit | 100 | Phantom-SoM | SoM | visual-hijack/click-loop | P5,P14 | P5,P14; 25 steps; actions click→type→click→back→click→back→click→back→…; final_url=http://100.95.81.103:9999/f/funny; reason=fail_no_progress |
| reddit | 100 | Phantom-SoM | Vision | abandon-after-N | P14 | P14; 6 steps; actions type→scroll→scroll→click→scroll→finish; final_url=http://100.95.81.103:9999/f/tifu/135257/tifu-with-lemon-pound-cake; reason=fail_finish_empty_answer |
| reddit | 124 | Phantom-SoM | DOM | search-loop | - | no P-rule; 14 steps; actions type→type→click→type→type→type→type→type→…; final_url=http://100.95.81.103:9999/submission_images/5b47b170a63d6de812d2ca101b; reason=fail_finish_eval_mismatch |
| reddit | 124 | Phantom-SoM | SoM | search-loop | - | no P-rule; 4 steps; actions type→type→type→finish; final_url=http://100.95.81.103:9999/search?q=Microsoft+revenue+1980s+; reason=fail_finish_eval_mismatch |
| reddit | 124 | Phantom-SoM | Vision | click-loop/no-text-grounding | P5,P14 | P5,P14; 12 steps; actions type→type→click→type→click→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9999/forums; reason=fail_no_progress |
| reddit | 131 | SoM | DOM | visual-missing | P6,P14 | P6,P14; 4 steps; actions type→click→click→finish; final_url=http://100.95.81.103:9999/f/dataisbeautiful; reason=fail_finish_eval_mismatch |
| reddit | 131 | SoM | Vision | click-loop/no-text-grounding | P5,P14 | P5,P14; 5 steps; actions type→click→click→click→click; final_url=http://100.95.81.103:9999/f/IAmA/119736/i-am-mark-humphery-jenner-a-fi; reason=fail_no_progress |
| reddit | 131 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 139 | SoM | DOM | visual-missing | P6 | P6; 1 steps; actions finish; final_url=http://100.95.81.103:9999/; reason=fail_early_finish |
| reddit | 139 | SoM | Vision | abandon-after-N | P5,P14 | P5,P14; 6 steps; actions scroll→scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9999/; reason=fail_no_progress |
| reddit | 139 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 142 | SoM | DOM | visual-missing | P5,P13,P14 | P5,P13,P14; 11 steps; actions type→type→type→type→type→type→type→type→…; final_url=http://100.95.81.103:9999/search?q=Creed+III+release+date+; reason=fail_no_progress |
| reddit | 142 | SoM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9999/f/movies/86107; reason=fail_early_finish |
| reddit | 142 | SoM | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 148 | Vision | DOM | visual-missing | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9999/submission_images/57601f433cafd96a7d3883beea; reason=fail_early_finish |
| reddit | 148 | Vision | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9999/f/food/60745/homemade-arancini; reason=fail_early_finish |
| reddit | 148 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 150 | Vision | DOM | visual-missing | P14 | P14; 7 steps; actions click→click→click→scroll→type→click→finish; final_url=http://100.95.81.103:9999/f/OldSchoolCool/35826/myself-center-in-1966/; reason=fail_finish_eval_mismatch |
| reddit | 150 | Vision | SoM | visual-hijack/click-loop | P14 | P14; 7 steps; actions click→click→click→scroll→type→click→finish; final_url=http://100.95.81.103:9999/f/OldSchoolCool/35826/myself-center-in-1966/; reason=fail_finish_eval_mismatch |
| reddit | 150 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 152 | Vision | DOM | visual-missing | - | no P-rule; 5 steps; actions scroll→type→type→type→type; final_url=http://100.95.81.103:9999/f/OldSchoolCool/15059; reason=fail_no_progress |
| reddit | 152 | Vision | SoM | other | - | no P-rule; 4 steps; actions scroll→type→type→type; final_url=http://100.95.81.103:9999/f/OldSchoolCool/15059; reason=fail_no_progress |
| reddit | 152 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |
| reddit | 162 | Phantom-SoM | DOM | search-loop | P14 | P14; 30 steps; actions type→click→type→click→type→type→click→click→…; final_url=http://100.95.81.103:9999/; reason=fail_max_steps_search_repeat |
| reddit | 162 | Phantom-SoM | SoM | search-loop | P13 | P13; 6 steps; actions type→type→type→type→type→finish; final_url=http://100.95.81.103:9999/search?q=retirement+account+vs+brokerage+acc; reason=fail_finish_claim_missing |
| reddit | 162 | Phantom-SoM | Vision | search-loop | P13 | P13; 5 steps; actions type→type→type→type→finish; final_url=http://100.95.81.103:9999/search?q=%2Ff%2Fwallstreetbets; reason=fail_finish_claim_missing |
| reddit | 179 | Vision | DOM | visual-missing | P6 | P6; 30 steps; actions click→type→type→type→type→type→type→click→…; final_url=http://100.95.81.103:9999/search?q=Missouri+city+discussion+; reason=fail_max_steps_search_repeat |
| reddit | 179 | Vision | SoM | abandon-after-N | - | no P-rule; 30 steps; actions type→type→click→type→type→type→type→type→…; final_url=http://100.95.81.103:9999/search?q=St.+Louis+forum+; reason=fail_max_steps_search_repeat |
| reddit | 179 | Vision | Phantom-SoM | trace-unavailable | - | summary-only: step trace unavailable after run clear; counted from adjusted_success summary. |

## B1 Disagreement Analysis

This section repeats the B0 disagreement analysis for **B1 = Qwen3-VL-4B local** on the completed B1 3-mode runs only: `B1_3mode_classifieds_20260413` and `B1_3mode_reddit_20260413`. Phantom-SoM is not included in the B1 table because the B1 phantom runs are cleared / trace-unavailable; this section therefore measures DOM/SoM/Vision capability-by-representation interaction.

### B1 disagreement task IDs (cls + red)

- Total B1 one-arm-only disagreement tasks: **45** (`classifieds` 33, `reddit` 12).
- Failure-side `(task, mode)` pairs diagnosed: **90 / 90** have step traces.
- Cluster categories observed under the shared paper taxonomy: **9**; low-frequency P-rule-only cases are folded into `other`.

| Site | DOM-only success | SoM-only success | Vision-only success |
|---|---|---|---|
| classifieds | 12: `10, 15, 25, 45, 50, 64, 83, 101, 164, 189, 196, 210` | 15: `17, 19, 40, 48, 93, 111, 112, 127, 130, 135, 170, 173, 174, 184, 221` | 6: `44, 79, 110, 131, 151, 220` |
| reddit | 7: `0, 6, 18, 58, 100, 188, 189` | 3: `77, 131, 171` | 2: `120, 201` |

### Per-mode failure distribution (B1)

| Mode | N fail | search-loop | click-loop | early-finish/wrong-commit | abandon-after-N | visual-missing | visual-hijack/click-loop | click-loop/no-text-grounding | element-misground | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DOM | 26 | 2 (7.7%) | 1 (3.8%) | 1 (3.8%) | 1 (3.8%) | 21 (80.8%) | - | - | - | - |
| SoM | 27 | - | - | 7 (25.9%) | 1 (3.7%) | - | 19 (70.4%) | - | - | - |
| Vision | 37 | - | - | 6 (16.2%) | 6 (16.2%) | - | - | 11 (29.7%) | 9 (24.3%) | 5 (13.5%) |

Site split:

| Site | Mode | search-loop | click-loop family | early-finish | visual-missing | element-misground | other |
|---|---|---:|---:|---:|---:|---:|---:|
| classifieds | DOM | 2 | 1 | 1 | 16 | 0 | 1 |
| classifieds | SoM | 0 | 13 | 4 | 0 | 0 | 1 |
| classifieds | Vision | 0 | 10 | 5 | 0 | 4 | 8 |
| reddit | DOM | 0 | 0 | 0 | 5 | 0 | 0 |
| reddit | SoM | 0 | 6 | 3 | 0 | 0 | 0 |
| reddit | Vision | 0 | 1 | 1 | 0 | 5 | 3 |

Pair membership uses `adjusted_success`; the compact diagnostics preserve the raw trajectory termination string, so a few failed-side rows still display `reason=success` when raw success was later adjusted false.

Representative B1 diag excerpts:

- `classifieds task 45 SoM` (winner: DOM, cluster: `visual-hijack/click-loop`): P5,P13,P14; 7 steps; actions type→scroll→scroll→scroll→type→type→type; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=putty+knife+; reason=fail_no_progress. Intent: I recall seeing this exact item on the site, help me find the most recent post of it.
- `reddit task 100 SoM` (winner: DOM, cluster: `visual-hijack/click-loop`): P5,P14; 16 steps; actions click→scroll→scroll→scroll→type→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9999/submission_images/ba413f846d564b8c3997acd1b7; reason=fail_finish_empty_answer. Intent: Navigate to the comments section of a post that contains a picture of a cake.
- `classifieds task 17 DOM` (winner: SoM, cluster: `visual-missing`): P5,P6,P14; 10 steps; actions select_option→type→scroll→click→back→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=bike+with+red; reason=fail_no_progress. Intent: Show me the cheapest bike with red handlebars between $900-950.
- `reddit task 131 DOM` (winner: SoM, cluster: `visual-missing`): P5,P6,P14; 6 steps; actions type→click→type→click→click→click; final_url=http://100.95.81.103:9999/search?q=Finance+; reason=fail_no_progress. Intent: Subscribe to the forum which is most likely to include discussions related to the image.
- `classifieds task 83 Vision` (winner: DOM, cluster: `element-misground`): P1,P5; 8 steps; actions click→click→scroll→click→scroll→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=19&sOrder=i_; reason=fail_no_progress. Intent: Tell me the last name of the lister who posted the item on the page that does not include jewelry in its image?
- `reddit task 120 DOM` (winner: Vision, cluster: `visual-missing`): no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9999/submission_images/361ec602ee63f8d052217fe657; reason=fail_early_finish. Intent: According to the image, which country in this picture has the worst rail system?
- `classifieds task 10 Vision` (winner: DOM, cluster: `early-finish/wrong-commit`): no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/; reason=fail_early_finish. Intent: What is the seat height in inches of the smaller piece of furniture on this page?
- `classifieds task 10 SoM` (winner: DOM, cluster: `visual-hijack/click-loop`): P14; 3 steps; actions select_option→select_option→select_option; final_url=http://100.95.81.103:9980/; reason=fail_no_progress. Intent: What is the seat height in inches of the smaller piece of furniture on this page?
- `classifieds task 15 SoM` (winner: DOM, cluster: `early-finish/wrong-commit`): no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=15342; reason=fail_early_finish. Intent: What is the email of the seller of the guitar in the red case on this page?
- `classifieds task 25 Vision` (winner: DOM, cluster: `abandon-after-N`): P14; 4 steps; actions type→scroll→scroll→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=red+boat; reason=fail_finish_eval_mismatch. Intent: How many red boats were posted on 16th November 2023?

### B0-vs-B1 contrast table

| Mode / pattern | B0 count | B0 share | B1 count | B1 share | Shift | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| DOM visual-missing | 34/46 | 73.9% | 21/26 | 80.8% | +6.9 pp | Universal: visual-bound tasks defeat text-only DOM in both models. |
| DOM search-loop | 3/46 | 6.5% | 2/26 | 7.7% | +1.2 pp | Capability/denominator-sensitive: the exclusive slice is visual-heavy, so search-loop is not dominant here despite the whole-run reddit gradient. |
| SoM early-finish/wrong-commit | 16/30 | 53.3% | 7/27 | 25.9% | -27.4 pp | B0 confidently commits wrong more often; B1 shifts away from this pattern. |
| SoM visual-hijack/click-loop | 8/30 | 26.7% | 19/27 | 70.4% | +43.7 pp | Capability interaction: B1 is more mark/loop vulnerable, matching the §100 attention-hijack hypothesis. |
| Vision text/grounding loops | 11/41 | 26.8% | 11/37 | 29.7% | +2.9 pp | Universal Vision weakness: screen evidence without stable text/element grounding causes repeated plausible clicks. |
| Vision element-misground | 12/41 | 29.3% | 9/37 | 24.3% | -4.9 pp | B1 remains vulnerable to coordinate/target errors; high full-run P1 rate shows this beyond the exclusive slice. |

### Implications for paper Section 5 capability sub-claim

1. **DOM visual-missing is universal across capability.** B0 DOM visual-missing was 34/46 failure pairs; B1 DOM is 21/26. The model size/capability change does not remove the representational blind spot when the task requires image/color/page-screen evidence.
2. **SoM failure mode shifts with capability.** In B0, SoM failures in the exclusive slice were mostly early/wrong commit (16/30) with fewer mark-driven loops (8/30). In B1, SoM flips: visual-hijack/click-loop is 19/27 while early/wrong commit falls to 7/27. This is direct evidence for a capability × representation interaction: the smaller local model is more likely to be captured by the mark/action surface and repeat on the wrong page.
3. **Vision failures are representation-limited rather than purely capability-limited.** B0 and B1 both show substantial Vision click/grounding and element-targeting failures. The screenshot supplies visual evidence, but without DOM/mark text the agent lacks stable symbolic anchors for multi-step correction.
4. **Phantom-SoM remains unresolved for B1 step-level mechanism.** The B1 phantom runs are cleared, so this B1 section cannot diagnose Phantom trajectories. Use this B1 section for DOM/SoM/Vision capability interaction and keep Phantom mechanism claims anchored to B0 §103 macro metrics until B1 phantom traces are restored.

### B1 compact pair diagnostics

| Site | Task | Winner | Failed mode | Category | P-rules | Diag excerpt |
|---|---:|---|---|---|---|---|
| classifieds | 10 | DOM | SoM | visual-hijack/click-loop | P14 | P14; 3 steps; actions select_option→select_option→select_option; final_url=http://100.95.81.103:9980/; reason=fail_no_progress |
| classifieds | 10 | DOM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/; reason=fail_early_finish |
| classifieds | 15 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=15342; reason=fail_early_finish |
| classifieds | 15 | DOM | Vision | early-finish/wrong-commit | - | no P-rule; 2 steps; actions scroll→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=21; reason=fail_early_finish |
| classifieds | 17 | SoM | DOM | visual-missing | P5,P6,P14 | P5,P6,P14; 10 steps; actions select_option→type→scroll→click→back→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=bike+with+red; reason=fail_no_progress |
| classifieds | 17 | SoM | Vision | abandon-after-N | P5,P14 | P5,P14; 7 steps; actions type→scroll→scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=bike+with+red; reason=fail_no_progress |
| classifieds | 19 | SoM | DOM | abandon-after-N | P5,P14 | P5,P14; 8 steps; actions select_option→type→scroll→scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=sea+painting+; reason=fail_no_progress |
| classifieds | 19 | SoM | Vision | other | P14 | P14; 4 steps; actions click→select_option→select_option→select_option; final_url=http://100.95.81.103:9980/; reason=fail_no_progress |
| classifieds | 25 | DOM | SoM | visual-hijack/click-loop | P14 | P14; 3 steps; actions type→scroll→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=red+boats+; reason=fail_finish_eval_mismatch |
| classifieds | 25 | DOM | Vision | abandon-after-N | P14 | P14; 4 steps; actions type→scroll→scroll→finish; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=red+boat; reason=fail_finish_eval_mismatch |
| classifieds | 40 | SoM | DOM | search-loop | P5,P14 | P5,P14; 13 steps; actions type→click→type→scroll→type→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=3; reason=fail_no_progress |
| classifieds | 40 | SoM | Vision | click-loop/no-text-grounding | P5,P14 | P5,P14; 10 steps; actions type→scroll→click→back→click→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=18607; reason=fail_no_progress |
| classifieds | 44 | Vision | DOM | visual-missing | P6 | P6; 5 steps; actions type→type→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=53620; reason=fail_finish_wrong_url_not_found |
| classifieds | 44 | Vision | SoM | visual-hijack/click-loop | P5,P14 | P5,P14; 15 steps; actions type→click→type→type→type→type→type→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_no_progress |
| classifieds | 45 | DOM | SoM | visual-hijack/click-loop | P5,P13,P14 | P5,P13,P14; 7 steps; actions type→scroll→scroll→scroll→type→type→type; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=putty+knife+; reason=fail_no_progress |
| classifieds | 45 | DOM | Vision | click-loop/no-text-grounding | P5,P14 | P5,P14; 4 steps; actions type→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=HYDE+NO.1000+; reason=fail_no_progress |
| classifieds | 48 | SoM | DOM | visual-missing | P6 | P6; 4 steps; actions select_option→type→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=44305; reason=fail_finish_empty_answer |
| classifieds | 48 | SoM | Vision | click-loop/no-text-grounding | P14 | P14; 5 steps; actions click→click→click→click→click; final_url=http://100.95.81.103:9980/; reason=fail_no_progress |
| classifieds | 50 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=15485; reason=fail_early_finish |
| classifieds | 50 | DOM | Vision | early-finish/wrong-commit | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=15485; reason=fail_early_finish |
| classifieds | 64 | DOM | SoM | visual-hijack/click-loop | P5,P13,P14 | P5,P13,P14; 9 steps; actions type→type→type→scroll→scroll→type→type→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_no_progress |
| classifieds | 64 | DOM | Vision | click-loop/no-text-grounding | P14 | P14; 13 steps; actions type→type→type→click→type→click→click→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_incomplete_or_stuck |
| classifieds | 79 | Vision | DOM | visual-missing | P6 | P6; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=9&sOrder=dt_; reason=fail_early_finish |
| classifieds | 79 | Vision | SoM | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=9&sOrder=dt_; reason=fail_early_finish |
| classifieds | 83 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=49779; reason=fail_early_finish |
| classifieds | 83 | DOM | Vision | element-misground | P1,P5 | P1,P5; 8 steps; actions click→click→scroll→click→scroll→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=19&sOrder=i_; reason=fail_no_progress |
| classifieds | 93 | SoM | DOM | visual-missing | P6 | P6; 7 steps; actions type→scroll→click→type→type→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=33622; reason=fail_finish_wrong_url_not_found |
| classifieds | 93 | SoM | Vision | abandon-after-N | P5 | P5; 6 steps; actions scroll→scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sRegion=7361885&sCateg; reason=fail_no_progress |
| classifieds | 101 | DOM | SoM | visual-hijack/click-loop | P2,P5,P14 | P2,P5,P14; 12 steps; actions select_option→type→click→back→click→click→click→back→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=painting+&sCa; reason=fail_no_progress |
| classifieds | 101 | DOM | Vision | click-loop/no-text-grounding | P14 | P14; 5 steps; actions click→click→click→click→click; final_url=http://100.95.81.103:9980/; reason=fail_no_progress |
| classifieds | 110 | Vision | DOM | visual-missing | P10 | P10; 4 steps; actions type→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=34406; reason=fail_finish_eval_mismatch |
| classifieds | 110 | Vision | SoM | visual-hijack/click-loop | P5,P14 | P5,P14; 18 steps; actions type→scroll→click→back→click→back→click→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=mario+kart+; reason=fail_no_progress |
| classifieds | 111 | SoM | DOM | search-loop | P14 | P14; 30 steps; actions type→click→type→scroll→type→type→type→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=12; reason=fail_max_steps_search_repeat |
| classifieds | 111 | SoM | Vision | element-misground | P1,P5,P14 | P1,P5,P14; 5 steps; actions type→click→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=hockey; reason=fail_no_progress |
| classifieds | 112 | SoM | DOM | visual-missing | P5,P14 | P5,P14; 6 steps; actions type→scroll→type→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=basketball+; reason=fail_no_progress |
| classifieds | 112 | SoM | Vision | click-loop/no-text-grounding | P5,P14 | P5,P14; 6 steps; actions type→scroll→scroll→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=basketball; reason=fail_no_progress |
| classifieds | 127 | SoM | DOM | click-loop | P5,P14 | P5,P14; 7 steps; actions type→scroll→scroll→click→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=item&id=68187; reason=fail_no_progress |
| classifieds | 127 | SoM | Vision | click-loop/no-text-grounding | P5,P14 | P5,P14; 5 steps; actions type→scroll→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=MCAT; reason=fail_no_progress |
| classifieds | 130 | SoM | DOM | visual-missing | P5,P14 | P5,P14; 12 steps; actions scroll→scroll→scroll→scroll→type→type→type→type→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=7657; reason=fail_no_progress |
| classifieds | 130 | SoM | Vision | element-misground | P1 | P1; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_no_progress |
| classifieds | 131 | Vision | DOM | visual-missing | - | no P-rule; 3 steps; actions type→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=65634; reason=fail_finish_wrong_url_not_found |
| classifieds | 131 | Vision | SoM | abandon-after-N | - | no P-rule; 30 steps; actions type→click→type→type→click→type→type→click→…; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=dogs+; reason=fail_max_steps_target_unreachable |
| classifieds | 135 | SoM | DOM | visual-missing | P6,P14 | P6,P14; 30 steps; actions type→click→back→click→type→type→type→type→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=success |
| classifieds | 135 | SoM | Vision | abandon-after-N | P5 | P5; 6 steps; actions scroll→scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=success |
| classifieds | 151 | Vision | DOM | visual-missing | P6,P14 | P6,P14; 11 steps; actions scroll→scroll→scroll→scroll→type→type→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_no_progress |
| classifieds | 151 | Vision | SoM | visual-hijack/click-loop | P14 | P14; 6 steps; actions click→click→scroll→scroll→click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=9146; reason=fail_finish_empty_answer |
| classifieds | 164 | DOM | SoM | visual-hijack/click-loop | P5,P14 | P5,P14; 5 steps; actions type→click→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=item&id=5636; reason=success |
| classifieds | 164 | DOM | Vision | abandon-after-N | P5 | P5; 6 steps; actions scroll→scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=10&iPage=4y; reason=success |
| classifieds | 170 | SoM | DOM | visual-missing | P6,P14 | P6,P14; 30 steps; actions type→type→type→type→click→click→back→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=fail_max_steps_target_unreachable |
| classifieds | 170 | SoM | Vision | click-loop/no-text-grounding | P5,P14 | P5,P14; 18 steps; actions click→scroll→click→back→scroll→click→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=11; reason=fail_no_progress |
| classifieds | 173 | SoM | DOM | visual-missing | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=14834; reason=fail_early_finish |
| classifieds | 173 | SoM | Vision | element-misground | P1 | P1; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=17&sOrder=i_; reason=fail_no_progress |
| classifieds | 174 | SoM | DOM | visual-missing | P5,P6 | P5,P6; 7 steps; actions scroll→scroll→scroll→scroll→scroll→type→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=8383; reason=fail_finish_eval_mismatch |
| classifieds | 174 | SoM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=17&sOrder=i_; reason=fail_early_finish |
| classifieds | 184 | SoM | DOM | visual-missing | P6,P14 | P6,P14; 30 steps; actions select_option→type→type→type→type→click→back→type→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=15047; reason=fail_max_steps_search_repeat |
| classifieds | 184 | SoM | Vision | click-loop/no-text-grounding | P14 | P14; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9980/; reason=fail_no_progress |
| classifieds | 189 | DOM | SoM | visual-hijack/click-loop | P5 | P5; 5 steps; actions scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=24&sShowAs=g; reason=success |
| classifieds | 189 | DOM | Vision | abandon-after-N | P5 | P5; 5 steps; actions scroll→scroll→scroll→scroll→scroll; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=24&sShowAs=g; reason=success |
| classifieds | 196 | DOM | SoM | visual-hijack/click-loop | P7,P10,P14 | P7,P10,P14; 27 steps; actions select_option→type→click→scroll→type→type→scroll→click→…; final_url=http://100.95.81.103:9980/index.php?page=search&sOrder=dt_pub_date&iOr; reason=success |
| classifieds | 196 | DOM | Vision | other | P14 | P14; 4 steps; actions click→select_option→select_option→select_option; final_url=http://100.95.81.103:9980/; reason=success |
| classifieds | 210 | DOM | SoM | visual-hijack/click-loop | P14 | P14; 6 steps; actions type→click→click→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=lamb+; reason=fail_incomplete_or_stuck |
| classifieds | 210 | DOM | Vision | click-loop/no-text-grounding | P5,P14 | P5,P14; 12 steps; actions click→scroll→click→click→scroll→click→scroll→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=search&sCategory=16; reason=fail_no_progress |
| classifieds | 220 | Vision | DOM | visual-missing | P6,P14 | P6,P14; 13 steps; actions type→scroll→click→click→type→scroll→click→scroll→…; final_url=http://100.95.81.103:9980/index.php?page=item&id=76756#contact-in; reason=fail_finish_eval_mismatch |
| classifieds | 220 | Vision | SoM | visual-hijack/click-loop | P5,P14 | P5,P14; 4 steps; actions type→click→click→click; final_url=http://100.95.81.103:9980/index.php?page=search&sPattern=red+microwave; reason=success |
| classifieds | 221 | SoM | DOM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions scroll→finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=48575; reason=fail_early_finish |
| classifieds | 221 | SoM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9980/index.php?page=item&id=48575; reason=fail_early_finish |
| reddit | 0 | DOM | SoM | visual-hijack/click-loop | P14 | P14; 6 steps; actions click→click→click→click→click→click; final_url=http://100.95.81.103:9999/f/food/18823/i-ate-maple-pecan-croissant; reason=fail_incomplete_or_stuck |
| reddit | 0 | DOM | Vision | other | - | no P-rule; 5 steps; actions click→click→click→click→click; final_url=http://100.95.81.103:9999/f/food; reason=fail_no_progress |
| reddit | 6 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 9 steps; actions click→type→click→click→click→click→click→type→…; final_url=http://100.95.81.103:9999/search?q=cooking+post+with+pork+in+pan+; reason=fail_finish_eval_mismatch |
| reddit | 6 | DOM | Vision | other | - | no P-rule; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9999/forums/all; reason=fail_no_progress |
| reddit | 18 | DOM | SoM | visual-hijack/click-loop | P5,P14 | P5,P14; 10 steps; actions click→click→click→click→back→click→type→click→…; final_url=http://100.95.81.103:9999/search?q=colorful+keyboard+; reason=fail_no_progress |
| reddit | 18 | DOM | Vision | click-loop/no-text-grounding | P5 | P5; 4 steps; actions click→click→click→click; final_url=http://100.95.81.103:9999/forums/all; reason=fail_no_progress |
| reddit | 58 | DOM | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions type→finish; final_url=http://100.95.81.103:9999/search?q=most+popular+novel+adapted+anime+20; reason=fail_early_finish |
| reddit | 58 | DOM | Vision | other | P14 | P14; 4 steps; actions scroll→back→back→back; final_url=http://100.95.81.103:9999/f/dataisbeautiful/38990%20%7CAND%7C%20http:/; reason=fail_incomplete_or_stuck |
| reddit | 77 | SoM | DOM | visual-missing | P5,P14 | P5,P14; 30 steps; actions type→click→scroll→scroll→scroll→back→click→scroll→…; final_url=http://100.95.81.103:9999/submission_images/5bcb8d62749031657890e4b43b; reason=fail_max_steps_click_back_loop |
| reddit | 77 | SoM | Vision | element-misground | P1,P5,P14 | P1,P5,P14; 7 steps; actions click→click→type→click→scroll→scroll→scroll; final_url=http://100.95.81.103:9999/submission_images/5bcb8d62749031657890e4b43b; reason=fail_no_progress |
| reddit | 100 | DOM | SoM | visual-hijack/click-loop | P5,P14 | P5,P14; 16 steps; actions click→scroll→scroll→scroll→type→scroll→scroll→scroll→…; final_url=http://100.95.81.103:9999/submission_images/ba413f846d564b8c3997acd1b7; reason=fail_finish_empty_answer |
| reddit | 100 | DOM | Vision | element-misground | P1,P5,P14 | P1,P5,P14; 4 steps; actions click→click→click→click; final_url=http://100.95.81.103:9999/; reason=fail_no_progress |
| reddit | 120 | Vision | DOM | visual-missing | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9999/submission_images/361ec602ee63f8d052217fe657; reason=fail_early_finish |
| reddit | 120 | Vision | SoM | early-finish/wrong-commit | - | no P-rule; 2 steps; actions click→finish; final_url=http://100.95.81.103:9999/submission_images/361ec602ee63f8d052217fe657; reason=fail_early_finish |
| reddit | 131 | SoM | DOM | visual-missing | P5,P6,P14 | P5,P6,P14; 6 steps; actions type→click→type→click→click→click; final_url=http://100.95.81.103:9999/search?q=Finance+; reason=fail_no_progress |
| reddit | 131 | SoM | Vision | element-misground | P1,P14 | P1,P14; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9999/; reason=fail_incomplete_or_stuck |
| reddit | 171 | SoM | DOM | visual-missing | P6 | P6; 1 steps; actions finish; final_url=http://100.95.81.103:9999/; reason=fail_early_finish |
| reddit | 171 | SoM | Vision | element-misground | P1,P14 | P1,P14; 3 steps; actions click→click→click; final_url=http://100.95.81.103:9999/; reason=fail_incomplete_or_stuck |
| reddit | 188 | DOM | SoM | visual-hijack/click-loop | P14 | P14; 11 steps; actions click→click→click→click→click→type→click→click→…; final_url=http://100.95.81.103:9999/forums; reason=fail_no_progress |
| reddit | 188 | DOM | Vision | element-misground | P1,P5,P14 | P1,P5,P14; 5 steps; actions click→click→click→click→click; final_url=http://100.95.81.103:9999/forums; reason=fail_no_progress |
| reddit | 189 | DOM | SoM | visual-hijack/click-loop | P5,P14 | P5,P14; 6 steps; actions click→click→click→click→click→click; final_url=http://100.95.81.103:9999/forums; reason=fail_no_progress |
| reddit | 189 | DOM | Vision | early-finish/wrong-commit | - | no P-rule; 1 steps; actions finish; final_url=http://100.95.81.103:9999/; reason=fail_early_finish |
| reddit | 201 | Vision | DOM | visual-missing | P5,P6,P14 | P5,P6,P14; 4 steps; actions type→click→click→click; final_url=http://100.95.81.103:9999/search?q=bed+; reason=fail_no_progress |
| reddit | 201 | Vision | SoM | visual-hijack/click-loop | P5 | P5; 4 steps; actions scroll→click→click→click; final_url=http://100.95.81.103:9999/f/explainlikeimfive; reason=fail_no_progress |
