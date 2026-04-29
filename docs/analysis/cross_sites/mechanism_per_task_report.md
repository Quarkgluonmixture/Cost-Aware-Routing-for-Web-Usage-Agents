# Per-task mechanism evidence (E1-E4)

This report explains why mode swaps move outcomes by using per-task and per-step evidence. Element ids are excluded because they are not stable across navigation steps or observation modes. Click evidence uses URL-changing transitions `(pre_url_signature, post_url_signature)`, trajectory evidence uses URL signatures per step, confidence evidence reads existing per-run calibration outputs, and action vocabulary evidence uses normalized action types.

## E1 Click-target divergence

E1 asks whether modes click into the same server-determined page transitions. Jaccard is computed over each task's set of URL-changing click transitions, then averaged across paired tasks.

| site | contrast | N | mean Jaccard | std | median | mean divergence | left size | right size | union size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reddit | axis_1_text (DOM vs P-text) | 210 | 0.463 | 0.472 | 0.250 | 0.537 | 1.105 | 0.952 | 1.795 |
| reddit | axis_2_prompt (P-text vs P-SoM) | 210 | 0.484 | 0.485 | 0.333 | 0.516 | 0.952 | 0.833 | 1.590 |
| reddit | axis_3_image (P-SoM vs SoM) | 210 | 0.492 | 0.491 | 0.333 | 0.508 | 0.833 | 0.548 | 1.262 |
| reddit | compound_DOM_to_PSoM (DOM vs P-SoM) | 210 | 0.421 | 0.471 | 0.000 | 0.579 | 1.105 | 0.833 | 1.743 |
| reddit | axis_2_prompt_alt (DOM vs P-prompt) | 138 | 0.502 | 0.470 | 0.500 | 0.498 | 1.109 | 0.891 | 1.717 |
| reddit | axis_1_text_alt (P-prompt vs P-SoM) | 138 | 0.486 | 0.485 | 0.310 | 0.514 | 0.891 | 0.862 | 1.536 |
| classifieds | axis_1_text (DOM vs P-text) | 234 | 0.561 | 0.476 | 1.000 | 0.439 | 0.697 | 0.650 | 1.158 |
| classifieds | axis_2_prompt (P-text vs P-SoM) | 234 | 0.542 | 0.485 | 1.000 | 0.458 | 0.650 | 0.756 | 1.231 |
| classifieds | axis_3_image (P-SoM vs SoM) | 234 | 0.482 | 0.488 | 0.292 | 0.518 | 0.756 | 0.517 | 1.167 |
| classifieds | compound_DOM_to_PSoM (DOM vs P-SoM) | 234 | 0.531 | 0.478 | 0.633 | 0.469 | 0.697 | 0.756 | 1.261 |
| classifieds | axis_2_prompt_alt (DOM vs P-prompt) | 0 (pending) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| classifieds | axis_1_text_alt (P-prompt vs P-SoM) | 0 (pending) | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

Per-axis interpretation:
- axis_1_text: reddit Jaccard 0.463; classifieds Jaccard 0.561. Lower values indicate that the modes use different URL-changing click decisions.
- axis_2_prompt: reddit Jaccard 0.484; classifieds Jaccard 0.542. Lower values indicate that the modes use different URL-changing click decisions.
- axis_3_image: reddit Jaccard 0.492; classifieds Jaccard 0.482. Lower values indicate that the modes use different URL-changing click decisions.
- compound_DOM_to_PSoM: reddit Jaccard 0.421; classifieds Jaccard 0.531. Lower values indicate that the modes use different URL-changing click decisions.
- axis_2_prompt_alt: reddit Jaccard 0.502; classifieds Jaccard n/a. Lower values indicate that the modes use different URL-changing click decisions.
- axis_1_text_alt: reddit Jaccard 0.486; classifieds Jaccard n/a. Lower values indicate that the modes use different URL-changing click decisions.

Case-study anchors from E2 below should be read with E1: tasks with low click-transition overlap often diverge before the final answer, not merely at finish time.

## E2 Trajectory boundary divergence

E2 filters to symmetric-difference tasks, where exactly one side of the contrast has adjusted success. It then records the first step where URL signatures differ. Early divergence is step <= 3; late divergence is step >= 10.

| site | contrast | symmetric diff N | median first step | early rate | late rate | case tasks |
|---|---|---:|---:|---:|---:|---|
| reddit | DOM_vs_P-text | 16 | 0.000 | 87.5% | 0.0% | 15, 81, 107 |
| reddit | P-text_vs_Phantom-SoM | 15 | 0 | 100.0% | 0.0% | 7, 167, 26 |
| reddit | Phantom-SoM_vs_SoM | 23 | 0 | 95.2% | 0.0% | 7, 0, 14 |
| reddit | DOM_vs_Phantom-SoM | 23 | 0 | 91.3% | 0.0% | 7, 81, 15 |
| reddit | DOM_vs_Phantom-prompt | 13 | 1 | 76.9% | 0.0% | 7, 19, 17 |
| reddit | Phantom-prompt_vs_Phantom-SoM | 8 | 0 | 100.0% | 0.0% | 15, 77, 17 |
| classifieds | DOM_vs_P-text | 21 | 1 | 81.0% | 0.0% | 63, 201, 98 |
| classifieds | P-text_vs_Phantom-SoM | 26 | 1.000 | 92.3% | 0.0% | 17, 103, 79 |
| classifieds | Phantom-SoM_vs_SoM | 42 | 1 | 94.9% | 0.0% | 14, 17, 49 |
| classifieds | DOM_vs_Phantom-SoM | 23 | 1 | 73.9% | 4.3% | 17, 201, 63 |
| classifieds | DOM_vs_Phantom-prompt | 0 | n/a | n/a | n/a |  |
| classifieds | Phantom-prompt_vs_Phantom-SoM | 0 | n/a | n/a | n/a |  |

E2 case studies:
- reddit DOM_vs_P-text task_15: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 12 vs 8.
- reddit DOM_vs_P-text task_81: first divergent step 6, trajectory Jaccard 1.000, left_success=True, right_success=False, steps 8 vs 6.
- reddit DOM_vs_P-text task_107: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 4.
- reddit P-text_vs_Phantom-SoM task_7: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 4 vs 5.
- reddit P-text_vs_Phantom-SoM task_26: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 9 vs 6.
- reddit P-text_vs_Phantom-SoM task_167: first divergent step 3, trajectory Jaccard 0.333, left_success=False, right_success=True, steps 6 vs 30.
- reddit Phantom-SoM_vs_SoM task_0: first divergent step 6, trajectory Jaccard 0.667, left_success=True, right_success=False, steps 12 vs 30.
- reddit Phantom-SoM_vs_SoM task_7: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 5 vs 11.
- reddit Phantom-SoM_vs_SoM task_14: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 7.
- reddit DOM_vs_Phantom-SoM task_7: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 5.
- reddit DOM_vs_Phantom-SoM task_15: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 12 vs 30.
- reddit DOM_vs_Phantom-SoM task_81: first divergent step 6, trajectory Jaccard 1.000, left_success=True, right_success=False, steps 8 vs 6.
- reddit DOM_vs_Phantom-prompt task_7: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 3.
- reddit DOM_vs_Phantom-prompt task_17: first divergent step 0, trajectory Jaccard 0.200, left_success=False, right_success=True, steps 30 vs 16.
- reddit DOM_vs_Phantom-prompt task_19: first divergent step 4, trajectory Jaccard 1.000, left_success=True, right_success=False, steps 5 vs 4.
- reddit Phantom-prompt_vs_Phantom-SoM task_15: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 30.
- reddit Phantom-prompt_vs_Phantom-SoM task_17: first divergent step 0, trajectory Jaccard 0.222, left_success=True, right_success=False, steps 16 vs 30.
- reddit Phantom-prompt_vs_Phantom-SoM task_77: first divergent step 2, trajectory Jaccard 1.000, left_success=True, right_success=False, steps 30 vs 5.
- classifieds DOM_vs_P-text task_63: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 14 vs 10.
- classifieds DOM_vs_P-text task_98: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 5 vs 5.
- classifieds DOM_vs_P-text task_201: first divergent step 7, trajectory Jaccard 0.500, left_success=False, right_success=True, steps 12 vs 11.
- classifieds P-text_vs_Phantom-SoM task_17: first divergent step 0, trajectory Jaccard 0.125, left_success=True, right_success=False, steps 6 vs 30.
- classifieds P-text_vs_Phantom-SoM task_79: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 5 vs 3.
- classifieds P-text_vs_Phantom-SoM task_103: first divergent step 7, trajectory Jaccard 0.500, left_success=False, right_success=True, steps 8 vs 12.
- classifieds Phantom-SoM_vs_SoM task_14: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 2 vs 2.
- classifieds Phantom-SoM_vs_SoM task_17: first divergent step 5, trajectory Jaccard 1.000, left_success=False, right_success=True, steps 30 vs 5.
- classifieds Phantom-SoM_vs_SoM task_49: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 5 vs 3.
- classifieds DOM_vs_Phantom-SoM task_17: first divergent step 0, trajectory Jaccard 0.125, left_success=True, right_success=False, steps 6 vs 30.
- classifieds DOM_vs_Phantom-SoM task_63: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 14 vs 11.
- classifieds DOM_vs_Phantom-SoM task_201: first divergent step 10, trajectory Jaccard 0.400, left_success=False, right_success=True, steps 12 vs 18.

## E3 Confidence calibration cross-condition aggregator

E3 reads existing `analyze_confidence_calibration.py` outputs under `analysis/signals/combined/tables`. It does not recompute calibration. B1 runs expose per-mode token and verbalized calibration in `per_mode_summary.csv`; B0 API runs expose verbalized and behavioral AUROC but no token-level calibration in the existing outputs.

| model | site | mode | ECE token | ECE verbal | AUROC token | AUROC verbal | AUROC behavioral max | FP rate | best signals |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 | classifieds | DOM | n/a | n/a | n/a | 0.742 | 0.769 | n/a | verb=ep_mean_verbalized; beh=action_diversity |
| B0 | classifieds | P-text | n/a | n/a | n/a | 0.737 | 0.733 | n/a | verb=ep_mean_verbalized; beh=action_diversity |
| B0 | classifieds | P-SoM | n/a | n/a | n/a | 0.701 | 0.728 | n/a | verb=ep_mean_verbalized; beh=action_diversity |
| B0 | classifieds | SoM | n/a | n/a | n/a | 0.709 | 0.697 | n/a | verb=ep_mean_verbalized; beh=action_diversity |
| B0 | classifieds | Vision | n/a | n/a | n/a | 0.763 | 0.773 | n/a | verb=ep_mean_verbalized; beh=max_repeat_streak |
| B0 | reddit | DOM | n/a | n/a | n/a | 0.817 | 0.682 | n/a | verb=ep_mean_verbalized; beh=max_repeat_streak |
| B0 | reddit | P-text | n/a | n/a | n/a | 0.793 | 0.698 | n/a | verb=ep_mean_verbalized; beh=url_revisit_count |
| B0 | reddit | P-SoM | n/a | n/a | n/a | 0.720 | 0.694 | n/a | verb=ep_mean_verbalized; beh=max_repeat_streak |
| B0 | reddit | P-prompt | n/a | n/a | n/a | n/a | n/a | n/a |  |
| B0 | reddit | SoM | n/a | n/a | n/a | 0.719 | 0.681 | n/a | verb=ep_mean_verbalized; beh=action_diversity |
| B0 | reddit | Vision | n/a | n/a | n/a | 0.778 | 0.709 | n/a | verb=ep_mean_verbalized; beh=max_repeat_streak |
| B1 | classifieds | DOM | 0.837 | 0.635 | 0.674 | 0.683 | 0.760 | n/a | tok=ep_max_entropy; verb=ep_mean_verbalized; beh=max_repeat_streak |
| B1 | classifieds | P-SoM | n/a | n/a | 0.689 | 0.715 | 0.788 | n/a | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=action_diversity |
| B1 | classifieds | SoM | 0.781 | 0.606 | 0.653 | 0.755 | 0.727 | n/a | tok=ep_mean_logprob; verb=ep_mean_verbalized; beh=url_revisit_max |
| B1 | classifieds | Vision | 0.839 | 0.602 | 0.541 | 0.757 | 0.816 | n/a | tok=ep_max_entropy; verb=ep_mean_verbalized; beh=url_revisit_max |
| B1 | reddit | DOM | 0.828 | 0.666 | 0.548 | 0.724 | 0.620 | n/a | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=url_revisit_count |
| B1 | reddit | SoM | 0.850 | 0.740 | 0.589 | 0.638 | 0.613 | n/a | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=url_revisit_max |
| B1 | reddit | Vision | 0.874 | 0.678 | 0.660 | 0.698 | 0.862 | n/a | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=url_revisit_count |

E3 highlights:
- B0/classifieds: honest-commit mode None (ECE n/a); best-signal mode B0/classifieds/Vision (AUROC 0.773).
- B0/reddit: honest-commit mode None (ECE n/a); best-signal mode B0/reddit/DOM (AUROC 0.817).
- B1/classifieds: honest-commit mode B1/classifieds/Vision (ECE 0.602); best-signal mode B1/classifieds/Vision (AUROC 0.816).
- B1/reddit: honest-commit mode B1/reddit/DOM (ECE 0.666); best-signal mode B1/reddit/Vision (AUROC 0.862).

Layer 0b FP cross-reference: B0 FP rates are attached for cells present in `sr_fp_per_mode.json`. Because B0 ECE is absent from the existing analyzer outputs, low-ECE versus low-FP claims should be made only for B1 calibration cells or deferred until B0 calibration tables are generated.

## E4 Action vocabulary distribution

E4 expands Layer 1 macro behavior from a few hand-picked action metrics to the full normalized action vocabulary. Fractions below are pooled over all steps in each B0 site/mode cell.

| cell | click | type | scroll | select | wait | back | forward | finish | tab_focus | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0/classifieds/DOM | 0.250 | 0.293 | 0.226 | 0.075 | 0.000 | 0.093 | 0.000 | 0.049 | 0.014 | 0.000 |
| B0/classifieds/P-text | 0.222 | 0.329 | 0.246 | 0.078 | 0.000 | 0.065 | 0.000 | 0.052 | 0.007 | 0.000 |
| B0/classifieds/P-SoM | 0.255 | 0.291 | 0.236 | 0.066 | 0.000 | 0.099 | 0.000 | 0.043 | 0.010 | 0.000 |
| B0/classifieds/SoM | 0.260 | 0.282 | 0.200 | 0.060 | 0.001 | 0.099 | 0.000 | 0.091 | 0.006 | 0.000 |
| B0/classifieds/Vision | 0.271 | 0.353 | 0.246 | 0.037 | 0.000 | 0.030 | 0.000 | 0.057 | 0.005 | 0.000 |
| B0/reddit/DOM | 0.358 | 0.378 | 0.142 | 0.001 | 0.000 | 0.059 | 0.000 | 0.037 | 0.026 | 0.000 |
| B0/reddit/P-text | 0.337 | 0.350 | 0.198 | 0.011 | 0.000 | 0.051 | 0.000 | 0.039 | 0.015 | 0.000 |
| B0/reddit/P-SoM | 0.381 | 0.290 | 0.149 | 0.011 | 0.000 | 0.085 | 0.000 | 0.042 | 0.042 | 0.000 |
| B0/reddit/Phantom-prompt | 0.333 | 0.271 | 0.228 | 0.008 | 0.000 | 0.097 | 0.000 | 0.041 | 0.022 | 0.000 |
| B0/reddit/SoM | 0.391 | 0.226 | 0.156 | 0.015 | 0.000 | 0.096 | 0.000 | 0.066 | 0.050 | 0.000 |
| B0/reddit/Vision | 0.326 | 0.254 | 0.255 | 0.006 | 0.000 | 0.033 | 0.000 | 0.064 | 0.062 | 0.000 |

Paired action-fraction shifts by axis (right-minus-left):

| site | axis | N | top shift 1 | top shift 2 | top shift 3 |
|---|---|---:|---|---|---|
| reddit | axis_1_text | 210 | scroll 0.034 | type -0.023 | select_option 0.016 |
| reddit | axis_2_prompt | 210 | tab_focus 0.083 | type -0.066 | scroll -0.038 |
| reddit | axis_3_image | 210 | finish 0.071 | scroll -0.034 | click -0.031 |
| reddit | compound_DOM_to_PSoM | 210 | type -0.089 | tab_focus 0.071 | back 0.013 |
| reddit | axis_2_prompt_alt | 138 | scroll 0.028 | type -0.022 | back 0.020 |
| reddit | axis_1_text_alt | 138 | tab_focus 0.124 | type -0.066 | scroll -0.046 |
| classifieds | axis_1_text | 234 | back -0.015 | type 0.015 | tab_focus -0.009 |
| classifieds | axis_2_prompt | 234 | type -0.037 | click 0.021 | back 0.021 |
| classifieds | axis_3_image | 234 | finish 0.135 | scroll -0.095 | click -0.016 |
| classifieds | compound_DOM_to_PSoM | 234 | scroll 0.027 | type -0.022 | click 0.014 |
| classifieds | axis_2_prompt_alt | 0 |  |  |  |
| classifieds | axis_1_text_alt | 0 |  |  |  |

Uncommon-action highlights:
- reddit select_option: SoM 0.015 vs DOM 0.001 (20.5x).
- reddit tab_focus: Vision 0.062 vs P-text 0.015 (4.3x).
- classifieds back: P-SoM 0.099 vs Vision 0.030 (3.3x).
- reddit back: P-prompt 0.097 vs Vision 0.033 (3.0x).
- classifieds finish: SoM 0.091 vs P-SoM 0.043 (2.1x).
- classifieds select_option: P-text 0.078 vs Vision 0.037 (2.1x).

## Mechanism evidence for paper Section 5

DOM and P-SoM click transitions diverge at event granularity: compound click-target Jaccard is 0.421 on reddit and 0.531 on classifieds.

Boundary divergence is usually visible early among symmetric-difference tasks: DOM vs P-SoM early rates are reddit 91.3% and classifieds 73.9%.

The strongest calibration/routing cell is B1/reddit/Vision with max AUROC 0.862; B0 token ECE is unavailable in existing per-run outputs.

The largest action-vocabulary shift is classifieds axis_3_image finish (0.135 right-minus-left).

Together, E1 and E2 support a decision-path account: mode swaps change which URL transitions are attempted and how early trajectories split on tasks where outcomes disagree. E3 keeps the commitment-confidence claim separate from path choice: confidence evidence is useful, but existing B0 outputs support it mainly through verbalized and behavioral AUROC rather than token calibration. E4 shows whether those path changes are accompanied by broad policy-shape shifts in the action vocabulary, or whether the same action mix hides different click targets.

## Appendix A: E1 click-set size histograms

These histograms summarize how many URL-changing click transitions each task produced. The key is set size; the value is the number of tasks with that size.

| site | axis | left mode | left hist | right mode | right hist |
|---|---|---|---|---|---|
| reddit | axis_1_text | DOM | `{"0": 99, "1": 53, "2": 28, "3": 14, "4": 8, "5": 4, "6": 1, "7": 1, "8": 2}` | P-text | `{"0": 106, "1": 56, "2": 20, "3": 16, "4": 7, "5": 3, "6": 1, "7": 1}` |
| reddit | axis_2_prompt | P-text | `{"0": 106, "1": 56, "2": 20, "3": 16, "4": 7, "5": 3, "6": 1, "7": 1}` | P-SoM | `{"0": 113, "1": 58, "10": 1, "2": 20, "3": 10, "4": 4, "5": 3, "6": 1}` |
| reddit | axis_3_image | P-SoM | `{"0": 113, "1": 58, "10": 1, "2": 20, "3": 10, "4": 4, "5": 3, "6": 1}` | SoM | `{"0": 129, "1": 58, "2": 17, "3": 4, "5": 1, "6": 1}` |
| reddit | compound_DOM_to_PSoM | DOM | `{"0": 99, "1": 53, "2": 28, "3": 14, "4": 8, "5": 4, "6": 1, "7": 1, "8": 2}` | P-SoM | `{"0": 113, "1": 58, "10": 1, "2": 20, "3": 10, "4": 4, "5": 3, "6": 1}` |
| reddit | axis_2_prompt_alt | DOM | `{"0": 64, "1": 36, "2": 19, "3": 8, "4": 6, "5": 3, "8": 2}` | P-prompt | `{"0": 70, "1": 41, "2": 14, "3": 6, "4": 3, "5": 2, "7": 2}` |
| reddit | axis_1_text_alt | P-prompt | `{"0": 70, "1": 41, "2": 14, "3": 6, "4": 3, "5": 2, "7": 2}` | P-SoM | `{"0": 81, "1": 30, "10": 1, "2": 11, "3": 8, "4": 3, "5": 3, "6": 1}` |
| classifieds | axis_1_text | DOM | `{"0": 147, "1": 46, "2": 21, "3": 11, "4": 6, "5": 2, "8": 1}` | P-text | `{"0": 143, "1": 51, "2": 25, "3": 11, "4": 2, "5": 2}` |
| classifieds | axis_2_prompt | P-text | `{"0": 143, "1": 51, "2": 25, "3": 11, "4": 2, "5": 2}` | P-SoM | `{"0": 136, "1": 50, "2": 30, "3": 10, "4": 5, "5": 2, "7": 1}` |
| classifieds | axis_3_image | P-SoM | `{"0": 136, "1": 50, "2": 30, "3": 10, "4": 5, "5": 2, "7": 1}` | SoM | `{"0": 159, "1": 50, "2": 14, "3": 4, "4": 6, "7": 1}` |
| classifieds | compound_DOM_to_PSoM | DOM | `{"0": 147, "1": 46, "2": 21, "3": 11, "4": 6, "5": 2, "8": 1}` | P-SoM | `{"0": 136, "1": 50, "2": 30, "3": 10, "4": 5, "5": 2, "7": 1}` |
| classifieds | axis_2_prompt_alt | DOM | `{}` | P-prompt | `{}` |
| classifieds | axis_1_text_alt | P-prompt | `{}` | P-SoM | `{}` |

## Appendix B: E2 first-divergence histograms

The histogram key is first divergent step. Step 0 means the two modes start from different URL signatures or immediately navigate differently.

| site | contrast | histogram |
|---|---|---|
| reddit | DOM_vs_P-text | `{"0": 10, "1": 2, "2": 1, "3": 1, "5": 1, "6": 1}` |
| reddit | P-text_vs_Phantom-SoM | `{"0": 12, "1": 1, "2": 1, "3": 1}` |
| reddit | Phantom-SoM_vs_SoM | `{"0": 13, "1": 7, "6": 1}` |
| reddit | DOM_vs_Phantom-SoM | `{"0": 13, "1": 5, "2": 1, "3": 2, "4": 1, "6": 1}` |
| reddit | DOM_vs_Phantom-prompt | `{"0": 5, "1": 3, "2": 2, "4": 3}` |
| reddit | Phantom-prompt_vs_Phantom-SoM | `{"0": 6, "2": 1}` |
| classifieds | DOM_vs_P-text | `{"0": 5, "1": 8, "2": 2, "3": 2, "4": 1, "6": 2, "7": 1}` |
| classifieds | P-text_vs_Phantom-SoM | `{"0": 10, "1": 6, "2": 3, "3": 5, "4": 1, "7": 1}` |
| classifieds | Phantom-SoM_vs_SoM | `{"0": 11, "1": 18, "2": 5, "3": 3, "4": 1, "5": 1}` |
| classifieds | DOM_vs_Phantom-SoM | `{"0": 8, "1": 6, "10": 1, "2": 1, "3": 2, "4": 2, "5": 2, "9": 1}` |
| classifieds | DOM_vs_Phantom-prompt | `{}` |
| classifieds | Phantom-prompt_vs_Phantom-SoM | `{}` |

## Appendix C: E3 source provenance

Each E3 row is sourced from existing analyzer outputs; calibration source is null when the run exposes AUROC but not per-mode ECE/MCE/Brier.

| cell | AUROC source | calibration source | token n | verbal n | behavioral n |
|---|---|---|---:|---:|---:|
| B0/classifieds/DOM | `results/visualwebarena/phase1/B0_3mode_classifieds_20260413/analysis/signals/combined/tables/cross_mode_auroc.csv` | n/a | n/a | 234 | 234 |
| B0/classifieds/P-text | `results/visualwebarena/phase1/B0_phantom_text_classifieds_20260427/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | n/a | 234 | 234 |
| B0/classifieds/Phantom-SoM | `results/visualwebarena/phase1/B0_phantom_som_classifieds_20260426/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | n/a | 234 | 234 |
| B0/classifieds/SoM | `results/visualwebarena/phase1/B0_3mode_classifieds_20260413/analysis/signals/combined/tables/cross_mode_auroc.csv` | n/a | n/a | 234 | 234 |
| B0/classifieds/Vision | `results/visualwebarena/phase1/B0_3mode_classifieds_20260413/analysis/signals/combined/tables/cross_mode_auroc.csv` | n/a | n/a | 234 | 234 |
| B0/reddit/DOM | `results/visualwebarena/phase1/B0_3mode_reddit_20260422/analysis/signals/combined/tables/cross_mode_auroc.csv` | n/a | n/a | 209 | 210 |
| B0/reddit/P-text | `results/visualwebarena/phase1/B0_phantom_text_reddit_20260427/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | n/a | 208 | 210 |
| B0/reddit/Phantom-SoM | `results/visualwebarena/phase1/B0_phantom_som_reddit_20260428/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | n/a | 205 | 210 |
| B0/reddit/Phantom-prompt | `results/visualwebarena/phase1/B0_phantom_prompt_reddit_20260429/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | n/a | n/a | n/a |
| B0/reddit/SoM | `results/visualwebarena/phase1/B0_3mode_reddit_20260422/analysis/signals/combined/tables/cross_mode_auroc.csv` | n/a | n/a | 199 | 210 |
| B0/reddit/Vision | `results/visualwebarena/phase1/B0_3mode_reddit_20260422/analysis/signals/combined/tables/cross_mode_auroc.csv` | n/a | n/a | 210 | 210 |
| B1/classifieds/DOM | `results/visualwebarena/phase1/B1_3mode_classifieds_20260413/analysis/signals/combined/tables/cross_mode_auroc.csv` | `results/visualwebarena/phase1/B1_3mode_classifieds_20260413/analysis/signals/combined/tables/per_mode_summary.csv` | 234 | 234 | 234 |
| B1/classifieds/Phantom-SoM | `results/visualwebarena/phase1/B1_phantom_som_classifieds_20260428/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 230 | 230 | 230 |
| B1/classifieds/SoM | `results/visualwebarena/phase1/B1_3mode_classifieds_20260413/analysis/signals/combined/tables/cross_mode_auroc.csv` | `results/visualwebarena/phase1/B1_3mode_classifieds_20260413/analysis/signals/combined/tables/per_mode_summary.csv` | 234 | 234 | 234 |
| B1/classifieds/Vision | `results/visualwebarena/phase1/B1_3mode_classifieds_20260413/analysis/signals/combined/tables/cross_mode_auroc.csv` | `results/visualwebarena/phase1/B1_3mode_classifieds_20260413/analysis/signals/combined/tables/per_mode_summary.csv` | 234 | 234 | 234 |
| B1/reddit/DOM | `results/visualwebarena/phase1/B1_3mode_reddit_20260413/analysis/signals/combined/tables/cross_mode_auroc.csv` | `results/visualwebarena/phase1/B1_3mode_reddit_20260413/analysis/signals/combined/tables/per_mode_summary.csv` | 210 | 210 | 210 |
| B1/reddit/SoM | `results/visualwebarena/phase1/B1_3mode_reddit_20260413/analysis/signals/combined/tables/cross_mode_auroc.csv` | `results/visualwebarena/phase1/B1_3mode_reddit_20260413/analysis/signals/combined/tables/per_mode_summary.csv` | 210 | 210 | 210 |
| B1/reddit/Vision | `results/visualwebarena/phase1/B1_3mode_reddit_20260413/analysis/signals/combined/tables/cross_mode_auroc.csv` | `results/visualwebarena/phase1/B1_3mode_reddit_20260413/analysis/signals/combined/tables/per_mode_summary.csv` | 210 | 210 | 210 |

## Appendix D: E4 full action-shift matrix

All values are paired per-task action-fraction shifts in the cascade direction, right-minus-left.

| site | axis | click | type | scroll | select | wait | back | forward | finish | tab_focus | other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reddit | axis_1_text | -0.010 | -0.023 | 0.034 | 0.016 | 0.000 | 0.001 | 0.000 | -0.006 | -0.012 | 0.000 |
| reddit | axis_2_prompt | -0.001 | -0.066 | -0.038 | -0.003 | 0.000 | 0.011 | 0.000 | 0.014 | 0.083 | 0.000 |
| reddit | axis_3_image | -0.031 | -0.025 | -0.034 | 0.013 | 0.000 | -0.005 | 0.000 | 0.071 | 0.011 | 0.000 |
| reddit | compound_DOM_to_PSoM | -0.012 | -0.089 | -0.003 | 0.012 | 0.000 | 0.013 | 0.000 | 0.009 | 0.071 | 0.000 |
| reddit | axis_2_prompt_alt | -0.017 | -0.022 | 0.028 | 0.011 | 0.000 | 0.020 | 0.000 | -0.010 | -0.010 | 0.000 |
| reddit | axis_1_text_alt | -0.012 | -0.066 | -0.046 | -0.000 | 0.000 | -0.014 | 0.000 | 0.015 | 0.124 | 0.000 |
| classifieds | axis_1_text | -0.007 | 0.015 | 0.008 | 0.005 | 0.000 | -0.015 | 0.000 | 0.004 | -0.009 | 0.000 |
| classifieds | axis_2_prompt | 0.021 | -0.037 | 0.018 | -0.013 | 0.000 | 0.021 | 0.000 | -0.015 | 0.005 | 0.000 |
| classifieds | axis_3_image | -0.016 | 0.007 | -0.095 | -0.010 | 0.001 | -0.013 | 0.000 | 0.135 | -0.007 | 0.000 |
| classifieds | compound_DOM_to_PSoM | 0.014 | -0.022 | 0.027 | -0.008 | 0.000 | 0.005 | 0.000 | -0.011 | -0.005 | 0.000 |
| classifieds | axis_2_prompt_alt | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| classifieds | axis_1_text_alt | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |

## Appendix E: E4 ranked per-cell actions

Top action types per B0 cell by pooled step fraction.

| cell | rank | action_type | fraction |
|---|---:|---|---:|
| B0/classifieds/DOM | 1 | type | 0.293 |
| B0/classifieds/DOM | 2 | click | 0.250 |
| B0/classifieds/DOM | 3 | scroll | 0.226 |
| B0/classifieds/DOM | 4 | back | 0.093 |
| B0/classifieds/DOM | 5 | select_option | 0.075 |
| B0/classifieds/P-text | 1 | type | 0.329 |
| B0/classifieds/P-text | 2 | scroll | 0.246 |
| B0/classifieds/P-text | 3 | click | 0.222 |
| B0/classifieds/P-text | 4 | select_option | 0.078 |
| B0/classifieds/P-text | 5 | back | 0.065 |
| B0/classifieds/Phantom-SoM | 1 | type | 0.291 |
| B0/classifieds/Phantom-SoM | 2 | click | 0.255 |
| B0/classifieds/Phantom-SoM | 3 | scroll | 0.236 |
| B0/classifieds/Phantom-SoM | 4 | back | 0.099 |
| B0/classifieds/Phantom-SoM | 5 | select_option | 0.066 |
| B0/classifieds/SoM | 1 | type | 0.282 |
| B0/classifieds/SoM | 2 | click | 0.260 |
| B0/classifieds/SoM | 3 | scroll | 0.200 |
| B0/classifieds/SoM | 4 | back | 0.099 |
| B0/classifieds/SoM | 5 | finish | 0.091 |
| B0/classifieds/Vision | 1 | type | 0.353 |
| B0/classifieds/Vision | 2 | click | 0.271 |
| B0/classifieds/Vision | 3 | scroll | 0.246 |
| B0/classifieds/Vision | 4 | finish | 0.057 |
| B0/classifieds/Vision | 5 | select_option | 0.037 |
| B0/reddit/DOM | 1 | type | 0.378 |
| B0/reddit/DOM | 2 | click | 0.358 |
| B0/reddit/DOM | 3 | scroll | 0.142 |
| B0/reddit/DOM | 4 | back | 0.059 |
| B0/reddit/DOM | 5 | finish | 0.037 |
| B0/reddit/P-text | 1 | type | 0.350 |
| B0/reddit/P-text | 2 | click | 0.337 |
| B0/reddit/P-text | 3 | scroll | 0.198 |
| B0/reddit/P-text | 4 | back | 0.051 |
| B0/reddit/P-text | 5 | finish | 0.039 |
| B0/reddit/Phantom-SoM | 1 | click | 0.381 |
| B0/reddit/Phantom-SoM | 2 | type | 0.290 |
| B0/reddit/Phantom-SoM | 3 | scroll | 0.149 |
| B0/reddit/Phantom-SoM | 4 | back | 0.085 |
| B0/reddit/Phantom-SoM | 5 | finish | 0.042 |
| B0/reddit/Phantom-prompt | 1 | click | 0.333 |
| B0/reddit/Phantom-prompt | 2 | type | 0.271 |
| B0/reddit/Phantom-prompt | 3 | scroll | 0.228 |
| B0/reddit/Phantom-prompt | 4 | back | 0.097 |
| B0/reddit/Phantom-prompt | 5 | finish | 0.041 |
| B0/reddit/SoM | 1 | click | 0.391 |
| B0/reddit/SoM | 2 | type | 0.226 |
| B0/reddit/SoM | 3 | scroll | 0.156 |
| B0/reddit/SoM | 4 | back | 0.096 |
| B0/reddit/SoM | 5 | finish | 0.066 |
| B0/reddit/Vision | 1 | click | 0.326 |
| B0/reddit/Vision | 2 | scroll | 0.255 |
| B0/reddit/Vision | 3 | type | 0.254 |
| B0/reddit/Vision | 4 | finish | 0.064 |
| B0/reddit/Vision | 5 | tab_focus | 0.062 |

## Appendix F: validation detail

| validation check | value | pass |
|---|---|---|
| axis1 N reddit | 210 / 210 | True |
| axis1 N classifieds | 234 / 234 | True |
| E1 Jaccard range classifieds/axis_1_text | [0, 1] | True |
| E1 Jaccard range classifieds/axis_2_prompt | [0, 1] | True |
| E1 Jaccard range classifieds/axis_3_image | [0, 1] | True |
| E1 Jaccard range classifieds/compound_DOM_to_PSoM | [0, 1] | True |
| E1 Jaccard range reddit/axis_1_text | [0, 1] | True |
| E1 Jaccard range reddit/axis_1_text_alt | [0, 1] | True |
| E1 Jaccard range reddit/axis_2_prompt | [0, 1] | True |
| E1 Jaccard range reddit/axis_2_prompt_alt | [0, 1] | True |
| E1 Jaccard range reddit/axis_3_image | [0, 1] | True |
| E1 Jaccard range reddit/compound_DOM_to_PSoM | [0, 1] | True |
| E2 first step range classifieds/DOM_vs_P-text | [0, 30] | True |
| E2 first step range classifieds/DOM_vs_Phantom-SoM | [0, 30] | True |
| E2 first step range classifieds/DOM_vs_Phantom-prompt | [0, 30] | True |
| E2 first step range classifieds/P-text_vs_Phantom-SoM | [0, 30] | True |
| E2 first step range classifieds/Phantom-SoM_vs_SoM | [0, 30] | True |
| E2 first step range classifieds/Phantom-prompt_vs_Phantom-SoM | [0, 30] | True |
| E2 first step range reddit/DOM_vs_P-text | [0, 30] | True |
| E2 first step range reddit/DOM_vs_Phantom-SoM | [0, 30] | True |
| E2 first step range reddit/DOM_vs_Phantom-prompt | [0, 30] | True |
| E2 first step range reddit/P-text_vs_Phantom-SoM | [0, 30] | True |
| E2 first step range reddit/Phantom-SoM_vs_SoM | [0, 30] | True |
| E2 first step range reddit/Phantom-prompt_vs_Phantom-SoM | [0, 30] | True |
| E4 action sum B0/classifieds/DOM | 1.000000 | True |
| E4 action sum B0/classifieds/P-text | 1.000000 | True |
| E4 action sum B0/classifieds/Phantom-SoM | 1.000000 | True |
| E4 action sum B0/classifieds/SoM | 1.000000 | True |
| E4 action sum B0/classifieds/Vision | 1.000000 | True |
| E4 action sum B0/reddit/DOM | 1.000000 | True |
| E4 action sum B0/reddit/P-text | 1.000000 | True |
| E4 action sum B0/reddit/Phantom-SoM | 1.000000 | True |
| E4 action sum B0/reddit/Phantom-prompt | 1.000000 | True |
| E4 action sum B0/reddit/SoM | 1.000000 | True |
| E4 action sum B0/reddit/Vision | 1.000000 | True |
| E1_any_click_divergence_gt_0.1 | threshold > 0.1 | True |
| E2_any_early_or_late_rate_gt_0.1 | threshold > 0.1 | True |
| E3_any_AUROC_effect_gt_0.1 | threshold > 0.1 | True |
| E4_any_action_shift_gt_0.1 | threshold > 0.1 | True |

## Validation

Overall pass: True.

| check | result |
|---|---|
| E1 N reddit | 210 / 210 |
| E1 N classifieds | 234 / 234 |
| E3 cells | 18 / 18 |
| P-prompt status | {"B0_phantom_prompt_reddit_20260429": {"episodes": 138, "policy": "P-prompt is not included in E1-E4 contrasts.", "status": "partial / pending"}} |
