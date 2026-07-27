# Per-task mechanism evidence (E1-E4)

This report explains why mode swaps move outcomes by using per-task and per-step evidence. Element ids are excluded because they are not stable across navigation steps or observation modes. Click evidence uses URL-changing transitions `(pre_url_signature, post_url_signature)`, trajectory evidence uses URL signatures per step, confidence evidence reads existing per-run calibration outputs, and action vocabulary evidence uses normalized action types.

## E1 Click-target divergence

E1 asks whether modes click into the same server-determined page transitions. Jaccard is computed over each task's set of URL-changing click transitions, then averaged across paired tasks.

| site | contrast | N | mean Jaccard | std | median | mean divergence | left size | right size | union size |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| reddit | axis_1_text (DOM vs P-text) | 205 | 0.333 | 0.441 | 0.000 | 0.667 | 1.610 | 1.527 | 2.849 |
| reddit | axis_2_prompt (P-text vs P-SoM) | 205 | 0.409 | 0.448 | 0.167 | 0.591 | 1.527 | 1.624 | 2.785 |
| reddit | axis_3_image (P-SoM vs SoM) | 205 | 0.313 | 0.429 | 0.000 | 0.687 | 1.624 | 1.254 | 2.610 |
| reddit | compound_DOM_to_PSoM (DOM vs P-SoM) | 205 | 0.308 | 0.425 | 0.000 | 0.692 | 1.610 | 1.624 | 2.937 |
| reddit | axis_2_prompt_alt (DOM vs P-prompt) | 205 | 0.313 | 0.434 | 0.000 | 0.687 | 1.610 | 1.824 | 3.151 |
| reddit | axis_1_text_alt (P-prompt vs P-SoM) | 205 | 0.330 | 0.428 | 0.000 | 0.670 | 1.824 | 1.624 | 3.107 |
| classifieds | axis_1_text (DOM vs P-text) | 224 | 0.315 | 0.436 | 0.000 | 0.685 | 1.304 | 1.549 | 2.616 |
| classifieds | axis_2_prompt (P-text vs P-SoM) | 224 | 0.340 | 0.439 | 0.000 | 0.660 | 1.549 | 1.473 | 2.714 |
| classifieds | axis_3_image (P-SoM vs SoM) | 224 | 0.313 | 0.445 | 0.000 | 0.687 | 1.473 | 1.121 | 2.406 |
| classifieds | compound_DOM_to_PSoM (DOM vs P-SoM) | 224 | 0.332 | 0.436 | 0.000 | 0.668 | 1.304 | 1.473 | 2.469 |
| classifieds | axis_2_prompt_alt (DOM vs P-prompt) | 224 | 0.313 | 0.428 | 0.000 | 0.687 | 1.304 | 1.402 | 2.460 |
| classifieds | axis_1_text_alt (P-prompt vs P-SoM) | 224 | 0.306 | 0.433 | 0.000 | 0.694 | 1.402 | 1.473 | 2.647 |

Per-axis interpretation:
- axis_1_text: reddit Jaccard 0.333; classifieds Jaccard 0.315. Lower values indicate that the modes use different URL-changing click decisions.
- axis_2_prompt: reddit Jaccard 0.409; classifieds Jaccard 0.340. Lower values indicate that the modes use different URL-changing click decisions.
- axis_3_image: reddit Jaccard 0.313; classifieds Jaccard 0.313. Lower values indicate that the modes use different URL-changing click decisions.
- compound_DOM_to_PSoM: reddit Jaccard 0.308; classifieds Jaccard 0.332. Lower values indicate that the modes use different URL-changing click decisions.
- axis_2_prompt_alt: reddit Jaccard 0.313; classifieds Jaccard 0.313. Lower values indicate that the modes use different URL-changing click decisions.
- axis_1_text_alt: reddit Jaccard 0.330; classifieds Jaccard 0.306. Lower values indicate that the modes use different URL-changing click decisions.

Case-study anchors from E2 below should be read with E1: tasks with low click-transition overlap often diverge before the final answer, not merely at finish time.

## E2 Trajectory boundary divergence

E2 filters to symmetric-difference tasks, where exactly one side of the contrast has adjusted success. It then records the first step where URL signatures differ. Early divergence is step <= 3; late divergence is step >= 10.

| site | contrast | symmetric diff N | median first step | early rate | late rate | case tasks |
|---|---|---:|---:|---:|---:|---|
| reddit | DOM_vs_P-text | 16 | 0.000 | 93.8% | 0.0% | 14, 100, 15 |
| reddit | P-text_vs_P-SoM | 21 | 0 | 100.0% | 0.0% | 11, 100, 12 |
| reddit | P-SoM_vs_SoM | 21 | 0.000 | 95.0% | 0.0% | 4, 21, 11 |
| reddit | DOM_vs_P-SoM | 21 | 0 | 100.0% | 0.0% | 2, 58, 11 |
| reddit | DOM_vs_P-prompt | 18 | 1.000 | 88.9% | 0.0% | 2, 98, 100 |
| reddit | P-prompt_vs_P-SoM | 19 | 0 | 100.0% | 0.0% | 11, 93, 12 |
| classifieds | DOM_vs_P-text | 22 | 1.000 | 95.5% | 0.0% | 132, 75, 137 |
| classifieds | P-text_vs_P-SoM | 18 | 2.000 | 72.2% | 0.0% | 16, 12, 60 |
| classifieds | P-SoM_vs_SoM | 46 | 1.000 | 93.5% | 0.0% | 17, 75, 49 |
| classifieds | DOM_vs_P-SoM | 22 | 1.000 | 95.5% | 0.0% | 16, 103, 105 |
| classifieds | DOM_vs_P-prompt | 33 | 1 | 84.8% | 3.0% | 1, 149, 16 |
| classifieds | P-prompt_vs_P-SoM | 23 | 1 | 91.3% | 0.0% | 1, 201, 64 |

E2 case studies:
- reddit DOM_vs_P-text task_14: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 30 vs 30.
- reddit DOM_vs_P-text task_15: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 30.
- reddit DOM_vs_P-text task_100: first divergent step 8, trajectory Jaccard 0.429, left_success=True, right_success=False, steps 30 vs 30.
- reddit P-text_vs_P-SoM task_11: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 30.
- reddit P-text_vs_P-SoM task_12: first divergent step 0, trajectory Jaccard 0.250, left_success=False, right_success=True, steps 30 vs 30.
- reddit P-text_vs_P-SoM task_100: first divergent step 2, trajectory Jaccard 0.800, left_success=False, right_success=True, steps 30 vs 30.
- reddit P-SoM_vs_SoM task_4: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 10.
- reddit P-SoM_vs_SoM task_11: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 30 vs 6.
- reddit P-SoM_vs_SoM task_21: first divergent step 6, trajectory Jaccard 0.333, left_success=False, right_success=True, steps 17 vs 27.
- reddit DOM_vs_P-SoM task_2: first divergent step 0, trajectory Jaccard 0.200, left_success=True, right_success=False, steps 11 vs 28.
- reddit DOM_vs_P-SoM task_11: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 30.
- reddit DOM_vs_P-SoM task_58: first divergent step 2, trajectory Jaccard 0.333, left_success=True, right_success=False, steps 30 vs 32.
- reddit DOM_vs_P-prompt task_2: first divergent step 0, trajectory Jaccard 0.200, left_success=True, right_success=False, steps 11 vs 30.
- reddit DOM_vs_P-prompt task_98: first divergent step 6, trajectory Jaccard 1.000, left_success=True, right_success=False, steps 30 vs 30.
- reddit DOM_vs_P-prompt task_100: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 30 vs 30.
- reddit P-prompt_vs_P-SoM task_11: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 30 vs 30.
- reddit P-prompt_vs_P-SoM task_12: first divergent step 0, trajectory Jaccard 0.200, left_success=False, right_success=True, steps 30 vs 30.
- reddit P-prompt_vs_P-SoM task_93: first divergent step 2, trajectory Jaccard 0.125, left_success=True, right_success=False, steps 15 vs 11.
- classifieds DOM_vs_P-text task_75: first divergent step 8, trajectory Jaccard 1.000, left_success=True, right_success=False, steps 10 vs 30.
- classifieds DOM_vs_P-text task_132: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 13 vs 30.
- classifieds DOM_vs_P-text task_137: first divergent step 0, trajectory Jaccard 0.111, left_success=False, right_success=True, steps 6 vs 17.
- classifieds P-text_vs_P-SoM task_12: first divergent step 8, trajectory Jaccard 0.750, left_success=True, right_success=False, steps 14 vs 30.
- classifieds P-text_vs_P-SoM task_16: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 3 vs 4.
- classifieds P-text_vs_P-SoM task_60: first divergent step 0, trajectory Jaccard 0.118, left_success=True, right_success=False, steps 20 vs 30.
- classifieds P-SoM_vs_SoM task_17: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 8 vs 7.
- classifieds P-SoM_vs_SoM task_49: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 4 vs 3.
- classifieds P-SoM_vs_SoM task_75: first divergent step 8, trajectory Jaccard 1.000, left_success=True, right_success=False, steps 10 vs 9.
- classifieds DOM_vs_P-SoM task_16: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 2 vs 4.
- classifieds DOM_vs_P-SoM task_103: first divergent step 7, trajectory Jaccard 0.600, left_success=True, right_success=False, steps 19 vs 7.
- classifieds DOM_vs_P-SoM task_105: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 20 vs 7.
- classifieds DOM_vs_P-prompt task_1: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 10 vs 10.
- classifieds DOM_vs_P-prompt task_16: first divergent step 0, trajectory Jaccard 0.000, left_success=False, right_success=True, steps 2 vs 4.
- classifieds DOM_vs_P-prompt task_149: first divergent step 10, trajectory Jaccard 1.000, left_success=False, right_success=True, steps 19 vs 10.
- classifieds P-prompt_vs_P-SoM task_1: first divergent step 0, trajectory Jaccard 0.000, left_success=True, right_success=False, steps 10 vs 9.
- classifieds P-prompt_vs_P-SoM task_64: first divergent step 0, trajectory Jaccard 0.182, left_success=True, right_success=False, steps 18 vs 17.
- classifieds P-prompt_vs_P-SoM task_201: first divergent step 9, trajectory Jaccard 0.667, left_success=True, right_success=False, steps 28 vs 30.

## E3 Confidence calibration cross-condition aggregator

E3 reads existing `analyze_confidence_calibration.py` outputs under `analysis/signals/combined/tables`. It does not recompute calibration. B1 runs expose per-mode token and verbalized calibration in `per_mode_summary.csv`; B0 API runs expose verbalized and behavioral AUROC but no token-level calibration in the existing outputs.

| model | site | mode | ECE token | ECE verbal | AUROC token | AUROC verbal | AUROC behavioral max | canonical SR | best signals |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| B0 | classifieds | DOM | n/a | n/a | 0.555 | 0.603 | 0.673 | 17.411 | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=action_diversity |
| B0 | classifieds | P-SoM | n/a | n/a | 0.664 | 0.723 | 0.766 | 15.625 | tok=ep_min_margin; verb=ep_mean_verbalized; beh=action_diversity |
| B0 | classifieds | P-prompt | n/a | n/a | 0.612 | 0.593 | 0.683 | 19.643 | tok=ep_min_margin; verb=ep_min_verbalized; beh=action_diversity |
| B0 | classifieds | P-text | n/a | n/a | 0.620 | 0.601 | 0.708 | 15.625 | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=action_diversity |
| B0 | classifieds | SoM | n/a | n/a | 0.651 | 0.657 | 0.753 | 27.232 | tok=ep_min_margin; verb=ep_mean_verbalized; beh=action_diversity |
| B0 | classifieds | Vision | n/a | n/a | 0.620 | 0.692 | 0.764 | 25.000 | tok=ep_min_logprob; verb=ep_min_verbalized; beh=action_diversity |
| B0 | reddit | DOM | n/a | n/a | 0.614 | 0.760 | 0.685 | 14.634 | tok=ep_min_margin; verb=ep_mean_verbalized; beh=max_repeat_streak |
| B0 | reddit | P-SoM | n/a | n/a | 0.607 | 0.737 | 0.673 | 11.220 | tok=ep_min_margin; verb=ep_min_verbalized; beh=url_revisit_count |
| B0 | reddit | P-prompt | n/a | n/a | 0.575 | 0.658 | 0.644 | 12.683 | tok=ep_min_margin; verb=ep_min_verbalized; beh=url_revisit_count |
| B0 | reddit | P-text | n/a | n/a | 0.688 | 0.695 | 0.654 | 13.659 | tok=ep_min_logprob; verb=ep_min_verbalized; beh=url_revisit_max |
| B0 | reddit | SoM | n/a | n/a | 0.588 | 0.576 | 0.714 | 14.634 | tok=ep_min_margin; verb=ep_mean_verbalized; beh=url_revisit_count |
| B0 | reddit | Vision | n/a | n/a | 0.788 | 0.854 | 0.877 | 7.805 | tok=ep_min_margin; verb=ep_mean_verbalized; beh=url_revisit_count |
| B1 | classifieds | DOM | n/a | n/a | 0.753 | 0.702 | 0.870 | 6.250 | tok=ep_max_entropy; verb=ep_mean_verbalized; beh=action_diversity |
| B1 | classifieds | P-SoM | n/a | n/a | 0.704 | 0.727 | 0.631 | 6.696 | tok=ep_max_entropy; verb=ep_mean_verbalized; beh=action_diversity |
| B1 | classifieds | P-prompt | n/a | n/a | 0.631 | 0.659 | 0.727 | 6.696 | tok=ep_min_logprob; verb=ep_min_verbalized; beh=action_diversity |
| B1 | classifieds | P-text | n/a | n/a | 0.743 | 0.723 | 0.828 | 7.589 | tok=ep_max_entropy; verb=ep_mean_verbalized; beh=action_diversity |
| B1 | classifieds | SoM | n/a | n/a | 0.635 | 0.737 | 0.527 | 14.286 | tok=ep_mean_logprob; verb=ep_mean_verbalized; beh=action_diversity |
| B1 | classifieds | Vision | n/a | n/a | 0.533 | 0.806 | 0.707 | 12.500 | tok=ep_mean_margin; verb=ep_mean_verbalized; beh=action_diversity |
| B1 | reddit | DOM | n/a | n/a | 0.595 | 0.683 | 0.634 | 6.829 | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=url_revisit_count |
| B1 | reddit | P-SoM | n/a | n/a | 0.503 | 0.645 | 0.546 | 6.829 | tok=ep_mean_margin; verb=ep_min_verbalized; beh=max_repeat_streak |
| B1 | reddit | P-prompt | n/a | n/a | 0.706 | 0.646 | 0.659 | 6.341 | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=url_revisit_count |
| B1 | reddit | P-text | n/a | n/a | 0.688 | 0.685 | 0.641 | 6.829 | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=url_revisit_count |
| B1 | reddit | SoM | n/a | n/a | 0.506 | 0.667 | 0.519 | 8.293 | tok=ep_mean_entropy; verb=ep_mean_verbalized; beh=max_repeat_streak |
| B1 | reddit | Vision | n/a | n/a | n/a | n/a | n/a | 2.927 |  |
| B2 | classifieds | DOM | n/a | n/a | 0.682 | 0.671 | 0.915 | 1.339 | tok=ep_mean_margin; verb=ep_mean_verbalized; beh=url_revisit_count |
| B2 | classifieds | P-SoM | n/a | n/a | 0.836 | 0.367 | 0.881 | 0.893 | tok=ep_max_entropy; verb=ep_min_verbalized; beh=action_diversity |
| B2 | classifieds | P-prompt | n/a | n/a | 0.652 | 0.497 | 0.933 | 1.786 | tok=ep_min_logprob; verb=ep_min_verbalized; beh=max_repeat_streak |
| B2 | classifieds | P-text | n/a | n/a | 0.919 | 0.934 | 0.659 | 0.446 | tok=ep_max_entropy; verb=ep_min_verbalized; beh=max_repeat_streak |
| B2 | classifieds | SoM | n/a | n/a | 0.637 | 0.663 | 0.647 | 2.232 | tok=ep_max_entropy; verb=ep_mean_verbalized; beh=url_revisit_count |
| B2 | classifieds | Vision | n/a | n/a | 0.807 | 0.655 | 0.620 | 2.232 | tok=ep_min_logprob; verb=ep_mean_verbalized; beh=url_revisit_max |

E3 highlights:
- B0/classifieds: honest-commit mode None (ECE n/a); best-signal mode B0/classifieds/P-SoM (AUROC 0.766).
- B0/reddit: honest-commit mode None (ECE n/a); best-signal mode B0/reddit/Vision (AUROC 0.877).
- B1/classifieds: honest-commit mode None (ECE n/a); best-signal mode B1/classifieds/DOM (AUROC 0.870).
- B1/reddit: honest-commit mode None (ECE n/a); best-signal mode B1/reddit/P-prompt (AUROC 0.706).
- B2/classifieds: honest-commit mode None (ECE n/a); best-signal mode B2/classifieds/P-text (AUROC 0.934).

Outcome 0a SR cross-reference: canonical per-mode SR values are attached from `sr_per_mode.json`. Because B0 ECE is absent from the existing analyzer outputs, calibration claims remain limited to the fields actually emitted by each baseline.

## E4 Action vocabulary distribution

E4 expands the Macro dimension from a few hand-picked action metrics to the full normalized action vocabulary. Fractions below are pooled over all steps in each B0 site/mode cell.

| cell | click | type | scroll | select | wait | back | forward | finish | tab_focus | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0/classifieds/DOM | 0.306 | 0.275 | 0.205 | 0.064 | 0.001 | 0.085 | 0.000 | 0.047 | 0.015 | 0.002 |
| B0/classifieds/P-SoM | 0.313 | 0.218 | 0.258 | 0.062 | 0.001 | 0.091 | 0.000 | 0.042 | 0.004 | 0.010 |
| B0/classifieds/P-prompt | 0.325 | 0.252 | 0.206 | 0.061 | 0.001 | 0.087 | 0.000 | 0.048 | 0.016 | 0.004 |
| B0/classifieds/P-text | 0.311 | 0.228 | 0.214 | 0.074 | 0.000 | 0.108 | 0.000 | 0.045 | 0.004 | 0.016 |
| B0/classifieds/SoM | 0.312 | 0.252 | 0.170 | 0.101 | 0.001 | 0.079 | 0.000 | 0.054 | 0.003 | 0.027 |
| B0/classifieds/Vision | 0.353 | 0.164 | 0.295 | 0.080 | 0.000 | 0.051 | 0.000 | 0.044 | 0.006 | 0.005 |
| B0/reddit/DOM | 0.465 | 0.149 | 0.191 | 0.023 | 0.000 | 0.080 | 0.000 | 0.025 | 0.042 | 0.024 |
| B0/reddit/P-SoM | 0.442 | 0.132 | 0.126 | 0.024 | 0.003 | 0.074 | 0.000 | 0.018 | 0.152 | 0.029 |
| B0/reddit/P-prompt | 0.466 | 0.153 | 0.149 | 0.020 | 0.004 | 0.106 | 0.000 | 0.027 | 0.038 | 0.036 |
| B0/reddit/P-text | 0.432 | 0.150 | 0.127 | 0.015 | 0.001 | 0.081 | 0.000 | 0.016 | 0.150 | 0.029 |
| B0/reddit/SoM | 0.483 | 0.115 | 0.098 | 0.044 | 0.002 | 0.063 | 0.000 | 0.025 | 0.147 | 0.024 |
| B0/reddit/Vision | 0.319 | 0.072 | 0.368 | 0.008 | 0.002 | 0.045 | 0.000 | 0.015 | 0.138 | 0.033 |

Paired action-fraction shifts by axis (right-minus-left):

| site | axis | N | top shift 1 | top shift 2 | top shift 3 |
|---|---|---:|---|---|---|
| reddit | axis_1_text | 205 | tab_focus 0.077 | scroll -0.049 | finish -0.028 |
| reddit | axis_2_prompt | 205 | type -0.018 | click 0.014 | scroll 0.007 |
| reddit | axis_3_image | 205 | finish 0.035 | scroll -0.026 | click 0.023 |
| reddit | compound_DOM_to_PSoM | 205 | tab_focus 0.077 | scroll -0.043 | finish -0.030 |
| reddit | axis_2_prompt_alt | 205 | scroll -0.038 | back 0.023 | click 0.018 |
| reddit | axis_1_text_alt | 205 | tab_focus 0.074 | finish -0.025 | back -0.020 |
| classifieds | axis_1_text | 224 | type -0.031 | back 0.013 | scroll 0.013 |
| classifieds | axis_2_prompt | 224 | scroll 0.012 | back -0.011 | finish 0.009 |
| classifieds | axis_3_image | 224 | scroll -0.054 | finish 0.034 | select_option 0.018 |
| classifieds | compound_DOM_to_PSoM | 224 | scroll 0.025 | type -0.024 | tab_focus -0.010 |
| classifieds | axis_2_prompt_alt | 224 | type -0.016 | finish 0.015 | scroll -0.009 |
| classifieds | axis_1_text_alt | 224 | scroll 0.034 | click -0.016 | finish -0.009 |

Uncommon-action highlights:
- classifieds other: SoM 0.027 vs DOM 0.002 (13.5x).
- reddit select_option: SoM 0.044 vs Vision 0.008 (5.7x).
- classifieds tab_focus: P-prompt 0.016 vs SoM 0.003 (5.6x).
- reddit tab_focus: P-SoM 0.152 vs P-prompt 0.038 (4.0x).
- reddit scroll: Vision 0.368 vs SoM 0.098 (3.8x).
- reddit back: P-prompt 0.106 vs Vision 0.045 (2.4x).
- reddit type: P-prompt 0.153 vs Vision 0.072 (2.1x).
- classifieds back: P-text 0.108 vs Vision 0.051 (2.1x).

## Mechanism evidence for paper Section 5

Available DOM and P-SoM click transitions diverge at event granularity: compound click-target Jaccard is 0.308 on reddit; 0.332 on classifieds.

Boundary divergence is usually visible early among symmetric-difference tasks: DOM vs P-SoM early rates are reddit 100.0% and classifieds 95.5%.

The strongest calibration/routing cell is B2/classifieds/P-text with max AUROC 0.934; B0 token ECE is unavailable in existing per-run outputs.

The largest action-vocabulary shift is reddit axis_1_text tab_focus (0.077 right-minus-left).

Together, E1 and E2 support a decision-path account: mode swaps change which URL transitions are attempted and how early trajectories split on tasks where outcomes disagree. E3 keeps the commitment-confidence claim separate from path choice: confidence evidence is useful, but existing B0 outputs support it mainly through verbalized and behavioral AUROC rather than token calibration. E4 shows whether those path changes are accompanied by broad policy-shape shifts in the action vocabulary, or whether the same action mix hides different click targets.

## Appendix A: E1 click-set size histograms

These histograms summarize how many URL-changing click transitions each task produced. The key is set size; the value is the number of tasks with that size.

| site | axis | left mode | left hist | right mode | right hist |
|---|---|---|---|---|---|
| reddit | axis_1_text | DOM | `{"0": 75, "1": 48, "11": 1, "2": 35, "3": 16, "4": 14, "5": 8, "6": 3, "7": 3, "9": 2}` | P-text | `{"0": 76, "1": 47, "2": 37, "3": 18, "4": 11, "5": 8, "6": 5, "7": 1, "8": 1, "9": 1}` |
| reddit | axis_2_prompt | P-text | `{"0": 76, "1": 47, "2": 37, "3": 18, "4": 11, "5": 8, "6": 5, "7": 1, "8": 1, "9": 1}` | P-SoM | `{"0": 74, "1": 45, "2": 41, "3": 14, "4": 11, "5": 11, "6": 3, "7": 2, "8": 3, "9": 1}` |
| reddit | axis_3_image | P-SoM | `{"0": 74, "1": 45, "2": 41, "3": 14, "4": 11, "5": 11, "6": 3, "7": 2, "8": 3, "9": 1}` | SoM | `{"0": 80, "1": 58, "2": 32, "3": 16, "4": 12, "5": 4, "6": 2, "7": 1}` |
| reddit | compound_DOM_to_PSoM | DOM | `{"0": 75, "1": 48, "11": 1, "2": 35, "3": 16, "4": 14, "5": 8, "6": 3, "7": 3, "9": 2}` | P-SoM | `{"0": 74, "1": 45, "2": 41, "3": 14, "4": 11, "5": 11, "6": 3, "7": 2, "8": 3, "9": 1}` |
| reddit | axis_2_prompt_alt | DOM | `{"0": 75, "1": 48, "11": 1, "2": 35, "3": 16, "4": 14, "5": 8, "6": 3, "7": 3, "9": 2}` | P-prompt | `{"0": 65, "1": 48, "2": 31, "3": 27, "4": 12, "5": 10, "6": 4, "7": 4, "8": 3, "9": 1}` |
| reddit | axis_1_text_alt | P-prompt | `{"0": 65, "1": 48, "2": 31, "3": 27, "4": 12, "5": 10, "6": 4, "7": 4, "8": 3, "9": 1}` | P-SoM | `{"0": 74, "1": 45, "2": 41, "3": 14, "4": 11, "5": 11, "6": 3, "7": 2, "8": 3, "9": 1}` |
| classifieds | axis_1_text | DOM | `{"0": 81, "1": 71, "2": 31, "3": 26, "4": 4, "5": 6, "6": 1, "7": 3, "8": 1}` | P-text | `{"0": 87, "1": 52, "2": 33, "3": 22, "4": 12, "5": 7, "6": 4, "7": 3, "8": 1, "9": 3}` |
| classifieds | axis_2_prompt | P-text | `{"0": 87, "1": 52, "2": 33, "3": 22, "4": 12, "5": 7, "6": 4, "7": 3, "8": 1, "9": 3}` | P-SoM | `{"0": 88, "1": 62, "11": 1, "2": 28, "3": 17, "4": 11, "5": 8, "6": 3, "7": 2, "8": 2, "9": 2}` |
| classifieds | axis_3_image | P-SoM | `{"0": 88, "1": 62, "11": 1, "2": 28, "3": 17, "4": 11, "5": 8, "6": 3, "7": 2, "8": 2, "9": 2}` | SoM | `{"0": 107, "1": 59, "10": 1, "2": 25, "3": 13, "4": 9, "5": 5, "6": 3, "7": 2}` |
| classifieds | compound_DOM_to_PSoM | DOM | `{"0": 81, "1": 71, "2": 31, "3": 26, "4": 4, "5": 6, "6": 1, "7": 3, "8": 1}` | P-SoM | `{"0": 88, "1": 62, "11": 1, "2": 28, "3": 17, "4": 11, "5": 8, "6": 3, "7": 2, "8": 2, "9": 2}` |
| classifieds | axis_2_prompt_alt | DOM | `{"0": 81, "1": 71, "2": 31, "3": 26, "4": 4, "5": 6, "6": 1, "7": 3, "8": 1}` | P-prompt | `{"0": 95, "1": 50, "10": 1, "2": 40, "3": 12, "4": 10, "5": 6, "6": 5, "7": 3, "8": 1, "9": 1}` |
| classifieds | axis_1_text_alt | P-prompt | `{"0": 95, "1": 50, "10": 1, "2": 40, "3": 12, "4": 10, "5": 6, "6": 5, "7": 3, "8": 1, "9": 1}` | P-SoM | `{"0": 88, "1": 62, "11": 1, "2": 28, "3": 17, "4": 11, "5": 8, "6": 3, "7": 2, "8": 2, "9": 2}` |

## Appendix B: E2 first-divergence histograms

The histogram key is first divergent step. Step 0 means the two modes start from different URL signatures or immediately navigate differently.

| site | contrast | histogram |
|---|---|---|
| reddit | DOM_vs_P-text | `{"0": 14, "3": 1, "8": 1}` |
| reddit | P-text_vs_P-SoM | `{"0": 16, "1": 3, "2": 2}` |
| reddit | P-SoM_vs_SoM | `{"0": 12, "1": 4, "2": 2, "3": 1, "6": 1}` |
| reddit | DOM_vs_P-SoM | `{"0": 14, "1": 6, "2": 1}` |
| reddit | DOM_vs_P-prompt | `{"0": 8, "1": 6, "2": 1, "3": 1, "5": 1, "6": 1}` |
| reddit | P-prompt_vs_P-SoM | `{"0": 15, "1": 3, "2": 1}` |
| classifieds | DOM_vs_P-text | `{"0": 6, "1": 10, "2": 3, "3": 2, "8": 1}` |
| classifieds | P-text_vs_P-SoM | `{"0": 3, "1": 5, "2": 4, "3": 1, "4": 1, "7": 2, "8": 2}` |
| classifieds | P-SoM_vs_SoM | `{"0": 14, "1": 19, "2": 8, "3": 2, "4": 1, "5": 1, "8": 1}` |
| classifieds | DOM_vs_P-SoM | `{"0": 6, "1": 8, "2": 6, "3": 1, "7": 1}` |
| classifieds | DOM_vs_P-prompt | `{"0": 9, "1": 12, "10": 1, "2": 3, "3": 4, "5": 1, "6": 1, "7": 1, "8": 1}` |
| classifieds | P-prompt_vs_P-SoM | `{"0": 6, "1": 9, "2": 3, "3": 3, "4": 1, "9": 1}` |

## Appendix C: E3 source provenance

Each E3 row is sourced from existing analyzer outputs; calibration source is null when the run exposes AUROC but not per-mode ECE/MCE/Brier.

| cell | AUROC source | calibration source | token n | verbal n | behavioral n |
|---|---|---|---:|---:|---:|
| B0/classifieds/DOM | `results/visualwebarena/phase1/B0_dom_classifieds_20260525_194618_553890342_530647_R21557/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B0/classifieds/P-SoM | `results/visualwebarena/phase1/B0_phantom_som_classifieds_20260527_191300_844420226_914570_R32031/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B0/classifieds/P-prompt | `results/visualwebarena/phase1/B0_phantom_prompt_classifieds_20260528_040546_107246795_987141_R14655/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 223 | 224 |
| B0/classifieds/P-text | `results/visualwebarena/phase1/B0_phantom_text_classifieds_20260526_233303_901232655_764510_R31183/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B0/classifieds/SoM | `results/visualwebarena/phase1/B0_som_classifieds_20260526_041601_863239369_602235_R5313/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B0/classifieds/Vision | `results/visualwebarena/phase1/B0_vision_classifieds_20260526_141916_610351680_689390_R32024/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 222 | 224 |
| B0/reddit/DOM | `results/visualwebarena/phase1/B0_dom_reddit_20260625_154833_928747130_2827521_R11344/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B0/reddit/P-SoM | `results/visualwebarena/phase1/B0_phantom_som_reddit_20260701_223127_661875492_3649813_R28173/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B0/reddit/P-prompt | `results/visualwebarena/phase1/B0_phantom_prompt_reddit_20260709/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B0/reddit/P-text | `results/visualwebarena/phase1/B0_phantom_text_reddit_20260629_140253_060787566_3384189_R32139/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 76 | 76 | 76 |
| B0/reddit/SoM | `results/visualwebarena/phase1/B0_som_reddit_20260627_035453_162107997_3024022_R20936/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B0/reddit/Vision | `results/visualwebarena/phase1/B0_vision_reddit_20260628_094255_184327569_3222015_R17559/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B1/classifieds/DOM | `results/visualwebarena/phase1/B1_dom_classifieds_20260603_103630_477435114_112846_R17188/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B1/classifieds/P-SoM | `results/visualwebarena/phase1/B1_phantom_som_classifieds_20260606_165421_042595838_568395_R26199/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B1/classifieds/P-prompt | `results/visualwebarena/phase1/B1_phantom_prompt_classifieds_20260607_135946_736335864_683961_R32516/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B1/classifieds/P-text | `results/visualwebarena/phase1/B1_phantom_text_classifieds_20260605_194554_941872185_431169_R933/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B1/classifieds/SoM | `results/visualwebarena/phase1/B1_som_classifieds_20260604_072456_562166453_226675_R31705/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B1/classifieds/Vision | `results/visualwebarena/phase1/B1_vision_classifieds_20260605_012235_349047872_327631_R28622/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B1/reddit/DOM | `results/visualwebarena/phase1/B1_dom_reddit_20260703/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B1/reddit/P-SoM | `results/visualwebarena/phase1/B1_phantom_som_reddit_20260711/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B1/reddit/P-prompt | `results/visualwebarena/phase1/B1_phantom_prompt_reddit_20260713/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B1/reddit/P-text | `results/visualwebarena/phase1/B1_phantom_text_reddit_20260710/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B1/reddit/SoM | `results/visualwebarena/phase1/B1_som_reddit_20260706/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 205 | 205 | 205 |
| B1/reddit/Vision | `results/visualwebarena/phase1/B1_vision_reddit_20260708_002122_732634080_205180_R16847/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | n/a | n/a | n/a |
| B2/classifieds/DOM | `results/visualwebarena/phase1/B2_dom_classifieds_20260609_214713_553762009_985526_R21521/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B2/classifieds/P-SoM | `results/visualwebarena/phase1/B2_phantom_som_classifieds_20260615_044451_093238285_1626673_R22577/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 222 | 224 |
| B2/classifieds/P-prompt | `results/visualwebarena/phase1/B2_phantom_prompt_classifieds_20260616_142027_795794905_1801050_R10175/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B2/classifieds/P-text | `results/visualwebarena/phase1/B2_phantom_text_classifieds_20260614_020803_377049301_1495224_R14219/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 221 | 224 |
| B2/classifieds/SoM | `results/visualwebarena/phase1/B2_som_classifieds_20260611_210828_923656661_1218867_R3380/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |
| B2/classifieds/Vision | `results/visualwebarena/phase1/B2_vision_classifieds_20260612_221910_098760264_1351451_R9288/analysis/signals/combined/tables/auroc_all_metrics.csv` | n/a | 224 | 224 | 224 |

## Appendix D: E4 full action-shift matrix

All values are paired per-task action-fraction shifts in the cascade direction, right-minus-left.

| site | axis | click | type | scroll | select | wait | back | forward | finish | tab_focus | other |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| reddit | axis_1_text | -0.016 | 0.007 | -0.049 | -0.001 | 0.000 | 0.007 | 0.000 | -0.028 | 0.077 | 0.003 |
| reddit | axis_2_prompt | 0.014 | -0.018 | 0.007 | 0.004 | 0.002 | -0.004 | -0.000 | -0.002 | -0.000 | -0.003 |
| reddit | axis_3_image | 0.023 | -0.017 | -0.026 | 0.014 | -0.000 | -0.009 | 0.000 | 0.035 | -0.017 | -0.003 |
| reddit | compound_DOM_to_PSoM | -0.002 | -0.011 | -0.043 | 0.003 | 0.003 | 0.003 | 0.000 | -0.030 | 0.077 | 0.000 |
| reddit | axis_2_prompt_alt | 0.018 | -0.007 | -0.038 | -0.002 | 0.003 | 0.023 | 0.000 | -0.006 | 0.003 | 0.007 |
| reddit | axis_1_text_alt | -0.019 | -0.004 | -0.005 | 0.005 | 0.000 | -0.020 | 0.000 | -0.025 | 0.074 | -0.007 |
| classifieds | axis_1_text | -0.000 | -0.031 | 0.013 | 0.011 | -0.001 | 0.013 | 0.000 | -0.004 | -0.010 | 0.009 |
| classifieds | axis_2_prompt | -0.009 | 0.007 | 0.012 | -0.005 | 0.000 | -0.011 | 0.000 | 0.009 | 0.001 | -0.005 |
| classifieds | axis_3_image | 0.007 | 0.004 | -0.054 | 0.018 | 0.002 | -0.016 | 0.000 | 0.034 | -0.002 | 0.007 |
| classifieds | compound_DOM_to_PSoM | -0.010 | -0.024 | 0.025 | 0.006 | -0.000 | 0.002 | 0.000 | 0.005 | -0.010 | 0.005 |
| classifieds | axis_2_prompt_alt | 0.006 | -0.016 | -0.009 | 0.008 | 0.000 | -0.001 | 0.000 | 0.015 | -0.002 | -0.000 |
| classifieds | axis_1_text_alt | -0.016 | -0.008 | 0.034 | -0.002 | -0.000 | 0.003 | 0.000 | -0.009 | -0.008 | 0.005 |

## Appendix E: E4 ranked per-cell actions

Top action types per B0 cell by pooled step fraction.

| cell | rank | action_type | fraction |
|---|---:|---|---:|
| B0/classifieds/DOM | 1 | click | 0.306 |
| B0/classifieds/DOM | 2 | type | 0.275 |
| B0/classifieds/DOM | 3 | scroll | 0.205 |
| B0/classifieds/DOM | 4 | back | 0.085 |
| B0/classifieds/DOM | 5 | select_option | 0.064 |
| B0/classifieds/P-SoM | 1 | click | 0.313 |
| B0/classifieds/P-SoM | 2 | scroll | 0.258 |
| B0/classifieds/P-SoM | 3 | type | 0.218 |
| B0/classifieds/P-SoM | 4 | back | 0.091 |
| B0/classifieds/P-SoM | 5 | select_option | 0.062 |
| B0/classifieds/P-prompt | 1 | click | 0.325 |
| B0/classifieds/P-prompt | 2 | type | 0.252 |
| B0/classifieds/P-prompt | 3 | scroll | 0.206 |
| B0/classifieds/P-prompt | 4 | back | 0.087 |
| B0/classifieds/P-prompt | 5 | select_option | 0.061 |
| B0/classifieds/P-text | 1 | click | 0.311 |
| B0/classifieds/P-text | 2 | type | 0.228 |
| B0/classifieds/P-text | 3 | scroll | 0.214 |
| B0/classifieds/P-text | 4 | back | 0.108 |
| B0/classifieds/P-text | 5 | select_option | 0.074 |
| B0/classifieds/SoM | 1 | click | 0.312 |
| B0/classifieds/SoM | 2 | type | 0.252 |
| B0/classifieds/SoM | 3 | scroll | 0.170 |
| B0/classifieds/SoM | 4 | select_option | 0.101 |
| B0/classifieds/SoM | 5 | back | 0.079 |
| B0/classifieds/Vision | 1 | click | 0.353 |
| B0/classifieds/Vision | 2 | scroll | 0.295 |
| B0/classifieds/Vision | 3 | type | 0.164 |
| B0/classifieds/Vision | 4 | select_option | 0.080 |
| B0/classifieds/Vision | 5 | back | 0.051 |
| B0/reddit/DOM | 1 | click | 0.465 |
| B0/reddit/DOM | 2 | scroll | 0.191 |
| B0/reddit/DOM | 3 | type | 0.149 |
| B0/reddit/DOM | 4 | back | 0.080 |
| B0/reddit/DOM | 5 | tab_focus | 0.042 |
| B0/reddit/P-SoM | 1 | click | 0.442 |
| B0/reddit/P-SoM | 2 | tab_focus | 0.152 |
| B0/reddit/P-SoM | 3 | type | 0.132 |
| B0/reddit/P-SoM | 4 | scroll | 0.126 |
| B0/reddit/P-SoM | 5 | back | 0.074 |
| B0/reddit/P-prompt | 1 | click | 0.466 |
| B0/reddit/P-prompt | 2 | type | 0.153 |
| B0/reddit/P-prompt | 3 | scroll | 0.149 |
| B0/reddit/P-prompt | 4 | back | 0.106 |
| B0/reddit/P-prompt | 5 | tab_focus | 0.038 |
| B0/reddit/P-text | 1 | click | 0.432 |
| B0/reddit/P-text | 2 | type | 0.150 |
| B0/reddit/P-text | 3 | tab_focus | 0.150 |
| B0/reddit/P-text | 4 | scroll | 0.127 |
| B0/reddit/P-text | 5 | back | 0.081 |
| B0/reddit/SoM | 1 | click | 0.483 |
| B0/reddit/SoM | 2 | tab_focus | 0.147 |
| B0/reddit/SoM | 3 | type | 0.115 |
| B0/reddit/SoM | 4 | scroll | 0.098 |
| B0/reddit/SoM | 5 | back | 0.063 |
| B0/reddit/Vision | 1 | scroll | 0.368 |
| B0/reddit/Vision | 2 | click | 0.319 |
| B0/reddit/Vision | 3 | tab_focus | 0.138 |
| B0/reddit/Vision | 4 | type | 0.072 |
| B0/reddit/Vision | 5 | back | 0.045 |

## Appendix F: validation detail

| validation check | value | pass |
|---|---|---|
| axis1 N reddit | 205 / 205 | True |
| axis1 N classifieds | 224 / 224 | True |
| E1 Jaccard range classifieds/axis_1_text | [0, 1] | True |
| E1 Jaccard range classifieds/axis_1_text_alt | [0, 1] | True |
| E1 Jaccard range classifieds/axis_2_prompt | [0, 1] | True |
| E1 Jaccard range classifieds/axis_2_prompt_alt | [0, 1] | True |
| E1 Jaccard range classifieds/axis_3_image | [0, 1] | True |
| E1 Jaccard range classifieds/compound_DOM_to_PSoM | [0, 1] | True |
| E1 Jaccard range reddit/axis_1_text | [0, 1] | True |
| E1 Jaccard range reddit/axis_1_text_alt | [0, 1] | True |
| E1 Jaccard range reddit/axis_2_prompt | [0, 1] | True |
| E1 Jaccard range reddit/axis_2_prompt_alt | [0, 1] | True |
| E1 Jaccard range reddit/axis_3_image | [0, 1] | True |
| E1 Jaccard range reddit/compound_DOM_to_PSoM | [0, 1] | True |
| E2 first step range classifieds/DOM_vs_P-SoM | [0, 30] | True |
| E2 first step range classifieds/DOM_vs_P-prompt | [0, 30] | True |
| E2 first step range classifieds/DOM_vs_P-text | [0, 30] | True |
| E2 first step range classifieds/P-SoM_vs_SoM | [0, 30] | True |
| E2 first step range classifieds/P-prompt_vs_P-SoM | [0, 30] | True |
| E2 first step range classifieds/P-text_vs_P-SoM | [0, 30] | True |
| E2 first step range reddit/DOM_vs_P-SoM | [0, 30] | True |
| E2 first step range reddit/DOM_vs_P-prompt | [0, 30] | True |
| E2 first step range reddit/DOM_vs_P-text | [0, 30] | True |
| E2 first step range reddit/P-SoM_vs_SoM | [0, 30] | True |
| E2 first step range reddit/P-prompt_vs_P-SoM | [0, 30] | True |
| E2 first step range reddit/P-text_vs_P-SoM | [0, 30] | True |
| E4 action sum B0/classifieds/DOM | 1.000000 | True |
| E4 action sum B0/classifieds/P-SoM | 1.000000 | True |
| E4 action sum B0/classifieds/P-prompt | 1.000000 | True |
| E4 action sum B0/classifieds/P-text | 1.000000 | True |
| E4 action sum B0/classifieds/SoM | 1.000000 | True |
| E4 action sum B0/classifieds/Vision | 1.000000 | True |
| E4 action sum B0/reddit/DOM | 1.000000 | True |
| E4 action sum B0/reddit/P-SoM | 1.000000 | True |
| E4 action sum B0/reddit/P-prompt | 1.000000 | True |
| E4 action sum B0/reddit/P-text | 1.000000 | True |
| E4 action sum B0/reddit/SoM | 1.000000 | True |
| E4 action sum B0/reddit/Vision | 1.000000 | True |
| E1_any_click_divergence_gt_0.1 | threshold > 0.1 | True |
| E2_any_early_or_late_rate_gt_0.1 | threshold > 0.1 | True |
| E3_any_AUROC_effect_gt_0.1 | threshold > 0.1 | True |
| E4_any_action_shift_gt_0.1 | threshold > 0.1 | False |

## Validation

Overall pass: False.

| check | result |
|---|---|
| E1 N reddit | 205 / 205 |
| E1 N classifieds | 224 / 224 |
| E3 cells | 30 / 30 |
| P-prompt status | {"B0_phantom_prompt_classifieds_20260528_040546_107246795_987141_R14655": {"episodes": 224, "policy": "P-prompt is not included in E1-E4 contrasts.", "status": "complete but excluded by design"}, "B0_phantom_prompt_reddit_20260709": {"episodes": 205, "policy": "P-prompt is not included in E1-E4 contrasts.", "status": "complete but excluded by design"}, "B1_phantom_prompt_classifieds_20260607_135946_736335864_683961_R32516": {"episodes": 224, "policy": "P-prompt is not included in E1-E4 contrasts.", "status": "complete but excluded by design"}, "B1_phantom_prompt_reddit_20260713": {"episodes": 205, "policy": "P-prompt is not included in E1-E4 contrasts.", "status": "complete but excluded by design"}, "B2_phantom_prompt_classifieds_20260616_142027_795794905_1801050_R10175": {"episodes": 224, "policy": "P-prompt is not included in E1-E4 contrasts.", "status": "complete but excluded by design"}, "B2_phantom_prompt_reddit_20260723": {"episodes": 205, "policy": "P-prompt is not included in E1-E4 contrasts.", "status": "complete but excluded by design"}} |
