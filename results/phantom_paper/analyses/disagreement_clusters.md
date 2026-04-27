# Disagreement Failure Clusters

Generated: 2026-04-27 18:08 UTC

Scope: B0 VWA `classifieds` and `reddit` only. Shopping data was not read. The primary set is the four-arm phantom-paper comparison: DOM, SoM, Vision, and Phantom-SoM. A short reddit Phantom-DOM ablation snapshot is included separately because that run is active/partial.

Definition: a disagreement task has at least one adjusted-success mode and at least one adjusted-failure mode on the same task. The cluster counts below are over the failure-side `(task, mode)` pairs for those disagreement tasks.

Important trace note: completed Phantom-SoM runs currently have summary JSON under `.bak_pre_rederive` but no `*_steps_v2.jsonl` traces in this checkout, so Phantom-SoM failure pairs are counted in the success-vector disagreement set but cannot receive P1-P14 per-step hits here.

## Inputs

| Site | Mode | Run dir | Condition | episodes scanned by P1-P14 | episodes with hits |
|---|---|---|---|---:|---:|
| classifieds | DOM | `results/visualwebarena/phase1/B0_3mode_classifieds_20260413` | `phase1_dom_router_0` | 234 | 187 |
| classifieds | SoM | `results/visualwebarena/phase1/B0_3mode_classifieds_20260413` | `phase1_som_router_0` | 234 | 102 |
| classifieds | Vision | `results/visualwebarena/phase1/B0_3mode_classifieds_20260413` | `phase1_vision_router_0` | 234 | 189 |
| classifieds | Phantom-SoM | `results/visualwebarena/phase1/B0_phantom_classifieds_20260426` | `phase1_phantom_som_router_0` | 0 | 0 |
| reddit | DOM | `results/visualwebarena/phase1/B0_3mode_reddit_20260422` | `phase1_dom_router_0` | 210 | 156 |
| reddit | SoM | `results/visualwebarena/phase1/B0_3mode_reddit_20260422` | `phase1_som_router_0` | 210 | 113 |
| reddit | Vision | `results/visualwebarena/phase1/B0_3mode_reddit_20260422` | `phase1_vision_router_0` | 210 | 153 |
| reddit | Phantom-SoM | `results/visualwebarena/phase1/run_reddit_1777238854_ef9c4b` | `phase1_phantom_som_router_0` | 0 | 0 |
| reddit | Phantom-DOM | `results/visualwebarena/phase1/B0_phantom_dom_reddit_20260427` | `phase1_phantom_dom_router_0` | 59 | 46 |

## Primary Summary

| Site | same-task N | disagreement tasks | failure-side pairs | pairs with step trace | no-trace pairs |
|---|---:|---:|---:|---:|---:|
| classifieds | 234 | 69 | 165 | 116 | 49 |
| reddit | 210 | 42 | 89 | 70 | 19 |
| **Total** | 444 | 111 | 254 | 186 | 68 |

## P1-P14 Distribution

Counts are over failure-side pairs with step traces. `pair_count` counts a pair once per rule; `hit_count` counts all step-level hits.

| Rule | pair_count | hit_count |
|---|---:|---:|
| P1 | 20 | 57 |
| P2 | 2 | 2 |
| P4 | 1 | 1 |
| P5 | 48 | 48 |
| P6 | 32 | 32 |
| P7 | 1 | 1 |
| P10 | 3 | 3 |
| P12 | 2 | 2 |
| P13 | 12 | 12 |
| P14 | 83 | 120 |

## High-Level Failure Clusters

| Cluster | Label | total | classifieds | reddit |
|---|---|---:|---:|---:|
| `no-step-trace` | Summary only; step trace unavailable | 68 | 49 | 19 |
| `navigation-loop` | Navigation/self-loop after reaching or choosing a page | 63 | 41 | 22 |
| `premature-or-wrong-finish` | Premature finish / wrong answer / eval mismatch | 49 | 31 | 18 |
| `visual-info-missing` | Visual information missing from the representation | 31 | 19 | 12 |
| `element-targeting` | Element grounding / wrong target selection | 23 | 20 | 3 |
| `unmatched-stuck` | Stuck/no-progress without P-rule hit | 10 | 1 | 9 |
| `no-hard-rule-hit` | No P1-P14 hard-rule hit | 4 | 2 | 2 |
| `search-browse-strategy` | Search-over-browse or insufficient exploration | 3 | 0 | 3 |
| `answer-memory` | Answer synthesis or numeric memory failure | 1 | 1 | 0 |
| `benchmark-url-quirk` | Benchmark/API URL quirk | 1 | 1 | 0 |
| `max-step-exhaustion` | Max-step exhaustion without specific P-rule | 1 | 0 | 1 |

## Mode Exposure

| Site | Mode | failure-side disagreement pairs |
|---|---|---:|
| classifieds | DOM | 44 |
| classifieds | Phantom-SoM | 49 |
| classifieds | SoM | 27 |
| classifieds | Vision | 45 |
| reddit | DOM | 22 |
| reddit | Phantom-SoM | 19 |
| reddit | SoM | 20 |
| reddit | Vision | 28 |

## Reading

- The largest hard-rule bucket is navigation/self-loop (`P5`/`P14`), especially on reddit: agents often commit to a page/post and continue acting on the same URL even when another mode found the correct target.
- `P6` marks genuine representation mismatch: DOM can be inside a disagreement task but still lack necessary color/image evidence. Treat this as observation-limit evidence, not an infrastructure bug.
- A large `premature-or-wrong-finish` residue remains outside P1-P14. Those pairs usually fail by answering/finishing from an incomplete page state rather than by a detectable click/scroll pathology.
- Phantom-SoM contributes to the disagreement set, but its completed step traces are not present in this checkout. Restore the cleared `*_steps_v2.jsonl` files if per-step Phantom-SoM clusters are needed.

## Classifieds Details

| Cluster | count | top modes | example tasks |
|---|---:|---|---|
| `no-step-trace` | 49 | Phantom-SoM:49 | 5, 11, 14, 16, 24, 40, 49, 52 |
| `navigation-loop` | 41 | Vision:19, DOM:12, SoM:10 | 5, 13, 16, 25, 40, 45, 46, 52 |
| `premature-or-wrong-finish` | 31 | SoM:14, DOM:10, Vision:7 | 11, 14, 14, 16, 25, 40, 50, 60 |
| `element-targeting` | 20 | Vision:17, DOM:2, SoM:1 | 10, 13, 15, 17, 44, 49, 50, 93 |
| `visual-info-missing` | 19 | DOM:19 | 11, 24, 49, 52, 60, 61, 62, 64 |
| `no-hard-rule-hit` | 2 | SoM:1, Vision:1 | 167, 167 |
| `answer-memory` | 1 | SoM:1 | 93 |
| `unmatched-stuck` | 1 | DOM:1 | 130 |
| `benchmark-url-quirk` | 1 | Vision:1 | 196 |

## Reddit Details

| Cluster | count | top modes | example tasks |
|---|---:|---|---|
| `navigation-loop` | 22 | Vision:12, SoM:6, DOM:4 | 0, 2, 4, 7, 19, 26, 79, 79 |
| `no-step-trace` | 19 | Phantom-SoM:19 | 2, 4, 14, 69, 79, 81, 120, 129 |
| `premature-or-wrong-finish` | 18 | SoM:8, Vision:6, DOM:4 | 58, 69, 69, 72, 81, 94, 94, 94 |
| `visual-info-missing` | 12 | DOM:12 | 2, 4, 7, 14, 26, 107, 131, 139 |
| `unmatched-stuck` | 9 | Vision:4, SoM:3, DOM:2 | 0, 18, 40, 107, 151, 152, 152, 170 |
| `element-targeting` | 3 | Vision:3 | 7, 42, 81 |
| `search-browse-strategy` | 3 | Vision:2, SoM:1 | 14, 162, 162 |
| `no-hard-rule-hit` | 2 | SoM:1, Vision:1 | 31, 31 |
| `max-step-exhaustion` | 1 | SoM:1 | 179 |

## Reddit Phantom-DOM Ablation Snapshot

Partial same-task comparison over DOM, SoM, Phantom-DOM where Phantom-DOM summaries exist: N=59 tasks. Disagreement failure-side pairs: 15. This is a read-only snapshot of an active/partial run, so use it directionally.

| Cluster | count | modes |
|---|---:|---|
| `visual-info-missing` | 5 | DOM:5 |
| `navigation-loop` | 4 | Phantom-DOM:3, SoM:1 |
| `no-hard-rule-hit` | 3 | Phantom-DOM:2, SoM:1 |
| `premature-or-wrong-finish` | 2 | DOM:1, SoM:1 |
| `unmatched-stuck` | 1 | SoM:1 |

## Compact Pair Diagnostics

Success vectors use mode order `DOM SoM Vision Phantom-SoM` for the primary table. `1` means adjusted success; `0` means adjusted failure.

| Site | Task | Mode | Vector | Cluster | P-rules | Reason bucket / note | Intent |
|---|---:|---|---|---|---|---|---|
| classifieds | 5 | SoM | `1010` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Navigate to my listing of the white car and delete it. |
| classifieds | 5 | Phantom-SoM | `1010` | `no-step-trace` | - | step trace unavailable | Navigate to my listing of the white car and delete it. |
| classifieds | 10 | Vision | `1101` | `element-targeting` | P1,P5,P14 | fail_no_progress | What is the seat height in inches of the smaller piece of furniture on this page? |
| classifieds | 11 | DOM | `0010` | `visual-info-missing` | P5,P6,P14 | fail_no_progress | What is the size of the wheels in inches of the first blue bike on this page? |
| classifieds | 11 | SoM | `0010` | `premature-or-wrong-finish` | - | fail_early_finish | What is the size of the wheels in inches of the first blue bike on this page? |
| classifieds | 11 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | What is the size of the wheels in inches of the first blue bike on this page? |
| classifieds | 13 | DOM | `0101` | `navigation-loop` | P14 | fail_finish_claim_missing | What is the color of the most expensive item in the "Boats" category? |
| classifieds | 13 | Vision | `0101` | `element-targeting` | P1,P5,P14 | fail_no_progress | What is the color of the most expensive item in the "Boats" category? |
| classifieds | 14 | DOM | `0100` | `premature-or-wrong-finish` | - | fail_early_finish | What is the email of the seller of the painting in the second row? |
| classifieds | 14 | Vision | `0100` | `premature-or-wrong-finish` | - | fail_early_finish | What is the email of the seller of the painting in the second row? |
| classifieds | 14 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | What is the email of the seller of the painting in the second row? |
| classifieds | 15 | Vision | `1101` | `element-targeting` | P1,P5 | fail_no_progress | What is the email of the seller of the guitar in the red case on this page? |
| classifieds | 16 | DOM | `0010` | `navigation-loop` | P5,P14 | fail_no_progress | What is the email of the seller of the item with the coffee mug in the picture on this page? |
| classifieds | 16 | SoM | `0010` | `premature-or-wrong-finish` | - | fail_early_finish | What is the email of the seller of the item with the coffee mug in the picture on this page? |
| classifieds | 16 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | What is the email of the seller of the item with the coffee mug in the picture on this page? |
| classifieds | 17 | Vision | `1101` | `element-targeting` | P1,P5,P12,P14 | fail_no_progress | Show me the cheapest bike with red handlebars between $900-950. |
| classifieds | 24 | DOM | `0110` | `visual-info-missing` | P5,P6,P14 | success | How many miles does the black truck on this page have? |
| classifieds | 24 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | How many miles does the black truck on this page have? |
| classifieds | 25 | SoM | `1001` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | How many red boats were posted on 16th November 2023? |
| classifieds | 25 | Vision | `1001` | `navigation-loop` | P14 | fail_finish_eval_mismatch | How many red boats were posted on 16th November 2023? |
| classifieds | 40 | DOM | `0010` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Search for "dishwasher" and tell me the brand of the most recent listing of a stainless steel o… |
| classifieds | 40 | SoM | `0010` | `premature-or-wrong-finish` | - | fail_early_finish | Search for "dishwasher" and tell me the brand of the most recent listing of a stainless steel o… |
| classifieds | 40 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Search for "dishwasher" and tell me the brand of the most recent listing of a stainless steel o… |
| classifieds | 44 | Vision | `1101` | `element-targeting` | P1,P5,P14 | fail_no_progress | I recall seeing this exact item on the site, help me find the most recent post of it. |
| classifieds | 45 | Vision | `1101` | `navigation-loop` | P5,P13,P14 | fail_no_progress | I recall seeing this exact item on the site, help me find the most recent post of it. |
| classifieds | 46 | Vision | `1101` | `navigation-loop` | P13,P14 | fail_no_progress | I recall seeing this exact item on the site, help me find the most recent post of it. |
| classifieds | 49 | DOM | `0100` | `visual-info-missing` | P6 | fail_finish_eval_mismatch | How much RAM (in GB) does the item with blue LED lights on this page have? |
| classifieds | 49 | Vision | `0100` | `element-targeting` | P1 | fail_finish_eval_mismatch | How much RAM (in GB) does the item with blue LED lights on this page have? |
| classifieds | 49 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | How much RAM (in GB) does the item with blue LED lights on this page have? |
| classifieds | 50 | SoM | `1001` | `premature-or-wrong-finish` | - | fail_early_finish | What is the email of the seller of the red palette on this page? |
| classifieds | 50 | Vision | `1001` | `element-targeting` | P1,P5 | fail_no_progress | What is the email of the seller of the red palette on this page? |
| classifieds | 52 | DOM | `0100` | `visual-info-missing` | P5,P6,P14 | fail_no_progress | Find me the most recent painting in the "Arts + crafts" category of something that looks close … |
| classifieds | 52 | Vision | `0100` | `navigation-loop` | P14 | fail_incomplete_or_stuck | Find me the most recent painting in the "Arts + crafts" category of something that looks close … |
| classifieds | 52 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Find me the most recent painting in the "Arts + crafts" category of something that looks close … |
| classifieds | 55 | Phantom-SoM | `1110` | `no-step-trace` | - | step trace unavailable | Find the most recently listed item in the "Collectibles" category that depicts the inventors of… |
| classifieds | 60 | DOM | `0001` | `visual-info-missing` | P6,P14 | fail_incomplete_or_stuck | Find the most expensive video game item where I can roleplay the situation in the image. |
| classifieds | 60 | SoM | `0001` | `premature-or-wrong-finish` | - | fail_finish_wrong_url_not_found | Find the most expensive video game item where I can roleplay the situation in the image. |
| classifieds | 60 | Vision | `0001` | `navigation-loop` | P14 | fail_no_progress | Find the most expensive video game item where I can roleplay the situation in the image. |
| classifieds | 61 | DOM | `0010` | `visual-info-missing` | P6 | fail_max_steps_target_unreachable | Find the most expensive video game item where I can roleplay the situation in the image. |
| classifieds | 61 | SoM | `0010` | `navigation-loop` | P13,P14 | fail_no_progress | Find the most expensive video game item where I can roleplay the situation in the image. |
| classifieds | 61 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Find the most expensive video game item where I can roleplay the situation in the image. |
| classifieds | 62 | DOM | `0101` | `visual-info-missing` | P5,P6,P14 | fail_no_progress | Find the most expensive video game item with the character from the image on its display. |
| classifieds | 62 | Vision | `0101` | `navigation-loop` | P14 | fail_finish_wrong_url_not_found | Find the most expensive video game item with the character from the image on its display. |
| classifieds | 63 | Vision | `1100` | `navigation-loop` | P5,P13,P14 | fail_no_progress | Find the most expensive video game item with the character from the image on its display. |
| classifieds | 63 | Phantom-SoM | `1100` | `no-step-trace` | - | step trace unavailable | Find the most expensive video game item with the character from the image on its display. |
| classifieds | 64 | DOM | `0001` | `visual-info-missing` | P5,P6,P13,P14 | fail_no_progress | Find the video game item that costs exactly $500 where I can roleplay the situation in the imag… |
| classifieds | 64 | SoM | `0001` | `navigation-loop` | P14 | fail_max_steps_target_unreachable | Find the video game item that costs exactly $500 where I can roleplay the situation in the imag… |
| classifieds | 64 | Vision | `0001` | `navigation-loop` | P5,P13,P14 | fail_no_progress | Find the video game item that costs exactly $500 where I can roleplay the situation in the imag… |
| classifieds | 79 | DOM | `0101` | `visual-info-missing` | P5,P6,P14 | fail_no_progress | Find me an item on this page that references the person in the image. |
| classifieds | 79 | Vision | `0101` | `navigation-loop` | P5,P14 | fail_no_progress | Find me an item on this page that references the person in the image. |
| classifieds | 93 | DOM | `0001` | `visual-info-missing` | P6,P10 | fail_finish_wrong_url_left_target | Find the electronics item on the page which is compatible with this image. |
| classifieds | 93 | SoM | `0001` | `answer-memory` | P10 | fail_finish_wrong_url_left_target | Find the electronics item on the page which is compatible with this image. |
| classifieds | 93 | Vision | `0001` | `element-targeting` | P1,P14 | fail_finish_wrong_url_not_found | Find the electronics item on the page which is compatible with this image. |
| classifieds | 94 | Vision | `1100` | `element-targeting` | P1,P5 | fail_no_progress | Find the animal on this page that has someone riding it in the image. |
| classifieds | 94 | Phantom-SoM | `1100` | `no-step-trace` | - | step trace unavailable | Find the animal on this page that has someone riding it in the image. |
| classifieds | 98 | SoM | `1000` | `premature-or-wrong-finish` | - | fail_early_finish | How many hours are on the engine of the most recently listed red boat? |
| classifieds | 98 | Vision | `1000` | `element-targeting` | P1,P5,P14 | fail_no_progress | How many hours are on the engine of the most recently listed red boat? |
| classifieds | 98 | Phantom-SoM | `1000` | `no-step-trace` | - | step trace unavailable | How many hours are on the engine of the most recently listed red boat? |
| classifieds | 101 | DOM | `0100` | `visual-info-missing` | P6 | fail_finish_wrong_url_not_found | Explore the "Art + crafts" category, and find the most expensive painting of the city in this i… |
| classifieds | 101 | Vision | `0100` | `navigation-loop` | P14 | fail_no_progress | Explore the "Art + crafts" category, and find the most expensive painting of the city in this i… |
| classifieds | 101 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Explore the "Art + crafts" category, and find the most expensive painting of the city in this i… |
| classifieds | 106 | DOM | `0100` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | What is the email of the seller of the most expensive item in Photo + videos that has an animal… |
| classifieds | 106 | Vision | `0100` | `element-targeting` | P1,P5,P12,P14 | fail_no_progress | What is the email of the seller of the most expensive item in Photo + videos that has an animal… |
| classifieds | 106 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | What is the email of the seller of the most expensive item in Photo + videos that has an animal… |
| classifieds | 110 | Vision | `1101` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Search for "mario kart" and tell me how many games are in the image of the most recently listed… |
| classifieds | 111 | DOM | `0100` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Search for "hockey" and tell me the team name of the jersey on the most recently listed item. |
| classifieds | 111 | Vision | `0100` | `premature-or-wrong-finish` | - | fail_early_finish | Search for "hockey" and tell me the team name of the jersey on the most recently listed item. |
| classifieds | 111 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Search for "hockey" and tell me the team name of the jersey on the most recently listed item. |
| classifieds | 112 | DOM | `0010` | `navigation-loop` | P5,P14 | fail_no_progress | Search for "basketball" and navigate to the cheapest item that has a man in a suit in its image. |
| classifieds | 112 | SoM | `0010` | `premature-or-wrong-finish` | - | fail_early_finish | Search for "basketball" and navigate to the cheapest item that has a man in a suit in its image. |
| classifieds | 112 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Search for "basketball" and navigate to the cheapest item that has a man in a suit in its image. |
| classifieds | 115 | DOM | `0100` | `element-targeting` | P2,P6,P14 | fail_max_steps_target_unreachable | Search for "brace" and navigate to the most recently listed item that matches the body part in … |
| classifieds | 115 | Vision | `0100` | `element-targeting` | P1,P14 | fail_no_progress | Search for "brace" and navigate to the most recently listed item that matches the body part in … |
| classifieds | 115 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Search for "brace" and navigate to the most recently listed item that matches the body part in … |
| classifieds | 116 | SoM | `1010` | `navigation-loop` | P14 | fail_finish_wrong_url_not_found | Search for "brace" and navigate to the cheapest item that matches the body part in this image. |
| classifieds | 116 | Phantom-SoM | `1010` | `no-step-trace` | - | step trace unavailable | Search for "brace" and navigate to the cheapest item that matches the body part in this image. |
| classifieds | 120 | DOM | `0100` | `visual-info-missing` | P5,P6 | fail_no_progress | What is written on the down tube of the bike with the purple frame on this page? |
| classifieds | 120 | Vision | `0100` | `premature-or-wrong-finish` | - | fail_early_finish | What is written on the down tube of the bike with the purple frame on this page? |
| classifieds | 120 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | What is written on the down tube of the bike with the purple frame on this page? |
| classifieds | 124 | DOM | `0110` | `navigation-loop` | P5 | fail_no_progress | Navigate to the item on this page whose image is set on grass. |
| classifieds | 124 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | Navigate to the item on this page whose image is set on grass. |
| classifieds | 125 | DOM | `0110` | `element-targeting` | P4 | fail_finish_empty_answer | Navigate to the item on this page whose image is a desktop screenshot. |
| classifieds | 125 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | Navigate to the item on this page whose image is a desktop screenshot. |
| classifieds | 127 | DOM | `0100` | `navigation-loop` | P10,P14 | fail_finish_wrong_url_not_found | Search for "MCAT" and navigate to the prep book that has 2020-2021 on the cover. |
| classifieds | 127 | Vision | `0100` | `element-targeting` | P1 | fail_finish_claim_missing | Search for "MCAT" and navigate to the prep book that has 2020-2021 on the cover. |
| classifieds | 127 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Search for "MCAT" and navigate to the prep book that has 2020-2021 on the cover. |
| classifieds | 130 | DOM | `0100` | `unmatched-stuck` | - | fail_max_steps_click_back_loop | Navigate to the item on this page whose image is taken during a sunset. |
| classifieds | 130 | Vision | `0100` | `element-targeting` | P1 | fail_no_progress | Navigate to the item on this page whose image is taken during a sunset. |
| classifieds | 130 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Navigate to the item on this page whose image is taken during a sunset. |
| classifieds | 132 | DOM | `0100` | `visual-info-missing` | P6 | fail_max_steps_click_back_loop | Navigate to the item on this page whose image has 3 black animals. |
| classifieds | 132 | Vision | `0100` | `element-targeting` | P1 | fail_finish_empty_answer | Navigate to the item on this page whose image has 3 black animals. |
| classifieds | 132 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Navigate to the item on this page whose image has 3 black animals. |
| classifieds | 139 | SoM | `1010` | `navigation-loop` | P14 | fail_max_steps_target_unreachable | I recall seeing this exact item on the site, help me find the most recent post of it. I recall … |
| classifieds | 139 | Phantom-SoM | `1010` | `no-step-trace` | - | step trace unavailable | I recall seeing this exact item on the site, help me find the most recent post of it. I recall … |
| classifieds | 149 | DOM | `0100` | `navigation-loop` | P5 | fail_no_progress | For the item with a gold colored controller, tell me the number of games displayed in the image. |
| classifieds | 149 | Vision | `0100` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | For the item with a gold colored controller, tell me the number of games displayed in the image. |
| classifieds | 149 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | For the item with a gold colored controller, tell me the number of games displayed in the image. |
| classifieds | 151 | DOM | `0110` | `visual-info-missing` | P5,P6 | fail_no_progress | Navigate to the item on this page whose image has pink sticky notes. |
| classifieds | 151 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | Navigate to the item on this page whose image has pink sticky notes. |
| classifieds | 152 | DOM | `0010` | `navigation-loop` | P5 | fail_no_progress | Navigate to the item on this page whose image includes a human hand. |
| classifieds | 152 | SoM | `0010` | `navigation-loop` | P14 | fail_finish_wrong_url_not_found | Navigate to the item on this page whose image includes a human hand. |
| classifieds | 152 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Navigate to the item on this page whose image includes a human hand. |
| classifieds | 160 | DOM | `0100` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Add a comment on the item on this page whose image is taken inside a garage with the title "Mor… |
| classifieds | 160 | Vision | `0100` | `premature-or-wrong-finish` | - | fail_early_finish | Add a comment on the item on this page whose image is taken inside a garage with the title "Mor… |
| classifieds | 160 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Add a comment on the item on this page whose image is taken inside a garage with the title "Mor… |
| classifieds | 165 | DOM | `0100` | `visual-info-missing` | P6,P13 | fail_max_steps_target_unreachable | Navigate to the listing on this page whose image includes a car the same color as the item in t… |
| classifieds | 165 | Vision | `0100` | `navigation-loop` | P5 | fail_no_progress | Navigate to the listing on this page whose image includes a car the same color as the item in t… |
| classifieds | 165 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Navigate to the listing on this page whose image includes a car the same color as the item in t… |
| classifieds | 166 | DOM | `0100` | `visual-info-missing` | P5,P6 | fail_no_progress | Navigate to the listing on this page whose image includes an instrument the same color as the i… |
| classifieds | 166 | Vision | `0100` | `navigation-loop` | P5 | fail_no_progress | Navigate to the listing on this page whose image includes an instrument the same color as the i… |
| classifieds | 166 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Navigate to the listing on this page whose image includes an instrument the same color as the i… |
| classifieds | 167 | SoM | `1000` | `no-hard-rule-hit` | - | success | Navigate to the listing on this page whose image includes an instrument the same color as the i… |
| classifieds | 167 | Vision | `1000` | `no-hard-rule-hit` | - | success | Navigate to the listing on this page whose image includes an instrument the same color as the i… |
| classifieds | 167 | Phantom-SoM | `1000` | `no-step-trace` | - | step trace unavailable | Navigate to the listing on this page whose image includes an instrument the same color as the i… |
| classifieds | 170 | SoM | `1001` | `navigation-loop` | P14 | fail_max_steps_target_unreachable | Navigate to the most expensive item in the "Cell phones" category which includes in its design … |
| classifieds | 170 | Vision | `1001` | `navigation-loop` | P5,P14 | fail_no_progress | Navigate to the most expensive item in the "Cell phones" category which includes in its design … |
| classifieds | 173 | DOM | `0110` | `premature-or-wrong-finish` | - | fail_early_finish | For the item on this page with a website address in the image, tell me the address. |
| classifieds | 173 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | For the item on this page with a website address in the image, tell me the address. |
| classifieds | 174 | SoM | `1000` | `premature-or-wrong-finish` | - | fail_early_finish | For the item on this page which includes a Black Friday logo in the image, tell me the most spe… |
| classifieds | 174 | Vision | `1000` | `element-targeting` | P1,P5 | fail_no_progress | For the item on this page which includes a Black Friday logo in the image, tell me the most spe… |
| classifieds | 174 | Phantom-SoM | `1000` | `no-step-trace` | - | step trace unavailable | For the item on this page which includes a Black Friday logo in the image, tell me the most spe… |
| classifieds | 183 | DOM | `0011` | `visual-info-missing` | P5,P6,P14 | fail_no_progress | Navigate to the most recently listed item in the "Collectibles" category with the same characte… |
| classifieds | 183 | SoM | `0011` | `navigation-loop` | P14 | fail_incomplete_or_stuck | Navigate to the most recently listed item in the "Collectibles" category with the same characte… |
| classifieds | 184 | SoM | `1000` | `navigation-loop` | P14 | fail_max_steps | Navigate to the most expensive item in the "Collectibles" category with the same exact item fro… |
| classifieds | 184 | Vision | `1000` | `navigation-loop` | P14 | fail_max_steps | Navigate to the most expensive item in the "Collectibles" category with the same exact item fro… |
| classifieds | 184 | Phantom-SoM | `1000` | `no-step-trace` | - | step trace unavailable | Navigate to the most expensive item in the "Collectibles" category with the same exact item fro… |
| classifieds | 187 | DOM | `0100` | `premature-or-wrong-finish` | - | fail_finish_wrong_url_not_found | Navigate to the item on the page where Lightning McQueen is in the image. |
| classifieds | 187 | Vision | `0100` | `element-targeting` | P1 | fail_no_progress | Navigate to the item on the page where Lightning McQueen is in the image. |
| classifieds | 187 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Navigate to the item on the page where Lightning McQueen is in the image. |
| classifieds | 189 | DOM | `0011` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Navigate to the item on the page where Mario is in the image. |
| classifieds | 189 | SoM | `0011` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Navigate to the item on the page where Mario is in the image. |
| classifieds | 191 | SoM | `1010` | `element-targeting` | P2 | success | Navigate to the item on this page with a blonde horse. |
| classifieds | 191 | Phantom-SoM | `1010` | `no-step-trace` | - | step trace unavailable | Navigate to the item on this page with a blonde horse. |
| classifieds | 192 | DOM | `0010` | `navigation-loop` | P14 | fail_no_progress | For the listing on this page with books of cars, tell me the primary color of the cars. |
| classifieds | 192 | SoM | `0010` | `premature-or-wrong-finish` | - | fail_early_finish | For the listing on this page with books of cars, tell me the primary color of the cars. |
| classifieds | 192 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | For the listing on this page with books of cars, tell me the primary color of the cars. |
| classifieds | 194 | DOM | `0010` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Navigate to the item on the page where Mickey Mouse is in the image. |
| classifieds | 194 | SoM | `0010` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Navigate to the item on the page where Mickey Mouse is in the image. |
| classifieds | 194 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Navigate to the item on the page where Mickey Mouse is in the image. |
| classifieds | 195 | Vision | `1101` | `navigation-loop` | P14 | success | Navigate to the most expensive yellow motorcycle in the "Motorcycles" category that costs at le… |
| classifieds | 196 | Vision | `1100` | `benchmark-url-quirk` | P7 | success | Navigate to the most expensive red truck in the "Cars + trucks" category from Maryland that is … |
| classifieds | 196 | Phantom-SoM | `1100` | `no-step-trace` | - | step trace unavailable | Navigate to the most expensive red truck in the "Cars + trucks" category from Maryland that is … |
| classifieds | 201 | DOM | `0001` | `visual-info-missing` | P5,P6,P14 | fail_no_progress | Show me the latest listing of a snare drum with a black and red color scheme. |
| classifieds | 201 | SoM | `0001` | `premature-or-wrong-finish` | - | fail_finish_wrong_url_not_found | Show me the latest listing of a snare drum with a black and red color scheme. |
| classifieds | 201 | Vision | `0001` | `navigation-loop` | P14 | fail_finish_wrong_url_not_found | Show me the latest listing of a snare drum with a black and red color scheme. |
| classifieds | 209 | DOM | `0100` | `premature-or-wrong-finish` | - | fail_early_finish | What is the cheapest price that I can pay for 31 of these wooden pallets? |
| classifieds | 209 | Vision | `0100` | `premature-or-wrong-finish` | - | fail_early_finish | What is the cheapest price that I can pay for 31 of these wooden pallets? |
| classifieds | 209 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | What is the cheapest price that I can pay for 31 of these wooden pallets? |
| classifieds | 210 | DOM | `0100` | `premature-or-wrong-finish` | - | fail_finish_wrong_url_not_found | Find me the cheapest lamb in the "Farm + garden" category on this site. |
| classifieds | 210 | Vision | `0100` | `navigation-loop` | P14 | fail_incomplete_or_stuck | Find me the cheapest lamb in the "Farm + garden" category on this site. |
| classifieds | 210 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Find me the cheapest lamb in the "Farm + garden" category on this site. |
| classifieds | 214 | Vision | `1100` | `navigation-loop` | P13,P14 | fail_max_steps_target_unreachable | Show me the most expensive phone with a theme matching that of the image. |
| classifieds | 214 | Phantom-SoM | `1100` | `no-step-trace` | - | step trace unavailable | Show me the most expensive phone with a theme matching that of the image. |
| classifieds | 217 | DOM | `0010` | `navigation-loop` | P5,P14 | fail_no_progress | Find me the most recent listing offering a book titled "Captain's Logs: The Complete Trek Voyag… |
| classifieds | 217 | SoM | `0010` | `premature-or-wrong-finish` | - | fail_early_finish | Find me the most recent listing offering a book titled "Captain's Logs: The Complete Trek Voyag… |
| classifieds | 217 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Find me the most recent listing offering a book titled "Captain's Logs: The Complete Trek Voyag… |
| classifieds | 220 | DOM | `0110` | `visual-info-missing` | P6 | fail_finish_eval_mismatch | Find the cheapest red-colored microwave that is $10 and under and still works. |
| classifieds | 220 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | Find the cheapest red-colored microwave that is $10 and under and still works. |
| classifieds | 221 | DOM | `0110` | `premature-or-wrong-finish` | - | fail_early_finish | How many bowls are there in this set of bowls? |
| classifieds | 221 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | How many bowls are there in this set of bowls? |
| classifieds | 222 | SoM | `1000` | `premature-or-wrong-finish` | - | fail_early_finish | Based on the measuring tape in the image, is the stated diameter of this item correct? |
| classifieds | 222 | Vision | `1000` | `premature-or-wrong-finish` | - | fail_early_finish | Based on the measuring tape in the image, is the stated diameter of this item correct? |
| classifieds | 222 | Phantom-SoM | `1000` | `no-step-trace` | - | step trace unavailable | Based on the measuring tape in the image, is the stated diameter of this item correct? |
| classifieds | 233 | DOM | `0110` | `navigation-loop` | P14 | fail_max_steps_search_repeat | Are the characters in the image on Reddit shown in the listing on the classifieds site? If they… |
| classifieds | 233 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | Are the characters in the image on Reddit shown in the listing on the classifieds site? If they… |
| reddit | 0 | SoM | `1001` | `unmatched-stuck` | - | fail_max_steps_click_back_loop | I'm trying to find this post. Navigate to the comment section for it. |
| reddit | 0 | Vision | `1001` | `navigation-loop` | P5,P14 | fail_no_progress | I'm trying to find this post. Navigate to the comment section for it. |
| reddit | 2 | DOM | `0100` | `visual-info-missing` | P6 | fail_finish_wrong_url_not_found | I'm trying to find this post. Navigate to the comment section for it. |
| reddit | 2 | Vision | `0100` | `navigation-loop` | P14 | fail_incomplete_or_stuck | I'm trying to find this post. Navigate to the comment section for it. |
| reddit | 2 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | I'm trying to find this post. Navigate to the comment section for it. |
| reddit | 4 | DOM | `0100` | `visual-info-missing` | P5,P6,P14 | fail_no_progress | I'm trying to do some research about the post with this image. Tell me how many comments mentio… |
| reddit | 4 | Vision | `0100` | `navigation-loop` | P5,P14 | fail_no_progress | I'm trying to do some research about the post with this image. Tell me how many comments mentio… |
| reddit | 4 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | I'm trying to do some research about the post with this image. Tell me how many comments mentio… |
| reddit | 7 | DOM | `0001` | `visual-info-missing` | P6 | success | I'm doing some research about the post with this image. Navigate to the comment url of the reci… |
| reddit | 7 | SoM | `0001` | `navigation-loop` | P14 | success | I'm doing some research about the post with this image. Navigate to the comment url of the reci… |
| reddit | 7 | Vision | `0001` | `element-targeting` | P1,P5,P14 | success | I'm doing some research about the post with this image. Navigate to the comment url of the reci… |
| reddit | 14 | DOM | `0100` | `visual-info-missing` | P6,P14 | fail_incomplete_or_stuck | Can you help me find and navigate to the comments section of an image post that has something l… |
| reddit | 14 | Vision | `0100` | `search-browse-strategy` | P13 | fail_incomplete_or_stuck | Can you help me find and navigate to the comments section of an image post that has something l… |
| reddit | 14 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Can you help me find and navigate to the comments section of an image post that has something l… |
| reddit | 18 | Vision | `1101` | `unmatched-stuck` | - | fail_no_progress | Navigate to a subreddit where I can find the most similar content to this image. |
| reddit | 19 | Vision | `1101` | `navigation-loop` | P14 | fail_no_progress | Find and upvote this post. I think it was posted in /f/GetMotivated. |
| reddit | 26 | DOM | `0101` | `visual-info-missing` | P5,P6,P14 | success | Find this post and tell me what the top comment says. |
| reddit | 26 | Vision | `0101` | `navigation-loop` | P5,P14 | success | Find this post and tell me what the top comment says. |
| reddit | 31 | SoM | `1001` | `no-hard-rule-hit` | - | success | Find the user who posted this and navigate to the comments section of one of their other posts. |
| reddit | 31 | Vision | `1001` | `no-hard-rule-hit` | - | success | Find the user who posted this and navigate to the comments section of one of their other posts. |
| reddit | 40 | Vision | `1101` | `unmatched-stuck` | - | fail_no_progress | I'd like to find the subreddit for the city this photo was taken in. Can you navigate to it? |
| reddit | 42 | Vision | `1101` | `element-targeting` | P1 | fail_finish_wrong_url_not_found | Navigate to the comments section of the first image post in /f/MechanicalKeyboards as sorted by… |
| reddit | 58 | Vision | `1101` | `premature-or-wrong-finish` | - | fail_early_finish | Who is the author of the most popular novel adapted anime in year 2012? |
| reddit | 69 | SoM | `1000` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Make a comment in this post explaining what the picture is about. |
| reddit | 69 | Vision | `1000` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Make a comment in this post explaining what the picture is about. |
| reddit | 69 | Phantom-SoM | `1000` | `no-step-trace` | - | step trace unavailable | Make a comment in this post explaining what the picture is about. |
| reddit | 72 | Vision | `1101` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Make a comment in this post explaining what the picture is about. |
| reddit | 79 | SoM | `1000` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Navigate to the comments section of any post that contains Rick and Morty in the picture. |
| reddit | 79 | Vision | `1000` | `navigation-loop` | P14 | fail_finish_empty_answer | Navigate to the comments section of any post that contains Rick and Morty in the picture. |
| reddit | 79 | Phantom-SoM | `1000` | `no-step-trace` | - | step trace unavailable | Navigate to the comments section of any post that contains Rick and Morty in the picture. |
| reddit | 81 | SoM | `1000` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Upvote all the posts that have a picture which contains cat on this page. |
| reddit | 81 | Vision | `1000` | `element-targeting` | P1 | fail_no_progress | Upvote all the posts that have a picture which contains cat on this page. |
| reddit | 81 | Phantom-SoM | `1000` | `no-step-trace` | - | step trace unavailable | Upvote all the posts that have a picture which contains cat on this page. |
| reddit | 94 | DOM | `0001` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Is there snow in the picture from the hottest post in f/EarthPorn? |
| reddit | 94 | SoM | `0001` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Is there snow in the picture from the hottest post in f/EarthPorn? |
| reddit | 94 | Vision | `0001` | `premature-or-wrong-finish` | - | fail_early_finish | Is there snow in the picture from the hottest post in f/EarthPorn? |
| reddit | 100 | DOM | `0001` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Navigate to the comments section of a post that contains a picture of a cake. |
| reddit | 100 | SoM | `0001` | `navigation-loop` | P5,P14 | fail_no_progress | Navigate to the comments section of a post that contains a picture of a cake. |
| reddit | 100 | Vision | `0001` | `navigation-loop` | P14 | fail_finish_empty_answer | Navigate to the comments section of a post that contains a picture of a cake. |
| reddit | 107 | DOM | `0011` | `visual-info-missing` | P6 | fail_max_steps_click_back_loop | Subscribe to a forum that contains the following picture. |
| reddit | 107 | SoM | `0011` | `unmatched-stuck` | - | fail_max_steps_click_back_loop | Subscribe to a forum that contains the following picture. |
| reddit | 120 | DOM | `0110` | `premature-or-wrong-finish` | - | fail_early_finish | According to the image, which country in this picture has the worst rail system? |
| reddit | 120 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | According to the image, which country in this picture has the worst rail system? |
| reddit | 124 | DOM | `0001` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Find me a post in /f/dataisbeautiful that illustrates the Microsoft revenue in 1985, and tell m… |
| reddit | 124 | SoM | `0001` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | Find me a post in /f/dataisbeautiful that illustrates the Microsoft revenue in 1985, and tell m… |
| reddit | 124 | Vision | `0001` | `navigation-loop` | P5,P14 | fail_no_progress | Find me a post in /f/dataisbeautiful that illustrates the Microsoft revenue in 1985, and tell m… |
| reddit | 129 | Phantom-SoM | `1110` | `no-step-trace` | - | step trace unavailable | Subscribe to the forum which is most likely to include discussions related to the image. |
| reddit | 130 | SoM | `1001` | `premature-or-wrong-finish` | - | fail_early_finish | Subscribe to the forum which is most likely to include discussions related to the image. |
| reddit | 130 | Vision | `1001` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Subscribe to the forum which is most likely to include discussions related to the image. |
| reddit | 131 | DOM | `0100` | `visual-info-missing` | P6,P14 | fail_finish_eval_mismatch | Subscribe to the forum which is most likely to include discussions related to the image. |
| reddit | 131 | Vision | `0100` | `navigation-loop` | P5,P14 | fail_no_progress | Subscribe to the forum which is most likely to include discussions related to the image. |
| reddit | 131 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Subscribe to the forum which is most likely to include discussions related to the image. |
| reddit | 139 | DOM | `0100` | `visual-info-missing` | P6 | fail_early_finish | I don't like the author of this image from one of the hot posts in /f/memes. Can you help me bl… |
| reddit | 139 | Vision | `0100` | `navigation-loop` | P5,P14 | fail_no_progress | I don't like the author of this image from one of the hot posts in /f/memes. Can you help me bl… |
| reddit | 139 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | I don't like the author of this image from one of the hot posts in /f/memes. Can you help me bl… |
| reddit | 142 | DOM | `0100` | `navigation-loop` | P5,P13,P14 | fail_no_progress | Leave a comment in this post with the release date of the movie in the image in mm-dd-yyyy form… |
| reddit | 142 | Vision | `0100` | `premature-or-wrong-finish` | - | fail_early_finish | Leave a comment in this post with the release date of the movie in the image in mm-dd-yyyy form… |
| reddit | 142 | Phantom-SoM | `0100` | `no-step-trace` | - | step trace unavailable | Leave a comment in this post with the release date of the movie in the image in mm-dd-yyyy form… |
| reddit | 148 | DOM | `0010` | `premature-or-wrong-finish` | - | fail_early_finish | Leave a comment in this post with the text as the number of arancinis in the image. |
| reddit | 148 | SoM | `0010` | `premature-or-wrong-finish` | - | fail_early_finish | Leave a comment in this post with the text as the number of arancinis in the image. |
| reddit | 148 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Leave a comment in this post with the text as the number of arancinis in the image. |
| reddit | 150 | DOM | `0010` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Leave a comment in this post with the text as the number of adults in the image. |
| reddit | 150 | SoM | `0010` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Leave a comment in this post with the text as the number of adults in the image. |
| reddit | 150 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Leave a comment in this post with the text as the number of adults in the image. |
| reddit | 151 | DOM | `0110` | `unmatched-stuck` | - | fail_no_progress | Leave a comment in this post with the text as the number of adults in the image. |
| reddit | 151 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | Leave a comment in this post with the text as the number of adults in the image. |
| reddit | 152 | DOM | `0010` | `unmatched-stuck` | - | fail_no_progress | Leave a comment in this post with the text as the number of adults in the image. |
| reddit | 152 | SoM | `0010` | `unmatched-stuck` | - | fail_no_progress | Leave a comment in this post with the text as the number of adults in the image. |
| reddit | 152 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Leave a comment in this post with the text as the number of adults in the image. |
| reddit | 155 | Vision | `1101` | `navigation-loop` | P5,P14 | fail_no_progress | I saw a really funny movie last week (the image is the movie poster). Can you navigate to the c… |
| reddit | 160 | Vision | `1100` | `premature-or-wrong-finish` | - | fail_early_finish | Can you subscribe to all subreddits that start with the letter 'i' and have a female usb to mal… |
| reddit | 160 | Phantom-SoM | `1100` | `no-step-trace` | - | step trace unavailable | Can you subscribe to all subreddits that start with the letter 'i' and have a female usb to mal… |
| reddit | 161 | DOM | `0110` | `visual-info-missing` | P5,P6,P14 | fail_no_progress | Take me to the post for the joke on this page that is most likely to take place in the followin… |
| reddit | 161 | Phantom-SoM | `0110` | `no-step-trace` | - | step trace unavailable | Take me to the post for the joke on this page that is most likely to take place in the followin… |
| reddit | 162 | DOM | `0001` | `navigation-loop` | P14 | fail_max_steps_search_repeat | Can you give me the URL for a gif I could send to my friend about the contrast of investing in … |
| reddit | 162 | SoM | `0001` | `search-browse-strategy` | P13 | fail_finish_claim_missing | Can you give me the URL for a gif I could send to my friend about the contrast of investing in … |
| reddit | 162 | Vision | `0001` | `search-browse-strategy` | P13 | fail_finish_claim_missing | Can you give me the URL for a gif I could send to my friend about the contrast of investing in … |
| reddit | 170 | DOM | `0101` | `visual-info-missing` | P5,P6,P14 | fail_no_progress | Can you subscribe to the forum that is most likely to contain posts like this? |
| reddit | 170 | Vision | `0101` | `unmatched-stuck` | - | fail_incomplete_or_stuck | Can you subscribe to the forum that is most likely to contain posts like this? |
| reddit | 171 | Phantom-SoM | `1110` | `no-step-trace` | - | step trace unavailable | Can you subscribe to the forum that is most likely to contain posts like this? |
| reddit | 178 | Vision | `1101` | `unmatched-stuck` | - | fail_incomplete_or_stuck | Can you subscribe to the forum for this city? |
| reddit | 179 | DOM | `0010` | `visual-info-missing` | P6 | fail_max_steps_search_repeat | Can you take me to the page that shows the most controversial posts of the past month in the fo… |
| reddit | 179 | SoM | `0010` | `max-step-exhaustion` | - | fail_max_steps_search_repeat | Can you take me to the page that shows the most controversial posts of the past month in the fo… |
| reddit | 179 | Phantom-SoM | `0010` | `no-step-trace` | - | step trace unavailable | Can you take me to the page that shows the most controversial posts of the past month in the fo… |
| reddit | 182 | DOM | `0011` | `visual-info-missing` | P6 | success | I am a student at the school in the image. Can you make a post asking for the best Indian food … |
| reddit | 182 | SoM | `0011` | `premature-or-wrong-finish` | - | fail_finish_eval_mismatch | I am a student at the school in the image. Can you make a post asking for the best Indian food … |
| reddit | 188 | SoM | `1011` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Can you subscribe me to the most suitable forum where I can find more posts like in the image? |
| reddit | 189 | SoM | `1011` | `navigation-loop` | P14 | fail_finish_eval_mismatch | Can you subscribe me to the most suitable forum where I can find more posts like in the image? |
| reddit | 200 | SoM | `1011` | `premature-or-wrong-finish` | - | fail_finish_wrong_url_not_found | Can you take me to the post on this page that is most related to the picture on this page? |

## Ablation Pair Diagnostics

Success vectors use mode order `DOM SoM Phantom-DOM`.

| Task | Mode | Vector | Cluster | P-rules | Reason bucket / note |
|---:|---|---|---|---|---|
| 0 | SoM | `101` | `unmatched-stuck` | - | fail_max_steps_click_back_loop |
| 2 | DOM | `010` | `visual-info-missing` | P6 | fail_finish_wrong_url_not_found |
| 2 | Phantom-DOM | `010` | `no-hard-rule-hit` | - | - |
| 4 | DOM | `010` | `visual-info-missing` | P5,P6,P14 | fail_no_progress |
| 4 | Phantom-DOM | `010` | `navigation-loop` | P5,P14 | - |
| 14 | DOM | `010` | `visual-info-missing` | P6,P14 | fail_incomplete_or_stuck |
| 14 | Phantom-DOM | `010` | `navigation-loop` | P5,P14 | - |
| 15 | DOM | `001` | `visual-info-missing` | P5,P6,P14 | fail_no_progress |
| 15 | SoM | `001` | `navigation-loop` | P14 | fail_finish_empty_answer |
| 18 | Phantom-DOM | `110` | `no-hard-rule-hit` | - | - |
| 26 | DOM | `010` | `visual-info-missing` | P5,P6,P14 | success |
| 26 | Phantom-DOM | `010` | `navigation-loop` | P5,P14 | - |
| 31 | SoM | `101` | `no-hard-rule-hit` | - | success |
| 36 | DOM | `001` | `premature-or-wrong-finish` | - | fail_finish_wrong_url_not_found |
| 36 | SoM | `001` | `premature-or-wrong-finish` | - | fail_finish_wrong_url_not_found |
