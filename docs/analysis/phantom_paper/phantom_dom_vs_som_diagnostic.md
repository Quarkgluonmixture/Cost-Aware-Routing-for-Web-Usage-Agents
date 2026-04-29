# P-text vs Phantom-SoM Mechanism Diagnostic

## 1. Summary

This audit compares fresh B0 P-text and Phantom-SoM runs on classifieds and reddit. The old "prompt only changes commitment confidence" theory is too narrow. On classifieds, P-text and Phantom-SoM have identical adjusted SR (34/234, 14.53%) but solve substantially different task sets: only 21 adjusted successes overlap, giving Jaccard 0.447. On reddit, the premise that the two adjusted SRs are equal is not supported by episode-level `adjusted_success`: P-text has 25/210 adjusted successes (11.90%), while Phantom-SoM has 29/210 (13.81%), with Jaccard 0.543. The best update is: the prompt family acts as a task-conditional decision prior over search terms, clicks, backtracking, and finish timing; commitment calibration is a real second-order subeffect, but not the only prompt knob.

Important data note: `condition_summary_v2.json` stores raw `success_rate`. For reddit P-text it reports raw 29/210 = 13.81%, but the episode summaries contain 4 raw-success false positives, so adjusted SR is 25/210 = 11.90%. All overlap and oracle-style claims below use episode-level `adjusted_success` unless a row is explicitly marked raw.

## 2. SR Overlap

| Site | Metric | P-text successes | Phantom-SoM successes | Shared | DOM-only task IDs | SoM-only task IDs | Jaccard |
|---|---:|---:|---:|---:|---|---|---:|
| classifieds | raw success | 39 | 37 | 27 | 2, 5, 17, 60, 101, 113, 115, 116, 137, 174, 184, 196 | 79, 103, 153, 181, 183, 195, 215, 220, 222, 233 | 0.551 |
| classifieds | adjusted success | 34 | 34 | 21 | 2, 5, 17, 60, 101, 113, 115, 116, 137, 164, 174, 184, 196 | 79, 103, 135, 153, 181, 183, 191, 194, 195, 215, 220, 222, 233 | 0.447 |
| reddit | raw success | 29 | 30 | 23 | 42, 124, 130, 132, 153, 181 | 94, 131, 139, 151, 161, 167, 170 | 0.639 |
| reddit | adjusted success | 25 | 29 | 19 | 42, 124, 130, 132, 153, 181 | 7, 26, 72, 94, 131, 139, 151, 161, 167, 170 | 0.543 |

The overlap is not high. Even where SR is exactly tied on classifieds, the solved-task pool is closer to "two partially overlapping policies" than to "the same policy with minor noise." Reddit is also not a high-overlap case after adjustment: Phantom-SoM has four more adjusted successes, and ten successes are unique to the SoM-prompt arm.

The raw-to-adjusted gap is still consistent with a commitment-calibration effect: P-text has more raw-success false positives than Phantom-SoM on both sites (classifieds 5 vs 3; reddit 4 vs 1). But because the adjusted success pools diverge strongly, the prompt effect cannot be reduced to a uniform tendency to finish earlier or later.

## 3. Trajectory Metrics

| Site | Condition | N | Raw / Adj SR | Steps avg / median / max | Search-loop eps | Self-correction | Page-change rate | Unique URLs avg / median / max |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| classifieds | P-text | 234 | 16.67 / 14.53 | 11.16 / 7 / 30 | 171 (73.1%) | 134 hits in 54 eps (23.1%) | 82.4% | 3.67 / 3 / 13 |
| classifieds | Phantom-SoM | 234 | 15.81 / 14.53 | 12.05 / 8 / 30 | 180 (76.9%) | 229 hits in 60 eps (25.6%) | 83.8% | 3.63 / 3 / 14 |
| reddit | P-text | 210 | 13.81 / 11.90 | 11.45 / 7 / 30 | 104 (49.5%) | 103 hits in 37 eps (17.6%) | 84.3% | 3.70 / 3 / 21 |
| reddit | Phantom-SoM | 210 | 14.29 / 13.81 | 9.90 / 6 / 30 | 75 (35.7%) | 112 hits in 39 eps (18.6%) | 80.9% | 2.66 / 2 / 21 |

| Site | Condition | Action distribution over steps |
|---|---|---|
| classifieds | P-text | click 581 (22.2%), type 860 (32.9%), scroll 643 (24.6%), select_option 204 (7.8%), back 169 (6.5%), tab_focus 18 (0.7%), finish 137 (5.2%) |
| classifieds | Phantom-SoM | click 718 (25.5%), type 819 (29.1%), scroll 666 (23.6%), select_option 186 (6.6%), back 280 (9.9%), tab_focus 28 (1.0%), finish 122 (4.3%) |
| reddit | P-text | click 811 (33.7%), type 842 (35.0%), scroll 475 (19.8%), select_option 27 (1.1%), back 122 (5.1%), tab_focus 35 (1.5%), finish 93 (3.9%) |
| reddit | Phantom-SoM | click 793 (38.1%), type 604 (29.0%), scroll 309 (14.9%), select_option 22 (1.1%), back 177 (8.5%), tab_focus 87 (4.2%), finish 88 (4.2%) |

The trajectory differences are site-modulated. On classifieds, SoM prompt adds slightly more steps, search-loop episodes, self-correction language, clicks, and backs, while keeping adjusted SR equal. On reddit, SoM prompt is more efficient: fewer steps, fewer search-loop episodes, lower URL diversity, less typing, more direct clicking/backtracking/tab-focus behavior, and four additional adjusted successes.

## 4. Failure Pattern Distribution

Pattern labels below use the shared 9-category taxonomy as a heuristic proxy over fresh step traces, not manual per-task adjudication. Rules are deterministic and intentionally conservative: terminal failed `finish` maps to early-finish/wrong-commit; repeated search-page trajectories map to DOM search-loop; repeated no-change clicks/back/image-page loops map to SoM visual-hijack/click-loop; visual-reference failures without stronger loop evidence map to DOM visual-missing. Vision-specific categories are retained for compatibility but are expected to be zero for these two text-only arms.

| Site | Condition | Failed N | DOM visual-missing | DOM search-loop | SoM early-finish/wrong-commit | SoM visual-hijack/click-loop | Vision text/grounding loops | Vision element-misground | Element-misground | Abandon-after-N | Misc |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| classifieds | P-text | 200 | 4 (2.0%) | 81 (40.5%) | 103 (51.5%) | 6 (3.0%) | 0 | 0 | 5 (2.5%) | 1 (0.5%) | 0 |
| classifieds | Phantom-SoM | 200 | 2 (1.0%) | 94 (47.0%) | 89 (44.5%) | 8 (4.0%) | 0 | 0 | 6 (3.0%) | 1 (0.5%) | 0 |
| reddit | P-text | 185 | 28 (15.1%) | 60 (32.4%) | 76 (41.1%) | 10 (5.4%) | 0 | 0 | 11 (5.9%) | 0 | 0 |
| reddit | Phantom-SoM | 181 | 23 (12.7%) | 44 (24.3%) | 67 (37.0%) | 19 (10.5%) | 0 | 0 | 28 (15.5%) | 0 | 0 |

This distribution weakens the old "prompt only changes commitment confidence" account. P-text indeed has more early-finish/wrong-commit failures on both sites, which preserves the commitment-calibration subclaim. But Phantom-SoM also changes search-loop frequency and element-level behavior: it increases search-loop on classifieds but reduces it sharply on reddit; it also shifts reddit failures toward more click/back/image-page or element-misground patterns. That is a policy-shape difference, not only a finish-threshold difference.

As reference context from `cross_site_pattern_consolidation.md`, the broader B0-to-B1 SoM visual-hijack/click-loop shift is cross-site but site-modulated (+50.0 pp on classifieds, +33.3 pp on reddit). The present audit is compatible with that larger mechanism: prompt and representation choices reshape failure geometry, but the direction depends on site structure and task type.

## 5. Trajectory Snippet Case Studies

### Classifieds task 2: P-text succeeds, Phantom-SoM misses the target item

Intent: find the most recently posted Jewelry item with a red gem priced $300-$600.

P-text succeeds in 6 steps: select Jewelry, search "red gem", refine the query/price, click item `10066`, then finish with "Eye Catching Dazzling Center Of Attention Cocktail Ring, $425.00, published 2023/11/16." Phantom-SoM follows the same start but inserts an extra max-price step, clicks item `69296`, self-corrects that jade is not a red gem, then clicks item `30365` and finishes with a valid-looking but older item from 2023/10/20. This is a task-specific ranking/selection difference, not a pure commitment threshold issue.

Minimal trajectories:

| Mode | Outcome | First steps |
|---|---|---|
| P-text | adjusted success | select_option `/`; type `red gem` in Jewelry; refine search; type price filter; click `/item&id=10066`; finish correct item |
| Phantom-SoM | fail | select_option `/`; type `red gem`; refine search; type price filter; type max price; click `/item&id=69296`; back after rejecting jade; click `/item&id=30365`; finish older item |

### Classifieds task 79: Phantom-SoM succeeds, P-text gets stuck on wrong item

Intent: find an item on the page that references the person in the image.

P-text searches in Books, clicks a Bible map item, then repeatedly types/searches from the wrong item page. Phantom-SoM searches similarly but clicks item `44608` and finishes after identifying the title "JESUS of NAZARETH." Here the SoM prompt did not simply delay or accelerate finish; it changed which search result was treated as matching the visual reference.

Minimal trajectories:

| Mode | Outcome | First steps |
|---|---|---|
| P-text | fail | type search from Books page; click `/item&id=72515`; type from item page; type "Jesus statue"; type again, still on wrong page |
| Phantom-SoM | adjusted success | type religious/Jesus search; click `/item&id=44608`; finish "JESUS of NAZARETH" |

### Classifieds task 50: both succeed, but Phantom-SoM takes a much longer route

Intent: report the seller email for the red palette.

Both modes finish with `sofia_kumar@example.com`. P-text searches directly and finishes in 3 steps. Phantom-SoM scrolls, searches, paginates, scrolls through page 2, then eventually reaches the same listing and finishes in 17 steps. This is a clean example of equal outcome with materially different exploration policy.

Minimal trajectories:

| Mode | Outcome | First steps |
|---|---|---|
| P-text | adjusted success, 3 steps | type "red palette"; click `/item&id=65673`; finish seller email |
| Phantom-SoM | adjusted success, 17 steps | scroll category; scroll; type search; scroll; click page 2; several scroll/search steps; final click item; finish same seller email |

### Reddit task 132: P-text succeeds, Phantom-SoM enters image-page loop

Intent: count comments for a top-50 hot /f/nyc image post.

P-text searches /f/nyc and "cat near toilet," clicks the post page, and finishes with 146 comments. Phantom-SoM searches "cat in bathroom," repeatedly clicks the direct image URL and backs out to search results, then hits 30 steps without reaching the post metadata. This is a search/click loop difference induced by the prompt-conditioned interpretation of the same marks text.

Minimal trajectories:

| Mode | Outcome | First steps |
|---|---|---|
| P-text | adjusted success | click forums; search `/f/nyc`; recover from bad query; search `cat near toilet`; click `/f/nyc/88264`; search/click again; finish 146 |
| Phantom-SoM | fail | search `cat in bathroom`; click direct `/submission_images/...`; back; click same image; back; repeat image-page loop until max steps |

### Reddit task 7: Phantom-SoM succeeds, P-text has raw success but adjusted failure

Intent: navigate to the comment URL of the recipe posted by the OP.

P-text searches "Christmas cake recipe" and scrolls search results; it is raw-success in the evaluator but adjusted false positive in the episode data. Phantom-SoM searches the broader "cake recipe," clicks directly into the OP comment permalink, clicks/settles on the same permalink, and finishes with the exact comment URL. This is a concrete example of the SoM prompt changing search phrasing and target selection.

Minimal trajectories:

| Mode | Outcome | First steps |
|---|---|---|
| P-text | raw success, adjusted failure | type `Christmas cake recipe`; scroll search; scroll; scroll |
| Phantom-SoM | adjusted success | type `cake recipe`; click `/f/food/18987/-/comment/313939`; click same permalink; click same permalink; finish exact URL |

### Reddit task 139: Phantom-SoM blocks the correct author, P-text blocks the wrong one

Intent: block the author of a hot /f/memes image post.

P-text immediately clicks a candidate post, follows user `elch3w`, clicks block, and finishes on the block list, but adjusted outcome is false because it blocked the wrong author. Phantom-SoM first navigates through forums to /f/memes, scrolls to identify the target image, follows user `Jamminmb`, and blocks that author successfully. This supports the task-conditional prompt-prior theory: same observation substrate, different candidate-selection path.

Minimal trajectories:

| Mode | Outcome | First steps |
|---|---|---|
| P-text | fail | click candidate `/f/memes/41616/...`; click `/user/elch3w`; click block; click block list; finish |
| Phantom-SoM | adjusted success | click home/forums; click `/f/memes`; scroll; click `/user/Jamminmb`; click block; click block list; finish |

## 6. Theory Update

### Theory A: prompt -> commitment confidence

Status: partially supported but insufficient. P-text has more raw-to-adjusted false positives than Phantom-SoM on both sites (classifieds 5 vs 3; reddit 4 vs 1), and the failure proxy labels more P-text failures as early-finish/wrong-commit. That preserves the commitment-confidence subclaim. However, it cannot explain the low overlap on classifieds or the reddit adjusted-SR gap.

### Theory B: prompt has negligible SR effect

Status: falsified for task pools. If prompt effects were negligible, adjusted success overlap should be very high. Instead, Jaccard is 0.447 on classifieds and 0.543 on reddit. Even in the one site where adjusted SR is numerically equal, 13 successes move in each direction. Same total SR hides different task-specific wins and losses.

### Theory C: prompt changes task-specific decisions over the same marks text

Status: best supported. The prompt family changes search query wording, candidate ordering, when the agent backs out of a direct image URL, and which marked item it treats as semantically relevant. The case studies show both directions: DOM prompt wins some price/category/listing tasks; SoM prompt wins some image-reference and forum-author tasks. This is closer to task-conditional preference shaping than to a scalar finish threshold.

### Theory D: representation is still the primary Phantom mechanism

Status: still plausible but now needs cleaner wording. Both P-text and Phantom-SoM share `[SOM_MARKS]` text and both are competitive text-only arms; the observation representation remains the main reason a no-image mode works at all. But within that shared representation, prompt wording is not inert. It is a second axis that reorders task-specific decisions and produces moderate, not tiny, task-pool divergence.

### Recommended Section 1 / Section 5 prose changes

Do not say: "the prompt knob mainly changes commitment confidence and not SR." That was a useful stale-data hypothesis, but fresh full-run evidence is broader.

Safer replacement:

> Holding the marks-text observation fixed, DOM-family and SoM-family prompts achieve similar aggregate SR on classifieds and nearby SR on reddit, but they do not solve the same tasks. The prompt family acts less like a uniform confidence knob and more like a task-conditional decision prior: it changes search phrasing, candidate selection, backtracking, and finish calibration. The raw-to-adjusted false-positive gap suggests a commitment-calibration component, but the low P-text/Phantom-SoM overlap shows that prompt wording also changes which tasks are solved.

For Section 5, frame the two-knob mechanism as:

1. Representation knob: AXTree vs `[SOM_MARKS]` changes the action surface and trajectory basin.
2. Prompt-family knob: DOM prompt vs SoM prompt changes task-conditional preferences within the same marks-text surface.
3. Commitment confidence: a measurable subeffect, visible in raw-to-adjusted FP gaps, but not the whole mechanism.

This preserves the paper's integrity: Phantom remains a drop-in text-only routing arm, but the mechanism claim should move from "prompt tunes when it commits" to "prompt tunes how it searches, selects, backtracks, and commits."
