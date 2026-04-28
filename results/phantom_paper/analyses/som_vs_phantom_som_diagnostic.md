# SoM vs Phantom-SoM Mechanism Diagnostic (Image-Dimension Audit)

## 1. Summary

This audit compares fresh B0 SoM and Phantom-SoM runs on classifieds and reddit. The two arms share the SoM-family system prompt and the same `[SOM_MARKS]` text; the only intended treatment difference is that SoM receives the marked screenshot image, while Phantom-SoM does not. The image effect is strongly site-modulated: on classifieds, image-on SoM is better by +6.84 adjusted pp (50/234, 21.37%, versus 34/234, 14.53%); on reddit, image-on SoM is lower by -3.33 adjusted pp (22/210, 10.48%, versus 29/210, 13.81%). The solved-task overlap is low in both sites (adjusted Jaccard 0.333 on classifieds, 0.378 on reddit), so the image does not act like a scalar "more visual grounding" knob. The best theory update is that Axis 3 is multi-dimensional: images can provide spatial grounding, visual context, element disambiguation, and state recognition, but can also induce false visual confidence, wrong-region attention, image-page detours, and click loops.

Method note: overlap uses episode-level `success` and `adjusted_success`. Failure and image-dimension labels below are heuristic labels assigned from the step traces, intents, and outcome differences. They are measured over the fresh run artifacts, but they are not a full manual adjudication of every task.

## 2. SR Overlap (raw + adj, per site)

| Site | Metric | SoM successes | Phantom-SoM successes | Shared | SoM-only task IDs | Phantom-SoM-only task IDs | Jaccard |
|---|---:|---:|---:|---:|---|---|---:|
| classifieds | raw success | 54 | 37 | 26 | 13, 14, 17, 49, 52, 62, 63, 101, 106, 111, 115, 120, 124, 125, 127, 130, 132, 149, 151, 160, 165, 166, 173, 187, 196, 209, 210, 221 | 50, 68, 93, 170, 181, 183, 194, 201, 215, 217, 222 | 0.400 |
| classifieds | adjusted success | 50 | 34 | 21 | 13, 14, 17, 24, 49, 52, 62, 63, 101, 106, 111, 115, 120, 124, 125, 127, 130, 132, 149, 151, 160, 165, 166, 173, 187, 196, 209, 210, 221 | 50, 68, 93, 135, 170, 181, 183, 191, 194, 201, 215, 217, 222 | 0.333 |
| reddit | raw success | 25 | 30 | 17 | 2, 4, 14, 18, 42, 120, 142, 160 | 0, 15, 36, 94, 107, 157, 162, 167, 179, 182, 188, 189, 200 | 0.447 |
| reddit | adjusted success | 22 | 29 | 14 | 2, 4, 14, 18, 42, 120, 142, 160 | 0, 7, 15, 31, 36, 94, 107, 157, 162, 167, 179, 182, 188, 189, 200 | 0.378 |

The overlap is the first warning against a one-axis image theory. On classifieds, image-on SoM has 29 adjusted successes that Phantom-SoM misses, while Phantom-SoM still has 13 adjusted successes that SoM misses. That is a large net positive image effect, but not a universal improvement. On reddit, the direction reverses: SoM has only 8 adjusted unique successes, while Phantom-SoM has 15, meaning that image removal helps a non-trivial set of text-dominated post/comment tasks.

The site asymmetry matches the paper's conservative framing. Classifieds tasks often ask about visual listing structure, object appearance, and page layout, where screenshots can provide missing information. Reddit tasks more often require finding the correct post, comment, subreddit, or URL in a text-heavy interface; the image can become a misleading attractor.

## 3. Trajectory Metrics

| Site | Condition | N | Raw / Adj SR | Steps avg / median / max | Search-loop eps | Self-correction | Page-change rate | Unique URLs avg / median / max | Cost | P95 step latency |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| classifieds | SoM | 234 | 23.08 / 21.37 | 8.60 / 5 / 30 | 151 (64.5%) | 181 hits in 48 eps (20.5%) | 85.3% | 3.03 / 2 / 30 | $0.0415 | 74.0s |
| classifieds | Phantom-SoM | 234 | 15.81 / 14.53 | 12.05 / 8 / 30 | 180 (76.9%) | 229 hits in 60 eps (25.6%) | 83.8% | 3.63 / 3 / 14 | $0.0441 | 18.2s |
| reddit | SoM | 210 | 11.90 / 10.48 | 8.05 / 5 / 30 | 66 (31.4%) | 134 hits in 32 eps (15.2%) | 82.0% | 2.08 / 2 / 8 | $0.0409 | 58.9s |
| reddit | Phantom-SoM | 210 | 14.29 / 13.81 | 9.90 / 6 / 30 | 75 (35.7%) | 112 hits in 39 eps (18.6%) | 80.9% | 2.66 / 2 / 21 | $0.0381 | 51.4s |

| Site | Condition | Action distribution over steps |
|---|---|---|
| classifieds | SoM | type 567, select_option 121, scroll 402, finish 184, click 524, back 199, wait 2, tab_focus 13 |
| classifieds | Phantom-SoM | type 819, select_option 186, scroll 666, click 718, finish 122, back 280, tab_focus 28 |
| reddit | SoM | click 662, back 162, scroll 263, type 383, finish 111, select_option 26, tab_focus 84 |
| reddit | Phantom-SoM | click 793, back 177, scroll 309, finish 88, type 604, tab_focus 87, select_option 22 |

On classifieds, image-on SoM is both more successful and more efficient in steps. It needs fewer average steps (8.60 versus 12.05), fewer search-loop episodes (64.5% versus 76.9%), and fewer unique URLs, consistent with the screenshot resolving listing structure or visual object identity before the model over-searches. The latency tradeoff is large: SoM's p95 step latency is 74.0s versus 18.2s for Phantom-SoM, despite similar per-episode dollar cost ($0.0415 versus $0.0441).

On reddit, image-on SoM is shorter but less successful. It averages 8.05 steps versus 9.90 for Phantom-SoM and has lower URL diversity, yet its adjusted SR is 3.33 pp lower. This is the signature of harmful visual commitment: the image can make the agent pick or finish on an attractive candidate before doing enough textual verification.

## 4. Failure Pattern Distribution

The following failure labels are trace-derived proxies. They should be read as descriptive evidence for mechanism formation, not as final human-coded categories.

| Site | Condition | Failed N | Early-finish / wrong-commit | Search-loop | Visual-hijack / click-loop | Element-misground / wrong target | Missing visual context / disambiguation |
|---|---|---:|---:|---:|---:|---:|---:|
| classifieds | SoM | 184 | 135 (73.4%) | 41 (22.3%) | 3 (1.6%) | 5 (2.7%) | 0 |
| classifieds | Phantom-SoM | 200 | 89 (44.5%) | 73 (36.5%) | 8 (4.0%) | 8 (4.0%) | 22 (11.0%) |
| reddit | SoM | 188 | 93 (49.5%) | 31 (16.5%) | 13 (6.9%) | 51 (27.1%) | 0 |
| reddit | Phantom-SoM | 181 | 67 (37.0%) | 24 (13.3%) | 16 (8.8%) | 51 (28.2%) | 23 (12.7%) |

Classifieds shows the helpful side of the image most clearly. Phantom-SoM has a visible missing-context bucket (22 failures, 11.0%) and more search-loop failures than SoM. Removing the image makes the same SoM-family prompt spend more time typing, scrolling, and checking candidate pages, which helps some tasks but loses many visually grounded listing tasks.

Reddit shows the harmful side. SoM has more early-finish/wrong-commit failures than Phantom-SoM (93 versus 67) and a material visual-hijack/click-loop bucket (13 failures, 6.9%). Phantom-SoM still has image-reference failures, because the tasks can describe an image even when the model is not sent that image; however, its lower early-commit count and higher adjusted SR suggest that no-image runs often compensate with broader textual search.

## 5. Trajectory Snippet Case Studies (paper Section 5 use)

### Classifieds task 14: image helps row/layout grounding

Intent: "What is the email of the seller of the painting in the second row?"

| Mode | Outcome | Minimal trajectory |
|---|---|---|
| SoM | adjusted success | Clicks `/index.php?page=item&id=32385`, then finishes `olga.jones341@example.com`. |
| Phantom-SoM | fail | Clicks `/index.php?page=item&id=60133`, then finishes `john.dubois394@example.com`. |

The screenshot supplies row position and visual layout. With only text marks, Phantom-SoM picks a plausible but wrong painting. This is spatial grounding plus element disambiguation, not merely "avoid hijack."

### Classifieds task 24: image gives object/color context

Intent: "How many miles does the black truck on this page have?"

| Mode | Outcome | Minimal trajectory |
|---|---|---|
| SoM | adjusted success | From the gallery page, finishes in one step with `109999`. |
| Phantom-SoM | adjusted false positive | Scrolls repeatedly and says the black truck is not visible. |

Here the image carries the color and vehicle recognition needed to answer directly. The no-image trace shows that `[SOM_MARKS]` text alone does not reliably expose the same visual state.

### Classifieds task 50: image hurts through premature false confidence

Intent: "What is the email of the seller of the red palette on this page?"

| Mode | Outcome | Minimal trajectory |
|---|---|---|
| SoM | fail | Scrolls, then finishes that the seller email is not provided and only a phone number is shown. |
| Phantom-SoM | adjusted success | Searches and paginates through listings, reaches item `65673`, and finishes `sofia_kumar@example.com`. |

The image-on run appears to commit to the current page state too early. Removing the image forces a slower textual/listing search path that finds the seller email.

### Classifieds task 103: both solve, but image accelerates the path

Intent: "Explore the Arts + crafts category, and find the most recently listed item portraying the person in this image."

| Mode | Outcome | Minimal trajectory |
|---|---|---|
| SoM | adjusted success | Selects Arts + crafts, types `Abraham Lincoln statue`, clicks item `77577`, finishes `bronze Lincoln stands`. |
| Phantom-SoM | adjusted success | Selects the category, searches and scrolls across more results, eventually reaches the same item and finishes. |

Both arms can solve the task, but the image helps SoM translate the visual reference into a precise query and shortens the route.

### Reddit task 4: image helps select the actual post, not the image file

Intent: "I'm trying to do some research about the post with this image. Tell me how many comments mention the word wheat."

| Mode | Outcome | Minimal trajectory |
|---|---|---|
| SoM | adjusted success | Searches `wheat`, opens `/f/Washington/16163/wheat-field-in-central-wa`, checks comments, finishes `1`. |
| Phantom-SoM | fail | Searches `wheat`, opens a direct `/submission_images/...` image URL, scrolls the image page, and cannot inspect comments. |

This is a helpful image case on reddit: the screenshot helps link the visual reference to the post/comment context rather than the raw image asset.

### Reddit task 18: image helps forum recognition

Intent: "Navigate to a subreddit where I can find the most similar content to this image."

| Mode | Outcome | Minimal trajectory |
|---|---|---|
| SoM | adjusted success | Clicks `/f/MechanicalKeyboards` and finishes in the relevant subreddit. |
| Phantom-SoM | fail | Opens the direct image, backs out, searches `pastel mechanical keyboard`, and loops for 30 steps. |

The image supplies enough visual category information to identify the forum directly. This is visual context disambiguation and state/action recognition.

### Reddit task 0: image hurts through direct-image looping

Intent: "I'm trying to find this post. Navigate to the comment section for it."

| Mode | Outcome | Minimal trajectory |
|---|---|---|
| SoM | fail | Repeatedly clicks the direct sushi image `/submission_images/a731...` and backs to `/f/food`, never reaching the post comments. |
| Phantom-SoM | adjusted success | After early image/back actions, eventually opens `/f/food/82896/i-ate-sushi-platter` and reaches the comments. |

The marked image acts as an attractor toward the image asset rather than the post page. This is a visual-hijack/click-loop failure.

### Reddit task 7: image hurts through wrong visual target

Intent: "Navigate to the comment url of the recipe posted by the OP."

| Mode | Outcome | Minimal trajectory |
|---|---|---|
| SoM | adjusted false positive | Navigates food pages and clicks the wrong cake image/post `/f/food/18811/...`, without finishing the exact comment URL. |
| Phantom-SoM | adjusted success | Searches `cake recipe`, opens exact comment permalink `/f/food/18987/-/comment/313939`, and finishes the URL. |

The image-on path follows a visually plausible cake target but misses the OP recipe comment. The no-image path relies more on text search and lands on the exact permalink.

### Reddit task 31: image hurts through search/user detour

Intent: "Find the user who posted this and navigate to comments section of one of their other posts."

| Mode | Outcome | Minimal trajectory |
|---|---|---|
| SoM | adjusted false positive | Searches lobster, repeatedly tries `/user/savage-dragon`, gets 404s, and loops until 30 steps. |
| Phantom-SoM | adjusted success | Searches `colorful lobster image`, opens `/f/nosleep/109250/...`, clicks `/user/RaynaClay`, opens `/comments`, and finishes. |

This is not a simple "image bad" case. The image may help name the object, but it also sends the agent down a wrong-user path. The no-image run uses broader text evidence and recovers.

## 6. Image Effect Dimension Taxonomy + Counts

Counts below are primary labels over adjusted-exclusive task sets. Benefit counts are SoM adjusted successes that Phantom-SoM misses. Harm counts are Phantom-SoM adjusted successes that SoM misses. A task can plausibly involve multiple dimensions, but each task is assigned one primary label so that counts are additive.

### Benefit dimensions: image-on succeeds, image-off fails

| Site | Dimension | Count | Task IDs |
|---|---|---:|---|
| classifieds | Visual context disambiguation | 13 | 13, 52, 101, 115, 124, 125, 130, 132, 151, 165, 166, 173, 187 |
| classifieds | Element disambiguation / target selection | 10 | 14, 17, 62, 63, 106, 111, 127, 149, 209, 221 |
| classifieds | Spatial grounding / layout | 5 | 24, 49, 120, 196, 210 |
| classifieds | State/action recognition | 1 | 160 |
| reddit | State/action recognition | 6 | 2, 4, 14, 42, 142, 160 |
| reddit | Visual context disambiguation | 2 | 18, 120 |

### Harm dimensions: image-off succeeds, image-on fails

| Site | Dimension | Count | Task IDs |
|---|---|---:|---|
| classifieds | False confidence / wrong commit | 3 | 50, 215, 217 |
| classifieds | False visual confidence / wrong commit | 5 | 68, 93, 194, 201, 222 |
| classifieds | Image-induced search detour | 5 | 135, 170, 181, 183, 191 |
| reddit | Visual-hijack / click-loop | 2 | 0, 167 |
| reddit | Visual misdirection / wrong target | 1 | 7 |
| reddit | False visual confidence / wrong commit | 9 | 15, 36, 94, 157, 162, 182, 188, 189, 200 |
| reddit | Image-induced search detour | 3 | 31, 107, 179 |

This taxonomy gives a more faithful mechanism than a single visual-hijack axis. On classifieds, the positive dimensions dominate: visual context, element disambiguation, and spatial layout jointly explain most of the 29 SoM-only adjusted successes. On reddit, the negative dimensions dominate: false visual confidence and image-induced detours explain most of the 15 Phantom-SoM-only adjusted successes, while direct visual-hijack is only a minority pattern.

## 7. Theory Update (Axis 3: image as multi-dimensional)

The old simplified Axis 3 story, "image mainly causes visual-hijack," should be retired. It captures one real reddit failure family, but it misses the larger classifieds effect and overstates a single failure mode.

The updated theory should be:

1. Images add several helpful channels: spatial grounding, visual context disambiguation, element/target selection, and state/action recognition.
2. Images also add several harmful channels: visual-hijack/click-loop, visual misdirection, false visual confidence or early commitment, and search/user detours caused by over-attending to a visual candidate.
3. The net SR effect is site-modulated. In visually rich classifieds listings, helpful channels dominate, producing a +6.84 pp adjusted SR gain for SoM over Phantom-SoM. In text-dominated reddit post/comment tasks, harmful channels modestly dominate, producing a -3.33 pp adjusted SR loss for SoM relative to Phantom-SoM.
4. The effect is task-conditional rather than uniform. Adjusted Jaccard is only 0.333 on classifieds and 0.378 on reddit, so image-on and image-off runs solve meaningfully different task pools.
5. Cost and latency are part of the deployment mechanism. SoM and Phantom-SoM have similar average dollar cost in classifieds, but Phantom-SoM has much lower p95 step latency (18.2s versus 74.0s). On reddit, Phantom-SoM is both slightly cheaper ($0.0381 versus $0.0409) and higher SR, although the observed 3.33 pp gap should still be interpreted conservatively given run-to-run variance.

Section 5 prose should therefore frame the image axis as a mixture of affordances and hazards. A suitable revision is:

> Marked screenshots are not a monotone visual upgrade. They provide spatial grounding, visual context, state recognition, and element disambiguation, which dominate on visually rich classifieds tasks. The same channel can also create false visual confidence, wrong-target attention, and direct-image loops, which become more costly on text-heavy reddit tasks. Phantom-SoM isolates this image channel: holding the SoM prompt fixed, removing the image lowers classifieds adjusted SR from 21.37% to 14.53% but raises reddit adjusted SR from 10.48% to 13.81%. The image effect is therefore site- and task-modulated, not a single visual-hijack axis.
