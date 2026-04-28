# B1 Capability Profile — Qwen3-VL-4B Local Baseline

This profile summarizes the completed B1 three-mode VisualWebArena baselines for classifieds and reddit. It is intended as Section 6 cross-model generalization prep. It uses episode-level `adjusted_success` for SR, `condition_summary_v2.json` for cost/latency/energy, and the shared P1-P14/cluster taxonomy used by `docs/analysis/phantom_paper/disagreement_clusters.md`. No experiments were run.

## 1. SR Snapshot (cls + red, raw + adjusted)

| Site | Mode | N | Raw SR | Adjusted SR | FP gap | B0 adjusted SR | B1-B0 delta |
|---|---|---:|---:|---:|---:|---:|---:|
| classifieds | DOM | 234 | 26 (11.11%) | 20 (8.55%) | 6 (2.56 pp) | 14.10% | -5.56 pp |
| classifieds | SoM | 234 | 41 (17.52%) | 32 (13.68%) | 9 (3.85 pp) | 21.37% | -7.69 pp |
| classifieds | Vision | 234 | 26 (11.11%) | 17 (7.26%) | 9 (3.85 pp) | 13.68% | -6.41 pp |
| reddit | DOM | 210 | 21 (10.00%) | 16 (7.62%) | 5 (2.38 pp) | 9.52% | -1.90 pp |
| reddit | SoM | 210 | 17 (8.10%) | 12 (5.71%) | 5 (2.38 pp) | 10.48% | -4.76 pp |
| reddit | Vision | 210 | 10 (4.76%) | 5 (2.38%) | 5 (2.38 pp) | 6.67% | -4.29 pp |

B1 is below B0 in every site/mode cell. The gap is largest for classifieds SoM (-7.69 pp) and reddit SoM (-4.76 pp), which matters because SoM remains the strongest B1 arm on classifieds but becomes weaker than DOM on reddit.

## 2. Cost / Latency / Energy / CO2 Snapshot

| Site | Mode | B1 cost/task | B0 cost/task | B1 p95 step latency | B0 p95 step latency | B1 avg energy kWh | B1 avg CO2e kg | B0 energy/CO2 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| classifieds | DOM | $0.0398 | $0.0427 | 43.2s | 37.5s | 0.005216 | 0.001147 | unavailable for proxy API |
| classifieds | SoM | $0.0347 | $0.0415 | 30.2s | 74.0s | 0.001986 | 0.000437 | unavailable for proxy API |
| classifieds | Vision | $0.0133 | $0.0248 | 64.5s | 45.0s | 0.001942 | 0.000427 | unavailable for proxy API |
| reddit | DOM | $0.0536 | $0.0516 | 87.9s | 73.6s | 0.003770 | 0.000829 | unavailable for proxy API |
| reddit | SoM | $0.0435 | $0.0409 | 131.5s | 58.9s | 0.005119 | 0.001126 | unavailable for proxy API |
| reddit | Vision | $0.0137 | $0.0227 | 53.5s | 55.6s | 0.001276 | 0.000281 | unavailable for proxy API |

B1 local GPU runs expose real NVML energy fields; B0 proxy API runs have cost and latency but not comparable local energy. B1 is cheaper for Vision, roughly similar for DOM/SoM cost, but much slower in wall-clock latency because local Qwen3-VL-4B inference is not proxy-hosted.

## 3. Failure Pattern Distribution (per mode)

The table below reports all adjusted-failed B1 baseline episodes, not only one-arm-only disagreement tasks. Categories use the shared 9-pattern taxonomy: visual-missing, early-finish/wrong-commit, visual-hijack/click-loop, click-loop/no-text-grounding, element-misground, abandon-after-N, search-loop, click-loop, other.

| Site | Mode | Failed N | Top pattern 1 | Top pattern 2 | Top pattern 3 |
|---|---|---:|---|---|---|
| classifieds | DOM | 214 | visual-missing: 115 (53.7%) | search-loop: 41 (19.2%) | click-loop: 34 (15.9%) |
| classifieds | SoM | 202 | visual-hijack/click-loop: 98 (48.5%) | other: 53 (26.2%) | search-loop: 40 (19.8%) |
| classifieds | Vision | 217 | click-loop/no-text-grounding: 126 (58.1%) | element-misground: 47 (21.7%) | other: 37 (17.1%) |
| reddit | DOM | 194 | visual-missing: 83 (42.8%) | search-loop: 53 (27.3%) | click-loop: 38 (19.6%) |
| reddit | SoM | 198 | visual-hijack/click-loop: 123 (62.1%) | other: 40 (20.2%) | search-loop: 28 (14.1%) |
| reddit | Vision | 205 | element-misground: 97 (47.3%) | click-loop/no-text-grounding: 66 (32.2%) | other: 38 (18.5%) |

Representative B1 failure snippets:

- `classifieds DOM task 0` (visual-missing): P5,P6,P14; 8 steps; actions type->scroll->click->scroll->click->scroll->scroll->scroll; reason=unknown. Intent: Find me the cheapest blue kayak on this site.
- `classifieds SoM task 0` (visual-hijack/click-loop): P5,P14; 6 steps; actions type->scroll->type->scroll->scroll->scroll; reason=unknown. Intent: Find me the cheapest blue kayak on this site.
- `classifieds Vision task 0` (click-loop/no-text-grounding): P14; 4 steps; actions type->scroll->scroll->finish; reason=unknown. Intent: Find me the cheapest blue kayak on this site.
- `reddit DOM task 102` (visual-missing): P6; 2 steps; actions click->finish; reason=unknown. Intent: How many red keys are in the picture? Leave a comment replying to user "liverblow" with the following format: "It has <number of r
- `reddit SoM task 0` (visual-hijack/click-loop): P14; 6 steps; actions click->click->click->click->click->click; reason=unknown. Intent: I'm trying to find this post. Navigate to the comment section for it.
- `reddit Vision task 100` (element-misground): P1,P5,P14; 4 steps; actions click->click->click->click; reason=unknown. Intent: Navigate to the comments section of a post that contains a picture of a cake.

## 4. B0 vs B1 Per-Site Contrast (disaggregated)

### 4.1 SR delta by site and mode

| Site | Mode | B0 adjusted | B1 adjusted | Delta | B1-only successes | B0-only successes |
|---|---|---:|---:|---:|---|---|
| classifieds | DOM | 14.10% | 8.55% | -5.56 pp | 6: `64, 83, 101, 164, 189, 210` | 19: `17, 44, 46, 48, 54, 55, 63, 110, 116, 126, 139, 167...` |
| classifieds | SoM | 21.37% | 13.68% | -7.69 pp | 13: `5, 19, 40, 78, 93, 98, 112, 135, 152, 170, 174, 184...` | 31: `10, 14, 15, 24, 44, 45, 46, 49, 52, 54, 55, 62...` |
| classifieds | Vision | 13.68% | 7.26% | -6.41 pp | 10: `13, 44, 78, 79, 106, 110, 120, 131, 187, 188` | 25: `5, 11, 16, 24, 40, 48, 54, 55, 61, 103, 112, 116...` |
| reddit | DOM | 9.52% | 7.62% | -1.90 pp | 4: `6, 36, 100, 179` | 8: `19, 31, 79, 81, 130, 155, 171, 200` |
| reddit | SoM | 10.48% | 5.71% | -4.76 pp | 4: `36, 69, 77, 179` | 14: `2, 4, 14, 18, 19, 26, 58, 120, 139, 142, 151, 155...` |
| reddit | Vision | 6.67% | 2.38% | -4.29 pp | 4: `69, 72, 160, 201` | 13: `107, 129, 148, 150, 151, 152, 161, 171, 179, 182, 188, 189...` |

### 4.2 Failure-pattern shift by site and mode

| Site | Mode | Pattern | B0 share | B1 share | Shift |
|---|---|---|---:|---:|---:|
| classifieds | DOM | visual-missing | 50.7% | 53.7% | +3.0 pp |
| classifieds | DOM | other | 20.9% | 9.8% | -11.1 pp |
| classifieds | DOM | search-loop | 7.0% | 19.2% | +12.2 pp |
| classifieds | DOM | click-loop | 17.4% | 15.9% | -1.5 pp |
| classifieds | DOM | abandon-after-N | 3.0% | 0.9% | -2.1 pp |
| classifieds | SoM | visual-hijack/click-loop | 33.2% | 48.5% | +15.4 pp |
| classifieds | SoM | other | 47.8% | 26.2% | -21.6 pp |
| classifieds | SoM | search-loop | 13.0% | 19.8% | +6.8 pp |
| classifieds | SoM | abandon-after-N | 4.9% | 2.5% | -2.4 pp |
| classifieds | SoM | element-misground | 1.1% | 3.0% | +1.9 pp |
| classifieds | Vision | click-loop/no-text-grounding | 29.2% | 58.1% | +28.9 pp |
| classifieds | Vision | element-misground | 32.2% | 21.7% | -10.5 pp |
| classifieds | Vision | search-loop | 21.8% | 3.2% | -18.6 pp |
| classifieds | Vision | other | 16.8% | 17.1% | +0.2 pp |
| reddit | DOM | visual-missing | 41.6% | 42.8% | +1.2 pp |
| reddit | DOM | search-loop | 17.9% | 27.3% | +9.4 pp |
| reddit | DOM | click-loop | 19.5% | 19.6% | +0.1 pp |
| reddit | DOM | other | 18.4% | 9.8% | -8.6 pp |
| reddit | DOM | abandon-after-N | 2.6% | 0.5% | -2.1 pp |
| reddit | SoM | visual-hijack/click-loop | 46.8% | 62.1% | +15.3 pp |
| reddit | SoM | other | 36.2% | 20.2% | -16.0 pp |
| reddit | SoM | search-loop | 14.4% | 14.1% | -0.2 pp |
| reddit | SoM | abandon-after-N | 2.7% | 2.5% | -0.1 pp |
| reddit | SoM | element-misground | 0.0% | 1.0% | +1.0 pp |
| reddit | Vision | element-misground | 18.4% | 47.3% | +28.9 pp |
| reddit | Vision | click-loop/no-text-grounding | 45.4% | 32.2% | -13.2 pp |
| reddit | Vision | other | 21.4% | 18.5% | -2.9 pp |
| reddit | Vision | search-loop | 14.8% | 1.5% | -13.3 pp |
| reddit | Vision | abandon-after-N | 0.0% | 0.5% | +0.5 pp |

The largest site-stable qualitative shift is SoM: B1 has more mark/loop vulnerability, especially on classifieds. Vision shifts are site-modulated: classifieds Vision accumulates more no-text-grounding loops, while reddit Vision keeps a high element-misground share.

## 5. Capability Fingerprint

### 5.1 Category-level weakness

| Site | Mode | Category | B0 adjusted | B1 adjusted | Delta |
|---|---|---|---:|---:|---:|
| classifieds | DOM | B | 27.9% | 13.2% | -14.7 pp |
| classifieds | SoM | B | 30.9% | 16.2% | -14.7 pp |
| classifieds | Vision | B | 19.1% | 7.4% | -11.8 pp |
| classifieds | Vision | A | 11.1% | 0.0% | -11.1 pp |
| classifieds | SoM | C | 20.8% | 14.6% | -6.2 pp |
| classifieds | DOM | D | 11.6% | 7.0% | -4.7 pp |
| reddit | SoM | B | 17.9% | 7.1% | -10.7 pp |
| reddit | Vision | B | 10.7% | 1.2% | -9.5 pp |
| reddit | DOM | B | 15.5% | 10.7% | -4.8 pp |
| reddit | SoM | C | 6.2% | 5.3% | -0.9 pp |
| reddit | Vision | C | 4.4% | 3.5% | -0.9 pp |
| reddit | DOM | A | 0.0% | 0.0% | +0.0 pp |

B1 is especially weak on visual/reference-image categories when the observation representation does not provide stable grounding. On classifieds, the largest drops include SoM category B/D and DOM visual categories; on reddit, small denominators make some category deltas volatile, but B1 Vision is consistently weak.

### 5.2 Where B1 is better

- B1 classifieds task 5 SoM: visual-hijack/click-loop; P5,P14; 13 steps; actions click->click->click->scroll->scroll->scroll->click->scroll->...; reason=unknown; intent=Navigate to my listing of the white car and delete it.
- B1 classifieds task 64 DOM: visual-missing; P6,P14; 17 steps; actions type->type->type->type->type->type->scroll->type->...; reason=unknown; intent=Find the video game item that costs exactly $500 where I can roleplay the situation in the image.
- B1 reddit task 6 DOM: visual-missing; P6; 3 steps; actions type->click->finish; reason=unknown; intent=I'm trying to do some research about the post with this image. Tell me how many comments this post h
- B1 reddit task 69 Vision: element-misground; P1; 3 steps; actions click->click->click; reason=unknown; intent=Make a comment in this post explaining what the picture is about.

These B1-only successes are isolated rather than a global advantage. They mostly reflect stochastic/trajectory choices where B0 over-searches or commits to a different plausible page, not a broad capability edge.

### 5.3 Where B1 loses capability

- B1 classifieds task 10 SoM: visual-hijack/click-loop; P14; 3 steps; actions select_option->select_option->select_option; reason=unknown; intent=What is the seat height in inches of the smaller piece of furniture on this page?
- B1 classifieds task 17 DOM: visual-missing; P5,P6,P14; 10 steps; actions select_option->type->scroll->click->back->scroll->scroll->scroll->...; reason=unknown; intent=Show me the cheapest bike with red handlebars between $900-950.
- B1 reddit task 19 DOM: visual-missing; P6,P12,P14; 7 steps; actions click->click->back->click->click->click->click; reason=unknown; intent=Find and upvote this post. I think it was posted in /f/GetMotivated.
- B1 reddit task 107 Vision: click-loop/no-text-grounding; P5,P14; 6 steps; actions click->click->scroll->click->click->click; reason=unknown; intent=Subscribe to a forum that contains the following picture.

B1 loses most often when the task requires sustained multi-step exploration, stable post/listing localization, or correcting an initially plausible but wrong target. This matches the disagreement evidence: DOM visual-missing is universal, while B1 SoM is more likely to get captured by marks and B1 Vision often lacks text anchors.

### 5.4 Mode preference inside B1

- Classifieds: SoM is clearly B1's best mode (13.68% adjusted), ahead of DOM (8.55%) and Vision (7.26%). The marked image/text bundle helps OSClass listing tasks despite SoM visual-hijack loops.
- Reddit: DOM is B1's best mode (7.62% adjusted), ahead of SoM (5.71%) and Vision (2.38%). Reddit tasks punish weak visual grounding and comment/post localization; SoM marks do not compensate enough for the smaller model.
- Vision-only B1 is consistently the weakest arm, especially on reddit, where screen evidence without symbolic anchors produces low SR and high element/click grounding failures.

## 6. Phantom B1 Placeholder (queue_b1_after_b0 trigger after update)

B1 Phantom-SoM and Phantom-DOM runs are currently pending / cleared and are not included in this profile. Once `queue_b1_after_b0` completes, update this document with:

- B1 Phantom-SoM and Phantom-DOM raw + adjusted SR for classifieds and reddit.
- Phantom-vs-DOM / Phantom-vs-SoM overlap and unique-task counts.
- B1 phantom cost/latency/energy, including whether text-only Phantom retains DOM-like energy/cost and whether it avoids SoM visual-hijack loops.
- Updated Section 6 cross-model claim: whether Phantom remains an independent arm for the smaller local model, or whether the effect depends on B0-level capability.
