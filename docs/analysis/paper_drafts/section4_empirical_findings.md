# 4. Empirical Findings

This section reports empirical evidence that web-agent observation representations should be treated as routing arms, not only as fidelity levels. The key surprise is that **Set-of-Mark text alone**, with the marked screenshot removed, does not collapse to a DOM-like baseline. Instead, it behaves as a distinct text-only arm whose successes only partially overlap with DOM, full SoM, and vision-only observations. We refer to this arm as **Phantom-SoM**: the agent receives the `[SOM_MARKS]` textual element list and the SoM-style prompt, but no image.

Throughout this section, we distinguish three measurement conventions. **Raw SR** is the evaluator success rate in `condition_summary_v2.json`. **Adjusted SR** subtracts confirmed false-positive terminal answers on not-applicable or evaluator-mismatch tasks. **Same-task adjusted SR** uses the same task set for all arms within a site. Unless otherwise noted, claims use same-task adjusted SR on completed B0 VisualWebArena classifieds and reddit runs. We also treat small cell-to-cell differences cautiously: under same-condition repeats, we observe roughly **+/-5% task-set variance**, so individual differences below about **2 pp** should be interpreted as noise-floor evidence rather than stable rankings.

## 4.1 Setup

We evaluate a single strong API-backed web agent, denoted **B0**, on two completed VisualWebArena sites: classifieds and reddit. The completed B0 pool contains **234 classifieds tasks** and **210 reddit tasks** for each reported observation condition:

| Arm | Observation | Prompt family | Image input | Intended contrast |
|---|---|---|---|---|
| DOM | AXTree / DOM-derived text | DOM | No | Hierarchical text baseline |
| SoM | `[SOM_MARKS]` text plus marked screenshot | SoM | Yes | Full Set-of-Mark baseline |
| Vision | Screenshot without SoM marks | Vision | Yes | Visual-only baseline |
| Phantom-SoM | `[SOM_MARKS]` text only | SoM | No | Isolated marks-text representation |
| Phantom-DOM | `[SOM_MARKS]` text only | DOM | No | Prompt-family control for marks text |

The first three arms are the original Phase 1 representation baselines. Phantom-SoM is the new ablation arm. Phantom-DOM is a prompt-family control: it receives the same marks-text-only observation as Phantom-SoM but uses the DOM prompt. We report all five modes for descriptive SR, cost, and latency. For the main routing-value claim, we keep the primary drop-one oracle on the four-arm comparison used throughout the paper: DOM, SoM, Vision, and Phantom-SoM.

The original intuition was that Phantom-SoM should be either a broken SoM configuration or a weak DOM surrogate: it keeps a prompt that says the agent is operating with marked visual context, but removes the marked screenshot. The empirical results reject that collapse story. Phantom-SoM is lower than full SoM on classifieds, where marked screenshots carry clear visual grounding value, but it matches or modestly exceeds full SoM on reddit under adjusted SR.

## 4.2 Single-Mode SR, Cost, and Latency

The single-mode success rates show a site-modulated effect. On classifieds, full SoM remains the strongest individual representation. On reddit, Phantom-SoM is at least competitive with the strongest baselines, while using no image input. The table reports adjusted SR, because Figures 1, 2, 7, and 8 use episode-level `adjusted_success` for the paper comparisons. The latency column is p95 step latency from `condition_summary_v2.json`; cost is average total cost per task.

| Site | Arm | Adjusted SR | Avg cost | p95 step latency | Metric |
|---|---|---:|---:|---:|---|
| Classifieds | DOM | 14.10 | $0.043 | 37.5s | N=234 |
| Classifieds | SoM | **21.37** | $0.042 | 74.0s | N=234 |
| Classifieds | Vision | 13.68 | $0.025 | 45.0s | N=234 |
| Classifieds | Phantom-DOM | 14.53 | $0.040 | 12.8s | N=234 |
| Classifieds | Phantom-SoM | 14.53 | $0.044 | 18.2s | N=234 |
| Reddit | DOM | 9.52 | $0.052 | 73.6s | N=210 |
| Reddit | SoM | 10.48 | $0.041 | 58.9s | N=210 |
| Reddit | Vision | 6.67 | $0.023 | 55.6s | N=210 |
| Reddit | Phantom-DOM | 11.90 | $0.046 | 58.1s | N=210 |
| Reddit | Phantom-SoM | **13.81** | $0.038 | 51.4s | N=210 |

The classifieds result is the expected sanity check: when tasks benefit from visual page layout and product imagery, the marked screenshot adds useful grounding and full SoM is clearly best (**SoM 21.37 vs Phantom-SoM 14.53; N=234; adjusted**). Phantom-SoM is close to DOM on classifieds (**14.53 vs 14.10**), but this is not a dominance claim; it is inside the noise floor and far below full SoM.

The reddit result is the counterintuitive case. Removing the image does not eliminate the value of the SoM representation: Phantom-SoM matches or modestly exceeds full SoM and DOM on adjusted SR (**13.81 vs SoM 10.48 vs DOM 9.52; N=210; adjusted**). Given the variance we observe in repeats, the **+3.33 pp** gap over SoM is near the boundary of what should be treated as stable. We interpret this as evidence that Phantom-SoM is competitive on text-dominated reddit threads, not as an unconditional single-cell dominance claim. The more robust pattern is the cross-site asymmetry: **classifieds favors full SoM; reddit does not**. We treat that asymmetry as mechanism evidence rather than a setup bug: Section 5 shows a related site-modulated capability shift, with B0-to-B1 SoM visual-hijack/click-loop increasing by **+50.0 pp** on classifieds and **+33.3 pp** on reddit.

This pattern suggests that the `[SOM_MARKS]` list is doing more than serving as a caption for a screenshot. It is a compact, flat, indexed text representation. Compared with AXTree-style DOM text, it removes much of the hierarchical nesting and metadata, and presents candidate actions as a linear set of marked elements. The outcome is not uniformly better, but it can push the agent toward a different solution basin.

The cost and latency columns make the routing tradeoff concrete. On classifieds, Phantom-SoM's average cost is effectively in the same band as DOM and SoM (**$0.044 vs $0.043 vs $0.041**), but its p95 step latency is much lower than full SoM (**18.2s vs 74.0s**, roughly 4x faster). On reddit, Phantom-SoM is the cheapest of the main text/SoM-style arms (**$0.038 vs SoM $0.041 vs DOM $0.052**) and remains faster at p95 step latency than full SoM (**51.4s vs 58.9s**). These numbers support the cost-aware routing interpretation in Figures 7 and 9 without requiring Phantom-SoM to win every site.

Raw SR tells the same high-level story but should not be mixed with adjusted SR. Some arms lose points after false-positive adjustment. Because the paper claim concerns deployable task success rather than answer attempts that only appear correct under a noisy evaluator, we use adjusted SR for the main empirical comparisons.

## 4.3 Drop-One Oracle

Single-mode SR can hide routing value. A representation may have modest average SR while still solving tasks that the other arms miss. We therefore compute a drop-one oracle: form the oracle union over the four primary arms, remove one arm, and measure how much oracle SR falls. This loss is the arm's incremental contribution to the routing pool.

| Site | Largest loss | Second | Third | Fourth | Metric |
|---|---:|---:|---:|---:|---|
| Classifieds | SoM -8.55 pp | Vision -3.42 pp | Phantom-SoM -2.56 pp | DOM -2.14 pp | Drop-one oracle loss, N=234, adjusted |
| Reddit | Phantom-SoM -3.33 pp | DOM -1.90 pp | SoM -1.90 pp | Vision -1.43 pp | Drop-one oracle loss, N=210, adjusted |

The classifieds oracle is consistent with the single-mode story: full SoM contributes the most unique oracle value, followed by vision. Phantom-SoM still has a non-zero loss (**2.56 pp; N=234**), but the main effect on classifieds belongs to visual grounding.

The reddit oracle is the stronger routing signal. Phantom-SoM has the largest nominal drop-one loss in the fresh four-arm oracle (**3.33 pp; N=210**), while DOM and SoM each contribute **1.90 pp** and Vision contributes **1.43 pp**. Because these are small absolute task counts, we do not read the ordering as a precise rank claim. The important point is that Phantom-SoM is comparable to the top routing-value arms and is not subsumed by DOM, SoM, or Vision.

The overlap view supports the same conclusion. In the four-arm oracle, Phantom-SoM contributes a concrete reddit-only set of seven tasks (**7, 15, 36, 94, 157, 162, 167**) and a non-zero classifieds set as well. Two examples illustrate the kind of work this arm is doing. On reddit task 7, Phantom-SoM searched for the cake-recipe post and navigated directly to the OP recipe comment permalink. On reddit task 162, it searched within /f/wallstreetbets, scrolled hot posts, and returned the GIF URL for the retirement-account-versus-brokerage-account prompt. These are not proof of a universal mechanism by themselves, but they make the drop-one value concrete: the arm is adding recoverable successes, not only shifting aggregate percentages.

The main empirical claim is therefore not that Phantom-SoM dominates the other modes. It does not. The claim is that it is an **independent routing arm**: it opens a distinct task pool at text-only cost, with the strongest relative benefit on the text-dominated reddit site and a clear visual-grounding disadvantage on classifieds.

## 4.4 Two-Knob Ablation

The five-mode result raises a confound: is Phantom-SoM useful because of the `[SOM_MARKS]` text representation, or because the SoM prompt changes the agent's confidence and behavior even without an image? Phantom-DOM separates these factors. The full clean Phantom-DOM runs are reported above for SR, cost, and latency; for behavioral mechanism, we use the verified same-task reddit subset of **N=48**, where all four cells of the prompt-by-representation ablation were manually checked.

> **Text format shapes how the agent explores. Prompt wording tunes when the agent commits.**

The first knob is exploration shape. On the same-task reddit ablation subset, replacing AXTree text with `[SOM_MARKS]` text shifts macro behavior away from DOM-like search loops and toward Phantom-SoM-like quick decisions. The verified search-loop rate is **22.7% for DOM** but **10.8% for Phantom-SoM and 10.8% for Phantom-DOM** (**N=48; behavior metric; same-task subset**). The prompt change alone does not pull Phantom-DOM back to DOM-like exploration. This supports the representation-driven part of the hypothesis: the flat marks list, not only the SoM prompt, changes the trajectory distribution.

The second knob is commitment confidence. On the same N=48 subset, DOM and Phantom-DOM have identical raw-to-adjusted SR gaps, while Phantom-SoM has a smaller gap:

| Prompt family | Arm | Raw SR | Adjusted SR | FP gap | N/A FP | Metric |
|---|---|---:|---:|---:|---:|---|
| DOM prompt | DOM | 18.75 | 12.50 | 6.25 pp | 3 | N=48, raw/adjusted |
| DOM prompt | Phantom-DOM | 18.75 | 12.50 | 6.25 pp | 3 | N=48, raw/adjusted |
| SoM prompt | Phantom-SoM | 18.75 | 16.67 | 2.08 pp | 1 | N=48, raw/adjusted |
| SoM prompt | SoM | 22.92 | 16.67 | 6.25 pp | 3 | N=48, raw/adjusted |

The aggregate SR equality should not be overread as task-level identity: equal counts such as 6/48 can occur with different solved-task sets. The robust signal is the false-positive pattern. DOM-prompt arms have the larger false-positive gap (**DOM and Phantom-DOM: 3 N/A false positives, 6.25 pp gap; N=48**). The SoM-prompt Phantom arm has fewer N/A false positives (**1 N/A false positive, 2.08 pp gap; N=48**). This indicates that prompt wording affects terminal-action calibration: when the model decides it has enough evidence to `finish`.

The two-knob account reconciles the apparent tension. The representation is the novel routing axis because it changes the agent's default exploration path. The prompt is a secondary but real tuning knob because it changes commitment confidence. Both are needed to explain the ablation. A representation-only story misses the FP gap, while a prompt-only story cannot explain why Phantom-DOM follows Phantom-SoM rather than DOM on search-loop behavior.

These findings explain why Phantom-SoM can be valuable despite not winning every single-mode comparison. Routing benefits depend on complementarity, not only average SR. A flat marks list can be worse for tasks that need hierarchy or visual layout, yet better for tasks where the same hierarchy induces over-searching. The practical implication is a cost-aware cascade: try cheap text representations first, use behavioral signals to detect when their exploration is unproductive, and escalate to full SoM when visual grounding is likely to matter.
