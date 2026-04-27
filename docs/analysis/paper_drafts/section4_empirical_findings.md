# 4. Empirical Findings

This section reports the first empirical evidence for cost-aware routing over web-agent observation representations. The key surprise is that **Set-of-Mark text alone**, with the marked screenshot removed, does not collapse to a DOM-like baseline. Instead, it behaves as a fourth routing arm whose successes only partially overlap with DOM, full SoM, and vision-only observations. We refer to this arm as **Phantom-SoM**: the agent receives the `[SOM_MARKS]` textual element list and the SoM-style prompt, but no image.

Throughout this section, we distinguish three measurement conventions. **Raw SR** is the evaluator success rate directly recorded by the episode summary. **Adjusted SR** subtracts confirmed false-positive terminal answers on not-applicable or evaluator-mismatch tasks. **Same-task adjusted SR** uses the same task set for all arms within a site. Unless otherwise noted, claims in Sections 4.2 and 4.3 use same-task adjusted SR on the completed B0 VisualWebArena classifieds and reddit runs. We report the denominator and metric convention with each claim because the difference between raw and adjusted rates is material for this benchmark.

## 4.1 Setup

We evaluate a single strong API-backed web agent, denoted **B0**, on two completed VisualWebArena sites: classifieds and reddit. The completed B0 pool contains **234 classifieds tasks** and **210 reddit tasks** for each of the four compared observation arms (**N=234 classifieds; N=210 reddit; same-task adjusted unless marked otherwise**):

| Arm | Observation | Prompt family | Image input | Intended contrast |
|---|---|---|---|---|
| DOM | AXTree / DOM-derived text | DOM | No | Hierarchical text baseline |
| SoM | `[SOM_MARKS]` text plus marked screenshot | SoM | Yes | Full Set-of-Mark baseline |
| Vision | Screenshot without SoM marks | Vision | Yes | Visual-only baseline |
| Phantom-SoM | `[SOM_MARKS]` text only | SoM | No | Isolated marks-text representation |

The first three arms are the original Phase 1 representation baselines. Phantom-SoM is the new ablation arm. The original intuition was that Phantom-SoM should be either a broken SoM configuration or a weak DOM surrogate: it keeps a prompt that says the agent is operating with marked visual context, but removes the marked screenshot. The empirical results reject that intuition. Phantom-SoM is lower than full SoM on classifieds but competitive on reddit, and its task-success pool is not subsumed by DOM, SoM, or vision (**N=234/210; adjusted/drop-one oracle evidence below**).

We additionally use a partial reddit ablation, **Phantom-DOM**, to separate the effects of representation format from prompt wording. Phantom-DOM receives the same `[SOM_MARKS]` text as Phantom-SoM and no image, but uses the DOM prompt rather than the SoM prompt. This 2-by-2 contrast is currently available on a **same-task reddit subset of N=48**. The N=48 ablation is not used to claim final task-level SR superiority; it is used only for mechanism: which knob changes exploration strategy, and which knob changes commitment confidence.

## 4.2 Single-Mode SR

The single-mode success rates show that Phantom-SoM is not simply a failed or degenerate arm. On classifieds, full SoM remains the strongest individual representation, while Phantom-SoM trails the other arms. On reddit, however, Phantom-SoM is the strongest single arm by same-task adjusted SR.

| Site | DOM | SoM | Vision | Phantom-SoM | Metric |
|---|---:|---:|---:|---:|---|
| Classifieds | 14.10 | **21.37** | 13.68 | 11.97 | Same-task adjusted SR, N=234 |
| Reddit | 9.52 | 10.48 | 6.67 | **10.95** | Same-task adjusted SR, N=210 |

The classifieds result is the expected sanity check: when tasks benefit from visual page layout, the marked screenshot adds useful grounding and full SoM is best (**SoM 21.37 vs Phantom-SoM 11.97; N=234; adjusted**). The reddit result is the counterintuitive case: removing the image does not eliminate the value of the SoM representation. Phantom-SoM slightly exceeds full SoM and DOM on adjusted SR (**Phantom-SoM 10.95 vs SoM 10.48 vs DOM 9.52; N=210; adjusted**). The magnitude is small in absolute SR, but the direction matters because Phantom-SoM is text-only and avoids image-token cost.

This pattern suggests that the `[SOM_MARKS]` list is doing more than serving as a caption for a screenshot. It is a compact, flat, indexed text representation. Compared with AXTree-style DOM text, it removes much of the hierarchical nesting and metadata, and presents candidate actions as a linear set of marked elements. The outcome is not uniformly better, but it can push the agent toward a different solution basin. The rest of this section tests that routing interpretation directly.

Raw SR tells the same high-level story but should not be mixed with the adjusted table. For example, the current episode summaries record higher raw rates for some baseline arms before false-positive adjustment. Because the paper claim concerns deployable task success rather than answer attempts that only appear correct under a noisy evaluator, we use the adjusted rates above for the main empirical comparisons (**raw/adjusted distinction; N=234/210**).

## 4.3 Drop-One Oracle

Single-mode SR can hide routing value. A representation may have modest average SR while still solving tasks that the other arms miss. We therefore compute a drop-one oracle: form the oracle union over all four arms, remove one arm, and measure how much oracle SR falls. This loss is the arm's incremental contribution to the routing pool.

| Site | Largest loss | Second | Third | Fourth | Metric |
|---|---:|---:|---:|---:|---|
| Classifieds | SoM -7.69 pp | Vision -3.85 pp | DOM -2.14 pp | Phantom-SoM -1.71 pp | Drop-one oracle loss, N=234, adjusted |
| Reddit | SoM -2.86 pp | **Phantom-SoM -2.38 pp** | Vision -1.90 pp | DOM -1.43 pp | Drop-one oracle loss, N=210, adjusted |

The classifieds oracle is consistent with the single-mode story: full SoM contributes the most unique oracle value, vision contributes next, and Phantom-SoM has the smallest unique loss (**N=234; adjusted**). Even there, however, Phantom-SoM is not zero. Removing it still loses 1.71 percentage points of oracle success, meaning that a non-empty set of tasks is solved only by the marks-text-only arm.

The reddit oracle is the stronger result. Phantom-SoM has the second-largest drop-one loss, behind only full SoM (**-2.38 pp vs SoM -2.86 pp; N=210; adjusted**). This directly rejects the "Phantom-SoM is noise" hypothesis. If the arm were merely a degraded DOM or an unreliable SoM prompt artifact, dropping it should have little effect after DOM, SoM, and vision are already in the pool. Instead, it contributes more incremental oracle value than DOM and vision on reddit.

Qualitative audits of only-solved tasks point to different specialization mechanisms. Full SoM and vision disproportionately cover screen-layout or visual-reference cases, where spatial grounding matters (**classifieds only-set: SoM page-screen 61%, vision page-screen 56%; reddit only-set: SoM ref-image 83%, vision page-screen 75%; audit categories, adjusted only-set**). DOM's successes are tied to sustained hierarchical exploration, especially on page-screen reddit cases. Phantom-SoM, by contrast, tends toward compact quick decisions from the flat marks list, including a classifieds only-set skew toward reference-image tasks (**classifieds Phantom-SoM only-set: ref-image 75%; audit category; adjusted only-set**).

The main empirical claim is therefore not that Phantom-SoM dominates the other modes. It does not. The claim is that it is an **independent routing arm**: it opens a distinct task pool at nearly DOM-like text-only cost. This is the core motivation for routing over observation representations rather than treating SoM text as inseparable from a marked image.

## 4.4 Two-Knob Ablation

The four-arm result raises a confound: is Phantom-SoM useful because of the `[SOM_MARKS]` text representation, or because the SoM prompt changes the agent's confidence and behavior even without an image? The Phantom-DOM ablation separates these factors. Phantom-DOM uses the **DOM prompt** with the **same `[SOM_MARKS]` text-only observation** used by Phantom-SoM. On the currently verified reddit subset, it reveals a two-knob mechanism:

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

These findings also explain why Phantom-SoM can be valuable despite not winning every single-mode comparison. Routing benefits depend on complementarity, not only average SR. A flat marks list can be worse for tasks that need hierarchy or visual layout, yet better for tasks where the same hierarchy induces over-searching. The practical implication for P79 is a cost-aware cascade: try cheap text representations first, use behavioral signals to detect when their exploration is unproductive, and escalate to full SoM only when the image is likely to add grounding value.
