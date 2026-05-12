# 5. Mechanism Interpretability

## 5.1 Overview and Theoretical Framing

Why does Phantom-SoM sometimes achieve DOM-like cost while retaining part of the SoM signal? The mechanism evidence points to a phantom routing space in the residual stream: when the model receives flat Set-of-Mark text without the annotated image, it does not simply collapse to DOM. Instead, it occupies a mode whose text-axis geometry is close to DOM/P-text and whose image-axis geometry remains separated from full SoM.

This section is the Zoom-4 layer of the paper's four-level account. Zoom 1 defines the architectural intervention, "skip the annotated image"; Zoom 2 measures the behavioral axes of text payload, prompt family, and image presence; Zoom 3 links the observed behavior to Mirage-style no-image visual reasoning and prompt-format sensitivity; Zoom 4 asks where the resulting mode is represented and whether it is causally used by the model. We index layers L0-L36, where L0 is the embedding-block output and L1-L36 are the 36 transformer decoder block outputs.

The analysis builds on the linear-readable and steerable circuit framework of Wu et al., which uses mode means, PCA geometry, and mean-difference activation steering to study tool selection, and on work showing middle-layer cross-modal information flow in VLMs \citep{wu2026toolcalling,kaduri2024whatsintheimage}. Our setting is not a replication of those papers. It is a multimodal web-agent application of the same representation-level question: whether a behaviorally useful routing arm is linearly readable, partially steerable, and causally active inside the model.

Four mechanism claims organize the evidence. First, observation modes are geometrically separable in the residual stream: the PCA cosine-gap analysis reaches AUROC 1.0 on the main mode contrasts. Second, Phantom-SoM is a mid-layer text-axis sibling of DOM/P-text, not an image-axis sibling of full SoM. Third, SoM-to-no-image activation patching displaces target continuations at mid layers L11-L17 with consistent magnitude across classifieds and reddit. Fourth, the shortcut trigger is flat element-list format rather than a specific token pattern; AXTree hierarchy is the unique format that preserves the early L04 image-axis peak in the aggregate.

The cross-site evidence stack is deliberately defensive. Per-task H1 fragility shows the dichotomy is an aggregate mechanism rather than a deterministic per-task law. Reverse-tier classifieds runs defend against strong-tier selection bias. Reddit format variation replicates the shortcut direction with cleaner mid-layer peaks. Reddit Method 4.2 replicates the Mirage signature: Phantom-SoM remains close to DOM on the text axis while separating from SoM on the image axis. Paper 1 uses these results for mechanism interpretation only; routing implementation is deferred to paper 2, consistent with the paper-planning scope split.

## 5.2 Method 4.2: PCA Cosine Gap

Method 4.2 extracts hidden states from Qwen3-VL-4B B1 runs and compares observation modes by layer. For each mode pair and layer, we compute the cosine gap between hidden-state means, evaluate AUROC by projecting examples onto the mean-difference direction, and summarize per-mode geometry through PCA top-10 variance. The classifieds baseline contains 288 examples, formed from 24 strong-tier tasks, two archived steps, and six modes, over 37 indexed layers.

The robustness suite passes all five checks in the plan. Label permutation leaves the real AUROC 9.8 standard deviations above the permuted baseline. Per-task analysis is positive for all 24 tasks. Step 2 and step 5 curves are invariant at the mechanism level. The L23 silhouette score is at least 0.5, showing nontrivial clustering. Bootstrap 95% confidence intervals are tight, with widths of roughly 4-15% of the corresponding means.

The key classifieds snapshot is the L17 cosine-gap table:

| Pair at L17 | Cosine gap | 95% CI | AUROC |
|---|---:|---:|---:|
| P-SoM <-> P-text | 0.0028 | [0.0027, 0.0029] | 1.000 |
| DOM <-> P-prompt | 0.0013 | [0.0012, 0.0014] | 1.000 |
| P-SoM <-> SoM | 0.0413 | [0.0403, 0.0422] | 1.000 |
| DOM <-> Vision | 0.0547 | [0.0531, 0.0563] | 1.000 |

The reddit replication lands the same qualitative geometry. At L17, P-SoM is close to DOM with cosine gap 0.0098 and close to P-text with gap 0.0027, while P-SoM-to-SoM remains much larger at 0.0423 and P-SoM-to-Vision at 0.0457. The DOM-to-Vision image-axis peak is L04 with cosine gap 0.0687 and AUROC 1.0.

This is the Mirage signature in geometric form. Phantom-SoM is not represented as a weakened image mode. At the mid-layer disruption locus, it is a text-axis sibling of DOM/P-text, while the image-axis distance to full SoM remains large.

## 5.3 Method 4.4: Mean-Difference Activation Steering

Method 4.4 tests whether the readable mode direction can be used as a steering direction. For each layer, we form a mean-difference vector between Phantom-SoM-like and DOM-like hidden states, add it to each input at generation time with scaling factor $\alpha$, and evaluate whether the continuation moves toward the target mode while preserving the JSON action envelope. Following HDMI's evaluation vocabulary, reliability is the harmonic mean of completeness and selectivity, not a raw shift rate \citep{khorasani2026hdmi}.

The v2 sweep covers layers [11, 17, 23, 29, 33, 34] and $\alpha \in [1,2,5,10,20]$, for 45 completed cells in the plan summary. The original L17, $\alpha=5$ smoke result reported H-mean 0.44, but the full sweep lowers that cell to 0.16. The plan records this as a smoke-variance artifact from notes 126/127: a 4-cell smoke was too small to support a sweet-spot claim.

The strongest full-sweep cell is L33, $\alpha=10$, with H-mean 0.33. Its completeness is 38% and its selectivity is 29%. The layer profile is the important result: mid layers L11-L23 preserve the JSON envelope with 100% selectivity but have low completeness, while late-layer L33 produces the largest shifts but frequently over-steers the continuation out of the expected JSON format.

This creates a probe-causal dissociation. The mid-layer geometry is cleanly readable and causally implicated by patching, but fixed mean-difference steering is only partially reliable. The 0.33 H-mean is therefore an evidence ceiling for Method 4.4, not a final control result. Section 8 should treat LA-HDMI and SAE feature steering as future work motivated by this ceiling, without claiming that either method has already improved it.

## 5.4 Stage 2/3: Activation Patching for a Causal Mid-Layer Mechanism

Activation patching provides the causal test. For each task, the clean/source run and corrupt/target run use the same archived browser step and deterministic 50-token continuation. In the core SoM-to-Phantom-SoM setup, the source prompt is `som`: task instruction, SoM prompt family, flat `[SOM_MARKS]` text, and annotated screenshot. The target prompt is `phantom_som`: the same instruction, same prompt family, and same `[SOM_MARKS]` text, but no image. Source hidden states are cached by layer, injected into the final input-token position of the target on the first forward pass, and subsequent decoding proceeds normally through the model cache.

Each patched continuation is scored against the unpatched source and target continuations. The main disruption statistic is the drop in `token_overlap_to_target`; Levenshtein distance to target is the paired backup. Layer-wise tests compare each grid layer to the final-layer reference using task-paired differences and Holm-Bonferroni correction across the canonical grid. Random-injection controls replace source hidden states with Gaussian tensors matched to source activation mean and standard deviation.

The Stage 2 P-SoM<->SoM dashboard now contains ten completed cells, including reddit F/G and the reddit random control:

| Cell | Site | Direction | Mid-layer target-overlap drop | Holm status |
|---|---|---|---:|---|
| A | cls | SoM->P-SoM forward | -0.32 at L17 | significant |
| B | cls | P-SoM->SoM reverse | -0.16 at L17 | significant |
| C | cls | reverse-tier forward | -0.02 at L17 | null |
| D | cls | strong-tier reverse | -0.18 at L17 | significant |
| E | cls | random injection | -0.03 uniform | negative control |
| F | reddit | SoM->P-SoM forward | -0.21 at L17 | significant |
| G | reddit | P-SoM->SoM reverse | -0.18 at L17 | significant |
| Cr/Dr | reddit | 2x2 controls | -0.15 to -0.18 | significant |
| Er | reddit | random injection | approximately 0 uniform | negative control |

Stage 3 extends this from P-SoM to the three no-image arms, testing whether the image-feature axis is shared across DOM, P-text, and P-prompt targets. The table below reports per-task-paired Δoverlap-to-target from the patching_continuation_results.json under each cell directory, with the layer at which the disruption peaks.

| Site | SoM->DOM | SoM->P-text | SoM->P-prompt | best-L Δ range |
|---|---:|---:|---:|---:|
| cls | -0.309 at L17, -0.352 at L18 (best) | -0.255 at L17, -0.270 at L12 (best) | -0.223 at L17, -0.273 at L13 (best) | [-0.273, -0.352] |
| reddit | -0.335 at L11, -0.255 at L17, -0.338 at L14 (best) | -0.244 at L11, -0.236 at L17, -0.330 at L15 (best) | -0.233 at L11, -0.191 at L17, -0.322 at L14 (best) | [-0.322, -0.338] |

All six Stage 3 cells are now closed. Two observations carry the cross-site claim. First, every cell's best layer falls inside the L12-L18 mid-layer window, and every cell's best Δoverlap-to-target is between -0.27 and -0.35. The mid-layer fusion locus is therefore not a single layer index but a tight 7-layer window that transfers across cls and reddit. Second, the interpretation is additive rather than arm-specific: a SoM source state displaces DOM, P-text, and P-prompt targets toward the source with similar magnitude, implying a shared image-feature substrate across all three no-image arms. The negative controls, Cell E at -0.03 and Cell Er near zero, rule out a generic nonzero-injection explanation.

## 5.5 Image-Axis Peak-Layer Dichotomy and H1 Format Variation

The cleanest single-pair signature is the image-axis peak-layer dichotomy. Across eight image-presence contrasts, the no-image side's text format predicts the peak layer with zero overlap. If the no-image side is AXTree text, the image-axis cosine gap peaks at L04 in all four pairs: DOM<->Vision, DOM<->SoM, P-prompt<->Vision, and P-prompt<->SoM. If the no-image side is `[SOM_MARKS]` or another flat marks text, the peak shifts to L17-L36 in all four pairs: P-text<->Vision, P-text<->SoM, P-SoM<->Vision, and P-SoM<->SoM.

The refined H1 is a pretraining co-occurrence shortcut: when the input contains a flat element-region list, the model activates a visual-grounding pathway even if the image is absent. Prompt-format sensitivity makes this plausible at the input level \citep{sclar2024promptformat}; Method 4.2 shows it as a layer-resolved internal signature.

The format-variation grid contains ten modes: six marks-like variants, two controls, and DOM/SoM baselines. In the classifieds strong-tier baseline, all six marks-like variants peak at L36, the hash-ID control also peaks at L36, the plain-sentence control peaks at L17, and the DOM baseline preserves the L04 peak. Because L36 is the boundary layer, this is best read as a strong late/monotonic signature rather than as a precise late-layer mechanism.

The classifieds reverse-tier run reproduces the strong-tier shape. The six marks-like variants and hash-ID control again peak at L36, the plain-sentence control moves to L22, and DOM remains at L04. This defends H1 against the selection-bias concern that strong-tier curation alone created the pattern.

The reddit format run is cleaner for the mid-layer interpretation. Four of six marks-like variants peak at L17, the plain-sentence control peaks at L17, hash-ID control returns to L04, and DOM remains at L04. The plan flags the two L04 marks-like reddit variants as small-n caveats rather than a reversal. Cross-site, the safe claim is directional: flat list formats tend to delay image-axis separation into mid/late layers, while AXTree hierarchy uniquely preserves the early L04 image-axis peak. The reddit curve reveals the true L11-L17 fusion locus more clearly than the classifieds L36 boundary artifact.

## 5.6 Convergent Four-Vertical-Defense Evidence Stack

The first defense is per-task fragility. On 45 classifieds task-step pairs, only 11% satisfy the strict per-task dichotomy, even though aggregate marks-like peaks are later than AXTree peaks. This prevents over-claiming: H1 is a population-level mechanism signature with task variability, not a deterministic rule for every trajectory.

The second defense is selection-bias robustness. The classifieds reverse-tier run replicates the strong-tier H1 pattern, including L36 marks-like peaks and L04 DOM baseline. The shortcut signature is therefore not an artifact of selecting tasks where SoM beats DOM.

The third defense is cross-site H1. Reddit does not reproduce the exact boundary-layer shape, but it reproduces the direction of the indexed-list shortcut with a cleaner L17 mid-layer peak for four of six marks-like formats. The site changes the curve shape, not the basic interpretation.

The fourth defense is cross-site Mirage geometry. Reddit Method 4.2 reproduces the central relation: P-SoM is close to DOM/P-text at L17 and far from SoM/Vision on the image axis, with AUROC 1.0 on the key contrasts. This supports cross-site generalization of the mechanism claim, not B0/B1 capability scaling.

Two additional defenses remain deferred rather than folded into the claim: P2 cross-family Phi-3.5-Vision and P3 larger Qwen2-VL-7B. The current evidence is sufficient for the single-model, cross-site Qwen3-VL-4B mechanism section; family and capacity generalization belong in future work or Section 7.

## 5.7 Mid-Layer Mechanism Versus Behavioral Hero Status

The Method 4.2 geometry assigns the four phantom-boundary modes into two text-format clusters at the mid-layer fusion locus. At L17 on classifieds, P-text and Phantom-SoM share an essentially identical residual-stream representation (cosine gap 0.0028, 95% CI [0.0027, 0.0029]), and DOM and P-prompt share another (cosine gap 0.0013, 95% CI [0.0012, 0.0014]). The prompt family makes no measurable contribution to the L11-L17 geometry; only the text payload (hierarchical AXTree versus flat indexed list) separates the modes at this depth. The reddit replication preserves the same clustering, with Phantom-SoM-to-P-text cosine gap 0.0027 at L17.

This creates an explanatory puzzle the Section 1 forest plot cannot resolve at the mechanism layer alone. The drop-one oracle figure (`fig_meta_forest.png`) places Phantom-SoM as the unique hero with confidence interval strictly above zero on the classifieds and reddit panels, while the P-text and P-prompt drop-one effects straddle zero in the same meta-analysis. If P-text and Phantom-SoM share an identical mid-layer signature, why does only Phantom-SoM earn drop-one hero status?

The resolution is that the phantom routing space is structured as a $2\times 2$ grid in (text-format) $\times$ (prompt-family), and the mid-layer mechanism only measures the text-format axis. The prompt-family axis acts at the policy and action-vocabulary level rather than at L11-L17 residual stream geometry. The behavioral evidence is consistent with this. With the text payload held fixed at the flat indexed list, Phantom-SoM and P-text still have low click-target overlap on reddit (P-text-versus-P-SoM click-target Jaccard 0.484; the standard behavioral notation is in Section 4), and Phantom-SoM shifts the macro action vocabulary relative to P-text by typing -0.0658 and tab-focus +0.0828. These are decision-prior shifts that the cosine-gap geometry does not capture.

Phantom-SoM uniquely earns hero status because it is the only phantom-boundary mode that occupies both axis-1 (flat-text mid-layer cluster, which the mechanism explains) and axis-2 (referential SoM-style prompt commitment, which the mechanism does not). P-text has axis-1 but not axis-2, P-prompt has axis-2 but not axis-1, and DOM has neither. The mid-layer mechanism is necessary but not sufficient for behavioral hero status, and the prompt-family decision prior is an orthogonal contributor visible only at policy-level macro and micro evidence rather than at L17 residual geometry.

This dissociation is itself a paper-grade finding rather than a limitation. It establishes that observation-mode routing decisions cannot be read off the mid-layer signal alone; a deployment-time router would have to jointly model the geometric mid-layer cluster and the prompt-level decision prior. Paper 2 inherits this as a structural constraint on the routing implementation it owns. The same dissociation also constrains future single-axis interventions in Section 8: LA-HDMI and SAE feature steering operate on the residual stream and can target the axis-1 mechanism, but axis-2 prompt-prior effects require prompt-level rather than activation-level intervention.

## 5.8 Discussion and Limits

The main limit is the Method 4.4 ceiling. The cosine-gap and patching evidence point to L11-L17 as the readable and causally active fusion region, while the best fixed mean-difference steering cell is late, L33 with $\alpha=10$, and has H-mean 0.33 because completeness and selectivity trade off. This supports a mechanism interpretation but not a strong deployment-time steering claim.

The second limit is layer precision. Classifieds H1 peaks often hit L36, while reddit reveals cleaner L17 peaks. The robust claim is therefore an effect-direction claim: AXTree hierarchy preserves early image-axis separation, and flat element-list formats delay that separation into mid/late computation. We should not claim that every site or task has an identical peak layer.

Literature positioning should stay modest. Section 5 applies the linear-readable, steerable, and mid/late-layer circuit framework to multimodal web-agent observation modes \citep{wu2026toolcalling,kaduri2024whatsintheimage,khorasani2026hdmi}; NEEDS_BIB: \citep{fayyaz2026steermoe}. It should not claim novelty as the first such circuit or the first use of marked text. The contribution is controlled scientific characterization of the phantom boundary.

Finally, AXTree hierarchy is the unique defeating format in the aggregate, but the reason hierarchy defeats the shortcut remains open. The plan records one attribution-pending hypothesis: hierarchy or indentation tokens may redirect cross-modal attention before the flat-list shortcut activates. That should be treated as a supplement question, not as a Section 5 finding.

## NOTE FOR HUMAN

Bibkeys needed: NEEDS_BIB: `fayyaz2026steermoe`. Existing keys found and used: `wu2026toolcalling`, `khorasani2026hdmi`, `kaduri2024whatsintheimage`, `sclar2024promptformat`.

Behavioral content to relocate from current `section5_mechanism_reddit.md`: lines 17-75 should move to Section 4 or a new behavioral-routing subsection. Specifically, lines 17-23 are reddit substrate framing; lines 25-35 are Axis 1 text-payload behavior; lines 37-47 are Axis 2 prompt behavior; lines 49-59 are Axis 3 image behavior; lines 61-67 are compound P-SoM versus DOM behavior; lines 69-75 are scope/noise limitations. Lines 1-15 are method material that was retained conceptually but must use the new L0-L36 layer convention. Line 77 should be deleted or replaced because routing implementation is now paper-2, not paper-1 Section 6.

Stage 3 numbers verified 2026-05-12 from full per-task paired-test computation on `patching_continuation_results.json` (each cell, 24 tasks × 36 layers). H-d-cls best L18 Δ=-0.352, H-d-red best L14 Δ=-0.338, H-t-cls best L12 -0.270, H-t-red best L15 -0.330, H-p-cls best L13 -0.273, H-p-red best L14 -0.322. All 6 cells' best layer lands in L12-L18 mid-layer window, Δ range [-0.27, -0.35]. The L17-only column previously cited in plan §5.2 reads -0.309/-0.255/-0.223 (cls) and -0.255/-0.236/-0.191 (reddit); plan §5.2 has been updated to record best-layer Δ instead of L17-only Δ.

Pending items: (a) Bibkey `fayyaz2026steermoe` for SteerMoE expert routing remains NEEDS_BIB; check `paper.bib` and add if missing. (b) Method 4.4 sweep description should be "45 completed cells out of a 6x5 layer-alpha grid plus 3 placeholder cells that did not finish", not "45/48-cell sweep" (the 48-cell wording in plan §5.3 implies a 48-cell denominator that was never executed). (c) After this section settles, behavioral content in current `section5_mechanism_reddit.md` lines 17-75 should be relocated to Section 4; that file should then be deleted or renamed to a stub pointing to the new Section 5.
