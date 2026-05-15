## 3. Phantom-SoM: Definition and Ablation Setup

### 3.1 Set-of-Mark Bundle

Set-of-Mark (SoM) prompting converts a screenshot into an indexed visual interface. The standard bundle has two synchronized parts: a marked image, where page regions are overlaid with bounding boxes and numeric IDs, and a text legend that maps those IDs to element descriptions [Yang et al. 2023]. We serialize the text component as:

```text
[SOM_MARKS]
[id=N] role 'label'
...
[/SOM_MARKS]
```

Full SoM gives both pieces to the agent at the same step. The prompt says the `[SOM_MARKS]` list and annotated screenshot refer to one another, and the action schema asks the model to click, type, or select by `element_id` when possible. VisualWebArena and SeeAct use the same broad pattern: visual evidence is paired with grounding information so the model can convert perception into browser actions [Koh et al. 2024; Zheng et al. 2024].

This bundle is the assumption Phantom-SoM ablates. The question is not whether marked screenshots are useful; Section 4 shows that they often are. The question is whether the text half of the bundle is only an image key, or itself a distinct text representation.

### 3.2 Phantom-SoM

We define **Phantom-SoM** as:

```text
Phantom-SoM(page) =
  prompt = SoM prompt
  text   = SOM_MARKS(page)
  image  = None
```

Phantom-SoM uses the same SoM prompt family as full SoM and the same `[SOM_MARKS]` text, but removes the page screenshot passed to the model. In code, `p79/experiment/som.py::prepare_observation_for_mode` handles `mode in ("phantom_som", "phantom_dom", "phantom_text")` by calling `_build_som_result(...)`, then returning the generated `som_text` with `marked_image=None` (`phantom_dom` is the legacy mode value retained as alias for paper-grade run dirs; `phantom_text` is the current canonical name for P-text). The rendered screenshot path is retained for debugging; the model does not receive it.

The critical property is that the prompt remains the SoM prompt. It still describes an annotated screenshot with numbered boxes, even though the observation channel contains no page screenshot. We call this the **mirage prompt** property: the behavioral scaffold of SoM is preserved while the visual substrate is removed.

Phantom-SoM is a cost intervention, and the structure of the saving is best stated relative to two different baselines.

**Relative to DOM**, Phantom-SoM is essentially free. The `[SOM_MARKS]` block is produced by a regex filter over the VisualWebArena accessibility-tree text that the DOM baseline already consumes. VWA serializes interactive elements with bracketed numeric IDs of the form `[N] role 'label'`; in our implementation `_extract_text_marks` (see `p79/experiment/som.py`) walks `obs_text` line by line, keeps the lines that match `\[\d+\]`, and returns `(id, label)` pairs that are wrapped in a `[SOM_MARKS] ... [/SOM_MARKS]` block. There is no bounding-box lookup and no image work in this path; bounding boxes are only used by full SoM when drawing numeric labels onto the screenshot. Empirically this leaves text length roughly unchanged: holding the system prompt fixed at the DOM family, median total input is 3437 tokens for DOM versus 3661 for P-text on reddit, and 3008 versus 2948 on classifieds — within ±7% on both sites. The two formats see the same accessibility content; what differs is the surface form (flat indexed list versus nested hierarchy with url/tab metadata). We treat this as a representation property and study its behavioral effect mechanistically in Section 5; for cost accounting the implication is that switching DOM → Phantom-SoM at deployment time costs at most a regex pass over the same observation.

**Relative to full SoM**, Phantom-SoM saves two real layers of cost. (i) The on-server annotation step that draws numeric labels onto the page screenshot is unique to full SoM and is omitted in a Phantom-SoM deployment; in our research code we retain the marked image on disk for debugging, which is why both modes report ~30 ms median obs-prepare latency, but a production variant skips the draw entirely and recovers roughly 30 ms and on the order of $2e-5 per step. (ii) The marked screenshot is no longer encoded as image tokens at inference, removing the visual-encoding stage. Comparing step-level `tokens.input` medians between full SoM and P-text gives a same-prompt image-channel estimate of 733 input tokens per step on reddit (SoM 4275 versus P-text 3542; P-text partial live run, 145 episodes) and 1064 on classifieds (4034.5 versus 2970.5; 234 episodes). We attribute this median gap to the marked screenshot under our backend tokenization. These are the tokens that drive prompt-processing time, memory pressure, and time-to-first-token in multimodal serving (see Section 2.4); skipping them is the dominant component of the cost difference between full SoM and Phantom-SoM.

The combined picture is that Phantom-SoM sits at roughly DOM cost (its observation is a text filter of the same AXTree) while replacing the visual-evidence half of SoM with nothing at all. This is also a deployment-level claim, not only an analytical one: an existing full-SoM agent can be converted into a Phantom-SoM agent by changing only what the server forwards to the model — keep the `[SOM_MARKS]` text that is already being produced from the accessibility tree, stop drawing labels onto the screenshot, and stop attaching the marked image to the inference request. The model interface, the prompt, the action schema, and the evaluator are unchanged. There is no retraining, no new data path, and no marks-side prompt edit; the only mutation is on the backend annotation pipeline, after the AXTree filter and before the model call. We use this property in Section 4 to interpret cost-versus-success comparisons as deployment-time tradeoffs rather than research-only configurations, and in Section 5 to argue that Phantom-SoM's behavior is a property of the format the model already saw inside SoM, not an emergent capability that requires new infrastructure.

### 3.3 P-text

**P-text** is the disambiguation ablation:

```text
P-text(page) =
  prompt = DOM prompt
  text   = SOM_MARKS(page)
  image  = None
```

Its observation is identical to Phantom-SoM: `[SOM_MARKS]` text only, no page screenshot. The only intended change is the system prompt. In both B0 (`p79/agents/proxy_api_agent.py`) and B1 (`p79/agents/qwen3vl_agent.py`), `_system_prompts["phantom_som"]` maps to the SoM prompt, while `_system_prompts["phantom_dom"]` (and the alias `_system_prompts["phantom_text"]`) maps to the DOM prompt. For `som`, `phantom_som`, `phantom_dom`, and `phantom_text`, the agent passes through the `[SOM_MARKS]...[/SOM_MARKS]` text directly.

This cell separates representation from prompt wording. If P-text behaves like Phantom-SoM, the flat marks text is driving behavior. If it behaves like DOM, the prompt is doing more of the work.

### 3.4 The Complete 2×2 Ablation Matrix and the P-prompt Fourth Cell

The core ablation is a complete prompt-by-representation matrix:

| | DOM prompt | SoM prompt |
|---|---|---|
| AXTree obs | DOM | **P-prompt** |
| `[SOM_MARKS]` obs | P-text | Phantom-SoM |

Full SoM is adjacent to this 2×2: it uses the SoM prompt, the same `[SOM_MARKS]` text, and the marked screenshot. Vision is a separate screenshot-only baseline. Together with the 2×2 above, this yields the six modes we evaluate empirically: DOM, P-text, P-prompt, Phantom-SoM, full SoM, and Vision.

### 3.4.1 P-prompt (AXTree observation + SoM prompt)

**P-prompt** is the fourth cell of the 2×2:

```text
P-prompt(page) =
  prompt = SoM prompt
  text   = AXTree(page)
  image  = None
```

The agent receives the AXTree accessibility-tree text the DOM baseline already consumes, but under the SoM system prompt — i.e. the model is told to act on an annotated screenshot with numeric IDs even though no such image is present and the text is hierarchical AXTree, not the flat `[SOM_MARKS]` form. In code, `_system_prompts["phantom_prompt"]` maps to the SoM prompt family and the observation channel passes through the unmodified AXTree text (no `[SOM_MARKS]` filter applied).

This cell is intentionally a mismatched-format-against-prompt design point. The SoM system prompt expects flat indexed `[N]` references to elements, while AXTree text uses an independent hierarchical accessibility-tree ID space; the two ID systems do not in general agree on a given element's identifier. There are three substantive reasons we include P-prompt rather than excluding it as a malformed hybrid:

(i) **The 2×2 isolates the prompt-format axis independently of the text-representation axis.** Without P-prompt, the prompt wording effect can only be measured along `[SOM_MARKS]` text (P-text vs Phantom-SoM). P-prompt gives the corresponding contrast on AXTree text (DOM vs P-prompt), so the 2×2 is identified at both rows rather than only at one.

(ii) **The "mismatched ID parsing failure" is an empirical question, not a design exclusion.** Modern multimodal LLMs handle prompt-vs-text disagreement by their own parser fallbacks — Section 4 reports whether P-prompt accuracy collapses (consistent with mismatched-ID failure) or partially holds (consistent with the model adapting prompt instructions to the AXTree ID space). Either outcome is informative for the phantom-space framing. Pre-excluding P-prompt would force this question into a no-evidence appendix.

(iii) **P-prompt anchors the H3 axis-2 structural claim.** The pre-registered H3(ii) hypothesis (`preregistration.md` §2) tests whether |P-prompt ∖ P-SoM| unique-task contribution is non-zero — i.e., whether the SoM-style prompt elicits behavior P-SoM does not when paired with hierarchical text. Without P-prompt cell data, H3 axis-2 cannot be tested, and the paper's structural claim collapses to a single-axis (R2) framing.

We acknowledge that under naive evaluation P-prompt may show degraded `[SOM_MARKS]`-style action parsing. Our pre-registered analysis treats this as a feature, not a confound: the cell tests whether prompt instructions can elicit task behavior on a representation the prompt does not literally match. Section 4 reports P-prompt's SR alongside its action-parse-success rate as transparency.

### 3.4.2 Contrast Logic

Each contrast isolates a different factor:

- **DOM vs P-text** holds the prompt family fixed at DOM and changes the text-payload structure from AXTree to `[SOM_MARKS]`.
- **DOM vs P-prompt** holds the text observation fixed at AXTree and changes only the prompt family.
- **P-text vs Phantom-SoM** holds the text observation fixed at `[SOM_MARKS]` and changes only the prompt family. Together with DOM vs P-prompt, this gives two independent measurements of the prompt-format axis.
- **DOM vs Phantom-SoM** changes both axes — text-payload structure AND prompt — measuring the combined within-phantom diagonal.
- **Full SoM vs Phantom-SoM** holds prompt and marks text fixed and adds the implemented marked-image channel.
- **Full SoM vs DOM** measures the combined effect of SoM prompt, marks text, and marked screenshot relative to the standard text baseline.

The 2×2 is not a routing policy by itself. It is a causal scaffold for Section 5: text-payload structure shapes exploration, while prompt wording tunes commitment confidence. Section 6 promotes this scaffold to a token-monotonic cascade — DOM → P-text (axis 1, text-payload swap, no token increase) → Phantom-SoM (axis 2, system-prompt swap, no data-token increase) → full SoM (axis 3, image embedding cost). P-prompt sits off this monotonic path as the diagonal cell that completes the prompt-format identification.

### 3.5 Implementation and Measurement Protocol

All SoM-derived conditions share the same text-marks extractor. `_extract_text_marks` reads `obs_text` (the VisualWebArena accessibility-tree serialization the DOM baseline already uses) line by line, keeps each line whose label matches `\[\d+\]`, and produces `(id, label)` pairs up to a configured cap. `_build_som_result` then wraps those pairs in a `[SOM_MARKS] ... [/SOM_MARKS]` block. This text path **does not require bounding boxes**: the IDs come from the accessibility tree, not from a separate vision pipeline. Bounding boxes are only consulted by full SoM, which uses `obs_nodes_info` to draw numeric labels onto the page screenshot. Phantom-SoM and P-text reuse the exact `[SOM_MARKS]` text and drop the page screenshot; Marks are not re-filtered specifically for Phantom, and the source page state is unchanged.

Reference images supplied by a task configuration are separate from the observation mode. These task-provided target images are passed to all modes as task input; Phantom-SoM removes only the current-page browser screenshot.

Each episode starts from `environment.reset(task.config_file)`, and paper-grade condition comparisons use freshly reset site state to avoid cross-condition contamination. The April 27 Magento base-url/auth fix addressed an unrelated shopping-state reliability issue; this paper uses completed classifieds and reddit runs under the reset protocol.

When comparing arms, we use same-task subsets: a task contributes only when the relevant conditions have completed it. The reported outcome is the canonical **success** field on each episode summary, with no post-hoc adjustment. Two upstream interventions ensure this raw number is the paper-grade outcome rather than a contamination of false positives: (i) tasks with `eval.fuzzy_match == "N/A"` are excluded at the task-load layer (`tasks.py::_is_na_task`, default `task.exclude_na_tasks: true`), so not-applicable tasks never enter the comparison universe; (ii) the VisualWebArena LLM-judge helper functions (`llm_fuzzy_match` and `llm_ua_match`) are patched at the evaluator level to return `0.0` deterministically when the predicted answer is empty or whitespace-only (VWA submodule branch `p79-patches` commit `f0c835b`), which closes the failure mode where the runner's fake-stop with `answer=""` was being credited as a correct answer by GPT-4o-mini. The combined effect is that `success` is now the canonical outcome with no per-condition false-positive subtraction layer; the historical `compute_adjusted_success` post-hoc layer and its `na_fp`/`eval_fp`/`adjusted_success` derived fields have been retired (see `analysis.py` and `EpisodeSummaryV2`). Section 4 reports results under these conventions; Section 5 uses the same traces for mechanism analysis.

### 3.5.1 Cross-baseline implementation asymmetries (disclosure)

Three baseline-level implementation differences are not eliminated by the design above and we disclose them here rather than treat them as bugs.

**Effective scroll action vocabulary.** Although all three baselines see the eight-action schema described above, the *effective* scroll action space differs between B0 (proxy) and B1/B2 (local). B0's system prompt emits `scroll_direction ∈ {"up", "down"}`, and the agent post-processes the response into a fixed magnitude `delta = [0, ±0.8]` (`proxy_api_agent.py::step`, the `scroll_direction → delta` translation block). B1 and B2 instead receive a `delta = [dx, dy]` schema and the model selects the magnitude per step. Cross-baseline SR differences on scroll-heavy tasks (notably reddit search-loop episodes) therefore mix a capability gap with an action-vocabulary asymmetry; we do not claim byte-equivalence at the action-space level and Section 4 does not interpret scroll-driven SR deltas as pure capability effects.

**Per-mode input-token decomposition.** B1 and B2 record `input_image_tokens` and `input_text_tokens` separately from the local processor (`qwen3vl_agent.py:521`, `gemma3vl_agent.py:226-229`), so the cost saved by removing the page screenshot from a marks-only mode is directly measurable. The B0 proxy returns only an aggregate `usage.input_tokens` (no per-modality decomposition; `proxy_api_agent.py::meta`), and the runner therefore writes `input_image_tokens = 0` for B0 in every step record. Absolute B0 cost numbers per mode are not comparable to B1/B2 absolute numbers at the per-image-token granularity. The "Phantom-SoM cost ≈ DOM" property is therefore measured *within* each baseline as the cross-mode delta `Δusage(mode_with_image) − Δusage(mode_without_image)` rather than as an absolute per-image cost across baselines.

**Parse-error recovery scaffold (provisional).** B0 currently routes parse failures through an auxiliary GLM-5.1 extraction call (`proxy_api_agent.py::_call_glm_extract`, enabled by `use_glm_fallback: true` in B0 configs), which converts a malformed B0 response into a valid action. The architectural cause is that the proxy API does not forward `tool_choice` to the underlying Bedrock model, so B0 cannot be forced into structured JSON output. B1 and B2 (local Hugging Face models) have no such fallback path. We record `glm_fallback_used` and `glm_original_fail_reason` on every step so the rescue rate is auditable, and we are evaluating whether to retire the GLM rescue and report B0 with pure parse failures instead. *(Specific paper text for this paragraph pending advisor decision; the rescue rate measured on the current B0 archive is ≈ 1.49 % of steps. Final wording will reflect whichever option lands: switching to a clean structured API path, or keeping the GLM rescue with disclosure and downstream filter.)*
