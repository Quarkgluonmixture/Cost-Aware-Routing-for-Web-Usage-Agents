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

Phantom-SoM uses the same SoM prompt family as full SoM and the same `[SOM_MARKS]` text, but removes the page screenshot passed to the model. In code, `p79/experiment/som.py::prepare_observation_for_mode` handles `mode in ("phantom_som", "phantom_dom")` by calling `_build_som_result(...)`, then returning the generated `som_text` with `marked_image=None`. The rendered screenshot path is retained for debugging; the model does not receive it.

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

Its observation is identical to Phantom-SoM: `[SOM_MARKS]` text only, no page screenshot. The only intended change is the system prompt. In both B0 (`p79/agents/proxy_api_agent.py`) and B1 (`p79/agents/qwen3vl_agent.py`), `_system_prompts["phantom_som"]` maps to the SoM prompt, while `_system_prompts["phantom_dom"]` maps to the DOM prompt. For `som`, `phantom_som`, and `phantom_dom`, the agent passes through the `[SOM_MARKS]...[/SOM_MARKS]` text directly.

This cell separates representation from prompt wording. If P-text behaves like Phantom-SoM, the flat marks text is driving behavior. If it behaves like DOM, the prompt is doing more of the work.

### 3.4 The 2x2 Ablation Matrix and Excluded Hybrid

The core ablation is a prompt-by-representation matrix:

| | DOM prompt | SoM prompt |
|---|---|---|
| AXTree obs | DOM | *excluded — see below* |
| `[SOM_MARKS]` obs | P-text | Phantom-SoM |

Full SoM is adjacent to this 2x2: it uses the SoM prompt, the same `[SOM_MARKS]` text, and the marked screenshot. Vision is a separate screenshot-only baseline.

The fourth cell — AXTree observation paired with the SoM prompt — is intentionally excluded from Paper 1 because it is not a self-consistent design point. The SoM system prompt instructs the agent to interact via `[SOM_MARKS]` IDs (e.g. `click [42]` referring to the SoM-marked element 42), but AXTree text uses an independent accessibility-tree ID space; an action like `click [42]` becomes parsing-ambiguous when the two ID systems do not match. This hybrid mode (i) has no clean LLM mechanism, (ii) confounds the prompt-effect ablation with mismatched-ID parsing failure, and (iii) does not reduce token cost relative to P-text. We treat the 5-mode set (DOM, P-text, Phantom-SoM, full SoM, plus Vision as a separate screenshot-only arm) as the diagonal axis-by-axis path through the 2×2×2 (text-payload-structure × prompt × image) design cube; the four mismatched-prompt-representation hybrids are excluded for the same reason.

Each contrast isolates a different factor:

- **DOM vs P-text** holds the prompt family fixed at DOM and changes the text-payload structure from AXTree to `[SOM_MARKS]`.
- **Phantom-SoM vs P-text** holds the text observation fixed and changes only the prompt family.
- **Full SoM vs Phantom-SoM** holds prompt and marks text fixed and adds the implemented marked-image channel.
- **Full SoM vs DOM** measures the combined effect of SoM prompt, marks text, and marked screenshot relative to the standard text baseline.

The 2x2 is not a routing policy by itself. It is a causal scaffold for Section 5: text-payload structure shapes exploration, while prompt wording tunes commitment confidence. Section 6 promotes this scaffold to a token-monotonic cascade — DOM → P-text (axis 1, text-payload swap, no token increase) → Phantom-SoM (axis 2, system-prompt swap, no data-token increase) → full SoM (axis 3, image embedding cost) — so a routing trigger never has to "add then remove" tokens.

### 3.5 Implementation and Measurement Protocol

All SoM-derived conditions share the same text-marks extractor. `_extract_text_marks` reads `obs_text` (the VisualWebArena accessibility-tree serialization the DOM baseline already uses) line by line, keeps each line whose label matches `\[\d+\]`, and produces `(id, label)` pairs up to a configured cap. `_build_som_result` then wraps those pairs in a `[SOM_MARKS] ... [/SOM_MARKS]` block. This text path **does not require bounding boxes**: the IDs come from the accessibility tree, not from a separate vision pipeline. Bounding boxes are only consulted by full SoM, which uses `obs_nodes_info` to draw numeric labels onto the page screenshot. Phantom-SoM and P-text reuse the exact `[SOM_MARKS]` text and drop the page screenshot; Marks are not re-filtered specifically for Phantom, and the source page state is unchanged.

Reference images supplied by a task configuration are separate from the observation mode. These task-provided target images are passed to all modes as task input; Phantom-SoM removes only the current-page browser screenshot.

Each episode starts from `environment.reset(task.config_file)`, and paper-grade condition comparisons use freshly reset site state to avoid cross-condition contamination. The April 27 Magento base-url/auth fix addressed an unrelated shopping-state reliability issue; this paper uses completed classifieds and reddit runs under the reset protocol.

When comparing arms, we use same-task subsets: a task contributes only when the relevant conditions have completed it. We report **adjusted SR**, which starts from raw evaluator success and removes `na_fp` for not-applicable tasks that appear correct without agent-initiated finish, and `eval_fp` for evaluator matches caused by ineffective or non-finished trajectories. Section 4 reports results under these conventions; Section 5 uses the same traces for mechanism analysis.
