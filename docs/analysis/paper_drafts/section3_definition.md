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

Phantom-SoM is also a cost intervention, but the savings come from a different place than one might assume. We decompose the SoM bundle into three layers and ask which a Phantom-SoM deployment retains:

1. **Text observation**. The `[SOM_MARKS]` block is structurally different from a serialized AXTree (flat indexed list versus nested hierarchy with url/tab metadata) but has comparable token length: holding the system prompt fixed at the DOM family, median total input is 3437 tokens for DOM versus 3661 for Phantom-DOM on reddit, and 3008 versus 2948 on classifieds — within ±7% on both sites. Layer 1 is therefore **not a meaningful cost saving** on its own; what it does provide is a different token geometry, which we treat as a representation property and study mechanistically in Section 5.

2. **On-server annotation pipeline**, which has two steps: (2a) extracting element bounding boxes from the accessibility tree to produce `[SOM_MARKS]`, and (2b) drawing numeric labels onto the page screenshot to produce the marked image. Step (2a) is required for Phantom-SoM because it generates the text observation. Step (2b) is **specific to full SoM**: a Phantom-SoM deployment skips it. In our measurements full SoM and Phantom-SoM both report ~30 ms obs-prepare median latency because the research code keeps the marked screenshot on disk for debugging (`p79/experiment/som.py::prepare_observation_for_mode` retains the rendered path), but a production deployment can omit step (2b) and recover roughly 30 ms and on the order of $2e-5 per step. This contribution is small relative to inference cost, but it is a real saving and not an artifact.

3. **Image tokens at inference**. This is the dominant cost difference. Phantom-SoM does not send the marked screenshot to the model, removing the visual encoding stage entirely. Comparing same-prompt conditions, full SoM exceeds Phantom-DOM by roughly 600 tokens per step on reddit and 1100 on classifieds, and we attribute that gap to image tokens under our backend tokenization. These tokens otherwise drive prompt-processing time, memory pressure, and time-to-first-token in multimodal serving (see Section 2.4).

A Phantom-SoM deployment therefore saves layer (2b) and layer (3) relative to full SoM, while layer (1) is roughly unchanged from the DOM baseline in token count but different in structure. Section 4 tests whether this image-free condition also creates independent routing value, and Section 5 examines what the layer-(1) structural difference does to model behavior.

### 3.3 Phantom-DOM

**Phantom-DOM** is the disambiguation ablation:

```text
Phantom-DOM(page) =
  prompt = DOM prompt
  text   = SOM_MARKS(page)
  image  = None
```

Its observation is identical to Phantom-SoM: `[SOM_MARKS]` text only, no page screenshot. The only intended change is the system prompt. In both B0 (`p79/agents/proxy_api_agent.py`) and B1 (`p79/agents/qwen3vl_agent.py`), `_system_prompts["phantom_som"]` maps to the SoM prompt, while `_system_prompts["phantom_dom"]` maps to the DOM prompt. For `som`, `phantom_som`, and `phantom_dom`, the agent passes through the `[SOM_MARKS]...[/SOM_MARKS]` text directly.

This cell separates representation from prompt wording. If Phantom-DOM behaves like Phantom-SoM, the flat marks text is driving behavior. If it behaves like DOM, the prompt is doing more of the work.

### 3.4 The 2x2 Ablation Matrix

The core ablation is a prompt-by-representation matrix:

| | DOM prompt | SoM prompt |
|---|---|---|
| AXTree obs | DOM | unused in Paper 1 |
| `[SOM_MARKS]` obs | Phantom-DOM | Phantom-SoM |

Full SoM is adjacent to this 2x2: it uses the SoM prompt, the same `[SOM_MARKS]` text, and the marked screenshot. Vision is a separate screenshot-only baseline.

Each contrast isolates a different factor:

- **DOM vs Phantom-DOM** holds the prompt family fixed at DOM and changes the text representation from AXTree to `[SOM_MARKS]`.
- **Phantom-SoM vs Phantom-DOM** holds the text observation fixed and changes only the prompt family.
- **Full SoM vs Phantom-SoM** holds prompt and marks text fixed and adds the implemented marked-image channel.
- **Full SoM vs DOM** measures the combined effect of SoM prompt, marks text, and marked screenshot relative to the standard text baseline.

The 2x2 is not a routing policy by itself. It is a causal scaffold for Section 5: text representation shapes exploration, while prompt wording tunes commitment confidence.

### 3.5 Implementation and Measurement Protocol

All SoM-derived conditions use the same mark-generation pipeline. `_extract_text_marks` reads numbered element lines from the VisualWebArena accessibility text, up to the configured cap, and `_build_som_result` emits the `[SOM_MARKS]` block. When `obs_nodes_info` provides bounding boxes, full SoM draws numeric labels on the screenshot. Phantom-SoM and Phantom-DOM reuse this exact text and drop only the page screenshot. Marks are not re-filtered specifically for Phantom; the source page state is unchanged.

Reference images supplied by a task configuration are separate from the observation mode. These task-provided target images are passed to all modes as task input; Phantom-SoM removes only the current-page browser screenshot.

Each episode starts from `environment.reset(task.config_file)`, and paper-grade condition comparisons use freshly reset site state to avoid cross-condition contamination. The April 27 Magento base-url/auth fix addressed an unrelated shopping-state reliability issue; this paper uses completed classifieds and reddit runs under the reset protocol.

When comparing arms, we use same-task subsets: a task contributes only when the relevant conditions have completed it. We report **adjusted SR**, which starts from raw evaluator success and removes `na_fp` for not-applicable tasks that appear correct without agent-initiated finish, and `eval_fp` for evaluator matches caused by ineffective or non-finished trajectories. Section 4 reports results under these conventions; Section 5 uses the same traces for mechanism analysis.
