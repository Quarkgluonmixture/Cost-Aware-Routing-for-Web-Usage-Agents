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

Phantom-SoM is also a cost intervention. The `[SOM_MARKS]` block is a flat indexed list rather than a full AXTree; in our notes it is roughly half the AXTree token length on average, and it avoids page image tokens entirely. Section 4 tests whether this cheaper condition also creates independent routing value.

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
