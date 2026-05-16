# Magento Custom-Option Radio: form_value_changed 漏检 Audit

**Date**: 2026-04-29
**Scope**: B0/B1 shopping (VWA), all observation modes
**Triggered by**: `/diag B0 vwa:shopping_task_0 DOM` — discovered click on Style 3/4 radios flips AXTree `checked` but runner reports `action_success=False, page_changed=False`, triggering false `action_failed` cascade and cycle early stop.
**Status**: Root cause identified with HTML evidence; concrete fix recommended; not yet applied.

---

## 1. Hypothesis verification (which one was right)

**Original hypotheses:**
1. ARIA shim: swatch is `<div role="radio">` not real `<input type="radio">`, so `_FORM_SNAPSHOT_JS` never sees it.
2. Async select update: hidden `<select>` updates after the snapshot is taken.

**Both wrong.** Real cause: **dict-key collision in `_form_fields_changed`** for same-name radio button groups.

### HTML evidence (live probe)

Probe URL: `http://100.95.81.103:7770/anime-throw-blanket-flannel-fleece-blanket-super-soft-cozy-warm-for-bedding-couch-sofa-plush-blanket-for-kids-adults-gift-50-x40.html`

This product is **not** a Magento configurable product (no `<select>`, no `super_attribute[]`). It's a product with **custom options** (`name="options[10527]"`). Markup excerpt:

```html
<input type="radio" class="radio admin__control-radio required product-custom-option"
       name="options[10527]" id="options_10527_2" value="64440">  <!-- Style 3 -->
<input type="radio" class="radio admin__control-radio required product-custom-option"
       name="options[10527]" id="options_10527_3" value="64441">  <!-- Style 4 -->
<input type="radio" class="radio admin__control-radio required product-custom-option"
       name="options[10527]" id="options_10527_4" value="64442">  <!-- Style 5 -->
```

So `_FORM_SNAPSHOT_JS`'s `document.querySelectorAll('input, textarea, select')` **does** see all 3 radios. Each entry produced is:

```js
{ tag: 'input', type: 'radio', name: 'options[10527]', value: '64440'/'64441'/'64442',
  checked: false/true/false, idx: 0 }
```

### Why idx is 0 for all three

Each radio sits in its own wrapper div (`<div class="field choice admin__field admin__field-option">`), as the only child input. So `Array.from(parent.children).indexOf(el) === 0` for **all three** Style radios. Same for Size radios.

Verified via BeautifulSoup-equivalent traversal:

| input | name | value | parent class | idx_in_parent |
|---|---|---|---|---|
| Style 3 | options[10527] | 64440 | field choice admin__field admin__field-option | **0** |
| Style 4 | options[10527] | 64441 | field choice admin__field admin__field-option | **0** |
| Style 5 | options[10527] | 64442 | field choice admin__field admin__field-option | **0** |
| 50"x40" | options[10528] | 64443 | field choice admin__field admin__field-option | **0** |
| 60"x50" | options[10528] | 64444 | field choice admin__field admin__field-option | **0** |
| 80"x60" | options[10528] | 64445 | field choice admin__field admin__field-option | **0** |

### The collision

`p79/experiment/state_change.py:_form_fields_changed` builds `before_map`/`after_map` keyed by `(tag, type, name, idx)`:

```python
def _key(f):
    return (str(f.get("tag","")), str(f.get("type","")),
            str(f.get("name","")), int(f.get("idx", 0)))
before_map = {_key(f): f for f in before_fields}
```

For 3 same-name radios, all three produce the **same 4-tuple key** `('input', 'radio', 'options[10527]', 0)`. Python dict comprehension keeps only the **last** value written → `before_map` retains **only Style 5**. Same for `after_map`.

Comparison loop iterates `before_map.items()` and compares to `after_map[k]`:

| scenario | before_map[K] | after_map[K] | result |
|---|---|---|---|
| Click Style 3 (value=64440) | Style 5 unchecked | Style 5 unchecked | **NOT changed** ✗ |
| Click Style 4 (value=64441) | Style 5 unchecked | Style 5 unchecked | **NOT changed** ✗ |
| Click Style 5 (value=64442) | Style 5 unchecked | Style 5 **checked: true** | changed ✓ |

Agent in `task 0` clicked Style 3 (step 7) and Style 4 (step 9) — both **silent** to the change detector. Step 10 DOM ground truth confirms the clicks succeeded (`[29594] checked: true → false; [29598] checked: true`), but `page_change_reasons=[]` for all of them.

This is **independent of** the Magento configurable / custom option distinction; any same-name radio group where each radio is the sole child of its wrapper hits this bug.

---

## 2. Repro confirming the bug

Code path: `runner/main.py:901-958` → `snapshot_form_fields()` → `_FORM_SNAPSHOT_JS` → `build_page_state()` → `detect_page_state_change()` → `_form_fields_changed()`.

End-to-end trace from `B0_dom_shopping_20260428` task 0:

| step | action | runner reported | next-step DOM ground truth | dict has |
|---|---|---|---|---|
| 7 | click 29594 (Style 3, value=64440) | `action_success=False, page_change_reasons=[]` | `[29594] checked: true` | only Style 5 (unchanged) |
| 8 | click 29594 (repeat) | `action_success=False, page_change_reasons=[]` | unchanged | only Style 5 (unchanged) |
| 9 | click 29598 (Style 4, value=64441) | `action_success=False, page_change_reasons=[]` | `[29598] checked: true; [29594] false` | only Style 5 (unchanged) |
| 10 | click 29612 (size 50"x40", value=64443) | `action_success=False, page_change_reasons=[]` | (episode terminated by cycle) | only "80"x60"" (unchanged) |

`trigger_distribution`: `action_failed=3, page_unchanged_streak=2, no_progress_streak=2` → cycle/no-progress condition triggered, episode killed at step 10 (`agent_finished=False`).

---

## 3. Recommended fix

**Minimal-risk fix**: include `value` in the dict key for radio/checkbox elements. Real `<input>`/`<select>` form changes still detected via the value/checked/selectedIndex comparison loop; same-name radio groups become individually addressable.

### Diff against `p79/experiment/state_change.py`

```python
# BEFORE (lines 98-104)
def _key(f: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return (
        str(f.get("tag", "")),
        str(f.get("type", "")),
        str(f.get("name", "")),
        int(f.get("idx", 0)),
    )

# AFTER
def _key(f: Dict[str, Any]) -> Tuple[str, str, str, str, int]:
    # For radio/checkbox, include value to distinguish same-name group members
    # (each radio in a same-name group typically sits as the sole child of
    # its wrapper, so idx_in_parent=0 collides — see swatch_form_change_audit.md)
    ftype = str(f.get("type", ""))
    discriminator = str(f.get("value", "")) if ftype in ("radio", "checkbox") else ""
    return (
        str(f.get("tag", "")),
        ftype,
        str(f.get("name", "")),
        discriminator,
        int(f.get("idx", 0)),
    )
```

### Why not the alternatives

- **Switch idx to use document-order** (e.g. via `data-fingerprint` injected by JS): more code, same outcome.
- **Aggregate radios by name and compare "which value is checked"**: more semantic but bigger refactor; current 4-tuple comparison still serves text inputs and selects fine.
- **Always include id**: ids may be missing on `<input type="text">` without `name`; not all Magento custom options have stable ids.
- **Async wait before snapshot** (the original hypothesis 2): not the bug. Snapshot is timed correctly; the dict construction loses information.

### Side effects to check

- **`type="checkbox"` groups** — Magento product compare and review-form ratings (5 same-name radios, `name="ratings[4]"`). Same fix applies, no regression.
- **Generic `<input>`** with no name — already handled via `name: el.name || el.id || ''` JS fallback. No change.
- **Tests** — `tests/` covers `state_change` indirectly through smoke tests. After the fix, add a unit test in `tests/test_state_change.py` (create one if missing) with a synthetic snapshot of two radios sharing `name`/`idx` but different `value`/`checked`.

### Apply the fix?

Recommendation: **apply**, but treat as a paper-relevant scaffold change.

- Affects accuracy of `action_success`, `page_changed`, `page_change_reasons`, and downstream `wasted_cost`, `no_op_rate`, router `trigger_distribution`. These signals appear in shopping baseline tables.
- Does **not** affect agent's observation (AXTree was already correct). Affects only runner-side bookkeeping and cycle-detection thresholds.
- For paper-grade clean re-run: shopping B0/B1 DOM/SoM/Vision should be re-run after the fix to get accurate `action_failed` counts. Vision mode is unaffected by the swatch issue (no DOM-targeted clicks), but DOM/SoM definitely affected.

---

## 4. Blast radius (B0_dom_shopping_20260428)

Scanned all 465 task episodes for swatch-loop signature: ≥2 consecutive `action_type=click` steps with `action_success=False, page_changed=False, page_change_reasons=[]` on the same product page (`*.html`) with at least one tiny element_bbox (≤30×30 px).

**11 episodes match (2.4% of run), all `success=False`.**

| task | streak_len | step_idxs | element_ids | agent_finished | total_steps |
|---:|:-:|---|---|:-:|:-:|
| 0 | 4 | 7-10 | 29594, 29594, 29598, 29612 | ✗ | 11 |
| 46 | 4 | 6-9 | 12541, 12552, 12541, 12552 | ✗ | 10 |
| 245 | 2 | 4-5 | 9054, 9068 | ✓ | 10 |
| 262 | 2 | 5-6 | 10233, 10233 | ✗ | 7 |
| 281 | 3 | 2-4 | 8796 ×3 | ✗ | 5 |
| 288 | 4 | 1-4 | 5838, 5827, 5827, 5838 | ✗ | 5 |
| 329 | 3 | 3-5 | 9370 ×3 | ✗ | 6 |
| 349 | 2 | 3-4 | 5168 ×2 | ✗ | 9 |
| 383 | 2 | 4-5 | 9381, 9487 | ✓ | 16 |
| 415 | 2 | 5-6 | 7051 ×2 | ✗ | 7 |
| 458 | 2 | 14-15 | 32961, 33014 | ✗ | 18 |

Full data: `logs/codex/swatch_blast_radius.json`.

**Direct impact**: 9/11 episodes terminated without reaching a `finish` action — cycle/no-progress early stop killed them mid-flow on a product page. Without this bug at minimum 9 episodes would have continued (whether they would have eventually succeeded depends on color-perception P6 and other factors, but they would have at least had a chance to add-to-cart).

**Indirect impact**: false `action_failed` inflates `trigger_distribution` (router escalation triggers, B1 phantom-SoM router signal AUROC). Across the 11 affected episodes the false-positive `action_failed` count is at least ~30 (sum of streak_len). For the run-level metric this is ~30 / total_steps (run_summary).

**Same bug for B1 / SoM mode**: the snapshot/diff layer is shared. SoM mode also looks at AXTree-equivalent observations and uses the same env. B1 4B does not change the failure mechanism. Therefore the fix is needed across all 6 (B0/B1 × dom/som/vision) shopping conditions, but Vision rarely targets these radios (it works in normalized coords on the screenshot rather than `element_id`), so the Vision blast radius will be smaller.

---

## 5. Decision

**Recommended action**:
1. Apply the `_key` fix in `state_change.py`.
2. Add a unit test for same-name radio groups.
3. Re-run B0 DOM shopping on the affected 11 task subset first as a sanity check (cheap: ~11 episodes × <1 min/ep with proxy API).
4. If sanity check shows the swatch-loop signature is gone and at least some of those 11 advance further, decide whether to re-run the full B0/B1 shopping suite for paper-grade clean numbers.

**Out of scope here**:
- Color perception remains a separate problem (P6) — Style 1-5 swatch labels don't expose color in DOM. This audit is about scaffold correctness, not task tractability.
- Agent's "focus interpreted as checked" hallucination (also surfaced in /diag) is a separate model-level issue and won't be fixed by this change. After the fix, agent will at least receive correct `action_failed=False` feedback when its swatch click works.

---

## Appendix: Files referenced

- `p79/experiment/state_change.py:93-125` — `_form_fields_changed`
- `p79/envs/vwa_wrapper.py:570-609` — `_FORM_SNAPSHOT_JS` / `snapshot_form_fields`
- `p79/experiment/runner/main.py:901-923` — snapshot before/after action
- `results/visualwebarena/phase1/B0_dom_shopping_20260428/phase1_dom_router_0/episodes/shopping_task_0_steps_v2.jsonl` — primary trace
- `logs/codex/swatch_blast_radius.json` — full blast-radius scan
