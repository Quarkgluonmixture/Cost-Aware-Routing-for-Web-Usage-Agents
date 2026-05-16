# Cluster 1 Design — Locator-route mouse dispatch (B-01/02/03/04/05/25/32/33)

**Date**: 2026-04-30
**Status**: design only, no code yet
**Resolves**: B-01 TYPE / B-02 §106 / B-03 CLEAR / B-04 HOVER / B-05 UPLOAD / B-25 role=link non-`<a>` / B-32 button-AJAX / **B-33 family** (a-f sub-modes)
**Estimated blast radius freed**: ~5.5% of all paper-grade ep (combined click bbox bugs from Tier 10 sweep)

---

## 1. Problem statement

Per Tier 10 dispatch-effective-target audit (`probe_tier10_dispatch_target.json`), framework's id-based action dispatch in `external/visualwebarena/browser_env/actions.py:1280-1430` exhibits **94.4% off-target rate** on failed click steps:

```python
# actions.py:1306-1310 (CLICK with element_id)
if action["element_id"]:
    element_id = action["element_id"]
    element_center = obseration_processor.get_element_center(element_id)
    execute_mouse_click(element_center[0], element_center[1], page)
```

`get_element_center` (processors.py:786) returns `(x + w/2, y + h/2) / viewport_size` — the **geometric center of `union_bound`**. This center hits child elements (span.date / svg / h2) instead of the actionable parent (`<a>` / `<button>` / `<input>`) in 5 sub-modes (B-33a-f). Same pattern affects TYPE, CLEAR, HOVER, UPLOAD.

## 2. Strategy: P79-side wrapper, NO upstream patch

The framework's `actions.py` is upstream code; patching it forks visualwebarena. **Better approach** (proven by §51 SELECT_OPTION JS workaround in `p79/envs/vwa_wrapper.py:350-435`):

- Intercept actions at P79 wrapper layer **before** they reach `execute_action()` framework dispatch
- For id-based actions, use `obs_nodes_info[eid]["union_bound"]` to find pixel center
- Use Playwright `page.evaluate_handle` + JS `elementFromPoint(x, y)` to get the actual DOM element at center
- **Walk up the DOM tree** to find the nearest actionable ancestor (`<a>` / `<button>` / `<input>` / `[role=link]` / `[role=button]`)
- Invoke action via Playwright locator on the resolved element handle

This bypasses framework dispatch entirely for id-based actions while preserving framework dispatch as fallback for `pw_code` / `coordinate` paths.

## 3. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│ Agent emits action_json:                                         │
│ {"action_type": "click", "element_id": 42, "thought": "..."}     │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│ P79 vwa_wrapper.step()  [NEW dispatch interception]              │
│                                                                  │
│ if action_type ∈ {click, type, clear, hover, upload}             │
│    and "element_id" in action_json:                              │
│       1. Look up obs_nodes_info[eid]["union_bound"]              │
│       2. compute pixel center (x_px, y_px)                       │
│       3. JS resolve: elementFromPoint(x_px, y_px)                │
│       4. JS walk-up: find actionable ancestor by action type:    │
│          - click/hover: <a>, <button>, [role=link/button]        │
│          - type/clear:  <input>, <textarea>, [contenteditable]   │
│          - upload:      <input type=file>                        │
│       5. Get Playwright JSHandle for actionable element          │
│       6. Dispatch via Playwright:                                │
│          - locator.click() / hover() / fill() / clear()          │
│          - locator.set_input_files() for upload                  │
│       7. Convert P79 dispatch result → framework's expected      │
│          (action_success bool, page_changed delta)               │
│                                                                  │
│ else: fall through to framework execute_action() (unchanged)     │
└──────────────────────────────┬───────────────────────────────────┘
                               ↓
┌──────────────────────────────────────────────────────────────────┐
│ Framework execute_action() — only reached for non-id paths        │
│ (pw_code, coordinate, scroll, key_press, navigation)             │
└──────────────────────────────────────────────────────────────────┘
```

## 4. Per-action JS resolution rules

### 4.1 CLICK / HOVER (B-02, B-25, B-31, B-32, B-33 family)

```javascript
function resolveClickTarget(x, y) {
  let el = document.elementFromPoint(x, y);
  if (!el) return null;

  // Walk up to find actionable ancestor (max 6 levels to bound complexity)
  for (let i = 0; i < 6 && el && el !== document.body; i++) {
    if (el.tagName === 'A' && el.href) return el;
    if (el.tagName === 'BUTTON') return el;
    const role = el.getAttribute('role');
    if (role === 'link' || role === 'button' || role === 'menuitem' || role === 'tab') return el;
    if (el.tagName === 'INPUT' && (el.type === 'submit' || el.type === 'button' || el.type === 'checkbox' || el.type === 'radio')) return el;
    // Element with onclick handler
    if (el.onclick !== null) return el;
    el = el.parentElement;
  }
  return null;  // No actionable ancestor found within walk-up budget
}
```

**Dispatch**: `await locator(actionable_handle).click()` (Playwright actionability check + real mouse event)

**Fallback**: if `resolveClickTarget` returns null (e.g., bbox center hit pure decoration), fall through to framework's `execute_mouse_click(center)` — preserves current behavior for unrecognized targets.

### 4.2 TYPE / CLEAR (B-01, B-03)

```javascript
function resolveInputTarget(x, y) {
  let el = document.elementFromPoint(x, y);
  if (!el) return null;

  for (let i = 0; i < 6 && el && el !== document.body; i++) {
    if (el.tagName === 'INPUT' && el.type !== 'hidden' && el.type !== 'submit' && el.type !== 'button') return el;
    if (el.tagName === 'TEXTAREA') return el;
    if (el.isContentEditable) return el;
    // Label associated with an input via for=""
    if (el.tagName === 'LABEL' && el.htmlFor) {
      const target = document.getElementById(el.htmlFor);
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return target;
    }
    el = el.parentElement;
  }
  return null;
}
```

**Dispatch for TYPE**: `await locator(input_handle).fill(text)` — auto-clears previous content + sets value + dispatches `input`/`change` events. **No `Meta+A` / `Backspace` / `keyboard.type` 三连击** — eliminates 全选变蓝 (§52/§64) entirely.

**Dispatch for CLEAR**: `await locator(input_handle).fill('')` or `locator.clear()`.

**Critical correctness**: framework's TYPE pattern is `mouse.click + Meta+A + Backspace + keyboard.type`. The Meta+A is what causes 全选变蓝 when click hits non-input. With `locator.fill()`, **no global key shortcut is sent at all** — even if our walk-up fails, the worst case is the framework fallback path (which is what runs today).

### 4.3 UPLOAD (B-05)

```javascript
function resolveUploadTarget(x, y) {
  let el = document.elementFromPoint(x, y);
  for (let i = 0; i < 6 && el && el !== document.body; i++) {
    if (el.tagName === 'INPUT' && el.type === 'file') return el;
    // Magento "browse" button often is <button> labeled "Choose File" near the file input
    if (el.tagName === 'BUTTON' || (el.tagName === 'A' && el.classList.contains('upload'))) {
      // Look for sibling/descendant file input
      const parent = el.parentElement;
      if (parent) {
        const fileInput = parent.querySelector('input[type=file]');
        if (fileInput) return fileInput;
      }
    }
    el = el.parentElement;
  }
  return null;
}
```

**Dispatch**: `await locator(file_input_handle).set_input_files(action.text)` (Playwright handles `expect_file_chooser` automatically with `set_input_files`).

### 4.4 SELECT_OPTION (B-06)

**Already implemented** in `vwa_wrapper.py:350-435`. **No change needed** in Cluster 1 — keep existing logic.

## 5. Implementation skeleton

New helper module `p79/envs/locator_dispatch.py`:

```python
"""Locator-route dispatch wrapper for B-01/02/03/04/05/25/32/33 (Cluster 1).

Replaces framework's mouse.click(union_bound_center) pattern with Playwright
locator-based dispatch on the resolved actionable DOM ancestor.
"""

# JS for finding actionable ancestor (different per action type)
_JS_RESOLVE_CLICK = """([x, y]) => { ... walk-up logic ... }"""
_JS_RESOLVE_INPUT = """([x, y]) => { ... walk-up logic ... }"""
_JS_RESOLVE_UPLOAD = """([x, y]) => { ... walk-up logic ... }"""

def dispatch_id_based_click(page, x_px, y_px, *, action_type: str) -> dict:
    """Returns {success: bool, target_tag: str, fallback_used: bool}."""
    js = _JS_RESOLVE_CLICK if action_type in ("click", "hover") else _JS_RESOLVE_INPUT
    handle = page.evaluate_handle(js, [x_px, y_px])
    if handle is None or handle.json_value() is None:
        return {"success": False, "fallback_used": True}
    locator = page.locator(":root").locator(handle)  # adapt JSHandle to Locator
    if action_type == "click":
        locator.click(timeout=5000)
    elif action_type == "hover":
        locator.hover(timeout=5000)
    elif action_type == "type":
        locator.fill(action_text, timeout=5000)
    elif action_type == "clear":
        locator.fill("", timeout=5000)
    elif action_type == "upload":
        locator.set_input_files(action_text, timeout=5000)
    return {"success": True, "fallback_used": False, "target_tag": ...}
```

Hook into `vwa_wrapper.py:step()` at the same point where SELECT_OPTION wrapping happens (line ~350). Add 5 more `elif action_type == ...` branches before the framework `super().step(action_dict)` fall-through.

## 6. Risk + mitigation

### R1: walk-up depth=6 misses deeper actionable ancestor
- **Mitigation**: 6 levels covers virtually all known patterns (listing card has anchor 2 levels up; submission header has anchor 1 level up; subscribe button span 1 level up). If pilot Cluster-1 wave shows residual scaffold, extend to 8.

### R2: `el.onclick !== null` heuristic is unreliable (most onclick is set via addEventListener)
- **Mitigation**: skip the `onclick` heuristic; rely only on tag + role. Real-world Magento/Postmill/Reddit use semantic tags + roles uniformly.

### R3: Playwright's `page.locator(":root").locator(handle)` adapter API may not exist
- **Mitigation**: alternate path — call `handle.click()`, `handle.fill()` etc. directly on the JSHandle. ElementHandle in Playwright has its own `.click()` / `.fill()` methods.

### R4: `locator.fill(text)` sends `input` event but some sites need `change` + `blur`
- **Mitigation**: after fill, call `locator.dispatch_event('change')` + `locator.blur()`. Already a known Playwright idiom.

### R5: locator-route bypass of framework's `execute_mouse_click` may break runner state tracking
- **Mitigation**: ensure dispatch wrapper updates `last_obs_nodes_info` / `state_digest` consistently. Pattern from SELECT_OPTION (§51) already proven.

### R6: Pilot Cluster-1 wave needs to compare BEFORE/AFTER on same task
- **Mitigation**: run Cluster-1 patch on same 30-task pilot subset (reddit + shopping, cls deferred). Compare:
  - Task-level success match: should improve on tasks where B-33 was active (listing card click, button-AJAX, type-on-non-input)
  - Aggregate SR delta: expect **+2 to +5pp improvement** if Tier 10 estimate (~5.5% blast) is right

## 7. Validation plan

After Cluster 1 patch lands:

1. **Unit-level**: pytest for resolve_*_target JS via Playwright fixture (5-10 cases per action type)
2. **Integration**: pilot Cluster-1 wave, 30 task × 2 site (reddit + shopping), compare to current pilot wave-2 baseline (T=0 paper-grade behavior)
3. **Tier 10 re-sweep**: re-run `probe_tier10_dispatch_target.py` after patch — expect ON_TARGET ≥ 80% (vs current 5.6%)
4. **Production indicator**: cycle-detect early-stop frequency should drop (no more "URL stuck after 5 clicks" because each click reaches actual target)

## 8. Out of scope for Cluster 1

- B-09 page_changed split → **Cluster 2** (separate ~50 LOC patch in state_change.py + runner)
- B-11 fuzzy cycle hash → **Cluster 3** (separate ~40 LOC patch in helpers.py)
- B-22 program_html selector brittleness → Section 4 limitation cite, no code fix in Phase A
- B-37 RNG seeding → **already shipped in Cluster 4**

## 9. Open questions for review

- [ ] **Locator vs ElementHandle API choice**: which Playwright dispatch API is most stable for use with `JSHandle` from `evaluate_handle`? Need 5-min spike to validate.
- [ ] **First-click-then-fill** for sites that lazy-load form on focus: do we need to click the input first to trigger any reveal logic, then fill?
- [ ] **Confirmation dialogs**: §53 `confirm()` auto-accept needs to be preserved — check that locator-route doesn't bypass dialog handler registration.
- [ ] **target=_blank popup wait** (B-07): should we use `expect_popup` context manager around Cluster-1 click? Or rely on existing post-action `pages.length` check?

These should be resolved during implementation prototyping (~30-60 min spike), not blocking design ratification.

---

## 10. LOC estimate

| File | Lines | Purpose |
|---|---:|---|
| `p79/envs/locator_dispatch.py` (new) | ~120 | Helper module: 4 JS strings + 1 dispatcher function |
| `p79/envs/vwa_wrapper.py` | +30 | Hook 5 new action-type branches before framework fall-through |
| `tests/test_locator_dispatch.py` (new) | ~80 | Unit tests with mocked Playwright fixture |
| **Total** | **~230** | within Phase A budget |

## 11. Decision log

- **2026-04-30 13:50 BST**: Cluster 4 (B-37) shipped, pilot wave-2 PASS Δ=0pp on N=58 ep → green-light Cluster 1 design.
- Design pattern adopted: P79-wrapper interception (per §51 SELECT_OPTION) instead of upstream `actions.py` patch.
- LOC budget ~230, well within Phase A overall budget (~175 originally projected; revised slightly upward for proper test coverage).
