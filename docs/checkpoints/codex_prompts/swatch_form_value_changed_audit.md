# Magento Swatch click → page_change_reasons 漏检 audit

## Background

Diagnosing `B0_dom_shopping_20260428` task 0 (intent: "Buy the least expensive red blanket from Blankets & Throws") surfaced a probable scaffold bug: clicking a Magento configurable-product **swatch** (Style/Size radio) flips the AXTree `checked: false → checked: true`, but the runner reports `action_success=False, page_changed=False, page_change_reasons=[]`. Repeated `action_failed` then triggers cycle/no_progress early stop, so episodes are killed even when the swatch click physically succeeded.

This bug is hypothesized to affect **all shopping product pages with configurable options** (color/size/style) — both B0 and B1 — and is independent of P6 "visual task DOM impossible". The failure mode hides a successful click behind a misleading `action_failed` signal, which then causes the agent to repeat the same click 3-4 times until early stop terminates the episode.

## Trace evidence (already verified)

Run dir: `results/visualwebarena/phase1/B0_dom_shopping_20260428/phase1_dom_router_0/`
Episode: `episodes/shopping_task_0_steps_v2.jsonl`

| step | action | runner reported | next-step DOM ground truth |
|---|---|---|---|
| 7 | click 29594 (Style 3 radio) | `action_success=False, page_changed=False, page_change_reasons=[]` | step 8 DOM: `[29594] radio 'Style 3' checked: true` ✅ click did succeed |
| 9 | click 29598 (Style 4 radio) | `action_success=False, page_changed=False, page_change_reasons=[]` | step 10 DOM: `[29598] radio 'Style 4' checked: true; [29594] checked: false` ✅ click did succeed |

URL: `http://100.95.81.103:7770/anime-throw-blanket-flannel-fleece-blanket-super-soft-cozy-warm-for-bedding-couch-sofa-plush-blanket-for-kids-adults-gift-50-x40.html` (DGX-reachable shopping site).

## Code under audit

- `p79/experiment/state_change.py:_form_fields_changed` (lines 93-125) — should report `form_value_changed` when `bf.get("checked") != af.get("checked")` for matched fields
- `p79/envs/vwa_wrapper.py:_FORM_SNAPSHOT_JS` (lines 570-599) — JS that runs `document.querySelectorAll('input, textarea, select')` and records `entry.checked = el.checked` for `input[type=radio|checkbox]`, `selectedIndex` for `<select>`
- `p79/experiment/runner/main.py:901-923` — calls `snapshot_form_fields()` before/after `environment.step(action)` and feeds into `build_page_state`

## Hypothesis to verify

Magento configurable-product swatches use `<div role="radio">` ARIA shims rather than real `<input type="radio">`. The visible click target is a `<div class="swatch-option">`; the underlying option value is stored in a hidden `<select name="super_attribute[ATTR_ID]">` which is updated by JavaScript after the click. Two failure paths possible:

1. **No real input/select captured** — swatch is pure `<div role="radio" aria-checked="...">` with no `<select>` underneath, so `_FORM_SNAPSHOT_JS` never sees it. AXTree picks up `aria-checked` because browsergym walks the accessibility tree, not the DOM.
2. **Select exists but JS update is async** — clicking the swatch dispatches an async event that updates `select.selectedIndex` shortly after; `snapshot_form_fields()` runs immediately after `environment.step(action)` returns, possibly before the JS finishes.

## Tasks

### Task 1 — Probe the actual Magento swatch HTML structure

Use a Playwright session (or `requests` + cookies from `.auth/shopping_state.json` then HTML inspection) on the URL above. Verify:

- Is there an `<input type="radio">` for each Style/Size option, OR only `<div role="radio">`?
- Is there a hidden `<select name="super_attribute[10527]">` (color) and `super_attribute[10528]` (size)?
- After programmatically clicking a swatch (Playwright), does (a) the hidden `<select>.selectedIndex` change synchronously, (b) `aria-checked` flip on the `<div role="radio">`, (c) any real `<input type="radio">.checked` flip?

If using Playwright, run with `headless=True` and use the cookies from `.auth/shopping_state.json`. Site URL: `http://100.95.81.103:7770/`. Login is preserved via storage state.

### Task 2 — Reproduce the bug end-to-end

Write a minimal repro that runs the same `_FORM_SNAPSHOT_JS` against the page before+after a swatch click, and compare via `_form_fields_changed`. Confirm whether `form_value_changed` is or is not reported. If the snapshot misses the change, identify which of the two hypotheses applies.

### Task 3 — Recommend fix

Based on what Task 1/2 finds, recommend ONE concrete fix. Constraints:

- Must not break existing detection for real `<input>` / `<select>` form changes (currently working for search box, sort dropdown, etc.)
- Must not introduce false positives (e.g. detecting transient hover state as a change)
- Prefer extending `_FORM_SNAPSHOT_JS` to also capture `[role="radio"][aria-checked]` and `[role="checkbox"][aria-checked]` if hypothesis 1 is correct
- If hypothesis 2 (async), consider a short `page.wait_for_timeout(50)` between `step()` and `snapshot_form_fields()`, OR use Playwright's `page.wait_for_load_state('networkidle', timeout=200)`. Discuss tradeoff with overall step latency.
- Out of scope: redesigning the page_changed detector. Just fix the snapshot/diff layer.

### Task 4 — Estimate blast radius

Count, across the existing `B0_dom_shopping_20260428` run, how many episodes have `>=2 consecutive steps` with `action_success=False, page_changed=False, page_change_reasons=[], obs_url unchanged, action_type=click` followed by a re-click on a different `element_id`. This is the signature of "swatch click looped". Report the task IDs and how many likely lost their episodes to this bug.

## Output format

A single markdown report saved to `docs/analysis/cross_sites/swatch_form_change_audit.md` with sections:
1. Hypothesis verification (which one was right, with evidence — Playwright snippet output or HTML excerpt)
2. Repro confirming the bug
3. Recommended fix (concrete diff against `p79/envs/vwa_wrapper.py` or `p79/experiment/state_change.py`)
4. Blast radius (task ID list + count of affected episodes)
5. Decision: proceed with the fix? If yes, follow-up on whether to re-run shopping baseline post-fix (paper-grade implications — this changes action_success accounting)

## Constraints / not in scope

- Do NOT modify code in this round; deliver the audit + recommendation only. User will decide whether to apply the fix and re-run.
- Do NOT touch `_form_fields_changed` matching key — the `(tag, type, name, idx)` heuristic is fine for real form elements; problem is upstream in JS capture.
- The shopping site is shared (DGX accesses via Tailscale `100.95.81.103:7770`). Don't run destructive operations (add to cart, place order). Read-only HTML/JS probes only.
- Run on DGX Spark (`spark-9ea3`). Use `.venv/bin/python3` and ensure `PYTORCH_NVML_BASED_CUDA_CHECK=1` (though Playwright probe shouldn't need GPU).
