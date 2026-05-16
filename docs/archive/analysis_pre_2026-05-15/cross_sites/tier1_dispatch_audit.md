# Tier 1 — WA/VWA Dispatch + AXTree Extraction Audit

Audit date: 2026-04-30. Scope inspected locally: `external/visualwebarena/browser_env`. The requested `external/webarena/browser_env` path is not present in this workspace, so WebArena same-fork claims below are limited to the VWA copy bundled in this repo. BrowserGym source is also not vendored locally; Section C uses best-effort online links to ServiceNow/BrowserGym main and labels those separately.

## Executive Summary

- 4 high-suspicion candidate bugs beyond the already confirmed §106 click-center bug: element-id `TYPE`, `CLEAR`, `HOVER`, and `UPLOAD` all inherit the same `element_id -> union_bound center -> low-level mouse` dispatch surface.
- 2 medium-suspicion dispatch concerns: `SELECT_OPTION` uses locator APIs but drops parsed option arguments, and tab activation after click/new-tab has weak popup/target-blank semantics.
- 5 structural AXTree concerns: IDs are CDP AX `nodeId` values rather than backend DOM IDs, bounds come from current `getBoundingClientRect()`, inclusion is broad and role/tag-agnostic, current-viewport pruning rewrites the tree, and metadata is rebuilt only after action execution.

Recommended priority: fix §106 first with bid/locator click; then extend the same element-id locator route to `TYPE`, `CLEAR`, `HOVER`, and `UPLOAD`; then repair Playwright `select_option` argument forwarding.

## Section A — ActionType Dispatch Matrix

| ActionType | VWA dispatch API | Ref | Suspicion | Reasoning |
|---|---|---:|---|---|
| `NONE` | local no-op | `actions.py:1280-1281` | INFO | No browser call. |
| `SCROLL` | `page.evaluate("(document.scrollingElement ...).scrollTop +=/-= window.innerHeight")` | `actions.py:889-899`, `1283-1285` | MEDIUM | Low-level page scroll, not element identity. Can miss scrollable nested containers/modals because it always targets document scrolling. BrowserGym exposes wheel-style `scroll(dx, dy)` and `scroll_at(...)` online, but this is a different design surface. |
| `KEY_PRESS` | `page.keyboard.press(key)` | `actions.py:917-921`, `1286-1288` | MEDIUM | Low-level keyboard dispatch to current focus. Correct for global shortcuts, brittle when expected target focus was lost by prior center-click failure. |
| `MOUSE_CLICK` | `page.mouse.click(x, y)` | `actions.py:954-962`, `1290-1291` | INFO | Explicit coordinate action; coordinate brittleness is expected by action-space design. |
| `KEYBOARD_TYPE` | `page.keyboard.type(text)` | `actions.py:1041-1044`, `1300-1301` | MEDIUM | Explicit keyboard action to current focus. No element identity, but silent if focus is wrong. |
| `MOUSE_HOVER` | `page.mouse.move(x, y)` | `actions.py:935-941`, `1298-1299` | INFO | Explicit coordinate hover; expected coordinate action. |
| `CLICK` | id path: `get_element_center()` then `page.mouse.click`; role path: `get_by_role(...).focus()` then focused locator click; pw path: `locator.click()` | `actions.py:1302-1321`; center at `processors.py:786-795` | HIGH | §106. The default id-based prompt path uses AX/SoM ID geometry instead of locator actionability. Probe already measured 27 episodes / 1.6% blast radius, DOM:SoM = 1.7x. |
| `TYPE` | id path: center click, `Meta+A`, `Backspace`, `page.keyboard.type(text)`; role path: locator focus then keyboard; pw path: locator click, keypress clear, `locator.type(text)` | `actions.py:1341-1367` | HIGH | Same geometry bug as §106, now compounded by destructive keyboard operations. If the center hits the wrong element or no input receives focus, `Meta+A`/`Backspace` can clear the wrong field or page state while `success=True`. BrowserGym online uses `fill(bid, value)` via `get_elem_by_bid(...).fill(...)`, not a center click. |
| `HOVER` | id path: `get_element_center()` then `page.mouse.move`; role path: focus; pw path: `locator.hover()` | `actions.py:1322-1339` | HIGH | Same center mismatch on menus/tooltips/dropdowns. For hover-only UI, a parent bbox center can fail silently to open the intended submenu. BrowserGym online uses `elem.hover(...)` for bid hover. |
| `PAGE_FOCUS` / `TAB_FOCUS` | `browser_ctx.pages[index]`; `page.bring_to_front()` | `actions.py:1373-1375`; parser `1810-1817` | INFO | Robust when index exists. No element identity. |
| `NEW_TAB` | `browser_ctx.new_page()` | `actions.py:1376-1377` | MEDIUM | Explicit new-tab action is fine, but popup/target-blank produced by `CLICK` only relies on post-action page-count polling at `1417-1421`; no `expect_popup`, load wait, or target-blank recovery. This aligns with the existing 12 ep / 13% `POPUP_OR_TARGET_BLANK` probe category. |
| `GO_BACK` | `page.go_back()` | `actions.py:1378-1379` | LOW | Native page navigation. |
| `GO_FORWARD` | `page.go_forward()` | `actions.py:1380-1381` | LOW | Native page navigation. |
| `GOTO_URL` / `GOTO` | `page.goto(action["url"])` | `actions.py:1382-1383`; parser `1798-1803` | LOW | Native navigation; no element identity. |
| `PAGE_CLOSE` | `page.close()`, switch/open fallback page | `actions.py:1384-1389` | INFO | Browser-level operation. |
| `CHECK` | `locator.check()` from Playwright code | `actions.py:1251-1256`, `1400-1408` | LOW | Locator-based and actionability-checked, but only available through Playwright-code actions, not id-based AX actions. |
| `SELECT_OPTION` | `locator.select_option()` from Playwright code | `actions.py:1227-1235`, `1391-1399` | MEDIUM | Locator-based, but dispatch calls `execute_playwright_select_option(locator_code, page)` without forwarding the parsed select arguments from `parsed_code[-1]`. `page.locator(...).select_option("x")` can become `locator.select_option()` and fail or clear selection. |
| `STOP` / `FINISH` | local answer only | `actions.py:519-522`, parser `1820-1826` | INFO | No browser call. |
| `CLEAR` | id path only in sync: center click, `Meta+A`, `Backspace` | `actions.py:1292-1297` | HIGH | Same geometry bug as `TYPE` with destructive side effects. The action creator accepts role/pw fields (`actions.py:676-693`) but sync dispatch ignores them. |
| `UPLOAD` | intended id path: center click inside `expect_file_chooser`; creation/parsing bug sets action type to `TYPE` | `actions.py:975-984`, `1409-1412`, creator `697-717`, parser `1743-1760` | HIGH | Uses center click for file chooser, so wrong bbox center times out or selects wrong element. Additionally `create_upload_action()` writes `ActionTypes.TYPE` at `actions.py:708`, making id-based upload parse into type dispatch rather than the `UPLOAD` branch. |

## Section B — AXTree Extraction Findings

### Q1 Role Inclusion Rules

VWA calls CDP `Accessibility.getFullAXTree` and walks every returned AX node after duplicate `nodeId` removal (`processors.py:409-421`). Inclusion is not a positive allowlist of `link/button/textbox`; it is broad. `parse_accessibility_tree()` emits any node whose string is non-empty, then filters only a small set of nameless nodes:

```python
role = node["role"]["value"]
name = node["name"]["value"]
...
if not name.strip():
    if not properties:
        if role in ["generic", "img", "list", "strong", "paragraph", "banner",
                    "navigation", "Section", "LabelText", "Legend", "listitem"]:
            valid_node = False
    elif role in ["listitem"]:
        valid_node = False
```

Ref: `external/visualwebarena/browser_env/processors.py:529-568`. Properties listed in `IGNORED_ACTREE_PROPERTIES` are suppressed: `focusable`, `editable`, `readonly`, `level`, `settable`, `multiline`, `invalid` (`constants.py:288-296`). Finding: named `link`, `button`, `heading`, `textbox`, `searchbox`, `radio`, `checkbox`, `option`, etc. are included if CDP exposes them. Paper relevance: HIGH, because the benchmark exposes semantic nodes without guaranteeing the exposed node has a locator-safe click target.

For `image_som`, the inclusion surface is a separate JS selector list: `a[href]`, `button`, inputs, `textarea`, `select`, `[contenteditable]`, `[role="button"]`, `[role="link"]`, checkboxes/menuitems/tabs/draggables, plus some site-specific selectors (`processors.py:813-818`). Paper relevance: HIGH, because SoM and AXTree have different ID systems and inclusion policies.

### Q2 Element ID Assignment

AXTree text observations use CDP AX `nodeId` as the visible `element_id`: `dfs(0, accessibility_tree[0]["nodeId"], 0)` and child recursion passes `child_node_id` as `obs_node_id` (`processors.py:581-597`). `obs_nodes_info[obs_node_id]` stores `backend_id` and `union_bound`, but the visible id is not `backendDOMNodeId` (`processors.py:572-575`). HTML observations use sequential DOM-tree cursor indices (`processors.py:366-386`). SoM uses a fresh drawing counter (`processors.py:946-964`, `1138-1139`).

Finding: IDs are regenerated from the current observation extraction, not a stable cross-step DOM identity. There is no backend-id lookup at dispatch time; dispatch only reads stale metadata from the previous observation and computes a coordinate. Paper relevance: HIGH, because cross-step ID comparisons and failure analysis must treat IDs as observation-local.

### Q3 `union_bound` Source

The bound source is a CDP `DOM.resolveNode` followed by `Runtime.callFunctionOn` that returns either a text-range rect for text nodes or `this.getBoundingClientRect().toJSON()` for elements (`processors.py:157-188`). AXTree extraction stores `[x, y, width, height]` directly from that response (`processors.py:430-446`). `get_element_center()` divides the rect center by viewport width/height (`processors.py:786-795`).

Finding: it is a current viewport `getBoundingClientRect()` wrapper, not a robust hit-test target. It does not handle multi-rect inline links via `getClientRects()`, does not choose an actionable descendant, and does not compensate for a CDP AX node whose backend DOM node is a semantic parent/card. Paper relevance: HIGH; this is the core §106 mechanism and applies to `TYPE/HOVER/CLEAR/UPLOAD` too.

### Q4 Stale-Cache Risk

`envs.step()` executes the action using `self.observation_handler.action_processor` from the previous observation, then rebuilds observations and metadata (`envs.py:282-295`). `TextObervationProcessor.process()` overwrites `self.obs_nodes_info` and `meta_data["obs_nodes_info"]` on each observation (`processors.py:751-764`). There is no persistent cache across completed steps, but there is an unavoidable stale window between observation N and action N: AJAX rerenders, lazy layout shifts, popups, or animation can invalidate the saved `union_bound` before dispatch.

Finding: no long-lived stale cache after step completion, but action-time geometry can be stale and remains silent if the coordinate still clicks some non-target surface. If `page.mouse.click` succeeds but page state does not change, `envs.step()` reports `success=True` (`envs.py:280-291`). Paper relevance: HIGH.

### Q5 `role="link"` Non-`<a>` Handling

The AXTree path has no tag/href validation; it trusts CDP role/name (`processors.py:529-575`). A `<button role="link">`, `<span role="link" onclick>`, or heading/card exposed by ARIA as `link` can be included if named and visible. The SoM selector explicitly includes `[role="link"]` (`processors.py:813-816`). Dispatch then clicks/moves the center coordinate, not a DOM locator.

Finding: non-anchor links are not inherently wrong; browsers can route click handlers on role-bearing elements. The failure mode is that the role-bearing accessible node can correspond to a parent/card bbox or nested layout region. Center clicking may hit a child image, whitespace, or overlay rather than the actual event listener. Paper relevance: HIGH for explaining listing-card and ARIA-card failures; MEDIUM for plain `<button role="link">` where the button box is compact.

## Section C — Cross-Fork Comparison (Best Effort)

Local BrowserGym source is not available in this repository. Recommendation: clone `ServiceNow/BrowserGym` for a pinned, reproducible comparison before claiming exact version-level diffs in the paper.

Online best effort against ServiceNow/BrowserGym main shows the relevant architectural difference:

- `get_elem_by_bid(page, bid, ...)` resolves a BrowserGym id through `get_by_test_id(...)`, including nested frame traversal, then returns a Playwright `Locator`: [utils.py#L6-L49](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/action/utils.py#L6-L49).
- `click(bid)` calls `elem.click(..., timeout=500)`: [functions.py#L145-L165](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/action/functions.py#L145-L165).
- `fill(bid, value)`, `hover(bid)`, `press(bid, key_comb)`, `clear(bid)`, `select_option(bid, options)`, and `upload_file(bid, file)` similarly route bid actions through locators: [functions.py#L59-L89](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/action/functions.py#L59-L89), [L191-L230](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/action/functions.py#L191-L230), [L246-L256](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/action/functions.py#L246-L256), [L126-L142](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/action/functions.py#L126-L142), [L615-L634](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/action/functions.py#L615-L634).
- BrowserGym still exposes explicit coordinate primitives such as `mouse_click(x, y)`, but they are separate from bid actions: [functions.py#L359-L373](https://github.com/ServiceNow/BrowserGym/blob/main/browsergym/core/src/browsergym/core/action/functions.py#L359-L373).

Diff summary: VWA id-based `CLICK/TYPE/HOVER/CLEAR/UPLOAD` use observation-time bounds and low-level mouse/keyboard; BrowserGym bid-based equivalents use current locators and Playwright actionability checks. Drop-in patch candidate: introduce an injected stable attribute for every exposed AX/SoM node and replace id-based center dispatch with `locator("[attr='id']").click/fill/hover/clear` where a unique element can be resolved. Non-drop-in risk: VWA currently exposes CDP AX `nodeId`, not BrowserGym bid, so injection must happen during observation extraction and handle AX nodes whose backend DOM node is text or a parent semantic node.

## Section D — Recommended Fix Priority

1. §106 `CLICK` center mismatch. Fix sketch: during AX extraction, assign a stable per-observation `data-vwa-bid`/`bid` to the backend DOM element, store it in `obs_nodes_info`, and make id-click use `page.locator(...).click(timeout=...)`. Fallback to coordinate only for explicit coordinate actions. Blast radius: measured 27 episodes / 1.6%, DOM:SoM = 1.7x.

2. Candidate §107 `TYPE` center-focus mismatch. Fix sketch: route id-based type to `locator.fill(text_without_trailing_enter)` or `locator.click(); locator.press(...)` when Enter is requested. Avoid global `Meta+A` and `Backspace` unless a locator is focused and actionability succeeded. Blast radius estimate: needs probe; likely broad on forms/search boxes and can poison subsequent steps.

3. Candidate §108 `CLEAR` destructive wrong-target mismatch. Fix sketch: honor id/role/pw dispatch uniformly and use `locator.clear(timeout=...)` for id-based inputs. If target is not clearable, return explicit failure. Blast radius estimate: needs probe; lower frequency than type, high severity when invoked.

4. Candidate §109 `HOVER` menu mismatch. Fix sketch: use `locator.hover(timeout=...)` for id-based hover and require explicit failure when hidden/covered. Blast radius estimate: needs probe on menu/dropdown tasks; likely site-clustered.

5. Candidate §110 `UPLOAD` + `SELECT_OPTION` form dispatch gaps. Fix sketch: change `create_upload_action()` to `ActionTypes.UPLOAD`, use `locator.click()` inside `expect_file_chooser`, and forward parsed `select_option` args/kwargs from `parsed_code[-1]` into `execute_playwright_select_option`. Blast radius estimate: upload rare; select-option likely under-measured because most public prompts use id-based click/type rather than Playwright select.

## Section E — Open Questions for Follow-Up

- Empirically probe `TYPE`: count cases where center click does not focus the target input or clears a neighboring field.
- Empirically probe nested scroll surfaces: compare document `scrollTop` dispatch against wheel/locator scroll inside modals, menus, and result panes.
- Probe `HOVER` on menu tasks: detect hover action with no AX/DOM/menu expansion.
- Probe `SELECT_OPTION` Playwright-code path with a controlled `<select>` to confirm current argument dropping behavior.
- Clone BrowserGym at a pinned commit and diff the exact current action/observation implementation against this VWA fork before writing final paper claims.
