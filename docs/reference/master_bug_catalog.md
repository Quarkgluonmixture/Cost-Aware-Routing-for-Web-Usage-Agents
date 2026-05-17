# Master Bug Catalog — 5-Tier Audit + Conversation Findings

**Purpose**: Single source of truth for **every** suspected scaffold/dispatch/observation/evaluator bug across the 5-tier audit system, the click-dispatch probe (§106), historical 实验笔记 entries, and conversation-level discoveries. Includes **disputed** and **rejected** entries — 有据可查，避免重复争论。

**Last updated**: 2026-04-30
**Maintainer**: Claude session (auto-update on new probe / tier verification)

---

## Status Legend

| Tag | Meaning |
|---|---|
| ✅ **CONFIRMED** | Replay-verified or root-cause traced; in fix scope |
| ⚠️ **DISPUTED** | Audit-claimed but probe evidence questionable (e.g. codex 没忠实 replay, blast radius 不可信) |
| ❌ **NOT_A_BUG** | Audit signature 是 by-design 行为 / 0% replay scaffold rate |
| 🔄 **UNVERIFIED** | Static audit only, no replay 验证 |
| 🛠️ **FIXED** | Already patched (with commit ref) |

---

## Origin Index

| Origin | What it audits | Methodology |
|---|---|---|
| **§52 / §64 / §105 / §106** | Historical 实验笔记 findings | Live debugging during experiments |
| **Tier 1** (`tier1_dispatch_audit`) | Static dispatch & AXTree code review | Read `external/visualwebarena/browser_env/{actions,processors,envs}.py` |
| **Tier 2** (`tier2_silent_failure_catalog`) | Signal-based silent failure mining | Scan 4493 ep / 46844 step over 12 paper-grade runs |
| **Tier 3** (Gemini Deep Research) | Lit-grounded 5-category taxonomy | Lit review only, no code/data |
| **Tier 4** (`tier4_invariant_audit`) | Invariant-violation mining | 10 logical invariants × 4501 ep |
| **Tier 5** (`tier5_evaluator_audit`) | Evaluator-side static audit | Read `evaluation_harness/{evaluators,helper_functions}.py` |
| **Click probe** (Apr 30 morning) | Re-classify §106 ep | Playwright replay 27 ep |
| **Audit verification probe** (`probe_audit_verification`) | Re-verify Tier 1/2/4 signatures | Playwright replay 50 cases (codex; **partial replay quality issue**) |
| **Conversation findings** | User pushback + self-replay | Interactive cross-check |

---

## Master Bug List

Sorted by **(severity × paper-impact)**. Each entry: claim → evidence → status → fix.

### B-01. TYPE silent failure (Meta+A / Ctrl+A 全选变蓝 + dispatch on non-input)

- **Origin**: §52 (2026-04-14, B0 TYPE 全选变蓝根因) + §64 (2026-04-15, P79 自写 Control+a 逻辑) + Tier 1 candidate_1 + Tier 2 `type_silent_failure` + probe verification (15/15 scaffold)
- **File**: `external/visualwebarena/browser_env/actions.py:1341` + `p79/envs/vwa_wrapper.py:240`
- **Mechanism**: id-based TYPE = `mouse.click(union_bound_center)` → `Meta+A` (or `Ctrl+A` on Linux) → `Backspace` → `keyboard.type(text)`. If center 命中非 editable 元素：focus 没换到 input → `Meta+A` 全选当前 page text → `Backspace` 删页面字 / type 进 wrong element → runner 看 dispatch return success=True 但 input.value 空。Page 出现"全屏蓝"高亮（§52 现象）。
- **Blast radius**: Tier 2 `type_silent_failure` 549 ep / 12.22% of failed traces. Probe scaffold fraction = **100%** (15/15 case). Site breakdown: cls 291, red 172, shop 86. Mode breakdown: DOM 224, P-text 76, P-SoM 79, SoM 78, Vision 80, P-prompt 12.
- **Self-correct caveat**: B0 (235B) 偶尔自纠正 retry → real SR impact < dispatch failure rate. B1 (4B) 几乎不自纠正 → blast radius mostly落 B1 SR。**asymmetric noise** between baselines.
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (locator-route via `p79/envs/locator_dispatch.py::dispatch_id_based_type` + `dispatch_id_based_click`); previously CONFIRMED + DOUBLE-VERIFIED via probe + self-verify
- **Verification trail**:
  - Codex `probe_audit_verification`: 15/15 SCAFFOLD (100%)
  - **Self-verify** `probe_b01_b13_self_verify.py` (2026-04-30, 12 cls cases with CORRECT bbox center `x+w/2,y+h/2`): 11/11 replay-ok cases SCAFFOLD_TYPE_BUG (1.0 strict scaffold fraction). 0 cases hit editable element at center. **Bbox bug didn't affect this category** because TYPE's bug is "center hits non-editable" — wrong bbox formula still yields wrong element. Caveat: cls-only sample; red/shop blast extrapolation assumes cls-like rate.
- **Fix**: route via Playwright `locator.fill()`（自带 actionability check + 自动清空 + dispatch input event）, ~30 行 in `p79/backends/action_utils.py`. §64 之前的 `activeElement.tagName` check 只 patch 了 Vision 路径，没 cover backend dispatch 层。
- **Paper impact**: Section 4 limitation table cite (asymmetric blast across baselines + DOM/SoM/Vision modes); Section 6 routing 章节考虑 mode-switch gate trigger by post-action `input.value` empty。

---

### B-02. §106 union_bound center mismatch (coordinate click dispatch)

- **Origin**: `docs/reference/106 VWA WebArena Coordinate-Based Click Dispatch Bug.md` + click probe (Apr 30 morning, 27 ep replay)
- **File**: `external/visualwebarena/browser_env/actions.py` (CLICK dispatch + processors.py:157 union_bound)
- **Mechanism**: id-based CLICK 用 `page.mouse.click(union_bound_center)`. union_bound 来自 step N 的 observation snapshot，但 dispatch 在 step N+1 issue 时元素位置可能 stale (AJAX / lazy render / layout shift). Click 落空 / 命中错元素 → page 没变 → agent 反复 click 同 element_id → click-loop。
- **Blast radius**:
  - Initial estimate (signature scan): 5.4%
  - Probe (Playwright replay 27 ep): **真 §106 = 1.6%** — 其余是 agent decision error / popup / AJAX
  - **REAL** 5-mechanism click-loop taxonomy from `logs/codex/click_root_cause_probe.json` (93 cases): **AXTREE_MAPPING_ERROR 55.9%** (B-33 NEW), UNION_BOUND_BUG / §106 proper **18.3%** (this entry), POPUP_OR_TARGET_BLANK 12.9% (B-07), BUTTON_OR_AJAX 7.5% (B-32), MISSING/transient 5.4%.
  - **PRIOR catalog (now retracted)**: 29% / 24% (searchbox-no-type) / 12% (heading-as-link) / 13% / 8% / 10% — these numbers were **fabricated from conversation memory**, not present in artifact. Retraction recorded in update log.
  - DOM:SoM hit-rate ratio ≈ **1.5×** on B0 reddit (asymmetric across modes)
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (locator-route in same patch as B-01); previously CONFIRMED with re-verified blast radius
- **Fix**: 同 B-01，route id-based click through injected locator + `locator.click()` actionability check. Tier 1 candidate_1/2/3/4 共享同一架构问题。
- **Paper impact**: Section 4 限制条款 cite (跨模式 1.5× 对称性 → 不能完全 cancel cross-mode SR comparison); Section 5 mechanism 用 `silent-failure-aware-but-not-driven` 论证 (ref Section 5 reddit prose §5.1.7).

---

### B-03. CLEAR dispatch (shares §106 architecture)

- **Origin**: Tier 1 candidate_2 + **static-verify** (2026-04-30, code reading)
- **File**: `external/visualwebarena/browser_env/actions.py:1292-1296`
- **Mechanism**: id-based CLEAR — code 字面 `execute_mouse_click(element_center[0], element_center[1], page)` then `execute_key_press("Meta+A", page)` then `execute_key_press('Backspace', page)`. **Same pattern as TYPE** — click union_bound center → Meta+A → Backspace. Creator accepts role/pw fields but sync dispatch ignores them.
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (shared locator-route); previously CONFIRMED via static read
- **Fix**: 用 `locator.fill('')` 替代（等价 `locator.clear()` in Playwright ≥ 1.34，源码实测为 `fill('')`，跟 prose 自纠正 — /stress A1.3 F5 backlog sweep 2026-05-15）；同 B-01 patch
- **Paper impact**: 跟 B-01 一起处理，不 separately framing

---

### B-04. HOVER dispatch (shares §106 architecture)

- **Origin**: Tier 1 candidate_3 + **static-verify** (2026-04-30, code reading)
- **File**: `external/visualwebarena/browser_env/actions.py:1322-1340`
- **Mechanism**: id-based HOVER — code 字面 `if action["element_id"]:` → `execute_mouse_hover(element_center[0], element_center[1], page)`. Same union_bound center as B-01/B-02/B-03.
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (shared locator-route); previously CONFIRMED static (low blast)
- **Fix**: route through `locator.hover(timeout=...)` with explicit failure on hidden/detached/covered. Same patch wave as B-01.

---

### B-05. UPLOAD dispatch + creator type mismatch

- **Origin**: Tier 1 candidate_4 + **static-verify** (2026-04-30, code reading)
- **File**: `external/visualwebarena/browser_env/actions.py:1409-1414` + `:708-720`
- **Mechanism**: 两个 bug 串联 confirmed:
  - **Creator bug** (line 715): `create_upload_action()` 字面 `"action_type": ActionTypes.TYPE` — should be `ActionTypes.UPLOAD`. Parsed id-based upload bypass UPLOAD branch entirely.
  - **Dispatch bug** (line 1413): `execute_upload(element_center[0], element_center[1], action["text"], page)` — center-based click before file_chooser, shares §106
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (shared locator-route + creator fix); previously CONFIRMED static
- **Fix**: set creator to `ActionTypes.UPLOAD`; use `locator.click()` inside `expect_file_chooser`. Low priority (VWA upload tasks 极少).

---

### B-06. SELECT_OPTION arg-drop

- **Origin**: Tier 1 candidate_5 + Tier 2 `select_option_silent_failure` + codex probe (8 case) + **self-replay probe** (2026-04-30, `probe_b08_b06_self_replay.json`, 20 cases site-diverse)
- **File**: `external/visualwebarena/browser_env/actions.py:1391-1395`
- **Mechanism**: SELECT_OPTION dispatch — line 1393 `parsed_code = parse_playwright_code(action["pw_code"])`, line 1394 `locator_code = parsed_code[:-1]` strips the last element (the `select_option(...)` call with args), line 1395 `execute_playwright_select_option(locator_code, page)` 用 default `pw_action_args=[]`/`pw_action_kwargs={}` → 等价 `locator.select_option()` 无参. Dropdown clears or no-op.
- **Blast radius**:
  - Tier 2 claim: 149 ep / 3.32%
  - Codex probe scaffold fraction: 0.286 (2/8 真 arg-drop) — **over-estimate due to codex bbox center bug + cls-only sample**
  - Self-replay (20 cases, 11 cls + 6 red + 3 shop, 修复 bbox 中心计算): scaffold fraction = **0.10** (2/20 native `<select>` arg-drop, 18/20 OTHER_CUSTOM_DROPDOWN — agent 试图 select_option 但 target 是 ARIA combobox / custom div, 走 click 路径不走 select_option dispatch)
  - **Real extrapolated blast radius: 15 ep** (149 × 0.10), much smaller than codex's 43 ep estimate
  - Site context: cls 117 (78%) / red 27 / shop 5 — paper 数据里 majority target 是 osclass custom location dropdown / reddit ARIA combobox, **native `<select>` 极少触发 SELECT_OPTION 路径**
- **Status**: ✅ **CONFIRMED** (real bug at line 1395) but ⚠️ **blast radius 1/3 of codex claim**
- **Existing P79 mitigation (§51, 2026-04-14)**: `p79/envs/vwa_wrapper.py` 已 implement JS workaround for native `<select>`: 从 `obs_nodes_info[str(eid)]["union_bound"]` 取像素中心 → JS `elementFromPoint(x,y)` 定位 SELECT → `el.value = opt.value; el.dispatchEvent(new Event('change'))`. 加 `_inject_select_options()` 在 AXTree combobox 行下注入 `[OPTIONS] "opt1", "opt2", ...`. **P79 已绕开 framework 的 arg-drop bug** — framework 路径只在 P79 wrapper 失败时 fallback 触发，进一步降低真实 blast radius.
- **Fix**: forward `parsed_code[-1].arguments` and `.keywords` into `execute_playwright_select_option()` — 一行代码 + regression test (upstream cleanup, 不影响 P79 实际行为). Low priority (15 ep / 4501 ≈ 0.3% of all ep).
- **Paper impact**: appendix only; not Phase A.
- **Methodology note**: self-replay 用 `bbox = [x,y,w,h]` 正确解析 (codex 用了错误的 `(x+w)/2,(y+h)/2`，命中 wrong elements 导致 over-classification 为 SCAFFOLD).

---

### B-07. NEW_TAB / target=_blank popup handling

- **Origin**: Tier 1 candidate_6 + click probe taxonomy (popup target_blank 13%) + **static-verify** (2026-04-30, code reading)
- **File**: `external/visualwebarena/browser_env/actions.py:1416-1422`
- **Mechanism**: 字面 `page.wait_for_timeout(int(sleep_after_execution * 1000))` then `num_tabs_now = len(browser_ctx.pages)`, only `if num_tabs_now > num_tabs_before: page = browser_ctx.pages[-1]; page.bring_to_front()`. **No `expect_popup` / no load wait** — fully confirmed.
- **Blast radius**: click-loop 13% mechanism share (from §106 click probe taxonomy)
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (locator + popup wrapping); previously CONFIRMED static
- **Fix**: wrap locator clicks with `expect_popup` fallback + bring popup to front after load.

---

### B-08. SCROLL silent failure (mostly legit at-bottom)

- **Origin**: Tier 2 `scroll_silent_failure` + codex probe (10 case) + **self-replay probe** (2026-04-30, `probe_b08_b06_self_replay.json`, 20 cases)
- **File**: framework `scroll` action path
- **Mechanism**: Tier 2 claim — `scroll` 后 viewport 不变 / obs_text similarity > 0.95. **Counterclaim**: 大多数是 page-already-at-bottom 合法 no-op.
- **Blast radius**:
  - Tier 2 claim: 667 ep / 14.85%
  - Codex probe scaffold fraction: 0.3 (3/10) — **over-estimate**
  - Self-replay (20 cases): scaffold fraction = **0.053** (1/19 replay-ok cases真 SCAFFOLD_SCROLL_BUG: cls task 5 step 14 — page room 568px ≈ expected 576px scroll，但 logged scroll_y 卡住; 18/19 LEGIT_AT_BOTTOM)
  - Codex over-estimate factor: **5.7×**
  - **Real extrapolated blast radius: 35 ep** (667 × 0.053), much smaller than codex's 200 ep estimate
- **Status**: ✅ **CONFIRMED** (real bug exists) but blast radius 1/6 of codex claim
- **Fix**: 改 `page.mouse.wheel()` → `page.evaluate("window.scrollBy(...)")` + 等待 scroll event resolve. Low priority (35 ep / 4501 ≈ 0.8% of all ep).
- **Paper impact**: appendix; ⚠️ **codex misled us — self-replay shows blast radius is much smaller, NOT a top-3 fix priority**
- **Methodology note**: self-replay 用 logged `state_digest.scroll_y_before/after` (避免重放前置步骤) + 抓 obs_url page geometry 判断是否真 at-bottom，不依赖 fresh-state replay scaling.

---

### B-09. State-change-but-obs-same (page_changed false trigger)

- **Origin**: Tier 4 I10 (288 violations) + probe verification (5 case) + **conversation root-cause trace** (`p79/experiment/state_change.py:detect_page_state_change`)
- **File**: `p79/experiment/state_change.py` (detect_page_state_change reasons emission) + `p79/experiment/runner/main.py:966` (`page_changed = bool(reasons)`)
- **Mechanism**: `page_changed = bool(12-signal union)`. 两个 reason 会让 `page_changed=True` 但 agent obs_text 不变：
  - **`form_value_changed`** — 编辑 form 字段时 form value 变了，但 AXTree dump 显示的是 type/role 不是 value → text 仍 identical
  - **`scroll_changed` ≥ 5px** — 视觉 scroll 但 viewport 文本同
  - **SequenceMatcher false trigger** — 长 corpus 上 0.95 阈值偶发 < 0.95 但视觉不可见
- **Blast radius**: Tier 4 claim 288 step. Probe scaffold fraction = **0.8** (3/5 LOGGER_BUG, 1/5 OBS_CACHE_BUG, 1/5 INVISIBLE_CHANGE).
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (`state_change.py` adds `agent_visible_changed` vs `runner_page_changed` split, paper-grade SR uses agent_visible_changed); previously CONFIRMED root-cause traced
- **Fix**: **结构性重分**, 不补 logger:
  - `runner_page_changed` (runner-internal, 12 reasons union) — 用于 cycle break / retry decision
  - `agent_visible_changed` (agent-facing, url/title/visible_text/scroll only) — 用于 fig0a SR / 早停 detect
  - Step JSONL 里的 `page_changed` 改成 `agent_visible_changed`
- **Paper impact**: 影响 fig0a SR 派生 + cycle-detect + search-loop 检测; fix 后建议**重跑 Tier 4 invariant audit**（I1/I8/I10 都 page_changed-derived）

---

### B-10. §105 Magento custom-option radio swatch 漏检 🛠️ FIXED

- **Origin**: §105 (2026-04-29) + `docs/analysis/cross_sites/swatch_form_change_audit.md`
- **File**: `p79/experiment/state_change.py:_key` (form_field key generation)
- **Mechanism**: `_form_fields_changed` 用 `(tag, type, name, idx)` 作 dict key. 同 name radio group 每个 radio 是独立 wrapper 的唯一 child → idx 全 0 → 同 name 的 radio 互相覆盖 → click 真生效但 `page_change_reasons=[]` → 误报 action_failed → cycle 早停。
- **Blast radius**: B0_dom_shopping_20260428 **11/465 ep (2.4%)** 全失败, 9/11 被 cycle 早停未 finish. DOM/SoM 共享 snapshot 层都受影响 (Vision 不受影响).
- **Status**: 🛠️ **FIXED** (`state_change.py:_key` 加 value discriminator)
- **Paper impact**: 触发了 14-cell rerun (整轮 dom shopping debug + paper-grade re-run B0/B1 DOM/SoM)

---

### B-11. 广义早停 — fuzzy cycle detect missing

- **Origin**: Conversation pushback (Apr 30, user 提点 #5) + Tier 4 I8 (max-step truncate at click 201 violations) + I3 (repeat click 481 violations)
- **File**: `p79/experiment/runner/helpers.py` (`_repeat_action_count` cycle detector)
- **Mechanism**: 当前 cycle detect 是**精确字符串匹配** — 同 action_type + 同 target_id + 同 args. Agent 在 search-loop 里**变 query**（"blue kayak" → "kayak blue" → "kayak boat"），cycle detect miss. 死循环 ≥ 30 步 hit max_step truncate 才停 → cost inflate (~72K input token wasted per loop).
- **Blast radius**:
  - I8 max-step truncate at click: 201 ep
  - I3 repeat click: 481 ep
  - Estimated trace impact: ~15-20% of failed traces (overlap with TYPE/SCROLL silent failure)
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (fuzzy cycle detection with `_action_signature_fuzzy` + min_reps=5); SUPERSEDED by B-38 (early-stop disabled per advisor 5/5 — detection now diagnostic-only). Previously CONFIRMED via user observation + Tier 4.
- **Fix**: fuzzy cycle hash on `(action_type, current_url_path_only_no_query, target_axtree_role)`; 同 hash 连续 ≥ 3 次直接早停。Independent patch from B-09.
- **Paper impact**: cost calculation 当前**系统性 inflate** failed traces' wasted-cost — fix 后 fig0e wasted_cost_usd 可能下降; 影响 4-fold drop-in property (a) 的 cost ≈ DOM claim

---

### B-12. AXTree element_id observation-local re-numbering ❌ NOT_A_BUG

- **Origin**: Tier 4 I9 (1127 violations) + probe verification (7 case, 0.833 codex-claimed scaffold) + **conversation pushback** (user #2, #3)
- **File**: `external/visualwebarena/browser_env/processors.py:581` (Tier 1 axtree_finding Q2)
- **Mechanism**: AXTree visible element_id = CDP Accessibility nodeId, **observation-local enumeration**, 不是稳定 backend identity. SoM marks 同样从 AXTree action-able 子集 enumerate — 页面 reflow / scroll / AJAX update 后枚举重排。
- **Why NOT_A_BUG**: agent 在 step N+1 issue `click(element_id=X)` 时引用的是 step N+1 的 AXTree dump，**不会跨 step reference**. "step N [7]=Search button → step N+1 [7]=Logout link" 是设计行为不是 stale-cache。Codex 的 STALE_NODEID_REUSE 分类是 misnaming.
- **Status**: ❌ **NOT_A_BUG** (BrowserGym public API contract: element_id is observation-local)
- **Paper impact**: paper limitation table acknowledge; 不进 fix scope. 但 **`current_viewport_only` pruning** (Tier 1 axtree_finding "current_viewport_only pruning") 提到 "nodes outside viewport or below 0.6 overlap removed; children spliced into parent" — 这会改变 ID 可用性 between scrolls，可能影响 §80 in_viewport_ratio 运算符优先级 bug (bug §80 一直没修，记录在 CLAUDE.md "DOM 仍有信息优势" 段)。

---

### B-13. action_fail-but-page-changed (runner false negative) ❌ NOT_A_BUG

- **Origin**: Tier 4 I2 (25 violations) + codex probe (5 case) + **self-verify** (`probe_b01_b13_self_verify.py`, 8 case state_digest analysis)
- **Verification trail**:
  - Codex: 0/5 (3 REPLAY_FAIL + 2 REPLAY_DID_NOT_CHANGE) — **weak evidence**, codex's REPLAY_FAIL might artifact replay infra issues not bug absence
  - Self-verify: 0/8 RUNNER_FALSE_NEGATIVE, 6/8 PAGE_CHANGED_FALSE_TRIGGER, 2/8 REPLAY_FAIL — **strong evidence via state_digest log analysis** (no replay needed)
- **Strong finding**: 6/8 I2 violations are actually **B-09 page_changed false trigger contamination** — runner correctly reports `action_success=False`, but `page_changed=True` is bogus (no url/title/scroll/text_sim real change signals). **I2 and B-09 share root cause** (the 12-reasons `page_changed` union includes false triggers like `form_value_changed` and `dom_complexity_changed`).
- **Status**: ❌ **NOT_A_BUG** (no runner false negatives) — but contributes ~25 ep to B-09's blast radius (288 → 313 ep effective)
- **Paper impact**: 移出 fix scope; fix B-09 自动消化 B-13's I2 violations.

---

### B-14. AXTree drift same URL (I6) — ❌ NOT_A_BUG (mostly viewport scroll natural drift)

- **Origin**: Tier 4 I6 (6002 violations — **最大 invariant**) + **static-verify case study** (2026-04-30)
- **Mechanism**: 连续两 step AXTree text similarity < 0.7 但 obs_url 没变.
- **Status**: ❌ **NOT_A_BUG via case study spot check** (2026-04-30). All 8 case_study_examples show `action=scroll` — Tier 4 prompt excluded `goto/click` but **not scroll**. Scroll changes viewport; `current_viewport_only=True` (Tier 1 axtree_finding) prunes AXTree to new viewport content; similarity drops naturally below 0.7. **6002 violations is mostly expected behavior**, not bugs.
- **Paper impact**: 不进 fix scope. Acknowledge in Section 4 limitation table as known viewport-pruning side effect.
- **Lesson**: Tier 4 invariant definitions need to exclude scroll from "non-navigating actions" baseline if measuring scaffold bugs vs design behaviors.

---

### B-15. Finish-but-eval-reject (I7 / Tier 2 finish_wrong_state) — agent error not scaffold

- **Origin**: Tier 4 I7 (1552 violations) + Tier 2 `finish_wrong_state` (1972 ep / 43.89% — **最大 silent failure category**) + **static-verify case study** (2026-04-30)
- **Mechanism**: agent 自报 finish 但 task evaluator 拒收 (FP). Case study spot check shows `action=finish; answer=18000.00` — agent gives wrong answer, not scaffold issue.
- **Blast radius**: 1972 ep / 43.89%. Site: cls 1033, red 710, shop 229. Mode: DOM 604, SoM 447, Vision 296, P-text 253, P-SoM 285, P-prompt 87.
- **Status**: ✅ **CONFIRMED as agent error** via case study (not scaffold/runner bug). Already partially covered by §95 eval_fp + §78a na_fp filter体系.
- **Fix**: agent-side, not scaffold. Section 4 cite §95 体系处理.
- **Paper impact**: Tier 5 ua_match audit 复用此 ep set 验证 GPT-judge variance.

---

### B-16. Long step unexplained (I4) — Playwright timeout swallow signal

- **Origin**: Tier 4 I4 (828 violations) + **static-verify case study** (2026-04-30)
- **Mechanism**: env_step_ms ≥ 30s 但 action_type ∉ {wait, type-长文}. 暗示 hidden Playwright timeout 被吞 (Tier 3 5-category taxonomy "Actionability Check Masking and Timeout Swallowing").
- **Status**: ✅ **CONFIRMED via case study** (2026-04-30): all 8 case_study_examples show `action=tab_focus` with env_step_ms > 30s — `tab_focus` should never take 30s legitimately, so this is real Playwright timeout being swallowed by runner without surfacing exception.
- **Fix**: surface Playwright timeout exceptions instead of swallowing; add timeout reason to step record.
- **Paper impact**: Section 4 limitation table cite — explains some "no_progress" / cycle-truncate trajectories are actually 30s+ stuck steps not agent decision loops.

---

### B-17. Repeat click no cycle break (I3)

- **Origin**: Tier 4 I3 (481 violations) + click probe taxonomy
- **Mechanism**: 同 element_id click ≥ 3 次连续 — agent 死循环, cycle-detect 没提前停.
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (covered by B-11 fuzzy cycle); SUPERSEDED by B-38 (early-stop disabled). Previously CONFIRMED overlap.
- **Fix**: same as B-11 (fuzzy cycle hash includes target role, not just exact element_id).

---

### B-18. Max-step truncate at click (I8)

- **Origin**: Tier 4 I8 (201 violations)
- **Mechanism**: episode 在 max_step 截断 但 last action 是 click 类 — max_iter masking silent failure (Tier 3 Type 5).
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (covered by B-11 fix); SUPERSEDED by B-38. Previously CONFIRMED downstream.
- **Fix**: same as B-11.

---

### B-19. Cross-step trajectory anomaly (Tier 2)

- **Origin**: Tier 2 `cross_step_trajectory_anomaly` (353 ep / 7.86%)
- **Mechanism**: trajectory-level pattern signature (e.g. action sequence cycles, no_progress repeats). Heterogeneous root cause.
- **Status**: ⚠️ **CONFIRMED but redundant with B-11/B-17/B-18** — Tier 2 mode breakdown {DOM 183, P-SoM 47, P-prompt 5, P-text 45, SoM 62, Vision 11} matches B-11 cycle pattern. 353 ep is **upper bound** for fuzzy cycle scope, large overlap with B-11 481 violations.
- **Fix**: 去重 against B-11; same fuzzy-cycle patch resolves majority. Once B-11 patch lands, re-scan to estimate residual.

---

### B-20. ua_match GPT-judge drift modes (Tier 5)

- **Origin**: Tier 5 evaluator audit + §78a (na_fp 体系) + **static-verify** (2026-04-30, code reading at helper_functions.py:623 + JSON evidence)
- **File**: `external/visualwebarena/evaluation_harness/helper_functions.py:623`
- **Model setting**: `gpt-4o-mini`, `temperature=0`, `top_p=1.0`, `max_tokens=768`, **`fixed_seed=False`** — confirmed in audit JSON. Even with temp=0, no fixed seed means model version drift can change verdict across runs.
- **Drift modes** (4 identified, all verified in static audit):
  1. **N/A task accept bias** — agent 说 "task is impossible" GPT 倾向 accept as success. **Covered** by §78a na_fp filter.
  2. **Active-finish blank/generic N/A gap** — agent finish without "N/A" keyword but answer 是空 / 通用. **NOT covered**. Spot-check found adjusted-true N/A rows with active finish.
  3. **Ambiguous task / LLM nondeterminism** — temperature=0 set but no seed; model alias overridable by `VWA_EVAL_MODEL` / `OPENAI_EVAL_MODEL` env. **NOT covered**.
  4. **Parser substring brittleness** — `llm_ua_match` checks "different" before "same"; noncompliant responses misparsed. **NOT covered**.
- **Blast radius**: would need probe — 同 trajectory 跑 N=5 GPT-judge 测 verdict 方差. Likely small (< 5% of na_fp tasks) but real.
- **Status**: ✅ **CONFIRMED via static read + evidence chain** (2026-04-30); 3/4 drift modes outside current FP filter coverage.
- **Fix**: prompt template wording + multi-sample majority vote OR replace with deterministic rule.
- **Paper impact**: Section 4 限制条款 cite (3/4 drift mode 不在现 FP filter 覆盖); 影响 raw_success vs adjusted_success 校准 baseline.

---

### B-21. string_match `fuzzy_threshold = 1.0` (Tier 5) — actually GPT-judged binary, not numerical threshold

- **Origin**: Tier 5 evaluator audit + **static-verify** (2026-04-30, evaluators.py:203 + audit JSON)
- **File**: `external/visualwebarena/evaluation_harness/evaluators.py:203` + `helper_functions.py:581`
- **Mechanism (corrected)**: `fuzzy_threshold = 1.0` is **misleading naming** — this is NOT a Levenshtein / Jaccard / edit-distance threshold. It's **GPT-judged binary**: `fuzzy_match` returns 1.0 only when GPT labels output as 'correct'. Runner success requires score >= 1.0. Effectively same logic as ua_match, just a separate path.
- **Normalization**: `lowercase=True, strip_whitespace=False, strip_surrounding_quotes=True, strip_punctuation=False`. **No whitespace stripping** — "answer " (trailing space) misses target "answer".
- **Edge cases verified in audit JSON**:
  - "$5.99" vs "5.99": `must_include` may match via tokenization, `exact_match` fails
  - "5.99" vs "$5.99": `must_include` misses ($5.99 not substring of 5.99), exact_match fails
  - "five point nine nine dollars" vs "5.99": deterministic paths miss; GPT fuzzy may accept if configured
- **Multi-answer logic**: outer reference lists are conjunctive; alternatives use literal `' |OR| '` delimiter; `one_of` does substring search (FP risk on short tokens).
- **Paper-grade sample**: 1316 string_match rows / 228 raw / 141 adjusted / 87 na_fp removed.
- **Blast radius**: needs Levenshtein scan — but bounded by 228-141=87 ep already removed by FP filter.
- **Status**: ✅ **CONFIRMED via static read + audit JSON** (2026-04-30). The bug is **less about numerical threshold and more about fuzzy_match degenerating to GPT-judge binary** (same drift modes as B-20).
- **Fix**: split fuzzy_match from GPT-judge into deterministic Levenshtein/Jaccard with explicit threshold; add whitespace stripping in clean_answer.
- **Paper impact**: Section 4 cite — string_match's "fuzzy" tier is effectively ua_match #2.

---

### B-22. program_html selector brittleness (Tier 5)

- **Origin**: Tier 5 evaluator audit + **static-verify** (2026-04-30, audit JSON `task_pool_audit` + `selector_category_counts_raw_configs`)
- **File**: `external/visualwebarena/evaluation_harness/evaluators.py:345` + raw configs
- **Counts verified**: 1068 program_html tasks / 1598 target checks. Selector pattern: 1100 `raw_js_document`, 233 `full_page`, 143 `func_get_query_text_css`, 63 `func_shopping_helper`, 31 `func_reddit_helper`, 16 `lambda_js`. **`static_brittle_flag_count: 562`** (35% of 1598).
- **Specific concerns confirmed**:
  - No `wait_for_selector`; only fixed 3-second sleep after non-last navigation
  - last-url checks receive no post-action wait
  - Raw `querySelector`/`querySelectorAll` selectors depend on exact classes, positions, child structure
  - Helper exceptions are swallowed → empty strings/zero/empty dicts (silent FN)
  - reddit latest-comment-by-username helpers are non-causal — match stale comments
  - Full-page scans can match incidental or stale text
- **Concrete examples** (from `task_pool_audit`):
  - vwa cls task 4: `func:get_query_text(__page__, '.price'); func:get_query_text(__page__, '.desc')` — class-only selectors
  - vwa shopping task 0: `document.querySelector(".order-details-items.ordered").outerText` — Magento class contract + async order-page rendering
- **Blast radius**: 562 / 1598 brittle (~35%); needs replay verification per task to estimate true FN rate.
- **Status**: ✅ **CONFIRMED via static read + concrete examples** (2026-04-30)
- **Fix**: replace querySelector with locator + `wait_for_selector`; surface helper exceptions; add causality check in latest-comment helpers.
- **Paper impact**: same direction as B-21 — likely low SR estimation; Section 4 cite.

---

### B-23. AXTree property suppression (Tier 1 finding Q1 ignored properties)

- **Origin**: Tier 1 axtree_finding "Q1 ignored properties" + **static-verify** (2026-04-30, code reading)
- **File**: `external/visualwebarena/browser_env/constants.py:288-296`
- **Mechanism**: 字面 `IGNORED_ACTREE_PROPERTIES = ("focusable", "editable", "readonly", "level", "settable", "multiline", "invalid")`. Used in `processors.py:dfs()` line 535-541: `if property["name"] in IGNORED_ACTREE_PROPERTIES: continue`. Agent 看不到 actionability / form-state clues.
- **Status**: ✅ **CONFIRMED via static read** (2026-04-30, double-verified)
- **Fix**: minimally include `editable`, `readonly`, `multiline` for input/textarea/select role elements
- **Paper impact**: relates to TYPE silent failure (B-01) — agent 不知道 element 是不是 editable 就 type → 命中非 input

---

### B-24. AXTree role inclusion non-strict (Tier 1 finding Q1)

- **Origin**: Tier 1 axtree_finding "Q1 role inclusion rules" + **static-verify** (2026-04-30, code reading)
- **File**: `external/visualwebarena/browser_env/processors.py:529-560` (dfs function)
- **Mechanism**: 字面 — `dfs()` walks `node["role"]["value"]` and `node["name"]["value"]`, only filters by exact role list `["generic", "img", "list", "strong", "paragraph", "banner", "navigation", ...]` AND only when `name.strip()` is empty. Non-strict allowlist — any node with non-empty role+name gets included.
- **Status**: ✅ **CONFIRMED via static read** (2026-04-30, design choice contributes to AXTree bloat)
- **Paper impact**: may explain Section 5 reddit Axis 1 mechanism — AXTree hierarchy bloat → sidebar forum link 埋深 → attention dilution. **Section 5 prose 已 cover**, 不单独 frame.

---

### B-25. role='link' non-`<a>` elements (Tier 1 finding Q5)

- **Origin**: Tier 1 axtree_finding "Q5 role='link' non-<a>" + **static-verify** (2026-04-30, code reading)
- **File**: `external/visualwebarena/browser_env/processors.py:813` (`get_page_bboxes`)
- **Mechanism**: 字面 `interactableSelectors` array 包含 `'[role="button"]', '[role="link"]', '[role="checkbox"]', '[role="menuitem"]', '[role="tab"]'` — non-anchor "link"-roled elements 进 SoM observations. Combined with no tag/href validation. Non-anchor links 被 bbox-center mouse action dispatch (而不是 locator actionability) — 加剧 §106 click dispatch bug.
- **Status**: ✅ **CONFIRMED via static read** (2026-04-30, double-verified)
- **Paper impact**: cite as part of B-02 root cause; 不单独 fix.

---

### B-26. `current_viewport_only` 0.6 overlap operator precedence bug (§80) 🛠️ **FIXED 2026-04-19**

- **Origin**: §80 in 实验笔记 + CLAUDE.md project knowledge
- **File**: `external/visualwebarena/browser_env/processors.py:218` (in_viewport_ratio)
- **Mechanism**: upstream `overlap_w * overlap_h / w * h` 实际是 `((ow*oh)/w)*h` (operator precedence) — 阈值 0.6 形同虚设, 任何部分可见元素都被保留并给出**完整文本**.
- **Blast radius**: affects all DOM/SoM 模式 (Vision 不受影响 — 不靠 AXTree).
- **Status**: 🛠️ **FIXED 2026-04-19** (commit `3f9ceca` on VWA submodule branch `p79-patches`). One-line paren fix: `ratio = (overlap_w * overlap_h) / (width * height)`. All B0+B1 DOM/SoM conditions re-run after fix (§80 decision). Earlier classification "NOT_FIXED_BY_DESIGN" superseded by /stress A1.18 audit 2026-05-16 (gemini OOB catch confirmed by codex F8): code had been fixed but paper §4.X.5 prose + this catalog row remained stale until A1.18 sweep.
- **Paper impact**: Section 4.X.5 prose rewritten 2026-05-16 to reflect FIXED status; §1 hero claim no longer depends on viewport bug as DOM-advantage confound source.

---

### B-28. §50 scroll direction confusion — agent prompt limitation 🛠️ MITIGATED (schema only)

- **Origin**: §50 (2026-04-14), 实验笔记 `[bug][finding]` + `B0_DOM_digest.md:98`
- **Mechanism**: 235B model 经常猜错 scroll `delta=[dx, dy]` 的方向（`dy<0` 向上 vs 向下），连续 3 次 scroll page_changed=False 触发 cycle 截断。原始 schema 暴露 `delta` 数值 → model 按自然语言理解（不一致）。
- **Mitigation (already shipped)**: §67 — Tool schema 把 `delta: [dx, dy]` 替换成 `scroll_direction: enum("up", "down")`. Mitigated but **not eliminated** (B0/B1 schema 不完全对称, B0 仍可能受影响).
- **Status**: 🛠️ **MITIGATED — schema only** (paper-disclosed limitation, B0_DOM_digest.md §6)
- **⚠️ Sharpened by /stress A1.1 2026-05-15**: schema fix 没解决**effective action vocabulary asymmetry**。`proxy_api_agent.py:749-752` 在 model 输出后 hard-code `scroll_direction → delta=[0, ±0.8]`，B0 实际 scroll 步长 binary `±0.8` 不可调；B1/B2 prompt 仍教 `delta=[dx, dy]` (`qwen3vl_agent.py:181`)，model 自由出任意 dy magnitude → continuous space。Cross-baseline SR 比较 (尤其 reddit search-loop / scroll-heavy 任务)：B0 SR 若 ≠ B1/B2，无法区分 "capability gap" vs "binary scroll vocab 残废"。Paper §3 footnote 需 disclose: "B0 effective scroll = {up, down} binary at ±0.8 magnitude (post-process clamp); B1/B2 effective scroll = continuous `dy ∈ [-1, 1]`. Cross-baseline scroll behavior is not byte-equivalent at action-space level."
- **Paper impact**: Section 4 limitation table cite — 需补 "effective action vocab asymmetry on scroll axis" 段。Either disclose (0 effort) 或修代码 (B1/B2 也 clamp 到 ±0.8 + 重跑 Phase 1a 数据)。User decision pending — disclosure 推荐 default。
- **Note**: This is **agent prompt schema + post-process clamp** issue, not scaffold bug. /stress A1.1 finding F2 触发此 status sharpening。

---

### B-29. §55 delete success signal missing — site UX limitation, not fixed

- **Origin**: §55 (2026-04-14), 实验笔记 `[bug]` + `B0_DOM_digest.md §6`
- **Mechanism**: classifieds delete operations use `onclick="javascript:return confirm('...')"`. After confirm acceptance, success signal is shown via flash message that disappears before next AXTree snapshot — agent **cannot perceive delete success**. All delete tasks struggle.
- **Status**: 🛠️ **NOT_FIXED_BY_DESIGN** ("非结构性缺失", §55 本身记录"暂不修复")
- **Mitigation**: classifieds delete tasks 用 `require_reset=true` flag + `program_html` evaluator (404 check after delete) — evaluator 自己 navigate to deleted item URL 验证 404, not relying on agent finish. So **task-pool level mitigated** even if agent UX broken.
- **Paper impact**: cite as known site UX brittleness; affects classifieds task subset (~10% of cls).

---

### B-30. searchbox-no-type pattern — partially redeemed as B-33c

- **Status**: ⚠️ **PARTIAL RETRACT + REINSTATE as B-33c sub-mode 2026-04-30**
- **Original retraction reason**: 24% number was conversation-memory, not in artifact directly
- **Re-investigation finding**: artifact's AXTREE_MAPPING_ERROR sub-pattern `input.form-control` (18/52 = 35% of that umbrella, **19.4% of all click-loop**) IS the search-input-click-no-type pattern. Earlier 24% was directionally correct, just sourced from wrong taxonomy level.
- **Redirect**: see **B-33c** in B-33 family for current-state catalog entry
- **Lesson**: When an entry is based on memory but the underlying pattern exists, retract the entry but track via a verified parent (B-33c). Don't lose the signal.

---

### B-31. heading-as-link pattern — partially redeemed as B-33b

- **Status**: ⚠️ **PARTIAL RETRACT + REINSTATE as B-33b sub-mode 2026-04-30**
- **Original retraction reason**: 12% number was conversation-memory
- **Re-investigation finding**: artifact's AXTREE_MAPPING_ERROR sub-pattern `h2.(no-cls)` (11/52 = 21% of umbrella, **11.8% of all click-loop**) IS the heading-as-link pattern. The 12% was approximately right.
- **Redirect**: see **B-33b** in B-33 family for current-state catalog entry

---

### B-32. button-AJAX silent — confirmed real, reddit subscribe pattern

- **Origin**: `logs/codex/click_root_cause_probe.json` (7/93 = 7.5%) + spot-check verified
- **Mechanism**: All 7 cases are reddit `<button>` Subscribe / Unsubscribe — bbox center hits `span.subscribe-button__label-text` (child span) instead of the `<button>` parent. Click fires but goes to span (no JS action attached), AJAX state of subscribe doesn't update → agent re-clicks same button.
  - Sub-mode of B-33 AXTree-mapping-error (bbox is button, center hits span)
  - Plus AJAX timing: even if click reached button, subscribe state change is async — observation snapshot may miss it (related to B-09)
- **Blast radius**: 7/93 click-loop = **7.5% of click-loop** ≈ 0.4% of all ep
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (locator-route Cluster 1 + state_change Cluster 2); previously CONFIRMED narrow scope
- **Site/mode locality**: 100% reddit, mostly DOM/SoM (not Vision since Vision uses normalized coords)
- **Fix**: locator-route mouse dispatch (Cluster 1, B-01 patch) + observation refresh after click (B-09 patch) — covered by Phase A clusters
- **Paper impact**: narrow, no separate framing; absorbed by Cluster 1 + Cluster 2 fixes

---

### B-33. 🆕 AXTree-DOM dispatch-target mapping family (REAL #1 click-dispatch bug umbrella)

- **Origin**: `logs/codex/click_root_cause_probe.json` (52/93 = **55.9%** of click-loop) + **Tier 10 sweep `probe_tier10_dispatch_target.py`** (18 click cases, 94.4% off-target) — discovered 2026-04-30
- **Mechanism**: AXTree assigns `union_bound` bbox to element_id `[N]`, but `mouse.click(bbox_center)` consistently hits a CHILD element instead of the AXTree-nominal target. Pattern is umbrella over multiple sub-modes:

  **B-33a Listing-card child span** (cls Magento listing pages, ~10/52 in artifact + 3/18 Tier 10 click)
  - bbox = listing card `<li>` bounding box
  - center hits `span.date` / `span.location` / `span.price` / `span.desc` / `div.listing-attributes`
  - examples: cls task 117/123/204 (date/location spans)

  **B-33b Heading-as-link** (cls reddit, 11/52 in artifact + 1/18 Tier 10 click)
  - bbox encompasses heading `<h2>` containing inner anchor
  - center hits `<h2>` directly; agent expected click would navigate but heading has no onclick
  - **NOT a fabricated pattern** (despite earlier B-31 retraction): real artifact 21% of AXTree mapping errors

  **B-33c Search-input click without type follow-up** (18/52 in artifact + 1/18 Tier 10 click)
  - bbox center hits `input.form-control` / `input.input-text` (search box)
  - Click focuses input but agent doesn't follow with `type` action → cycles
  - Mixed scaffold/agent: bbox is correct (it's an input) but agent doesn't continue to type
  - **NOT fully fabricated** (B-30 was directionally correct): real artifact 35% of AXTree mapping errors

  **B-33d Icon-or-image inside link/button** (5/18 Tier 10 click = **28%** — NEW sub-mode not in artifact)
  - bbox is `<a>` or `<button>` containing `<svg>` / `<img>` / `<i>` icon
  - center hits `<svg>` / `<img>` which has no onclick handler
  - click fires but doesn't navigate
  - cls/red/shop all affected

  **B-33e Reddit submission header** (16/17 in §106 / B-02 sub-mode)
  - bbox is `header.submission__header` block, center hits header padding
  - Same architecture as B-33 but classified as UNION_BOUND_BUG by codex

  **B-33f Block parent (div / li / main)** (5/18 Tier 10 click = 28%)
  - bbox is anchor, but click center falls in block-level parent (li, div, main, ul) due to inline gaps or padding
  - Same as §106 inline gap pattern

- **Aggregate blast radius**:
  - In artifact 93 click-loop cases: B-33 family = 52 + ~16 (B-33e from §106 sub-mode) = **~68/93 = 73%** of click-loop bugs
  - In Tier 10 sweep on FAILED click steps: 17/18 = **94.4% off-target**
  - Effective ALL ep impact: 5.4% click-loop signature × 73% = **~3.9% of all paper-grade ep**, plus all UNION_BOUND_BUG cases (1.6%) = **~5.5% combined bbox-mapping bug rate**
- **Status**: 🛠️ **FIXED commit `3c15cd7`** (locator-route Cluster 1 — bypasses union_bound center → real Playwright locator dispatch). Previously CONFIRMED via artifact + Tier 10 sweep. 🚨 **Largest single dispatch contaminant family** in framework.
- **Fix**: Cluster 1 (locator-route via Playwright `locator.click()`) **automatically resolves all sub-modes a-f** — Playwright's locator finds the actual `<a>`/`<button>` element via accessibility tree + actionability check, bypassing geometric-center mouse dispatch entirely
- **Paper impact**: **Section 4 limitation table PRIMARY cite** — replaces §106's 1.6% framing with broader 5.5% combined bbox-mapping family. Cross-mode bias: DOM/SoM/P-text/P-SoM all affected; Vision largely unaffected (uses normalized coordinate fallback). DOM:SoM ratio is small (similar bug rates) so cross-mode SR comparisons partially cancel.

---

### Tier 10 audit summary — dispatch-effective-target across action types

`probe_tier10_dispatch_target.py` (2026-04-30, 54 cases × 3 action types):

| Action | ON_TARGET | OFF_TARGET | Top off-target sub-modes |
|---|---:|---:|---|
| click | 5.6% (1/18) | **94.4%** | ICON_OR_IMG_INSIDE 28%, BLOCK_PARENT 28%, SPAN 22%, HEADING 6% |
| type | 0% (0/17 replay-ok) | **100%** | BLOCK_PARENT 41%, OTHER (div wrappers) 53%, ICON 6% — confirms B-01 |
| select_option | 0% (0/18) | **100%** | OTHER (custom dropdown) 67%, OTHER_SPAN (ARIA combobox) 33% — confirms B-06 broad |

**Conclusion**: All 3 main dispatch action types have ≥94% off-target rate when action_success=False. Confirms framework's `mouse.click(union_bound_center)` + `keyboard.type` pattern is **systematically wrong target** in failure cases. Cluster 1 locator-route fix is justified as the highest-leverage Phase A patch.

Hover and Clear actions had 0 cases collected (VWA agents almost never use these) — out of scope for Tier 10.

---

### B-34. Tier 7 stale auth file masks refresh subprocess crash 🆕

- **Origin**: Tier 7 audit (2026-04-30, static read of `p79/utils/auth_refresh.py:148`)
- **File**: `p79/utils/auth_refresh.py:148-150`
- **Mechanism**: `if r.returncode == 0 and auth_file.exists():` returns True. 但如果 subprocess crashed AFTER login succeeded but BEFORE `ctx.storage_state(path=...)` line wrote the new file, **the previous successful refresh's stale auth_file still exists** → condition passes → caller (runner / watchdog) believes auth refreshed but cookies are days/hours old.
- **Severity**: low frequency (subprocess rarely crashes mid-execution) but **silent failure mode** when it happens — no log/warning to operator
- **Status**: ✅ **CONFIRMED via static read**
- **Fix**: write auth_file path to a tempfile-with-PID first (e.g. `auth_file.with_suffix('.json.refreshing.{pid}')`), then atomically `os.replace()` to final path only after subprocess exits cleanly. ~10 LOC.
- **Paper impact**: low — but worth Section 4 limitation cite as known auth pipeline brittleness

---

### B-35. Tier 7 auth refresh interval is episode-count only, not time-based 🆕

- **Origin**: Tier 7 audit (2026-04-30, `p79/utils/auth_refresh.py:170-183` `should_refresh`)
- **File**: `p79/utils/auth_refresh.py:170-183`
- **Mechanism**: `should_refresh()` returns True only when `episodes_since_refresh >= interval`. **No time-based check**. PHP session `gc_maxlifetime=1440s` (§39) means cls/shopping sessions expire after 24 minutes. If a single difficult episode runs 30+ min (e.g., max_step=30 × 60s/step latency), session expires mid-episode but auth refresh isn't triggered until episode count crosses interval.
- **Compounding factor**: B-16 long-step Playwright timeout swallow (env_step_ms > 30s tab_focus etc.) makes some episodes very long, increasing exposure.
- **Severity**: medium — affects long episodes on cls/shopping
- **Status**: 🛠️ **FIXED commit `<TBD>`** (笔记 §116.9, 2026-05-08): `should_refresh()` adds `seconds_since_refresh` parameter + config `time_interval_seconds: 1200` (below PHP gc_maxlifetime 1440s). Runner tracks `_auth_last_refresh_ts` per site. Previously CONFIRMED static.
- **Fix**: add `time_since_refresh` check in `should_refresh()` with threshold ~1200s (below gc_maxlifetime 1440s). ~15 LOC.
- **Paper impact**: medium — could explain some inconsistent SR in tasks with long action sequences

---

### B-36. Tier 8 image compression scale-to-0.4 may break SoM mark readability 🆕

- **Origin**: Tier 8 audit (2026-04-30, `p79/backends/image_utils.py:50-67`)
- **File**: `p79/backends/image_utils.py:14-18` (`COMPRESSION_PRESETS`)
- **Mechanism**: cascading presets `[(85,1.0), (70,1.0), (55,0.9), (40,0.8), (30,0.7), (25,0.6), (20,0.5), (20,0.4)]`. For large screenshots (1920×1080) with many SoM marks, payload may exceed 5MB even at q=85, falling through to q=20 + scale=0.4 — that's 768×432 viewport at JPEG quality 20. **SoM mark numbers (typically 12-14px font) become 5-6px → unreadable for Vision/SoM agents**.
- **Asymmetric impact**: only affects SoM/Vision modes (DOM mode doesn't use image). Among visual tasks with high mark density (reddit threads with 50+ marks), this is the highest-risk subset.
- **Severity**: low frequency (most pages compress at q=85 or q=70 fine) but **systematic for high-mark-density Vision/SoM tasks**
- **Status**: ✅ **CONFIRMED via static read**
- **Fix**: gate scale-down with mark-density threshold; for SoM mode, fall back to **content-aware crop** (split image into quadrants if too big) instead of resize. ~30 LOC.
- **Paper impact**: low — Section 4 cite as Vision/SoM noise source on dense pages
- **Existing related**: §94 mentions SoM mark count 80→200 expansion which exacerbates this — denser marks more likely to hit compression cliff

---

### B-37. Tier 12 seed=42 is metadata only — **B0 explicitly stochastic by config** 🚨🆕

- **Origin**: Tier 12 audit (2026-04-30) — discovered via grep + config inspection
- **Triple-evidence verification** (8th-pass static, 2026-04-30):

  **Evidence 1 — zero RNG seeding in p79**:
  - `grep -rn "random.seed\|np.random.seed\|torch.manual_seed\|set_seed\|seed_everything" p79/` returns **ZERO matches**
  - `self.seed=42` written into step records / condition_meta but never to any RNG

  **Evidence 2 — B0 configs ALL use temperature=0.1** (not 0!):
  - `grep "temperature" configs/exp_v2_B0_*.yaml`: **18/18 configs set `temperature: 0.1`**
  - `proxy_api_agent.py:600`: payload uses `gen_cfg.get("temperature", 0.1)` default 0.1
  - At temperature=0.1 with no seed, model **explicitly samples** stochastically — not greedy. Each API call produces different output.
  - This is **B0 non-deterministic by design**, regardless of seed propagation.

  **Evidence 3 — payload omits `seed` parameter**:
  - `proxy_api_agent.py:596-601`: `payload = {model, messages, max_tokens, temperature}`. **No `seed`. No `top_p` even.**
  - Anthropic API native protocol (used by proxy) **has no `seed` parameter at all** — it's an OpenAI-only feature
  - So even if we wanted to forward seed to B0, the API doesn't accept it

  **B1 partial determinism**:
  - `qwen3vl_agent.py:508`: `do_sample=False` (greedy top-1) → mostly deterministic on same hardware/input
  - But: no `torch.manual_seed(seed)` for non-LLM RNG (image preprocessing, weight init reproducibility, etc.)
  - Same-host re-run should mostly match for B1 trajectories, but **CUDA floating-point non-determinism** can flip top-1 logit comparisons in rare cases

- **Severity**: 🚨 **CRITICAL for paper reproducibility claim**:
  - **B0 (235B Qwen3-VL via proxy)**: explicitly stochastic at temp=0.1, **trajectories DIFFER between runs by design**
  - **B1 (4B Qwen3-VL local)**: greedy → mostly stable but **not strictly deterministic** without torch seeding
- **Status**: 🛠️ **PARTIALLY FIXED (2026-05-08 audit re-verify)**
  - ✅ `runner/main.py:81-94` — `random.seed(seed) + np.random.seed(seed) + torch.manual_seed(seed)` deployed (Phase A Cluster 4 fix)
  - ✅ `proxy_api_agent.py:226,603` — `temperature: 0.0` (NOT 0.1 as old catalog said), `top_p: 1.0` explicit
  - ✅ `proxy_api_agent.py:609-614` — seed forwarded to API as `payload["seed"]` if provider supports OpenAI-compat
  - ✅ `qwen3vl_agent.py:519` — `do_sample=False` (greedy)
  - ⚠️ Anthropic API native protocol still has no `seed` parameter — B0 best-effort but not guaranteed deterministic
  - ⚠️ CUDA floating-point non-determinism remains (sm_121 vs sm_80 / sm_70 cross-machine drift); paper §3 disclose
- **Fix options**:
  - **(a) Code fix B0**: change `temperature: 0.1 → 0` in 18 configs + add OpenAI-format proxy with seed support. ~30 LOC + config sweep. **Requires re-running all B0 data (impossible for archived runs).**
  - **(b) Code fix B1 only**: add `torch.manual_seed + torch.cuda.manual_seed_all` at runner condition iteration start. ~10 LOC. Tightens B1 reproducibility for FUTURE runs.
  - **(c) Paper revision**: Section 4 reframe — "B0 uses temperature=0.1 with no seed forwarding (proxy API limitation); B1 uses greedy decoding (do_sample=False). We report seed=42 as configuration metadata. Trajectories are not strictly deterministic across re-runs, but action distributions are stable in expectation; SR aggregates are reported with N=1 per task per condition." **Plus add B-37 to known limitations.**
- **Recommended path**: **(b) + (c) hybrid** — add B1 torch seeding for future runs (cheap), revise paper claim to honest disclosure for archived data (no re-run needed)
- **Paper impact**: 🚨 **VERY HIGH** — affects reproducibility claim across whole paper. Section 4 limitation table must explicitly cite B-37 with B0 temperature disclosure.

- **Bonus finding**: temperature=0.1 across 18 configs is **a deliberate choice** (someone changed default from 0 → 0.1 at some point — git blame would show when). This may have been to **avoid mode collapse** in greedy decoding for vision tasks. Trade-off: better task variety per call vs reproducibility. Worth a paper Section 4 footnote.

---

### B-27. SoM mark numbering not stable across observations

- **Origin**: Conversation pushback (user #3) + `p79/experiment/som.py`
- **Mechanism**: SoM marks 从 AXTree action-able 子集按遍历顺序 enumerate — page reflow / scroll / AJAX update 后重排. SoM `[3]` 不保证是同一元素 across observations.
- **Status**: ❌ **NOT_A_BUG** (by design, same as B-12)
- **Paper impact**: Section 5 reddit Axis 1 mechanism 不依赖 SoM mark stability — 用 flat-list-reduces-attention-dilution 论证 (not ID-stability).

---

## Fix Scope Decision Matrix

| Bug | Status | Blast radius | Fix LOC | Paper impact | Phase |
|---|---|---:|---:|---|---|
| B-01 TYPE silent failure | ✅ | 12.22% / 549 ep | ~30 | High (asym across baselines) | **A (now)** |
| B-09 page_changed false trigger | ✅ | 5.7% / 288 ep | ~50 | Medium (fig0a SR derive) | **A (now)** |
| B-11 / B-17 / B-18 fuzzy cycle | ✅ | ~15-20% trace | ~40 | High (cost inflate) | **A (now)** |
| B-02 §106 click dispatch | ✅ | 1.6% (real) | shared with B-01 | High (cross-mode 1.5×) | **A (now)** |
| B-10 §105 swatch | 🛠️ | 2.4% | (fixed) | (done) | **DONE** |
| B-03 CLEAR | ✅ static | inherits B-01 100% | shared with B-01 patch | Low | **A (with B-01)** |
| B-04 HOVER | ✅ static | low frequency | shared with B-01 patch | Low | **A (with B-01)** |
| B-05 UPLOAD double bug | ✅ static | very rare action | shared + creator fix | Low | **A (with B-01)** |
| B-07 NEW_TAB / target=_blank | ✅ static | 13% click-loop share | wrap with expect_popup | Low | **A (with B-01)** |
| B-06 SELECT_OPTION arg-drop | ✅ (small) | 0.3% / 15 ep (self-replay) | 1 line | Low (appendix) | **D (defer)** |
| B-08 SCROLL silent | ✅ (small) | 0.8% / 35 ep (self-replay) | ~20 | Low (was thought Medium; not Vision-only) | **D (defer)** |
| B-15 finish_wrong_state | ✅ agent error | 43.89% | (FP filter, §95) | High (raw vs adj) | **C (post-Phase A)** |
| B-16 long step unexplained | ✅ static | 1.85% | ~15 | Low | **C** |
| B-20 ua_match GPT drift | ✅ static | unknown (probe needed) | prompt template | High (Section 4 limit) | **C (probe optional)** |
| B-21 string_match GPT-judged | ✅ static | bounded by 87 ep already filtered | ~30 | Medium | **C** |
| B-22 program_html brittleness | ✅ static | 562/1598 brittle | ~80 | Medium | **C** |
| B-23 AXTree property suppression | ✅ static | (design issue) | ~5 | Low | **D (research-only)** |
| B-12 element_id role drift | ❌ | — | — | acknowledge | **NONE** |
| B-13 action_fail/page_changed | ❌ self-verified | (rolled into B-09) | — | — | **NONE** |
| B-14 AXTree drift same URL | ❌ self-verified | (scroll viewport prune) | — | — | **NONE** |
| B-19 cross-step anomaly | ⚠️ redundant | 353 ep (overlap B-11) | (de-dup) | — | **NONE** |
| B-24/25/27 design notes | ✅ static | — | — | Section 4 cite | **NONE** |
| B-26 §80 in_viewport_ratio | 🛠️ **FIXED 2026-04-19** | all DOM/SoM | commit `3f9ceca` | Section 4.X.5 (rewritten 2026-05-16) | **DONE** |
| B-28 scroll direction | 🛠️ MITIGATED (§67) | partial B0 | (schema fix done) | Section 4 cite | **NONE** |
| B-29 delete signal missing | 🛠️ NO_FIX | ~10% cls | (require_reset+program_html mitigates) | Section 4 cite | **NONE** |
| B-30 searchbox-no-type | ⚠️ partial retract → B-33c | 19.4% click-loop | shared with B-01 (Cluster 1) | Section 4 cite | **A via B-33c** |
| B-31 heading-as-link | ⚠️ partial retract → B-33b | 11.8% click-loop | shared with B-01 (Cluster 1) | Section 4 cite | **A via B-33b** |
| B-32 button-AJAX silent | ✅ confirmed (reddit subscribe) | 0.4% all ep | shared with B-01 + B-09 | Section 4 cite | **A (Cluster 1+2)** |
| **B-33 AXTree-DOM mapping family** | ✅ CONFIRMED via artifact + Tier 10 | **5.5% all ep (combined click bbox bugs)** | shared with B-01 (Cluster 1 locator-route) | **Section 4 PRIMARY cite** | **A (Cluster 1)** |
| B-34 stale auth file mask | ✅ static (Tier 7) | low (subprocess crash rare) | ~10 | low | **C (Tier 7 cluster)** |
| B-35 auth refresh no time-based | ✅ static (Tier 7) | medium (long ep on cls/shop) | ~15 | medium | **B (with §107 wave)** |
| B-36 image compression cliff | ✅ static (Tier 8) | low (high-density SoM only) | ~30 | low | **C** |
| **B-37 seed not propagated to RNG** | ✅ static (Tier 12) | **paper reproducibility claim** | ~25 | **VERY HIGH (Section 4)** | **A or paper revision** |

---

## Codex Replay Quality Issue (probe_audit_verification meta-concern)

**Date**: 2026-04-30 (discovered during conversation)
**Symptom**: codex `probe_audit_verification` reported scaffold fractions for 6 categories. Self-replay on cls task 0 step 5 (Vision, B0_3mode_classifieds) revealed:
- Codex reported "scroll 576/2727 -> 1296"
- Self-replay (faithful 3-prior-scroll replay): start position **y=1728** (not 576), max scrollHeight **2635** (not 2727)
- Codex 数字 looks consistent with 没忠实 replay 前置 3 个 scroll，从 fresh y=0 单 scroll 跑了一次

**Implication**:
- TYPE 100% scaffold rate: ✅ trustworthy (cross-checked with §52/§64; signature is robust)
- I9 83%: ❌ rejected for being NOT_A_BUG (by-design observation-local re-numbering)
- I10 80%: ✅ trustworthy (root cause independently traced to `form_value_changed` false trigger)
- I2 0%: ✅ trustworthy (3/5 REPLAY_FAIL acknowledged in classification)
- **SCROLL 0.3 / SELECT 0.286**: ⚠️ **数字不可信** — codex possibly didn't replay prior steps; need self-replay ≥ 20 case per category to recalibrate
- All 50-cases-total cap may have been too small for low-scaffold-rate categories (CI ±15% on 10-case sample at 0.3 rate)

**Action**: re-do SCROLL + SELECT with explicit faithful prior-step replay; document codex-replay limitation as known noise source for future audit cycles.

**Update (2026-04-30, after self-verify)**: Codex's reliability is **bounded to Playwright-replay-based probes** (the `probe_audit_verification` 50-case probe). Static audits (Tier 1/2/4/5) and JSONL signal/invariant mining do NOT depend on replay/bbox. The bbox bug also has **asymmetric impact** across categories:
- **Unaffected**: TYPE (bug = "wrong element type", any bbox formula yields wrong element); I9/I10 (work on logged AXTree text comparisons)
- **Severely affected**: SCROLL/SELECT (need correct DOM element identification at bbox)

So out of codex's 6 probed categories, only SCROLL+SELECT had over-classification; the other 4 hold up under self-verification.

---

## Cross-Architecture Bug Sharing

```
                 §106 union_bound center mismatch
                             |
        ┌────────────────────┼────────────────────┐
        ↓                    ↓                    ↓
    B-01 TYPE          B-03 CLEAR            B-04 HOVER
    B-02 CLICK         B-05 UPLOAD           (Tier 1
    (probe verified)   (UNVERIFIED)           candidates)
                             |
                             ↓
                        B-25 role='link' non-<a>
                        (forces bbox dispatch even
                         when locator could work)

    Fix B-01 + B-02 with locator route → automatically resolves B-03/04/05
```

```
                  page_changed = bool(12-reasons)
                             |
                ┌────────────┼────────────┐
                ↓            ↓            ↓
         form_value     scroll_changed  text_similarity
         changed         (≥5px)         < 0.95
                             |
                             ↓
                    B-09 I10 false trigger
                    (cycle-detect / SR derive contaminated)

    Fix: split runner_page_changed (12-union) vs agent_visible_changed
         (url+title+text+scroll only); JSONL record latter
```

```
                       cycle detect = exact match
                             |
                ┌────────────┼────────────┐
                ↓            ↓            ↓
         B-11 search-     B-17 repeat    B-18 max-step
         loop (variant     click          truncate at
         queries miss     (exact id     click (cycle
         exact match)     match miss)    miss → max iter)

    Fix: fuzzy hash on (action_type, url_path_no_query, axtree_role)
         resolves all three with one patch
```

---

## Tier Coverage Gap Analysis — 5-Tier 之外可能漏掉的 bug 维度

Current 5-tier audit covered: Tier 1 (static dispatch + AXTree code) / Tier 2 (silent failure JSONL signal mining) / Tier 3 (literature taxonomy) / Tier 4 (invariant violations on logged steps) / Tier 5 (evaluator-side static audit). The B-33 discovery (AXTREE_MAPPING_ERROR 55.9% via click probe artifact, NOT covered by Tier 1's "AXTree element_id assignment Q2" finding) shows 5 tiers had a **dispatch-effective-target audit gap** — Tier 1 only checks how `element_id` is assigned to AX node IDs, not how `union_bound` actually maps to physical click targets. **This pattern of gaps suggests other untouched dimensions:**

### Tier 0: Backend / model serving layer
- vLLM / Ollama / OpenAI proxy retry & timeout behavior
- Token accounting accuracy (input_text vs input_image split)
- EOS handling, max_tokens enforcement, sampling params (already partially in §47)
- Backend warm-up / cold-start variance affecting latency metrics
- Prompt cache effects (B0 proxy server-side cache, B1 vLLM prefix cache)
**Risk**: low for SR but real for cost/latency derived metrics

### Tier 6: Data infrastructure layer
- `read_jsonl_dedup` correctness (restart resume + dedup logic)
- `logger_v2.py` JSONL fsync atomicity (partial line on crash)
- `condition_summary_v2.json` aggregation race conditions
- `analyze_run` rederive idempotency
- Schema migration v2→v3 backward compatibility
**Risk**: medium — silent data loss / counts wrong

### Tier 7: Authentication / session layer
- `auth_refresh.py` storage_state aging (cookies expire mid-run)
- Cross-mode auth race conditions (already partially in CLAUDE.md hard rule)
- PHP session `gc_maxlifetime=1440s` (§39)
- Magento FPC TTL (§104 fixed)
- Tailscale Network reachability flicker
**Risk**: medium — episodes silently fail / login-redirect loops

### Tier 8: Image preprocessing pipeline (Vision mode)
- Image resize / aspect ratio handling (`p79/backends/image_utils.py`)
- Image token count for cost calculation
- Image quality/encoding (JPEG quality, PIL resampling)
- Reference image leakage between modes (§46 fixed)
- BLIP-2 captioning on visual tasks (§59 fixed but might still affect)
**Risk**: medium for Vision SR, low for DOM/SoM

### Tier 9: Cross-experiment isolation
- Docker container shared state across runs (cart, comments, listings)
- User account login overlap (already CLAUDE.md hard rule)
- File system race (multiple runs writing to same results subtree)
- VWA reset script reliability (`scripts/queues/queue_baseline.sh` reset path)
- Restart resume preserving wrong state across rederive
**Risk**: high if violated, but largely mitigated by hard rules

### Tier 10: AXTree-DOM dispatch-effective-target mapping (NEWLY DISCOVERED via B-33)
- How `union_bound` field gets populated for each AX node (per `processors.py:157-297`)
- Which DOM child element actually receives `mouse.click(center)` (B-33 discovery: usually NOT the AX node's nominal target)
- AXTree node → DOM node identity preserved across observations
- Click-effective-target analysis for B-02 / B-33 / B-32 / B-25 family
**Risk**: HIGH — B-33 is 3.0% of all ep, biggest single click contaminant

### Tier 11: Site-side state-machine bugs
- Magento FPC stale page (§104 fixed)
- Osclass cookie/session edge cases
- Postmill cycle redirect chains
- Site-side rate limiting / 503 retry (§40 fixed)
- Database state reset reliability per task class
**Risk**: medium — site-specific but already monitored

### Tier 12: Determinism / reproducibility
- Random seed propagation (Python random, NumPy, torch)
- LLM sampling determinism (temperature=0 not enough — §47 mentions, B-20 ua_match cite)
- Episode replay reproducibility (re-running same task gives same trajectory?)
- Time-dependent state (timestamps, "today" in tasks)
**Risk**: medium — affects whether observed differences are signal vs noise

### Tier 13: Module fallback hooks (M1-M4, Phase 3 not active)
- `m1_dom_select_fallback` correctness when activated
- `m2_dom_first_input_fallback` race conditions
- `m3_failure_trigger_retry` logic
- `m4_two_stage_generation_grounding` interaction with backend
**Risk**: high if activated, deferred to Phase 3

---

**Recommended next-pass audit priority** (ranked by paper impact × discovery probability):
1. **Tier 10 dispatch-effective-target** (HIGH) — already partially probed; complete by reading `processors.py` union_bound population logic + scanning all element_id click cases for child-vs-parent mismatch
2. **Tier 6 data infrastructure** (MEDIUM) — silent data loss could affect SR computation; check via `read_jsonl_dedup` corner cases + restart resume tests
3. **Tier 12 determinism** (MEDIUM) — paper-grade reproducibility claim depends on this
4. **Tier 8 image preprocessing** (MEDIUM) — affects Vision-only mode, important for axis 3 mechanism analysis
5. Tier 0 / 7 / 11 — lower priority, mostly already monitored or out-of-scope

Each tier audit costs ~30 min static + 30 min probe. Total budget for full coverage ~6 hours. **Don't audit until Phase A patches land** — current 33-entry catalog is enough for Phase A; deeper audits should follow Phase A re-run to capture new emergent bugs.

---

## Update Log

- **2026-04-30 first pass**: Initial document; consolidates 5-tier audit (B-01 to B-22), conversation findings (B-09 root cause, B-11 fuzzy cycle, B-12/B-27 NOT_A_BUG), historical 实验笔记 entries (§52/§64/§80/§105/§106), and codex-replay-quality meta-concern.
- **2026-04-30 ninth pass (B-37 verification)**: Reached B-37 verification verdict via 3-evidence static convergence (no live API call needed, .env not readable in audit context):
  - **Evidence 1**: `grep -rn "random.seed\|np.random.seed\|torch.manual_seed\|set_seed\|seed_everything" p79/` → ZERO matches
  - **Evidence 2**: 18/18 B0 configs explicit `temperature: 0.1`; `proxy_api_agent.py:600` defaults to 0.1
  - **Evidence 3**: payload omits `seed`; Anthropic API native protocol has no `seed` parameter
  - **Git blame**: temperature=0.1 set at commit `557f47fe` (2026-04-09 by Quarkgluonmixture), unchanged since. Deliberate design choice (probably to avoid mode collapse), but **never disclosed in paper Section 4**.
  - **B0 verdict**: explicitly stochastic by design; SR data is N=1 sample from a stochastic process, not deterministic trajectory
  - **B1 verdict**: greedy (do_sample=False) → mostly deterministic but no torch seed = CUDA non-determinism risk
  - **Recommended action**: Section 4 limitation table must add explicit B-37 disclosure ("B0 uses temperature=0.1 with no seed forwarding; trajectories vary across re-runs"). Future-proof B1 by adding torch.manual_seed at runner condition iteration. **No archived data re-run needed** — historical data already represents one stochastic sample, paper should report it as such.

- **2026-04-30 eighth pass (Tier 6/7/8/12 supplementary audit)**: Static read of remaining 4 tier dimensions:
  - **Tier 6 data infrastructure**: ✅ no critical bugs. `logger_v2.py` uses `tmp + os.replace + fsync` for all atomic writes; `io_utils.py` `dedup_restart_lines` correctly handles step_idx=0 boundaries. Schema migration scaffolding exists for future v2→v3.
  - **Tier 7 auth/session**: 2 NEW BUGS:
    - **B-34** stale auth file masks subprocess crash (`auth_refresh.py:148` checks `auth_file.exists()` only)
    - **B-35** episode-count-only refresh with no time-based check (gc_maxlifetime 1440s risk for long episodes)
  - **Tier 8 image preprocessing**: 1 minor bug:
    - **B-36** image compression cascade can drop to scale=0.4 + q=20 → SoM mark numbers unreadable on dense pages (~50+ marks)
  - **Tier 12 determinism**: 1 CRITICAL bug:
    - **B-37 🚨** `grep` for `random.seed / np.random.seed / torch.manual_seed / set_seed / seed_everything` across `p79/` returns **ZERO matches**. `self.seed=42` is purely metadata — never propagated to any RNG. Paper's reproducibility claim is unsupported. This is the most paper-impactful bug found in audit since B-33.
  - **No bugs found in Tier 6** (data infra is rigorous via tmp+os.replace+fsync pattern).
  - **Catalog now 37 entries** (was 33 + 4 new).

- **2026-04-30 seventh pass (Tier 10 dispatch-effective-target sweep)**: `scripts/maintenance/probe_tier10_dispatch_target.py` (54 cases × 3 action types) revealed:
  - **click ON_TARGET only 5.6%** — 17/18 failed clicks land on wrong DOM target via `mouse.click(union_bound_center)`. New sub-modes discovered:
    - **B-33d ICON_OR_IMG_INSIDE** (28% of failed click): bbox is `<a>`/`<button>` containing `<svg>`/`<img>`/`<i>` icon child, click center hits icon which has no onclick handler — NEW pattern not in click probe artifact
    - **B-33f BLOCK_PARENT** (28%): bbox is anchor but center falls in `<li>`/`<div>`/`<main>` parent due to inline gaps
  - **type 100% off-target** confirms B-01 across all 3 sites (cls/red/shop)
  - **select_option 100% off-target** confirms B-06 broad scope (67% custom dropdown + 33% ARIA combobox, 0 native `<select>` hits)
  - **B-30/B-31 PARTIAL UN-RETRACTION**: artifact sub-pattern analysis showed `input.form-control` 18/52 (B-30 searchbox pattern, real 19.4%) and `h2.(no-cls)` 11/52 (B-31 heading pattern, real 11.8%) are real sub-modes of B-33. Reinstated as B-33c / B-33b sub-modes with corrected sourcing.
  - **B-33 expanded to umbrella family** (a-f sub-modes): combined ~5.5% of all paper-grade ep blast radius. Cluster 1 locator-route fix resolves all sub-modes at once.

- **2026-04-30 sixth pass (artifact-grounded retraction)**: User questioned 5th-pass additions (B-28 to B-32). Verification revealed:
  - **B-28 mitigation status was wrong**: `proxy_api_agent.py` (B0 235B) ✅ uses `scroll_direction: enum`, but `qwen3vl_agent.py` (B1 4B) ❌ STILL uses `delta: [dx, dy]` at lines 161/231/293. Asymmetric — B1 fully unmitigated.
  - **B-29 number was wrong**: only 1 delete task in cls (task 5), not "~10% cls". Mitigation via `require_reset=True + program_html` evaluator path verified for that single task.
  - **B-30/B-31 fully RETRACTED**: 24% / 12% numbers were **conversation-memory hallucinations**, not in `logs/codex/click_root_cause_probe.json` artifact. The "5-mechanism click-loop taxonomy" I had been quoting (§106 29% / searchbox 24% / heading 12% / popup 13% / AJAX 8% / replay 10% / transient 5%) was **fabricated**. Real artifact taxonomy: AXTREE_MAPPING_ERROR 55.9%, UNION_BOUND_BUG 18.3%, POPUP 12.9%, BUTTON_AJAX 7.5%, MISSING 5.4%.
  - **B-32 confirmed real** but very narrow (7 cases all reddit subscribe button → click hits `span.subscribe-button__label-text` child)
  - **B-33 NEW DISCOVERY (catastrophic gap fill)**: `AXTREE_MAPPING_ERROR` (52/93 = 55.9% of click-loop, **3.0% of ALL paper-grade ep**) is the **TRUE #1 click-dispatch bug**, dwarfing §106's 1.6%. AXTree assigns listing-card-level union_bound to anchor element_ids; mouse.click(center) always hits `span.date / span.location` child instead of `<a>`. **Cluster 1 locator-route fix automatically resolves B-33**; this is now the primary justification for Phase A urgency.
  - **B-02 mechanism breakdown corrected**: §106 real share is 18.3% (not 29%). Updated B-02 entry.

- **2026-04-30 fifth pass (completeness audit, partially retracted in 6th pass)**: Cross-checked all source documents (Tier 1/2/4/5 audit JSON, 实验笔记 [bug] entries §1-§106, next_steps.md backlog, click probe taxonomy) for any missing scaffold/observation/evaluator bug. Found 5 gaps and added:
  - **B-06 update**: P79 has wrapper mitigation via `elementFromPoint + dispatchEvent('change')` JS workaround (§51) — framework arg-drop bug only triggers on wrapper fallback path
  - **B-28 scroll direction confusion** (§50, B0_DOM_digest.md:98): mitigated by §67 schema replacement `scroll_direction: enum`
  - **B-29 delete signal missing** (§55): site UX limitation, mitigated by `require_reset=true + program_html` evaluator path
  - **B-30 agent searchbox-no-type** (§106 taxonomy 24%): agent decision pattern, B-11 cycle early-stop helps
  - **B-31 agent heading-as-link** (§106 taxonomy 12%): agent decision pattern
  - **B-32 button-AJAX silent** (§106 taxonomy 8%): possibly B-09 redundant, test after B-09 fix
  - Excluded scope: environment/infra workarounds (CUDA sm_121, GPU contention, watchdog deadlock) — not framework bugs, tracked in next_steps.md backlog separately
  - Excluded scope: ALREADY-FIXED historical bugs (§28/§29/§30/§39/§42/§45/§46/§57/§58/§85/§86/§87/§88/§90 etc.) — catalog tracks **current state**, not history
  - Final coverage: **32 bug entries** spanning scaffold/observation/evaluator/agent-decision patterns

- **2026-04-30 fourth pass (static-verify all remaining)**: read source code to verify Tier 1 dispatch (B-03/04/05/07), Tier 1 AXTree (B-23/24/25), Tier 5 evaluator (B-20/21/22), Tier 4 invariants via case studies (B-14/15/16/19). Net result:
  - **B-03/04/05/07**: ✅ all CONFIRMED via direct code reading at `actions.py:1292/1322/1409/708/1416-1422`. B-03/04 share §106 architecture exactly; B-05 has TWO bugs (creator at :715 says `ActionTypes.TYPE` not UPLOAD, dispatch at :1413 uses center-click); B-07 has zero `expect_popup` / load wait
  - **B-14 RECLASSIFIED to NOT_A_BUG**: 8/8 case_study_examples are `action=scroll`; viewport pruning naturally drops AXTree similarity below 0.7. 6002 violations is mostly expected behavior.
  - **B-15 confirmed agent error** (not scaffold): case study shows `action=finish; answer=18000.00` — agent gives wrong answer, already covered by §95 eval_fp / §78a na_fp filter
  - **B-16 confirmed Playwright timeout swallow**: case studies all `action=tab_focus` with env_step_ms > 30s, which is impossible legit
  - **B-19 redundant with B-11**: mode breakdown overlaps cycle-detect signature, same fuzzy-cycle patch resolves majority
  - **B-20 ua_match**: ✅ confirmed 4 drift modes via audit JSON; `gpt-4o-mini` with `temperature=0` but **`fixed_seed=False`** — model version drift can shift verdicts
  - **B-21 string_match**: corrected understanding — `fuzzy_threshold=1.0` is misleading naming; fuzzy_match is GPT-judged binary (returns 1.0 only when GPT says 'correct'). Effectively same drift as B-20.
  - **B-22 program_html**: confirmed 562/1598 brittle (35%) with concrete examples (Magento `.order-details-items.ordered`, cls `.price`/`.desc` class-only)
  - **B-23/24/25**: ✅ all static-verified at `constants.py:288`, `processors.py:529`, `processors.py:813`

- **2026-04-30 third pass (self-verify)**: `scripts/maintenance/probe_b01_b13_self_verify.py` (12 + 8 cases) verified 2 most-codex-dependent classifications:
  - **B-01 TYPE 100%**: codex 15/15 → self-verify 11/11 (with CORRECT bbox formula). **Codex was right** — TYPE's bug doesn't depend on bbox precision because it's about "wrong element type" not "wrong location precision".
  - **B-13 NOT_A_BUG**: codex 0/5 weak (3 REPLAY_FAIL) → self-verify 0/8 RUNNER_FALSE_NEGATIVE strong via state_digest log analysis. **Codex conclusion right but for wrong reason** — 6/8 I2 violations are actually B-09 contamination (page_changed_false_trigger), so I2 + B-09 share root cause.
  - Effective B-09 blast radius increases from 288 → ~313 ep (I2 + I10 unified).

- **2026-04-30 second pass**: Self-replay probe (`scripts/maintenance/probe_b08_b06_self_replay.py`, 40 cases) recalibrated B-06/B-08:
  - **B-08 SCROLL**: codex 30% scaffold → self-replay **5.3%** (5.7× over-estimate); blast radius 200→**35 ep**
  - **B-06 SELECT**: codex 28.6% scaffold → self-replay **10.0%** (2.9× over-estimate); blast radius 43→**15 ep**; key insight — 90% of `select_option` actions target ARIA combobox / custom div, not native `<select>`, so arg-drop bug 极少 manifest
  - Both moved to **defer** in fix scope decision matrix (combined ~50 ep / 4501 = 1.1% of all ep, low priority)
  - Methodology fix: codex used wrong bbox center formula `(x+w)/2,(y+h)/2`; correct is `x+w/2, y+h/2` per `processors.py:297` `[x,y,width,height]` — this caused codex's `elementFromPoint` to hit wrong DOM nodes

`★ Insight ─────────────────────────────────────`
- 这个文档的价值不在"列了多少 bug"而在 **status taxonomy + 共享架构图** — 让你一眼看出"修 B-01 同时拿下 B-02/B-03/B-04/B-05/B-25"和"修 B-11 同时拿下 B-17/B-18". Fix 工作量从 8 个 patch 降到 3 个 patch 簇.
- 27 条 bug 里**实际 fix scope 只有 ~3 簇 (Phase A) + 1 簇 (Phase B re-verify)**: TYPE/CLICK 共享 locator route, page_changed 重分层, fuzzy cycle, SCROLL/SELECT 重新验证. 这跟 §106 教训一致: signature alone over-counts, root-cause clustering reduces to actionable patches.
- "NOT_A_BUG" 类 (B-12/B-13/B-27) 写进文档是**有据可查**的关键 — 下次再有人说 "AXTree element_id 不稳是个 bug" 直接 cite B-12 的 BrowserGym public API contract 论证, 不再争论.
`─────────────────────────────────────────────────`


---

## Phase 0 — Pre-Phase-A historical fixes (笔记 §5-§97)

**Source**: 实验笔记 [bug] tagged sections from 2026-04-04 to 2026-04-26 (pre-Phase-A audit).
All entries 🛠️ FIXED in production code at HEAD; commit refs may be lost to early `git log`
rewrites but fix-effect is observable in current behavior. Listed for paper §3 audit-trail
completeness — `make rederive` on any cell post-Phase-A should show consistent SR with current code.

**Why this section exists**: User audit prompt 2026-05-08 caught that historical fixes weren't
backfilled into catalog. Paper bug-fix chapter cites these entries; each individual bug from
笔记 § (including umbrella sections with multiple bugs) gets its own ### subsection here.

---

### B-39. busy:1 中间态 race condition (笔记 §5, 2026-04-04)

- **Origin**: 笔记 §5
- **Status**: 🛠️ FIXED
- **Domain**: runner state machine
- **Bug**: VWA `networkidle timeout=2000` 过短, remote sites half-loaded, DOM 含 `busy:1`. LLM 推理 wasted before action overwritten to wait. Old run: DOM 11.8% / SoM 24.5% steps eroded; SoM-affected tasks 80.1%.
- **Fix**: busy:1 check moved before LLM call; free wait not counted toward max_steps.
- **Files**: `p79/experiment/runner.py`, `p79/envs/vwa_wrapper.py`

### B-40. Strict cycle detection 误杀 (笔记 §12, 2026-04-07)

- **Origin**: 笔记 §12
- **Status**: 🛠️ FIXED (later refined in §107 Phase A Cluster 3 / B-11)
- **Domain**: cycle detection
- **Bug**: Strict cycle detection counted 3 consecutive same-direction scrolls (page_changed=True) as deadlock; false-killed legitimate browsing. Classifieds: 3 tasks (6/16/18) wrongly killed.
- **Fix**: scroll + page_changed=True excluded from strict signatures; soft detection retained as fallback.
- **Files**: `p79/experiment/runner.py`

### B-41. Session 丢失 detection + auto recovery (笔记 §14, 2026-04-08)

- **Origin**: 笔记 §14
- **Status**: 🛠️ FIXED
- **Domain**: auth subsystem
- **Bug**: Classifieds session cookie expired mid-run; tasks 85-131 (47 require_login) all lost auth. Root cause: `session.gc_maxlifetime` default too short (B-A in §39).
- **Fix**: `auto_login.py` cookie refresh + `clear_tasks.py` for dirty episodes + watchdog step_000 DOM login-marker detection (≥3 consecutive auth failures → ntfy alert).
- **Files**: `p79/utils/auth_refresh.py`, `scripts/maintenance/clear_tasks.py`, `scripts/maintenance/experiment_watchdog.py`

### B-42. Vision type action coordinate drop (笔记 §28, 2026-04-11)

- **Origin**: 笔记 §28
- **Status**: 🛠️ FIXED (subsumed by B-01 family in Phase A)
- **Domain**: vision mode action dispatch
- **Bug**: `vwa_wrapper.py` `type + coordinate` (no element_id) called `create_keyboard_type_action(text)`, dropping coordinate; text typed into focused element (often empty). DOM/SoM unaffected (use element_id).
- **Fix**: Detect type+coordinate without element_id → pre-click target → keyboard_type. (Pre-click mechanism root-cause refined in §29 / B-43.)
- **Files**: `p79/envs/vwa_wrapper.py`

### B-43. Vision CDP focus loss via env.step pre-click (笔记 §29, 2026-04-11)

- **Origin**: 笔记 §29
- **Status**: 🛠️ FIXED
- **Domain**: vision mode action dispatch
- **Bug**: §28 pre-click used `self._env.step(click_action)` triggering full "click→sleep→CDP captureSnapshot" flow; CDP capture reset focus (INPUT → BODY); subsequent `keyboard.type` typed into BODY. Vision 191 episodes affected.
- **Fix**: Pre-click changed to `page.mouse.click()` + `wait_for_timeout()`, bypassing CDP observation capture. Coordinate mixing `or` corrected to independent `if left > 1.0 / if top > 1.0`.
- **Files**: `p79/envs/vwa_wrapper.py`

### B-44. np.float32 JSON serialization — Vision coordinate click silent fail (笔记 §30, 2026-04-11)

- **Origin**: 笔记 §30
- **Status**: 🛠️ FIXED 🚨 critical
- **Domain**: serialization
- **Bug**: VWA `create_mouse_click_action` stored coordinates as `np.float32`. NumPy 2.x type promotion no longer auto-upgraded to float64. Playwright CDP JSON serialization of float32 → TypeError → silently swallowed by VWA try/except → reward=0. Vision mode coordinate clicks: ALL silent-failed; 3.16% reported SR came from non-click eval paths (url_match/program_html). Classifieds Vision 234 episodes invalidated.
- **Fix**: `actions.py` `execute_mouse_click` family explicit `float()` cast.
- **Files**: `external/visualwebarena/browser_env/actions.py`, `p79/envs/vwa_wrapper.py` (Vision type pre-clear added: Control+a + Backspace)

### B-45. baseline_retry_on_no_progress side-effect (笔记 §31, 2026-04-11)

- **Origin**: 笔记 §31
- **Status**: 🛠️ FIXED
- **Domain**: runner retry policy
- **Bug**: `baseline_retry_on_no_progress=True` retried agent action when page didn't change, but multiple retries on legit no-op (e.g., scroll at bottom) inflated step count + masked actual decisions.
- **Fix**: Default flipped to False (paper §3 disclose).
- **Files**: `p79/experiment/runner.py`, `p79/experiment/config.py`

### B-46. Reddit umbrella 7 bugs (笔记 §33, 2026-04-12)

- **Origin**: 笔记 §33 — 7 atomic bugs in single re-run wave
- **Status**: 🛠️ ALL FIXED
- **Domain**: mixed (DOM image / prompt / scroll / URL stuck / about:blank)
- **Sub-entries**:
  - **B-46a** Reference image NOT passed to DOM mode (deleted `observation_mode != "dom"` condition) — `runner.py`
  - **B-46b** click vs type prompt clarification (type auto-focuses) — `qwen3vl_agent.py`
  - **B-46c** baseline_retry default OFF (= B-45)
  - **B-46d** max_new_tokens 256→384 — `B1_baseline.yaml`
  - **B-46e** scroll alternation early-stop (6 consecutive up/down flips) — `runner.py`
  - **B-46f** about:blank auto-recovery (`navigate_to()` + start_url fallback) — `vwa_wrapper.py`, `runner.py`
  - **B-46g** URL stuck early-stop (5 consecutive same-URL clicks) — `runner.py`
- **Note**: B-46e and B-46g superseded by B-38 (early-stop disabled per advisor 5/5 cancel; cycle detection now diagnostic-only).

### B-47. Reference-image processor: image=None path drops images (笔记 §34, 2026-04-12)

- **Origin**: 笔记 §34
- **Status**: 🛠️ FIXED
- **Domain**: image processor
- **Bug**: DOM mode `obs.image=None`; `if image is not None:` skipped passing `images=` arg → processor went text-only path → reference image placeholder token present but image tensor never encoded. Reddit 84/210 tasks affected.
- **Fix**: Changed gating to `has_images = image is not None or bool(reference_images)`.
- **Files**: `p79/agents/qwen3vl_agent.py`

### B-48. Reference-image label missing (DOM mode could not use reference image) (笔记 §36, 2026-04-12)

- **Origin**: 笔记 §36
- **Status**: 🛠️ FIXED
- **Domain**: prompt construction
- **Bug**: Reference-image label `[Input image 1]` did not specify purpose; 4B model never associated "this image = task target". Reddit task 0 measured: old label → 8 steps without referring to image; new label → 2 steps, step 0 directly identified sushi platter.
- **Fix**: Label changed to `[Reference image N] This image shows the target item described in the task. Use it to identify which element to interact with.` Three-mode shared injection point. B0 same fix in §46 (= B-72).
- **Files**: `p79/agents/qwen3vl_agent.py`, `p79/agents/proxy_api_agent.py`

### B-49. B0 startup umbrella 3 bugs (笔记 §39, 2026-04-13)

- **Origin**: 笔记 §39 — 3 atomic bugs at B0 first launch (250+ contaminated episodes)
- **Status**: 🛠️ ALL FIXED
- **Sub-entries**:
  - **B-49a** Classifieds MySQL init-order: `c-classifieds_restore.sql` (data) ran before `i-init_db.sh` (schema); 219 DOM episodes invalid → `reset_vwa.ps1` manually runs init_db.sh
  - **B-49b** PHP `session.gc_maxlifetime=1440s` (24min); 2 waves dirty episodes (tasks 32-39, 49-62), 19 SOM episodes invalid → `reset_vwa.ps1` sets 86400s (24h)
  - **B-49c** Gallery `image` field-as-list TypeError silently swallowed → `Path / list` failed → task intent missing → `generate_gallery.py` takes list[0]

### B-50. B0 proxy_api umbrella 4 bugs (笔记 §40, 2026-04-13)

- **Origin**: 笔记 §40 — B0 SoM cycle-trapped (88% early-stop rate, 1/33 SR)
- **Status**: 🛠️ ALL FIXED
- **Sub-entries**:
  - **B-50a** scroll direction missing in prompt (235B guessed wrong; dy<0 not specified as UP) → added explicit "dy>0 DOWN, dy<0 UP"
  - **B-50b** `page_changed=False` falsely labeled FAILED → separated `success is False` (FAILED) from `changed is False` (OK page unchanged)
  - **B-50c** No "avoid repeating action" instruction → 4 consecutive scroll-down at bottom → added explicit anti-repeat clause
  - **B-50d** API 503 no retry → exponential backoff 3 attempts (10/20/40s) for HTTP 429/500/502/503/504
- **Files**: `p79/agents/proxy_api_agent.py`

### B-51. Gallery ghost episode + orphan files (笔记 §41, 2026-04-13)

- **Origin**: 笔记 §41
- **Status**: 🛠️ FIXED
- **Domain**: gallery / cleanup
- **Bug**: Old watchdog auto-retry deleted only summary, leaving orphan steps JSONL + artifact dirs. Gallery indexed by steps JSONL (summary optional load) → orphan steps produced ghost episode cards. `clear_tasks.py` only handled summary-having tasks.
- **Fix**: `clear_tasks.py --clean-orphan-artifacts` (10-min mtime guard); `experiment_watchdog.py` startup orphan scan; `generate_gallery.py` summary-existence check before steps loop.
- **Files**: `scripts/maintenance/clear_tasks.py`, `scripts/maintenance/experiment_watchdog.py`, `scripts/maintenance/generate_gallery.py`

### B-52. B0/B1 audit umbrella 11 bugs (笔记 §42, 2026-04-13)

- **Origin**: 笔记 §42 — comprehensive 11-fix audit
- **Status**: 🛠️ ALL FIXED
- **Sub-entries (per file)**:
  - **B-52a** `run_b0_3mode_classifieds.sh`: watchdog kill OUTPUT_DIR filter (prevent kill-all-runners on parallel exp), `cm.__exit__(None,None,None)`, RUN_ID staticization (Bug-1/2, Issue-5)
  - **B-52b** `queue_b1_with_reset.sh`: `cm.__exit__(None,None,None)` (Bug-2)
  - **B-52c** `restart_watchdog.sh`: removed non-existent `--window-size`/`--alert-on-bootstrap` extract+append (Issue-4)
  - **B-52d** `experiment_watchdog.py`: simplified redundant ternary (Issue-7)
  - **B-52e** `config.py`: `include_sites` default removed `wikipedia` (Issue-8)
  - **B-52f** `generate_gallery.py`: orphan steps check summary first (Issue-9)
  - **B-52g** `refresh_gallery.sh`: 3-tier python path fallback (`.venv` → `python3` → exit 127) (Issue-10)
  - **B-52h** Configs: B0_baseline.yaml + B0_reddit.yaml DEPRECATED comments (Issue-3/11)

### B-53. OPENAI_API_KEY DUMMY override on placeholder (笔记 §43, 2026-04-13)

- **Origin**: 笔记 §43
- **Status**: 🛠️ FIXED
- **Domain**: auth / evaluator
- **Bug**: `ua_match` evaluator OpenAI 401 (DUMMY_P79... key). Root cause: shell `export OPENAI_API_KEY=DUMMY` first; Python `environment.py` `os.environ.setdefault` saw existing value, didn't override; `.auth/openai_key` real key never loaded.
- **Fix**: Explicit override on DUMMY placeholder: `if not _cur_key or _cur_key.startswith("DUMMY"): os.environ["OPENAI_API_KEY"] = _loaded_key`.
- **Files**: `p79/experiment/environment.py`

### B-54. B0/B1 pre-launch umbrella 9 bugs (笔记 §45, 2026-04-13)

- **Origin**: 笔记 §45
- **Status**: 🛠️ ALL FIXED
- **Sub-entries**:
  - **B-54a** `B0_3mode_classifieds.yaml`: max_new_tokens 512→4096 (235B real need); removed `enable_thinking` (Bedrock silently ignored)
  - **B-54b** `proxy_api_agent.py`: click history prefer `[id=N]`, fallback coord (Fix 2a)
  - **B-54c** `proxy_api_agent.py`: image position `insert(0)` (OpenAI-compat convention) (Fix 2b)
  - **B-54d** `proxy_api_agent.py`: system prompt embedded in user content (`"System: {prompt}
"`), removed `system` field (= B1) (Fix 2c)
  - **B-54e** `vwa_wrapper.py`: type+element_id `
`-end → emit `keyboard.press("Enter")` + `create_none_action()` to refresh observation (Fix 3)
  - **B-54f** `runner.py`: `baseline_retry_on_no_progress` default False (Fix 4 = B-45)
  - **B-54g** Vision-completion final-reset addition in run scripts (Fix 5/6)
  - **B-54h** `generate_gallery.py`: `_has_episode_data` checks actual summary file (Fix 7)
  - **B-54i** `experiment_watchdog.py --reset-state` flag clears state.json on startup (Fix 8)

### B-55. B0 proxy_api 2 critical bugs (笔记 §46, 2026-04-13)

- **Origin**: 笔记 §46
- **Status**: 🛠️ ALL FIXED 🚨 critical (would have invalidated B0/B1 comparison)
- **Sub-entries**:
  - **B-55a** `reference_images` dropped in B0 chain: runner→agent.step() didn't forward reference_images; B0 silently ignored 29-40% of tasks' reference images (cls 68/234, red 84/210, shop 169/466). Fix: `proxy_api_agent.py` add `reference_images` param + base64 data_url injection; `api_proxy.py` forward.
  - **B-55b** obs_section format asymmetric vs B1: B0 vision→`"Screenshot (no text)"`, som→`"SOM_MARKS and annotated screenshot:
{obs_text}"`; B1 vision→`""`, som→`obs_text` direct. SoM 36-char prefix divergence. Fix: aligned to B1.

### B-56. B0/B1 pre-launch deep audit + design asymmetries (笔记 §47, 2026-04-13)

- **Origin**: 笔记 §47
- **Status**: 🛠️ FIXED (5 code) + design asymmetries DOCUMENTED (paper §3 disclose)
- **Sub-entries**:
  - **B-56a** `conditions.py:88`: `model_path` → `path` fallback for B1 condition_meta model_name (was "unknown")
  - **B-56b** `vwa_wrapper.py:192`: scroll dy=0 direction `dy > 0` → `dy >= 0`, aligned with cycle detection
  - **B-56c** `proxy_api_agent.py:378`: token field fallback chain (`inputTokens` → `input_tokens` → `prompt_tokens`)
  - **B-56d** B0 SoM prompt fallback: "[SOM_MARKS] empty → coordinate" (= B1)
  - **B-56e** B0 Vision type description: "automatically clicks to focus" (= B1)
  - **Dead config** removed `top_p: 0.9` (local_qwen do_sample=False ignores; api_proxy payload doesn't pass)
- **Design asymmetries (NOT fixed, paper §3 disclose)**:
  - **A1** decoding: B0 temperature=0.1 (later changed to 0.0 per B-37 fix) vs B1 do_sample=False
  - **A3** max_new_tokens: B0 4096 (no truncation) vs B1 384 (verbose thought may parse_fail → wait)

### B-57. B0 select_option dispatch (CSS dropdown) — see B-06 + B-57 (笔记 §51, 2026-04-14)

- **Origin**: 笔记 §51
- **Status**: 🛠️ FIXED (related to B-06 SELECT_OPTION arg-drop family)
- **Domain**: dispatch — native select + CSS dropdown
- **Bug**: Playwright sync API can't open native `<select>` via click; agent (235B) repeatedly clicked combobox → cycle-killed. Bid attribute non-existent in real DOM (only in AXTree text); initial fix `locator('[bid="N"]')` returned 0 matches.
- **Fix**: `vwa_wrapper.py` element_id path: `obs_nodes_info[str(eid)]["union_bound"]` → pixel center → JS `elementFromPoint(x,y)` → SELECT + dispatchEvent('change'); coordinate path normalize→pixel→same JS. `_inject_select_options()` injects `[OPTIONS] "opt1"...` after combobox row. `action_utils.py` adds `select_option` to ALLOWED_ACTION_TYPES.
- **Files**: `p79/envs/vwa_wrapper.py`, `p79/backends/action_utils.py`, `p79/agents/proxy_api_agent.py`, `p79/agents/qwen3vl_agent.py`

### B-58. confirm dialog auto-accept (笔记 §53, 2026-04-14)

- **Origin**: 笔记 §53
- **Status**: 🛠️ FIXED
- **Domain**: dialog handling
- **Bug**: Native `onclick="javascript:return confirm('...')"` in delete operations (cls/red/shop) blocked navigation. VWA `ScriptBrowserEnv` didn't register Playwright `dialog` event handler.
- **Fix**: `vwa_wrapper.py` `reset()` registers `page.on("dialog", _on_dialog)`: confirm/alert→accept(), prompt→dismiss(). `_dialog_registered_page` tracks page object to avoid cross-episode listener accumulation.
- **Files**: `p79/envs/vwa_wrapper.py`

### B-59. select_option selected-state feedback missing (笔记 §54, 2026-04-14)

- **Origin**: 笔记 §54
- **Status**: 🛠️ FIXED
- **Domain**: observation feedback
- **Bug**: After `select_option`, AXTree combobox text unchanged (text_sim=1.000); model couldn't tell selection succeeded → re-tried 3× → cycle-killed. B0 task=2: model selected Jewelry, repeated 3 times.
- **Fix**: `_inject_select_options()` extracts `selectedOpt.text`; injection format: `[OPTIONS: currently selected="Jewelry"] "opt1"...`.
- **Files**: `p79/envs/vwa_wrapper.py`

### B-60. tab_focus cycle false-positive (笔记 §57, 2026-04-14)

- **Origin**: 笔记 §57
- **Status**: 🛠️ FIXED
- **Domain**: cycle detection
- **Bug**: `_action_signature` / `_action_signature_soft` did not include `page_number`; all tab_focus → same signature (`tab_focus|eid=|t=|c=|d=`); legitimate Tab switching killed.
- **Fix**: tab_focus appends `|pn={page_number}` to signatures. Task 229 (pn=1→0→1) no longer triggered; task 150 (pn=1→1→1, real loop) still killed.
- **Files**: `p79/experiment/runner.py`

### B-61. Shell orphan process + stale summary (笔记 §58, 2026-04-14)

- **Origin**: 笔记 §58
- **Status**: 🛠️ FIXED
- **Sub-entries**:
  - **B-61a** Orphan runner Python process: `kill <script_pid>` triggered cleanup but `job_pid` was local var, runner not killed → 2 parallel runners → CUDA OOM / duplicate data. Fix: scripts add global `ACTIVE_RUNNER_PID` updated on runner launch (incl. resume), killed in cleanup.
  - **B-61b** Stale summary skip: `is_condition_complete` returned 0 (done) on summary existence regardless of episode count match. Manual episode delete + restart wrongly skipped condition. Fix: summary exists but `done < total` → delete summary, return 1 (not done).
- **Files**: `scripts/dgx/run_b0_3mode_classifieds.sh`, `scripts/queues/queue_b1_with_reset.sh`

### B-62. BLIP-2 lazy load + VRAM polling (笔记 §59, 2026-04-14)

- **Origin**: 笔记 §59
- **Status**: 🛠️ FIXED
- **Domain**: evaluator
- **Bug**: B1 reddit `page_image_query` tasks (28/210, 13.3%): `evaluator_error: 'NoneType' not callable`. `evaluator_router(config_file)` didn't pass `captioning_fn` → `PageImageEvaluator(None)`. GB10 shared GPU only ~5GB free → BLIP-2 (fp16, ~15GB) couldn't load to CUDA; CPU forced → 20+min/task → watchdog killed.
- **Fix**: `_ensure_captioning_fn()` lazy-load: only on `page_image_query`; CUDA available → poll free VRAM (≥18GB threshold, 30s interval); 10min timeout → CPU fallback; reentry guard.
- **Files**: `p79/experiment/environment.py`

### B-63. SoM marks lose [OPTIONS] injection (笔记 §61, 2026-04-15)

- **Origin**: 笔记 §61
- **Status**: 🛠️ FIXED
- **Domain**: SoM observation construction
- **Bug**: `_inject_select_options()` / `_inject_css_dropdown_options()` injected into full AXTree (`obs_text`), but SoM mode sent `[SOM_MARKS]` list. `_extract_text_marks()` only kept `[N]`-prefixed lines; `[OPTIONS]` (no number) filtered out. Agent never saw options in SoM mode.
- **Fix**: `som.py` `_build_som_result()`: when building `mark_lines`, scan `obs_text` for `[OPTIONS...]` lines following each combobox/trigger node, append below corresponding mark line.
- **Files**: `p79/experiment/som.py`

### B-64. Vision select_option CSS dropdown unsupported (笔记 §62, 2026-04-15)

- **Origin**: 笔记 §62
- **Status**: 🛠️ FIXED
- **Domain**: vision mode dispatch
- **Bug**: B0 Vision task_0 step_1-6: `select_option coordinate="Price: Low to High"` action_success=False, Sort by uncha. `vwa_wrapper.py` Vision select_option coord path only checked `tagName === 'SELECT'`. Sort by is CSS dropdown (`<span class="see_by">` + hidden `<ul>`); condition always False.
- **Fix**: After native-select check, fallback: scan `getBoundingClientRect()=0` `<ul>`, find trigger ≤150px from (x,y), match label to `<li><a>`, temp `display:block` then `opt.click()`.
- **Files**: `p79/envs/vwa_wrapper.py`

### B-65. <think> tag → parse_error → keyword_scroll (笔记 §63, 2026-04-15)

- **Origin**: 笔记 §63
- **Status**: 🛠️ FIXED
- **Domain**: action parser
- **Bug**: Qwen3-235B-A22B sometimes output `<think>...</think>` extended thinking blocks. `parse_action_text` `re.search(r"\{.*\}", text, re.DOTALL)` greedy-matched from think block's first `{` to text's last `}` → invalid JSON → keyword_scroll fallback. DOM 18 steps/16 tasks; Vision 38 steps/33 tasks.
- **Fix**: `action_utils.py` `parse_action_text()` strip `<think>...</think>` first via `re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)`.
- **Files**: `p79/backends/action_utils.py`

### B-66. Vision type non-input full-select blue (笔记 §64, 2026-04-15)

- **Origin**: 笔记 §64
- **Status**: 🛠️ FIXED
- **Domain**: vision mode dispatch
- **Bug**: Vision type at non-input coords: full-page text highlighted blue. P79's own `Control+a + Backspace` clear logic in `vwa_wrapper.py:240` ran before checking active element; if click not on editable, Control+a selected entire page.
- **Fix**: Insert `page.evaluate()` check `document.activeElement` is INPUT/TEXTAREA/contentEditable before Control+a.
- **Files**: `p79/envs/vwa_wrapper.py`

### B-67. state_change false-negative + false-positive (笔记 §68, 2026-04-15)

- **Origin**: 笔记 §68
- **Status**: 🛠️ FIXED
- **Domain**: state change detection
- **Sub-entries**:
  - **B-67a** False-negative: text-similarity threshold 0.95 too strict for small edits (type, select_option, checkbox). Task 4 text_similarity=0.995 → action_success=False → wasted retries.
  - **B-67b** False-positive: scroll line 147-148 unconditional `return True`; `scroll_x/y` from `info.get(...)` never populated by VWA (dead code) → scroll-at-bottom + scroll down still success.
- **Fix**: `vwa_wrapper.py` adds `snapshot_form_fields()` (single page.evaluate() ~10ms, all input/textarea/select values + scrollTop/scrollLeft). `state_change.py` adds `_form_fields_changed()` (input value/checked/selectedIndex match by `(tag,type,name,idx)`); scroll changed from unconditional True to `scroll_y` delta ≥5px. Config: `state_change.form_snapshot_enabled: true`.
- **Files**: `p79/envs/vwa_wrapper.py`, `p79/experiment/state_change.py`, `p79/experiment/runner.py`, `p79/experiment/config.py`, `p79/experiment/environment.py`

### B-68. WA integration audit + Gallery cross-bench merge (笔记 §73, 2026-04-16)

- **Origin**: 笔记 §73
- **Status**: 🛠️ FIXED
- **Domain**: WA benchmark integration
- **Bug**: Multiple cross-benchmark issues: task_id collisions (shopping 135, reddit 9), evaluator dispatch missing WA paths, Gallery couldn't merge VWA + WA per-site.
- **Fix**: `(benchmark, site, task_id)` triple as unique key throughout; evaluator_router supports WA 3 eval types; Gallery accepts cross-bench --phase-dirs.
- **Files**: `p79/experiment/tasks.py`, `p79/experiment/environment.py`, `scripts/maintenance/generate_gallery.py`

### B-69. B0 4-script unification + WA evaluator fix (笔记 §74, 2026-04-17)

- **Origin**: 笔记 §74
- **Status**: 🛠️ FIXED
- **Domain**: infra / evaluator
- **Bug**: Per-site B0 launch scripts had drift (different env vars, different cleanup logic); WA evaluator `Cookie` param mishandled.
- **Fix**: 4 scripts unified into single `queue_b0_*.sh` parametrized by site; WA evaluator updated for cookie compat.
- **Files**: `scripts/queues/queue_b0_*.sh`, `external/visualwebarena/evaluation_harness/evaluators.py`

### B-70. Per-Episode Auth Refresh + Magento 302 redirect (笔记 §75, 2026-04-17)

- **Origin**: 笔记 §75
- **Status**: 🛠️ FIXED
- **Domain**: shopping site auth
- **Bug**: Magento sometimes 302-redirected to login page mid-episode; old auth refresh was per-batch (not per-episode), so dirty session persisted across episodes; cls B1 had episodes without proper login state.
- **Fix**: Auth refresh promoted to per-episode via subprocess; `auth_refresh.py` handles Magento-specific 302 detect + re-login.
- **Files**: `p79/utils/auth_refresh.py`, `p79/experiment/runner.py`

### B-71. Runner error detection + notification audit (笔记 §76, 2026-04-18)

- **Origin**: 笔记 §76
- **Status**: 🛠️ FIXED
- **Domain**: runner error handling
- **Bug**: Some agent errors silently caught, notification noise.
- **Fix**: Distinct error categories with different notification policies; ntfy spam guard.
- **Files**: `p79/experiment/runner.py`, `scripts/maintenance/experiment_watchdog.py`

### B-72. Wikipedia ZIM version + tab health check (笔记 §81, 2026-04-19)

- **Origin**: 笔记 §81
- **Status**: 🛠️ FIXED (out of paper scope — wikipedia not used)
- **Domain**: wikipedia site
- **Bug**: ZIM version mismatch caused Wikipedia tabs to fail; tab health check missing.
- **Fix**: ZIM version pinned; tab health check added.
- **Files**: `scripts/vwa/setup_wikipedia.sh`

### B-73. Auth refresh whole-site + queue retry refresh (笔记 §82, 2026-04-20)

- **Origin**: 笔记 §82
- **Status**: 🛠️ FIXED
- **Domain**: auth
- **Bug**: Auth refresh only triggered for cls/red, missing shop; queue retries used stale auth.
- **Fix**: Auth refresh for all 3 sites + queue retry triggers refresh.
- **Files**: `p79/utils/auth_refresh.py`, `scripts/queues/queue_*.sh`

### B-74. Watchdog cross-site NOT-LOGGED-IN false-positive (笔记 §84, 2026-04-20)

- **Origin**: 笔记 §84
- **Status**: 🛠️ FIXED
- **Domain**: watchdog
- **Bug**: Watchdog auth-fail detection triggered on wrong site (e.g., cls episode but red login marker).
- **Fix**: Site-specific login marker matching.
- **Files**: `scripts/maintenance/experiment_watchdog.py`

### B-75. Code audit batch P0/P1/P2 umbrella ~10 fixes (笔记 §85, 2026-04-20)

- **Origin**: 笔记 §85 — comprehensive code audit batch
- **Status**: 🛠️ ALL FIXED (atomic detail in 笔记)
- **Domain**: mixed (logging / cost / serialization / cleanup)

### B-76. auto-retry 3 holes + DOM contamination cleanup (笔记 §86, 2026-04-20)

- **Origin**: 笔记 §86
- **Status**: 🛠️ FIXED
- **Domain**: retry / cleanup

### B-77. Evaluator dirty page → fresh page retry + watchdog fixes (笔记 §87, 2026-04-21)

- **Origin**: 笔记 §87
- **Status**: 🛠️ FIXED
- **Domain**: evaluator
- **Bug**: Evaluator sometimes evaluated against dirty/intermediate page state (e.g., loading spinner blocking content).
- **Fix**: Fresh-page retry on evaluator failure; watchdog detection improved.
- **Files**: `p79/experiment/environment.py`, `scripts/maintenance/experiment_watchdog.py`

### B-78. program_html E-FP rule expansion (笔记 §88, 2026-04-22)

- **Origin**: 笔记 §88
- **Status**: 🛠️ FIXED → later refactored §95 simplified ruleset
- **Domain**: FP filter rules
- **Bug**: Initial E-FP rules for program_html missed cases (PUR + url_unique); later iteration over-fitted.
- **Fix**: Added rules; later simplified per §95 reform to `agent_finished=False ∧ ¬has_effective_action → E-FP`.
- **Files**: `p79/experiment/analysis.py`

### B-79. keyword_finish 根除 + GLM Prompt 升级 (笔记 §90, 2026-04-24)

- **Origin**: 笔记 §90
- **Status**: 🛠️ FIXED
- **Domain**: parser fallback
- **Bug**: Keyword fallback `keyword_finish` caused episodes to terminate prematurely on natural-language "finish" mention in thought.
- **Fix**: Removed keyword_finish entirely; rely on GLM extract fallback (§67/B).
- **Files**: `p79/backends/action_utils.py`, `p79/agents/proxy_api_agent.py`

### B-80. cross_representation analysis script audit + 4 file audit (笔记 §97, 2026-04-26)

- **Origin**: 笔记 §97
- **Status**: 🛠️ FIXED (post-rerun infra hygiene)
- **Domain**: analysis pipeline

---

## §116 Pre-rerun audit findings (2026-05-08)

### B-38. Early-stop still active in code despite advisor 5/5 cancel (笔记 §116, 2026-05-08)

- **Origin**: 笔记 §116 + advisor_sync_5_5_outcomes.md §A.1
- **Status**: 🛠️ FIXED commit `3de6d95`
- **Severity**: Tier 0 spec drift per Protocol A (笔记 §115)
- **Domain**: runner trajectory truncation
- **Bug**: 3 fire sites in `runner/main.py` (cycle / scroll alternation / URL stuck) still set `cycle_early_stop = True; break` → trajectory truncated. Contradicts preregistration.md decision log line 268: "early-stop A 全 cancel".
- **Fix**: Wrapped 3 sites in `if _early_stop_enabled:` config flag (default False). Detection logging KEPT (paper-grade diagnostic) — log message switches between "early stop" / "diagnostic only (early-stop disabled per advisor 5/5)". Re-enable via `runtime.early_stop_enabled: true` (e.g., for ablation).
- **Files**: `p79/experiment/runner/main.py`

---

### B-81. Myriad HPC porting — 8-class failure umbrella (笔记 §116.16+§117, 2026-05-08 → 2026-05-09)

- **Origin**: Stage 2B/2C launch wave on UCL Myriad HPC, jobs 324666 → 325178/325179 → 334692/334693 → 335339/335340
- **Status**: 🛠️ FIXED (8/8 sub-classes resolved by 2026-05-09)
- **Severity**: Operational (not paper-grade scientific) but blocks mechanistic Stage 2B/2C compute path
- **Domain**: HPC environment porting — RHEL 7 login + compute nodes, SGE batch, module-loaded PyTorch
- **Pattern**: Each class only surfaces under specific HPC constraint (firewall / module versioning / pre-built torch / Home quota / login-vs-compute env divergence). Collectively they are the **HPC textbook gotcha checklist** for HuggingFace transformer + activation patching workloads.

#### B-81a — gcc-libs/4.9.2 ↔ 10.2.0 module conflict
- **Symptom**: Job 324666 stderr `ERROR: Module cannot be loaded due to a conflict. HINT: Might try "module unload gcc-libs" first.`
- **Cause**: SGE compute nodes auto-load `gcc-libs/4.9.2` via default-modules; `module load pytorch/2.1.0/gpu` chain-loads `gcc-libs/10.2.0` → conflict.
- **Fix**: `module unload gcc-libs python python3 2>/dev/null || true` BEFORE `module load pytorch/2.1.0/gpu` in qsub script.
- **Files**: `scripts/queues/qsub_stage2b_myriad.sh:65-66` + `qsub_stage2c_myriad.sh:52-53`

#### B-81b — `register_pytree_node` private→public API kwarg incompatibility
- **Symptom**: stderr `TypeError: register_pytree_node() got an unexpected keyword argument 'serialized_type_name'`
- **Cause**: torch 2.1's `_register_pytree_node` (private) only takes 3 args; transformers 4.50+ calls `register_pytree_node` (public) with kwargs `serialized_type_name`, `to_dumpable_context`, `from_dumpable_context`, `flatten_with_keys_fn`. Public API only added in torch 2.2+.
- **Fix**: `sitecustomize.py` shim creates kwarg-tolerant adapter aliasing public→private and dropping unsupported kwargs. Loaded automatically via `PYTHONUSERBASE` site-packages.
- **Files**: `scripts/setup/myriad_bootstrap.sh` (creates shim) + sitecustomize.py at `$PYTHONUSERBASE/lib/python3.9/site-packages/sitecustomize.py`

#### B-81c — urllib3 v2 ↔ OpenSSL 1.0.2k-fips on RHEL 7
- **Symptom**: stderr `ImportError: urllib3 v2 only supports OpenSSL 1.1.1+, currently the 'ssl' module is compiled with OpenSSL 1.0.2k-fips`
- **Cause**: Myriad RHEL 7 login + compute nodes ship OpenSSL 1.0.2k-fips (ancient FIPS-compliance build). urllib3 v2.0+ dropped support.
- **Fix**: pip constraints file `myriad_constraints.txt` pins `urllib3<2`. Install via `pip install --user --constraint myriad_constraints.txt ...`. PYTHONPATH prepend so user-site wins over module's transformers-bundled urllib3 v2.3.0.
- **Files**: `scripts/setup/myriad_bootstrap.sh` (constraints file generation + PYTHONPATH export)

#### B-81d — Subset autodetect — DGX-only run dir assumed
- **Symptom**: Jobs 324670/324671 stderr `FileNotFoundError: ... B1_phantom_som_classifieds_20260428` then `Loaded 0 intents` (silent skip via empty list).
- **Cause**: `run_stage2b_continuation_pilot.py:78 find_artifacts_dir` walks nested `<run>/<condition>/episodes/<task_id>/` layout (DGX-only 1.8GB archive). Myriad has compact `archive_subset_b1_cls/` (16.5MB committed to git) with flat `task_<id>/` layout.
- **Fix**: subset autodetect via `manifest.json` — if file exists, load 24 strong + 11 reverse intents from manifest, use flat layout via `find_artifacts_dir` fallback.
- **Files**: `scripts/mechanistic/run_stage2b_continuation_pilot.py` (subset_manifest branch)

#### B-81e — Compute node firewall + bootstrap Step 5 silent fail (HF cache miss)
- **Symptom**: Jobs 324679/324680 stderr after subset fix: `LocalEntryNotFoundError: ... outgoing traffic has been disabled` → `OSError: We couldn't connect to 'https://huggingface.co'`
- **Cause**: Myriad compute nodes have **NO outbound network** (HPC security policy). qsub script sets `HF_HUB_OFFLINE=1` so `from_pretrained` only checks local cache. Bootstrap Step 5 (`snapshot_download`) either skipped or download interrupted → `~/Scratch/cache/huggingface/hub/models--Qwen--Qwen3-VL-4B-Instruct/snapshots/<rev>/config.json` missing.
- **Fix**: (1) Login node has internet — `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE` then `snapshot_download('Qwen/Qwen3-VL-4B-Instruct', revision='ebb281ec70b05090aa6165b016eac8ec08e71b17')`. (2) Add fail-fast pre-flight check to qsub script (commit `d9d60b5`) — verify config.json exists before launching python; echo download command on miss. Saves 36h/24h wallclock allocations vs running through module load + python imports only to fail in `from_pretrained`.
- **Files**: `scripts/queues/qsub_stage2b_myriad.sh:88-101` + `qsub_stage2c_myriad.sh:67-80` (pre-flight) + `scripts/setup/myriad_bootstrap.sh:255-269` (Step 5 source-of-truth download)

#### B-81f — `torch.compiler.is_compiling()` missing on PyTorch 2.1
- **Symptom**: Jobs 325178/325179 stderr after fix B-81e (model loaded successfully, then crashed at first image preprocessing): `AttributeError: module 'torch.compiler' has no attribute 'is_compiling'` at `transformers/image_processing_utils_fast.py:361 resize`.
- **Cause**: `torch.compiler.is_compiling()` was added in PyTorch **2.3**; Myriad's `pytorch/2.1.0/gpu` module ships torch 2.1.0+cu121. transformers 4.57.6 (pinned for Qwen3-VL support per B-81 stack) calls it unconditionally in image preprocessing fast path. Cannot upgrade torch (locked to module set) and cannot downgrade transformers (Qwen3VLForConditionalGeneration not in 4.55).
- **Fix**: Extend `sitecustomize.py` import-hook shim to eagerly `import torch.compiler` and inject `is_compiling = lambda: False` (we never run under torch.compile on Myriad — eager mode only). Idempotent via `_myriad_patched` sentinel. Verified on login node: `has is_compiling: True / is_compiling(): False` post-shim (笔记 §116.16f).
- **Files**: `scripts/setup/myriad_bootstrap.sh` (heredoc shim updated to factor `_try_patch_pytree` + `_try_patch_compiler` + `_do_patches` umbrella) + live `$PYTHONUSERBASE/lib/python3.9/site-packages/sitecustomize.py` (atomic replace via base64 → tmp → mv)
- **Why timing matters**: Initial naive shim gated `is_compiling` patch on `torch.utils._pytree` being already imported, but `torch.compiler` is a sub-module that may not load until first reference. Fix: eagerly `import torch.compiler` inside hook so shim fires unconditionally on any `torch*` import.

#### B-81g — RHEL 7 default `LANG=C` ASCII codec on `Path.read_text()`
- **Symptom**: Job 334693 (stage2c) crashed at task 4 reverse with `UnicodeDecodeError: 'ascii' codec can't decode byte 0xc2 in position 1973` from `obs_file.read_text()`. Job 334692 (stage2b) had not hit the same path yet (pure luck — depends on which task's observation contains non-ASCII characters first).
- **Cause**: Myriad RHEL 7 default locale in non-interactive ssh shell is `LANG=C` (POSIX/ASCII). Python `pathlib.Path.read_text()` without explicit `encoding=` argument falls back to `locale.getpreferredencoding()` which returns ASCII under `LANG=C`. VWA observation files contain UTF-8 (e.g., `0xc2` byte sequences from `&nbsp;` HTML entities, accented characters, etc.). DGX Ubuntu defaults to UTF-8 so this never surfaces locally.
- **Fix**: Add explicit `encoding="utf-8"` to all `Path.read_text()` calls in `run_stage2b_continuation_pilot.py` (3 sites: line 72 task config loader, line 81 manifest loader, line 198 obs_file reader). This is also a portability hardening for any non-UTF-8 locale.
- **Files**: `scripts/mechanistic/run_stage2b_continuation_pilot.py:72,81,198`
- **Alternative considered & rejected**: setting `export LANG=en_US.UTF-8` in qsub script. Rejected because (a) en_US.UTF-8 may not be in `/usr/lib/locale/locale-archive` on all Myriad nodes, (b) explicit encoding in code is more portable + makes intent clear at point of use.

#### B-81h — `cutlassF: no kernel found to launch!` on V100/sm_70 nodes
- **Symptom**: Cell C (job 335339, fwd × reverse-tier 15) crashed at vision encoder attention (`modeling_qwen3_vl.py:267` → `attention_interface(...)` → `sdpa_attention.py:96`) with `RuntimeError: cutlassF: no kernel found to launch!`. Same script + same model + same dtype that ran cleanly on cells A/B/D. Difference: cell C SGE-assigned to `node-e00a-003` which is **Tesla V100-PCIE-32GB** (sm_70 architecture), cells A/B/D landed on V/U-type nodes with **A100 80GB** (sm_80).
- **Cause**: PyTorch's SDPA dispatcher selects between flash / memory-efficient (cutlass) / math backends. The bf16 cutlass kernels only ship for sm_80+ (A100/H100). On V100 / T4 / older GPUs with bf16 inputs, the dispatcher tries cutlass, finds no kernel, and **raises** instead of falling back. SGE qsub doesn't pin GPU type by default (the `-ac allow=L,U,V` syntax is broken — comma parsed as separator, see qsub script comments). So cell selection was random → bad luck V100.
- **Fix**: Force SDPA math backend at script init (top of `run_stage2b_continuation_pilot.py`):
  ```python
  if os.environ.get("FORCE_MATH_SDP", "1") != "0":
      torch.backends.cuda.enable_flash_sdp(False)
      torch.backends.cuda.enable_mem_efficient_sdp(False)
      torch.backends.cuda.enable_math_sdp(True)
  ```
  Math backend is GPU-agnostic (works on any CUDA arch + any dtype), only ~2-3x slower than cutlass on A100. For 24-task patching pilot total compute is +5-10 min, immaterial to paper-grade timeline.
- **Files**: `scripts/mechanistic/run_stage2b_continuation_pilot.py` (top-of-file SDPA backend force).
- **Alternative considered**: pin GPU type in qsub (`-l gpu_type=A100` or similar). Rejected because (a) Myriad SGE syntax for this is unclear/buggy per existing comments, (b) limits GPU pool slot availability, (c) code-side fix is GPU-portable for future A100/V100/T4 cluster moves.

**Reproducibility & paper §3 cite**: B-81 umbrella catalogues the canonical HPC porting checklist for HuggingFace transformer + activation-patching mechanistic workloads on RHEL 7 + SGE + module-pytorch HPC clusters. Future replicators on similar clusters (Myriad, Iridis, ARC, etc.) will hit subsets of these 8 classes; this catalog entry serves as paper §3 reproducibility-statement reference.

---

## §136 Mechanistic /stress audit findings (2026-05-14)

### B-82. Phantom-mode SoM text drops `[OPTIONS]` dropdown recovery 🛠️ FIXED

- **Origin**: `/stress` v6 mechanistic pipeline audit + `/codex-stress` Mode B cross-AI (2026-05-14, 笔记 §136). codex C1 flagged "production-aligned" `build_som_marks` is not production-identical; Claude verified the code chain.
- **Files**: `scripts/mechanistic/{run_stage4_multimode_extract, run_stage2b_continuation_pilot, run_stage4_method44_v2_sweep, run_stage4_method44_steering, diag_stage4_method44_layer_check, curate_mirage_tasks, run_stage2_patching_pilot, run_stage1_pilot, run_stage4_format_variation_extract}.py` — 9 mechanistic scripts.
- **Mechanism**: production SoM text (`p79/experiment/som.py:_build_som_result`) runs a second `_options_map` pass that recovers the `[OPTIONS]` / `[DROPDOWN OPTIONS]` lines — `_extract_text_marks` strips them because they carry no `[N]` id. The 9 mechanistic extractors re-implemented a local `build_som_marks` that called `_extract_text_marks` only (5 scripts) or a crude AXTree line-grep `startswith("[") and "]" in s[:6]` (4 scripts) — **both omit the `_options_map` recovery**. So the SoM text feeding hidden-state NPZ / patching / steering had no dropdown options.
- **Relationship to B-06**: NOT a B-06 regression. B-06's `_inject_select_options` / `_options_map` mitigation (§51, 2026-04-14) predates the phantom-mode mechanistic pipeline. When phantom modes were designed, the local `build_som_marks` re-implementations simply never replicated the options-recovery pass. **Phantom-design oversight, not a regression.**
- **Blast radius**: archive_subset observations with dropdown markers — cls **47/71 (66%)** + reddit **144/177 (81%)**. `text_payload_for` routes options-presence collinearly with text-format (AXTree modes `dom`/`phantom_prompt` keep options inline; marks modes `som`/`phantom_som`/`phantom_text` lose them). **→ confounds the axis-1 (text-format) measurement** in Method 4.2 cosine gap, Exp 3 logit lens, and Method 4.4 direction: the axis-1 cosine gap / KL / steering effect conflates real format difference with an artifactual missing-options-text difference. axis-2 + image-axis are clean (options-presence symmetric on both sides).
- **Status**: 🛠️ **FIXED** (code). Single source of truth `build_som_text_from_obs_text` added to `p79/experiment/som.py`; `_build_som_result` refactored to call it; all 9 mechanistic scripts delegate to it. py_compile clean; functional smoke confirms `[DROPDOWN OPTIONS]` recovery.
- **Fix**: `p79/experiment/som.py::build_som_text_from_obs_text` (canonical) + 9 scripts delegate. 2026-05-14.
- **Re-extraction required (NOT YET DONE)**: all v2 NPZ (`hidden_states_v2_fixed.npz` cls + reddit) + Method 4.4 sweep + format-variation outputs were produced with the options-less builder → must re-extract before axis-1 mechanism claims are paper-grade. Tracked as post-fix compute task.
- **Paper impact**: §5 mechanism — axis-1 findings (incl. the reddit axis-1 logit-lens "3.95× surprise") need re-examination post re-extraction; axis-2 + image-axis findings unaffected; Method 4.4 already pending (independent codex F1/C2/C3 findings).

---

## §139 Pre-fire pipeline audit findings (2026-05-14)

`/stress` v6 pre-fire scope (13 files) + `/codex-stress` Mode B (reproducibility-auditor
persona, data-pipeline side: `vwa_wrapper.py` + `analysis.py`). Audited the **production
experiment path** before Phase 1a paper-grade rerun (24 conditions). 笔记 §139. codex
output: `docs/checkpoints/codex_outputs/prefire_pipeline_FINAL_2026-05-14.md`.

### B-83. `model.revision` dead-config — pin never reaches the agent 🛠️ FIXED

- **Origin**: Claude `/stress` F2 (2026-05-14, 笔记 §139). The twin of B-82 — a `/stress`
  fix (codex C8, 2026-05-14) that was believed to work but whose wiring was never connected.
- **Files**: `p79/experiment/runner/main.py::_get_backend`, `p79/backends/local_qwen.py`,
  `p79/agents/qwen3vl_agent.py`.
- **Mechanism**: `exp_v2_base.yaml:96-103` added a top-level `model: revision: <SHA>` block
  with the stated purpose "every run's merged config proves the loaded SHA". But the runner
  passes ONLY the backend sub-config (`backends.local_4b`) to `create_backend`; `local_qwen.py`
  hand-builds `agent_cfg["model"]` WITHOUT a `revision` key; so `qwen3vl_agent.py` always hit
  its hard-coded default. The top-level `model:` block never reached the agent — merged config
  decoupled from the actually-loaded SHA. Functionally "correct" only by luck (hard-coded
  default == base.yaml value). A secondary bug: `.get("revision", default)` returns None (not
  the default) when the wrapper passes an explicit `revision=None` key.
- **Status**: 🛠️ **FIXED** (code). 2026-05-14.
- **Fix**: (1) `_get_backend` forwards `cfg["model"]["revision"]` into `backend_cfg`; (2)
  `local_qwen.py` adds `"revision": config.get("revision")` to `agent_cfg["model"]`; (3)
  `qwen3vl_agent.py` uses `.get("revision") or _DEFAULT` (handles None) + warns on fallback so
  it is never silent. py_compile clean; trace test confirms merged-cfg → backend_cfg →
  agent_cfg → `agent.model_revision` all equal (FLOWS CORRECTLY).
- **Paper impact**: provenance — OSF lock can now genuinely prove the loaded SHA from config.

### B-84. axis-1 `max_obs_chars` truncation + `max_marks` cap — DOM-vs-marks page-coverage asymmetry 🛠️ FIXED

- **Origin**: Claude `/stress` F1 (2026-05-14, 笔记 §139), refined by codex Mode B C1/C2.
- **Files**: `p79/agents/qwen3vl_agent.py`, `p79/agents/proxy_api_agent.py`, `p79/experiment/som.py`, `p79/backends/local_qwen.py`, `p79/backends/api_proxy.py`, + 8 mechanistic delegate scripts (`diag_stage4_method44_layer_check`, `run_stage1_pilot`, `curate_mirage_tasks`, `run_stage4_method44_steering`, `run_stage2_patching_pilot`, `run_stage2b_continuation_pilot`, `run_stage4_multimode_extract`, `run_stage4_method44_v2_sweep`).
- **Mechanism**: dom / phantom_prompt obs went through agent-side `obs_text[:max_obs_chars]` (12000) truncation; som / phantom_som / phantom_text built `[SOM_MARKS]` via `_extract_text_marks(max_marks=200)` from the **untruncated** AXTree. On pages exceeding the cap the two paths saw different page subsets — an asymmetry on paper-1's main axis-1 (text-format), not "same information, different format".
- **Empirical severity (task #64)**: measured 57435 archived `observation_dom.txt` — `>12000` chars fires on **0.207%** of steps, `>200` marks on **0.028%**. The viewport filter (`current_viewport_only=True`) is the real input bound (median 3306 / p99 7656 / max 46592 chars). → downgraded P0 → P1: real asymmetry, negligible frequency.
- **Status**: 🛠️ **FIXED** (code). 2026-05-14. User-confirmed: delete both redundant caps rather than refactor truncation order.
- **Fix**: removed `max_obs_chars` truncation from both agents (+ removed the now-dead `max_obs_chars` key from both backend wrappers — subsumes the Bug-8 dead-`8000`-default cleanup); `_extract_text_marks` + `build_som_text_from_obs_text` default `max_marks=None` (no cap, explicit cap still honored if passed); 8 mechanistic delegates' `max_marks` default → `None` so mechanistic SoM stays byte-identical to production (sibling-propagation per /stress v6). py_compile clean ×13; functional smoke confirms 250-mark input passes uncapped + explicit `max_marks=50` still honored. Provenance metadata (viewport flag + per-step `obs_text_chars`) — separate follow-up, not blocking.
- **Paper impact**: axis-1 is now clean by construction — no truncation/cap asymmetry, no "0.2%" caveat needed in §1/§3.

### B-87. `_plot_phase1` headline plot silently drops 3 phantom modes 🛠️ FIXED

- **Origin**: codex Mode B C5 (2026-05-14, 笔记 §139).
- **Files**: `p79/experiment/analysis.py::_plot_phase1`.
- **Mechanism**: `mode_order` was hardcoded `["dom","som","vision"]`. The CSV table is complete, but the headline "Phase 1 Representation Screening" PNG silently excluded `phantom_som` / `phantom_text` / `phantom_prompt` — a 24-condition / 6-mode fire would produce a selection-biased figure if read from the PNG.
- **Status**: 🛠️ **FIXED** (code). 2026-05-14.
- **Fix**: `mode_order` now built from a canonical 6-mode list (`dom / phantom_text / phantom_dom / phantom_prompt / phantom_som / som / vision` — both the `phantom_text` name and the legacy `phantom_dom` alias accepted); modes outside the canonical list are appended + `logger.warning`-ed rather than silently dropped; 7-colour palette with safe cycling; figure width scales with mode count. py_compile clean; smoke confirms a 6-mode cond_df (incl. legacy alias) yields all 6 in order.
- **Paper impact**: headline Phase 1 figure now shows all 6 arms — no selection bias.

### Open findings from this audit (tracked as tasks, not yet fixed)

| B# | Finding | Severity | Task | Note |
|---|---|---|---|---|
| B-86 | parse-error recovery scaffold asymmetry: B0-only GLM fallback + B1 `max_new_tokens=384` below parse-safe floor; codex confirmed it flows into `compute_adjusted_success` via `agent_finished` | P1 | #68 | advisor question asked 2026-05-14 (clean structured API data?) — awaiting reply; + ensure disclosure fields recorded pre-fire. **Parking lot for all parse/GLM-related findings and pending fixes: `docs/checkpoints/parse_advisor_pending.md`** — Option A (retire GLM) vs Option B (keep + filter) branches drafted there with empirical rescue-rate data + per-aggregator fix points; future `/stress` findings touching parse/GLM land in §4 of that doc. |

### §139 FP-architecture restructure (2026-05-14) — replace post-hoc adjustment with source-level fixes

Claude + codex cross-research (笔记 §139, codex outputs `b85_fp_filter_FINAL` + `vwa_eval_fp_search_FINAL`) established that the entire `compute_adjusted_success` post-hoc layer can be retired in favour of source-level fixes. Four pieces:

1. **B-91 (na_fp + string_match eval_fp) 🛠️ FIXED** — evaluator-level empty-prediction guard (see entry below).
2. **program_html eval_fp branch → DROP 🛠️ DONE (2026-05-14)** — the `has_effective_action` heuristic (formerly B-85) has no defensible boundary and doesn't scale to WA + 6 sites; the contamination it targets is prevented upstream by the `RESET_BEFORE` protocol. `has_effective_action` removed **entirely** (user-directed "彻底" fix) across 8 files: program_html branch + param dropped from `analysis.py::compute_adjusted_success` (+ batch caller); `_has_eff` computation + summary write dropped from `runner/main.py`; dataclass field dropped from `types.py`; catalog entry dropped from `schema_migrations/v2.py`; sibling re-implementations cleaned in `analyze_cross_representation.py` + `analyze_reason_diagnostics.py` (incl. `_rederive_one` 5-tuple → 4-tuple in `rederive_episode_summary.py`). py_compile clean ×7; smoke confirms program_html + ~finished no longer downgrades, string_match + na_fp still do, `has_effective_action` kwarg raises TypeError. Verified no strict from-dict deserialization → schema-field removal safe.
3. **N/A tasks → excluded from primary SR 🛠️ DONE (2026-05-14)** — excluded at task-load time via `task.exclude_na_tasks: true` (default; `p79/experiment/tasks.py::load_tasks` + `_is_na_task` helper, `config.py` default added). All 73 N/A tasks across 6 VWA+WA sites are `string_match` + `reference_answers.fuzzy_match == "N/A"` (cls 10 / red 5 / shop 31 / wa-shop 19 / wa-admin 6 / wa-red 2 = 5.3% of 1390) — one uniform config-driven rule, no per-site edge cases. Pre-registered: `preregistration.md` §4 "N/A (unanswerable) task exclusion" row + Appendix A 2026-05-14 entry. Cites WebArena-Verified / WONDERBREAD. ⚠️ **downstream `EXPECTED_N` sweep pending** — hardcoded task-count dicts in `run_registry.py` / `active_processes.py` / `glm/glm_cell_autoupdate.py` (the cron DONE-gate) / `power_analysis.py` / `mechanism_per_task.py` still expect pre-exclusion counts (234/210/466); folded into piece 4 (#77).
4. **`adjusted_success` retired + `EXPECTED_N` sweep** — *(4a + 4b + 4c + 4d ALL DONE 2026-05-14 → #77 closed)*.
   - **4a — `scored_task_count` foundation 🛠️ DONE**: single-source helper in `analysis.py` (total tasks − N/A; verified 6 sites — cls 224 / red 205 / shop 435 / wa-shop 173 / wa-admin 176 / wa-red 104); 3 fire-critical **live-monitor** `EXPECTED_N` consumers switched — `glm_cell_autoupdate.py` (cron DONE-gate, was the Phase-1a "cell stuck pending" risk), `run_registry.py` (`is_complete`), `active_processes.py` (`make active`).
   - **4c — post-hoc layer fully retired 🛠️ DONE**: `compute_adjusted_success` + `compute_adjusted_success_batch` deleted from `analysis.py`; the `analyze_run` wiring gutted (na_fp/eval_fp compute + `success` override + the zero-reader dead derived cols `wasted_cost_usd_adjusted` / `cost_efficiency_ratio_adjusted` / `success_rate_raw`) — `success` is now the canonical outcome with no override; runner stops computing `adjusted_success`/`fp_reason` (+ dead `_na_ids_cache` removed); `EpisodeSummaryV2` + `schema_migrations/v2.py` fields removed (safe — instantiation is field-by-field, no `**dict`); breaking callers fixed — `analyze_confidence_calibration.py` collapsed to alias (`--no-adjust` now a documented no-op), `rederive_episode_summary.py` stops re-deriving (5-field → 4-field), `analyze_reason_diagnostics.py` + `analyze_cross_representation.py::_mark_false_positives` mirror collapsed to `== success` aliases for output-schema stability; 4 retired-function tests in `test_router_and_metrics.py` removed. 11 files, py_compile clean ×10, 41 pytest pass incl. end-to-end runner+analysis.
   - **4b — non-live `EXPECTED_N` consumers 🛠️ DONE**: 5 of 6 switched to `scored_task_count` — `fig1ab_cascade_diamond.py`, `fig3_regional_carbon.py` (label now derives from the count), `axis_effect_size.py`, `axis1_microbehavior.py`, `mechanism_per_task.py` (all validate/label *live phase1 rerun* data → post-#76 counts cls 224 / red 205). `power_analysis.py` **kept hardcoded with a comment** — it is a pre-registered design-time power computation and the committed prereg power section is locked to the pre-exclusion design N (234/210/466); the ~4% N reduction is a negligible MDE shift and updating it would desync the prereg. py_compile clean ×6, computed counts verified.
   - **4d — downstream reader cleanup 🛠️ DONE**: `aggregate_sr_fp_per_mode.py` collapsed (was warning per-row post-4c since it reads raw `*_summary_v2.json` — now reports the single canonical SR, FP count structurally 0, dual keys kept == each other for schema stability); pure `.get("adjusted_success", …success)` graceful-fallback dead-key lookups removed across 13 files (`active_processes.py`, `layered_status.py`, `aggregate_phantom_lift.py`, `hero_claim_bootstrap.py` + 8 figure scripts `fig0c/0d/0e/0f/2f/3a/3d` + `fig_phantom_structure_venn`); `mechanism_per_task.py::summary_success` dead branch removed; stale comments/labels fixed. Deliberately kept as documented `== success` aliases (output-schema stability, no correctness impact): the `adjusted_success` columns in `analyze_reason_diagnostics.py` + `analyze_cross_representation.py` CSV outputs, `layered_status.py`'s `adjusted_successes` key, `compute_wasted_cost`'s optional `adjusted_success=None` param, `generate_gallery.py`'s FP-badge JS (now inert). py_compile clean, 47 pytest pass. **#77 fully closed.**

**B-85** (`has_effective_action` only counts `type`/`select_option`) — 🔄 **SUPERSEDED** by this restructure (piece 2). Claude+codex research: of 13 archived episodes the current logic would eval_fp-downgrade, only 4 are legitimately click-only (vote/delete) and 8 are stale comment-task FPs the narrow filter accidentally caught — confirming the heuristic is unfixable in a scalable way. Resolution = drop the branch, not patch the heuristic.

### B-88. reference images in phantom "no-image" modes ❌ NOT_A_BUG

- **Origin**: Claude `/stress` F5 (2026-05-14, 笔记 §139); reclassified after user review.
- **Why not a bug**: `qwen3vl_agent.py` / `proxy_api_agent.py` inject `reference_images` unconditionally (not mode-gated) — so DOM mode receives them too, not just phantom modes. Reference images are **task-spec inputs** (the product photo for a "find this item" task — the task is unachievable without them in any mode), not page observations. They are uniformly present across all 6 modes and are therefore orthogonal to the image axis (which is strictly about page-screenshot present/absent). No asymmetry exists. F5 mis-framed it by overlooking that DOM also receives reference images.
- **Action**: paper-prose clarification only — define the image axis as "page screenshot present/absent" and state reference images are uniform task-spec inputs outside the axis. No code change.

### B-90. `_extract_text_marks` `re.search` any-position `[N]` match — over-inclusion 📊 EVALUATED, no fix

- **Origin**: flagged in prior audits; re-evaluated for §139 (task #72).
- **Evaluation (2026-05-14)**: measured 3000 sampled archived `observation_dom.txt` — of 155771 lines matched by the current `re.search(r"\[(\d+)\]")`, **155770 are also matched by an anchored `^\s*\[(\d+)\]`**. Over-inclusion = **1 line = 0.001%** (a tab title ending in `[7]`). VWA AXTree element lines reliably carry `[N]` at line start.
- **Status**: 📊 **EVALUATED → no fix** (user decision 2026-05-14). Over-inclusion is empirically negligible; tightening to an anchored regex would change the fire path for a 1-in-155771 effect — not worth it. Documented as known-negligible.

### B-89. `cost_efficiency_ratio` stays raw while `success_rate` is overridden to adjusted 🛠️ FIXED

- **Origin**: codex Mode B C6 (2026-05-14, 笔记 §139).
- **Files**: `p79/experiment/analysis.py` (`analyze_experiment` adjusted-override block), `p79/experiment/metrics.py` (comment).
- **Mechanism**: `metrics.py::aggregate_condition_metrics` computes `cost_efficiency_ratio` from raw `success`. `analysis.py` overrides `cond_df["success_rate"]` to adjusted-success but never recomputes the ratio → any cost table using `cost_efficiency_ratio` mixes raw-success economics with adjusted-success SR conclusions.
- **Status**: 🛠️ **FIXED** (code). 2026-05-14.
- **Fix**: `analysis.py` adds `cost_efficiency_ratio_adjusted` to `cond_df` (computed from `ep_df["adjusted_success"]` × `total_cost_usd`, guarded on column presence) in the same block that overrides `success_rate`; raw `cost_efficiency_ratio` kept untouched; `metrics.py` comment updated to point at the adjusted field. py_compile clean; smoke confirms the grouped ratio (c1 0.667 / c2 0.200).
- **Paper impact**: adjusted-success cost tables can now use a matching adjusted ratio instead of silently mixing bases.

### B-91. VWA evaluator credits empty-answer predictions — na_fp + string_match eval_fp root cause 🛠️ FIXED

- **Origin**: Claude /stress B-85 follow-up → Claude code-read + codex cross-investigation (2026-05-14, 笔记 §139). codex outputs `b85_fp_filter_FINAL` + `vwa_eval_fp_search_FINAL`.
- **Files**: `external/visualwebarena/evaluation_harness/helper_functions.py` (`llm_fuzzy_match` + `llm_ua_match`). VWA submodule commit `f0c835b` on branch `p79-patches`.
- **Mechanism**: `StringEvaluator.__call__` takes `pred = last_action["answer"]` (`evaluators.py:212-213`). When the agent never submits a real finish, VWA `run.py:425-427` and the P79 runner (`runner/main.py:1426-1432`) both append a fake stop action with `answer=""`. The empty `pred` then reaches the LLM judges — `llm_fuzzy_match` (fuzzy `string_match`, `evaluators.py:274`) and `llm_ua_match` (N/A tasks, `evaluators.py:266`) — neither of which guarded against an empty prediction. GPT-4o-mini handed an empty answer can return `'correct'` / `'same'` → false-positive success. The deterministic string approaches (`exact_match` / `must_include` / `one_of` / `required_values`) already return 0 on empty `pred` — only the two LLM-judge paths FP. **na_fp and string_match eval_fp share this single root cause.**
- **Literature context**: a known severe VWA/WebArena issue — WebArena-Verified classifies N/A scoring as an "evaluation mechanism issue" (the harness credits `"N/A"`, cannot distinguish a reasoned unachievable judgement from an early exit); PAE reports ~50% of WebArena "successes" are evaluator false positives; WONDERBREAD filters impossible tasks. No upstream VWA fix exists (`git log` on `evaluators.py`/`helper_functions.py` confirms).
- **Status**: 🛠️ **FIXED** (code, VWA submodule). 2026-05-14.
- **Fix**: deterministic `if not pred or not pred.strip(): return 0.0` guard at the top of both `llm_fuzzy_match` and `llm_ua_match` — return 0.0 before the LLM call. Source-level, deterministic, no judge-prompt change. py_compile clean; guard logic verified (empty/whitespace/None → 0.0; real answer / `"N/A"` → falls through). Strictly more correct than the prior post-hoc `compute_adjusted_success` na_fp/string_match downgrade: it also catches "agent finished but with an empty answer", which the old `¬agent_finished` keying missed.
- **Supersedes**: P79's post-hoc `na_fp` + `eval_fp(string_match)` downgrade in `compute_adjusted_success` (part of the §139 FP-architecture restructure — `adjusted_success` retired once all 4 pieces land).
- **Paper impact**: `success` is correct at the evaluator boundary; FP handling moves from a post-hoc adjustment layer to a documented source-level fix, defensible by citing the WebArena-Verified / PAE literature.

Non-bug verify item ✅ VERIFIED (task #73, 2026-05-14): `phantom_text` config uses legacy
`observation_mode: phantom_dom` → `condition_id = phase1_phantom_dom_router_0`. Confirmed safe:
`run_registry.canonical_mode()` resolves both `phantom_dom` and `phantom_text` → `P-text` at
the registry boundary; the primary Phase 1a behavioral aggregator (`aggregate_sr_fp_per_mode.py`)
operates entirely in canonical `PAPER_MODES` space, so the raw alias never leaks. (Minor,
non-blocking: a few mechanistic NPZ analysis scripts hardcode `phantom_text` — separate from
the behavioral fire.)

### §139 audit status (2026-05-14): B-83 / B-84 / B-87 / B-89 🛠️ FIXED · B-88 ❌ NOT_A_BUG · B-90 📊 EVALUATED no-fix · B-85 / B-86 open

---

### §141 /stress A1.1+A1.2+A1.3 audit wave (2026-05-15)

Three audit surfaces, 14+ findings filed, 7 fixed in code, parse/GLM-touching items routed to `parse_advisor_pending.md` parking lot.

#### A1.1 (agent layer — `p79/agents/`) — Claude /stress + codex Mode B

- **B-92** Gemma3-VL prompt parity import-time fragility 🛠️ **FIXED commit `11d6fd9`** — `Qwen3VLAgent._make_*_prompt` converted to `@staticmethod`; Gemma callsites drop `(None)` bound-method-on-None pattern; 4 invariant tests in `test_agents_prompt_parity.py`.
- **B-93** B0 scroll vocab effective binary vs B1/B2 continuous (B-28 schema-only mitigation sharpener) 🛠️ **DISCLOSED** — paper §3.5.1 disclosure in commit `11d6fd9`; B-28 catalog entry updated to "MITIGATED schema only".
- **B-94** B0 `input_image_tokens` absent → "cost ≈ DOM" paper hero (4-fold drop-in property a) not apples-to-apples 🛠️ **DISCLOSED** — paper §3.5.1 footnote in commit `11d6fd9`; cross-baseline cost comparison should use `Δusage_between_modes` not absolute proxy `input_tokens`.
- **A1.1 Claude F3 / Codex C3 / Codex C2 sub-axes** — parse / GLM-related; routed to `parse_advisor_pending.md`.

#### A1.2 (backends — `p79/backends/`) — Claude /stress + codex Mode B (reproducibility auditor)

- **B-95** image_utils last-resort fallback redundant double-encode under default config 🛠️ **FIXED commit `2a05e9a`** — `encode_image_data_url` adds `over_cap` field to return dict; default-config fallback reuses loop tail's b64 instead of re-encoding identical JPEG; logger.warning when over_cap fires; 5 invariant tests in `test_image_utils_over_cap.py`.
- **B-96** factory backend dispatch silent default to LocalQwen on missing `type` key 🛠️ **FIXED commit `9316c8a`** — `factory.create_backend` raises ValueError on missing `type`; verified Phase 1a configs all explicit.
- **B-97** HeuristicDomBackend cfg-drop in factory dispatch 🛠️ **FIXED commit `9316c8a`** — `HeuristicDomBackend.__init__` now accepts `(backend_id, config)` like other backends; factory dispatches via normal constructor; class-level `backend_id` default removed (dead code).
- **B-98** MockBackend scroll delta mismatch with LocalQwen/Gemma mock_mode 🛠️ **FIXED commit `9316c8a`** — `factory.MockBackend.step` delta 0.5 → 0.8; 5 invariant tests in `test_factory_dispatch.py`.
- **B-99** schema_version identity split: `EPISODE_SUMMARY_V2_DEFAULTS["schema_version"] = "v2"` vs runtime `SCHEMA_VERSION_V2 = "2.0"` 🛠️ **FIXED commit `<pending>`** — v2.py imports `SCHEMA_VERSION_V2` so fill_defaults backfill uses the runtime constant; 1 invariant test in `test_som_and_schema.py`. Latent (fill_defaults currently 0 callers) but closed before v3 migration.
- **A1.2 Codex C2/C4** silent zero-fill in backend meta — covers `glm_fallback_used` axis (parse) + `image_payload_bytes` already (commit `2a05e9a`) + timing decomposition / confidence not yet. **Backend meta contract design pending advisor** since `glm_fallback_used` axis tied to GLM decision — see `parse_advisor_pending.md` §3 Option B aggregator changes. Hold.

#### A1.4 (som.py + mark extraction) — Claude /stress + codex Mode B (mechinterp implementer)

- **B-102** Two-layer silent fallback to DOM on unknown observation mode 🛠️ **FIXED commit `<pending>`** — `som.py.prepare_observation_for_mode` + 3 baseline agents' `_system_prompts.get(mode, dom_default)` silently accepted typo modes (e.g. `'phantum_som'`) and ran DOM-like obs / DOM prompt while downstream `condition_id` label still recorded the typo. Now: `KNOWN_OBSERVATION_MODES` frozenset at som.py + strict ValueError on unknown mode at all 4 dispatch points; 3 invariant tests guarding both layers (`test_prepare_observation_rejects_unknown_mode` + `test_prepare_observation_accepts_all_known_modes` + `test_agent_layer_strict_rejects_unknown_mode` source-grep guard).
- **B-103** Mechanistic extractor `_build_user_text` missing `Accessibility Tree:\n` prefix for DOM / phantom_prompt 🛠️ **FIXED commit `<pending>`** — production agent (`qwen3vl_agent.py:441-450`) prepends the header; mechanistic / cross-family extractors were missing it, so any NPZ extracted for `dom` / `phantom_prompt` modes was not byte-identical to production. Fixed in 3 files: `p79/mechanistic/extract_hidden_states.py:_build_user_text` + `scripts/mechanistic/run_stage4_h1_qwen2vl.py:_build_user_text` + `scripts/mechanistic/run_stage4_h1_phi35.py:_build_user_text`. User 2026-05-15 policy update: "these all need fixing, not just cataloged" — earlier "wait for §5 re-activation" framing was wrong (advisor §138 says **冻结存档** = frozen archive, no re-activation planned). Code is fixed in place for any non-mechanism reuse; archived NPZ from before this commit retains the byte-divergence and is documented as legacy. Codex Mode B (mechinterp implementer persona) caught this in trace mid-investigation; Claude missed. Test: `test_mechanistic_build_user_text_has_accessibility_tree_prefix`.
- **B-104** B-92 propagation gap — 5 callsites still passed `self`/`None` to now-@staticmethod prompt methods 🛠️ **FIXED commit `<pending>`** — discovered while sweeping B-103. `extract_hidden_states.py:73,74,82` passed `self`; `run_stage4_h1_qwen2vl.py:153,154` + `run_stage4_h1_phi35.py:179,180` passed `None`. All would TypeError on first instantiation against the @staticmethod descriptor (commit `11d6fd9` made `_make_*_prompt` no-arg). Mechanism §5 paused so none of these instantiated in production — latent until manual mechanism re-run. Fixed in same 3 files; test `test_no_legacy_self_or_none_prompt_callsites` guards future regressions across all of `p79/` + `scripts/`.
- A1.4 Claude F1 cross-family `_options_map` skip / F3 inject window / F4 `_collect_bbox_map` recursion / F5 `apply_som` legacy / F6 `_options_map` telemetry — sweep pending (next batch per user 2026-05-15 "都要修" policy).

#### A1.3 backlog sweep (2026-05-15) — per user "都要修" policy

- **B-105** A1.3 F2 walk-up depth telemetry — `WALK_UP_MAX_DEPTH = 6` exposed as named module-level constant, interpolated into all 3 JS resolvers via f-string. Single-source for any future tuning / calibration. 🛠️ **FIXED commit `<pending>`**. Tests: `test_walk_up_max_depth_is_named_constant`.
- **B-106** A1.3 F3 `<a>` no-href + actionable extension — `_JS_RESOLVE_CLICK` now also accepts `<a>` tags with `onclick` handler (either DOM property or `onclick` attribute), so JS-only links no longer fall through to framework bbox-center fallback. `<button type=reset>` discrimination remains the agent's responsibility (not dispatch); paper §3.5.1 cross-baseline disclosure covers the principle. 🛠️ **FIXED commit `<pending>`**. Tests: `test_click_resolver_accepts_anchor_without_href_with_onclick`.
- **B-107** A1.3 F4 ARIA actionable role accept list extended — added `menuitemradio` / `switch` / `treeitem` / `gridcell` / `radio` / `checkbox` (non-native) / `combobox` / `slider` to `_ACTIONABLE_ARIA_ROLES_JS` constant. Modern web apps using ARIA-only widgets no longer fall through to bbox-center fallback. 🛠️ **FIXED commit `<pending>`**. Tests: `test_click_resolver_accepts_extended_aria_roles`.
- A1.3 F5 catalog B-03 prose self-correction (locator.clear → fill('')) — inline above in B-03 entry. 🛠️ DONE.

#### A1.4 backlog sweep (2026-05-15) — per user "都要修" policy

- **B-108** A1.4 F3 `_options_map` look-ahead window 2-line cap → next-mark-id boundary only 🛠️ **FIXED commit `<pending>`** — `build_som_text_from_obs_text` (som.py:79-92) loop now scans until next mark id appears, removing the fixed 2-line distance cap. Closes the cross-file silent contract between vwa_wrapper inject distance and som.py recovery distance. Tests: `test_options_map_lookahead_is_unbounded_by_distance` + `test_options_map_boundary_is_next_mark_id`.
- **B-109** A1.4 F4 `_collect_bbox_map` cyclic-reference / depth guard 🛠️ **FIXED commit `<pending>`** — adds `_visited: set` (by id()) + `_depth` cap (`_BBOX_TRAVERSAL_MAX_DEPTH = 50`). Cyclic raw obs dicts no longer RecursionError. Tests: `test_collect_bbox_map_handles_cyclic_reference` + `test_collect_bbox_map_respects_depth_cap`.
- **B-110** A1.4 F6 `apply_som` legacy DeprecationWarning 🛠️ **FIXED commit `<pending>`** — any caller still routing through the legacy `include_full_axtree=True` path surfaces at runtime. Test: `test_apply_som_emits_deprecation_warning`.
- A1.4 F1 cross-family `_options_map` skip — **documented as intentional design choice** (commit `<pending>`): cross-family scripts test format variation; raw marks without `[SOM_MARKS]` wrapper is the goal; `_options_map` recovery would re-introduce wrapper. Header comments added in `run_stage4_h1_qwen2vl.py` + `run_stage4_h1_phi35.py` explaining the tradeoff. Mechanism §5 paused per advisor §138; if §5 re-activates, cross-family scripts must either switch to canonical wrappers (accepting format-loss) or inject `[OPTIONS]` per-formatter. Not a code fix.
- A1.4 F5 `_FONT_CACHE` documented as module-level by design — bounded ~150KB upper bound, no eviction needed. Inline comment added. Not a code fix.

### §142 /stress A1.1 v7 re-run with cross-AI Mode B + Mode C (2026-05-15 evening)

User re-ran /stress on A1.1 (`p79/agents/`) after skill v7 land (Mode C `/gemini-stress` added). Goal: deeper code-audit at milestone scope (5-7 files / ≥5 findings / ≥2 OOB) with Claude + codex (ML systems engineer persona) + gemini (claim-audit persona) tri-AI. 5 fixes land + paper-grade contamination scale recalibrated via codex Q1-Q6 archived data scan.

- **B-111** B0/B1/B2 prompt parity broken by shopping-domain examples (`Blankets & Throws` / `Home & Kitchen` / `Electronics` / `Jewelry & Watches`) 🛠️ **FIXED commit `<pending>`** — `gemma3vl_agent.py:23-27` asserted "byte-identical prompts cross-baseline" as hard paper-grade requirement, but `proxy_api_agent.py._get_system_prompts` carried separately-authored prompt with shopping-specific examples while B1/B2 shared Qwen3 staticmethod prompts that ALSO included two shopping examples in the `select_option` schema. Two-part fix: (1) `ProxyApiAgent._get_system_prompts` now reuses `Qwen3VLAgent._make_*_prompt()` verbatim → byte-identical B0/B1/B2; (2) deleted `(e.g., "Electronics", "Jewelry & Watches")` from Qwen3 DOM + SoM prompts. Three baseline now byte-identical AND free of shopping-domain prior leak on classifieds + reddit Phase 1a workload. Tests: `test_b0_prompts_byte_identical_to_b1_b2` + `test_b0_prompts_have_no_shopping_specific_examples` in `test_agents_prompt_parity.py`. Catches reviewer-3 attack "B0 prompt含 shopping leak → cross-baseline 比较被 domain prior 污染". Claude /stress F1 + Gemini F1 confirmed.
- **B-112** B0 image telemetry (over_cap / payload_bytes / quality / compressed) lived in agent meta but runner step_record dropped them → Q5 audit of B0 `image_over_cap` fire rate **structurally impossible** from archived JSONL 🛠️ **FIXED commit `<pending>`** — added `image_meta: Optional[Dict[str, Any]] = None` to `StepRecordV2`; runner.main step_record builder lifts the five image_* keys from agent meta into the new field (skipping `None`-valued keys). Codex Mode B C1 caught — Claude completely missed. Tests: `test_step_record_v2_has_image_meta_field` + `test_runner_persists_image_meta_from_agent_meta` in `test_stress_a1_1_fixes.py`.
- **B-113** B0 image encode failure silently degraded to text-only episode vs B1/B2 raise → cross-baseline missingness pattern asymmetric, invisible from JSONL 🛠️ **FIXED commit `<pending>`** — `proxy_api_agent.py` step() adds `_image_encode_error_count` tracker incremented in both reference-image and screenshot encode catches; meta surfaces it as `image_encode_error` field (`None` when 0 to keep JSONL clean), persisted via B-112's runner step_record block. B0 keeps lenient try/except (proxy-side transient encode errors should not abort episode) but downstream audit can now symmetric-exclude or paper §3.5.1 disclose missingness asymmetry. Codex Mode B C2 caught. Test: `test_proxy_agent_meta_carries_image_encode_error_field`.
- **B-114** `parse_action_text` scroll/back keyword fallback executed downstream despite `valid=False` → silent partial automation 🛠️ **FIXED commit `<pending>`** — `action_utils.parse_action_text` returned `action_type=scroll / back` with `valid=False` on unparseable text containing those keywords. Runner does NOT gate `env.step` on `parse_valid`, so the fallback action was actually executed — codex Q4 archived data confirmed `keyword_scroll/back` steps with `action_success=True` (= ~1.5% of fp-population steps were substring-match driven, not model-action driven). Now ALL parse failures fall through to wait (mirrors §67 `keyword_finish` removal). Tests: `test_parse_unparseable_with_scroll_word_falls_to_wait` + `test_parse_unparseable_with_back_word_falls_to_wait` in `test_action_utils.py`. Codex C3 caught — Claude flagged "unclear" in F5, codex archived-data verification confirmed.
- **B-115** Runner top-level `cfg.model.revision` cross-backend leak — could silently overwrite Gemma3 backend (`local_gemma`) revision=None with the Qwen-specific SHA 🛠️ **FIXED commit `<pending>`** — `runner.main:_get_backend` revision-forward block now gated on `backend_cfg["type"] in {"local_qwen", "api_proxy"}` (the Qwen-class backend types that the top-level field was historically meant for). Gemma3 backend's `revision` must come from its own backend cfg (`local_gemma.py:38` already declares it from config). Previously a misconfigured Gemma3 backend cfg without an explicit revision would silently inherit Qwen3 SHA → load wrong base model. Test: `test_runner_revision_forward_is_qwen_class_gated` (source-level invariant in `test_stress_a1_1_fixes.py`). Codex Q3 cross-validate caught this latent leak — Claude completely missed.

- **B-116** Cross-baseline max_new_tokens parity (B1/B2 384 → 4096) 🛠️ **FIXED commit `<pending>`** — Codex Mode B Q2 archived-data scan quantified B1 384-cap silent-truncation rate at 0.017% (4/23307 steps). Memory guard rail "B0/B1 设计不对称是已知论文披露即可代码不改" retired (2026-05-15 evening) after user disclosed 学长 negotiation on logprob availability + official API channel — Class 1 (inherent deployment) vs Class 2 (historical sloppy) split makes the previous "disclose-only" stance untenable for purely codebase-controlled fields like max_new_tokens. 67 configs (37 B1 + 30 B2) updated via sed: `max_new_tokens: 384 → 4096`; all 110 `exp_v2_*.yaml` now uniformly at 4096. Test: `test_configs_b0_b1_b2_max_new_tokens_parity` (parses every config + asserts unified cap). **Trade-off audit**: `max_new_tokens` is a CAP not a forced length → decoder still stops at model EOS → 99.98% steps unchanged; only the 0.02% outlier steps (long thought + JSON envelope) now land instead of truncate. GPU mem unchanged (dynamic KV cache in `transformers.generate()`). Outlier wall time per cell ≈ +10 min (acceptable). **Paper-grade impact**: §142 F3 disclosure obligation deleted from paper §3.5.1 backlog; cross-baseline `parse_fail` rate now apples-to-apples. **Timing**: Phase 1a clean rerun未启动 (`phase1_plan §B0+§B2` 都 unticked) → no archived-data divergence — perfect window.

### §142 audit status (2026-05-15 evening): B-111 / B-112 / B-113 / B-114 / B-115 / B-116 🛠️ FIXED · §142 retro check pending 7 days

## §143 Post-Batch-1-5 propagation audit (cross-AI Mode B+C, 2026-05-15 late evening)

Trigger: user "再看看整体文档,看下有没有什么需要改的,不确定的,重要的决策,和现在不符合的等,然后用stressBC一起". Anchor = post-Batch-1-4b + Batch 5 residual cleanup (5 commits c188e8e → 2031212). Cross-AI dispatched in parallel: codex Mode B (`gpt-5.5`, code/pipeline anchor) + gemini Mode C (`gemini-3.1-pro-preview`, prose/design anchor). Both used Bug Table output spec v7.2 added this session.

Codex 10 findings (4 P0 / 3 P1 / 3 P2). Gemini 7 findings (2 P0 / 2 P1 / 1 P2 + 2 cross-cutting). Pattern: **prose-layer narrative propagated by Batch 1-5, implementation-layer (analysis pipeline / make gates / manifest registry / provenance / configs) still in 24/4/B0+B1/old-VWA-SHA world**. OSF lock NOT ready — many P0s gate it.

- **B-117** Makefile VWA SHA lock pin stale 🛠️ **FIXED commit `7b6b715`** — `Makefile:140` `verify-version-locks` expected `832f037e` (pre-B-91), actual submodule HEAD `f0c835b35191e2ff8d46993d9279674a0956ef14` (branch `p79-patches`, B-91 source-level FP guard). Pre-launch-check would fail OR force unsafe VWA downgrade (B-91 patch loss = na_fp reintroduced). Bumped LOCK_SHA + `locked_versions.md` row + changelog 2026-05-15. Codex Mode B P0-2 caught.
- **B-118** `configs/exp_v2_base.yaml:79` `hardware_profile: "dgx_spark"` 🛠️ **FIXED commit `7b6b715`** — DGX power profile selected by energy_tracker while canonical paper-grade run is A100 self-host. Paper §8 GPU-hour / kWh / CO2 numbers would mis-label host (paper §7 declares A100, §8 power says DGX). Updated to `a100_pcie_40gb`. Codex Mode B P1-1 caught.
- **B-119** `snapshot_env.py` Qwen-only + missing external VWA helper hash + post-runner.run() timing 🛠️ **FIXED commit `7b6b715`** — three sub-issues per codex Mode B P1-2: (a) `DEFAULT_MODELS` only listed Qwen → B2 run snapshots recorded Qwen revision (default) instead of Gemma → replicator would use wrong model; (b) `EVALUATOR_SOURCE_FILES` excluded external `helper_functions.py` (where B-91 lives) → evaluator_code_combined_sha256 couldn't detect B-91 patch regressions; (c) `capture_env_snapshot` invoked AFTER `runner.run()` → crashed runs had no provenance. Fix: added `google/gemma-3-4b-it` to DEFAULT_MODELS; added `external/visualwebarena/evaluation_harness/helper_functions.py` to EVALUATOR_SOURCE_FILES; moved `capture_env_snapshot` call to before `runner.run()` using `runner.output_root` (set in `__init__`).
- **B-120** `preregistration_decision_test.py` hardcoded 24-cond / 4-cell / B0+B1 only / K=3 🛠️ **FIXED commit `4f50cc6`** — paper hero verdict gate script ignored Batch 1 B2 / k=6 / Decision 3A propagation. `PHASE_1A_CELLS` expanded 4 → 6 tuples (added (cls,B2) + (red,B2)); docstring rescoped 24/4 → 36/6 with B2 rationale; K_h1/K_h3 default 3 → 4 (descriptive "4-5/6 = strong" per propagated prereg §4); banner notes Decision 3A FE-vs-current-DL-impl gap pending advisor (links B-130 escalation). Without fix: 36-cond rerun → script silently drops B2 data, pools 4 cells, outputs wrong R-rule verdict → paper §1 hero based on broken logic. Codex Mode B P0-1 caught. **Note**: parallel session also used "B-120" label for `api_proxy revision pin` fix in `runner/main.py:157`; bug-number conflict logged here for catalog audit — both fixes land but different scopes (this entry = decision-test rescope; sibling entry = revision pin).
- **B-121** `run_manifest.yaml` 0 paper-grade cells + no B2 entries + stale pre-N/A counts 🛠️ **FIXED commit `4f50cc6`** — manifest had 0 `grade: paper-grade` entries (bulk-archived 2026-05-04 for advisor sync); after A100 rerun lands, paper-grade promotion needs 36 new cells (B0+B1+B2 × cls+red × 6 modes). Archived B0/B1 entries still showed pre-N/A-exclusion task counts (234/210/466) while post-§139.8 dynamic counts via `p79.experiment.analysis.scored_task_count` are 224/205/435. Fix (header-only, body unchanged): added 2026-05-15 SCOPE UPDATE comment block documenting promotion workflow + dynamic vs hardcoded N reconciliation. Manifest body deferred to A100 rerun completion. Codex Mode B P0-3 caught.
- **B-122** `per_task_sr.csv` no producer 🛠️ **FIXED commit `4f50cc6`** — launcher commanded users to feed `preregistration_decision_test.py --per-task-csv results/phantom_paper/per_task_sr.csv` but no script generated that file (`aggregate_phantom_lift.py` produced `phantom_lift.csv`, different schema). User following launcher prompt would hit `FileNotFoundError`. New `scripts/analysis/generate_per_task_sr.py` (B-122 producer, ~200 lines): reads `run_registry.get_all_cells(grade_filter=["paper-grade"])`, iterates condition_dir/episodes/*_summary_v2.json, pivots to wide format (one row per (cell_id, task_id), cols = sr_<mode> × 6 + cost_dom + cost_psom). Launcher updated to call producer first, then decision test. Codex Mode B P0-4 caught.
- **B-123** B2 "matched-capability cross-family control" wording overclaim 🛠️ **FIXED commit `c53e9a0`** — 4B parameter parity ≠ matched capability (Gemma vs Qwen have different vision encoders, pretraining recipes, instruction-tuning configs). Reviewer attack "B1/B2 difference is family effect or just Gemma-4B vs Qwen-4B inherent strength gap?" — no anchor to disambiguate without MMMU/VQA zero-shot benchmark. Fix: 1h wording downgrade across `model_card` (multiple) / `dataset_card §63` / `locked_versions B2 section` / `preregistration scope statement + §2.4 power + §6 commit (10) + §7 external validity` / `section1_intro L13 + L15` / `section8 §8.1`: "matched-capability cross-family control" → "cross-family robustness check at 4B parameter parity". MMMU/VQA anchor option (1 day) deferred to advisor decision. Gemini Mode C P0-1 caught.
- **B-124** Mechanism §5 paper-2 deferral phantom-limb residuals 🛠️ **FIXED commit `c53e9a0`** — Batch 4b marked `section5_mechanism.md` as paper-2 working draft, but two phantom-limb residuals remained: (a) `section8 §8.6 "Sparse-mechanism caveat"` still discussed activation-patching effects (L11/L17 task-conditional sparseness) as paper-1 limitation; (b) `osf_lock_manifest §2.1 "Mechanistic 24+15 candidates"` lock-artifact row would have errantly locked paper-2 mechanism artifacts into paper-1 OSF DOI. Fix: section8 §8.6 rewritten "Mechanism evidence — deferred to follow-up paper" pointing to paper-2 working draft; osf_lock_manifest §2.1 row strikethrough+retire-tag with "retired 2026-05-15 (B-124 phantom-limb purge)". Gemini Mode C P1-4 caught.
- **B-125** section1 router contribution Behavioral Divergence bridge missing 🛠️ **FIXED commit `a67deae`** — mechanism §5 paper-2 deferral left §6 router without theoretical justification why TF-IDF + binary features should work as routing signals. Reviewer attack "你就训了个 classifier 在 task description 上 overfit,没原理". Fix: prepended behavioural-prediction bridge sentence at start of third-contribution paragraph in `section1_intro.md`: "The behavioural two-knob account predicts a routing signal at the task level — quick-decide tasks favour flat `[SOM_MARKS]`, sustained-exploration tasks favour AXTree hierarchy". Bridges behavioural account (§5) → router operationalisation (§6) without leaning on activation-patching evidence. Gemini Mode C P1-3 caught.
- **B-126** `power_analysis.py` 16-cell K-of-N hard-gate framing obsolete 🛠️ **FIXED commit `a67deae`** — docstring still said "K_h1=12 of 16 cells, K_h3=11 of 16" hard-gate framing, but preregistration retired this as transparency-only count 2026-05-14 (Decision 3A); at k=6 the ratios remain indistinguishable (⌈0.75×6⌉=⌈0.67×6⌉=5 — same fake-precision argument). User pasting `power_analysis.md` output into paper appendix would have numbers corresponding to non-existent K-of-N gates. Fix: docstring rewrite — 16-cell → 6-cell scope, K-of-N retire banner, descriptive 4-5/6 = strong-consistency benchmark per prereg §4 K-of-N row. Codex Mode B P2-9 caught.
- **B-127** `model_card.md` B0/B1 framing as "intentional scientific design" hides confounder 🛠️ **FIXED commit `a67deae`** — model_card framed B0 (API 235B) vs B1 (Local 4B) cost asymmetry as "intentional scientific design", but B0 has GLM-5.1 parse-error rescue scaffold (`proxy_api_agent.py::_call_glm_extract`) that B1/B2 don't, introducing hidden infrastructure confounder for any B0-vs-B1 scale claim. Fix: added explicit confounder bullet acknowledging GLM rescue, scale-claim softening requirement, pointer to section3 §3.5.1 + section8 limitations. Gemini Mode C P2-5 caught.
- **B-128** B2 shop VWA configs missing (only WA shop B2 existed) 🛠️ **FIXED commit `af89309`** — launcher advertised Phase 1b B2 × shop × 6 modes but `configs/exp_v2_B2_*_shopping.yaml` VWA files didn't exist (only WA = WebArena variants existed, different benchmark). Phase 1a unaffected; Phase 1b launch would have failed missing-config. Fix: templated 6 new B2 VWA shop configs (`exp_v2_B2_{dom,som,vision,phantom,phantom_text,phantom_prompt}_shopping.yaml`) from B1 shop templates, swapping `local_4b` backend → `local_gemma`. Codex Mode B P2-1 caught.
- **B-129** `queue_chain.sh` same-baseline same-site collision check missing 🛠️ **FIXED commit `af89309`** — cross-baseline check (B0 vs B1 vs B2 on same site) existed; same-baseline different-mode concurrent runs (e.g. `B0_dom_reddit + B0_som_reddit` launched manually outside master gate) shared site user account → RESET_BEFORE race + cross-mode episode contamination. Master gate `queue_phase1_paper_grade.sh` already blocks active runs; this defends against manual bypass. Fix: second `pgrep` check after cross-baseline loop matching same-baseline same-site, with same wait-loop semantics. `bash -n` syntax pass. Codex Mode B P1-3 caught.
- **B-130** Decision 3A FE estimand challenged by Gemini Mode C — ESCALATED to advisor (NOT auto-fixed) ⏳ **OPEN** parking lot `docs/checkpoints/parse_advisor_pending.md §8` — Decision 3A (FE inverse-variance pool over 6 planned cells, locked 2026-05-14) attacked from generalization-claim-coupling angle by Gemini Mode C P0-2: "FE estimand 把 paper §1 'phantom routing space generalizable property' claim 阉割了 — FE 只能证 在这 6 cells valid, no broader Web Agents generalization. 回滚 RE+Knapp-Hartung at k=6". 3-option tradeoff (keep FE + soften §1 wording / RE+Knapp-Hartung / both as primary+sensitivity) escalated to advisor; OSF lock email blocked until decision. Implementation drift also flagged (`aggregate_phantom_meta.py` + `preregistration_decision_test.py` currently compute DL — third option, code↔prose drift). Will NOT modify estimator unilaterally — Decision 3A advisor-witness-locked. Consolidated into `parse_advisor_pending.md §8` 2026-05-15 (was standalone `_status/issues/issue_decision_3a_fe_re_review_2026-05-15.md`, removed since user directed "B-130 advisor decision 放 parse_advisor_pending.md").

### §143 audit status (2026-05-15 late evening): B-117 / B-118 / B-119 / B-120 / B-121 / B-122 / B-123 / B-124 / B-125 / B-126 / B-127 / B-128 / B-129 🛠️ FIXED · B-130 ⏳ OPEN (advisor decision required) · §143 retro check pending 7 days

**Cross-AI verification chronicle**: Mode B (codex `gpt-5.5`, 7m12s, 10 findings / 4 P0 / 3 OOB — PASS pre-fire scope); Mode C (gemini `gemini-3.1-pro-preview`, 4m30s, 7 findings / 2 P0 / 3 OOB — PASS pre-fire scope despite slightly-fast runtime, quality met threshold).

**Structural finding (3-AI agreement, highest confidence)**: post-Batch-1-5 propagation is **asymmetric** — narrative layer (preregistration prose + paper drafts + planning docs) fully aligned to 3-baseline / 36-cond / k=6 / A100 / mechanism-paper-2 / router-paper-1 scope; implementation layer (decision script / make gates / manifest registry / per_task_sr producer / configs / power_analysis K rules / snapshot Gemma support) **was lagging across 10 places** until B-117~B-129 fixes landed this session.

**Cross-AI complementarity validated**: Gemini caught what code-only audit missed (B-123 B2 wording overclaim, B-124 phantom-limb prose residuals, B-125 router theory bridge, B-127 confounder admission, B-130 FE estimand challenge); codex caught what prose-only audit missed (B-117 Makefile SHA, B-118 hardware profile, B-119 snapshot provenance, B-120 decision test, B-121 manifest, B-122 producer missing, B-126 power_analysis script staleness, B-128 missing configs, B-129 queue collision). Three lineages' blind spots stacked additively — exactly what cross-AI design predicts.

**Cross-AI verification chronicle**: Mode B (codex, 120s, 5 findings / 2 OOB) — PASS Phase 1+2+3 sanity; Mode C (gemini, 120s, 3 weak + 2 strong) — PASS (borderline count but P0 quality + paper-layer disclosure status check unique value); two-AI diff produced. Codex archived-data Q1-Q6 recalibration: B1 384 cap actual silent-truncation rate = 0.017% (not "high probability" as Claude F3 assumed); B0 GLM rescue rate = 1.49% (parse parking lot data). F4 B0 reproducibility seed no-op **defused** by gemini reading preregistration §7 disclosure ("server-side determinism is best-effort"). F2 cross-model AUROC routing claim → paper §4 estimand decision pending advisor.

**Codex caught Claude completely missed**: B-112 (image telemetry dropped in step_record) + B-115 (runner revision cross-backend leak) + B-114 archived-data confirmation. **Gemini caught Claude missed**: F4 already disclosed in preregistration §7 (so the seed no-op attack is paper-defused). **Claude unique**: F1 prompt parity grep-level byte-difference verification (codex implicit, gemini paper-level claim attack).

#### A1.3 (envs — `p79/envs/`) — Claude /stress (codex Mode B incomplete twice)

- locator_dispatch.py F1 (handle dispose leak in action-raise path) + F6 (walk-fail reason category in error field) — pending land per user 2026-05-15 sequence.
- F2 (walk-up depth 6 hardcoded, no telemetry) + F3 (`<a>` no-href + `<button type=reset>` boundary) + F4 (ARIA role accept list incomplete) — backlog.
- F5 (catalog B-03 prose "locator.clear()" vs code `fill('')` mismatch) — catalog prose 2-min fix.

#### Status counts increment

| Audit | New entries | FIXED in commits | DISCLOSED to paper | Parking lot | Pending fix |
|---|---|---|---|---|---|
| A1.1 | 5 | 1 (B-92) | 2 (B-93, B-94) | 2 (Claude F3, Codex C3, Codex C2 GLM axis) | 0 |
| A1.2 | 5 | 4 (B-95, B-96, B-97, B-98, B-99) | 0 | 1 (Codex C2/C4 backend meta contract) | 0 |
| A1.3 | 6 | 0 (in progress) | 0 | 0 | 6 (F1/F2/F3/F4/F5/F6) |

---

## Updated Status Counts (post-§116 audit + Phase 0 backfill)

| Tag | Count | Notes |
|---|---|---|
| ✅ **CONFIRMED** | ~28 | B-01 to B-37 mostly CONFIRMED but unfixed (Phase A audit findings); fix scope decided per evaluator_change_protocol.md Tier classification |
| 🛠️ **PARTIALLY FIXED** | 1 | B-37 (seed determinism, multi-component) |
| ⚠️ **DISPUTED** | 0 | — |
| ❌ **NOT_A_BUG** | 4 | B-12 / B-13 / B-14 / B-27 |
| 🔄 **UNVERIFIED** | 0 | All Phase A entries CONFIRMED via static read |
| 🛠️ **FIXED** | ~54 | B-10 (§105) + B-26 (FIXED 2026-04-19 commit `3f9ceca`) + B-28 (MITIGATED) + B-29 (NOT FIXED BY DESIGN) + B-38 (§116) + B-81a-h (HPC porting umbrella, 8 sub-classes) + Phase 0 historical (B-39 to B-80, including umbrella sub-entries) + Phase A patches via commits 3c15cd7 onwards |

**Pre-rerun rule reaffirmed**: All 🛠️ FIXED bugs must have their fix in code at HEAD before
16-cell rerun launch (笔记 §116 / pre_rerun_audit.md §F). UNVERIFIED entries have been
triaged to either ✅ CONFIRMED (with paper §3 disclosure) or 🛠️ FIXED. Remaining ~28 CONFIRMED
entries are documented in catalog with fix-scope decision per evaluator_change_protocol.md
Tier classification (Tier 0/1 deferred to post-rerun if not blocking).

### §144 /stress A1.1 v8 with 3-AI cross-AI cycle (2026-05-15 evening, Commit A)

User re-fired /stress on `p79/agents/` (A1.1) with full Claude+codex+gemini 3-AI cycle. Skill v7.3 hard constraint enforced: unified bug table presentation BEFORE fix work; user confirmed fix scope; 7 code fixes + 1 prereg disclosure landed in Commit A. 5 P0/P1 items deferred to `parse_advisor_pending.md` pending 学长 sync on (a) official Qwen API channel (P0-2) (b) DashScope logprob availability (P0-7 + P1-4) (c) GLM scaffold drop policy (P1-1) (d) full T=0 reproducibility audit scope (P0-9 partial).

- **B-131** Runner `_QWEN_CLASS_BACKEND_TYPES` cross-backend leak — api_proxy still in injection set after B-115 closed local_gemma leak → B0 235B's `condition_meta.json` reported B1 4B SHA `ebb281ec...` as fake revision pin 🛠️ **FIXED commit `<pending>`** — `runner.main._get_backend` injection set narrowed to `{"local_qwen"}` only. Codex Mode B F2 (reproducibility-auditor persona) caught — sibling-propagation gap from B-115 (skill v6 sibling-script propagation check 实证 case). B0 provider provenance now requires separate `provider_snapshot_id` field (out of scope, parking lot). Test: `test_runner_revision_forward_is_qwen_class_gated` (updated assertion to `{'local_qwen'}` only).

- **B-132** Multi-seed `condition_id` schema drift in runner — `effective_cid = condition.condition_id + "_seedN"` used for directory + `condition_meta.json`, but `StepRecordV2` + `EpisodeSummaryV2` wrote unsuffixed `condition.condition_id` → multi-seed mode produces silent join/aggregate collision across seeds 🛠️ **FIXED commit `<pending>`** — `_run_episode` signature accepts `effective_cid: Optional[str] = None` (defaults to `condition.condition_id` for backward-compat with single-seed mode); `_run_and_record_episode` threads `effective_cid` parameter from runner main loop; both `StepRecordV2.condition_id` (main.py:1228) + `EpisodeSummaryV2.condition_id` (main.py:1555) use `effective_cid`. Codex Mode B F1 caught — latent until OSF multi-seed lock manifest activated. Test added (separate test PR planned).

- **B-133** Cross-baseline image-encode lenient asymmetry — B0 (proxy_api_agent) had try/except around PIL encode + counted `_image_encode_error_count` + continued text-only; B1/B2 (qwen3vl + gemma3vl) raised on same root cause → same PIL exception produced asymmetric episode outcome (B0 SoM step degraded to P-SoM step vs B1/B2 episode-killed) 🛠️ **FIXED commit `<pending>`** — B1 + B2 both wrapped image encode in try/except matching B0; emit `image_encode_error` count via meta dict (persisted via B-112 runner image_meta wiring). All 3 baselines now lenient + audit-able. Downstream `aggregate_*.py` must symmetric-exclude steps with `image_encode_error > 0` (paper-grade contamination flag, watchdog-auto-clean parallel — separate PR). /stress A1.1 v8 **3-AI overlap**: Claude F1 + codex C5 + gemini G4 all caught this disease class from different angles (outcome asymmetry / image_meta mandatory schema / SoM→P-SoM degradation). Highest confidence finding of the audit. Test: `test_b1_b2_image_encode_lenient_cross_baseline_align`.

- **B-134** Runner discards `validate_action()` bool result — `action, _ = validate_action(action)` at runner.main:954 + :988 threw away the bool; `parse_valid` was computed solely from `meta.get("valid", True)` → if backend returned malformed action (unknown action_type like "clik") with valid=True, runner rescued to wait but JSONL recorded `parse_valid=True / parse_failure_reason=None` → schema source-of-truth split between agent self-report and runner-executed action → cross-baseline parse_fail rate not comparable 🛠️ **FIXED commit `<pending>`** — saved `runner_valid_post_backend` bool; `parse_valid = agent_parse_valid AND runner_valid_post_backend`; emit `failure_reason="runner_invalid_action"` when backend self-reported valid but runner had to rescue. Failure taxonomy now distinguishes agent-side invalid from runner-rescued no-ops. Codex Mode B F3 caught — Claude completely missed during Mode A. Test: `test_runner_validate_action_bool_saved_into_parse_valid`.

- **B-135** B0 `max_new_tokens` default 512 vs B1/B2 default 4096 — `proxy_api_agent.py:505` `gen_cfg.get("max_new_tokens", 512)` falls through to 512 when config omits the key 🛠️ **FIXED commit `<pending>`** — default aligned to 4096. Current 107 active configs all explicit-set (codex Mode B Q2 archived-data confirm; B-116 enforced parity test guards this), so no behavior change for current Phase 1a; defense-in-depth fix for any future config missing the key (silent truncation → parse_error → GLM fallback fires → cross-baseline parse_fail asymmetric). Claude Mode A F3 caught.

- **B-136** Revision strict mode (3 agents) — qwen3vl_agent.py had hardcoded `_DEFAULT_REVISION = "ebb281ec..."` fallback with logger.warning; gemma3vl_agent.py had no default but warned-and-loaded-HF-HEAD when revision unset → silent provenance fallback (log warning ≠ commit-recorded SHA); run_meta records loaded SHA but config does NOT, so OSF artifact ≠ commit history → reviewers cannot reproduce 🛠️ **FIXED commit `<pending>`** — both agents now `raise RuntimeError("model.revision must be pinned in config")` if revision is missing/None. Single cross-baseline policy: paper-grade configs MUST explicitly pin SHA in yaml. B-83 backend-wrapper None-passing trap still handled via truthy `not self.model_revision` check. Base yaml verified: `local_qwen` inherits revision from top-level `model.revision: "ebb281ec..."` (line 103); `local_gemma` has its own `revision: "093f9f388b..."` (line 133). Claude Mode A F5 caught. Test: `test_qwen_gemma_revision_strict_mode`.

- **B-137** Base yaml temperature asymmetry — `local_4b` had `temperature: 0.1`; `local_gemma` and `api_strong` had 0.0 → code path uses `do_sample=False` so value never reaches generate(), but run_meta records yaml field verbatim → reviewer reading metadata sees B1 0.1 vs B0/B2 0.0 → false-alarm cross-baseline decoding config asymmetry 🛠️ **FIXED commit `<pending>`** — all 3 baseline backend blocks (`local_4b` / `local_gemma` / `api_strong`) now `temperature: 0.0`. Per-condition B0 configs already overrode to 0.0; base default aligned for any config that omits the key. Codex Mode B F8 caught (config-vs-code contract drift class). Test: `test_base_yaml_three_baselines_temperature_uniform_zero`.

### §144 audit status (2026-05-15 evening Commit A): B-131 / B-132 / B-133 / B-134 / B-135 / B-136 / B-137 🛠️ FIXED · 5 items → parse_advisor_pending.md · §144 retro check pending 7 days

### §144 Commit B (2026-05-15 late evening) — remaining P1 polish

- **B-138** T=0 greedy consistency probe (B0 paper-§3.5 reproducibility lightweight verification) 🛠️ **FIXED commit `<pending>`** — new script `scripts/maintenance/probe_b0_greedy_consistency.py`. POST same payload N=10 to B0 proxy, hash responses byte-for-byte + extract action sig, classify into 3-tier verdict: MECHANICAL_GREEDY (≥99% byte-identical → no disclosure) / SEMANTIC_GREEDY_WITH_NOISE (≥90% byte-identical → paper §3.5 disclose reproducibility rate) / NON_DETERMINISTIC (<90% byte-identical → escalate advisor for full audit). No VWA dep, ~30min runtime. Full VWA-task-level T=0 audit deferred to `parse_advisor_pending.md §4` (advisor budget decision). Output JSON to `docs/checkpoints/reproducibility/b0_greedy_probe_<HHMMSS>.json`. Claude Mode A F4 caught (P0-9 partial defer). Test: `test_b138_probe_script_exists_and_compiles` (script existence + py_compile + 3-tier verdict labels invariant).

- **B-139** B2 image_token_count_method meta flag 🛠️ **FIXED commit `<pending>`** — `gemma3vl_agent.py:258-261` previously had silent conditional switch between `n_images * 256` estimate and exact `processor.image_token_id` count via `if img_tok_id is not None:` overwrite. Meta now emits `image_token_count_method` enum ("exact_id_match" vs "estimate_256_per_image") via runner B-140 image_meta block — transformers version drift between estimate and exact no longer changes cost-accounting semantics without audit trail. Claude Mode A F6 caught. Test: `test_b2_image_token_count_method_emitted_in_meta`.

- **B-140** image_meta mandatory dict schema 🛠️ **FIXED commit `<pending>`** — runner.main.py:1355-1377 previously had conditional `if _image_meta_payload: step_record["image_meta"] = ...` (image_meta absent when no fields populated). Now MANDATORY (always emitted) with explicit `pipeline` enum ("proxy_jpeg_data_url" / "hf_processor_pil" / "unknown") + 5 telemetry fields + B-139 `image_token_count_method` field — even None values explicit. Closes "unsupported vs missing vs failed extraction" ambiguity for downstream analysis. Codex Mode B C5 caught — paired with B-133 image-encode lenient align (Commit A) so all 3 baselines emit symmetric image_meta. Test: `test_runner_image_meta_is_mandatory_dict`.

- **B-141** parser robust JSON repair 🛠️ **FIXED commit `<pending>`** — `action_utils.parse_action_text` previously used greedy regex `re.search(r"\{.*\}", text, re.DOTALL)` capturing first-{ to last-} which silently mismatched on outputs like `"{action} then maybe {alternative}"` (captured whole spanned region → JSON decode error) and `"{action} // notes"` (captured trailing comment → error). Replaced with two-tier scan: (1) fenced ```json {...} ``` block (markdown wrapping common when models echo "Output ONLY valid JSON" instruction), (2) `json.JSONDecoder().raw_decode()` walking each `{` position to find ALL valid JSON candidates. New failure_reasons: `repaired_fenced` / `repaired_raw_decode` / `repaired_multiple_identical` / `multiple_actions` (ambiguity when ≥2 distinct valid actions). Cross-baseline parse_fail rate now driven by model output structure, not by parser heuristic brittleness. Codex Mode B C6 caught. Tests: `test_parse_fenced_json_block_repair` + `test_parse_multiple_identical_actions_repairs_to_one` + `test_parse_multiple_distinct_actions_flags_ambiguity` + `test_parse_invalid_action_after_raw_decode_emits_repaired_invalid` + `test_parse_no_json_at_all_falls_to_parse_failed` + existing `test_parse_repaired_regex_valid` updated to `repaired_raw_decode`.

- **B-142** validate_action coord/delta shape+range check 🛠️ **FIXED commit `<pending>`** — `action_utils.validate_action` previously only checked coord presence (`coord is None`), so payloads like `{"action_type":"click","coordinate":[2,"x"]}` or `[None, None]` or `"42,7"` passed validation and reached env.step → schema failure converted into env behavior/no-progress → cross-baseline parse_fail vs no_op_rate taxonomy polluted. New helpers `_is_valid_coordinate_pair` + `_is_valid_delta_pair` enforce: 2 finite floats (no NaN/inf/non-numeric), normalized [0,1] range OR pixel non-negative finite, scroll delta finite pair, tab_focus.page_number must be int ≥ 0, select_option.element_id must be int (not truthy string). Per-action strict schema; malformed coord/delta → wait+invalid. Codex Mode B C7 caught. Tests: 9 new in `test_action_utils.py` (malformed coord pair / wrong length / NaN+inf / valid normalized + pixel / scroll delta malformed/valid / tab_focus int check / select_option int element_id).

- **B-143** Network retry latency separate meta 🛠️ **FIXED commit `<pending>`** — `proxy_api_agent.py:545-580` retry loop previously only `time.sleep(wait)` without surfacing the overhead. B0 one transient 429 + 3 retry = +70s wallclock conflated into step latency_ms with no way to subtract for cross-baseline fair comparison (B1/B2 have no network retry). Now meta emits `network_retry_count` + `network_retry_wait_ms`; runner step_record latency_ms dict adds `total_minus_retry = total - network_retry_wait_ms`. Retry overhead is scaffold-level — NOT in agent cost accounting (cost is token-based, unchanged). Cross-baseline latency comparisons + §C router latency feature now use `total_minus_retry`. Claude Mode A F7 caught. Test: `test_proxy_agent_emits_network_retry_meta` + `test_runner_emits_latency_ms_minus_retry`.

### §144 Commit B audit status (2026-05-15 late evening): B-138 / B-139 / B-140 / B-141 / B-142 / B-143 🛠️ FIXED · §144 retro check pending 7 days

**Tests delta**: 25 (Commit A) → 57 (Commit B) — +32 invariant tests across `test_stress_a1_1_fixes.py` (+5 Commit B section) + `test_action_utils.py` (+5 B-141 + 9 B-142) + existing test updates. 100% pass rate maintained.

**Round counts (Commit B finalized)**:

| Tag | Count | Notes |
|---|---|---|
| 🛠️ **FIXED Commit A** | 7 | B-131 / B-132 / B-133 / B-134 / B-135 / B-136 / B-137 |
| 🛠️ **FIXED Commit B** | 6 | B-138 / B-139 / B-140 / B-141 / B-142 / B-143 |
| ⏳ **PARKING LOT** (advisor sync) | 5 | P0-2 / P0-7 / P0-9 full audit / P1-1 / P1-4 |
| 📝 **DISCLOSURE** (paper-side) | 2 | P0-1 prereg §2.6 (landed Commit A) / P2-1 paper §3.4.1 (landed Commit C, commit pending) |

**Cross-AI verification chronicle**: Mode B (codex, ~4min, 8 findings / 5 OOB / reproducibility-auditor persona) — PASS Phase 1+2+3 sanity; Mode C (gemini, ~23min, 5 findings / 3 OOB / broad-reviewer persona) — PASS Phase 1+2+3. Cross-AI agreement: 1 3-AI overlap (P0-5 image-encode lenient — Claude F1 + codex C5 angle + gemini G4 different angle, all same root); 1 2-AI overlap (P0-7 confidence meta — Claude F2 + codex C4 schema angle); 17 1-AI unique catches (5 Claude / 6 codex / 5 gemini). **Gemini standout**: 80% of findings unique catch; G2 reference_images "leak" attack initially flagged as P0 but user confirmed it's intentional design (DOM also has ref_img; cost ≈ DOM stays fair) → routed to prereg §2.6 disclosure instead of code fix. P2-1 (P-prompt header contradiction) also user-confirmed intentional manipulation → routed to paper §3.4.1 disclosure (Commit C deferred).

**Skill v7.3 hard-constraint enforcement empirical**: user caught Claude attempting to fix mid-stream (after Mode A+B but before Mode C) → hard constraint added to `.claude/skills/stress/SKILL.md` v7.3 + `stress_skill_replica.md` mirror: "present unified 3-AI bug list BEFORE any fix work, await user fix-scope confirmation". v7.2 Bug Table 3-column canonical format (Bug / Blast Radius / Launch 卡?) enforced; Source/OOB/Agreement encoded into `Pn-i-<suffix>` id column (A=Claude, B=codex, C=gemini, `*`=OOB). Cosmetic v6+v7.1 changelog/breadcrumb trim concurrently landed: Versioning section compressed (8 entries × multi-line → 9 entries × 1-liner), inline `(v6, 2026-05-14)` breadcrumbs removed from section headers (load-bearing dates kept for HHMMSS / parallel dispatch / v7.3 hard constraint), "Today's audit caught X" historical anecdotes deleted (motivation moved to commit messages / 实验笔记).

**Codex caught Claude completely missed**: B-131 (api_proxy revision inject — sibling-propagation gap from B-115), B-132 (multi-seed condition_id schema drift), B-134 (validate_action bool discard). **Gemini caught Claude+codex completely missed**: P0-1 (ref_images "leak" — turned out intentional + cost-fair), P0-2 (tool_calling prompt mutation), P2-1 (P-prompt header contradiction — turned out intentional manipulation). **Claude unique** (B-135 + B-136 + B-137 batch): max_new_tokens default, revision strict, temperature config metadata — all paper-grade reproducibility hardening invisible to single-AI review without explicit yaml-cross-code-cross-paper triple-grep.

#### Round counts (superseded — see consolidated Commit B finalized table above)

## §145 Pre_run/ folder residual audit (2026-05-15 late evening, user-requested)

User asked "看看 pre run 里面文件夹的内容,有没有不够是当前 2 paper 的?或者不正确的". Audit pass on `docs/checkpoints/pre_run/` 12 files surfaced 11 stale items across 2 root patterns (Batch 1-5 propagation missed protocol/license files): (1) paper-2 mechanism scope separation incomplete (mirage subsets + Stage 2 outputs still listed as paper-1 release scope); (2) §139.8 FP architecture retire banner not applied to protocol files.

**Bug-number collision note**: parallel session §144 used B-137 through B-132 concurrently with this session's §143 (B-117 through B-130) and this session's first commit body label "B-132/133/134" (commit `4bed340`). To stop the collision growing, this audit's catalog entries use B-150~B-152 (clear gap past B-132). Commit `4bed340` commit message body retains the original B-132/133/134 labels for git-history fidelity; reader should map "commit 4bed340 B-132" → "catalog B-150" / "commit 4bed340 B-133" → "catalog B-151" / "commit 4bed340 B-134" → "catalog B-152".

- **B-150** pre_run/ paper-2 mechanism scope separation propagation incomplete (= commit `4bed340` body label "B-132") — 5 files (release_redaction / ethics / locked_versions / model_card / osf_lock_manifest) 🛠️ **FIXED commit `4bed340`** — Batch 4b marked section5_mechanism.md paper-2 working draft + Batch 4 retired osf §2.1 mechanistic candidates, but 5 other pre_run/ files still listed mechanism §5 artifacts (mirage subsets / Stage 2 outputs / mechanistic claims) as paper-1 release scope. User decision 2026-05-15 (option α): completely archive existing mechanism artifacts, paper-2 will re-run with new data; paper-1 OSF DOI does NOT cite paper-2 forward artifacts. Cleanup: release_redaction mirage subsets moved EXCLUDED + ethics §3 paper §5 reproducibility deferred + locked_versions B1 section header "mechanistic + B1 baseline" → "paper-1 within-family local baseline" + model_card B1 modality "Stage 2 mechanistic" → "paper-2 future" + osf provenance table A100 promoted to paper-1 canonical / DGX → archive-only / A100 "Phase 2 mechanistic" row retired + osf §1 dangling "× 2 models × 6 modes" residual from Batch 4 edit cleaned.
- **B-151** `reeval_audit_protocol.md` §139.8 FP retire banner missing (= commit `4bed340` body label "B-133") 🛠️ **FIXED commit `4bed340`** — protocol body still referenced `adjusted_success` / `fp_reason` / `na_fp` / `eval_fp` rewrite_set entries (L28/29/35/42/64/81/96) as live re-derive fields. Added top-of-file banner: (a) `adjusted_success ≡ success` post-fix (B-91 patch), `fp_reason` no longer written, (b) protocol still applies to archive data re-derive (Appendix D contamination disclosure), (c) post-§139.8 canonical A100 rerun records evaluator SHA only. Body retained for backward compatibility with archive-era schema.
- **B-152** `evaluator_change_protocol.md` §139.8 FP retire banner missing (= commit `4bed340` body label "B-134") 🛠️ **FIXED commit `4bed340`** — protocol described §95 reform (post-hoc `compute_adjusted_success` + 3-layer `na_fp/eval_fp/visual_fp`) as current FP framework. Added top-of-file banner: current FP is source-level (B-91 evaluator patch + N/A excluded at task-load + program_html eval_fp branch dropped + visual_fp retired) + Tier classification (T0-T3) framework still valid for new evaluator changes + §95 reform retained for historical context.

P2 cosmetic items (PR-8/9/10/11 in audit Bug Table) folded into B-150 commit batch:
- `pre_rerun_audit.md` L38: 24-condition → 36-condition (PR-8)
- `ethics_license_coi_statements.md` L88: 16 cells → 36 conditions / 6 cells (PR-10)
- `ethics_license_coi_statements.md` L36-37: DGX + Tailscale-only → A100 canonical + DGX archive-only (PR-9)
- `osf_lock_manifest.md` L65-67: DGX → archive reference / A100 → paper-1 canonical promoted (PR-11)

### §145 audit status (2026-05-15 late evening, post §144): B-150 / B-151 / B-152 🛠️ FIXED · 1 advisor-pending Thread (3 / paper-scope-separation) NOT needed — user already decided option α (mechanism artifacts completely archive) · §145 retro check pending 7 days

**No cross-AI verification** — this audit was a Claude-only review pass requested by user. The 11 PR items were entirely propagation residuals (no new methodology / claim attacks needed), so cross-AI Mode B+C dispatch unnecessary. Bug Table was presented to user using SKILL v7.2 3-col format; user confirmed option α for mechanism artifacts; fixes landed in single commit.

**Pattern observation (cross §142 / §143 / §144 / §145)**: Each propagation batch surfaces a wider residual perimeter — Batch 1-5 covered prereg + paper drafts, §143 cross-AI surfaced 13 implementation-layer bugs, §144 cross-AI surfaced 7 more code/schema-layer bugs, §145 (Claude-only audit) surfaced 11 protocol/license-layer residuals. Lesson: full audit surface includes (a) prose / preregistration, (b) implementation / scripts / configs, (c) protocol / license / governance docs. Cross-AI Mode B+C covers (a)+(b); folder-pass Claude review covers (c). All three needed for paper-grade lock readiness.

---

### §146 — /stress A1.2 v8 `p79/backends/` second-pass 3-AI audit + Commit D+E (2026-05-16)

Continuation of §144 audit cadence: A1.1 (p79/agents/) wrapped, this round audits backend wrapper layer for post-A1.1 sibling-propagation drift + cross-baseline parity. Prior A1.2 round (commits `9316c8a` / `2a05e9a` / `50e10b3`) had landed F1-F6 + B-99; second-pass surfaced 10 new findings (5 OOB).

- **B-144** Multi-seed backend cache freezes first seed 🛠️ **FIXED commit `<TBD>`** — `runner/main.py:141-143` `_backends` cache key was `backend_id` only, so seed-loop `:333-338` updates to `self.seed` never propagated to cached agent's `agent_cfg["seed"]`. Multi-seed paper runs would produce distinct `condition_meta.seed` rows but identical model-side seed → mislabeled same-seed duplicates. Fix: key changed to `Dict[Tuple[str, int], Any]`; `_get_backend` builds `cache_key = (backend_id, int(self.seed))`. Each seed switch reconstructs backend with the correct seed forwarded. Source: codex Mode B P0-B1 (OOB).
- **B-145** GLM fallback default disabled + deprecation 🛠️ **FIXED commit `<TBD>`** — `configs/exp_v2_base.yaml:160` `use_glm_fallback: true → false` + `proxy_api_agent.py` GLM block "MARKED FOR FULL RETIRE" docstring + `DeprecationWarning` on enable. Cross-baseline cost-fairness violation: B1/B2 have no equivalent recovery model. User Q1 (2026-05-16) confirmed module slated for full retire pending Qwen official API channel (which exposes tool_choice + removes parse-error root cause). Source: codex Mode B P0-B2 (OOB).
- **B-146** Gemma → Qwen import decouple 🛠️ **FIXED commit `<TBD>`** — `gemma3vl_agent.py:11` `from p79.agents.qwen3vl_agent import Qwen3VLAgent, _wait_for_vram` pulled `qwen3vl_agent.py:7` `from qwen_vl_utils import process_vision_info` transitively → "B2-only" Gemma launch required Qwen deps. Mock/import tests didn't catch (lazy import inside `if not self.mock_mode:`); ImportError only surfaced at first real launch. Fix: new `p79/agents/_shared_vl_utils.py` module exports `make_dom_prompt` / `make_som_prompt` / `make_vision_prompt` / `format_history` / `compute_confidence` / `wait_for_vram` with minimal deps (torch + logging only). Gemma imports directly from shared; Qwen classmethods preserved as backward-compat delegates so all external callers (proxy_api_agent, mechanistic scripts, tests/test_agents_prompt_parity) keep working. Cross-baseline byte-identical prompts verified via new invariant test. Source: codex Mode B P1-B4 (OOB).
- **B-147** `api_proxy.py:28` max_new_tokens default 512 → 4096 🛠️ **FIXED commit `<TBD>`** — backend wrapper masked agent layer's 4096 default (B-135) with stale 512 → yaml-silent configs would silently truncate ~400-1500 tok thought+JSON envelope. All current paper-grade yaml explicit-set 4096 so no data污染, but defense-in-depth fails; future config refactor regression risk. Aligned with local_qwen/local_gemma wrappers + agent layer; invariant test asserts 3-wrapper alignment. Source: Claude inline F1-A (OOB).
- **B-148** `api_proxy.py` `api_key_env` allowlist guard 🛠️ **FIXED commit `<TBD>`** — yaml `api_key_env` could be overridden to any string (e.g. `AWS_SECRET_ACCESS_KEY`); subsequent verbose error trace or debug log would exfiltrate the secret to logs/stdout. Module-level `_ALLOWED_API_KEY_ENVS = {"PROXY_API_KEY", "DASHSCOPE_API_KEY", "GLM_API_KEY"}` + `_validate_api_key_env()` raises `ValueError` on non-allowlist value at backend init. Source: gemini Mode C P1-C4 (OOB).
- **B-149** api_proxy mock action aligned scroll 🛠️ **FIXED commit `<TBD>`** — `api_proxy.py:58-71` mock_mode returned `click element_id=1` while local_qwen/local_gemma/MockBackend all returned `scroll [0, 0.8]`. `factory.py:15` comment marked scroll as canonical. Mock parity invariant tests silently failed only on proxy path. Aligned to scroll. Source: Claude inline F3-A + gemini Mode C C5 (dual-AI catch).
- **B-153** Stale comment `local_qwen.py:27-28` realigned with B-136 🛠️ **FIXED commit `<TBD>`** — comment claimed "falls back to default + warns" but B-136 (Commit A) made agent RAISE `RuntimeError` on missing revision. Updated comment to reference B-136 strict mode. P2 cosmetic but stale comment misleads readers re: paper-grade revision pin enforcement. Source: Claude inline F2-A.
- **B-154** `factory.MockBackend.backend_type` baseline-tagged 🛠️ **FIXED commit `<TBD>`** — bare `"mock"` string couldn't distinguish mock_B0 vs mock_B1 vs mock_B2 in `type: mock` direct dispatch. Now `f"mock_{self.backend_id}"`. Source: Claude inline F6-A.
- **B-155** PIL version pinned ≥10 + import-time guard 🛠️ **FIXED commit `<TBD>`** — `image_utils.py:55` `Image.Resampling.LANCZOS` is PIL 9.0+ enum API; mixed PIL across DGX / Condenser / Myriad → different resize kernel + boundary padding → image bytes diverge → paper-grade bit-identical reproducibility impossible. `pyproject.toml` `pillow>=10.0,<12.0` + `image_utils.py` import-time `assert PIL.__version__ >= (10, 0)`. Source: Claude inline F7-A (OOB).

#### §146 audit status (2026-05-16 evening, Commit D + Commit E batch)

| Tag | Count | Notes |
|---|---|---|
| 🛠️ **FIXED Commit D** (P0 + P1 + P2-A2) | 7 | B-144 / B-145 / B-146 / B-147 / B-148 / B-149 / B-153 |
| 🛠️ **FIXED Commit E** (P2 polish) | 2 | B-154 / B-155 |
| ⚠️ **DISSOLVED by-decision** | 1 | P0-C2 model_calls hardcoded — GLM fallback retire makes 1 the truthful value |
| 📝 **DISCLOSURE 不修** | 1 | P1-B3 decoding override cascade — B-137 T=0 uniform + B1/B2 do_sample=False is reproducibility-by-design; paper §3.4 / methodology discloses |
| ⏳ **DEFER (Phase 2/3)** | 3 | P0-C1 HeuristicDomBackend 2-arg contract (Phase 2 router substrate) / P1-B5 dom_mode B2 propagation (Phase 2 router) / P2-A5+C3 stage_prefix triplication (Phase 3 M4 module) |

**Cross-AI verification chronicle**: Mode B (codex, ~3min, 5 findings / 3 OOB / reproducibility-auditor persona) — PASS Phase 1+2+3 sanity; Mode C (gemini, ~3min, 5 findings / 2 OOB / broad-reviewer persona) — PASS Phase 1+2+3 (tolerated read_file ignore-pattern + run_shell_command not found tool errors and still emitted substantive attacks). Cross-AI agreement: 1 dual-AI overlap (mock divergence — Claude F3-A + gemini C5 same finding); 13 unique catches (Claude 5, codex 5, gemini 3). Codex caught Claude completely missed: B-144 multi-seed cache (paper-grade blocker) + B-146 Gemma → Qwen import dep. Gemini caught Claude+codex completely missed: B-148 api_key_env exfiltration (security angle) + P0-C2 model_calls hardcoded (cost-fair / paper-disclosure angle).

**Skill v7.4 enforcement empirical**: User mid-flow asked for bilingual Blast Radius (round 1 pure English caused cognitive load) and a 推荐修改 section pre-empting "你打算怎么改" round-trip. Added to `.claude/skills/stress/SKILL.md` as v7.4 (HARD spec: bilingual MUST + 推荐修改 REQUIRED for all P0+P1). Replica synced. Driver: 2026-05-16 user feedback `用中英双语讲blast` + `增加一个：推荐修改`.

**B-number collision avoidance**: parallel session §145 used B-150 / B-151 / B-152; this round skipped those numbers, landing P2 polish on B-153 / B-154 / B-155.

**Pattern (cross §142 / §143 / §144 / §145 / §146)**: §146 confirms /stress is iterative — `p79/backends/` had a first-pass round 7 days prior (F1-F6 fixes) but second-pass after A1.1 land still surfaced 10 new findings (5 OOB) including paper-grade multi-seed cache blocker missed in first pass. Implication: each round of upstream changes (B-131~B-143) can produce new sibling-propagation gaps; periodic second-pass /stress is structural, not redundant.

---

### §147 — /stress A1.3 v8 `p79/envs/` second-pass 3-AI audit + Commit F (2026-05-16)

Continuation of /stress cadence: A1.2 (`p79/backends/`) wrapped, this round audits VWA env wrapper layer (`vwa_wrapper.py` 939 LOC + `locator_dispatch.py` 316 LOC). Prior A1.3 round (`131949b` / `9fb9b38` / `b1a31a2`) had landed F1-F6 + B-105~B-107 on locator_dispatch; this round surfaced 19 distinct findings (11 OOB), 6 fixes after user Q&A scope review.

- **B-156** locator-route telemetry → step_record 🛠️ **FIXED commit `<TBD>`** — `StepRecordV2.locator_route_meta: Optional[Dict[str, Any]]` field added; vwa_wrapper.py click + type branches stamp `_lr_result` (with `action_kind` discriminator) into `info["locator_route_meta"]`; runner step_record reads `next_info.get("locator_route_meta")`. Paper §3 evidence-layer audit of Cluster 1 (B-01/02/33) ON_TARGET rate previously structurally impossible from JSONL alone. Source: Claude F5-A + codex P2-B7 dual catch.
- **B-157** locator-route new-tab handling 🛠️ **FIXED commit `<TBD>`** — pre-fix: locator-route real click + `create_none_action()` skipped VWA framework's `num_tabs_now > num_tabs_before` tab-switch (browser_env/actions.py:1417-1421), so element-id clicks opening `target=_blank` / `window.open` left observation bound to the old page. Fix: snapshot `_num_tabs_before` before dispatch; on success detect new tab + `bring_to_front()` + `self._env.page = pages[-1]` mirroring VWA framework. Source: codex P1-B1 (OOB).
- **B-158** dialog handler context-level registration 🛠️ **FIXED commit `<TBD>`** — `_dialog_registered_page` (per-page) → `_dialog_registered_context` (per-context); `ctx.on("page", lambda new_page: new_page.on("dialog", _on_dialog))` ensures every new tab (including B-157 switched-to ones) inherits the auto-handler. Classifieds delete confirm() dialogs on new-tab paths no longer hang. Source: codex P1-B2 (OOB).
- **B-159** asyncio loop fail-loud guard 🛠️ **FIXED commit `<TBD>`** — `_lazy_init()` + `reset()` `_asyncio.get_running_loop()` no longer silently passes through on detection; raises `RuntimeError` with actionable message ("Run wrapper in subprocess; pytest-asyncio / notebook contexts must isolate"). Replaces cryptic "Sync API inside the asyncio loop" mid-init crash. Phase 1a queue scripts unaffected. Source: codex P1-B3 (OOB).
- **B-160** navigate_to URL injection vector closed 🛠️ **FIXED commit `<TBD>`** — `create_playwright_action(f'page.goto("{url}")')` → `create_playwright_action(f"page.goto({json.dumps(url)})")`. VWA `create_playwright_action` evaluates action_str as Python code against `page`; pre-fix any `"` in url broke out of the string literal → arbitrary Python code execution. `json.dumps` emits a safe Python string literal. Phase 1a callers trusted (URLs from config files); architectural hardening. Source: Claude F1-A (OOB).
- **B-161** Shadow DOM elementFromPoint penetration 🛠️ **FIXED commit `<TBD>`** — `_JS_SHADOW_DESCENT_FN` helper + `_pierceElementFromPoint(cx, cy)` recursively descends into shadow roots (depth ≤ 5) when first hit is a shadow host; 3 `_JS_RESOLVE_*` resolvers all use the pierced lookup. Walk-up adds `el.parentElement || el.getRootNode().host` to escape shadow boundaries. Reddit redesign / modern SPA / web components no longer fall back to framework bbox-center (B-33 buggy path). Phase 1a (cls/red 旧站非 shadow) 不变; Phase 1b cross-family / future site expansion 受惠。 Source: gemini C4 (OOB).

#### §147 audit status (2026-05-16 evening, Commit F batch)

| Tag | Count | Notes |
|---|---|---|
| 🛠️ **FIXED Commit F** | 6 | B-156 ~ B-161 |
| ⚠️ **DISSOLVED by-design** | 3 | C1 invisible turn (paper §3 semantics) / F2+B4 coord normalize (heuristic is intentional) / F3 inject_options regex (dict double-guard already covers) |
| ↓ **DEMOTED P2 latent debt** | 1 | C2 select stale coord (framework-level, all element_id actions, not select-specific) |
| ⏳ **DEFER P2 polish** | 9 | F4 horizontal scroll / F6 silent swallow / F7 fuzzy threshold / C3 asyncio hijack (Phase 2 future) / C5 hover/upload dead code / C6 P79Observation polymorphism / C7 select redundant env.step / P2-B5 form 200-byte truncation / P2-B6 native select empty-value filter |

**Cross-AI verification chronicle**: Mode B (codex, ~5min, 7 findings / 4 OOB / reproducibility-auditor) — PASS Phase 1+2+3; Mode C (gemini, ~6min, 7 findings / 3 OOB / broad-reviewer) — PASS Phase 1+2+3. Cross-AI agreement: 1 dual-AI overlap (Claude F5 + codex B7 telemetry), 18 unique. Codex caught Claude+gemini missed: B-157 new-tab loss + B-158 multi-tab dialog handler + B-159 asyncio passthrough (all mid-fire-rate paper-grade bugs). Gemini caught Claude+codex missed: B-161 shadow DOM penetration (cross-family expansion blocker) + C1/C2/C5/C6/C7 architecture/contract issues. Claude caught Claude-only: B-160 navigate_to injection (architectural security) + dropdown regex (turned out dict-double-guarded).

**User Q&A as audit layer**: 4 of my findings (C1 invisible turn / F2+B4 coord / F3 regex / C2 select coord) were challenged by user; 3 dissolved by-design and 1 demoted P2. User-level design intent review is **structurally distinct** from cross-AI consensus — cross-AI can amplify "this looks broken" but only user can answer "this is intentional". By-decision dissolution + demotion saved ~3-4h speculative fix effort.

**B-number coordination**: Continuous gap past §146 (B-155). Next available: B-162+.

**Pattern (cross §142 / §143 / §144 / §145 / §146 / §147)**: Three-pass /stress audit value confirmed — `p79/envs/` had first-pass 6 fixes 7 days ago, second-pass 6 new P1 fixes today, plus 3 dissolved by user review. Each pass surfaces a different layer: first-pass = correctness primitives (dispose / regex / ARIA roles), second-pass = cross-baseline parity + multi-tab + security architecture, user-layer = design intent vs accidental behavior. All three needed for paper-grade lock readiness.

#### §148 cross-system docker audit (2026-05-15 / 16, A100 self-host pre-fire)

3-AI stress (Claude self + codex Mode B + gemini Mode C with `--yolo` after permission misattribution retract). 16 findings consolidated → 15 act-on + 1 WONTFIX. P0×7 / P1×3 / P2×2 / P3×3. Full trace: `docs/checkpoints/process/cross_system_docker_audit_3way_diff_2026-05-15.md` (gitignored).

- **B-162** auto_login.py CWD-relative cookie write 🚧 **DOC-ONLY** (Claude NEW5) — `external/visualwebarena/browser_env/auto_login.py:62,82` write Playwright session state to CWD-relative `.auth/`. When invoked from `external/visualwebarena/` cwd (e.g. accidental cwd-switch), cookies land at `external/visualwebarena/.auth/` instead of repo-root `.auth/` where P79 runner expects. P79 hot path uses `p79/utils/auth_refresh.py` (writes absolute repo-root path) — so 0pp impact normally. Document warning: don't invoke `auto_login.py` directly; prefer `p79/utils/auth_refresh.py` for paper-grade cookie generation. Already hit once 2026-05-15 during A100 first smoke (manual cp upward to fix). Phase 1a 0pp; future-maintenance footgun. Source: Claude NEW5 in cross-system docker audit.
- **B-163** `_DEFAULT_BASE_URLS` fallback 🛠️ **FIXED commit `<TBD>`** (Claude NEW2) — `p79/utils/auth_refresh.py:23-28` had quark Tailscale `100.95.81.103` as fallback IP. CLAUDE.md hard rule #3 forbids bare `python run_experiment.py` (queue scripts mandatory, they source vwa_env_remote.sh setting env), so 0pp on hot path. But for safety changed defaults to `localhost` — loud-fail-on-unset (localhost not running anything on DGX) safer than silent-route-to-quark-prod. Single edit, ~5 min. Phase 1a 0pp expected, safety hardening. Source: Claude NEW2.

---

### §149 — /stress A1.4a v8 `p79/experiment/` orchestrator core 3-AI audit + 4-commit G1-G4 (2026-05-16)

phase1_plan A1.4 拆 3 chunks (A1.4a orchestrator / A1.4b data plane / A1.4c auxiliary), 本 § = **A1.4a only** (5 file / 2521 LOC: `runner/main.py` 1695 + `runner/helpers.py` 299 + `router.py` 149 + `conditions.py` 208 + `tasks.py` 170)。Pre-fire scope (≥7 findings / ≥3 OOB)。3-AI cycle: Mode A Claude 8 finding/4 OOB, Mode B codex repro-auditor 8/5 OOB (5 min), Mode C gemini broad-reviewer 7/4 OOB (3 min); 23 raw → 21 distinct (4 dual-catch), user Q&A 5 dissolve / 4 demote-defer / 1 expand-scope。

- **B-164** backend cfg deep copy 🛠️ **FIXED commit `05ff344`** (codex B5, OOB) — `p79/experiment/runner/main.py:165` `dict(backend_cfg)` → `copy.deepcopy(backend_cfg)`. Pre-fix shallow copy shared nested ``generation`` / ``model_kwargs`` / ``headers`` dicts/lists with `self.cfg`, so constructor side effects mutated shared state across (condition, seed) iterations through `self.cfg`. Symptom: seed=42 single-run ≠ seed=[42,43] runs[0] even though B-144 cache key tuple was correct. Renumbered from working-name B-162 after §148 parallel-session collision (see §148 B-162 above for unrelated `auto_login.py` issue).
- **B-165** fallback_finish reward override guard 🛠️ **FIXED commit `05ff344`** (Claude F2 + codex B3 dual-catch, OOB) — `runner/main.py:1562-1585` reward override now requires `_real_finish` guard (`parse_valid AND not fallback_finish`). Pre-fix the keyword-rescue 'finish' (fallback_finish=True, parse_valid=False) ALSO triggered score=0→1 override because action_type literally == 'finish'. Cross-baseline SR contamination: B0 235B rarely needs keyword rescue, B1/B2 4B frequently does → B1/B2 SR systematically inflated 1-3pp by override differential. Renumbered B-163 → B-165 (collision avoidance).
- **B-166** trajectory_incomplete telemetry 🛠️ **FIXED commit `05ff344`** (Claude F4, OOB) — `runner/main.py:1545-1568` adds `trajectory_incomplete` flag (True when fake stop action appended for missing finish); episode_summary stamps it as transparency metric. Episodes that exhaust max_steps without explicit `finish` receive empty `answer=""` → string_match VWA evaluator scores 0 regardless of agent capability. B1/B2 4B baselines time out far more than B0 235B → cross-baseline SR rank contains a timeout-rate confound. Path A disclosure (SR canonical, no adjustment; paper §3.5 will report `trajectory_incomplete_rate` per cell). Renumbered B-164 → B-166.
- **B-167** invalid_action 7-category 细分 + unknown_failure bucket 🛠️ **FIXED commit `10b6e4c`** (Claude F3 expanded scope, OOB) — new `validate_action_detailed(action) → (action, valid, reason)` in `p79/backends/action_utils.py` emits sub-category reasons {invalid_schema_dict, invalid_action_type, invalid_element_id, invalid_coord, invalid_select_option}; backward-compat `validate_action` 2-tuple wrapper retained (20+ callers unchanged). `parse_action_text` Path 1 (clean JSON) propagates detailed reason. `_normalize_error_category` rewritten from 5 → 10 categories + `unknown_failure` future-proof bucket (was silently invalid_action catch-all). Episode_summary stamps `unknown_failure_reasons: Dict[str, int]` Counter (filtered by error_category=='unknown_failure'). Paper §3.5 cross-baseline error taxonomy now informative. Router-aware escalation policy (per-category target mode mapping: invalid_element_id → SoM / invalid_coord → vision) deferred to Phase 2 / paper-2 scope. Coord-present-but-malformed priority beats invalid_element_id (specific reason wins). 2 existing `tests/test_action_utils.py` cases updated for new reasons. Renumbered B-165 → B-167.
- **B-168** partial-step crash → JSONL aggregate recovery 🛠️ **FIXED commit `1495a6e`** (codex B1, OOB) — `runner/main.py:618-684` `_run_and_record_episode` except path now calls `read_jsonl_dedup(_jsonl_path)` + new helper `ExperimentRunner._aggregate_partial_steps(step_records)` to compute steps/tokens/cost/latency/retries/no_op/page_unchanged from JSONL rows already written before mid-episode crash. Pre-fix: 12-step episode that crashed at step 13 → outer except wrote `steps=0,total_cost=0` even though 12 step JSONL rows were already on disk → same-episode JSONL-vs-summary truth split in paper §3 evidence layer. Summary now includes `trajectory_incomplete=True` + `partial_recovery_step_count=N`. Renumbered B-166 → B-168.
- **B-169** resume identity tuple check + quarantine 🛠️ **FIXED commit `1495a6e`** (codex B2, OOB) — `runner/main.py:367-432` resume gate now: (a) load summary, (b) call `ExperimentRunner._validate_resume_identity(loaded, expected)` against 6-field tuple `(schema_version, run_id, condition_id, seed, benchmark_site, task_id)`, (c) mismatch → quarantine to `<episodes>/quarantine/<basename>.<ts>.json` + log warning + fall through to re-run, (d) match → existing acceptance logic. Pre-fix: any file at the expected path was accepted, so output_root reuse with changed `run_id`/`seed`/`include_sites` silently ingested stale summaries (yesterday's run_42 ingested into today's run_43 aggregate). Quarantine via `shutil.move` for cross-fs safety. Renumbered B-167 → B-169.

#### §149 audit status (2026-05-16 Commit G1-G3 batch)

| Tag | Count | Notes |
|---|---|---|
| 🛠️ **FIXED G1** | 3 | B-164 / B-165 / B-166 — backend deepcopy / fallback_finish guard / trajectory_incomplete |
| 🛠️ **FIXED G2** | 1 | B-167 — invalid_action 7-category 细分 + unknown_failure bucket |
| 🛠️ **FIXED G3** | 2 | B-168 / B-169 — partial-step JSONL aggregate / resume identity check |
| ⚠️ **DISSOLVED by-design** | 5 | D4 dom_size cross-mode (empirical 1.00× ratio) / D2 phantom som_on (paper §3 by-design) / D2 Phase 2 Frankenstein (paper-2 defer) / C1 escalation monotonicity (cost-aware by-design) / C5 heuristic injection assertion (default-off + git review) / C7 silent escalation reset (code path unreachable) |
| ↓ **DEMOTED P2 latent** | 3 | D3 task ordering manifest (empirically stable order) / F7 zero-step watchdog race (race window small) / B7 placeholder provenance (resolved JSON on disk) |
| ⏳ **DEFER** | 4 | F8 cycle min_reps (advisor-canceled early-stop, reporting-only) / B4 schema v2.0 drift (bump at paper revision) / B8 analysis timeout status (warning log sufficient) / C4 phase 3 synergy ceiling (paper-2 deferred) |
| 🔬 **Mini-investigation** | 1 | F1 about:blank systematic study — `_status/issues/issue_about_blank_systematic_2026-05-16.md` + `scripts/analysis/about_blank_frequency.py` (no runner patch yet) |

**Cross-AI verification chronicle**: Mode B (codex repro-auditor, ~5min, 8 findings / 5 OOB) — PASS Phase 1+2+3 (file 2.4MB inflated by codex working log capture; actual audit lines 1-327 ≈ 25KB clean); Mode C (gemini broad-reviewer, ~3min, 7 findings / 4 OOB) — PASS Phase 1+2+3. Cross-AI agreement: 4 dual-catch overlaps {fallback_finish reward override / som_on (different sites) / task ordering / dom_size cross-mode}, 17 unique. Codex 独家 catch: schema v2.0 drift / shallow config copy / partial-step JSONL recovery / resume identity / placeholder provenance / analysis subprocess timeout (all reproducibility-layer paper-grade bugs Claude+gemini systematically missed). Gemini 独家 catch: escalation monotonicity / trigger collinearity inflation / phase 3 synergy gap / heuristic injection (all design-layer issues).

**User Q&A as audit layer**: 6 user questions changed fix list: D1 fallback_finish confirmed (Q1) / D2 phantom som_on dissolved by-design (Q2) / F1 changed to mini-investigation path not direct patch (Q3) / F3 scope expanded 4× from "add unknown_failure bucket" to "7-category 细分 + upstream validate_action_detailed emission" (Q4) / F4 disclosure path A confirmed (Q5) / D4 dom_size cross-mode dissolved by empirical fact — Claude+gemini both held wrong "AXTree → flat -50%" mental model, user spot-check showed dom=3804 / som=3879 / vision=3805 char ratio 1.00-1.02× (Q6). 21 finding 中 5 dissolve (24%) per user, 1 scope-expand. Saved ~5-7h speculative work; gained ~2-3h scope-expand for paper §3.5 value.

**B-number coordination + collision pattern**: §148 parallel session reserved B-162 / B-163 for cross-system docker audit (commit `b04f7b2`). My G1/G2/G3 commit messages and initial code referenced B-162~B-167 (per §147 "Next available: B-162+"). Renumbered to B-164~B-169 via sed after collision detected: `s/B-167/B-169/g; s/B-166/B-168/g; s/B-165/B-167/g; s/B-164/B-166/g; s/B-163/B-165/g; s/B-162/B-164/g` (reverse order to avoid double-replacement). Commit-message history (`05ff344`, `10b6e4c`, `1495a6e`) references old "B-162~B-167"; in-tree code + tests + this catalog entry use corrected B-164~B-169. **Lesson**: B-number reservation via §-section is not canonical until catalog commits land. Cross-session parallel work pattern (also hit §145 with B-150/151/152 collision).

**Pattern (cross §142-§149)**: orchestrator audit blast radius significantly higher than backends/envs/agents because `p79/experiment/runner/` is the paper §3 evidence-layer trunk. 3-AI + user Q&A 4 layers needed for full coverage: Claude finds silent attribution + fallback override; codex finds reproducibility-layer (JSONL/summary divergence, schema drift, shallow copy, identity check); gemini finds design-layer (cost-aware framing, control-flow contracts); user finds design intent (dissolves AI-misread "bugs"). Single-AI audit would have missed >50% of actionable findings.

**A1.4a fully closed**. A1.4b-i (analysis.py canonical paper §3 producer, 1557 LOC) covered next; see §150 below.

---

### §150 /stress A1.4b-i — `p79/experiment/analysis.py` paper §3 canonical (2026-05-16)

- **B-170** McNemar/Wilcoxon merge key cross-site collision 🛠️ **FIXED commit `60e6ce5`** (Claude A1, OOB) — `p79/experiment/analysis.py:851-853` `merge(df_a, df_b, on="task_id")` cross-paired tasks across cls/red/shop because empirical task_id ranges all overlap [0,209] (verified via `wc -l` + `python -c json.load` over `test_*.raw.json`). Fix: `on=["benchmark_site", "task_id"]` when site column present; spot-check unit test verifies 2×2 site×task → 4 rows (broken) → 2 rows (fixed).
- **B-171** "(adjusted)" stale prose remnant 🛠️ **FIXED commit `60e6ce5`** (Claude A5 + gemini C1, cross-AI) — 4 plot titles in `analysis.py:545, 1267, 1300, 1346` said "(adjusted)" but §139.8 retired the post-hoc adjustment layer. Replacement: "(N/A excluded at task-load)". `_plot_phase1` headline also gets "partial conditions hatched (//)" suffix when B-179 hatched-bar branch fires.
- **B-172** Wilcoxon skip silent-CSV-row 🛠️ **FIXED commit `60e6ce5`** (Claude A6) — `analysis.py:908-910` `continue` on `len(common_idx)<5` only wrote to `results["notes"]`, leaving `statistical_tests.csv` row-less; downstream "no diff" reading silent. Fix: emit CSV row with `p_value=None` + `skipped_reason="insufficient_paired_samples_n{N}"` so auditors can re-discover skipped pairs.
- **B-173** Pareto strict-< tie semantics 🛠️ **FIXED commit `60e6ce5`** (Claude A8 + gemini C5, cross-AI; doc-only) — `_compute_pareto_front:89-106` uses `val < best_min` which silently drops tied points (same max + same min). Standard Pareto definition includes both. Edge case at observed N≥24 cells; fix is docstring "ties broken by sort order" + paper figure caption disclosure (no code change).
- **B-174** `_to_mapping` silent JSON parse swallow 🛠️ **FIXED commit `60e6ce5`** (Claude A9, OOB) — `analysis.py:240-250` `except: return {}` made malformed `trigger_distribution` / `state_change_reason_distribution` look like "no event fired" in pivot tables. Fix: log warning + append to `_TO_MAPPING_PARSE_FAILURES` collector; `analyze_run` emits `analysis/parse_failures.csv` so audit sees dropped values. `_flatten_*` callsites pass `context=` for traceability.
- **B-175** TOST column inversion-trap label 🛠️ **FIXED commit `60e6ce5`** (codex B4) — `aggregate_phantom_lift.py:913` markdown column "TOST sig (0.05)" placed next to "sig (Holm 0.05)" invited the exact inversion the formula prevents (TOST sig = equivalence accepted, NOT lift significance). Rename to `equiv_within_1pp` + disambiguating footnote.
- **B-176** Bootstrap RNG seed disclosure 🛠️ **FIXED commit `60e6ce5`** (codex B9) — `analysis.py:816` seed=42 + B=10_000 pinned but not disclosed in paper §3.5 prose. Inline comment added; paper §3.5 prose follow-up tracked.
- **B-177** Phase 2 net-saving 漏算 `obs_prepare` cost 🛠️ **FIXED commit `824e55a`** (codex B1, OOB) — `analysis.py:1538-1543` `_plot_phase2` reconstructed `routed_total_cost = avg_total_model_cost_usd + avg_router_overhead_cost_usd` (2 components) but runner cost decomposition is `total = model + router_overhead + obs_prepare` (3 components, `runner/main.py:1839`). Every Phase 2 net-saving JSON / CSV biased UPWARD by exactly the routed obs-prepare cost. Fix: use canonical `avg_total_cost_usd` directly + emit 4-way decomposition with new `routed_obs_prepare_cost` field; sanity invariant logs warning if components don't sum back to canonical total within 1e-9 USD.
- **B-178** Holm-Bonferroni step-down in `_compute_statistical_tests` 🛠️ **FIXED commit `824e55a`** (Claude A2 + gemini C2, cross-AI) — `analysis.py:794-844` previously emitted only raw pairwise p-values; paper §3.6 prose claims Holm-corrected paired tests so pre-fix any reader pasting these into the paper inflated FWER for 36 conditions × 630 pairs. Add `_holm_correct` step-down helper applied within each (test, metric) sub-family (McNemar success / Wilcoxon cost / Wilcoxon latency are separate families); emits `p_value_holm`, `significant_05_holm`, `holm_family`, `holm_family_m` columns + `holm_corrected.families` JSON metadata. Raw p-values preserved for transparency.
- **B-179** `_synthesize_condition_summary` schema-incomplete + `or 0` silent None→0 🛠️ **FIXED commit `de85d5a`** (Claude A3 + codex B6 + gemini C4, P0 triple-cross) — pre-fix `_synthesize_condition_summary:126-228` hand-aggregated with `e.get(k, 0) or 0` (B0 cost / energy systematic underestimation) AND was missing `avg_total_latency_ms`, `avg_obs_prepare_cost_usd`, `avg_input_cost_usd`, `avg_output_cost_usd`, `avg_busy_wait_total_ms`, `energy_partial_*` fields, AND hard-zeroed `avg_router_overhead_cost_usd` / `wasted_*` / `benchmark_noise_rate`. Headline `_plot_phase1` consumed mixed partial+complete rows with no visual distinction. Fix: synth delegates to `aggregate_condition_metrics(ep_summaries)` (the runner's canonical aggregator); `_synthesized=True` flag drives `//` hatch on partial bars in `_plot_phase1`.
- **B-180** `read_jsonl_dedup` summary-aware identity check 🛠️ **FIXED commit `de85d5a`** (codex B7) — `io_utils.py:read_jsonl_dedup` "last step_idx=0 wins" heuristic didn't validate against sibling summary. Restart-crash (new segment writes step_idx=0 then crashes before summary overwrite) silently kept partial segment while summary still pointed at old run. Fix: optional `summary_path=` kwarg validates 6-field identity tuple `(schema_version, run_id, condition_id, seed, benchmark_site, task_id)` + steps cardinality; mismatches log warning (no raise) so analysis still proceeds. `_collect_step_records` derives summary_path from `*_steps_v2.jsonl` ↔ `*_summary_v2.json` naming.
- **B-181** Mixed-phase fail-closed validator 🛠️ **FIXED commit `3f83a52`** (codex B5) — `analyze_run` post-cond_df construction now classifies every `condition_id` prefix into {phase1, phase2, phase3, unknown}; >1 phase detected → write `analysis/phase_mix_warning.txt` + drop out-of-phase rows from cross-cond plots so downstream `_plot_phase{1,2,3}` see single-phase clean cond_df. Pre-fix: re-launching Phase 2 into Phase 1 dir silently mixed flat arms into Pareto plots.
- **B-182** Holm family scope labels in `aggregate_phantom_meta` 🛠️ **FIXED commit `3f83a52`** (codex B3) — `aggregate_phantom_meta.py:287` table header now has `family_scope` (APPENDIX_RE_SENSITIVITY_m{1,2,3}) + `gating_status` (appendix-only / exploratory) columns. Pre-fix prose "Pre-registered family gating" with Holm over SECONDARY pooled tests (m=3) contradicted `preregistration.md:292-320` PRIMARY family m=1 (FE superiority, paper PRIMARY actually lives in B-184 missing `phase1_prereg_gate.csv`). Disclosure paragraph added: RE meta in this script is appendix sensitivity, paper gate is elsewhere.
- **B-183** Per-episode P95 latency figure caption disclosure 🛠️ **FIXED commit `3f83a52`** (Claude A7 half-defused) — `_analyze_condition:604-617` "Latency Distribution" histogram caption pre-fix did not disclose that each x-axis value is per-episode P95 (NOT per-step). `metrics.py:344-347` docstring already explained this but figure was opaque. Fix: caption now reads "Per-episode P95 step latency (s) (NOT per-step distribution)" + title prefixed "Per-episode P95 Latency Distribution".

#### §150 NEW ARTIFACTS — partial close (B-184 landed; B-185/B-186 still deferred)

- **B-184** `phase1_prereg_gate.{csv,json,md}` 🛠️ **FIXED** (codex B2, P0 OOB) — canonical FE drop-one over 6 cells + one-sided z superiority test. New producer `scripts/analysis/aggregate_phase1_prereg_gate.py` implements prereg §1 H1 PRIMARY (line 68-86 lock): per-cell drop-one θ_i with paired B=1000 bootstrap SE (seed=42 per B-176), FE pool `w_i=1/SE_i²` → `θ_FE = Σw_iθ_i/Σw_i`, `SE_FE = sqrt(1/Σw_i)`, one-sided `z = (θ_FE − 1.0) / SE_FE`, reject H0 when `p_one_sided < 0.05`. Emits 3 files (CSV per-cell + pooled / JSON metadata / Markdown paper-citable table). Wired into Makefile `_aggregate` chain (runs BEFORE `phantom-lift`); `phantom-lift` retained as appendix-exploratory. Pre-data behavior: `gate_status="INSUFFICIENT_DATA"` (no cell has all 6 modes yet, Phase 1a rerun in flight); producer gracefully degrades, does not block `make analysis`. 17 unit tests verify: ϕ_CDF accuracy / per-task drop-one indicator (oracle_6 ⊇ oracle_5_no_psom invariant) / bootstrap determinism with seed pin / SE scales with √n / FE pool arithmetic at equal weights → mean / inverse-variance weighting pulls toward low-SE cell / z=0 at θ_FE=δ → p=0.5 not pass / k=1 returns None / partial-data status / writers round-trip.
- **B-185** `claim_manifest.json` ⏳ **DEFER** (codex B8, P1, 0.5-1 day scope) — paper-claim → producer + input SHA mapping; depends on B-184. Same issue file.
- **B-186** `hero_metrics.json` ⏳ **DEFER** (gemini C3, P1, 0.5 day scope) — single JSON collecting 4-fold drop-in property (a)cost (b)latency (c)AUROC (d)drop-one for P-SoM. All 4 metrics already produced but scattered. Same issue file.

#### §150 audit status (2026-05-16 Commit G1-G4 batch)

| Tag | Count | Notes |
|---|---|---|
| 🛠️ **FIXED G1** | 7 | B-170 ... B-176 — paper §3 reviewer-3 trap fixes (task_id merge / "(adjusted)" prose / Wilcoxon CSV row / Pareto doc / `_to_mapping` warn / TOST column / RNG seed) |
| 🛠️ **FIXED G2** | 2 | B-177 / B-178 — Phase 2 obs_prepare drop + Holm in `analyze_run` |
| 🛠️ **FIXED G3** | 2 | B-179 / B-180 — synth canonical + JSONL identity check |
| 🛠️ **FIXED G4** | 3 | B-181 / B-182 / B-183 — phase fail-closed + Holm family labels + P95 caption |
| 🛠️ **FIXED (new artifact)** | 1 | B-184 — phase1_prereg_gate (canonical paper §1 H1 PRIMARY gate; producer + 17 tests landed; live status INSUFFICIENT_DATA pending Phase 1a data) |
| ⏳ **DEFER (new artifact build)** | 2 | B-185 / B-186 — claim_manifest / hero_metrics (1-1.5 day remaining scope; A1.4b-i follow-up issue) |

**Cross-AI verification chronicle**: Mode B (codex repro-auditor, ~5-10min, 9 findings / 2 explicit OOB + 4 borderline + 3 strong-claims anti-attacks confirming TOST formula + DL τ² spotcheck + sibling propagation isolation) — PASS Phase 1+2+3 (18.7KB output, all proper structure). Mode C (gemini broad-reviewer, ~3min, 5 findings / 3 OOB, structural caveat C1 header truncated — content recoverable from "Key Findings Overview") — PASS Phase 1+2+3 with structural caveat. Cross-AI overlap matrix: 4 cross-validated (A2/C2, A4/C3, A5/C1, A8/C5), 5 Mode A unique (A1/A6/A7/A9 + parts of A3), 8 codex unique (B1/B2/B3/B4/B5/B7/B8/B9), 1 gemini unique (C4). Codex 独家 catch P0 OOB: B1 Phase 2 obs_prepare drop (paper §3.6 numerical leak) + B2 estimand drift (paper H1 hero uses wrong estimand). Gemini unique catch: C4 partial/complete condition mixing (Claude A10 candidate that I had downgraded; gemini elevated correctly).

**User Q&A as audit layer**: User accepted ALL findings with "其他同意" + "历史归档不需要处理" (no regenerate of old statistical_tests.json) + "make analysis看看还缺少什么" (inventory check for new artifacts). Spot-check inventory found 3 paper-grade canonical artifacts MISSING (B-184/B-185/B-186) — flagged for follow-up issue (0.5-3 day scope, not tonight). Tonight scope = 14 code-fixes across 4 commits.

**A1.4b-i scope confirmed (split-large-scope rule)**: total = 2463 LOC / 7 files in `p79/experiment/` data plane. Split into A1.4b-i (analysis.py 1557 LOC standalone, paper §3 critical) + A1.4b-ii (logger_v2 / io_utils / types / metrics / schema_migrations, 906 LOC). A1.4b-i fully closed; A1.4b-ii queued. A1.4c (auxiliary modules) also queued.

**Pattern (cross §142-§150)**: paper §3 evidence-layer canonical producer (`analysis.py`) blast radius higher than orchestrator core because every paper figure / table / prose claim flows through here. Cross-AI value compounded: 3 AIs caught 17 unique findings vs single-AI ≤9. Codex reproducibility-auditor persona empirically high-value for stats/analysis scripts (caught 8 vs 5 from Claude). Gemini broad-reviewer persona high-value for paper ↔ code mismatch (caught C1 prose drift that I had not surfaced as paper-grade). User Q&A continues to be necessary 4th layer (saved ~6-9h on B-184/B-185/B-186 by punting from "fix now" to "issue tracker").

**Next available B-number**: ~~B-187+~~ — superseded by §152 below (B-187 ~ B-200 landed).

---

### §152 /stress A1.4b-ii — `p79/experiment/` data plane (2026-05-16 evening)

- **B-187** delete dead `compute_energy_step` 🛠️ **FIXED commit `2c2bd41`** (gemini v5 #1 + codex B-ii-5 + user spot-check, P2) — 0 production callers. Real CO2 pipeline via `LightweightEnergyTracker.estimate_step` (`runner/main.py:1472`); empirical co2e/kwh = 0.22 kg/kWh = UK 220 g/kWh verified across 5 episode samples. Deleted helper read wrong YAML key (`co2e_kg_per_kwh: null` vs `carbon_intensity_g_per_kwh: 220`).
- **B-188** delete dead `compute_waste_breakdown` + dead `adjusted_success=` kwarg 🛠️ **FIXED commit `2c2bd41`** (gemini v5 #4 + codex B-ii-8 + Claude D5, P2) — 0 callers, math invariant violation (success episode wasted=0 but no_op/page_unchanged unconditionally summed). Per user "选 A": keep `compute_wasted_cost` binary as canonical, fine-grained richer impl in `analyze_reason_diagnostics.py`. §139.8 `adjusted_success` retired.
- **B-189** paper §3.5 seed=42 + B=1000 prose disclosure 🛠️ **FIXED commit `2c2bd41`** (gemini v1 G5, P2 OOB) — names 3 sharing scripts (`_compute_statistical_tests`, `aggregate_phantom_lift`, `aggregate_phase1_prereg_gate`).
- **B-190** `STEP_RECORD_V2_DEFAULTS` catalog 🛠️ **FIXED commit `add14ed`** (Claude D1 + codex B-ii-1, P0 OOB) — mirrors EpisodeSummary catalog; 5 paper-grade-critical step optionals (`parse_valid` / `parse_failure_reason` / `fallback_finish` / `image_meta` / `locator_route_meta`) explicit defaults; `fill_step_defaults()` helper.
- **B-191** `schema_migrations.migrate` `deepcopy(record)` 🛠️ **FIXED commit `add14ed`** (Claude D3, P0 OOB) — same defensive class as B-164. Nested dicts isolated from caller.
- **B-192** wire `fill_defaults` into `_collect_episode_summaries` 🛠️ **FIXED commit `add14ed`** (Claude D2, P1 OOB) — framework not dead infra anymore. Legacy summaries get baseline-typed defaults per user "历史归档" decision.
- **B-193** trajectory telemetry → EpisodeSummaryV2 + aggregator emit rates 🛠️ **FIXED commit `d7850bb`** (codex B-ii-2, P1 OOB, **biggest catch**) — A1.4a B-166/B-167/B-168 runner-stamped telemetry was never aggregated. `EpisodeSummaryV2` + defaults catalog + `aggregate_condition_metrics` now emit `trajectory_incomplete_rate` / `partial_recovery_rate` / `unknown_failure_reason_distribution`. Paper §3.5 transparency claim now structurally producible.
- **B-194** exception path `wasted_cost = total` not 0 🛠️ **FIXED commit `d7850bb`** (codex B-ii-3, P1 OOB) — force-zero was inconsistent with `compute_wasted_cost(success=False)` canonical semantic + B-168 recovered partial cost wasted in vain.
- **B-195** obs_prepare cost field comment 🛠️ **FIXED commit `fdd06cd`** (gemini v1 G1, P1 OOB) — aggregator now annotated; step-level latency pivot is in `step_metrics.csv` (B-195b emit median+p95 obs-prepare ms deferred pending paper §3 decision).
- **B-196** JSONL integrity report 🛠️ **FIXED commit `fdd06cd`** (codex B-ii-4, P1) — `_JSONL_INTEGRITY_LOG` module-level counter; `analyze_run` emits `analysis/jsonl_integrity_report.csv` with corrupt_lines + dedup_discarded + summary_identity_mismatch per file. Closes paper §3 denominator transparency gap.
- **B-197** `cost_efficiency_ratio` → None when no cost 🛠️ **FIXED commit `fdd06cd`** (Claude D4 + gemini G4, P1) — B1 local condition (cost=0) no longer silently emits "0% efficiency".
- **B-198** `logger_v2._fsync_dir` 3 callsites 🛠️ **FIXED commit `fdd06cd`** (Claude D6, P1) — dir entry fsync after `os.replace` (best-effort, swallows OSError on unsupported FS).
- **B-199** `detect_benchmark_noise` + 2 categories + category distribution 🛠️ **FIXED commit `fdd06cd`** (codex B-ii-7 + gemini G3, P1/P2) — `api_rate_limit` + `auth_expired_or_session_invalid` 加入 (forward-risk), `ERR_CONNECTION_REFUSED` 重排到 connection_error 之前避免 navigation_error shadow. Aggregator emits `benchmark_noise_category_distribution`.
- **B-200** `p95` filters None + NaN 🛠️ **FIXED commit `fdd06cd`** (codex B-ii-6, P2) — pre-fix `p95([None, None, 0, 0])` raised TypeError; `p95([NaN, 1, 2])` silently returned 1.9. Now: filter, return 0.0 on empty valid.

#### §152 audit status

| Tag | Count | Notes |
|---|---|---|
| 🛠️ **FIXED G1** | 3 | B-187/B-188/B-189 — dead-code cleanup |
| 🛠️ **FIXED G2** | 3 | B-190/B-191/B-192 — schema integrity |
| 🛠️ **FIXED G3** | 2 | B-193/B-194 — trajectory telemetry aggregator + exception wasted fix |
| 🛠️ **FIXED G4** | 6 | B-195~B-200 — defensive validators (obs_prepare / JSONL integrity / cost_eff None / dir fsync / noise categories / p95 policy) |
| ⏳ **DEFER (analyzer source-of-truth promotion)** | 0 | wasted_cost richer breakdown stayed in `analyze_reason_diagnostics.py` (B-188 user选A) |
| 🟢 **PASSED defuse** | 3 | codex B-ii-5 (compute_energy_step 0 caller) / B-ii-8 (compute_waste_breakdown 0 caller) / B2 cost split (current 0 mixed-backend run dirs) |

**Cross-AI verification chronicle**: Mode A (Claude inline /stress, 7 findings 3 OOB) — PASS. Mode B (codex repro-auditor, 8 findings + 3 strong-defuse + 3 quantification sections, 16.3KB) — PASS. Mode C (gemini broad-reviewer) — chatter failure on direct dispatch; **fixed via Path C wrapper** (commit `202421d` `scripts/maintenance/gemini_stress_clean.sh` + 3-rule prompt discipline). gemini v5 wrapper-clean dispatch surfaced 4 findings 2 OOB; 2 of those (CO2e key mismatch / wasted_cost paradox) downgraded to P2 after user "carbon 我记得是有计算的吧" spot-check confirmed both target dead code.

**B-numbers consumed**: B-187 through B-200 (14 contiguous, no collisions).

**Next available B-number**: B-201+. B-185/B-186 still reserved by `_status/issues/issue_phase1_canonical_artifacts_2026-05-16.md` follow-up issue but not yet built.

---

## A1.4c /stress audit (2026-05-16) — B-202 to B-210 (10 entries)

### B-202. `_extract_focused_tag` regex empirically dead on real AXTree 🛠️ FIXED
- **Source**: A1.4c Mode A Finding 2 (Claude)
- **Code**: `p79/experiment/state_change.py:32-37` (DELETED 2026-05-16)
- **Attack**: Regex `r"focused.*?\b(\w+)(?:\s|$)"` requires a word char AFTER `focused`, but real AXTree puts `focused` at line end (`[14] textbox 'Search' focused`). Empirical: 30/30 production step records in `B2_dom_reddit_20260516/.../reddit_task_2_steps_v2.jsonl` had `active_element_tag = None`.
- **Test false-positive**: `test_build_page_state_with_focused` used synthetic `"focused input"` string (not real AXTree) to make regex match — classic "test validates with synthetic data, production never triggers" anti-pattern.
- **Fix**: Deleted `_extract_focused_tag` + 4 dependent tests + `active_element_tag` field emission (see B-204).

### B-203. `apply_som` deprecated 0-caller dead code 🛠️ FIXED
- **Source**: A1.4c Mode A Finding 11 (Claude)
- **Code**: `p79/experiment/som.py:407-432` (DELETED 2026-05-16)
- **Attack**: `apply_som` was 0-caller in production (grep confirmed only `tests/test_som_and_schema.py` referenced it). F6 had earlier added a DeprecationWarning expecting a forgotten caller to surface, but the only "caller" was the test that verified the warning fired — circular dead code.
- **Fix**: Deleted function + `test_apply_som_emits_deprecation_warning` + replaced `test_som_degrades_without_bbox` to use `prepare_observation_for_mode` (canonical path).

### B-204. `active_element_tag` field structurally always None in production 🛠️ FIXED
- **Source**: A1.4c Mode A Finding 2 (Claude, downstream of B-202)
- **Code**: `p79/experiment/state_change.py:87` (DELETED 2026-05-16) + downstream
- **Attack**: Field emitted by `build_page_state` but no `scripts/analysis/` consumer (grep confirmed); test was the only reader, and `_extract_focused_tag` always returned None on real AXTree (B-202). Dead-field exposure.
- **Fix**: Removed from `build_page_state` state dict; updated 3 tests (test_external_module_integration, test_state_change).

### B-205. Wrapper `--approval-mode plan` silently degrades audit on shell-read prompts 🛠️ FIXED
- **Source**: A1.4c Q4 empirical 3-trial cross-AI test
- **Code**: `scripts/maintenance/gemini_stress_clean.sh:93` (REWRITTEN 2026-05-16)
- **Attack**: 2026-05-16 10:02 wrapper rewrite set `--approval-mode plan` default. Plan-mode silently blocks gemini's shell/Read tools — when prompt asks gemini to `cat docs/checkpoints/paper_drafts/*.md`, tool calls return `Unauthorized` and gemini hallucinates "I read X and produced N findings saved to Y" meta-summary into `.response`. Wrapper extracts the meta-summary into the audit output file (1488 B vs 6699 B on the same prompt with `--yolo`).
- **Trial empirical**: A (plan): 326ms / 0 tokens (cache hit ⚠️) / 2343B; B (yolo direct): 2207B; C (yolo wrapper): 1523ms / 4940 tokens real / 2266B. Winner = Trial C — same wrapper interface + real model call + clean format.
- **Fix**: Ship Trial C — `--yolo` default + `GEMINI_APPROVAL_MODE=plan` env opt-in for inline-context-only prompts + comment line 73-84 rewritten to reflect memory 2026-05-16 retract directive (`gemini --yolo -p ≡ codex --sandbox danger-full-access`).
- **Diagnostic lesson**: `Unauthorized/Blocked/Tool not found` keywords → debug PERMISSION before CAPABILITY (per memory `feedback_cross_ai_audit.md`).

### B-206. Carbon emission provenance missing in paper §3/§8 🛠️ FIXED (prose disclosure)
- **Source**: A1.4c Mode A Finding 4 + Mode C F2 (convergent)
- **Code**: `docs/checkpoints/paper_drafts/section8_limitations.md` §8.7 (EDITED 2026-05-16)
- **Attack**: §8.7 mentioned "kg-CO2 estimates" but did not disclose carbon intensity value, formula, region, or PUE. Reviewer attack: "greenwashing" — non-reproducible CO2 numbers. Double-source: `REGION_INTENSITY_G_PER_KWH["uk"] = 257` in `energy_tracker.py:49` vs explicit `carbon_intensity_g_per_kwh: 220` in `configs/exp_v2_base.yaml:81` (the 220 wins via priority chain at `energy_tracker.py:238-243`).
- **Fix**: §8.7 added: formula `co2e_kg = total_energy_kwh × 0.220`, region `UK national grid 2024 average`, decorative-vs-active explanation of 257-vs-220, PUE=1.0 (dock-power only) caveat with `strubell2019energy` cite.

### B-207. Paper §4 "Adjusted SR" semantic collision with §139.8 FP-retire 🛠️ FIXED (paper-wide rename)
- **Source**: A1.4c Mode A Finding 7 + Mode C F3 (convergent, gemini upgraded to P0)
- **Code**: `docs/checkpoints/paper_drafts/section{1, 4}_*.md` (EDITED 2026-05-16)
- **Attack**: §4 line 5 defined 3-tier "Raw SR / Adjusted SR / Same-task adjusted SR", but §3.5 / §8.2 retired post-hoc `compute_adjusted_success` (per §139.8 lab notes). Reviewer attack: term carries "adjustment" connotation suggesting data manipulation; semantic double-standard between §4 and §8.
- **Fix**: §4 line 5 collapse 3-tier definition → single canonical `VWA-Success (N/A excluded)`; 6 inline rename occurrences in §4 + 2 in §1; §4 line 48 + line 75 inline-flag the FP-decomposition paragraphs as pre-§139.8 archive analysis retained for Appendix D.

### B-208. M3 / baseline retry stale action attribution 📋 DISCLOSED (prose-only, paper-1 latent)
- **Source**: A1.4c Mode B OOB-2 (codex)
- **Code**: `p79/experiment/runner/main.py:1357-1407` + `step_record.action` field semantics
- **Attack**: When M3 fallback retry (`module_flags.m3_failure_trigger_retry`) or `runtime.baseline_retry_on_no_progress` triggers, runner adopts retry's `next_obs/reward/state_after` and sets `action_success = retry_success`, but `step_record.action` keeps the original (failed) action. Any aggregator computing per-action-type success rate without filtering on `retry_action_applied=True` attributes retry-scroll success to the original click.
- **Scope re-evaluation (User Q&A defuse)**: Phase 1a runs neither flag enabled by default (`modules.py:54` M3 flag default False + `config.py:213` baseline_retry default False) → retry never fires in paper-1 → step_record.action_success bug is structurally latent on Phase 1a data. Active only in paper-2 Phase 3 M3 ablation studies.
- **Fix**: Paper §3.5.1 prose disclosure added (defines `action_success` semantics under optional retry + filtering rule); schema fix `executed_action_chain` field queued as paper-2 prerequisite, not paper-1 remediation.

### B-209. SoM degradation 2 paths conflated under single `degraded_som=True` bool 📋 DEFERRED
- **Source**: A1.4c Mode A Finding 5 + Mode C F5 (convergent)
- **Code**: `p79/experiment/som.py:322-333` (zero-marks → raw image fallback) vs `som.py:390-397` (image render fail → no image)
- **Attack**: Both code paths set `degraded_som=True` but give qualitatively different signals to the agent (raw image still consumed in zero-marks path; no image at all in render-fail path). Paper §3 does not distinguish; degradation rate per path not disclosed. Reviewer attack: SoM SR table could be contaminated by vision-fallback in zero-marks path.
- **Status**: DEFERRED to schema fix (add `degraded_som_zero_marks_count` + `degraded_som_render_fail_count` separate fields + paper §3 Table column "% Degraded Steps"). Scope too large for the 0.5h A1.4c remediation budget; queue for Phase 1a data review.

### B-210. ChecklistManagerLite `update_after_action` is step-success counter, not semantic decomposition tracker ✅ DEFUSED by paper-scope
- **Source**: A1.4c Mode A OOB-A + Mode C F1 (convergent construct-validity attack)
- **Code**: `p79/experiment/checklist_module.py:139-159`
- **Attack**: `update_after_action(action_success=bool)` advances first pending → in_progress → completed regardless of whether the action semantically achieved any checklist item. Empirical reproducer: 5 sequential `action_success=True` calls promote 2 checklist items to completed (counter pattern: every 2 successes complete 1 item, in **reverse** order due to a `target_idx` overwrite bug at line 141-146). Gemini construct-validity attack: paper claims "task understanding" but metric is "step-success-counter / item-count".
- **User Q&A defuse**: User confirmed `paper §3/§4 figures + tables` do not actually reference `checklist_completion_rate`; metric exists in code (`analysis.py:357-420`) and `condition_summary_v2.json` schema but not in any published claim. Therefore reviewer attack lacks bite — there's nothing to attack.
- **Status**: DEFUSED. Checklist code retained as latent infrastructure for paper-2 Phase 3 M4 ablation; no paper-1 remediation needed.


---

## A1.5 /stress audit (2026-05-16) — B-211 to B-229 (19 entries; 9 fixed, 5 deferred, 5 disclosed-only)

> Cross-AI: Mode A Claude 9 findings / 3 OOB + Mode B codex 7 findings / 5 OOB + Mode C gemini 5 findings / 2 OOB = 21 unique attack vectors (B-211~B-229 here; 2 already-tracked per A1.4c not re-counted). Cumulative session work touched 8 files: `p79/utils/auth_refresh.py` (rewrite) / `p79/utils/log_cleanup.py` (dry_run default) / `p79/experiment/runner/main.py` (auth gate + in-progress marker) / `scripts/maintenance/experiment_watchdog.py` (delete ImportError fallback + atomic state + PID guard) / `scripts/maintenance/cleanup_logs.py` (--confirm flag) / `scripts/maintenance/clear_tasks.py` (atomic digest) / `scripts/queues/queue_baseline.sh` (auth gate).

### B-211. Plaintext VWA credentials in tracked code 双点 leak 🛠️ FIXED
- **Source**: Claude Mode A Finding 1 (OOB-A) + codex CV-1 strengthens
- **Code**: `p79/utils/auth_refresh.py:16-21` + `scripts/maintenance/experiment_watchdog.py:132-136`
- **Attack**: OWASP "hardcoded credentials" — `_ACCOUNTS` plaintext in tracked files (Phase 1a credentials = `Password.123` / `test1234` / `admin1234`). Subprocess `python -c "<script>"` 把 password 嵌入 cmdline → `ps -ef` / `/proc/<pid>/cmdline` 可见. Even though VWA reference test accounts (no real prod creds), reviewer optics is OWASP red flag.
- **Fix**: `_ACCOUNT_ENV_KEYS` dict mapping site → (USER, PASS) env var names; `_load_account()` raises `AuthRefreshConfigError` on missing env. Watchdog ImportError fallback ENTIRELY DELETED (Item 8 default: silent fallback dangerous). Required env vars: `VWA_<SITE>_USER` + `VWA_<SITE>_PASS` for each of {classifieds, reddit, shopping, shopping_admin}, set via gitignored `scripts/vwa_env_remote.sh`.

### B-212. LOGIN_FAILED URL substring false-positive 🛠️ FIXED
- **Source**: Claude Mode A Finding 2 (OOB-B) + codex CV-2 weak-defuse (pre-fix archive evidence)
- **Code**: `p79/utils/auth_refresh.py:130-141` (pre-fix substring `login_marker in final_url.lower()`)
- **Attack**: substring match systematically false-positives where marker is a path prefix of post-login URL. shopping_admin marker `/admin` matches `/admin/dashboard/index/*` → 永远误判 LOGIN_FAILED. reddit `/login` matches `/login_success?next=...`. classifieds pre-2026-04-26 fix was `/index.php` after `.split('?')[0]` → matched every post-login user page.
- **Fix**: structured `urlparse` + path-equal AND login_qs-subset check. Examples: cls (`path=/index.php`, query `page=login`) — post-login user page has same path but different `page` value → NOT still on login. shopping_admin (`path=/admin`) — dashboard has `path=/admin/dashboard/index` after rstrip → path differs → NOT still on login. Implementation inlined into subprocess script literal at `auth_refresh.py:155-200`.

### B-213. `cleanup_all` destructive `dry_run=False` default 🛠️ FIXED
- **Source**: Claude Mode A Finding 3 + codex CV-3 disagree-defuse (`cleanup_all` IS reachable via CLI)
- **Code**: `p79/utils/log_cleanup.py:200-215` (default `dry_run=False`) + `scripts/maintenance/cleanup_logs.py:115` (passes config without confirmation)
- **Attack**: `python scripts/maintenance/cleanup_logs.py` (default `--dir all`) runs `cleanup_all → cleanup_results(max_run_age=90) → shutil.rmtree(run_dir)` 不可逆. Paper-1 archive baseline runs > 90d 全删 → paper §4 figure source 灭失. Trigger = manual CLI invocation (no cron / no Makefile auto-trigger), but documented in README so误调路径 active.
- **Fix**: `cleanup_all` 默认 `dry_run=True` + 新加 `confirmed: bool = False` kwarg; even if caller sets `config.dry_run=False`,需 `confirmed=True` 才真删 (B-213 safety gate). CLI 加 `--confirm` flag,只有 `--confirm and not --dry-run` 才传 `confirmed=True`. `--dry-run` retained for backward-compat.

### B-214. env propagation leaks parent env into login subprocess 📋 DEFERRED
- **Source**: Claude Mode A Finding 4 (OOB-C)
- **Code**: `p79/utils/auth_refresh.py:153` `env = {**os.environ, "DATASET": dataset}`
- **Attack**: subprocess inherits `OPENAI_API_KEY` / `PROXY_API_KEY` / `AWS_*` / `GITHUB_TOKEN`. crash → stack trace 或 `/proc/<pid>/environ` 可读.
- **Status**: DEFERRED per user Q9. theoretical leak,无 incident 实证. comment added at code site documenting future tightening to minimal env.

### B-215. asyncio locale-fragile English pattern match 📋 DEFERRED
- **Source**: Claude Mode A Finding 5 (OOB-D)
- **Code**: `p79/utils/asyncio_workarounds.py:32` literal `"future exception was never retrieved"`
- **Attack**: Python upstream message change → filter 静默失败 → logs flooded with TargetClosedError.
- **Status**: DEFERRED per default (Q13). upstream message 稳定多年.

### B-216. `episode_*.jsonl` cleanup pattern naming-collision 📋 DEFERRED
- **Source**: Claude Mode A Finding 6
- **Code**: `p79/utils/log_cleanup.py:237-239` `temp_patterns = ["episode_*.jsonl", ...]`
- **Status**: DEFERRED per default (Q14). 0-impact currently — production JSONL is `<site>_task_<n>_steps_v2.jsonl`, no `episode_*.jsonl` producer in tree.

### B-217. `auth_refresh.py` subprocess timeout=30s marginal 📋 DEFERRED
- **Source**: Claude Mode A Finding 7
- **Code**: `auth_refresh.py:155-159`
- **Status**: DEFERRED per default (Q15). `AUTH_REFRESH_TIMEOUT` env override added preemptively as documented field (default 30s, opt-in 60s+ for slow networks).

### B-218. torch_cuda_workarounds patch guard lost on `importlib.reload(torch)` 📋 DEFERRED
- **Source**: Claude Mode A Finding 8
- **Code**: `p79/utils/torch_cuda_workarounds.py:60`
- **Status**: DEFERRED per default (Q16). paper-1 workflow 不用 jupyter reload.

### B-219. env_snapshot.json overwrite race with same-run_id resume 📋 DEFERRED
- **Source**: Claude Mode A Finding 9
- **Code**: `p79/cli/run_experiment.py:50-57`
- **Status**: DEFERRED per default (Q17). resume case 罕见.

### B-220. Runner auth-refresh failure non-blocking 🛠️ FIXED
- **Source**: codex Mode B OOB-01
- **Code**: `p79/experiment/runner/main.py:953-969` (pre-fix `if ok: ... else: logger.warning("...continuing with stale session")`)
- **Attack**: refresh failure 仅 log warning + 继续 `environment.reset()` + episode → NOT-LOGGED-IN session 进 step_record + condition_summary_v2.json. SR 在未登录 session 上算 → cross-mode 差异不可解释.
- **Fix**: replaced soft `refresh_site_auth(...) -> ok/warn` with `auth_required_gate(...)` (new helper in `auth_refresh.py`). Raises `AuthRefreshFailure` after retry-budget exhausted → episode aborts → outer episode-safe wrapper records as auth-aborted → watchdog picks up via state + retries condition.

### B-221. Watchdog contamination cleanup conditional on later positive login 📋 DOCUMENTED (architectural, not code-fix scope)
- **Source**: codex Mode B OOB-02
- **Code**: `scripts/maintenance/experiment_watchdog.py:1434-1510`
- **Attack**: `if session_ok is True: contaminated = session_contaminated.pop(site, [])` — vision/SoM modes 难触发 session_ok=True. condition 完成前无 positive signal → contaminated list 永驻不清.
- **Status**: partial mitigation via B-220 (runner-side gate prevents new contaminated episodes from accumulating in first place). True "condition finalization gate" (failing condition with unresolved contamination) is queued for paper-2 prep — current B-220 + B-224 closure substantially reduces contamination rate.

### B-222. Watchdog orphan-cleanup 10min mtime 无 runner-liveness check 🛠️ FIXED
- **Source**: codex Mode B OOB-03
- **Code**: `scripts/maintenance/experiment_watchdog.py:1154-1187` + `p79/experiment/runner/main.py:971-996`
- **Attack**: 长 episode (image render hang / browser stuck) 或挂住的 episode 容易 > 10min mtime,watchdog restart 把 runner 正在写的 artifacts 当 orphan 删 → episode 中段 state inconsistent + summary 不一致.
- **Fix**: (a) primary guard — watchdog `pgrep -fa "run_experiment.*${run_dir.name}"` probe; live runner detected → SKIP all orphan cleanup this cycle; (b) secondary guard — runner writes `.in_progress` marker at `episode_dir.mkdir` time (after auth gate passes) + removes at successful `episode_summary` return; watchdog orphan-pruning skips dirs/files where corresponding `.in_progress` marker exists.

### B-223. Watchdog state non-atomic write + silent reset on corrupt JSON 🛠️ FIXED
- **Source**: codex Mode B OOB-04
- **Code**: `scripts/maintenance/experiment_watchdog.py:927-971`
- **Attack**: crash mid `path.write_text(json.dumps(...))` → truncated file → next `_load_state` `except Exception: return {}` → retry budget + contamination memory + seen_keys 全丢.
- **Fix**: atomic write pattern (tmp + fsync + `os.replace` + dir fsync). Same pattern as `LoggerV2._fsync_dir` per B-198.

### B-224. Queue post-reset auth-refresh failure warn-only 🛠️ FIXED
- **Source**: codex Mode B OOB-05
- **Code**: `scripts/queues/queue_baseline.sh:192-206`
- **Attack**: post-reset `refresh_site_auth` 失败仅 `echo "[warn]"` + `setsid nohup ... run_experiment.py` 照样启 → 前 3 task 极可能未登录.
- **Fix**: replaced with `auth_required_gate` call via embedded Python heredoc. Failure → `exit 1` (abort launch). `AUTH_GATE_BYPASS=1` env var available as explicit opt-out (paper-grade dirty, watchdog reactive only).

### B-225. Watchdog inline fallback hardcoded `100.95.81.103` 🛠️ FIXED
- **Source**: codex Mode B OOB-06
- **Code**: `scripts/maintenance/experiment_watchdog.py:138-142, 165` (pre-fix)
- **Attack**: literal IP 跟 quark Tailscale 老 IP 绑定. A100 / 换 Tailscale IP / quark IP 变动时 fallback 跑错域 → 写错的 storage state 进 `.auth/` → 后续 episode 用错 cookies NOT-LOGGED-IN.
- **Fix**: per Item 8 user decision, entire `experiment_watchdog._auto_refresh_auth` ImportError fallback block (~92 lines) DELETED. ImportError now → fatal print + return False (loud fail). B-211 + B-225 closed simultaneously via single deletion (双点 cleanup).

### B-226. `clear_tasks.py` digest non-atomic rewrite after delete 🛠️ FIXED
- **Source**: codex Mode B OOB-07
- **Code**: `scripts/maintenance/clear_tasks.py:243`
- **Attack**: `f.unlink()` + `shutil.rmtree(d)` 先删 episode files,再 `jsonl_file.write_text(...)` 覆盖 digest. crash 中 → digest 半写或残留旧记录 + episode 已 gone → 后续 gallery/analysis 不一致.
- **Fix**: atomic temp-write + fsync + `os.replace` + dir fsync (same pattern as B-198 / B-223).

### B-227. Evaluator integrity via `p79-patches` proprietary fork 📋 DEFERRED (companion bug paper)
- **Source**: gemini Mode C OOB-F
- **Code**: paper §3.5 line 113 + VWA submodule branch `p79-patches` commit `f0c835b`
- **Attack**: paper claims VWA SR but actual evaluator is forked + patched (empty-prediction → 0.0 guard). Reviewer "evaluator-hacking" attack: even with legitimate motivation (fixes GPT-4o-mini empty-pred误判), paper-grade評測 must publish patch scope + SBOM + 原 VWA 对照 sample.
- **Status**: per user Item 1 decision — DEFERRED to **independent bug研究 paper** (workshop-targeted, agisdk-style cross-benchmark bug aggregation per `next_steps.md §11`). Companion paper hosts complete Appendix-style disclosure (patch list + diff + before-after sample retest + cross-link to B-91 / B-01/02/33 / B-90 / B-209 fixes). Main paper §3.5 prose already references `f0c835b` commit + branch name as in-paper reproducibility pointer; full SBOM is companion-paper substrate.

### B-228. PUE=1.0 carbon greenwashing attack 📋 DEFERRED (70% pre-defused)
- **Source**: gemini Mode C #4
- **Code**: paper §8.7 (per B-206 commit `a2962fd`)
- **Attack**: PUE=1.0 "physically impossible". Real university HPC PUE 1.3-1.5.
- **Status**: DEFERRED per user Q11. B-206 prose already discloses "PUE=1.0 (dock-power only); typical 1.3-1.5" as lower-bound caveat. Gemini missed the framing. 30% gap = paper §1/§4 not明示 "carbon estimate is lower bound, multiply by 1.3-1.5 PUE for facility-effective" but defer accepted.

### B-229. Parse error taxonomy needs source decomposition 📋 DEFERRED (paper-grade analysis, post-Phase-1a)
- **Source**: gemini Mode C #5 + user Item 12 refinement
- **Code**: `p79/backends/action_utils.py::validate_action_detailed` + `step_record.parse_failure_reason` (single-axis classification)
- **Attack**: current `parse_failure_reason` doesn't separate (M) model output error / (P) B0 proxy API corruption / (S) scaffold parsing error / (R) scaffold-recoverable rejection. Reviewer integrity-report meta-attack: "你 corrupt-line 标准是什么? 跨 mode 不均匀 = cherry-picking".
- **Status**: DEFERRED to dedicated archive analysis task. Plan: extend `validate_action_detailed` with `model_output_class` + `scaffold_recoverable` fields; add `scripts/analysis/parse_error_taxonomy.py` cross-archive distribution analysis by (backend, mode, site). Output lives in companion bug paper (B-227 family). Effort: 2-3h script + 0.5h schema fields.

**B-numbers consumed**: B-211 through B-229 (19 contiguous, no collisions). Cumulative session work (A1.4a+A1.4b-i+A1.4b-ii+A1.4c+A1.5) = B-140 through B-229 = 90 unique entries (some defused / deferred / disclosed; the rest fixed in code).

**Next available B-number**: B-237+ (after A1.13 batch below).


---

## A1.13 + A1.14 /stress audit (2026-05-16) — B-230 to B-236 (7 entries; 7 fixed, 0 deferred, 0 disclosed-only)

> Cross-AI: Mode A Claude 8 findings / 5 OOB + Mode B codex retry 8 findings / 0 OOB labels (conservative) + Mode C gemini 6 findings / 3 OOB = 22 attack vectors collapsed to 10 unique (5 Q-options) → 7 fixed this batch. Scope = 7 queue/orchestrator/preflight scripts (`queue_baseline.sh` + `queue_chain.sh` + `queue_phantom_{som,text,prompt}.sh` + `queue_phase1_paper_grade.sh` + `preflight_v2.sh`). Q4 user decision: lib extraction over inline copy → `scripts/queues/_lib_paper_grade_gates.sh` (174 LOC) centralizes 5 helpers; 4 queue scripts shrink -274 LOC (-26%). Sibling-propagation defect class definitively closed.

### B-230. Auth gate hard-fail not propagated to phantom queue scripts 🛠️ FIXED (sibling propagation)
- **Source**: Claude Mode A Finding A1 + codex C-1 + gemini G-1 (3-AI overlap, P0)
- **Code**: `scripts/queues/queue_phantom_som.sh:155-166` + `queue_phantom_text.sh:194-205` + `queue_phantom_prompt.sh:151-162` — pre-fix soft `refresh_site_auth` + `[warn] post-reset auth refresh failed; watchdog will retry reactively after streak=3`
- **Attack**: B-224 fix (`auth_required_gate` hard-fail) only landed in `queue_baseline.sh`; 18 phantom conditions (50% of Phase 1a 36-cond matrix = P-text + P-prompt + P-SoM × 6 cells) shipped with the pre-fix soft-warn → first 1-3 phantom tasks ran NOT-LOGGED-IN post-reset before watchdog reactive cleanup (streak=3 ≈ 3-5 min). Paper §1 phantom-vs-baseline SR delta partially confounded by auth state, not by observation mode alone — disentangle 不可能。
- **Fix**: `reset_and_auth_gate()` helper centralized in `scripts/queues/_lib_paper_grade_gates.sh:114-160`. All 4 queue scripts source the lib + call this single function in the RESET_BEFORE branch. Q4 A decision (lib over inline copy) prevents future sibling drift on subsequent gate additions.
- **Files**: `scripts/queues/_lib_paper_grade_gates.sh` (new) + `queue_baseline.sh` (refactored) + `queue_phantom_som.sh` + `queue_phantom_text.sh` + `queue_phantom_prompt.sh`. 0 leftover `refresh_site_auth` references in phantom scripts post-fix (verified `grep -c refresh_site_auth` returns 0/0/0).

### B-231. BUG-2 A100 URL-locality preflight not propagated to phantom queue scripts 🛠️ FIXED (sibling propagation)
- **Source**: Claude Mode A Finding A1 + codex C-2 + gemini G-1 (3-AI overlap, P0)
- **Code**: `scripts/queues/queue_baseline.sh:114-122` had `BUG-2 preflight: assert all site URLs are local on A100` block (codex stress v6 C2 fix); `queue_phantom_{som,text,prompt}.sh` had no equivalent.
- **Attack**: Empirical `grep -c "BUG-2 preflight" scripts/queues/queue_phantom_*.sh` returned 0/0/0. 18 phantom conditions on A100 with stale prod URLs in env (e.g. user switched sessions DGX → A100 without updating `vwa_env_remote.sh`) would silently hit quark prod docker via Tailscale → 50% silent deployment substitution. Memory `project_paper_grade_target_host` (2026-05-14 standing decision) defines paper-grade target = A100 self-hosted docker exclusively.
- **Fix**: `assert_a100_url_locality()` helper in `_lib_paper_grade_gates.sh:38-50`. Same `*condense*` hostname / `/home/ubuntu/workspace/p79` dir detection; on A100 hosts, refuses launch if any of `CLASSIFIEDS/REDDIT/SHOPPING/WIKIPEDIA` non-local. Called from all 4 queue scripts post-`init_paper_grade_env`. Verification: 4-script `grep -c assert_a100_url_locality` returns 1/1/1/1.

### B-232. queue_chain.sh C3 sentinel file-existence-only (no content validity) 🛠️ FIXED
- **Source**: Claude Mode A Finding A2 + codex C-3 + gemini G-3 (3-AI overlap, P0 OOB)
- **Code**: `scripts/queues/queue_chain.sh:178-182` (pre-fix `if [[ -f "${cand}" ]]; then summary_found="${cand}"; break; fi`)
- **Attack**: File-presence alone insufficient. Three concrete failure modes: (a) FORCE_NEW=1 same-second collision (pre-B-234 1-second timestamp precision) → second launcher sees first's partial JSON → sentinel pass → chain advance with empty cell. (b) Mid-write SIGKILL (OOM / sysadmin reboot) → JSON truncated → `read_jsonl_dedup` silently skips bad rows → aggregator reads 0 tasks → paper §1 numbers based on partial data. (c) Stale prior-run dir reused at same `run_id/cond_id` path.
- **Fix**: `queue_chain.sh:178-204` now validates (a) `[[ -s file ]]` non-empty, (b) `json.load(open(cand))` parses, (c) `condition_id` field matches expected `${cond_id}`, (d) `total_tasks` (with fallback to `num_tasks` / `scored_task_count`) is `int > 0`. Validation failure → log + abort chain + ntfy "queue_chain ABORT: sentinel validation failed" (distinguishable from prior "no summary" message). Smoke: in-line Python heredoc runs on every cell completion before advancing.

### B-233. Watchdog liveness invariant not enforced in queue_chain wait loop 🛠️ FIXED
- **Source**: Claude Mode A Finding A3 + codex C-4 + gemini G-6 (3-AI overlap, P0)
- **Code**: `scripts/queues/queue_chain.sh:67-80` (pre-fix `wait_for_runner_done` watched runner PID only)
- **Attack**: `queue_baseline.sh:270-278` codex stress v6 C5 declares "watchdog FATAL for paper-grade launch" — but invariant enforced ONLY at launch time. Mid-run watchdog death (typical roots: ntfy curl SIGPIPE, OOM, glm config bug, NPE on bad state JSON) silent: runner continues, chain wait loop continues polling runner PID, C3 sentinel passes when runner finishes, cell marked done. But watchdog-dead window (potentially 24-48h on weekend chain runs) loses reactive auth refresh + idle alerts + auto-clean → mid-run auth drift → NOT-LOGGED-IN tasks → adjusted_success bias. Multi-day chain failure mode is paper-grade-fatal.
- **Fix**: `queue_chain.sh:wait_for_runner_done` (updated 67-93) adds watchdog liveness check inside polling loop. If watchdog dead while runner alive: log FATAL + `pkill -f run_experiment.py.*${pattern}` (kill runner) + ntfy "queue_chain ABORT (${label}): watchdog died after ${elapsed}s" + `exit 1`. Q2 A decision (abort over restart): paper-grade > compute reclaim. User triggers manual restart after diagnosing watchdog root cause.

### B-234. FORCE_NEW=1 same-second RUN_ID collision risk 🛠️ FIXED
- **Source**: codex C-5 + gemini G-2 (2-AI overlap; Claude A missed). P1 OOB.
- **Code**: pre-fix 5 queue scripts used `TS_FULL="$(date +%Y%m%d_%H%M%S)"` (1-second precision)
- **Attack**: master orchestrator `queue_phase1_paper_grade.sh` fires cls + red chains via `nohup ... &` near-simultaneously. Manual retry / re-fire in tight loop also susceptible. Two launchers in same second → identical RUN_ID → identical `run_dir` / runner log / watchdog state JSON → two runners attached to same docker user account → session race + log overwrite.
- **Fix**: `mint_run_id()` helper in `_lib_paper_grade_gates.sh:74-105`. Format: `${CFG_NAME}_YYYYMMDD_HHMMSS_PIDxxxx_Rxxxxx` (PID = `$$`, R = `$RANDOM` 0-32767). Same-second 3-call empirical smoke test (2026-05-16 15:49:02 PID 3598120) produced 3 distinct RUN_IDs `R20797 / R17648 / R25754` — collision probability ~1/32767² ≈ 10⁻⁹ even within same second + same PID.

### B-235. Preflight skips VWA submodule branch / commit / B-91 patch verification 🛠️ FIXED
- **Source**: Claude Mode A Finding A6 + codex C-6 + gemini G-4 (3-AI overlap, P1 OOB)
- **Code**: pre-fix `scripts/preflight_v2.sh:343-376` `check_vwa_evaluator_import` only `from evaluation_harness import evaluator_router`; Makefile `verify-version-locks` target had SHA pin but queue path didn't enforce it.
- **Attack**: user `git submodule update --remote` accidentally switches to `main` branch → loses B-91 fix (LLM judge `pred=""` guard) → N/A tasks revert to old judge → visual_fp ~2-3pp regression. Preflight passes (import works on any branch). OSF reproducibility audit: reviewer `git submodule init` defaults to `main` → reproduction fails silently.
- **Fix**: `preflight_v2.sh:343-381` new `check_vwa_submodule_lock()` function. Verifies (a) submodule dir is git repo, (b) branch = `p79-patches`, (c) SHA = `f0c835b35191e2ff8d46993d9279674a0956ef14`, (d) `grep -cP '^\s*if not pred or not pred\.strip\(\):' evaluation_harness/helper_functions.py` = 2 (the B-91 guard form, both LLM-judge functions in `helper_functions.py:589 + :634`). Wired into `main()` between `check_torch_cuda` and `check_vwa_evaluator_import`. Verified smoke: current state (branch `p79-patches`, SHA matches, guard count = 2) passes.

### B-236. queue_baseline.sh QUARK_TZ export vestigial dead code 🛠️ REMOVED
- **Source**: Claude Mode A Finding A4 (Claude-unique OOB; codex C-2 / gemini G-1 raised propagation gap but missed correctness gap)
- **Code**: `scripts/queues/queue_baseline.sh:104-108` (pre-fix `export QUARK_TZ="${QUARK_TZ:-Europe/London}"` with comment "BUG-6 fix, 3-AI agree 2026-05-16. Postmill timestamps render in container TZ → reddit task must_include break across midnight boundary")
- **Attack**: Client-side `export QUARK_TZ` does NOT influence docker container TZ (container TZ controlled by `docker-compose TZ:` flag or `/etc/timezone` in image). Empirical: `grep -r "QUARK_TZ" p79/ scripts/` returns single-line hit only (the export itself; no runner consumer). Variable name "QUARK_TZ" also outdated: paper-grade fires on A100 self-hosted docker (memory `project_paper_grade_target_host`, standing decision 2026-05-14) — no quark in launch path. Two failure modes: (a) cargo-cult fix attribution: paper §139 BUG-6 entry mistakenly marked fixed; (b) misleading future audits — next reviewer asks "why is this here, who reads QUARK_TZ?" and gets no answer.
- **Fix**: 5 lines deleted (`queue_baseline.sh:104-108` QUARK_TZ block), replaced with 8-line `BUG-6 NOTE` audit-trail comment explaining pre/post-2026-05-14 era + A100 host UTC reality + residual cross-midnight relative-timestamp drift bounded to ~5/210 reddit tasks (disclose-only in paper §限制, not code-fixable here). Memory file `project_paper_grade_target_host.md` documents the broader vestigial-marker class for future audits.

**B-numbers consumed**: B-230 through B-236 (7 contiguous, no collisions). Cumulative session work (A1.4a+A1.4b-i+A1.4b-ii+A1.4c+A1.5+A1.13) = B-140 through B-236 = 97 unique entries.

**Next available B-number**: B-237+.

## §158 /stress A1.6 `p79/experiment/analysis.py` — FP architecture hard-delete sweep (2026-05-16)

Audit: Mode A (Claude) F1-F9 + Mode B (codex reproducibility-auditor) F10-F17 + Mode C (gemini broad-reviewer) B-118~B-124 = 17 findings. User overrule "selective-retain-for-output-schema-stability" policy (2026-05-14) → hard-delete all retired-layer remnants. `docs/analysis/` archived to `docs/archive/analysis_pre_2026-05-15/`. Chronicle: 实验笔记 §158.

### B-237. `aggregate_sr_fp_per_mode.py` emits dual `n_raw_success` + `n_adjusted_success` (恒等 post-§139.8) 🛠️ FIXED
- **Source**: Mode A F5 + Mode B F10 (collector advertise gap) — 2-AI overlap
- **Code**: `scripts/analysis/aggregate_sr_fp_per_mode.py:78-90, 117` emit `n_raw_success = n_success` + `n_adjusted_success = n_success` + `raw_sr_pct` + `adjusted_sr_pct` 双列;markdown table L117 `{raw_sr_pct} | {adjusted_sr_pct}` 渲染恒等双列
- **Attack**: Reviewer 看到 sr_fp_per_mode.json 两 SR field 恒等会疑似 pipeline bug 或冗余记账。Paper §3.5 disclosure "post-hoc retired" 与 schema 两 SR field 矛盾
- **Fix**: 整 file rewrite → 单列 `n_success` + `sr_pct` + 新增 `expected_n` / `complete` / `completeness_ratio` (Q9 partial-cell handling piggy)。Output path `sr_fp_per_mode.json` → `sr_per_mode.json`;`schema_version: v2-2026-05-16-fp-retire`

### B-238. `analyze_cross_representation._mark_false_positives` thin alias-setter (0-emit cargo cult) 🛠️ FIXED
- **Source**: Mode A F6 + Mode B F14 — 2-AI overlap
- **Code**: `scripts/analysis/analyze_cross_representation.py:370-391, 546-549, 1338, 1575-1576, 1731-1736` `_mark_false_positives` 自 docstring 标 "thin alias-setter for output-schema stability";emit `*_na_fp = False` / `*_eval_fp = False` / `na_fp_count = 0` / `eval_fp_count = 0`;a2/a3/a5/a6/r2/r3 都有 `has_adj` mirror branches
- **Attack**: Output schema 13 处恒 0/False emit + 整 `_success_adj` 路径 mirror raw 路径,readers grep 找 FP detection 看到一个 "always returns 0" 的函数 → cargo-cult perception
- **Fix**: `_mark_false_positives` 函数整删 + `_compute_set_metrics` / `_compute_exclusive_sets` / `_build_oracle_rows` `success_suffix` 参数整删 + a2/a3/a5/a6/r2/r3 `has_adj` mirror branches 全清 + `write_summary` `*_adjusted` keys + caller site `_mark_false_positives` 调用整删

### B-239. `collect_analysis_summary.py` 顶层 collector 读已 retired keys → JSON null/missing 😵 🛠️ FIXED
- **Source**: Mode B F10 (Claude+gemini 漏) — codex-unique OOB
- **Code**: `scripts/analysis/collect_analysis_summary.py:14-16, 91-97, 125-142, 201, 295` reads `adjusted_success_rates` / `na_fp_count` / `eval_fp_count` / `na_reference_tasks.csv` / `adjusted_success`(L295 字面 sum)。Source `analysis.py:1412-1422` (post-§139.8) 不再写这些 key → collector 输出全是 null
- **Attack**: Reviewer 拿 paper-grade consolidated JSON 看到 null 字段无法分辨 "已 retire" vs "pipeline 失败" → credibility 损失
- **Fix**: Hard-drop 4 retired field reads + `raw_success_rate` → `success_rate` + `intent_feature_sr` 内 `adjusted_success` → `success` + `exclusive_sets_adjusted` → `exclusive_sets`

### B-240. `analyze_confidence_calibration.py` in-memory alias 漂移 🛠️ FIXED
- **Source**: Mode A F5 (Claude-unique)
- **Code**: `scripts/analysis/analyze_confidence_calibration.py:2228-2238` set `ep_df["raw_success"] = ep_df["success"]` + `ep_df["adjusted_success"] = ep_df["success"]` + `ep_df["fp_reason"] = ""` + emit `label_mode` / `n_adjusted` / `n_success_raw` / `n_success_adjusted` JSON fields;`--no-adjust` CLI flag (documented no-op)
- **Attack**: Downstream consumer 看 `n_success_raw` / `n_success_adjusted` 两 field 恒等 = 疑似 bug 或冗余。`--no-adjust` flag 出现在 `--help` 表暗示 "可选 adjust",实际 retire 后无意义
- **Fix**: 3 alias columns 整删 + 4 JSON output keys 整删 → 单 `n_success`;`--no-adjust` argparse 行整删

### B-241. `analyze_reason_diagnostics.py` episode_reason_rows.csv 仍 emit `adjusted_success` / `fp_reason` 列 🛠️ FIXED
- **Source**: Mode A F1 + downstream
- **Code**: `scripts/analysis/analyze_reason_diagnostics.py:2003-2005, 2045-2046, 2159-2162` set `adjusted_success = success` + `fp_reason = ""` + `adjusted_reason_bucket = reason_bucket` + 写入 episode_reason_rows.csv;L2293-2335 `condition_overview.csv` emit `adjusted_success_count` / `adjusted_success_rate` / `fp_count`(后者读 `fp_reason` 恒空 → 恒 0);L1488-1525 `state_change_by_outcome.csv` group by `adjusted_success`;L1612-1755 `intent_feature_sr` / `task_type_mode_sr` / `temporal_sr` 计算 `r.get("adjusted_success")` 恒等 raw
- **Attack**: CSV 输出 schema 含恒等 / 恒空 alias 列让 reviewer 误读为 "fp_count=0 是 paper-grade 结论",实际是 silent retire emission
- **Fix**: 3 alias variable + 4 CSV column emit (`adjusted_success` / `fp_reason` / `adjusted_reason_bucket` / `adjusted_success_count` / `adjusted_success_rate` / `fp_count`) + state_change_by_outcome `adjusted_success` → `success` + intent/task/temporal SR 读 `success`;`temporal_sr.csv` `adjusted_sr_pct` → `sr_pct`

### B-242. `layered_status.py` mode_stats 双 emit `raw_successes` / `adjusted_successes` + "0b FP rate" markdown section 🛠️ FIXED
- **Source**: Mode A F5 + Mode B F14
- **Code**: `scripts/analysis/layered_status.py:181-209` mode_stats emit `raw_successes`/`adjusted_successes`/`raw_sr`/`adjusted_sr`/`fp_rate` keys (5 dual);L300-328 markdown write "0b FP rate (raw success - adjusted success)" section emit 恒 0% rate
- **Attack**: layered_evidence_status.md "0b" section paper §3 layered-evidence 渲染恒 0 → reviewer "FP rate 全 0,这数据可信吗?"
- **Fix**: mode_stats 单 emit `n_success` + `sr`;markdown "0b FP rate" 整 section 删除 + 注释 trace

### B-243. `figures/fig0a_sr_per_mode_heatmap.py` annotation prints adjusted SR + raw SR + `fp=` 🛠️ FIXED
- **Source**: Mode B F14
- **Code**: `scripts/analysis/figures/fig0a_sr_per_mode_heatmap.py:79-87` cell annotation `{adj:.1f}%\n({raw:.1f}% raw)\nN={n}, fp={fp}`;title "Adjusted success rate (%)"
- **Attack**: paper §1 hero figure 三元注释训练 reviewer 把 adjusted / raw / fp 当 active quantity,与 paper §3.5 retire 声明矛盾
- **Fix**: annotation 简化为 `{sr:.1f}%\nN={n} ({mark})` 其中 mark = "✓" if complete else "{n}/{expected}";title "Success rate (%)";source path 切到 `sr_per_mode.json` (B-237 联动);colorbar label "SR (%)"

### B-244. `figures/fig0b_fp_rate_per_mode.py` 整 figure 围绕已退役 FP rate (always 0 post-§139.8) 🛠️ DELETED
- **Source**: Mode B F14 + Claude F6
- **Code**: `scripts/analysis/figures/fig0b_fp_rate_per_mode.py` 整 file (docstring + title + footnote 都定义 FP rate = raw - adjusted)
- **Attack**: figure 永远画全 0 条 → reviewer 第一眼问 "为啥所有 mode FP=0?";paper §1/§4 无引用 (`grep -rn "fig0b_fp_rate" docs/checkpoints/paper_drafts/` 空)
- **Fix**: 整 file `git rm`;`Makefile:250` line 删除 + comment "retired §139.8 + /stress A1.6"

### B-245. `generate_gallery.py` FP-badge JS (V-FP/N-FP/E-FP coloring) inert 但 render 🛠️ FIXED
- **Source**: Mode A F6
- **Code**: `scripts/maintenance/generate_gallery.py:131-156, 354-355, 614-617, 904-905, 957-960` FP-badge JS + CSS `.fp-indicator` + episode payload `adjusted_success` / `fp_reason` 字段
- **Attack**: Gallery HTML output 仍尝试 render FP-badge (post-§139.8 inert because `fp_reason` 恒空) → JS dead code + readers 疑惑 fp-indicator 何时显示
- **Fix**: `_load_reason_rows` 内 `adjusted_success` / `fp_reason` 字段读 + 2 处 dict payload 字段 + Home table cell `fpTag` 注释化 + Episode view `if(e.fp_reason)` 块整删 + CSS `.fp-indicator` rule 注释化

### B-246. `analysis.py::_load_na_task_ids` warning text "na_fp detection will be silently disabled" 🛠️ FIXED
- **Source**: Claude F7
- **Code**: `p79/experiment/analysis.py:29-33, 45-49` warning literally references 已 retire "na_fp detection" 功能
- **Attack**: Operator 读 log "na_fp detection disabled" 去找 code 找不到 → 困惑;warning 文案 lifespan 比功能本身还久 = stale-comment 病
- **Fix**: 改 "scored_task_count: N/A config not found ... will fall back to 0" (与 strict mode 协同)

### B-247. `analysis.py` `is_na_reference` 计算块 + `na_reference_tasks.csv` 写出 + `na_reference_task_count` JSON field 🛠️ FIXED
- **Source**: Claude F2
- **Code**: `p79/experiment/analysis.py:1271-1291, 1383-1385, 1401, 582` — `is_na_reference` per-episode flag + CSV emit + JSON summary field + cumulative_success_rate caption "N/A excluded at task-load"
- **Attack**: `exclude_na_tasks=True` default → episodes 不含 N/A → CSV 恒空。Caption 写死 "N/A excluded" 与 CSV 存在矛盾 (一个 surface 暗示 N/A 已排除,另一个 surface 在记录 N/A) → reviewer 比对立刻 catch invariant violation
- **Fix**: is_na_reference 计算块整删 (含 fallback `ep_df["is_na_reference"] = False`) + `na_reference_tasks.csv` emit 删 + `na_reference_task_count` JSON field 删 + dead `noise_dir` mkdir 删 (now empty dir) + caption 简化为 "Success Rate — {cond_id}"

### B-248. `analysis.py` `raw_success` in-memory alias 浪费 + downstream `n_success_raw` 误读 🛠️ FIXED
- **Source**: Claude F5
- **Code**: `p79/experiment/analysis.py:1297-1300` `ep_df["raw_success"] = ep_df["success"].copy()` 保留 alias for backward-compat
- **Attack**: 历史 raw_success = "未 adjusted",§139.8 后 == canonical → `analyze_confidence_calibration.py:2357` emit `n_success_raw` 字段下游 consumer 看到与 `n_success` 恒等会疑似 bug。Alias 保留是 cargo-cult 而非 backward-compat (no current reader expects different semantics)
- **Fix**: 列整删 (sweep readers in B-240)

### B-249. `scored_task_count` silent 0-fallback → 假阳性 "complete" propagation 🛠️ FIXED
- **Source**: Mode A F4 + Mode B F11/F12 — 2-AI overlap (codex 加 specific call-sites)
- **Code**: `p79/experiment/analysis.py:60-77` return 0 on missing config (silent + warning);propagation: `run_registry.py:152` `is_complete = actual_n >= expected_n` (`0 >= 0 = True`);`fig1ab_cascade_diamond.py:146-153` `if n >= min(200, expected): return "complete"` (expected=0 → n=0 也 complete);`active_processes.py:34-40` 同款 fallback (`make active` 显示 0/0)
- **Attack**: Phase 1a launch 后任何 config drift / Docker mount 错 → silent 把 missing cells 标 "complete" → paper-grade promote 通过 empty data。比 fail-loud 危险 = fail-silent + false-positive on completeness
- **Fix**: `scored_task_count(strict=True)` kwarg 加 → raise FileNotFoundError;paper-grade callers (7 个文件 EXPECTED_N module-level) 全切 strict;`run_registry.is_complete` 加 `expected_n > 0` guard;`fig1ab_cascade_diamond.prompt_status` 加 `assert expected > 0`

### B-250. N/A 定义 DRY drift — `_load_na_task_ids` 与 `_is_na_task` 重复 🛠️ FIXED
- **Source**: Claude F3
- **Code**: `p79/experiment/analysis.py:40-43` 与 `p79/experiment/tasks.py:24-25` 双重定义 `eval.reference_answers.fuzzy_match == "N/A"`
- **Attack**: 改 N/A definition (e.g., 加 `exact_match == "N/A"` 覆盖)单改一处 → exclusion-at-load 与 analysis-time 集合 silent 差一。Future-refactor entrance for bug。
- **Fix**: extract `_resolve_site_config` + `_load_site_tasks` helpers;`_load_na_task_ids` 从 `tasks.py` import `_is_na_task` 复用 (local-import 避免 cycle);测试 `test_na_definition_single_sourced` guard 防 future regression

### B-251. Denominator transition gap (3-AI overlap) — docstring / report prose / paper §4 三 surface 仍 pre-exclusion 234/210 🛠️ FIXED
- **Source**: Mode A F8 + Mode B F16 + Mode C B-118 — **3-AI overlap (highest paper-grade confidence)**
- **Code**: `p79/experiment/analysis.py:57-58` docstring quote `EXPECTED_N = {classifieds: 234, ...}` (pre-exclusion);`scripts/analysis/axis1_microbehavior.py:713-718` report prose 仍 hardcode `/210` `/234`;`section4_empirical_findings.md:23` paper §4 hero table N=234/210
- **Attack**: §139.8 sweep 时只清了 12 个执行路径,3 surface 的 narrative footprint 没改 → reviewer 看 §4 hero table 写 "Cls SR = 38/234 = 16.2%",但 prereg + N/A exclusion 说 N=224 → 算 38/224 = 16.96% → 立刻拒。3-AI 互验 = 真 paper-grade P0
- **Fix**: (a) `analysis.py:55-69` `scored_task_count` docstring 加 "Post-exclusion: cls=224, red=205, shop=435" 明确 + 引用 strict mode rationale;(b) `axis1_microbehavior.py:711-723` report prose 改用动态 `EXPECTED_N[site]` 替换 hardcode (这次会 emit 224/205);(c) **Paper §4 hero table** defer 等 Phase 1a clean-run data land 后一次性重算 (numbers will change anyway)

### B-252. `power_analysis.py` K-of-16 body section obsolete prose (header retired vs body 矛盾) 🛠️ FIXED
- **Source**: Mode B F17
- **Code**: `scripts/analysis/power_analysis.py:147-180` body still emit "## Family-wise power (K-of-N rule)" table with K_h1=12/16 / K_h3=11/16 + family-wise power calculations;但 file header L5-11 已声明 "K-of-N rule RETIRED 2026-05-14 as gate per preregistration.md §4 Decision 3A"
- **Attack**: Self-contradicting file (header retired-status statement vs body emit retired prose) → reviewer 看 appendix generator output 看到 "K=12/16" decision rule 复活 → 信度受损
- **Fix**: body L147-170 (K-of-N table + family-wise power calculations + K_h1/K_h3 interpretation) 整段删 + L164-167 reviewer-defensible claim 重写去 K-of-N reference + 注释 trace。**保留** 234/210/466 pre-exclusion design N (prereg-locked,故意 desync 风险 > MDE 位移)

### B-253. test coverage gap for FP architecture invariants 🛠️ FIXED
- **Source**: Mode A F9 (Claude-unique)
- **Code**: 无 invariant 测试 — 重构任何 silent drift 无拦截网
- **Attack**: Future refactor (advisor 改 N/A definition / 改 prereg / lazy refactor 顺手加 alias) 没 fail-loud signal
- **Fix**: `tests/test_fp_architecture_invariants.py` NEW with 9 invariants: scored_task_count post-exclusion values / strict raise / non-strict fallback / N/A single-source / `_is_na_task` contract / `EpisodeSummaryV2` 无 retired fields / `analysis.py` 无 `compute_adjusted_success*` / `analyze_cross_representation` 无 `_mark_false_positives` / `exclude_na_tasks` default True。`pytest` 全 9 pass

**B-numbers consumed**: B-237 through B-253 (17 contiguous, no collisions). Cumulative session work A1.4a+A1.4b-i+A1.4b-ii+A1.4c+A1.5+A1.13+A1.6 = B-140 through B-253 = 114 unique entries.

**Next available B-number**: B-254+.

**Deferred (per user 2026-05-16 decision)**:
- Mode A F1 (mechanism_per_task `adjusted_success` key emit) — §5 advisor-defer 2026-05-14
- Mode B F15 (mechanism `B1_STEP_DIRS` unused) — §5 deferred
- Mode C B-122 visual_fp 删除影响 — user explicit "不需要考虑"
- Mode C B-119 B-91 whitespace bypass — defer (待 Phase 1a clean-run pred 分布数据)
- Mode C B-121 B-86 GLM asymmetry — defer (待 advisor 2026-05-14 question reply)
- Mode C B-120 preregistration "lock" 2026-05-14 timing prose — defer (paper §3 prose decision)

---

## /stress A1.7 fix-batch — `conditions.py` + `configs/*.yaml` (2026-05-16)

Phase 1a fire prep audit per phase1_plan §A1 ladder item 7。3-AI cross-AI cycle (Mode A Claude + Mode B codex ML-systems / Mode C gemini reproducibility) → 17 unique findings → 12 fix-tagged entries B-261~B-272 (+ 2 subsumed)。User Q11 directive: "所有 P0+P1 全修再 fire"。Chronicle entry: 实验笔记 §159。

### B-261. phantom_dom legacy alias retired → phantom_text canonical 🛠️ FIXED
- **Source**: A1.7 Mode A F1 + Mode B F1 ext + Mode C F2 cross-validate (3-AI confirmed). User explicit fix directive 2026-05-16.
- **Code**: `configs/exp_v2_B*_phantom_*.yaml` (×19 files) had `observation_mode: ["phantom_dom"]` + `B1_phantom_dom_classifieds.yaml` dup with `B1_phantom_text_classifieds.yaml` → same `condition_id=phase1_phantom_dom_router_0` (per `conditions.py:126`); + `resume: true` (line 17) enabled silent episode reuse across paper-§3-named arms.
- **Attack**: paper §3 P-text figure 底层数据可能是 phantom_dom 旧 episode 数据 (resume:true ×legacy alias × shared condition_id);Mode B codex 升级 attack vector vs Claude original silent-overwrite framing.
- **Fix**: (a) 19 yaml `obs_mode` sed `phantom_dom → phantom_text` + 删 legacy alias comment + rm dup `B1_phantom_dom_classifieds.yaml`;(b) `conditions.py:84-103` 加 obs_mode enum guard + `_DEPRECATED_OBS_MODES` dict `phantom_dom → phantom_text` raise (fail-loud);(c) `phase1a_status.sh:60-66` 改 phantom_text → `phase1_phantom_text_router_0` (canonical);(d) `queue_phantom_text.sh` 删 legacy fallback (legacy yaml 已删,fail-loud)。Archive `phase1_phantom_dom_router_0/` dirs historical (3 pre-A1.7 instances) — run_registry.py 已 backward-compat `phantom_dom + phantom_text → "P-text"` (line 23-24)

### B-262. B0 use_glm_fallback per-yaml override base B-145 fix 📋 ADVISOR PENDING
- **Source**: A1.7 Mode A F2 + Mode B F2 + Mode C cross-validate (~41 B0 per-yaml). User directive 2026-05-16: write to `docs/checkpoints/parse_advisor_pending.md`.
- **Code**: `configs/exp_v2_base.yaml:166` B-145 fix `use_glm_fallback: false` + "never enable for paper-grade runs"。**~41 B0 per-yaml** override `use_glm_fallback: true` (cls/red/shop/wa-* × 6 modes + pilot)。Empirical GLM rescue trigger rate 1.488% (453/30437 archive steps per `parse_advisor_pending.md §2`).
- **Blast**: Cross-baseline cost-fairness violation;B0 effective backend = Qwen3-VL-235B + GLM-5.1 recovery;B1/B2 single model only。Reviewer 拿 run_meta `use_glm_fallback=true` 看到 paper §3 "B-145 fix" 与 effective config 矛盾.
- **Status**: routed to `parse_advisor_pending.md §4` Thread 1 (B-86 GLM thread, opened 2026-05-14)。Option A/B/C decision pending advisor sync on Qwen official API channel (exposes `tool_choice` → removes parse-error root cause → GLM rescue auto-defuses)。
- **Fix path**: see `parse_advisor_pending.md` §3 Decision branches — code-align (Option B) is 5min sed when advisor signals。Not unilateral.

### B-263. configs/exp_v2_phase1.yaml master dead modes retired 🛠️ FIXED
- **Source**: A1.7 Mode A F3 + Mode C F3 cross-validate.
- **Code**: `configs/exp_v2_phase1.yaml:7-10` had `primary.som: [false, true]` + `primary.observation_mode: ["dom_only", "hybrid"]` (Phase 1 v1 router design artifacts,不在 paper-1 6-mode enum)。CLAUDE.md:45 + README.md:37 列为"实验入口"。`conditions.py:84` 无 enum validation → silent generation `phase1_dom_only_router_0` / `phase1_hybrid_router_0` invalid mode tags propagating downstream.
- **Attack**: Latent footgun: 新人 / fallback automation 用 `--config configs/exp_v2_phase1.yaml` → results/episodes/condition_meta 全部 invalid mode tag → aggregator filter `LIKE 'phase1_dom_router_0'` silent drop OR 误分类入 paper §1。
- **Fix**: (a) rm `configs/exp_v2_phase1.yaml`;(b) CLAUDE.md:45 + README.md:37 redirect to per-condition yaml + footnote "master phase1.yaml retired B-263 2026-05-16";(c) `conditions.py:85-103` `_DEPRECATED_OBS_MODES` 显式 raise on `dom_only` / `hybrid` (fail-loud + 错误信息引用 valid enum)。

### B-264. N_conditions "三头案" Contract Drift unified → 42 🛠️ FIXED
- **Source**: A1.7 Mode C G1 (gemini-specific OOB,Mode A F4 升级版)。User directive 2026-05-16 "preregistration.md 没有OSF 公开，是pending，应该是42"。
- **Code**: 三 doc 三数字: yaml header 40+ files "24-condition Phase 1a scope (2 sites × 2 models × 6 modes)" / `preregistration.md §4 N_conditions` row "36 operational conditions" / `phase1_plan §A` "42 conditions = 36 Pass-1 baseline + 6 Pass-2 router"。3 来源互相矛盾,reviewer 不论拿哪个 doc 都跟其他 2 doc 矛盾。
- **Attack**: Contract integrity 致命。Phase 1a fire 之后 reviewer audit 现场暴露 = trust loss + post-hoc 嫌疑 (H10 router 在 §2 描述但 §4 数字没算 → 像 fire 完才决定加 H10)。preregistration 是 pre-OSF pending → 可 amend。
- **Fix**: (a) 12 yaml header sed → "42-condition Phase 1a scope: Pass-1 baseline 36 (2 sites × 3 models × 6 modes; Gemma3-VL added 2026-05-14) + Pass-2 learned router 6";(b) `preregistration.md §4 N_conditions row` 改为 42 split Pass-1/Pass-2;(c) DOI scope claim §1 updated;(d) `scope_revision_2026_05_16` entry added 解释 accounting fix (H10 was already §2 gating hypothesis,no new experimental scope)。**B-267 subsumed by this unification**。

### B-265. vision mode silent text-fallback on missing image 🛠️ FIXED
- **Source**: A1.7 Mode B F4 (NEW codex-specific OOB).
- **Code**: `p79/experiment/som.py:269-276` (pre-fix) vision branch `marked_image=getattr(obs, "image", None)` — None silently 传到 agent → vision mode 变 text-shaped when screenshot capture failed / payload drop。Yaml 层无 `require_image` knob;step_record 无 `image_present` field。
- **Attack**: paper §3 "vision = raw screenshot only" claim 不能从 config 或 step trace 证明。Reviewer "你 vision condition 我怎么知道不是被 force-fallback 成 dom 跑的?" 无 defuse。
- **Fix**: `som.py:269-289` vision branch 加 fail-loud raise `if image is None: raise ValueError("Vision mode requires image observation but obs.image is None. Paper §3 ... contract violated.")` (B-265 fix)。Failed episode 被 runner outer try/except 捕获 → marked `image_missing` reason → step_record audit trail。Reviewer 可证 "image-only contract enforced"。

### B-266. pilot T0 yamls abandoned 🛠️ DELETED
- **Source**: A1.7 Mode B F8 + user explicit "pilot放弃" 2026-05-16.
- **Code**: `configs/exp_v2_B0_dom_pilot_T0_{classifieds,reddit,shopping}.yaml` (Apr 30 pilot); pilot header claim "T=0 + RNG seeding" 但 yaml 无 explicit `seed` field + `use_glm_fallback: true` (B-262 family); `queue_pilot_t0.sh` + `compare_pilot_t0_vs_paper_grade.py` 引用。
- **Attack**: pilot purpose (isolate T=0 vs T=0.1 effect) 被 GLM fallback 污染;paper §3 引 pilot 数据 reviewer 一查 yaml 立即 invalidate。Pilot superseded by Phase 1a fresh fire on A100。
- **Fix**: `git rm` 3 pilot yamls + `queue_pilot_t0.sh` 加 deprecation banner + `exit 2` (refuses launch); archive pilot results preserved at `results/visualwebarena/phase1/B0_*_pilot_T0_*/` (historical reference,not paper-grade)。

### B-267. H10 learned router in preregistration §2 但 §4 count omit 📋 SUBSUMED by B-264
- **Source**: A1.7 Mode C G2 (gemini-specific).
- **Code**: `docs/checkpoints/pre_run/preregistration.md §2` 详描 H10 hypothesis testing;`§4 N_conditions` table 只列 36 baseline,不含 H10 的 6 router conditions。
- **Attack**: "Post-hoc 嫌疑": reviewer 看 §2 H10 preregistered → §4 condition count 没匹配数字 → "你跑完 baseline 才决定加 H10?" Pre-registration breach (weak version)。
- **Fix**: Subsumed by B-264 unified 42 = 36 + 6 (H10 row now explicit in §4 + scope_revision_2026_05_16 entry explains accounting fix not new scope)。

### B-268. router_learned yaml TODO ↔ runner main.py:1017 dispatch land drift 🛠️ FIXED
- **Source**: A1.7 Mode A F5 (Claude-unique).
- **Code**: `configs/exp_v2_B*_router_learned_*.yaml:5-8` 注释 "⚠️ RUNTIME LR INTEGRATION TODO ... pending separate session"。但 `p79/experiment/runner/main.py:1011-1059` 已实现 `obs_mode == "learned"` dispatch (commit `5c33103` 2026-05-16 by parallel router session)。文档 stale。
- **Attack**: Documentation drift → launch operator 信 TODO 注释 → 延迟 Pass-2 fire OR 不知 `lr_model_path` artifact 必须存在前置。
- **Fix**: (a) B0/B1 cls/red 4 yaml 注释更新 "✅ RUNTIME LR INTEGRATION DONE (B-268 verify 2026-05-16)" + 引用 `runner main.py:1011-1059` + `train_l1_router.py artifact verified at lr_model_path (May 16 12:28 land)`;(b) **B2 (Gemma3-VL) artifact MISSING**: 2 B2 yaml 注释 "🚧 LR ARTIFACT BLOCKED — `B2_classifieds_lr.pkl` / `B2_reddit_lr.pkl` 未生成 (Pass-1 B2 数据 land 后 train via `train_l1_router.py --baseline B2`)"。Phase 1a Pass-2 fire 6 cell → 实际 4 cell available + 2 BLOCKED 待 B2 train.

### B-269. baselines.run_b0 dead-code retired 🛠️ FIXED
- **Source**: A1.7 Mode A F6 + Mode B F7 cross-validate.
- **Code**: `conditions.py:298-312` `if baselines.get("run_b0", False)` 追加 extra `b0_strong_upper_bound` condition (Phase 1 v1 "B1-only + B0 upper bound" design)。119 yaml 末尾携带 `baselines: run_b0: false` block。
- **Attack**: Dead-code perpetuation + misleading framing (reviewer 误以为 Phase 1a B1-only + B0 disabled);risk accident set True → 跑出 duplicate b0_strong_upper_bound condition 跟 phase1_dom_router_0 同 run_dir。
- **Fix**: (a) sed delete `baselines:\n  run_b0: false` block 跨 119 yaml;(b) `conditions.py:298-312` 加 `if phase == "phase1" and baselines.get("run_b0"): raise ValueError` (Phase 2/3 仍 allow,paper-2 substrate);(c) error message 引导 user 用 per-baseline yaml。

### B-270. min_free_vram_gb=0 disable OOM safety in B1/B2 per-yaml 🛠️ FIXED
- **Source**: A1.7 Mode A F7 + Mode B F3 cross-validate.
- **Code**: 全 B1/B2 per-yaml 显式 `min_free_vram_gb: 0` (76 files 受影响);base.yaml 无 default → 0 disables safety gate.
- **Attack**: Shared GPU mid-experiment OOM → silent failure 看起来像 model weakness。A100-PCIE-40GB self-hosted VWA Docker 共占,leeway 紧。
- **Fix**: (a) `configs/exp_v2_base.yaml:107-130` `local_4b` + `local_gemma` 加 `min_free_vram_gb: 12` default (4B model bf16 ~10GB + VWA Docker margin);(b) sed delete 76 per-yaml `min_free_vram_gb: 0` override (deep-merge inherits 12 from base);(c) per-condition override allowed but discouraged via comment。

### B-271. B1/B2 per-yaml decoding fields not explicitly pinned 🛠️ FIXED (defense in depth)
- **Source**: A1.7 Mode B F6 (NEW codex-specific).
- **Code**: Pre-fix B0 per-yaml 显式 `temperature: 0.0`;B1/B2 per-yaml 只 set `max_new_tokens: 4096`,decoding fields 全 inherit base。`config.py:95-104` `_merge_dict` is **deep merge** → defused current bug,but hypothetical: if base.yaml 改 default,B1/B2 drift but B0 不动.
- **Attack**: cost-aware paper 最怕 success/cost differences = decoding-policy differences hidden in default inheritance。
- **Fix**: sed insert `temperature: 0.0  # B-271 (2026-05-16, A1.7): explicit pin for paper-grade audit parity with B0` after `max_new_tokens: 4096` 跨 40 B1/B2 cls/red/shop/wa per-yaml。Defense in depth — base.yaml 已 0.0 但 per-yaml 显式 pin 让 run_meta 三 baseline 完全 symmetric record。

### B-272. phantom_<site>.yaml filename asym → phantom_som_<site>.yaml 🛠️ FIXED
- **Source**: A1.7 Mode A F9 (Claude-unique).
- **Code**: Pre-fix filenames asym: `phantom_classifieds.yaml` (= phantom_som,content `obs_mode="phantom_som"`) vs sibling `phantom_text_classifieds.yaml` / `phantom_prompt_classifieds.yaml` (有 mode 后缀)。Glob `exp_v2_B0_phantom_*_classifieds.yaml` miss phantom_som。
- **Attack**: 视觉混淆 + 自动化 glob skip phantom_som condition。
- **Fix**: (a) git mv 18 yamls `B*_phantom_<site>.yaml` → `B*_phantom_som_<site>.yaml` (跨 3 站 × 3 模型 × WA);(b) `queue_phantom_som.sh:59` `CFG_NAME="${BASELINE}_phantom"` → `"${BASELINE}_phantom_som"`;(c) `rsync_results_from_hub.sh` comment refresh (just doc reference). 现 3 phantom arms 全有 mode 后缀对称: phantom_som / phantom_text / phantom_prompt。

**B-numbers consumed**: B-261 through B-272 (12 contiguous + 2 subsumed: B-267 → B-264, namespace overlap → B-261)。Cumulative session work A1.4a+A1.4b-i+A1.4b-ii+A1.4c+A1.5+A1.13+A1.6+A1.7 = B-140 through B-272 = ~126 unique entries. Gap B-254~B-260 reserved (in case A1.6 amendments).

**Next available B-number**: B-280+ (after A1.16 batch below).

**Renumber drama** (chronicle §159.4): initial draft B-230~B-243 → conflict A1.13+A1.14; second pass B-237~B-249 → conflict A1.6 uncommitted; final B-261~B-272 safe gap above all parallel sessions。Lesson: pre-batch grep `git diff HEAD docs/reference/master_bug_catalog.md | grep "^+### B-"` 加 sequential allocator (future cron 实现)。


---

## A1.16 /stress audit (2026-05-16) — B-273 to B-279 (7 entries; 7 fixed, 0 deferred, 5 mechanism-script-deferred per Q16 user decision)

> Cross-AI: Mode A Claude 8 findings / 5 OOB + Mode B codex retry 9 findings / 2 OOB labels + Mode C gemini 7 findings / 2 OOB + 2 P0 unique attack vectors = 22 raw → **11 unique** after dedup. Scope = `scripts/provenance/snapshot_*` (3 files / 519 LOC paper §3 reproducibility anchors). User Q1 Option A (paper-1 critical, 7 fix); Q16 (i) mechanism `numerical_determinism_check.py` 5 deferred bugs (P0-2/P0-3/P1-5/P1-6/P1-7) tracked via `phase1_plan.md` §A1 pointer per advisor 2026-05-14 mechanism defer.

### B-273. snapshot_vwa.sh `/` probe → session-stateful body_sha256 🛠️ FIXED
- **Source**: Claude Mode A A-1 + codex C-6 + gemini G-5 (3-AI overlap, P0)
- **Code**: `scripts/provenance/snapshot_vwa.sh:88-91` pre-fix `url = f"http://${VWA_HOST}:{port}/"` then `curl ... | sha256sum`
- **Attack**: Magento `/` 含 cart token + Postmill `/` 含 user feed + OSClass `/` 含 last_login → 同站同脚本两次 capture body_sha256 都不同。Paper §3 Appendix D "byte-equivalence" claim 在 byte 一层就站不住,reviewer diff snapshot.json 永远 mismatch → "你这论文不可复现"。
- **Fix**: 改 probe path 到 `/robots.txt` (server-config static,no session cookies,deterministic for given VWA submodule source)。配 3 env override (`VWA_PROBE_PATH_CLS/RED/SHOP`) + `-b /dev/null -c /dev/null` disable cookies。`probe_kind: "static-asset"` 写进 snap["sites"] 标识当前 probe 类型 (paper-2 reviewer 可读 schema)。

### B-274. snapshot_vwa.sh bash heredoc command injection vector 🛠️ FIXED
- **Source**: Claude Mode A A-2 + codex C-5 + gemini G-6 (3-AI overlap, severity split P0/P1/P2)
- **Code**: `scripts/provenance/snapshot_vwa.sh:39, 54-56, 88` pre-fix `python3 - <<PYEOF` + `"host": "$HOST"` 等 bash interpolation
- **Attack**: `VWA_HOST='x";import os;os.system("evil");#'` → Python source 执行任意代码。Paper-grade OSF reviewer 跑 untrusted env 才有 real risk 但开源代码 lint signal 弱;3-AI 严重性分歧 (Claude P0 / codex P1 / gemini P2),P2 接受 (paper-grade lint not real attack surface)。
- **Fix**: heredoc 改 `<<'PYEOF'` quoted form (closes bash interpolation) + env export 全 host vars + Python 内 `os.environ.get()` 取值。所有外部值 typed-validated (`int(os.environ["CLS_PORT"])`)。

### B-275. snapshot_env.py HF SHA captures registry HEAD, not loaded cache 🛠️ FIXED
- **Source**: Claude Mode A A-3 + codex C-1 + gemini G-1 (3-AI overlap, P1)
- **Code**: `scripts/provenance/snapshot_env.py:123-130` pre-fix `HfApi().model_info(model_id).sha`
- **Attack**: HF 滚 main 分支 → snapshot 记 NEW SHA, runner 用 cache OLD SHA. Paper §3 "model SHA pinned at launch" misleading — pinned 的是 registry HEAD 不是 actually-used revision. Reviewer `from_pretrained(model_id, revision=<paper-SHA>)` 拉到 new weights → SR 不匹配 paper 数字。
- **Fix**: 新 `_loaded_revision_from_cache(model_id)` 用 `huggingface_hub.scan_cache_dir()` 读 local cache 最 recent revision。`_capture_model_revisions()` 输出双字段: `loaded_revision` (PRIMARY) + `registry_head` (SECONDARY) + `divergence` 字段 `match | runner_used_stale_cache | registry_unavailable | no_local_cache | gated_no_token`. Smoke verified: B1 Qwen3-VL-4B-Instruct loaded=`ebb281ec70b05090...` matches registry HEAD → `divergence: match`。
- **Note**: runner-side enforce (`from_pretrained(model_id, revision=<pinned_sha>)`) tracked separately in phase1_plan §A1 pointer (downstream change in runner, not snapshot scope)。

### B-276. snapshot_env.py Gemma gated → silent "unavailable" 🛠️ FIXED
- **Source**: Claude Mode A A-4 + codex C-2 (Claude+codex 2-AI, gemini missed)
- **Code**: `scripts/provenance/snapshot_env.py:58-61` `DEFAULT_MODELS` 含 `google/gemma-3-4b-it` (gated); `:124, 128-131` pre-fix `HfApi()` 无 token + `_safe` 静默 → "unavailable"
- **Attack**: B2 model SHA 永远 unavailable, paper §3 / Appendix D B2 anchor 缺失。Reviewer 不能 verify B2 → "凭啥说你 B2 真用了 gemma-3-4b-it?" Cross-family claim 失锚。
- **Fix**: 新 `DEFAULT_GATED_MODELS` list (paper-baseline gated)。`_capture_model_revisions()` 检查 `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` env, 缺失 + gated 模型 → 立即 set `_critical_error` + errors entry + log.error()。`--strict` CLI flag enables exit non-zero for paper-grade Phase 1a launch wrap (caller `bash queue_phase1_paper_grade.sh` 可加 `snapshot_env --strict` 在 fire 前 gate)。Smoke: B2 missing token → `divergence: gated_no_token` + 1 errors entry surfaced。

### B-277. snapshot_env.py evaluator combined_sha256 silent MISSING + order-sensitive 🛠️ FIXED
- **Source**: Claude Mode A A-5 + codex C-3 + C-4 + gemini G-3 (3-AI overlap, P1)
- **Code**: `scripts/provenance/snapshot_env.py:139-152` pre-fix `if not f.exists(): per_file[rel_path]="MISSING"; continue` + `for rel_path in EVALUATOR_SOURCE_FILES` (list-order iteration)
- **Attack**: (1) Reviewer 没 `git submodule init` → `helper_functions.py` MISSING → silent skip → `combined_sha256` 用剩 3 文件计算 → reviewer 跟 paper 的 hash 对不上 **不报错**. (2) Future PR 重排 list → hash 改 byte 不变 → false-positive regression alert。
- **Fix**: 新 `_evaluator_combined_sha()` (a) `raise FileNotFoundError` on MISSING (paper-grade fail-loud) (b) `EVALUATOR_SOURCE_FILES = sorted([...])` 强制 canonical order (c) hash form: `sha256(rel_path \0 byte_len \0 content \0)` path-aware sentinel-delimited (defeats rename-but-same-content attack + size-detect)。`schema_version: "2026-05-16-canonical-v2"` 字段标记新格式。

### B-278. snapshot_env.py evaluator scope 过窄 (configs + utils missing) 🛠️ FIXED
- **Source**: gemini Mode C G-3 (gemini-unique OOB attack vector)
- **Code**: `scripts/provenance/snapshot_env.py:45-52` pre-fix 4-file list (analysis + environment + metrics + helper_functions)
- **Attack**: YAML `max_steps` 改 → SR 基准变, 但 `combined_sha256` 不变 → paper §3 "scoring logic pinnable" 失效。`p79/utils/auth_refresh.py` 决定 N/A task 跑没跑也直接影响 SR。
- **Fix**: list 扩展到 6 explicit files + `EVALUATOR_CONFIG_GLOB = "configs/exp_v2_*.yaml"` 加 119 config files + 新 `evaluators.py` (string_match / url_match / program_html) + `auth_refresh.py` (auth gate task-level)。Plus inline `pip_freeze_lock` 全 deps lock 进 snap (Q6 option C 推荐) — paper §3 full env reproducibility. Smoke verified: 6 core + 119 configs = 125 files captured。

### B-279. snapshot_vwa.sh docker image_id non-portable + VWA source SHA missing 🛠️ FIXED
- **Source**: Claude Mode A A-8 + codex C-7 (Claude+codex 2-AI, P1)
- **Code**: `scripts/provenance/snapshot_vwa.sh:69` pre-fix `docker inspect --format={{.Image}} ${cid}` returns local layer storage hash (non-portable across hosts); 完全无 VWA submodule SHA / Dockerfile / docker-compose fingerprint
- **Attack**: (1) image_id mismatch 跨 dgx vs a100 even with same registry pull → reviewer 以为 deploy 不同实际 manifest 一致。(2) Rebuild docker with same tag but new VWA source → snap looks unchanged, evaluator code 实际不同 → byte-equivalence claim broken。
- **Fix**: (a) `repo_digests` 升级为 PRIMARY (registry-canonical, portable) + `image_id_full` 标 "local-only, non-portable" comment;(b) 新 `snap["vwa_source"]` section: `submodule_sha` + `submodule_branch` + `submodule_dirty` (含 dirty_files list) + `dockerfile_combined_sha256` (canonical sha256 over sorted Dockerfile + docker-compose*.yml files); rglob covers nested dirs。

**B-numbers consumed**: B-273 through B-279 (7 contiguous, no collisions). Cumulative session work (A1.4a+A1.4b-i+A1.4b-ii+A1.4c+A1.5+A1.13+A1.6+A1.7+A1.16) = B-140 through B-279 = ~133 unique entries.

**Mechanism-script bugs DEFERRED** (Q16 user decision =留 + pointer): 5 bugs in `scripts/provenance/numerical_determinism_check.py` 跟 paper §5 cross-machine mechanism quote 绑定; advisor 2026-05-14 显式 defer mechanism。Bug list tracked via `phase1_plan.md §A1` pointer "A1.16-mechanism subset deferred per paper-2 scope":
- (D-1) TF32 matmul blindness (gemini G-2, P0) — `numerical_determinism_check.py:79-81` 不设 `torch.backends.cuda.matmul.allow_tf32`
- (D-2) Model loading dtype non-determinism (gemini G-4, P0) — `:75` `HiddenStateExtractor` 不传 `torch_dtype`
- (D-3) `external_code` path typo (Claude A-6, P1) — `:67` silent fallback to episode summary intent
- (D-4) pass_threshold default 1e-2 vs docstring 1e-3 (Claude A-7 + codex C-8, P1) — `:201-202`
- (D-5) Capture input not SHA-pinned across machines (codex C-9, P1 OOB) — `:59-77`

These 5 bugs are paper-1 out-of-scope (mechanism deferred); if paper-2 mechanism work resumes, this batch becomes paper-grade gate for cross-machine determinism quotes.

**Next available B-number**: B-280+.

**Phase 1a fire green-light (per user Q11)**: all 12 P0+P1 fix landed + smoke verified (pytest 16/16 + conditions.py enum + som.py vision raise + base.yaml deep-merge inheritance)。Remaining advisor blocker: B-262 (glm_fallback) per `parse_advisor_pending.md` Thread 1。

---

## /stress A1.8 fix-batch — schema + JSONL + dedup substrate (2026-05-16)

Phase 1a fire prep audit per phase1_plan §A1 ladder item 8。3-AI cross-AI cycle (Claude Mode A 9 + codex Mode B 7 + gemini Mode C 4 / 18 unique after dedupe) → 18 fix-tagged entries B-280~B-297。User Q-all directive "全推荐" 2026-05-16: all defaults accepted。Chronicle: 实验笔记 §161。

### B-280. `validate_step_record_v2` shallow → dataclass-derive + per-field type + critical optionals 🛠️ FIXED
- **Source**: A1.7 Mode A F1 + Mode C G3 cross-validate
- **Code**: `p79/experiment/types.py:195-231` pre-fix 25 hardcoded REQUIRED + schema_version equality only;无 type check + 12 optionals 漏(`parse_valid`/`image_meta`/`locator_route_meta`...)
- **Attack**: malformed step record `{"som": null, "latency_ms": "0", "image_meta": "not-dict"}` 静默通过 → paper §3 evidence layer 数据基础 reviewer-defensible 度低
- **Fix**: 重写 `validate_step_record_v2` 用 `dataclasses.fields(StepRecordV2)` 派生 REQUIRED + per-field `_STEP_FIELD_TYPES` mapping + `PAPER_GRADE_STEP_OPTIONAL_KEYS` 必 present-or-None。Plus new `validate_episode_summary_v2` + `validate_run_summary_v2` 同设计

### B-281. `REQUIRED_STEP_FIELDS_V2` hand-maintained set drifts from dataclass 🛠️ FIXED (subsumed in B-280)
- **Source**: A1.7 Mode A F2 + Mode B F1
- **Fix**: `_required_field_names(cls)` helper 派生 frozenset from dataclasses.fields with MISSING defaults。Now StepRecordV2 / EpisodeSummaryV2 / RunSummaryV2 各自动 derived

### B-282. SCHEMA_VERSION_V2 "2.0" vs `_CHAIN=["v2"]` — v3 migration latent broken 🛠️ FIXED
- **Source**: A1.7 Mode A F3 (Claude-unique OOB)
- **Code**: `types.py:6` `SCHEMA_VERSION_V2 = "2.0"`(semver)vs `schema_migrations/__init__.py:43` `_CHAIN = ["v2"]`(short)
- **Attack**: V3 work land 时,migrate("2.0", "3.0") raise unknown version,**任意 v3 work 启动即引爆**
- **Fix**: 全 unify semver `_CHAIN = ["2.0"]` + docstring example 更新 + future `_CHAIN.append("3.0")` 路径 documented。Archive `schema_version="2.0"` 兼容

### B-283. `bool(row.get("success", False))` truthy on string "false" 🛠️ FIXED (paper §1 hero protection)
- **Source**: A1.7 Mode B F3 (codex-unique OOB, Mode A + C 完全 missed)
- **Code**: `aggregate_sr_fp_per_mode.py:77-80` + `aggregate_phantom_lift.py:109-116` 用 `bool(...)`
- **Attack**: **Paper §1 hero SR 静默抬高 vector** — JSON string "false" Python truthy,missing field default False → SR 静默压低
- **Fix**: 新 `p79/experiment/io_utils.py::load_episode_summary_strict(path, mode)` — strict mode raise on `success != bool` / `task_id != int`;lenient mode log + return None。2 个 paper-1 aggregator 切 strict (Phase 1a fire ENV P79_STRICT=1 启用)

### B-284. "Ghost fields" runner-write off-catalog → 加 5 field to dataclass + DEFAULTS 🛠️ FIXED
- **Source**: A1.7 Mode C G1 (gemini-unique OOB)
- **Code**: `runner/main.py:1663-1669` 写 `retry_action_applied` / `retry_action_type` / `glm_fallback_used` / `glm_fallback_latency_ms` / `glm_original_fail_reason`;但 **不在 dataclass + DEFAULTS**
- **Attack**: paper §3.5.1 cite 这些 field 做 GLM de-biasing 但 catalog 找不到 → "黑户字段"
- **Fix**: 5 field 加 `StepRecordV2` dataclass + `STEP_RECORD_V2_DEFAULTS`(均 Optional[T] = None default)。Dataclass + DEFAULTS 现 43=43 fully synced。`fill_step_defaults` 现 backfill 这 5 field,archive backward-compat

### B-285. EpisodeSummaryV2 无 validator function 🛠️ FIXED
- **Source**: A1.7 Mode A F4 + Mode B F2 cross-validate
- **Fix**: 新 `validate_episode_summary_v2(record)` — REQUIRED 27 fields + per-field type check on hero fields (`success: bool` / `score: int|float` / `steps: int` / `task_id: int`)。LoggerV2.write_episode_summary 入口 caller 应 invoke(未自动 wire,paper-1 fire 后 add)

### B-286. rederive_episode_summary.py 不验 step ↔ summary identity 🛠️ FIXED
- **Source**: A1.7 Mode B F6 (codex-unique OOB)
- **Code**: `scripts/maintenance/rederive_episode_summary.py:132` `read_jsonl_dedup(steps_path)` 不传 summary_path → 跳过 B-180 identity check
- **Attack**: restart/crash 留下 mismatched files → rederive 用 task B steps 覆盖 task A summary,无 audit log → **paper-grade evidence 不可逆 corruption**
- **Fix**: 改为 `read_jsonl_dedup(steps_path, summary_path=summary_path)`;mismatch flag → skip rederive + log。Rederive 比 analysis read stricter (mutates evidence)

### B-287. `dedup_restart_lines` 文件无 step_idx=0 silent keep-all 🛠️ FIXED
- **Source**: A1.7 Mode A F5 + Mode B F5 cross-validate
- **Code**: `io_utils.py:32-35` 找 LAST `step_idx==0`,若不存在则 `last_run_start=0` → keep all
- **Attack**: First-step crash + restart 续写 → dedup 不识别 restart → 双 run 数据混入
- **Fix**: 加 decrement-boundary fallback(若无 step_idx=0,扫描 prev > curr 边界 as restart);新 `_assert_step_idx_monotonic()` invariant post-dedup 推断 corruption + 写 integrity log `step_idx_non_monotonic: bool` flag

### B-288. `read_jsonl_dedup` 不 catch UnicodeDecodeError 🛠️ FIXED
- **Source**: A1.7 Mode A F11 + Mode B F4 (downstream cascade)
- **Code**: `io_utils.py:64` `open(path, "r", encoding="utf-8")` — UnicodeDecodeError 不在 catch 范围
- **Attack**: 单个 invalid UTF-8 byte → analyze pipeline crash → paper §3 整 cell skip
- **Fix**: `open(..., errors="replace")` in `io_utils.py` + 4 places in `scripts/analysis/validate_run.py:545/581/645/765`

### B-289. LoggerV2.__init__ 不 fsync condition_dir 🛠️ FIXED
- **Source**: A1.7 Mode A F7 (Claude-unique OOB)
- **Code**: `logger_v2.py:31-32` `episodes_dir.mkdir(parents=True)` — 无 _fsync_dir
- **Attack**: Mirror B-198: mkdir 后 crash → reboot 见不到 episodes/ → run data partial visible
- **Fix**: `_fsync_dir(self.condition_dir) + _fsync_dir(self.episodes_dir)` in __init__

### B-290. dedup discarded earlier segments 无 audit trail 🛠️ DEFERRED (sidecar archive)
- **Source**: A1.7 Mode A F8 (Claude-unique)
- **Status**: Integrity log entry 新增 `discarded_segments_archive: Optional[str]` field(currently None);完整 sidecar `.restart_archive.jsonl` 写实现 deferred(non-blocker,Phase 1a fire 不影响)
- **Fix**: Schema-level prep land,write-side 待 future audit demand

### B-291. `image_meta=None` 语义二义性 → `image_meta_recorded: bool` 加 separator 🛠️ FIXED
- **Source**: A1.7 Mode C G2 (gemini-unique OOB)
- **Code**: `schema_migrations/v2.py:134` `image_meta: None` default → vision-mode-no-image vs old-data-missing 不区分
- **Fix**: `StepRecordV2` 加 `image_meta_recorded: bool = False`;DEFAULTS 加 False(archive lineage flag)。Paper §3 image_over_cap claim 现可证 archive vs new fire 区别

### B-292. write_step fsync ~14min/Phase 1a overhead 🛠️ DISCLOSED (no code change)
- **Source**: A1.7 Mode A F6 (Claude-unique)
- **Status**: B-198 hero design (per-step crash atomicity) — 不动 code
- **Fix**: `docs/checkpoints/paper_drafts/section3_definition.md` "Schema integrity guarantees" 段加 paragraph: "Per-step fsync is intentional ... 14 minutes of disk-flush time ... we do not subtract this from latency_ms.total because all 3 baselines incur equal overhead ... reviewers should account for durability-first write path"

### B-293. read_jsonl_dedup `summary_path=None` 时 `summary_identity_mismatch=False` 误导 🛠️ FIXED
- **Source**: A1.7 Mode A F9 (Claude-unique)
- **Code**: `io_utils.py:79-80` pre-fix `identity_mismatch=False`(无 summary 也写 False)
- **Fix**: `Optional[bool] = None` — None = "未检",False/True = "检了+matched/mismatch"。`jsonl_integrity_report.csv` reviewer audit 现可 filter `is True` 拿到真 mismatch

### B-294. `schema_migrations.migrate()` dead code (0 callers) 🛠️ FIXED (tests added)
- **Source**: A1.7 Mode A F10 (Claude-unique)
- **Fix**: 新 `tests/test_schema_migrations.py` 6 tests: SCHEMA_VERSION align _CHAIN + migrate identity + unknown version raise + downgrade refuse + mock v2→vtest chain + deepcopy + idempotency。Future v3 work 可信 framework + test gate

### B-295. Step schema tests 不覆盖 type drift / drift 🛠️ FIXED
- **Source**: A1.7 Mode B F1
- **Fix**: 新 `tests/test_step_record_validation.py` 35 tests parametrized:
  - 9 type-mismatch cases (som=None / latency_ms=str / task_id=str / seed=float / action_success=str / page_changed=int / reward=str / tokens=list / som=str)
  - 5 critical-optional-missing cases (per `PAPER_GRADE_STEP_OPTIONAL_KEYS`)
  - 7 episode-summary string-truthy success cases ("false"/"true"/"0"/0/1/None/1.0)
  - 2 episode-summary other-type cases (steps=str / score=str)
  - 2 run-summary type cases (condition_metrics=dict / assumptions=list)
  - 3 happy-path cases (valid step / episode / run summary)
  - 2 schema_version cases ("v2" / "1.0" / float 2.0)

### B-296. RunSummaryV2 无 validator function 🛠️ FIXED
- **Source**: A1.7 Mode A G1 (Claude-unique gap)
- **Fix**: 新 `validate_run_summary_v2(record)` — REQUIRED 8 fields + `condition_metrics: list` + `assumptions: dict` + `total_episodes: int`。runner main.py write run_summary 入口 future wire (paper-1 fire 后 add)

### B-297. `aggregate_failure_modes.py` regex `B[01]` 漏 B2 + COND_MODE_MAP 漏 phantom_dom 🛠️ FIXED
- **Source**: A1.7 Mode B F7 (codex-unique OOB)
- **Code**: `scripts/analysis/aggregate_failure_modes.py:74` regex hardcoded `B[01]`;line 86-93 mode map 无 phantom_dom
- **Attack**: Phase 1a 3-baseline B2 (Gemma3-VL) failure data + archive phantom_dom failure (3 现存 dir) silent vanish
- **Fix**: regex `B[0-2]` + COND_MODE_MAP 加 `"phantom_dom": "P-text"`(B-261 alias backward-compat)

**B-numbers consumed**: B-280 through B-297 (18 contiguous, no collisions with A1.16 B-273~B-279)。Cumulative session A1.4a+b-i+b-ii+c + A1.5 + A1.13+14 + A1.6 + A1.7 + A1.16 + A1.8 = B-140 through B-297 = ~158 unique entries。

**Next available B-number**: ~~B-298+~~ → superseded by A1.17 Chunk 1 below.

**Smoke verification**: pytest 316/316 PASS (= 281 baseline A1.7 + 35 new A1.8 negative tests)。

**Phase 1a fire green-light**: schema substrate fully paper-grade-defensible post-A1.8。剩 advisor blocker: B-262 (GLM fallback per parse_advisor_pending.md Thread 1); B-130 (FE/RE estimand per Thread 2); B-268 B2 LR artifact (Pass-2 B2 router cells 待 Pass-1 B2 fire 后 train via `train_l1_router.py --baseline B2`)。

---

## /stress A1.17 Chunk 1 — VWA setup + RESET_BEFORE protocol launch-blocker batch (2026-05-16)

3-AI cross-AI cycle (Mode A Claude + Mode B codex + Mode C gemini) on `scripts/vwa/` + `RESET_BEFORE` protocol pre-fire audit. **22 attacks consolidated, Chunk 1 lands 9 P0 + glm-absorb-P1 fixes for Phase 1a launch readiness; Chunk 2 (12 P1 + Option K Trajectory Event Log) deferred to subsequent session per user split-scope decision** (memory `feedback_split_large_scope`).

Cross-AI agreement: 1 3-AI overlap (cls reset sentinel narrow — Chunk 2), 6 2-AI overlaps, 15 1-AI uniques. Codex Mode B critical unique catch = B-302 (queue_chain schema mismatch, LAUNCH BLOCKER — would abort chain after cell 1 完成). Gemini Mode C unique OOB = TZ drift / disk underestimate / Magento indexer race (all Chunk 2).

### B-298. `_lib_paper_grade_gates.sh:50-61` A100 URL-locality preflight hostname pattern miss [P0 — 2-AI overlap A+B OOB] 🛠️ FIXED
- **Attack**: Pre-fix predicate `hostname == *condense* OR -d /home/ubuntu/workspace/p79` fails on canonical target VM `a100-jiaming-test` (hostname has no "condense" substring); only directory fallback held, fragile to user/repo-path layout changes.
- **Cascade**: BUG-2 (B-225) URL-locality preflight 静默不激活 on canonical A100 → quark Tailscale URL leaks through → paper-grade run 数据来自非 A100 站点 → cross-host contam silent into OSF artifact.
- **Fix**: 5-way predicate `*a100* OR *condense* OR P79_PAPER_GRADE_HOST=1 OR cwd=*workspace/p79* OR -d /home/ubuntu/workspace/p79`. Adds explicit `[preflight] gate ACTIVE on host=$HOSTNAME` stderr log proving gate ran. Also adds HOMEPAGE to checked var list (was missing, P2-5 absorbed).

### B-299. `reset_vwa_sites.sh:91-99` shopping reset stub silently returns 0 [P0 — 2-AI overlap A+B] 🛠️ FIXED
- **Attack**: `_reset_vwa_local_shopping` body = `echo "NOT YET IMPLEMENTED"; return 0` → gate at `_lib_paper_grade_gates.sh:143` treats rc==0 as success → Phase 1b shopping fires proceed against dirty Magento (cart/session/search-cache from prior condition).
- **Cascade**: B0/B1/B2 + dom/som/vision/phantom 差异混入 cart 残留 → OSF artifact "reset-clean" but actually cross-condition state accumulation.
- **Fix**: Return 78 (sentinel "not implemented" rc); `_lib_paper_grade_gates.sh:reset_and_auth_gate` branches on rc==78 with specific operator-facing message "implement reset_vwa_local_${site} body before paper-grade Phase 1b launch". Phase 1a unaffected (cls+red don't call shopping stub).

### B-300. `reset_vwa_sites.sh:113-120` Reset auto mode SSH-key routing flip [P0 — 2-AI overlap A+B OOB] 🛠️ FIXED
- **Attack**: Pre-fix auto detect `[[ -f ~/.ssh/vwa_windows ]]` → remote; A100 VM with legacy SSH key from dotfiles/rsync silently routes to quark Windows PowerShell reset rather than local A100 docker.
- **Cascade**: BUG-3 OSClass POST endpoint + mutation sentinel 全跳过 → cls 容器没真 reset → paper-grade condition 从前一 condition 的 dirty state 继续跑.
- **Fix**: Hostname-first auto-detect: `*a100*` / `*condense*` / `P79_PAPER_GRADE_HOST=1` / `/home/ubuntu/workspace/p79` → force `mode=local` regardless of SSH key presence. SSH key heuristic only kicks in on non-A100 hosts. Resolved mode logged to stderr.

### B-301. `queue_chain.sh:178-180` `|| true` masks queue script reset/auth failure [P1 — codex OOB unique] 🛠️ FIXED
- **Attack**: Pre-fix `out=$(... 2>&1 || true)` discarded queue script rc; reset/auth FATAL was hidden because run_id had been printed before reset → chain proceeded to `wait_for_runner_done` finding no runner → declared "runner done" instantly → fell through to silent sentinel check.
- **Cascade**: Cascades with B-302 schema sentinel mismatch → silent partial-data advancement; reset failure looks like normal idempotent skip.
- **Fix**: `set +e; out=$(...); queue_rc=$?; set -e` explicit rc capture. rc!=0 + no run_id minted → FATAL + full output dump + abort. rc!=0 + run_id minted (legacy idempotent-skip case) → warn + continue.

### B-302. `queue_chain.sh:217-219` completion sentinel schema mismatch [P0 — codex OOB unique LAUNCH BLOCKER] 🛠️ FIXED
- **Attack**: Pre-fix sentinel queried `total_tasks / num_tasks / scored_task_count` — empirically verified 2026-05-16 across 5 sample `condition_summary_v2.json` files: top-level schema has `episodes: int` (234/210 cls/red task count) + `condition_id` + 30+ metric keys but **NONE of total_tasks / num_tasks / scored_task_count**. All three .get() return None → total=0 → fail validation → chain abort.
- **Cascade**: Phase 1a launch → cell 1 finishes → sentinel reject "no valid condition_summary_v2.json" → user 被迫 `--no-reset` manual relaunch → cross-condition contam risk soars. **Without this catch, Phase 1a fire would self-abort on first cell completion.**
- **Fix**: Use `episodes` as canonical field + `expected_n` hardcoded per site (cls=234 / red=210 / shop=466 / wa_*) parsed from `run_id` pattern. Validation: `episodes > 0` AND `episodes >= 90% of expected_n` (allows interrupt+resume partial cells, rejects smoke/early-fail). Legacy fallbacks kept for forward-compat. 3 case smoke tested: valid 234→exit 0, ep=0→exit 3, 100/234=43%→exit 4.

### B-303. `phase1a_relaunch_missing.sh:117-122` FORCE_NEW chain-level leakage [P0 — 2-AI overlap A+C] 🛠️ FIXED
- **Attack**: Pre-fix any single PARTIAL condition in chain pulled `force_new=0` for ALL conditions in that chain → PENDING (fresh) conditions went through `mint_run_id` resume-by-glob branch, potentially inheriting stale partial dirs from prior fires.
- **Cascade**: PENDING 应 fresh 的 condition 被静默降级为 resume → JSONL 掺杂两次 fire 的 episodes → paper-grade integrity violation.
- **Fix**: Split each site chain into 2 sub-chains via new `split_by_resume()` helper. Fresh sub-chain: FORCE_NEW=1 + RESET_BEFORE=1 (paper-grade clean launch). Resume sub-chain: FORCE_NEW=0 + RESET_BEFORE=0 (see B-304). Up to 4 sub-chains total (cls_fresh / cls_resume / red_fresh / red_resume).

### B-304. `phase1a_relaunch_missing.sh:125` resume+reset trajectory discontinuity [P1 — codex OOB unique, P1-5-B Tier 1 α'] 🛠️ FIXED
- **Attack**: Pre-fix combined `FORCE_NEW=0` (resume from existing run_dir at episode 120) with `RESET_BEFORE=1` (queue_chain default — wipes site state) → episodes 0-119 ran on cumulative dirty trajectory; episodes 120-233 ran on fresh-reset trajectory; same `condition_summary_v2.json` aggregated both as if single condition. Within-cell state trajectory discontinuity.
- **Cascade**: VWA tasks are NOT i.i.d. — "comment on listing I posted yesterday" type tasks see clean cls in episode 120+ but agent expects mutations from episodes 0-119; SR estimate biased toward state-independent task types post-interrupt.
- **Fix**: Per user 3-AI brainstorm decision (Tier 1 α' = code fix B + paper §3 reframe + GLMM covariate + Fisher homogeneity), resume sub-chains now use `RESET_BEFORE=0` — preserves trajectory continuity. Paper §3 disclosure: "PARTIAL cells resumed without additional reset; trajectory continuity preserved; fresh cells reset before launch."
- **Deferred to Chunk 2**: Option K Trajectory Event Log schema (unified perturbation event tracking for both reset interrupts AND auth-loss/auto-clear events per user cross-talk insight — generalizes Tier 1 stack to auth-loss problem at zero additional cost).

### B-305. `launch.sh:30` help text stale "B0 | B1 | Claude" [P2 — Claude unique] 🛠️ FIXED
- **Attack**: Help text lists `BASELINE: B0 | B1 | Claude`; CLAUDE.md baseline scope updated 2026-05-14 to B2=Gemma3-VL (跨族第三 model). New users following help text run `BASELINE=Claude` → case match fail → exit 65 confusion.
- **Fix**: Updated to `BASELINE: B0 | B1 | B2`.

### B-306. `glm_pre_launch_check.py` retirement + deterministic shell asserts in launch.sh [P1 absorbed (P1-3+P1-10+P1-11) — Claude+codex] 🛠️ FIXED (file DELETED)
- **Attack**: 3 cascading bugs in single file:
  - P1-3 (Claude+codex): `glm_pre_launch_check.py:152-155` returns 0/1 only (never 2); `launch.sh:130` expected BLOCK=exit 2 → BLOCK verdicts from GLM collapsed to WARN+y/N prompt → paper-grade hard rule violations operator-y-bypassable.
  - P1-10 (Claude unique): `:96` non-greedy regex `r"\{.*?\}"` picks shortest brace pair → JSON parse failure on nested GLM responses → fail_default.
  - P1-11 (Claude unique): `:47-50` GLM config missing → `return True` (fail-open) asymmetric with line 87-88 fail-closed → fresh A100 clone silently skips pre-launch check.
- **Fix decision** (per user 2026-05-16): retire entire file + replace with deterministic shell asserts in `launch.sh:113-160`. Rationale: 4/5 hard rules already deterministically enforced upstream (queue_chain.sh:142-171 3-way collision / queue_chain.sh:7-9 RESET default / queue_chain.sh:117-122 script-existence / queue_baseline.sh BENCHMARK branch); only rule unique to glm_pre_launch_check was config-↔-site benchmark match (now `grep "benchmark: ${EXPECTED_BENCH}" ${cfg}`). GLM dependency removed = LLM variance / API outage / non-deterministic gate eliminated from paper-grade launch path. Per codex C-8 defuse "GLM should be advisory only".
- **Files touched**: deleted `scripts/maintenance/glm/glm_pre_launch_check.py` (159 LOC); updated `launch.sh:113-160` (deterministic 3-rule check); removed `Makefile:35,527-536` target + .PHONY entry; struck through `scripts/maintenance/README.md:15` table row.

**B-numbers consumed**: B-298 through B-306 (9 contiguous, no collisions with A1.8 B-280~B-297)。

**Deferred to A1.17 Chunk 2** (~11h investment, separate commit):
- P1-1 (3-AI overlap) cls reset sentinel multi-table expansion (`oc_t_item + comment + user`)
- P1-2 (A+B) `a100_self_host_vwa.sh` deploy_reddit/shopping bad compose paths
- P1-6 (gemini OOB unique) reddit reset `docker run` missing `-e TZ` flag
- P1-7 (gemini OOB unique) `REQ_GB=130` → 250 (217GB actual)
- P1-8 (gemini OOB unique) Magento `indexer:reindex` async no-wait poll
- P1-12 (Claude unique) BUG-5 sibling: cls DB seed `|| true` strip
- **Option K Trajectory Event Log** (user cross-talk insight): `p79/experiment/logger_v2.py` `log_trajectory_event()` API + hooks in `experiment_watchdog.py` (auth-clear) + `_lib_paper_grade_gates.sh` (reset event). Generalizes Tier 1 stack to cover BOTH P1-5-B reset events AND auth-loss/auto-clear events at zero analysis-cost — ~2-3h additional schema work.

**Smoke verification**: bash -n PASS all 5 scripts (`_lib_paper_grade_gates.sh` / `queue_chain.sh` / `reset_vwa_sites.sh` / `phase1a_relaunch_missing.sh` / `launch.sh`). Makefile help target compiles. B-302 sentinel python 3-case smoke (valid 234 / ep=0 / partial 100/234=43%) all exit-codes correct.

**Phase 1a fire green-light post-Chunk 1**: launch path now produces correct sentinel validation (B-302 schema fix is the critical unblocker). Remaining advisor blockers still B-262 (GLM fallback) + B-130 (FE/RE estimand). Chunk 2 fixes are paper-grade quality, NOT launch blockers — Phase 1a can fire on Chunk 1 alone.

**Next available B-number**: B-307+.

## §159 /stress A1.18 VWA submodule `p79-patches` — 3-AI audit + full sweep (2026-05-16)

15 findings across Mode A (Claude F1-F7) + Mode B (codex F8-F15) + Mode C (gemini P0-1, P0-2, P1-1, P1-2, P2-1, P2-2 + F6 OOB). 3-AI overlap on P0-4 networkidle asymmetric timing (highest confidence). 2-AI overlaps on P0-1 viewport paradox (BC*), P0-2 IP propagation (AB*, codex deepened 1→913 hits), P0-3 eval model swap (AB*), P0-5 composite commit (AB*), P0-6 SBOM (BC), P0-7 Meta+A — user clarified intentional design (A), P1-5 float coercion (AB).

### B-254. Viewport paradox: paper §4.X.5 "we do not fix" vs code FIXED 🛠️ DOC-FIXED (B-26 + chronicle §80 stale-doc closure)
- **Source**: Mode C gemini P0-1 (OOB) + Mode B codex F8 (confirm) — Mode A Claude missed (saw file:line match in disclosure but didn't read prose body)
- **Code**: `external/visualwebarena/browser_env/processors.py:218` HEAD `f0c835b`: `ratio = (overlap_w * overlap_h) / (width * height)` (FIXED by commit `3f9ceca` 2026-04-19); paper `section4_limitations_disclosure.md:113-115` said "We do **not** fix this bug"
- **Attack**: paper §1 hero claim invariance defense uses "DOM systematically helps from no-op viewport filter" as confound source disclosure; if reviewer git-show §218 sees the fix, the entire §4.X.5 framing collapses
- **Fix**: §4.X.5 rewritten (title "FIXED 2026-04-19" + prose reflects fix applied + 0.6 threshold math + all DOM/SoM re-run after fix); master_bug_catalog B-26 status flipped 🛠️ NO_FIX → 🛠️ FIXED + commit `3f9ceca` reference. §80 chronicle decision had landed; paper §4 + catalog metadata never updated until A1.18.

### B-255. Hardcoded Quark Tailscale IP propagation across 913 task configs + 8 scripts 🛠️ FIXED (full clean for A100 reproducibility)
- **Source**: Mode A Claude F1 (single-point framing) + Mode B codex F10 (OOB deepening, 794 hits in test_shopping.json alone)
- **Code**: `external/visualwebarena/config_files/vwa/test_*.json` 913 files / 3882 total IP occurrences + `envs.py:145` chromium launch arg + 8 P79 scripts (`scripts/maintenance/{retry_b1_single_task.sh,reset_vwa_sites.sh,glm/myriad_watcher.py,auto_pull_myriad_cell.sh,experiment_watchdog.py}` / `scripts/queues/queue_pilot_t0.sh` / `scripts/myriad/{smoke_compute.qsub,smoke_login.sh}` / `scripts/setup/a100_self_host_vwa.sh` / `scripts/provenance/snapshot_vwa.sh` / `p79/utils/auth_refresh.py`)
- **Attack**: Phase 1a 在 A100 跑 → quark Tailscale 私 IP 不可达 → 任何 task config `start_url` field 直接指向不可达地址 → reset 站点必失败,reproduce killer for any host outside Tailscale net
- **Fix**: 913 task configs rewritten with `__SHOPPING__/__CLASSIFIEDS__/__REDDIT__/__WIKIPEDIA__` placeholder substitution (`tasks.py::_placeholder_mapping()` already infra-ready); `envs.py:143-156` env-driven via `VWA_CHROMIUM_LAUNCH_ARGS`; 8 scripts converted to env-required / localhost defaults / `${VAR:?...}` fail-loud / placeholder examples. **Zero tracked-file IP hits remaining** (excluding `_deprecated/` + gitignored `scripts/vwa_env_remote.sh`).

### B-256. Eval model `gpt-4-1106-preview` → `gpt-4o-mini` silently undisclosed 🛠️ DOC-DISCLOSED (test deferred)
- **Source**: Mode A Claude F2 (OOB) + Mode C gemini P2-1
- **Code**: `external/visualwebarena/evaluation_harness/helper_functions.py:603-607, 654-658` `eval_model = ... or "gpt-4o-mini"` (default); upstream VWA was `gpt-4-1106-preview`. Commit `3f9ceca` body just says "default gpt-4o-mini" with no rationale
- **Attack**: Cross-paper SR (VWA / WebArena-Verified / PAE 都用 gpt-4-turbo) 不可直接比;judge differential 1-3pp 估算
- **Fix (disclosure)**: paper §3.5 加 "LLM-judge model disclosure" paragraph 明确 swap + cross-paper comparability scope; paper §8.2 retract "canonical" → "internal-P79 paper-grade outcome". **Deferred**: user wants test gpt-4-1106-preview availability first (session permission-blocked, user-runnable 1-line test in chronicle §159.6)

### B-257. Asymmetric `networkidle` wait — Image processor only, Text processor not, plus internal ordering bug 🛠️ FIXED (3-AI overlap)
- **Source**: Mode A Claude F4 + Mode C gemini P0-2 + Mode B codex F14 deepening (internal-ordering within ImageObservationProcessor too)
- **Code**: `external/visualwebarena/browser_env/processors.py:1118-1124` ImageObservationProcessor `process()` waited 2s networkidle; TextObservationProcessor didn't wait at all. Plus codex F14: within ImageObservationProcessor `browser_info` captured L1113 BEFORE wait L1121 → image-internal async (metadata vs screenshot from different render states)
- **Attack**: paper §3 "comparable observations of same page state" claim fails. DOM/SoM AXTree pre-idle + Vision screenshot post-idle + SoM image post-idle. P-SoM 1.7-3.3pp lift could be partly mode-timing confound
- **Fix**: Single shared barrier — moved `page.wait_for_load_state("networkidle", timeout=2000)` to `ObservationHandler.get_observation` BEFORE both text + image processors; removed local ImageObservationProcessor wait. All processors observe same post-idle state.

### B-258. Composite-commit `3f9ceca` selective disclosure (6 fix in 1 commit, paper §4 lists 1) 🛠️ DOC-FIXED
- **Source**: Mode A Claude F1 (composite-commit angle) + Mode C gemini P1-2
- **Code**: Commit `3f9ceca` body lists 5 piece (processors / envs / actions float / helper_functions VWA_EVAL_MODEL / openai_utils lazy init); actual diff also includes 6th piece (Meta+A clear-before-type to 5 type sites). Paper §4.X.5 pre-A1.18 only disclose 1 piece (viewport, also stale per B-254). DGX routing + NumPy actions + eval model swap + lazy init + Meta+A all undisclosed
- **Attack**: reviewer git log shows commit lists 5, paper lists 1 → selective bias signal undermines paper §3 reproducibility claim
- **Fix**: paper §4.X.11 NEW full disclosure table — 5 commits × Subject × Behavioural impact × Affected files × Paper §-disclosure pointer. §4.X.12 NEW hardcoded IP propagation specific disclosure (793 task config hits documented).

### B-259. SBOM/OSF lock missing diff-hash + per-commit behavioural fingerprint 🛠️ FIXED
- **Source**: Mode B codex F9 (computed sha256 `5ca914c7...`) + Mode C gemini P1-1
- **Code**: `preregistration.md:545` records SHA at lock time, `osf_lock_manifest.md:38` table no VWA submodule row, `locked_versions.md:16` HEAD only no diff hash. SHA + branch name mutable under force-push; diff sha256 immutable witness
- **Attack**: reviewer asks "how do you guarantee B-91 was the only evaluator change" — no SBOM-level answer
- **Fix**: `locked_versions.md` VisualWebArena table 4 new rows (HEAD + upstream base SHA + diff sha256 + 5-commit list); `osf_lock_manifest.md §2.1` 3 new rows; `preregistration.md §7` 3-layer hard rule (HEAD match + upstream base resolves + diff sha matches), enforced per Phase 1a run

### B-260. Meta+A clear-before-type behavioral extension undisclosed (B-01 wrapper-layer override missing from paper) 🛠️ DOC-DISCLOSED (intentional design per user rationale)
- **Source**: Mode A Claude F3 (OOB) + Mode C gemini F6
- **Code**: `external/visualwebarena/browser_env/actions.py:1346-1521` `Meta+A + Backspace` clear-before-type 加 to 5 type sites (sync element_id / sync element_role+name / sync pw_code / async element_role+name / async pw_code) by commit `3f9ceca`. Commit body 不 list Meta+A. P79 wrapper `vwa_wrapper.py` Cluster 1 fix overrides with `locator.fill()` atomic clear-and-type, bypassing the Meta+A path (paper-grade data unaffected). Submodule fallback retained for raw-VWA reproducers
- **User rationale** (clarification 2026-05-16): "agent 在编辑时看不到输入框的文本,重复输入会接在上一次后面;backspace 不是全选变蓝原因,是 meta-a" → Meta+A clear-before-type is **intentional P79 design** (clear-before-retry semantics) + wrapper `locator.fill()` is canonical implementation (avoids 全选变蓝 side effect when click misses input)
- **Fix (disclosure)**: paper §3.5 "Type-action clear-before-type behaviour" paragraph documents double-layer (wrapper canonical + submodule fallback for raw-VWA reproducibility) + cross-paper VWA comparability scope

### B-261. Softened `assert "correct" in response` → silent return 0 + substring FP risk 🛠️ FIXED (tighten + log)
- **Source**: Mode A Claude F5
- **Code**: `external/visualwebarena/evaluation_harness/helper_functions.py:619-624, 668-672` upstream was `assert "correct" in response, response` (crash-loud); P79 patch softened to `elif: 1.0 else: return 0.0` silent. Substring "the correct answer is X but student wrote Y" passes → FP; "n/a" / "skipped" responses silent 0 → FN; lost evaluator diagnostic signal
- **Fix**: Tighten match (still primarily "correct" in response but log unexpected response cases to `evaluator_unexpected_response_log.csv` via new `_log_unexpected_judge_response()` helper, gitignored runtime artifact for post-hoc audit)

### B-262. Lazy OpenAI client init unlocked + no env-drift check 🛠️ FIXED
- **Source**: Mode B codex F11
- **Code**: `external/visualwebarena/llms/providers/openai_utils.py:15-32` `_require_openai_clients()` mutates module globals (client, aclient) without lock; if `OPENAI_BASE_URL` changes between calls, last-writer-wins
- **Attack**: concurrent thread / proxy endpoint drift mixes old/new clients → batch judge poison
- **Fix**: threading.Lock + sha256(api_key + base_url) fingerprint stored as `_clients_env_fingerprint`; reinit on env change

### B-263. Async OpenAI response-shape mismatch — caller dict-indexes SDK objects 🛠️ FIXED (OOB)
- **Source**: Mode B codex F12 (OOB)
- **Code**: `external/visualwebarena/llms/providers/openai_utils.py:96` (completion async returns SDK object), `:151` (caller `x["choices"][0]["text"]` dict indexing); fallback `:111` returns chat-shaped dict missing `"text"`. Symmetric issue in chat helpers `:213-236, :276`
- **Attack**: async batch judge path crashes on success (SDK obj not subscriptable) or on fallback (missing key); non-deterministic evaluator failure invisible to caller
- **Fix**: Throttlers (`_throttled_openai_completion_acreate` + `_throttled_openai_chat_completion_acreate`) return `str` directly on both success (`resp.choices[0].text` / `.message.content`) and fallback (`""`); callers use plain list construction

### B-264. Async action dispatcher signature lacks `obseration_processor` + UPLOAD factory misset 🛠️ FIXED (OOB)
- **Source**: Mode B codex F13 (OOB)
- **Code**: `external/visualwebarena/browser_env/actions.py:1427-1449` `aexecute_action(action, page, browser_ctx)` signature lacks `obseration_processor` param but body L1448 (CLEAR) + L1572 (UPLOAD) reference it → `NameError` at runtime. Also L1449 `await execute_mouse_click(...)` awaiting sync function. Plus `create_upload_action:697-715` sets `ActionTypes.TYPE` not `UPLOAD` → UPLOAD branch in dispatcher unreachable (uploads silently executed as text-entry)
- **Attack**: any async VWA execution path through CLEAR/UPLOAD crashes; any upload task on async path silently misbehaves
- **Fix**: `aexecute_action` signature added `obseration_processor: ObservationProcessor | None = None` param + branches `raise RuntimeError` when missing for CLEAR/UPLOAD + use `aexecute_mouse_click` / `aexecute_key_press` truly async primitives; `create_upload_action` set `ActionTypes.UPLOAD`

### B-265. Float coercion sibling propagation gap — async hover + async upload not cast 🛠️ FIXED
- **Source**: Mode A Claude F6 + Mode B codex F15
- **Code**: `external/visualwebarena/browser_env/actions.py:949` async `aexecute_mouse_hover` passes raw `left * viewport_size["width"]` (no float()); `:993` async `aexecute_upload` same. Commit `3f9ceca` only added float() cast to 4 sync sites
- **Attack**: NumPy 2.0 env where np.float32 leaks in → Playwright JSON serializer silent fail on async-only path
- **Fix**: 2 sites wrapped `float(left * viewport_size["width"])`

### B-266. `prepare.sh` python resolver Unix-only — Quark = Windows host fail 🛠️ FIXED
- **Source**: Mode A Claude F7
- **Code**: `external/visualwebarena/prepare.sh:7-30` resolve_python tries `.venv/bin/python` → `python3` → `python` only
- **Attack**: Windows reproducer (Quark itself is Windows!) / conda env without Unix python fails
- **Fix**: Added Windows `py -3` fallback (last in resolution chain)

### B-267. Paper §8.2 "canonical" wording overclaims cross-paper comparability 🛠️ DOC-FIXED
- **Source**: Mode C gemini P2-1
- **Code**: `section8_limitations.md:7` "raw `success` from the fixed evaluator is canonical"
- **Attack**: "canonical" implies cross-paper comparability; B-91 fixes empty-pred FP but does NOT bridge gpt-4-turbo→gpt-4o-mini capability drift
- **Fix**: "canonical" → "internal-P79 paper-grade outcome" + explicit cross-paper SR comparison scope statement

### B-268. Hardcoded private IP in launched chromium ID (envs.py launch arg only — single-point, separate from B-255 task config propagation) 🛠️ FIXED
- **Source**: Mode A Claude F1 (single-point framing, deepened by codex to B-255 multi-site)
- **Code**: `external/visualwebarena/browser_env/envs.py:144-145` chromium.launch `args=["--host-resolver-rules=MAP metis.lti.cs.cmu.edu 100.95.81.103"]`
- **Attack**: chromium launch arg leaks private IP into committed code (separately from task config data leak B-255); soft privacy concern + cross-host portability blocker
- **Fix**: env-driven via `VWA_CHROMIUM_LAUNCH_ARGS` (space-separated arg list, default empty); reproducers set their own `--host-resolver-rules` MAP rule

**B-numbers consumed**: B-254 through B-268 (15 contiguous, no collisions). Cumulative session work A1.4a+A1.4b-i+A1.4b-ii+A1.4c+A1.5+A1.13+A1.6+A1.18 = B-140 through B-268 = 129 unique entries.

**Next available B-number**: B-269+.

**Deferred (per user 2026-05-16 decision)**:
- B-256 (eval model) gpt-4-1106-preview availability test — user-runnable 1-line test deferred; disclosure-only path applied handles both deprecated and cost cases

---

---

## /stress A1.17 Chunk 2 — paper-grade quality + Option K Trajectory Event Log (2026-05-16)

Chunk 2 of A1.17 audit cycle (Chunk 1 = launch-blockers, this Chunk = paper-grade quality + Option K user-insight generalization). Worked in `git worktree add ../p79-a1.17-chunk2` per user worktree-per-session protocol decision 2026-05-16 (avoid B-#/§-race + working-tree pathspec friction across parallel Claude audit sessions). Just-in-time check from `master:docs/reference/master_bug_catalog.md` confirmed B-306 latest header → Chunk 2 starts B-307.

### B-307. `reset_vwa_sites.sh:55-56` cls reset sentinel narrow (single-table) [P1 — 3-AI overlap A+B+C OOB] 🛠️ FIXED
- **Attack**: Pre-fix verified only `oc_t_item_comment` (`b_active=1` count==0). OSClass reset endpoint regression could clear comments but leave `oc_t_item` (listings posted by prior episodes), `oc_t_user` (registered users), `oc_t_item_meta` — sentinel passes while ablation surface still contaminated.
- **Cascade**: P79 cls task mix includes search-listing / post-listing / user-profile types; prior episode's listing residual = next episode's search returns stale data → 3-5pp SR drift (gemini est) / 0.2-0.8pp bounded (codex on require_reset subset).
- **Fix**: Multi-table sentinel — query 3 highest-mutation-surface tables (`oc_t_item_comment` + `oc_t_item` filter `fk_i_user_id > 0` = user-posted only + `oc_t_user` filter `NOT LIKE '%admin%'` = non-admin only with conservative `>20` threshold). All 3 must pass; aggregate failure report in single pass.

### B-308. `a100_self_host_vwa.sh:133-156 + 195-200` deploy_reddit/shopping bad compose paths + smoke `|| true` [P1 — 2-AI A+B] 🛠️ FIXED
- **Attack**: Pre-fix `deploy_reddit` and `deploy_shopping` looked up `${VWA_DIR}/reddit/` + `${VWA_DIR}/shopping/` which DON'T EXIST — VWA reddit uses postmill `docker run` from loaded image (no compose dir), shopping uses `shopping_final_0712` `docker run` similarly. Pre-fix: return 1 + smoke `|| true` swallow + script still prints "=== A100 self-host VWA setup DONE ===".
- **Cascade**: A100 first-time bring-up runbook unusable; operator goes manual debug + may fallback to quark Tailscale URLs → cascade to B-298 hostname-locality bypass risk.
- **Fix**: deploy_classifieds / deploy_reddit / deploy_shopping all delegate to `bash scripts/vwa/start_vwa_docker.sh --sites <site> --hostname localhost` (single source of truth, leverages B-311 indexer poll + B-312 cls DB seed retry + Magento base_url DB-side verify). Smoke check: aggregate failures into `smoke_failed` boolean; exit 1 if any site failed (vs pre-fix swallowing). Smoke probe path switched to `/robots.txt` per B-273 sibling-propagation note.

### B-309. `reset_vwa_sites.sh:71` reddit reset missing `-e TZ` flag [P1 — gemini OOB unique] 🛠️ FIXED
- **Attack**: Pre-fix `docker run -d --name vwa-reddit -p 9999:80 postmill-...` did NOT pass `-e TZ`; `start_vwa_docker.sh:217` initial start DID pass `-e TZ="${QUARK_TZ:-Europe/London}"`. Reset-recreated container ran in UTC.
- **Cascade**: Reddit task mix contains relative-time tasks ("within the last hour" type); reset before vs after = system time changes by Europe/London offset (typically ±1h vs UTC) → ablation 严谨性 broken + systematic noise.
- **Fix**: Added `-e TZ="${VWA_REDDIT_TZ:-${QUARK_TZ:-Europe/London}}"` to docker run command (parity with initial start; renamed env var to `VWA_REDDIT_TZ` for clarity, fallback to legacy `QUARK_TZ` for back-compat).

### B-310. `a100_self_host_vwa.sh:113` REQ_GB=130 vs actual ~217GB needed [P1 — gemini OOB unique] 🛠️ FIXED
- **Attack**: Pre-fix `REQ_GB=130` + WARN-and-continue on shortage. Empirical from `setup_vwa.sh` wget comments: shopping 68 + reddit 53 + wikipedia 95 + classifieds 0.025 = ~216GB raw download. Plus ~30GB docker layer decompression overhead → ~246GB needed.
- **Cascade**: 130-217GB disk free → pre-flight WARN-continues → wget mid-download ENOSPC → corrupted tar/zim files → docker load may silently succeed with missing layers → containers fail at runtime with cryptic errors. Current A100 VM has 485GB free so not currently blocking, but blocks future host migrations.
- **Fix**: REQ_GB=250 (217 + 30 + 3 safety) + WARN → FATAL `exit 1` (paper-grade fail-fast over mid-setup ENOSPC corruption).

### B-311. `start_vwa_docker.sh:202-203` Magento `indexer:reindex` async no-wait [P1 — gemini OOB unique] 🛠️ FIXED
- **Attack**: Pre-fix command `magento indexer:reindex >/dev/null 2>&1 || echo WARN` was fire-and-forget. Reindex on 68GB shopping image takes 5-10min; script returned immediately. If agent runner started before reindex completed, search-autocomplete / category-update tasks would return empty results (no model error, just silent empty data) → SR confounded by infra not model behavior.
- **Cascade**: Phase 1b shop critical — any cls site agnostic task type that depends on category indexing or search-autocomplete behavior would silently fail until reindex completed in background.
- **Fix**: Poll `magento indexer:status` until all rows say "Ready" or 10min timeout (60 × 10s iter). Log timing on success ("all Ready after Xs"); WARN if timeout reached. Maintains backward-compat with pre-fix command (still issued, just now followed by status poll).

### B-312. `start_vwa_docker.sh:280` cls DB seed `|| true` swallows SQL failure (BUG-5 sibling) [P1 — Claude unique] 🛠️ FIXED
- **Attack**: Pre-fix `docker exec classifieds_db mysql ... osclass_craigslist.sql >/dev/null 2>&1 || true` had the BUG-5 anti-pattern that shopping Magento patches (line 190-191) already had stripped. Sibling-propagation defect: same `|| true` in cls DB seed → SQL load failure (e.g. DB warming race) silently swallowed → empty cls DB → all cls tasks 0% SR.
- **Fix**: Strip `|| true`; 3-retry with 5s sleep between attempts (handles DB warm-up race); after 3 retries fail → FATAL return 1 with explicit message "cls site will be empty (all cls tasks would 0% SR)" + abort startup loudly rather than silent broken site.

### B-313. Option K Trajectory Event Log `p79/experiment/logger_v2.py` schema + API [Schema extension — user cross-talk insight 2026-05-16] 🛠️ FIXED
- **Spec**: Append-only JSONL at `condition_dir/trajectory_events.jsonl`. Each event = single line: `{event_type, task_index, wallclock_ts, metadata}`. event_type values: `"reset_post_interrupt"` / `"task_auto_cleared"` / `"auth_refresh_no_clear"` / `"runner_restart"` / `"watchdog_intervention"`. task_index = episode/task index at event time; None for cell-level events. metadata = event-specific dict (reset rc, auth_refresh_method, cleared_task_count, etc.).
- **Why**: P1-5-B reset-discontinuity and auth-loss/auto-clear are isomorphic bug classes — both cause JSONL ↔ site state inconsistency, just in opposite directions. Unified event log enables paper §4 GLMM bias absorption: aggregator emits per-episode `is_after_reset` + `had_auth_clear` + `prior_event_count` columns from this trail. Tier 1 stack (1-gemini GLMM / 4-gemini Fisher / 2-gemini §3 reframe) generalizes to BOTH perturbation classes at zero additional analysis cost (user cross-talk insight 2026-05-16).
- **API**: `LoggerV2.log_trajectory_event(event_type, task_index, metadata)` instance method + module-level `log_trajectory_event_external(condition_dir, event_type, task_index, metadata)` helper for out-of-band callers (bash heredoc / watchdog / future runner-side reset hook). External helper gracefully no-ops when condition_dir doesn't exist yet (e.g., reset gate before runner creates dir).
- **Smoke verified**: missing dir → silent no-op; existing dir → JSONL written with correct schema (2 events test).

### B-314. Option K hooks — `experiment_watchdog.py` auth-clear + `_lib_paper_grade_gates.sh` reset event [Schema integration — user cross-talk insight] 🛠️ FIXED (partial — runner-side reset pickup deferred)
- **Hook 1 (watchdog auto-clean)**: `experiment_watchdog.py:1402+` after `_persist_state()` — logs `task_auto_cleared` event with metadata (`reason` = classified error type, `retry_attempt`, `is_noise`, `is_auth_loss` bool flag, `purged_digest_records`). Best-effort import + try/except — failure is non-fatal (paper-§4 enrichment, not blocking). Captures ALL auto-clean paths: code_bug, noise, session, auth — `is_auth_loss` bool flag differentiates auth-loss subset.
- **Hook 2 (reset gate staging)**: `_lib_paper_grade_gates.sh:reset_and_auth_gate` after successful reset+sleep — writes `reset_post_interrupt` event to STAGING file at `${repo_dir}/logs/trajectory_events_staging/RUN_${RUN_ID}.jsonl` (condition_dir doesn't exist yet at gate time). Staging-file approach documented; runner-side pickup + merge into final `condition_dir/trajectory_events.jsonl` deferred to follow-up (see phase1_plan §A1 pending).
- **Deferred to follow-up**: (i) runner-side staging pickup on startup; (ii) paper §3 reframe to "Multi-Epoch Sequential Benchmark Protocol" (2-gemini, 1h); (iii) aggregator covariate emission in `aggregate_sr_fp.py` (4h, post Phase 1a data land); (iv) Fisher homogeneity rebuttal script (3h).

**B-numbers consumed**: B-307 through B-314 (8 contiguous, latest-checked from master HEAD at chunk start).

**Smoke verification**:
- bash -n PASS: `_lib_paper_grade_gates.sh` / `reset_vwa_sites.sh` / `start_vwa_docker.sh` / `a100_self_host_vwa.sh`
- py_compile PASS: `logger_v2.py` / `experiment_watchdog.py`
- API end-to-end: `log_trajectory_event_external` 2-event JSONL write verified (graceful no-op on missing dir + correct schema on present dir)

**Worktree**: `/home/jiaming/workspace/p79-a1.17-chunk2` branched from master `3e3ac8f` (Chunk 1 commit) on branch `a1.17-chunk2`. Merge back to master post-commit + worktree cleanup.

**Cross-AI value summary (combined Chunks 1+2)**: 22 attacks consolidated across 3 AI lineages (Mode A Claude self / Mode B codex / Mode C gemini). 17 fixes landed (9 in Chunk 1 + 8 in Chunk 2). Most-critical 1-AI unique catches: codex P0-4 (B-302 LAUNCH BLOCKER schema mismatch) + gemini × 3 OOB (B-309 TZ + B-310 disk + B-311 indexer) + user cross-talk insight (Option K generalization). **Paper-grade integrity sweep: Phase 1a launch-readiness post-Chunk 1 confirmed; paper-grade quality post-Chunk 2; Tier 1 analysis-layer fixes (paper §3 reframe + GLMM + Fisher) deferred to paper §4 codex round + post-data analysis**.

**Next available B-number**: B-315+.

### B-320. HARDWARE_PROFILES dict 缺 `"a100_pcie_40gb"` key → silent m2 fallback 🛠️ FIXED (OOB, 3-AI overlap)

Config `exp_v2_base.yaml:79` canonical key `"a100_pcie_40gb"` not in `p79/experiment/energy_tracker.py:HARDWARE_PROFILES` dict (only `"a100"`). `.get(key, HARDWARE_PROFILES["m2"])` fallback → laptop m2 profile (5W/22W) reported for paper-grade A100 fire whenever pynvml unavailable / sampling thread cold-start / `kwh_per_step` unset → ~14× energy/CO2 under-quote (load 22W vs 300W). 3-AI overlap (Mode A F1 + Mode C #2 explicit OOB + Mode B implicit). Fix: alias key `"a100_pcie_40gb"` (same value as `"a100"` baseline; PCIe variant TDP unchanged) + fail-loud `ValueError` in `__init__` on unknown profile when energy enabled.

### B-321. `_average_measured_power` window 混 pre-step idle samples → fast-step energy under-quote 🛠️ FIXED (OOB)

`energy_tracker.py:302-313` sampling thread does not know step boundary → `cutoff = now - duration - 1s` returns mean over ALL samples in window. Fast step (200ms B0 latency, 500ms sample interval) → window mostly pre-step idle → step power averaged with idle, biased toward A100 idle 50W not inference 300W. Total energy ≠ Σ(per_step_energy). Fix: add `step_start_monotonic` param to `estimate_step` + strict `[step_start, step_start+duration]` window bound + emit `window_sample_count` + `energy_window_partial` flag. Legacy callers (None) fall back to pre-fix sliding window (zero behavior change). Runner wires `time.monotonic()` at step start.

### B-322. `aggregate_condition_metrics` entry string-truthy attack (A1.8 B-283 sibling propagation) 🛠️ FIXED (OOB)

A1.8 B-283 fixed string-truthy at `load_episode_summary_strict()`, but `aggregate_condition_metrics` was called from 3 sites (`runner/main.py:636`, `rederive_episode_summary.py:280`, `analysis.py:200`) passing raw dicts bypassing strict loader → defense-in-depth violation. JSON literal `"success": "false"` → Python `bool("false") = True` → paper §1 SR inflated. Fix: `_assert_strict_aggregator_types()` entry guard checks `success` / `benchmark_noise` / `score` types; `_collect_episode_summaries` switched to `load_episode_summary_strict(mode="lenient")` at load boundary; `analysis.py` 3-way coercion drift (`pd.to_numeric` / `astype(bool)` / etc) now operates on bool-validated source.

### B-323. `runner/main.py:911` swallows `write_episode_summary` failure → memory/disk split-brain 🛠️ FIXED (OOB)

`try/except + logger.error` ate disk-write failures → in-memory aggregate counted episode, but `episodes/*_summary_v2.json` missing on disk → `analyze_run()` re-scan path produced different denominators than runner live path → paper §1/§3 disk-vs-memory split-brain on NFS / crash / disk-full. Fix: paper-grade mode (`cfg.paper_grade=True`) raises `RuntimeError`; dev mode still swallows + logs for backwards compat.

### B-324. `image_meta_recorded` schema ghost: A1.8 加 field 但 runner 不写 🛠️ FIXED (OOB)

`p79/experiment/types.py:191` + `schema_migrations/v2.py` STEP_RECORD_V2_DEFAULTS include `image_meta_recorded: bool` per A1.8 B-291 separator design, but `grep -c image_meta_recorded p79/experiment/runner = 0` → runner never wrote → A1.8 schema separator structurally inert → paper §3 image-axis telemetry disclosure layer broken. Fix: runner sets `image_meta_recorded = bool(decision_mode in {"som","vision","phantom_som"} AND image_payload_bytes present AND no encode_error)`.

### B-325. `aggregate_phantom_lift.load()` lenient corrupt rows → §1 oracle lift denominator pollution 🛠️ FIXED (OOB)

Pre-fix `o.add(tid)` before load attempt → corrupt JSON task_id counted as observed failure in drop-one oracle denominator → paper §1 hero "Phantom-SoM +3.33pp reddit drop-one oracle lift" silently polluted by corrupt rows. Fix: strict-by-default flipped (default `P79_STRICT=1`). Corrupt → EXCLUDED from BOTH observed + success sets (corrupt = missing-data, not failure). Legacy `P79_STRICT=0` env override preserved for lenient legacy inspection mode.

### B-326. Paper §1 hero `B=10000 task resamples` vs prereg §81/409 `B=1000` + code `PREREG_B=1000` 🛠️ FIXED (3-way alignment)

`docs/checkpoints/paper_drafts/section1_intro.md:7` prose declared `B=10000 task resamples` for hero P=0.998 CI, but `scripts/analysis/aggregate_phase1_prereg_gate.py:68` had `PREREG_B = 1000` (B-176 prereg lock) + preregistration §81+409 declared `1000-resample`. 3-way mismatch → reviewer replication would get P=0.99 (not 0.998) → paper §1 hero fabricated precision by 10× resamples. Mode C (gemini) unique catch (Claude+codex missed cross-prose/code/prereg alignment). Fix: paper §1:7 prose `B=10000` → `B=1000` to align with prereg + code (prereg remains pending → no advisor sync required). Hero P-value will re-compute at B=1000 after Phase 1a re-fire.

### B-327. `success_rate` denominator includes `benchmark_noise=True` → SR conflated with infra stability 🛠️ FIXED

`metrics.py:334` `success_rate = sum(success) / len(episode_summaries)` raw counted api_rate_limit / playwright crash / auth expired episodes as task failures. Reddit (high noise rate) → mid-pp SR moves between Phantom-SoM and full SoM look like real gains but may be infra-stability variance. Fix: emit `clean_success_rate = sum(success ∧ ¬noise) / sum(¬noise)` field; appendix discloses raw_SR; paper §1 hero will use clean_SR per Q8 (A).

### B-328. `estimate_step_flops` formula `2 N d² L × 4` ~3× under-estimate vs Hoffmann standard 🛠️ FIXED (delete dead helper)

Hand-wave "4× multiplier" cover QKVO+FFN = 8 N d²/layer vs Hoffmann standard 24 N d² (SwiGLU 28). 0 production callers (`grep -r estimate_step_flops p79 scripts tests | wc -l = 1` = self-definition only). Paper §3 does not quote FLOPs/step. Mode A F4 + Mode B F8 both flag formula wrong + dead code. Fix: delete dead helper + replace with comment block explaining standard transformer FLOPs decomposition for future implementers.

### B-329. `VwaEvaluator.evaluate` retry with `fresh_page` 丢 program_html DOM state 🛠️ FIXED (OOB)

`environment.py:220` retry uses `page.context.new_page() + goto(target)` → stateless server-side render. For `eval_types: program_html` (VWA classifieds ~30%+ tasks), DOM state (cart count, posted listing, modified field) NOT preserved → retry false-negative when agent actually succeeded but original page nav error → paper §1 SR silently under-quoted. url_match / string_match / ua_match unaffected (target URL / answer-text-based). Fix: program_html tasks skip retry entirely; bail as `evaluator_nav_error_program_html` so aggregator can exclude from denominator (consistent with N/A task-load exclusion).

### B-330. H3 axis2 universe `universe_5` → `universe_6` (six-arm complete-case) 🛠️ FIXED (OOB)

Pre-fix `aggregate_phantom_lift.py:599-621` H3 axis1+axis2 effect indexed against `universe = sorted(common)` = `universe_5` (DOM ∩ SoM ∩ Vision ∩ P-text ∩ P-SoM) → axis2 estimand drift when P-prompt missing on tasks in universe_5. Per user paper §1 framing 2026-05-16: P-text + P-prompt are co-equal axis-decomposition arms (not asymmetric "P-prompt is THE axis"). Fix: switch H3 axis1+axis2 to `universe_6` (six-arm complete-case: DOM ∩ SoM ∩ Vision ∩ P-text ∩ P-SoM ∩ P-prompt). Strictest denominator (smallest N) but estimand matches H3 claim precisely. Fallback to None (not universe_5) when universe_6 unavailable (prevents silent estimand drift on partial cells).

### B-331. `run_summary_v2.json` write 不走 LoggerV2 atomic+fsync chain 🛠️ FIXED (OOB)

`runner/main.py:710` plain `json.dump` could truncate on crash mid-write while `condition_summary` (`logger_v2.py:86-91`) used atomic+fsync → asymmetric durability across writers (paper §1 data inconsistency on post-run crash). Fix: extract shared `write_run_summary_atomic(path, payload)` helper in logger_v2.py (tmp + flush + os.fsync + os.replace + fsync_dir chain matching condition_summary). Runner calls helper instead of plain `json.dump`.

### B-332. `p50_obs_prepare_ms` / `p95_obs_prepare_ms` 结构性 missing 🛠️ FIXED (OOB)

Paper §3.2 quotes `"~30ms median obs-prepare latency"` but `aggregate_condition_metrics` emits only USD aggregate (`avg_total_obs_prepare_cost_usd`), no ms quantile → paper §3.2 number structurally not producible from current pipeline (B-195b deferred). Fix: aggregate p50/p95 across each episode's `obs_prepare_latency_ms_list` field if present. Field assumed populated by runner per-step `latency_ms.obs_prepare` aggregation at episode close.

### B-333. Scroll vocabulary asymmetry: B0 semantic `scroll_direction → ±0.8 fixed` vs B1/B2 free `delta [dx, dy]` 🛠️ DISCLOSED (paper §1 footnote)

B0 (`proxy_api_agent.py:69-696`) emits `scroll_direction ∈ {up, down}` → agent layer pop+converts to fixed magnitude `[0, ±0.8]`. B1/B2 (`_shared_vl_utils.py:117/188/251`) emit `delta: [dx, dy]` predicted per-step by VLM (variable magnitude). Reddit (scroll-heavy site) B0 vs B1/B2 SR delta confounded: capability vs action-vocab flexibility. Mode C catch + Q14 spot-check confirms schema asymmetry. Fix: paper §1 disclosure paragraph (B-333 + paper §3.5.1 cross-baseline asymmetry). Not standardize — proxy-as-deployed semantics preserved; reviewer attack documented.

### B-334. Energy/CO2 platform asymmetry (aarch64 RAPL=0 silent) 🛠️ DISCLOSED (paper §3.5.1)

`p79/experiment/energy_tracker.py:294-299` `total_w = gpu_w + (rapl_w or 0.0)`. RAPL Intel/AMD-only `/sys/class/powercap/intel-rapl/`; aarch64 (NVIDIA Grace ARM) → RAPL unavailable → CPU power=0 silent → cross-platform comparability broken. DGX Spark dev numbers (Grace CPU) not directly comparable to A100 paper-grade fire (Intel host RAPL fires). Fix: paper §3.5.1 footnote disclosure + chronicle entries labelled DGX-dev (excluded from §4 totals). Phase 1a fire host all A100/Intel.

### B-335. `detect_benchmark_noise` 503 misclassified to docker_service_error (no URL context) 🛠️ FIXED

B0 proxy short error `"503 Service Unavailable"` (no AWS gateway URL) was uniformly bucketed to `docker_service_error` → paper §3.4 noise breakdown wrong attribution. Fix: split bare "502"/"503"/"service unavailable" out of docker bucket into new `unclassified_5xx` category. Specific `docker`/`container` URL signatures still classify as docker_service_error.

### B-336. `kwh_per_step` mode duration-blind, paper §3 contract violation 🛠️ FIXED (deprecation raise)

`energy_tracker.py:358-367` `if self.kwh_per_step is not None: kwh = float(...)` returns same kwh regardless of step duration → 5s vs 60s step report same energy → incompatible with paper §3 per-step energy claim (which implies duration-proportional). Fix: `kwh_per_step is not None` paths raise in `__init__` (deprecation hard-block). Config key remains in DEFAULT_CONFIG + yaml schema at value None for backwards compat.

### B-337. `NullEvaluator` score=0 indistinguishable from real failure 🛠️ DEFER (low blast, Q18B)

`environment.py:63-65/176-177` evaluator_unavailable returns `score=0.0, error="evaluator_unavailable"` → runner derives success=False → counted as task failure in SR aggregate → paper §1 SR silently under-quoted if evaluator infra fails. Per Q18B defer until evaluator infra empirically fails in prod (current VWA install OK). Future: aggregator filter `evaluator_error` non-empty episodes from denominator (consistent with N/A task-load exclusion B-91).

### B-338. `cost_usd` nested key validation gap 🛠️ FIXED

`validate_step_record_v2` checked `cost_usd is dict` only; runner writer drift (e.g. rename `"model"` → `"llm"`) would silently zero out `compute_component_breakdown.get("model", 0)` paper §3 model cost number. Fix: validator requires `cost_usd` to contain {input, output, model, router_overhead, total} nested keys. Two existing test fixtures updated to include the 5 keys.

### B-339. `_estimate_power_watts` profile_fallback uniform 0.6 utilization 🛠️ DISCLOSED (paper §3 footnote)

`energy_tracker.py:388` `power = idle + (load - idle) * 0.6` assumes fixed 0.6 util on profile_fallback path, ignoring mode-specific load (vision ViT heavier than DOM). Per Q20B disclose-only (profile_fallback rare in paper-grade fire because pynvml available); paper §3 footnote documents assumption. Mode-specific util multiplier deferred.

### B-340. B0 GLM fallback fail-loud in paper_grade mode 🛠️ FIXED (defense-in-depth)

`use_glm_fallback: false` is yaml default + DeprecationWarning fires when enabled, but warning easy to miss in noisy log. Fix: `ProxyApiAgent.__init__` `RuntimeError` raise when `paper_grade=True AND use_glm_fallback=True`. Required propagating `paper_grade` flag through 3 layers: runner → backend factory → `ApiProxyBackend` → `ProxyApiAgent`. Defense-in-depth against config drift / accidental yaml override during paper-grade fire. Mode C catch was stale (didn't see config default) but defense-in-depth still valuable.

### B-341. RAPLReader `open` no `errors=` (A1.8 B-288 sibling propagation) 🛠️ FIXED

`energy_tracker.py:181` bare `open(self._energy_file, "r")` could raise `UnicodeDecodeError` on kernel mid-write race for `/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj` → outer bare `except Exception` swallowed → silent None → fallback to broken profile path. Fix: `open(..., errors="replace")` mirror A1.8 B-288. `int()` on result still fails fast on non-digit; UnicodeDecodeError path closed.

### B-342. `test_phase1_prereg_gate.py` fixture missing `schema_version` → 5 tests FAILED 🛠️ FIXED (A1.12 P0-1 ABC* 3-AI overlap)

`tests/test_phase1_prereg_gate.py:38 _make_episodes_dir` writes only `{"task_id": tid, "success": ...}` to fixture summaries. B-283 `load_episode_summary_strict[lenient]` requires `schema_version` (str) per `io_utils.py:61`. Without it: all 100 mock summaries treated as corrupt-skip → `theta_pp` collapses to 0.0 → `assert theta_pp == 5.0` fail. 5/17 tests red → `make pre-launch-check` step 7 `pytest -x -q` exits ≠ 0 → Phase 1a launch hard-blocked. Fix: add `"schema_version": "2.0"` to fixture dict. **3-AI overlap**: Claude verified via pytest, codex P0-1, gemini P0-1.

### B-343. `test_phase1_prereg_gate.py` 6-cell topology drift (B0+B1 × shop vs Phase 1a B0+B1+B2 × cls+red) 🛠️ FIXED (A1.12 P0-2 A unique)

`test_build_gate_six_cells_passes:212-214` + 3 sibling tests (`_at_threshold_fails`, `test_write_csv_per_cell_and_pooled_rows`, `test_write_md_renders`) construct 6-cell fixture via `for b in ["B0", "B1"] for s in ["classifieds", "reddit", "shopping"]`. Phase 1a canonical scope = `{B0, B1, B2} × {cls, red}` (CLAUDE.md + phase1_plan.md). Wrong topology: (a) shopping is Phase 1b deferred, (b) B2 Gemma3-VL missing despite 2026-05-14 advisor decision. Fix: 4 sites changed to `for b in ["B0", "B1", "B2"] for s in ["classifieds", "reddit"]`.

### B-344. `p79.experiment.io_utils.load_episode_summary_strict` + `read_jsonl_dedup` ZERO direct tests 🛠️ FIXED (A1.12 P0-3 A unique)

The very B-283 strict-load module that caused B-342 launch-blocker had no direct test. `read_jsonl_dedup` (B-180 identity tuple / B-196 integrity log / B-287 step_idx monotonic / B-288 UnicodeDecodeError / B-293 Optional[bool] semantic) was only indirectly exercised via `test_stress_a1_4b_ii_g4_fixes:B-196 end-to-end`. Any future refactor of strict/lenient boundary, identity-tuple keys, or integrity-log shape could silently degrade aggregator behaviour. Fix: new `tests/test_io_utils_strict_load.py` (14 tests) covering valid/lenient/strict + JSON corrupt + identity mismatch detection + monotonic step_idx + empty file handling.

### B-345. `tests/` ZERO shell-script smoke tests (B-303/B-304/B-224 no net) 🛠️ FIXED (A1.12 P0-4 C* OOB)

Paper-grade shell layer (`_lib_paper_grade_gates.sh`, `queue_phase1_paper_grade.sh`, `queue_chain.sh`, `queue_baseline.sh`, `queue_phantom_*.sh`, `scripts/preflight_v2.sh`) had no regression net despite A1.13/A1.14 batch fixing B-303 chain leakage / B-304 resume discontinuity / B-224 auth gate hard-fail propagation in shell. Pass-1 1-2 week run + Phase 1b shopping reruns at risk for silent shell regression. Fix: new `tests/test_paper_grade_gates_shell.py` (14 tests): `bash -n` syntax across 9 scripts + lib `declare -F` function-export contract + `mint_run_id` FORCE_NEW happy path + back-to-back nanosec collision defense (A1.13 P1-2) + 3-baseline collision check presence in queue_chain.

### B-346. VWA submodule B-91 evaluator empty-prediction guard ZERO test 🛠️ FIXED (A1.12 P0-5 AC)

B-91 fix in `external/visualwebarena/evaluation_harness/helper_functions.py:589 + :677` (both `llm_fuzzy_match` + `llm_ua_match`) returns 0.0 on `pred=""` or whitespace-only. Submodule SHA pin (eb5cbd8) catches file content drift but not behavior; preflight SHA check is OSF lock layer, no unit test. If submodule got `git reset` / merge from upstream main, guard could vanish + N/A task SR inflated by FP. Fix: new `tests/test_vwa_evaluator_b91_guard.py` (6 tests): direct runtime exercise of guard with `pred=""` / `pred="   "` for both fuzzy + ua matchers + cross-check on submodule SHA == A1.18 lock + B-91 guard source-grep at ≥2 callsites.

### B-347. `test_fe_pool_handles_zero_se_via_floor` locks stale 1e-9 floor semantics (impl uses 1pp) 🛠️ FIXED (A1.12 P1-1 B)

`tests/test_phase1_prereg_gate.py:188` docstring says "1e-9 floor" + expects `θ_FE ≈ 2.0`. Current `scripts/analysis/aggregate_phase1_prereg_gate.py:187 _fe_pool` uses `np.where(ses <= 0, 1.0, ses)` (1.0pp floor) — the prereg-disclosure decision preventing degenerate cells from hijacking pool. With 1pp floor: ses=[1.0, 0.5] → weights=[1, 4] → θ_FE = (2 + 12)/5 = 2.8 ≠ 2.0. Test↔implementation drift baked. Fix: update expected to 2.8 + add `n_zero_se_floored_cells == 1` assertion + docstring rewrite citing prereg disclosure.

### B-348. 13 `test_stress_a1_*_fixes.py` source-regex哨兵替代 behavior testing 🛠️ PARTIAL FIX (A1.12 P1-2 AB renatrofit 3 files)

14 stress regression files (~3000 LOC) ~50%+ tests use `re.search` / `read_text` / `in src` source-grep instead of runtime behavior assertion. Failure mode: (a) comment containing `DeprecationWarning` string passes without `warnings.warn` ever firing, (b) `image_meta` built but never attached in runtime path, (c) benign refactor renaming variable breaks regex but behavior intact. Pattern empirically caught by codex P1-4. **Partial fix**: new `tests/test_stress_behavioral_retrofit.py` (5 tests) adds runtime PAIR tests for B-145 GLM DeprecationWarning + B-340 paper_grade hard-block + B-146 Gemma sys.modules decoupling + B-92 Qwen prompt @staticmethod runtime callable + B-144 backend cache (seed) key-distinguishing. Source-grep tests retained for refactor-time string drift; behavior tests for runtime semantic regressions. Full audit of remaining 14 files deferred.

### B-349. `p79.backends.{local_qwen, local_gemma, api_proxy}` ZERO direct tests 🛠️ FIXED (A1.12 P1-3 A unique)

3 production backend classes (B0 proxy / B1 LocalQwen / B2 LocalGemma) had no direct invariant test. `test_agents_prompt_parity` covered prompt-string equality; `test_factory_dispatch:test_mock_backends_agree_on_scroll_delta` covered only LocalQwen (B1). LocalGemma (B2, added 2026-05-14) and ApiProxy (B0) zero coverage at backend layer. Mock-mode contract drift could land B2 launch OOM / B0 401-handling regression with no test signal. Fix: new `tests/test_backends_mock_dispatch_parity.py` (11 tests): 3-backend mock_mode step()  parity (action_type/delta/coordinate_type identical) + factory dispatch + `_agent is None` mock_mode confirmation + missing-key contract.

### B-350. pytest config hygiene gaps (no strict-markers / no filterwarnings / no conftest / Makefile `-x` fail-fast hides failures) 🛠️ FIXED (A1.12 P1-4 A unique)

`pyproject.toml [tool.pytest.ini_options]` had only `testpaths` + `pythonpath`. Missing `strict-markers` → typo'd `@pytest.mark.local_dat` silently always-runs. No `-rs` → `pytest.importorskip("pandas")` silent skip never surfaces. `Makefile test:` used `-x -q` fail-fast → user sees only first fail (hid 4/5 of A1.12 P0-1 batch). No `conftest.py` or `tests/__init__.py` → no shared cleanup of `_JSONL_INTEGRITY_LOG` / `_GLOBAL_REGISTRY` module globals. Fix: `addopts = "--strict-markers -ra -rs"` + `markers = ["local_data: ...", "external: ..."]` + Makefile `-x` removed + `--tb=short`.

### B-351. `tests/analysis/test_run_registry.py` depends on live workspace `results/` + manifest (fresh clone fails) 🛠️ FIXED (A1.12 P1-5 B)

`test_load_manifest_succeeds:14` asserts `len(manifest["cells"]) >= 10` hardcoded; `test_episodes_dir_exists:41` asserts `complete[0].episodes_dir.exists()`. Fresh OSF clone / new contributor / CI without local `results/` → fail even if registry code is correct. Stale local results can let broken path resolution silently pass. Fix: split into (a) 3 pure-logic tests (canonical_mode dict lookup + missing baseline returns None) — always run; (b) 4 `@pytest.mark.local_data + skipif RUN_LOCAL_DATA_TESTS!=1` probes for legacy local-host assertions.

### B-352. `p79.policies.learned_router` Pass-2 entry point ZERO test 🚧 DEFERRED (A1.12 P1-6 B → T1-4=B Pass-1 land 后再加)

`runner/main.py:1017` dispatches `condition.observation_mode == "learned"` to `p79.policies.learned_router`; `load_lr_pipeline:156` + `extract_task_features:172` + `predict_mode:209` untested. Pass-1 (36 baseline conditions) launches without router → not blocked. Pass-2 (6 router conditions) fires 3-5 days after Pass-1, then feature column order / missing pickle / M3 retry could regress with no pytest signal. T1-4=B decision: defer test to Pass-1 land + 1 week before Pass-2 fire (deadline-driven, avoid stale LR pipeline schema).

### B-353. `test_external_module_integration.py` 名实不符 (no actual external integration) 🛠️ FIXED (A1.12 P1-7 B)

File name implies VWA / browser_env / evaluator integration but all tests use in-process `P79Observation` + PIL images + pure helpers — 0 hits on `create_environment` / `VWAWrapper` live / `evaluator_router` / auth gate. Misleading scope hides real integration test gap. Fix: (a) clarifying docstring header on existing file (filename preserved for git blame), (b) new `tests/test_external_vwa_smoke.py` (4 tests) with `pytest.mark.external + skipif RUN_EXTERNAL_TESTS!=1`: VWA browser_env import + evaluator_router callable + P79Observation/VWAWrapper class import + runner can import VWA pathway.

### B-354. Optional deps policy contradiction: `[analysis]` not in `[dev]` + top-level pandas import 🛠️ FIXED (A1.12 P1-8 B)

`pyproject.toml [analysis]` = `pandas/matplotlib/scipy` but `[dev]` = only `pytest`. `test_stress_a1_4b_i_g1_fixes.py:21` + `test_stress_a1_4b_i_g2_fixes.py:13` have top-level `import pandas as pd` → fresh CI install `pip install -e ".[dev]"` collection-time ImportError. Same CI host with `[analysis]` extras → 6 files `pytest.importorskip("pandas/matplotlib/scipy")` silent skip critical analysis end-to-end. Fix: new `[test]` aggregate extras (pytest + pandas + matplotlib + scipy) + 2 top-level imports converted to `pd = pytest.importorskip("pandas")` module-level + `addopts -rs` surfaces silent skip.

### B-355. `p79.experiment.config.normalize_config` / DEFAULT_CONFIG merge ZERO direct test 📋 NOTED (A1.12 P2-1 A defer)

Only spot-checked via `test_fp_architecture_invariants:test_exclude_na_tasks_default_true`. Full DEFAULT_CONFIG merge / YAML override / nested-key precedence untested. Not blocking; flagged for backlog.

### B-356. `p79.utils.auth_refresh` (watchdog auth-clear) ZERO test 📋 NOTED (A1.12 P2-2 A defer)

Paper-grade clean-run 6-layer defense core. Pass-1 1-2 week run will trigger auth_expired_or_session_invalid; auth_refresh regression would cause silent episode failure with wrong benchmark_noise category. Not blocking; flagged for backlog.

### B-357. `test_step_record_validation` only covers `success/score/steps` negative type — misses cost/tokens/latency 📋 NOTED (A1.12 P2-3 B defer)

`validate_episode_summary_v2` could accept `total_cost_usd="0.10"` (str not float) → downstream aggregation crash or coerce weirdly. Paper cost-efficiency + wasted-cost tables = downstream corruption surface. Not blocking; flagged for backlog.

### B-358. `test_smoke_page_unchanged_rate_excludes_finish` vacuous on zero-step runner output 📋 NOTED (A1.12 P2-4 B defer)

`if n_total == 0: return` early-return bypasses invariant. Low probability (other smoke checks file existence), but not enforcing minimum schema shape. Not blocking; flagged for backlog.

---

## /stress A1.10 fix-batch — `p79/experiment/{router, modules, state_change, checklist_module, tasks, config}` + consumers + paper §1/§3/§4 prose (2026-05-16)

3-AI cycle (Mode A + Mode B + Mode C) + 17 fixes (P0×8 + P1×9) + 5 prose disclosure additions. P0-5 schema bump + P1-6 state_digest after-fields deferred per user Q3=B. P2 items (5) defer paper-2. Phase 0 self-audit + Phase 1+2+3 cross-AI verification all PASS.

**Source distribution**:
- Claude (Mode A): 12 findings, 4 OOB — router internals + state_change substrate
- codex (Mode B): 7 findings, 4 OOB — consumer path (runner main.py call site + analyzer aggregator + aggregate_phantom_lift)
- gemini (Mode C): 9 findings, 4 OOB — paper §1/§3 prose vs code reality hallucinations

**Cross-AI agreement**: 3-AI overlap = 1 (P0-1 router thresholds dead + aggregator no audit + §4.X.5 staleness undisclosed); 2-AI overlap = 2 (P0-2 B-09 split incomplete; P1-2 regex sibling propagation); 1-AI unique = 17.

### B-359. Router numeric thresholds empirically dead under cleaned-AXTree regime (P0-1-ABC*) 🛠️ FIXED (3-AI overlap OOB)
- **Source**: A1.10 Mode A F1 + Mode B F6 + Mode C F7 cross-validate.
- **Code**: `p79/experiment/router.py:32,43,44` defines `dom_size_threshold=12000` / `dom_complexity_trigger=500` / `text_length_trigger=12000`. 5001-step empirical (B1 cls 3-mode 20260413): `state_digest.text_length` p50=3113 p95=4675 max=46591 → **pct > 12000 = 0.14 %**; `state_digest.dom_complexity` p95=73 max=81 → **pct > 500 = 0.00 %**. Streak counters fire 25.39 % — actual routing signal source.
- **Attack**: Paper §3.5/§6 prose implies "size-aware escalation" but router fires 100 % on streak signals, 0.x % on numeric thresholds. Reviewer grep `'"dom_size_exceeds_threshold"' phase1_*/episodes/*.jsonl | wc -l` catches.
- **Fix**: (a) `docs/checkpoints/paper_drafts/section3_definition.md` §3.5 + `section4_limitations_disclosure.md` §4.X.5 added paragraph "rule-based router numeric thresholds — empirically dead under cleaned-AXTree regime" disclosing empirical fire rates + attributing routing to streak counters + cross-referencing §4.X.5 viewport-fix as upstream cause; (b) `scripts/analysis/aggregate_phantom_lift.py` added `audit_router_fire_rate()` + `--audit-fire-rate <run_root>` CLI gate that fails when max numeric trigger > 0.5 %. Empirical audit on B1 cls 3237 steps: dom_size 0.22 % / dom_complexity 0.00 % / text_length 0.19 % / streak 25.39 % → disclosure_consistent=True. (A) recalibrate path defer paper-2 per Q4=A.

### B-360. B-09 page_changed/agent_visible_changed split not propagated to router input + analyzer (P0-2-AB*) 🛠️ FIXED (2-AI overlap OOB)
- **Source**: A1.10 Mode A F2 (router input) + Mode B F3 (analyzer 5 callsites) dual-catch.
- **Code**: `runner/main.py:1802` `prev_page_changed = page_changed` (raw any-reason) passed to `router.decide()`; `analyze_reason_diagnostics.py:511/553/798/1410/1943` consumed raw `page_changed` for wasted/stuck/page_change_rate/ax_page_change_rate/per-task counts.
- **Attack**: Router escalation suppressed by form_value_changed / dom_complexity_changed reasons agent cannot perceive → per-baseline asymmetric routing (dom mode sees more form_value reasons than vision); analyzer diagnostics rollups polluted by RUNNER_INTERNAL_REASONS → paper §3.5/§6 cross-mode comparison not apples-to-apples.
- **Fix**: (a) `runner/main.py:1844-1845` switched `prev_page_changed = is_agent_visible_change(page_change_reasons)` (module-imported helper already at top of file); (b) `analyze_reason_diagnostics.py` added module-level `_progress_changed(step)` helper preferring `agent_visible_changed` field with fallback to `page_changed` for legacy archive records; all 5 sites switched to helper. Cls B0/B1 router_on archive marked paper-2 backlog (Q5=A defer re-fire); paper §4.X disclosure added.

### B-361. DEFAULT_CONFIG observation_mode 3-mode vs paper-1 6-mode universe (P0-3-A) 🛠️ FIXED
- **Source**: A1.10 Mode A F7. Mode B verified current launch yamls override → defense-in-depth not active leak.
- **Code**: `p79/experiment/config.py:23` DEFAULT_CONFIG `observation_mode: ["dom","som","vision"]` — paper §1 hero claims 6-mode universe; yaml override-forgotten silently generated 3-mode subset run.
- **Fix**: DEFAULT_CONFIG fallback raised to paper-1 canonical 6-mode list `{dom, som, vision, phantom_som, phantom_text, phantom_prompt}`. Per-condition yamls retain legitimate 1-mode override (e.g. `B0_dom_classifieds.yaml`); 6-mode discipline enforced at launch-orchestrator layer.

### B-362. Sibling regex unanchored propagation across 7 callsites (P1-2-AB*) 🛠️ FIXED (2-AI overlap OOB)
- **Source**: A1.10 Mode A F4 (state_change.py:21) + Mode B F7 (sibling propagation som.py:46/89/98 + action_utils.py:303 + vwa_wrapper.py:959/1008).
- **Code**: Pre-fix `re.search(r"\[(\d+)\]", line)` unanchored across 7 callsites — A1.4 SOM regex fix did not propagate to siblings. Lines like `[10] StaticText 'see [4] section'` matched twice (once each); StaticText labels containing only bracketed digits could false-positive.
- **Fix**: `p79/experiment/som.py` added canonical `MARK_ID_DETECT_RE = re.compile(r"^\s*\[(\d+)\]\s+\w")` + `extract_mark_id(line)` + `is_mark_line(line)` helpers + `_MARK_ID_PREFIX_STRIP_RE` for label cleanup. All 7 callsites switched to helpers (state_change._extract_interactive_count / som._extract_text_marks / som._options_map options scan / action_utils.first_element_id_by_keyword / vwa_wrapper._inject_select_options dropdown injectors). 4 regression tests in `test_stress_a1_10_fixes.py`.

### B-363. `_TEXT_TRUNCATION_LIMIT=5000` content_change similarity computed on truncated prefix (P1-1-A) 🛠️ FIXED
- **Source**: A1.10 Mode A F3.
- **Code**: `state_change.py:9` capped visible_text to 5000 chars; empirical p95=4675 max=46591, ~5 % pages exceeded → SequenceMatcher on first 5000 chars only, long cls listings pages with shared nav prefix → content_changed=False even when listing content updates.
- **Fix**: `_TEXT_TRUNCATION_LIMIT` raised to 20000 (covers p99 ≈ 8000 × 2× safety) + content-hash equality fallback when either page ≥ truncation limit (md5 byte-exact comparison, O(n) vs O(n²) SequenceMatcher). 2 regression tests verifying equal-pages → similarity 1.0, different-pages → 0.0 + content_changed.

### B-364. `dom_complexity` field misnamed (line count, not element count) (P1-3-A) 📝 PROSE-ONLY
- **Source**: A1.10 Mode A F5.
- **Code**: `state_change.py:77` `dom_complexity = text.count("\n") + 1` — AXTree text line count, not DOM element count. Paper §3 / analysis scripts referencing `dom_complexity > 500` reviewer would interpret as DOM elements.
- **Fix**: `section3_definition.md` §3.5 added "dom_complexity field name disclosure" paragraph clarifying field is line count, schema v2 preserves name for archive backward compat, schema v3 paper-2 prep will rename to `axtree_line_count`. Cosmetic only — empirical impact ≈ 0 since trigger dead anyway (see B-359).

### B-365. `_extract_modal_state` over-matches "dialog" substring anywhere (P1-5-A) 🛠️ FIXED
- **Source**: A1.10 Mode A F8.
- **Code**: `state_change.py:36` pre-fix `any(k in low for k in ("dialog","modal","popup","overlay","aria-modal"))` matched any of these substrings anywhere in AXTree dump including reddit subforum descriptions like `'Open dialog about features'` → modal_present flipped noisily → modal_state_changed (AGENT_VISIBLE_REASON) became polluted.
- **Fix**: Strict regex `\b(?:role|aria-modal)\s*[=:]\s*[\"']?(?:dialog|alertdialog|modal)\b` requires role/aria-modal attribute context. 3 regression tests (substring rejection / role match / aria-modal match).

### B-366. `_form_fields_changed` discriminator only set for radio/checkbox (P1-4-A*) 🛠️ FIXED (OOB)
- **Source**: A1.10 Mode A F6.
- **Code**: `state_change.py:89-102` pre-fix discriminator = `value` only when `type in ("radio","checkbox")`; text/textarea with empty `name=""` (cls search filter pattern) at idx=0 different wrappers collapsed to identical `(input,text,"","",0)` key → form_value changes between such fields silently missed.
- **Fix**: Discriminator now `str(f.get("value",""))` for ALL field types. 2 regression tests (empty-name text input value change detected / identical fields no-change).

### B-367. `clean_success_rate` extension to A1.9 B-327 verified via test_b327 (already landed) ✓ NO ACTION
- **Source**: A1.10 cross-verification only. Already shipped in A1.9.

### B-368. Learned-router → rule router double dispatch (P0-4-B*) 🛠️ FIXED (codex unique OOB)
- **Source**: A1.10 Mode B F1.
- **Code**: `runner/main.py:1082-1087` learned router `_dc_replace`d `condition.observation_mode = predicted_mode` at episode start, but `1232-1239` `self.router.decide(router_enabled=condition.router_on=True, ...)` rule router still ran and could re-pick mode via streak/threshold escalation → paper §6 learned-router cells reported hybrid (LR + rule) policy, not pure LR oracle validation. H10 Pareto attribution corrupted.
- **Fix**: `runner/main.py:1232+` added `_is_learned_cell` guard reading `condition.metadata.router_variant == "v7_learned"`; if learned cell: `decision_mode = condition.observation_mode` (LR prediction landed at episode start), `triggers = ["v7_learned_route"]`, `overhead = {..., "rule_router_skipped": 1.0}` flag for analyzer transparency. Smoke test verifies conditions.py emits `router_variant=v7_learned` marker.

### B-369. Retry mixed with primary action identity in step record (P0-5-B*) 📝 DEFERRED PROSE-ONLY
- **Source**: A1.10 Mode B F2.
- **Code**: `runner/main.py:1466-1516,1631-1634,1700-1701` retry: `action_success=retry_success` + `page_changed=bool(retry_reasons)` + `page_change_reasons=retry_reasons+[retry_tag]` but `action_type` / `action` payload still **original failed action**. Step JSONL row mixes primary action identity + retry outcome semantics; downstream `action_type`-sliced aggregations attribute "successful click" when actually the retry scroll succeeded.
- **Fix**: Per user Q3=B, schema v2 → v2.2 bump deferred to paper-2 (M3 ablation prerequisite). `section3_definition.md` §3.5 retry-attribution paragraph extended with explicit disclosure of action-identity vs outcome mixing semantics for paper-2 M3 aggregators (filter on `retry_action_applied==True`/`False` separately, NOT inspect `action_type` of retry rows as primary action). Paper-1 baseline M3 off default → latent (per existing §3.5 disclosure).

### B-370. state_digest before-only snapshot (P1-6-B*) 📝 DEFERRED (gated on B-369 schema bump)
- **Source**: A1.10 Mode B F4.
- **Code**: `runner/main.py:1680-1689` `state_digest` emits only `url_before`/`url_after`/`title_before`/`title_after`/`dom_complexity`/`text_length`/`scroll_y_before`/`scroll_y_after`. Pre-fix can't re-compute `dom_complexity_changed`/`text_length_changed` delta from JSONL → provenance not closed for re-analysis of router threshold fire rate.
- **Fix**: Deferred per Q16=A "merge with P0-5 schema bump"; since Q3=B defers P0-5, P1-6 also defers. Schema v2.2 paper-2 prep will add `*_before` + `*_after` for all signal fields.

### B-371. Analyzer drops router metrics from episode_reason_rows.csv (P1-7-B) 🛠️ FIXED
- **Source**: A1.10 Mode B F5.
- **Code**: `analyze_reason_diagnostics.py:2034-2130` episode_row dict omitted runner's emitted `escalation_count` + `trigger_distribution`; paper §3.5/§6 router rollups required join-back to raw summary.
- **Fix**: Pre-loop computes per-task `_task_escalation_count` (steps where `router.decision != observation_mode`), `_trigger_counter` (Counter of router.trigger_reason values), `_rule_router_skipped_count` (from B-368 overhead flag). Three new episode_row fields: `escalation_count`, `trigger_distribution_json`, `rule_router_skipped_steps`.

### B-372. Paper §1 rule-based router task-attribute hallucination (P0-6-C*) 📝 PROSE-FIXED (gemini unique OOB)
- **Source**: A1.10 Mode C F1.
- **Code**: `section1_intro.md:13` prose "rule-based router that selects mode by task attribute (instruction tokens, presence of reference image, finish-string-match flag)" — but `router.py:46 RuleBasedRouter.decide` reads only dynamic step feedback (unchanged_streak / no_progress_streak / dom_size / action_failed); the "task attribute" features described are the **learned router** features. Paper §1 conflates rule-based + learned router.
- **Fix**: §1 prose rewritten to "rule-based escalation router that switches mode reactively on dynamic step feedback (unchanged-page streaks, action-failure streaks, raw DOM-size signal), implementing the v3/v4/v5 single-step `modes[idx+1]` escalation … the learned classifier predicts best-mode-per-task from TF-IDF task-instruction features plus binary task features (presence of reference image, intent keyword regexes, AXTree element count) under 5-fold site-stratified CV. The two routers consume disjoint signal sources — the rule-based one observes the running episode, the learned one observes the task statically".

### B-373. Paper §1 lists rule-based cascade as paper-1 contrib but DEFERRED paper-2 (P0-7-C*) 📝 PROSE-FIXED (gemini unique OOB)
- **Source**: A1.10 Mode C F2.
- **Code**: `section1_intro.md:13` listed "(a) rule-based router + (b) learned classifier" — but `conditions.py:126` "cascade DEFERRED to paper-2 per Q3 decision 2026-05-16"; paper-1 Phase 1a actual yaml router_kind = "learned" only. v3/v4/v5 single-step escalation runs but v6 pareto-cascade not.
- **Fix**: §1 prose now distinguishes "v3/v4/v5 single-step `modes[idx+1]` escalation" (paper-1) from "v6 pareto-cascade variant (`router.py::latch_after_fallback`) deferred to a follow-up paper". Same prose edit as B-372 (single paragraph touched both findings).

### B-374. Paper §3.4.1 "two ID systems disagree" hallucination (P0-8-C*) 📝 PROSE-FIXED (gemini unique OOB)
- **Source**: A1.10 Mode C F3.
- **Code**: `section3_definition.md:80` claimed "AXTree text uses an independent hierarchical accessibility-tree ID space; the two ID systems do not in general agree on a given element's identifier" — but §3.2 line 35 itself states SOM_MARKS is regex-filtered from AXTree (shares IDs). The "ID space disagreement" mechanism was a fabricated explanation for P-prompt cross-domain behavior.
- **Fix**: §3.4.1 prose rewritten: "the SoM system prompt expects a **flat compressed listing** of `[N]` references (the `[SOM_MARKS]` block produced by `_extract_text_marks` regex-filtering the AXTree), while the AXTree text **embeds the same `[N]` IDs inside a nested hierarchical structure** with role labels, parent-child indentation, and url/tab metadata. The two surface formats expose the same underlying AXTree IDs in incompatible structural contexts; the mismatch is one of representation format, not of identifier values".

### B-375. Paper §3.4.2 "token-monotonic cascade" prose vs v6 jump-and-latch reality (P1-8-C*) 📝 PROSE-FIXED (gemini unique OOB)
- **Source**: A1.10 Mode C F4.
- **Code**: `section3_definition.md:101` described "Section 6 promotes this scaffold to a token-monotonic cascade — DOM → P-text → Phantom-SoM → full SoM" — but `router.py:130-141` v6 cascade directly `decision = self.safe_fallback_target` (phantom_som default) + `fallback_latched = True`, single-step jump + lock, NOT 4-step monotonic.
- **Fix**: §3.4.2 prose rewritten to two-router operating-point story: v3/v4/v5 single-step escalation (paper-1) + v6 jump-and-latch (paper-2 deferred); explicit line references `router.py:130-141` and `router.py:142-156`.

### B-376. Form-value interactive-elements still RUNNER_INTERNAL per design (P1-9-C*) 📝 PROSE-ONLY (Q8=B keep + disclose)
- **Source**: A1.10 Mode C F5.
- **Code**: `state_change.py:138-152` keeps `form_value_changed` + `interactive_elements_changed` in RUNNER_INTERNAL_REASONS (excluded from AGENT_VISIBLE_REASONS). Mode C argued they should be agent-visible since AXTree text `[N] checkbox checked` is technically visible to agent.
- **Fix**: Per user Q8=B keep current B-09 split semantics + paper §4.X disclosure. Cls/reddit form-heavy task SR may mild-underestimate on similarity > 0.95 cases where form_value is the only signal — disclosure pending advisor sync on whether to add a sensitivity check at re-analysis time.

### B-377. Paper §4.X.12 reproducer env var names wrong (P1-10-C*) 📝 PROSE-FIXED (gemini unique OOB)
- **Source**: A1.10 Mode C F6.
- **Code**: `section4_limitations_disclosure.md:267` told reproducer to set `*_BASE_URL` env vars; `tasks.py:55-62` `_placeholder_mapping` actually reads bare names (`REDDIT` / `SHOPPING` / `CLASSIFIEDS` / `GITLAB` / `WIKIPEDIA` / `MAP` / `HOMEPAGE` / `SHOPPING_ADMIN`). Reproducer following paper would set non-read env vars → silent localhost fallback → network unreachable crash.
- **Fix**: §4.X.12 prose updated with the canonical bare-name env vars and explicit note about the corrected names replacing the earlier `_BASE_URL`-suffix hint.

### B-378. Paper §3.2 phantom_dom "alias" prose vs fail-loud reality (P1-11-C) 📝 PROSE-FIXED
- **Source**: A1.10 Mode C F8.
- **Code**: `section3_definition.md:29` called `phantom_dom` "the legacy mode value retained as alias for paper-grade run dirs" — but `conditions.py:96-117` A1.7 B-261 raises `ValueError` fail-loud when encountered in yaml. Soft-alias prose claim vs hard-block code reality.
- **Fix**: §3.2 prose updated to "deprecated legacy mode value — A1.7 B-261 enforces fail-loud ValueError raise in conditions.py:96-117 when encountered in any new yaml to prevent resume:true-driven silent overwrite of phantom_text data; pre-A1.7 archive run dirs phase1_phantom_dom_router_0/ remain readable via run_registry.py backward-compat normalization phantom_dom + phantom_text → 'P-text'".

### B-379. P2 — checklist regex destroys task intent semantics (defer paper-2) 📋 NOTED (Q21=C defer)
- **Source**: A1.10 Mode A F9.
- **Code**: `checklist_module.py:100` regex `\b(and|then|,|->)\b` splits "Search for blue and green shoes" → ["Search for blue", "green shoes"]. Phase 1a checklist disabled.
- **Fix**: Deferred paper-2 (Phase 3 checklist ablation prerequisite).

### B-380. P2 — config.py double-default for router thresholds (defer) 📋 NOTED (Q22=C defer)
- **Source**: A1.10 Mode A F10.
- **Code**: `config.py:38` DEFAULT_CONFIG + `config.py:195` normalize_config both `setdefault("dom_size_threshold", 12000)`. Hygiene only — setdefault doesn't override, behavior correct; maintenance hazard.
- **Fix**: Deferred. setdefault is idempotent so no active bug.

### B-381. P2 — RouterState.dom_complexity_history bounded BY runner not router (defer) 📋 NOTED (Q23=C defer)
- **Source**: A1.10 Mode A F11.
- **Code**: `router.py:15-16` lists; `runner/main.py:1376-1378` bounds. Non-runner caller (test, future phase4) would unbound-grow.
- **Fix**: Deferred — current single runner caller correct.

### B-382. P2 — VALID_OBS_MODES hardcoded vs runner sentinel handler (defer) 📋 NOTED (Q24=C defer)
- **Source**: A1.10 Mode A F12.
- **Code**: `conditions.py:91-95` whitelist + `runner/main.py:1044` sentinel handler. Adding new sentinel requires syncing 3 places.
- **Fix**: Deferred to paper-2 router experiment integration.

### B-383. Paper §3.2 vs §4.5.4 token gap 733 vs 778 — different pairs not data drift (P2-5-C) 📝 PROSE-FIXED (Q25=A unify)
- **Source**: A1.10 Mode C F9.
- **Code**: `section3_definition.md:37` "733 input tokens per step on reddit (SoM 4275 versus P-text 3542)"; `section4_empirical_findings.md:128` "SoM versus P-SoM observed gap 778 tokens/step". Different denominators (P-text vs P-SoM), not a math error. Mode C misidentified as inconsistency.
- **Fix**: `section4_empirical_findings.md:128` extended with "Note this measures SoM against the screenshot-free P-SoM arm, which differs from §3.2's SoM-vs-P-text image-channel estimate of 733 tokens/step (4275 vs 3542); the two numbers compare SoM against different phantom siblings rather than disagreeing on the same comparison".

---

**B-numbers consumed**: B-359 through B-383 (25 entries spanning 17 active fixes + 5 deferred + 3 prose-only).
**Cumulative session**: A1.4a+b-i+b-ii+c + A1.5 + A1.13+14 + A1.6 + A1.7 + A1.16 + A1.8 + A1.17 + A1.18 + A1.12 + A1.9 + A1.10 = B-140 through B-383 = ~244 unique entries.
**Smoke verification**: pytest 406/406 PASS (398 baseline A1.12 - 1 stale-deferred + 15 new A1.10 negative tests). Compile-check 9 modified files: all PASS.
**Phase 1a fire green-light**: substrate fully paper-grade-defensible post-A1.10. Remaining advisor blockers: B-262 (GLM fallback per parse_advisor_pending.md Thread 1); B-130 (FE/RE estimand per Thread 2); B-369 (schema v2.2 bump for retry attribution, paper-2 prerequisite).

---

## A1.15 /stress audit (2026-05-16) — B-384 to B-394 (11 entries; Pre-fire 闭环 batch, all FIXED)

Scope: watchdog stack + 6-layer auto-clean protocol (`experiment_watchdog.py` 1773 LOC + 3 control scripts). 3-AI cycle (Mode A Claude 11 findings 3 OOB / Mode B codex 9 findings 6 OOB / Mode C gemini 10 findings 5 OOB) = 22 unique attack vectors → 4 P0 + 11 P1 + 6 P2. Pre-fire 闭环 batch lands 11 fixes (5 P0 + 4 P1 + 2 substrate) closing Option K Trajectory Event Log critical path (B-313+B-314 schema laid down A1.17; value-extraction half deferred until this batch).

Cross-AI value (A1.15-specific): codex unique P0 = **B-385 session auto-clean missing trajectory hook** (separate code path from retry; B-314 only covered retry → paper §4 covariate trail systematically false-negative on connected NOT-LOGGED-IN waves) + **B-391 `_run_auto_digest` silent dead path** (`scripts_dir = .parent` resolves to nonexistent `scripts/maintenance/analysis/`). Gemini unique P1 = scan_summaries O(N) glob + restart_watchdog /proc race + 5000-char DOM read limit. Claude unique P0 OOB = **B-387 reddit DOM regex inert** (empirically verified 5/5 reddit DOM 0 match for `link 'Logout'`; Postmill uses dropdown menu) + outcome-dependent retry SR-bias disclosure (T5, paper §3 stub). User strategic decision T8=(a) "Pre-fire 闭环" commits to trajectory_events.jsonl → aggregator covariate trail before fire.

### B-384. `experiment_watchdog.py:1759+` Option K Hook C — session-cleanup path emit `task_auto_cleared` [P0 — codex unique OOB] 🛠️ FIXED
- **Attack**: B-314 (A1.17 Chunk 2) added trajectory event hook to the retry path (L1411-1432), but the `session_contaminated[site]` mass-cleanup loop (L1494-1525, high-frequency auth-loss class entry point) had ZERO `log_trajectory_event_external` call. Codex caught this in A1.15 grep — `log_trajectory_event_external` only appeared once in watchdog. Session-loss waves clear connected/correlated tasks (not white noise) → paper §4 GLMM `had_auth_clear` covariate systematically false-negative with directional bias.
- **Cascade**: 14 reddit cells × N session-loss waves during Phase 1a = potentially dozens of unrecorded auto-clean events; aggregator output omits `had_auth_clear` for affected episodes → §4 GLMM regression coefficient biased toward zero → main treatment effect contaminated by uncontrolled-for auth-clear perturbation.
- **Fix**: Per-task emit inside session cleanup loop with wave-context metadata (`is_auth_loss=True`, `cleared_in_session_wave=True`, `wave_size`, `wave_task_index`, `is_noise=True`, `site`). Best-effort try/except wrapper per T2'=(a) decision. Distinguishes from retry-path emit via `cleared_in_session_wave` bool.

### B-385. `p79/experiment/logger_v2.py:106-134` Option K schema doc update + P0-4 reframe note [Schema clarification + codex P0-4 reframe] 🛠️ FIXED
- **Attack**: Schema doc listed event_type `"auth_clear_task"` but actual hook emits `"task_auto_cleared"` (mismatch from B-314 implementation). P0-4 condition_finalize race (codex unique OOB at A1.15) lacked schema guidance — runner-side `condition_finalizing.lock` would have been heavy fix.
- **Fix**: Updated schema event_type enum to actual emit values; added metadata key catalog; documented post-hoc race detection (aggregator intersection of condition_summary task_ids with `task_auto_cleared` events = race-cleared episodes, no new event_type needed). Race fully resolved via aggregator covariate `had_finalize_race_clear` rather than via runner mutex primitive — best-effort emit substrate sufficient.

### B-386. `experiment_watchdog.py:1621-1657` Option K race-ordering disclosure + paper §4.X.13 stub [P0 — 2-AI A+B race window] 🛠️ FIXED (best-effort + disclosure per T2'=a)
- **Attack**: B-314 event write happens AFTER destructive ops (`unlink` + `rmtree` + `_purge_digest_records` + `_persist_state`). `restart_watchdog.sh kill -9` race window (2-3s) drops event while filesystem mutation persists → paper §4 covariate undercount.
- **Decision T2'=(a)**: Best-effort + paper §3 disclose (vs 2-phase commit which would add 1.5h with marginal ROI given mode-symmetric drop direction).
- **Fix**: Enhanced inline comment explicitly documenting race window + paper §3 disclosure pointer + `cleared_in_session_wave: False` metadata addition (distinguishes from B-384 session path). Paper §4.X.13 stub added to `section4_limitations_disclosure.md` documenting best-effort enrichment + sensitivity analysis Supp Table S-trajectory-loss planned post-data.

### B-387. `experiment_watchdog.py:34-91, 303-317` Reddit DOM regex inert — per-site `_SITE_AUTH_REGEX` dict [P0 — Claude unique OOB, empirically verified] 🛠️ FIXED
- **Attack**: Pre-fix single `_LOGIN_PRESENT_RE = link\s+'(?:Logout|Log out|Sign Out)'` designed for OSClass/Magento DOM serializer format. Reddit Postmill DOM uses dropdown menu structure (`"Profile"`, `"My account"`, `"User settings"`, `"Block list"`) rather than explicit `link 'Logout'`. **Empirically verified 2026-05-16**: 5/5 reddit step_000 DOM files (6.9-7.2KB, sample from `B0_phantom_som_reddit_20260428`) matched neither old regex → `_check_session_health` returned `None` for ALL reddit tasks → `session_loss_streak[reddit]` never incremented → session_contaminated never accumulated → auto-clean protocol completely inert for 14 reddit cells. Paper §4 "auto-clean protects all 42 cells" claim audit replay falsified.
- **Fix**: Per-site `_SITE_AUTH_REGEX: Dict[str, Tuple[Pattern, Pattern]]` with site-specific (logged-out, logged-in) markers — classifieds = OSClass `link 'Login/Logout'`, shopping = Magento `link 'Sign In/Sign Out/My Account'`, shopping_admin = `link 'Login/Account'`, reddit = `link 'Log in/Sign up'` for logged-out + `DROPDOWN OPTIONS|"My account"|"User settings"|"Block list"|link 'Profile'` for logged-in. `_check_session_health` dispatches via site key with classifieds fallback.
- **Smoke verified**: 5/5 reddit DOM files now correctly detected as logged-in (was 0/5).

### B-388. `p79/experiment/logger_v2.py:190+` runner-side staging pickup helper `merge_staging_trajectory_events` [Merge (i) — closes deferred follow-up from B-314] 🛠️ FIXED
- **Attack**: B-314 reset gate writes `reset_post_interrupt` events to `${repo_root}/logs/trajectory_events_staging/RUN_${RUN_ID}.jsonl` because condition_dir doesn't exist yet. Without pickup the staging file accumulated events but never made it into per-condition `trajectory_events.jsonl` → paper §4 aggregator received zero events from reset class → Option K Tier 1 stack analysis layer was effectively dead for reset perturbation.
- **Fix**: New module helper `merge_staging_trajectory_events(condition_dir, run_id, repo_root)`. Idempotent via "fresh-dir-only" guard. Cell-level events duplicated across each condition_dir under same RUN_ID by design (each condition's covariate view sees its own copy). Preserves original wallclock_ts + adds `merged_from_staging=True` + `staging_run_id` metadata. Atomic per-event append + fsync.
- **Call site**: Runner `p79/experiment/runner/main.py:513-540` immediately after `condition_dir.mkdir(parents=True, exist_ok=True)`. Best-effort try/except — failure surfaces warning but does not abort.

### B-389. `scripts/analysis/aggregate_trajectory_covariates.py` (NEW) Option K covariate aggregator [Aggregator (iii) — closes deferred follow-up] 🛠️ FIXED
- **Attack**: Without aggregator implementation, trajectory_events.jsonl files would accumulate during Phase 1a fire but never feed into paper §4 GLMM — making the entire Option K substrate dead artifacts.
- **Fix**: New script `aggregate_trajectory_covariates.py` (~233 LOC). Reads `trajectory_events.jsonl` + `condition_summary_v2.json` per condition_dir; emits per-episode JSONL + CSV at `<condition_dir>/analysis/trajectory_covariates.{jsonl,csv}` with columns: `is_after_reset` / `had_auth_clear` / `had_finalize_race_clear` (post-hoc race detection per B-385) / `cleared_in_session_wave` / `session_wave_size` / `prior_event_count` / `n_task_events` / `ep_wallclock_start`. Defensive schema parsing handles both v2 (`episode_summaries: list`) and legacy (`episodes: int` count → filesystem scan fallback). CLI: `--run-dir <run> [--condition <cid>]`.

### B-390. End-to-end Pre-fire 闭环 smoke test (B-313~B-389 chain) [Substrate verification] 🛠️ FIXED
- **Test**: Fake condition_dir → write staging file with `reset_post_interrupt` → `merge_staging_trajectory_events` (B-388) pickup → emit 4 `task_auto_cleared` events (1 retry-path + 3 session-wave per B-384 metadata) → write `condition_summary_v2.json` with 5-task episode list → run aggregator (B-389) → verify all 5 covariate column values + idempotency (re-merge returns 0) + race detection (4 tasks with `had_finalize_race_clear=True`, 1 clean task False) + legacy schema fallback on archived run.
- **All assertions PASS**: covariate columns correct; idempotent re-merge; archived run (`episodes: int(3)`) aggregator does not crash (3 episodes emit via filesystem scan).

### B-391. `experiment_watchdog.py:1191, 305-435, 1218` `_run_auto_digest` silent dead path [P1 — codex unique OOB, empirically confirmed] 🛠️ FIXED
- **Attack**: Pre-fix `scripts_dir = Path(__file__).parent` resolved to `scripts/maintenance/`, but L1005 looked for `scripts/maintenance/analysis/analyze_reason_diagnostics.py` — directory doesn't exist (verified `ls`). `diag_script.exists()` permanently False → silent return None → watchdog `_run_auto_digest` dead path through entire Phase 1a wall. `_run_post_condition_analysis` (L532) correctly uses `.parent.parent = scripts/` — sibling inconsistency from §99 reorg. Additionally `_DIGEST_MODES = ("dom","som","vision")` hardcoded didn't include Phase 1a phantom modes (P-text/P-prompt/P-SoM).
- **Fix**: (a) `_run_auto_digest` scripts_dir = `.parent.parent` mirror L532; (b) `digest_script = scripts_dir / "maintenance" / "glm" / "glm_batch_digest.py"` (post-2026-05-02 GLM sidecar reorg); (c) new helper `_get_active_digest_modes(run_dir, ...)` runtime-derives modes from `condition_meta.json["observation_mode"]` to cover all 6 Phase 1a modes.

### B-392. `experiment_watchdog.py:145-280, 1791+` `_purge_digest_records_batch` + fsync [P1 — 2-AI A+C purge perf+durability] 🛠️ FIXED
- **Attack**: (a) gemini OOB: pre-fix `_purge_digest_records` called per-task in session cleanup loop B-384 → O(N·M) read+filter+rewrite (100 records × multi-MB digest = ~200MB I/O) → watchdog hangs minutes during session-restore mass cleanup. (b) Claude: rename `tmp_file.replace(digest_file)` no fsync — durability sibling inconsistency with `_save_state` atomic+fsync+dirsync pattern.
- **Fix**: New `_purge_digest_records_batch(digest_dir, obs_mode, keys_to_remove: Set[Tuple[str, int]])` reads once, filters by set membership, writes once with fsync+replace+dirsync. Session cleanup loop collects `_purge_keys_by_mode: Dict[str, Set]` during destructive op pass then runs ONE batch purge per mode after the loop. Single-key `_purge_digest_records` retained as thin wrapper for retry-path B-314 call site. Idempotent: empty keys or no matches returns 0 without touching disk.

### B-393. `experiment_watchdog.py:618-700, 1308-1316` `_load_state` fail-closed on corrupt state [P1 — 2-AI B+C corrupt state silent reset] 🛠️ FIXED
- **Attack**: Pre-fix `except Exception: return {}` silently reset on ANY parse/I/O error → `error_retry_counts` cleared (= "已耗尽 retry 任务被重新调度") + `session_contaminated` cleared (= "受损 episode 永留在 result set"). B-223 (A1.5) landed atomic write via `_save_state`; read-side never matched → durability sibling inconsistency.
- **Fix**: Distinguish `FileNotFoundError` (OK return {} — clean first launch) from `JSONDecodeError/OSError/UnicodeDecodeError` (rename `path.with_suffix(".corrupt.<ts>")` + urgent ntfy + raise SystemExit unless caller passed `reset_state=True`). Forwarded `reset_state` parameter through main(). Fail-closed default is paper-grade-correct; operator can opt-in to discard via `--reset-state`.

### B-394. `scripts/maintenance/wait_for_reddit_then_rederive.sh` DELETED + Makefile `watch-reddit` target retired [P1 — 3-AI overlap, T6=(a) decision] 🛠️ FIXED
- **Attack**: 3-AI overlap finding. Header L14 comment claimed "Step 5: Does NOT auto-start B1 shopping queue" but body L52-62 actually DOES auto-launch `queue_b1_with_reset.sh`. Hardcoded `B1` baseline + `RUN_DIR=20260413` (April-specific). B2 Gemma3-VL joined baseline set 2026-05-14 → script stale + cross-baseline collision risk (CLAUDE.md "B0 XOR B1 XOR B2 同 site" hard rule violation).
- **Decision T6=(a)**: Delete (vs generic refactor which would preserve problematic pattern).
- **Fix**: `git rm scripts/maintenance/wait_for_reddit_then_rederive.sh`; Makefile `watch-reddit` target body replaced with retirement comment + .PHONY removed + header doc comment updated; `schedule-list` target pgrep pattern updated (removed `wait_for_reddit`, added `queue_b2|queue_chain` for B2 + chain coverage); `scripts/maintenance/README.md` entry strikethroughed. Paper-grade reddit workflow uses `queue_chain.sh` (3-baseline aware, collision-safe).

**B-numbers consumed**: B-384 through B-394 (11 contiguous, latest-checked from master HEAD `ddc29ff` at chunk start; reverse-pass renumbered from initial B-315~B-325 after detecting parallel A1.9/A1.10/A1.12 occupation of B-315~B-383).

**Smoke verification**:
- py_compile PASS: `experiment_watchdog.py` / `logger_v2.py` / `runner/main.py` / `aggregate_trajectory_covariates.py`
- Makefile syntax PASS (`make -n schedule-list` / `make -n help` / `make watch-reddit` correctly errors "No rule")
- B-387 empirical: 5/5 reddit DOM 0→5 logged-in detection ✓
- B-390 end-to-end smoke: 5 covariate rows correct + idempotency + race detection + archived sanity ✓

**Worktree**: `/home/jiaming/workspace/p79-a1.15` branched from master `057f7aa` on branch `a1.15-stress`. Merged in master `ddc29ff` (= parallel A1.9+A1.10+A1.12 land) before docs append; 1 conflict in `logger_v2.py` (B-331 atomic + B-388 helper both at file tail) — resolved keeping both. Merge back to master via `git merge --no-ff a1.15-stress` (per A1.17 worktree workflow §163.5).

**Pre-fire 闭环 completion status**:
- ✅ Schema (B-313 prior + B-385 doc update)
- ✅ API (B-313 prior + B-388 staging-merge helper)
- ✅ Hook A retry path (B-314 prior + B-386 race-window comment)
- ✅ Hook B reset gate (B-314 prior)
- ✅ Hook C session-cleanup path 🆕 (B-384)
- ✅ Hook D race-window detection — post-hoc via aggregator 🆕 (B-385)
- ✅ Merge runner-side staging pickup 🆕 (B-388)
- ✅ Aggregator covariate emission 🆕 (B-389 + B-390 smoke)
- 🟡 Paper §3 Multi-Epoch reframe (deferred (ii), codex round + advisor sync)
- 🟡 Paper §4 GLMM regression (deferred (iii) downstream, post-data, reads aggregator output)
- 🟡 Fisher rebuttal (deferred (iv), post-data)

**Cumulative session**: A1.4a+b-i+b-ii+c + A1.5 + A1.13+14 + A1.6 + A1.7 + A1.16 + A1.8 + A1.17 + A1.18 + A1.12 + A1.9 + A1.10 + A1.15 = B-140 through B-394 = ~255 unique entries.
**Phase 1a fire green-light unchanged**: substrate paper-grade post-A1.15 Pre-fire 闭环 land. Remaining advisor blockers unchanged: B-262 (GLM fallback Thread 1); B-130 (FE/RE estimand Thread 2); B-369 (schema v2.2 retry attribution, paper-2 prerequisite).
**Next available B-number**: B-395+.

---

## A1.1 batch (B-395~B-405, 3-AI cross-audit, 2026-05-16)

3-AI cross-AI /stress cycle (Claude Mode A + codex Mode B + gemini Mode C) on `p79/agents/` step contract. User Phase A directive: 永远最 clean paper grade; GLM rescue paper-grade run 全禁。 Commits: `0f3a7c2` (batch 1: B-395~B-401) + `d765dbf` (batch 2: B-402/403/405).

### B-395. `paper_grade` flag end-to-end wire — 3-AI overlap P0-1 [A+B+C — highest confidence] 🛠️ FIXED commit `0f3a7c2`
- **Attack**: B-340 hard-block at `proxy_api_agent.py:179-186` raises iff `config["paper_grade"]=True`, but no upstream path ever sets the flag (0 grep hits across `configs/`, `scripts/queues/*.sh`, env-var exports). `runner/main.py:209` setdefault'd to `False`. Claude Mode A code-trace + codex Mode B grep (`configs/exp_v2_B0_som_reddit.yaml:36 use_glm_fallback:true` + 39 sibling configs) + gemini Mode C paper-§3 fairness rhetoric attack all converged on same root.
- **Fix**: `config.py:normalize_config` merges env `P79_PAPER_GRADE` (1/true/yes/on) → top-level `cfg["paper_grade"]=True`. `queue_phase1_paper_grade.sh:67` + `queue_phase1_router_paper_grade.sh:58` export the env. Reaches existing B-340 raise; agent fail-fast at init if any yaml still has `use_glm_fallback:true`.

### B-396. 39 live B0 yaml configs `use_glm_fallback: true → false` (P0-1 defense-in-depth sibling) 🛠️ FIXED commit `0f3a7c2`
- **Attack**: All 39 paper-grade B0 yamls (cls/red/shop/WA × 6 modes incl. `exp_v2_B0_router_learned_classifieds.yaml:62`) explicitly opt-in to GLM fallback at config layer, masking B-395 fix.
- **Fix**: `sed -i 's/use_glm_fallback: true/use_glm_fallback: false/'` mass-flip + per-line audit comment "B-396 (/stress A1.1 P0-1 3-AI overlap, 2026-05-16): paper-grade hard-off (was true)". `configs/exp_v2_base.yaml:173 use_glm_fallback: false` (default) unchanged. Defense-in-depth: config-off + B-395 hard-block reachable + B-340 RuntimeError raise.

### B-397. `image_meta_recorded` cross-baseline asymmetry — 2-AI A+B overlap P0-2 🛠️ FIXED commit `0f3a7c2`
- **Attack**: B-324 truth source `image_payload_bytes is not None` only fires on B0 (proxy JPEG pipeline); B1/B2 HF-processor PIL path has no payload_bytes → `image_meta_recorded` permanently False on ALL B1/B2 SoM/vision steps. Image-mode set also incorrectly included `phantom_som` (per `som.py:322-323` phantom_som strips image). Any aggregator filter on `image_meta_recorded == True` silently excludes B1/B2 image-axis data.
- **Fix**: `runner/main.py:1768-1785` backend-aware OR — `input_image_tokens > 0` (B1/B2) ∨ `image_payload_bytes is not None` (B0); image-mode set narrowed to `{"som", "vision"}` only (phantom_som excluded — no image by design).

### B-398. `glm_fallback_attempted` unconditional persistence — 2-AI A+B overlap P0-3 🛠️ FIXED commit `0f3a7c2`
- **Attack**: `runner/main.py:1703 if meta.get("glm_fallback_used"):` truthy-check skipped attempted-but-failed cases. JSONL collapsed `attempted=True, used=False` → same shape as `never tried` (both default None per `STEP_RECORD_V2_DEFAULTS`). Reviewer audit "GLM hit rate" returned 20/100=20% instead of 20/30=67% — off by ~47%.
- **Fix**: New field `glm_fallback_attempted: Optional[bool]` in `types.py:135` + `STEP_RECORD_V2_DEFAULTS:149`. Runner emits all 4 fields when `attempted=True` regardless of success/failure (`runner/main.py:1722-1735`). Audit trail fully reconstructable from JSONL.

### B-399. `total_minus_retry` failed-attempt elapsed accounting — Mode A unique P1-1 🛠️ FIXED commit `0f3a7c2`
- **Attack**: `proxy_api_agent.py:592` retry loop accumulated only `wait * 1000.0` (sleep time) into `_retry_wait_ms_total`. Pre-fix: 120s timeout + 10s sleep + 30s success → total=160s, network_retry_wait_ms=10s, `total_minus_retry=150s` — true scaffold overhead was 130s. Retry-frequent sites systematically inflated.
- **Fix**: Capture `_attempt_start = time.time()` per retry; failed attempts (`Timeout/ConnectionError` OR retryable status code) accumulate `(time.time() - _attempt_start) * 1000 + wait * 1000` to `_retry_wait_ms_total`. Success path attempt's elapsed remains LEGITIMATE network cost (not scaffold) — not accumulated.

### B-400. `image_payload_bytes_{total,ref,screenshot}` cost field — 2-AI A+C overlap P1-2 🛠️ FIXED commit `0f3a7c2`
- **Attack**: B0 meta `image_payload_bytes` (line 767) tracked screenshot ONLY; reference-image payloads (1-3 product photos × 30-100KB each) were silently omitted. Paper §1 cost claim under-reported B0 absolute egress for any task with reference images. Mode A F5 + Mode C F7 (Gemini) two-AI overlap.
- **Fix**: `proxy_api_agent.py` `_ref_payload_bytes_total` accumulator over ref-image loop; meta emits 3 fields: `image_payload_bytes_screenshot` (back-compat = old field semantic), `image_payload_bytes_ref`, `image_payload_bytes_total` (sum). Runner `_image_meta_payload` (`main.py:1769-1773`) pipes all 3 to step_record. Aggregator should prefer `_total` for cross-task cost comparison.

### B-401. B0 latency-split honest `None` — Mode A unique P1-3 🛠️ FIXED commit `0f3a7c2`
- **Attack**: B0 meta did not emit `preprocess_ms` or `generate_ms` keys (API boundary has no preprocess/generate split exposed). `runner/main.py:1638 float(meta.get("preprocess_ms", 0.0))` default-coerced to 0.0 → B0 latency rows looked like "preprocessing=0, generate=0, backend_infer=full_network" while B1/B2 had real numbers. Schema-dishonest.
- **Fix**: B0 meta explicit `"preprocess_ms": None, "generate_ms": None` (`proxy_api_agent.py:766-770`). Runner records `None` instead of default 0.0 (`runner/main.py:1672-1683`). Paper §3 latency-split disclosure can now honestly note "B0 API boundary not exposed; latency-split N/A".

### B-402. B0 prompt builders direct import from `_shared_vl_utils` — Mode A unique P1-4 🛠️ FIXED commit `d765dbf`
- **Attack**: `proxy_api_agent.py:344 from p79.agents.qwen3vl_agent import Qwen3VLAgent` transitively imported `transformers.Qwen3VLForConditionalGeneration` + `qwen_vl_utils.process_vision_info` heavy deps. B0 is a pure network-call agent that needs none of them. Contradicts B-146 _shared_vl_utils extraction intent ("Gemma3VLAgent failed at first launch in environments without Qwen deps").
- **Fix**: Direct `from p79.agents._shared_vl_utils import make_dom_prompt, make_som_prompt, make_vision_prompt`. `test_agents_prompt_parity` (9/9 PASS) confirms byte-identical via shared SOT.

### B-403. `image_encode_error_step_count` symmetric-exclude — Mode B unique P1-9 🛠️ FIXED commit `d765dbf`
- **Attack**: Agent comments (`qwen3vl_agent.py:355-363` + `gemma3vl_agent.py:330-336`) mandated "aggregate_*.py MUST symmetric-exclude steps with image_encode_error > 0" for paper-grade cross-baseline SR comparability. Pre-fix: 0 aggregator implemented exclusion. `aggregate_sr_fp_per_mode.py:78` + `aggregate_phantom_lift.py:117` consumed only `success` from condition summary — infra failures (PIL decode / base64 OOM on B0 proxy) silently scored as model/task failures.
- **Fix**: `EpisodeSummaryV2.image_encode_error_step_count: int = 0` field (`types.py:213`) + STEP_RECORD_V2_DEFAULTS mirrored (`schema_migrations/v2.py:79`). Runner sums step_records image_meta image_encode_error per episode (`runner/main.py:2169-2174`). `aggregate_sr_fp_per_mode.aggregate_cell` emits 5 new transparency columns: `n_image_encode_error_episodes`, `image_encode_error_episode_rate`, `n_clean`, `n_success_clean`, `sr_pct_clean`. Reviewer can compare `sr_pct` vs `sr_pct_clean` — gap >> 0 = infra-failure contamination.

### B-404. P2-3 `phantom_dom` alias paper §3 disclosure 🟢 ALREADY SATISFIED — no action
- **Attack** (Gemini Mode C F3): code has `phantom_dom` + `phantom_text` (two mode names, identical dispatch) but prereg收敛 "3 arms" → reviewer audit could read as "4 names → 3 structures, author 注水"
- **Status**: `section3_definition.md:29` already explicitly discloses: "`phantom_dom` is the deprecated legacy mode value — A1.7 B-261 enforces fail-loud `ValueError` raise in `conditions.py:96-117` ... `phantom_text` is the current canonical name". Additionally `pre_run/preregistration.md:39` proactively rejects "3-arm strict" framing ("Phantom space is a structural claim, not a 3-arm deployment claim"). No additional prose change required; recorded as defused-by-existing-disclosure.

### B-405. `aggregate_cross_site._get_adjusted_sr` archive-only warning — Mode B unique P2-4 🛠️ FIXED commit `d765dbf`
- **Attack**: `aggregate_cross_site.py:154` `_get_adjusted_sr()` helper + line 205 output row `adjusted_sr` column survived §139.8 retirement of post-hoc adjusted_success layer. Old cross-site tables silently re-introduced retired FP-filter framework if user re-ran old script.
- **Fix**: Helper docstring labels archive-only-path; populated values now log `[B-405 legacy-archive]` warning to stderr; output column retained for back-compat with `cross_representation_summary` archives. Paper-grade callers reading the warning know to cite `raw_sr` not `adjusted_sr`. Future v3 schema bump can drop the column entirely.

**B-numbers consumed**: B-395 through B-405 (11 contiguous; gap B-384~B-394 reserved per user directive 2026-05-16 mid-batch renumber for parallel chunk allocation).

**Smoke verification**:
- py_compile PASS: `config.py` / `runner/main.py` / `types.py` / `schema_migrations/v2.py` / `proxy_api_agent.py` / `aggregate_sr_fp_per_mode.py` / `aggregate_cross_site.py`
- Tests 39/39 PASS: `test_agents_prompt_parity` (9) + `test_stress_a1_2_fixes` (16) + `test_stress_a1_4b_ii_g2_fixes` (6) + `test_runner_smoke` (7) + `test_runner_integration` (1)
- B-395 + B-396 defense-in-depth verification: 39 yaml configs `use_glm_fallback:false` post-flip + B-395 env wire reaches B-340 raise

**Cross-AI overlap summary** (paper-grade signal calibration):
- 3-AI overlap (Mode A+B+C): B-395 (paper_grade inert) + Mode A F8 ↔ Mode B F2 ↔ Mode C F4 (B0 determinism gap, prose-only disclosure deferred next codex round)
- 2-AI A+B overlap: B-397 (image_meta_recorded) + B-398 (glm_attempted lost)
- 2-AI A+C overlap: B-400 (image_payload_bytes excludes ref)
- Mode A unique (code-only): B-399 (total_minus_retry bias), B-401 (preprocess/generate=0), B-402 (transitive import)
- Mode B unique (sys-engineer reproducibility): B-403 (image_encode_error no aggregator), B-405 (adjusted_sr leak)
- Mode C unique (paper-prose): P0-4 (rule router §1 dead-trigger), P0-6 (FP-gap behavioral overinterpretation), P1-5 (tokenizer-level byte identity), P1-6 (B0 determinism §3 move) — **DEFERRED to next codex round**
- Mode B 2 paper-2-scope catches (cross-family `--model-revision` default None + NVRTC fallback no telemetry) → defer to paper-2 mechanism resume per Phase A scope discipline

**Deferred to next session**:
- Mode C prose codex round (P0-4 / P0-5 / P0-6 / P1-5 / P1-6) — paper §1 + §3 prose edits + codex round verification
- Mode B paper-2 scope (P1-7 cross-family revision, P1-8 NVRTC telemetry, P1-10 confidence in summary) — paper-2 mechanism resume hard gate

**Phase 1a fire green-light**: substrate paper-grade post-A1.1 batch (3-AI cross-audit). GLM hard-block fully reachable (B-395 + B-396 + B-340 stacked); cross-baseline schema asymmetries closed (B-397/398/400/401); audit-trail intact (B-398/403/405). Phase A "永远最 clean paper grade" directive satisfied for §A1.1 scope.

---

## /stress A1.2 batch — `p79/backends/` cross-baseline contract (2026-05-16 late-night)

3-AI cycle: Mode A Claude (5 findings, 3 OOB) / Mode B codex (7 findings, 5 OOB, 15135 bytes) / Mode C gemini (5 findings, 3 OOB, 7069 bytes) = 16 unified findings → 11 code fixes landed (4 prose deferred to next codex round per user Q&A `disclose-only` decisions).

**Cross-AI agreement signal**:
- 3-AI overlap: 1 finding (`heuristic_only` B2 missing branch, B-408)
- 2-AI A+B overlap: 1 (B-406 coord_type=normalized but pixel coord, 851 empirical rows match)
- 2-AI A+C overlap: 1 (`tokens.input_image` B0=0 / B1=1280 asymmetry, P0-4 deferred prose)
- 1-AI unique: 12 findings unique to each lineage (Mode A 4 / Mode B 5 / Mode C 3)

### B-406. `_is_valid_coordinate_pair` coord_type-aware strict enforcement — 2-AI A+B P0 OOB 🛠️ FIXED commit `4c559d2`
- **Attack**: `p79/backends/action_utils.py:140` pre-fix `allow_pixel=True` (default) accepted any non-negative finite pair, even when `coordinate_type="normalized"`. Mode A empirical spot-check + Mode B independent jq query both confirmed 851/3561 (24%) normalized-declared rows with coord >1; B0 15.6% / B1 35.3% (2.3× cross-baseline asymmetry); all parse_valid=true. env wrapper `vwa_wrapper.py:336` silent auto-normalize → schema violation collapsed into env behavior → paper §3.5 error taxonomy contaminated + §1 hero number cross-baseline averaging biased.
- **Fix**: `_is_valid_coordinate_pair(coord, coordinate_type=None, allow_pixel=True)` 加第 2 形参; `coordinate_type="normalized"` → strict [0,1] enforcement; `coordinate_type="pixel"` → non-negative finite; None falls back legacy. Click / type / select_option all propagate `coordinate_type` from action dict.

### B-407. `type` action target required — 1-AI Mode B P0 OOB 🛠️ FIXED commit `4c559d2`
- **Attack**: `p79/backends/action_utils.py:261` pre-fix `type` branch only checked `text`, accepted `{"action_type":"type","text":"\n"}` parse_valid=true. Empirical 23 rows in archive (B0 cls task 204 step 1 etc.). env wrapper cannot execute targetless type → schema-violation silently grouped under no_progress → paper §3.5 taxonomy + cross-baseline SR contaminated.
- **Fix**: type branch end `if not (has_id or coord_valid_shape): return invalid_element_id`.

### B-408. `dom_mode` canonical enum + B2 fail-loud — 3-AI A+B+C P1 OOB 🛠️ FIXED commit `abb0900`
- **Attack**: 3-AI overlap finding. Pre-fix 3 drift sources: (1) `tests/test_runner_smoke.py:80` 用 `"heuristic"` (no _only) / (2) `api_proxy.py:99` + `local_qwen.py:57` 只匹配 `"heuristic_only"` → silent no-op for `"heuristic"` / (3) `local_gemma.py:21` comment "No dom_mode/heuristic branch" — B2 has no branch at all. Same primitive 3 semantics across baselines.
- **Fix**: `p79/backends/factory.py:_ALLOWED_DOM_MODES = {"llm", "heuristic_only"}` enum validated at dispatch for local_qwen/local_gemma/api_proxy. B2 `local_gemma.py.__init__` raises `NotImplementedError` on heuristic_only (B2 has no HeuristicDomBackend wrapper). 4 test files updated to canonical `"heuristic_only"` (test_runner_smoke / test_stress_behavioral_retrofit / test_runner_integration / test_backends_mock_dispatch_parity).

### B-409. `multiple_actions` dedup signature widened — 1-AI Mode B P1 OOB 🛠️ FIXED commit `4c559d2`
- **Attack**: `action_utils.py:111` pre-fix dedup key `(action_type, element_id, text)` collapsed two different-coord clicks as "identical" → executed first → system bias toward first hallucinated candidate (vision-mode especially exposed).
- **Fix**: Per user Q2=A "full-field" signature; dedup key now includes coordinate / coordinate_type / delta / scroll_direction / answer / option_label / option_value / option_index / page_number (12 fields total).

### B-410. Yaml `temperature/top_p` dead-config warning B1/B2 — 1-AI Mode A P1 🛠️ FIXED commit `abb0900`
- **Attack**: `local_qwen.py:42` + `local_gemma.py:42` forward yaml `temperature` to agent, but `qwen3vl_agent.py:299` + `gemma3vl_agent.py:279` hardcode `do_sample=False` → yaml temp silently dead on B1/B2. B0 `proxy_api_agent.py:572` honors yaml temp. If yaml drifts to `temperature: 0.7`, only B0 goes non-deterministic → cross-baseline reproducibility asymmetric, no fail-fast guard.
- **Fix**: LocalQwen / LocalGemma `__init__` emit `logger.warning` if temp != 0.0 or top_p != 1.0.

### B-411. `paper_grade` flag wire-through to local backends — 1-AI Mode A P1 🛠️ FIXED commit `abb0900`
- **Attack**: `api_proxy.py:94` forwards `paper_grade` to ProxyApiAgent (consumed at line 179 for B-340 GLM hard-block); `local_qwen.py:22-52` + `local_gemma.py:34-52` agent_cfg DROP paper_grade. Currently inert (local agents have no paper_grade gate yet), but **latent contract gap** for future paper-grade-only guards (`torch.use_deterministic_algorithms` / `CUBLAS_WORKSPACE_CONFIG` / cuDNN benchmark off / revision drift fail-fast).
- **Fix**: `local_qwen.py:62` + `local_gemma.py:75` agent_cfg add `"paper_grade": bool(config.get("paper_grade", False))`. Defense-in-depth wire-up; no current behavior change.

### B-412. Naked scroll require targeting field — 1-AI Mode B P1 🛠️ FIXED commit `4c559d2`
- **Attack**: `action_utils.py:270` pre-fix `{"action_type":"scroll"}` parse_valid=true (delta None branch fell through). VWA env `vwa_wrapper.py:356` requires `delta` OR `scroll_direction` to execute. Empirical 2 rows in archive (B1 cls task_174 step_1). B0 `proxy_api_agent.py:745` had `scroll_direction→delta` conversion, B1/B2 lacked it → cross-baseline asymmetric.
- **Fix**: Validator now requires at least one of `delta` / `scroll_direction in {up,down}` / `direction in {up,down,left,right}` (WebArena-legacy alias kept for cross-benchmark compat via `vwa_wrapper.py:800` consumer). Single source of truth at validator.

### B-413. Repair path detailed reason propagation — 1-AI Mode B P1 🛠️ FIXED commit `4c559d2`
- **Attack**: `action_utils.py:86` + `:106` repair path (fenced JSON / raw_decode) used 2-tuple `validate_action()` and discarded reason → all sub-category failures collapsed to generic `invalid_action_repaired`. Markdown/prose-wrapped JSON case (common VLM output pattern) hid real sub-category from paper §3.5 taxonomy.
- **Fix**: Repair path now uses 3-tuple `validate_action_detailed()` and stores `(action, is_valid, reason)`. Invalid-only fallback propagates FIRST candidate's specific reason (`invalid_action_type` / `invalid_element_id` / etc.) instead of generic.

### B-414. `first_element_id_by_keyword` role-anchored — 1-AI Mode B P2 OOB 🛠️ FIXED commit `4c559d2`
- **Attack**: `action_utils.py:306` pre-fix substring match against whole line; `[12] StaticText 'click the blue button'` matched "button" → returned eid=12 (StaticText not clickable). Sibling propagation: 7 callsites (heuristic.py x2, modules.py M1, runner/helpers M2, ...) inherited the bug from single primitive.
- **Fix**: Parse role token after `[N]` via anchored regex `_ROLE_RE = re.compile(r"\[\s*\d+\s*\]\s+(\S+)")`. Match keyword ONLY against whitespace/underscore-normalized role string. Single-source fix propagates to 7 callsites.

### B-415. MockBackend tag canonical naming — 1-AI Mode A P2 🛠️ FIXED commit `abb0900`
- **Attack**: 4 mock naming schemes (factory `mock_<id>` / local_qwen `local_qwen_mock` / local_gemma `local_gemma_mock` / api_proxy `api_proxy_mock`) → cross-baseline mock invariance tests grepping one pattern miss the other 3.
- **Fix**: All 4 emit `f"mock_{self.backend_id}"`. `test_local_gemma_mock_mode_emits_canonical_scroll` updated to assert `mock_b2_mock`.

### B-416. `image_utils.encode_image_data_url` `.encode("utf-8")` redundant — 1-AI Mode A P2 🛠️ FIXED commit `abb0900`
- **Attack**: `b64` is already ASCII (line 52 `base64.b64encode().decode("ascii")`), so `len(b64.encode("utf-8")) == len(b64)`. Per-loop redundant encode call.
- **Fix**: Replace `len(b64.encode("utf-8"))` → `len(b64)`. ~6-8 presets × ~80k step calls/condition = ~640k allocs saved per Phase 1a fire. No behavior change.

**B-numbers consumed (A1.2)**: B-406 through B-416 (11 contiguous; 4 deferred prose B-IDs reserved B-417~B-420).

**Smoke verification**:
- py_compile PASS: `action_utils.py` / `factory.py` / `local_qwen.py` / `local_gemma.py` / `api_proxy.py` / `image_utils.py`
- Tests 414/414 PASS (411 prior + 3 new: test_validate_scroll_accepts_direction_aliases new; B-412 + B-413 + B-415 + B-408 4 tests updated to new contracts)

### Deferred to next codex prose round (paper §3.5 + §4 disclosure batch)

- **P0-3 (Mode A) max_image_payload_bytes asymmetric**: 39 B0 yamls set 5MB cap / 0 B1+B2 yamls. B0 path `image_utils.encode_image_data_url` JPEG base64; B1/B2 path direct PIL → HF processor unbounded. User Q1=A disclose-only. Paper §3.5 disclose + §4 cost-figure caption update. ~30 min codex round.
- **P0-4 (Mode A+C 2-AI) tokens.input_image asymmetry**: Empirical B0 som task_167 step_0 `input_image=0`, `input=4749` includes image; B1 same task `input_text=2636, input_image=1280` explicit split. User Q3=A disclose-only. Paper §3.5 + §4 cost-fairness explicit "B0 image tokens not observable; cross-baseline cost only valid at total tokens layer". ~25 min codex round.
- **P1-7 (Mode C) historical B0 GLM rescue framing**: paper §1 + §3.5.1 figure caption explicit "data source = post-A1.1 paper-grade re-fire (P79_PAPER_GRADE=1, GLM hard-block), legacy 20260413 archive 不进 main figure". ~30 min codex round.
- **P2-2 (Mode C) invalid-action sub-category paper §3.5/§4 figure**: paper §4 mechanism add sub-category pie chart leveraging `validate_action_detailed` 6 sub-reason. ~4 hour analysis + figure work (P2 defer-able).

### Deferred to test infra round

- **P2-5 (Mode C) MockBackend `mock_strategy` parameter**: smoke test "False Stability" — mock all emit 0.8 but live LLM free `[dx, dy]`. Adding `mock_strategy` param for dynamic vs fixed scroll. ~2h test infra (low ROI 当前).

**Phase 1a fire green-light reaffirmed**: A1.2 substrate post-A1.1+A1.2 batches (3-AI cross-audit). Cross-baseline validator strict (B-406/407/409/412/413); contract symmetric (B-408/410/411); cleanup landed (B-414/415/416). Phase A "永远最 clean paper grade" directive satisfied for §A1.2 scope.

---

## /stress A1.3 v9 batch — `p79/envs/` env-layer scaffold + D1 heuristic delete (2026-05-17)

A1.3 v9 third-pass audit (after §147 v8 6-fix batch B-156~B-161). Mode A + Mode B + Mode C all 3 cycles + 1 deeper round (heuristic family + scaffold completeness). User decided D1=A delete heuristic + D2 (paper §3 GRL framing) deferred to parallel session.

**3-AI cross-audit landings (initial + deeper)**:
- Initial Mode B (codex) 8 findings / 5 OOB (15135 bytes), Mode C (gemini) 7 findings + 3 OOB (8038 bytes), Mode A (Claude) 7 findings + 3 OOB.
- Deeper Mode B (codex) PARTIAL after 2 retries — numeric receipts only (`stress_a1_3_deeper_spotcheck_20260517-003217.txt` 7100 bytes); Mode C (gemini) 7159 bytes 13-item scaffold table + Option B framing; Mode A 22-item enumeration + 0/53924 heuristic empirical.
- Cross-AI overlap: 0 × 3-AI / 4 × 2-AI A+C / 1 × 2-AI A+B / 12 × 1-AI unique.

### B-425. HeuristicDomBackend family retirement — D1=A user decision 🛠️ FIXED commit `5799fda`
- **Attack**: 3-AI deeper audit confirmed 0/53924 step rows + 0/119 yaml configs + 0 paper §3 mention. ~150 LOC + 8 tests defending unreachable code paths. A1.2 B-408 enum + B-414 anchored-role fix were defensive engineering on dead code.
- **Fix**: Deleted `p79/backends/heuristic.py` + `p79/experiment/modules.py` (M1/M2/M3/M4 functions). Removed `_ALLOWED_DOM_MODES` enum + `heuristic_dom` factory branch (replaced with actionable ValueError); removed `dom_mode == heuristic_only` branches in local_qwen/api_proxy; removed local_gemma NotImplementedError raise (B-408 contract symmetry now trivial). Runner M1-M3 calls inlined as simplified baseline-retry generator (M3 module retired; baseline-retry path preserved). Tests: 2 heuristic_dom dispatch tests deleted + 1 retirement-contract test added.
- **Backward-compat**: `dom_mode` field still accepted in yaml/config (no-op); `first_element_id_by_keyword` + `extract_candidate_query` kept in action_utils.py because `_build_exploration_fallback_action` in helpers.py (anti-repeat + no-early-finish runtime controls) still uses them.
- **Paper-2 resume cost**: ~1h (git revert + re-test).

### B-418. Locator-route TYPE branch new-tab switch parity — Mode B P1-2 OOB sibling propagation of B-157 🛠️ FIXED commit `5fa579f`
- **Attack**: `vwa_wrapper.py:418-447` type+Enter branch missing the B-157 click-branch `_num_tabs_before` snapshot + new-tab detect + `bring_to_front()`. Pre-fix: search form Enter + form-submit `target=_blank` opened new tab silently; observation stayed bound to old page → state_change false-no-progress → cross-baseline taxonomy contamination.
- **Fix**: Mirror click branch tab-switch logic (snapshot before dispatch, detect, switch, stamp `_locator_route_meta["new_tab_switched"]`).

### B-419. `snapshot_form_fields()` error sentinel — Mode B P1-3 OOB 🛠️ FIXED commit `5fa579f`
- **Attack**: `vwa_wrapper.py:745-753` exception path returned bare `empty: {fields:[]}` indistinguishable from "page has no form fields". state_change.py:111-112 collapsed both cases to `unchanged` → silent navigation-race suppression.
- **Fix**: Exception path stamps `snapshot_error: "<ExcClass>: <msg>"`; success path also gets `snapshot_error: None` so downstream readers can disambiguate key-missing vs explicit-None.

### B-420. `select_option_meta` env_dispatch_meta — Mode B P1-5 OOB 🛠️ FIXED commit `5fa579f`
- **Attack**: `vwa_wrapper.py:479-619` bare `except: logger.warning + create_none_action()` swallowed JS exceptions + missing-obs cases + no-match cases. Empirical 195/738 archive rows (action_success=False, page_change=False) were taxonomy-blind.
- **Fix**: New `_select_option_meta` dict symmetric with `_locator_route_meta`, captures `{action_kind, dispatch_path: element_id|coordinate|missing_obs, success, error}`. Schema field added to `StepRecordV2` + `STEP_RECORD_V2_DEFAULTS` + `PAPER_GRADE_STEP_OPTIONAL_KEYS`. Runner persists at step_record write-out (paper §3.5 select_option sub-taxonomy now audit-able).

### B-421. `locator_route_meta` regression test — Mode A P1-8 🛠️ FIXED commit `5fa579f`
- **Attack**: `locator_route_meta` archive coverage 0/53924 (pre-B-156 data) — fix could regress without detection (no integration test asserts wire is connected).
- **Fix**: `tests/test_vwa_wrapper_telemetry.py` NEW — 3 tests monkeypatching fake `_lr_click`/`_lr_type` modules. Asserts `info["locator_route_meta"]` non-None on id-based click + type, None on scroll. No real Playwright needed.

### B-422. Injection threshold harmonize — Mode A P1-11 🛠️ FIXED commit `5fa579f`
- **Attack**: `vwa_wrapper.py:1049` CSS dropdown inline 150 px / `vwa_wrapper.py:1131` native select inline 100 px — two magic numbers same primitive, no doc trail.
- **Fix**: Named module-level constants `_INJECT_DISTANCE_CSS_DROPDOWN_PX = 150` + `_INJECT_DISTANCE_NATIVE_SELECT_PX = 100`. Docstring explains why values differ (CSS popup triggers larger; native combobox tightly bound).

### B-423. `_on_dialog` beforeunload accept policy — Mode B P2-1 OOB 🛠️ FIXED commit `5fa579f`
- **Attack**: `vwa_wrapper.py:757-771` `prompt/beforeunload: dismiss` — beforeunload dismissal = "stay on page" → silent cancel on go_back / form-submit navigation after dirty form edit. Asymmetric vs confirm/alert accept policy.
- **Fix**: Moved beforeunload from dismiss → accept (paper-grade agents WANT navigation to proceed). Prompt still dismisses (no agent-authored text input).

### B-424. Form snapshot value hash — Mode B P2-2 / closes §147 P2-B5 🛠️ FIXED commit `5fa579f`
- **Attack**: `_FORM_SNAPSHOT_JS:731` `entry.value.substring(0, 200)` truncates at 200 chars. Codex Mode B empirical: 25/7271 type actions >200 chars (max 807). Long-value suffix edits silently collapsed to "unchanged" → cycle detection breaks.
- **Fix**: JS adds `value_len` + lightweight `value_djb2` hash; state_change.py `_form_fields_changed` compares the tuple. Legacy snapshots without these keys default to None == None (no-op against archived data).

### B-417. iframe descent in locator dispatch JS resolvers ⏳ DEFER follow-up — Mode B P1-1 OOB
- **Attack**: `document.elementFromPoint` + `_inject_*` + `_FORM_SNAPSHOT_JS` all top-document only. iframe target elements fall back to bbox-center or vanish.
- **Defer rationale**: Phase 1a cls+red 0pp empirical impact (旧 sites 非 iframe-heavy); Phase 1b shop + cross-family expansion may need. 2h+ code + cross-origin regression risk → separate commit / future follow-up.
- **Tracked**: User's parallel GRL session (`docs/checkpoints/stress_grl_audit_2026-05-17.md`).

**B-numbers consumed (A1.3 v9)**: B-417~B-425 (9 contiguous; B-417 reserved-but-deferred).

**Smoke verification**:
- py_compile PASS: backends/* + envs/* + experiment/runner/main + experiment/state_change + experiment/types + experiment/schema_migrations/v2
- Tests **417/417 PASS** (was 414 pre-A1.3 v9; +4 new tests this round — 3 from B-421 telemetry regression + 1 schema validation update for B-420)

### Deferred to user's parallel GRL audit session

D2 (Paper §3 GRL framing decision A/B/C) deferred to user's separate session. User had previous advisor sync: bug fixes → future cross-benchmark workshop sub-paper (e.g. agisdk pattern). User identified additional concern: "dropdown 修复 是独立于传统 bug 修复之外的" — `[DROPDOWN OPTIONS]` injection is net-new capability not VWA bug fix. GRL completeness audit running in parallel session.

**Phase 1a fire green-light extended (post-A1.3 v9)**: substrate paper-grade after A1.1 + A1.2 + A1.3 v9 (21+8+9 = 38 fixes B-395~B-425). Heuristic family retired (paper-2 forward stub via git history). env-layer scaffold ON_TARGET telemetry (locator + select_option) now JSONL audit-able. Beforeunload policy symmetric. Form snapshot full-fidelity.

**Next available B-number**: B-426+ (consumed below by A1.19).

## A1.19 `scripts/analysis/aggregate_*.py` — pre-fire 分析管线 3-AI cycle (2026-05-17 01:00 deep night)

13-fix batch (4 P0 + 9 P1) from cross-AI Mode A + Mode B (codex stats methodologist) + Mode C (gemini prose/design-layer). Chronicle entry §172. Scope `phase1_plan.md` §A1.19. All P0 cleared as paper-§1-hero / OSF-lock blockers; 4 P1 deferred (P1-2 / P1-10 / P1-11 / P1-12 per Q&A bottom-tier auto-default).

### B-426. SE=0 floor 1.0pp 未在 prereg 声明 — Mode A+C P0 2-AI overlap 🛠️ FIXED commit `<pending>`
- **Attack**: `aggregate_phase1_prereg_gate.py:185-187` silent `ses = np.where(ses <= 0, 1.0, ses)` floor. Paper §1 PRIMARY gate verdict 实质 depends on 1.0pp const but prereg.md H1 spec 无 disclosure → reviewer "estimand drift / preregistration breach" 攻击 + OSF lock blocking.
- **Data-grounded design**: 查 archive `results/phantom_paper/meta_phantom_lift.md` 2026-05-09 archive 3 P-SoM cells SE: B0 cls 0.981pp / B0 red 1.096pp / B1 cls 0.766pp → median ≈ 0.98pp ≈ 1.0pp at N≈200-234, p≈10-22%. 1.0pp 实际 empirically-calibrated, 不是 arbitrary. Const 优于 N-aware (后者 cells 不同 floor 引入 noise).
- **Fix**: `preregistration.md §2 H1` 加 "Degenerate-cell SE floor protocol" 段, Agresti-Coull-style finite floor 命名, archive median 锚定, "no post-hoc tuning permitted post-data-lock"。`aggregate_phase1_prereg_gate.py:185-187` 加 comment cite prereg。Implementation invariant: only applies when bootstrap SE_i = 0 exactly; `n_zero_se_floored_cells` emitted in payload。

### B-427. `use_adjusted` default 走 §139.8 retired adjusted_sr — A+B+C P0 3-AI overlap 🛠️ FIXED commit `<pending>`
- **Attack**: `aggregate_cross_site.py:341` + `compare_b0_b1.py:340` 默认 `use_adjusted = not args.no_adjusted` = True → `make analysis` 默认 invoke 都走 retired adjusted_sr path; §139.8 (2026-05-16) 已 retire 整 post-hoc layer. Codex P1-6-B sibling propagation 把 Claude F2 扩到 `compare_b0_b1.py`. Gemini C2 reframe 为 "preregistration breach".
- **Fix**: 两 scripts 加 `--use-legacy-adjusted` explicit opt-in (default False), `--no-adjusted` retained as DEPRECATED no-op for Makefile backward-compat. `use_adjusted = args.use_legacy_adjusted` 直接 binding.

### B-428. `evaluate_h2_cost` double bug: margin 10→20 + K-of-N→ALL — Mode C P0 OOB framing critique 🛠️ FIXED commit `<pending>`
- **Attack (reframed)**: gemini Mode C 原 claim `aggregate_phase1_prereg_gate.py` 缺 Cost Falsification check. Verify: 该 file 是 H1-only PRIMARY gate, framing decision (R1-R5) 在 `preregistration_decision_test.py`. **真 bug 在 preregistration_decision_test.py**: (a) `evaluate_h2_cost(cost_margin_pct: float = 10.0)` L517 + CLI default L796 = 10.0% vs prereg L131-132 lock 20.0% (1.20× ratio) → margin 2× 严格于 prereg → false-falsification rate 膨胀 when median ratio 1.10-1.20×; (b) `consistent: pass_count >= transparency_K_h2` L551 (K-of-N) vs prereg L131-132 + L368 "if ANY condition violated → falsified" 严格 ALL-pass semantics → K-of-N 错把 H1/H3 transparency 移植到 H2(a) falsification.
- **Fix**: `cost_margin_pct` default 10.0 → 20.0; `transparency_K_h2` arg DeprecationWarning + ignored; `consistent` 改为 `pass_count == n_cells_total`; new fields `n_cells_falsified` + `semantics: "strict_all_pass_falsification_check"` + `prereg_anchor`。CLI `--H2-cost-margin-pct` default fixed. Smoke test confirms strict_all_pass semantics on synthetic r1_pass scenario.

### B-429. `aggregate_phantom_lift.py:664-670` mixed-universe lift — Mode B P0 OOB 🛠️ FIXED commit `<pending>`
- **Attack**: P-SoM CI 用 `u_psom` over `(in_3_psom, in_4_psom)` (L457-482), 但 exported lift `sr_4_psom - sr_3` (L664) where `sr_3` over `common = universe_5` (L470). Point estimate 跟 CI 不同 denominator → 同行 "+P-SoM lift [CI]" mathematically inconsistent. 同 bug P-prompt L667 + 6-vs-3 L670 + 6-vs-5 L673. F07 audit 2026-05-09 仅 fix CI side, point estimate 滑过.
- **Fix**: rewrite 4 lift formulas to use per-comparison universe — `sr_4_psom - sr_3_psom_only` (over u_psom), `sr_4_pprompt - sr_3_pprompt_only` (over u_pprompt), `sr_6 - sr_3_u6` (over universe_6), `sr_6 - sr_5_u6` (over universe_6). 加 `*_universe` + `*_n_universe` columns 明示 per-row estimand. Appendix exploratory only — paper §1 hero gate `aggregate_phase1_prereg_gate.py` 不受影响.

### B-430. `preregistration_decision_test.py:368,455` Python `hash()` non-deterministic — Mode B P1 OOB reproducibility 🛠️ FIXED commit `<pending>`
- **Attack**: `cell_seed = bootstrap_seed + (hash((cell_id, "h1_drop_one")) % 100000)` 用 Python built-in `hash()`. `PYTHONHASHSEED` 影响 per-cell bootstrap stream → 同 `--seed 42` 跑 2 次 produces 4.22569 vs 4.23549 (codex empirical verify during audit run). OSF artifact byte-reproduce 失败.
- **Fix**: swap `hash((cell_id, X))` → `int(hashlib.sha256(f"{cell_id}|X".encode()).hexdigest()[:8], 16) % 100000`. `hashlib` already imported at L89. Smoke retest confirms byte-reproducible (4.276338513017495 == 4.276338513017495 same-seed runs).

### B-431. `aggregate_phantom_meta.py` Hartung-Knapp-Sidik-Jonkman row missing — Mode B P1 OOB 🛠️ FIXED commit `<pending>`
- **Attack**: DL random-effects pool uses Wald z + 1.96 CI (`preregistration_decision_test.py:221-227` canonical). At k=6 (Phase 1a) DL-Wald anti-conservative per IntHout et al. 2014 (false-positive rate inflated) + Veroniki et al. 2016 (DL τ² downward-biased at k<10). Decision-grade RE inference at k≤10 should use Hartung-Knapp `t_{k-1}` on Hartung's residual-variance estimator.
- **Fix**: new `hartung_knapp_sidik_jonkman()` function in `aggregate_phantom_meta.py`; emits HKSJ section in md output table after DL-Wald table; DL-Wald row marked "legacy descriptive only — cite HKSJ for decision-grade". scipy.stats.t when available + hard-coded t-table fallback for k=2..10. Smoke on archive 3-cell: θ_RE=2.33pp, SE_HK=0.46pp, t=5.03, p_one_sided=0.019 sig (CI wider than DL due to t_crit=4.30 at k=3).

### B-432. `aggregate_failure_modes.py` "5-bucket" docstring vs 7-bucket code — Mode A+C P1 2-AI overlap 🛠️ FIXED commit `<pending>`
- **Attack**: docstring + filename + paper §5 prose 说 "5-bucket paper taxonomy" 但 PAPER_TAXONOMY 字典 has 7 keys (5 core + max-steps-other + error/noise) + `other-failure` catch-all = 8 effective. Reviewer 5-vs-7 一眼对账 catch → "撰写粗糙感", 削弱 §5 mechanism support credibility.
- **Fix**: docstring 改 "7-bucket paper-grade taxonomy (5 core + 2 catch-alls) + 1 dynamic". Code 不变 (per Q9=a recommend — 保留 7 buckets, info richer; paper §5 prose reconcile 留 codex round per Q11=C).

### B-433. `aggregate_trajectory_covariates.py:186-194` 字符串-ts lexicographic comparison fragile — Mode A P1 OOB 🛠️ FIXED commit `<pending>`
- **Attack**: `str(rev.get("wallclock_ts","")) < ep_start` 假设 ISO-8601 well-formed 双侧 + identical TZ format. Format drift (epoch float / Z-suffix / TZ format) silently flip `is_after_reset` direction → paper §4 GLMM covariate-adjusted SR estimate biased. B-389 schema 没强制 ISO-8601.
- **Fix**: new `_parse_ts()` helper with `datetime.fromisoformat()` + Z-suffix normalization; entry-guard prints stderr WARN + returns None for unparseable (covariates degrade to "no prior info" — correct fallback rather than wrong-direction flip). Replace both call sites L186-194 + L234-238.

### B-434. `lib/run_registry.py` LEGACY_MODE_ALIAS 双映射 silent merge — Mode A P1 🛠️ FIXED commit `<pending>`
- **Attack**: `phantom_dom→P-text` AND `phantom_text→P-text` 同 P-text canonical. If run_manifest.yaml 同时有 archive `phase1_phantom_dom_router_0/` (alias→P-text) + paper-grade `phantom_text_router_0/` (alias→P-text) for same (baseline, site), `get_cells(mode='P-text')` silent merge 两个 different conditions.
- **Fix**: `_all_cells_unfiltered` 加 cross-tier alias-collision detection. 每 (baseline, site, canonical_mode) 跨 grade tier 若 has BOTH `paper-grade` + `archived` AND source mode strings 不同 → `warnings.warn(RuntimeWarning)` 提示 audit manifest. LEGACY_MODE_ALIAS 保留 (archive backward-compat needed).

### B-435. `aggregate_routing_auroc.py:91-96` cond_dirs first-match silent mislabel — Mode A P1 🛠️ FIXED commit `<pending>`
- **Attack**: `mode = cond_dirs[0].name.replace(...)` first-match. Multi-condition runs (e.g., 3-mode runs with DOM+SoM+Vision conditions) silently mislabeled with only first condition's mode, AUROC 数字 attribution drift.
- **Fix**: if `len(cond_dirs) > 1` → explicit skip + stderr message asking user to re-run `analyze_confidence_calibration.py` for per-condition `cross_mode_auroc.csv`. Single-condition runs unchanged path.

### B-436. `aggregate_failure_modes.py` multi-rerun additive counting — Mode A P1 🛠️ FIXED commit `<pending>`
- **Attack**: `RUN_RE` 只 baseline+site prefix. Same (baseline, site, mode) cell 跨 multiple paper-grade run dirs (B-184 rerun cycles) → cell_totals additively counted → failure_count silently inflated 1.5-2×. `source_runs.append` tracked runs 但无 dedup gate.
- **Fix**: per-cell `seen_runs_per_cell: dict[tuple, set[str]]` guard. 已 counted 的 run 直接 skip. Run-level mark applied after row loop. Multi-run detection 末端 stderr WARN 列 cells + count + 建议 audit run_manifest.

### B-437. `aggregate_phantom_lift.py:953,965-966,975-986` 过时 "PRIMARY family — H1 Hero" wording — Mode B P1 🛠️ FIXED commit `<pending>`
- **Attack**: 文件 B-184 demoted 为 appendix exploratory but output md prose 还说 "PRIMARY family — H1 (Hero deployment claim, P-SoM)" + "Section 1/4 hook". Reviewer 看 `phantom_lift.md` 误以为是 paper §1 hero source → "estimand drift between prereg gate (P-SoM drop-one) and appendix (3→5 oracle lift)".
- **Fix**: md prose rewrite: title 改 "APPENDIX EXPLORATORY", 加 prominent ⚠️ 顶部 warning 引用 `phase1_prereg_gate.md` as canonical; "PRIMARY family" 全改 "APPENDIX legacy exploratory family"; "Section 1/4 hook" → "Appendix sensitivity, see phase1_prereg_gate.md for §1 hero".

### B-438. Superiority test δ=1.0pp + TOST δ=1.0pp 同表 cognitive conflict — Mode C P1 🛠️ FIXED commit `<pending>`
- **Attack**: `aggregate_phantom_lift.py` md output 表格同时报告 Holm-McNemar superiority p (sig ✅) + TOST equivalence p. 同 δ=1.0pp 既作 superiority 门槛 (H0: θ ≤ 0 vs H1: θ > 0) 又作 equivalence boundary (H0: |θ| ≥ 1 vs H1: |θ| < 1). 统计 reviewer cognitive conflict — superiority/equivalence 反向 hypotheses 同表混报.
- **Fix**: md output 表格前加 strong warning block 显式 disambiguate 两个 test 的 hypothesis: superiority asks "is it big enough?", equivalence asks "is it small enough?". 注明 disjoint hypotheses + reviewer 必须 reference each test's row label.

**B-numbers consumed (A1.19)**: B-426~B-438 (13 contiguous, all FIXED).

**Smoke verification (A1.19)**:
- py_compile PASS: `aggregate_phase1_prereg_gate.py / aggregate_cross_site.py / compare_b0_b1.py / preregistration_decision_test.py / aggregate_phantom_lift.py / aggregate_phantom_meta.py / aggregate_trajectory_covariates.py / aggregate_routing_auroc.py / aggregate_failure_modes.py / lib/run_registry.py`
- Reproducibility retest (B-430): same `--seed 42` produces byte-identical pooled_effect 4.276338513017495 (was 4.22569 vs 4.23549 pre-fix)
- P0-3 smoke (B-428): synthetic r1_pass scenario emits `consistent=true` with new strict ALL-pass semantics + margin=20.0pp; framing rule returns R1
- HKSJ smoke (B-431): archive 3-cell P-SoM CI [0.34, 4.33] vs DL-Wald [1.30, 3.37] — HKSJ wider as expected at k=3
- Tests **413/414 PASS** (1 pre-existing failure NOT caused by A1.19 — confirmed via git-stash test isolation; failure source is parallel GRL audit `stress_grl_audit_2026-05-17.md` adding `PAPER_GRADE_STEP_OPTIONAL_KEYS` validation without test-fixture sync)

### Deferred A1.19 P1 (per Q&A bottom-tier auto-default)
- **P1-2 DL meta SE direct storage** (`aggregate_phantom_meta.py:191-193` + `preregistration_decision_test.py:208,219`) — defer needs advisor batch + cross-script schema bump
- **P1-10 `axis_effect_size.py` hard-coded archive paths + B2 empty** — defer per advisor 2026-05-14 "mechanism 暂搁" paper-2 scope
- **P1-11 drop-one vs 3→5 lift terminology drift** (`section1_intro.md` prose) — defer codex round
- **P1-12 P-prompt baseline exclusion in prereg** — defer advisor confirm + Phase 1a data land

**A1.19 follow-up tracker**: P0-3 reframed bug discovery (`evaluate_h2_cost` margin 10→20 + K-of-N→ALL) means historical synthetic smoke runs of `preregistration_decision_test.py` pre-2026-05-17 used **2× stricter margin** → if any analysis cited "consistent: true" with margin=10%, re-run with corrected default. Negligible if no paper prose cites synthetic smoke output, but check codex round artifacts.

**Phase 1a fire green-light extended (post-A1.19)**: substrate paper-grade after A1.1+A1.2+A1.3 v9+A1.19 = 51 fixes (B-395~B-438, 5 deferred: B-417 + A1.19 P1-2/10/11/12). Analysis pipeline paper-§1-hero gate cleansed of OSF-lock blockers; appendix exploratory file (`aggregate_phantom_lift`) explicitly demoted in md prose. Remaining advisor blockers unchanged: B-262 GLM channel migration, B-130 FE/RE estimand, B-369 schema v2.2 retry, + new A1.19 advisor batch (P0-1 prereg amend SE floor protocol formal sign-off + P0-3 cost_margin amend if applicable).

**Next available B-number**: B-449+ (consumed below by A1.25 GRL Chunk 1).

---

## A1.25 GRL (Generated Runtime Layer) audit — Chunk 1 batch (2026-05-17 deep night)

User-invoked /stress on full GRL surface (P79 runtime 容错层 on top of VWA upstream `89f5af2`); 15 user-listed items + 6 net-new discovered (hover/clear/upload locator routes DEAD CODE / CSS dropdown asymmetry / min_free_vram OOM gate / get_all_tab_titles helper). 5-chunk decomposition tracker: `docs/checkpoints/stress_grl_audit_2026-05-17.md`. Chunk 1 (locator_dispatch + action routing) 3-AI cycle:
- **Mode A (Claude)**: 10 findings / 6 OOB
- **Mode B (codex, /codex-stress)**: 8 findings / 5 OOB / PASS Phase 1+2+3 (caught retry-overwrite primary telemetry deletion — `runner/main.py:1567`)
- **Mode C (gemini, /gemini-stress)**: 7 findings / 3 OOB / PASS Phase 1+2+3 (caught 94.4% hero number missing from paper §3 prose entirely)

**Unified bug list**: 25 findings (4-overlap 3-AI, 3 overlap 2-AI, 18 unique). User confirmed Q1=A scope: 全 P0 (8) + low-risk P1 (P1-6/P1-8) + Q5=B (Phase 1a launches AFTER all 4 chunks complete). B-numbers B-439-448 below.

### B-439. hover/clear/upload locator-route dispatch DEAD CODE — Claude+Gemini overlap P0 OOB 🛠️ FIXED commit `<TBD>`
- **Attack**: `dispatch_id_based_hover/clear/upload` defined in `p79/envs/locator_dispatch.py` but ZERO production callsites (grep verified). Tests imported them but production never invoked. Production hover/clear/upload action_types absent from results JSONL. Paper §3 / handoff implicitly claiming locator-route covers 5 action types was 60% paperware.
- **Fix**: Delete 3 functions + delete 5 test cases + retain `_JS_RESOLVE_UPLOAD` JS constant (substrate for future workshop sub-paper expansion); paper §3.5.2 disclose "locator-route applied to click + type only".

### B-440. Retry overwrites primary locator telemetry — codex Mode B P0-2 OOB 🛠️ FIXED commit `<TBD>`
- **Attack**: `runner/main.py:1567` retry path `next_info = retry_info` overwrote primary action's `_locator_route_meta`. Step_record only ever showed retry meta (or None). Cross-baseline B0/B1/Gemma3-VL retry-trigger rate asymmetry → biased paper §3 ON_TARGET denominator. Codex Mode B unique catch — Claude+Gemini didn't read runner/main.py.
- **Fix**: `types.py` add `locator_route_meta_primary` + `locator_route_meta_retry` to StepRecordV2 + PAPER_GRADE_STEP_OPTIONAL_KEYS; `runner/main.py:1448` snapshot primary meta BEFORE retry block; write both fields. Legacy `locator_route_meta` field retains "value at step write time" semantics for archive backward-compat. `schema_migrations/v2.py STEP_RECORD_V2_DEFAULTS` adds None defaults for both.

### B-441. `image_encode_error_step_count` never stamped — codex Mode B P0-6 OOB 🛠️ FIXED commit `<TBD>`
- **Attack**: A1.1 B-403 schema declared the field for cross-baseline (B0 proxy JPEG vs B1/B2 HF) symmetric exclusion. Runner never stamped it → downstream `aggregate_sr_fp_per_mode.py:112-116` defaulted missing → 0 → every episode looked "clean" → filter was structurally fake. Paper §3 image_meta-based cross-baseline filtering claim unsupported by data layer.
- **Fix**: `runner/main.py:2272+` (post-trajectory_incomplete stamp) add `episode_summary["image_encode_error_step_count"] = sum(...)` over step_records where `image_meta.image_encode_error is not None`.

### B-442. Vision-mode TYPE focus-click bypasses locator walk-up — Claude+Gemini overlap P0 OOB 🛠️ FIXED commit `<TBD>`
- **Attack**: `vwa_wrapper.py:376-417` used direct `page.mouse.click(px, py)` — exact B-01 bbox-pattern. DOM/SoM mode TYPE got walk-up (94.4% → >80% fix); vision-mode TYPE remained B-01-prone. `is_editable` Control+a guard at :405-413 prevented 全选变蓝 symptom but not focus-落空 root cause. Cross-mode unfair execution; paper §4 cross-mode SR comparison + phantom routing space 4-fold drop-in property contaminated.
- **Fix**: Add `dispatch_coord_based_type` in `locator_dispatch.py` (walk-up via `_JS_RESOLVE_INPUT` from pixel coord, reuses B-161 shadow DOM pierce + 6-level walk-up); `vwa_wrapper.py:376-417` route through new function; legacy direct-click + keyboard.type path retained as walk-fail fallback (preserves backward-compat on edge cases).

### B-443. INPUT type=image / reset / area / contenteditable missing — Claude P1-6 🛠️ FIXED commit `<TBD>`
- **Attack**: `locator_dispatch.py:110-112` `_JS_RESOLVE_CLICK` ARIA accept list omitted `<input type=image>` (Magento Add-to-Cart sprite), `<input type=reset>` (form reset), `<area>` (image map), `[contenteditable]` divs. Walk-fail → bbox-center fallback = silent B-33 regression on those specific clicks.
- **Fix**: Extend `_JS_RESOLVE_CLICK` accept list to include `el.type === 'image' || el.type === 'reset'`, `el.tagName === 'AREA' && el.href`, `el.isContentEditable`.

### B-444. Validator shallow nested telemetry semantics — codex Mode B P1-8 OOB 🛠️ FIXED commit `<TBD>`
- **Attack**: `validate_step_record_v2` checked key presence only. `locator_route_meta = {}` or `{"success": "false"}` (string instead of bool) passed silently → downstream denominator logic treated malformed records as falsey/truthy depending on implementation = silent pipeline corruption.
- **Fix**: `types.py validate_step_record_v2` add nested semantic checks for `locator_route_meta*` + `select_option_meta`: success must be bool-or-None, action_kind must be enum value, dict-or-None type shape enforced. Fail-loud at write boundary.

### B-445. Coord `(0, 0)` / `(0, 0.5)` framework asymmetric bug — codex Mode B P1-9 OOB 🛠️ FIXED commit `<TBD>` (submodule)
- **Attack**: `external/visualwebarena/browser_env/actions.py:651-672` `create_mouse_click_action` used `if left and top:` truthiness test. `(0, 0)` reclassified to id-based CLICK without id (downstream error); `(0, 0.5)` raised ValueError. Mode-asymmetric framework bug: vision/coord agents can legitimately emit boundary coords.
- **Fix**: Replace truthiness with explicit `is not None` comparison; both branches updated symmetrically.

### B-446. SELECT_OPTION upstream drops selected args — codex Mode B P0-7 🛠️ FIXED commit `<TBD>` (submodule)
- **Attack**: `external/visualwebarena/browser_env/actions.py:1398-1402` parsed `parsed_code[-1]["arguments"]` then discarded → upstream `execute_playwright_select_option` called `locator.select_option()` with empty defaults → chosen option never applied. Combined with P79 `_select_option_meta.success=True` recording "dispatched" not "matched" (B-420), paper §3.5 select_option sub-taxonomy completely unreliable.
- **Fix**: Extract `_so_args` + `_so_kwargs` from `parsed_code[-1]["arguments"]` / `["keywords"]` and pass to `execute_playwright_select_option(locator_code, page, pw_action_args=_so_args, pw_action_kwargs=_so_kwargs)`.

### B-447. UPLOAD upstream parser doubly broken — codex Mode B P0-8 OOB 🛠️ FIXED commit `<TBD>` (submodule)
- **Attack**: `external/visualwebarena/browser_env/actions.py:1690` `create_upload_action(pw_code=playwright_code)` missing required `text` arg → TypeError. `actions.py:1774-1776` id-based upload regex literally `r"type ?\[(\d+)\] ?..."` not `r"upload ?..."`. Both branches dead. Combined with P79 B-439 (no production dispatch wiring) = triply-dead action family.
- **Fix**: (a) playwright-code branch: extract text via `re.search(r'upload\((?:"|\')(.+?)(?:"|\')\)', playwright_code)`; (b) id-based regex anchor change `type` → `upload`; (c) error message refined.

### B-448. `aggregate_locator_route_metrics.py` new aggregator — Claude+Codex Mode B P0-1+P1-7 🛠️ FIXED commit `<TBD>`
- **Attack**: Paper §3 evidence layer for B-01/02/33 walk-up fix had NO aggregator path. `rg locator_route_meta p79/experiment/metrics.py scripts/analysis` returned nothing. Step JSONL had data but no condition-level rollup → "field-存在主义". Paper §3 ON_TARGET rate structurally unproducible from current scripts.
- **Fix**: New `scripts/analysis/aggregate_locator_route_metrics.py` reads dedup'd step JSONL, emits per (site, model, mode) counts: invoked / walk_success / walk_fail / retry_overwritten / walk_success_rate. B-440-aware (prefers `_primary` field, falls back to legacy `locator_route_meta` for archive). Paper §3.5.2 hero number table placeholder added; populated post-Phase-1a clean-rerun.

**B-numbers consumed (A1.25 Chunk 1)**: B-439~B-448 (10 contiguous, all FIXED).

**Smoke verification (A1.25 Chunk 1)**:
- py_compile PASS: locator_dispatch + vwa_wrapper + types + runner/main + schema_migrations/v2 + aggregate_locator_route_metrics + external/visualwebarena/browser_env/actions
- Tests **401/401 PASS** (8 skipped intentional; fixture update for B-440 split fields closed pre-existing failure noted in A1.19 chronicle)

**Chunks 2-4 pending**: per Q5=B (wait-fix-all) Phase 1a launches AFTER all 4 chunks audited + fixed. Tracker: `docs/checkpoints/stress_grl_audit_2026-05-17.md`.

**Next available B-number**: B-449+.

---

## /stress A1.4 SoM extraction chain unified bug batch (2026-05-17)

3-AI cross-audit (Mode A Claude + Mode B codex + Mode C gemini, scope SoM
extraction chain `som.py` + agents + mechanistic extractor + schema +
paper §3 prose). Verification status: A PASS / B PASS (B4 hallucinated test
failure noted) / C PASS (C2 framing overstated, but core finding verified).

User picks (5 P0+P1 triage Q):
- Q1 paper-1 没 rule-based router → P0-4-C* (gemini router framing) **DROPPED**
- Q2 reference images 所有 mode 都该有 (paper §3.5 line 107 已 disclose) → P0-1-B* (codex) **DROPPED**
- Q3 degraded_som 是否真的需要 → **no, delete** (B-449)
- Q4 fire timing → wait-all-fix before Phase 1a
- Q5 P0-5 cross-pipeline coherence → pre-fire 修 (B-451)

Auto-defer: P1-2-B (select_option JS contract) + P1-5-B (CSS dropdown multi-menu)
**DEFERRED to user parallel /stress A1.25 GRL session** — both touch
injection/dispatch layer where user is actively working (B-445~B-447 in VWA
submodule).

### B-449. Delete `degraded_som` schema field (overloaded 3-meaning bool) — Claude+Gemini overlap P0-2-A*+C* OOB 🛠️ FIXED commit `3a2d204`
- **Attack**: `SomResult.degraded_som` single bool encoded three semantically distinct states: (a) zero-marks vision-fallback; (b) PIL render-fail phantom-fallback; (c) phantom-mode inheriting (b) where "no image" is design intent. Paper §3.5 line 109 prose committed to "split Path A / Path B if B2 SoM degraded rate > 5%" but schema had no field to split on. Empirical 0/6471 archive fires (the bool carried zero production signal).
- **Fix**: Delete `degraded_som` from `SomResult` + `step_record.som`. Aggregator-side `mark_count == 0` derives Path A (zero-marks); PIL render-fail Path B is logged via `logger.warning` only — empirical 0/6471 means no signal lost. `analyze_reason_diagnostics.py` updated to count SoM-mode steps with `mark_count == 0`. Paper §3.5 line 109 prose rewritten to reference the new canonical signals.

### B-450. Add `select_option_meta_primary` + `select_option_meta_retry` schema fields — Codex Mode B P0-3 🛠️ FIXED commit `3a2d204`
- **Attack**: `runner/main.py:1941` writes `step_record["select_option_meta_primary"]`, but `StepRecordV2` dataclass / `STEP_RECORD_V2_DEFAULTS` / `PAPER_GRADE_STEP_OPTIONAL_KEYS` only listed `select_option_meta`. Codex grep: schema 0 / types 0 / runner 1 mention — ghost field outside canonical schema. `fill_step_defaults` did not backfill; archive readers could not produce per-step primary/retry split for paper §3.5 select_option sub-taxonomy. Asymmetric vs the locator_route_meta full-landed primary/retry pair (B-440).
- **Fix**: Add `select_option_meta_primary: Optional[Dict[str, Any]] = None` + `select_option_meta_retry: Optional[Dict[str, Any]] = None` to `StepRecordV2`; add both keys to `STEP_RECORD_V2_DEFAULTS` + `PAPER_GRADE_STEP_OPTIONAL_KEYS`. Test fixture (`test_som_and_schema.py`) updated.

### B-451. Cross-pipeline mode→prompt dispatch coherence — Claude P0-5-A* OOB 🛠️ FIXED commit `e5af0e7`
- **Attack**: B0 (`proxy_api_agent._get_system_prompts`), B1 (`qwen3vl_agent.__init__._system_prompts`), B2 (`gemma3vl_agent.__init__._system_prompts`), and mechanistic extractor (`extract_hidden_states.__init__._mode_to_prompt`) each hand-rolled the same 7-key mode → prompt dispatch dict locally. Four copies = four silent-drift surfaces. B-103 (DOM/phantom_prompt missing `Accessibility Tree:\n` prefix in mechanistic path) was caused by exactly this drift, caught only after NPZ data was already extracted. Mechanism §5 frozen per advisor §138; this is forward-investment for paper-2 unfreeze.
- **Fix**: New canonical factory `_shared_vl_utils.build_mode_prompt_dispatch_table()` returns the canonical 7-key dict. All 4 consumers replaced with single call. Test `test_b0_b1_b2_mode_dispatch_keys_identical` updated to verify canonical table directly + grep source for `_shared_build_mode_prompt_dispatch_table()` call.

### B-452. Undeclared coordinate_type stamped "normalized" for pixel inputs — Codex Mode B P1-1-B OOB 🛠️ FIXED commit `901956d`
- **Attack**: `action_utils.py:299/317/345` "auto-add coordinate_type when missing" branch stamped `coordinate_type="normalized"` for every valid positive-finite coord — including obvious pixel pairs like `[100, 200]`. env wrapper `vwa_wrapper.py:352-358` then silently auto-normalizes by viewport division, but step JSONL audit trail claimed "normalized". Cross-baseline coord-failure analysis + paper §3 error-taxonomy mislabeled.
- **Fix**: New helper `_infer_coordinate_type(coord)` returns `"pixel"` when any component > 1.0 else `"normalized"`. The 3 auto-stamp branches (click / type / select_option) now use the inferred type. Explicit declaration is preserved (no inference override). Regression test `test_undeclared_coord_infers_pixel_not_blind_normalized` covers all 3 branches + explicit-declaration preservation.

### B-453. Select_option JS dispatch success semantics — Codex Mode B P1-2-B OOB 🛠️ DEFERRED (user A1.25 GRL session)
- **Attack**: `vwa_wrapper.py:593-606/644-649/662-710` `_FUZZY_MATCH_JS` evaluate branches mostly `return;` without `{matched: bool}`. Python sets `_select_option_meta["success"] = True` after `page.evaluate()` completes. Paper §3.5 select_option dispatch sub-taxonomy uses this field as evidence layer — no-match / wrong-match / native-select-absent all logged as success.
- **Deferral rationale**: User's parallel /stress A1.25 GRL session (commits B-445~B-447 in VWA submodule) is actively working on the dispatch/injection layer. Avoiding mid-edit collision; B-453 stays open until that session lands.

### B-454. `_collect_bbox_map` bbox unit contract docstring — Claude P1-3-A OOB 🛠️ FIXED commit `901956d`
- **Attack**: `_normalize_bbox` heuristic `max(|x|) <= 1.0` → normalized→pixel scale. Production path (`obs.obs_nodes_info[*].union_bound`, pixel) bypasses `_collect_bbox_map`; legacy fallback path bbox source unit was undocumented. Future cross-benchmark integration with normalized bboxes would silently scale up.
- **Fix**: Add docstring contract to `_collect_bbox_map` declaring pixel-coordinate expectation + describing `_normalize_bbox` as defensive fallback. Doc-only; no behavior change.

### B-455. CSS dropdown injection same-eid menu overwrite — Codex Mode B P1-5-B 🛠️ DEFERRED (user A1.25 GRL session)
- **Attack**: `vwa_wrapper.py:1076-1088` `injections: dict = {}` assignment `injections[best_eid] = dd['options']` (overwrite-not-append) — multiple hidden `<ul>` menus mapped to one trigger lose all but the last. classifieds + reddit nav often has clustered hidden menus.
- **Deferral rationale**: Same as B-453 — user's parallel GRL session owns injection layer.

### B-456. `p95(empty)=0.0` opt-in strict mode for figure renderers — Gemini Mode C P1-8 OOB 🛠️ FIXED commit `901956d`
- **Attack**: `metrics.py::p95` returns 0.0 on empty valid set (B-200 legacy contract). Paper §4 "Latency P95 robustness" disclosure says "per-arm p95=0.0 indicates catastrophic empty input". But cross-arm/cross-mode aggregator (e.g. fig_latency_scatter mean(p95)) mathematically treats 0.0 as "fast" — falsely advantaging the most-failing arm in fleet average.
- **Fix**: New `strict: bool = False` keyword on `p95(values, *, strict=...)`. `strict=True` raises ValueError on empty so renderers explicitly handle "N/A". Default `strict=False` preserves legacy 0.0 contract (no caller change). Regression test `test_b456_p95_strict_mode_raises_on_empty`.

### B-457. Paper §3.5 line 105 regex anchoring prose precision — Claude P1-4 🛠️ FIXED commit `<TBD>`
- **Attack**: Paper §3.5 line 105 described `_extract_text_marks` as "keeps each line whose label matches `\[\d+\]`" (unanchored). Production regex is `^\s*\[(\d+)\]\s+\w` (anchored line-start + word-prefix), defined as `MARK_ID_DETECT_RE` in `som.py:52` per /stress A1.10 P1-2 sibling propagation. Empirical current archive: 59 == 59 (unanchored coincides with anchored on indented AXTree) but prose underspecified. Reviewer running paper-stated regex on future cross-benchmark data could get different mark counts.
- **Fix**: Paper §3.5 line 105 prose rewritten to specify "first non-whitespace token is the bracketed numeric id followed by a role label — the canonical anchored regex `^\s*\[(\d+)\]\s+\w` defined as `MARK_ID_DETECT_RE`" + explicit `[N]`-embedded-in-StaticText exclusion clause.

### B-458. `condition_map.md` missing phantom condition_ids — Gemini Mode C P1-6 OOB 🛠️ FIXED commit `<TBD>`
- **Attack**: `docs/reference/condition_map.md` hardcoded `condition_id` to 3 baseline values (`phase1_dom_router_0` / `phase1_som_router_0` / `phase1_vision_router_0`) — completely missing all 4 phantom condition_ids (paper §3 hero P-SoM/P-text/P-prompt). `diag` / `write-analysis` / `report` skills consume this as single source of truth; all phantom-mode runs silently treated as unknown by automated pipeline status aggregation.
- **Fix**: condition_map.md updated to list 7-mode universe: 3 baseline + 4 phantom (`phantom_som` / `phantom_text` / `phantom_dom` legacy alias / `phantom_prompt`). Inline disclosure paragraph added.

**B-numbers consumed (A1.4 batch)**: B-449~B-458 (10 contiguous; B-453 + B-455 DEFERRED to A1.25 GRL session, rest FIXED).

**Smoke verification (A1.4 batch)**:
- py_compile PASS: som.py + types.py + schema_migrations/v2.py + runner/main.py + analyze_reason_diagnostics.py + _shared_vl_utils.py + qwen3vl_agent.py + gemma3vl_agent.py + proxy_api_agent.py + extract_hidden_states.py + action_utils.py + metrics.py
- Tests **412/412 PASS** (8 skipped intentional; 6 deselected — `test_vwa_evaluator_b91_guard.py` excluded due to environmental SHA drift from user A1.25 GRL session, NOT related to A1.4 batch; +2 new tests from B-452 + B-456)

**Next available B-number**: B-459+ (consumed below by A1.20).

## A1.20 `scripts/analysis/figures/*.py` — figure-script pre-fire 3-AI cycle (2026-05-17 02:00 deep night)

19-fix batch (11 P0 + 8 P1) from cross-AI Mode A (Claude 10 figures) + Mode B (codex ML systems engineer, 6 complementary figures) + Mode C (gemini prose/design-layer, 5 prose+caption artifacts). Chronicle entry §173. Scope `phase1_plan.md §A1.20`. **B-numbers reservation collided with parallel GRL session A1.25 (consumed B-449~B-458); A1.20 batch renumbered B-459~B-477**. P0-7 fig1ab/fig1c full aggregator-extraction REFACTOR landed as MINIMAL PATCH (latest-glob→run_registry + N validation + strict `is True` + masked None bars); full refactor (extract to new aggregate_cascade_metrics.py + aggregate_strategy_gradient.py) deferred to dedicated session. P1-3 SE from quantile CI deferred per Q12=C with A1.19 P1-2 advisor batch.

### B-459. fig_meta_forest HKSJ schema-skew (CSV missing HKSJ cols) — Mode A+C P0 2-AI overlap 🛠️ FIXED commit `<pending>`
- **Attack**: `fig_meta_forest.py:88-95` reads `meta_phantom_lift.csv` but A1.19 B-431 HKSJ rows landed in MD output ONLY (CSV schema 18 cols 无 `se_hk / ci_lo_hk / ...`). Paper §1 forest plot 仍 render anti-conservative DL-Wald diamond as decision-grade despite Wald z+1.96 at k=6 anti-conservative per IntHout 2014. Sibling propagation gap A1.19 fix → figure layer.
- **Fix**: `aggregate_phantom_meta.py:303` add 8 HKSJ columns to CSV schema (`hk_theta_re / hk_se_hk / hk_ci_lo / hk_ci_hi / hk_t_stat / hk_t_crit / hk_df / hk_p_one_sided`); `fig_meta_forest.py:_panel_render` add HKSJ diamond (white outlined, smaller, alongside DL-Wald) with explicit caption "HKSJ decision-grade at k≤10 per IntHout 2014"; sync title + footer text.

### B-460. `fig0c_phantom_lift_bars.py:78-83` mixed-universe bar rendering — Mode A P0 OOB 🛠️ FIXED commit `<pending>`
- **Attack**: bar height `sr_psom - sr3` mixed `sr_psom` (over u_psom universe per A1.19 B-429) with `sr3` (over universe_5 baseline). CI `sr3 + ci_lo_psom_lift` used same universe mix. A1.19 B-429 only fix CSV columns, figure rendering 仍 mixed.
- **Fix**: derive `sr3_psom_universe = sr_psom - lift_psom_pp` from A1.19 B-429 corrected `lift_4psom_vs_3_pp` (= `sr_4_psom - sr_3_psom_only`). CI bars rooted at per-comparison universe baseline. Math consistent within row.

### B-461. B2 silent missing across 12 figures (sibling propagation) — Mode A+B+C P0 3-AI overlap 🛠️ FIXED commit `<pending>`
- **Attack**: Phase 1a 6-cell scope (B0/B1/B2 × cls/red) shown as 4-cell across 12/26 figures (Claude 6 + Codex 6) hard-coded `PANELS = [_panel(B0, ...), _panel(B1, ...)]` or `for baseline in ("B0", "B1")`. Paper `section1_intro.md:65` claims "evaluate on three baselines B0/B1/B2" but core figures (Oracle/Forest/Venn) only render 4 panels. "Ghost baseline" reviewer攻击.
- **Fix**: new `scripts/analysis/figures/lib/__init__.py` + `scripts/analysis/figures/lib/panels.py` shared helper `paper_grade_panels(sites=..., baselines=...)` pulls from `run_registry.BASELINES` + `scored_task_count` canonical N. 12 figures × replace hardcoded PANELS with helper call. Placeholder rendering for incomplete cells (`is_placeholder=True` panels show "pending Phase 1a" tile rather than silent skip). Layout grows automatically (2×2 → 2×3 / 3×2). Future baseline addition = 1-file change in run_registry, all 12 figures auto-update.

### B-462. `fig_meta_forest.py:41-50` HERO label drift (A1.19 B-437 propagation gap) — Mode A+C P0 2-AI overlap 🛠️ FIXED commit `<pending>`
- **Attack**: ARMS + ROLE_BADGE labels `4psom_vs_3` arm 为 "DEPLOYMENT HERO (H1, gating)" — A1.19 B-437 already demoted source aggregator's md prose 到 "APPENDIX legacy exploratory family". Figure label still claims H1 gating. Same estimand schizophrenia as A1.19 B-437 but in figure layer.
- **Fix**: ARMS role 全改 "APPENDIX_EXPLORATORY"; ROLE_BADGE 改 "APPENDIX exploratory (3→5-mode legacy lift; cf. phase1_prereg_gate.{csv,md} for H1 PRIMARY)"; HERO frame emphasis (spine 2.0pt black) 移除 (all 3 arms 同 gray-outline diamond); legacy keys (HERO/ABLATION) retained as backward-compat aliases with warn label.

### B-463. `fig0e_category_mode_heatmap.py:53` archived-only source no live producer — Mode B P0 OOB 🛠️ DEFERRED commit `<pending>` per Q3=B
- **Attack**: reads `docs/analysis/cross_sites/codex_audit_*.json` but Makefile `_aggregate` 无 live producer. Only archived copies under `docs/archive/analysis_pre_2026-05-15/cross_sites/` exist. Clean rerun on different machine = silent stale or crash. Paper §1 category claim 不可重现.
- **Defer rationale (per Q3=B 推荐)**: `fig0e_category_mode_heatmap` removed from `Makefile _figures` target (commented out L260). Paper §1 不再 cite category heatmap evidence (category sub-claim deferred until advisor confirms taxonomy + new `aggregate_category_mode.py` producer is built). Lower-risk than option (a) "build live producer + advisor confirm".

### B-464. `fig0f_overlap_stacked_bar.py:100` uniqueness inflation direction-bias — Mode B P0 OOB 🛠️ FIXED commit `<pending>`
- **Attack**: uniqueness 仅 computed over **available** modes, not declared 6-mode universe. If P-prompt/B2/missing cells absent, unique counts inflate **exactly in direction of "hidden 4th arm" structural claim** — confirmation bias direction-aligned with paper §1 hypothesis. Reviewer audit code 立刻 catch.
- **Fix**: `draw_panel` 加 `require_six_mode_complete=True` default. Incomplete cells render explicit "INCOMPLETE CELL — uniqueness inflation risk" placeholder text rather than silent compute over <6 modes. Override via `P79_FIG0F_ALLOW_INCOMPLETE=1` env var for Phase 1a inspection sensitivity. Fail-loud rather than silently bias.

### B-465. `fig1ab/fig1c` render-time aggregator anti-pattern — Mode B P0 OOB 🛠️ PARTIAL FIX commit `<pending>` (minimal patch landed, full refactor deferred)
- **Attack**: `fig1ab_cascade_diamond.py:107` + `fig1c_strategy_gradient.py:129` compute mechanism stats LIVE from step JSONL inside renderer. `fig1ab:147` P-prompt uses `sorted(RESULTS.glob(...))[-1]` latest-glob 不是 run_registry → silently pulls in-flight or archived runs. Provenance break: no frozen CSV/JSON gate, weak N check.
- **Minimal patch (this round)**: `prompt_status()` now uses `get_cells(baseline="B0", site=site, mode="P-prompt")` instead of latest-glob — single paper-grade source. Strict `success is True` via `mode_metrics` (P1-2 sibling). Full refactor (extract stats to new `aggregate_cascade_metrics.py` + `aggregate_strategy_gradient.py` aggregators, figure reads CSV) **DEFERRED** per scope-band (0.5-1d) — separate focused session.

### B-466. `section1_intro.md` "Zero image tokens" prose vs prereg §2.6 reference_image — Mode C P0 OOB 🛠️ FIXED commit `<pending>`
- **Attack**: Paper §1 prose claims "agent receives ... no image" + "no image tokens" + "no marked image". `preregistration.md §2.6`: "phantom modes preserve task-supplied reference_images to maintain task tractability". Technical inaccuracy: prose 卖点 phrase 跟 prereg lock 直接冲突.
- **Fix**: section1_intro.md `[^image-scope]` footnote 加 explicit clarification: "no image" means "no per-step marked page screenshot" (per-step encoding pipeline cost gone); task-supplied reference_image tokens preserved per prereg §2.6 + identical across all 6 modes (enter once at episode start). Inline phrasing 改 "no marked screenshot" / "no per-step screenshot encoding cost" / "no per-step marked screenshot, no extra inference modality at every browser step".

### B-467. `section1_intro.md:7` H1 estimand schizophrenia (4-mode prose vs 6-mode prereg gate) — Mode C P0 OOB 🛠️ FIXED commit `<pending>`
- **Attack**: prose cites "3.33pp / 2.56pp" hero numbers as if they are H1 PRIMARY gate, but those are **archive 4-mode universe** {DOM, SoM, Vision, P-SoM} drop-one. prereg-locked H1 PRIMARY = P-SoM drop-one over **6-mode universe** {DOM, SoM, Vision, P-text, P-prompt, P-SoM} with FE inverse-variance pooling over 6 planned cells. Estimand drift between intro hero number + prereg gate target.
- **Fix**: section1_intro.md `[^hero-estimand-scope]` footnote 加 explicit caveat — 3.33/2.56 是 archive 4-mode universe, prereg H1 是 6-mode FE pool, Phase 1a 数据 land 后 paragraph 改 cite `phase1_prereg_gate.{csv,md}` 6-mode FE result. Inline clarifying "The numbers reported here are from the **archive 4-mode universe** {DOM, SoM, Vision, P-SoM}".

### B-468. Hardcoded `234/210` expected vs canonical `scored_task_count = 224/205` — Mode A+B P1 2-AI overlap 🛠️ FIXED commit `<pending>`
- **Attack**: 4 figures (Claude fig0c/0d + Codex fig0e/0f) hardcoded `_panel(..., 234)` / `_panel(..., 210)` vs canonical `scored_task_count(site, "visualwebarena", strict=True) = 224/205` post-§139.8. fig0c L131 `int(234*0.9)=210` partial threshold vs actual 224 obs → "near-complete" note 误导.
- **Fix**: `lib/panels.py` exposes `expected_n_canonical(site)` calling `scored_task_count`. `paper_grade_panels()` 自动注入 — replaces hardcoded 234/210 in PANELS construction across 4 figures.

### B-469. `bool(record.get("success", False))` truthy in 7 figures — Mode A+B P1 2-AI overlap 🛠️ FIXED commit `<pending>`
- **Attack**: 7 figures × `bool(record.get("success", False))` (B-283 sibling propagation). String `"false"` truthy under `bool()` → SR silently inflated in figure rendering. Archive / schema drift risk.
- **Fix**: 7 figures × replace `bool(record.get(...))` → `record.get("success") is True` strict. Files: fig0c_drop_one_oracle:109 / fig0d_taskpool_jaccard:82 / fig0f_overlap_stacked_bar:80 / fig_phantom_structure_venn:80 / fig3d_cost_sr_frontier:116 / fig3a_token_cost_intra_baseline:103 (note: fig0c_phantom_lift_bars reads CSV not summary JSON, unaffected).

### B-470. SE from quantile CI in fig_meta_forest — Mode A P1 🛠️ DEFERRED per Q12=C
- **Attack**: `fig_meta_forest.py:75` `se = (CI_hi - CI_lo) / (2 * 1.96)` quantile→normal-approx SE same as A1.19 P1-2-AB Claude+Codex finding in figure layer sibling. A1.19 P1-2 deferred per Q7=C advisor batch.
- **Defer rationale**: same advisor sync ticket as A1.19 P1-2. Fix together when advisor confirms DL meta SE protocol + aggregate_phantom_lift schema bump emits `bootstrap_se_pp` column. Track as backlog with A1.19 P1-2.

### B-471. `fig0c_drop_one_oracle.py:84-87` SECTION103_LOSS hardcoded stale — Mode A+C P1 🛠️ FIXED commit `<pending>` per Q13=c
- **Attack**: drift-detection dict hardcoded B0 cls/red only with stale numbers (B0 cls=1.71 vs intro current 3.33). Drift detection mechanism 本身 stale; B1+B2 cells blind.
- **Fix per Q13=c (推荐 drop entirely)**: SECTION103_LOSS = {} empty dict. Drift detection now lives in aggregator-level `validate_run.py` + post-flight QA. Figure-internal sanity check retired (redundant with aggregator gate).

### B-472. `fig_failure_modes_per_cell.py:76` B2 baseline_order sorts as 9 — Mode A P1 🛠️ FIXED commit `<pending>`
- **Attack**: `baseline_order = {"B0": 0, "B1": 1}.get(b, 9)` — B2 sorts 9 catch-all instead of canonical 2. B2 panels scattered to end not adjacent to B0/B1.
- **Fix**: add `"B2": 2` key → `{"B0": 0, "B1": 1, "B2": 2}`. Canonical order.

### B-473. `fig_failure_modes_per_cell.py:125` footer "5-bucket" stale — Mode A P1 🛠️ FIXED commit `<pending>`
- **Attack**: footer 写 "5-bucket paper taxonomy" — A1.19 B-432 changed `aggregate_failure_modes.py` docstring to "7-bucket + 1 dynamic". Figure footer 不同步 paper §5 reviewer 对账 inconsistency.
- **Fix**: footer 改 "7-bucket paper-grade taxonomy (5 core + 2 catch-alls) + 1 dynamic — see aggregate_failure_modes.py docstring (A1.19 B-432)".

### B-474. `fig2_micro_divergence_heatmap.py:112` per-baseline color normalization → cross-baseline incomparable — Mode B P1 OOB 🛠️ FIXED commit `<pending>`
- **Attack**: `lim/fmax/repeat_lim` from each rendered baseline's finite values → identical colors mean DIFFERENT effect sizes across `fig2_*_B0.png` vs `fig2_*_B1.png`. Viewer reads cross-baseline color intensity as comparable when not. Classic per-panel vmin/vmax trap.
- **Fix**: new `compute_global_limits()` precomputes global vmin/vmax across **all baselines/sites/contrasts**. `render_baseline()` accepts `global_limits` param + uses fixed `vmin/vmax`. main() loops over `BASELINES` registry (B0+B1+B2) + writes per-baseline PNGs sharing global colormap norm. Footer disclosure "vmin/vmax now GLOBAL across baselines for cross-baseline visual comparability".

### B-475. `fig1c_strategy_gradient.py:216` None→0 bars + per-panel auto-scale (missing-as-zero rendering bias) — Mode B P1 OOB 🛠️ FIXED commit `<pending>`
- **Attack**: `heights = [0 if value is None else value for value in metric_values]` + per-panel y-auto-scale make absence look like near-zero. Reviewer can't distinguish "no effect" from "not observed" qualitatively.
- **Fix**: `draw_panel(..., row_ymax=...)` accepts metric-level y-limit (passed from main loop). None values: `bar.set_visible(False)` + explicit "n/a (not observed)" text annotation in bar slot. Fixed y-limit per metric row replaces auto-scale.

### B-476. `fig0c_drop_one_oracle.py` figure caption N denominator transparency — Mode C P1 🛠️ FIXED commit `<pending>`
- **Attack**: gemini Mode C denominator drift: intro prose N=210 vs figure n_common can drift (if listwise-deletion excludes any tasks, denominator differs). Reader 对账 prose vs figure number catches inconsistency.
- **Fix**: footer text 加 explicit "**N=common observed across all 6 modes per panel** (denominator value shown in each panel title; pp lifts are expressed as percentages of THIS panel's N_common, not against site's expected_n)". Canonical scored_task_count per site (cls=224/red=205) also noted.

### B-477. P2 backlog (5 items) — DEFERRED 🟢 backlog
- P2-1 fig0a raw vs `sr_pct_clean` (B-403 image_encode_error 排除)
- P2-2 fig0d colorbar "adjusted-success" label (FIXED via aggregator label fix during P0-3 batch — already updated to "success canonical, post-§139.8")
- P2-3 fig0g `vmin=0.4/vmax=0.9` + 0.7 threshold hardcoded
- P2-4 fig3a API$ vs electricity$ legend drift
- P2-5 fig_phantom_structure_venn caption "directly visualizes H3" vs 2-way unique mismatch

**B-numbers consumed (A1.20)**: B-459~B-477 (19 contiguous; B-463 DEFER per Q3=B; B-470 DEFER per Q12=C; remaining 17 FIXED).

**Smoke verification (A1.20)**:
- py_compile PASS: 14 figure scripts + lib/panels.py + lib/__init__.py + aggregate_phantom_meta.py
- lib/panels.py smoke: `paper_grade_panels()` returns 6 PanelSpec (3 baselines × 2 sites) with canonical N=224/205, all placeholder=True pre-Phase-1a-fire (correct)
- Tests **417/418 PASS** (1 pre-existing fail from parallel GRL session NOT A1.20-caused; confirmed via test isolation in A1.19 commit `124d3a5`; 4 new tests landed from A1.4 batch — verified independent of A1.20 fix)
- Production figures dated 2026-05-10 (pre-A1.19+A1.20) — next `make analysis` regenerates with all fixes
- Makefile `_figures` target: fig0e commented out (P0-5 defer per Q3=B); fig2 main() now writes B0+B1+B2 (was B0+B1 only)

### Deferred A1.20 P0/P1 (per Q&A bottom-tier auto-default + size scope)
- **B-463 P0-5 fig0e_category_mode_heatmap** — Q3=B defer: removed from Makefile `_figures` target. Re-enable when `aggregate_category_mode.py` live producer is built + advisor confirms category taxonomy.
- **B-465 P0-7 fig1ab/fig1c full aggregator extraction** — minimal patch landed (latest-glob → run_registry + N check + strict success); full refactor (new aggregate_cascade_metrics.py + aggregate_strategy_gradient.py + figures read CSV) deferred to dedicated session (0.5-1d scope per codex Mode B).
- **B-470 P1-3 SE from quantile CI** — Q12=C defer: bundled with A1.19 P1-2 advisor sync batch.
- **B-477 P2 (5 items)** — backlog per Q19=C.

**Phase 1a fire green-light extended (post-A1.20)**: substrate paper-grade after A1.1+A1.2+A1.3 v9+A1.4+A1.19+A1.20 + parallel GRL chunks = **68+ fixes** across multi-session 6-day audit sprint. Analysis pipeline + figure layer paper-§1-hero gate cleansed of OSF-lock blockers. Paper §1 intro prose now footnotes the "no image" semantic correction + the 4-mode-vs-6-mode estimand caveat (will be cleanly cited from `phase1_prereg_gate.{csv,md}` after Phase 1a data lands). Shared `scripts/analysis/figures/lib/panels.py` infrastructure closes B2-sibling-propagation reservoir for future baseline additions. Remaining advisor blockers: B-262 GLM channel + B-130 FE/RE estimand + B-369 schema v2.2 retry + A1.19 advisor batch (B-426 SE floor protocol + B-428 cost_margin amend) + A1.20 sibling (B-470 quantile-CI SE).

**Next available B-number**: B-478+.
