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
- **Fix**: 用 `locator.clear()` 替代；同 B-01 patch
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

### B-26. `current_viewport_only` 0.6 overlap operator precedence bug (§80)

- **Origin**: §80 in 实验笔记 + CLAUDE.md project knowledge
- **File**: `external/visualwebarena/browser_env/processors.py:218` (in_viewport_ratio)
- **Mechanism**: `overlap_w * overlap_h / w * h` 实际是 `((ow*oh)/w)*h` (operator precedence) — 阈值 0.6 形同虚设, 任何部分可见元素都被保留并给出**完整文本**.
- **Blast radius**: affects all DOM/SoM 模式 (Vision 不受影响 — 不靠 AXTree).
- **Status**: 🛠️ **NOT FIXED BY DESIGN** (CLAUDE.md "不修(上游代码 + 无完美阈值 + 不影响纵向对比)")
- **Paper impact**: Section 4 cite as known DOM information advantage source; explains why DOM SR > Vision SR partly artifact-driven.

---

### B-28. §50 scroll direction confusion — agent prompt limitation 🛠️ MITIGATED

- **Origin**: §50 (2026-04-14), 实验笔记 `[bug][finding]` + `B0_DOM_digest.md:98`
- **Mechanism**: 235B model 经常猜错 scroll `delta=[dx, dy]` 的方向（`dy<0` 向上 vs 向下），连续 3 次 scroll page_changed=False 触发 cycle 截断。原始 schema 暴露 `delta` 数值 → model 按自然语言理解（不一致）。
- **Mitigation (already shipped)**: §67 — Tool schema 把 `delta: [dx, dy]` 替换成 `scroll_direction: enum("up", "down")`. Mitigated but **not eliminated** (B0/B1 schema 不完全对称, B0 仍可能受影响).
- **Status**: 🛠️ **MITIGATED** (paper-disclosed limitation, B0_DOM_digest.md §6)
- **Paper impact**: Section 4 limitation table cite — B0 SR partial loss to scroll-direction confusion is acknowledged design choice, all conditions affected uniformly.
- **Note**: This is **agent prompt schema** issue, not scaffold bug. Listed for completeness (有据可查 per user instruction).

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
| B-26 §80 in_viewport_ratio | 🛠️ NO_FIX | all DOM | (upstream) | Section 4 cite | **NONE** |
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

### B-81. Myriad HPC porting — 5-class failure umbrella (笔记 §116.16, 2026-05-08)

- **Origin**: Stage 2B/2C launch wave on UCL Myriad HPC, jobs 324666 → 325178/325179
- **Status**: 🛠️ FIXED (5/5 sub-classes resolved by 2026-05-08)
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

**Reproducibility & paper §3 cite**: B-81 umbrella catalogues the canonical HPC porting checklist for HuggingFace transformer + activation-patching mechanistic workloads on RHEL 7 + SGE + module-pytorch HPC clusters. Future replicators on similar clusters (Myriad, Iridis, ARC, etc.) will hit subsets of these 5 classes; this catalog entry serves as paper §3 reproducibility-statement reference.

---

## Updated Status Counts (post-§116 audit + Phase 0 backfill)

| Tag | Count | Notes |
|---|---|---|
| ✅ **CONFIRMED** | ~28 | B-01 to B-37 mostly CONFIRMED but unfixed (Phase A audit findings); fix scope decided per evaluator_change_protocol.md Tier classification |
| 🛠️ **PARTIALLY FIXED** | 1 | B-37 (seed determinism, multi-component) |
| ⚠️ **DISPUTED** | 0 | — |
| ❌ **NOT_A_BUG** | 4 | B-12 / B-13 / B-14 / B-27 |
| 🔄 **UNVERIFIED** | 0 | All Phase A entries CONFIRMED via static read |
| 🛠️ **FIXED** | ~50 | B-10 (§105) + B-26 (NOT FIXED BY DESIGN) + B-28 (MITIGATED) + B-29 (NOT FIXED BY DESIGN) + B-38 (§116) + B-81a-e (HPC porting umbrella, 5 sub-classes) + Phase 0 historical (B-39 to B-80, including umbrella sub-entries) + Phase A patches via commits 3c15cd7 onwards |

**Pre-rerun rule reaffirmed**: All 🛠️ FIXED bugs must have their fix in code at HEAD before
16-cell rerun launch (笔记 §116 / pre_rerun_audit.md §F). UNVERIFIED entries have been
triaged to either ✅ CONFIRMED (with paper §3 disclosure) or 🛠️ FIXED. Remaining ~28 CONFIRMED
entries are documented in catalog with fix-scope decision per evaluator_change_protocol.md
Tier classification (Tier 0/1 deferred to post-rerun if not blocking).
