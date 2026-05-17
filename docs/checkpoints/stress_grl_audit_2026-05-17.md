---
date: 2026-05-17
status: active
audit_type: /stress pre-fire (GRL Generated Runtime Layer)
scope: P79 runtime 容错层 (15 declared items + discovery sweep for net-new)
session: stress_grl_audit_2026-05-17
related_docs:
  - docs/reference/master_bug_catalog.md (B-01~B-272 catalog)
  - docs/checkpoints/paper_drafts/section3_definition.md
  - docs/checkpoints/paper_drafts/section4_empirical_findings.md
  - docs/checkpoints/paper_drafts/section4_limitations_disclosure.md
---

# /stress GRL Generated Runtime Layer 完全审计 (2026-05-17)

## 0. Audit 框架

**Scope decision**: pre-fire (8-12 files, ≥7 findings, ≥3 OOB per chunk × 5 chunks ≈ 35-50 total)

**Driver**: user 直接 invoke /stress on GRL 15-item table, 中途 hint "潜在 bug / 潜在 GRL 没列". 拆 5 chunks 因为单 stress 上限 + discovery-first 顺序需要.

**Chunk 架构** (5 chunks):
| # | Chunk | Scope files | Status |
|---|-------|-------------|--------|
| 0 | Discovery sweep — net-new GRL + undocumented changes | grep entire `p79/` + `vwa_p79/` + commits since `89f5af2` | active |
| 1 | locator_dispatch + action routing (Items 1/2/3/4) | `p79/envs/locator_dispatch.py`, `p79/envs/vwa_wrapper.py` click/type/upload | pending |
| 2 | Observation enrichment (Items 7/8/9/14) | `p79/envs/vwa_wrapper.py:872-1041, 714-753`, `p79/experiment/som.py` | pending |
| 3 | Action policy + safety + cross-baseline (Items 10/11/12/13/15) | `p79/envs/vwa_wrapper.py:79-415, 757-771`, `p79/backends/action_utils.py:174-437`, `p79/backends/heuristic.py` | pending |
| 4 | VWA upstream patches (Items 5/6) | `external/visualwebarena/browser_env/processors.py`, `p79/envs/vwa_wrapper.py:139-184, 683` | pending |

**Per-chunk workflow** (per /stress v7.3 hard constraint):
1. Claude Mode A (scripts-first hostile read, ≥5 findings + ≥2 OOB per chunk)
2. Dispatch Mode B (`/codex-stress`) + Mode C (`/gemini-stress`) **in parallel**
3. Post-flight verification (Phase 1+2+3) on each
4. **Present unified 3-AI bug list** to user (P0/P1/P2 × 3-col actionable summary)
5. **Await user fix-scope confirmation** before any Edit/Write
6. Apply fixes per confirmed scope; verify py_compile + tests
7. Update master_bug_catalog + chronicle + this tracker
8. Move to next chunk

---

## 1. Chunk 0 — Discovery sweep ✅ COMPLETE (2026-05-17 first turn)

**6 net-new GRL items discovered** beyond user's 15-item list:

| # | New item | File:line | Severity | Chunk |
|---|----------|-----------|----------|-------|
| N1 | **`dispatch_id_based_hover()`** locator route | `p79/envs/locator_dispatch.py:273` | **HIGH** — action surface divergence | 1 |
| N2 | **`dispatch_id_based_clear()`** locator route | `p79/envs/locator_dispatch.py:310` | **HIGH** — fill('') vs clear() semantics | 1 |
| N3 | **`dispatch_id_based_upload()`** locator route | `p79/envs/locator_dispatch.py:322` | **HIGH** — set_input_files vs framework | 1 |
| N4 | **`_inject_css_dropdown_options()`** ↔ `_inject_select_options` asymmetry | `vwa_wrapper.py:872 / 975` | MEDIUM — CSS vs HTML select | 2 |
| N5 | **B-270 min_free_vram_gb=0 OOM gate** (FIXED) | `p79/backends/local_qwen.py:69` | MEDIUM — per-yaml config asymmetry | 3 |
| N6 | **`get_all_tab_titles()`** observability helper | `vwa_wrapper.py:221` | LOW — non-behavioral | 3 |

**Plus context from sweep**:
- **280+ B-### entries already in catalog** (B-01~B-310) from prior /stress cycles A1.1~A1.18 — Chunks 1-4 must NOT re-litigate already-fixed items
- **B-91 LLM judge guard** lives in VWA submodule p79-patches branch `f0c835b` — Chunk 4 must include this
- **B-340 GLM fallback deprecation** — proxy_api_agent has GLM JSON-repair, qwen3vl_agent has none → B0 vs B1 asymmetric action surface on parse error. Status: deprecated but path may still fire under `paper_grade=false` runs
- **B-133 image encoding lenient** — both agents log+count+continue (symmetric) per catalog
- **B-410 temperature/top_p warning asymmetry** — both backends log warning if yaml deviates, neither respects values (qwen3vl hardcodes `do_sample=False`) → known asymmetry, cataloged
- **B-366 SoM mark_id regex anchored fix** — already FIXED, anchored `^\s*\[(\d+)\]\s+\w`
- **B-365 6-mode canonical universe** — already FIXED in config.py

**Updated chunk scope**:
- **Chunk 1** = Items 1/2/3/4 (user) + **N1+N2+N3** (hover/clear/upload locator dispatch) — total 7 items
- **Chunk 2** = Items 7/8/9/14 (user) + **N4** (CSS dropdown ↔ select asymmetry)
- **Chunk 3** = Items 10/11/12/13/15 (user) + **N5+N6** (vram + tab titles) + **B-340 GLM asymmetry verify** (cross-baseline)
- **Chunk 4** = Items 5/6 (user) + **B-91 LLM judge guard** + **VWA submodule patch SBOM verify**

---

## 2. Chunk 1-4 baseline (pre-existing Explore agent finding 2026-05-17)

> Source: `Agent` Explore sweep at audit kickoff. Each item below was mapped to file:line + cross-baseline status + paper disclosure status. **Findings consumed into Chunk 1-4 deep dive**.

### Chunk 1 baseline (Items 1-4)

| # | Item | Primary file:line | B-xxx | Paper § | Cross-baseline |
|---|------|-------------------|-------|---------|----------------|
| 1 | Walk-up click 94.4% off-target | `p79/envs/locator_dispatch.py:1-340`, `p79/envs/vwa_wrapper.py:275-321` | B-01~B-05 | §3:123 | symmetric |
| 2 | Walk-up type + Meta+A 全选变蓝 | `p79/envs/vwa_wrapper.py:414-417, 428-452` | B-01 cluster | §3:123 | symmetric |
| 3 | Shadow DOM pierce | `p79/envs/locator_dispatch.py:56-87` + 3 resolvers | B-161 (gemini C4 OOB) | UNDISCLOSED | symmetric |
| 4 | Tab switch mirror | `p79/envs/vwa_wrapper.py:275-309` | B-157 (codex P1-B1 OOB) | UNDISCLOSED | symmetric |

### Chunk 2 baseline (Items 7, 8, 9, 14)

| # | Item | Primary file:line | B-xxx | Paper § | Cross-baseline |
|---|------|-------------------|-------|---------|----------------|
| 7 | `_inject_select_options` | `p79/envs/vwa_wrapper.py:975-1041` | (untagged) | UNDISCLOSED | symmetric |
| 8 | `_inject_css_dropdown_options` | `p79/envs/vwa_wrapper.py:872-973` | B-361 family | UNDISCLOSED | symmetric |
| 9 | `_FUZZY_MATCH_JS` 3-tier | `p79/envs/vwa_wrapper.py:29-61` + lines 510, 570 | (untagged) | UNDISCLOSED | symmetric |
| 14 | Form snapshot 200-char | `p79/envs/vwa_wrapper.py:714-743 (.substring(0,200))` | (untagged) | UNDISCLOSED | symmetric |

### Chunk 3 baseline (Items 10, 11, 12, 13, 15)

| # | Item | Primary file:line | B-xxx | Paper § | Cross-baseline |
|---|------|-------------------|-------|---------|----------------|
| 10 | `is_editable` Control+a guard | `p79/envs/vwa_wrapper.py:405-415` | B-01 cluster | §3:123 indirect | symmetric |
| 11 | Coord auto-normalize 3 sites | `p79/envs/vwa_wrapper.py:336-352` + `action_utils.py:174-220` | B-406 (15.6% B0 / 35.3% B1) | UNDISCLOSED | **B0+B1 asymmetric (2.3×)** |
| 12 | Dialog auto-accept | `p79/envs/vwa_wrapper.py:757-771` | B-158 (codex P1-B2 OOB) | §3:135 footnote | symmetric (uncertain) |
| 13 | `sleep_after_execution=0.5` | `p79/envs/vwa_wrapper.py:79, 88`; `environment.py:294` | (untagged) | UNDISCLOSED | symmetric |
| 15 | `extract_candidate_query` keyword backend | `p79/backends/action_utils.py:428-437`, `heuristic.py:33`, `runner/helpers.py:187` | (untagged, slated DELETE) | UNDISCLOSED | **B1-only** |

### Chunk 4 baseline (Items 5, 6)

| # | Item | Primary file:line | B-xxx | Paper § | Cross-baseline |
|---|------|-------------------|-------|---------|----------------|
| 5 | B-26 viewport ratio | `external/visualwebarena/browser_env/processors.py:218` (fixed) | B-26 | §4:120 | symmetric |
| 6 | B-159 asyncio + B-160 navigate JSON escape | `p79/envs/vwa_wrapper.py:139-184` + 683 | B-159 (codex P1-B3 OOB), B-160 (Claude F1-A OOB) | UNDISCLOSED | symmetric |

---

## 3. Top-level pre-finding 5 disclosure gaps (Explore agent flagged)

These will be re-attacked in detail per chunk:

1. **B-161 Shadow DOM** — Reddit redesign + web components, entirely undisclosed in paper drafts
2. **Fuzzy `select_option` 3-tier** — agents rely on label matching, undisclosed
3. **Dialog auto-accept** — Classifieds delete confirm/alert, undisclosed scope
4. **Coord auto-normalize semantics** — silent pixel↔normalized conversion, undisclosed
5. **Keyword extraction (M1/M2 heuristic)** — slated delete but still active, undisclosed fallback

## 4. Top-level pre-finding 3 cross-baseline asymmetry risks (Explore flagged)

1. **B-406 coord_type B0/B1 2.3× empirical skew** — 851 rows of normalized-declared coord >1, pre-fix asymmetric across baselines. Paper §3.5 error taxonomy + §1 hero number could be biased.
2. **M1/M2 keyword fallback B1-only** — `extract_candidate_query` invoked by B1 heuristic backend, NOT by B0 proxy_api_agent. Task success rate may be inflated for B1.
3. **Dialog auto-accept scope unknown** — only `vwa_wrapper._on_dialog` disclosed; B0/B1/Gemma3-VL invocation path uniformity unverified.

## 5. Top-level pre-finding 3 stealth modifications (Explore flagged)

1. **B-161 shadow DOM** scattered across 3 resolvers + helper, single grep doesn't reveal
2. **B-406 coord validation split** vwa_wrapper.py (permissive auto-normalize) + action_utils.py (strict per-action) different semantics
3. **Keyword extraction sibling propagation** function ~10 lines but architectural choice (B1-route-through-keyword, B0-no-keyword) only visible from grep callsites

---

## 6. Cumulative unified bug list (will fill after each chunk completes)

**Round 1 — Chunk 0 discovery**: ✅ Complete — 6 net-new GRL items folded into Chunks 1-4 scope (above §1)

**Round 2 — Chunk 1 locator_dispatch** — Mode A complete (2026-05-17), Mode B+C dispatching

### Chunk 1 Mode A findings (Claude self-review, scripts-first)

> Scope: pre-fire (8-12 files, ≥7 findings, ≥3 OOB). Artifacts read: `p79/envs/locator_dispatch.py` (358 lines full), `p79/envs/vwa_wrapper.py:1-460` (click/type/dispatch paths), `docs/reference/master_bug_catalog.md` (B-01~B-161 entries).

#### Finding A1 — Vision mode TYPE focus-click bypasses locator walk-up entirely [P0 — OOB]

**Claim** — Paper §3 (line 123, master_bug_catalog B-01 framing) says "locator-route bypass dispatches via Playwright locator + actionability check, avoids 全选变蓝 §52/§64". Implies all TYPE actions go through the walk-up.

**代码现实** — `p79/envs/vwa_wrapper.py:376-417`:
```python
elif action_type == "type" and "text" in action_json and "element_id" not in action_json:
    # Vision mode: click coordinate first to focus, then keyboard type.
    coord = action_json.get("coordinate")
    if coord is not None and isinstance(coord, (list, tuple)) and len(coord) == 2:
        # ... normalize ...
        self._env.page.mouse.click(  # ← DIRECT mouse.click, no walk-up
            left * self.viewport_width,
            top * self.viewport_height,
        )
        # ... Control+a guard (line 405-416) ...
    action = create_keyboard_type_action(action_json["text"])
```

vs `p79/envs/vwa_wrapper.py:418-458` (DOM/SoM id-based TYPE):
```python
from p79.envs.locator_dispatch import dispatch_id_based_type as _lr_type
_lr_result = _lr_type(self._env.page, self._last_obs_nodes_info, element_id, ...)
# ← walk-up via locator_dispatch
```

`p79/envs/locator_dispatch.py:121-144` `_JS_RESOLVE_INPUT` walk-up logic exists but is invoked ONLY through `dispatch_id_based_type`, which is invoked ONLY when `element_id` is present. Vision-mode action JSON has NO element_id (only coord) → walk-up never runs.

**攻击** — Paper §3 hero claim "B-01 fixed" 是 cross-mode false 的: DOM/SoM mode TYPE = walk-up ✓, vision mode TYPE = 仍走 bbox-pattern (`page.mouse.click(px, py)` 落点取决于 agent 的 coord 精度, 若 coord 落在 child span/div, focus 不进 input → `keyboard.type` 类型到错误元素 / page body). 行 405-413 的 `is_editable` guard 只 prevent **全选变蓝 visible symptom**, 没 fix **focus 落空** root cause. **跨 mode 不对称**: vision mode SR vs DOM/SoM mode SR 的 gap 里, 一部分是 B-01 在 vision mode 仍未修. Paper §3 router signal / phantom routing space 4-fold drop-in property 因 vision mode 仍是 buggy baseline 被污染.

**Defuse** — 任一: (a) 在 vision-mode TYPE 加 walk-up (复用 `_JS_RESOLVE_INPUT` with coord-from-pixel), 测试 vision mode TYPE SR 是否提升; (b) 在 paper §3 disclose "locator-route applies to id-based actions only; vision-mode coord-based TYPE retains framework dispatch + page-level focus heuristic"; (c) 重跑一组 vision mode TYPE-heavy task 量化 walk-fail 频率.

**Effort** — (a) 1-2h code + 1 cell 重跑; (b) 30min prose + footnote; (c) 半天分析

**Confidence** — high (code reading 直接证据)

#### Finding A2 — Locator walk-fail fallback = SAME B-33 buggy path (recursive-bug claim) [P0 — OOB]

**Claim** — Paper §3:123 implies "Cluster 1 locator-route 把 94.4% off-target lift to >80% ON_TARGET". Reader expects: all id-based clicks go through walk-up; if walk-up succeeds, click is on-target.

**代码现实** — `p79/envs/vwa_wrapper.py:316-321`:
```python
else:
    logger.debug("locator-route click fallback: eid=%s reason=%s",
                 eid, _lr_result.get("error", "")[:80])
    action = create_id_based_action(f"click [{eid}]")  # ← framework dispatch = B-33 buggy path
```

When `dispatch_id_based_click` returns `success=False` (walk-fail: `no_actionable_within_walk`, `obs_nodes_info missing union_bound`, etc.), the code routes through `create_id_based_action()` → VWA framework's `execute_id_based_action` → `mouse.click(union_bound_center)` = the EXACT B-33 pattern (94.4% off-target) the locator-route was meant to retire.

**攻击** — Paper §3 "lift to >80% ON_TARGET" 是 success-conditional 指标, 不是 unconditional. 假设 walk-fail rate 是 20% (尚未量化), 实际 ON_TARGET rate 在 entire population 是 `0.20 × bbox_center_rate + 0.80 × walk_up_rate ≈ 0.20 × 0.056 + 0.80 × 0.9 = 0.731` (低于 >80% 声明). 更糟: walk-fail 案例 disproportionately 是 hard-to-resolve (e.g., shadow DOM that B-161 没 cover, exotic widgets), 这些恰恰是 B-33 buggy 的 high-yield 场景, 所以 conditional rate 偏低. Paper §3 hero number 应该是 **dual rate** (locator-success ON_TARGET 91% + walk-fail bbox ON_TARGET 6% + overall 76%) 而不是单 success-conditional.

**Defuse** — 在 step_record/run_summary 加 walk-fail rate metric (utilise `_locator_route_meta.success` 已 captured at vwa_wrapper.py:296). Per-cell 报告 P(walk-up succeed) × P(on-target | succeed) + P(walk-up fail) × P(on-target | fail). Paper §3 hero number 更新为 unconditional rate.

**Effort** — 1h analysis (telemetry 已 in JSONL) + 1-2h paper §3 prose update.

**Confidence** — high

#### Finding A3 — Triple-sleep on locator-route success (latency variance inflation) [P1 — OOB]

**Claim** — Paper §4 Mode C OOB-2 attack target (memory `feedback_cross_ai_audit.md`): `sleep_after_execution=0.5` compresses latency variance. P79 prose implies single 500ms sleep per step.

**代码现实** — In locator-route success path `p79/envs/vwa_wrapper.py:289-315`:
```python
_lr_result = _lr_click(..., sleep_after_ms=int(self.sleep_after_execution * 1000))
# ← Sleep #1 (500ms inside dispatch_id_based_click line 214)
...
if _lr_result.get("success"):
    ...
    action = create_none_action()  # ← will trigger VWA framework sleep
```

After this branch, the wrapper later (line >461, beyond Read window) calls `self._env.step(action)`. VWA's `ScriptBrowserEnv.step` runs the framework's `sleep_after_execution=0.5` AGAIN inside `execute_action()` (in vwa_p79/browser_env/actions.py). So actions on locator-route success path eat **two 500ms sleeps** (locator dispatch + VWA env.step), totalling 1000ms post-click, while framework-fallback path only eats 1 × 500ms.

**攻击** — (a) Paper §4 latency comparison cross-mode is biased: DOM/SoM modes (high locator-route hit rate) consistently +500ms per successful click vs vision mode (no locator-route). The 4-fold drop-in property "latency ~50% lower than full SoM" claim may be conflated with this extra sleep. (b) Total budget per episode (50 steps × locator-success rate × 500ms extra = ~12s) is non-trivial relative to median episode latency. (c) Reviewer who probes sleep symmetry will quickly find this.

**Defuse** — Either remove the in-dispatch `sleep_after_ms` parameter from locator_dispatch.py (let VWA env.step's sleep do the work uniformly), or skip the framework sleep on `create_none_action()` path (VWA env.step on NONE could be tagged "no-sleep"). Then re-baseline latency numbers.

**Effort** — 30min code + re-aggregate latency stats (no re-run needed if step_record has per-action timestamps).

**Confidence** — medium (sleep#2 path inferred from VWA upstream architecture; should spot-check VWA actions.py to confirm)

#### Finding A4 — Coord auto-normalize silently misinterprets edge values [P1 — OOB]

**Claim** — `vwa_wrapper.py:336-352` "auto-normalize normalized↔pixel coords per viewport" with prose "Accept either normalized [0-1] or pixel coordinates".

**代码现实** — `p79/envs/vwa_wrapper.py:336-352`:
```python
if left > 1.0:
    left = left / float(self.viewport_width)
if top > 1.0:
    top = top / float(self.viewport_height)
```

Edge cases:
- **Agent outputs `left=1.001`** (normalized, near-right-edge fuzzy precision): becomes `1.001/1280 = 7.82e-4` → leftmost pixel column. Click far from intent.
- **Agent outputs `left=0.5, top=400`** (mixed format, x normalized, y pixel): becomes `(0.5, 400/720=0.556)`. Per-dim normalization "saves" the action, but no warning logged. Reviewer would not catch silent misinterpretation.
- **Pixel value `≤1.0` (e.g., `left=0`)**: stays `0`, then bumped to `eps=1e-6` (line 346) → leftmost. If agent meant pixel-0 (page-top), behavior matches. If agent meant normalized-0 (page-left), behavior matches. Coincidentally identical, but other edge values diverge.

**攻击** — B-406 already noted 2.3× B0/B1 asymmetry rate (15.6% vs 35.3%) for coord >1 normalized-declared. The auto-normalize at vwa_wrapper.py:336 is the **enabling silent mechanism**: it never fail-louds, never logs warning, so cross-baseline divergence in coord-output format remains invisible until post-hoc analysis. Per memory `feedback_cross_baseline_symmetry` retract: this is the kind of asymmetry that should be **code-aligned**, not "disclose-only".

**Defuse** — (a) Add `logger.warning("coord auto-normalize: left=%s > 1.0, treating as pixel", left)` on dim-mismatch hit so cross-baseline asymmetry surfaces in logs; (b) Per-mode strict mode (DOM/SoM `coord_type='normalized'` enforced, vision `coord_type='pixel'` enforced, fail-loud otherwise) — already partially done at `action_utils.py:174-220` (B-406 fix); make wrapper consistent by deleting the permissive auto-normalize at vwa_wrapper.py:336-352.

**Effort** — 1h code + 1 cell smoke run to verify no regression.

**Confidence** — high (code reading + B-406 catalog cross-link)

#### Finding A5 — Hardcoded 5000ms locator timeout, no prose disclosure [P1]

**Claim** — Implicit: locator-route is faster + more reliable than framework dispatch.

**代码现实** — `p79/envs/locator_dispatch.py:213, 260, 262, 299, 349`:
```python
as_element.click(timeout=5000)
as_element.fill(fill_text, timeout=5000)
as_element.press("Enter", timeout=5000)
as_element.hover(timeout=5000)
as_element.set_input_files(file_path, timeout=5000)
```

5000ms = 5s per dispatch, hardcoded. No prose disclosure. Adds tail latency on slow-rendering pages (e.g., shopping product page with deferred JS, reddit infinite-scroll mid-render).

**攻击** — Latency distribution per mode is sensitive to this constant. If a page renders in 200ms, no impact. If a page is borderline 4500ms, action succeeds within budget. If page is 5100ms, action fails → walk-fail → framework fallback (= sleep_after_execution another 500ms + framework click). So **tail latency cliff at 5000ms** is silent. Paper §4 latency CDF might show invisible cliff.

**Defuse** — (a) Expose timeout as config field with default 5000ms, document in `configs/exp_v2_phase1.yaml` + paper §4 footnote; (b) Per-action breakdown of `walk_fail_reason` (existing `_locator_route_meta.error` field) to count `TimeoutError:` walk-fails — quantifies how often the 5s cliff is hit.

**Effort** — 1-2h config plumbing + analysis aggregator extension.

**Confidence** — high

#### Finding A6 — INPUT type="image" / type="reset" fall through walk-fail [P1]

**代码现实** — `locator_dispatch.py:110-112`:
```python
if (el.tagName === 'INPUT' && (el.type === 'submit' || el.type === 'button' ||
    el.type === 'checkbox' || el.type === 'radio')) return el;
```

Misses `<input type="image">` (image submit button — common in legacy forms / e-commerce), `<input type="reset">` (form reset button). Both are actionable click targets. Walk-up falls through → walk-fail → framework bbox-center fallback = B-33 buggy.

**攻击** — Shopping site (Magento) uses `<input type="image">` for "Add to Cart" sprite buttons in some templates. Reddit uses `<button type="reset">` rarely but search-clear may render as such. Each missed actionable type → silent B-33 regression on those specific clicks. Paper §3 "94.4% → >80%" hero might mask a non-zero per-site walk-fail subset.

**Defuse** — Extend `_JS_RESOLVE_CLICK` line 110-112 to include `image`, `reset`, also any `[contenteditable]` and `<area>` (image map) for completeness.

**Effort** — 10min code + sanity smoke.

**Confidence** — high

#### Finding A7 — Hover/Clear/Upload locator routes are **DEAD CODE** (no production callsite) [P0 — OOB]

**Claim verification** — Updated post grep callsite analysis 2026-05-17.

**代码现实** — Grep result (callsites for `dispatch_id_based_hover|clear|upload`):
```
p79/envs/locator_dispatch.py  ← definitions only (lines 273, 310, 322)
tests/test_locator_dispatch.py ← test imports + invocations only (lines 14-16, 69, 75, 121-128, 136)
```
**Zero production callsites** in `p79/envs/vwa_wrapper.py` or anywhere in `p79/`. Lines 460-740 of vwa_wrapper.py contain action_type branches for `back/forward/tab/finish/select_option/wait/etc.` but NOT for `hover/clear/upload` — those action types fall through to the generic `_json_to_id_action_str` → `create_id_based_action` framework fallback at line 696-706.

**攻击** — (a) **Paper-grade disclosure trap**: if paper §3 says "P79 adds locator-route dispatch for click/type/hover/clear/upload to fix B-33 family", that's false — only click/type fire. Hover/clear/upload silently use VWA framework dispatch = potentially still buggy. (b) **User audit-scope inclusion was incorrect**: items N1/N2/N3 from discovery exist in code but inert in production → workshop sub-paper "VWA upstream bug fix" scaffold would be embarrassed if reviewer greps callsites and finds dead implementations. (c) **Test-coverage paradox**: tests cover the dead code but production never exercises it → CI green != production correctness. The architectural sin is "implemented and tested but never wired up".

**Defuse** — Decision tree:
1. Are hover/clear/upload action types emitted by any agent (B0/B1/Gemma3-VL) in Phase 1a? Grep step_record JSONL: `jq '.action.action_type' results/visualwebarena/phase1/*/step_*.jsonl | sort -u` — if absent → delete dead code + paper §3 disclose "locator-route applied to click + type only".
2. If emitted but framework-dispatched → choose: wire up locator dispatch (1-2h code per action type) OR explicitly disclose "hover/clear/upload retain framework dispatch; no B-33-class fix applied".
3. Workshop sub-paper "cross-benchmark bug fix" framing must NOT claim hover/clear/upload patches without enabling the dispatch.

**Effort** — 30min grep + decision; 1-2h per action type to wire up if needed.

**Confidence** — high (grep confirmed zero callsites)

#### Finding A9 — `select_option` vision-mode bypasses walk-up + `success` label semantically wrong [P1 — OOB]

**Claim** — `_select_option_meta.success: True` (vwa_wrapper.py:684) implies "the option was selected".

**代码现实** — `p79/envs/vwa_wrapper.py:628-688`:
```python
elif "coordinate" in action_json:
    _select_option_meta["dispatch_path"] = "coordinate"
    try:
        coord = action_json["coordinate"]
        # ... compute x_px, y_px ...
        self._env.page.evaluate(
            _FUZZY_MATCH_JS + """([x, y, label]) => {
                const el = document.elementFromPoint(x, y);  // ← NO walk-up
                if (el && el.tagName === 'SELECT') { ... }
                // ... CSS dropdown fallback ...
            }""", [x_px, y_px, option_label],
        )
        ...
        _select_option_meta["success"] = True  # ← "dispatched", not "matched"
```
And the comment from element_id branch confirms (line 619-623):
```python
# B-420: JS evaluate completed without raise (success determination is
# best-effort — actual option match outcome is silently `return;` from JS).
# Recording "dispatched" rather than "matched" since the current JS does
# not surface match-vs-no-match.
```

**攻击** — (a) `success` field is misleading: success=True means "JS evaluate didn't throw", not "option was applied". A page with no matching <select> at that coord returns silently from JS, but Python reports success=True. Paper §3.5 select_option sub-taxonomy will mis-attribute "agent dispatched correctly" cases that actually no-op'd. (b) Vision-mode walk-up bypass identical to A1 (vision-mode TYPE): `document.elementFromPoint(x, y)` first hit only, no parent walk. If coord points to child label/div, native <select> not hit → falls into CSS dropdown fallback path which may also miss. Silent no-op masquerades as success.

**Defuse** — (a) JS should return a status enum (`'matched_native' / 'matched_custom' / 'no_target' / 'no_match'`); Python reads return value into `_select_option_meta.match_outcome` field. (b) Add walk-up before `elementFromPoint` hit, mirroring `_JS_RESOLVE_CLICK`. (c) Paper §3.5 sub-taxonomy needs the 4-bucket distinction not the 2-bucket success/fail.

**Effort** — 1h JS + Python refactor + telemetry plumb.

**Confidence** — high (B-420 comment explicit about the semantic gap)

#### Finding A10 — Post-type Enter triggers **second env.step()** inside one logical step [P1]

**代码现实** — `vwa_wrapper.py:715-724`:
```python
if _type_needs_enter and self._env is not None:
    try:
        self._env.page.keyboard.press("Enter")
        self._env.page.wait_for_timeout(int(self.sleep_after_execution * 1000))
        re_obs, _, _, _, re_info = self._env.step(create_none_action())  # ← SECOND step
        obs, info = re_obs, re_info
    ...
```

When `_type_needs_enter=True` (DOM/SoM id-based type WITHOUT locator-route success path — see line 451 which suppresses `_type_needs_enter` on locator success), this triggers an Enter + 500ms wait + another env.step. Counts in metrics:
- 1 logical agent action → 2 framework env.step() calls
- Per-step `total_cost_usd` / `latency_ms` attribution: cost gets divided across logical action but framework records 2 step events
- step_record JSONL emits 2 entries with same agent action source?

**攻击** — Latency comparison cross-mode (DOM/SoM type-heavy vs vision type-heavy) is biased by how often this second-step fires. If DOM/SoM mode has X% Enter-trailing type actions, total latency is inflated by `X% × 500ms`. Paper §4 latency table needs re-attribution.

**Defuse** — Verify whether step_record emits 1 entry or 2 per `_type_needs_enter` case; if 2, fold into one with `total_latency_ms` summing both env.step latencies. If 1 (current), document the hidden +1 env.step as paper §4 footnote.

**Effort** — 30min code reading runner/main.py to confirm step_record behavior + 1 footnote.

**Confidence** — medium (existence confirmed; step_record handling needs verify)

#### Finding A8 — ARIA actionable role list 硬编码 JS const, drift risk [P2]

**代码现实** — `locator_dispatch.py:50-54` hard-codes 14 ARIA roles. ARIA 1.2 → 1.3 spec drift (e.g., new `dialog button`-style roles, `tabpanel` actionable in some pattern libraries) silently leaves walk-fails on novel sites.

**攻击** — Not Phase 1a paper-grade blocker (current sites use classical roles), but paper §1 generalization claim ("phantom routing space generalizes to other VWA-like benchmarks") could break on agisdk-style cross-benchmark workshop sub-paper (different sites = different ARIA usage).

**Defuse** — Move to YAML config + document as "currently-supported ARIA roles" in paper §4 footnote. Cross-benchmark workshop paper should explicitly test role coverage.

**Effort** — 2h refactor + paper sentence.

**Confidence** — medium (drift is real but rate uncertain)

---

### Phase 0 self-audit (Claude /stress own output)

- **Scope declared**: pre-fire (8-12 files, ≥7 findings, ≥3 OOB) ✓
- **Artifacts cited**: 3 (`locator_dispatch.py`, `vwa_wrapper.py:1-460`, `master_bug_catalog.md`) — minimum acceptable for pre-fire; vwa_wrapper.py:460-end and `paper_drafts/section3_definition.md` should have been read in Mode A round; flagged for Mode B+C complementary scope
- **Findings filed**: 8 (target 7) ✓
- **OOB count**: 5 (A1 vision-mode walk-up bypass / A2 walk-fail = recursive bug / A3 triple-sleep / A4 coord misinterpret / A7 hover-clear-upload unverified) (target 3) ✓
- **Specificity**: all findings quote file:line + code snippets ✓
- **Bilingual**: 中文 attack + English code refs ✓

⚠️ **Self-audit gap**: did NOT read paper §3 prose to compare against locator-route claim. Mode C (gemini) should explicitly verify paper §3.5 + §4.X.5 claims vs my Mode A code-reality findings — this is the cross-AI complementary value.

### Chunk 1 Mode B + C dispatched (2026-05-17 01:18:07)

- **Pre-flight smoke test**: codex `READY` ✓, gemini `READY` ✓ (both `--approval-mode plan` Pro tier)
- **Handoff file**: `docs/checkpoints/codex_prompts/grl_chunk1_handoff_2026-05-17_011807.md` (Claude's 10 findings + complementary scope assigned to each AI)
- **Mode B prompt**: `docs/checkpoints/codex_prompts/grl_chunk1_2026-05-17_011807.md` (codex = code/pipeline/reproducibility auditor; scope = runner main.py + types.py + vwa_p79 actions.py + test_locator_dispatch.py)
- **Mode C prompt**: `docs/checkpoints/gemini_prompts/grl_chunk1_2026-05-17_011807.md` (gemini = prose/design/disclosure broad-reviewer; scope = section3_definition.md + section4_empirical_findings.md + section4_limitations_disclosure.md)
- **Dispatched in parallel** (background tasks: brb7r2ndv codex + b9vxjww9q gemini); ETA 5-15 min wallclock each

### Mode C (gemini) — PASS (2026-05-17 01:21, ~3min runtime)

**Phase 1 (I/O sanity)**: 9885 bytes ✓, ends cleanly with `=== GEMINI_STRESS_AUDIT_CHUNK1_DONE ===` ✓, severity tags P0/P1/P2 ✓, required sections (Distance + leverage) ✓
**Phase 2 (depth/scope)**: 7 findings (target 7) ✓, 3 OOB (target 3) ✓, all file:line specific ✓, hostile-area-chair persona ✓, no Claude-context leak ✓
**Phase 3 (runtime)**: 2.6s (gemini-3.1-pro-preview Pro tier; sub-5min spot-check band) ✓

**Gemini findings**:
- **G1 Paperware action surface** (= Claude A7): hover/clear/upload dead code, paper §3 implies support
- **G2 Selective disclosure of B-33 recursive bug** (= Claude A2) [OOB]: walk-fail fallback to bbox-center bug, paper hides
- **G3 Vision-mode asymmetric handicap** (= Claude A1) [OOB]: cross-mode unfairness in execution layer
- **G4 Missing 94.4% → >80% data in paper** [Gemini-unique catch!]: grep `section3_definition.md` empty for that number, paper doesn't commit to hero metric
- **G5 Mirage Trap cognitive dissonance** [Gemini-unique OOB]: §3.2 Mirage attribute + walk-fail = silent model hallucination loop, undiscussed
- **G6 select_option pseudo-success semantic drift** (= Claude A9)
- **G7 External validity overfitting to VWA ARIA** [Gemini-unique OOB]: workshop sub-paper "generalizes" claim breaks on Shadow-DOM-heavy sites

### Mode B (codex) — PENDING (trace 339KB at 01:21, codex reading deep)

Awaiting harness notification. Trace log path: `docs/checkpoints/codex_outputs/grl_chunk1_trace_2026-05-17_011807.log`. Final report (atomic `-o` write): `docs/checkpoints/codex_outputs/grl_chunk1_FINAL_2026-05-17_011807.md` (not yet created).

**Next**: when codex completes, verify Phase 1+2+3 → assemble unified 3-AI bug list (Mode A 10 + Mode B N + Mode C 7) → present P0/P1/P2 × 3-col table → await user fix-scope confirmation per /stress v7.3 hard constraint.

**Round 3 — Chunk 2 observation enrichment**: Mode A pre-read complete 2026-05-17 (pipelined during Chunk 1 B+C wait); preview findings (≥4):

- **B1-prev** `_inject_css_dropdown_options` proximity-match has NO distance threshold (`vwa_wrapper.py:1052-1077`): `min(node_centers, key=...)` always returns nearest AXTree node, even if dropdown is far away. Agent gets [DROPDOWN OPTIONS] attached to wrong trigger if no nearby element. Paper §3 phantom_som assumes correct attribution.
- **B2-prev** `_FUZZY_MATCH_JS` 2-keyword threshold hardcoded (`vwa_wrapper.py:59`): `bestS >= 2`. Short labels ("Yes"/"No"/"OK") never reach 2 keywords → silent null match. Cross-benchmark workshop sub-paper "generalizes to other VWA-like benchmarks" breaks for low-keyword option sets.
- **B3-prev** `_inject_css_dropdown_options` requires 2-20 items hardcoded (`vwa_wrapper.py:1037`): single-option filters (silently excluded), 50-state dropdowns (silently excluded). No justification for bounds. Affects which sites get [DROPDOWN OPTIONS] enrichment.
- **B4-prev** `_FUZZY_MATCH_JS` stops list is English-only hardcoded (`vwa_wrapper.py:41`): `['the','a','an','to','by','of','in','on','for','and','or','is','it']`. Non-English option labels (none in current sites; would break for any future i18n test).

> Will dispatch Chunk 2 B+C **AFTER** Chunk 1 unified bug list presented + user fix-scope confirmed. Pipeline strategy: prep Mode A only, sequential B+C per /stress v7.3 hard constraint.

**Round 4 — Chunk 3 action policy**: pending

**Round 5 — Chunk 4 VWA upstream patches**: pending

**Round 3 — Chunk 2 observation enrichment**: pending

**Round 4 — Chunk 3 action policy**: pending

**Round 5 — Chunk 4 VWA upstream patches**: pending

**Total bug count**: TBD

---

## 7. Cross-link

- Chronicle §173 (Chunk 1) + §175 (Chunk 2) + §176 (Chunk 3) + §178 (Chunk 4) + §179 (synthesis, this section)
- Phase 1 plan §A1.25 GRL audit (item 8 — closed 2026-05-17 morning)
- Master bug catalog: actual claim B-439~B-448 (Chunk 1) + B-479~B-484 (Chunk 2) + B-506~B-511 (Chunk 3) + B-535~B-541 (Chunk 4) = **29 fixes** across 4 commits (`5d8fc2f` + `87874f2` + `01d45cf` + `25191a9`)
- Workshop sub-paper substrate: see §8.2 below for viability decision

---

## 8. Cross-chunk synthesis (CLOSURE 2026-05-17 ~11:30 BST)

### 8.1 Unified bug list summary — 29 fixes by severity + paper §-disclosure

| Severity | Count | Examples |
|---|---|---|
| **P0** (paper-grade emergency / launch-blocking) | 6 | B-439 hover/clear/upload dead-code retire; B-440 locator_route_meta retry-split; B-481 select_option structured fuzzy return (4-AI overlap); B-506 element_id strict reject; **B-535 LLM judge polarity inversion** (paper-grade emergency, codex Mode B F1 unique); B-536 SBOM全栈 re-lock |
| **P1** (paper-grade quality) | ~19 | B-441 walk-up extension to AREA/contenteditable; B-446/B-538 sync+async SELECT_OPTION args forward; B-447 UPLOAD parser+factory + B-539 field decouple; B-479 CSS dropdown multi-menu accumulate; B-480 select_option_meta_retry symmetric write; B-507 anti_repeat hard-block disclose; B-508 coord dual-format disclose (user reframe); B-509 dialog telemetry; B-510 runtime_sleep_ms field; B-511 option_index bounds; B-541 P79 wrapper reliability disclosure; ... |
| **P2** (defer-able to Phase 1b prep) | ~4 | DOM/Vision dispatch fallback; start-URL health-check fail-open; auto tab-switch framing; is_editable guard symmetric |

**Paper §-disclosure coverage** (post-A1.25):
- §3.5 GRL action-layer (B-445/446/447/535/538/539/540 commit fields populated)
- §3.5.1 cross-baseline asymmetry — 12+ items disclosed (incl. +6 new from A1.25: coord dual-format, dialog telemetry, runtime_sleep, anti_repeat disclose, P79 wrapper reliability, evaluator-patch policy)
- §3.5.3 **NEW** Observation enrichment surface — `[OPTIONS]` / `[DROPDOWN OPTIONS]` / fuzzy match / select_option dispatch (Chunk 2)
- §4.X.11 VWA submodule SBOM patch table — extended with 2 rows (c1765ee + 1c3a615), 8 commits total
- Catalog cross-references all updated; only Chunk 3 shared-files (runner/main.py + types.py + schema_migrations/v2.py + section3) retain draft B-485-B-490 inline labels with catalog cross-reference disclosure

### 8.2 Workshop sub-paper viability decision

**Original framing** (user kickoff): 15 GRL items split into **bucket A (workshop sub-paper)** = walk-up click 94.4% off-target family + select_option/UPLOAD dispatch fixes + shadow-DOM pierce + tab switch + viewport ratio + asyncio + navigate JSON; **bucket B (paper-1 disclosure)** = options injection + fuzzy match + is_editable + coord normalize + dialog accept + sleep + form snapshot + extract_candidate.

**Post-A1.25 viability assessment**:

| Workshop sub-paper component | Post-A1.25 status | Viability |
|---|---|---|
| **Walk-up click ON_TARGET fix family** (B-01/02/33 cluster + B-439~B-448 + B-445/446/447 + B-538/539) | ✅ Comprehensive — 8+ B-### entries, evidence layer B-440 + B-448 aggregator + paper §3.5.2 disclosure | **HIGH** — most coherent workshop story |
| **Cross-benchmark generalizability of observation enrichment** | Paper-1 §3.5.3 disclose as "standard VWA agent practice"; cross-benchmark deferred to workshop per user Chunk 2 Q1 reframing | **MEDIUM** — needs cross-benchmark data fire |
| **LLM judge polarity bug** (B-91 + B-535) | Paper-1 §3.5 evaluator-patch policy disclosure | **HIGH** — single-finding workshop "VWA evaluator bug worth republishing across VWA-derived papers" |
| **Action policy + safety primitives** (dialog telemetry, runtime_sleep, anti_repeat hard-block) | Paper-1 §3.5.1 disclosure; no separable workshop story | **LOW** — too embedded in paper-1 cross-baseline framing |
| **VWA submodule SBOM machinery** (lock files, preflight, OSF reproducibility) | Paper-1 §4.X.11; meta-infrastructure | **LOW** — process paper, not research result |

**推荐 workshop sub-paper scope** (high-viability only):
- **Track A (primary)**: GRL walk-up click ON_TARGET fix family + paper-grade evidence layer (B-440 split + B-448 aggregator) + cross-mode coverage analysis. ~3-4K word workshop paper.
- **Track B (auxiliary)**: VWA LLM judge polarity bug (B-91 + B-535) as standalone short paper / workshop note. ~1.5K word.
- **Track A + B co-submit** to same workshop OR distribute across 2 (different audiences: walk-up click is methodology-paper venue; LLM judge bug is evaluation-systems venue).

**推荐 NOT in workshop**: cross-benchmark generalizability (lacks data), action policy / safety primitives (too tangled with paper-1), SBOM machinery (process, not result).

### 8.3 Phase 1a launch trigger criteria + go/no-go

Per user Q2=Soften + Q3=(a) "continue per Q2=Soften → final synthesis then Phase 1a fire":

**Required pre-launch checks** (all must PASS):

| # | Check | Status | Action if FAIL |
|---|---|---|---|
| 1 | All A1.25 GRL Chunks 1-4 committed | ✅ PASS (4 commits) | — |
| 2 | All Chunks' P0 + launch-blocking P1 fixes landed (Q2=Soften) | ✅ PASS (6 P0 + ~13 P1 = ~19 launch-blocking lands) | — |
| 3 | py_compile + pytest all clean (zero regression) | ✅ PASS (421/421 PASS) | Debug + re-land |
| 4 | Submodule SBOM re-locked + lock files synced | ✅ PASS (B-536) | Re-sed sweep |
| 5 | Paper §3.5.3 + §3.5.1 + §4.X.11 disclosures landed | ✅ PASS (B-484/B-507/B-508/B-509/B-541/B-537) | Add prose |
| 6 | Memory `reference_vwa_submodule_p79_patches.md` updated | ✅ PASS | Re-write |
| 7 | preflight_v2.sh full execution at launch time | ⏳ NOT-RUN (runs at launch script) | Investigate failure |
| 8 | Quark Tailscale / A100 VM connectivity verified | ⏳ NOT-RUN | Restore connection |
| 9 | `make active` shows no concurrent paper-grade fire | ⏳ Should be 0 | Wait for any active to finish |

**Go/no-go assessment**: **GO** for Phase 1a launch trigger per Q2=Soften criteria. Items 7-9 are runtime checks executed by the launch script itself (e.g. `make launch BASELINE=B0 SITE=classifieds MODE=dom` runs preflight_v2.sh as first step), not pre-commit checks.

**Residual concerns (Phase 1b prep window)**:
- P2-1-C* DOM/Vision dispatch fallback asymmetry (Chunk 3) — audit-able from Phase 1a JSONL post-fire
- P2-2-B start-URL health-check fail-open (Chunk 3)
- P2-3-C* auto tab-switch framing §3.5.2 (Chunk 3)
- P1-2 observation_enrichment_meta telemetry (Chunk 2 deferred)
- P1-6 native-vs-CSS selected-state asymmetry (Chunk 2 deferred)
- P1-7 §3.5.1 dom_size prose recalibration (Chunk 2 deferred)

**Phase 1a launch invocation** (per launch_checklist.md): `make launch BASELINE=<B0|B1|B2> SITE=<classifieds|reddit> MODE=<dom|som|vision|phantom_*>` per condition; 36 conditions / 6 cells total. Recommended order: B0 first (lowest local resource footprint per CLAUDE.md hard rule "same site only one baseline at a time") → B1 → B2.

### 8.4 Cross-batch learnings — institutionalize for next /stress

**Concurrent multi-session protocol learnings (3 collision events this sprint)**:

1. **B-### reservation race** (Chunks 2→4 cumulative):
   - Chunk 2: B-478 gap left for parallel timing (no actual collision avoided)
   - Chunk 3: B-485-B-490 draft collided with parallel A1.5b's B-485-B-505 (mid-fix-apply discovery, sed-rename safe files + catalog cross-reference for shared)
   - Chunk 4: avoided collision by re-grep "Next available" IMMEDIATELY before inline tagging (no collision occurred)
   - **Institutional fix**: stress SKILL.md add "Phase 0.5: re-grep Next available B-number IMMEDIATELY BEFORE inline reference tagging; do NOT use Phase-0-recon B-numbers if any /stress is concurrently active". Or stronger: atomic claim via touch `tmp/B-### reservation` lockfile.

2. **§ chronicle race** (3 duplicate § events):
   - §173 duplicate (me A1.25 Chunk 1 + parallel A1.20)
   - §175 duplicate (me A1.25 Chunk 2 + parallel A1.5b Phase 1 Chunk 5b)
   - **Institutional fix**: same as B-### — re-grep latest § immediately before append.

3. **Codex usage limit recovery pattern** (Chunk 2 only, didn't recur):
   - Codex 5h-window quota exhausted mid-Chunk-2; auto-retry pattern via `setsid nohup` + Tier 1 file marker + ntfy → saved Chunk 2's 6 paper-grade findings
   - **Institutional fix**: stress SKILL.md add Mode B retry fallback section documenting the detached-worker pattern.

4. **User Q1 reframing pattern** (Chunk 2 + Chunk 3 + Chunk 4):
   - Chunk 2 Q1: "phantom enrichment = standard VWA practice" (downgrade P0→P1 disclosure-only)
   - Chunk 3 Q3: "coord dual-format = intentional design" (drop wrapper-as-authoritative refactor)
   - Chunk 4 Q1: "LLM judge consistent with B-91 precedent → fix + disclose" (initial Mode A wrong, user precedent argument)
   - **Institutional fix**: stress SKILL.md add Phase 0.4 "check existing P79 precedents for analogous fix" before triage — Mode A would have caught B-91 precedent for Chunk 4 without user intervention.

5. **Mode B retry success rate**: 1/1 (Chunk 2 only retry-fire; saved 6 findings). Pattern is solid.

**Mode B / Mode C OOB catch stats this sprint** (~29 fixes total):
- Mode B unique OOB catches: ~12 fixes (e.g. B-481 select_option semantic + B-509 dialog + B-510 sleep + B-511 option_index + B-535 LLM judge polarity)
- Mode C unique OOB catches: ~6 fixes (e.g. B-507 anti_repeat + B-541 wrapper reliability + Chunk 4 SBOM)
- 3-AI overlap (Mode A + B + C): ~6 fixes (high-confidence cross-validation, P0-1 Chunk 2 + P0-2 Chunk 3 + P0-2 Chunk 4)
- Single-AI unique (Mode A only): ~5 fixes (smaller P2 items)

**Cross-AI ROI = strong**: each chunk's Mode B/C OOB catches save 1-3 paper-grade unblockers Mode A solo would have missed.

---

**A1.25 GRL audit batch — CLOSED**. Ready for Phase 1a launch trigger per Q3=(a) timeline (expected fire @ 5/17 evening BST).
