# B0 phantom_text classifieds — /diag digest (R31183 canonical)

**Run**: `B0_phantom_text_classifieds_20260526_233303_901232655_764510_R31183` (manifest-bound authoritative, Pass-1 paper-grade)
**Condition**: `phase1_phantom_text_router_0`
**Mode**: `phantom_text` (= P-text; **`[SOM_MARKS]` index text + DOM-prompt**; no image to agent; reference image attached if task has one). Source: `som.py:399-405` + `som.py:443-471` + paper §1 line 5. **text-mismatched arm**: prompt expects AXTree hierarchical references but obs provides `[SOM_MARKS]` interactive-only subset (regex filter over AXTree, flat 1..K list). Symmetric counterpart = P-prompt (`phantom_prompt`) = AXTree text + SoM-prompt.
**Site / Model**: classifieds / B0 (Qwen3-VL-235B-A22B via AWS Bedrock proxy)
**N episodes**: 224 / **SR = 35/224 = 15.6%**
**Ruleset version**: `4-domsomvis-b1860coord` (only dom+som+vision discovered; **ptext = 4th-mode discover product of THIS digest**)
**Diag date**: 2026-05-27

> ⚠️ **per-condition digest, NOT cross-mode comparable**: ruleset is 3-mode discover; ptext-specific patterns surfaced here will drive proposed P33-P39 → bump RULESET_VERSION to `5-...ptext-discover` → full re-scan all 4 conditions before any cross-mode quantitative claim. See [[CLAUDE.md]] "discover-then-freeze 协议".

---

## §1. Three-class stats

| Category | Count | % failed |
|---|---|---|
| **scaffold-bug** | 1 (task 9) | 0.5% |
| **agent-limit** | 188 | 99.5% |
| **benchmark-FP** | 0 | 0% |
| **unclear** | 0 | — |

> Tier-1 + Tier-2 (47 ep deep-dive) combined; remaining ~141 failed_hit episodes inherit Tier-1 categorization (mostly agent-limit, dominant rules P31/P17/P5).

**ptext_specific failures (Tier-2 sample)**: **32 / 45 (71%)** of no-hit failed are ptext-mode-unique — visual-attribute / image-grounded / row-layout / hallucination patterns that dom mode wouldn't suffer (because dom doesn't fake having visual info).

---

## §2. Tier-1 per-rule distribution (failed-only)

| Rule | Failed | Success-fire (FP risk) | Notes |
|---|---|---|---|
| **P31 budget耗尽未完成** | 64 | 1 | dominant; 29% of failed |
| **P17 click-back振荡** | 41 | 5 | 高 FP (success-fire 5 in 11 success_hit) |
| **P5 感知缺失循环** | 34 | 4 | phantom_text scroll-stuck 是正常视觉扫描 → FP-prone |
| **P19 url_match过早 finish** | 30 | 0 | clean |
| **P4 根节点误操作** | 28 | 0 | clean |
| **P7 sCity=州名** | 19 | 0 | clean |
| **P14 URL自环** | 15 | 4 | 跟 P5 同类 phantom_text FP |
| **P20 评测页未访问** | 14 | 0 | clean |
| **P18 漏排序** | 10 | 0 | clean |
| **P25 跨站任务跳过** | 10 | 0 | clean |
| **P23 oldest=价格** | 9 | 0 | clean |
| **P10 跨步记忆** | 8 | 3 | typed-text composition FP |

P-rule FP-prone in ptext: **P17 / P5 / P14 / P10 / P30 / P31** (audit results §4).
Mode-gated visual rules (P6/P15/P16/P21 `mode != "dom"`): all **0 fire on ptext** (correct gating).

---

## §3. Tier-2 主深挖 — ptext-specific 失败模式 (45 no-hit, 32 ptext-specific)

> ⚠️ **MECHANISM RE-AUDIT PENDING** (2026-05-27 fix): 本 §3 mechanism attribution 写于 line 5 P-text 定义错误前提下 (had: "AXTree + SoM-prompt"; actually `som.py:399-405` canonical = **`[SOM_MARKS]` + DOM-prompt**)。**Failure phenomenon (visual-attribute blind / image-grounded fail / gallery-row / hallucination) 仍 valid 作 episode-level observed events**, 但**机制归因** ("prompt 风格 SoM 暗示 vision" / "prompt 期待 AXTree obs 给 [SOM_MARKS]") 需 follow-up rewrite。Real mechanism direction = obs `[SOM_MARKS]` flat interactive-only subset 失去 AXTree hierarchical text context (headings / labels / descriptions), 加 DOM-prompt 指导 reference AXTree-style elements 但 obs 是 flatten subset → context 损失驱动 hallucination 更频繁 vs dom (dom obs = full AXTree)。Failure task list (§3.A-§3.E) + episode evidence valid; mechanism prose 待重写。

### A. Visual-attribute blind (color / shape / cover / thumbnail) — 16+ tasks

**Mechanism (pending rewrite per §3 audit warning above)**: 原写"phantom_text 把 prompt 设计成'使用 element_id 引用' (SoM 风格) 但 obs 是纯 AXTree 文本" — **错**, 反了 (P-text 实际是 `[SOM_MARKS]` obs + DOM-prompt). Failure phenomenon 真实: agent **明知**该判定颜色 / 形状 / 封面 / 缩略图但 obs 不含视觉字段 (P-text 无图 by construction), 退化为关键词 matching / hallucination — 该 phenomenon **不需 mechanism 修正**, 因为 dom 也无图也会 visual blind, P-text 区别在于 `[SOM_MARKS]` subset 失去 AXTree 周边 text context 进一步弱化 keyword recovery。

Sub-categories:
- **color attributes** (red / blue / green / purple / neon green / dark color):
  tasks 21, 49, 50, 56, 58, 82, 89, 117 (8 ep)
- **image content** ('whose image is' / 'has an image' / 'in the picture'):
  tasks 3, 16, 47, 96, 106, 113 (6 ep)
- **cover art** ('on the cover'):
  tasks 81 (book hurricane cover)
- **shape / appearance** ('animal shape'):
  tasks 97
- **dollar bill image content / brand on down tube / jersey number on jersey**:
  tasks 119 (dollar denomination), 120 (canyon brand → KONA hallucination), 128 (jersey number)

### B. Image-grounded task with task reference image (asymmetric image access)

**Mechanism**: `task_config.image != null` — agent receives reference image (e.g. 137-164KB JPEG of a vacuum / penguin / football / Pittsburgh) but **cannot see listing display images** in search results. Agent identifies the visual anchor correctly but fails to match it to the right listing.

Tasks: 47 (boat image), 51 (Arts+crafts painting matching reference), 59 (penguin video game cover), 66 (football cover), 67 (basketball cover), 96 (vacuum reference), 117 (blue bike reference) — 7 ep

> 💡 This is the **most ptext-specific** failure mode — symmetric image-access (task-ref AND display-img) would route to vision/som; symmetric no-image would be dom (no mismatch). ptext sits in the asymmetric gap.

### C. Gallery row layout from AXTree linear sequence

**Mechanism**: `sShowAs=gallery` URL state means visual 2D row/column layout; AXTree linearizes → "second row first item" maps to "second item in flat list" or "iPage=3" (row→page confusion).

Tasks: 14 (gallery row painting), 23 (gallery row Toyota), 41 (gallery price range range)

### D. Visual hallucination ('I am viewing the image' under no-image mode)

**Mechanism (pending rewrite per §3 audit warning)**: 原写 "SoM-style prompt that implies vision" — **错**, P-text 用 DOM-prompt 不 imply vision。Real candidate mechanism: agent 在 `[SOM_MARKS]` obs 看到 `[N] image "alt-text"` 节点 (AXTree image elements 含 alt text, regex 提取后仍保留 `image` role 标识) → 误把 alt-text + role hint 当 "I see image" 触发 → emits `thought = "I can see..."` then makes up brand/content (task 120 'KONA' for canyon-brand bike). **Phenomenon (hallucination) valid 作 episode evidence**, mechanism 待 follow-up rewrite + obs trace verify。

Tasks: 120 (KONA hallucination), partial: 47, 96 (thoughts implying viewing)

### E. Visual abandon at correct page

**Mechanism**: start_url already at correct item; early step image-click action fails (no SoM mark) → no_progress → agent navigates away from correct page → wanders and ends at wrong item.

Tasks: 128 (jersey numbers, started at item/74603, abandoned to item/41160)

### F. Non-ptext-specific (would fail in dom too) — 13/45 = 29%

- Sort-step skipped for time-based intent: task 20 (no `sOrder=dt_pub_date`)
- Wrong item selected in search results: 17, 34, 78 (filter scope / search strategy errors)
- Year mismatch rationalization: 127 ('2021-2022' close enough to '2020-2021')
- Date filter UI doesn't exist in classifieds: 26
- Form star-rating UI cross-mode failure: 95 (same task fails in som too)
- Price-related search strategy: 31, 43, 64

---

## §4. P-rule FP audit (success_hit batch, 5 ep) — **100% FP rate (5/5 hit_causal=false)**

| Task | Rules fire | Audit | True root cause |
|---|---|---|---|
| 4 | P10 | **false** | typed-text composition includes '$2021' (car year) ≠ thought numbers (price tokenizer) — `DATE_CONTEXT_RE` doesn't strip typed numbers |
| 12 | P30 | **false** | `eval_type=string_match` not `url_match`; `reference_url` is eval pointer, agent finishing elsewhere with correct answer passes |
| 52 | P5, P14 | **false** | phantom_text scroll-stuck during legitimate visual scan = normal mode behavior |
| 102 | P5, P14, P17 | **false** | same + P17 counts continuous click→scroll→finish on same URL as 3 visits |
| 151 | P17, P31 | **false** | `success=true + trajectory_incomplete=true` is legal (url_match budget exhaust passes if last page = reference); P17 9× visit = blind gallery scan w/o thumbnails |

### Failed-hit causal verify (5 ep) — **2/5 hit_causal=true, 3/5 false**

| Task | Rules | Audit | Notes |
|---|---|---|---|
| 7 | P17, P31 | **P17 true** | Color filter absent → revisit cycle on same item → budget out → P17 真 causal |
| 11 | P17 | **false** | Real cause: color perception miss; P17 count is presence-only (4 visits = 2 actual revisits + 2 scroll on same URL) |
| 21 | P5, P14 | **false** | Real cause: dark color visual blind (phantom_text); P5/P14 = surface symptoms |
| 64 | P5, P14, P19, P31 | **P19 true** | Real cause: `sPriceMin` missing → target $500 item not displayed; P19 detects properly |
| 26 | P4, P2, P31 | **false** | Real cause: classifieds has no date filter UI; P4/P2 = mid-trajectory noise |

→ **Combined FP rate**: 5/5 (success_hit) + 3/5 (failed_hit) = **8/10 audit episodes had FP rule fires** = current ruleset has substantial presence-only signals on phantom_text, similar to P14 v3 lesson in §299.

---

## §5. Representative episodes (per-class)

**scaffold-bug (rare!)**:
- **Task 9 (B-number candidate)** — listing creation form: type description success → page auto-scrolls → price INPUT (eid=20) bbox lost → 2× `obs_nodes_info missing union_bound` errors on type → agent fallback types price into description field (eid=18) → form abandoned. **Fix path**: when `union_bound` missing on type, inject pre-emptive `scroll` to bring element into viewport, re-fetch obs, retry. ptext-specific because som/vision modes annotate viewport elements visually.

**agent-limit (ptext visual blind)**:
- **Task 50** — 'red palette' makeup seller email; agent admits "not confirmed to be red" yet still finishes; selected wrong item by title-keyword match only.
- **Task 82** — '$1234 most expensive purple hard-case book'; AXTree no color/material field; agent hallucinates 'Games for the Classroom' is purple hard-case based on price ranking alone.
- **Task 120** — bike down-tube brand; agent navigates to image URL `/uploads/76122.png`, types "I am viewing the image", reports KONA (correct: canyon).

**agent-limit (non-ptext-specific)**:
- **Task 20** — 'most recently posted white Xbox'; URL never contains `sOrder=dt_pub_date`; selected default-order first result, missed actual most-recent item.
- **Task 127** — '2020-2021 Kaplan MCAT cover'; agent finds '2021-2022' edition, rationalizes "close enough", finishes.

---

## §6. 🔁 Self-evolving — proposed P33–P39 + v2 修正

### New rules (discover-then-freeze, NOT落码 yet)

| Rule | Trigger | Coverage | mode-gate |
|---|---|---|---|
| **P33** visual-attribute-task ptext-blind | `mode in {phantom_text, phantom_som}` + intent含 `(color\|on the cover\|whose image\|in the picture\|shape of\|thumbnail)` regex + finish 失败 | ~16 tasks (visual-attribute super-cluster) | phantom_text/som |
| **P34** gallery row layout AXTree blind | `start_url 含 sShowAs=gallery` + intent 含 `\b\d+(st\|nd\|rd\|th)\s+row\b` + finish 失败 | tasks 14, 23, 41 | mode-agnostic |
| **P35** scaffold: union_bound missing on type → no scroll | `action_success=False` + `locator_route_meta.error == 'obs_nodes_info missing union_bound'` + `action_type=type` + 后续 form abandoned | task 9 (potentially form-submit class) | **is_scaffold=True** (B-number candidate) |
| **P36** visual hallucination in no-image mode | `mode==phantom_text` + thought/finish 含 `(viewing the image\|I can see\|I am looking at)` | task 120 (likely more in 6-mode batch) | phantom_text |
| **P37** sort-step skipped for time-based intent | intent 含 `most recent\|latest\|newest` + obs_url 全程无 `sOrder=dt_pub_date` + `agent_finished=True` + 失败 | task 20 | mode-agnostic |
| **P38** year mismatch rationalization | intent 含 exact year pattern `YYYY-YYYY` + finish thought 含 `close enough\|as required` + finish ≠ reference | task 127 | mode-agnostic |
| **P39** visual-abandon-correct-page | `mode==phantom_text` + start_url ∈ item/* + step_0/1 action_failed/no_progress + 后续 obs_url 离开 start_url + finish 失败 | task 128 | phantom_text |

### v2 fixes for existing rules (HIGH PRIORITY — addresses 100% FP in success_hit audit)

| Rule | Fix | Justification |
|---|---|---|
| **P30 v2** | gate by `eval_types == ['url_match']`; skip on string_match | task 12 FP — string_match uses `reference_url` as eval pointer not agent finish location |
| **P31 v2** | add success-safe: `if success and trajectory_incomplete: return []` | task 151 FP — url_match budget exhaust passes if last obs_url = reference |
| **P17 v2** | rewrite `item_visits` counter: count click→back round-trips, not URL appearances. Continuous scroll/finish on same URL ≠ revisit. | tasks 11, 102, 151 — current `item_visits[id] += 1` per obs_url over-counts 2-3× |
| **P5 / P14 v2** | raise no_progress threshold for `mode==phantom_text` OR add gate skipping select_option/scroll-stuck during visual scan | tasks 52, 102 — phantom_text scroll/sort retries are normal mode behavior, not failure signal |
| **P10 v2** | skip if `action_type=type` + `action_success=True` + typed text is prose composition (full description rewrite legitimately introduces numbers ≠ thought numbers) | task 4 FP — typed '$2021' for car-year ≠ thought price numbers |

> 不立刻落码 — discover-then-freeze: ptext 是 4th-mode discover (only dom+som+vision in ruleset `4-domsomvis-b1860coord`). P-prompt + P-SoM 还未 discovered. **Once 6-mode 齐 → bump RULESET_VERSION = `5-...ptext-discover` or `6-6mode-freeze` → full re-scan all conditions → only then cross-mode quantitative compare.**

---

## §7. Actionable

### B-number candidates

- **B-1868** (proposed): scaffold: `obs_nodes_info missing union_bound` on type → no scroll recovery. **Fix path**: when `union_bound` missing on form INPUT, inject `scroll` to bring element into viewport, re-fetch obs, retry type. ptext-specific exposure (som/vision have visual fallback via image bbox). cross-ref task 9; cross-link `master_bug_catalog.md` candidate entry.

### AMENDMENT_08 candidate exclude (if benchmark-FP confirmed in 6-mode)

- 0 benchmark-FP confirmed in this 47-ep sample (5 success_hit FP audit + 5 failed_hit verify = all FP are P-rule presence, NOT eval-judge); ptext doesn't introduce new task-exclusion candidates beyond existing B-1864/B-1865/B-1866.

### Cross-mode 6-mode discover horizon

- Phase 1a chain currently running B0 phantom_som R14849 (next condition); after phantom_som + phantom_prompt → 4 P-mode discover ptext only product of THIS digest; need som-family complete + ptext + dom + som + vision full discover before bump RULESET_VERSION to `5-6mode-freeze` and rescan.

---

## §8. Cross-link

- 笔记 §302 (vision MoE 双锚反转 + codex Mode B) — Risk 6 layered noise; B0 14pp SR floor 已 disclose
- 笔记 §302.8 (N=5 cross-provider control) — provider-dependent noise; B-1867 audit gap
- 笔记 §299 (R21557 dom + R5313 som diag) — P14 v3 success-fire 15→0 fix lesson, **template for P5/P17/P30/P31 v2** here
- master_bug_catalog B-1867 (audit gap, ongoing) + B-21 / B-1864 / B-1865 / B-1866 (benchmark-FP class) — NO new exclusion candidates from ptext
- next_steps §0 ④ (Phase 1a fire chain status) — chain continues to phantom_som R14849 after R31183
- ruleset `scripts/analysis/diag_pattern_match.py` — current `4-domsomvis-b1860coord`; needs bump after 6-mode discover complete + rescan
