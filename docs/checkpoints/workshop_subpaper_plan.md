# Workshop Sub-Paper Planning Notebook

**Status**: planning substrate (NOT paper draft); 2-track post-/stress A1.25 GRL audit closure 2026-05-17
**Origin**: synthesis §8.2 of `stress_grl_audit_2026-05-17.md` (⚠️ 该文件 2026-05-19 已删除，见 git 历史) recommended 2 HIGH-viability workshop tracks
**Parent paper-1**: VWA classifieds + reddit phantom routing space study (paper-1, EMNLP / ARR 5/25 target per memory `project_paper_hook.md`)
**Workshop relationship**: complementary, not subset — these tracks isolate GRL-layer findings that would dilute paper-1's phantom routing hero claim if folded in
**Last update**: 2026-05-17 (this document IS the first draft of the plan)

---

## 0. Two-track decision summary

| Track | Title (working) | Findings core | Target venue | Viability | Est. effort | Earliest sub. |
|---|---|---|---|---|---|---|
| **A** | Walk-up Click `ON_TARGET` — Closing the 94.4% Off-Target Gap in VisualWebArena | B-01/02/33 walk-up cluster + B-439~B-448 + B-445/446/447 + B-538/539 = ~12 fixes | Methodology / benchmarking workshop (ACL ARR Workshop / NeurIPS WebAgent / ICLR LMRL etc.) | HIGH | 3-4 weeks (post Phase 1a data lands) | 2026-06 deadline cycle |
| **B** | A Polarity Bug in the VisualWebArena LLM Judge — Implications for Cross-Paper SR Comparability | B-91 (empty-pred guard) + B-535 (substring polarity inversion) | Evaluation-systems workshop note OR short paper | HIGH | 1-2 weeks (mostly prose, narrow scope) | 2026-06 deadline cycle |

Co-submission is feasible (different audiences); recommend NOT bundling — independent narratives + independent reviewer pools.

### 0.1 Candidate additional GRL-layer findings (Protocol Reset era + post-fire /diag, 2026-05-21 / 2026-05-26)

The Protocol Reset (实验笔记 §242-251, Amendment 01) surfaced two further GRL-layer ("reliability not policy") findings; 2026-05-26 /diag (R21557 dom + R5313 som, 笔记 §299) surfaced a third. Candidate workshop material, cross-ref main-paper §3.5.1:

- **B-1794 — B0 forced-tool-call schema ≡ validator (serialization-adapter reliability)**: under `tool_choice="required"` the proxy emits a *minimal* tool call dropping optional but semantically-required fields (element_id on type/search; 235B has a competing search-via-URL prior). Fix = per-action conditional `required` mirroring `validate_action_detailed` exactly (commit `681b9cf` + 10 invariant tests). This is a **serialization-adapter reliability fix** — it makes B0's forced tool-calling conform to the same semantic gate B1/B2 prose-JSON already pass — squarely GRL "reliability not policy". **GRL boundary clarified**: schema≡validator alignment is IN-bounds (serialization adapter); obs-enrichment to disambiguate anonymous `textbox ''` inputs would be OUT-of-bounds (changes representation/task-policy + pollutes phantom-mode comparison + re-diverges from upstream).
- **Protocol Reset accounting (two-budget step counting + three-column cost)**: restores upstream "N agent decisions" budget semantics + cleanly separates canonical vs protocol-wasted spend — a reliability/measurement-layer contribution (main-paper §3.5.1 disclosure).
- **Pre-Fire-6 /stress accounting-boundary cluster (B-1796…B-1802, 2026-05-21)**: the 3-AI pre-fire audit found the accounting/serialization layer had several "computed-but-not-published / step-but-not-rolled-up" reliability gaps — episode-summary as the strict accounting boundary (`cost_unit_basis` step→episode→condition→cross_site chain, B-1798), crash-atomic billed-cost ledger (B-1800), cross-site published-artifact correctness + fail-loud on basis-less paper-grade rows (B-1801), backend-serialization provenance for B1/B2 (`action_source` + `text_parse_path`, B-1797), and the schema≡validator `select_option` VISION asymmetry (B-1796). All GRL "reliability not policy". These strengthen the same "reproducible-accounting-as-contribution" workshop angle as the two above (main-paper §3.5.1 disclosure; catalog B-1796…B-1802; chronicle §252).
- **B-1864 — cls rating widget radio input AXTree unexposed (wrapper-grounding reliability, /diag R21557 dom 2026-05-26)**: Classifieds CSS 星级评分 widget 是 `<input type="radio">` 但 P79 AXTree serialization 完全不暴露该节点 — agent 看到的 group 内只有 textbox(rating-title/comment) + Send button + `StaticText 'Rating'`, **0 个 radio 节点**, click 评分区域返回 `walk_fail:no_actionable_within_walk`。cls 234 task 中 4 task (task_95/104/133/180 = 1.7%) 跨 dom/som/vision **universal-fail**。Fix path = `p79/envs/vwa_wrapper.py::_to_p79_obs` 加 `_inject_radio_groups()`, 跟 `_inject_select_options` + `_inject_css_dropdown_options` (paper §3.5.1 line 214) 完全同 wrapper-enrichment pattern — query live page `input[type="radio"]` + 匹配 nearest AXTree 节点 + 注入 `[RADIO OPTIONS: 1-star ... 5-star]`。Serialization adapter, 不改 prompt / 不改 action policy / 不改 task → **GRL boundary in-bounds**。catalog B-1864。这条加进 §0.1 list 后, Track A 的 GRL reliability pattern cluster = schema≡validator + accounting + wrapper-enrichment (dropdown / select-options / radio-groups) 三层完整。

Whether either becomes a standalone workshop track (vs main-paper §3.5.1 disclosure only) is an advisor-sync decision; the full GRL workshop writeup stays on the Track A/B 2026-06 timeline (NOT Fire-6-blocking). Evidence scripts: `scripts/spike/probe_b0_*.py` + `tests/test_b0_schema_validator_consistency.py` + `tests/test_accounting_reset.py`.

---

## 1. Track A — Walk-up Click ON_TARGET

### 1.1 Hero claim

> P79 reduces VWA framework's 94.4% off-target click rate (on failed clicks) to a measurable `walk_success_rate` >80% (TBD post-Phase-1a) via DOM walk-up locator dispatch. The fix exposes a paper-grade evidence layer (`step_record.locator_route_meta_primary`) that cross-baseline studies can audit per (site, model, mode) cell. Without this fix, action-failure attribution mixes capability deficits with framework dispatch bugs.

### 1.2 Target venue candidates

- **Primary**: ACL ARR Workshop (e.g. Web Agents / Tool Use / VLM benchmarking) — natural home for "fix to VWA benchmark execution"
- **Secondary**: NeurIPS Workshop on Foundation Models for Decision Making — broader scope
- **Tertiary**: ICLR LMRL or Open-Source Software Track — niche but appropriate for methodology + open-source patch contribution

### 1.3 Evidence anchors

- **Pre-fix measurement**: Tier 10 sweep `scripts/maintenance/probe_tier10_dispatch_target.py` (2026-04-30) — 94.4% off-target on failed clicks
- **Post-fix code**: `p79/envs/locator_dispatch.py` walk-up resolver (anchored regex + ancestor walk for `<a>`, `<button>`, `[role=link]`, `[role=button]`, `<input type=submit/button/checkbox/radio/image/reset>`, `<summary>`, `<area>`, `[contenteditable]`, plus shadow-DOM pierce per B-161)
- **Evidence layer**: `step_record.locator_route_meta_primary` (B-440 split) + aggregator `scripts/analysis/aggregate_locator_route_metrics.py` (B-448) → walk_success_rate per cell
- **Submodule patches**: VWA `actions.py` upstream fixes B-445 (create_mouse_click_action truthiness) + B-446/B-538 (sync+async SELECT_OPTION args forward) + B-447/B-539 (UPLOAD parser/factory + field decouple)
- **Phase 1a fire data** (pending): per (site, model, mode) walk_success_rate table for paper §3.5.2 currently-empty placeholder

### 1.4 Section structure (draft, ~3500-4000 words)

1. **Introduction** (~400w): VWA upstream framework dispatches clicks via `mouse.click(union_bound_center)`; on partially-visible AXTree elements or compound `<span>` children, this misses the actionable target. Tier 10 sweep finds 94.4% off-target on failed clicks. This paper presents a walk-up locator dispatch fix + paper-grade evidence layer.
2. **Background** (~500w): VWA action dispatch surface + AXTree + Playwright `Locator` vs `mouse.click` semantics + prior work (WebArena, AgentBench, AgiSDK — do they have this bug?)
3. **Method** (~800w): Walk-up resolver design — JS-side DOM walk for actionable ancestor with allowlist of `<a> | <button> | [role=link] | [role=button] | <input type=submit/...> | <summary> | <area> | [contenteditable]` + shadow-DOM pierce. Cross-mode coverage table (DOM/SoM walk-up via element_id; vision-mode coord via `_JS_RESOLVE_INPUT`). Evidence layer schema (`locator_route_meta_primary`).
4. **Measurement** (~700w): Per (site, model, mode) walk_success_rate from Phase 1a data. Compare pre-fix archive vs post-fix archive (B-440 backward-compat allows historical baseline). Decompose action-failure attribution: framework-dispatch-bug vs capability-deficit vs label-mismatch vs DOM-mutation.
5. **Submodule upstream fixes** (~500w): VWA `actions.py` sibling fixes B-445/446/447/538/539 closing 3 upstream-contract violations (boundary-click truthiness, SELECT_OPTION args drop, UPLOAD dispatch + field encoding). Cross-paper reproducibility implication.
6. **Discussion** (~400w): Cross-benchmark generalizability (WebArena baseline, AgiSDK preview — would walk-up resolver port?). When is this fix paper-grade material (any VWA-derived paper claiming click SR > X%).
7. **Limitations** (~300w): Walk-success ≠ ON_TARGET (we lack per-step gallery labeling); reads action-policy as oracle (no preference learning interaction with walk-up). Phase 1b future scope.
8. **Conclusion** (~200w): Open-source patch contribution + evidence layer + reusable across VWA-derived studies.

### 1.5 Open questions for advisor sync

- **Q-A1**: Workshop venue priority? ARR Workshop most natural but possibly long review cycle; NeurIPS workshop deadline ~2026-08-30 if narrowly relevant.
- **Q-A2**: Should Phase 1a fire data be IN the paper (more compelling) or referenced "see paper-1" (faster submission)? Tradeoff: in-paper = wait for Phase 1a complete (~5/17 evening + analysis ~5/19); referenced = submit immediately post-§3.5.2 prose stabilization.
- **Q-A3**: Submodule fix authorship — should we upstream the patches to VWA repo via PR (would strengthen workshop "open-source contribution" angle) OR keep as P79 fork (faster, no upstream maintainer dependency)?
- **Q-A4**: Cross-benchmark scope — do we include WebArena + AgiSDK preliminary port-attempt analysis in the paper, or treat as future work?

### 1.6 Timeline (working, post-Phase-1a)

- 2026-05-17 evening: Phase 1a fire trigger
- 2026-05-19/20: Phase 1a data analysis + walk_success_rate per cell extracted
- 2026-05-22-26: Track A first draft (sections 3-5 are most code-tight; sections 1-2 + 6-8 are prose)
- 2026-05-27-30: codex round + advisor review
- 2026-06-01-05: revisions + final
- 2026-06-06+: submission

---

## 2. Track B — VWA LLM Judge Polarity Bug

### 2.1 Hero claim

> The VWA `llm_fuzzy_match` evaluator (`evaluation_harness/helper_functions.py`) contains a long-standing polarity bug: the substring check `if "correct" in response: return 1.0` accepts judge responses like `"incorrect"`, `"partially correct"`, `"not correct"` (which substring-match) all as 1.0. Combined with `llm_ua_match`'s analogous `"not the same"` substring-match-as-positive, this systematically inflates Success Rate on the VWA benchmark. Every VWA-derived paper (VisualWebArena, WebArena-Verified, PAE, Aviator-Web et al.) that uses upstream's judge logic is affected. We document the bug, present a minimal patch (invert check order, strict negative-first, no ambiguous middle), and quantify the cross-paper SR impact via re-evaluation of the published archive.

### 2.2 Target venue candidates

- **Primary**: An evaluation-systems workshop note (e.g. ICML / NeurIPS Workshop on Evaluation, or LREC if AI-evals scoped, or a dedicated benchmarks-track short paper at a major venue)
- **Secondary**: Standalone arxiv preprint + GitHub issue / PR cross-link (community-first dissemination)
- **Tertiary**: ML reproducibility workshop (MLRC, ReProRepro)

### 2.3 Evidence anchors

- **Bug discovery**: codex `/codex-stress` Mode B F1 (2026-05-17 Chunk 4) — monkeypatch verified `llm_fuzzy_match("incorrect", ...) == 1.0`, `llm_fuzzy_match("partially correct", ...) == 1.0`; `llm_ua_match("not the same", ...) == 1.0`
- **Code provenance**: bug exists in upstream VWA `89f5af2` baseline (not P79-introduced); B-91 (`f0c835b`) prior P79 patch closed empty-pred guard but NOT this substring polarity
- **Fix**: P79 submodule commit `1c3a615` (B-535) inverts check order — negative phrases FIRST (`"incorrect"` / `"partially correct"` / `"not correct"` for fuzzy; `"different"` / `"not the same"` / `"not same"` for ua), then positive, then fail-closed 0.0
- **Cross-paper impact quantification (PENDING)**: re-evaluate published archives where available — VWA paper raw outputs (not public?), WebArena-Verified (public archive on HuggingFace?), PAE (public archive?). For each, compute SR_published vs SR_post-fix-judge

### 2.4 Section structure (draft, ~1500 words)

1. **Introduction** (~250w): VWA / WebArena-family benchmarks use LLM-as-judge for fuzzy and N/A task scoring. We identify a polarity bug in the upstream judge logic that systematically inflates SR by silently scoring negative judge responses as positive.
2. **The bug** (~300w): `helper_functions.py:626-628` substring matcher. Monkeypatch demonstration. Quote 4 judge response patterns + verified scoring.
3. **Cross-paper impact** (~400w): Catalogue of papers using upstream judge logic (VWA, WebArena, WebArena-Verified, PAE, Aviator-Web, etc.). For each, estimate impact magnitude (qualitative if data unavailable; quantitative if re-evaluation possible).
4. **Fix** (~250w): Minimal patch (invert order, negative-first, strict binary, fail-closed default). Backward-compat preserved for downstream callers. P79 disclosure pattern (same precedent as B-91 empty-pred guard).
5. **Recommendation** (~200w): Upstream PR to VWA repo. Re-evaluation protocol for prior papers' archives. Disclosure pattern for evaluator patches in benchmark-derived studies.
6. **Conclusion** (~100w): Cross-paper SR comparability gap closure.

### 2.5 Open questions for advisor sync

- **Q-B1**: Should we attempt re-evaluation of public archive (WebArena-Verified?) to quantify impact, or stay qualitative ("this many papers use the buggy judge — impact is non-negligible") for first cut?
- **Q-B2**: Upstream PR to VWA — should we submit + cite ours, or P79-fork-only? Upstream PR strengthens "we contributed to community" framing.
- **Q-B3**: Track B as standalone OR appendix-to-Track-A? Pro standalone: distinct narrative + audience. Pro appendix: single-paper economy of attention.
- **Q-B4**: Scope of "polarity bug family" — should we also discuss `llm_ua_match`'s `"not the same"` case (covered by B-535 sibling fix) + any other VWA scoring sites we haven't audited?

### 2.6 Timeline

- 2026-05-17 morning: bug + fix landed (B-535 in commit `25191a9`)
- 2026-05-19: monkeypatch demonstration script (standalone reproducer)
- 2026-05-19-22: cross-paper impact quantification (if Q-B1 = yes) + qualitative catalogue (if Q-B1 = no)
- 2026-05-23-25: first draft + codex round + advisor review
- 2026-05-26-30: revisions + final
- 2026-06-01+: submission

---

## 3. Co-submission strategy

**推荐**: Independent submissions to DIFFERENT venues.
- Track A → ARR Workshop / NeurIPS WebAgent (methodology audience, larger venue traction)
- Track B → ML reproducibility workshop / ARR Workshop / arxiv-first (evaluation audience, narrow focus rewards narrow venue)

**Not推荐**: Bundle into single paper. Reasons:
- Different audiences (methodology vs evaluation systems)
- Track B is small enough to stand alone (1500w); bundling dilutes either
- Independent peer review on each isolates feedback streams

**Both can cite paper-1** (when ready): "the post-fix evaluator + walk-up resolver are used in the main paper-1 study (cite when public)".

---

## 4. Pre-submission checklist (both tracks)

- [ ] Paper-1 §3.5.2 walk_success_rate table populated post-Phase-1a (Track A blocker)
- [ ] Cross-paper SR comparability re-evaluation script (Track B, if Q-B1 = yes)
- [ ] Submodule fixes pushed upstream OR maintained as P79 fork (advisor decision Q-A3 + Q-B2)
- [ ] OSF / GitHub repo public for code+evidence-layer artifacts
- [ ] BibTeX entries prepared (paper-1 + VWA + WebArena-Verified + PAE)
- [ ] codex round + advisor review for both tracks
- [ ] Final word count + figure quota check per target venue spec

---

## 5. Cross-links

- **Audit substrate**: `stress_grl_audit_2026-05-17.md §8` (closure synthesis；⚠️ 文件 2026-05-19 已删除，见 git 历史 / [[实验笔记]])
- **Chronicle**: `docs/checkpoints/实验笔记.md §179` (audit closure entry)
- **Catalog**: `docs/reference/master_bug_catalog.md` B-439~B-448 (Chunk 1) + B-479~B-484 (Chunk 2) + B-506~B-511 (Chunk 3) + B-535~B-541 (Chunk 4); also B-01/02/33/91/156/157/158/161 walk-up cluster ancestors
- **SBOM**: `docs/checkpoints/pre_run/osf_lock_manifest.md` + `locked_versions.md` (8-commit table + sha256 lock)
- **Paper-1**: `docs/checkpoints/paper_drafts/section3_definition.md` §3.5 evaluator-patch policy + §3.5.2 locator-route dispatch + §3.5.3 observation enrichment; §4.X.11 VWA SBOM table
- **Strategy parent**: `docs/checkpoints/paper_planning.md` (paper-1 strategy; this workshop plan is forward-prep tangent)
- **Memory**: `reference_vwa_submodule_p79_patches.md` (8-commit SBOM); `feedback_cross_ai_audit.md` (audit pattern that surfaced B-535)

---

## 6. Tracking

This document is iterated **per advisor sync** (similar to `paper_planning.md` cadence). Initial creation 2026-05-17 post A1.25 closure. Next iteration trigger: advisor decision on Q-A1/A2/A3/A4 + Q-B1/B2/B3/B4.
