# Advisor-pending parking lot (parse / GLM + statistical estimand + future)

> **Purpose**: canonical parking lot for findings / fixes / disclosures that need an advisor decision before they can land. Originally scoped to the GLM-5.1 fallback question (2026-05-14, B-86); broadened 2026-05-15 to cover the Decision 3A FE vs RE estimand question (B-130) + any future blocked-on-advisor items.
>
> **Status**: 2 open threads as of 2026-05-15:
> - **Thread 1 (GLM, B-86)**: open since 2026-05-14 advisor sync (sections §1-§5 + §7 below)
> - **Thread 2 (Decision 3A FE vs RE, B-130)**: open since 2026-05-15 cross-AI Mode C audit (section §8 below)
>
> **Rule**: any future `/stress A1.x` / `/stress A2.x` / cross-AI finding that is **blocked on advisor decision** (cannot be unilaterally landed because the choice is methodology / scope / locked-decision territory) lands here under its thread, **not in the live catalog or fix queue**. Move out of parking only after the relevant advisor reply.

---

## 1. Context — the problem advisor is deciding

AWS API Gateway proxy (`execute-api.eu-west-2` → Bedrock) **does not forward `tool_choice` / structured-output flag** to the underlying model. B0 (Qwen3-VL-235B) therefore cannot be forced to emit valid JSON — its outputs are JSON-shaped but unstable, and a measurable fraction of step responses fail to parse cleanly.

The current B0 path patches this with an **auxiliary GLM-5.1 extraction call** (`proxy_api_agent.py::_call_glm_extract`, enabled by `use_glm_fallback: true` in B0 configs). When `parse_action_text` fails on raw B0 output, the runtime ships the raw text to GLM-5.1 which extracts a structured action JSON, and that JSON becomes the step's action.

**Why this is paper-grade contaminating**:
- B0 reported success rate mixes "235B got it right" with "235B failed to format + GLM-5.1 patched it".
- The verbalized `confidence` field on a GLM-rescued step comes from GLM-5.1, not B0 — so any routing or analysis feature derived from confidence is contaminated.
- B1 (Qwen3-VL-4B) and B2 (Gemma3-VL) are local HF models with `do_sample=False` greedy decoding; they never invoke GLM. Cross-baseline comparison therefore mixes two different "decoding pathways".

Advisor 2026-05-14 sync question: **can the proxy API path be replaced with one that returns clean structured JSON natively, so the GLM rescue can be retired?**

---

## 2. Empirical data (advisor decision input)

Measured across all archived Phase 1 B0/B1 runs in `results/visualwebarena/phase1/`:

| Metric | Value | Source |
|---|---|---|
| B0 GLM rescue trigger rate | **1.488 %** (453 / 30 437 steps) | T4b grep |
| B0 raw `parse_failed` (pre-rescue) | 0.309 % (94 / 30 437) | T4b grep |
| B0 post-rescue `parse_failed` (final invalid) | 0.003 % (1 / 30 437) | T4b grep |
| B1 true `parse_failed` rate (no rescue) | **0.060 %** (14 / 23 307) | T4b grep |
| B1 keyword-fallback rate | 0.009 % (2 / 23 307) | T4b grep |
| B0 `repaired_regex` (post-rescue, mid-tier salvage) | 0.154 % (47 / 30 437) | T4b grep |

Reading: removing GLM from B0 would convert ≈ 1.49 % of steps from "valid action" to "wait" (the `parse_action_text` final fallback). B1 already runs without any rescue at 0.060 % parse-fail rate, so the local-greedy paths are already at the noise floor.

---

## 3. Decision branches — what changes when advisor replies

### Option A: switch to clean structured-API path (retire GLM entirely)

**Code changes**:
- Delete `_load_glm_config` + `_call_glm_extract` + GLM init block (`proxy_api_agent.py:147-152, 167-287`)
- Delete GLM rescue branch in `step()` body (`proxy_api_agent.py:731-746`)
- Delete meta fields `glm_fallback_used` / `glm_fallback_attempted` / `glm_fallback_latency_ms` / `glm_original_fail_reason` (`proxy_api_agent.py:787-790`)
- Remove `urllib.request` import (only used by GLM call)
- Backend wrapper: drop `use_glm_fallback` / `glm_config` keys from `api_proxy.py:40-42`
- Configs: remove `use_glm_fallback: true` from ≈ 20 B0 yaml files (`grep -l use_glm_fallback configs/`)
- Runner: drop `glm_fallback_used` / `glm_fallback_latency_ms` / `glm_original_fail_reason` step_record fields (`runner/main.py:1279-1282`)
- Tests: remove any GLM-related test (none currently exist)

**Paper changes**:
- §3.5.1 GLM placeholder → "B0 uses the proxy API's structured-output mode (tool_choice forwarded to Bedrock); parse failures default to `wait` and are counted in the step's failure_reason field. B1/B2 use local greedy decoding."
- §4 limitation table: remove the GLM-rescue-rate disclosure row.

**Data changes**:
- Archived runs still carry GLM-rescued steps. Decision: either (a) treat archived B0 numbers as pre-fix data and re-run on the new structured-API B0, or (b) keep archived numbers with a clearer footnote that ≈ 1.5 % of steps were GLM-shaped.
- Need to decide in advisor sync.

**Effort**: 2-3 h code + paper text + config sweep.

---

### Option B: keep GLM + add downstream filter + disclose

**Code changes**: none on the rescue path itself.

**Aggregator changes**:
- Add `glm_fallback_used` filter as a first-class option to:
  - `scripts/analysis/aggregate_phantom_lift.py:100-113`
  - `scripts/analysis/aggregate_cross_site.py:190-210`
  - `scripts/analysis/compare_b0_b1.py:171-184`
  - `scripts/analysis/aggregate_sr_fp_per_mode.py:62-94`
- Per-condition output gains two SR numbers: `SR_raw` (current) and `SR_glm_filtered` (drop GLM-rescued steps from the trajectory or count them as wait).
- Decide policy: drop rescued steps (deflates B0 step count), or treat them as `valid=False` (deflates B0 SR).

**Paper changes**:
- §3.5.1 GLM placeholder → final wording: "B0 routes parse failures through an auxiliary GLM-5.1 extraction call (≈ 1.49 % of steps; recorded as `glm_fallback_used`). B1/B2 have no such fallback. Cross-baseline SR comparisons in Section 4 are reported both raw (B0 includes GLM-rescued steps) and GLM-filtered (B0 GLM-rescued steps reclassified as `wait`); the qualitative phantom-routing-space claims hold under both."
- §4 limitation table: cite the 1.49 % rescue rate explicitly.

**Effort**: 2-3 h aggregator + paper text.

---

### Option C: hybrid (advisor selects)

E.g. "use structured API where supported, keep GLM only as last-resort with a stricter ceiling". TBD; depends on what the proxy API can actually deliver.

---

## 4. Pending findings that route into this parking lot

### From `/stress A1.1` (2026-05-15)

- **Claude F3 (A1.1)** — GLM downstream filter 0 hits in `scripts/analysis/aggregate_*`. Grepped, confirmed by codex Mode B. → Option B child task; no action under Option A.
- **Codex C3 (A1.1)** — `parse_action_text` greedy regex `re.search(r'\{.*\}', DOTALL)` + lowercase-only `<think>` strip. Measured fail rate on B1 (no rescue) = **0.060 %** — well below 0.1 % no-fix threshold (B-90 pattern). Under Option A, the whole regex/keyword fallback chain in `action_utils.py:38-69` can be simplified to "strict json.loads → wait". Under Option B, leave as-is + paper §3 mentions the keyword-fallback rate.

### From `/stress A1.2` (2026-05-15) — codex Mode B

- **Codex C2/C4 (A1.2)** — silent zero-fill of meta fields. **The `glm_fallback_used` field is one axis** of this finding (alongside timing decomposition + image-token + confidence). Under Option A the field disappears entirely; under Option B it stays and the backend meta contract must include it as a required-on-B0 field.

### From `/stress A1.1` 3-AI round (2026-05-15 evening) — Claude + codex + gemini

User explicitly routed these to advisor-pending parking lot; do NOT act unilaterally.

- **P0-2 (gemini G1)** — B0 `use_tool_calling=True` in `proxy_api_agent.py:112-117` replaces system prompt `"Output ONLY valid JSON..."` with `"Use the web_action tool..."` + injects `tools` + `tool_choice` into payload. Plan A (tool-use API) dormant since dashscope proxy era; Plan B (GLM fallback) active. **User decision pending**: advisor sync on **official Qwen API channel** — if accessible, structured-output reliability becomes paper-grade comparable; if not, Plan A stays dormant + CI assert `use_tool_calling=False` for paper-grade configs. Currently `configs/exp_v2_base.yaml` has `use_tool_calling: false`, no active config overrides.
- **P0-7 (Claude F2 + codex C4)** — B0 meta dict missing `mean_logprob / min_logprob / mean_margin / min_margin / mean_entropy / max_entropy` 6 fields vs B1/B2 emit all 6 via `_compute_confidence`. §C learned router cross-baseline input space asymmetric. **User decision pending**: advisor sync on whether **DashScope OpenAI-format proxy** exposes token-level logprobs (it can in principle); if yes, `proxy_api_agent.step` can enable `logprobs=True, top_logprobs=2` and reuse `Qwen3VLAgent._compute_confidence` algebra on Python list. If no, declare in preregistration §3 as known cross-baseline asymmetry.
- **P0-9 partial defer (Claude F4 + Gemini concurrent angle)** — T=0/top_p=1 on proxy ≠ HF `do_sample=False`. **API probe verification script** (B-125, this round) measures step-level consistency cheaply (~30min, no VWA dep). **Full T=0 reproducibility audit** (paper-grade VWA-task-level consistency over N=10 same-(task,seed) runs) deferred to advisor sync to confirm scope budget.
- **P1-1 (gemini G3)** — GLM-5.1 scaffold `_call_glm_extract` rescues invalid JSON for B0 only; B1/B2 lack equivalent. **User decision pending**: advisor sync on whether to **drop parse_error rescue entirely** vs keep + transparently report pre-fallback rate. Tied to Plan A vs Plan B (parse_advisor_pending.md §1 master question).
- **P1-4 (codex C4)** — Confidence schema currently `confidence: Optional[Dict[str, Any]] = None` in `types.py:80-83`. Should become mandatory dict with fixed keys (`mean_logprob` etc) + `confidence_supported / confidence_extraction_error / confidence_source` audit fields. **Paired with P0-7** — both unlock at advisor decision on logprob availability.

### From paper drafts (already prose-staged)

- `paper_drafts/section3_definition.md` §3.5.1 — third paragraph "Parse-error recovery scaffold (provisional)" is a **placeholder**. Current text disclosed advisor-pending status + measured rescue rate. Replace with Option A or Option B final wording on advisor reply.

### Catalog cross-links

- `docs/reference/master_bug_catalog.md` B-86 entry — open, advisor-pending. Status update when advisor decides.

---

## 5. Forward append rule (for future `/stress A1.x` rounds)

Any new finding from `/stress` audits that:
- references `parse_action_text` / `_call_glm_extract` / `_load_glm_config` / `use_glm_fallback`, or
- touches `proxy_api_agent.py` parse / GLM blocks, or
- touches `action_utils.py` fallback chain (regex repair / keyword scroll/back), or
- emits `glm_fallback_*` step_record fields,

→ **append to §4 of this doc** instead of acting on it. Tag with the audit round (`A1.x`), severity, and the option (A/B) under which it would be acted upon.

---

## 6. Decision log

| Date | Thread | Decision | By |
|---|---|---|---|
| 2026-05-14 | GLM (B-86) | Advisor sync — question asked about clean structured API path; B-86 open | user + advisor |
| 2026-05-15 | GLM (B-86) | User directive: do not delete GLM yet; parking lot established | user |
| 2026-05-15 | GLM (B-86) | A1.1+A1.2 findings sorted into Option A / Option B branches above | claude /stress |
| 2026-05-15 | FE/RE (B-130) | Gemini Mode C P0-2 attack on Decision 3A; 3-option escalation drafted (§8 below) | claude /stress + gemini cross-AI |
| (TBD GLM) | GLM (B-86) | Option A / B / C selected → trigger code+paper fix per branch | advisor |
| (TBD FE/RE) | FE/RE (B-130) | Option (a) keep FE / (b) RE+Knapp-Hartung / (c) report both — selection → trigger §8 fix per branch | advisor |

---

## 7. Cross-references

- Catalog: `docs/reference/master_bug_catalog.md` B-86 (GLM thread) + B-130 (FE/RE thread, §143 audit batch)
- Paper drafts:
  - GLM: `docs/checkpoints/paper_drafts/section3_definition.md` §3.5.1
  - FE/RE: `docs/checkpoints/paper_drafts/section1_intro.md` (§1 hero hook generalization wording) + §8 statistical methods (`section8_limitations.md` §8.5 if added) + `preregistration.md` §2 estimand / §2.4 power / §4 pooling row / Appendix A Decision 3A entry / `osf_lock_manifest.md` §2.2
- Advisor sync: `docs/checkpoints/_status/issues/issue_advisor_sync_2026-05-14.md` (ADVISOR_SYNC.md retired 2026-05-15, commit `f64bc9d`)
- Audit chronicle: `docs/checkpoints/实验笔记.md` §142 (B-111~B-116 fix batch + Mode B+C v7) + §143 (this audit + cross-AI Mode C P0-2 chronicle)
- Codex outputs (GLM thread):
  - `docs/checkpoints/codex_outputs/A1_1_FINAL_2026-05-15.md` — A1.1 cross-validate confirming GLM filter gap
  - `docs/checkpoints/codex_outputs/A1_2_trace_2026-05-15.log` — A1.2 codex full synthesis (stdout, `-o` was empty)
- Cross-AI outputs (FE/RE thread):
  - `docs/checkpoints/codex_outputs/post_batch5_FINAL_2026-05-15_201813.md` — Mode B 10 findings (code-anchored, none on FE/RE — code-only audit didn't reach methodology layer)
  - `docs/checkpoints/gemini_outputs/post_batch5_2026-05-15_201813.md` — Mode C 7 findings incl. P0-2 attack on Decision 3A

---

## 8. Statistical estimand — Decision 3A FE vs RE rollback (B-130)

**Thread**: FE vs RE meta-analysis estimand (B-130, opened 2026-05-15 evening).

### 8.1 Context — what got locked 2026-05-14 (Decision 3A)

Per `preregistration.md §2.4` + §4 row "Pooling estimator + heterogeneity pre-spec" + Appendix A 2026-05-14 entry:

- **Estimand**: fixed-effects inverse-variance pooled average θ_FE over the 4 (now 6 per B2) *planned* (site, model) cells. The cells are the design, not a population sample → no between-cell variance τ² in the estimand → no DerSimonian-Laird, no REML.
- **Rationale (Claude v6 + codex cross-think)**: avoids DL τ² downward bias + RE Wald anti-conservatism at k<10 (Veroniki et al. 2016 / IntHout et al. 2014). FE is sound at any k under CLT on per-cell θ_i.
- **Witness**: paper_planning §19 + 实验笔记 §142 — advisor email pending lock.

### 8.2 Gemini Mode C P0-2 attack 2026-05-15 (cold cross-AI, prose-anchored)

> **Quote**: `osf_lock_manifest.md` "Fixed-effects inverse-variance pooled average... (decision '3A' 2026-05-14 — NOT DerSimonian-Laird; the cells are the design not a population, so no τ²)."
>
> **Attack**: 这是一个致命的统计学自杀 (statistical trap). 放弃 Random Effects (REML+HK/DL) 转而使用 Fixed-Effects (FE) 意味着你假设所有 6 个 cells 存在**唯一真实的 effect size**. 这在统计学上直接剥夺了论文 generalizability 的合法性. Reviewer 会攻击: FE 只能证明 "在这 3 个特定模型和 2 个特定网站上有效", 无法泛化到 "Web Agents" 这一 broader population.
>
> **Defuse**: 立即推翻 Decision 3A,回滚到 Random Effects meta-analysis. 即便 k=6 较小,RE 配合 Knapp-Hartung 调整也远比强制假设同质性的 FE 更符合顶级会议的统计严谨性标准.
>
> **Severity**: P0. **Effort**: 2h (重写 `aggregate_phantom_meta.py` 的 pooling 逻辑) + advisor email lock (1 day).

### 8.3 Why this is real (not just model-disagreement)

The FE-vs-RE choice is **estimand definition**, not estimator tuning:

- **FE estimand**: "average drop-one over EXACTLY these 6 planned cells (cls/red × B0/B1/B2)". Inference scope = these 6 cells, period.
- **RE estimand**: "average drop-one over a HYPOTHETICAL population of (site, model) cells, with the 6 observed cells as a sample". Inference scope = "Web Agents on VWA-style tasks" broadly.

Paper §1 hero hook currently says:
> "We characterize the **phantom routing space**: configurations on the 'skip annotated image' boundary..."

The implicit-generalization framing ("phantom routing space exists for Qwen+Gemma agents on VWA-style tasks") is RE-shaped, not FE-shaped. If the estimator says "we only learned about these 6 cells", the hook overpromises.

Veroniki/IntHout fragility is real at k<10, but **it does not vanish by switching to FE**:
- DL τ² downward-biased at k=4 → eases (but not gone) at k=6
- RE Wald anti-conservative at k=4 → eases (but not gone) at k=6
- Knapp-Hartung adjustment uses t-distribution at k-1 df (5 at k=6), restoring conservativeness

Gemini's recommendation (RE+Knapp-Hartung at k=6) is **statistically more conservative** than the 2026-05-14 FE choice while preserving generalization-claim language.

### 8.4 Decision branches — what changes when advisor replies

#### Option (a) — Keep Decision 3A FE + soften paper §1 generalization wording

**Code changes**: minimal. `aggregate_phantom_meta.py` still needs migration DL → FE to match prereg (current code↔prose drift). `preregistration_decision_test.py` similarly DL → FE.

**Paper changes**:
- §1 hero hook: rephrase to scope-explicit "characterizes phantom routing space on cls + red × B0/B1/B2" (drop "is a generalizable property" / "Web Agents broadly" framing)
- §8 limitations: add explicit FE-estimand scope statement

**Effort**: 4-6h paper rewrite + 2h code FE migration.

---

#### Option (b) — Roll back to RE+Knapp-Hartung at k=6 (Gemini recommendation)

**Code changes**:
- Rewrite `aggregate_phantom_meta.py` pooling: DerSimonian-Laird τ² estimation + Knapp-Hartung t-based CI at k-1=5 df
- Rewrite `preregistration_decision_test.py` H1/H3 estimand sections to match
- Update `osf_lock_manifest §2.2` H1 row from FE → RE+HK

**Paper changes**:
- preregistration §2.4 + §4 + Appendix A 2026-05-14 entry: amend to note Decision 3A reversed 2026-05-15+; cite Knapp-Hartung 2003 + reasoning re k<10 fragility mitigation
- §1 hero hook: retains generalization language under RE estimand

**Effort**: 2-4h code + 4h paper text + advisor email lock (1 day).

---

#### Option (c) — Report both FE + RE (primary + sensitivity)

**Code changes**: aggregator produces both FE and RE+HK outputs.

**Paper changes**:
- §4 main text: primary = (chosen by advisor); the other = sensitivity row in Appendix
- §8 limitations: full disclosure of estimand-choice sensitivity

**Effort**: 4h code + 6h paper text — most defensive but doubles §1+§4 prose.

### 8.5 Tradeoff matrix

| Estimator | Pros | Cons | Reviewer attack |
|---|---|---|---|
| **FE (Decision 3A, 2026-05-14)** | Sound at any k; no τ² estimation needed; clean | Inference limited to 6 cells; generalization claim 受限 | "你只测了 3 模型 × 2 站点,凭什么 claim 'phantom routing space is generalizable property'?" |
| **RE+Knapp-Hartung (Gemini P0-2 recommend)** | Restores generalization claim; HK adjustment fixes anti-conservativeness at k<10 | Still has DL τ² downward bias at k=6; FE vs RE point estimate may diverge if I² > 25% | "你用 RE 在 k=6 上 pool — IntHout 2014 / Veroniki 2016 都说 k<10 时 RE 不稳" |
| **DL random-effects** (current script impl) | What `aggregate_phantom_meta.py` actually computes today; matches archive | Most fragile at k<10; both biases active | "你 prereg 说 FE,代码跑 DL — code↔prose mismatch" (B-130 sub-finding) |

**Third bug exposed**: prereg says FE (Decision 3A), but `aggregate_phantom_meta.py` + `preregistration_decision_test.py` actually compute DL — code↔prose drift independent of FE/RE question.

### 8.6 Suggested advisor email/sync agenda

> Post-Batch-1-5 cross-AI audit surfaced a Decision 3A challenge from an angle our 2026-05-14 lock didn't cover:
>
> Gemini Mode C (independent prose audit): "FE estimand 把 paper §1 generalization claim 阉割了 — RE+Knapp-Hartung at k=6 才能保留 'phantom routing space is generalizable' 这种顶会 framing"
>
> Three options (§8.4):
>   (a) Keep Decision 3A FE + paper §1 hook 软化到 "characterizes phantom routing space on these 6 cells"
>   (b) Roll back to RE+Knapp-Hartung at k=6 (Gemini recommendation) — restore generalization-claim framing
>   (c) Report both as primary + sensitivity
>
> Need advisor decision before OSF DOI lock email goes out.

### 8.7 Affected gates

- 🔴 OSF lock email — locks estimand choice into DOI artifact; cannot send until decision
- 🔴 `aggregate_phantom_meta.py` — currently runs DL (matches no prereg version cleanly)
- 🔴 `scripts/analysis/preregistration_decision_test.py` — same DL-vs-FE-vs-RE inconsistency
- 🟠 `osf_lock_manifest.md §2.2` — was updated 2026-05-15 Batch 4 to FE wording; will need rewrite if (b) or (c)
- 🟠 Paper §1 hero claim language — generalization-claim coupling

### 8.8 Forward append rule for FE/RE thread

Any new finding from `/stress` / cross-AI that:
- references FE vs RE / DerSimonian-Laird / Knapp-Hartung / τ² / between-cell variance
- touches `aggregate_phantom_meta.py` pooling code OR `preregistration_decision_test.py` meta block
- challenges paper §1 hook generalization scope

→ append to this §8 instead of acting on it. Will not modify estimator unilaterally — Decision 3A advisor-witness-locked.

### 8.9 Status

⏳ **Open** — pending advisor sync / email lock decision (a)/(b)/(c).

After advisor decision lands → update preregistration §2.4 + §4 + Appendix A + `osf_lock_manifest §2.2` + the 2 affected analysis scripts + §6 Decision log row above → status: decided/closed.
