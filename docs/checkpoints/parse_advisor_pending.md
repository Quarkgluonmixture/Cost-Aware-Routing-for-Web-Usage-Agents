# Parse / GLM parking lot — pending advisor decision

> **Purpose**: collect every parse-related finding / fix / disclosure that needs the advisor's outcome on the "GLM-5.1 fallback vs clean structured-API B0 path" question before it can land.
>
> **Status**: open since 2026-05-14 advisor sync (B-86 question asked); awaiting advisor reply.
>
> **Rule**: any future `/stress A1.x` finding that touches the parse / GLM area lands here, **not in the live catalog or fix queue**. Move out of parking only after advisor decision.

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

| Date | Decision | By |
|---|---|---|
| 2026-05-14 | Advisor sync — question asked about clean structured API path; B-86 open | user + advisor |
| 2026-05-15 | User directive: do not delete GLM yet; parking lot established | user |
| 2026-05-15 | A1.1+A1.2 findings sorted into Option A / Option B branches above | claude /stress |
| (advisor reply date TBD) | Option A / B / C selected → trigger code+paper fix per branch | advisor |

---

## 7. Cross-references

- Catalog: `docs/reference/master_bug_catalog.md` B-86
- Paper draft: `docs/checkpoints/paper_drafts/section3_definition.md` §3.5.1
- Advisor sync: `docs/checkpoints/_status/issues/issue_advisor_sync_2026-05-14.md` (ADVISOR_SYNC.md retired 2026-05-15, commit `f64bc9d`)
- Audit chronicle: `docs/checkpoints/实验笔记.md` (pending §141 from this session)
- Codex outputs:
  - `docs/checkpoints/codex_outputs/A1_1_FINAL_2026-05-15.md` — A1.1 cross-validate confirming GLM filter gap
  - `docs/checkpoints/codex_outputs/A1_2_trace_2026-05-15.log` — A1.2 codex full synthesis (stdout, `-o` was empty)
