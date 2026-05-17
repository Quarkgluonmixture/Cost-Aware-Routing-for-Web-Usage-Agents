---
type: issue
category: code-migration
status: resolved
priority: high
resolved: 2026-05-18
created: 2026-05-17
updated: 2026-05-18
action: B0 proxy migration code-side complete — Phase 1a fire substrate ready
---

# B0 proxy migration — GLM rescue retire + OpenAI-style tool_calling enabled

Code-side companion to §208 A2.2 Tier 1 B-901 prose retire. Closes the
B-262 / B-86 advisor-pending thread on B0 parse-error rescue, replacing
GLM-5.1 scaffold with native AWS proxy tool_calling.

## Resolution summary

**Probe-driven migration** (NOT advisor-driven): user "都搞定了哈" reply
covered API quota + DGX disk only; AWS proxy `tool_choice` + `logprobs`
capability had been available all along — gap was at P79 payload format
layer (Anthropic-style schema → HTTP 400; OpenAI-style → HTTP 200 +
clean `tool_calls[0].function.arguments`).

## Commits (3 chunks)

- **Chunk A** `e7546f7` — code + configs + tests (1626 insertions /
  281 deletions, 9 files): `p79/agents/proxy_api_agent.py` OpenAI tools
  swap + GLM 155 LOC delete + logprob extract; `configs/exp_v2_base.yaml`
  `use_tool_calling: true` flip; `tests/test_proxy_openai_transport.py`
  5 monkeypatch cases; 4 probe scripts.
- **Chunk B** `cec0ab2` — probe artifacts (5713 insertions / 3 files):
  `docs/checkpoints/probes/proxy_capability_{230807,v2_223704}.json`
  + `proxy_full_stack_225749.json`.
- **Chunk C** `22f7f88` — chronicle + catalog (201 insertions / 2 files):
  `实验笔记.md §210` + `master_bug_catalog A1.2-followup` section.

## Q1=A pilot gate empirical (N=30, A100)

- emit_rate = 100% (30/30)
- schema_valid_rate = 100% (30/30)
- logprobs_present_rate = 100% (30/30)
- action_type distribution: 15 click + 15 select_option (both schemas pass)
- cost $0.00179 ± $0.0001 / elapsed 1216±258ms / logprob tokens 57.5±19.8

## Cross-baseline confidence schema post-migration

| Baseline | mean_logprob | min_logprob | mean_margin | min_margin | mean_entropy | max_entropy |
|---|---|---|---|---|---|---|
| B0 (proxy + tool_use auto) | ✓ | ✓ | ✓ | ✓ | None | None |
| B1 (HF Qwen3-VL-4B) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| B2 (HF Gemma3-VL-4B) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

Entropy=None on B0 per top-2 logprobs truncation (full-vocab entropy
unobservable). Disclosed in `_compute_confidence_from_proxy_logprobs`
docstring. §C router can use 4 fields cross-baseline; B1/B2 within-
baseline can use 6.

## Closes

- B-86 (parse_advisor_pending.md Thread 1 GLM rescue) — Option A
  selected, code path deleted entirely
- B-262 (GLM channel migration advisor-pending) — Option A landed via
  probe-driven empirical (not advisor email)
- A2.2 §208 B-901 prose retire — concurrent code-side work referenced
  in paper §3.5.1 "a concurrent code-side session lands _WEB_ACTION_TOOL
  OpenAI-style + Path-1 parser reading tool_calls"

## Phase 1a fire impact

✓ Substrate ready — `use_tool_calling: true` default, transport tests
freeze contract, probe artifacts on disk for audit trail.

## Cross-link

- `docs/checkpoints/实验笔记.md §210`
- `docs/reference/master_bug_catalog.md` (A1.2-followup section)
- `docs/checkpoints/parse_advisor_pending.md` Thread 1 (now ready to close)
- §208 A2.2 catalog entry B-901 (prose retire, paper §3.5.1)
