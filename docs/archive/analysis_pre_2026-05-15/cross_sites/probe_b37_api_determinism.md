# B-37 API Determinism Probe

**Audit date**: 2026-04-30
**Model**: qwen.qwen3-vl-235b-a22b
**N calls**: 5 × temperature=0

## Verdict: **TOKEN-LEVEL NON-DETERMINISTIC, DECISION-LEVEL CONVERGENT**

5/5 calls produced 5 distinct byte-level outputs at T=0+top_p=1.0+seed=42 forwarded.
**However** all 5 picked the same action: `click [element_id=5]` (cheapest blue
kayak). Output token counts varied (38, 45, 46, 49, 49 tokens) — model is
genuinely sampling differently across calls.

### Implications for Phase A paper claim

**B0 reproducibility ≠ B1 reproducibility**:
- B1 (4B local + do_sample=False + torch.manual_seed): byte-deterministic by construction
- B0 (235B proxy + T=0 + top_p=1.0 + seed forwarded): **NOT byte-deterministic**, but
  decision-level convergent (action selection appears stable across replicates)

**Action-level consequences for paper SR**:
- ✅ Same task → same action → same downstream observation → same next-step decision (mostly)
- ⚠️ Token-level wording variance affects: `string_match` evaluator (exact answer
  match), `finish.answer` text (e.g., "$320 Blue Inflatable Kayak" vs "$320 blue
  inflatable kayak"), thought-text bootstrap CI from agent reasoning
- 🔴 Cannot claim B0 is "fully reproducible at seed=42" in paper Section 4

### Recommended Section 4 disclosure (1 paragraph)

> "We use temperature=0 with top_p=1.0 across all baselines and forward seed=42
> via best-effort proxy headers. B1 (Qwen3-VL-4B local) is byte-deterministic
> by construction (do_sample=False, manual torch seeding). B0 (Qwen3-VL-235B
> via proxy API) is **not byte-deterministic** in our verification probe (5/5
> calls produced 5 distinct outputs at temperature=0; provider's deterministic
> infrastructure is unverifiable from client). However, **decision-level
> convergence is empirically observed** — across 5 replicates of one
> representative prompt, all 5 selected the same action (`click [element_id=5]`).
> SR-level conclusions are therefore stable to proxy non-determinism, but
> token-level metrics (string_match exact-answer match, thought-text similarity)
> may have residual variance not captured by our task-level bootstrap CI."



## Per-call detail

| Call | Status | Digest | Length | Elapsed |
|---:|---:|:---|---:|---:|
| 1 | 200 | `f83bc369231c8a8c` | 477 | 2761ms |
| 2 | 200 | `ee9b2b6cf3a0ce4e` | 455 | 834ms |
| 3 | 200 | `72f07ca81a7a5b86` | 461 | 897ms |
| 4 | 200 | `127521ee4a72359d` | 469 | 885ms |
| 5 | 200 | `c26643e84e46ee7c` | 468 | 835ms |

## Output texts

### Call 1 (digest `f83bc369231c8a8c`)
```
{"content":"{\n  \"action_type\": \"click\",\n  \"element_id\": 5,\n  \"thought\": \"The $320 Blue Inflatable Kayak is the cheapest option among the listed blue kayaks.\"\n}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":46,"cost":0.00039200000000000004},"metadata":{"remaining_quota":{"llm_cost":332.60454799999997,"total_cost":332.60454799999997,"budget_limit":350,"remaining_budget":17.39545200000001,"budget_usage_percent":95.02987085714285}}}
```

### Call 2 (digest `ee9b2b6cf3a0ce4e`)
```
{"content":"{\"action_type\": \"click\", \"element_id\": 5, \"thought\": \"The $320 blue inflatable kayak is the cheapest option available in the search results.\"}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":38,"cost":0.000352},"metadata":{"remaining_quota":{"llm_cost":332.60490000000004,"total_cost":332.60490000000004,"budget_limit":350,"remaining_budget":17.395099999999978,"budget_usage_percent":95.02997142857143}}}
```

### Call 3 (digest `72f07ca81a7a5b86`)
```
{"content":"{\"action_type\": \"click\", \"element_id\": 5, \"thought\": \"The cheapest blue kayak listed is the $320 Blue Inflatable Kayak, so I will click on it to view more details.\"}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":49,"cost":0.000407},"metadata":{"remaining_quota":{"llm_cost":332.605307,"total_cost":332.605307,"budget_limit":350,"remaining_budget":17.394693000000014,"budget_usage_percent":95.0300877142857}}}
```

### Call 4 (digest `127521ee4a72359d`)
```
{"content":"{\"action_type\": \"click\", \"element_id\": 5, \"thought\": \"The cheapest blue kayak listed is the $320 Blue Inflatable Kayak, which matches the goal of finding the lowest-priced option.\"}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":49,"cost":0.000407},"metadata":{"remaining_quota":{"llm_cost":332.605714,"total_cost":332.605714,"budget_limit":350,"remaining_budget":17.39428600000002,"budget_usage_percent":95.030204}}}
```

### Call 5 (digest `c26643e84e46ee7c`)
```
{"content":"{\n  \"action_type\": \"click\",\n  \"element_id\": 5,\n  \"thought\": \"The $320 Blue Inflatable Kayak is the cheapest option available among the search results.\"\n}","model":"qwen.qwen3-vl-235b-a22b","usage":{"inputTokens":162,"outputTokens":45,"cost":0.00038700000000000003},"metadata":{"remaining_quota":{"llm_cost":332.606101,"total_cost":332.606101,"budget_limit":350,"remaining_budget":17.393898999999966,"budget_usage_percent":95.03031457142858}}}
```
