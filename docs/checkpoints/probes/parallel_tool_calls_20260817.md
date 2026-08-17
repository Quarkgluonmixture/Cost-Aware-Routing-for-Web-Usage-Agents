# Probe — does the AWS proxy honour `parallel_tool_calls: false`? (2026-08-17)

**Verdict: no. The field is silently ignored. Client-side handling is the only lever.**

Driver: Gemini Mode C Finding 2 (`/stress` 2026-08-17) argued that suppressing parallel
emission upstream is strictly better than dropping the extra call client-side, and that
deferring the 60-second test before a $48 fire was reckless. Correct instinct — so it was
tested rather than assumed.

## Method

`scripts`-free one-off probe run on the A100 fire host against the production endpoint
(`.../model-api/invoke`, `qwen.qwen3-vl-235b-a22b`, `tool_choice="required"`, T=0), with a
prompt deliberately written to invite two actions:

> Page: `[7] textbox 'Search'` / `[9] button 'Go'`
> Type 'laptop' into the search box **and then** press the Go button.

Three arms. Arm C is the one that matters and was not in Gemini's proposal: if the proxy
merely **ignores unknown keys**, then B alone would 200 and look like a fix while changing
nothing. Prior art in `proxy_api_agent.py:739-741` says this proxy does exactly that to
`tool_choice={"type":"function"}`.

## Result

| Arm | payload | HTTP | `tool_calls` | `web_action` count |
|---|---|---|---|---|
| A control | production shape, no flag | 200 | 2 | **2** |
| B | `+ parallel_tool_calls: false` | 200 | 2 | **2** |
| C | `+ parallel_tool_calls: true` | 200 | 2 | **2** |

B ≡ C ≡ A. The flag changes nothing in either direction ⇒ the proxy accepts the key and
discards it. There is no upstream switch to throw.

## Consequences

1. **The upstream defuse is closed.** Taking the first `web_action` client-side
   (`B-1980-followup`) is not a workaround chosen over a cleaner fix; it is the only
   available handling.
2. **Parallel emission is task-driven and reproducible, not a rare glitch.** The control
   arm — today's exact production configuration — returns two `web_action` calls whenever
   the instruction implies two steps. It is not a low-probability drift artefact.
   Consequence for the v1 B-1980 guard: a `raise` on `len(web_action) > 1` does not
   *risk* killing a 466-task condition, it **guarantees** it.
3. **The dropped call carries model intent, not noise.** Arm A's second call is the "press
   Go" half of a two-step plan. This supports recording the discarded payload rather than
   only a count (Gemini Finding 3), and it retires the word "speculative" as a description
   of what is being dropped (Gemini Findings 6/7): call 2 is an attempted **macro-action**
   that the one-action-per-step runner cannot honour.

## What this probe does NOT establish

- It does **not** measure the in-the-wild rate of parallel emission during a real episode.
  The prompt here was written to provoke it. The rate under actual VWA task prompts is what
  `parallel_web_action_dropped` now records per step, and it must be read off a real run
  before any claim about frequency is made.
- It does **not** speak to whether pre-2026-08-10 archived B0 had the same behaviour. That
  remains unobservable — the old code discarded the extra call without leaving a trace.
