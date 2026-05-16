# P1 (rule-based) router archive simulation — SANITY-CHECK ONLY

> ⚠️ **NOT preregistration lock substrate.** Archive uses same task IDs Phase 1a will use; outcomes are pre-§107 + pre-§139.8 fix. Numbers below are *directional* — they answer 'does P1 v3 design have honest phantom-dependency signal?', NOT 'P1 SR is X%'. Real P1 SR claim must come from Phase 1a fresh-data 5-fold CV. See `proposals_v4.md` §A for methodology reframe.

Run date: `2026-05-16T09:21:15.511002Z`

## Method recap

- **P1 v3 rule** (from `proposals_v4.md` decide_p1_v3):
  - if `intent` matches search regex (`find|search|locate|how many|how much`) → `dom`
  - elif `dom_size > 12000` OR `dom_complexity > 500` → `som`
  - else → `phantom_som`
- **P1 3-mode counterfactual**: same tree but else → `dom` (drops phantom dependency)
- Features: step-0 `state_digest.{text_length, dom_complexity}` + task config `intent`
- Outcome: archive `success` per (task_id, mode)

## Per-cell results

### B0_classifieds

- n tasks (intersection): **234**, modes retained: `['dom', 'som', 'vision', 'phantom_text', 'phantom_som']`
- features extracted: **234** tasks (config + step-0 parsed)
- skipped modes: ['phantom_prompt: only 4 ep (< 50)']

**Router comparison**:

| Router | SR (%) | N | Routed mode distribution |
|---|---:|---:|---|
| P1_v3_6mode | 15.81 | 234 | phantom_som=123 (52.6%), dom=111 (47.4%) |
| P1_3mode | 14.96 | 234 | dom=234 (100.0%) |
| always_dom | 14.96 | 234 | dom=234 (100.0%) |
| always_som | 23.08 | 234 | som=234 (100.0%) |
| always_phantom_som | 15.81 | 234 | phantom_som=234 (100.0%) |
| oracle | 35.90 | 234 | dom=185 (79.1%), som=29 (12.4%), vision=11 (4.7%), phantom_text=7 (3.0%), phantom_som=2 (0.9%) |

**Per-routed-mode SR** (P1_v3_6mode breakdown — when router routes to X, what's the archive SR?):

| Routed mode | N | SR (%) |
|---|---:|---:|
| phantom_som | 123 | 15.45 |
| dom | 111 | 16.22 |

**P1 phantom dependency**: 52.6% of tasks routed to phantom_som by v3.
**P1 v3 (6-mode) vs P1 3-mode delta**: +0.85pp (phantom HELPS)

### B0_reddit

- n tasks (intersection): **210**, modes retained: `['dom', 'som', 'vision', 'phantom_text', 'phantom_prompt', 'phantom_som']`
- features extracted: **210** tasks (config + step-0 parsed)

**Router comparison**:

| Router | SR (%) | N | Routed mode distribution |
|---|---:|---:|---|
| P1_v3_6mode | 14.29 | 210 | phantom_som=156 (74.3%), dom=54 (25.7%) |
| P1_3mode | 11.43 | 210 | dom=210 (100.0%) |
| always_dom | 11.43 | 210 | dom=210 (100.0%) |
| always_som | 11.90 | 210 | som=210 (100.0%) |
| always_phantom_som | 14.29 | 210 | phantom_som=210 (100.0%) |
| oracle | 25.24 | 210 | dom=181 (86.2%), som=10 (4.8%), phantom_text=8 (3.8%), vision=5 (2.4%), phantom_prompt=4 (1.9%), phantom_som=2 (1.0%) |

**Per-routed-mode SR** (P1_v3_6mode breakdown — when router routes to X, what's the archive SR?):

| Routed mode | N | SR (%) |
|---|---:|---:|
| phantom_som | 156 | 13.46 |
| dom | 54 | 16.67 |

**P1 phantom dependency**: 74.3% of tasks routed to phantom_som by v3.
**P1 v3 (6-mode) vs P1 3-mode delta**: +2.86pp (phantom HELPS)

## Cross-cell summary

| Cell | always_dom | P1_3mode | P1_v3_6mode | Δ (6-3) | always_phantom_som | oracle |
|---|---:|---:|---:|---:|---:|---:|
| B0_classifieds | 14.96 | 14.96 | 15.81 | +0.85 | 15.81 | 35.90 |
| B0_reddit | 11.43 | 11.43 | 14.29 | +2.86 | 14.29 | 25.24 |

## Interpretation guide

- If P1_v3_6mode > P1_3mode by ≥ 1pp → phantom dependency adds archive SR (directional confidence that v3 design is sensible)
- If P1_v3_6mode ≈ P1_3mode (within ±0.5pp) → phantom path SR ≈ DOM path SR on routed-tasks; phantom value is cost not SR (paper §6 should frame Pareto, not lift)
- If P1_v3_6mode < P1_3mode by > 1pp → phantom_som hurts on the 'simple non-search' task slice; v3 rule needs revision (likely v5: tighter complexity threshold or alternative else-branch)

**Reminder**: this is correlated-population evidence on B0 only. Phase 1a 6-cell fresh data is the paper-grade test.
