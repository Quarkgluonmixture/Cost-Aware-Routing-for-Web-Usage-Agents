---
type: analysis
status: rolling
purpose: does any reddit claim depend on successes credited by accumulated site state
producer: scripts/analysis/leakage_sensitivity.py
---

# Sidebar-leakage sensitivity

Regenerate: `.venv/bin/python3 scripts/analysis/leakage_sensitivity.py`

`reddit_sidebar_leakage_audit` finds **6 scored successes** credited without the episode ever visiting the forum the evaluator reads. `require_reset` is a no-op on reddit, so subscriptions accumulate across a run's 205 episodes and a later task can be scored on state an earlier one created — an *environmental* credit, not a behavioural one.

Below, each of those successes is set to **0** and every reddit contrast is recomputed. The denominator is unchanged: a leaked success is an attempted-and-not-accomplished task, so 0 is the correct value and dropping the row would make the columns incomparable. Paired bootstrap, 10,000 resamples, seed 20260802 — identical to `fusion_premium`.

Removed: `red_B0`·DOM·task 171, `red_B0`·Vision·task 189, `red_B1`·SoM·task 189, `red_B2`·DOM·task 178, `red_B2`·DOM·task 188, `red_B2`·DOM·task 189.

## 1. Fusion contrasts, before and after

| cell | contrast | before | 95% CI | after | 95% CI | shift | verdict |
|---|---|---|---|---|---|---|---|
| `red_B0` | SoM − Vision | +7.39pp | [+2.46, +12.32] | **+7.88pp** | [+2.96, +12.81] | +0.49pp | unchanged |
| `red_B0` | SoM − DOM | +0.49pp | [-3.94, +4.93] | **+0.99pp** | [-3.45, +5.42] | +0.49pp | unchanged |
| `red_B1` | SoM − Vision | +4.93pp | [+1.48, +8.87] | **+4.43pp** | [+0.99, +7.88] | -0.49pp | unchanged |
| `red_B1` | SoM − DOM | +1.48pp | [-1.48, +4.43] | **+0.99pp** | [-1.48, +3.94] | -0.49pp | unchanged |
| `red_B2` | SoM − Vision | -0.99pp | [-3.45, +1.48] | **-0.99pp** | [-3.45, +1.48] | +0.00pp | unchanged |
| `red_B2` | SoM − DOM | -2.96pp | [-5.91, -0.49] | **-1.48pp** | [-3.45, +0.49] | +1.48pp | **flips** — excludes 0 → includes 0 |

## 2. Per-mode SR and the best single mode

| cell | mode | SR before | SR after | Δ |
|---|---|---|---|---|
| `red_B0` | Vision | 7.39% | 6.90% | -0.49pp |
| `red_B0` | DOM | 14.29% | 13.79% | -0.49pp |
| `red_B1` | SoM | 7.39% | 6.90% | -0.49pp |
| `red_B2` | DOM | 3.94% | 2.46% | -1.48pp |

Modes not listed are untouched. The best single mode is **unchanged in every cell** (SoM, SoM, DOM respectively).

## 3. What this changes

**1 verdict(s) depend on the leaked episodes**: `red_B2` SoM − DOM.

`red_B2` SoM − DOM was the **only** interval in the eight-cell fusion table lying entirely on the negative side — the single piece of evidence that the fused mode is *significantly worse* than a single channel anywhere. Three of the eight successes behind `red_B2`·DOM are leaked (37.5% of that arm's successes, the highest share of any arm), and with them removed the interval crosses zero. **The claim that fusion is significantly beaten in some cell rests on accumulated site state.**

What does **not** move: the modality reversal. `red_B0` and `red_B1` SoM − Vision both still exclude zero, and `red_B0` moves *further* from zero. The leak count is also asymmetric in a way that works against the fusion arm rather than for it — 4 of the 6 leaks are on DOM, 1 on Vision, 1 on SoM — so removing them can only help the fused channel's comparisons, which is the direction that disfavours the paper's own caution.

⚠️ **Scope.** This covers VWA reddit only. `audit_reddit_sidebar_leakage.py` reads `external/visualwebarena/config_files/vwa/test_reddit`; the WebArena reddit cells use a different task set and have not been audited for the same defect. The mechanism (`require_reset` gated on classifieds) applies to any Postmill site, so absence of an audit is not evidence of absence of leakage.

