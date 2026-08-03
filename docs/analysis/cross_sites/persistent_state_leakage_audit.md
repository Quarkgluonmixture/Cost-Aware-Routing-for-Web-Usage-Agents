---
type: analysis
status: complete
purpose: for every success scored on persistent state, did this episode create it
post_hoc_exploratory: true
producer: scripts/analysis/audit_persistent_state_leakage.py
---

# Did the episode create the state it was scored on?

Regenerate: `.venv/bin/python3 scripts/analysis/audit_persistent_state_leakage.py`

`require_reset` is a no-op on reddit, so within a run every episode shares one Postmill instance and one account. A task scored by reading persistent state can therefore pass on state an earlier episode left behind. `reddit_sidebar_leakage_audit` measured that on VisualWebArena with seven hand-picked task ids; §8b hand-traced two WebArena episodes and said plainly it was *not* an audit. This derives the target from each evaluator's own configuration and runs on both benchmarks — reproducing the VWA result is what makes the WA numbers readable.

| cell | successes scored on persistent state | earned | **leaked** | leak share |
|---|---|---|---|---|
| `wa_B0` | 39 | 39 | **0** | 0.0% |
| `wa_B1` | 22 | 22 | **0** | 0.0% |
| `red_B0` | 18 | 7 | **11** | 61.1% |
| `red_B1` | 15 | 7 | **8** | 53.3% |
| `red_B2` | 4 | 1 | **3** | 75.0% |

**WebArena: 0 leaked.** VisualWebArena: 22.

## ⚠️ This implementation is NOT calibrated — read the zero accordingly

`reddit_sidebar_leakage_audit` establishes **6** leaked successes on VisualWebArena. This script reports **22** on the same data, so it **over-flags by roughly 3.7x** and its VWA count must not be quoted. Three successive filters were tried (86 → 30 → 22) and the generalisation was then abandoned rather than tuned further, because fitting a heuristic to an answer you already hold is not validation.

**What the WebArena zero is still worth.** The error is one-sided: every version of this test flagged *more* than the truth, never fewer. A test that cries wolf 3.7x too often on the benchmark where the answer is known, and then finds **nothing at all** on WebArena, is evidence that WebArena carries little of this defect — weaker than a calibrated audit, stronger than the two-episode hand check §8b had. It is **not** a clean bill of health, and the ⚠️ unaudited marks on the WA cells should stay until a criterion that reproduces the 6 exists.

### `red_B0` — leaked successes by mode

| mode | scored on state | leaked | tasks |
|---|---|---|---|
| DOM | 5 | **3** | 171, 188, 189 |
| SoM | 3 | **1** | 188 |
| Vision | 2 | **2** | 188, 189 |
| P-text | 3 | **2** | 188, 189 |
| P-prompt | 3 | **2** | 188, 189 |
| P-SoM | 2 | **1** | 188 |

### `red_B1` — leaked successes by mode

| mode | scored on state | leaked | tasks |
|---|---|---|---|
| DOM | 2 | **1** | 188 |
| SoM | 3 | **2** | 188, 189 |
| P-text | 4 | **2** | 188, 189 |
| P-prompt | 2 | **1** | 189 |
| P-SoM | 4 | **2** | 188, 189 |

### `red_B2` — leaked successes by mode

| mode | scored on state | leaked | tasks |
|---|---|---|---|
| DOM | 3 | **3** | 178, 188, 189 |

## How to read a zero

A zero in the leaked column means *no success on a state-scored task happened without the episode reaching the object the evaluator reads*. It does **not** mean the run is free of state carry-over — an episode can visit the forum AND benefit from an earlier subscription, and this test scores that as earned. The test is one-sided by design: it catches successes that are certainly unearned, not all of those that might be.
