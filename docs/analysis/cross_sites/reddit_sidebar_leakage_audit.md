# reddit sidebar-task leakage audit

- **9 tasks** are scored by reading `#sidebar > section > ul` (the subscribed-forum list)
- `require_reset` is a **no-op on reddit** (`envs.py:172` gates it on `"classifieds" in sites`, `TODO(jykoh)` for the rest), so subscriptions accumulate across the 205 episodes of a run
- **earned** = the episode visited the required forum · **LEAKED** = scored success without ever visiting it · **passive-satisfiable** = `must_exclude`-only eval, satisfied by doing nothing
- verdicts are mechanical and per-episode. Whether to exclude these tasks is a **preregistration-level decision** and is not made here.

## Verdict counts (scored universe only)

| verdict | n |
|---|---|
| **LEAKED** | **6** |
| earned | 31 |
| failed | 107 |

Plus **13** passive-satisfiable successes on protocol-excluded task(s) [160] — already outside the scored universe via AMENDMENT_08, listed for completeness.

## Per-cell impact on the scored success count

| cell · mode | scored successes | of which LEAKED | leaked share |
|---|---|---|---|
| B0 · DOM | 29 | 1 | 3.4% |
| B0 · SoM | 30 | 0 | 0.0% |
| B0 · Vision | 15 | 1 | 6.7% |
| B0 · P-text | 27 | 0 | 0.0% |
| B0 · P-SoM | 22 | 0 | 0.0% |
| B0 · P-prompt | 25 | 0 | 0.0% |
| B1 · DOM | 12 | 0 | 0.0% |
| B1 · SoM | 15 | 1 | 6.7% |
| B1 · Vision | 5 | 0 | 0.0% |
| B1 · P-text | 12 | 0 | 0.0% |
| B1 · P-SoM | 12 | 0 | 0.0% |
| B1 · P-prompt | 11 | 0 | 0.0% |
| B2 · DOM | 8 | 3 | 37.5% |
| B2 · SoM | 2 | 0 | 0.0% |
| B2 · Vision | 4 | 0 | 0.0% |
| B2 · P-text | 4 | 0 | 0.0% |
| B2 · P-SoM | 1 | 0 | 0.0% |
| B2 · P-prompt | 0 | 0 | — |

## Every leaked success

| cell | mode | task | eval wants sidebar to contain | forums visited |
|---|---|---|---|---|
| B0_reddit | DOM | 171 | `mechanicalkeyboards` | 1 forums, **none of them the target** |
| B0_reddit | Vision | 189 | `deeplearning | machinelearning | singularity` | 1 forums, **none of them the target** |
| B1_reddit | SoM | 189 | `deeplearning | machinelearning | singularity` | 1 forums, **none of them the target** |
| B2_reddit | DOM | 178 | `nyc` | 3 forums, **none of them the target** |
| B2_reddit | DOM | 188 | `iphone | technology` | 2 forums, **none of them the target** |
| B2_reddit | DOM | 189 | `deeplearning | machinelearning | singularity` | 2 forums, **none of them the target** |

## Earned successes (kept)

- B0_reddit · DOM · task 178 — visited `nyc`
- B0_reddit · DOM · task 188 — visited `iphone`
- B0_reddit · DOM · task 189 — visited `machinelearning`
- B0_reddit · DOM · task 190 — visited `art`
- B0_reddit · P-SoM · task 178 — visited `nyc`
- B0_reddit · P-SoM · task 188 — visited `iphone, technology`
- B0_reddit · P-prompt · task 178 — visited `nyc`
- B0_reddit · P-prompt · task 188 — visited `iphone`
- B0_reddit · P-prompt · task 189 — visited `deeplearning`
- B0_reddit · P-text · task 178 — visited `nyc`
- B0_reddit · P-text · task 188 — visited `iphone`
- B0_reddit · P-text · task 189 — visited `machinelearning`
- B0_reddit · SoM · task 171 — visited `mechanicalkeyboards`
- B0_reddit · SoM · task 178 — visited `nyc`
- B0_reddit · SoM · task 188 — visited `iphone`
- B0_reddit · Vision · task 188 — visited `technology`
- B1_reddit · DOM · task 171 — visited `mechanicalkeyboards`
- B1_reddit · DOM · task 188 — visited `iphone`
- B1_reddit · P-SoM · task 171 — visited `mechanicalkeyboards`
- B1_reddit · P-SoM · task 178 — visited `nyc`
- B1_reddit · P-SoM · task 188 — visited `iphone`
- B1_reddit · P-SoM · task 189 — visited `machinelearning`
- B1_reddit · P-prompt · task 171 — visited `mechanicalkeyboards`
- B1_reddit · P-prompt · task 189 — visited `machinelearning`
- B1_reddit · P-text · task 171 — visited `mechanicalkeyboards`
- B1_reddit · P-text · task 178 — visited `nyc`
- B1_reddit · P-text · task 188 — visited `iphone`
- B1_reddit · P-text · task 189 — visited `machinelearning`
- B1_reddit · SoM · task 171 — visited `mechanicalkeyboards`
- B1_reddit · SoM · task 188 — visited `iphone`
- B2_reddit · SoM · task 170 — visited `sports`
