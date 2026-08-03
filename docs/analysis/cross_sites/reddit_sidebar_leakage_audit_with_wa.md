# reddit sidebar-task leakage audit

- **9 tasks** are scored by reading `#sidebar > section > ul` (the subscribed-forum list)
- `require_reset` is a **no-op on reddit** (`envs.py:172` gates it on `"classifieds" in sites`, `TODO(jykoh)` for the rest), so subscriptions accumulate across the 205 episodes of a run
- **earned** = the episode visited the required forum · **LEAKED** = scored success without ever visiting it · **passive-satisfiable** = `must_exclude`-only eval, satisfied by doing nothing
- verdicts are mechanical and per-episode. Whether to exclude these tasks is a **preregistration-level decision** and is not made here.

## Verdict counts (scored universe only)

| verdict | n |
|---|---|
| **LEAKED** | **6** |
| earned | 68 |
| failed | 120 |

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
- B0_wa_reddit · DOM · task 596 — visited `books`
- B0_wa_reddit · DOM · task 597 — visited `consoles`
- B0_wa_reddit · DOM · task 598 — visited `pittsburgh`
- B0_reddit · P-SoM · task 178 — visited `nyc`
- B0_reddit · P-SoM · task 188 — visited `iphone, technology`
- B0_wa_reddit · P-SoM · task 595 — visited `space`
- B0_wa_reddit · P-SoM · task 596 — visited `books`
- B0_wa_reddit · P-SoM · task 597 — visited `consoles`
- B0_wa_reddit · P-SoM · task 598 — visited `pittsburgh`
- B0_wa_reddit · P-SoM · task 599 — visited `machinelearning`
- B0_reddit · P-prompt · task 178 — visited `nyc`
- B0_reddit · P-prompt · task 188 — visited `iphone`
- B0_reddit · P-prompt · task 189 — visited `deeplearning`
- B0_reddit · P-text · task 178 — visited `nyc`
- B0_reddit · P-text · task 188 — visited `iphone`
- B0_reddit · P-text · task 189 — visited `machinelearning`
- B0_wa_reddit · P-text · task 595 — visited `space`
- B0_wa_reddit · P-text · task 596 — visited `books`
- B0_wa_reddit · P-text · task 597 — visited `consoles`
- B0_wa_reddit · P-text · task 598 — visited `pittsburgh`
- B0_wa_reddit · P-text · task 599 — visited `machinelearning`
- B0_reddit · SoM · task 171 — visited `mechanicalkeyboards`
- B0_reddit · SoM · task 178 — visited `nyc`
- B0_reddit · SoM · task 188 — visited `iphone`
- B0_wa_reddit · SoM · task 596 — visited `books`
- B0_wa_reddit · SoM · task 597 — visited `consoles`
- B0_wa_reddit · SoM · task 598 — visited `pittsburgh`
- B0_reddit · Vision · task 188 — visited `technology`
- B0_wa_reddit · Vision · task 596 — visited `books`
- B0_wa_reddit · Vision · task 597 — visited `consoles`
- B0_wa_reddit · Vision · task 598 — visited `pittsburgh`
- B1_reddit · DOM · task 171 — visited `mechanicalkeyboards`
- B1_reddit · DOM · task 188 — visited `iphone`
- B1_wa_reddit · DOM · task 595 — visited `space`
- B1_wa_reddit · DOM · task 597 — visited `consoles`
- B1_wa_reddit · DOM · task 598 — visited `pittsburgh`
- B1_wa_reddit · DOM · task 599 — visited `machinelearning`
- B1_reddit · P-SoM · task 171 — visited `mechanicalkeyboards`
- B1_reddit · P-SoM · task 178 — visited `nyc`
- B1_reddit · P-SoM · task 188 — visited `iphone`
- B1_reddit · P-SoM · task 189 — visited `machinelearning`
- B1_wa_reddit · P-SoM · task 595 — visited `space`
- B1_wa_reddit · P-SoM · task 596 — visited `books`
- B1_wa_reddit · P-SoM · task 597 — visited `consoles`
- B1_wa_reddit · P-SoM · task 598 — visited `pittsburgh`
- B1_wa_reddit · P-SoM · task 599 — visited `machinelearning`
- B1_reddit · P-prompt · task 171 — visited `mechanicalkeyboards`
- B1_reddit · P-prompt · task 189 — visited `machinelearning`
- B1_reddit · P-text · task 171 — visited `mechanicalkeyboards`
- B1_reddit · P-text · task 178 — visited `nyc`
- B1_reddit · P-text · task 188 — visited `iphone`
- B1_reddit · P-text · task 189 — visited `machinelearning`
- B1_wa_reddit · P-text · task 595 — visited `space`
- B1_wa_reddit · P-text · task 596 — visited `books`
- B1_wa_reddit · P-text · task 597 — visited `consoles`
- B1_wa_reddit · P-text · task 598 — visited `pittsburgh`
- B1_wa_reddit · P-text · task 599 — visited `machinelearning`
- B1_reddit · SoM · task 171 — visited `mechanicalkeyboards`
- B1_reddit · SoM · task 188 — visited `iphone`
- B1_wa_reddit · SoM · task 597 — visited `consoles`
- B1_wa_reddit · SoM · task 598 — visited `pittsburgh`
- B1_wa_reddit · SoM · task 599 — visited `machinelearning`
- B1_wa_reddit · Vision · task 596 — visited `books`
- B2_reddit · SoM · task 170 — visited `sports`


## WebArena reddit (tasks 595-599) — first audit, 2026-08-03

- **5 tasks** scored by `#sidebar > section`, same Postmill image and the same `require_reset` no-op as VWA reddit
- 50 scored episodes across both WA backbones x 6 modes: **0 LEAKED**, 37 earned, 13 failed

⚠️ **What this zero does and does not mean.** The `earned` test is *did the episode visit the required forum* — the same test VWA uses, so the two are comparable, and it is a **lower bound on leakage**. Visiting is not subscribing: an episode can arrive at a forum an *earlier* episode already subscribed to, read `Unsubscribe` on the button, and finish without acting. That is a leak this test scores as earned.

**The leakage window is open on WA.** Within one run the target forums are reached by many non-target tasks (on `B1`/DOM: `books` by 13 other tasks, `machinelearning` by 8, `pittsburgh` by 4, `consoles` by 2; only `space` is touched by its own task alone). So the mechanism is available here; what the audit establishes is that **no scored success was obtained without ever reaching the forum**, not that no success inherited a subscription.

**One arrival-already-subscribed case is confirmed by hand**: `B1`/DOM task 597, whose final step reads *"a visible 'Unsubscribe 1 subscriber' button, indicating the user is already subscribed ... so the task is complete"*. It is scored `earned` here because it did visit `consoles`.

**A text heuristic was tried and rejected.** Flagging episodes whose reasoning says *already subscribed* without saying *clicked subscribe* returns 6 of 37 — but it misses the hand-confirmed 597 above, because that episode says both (it deliberates: *I need to click the 'Unsubscribe'* and *I will click the 'Subscribe'*). Model self-report cannot separate deliberation from action, so no count is published from it. Deciding this mechanically needs the subscription state before and after each click, which `state_digest` does not carry.
