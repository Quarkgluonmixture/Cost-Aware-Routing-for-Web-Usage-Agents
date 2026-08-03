# Per-mode four-dimension evidence profile

- generated: `2026-08-03T10:42:51+00:00`
- schema: `2026-08-02-per-mode-four-dimension-profile-v2`
- **post_hoc_exploratory=True / h10_eligible=False**
- 笔记 §108 evidence layer, cross-mode axis (the paper-headline axis per `paper_section2_framework.canvas`). `INDEX.md §7`: the framework was defined but only the **Macro** dimension had ever been computed per mode. This is the first run of all four.
- ⚠️ No number here is taken from the canvas — its cells are a 2026-05-03 snapshot and several were later retracted (it still shows `drop-one 1.7-3.8pp`; at k=6 **H1 FAILED**, θ_FE 0.7897, p=0.807, §395.6).
- cost comparable **within a cell only** (B0 = proxy API bill; B1/B2 = electricity-derived), hence the `cost_rel_dom` column.

## Data layers and what was excluded

Outcome + Efficiency read episode summaries and use **every** scored task. Macro + Micro read step JSONL and must drop episodes whose step file does not belong to their summary.

- **B0_reddit / P-SoM**: excluded from Macro+Micro — tasks `[87, 149]` (2 episodes). Cause: quarantine → resume-rerun wrote a new summary but left the original interrupted step file in place.

Blast radius measured by `audit_steps_summary_identity.py`: **2 of 7686 episodes** across all 36 combinations.

## Cross-cell consistency — which per-mode differences survive

A per-mode difference only counts if it holds across cells. `unanimous` = the same mode is the extreme in all 6 cells.

⚙️ = the extreme follows from how the mode is **built** (tautology). ◆ = the magnitude is real but its **direction was predictable** from the design. Neither may be cited as a behavioural finding. `tie` counts cells where two or more modes share the extreme — those cells contribute a fractional count, so ordering can never manufacture a unanimous row.

| dim | metric | highest | in | lowest | in | tie | top÷2nd (min–max) | unanimous |
|---|---|---|---|---|---|---|---|---|
| Outcome | success rate % | SoM | 4.5/6 | Vision | 2/6 | 2 | 1.03–2.00× |  |
| Outcome | solves | SoM | 4.5/6 | Vision | 2/6 | 2 | 1.03–2.00× |  |
| Outcome | unique solves (no other mode got it) | Vision | 2.5/6 | P-text | 1.83/6 | 7 | 1.11–5.00× |  |
| Macro | steps / episode | P-SoM | 2/6 | SoM | 5/6 | — | 1.00–1.05× |  |
| Macro | episodes exhausting the step budget | Vision | 2/6 | SoM | 5/6 | — | 1.01–1.09× |  |
| Macro | click fraction | P-SoM | 3/6 | Vision | 4/6 | — | 1.00–1.16× |  |
| Macro | type fraction | P-text | 4/6 | Vision | 4/6 | — | 1.03–1.18× |  |
| Macro | scroll fraction | Vision | 6/6 | SoM | 3/6 | — | 1.25–6.77× | ◆ **high: Vision 6/6** (arch. downstream) |
| Macro | search-loop rate | DOM | 2/6 | Vision | 4/6 | — | 1.01–1.10× |  |
| Macro | URL-revisit step rate | Vision | 6/6 | SoM | 4/6 | — | 1.02–1.05× | **high: Vision 6/6** |
| Micro | parse-invalid step rate | SoM | 2/6 | Vision | 4.5/6 | 1 | 1.03–2.40× |  |
| Micro | action-execution failure rate | Vision | 6/6 | SoM | 2/6 | — | 1.06–1.60× | ◆ **high: Vision 6/6** (arch. downstream) |
| Micro | action failure | action was a click | Vision | 3/6 | P-text | 4/6 | — | 1.00–1.36× |  |
| Micro | action failure | action was a type | DOM | 2/6 | Vision | 4/6 | — | 1.09–2.19× |  |
| Micro | page-unchanged (no-op) step rate | Vision | 6/6 | P-text | 3/6 | — | 1.07–1.58× | ◆ **high: Vision 6/6** (arch. downstream) |
| Micro | scroll action that did not move the viewport | Vision | 5/6 | SoM | 5/6 | — | 1.23–3.50× |  |
| Micro | no-op despite a SUCCEEDING action | SoM | 5/6 | P-text | 2/6 | — | 1.02–1.50× |  |
| Micro | page changed but channel did not show it | Vision | 2/6 | Vision | 4/6 | — | 1.03–1.22× |  |
| Micro | locator fallback rate | P-prompt | 2/6 | Vision | 6/6 | — | 1.00–1.43× | ⚙️ **low: Vision 6/6** (by construction) |
| Micro | consecutive same-action rate | Vision | 4/6 | SoM | 2/6 | — | 1.01–1.19× |  |
| Micro | episodes ending in finish | SoM | 5/6 | P-SoM | 2/6 | — | 1.01–1.21× |  |
| Efficiency | billed cost / episode | SoM | 5/6 | Vision | 6/6 | — | 1.00–1.12× | ⚙️ **low: Vision 6/6** (by construction) |
| Efficiency | cost relative to DOM (within cell) | SoM | 5/6 | Vision | 6/6 | — | 1.00–1.12× | ⚙️ **low: Vision 6/6** (by construction) |
| Efficiency | latency / episode (s) | P-text | 3/6 | SoM | 3/6 | — | 1.00–1.10× |  |
| Efficiency | latency canonical / episode (s) | P-text | 3/6 | SoM | 3/6 | — | 1.00–1.04× |  |
| Efficiency | tokens / episode | SoM | 5/6 | Vision | 6/6 | — | 1.03–1.13× | ⚙️ **low: Vision 6/6** (by construction) |

Why each ◆ row is architecturally downstream. **These are author-written causal assertions, and each one deletes a whole row of evidence, so each needs its own support.** All three were tested on 2026-08-02 and none survived as originally written; the entries below are the rewritten versions and carry the test that forced the rewrite:

- `scroll fraction` — viewport-only observation with no AXTree to enumerate off-screen targets pushes toward scrolling; the 1.2-6.8x magnitude is real but its DIRECTION was predictable from the design. ⚠️ HALF REFUTED: this entry previously gave a second mechanism, re-orienting after a no-op. Measured on B0 x {cls, red} x {dom, som, vision}, the share of scroll steps whose predecessor was a no-op sits AT OR BELOW the base rate of no-ops in the same run in all six combinations (Vision on classifieds: 18.5% against a 36.4% base, 17.9 points below chance). Scrolls are not preferentially preceded by no-ops anywhere. Only the viewport-enumeration mechanism survives and the marking now rests on it alone.
- `action-execution failure rate` — coordinate addressing has no element-identity guarantee, so a higher miss rate is the expected direction. ⚠️ WEAKENED: if that mechanism drove the total, the excess should concentrate in spatially-targeted actions. It does not. Vision leads on click-conditional failure in only 3/6 cells and is the LOWEST mode on type-conditional failure in 4/6. The direction of the total is still predictable; the stated decomposition is not what produces it.
- `page-unchanged (no-op) step rate` — largely downstream of the action-failure row above. ⚠️ REFINED: measured, `action_success = False` implies `page_changed = False` in 100% of steps across all six B0 combinations checked, so action failure is a strict SUBSET of no-op and the metric decomposes exactly into it plus a successful-but-inert residual. Vision's high rate is dominated by the first term (85-95% of its no-ops), but the residual is NOT Vision-led — see the separate `noop_inert_rate` row, where SoM is the extreme in 5/6 cells.

Why each ⚙️ row is architectural:

- `locator fallback rate` — Vision emits coordinates (`coordinate_type: qwen_0_1000`) and has zero element ids, so it barely enters the element-id locator path at all — the residual 0.002-0.011 is not a lower fallback rate on the same mechanism.
- `billed cost / episode` — billed cost is dominated by input tokens, so this inherits the token property above.
- `cost relative to DOM (within cell)` — same quantity as mean_cost_usd, expressed against DOM.
- `tokens / episode` — Vision carries no AXTree text, so its token count is lower by construction.

Metrics added 2026-08-02 and **not yet adjudicated** either way. Absence from the ⚙️ and ◆ lists above means *nobody has ruled on this*, not *verified clean*, and an unflagged unanimous row must not be read as an endorsed behavioural finding:

- `URL-revisit step rate` — Vision is the extreme in every cell, the only unflagged unanimous row in the grid. A plausible architectural story exists (a channel that cannot enumerate off-screen targets navigates more exploratorily and so returns to pages it has seen), and it has not been tested. Do not cite as a behavioural finding until it is.
- `episodes exhausting the step budget` — SoM is the extreme (lowest) in 5/6. Reads with `n_steps` (lowest, 5/6) and `finish_rate` (highest, 5/6) as one signature rather than three findings: the fused mode terminates sooner and more often by choice. Count it once.
- `no-op despite a SUCCEEDING action` — SoM highest in 5/6. This is the residual of `no_change_rate` after action failures are removed, so it is the part of that metric the ◆ marking above does NOT cover.
- `page changed but channel did not show it` — no signal: Vision is the extreme at both ends (highest in 2/6, lowest in 4/6). A two-cell probe on 2026-08-02 read Vision as uniformly highest and that reading did not survive the full grid. Reported so the absence is on the record.
- `action failure | action was a click` — diagnostic for the `action_fail_rate` marking above, not a standalone finding.
- `action failure | action was a type` — diagnostic for the `action_fail_rate` marking above, not a standalone finding.

## Outcome

### B0_classifieds

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| success rate % | 17.41 | 27.23 | 25.00 | 15.62 | 19.64 | 15.62 |
| solves | 39 | 61 | 56 | 35 | 44 | 35 |
| unique solves (no other mode got it) | 4 | 6 | 9 | 2 | 6 | 2 |

### B0_reddit

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| success rate % | 14.29 | 14.78 | 7.39 | 13.30 | 12.32 | 10.84 |
| solves | 29 | 30 | 15 | 27 | 25 | 22 |
| unique solves (no other mode got it) | 3 | 4 | 4 | 2 | 2 | 2 |

### B1_classifieds

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| success rate % | 6.25 | 14.29 | 12.50 | 7.59 | 6.70 | 6.70 |
| solves | 14 | 32 | 28 | 17 | 15 | 15 |
| unique solves (no other mode got it) | 1 | 10 | 9 | 1 | 2 | 3 |

### B1_reddit

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| success rate % | 5.91 | 7.39 | 2.46 | 5.91 | 5.42 | 5.91 |
| solves | 12 | 15 | 5 | 12 | 11 | 12 |
| unique solves (no other mode got it) | 1 | 1 | 2 | 1 | 2 | 0 |

### B2_classifieds

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| success rate % | 1.34 | 2.23 | 2.23 | 0.45 | 1.79 | 0.89 |
| solves | 3 | 5 | 5 | 1 | 4 | 2 |
| unique solves (no other mode got it) | 1 | 5 | 5 | 0 | 0 | 1 |

### B2_reddit

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| success rate % | 3.94 | 0.99 | 1.97 | 1.97 | 0.00 | 0.49 |
| solves | 8 | 2 | 4 | 4 | 0 | 1 |
| unique solves (no other mode got it) | 6 | 1 | 3 | 1 | 0 | 1 |

## Macro

### B0_classifieds

trajectory metrics on the **224 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| steps / episode | 15.61 | 13.67 | 15.88 | 15.83 | 14.96 | 16.23 |
| episodes exhausting the step budget *(task-macro)* | 0.2679 | 0.2589 | 0.2946 | 0.2946 | 0.2902 | 0.3214 |
| episodes exhausting the step budget *(pooled-step)* | 0.2679 | 0.2589 | 0.2946 | 0.2946 | 0.2902 | 0.3214 |
| click fraction *(task-macro)* | 0.3217 | 0.3191 | 0.3534 | 0.3215 | 0.3279 | 0.3122 |
| click fraction *(pooled-step)* | 0.3061 | 0.3123 | 0.3531 | 0.3113 | 0.3254 | 0.3133 |
| type fraction *(task-macro)* | 0.2245 | 0.2051 | 0.1461 | 0.1937 | 0.2083 | 0.2007 |
| type fraction *(pooled-step)* | 0.2746 | 0.2522 | 0.1645 | 0.2276 | 0.2519 | 0.2184 |
| scroll fraction *(task-macro)* | 0.1834 | 0.1546 | 0.2603 | 0.1960 | 0.1740 | 0.2085 |
| scroll fraction *(pooled-step)* | 0.2051 | 0.1702 | 0.2952 | 0.2140 | 0.2057 | 0.2583 |
| search-loop rate | 0.8125 | 0.6920 | 0.7277 | 0.8080 | 0.7589 | 0.7902 |
| URL-revisit step rate *(task-macro)* | 0.6059 | 0.5576 | 0.6389 | 0.6027 | 0.5826 | 0.5996 |
| URL-revisit step rate *(pooled-step)* | 0.7014 | 0.6910 | 0.7489 | 0.7033 | 0.7030 | 0.7159 |

### B0_reddit

trajectory metrics on the **201 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 2, 'SoM': 2, 'Vision': 2, 'P-text': 2, 'P-prompt': 2, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| steps / episode | 20.18 | 20.08 | 23.55 | 23.22 | 19.87 | 22.90 |
| episodes exhausting the step budget *(task-macro)* | 0.4975 | 0.5124 | 0.6716 | 0.6468 | 0.4627 | 0.5920 |
| episodes exhausting the step budget *(pooled-step)* | 0.4975 | 0.5124 | 0.6716 | 0.6468 | 0.4627 | 0.5920 |
| click fraction *(task-macro)* | 0.4552 | 0.4740 | 0.3395 | 0.4420 | 0.4750 | 0.4529 |
| click fraction *(pooled-step)* | 0.4651 | 0.4792 | 0.3179 | 0.4322 | 0.4680 | 0.4401 |
| type fraction *(task-macro)* | 0.1484 | 0.1228 | 0.0792 | 0.1532 | 0.1416 | 0.1371 |
| type fraction *(pooled-step)* | 0.1486 | 0.1160 | 0.0716 | 0.1500 | 0.1540 | 0.1323 |
| scroll fraction *(task-macro)* | 0.1662 | 0.0994 | 0.3432 | 0.1166 | 0.1264 | 0.1243 |
| scroll fraction *(pooled-step)* | 0.1913 | 0.0994 | 0.3692 | 0.1260 | 0.1467 | 0.1264 |
| search-loop rate | 0.4726 | 0.3632 | 0.2388 | 0.3881 | 0.4279 | 0.3383 |
| URL-revisit step rate *(task-macro)* | 0.7209 | 0.7306 | 0.8171 | 0.7746 | 0.7044 | 0.7747 |
| URL-revisit step rate *(pooled-step)* | 0.8129 | 0.8370 | 0.8697 | 0.8453 | 0.7922 | 0.8392 |

### B1_classifieds

trajectory metrics on the **224 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| steps / episode | 21.38 | 18.01 | 20.17 | 22.46 | 21.40 | 21.26 |
| episodes exhausting the step budget *(task-macro)* | 0.5848 | 0.4911 | 0.5982 | 0.6250 | 0.5938 | 0.5759 |
| episodes exhausting the step budget *(pooled-step)* | 0.5848 | 0.4911 | 0.5982 | 0.6250 | 0.5938 | 0.5759 |
| click fraction *(task-macro)* | 0.2456 | 0.3684 | 0.3305 | 0.2807 | 0.3749 | 0.4359 |
| click fraction *(pooled-step)* | 0.2585 | 0.4405 | 0.3718 | 0.2867 | 0.4071 | 0.4767 |
| type fraction *(task-macro)* | 0.3449 | 0.2109 | 0.0713 | 0.3672 | 0.2093 | 0.2123 |
| type fraction *(pooled-step)* | 0.3802 | 0.2248 | 0.0564 | 0.3899 | 0.2020 | 0.2094 |
| scroll fraction *(task-macro)* | 0.1714 | 0.1375 | 0.3599 | 0.1482 | 0.1626 | 0.1545 |
| scroll fraction *(pooled-step)* | 0.2048 | 0.1832 | 0.4520 | 0.1738 | 0.2061 | 0.1844 |
| search-loop rate | 0.7768 | 0.6429 | 0.6027 | 0.8036 | 0.7232 | 0.7366 |
| URL-revisit step rate *(task-macro)* | 0.6981 | 0.6553 | 0.7453 | 0.7162 | 0.7103 | 0.7153 |
| URL-revisit step rate *(pooled-step)* | 0.8084 | 0.8213 | 0.9002 | 0.8036 | 0.8133 | 0.8184 |

### B1_reddit

trajectory metrics on the **203 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| steps / episode | 23.44 | 22.38 | 23.24 | 25.56 | 23.72 | 25.17 |
| episodes exhausting the step budget *(task-macro)* | 0.6798 | 0.6700 | 0.6847 | 0.7833 | 0.6946 | 0.7734 |
| episodes exhausting the step budget *(pooled-step)* | 0.6798 | 0.6700 | 0.6847 | 0.7833 | 0.6946 | 0.7734 |
| click fraction *(task-macro)* | 0.4710 | 0.5738 | 0.4341 | 0.4762 | 0.5454 | 0.5801 |
| click fraction *(pooled-step)* | 0.4984 | 0.6549 | 0.4790 | 0.4842 | 0.5888 | 0.6198 |
| type fraction *(task-macro)* | 0.2383 | 0.1290 | 0.0285 | 0.2456 | 0.1488 | 0.1658 |
| type fraction *(pooled-step)* | 0.2458 | 0.1215 | 0.0242 | 0.2575 | 0.1499 | 0.1793 |
| scroll fraction *(task-macro)* | 0.0663 | 0.0954 | 0.2701 | 0.0674 | 0.0517 | 0.0560 |
| scroll fraction *(pooled-step)* | 0.0719 | 0.1200 | 0.3137 | 0.0736 | 0.0530 | 0.0573 |
| search-loop rate | 0.5961 | 0.4089 | 0.1478 | 0.6059 | 0.5665 | 0.5567 |
| URL-revisit step rate *(task-macro)* | 0.7536 | 0.7470 | 0.8123 | 0.7928 | 0.7572 | 0.7776 |
| URL-revisit step rate *(pooled-step)* | 0.8283 | 0.8635 | 0.9046 | 0.8406 | 0.8332 | 0.8446 |

### B2_classifieds

trajectory metrics on the **224 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| steps / episode | 27.38 | 24.37 | 28.25 | 26.85 | 27.84 | 28.38 |
| episodes exhausting the step budget *(task-macro)* | 0.8170 | 0.6875 | 0.8884 | 0.8036 | 0.8304 | 0.8527 |
| episodes exhausting the step budget *(pooled-step)* | 0.8170 | 0.6875 | 0.8884 | 0.8036 | 0.8304 | 0.8527 |
| click fraction *(task-macro)* | 0.5197 | 0.4878 | 0.3359 | 0.3840 | 0.5144 | 0.5231 |
| click fraction *(pooled-step)* | 0.5293 | 0.5246 | 0.3355 | 0.3991 | 0.5272 | 0.5415 |
| type fraction *(task-macro)* | 0.1423 | 0.1430 | 0.1276 | 0.1682 | 0.0777 | 0.0819 |
| type fraction *(pooled-step)* | 0.1459 | 0.1400 | 0.1338 | 0.1814 | 0.0773 | 0.0841 |
| scroll fraction *(task-macro)* | 0.0309 | 0.0333 | 0.3232 | 0.0431 | 0.0477 | 0.0331 |
| scroll fraction *(pooled-step)* | 0.0308 | 0.0322 | 0.3236 | 0.0451 | 0.0495 | 0.0340 |
| search-loop rate | 0.6473 | 0.6161 | 0.7098 | 0.6295 | 0.6295 | 0.6830 |
| URL-revisit step rate *(task-macro)* | 0.8235 | 0.8284 | 0.8956 | 0.8324 | 0.8571 | 0.8577 |
| URL-revisit step rate *(pooled-step)* | 0.8415 | 0.8663 | 0.9047 | 0.8505 | 0.8721 | 0.8668 |

### B2_reddit

trajectory metrics on the **203 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| steps / episode | 28.39 | 26.34 | 26.91 | 27.27 | 27.68 | 27.87 |
| episodes exhausting the step budget *(task-macro)* | 0.8966 | 0.7833 | 0.8276 | 0.8177 | 0.8424 | 0.8818 |
| episodes exhausting the step budget *(pooled-step)* | 0.8966 | 0.7833 | 0.8276 | 0.8177 | 0.8424 | 0.8818 |
| click fraction *(task-macro)* | 0.7236 | 0.7021 | 0.3492 | 0.6602 | 0.7034 | 0.7080 |
| click fraction *(pooled-step)* | 0.7314 | 0.7489 | 0.3450 | 0.6994 | 0.7391 | 0.7379 |
| type fraction *(task-macro)* | 0.1084 | 0.1028 | 0.0811 | 0.0987 | 0.0806 | 0.0937 |
| type fraction *(pooled-step)* | 0.1150 | 0.1066 | 0.0849 | 0.1053 | 0.0796 | 0.0979 |
| scroll fraction *(task-macro)* | 0.0417 | 0.0433 | 0.3441 | 0.0380 | 0.0518 | 0.0479 |
| scroll fraction *(pooled-step)* | 0.0399 | 0.0406 | 0.3540 | 0.0356 | 0.0536 | 0.0495 |
| search-loop rate | 0.2266 | 0.1478 | 0.1379 | 0.2660 | 0.2709 | 0.2167 |
| URL-revisit step rate *(task-macro)* | 0.8702 | 0.8574 | 0.9114 | 0.8848 | 0.8734 | 0.8817 |
| URL-revisit step rate *(pooled-step)* | 0.8775 | 0.8906 | 0.9301 | 0.9033 | 0.8831 | 0.9010 |

## Micro

### B0_classifieds

trajectory metrics on the **224 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| parse-invalid step rate *(task-macro)* | 0.0007 | 0.0021 | 0.0000 | 0.0000 | 0.0009 | 0.0005 |
| parse-invalid step rate *(pooled-step)* | 0.0006 | 0.0013 | 0.0000 | 0.0000 | 0.0012 | 0.0008 |
| action-execution failure rate *(task-macro)* | 0.1334 | 0.0817 | 0.1499 | 0.1012 | 0.1325 | 0.1118 |
| action-execution failure rate *(pooled-step)* | 0.1733 | 0.1235 | 0.2238 | 0.1331 | 0.1940 | 0.1581 |
| action failure | action was a click *(task-macro)* | 0.1635 | 0.1016 | 0.1533 | 0.1004 | 0.1594 | 0.1298 |
| action failure | action was a click *(pooled-step)* | 0.2664 | 0.2155 | 0.2842 | 0.1540 | 0.3110 | 0.2230 |
| action failure | action was a type *(task-macro)* | 0.0349 | 0.0067 | 0.0048 | 0.0221 | 0.0261 | 0.0157 |
| action failure | action was a type *(pooled-step)* | 0.0896 | 0.0130 | 0.0291 | 0.0409 | 0.0723 | 0.0189 |
| page-unchanged (no-op) step rate *(task-macro)* | 0.2413 | 0.2156 | 0.2629 | 0.2094 | 0.2463 | 0.2206 |
| page-unchanged (no-op) step rate *(pooled-step)* | 0.2203 | 0.1764 | 0.2671 | 0.1777 | 0.2412 | 0.1999 |
| scroll action that did not move the viewport *(task-macro)* | 0.0958 | 0.0178 | 0.0718 | 0.0712 | 0.0776 | 0.0780 |
| scroll action that did not move the viewport *(pooled-step)* | 0.2678 | 0.1036 | 0.1962 | 0.2016 | 0.2845 | 0.2620 |
| no-op despite a SUCCEEDING action *(task-macro)* | 0.1079 | 0.1339 | 0.1130 | 0.1082 | 0.1138 | 0.1088 |
| no-op despite a SUCCEEDING action *(pooled-step)* | 0.0469 | 0.0529 | 0.0433 | 0.0446 | 0.0472 | 0.0418 |
| page changed but channel did not show it *(task-macro)* | 0.0464 | 0.0511 | 0.0579 | 0.0415 | 0.0486 | 0.0424 |
| page changed but channel did not show it *(pooled-step)* | 0.0433 | 0.0595 | 0.0602 | 0.0405 | 0.0464 | 0.0364 |
| locator fallback rate *(task-macro)* | 0.0784 | 0.0274 | 0.0025 | 0.0461 | 0.0787 | 0.0370 |
| locator fallback rate *(pooled-step)* | 0.1001 | 0.0438 | 0.0042 | 0.0592 | 0.1167 | 0.0490 |
| consecutive same-action rate *(task-macro)* | 0.3242 | 0.3000 | 0.3957 | 0.3102 | 0.3215 | 0.3318 |
| consecutive same-action rate *(pooled-step)* | 0.4465 | 0.4579 | 0.5206 | 0.3995 | 0.4475 | 0.4522 |
| episodes ending in finish | 0.7366 | 0.7411 | 0.7054 | 0.7054 | 0.7188 | 0.6875 |

### B0_reddit

trajectory metrics on the **201 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 2, 'SoM': 2, 'Vision': 2, 'P-text': 2, 'P-prompt': 2, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| parse-invalid step rate *(task-macro)* | 0.0002 | 0.0008 | 0.0000 | 0.0005 | 0.0026 | 0.0027 |
| parse-invalid step rate *(pooled-step)* | 0.0002 | 0.0012 | 0.0000 | 0.0006 | 0.0040 | 0.0028 |
| action-execution failure rate *(task-macro)* | 0.2215 | 0.2515 | 0.3818 | 0.2921 | 0.1942 | 0.2851 |
| action-execution failure rate *(pooled-step)* | 0.2837 | 0.3385 | 0.4571 | 0.3544 | 0.2519 | 0.3480 |
| action failure | action was a click *(task-macro)* | 0.1939 | 0.1266 | 0.1931 | 0.1259 | 0.1462 | 0.1176 |
| action failure | action was a click *(pooled-step)* | 0.3026 | 0.2110 | 0.3236 | 0.1859 | 0.2370 | 0.1797 |
| action failure | action was a type *(task-macro)* | 0.0512 | 0.0224 | 0.0177 | 0.0620 | 0.0461 | 0.0344 |
| action failure | action was a type *(pooled-step)* | 0.1891 | 0.0598 | 0.0560 | 0.2186 | 0.1626 | 0.1248 |
| page-unchanged (no-op) step rate *(task-macro)* | 0.2938 | 0.3284 | 0.4304 | 0.3391 | 0.2700 | 0.3353 |
| page-unchanged (no-op) step rate *(pooled-step)* | 0.3081 | 0.3620 | 0.4708 | 0.3694 | 0.2787 | 0.3656 |
| scroll action that did not move the viewport *(task-macro)* | 0.1607 | 0.0846 | 0.2681 | 0.1456 | 0.1494 | 0.1596 |
| scroll action that did not move the viewport *(pooled-step)* | 0.3776 | 0.2993 | 0.5046 | 0.5510 | 0.4727 | 0.5361 |
| no-op despite a SUCCEEDING action *(task-macro)* | 0.0723 | 0.0769 | 0.0487 | 0.0470 | 0.0757 | 0.0502 |
| no-op despite a SUCCEEDING action *(pooled-step)* | 0.0244 | 0.0235 | 0.0137 | 0.0150 | 0.0268 | 0.0176 |
| page changed but channel did not show it *(task-macro)* | 0.0755 | 0.0918 | 0.0637 | 0.0872 | 0.0932 | 0.0973 |
| page changed but channel did not show it *(pooled-step)* | 0.0880 | 0.1045 | 0.0774 | 0.0863 | 0.0850 | 0.0976 |
| locator fallback rate *(task-macro)* | 0.1185 | 0.0675 | 0.0063 | 0.0642 | 0.0827 | 0.0704 |
| locator fallback rate *(pooled-step)* | 0.1481 | 0.0835 | 0.0055 | 0.0744 | 0.0999 | 0.0778 |
| consecutive same-action rate *(task-macro)* | 0.4282 | 0.4910 | 0.5563 | 0.5179 | 0.4211 | 0.5017 |
| consecutive same-action rate *(pooled-step)* | 0.5192 | 0.5995 | 0.6349 | 0.5802 | 0.4883 | 0.5695 |
| episodes ending in finish | 0.5075 | 0.4876 | 0.3284 | 0.3582 | 0.5373 | 0.4080 |

### B1_classifieds

trajectory metrics on the **224 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| parse-invalid step rate *(task-macro)* | 0.0031 | 0.0016 | 0.0000 | 0.0004 | 0.0044 | 0.0010 |
| parse-invalid step rate *(pooled-step)* | 0.0019 | 0.0007 | 0.0000 | 0.0006 | 0.0029 | 0.0004 |
| action-execution failure rate *(task-macro)* | 0.2846 | 0.2487 | 0.4540 | 0.2588 | 0.3095 | 0.3218 |
| action-execution failure rate *(pooled-step)* | 0.3737 | 0.3783 | 0.6386 | 0.3205 | 0.3895 | 0.4017 |
| action failure | action was a click *(task-macro)* | 0.1546 | 0.2034 | 0.3447 | 0.1821 | 0.2716 | 0.2740 |
| action failure | action was a click *(pooled-step)* | 0.3110 | 0.4395 | 0.7560 | 0.4064 | 0.4362 | 0.4167 |
| action failure | action was a type *(task-macro)* | 0.1527 | 0.0676 | 0.0307 | 0.1396 | 0.0814 | 0.1360 |
| action failure | action was a type *(pooled-step)* | 0.2812 | 0.2558 | 0.2196 | 0.2188 | 0.1829 | 0.3721 |
| page-unchanged (no-op) step rate *(task-macro)* | 0.3588 | 0.3531 | 0.5476 | 0.3173 | 0.3799 | 0.3811 |
| page-unchanged (no-op) step rate *(pooled-step)* | 0.3912 | 0.4023 | 0.6549 | 0.3364 | 0.4066 | 0.4196 |
| scroll action that did not move the viewport *(task-macro)* | 0.2189 | 0.1488 | 0.2778 | 0.1593 | 0.1781 | 0.1690 |
| scroll action that did not move the viewport *(pooled-step)* | 0.6667 | 0.6116 | 0.6557 | 0.5092 | 0.6700 | 0.5957 |
| no-op despite a SUCCEEDING action *(task-macro)* | 0.0741 | 0.1043 | 0.0936 | 0.0584 | 0.0704 | 0.0593 |
| no-op despite a SUCCEEDING action *(pooled-step)* | 0.0175 | 0.0240 | 0.0164 | 0.0159 | 0.0171 | 0.0178 |
| page changed but channel did not show it *(task-macro)* | 0.0111 | 0.0135 | 0.0182 | 0.0095 | 0.0122 | 0.0149 |
| page changed but channel did not show it *(pooled-step)* | 0.0161 | 0.0141 | 0.0212 | 0.0084 | 0.0113 | 0.0152 |
| locator fallback rate *(task-macro)* | 0.1752 | 0.1178 | 0.0111 | 0.2018 | 0.1743 | 0.2033 |
| locator fallback rate *(pooled-step)* | 0.2292 | 0.1673 | 0.0164 | 0.2467 | 0.2116 | 0.2537 |
| consecutive same-action rate *(task-macro)* | 0.4428 | 0.3905 | 0.5300 | 0.4608 | 0.4088 | 0.4615 |
| consecutive same-action rate *(pooled-step)* | 0.5539 | 0.5444 | 0.7389 | 0.5414 | 0.5005 | 0.5507 |
| episodes ending in finish | 0.4107 | 0.5089 | 0.4018 | 0.3839 | 0.4062 | 0.4241 |

### B1_reddit

trajectory metrics on the **203 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| parse-invalid step rate *(task-macro)* | 0.0021 | 0.0063 | 0.0141 | 0.0018 | 0.0010 | 0.0019 |
| parse-invalid step rate *(pooled-step)* | 0.0025 | 0.0040 | 0.0064 | 0.0012 | 0.0006 | 0.0023 |
| action-execution failure rate *(task-macro)* | 0.2266 | 0.2968 | 0.5316 | 0.2825 | 0.2952 | 0.3318 |
| action-execution failure rate *(pooled-step)* | 0.2660 | 0.3698 | 0.6397 | 0.3132 | 0.3468 | 0.3812 |
| action failure | action was a click *(task-macro)* | 0.2097 | 0.2284 | 0.3343 | 0.1951 | 0.2776 | 0.2973 |
| action failure | action was a click *(pooled-step)* | 0.2656 | 0.3314 | 0.6279 | 0.2683 | 0.3767 | 0.3716 |
| action failure | action was a type *(task-macro)* | 0.0488 | 0.0617 | 0.0100 | 0.1072 | 0.0562 | 0.0766 |
| action failure | action was a type *(pooled-step)* | 0.0966 | 0.3261 | 0.0789 | 0.2111 | 0.1343 | 0.2140 |
| page-unchanged (no-op) step rate *(task-macro)* | 0.2737 | 0.3575 | 0.5817 | 0.3189 | 0.3402 | 0.3681 |
| page-unchanged (no-op) step rate *(pooled-step)* | 0.2782 | 0.3815 | 0.6496 | 0.3211 | 0.3580 | 0.3883 |
| scroll action that did not move the viewport *(task-macro)* | 0.0897 | 0.1106 | 0.3016 | 0.1559 | 0.1143 | 0.1459 |
| scroll action that did not move the viewport *(pooled-step)* | 0.5877 | 0.6587 | 0.7412 | 0.6414 | 0.4745 | 0.6860 |
| no-op despite a SUCCEEDING action *(task-macro)* | 0.0471 | 0.0607 | 0.0501 | 0.0363 | 0.0450 | 0.0363 |
| no-op despite a SUCCEEDING action *(pooled-step)* | 0.0122 | 0.0117 | 0.0100 | 0.0079 | 0.0112 | 0.0070 |
| page changed but channel did not show it *(task-macro)* | 0.0505 | 0.0610 | 0.0223 | 0.0420 | 0.0547 | 0.0591 |
| page changed but channel did not show it *(pooled-step)* | 0.0504 | 0.0673 | 0.0387 | 0.0432 | 0.0709 | 0.0624 |
| locator fallback rate *(task-macro)* | 0.1426 | 0.1764 | 0.0044 | 0.1638 | 0.2007 | 0.2082 |
| locator fallback rate *(pooled-step)* | 0.1610 | 0.2102 | 0.0028 | 0.1820 | 0.2272 | 0.2368 |
| consecutive same-action rate *(task-macro)* | 0.4650 | 0.5520 | 0.6320 | 0.5105 | 0.4936 | 0.5442 |
| consecutive same-action rate *(pooled-step)* | 0.5320 | 0.6896 | 0.7462 | 0.5555 | 0.5661 | 0.6106 |
| episodes ending in finish | 0.3153 | 0.3202 | 0.2906 | 0.2167 | 0.3054 | 0.2217 |

### B2_classifieds

trajectory metrics on the **224 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| parse-invalid step rate *(task-macro)* | 0.0259 | 0.0876 | 0.0067 | 0.0761 | 0.0518 | 0.0522 |
| parse-invalid step rate *(pooled-step)* | 0.0201 | 0.0504 | 0.0068 | 0.0411 | 0.0358 | 0.0333 |
| action-execution failure rate *(task-macro)* | 0.5089 | 0.5594 | 0.6704 | 0.4409 | 0.4947 | 0.4253 |
| action-execution failure rate *(pooled-step)* | 0.5214 | 0.5947 | 0.6887 | 0.4370 | 0.5034 | 0.4245 |
| action failure | action was a click *(task-macro)* | 0.5548 | 0.4755 | 0.7732 | 0.4382 | 0.5692 | 0.5069 |
| action failure | action was a click *(pooled-step)* | 0.6945 | 0.6794 | 0.8479 | 0.5479 | 0.7108 | 0.5870 |
| action failure | action was a type *(task-macro)* | 0.2067 | 0.1103 | 0.3550 | 0.2271 | 0.1675 | 0.1087 |
| action failure | action was a type *(pooled-step)* | 0.3732 | 0.2579 | 0.6517 | 0.4170 | 0.4938 | 0.2879 |
| page-unchanged (no-op) step rate *(task-macro)* | 0.5265 | 0.5840 | 0.6850 | 0.4527 | 0.5042 | 0.4328 |
| page-unchanged (no-op) step rate *(pooled-step)* | 0.5259 | 0.6010 | 0.6923 | 0.4393 | 0.5060 | 0.4264 |
| scroll action that did not move the viewport *(task-macro)* | 0.0989 | 0.0672 | 0.4434 | 0.1297 | 0.1372 | 0.0993 |
| scroll action that did not move the viewport *(pooled-step)* | 0.2540 | 0.2330 | 0.5972 | 0.3727 | 0.5534 | 0.4259 |
| no-op despite a SUCCEEDING action *(task-macro)* | 0.0176 | 0.0246 | 0.0146 | 0.0118 | 0.0095 | 0.0075 |
| no-op despite a SUCCEEDING action *(pooled-step)* | 0.0046 | 0.0062 | 0.0036 | 0.0023 | 0.0026 | 0.0019 |
| page changed but channel did not show it *(task-macro)* | 0.0305 | 0.0262 | 0.0167 | 0.0293 | 0.0275 | 0.0265 |
| page changed but channel did not show it *(pooled-step)* | 0.0210 | 0.0197 | 0.0149 | 0.0252 | 0.0175 | 0.0167 |
| locator fallback rate *(task-macro)* | 0.3390 | 0.3268 | 0.1241 | 0.2922 | 0.3121 | 0.2728 |
| locator fallback rate *(pooled-step)* | 0.3557 | 0.3507 | 0.1308 | 0.3124 | 0.3271 | 0.2852 |
| consecutive same-action rate *(task-macro)* | 0.6299 | 0.6273 | 0.5180 | 0.6021 | 0.6329 | 0.6489 |
| consecutive same-action rate *(pooled-step)* | 0.6518 | 0.6832 | 0.5242 | 0.6142 | 0.6491 | 0.6586 |
| episodes ending in finish | 0.1250 | 0.1518 | 0.1027 | 0.0625 | 0.0714 | 0.0536 |

### B2_reddit

trajectory metrics on the **203 tasks every mode has a usable trajectory for** (paired); dropped for pairing: {'DOM': 0, 'SoM': 0, 'Vision': 0, 'P-text': 0, 'P-prompt': 0, 'P-SoM': 0}

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| parse-invalid step rate *(task-macro)* | 0.0224 | 0.0431 | 0.0163 | 0.0786 | 0.0660 | 0.0501 |
| parse-invalid step rate *(pooled-step)* | 0.0128 | 0.0230 | 0.0145 | 0.0403 | 0.0333 | 0.0267 |
| action-execution failure rate *(task-macro)* | 0.4919 | 0.3835 | 0.6399 | 0.2846 | 0.6041 | 0.3157 |
| action-execution failure rate *(pooled-step)* | 0.4942 | 0.3930 | 0.6855 | 0.2759 | 0.6081 | 0.3134 |
| action failure | action was a click *(task-macro)* | 0.5024 | 0.3444 | 0.6132 | 0.2631 | 0.6211 | 0.2965 |
| action failure | action was a click *(pooled-step)* | 0.5378 | 0.3715 | 0.6753 | 0.2449 | 0.6713 | 0.2728 |
| action failure | action was a type *(task-macro)* | 0.1109 | 0.0897 | 0.2760 | 0.0815 | 0.1260 | 0.1226 |
| action failure | action was a type *(pooled-step)* | 0.4087 | 0.4649 | 0.7177 | 0.3156 | 0.4653 | 0.4296 |
| page-unchanged (no-op) step rate *(task-macro)* | 0.5013 | 0.4027 | 0.6686 | 0.2917 | 0.6095 | 0.3250 |
| page-unchanged (no-op) step rate *(pooled-step)* | 0.4968 | 0.3987 | 0.6912 | 0.2779 | 0.6097 | 0.3150 |
| scroll action that did not move the viewport *(task-macro)* | 0.1564 | 0.0526 | 0.5473 | 0.0944 | 0.1149 | 0.1368 |
| scroll action that did not move the viewport *(pooled-step)* | 0.4609 | 0.3226 | 0.7084 | 0.3807 | 0.4120 | 0.4929 |
| no-op despite a SUCCEEDING action *(task-macro)* | 0.0093 | 0.0192 | 0.0287 | 0.0071 | 0.0054 | 0.0093 |
| no-op despite a SUCCEEDING action *(pooled-step)* | 0.0026 | 0.0056 | 0.0057 | 0.0020 | 0.0016 | 0.0016 |
| page changed but channel did not show it *(task-macro)* | 0.0441 | 0.0519 | 0.0046 | 0.0413 | 0.0420 | 0.0335 |
| page changed but channel did not show it *(pooled-step)* | 0.0579 | 0.0448 | 0.0030 | 0.0333 | 0.0506 | 0.0361 |
| locator fallback rate *(task-macro)* | 0.4430 | 0.3054 | 0.0784 | 0.2016 | 0.5026 | 0.2395 |
| locator fallback rate *(pooled-step)* | 0.4517 | 0.3125 | 0.0818 | 0.2076 | 0.5307 | 0.2517 |
| consecutive same-action rate *(task-macro)* | 0.7702 | 0.7597 | 0.5971 | 0.7278 | 0.7237 | 0.7251 |
| consecutive same-action rate *(pooled-step)* | 0.7772 | 0.7967 | 0.6255 | 0.7468 | 0.7400 | 0.7468 |
| episodes ending in finish | 0.0739 | 0.1626 | 0.1527 | 0.0591 | 0.0443 | 0.0493 |

## Efficiency

### B0_classifieds

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| billed cost / episode | 0.06962 | 0.07236 | 0.06481 | 0.06919 | 0.06853 | 0.07206 |
| cost relative to DOM (within cell) | 1.0000 | 1.0393 | 0.9309 | 0.9938 | 0.9843 | 1.0350 |
| latency / episode (s) | 114.96 | 106.72 | 126.29 | 123.85 | 109.38 | 121.14 |
| latency canonical / episode (s) | 114.0638 | 106.0076 | 125.2114 | 120.4008 | 107.7716 | 117.8978 |
| tokens / episode | 63045 | 67000 | 58302 | 62154 | 62555 | 65334 |

### B0_reddit

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| billed cost / episode | 0.10147 | 0.11045 | 0.09807 | 0.10577 | 0.10163 | 0.10814 |
| cost relative to DOM (within cell) | 1.0000 | 1.0885 | 0.9665 | 1.0424 | 1.0016 | 1.0657 |
| latency / episode (s) | 571.95 | 461.37 | 449.78 | 631.38 | 498.45 | 561.99 |
| latency canonical / episode (s) | 552.5376 | 451.6388 | 418.4608 | 562.0760 | 447.7186 | 532.0163 |
| tokens / episode | 93303 | 103031 | 88872 | 96214 | 93817 | 99005 |

### B1_classifieds

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| billed cost / episode | 0.05951 | 0.06028 | 0.04316 | 0.05879 | 0.06304 | 0.05970 |
| cost relative to DOM (within cell) | 1.0000 | 1.0128 | 0.7251 | 0.9878 | 1.0592 | 1.0031 |
| latency / episode (s) | 308.93 | 261.97 | 269.78 | 313.54 | 301.37 | 311.71 |
| latency canonical / episode (s) | 308.9298 | 261.9671 | 269.7754 | 313.5402 | 301.3677 | 311.7099 |
| tokens / episode | 61904 | 62953 | 44524 | 61074 | 65628 | 61985 |

### B1_reddit

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| billed cost / episode | 0.07330 | 0.08000 | 0.05240 | 0.06948 | 0.07656 | 0.07480 |
| cost relative to DOM (within cell) | 1.0000 | 1.0914 | 0.7149 | 0.9478 | 1.0445 | 1.0204 |
| latency / episode (s) | 602.63 | 609.05 | 456.57 | 598.58 | 614.02 | 616.56 |
| latency canonical / episode (s) | 602.6264 | 609.0501 | 456.5728 | 598.5846 | 614.0174 | 616.5584 |
| tokens / episode | 76462 | 83517 | 53994 | 72222 | 79896 | 77722 |

### B2_classifieds

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| billed cost / episode | 0.07676 | 0.09075 | 0.07065 | 0.07320 | 0.08453 | 0.08456 |
| cost relative to DOM (within cell) | 1.0000 | 1.1822 | 0.9204 | 0.9536 | 1.1011 | 1.1016 |
| latency / episode (s) | 402.32 | 374.25 | 417.76 | 399.45 | 396.41 | 411.03 |
| latency canonical / episode (s) | 402.3194 | 374.2533 | 417.7628 | 399.4490 | 396.4071 | 411.0288 |
| tokens / episode | 79653 | 95081 | 73126 | 75946 | 87948 | 87931 |

### B2_reddit

| metric | DOM | SoM | Vision | P-text | P-prompt | P-SoM |
|---|---|---|---|---|---|---|
| billed cost / episode | 0.09479 | 0.11160 | 0.06833 | 0.08852 | 0.09940 | 0.09451 |
| cost relative to DOM (within cell) | 1.0000 | 1.1773 | 0.7209 | 0.9338 | 1.0486 | 0.9970 |
| latency / episode (s) | 669.92 | 623.01 | 550.02 | 677.61 | 599.28 | 639.96 |
| latency canonical / episode (s) | 669.9164 | 623.0092 | 550.0212 | 677.6121 | 599.2793 | 639.9556 |
| tokens / episode | 99015 | 117379 | 70826 | 92346 | 104021 | 98699 |

## Reading notes

1. **8 of 26 metrics have a unanimous extreme mode — but they fall into three classes, not two: 1 empirical, 3 architecturally downstream, 4 tautological.** The earlier revision of this file used a binary split and put the downstream group on the empirical side; Gemini (cross-AI Mode C, 2026-07-29) attacked that, correctly.
   **Empirical — not predictable from the design:**
   - `URL-revisit step rate` (Macro): **Vision** highest in 6/6, 1.0–1.1× the next mode
   **◆ Architecturally downstream — real magnitudes, predictable direction.** One causal chain explains all of them: coordinate-only addressing → more off-target clicks → page unchanged → scroll to re-orient. Citing these as behavioural discoveries overstates them; promoting one requires a baseline for what a coordinate-addressed agent *should* score, which this profile does not provide.
   - `scroll fraction` (Macro): **Vision** highest in 6/6, 1.2–6.8× the next mode
   - `action-execution failure rate` (Micro): **Vision** highest in 6/6, 1.1–1.6× the next mode
   - `page-unchanged (no-op) step rate` (Micro): **Vision** highest in 6/6, 1.1–1.6× the next mode
   **⚙️ By construction — do NOT cite as findings:** `locator fallback rate`, `billed cost / episode`, `cost relative to DOM (within cell)`, `tokens / episode`
2. **Mechanism claims are not established here.** These are Evidence-layer observations. Reading `scroll_frac` as "viewport-only forces scrolling" is an Explanation-layer hypothesis and the canvas's own reviewer caveat ("Evidence ≠ Explanation") applies: the two must be written separately and linked explicitly, not merged.
3. **Vision is structurally off the 2×2 grid** (no AXTree text), which is why it never appeared in the earlier per-axis analyses (§103). Any Vision row here is the first time that mode has been profiled on this dimension.
4. **Two estimands are reported for every step-level rate.** `task-macro` is the mean over episodes of a within-episode rate; `pooled-step` is total numerator over total denominator. They weight long and short episodes differently and can diverge substantially, so neither is reported alone.
