---
type: analysis
status: complete
purpose: whether a retry-or-switch routing decision has a different supply profile than which-mode routing, and whether the licensed one-arm claim survives every base
scope_warning: three arms of one cell (B0 x VWA-classifieds, n=224). The replicate pairs are run-to-run INCLUDING environment drift (dom 2 days apart, som 69), so retry gains are an UPPER bound on an immediate retry. Everything is oracle-conditioned and needs failure detection to deploy.
producer: scripts/analysis/retry_vs_switch_label_supply.py
---

# Retry-or-switch: label supply, and whether the one-arm claim is base-dependent

Regenerate: `.venv/bin/python3 scripts/analysis/retry_vs_switch_label_supply.py --write-md`  ·  re-render prose only: `--from-json <path> --write-md`

## 1. The one-arm margin, read from every base

`noise_floor_inventory.md` §2 licenses one sentence on this cell — *at the one-arm margin a distinct representation is worth no more than a rerun of the same representation*. It is computed from the cell's **best** single mode, which is also the base least favourable to switching: the strongest arm leaves the others least to add. Read from every base, the sentence is starting-point dependent.

| base | base SR | +1 rerun | +1 distinct representation | switch / retry |
|---|---:|---:|---:|---:|
| `dom.b` | 15.18% | **7.14pp** | 15.18–17.41pp | 2.12–2.44× |
| `dom.a` | 17.41% | **4.91pp** | 15.18–16.96pp | 3.09–3.45× |
| `vision.b` | 24.11% | **7.59pp** | 6.25–11.16pp | 0.82–1.47× |
| `vision.a` | 25.00% | **6.70pp** | 7.14–10.71pp | 1.07–1.60× |
| `som.a` | 27.23% | **7.59pp** | 4.91–7.14pp | 0.65–0.94× |
| `som.b` | 29.46% | **5.36pp** | 3.12–6.25pp | 0.58–1.17× |

Rerun gain moves over 4.91–7.59pp with no trend in the base, while switch gain moves over 3.12–17.41pp and tracks it. That asymmetry has a reading: what a repetition buys is a property of the serving path and the environment, roughly independent of which representation is being repeated, whereas what a switch buys is a function of what the current representation is missing.

⚠️ Both generations of each switch target are reported because pairing a base with the same-generation arm or the other one is a free choice; quoting one would hide the drift sensitivity that choice carries.

## 2. Supply: the decision set is large, the learnable part is not

The which-mode label needs a task **some mode solved**. A retry-or-switch label needs only that the base attempt **failed**, which is far more common — so the decision set should be larger. It is. That turns out not to be the binding constraint.

| base | decision set | retry only | switch only | both | neither | contested | % of cell |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dom.a` | 185 (82.6%) | 1 | 50 | 10 | 124 | 51 | 22.77% |
| `dom.b` | 190 (84.8%) | 6 | 50 | 10 | 124 | 56 | 25.00% |
| `som.a` | 163 (72.8%) | 6 | 22 | 11 | 124 | 28 | 12.50% |
| `som.b` | 158 (70.5%) | 4 | 22 | 8 | 124 | 26 | 11.61% |
| `vision.a` | 168 (75.0%) | 3 | 29 | 12 | 124 | 32 | 14.29% |
| `vision.b` | 170 (75.9%) | 4 | 29 | 13 | 124 | 33 | 14.73% |

`neither` is [124] out of n=224 on every base: the same tasks, no matter which arm starts. The decision set is large because failures are abundant, but most of it carries no preference to learn — both actions fail together.

Against the same-arm-count which-mode contested set (**24.11%** of the cell, recomputed on these three arms), retry-or-switch offers **25.00%** — **1.04×**. Redefining the label does not escape the ceiling. What bounds both is the number of tasks the agent can solve at all, which is the same circularity the draft's §7 names, reached from a second direction.

## 3. A fixed budget of six arms, spent two ways

`noise_floor_inventory.md` declines this explicitly — *Not licensed. 'The whole 6-mode ceiling gain is noise.' We hold one rerun arm, not five.* Three replicated arms exist now, so the six-arm contrast is computable.

| budget | union SR | note |
|---|---:|---|
| 6 representations × 1 generation | **43.30%** | best single `som` @ 27.23% |
| 3 representations × 2 generations | **44.64%** | `dom.a`, `dom.b`, `som.a`, `som.b`, `vision.a`, `vision.b` |

Solved only by the six-representation budget: 3 tasks `[68, 117, 142]`. Only by the 3×2 budget: 6 tasks `[22, 50, 78, 147, 161, 206]`.

The six-representation union reproduces the published **43.30%** six-mode oracle for this cell exactly, which is the correctness check on this whole read path.

⚠️ **The gap is not a difference.** It is 1.34pp = 3 tasks, against a same-condition discordance of 12–14% on this cell. The defensible statement is that the two ways of spending six arms are **indistinguishable here**, not that repetition wins.

⚠️ budgets may share up to 3 arms; not an independent contrast. And the repetition budget covers only three distinct representations, so this is a question about how to spend an arm budget, not a claim that repetition is equivalent to representation diversity.

⚠️ The 3×2 union is if anything **flattered**: its arms are separated by days to months, so it collects environment drift that a same-day budget would not.
