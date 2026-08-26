# Launch intent — B5 × reddit chain (declared 2026-08-26, BEFORE fire)

Same discipline as `floor_chain_launch_intent_20260817.md` and
`reframe_chain_launch_intent_20260819.md`, and for the same reason (§469.7):
what a run is *for* has to be fixed before its number is known.

**Every cell below gets reported, whatever its number says.** A cell that lands
and is omitted requires a written reason in this file, in the same commit.

## What this chain is for

The reframe chain answered NAACL attack surface **#3** ("is the baseline strong
enough") on classifieds: B5 = GPT-5.6 terra scored **23.66% / 25.00%** on
`cls·dom` against B0's 17.41%, so the study's capability ceiling is no longer
set by an open-weight backbone alone.

It left **#2 (cross-site generalisation) untouched for B5**. Every B5 observation
is from one site. This chain buys the second site and nothing else.

| Claim | Where it stands after the reframe chain | The soft spot | Cell |
|---|---|---|---|
| **#3** Baseline strength | answered on `cls` (B5 > B0 by 6-8pp) | one site | B5-red-dom |
| **#2** Cross-site | B0/B1/B2 span two sites; **B5 spans one** | the strongest backbone is the one with least site coverage | all three |
| **#6** Does the mode structure generalise across families | Phase C tests it on `cls` | untested off `cls` for any API model but B0 | B5-red-{dom,som,vision} = one arm per side |

Three modes, not six: `dom` / `som` / `vision` are **one arm per side** of the
text | combined | visual partition (`TERMS.md §1.1`). The three phantom arms
subdivide the text side, which is a `cls` question already answered there; the
side-level question is what has no second-site B5 answer at all.

## The cells, in fire order

| # | Cell | n | Est. cost (billed / real) | Est. wall-clock |
|---|---|---|---|---|
| **R1** | B5 × red × dom | 203 | $41 / **$52** | ~40 h |
| **R2** | B5 × red × som | 203 | $41 / **$52** | ~40 h |
| **R3** | B5 × red × vision | 203 | $41 / **$52** | ~40 h |

Cost basis: B5 `cls·dom` measured **$0.1374/episode** (A1, 224 ep, $30.78),
scaled by the **1.48×** reddit/classifieds per-episode ratio measured on B0
across all six modes (1.45-1.52×; driven by step count, 19.7-23.2 vs 13.6-16.2).

## ⚠️ Price drift, disclosed before the fact

The proxy re-prices. Read from the registry, never from a document (§471.2):

| date | terra input / output |
|---|---|
| 2026-08-16 | 0.001 / 0.005 |
| 2026-08-19 | 0.002 / 0.012 |
| **2026-08-26** | **0.0025 / 0.015** |

**2.5× / 3× in ten days.** The configs bill at the 08-19 values and are left
that way **on purpose**: A1, A2 and Phase C all bill at 0.002/0.012, and
changing it mid-chain would make the cost column incomparable across the very
cells it exists to compare. Consequence, stated so it cannot be rediscovered as
a surprise:

> **Every B5 cost figure in this project under-states actual spend by 25%.**
> Token counts are the raw observation; price is a post-hoc parameter. Any
> published B5 cost must be recomputed from tokens at the tier price *in force
> when the cell ran*, not from the recorded dollar column.

This is not confined to B5 — a registry sweep on 2026-08-26 found **all 60
API configs** off the live price, in both directions: B0 **+2.5%**, B4
**−8.3%**, B5 **+25%**. B0's and B4's drifts are small enough not to move any
published comparison (all cost comparisons are within-price), but the same
recompute-from-tokens rule applies to them.

## Budget, and why the ceiling is a live quota check

```
balance 2026-08-26          $383.62   (probe, not a document)
Phase C  (still to run)     $192.38   real
this chain (3 cells)        $154.81   real
                            ───────
remaining if both complete   $36.43
```

**$36 of headroom is thin**, and the recorded-cost ceiling that guarded the
previous chains cannot see it — a ceiling computed from the billed column
under-reads real spend by 25%, i.e. exactly the failure mode that would let a
chain run past an empty pool. So this chain gates on the **live remaining
quota** instead, probed before each cell:

- remaining quota **< $60** before a cell → halt, do not start it
- any cell finishing with episodes ≠ 203 → halt
- wall-clock past **2026-09-04** → halt (leaves the 09-05 thesis deadline clear)
- another site chain found running → refuse (host-global lease)

`resume: true` is set, so a mid-cell exhaustion is recoverable after a top-up
rather than data loss.

## Power, declared up front

`d ≈ n × SR × 0.59` (§468 / B-1972); `d < 10` ⇒ inventory, not an interval.
B5's reddit SR is unknown, so it is projected from B0's measured red/cls SR
ratio per mode, applied to B5's `cls·dom` 23.66%:

| Cell | B0 red/cls SR ratio | projected B5 red SR | projected d | verdict |
|---|---|---|---|---|
| dom | 0.84 | ~19.9% | **≈ 24** | interval |
| som | 0.54 | ~12.8% | **≈ 15** | interval |
| vision | 0.31 | ~7.3% | **≈ 8.7** | **inventory only** |

**`vision` is declared under-powered before it runs.** It is included anyway
because a side-level comparison missing the visual side is not a side-level
comparison, and because omitting the arm whose projection is weakest is exactly
the selection this file exists to prevent. It will be reported as inventory.

The projection itself is an assumption, not a measurement: it presumes B5's
site sensitivity resembles B0's. If B5's reddit SR lands far off these, the
power verdicts move with it — which is why they are written down now.

## What is deliberately NOT in this chain

- **The three phantom arms on reddit for B5.** The text-side subdivision is a
  `cls` question; B0 already carries reddit's phantom arms (and their floors).
- **A replicate of any cell.** The API reproducibility floor already has two
  independent models (B0 10.3-14.3%, B5 12.95%). A third measurement of the
  same quantity buys less than a second site does.
- **sol or luna.** `sol` was pre-declared as needed *only if* terra ≈ B0; terra
  came in 6-8pp above, so the condition never triggered. Switching tiers now
  would also make B5 two models and break every cross-cell comparison
  (`reframe_chain_launch_intent_20260819.md`), and at 2.5× terra it does not
  fit the balance.

## Prediction, recorded before the fact

From the B0-ratio projection: **dom ≈ 20%, som ≈ 13%, vision ≈ 7%**, i.e. B5
stays above B0's reddit numbers (14.6 / 14.6 / 7.8%) on dom and som and roughly
ties on vision.

The interesting outcome is the ordering. On `cls`, B0 runs som > vision > dom
(27.2 / 25.0 / 17.4). If B5 on reddit comes back **dom > som > vision**, the
"which side wins" answer is site-dependent *and* model-dependent, and no
single-site claim about side ranking survives. That would be a negative result
about this study's own framing, so it is written down before the data.
