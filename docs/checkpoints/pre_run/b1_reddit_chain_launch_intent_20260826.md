# Launch intent — B1 × reddit replicate (declared 2026-08-26, BEFORE fire)

Same discipline as `floor_chain_launch_intent_20260817.md`,
`reframe_chain_launch_intent_20260819.md` and
`b5_reddit_chain_launch_intent_20260826.md` (§469.7): the cells and what each
outcome would mean are fixed here, before any number exists.

**Every cell below gets reported, whatever its number says.** A cell that lands
and is omitted requires a written reason in this file, in the same commit.

## What this chain is for — falsifying C1, not refining it

C1 (`docs/analysis/cross_sites/serving_mode_floor.md`) says the run-to-run
reproducibility floor groups by **how the backbone is served**:

| | arms | families | sites | floor |
|---|---|---|---|---|
| API-served | 10 | Qwen, OpenAI | 2 | 7.39–14.29% |
| locally served | 3 | Qwen | **1** | 0.00–3.12% |

The API group spans two sites. **The local group does not.** Every local
measurement in this project is `classifieds`. So the reading "the floor is low
locally" is, strictly, "the floor is low locally *on classifieds*", and a
one-site control group is the weakest part of C1.

**This chain does not buy precision. It buys falsifiability.** Declared before
the fire, because it is the whole point:

| B1 × reddit floor lands at | what it means for C1 |
|---|---|
| **0–3%** | the grouping holds across two sites on both arms of the comparison. C1 goes from "API spans 2 sites, local spans 1" to a symmetric claim. |
| **≥7.39%** (inside the API range) | **C1 is dead as stated.** The floor would then track site or workload, not serving path, and `serving_mode_floor.md` must be retracted rather than hedged. |
| **3–7.39%** (the gap) | C1 survives only as "the groups separate on classifieds"; the cross-site version is not available and the honest claim shrinks. **This is the outcome that is easiest to spin and hardest to report, so it is written down first.** |

## The cells, in fire order

| # | Cell | n | Est. wall-clock | Cost |
|---|---|---|---|---|
| **R1** | B1 × red × `som` | 205 | ~35 h | **$0** (local bf16) |
| **R2** | B1 × red × `dom` | 205 | ~35 h | **$0** |

`som` first because it carries the most power; `dom` second because it is the
only local arm anywhere that measured **non-zero** (`B1.cls.dom` = 3.12%), so
its reddit counterpart is the sharpest test of whether that 3.12% is a stable
property or a one-off.

`vision` is **excluded**: B1's reddit vision SR is 2.93%, giving d ≈ 3.5. That
is low enough that the measurement could not distinguish a true zero from a
missed flip, and running it would add a row that reads like evidence without
being any.

**Ordering is load-bearing.** If the deadline cuts the chain, it cuts `dom`, and
the surviving cell is the higher-powered one. That is deliberate — the fallback
is a worse version of the same test, not a different test.

## Power, declared up front

`d ≈ n × SR × 0.59` (§468 / B-1972); `d < 10` ⇒ inventory, not an interval.
Projected from the archived B1 reddit SRs at n=203:

| Cell | archive SR | projected d | verdict |
|---|---|---|---|
| `som` | 8.29% | **9.9** | **inventory only** (0.1 short of the bar) |
| `dom` | 6.83% | **8.2** | **inventory only** |

**Both cells are declared underpowered before they run.** They are run anyway
because C1's claim is about a *grouping* — whether the two ranges overlap — and
an inventory-grade point still answers that. It cannot tighten the local range;
it can move a point across a boundary. An earlier note in this session called
this chain "the most worthwhile buy in the window" on cost-and-direction
grounds; that was the wrong reason and is corrected here — the reason is
falsifiability, and the power table is why it can only be that.

## Halt conditions

- upstream `_b5_reddit_chain.sh` HALTs, or Phase C/B5-reddit stalls → halt, do
  not wait silently (the 2026-08-26 03:10 lesson: a downstream chain waiting on
  a dead upstream waits until its own deadline)
- any cell finishes with episodes ≠ **205** (the COLLECTION denominator; the
  scored one is 203 — B-1992 was exactly this confusion)
- wall-clock past **2026-09-06** → halt regardless of progress
- another site chain found running → refuse (host-global lease)

No quota gate: B1 runs on local weights and spends nothing on the proxy. That is
also why this chain can be armed now rather than weighed against the balance.

## Prediction, recorded before the fact

B1 on classifieds gave 0.00 / 0.00 / 3.12%, and a controlled step-level probe
(§298.2) returned determinism 133/133. If local serving is genuinely
deterministic then reddit should land in the same place, and the honest
expectation is **0–3%**, i.e. the outcome that *supports* C1.

Writing that down matters because it is the self-serving prediction. The row
that would cost the most — ≥7.39% — is the one this chain exists to expose, and
predicting the comfortable outcome in advance is what makes the uncomfortable
one reportable if it arrives.
